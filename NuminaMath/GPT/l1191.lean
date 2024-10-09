import Mathlib

namespace vince_bus_ride_distance_l1191_119155

/-- 
  Vince's bus ride to school is 0.625 mile, 
  given that Zachary's bus ride is 0.5 mile 
  and Vince's bus ride is 0.125 mile longer than Zachary's.
--/
theorem vince_bus_ride_distance (zachary_ride : ℝ) (vince_longer : ℝ) 
  (h1 : zachary_ride = 0.5) (h2 : vince_longer = 0.125) 
  : zachary_ride + vince_longer = 0.625 :=
by sorry

end vince_bus_ride_distance_l1191_119155


namespace jane_reads_105_pages_in_a_week_l1191_119146

-- Define the pages read in the morning and evening
def pages_morning := 5
def pages_evening := 10

-- Define the number of pages read in a day
def pages_per_day := pages_morning + pages_evening

-- Define the number of days in a week
def days_per_week := 7

-- Define the total number of pages read in a week
def pages_per_week := pages_per_day * days_per_week

-- The theorem that sums up the proof
theorem jane_reads_105_pages_in_a_week : pages_per_week = 105 := by
  sorry

end jane_reads_105_pages_in_a_week_l1191_119146


namespace work_hours_to_pay_off_debt_l1191_119109

theorem work_hours_to_pay_off_debt (initial_debt paid_amount hourly_rate remaining_debt work_hours : ℕ) 
  (h₁ : initial_debt = 100) 
  (h₂ : paid_amount = 40) 
  (h₃ : hourly_rate = 15) 
  (h₄ : remaining_debt = initial_debt - paid_amount) 
  (h₅ : work_hours = remaining_debt / hourly_rate) : 
  work_hours = 4 :=
by
  sorry

end work_hours_to_pay_off_debt_l1191_119109


namespace negation_of_exists_l1191_119168

theorem negation_of_exists (x : ℝ) : ¬ (∃ x : ℝ, x^2 - x + 2 > 0) = ∀ x : ℝ, x^2 - x + 2 ≤ 0 := by
  sorry

end negation_of_exists_l1191_119168


namespace factorize_x4_minus_3x2_plus_1_factorize_a5_plus_a4_minus_2a_plus_1_factorize_m5_minus_2m3_minus_m_minus_1_l1191_119179

-- Problem 1: Prove the factorization of x^4 - 3x^2 + 1
theorem factorize_x4_minus_3x2_plus_1 (x : ℝ) : 
  x^4 - 3 * x^2 + 1 = (x^2 + x - 1) * (x^2 - x - 1) := 
by
  sorry

-- Problem 2: Prove the factorization of a^5 + a^4 - 2a + 1
theorem factorize_a5_plus_a4_minus_2a_plus_1 (a : ℝ) : 
  a^5 + a^4 - 2 * a + 1 = (a^2 + a - 1) * (a^3 + a - 1) := 
by
  sorry

-- Problem 3: Prove the factorization of m^5 - 2m^3 - m - 1
theorem factorize_m5_minus_2m3_minus_m_minus_1 (m : ℝ) : 
  m^5 - 2 * m^3 - m - 1 = (m^3 + m^2 + 1) * (m^2 - m - 1) := 
by
  sorry

end factorize_x4_minus_3x2_plus_1_factorize_a5_plus_a4_minus_2a_plus_1_factorize_m5_minus_2m3_minus_m_minus_1_l1191_119179


namespace compute_ratio_l1191_119172

variable {p q r u v w : ℝ}

theorem compute_ratio
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0) 
  (h1 : p^2 + q^2 + r^2 = 49) 
  (h2 : u^2 + v^2 + w^2 = 64) 
  (h3 : p * u + q * v + r * w = 56) : 
  (p + q + r) / (u + v + w) = 7 / 8 := 
sorry

end compute_ratio_l1191_119172


namespace find_values_l1191_119191

theorem find_values (x y : ℝ) (h1 : (x + y)^2 = 1) (h2 : (x - y)^2 = 49) : 
  x^2 + y^2 = 25 ∧ x * y = -12 :=
by 
  sorry

end find_values_l1191_119191


namespace project_completion_time_l1191_119132

def work_rate_A : ℚ := 1 / 20
def work_rate_B : ℚ := 1 / 30
def total_project_days (x : ℚ) : Prop := (work_rate_A * (x - 10) + work_rate_B * x = 1)

theorem project_completion_time (x : ℚ) (h : total_project_days x) : x = 13 := 
sorry

end project_completion_time_l1191_119132


namespace base12_remainder_l1191_119195

theorem base12_remainder (x : ℕ) (h : x = 2 * 12^3 + 7 * 12^2 + 4 * 12 + 5) : x % 5 = 2 :=
by {
    -- Proof would go here
    sorry
}

end base12_remainder_l1191_119195


namespace mean_days_correct_l1191_119118

noncomputable def mean_days (a1 a2 a3 a4 a5 d1 d2 d3 d4 d5 : ℕ) : ℚ :=
  (a1 * d1 + a2 * d2 + a3 * d3 + a4 * d4 + a5 * d5 : ℚ) / (a1 + a2 + a3 + a4 + a5)

theorem mean_days_correct : mean_days 2 4 5 7 4 1 2 4 5 6 = 4.05 := by
  sorry

end mean_days_correct_l1191_119118


namespace correct_operation_l1191_119192

theorem correct_operation :
  ¬ ( (-3 : ℤ) * x ^ 2 * y ) ^ 3 = -9 * (x ^ 6) * y ^ 3 ∧
  ¬ (a + b) * (a + b) = (a ^ 2 + b ^ 2) ∧
  (4 * x ^ 3 * y ^ 2) * (x ^ 2 * y ^ 3) = (4 * x ^ 5 * y ^ 5) ∧
  ¬ ((-a) + b) * (a - b) = (a ^ 2 - b ^ 2) :=
by
  sorry

end correct_operation_l1191_119192


namespace find_ratio_l1191_119163

theorem find_ratio (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 → 
  P / (x + 6) + Q / (x * (x - 5)) = (x^2 - x + 15) / (x^3 + x^2 - 30 * x)) :
  Q / P = 5 / 6 := sorry

end find_ratio_l1191_119163


namespace total_dividend_received_l1191_119117

noncomputable def investmentAmount : Nat := 14400
noncomputable def faceValue : Nat := 100
noncomputable def premium : Real := 0.20
noncomputable def declaredDividend : Real := 0.07

theorem total_dividend_received :
  let cost_per_share := faceValue * (1 + premium)
  let number_of_shares := investmentAmount / cost_per_share
  let dividend_per_share := faceValue * declaredDividend
  let total_dividend := number_of_shares * dividend_per_share
  total_dividend = 840 := 
by 
  sorry

end total_dividend_received_l1191_119117


namespace purple_balls_correct_l1191_119133

-- Define the total number of balls and individual counts
def total_balls : ℕ := 100
def white_balls : ℕ := 20
def green_balls : ℕ := 30
def yellow_balls : ℕ := 10
def red_balls : ℕ := 37

-- Probability that a ball chosen is neither red nor purple
def prob_neither_red_nor_purple : ℚ := 0.6

-- The number of purple balls to be proven
def purple_balls : ℕ := 3

-- The condition used for the proof
def condition : Prop := prob_neither_red_nor_purple = (white_balls + green_balls + yellow_balls) / total_balls

-- The proof problem statement
theorem purple_balls_correct (h : condition) : 
  ∃ P : ℕ, P = purple_balls ∧ P + red_balls = total_balls - (white_balls + green_balls + yellow_balls) :=
by
  have P := total_balls - (white_balls + green_balls + yellow_balls + red_balls)
  existsi P
  sorry

end purple_balls_correct_l1191_119133


namespace number_chosen_l1191_119147

theorem number_chosen (x : ℤ) (h : x / 4 - 175 = 10) : x = 740 := by
  sorry

end number_chosen_l1191_119147


namespace range_of_m_l1191_119177

-- Definitions
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := (x^2 - 4*x + 4 - m^2) ≤ 0

-- Theorem Statement
theorem range_of_m (m : ℝ) (h_m : m > 0) : 
  (¬(∃ x, ¬p x) → ¬(∃ x, ¬q x m)) → m ≥ 8 := 
sorry -- Proof not required

end range_of_m_l1191_119177


namespace unique_solution_l1191_119123

noncomputable def solve_system (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ) (x1 x2 x3 : ℝ) : Prop :=
  (a11 * x1 + a12 * x2 + a13 * x3 = 0) ∧
  (a21 * x1 + a22 * x2 + a23 * x3 = 0) ∧
  (a31 * x1 + a32 * x2 + a33 * x3 = 0)

theorem unique_solution 
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ)
  (h1 : 0 < a11) (h2 : 0 < a22) (h3 : 0 < a33)
  (h4 : a12 < 0) (h5 : a13 < 0) (h6 : a21 < 0)
  (h7 : a23 < 0) (h8 : a31 < 0) (h9 : a32 < 0)
  (h10 : 0 < a11 + a12 + a13) (h11 : 0 < a21 + a22 + a23) (h12 : 0 < a31 + a32 + a33) :
  ∀ (x1 x2 x3 : ℝ), solve_system a11 a12 a13 a21 a22 a23 a31 a32 a33 x1 x2 x3 → (x1 = 0 ∧ x2 = 0 ∧ x3 = 0) :=
by
  sorry

end unique_solution_l1191_119123


namespace length_of_bridge_l1191_119189

-- Define the conditions
def train_length : ℕ := 130 -- length of the train in meters
def train_speed : ℕ := 45  -- speed of the train in km/hr
def crossing_time : ℕ := 30  -- time to cross the bridge in seconds

-- Prove that the length of the bridge is 245 meters
theorem length_of_bridge : 
  (train_speed * 1000 / 3600 * crossing_time) - train_length = 245 := 
by
  sorry

end length_of_bridge_l1191_119189


namespace eggs_per_hen_per_day_l1191_119134

theorem eggs_per_hen_per_day
  (hens : ℕ) (days : ℕ) (neighborTaken : ℕ) (dropped : ℕ) (finalEggs : ℕ) (E : ℕ) 
  (h1 : hens = 3) 
  (h2 : days = 7) 
  (h3 : neighborTaken = 12) 
  (h4 : dropped = 5) 
  (h5 : finalEggs = 46) 
  (totalEggs : ℕ := hens * E * days) 
  (afterNeighbor : ℕ := totalEggs - neighborTaken) 
  (beforeDropping : ℕ := finalEggs + dropped) : 
  totalEggs = beforeDropping + neighborTaken → E = 3 := sorry

end eggs_per_hen_per_day_l1191_119134


namespace zander_stickers_l1191_119100

theorem zander_stickers (total_stickers andrew_ratio bill_ratio : ℕ) (initial_stickers: total_stickers = 100) (andrew_fraction : andrew_ratio = 1 / 5) (bill_fraction : bill_ratio = 3 / 10) :
  let andrew_give_away := total_stickers * andrew_ratio
  let remaining_stickers := total_stickers - andrew_give_away
  let bill_give_away := remaining_stickers * bill_ratio
  let total_given_away := andrew_give_away + bill_give_away
  total_given_away = 44 :=
by
  sorry

end zander_stickers_l1191_119100


namespace quadratic_inequality_false_range_l1191_119138

theorem quadratic_inequality_false_range (a : ℝ) :
  (¬ ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (a < 0 ∨ a ≥ 3) :=
by
  sorry

end quadratic_inequality_false_range_l1191_119138


namespace fifteenth_term_l1191_119142

variable (a b : ℤ)

def sum_first_n_terms (n : ℕ) : ℤ := n * (2 * a + (n - 1) * b) / 2

axiom sum_first_10 : sum_first_n_terms 10 = 60
axiom sum_first_20 : sum_first_n_terms 20 = 320

def nth_term (n : ℕ) : ℤ := a + (n - 1) * b

theorem fifteenth_term : nth_term 15 = 25 :=
by
  sorry

end fifteenth_term_l1191_119142


namespace eval_expr_l1191_119174

theorem eval_expr : 3 + 3 * (3 ^ (3 ^ 3)) - 3 ^ 3 = 22876792454937 := by
  sorry

end eval_expr_l1191_119174


namespace kiwi_count_l1191_119139

theorem kiwi_count (s b o k : ℕ)
  (h1 : s + b + o + k = 340)
  (h2 : s = 3 * b)
  (h3 : o = 2 * k)
  (h4 : k = 5 * s) :
  k = 104 :=
sorry

end kiwi_count_l1191_119139


namespace triangle_area_ratio_l1191_119184

theorem triangle_area_ratio (a b c a' b' c' r : ℝ)
    (h1 : a^2 + b^2 = c^2)
    (h2 : a'^2 + b'^2 = c'^2)
    (h3 : r = c' / 2)
    (S : ℝ := (1/2) * a * b)
    (S' : ℝ := (1/2) * a' * b') :
    S / S' ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end triangle_area_ratio_l1191_119184


namespace steve_writes_24_pages_per_month_l1191_119187

/-- Calculate the number of pages Steve writes in a month given the conditions. -/
theorem steve_writes_24_pages_per_month :
  (∃ (days_in_month : ℕ) (letter_interval : ℕ) (letter_minutes : ℕ) (page_minutes : ℕ) 
      (long_letter_factor : ℕ) (long_letter_minutes : ℕ) (total_pages : ℕ),
    days_in_month = 30 ∧ 
    letter_interval = 3 ∧ 
    letter_minutes = 20 ∧ 
    page_minutes = 10 ∧ 
    long_letter_factor = 2 ∧ 
    long_letter_minutes = 80 ∧ 
    total_pages = 24 ∧ 
    (days_in_month / letter_interval * (letter_minutes / page_minutes)
      + long_letter_minutes / (long_letter_factor * page_minutes) = total_pages)) :=
sorry

end steve_writes_24_pages_per_month_l1191_119187


namespace difference_of_squares_65_35_l1191_119150

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := 
  sorry

end difference_of_squares_65_35_l1191_119150


namespace partI_solution_set_l1191_119116

def f (x : ℝ) (a : ℝ) : ℝ := abs (x + a) - abs (x - a^2 - a)

theorem partI_solution_set (x : ℝ) : 
  (f x 1 ≤ 1) ↔ (x ≤ -1) :=
sorry

end partI_solution_set_l1191_119116


namespace ellipse_equation_hyperbola_vertices_and_foci_exists_point_P_on_x_axis_angles_complementary_l1191_119130

noncomputable def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 - y^2 / 2 = 1

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / 2 = 1

def point_on_x_axis (P : ℝ × ℝ) : Prop :=
  P.snd = 0

def angles_complementary (P A B : ℝ × ℝ) : Prop :=
  let kPA := (A.snd - P.snd) / (A.fst - P.fst)
  let kPB := (B.snd - P.snd) / (B.fst - P.fst)
  kPA + kPB = 0

theorem ellipse_equation_hyperbola_vertices_and_foci :
  (∀ x y : ℝ, hyperbola_eq x y → ellipse_eq x y) :=
sorry

theorem exists_point_P_on_x_axis_angles_complementary (F2 A B : ℝ × ℝ) :
  F2 = (1, 0) → (∃ P : ℝ × ℝ, point_on_x_axis P ∧ angles_complementary P A B) :=
sorry

end ellipse_equation_hyperbola_vertices_and_foci_exists_point_P_on_x_axis_angles_complementary_l1191_119130


namespace emily_sold_toys_l1191_119124

theorem emily_sold_toys (initial_toys : ℕ) (remaining_toys : ℕ) (sold_toys : ℕ) 
  (h_initial : initial_toys = 7) 
  (h_remaining : remaining_toys = 4) 
  (h_sold : sold_toys = initial_toys - remaining_toys) :
  sold_toys = 3 :=
by sorry

end emily_sold_toys_l1191_119124


namespace at_least_one_is_half_l1191_119148

theorem at_least_one_is_half (x y z : ℝ) (h : x + y + z - 2 * (x * y + y * z + z * x) + 4 * x * y * z = 1 / 2) :
  x = 1 / 2 ∨ y = 1 / 2 ∨ z = 1 / 2 :=
sorry

end at_least_one_is_half_l1191_119148


namespace men_employed_l1191_119160

theorem men_employed (M : ℕ) (W : ℕ)
  (h1 : W = M * 9)
  (h2 : W = (M + 10) * 6) : M = 20 := by
  sorry

end men_employed_l1191_119160


namespace cubes_sum_identity_l1191_119151

variable {a b : ℝ}

theorem cubes_sum_identity (h : (a / (1 + b) + b / (1 + a) = 1)) : a^3 + b^3 = a + b :=
sorry

end cubes_sum_identity_l1191_119151


namespace number_of_chairs_l1191_119159

theorem number_of_chairs (x t c b T C B: ℕ) (r1 r2 r3: ℕ)
  (h1: x = 2250) (h2: t = 18) (h3: c = 12) (h4: b = 30) 
  (h5: r1 = 2) (h6: r2 = 3) (h7: r3 = 1) 
  (h_ratio1: T / C = r1 / r2) (h_ratio2: B / C = r3 / r2) 
  (h_eq: t * T + c * C + b * B = x) : C = 66 :=
by
  sorry

end number_of_chairs_l1191_119159


namespace stamps_ratio_l1191_119135

noncomputable def number_of_stamps_bought := 300
noncomputable def total_stamps_after_purchase := 450
noncomputable def number_of_stamps_before_purchase := total_stamps_after_purchase - number_of_stamps_bought

theorem stamps_ratio : (number_of_stamps_before_purchase : ℚ) / number_of_stamps_bought = 1 / 2 := by
  have h : number_of_stamps_before_purchase = total_stamps_after_purchase - number_of_stamps_bought := rfl
  rw [h]
  norm_num
  sorry

end stamps_ratio_l1191_119135


namespace number_of_parallelograms_l1191_119194

theorem number_of_parallelograms : 
  (∀ b d k : ℕ, k > 1 → k * b * d = 500000 → (b * d > 0 ∧ y = x ∧ y = k * x)) → 
  (∃ N : ℕ, N = 720) :=
sorry

end number_of_parallelograms_l1191_119194


namespace perimeter_of_quadrilateral_eq_fifty_l1191_119149

theorem perimeter_of_quadrilateral_eq_fifty
  (a b : ℝ)
  (h1 : a = 10)
  (h2 : b = 15)
  (h3 : ∀ (p q r s : ℝ), p + q = r + s) : 
  2 * a + 2 * b = 50 := 
by
  sorry

end perimeter_of_quadrilateral_eq_fifty_l1191_119149


namespace missing_jar_size_l1191_119115

theorem missing_jar_size (total_ounces jars_16 jars_28 jars_unknown m n p: ℕ) (h1 : m = 3) (h2 : n = 3) (h3 : p = 3)
    (total_jars : m + n + p = 9)
    (total_peanut_butter : 16 * m + 28 * n + jars_unknown * p = 252)
    : jars_unknown = 40 := by
  sorry

end missing_jar_size_l1191_119115


namespace Paula_needs_52_tickets_l1191_119154

theorem Paula_needs_52_tickets :
  let g := 2
  let b := 4
  let r := 3
  let f := 1
  let t_g := 4
  let t_b := 5
  let t_r := 7
  let t_f := 3
  g * t_g + b * t_b + r * t_r + f * t_f = 52 := by
  intros
  sorry

end Paula_needs_52_tickets_l1191_119154


namespace num_students_earning_B_l1191_119113

open Real

theorem num_students_earning_B (total_students : ℝ) (pA : ℝ) (pB : ℝ) (pC : ℝ) (students_A : ℝ) (students_B : ℝ) (students_C : ℝ) :
  total_students = 31 →
  pA = 0.7 * pB →
  pC = 1.4 * pB →
  students_A = 0.7 * students_B →
  students_C = 1.4 * students_B →
  students_A + students_B + students_C = total_students →
  students_B = 10 :=
by
  intros h_total_students h_pa h_pc h_students_A h_students_C h_total_eq
  sorry

end num_students_earning_B_l1191_119113


namespace reduction_for_1750_yuan_max_daily_profit_not_1900_l1191_119161

def average_shirts_per_day : ℕ := 40 
def profit_per_shirt_initial : ℕ := 40 
def price_reduction_increase_shirts (reduction : ℝ) : ℝ := reduction * 2 
def daily_profit (reduction : ℝ) : ℝ := (profit_per_shirt_initial - reduction) * (average_shirts_per_day + price_reduction_increase_shirts reduction)

-- Part 1: Proving the reduction that results in 1750 yuan profit
theorem reduction_for_1750_yuan : ∃ x : ℝ, daily_profit x = 1750 ∧ x = 15 := 
by {
  sorry
}

-- Part 2: Proving that the maximum cannot reach 1900 yuan
theorem max_daily_profit_not_1900 : ∀ x : ℝ, daily_profit x ≤ 1800 ∧ (∀ y : ℝ, y ≥ daily_profit x → y < 1900) :=
by {
  sorry
}

end reduction_for_1750_yuan_max_daily_profit_not_1900_l1191_119161


namespace find_m_l1191_119106

theorem find_m (x y m : ℝ) (h1 : 2 * x + y = 1) (h2 : x + 2 * y = 2) (h3 : x + y = 2 * m - 1) : m = 1 :=
by
  sorry

end find_m_l1191_119106


namespace terminal_side_quadrant_l1191_119157

theorem terminal_side_quadrant (k : ℤ) : 
  ∃ quadrant, quadrant = 1 ∨ quadrant = 3 ∧
  ∀ (α : ℝ), α = k * 180 + 45 → 
  (quadrant = 1 ∧ (∃ n : ℕ, k = 2 * n)) ∨ (quadrant = 3 ∧ (∃ n : ℕ, k = 2 * n + 1)) :=
by
  sorry

end terminal_side_quadrant_l1191_119157


namespace students_speak_both_l1191_119137

theorem students_speak_both (total E T N : ℕ) (h1 : total = 150) (h2 : E = 55) (h3 : T = 85) (h4 : N = 30) :
  E + T - (total - N) = 20 := by
  -- Main proof logic
  sorry

end students_speak_both_l1191_119137


namespace total_minutes_of_game_and_ceremony_l1191_119186

-- Define the components of the problem
def game_hours : ℕ := 2
def game_additional_minutes : ℕ := 35
def ceremony_minutes : ℕ := 25

-- Prove the total minutes is 180
theorem total_minutes_of_game_and_ceremony (h: game_hours = 2) (ga: game_additional_minutes = 35) (c: ceremony_minutes = 25) :
  (game_hours * 60 + game_additional_minutes + ceremony_minutes) = 180 :=
  sorry

end total_minutes_of_game_and_ceremony_l1191_119186


namespace evaluate_expression_l1191_119141

theorem evaluate_expression : abs (abs (abs (-2 + 2) - 2) * 2) = 4 := 
by
  sorry

end evaluate_expression_l1191_119141


namespace sign_up_ways_l1191_119101

theorem sign_up_ways : (3 ^ 4) = 81 :=
by
  sorry

end sign_up_ways_l1191_119101


namespace expected_value_of_win_is_3_5_l1191_119197

noncomputable def expected_value_win : ℝ :=
  (1/8) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

theorem expected_value_of_win_is_3_5 :
  expected_value_win = 3.5 :=
by
  sorry

end expected_value_of_win_is_3_5_l1191_119197


namespace elevation_angle_second_ship_l1191_119131

-- Assume h is the height of the lighthouse.
def h : ℝ := 100

-- Assume d_total is the distance between the two ships.
def d_total : ℝ := 273.2050807568877

-- Assume θ₁ is the angle of elevation from the first ship.
def θ₁ : ℝ := 30

-- Assume θ₂ is the angle of elevation from the second ship.
def θ₂ : ℝ := 45

-- Prove that angle of elevation from the second ship is 45 degrees.
theorem elevation_angle_second_ship : θ₂ = 45 := by
  sorry

end elevation_angle_second_ship_l1191_119131


namespace explicit_form_correct_l1191_119125

-- Define the original function form
def f (a b x : ℝ) := 4*x^3 + a*x^2 + b*x + 5

-- Given tangent line slope condition at x = 1
axiom tangent_slope : ∀ (a b : ℝ), (12 * 1^2 + 2 * a * 1 + b = -12)

-- Given the point (1, f(1)) lies on the tangent line y = -12x
axiom tangent_point : ∀ (a b : ℝ), (4 * 1^3 + a * 1^2 + b * 1 + 5 = -12)

-- Definition for the specific f(x) found in solution
def f_explicit (x : ℝ) := 4*x^3 - 3*x^2 - 18*x + 5

-- Finding maximum and minimum values on interval [-3, 1]
def max_value : ℝ := -76
def min_value : ℝ := 16

theorem explicit_form_correct : 
  ∃ a b : ℝ, 
  (∀ x, f a b x = f_explicit x) ∧ 
  (max_value = 16) ∧ 
  (min_value = -76) := 
by
  sorry

end explicit_form_correct_l1191_119125


namespace trigonometric_identity_l1191_119103

theorem trigonometric_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  (Real.cos (A / 2)) ^ 2 = (Real.cos (B / 2)) ^ 2 + (Real.cos (C / 2)) ^ 2 - 2 * (Real.cos (B / 2)) * (Real.cos (C / 2)) * (Real.sin (A / 2)) :=
sorry

end trigonometric_identity_l1191_119103


namespace hawks_first_half_score_l1191_119180

variable (H1 H2 E : ℕ)

theorem hawks_first_half_score (H1 H2 E : ℕ) 
  (h1 : H1 + H2 + E = 120)
  (h2 : E = H1 + H2 + 16)
  (h3 : H2 = H1 + 8) :
  H1 = 22 :=
by
  sorry

end hawks_first_half_score_l1191_119180


namespace line_passing_through_first_and_third_quadrants_l1191_119170

theorem line_passing_through_first_and_third_quadrants (k : ℝ) (h_nonzero: k ≠ 0) : (k > 0) ↔ (∃ (k_value : ℝ), k_value = 2) :=
sorry

end line_passing_through_first_and_third_quadrants_l1191_119170


namespace AM_GM_inequality_equality_case_of_AM_GM_l1191_119119

theorem AM_GM_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : (x / y) + (y / x) ≥ 2 :=
by
  sorry

theorem equality_case_of_AM_GM (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : ((x / y) + (y / x) = 2) ↔ (x = y) :=
by
  sorry

end AM_GM_inequality_equality_case_of_AM_GM_l1191_119119


namespace train_length_l1191_119175

variable (L : ℝ) -- The length of the train

def length_of_platform : ℝ := 250 -- The length of the platform

def time_to_cross_platform : ℝ := 33 -- Time to cross the platform in seconds

def time_to_cross_pole : ℝ := 18 -- Time to cross the signal pole in seconds

-- The speed of the train is constant whether it crosses the platform or the signal pole.
-- Therefore, we equate the expressions for speed and solve for L.
theorem train_length (h1 : time_to_cross_platform * L = time_to_cross_pole * (L + length_of_platform)) :
  L = 300 :=
by
  -- Proof will be here
  sorry

end train_length_l1191_119175


namespace polynomial_value_l1191_119181

variables (x y p q : ℝ)

theorem polynomial_value (h1 : x + y = -p) (h2 : xy = q) :
  x * (1 + y) - y * (x * y - 1) - x^2 * y = pq + q - p :=
by
  sorry

end polynomial_value_l1191_119181


namespace selection_count_Group3_selection_count_Group4_selection_count_Group5_probability_A_or_B_l1191_119120

/-
  Conditions:
-/
def Group3 : ℕ := 18
def Group4 : ℕ := 12
def Group5 : ℕ := 6
def TotalParticipantsToSelect : ℕ := 12
def TotalFromGroups345 : ℕ := Group3 + Group4 + Group5

/-
  Questions:
  1. Prove that the number of people to be selected from each group using stratified sampling:
\ 2. Prove that the probability of selecting at least one of A or B from Group 5 is 3/5.
-/

theorem selection_count_Group3 : 
  (Group3 * TotalParticipantsToSelect / TotalFromGroups345) = 6 := 
  by sorry

theorem selection_count_Group4 : 
  (Group4 * TotalParticipantsToSelect / TotalFromGroups345) = 4 := 
  by sorry

theorem selection_count_Group5 : 
  (Group5 * TotalParticipantsToSelect / TotalFromGroups345) = 2 := 
  by sorry

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_A_or_B : 
  (combination 6 2 - combination 4 2) / combination 6 2 = 3 / 5 := 
  by sorry

end selection_count_Group3_selection_count_Group4_selection_count_Group5_probability_A_or_B_l1191_119120


namespace longest_side_similar_triangle_l1191_119164

theorem longest_side_similar_triangle 
  (a b c : ℕ) (p : ℕ) (longest_side : ℕ)
  (h1 : a = 6) (h2 : b = 7) (h3 : c = 9) (h4 : p = 110) 
  (h5 : longest_side = 45) :
  ∃ x : ℕ, (6 * x + 7 * x + 9 * x = 110) ∧ (9 * x = longest_side) :=
by
  sorry

end longest_side_similar_triangle_l1191_119164


namespace NorrisSavings_l1191_119122

theorem NorrisSavings : 
  let saved_september := 29
  let saved_october := 25
  let saved_november := 31
  let saved_december := 35
  let saved_january := 40
  saved_september + saved_october + saved_november + saved_december + saved_january = 160 :=
by
  sorry

end NorrisSavings_l1191_119122


namespace sufficient_but_not_necessary_condition_l1191_119162

theorem sufficient_but_not_necessary_condition (a : ℝ) : 
  (a > 0) → (|2 * a + 1| > 1) ∧ ¬((|2 * a + 1| > 1) → (a > 0)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1191_119162


namespace venue_cost_correct_l1191_119183

noncomputable def cost_per_guest : ℤ := 500
noncomputable def johns_guests : ℤ := 50
noncomputable def wifes_guests : ℤ := johns_guests + (60 * johns_guests) / 100
noncomputable def total_wedding_cost : ℤ := 50000
noncomputable def guests_cost : ℤ := wifes_guests * cost_per_guest
noncomputable def venue_cost : ℤ := total_wedding_cost - guests_cost

theorem venue_cost_correct : venue_cost = 10000 := 
  by
  -- Proof can be filled in here.
  sorry

end venue_cost_correct_l1191_119183


namespace range_of_m_l1191_119143

def P (m : ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ (x1 ^ 2 + m * x1 + 1 = 0) ∧ (x2 ^ 2 + m * x2 + 1 = 0) ∧ (x1 < 0) ∧ (x2 < 0)

def Q (m : ℝ) : Prop :=
  ∀ (x : ℝ), 4 * x ^ 2 + 4 * (m - 2) * x + 1 ≠ 0

def P_or_Q (m : ℝ) : Prop :=
  P m ∨ Q m

def P_and_Q (m : ℝ) : Prop :=
  P m ∧ Q m

theorem range_of_m (m : ℝ) : P_or_Q m ∧ ¬P_and_Q m ↔ m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3 :=
by {
  sorry
}

end range_of_m_l1191_119143


namespace union_of_A_and_B_l1191_119111

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x < 4}

theorem union_of_A_and_B : A ∪ B = {x | x > 1} := 
by 
  sorry

end union_of_A_and_B_l1191_119111


namespace score_order_l1191_119128

variable (A B C D : ℕ)

-- Condition 1: B + D = A + C
axiom h1 : B + D = A + C
-- Condition 2: A + B > C + D + 10
axiom h2 : A + B > C + D + 10
-- Condition 3: D > B + C + 20
axiom h3 : D > B + C + 20
-- Condition 4: A + B + C + D = 200
axiom h4 : A + B + C + D = 200

-- Question to prove: Order is Donna > Alice > Brian > Cindy
theorem score_order : D > A ∧ A > B ∧ B > C :=
by
  sorry

end score_order_l1191_119128


namespace not_prime_p_l1191_119188

theorem not_prime_p (x k p : ℕ) (h : x^5 + 2 * x + 3 = p * k) : ¬ (Nat.Prime p) :=
by
  sorry -- Placeholder for the proof

end not_prime_p_l1191_119188


namespace triangle_perimeter_l1191_119198

theorem triangle_perimeter (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (AB AC : ℝ) (angle_A : ℝ)
  (h1 : AB = 4) (h2 : AC = 4) (h3 : angle_A = 60) : 
  AB + AC + AB = 12 :=
by {
  sorry
}

end triangle_perimeter_l1191_119198


namespace pump_fill_time_without_leak_l1191_119190

def time_with_leak := 10
def leak_empty_time := 10

def combined_rate_with_leak := 1 / time_with_leak
def leak_rate := 1 / leak_empty_time

def T : ℝ := 5

theorem pump_fill_time_without_leak
  (time_with_leak : ℝ)
  (leak_empty_time : ℝ)
  (combined_rate_with_leak : ℝ)
  (leak_rate : ℝ)
  (T : ℝ)
  (h1 : combined_rate_with_leak = 1 / time_with_leak)
  (h2 : leak_rate = 1 / leak_empty_time)
  (h_combined : 1 / T - leak_rate = combined_rate_with_leak) :
  T = 5 :=
by {
  sorry
}

end pump_fill_time_without_leak_l1191_119190


namespace gcd_of_cubic_sum_and_linear_is_one_l1191_119167

theorem gcd_of_cubic_sum_and_linear_is_one (n : ℕ) (h : n > 27) : Nat.gcd (n^3 + 8) (n + 3) = 1 :=
sorry

end gcd_of_cubic_sum_and_linear_is_one_l1191_119167


namespace exponent_division_l1191_119171

theorem exponent_division : (19 ^ 11) / (19 ^ 8) = 6859 :=
by
  -- Here we assume the properties of powers and arithmetic operations
  sorry

end exponent_division_l1191_119171


namespace seven_lines_divide_into_29_regions_l1191_119156

open Function

theorem seven_lines_divide_into_29_regions : 
  ∀ n : ℕ, (∀ l m : ℕ, l ≠ m → l < n ∧ m < n) → 1 + n + (n.choose 2) = 29 :=
by
  sorry

end seven_lines_divide_into_29_regions_l1191_119156


namespace minimum_f_l1191_119136

def f (x : ℝ) : ℝ := |x - 2| + |5 - x|

theorem minimum_f : ∃ x, f x = 3 :=
by
  use 3
  unfold f
  sorry

end minimum_f_l1191_119136


namespace nth_equation_pattern_l1191_119165

theorem nth_equation_pattern (n: ℕ) :
  (∀ k : ℕ, 1 ≤ k → ∃ a b c d : ℕ, (a * c ≠ 0) ∧ (b * d ≠ 0) ∧ (a = k) ∧ (b = k + 1) → 
    (a + 3 * (2 * a)) / (b + 3 * (2 * b)) = a / b) :=
by
  sorry

end nth_equation_pattern_l1191_119165


namespace find_line_equation_l1191_119114

theorem find_line_equation : 
  ∃ (m : ℝ), (∀ (x y : ℝ), (2 * x + y - 5 = 0) → (m = -2)) → 
  ∀ (x₀ y₀ : ℝ), (x₀ = -2) ∧ (y₀ = 3) → 
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * x₀ + b * y₀ + c = 0) ∧ (a = 1 ∧ b = -2 ∧ c = 8) := 
by
  sorry

end find_line_equation_l1191_119114


namespace find_k_l1191_119199

noncomputable def series_sum (k : ℝ) : ℝ :=
  3 + ∑' (n : ℕ), (3 + (n + 1) * k) / 4^(n + 1)

theorem find_k : ∃ k : ℝ, series_sum k = 8 ∧ k = 9 :=
by
  use 9
  have h : series_sum 9 = 8 := sorry
  exact ⟨h, rfl⟩

end find_k_l1191_119199


namespace find_x_squared_plus_y_squared_l1191_119126

theorem find_x_squared_plus_y_squared (x y : ℝ) 
  (h1 : (x - y)^2 = 49) (h2 : x * y = -12) : x^2 + y^2 = 25 := 
by 
  sorry

end find_x_squared_plus_y_squared_l1191_119126


namespace log_b_243_values_l1191_119196

theorem log_b_243_values : 
  ∃! (s : Finset ℕ), (∀ b ∈ s, ∃ n : ℕ, b^n = 243) ∧ s.card = 2 :=
by 
  sorry

end log_b_243_values_l1191_119196


namespace median_of_consecutive_integers_l1191_119107

theorem median_of_consecutive_integers (a n : ℤ) (N : ℕ) (h1 : (a + (n - 1)) + (a + (N - n)) = 110) : 
  (2 * a + N - 1) / 2 = 55 := 
by {
  -- The proof goes here.
  sorry
}

end median_of_consecutive_integers_l1191_119107


namespace sufficient_not_necessary_condition_l1191_119176

variable (a b : ℝ)

theorem sufficient_not_necessary_condition (h : a > |b|) : a^2 > b^2 :=
by 
  sorry

end sufficient_not_necessary_condition_l1191_119176


namespace part1_l1191_119178

theorem part1 (m : ℕ) (n : ℕ) (h1 : m = 6 * 10 ^ n + m / 25) : ∃ i : ℕ, m = 625 * 10 ^ (3 * i) := sorry

end part1_l1191_119178


namespace additional_tobacco_acres_l1191_119110

def original_land : ℕ := 1350
def original_ratio_units : ℕ := 9
def new_ratio_units : ℕ := 9

def acres_per_unit := original_land / original_ratio_units

def tobacco_old := 2 * acres_per_unit
def tobacco_new := 5 * acres_per_unit

theorem additional_tobacco_acres :
  tobacco_new - tobacco_old = 450 := by
  sorry

end additional_tobacco_acres_l1191_119110


namespace values_of_b_for_real_root_l1191_119108

noncomputable def polynomial_has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^5 + b * x^4 - x^3 + b * x^2 - x + b = 0

theorem values_of_b_for_real_root :
  {b : ℝ | polynomial_has_real_root b} = {b : ℝ | b ≤ -1 ∨ b ≥ 1} :=
sorry

end values_of_b_for_real_root_l1191_119108


namespace harry_total_hours_l1191_119140

variable (x h y : ℕ)

theorem harry_total_hours :
  ((h + 2 * y) = 42) → ∃ t, t = h + y :=
  by
    sorry -- Proof is omitted as per the instructions

end harry_total_hours_l1191_119140


namespace even_digits_count_1998_l1191_119121

-- Define the function for counting the total number of digits used in the first n positive even integers
def totalDigitsEvenIntegers (n : ℕ) : ℕ :=
  let totalSingleDigit := 4 -- 2, 4, 6, 8
  let numDoubleDigit := 45 -- 10 to 98
  let digitsDoubleDigit := numDoubleDigit * 2
  let numTripleDigit := 450 -- 100 to 998
  let digitsTripleDigit := numTripleDigit * 3
  let numFourDigit := 1499 -- 1000 to 3996
  let digitsFourDigit := numFourDigit * 4
  totalSingleDigit + digitsDoubleDigit + digitsTripleDigit + digitsFourDigit

-- Theorem: The total number of digits used when the first 1998 positive even integers are written is 7440.
theorem even_digits_count_1998 : totalDigitsEvenIntegers 1998 = 7440 :=
  sorry

end even_digits_count_1998_l1191_119121


namespace incorrect_statement_l1191_119153

theorem incorrect_statement :
  let statementA := "The shortest distance between two points is a line segment."
  let statementB := "Vertical angles are congruent."
  let statementC := "Complementary angles of the same measure are congruent."
  let statementD := "There is only one line passing through a point outside a given line that is parallel to the given line."
  (statementA = "correct") ∧ 
  (statementB = "correct") ∧ 
  (statementC = "correct") ∧ 
  (statementD = "incorrect") :=
by
  let statementA := "The shortest distance between two points is a line segment."
  let statementB := "Vertical angles are congruent."
  let statementC := "Complementary angles of the same measure are congruent."
  let statementD := "There is only one line passing through a point outside a given line that is parallel to the given line."
  have hA : statementA = "correct" := sorry
  have hB : statementB = "correct" := sorry
  have hC : statementC = "correct" := sorry
  have hD : statementD = "incorrect" := sorry
  exact ⟨hA, hB, hC, hD⟩

end incorrect_statement_l1191_119153


namespace distance_AB_bounds_l1191_119127

noncomputable def distance_AC : ℕ := 10
noncomputable def distance_AD : ℕ := 10
noncomputable def distance_BE : ℕ := 10
noncomputable def distance_BF : ℕ := 10
noncomputable def distance_AE : ℕ := 12
noncomputable def distance_AF : ℕ := 12
noncomputable def distance_BC : ℕ := 12
noncomputable def distance_BD : ℕ := 12
noncomputable def distance_CD : ℕ := 11
noncomputable def distance_EF : ℕ := 11
noncomputable def distance_CE : ℕ := 5
noncomputable def distance_DF : ℕ := 5

theorem distance_AB_bounds (AB : ℝ) :
  8.8 < AB ∧ AB < 19.2 :=
sorry

end distance_AB_bounds_l1191_119127


namespace max_expr_value_l1191_119193

theorem max_expr_value (a b c d : ℝ) (h_a : -8.5 ≤ a ∧ a ≤ 8.5)
                       (h_b : -8.5 ≤ b ∧ b ≤ 8.5)
                       (h_c : -8.5 ≤ c ∧ c ≤ 8.5)
                       (h_d : -8.5 ≤ d ∧ d ≤ 8.5) :
                       a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 306 :=
sorry

end max_expr_value_l1191_119193


namespace omega_min_value_l1191_119158

def min_omega (ω : ℝ) : Prop :=
  ω > 0 ∧ ∃ k : ℤ, (k ≠ 0 ∧ ω = 8)

theorem omega_min_value (ω : ℝ) (h1 : ω > 0) (h2 : ∃ k : ℤ, k ≠ 0 ∧ (k * 2 * π) / ω = π / 4) : 
  ω = 8 :=
by
  sorry

end omega_min_value_l1191_119158


namespace max_divisor_f_l1191_119112

-- Given definition
def f (n : ℕ) : ℕ := (2 * n + 7) * 3 ^ n + 9

-- Main theorem to be proved
theorem max_divisor_f :
  ∃ m : ℕ, (∀ n : ℕ, 0 < n → m ∣ f n) ∧ m = 36 :=
by
  -- The proof would go here
  sorry

end max_divisor_f_l1191_119112


namespace tournament_teams_l1191_119169

theorem tournament_teams (n : ℕ) (H : 240 = 2 * n * (n - 1)) : n = 12 := 
by sorry

end tournament_teams_l1191_119169


namespace evaluate_expression_l1191_119145

theorem evaluate_expression :
  let a := 3 * 4 * 5
  let b := (1 : ℝ) / 3
  let c := (1 : ℝ) / 4
  let d := (1 : ℝ) / 5
  (a : ℝ) * (b + c - d) = 23 := by
  sorry

end evaluate_expression_l1191_119145


namespace eighth_L_prime_is_31_l1191_119102

def setL := {n : ℕ | n > 0 ∧ n % 3 = 1}

def isLPrime (n : ℕ) : Prop :=
  n ∈ setL ∧ n ≠ 1 ∧ ∀ m ∈ setL, (m ∣ n) → (m = 1 ∨ m = n)

theorem eighth_L_prime_is_31 : 
  ∃ n ∈ setL, isLPrime n ∧ 
  (∀ k, (∃ m ∈ setL, isLPrime m ∧ m < n) → k < 8 → m ≠ n) :=
by sorry

end eighth_L_prime_is_31_l1191_119102


namespace correct_statement_is_c_l1191_119173

-- Definitions corresponding to conditions
def lateral_surface_of_cone_unfolds_into_isosceles_triangle : Prop :=
  false -- This is false because it unfolds into a sector.

def prism_with_two_congruent_bases_other_faces_rectangles : Prop :=
  false -- This is false because the bases are congruent and parallel, and all other faces are parallelograms.

def frustum_complemented_with_pyramid_forms_new_pyramid : Prop :=
  true -- This is true, as explained in the solution.

def point_on_lateral_surface_of_truncated_cone_has_countless_generatrices : Prop :=
  false -- This is false because there is exactly one generatrix through such a point.

-- The main proof statement
theorem correct_statement_is_c :
  ¬lateral_surface_of_cone_unfolds_into_isosceles_triangle ∧
  ¬prism_with_two_congruent_bases_other_faces_rectangles ∧
  frustum_complemented_with_pyramid_forms_new_pyramid ∧
  ¬point_on_lateral_surface_of_truncated_cone_has_countless_generatrices :=
by
  -- The proof involves evaluating all the conditions above.
  sorry

end correct_statement_is_c_l1191_119173


namespace lg_45_eq_l1191_119182

variable (m n : ℝ)
axiom lg_2 : Real.log 2 = m
axiom lg_3 : Real.log 3 = n

theorem lg_45_eq : Real.log 45 = 1 - m + 2 * n := by
  -- proof to be filled in
  sorry

end lg_45_eq_l1191_119182


namespace initial_deposit_l1191_119166

theorem initial_deposit (x : ℝ) 
  (h1 : x - (1 / 4) * x - (4 / 9) * ((3 / 4) * x) - 640 = (3 / 20) * x) 
  : x = 2400 := 
by 
  sorry

end initial_deposit_l1191_119166


namespace non_zero_number_is_nine_l1191_119104

theorem non_zero_number_is_nine (x : ℝ) (h1 : x ≠ 0) (h2 : (x + x^2) / 2 = 5 * x) : x = 9 :=
by
  sorry

end non_zero_number_is_nine_l1191_119104


namespace count_numbers_with_digit_7_count_numbers_divisible_by_3_or_5_l1191_119152

-- Statement for Question 1
theorem count_numbers_with_digit_7 :
  ∃ n, n = 19 ∧ (∀ k, (k < 100 → (k / 10 = 7 ∨ k % 10 = 7) ↔ k ≠ 77)) :=
sorry

-- Statement for Question 2
theorem count_numbers_divisible_by_3_or_5 :
  ∃ n, n = 47 ∧ (∀ k, (k < 100 → (k % 3 = 0 ∨ k % 5 = 0)) ↔ (k % 15 = 0)) :=
sorry

end count_numbers_with_digit_7_count_numbers_divisible_by_3_or_5_l1191_119152


namespace sheila_weekly_earnings_is_288_l1191_119144

-- Define the conditions as constants.
def sheilaWorksHoursPerDay (d : String) : ℕ :=
  if d = "Monday" ∨ d = "Wednesday" ∨ d = "Friday" then 8
  else if d = "Tuesday" ∨ d = "Thursday" then 6
  else 0

def hourlyWage : ℕ := 8

-- Calculate total weekly earnings based on conditions.
def weeklyEarnings : ℕ :=
  (sheilaWorksHoursPerDay "Monday" + sheilaWorksHoursPerDay "Wednesday" + sheilaWorksHoursPerDay "Friday") * hourlyWage +
  (sheilaWorksHoursPerDay "Tuesday" + sheilaWorksHoursPerDay "Thursday") * hourlyWage

-- The Lean statement for the proof.
theorem sheila_weekly_earnings_is_288 : weeklyEarnings = 288 :=
  by
    sorry

end sheila_weekly_earnings_is_288_l1191_119144


namespace find_function_perfect_square_condition_l1191_119185

theorem find_function_perfect_square_condition (g : ℕ → ℕ)
  (h : ∀ m n : ℕ, ∃ k : ℕ, (g m + n) * (g n + m) = k * k) :
  ∃ c : ℕ, ∀ m : ℕ, g m = m + c :=
sorry

end find_function_perfect_square_condition_l1191_119185


namespace ant_probability_after_10_minutes_l1191_119129

-- Definitions based on the conditions given in the problem
def ant_start_at_A := true
def moves_each_minute (n : ℕ) := n == 10
def blue_dots (x y : ℤ) : Prop := 
  (x == 0 ∨ y == 0) ∧ (x + y) % 2 == 0
def A_at_center (x y : ℤ) : Prop := x == 0 ∧ y == 0
def B_north_of_A (x y : ℤ) : Prop := x == 0 ∧ y == 1

-- The probability we need to prove
def probability_ant_at_B_after_10_minutes := 1 / 9

-- We state our proof problem
theorem ant_probability_after_10_minutes :
  ant_start_at_A ∧ moves_each_minute 10 ∧ blue_dots 0 0 ∧ blue_dots 0 1 ∧ A_at_center 0 0 ∧ B_north_of_A 0 1
  → probability_ant_at_B_after_10_minutes = 1 / 9 := 
sorry

end ant_probability_after_10_minutes_l1191_119129


namespace faye_has_62_pieces_of_candy_l1191_119105

-- Define initial conditions
def initialCandy : Nat := 47
def eatenCandy : Nat := 25
def receivedCandy : Nat := 40

-- Define the resulting number of candies after eating and receiving more candies
def resultingCandy : Nat := initialCandy - eatenCandy + receivedCandy

-- State the theorem and provide the proof
theorem faye_has_62_pieces_of_candy :
  resultingCandy = 62 :=
by
  -- proof goes here
  sorry

end faye_has_62_pieces_of_candy_l1191_119105
