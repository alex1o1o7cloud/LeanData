import Mathlib

namespace NUMINAMATH_GPT_sqrt_4_of_10000000_eq_l1265_126571

noncomputable def sqrt_4_of_10000000 : Real := Real.sqrt (Real.sqrt 10000000)

theorem sqrt_4_of_10000000_eq :
  sqrt_4_of_10000000 = 10 * Real.sqrt (Real.sqrt 10) := by
sorry

end NUMINAMATH_GPT_sqrt_4_of_10000000_eq_l1265_126571


namespace NUMINAMATH_GPT_find_x_l1265_126592

theorem find_x (x : ℤ) (h : 3 * x = (26 - x) + 10) : x = 9 :=
by
  -- proof steps would be provided here
  sorry

end NUMINAMATH_GPT_find_x_l1265_126592


namespace NUMINAMATH_GPT_sales_tax_difference_l1265_126562

theorem sales_tax_difference (price : ℝ) (tax_rate1 tax_rate2 : ℝ) :
  price = 50 → tax_rate1 = 0.075 → tax_rate2 = 0.065 →
  (price * tax_rate1 - price * tax_rate2 = 0.5) :=
by
  intros
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l1265_126562


namespace NUMINAMATH_GPT_evaluate_g_at_3_l1265_126597

def g (x: ℝ) := 5 * x^3 - 4 * x^2 + 3 * x - 7

theorem evaluate_g_at_3 : g 3 = 101 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_g_at_3_l1265_126597


namespace NUMINAMATH_GPT_inequality_am_gm_l1265_126559

theorem inequality_am_gm (a b : ℝ) (h₀ : 0 < a) (h₁ : a < 1) (h₂ : 0 < b) (h₃ : b < 1) :
  1 + a + b > 3 * Real.sqrt (a * b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l1265_126559


namespace NUMINAMATH_GPT_ratio_of_60_to_12_l1265_126572

theorem ratio_of_60_to_12 : 60 / 12 = 5 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_60_to_12_l1265_126572


namespace NUMINAMATH_GPT_tennis_handshakes_l1265_126525

-- Define the participants and the condition
def number_of_participants := 8
def handshakes_no_partner (n : ℕ) := n * (n - 2) / 2

-- Prove that the number of handshakes is 24
theorem tennis_handshakes : handshakes_no_partner number_of_participants = 24 := by
  -- Since we are skipping the proof for now
  sorry

end NUMINAMATH_GPT_tennis_handshakes_l1265_126525


namespace NUMINAMATH_GPT_reciprocal_of_repeating_decimal_6_l1265_126505

-- Define a repeating decimal .\overline{6}
noncomputable def repeating_decimal_6 : ℚ := 2 / 3

-- The theorem statement: reciprocal of .\overline{6} is 3/2
theorem reciprocal_of_repeating_decimal_6 :
  repeating_decimal_6⁻¹ = (3 / 2) :=
sorry

end NUMINAMATH_GPT_reciprocal_of_repeating_decimal_6_l1265_126505


namespace NUMINAMATH_GPT_tammy_avg_speed_second_day_l1265_126508

theorem tammy_avg_speed_second_day (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) :
  v + 0.5 = 4 :=
sorry

end NUMINAMATH_GPT_tammy_avg_speed_second_day_l1265_126508


namespace NUMINAMATH_GPT_symmetry_about_origin_l1265_126528

def Point : Type := ℝ × ℝ

def A : Point := (2, -1)
def B : Point := (-2, 1)

theorem symmetry_about_origin (A B : Point) : A = (2, -1) ∧ B = (-2, 1) → B = (-A.1, -A.2) :=
by
  sorry

end NUMINAMATH_GPT_symmetry_about_origin_l1265_126528


namespace NUMINAMATH_GPT_non_egg_laying_chickens_count_l1265_126532

noncomputable def num_chickens : ℕ := 80
noncomputable def roosters : ℕ := num_chickens / 4
noncomputable def hens : ℕ := num_chickens - roosters
noncomputable def egg_laying_hens : ℕ := (3 * hens) / 4
noncomputable def hens_on_vacation : ℕ := (2 * egg_laying_hens) / 10
noncomputable def remaining_hens_after_vacation : ℕ := egg_laying_hens - hens_on_vacation
noncomputable def ill_hens : ℕ := (1 * remaining_hens_after_vacation) / 10
noncomputable def non_egg_laying_chickens : ℕ := roosters + hens_on_vacation + ill_hens

theorem non_egg_laying_chickens_count : non_egg_laying_chickens = 33 := by
  sorry

end NUMINAMATH_GPT_non_egg_laying_chickens_count_l1265_126532


namespace NUMINAMATH_GPT_riza_son_age_l1265_126520

theorem riza_son_age (R S : ℕ) (h1 : R = S + 25) (h2 : R + S = 105) : S = 40 :=
by
  sorry

end NUMINAMATH_GPT_riza_son_age_l1265_126520


namespace NUMINAMATH_GPT_measure_of_angle_C_l1265_126509

theorem measure_of_angle_C (C D : ℝ) (h1 : C + D = 180) (h2 : C = 7 * D) : C = 157.5 := 
by 
  sorry

end NUMINAMATH_GPT_measure_of_angle_C_l1265_126509


namespace NUMINAMATH_GPT_equal_five_digit_number_sets_l1265_126568

def five_digit_numbers_not_div_5 : ℕ :=
  9 * 10^3 * 8

def five_digit_numbers_first_two_not_5 : ℕ :=
  8 * 9 * 10^3

theorem equal_five_digit_number_sets :
  five_digit_numbers_not_div_5 = five_digit_numbers_first_two_not_5 :=
by
  repeat { sorry }

end NUMINAMATH_GPT_equal_five_digit_number_sets_l1265_126568


namespace NUMINAMATH_GPT_average_eq_16_l1265_126585

noncomputable def x : ℝ := 20
noncomputable def y : ℝ := 12

theorem average_eq_16 (h1 : 3 = 0.15 * x) (h2 : 3 = 0.25 * y) : (x + y) / 2 = 16 := by
  sorry

end NUMINAMATH_GPT_average_eq_16_l1265_126585


namespace NUMINAMATH_GPT_evaluate_expression_l1265_126594

theorem evaluate_expression (x : ℤ) (h : x = 2) : 20 - 2 * (3 * x^2 - 4 * x + 8) = -4 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1265_126594


namespace NUMINAMATH_GPT_hyperbola_lattice_points_count_l1265_126524

def is_lattice_point (x y : ℤ) : Prop :=
  x^2 - 2 * y^2 = 2000^2

def count_lattice_points (points : List (ℤ × ℤ)) : ℕ :=
  points.length

theorem hyperbola_lattice_points_count : count_lattice_points [(2000, 0), (-2000, 0)] = 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_lattice_points_count_l1265_126524


namespace NUMINAMATH_GPT_monotone_decreasing_f_find_a_value_l1265_126593

-- Condition declarations
variables (a b : ℝ) (h_a_pos : a > 0) (max_val min_val : ℝ)
noncomputable def f (x : ℝ) := x + (a / x) + b

-- Problem 1: Prove that f is monotonically decreasing in (0, sqrt(a)]
theorem monotone_decreasing_f : 
  (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 ≤ Real.sqrt a → f a b x1 > f a b x2) :=
sorry

-- Conditions for Problem 2
variable (hf_inc : ∀ x1 x2 : ℝ, Real.sqrt a ≤ x1 ∧ x1 < x2 → f a b x1 < f a b x2)
variable (h_max : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f a b x ≤ 5)
variable (h_min : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f a b x ≥ 3)

-- Problem 2: Find the value of a
theorem find_a_value : a = 6 :=
sorry

end NUMINAMATH_GPT_monotone_decreasing_f_find_a_value_l1265_126593


namespace NUMINAMATH_GPT_triangle_area_ABC_l1265_126552

-- Define the vertices of the triangle
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, 9)
def C : ℝ × ℝ := (7, 6)

-- Define a function to calculate the area of a triangle given its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Prove that the area of the triangle with the given vertices is 15
theorem triangle_area_ABC : triangle_area A B C = 15 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_triangle_area_ABC_l1265_126552


namespace NUMINAMATH_GPT_frank_money_made_l1265_126518

theorem frank_money_made
  (spent_on_blades : ℕ)
  (number_of_games : ℕ)
  (cost_per_game : ℕ)
  (total_cost_games := number_of_games * cost_per_game)
  (total_money_made := spent_on_blades + total_cost_games)
  (H1 : spent_on_blades = 11)
  (H2 : number_of_games = 4)
  (H3 : cost_per_game = 2) :
  total_money_made = 19 :=
by
  sorry

end NUMINAMATH_GPT_frank_money_made_l1265_126518


namespace NUMINAMATH_GPT_age_ratio_l1265_126584

theorem age_ratio (V A : ℕ) (h1 : V - 5 = 16) (h2 : V * 2 = 7 * A) :
  (V + 4) * 2 = (A + 4) * 5 := 
sorry

end NUMINAMATH_GPT_age_ratio_l1265_126584


namespace NUMINAMATH_GPT_increasing_interval_l1265_126563

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval : {x : ℝ | 2 < x} = { x : ℝ | (x - 3) * Real.exp x > 0 } :=
by
  sorry

end NUMINAMATH_GPT_increasing_interval_l1265_126563


namespace NUMINAMATH_GPT_arithmetic_square_root_of_16_l1265_126531

theorem arithmetic_square_root_of_16 : ∃! (x : ℝ), x^2 = 16 ∧ x ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_square_root_of_16_l1265_126531


namespace NUMINAMATH_GPT_initial_amount_is_53_l1265_126547

variable (X : ℕ) -- Initial amount of money Olivia had
variable (ATM_collect : ℕ := 91) -- Money collected from ATM
variable (supermarket_spent_diff : ℕ := 39) -- Spent 39 dollars more at the supermarket
variable (money_left : ℕ := 14) -- Money left after supermarket

-- Define the final amount Olivia had
def final_amount (X ATM_collect supermarket_spent_diff : ℕ) : ℕ :=
  X + ATM_collect - (ATM_collect + supermarket_spent_diff)

-- Theorem stating that the initial amount X was 53 dollars
theorem initial_amount_is_53 : final_amount X ATM_collect supermarket_spent_diff = money_left → X = 53 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_initial_amount_is_53_l1265_126547


namespace NUMINAMATH_GPT_range_of_a_l1265_126550

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1265_126550


namespace NUMINAMATH_GPT_range_of_a_l1265_126511

theorem range_of_a (a : ℝ) : 
  (M = {x : ℝ | 2 * x + 1 < 3}) → 
  (N = {x : ℝ | x < a}) → 
  (M ∩ N = N) ↔ a ≤ 1 :=
by
  let M := {x : ℝ | 2 * x + 1 < 3}
  let N := {x : ℝ | x < a}
  simp [Set.subset_def]
  sorry

end NUMINAMATH_GPT_range_of_a_l1265_126511


namespace NUMINAMATH_GPT_largest_unreachable_integer_l1265_126548

theorem largest_unreachable_integer : ∃ n : ℕ, (¬ ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ 8 * a + 11 * b = n)
  ∧ ∀ m : ℕ, m > n → (∃ a b : ℕ, 0 < a ∧ 0 < b ∧ 8 * a + 11 * b = m) := sorry

end NUMINAMATH_GPT_largest_unreachable_integer_l1265_126548


namespace NUMINAMATH_GPT_problem_1_problem_2_l1265_126501

def A := {x : ℝ | 1 < 2 * x - 1 ∧ 2 * x - 1 < 7}
def B := {x : ℝ | x^2 - 2 * x - 3 < 0}

theorem problem_1 : A ∩ B = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

theorem problem_2 : (A ∪ B)ᶜ = {x : ℝ | x ≤ -1 ∨ x ≥ 4} :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1265_126501


namespace NUMINAMATH_GPT_geometric_series_sum_example_l1265_126503

def geometric_series_sum (a r : ℤ) (n : ℕ) : ℤ :=
  a * ((r ^ n - 1) / (r - 1))

theorem geometric_series_sum_example : geometric_series_sum 2 (-3) 8 = -3280 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_example_l1265_126503


namespace NUMINAMATH_GPT_arthur_money_left_l1265_126598

theorem arthur_money_left {initial_amount spent_fraction : ℝ} (h_initial : initial_amount = 200) (h_fraction : spent_fraction = 4 / 5) : 
  (initial_amount - spent_fraction * initial_amount = 40) :=
by
  sorry

end NUMINAMATH_GPT_arthur_money_left_l1265_126598


namespace NUMINAMATH_GPT_free_fall_time_l1265_126565

theorem free_fall_time (h : ℝ) (t : ℝ) (h_eq : h = 4.9 * t^2) (h_val : h = 490) : t = 10 :=
by
  sorry

end NUMINAMATH_GPT_free_fall_time_l1265_126565


namespace NUMINAMATH_GPT_tenth_term_of_arithmetic_progression_l1265_126512

variable (a d n T_n : ℕ)

def arithmetic_progression (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem tenth_term_of_arithmetic_progression :
  arithmetic_progression 8 2 10 = 26 :=
  by
  sorry

end NUMINAMATH_GPT_tenth_term_of_arithmetic_progression_l1265_126512


namespace NUMINAMATH_GPT_roots_exist_l1265_126560

theorem roots_exist (a : ℝ) : ∃ x : ℝ, a * x^2 - x = 0 := by
  sorry

end NUMINAMATH_GPT_roots_exist_l1265_126560


namespace NUMINAMATH_GPT_div_by_seven_equiv_l1265_126583

-- Given integers a and b, prove that 10a + b is divisible by 7 if and only if a - 2b is divisible by 7.
theorem div_by_seven_equiv (a b : ℤ) : (10 * a + b) % 7 = 0 ↔ (a - 2 * b) % 7 = 0 := sorry

end NUMINAMATH_GPT_div_by_seven_equiv_l1265_126583


namespace NUMINAMATH_GPT_dividend_rate_correct_l1265_126543

def stock_price : ℝ := 150
def yield_percentage : ℝ := 0.08
def dividend_rate : ℝ := stock_price * yield_percentage

theorem dividend_rate_correct : dividend_rate = 12 := by
  sorry

end NUMINAMATH_GPT_dividend_rate_correct_l1265_126543


namespace NUMINAMATH_GPT_multiple_7_proposition_l1265_126542

theorem multiple_7_proposition : (47 % 7 ≠ 0 ∨ 49 % 7 = 0) → True :=
by
  intros h
  sorry

end NUMINAMATH_GPT_multiple_7_proposition_l1265_126542


namespace NUMINAMATH_GPT_find_n_l1265_126579

theorem find_n (n : ℕ) (hn : (Nat.choose n 2 : ℚ) / 2^n = 10 / 32) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1265_126579


namespace NUMINAMATH_GPT_average_temperature_l1265_126502

theorem average_temperature :
  ∀ (T : ℝ) (Tt : ℝ),
  -- Conditions
  (43 + T + T + T) / 4 = 48 → 
  Tt = 35 →
  -- Proof
  (T + T + T + Tt) / 4 = 46 :=
by
  intros T Tt H1 H2
  sorry

end NUMINAMATH_GPT_average_temperature_l1265_126502


namespace NUMINAMATH_GPT_find_triangle_area_l1265_126557

noncomputable def triangle_area_problem
  (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 14) : ℝ :=
  (1 / 2) * a * b

theorem find_triangle_area
  (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 14) :
  triangle_area_problem a b h1 h2 = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_find_triangle_area_l1265_126557


namespace NUMINAMATH_GPT_election_votes_l1265_126586

theorem election_votes (V : ℝ) (h1 : 0.56 * V - 0.44 * V = 288) : 0.56 * V = 1344 :=
by 
  sorry

end NUMINAMATH_GPT_election_votes_l1265_126586


namespace NUMINAMATH_GPT_points_on_ellipse_l1265_126575

noncomputable def x (t : ℝ) : ℝ := (3 - t^2) / (1 + t^2)
noncomputable def y (t : ℝ) : ℝ := 4 * t / (1 + t^2)

theorem points_on_ellipse : ∀ t : ℝ, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  (x t / a)^2 + (y t / b)^2 = 1 := 
sorry

end NUMINAMATH_GPT_points_on_ellipse_l1265_126575


namespace NUMINAMATH_GPT_cos_sum_series_l1265_126533

theorem cos_sum_series : 
  (∑' n : ℤ, if (n % 2 = 1 ∨ n % 2 = -1) then (1 : ℝ) / (n : ℝ)^2 else 0) = (π^2) / 8 := by
  sorry

end NUMINAMATH_GPT_cos_sum_series_l1265_126533


namespace NUMINAMATH_GPT_democrats_and_republicans_seating_l1265_126500

theorem democrats_and_republicans_seating : 
  let n := 6
  let factorial := Nat.factorial
  let arrangements := (factorial n) * (factorial n)
  let circular_table := 1
  arrangements * circular_table = 518400 :=
by 
  sorry

end NUMINAMATH_GPT_democrats_and_republicans_seating_l1265_126500


namespace NUMINAMATH_GPT_first_year_fee_correct_l1265_126513

noncomputable def first_year_fee (n : ℕ) (annual_increase : ℕ) (sixth_year_fee : ℕ) : ℕ :=
  sixth_year_fee - (n - 1) * annual_increase

theorem first_year_fee_correct (n annual_increase sixth_year_fee value : ℕ) 
  (h_n : n = 6) (h_annual_increase : annual_increase = 10) 
  (h_sixth_year_fee : sixth_year_fee = 130) (h_value : value = 80) :
  first_year_fee n annual_increase sixth_year_fee = value :=
by {
  sorry
}

end NUMINAMATH_GPT_first_year_fee_correct_l1265_126513


namespace NUMINAMATH_GPT_opposite_of_neg_2023_l1265_126510

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_2023_l1265_126510


namespace NUMINAMATH_GPT_tax_free_value_is_500_l1265_126539

-- Definitions of the given conditions
def total_value : ℝ := 730
def paid_tax : ℝ := 18.40
def tax_rate : ℝ := 0.08

-- Definition of the excess value
def excess_value (E : ℝ) := tax_rate * E = paid_tax

-- Definition of the tax-free threshold value
def tax_free_limit (V : ℝ) := total_value - (paid_tax / tax_rate) = V

-- The theorem to be proven
theorem tax_free_value_is_500 : 
  ∃ V : ℝ, (total_value - (paid_tax / tax_rate) = V) ∧ V = 500 :=
  by
    sorry -- Proof to be completed

end NUMINAMATH_GPT_tax_free_value_is_500_l1265_126539


namespace NUMINAMATH_GPT_perp_lines_l1265_126519

noncomputable def line_1 (k : ℝ) : ℝ → ℝ → ℝ := λ x y => (k - 3) * x + (5 - k) * y + 1
noncomputable def line_2 (k : ℝ) : ℝ → ℝ → ℝ := λ x y => 2 * (k - 3) * x - 2 * y + 3

theorem perp_lines (k : ℝ) : 
  let l1 := line_1 k
  let l2 := line_2 k
  (∀ x y, l1 x y = 0 → l2 x y = 0 → (k = 1 ∨ k = 4)) :=
by
    sorry

end NUMINAMATH_GPT_perp_lines_l1265_126519


namespace NUMINAMATH_GPT_f_prime_at_1_l1265_126526

def f (x : ℝ) : ℝ := 3 * x^3 - 4 * x^2 + 10 * x - 5

theorem f_prime_at_1 : (deriv f 1) = 11 :=
by
  sorry

end NUMINAMATH_GPT_f_prime_at_1_l1265_126526


namespace NUMINAMATH_GPT_bobby_shoes_multiple_l1265_126556

theorem bobby_shoes_multiple (B M : ℕ) (hBonny : 13 = 2 * B - 5) (hBobby : 27 = M * B) : 
  M = 3 :=
by 
  sorry

end NUMINAMATH_GPT_bobby_shoes_multiple_l1265_126556


namespace NUMINAMATH_GPT_triangle_proof_l1265_126587

theorem triangle_proof (a b : ℝ) (cosA : ℝ) (ha : a = 6) (hb : b = 5) (hcosA : cosA = -4 / 5) :
  (∃ B : ℝ, B = 30) ∧ (∃ area : ℝ, area = (9 * Real.sqrt 3 - 12) / 2) :=
  by
  sorry

end NUMINAMATH_GPT_triangle_proof_l1265_126587


namespace NUMINAMATH_GPT_preceding_integer_binary_l1265_126540

--- The conditions as definitions in Lean 4

def M := 0b101100 -- M is defined as binary '101100' which is decimal 44
def preceding_binary (n : Nat) : Nat := n - 1 -- Define a function to get the preceding integer in binary

--- The proof problem statement in Lean 4
theorem preceding_integer_binary :
  preceding_binary M = 0b101011 :=
by
  sorry

end NUMINAMATH_GPT_preceding_integer_binary_l1265_126540


namespace NUMINAMATH_GPT_jane_last_segment_speed_l1265_126581

theorem jane_last_segment_speed :
  let total_distance := 120  -- in miles
  let total_time := (75 / 60)  -- in hours
  let segment_time := (25 / 60)  -- in hours
  let speed1 := 75  -- in mph
  let speed2 := 80  -- in mph
  let overall_avg_speed := total_distance / total_time
  let x := (3 * overall_avg_speed) - speed1 - speed2
  x = 133 :=
by { sorry }

end NUMINAMATH_GPT_jane_last_segment_speed_l1265_126581


namespace NUMINAMATH_GPT_greatest_value_a_plus_b_l1265_126589

theorem greatest_value_a_plus_b (a b : ℝ) (h1 : a^2 + b^2 = 130) (h2 : a * b = 45) : a + b = 2 * Real.sqrt 55 :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_a_plus_b_l1265_126589


namespace NUMINAMATH_GPT_wall_width_l1265_126541

theorem wall_width (w h l : ℝ)
  (h_eq_6w : h = 6 * w)
  (l_eq_7h : l = 7 * h)
  (V_eq : w * h * l = 86436) :
  w = 7 :=
by
  sorry

end NUMINAMATH_GPT_wall_width_l1265_126541


namespace NUMINAMATH_GPT_number_of_different_towers_l1265_126516

theorem number_of_different_towers
  (red blue yellow : ℕ)
  (total_height : ℕ)
  (total_cubes : ℕ)
  (discarded_cubes : ℕ)
  (ways_to_leave_out : ℕ)
  (multinomial_coefficient : ℕ) : 
  red = 3 → blue = 4 → yellow = 5 → total_height = 10 → total_cubes = 12 → discarded_cubes = 2 →
  ways_to_leave_out = 66 → multinomial_coefficient = 4200 →
  (ways_to_leave_out * multinomial_coefficient) = 277200 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_number_of_different_towers_l1265_126516


namespace NUMINAMATH_GPT_prove_cardinality_l1265_126527

-- Definitions used in Lean 4 Statement adapted from conditions
variable (a b : ℕ)
variable (A B : Finset ℕ)

-- Hypotheses
variable (ha : a > 0)
variable (hb : b > 0)
variable (h_disjoint : Disjoint A B)
variable (h_condition : ∀ i ∈ (A ∪ B), i + a ∈ A ∨ i - b ∈ B)

-- The statement to prove
theorem prove_cardinality (a b : ℕ) (A B : Finset ℕ)
  (ha : a > 0) (hb : b > 0) (h_disjoint : Disjoint A B)
  (h_condition : ∀ i ∈ (A ∪ B), i + a ∈ A ∨ i - b ∈ B) :
  a * A.card = b * B.card :=
by 
  sorry

end NUMINAMATH_GPT_prove_cardinality_l1265_126527


namespace NUMINAMATH_GPT_largest_possible_a_l1265_126578

theorem largest_possible_a (a b c e : ℕ) (h1 : a < 2 * b) (h2 : b < 3 * c) (h3 : c < 5 * e) (h4 : e < 100) : a ≤ 2961 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_a_l1265_126578


namespace NUMINAMATH_GPT_decimal_to_base8_conversion_l1265_126590

-- Define the base and the number in decimal.
def base : ℕ := 8
def decimal_number : ℕ := 127

-- Define the expected representation in base 8.
def expected_base8_representation : ℕ := 177

-- Theorem stating that conversion of 127 in base 10 to base 8 yields 177
theorem decimal_to_base8_conversion : Nat.ofDigits base (Nat.digits base decimal_number) = expected_base8_representation := 
by
  sorry

end NUMINAMATH_GPT_decimal_to_base8_conversion_l1265_126590


namespace NUMINAMATH_GPT_sum_first_11_terms_of_arithmetic_sequence_l1265_126582

noncomputable def sum_arithmetic_sequence (n : ℕ) (a1 an : ℤ) : ℤ :=
  n * (a1 + an) / 2

theorem sum_first_11_terms_of_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h1 : S n = sum_arithmetic_sequence n (a 1) (a n))
  (h2 : a 3 + a 6 + a 9 = 60) : S 11 = 220 :=
sorry

end NUMINAMATH_GPT_sum_first_11_terms_of_arithmetic_sequence_l1265_126582


namespace NUMINAMATH_GPT_hillary_descending_rate_is_1000_l1265_126595

-- Definitions from the conditions
def base_to_summit_distance : ℕ := 5000
def hillary_departure_time : ℕ := 6
def hillary_climbing_rate : ℕ := 800
def eddy_climbing_rate : ℕ := 500
def hillary_stop_distance_from_summit : ℕ := 1000
def hillary_and_eddy_pass_time : ℕ := 12

-- Derived definitions
def hillary_climbing_time : ℕ := (base_to_summit_distance - hillary_stop_distance_from_summit) / hillary_climbing_rate
def hillary_stop_time : ℕ := hillary_departure_time + hillary_climbing_time
def eddy_climbing_time_at_pass : ℕ := hillary_and_eddy_pass_time - hillary_departure_time
def eddy_climbed_distance : ℕ := eddy_climbing_rate * eddy_climbing_time_at_pass
def hillary_distance_descended_at_pass : ℕ := (base_to_summit_distance - hillary_stop_distance_from_summit) - eddy_climbed_distance
def hillary_descending_time : ℕ := hillary_and_eddy_pass_time - hillary_stop_time 

def hillary_descending_rate : ℕ := hillary_distance_descended_at_pass / hillary_descending_time

-- Statement to prove
theorem hillary_descending_rate_is_1000 : hillary_descending_rate = 1000 := 
by
  sorry

end NUMINAMATH_GPT_hillary_descending_rate_is_1000_l1265_126595


namespace NUMINAMATH_GPT_unique_2_digit_cyclic_permutation_divisible_l1265_126546

def is_cyclic_permutation (n : ℕ) (M : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i < n → j < n → M i = M j

def M (a : Fin 2 → ℕ) : ℕ := a 0 * 10 + a 1

theorem unique_2_digit_cyclic_permutation_divisible (a : Fin 2 → ℕ) (h0 : ∀ i, a i ≠ 0) :
  (M a) % (a 1 * 10 + a 0) = 0 → 
  (M a = 11) :=
by
  sorry

end NUMINAMATH_GPT_unique_2_digit_cyclic_permutation_divisible_l1265_126546


namespace NUMINAMATH_GPT_solve_system_of_equations_l1265_126577

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x * y * (x + y) = 30 ∧ x^3 + y^3 = 35 ∧ ((x = 3 ∧ y = 2) ∨ (x = 2 ∧ y = 3)) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1265_126577


namespace NUMINAMATH_GPT_not_right_triangle_angle_ratio_l1265_126538

theorem not_right_triangle_angle_ratio (A B C : ℝ) (h₁ : A / B = 3 / 4) (h₂ : B / C = 4 / 5) (h₃ : A + B + C = 180) : ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
sorry

end NUMINAMATH_GPT_not_right_triangle_angle_ratio_l1265_126538


namespace NUMINAMATH_GPT_total_homework_time_l1265_126558

variable (num_math_problems num_social_studies_problems num_science_problems : ℕ)
variable (time_per_math_problem time_per_social_studies_problem time_per_science_problem : ℝ)

/-- Prove that the total time taken by Brooke to answer all his homework problems is 48 minutes -/
theorem total_homework_time :
  num_math_problems = 15 →
  num_social_studies_problems = 6 →
  num_science_problems = 10 →
  time_per_math_problem = 2 →
  time_per_social_studies_problem = 0.5 →
  time_per_science_problem = 1.5 →
  (num_math_problems * time_per_math_problem + num_social_studies_problems * time_per_social_studies_problem + num_science_problems * time_per_science_problem) = 48 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_total_homework_time_l1265_126558


namespace NUMINAMATH_GPT_find_C_l1265_126515

theorem find_C (C : ℤ) (h : 2 * C - 3 = 11) : C = 7 :=
sorry

end NUMINAMATH_GPT_find_C_l1265_126515


namespace NUMINAMATH_GPT_calculate_expression_l1265_126534

theorem calculate_expression (a : ℤ) (h : a = -2) : a^3 - a^2 = -12 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1265_126534


namespace NUMINAMATH_GPT_trailing_zeroes_500_fact_l1265_126504

-- Define a function to count multiples of a given number in a range
def countMultiples (n m : Nat) : Nat :=
  m / n

-- Define a function to count trailing zeroes in the factorial
def trailingZeroesFactorial (n : Nat) : Nat :=
  countMultiples 5 n + countMultiples (5^2) n + countMultiples (5^3) n + countMultiples (5^4) n

theorem trailing_zeroes_500_fact : trailingZeroesFactorial 500 = 124 :=
by
  sorry

end NUMINAMATH_GPT_trailing_zeroes_500_fact_l1265_126504


namespace NUMINAMATH_GPT_minimum_moves_l1265_126506

theorem minimum_moves (n : ℕ) : 
  n > 0 → ∃ k l : ℕ, k + 2 * l ≥ ⌊ (n^2 : ℝ) / 2 ⌋₊ ∧ k + l ≥ ⌊ (n^2 : ℝ) / 3 ⌋₊ :=
by 
  intro hn
  sorry

end NUMINAMATH_GPT_minimum_moves_l1265_126506


namespace NUMINAMATH_GPT_avg_annual_growth_rate_optimal_selling_price_l1265_126591

-- Define the conditions and question for the first problem: average annual growth rate.
theorem avg_annual_growth_rate (initial final : ℝ) (years : ℕ) (growth_rate : ℝ) :
  initial = 200 ∧ final = 288 ∧ years = 2 ∧ (final = initial * (1 + growth_rate)^years) →
  growth_rate = 0.2 :=
by
  -- Proof will come here
  sorry

-- Define the conditions and question for the second problem: setting the selling price.
theorem optimal_selling_price (cost initial_volume : ℕ) (initial_price : ℝ) 
(additional_sales_per_dollar : ℕ) (desired_profit : ℝ) (optimal_price : ℝ) :
  cost = 50 ∧ initial_volume = 50 ∧ initial_price = 100 ∧ additional_sales_per_dollar = 5 ∧
  desired_profit = 4000 ∧ 
  (∃ p : ℝ, (p - cost) * (initial_volume + additional_sales_per_dollar * (initial_price - p)) = desired_profit ∧ p = optimal_price) →
  optimal_price = 70 :=
by
  -- Proof will come here
  sorry

end NUMINAMATH_GPT_avg_annual_growth_rate_optimal_selling_price_l1265_126591


namespace NUMINAMATH_GPT_simplest_form_expression_l1265_126507

theorem simplest_form_expression (x y a : ℤ) : 
  (∃ (E : ℚ → Prop), (E (1/3) ∨ E (1/(x-2)) ∨ E ((x^2 * y) / (2*x)) ∨ E (2*a / 8)) → (E (1/(x-2)) ↔ E (1/(x-2)))) :=
by 
  sorry

end NUMINAMATH_GPT_simplest_form_expression_l1265_126507


namespace NUMINAMATH_GPT_mismatching_socks_count_l1265_126521

-- Define the conditions given in the problem
def total_socks : ℕ := 65
def pairs_matching_ankle_socks : ℕ := 13
def pairs_matching_crew_socks : ℕ := 10

-- Define the calculated counts as per the conditions
def matching_ankle_socks : ℕ := pairs_matching_ankle_socks * 2
def matching_crew_socks : ℕ := pairs_matching_crew_socks * 2
def total_matching_socks : ℕ := matching_ankle_socks + matching_crew_socks

-- The statement to prove
theorem mismatching_socks_count : total_socks - total_matching_socks = 19 := by
  sorry

end NUMINAMATH_GPT_mismatching_socks_count_l1265_126521


namespace NUMINAMATH_GPT_determine_x_l1265_126576

theorem determine_x (x : ℝ) (hx : 0 < x) (h : (⌊x⌋ : ℝ) * x = 120) : x = 120 / 11 := 
sorry

end NUMINAMATH_GPT_determine_x_l1265_126576


namespace NUMINAMATH_GPT_price_per_can_of_spam_l1265_126551

-- Definitions of conditions
variable (S : ℝ) -- The price per can of Spam
def cost_peanut_butter := 3 * 5 -- 3 jars of peanut butter at $5 each
def cost_bread := 4 * 2 -- 4 loaves of bread at $2 each
def total_cost := 59 -- Total amount paid

-- Proof problem to verify the price per can of Spam
theorem price_per_can_of_spam :
  12 * S + cost_peanut_butter + cost_bread = total_cost → S = 3 :=
by
  sorry

end NUMINAMATH_GPT_price_per_can_of_spam_l1265_126551


namespace NUMINAMATH_GPT_max_gcd_2015xy_l1265_126553

theorem max_gcd_2015xy (x y : ℤ) (coprime : Int.gcd x y = 1) :
    ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
sorry

end NUMINAMATH_GPT_max_gcd_2015xy_l1265_126553


namespace NUMINAMATH_GPT_find_tuesday_temp_l1265_126537

variable (temps : List ℝ) (avg : ℝ) (len : ℕ) 

theorem find_tuesday_temp (h1 : temps = [99.1, 98.2, 99.3, 99.8, 99, 98.9, tuesday_temp])
                         (h2 : avg = 99)
                         (h3 : len = 7)
                         (h4 : (temps.sum / len) = avg) :
                         tuesday_temp = 98.7 := 
sorry

end NUMINAMATH_GPT_find_tuesday_temp_l1265_126537


namespace NUMINAMATH_GPT_one_cow_empties_pond_in_75_days_l1265_126536

-- Define the necessary variables and their types
variable (c a b : ℝ) -- c represents daily water inflow from the spring
                      -- a represents the total volume of the pond
                      -- b represents the daily consumption per cow

-- Define the conditions
def condition1 : Prop := a + 3 * c = 3 * 17 * b
def condition2 : Prop := a + 30 * c = 30 * 2 * b

-- Target statement we want to prove
theorem one_cow_empties_pond_in_75_days (h1 : condition1 c a b) (h2 : condition2 c a b) :
  ∃ t : ℝ, t = 75 := 
sorry -- Proof to be provided


end NUMINAMATH_GPT_one_cow_empties_pond_in_75_days_l1265_126536


namespace NUMINAMATH_GPT_cost_of_one_dozen_pens_is_780_l1265_126599

-- Defining the cost of pens and pencils
def cost_of_pens (n : ℕ) := n * 65

def cost_of_pencils (m : ℕ) := m * 13

-- Given conditions
def total_cost (x y : ℕ) := cost_of_pens x + cost_of_pencils y

theorem cost_of_one_dozen_pens_is_780
  (h1 : total_cost 3 5 = 260)
  (h2 : 65 = 5 * 13)
  (h3 : 65 = 65) :
  12 * 65 = 780 := by
    sorry

end NUMINAMATH_GPT_cost_of_one_dozen_pens_is_780_l1265_126599


namespace NUMINAMATH_GPT_min_tan_expression_l1265_126522

open Real

theorem min_tan_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
(h_eq : sin α * cos β - 2 * cos α * sin β = 0) :
  ∃ x, x = tan (2 * π + α) + tan (π / 2 - β) ∧ x = 2 * sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_tan_expression_l1265_126522


namespace NUMINAMATH_GPT_erica_has_correct_amount_l1265_126564

-- Definitions for conditions
def total_money : ℕ := 91
def sam_money : ℕ := 38

-- Definition for the question regarding Erica's money
def erica_money := total_money - sam_money

-- The theorem stating the proof problem
theorem erica_has_correct_amount : erica_money = 53 := sorry

end NUMINAMATH_GPT_erica_has_correct_amount_l1265_126564


namespace NUMINAMATH_GPT_remainder_sum_of_integers_division_l1265_126566

theorem remainder_sum_of_integers_division (n S : ℕ) (hn_cond : n > 0) (hn_sq : n^2 + 12 * n - 3007 ≥ 0) (hn_square : ∃ m : ℕ, n^2 + 12 * n - 3007 = m^2):
  S = n → S % 1000 = 516 := 
sorry

end NUMINAMATH_GPT_remainder_sum_of_integers_division_l1265_126566


namespace NUMINAMATH_GPT_teresa_total_marks_l1265_126570

theorem teresa_total_marks :
  let science_marks := 70
  let music_marks := 80
  let social_studies_marks := 85
  let physics_marks := 1 / 2 * music_marks
  science_marks + music_marks + social_studies_marks + physics_marks = 275 :=
by
  sorry

end NUMINAMATH_GPT_teresa_total_marks_l1265_126570


namespace NUMINAMATH_GPT_lines_perpendicular_l1265_126569

-- Define the conditions: lines not parallel to the coordinate planes 
-- (which translates to k_1 and k_2 not being infinite, but we can code it directly as a statement on the product being -1)
variable {k1 k2 l1 l2 : ℝ} 

-- Define the theorem statement 
theorem lines_perpendicular (hk : k1 * k2 = -1) : 
  ∀ (x : ℝ), (k1 ≠ 0) ∧ (k2 ≠ 0) → 
  (∀ (y1 y2 : ℝ), y1 = k1 * x + l1 → y2 = k2 * x + l2 → 
  (k1 * k2 = -1)) :=
sorry

end NUMINAMATH_GPT_lines_perpendicular_l1265_126569


namespace NUMINAMATH_GPT_jack_jogging_speed_needed_l1265_126544

noncomputable def jack_normal_speed : ℝ :=
  let normal_melt_time : ℝ := 10
  let faster_melt_factor : ℝ := 0.75
  let adjusted_melt_time : ℝ := normal_melt_time * faster_melt_factor
  let adjusted_melt_time_hours : ℝ := adjusted_melt_time / 60
  let distance_to_beach : ℝ := 2
  let required_speed : ℝ := distance_to_beach / adjusted_melt_time_hours
  let slope_reduction_factor : ℝ := 0.8
  required_speed / slope_reduction_factor

theorem jack_jogging_speed_needed
  (normal_melt_time : ℝ := 10) 
  (faster_melt_factor : ℝ := 0.75) 
  (distance_to_beach : ℝ := 2) 
  (slope_reduction_factor : ℝ := 0.8) :
  jack_normal_speed = 20 := 
by
  sorry

end NUMINAMATH_GPT_jack_jogging_speed_needed_l1265_126544


namespace NUMINAMATH_GPT_symmetric_line_eq_l1265_126596

-- Defining a structure for a line using its standard equation form "ax + by + c = 0"
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Definition: A line is symmetric with respect to y-axis if it can be obtained
-- by replacing x with -x in its equation form.

def isSymmetricToYAxis (l₁ l₂ : Line) : Prop :=
  l₂.a = -l₁.a ∧ l₂.b = l₁.b ∧ l₂.c = l₁.c

-- The given condition: line1 is 4x - 3y + 5 = 0
def line1 : Line := { a := 4, b := -3, c := 5 }

-- The expected line l symmetric to y-axis should satisfy our properties
def expected_line_l : Line := { a := 4, b := 3, c := -5 }

-- The theorem we need to prove
theorem symmetric_line_eq : ∃ l : Line,
  isSymmetricToYAxis line1 l ∧ l = { a := 4, b := 3, c := -5 } :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_eq_l1265_126596


namespace NUMINAMATH_GPT_geom_seq_m_equals_11_l1265_126517

theorem geom_seq_m_equals_11 
  (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = 1) 
  (h2 : |q| ≠ 1) 
  (h3 : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h4 : a m = (a 1) * (a 2) * (a 3) * (a 4) * (a 5)) : 
  m = 11 :=
sorry

end NUMINAMATH_GPT_geom_seq_m_equals_11_l1265_126517


namespace NUMINAMATH_GPT_problem_statement_l1265_126567
-- Broader import to bring in necessary library components.

-- Definition of the equation that needs to be satisfied by the points.
def satisfies_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Definitions of the two lines that form the solution set.
def line1 (x y : ℝ) : Prop := y = -x - 2
def line2 (x y : ℝ) : Prop := y = -2 * x + 1

-- Prove that the set of points that satisfy the given equation is the union of the two lines.
theorem problem_statement (x y : ℝ) : satisfies_equation x y ↔ line1 x y ∨ line2 x y :=
sorry

end NUMINAMATH_GPT_problem_statement_l1265_126567


namespace NUMINAMATH_GPT_trigonometric_identity_l1265_126529

theorem trigonometric_identity (θ : ℝ) (h : Real.sin θ + Real.cos θ = Real.sqrt 2) :
  Real.tan θ + 1 / Real.tan θ = 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1265_126529


namespace NUMINAMATH_GPT_chocolates_brought_by_friend_l1265_126573

-- Definitions corresponding to the conditions in a)
def total_chocolates := 50
def chocolates_not_in_box := 5
def number_of_boxes := 3
def additional_boxes := 2

-- Theorem statement: we need to prove the number of chocolates her friend brought
theorem chocolates_brought_by_friend (C : ℕ) : 
  (C + total_chocolates = total_chocolates + (chocolates_not_in_box + number_of_boxes * (total_chocolates - chocolates_not_in_box) / number_of_boxes + additional_boxes * (total_chocolates - chocolates_not_in_box) / number_of_boxes) - total_chocolates) 
  → C = 30 := 
sorry

end NUMINAMATH_GPT_chocolates_brought_by_friend_l1265_126573


namespace NUMINAMATH_GPT_every_algorithm_must_have_sequential_structure_l1265_126514

def is_sequential_structure (alg : Type) : Prop := sorry -- This defines what a sequential structure is

def must_have_sequential_structure (alg : Type) : Prop :=
∀ alg, is_sequential_structure alg

theorem every_algorithm_must_have_sequential_structure :
  must_have_sequential_structure nat := sorry

end NUMINAMATH_GPT_every_algorithm_must_have_sequential_structure_l1265_126514


namespace NUMINAMATH_GPT_problem1_problem2_l1265_126545

-- Problem 1 equivalent proof problem
theorem problem1 : 
  (Real.sqrt 3 * Real.sqrt 6 - (Real.sqrt (1 / 2) - Real.sqrt 8)) = (9 * Real.sqrt 2 / 2) :=
by
  sorry

-- Problem 2 equivalent proof problem
theorem problem2 (x : Real) (hx : x = Real.sqrt 5) : 
  ((1 + 1 / x) / ((x^2 + x) / x)) = (Real.sqrt 5 / 5) :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1265_126545


namespace NUMINAMATH_GPT_proof_GP_product_l1265_126588

namespace GPProof

variables {a r : ℝ} {n : ℕ} (S S' P : ℝ)

def isGeometricProgression (a r : ℝ) (n : ℕ) :=
  ∀ i, 0 ≤ i ∧ i < n → ∃ k, ∃ b, b = (-1)^k * a * r^k ∧ k = i 

noncomputable def product (a r : ℝ) (n : ℕ) : ℝ :=
  a^n * r^(n*(n-1)/2) * (-1)^(n*(n-1)/2)

noncomputable def sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - (-r)^n) / (1 - (-r))

noncomputable def reciprocalSum (a r : ℝ) (n : ℕ) : ℝ :=
  (1 / a) * (1 - (-1/r)^n) / (1 + 1/r)

theorem proof_GP_product (hyp1 : isGeometricProgression a (-r) n) (hyp2 : S = sum a (-r) n) (hyp3 : S' = reciprocalSum a (-r) n) (hyp4 : P = product a (-r) n) :
  P = (S / S')^(n/2) :=
by
  sorry

end GPProof

end NUMINAMATH_GPT_proof_GP_product_l1265_126588


namespace NUMINAMATH_GPT_count_two_digit_powers_of_three_l1265_126523

theorem count_two_digit_powers_of_three : 
  ∃ (n1 n2 : ℕ), 10 ≤ 3^n1 ∧ 3^n1 < 100 ∧ 10 ≤ 3^n2 ∧ 3^n2 < 100 ∧ n1 ≠ n2 ∧ ∀ n : ℕ, (10 ≤ 3^n ∧ 3^n < 100) → (n = n1 ∨ n = n2) ∧ n1 = 3 ∧ n2 = 4 := by
  sorry

end NUMINAMATH_GPT_count_two_digit_powers_of_three_l1265_126523


namespace NUMINAMATH_GPT_number_of_new_galleries_l1265_126580

-- Definitions based on conditions
def number_of_pictures_first_gallery := 9
def number_of_pictures_per_new_gallery := 2
def pencils_per_picture := 4
def pencils_per_exhibition_signature := 2
def total_pencils_used := 88

-- Theorem statement according to the correct answer
theorem number_of_new_galleries 
  (number_of_pictures_first_gallery : ℕ)
  (number_of_pictures_per_new_gallery : ℕ)
  (pencils_per_picture : ℕ)
  (pencils_per_exhibition_signature : ℕ)
  (total_pencils_used : ℕ)
  (drawing_pencils_first_gallery := number_of_pictures_first_gallery * pencils_per_picture)
  (signing_pencils_first_gallery := pencils_per_exhibition_signature)
  (total_pencils_first_gallery := drawing_pencils_first_gallery + signing_pencils_first_gallery)
  (pencils_for_new_galleries := total_pencils_used - total_pencils_first_gallery)
  (pencils_per_new_gallery := (number_of_pictures_per_new_gallery * pencils_per_picture) + pencils_per_exhibition_signature) :
  pencils_per_new_gallery > 0 → pencils_for_new_galleries / pencils_per_new_gallery = 5 :=
sorry

end NUMINAMATH_GPT_number_of_new_galleries_l1265_126580


namespace NUMINAMATH_GPT_find_n_l1265_126530

theorem find_n {
    n : ℤ
   } (h1 : 0 ≤ n) (h2 : n < 103) (h3 : 99 * n ≡ 72 [ZMOD 103]) :
    n = 52 :=
sorry

end NUMINAMATH_GPT_find_n_l1265_126530


namespace NUMINAMATH_GPT_arithmetic_sequence_product_l1265_126574

theorem arithmetic_sequence_product {b : ℕ → ℤ} (d : ℤ) (h1 : ∀ n, b (n + 1) = b n + d)
    (h2 : b 5 * b 6 = 21) : b 4 * b 7 = -11 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_product_l1265_126574


namespace NUMINAMATH_GPT_monotonic_increasing_iff_l1265_126554

noncomputable def f (x b : ℝ) : ℝ := (x - b) * Real.log x + x^2

theorem monotonic_increasing_iff (b : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 1 → 0 ≤ (Real.log x - b/x + 1 + 2*x)) ↔ b ∈ Set.Iic (3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_increasing_iff_l1265_126554


namespace NUMINAMATH_GPT_absolute_value_condition_l1265_126561

theorem absolute_value_condition (a : ℝ) (h : |a| = -a) : a = 0 ∨ a < 0 :=
by
  sorry

end NUMINAMATH_GPT_absolute_value_condition_l1265_126561


namespace NUMINAMATH_GPT_probability_60_or_more_points_l1265_126549

theorem probability_60_or_more_points :
  let five_choose k := Nat.choose 5 k
  let prob_correct (k : Nat) := (five_choose k) * (1 / 2)^5
  let prob_at_least_3_correct := prob_correct 3 + prob_correct 4 + prob_correct 5
  prob_at_least_3_correct = 1 / 2 := 
sorry

end NUMINAMATH_GPT_probability_60_or_more_points_l1265_126549


namespace NUMINAMATH_GPT_example_function_not_power_function_l1265_126535

-- Definition of a power function
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the function y = 2x^(1/2)
def example_function (x : ℝ) : ℝ :=
  2 * x ^ (1 / 2)

-- The statement we want to prove
theorem example_function_not_power_function : ¬ is_power_function example_function := by
  sorry

end NUMINAMATH_GPT_example_function_not_power_function_l1265_126535


namespace NUMINAMATH_GPT_problem_statement_l1265_126555

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for f

-- Theorem stating the axis of symmetry and increasing interval for the transformed function
theorem problem_statement (hf_even : ∀ x, f x = f (-x))
  (hf_increasing : ∀ x₁ x₂, 3 < x₁ → x₁ < x₂ → x₂ < 5 → f x₁ < f x₂) :
  -- For y = f(x - 1), the following holds:
  (∀ x, (f (x - 1)) = f (-(x - 1))) ∧
  (∀ x₁ x₂, 4 < x₁ → x₁ < x₂ → x₂ < 6 → f (x₁ - 1) < f (x₂ - 1)) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1265_126555
