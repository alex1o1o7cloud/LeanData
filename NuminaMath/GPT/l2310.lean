import Mathlib

namespace expression_value_l2310_231059

noncomputable def compute_expression (ω : ℂ) (h : ω^9 = 1) (h2 : ω ≠ 1) : ℂ :=
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 + ω^72 + ω^76 + ω^80

theorem expression_value (ω : ℂ) (h : ω^9 = 1) (h2 : ω ≠ 1)
    : compute_expression ω h h2 = -ω^2 :=
sorry

end expression_value_l2310_231059


namespace length_of_BD_is_six_l2310_231011

-- Definitions of the conditions
def AB : ℕ := 6
def BC : ℕ := 11
def CD : ℕ := 6
def DA : ℕ := 8
def BD : ℕ := 6 -- adding correct answer into definition

-- The statement we want to prove
theorem length_of_BD_is_six (hAB : AB = 6) (hBC : BC = 11) (hCD : CD = 6) (hDA : DA = 8) (hBD_int : BD = 6) : 
  BD = 6 :=
by
  -- Proof placeholder
  sorry

end length_of_BD_is_six_l2310_231011


namespace prism_faces_l2310_231086

theorem prism_faces (E V F n : ℕ) (h1 : E + V = 30) (h2 : F + V = E + 2) (h3 : E = 3 * n) : F = 8 :=
by
  -- Actual proof omitted
  sorry

end prism_faces_l2310_231086


namespace trigonometric_expression_in_third_quadrant_l2310_231029

theorem trigonometric_expression_in_third_quadrant (α : ℝ) 
  (h1 : Real.sin α < 0) 
  (h2 : Real.cos α < 0) 
  (h3 : Real.tan α > 0) : 
  ¬ (Real.tan α - Real.sin α < 0) :=
sorry

end trigonometric_expression_in_third_quadrant_l2310_231029


namespace faye_total_crayons_l2310_231082

-- Define the number of rows and the number of crayons per row as given conditions.
def num_rows : ℕ := 7
def crayons_per_row : ℕ := 30

-- State the theorem we need to prove.
theorem faye_total_crayons : (num_rows * crayons_per_row) = 210 :=
by
  sorry

end faye_total_crayons_l2310_231082


namespace sum_of_squares_arithmetic_geometric_l2310_231022

theorem sum_of_squares_arithmetic_geometric (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 225) : x^2 + y^2 = 1150 :=
by
  sorry

end sum_of_squares_arithmetic_geometric_l2310_231022


namespace max_k_value_l2310_231079

noncomputable def circle_equation (x y : ℝ) : Prop :=
x^2 + y^2 - 8 * x + 15 = 0

noncomputable def point_on_line (k x y : ℝ) : Prop :=
y = k * x - 2

theorem max_k_value (k : ℝ) :
  (∃ x y, circle_equation x y ∧ point_on_line k x y ∧ (x - 4)^2 + y^2 = 1) →
  k ≤ 4 / 3 :=
by
  sorry

end max_k_value_l2310_231079


namespace small_order_peanuts_l2310_231084

theorem small_order_peanuts (total_peanuts : ℕ) (large_orders : ℕ) (peanuts_per_large : ℕ) 
    (small_orders : ℕ) (peanuts_per_small : ℕ) : 
    total_peanuts = large_orders * peanuts_per_large + small_orders * peanuts_per_small → 
    total_peanuts = 800 → 
    large_orders = 3 → 
    peanuts_per_large = 200 → 
    small_orders = 4 → 
    peanuts_per_small = 50 := by
  intros h1 h2 h3 h4 h5
  sorry

end small_order_peanuts_l2310_231084


namespace total_earnings_l2310_231043

theorem total_earnings (x y : ℝ) (h1 : 20 * x * y - 18 * x * y = 120) : 
  18 * x * y + 20 * x * y + 20 * x * y = 3480 := 
by
  sorry

end total_earnings_l2310_231043


namespace find_n_l2310_231021

theorem find_n (n : ℕ) : 
  (1/5 : ℝ)^35 * (1/4 : ℝ)^n = (1 : ℝ) / (2 * 10^35) → n = 18 :=
by
  intro h
  sorry

end find_n_l2310_231021


namespace quadratic_has_real_roots_iff_l2310_231081

theorem quadratic_has_real_roots_iff (m : ℝ) :
  (∃ x : ℝ, x^2 + 4 * x + m + 5 = 0) ↔ m ≤ -1 :=
by
  -- Proof omitted
  sorry

end quadratic_has_real_roots_iff_l2310_231081


namespace min_number_of_squares_l2310_231072

theorem min_number_of_squares (length width : ℕ) (h_length : length = 10) (h_width : width = 9) : 
  ∃ n, n = 10 :=
by
  sorry

end min_number_of_squares_l2310_231072


namespace seymour_flats_of_roses_l2310_231066

-- Definitions used in conditions
def flats_of_petunias := 4
def petunias_per_flat := 8
def venus_flytraps := 2
def fertilizer_per_petunia := 8
def fertilizer_per_rose := 3
def fertilizer_per_venus_flytrap := 2
def total_fertilizer := 314

-- Compute the total fertilizer for petunias and Venus flytraps
def total_fertilizer_petunias := flats_of_petunias * petunias_per_flat * fertilizer_per_petunia
def total_fertilizer_venus_flytraps := venus_flytraps * fertilizer_per_venus_flytrap

-- Remaining fertilizer for roses
def remaining_fertilizer_for_roses := total_fertilizer - total_fertilizer_petunias - total_fertilizer_venus_flytraps

-- Define roses per flat and the fertilizer used per flat of roses
def roses_per_flat := 6
def fertilizer_per_flat_of_roses := roses_per_flat * fertilizer_per_rose

-- The number of flats of roses
def flats_of_roses := remaining_fertilizer_for_roses / fertilizer_per_flat_of_roses

-- The proof problem statement
theorem seymour_flats_of_roses : flats_of_roses = 3 := by
  sorry

end seymour_flats_of_roses_l2310_231066


namespace circumference_of_jogging_track_l2310_231064

-- Definitions for the given conditions
def speed_deepak : ℝ := 4.5
def speed_wife : ℝ := 3.75
def meet_time : ℝ := 4.32

-- The theorem stating the problem
theorem circumference_of_jogging_track : 
  (speed_deepak + speed_wife) * meet_time = 35.64 :=
by
  sorry

end circumference_of_jogging_track_l2310_231064


namespace solve_fractional_equation_l2310_231056

theorem solve_fractional_equation (x : ℝ) (h : (x + 1) / (4 * x^2 - 1) = (3 / (2 * x + 1)) - (4 / (4 * x - 2))) : x = 6 := 
by
  sorry

end solve_fractional_equation_l2310_231056


namespace inequality_solutions_l2310_231073

theorem inequality_solutions (p p' q q' : ℕ) (hp : p ≠ p') (hq : q ≠ q') (hp_pos : 0 < p) (hp'_pos : 0 < p') (hq_pos : 0 < q) (hq'_pos : 0 < q') :
  (-(q : ℚ) / p > -(q' : ℚ) / p') ↔ (q * p' < p * q') :=
by
  sorry

end inequality_solutions_l2310_231073


namespace Zain_coins_total_l2310_231070

theorem Zain_coins_total :
  ∀ (quarters dimes nickels : ℕ),
  quarters = 6 →
  dimes = 7 →
  nickels = 5 →
  Zain_coins = quarters + 10 + (dimes + 10) + (nickels + 10) →
  Zain_coins = 48 :=
by intros quarters dimes nickels hq hd hn Zain_coins
   sorry

end Zain_coins_total_l2310_231070


namespace prove_range_of_xyz_l2310_231047

variable (x y z a : ℝ)

theorem prove_range_of_xyz 
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = a^2 / 2)
  (ha : 0 < a) :
  (0 ≤ x ∧ x ≤ 2 * a / 3) ∧ (0 ≤ y ∧ y ≤ 2 * a / 3) ∧ (0 ≤ z ∧ z ≤ 2 * a / 3) :=
sorry

end prove_range_of_xyz_l2310_231047


namespace rate_of_interest_l2310_231006

noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

theorem rate_of_interest :
  ∃ R : ℝ, simple_interest 8925 R 5 = 4016.25 ∧ R = 9 := 
by
  use 9
  simp [simple_interest]
  norm_num
  sorry

end rate_of_interest_l2310_231006


namespace calculate_value_l2310_231032

theorem calculate_value : (535^2 - 465^2) / 70 = 1000 := by
  sorry

end calculate_value_l2310_231032


namespace inverse_proportion_incorrect_D_l2310_231071

theorem inverse_proportion_incorrect_D :
  ∀ (x y x1 y1 x2 y2 : ℝ), (y = -3 / x) ∧ (y1 = -3 / x1) ∧ (y2 = -3 / x2) ∧ (x1 < x2) → ¬(y1 < y2) :=
by
  sorry

end inverse_proportion_incorrect_D_l2310_231071


namespace sqrt_of_square_neg_three_l2310_231048

theorem sqrt_of_square_neg_three : Real.sqrt ((-3 : ℝ)^2) = 3 := by
  sorry

end sqrt_of_square_neg_three_l2310_231048


namespace expand_expression_l2310_231077

theorem expand_expression : ∀ (x : ℝ), 2 * (x + 3) * (x^2 - 2*x + 7) = 2*x^3 + 2*x^2 + 2*x + 42 := 
by
  intro x
  sorry

end expand_expression_l2310_231077


namespace black_friday_sales_l2310_231008

variable (n : ℕ) (initial_sales increment : ℕ)

def yearly_sales (sales: ℕ) (inc: ℕ) (years: ℕ) : ℕ :=
  sales + years * inc

theorem black_friday_sales (h1 : initial_sales = 327) (h2 : increment = 50) :
  yearly_sales initial_sales increment 3 = 477 := by
  sorry

end black_friday_sales_l2310_231008


namespace four_digit_non_convertible_to_1992_multiple_l2310_231016

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_multiple_of_1992 (n : ℕ) : Prop :=
  n % 1992 = 0

def reachable (n m : ℕ) (k : ℕ) : Prop :=
  ∃ x y z : ℕ, 
    x ≠ m ∧ y ≠ m ∧ z ≠ m ∧
    (n + x * 10^(k-1) + y * 10^(k-2) + z * 10^(k-3)) % 1992 = 0 ∧
    n + x * 10^(k-1) + y * 10^(k-2) + z * 10^(k-3) < 10000

theorem four_digit_non_convertible_to_1992_multiple :
  ∃ n : ℕ, is_four_digit n ∧ (∀ m : ℕ, is_four_digit m ∧ is_multiple_of_1992 m → ¬ reachable n m 3) :=
sorry

end four_digit_non_convertible_to_1992_multiple_l2310_231016


namespace inequality_transformation_range_of_a_l2310_231089

-- Define the given function f(x) = |x + 2|
def f (x : ℝ) : ℝ := abs (x + 2)

-- State the inequality transformation problem
theorem inequality_transformation (x : ℝ) :  (2 * abs (x + 2) < 4 - abs (x - 1)) ↔ (-7 / 3 < x ∧ x < -1) :=
by sorry

-- State the implication problem involving m, n, and a
theorem range_of_a (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (hmn : m + n = 1) (a : ℝ) :
  (∀ x : ℝ, abs (x - a) - f x ≤ 1 / m + 1 / n) → (-6 ≤ a ∧ a ≤ 2) :=
by sorry

end inequality_transformation_range_of_a_l2310_231089


namespace Wolfgang_marble_count_l2310_231023

theorem Wolfgang_marble_count
  (W L M : ℝ)
  (hL : L = 5/4 * W)
  (hM : M = 2/3 * (W + L))
  (hTotal : W + L + M = 60) :
  W = 16 :=
by {
  sorry
}

end Wolfgang_marble_count_l2310_231023


namespace largest_interior_angle_l2310_231020

theorem largest_interior_angle (x : ℝ) (h₀ : 50 + 55 + x = 180) : 
  max 50 (max 55 x) = 75 := by
  sorry

end largest_interior_angle_l2310_231020


namespace exists_infinitely_many_solutions_l2310_231068

theorem exists_infinitely_many_solutions :
  ∃ m : ℕ, m > 0 ∧ (∀ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0) →
    (1/a + 1/b + 1/c + 1/(a*b*c) = m / (a + b + c))) :=
sorry

end exists_infinitely_many_solutions_l2310_231068


namespace range_of_a_l2310_231099

-- Defining the problem conditions
def f (x : ℝ) : ℝ := sorry -- The function f : ℝ → ℝ is defined elsewhere such that its range is [0, 4]
def g (a x : ℝ) : ℝ := a * x - 1

-- Theorem to prove the range of 'a'
theorem range_of_a (a : ℝ) : (a ≥ 1/2) ∨ (a ≤ -1/2) :=
sorry

end range_of_a_l2310_231099


namespace log_exp_sum_l2310_231012

theorem log_exp_sum :
  2^(Real.log 3 / Real.log 2) + Real.log (Real.sqrt 5) / Real.log 10 + Real.log (Real.sqrt 20) / Real.log 10 = 4 :=
by
  sorry

end log_exp_sum_l2310_231012


namespace number_of_dissimilar_terms_l2310_231018

theorem number_of_dissimilar_terms :
  let n := 7;
  let k := 4;
  let number_of_terms := Nat.choose (n + k - 1) (k - 1);
  number_of_terms = 120 :=
by
  sorry

end number_of_dissimilar_terms_l2310_231018


namespace six_digit_number_all_equal_l2310_231005

open Nat

theorem six_digit_number_all_equal (n : ℕ) (h : n = 21) : 12 * n^2 + 12 * n + 11 = 5555 :=
by
  rw [h]  -- Substitute n = 21
  sorry  -- Omit the actual proof steps

end six_digit_number_all_equal_l2310_231005


namespace race_total_distance_l2310_231028

theorem race_total_distance (D : ℝ) 
  (A_time : D / 20 = D / 25 + 1) 
  (beat_distance : D / 20 * 25 = D + 20) : 
  D = 80 :=
sorry

end race_total_distance_l2310_231028


namespace distinct_points_4_l2310_231038

theorem distinct_points_4 (x y : ℝ) :
  (x + y = 7 ∨ 3 * x - 2 * y = -6) ∧ (x - y = -2 ∨ 4 * x + y = 10) →
  (x, y) =
    (5 / 2, 9 / 2) ∨ 
    (x, y) = (1, 6) ∨
    (x, y) = (-2, 0) ∨ 
    (x, y) = (14 / 11, 74 / 11) :=
sorry

end distinct_points_4_l2310_231038


namespace quotient_larger_than_dividend_l2310_231095

-- Define the problem conditions
variables {a b : ℝ}

-- State the theorem corresponding to the problem
theorem quotient_larger_than_dividend (h : b ≠ 0) : ¬ (∀ a : ℝ, ∀ b : ℝ, (a / b > a) ) :=
by
  sorry

end quotient_larger_than_dividend_l2310_231095


namespace find_d_share_l2310_231035

def money_distribution (a b c d : ℕ) (x : ℕ) := 
  a = 5 * x ∧ 
  b = 2 * x ∧ 
  c = 4 * x ∧ 
  d = 3 * x ∧ 
  (c = d + 500)

theorem find_d_share (a b c d x : ℕ) (h : money_distribution a b c d x) : d = 1500 :=
by
  --proof would go here
  sorry

end find_d_share_l2310_231035


namespace wicket_keeper_older_than_captain_l2310_231015

variables (captain_age : ℕ) (team_avg_age : ℕ) (num_players : ℕ) (remaining_avg_age : ℕ)

def x_older_than_captain (captain_age team_avg_age num_players remaining_avg_age : ℕ) : ℕ :=
  team_avg_age * num_players - remaining_avg_age * (num_players - 2) - 2 * captain_age

theorem wicket_keeper_older_than_captain 
  (captain_age : ℕ) (team_avg_age : ℕ) (num_players : ℕ) (remaining_avg_age : ℕ) 
  (h1 : captain_age = 25) (h2 : team_avg_age = 23) (h3 : num_players = 11) (h4 : remaining_avg_age = 22) :
  x_older_than_captain captain_age team_avg_age num_players remaining_avg_age = 5 :=
by sorry

end wicket_keeper_older_than_captain_l2310_231015


namespace largest_n_for_negative_sum_l2310_231051

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ} -- common difference of the arithmetic sequence

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n + 1) * (a 0 + a n) / 2

theorem largest_n_for_negative_sum
  (h_arith_seq : is_arithmetic_sequence a d)
  (h_first_term : a 0 < 0)
  (h_sum_2015_2016 : a 2014 + a 2015 > 0)
  (h_product_2015_2016 : a 2014 * a 2015 < 0) :
  (∀ n, sum_of_first_n_terms a n < 0 → n ≤ 4029) ∧ (sum_of_first_n_terms a 4029 < 0) :=
sorry

end largest_n_for_negative_sum_l2310_231051


namespace quadratic_equation_with_given_root_l2310_231098

theorem quadratic_equation_with_given_root : 
  ∃ p q : ℤ, (∀ x : ℝ, x^2 + (p : ℝ) * x + (q : ℝ) = 0 ↔ x = 2 - Real.sqrt 7 ∨ x = 2 + Real.sqrt 7) 
  ∧ (p = -4) ∧ (q = -3) :=
by
  sorry

end quadratic_equation_with_given_root_l2310_231098


namespace g_at_zero_l2310_231026

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 4)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4)

theorem g_at_zero : g 0 = -Real.sqrt 2 :=
by
  -- proof to be completed
  sorry

end g_at_zero_l2310_231026


namespace tan_product_identity_l2310_231075

theorem tan_product_identity (A B : ℝ) (hA : A = 20) (hB : B = 25) (hSum : A + B = 45) :
    (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2 := 
  by
  sorry

end tan_product_identity_l2310_231075


namespace smallest_value_proof_l2310_231044

noncomputable def smallest_value (x : ℝ) (h : 0 < x ∧ x < 1) : Prop :=
  x^3 < x ∧ x^3 < 3*x ∧ x^3 < x^(1/3) ∧ x^3 < 1/x^2

theorem smallest_value_proof (x : ℝ) (h : 0 < x ∧ x < 1) : smallest_value x h :=
  sorry

end smallest_value_proof_l2310_231044


namespace students_end_year_10_l2310_231007

def students_at_end_of_year (initial_students : ℕ) (left_students : ℕ) (increase_percent : ℕ) : ℕ :=
  let remaining_students := initial_students - left_students
  let increased_students := (remaining_students * increase_percent) / 100
  remaining_students + increased_students

theorem students_end_year_10 : 
  students_at_end_of_year 10 4 70 = 10 := by 
  sorry

end students_end_year_10_l2310_231007


namespace compute_a2004_l2310_231063

def recurrence_sequence (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if n = 2 then 0
  else sorry -- We'll define recurrence operations in the proofs

theorem compute_a2004 : recurrence_sequence 2004 = -2^1002 := 
sorry -- Proof omitted

end compute_a2004_l2310_231063


namespace total_bottles_needed_l2310_231049

-- Definitions from conditions
def large_bottle_capacity : ℕ := 450
def small_bottle_capacity : ℕ := 45
def extra_large_bottle_capacity : ℕ := 900

-- Theorem statement
theorem total_bottles_needed :
  ∃ (num_large_bottles num_small_bottles : ℕ), 
    num_large_bottles * large_bottle_capacity + num_small_bottles * small_bottle_capacity = extra_large_bottle_capacity ∧ 
    num_large_bottles + num_small_bottles = 2 :=
by
  sorry

end total_bottles_needed_l2310_231049


namespace find_first_m_gt_1959_l2310_231052

theorem find_first_m_gt_1959 :
  ∃ m n : ℕ, 8 * m - 7 = n^2 ∧ m > 1959 ∧ m = 2017 :=
by
  sorry

end find_first_m_gt_1959_l2310_231052


namespace total_oil_leak_l2310_231053

-- Definitions for the given conditions
def before_repair_leak : ℕ := 6522
def during_repair_leak : ℕ := 5165
def total_leak : ℕ := 11687

-- The proof statement (without proof, only the statement)
theorem total_oil_leak :
  before_repair_leak + during_repair_leak = total_leak :=
sorry

end total_oil_leak_l2310_231053


namespace nuts_mixture_weight_l2310_231024

variable (m n : ℕ)
variable (weight_almonds per_part total_weight : ℝ)

theorem nuts_mixture_weight (h1 : m = 5) (h2 : n = 2) (h3 : weight_almonds = 250) 
  (h4 : per_part = weight_almonds / m) (h5 : total_weight = per_part * (m + n)) : 
  total_weight = 350 := by
  sorry

end nuts_mixture_weight_l2310_231024


namespace symmetry_condition_l2310_231010

theorem symmetry_condition (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∀ x y : ℝ, y = x ↔ x = (ax + b) / (cx - d)) ∧ 
  (∀ x y : ℝ, y = -x ↔ x = (-ax + b) / (-cx - d)) → 
  d + b = 0 :=
by sorry

end symmetry_condition_l2310_231010


namespace not_equal_77_l2310_231090

theorem not_equal_77 (x y : ℤ) : x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
  sorry

end not_equal_77_l2310_231090


namespace gillian_spent_multiple_of_sandi_l2310_231074

theorem gillian_spent_multiple_of_sandi
  (sandi_had : ℕ := 600)
  (gillian_spent : ℕ := 1050)
  (sandi_spent : ℕ := sandi_had / 2)
  (diff : ℕ := gillian_spent - sandi_spent)
  (extra : ℕ := 150)
  (multiple_of_sandi : ℕ := (diff - extra) / sandi_spent) : 
  multiple_of_sandi = 1 := 
  by sorry

end gillian_spent_multiple_of_sandi_l2310_231074


namespace toothpicks_grid_total_l2310_231093

theorem toothpicks_grid_total (L W : ℕ) (hL : L = 60) (hW : W = 32) : 
  (L + 1) * W + (W + 1) * L = 3932 := 
by 
  sorry

end toothpicks_grid_total_l2310_231093


namespace smallest_n_l2310_231004

theorem smallest_n (n : ℕ) :
  (1 / 4 : ℚ) + (n / 8 : ℚ) > 1 ↔ n ≥ 7 := by
  sorry

end smallest_n_l2310_231004


namespace find_xy_plus_yz_plus_xz_l2310_231033

theorem find_xy_plus_yz_plus_xz
  (x y z : ℝ)
  (h₁ : x > 0)
  (h₂ : y > 0)
  (h₃ : z > 0)
  (eq1 : x^2 + x * y + y^2 = 75)
  (eq2 : y^2 + y * z + z^2 = 64)
  (eq3 : z^2 + z * x + x^2 = 139) :
  x * y + y * z + z * x = 80 :=
by
  sorry

end find_xy_plus_yz_plus_xz_l2310_231033


namespace combined_alloy_tin_amount_l2310_231080

theorem combined_alloy_tin_amount
  (weight_A weight_B weight_C : ℝ)
  (ratio_lead_tin_A : ℝ)
  (ratio_tin_copper_B : ℝ)
  (ratio_copper_tin_C : ℝ)
  (amount_tin : ℝ) :
  weight_A = 150 → weight_B = 200 → weight_C = 250 →
  ratio_lead_tin_A = 5/3 → ratio_tin_copper_B = 2/3 → ratio_copper_tin_C = 4 →
  amount_tin = ((3/8) * weight_A) + ((2/5) * weight_B) + ((1/5) * weight_C) →
  amount_tin = 186.25 :=
by sorry

end combined_alloy_tin_amount_l2310_231080


namespace number_of_purple_balls_l2310_231091

theorem number_of_purple_balls (k : ℕ) (h : k > 0) (E : (24 - k) / (8 + k) = 1) : k = 8 :=
by {
  sorry
}

end number_of_purple_balls_l2310_231091


namespace cellphone_loading_time_approximately_l2310_231041

noncomputable def cellphone_loading_time_minutes : ℝ :=
  let T := 533.78 -- Solution for T from solving the given equation
  T / 60

theorem cellphone_loading_time_approximately :
  abs (cellphone_loading_time_minutes - 8.90) < 0.01 :=
by 
  -- The proof goes here, but we are just required to state it
  sorry

end cellphone_loading_time_approximately_l2310_231041


namespace evaluate_expression_l2310_231078

variable (x y : ℝ)
variable (h₀ : x ≠ 0)
variable (h₁ : y ≠ 0)
variable (h₂ : 5 * x ≠ 3 * y)

theorem evaluate_expression : 
  (5 * x - 3 * y)⁻¹ * ((5 * x)⁻¹ - (3 * y)⁻¹) = -1 / (15 * x * y) :=
sorry

end evaluate_expression_l2310_231078


namespace smallest_positive_period_pi_interval_extrema_l2310_231034

noncomputable def f (x : ℝ) := 4 * Real.sin x * Real.cos (x + Real.pi / 3) + Real.sqrt 3

theorem smallest_positive_period_pi : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

theorem interval_extrema :
  ∃ x_max x_min : ℝ, 
  -Real.pi / 4 ≤ x_max ∧ x_max ≤ Real.pi / 6 ∧ f x_max = 2 ∧
  -Real.pi / 4 ≤ x_min ∧ x_min ≤ Real.pi / 6 ∧ f x_min = -1 ∧ 
  (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 6 → f x ≤ 2 ∧ f x ≥ -1) :=
sorry

end smallest_positive_period_pi_interval_extrema_l2310_231034


namespace math_problem_l2310_231092

noncomputable def problem_statement : Prop := (7^2 - 5^2)^4 = 331776

theorem math_problem : problem_statement := by
  sorry

end math_problem_l2310_231092


namespace quadratic_solution_l2310_231069

theorem quadratic_solution (x : ℝ) (h_eq : x^2 - 3 * x - 6 = 0) (h_neq : x ≠ 0) :
    x = (3 + Real.sqrt 33) / 2 ∨ x = (3 - Real.sqrt 33) / 2 :=
by
  sorry

end quadratic_solution_l2310_231069


namespace tangent_line_to_ellipse_l2310_231054

theorem tangent_line_to_ellipse (k : ℝ) :
  (∀ x : ℝ, (x / 2 + 2 * (k * x + 2) ^ 2) = 2) →
  k^2 = 3 / 4 :=
by
  sorry

end tangent_line_to_ellipse_l2310_231054


namespace gerald_bars_l2310_231062

theorem gerald_bars (G : ℕ) 
  (H1 : ∀ G, ∀ teacher_bars : ℕ, teacher_bars = 2 * G → total_bars = G + teacher_bars) 
  (H2 : ∀ total_bars : ℕ, total_squares = total_bars * 8 → total_squares_needed = 24 * 7) 
  (H3 : ∀ total_squares : ℕ, total_squares_needed = 24 * 7) 
  : G = 7 :=
by
  sorry

end gerald_bars_l2310_231062


namespace taxi_ride_distance_l2310_231094

variable (t : ℝ) (c₀ : ℝ) (cᵢ : ℝ)

theorem taxi_ride_distance (h_t : t = 18.6) (h_c₀ : c₀ = 3.0) (h_cᵢ : cᵢ = 0.4) : 
  ∃ d : ℝ, d = 8 := 
by 
  sorry

end taxi_ride_distance_l2310_231094


namespace exists_positive_integer_m_l2310_231061

theorem exists_positive_integer_m (m : ℕ) (h_positive : m > 0) : 
  ∃ (m : ℕ), m > 0 ∧ ∃ k : ℕ, 8 * m = k^2 := 
sorry

end exists_positive_integer_m_l2310_231061


namespace expression_evaluates_to_4_l2310_231058

theorem expression_evaluates_to_4 :
  2 * Real.cos (Real.pi / 6) + (- 1 / 2 : ℝ)⁻¹ + |Real.sqrt 3 - 2| + (2 * Real.sqrt (9 / 4))^0 + Real.sqrt 9 = 4 := 
by
  sorry

end expression_evaluates_to_4_l2310_231058


namespace train_speed_is_60_kmph_l2310_231040

noncomputable def train_length : ℝ := 110
noncomputable def time_to_pass_man : ℝ := 5.999520038396929
noncomputable def man_speed_kmph : ℝ := 6

theorem train_speed_is_60_kmph :
  let man_speed_mps := man_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / time_to_pass_man
  let train_speed_mps := relative_speed - man_speed_mps
  let train_speed_kmph := train_speed_mps * (3600 / 1000)
  train_speed_kmph = 60 :=
by
  sorry

end train_speed_is_60_kmph_l2310_231040


namespace television_price_l2310_231003

theorem television_price (SP : ℝ) (RP : ℕ) (discount : ℝ) (h1 : discount = 0.20) (h2 : SP = RP - discount * RP) (h3 : SP = 480) : RP = 600 :=
by
  sorry

end television_price_l2310_231003


namespace number_of_possible_n_ge_2_satisfying_n_cubed_plus_two_is_prime_l2310_231001

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_possible_n_ge_2_satisfying_n_cubed_plus_two_is_prime 
  (n : ℕ) (h : n ≥ 2) : (∃ (a b : ℕ), a ≠ b ∧ is_prime (a^3 + 2) ∧ is_prime (b^3 + 2)) :=
by
  sorry

end number_of_possible_n_ge_2_satisfying_n_cubed_plus_two_is_prime_l2310_231001


namespace all_girls_probability_l2310_231025

-- Definition of the problem conditions
def probability_of_girl : ℚ := 1 / 2
def events_independent (P1 P2 P3 : ℚ) : Prop := P1 * P2 = P1 ∧ P2 * P3 = P2

-- The statement to prove
theorem all_girls_probability :
  events_independent probability_of_girl probability_of_girl probability_of_girl →
  (probability_of_girl * probability_of_girl * probability_of_girl) = 1 / 8 := 
by
  intros h
  sorry

end all_girls_probability_l2310_231025


namespace lines_parallel_iff_a_eq_neg2_l2310_231002

def line₁_eq (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def line₂_eq (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y - 1 = 0

theorem lines_parallel_iff_a_eq_neg2 (a : ℝ) :
  (∀ x y : ℝ, line₁_eq a x y → line₂_eq a x y) ↔ a = -2 :=
by sorry

end lines_parallel_iff_a_eq_neg2_l2310_231002


namespace derivative_at_two_l2310_231042

theorem derivative_at_two {f : ℝ → ℝ} (f_deriv : ∀x, deriv f x = 2 * x - 4) : deriv f 2 = 0 := 
by sorry

end derivative_at_two_l2310_231042


namespace triangle_area_divided_l2310_231031

theorem triangle_area_divided {baseA heightA baseB heightB : ℝ} 
  (h1 : baseA = 1) 
  (h2 : heightA = 1)
  (h3 : baseB = 2)
  (h4 : heightB = 1)
  : (1 / 2 * baseA * heightA + 1 / 2 * baseB * heightB = 1.5) :=
by
  sorry

end triangle_area_divided_l2310_231031


namespace shaggy_seeds_l2310_231039

theorem shaggy_seeds {N : ℕ} (h1 : 50 < N) (h2 : N < 65) (h3 : N = 60) : 
  ∃ L : ℕ, L = 54 := by
  let L := 54
  sorry

end shaggy_seeds_l2310_231039


namespace solve_for_y_l2310_231050

theorem solve_for_y (y : ℝ) (h : (↑(30 * y) + (↑(30 * y) + 17) ^ (1 / 3)) ^ (1 / 3) = 17) :
  y = 816 / 5 := 
sorry

end solve_for_y_l2310_231050


namespace not_possible_select_seven_distinct_weights_no_equal_subsets_l2310_231009

theorem not_possible_select_seven_distinct_weights_no_equal_subsets :
  ∀ (s : Finset ℕ), s ⊆ Finset.range 27 → s.card = 7 → ∃ (a b : Finset ℕ), a ≠ b ∧ a ⊆ s ∧ b ⊆ s ∧ a.sum id = b.sum id :=
by
  intro s hs hcard
  sorry

end not_possible_select_seven_distinct_weights_no_equal_subsets_l2310_231009


namespace log_graph_passes_fixed_point_l2310_231057

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem log_graph_passes_fixed_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  log_a a (-1 + 2) = 0 :=
by
  sorry

end log_graph_passes_fixed_point_l2310_231057


namespace expression_divisible_by_3_l2310_231076

theorem expression_divisible_by_3 (k : ℤ) : ∃ m : ℤ, (2 * k + 3)^2 - 4 * k^2 = 3 * m :=
by
  sorry

end expression_divisible_by_3_l2310_231076


namespace consecutive_digits_sum_190_to_199_l2310_231097

-- Define the digits sum function
def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define ten consecutive numbers starting from m
def ten_consecutive_sum (m : ℕ) : ℕ :=
  (List.range 10).map (λ i => digits_sum (m + i)) |>.sum

theorem consecutive_digits_sum_190_to_199:
  ten_consecutive_sum 190 = 145 :=
by
  sorry

end consecutive_digits_sum_190_to_199_l2310_231097


namespace relationship_between_p_and_q_l2310_231087

variable {a b : ℝ}

theorem relationship_between_p_and_q 
  (h_a : a > 2) 
  (h_p : p = a + 1 / (a - 2)) 
  (h_q : q = -b^2 - 2 * b + 3) : 
  p ≥ q := 
sorry

end relationship_between_p_and_q_l2310_231087


namespace emily_51_49_calculations_l2310_231000

theorem emily_51_49_calculations :
  (51^2 = 50^2 + 101) ∧ (49^2 = 50^2 - 99) :=
by
  sorry

end emily_51_49_calculations_l2310_231000


namespace combined_exceeds_limit_l2310_231017

-- Let Zone A, Zone B, and Zone C be zones on a road.
-- Let pA be the percentage of motorists exceeding the speed limit in Zone A.
-- Let pB be the percentage of motorists exceeding the speed limit in Zone B.
-- Let pC be the percentage of motorists exceeding the speed limit in Zone C.
-- Each zone has an equal amount of motorists.

def pA : ℝ := 15
def pB : ℝ := 20
def pC : ℝ := 10

/-
Prove that the combined percentage of motorists who exceed the speed limit
across all three zones is 15%.
-/
theorem combined_exceeds_limit :
  (pA + pB + pC) / 3 = 15 := 
by sorry

end combined_exceeds_limit_l2310_231017


namespace problem1_problem2_l2310_231036

theorem problem1 : 101 * 99 = 9999 := 
by sorry

theorem problem2 : 32 * 2^2 + 14 * 2^3 + 10 * 2^4 = 400 := 
by sorry

end problem1_problem2_l2310_231036


namespace inequality_positive_l2310_231027

theorem inequality_positive (x : ℝ) : (1 / 3) * x - x > 0 ↔ (-2 / 3) * x > 0 := 
  sorry

end inequality_positive_l2310_231027


namespace left_building_percentage_l2310_231014

theorem left_building_percentage (L R : ℝ)
  (middle_building_height : ℝ := 100)
  (total_height : ℝ := 340)
  (condition1 : L + middle_building_height + R = total_height)
  (condition2 : R = L + middle_building_height - 20) :
  (L / middle_building_height) * 100 = 80 := by
  sorry

end left_building_percentage_l2310_231014


namespace find_f_six_l2310_231030

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the function definition

axiom f_property : ∀ x y : ℝ, f (x - y) = f x * f y
axiom f_nonzero : ∀ x : ℝ, f x ≠ 0
axiom f_two : f 2 = 5

theorem find_f_six : f 6 = 1 / 5 :=
sorry

end find_f_six_l2310_231030


namespace find_N_l2310_231060

variables (k N : ℤ)

theorem find_N (h : ((k * N + N) / N - N) = k - 2021) : N = 2022 :=
by
  sorry

end find_N_l2310_231060


namespace certain_number_is_17_l2310_231055

theorem certain_number_is_17 (x : ℤ) (h : 2994 / x = 177) : x = 17 :=
by
  sorry

end certain_number_is_17_l2310_231055


namespace molecular_weight_compound_l2310_231088

theorem molecular_weight_compound :
  let weight_H := 1.008
  let weight_Cr := 51.996
  let weight_O := 15.999
  let n_H := 2
  let n_Cr := 1
  let n_O := 4
  (n_H * weight_H) + (n_Cr * weight_Cr) + (n_O * weight_O) = 118.008 :=
by
  sorry

end molecular_weight_compound_l2310_231088


namespace find_f1_verify_function_l2310_231083

theorem find_f1 (f : ℝ → ℝ) (h_mono : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 > f x2)
    (h1_pos : ∀ x : ℝ, 0 < x → f x > 1 / x^2)
    (h_eq : ∀ x : ℝ, 0 < x → (f x)^2 * f (f x - 1 / x^2) = (f 1)^3) :
    f 1 = 2 := sorry

theorem verify_function (f : ℝ → ℝ) (h_def : ∀ x : ℝ, 0 < x → f x = 2 / x^2) :
    (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 > f x2) ∧ (∀ x : ℝ, 0 < x → f x > 1 / x^2) ∧
    (∀ x : ℝ, 0 < x → (f x)^2 * f (f x - 1 / x^2) = (f 1)^3) := sorry

end find_f1_verify_function_l2310_231083


namespace minimum_balls_to_draw_l2310_231085

-- Defining the sizes for the different colors of balls
def red_balls : Nat := 40
def green_balls : Nat := 25
def yellow_balls : Nat := 20
def blue_balls : Nat := 15
def purple_balls : Nat := 10
def orange_balls : Nat := 5

-- Given conditions
def max_red_balls_before_18 : Nat := 17
def max_green_balls_before_18 : Nat := 17
def max_yellow_balls_before_18 : Nat := 17
def max_blue_balls_before_18 : Nat := 15
def max_purple_balls_before_18 : Nat := 10
def max_orange_balls_before_18 : Nat := 5

-- Sum of maximum balls of each color that can be drawn without ensuring 18 of any color
def max_balls_without_18 : Nat := 
  max_red_balls_before_18 + 
  max_green_balls_before_18 + 
  max_yellow_balls_before_18 + 
  max_blue_balls_before_18 + 
  max_purple_balls_before_18 + 
  max_orange_balls_before_18

theorem minimum_balls_to_draw {n : Nat} (h : n = max_balls_without_18 + 1) :
  n = 82 := by
  sorry

end minimum_balls_to_draw_l2310_231085


namespace find_base_side_length_l2310_231067

-- Regular triangular pyramid properties and derived values
variables
  (a l h : ℝ) -- side length of the base, slant height, and height of the pyramid
  (V : ℝ) -- volume of the pyramid

-- Given conditions
def inclined_to_base_plane_at_angle (angle : ℝ) := angle = 45
def volume_of_pyramid (V : ℝ) := V = 18

-- Prove the side length of the base
theorem find_base_side_length
  (h_eq : h = a * Real.sqrt 3 / 3)
  (volume_eq : V = 1 / 3 * (a * a * Real.sqrt 3 / 4) * h)
  (volume_given : V = 18) :
  a = 6 := by
  sorry

end find_base_side_length_l2310_231067


namespace perpendicular_bisector_midpoint_l2310_231065

theorem perpendicular_bisector_midpoint :
  let P := (-8, 15)
  let Q := (6, -3)
  let R := ((-8 + 6) / 2, (15 - 3) / 2)
  3 * R.1 - 2 * R.2 = -15 :=
by
  let P := (-8, 15)
  let Q := (6, -3)
  let R := ((-8 + 6) / 2, (15 - 3) / 2)
  sorry

end perpendicular_bisector_midpoint_l2310_231065


namespace trip_cost_is_correct_l2310_231096

-- Given conditions
def bills_cost : ℕ := 3500
def save_per_month : ℕ := 500
def savings_duration_months : ℕ := 2 * 12
def savings : ℕ := save_per_month * savings_duration_months
def remaining_after_bills : ℕ := 8500

-- Prove that the cost of the trip to Paris is 3500 dollars
theorem trip_cost_is_correct : (savings - remaining_after_bills) = bills_cost :=
sorry

end trip_cost_is_correct_l2310_231096


namespace speed_of_second_train_equivalent_l2310_231045

noncomputable def relative_speed_in_m_per_s (time_seconds : ℝ) (total_distance_m : ℝ) : ℝ :=
total_distance_m / time_seconds

noncomputable def relative_speed_in_km_per_h (relative_speed_m_per_s : ℝ) : ℝ :=
relative_speed_m_per_s * 3.6

noncomputable def speed_of_second_train (relative_speed_km_per_h : ℝ) (speed_of_first_train_km_per_h : ℝ) : ℝ :=
relative_speed_km_per_h - speed_of_first_train_km_per_h

theorem speed_of_second_train_equivalent
  (length_of_first_train length_of_second_train : ℝ)
  (speed_of_first_train_km_per_h : ℝ)
  (time_of_crossing_seconds : ℝ) :
  speed_of_second_train
    (relative_speed_in_km_per_h (relative_speed_in_m_per_s time_of_crossing_seconds (length_of_first_train + length_of_second_train)))
    speed_of_first_train_km_per_h = 36 := by
  sorry

end speed_of_second_train_equivalent_l2310_231045


namespace find_coefficients_l2310_231019

variable (P Q x : ℝ)

theorem find_coefficients :
  (∀ x, x^2 - 8 * x - 20 = (x - 10) * (x + 2))
  → (∀ x, 6 * x - 4 = P * (x + 2) + Q * (x - 10))
  → P = 14 / 3 ∧ Q = 4 / 3 :=
by
  intros h1 h2
  sorry

end find_coefficients_l2310_231019


namespace height_of_smaller_cone_l2310_231013

theorem height_of_smaller_cone (h_frustum : ℝ) (area_lower_base area_upper_base : ℝ) 
  (h_frustum_eq : h_frustum = 18) 
  (area_lower_base_eq : area_lower_base = 144 * Real.pi) 
  (area_upper_base_eq : area_upper_base = 16 * Real.pi) : 
  ∃ (x : ℝ), x = 9 :=
by
  -- Definitions and assumptions go here
  sorry

end height_of_smaller_cone_l2310_231013


namespace option_d_correct_l2310_231046

theorem option_d_correct (a b c : ℝ) (h : a > b ∧ b > c ∧ c > 0) : a / b < a / c :=
by
  sorry

end option_d_correct_l2310_231046


namespace range_of_m_n_l2310_231037

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ :=
  m * Real.exp x + x^2 + n * x

theorem range_of_m_n (m n : ℝ) :
  (∃ x : ℝ, f m n x = 0) ∧ (∀ x : ℝ, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
sorry

end range_of_m_n_l2310_231037
