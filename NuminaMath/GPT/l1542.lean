import Mathlib

namespace trigonometric_values_l1542_154231

-- Define cos and sin terms
def cos (x : ℝ) : ℝ := sorry
def sin (x : ℝ) : ℝ := sorry

-- Define the condition given in the problem statement
def condition (x : ℝ) : Prop := cos x - 4 * sin x = 1

-- Define the result we need to prove
def result (x : ℝ) : Prop := sin x + 4 * cos x = 4 ∨ sin x + 4 * cos x = -4

-- The main statement in Lean 4 to be proved
theorem trigonometric_values (x : ℝ) : condition x → result x := by
  sorry

end trigonometric_values_l1542_154231


namespace quotient_is_four_l1542_154229

theorem quotient_is_four (dividend : ℕ) (k : ℕ) (h1 : dividend = 16) (h2 : k = 4) : dividend / k = 4 :=
by
  sorry

end quotient_is_four_l1542_154229


namespace periodicity_of_m_arith_fibonacci_l1542_154230

def m_arith_fibonacci (m : ℕ) (v : ℕ → ℕ) : Prop :=
∀ n : ℕ, v (n + 2) = (v n + v (n + 1)) % m

theorem periodicity_of_m_arith_fibonacci (m : ℕ) (v : ℕ → ℕ) 
  (hv : m_arith_fibonacci m v) : 
  ∃ r : ℕ, r ≤ m^2 ∧ ∀ n : ℕ, v (n + r) = v n := 
by
  sorry

end periodicity_of_m_arith_fibonacci_l1542_154230


namespace simplify_expression_l1542_154239

theorem simplify_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : a^4 + b^4 = a + b) (h2 : a^2 + b^2 = 2) :
  (a^2 / b^2 + b^2 / a^2 - 1 / (a^2 * b^2)) = 1 := 
sorry

end simplify_expression_l1542_154239


namespace find_interest_rate_l1542_154245

-- Definitions from the conditions
def principal : ℕ := 1050
def time_period : ℕ := 6
def interest : ℕ := 378  -- Interest calculated as Rs. 1050 - Rs. 672

-- Correct Answer
def interest_rate : ℕ := 6

-- Lean 4 statement of the proof problem
theorem find_interest_rate (P : ℕ) (t : ℕ) (I : ℕ) 
    (hP : P = principal) (ht : t = time_period) (hI : I = interest) : 
    (I * 100) / (P * t) = interest_rate :=
by {
    sorry
}

end find_interest_rate_l1542_154245


namespace area_ACD_l1542_154243

def base_ABD : ℝ := 8
def height_ABD : ℝ := 4
def base_ABC : ℝ := 4
def height_ABC : ℝ := 4

theorem area_ACD : (1/2 * base_ABD * height_ABD) - (1/2 * base_ABC * height_ABC) = 8 := by
  sorry

end area_ACD_l1542_154243


namespace problem1_problem2_l1542_154246

noncomputable def interval1 (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}
noncomputable def interval2 : Set ℝ := {x | x < -1 ∨ x > 3}

theorem problem1 (a : ℝ) : (interval1 a ∩ interval2 = interval1 a) ↔ a ∈ {x | x ≤ -2} ∪ {x | 1 ≤ x} := by sorry

theorem problem2 (a : ℝ) : (interval1 a ∩ interval2 ≠ ∅) ↔ a < -1 / 2 := by sorry

end problem1_problem2_l1542_154246


namespace geometric_seq_arithmetic_example_l1542_154210

noncomputable def a_n (n : ℕ) (q : ℝ) : ℝ :=
if n = 0 then 1 else q ^ n

theorem geometric_seq_arithmetic_example {q : ℝ} (h₀ : q ≠ 0)
    (h₁ : ∀ n : ℕ, a_n 0 q = 1)
    (h₂ : 2 * (2 * (q ^ 2)) = 3 * q) :
    (q + q^2 + (q^3)) = 14 :=
by sorry

end geometric_seq_arithmetic_example_l1542_154210


namespace minimum_value_of_K_l1542_154224

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / Real.exp x

noncomputable def f_K (K x : ℝ) : ℝ :=
  if f x ≤ K then f x else K

theorem minimum_value_of_K :
  (∀ x > 0, f_K (1 / Real.exp 1) x = f x) → (∃ K : ℝ, K = 1 / Real.exp 1) :=
by
  sorry

end minimum_value_of_K_l1542_154224


namespace cashier_total_value_l1542_154241

theorem cashier_total_value (total_bills : ℕ) (ten_bills : ℕ) (twenty_bills : ℕ)
  (h1 : total_bills = 30) (h2 : ten_bills = 27) (h3 : twenty_bills = 3) :
  (10 * ten_bills + 20 * twenty_bills) = 330 :=
by
  sorry

end cashier_total_value_l1542_154241


namespace toothbrushes_difference_l1542_154282

theorem toothbrushes_difference
  (total : ℕ)
  (jan : ℕ)
  (feb : ℕ)
  (mar : ℕ)
  (apr_may_sum : total = jan + feb + mar + 164)
  (apr_may_half : 164 / 2 = 82)
  (busy_month_given : feb = 67)
  (slow_month_given : mar = 46) :
  feb - mar = 21 :=
by
  sorry

end toothbrushes_difference_l1542_154282


namespace average_marks_l1542_154257

variable (M P C B : ℕ)

theorem average_marks (h1 : M + P = 20) (h2 : C = P + 20) 
  (h3 : B = 2 * M) (h4 : M ≤ 100) (h5 : P ≤ 100) (h6 : C ≤ 100) (h7 : B ≤ 100) :
  (M + C) / 2 = 20 := by
  sorry

end average_marks_l1542_154257


namespace infinite_integer_solutions_l1542_154211

theorem infinite_integer_solutions 
  (a b c k D x0 y0 : ℤ) 
  (hD_pos : D = b^2 - 4 * a * c) 
  (hD_non_square : (∀ n : ℤ, D ≠ n^2)) 
  (hk_nonzero : k ≠ 0) 
  (h_initial_sol : a * x0^2 + b * x0 * y0 + c * y0^2 = k) :
  ∃ (X Y : ℤ), a * X^2 + b * X * Y + c * Y^2 = k ∧
  (∀ (m : ℕ), ∃ (Xm Ym : ℤ), a * Xm^2 + b * Xm * Ym + c * Ym^2 = k ∧
  (Xm, Ym) ≠ (x0, y0)) :=
sorry

end infinite_integer_solutions_l1542_154211


namespace sum_of_values_l1542_154207

noncomputable def f (x : ℝ) : ℝ :=
if x < 3 then 5 * x + 20 else 3 * x - 21

theorem sum_of_values (h₁ : ∃ x, x < 3 ∧ f x = 4) (h₂ : ∃ x, x ≥ 3 ∧ f x = 4) :
  ∃a b : ℝ, a = -16 / 5 ∧ b = 25 / 3 ∧ (a + b = 77 / 15) :=
by {
  sorry
}

end sum_of_values_l1542_154207


namespace weeks_to_work_l1542_154201

def iPhone_cost : ℕ := 800
def trade_in_value : ℕ := 240
def weekly_earnings : ℕ := 80

theorem weeks_to_work (iPhone_cost trade_in_value weekly_earnings : ℕ) :
  (iPhone_cost - trade_in_value) / weekly_earnings = 7 :=
by
  sorry

end weeks_to_work_l1542_154201


namespace find_original_price_l1542_154218

-- Definitions for the conditions mentioned in the problem
variables {P : ℝ} -- Original price per gallon in dollars

-- Proof statement assuming the given conditions
theorem find_original_price 
  (h1 : ∃ P : ℝ, P > 0) -- There exists a positive price per gallon in dollars
  (h2 : (250 / (0.9 * P)) = (250 / P + 5)) -- After a 10% price reduction, 5 gallons more can be bought for $250
  : P = 25 / 4.5 := -- The solution states the original price per gallon is approximately $5.56
by
  sorry -- Proof omitted

end find_original_price_l1542_154218


namespace curve_equation_l1542_154233

theorem curve_equation
  (a b : ℝ)
  (h1 : a * 0 ^ 2 + b * (5 / 3) ^ 2 = 2)
  (h2 : a * 1 ^ 2 + b * 1 ^ 2 = 2) :
  (16 / 25) * x^2 + (9 / 25) * y^2 = 1 := 
by {
  sorry
}

end curve_equation_l1542_154233


namespace tallest_boy_is_Vladimir_l1542_154269

noncomputable def Andrei_statement1 (Boris_tallest: Prop) : Prop := ¬ Boris_tallest
def Andrei_statement2 (Vladimir_shortest: Prop) : Prop := Vladimir_shortest

def Boris_statement1 (Andrei_oldest: Prop) : Prop := Andrei_oldest
def Boris_statement2 (Andrei_shortest: Prop) : Prop := Andrei_shortest

def Vladimir_statement1 (Dmitry_taller: Prop) : Prop := Dmitry_taller
def Vladimir_statement2 (Dmitry_older: Prop) : Prop := Dmitry_older

noncomputable def Dmitry_statement1 (Vladimir_statement1: Prop) (Vladimir_statement2: Prop) : Prop :=
  Vladimir_statement1 ∧ Vladimir_statement2
def Dmitry_statement2 (Dmitry_oldest: Prop) : Prop := Dmitry_oldest

axiom one_statement_true_per_boy :
  ∀ {P₁ P₂: Prop}, (P₁ ∨ P₂) ∧ ¬ (P₁ ∧ P₂)

axiom no_same_height_or_age :
  ∀ {h1 h2 h3 h4 a1 a2 a3 a4 : ℕ},
    (h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4) ∧
    (a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)

theorem tallest_boy_is_Vladimir :
  ∀ (Andrei_shortest Vladimir_shortest Boris_tallest Dmitry_taller Dmitry_oldest Vladimir_older : Prop),
    Dmitry_statement2 Dmitry_oldest → 
    Boris_statement2 Andrei_shortest → 
    Andrei_statement1 Boris_tallest → 
    Vladimir_statement2 Vladimir_older → 
    Dmitry_statement1 (Vladimir_statement1 Dmitry_taller) (Vladimir_statement2 Vladimir_older) →
    ¬ Dmitry_taller →
    ¬ Boris_tallest →
    Vladimir = "the tallest boy" :=
  sorry

end tallest_boy_is_Vladimir_l1542_154269


namespace compute_expression_l1542_154209

theorem compute_expression : 2 * (Real.sqrt 144)^2 = 288 := by
  sorry

end compute_expression_l1542_154209


namespace kanul_initial_amount_l1542_154200

theorem kanul_initial_amount (X Y : ℝ) (loan : ℝ) (R : ℝ) 
  (h1 : loan = 2000)
  (h2 : R = 0.20)
  (h3 : Y = 0.15 * X + loan)
  (h4 : loan = R * Y) : 
  X = 53333.33 :=
by 
  -- The proof would come here, but is not necessary for this example
sorry

end kanul_initial_amount_l1542_154200


namespace treasure_coins_problem_l1542_154287

theorem treasure_coins_problem (N m n t k s u : ℤ) 
  (h1 : N = (2/3) * (2/3) * (2/3) * (m - 1) - (2/3) - (2^2 / 3^2))
  (h2 : N = 3 * n)
  (h3 : 8 * (m - 1) - 30 = 81 * k)
  (h4 : m - 1 = 3 * t)
  (h5 : 8 * t - 27 * k = 10)
  (h6 : m = 3 * t + 1)
  (h7 : k = 2 * s)
  (h8 : 4 * t - 27 * s = 5)
  (h9 : t = 8 + 27 * u)
  (h10 : s = 1 + 4 * u)
  (h11 : 110 ≤ 81 * u + 25)
  (h12 : 81 * u + 25 ≤ 200) :
  m = 187 :=
sorry

end treasure_coins_problem_l1542_154287


namespace percentage_of_children_allowed_to_draw_l1542_154240

def total_jelly_beans := 100
def total_children := 40
def remaining_jelly_beans := 36
def jelly_beans_per_child := 2

theorem percentage_of_children_allowed_to_draw :
  ((total_jelly_beans - remaining_jelly_beans) / jelly_beans_per_child : ℕ) * 100 / total_children = 80 := by
  sorry

end percentage_of_children_allowed_to_draw_l1542_154240


namespace chocolate_bar_pieces_l1542_154237

theorem chocolate_bar_pieces (X : ℕ) (h1 : X / 2 + X / 4 + 15 = X) : X = 60 :=
by
  sorry

end chocolate_bar_pieces_l1542_154237


namespace investment_interest_rate_calculation_l1542_154265

theorem investment_interest_rate_calculation :
  let initial_investment : ℝ := 15000
  let first_year_rate : ℝ := 0.08
  let first_year_investment : ℝ := initial_investment * (1 + first_year_rate)
  let second_year_investment : ℝ := 17160
  ∃ (s : ℝ), (first_year_investment * (1 + s / 100) = second_year_investment) → s = 6 :=
by
  sorry

end investment_interest_rate_calculation_l1542_154265


namespace eight_digit_numbers_count_l1542_154223

theorem eight_digit_numbers_count :
  let first_digit_choices := 9
  let remaining_digits_choices := 10 ^ 7
  9 * 10^7 = 90000000 :=
by
  sorry

end eight_digit_numbers_count_l1542_154223


namespace geometric_sequence_fifth_term_l1542_154264

theorem geometric_sequence_fifth_term (a r : ℝ) (h1 : a * r^2 = 9) (h2 : a * r^6 = 1) : a * r^4 = 3 :=
by
  sorry

end geometric_sequence_fifth_term_l1542_154264


namespace no_real_solution_l1542_154288

theorem no_real_solution (x y : ℝ) (h: y = 3 * x - 1) : ¬ (4 * y ^ 2 + y + 3 = 3 * (8 * x ^ 2 + 3 * y + 1)) :=
by
  sorry

end no_real_solution_l1542_154288


namespace total_hike_time_l1542_154260

/-!
# Problem Statement
Jeannie hikes the 12 miles to Mount Overlook at a pace of 4 miles per hour, 
and then returns at a pace of 6 miles per hour. Prove that the total time 
Jeannie spent on her hike is 5 hours.
-/

def distance_to_mountain : ℝ := 12
def pace_up : ℝ := 4
def pace_down : ℝ := 6

theorem total_hike_time :
  (distance_to_mountain / pace_up) + (distance_to_mountain / pace_down) = 5 := 
by 
  sorry

end total_hike_time_l1542_154260


namespace starting_number_divisible_by_3_l1542_154296

theorem starting_number_divisible_by_3 (x : ℕ) (h₁ : ∀ n, 1 ≤ n → n < 14 → ∃ k, x + (n - 1) * 3 = 3 * k ∧ x + (n - 1) * 3 ≤ 50) :
  x = 12 :=
by
  sorry

end starting_number_divisible_by_3_l1542_154296


namespace car_travel_distance_l1542_154298

noncomputable def distance_traveled (diameter : ℝ) (revolutions : ℝ) : ℝ :=
  let pi := Real.pi
  let circumference := pi * diameter
  circumference * revolutions / 12 / 5280

theorem car_travel_distance
  (diameter : ℝ)
  (revolutions : ℝ)
  (h_diameter : diameter = 13)
  (h_revolutions : revolutions = 775.5724667489372) :
  distance_traveled diameter revolutions = 0.5 :=
by
  simp [distance_traveled, h_diameter, h_revolutions, Real.pi]
  sorry

end car_travel_distance_l1542_154298


namespace solve_for_x_l1542_154202

theorem solve_for_x (x : ℝ) : (2 / 7) * (1 / 4) * x - 3 = 5 → x = 112 := by
  sorry

end solve_for_x_l1542_154202


namespace probability_of_roll_6_after_E_l1542_154242

/- Darryl has a six-sided die with faces 1, 2, 3, 4, 5, 6.
   The die is weighted so that one face comes up with probability 1/2,
   and the other five faces have equal probability.
   Darryl does not know which side is weighted, but each face is equally likely to be the weighted one.
   Darryl rolls the die 5 times and gets a 1, 2, 3, 4, and 5 in some unspecified order. -/

def probability_of_next_roll_getting_6 : ℚ :=
  let p_weighted := (1 / 2 : ℚ)
  let p_unweighted := (1 / 10 : ℚ)
  let p_w6_given_E := (1 / 26 : ℚ)
  let p_not_w6_given_E := (25 / 26 : ℚ)
  p_w6_given_E * p_weighted + p_not_w6_given_E * p_unweighted

theorem probability_of_roll_6_after_E : probability_of_next_roll_getting_6 = 3 / 26 := sorry

end probability_of_roll_6_after_E_l1542_154242


namespace Tim_soda_cans_l1542_154277

noncomputable def initial_cans : ℕ := 22
noncomputable def taken_cans : ℕ := 6
noncomputable def remaining_cans : ℕ := initial_cans - taken_cans
noncomputable def bought_cans : ℕ := remaining_cans / 2
noncomputable def final_cans : ℕ := remaining_cans + bought_cans

theorem Tim_soda_cans :
  final_cans = 24 :=
by
  sorry

end Tim_soda_cans_l1542_154277


namespace find_y_l1542_154234

noncomputable def G (a b c d : ℝ) : ℝ := a ^ b + c ^ d

theorem find_y (h : G 3 y 2 5 = 100) : y = Real.log 68 / Real.log 3 := 
by
  have hG : G 3 y 2 5 = 3 ^ y + 2 ^ 5 := rfl
  sorry

end find_y_l1542_154234


namespace area_of_region_l1542_154252

theorem area_of_region : 
  (∃ (A : ℝ), A = 12 ∧ ∀ (x y : ℝ), |x| + |y| + |x - 2| ≤ 4 → 
    (0 ≤ y ∧ y ≤ 6 - 2*x ∧ x ≥ 2) ∨
    (0 ≤ y ∧ y ≤ 2 ∧ 0 ≤ x ∧ x < 2) ∨
    (0 ≤ y ∧ y ≤ 2*x + 2 ∧ -1 ≤ x ∧ x < 0) ∨
    (0 ≤ y ∧ y ≤ 2*x + 2 ∧ x < -1)) :=
sorry

end area_of_region_l1542_154252


namespace jonas_tshirts_count_l1542_154267

def pairs_to_individuals (pairs : Nat) : Nat := pairs * 2

variable (num_pairs_socks : Nat := 20)
variable (num_pairs_shoes : Nat := 5)
variable (num_pairs_pants : Nat := 10)
variable (num_additional_pairs_socks : Nat := 35)

def total_individual_items_without_tshirts : Nat :=
  pairs_to_individuals num_pairs_socks +
  pairs_to_individuals num_pairs_shoes +
  pairs_to_individuals num_pairs_pants

def total_individual_items_desired : Nat :=
  total_individual_items_without_tshirts +
  pairs_to_individuals num_additional_pairs_socks

def tshirts_jonas_needs : Nat :=
  total_individual_items_desired - total_individual_items_without_tshirts

theorem jonas_tshirts_count : tshirts_jonas_needs = 70 := by
  sorry

end jonas_tshirts_count_l1542_154267


namespace ratio_third_to_first_l1542_154258

theorem ratio_third_to_first (F S T : ℕ) (h1 : F = 33) (h2 : S = 4 * F) (h3 : (F + S + T) / 3 = 77) :
  T / F = 2 :=
by
  sorry

end ratio_third_to_first_l1542_154258


namespace complex_fraction_evaluation_l1542_154208

theorem complex_fraction_evaluation :
  ( 
    ((3 + 1/3) / 10 + 0.175 / 0.35) / 
    (1.75 - (1 + 11/17) * (51/56)) - 
    ((11/18 - 1/15) / 1.4) / 
    ((0.5 - 1/9) * 3)
  ) = 1/2 := 
sorry

end complex_fraction_evaluation_l1542_154208


namespace find_larger_integer_l1542_154249

noncomputable def larger_integer (a b : ℤ) := max a b

theorem find_larger_integer (a b : ℕ) 
  (h1 : a/b = 7/3) 
  (h2 : a * b = 294): 
  larger_integer a b = 7 * Real.sqrt 14 :=
by
  -- Proof goes here
  sorry

end find_larger_integer_l1542_154249


namespace area_of_base_of_cone_l1542_154221

theorem area_of_base_of_cone (semicircle_area : ℝ) (h1 : semicircle_area = 2 * Real.pi) : 
  ∃ (base_area : ℝ), base_area = Real.pi :=
by
  sorry

end area_of_base_of_cone_l1542_154221


namespace simplify_and_evaluate_l1542_154219

variable (a : ℝ)
variable (ha : a = Real.sqrt 3 - 1)

theorem simplify_and_evaluate : 
  (1 + 3 / (a - 2)) / ((a^2 + 2 * a + 1) / (a - 2)) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_l1542_154219


namespace race_distance_l1542_154205

theorem race_distance (x : ℝ) (D : ℝ) (vA vB : ℝ) (head_start win_margin : ℝ):
  vA = 5 * x →
  vB = 4 * x →
  head_start = 100 →
  win_margin = 200 →
  (D - win_margin) / vB = (D - head_start) / vA →
  D = 600 :=
by 
  sorry

end race_distance_l1542_154205


namespace caterpillars_and_leaves_l1542_154280

def initial_caterpillars : Nat := 14
def caterpillars_after_storm : Nat := initial_caterpillars - 3
def hatched_eggs : Nat := 6
def caterpillars_after_hatching : Nat := caterpillars_after_storm + hatched_eggs
def leaves_eaten_by_babies : Nat := 18
def caterpillars_after_cocooning : Nat := caterpillars_after_hatching - 9
def moth_caterpillars : Nat := caterpillars_after_cocooning / 2
def butterfly_caterpillars : Nat := caterpillars_after_cocooning - moth_caterpillars
def leaves_eaten_per_moth_per_day : Nat := 4
def days_in_week : Nat := 7
def total_leaves_eaten_by_moths : Nat := moth_caterpillars * leaves_eaten_per_moth_per_day * days_in_week
def total_leaves_eaten_by_babies_and_moths : Nat := leaves_eaten_by_babies + total_leaves_eaten_by_moths

theorem caterpillars_and_leaves :
  (caterpillars_after_cocooning = 8) ∧ (total_leaves_eaten_by_babies_and_moths = 130) :=
by
  -- proof to be filled in
  sorry

end caterpillars_and_leaves_l1542_154280


namespace find_n_l1542_154206

theorem find_n (n : ℕ) (h : 12^(4 * n) = (1/12)^(n - 30)) : n = 6 := 
by {
  sorry 
}

end find_n_l1542_154206


namespace perimeter_pentagon_l1542_154290

noncomputable def AB : ℝ := 1
noncomputable def BC : ℝ := Real.sqrt 2
noncomputable def CD : ℝ := Real.sqrt 3
noncomputable def DE : ℝ := 2

noncomputable def AC : ℝ := Real.sqrt (AB^2 + BC^2)
noncomputable def AD : ℝ := Real.sqrt (AC^2 + CD^2)
noncomputable def AE : ℝ := Real.sqrt (AD^2 + DE^2)

theorem perimeter_pentagon (ABCDE : List ℝ) (H : ABCDE = [AB, BC, CD, DE, AE]) :
  List.sum ABCDE = 3 + Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 10 :=
by
  sorry -- Proof skipped as instructed

end perimeter_pentagon_l1542_154290


namespace w1_relation_w2_relation_maximize_total_profit_l1542_154248

def w1 (x : ℕ) : ℤ := 200 * x - 10000

def w2 (x : ℕ) : ℤ := -(x ^ 2) + 1000 * x - 50000

def total_sales_vol (x y : ℕ) : Prop := x + y = 1000

def max_profit_volumes (x y : ℕ) : Prop :=
  total_sales_vol x y ∧ x = 600 ∧ y = 400

theorem w1_relation (x : ℕ) :
  w1 x = 200 * x - 10000 := 
sorry

theorem w2_relation (x : ℕ) :
  w2 x = -(x ^ 2) + 1000 * x - 50000 := 
sorry

theorem maximize_total_profit (x y : ℕ) :
  total_sales_vol x y → max_profit_volumes x y := 
sorry

end w1_relation_w2_relation_maximize_total_profit_l1542_154248


namespace cost_per_pizza_l1542_154220

theorem cost_per_pizza (total_amount : ℝ) (num_pizzas : ℕ) (H : total_amount = 24) (H1 : num_pizzas = 3) : 
  (total_amount / num_pizzas) = 8 := 
by 
  sorry

end cost_per_pizza_l1542_154220


namespace binom_9_5_eq_126_l1542_154284

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := by
  sorry

end binom_9_5_eq_126_l1542_154284


namespace mean_home_runs_l1542_154203

theorem mean_home_runs :
  let n_5 := 3
  let n_8 := 5
  let n_9 := 3
  let n_11 := 1
  let total_home_runs := 5 * n_5 + 8 * n_8 + 9 * n_9 + 11 * n_11
  let total_players := n_5 + n_8 + n_9 + n_11
  let mean := total_home_runs / total_players
  mean = 7.75 :=
by
  sorry

end mean_home_runs_l1542_154203


namespace ones_digit_of_prime_in_arithmetic_sequence_l1542_154216

theorem ones_digit_of_prime_in_arithmetic_sequence (p q r : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h1 : p < q) (h2 : q < r) 
  (arithmetic_sequence : q = p + 4 ∧ r = q + 4)
  (h : p > 5) : 
    (p % 10 = 3 ∨ p % 10 = 9) :=
sorry

end ones_digit_of_prime_in_arithmetic_sequence_l1542_154216


namespace ben_has_56_marbles_l1542_154217

-- We define the conditions first
variables (B : ℕ) (L : ℕ)

-- Leo has 20 more marbles than Ben
def condition1 : Prop := L = B + 20

-- Total number of marbles is 132
def condition2 : Prop := B + L = 132

-- The goal: proving the number of marbles Ben has is 56
theorem ben_has_56_marbles (h1 : condition1 B L) (h2 : condition2 B L) : B = 56 :=
by sorry

end ben_has_56_marbles_l1542_154217


namespace sample_size_is_59_l1542_154294

def totalStudents : Nat := 295
def samplingRatio : Nat := 5

theorem sample_size_is_59 : totalStudents / samplingRatio = 59 := 
by
  sorry

end sample_size_is_59_l1542_154294


namespace complement_intersection_subset_condition_l1542_154238

-- Definition of sets A, B, and C
def A := { x : ℝ | 3 ≤ x ∧ x < 7 }
def B := { x : ℝ | 2 < x ∧ x < 10 }
def C (a : ℝ) := { x : ℝ | x < a }

-- Proof problem 1 statement
theorem complement_intersection :
  ( { x : ℝ | x < 3 ∨ x ≥ 7 } ∩ { x : ℝ | 2 < x ∧ x < 10 } ) = { x : ℝ | 2 < x ∧ x < 3 ∨ 7 ≤ x ∧ x < 10 } :=
by
  sorry

-- Proof problem 2 statement
theorem subset_condition (a : ℝ) :
  ( { x : ℝ | 3 ≤ x ∧ x < 7 } ⊆ { x : ℝ | x < a } ) → (a ≥ 7) :=
by
  sorry

end complement_intersection_subset_condition_l1542_154238


namespace find_positive_X_l1542_154259

variable (X : ℝ) (Y : ℝ)

def hash_rel (X Y : ℝ) : ℝ :=
  X^2 + Y^2

theorem find_positive_X :
  hash_rel X 7 = 250 → X = Real.sqrt 201 :=
by
  sorry

end find_positive_X_l1542_154259


namespace Penelope_Candies_l1542_154226

variable (M : ℕ) (S : ℕ)
variable (h1 : 5 * S = 3 * M)
variable (h2 : M = 25)

theorem Penelope_Candies : S = 15 := by
  sorry

end Penelope_Candies_l1542_154226


namespace difference_in_dimes_l1542_154291

theorem difference_in_dimes : 
  ∀ (a b c : ℕ), (a + b + c = 100) → (5 * a + 10 * b + 25 * c = 835) → 
  (∀ b_max b_min, (b_max = 67) ∧ (b_min = 3) → (b_max - b_min = 64)) :=
by
  intros a b c h1 h2 b_max b_min h_bounds
  sorry

end difference_in_dimes_l1542_154291


namespace find_a_and_b_l1542_154236

noncomputable def a_and_b (x y : ℝ) (a b : ℝ) : Prop :=
  a = Real.sqrt x + Real.sqrt y ∧ b = Real.sqrt (x + 2) + Real.sqrt (y + 2) ∧
  ∃ n : ℤ, a = n ∧ b = n + 2

theorem find_a_and_b (x y : ℝ) (a b : ℝ)
  (h₁ : 0 ≤ x)
  (h₂ : 0 ≤ y)
  (h₃ : a_and_b x y a b)
  (h₄ : ∃ n : ℤ, a = n ∧ b = n + 2) :
  a = 1 ∧ b = 3 := by
  sorry

end find_a_and_b_l1542_154236


namespace trig_identity_solution_l1542_154255

theorem trig_identity_solution (α : ℝ) (h : Real.tan α = -1 / 2) :
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1 / 3 :=
by
  sorry

end trig_identity_solution_l1542_154255


namespace find_DF_l1542_154289

-- Conditions
variables {A B C D E F : Type}
variables {BC EF AC DF : ℝ}
variable (h_similar : similar_triangles A B C D E F)
variable (h_BC : BC = 6)
variable (h_EF : EF = 4)
variable (h_AC : AC = 9)

-- Question: Prove DF = 6 given the above conditions
theorem find_DF : DF = 6 :=
by
  sorry

end find_DF_l1542_154289


namespace fraction_division_l1542_154299

theorem fraction_division :
  (3 / 4) / (5 / 8) = 6 / 5 :=
by
  sorry

end fraction_division_l1542_154299


namespace proof_inequalities_l1542_154268

theorem proof_inequalities (A B C D E : ℝ) (p q r s t : ℝ)
  (h1 : A < B) (h2 : B < C) (h3 : C < D) (h4 : D < E)
  (h5 : p = B - A) (h6 : q = C - A) (h7 : r = D - A)
  (h8 : s = E - B) (h9 : t = E - D)
  (ineq1 : p + 2 * s > r + t)
  (ineq2 : r + t > p)
  (ineq3 : r + t > s) :
  (p < r / 2) ∧ (s < t + p / 2) :=
by 
  sorry

end proof_inequalities_l1542_154268


namespace distance_covered_at_40_kmph_l1542_154212

theorem distance_covered_at_40_kmph (x : ℝ) 
  (h₁ : x / 40 + (250 - x) / 60 = 5.5) :
  x = 160 :=
sorry

end distance_covered_at_40_kmph_l1542_154212


namespace max_theater_members_l1542_154215

theorem max_theater_members (N : ℕ) :
  (∃ (k : ℕ), (N = k^2 + 3)) ∧ (∃ (n : ℕ), (N = n * (n + 9))) → N ≤ 360 :=
by
  sorry

end max_theater_members_l1542_154215


namespace possible_m_value_l1542_154283

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3 - (1/2)*x - 1
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^3 + m / x

theorem possible_m_value :
  ∃ m : ℝ, (m = (1/2) - (1/Real.exp 1)) ∧
    (∀ x1 x2 : ℝ, 
      (f x1 = g (-x1) m) →
      (f x2 = g (-x2) m) →
      x1 ≠ 0 ∧ x2 ≠ 0 ∧
      m = x1 * Real.exp x1 - (1/2) * x1^2 - x1 ∧
      m = x2 * Real.exp x2 - (1/2) * x2^2 - x2) :=
by
  sorry

end possible_m_value_l1542_154283


namespace seunghyeon_pizza_diff_l1542_154274

theorem seunghyeon_pizza_diff (S Y : ℕ) (h : S - 2 = Y + 7) : S - Y = 9 :=
by {
  sorry
}

end seunghyeon_pizza_diff_l1542_154274


namespace candy_crush_ratio_l1542_154204

theorem candy_crush_ratio :
  ∃ m : ℕ, (400 + (400 - 70) + (400 - 70) * m = 1390) ∧ (m = 2) :=
by
  sorry

end candy_crush_ratio_l1542_154204


namespace solve_for_x_l1542_154232

theorem solve_for_x (x : ℝ) (h₀ : x^2 - 2 * x = 0) (h₁ : x ≠ 0) : x = 2 :=
sorry

end solve_for_x_l1542_154232


namespace sales_volume_conditions_l1542_154293

noncomputable def sales_volume (x : ℝ) (a k : ℝ) : ℝ :=
if 1 < x ∧ x ≤ 3 then a * (x - 4)^2 + 6 / (x - 1)
else if 3 < x ∧ x ≤ 5 then k * x + 7
else 0

theorem sales_volume_conditions (a k : ℝ) :
  (sales_volume 3 a k = 4) ∧ (sales_volume 5 a k = 2) ∧
  ((∃ x, 1 < x ∧ x ≤ 3 ∧ sales_volume x a k = 10) ∨ 
   (∃ x, 3 < x ∧ x ≤ 5 ∧ sales_volume x a k = 9)) :=
sorry

end sales_volume_conditions_l1542_154293


namespace freeze_alcohol_time_l1542_154295

theorem freeze_alcohol_time :
  ∀ (init_temp freeze_temp : ℝ)
    (cooling_rate : ℝ), 
    init_temp = 12 → 
    freeze_temp = -117 → 
    cooling_rate = 1.5 →
    (freeze_temp - init_temp) / cooling_rate = -129 / cooling_rate :=
by
  intros init_temp freeze_temp cooling_rate h1 h2 h3
  rw [h2, h1, h3]
  exact sorry

end freeze_alcohol_time_l1542_154295


namespace parametric_line_eq_l1542_154297

theorem parametric_line_eq (t : ℝ) : 
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x = 3 * t + 6 → y = 5 * t - 8 → y = m * x + b)) ∧ m = 5 / 3 ∧ b = -18 :=
sorry

end parametric_line_eq_l1542_154297


namespace max_value_x_plus_y_l1542_154228

theorem max_value_x_plus_y : ∀ (x y : ℝ), 
  (5 * x + 3 * y ≤ 9) → 
  (3 * x + 5 * y ≤ 11) → 
  x + y ≤ 32 / 17 :=
by
  intros x y h1 h2
  -- proof steps go here
  sorry

end max_value_x_plus_y_l1542_154228


namespace convert_500_to_base2_l1542_154261

theorem convert_500_to_base2 :
  let n_base10 : ℕ := 500
  let n_base8 : ℕ := 7 * 64 + 6 * 8 + 4
  let n_base2 : ℕ := 1 * 256 + 1 * 128 + 1 * 64 + 1 * 32 + 1 * 16 + 0 * 8 + 1 * 4 + 0 * 2 + 0
  n_base10 = 500 ∧ n_base8 = 500 ∧ n_base2 = n_base8 :=
by
  sorry

end convert_500_to_base2_l1542_154261


namespace file_size_correct_l1542_154214

theorem file_size_correct:
  (∀ t1 t2 : ℕ, (60 / 5 = t1) ∧ (15 - t1 = t2) ∧ (t2 * 10 = 30) → (60 + 30 = 90)) := 
by
  sorry

end file_size_correct_l1542_154214


namespace spade_problem_l1542_154266

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_problem : spade 2 (spade 3 (spade 1 4)) = -46652 := 
by sorry

end spade_problem_l1542_154266


namespace evie_shells_l1542_154273

theorem evie_shells (shells_per_day : ℕ) (days : ℕ) (gifted_shells : ℕ) 
  (h1 : shells_per_day = 10) 
  (h2 : days = 6)
  (h3 : gifted_shells = 2) : 
  shells_per_day * days - gifted_shells = 58 := 
by
  sorry

end evie_shells_l1542_154273


namespace five_letter_word_combinations_l1542_154244

open Nat

theorem five_letter_word_combinations :
  let first_letter_choices := 26
  let other_letter_choices := 26
  let total_combinations := first_letter_choices ^ 1 * other_letter_choices ^ 3
  total_combinations = 456976 := 
by
  let first_letter_choices := 26
  let other_letter_choices := 26
  let total_combinations := first_letter_choices ^ 1 * other_letter_choices ^ 3
  show total_combinations = 456976
  sorry

end five_letter_word_combinations_l1542_154244


namespace car_mpg_in_city_l1542_154222

theorem car_mpg_in_city:
  ∃ (h c T : ℝ), 
    (420 = h * T) ∧ 
    (336 = c * T) ∧ 
    (c = h - 6) ∧ 
    (c = 24) :=
by
  sorry

end car_mpg_in_city_l1542_154222


namespace inverse_of_3_mod_185_l1542_154281

theorem inverse_of_3_mod_185 : ∃ x : ℕ, 0 ≤ x ∧ x < 185 ∧ 3 * x ≡ 1 [MOD 185] :=
by
  use 62
  sorry

end inverse_of_3_mod_185_l1542_154281


namespace moles_of_HCl_needed_l1542_154247

-- Define the reaction and corresponding stoichiometry
def reaction_relates (NaHSO3 HCl NaCl H2O SO2 : ℕ) : Prop :=
  NaHSO3 = HCl ∧ HCl = NaCl ∧ NaCl = H2O ∧ H2O = SO2

-- Given condition: one mole of each reactant produces one mole of each product
axiom reaction_stoichiometry : reaction_relates 1 1 1 1 1

-- Prove that 2 moles of NaHSO3 reacting with 2 moles of HCl forms 2 moles of NaCl
theorem moles_of_HCl_needed :
  ∀ (NaHSO3 HCl NaCl : ℕ), reaction_relates NaHSO3 HCl NaCl NaCl NaCl → NaCl = 2 → HCl = 2 :=
by
  intros NaHSO3 HCl NaCl h_eq h_NaCl
  sorry

end moles_of_HCl_needed_l1542_154247


namespace purchase_price_of_jacket_l1542_154235

theorem purchase_price_of_jacket (S P : ℝ) (h1 : S = P + 0.30 * S)
                                (SP : ℝ) (h2 : SP = 0.80 * S)
                                (h3 : 8 = SP - P) :
                                P = 56 := by
  sorry

end purchase_price_of_jacket_l1542_154235


namespace inv_113_mod_114_l1542_154286

theorem inv_113_mod_114 :
  (113 * 113) % 114 = 1 % 114 :=
by
  sorry

end inv_113_mod_114_l1542_154286


namespace range_of_a_l1542_154213

theorem range_of_a (a : ℝ) (h : a - 2 * 1 + 4 > 0) : a > -2 :=
by
  -- proof is not required
  sorry

end range_of_a_l1542_154213


namespace total_lucky_stars_l1542_154279

theorem total_lucky_stars : 
  (∃ n : ℕ, 10 * n + 6 = 116 ∧ 4 * 8 + (n - 4) * 12 = 116) → 
  116 = 116 := 
by
  intro h
  obtain ⟨n, h1, h2⟩ := h
  sorry

end total_lucky_stars_l1542_154279


namespace unique_partition_no_primes_l1542_154262

open Set

def C_oplus_C (C : Set ℕ) : Set ℕ :=
  {z | ∃ x y, x ∈ C ∧ y ∈ C ∧ x ≠ y ∧ z = x + y}

def is_partition (A B : Set ℕ) : Prop :=
  (A ∪ B = univ) ∧ (A ∩ B = ∅)

theorem unique_partition_no_primes (A B : Set ℕ) :
  (is_partition A B) ∧ (∀ x ∈ C_oplus_C A, ¬Nat.Prime x) ∧ (∀ x ∈ C_oplus_C B, ¬Nat.Prime x) ↔ 
    (A = { n | n % 2 = 1 }) ∧ (B = { n | n % 2 = 0 }) :=
sorry

end unique_partition_no_primes_l1542_154262


namespace geometric_sequence_fourth_term_l1542_154254

theorem geometric_sequence_fourth_term (a₁ a₂ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 1/3) :
    ∃ a₄ : ℝ, a₄ = 1/243 :=
sorry

end geometric_sequence_fourth_term_l1542_154254


namespace nat_pairs_satisfy_conditions_l1542_154253

theorem nat_pairs_satisfy_conditions :
  ∃ (a b : ℕ), 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∧
  (∃ k : ℤ, (a^2 + b) = k * (b^2 - a)) ∧
  (∃ l : ℤ, (b^2 + a) = l * (a^2 - b)) := 
sorry

end nat_pairs_satisfy_conditions_l1542_154253


namespace least_value_xy_l1542_154270

theorem least_value_xy {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : (1 : ℚ) / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_value_xy_l1542_154270


namespace added_number_is_five_l1542_154272

variable (n x : ℤ)

theorem added_number_is_five (h1 : n % 25 = 4) (h2 : (n + x) % 5 = 4) : x = 5 :=
by
  sorry

end added_number_is_five_l1542_154272


namespace notebook_cost_correct_l1542_154271

def totalSpent : ℕ := 32
def costBackpack : ℕ := 15
def costPen : ℕ := 1
def costPencil : ℕ := 1
def numberOfNotebooks : ℕ := 5
def costPerNotebook : ℕ := 3

theorem notebook_cost_correct (h_totalSpent : totalSpent = 32)
    (h_costBackpack : costBackpack = 15)
    (h_costPen : costPen = 1)
    (h_costPencil : costPencil = 1)
    (h_numberOfNotebooks : numberOfNotebooks = 5) :
    (totalSpent - (costBackpack + costPen + costPencil)) / numberOfNotebooks = costPerNotebook :=
by
  sorry

end notebook_cost_correct_l1542_154271


namespace arrange_in_ascending_order_l1542_154275

theorem arrange_in_ascending_order (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : 5 * x < 0.5 * x ∧ 0.5 * x < 5 - x := by
  sorry

end arrange_in_ascending_order_l1542_154275


namespace lawn_width_l1542_154285

variable (W : ℝ)
variable (h₁ : 80 * 15 + 15 * W - 15 * 15 = 1875)
variable (h₂ : 5625 = 3 * 1875)

theorem lawn_width (h₁ : 80 * 15 + 15 * W - 15 * 15 = 1875) (h₂ : 5625 = 3 * 1875) : 
  W = 60 := 
sorry

end lawn_width_l1542_154285


namespace burger_cost_is_350_l1542_154251

noncomputable def cost_of_each_burger (tip steak_cost steak_quantity ice_cream_cost ice_cream_quantity money_left: ℝ) : ℝ :=
(tip - money_left - (steak_cost * steak_quantity + ice_cream_cost * ice_cream_quantity)) / 2

theorem burger_cost_is_350 :
  cost_of_each_burger 99 24 2 2 3 38 = 3.5 :=
by
  sorry

end burger_cost_is_350_l1542_154251


namespace find_a_for_arithmetic_progression_roots_l1542_154250

theorem find_a_for_arithmetic_progression_roots (x a : ℝ) : 
  (∀ (x : ℝ), x^4 - a*x^2 + 1 = 0) → 
  (∃ (t1 t2 : ℝ), t1 > 0 ∧ t2 > 0 ∧ (t2 = 9*t1) ∧ (t1 + t2 = a) ∧ (t1 * t2 = 1)) → 
  (a = 10/3) := 
  by 
    intros h1 h2
    sorry

end find_a_for_arithmetic_progression_roots_l1542_154250


namespace ratio_of_spinsters_to_cats_l1542_154256

theorem ratio_of_spinsters_to_cats :
  (∀ S C : ℕ, (S : ℚ) / (C : ℚ) = 2 / 9) ↔
  (∃ S C : ℕ, S = 18 ∧ C = S + 63 ∧ (S : ℚ) / (C : ℚ) = 2 / 9) :=
sorry

end ratio_of_spinsters_to_cats_l1542_154256


namespace map_length_represents_75_km_l1542_154278
-- First, we broaden the import to bring in all the necessary libraries.

-- Define the conditions given in the problem.
def cm_to_km_ratio (cm : ℕ) (km : ℕ) : ℕ := km / cm

def map_represents (length_cm : ℕ) (length_km : ℕ) : Prop :=
  length_km = length_cm * cm_to_km_ratio 15 45

-- Rewrite the problem statement as a theorem in Lean 4.
theorem map_length_represents_75_km : map_represents 25 75 :=
by
  sorry

end map_length_represents_75_km_l1542_154278


namespace bamboo_tube_middle_capacity_l1542_154292

-- Definitions and conditions
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem bamboo_tube_middle_capacity:
  ∃ a d, (arithmetic_sequence a d 0 + arithmetic_sequence a d 1 + arithmetic_sequence a d 2 = 3.9) ∧
         (arithmetic_sequence a d 5 + arithmetic_sequence a d 6 + arithmetic_sequence a d 7 + arithmetic_sequence a d 8 = 3) ∧
         (arithmetic_sequence a d 4 = 1) :=
sorry

end bamboo_tube_middle_capacity_l1542_154292


namespace problem_condition_l1542_154225

theorem problem_condition (a : ℝ) (x : ℝ) (h_a : -1 ≤ a ∧ a ≤ 1) :
  (x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) :=
sorry

end problem_condition_l1542_154225


namespace unit_digit_is_nine_l1542_154276

theorem unit_digit_is_nine (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : a ≠ 0) (h4 : a + b + a * b = 10 * a + b) : b = 9 := 
by 
  sorry

end unit_digit_is_nine_l1542_154276


namespace gervais_avg_mileage_l1542_154263
variable (x : ℤ)

def gervais_daily_mileage : Prop := ∃ (x : ℤ), (3 * x = 1250 - 305) ∧ x = 315

theorem gervais_avg_mileage : gervais_daily_mileage :=
by
  sorry

end gervais_avg_mileage_l1542_154263


namespace bacteria_growth_time_l1542_154227

-- Define the conditions and the final proof statement
theorem bacteria_growth_time (n0 n1 : ℕ) (t : ℕ) :
  (∀ (k : ℕ), k > 0 → n1 = n0 * 3 ^ k) →
  (∀ (h : ℕ), t = 5 * h) →
  n0 = 200 →
  n1 = 145800 →
  t = 30 :=
by
  sorry

end bacteria_growth_time_l1542_154227
