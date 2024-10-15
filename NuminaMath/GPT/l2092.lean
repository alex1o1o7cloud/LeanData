import Mathlib

namespace NUMINAMATH_GPT_probability_p_eq_l2092_209245

theorem probability_p_eq (p q : ℝ) (h_q : q = 1 - p)
  (h_eq : (Nat.choose 10 5) * p^5 * q^5 = (Nat.choose 10 6) * p^6 * q^4) : 
  p = 6 / 11 :=
by
  sorry

end NUMINAMATH_GPT_probability_p_eq_l2092_209245


namespace NUMINAMATH_GPT_sum_of_pairs_l2092_209210

theorem sum_of_pairs (a : ℕ → ℝ) (h1 : ∀ n, a n ≠ 0)
  (h2 : ∀ n, a n * a (n + 3) = a (n + 2) * a (n + 5))
  (h3 : a 1 * a 2 + a 3 * a 4 + a 5 * a 6 = 6) :
  a 1 * a 2 + a 3 * a 4 + a 5 * a 6 + a 7 * a 8 + a 9 * a 10 + a 11 * a 12 + 
  a 13 * a 14 + a 15 * a 16 + a 17 * a 18 + a 19 * a 20 + a 21 * a 22 + 
  a 23 * a 24 + a 25 * a 26 + a 27 * a 28 + a 29 * a 30 + a 31 * a 32 + 
  a 33 * a 34 + a 35 * a 36 + a 37 * a 38 + a 39 * a 40 + a 41 * a 42 = 42 := 
sorry

end NUMINAMATH_GPT_sum_of_pairs_l2092_209210


namespace NUMINAMATH_GPT_neg_q_necessary_not_sufficient_for_neg_p_l2092_209254

-- Proposition p: |x + 2| > 2
def p (x : ℝ) : Prop := abs (x + 2) > 2

-- Proposition q: 1 / (3 - x) > 1
def q (x : ℝ) : Prop := 1 / (3 - x) > 1

-- Negation of p and q
def neg_p (x : ℝ) : Prop := -4 ≤ x ∧ x ≤ 0
def neg_q (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 3

-- Theorem: negation of q is a necessary but not sufficient condition for negation of p
theorem neg_q_necessary_not_sufficient_for_neg_p :
  (∀ x : ℝ, neg_p x → neg_q x) ∧ (∃ x : ℝ, neg_q x ∧ ¬neg_p x) :=
by
  sorry

end NUMINAMATH_GPT_neg_q_necessary_not_sufficient_for_neg_p_l2092_209254


namespace NUMINAMATH_GPT_n_div_p_eq_27_l2092_209259

theorem n_div_p_eq_27 (m n p : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : p ≠ 0)
    (h4 : ∃ r1 r2 : ℝ, r1 * r2 = m ∧ r1 + r2 = -p ∧ (3 * r1) * (3 * r2) = n ∧ 3 * (r1 + r2) = -m)
    : n / p = 27 := sorry

end NUMINAMATH_GPT_n_div_p_eq_27_l2092_209259


namespace NUMINAMATH_GPT_rhombus_diagonal_difference_l2092_209241

theorem rhombus_diagonal_difference (a d : ℝ) (h_a_pos : a > 0) (h_d_pos : d > 0):
  (∃ (e f : ℝ), e > f ∧ e - f = d ∧ a^2 = (e/2)^2 + (f/2)^2) ↔ d < 2 * a :=
sorry

end NUMINAMATH_GPT_rhombus_diagonal_difference_l2092_209241


namespace NUMINAMATH_GPT_equation_of_plane_l2092_209286

-- Definitions based on conditions
def line_equation (A B C x y : ℝ) : Prop :=
  A * x + B * y + C = 0

def A_B_nonzero (A B : ℝ) : Prop :=
  A ^ 2 + B ^ 2 ≠ 0

-- Statement for the problem
noncomputable def plane_equation (A B C D x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem equation_of_plane (A B C D : ℝ) :
  (A ^ 2 + B ^ 2 + C ^ 2 ≠ 0) → (∀ x y z : ℝ, plane_equation A B C D x y z) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_plane_l2092_209286


namespace NUMINAMATH_GPT_front_view_correct_l2092_209204

-- Define the number of blocks in each column
def Blocks_Column_A : Nat := 3
def Blocks_Column_B : Nat := 5
def Blocks_Column_C : Nat := 2
def Blocks_Column_D : Nat := 4

-- Define the front view representation
def front_view : List Nat := [3, 5, 2, 4]

-- Statement to be proved
theorem front_view_correct :
  [Blocks_Column_A, Blocks_Column_B, Blocks_Column_C, Blocks_Column_D] = front_view :=
by
  sorry

end NUMINAMATH_GPT_front_view_correct_l2092_209204


namespace NUMINAMATH_GPT_geometric_sequence_a5_l2092_209256

theorem geometric_sequence_a5 (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = a n * r) (h2 : a 2 * a 8 = 4) : a 5 = 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a5_l2092_209256


namespace NUMINAMATH_GPT_sum_eq_twenty_x_l2092_209230

variable {R : Type*} [CommRing R] (x y z : R)

theorem sum_eq_twenty_x (h1 : y = 3 * x) (h2 : z = 3 * y) : 2 * x + 3 * y + z = 20 * x := by
  sorry

end NUMINAMATH_GPT_sum_eq_twenty_x_l2092_209230


namespace NUMINAMATH_GPT_candy_problem_l2092_209279

-- Define conditions and the statement
theorem candy_problem (K : ℕ) (h1 : 49 = K + 3 * K + 8 + 6 + 10 + 5) : K = 5 :=
sorry

end NUMINAMATH_GPT_candy_problem_l2092_209279


namespace NUMINAMATH_GPT_sum_of_k_values_l2092_209288

-- Conditions
def P (x : ℝ) : ℝ := x^2 - 4 * x + 3
def Q (x k : ℝ) : ℝ := x^2 - 6 * x + k

-- Statement of the mathematical problem
theorem sum_of_k_values (k1 k2 : ℝ) (h1 : P 1 = 0) (h2 : P 3 = 0) 
  (h3 : Q 1 k1 = 0) (h4 : Q 3 k2 = 0) : k1 + k2 = 14 := 
by
  -- Here we would proceed with the proof steps corresponding to the solution
  sorry

end NUMINAMATH_GPT_sum_of_k_values_l2092_209288


namespace NUMINAMATH_GPT_solve_for_A_l2092_209252

variable (a b : ℝ) 

theorem solve_for_A (A : ℝ) (h : (5 * a + 3 * b)^2 = (5 * a - 3 * b)^2 + A) : 
  A = 60 * a * b := by
  sorry

end NUMINAMATH_GPT_solve_for_A_l2092_209252


namespace NUMINAMATH_GPT_calculate_fraction_l2092_209267

def x : ℚ := 2 / 3
def y : ℚ := 8 / 10

theorem calculate_fraction :
  (6 * x + 10 * y) / (60 * x * y) = 3 / 8 := by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l2092_209267


namespace NUMINAMATH_GPT_number_of_boys_l2092_209232

theorem number_of_boys 
    (B : ℕ) 
    (total_boys_sticks : ℕ := 15 * B)
    (total_girls_sticks : ℕ := 12 * 12)
    (sticks_relation : total_girls_sticks = total_boys_sticks - 6) : 
    B = 10 :=
by
    sorry

end NUMINAMATH_GPT_number_of_boys_l2092_209232


namespace NUMINAMATH_GPT_basketball_probability_l2092_209228

-- Define the probabilities of A and B making a shot
def prob_A : ℝ := 0.4
def prob_B : ℝ := 0.6

-- Define the probability that both miss their shots in one round
def prob_miss_one_round : ℝ := (1 - prob_A) * (1 - prob_B)

-- Define the probability that A takes k shots to make a basket
noncomputable def P_xi (k : ℕ) : ℝ := (prob_miss_one_round)^(k-1) * prob_A

-- State the theorem
theorem basketball_probability (k : ℕ) : 
  P_xi k = 0.24^(k-1) * 0.4 :=
by
  unfold P_xi
  unfold prob_miss_one_round
  sorry

end NUMINAMATH_GPT_basketball_probability_l2092_209228


namespace NUMINAMATH_GPT_probability_neither_snow_nor_rain_in_5_days_l2092_209264

def probability_no_snow (p_snow : ℚ) : ℚ := 1 - p_snow
def probability_no_rain (p_rain : ℚ) : ℚ := 1 - p_rain
def probability_no_snow_and_no_rain (p_no_snow p_no_rain : ℚ) : ℚ := p_no_snow * p_no_rain
def probability_no_snow_and_no_rain_5_days (p : ℚ) : ℚ := p ^ 5

theorem probability_neither_snow_nor_rain_in_5_days
    (p_snow : ℚ) (p_rain : ℚ)
    (h1 : p_snow = 2/3) (h2 : p_rain = 1/2) :
    probability_no_snow_and_no_rain_5_days (probability_no_snow_and_no_rain (probability_no_snow p_snow) (probability_no_rain p_rain)) = 1/7776 := by
  sorry

end NUMINAMATH_GPT_probability_neither_snow_nor_rain_in_5_days_l2092_209264


namespace NUMINAMATH_GPT_production_relationship_l2092_209250

noncomputable def production_function (a : ℕ) (p : ℝ) (x : ℕ) : ℝ := a * (1 + p / 100)^x

theorem production_relationship (a : ℕ) (p : ℝ) (m : ℕ) (x : ℕ) (hx : 0 ≤ x ∧ x ≤ m) :
  production_function a p x = a * (1 + p / 100)^x := by
  sorry

end NUMINAMATH_GPT_production_relationship_l2092_209250


namespace NUMINAMATH_GPT_kevin_trip_distance_l2092_209235

theorem kevin_trip_distance :
  let D := 600
  (∃ T : ℕ, D = 50 * T ∧ D = 75 * (T - 4)) := 
sorry

end NUMINAMATH_GPT_kevin_trip_distance_l2092_209235


namespace NUMINAMATH_GPT_main_theorem_l2092_209292

noncomputable def problem_statement : Prop :=
  ∀ x : ℂ, (x ≠ -2) →
  ((15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 48) ↔
  (x = 12 + 2 * Real.sqrt 38 ∨ x = 12 - 2 * Real.sqrt 38 ∨
  x = -1/2 + Complex.I * Real.sqrt 95 / 2 ∨
  x = -1/2 - Complex.I * Real.sqrt 95 / 2)

-- Provide the main statement without the proof
theorem main_theorem : problem_statement := sorry

end NUMINAMATH_GPT_main_theorem_l2092_209292


namespace NUMINAMATH_GPT_scout_weekend_earnings_l2092_209270

-- Define the constants and conditions
def base_pay : ℝ := 10.0
def tip_per_delivery : ℝ := 5.0
def saturday_hours : ℝ := 4.0
def sunday_hours : ℝ := 5.0
def saturday_deliveries : ℝ := 5.0
def sunday_deliveries : ℝ := 8.0

-- Calculate total hours worked
def total_hours : ℝ := saturday_hours + sunday_hours

-- Calculate base pay for the weekend
def total_base_pay : ℝ := total_hours * base_pay

-- Calculate total number of deliveries
def total_deliveries : ℝ := saturday_deliveries + sunday_deliveries

-- Calculate total earnings from tips
def total_tips : ℝ := total_deliveries * tip_per_delivery

-- Calculate total earnings
def total_earnings : ℝ := total_base_pay + total_tips

-- Theorem to prove the total earnings is $155.00
theorem scout_weekend_earnings : total_earnings = 155.0 := by
  sorry

end NUMINAMATH_GPT_scout_weekend_earnings_l2092_209270


namespace NUMINAMATH_GPT_yearly_exports_calculation_l2092_209281

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

end NUMINAMATH_GPT_yearly_exports_calculation_l2092_209281


namespace NUMINAMATH_GPT_team_A_win_probability_l2092_209289

theorem team_A_win_probability :
  let win_prob := (1 / 3 : ℝ)
  let team_A_lead := 2
  let total_sets := 5
  let require_wins := 3
  let remaining_sets := total_sets - team_A_lead
  let prob_team_B_win_remaining := (1 - win_prob) ^ remaining_sets
  let prob_team_A_win := 1 - prob_team_B_win_remaining
  prob_team_A_win = 19 / 27 := by
    sorry

end NUMINAMATH_GPT_team_A_win_probability_l2092_209289


namespace NUMINAMATH_GPT_color_tv_cost_l2092_209243

theorem color_tv_cost (x : ℝ) (y : ℝ) (z : ℝ)
  (h1 : y = x * 1.4)
  (h2 : z = y * 0.8)
  (h3 : z = 360 + x) :
  x = 3000 :=
sorry

end NUMINAMATH_GPT_color_tv_cost_l2092_209243


namespace NUMINAMATH_GPT_green_tea_price_decrease_l2092_209275

def percentage_change (old_price new_price : ℚ) : ℚ :=
  ((new_price - old_price) / old_price) * 100

theorem green_tea_price_decrease
  (C : ℚ)
  (h1 : C > 0)
  (july_coffee_price : ℚ := 2 * C)
  (mixture_price : ℚ := 3.45)
  (july_green_tea_price : ℚ := 0.3)
  (old_green_tea_price : ℚ := C)
  (equal_mixture : ℚ := (1.5 * july_green_tea_price) + (1.5 * july_coffee_price)) :
  mixture_price = equal_mixture →
  percentage_change old_green_tea_price july_green_tea_price = -70 :=
by
  sorry

end NUMINAMATH_GPT_green_tea_price_decrease_l2092_209275


namespace NUMINAMATH_GPT_gcd_a_b_l2092_209226

def a := 130^2 + 250^2 + 360^2
def b := 129^2 + 249^2 + 361^2

theorem gcd_a_b : Int.gcd a b = 1 := 
by
  sorry

end NUMINAMATH_GPT_gcd_a_b_l2092_209226


namespace NUMINAMATH_GPT_find_largest_t_l2092_209220

theorem find_largest_t (t : ℝ) : 
  (15 * t^2 - 38 * t + 14) / (4 * t - 3) + 6 * t = 7 * t - 2 → t ≤ 1 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_find_largest_t_l2092_209220


namespace NUMINAMATH_GPT_reciprocal_of_neg_one_seventh_l2092_209238

theorem reciprocal_of_neg_one_seventh :
  (∃ x : ℚ, - (1 / 7) * x = 1) → (-7) * (- (1 / 7)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_one_seventh_l2092_209238


namespace NUMINAMATH_GPT_remainder_of_7n_div_4_l2092_209276

theorem remainder_of_7n_div_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
sorry

end NUMINAMATH_GPT_remainder_of_7n_div_4_l2092_209276


namespace NUMINAMATH_GPT_binary_operation_l2092_209221

theorem binary_operation : 
  let a := 0b11011
  let b := 0b1101
  let c := 0b1010
  let result := 0b110011101  
  ((a * b) - c) = result := by
  sorry

end NUMINAMATH_GPT_binary_operation_l2092_209221


namespace NUMINAMATH_GPT_distance_from_Beijing_to_Lanzhou_l2092_209236

-- Conditions
def distance_Beijing_Lanzhou_Lhasa : ℕ := 3985
def distance_Lanzhou_Lhasa : ℕ := 2054

-- Define the distance from Beijing to Lanzhou
def distance_Beijing_Lanzhou : ℕ := distance_Beijing_Lanzhou_Lhasa - distance_Lanzhou_Lhasa

-- Proof statement that given conditions imply the correct answer
theorem distance_from_Beijing_to_Lanzhou :
  distance_Beijing_Lanzhou = 1931 :=
by
  -- conditions and definitions are already given
  sorry

end NUMINAMATH_GPT_distance_from_Beijing_to_Lanzhou_l2092_209236


namespace NUMINAMATH_GPT_no_five_coin_combination_for_70_cents_l2092_209261

/-- Define the values of each coin type -/
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25

/-- Prove that it is not possible to achieve a total value of 70 cents with exactly five coins -/
theorem no_five_coin_combination_for_70_cents :
  ¬ ∃ a b c d e : ℕ, a + b + c + d + e = 5 ∧ a * penny + b * nickel + c * dime + d * quarter + e * quarter = 70 :=
sorry

end NUMINAMATH_GPT_no_five_coin_combination_for_70_cents_l2092_209261


namespace NUMINAMATH_GPT_china_nhsm_league_2021_zhejiang_p15_l2092_209246

variable (x y z : ℝ)

theorem china_nhsm_league_2021_zhejiang_p15 (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) : 
  (x ^ 4 + y ^ 2 * z ^ 2) / (x ^ (5 / 2) * (y + z)) + 
  (y ^ 4 + z ^ 2 * x ^ 2) / (y ^ (5 / 2) * (z + x)) + 
  (z ^ 4 + y ^ 2 * x ^ 2) / (z ^ (5 / 2) * (y + x)) ≥ 1 := 
sorry

end NUMINAMATH_GPT_china_nhsm_league_2021_zhejiang_p15_l2092_209246


namespace NUMINAMATH_GPT_bell_rings_before_geography_l2092_209249

def number_of_bell_rings : Nat :=
  let assembly_start := 1
  let assembly_end := 1
  let maths_start := 1
  let maths_end := 1
  let history_start := 1
  let history_end := 1
  let quiz_start := 1
  let quiz_end := 1
  let geography_start := 1
  assembly_start + assembly_end + maths_start + maths_end + 
  history_start + history_end + quiz_start + quiz_end + 
  geography_start

theorem bell_rings_before_geography : number_of_bell_rings = 9 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_bell_rings_before_geography_l2092_209249


namespace NUMINAMATH_GPT_alfred_bill_days_l2092_209265

-- Definitions based on conditions
def combined_work_rate := 1 / 24
def alfred_to_bill_ratio := 2 / 3

-- Theorem statement
theorem alfred_bill_days (A B : ℝ) (ha : A = alfred_to_bill_ratio * B) (hcombined : A + B = combined_work_rate) : 
  A = 1 / 60 ∧ B = 1 / 40 :=
by
  sorry

end NUMINAMATH_GPT_alfred_bill_days_l2092_209265


namespace NUMINAMATH_GPT_partI_solution_partII_solution_l2092_209217

-- Part (I)
theorem partI_solution (x : ℝ) (a : ℝ) (h : a = 5) : (|x + a| + |x - 2| > 9) ↔ (x < -6 ∨ x > 3) :=
by
  sorry

-- Part (II)
theorem partII_solution (a : ℝ) :
  (∀ x : ℝ, (|2*x - 1| ≤ 3) → (|x + a| + |x - 2| ≤ |x - 4|)) → (-1 ≤ a ∧ a ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_partI_solution_partII_solution_l2092_209217


namespace NUMINAMATH_GPT_find_four_digit_number_l2092_209255

theorem find_four_digit_number :
  ∃ A B C D : ℕ, 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧
    (1001 * A + 100 * B + 10 * C + A) = 182 * (10 * C + D) ∧
    (1000 * A + 100 * B + 10 * C + D) = 2916 :=
by 
  sorry

end NUMINAMATH_GPT_find_four_digit_number_l2092_209255


namespace NUMINAMATH_GPT_evaluate_expression_equals_three_plus_sqrt_three_l2092_209219

noncomputable def tan_sixty_squared_plus_one := Real.tan (60 * Real.pi / 180) ^ 2 + 1
noncomputable def tan_fortyfive_minus_twocos_thirty := Real.tan (45 * Real.pi / 180) - 2 * Real.cos (30 * Real.pi / 180)
noncomputable def expression (x y : ℝ) : ℝ := (x - (2 * x * y - y ^ 2) / x) / ((x ^ 2 - y ^ 2) / (x ^ 2 + x * y))

theorem evaluate_expression_equals_three_plus_sqrt_three :
  expression tan_sixty_squared_plus_one tan_fortyfive_minus_twocos_thirty = 3 + Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_evaluate_expression_equals_three_plus_sqrt_three_l2092_209219


namespace NUMINAMATH_GPT_bromine_is_liquid_at_25C_1atm_l2092_209205

-- Definitions for the melting and boiling points
def melting_point (element : String) : Float :=
  match element with
  | "Br" => -7.2
  | "Kr" => -157.4 -- Not directly used, but included for completeness
  | "P" => 44.1 -- Not directly used, but included for completeness
  | "Xe" => -111.8 -- Not directly used, but included for completeness
  | _ => 0.0 -- default case; not used

def boiling_point (element : String) : Float :=
  match element with
  | "Br" => 58.8
  | "Kr" => -153.4
  | "P" => 280.5 -- Not directly used, but included for completeness
  | "Xe" => -108.1
  | _ => 0.0 -- default case; not used

-- Define the condition of the problem
def is_liquid_at (element : String) (temperature : Float) (pressure : Float) : Bool :=
  melting_point element < temperature ∧ temperature < boiling_point element

-- Goal statement
theorem bromine_is_liquid_at_25C_1atm : is_liquid_at "Br" 25 1 = true :=
by
  sorry

end NUMINAMATH_GPT_bromine_is_liquid_at_25C_1atm_l2092_209205


namespace NUMINAMATH_GPT_contrapositive_proof_l2092_209202

theorem contrapositive_proof (a b : ℕ) : (a = 1 ∧ b = 2) → (a + b = 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_contrapositive_proof_l2092_209202


namespace NUMINAMATH_GPT_a8_eq_64_l2092_209229

variable (S : ℕ → ℕ)
variable (a : ℕ → ℕ)

axiom a1_eq_2 : a 1 = 2
axiom S_recurrence : ∀ (n : ℕ), S (n + 1) = 2 * S n - 1

theorem a8_eq_64 : a 8 = 64 := 
by
sorry

end NUMINAMATH_GPT_a8_eq_64_l2092_209229


namespace NUMINAMATH_GPT_road_renovation_l2092_209285

theorem road_renovation (x : ℕ) (h : 200 / (x + 20) = 150 / x) : 
  x = 60 ∧ (x + 20) = 80 :=
by {
  sorry
}

end NUMINAMATH_GPT_road_renovation_l2092_209285


namespace NUMINAMATH_GPT_percentage_of_allowance_spent_l2092_209282

noncomputable def amount_spent : ℝ := 14
noncomputable def amount_left : ℝ := 26
noncomputable def total_allowance : ℝ := amount_spent + amount_left

theorem percentage_of_allowance_spent :
  ((amount_spent / total_allowance) * 100) = 35 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_of_allowance_spent_l2092_209282


namespace NUMINAMATH_GPT_complement_intersection_l2092_209284

-- Definitions to set the universal set and other sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {2, 4, 5}

-- Complement of M with respect to U
def CU_M : Set ℕ := U \ M

-- Intersection of (CU_M) and N
def intersection_CU_M_N : Set ℕ := CU_M ∩ N

-- The proof problem statement
theorem complement_intersection :
  intersection_CU_M_N = {2, 5} :=
sorry

end NUMINAMATH_GPT_complement_intersection_l2092_209284


namespace NUMINAMATH_GPT_print_time_l2092_209271

theorem print_time (P R: ℕ) (hR : R = 24) (hP : P = 360) (T : ℕ) : T = P / R → T = 15 := by
  intros h
  rw [hR, hP] at h
  exact h

end NUMINAMATH_GPT_print_time_l2092_209271


namespace NUMINAMATH_GPT_ball_probability_l2092_209224

theorem ball_probability:
  let total_balls := 120
  let red_balls := 12
  let purple_balls := 18
  let yellow_balls := 15
  let desired_probability := 33 / 1190
  let probability_red := red_balls / total_balls
  let probability_purple_or_yellow := (purple_balls + yellow_balls) / (total_balls - 1)
  (probability_red * probability_purple_or_yellow = desired_probability) :=
sorry

end NUMINAMATH_GPT_ball_probability_l2092_209224


namespace NUMINAMATH_GPT_seq1_general_formula_seq2_general_formula_l2092_209296

-- Sequence (1): Initial condition and recurrence relation
def seq1 (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) = a n + (2 * n - 1)

-- Proving the general formula for sequence (1)
theorem seq1_general_formula (a : ℕ → ℕ) (n : ℕ) (h : seq1 a) :
  a n = (n - 1) ^ 2 :=
sorry

-- Sequence (2): Initial condition and recurrence relation
def seq2 (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = 3 * a n

-- Proving the general formula for sequence (2)
theorem seq2_general_formula (a : ℕ → ℕ) (n : ℕ) (h : seq2 a) :
  a n = 3 ^ n :=
sorry

end NUMINAMATH_GPT_seq1_general_formula_seq2_general_formula_l2092_209296


namespace NUMINAMATH_GPT_probability_of_both_white_l2092_209233

namespace UrnProblem

-- Define the conditions
def firstUrnWhiteBalls : ℕ := 4
def firstUrnTotalBalls : ℕ := 10
def secondUrnWhiteBalls : ℕ := 7
def secondUrnTotalBalls : ℕ := 12

-- Define the probabilities of drawing a white ball from each urn
def P_A1 : ℚ := firstUrnWhiteBalls / firstUrnTotalBalls
def P_A2 : ℚ := secondUrnWhiteBalls / secondUrnTotalBalls

-- Define the combined probability of both events occurring
def P_A1_and_A2 : ℚ := P_A1 * P_A2

-- Theorem statement that checks the combined probability
theorem probability_of_both_white : P_A1_and_A2 = 7 / 30 := by
  sorry

end UrnProblem

end NUMINAMATH_GPT_probability_of_both_white_l2092_209233


namespace NUMINAMATH_GPT_bus_passenger_count_l2092_209277

-- Definition of the function f representing the number of passengers per trip
def passengers (n : ℕ) : ℕ :=
  120 - 2 * n

-- The total number of trips is 18 (from 9 AM to 5:30 PM inclusive)
def total_trips : ℕ := 18

-- Sum of passengers over all trips
def total_passengers : ℕ :=
  List.sum (List.map passengers (List.range total_trips))

-- Problem statement
theorem bus_passenger_count :
  total_passengers = 1854 :=
sorry

end NUMINAMATH_GPT_bus_passenger_count_l2092_209277


namespace NUMINAMATH_GPT_bonus_received_l2092_209295

-- Definitions based on the conditions
def total_sales (S : ℝ) : Prop :=
  S > 10000

def commission (S : ℝ) : ℝ :=
  0.09 * S

def excess_amount (S : ℝ) : ℝ :=
  S - 10000

def additional_commission (S : ℝ) : ℝ :=
  0.03 * (S - 10000)

def total_commission (S : ℝ) : ℝ :=
  commission S + additional_commission S

-- Given the conditions
axiom total_sales_commission : ∀ S : ℝ, total_sales S → total_commission S = 1380

-- The goal is to prove the bonus
theorem bonus_received (S : ℝ) (h : total_sales S) : additional_commission S = 120 := 
by 
  sorry

end NUMINAMATH_GPT_bonus_received_l2092_209295


namespace NUMINAMATH_GPT_total_food_in_10_days_l2092_209247

theorem total_food_in_10_days :
  (let ella_food_per_day := 20
   let days := 10
   let dog_food_ratio := 4
   let ella_total_food := ella_food_per_day * days
   let dog_total_food := dog_food_ratio * ella_total_food
   ella_total_food + dog_total_food = 1000) :=
by
  sorry

end NUMINAMATH_GPT_total_food_in_10_days_l2092_209247


namespace NUMINAMATH_GPT_system_of_inequalities_solution_set_quadratic_equation_when_m_is_2_l2092_209215

theorem system_of_inequalities_solution_set : 
  (∀ x : ℝ, (2 * x - 1 < 7) → (x + 1 > 2) ↔ (1 < x ∧ x < 4)) := 
by 
  sorry

theorem quadratic_equation_when_m_is_2 : 
  (∀ x : ℝ, x^2 - 2 * x - 2 = 0 ↔ (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3)) := 
by 
  sorry

end NUMINAMATH_GPT_system_of_inequalities_solution_set_quadratic_equation_when_m_is_2_l2092_209215


namespace NUMINAMATH_GPT_shaded_region_area_correct_l2092_209269

noncomputable def hexagon_side : ℝ := 4
noncomputable def major_axis : ℝ := 4
noncomputable def minor_axis : ℝ := 2

noncomputable def hexagon_area := (3 * Real.sqrt 3 / 2) * hexagon_side^2

noncomputable def semi_ellipse_area : ℝ :=
  (1 / 2) * Real.pi * major_axis * minor_axis

noncomputable def total_semi_ellipse_area := 4 * semi_ellipse_area 

noncomputable def shaded_region_area := hexagon_area - total_semi_ellipse_area

theorem shaded_region_area_correct : shaded_region_area = 48 * Real.sqrt 3 - 16 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_shaded_region_area_correct_l2092_209269


namespace NUMINAMATH_GPT_part1_part2_l2092_209206

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^3 + k * Real.log x
noncomputable def f' (x : ℝ) (k : ℝ) : ℝ := 3 * x^2 + k / x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f x k - f' x k + 9 / x

-- Part (1): Prove the monotonic intervals and extreme values for k = 6:
theorem part1 :
  (∀ x : ℝ, 0 < x ∧ x < 1 → g x 6 < g 1 6) ∧
  (∀ x : ℝ, 1 < x → g x 6 > g 1 6) ∧
  (g 1 6 = 1) := sorry

-- Part (2): Prove the given inequality for k ≥ -3:
theorem part2 (k : ℝ) (hk : k ≥ -3) (x1 x2 : ℝ) (hx1 : x1 ≥ 1) (hx2 : x2 ≥ 1) (h : x1 > x2) :
  (f' x1 k + f' x2 k) / 2 > (f x1 k - f x2 k) / (x1 - x2) := sorry

end NUMINAMATH_GPT_part1_part2_l2092_209206


namespace NUMINAMATH_GPT_abs_neg_one_third_l2092_209272

theorem abs_neg_one_third : abs (- (1 / 3 : ℚ)) = 1 / 3 := 
by sorry

end NUMINAMATH_GPT_abs_neg_one_third_l2092_209272


namespace NUMINAMATH_GPT_interest_rate_compound_interest_l2092_209278

theorem interest_rate_compound_interest :
  ∀ (P A : ℝ) (t n : ℕ), 
  P = 156.25 → A = 169 → t = 2 → n = 1 → 
  (∃ r : ℝ, A = P * (1 + r / n) ^ (n * t) ∧ r * 100 = 4) :=
by
  intros P A t n hP hA ht hn
  use 0.04
  rw [hP, hA, ht, hn]
  sorry

end NUMINAMATH_GPT_interest_rate_compound_interest_l2092_209278


namespace NUMINAMATH_GPT_find_a6_l2092_209297

variable {a : ℕ → ℝ} -- Sequence a is indexed by natural numbers and the terms are real numbers.

-- Conditions
def a_is_geom_seq (a : ℕ → ℝ) := ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q)
def a1_eq_4 (a : ℕ → ℝ) := a 1 = 4
def a3_eq_a2_mul_a4 (a : ℕ → ℝ) := a 3 = a 2 * a 4

theorem find_a6 (a : ℕ → ℝ) 
  (h1 : a_is_geom_seq a)
  (h2 : a1_eq_4 a)
  (h3 : a3_eq_a2_mul_a4 a) : 
  a 6 = 1 / 8 ∨ a 6 = - (1 / 8) := 
by 
  sorry

end NUMINAMATH_GPT_find_a6_l2092_209297


namespace NUMINAMATH_GPT_ratio_of_speeds_l2092_209216

variable (a b : ℝ)

theorem ratio_of_speeds (h1 : b = 1 / 60) (h2 : a + b = 1 / 12) : a / b = 4 := 
sorry

end NUMINAMATH_GPT_ratio_of_speeds_l2092_209216


namespace NUMINAMATH_GPT_total_cost_kept_l2092_209260

def prices_all : List ℕ := [15, 18, 20, 15, 25, 30, 20, 17, 22, 23, 29]
def prices_returned : List ℕ := [20, 25, 30, 22, 23, 29]

def total_cost (prices : List ℕ) : ℕ :=
  prices.foldl (· + ·) 0

theorem total_cost_kept :
  total_cost prices_all - total_cost prices_returned = 85 :=
by
  -- The proof steps go here
  sorry

end NUMINAMATH_GPT_total_cost_kept_l2092_209260


namespace NUMINAMATH_GPT_geometric_sequence_expression_l2092_209208

theorem geometric_sequence_expression (a : ℕ → ℝ) (q : ℝ) (h_q : q = 4)
  (h_geom : ∀ n, a (n + 1) = q * a n) (h_sum : a 0 + a 1 + a 2 = 21) :
  ∀ n, a n = 4 ^ n :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_expression_l2092_209208


namespace NUMINAMATH_GPT_eliza_tom_difference_l2092_209207

theorem eliza_tom_difference (q : ℕ) : 
  let eliza_quarters := 7 * q + 3
  let tom_quarters := 2 * q + 8
  let quarter_difference := (7 * q + 3) - (2 * q + 8)
  let nickel_value := 5
  let groups_of_5 := quarter_difference / 5
  let difference_in_cents := nickel_value * groups_of_5
  difference_in_cents = 5 * (q - 1) := by
  sorry

end NUMINAMATH_GPT_eliza_tom_difference_l2092_209207


namespace NUMINAMATH_GPT_alcohol_solution_volume_l2092_209239

theorem alcohol_solution_volume (V : ℝ) (h1 : 0.42 * V = 0.33 * (V + 3)) : V = 11 :=
by
  sorry

end NUMINAMATH_GPT_alcohol_solution_volume_l2092_209239


namespace NUMINAMATH_GPT_points_on_line_l2092_209280

theorem points_on_line (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_points_on_line_l2092_209280


namespace NUMINAMATH_GPT_sum_of_products_of_roots_l2092_209290

noncomputable def poly : Polynomial ℝ := 5 * Polynomial.X^3 - 10 * Polynomial.X^2 + 17 * Polynomial.X - 7

theorem sum_of_products_of_roots :
  (∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ poly.eval p = 0 ∧ poly.eval q = 0 ∧ poly.eval r = 0) →
  (∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ ((p * q + p * r + q * r) = 17 / 5)) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_products_of_roots_l2092_209290


namespace NUMINAMATH_GPT_polar_to_cartesian_l2092_209268

theorem polar_to_cartesian (θ : ℝ) (ρ : ℝ) (x y : ℝ) :
  (ρ = 2 * Real.sin θ + 4 * Real.cos θ) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (x - 8)^2 + (y - 2)^2 = 68 :=
by
  intros hρ hx hy
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_l2092_209268


namespace NUMINAMATH_GPT_right_triangle_perimeter_l2092_209244

noncomputable def perimeter_of_right_triangle (x : ℝ) : ℝ :=
  let y := x + 15
  let c := Real.sqrt (x^2 + y^2)
  x + y + c

theorem right_triangle_perimeter
  (h₁ : ∀ a b : ℝ, a * b = 2 * 150)  -- The area condition
  (h₂ : ∀ a b : ℝ, b = a + 15)       -- One leg is 15 units longer than the other
  : perimeter_of_right_triangle 11.375 = 66.47 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_perimeter_l2092_209244


namespace NUMINAMATH_GPT_smallest_of_five_consecutive_even_sum_500_l2092_209263

theorem smallest_of_five_consecutive_even_sum_500 : 
  ∃ (n : Int), (n - 4, n - 2, n, n + 2, n + 4).1 = 96 ∧ 
  ((n - 4) + (n - 2) + n + (n + 2) + (n + 4) = 500) :=
by
  sorry

end NUMINAMATH_GPT_smallest_of_five_consecutive_even_sum_500_l2092_209263


namespace NUMINAMATH_GPT_intersection_when_a_eq_4_range_for_A_subset_B_l2092_209237

-- Define the conditions
def setA : Set ℝ := { x | (1 - x) / (x - 7) > 0 }
def setB (a : ℝ) : Set ℝ := { x | x^2 - 2 * x - a^2 - 2 * a < 0 }

-- First proof goal: When a = 4, find A ∩ B
theorem intersection_when_a_eq_4 :
  setA ∩ (setB 4) = { x : ℝ | 1 < x ∧ x < 6 } :=
sorry

-- Second proof goal: Find the range for a such that A ⊆ B
theorem range_for_A_subset_B :
  { a : ℝ | setA ⊆ setB a } = { a : ℝ | a ≤ -7 ∨ a ≥ 5 } :=
sorry

end NUMINAMATH_GPT_intersection_when_a_eq_4_range_for_A_subset_B_l2092_209237


namespace NUMINAMATH_GPT_largest_multiple_negation_greater_than_neg150_l2092_209209

theorem largest_multiple_negation_greater_than_neg150 (n : ℤ) (h₁ : n % 6 = 0) (h₂ : -n > -150) : n = 144 := 
sorry

end NUMINAMATH_GPT_largest_multiple_negation_greater_than_neg150_l2092_209209


namespace NUMINAMATH_GPT_Peter_drew_more_l2092_209291

theorem Peter_drew_more :
  ∃ (P : ℕ), 5 + P + (P + 20) = 41 ∧ (P - 5 = 3) :=
sorry

end NUMINAMATH_GPT_Peter_drew_more_l2092_209291


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2092_209283

theorem solution_set_of_inequality (x : ℝ) : (1 / |x - 1| ≥ 1) ↔ (0 ≤ x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2092_209283


namespace NUMINAMATH_GPT_total_brushing_time_in_hours_l2092_209248

-- Define the conditions as Lean definitions
def brushing_duration : ℕ := 2   -- 2 minutes per brushing session
def brushing_times_per_day : ℕ := 3  -- brushes 3 times a day
def days : ℕ := 30  -- for 30 days

-- Define the calculation of total brushing time in hours
theorem total_brushing_time_in_hours : (brushing_duration * brushing_times_per_day * days) / 60 = 3 := 
by 
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_total_brushing_time_in_hours_l2092_209248


namespace NUMINAMATH_GPT_nested_geometric_sum_l2092_209287

theorem nested_geometric_sum :
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4))))))))) = 1398100 :=
by
  sorry

end NUMINAMATH_GPT_nested_geometric_sum_l2092_209287


namespace NUMINAMATH_GPT_min_value_expression_l2092_209293

theorem min_value_expression (a b : ℝ) (h1 : a > 1) (h2 : b > 0) (h3 : a + b = 2) :
  (∃ x : ℝ, x = (1 / (a - 1)) + (1 / (2 * b)) ∧ x ≥ (3 / 2 + Real.sqrt 2)) :=
sorry

end NUMINAMATH_GPT_min_value_expression_l2092_209293


namespace NUMINAMATH_GPT_expand_and_simplify_l2092_209213

theorem expand_and_simplify :
  (x : ℝ) → (x^2 - 3 * x + 3) * (x^2 + 3 * x + 3) = x^4 - 3 * x^2 + 9 :=
by 
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l2092_209213


namespace NUMINAMATH_GPT_bread_cost_l2092_209299

theorem bread_cost {packs_meat packs_cheese sandwiches : ℕ} 
  (cost_meat cost_cheese cost_sandwich coupon_meat coupon_cheese total_cost : ℝ) 
  (h_meat_cost : cost_meat = 5.00) 
  (h_cheese_cost : cost_cheese = 4.00)
  (h_coupon_meat : coupon_meat = 1.00)
  (h_coupon_cheese : coupon_cheese = 1.00)
  (h_cost_sandwich : cost_sandwich = 2.00)
  (h_packs_meat : packs_meat = 2)
  (h_packs_cheese : packs_cheese = 2)
  (h_sandwiches : sandwiches = 10)
  (h_total_revenue : total_cost = sandwiches * cost_sandwich) :
  ∃ (bread_cost : ℝ), bread_cost = total_cost - ((packs_meat * cost_meat - coupon_meat) + (packs_cheese * cost_cheese - coupon_cheese)) :=
sorry

end NUMINAMATH_GPT_bread_cost_l2092_209299


namespace NUMINAMATH_GPT_proposition_C_correct_l2092_209266

theorem proposition_C_correct (a b c : ℝ) (h : a * c ^ 2 > b * c ^ 2) : a > b :=
sorry

end NUMINAMATH_GPT_proposition_C_correct_l2092_209266


namespace NUMINAMATH_GPT_triangle_isosceles_or_right_l2092_209222

theorem triangle_isosceles_or_right (a b c : ℝ) (A B C : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (triangle_abc : A + B + C = 180)
  (opposite_sides : ∀ {x y}, x ≠ y → x + y < 180) 
  (condition : a * Real.cos A = b * Real.cos B) :
  (A = B ∨ A + B = 90) :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_isosceles_or_right_l2092_209222


namespace NUMINAMATH_GPT_mode_and_median_of_data_set_l2092_209225

def data_set : List ℕ := [3, 5, 4, 6, 3, 3, 4]

noncomputable def mode_of_data_set : ℕ :=
  sorry  -- The mode calculation goes here (implementation is skipped)

noncomputable def median_of_data_set : ℕ :=
  sorry  -- The median calculation goes here (implementation is skipped)

theorem mode_and_median_of_data_set :
  mode_of_data_set = 3 ∧ median_of_data_set = 4 :=
  by
    sorry  -- Proof goes here

end NUMINAMATH_GPT_mode_and_median_of_data_set_l2092_209225


namespace NUMINAMATH_GPT_combined_perimeter_two_right_triangles_l2092_209201

theorem combined_perimeter_two_right_triangles :
  ∀ (h1 h2 : ℝ),
    (h1^2 = 15^2 + 20^2) ∧
    (h2^2 = 9^2 + 12^2) ∧
    (h1 = h2) →
    (15 + 20 + h1) + (9 + 12 + h2) = 106 := by
  sorry

end NUMINAMATH_GPT_combined_perimeter_two_right_triangles_l2092_209201


namespace NUMINAMATH_GPT_pyramid_z_value_l2092_209253

-- Define the conditions and the proof problem
theorem pyramid_z_value {z x y : ℕ} :
  (x = z * y) →
  (8 = z * x) →
  (40 = x * y) →
  (10 = y * x) →
  z = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_pyramid_z_value_l2092_209253


namespace NUMINAMATH_GPT_range_of_a_l2092_209223

theorem range_of_a
  (x0 : ℝ) (a : ℝ)
  (hx0 : x0 > 1)
  (hineq : (x0 + 1) * Real.log x0 < a * (x0 - 1)) :
  a > 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2092_209223


namespace NUMINAMATH_GPT_product_equation_l2092_209203

theorem product_equation (a b : ℝ) (h1 : ∀ (a b : ℝ), 0.2 * b = 0.9 * a - b) : 
  0.9 * a - b = 0.2 * b :=
by
  sorry

end NUMINAMATH_GPT_product_equation_l2092_209203


namespace NUMINAMATH_GPT_find_f_2000_l2092_209251

noncomputable def f : ℝ → ℝ := sorry

axiom f_property1 : ∀ (x y : ℝ), f (x + y) = f (x * y)
axiom f_property2 : f (-1/2) = -1/2

theorem find_f_2000 : f 2000 = -1/2 := 
sorry

end NUMINAMATH_GPT_find_f_2000_l2092_209251


namespace NUMINAMATH_GPT_paint_grid_l2092_209234

theorem paint_grid (paint : Fin 3 × Fin 3 → Bool) (no_adjacent : ∀ i j, (paint (i, j) = true) → (paint (i+1, j) = false) ∧ (paint (i-1, j) = false) ∧ (paint (i, j+1) = false) ∧ (paint (i, j-1) = false)) : 
  ∃! (count : ℕ), count = 8 :=
sorry

end NUMINAMATH_GPT_paint_grid_l2092_209234


namespace NUMINAMATH_GPT_range_of_m_l2092_209227

open Real

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 + 2 * x + m^2 > 0) ↔ -1 ≤ m ∧ m ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2092_209227


namespace NUMINAMATH_GPT_triangle_area_correct_l2092_209273

open Real

def triangle_area (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  0.5 * abs (v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v1.2 - v1.2 * v2.1 - v2.2 * v3.1 - v3.2 * v1.1)

theorem triangle_area_correct :
  triangle_area (4, 6) (-4, 6) (0, 2) = 16 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_correct_l2092_209273


namespace NUMINAMATH_GPT_inflation_over_two_years_real_interest_rate_l2092_209231

-- Definitions for conditions
def annual_inflation_rate : ℝ := 0.025
def nominal_interest_rate : ℝ := 0.06

-- Lean statement for the first problem: Inflation over two years
theorem inflation_over_two_years :
  (1 + annual_inflation_rate) ^ 2 - 1 = 0.050625 := 
by sorry

-- Lean statement for the second problem: Real interest rate after inflation
theorem real_interest_rate (inflation_rate_two_years : ℝ)
  (h_inflation_rate : inflation_rate_two_years = 0.050625) :
  (nominal_interest_rate + 1) ^ 2 / (1 + inflation_rate_two_years) - 1 = 0.069459 :=
by sorry

end NUMINAMATH_GPT_inflation_over_two_years_real_interest_rate_l2092_209231


namespace NUMINAMATH_GPT_sandy_saved_percentage_last_year_l2092_209240

noncomputable def sandys_saved_percentage (S : ℝ) (P : ℝ) : ℝ :=
  (P / 100) * S

noncomputable def salary_with_10_percent_more (S : ℝ) : ℝ :=
  1.1 * S

noncomputable def amount_saved_this_year (S : ℝ) : ℝ :=
  0.15 * (salary_with_10_percent_more S)

noncomputable def amount_saved_this_year_compare_last_year (S : ℝ) (P : ℝ) : Prop :=
  amount_saved_this_year S = 1.65 * sandys_saved_percentage S P

theorem sandy_saved_percentage_last_year (S : ℝ) (P : ℝ) :
  amount_saved_this_year_compare_last_year S P → P = 10 :=
by
  sorry

end NUMINAMATH_GPT_sandy_saved_percentage_last_year_l2092_209240


namespace NUMINAMATH_GPT_work_days_l2092_209294

/-- A needs 20 days to complete the work alone. B needs 10 days to complete the work alone.
    The total work must be completed in 12 days. We need to find how many days B must work 
    before A continues, such that the total work equals the full task. -/
theorem work_days (x : ℝ) (h0 : 0 ≤ x ∧ x ≤ 12) (h1 : 1 / 10 * x + 1 / 20 * (12 - x) = 1) : x = 8 := by
  sorry

end NUMINAMATH_GPT_work_days_l2092_209294


namespace NUMINAMATH_GPT_a_n_formula_b_n_geometric_sequence_l2092_209214

noncomputable def a_n (n : ℕ) : ℝ := 3 * n - 1

def S_n (n : ℕ) : ℝ := sorry -- Sum of the first n terms of b_n

def b_n (n : ℕ) : ℝ := 2 - 2 * S_n n

theorem a_n_formula (n : ℕ) : a_n n = 3 * n - 1 :=
by { sorry }

theorem b_n_geometric_sequence : ∀ n ≥ 2, b_n n / b_n (n - 1) = 1 / 3 :=
by { sorry }

end NUMINAMATH_GPT_a_n_formula_b_n_geometric_sequence_l2092_209214


namespace NUMINAMATH_GPT_calvin_buys_chips_days_per_week_l2092_209212

-- Define the constants based on the problem conditions
def cost_per_pack : ℝ := 0.50
def total_amount_spent : ℝ := 10
def number_of_weeks : ℕ := 4

-- Define the proof statement
theorem calvin_buys_chips_days_per_week : 
  (total_amount_spent / cost_per_pack) / number_of_weeks = 5 := 
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_calvin_buys_chips_days_per_week_l2092_209212


namespace NUMINAMATH_GPT_distinct_triangles_count_l2092_209258

theorem distinct_triangles_count (n : ℕ) (hn : 0 < n) : 
  (∃ triangles_count, triangles_count = ⌊((n+1)^2 : ℝ)/4⌋) :=
sorry

end NUMINAMATH_GPT_distinct_triangles_count_l2092_209258


namespace NUMINAMATH_GPT_oranges_in_each_box_l2092_209298

theorem oranges_in_each_box (O B : ℕ) (h1 : O = 24) (h2 : B = 3) :
  O / B = 8 :=
by
  sorry

end NUMINAMATH_GPT_oranges_in_each_box_l2092_209298


namespace NUMINAMATH_GPT_sum_of_final_two_numbers_l2092_209262

theorem sum_of_final_two_numbers (a b S : ℝ) (h : a + b = S) : 
  2 * (a + 3) + 2 * (b + 3) = 2 * S + 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_final_two_numbers_l2092_209262


namespace NUMINAMATH_GPT_total_shells_is_correct_l2092_209242

def morning_shells : Nat := 292
def afternoon_shells : Nat := 324
def total_shells : Nat := morning_shells + afternoon_shells

theorem total_shells_is_correct : total_shells = 616 :=
by
  sorry

end NUMINAMATH_GPT_total_shells_is_correct_l2092_209242


namespace NUMINAMATH_GPT_num_pairs_eq_seven_l2092_209211

theorem num_pairs_eq_seven :
  ∃ S : Finset (Nat × Nat), 
    (∀ (a b : Nat), (a, b) ∈ S ↔ (0 < a ∧ 0 < b ∧ a + b ≤ 100 ∧ (a + 1 / b) / (1 / a + b) = 13)) ∧
    S.card = 7 :=
sorry

end NUMINAMATH_GPT_num_pairs_eq_seven_l2092_209211


namespace NUMINAMATH_GPT_quadratic_factor_conditions_l2092_209274

theorem quadratic_factor_conditions (b : ℤ) :
  (∃ m n p q : ℤ, m * p = 15 ∧ n * q = 75 ∧ mq + np = b) → ∃ (c : ℤ), b = c :=
sorry

end NUMINAMATH_GPT_quadratic_factor_conditions_l2092_209274


namespace NUMINAMATH_GPT_determine_a_l2092_209257
open Set

-- Given Condition Definitions
def U : Set ℕ := {1, 3, 5, 7}
def M (a : ℤ) : Set ℕ := {1, Int.natAbs (a - 5)} -- using ℤ for a and natAbs to get |a - 5|

-- Problem statement
theorem determine_a (a : ℤ) (hM_subset_U : M a ⊆ U) (h_complement : U \ M a = {5, 7}) : a = 2 ∨ a = 8 :=
by sorry

end NUMINAMATH_GPT_determine_a_l2092_209257


namespace NUMINAMATH_GPT_value_of_x_l2092_209200

theorem value_of_x :
  ∀ (x : ℕ), 
    x = 225 + 2 * 15 * 9 + 81 → 
    x = 576 := 
by
  intro x h
  sorry

end NUMINAMATH_GPT_value_of_x_l2092_209200


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2092_209218

theorem solution_set_of_inequality :
  { x : ℝ | |1 - 2 * x| < 3 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2092_209218
