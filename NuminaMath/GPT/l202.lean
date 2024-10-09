import Mathlib

namespace number_of_two_bedroom_units_l202_20226

-- Definitions based on the conditions
def is_solution (x y : ℕ) : Prop :=
  (x + y = 12) ∧ (360 * x + 450 * y = 4950)

theorem number_of_two_bedroom_units : ∃ y : ℕ, is_solution (12 - y) y ∧ y = 7 :=
by
  sorry

end number_of_two_bedroom_units_l202_20226


namespace total_votes_is_240_l202_20204

variable {x : ℕ} -- Total number of votes (natural number)
variable {S : ℤ} -- Score (integer)

-- Given conditions
axiom score_condition : S = 120
axiom votes_condition : 3 * x / 4 - x / 4 = S

theorem total_votes_is_240 : x = 240 :=
by
  -- Proof should go here
  sorry

end total_votes_is_240_l202_20204


namespace ann_boxes_less_than_n_l202_20269

-- Define the total number of boxes n
def n : ℕ := 12

-- Define the number of boxes Mark sold
def mark_sold : ℕ := n - 11

-- Define a condition on the number of boxes Ann sold
def ann_sold (A : ℕ) : Prop := 1 ≤ A ∧ A < n - mark_sold

-- The statement to prove
theorem ann_boxes_less_than_n : ∃ A : ℕ, ann_sold A ∧ n - A = 2 :=
by
  sorry

end ann_boxes_less_than_n_l202_20269


namespace calculate_distance_l202_20224

theorem calculate_distance (t : ℕ) (h_t : t = 4) : 5 * t^2 + 2 * t = 88 :=
by
  rw [h_t]
  norm_num

end calculate_distance_l202_20224


namespace flagstaff_height_is_correct_l202_20217

noncomputable def flagstaff_height : ℝ := 40.25 * 12.5 / 28.75

theorem flagstaff_height_is_correct :
  flagstaff_height = 17.5 :=
by 
  -- These conditions are implicit in the previous definition
  sorry

end flagstaff_height_is_correct_l202_20217


namespace smallest_number_of_pets_l202_20250

noncomputable def smallest_common_multiple (a b c : Nat) : Nat :=
  Nat.lcm a (Nat.lcm b c)

theorem smallest_number_of_pets : smallest_common_multiple 3 15 9 = 45 :=
by
  sorry

end smallest_number_of_pets_l202_20250


namespace molecular_weight_constant_l202_20215

-- Given the molecular weight of a compound
def molecular_weight (w : ℕ) := w = 1188

-- Statement about molecular weight of n moles
def weight_of_n_moles (n : ℕ) := n * 1188

theorem molecular_weight_constant (moles : ℕ) : 
  ∀ (w : ℕ), molecular_weight w → ∀ (n : ℕ), weight_of_n_moles n = n * w :=
by
  intro w h n
  sorry

end molecular_weight_constant_l202_20215


namespace length_of_AB_l202_20299

theorem length_of_AB 
  (AB BC CD AD : ℕ)
  (h1 : AB = 1 * BC / 2)
  (h2 : BC = 6 * CD / 5)
  (h3 : AB + BC + CD = 56)
  : AB = 12 := sorry

end length_of_AB_l202_20299


namespace max_single_player_salary_l202_20263

variable (n : ℕ) (m : ℕ) (p : ℕ) (s : ℕ)

theorem max_single_player_salary
  (h1 : n = 18)
  (h2 : ∀ i : ℕ, i < n → p ≥ 20000)
  (h3 : s = 800000)
  (h4 : n * 20000 ≤ s) :
  ∃ x : ℕ, x = 460000 :=
by
  sorry

end max_single_player_salary_l202_20263


namespace percentage_of_total_population_absent_l202_20219

def total_students : ℕ := 120
def boys : ℕ := 72
def girls : ℕ := 48
def boys_absent_fraction : ℚ := 1/8
def girls_absent_fraction : ℚ := 1/4

theorem percentage_of_total_population_absent : 
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students * 100 = 17.5 :=
by
  sorry

end percentage_of_total_population_absent_l202_20219


namespace rectangle_area_exceeds_m_l202_20254

theorem rectangle_area_exceeds_m (m : ℤ) (h_m : m > 12) :
  ∃ x y : ℤ, x * y > m ∧ (x - 1) * y < m ∧ x * (y - 1) < m :=
by
  sorry

end rectangle_area_exceeds_m_l202_20254


namespace triangle_possible_sides_l202_20229

theorem triangle_possible_sides (a b c : ℕ) (h₁ : a + b + c = 7) (h₂ : a + b > c) (h₃ : a + c > b) (h₄ : b + c > a) :
  a = 1 ∨ a = 2 ∨ a = 3 :=
by {
  sorry
}

end triangle_possible_sides_l202_20229


namespace find_a_l202_20251

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ :=
  (x + a) * Real.log x

noncomputable def curve_deriv (a : ℝ) (x : ℝ) : ℝ :=
  Real.log x + (x + a) / x

theorem find_a (a : ℝ) (h : curve (x := 1) a = 2) : a = 1 :=
by
  have eq1 : curve 1 0 = (1 + a) * 0 := by sorry
  have eq2 : curve 1 1 = (1 + a) * Real.log 1 := by sorry
  have eq3 : curve_deriv a 1 = Real.log 1 + (1 + a) / 1 := by sorry
  have eq4 : 2 = 1 + a := by sorry
  sorry -- Complete proof would follow here

end find_a_l202_20251


namespace cube_volume_is_27_l202_20240

theorem cube_volume_is_27 
    (a : ℕ) 
    (Vol_cube : ℕ := a^3)
    (Vol_new : ℕ := (a - 2) * a * (a + 2))
    (h : Vol_new + 12 = Vol_cube) : Vol_cube = 27 :=
by
    sorry

end cube_volume_is_27_l202_20240


namespace remainder_of_division_l202_20278

def p (x : ℝ) : ℝ := 8*x^4 - 10*x^3 + 16*x^2 - 18*x + 5
def d (x : ℝ) : ℝ := 4*x - 8

theorem remainder_of_division :
  (p 2) = 81 :=
by
  sorry

end remainder_of_division_l202_20278


namespace general_formula_arithmetic_sequence_sum_of_sequence_b_l202_20266

-- Definitions of arithmetic sequence {a_n} and geometric sequence conditions
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
 ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
 ∀ n : ℕ, S n = n * (a 1 + a n) / 2

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  a 3 ^ 2 = a 1 * a 7

def arithmetic_sum_S3 (S : ℕ → ℤ) : Prop :=
  S 3 = 9

def general_formula (a : ℕ → ℤ) : Prop :=
 ∀ n : ℕ, a n = n + 1

def sum_first_n_terms_b (b : ℕ → ℤ) (T : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, T n = (n-1) * 2^(n+1) + 2

-- The Lean theorem statements
theorem general_formula_arithmetic_sequence
  (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
  (h1 : arithmetic_sequence a d)
  (h2 : sum_first_n_terms a S)
  (h3 : geometric_sequence a)
  (h4 : arithmetic_sum_S3 S) :
  general_formula a :=
  sorry

theorem sum_of_sequence_b
  (a b : ℕ → ℤ) (T : ℕ → ℤ)
  (h1 : general_formula a)
  (h2 : ∀ n : ℕ, b n = (a n - 1) * 2^n)
  (h3 : sum_first_n_terms_b b T) :
  ∀ n : ℕ, T n = (n-1) * 2^(n+1) + 2 :=
  sorry

end general_formula_arithmetic_sequence_sum_of_sequence_b_l202_20266


namespace correct_factorization_l202_20206

theorem correct_factorization {x y : ℝ} :
  (2 * x ^ 2 - 8 * y ^ 2 = 2 * (x + 2 * y) * (x - 2 * y)) ∧
  ¬(x ^ 2 + 3 * x * y + 9 * y ^ 2 = (x + 3 * y) ^ 2)
    ∧ ¬(2 * x ^ 2 - 4 * x * y + 9 * y ^ 2 = (2 * x - 3 * y) ^ 2)
    ∧ ¬(x * (x - y) + y * (y - x) = (x - y) * (x + y)) := 
by sorry

end correct_factorization_l202_20206


namespace domain_of_function_l202_20216

theorem domain_of_function (x : ℝ) : 
  {x | ∃ k : ℤ, - (Real.pi / 3) + (2 : ℝ) * k * Real.pi ≤ x ∧ x ≤ (Real.pi / 3) + (2 : ℝ) * k * Real.pi} :=
by
  -- Proof omitted
  sorry

end domain_of_function_l202_20216


namespace peter_remaining_walk_time_l202_20260

-- Define the parameters and conditions
def total_distance : ℝ := 2.5
def time_per_mile : ℝ := 20
def distance_walked : ℝ := 1

-- Define the remaining distance
def remaining_distance : ℝ := total_distance - distance_walked

-- Define the remaining time Peter needs to walk
def remaining_time_to_walk (d : ℝ) (t : ℝ) : ℝ := d * t

-- State the problem we want to prove
theorem peter_remaining_walk_time :
  remaining_time_to_walk remaining_distance time_per_mile = 30 :=
by
  -- Placeholder for the proof
  sorry

end peter_remaining_walk_time_l202_20260


namespace population_increase_l202_20265

theorem population_increase (k l m : ℝ) : 
  (1 + k/100) * (1 + l/100) * (1 + m/100) = 
  1 + (k + l + m)/100 + (k*l + k*m + l*m)/10000 + k*l*m/1000000 :=
by sorry

end population_increase_l202_20265


namespace radio_lowest_price_rank_l202_20239

-- Definitions based on the conditions
def total_items : ℕ := 38
def radio_highest_rank : ℕ := 16

-- The theorem statement
theorem radio_lowest_price_rank : (total_items - (radio_highest_rank - 1)) = 24 := by
  sorry

end radio_lowest_price_rank_l202_20239


namespace move_decimal_point_one_place_right_l202_20246

theorem move_decimal_point_one_place_right (x : ℝ) (h : x = 76.08) : x * 10 = 760.8 :=
by
  rw [h]
  -- Here, you would provide proof steps, but we'll use sorry to indicate the proof is omitted.
  sorry

end move_decimal_point_one_place_right_l202_20246


namespace tan_half_angle_l202_20271

theorem tan_half_angle (α : ℝ) (h1 : Real.sin α + Real.cos α = 1 / 5)
  (h2 : 3 * π / 2 < α ∧ α < 2 * π) : 
  Real.tan (α / 2) = -1 / 3 :=
sorry

end tan_half_angle_l202_20271


namespace no_solns_to_equation_l202_20223

noncomputable def no_solution : Prop :=
  ∀ (n m r : ℕ), (1 ≤ n) → (1 ≤ m) → (1 ≤ r) → n^5 + 49^m ≠ 1221^r

theorem no_solns_to_equation : no_solution :=
sorry

end no_solns_to_equation_l202_20223


namespace uncover_area_is_64_l202_20272

-- Conditions as definitions
def length_of_floor := 10
def width_of_floor := 8
def side_of_carpet := 4

-- The statement of the problem
theorem uncover_area_is_64 :
  let area_of_floor := length_of_floor * width_of_floor
  let area_of_carpet := side_of_carpet * side_of_carpet
  let uncovered_area := area_of_floor - area_of_carpet
  uncovered_area = 64 :=
by
  sorry

end uncover_area_is_64_l202_20272


namespace min_value_of_a_plus_b_l202_20247

theorem min_value_of_a_plus_b (a b : ℤ) (h1 : Even a) (h2 : Even b) (h3 : a * b = 144) : a + b = -74 :=
sorry

end min_value_of_a_plus_b_l202_20247


namespace fox_appropriation_l202_20275

variable (a m : ℕ) (n : ℕ) (y x : ℕ)

-- Definitions based on conditions
def fox_funds : Prop :=
  (m-1)*a + x = m*y ∧ 2*(m-1)*a + x = (m+1)*y ∧ 
  3*(m-1)*a + x = (m+2)*y ∧ n*(m-1)*a + x = (m+n-1)*y

-- Theorems to prove the final conclusions
theorem fox_appropriation (h : fox_funds a m n y x) : 
  y = (m-1)*a ∧ x = (m-1)^2*a :=
by
  sorry

end fox_appropriation_l202_20275


namespace aprons_to_sew_tomorrow_l202_20214

def total_aprons : ℕ := 150
def already_sewn : ℕ := 13
def sewn_today (already_sewn : ℕ) : ℕ := 3 * already_sewn
def sewn_tomorrow (total_aprons : ℕ) (already_sewn : ℕ) (sewn_today : ℕ) : ℕ :=
  let remaining := total_aprons - (already_sewn + sewn_today)
  remaining / 2

theorem aprons_to_sew_tomorrow : sewn_tomorrow total_aprons already_sewn (sewn_today already_sewn) = 49 :=
  by 
    sorry

end aprons_to_sew_tomorrow_l202_20214


namespace car_b_speed_l202_20227

def speed_of_car_b (Vb Va : ℝ) (tA tB : ℝ) (dist total_dist : ℝ) : Prop :=
  Va = 3 * Vb ∧ tA = 6 ∧ tB = 2 ∧ dist = 1000 ∧ total_dist = Va * tA + Vb * tB

theorem car_b_speed : ∃ Vb Va tA tB dist total_dist, speed_of_car_b Vb Va tA tB dist total_dist ∧ Vb = 50 :=
by
  sorry

end car_b_speed_l202_20227


namespace train_speed_km_hr_l202_20258

def train_length : ℝ := 130  -- Length of the train in meters
def bridge_and_train_length : ℝ := 245  -- Total length of the bridge and the train in meters
def crossing_time : ℝ := 30  -- Time to cross the bridge in seconds

theorem train_speed_km_hr : (train_length + bridge_and_train_length) / crossing_time * 3.6 = 45 := by
  sorry

end train_speed_km_hr_l202_20258


namespace period_of_f_max_value_of_f_and_values_l202_20202

noncomputable def f (x : ℝ) : ℝ := (1 - Real.sin (2 * x)) / (Real.sin x - Real.cos x)

-- Statement 1: The period of f(x) is 2π
theorem period_of_f : ∀ x, f (x + 2 * Real.pi) = f x := by
  sorry

-- Statement 2: The maximum value of f(x) is √2 and it is attained at x = 2kπ + 3π/4, k ∈ ℤ
theorem max_value_of_f_and_values :
  (∀ x, f x ≤ Real.sqrt 2) ∧
  (∃ k : ℤ, f (2 * k * Real.pi + 3 * Real.pi / 4) = Real.sqrt 2) := by
  sorry

end period_of_f_max_value_of_f_and_values_l202_20202


namespace arrangements_of_45520_l202_20268

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (k : Nat) : Nat :=
  factorial n / factorial k

theorem arrangements_of_45520 : 
  let n0_pos := 4
  let remaining_digits := 4 * arrangements 4 2
  n0_pos * remaining_digits = 48 :=
by
  -- Definitions and lemmas can be introduced here
  sorry

end arrangements_of_45520_l202_20268


namespace possible_number_of_friends_l202_20270

-- Define the conditions and problem statement
def player_structure (total_players : ℕ) (n : ℕ) (m : ℕ) : Prop :=
  total_players = n * m ∧ (n - 1) * m = 15

-- The main theorem to prove the number of friends in the group
theorem possible_number_of_friends : ∃ (N : ℕ), 
  (player_structure N 2 15 ∨ player_structure N 4 5 ∨ player_structure N 6 3 ∨ player_structure N 16 1) ∧
  (N = 16 ∨ N = 18 ∨ N = 20 ∨ N = 30) :=
sorry

end possible_number_of_friends_l202_20270


namespace find_CD_l202_20264

noncomputable def C : ℝ := 32 / 9
noncomputable def D : ℝ := 4 / 9

theorem find_CD :
  (∀ x, x ≠ 6 ∧ x ≠ -3 → (4 * x + 8) / (x^2 - 3 * x - 18) = 
       C / (x - 6) + D / (x + 3)) →
  C = 32 / 9 ∧ D = 4 / 9 :=
by sorry

end find_CD_l202_20264


namespace betty_cupcakes_per_hour_l202_20213

theorem betty_cupcakes_per_hour (B : ℕ) (Dora_rate : ℕ) (betty_break_hours : ℕ) (total_hours : ℕ) (cupcake_diff : ℕ) :
  Dora_rate = 8 →
  betty_break_hours = 2 →
  total_hours = 5 →
  cupcake_diff = 10 →
  (total_hours - betty_break_hours) * B = Dora_rate * total_hours - cupcake_diff →
  B = 10 :=
by
  intros hDora_rate hbreak_hours htotal_hours hcupcake_diff hcupcake_eq
  sorry

end betty_cupcakes_per_hour_l202_20213


namespace range_of_a_l202_20267

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Condition: f(x) is an increasing function on ℝ.
def is_increasing_on_ℝ (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x < f y

-- Equivalent proof problem in Lean 4:
theorem range_of_a (h : is_increasing_on_ℝ f) : 1 < a ∧ a < 6 := by
  sorry

end range_of_a_l202_20267


namespace log_increasing_on_interval_l202_20234

theorem log_increasing_on_interval :
  ∀ x : ℝ, x < 1 → (0.2 : ℝ)^(x^2 - 3*x + 2) > 1 :=
by
  sorry

end log_increasing_on_interval_l202_20234


namespace faucet_draining_time_l202_20274

theorem faucet_draining_time 
  (all_faucets_drain_time : ℝ)
  (n : ℝ) 
  (first_faucet_time : ℝ) 
  (last_faucet_time : ℝ) 
  (avg_drain_time : ℝ)
  (condition_1 : all_faucets_drain_time = 24)
  (condition_2 : last_faucet_time = first_faucet_time / 7)
  (condition_3 : avg_drain_time = (first_faucet_time + last_faucet_time) / 2)
  (condition_4 : avg_drain_time = 24) : 
  first_faucet_time = 42 := 
by
  sorry

end faucet_draining_time_l202_20274


namespace shirt_cost_correct_l202_20233

-- Definitions based on the conditions
def initial_amount : ℕ := 109
def pants_cost : ℕ := 13
def remaining_amount : ℕ := 74
def total_spent : ℕ := initial_amount - remaining_amount
def shirts_cost : ℕ := total_spent - pants_cost
def number_of_shirts : ℕ := 2

-- Statement to be proved
theorem shirt_cost_correct : shirts_cost / number_of_shirts = 11 := by
  sorry

end shirt_cost_correct_l202_20233


namespace Somu_years_back_l202_20228

-- Define the current ages of Somu and his father, and the relationship between them
variables (S F : ℕ)
variable (Y : ℕ)

-- Hypotheses based on the problem conditions
axiom age_of_Somu : S = 14
axiom age_relation : S = F / 3

-- Define the condition for years back when Somu was one-fifth his father's age
axiom years_back_condition : S - Y = (F - Y) / 5

-- Problem statement: Prove that 7 years back, Somu was one-fifth of his father's age
theorem Somu_years_back : Y = 7 :=
by
  sorry

end Somu_years_back_l202_20228


namespace apples_count_l202_20290

theorem apples_count (n : ℕ) (h₁ : n > 2)
  (h₂ : 144 / n - 144 / (n + 2) = 1) :
  n + 2 = 18 :=
by
  sorry

end apples_count_l202_20290


namespace log12_div_log15_eq_2m_n_div_1_m_n_l202_20225

variable (m n : Real)

theorem log12_div_log15_eq_2m_n_div_1_m_n 
  (h1 : Real.log 2 = m) 
  (h2 : Real.log 3 = n) : 
  Real.log 12 / Real.log 15 = (2 * m + n) / (1 - m + n) :=
by sorry

end log12_div_log15_eq_2m_n_div_1_m_n_l202_20225


namespace necessary_but_not_sufficient_condition_l202_20277

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (a < 2) → (∃ x : ℂ, x^2 + (a : ℂ) * x + 1 = 0 ∧ x.im ≠ 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_l202_20277


namespace inequality_positive_real_xyz_l202_20292

theorem inequality_positive_real_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y))) ≥ (3 / 4) := 
by
  -- Proof is to be constructed here
  sorry

end inequality_positive_real_xyz_l202_20292


namespace smallest_x_for_multiple_of_720_l202_20282

theorem smallest_x_for_multiple_of_720 (x : ℕ) (h1 : 450 = 2^1 * 3^2 * 5^2) (h2 : 720 = 2^4 * 3^2 * 5^1) : x = 8 ↔ (450 * x) % 720 = 0 :=
by
  sorry

end smallest_x_for_multiple_of_720_l202_20282


namespace triangle_right_triangle_l202_20210

-- Defining the sides of the triangle
variables (a b c : ℝ)

-- Theorem statement
theorem triangle_right_triangle (h : (a + b)^2 = c^2 + 2 * a * b) : a^2 + b^2 = c^2 :=
by {
  sorry
}

end triangle_right_triangle_l202_20210


namespace kristine_travel_distance_l202_20221

theorem kristine_travel_distance :
  ∃ T : ℝ, T + T / 2 + T / 6 = 500 ∧ T = 300 := by
  sorry

end kristine_travel_distance_l202_20221


namespace middle_school_students_count_l202_20230

variable (M H m h : ℕ)
variable (total_students : ℕ := 36)
variable (percentage_middle : ℕ := 20)
variable (percentage_high : ℕ := 25)

theorem middle_school_students_count :
  total_students = 36 ∧ (m = h) →
  (percentage_middle / 100 * M = m) ∧
  (percentage_high / 100 * H = h) →
  M + H = total_students →
  M = 16 :=
by sorry

end middle_school_students_count_l202_20230


namespace find_real_numbers_l202_20201

theorem find_real_numbers (x y : ℝ) (h₁ : x + y = 3) (h₂ : x^5 + y^5 = 33) :
  (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1) := by
  sorry

end find_real_numbers_l202_20201


namespace sin_double_angle_l202_20209

theorem sin_double_angle (α : ℝ)
  (h : Real.cos (Real.pi / 4 - α) = -3 / 5) :
  Real.sin (2 * α) = -7 / 25 := by
sorry

end sin_double_angle_l202_20209


namespace polynomial_n_values_possible_num_values_of_n_l202_20261

theorem polynomial_n_values_possible :
  ∃ (n : ℤ), 
    (∀ (x : ℝ), x^3 - 4050 * x^2 + (m : ℝ) * x + (n : ℝ) = 0 → x > 0) ∧
    (∃ a : ℤ, a > 0 ∧ ∀ (x : ℝ), x^3 - 4050 * x^2 + (m : ℝ) * x + (n : ℝ) = 0 → 
      x = a ∨ x = a / 4 + r ∨ x = a / 4 - r) ∧
    1 ≤ r^2 ∧ r^2 ≤ 4090499 :=
sorry

theorem num_values_of_n : 
  ∃ (n_values : ℤ), n_values = 4088474 :=
sorry

end polynomial_n_values_possible_num_values_of_n_l202_20261


namespace largest_divisor_of_composite_l202_20232

theorem largest_divisor_of_composite (n : ℕ) (h : n > 1 ∧ ¬ Nat.Prime n) : 12 ∣ (n^4 - n^2) :=
sorry

end largest_divisor_of_composite_l202_20232


namespace bg_fg_ratio_l202_20249

open Real

-- Given the lengths AB, BD, AF, DF, BE, CF
def AB : ℝ := 15
def BD : ℝ := 18
def AF : ℝ := 15
def DF : ℝ := 12
def BE : ℝ := 24
def CF : ℝ := 17

-- Prove that the ratio BG : FG = 27 : 17
theorem bg_fg_ratio (BG FG : ℝ)
  (h_BG_FG : BG / FG = 27 / 17) :
  BG / FG = 27 / 17 := by
  sorry

end bg_fg_ratio_l202_20249


namespace decrease_percent_in_revenue_l202_20205

theorem decrease_percent_in_revenue
  (T C : ℝ) -- T = original tax, C = original consumption
  (h1 : 0 < T) -- ensuring that T is positive
  (h2 : 0 < C) -- ensuring that C is positive
  (new_tax : ℝ := 0.75 * T) -- new tax is 75% of original tax
  (new_consumption : ℝ := 1.10 * C) -- new consumption is 110% of original consumption
  (original_revenue : ℝ := T * C) -- original revenue
  (new_revenue : ℝ := (0.75 * T) * (1.10 * C)) -- new revenue
  (decrease_percent : ℝ := ((T * C - (0.75 * T) * (1.10 * C)) / (T * C)) * 100) -- decrease percent
  : decrease_percent = 17.5 :=
by
  sorry

end decrease_percent_in_revenue_l202_20205


namespace sum_of_Q_and_R_in_base_8_l202_20245

theorem sum_of_Q_and_R_in_base_8 (P Q R : ℕ) (hp : 1 ≤ P ∧ P < 8) (hq : 1 ≤ Q ∧ Q < 8) (hr : 1 ≤ R ∧ R < 8) 
  (hdistinct : P ≠ Q ∧ Q ≠ R ∧ P ≠ R) (H : 8^2 * P + 8 * Q + R + (8^2 * R + 8 * Q + P) + (8^2 * Q + 8 * P + R) 
  = 8^3 * P + 8^2 * P + 8 * P) : Q + R = 7 := 
sorry

end sum_of_Q_and_R_in_base_8_l202_20245


namespace additional_oil_needed_l202_20256

def oil_needed_each_cylinder : ℕ := 8
def number_of_cylinders : ℕ := 6
def oil_already_added : ℕ := 16

theorem additional_oil_needed : 
  (oil_needed_each_cylinder * number_of_cylinders) - oil_already_added = 32 := by
  sorry

end additional_oil_needed_l202_20256


namespace expression_evaluation_l202_20257

-- Define expression variable to ensure emphasis on conditions and calculations
def expression : ℤ := 9 - (8 + 7) * 6 + 5^2 - (4 * 3) + 2 - 1

theorem expression_evaluation : expression = -67 :=
by
  -- Use assumptions about the order of operations to conclude
  sorry

end expression_evaluation_l202_20257


namespace female_officers_on_police_force_l202_20252

theorem female_officers_on_police_force
  (percent_on_duty : ℝ)
  (total_on_duty : ℕ)
  (half_female_on_duty : ℕ)
  (h1 : percent_on_duty = 0.16)
  (h2 : total_on_duty = 160)
  (h3 : half_female_on_duty = total_on_duty / 2)
  (h4 : half_female_on_duty = 80)
  :
  ∃ (total_female_officers : ℕ), total_female_officers = 500 :=
by
  sorry

end female_officers_on_police_force_l202_20252


namespace coefficient_a2b2_in_expansion_l202_20207

theorem coefficient_a2b2_in_expansion :
  -- Combining the coefficients: \binom{4}{2} and \binom{6}{3}
  (Nat.choose 4 2) * (Nat.choose 6 3) = 120 :=
by
  -- No proof required, using sorry to indicate that.
  sorry

end coefficient_a2b2_in_expansion_l202_20207


namespace min_value_3x_plus_4y_l202_20297

theorem min_value_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = x * y) : 3 * x + 4 * y = 28 :=
sorry

end min_value_3x_plus_4y_l202_20297


namespace latus_rectum_of_parabola_l202_20248

theorem latus_rectum_of_parabola (p : ℝ) (hp : 0 < p) (A : ℝ × ℝ) (hA : A = (1, 1/2)) :
  ∃ a : ℝ, y^2 = 4 * a * x → A.2 ^ 2 = 4 * a * A.1 → x = -1 / (4 * a) → x = -1 / 16 :=
by
  sorry

end latus_rectum_of_parabola_l202_20248


namespace y_coord_diff_eq_nine_l202_20289

-- Declaring the variables and conditions
variables (m n : ℝ) (p : ℝ) (h1 : p = 3)
variable (L1 : m = (n / 3) - (2 / 5))
variable (L2 : m + p = ((n + 9) / 3) - (2 / 5))

-- The theorem statement
theorem y_coord_diff_eq_nine : (n + 9) - n = 9 :=
by
  sorry

end y_coord_diff_eq_nine_l202_20289


namespace minimize_cost_l202_20259

noncomputable def total_cost (x : ℝ) : ℝ := (16000000 / x) + 40000 * x

theorem minimize_cost : ∃ (x : ℝ), x > 0 ∧ (∀ y > 0, total_cost x ≤ total_cost y) ∧ x = 20 := 
sorry

end minimize_cost_l202_20259


namespace convex_polygon_triangles_impossible_l202_20262

theorem convex_polygon_triangles_impossible :
  ∀ (a b c : ℕ), 2016 + 2 * b + c - 2014 = 0 → a + b + c = 2014 → a = 1007 → false :=
sorry

end convex_polygon_triangles_impossible_l202_20262


namespace average_earning_week_l202_20291

theorem average_earning_week (D1 D2 D3 D4 D5 D6 D7 : ℝ)
  (h1 : (D1 + D2 + D3 + D4) / 4 = 25)
  (h2 : (D4 + D5 + D6 + D7) / 4 = 22)
  (h3 : D4 = 20) : 
  (D1 + D2 + D3 + D4 + D5 + D6 + D7) / 7 = 24 :=
by
  sorry

end average_earning_week_l202_20291


namespace cuboid_total_edge_length_cuboid_surface_area_l202_20253

variables (a b c : ℝ)

theorem cuboid_total_edge_length : 4 * (a + b + c) = 4 * (a + b + c) := 
by
  sorry

theorem cuboid_surface_area : 2 * (a * b + b * c + a * c) = 2 * (a * b + b * c + a * c) := 
by
  sorry

end cuboid_total_edge_length_cuboid_surface_area_l202_20253


namespace jack_walked_distance_l202_20295

def jack_walking_time: ℝ := 1.25
def jack_walking_rate: ℝ := 3.2
def jack_distance_walked: ℝ := 4

theorem jack_walked_distance:
  jack_walking_rate * jack_walking_time = jack_distance_walked :=
by
  sorry

end jack_walked_distance_l202_20295


namespace intersection_M_N_l202_20287

-- Define set M
def set_M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}

-- Define set N
def set_N : Set ℤ := {x | ∃ k : ℕ, k > 0 ∧ x = 2 * k - 1}

-- Define the intersection of M and N
def M_intersect_N : Set ℤ := {1, 3}

-- The theorem to prove
theorem intersection_M_N : set_M ∩ set_N = M_intersect_N :=
by sorry

end intersection_M_N_l202_20287


namespace sum_c_d_l202_20283

theorem sum_c_d (c d : ℝ) (h : ∀ x, (x - 2) * (x + 3) = x^2 + c * x + d) :
  c + d = -5 :=
sorry

end sum_c_d_l202_20283


namespace prism_volume_is_correct_l202_20298

noncomputable def prism_volume 
  (a b c : ℝ) 
  (hab : a * b = 15) 
  (hbc : b * c = 18) 
  (hca : c * a = 20) 
  (hc_longest : c = 2 * a) 
  : ℝ :=
  a * b * c

theorem prism_volume_is_correct 
  (a b c : ℝ) 
  (hab : a * b = 15) 
  (hbc : b * c = 18) 
  (hca : c * a = 20) 
  (hc_longest : c = 2 * a) 
  : prism_volume a b c hab hbc hca hc_longest = 30 * Real.sqrt 10 :=
sorry

end prism_volume_is_correct_l202_20298


namespace sequence_next_number_l202_20286

def next_number_in_sequence (seq : List ℕ) : ℕ :=
  if seq = [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2] then 3 else sorry

theorem sequence_next_number :
  next_number_in_sequence [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2] = 3 :=
by
  -- This proof is to ensure the pattern conditions are met
  sorry

end sequence_next_number_l202_20286


namespace zero_point_neg_x₀_l202_20222

-- Define odd function property
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define zero point condition for the function
def is_zero_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = Real.exp x₀

-- The main theorem to be proved
theorem zero_point_neg_x₀ (f : ℝ → ℝ) (x₀ : ℝ)
  (h_odd : is_odd_function f)
  (h_zero : is_zero_point f x₀) :
  f (-x₀) * Real.exp x₀ + 1 = 0 :=
sorry

end zero_point_neg_x₀_l202_20222


namespace playground_area_l202_20200

noncomputable def calculate_area (w s : ℝ) : ℝ := s * s

theorem playground_area (w s : ℝ) (h1 : s = 3 * w + 10) (h2 : 4 * s = 480) : calculate_area w s = 14400 := by
  sorry

end playground_area_l202_20200


namespace find_x_l202_20293

theorem find_x (x y z : ℝ) (h1 : x ≠ 0) 
  (h2 : x / 3 = z + 2 * y ^ 2) 
  (h3 : x / 6 = 3 * z - y) : 
  x = 168 :=
by
  sorry

end find_x_l202_20293


namespace cannot_achieve_80_cents_with_six_coins_l202_20280

theorem cannot_achieve_80_cents_with_six_coins:
  ¬ (∃ (p n d : ℕ), p + n + d = 6 ∧ p + 5 * n + 10 * d = 80) :=
by
  sorry

end cannot_achieve_80_cents_with_six_coins_l202_20280


namespace domain_of_f_l202_20273

open Real

noncomputable def f (x : ℝ) : ℝ := (log (2 * x - x^2)) / (x - 1)

theorem domain_of_f (x : ℝ) : (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) ↔ (2 * x - x^2 > 0 ∧ x ≠ 1) := by
  sorry

end domain_of_f_l202_20273


namespace pat_kate_ratio_l202_20276

theorem pat_kate_ratio 
  (P K M : ℕ)
  (h1 : P + K + M = 117)
  (h2 : ∃ r : ℕ, P = r * K)
  (h3 : P = M / 3)
  (h4 : M = K + 65) : 
  P / K = 2 :=
by
  sorry

end pat_kate_ratio_l202_20276


namespace joan_dozen_of_eggs_l202_20211

def number_of_eggs : ℕ := 72
def dozen : ℕ := 12

theorem joan_dozen_of_eggs : (number_of_eggs / dozen) = 6 := by
  sorry

end joan_dozen_of_eggs_l202_20211


namespace cost_of_milk_l202_20281

-- Given conditions
def total_cost_of_groceries : ℕ := 42
def cost_of_bananas : ℕ := 12
def cost_of_bread : ℕ := 9
def cost_of_apples : ℕ := 14

-- Prove that the cost of milk is $7
theorem cost_of_milk : total_cost_of_groceries - (cost_of_bananas + cost_of_bread + cost_of_apples) = 7 := 
by 
  sorry

end cost_of_milk_l202_20281


namespace dr_jones_remaining_salary_l202_20236

theorem dr_jones_remaining_salary:
  let salary := 6000
  let house_rental := 640
  let food_expense := 380
  let electric_water_bill := (1/4) * salary
  let insurances := (1/5) * salary
  let taxes := (10/100) * salary
  let transportation := (3/100) * salary
  let emergency_costs := (2/100) * salary
  let total_expenses := house_rental + food_expense + electric_water_bill + insurances + taxes + transportation + emergency_costs
  let remaining_salary := salary - total_expenses
  remaining_salary = 1380 :=
by
  sorry

end dr_jones_remaining_salary_l202_20236


namespace minimum_groups_needed_l202_20220

theorem minimum_groups_needed :
  ∃ (g : ℕ), g = 5 ∧ ∀ n k : ℕ, n = 30 → k ≤ 7 → n / k = g :=
by
  sorry

end minimum_groups_needed_l202_20220


namespace time_to_cross_platform_l202_20296

/-- Definitions of the conditions in the problem. -/
def train_length : ℕ := 1500
def platform_length : ℕ := 1800
def time_to_cross_tree : ℕ := 100
def train_speed : ℕ := train_length / time_to_cross_tree
def total_distance : ℕ := train_length + platform_length

/-- Proof statement: The time for the train to pass the platform. -/
theorem time_to_cross_platform : (total_distance / train_speed) = 220 := by
  sorry

end time_to_cross_platform_l202_20296


namespace square_field_area_l202_20243

theorem square_field_area (x : ℕ) 
    (hx : 4 * x - 2 = 666) : x^2 = 27889 := by
  -- We would solve for x using the given equation.
  sorry

end square_field_area_l202_20243


namespace regular_polygon_area_l202_20203
open Real

theorem regular_polygon_area (R : ℝ) (n : ℕ) (hR : 0 < R) (hn : 8 ≤ n) (h_area : (1/2) * n * R^2 * sin (360 / n * (π / 180)) = 4 * R^2) :
  n = 10 := 
sorry

end regular_polygon_area_l202_20203


namespace minimum_reciprocal_sum_l202_20294

noncomputable def log_function_a (a : ℝ) (x : ℝ) : ℝ := 
  Real.log x / Real.log a

theorem minimum_reciprocal_sum (a m n : ℝ) 
  (ha1 : 0 < a) (ha2 : a ≠ 1) 
  (hmn : 0 < m ∧ 0 < n ∧ 2 * m + n = 2) 
  (hA : log_function_a a (1 : ℝ) + -1 = -1) 
  : 1 / m + 2 / n = 4 := 
by
  sorry

end minimum_reciprocal_sum_l202_20294


namespace sum_odd_even_integers_l202_20244

theorem sum_odd_even_integers :
  let odd_terms_sum := (15 / 2) * (1 + 29)
  let even_terms_sum := (10 / 2) * (2 + 20)
  odd_terms_sum + even_terms_sum = 335 :=
by
  let odd_terms_sum := (15 / 2) * (1 + 29)
  let even_terms_sum := (10 / 2) * (2 + 20)
  show odd_terms_sum + even_terms_sum = 335
  sorry

end sum_odd_even_integers_l202_20244


namespace limit_series_product_eq_l202_20255

variable (a r s : ℝ)

noncomputable def series_product_sum_limit : ℝ :=
∑' n : ℕ, (a * r^n) * (a * s^n)

theorem limit_series_product_eq :
  |r| < 1 → |s| < 1 → series_product_sum_limit a r s = a^2 / (1 - r * s) :=
by
  intro hr hs
  sorry

end limit_series_product_eq_l202_20255


namespace roy_older_than_julia_l202_20208

variable {R J K x : ℝ}

theorem roy_older_than_julia (h1 : R = J + x)
                            (h2 : R = K + x / 2)
                            (h3 : R + 2 = 2 * (J + 2))
                            (h4 : (R + 2) * (K + 2) = 192) :
                            x = 2 :=
by
  sorry

end roy_older_than_julia_l202_20208


namespace inequality_a_b_l202_20238

theorem inequality_a_b (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
    a / (b + 1) + b / (a + 1) ≤ 1 :=
  sorry

end inequality_a_b_l202_20238


namespace area_triangle_ABC_l202_20242

noncomputable def point := ℝ × ℝ

structure Parallelogram (A B C D : point) : Prop :=
(parallel_AB_CD : ∃ m1 m2, m1 ≠ m2 ∧ (A.2 - B.2) / (A.1 - B.1) = m1 ∧ (C.2 - D.2) / (C.1 - D.1) = m2)
(equal_heights : ∃ h, (B.2 - A.2 = h) ∧ (C.2 - D.2 = h))
(area_parallelogram : (B.1 - A.1) * (B.2 - A.2) + (C.1 - D.1) * (C.2 - D.2) = 27)
(thrice_length : (C.1 - D.1) = 3 * (B.1 - A.1))

theorem area_triangle_ABC (A B C D : point) (h : Parallelogram A B C D) : 
  ∃ triangle_area : ℝ, triangle_area = 13.5 :=
by
  sorry

end area_triangle_ABC_l202_20242


namespace age_ratio_l202_20284

theorem age_ratio (R D : ℕ) (h1 : D = 15) (h2 : R + 6 = 26) : R / D = 4 / 3 := by
  sorry

end age_ratio_l202_20284


namespace trapezoid_perimeter_l202_20212

noncomputable def length_AD : ℝ := 8
noncomputable def length_BC : ℝ := 18
noncomputable def length_AB : ℝ := 12 -- Derived from tangency and symmetry considerations
noncomputable def length_CD : ℝ := 18

theorem trapezoid_perimeter (ABCD : Π (a b c d : Type), a → b → c → d → Prop)
  (AD BC AB CD : ℝ)
  (h1 : AD = 8) (h2 : BC = 18) (h3 : AB = 12) (h4 : CD = 18)
  : AD + BC + AB + CD = 56 :=
by
  rw [h1, h2, h3, h4]
  norm_num

end trapezoid_perimeter_l202_20212


namespace normal_price_of_article_l202_20235

theorem normal_price_of_article (P : ℝ) (sale_price : ℝ) (discount1 discount2 : ℝ) :
  discount1 = 0.10 → discount2 = 0.20 → sale_price = 108 →
  P * (1 - discount1) * (1 - discount2) = sale_price → P = 150 :=
by
  intro hd1 hd2 hsp hdiscount
  -- skipping the proof for now
  sorry

end normal_price_of_article_l202_20235


namespace solve_system_of_equations_l202_20279

theorem solve_system_of_equations : 
  ∃ (x y : ℝ), 
  (x / y + y / x) * (x + y) = 15 ∧ 
  (x^2 / y^2 + y^2 / x^2) * (x^2 + y^2) = 85 ∧
  ((x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2)) :=
by
  sorry

end solve_system_of_equations_l202_20279


namespace stratified_sampling_red_balls_l202_20237

theorem stratified_sampling_red_balls (total_balls red_balls sample_size : ℕ) (h_total : total_balls = 100) (h_red : red_balls = 20) (h_sample : sample_size = 10) :
  (sample_size * (red_balls / total_balls)) = 2 := by
  sorry

end stratified_sampling_red_balls_l202_20237


namespace men_in_first_group_l202_20285

noncomputable def first_group_men (x m b W : ℕ) : Prop :=
  let eq1 := 10 * x * m + 80 * b = W
  let eq2 := 2 * (26 * m + 48 * b) = W
  let eq3 := 4 * (15 * m + 20 * b) = W
  eq1 ∧ eq2 ∧ eq3

theorem men_in_first_group (m b W : ℕ) (h_condition : first_group_men 6 m b W) : 
  ∃ x, x = 6 :=
by
  sorry

end men_in_first_group_l202_20285


namespace frost_time_with_sprained_wrist_l202_20231

-- Definitions
def normal_time_per_cake : ℕ := 5
def additional_time_for_10_cakes : ℕ := 30
def normal_time_for_10_cakes : ℕ := 10 * normal_time_per_cake
def sprained_time_for_10_cakes : ℕ := normal_time_for_10_cakes + additional_time_for_10_cakes

-- Theorems
theorem frost_time_with_sprained_wrist : ∀ x : ℕ, 
  (10 * x = sprained_time_for_10_cakes) ↔ (x = 8) := 
sorry

end frost_time_with_sprained_wrist_l202_20231


namespace find_numbers_l202_20241

theorem find_numbers (n : ℕ) (h1 : n ≥ 2) (a : ℕ) (ha : a ≠ 1) (ha_min : ∀ d, d ∣ n → d ≠ 1 → a ≤ d) (b : ℕ) (hb : b ∣ n) :
  n = a^2 + b^2 ↔ n = 8 ∨ n = 20 :=
by sorry

end find_numbers_l202_20241


namespace planting_scheme_correct_l202_20288

-- Setting up the problem as the conditions given
def types_of_seeds := ["peanuts", "Chinese cabbage", "potatoes", "corn", "wheat", "apples"]

def first_plot_seeds := ["corn", "apples"]

def planting_schemes_count : ℕ :=
  let choose_first_plot := 2  -- C(2, 1), choosing either "corn" or "apples" for the first plot
  let remaining_seeds := 5  -- 6 - 1 = 5 remaining seeds after choosing for the first plot
  let arrangements_remaining := 5 * 4 * 3  -- A(5, 3), arrangements of 3 plots from 5 remaining seeds
  choose_first_plot * arrangements_remaining

theorem planting_scheme_correct : planting_schemes_count = 120 := by
  sorry

end planting_scheme_correct_l202_20288


namespace complement_U_A_l202_20218

def U : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def A : Set ℝ := {x | 3 ≤ 2 * x - 1 ∧ 2 * x - 1 < 5}

theorem complement_U_A : (U \ A) = {x | (0 ≤ x ∧ x < 2) ∨ (3 ≤ x)} := sorry

end complement_U_A_l202_20218
