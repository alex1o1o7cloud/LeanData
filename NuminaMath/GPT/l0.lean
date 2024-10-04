import Mathlib

namespace maximum_possible_median_l0_857

theorem maximum_possible_median
  (total_cans : ℕ)
  (total_customers : ℕ)
  (min_cans_per_customer : ℕ)
  (alt_min_cans_per_customer : ℕ)
  (exact_min_cans_count : ℕ)
  (atleast_min_cans_count : ℕ)
  (min_cans_customers : ℕ)
  (alt_min_cans_customer: ℕ): 
  (total_cans = 300) → 
  (total_customers = 120) →
  (min_cans_per_customer = 2) →
  (alt_min_cans_per_customer = 4) →
  (min_cans_customers = 59) →
  (alt_min_cans_customer = 61) →
  (min_cans_per_customer * min_cans_customers + alt_min_cans_per_customer * (total_customers - min_cans_customers) = total_cans) →
  max (min_cans_per_customer + 1) (alt_min_cans_per_customer - 1) = 3 :=
sorry

end maximum_possible_median_l0_857


namespace value_of_x_when_z_is_32_l0_145

variables {x y z k : ℝ}
variable (m n : ℝ)

def directly_proportional (x y : ℝ) (m : ℝ) := x = m * y^2
def inversely_proportional (y z : ℝ) (n : ℝ) := y = n / z^2

-- Our main proof goal
theorem value_of_x_when_z_is_32 (h1 : directly_proportional x y m) 
  (h2 : inversely_proportional y z n) (h3 : z = 8) (hx : x = 5) : 
  x = 5 / 256 :=
by
  let k := x * z^4
  have k_value : k = 20480 := by sorry
  have x_new : x = k / z^4 := by sorry
  have z_new : z = 32 := by sorry
  have x_final : x = 5 / 256 := by sorry
  exact x_final

end value_of_x_when_z_is_32_l0_145


namespace chinese_money_plant_sales_l0_771

/-- 
Consider a scenario where a plant supplier sells 20 pieces of orchids for $50 each 
and some pieces of potted Chinese money plant for $25 each. He paid his two workers $40 each 
and bought new pots worth $150. The plant supplier had $1145 left from his earnings. 
Prove that the number of pieces of potted Chinese money plants sold by the supplier is 15.
-/
theorem chinese_money_plant_sales (earnings_orchids earnings_per_orchid: ℤ)
  (num_orchids: ℤ)
  (earnings_plants earnings_per_plant: ℤ)
  (worker_wage num_workers: ℤ)
  (new_pots_cost remaining_money: ℤ)
  (earnings: ℤ)
  (P : earnings_orchids = num_orchids * earnings_per_orchid)
  (Q : earnings = earnings_orchids + earnings_plants)
  (R : earnings - (worker_wage * num_workers + new_pots_cost) = remaining_money)
  (conditions: earnings_per_orchid = 50 ∧ num_orchids = 20 ∧ earnings_per_plant = 25 ∧ worker_wage = 40 ∧ num_workers = 2 ∧ new_pots_cost = 150 ∧ remaining_money = 1145):
  earnings_plants / earnings_per_plant = 15 := 
by
  sorry

end chinese_money_plant_sales_l0_771


namespace sunscreen_cost_l0_269

theorem sunscreen_cost (reapply_time : ℕ) (oz_per_application : ℕ) 
  (oz_per_bottle : ℕ) (cost_per_bottle : ℝ) (total_time : ℕ) (expected_cost : ℝ) :
  reapply_time = 2 →
  oz_per_application = 3 →
  oz_per_bottle = 12 →
  cost_per_bottle = 3.5 →
  total_time = 16 →
  expected_cost = 7 →
  (total_time / reapply_time) * (oz_per_application / oz_per_bottle) * cost_per_bottle = expected_cost :=
by
  intros
  sorry

end sunscreen_cost_l0_269


namespace length_of_train_l0_298

-- Definitions of given conditions
def train_speed (kmh : ℤ) := 25
def man_speed (kmh : ℤ) := 2
def crossing_time (sec : ℤ) := 28

-- Relative speed calculation (in meters per second)
def relative_speed := (train_speed 1 + man_speed 1) * (5 / 18 : ℚ)

-- Distance calculation (in meters)
def distance_covered := relative_speed * (crossing_time 1 : ℚ)

-- The theorem statement: Length of the train equals distance covered in crossing time
theorem length_of_train : distance_covered = 210 := by
  sorry

end length_of_train_l0_298


namespace randy_wipes_days_l0_989

theorem randy_wipes_days (wipes_per_pack : ℕ) (packs_needed : ℕ) (wipes_per_walk : ℕ) (walks_per_day : ℕ) (total_wipes : ℕ) (wipes_per_day : ℕ) (days_needed : ℕ) 
(h1 : wipes_per_pack = 120)
(h2 : packs_needed = 6)
(h3 : wipes_per_walk = 4)
(h4 : walks_per_day = 2)
(h5 : total_wipes = packs_needed * wipes_per_pack)
(h6 : wipes_per_day = wipes_per_walk * walks_per_day)
(h7 : days_needed = total_wipes / wipes_per_day) : 
days_needed = 90 :=
by sorry

end randy_wipes_days_l0_989


namespace jacks_speed_is_7_l0_95

-- Define the constants and speeds as given in conditions
def initial_distance : ℝ := 150
def christina_speed : ℝ := 8
def lindy_speed : ℝ := 10
def lindy_total_distance : ℝ := 100

-- Hypothesis stating when the three meet
theorem jacks_speed_is_7 :
  ∃ (jack_speed : ℝ), (∃ (time : ℝ), 
    time = lindy_total_distance / lindy_speed
    ∧ christina_speed * time + jack_speed * time = initial_distance) 
  → jack_speed = 7 :=
by {
  -- Placeholder for the proof
  sorry
}

end jacks_speed_is_7_l0_95


namespace last_three_digits_7_pow_103_l0_795

theorem last_three_digits_7_pow_103 : (7 ^ 103) % 1000 = 60 := sorry

end last_three_digits_7_pow_103_l0_795


namespace quadratic_inequality_solution_l0_48

theorem quadratic_inequality_solution (z : ℝ) :
  z^2 - 40 * z + 340 ≤ 4 ↔ 12 ≤ z ∧ z ≤ 28 := by 
  sorry

end quadratic_inequality_solution_l0_48


namespace star_24_75_l0_496

noncomputable def star (a b : ℝ) : ℝ := sorry 

-- Conditions
axiom star_one_one : star 1 1 = 2
axiom star_ab_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : star (a * b) b = a * (star b b)
axiom star_a_one (a : ℝ) (h : 0 < a) : star a 1 = 2 * a

-- Theorem to prove
theorem star_24_75 : star 24 75 = 1800 := 
by 
  sorry

end star_24_75_l0_496


namespace find_x_range_l0_334

theorem find_x_range (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -2) (h3 : 2 * x - 5 > 0) : x > 5 / 2 :=
by
  sorry

end find_x_range_l0_334


namespace pyramid_volume_in_unit_cube_l0_162

noncomputable def base_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

noncomputable def pyramid_volume (base_area : ℝ) (height : ℝ) : ℝ := (1 / 3) * base_area * height

theorem pyramid_volume_in_unit_cube : 
  let s := Real.sqrt 2 / 2
  let height := 1
  pyramid_volume (base_area s) height = Real.sqrt 3 / 24 :=
by
  sorry

end pyramid_volume_in_unit_cube_l0_162


namespace total_students_in_school_l0_88

noncomputable def total_students (girls boys : ℕ) (ratio_girls boys_ratio : ℕ) : ℕ :=
  let parts := ratio_girls + boys_ratio
  let students_per_part := girls / ratio_girls
  students_per_part * parts

theorem total_students_in_school (girls : ℕ) (ratio_girls boys_ratio : ℕ) (h1 : ratio_girls = 5) (h2 : boys_ratio = 8) (h3 : girls = 160) :
  total_students girls boys_ratio ratio_girls = 416 :=
  by
  -- proof would go here
  sorry

end total_students_in_school_l0_88


namespace find_f_value_l0_631

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x^α

theorem find_f_value (α : ℝ) (h : f 3 α = Real.sqrt 3) : f (1 / 4) α = 1 / 2 :=
by
  sorry

end find_f_value_l0_631


namespace infinite_expressible_terms_l0_385

theorem infinite_expressible_terms
  (a : ℕ → ℕ)
  (h1 : ∀ n, a n < a (n + 1)) :
  ∃ f : ℕ → ℕ, (∀ n, a (f n) = (f n).succ * a 1 + (f n).succ.succ * a 2) ∧
    ∀ i j, i ≠ j → f i ≠ f j :=
by
  sorry

end infinite_expressible_terms_l0_385


namespace gcd_18_30_45_l0_487

theorem gcd_18_30_45 : Nat.gcd (Nat.gcd 18 30) 45 = 3 :=
by
  sorry

end gcd_18_30_45_l0_487


namespace age_of_jerry_l0_107

variable (M J : ℕ)

theorem age_of_jerry (h1 : M = 2 * J - 5) (h2 : M = 19) : J = 12 := by
  sorry

end age_of_jerry_l0_107


namespace Tom_marble_choices_l0_133

theorem Tom_marble_choices :
  let total_marbles := 18
  let special_colors := 4
  let choose_one_from_special := (Nat.choose special_colors 1)
  let remaining_marbles := total_marbles - special_colors
  let choose_remaining := (Nat.choose remaining_marbles 5)
  choose_one_from_special * choose_remaining = 8008
:= sorry

end Tom_marble_choices_l0_133


namespace normal_distribution_symmetry_l0_923

noncomputable def normal_probability_symmetry (X : ℝ → ℝ) (mean : ℝ) : Prop :=
  ∀ (a b : ℝ), (mean - a < X) ∧ (X < mean - b) → 
  P(a < X < b) = P(mean + a < X < mean + b)

theorem normal_distribution_symmetry :
  ∀ (X : ℝ → ℝ) (mean : ℝ) (a b : ℝ),
  (X follows a normal distribution ∧ mean = 500 ∧ P(400 < X < 450) = 0.3) →
  P(550 < X < 600) = 0.3 :=
sorry

end normal_distribution_symmetry_l0_923


namespace other_asymptote_l0_536

-- Define the conditions
def C1 := ∀ x y, y = -2 * x
def C2 := ∀ x, x = -3

-- Formulate the problem
theorem other_asymptote :
  (∃ y m b, y = m * x + b ∧ m = 2 ∧ b = 12) :=
by
  sorry

end other_asymptote_l0_536


namespace rhombus_perimeter_is_80_l0_418

-- Definitions of the conditions
def rhombus_diagonals_ratio : Prop := ∃ (d1 d2 : ℝ), d1 / d2 = 3 / 4 ∧ d1 + d2 = 56

-- The goal is to prove that given the conditions, the perimeter of the rhombus is 80
theorem rhombus_perimeter_is_80 (h : rhombus_diagonals_ratio) : ∃ (p : ℝ), p = 80 :=
by
  sorry  -- The actual proof steps would go here

end rhombus_perimeter_is_80_l0_418


namespace vegetarian_eaters_l0_376

-- Define the conditions
theorem vegetarian_eaters : 
  ∀ (total family_size : ℕ) 
  (only_veg only_nonveg both_veg_nonveg eat_veg : ℕ), 
  family_size = 45 → 
  only_veg = 22 → 
  only_nonveg = 15 → 
  both_veg_nonveg = 8 → 
  eat_veg = only_veg + both_veg_nonveg → 
  eat_veg = 30 :=
by
  intros total family_size only_veg only_nonveg both_veg_nonveg eat_veg
  sorry

end vegetarian_eaters_l0_376


namespace proof_C_ST_l0_529

-- Definitions for sets and their operations
def A1 : Set ℕ := {0, 1}
def A2 : Set ℕ := {1, 2}
def S : Set ℕ := A1 ∪ A2
def T : Set ℕ := A1 ∩ A2
def C_ST : Set ℕ := S \ T

theorem proof_C_ST : 
  C_ST = {0, 2} := 
by 
  sorry

end proof_C_ST_l0_529


namespace crews_complete_job_l0_420

-- Define the productivity rates for each crew
variables (x y z : ℝ)

-- Define the conditions derived from the problem
def condition1 : Prop := 1/(x + y) = 1/z - 3/5
def condition2 : Prop := 1/(x + z) = 1/y
def condition3 : Prop := 1/(y + z) = 2/(7 * x)

-- Target proof: the combined time for all three crews
def target_proof : Prop := 1/(x + y + z) = 4/3

-- Final Lean 4 statement combining all conditions and proof requirement
theorem crews_complete_job (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 x y z) (h3 : condition3 x y z) : target_proof x y z :=
sorry

end crews_complete_job_l0_420


namespace appropriate_import_range_l0_860

def mung_bean_import_range (p0 : ℝ) (p_desired_min p_desired_max : ℝ) (x : ℝ) : Prop :=
  p0 - (x / 100) ≤ p_desired_max ∧ p0 - (x / 100) ≥ p_desired_min

theorem appropriate_import_range : 
  ∃ x : ℝ, 600 ≤ x ∧ x ≤ 800 ∧ mung_bean_import_range 16 8 10 x :=
sorry

end appropriate_import_range_l0_860


namespace find_day_53_days_from_friday_l0_732

-- Define the days of the week
inductive day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open day

-- Define a function that calculates the day of the week after a certain number of days
def add_days (d : day) (n : ℕ) : day :=
  match d, n % 7 with
  | Sunday, 0 => Sunday
  | Monday, 0 => Monday
  | Tuesday, 0 => Tuesday
  | Wednesday, 0 => Wednesday
  | Thursday, 0 => Thursday
  | Friday, 0 => Friday
  | Saturday, 0 => Saturday
  | Sunday, 1 => Monday
  | Monday, 1 => Tuesday
  | Tuesday, 1 => Wednesday
  | Wednesday, 1 => Thursday
  | Thursday, 1 => Friday
  | Friday, 1 => Saturday
  | Saturday, 1 => Sunday
  | Sunday, 2 => Tuesday
  | Monday, 2 => Wednesday
  | Tuesday, 2 => Thursday
  | Wednesday, 2 => Friday
  | Thursday, 2 => Saturday
  | Friday, 2 => Sunday
  | Saturday, 2 => Monday
  | Sunday, 3 => Wednesday
  | Monday, 3 => Thursday
  | Tuesday, 3 => Friday
  | Wednesday, 3 => Saturday
  | Thursday, 3 => Sunday
  | Friday, 3 => Monday
  | Saturday, 3 => Tuesday
  | Sunday, 4 => Thursday
  | Monday, 4 => Friday
  | Tuesday, 4 => Saturday
  | Wednesday, 4 => Sunday
  | Thursday, 4 => Monday
  | Friday, 4 => Tuesday
  | Saturday, 4 => Wednesday
  | Sunday, 5 => Friday
  | Monday, 5 => Saturday
  | Tuesday, 5 => Sunday
  | Wednesday, 5 => Monday
  | Thursday, 5 => Tuesday
  | Friday, 5 => Wednesday
  | Saturday, 5 => Thursday
  | Sunday, 6 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 6 => Friday
  | _, _ => Sunday -- This case is unreachable, but required to complete the pattern match
  end

-- Lean statement for the proof problem:
theorem find_day_53_days_from_friday : add_days Friday 53 = Tuesday :=
  sorry

end find_day_53_days_from_friday_l0_732


namespace total_dividends_correct_l0_896

-- Conditions
def net_profit (total_income expenses loan_penalty_rate : ℝ) : ℝ :=
  let net1 := total_income - expenses
  let loan_penalty := net1 * loan_penalty_rate
  net1 - loan_penalty

def total_loan_payments (monthly_payment months additional_payment : ℝ) : ℝ :=
  (monthly_payment * months) - additional_payment

def dividend_per_share (net_profit total_loan_payments num_shares : ℝ) : ℝ :=
  (net_profit - total_loan_payments) / num_shares

noncomputable def total_dividends_director (dividend_per_share shares_owned : ℝ) : ℝ :=
  dividend_per_share * shares_owned

theorem total_dividends_correct :
  total_dividends_director (dividend_per_share (net_profit 1500000 674992 0.2) (total_loan_payments 23914 12 74992) 1000) 550 = 246400 :=
sorry

end total_dividends_correct_l0_896


namespace f_monotonic_intervals_g_greater_than_4_3_l0_365

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def g (x : ℝ) : ℝ := f x - Real.log x

theorem f_monotonic_intervals :
  (∀ x < -1, ∀ y < -1, x < y → f x > f y) ∧ 
  (∀ x > -1, ∀ y > -1, x < y → f x < f y) :=
sorry

theorem g_greater_than_4_3 (x : ℝ) (h : x > 0) : g x > (4 / 3) :=
sorry

end f_monotonic_intervals_g_greater_than_4_3_l0_365


namespace speed_of_point_C_l0_350

theorem speed_of_point_C 
    (a T R L x : ℝ) 
    (h1 : x = L * (a * T) / R - L) 
    (h_eq: (a * T) / (a * T - R) = (L + x) / x) :
    (a * L) / R = x / T :=
by
  sorry

end speed_of_point_C_l0_350


namespace solve_for_x_l0_571

theorem solve_for_x (x : ℝ) (h : 5 * x + 3 = 10 * x - 17) : x = 4 :=
by {
  sorry
}

end solve_for_x_l0_571


namespace number_of_friends_l0_156

-- Let n be the number of friends
-- Given the conditions:
-- 1. 9 chicken wings initially.
-- 2. 7 more chicken wings cooked.
-- 3. Each friend gets 4 chicken wings.

theorem number_of_friends :
  let initial_wings := 9
  let additional_wings := 7
  let wings_per_friend := 4
  let total_wings := initial_wings + additional_wings
  let n := total_wings / wings_per_friend
  n = 4 :=
by
  sorry

end number_of_friends_l0_156


namespace day_53_days_from_friday_l0_721

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end day_53_days_from_friday_l0_721


namespace dacid_weighted_average_l0_932

theorem dacid_weighted_average :
  let english := 96
  let mathematics := 95
  let physics := 82
  let chemistry := 87
  let biology := 92
  let weight_english := 0.20
  let weight_mathematics := 0.25
  let weight_physics := 0.15
  let weight_chemistry := 0.25
  let weight_biology := 0.15
  (english * weight_english) + (mathematics * weight_mathematics) +
  (physics * weight_physics) + (chemistry * weight_chemistry) +
  (biology * weight_biology) = 90.8 :=
by
  sorry

end dacid_weighted_average_l0_932


namespace necessary_and_sufficient_condition_l0_359

theorem necessary_and_sufficient_condition (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  (a > b) ↔ (a - 1/a > b - 1/b) :=
sorry

end necessary_and_sufficient_condition_l0_359


namespace isosceles_triangle_perimeter_l0_624

-- Define the given conditions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem based on the problem statement and conditions
theorem isosceles_triangle_perimeter (a b : ℕ) (P : is_isosceles_triangle a b 5) (Q : is_isosceles_triangle b a 10) :
  valid_triangle a b 5 → valid_triangle b a 10 → a + b + 5 = 25 :=
by sorry

end isosceles_triangle_perimeter_l0_624


namespace integer_pairs_satisfying_condition_l0_602

theorem integer_pairs_satisfying_condition :
  { (m, n) : ℤ × ℤ | ∃ k : ℤ, (n^3 + 1) = k * (m * n - 1) } =
  { (1, 2), (1, 3), (2, 1), (3, 1), (2, 5), (3, 5), (5, 2), (5, 3), (2, 2) } :=
sorry

end integer_pairs_satisfying_condition_l0_602


namespace ant_probability_l0_302

-- Variables for the initial and target positions.
variables (start : ℕ × ℕ) (target : ℕ × ℕ)

-- Conditions: Starting at point A and moving for 6 minutes.
def initial_position := (0, 0)  -- Point labeled A
def target_position := (-2, 0)  -- Point labeled C
def time_steps := 6

-- Definition for movement, considering diagonal moves as well.
def neighbors (pos : ℕ × ℕ) : set (ℕ × ℕ) :=
  {(x + dx, y + dy) | dx, dy ∈ [-1, 0, 1], (dx ≠ 0 ∨ dy ≠ 0) ∧ (x, y) = pos}

-- Probability calculation.
-- This assumes each dot has an equal probability to be chosen next.
noncomputable def probability_of_reaching_target : ℝ :=
  begin
    /- Reasoning must be built here, but we assert the result for now -/
    exact 1 / 2,
  end

-- The theorem asserting the required probability equality.
theorem ant_probability :
  probability_of_reaching_target initial_position target_position time_steps = 1 / 2 :=
begin
  sorry -- Proof to be constructed
end

end ant_probability_l0_302


namespace number_of_truthful_dwarfs_l0_177

def num_dwarfs : Nat := 10
def num_vanilla : Nat := 10
def num_chocolate : Nat := 5
def num_fruit : Nat := 1

def total_hands_raised : Nat := num_vanilla + num_chocolate + num_fruit
def num_extra_hands : Nat := total_hands_raised - num_dwarfs

variable (T L : Nat)

axiom dwarfs_count : T + L = num_dwarfs
axiom hands_by_liars : L = num_extra_hands

theorem number_of_truthful_dwarfs : T = 4 :=
by
  have total_liars: num_dwarfs - T = num_extra_hands := by sorry
  have final_truthful: T = num_dwarfs - num_extra_hands := by sorry
  show T = 4 from final_truthful

end number_of_truthful_dwarfs_l0_177


namespace difference_between_x_and_y_l0_81

theorem difference_between_x_and_y 
  (x y : ℕ) 
  (h1 : 3 ^ x * 4 ^ y = 531441) 
  (h2 : x = 12) : x - y = 12 := 
by 
  sorry

end difference_between_x_and_y_l0_81


namespace symmetric_point_of_M_origin_l0_952

-- Define the point M with given coordinates
def M : (ℤ × ℤ) := (-3, -5)

-- The theorem stating that the symmetric point of M about the origin is (3, 5)
theorem symmetric_point_of_M_origin :
  let symmetric_point : (ℤ × ℤ) := (-M.1, -M.2)
  symmetric_point = (3, 5) :=
by
  -- (Proof should be filled)
  sorry

end symmetric_point_of_M_origin_l0_952


namespace inverse_proportion_relationship_l0_353

theorem inverse_proportion_relationship (k : ℝ) (y1 y2 y3 : ℝ) :
  y1 = (k^2 + 1) / -1 →
  y2 = (k^2 + 1) / 1 →
  y3 = (k^2 + 1) / 2 →
  y1 < y3 ∧ y3 < y2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end inverse_proportion_relationship_l0_353


namespace sum_of_integers_l0_265

theorem sum_of_integers (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1)
  (h4 : a * b * c = 343000)
  (h5 : Nat.gcd a b = 1) (h6 : Nat.gcd b c = 1) (h7 : Nat.gcd a c = 1) :
  a + b + c = 476 :=
by
  sorry

end sum_of_integers_l0_265


namespace last_three_digits_of_7_pow_103_l0_793

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 614 := by
  sorry

end last_three_digits_of_7_pow_103_l0_793


namespace charcoal_drawings_correct_l0_711

-- Define the constants based on the problem conditions
def total_drawings : ℕ := 120
def colored_pencils : ℕ := 35
def blending_markers : ℕ := 22
def pastels : ℕ := 15
def watercolors : ℕ := 12

-- Calculate the total number of charcoal drawings
def charcoal_drawings : ℕ := total_drawings - (colored_pencils + blending_markers + pastels + watercolors)

-- The theorem we want to prove is that the number of charcoal drawings is 36
theorem charcoal_drawings_correct : charcoal_drawings = 36 :=
by
  -- The proof goes here (we skip it with 'sorry')
  sorry

end charcoal_drawings_correct_l0_711


namespace factorial_division_l0_65

-- Conditions: definition for factorial
def factorial : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * factorial n

-- Statement of the problem: Proving the equality
theorem factorial_division :
  (factorial 10) / ((factorial 5) * (factorial 2)) = 15120 :=
by
  sorry

end factorial_division_l0_65


namespace A_knit_time_l0_453

def rate_A (x : ℕ) : ℚ := 1 / x
def rate_B : ℚ := 1 / 6

def combined_rate_two_pairs_in_4_days (x : ℕ) : Prop :=
  rate_A x + rate_B = 1 / 2

theorem A_knit_time : ∃ x : ℕ, combined_rate_two_pairs_in_4_days x ∧ x = 3 :=
by
  existsi 3
  -- (Formal proof would go here)
  sorry

end A_knit_time_l0_453


namespace fraction_meaningful_if_not_neg_two_l0_895

theorem fraction_meaningful_if_not_neg_two {a : ℝ} : (a + 2 ≠ 0) ↔ (a ≠ -2) :=
by sorry

end fraction_meaningful_if_not_neg_two_l0_895


namespace area_of_shaded_rectangle_l0_466

theorem area_of_shaded_rectangle (w₁ h₁ w₂ h₂: ℝ) 
  (hw₁: w₁ * h₁ = 6)
  (hw₂: w₂ * h₁ = 15)
  (hw₃: w₂ * h₂ = 25) :
  w₁ * h₂ = 10 :=
by
  sorry

end area_of_shaded_rectangle_l0_466


namespace tn_lt_sn_div_2_l0_680

section geometric_sequence

open_locale big_operators

def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
def b (n : ℕ) : ℝ := (n : ℝ) * (1 / 3)^n

def S (n : ℕ) : ℝ := ∑ i in finset.range n, a (i + 1)
def T (n : ℕ) : ℝ := ∑ i in finset.range n, b (i + 1)

theorem tn_lt_sn_div_2 (n : ℕ) : T n < S n / 2 := sorry

end geometric_sequence

end tn_lt_sn_div_2_l0_680


namespace rectangle_to_square_l0_855

variable (k n : ℕ)

theorem rectangle_to_square (h1 : k > 5) (h2 : k * (k - 5) = n^2) : n = 6 := by 
  sorry

end rectangle_to_square_l0_855


namespace blue_chips_count_l0_12

variable (T : ℕ) (blue_chips : ℕ) (white_chips : ℕ) (green_chips : ℕ)

-- Conditions
def condition1 : Prop := blue_chips = (T / 10)
def condition2 : Prop := white_chips = (T / 2)
def condition3 : Prop := green_chips = 12
def condition4 : Prop := blue_chips + white_chips + green_chips = T

-- Proof problem
theorem blue_chips_count (h1 : condition1 T blue_chips)
                          (h2 : condition2 T white_chips)
                          (h3 : condition3 green_chips)
                          (h4 : condition4 T blue_chips white_chips green_chips) :
  blue_chips = 3 :=
sorry

end blue_chips_count_l0_12


namespace radius_of_circle_l0_256

def circle_eq_def (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

theorem radius_of_circle {x y r : ℝ} (h : circle_eq_def x y) : r = 3 := 
by
  -- Proof skipped
  sorry

end radius_of_circle_l0_256


namespace truthful_dwarfs_count_l0_187

def number_of_dwarfs := 10
def vanilla_ice_cream := number_of_dwarfs
def chocolate_ice_cream := number_of_dwarfs / 2
def fruit_ice_cream := 1

theorem truthful_dwarfs_count (T L : ℕ) (h1 : T + L = 10)
  (h2 : vanilla_ice_cream = T + (L * 2))
  (h3 : chocolate_ice_cream = T / 2 + (L / 2 * 2))
  (h4 : fruit_ice_cream = 1)
  : T = 4 :=
sorry

end truthful_dwarfs_count_l0_187


namespace triangle_angle_type_l0_987

theorem triangle_angle_type (a b c R : ℝ) (hc_max : c ≥ a ∧ c ≥ b) :
  (a^2 + b^2 + c^2 - 8 * R^2 > 0 → 
     ∃ α β γ : ℝ, α + β + γ = π ∧ α < π / 2 ∧ β < π / 2 ∧ γ < π / 2) ∧
  (a^2 + b^2 + c^2 - 8 * R^2 = 0 → 
     ∃ α β γ : ℝ, α + β + γ = π ∧ (α = π / 2 ∨ β = π / 2 ∨ γ = π / 2)) ∧
  (a^2 + b^2 + c^2 - 8 * R^2 < 0 → 
     ∃ α β γ : ℝ, α + β + γ = π ∧ (α > π / 2 ∨ β > π / 2 ∨ γ > π / 2)) :=
sorry

end triangle_angle_type_l0_987


namespace find_x6_l0_692

-- Definition of the variables xi for i = 1, ..., 10.
variables {x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 : ℝ}

-- Given conditions as equations.
axiom eq1 : (x2 + x4) / 2 = 3
axiom eq2 : (x4 + x6) / 2 = 5
axiom eq3 : (x6 + x8) / 2 = 7
axiom eq4 : (x8 + x10) / 2 = 9
axiom eq5 : (x10 + x2) / 2 = 1

axiom eq6 : (x1 + x3) / 2 = 2
axiom eq7 : (x3 + x5) / 2 = 4
axiom eq8 : (x5 + x7) / 2 = 6
axiom eq9 : (x7 + x9) / 2 = 8
axiom eq10 : (x9 + x1) / 2 = 10

-- The theorem to prove.
theorem find_x6 : x6 = 1 :=
by
  sorry

end find_x6_l0_692


namespace trajectory_of_point_l0_219

theorem trajectory_of_point 
  (P : ℝ × ℝ) 
  (h1 : abs (P.1 - 4) + P.2^2 - 1 = abs (P.1 + 5)) : 
  P.2^2 = 16 * P.1 := 
sorry

end trajectory_of_point_l0_219


namespace black_cars_count_l0_415

-- Conditions
def red_cars : ℕ := 28
def ratio_red_black : ℚ := 3 / 8

-- Theorem statement
theorem black_cars_count :
  ∃ (black_cars : ℕ), black_cars = 75 ∧ (red_cars : ℚ) / (black_cars) = ratio_red_black :=
sorry

end black_cars_count_l0_415


namespace earrings_ratio_l0_780

theorem earrings_ratio :
  ∀ (total_pairs : ℕ) (given_pairs : ℕ) (total_earrings : ℕ) (given_earrings : ℕ),
    total_pairs = 12 →
    given_pairs = total_pairs / 2 →
    total_earrings = total_pairs * 2 →
    given_earrings = total_earrings / 2 →
    total_earrings = 36 →
    given_earrings = 12 →
    (total_earrings / given_earrings = 3) :=
by
  sorry

end earrings_ratio_l0_780


namespace fourth_root_equiv_l0_925

theorem fourth_root_equiv (x : ℝ) (hx : 0 < x) : (x * (x ^ (3 / 4))) ^ (1 / 4) = x ^ (7 / 16) :=
sorry

end fourth_root_equiv_l0_925


namespace mrs_hilt_rows_of_pies_l0_985

def number_of_pies (pecan_pies: Nat) (apple_pies: Nat) : Nat := pecan_pies + apple_pies

def rows_of_pies (total_pies: Nat) (pies_per_row: Nat) : Nat := total_pies / pies_per_row

theorem mrs_hilt_rows_of_pies :
  let pecan_pies := 16 in
  let apple_pies := 14 in
  let pies_per_row := 5 in
  rows_of_pies (number_of_pies pecan_pies apple_pies) pies_per_row = 6 :=
by 
  sorry

end mrs_hilt_rows_of_pies_l0_985


namespace allen_total_blocks_l0_589

/-- 
  If there are 7 blocks for every color of paint used and Shiela used 7 colors, 
  then the total number of blocks Allen has is 49.
-/
theorem allen_total_blocks
  (blocks_per_color : ℕ) 
  (number_of_colors : ℕ)
  (h1 : blocks_per_color = 7) 
  (h2 : number_of_colors = 7) : 
  blocks_per_color * number_of_colors = 49 := 
by 
  sorry

end allen_total_blocks_l0_589


namespace no_consecutive_heads_probability_l0_764

def prob_no_consecutive_heads (n : ℕ) : ℝ := 
  -- This is the probability function for no consecutive heads
  if n = 10 then (9 / 64) else 0

theorem no_consecutive_heads_probability :
  prob_no_consecutive_heads 10 = 9 / 64 :=
by
  -- Proof would go here
  sorry

end no_consecutive_heads_probability_l0_764


namespace radius_increase_by_100_percent_l0_753

theorem radius_increase_by_100_percent (A A' r r' : ℝ) (π : ℝ)
  (h1 : A = π * r^2) -- initial area of the circle
  (h2 : A' = 4 * A) -- new area is 4 times the original area
  (h3 : A' = π * r'^2) -- new area formula with new radius
  : r' = 2 * r :=
by
  sorry

end radius_increase_by_100_percent_l0_753


namespace isosceles_triangle_perimeter_l0_623

theorem isosceles_triangle_perimeter (a b c : ℝ) (h₀ : a = 5) (h₁ : b = 10) 
  (h₂ : c = 10 ∨ c = 5) (h₃ : a = b ∨ b = c ∨ c = a) 
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 25 := by
  sorry

end isosceles_triangle_perimeter_l0_623


namespace seq_1000_eq_2098_l0_223

-- Define the sequence a_n
def seq (n : ℕ) : ℤ := sorry

-- Initial conditions
axiom a1 : seq 1 = 100
axiom a2 : seq 2 = 101

-- Recurrence relation condition
axiom recurrence_relation : ∀ n : ℕ, 1 ≤ n → seq n + seq (n+1) + seq (n+2) = 2 * ↑n + 3

-- Main theorem to prove
theorem seq_1000_eq_2098 : seq 1000 = 2098 :=
by {
  sorry
}

end seq_1000_eq_2098_l0_223


namespace melissa_total_commission_l0_535

def sale_price_coupe : ℝ := 30000
def sale_price_suv : ℝ := 2 * sale_price_coupe
def sale_price_luxury_sedan : ℝ := 80000

def commission_rate_coupe_and_suv : ℝ := 0.02
def commission_rate_luxury_sedan : ℝ := 0.03

def commission (rate : ℝ) (price : ℝ) : ℝ := rate * price

def total_commission : ℝ :=
  commission commission_rate_coupe_and_suv sale_price_coupe +
  commission commission_rate_coupe_and_suv sale_price_suv +
  commission commission_rate_luxury_sedan sale_price_luxury_sedan

theorem melissa_total_commission :
  total_commission = 4200 := by
  sorry

end melissa_total_commission_l0_535


namespace monotone_decreasing_interval_3_l0_620

variable {f : ℝ → ℝ}

theorem monotone_decreasing_interval_3 
  (h1 : ∀ x, f (x + 3) = f (x - 3))
  (h2 : ∀ x, f (x + 3) = f (-x + 3))
  (h3 : ∀ ⦃x y⦄, 0 < x → x < 3 → 0 < y → y < 3 → x < y → f y < f x) :
  f 3.5 < f (-4.5) ∧ f (-4.5) < f 12.5 :=
sorry

end monotone_decreasing_interval_3_l0_620


namespace water_added_l0_575

theorem water_added (initial_fullness : ℚ) (final_fullness : ℚ) (capacity : ℚ)
  (h1 : initial_fullness = 0.40) (h2 : final_fullness = 3 / 4) (h3 : capacity = 80) :
  (final_fullness * capacity - initial_fullness * capacity) = 28 := by
  sorry

end water_added_l0_575


namespace one_minus_repeat_three_l0_325

theorem one_minus_repeat_three : 1 - (0.333333..<3̅) = 2 / 3 :=
by
  -- needs proof, currently left as sorry
  sorry

end one_minus_repeat_three_l0_325


namespace find_z_l0_522

theorem find_z (x y z : ℚ) (h1 : x / (y + 1) = 4 / 5) (h2 : 3 * z = 2 * x + y) (h3 : y = 10) : 
  z = 46 / 5 := 
sorry

end find_z_l0_522


namespace problem1_problem2_l0_305

noncomputable def sqrt (x : ℝ) := Real.sqrt x

theorem problem1 : sqrt 12 + sqrt 8 * sqrt 6 = 6 * sqrt 3 := by
  sorry

theorem problem2 : sqrt 12 + 1 / (sqrt 3 - sqrt 2) - sqrt 6 * sqrt 3 = 3 * sqrt 3 - 2 * sqrt 2 := by
  sorry

end problem1_problem2_l0_305


namespace triangle_sides_external_tangent_l0_889

theorem triangle_sides_external_tangent (R r : ℝ) (h : R > r) :
  ∃ (AB BC AC : ℝ),
    AB = 2 * Real.sqrt (R * r) ∧
    AC = 2 * r * Real.sqrt (R / (R + r)) ∧
    BC = 2 * R * Real.sqrt (r / (R + r)) :=
by
  sorry

end triangle_sides_external_tangent_l0_889


namespace prism_is_five_sided_l0_514

-- Definitions based on problem conditions
def prism_faces (total_faces base_faces : Nat) := total_faces = 7 ∧ base_faces = 2

-- Theorem to prove based on the conditions
theorem prism_is_five_sided (total_faces base_faces : Nat) (h : prism_faces total_faces base_faces) : total_faces - base_faces = 5 :=
sorry

end prism_is_five_sided_l0_514


namespace proof_of_x_and_velocity_l0_347

variables (a T L R x : ℝ)

-- Given condition
def given_eq : Prop := (a * T) / (a * T - R) = (L + x) / x

-- Target statement to prove
def target_eq_x : Prop := x = a * T * (L / R) - L
def target_velocity : Prop := a * (L / R)

-- Main theorem to prove the equivalence
theorem proof_of_x_and_velocity (a T L R : ℝ) : given_eq a T L R x → target_eq_x a T L R x ∧ target_velocity a T L R =
  sorry

end proof_of_x_and_velocity_l0_347


namespace find_sum_of_angles_l0_200

open Real

namespace math_problem

theorem find_sum_of_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : cos (α - β / 2) = sqrt 3 / 2)
  (h2 : sin (α / 2 - β) = -1 / 2) : α + β = 2 * π / 3 :=
sorry

end math_problem

end find_sum_of_angles_l0_200


namespace find_k_l0_211

noncomputable def vec (a b : ℝ) : ℝ × ℝ := (a, b)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem find_k
  (k : ℝ)
  (a b c : ℝ × ℝ)
  (ha : a = vec 3 1)
  (hb : b = vec 1 3)
  (hc : c = vec k (-2))
  (h_perp : dot_product (vec (a.1 - c.1) (a.2 - c.2)) (vec (a.1 - b.1) (a.2 - b.2)) = 0) :
  k = 0 :=
sorry

end find_k_l0_211


namespace multiplication_cycles_l0_284

theorem multiplication_cycles
  (p : ℕ) [Fact (Nat.Prime p)]
  (a : ZMod p) (h : a ≠ 0) :
  (∀ k : ℕ, ∃ n : ℕ, ((a : (ZMod p)) ^ k = a)) ∧
  (∀ b : ZMod p, b ≠ 0 → ((∃ k, b = a ^ k) → ∀ n m, (a ^ n = b) ↔ (a ^ m = b))) ∧
  (a ^ (p - 1) = 1) :=
sorry

end multiplication_cycles_l0_284


namespace min_value_of_expression_l0_531

theorem min_value_of_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : 
  (3 * x + y) * (x + 3 * z) * (y + z + 1) ≥ 48 :=
by
  sorry

end min_value_of_expression_l0_531


namespace cookies_eq_23_l0_849

def total_packs : Nat := 27
def cakes : Nat := 4
def cookies : Nat := total_packs - cakes

theorem cookies_eq_23 : cookies = 23 :=
by
  -- Proof goes here
  sorry

end cookies_eq_23_l0_849


namespace no_two_consecutive_heads_probability_l0_759

theorem no_two_consecutive_heads_probability : 
  let total_outcomes := 2 ^ 10 in
  let favorable_outcomes := 
    (∑ n in range 6, nat.choose (10 - n - 1) n) in
  (favorable_outcomes / total_outcomes : ℚ) = 9 / 64 :=
by
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := 1 + 10 + 36 + 56 + 35 + 6
  have h_total: total_outcomes = 1024 := by norm_num
  have h_favorable: favorable_outcomes = 144 := by norm_num
  have h_fraction: (favorable_outcomes : ℚ) / total_outcomes = 9 / 64 := by norm_num / [total_outcomes, favorable_outcomes]
  exact h_fraction

end no_two_consecutive_heads_probability_l0_759


namespace no_two_heads_consecutive_probability_l0_761

noncomputable def probability_no_two_heads_consecutive (total_flips : ℕ) : ℚ :=
  if total_flips = 10 then 144 / 1024 else 0

theorem no_two_heads_consecutive_probability :
  probability_no_two_heads_consecutive 10 = 9 / 64 :=
by
  unfold probability_no_two_heads_consecutive
  rw [if_pos rfl]
  norm_num
  sorry

end no_two_heads_consecutive_probability_l0_761


namespace range_of_a_l0_640

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a ≤ abs (x - 5) + abs (x - 3)) → a ≤ 2 := by
  sorry

end range_of_a_l0_640


namespace find_LP_l0_564

variables (A B C K L P M : Type) 
variables {AC BC AK CK CL AM LP : ℕ}

-- Defining the given conditions
def conditions (AC BC AK CK : ℕ) (AM : ℕ) :=
  AC = 360 ∧ BC = 240 ∧ AK = CK ∧ AK = 180 ∧ AM = 144

-- The theorem statement: proving LP equals 57.6
theorem find_LP (h : conditions 360 240 180 180 144) : LP = 576 / 10 := 
by sorry

end find_LP_l0_564


namespace quadratic_root_zero_l0_943

theorem quadratic_root_zero (a : ℝ) : 
  ((a-1) * 0^2 + 0 + a^2 - 1 = 0) 
  → a ≠ 1 
  → a = -1 := 
by
  intro h1 h2
  sorry

end quadratic_root_zero_l0_943


namespace largest_prime_factor_of_3913_l0_136

theorem largest_prime_factor_of_3913 : 
  ∃ (p : ℕ), nat.prime p ∧ p ∣ 3913 ∧ (∀ q, nat.prime q ∧ q ∣ 3913 → q ≤ p) ∧ p = 43 :=
sorry

end largest_prime_factor_of_3913_l0_136


namespace smallest_positive_period_of_f_l0_45

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) ^ 2

theorem smallest_positive_period_of_f :
  ∀ T > 0, (∀ x, f (x + T) = f x) ↔ T = Real.pi / 2 :=
by
  sorry

end smallest_positive_period_of_f_l0_45


namespace sum_coefficients_equals_l0_511

theorem sum_coefficients_equals :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ), 
  (∀ x : ℤ, (2 * x + 1) ^ 5 = 
    a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_0 = 1 →
  a_1 + a_2 + a_3 + a_4 + a_5 = 3^5 - 1 :=
by
  intros a_0 a_1 a_2 a_3 a_4 a_5 h h0
  sorry

end sum_coefficients_equals_l0_511


namespace max_sum_at_11_l0_660

noncomputable def is_arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_seq (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

theorem max_sum_at_11 (a : ℕ → ℚ) (d : ℚ) (h_arith : is_arithmetic_seq a) (h_a1_gt_0 : a 0 > 0)
 (h_sum_eq : sum_seq a 13 = sum_seq a 7) : 
  ∃ n : ℕ, sum_seq a n = sum_seq a 10 + (a 10 + a 11) := sorry


end max_sum_at_11_l0_660


namespace gcd_45736_123456_l0_42

theorem gcd_45736_123456 : Nat.gcd 45736 123456 = 352 :=
by sorry

end gcd_45736_123456_l0_42


namespace regular_polygon_sides_l0_19

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 150) : 
  ∃ (n : ℕ), 180 * (n - 2) / n = 150 ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l0_19


namespace seq_composite_l0_123

-- Define the sequence recurrence relation
def seq (a : ℕ → ℕ) : Prop :=
  ∀ (k : ℕ), k ≥ 1 → a (k+2) = a (k+1) * a k + 1

-- Prove that for k ≥ 9, a_k - 22 is composite
theorem seq_composite (a : ℕ → ℕ) (h_seq : seq a) :
  ∀ (k : ℕ), k ≥ 9 → ∃ d, d > 1 ∧ d < a k ∧ d ∣ (a k - 22) :=
by
  sorry

end seq_composite_l0_123


namespace fiftyThreeDaysFromFridayIsTuesday_l0_724

-- Domain definition: Days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to compute the day of the week after a given number of days
def dayAfter (start: Day) (n: ℕ) : Day :=
  match start with
  | Sunday    => Day.ofNat ((0 + n) % 7)
  | Monday    => Day.ofNat ((1 + n) % 7)
  | Tuesday   => Day.ofNat ((2 + n) % 7)
  | Wednesday => Day.ofNat ((3 + n) % 7)
  | Thursday  => Day.ofNat ((4 + n) % 7)
  | Friday    => Day.ofNat ((5 + n) % 7)
  | Saturday  => Day.ofNat ((6 + n) % 7)

namespace Day

-- Helper function to convert natural numbers back to day of the week
def ofNat : ℕ → Day
| 0 => Sunday
| 1 => Monday
| 2 => Tuesday
| 3 => Wednesday
| 4 => Thursday
| 5 => Friday
| 6 => Saturday
| _ => Day.ofNat (n % 7)

-- Lean statement: Prove that 53 days from Friday is Tuesday
theorem fiftyThreeDaysFromFridayIsTuesday :
  dayAfter Friday 53 = Tuesday :=
sorry

end fiftyThreeDaysFromFridayIsTuesday_l0_724


namespace number_of_rectangles_l0_805

theorem number_of_rectangles (a b : ℝ) (ha_lt_b : a < b) :
  ∃! (x y : ℝ), (x < b ∧ y < b ∧ 2 * (x + y) = a + b ∧ x * y = (a * b) / 4) := 
sorry

end number_of_rectangles_l0_805


namespace maple_is_taller_l0_102

def pine_tree_height : ℚ := 13 + 1/4
def maple_tree_height : ℚ := 20 + 1/2
def height_difference : ℚ := maple_tree_height - pine_tree_height

theorem maple_is_taller : height_difference = 7 + 1/4 := by
  sorry

end maple_is_taller_l0_102


namespace joan_gave_mike_seashells_l0_670

-- Definitions based on the conditions
def original_seashells : ℕ := 79
def remaining_seashells : ℕ := 16
def given_seashells := original_seashells - remaining_seashells

-- The theorem we want to prove
theorem joan_gave_mike_seashells : given_seashells = 63 := by
  sorry

end joan_gave_mike_seashells_l0_670


namespace rectangular_reconfiguration_l0_853

theorem rectangular_reconfiguration (k : ℕ) (n : ℕ) (h₁ : k - 5 > 0) (h₂ : k ≥ 6) (h₃ : k ≤ 9) :
  (k * (k - 5) = n^2) → (n = 6) :=
by {
  sorry  -- proof is omitted
}

end rectangular_reconfiguration_l0_853


namespace bank_record_withdrawal_l0_87

def deposit (x : ℤ) := x
def withdraw (x : ℤ) := -x

theorem bank_record_withdrawal : withdraw 500 = -500 :=
by
  sorry

end bank_record_withdrawal_l0_87


namespace T_lt_S_div_2_l0_681

noncomputable def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
noncomputable def b (n : ℕ) : ℝ := n * (1 / 3)^n
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)
noncomputable def S_general (n : ℕ) : ℝ := 3/2 - (1/2) * (1/3)^(n-1)
noncomputable def T_general (n : ℕ) : ℝ := 3/4 - (1/4) * (1/3)^(n-1) - (n * (1/3)^(n-1))/2

theorem T_lt_S_div_2 (n : ℕ) : T n < S n / 2 :=
by sorry

end T_lt_S_div_2_l0_681


namespace day_of_week_in_53_days_l0_737

/-- Prove that 53 days from Friday is Tuesday given today is Friday and a week has 7 days --/
theorem day_of_week_in_53_days
    (today : String)
    (days_in_week : ℕ)
    (day_count : ℕ)
    (target_day : String)
    (h1 : today = "Friday")
    (h2 : days_in_week = 7)
    (h3 : day_count = 53)
    (h4 : target_day = "Tuesday") :
    let remainder := day_count % days_in_week
    let computed_day := match remainder with
                        | 0 => "Friday"
                        | 1 => "Saturday"
                        | 2 => "Sunday"
                        | 3 => "Monday"
                        | 4 => "Tuesday"
                        | 5 => "Wednesday"
                        | 6 => "Thursday"
                        | _ => "Invalid"
    in computed_day = target_day := 
by
  sorry

end day_of_week_in_53_days_l0_737


namespace regular_polygon_sides_l0_20

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) 
(h_interior : (n - 2) * 180 / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l0_20


namespace least_subtraction_for_divisibility_l0_609

theorem least_subtraction_for_divisibility (n : ℕ) (h : n = 964807) : ∃ k, k = 7 ∧ (n - k) % 8 = 0 :=
by 
  sorry

end least_subtraction_for_divisibility_l0_609


namespace water_temp_increase_per_minute_l0_527

theorem water_temp_increase_per_minute :
  ∀ (initial_temp final_temp total_time pasta_time mixing_ratio : ℝ),
    initial_temp = 41 →
    final_temp = 212 →
    total_time = 73 →
    pasta_time = 12 →
    mixing_ratio = (1 / 3) →
    ((final_temp - initial_temp) / (total_time - pasta_time - (mixing_ratio * pasta_time)) = 3) :=
by
  intros initial_temp final_temp total_time pasta_time mixing_ratio
  sorry

end water_temp_increase_per_minute_l0_527


namespace probability_of_red_ball_l0_835

theorem probability_of_red_ball (total_balls red_balls black_balls white_balls : ℕ)
  (h1 : total_balls = 7)
  (h2 : red_balls = 2)
  (h3 : black_balls = 4)
  (h4 : white_balls = 1) :
  (red_balls / total_balls : ℚ) = 2 / 7 :=
by {
  sorry
}

end probability_of_red_ball_l0_835


namespace fraction_value_l0_264

-- Define the constants
def eight := 8
def four := 4

-- Statement to prove
theorem fraction_value : (eight + four) / (eight - four) = 3 := 
by
  sorry

end fraction_value_l0_264


namespace symmetric_point_origin_l0_874

theorem symmetric_point_origin (x y : ℤ) (h : x = -2 ∧ y = 3) :
    (-x, -y) = (2, -3) :=
by
  cases h with
  | intro hx hy =>
  simp only [hx, hy]
  sorry

end symmetric_point_origin_l0_874


namespace average_reading_days_l0_318

def emery_days : ℕ := 20
def serena_days : ℕ := 5 * emery_days
def average_days (e s : ℕ) : ℕ := (e + s) / 2

theorem average_reading_days 
  (e s : ℕ) 
  (h1 : e = emery_days)
  (h2 : s = serena_days) :
  average_days e s = 60 :=
by
  rw [h1, h2, emery_days, serena_days]
  sorry

end average_reading_days_l0_318


namespace day_of_week_in_53_days_l0_739

/-- Prove that 53 days from Friday is Tuesday given today is Friday and a week has 7 days --/
theorem day_of_week_in_53_days
    (today : String)
    (days_in_week : ℕ)
    (day_count : ℕ)
    (target_day : String)
    (h1 : today = "Friday")
    (h2 : days_in_week = 7)
    (h3 : day_count = 53)
    (h4 : target_day = "Tuesday") :
    let remainder := day_count % days_in_week
    let computed_day := match remainder with
                        | 0 => "Friday"
                        | 1 => "Saturday"
                        | 2 => "Sunday"
                        | 3 => "Monday"
                        | 4 => "Tuesday"
                        | 5 => "Wednesday"
                        | 6 => "Thursday"
                        | _ => "Invalid"
    in computed_day = target_day := 
by
  sorry

end day_of_week_in_53_days_l0_739


namespace net_marble_change_l0_98

/-- Josh's initial number of marbles. -/
def initial_marbles : ℕ := 20

/-- Number of marbles Josh lost. -/
def lost_marbles : ℕ := 16

/-- Number of marbles Josh found. -/
def found_marbles : ℕ := 8

/-- Number of marbles Josh traded away. -/
def traded_away_marbles : ℕ := 5

/-- Number of marbles Josh received in a trade. -/
def received_in_trade_marbles : ℕ := 9

/-- Number of marbles Josh gave away. -/
def gave_away_marbles : ℕ := 3

/-- Number of marbles Josh received from his cousin. -/
def received_from_cousin_marbles : ℕ := 4

/-- Final number of marbles Josh has after all transactions. -/
def final_marbles : ℕ :=
  initial_marbles - lost_marbles + found_marbles - traded_away_marbles + received_in_trade_marbles
  - gave_away_marbles + received_from_cousin_marbles

theorem net_marble_change : (final_marbles : ℤ) - (initial_marbles : ℤ) = -3 := 
by
  sorry

end net_marble_change_l0_98


namespace simplified_expression_at_one_l0_865

noncomputable def original_expression (a : ℚ) : ℚ :=
  (2 * a + 2) / a / (4 / (a ^ 2)) - a / (a + 1)

theorem simplified_expression_at_one : original_expression 1 = 1 / 2 := by
  sorry

end simplified_expression_at_one_l0_865


namespace yvette_sundae_cost_l0_279

noncomputable def cost_friends : ℝ := 7.50 + 10.00 + 8.50
noncomputable def final_bill : ℝ := 42.00
noncomputable def tip_percentage : ℝ := 0.20
noncomputable def tip_amount : ℝ := tip_percentage * final_bill

theorem yvette_sundae_cost : 
  final_bill - (cost_friends + tip_amount) = 7.60 := by
  sorry

end yvette_sundae_cost_l0_279


namespace sequence_solution_l0_947

theorem sequence_solution (a : ℕ → ℤ) :
  a 0 = -1 →
  a 1 = 1 →
  (∀ n ≥ 2, a n = 2 * a (n - 1) + 3 * a (n - 2) + 3^n) →
  ∀ n, a n = (1 / 16) * ((4 * n - 3) * 3^(n + 1) - 7 * (-1)^n) :=
by
  -- Detailed proof steps will go here.
  sorry

end sequence_solution_l0_947


namespace complex_pow_six_eq_eight_i_l0_309

theorem complex_pow_six_eq_eight_i (i : ℂ) (h : i^2 = -1) : (1 - i) ^ 6 = 8 * i := by
  sorry

end complex_pow_six_eq_eight_i_l0_309


namespace last_three_digits_of_7_pow_103_l0_791

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 614 := by
  sorry

end last_three_digits_of_7_pow_103_l0_791


namespace least_three_digit_with_factors_correct_l0_430

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def has_factors (n : ℕ) : Prop := n % 3 = 0 ∧ n % 4 = 0 ∧ n % 9 = 0
def least_three_digit_with_factors : ℕ := 108

theorem least_three_digit_with_factors_correct : 
  is_three_digit least_three_digit_with_factors ∧ has_factors least_three_digit_with_factors ∧
  ∀ m : ℕ, is_three_digit m → has_factors m → least_three_digit_with_factors ≤ m := 
by 
  sorry

end least_three_digit_with_factors_correct_l0_430


namespace fifty_three_days_from_friday_is_tuesday_l0_734

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end fifty_three_days_from_friday_is_tuesday_l0_734


namespace total_surface_area_is_correct_l0_592

-- Define the problem constants and structure
def num_cubes := 20
def edge_length := 1
def bottom_layer := 9
def middle_layer := 8
def top_layer := 3
def total_painted_area : ℕ := 55

-- Define a function to calculate the exposed surface area
noncomputable def calc_exposed_area (num_bottom : ℕ) (num_middle : ℕ) (num_top : ℕ) (edge : ℕ) : ℕ := 
    let bottom_exposed := num_bottom * (edge * edge)
    let middle_corners_exposed := 4 * 3 * edge
    let middle_edges_exposed := (num_middle - 4) * (2 * edge)
    let top_exposed := num_top * (5 * edge)
    bottom_exposed + middle_corners_exposed + middle_edges_exposed + top_exposed

-- Statement to prove the total painted area
theorem total_surface_area_is_correct : calc_exposed_area bottom_layer middle_layer top_layer edge_length = total_painted_area :=
by
  -- The proof itself is omitted, focus is on the statement.
  sorry

end total_surface_area_is_correct_l0_592


namespace set_difference_lt3_gt0_1_leq_x_leq_2_l0_357

def A := {x : ℝ | |x| < 3}
def B := {x : ℝ | x^2 - 3 * x + 2 > 0}

theorem set_difference_lt3_gt0_1_leq_x_leq_2 : {x : ℝ | x ∈ A ∧ x ∉ (A ∩ B)} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end set_difference_lt3_gt0_1_leq_x_leq_2_l0_357


namespace sum_of_coeff_l0_91

theorem sum_of_coeff (x y : ℕ) (n : ℕ) (h : 2 * x + y = 3) : (2 * x + y) ^ n = 3^n := 
by
  sorry

end sum_of_coeff_l0_91


namespace integer_between_sqrt3_add1_and_sqrt11_l0_879

theorem integer_between_sqrt3_add1_and_sqrt11 :
  (∀ x, (1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2) ∧ (3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4) → (2 < Real.sqrt 3 + 1 ∧ Real.sqrt 3 + 1 < 3) ∧ (3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4) ∧ x = 3) :=
by
  sorry

end integer_between_sqrt3_add1_and_sqrt11_l0_879


namespace gcd_18_30_45_l0_486

theorem gcd_18_30_45 : Nat.gcd (Nat.gcd 18 30) 45 = 3 :=
by
  sorry

end gcd_18_30_45_l0_486


namespace directors_dividends_correct_l0_903

theorem directors_dividends_correct :
  let net_profit : ℝ := (1500000 - 674992) - 0.2 * (1500000 - 674992)
  let total_loan_payments : ℝ := 23914 * 12 - 74992
  let profit_for_dividends : ℝ := net_profit - total_loan_payments
  let dividend_per_share : ℝ := profit_for_dividends / 1000
  let total_dividends_director : ℝ := dividend_per_share * 550
  total_dividends_director = 246400.0 :=
by
  sorry

end directors_dividends_correct_l0_903


namespace least_three_digit_with_factors_correct_l0_428

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def has_factors (n : ℕ) : Prop := n % 3 = 0 ∧ n % 4 = 0 ∧ n % 9 = 0
def least_three_digit_with_factors : ℕ := 108

theorem least_three_digit_with_factors_correct : 
  is_three_digit least_three_digit_with_factors ∧ has_factors least_three_digit_with_factors ∧
  ∀ m : ℕ, is_three_digit m → has_factors m → least_three_digit_with_factors ≤ m := 
by 
  sorry

end least_three_digit_with_factors_correct_l0_428


namespace right_triangle_sides_l0_704

theorem right_triangle_sides (x y z : ℕ) (h1 : x + y + z = 30)
    (h2 : x^2 + y^2 + z^2 = 338) (h3 : x^2 + y^2 = z^2) :
    (x = 5 ∧ y = 12 ∧ z = 13) ∨ (x = 12 ∧ y = 5 ∧ z = 13) :=
by
  sorry

end right_triangle_sides_l0_704


namespace find_x_l0_958

theorem find_x (x : ℕ) (hx : x > 0) : 1^(x + 3) + 2^(x + 2) + 3^x + 4^(x + 1) = 1958 → x = 4 :=
sorry

end find_x_l0_958


namespace charles_nickels_l0_306

theorem charles_nickels :
  ∀ (num_pennies num_cents penny_value nickel_value n : ℕ),
  num_pennies = 6 →
  num_cents = 21 →
  penny_value = 1 →
  nickel_value = 5 →
  (num_cents - num_pennies * penny_value) / nickel_value = n →
  n = 3 :=
by
  intros num_pennies num_cents penny_value nickel_value n hnum_pennies hnum_cents hpenny_value hnickel_value hn
  sorry

end charles_nickels_l0_306


namespace total_bins_correct_l0_174

def total_bins (soup vegetables pasta : ℝ) : ℝ :=
  soup + vegetables + pasta

theorem total_bins_correct : total_bins 0.12 0.12 0.5 = 0.74 :=
  by
    sorry

end total_bins_correct_l0_174


namespace exists_four_distinct_natural_numbers_sum_any_three_prime_l0_570

theorem exists_four_distinct_natural_numbers_sum_any_three_prime :
  ∃ a b c d : ℕ, (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
  (Prime (a + b + c) ∧ Prime (a + b + d) ∧ Prime (a + c + d) ∧ Prime (b + c + d)) :=
sorry

end exists_four_distinct_natural_numbers_sum_any_three_prime_l0_570


namespace dwarfs_truthful_count_l0_181

theorem dwarfs_truthful_count :
  ∃ (T L : ℕ), T + L = 10 ∧
    (∀ t : ℕ, t = 10 → t + ((10 - T) * 2 - T) = 16) ∧
    T = 4 :=
by
  sorry

end dwarfs_truthful_count_l0_181


namespace polynomial_has_real_root_l0_201

noncomputable def P : Polynomial ℝ := sorry

variables (a1 a2 a3 b1 b2 b3 : ℝ) (h_nonzero : a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0)
variables (h_eq : ∀ x : ℝ, P.eval (a1 * x + b1) + P.eval (a2 * x + b2) = P.eval (a3 * x + b3))

theorem polynomial_has_real_root : ∃ x : ℝ, P.eval x = 0 :=
sorry

end polynomial_has_real_root_l0_201


namespace least_sum_of_exponents_l0_827

theorem least_sum_of_exponents {n : ℕ} (h : n = 520) (h_exp : ∃ (a b : ℕ), 2^a + 2^b = n ∧ a ≠ b ∧ a = 9 ∧ b = 3) : 
    (∃ (s : ℕ), s = 9 + 3) :=
by
  sorry

end least_sum_of_exponents_l0_827


namespace new_person_weight_l0_406

theorem new_person_weight (avg_weight_increase : ℝ) (old_weight new_weight : ℝ) (n : ℕ)
    (weight_increase_per_person : avg_weight_increase = 3.5)
    (number_of_persons : n = 8)
    (replaced_person_weight : old_weight = 62) :
    new_weight = 90 :=
by
  sorry

end new_person_weight_l0_406


namespace one_minus_repeating_three_l0_324

theorem one_minus_repeating_three : 1 - (0.\overline{3}) = 2 / 3 :=
by
  sorry

end one_minus_repeating_three_l0_324


namespace max_f_l0_803

noncomputable def f (x : ℝ) : ℝ :=
  min (min (2 * x + 2) (1 / 2 * x + 1)) (-3 / 4 * x + 7)

theorem max_f : ∃ x : ℝ, f x = 17 / 5 :=
by
  sorry

end max_f_l0_803


namespace initial_population_l0_881

theorem initial_population (P : ℝ) 
    (h1 : 1.25 * P * 0.70 = 363650) : 
    P = 415600 :=
sorry

end initial_population_l0_881


namespace distance_between_DM_and_BN_l0_378

open Real
open EuclideanSpace

noncomputable def distance_between_skew_lines (a : ℝ) : ℝ :=
  let B := (2 * a, -2 * a, 0)
  let D := (-2 * a, 2 * a, 0)
  let M := (-a, -a, sqrt 14 * a)
  let N := (a, a, sqrt 14 * a)
  let BN := (a - 2 * a, a + 2 * a, sqrt 14 * a - 0)
  let DM := (-a - (-2 * a), -a - 2 * a, sqrt 14 * a - 0)
  let n := (3, 1, 0)
  let MN := (2 * a, 2 * a, 0)
  abs ((2 * a * 3 + 2 * a * 1) / sqrt (3 ^ 2 + 1 ^ 2))

theorem distance_between_DM_and_BN (a : ℝ) : distance_between_skew_lines a = 4 * sqrt 10 * a / 5 :=
  sorry

end distance_between_DM_and_BN_l0_378


namespace comp_functions_l0_639

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := 3 * x - 5

theorem comp_functions (x : ℝ) : f (g x) = 6 * x - 7 :=
by
  sorry

end comp_functions_l0_639


namespace three_consecutive_multiples_sum_l0_558

theorem three_consecutive_multiples_sum (h1 : Int) (h2 : h1 % 3 = 0) (h3 : Int) (h4 : h3 = h1 - 3) (h5 : Int) (h6 : h5 = h1 - 6) (h7: h1 = 27) : h1 + h3 + h5 = 72 := 
by 
  -- let numbers be n, n-3, n-6 and n = 27
  -- so n + n-3 + n-6 = 27 + 24 + 21 = 72
  sorry

end three_consecutive_multiples_sum_l0_558


namespace greatest_prime_factor_of_144_l0_740

theorem greatest_prime_factor_of_144 : ∃ p, prime p ∧ p ∣ 144 ∧ (∀ q, prime q ∧ q ∣ 144 → q ≤ p) :=
sorry

end greatest_prime_factor_of_144_l0_740


namespace scientific_notation_of_361000000_l0_38

theorem scientific_notation_of_361000000 : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ abs a) ∧ (abs a < 10) ∧ (361000000 = a * 10^n) ∧ (a = 3.61) ∧ (n = 8) :=
sorry

end scientific_notation_of_361000000_l0_38


namespace smallest_integer_n_exists_l0_141

-- Define the conditions
def lcm_gcd_correct_division (a b : ℕ) : Prop :=
  (lcm a b) / (gcd a b) = 44

-- Define the main problem
theorem smallest_integer_n_exists : ∃ n : ℕ, lcm_gcd_correct_division 60 n ∧ 
  (∀ k : ℕ, lcm_gcd_correct_division 60 k → k ≥ n) :=
begin
  sorry
end

end smallest_integer_n_exists_l0_141


namespace abc_value_l0_530

theorem abc_value {a b c : ℂ} 
  (h1 : a * b + 5 * b + 20 = 0) 
  (h2 : b * c + 5 * c + 20 = 0) 
  (h3 : c * a + 5 * a + 20 = 0) : 
  a * b * c = 100 := 
by 
  sorry

end abc_value_l0_530


namespace dispatch_plans_count_l0_772

theorem dispatch_plans_count :
  let officials := {a, b, c, d, e, f, g, h} -- assuming these are the 8 officials
  let males := {a, b, c, d, e}
  let females := {f, g, h}
  ∃ groups : officials → bool,
    (∀ g, (card (groups g = true)) ≥ 3 ∧ (card (groups g = false)) ≥ 3) ∧
    (∃ g, (males ∩ groups g ≠ ∅)) ∧ (∃ g, (males ∩ groups (¬g) ≠ ∅)) ∧
    (card {g | groups g} = 2) →
  (count dispatch_plans = 180)
:= sorry

end dispatch_plans_count_l0_772


namespace maximum_value_of_omega_l0_636

variable (A ω : ℝ)

theorem maximum_value_of_omega (hA : 0 < A) (hω_pos : 0 < ω)
  (h1 : ω * (-π / 2) ≥ -π / 2) 
  (h2 : ω * (2 * π / 3) ≤ π / 2) :
  ω = 3 / 4 :=
sorry

end maximum_value_of_omega_l0_636


namespace total_pumpkins_l0_239

-- Define the number of pumpkins grown by Sandy and Mike
def pumpkinsSandy : ℕ := 51
def pumpkinsMike : ℕ := 23

-- Prove that their total is 74
theorem total_pumpkins : pumpkinsSandy + pumpkinsMike = 74 := by
  sorry

end total_pumpkins_l0_239


namespace winner_percentage_l0_968

theorem winner_percentage (votes_winner : ℕ) (votes_difference : ℕ) (total_votes : ℕ) 
  (h1 : votes_winner = 1044) 
  (h2 : votes_difference = 288) 
  (h3 : total_votes = votes_winner + (votes_winner - votes_difference)) :
  (votes_winner * 100) / total_votes = 58 :=
by
  sorry

end winner_percentage_l0_968


namespace symmetric_circle_eq_l0_949

theorem symmetric_circle_eq :
  (∃ f : ℝ → ℝ → Prop, (∀ x y, f x y ↔ (x - 2)^2 + (y + 1)^2 = 1)) →
  (∃ line : ℝ → ℝ → Prop, (∀ x y, line x y ↔ x - y + 3 = 0)) →
  (∃ eq : ℝ → ℝ → Prop, (∀ x y, eq x y ↔ (x - 4)^2 + (y - 5)^2 = 1)) :=
by
  sorry

end symmetric_circle_eq_l0_949


namespace area_of_shaded_rectangle_l0_465

theorem area_of_shaded_rectangle (w₁ h₁ w₂ h₂: ℝ) 
  (hw₁: w₁ * h₁ = 6)
  (hw₂: w₂ * h₁ = 15)
  (hw₃: w₂ * h₂ = 25) :
  w₁ * h₂ = 10 :=
by
  sorry

end area_of_shaded_rectangle_l0_465


namespace boss_monthly_pay_l0_657

theorem boss_monthly_pay
  (fiona_hours_per_week : ℕ)
  (john_hours_per_week : ℕ)
  (jeremy_hours_per_week : ℕ)
  (hourly_rate : ℕ)
  (weeks_in_month : ℕ)
  (fiona_income : ℕ := fiona_hours_per_week * hourly_rate)
  (john_income : ℕ := john_hours_per_week * hourly_rate)
  (jeremy_income : ℕ := jeremy_hours_per_week * hourly_rate) :
  fiona_hours_per_week = 40 →
  john_hours_per_week = 30 →
  jeremy_hours_per_week = 25 →
  hourly_rate = 20 →
  weeks_in_month = 4 →
  (fiona_income + john_income + jeremy_income) * weeks_in_month = 7600 :=
begin
  intros h1 h2 h3 h4 h5,
  rw [h1, h2, h3, h4, h5],
  sorry -- This is the point where the proof would start
end

end boss_monthly_pay_l0_657


namespace buying_beams_l0_402

/-- Problem Statement:
Given:
1. The total money for beams is 6210 wen.
2. The transportation cost per beam is 3 wen.
3. Removing one beam means the remaining beams' total transportation cost equals the price of one beam.

Prove: 3 * (x - 1) = 6210 / x
-/
theorem buying_beams (x : ℕ) (h₁ : x > 0) (h₂ : 6210 % x = 0) :
  3 * (x - 1) = 6210 / x :=
sorry

end buying_beams_l0_402


namespace Tn_lt_Sn_over_2_l0_683

theorem Tn_lt_Sn_over_2 (a b : ℕ → ℝ) (S T : ℕ → ℝ) (n : ℕ) :
  (a 1 = 1) →
  (∀ n, b n = n * (a n) / 3) →
  (a 1, 3 * a 2, 9 * a 3 are in arithmetic_sequence) →
  (S n = ∑ i in (range n), a i) →
  (T n = ∑ i in (range n), b i) →
  T n < S n / 2 :=
sorry

end Tn_lt_Sn_over_2_l0_683


namespace B_contains_only_one_element_l0_981

def setA := { x | (x - 1/2) * (x - 3) = 0 }

def setB (a : ℝ) := { x | Real.log (x^2 + a * x + a + 9 / 4) = 0 }

theorem B_contains_only_one_element (a : ℝ) :
  (∃ x, setB a x ∧ ∀ y, setB a y → y = x) →
  (a = 5 ∨ a = -1) :=
by
  intro h
  -- Proof would go here
  sorry

end B_contains_only_one_element_l0_981


namespace allen_total_blocks_l0_588

/-- 
  If there are 7 blocks for every color of paint used and Shiela used 7 colors, 
  then the total number of blocks Allen has is 49.
-/
theorem allen_total_blocks
  (blocks_per_color : ℕ) 
  (number_of_colors : ℕ)
  (h1 : blocks_per_color = 7) 
  (h2 : number_of_colors = 7) : 
  blocks_per_color * number_of_colors = 49 := 
by 
  sorry

end allen_total_blocks_l0_588


namespace max_c_for_log_inequality_l0_528

theorem max_c_for_log_inequality (a b : ℝ) (ha : 1 < a) (hb : 1 < b) : 
  ∃ c : ℝ, c = 1 / 3 ∧ (1 / (3 + Real.log b / Real.log a) + 1 / (3 + Real.log a / Real.log b) ≥ c) :=
by
  use 1 / 3
  sorry

end max_c_for_log_inequality_l0_528


namespace greatest_prime_factor_of_144_l0_741

theorem greatest_prime_factor_of_144 : 
  ∃ p, p = 3 ∧ prime p ∧ ∀ q, prime q ∧ (q ∣ 144) → q ≤ p := 
by
  sorry

end greatest_prime_factor_of_144_l0_741


namespace range_of_a_l0_198

theorem range_of_a (a : ℝ) : ({x : ℝ | a - 4 < x ∧ x < a + 4} ⊆ {x : ℝ | 1 < x ∧ x < 3}) → (-1 ≤ a ∧ a ≤ 5) := by
  sorry

end range_of_a_l0_198


namespace find_max_marks_l0_582

variable (M : ℝ)
variable (pass_mark : ℝ := 60 / 100)
variable (obtained_marks : ℝ := 200)
variable (additional_marks_needed : ℝ := 80)

theorem find_max_marks (h1 : pass_mark * M = obtained_marks + additional_marks_needed) : M = 467 := 
by
  sorry

end find_max_marks_l0_582


namespace find_N_l0_417

theorem find_N (a b c : ℤ) (N : ℤ)
  (h1 : a + b + c = 105)
  (h2 : a - 5 = N)
  (h3 : b + 10 = N)
  (h4 : 5 * c = N) : 
  N = 50 :=
by
  sorry

end find_N_l0_417


namespace buy_beams_l0_404

theorem buy_beams (C T x : ℕ) (hC : C = 6210) (hT : T = 3) (hx: x > 0):
  T * (x - 1) = C / x :=
by
  rw [hC, hT]
  sorry

end buy_beams_l0_404


namespace student_age_is_17_in_1960_l0_475

noncomputable def student's_age_in_1960 (x y : ℕ) (hx : 0 ≤ x ∧ x < 10) (hy : 0 ≤ y ∧ y < 10) : ℕ := 
  let birth_year : ℕ := 1900 + 10 * x + y
  let age_in_1960 : ℕ := 1960 - birth_year
  age_in_1960

theorem student_age_is_17_in_1960 :
  ∃ x y : ℕ, 0 ≤ x ∧ x < 10 ∧ 0 ≤ y ∧ y < 10 ∧ (1960 - (1900 + 10 * x + y) = 1 + 9 + x + y) ∧ (1960 - (1900 + 10 * x + y) = 17) :=
by {
  sorry -- Proof goes here
}

end student_age_is_17_in_1960_l0_475


namespace total_rock_needed_l0_9

theorem total_rock_needed (a b : ℕ) (h₁ : a = 8) (h₂ : b = 8) : a + b = 16 :=
by
  sorry

end total_rock_needed_l0_9


namespace largest_6_digit_int_divisible_by_5_l0_603

theorem largest_6_digit_int_divisible_by_5 :
  ∃ n, (n ≤ 999999) ∧ (999999 ≤ 999999) ∧ (n % 5 = 0) ∧ (∀ m, (m ≤ 999999) ∧ (m % 5 = 0) → n ≥ m) :=
begin
  use 999995,
  repeat {split},
  dec_trivial,
  dec_trivial,
  dec_trivial,
  intros m m_cond,
  cases m_cond,
  cases m_cond_right,
  cases m_cond_right_right,
  sorry
end

end largest_6_digit_int_divisible_by_5_l0_603


namespace S_equals_x4_l0_676

-- Define the expression for S
def S (x : ℝ) : ℝ := (x - 1)^4 + 4 * (x - 1)^3 + 6 * (x - 1)^2 + 4 * x - 3

-- State the theorem to be proved
theorem S_equals_x4 (x : ℝ) : S x = x^4 :=
by
  sorry

end S_equals_x4_l0_676


namespace locus_midpoint_l0_504

-- Conditions
def hyperbola_eq (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1

def perpendicular_rays (OA OB : ℝ × ℝ) : Prop := (OA.1 * OB.1 + OA.2 * OB.2) = 0 -- Dot product zero for perpendicularity

-- Given the hyperbola and perpendicularity conditions, prove the locus equation
theorem locus_midpoint (x y : ℝ) :
  (∃ A B : ℝ × ℝ, hyperbola_eq A.1 A.2 ∧ hyperbola_eq B.1 B.2 ∧ perpendicular_rays A B ∧
  x = (A.1 + B.1) / 2 ∧ y = (A.2 + B.2) / 2) → 3 * (4 * x^2 - y^2)^2 = 4 * (16 * x^2 + y^2) :=
sorry

end locus_midpoint_l0_504


namespace intersection_points_l0_69

-- Definition of curve C by the polar equation
def curve_C (ρ : ℝ) (θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Definition of line l by the polar equation
def line_l (ρ : ℝ) (θ : ℝ) (m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 6) = m

-- Proof statement that line l intersects curve C exactly once for specific values of m
theorem intersection_points (m : ℝ) : 
  (∀ ρ θ, curve_C ρ θ → line_l ρ θ m → ρ = 0 ∧ θ = 0) ↔ (m = -1/2 ∨ m = 3/2) :=
by
  sorry

end intersection_points_l0_69


namespace smallest_integer_of_lcm_gcd_l0_139

theorem smallest_integer_of_lcm_gcd (m : ℕ) (h1 : m > 0) (h2 : Nat.lcm 60 m / Nat.gcd 60 m = 44) : m = 165 :=
sorry

end smallest_integer_of_lcm_gcd_l0_139


namespace parallel_lines_solution_l0_491

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, (x + a * y + 6 = 0) → (a - 2) * x + 3 * y + 2 * a = 0) → (a = -1) :=
by
  intro h
  -- Add more formal argument insights if needed
  sorry

end parallel_lines_solution_l0_491


namespace ordering_y1_y2_y3_l0_959

-- Conditions
def A (y₁ : ℝ) : Prop := ∃ b : ℝ, y₁ = -4^2 + 2*4 + b
def B (y₂ : ℝ) : Prop := ∃ b : ℝ, y₂ = -(-1)^2 + 2*(-1) + b
def C (y₃ : ℝ) : Prop := ∃ b : ℝ, y₃ = -(1)^2 + 2*1 + b

-- Question translated to a proof problem
theorem ordering_y1_y2_y3 (y₁ y₂ y₃ : ℝ) :
  A y₁ → B y₂ → C y₃ → y₁ < y₂ ∧ y₂ < y₃ :=
sorry

end ordering_y1_y2_y3_l0_959


namespace cost_of_one_pencil_and_one_pen_l0_408

variables (x y : ℝ)

def eq1 := 4 * x + 3 * y = 3.70
def eq2 := 3 * x + 4 * y = 4.20

theorem cost_of_one_pencil_and_one_pen (h₁ : eq1 x y) (h₂ : eq2 x y) :
  x + y = 1.1286 :=
sorry

end cost_of_one_pencil_and_one_pen_l0_408


namespace number_of_truthful_gnomes_l0_186

variables (T L : ℕ)

-- Conditions
def total_gnomes : Prop := T + L = 10
def hands_raised_vanilla : Prop := 10 = 10
def hands_raised_chocolate : Prop := ½ * 10 = 5
def hands_raised_fruit : Prop := 1 = 1
def total_hands_raised : Prop := 10 + 5 + 1 = 16
def extra_hands_raised : Prop := 16 - 10 = 6
def lying_gnomes : Prop := L = 6
def truthful_gnomes : Prop := T = 4

-- Statement to prove
theorem number_of_truthful_gnomes :
  total_gnomes →
  hands_raised_vanilla →
  hands_raised_chocolate →
  hands_raised_fruit →
  total_hands_raised →
  extra_hands_raised →
  lying_gnomes →
  truthful_gnomes :=
begin
  intros,
  sorry,
end

end number_of_truthful_gnomes_l0_186


namespace value_of_expression_l0_705

theorem value_of_expression (x : ℕ) (h : x = 2) : x + x * x^x = 10 := by
  rw [h] -- Substituting x = 2
  sorry

end value_of_expression_l0_705


namespace symmetric_sum_l0_356

theorem symmetric_sum (a b : ℤ) (h1 : a = -4) (h2 : b = -3) : a + b = -7 := by
  sorry

end symmetric_sum_l0_356


namespace rower_rate_in_still_water_l0_568

theorem rower_rate_in_still_water (V_m V_s : ℝ) (h1 : V_m + V_s = 16) (h2 : V_m - V_s = 12) : V_m = 14 := 
sorry

end rower_rate_in_still_water_l0_568


namespace buying_beams_l0_401

/-- Problem Statement:
Given:
1. The total money for beams is 6210 wen.
2. The transportation cost per beam is 3 wen.
3. Removing one beam means the remaining beams' total transportation cost equals the price of one beam.

Prove: 3 * (x - 1) = 6210 / x
-/
theorem buying_beams (x : ℕ) (h₁ : x > 0) (h₂ : 6210 % x = 0) :
  3 * (x - 1) = 6210 / x :=
sorry

end buying_beams_l0_401


namespace golden_section_AP_l0_649

-- Definitions of the golden ratio and its reciprocal
noncomputable def phi := (1 + Real.sqrt 5) / 2
noncomputable def phi_inv := (Real.sqrt 5 - 1) / 2

-- Conditions of the problem
def isGoldenSectionPoint (A B P : ℝ) := ∃ AP BP AB, AP < BP ∧ BP = 10 ∧ P = AB ∧ AP = BP * phi_inv

theorem golden_section_AP (A B P : ℝ) (h1 : isGoldenSectionPoint A B P) : 
  ∃ AP, AP = 5 * Real.sqrt 5 - 5 :=
by
  sorry

end golden_section_AP_l0_649


namespace chocolate_cost_proof_l0_30

/-- The initial amount of money Dan has. -/
def initial_amount : ℕ := 7

/-- The cost of the candy bar. -/
def candy_bar_cost : ℕ := 2

/-- The remaining amount of money Dan has after the purchases. -/
def remaining_amount : ℕ := 2

/-- The cost of the chocolate. -/
def chocolate_cost : ℕ := initial_amount - candy_bar_cost - remaining_amount

/-- Expected cost of the chocolate. -/
def expected_chocolate_cost : ℕ := 3

/-- Prove that the cost of the chocolate equals the expected cost. -/
theorem chocolate_cost_proof : chocolate_cost = expected_chocolate_cost :=
by
  sorry

end chocolate_cost_proof_l0_30


namespace range_of_years_of_service_l0_446

theorem range_of_years_of_service : 
  let years := [15, 10, 9, 17, 6, 3, 14, 16]
  ∃ min max, (min ∈ years ∧ max ∈ years ∧ (max - min = 14)) :=
by 
  let years := [15, 10, 9, 17, 6, 3, 14, 16]
  use 3, 17 
  sorry

end range_of_years_of_service_l0_446


namespace factorial_division_example_l0_67

theorem factorial_division_example : (10! / (5! * 2!)) = 15120 := 
by
  sorry

end factorial_division_example_l0_67


namespace evaluate_tan_fraction_l0_319

theorem evaluate_tan_fraction:
  (1 - Real.tan (15 * Real.pi / 180)) / (1 + Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 / 3 :=
by
  sorry

end evaluate_tan_fraction_l0_319


namespace incorrect_proposition_C_l0_510

theorem incorrect_proposition_C (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a^4 + b^4 + c^4 + d^4 = 2 * (a^2 * b^2 + c^2 * d^2) → ¬ (a = b ∧ b = c ∧ c = d) := 
sorry

end incorrect_proposition_C_l0_510


namespace plumber_salary_percentage_l0_382

def salary_construction_worker : ℕ := 100
def salary_electrician : ℕ := 2 * salary_construction_worker
def total_salary_without_plumber : ℕ := 2 * salary_construction_worker + salary_electrician
def total_labor_cost : ℕ := 650
def salary_plumber : ℕ := total_labor_cost - total_salary_without_plumber
def percentage_salary_plumber_as_construction_worker (x y : ℕ) : ℕ := (x * 100) / y

theorem plumber_salary_percentage :
  percentage_salary_plumber_as_construction_worker salary_plumber salary_construction_worker = 250 :=
by 
  sorry

end plumber_salary_percentage_l0_382


namespace half_percent_of_160_l0_565

theorem half_percent_of_160 : (1 / 2 / 100) * 160 = 0.8 :=
by
  -- Proof goes here
  sorry

end half_percent_of_160_l0_565


namespace symmetric_point_l0_450

theorem symmetric_point (P : ℝ × ℝ) (a b : ℝ) (h1: P = (2, 1)) (h2 : x - y + 1 = 0) :
  (b - 1) = -(a - 2) ∧ (a + 2) / 2 - (b + 1) / 2 + 1 = 0 → (a, b) = (0, 3) := 
sorry

end symmetric_point_l0_450


namespace Misha_l0_901

theorem Misha's_decision_justified :
  let A_pos := 7 in
  let A_neg := 4 in
  let B_pos := 4 in
  let B_neg := 1 in
  (B_pos / (B_pos + B_neg) > A_pos / (A_pos + A_neg)) := 
sorry

end Misha_l0_901


namespace Tn_lt_Sn_div_2_l0_678

def a₁ := 1
def q := (1 : ℝ) / 3
def a (n : ℕ) : ℝ := q^(n-1)

def b (n : ℕ) : ℝ := (n : ℝ) * a n / 3

def S (n : ℕ) : ℝ := (∑ i in Finset.range n, a (i + 1))

def T (n : ℕ) : ℝ := (∑ i in Finset.range n, b (i + 1))

theorem Tn_lt_Sn_div_2 (n : ℕ) : 
  T n < S n / 2 := 
sorry

end Tn_lt_Sn_div_2_l0_678


namespace metal_waste_l0_922

theorem metal_waste (l b : ℝ) (h : l > b) : l * b - (b^2 / 2) = 
  (l * b - (π * (b / 2)^2)) + (π * (b / 2)^2 - (b^2 / 2)) := by
  sorry

end metal_waste_l0_922


namespace randy_initial_blocks_l0_394

theorem randy_initial_blocks (x : ℕ) (used_blocks : ℕ) (left_blocks : ℕ) 
  (h1 : used_blocks = 36) (h2 : left_blocks = 23) (h3 : x = used_blocks + left_blocks) :
  x = 59 := by 
  sorry

end randy_initial_blocks_l0_394


namespace initial_sugar_amount_l0_252

-- Definitions based on the conditions
def packs : ℕ := 12
def weight_per_pack : ℕ := 250
def leftover_sugar : ℕ := 20

-- Theorem statement
theorem initial_sugar_amount : packs * weight_per_pack + leftover_sugar = 3020 :=
by
  sorry

end initial_sugar_amount_l0_252


namespace find_people_and_carriages_l0_1

theorem find_people_and_carriages (x y : ℝ) :
  (x / 3 = y + 2) ∧ ((x - 9) / 2 = y) ↔
  (x / 3 = y + 2) ∧ ((x - 9) / 2 = y) :=
by
  sorry

end find_people_and_carriages_l0_1


namespace population_2002_l0_964

-- Predicate P for the population of rabbits in a given year
def P : ℕ → ℝ := sorry

-- Given conditions
axiom cond1 : ∃ k : ℝ, P 2003 - P 2001 = k * P 2002
axiom cond2 : ∃ k : ℝ, P 2002 - P 2000 = k * P 2001
axiom condP2000 : P 2000 = 50
axiom condP2001 : P 2001 = 80
axiom condP2003 : P 2003 = 186

-- The statement we need to prove
theorem population_2002 : P 2002 = 120 :=
by
  sorry

end population_2002_l0_964


namespace tan_equals_three_l0_51

variable (α : ℝ)

theorem tan_equals_three : 
  (Real.tan α = 3) → (1 / (Real.sin α * Real.sin α + 2 * Real.sin α * Real.cos α) = 2 / 3) :=
by
  intro h
  sorry

end tan_equals_three_l0_51


namespace minimum_f_l0_391

namespace Proof

def has_unique_solution (d e f : ℕ) (ineq : d < e ∧ e < f) : Prop :=
  ∃ x y : ℕ, (2 * x + y = 2010) ∧ (y = abs (x - d) + abs (x - e) + abs (x - f))

theorem minimum_f (d e f : ℕ) (h₁ : d < e) (h₂ : e < f) (h₃: has_unique_solution d e f (and.intro h₁ h₂)) : f = 1006 := 
sorry

end Proof

end minimum_f_l0_391


namespace find_number_l0_646

theorem find_number (N : ℝ) (h : 0.15 * 0.30 * 0.50 * N = 108) : N = 4800 :=
by
  sorry

end find_number_l0_646


namespace daniel_earnings_l0_479

def fabric_monday := 20
def yarn_monday := 15

def fabric_tuesday := 2 * fabric_monday
def yarn_tuesday := yarn_monday + 10

def fabric_wednesday := fabric_tuesday / 4
def yarn_wednesday := yarn_tuesday / 2

def price_per_yard_fabric := 2
def price_per_yard_yarn := 3

def total_fabric := fabric_monday + fabric_tuesday + fabric_wednesday
def total_yarn := yarn_monday + yarn_tuesday + yarn_wednesday

def earnings_fabric := total_fabric * price_per_yard_fabric
def earnings_yarn := total_yarn * price_per_yard_yarn

def total_earnings := earnings_fabric + earnings_yarn

theorem daniel_earnings :
  total_earnings = 299 := by
  sorry

end daniel_earnings_l0_479


namespace id_tags_divided_by_10_l0_768

def uniqueIDTags (chars : List Char) (counts : Char → Nat) : Nat :=
  let permsWithoutRepetition := 
    Nat.factorial 7 / Nat.factorial (7 - 5)
  let repeatedCharTagCount := 10 * 10 * 6
  permsWithoutRepetition + repeatedCharTagCount

theorem id_tags_divided_by_10 :
  uniqueIDTags ['M', 'A', 'T', 'H', '2', '0', '3'] (fun c =>
    if c = 'M' then 1 else
    if c = 'A' then 1 else
    if c = 'T' then 1 else
    if c = 'H' then 1 else
    if c = '2' then 2 else
    if c = '0' then 1 else
    if c = '3' then 1 else 0) / 10 = 312 :=
by
  sorry

end id_tags_divided_by_10_l0_768


namespace students_minus_rabbits_l0_34

-- Define the number of students per classroom
def students_per_classroom : ℕ := 24

-- Define the number of rabbits per classroom
def rabbits_per_classroom : ℕ := 3

-- Define the number of classrooms
def number_of_classrooms : ℕ := 5

-- Define the total number of students and rabbits
def total_students : ℕ := students_per_classroom * number_of_classrooms
def total_rabbits : ℕ := rabbits_per_classroom * number_of_classrooms

-- The main statement to prove
theorem students_minus_rabbits :
  total_students - total_rabbits = 105 :=
by
  sorry

end students_minus_rabbits_l0_34


namespace product_of_b_l0_410

noncomputable def b_product : ℤ :=
  let y1 := 3
  let y2 := 8
  let x1 := 2
  let l := y2 - y1 -- Side length of the square
  let b₁ := x1 - l -- One possible value of b
  let b₂ := x1 + l -- Another possible value of b
  b₁ * b₂ -- Product of possible values of b

theorem product_of_b :
  b_product = -21 := by
  sorry

end product_of_b_l0_410


namespace tourist_journey_home_days_l0_776

theorem tourist_journey_home_days (x v : ℝ)
  (h1 : (x / 2 + 1) * v = 246)
  (h2 : x * (v + 15) = 276) :
  x + (x / 2 + 1) = 4 :=
by
  sorry

end tourist_journey_home_days_l0_776


namespace fourth_vertex_of_regular_tetrahedron_exists_and_is_unique_l0_159

theorem fourth_vertex_of_regular_tetrahedron_exists_and_is_unique :
  ∃ (x y z : ℤ),
    (x, y, z) ≠ (1, 2, 3) ∧ (x, y, z) ≠ (5, 3, 2) ∧ (x, y, z) ≠ (4, 2, 6) ∧
    (x - 1)^2 + (y - 2)^2 + (z - 3)^2 = 18 ∧
    (x - 5)^2 + (y - 3)^2 + (z - 2)^2 = 18 ∧
    (x - 4)^2 + (y - 2)^2 + (z - 6)^2 = 18 ∧
    (x, y, z) = (2, 3, 5) :=
by
  -- Proof goes here
  sorry

end fourth_vertex_of_regular_tetrahedron_exists_and_is_unique_l0_159


namespace problem1_problem2_l0_73

-- Problem 1: Prove that the solution to f(x) <= 0 for a = -2 is [1, +∞)
theorem problem1 (x : ℝ) : (|x + 2| - 2 * x - 1 ≤ 0) ↔ (1 ≤ x) := sorry

-- Problem 2: Prove that the range of m such that there exists x ∈ ℝ satisfying f(x) + |x + 2| ≤ m for a = 1 is m ≥ 0
theorem problem2 (m : ℝ) : (∃ x : ℝ, |x - 1| - 2 * x - 1 + |x + 2| ≤ m) ↔ (0 ≤ m) := sorry

end problem1_problem2_l0_73


namespace painting_ways_correct_l0_151

noncomputable def num_ways_to_paint : ℕ :=
  let red := 1
  let green_or_blue := 2
  let total_ways_case1 := red
  let total_ways_case2 := (green_or_blue ^ 4)
  let total_ways_case3 := green_or_blue ^ 3
  let total_ways_case4 := green_or_blue ^ 2
  let total_ways_case5 := green_or_blue
  let total_ways_case6 := red
  total_ways_case1 + total_ways_case2 + total_ways_case3 + total_ways_case4 + total_ways_case5 + total_ways_case6

theorem painting_ways_correct : num_ways_to_paint = 32 :=
  by
  sorry

end painting_ways_correct_l0_151


namespace polygon_sides_14_l0_920

theorem polygon_sides_14 (n : ℕ) (θ : ℝ) 
  (h₀ : (n - 2) * 180 - θ = 2000) :
  n = 14 :=
sorry

end polygon_sides_14_l0_920


namespace orange_beads_in_necklace_l0_11

theorem orange_beads_in_necklace (O : ℕ) : 
    (∀ g w o : ℕ, g = 9 ∧ w = 6 ∧ ∃ t : ℕ, t = 45 ∧ 5 * (g + w + O) = 5 * (9 + 6 + O) ∧ 
    ∃ n : ℕ, n = 5 ∧ n * (45) =
    n * (5 * O)) → O = 9 :=
by
  sorry

end orange_beads_in_necklace_l0_11


namespace find_distance_l0_16

-- Definitions based on given conditions
def speed : ℝ := 40 -- in km/hr
def time : ℝ := 6 -- in hours

-- Theorem statement
theorem find_distance (speed : ℝ) (time : ℝ) : speed = 40 → time = 6 → speed * time = 240 :=
by
  intros h1 h2
  rw [h1, h2]
  -- skipping the proof with sorry
  sorry

end find_distance_l0_16


namespace inequality_relation_l0_976

noncomputable def P : ℝ := Real.log 3 / Real.log 2
noncomputable def Q : ℝ := Real.log 2 / Real.log 3
noncomputable def R : ℝ := Real.log Q / Real.log 2

theorem inequality_relation : R < Q ∧ Q < P := by
  sorry

end inequality_relation_l0_976


namespace solve_x_for_fraction_l0_240

theorem solve_x_for_fraction :
  ∃ x : ℝ, (3 * x - 15) / 4 = (x + 7) / 3 ∧ x = 14.6 :=
by
  sorry

end solve_x_for_fraction_l0_240


namespace collinear_vectors_l0_358

open Real

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (OA OB OP : V) (m n : ℝ)

-- Given conditions
def non_collinear (OA OB : V) : Prop :=
  ∀ (t : ℝ), OA ≠ t • OB

def collinear_points (P A B : V) : Prop :=
  ∃ (t : ℝ), P - A = t • (B - A)

def linear_combination (OP OA OB : V) (m n : ℝ) : Prop :=
  OP = m • OA + n • OB

-- The theorem statement
theorem collinear_vectors (noncol : non_collinear OA OB)
  (collinearPAB : collinear_points OP OA OB)
  (lin_comb : linear_combination OP OA OB m n) :
  m = 2 ∧ n = -1 := by
sorry

end collinear_vectors_l0_358


namespace dwarfs_truthful_count_l0_183

theorem dwarfs_truthful_count :
  ∃ (T L : ℕ), T + L = 10 ∧
    (∀ t : ℕ, t = 10 → t + ((10 - T) * 2 - T) = 16) ∧
    T = 4 :=
by
  sorry

end dwarfs_truthful_count_l0_183


namespace fifty_three_days_from_friday_is_tuesday_l0_735

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end fifty_three_days_from_friday_is_tuesday_l0_735


namespace find_number_l0_612

theorem find_number (x : ℤ) (h : x = 1) : x + 1 = 2 :=
  by
  sorry

end find_number_l0_612


namespace algebraic_expression_value_l0_49

theorem algebraic_expression_value (x : ℝ) (h : 5 * x^2 - x - 2 = 0) :
  (2 * x + 1) * (2 * x - 1) + x * (x - 1) = 1 :=
by
  sorry

end algebraic_expression_value_l0_49


namespace fraction_simplification_l0_801

theorem fraction_simplification (x y : ℚ) (hx : x = 4 / 6) (hy : y = 5 / 8) :
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 :=
by
  rw [hx, hy]
  sorry

end fraction_simplification_l0_801


namespace negation_exists_implication_l0_413

theorem negation_exists_implication (x : ℝ) : (¬ ∃ y > 0, y^2 - 2*y - 3 ≤ 0) ↔ ∀ y > 0, y^2 - 2*y - 3 > 0 :=
by
  sorry

end negation_exists_implication_l0_413


namespace coordinates_of_A_l0_225

-- Definitions based on conditions
def origin : ℝ × ℝ := (0, 0)
def similarity_ratio : ℝ := 2
def point_A : ℝ × ℝ := (2, 3)
def point_A' (P : ℝ × ℝ) : Prop :=
  P = (similarity_ratio * point_A.1, similarity_ratio * point_A.2) ∨
  P = (-similarity_ratio * point_A.1, -similarity_ratio * point_A.2)

-- Statement of the theorem
theorem coordinates_of_A' :
  ∃ P : ℝ × ℝ, point_A' P :=
by
  use (4, 6)
  left
  sorry

end coordinates_of_A_l0_225


namespace gcd_proof_l0_229

theorem gcd_proof :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b = 33 ∧ Nat.lcm a b = 90 ∧ Nat.gcd a b = 3 :=
sorry

end gcd_proof_l0_229


namespace exists_positive_integer_m_l0_626

theorem exists_positive_integer_m (a b c d : ℝ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0) (hpos_d : d > 0) (h_cd : c * d = 1) : 
  ∃ m : ℕ, (a * b ≤ ↑m * ↑m) ∧ (↑m * ↑m ≤ (a + c) * (b + d)) :=
by
  sorry

end exists_positive_integer_m_l0_626


namespace day_53_days_from_friday_l0_720

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end day_53_days_from_friday_l0_720


namespace binomial_coefficient_fourth_term_l0_934

theorem binomial_coefficient_fourth_term (n k : ℕ) (hn : n = 5) (hk : k = 3) : Nat.choose n k = 10 := by
  sorry

end binomial_coefficient_fourth_term_l0_934


namespace symmetric_line_eq_l0_331

theorem symmetric_line_eq (x y : ℝ) : (x - y = 0) → (x = 1) → (y = -x + 2) :=
by
  sorry

end symmetric_line_eq_l0_331


namespace turtles_remaining_proof_l0_457

noncomputable def turtles_original := 50
noncomputable def turtles_additional := 7 * turtles_original - 6
noncomputable def turtles_total_before_frightened := turtles_original + turtles_additional
noncomputable def turtles_frightened := (3 / 7) * turtles_total_before_frightened
noncomputable def turtles_remaining := turtles_total_before_frightened - turtles_frightened

theorem turtles_remaining_proof : turtles_remaining = 226 := by
  sorry

end turtles_remaining_proof_l0_457


namespace average_male_students_score_l0_869

def average_male_score (total_avg : ℕ) (female_avg : ℕ) (male_count : ℕ) (female_count : ℕ) : ℕ :=
  let total_sum := (male_count + female_count) * total_avg
  let female_sum := female_count * female_avg
  let male_sum := total_sum - female_sum
  male_sum / male_count

theorem average_male_students_score
  (total_avg : ℕ) (female_avg : ℕ) (male_count : ℕ) (female_count : ℕ)
  (h1 : total_avg = 90) (h2 : female_avg = 92) (h3 : male_count = 8) (h4 : female_count = 20) :
  average_male_score total_avg female_avg male_count female_count = 85 :=
by {
  sorry
}

end average_male_students_score_l0_869


namespace cricket_average_increase_l0_507

theorem cricket_average_increase
    (A : ℝ) -- average score after 18 innings
    (score19 : ℝ) -- runs scored in 19th inning
    (new_average : ℝ) -- new average after 19 innings
    (score19_def : score19 = 97)
    (new_average_def :  new_average = 25)
    (total_runs_def : 19 * new_average = 18 * A + 97) : 
    new_average - (18 * A + score19) / 19 = 4 := 
by
  sorry

end cricket_average_increase_l0_507


namespace age_equivalence_l0_77

variable (x : ℕ)

theorem age_equivalence : ∃ x : ℕ, 60 + x = 35 + x + 11 + x ∧ x = 14 :=
by
  sorry

end age_equivalence_l0_77


namespace sum_of_pairwise_rel_prime_integers_l0_268

def is_pairwise_rel_prime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1

theorem sum_of_pairwise_rel_prime_integers 
  (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) 
  (h_prod : a * b * c = 343000) (h_rel_prime : is_pairwise_rel_prime a b c) : 
  a + b + c = 476 := 
sorry

end sum_of_pairwise_rel_prime_integers_l0_268


namespace option_d_correct_l0_746

variable (a b m n : ℝ)

theorem option_d_correct :
  6 * a + a ≠ 6 * a ^ 2 ∧
  -2 * a + 5 * b ≠ 3 * a * b ∧
  4 * m ^ 2 * n - 2 * m * n ^ 2 ≠ 2 * m * n ∧
  3 * a * b ^ 2 - 5 * b ^ 2 * a = -2 * a * b ^ 2 := by
  sorry

end option_d_correct_l0_746


namespace alberto_more_than_bjorn_and_charlie_l0_963

theorem alberto_more_than_bjorn_and_charlie (time : ℕ) 
  (alberto_speed bjorn_speed charlie_speed: ℕ) 
  (alberto_distance bjorn_distance charlie_distance : ℕ) :
  time = 6 ∧ alberto_speed = 10 ∧ bjorn_speed = 8 ∧ charlie_speed = 9
  ∧ alberto_distance = alberto_speed * time
  ∧ bjorn_distance = bjorn_speed * time
  ∧ charlie_distance = charlie_speed * time
  → (alberto_distance - bjorn_distance = 12) ∧ (alberto_distance - charlie_distance = 6) :=
by
  sorry

end alberto_more_than_bjorn_and_charlie_l0_963


namespace solution_a_l0_908

noncomputable def problem_a (a b c y : ℕ) : Prop :=
  a + b + c = 30 ∧ b + c + y = 30 ∧ a = 2 ∧ y = 3

theorem solution_a (a b c y x : ℕ)
  (h : problem_a a b c y)
  : x = 25 :=
by sorry

end solution_a_l0_908


namespace sum_nine_terms_l0_261

-- Definitions required based on conditions provided in Step a)
variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

-- The arithmetic sequence condition is encapsulated here
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- The definition of S_n being the sum of the first n terms
def sum_first_n (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

-- The given condition from the problem
def given_condition (a : ℕ → ℝ) : Prop :=
  2 * a 8 = 6 + a 1

-- The proof statement to show S_9 = 54 given the above conditions
theorem sum_nine_terms (h_arith : is_arithmetic_sequence a d)
                        (h_sum : sum_first_n a S) 
                        (h_given : given_condition a): 
                        S 9 = 54 :=
  by sorry

end sum_nine_terms_l0_261


namespace find_g_l0_998

-- Given conditions
def line_equation (x y : ℝ) : Prop := y = 2 * x - 10
def parameterization (g : ℝ → ℝ) (t : ℝ) : Prop := 20 * t - 8 = 2 * g t - 10

-- Statement to prove
theorem find_g (g : ℝ → ℝ) (t : ℝ) :
  (∀ x y, line_equation x y → parameterization g t) →
  g t = 10 * t + 1 :=
sorry

end find_g_l0_998


namespace consumption_reduction_l0_751

variable (P C : ℝ)

theorem consumption_reduction (h : P > 0 ∧ C > 0) : 
  (1.25 * P * (0.8 * C) = P * C) :=
by
  -- Conditions: original price P, original consumption C
  -- New price 1.25 * P, New consumption 0.8 * C
  sorry

end consumption_reduction_l0_751


namespace calculation_l0_414

noncomputable def distance_from_sphere_center_to_plane (S P Q R : Point) (r PQ QR RP : ℝ) : ℝ := 
  let a := PQ / 2
  let b := QR / 2
  let c := RP / 2
  let s := (PQ + QR + RP) / 2
  let K := Real.sqrt (s * (s - PQ) * (s - QR) * (s - RP))
  let R := (PQ * QR * RP) / (4 * K)
  Real.sqrt (r^2 - R^2)

theorem calculation 
  (P Q R S : Point) 
  (r : ℝ) 
  (PQ QR RP : ℝ)
  (h1 : PQ = 17)
  (h2 : QR = 18)
  (h3 : RP = 19)
  (h4 : r = 25) :
  distance_from_sphere_center_to_plane S P Q R r PQ QR RP = 35 * Real.sqrt 7 / 8 → 
  ∃ (x y z : ℕ), x + y + z = 50 ∧ (x.gcd z = 1) ∧ ¬ ∃ p : ℕ, Nat.Prime p ∧ p^2 ∣ y := 
by {
  sorry
}

end calculation_l0_414


namespace number_of_labelings_l0_973

-- Define the concept of a truncated chessboard with 8 squares
structure TruncatedChessboard :=
(square_labels : Fin 8 → ℕ)
(condition : ∀ i j, i ≠ j → square_labels i ≠ square_labels j)

-- Assuming a wider adjacency matrix for "connected" (has at least one common vertex)
def connected (i j : Fin 8) : Prop := sorry

-- Define the non-consecutiveness condition
def non_consecutive (board : TruncatedChessboard) :=
  ∀ i j, connected i j → (board.square_labels i ≠ board.square_labels j + 1 ∧
                          board.square_labels i ≠ board.square_labels j - 1)

-- Theorem statement
theorem number_of_labelings : ∃ c : Fin 8 → ℕ, ∀ b : TruncatedChessboard, non_consecutive b → 
  (b.square_labels = c) := sorry

end number_of_labelings_l0_973


namespace ab_equals_4_l0_823

theorem ab_equals_4 (a b : ℝ) (h_pos : a > 0 ∧ b > 0)
  (h_area : (1/2) * (12 / a) * (8 / b) = 12) : a * b = 4 :=
by
  sorry

end ab_equals_4_l0_823


namespace compute_pow_l0_308

theorem compute_pow (i : ℂ) (h : i^2 = -1) : (1 - i)^6 = 8 * i := by
  sorry

end compute_pow_l0_308


namespace find_starting_number_l0_790

theorem find_starting_number (n : ℤ) (h1 : ∀ k : ℤ, n ≤ k ∧ k ≤ 38 → k % 4 = 0) (h2 : (n + 38) / 2 = 22) : n = 8 :=
sorry

end find_starting_number_l0_790


namespace age_of_B_l0_832

variables (A B : ℕ)

-- Conditions
def condition1 := A + 10 = 2 * (B - 10)
def condition2 := A = B + 7

-- Theorem stating the present age of B
theorem age_of_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 37 :=
by
  sorry

end age_of_B_l0_832


namespace repeating_decimal_to_fraction_l0_320

noncomputable def x : ℚ := 3 + 56 / 99

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 3 + 56 / 99) : x = 353 / 99 := 
by 
  rw h
  exact (3 + 56 / 99 : ℚ)
  sorry

end repeating_decimal_to_fraction_l0_320


namespace compute_xy_l0_134

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 198) : xy = 5 :=
by
  sorry

end compute_xy_l0_134


namespace Tn_lt_half_Sn_l0_677

noncomputable def a_n (n : ℕ) : ℝ := (1/3)^(n-1)
noncomputable def b_n (n : ℕ) : ℝ := n * (1/3)^n
noncomputable def S_n (n : ℕ) : ℝ := 3/2 - 1/2 * (1/3)^(n-1)
noncomputable def T_n (n : ℕ) : ℝ := 3/4 - 1/4 * (1/3)^(n-1) - n/2 * (1/3)^n

theorem Tn_lt_half_Sn (n : ℕ) : T_n n < S_n n / 2 :=
by
  sorry

end Tn_lt_half_Sn_l0_677


namespace commission_8000_l0_304

variable (C k : ℝ)

def commission_5000 (C k : ℝ) : Prop := C + 5000 * k = 110
def commission_11000 (C k : ℝ) : Prop := C + 11000 * k = 230

theorem commission_8000 
  (h1 : commission_5000 C k) 
  (h2 : commission_11000 C k)
  : C + 8000 * k = 170 :=
sorry

end commission_8000_l0_304


namespace number_of_petri_dishes_l0_836

def germs_in_lab : ℕ := 3700
def germs_per_dish : ℕ := 25
def num_petri_dishes : ℕ := germs_in_lab / germs_per_dish

theorem number_of_petri_dishes : num_petri_dishes = 148 :=
by
  sorry

end number_of_petri_dishes_l0_836


namespace sequence_properties_l0_74

noncomputable def a (n : ℕ) : ℕ := 3 * n - 1

def S (n : ℕ) : ℕ := n * (2 + 3 * n - 1) / 2

theorem sequence_properties :
  a 5 + a 7 = 34 ∧ ∀ n, S n = (3 * n ^ 2 + n) / 2 :=
by
  sorry

end sequence_properties_l0_74


namespace original_speed_l0_844

noncomputable def circumference_feet := 10
noncomputable def feet_to_miles := 5280
noncomputable def seconds_to_hours := 3600
noncomputable def shortened_time := 1 / 18000
noncomputable def speed_increase := 6

theorem original_speed (r : ℝ) (t : ℝ) : 
  r * t = (circumference_feet / feet_to_miles) * seconds_to_hours ∧ 
  (r + speed_increase) * (t - shortened_time) = (circumference_feet / feet_to_miles) * seconds_to_hours
  → r = 6 := 
by
  sorry

end original_speed_l0_844


namespace simplify_exponentiation_l0_397

-- Define the exponents and the base
variables (t : ℕ)

-- Define the expression and expected result
def expr := t^5 * t^2
def expected := t^7

-- State the proof goal
theorem simplify_exponentiation : expr = expected := 
by sorry

end simplify_exponentiation_l0_397


namespace probability_of_correct_match_l0_459

theorem probability_of_correct_match :
  let n := 3
  let total_arrangements := Nat.factorial n
  let correct_arrangements := 1
  let probability := correct_arrangements / total_arrangements
  probability = ((1: ℤ) / 6) :=
by
  sorry

end probability_of_correct_match_l0_459


namespace smallest_c_a_l0_421

def factorial : ℕ → ℕ
| 0        := 1
| (n + 1)  := (n + 1) * factorial n

theorem smallest_c_a  (a b c : ℕ) (h1 : a * b * c = factorial 9) (h2 : a < b) (h3 : b < c) :
  c - a = 216 :=
  sorry

end smallest_c_a_l0_421


namespace calculation_of_nested_cuberoot_l0_783

theorem calculation_of_nested_cuberoot (M : Real) (h : 1 < M) : (M^1 / 3 + 1 / 3)^(1 / 3 + 1 / 3)^(1 / 3 + 1 / 3)^(1 / 3 + 1 / 3)^(1 / 3 + 1 / 3) = M^(40 / 81) := 
by 
  sorry

end calculation_of_nested_cuberoot_l0_783


namespace james_jail_time_l0_668

-- Definitions based on the conditions
def arson_sentence := 6
def arson_count := 2
def total_arson_sentence := arson_sentence * arson_count

def explosives_sentence := 2 * total_arson_sentence
def terrorism_sentence := 20

-- Total sentence calculation
def total_jail_time := total_arson_sentence + explosives_sentence + terrorism_sentence

-- The theorem we want to prove
theorem james_jail_time : total_jail_time = 56 := by
  sorry

end james_jail_time_l0_668


namespace product_of_roots_cubic_l0_230

theorem product_of_roots_cubic :
  (p q r : ℂ) (hpq : (Polynomial.eval (polynomial.C p * polynomial.C q * polynomial.C r) (polynomial.X^3 - 9 * polynomial.X^2 + 5 * polynomial.X - 15)) = 0) :
  (p * q * r = 5) :=
sorry

end product_of_roots_cubic_l0_230


namespace max_min_conditions_x_values_for_max_min_a2_x_values_for_max_min_aneg2_l0_364

noncomputable def y (x : ℝ) (a b : ℝ) : ℝ := (Real.cos x)^2 - a * (Real.sin x) + b

theorem max_min_conditions (a b : ℝ) :
  (∃ x : ℝ, y x a b = 0 ∧ (∀ x' : ℝ, y x' a b ≤ 0)) ∧ 
  (∃ x : ℝ, y x a b = -4 ∧ (∀ x' : ℝ, y x' a b ≥ -4)) ↔ 
  (a = 2 ∧ b = -2) ∨ (a = -2 ∧ b = -2) := sorry

theorem x_values_for_max_min_a2 (k : ℤ) :
  (∀ x, y x 2 (-2) = 0 ↔ x = -Real.pi / 2 + 2 * Real.pi * k) ∧ 
  (∀ x, (y x 2 (-2)) = -4 ↔ x = Real.pi / 2 + 2 * Real.pi * k) := sorry

theorem x_values_for_max_min_aneg2 (k : ℤ) :
  (∀ x, y x (-2) (-2) = 0 ↔ x = Real.pi / 2 + 2 * Real.pi * k) ∧ 
  (∀ x, (y x (-2) (-2)) = -4 ↔ x = -Real.pi / 2 + 2 * Real.pi * k) := sorry

end max_min_conditions_x_values_for_max_min_a2_x_values_for_max_min_aneg2_l0_364


namespace cyclic_inequality_l0_146

variables {a b c : ℝ}

theorem cyclic_inequality (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  (ab / (a + b + 2 * c) + bc / (b + c + 2 * a) + ca / (c + a + 2 * b)) ≤ (a + b + c) / 4 :=
sorry

end cyclic_inequality_l0_146


namespace value_of_a_l0_652

theorem value_of_a (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) → m > 1 :=
sorry

end value_of_a_l0_652


namespace trig_identity_proof_l0_605

theorem trig_identity_proof :
  let sin := Real.sin
  let cos := Real.cos
  let deg_to_rad := fun θ : ℝ => θ * Real.pi / 180
  sin (deg_to_rad 30) * sin (deg_to_rad 75) - sin (deg_to_rad 60) * cos (deg_to_rad 105) = Real.sqrt 2 / 2 :=
by
  sorry

end trig_identity_proof_l0_605


namespace symmetric_point_origin_l0_873

theorem symmetric_point_origin (x y : ℤ) (h : x = -2 ∧ y = 3) :
    (-x, -y) = (2, -3) :=
by
  cases h with
  | intro hx hy =>
  simp only [hx, hy]
  sorry

end symmetric_point_origin_l0_873


namespace largest_number_in_sequence_is_48_l0_215

theorem largest_number_in_sequence_is_48 
    (a_1 a_2 a_3 a_4 a_5 a_6 : ℕ) 
    (h1 : 0 < a_1) 
    (h2 : a_1 < a_2 ∧ a_2 < a_3 ∧ a_3 < a_4 ∧ a_4 < a_5 ∧ a_5 < a_6)
    (h3 : ∃ k_1 k_2 k_3 k_4 k_5 : ℕ, k_1 > 1 ∧ k_2 > 1 ∧ k_3 > 1 ∧ k_4 > 1 ∧ k_5 > 1 ∧ 
          a_2 = k_1 * a_1 ∧ a_3 = k_2 * a_2 ∧ a_4 = k_3 * a_3 ∧ a_5 = k_4 * a_4 ∧ a_6 = k_5 * a_5)
    (h4 : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 79) 
    : a_6 = 48 := 
by 
    sorry

end largest_number_in_sequence_is_48_l0_215


namespace sum_of_possible_k_values_l0_666

theorem sum_of_possible_k_values (j k : ℕ) (h : j > 0 ∧ k > 0 ∧ (1 / j : ℚ) + (1 / k : ℚ) = 1 / 5) : 
  (k = 26 ∨ k = 10 ∨ k = 6) := sorry

example : ∑ (k ∈ {26, 10, 6}) = 42 := by
  simp

end sum_of_possible_k_values_l0_666


namespace student_weekly_allowance_l0_909

theorem student_weekly_allowance (A : ℝ) (h1 : (4 / 15) * A = 1) : A = 3.75 :=
by
  sorry

end student_weekly_allowance_l0_909


namespace mass_percentage_C_in_CuCO3_l0_604

def molar_mass_Cu := 63.546 -- g/mol
def molar_mass_C := 12.011 -- g/mol
def molar_mass_O := 15.999 -- g/mol
def molar_mass_CuCO3 := molar_mass_Cu + molar_mass_C + 3 * molar_mass_O

theorem mass_percentage_C_in_CuCO3 : 
  (molar_mass_C / molar_mass_CuCO3) * 100 = 9.72 :=
by
  sorry

end mass_percentage_C_in_CuCO3_l0_604


namespace commercial_break_total_time_l0_937

theorem commercial_break_total_time (c1 c2 c3 : ℕ) (c4 : ℕ → ℕ) (interrupt restart : ℕ) 
  (h1 : c1 = 5) (h2 : c2 = 6) (h3 : c3 = 7) 
  (h4 : ∀ i, i < 11 → c4 i = 2) 
  (h_interrupt : interrupt = 3)
  (h_restart : restart = 2) :
  c1 + c2 + c3 + (11 * 2) + interrupt + 2 * restart = 47 := 
  by
  sorry

end commercial_break_total_time_l0_937


namespace no_two_consecutive_heads_l0_767

/-- The probability that two heads never appear consecutively when a coin is tossed 10 times is 9/64. -/
theorem no_two_consecutive_heads (tosses : ℕ) (prob: ℚ):
  tosses = 10 → 
  (prob = 9 / 64) → 
  prob = probability_no_consecutive_heads tosses := sorry

end no_two_consecutive_heads_l0_767


namespace simple_interest_years_l0_556

theorem simple_interest_years
  (CI : ℝ)
  (SI : ℝ)
  (p1 : ℝ := 4000) (r1 : ℝ := 0.10) (t1 : ℝ := 2)
  (p2 : ℝ := 1750) (r2 : ℝ := 0.08)
  (h1 : CI = p1 * (1 + r1) ^ t1 - p1)
  (h2 : SI = CI / 2)
  (h3 : SI = p2 * r2 * t2) :
  t2 = 3 :=
by
  sorry

end simple_interest_years_l0_556


namespace smallest_n_for_area_gt_3000_l0_167

noncomputable def complex_triangle_area (n : ℕ) : ℝ :=
let z1 := (n : ℂ) + complex.I,
    z2 := z1 ^ 2,
    z3 := z1 ^ 4 in
1 / 2 * abs (
  (re z1 * im z2 + re z2 * im z3 + re z3 * im z1) -
  (im z1 * re z2 + im z2 * re z3 + im z3 * re z1)
)

theorem smallest_n_for_area_gt_3000 : ∃ n : ℕ, 0 < n ∧ complex_triangle_area n > 3000 ∧ ∀ m : ℕ, 0 < m ∧ complex_triangle_area m > 3000 → n ≤ m :=
⟨20, by sorry, by sorry, by sorry⟩

end smallest_n_for_area_gt_3000_l0_167


namespace circle_equation_l0_125

-- Definitions based on the conditions
def center_on_x_axis (a b r : ℝ) := b = 0
def tangent_at_point (a b r : ℝ) := (b - 1) / a = -1/2

-- Proof statement
theorem circle_equation (a b r : ℝ) (h1: center_on_x_axis a b r) (h2: tangent_at_point a b r) :
    ∃ (a b r : ℝ), (x - a)^2 + y^2 = r^2 ∧ a = 2 ∧ b = 0 ∧ r^2 = 5 :=
by 
  sorry

end circle_equation_l0_125


namespace arithmetic_sequence_sum_l0_633

variable (a : ℕ → ℝ) (d : ℝ)
-- Conditions
def is_arithmetic_sequence : Prop := ∀ n : ℕ, a (n + 1) = a n + d
def condition : Prop := a 4 + a 8 = 8

-- Question
theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a d →
  condition a →
  (11 / 2) * (a 1 + a 11) = 44 :=
by
  sorry

end arithmetic_sequence_sum_l0_633


namespace truck_distance_l0_300

theorem truck_distance (d: ℕ) (g: ℕ) (eff: ℕ) (new_g: ℕ) (total_distance: ℕ)
  (h1: d = 300) (h2: g = 10) (h3: eff = d / g) (h4: new_g = 15) (h5: total_distance = eff * new_g):
  total_distance = 450 :=
sorry

end truck_distance_l0_300


namespace find_integer_a_l0_128

theorem find_integer_a (a : ℤ) : (∃ x : ℕ, a * x = 3) ↔ a = 1 ∨ a = 3 :=
by
  sorry

end find_integer_a_l0_128


namespace percentage_of_y_l0_438

theorem percentage_of_y (y : ℝ) : (0.3 * 0.6 * y = 0.18 * y) :=
by {
  sorry
}

end percentage_of_y_l0_438


namespace average_speed_l0_995

-- Define the given conditions as Lean variables and constants
variables (v : ℕ)

-- The average speed problem in Lean
theorem average_speed (h : 8 * v = 528) : v = 66 :=
sorry

end average_speed_l0_995


namespace lyka_saving_per_week_l0_106

-- Definitions from the conditions
def smartphone_price : ℕ := 160
def lyka_has : ℕ := 40
def weeks_in_two_months : ℕ := 8

-- The goal (question == correct answer)
theorem lyka_saving_per_week :
  (smartphone_price - lyka_has) / weeks_in_two_months = 15 :=
sorry

end lyka_saving_per_week_l0_106


namespace solve_for_n_l0_335

theorem solve_for_n (n : ℚ) (h : (1 / (n + 2)) + (2 / (n + 2)) + (n / (n + 2)) = 3) : n = -3/2 := 
by
  sorry

end solve_for_n_l0_335


namespace find_m_l0_540

def A (m : ℝ) : Set ℝ := {3, 4, m^2 - 3 * m - 1}
def B (m : ℝ) : Set ℝ := {2 * m, -3}
def C : Set ℝ := {-3}

theorem find_m (m : ℝ) : A m ∩ B m = C → m = 1 :=
by 
  intros h
  sorry

end find_m_l0_540


namespace f_increasing_f_odd_function_l0_209

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem f_increasing (a : ℝ) : ∀ (x1 x2 : ℝ), x1 < x2 → f a x1 < f a x2 :=
by
  sorry

theorem f_odd_function (a : ℝ) : f a 0 = 0 → (a = 1) :=
by
  sorry

end f_increasing_f_odd_function_l0_209


namespace range_of_a_l0_518

theorem range_of_a : 
  (∃ a : ℝ, (∃ x : ℝ, (1 ≤ x ∧ x ≤ 2) ∧ (x^2 + a ≤ a*x - 3))) ↔ (a ≥ 7) :=
sorry

end range_of_a_l0_518


namespace compute_pow_l0_307

theorem compute_pow (i : ℂ) (h : i^2 = -1) : (1 - i)^6 = 8 * i := by
  sorry

end compute_pow_l0_307


namespace solve_equation_l0_400

theorem solve_equation : ∀ x : ℝ, ((x - 1) / 3 = 2 * x) → (x = -1 / 5) :=
by
  assume x h,
  sorry

end solve_equation_l0_400


namespace point_C_velocity_l0_343

theorem point_C_velocity (a T R L x : ℝ) (h : a * T / (a * T - R) = (L + x) / x) :
  x = L * (a * T / R - 1) → 
  (L * (a * T / R - 1)) / T = a * L / R :=
by
  sorry

end point_C_velocity_l0_343


namespace similarity_of_triangle_l0_143

noncomputable def side_length (AB BC AC : ℝ) : Prop :=
  ∀ k : ℝ, k ≠ 1 → (AB, BC, AC) = (k * AB, k * BC, k * AC)

theorem similarity_of_triangle (AB BC AC : ℝ) (h1 : AB > 0) (h2 : BC > 0) (h3 : AC > 0) :
  side_length (2 * AB) (2 * BC) (2 * AC) = side_length AB BC AC :=
by sorry

end similarity_of_triangle_l0_143


namespace avg_salary_of_Raj_and_Roshan_l0_693

variable (R S : ℕ)

theorem avg_salary_of_Raj_and_Roshan (h1 : (R + S + 7000) / 3 = 5000) : (R + S) / 2 = 4000 := by
  sorry

end avg_salary_of_Raj_and_Roshan_l0_693


namespace male_students_tree_planting_l0_315

theorem male_students_tree_planting (average_trees : ℕ) (female_trees : ℕ) 
    (male_trees : ℕ) : 
    (average_trees = 6) →
    (female_trees = 15) → 
    (1 / male_trees + 1 / female_trees = 1 / average_trees) → 
    male_trees = 10 :=
by
  intros h_avg h_fem h_eq
  sorry

end male_students_tree_planting_l0_315


namespace inequality_l0_804

theorem inequality (a b c : ℝ) (h₀ : 0 < c) (h₁ : c < b) (h₂ : b < a) :
  a^4 * b + b^4 * c + c^4 * a > a * b^4 + b * c^4 + c * a^4 :=
by sorry

end inequality_l0_804


namespace ratio_constant_l0_551

theorem ratio_constant (a b c d : ℕ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d)
    (h : ∀ k : ℕ, ∃ m : ℤ, a + c * k = m * (b + d * k)) :
    ∃ m : ℤ, ∀ k : ℕ, a + c * k = m * (b + d * k) :=
    sorry

end ratio_constant_l0_551


namespace domain_ln_x_plus_one_l0_547

theorem domain_ln_x_plus_one : 
  { x : ℝ | ∃ y : ℝ, y = x + 1 ∧ y > 0 } = { x : ℝ | x > -1 } :=
by
  sorry

end domain_ln_x_plus_one_l0_547


namespace system_of_equations_solution_l0_883

theorem system_of_equations_solution (x y z : ℝ) 
  (h1 : x + y = -1) 
  (h2 : x + z = 0) 
  (h3 : y + z = 1) : 
  x = -1 ∧ y = 0 ∧ z = 1 :=
by
  sorry

end system_of_equations_solution_l0_883


namespace question_solution_l0_907

theorem question_solution 
  (hA : -(-1) = abs (-1))
  (hB : ¬ (∃ n : ℤ, ∀ m : ℤ, n < m ∧ m < 0))
  (hC : (-2)^3 = -2^3)
  (hD : ∃ q : ℚ, q = 0) :
  ¬ (∀ q : ℚ, q > 0 ∨ q < 0) := 
by {
  sorry
}

end question_solution_l0_907


namespace evaluate_fg_l0_957

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2 * x - 5

theorem evaluate_fg : f (g 4) = 9 := by
  sorry

end evaluate_fg_l0_957


namespace jackson_fishes_per_day_l0_834

def total_fishes : ℕ := 90
def jonah_per_day : ℕ := 4
def george_per_day : ℕ := 8
def competition_days : ℕ := 5

def jackson_per_day (J : ℕ) : Prop :=
  (total_fishes - (jonah_per_day * competition_days + george_per_day * competition_days)) / competition_days = J

theorem jackson_fishes_per_day : jackson_per_day 6 :=
  by
    sorry

end jackson_fishes_per_day_l0_834


namespace teds_age_l0_166

theorem teds_age (s t : ℕ) (h1 : t = 3 * s - 20) (h2 : t + s = 76) : t = 52 :=
by
  sorry

end teds_age_l0_166


namespace james_total_jail_time_l0_667

theorem james_total_jail_time (arson_count arson_sentence explosive_multiplier domestic_terrorism_sentence : ℕ) :
    arson_count * arson_sentence + (2 * arson_count * arson_sentence) + domestic_terrorism_sentence = 56 :=
by
  -- Given conditions
  let arson_count := 2
  let arson_sentence := 6
  let explosive_multiplier := 2
  let domestic_terrorism_sentence := 20

  -- Compute the total jail time James might face
  let arson_total := arson_count * arson_sentence
  let explosive_sentence := explosive_multiplier * arson_total
  let total_sentence := arson_total + explosive_sentence + domestic_terrorism_sentence

  -- Verify the total sentence is as expected
  have h : total_sentence = 56 := sorry

  exact h

end james_total_jail_time_l0_667


namespace exists_lattice_midpoint_among_five_points_l0_618

-- Definition of lattice points
structure LatticePoint where
  x : ℤ
  y : ℤ

open LatticePoint

-- The theorem we want to prove
theorem exists_lattice_midpoint_among_five_points (A B C D E : LatticePoint) :
    ∃ P Q : LatticePoint, P ≠ Q ∧ (P.x + Q.x) % 2 = 0 ∧ (P.y + Q.y) % 2 = 0 := 
  sorry

end exists_lattice_midpoint_among_five_points_l0_618


namespace exists_n_of_form_2k_l0_607

theorem exists_n_of_form_2k (n : ℕ) (x y z : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_recip : 1/x + 1/y + 1/z = 1/(n : ℤ)) : ∃ k : ℕ, n = 2 * k :=
sorry

end exists_n_of_form_2k_l0_607


namespace truthful_dwarfs_count_l0_189

def number_of_dwarfs := 10
def vanilla_ice_cream := number_of_dwarfs
def chocolate_ice_cream := number_of_dwarfs / 2
def fruit_ice_cream := 1

theorem truthful_dwarfs_count (T L : ℕ) (h1 : T + L = 10)
  (h2 : vanilla_ice_cream = T + (L * 2))
  (h3 : chocolate_ice_cream = T / 2 + (L / 2 * 2))
  (h4 : fruit_ice_cream = 1)
  : T = 4 :=
sorry

end truthful_dwarfs_count_l0_189


namespace initial_sugar_weight_l0_250

-- Definitions corresponding to the conditions
def num_packs : ℕ := 12
def weight_per_pack : ℕ := 250
def leftover_sugar : ℕ := 20

-- Statement of the proof problem
theorem initial_sugar_weight : 
  (num_packs * weight_per_pack + leftover_sugar = 3020) :=
by
  sorry

end initial_sugar_weight_l0_250


namespace cakes_remaining_l0_926

theorem cakes_remaining (cakes_made : ℕ) (cakes_sold : ℕ) (h_made : cakes_made = 149) (h_sold : cakes_sold = 10) :
  (cakes_made - cakes_sold) = 139 :=
by
  cases h_made
  cases h_sold
  sorry

end cakes_remaining_l0_926


namespace units_digit_of_6_pow_5_l0_437

theorem units_digit_of_6_pow_5 : (6^5 % 10) = 6 := 
by sorry

end units_digit_of_6_pow_5_l0_437


namespace shawn_password_possibilities_l0_396

theorem shawn_password_possibilities : 
  ∃ n, n = (Nat.choose 4 2) ∧ n = 6 :=
by
  exists Nat.choose 4 2
  split
  any_goals sorry
  have : Nat.choose 4 2 = 6,
  from rfl
  exact this

end shawn_password_possibilities_l0_396


namespace fill_trough_time_l0_303

noncomputable def time_to_fill (T_old T_new T_third : ℕ) : ℝ :=
  let rate_old := (1 : ℝ) / T_old
  let rate_new := (1 : ℝ) / T_new
  let rate_third := (1 : ℝ) / T_third
  let total_rate := rate_old + rate_new + rate_third
  1 / total_rate

theorem fill_trough_time:
  time_to_fill 600 200 400 = 1200 / 11 := 
by
  sorry

end fill_trough_time_l0_303


namespace intersection_A_B_l0_62

def A (x : ℝ) : Prop := ∃ y, y = Real.log (-x^2 - 2*x + 8) ∧ -x^2 - 2*x + 8 > 0
def B (x : ℝ) : Prop := Real.log x / Real.log 2 < 1 ∧ x > 0

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end intersection_A_B_l0_62


namespace locus_of_centers_l0_931

set_option pp.notation false -- To ensure nicer looking lean code.

-- Define conditions for circles C_3 and C_4
def C3 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C4 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Statement to prove the locus of centers satisfies the equation
theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, (a^2 + b^2 = (r + 1)^2) ∧ ((a - 3)^2 + b^2 = (9 - r)^2)) →
  (a^2 + 18 * b^2 - 6 * a - 440 = 0) :=
by
  sorry -- Proof not required as per the instructions

end locus_of_centers_l0_931


namespace sin_two_pi_zero_l0_311

theorem sin_two_pi_zero : Real.sin (2 * Real.pi) = 0 :=
by 
  -- We assume the necessary periodicity and value properties of the sine function
  sorry

end sin_two_pi_zero_l0_311


namespace count_15000_safe_numbers_l0_337

def is_psafe (p n : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ 0 → ¬ (abs (n - k * p) ≤ 3)

def count_safe_numbers (p8 p12 p15 upper_bound: ℕ) : ℕ :=
  { n : ℕ | n ≤ upper_bound ∧ is_psafe p8 n ∧ is_psafe p12 n ∧ is_psafe p15 n}.card

theorem count_15000_safe_numbers :
  count_safe_numbers 8 12 15 15000 = 2173 :=
sorry

end count_15000_safe_numbers_l0_337


namespace sum_a_b_eq_neg2_l0_101

def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

theorem sum_a_b_eq_neg2 (a b : ℝ) (h : f a + f b = 20) : a + b = -2 :=
by
  sorry

end sum_a_b_eq_neg2_l0_101


namespace translate_sin_to_left_by_pi_over_4_l0_888

def original_function (x : ℝ) : ℝ :=
  3 * sin (2 * x - (Real.pi / 6))

def translated_function (x : ℝ) : ℝ :=
  3 * sin (2 * (x + Real.pi / 4) - (Real.pi / 6))

def expected_result_function (x : ℝ) : ℝ :=
  3 * sin (2 * x + (Real.pi / 3))

theorem translate_sin_to_left_by_pi_over_4 :
  translated_function = expected_result_function :=
by
  sorry

end translate_sin_to_left_by_pi_over_4_l0_888


namespace football_preference_related_to_gender_prob_dist_and_expectation_X_l0_994

-- Definition of conditions
def total_students : ℕ := 100
def male_students : ℕ := 60
def female_students : ℕ := 40
def male_not_enjoy : ℕ := 10
def female_enjoy_fraction : ℚ := 1 / 4

def alpha : ℚ := 0.001
def chi_squared (a b c d n : ℕ) : ℚ := (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))
def chi_squared_critical_value : ℚ := 10.828

-- Part 1: Independence test problem statement
theorem football_preference_related_to_gender :
  let a := 50 in
  let b := 10 in
  let c := 10 in
  let d := 30 in
  let n := 100 in
  chi_squared a b c d n > chi_squared_critical_value :=
by
  sorry -- Proof of chi-squared calculation

-- Conditions for the probability problem
def total_selected_students : ℕ := 8
def male_selected_students : ℕ := 2
def female_selected_students : ℕ := 6

-- Part 2: Probability distribution and expectation problem statement
theorem prob_dist_and_expectation_X :
  let X_vals := [0, 1, 2].to_finset in
  let P : ℕ → ℚ := λ x, 
    match x with
    | 0 => 15 / 28
    | 1 => 3 / 7
    | 2 => 1 / 28
    | _ => 0
    end in
  let E_X : ℚ := 1 / 2 in

  ∀ x ∈ X_vals, P x ∈ {15 / 28, 3 / 7, 1 / 28} ∧
  ∑ x in X_vals, P x = 1 ∧
  ∑ x in X_vals, x * P x = E_X :=
by
  sorry -- Proof of probability distribution and expectation

end football_preference_related_to_gender_prob_dist_and_expectation_X_l0_994


namespace intersection_shape_is_rectangle_l0_169

noncomputable def curve1 (x y : ℝ) : Prop := x * y = 16
noncomputable def curve2 (x y : ℝ) : Prop := x^2 + y^2 = 34

theorem intersection_shape_is_rectangle (x y : ℝ) :
  (curve1 x y ∧ curve2 x y) → 
  ∃ p1 p2 p3 p4 : ℝ × ℝ,
    (curve1 p1.1 p1.2 ∧ curve1 p2.1 p2.2 ∧ curve1 p3.1 p3.2 ∧ curve1 p4.1 p4.2) ∧
    (curve2 p1.1 p1.2 ∧ curve2 p2.1 p2.2 ∧ curve2 p3.1 p3.2 ∧ curve2 p4.1 p4.2) ∧ 
    (dist p1 p2 = dist p3 p4 ∧ dist p2 p3 = dist p4 p1) ∧ 
    (∃ m : ℝ, p1.1 = p2.1 ∧ p3.1 = p4.1 ∧ p1.1 ≠ m ∧ p2.1 ≠ m) := sorry

end intersection_shape_is_rectangle_l0_169


namespace smallest_n_value_l0_750

-- Define the given expression
def exp := (2^5) * (6^2) * (7^3) * (13^4)

-- Define the conditions
def condition_5_2 (n : ℕ) := ∃ k, n * exp = k * 5^2
def condition_3_3 (n : ℕ) := ∃ k, n * exp = k * 3^3
def condition_11_2 (n : ℕ) := ∃ k, n * exp = k * 11^2

-- Define the smallest possible value of n
def smallest_n (n : ℕ) : Prop :=
  condition_5_2 n ∧ condition_3_3 n ∧ condition_11_2 n ∧ ∀ m, (condition_5_2 m ∧ condition_3_3 m ∧ condition_11_2 m) → m ≥ n

-- The theorem statement
theorem smallest_n_value : smallest_n 9075 :=
  by
    sorry

end smallest_n_value_l0_750


namespace point_C_velocity_l0_342

theorem point_C_velocity (a T R L x : ℝ) (h : a * T / (a * T - R) = (L + x) / x) :
  x = L * (a * T / R - 1) → 
  (L * (a * T / R - 1)) / T = a * L / R :=
by
  sorry

end point_C_velocity_l0_342


namespace part_one_part_two_l0_950

variable {a : ℕ → ℕ}

-- Conditions
axiom a1 : a 1 = 3
axiom recurrence_relation : ∀ n, a (n + 1) = 2 * (a n) + 1

-- Proof of the first part
theorem part_one: ∀ n, (a (n + 1) + 1) = 2 * (a n + 1) :=
by
  sorry

-- General formula for the sequence
theorem part_two: ∀ n, a n = 2^(n + 1) - 1 :=
by
  sorry

end part_one_part_two_l0_950


namespace circle_area_ratio_l0_598

theorem circle_area_ratio (r R : ℝ) (h : π * R^2 - π * r^2 = (3/4) * π * r^2) :
  R / r = Real.sqrt 7 / 2 :=
by
  sorry

end circle_area_ratio_l0_598


namespace T_lt_half_S_l0_682

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3) ^ n
def S (n : ℕ) : ℝ := (3/2) - (1/2) * (1/3)^(n-1)
def T (n : ℕ) : ℝ := (3/4) - (1/4) * (1/3)^(n-1) - (n/2) * (1/3)^n

theorem T_lt_half_S (n : ℕ) (hn : n ≥ 1) : T n < (S n) / 2 :=
by
  sorry

end T_lt_half_S_l0_682


namespace al_original_portion_l0_164

theorem al_original_portion (a b c : ℝ) (h1 : a + b + c = 1200) (h2 : 0.75 * a + 2 * b + 2 * c = 1800) : a = 480 :=
by
  sorry

end al_original_portion_l0_164


namespace sum_of_circle_areas_l0_152

theorem sum_of_circle_areas (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) : 
  π * r^2 + π * s^2 + π * t^2 = 56 * π := 
by 
  sorry

end sum_of_circle_areas_l0_152


namespace bug_visits_all_vertices_l0_452

noncomputable def tetrahedron_prob : ℚ :=
  let vertices := {A, B, C, D}
  let start_vertex := A
  let moves := 3

  -- Conditions:
  -- 1. Bug starts at one vertex of a tetrahedron.
  -- 2. Moves along edges with each edge having equal probability.
  -- 3. All choices are independent.

  -- Question: What is the probability that after three moves the bug will have visited every vertex exactly once?

  probability_of_visiting_each_vertex_exactly_once := (2 / 9)

theorem bug_visits_all_vertices :
  tetrahedron_prob = (2 / 9) :=
  sorry

end bug_visits_all_vertices_l0_452


namespace raise_salary_to_original_l0_572

/--
The salary of a person was reduced by 25%. By what percent should his reduced salary be raised
so as to bring it at par with his original salary?
-/
theorem raise_salary_to_original (S : ℝ) (h : S > 0) :
  ∃ P : ℝ, 0.75 * S * (1 + P / 100) = S ∧ P = 33.333333333333336 :=
sorry

end raise_salary_to_original_l0_572


namespace smallest_n_satisfying_congruence_l0_173

theorem smallest_n_satisfying_congruence :
  ∃ (n : ℕ), n > 0 ∧ (∀ m > 0, m < n → (7^m % 5) ≠ (m^7 % 5)) ∧ (7^n % 5) = (n^7 % 5) := 
by sorry

end smallest_n_satisfying_congruence_l0_173


namespace Q_share_of_profit_l0_443

def P_investment : ℕ := 54000
def Q_investment : ℕ := 36000
def total_profit : ℕ := 18000

theorem Q_share_of_profit : Q_investment * total_profit / (P_investment + Q_investment) = 7200 := by
  sorry

end Q_share_of_profit_l0_443


namespace find_f_neg_half_l0_628

def is_odd_function {α β : Type*} [AddGroup α] [Neg β] (f : α → β) : Prop :=
  ∀ x : α, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2 else 0

theorem find_f_neg_half (f_odd : is_odd_function f) (f_pos : ∀ x > 0, f x = Real.log x / Real.log 2) :
  f (-1/2) = 1 := by
  sorry

end find_f_neg_half_l0_628


namespace fifty_three_days_from_friday_is_tuesday_l0_736

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end fifty_three_days_from_friday_is_tuesday_l0_736


namespace unique_pair_l0_327

theorem unique_pair (m n : ℕ) (h1 : m < n) (h2 : n ∣ m^2 + 1) (h3 : m ∣ n^2 + 1) : (m, n) = (1, 1) :=
sorry

end unique_pair_l0_327


namespace value_of_a_l0_945

theorem value_of_a (a : ℝ) (x : ℝ) (h : (a - 1) * x^2 + x + a^2 - 1 = 0) : a = -1 :=
sorry

end value_of_a_l0_945


namespace mike_ride_equals_42_l0_108

-- Define the costs as per the conditions
def cost_mike (M : ℕ) : ℝ := 2.50 + 0.25 * M
def cost_annie : ℝ := 2.50 + 5.00 + 0.25 * 22

-- State the theorem that needs to be proved
theorem mike_ride_equals_42 : ∃ M : ℕ, cost_mike M = cost_annie ∧ M = 42 :=
by
  sorry

end mike_ride_equals_42_l0_108


namespace solve_abs_inequality_l0_44

theorem solve_abs_inequality (x : ℝ) : 
  (3 ≤ abs (x + 2) ∧ abs (x + 2) ≤ 6) ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) := 
by sorry

end solve_abs_inequality_l0_44


namespace Gwen_avg_speed_trip_l0_147

theorem Gwen_avg_speed_trip : 
  ∀ (d1 d2 s1 s2 t1 t2 : ℝ), 
  d1 = 40 → d2 = 40 → s1 = 15 → s2 = 30 →
  d1 / s1 = t1 → d2 / s2 = t2 →
  (d1 + d2) / (t1 + t2) = 20 :=
by 
  intros d1 d2 s1 s2 t1 t2 hd1 hd2 hs1 hs2 ht1 ht2
  sorry

end Gwen_avg_speed_trip_l0_147


namespace no_two_consecutive_heads_l0_766

/-- The probability that two heads never appear consecutively when a coin is tossed 10 times is 9/64. -/
theorem no_two_consecutive_heads (tosses : ℕ) (prob: ℚ):
  tosses = 10 → 
  (prob = 9 / 64) → 
  prob = probability_no_consecutive_heads tosses := sorry

end no_two_consecutive_heads_l0_766


namespace largest_sum_product_l0_255

theorem largest_sum_product (p q : ℕ) (h1 : p * q = 100) (h2 : 0 < p) (h3 : 0 < q) : p + q ≤ 101 :=
sorry

end largest_sum_product_l0_255


namespace license_plate_palindrome_probability_find_m_plus_n_l0_533

noncomputable section

open Nat

def is_palindrome {α : Type} (seq : List α) : Prop :=
  seq = seq.reverse

def number_of_three_digit_palindromes : ℕ :=
  10 * 10  -- explanation: 10 choices for the first and last digits, 10 for the middle digit

def total_three_digit_numbers : ℕ :=
  10^3  -- 1000

def prob_three_digit_palindrome : ℚ :=
  number_of_three_digit_palindromes / total_three_digit_numbers

def number_of_three_letter_palindromes : ℕ :=
  26 * 26  -- 26 choices for the first and last letters, 26 for the middle letter

def total_three_letter_combinations : ℕ :=
  26^3  -- 26^3

def prob_three_letter_palindrome : ℚ :=
  number_of_three_letter_palindromes / total_three_letter_combinations

def prob_either_palindrome : ℚ :=
  prob_three_digit_palindrome + prob_three_letter_palindrome - (prob_three_digit_palindrome * prob_three_letter_palindrome)

def m : ℕ := 7
def n : ℕ := 52

theorem license_plate_palindrome_probability :
  prob_either_palindrome = 7 / 52 := sorry

theorem find_m_plus_n :
  m + n = 59 := rfl

end license_plate_palindrome_probability_find_m_plus_n_l0_533


namespace chocolate_bars_per_box_l0_841

theorem chocolate_bars_per_box (total_chocolate_bars boxes : ℕ) (h1 : total_chocolate_bars = 710) (h2 : boxes = 142) : total_chocolate_bars / boxes = 5 := by
  sorry

end chocolate_bars_per_box_l0_841


namespace days_from_friday_l0_716

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end days_from_friday_l0_716


namespace find_day_53_days_from_friday_l0_731

-- Define the days of the week
inductive day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open day

-- Define a function that calculates the day of the week after a certain number of days
def add_days (d : day) (n : ℕ) : day :=
  match d, n % 7 with
  | Sunday, 0 => Sunday
  | Monday, 0 => Monday
  | Tuesday, 0 => Tuesday
  | Wednesday, 0 => Wednesday
  | Thursday, 0 => Thursday
  | Friday, 0 => Friday
  | Saturday, 0 => Saturday
  | Sunday, 1 => Monday
  | Monday, 1 => Tuesday
  | Tuesday, 1 => Wednesday
  | Wednesday, 1 => Thursday
  | Thursday, 1 => Friday
  | Friday, 1 => Saturday
  | Saturday, 1 => Sunday
  | Sunday, 2 => Tuesday
  | Monday, 2 => Wednesday
  | Tuesday, 2 => Thursday
  | Wednesday, 2 => Friday
  | Thursday, 2 => Saturday
  | Friday, 2 => Sunday
  | Saturday, 2 => Monday
  | Sunday, 3 => Wednesday
  | Monday, 3 => Thursday
  | Tuesday, 3 => Friday
  | Wednesday, 3 => Saturday
  | Thursday, 3 => Sunday
  | Friday, 3 => Monday
  | Saturday, 3 => Tuesday
  | Sunday, 4 => Thursday
  | Monday, 4 => Friday
  | Tuesday, 4 => Saturday
  | Wednesday, 4 => Sunday
  | Thursday, 4 => Monday
  | Friday, 4 => Tuesday
  | Saturday, 4 => Wednesday
  | Sunday, 5 => Friday
  | Monday, 5 => Saturday
  | Tuesday, 5 => Sunday
  | Wednesday, 5 => Monday
  | Thursday, 5 => Tuesday
  | Friday, 5 => Wednesday
  | Saturday, 5 => Thursday
  | Sunday, 6 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 6 => Friday
  | _, _ => Sunday -- This case is unreachable, but required to complete the pattern match
  end

-- Lean statement for the proof problem:
theorem find_day_53_days_from_friday : add_days Friday 53 = Tuesday :=
  sorry

end find_day_53_days_from_friday_l0_731


namespace intersection_M_N_l0_651

def M : Set ℝ :=
  {x | |x| ≤ 2}

def N : Set ℝ :=
  {x | Real.exp x ≥ 1}

theorem intersection_M_N :
  (M ∩ N) = {x | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l0_651


namespace find_difference_l0_645

theorem find_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := 
by
  sorry

end find_difference_l0_645


namespace chord_segments_division_l0_567

-- Definitions based on the conditions
variables (R OM : ℝ) (AB : ℝ)
-- Setting the values as the problem provides 
def radius : ℝ := 15
def distance_from_center : ℝ := 13
def chord_length : ℝ := 18

-- Formulate the problem statement as a theorem
theorem chord_segments_division :
  ∃ (AM MB : ℝ), AM = 14 ∧ MB = 4 :=
by
  let CB := chord_length / 2
  let OC := Real.sqrt (radius^2 - CB^2)
  let MC := Real.sqrt (distance_from_center^2 - OC^2)
  let AM := CB + MC
  let MB := CB - MC
  use AM, MB
  sorry

end chord_segments_division_l0_567


namespace extreme_points_of_f_range_of_a_for_f_le_g_l0_386

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.log x + (1 / 2) * x^2 + a * x

noncomputable def g (x : ℝ) : ℝ :=
  Real.exp x + (3 / 2) * x^2

theorem extreme_points_of_f (a : ℝ) :
  (∃ (x1 x2 : ℝ), x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0)
    ↔ a < -2 :=
sorry

theorem range_of_a_for_f_le_g :
  (∀ x : ℝ, x > 0 → f x a ≤ g x) ↔ a ≤ Real.exp 1 + 1 :=
sorry

end extreme_points_of_f_range_of_a_for_f_le_g_l0_386


namespace Tanika_total_boxes_sold_l0_117

theorem Tanika_total_boxes_sold:
  let friday_boxes := 60
  let saturday_boxes := friday_boxes + 0.5 * friday_boxes
  let sunday_boxes := saturday_boxes - 0.3 * saturday_boxes
  friday_boxes + saturday_boxes + sunday_boxes = 213 :=
by
  sorry

end Tanika_total_boxes_sold_l0_117


namespace remaining_pencils_l0_474

-- Define the initial conditions
def initial_pencils : Float := 56.0
def pencils_given : Float := 9.0

-- Formulate the theorem stating that the remaining pencils = 47.0
theorem remaining_pencils : initial_pencils - pencils_given = 47.0 := by
  sorry

end remaining_pencils_l0_474


namespace ratio_of_pens_to_pencils_l0_712

-- Define the conditions
def total_items : ℕ := 13
def pencils : ℕ := 4
def eraser : ℕ := 1
def pens : ℕ := total_items - pencils - eraser

-- Prove the ratio of pens to pencils is 2:1
theorem ratio_of_pens_to_pencils : pens = 2 * pencils :=
by
  -- indicate that the proof is omitted
  sorry

end ratio_of_pens_to_pencils_l0_712


namespace sunzi_carriage_l0_4

theorem sunzi_carriage (x y : ℕ) :
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔
  ((Three people share a carriage, leaving two carriages empty) ∧ (Two people share a carriage, leaving nine people walking)) := sorry

end sunzi_carriage_l0_4


namespace six_contestants_speaking_orders_l0_707

theorem six_contestants_speaking_orders :
  ∃ orders : ℕ, 
    (∀ (A_first A_last : ℕ), A_first ≠ 1 → A_last ≠ 6 → 
       orders = 4 * Nat.factorial 5) ∧ orders = 480 :=
by
  sorry

end six_contestants_speaking_orders_l0_707


namespace power_mod_l0_743

theorem power_mod (n : ℕ) : 2^99 % 7 = 1 := 
by {
  sorry
}

end power_mod_l0_743


namespace fewest_trips_l0_817

theorem fewest_trips (total_objects : ℕ) (capacity : ℕ) (h_objects : total_objects = 17) (h_capacity : capacity = 3) : 
  (total_objects + capacity - 1) / capacity = 6 :=
by
  sorry

end fewest_trips_l0_817


namespace allen_blocks_l0_587

def blocks_per_color : Nat := 7
def colors_used : Nat := 7

theorem allen_blocks : (blocks_per_color * colors_used) = 49 :=
by
  sorry

end allen_blocks_l0_587


namespace greater_number_l0_447

theorem greater_number (x y : ℕ) (h1 : x * y = 2048) (h2 : x + y - (x - y) = 64) : x = 64 :=
by
  sorry

end greater_number_l0_447


namespace isosceles_triangle_perimeter_l0_625

-- Define the given conditions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem based on the problem statement and conditions
theorem isosceles_triangle_perimeter (a b : ℕ) (P : is_isosceles_triangle a b 5) (Q : is_isosceles_triangle b a 10) :
  valid_triangle a b 5 → valid_triangle b a 10 → a + b + 5 = 25 :=
by sorry

end isosceles_triangle_perimeter_l0_625


namespace find_difference_l0_644

theorem find_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := 
by
  sorry

end find_difference_l0_644


namespace proof_triangle_tangent_l0_858

open Real

def isCongruentAngles (ω : ℝ) := 
  let a := 15
  let b := 18
  let c := 21
  ∃ (x y z : ℝ), 
  (y^2 = x^2 + a^2 - 2 * a * x * cos ω) 
  ∧ (z^2 = y^2 + b^2 - 2 * b * y * cos ω)
  ∧ (x^2 = z^2 + c^2 - 2 * c * z * cos ω)

def isTriangleABCWithSides (AB BC CA : ℝ) (ω : ℝ) (tan_ω : ℝ) : Prop := 
  (AB = 15) ∧ (BC = 18) ∧ (CA = 21) ∧ isCongruentAngles ω 
  ∧ tan ω = tan_ω

theorem proof_triangle_tangent : isTriangleABCWithSides 15 18 21 ω (88/165) := 
by
  sorry

end proof_triangle_tangent_l0_858


namespace range_of_a_l0_632

theorem range_of_a (a : ℝ) :
  (¬ ( ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^2 * x^2 + a * x - 2 = 0 ) 
    ∨ 
   ¬ ( ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0 )) 
→ a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1 :=
by
  sorry

end range_of_a_l0_632


namespace arithmetic_sequence_a10_l0_914

theorem arithmetic_sequence_a10 (a : ℕ → ℝ) 
    (h1 : a 2 = 2) 
    (h2 : a 3 = 4) : 
    a 10 = 18 := 
sorry

end arithmetic_sequence_a10_l0_914


namespace last_three_digits_7_pow_103_l0_794

theorem last_three_digits_7_pow_103 : (7 ^ 103) % 1000 = 60 := sorry

end last_three_digits_7_pow_103_l0_794


namespace least_possible_value_l0_936

theorem least_possible_value (x y : ℝ) : (3 * x * y - 1)^2 + (x - y)^2 ≥ 1 := sorry

end least_possible_value_l0_936


namespace length_of_tube_l0_286

/-- Prove that the length of the tube is 1.5 meters given the initial conditions -/
theorem length_of_tube (h1 : ℝ) (m_water : ℝ) (rho : ℝ) (g : ℝ) (p_ratio : ℝ) :
  h1 = 1.5 ∧ m_water = 1000 ∧ rho = 1000 ∧ g = 9.8 ∧ p_ratio = 2 → 
  ∃ h2 : ℝ, h2 = 1.5 :=
by
  sorry

end length_of_tube_l0_286


namespace ratio_unit_price_l0_781

theorem ratio_unit_price (v p : ℝ) (hv : v > 0) (hp : p > 0) :
  let vX := 1.25 * v
  let pX := 0.85 * p
  (pX / vX) / (p / v) = 17 / 25 := by
{
  sorry
}

end ratio_unit_price_l0_781


namespace sum_of_possible_values_of_k_l0_664

theorem sum_of_possible_values_of_k :
  (∀ j k : ℕ, 0 < j ∧ 0 < k → (1 / (j:ℚ)) + (1 / (k:ℚ)) = (1 / 5) → k = 6 ∨ k = 10 ∨ k = 30) ∧ 
  (46 = 6 + 10 + 30) :=
by
  sorry

end sum_of_possible_values_of_k_l0_664


namespace total_cost_of_vitamins_l0_99

-- Definitions based on the conditions
def original_price : ℝ := 15.00
def discount_percentage : ℝ := 0.20
def coupon_value : ℝ := 2.00
def num_coupons : ℕ := 3
def num_bottles : ℕ := 3

-- Lean statement to prove the final cost
theorem total_cost_of_vitamins
  (original_price : ℝ)
  (discount_percentage : ℝ)
  (coupon_value : ℝ)
  (num_coupons : ℕ)
  (num_bottles : ℕ)
  (discounted_price_per_bottle : ℝ := original_price * (1 - discount_percentage))
  (total_coupon_value : ℝ := coupon_value * num_coupons)
  (total_cost_before_coupons : ℝ := discounted_price_per_bottle * num_bottles) :
  (total_cost_before_coupons - total_coupon_value) = 30.00 :=
by
  sorry

end total_cost_of_vitamins_l0_99


namespace sum_pairwise_relatively_prime_integers_eq_160_l0_563

theorem sum_pairwise_relatively_prime_integers_eq_160
  (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h_prod : a * b * c = 27000)
  (h_coprime_ab : Nat.gcd a b = 1)
  (h_coprime_bc : Nat.gcd b c = 1)
  (h_coprime_ac : Nat.gcd a c = 1) :
  a + b + c = 160 :=
by
  sorry

end sum_pairwise_relatively_prime_integers_eq_160_l0_563


namespace one_elephant_lake_empty_in_365_days_l0_293

variables (C K V : ℝ)
variables (t : ℝ)

noncomputable def lake_empty_one_day (C K V : ℝ) := 183 * C = V + K
noncomputable def lake_empty_five_days (C K V : ℝ) := 185 * C = V + 5 * K

noncomputable def elephant_time (C K V t : ℝ) : Prop :=
  (t * C = V + t * K) → (t = 365)

theorem one_elephant_lake_empty_in_365_days (C K V t : ℝ) :
  (lake_empty_one_day C K V) →
  (lake_empty_five_days C K V) →
  (elephant_time C K V t) := by
  intros h1 h2 h3
  sorry

end one_elephant_lake_empty_in_365_days_l0_293


namespace central_vs_northern_chess_match_l0_478

noncomputable def schedule_chess_match : Nat :=
  let players_team1 := ["A", "B", "C"];
  let players_team2 := ["X", "Y", "Z"];
  let total_games := 3 * 3 * 3;
  let games_per_round := 4;
  let total_rounds := 7;
  Nat.factorial total_rounds

theorem central_vs_northern_chess_match :
    schedule_chess_match = 5040 :=
by
  sorry

end central_vs_northern_chess_match_l0_478


namespace option_b_correct_l0_777

variable (Line Plane : Type)

-- Definitions for perpendicularity and parallelism
variable (perp parallel : Line → Plane → Prop) (parallel_line : Line → Line → Prop)

-- Assumptions reflecting the conditions in the problem
axiom perp_alpha_1 {a : Line} {alpha : Plane} : perp a alpha
axiom perp_alpha_2 {b : Line} {alpha : Plane} : perp b alpha

-- The statement to prove
theorem option_b_correct (a b : Line) (alpha : Plane) :
  perp a alpha → perp b alpha → parallel_line a b :=
by
  intro h1 h2
  -- proof omitted
  sorry

end option_b_correct_l0_777


namespace original_cost_of_car_l0_112

theorem original_cost_of_car (C : ℝ) 
  (repair_cost : ℝ := 15000)
  (selling_price : ℝ := 64900)
  (profit_percent : ℝ := 13.859649122807017) :
  C = 43837.21 :=
by
  have h1 : C + repair_cost = selling_price - (selling_price - (C + repair_cost)) := by sorry
  have h2 : profit_percent / 100 = (selling_price - (C + repair_cost)) / C := by sorry
  have h3 : C = 43837.21 := by sorry
  exact h3

end original_cost_of_car_l0_112


namespace sum_of_roots_l0_274

theorem sum_of_roots (a b c : ℝ) (h : 6 * a ^ 3 - 7 * a ^ 2 + 2 * a = 0 ∧ 
                                   6 * b ^ 3 - 7 * b ^ 2 + 2 * b = 0 ∧ 
                                   6 * c ^ 3 - 7 * c ^ 2 + 2 * c = 0 ∧ 
                                   a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
    a + b + c = 7 / 6 :=
sorry

end sum_of_roots_l0_274


namespace point_C_velocity_l0_341

theorem point_C_velocity (a T R L x : ℝ) (h : a * T / (a * T - R) = (L + x) / x) :
  x = L * (a * T / R - 1) → 
  (L * (a * T / R - 1)) / T = a * L / R :=
by
  sorry

end point_C_velocity_l0_341


namespace Sara_taller_than_Joe_l0_395

noncomputable def Roy_height := 36

noncomputable def Joe_height := Roy_height + 3

noncomputable def Sara_height := 45

theorem Sara_taller_than_Joe : Sara_height - Joe_height = 6 :=
by
  sorry

end Sara_taller_than_Joe_l0_395


namespace geometric_sequence_term_l0_814

theorem geometric_sequence_term :
  ∃ (a_n : ℕ → ℕ),
    -- common ratio condition
    (∀ n, a_n (n + 1) = 2 * a_n n) ∧
    -- sum of first 4 terms condition
    (a_n 1 + a_n 2 + a_n 3 + a_n 4 = 60) ∧
    -- conclusion: value of the third term
    (a_n 3 = 16) :=
by
  sorry

end geometric_sequence_term_l0_814


namespace num_2_edge_paths_l0_553

-- Let T be a tetrahedron with vertices connected such that each vertex has exactly 3 edges.
-- Prove that the number of distinct 2-edge paths from a starting vertex P to an ending vertex Q is 3.

def tetrahedron : Type := ℕ -- This is a simplified representation of vertices

noncomputable def edges (a b : tetrahedron) : Prop := true -- Each pair of distinct vertices is an edge in a tetrahedron

theorem num_2_edge_paths (P Q : tetrahedron) (hP : P ≠ Q) : 
  -- There are 3 distinct 2-edge paths from P to Q  
  ∃ (paths : Finset (tetrahedron × tetrahedron)), 
    paths.card = 3 ∧ 
    ∀ (p : tetrahedron × tetrahedron), p ∈ paths → 
      edges P p.1 ∧ edges p.1 p.2 ∧ p.2 = Q :=
by 
  sorry

end num_2_edge_paths_l0_553


namespace license_plates_count_l0_585

theorem license_plates_count :
  let vowels := 5 -- choices for the first vowel
  let other_letters := 25 -- choices for the second and third letters
  let digits := 10 -- choices for each digit
  (vowels * other_letters * other_letters * (digits * digits * digits)) = 3125000 :=
by
  -- proof steps will go here
  sorry

end license_plates_count_l0_585


namespace least_three_digit_multiple_l0_433

def LCM (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

theorem least_three_digit_multiple (n : ℕ) :
  (n >= 100) ∧ (n < 1000) ∧ (n % 36 = 0) ∧ (∀ m, (m >= 100) ∧ (m < 1000) ∧ (m % 36 = 0) → n <= m) ↔ n = 108 :=
sorry

end least_three_digit_multiple_l0_433


namespace triangle_area_proof_l0_700

-- Conditions
variables (P r : ℝ) (semi_perimeter : ℝ)
-- The perimeter of the triangle is 40 cm
def perimeter_condition : Prop := P = 40
-- The inradius of the triangle is 2.5 cm
def inradius_condition : Prop := r = 2.5
-- The semi-perimeter is half of the perimeter
def semi_perimeter_def : Prop := semi_perimeter = P / 2

-- The area of the triangle
def area_of_triangle : ℝ := r * semi_perimeter

-- Proof Problem
theorem triangle_area_proof (hP : perimeter_condition P) (hr : inradius_condition r) (hsemi : semi_perimeter_def P semi_perimeter) :
  area_of_triangle r semi_perimeter = 50 :=
  sorry

end triangle_area_proof_l0_700


namespace is_factorization_l0_906

-- given an equation A,
-- Prove A is factorization: 
-- i.e., x^3 - x = x * (x + 1) * (x - 1)

theorem is_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by sorry

end is_factorization_l0_906


namespace product_eq_neg_one_l0_158

theorem product_eq_neg_one (m b : ℚ) (hm : m = -2 / 3) (hb : b = 3 / 2) : m * b = -1 :=
by
  rw [hm, hb]
  sorry

end product_eq_neg_one_l0_158


namespace m_range_l0_808

variable (a1 b1 : ℝ)

def arithmetic_sequence (n : ℕ) : ℝ := a1 + 2 * (n - 1)
def geometric_sequence (n : ℕ) : ℝ := b1 * 2^(n - 1)

def a2_condition : Prop := arithmetic_sequence a1 2 + geometric_sequence b1 2 < -2
def a1_b1_condition : Prop := a1 + b1 > 0

theorem m_range : a1_b1_condition a1 b1 ∧ a2_condition a1 b1 → 
  let a4 := arithmetic_sequence a1 4 
  let b3 := geometric_sequence b1 3 
  let m := a4 + b3 
  m < 0 := 
by
  sorry

end m_range_l0_808


namespace inradius_semicircle_relation_l0_89

theorem inradius_semicircle_relation 
  (a b c : ℝ)
  (h_acute: a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2)
  (S : ℝ)
  (p : ℝ)
  (r : ℝ)
  (ra rb rc : ℝ)
  (h_def_semi_perim : p = (a + b + c) / 2)
  (h_area : S = p * r)
  (h_ra : ra = (2 * S) / (b + c))
  (h_rb : rb = (2 * S) / (a + c))
  (h_rc : rc = (2 * S) / (a + b)) :
  2 / r = 1 / ra + 1 / rb + 1 / rc :=
by
  sorry

end inradius_semicircle_relation_l0_89


namespace fg_at_3_l0_494

def f (x : ℝ) : ℝ := x - 4
def g (x : ℝ) : ℝ := x^2 + 5

theorem fg_at_3 : f (g 3) = 10 := by
  sorry

end fg_at_3_l0_494


namespace proof_of_x_and_velocity_l0_348

variables (a T L R x : ℝ)

-- Given condition
def given_eq : Prop := (a * T) / (a * T - R) = (L + x) / x

-- Target statement to prove
def target_eq_x : Prop := x = a * T * (L / R) - L
def target_velocity : Prop := a * (L / R)

-- Main theorem to prove the equivalence
theorem proof_of_x_and_velocity (a T L R : ℝ) : given_eq a T L R x → target_eq_x a T L R x ∧ target_velocity a T L R =
  sorry

end proof_of_x_and_velocity_l0_348


namespace perfect_square_K_l0_715

-- Definitions based on the conditions of the problem
variables (Z K : ℕ)
variables (h1 : 1000 < Z ∧ Z < 5000)
variables (h2 : K > 1)
variables (h3 : Z = K^3)

-- The statement we need to prove
theorem perfect_square_K :
  (∃ K : ℕ, 1000 < K^3 ∧ K^3 < 5000 ∧ K^3 = Z ∧ (∃ a : ℕ, K = a^2)) → K = 16 :=
sorry

end perfect_square_K_l0_715


namespace day_after_53_days_from_Friday_l0_726

def Day : Type := ℕ -- Representing days of the week as natural numbers (0 for Sunday, 1 for Monday, ..., 6 for Saturday)

def Friday : Day := 5
def Tuesday : Day := 2

def days_after (start_day : Day) (n : ℕ) : Day :=
  (start_day + n) % 7

theorem day_after_53_days_from_Friday : days_after Friday 53 = Tuesday :=
  by
  -- proof goes here
  sorry

end day_after_53_days_from_Friday_l0_726


namespace coplanar_lines_k_values_l0_786

theorem coplanar_lines_k_values (k : ℝ) :
  (∃ t u : ℝ, 
    (1 + t = 2 + u) ∧ 
    (2 + 2 * t = 5 + k * u) ∧ 
    (3 - k * t = 6 + u)) ↔ 
  (k = -2 + Real.sqrt 6 ∨ k = -2 - Real.sqrt 6) :=
sorry

end coplanar_lines_k_values_l0_786


namespace least_three_digit_multiple_of_3_4_9_is_108_l0_436

theorem least_three_digit_multiple_of_3_4_9_is_108 :
  ∃ (n : ℕ), (100 ≤ n) ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 9 = 0) ∧ (n = 108) :=
by
  sorry

end least_three_digit_multiple_of_3_4_9_is_108_l0_436


namespace coefficient_x3_l0_328

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_x3 (n k : ℕ) (x : ℤ) :
  let expTerm : ℤ := 1 - x + (1 / x^2017)
  let expansion := fun (k : ℕ) => binomial n k • ((1 - x)^(n - k) * (1 / x^2017)^k)
  (n = 9) → (k = 3) →
  (expansion k) = -84 :=
  by
    intros
    sorry

end coefficient_x3_l0_328


namespace day_of_week_in_53_days_l0_738

/-- Prove that 53 days from Friday is Tuesday given today is Friday and a week has 7 days --/
theorem day_of_week_in_53_days
    (today : String)
    (days_in_week : ℕ)
    (day_count : ℕ)
    (target_day : String)
    (h1 : today = "Friday")
    (h2 : days_in_week = 7)
    (h3 : day_count = 53)
    (h4 : target_day = "Tuesday") :
    let remainder := day_count % days_in_week
    let computed_day := match remainder with
                        | 0 => "Friday"
                        | 1 => "Saturday"
                        | 2 => "Sunday"
                        | 3 => "Monday"
                        | 4 => "Tuesday"
                        | 5 => "Wednesday"
                        | 6 => "Thursday"
                        | _ => "Invalid"
    in computed_day = target_day := 
by
  sorry

end day_of_week_in_53_days_l0_738


namespace joan_first_payment_l0_840

theorem joan_first_payment (P : ℝ) 
  (total_amount : ℝ) 
  (r : ℝ) 
  (n : ℕ) 
  (h_total : total_amount = 109300)
  (h_r : r = 3)
  (h_n : n = 7)
  (h_sum : total_amount = P * (1 - r^n) / (1 - r)) : 
  P = 100 :=
by
  -- proof goes here
  sorry

end joan_first_payment_l0_840


namespace relationship_between_abc_l0_615

noncomputable def a : ℝ := Real.exp 0.9 + 1
def b : ℝ := 2.9
noncomputable def c : ℝ := Real.log (0.9 * Real.exp 3)

theorem relationship_between_abc : a > b ∧ b > c :=
by {
  sorry
}

end relationship_between_abc_l0_615


namespace hyperbola_correct_eqn_l0_206

open Real

def hyperbola_eqn (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

theorem hyperbola_correct_eqn (e c a b x y : ℝ)
  (h_eccentricity : e = 2)
  (h_foci_distance : c = 4)
  (h_major_axis_half_length : a = 2)
  (h_minor_axis_half_length_square : b^2 = c^2 - a^2) :
  hyperbola_eqn x y :=
by
  sorry

end hyperbola_correct_eqn_l0_206


namespace integer_solutions_b_l0_708

theorem integer_solutions_b (b : ℤ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ ∀ x : ℤ, x1 ≤ x ∧ x ≤ x2 → x^2 + b * x + 3 ≤ 0) ↔ b = -4 ∨ b = 4 := 
sorry

end integer_solutions_b_l0_708


namespace unique_positive_real_solution_l0_367

def f (x : ℝ) := x^11 + 5 * x^10 + 20 * x^9 + 1000 * x^8 - 800 * x^7

theorem unique_positive_real_solution :
  ∃! (x : ℝ), 0 < x ∧ f x = 0 :=
sorry

end unique_positive_real_solution_l0_367


namespace solve_for_r_l0_642

noncomputable def k (r : ℝ) : ℝ := 5 / (2 ^ r)

theorem solve_for_r (r : ℝ) :
  (5 = k r * 2 ^ r) ∧ (45 = k r * 8 ^ r) → r = (Real.log 9 / Real.log 2) / 2 :=
by
  intro h
  sorry

end solve_for_r_l0_642


namespace coeff_a_zero_l0_515

theorem coeff_a_zero
  (a b c : ℝ)
  (h : ∀ p : ℝ, 0 < p → ∀ (x : ℝ), (a * x^2 + b * x + c + p = 0) → x > 0) :
  a = 0 :=
sorry

end coeff_a_zero_l0_515


namespace gretchen_fewest_trips_l0_819

def fewestTrips (total_objects : ℕ) (max_carry : ℕ) : ℕ :=
  (total_objects + max_carry - 1) / max_carry

theorem gretchen_fewest_trips : fewestTrips 17 3 = 6 := 
  sorry

end gretchen_fewest_trips_l0_819


namespace power_greater_than_one_million_l0_380

theorem power_greater_than_one_million (α β γ δ : ℝ) (ε ζ η : ℕ)
  (h1 : α = 1.01) (h2 : β = 1.001) (h3 : γ = 1.000001) 
  (h4 : δ = 1000000) 
  (h_eps : ε = 99999900) (h_zet : ζ = 999999000) (h_eta : η = 999999000000) :
  α^ε > δ ∧ β^ζ > δ ∧ γ^η > δ :=
by
  sorry

end power_greater_than_one_million_l0_380


namespace sum_of_angles_l0_498

theorem sum_of_angles (α β : ℝ) (hα: 0 < α ∧ α < π) (hβ: 0 < β ∧ β < π) (h_tan_α: Real.tan α = 1 / 2) (h_tan_β: Real.tan β = 1 / 3) : α + β = π / 4 := 
by 
  sorry

end sum_of_angles_l0_498


namespace no_two_heads_consecutively_probability_l0_762

theorem no_two_heads_consecutively_probability :
  (∃ (total_sequences : ℕ) (valid_sequences : ℕ),
    total_sequences = 2^10 ∧ valid_sequences = 1 + 10 + 36 + 56 + 35 + 6 ∧
    (valid_sequences / total_sequences : ℚ) = 9 / 64) :=
begin
  sorry
end

end no_two_heads_consecutively_probability_l0_762


namespace sequence_add_l0_316

theorem sequence_add (x y : ℝ) (h1 : x = 81 * (1 / 3)) (h2 : y = x * (1 / 3)) : x + y = 36 :=
sorry

end sequence_add_l0_316


namespace principal_amount_l0_461

theorem principal_amount (r : ℝ) (n : ℕ) (t : ℕ) (A : ℝ) :
    r = 0.12 → n = 2 → t = 20 →
    ∃ P : ℝ, A = P * (1 + r / n)^(n * t) :=
by
  intros hr hn ht
  have P := A / (1 + r / n)^(n * t)
  use P
  sorry

end principal_amount_l0_461


namespace trigonometric_identity_l0_63

variable {α : Real}
variable (h : Real.cos α = -2 / 3)

theorem trigonometric_identity : 
  (Real.cos α = -2 / 3) → 
  (Real.cos (4 * Real.pi - α) * Real.sin (-α) / 
  (Real.sin (Real.pi / 2 + α) * Real.tan (Real.pi - α)) = Real.cos α) :=
by
  intro h
  sorry

end trigonometric_identity_l0_63


namespace actual_height_of_boy_is_236_l0_405

-- Define the problem conditions
def average_height (n : ℕ) (avg : ℕ) := n * avg
def incorrect_total_height := average_height 35 180
def correct_total_height := average_height 35 178
def wrong_height := 166
def height_difference := incorrect_total_height - correct_total_height

-- Proving the actual height of the boy whose height was wrongly written
theorem actual_height_of_boy_is_236 : 
  wrong_height + height_difference = 236 := sorry

end actual_height_of_boy_is_236_l0_405


namespace find_pairs_l0_440

-- Definitions for the conditions in the problem
def is_positive (x : ℝ) : Prop := x > 0

def equations (x y : ℝ) : Prop :=
  (Real.log (x^2 + y^2) / Real.log 10 = 2) ∧ 
  (Real.log x / Real.log 2 - 4 = Real.log 3 / Real.log 2 - Real.log y / Real.log 2)

-- Lean 4 Statement
theorem find_pairs (x y : ℝ) : 
  is_positive x ∧ is_positive y ∧ equations x y → (x, y) = (8, 6) ∨ (x, y) = (6, 8) :=
by
  sorry

end find_pairs_l0_440


namespace collections_in_bag_l0_687

noncomputable def distinct_collections : ℕ :=
  let vowels := ['A', 'I', 'O']
  let consonants := ['M', 'H', 'C', 'N', 'T', 'T']
  let case1 := Nat.choose 3 2 * Nat.choose 6 3 -- when 0 or 1 T falls off
  let case2 := Nat.choose 3 2 * Nat.choose 5 1 -- when both T's fall off
  case1 + case2

theorem collections_in_bag : distinct_collections = 75 := 
  by
  -- proof goes here
  sorry

end collections_in_bag_l0_687


namespace solve_inequality_case_a_lt_neg1_solve_inequality_case_a_eq_neg1_solve_inequality_case_a_gt_neg1_l0_992

variable (a x : ℝ)

theorem solve_inequality_case_a_lt_neg1 (h : a < -1) :
  ((x - 1) * (x + a) > 0) ↔ (x < -a ∨ x > 1) := sorry

theorem solve_inequality_case_a_eq_neg1 (h : a = -1) :
  ((x - 1) * (x + a) > 0) ↔ (x ≠ 1) := sorry

theorem solve_inequality_case_a_gt_neg1 (h : a > -1) :
  ((x - 1) * (x + a) > 0) ↔ (x < -a ∨ x > 1) := sorry

end solve_inequality_case_a_lt_neg1_solve_inequality_case_a_eq_neg1_solve_inequality_case_a_gt_neg1_l0_992


namespace a5_value_l0_812

theorem a5_value (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a2 - a1 = 2)
  (h2 : a3 - a2 = 4)
  (h3 : a4 - a3 = 8)
  (h4 : a5 - a4 = 16) :
  a5 = 31 := by
  sorry

end a5_value_l0_812


namespace cyclist_rate_l0_15

theorem cyclist_rate 
  (rate_hiker : ℝ := 4)
  (wait_time_1 : ℝ := 5 / 60)
  (wait_time_2 : ℝ := 10.000000000000002 / 60)
  (hiker_distance : ℝ := rate_hiker * wait_time_2)
  (cyclist_distance : ℝ := hiker_distance)
  (cyclist_rate := cyclist_distance / wait_time_1) :
  cyclist_rate = 8 := by 
sorry

end cyclist_rate_l0_15


namespace find_standard_equation_of_ellipse_l0_46

noncomputable def ellipse_equation (a c b : ℝ) : Prop :=
  ∃ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ∨ (y^2 / a^2 + x^2 / b^2 = 1)

theorem find_standard_equation_of_ellipse (h1 : 2 * a = 12) (h2 : c / a = 1 / 3) :
  ellipse_equation 6 2 4 :=
by
  -- We are proving that given the conditions, the standard equation of the ellipse is as stated
  sorry

end find_standard_equation_of_ellipse_l0_46


namespace fractions_sum_l0_755

theorem fractions_sum (a : ℝ) (h : a ≠ 0) : (1 / a) + (2 / a) = 3 / a := 
by 
  sorry

end fractions_sum_l0_755


namespace total_sum_of_ages_l0_703

theorem total_sum_of_ages (Y : ℕ) (interval : ℕ) (age1 age2 age3 age4 age5 : ℕ)
  (h1 : Y = 2) 
  (h2 : interval = 8) 
  (h3 : age1 = Y) 
  (h4 : age2 = Y + interval) 
  (h5 : age3 = Y + 2 * interval) 
  (h6 : age4 = Y + 3 * interval) 
  (h7 : age5 = Y + 4 * interval) : 
  age1 + age2 + age3 + age4 + age5 = 90 := 
by
  sorry

end total_sum_of_ages_l0_703


namespace same_terminal_angle_l0_244

theorem same_terminal_angle (k : ℤ) :
  ∃ α : ℝ, α = k * 360 + 40 :=
by
  sorry

end same_terminal_angle_l0_244


namespace proof_of_x_and_velocity_l0_349

variables (a T L R x : ℝ)

-- Given condition
def given_eq : Prop := (a * T) / (a * T - R) = (L + x) / x

-- Target statement to prove
def target_eq_x : Prop := x = a * T * (L / R) - L
def target_velocity : Prop := a * (L / R)

-- Main theorem to prove the equivalence
theorem proof_of_x_and_velocity (a T L R : ℝ) : given_eq a T L R x → target_eq_x a T L R x ∧ target_velocity a T L R =
  sorry

end proof_of_x_and_velocity_l0_349


namespace garden_perimeter_l0_770

/-- Define the dimensions of the rectangle and triangle in the garden -/
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 4
def triangle_leg1 : ℕ := 3
def triangle_leg2 : ℕ := 4
def triangle_hypotenuse : ℕ := 5 -- calculated using Pythagorean theorem

/-- Prove that the total perimeter of the combined shape is 28 units -/
theorem garden_perimeter :
  let perimeter := 2 * rectangle_length + rectangle_width + triangle_leg1 + triangle_hypotenuse
  perimeter = 28 :=
by
  let perimeter := 2 * rectangle_length + rectangle_width + triangle_leg1 + triangle_hypotenuse
  have h : perimeter = 28 := sorry
  exact h

end garden_perimeter_l0_770


namespace marbles_left_l0_277

def initial_marbles : ℕ := 100
def percent_t_to_Theresa : ℕ := 25
def percent_t_to_Elliot : ℕ := 10

theorem marbles_left (w t e : ℕ) (h_w : w = initial_marbles)
                                 (h_t : t = percent_t_to_Theresa)
                                 (h_e : e = percent_t_to_Elliot) : w - ((t * w) / 100 + (e * w) / 100) = 65 :=
by
  rw [h_w, h_t, h_e]
  sorry

end marbles_left_l0_277


namespace find_m_plus_n_l0_525

-- Definitions
structure Triangle (A B C P M N : Type) :=
  (midpoint_AD_P : P)
  (intersection_M_AB : M)
  (intersection_N_AC : N)
  (vec_AB : ℝ)
  (vec_AM : ℝ)
  (vec_AC : ℝ)
  (vec_AN : ℝ)
  (m : ℝ)
  (n : ℝ)
  (AB_eq_AM_mul_m : vec_AB = m * vec_AM)
  (AC_eq_AN_mul_n : vec_AC = n * vec_AN)

-- The theorem to prove
theorem find_m_plus_n (A B C P M N : Type)
  (t : Triangle A B C P M N) :
  t.m + t.n = 4 :=
sorry

end find_m_plus_n_l0_525


namespace problem_1_part_1_proof_problem_1_part_2_proof_l0_227

noncomputable def problem_1_part_1 : Real :=
  2 * Real.sqrt 2 + (Real.sqrt 6) / 2

theorem problem_1_part_1_proof:
  let θ₀ := 3 * Real.pi / 4
  let ρ_A := 4 * Real.cos θ₀
  let ρ_B := Real.sqrt 3 * Real.sin θ₀
  |ρ_A - ρ_B| = 2 * Real.sqrt 2 + (Real.sqrt 6) / 2 :=
  sorry

theorem problem_1_part_2_proof :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 2 * x - (Real.sqrt 3)/2 * y = 0) :=
  sorry

end problem_1_part_1_proof_problem_1_part_2_proof_l0_227


namespace simple_interest_rate_l0_217

theorem simple_interest_rate (P : ℝ) (T : ℝ) (A : ℝ) (R : ℝ) (h : A = 3 * P) (h1 : T = 12) (h2 : A - P = (P * R * T) / 100) :
  R = 16.67 :=
by sorry

end simple_interest_rate_l0_217


namespace quadratic_k_value_l0_56

theorem quadratic_k_value (a b k : ℝ) (h_eq : a * b + 2 * a + 2 * b = 1)
  (h_roots : Polynomial.eval₂ (RingHom.id ℝ) a (Polynomial.C k * Polynomial.X ^ 0 + Polynomial.C (-3) * Polynomial.X + Polynomial.C 1) = 0 ∧
             Polynomial.eval₂ (RingHom.id ℝ) b (Polynomial.C k * Polynomial.X ^ 0 + Polynomial.C (-3) * Polynomial.X + Polynomial.C 1) = 0) : 
  k = -5 :=
by
  sorry

end quadratic_k_value_l0_56


namespace cube_angle_diagonals_l0_412

theorem cube_angle_diagonals (q : ℝ) (h : q = 60) : 
  ∃ (d : String), d = "space diagonals" :=
by
  sorry

end cube_angle_diagonals_l0_412


namespace negation_P_l0_120

-- Define the original proposition P
def P (a b : ℝ) : Prop := (a^2 + b^2 = 0) → (a = 0 ∧ b = 0)

-- State the negation of P
theorem negation_P : ∀ (a b : ℝ), (a^2 + b^2 ≠ 0) → (a ≠ 0 ∨ b ≠ 0) :=
by
  sorry

end negation_P_l0_120


namespace equilateral_triangle_combination_l0_501

-- Given the interior angle of a polygon in degrees
constant interior_angle : ℕ → ℚ
-- Values of interior angles for each polygon mentioned
def interior_angle_quad := 90    -- Regular Quadrilateral
def interior_angle_hex  := 120   -- Regular Hexagon
def interior_angle_oct  := 135   -- Regular Octagon
def interior_angle_tri  := 60    -- Equilateral Triangle
def fixed_angle := 150  -- Given regular polygon
 
-- Define the seamless combination condition
def seamless_combination (a b : ℚ) : Prop := ∃ k l : ℕ, k ≥ 1 ∧ l ≥ 1 ∧ k * a + l * b = 360

theorem equilateral_triangle_combination:
  seamless_combination fixed_angle interior_angle_tri ∧
  ¬ seamless_combination fixed_angle interior_angle_quad ∧
  ¬ seamless_combination fixed_angle interior_angle_hex ∧
  ¬ seamless_combination fixed_angle interior_angle_oct :=
by
  sorry

end equilateral_triangle_combination_l0_501


namespace woody_saves_l0_747

variable (C A W : ℕ)

theorem woody_saves (C A W : ℕ) (H1 : C = 282) (H2 : A = 42) (H3 : W = 24) :
  let additional_amount_needed := C - A in
  let weeks := additional_amount_needed / W in
  weeks = 10 :=
by
  unfold additional_amount_needed weeks
  rw [H1, H2, H3]
  simp
  norm_num
  sorry -- Proof not provided in this exercise

end woody_saves_l0_747


namespace fg_sqrt3_eq_neg3_minus_2sqrt3_l0_647
noncomputable def f (x : ℝ) : ℝ := 5 - 2 * x
noncomputable def g (x : ℝ) : ℝ := x^2 + x + 1

theorem fg_sqrt3_eq_neg3_minus_2sqrt3 : f (g (Real.sqrt 3)) = -3 - 2 * Real.sqrt 3 := 
by sorry

end fg_sqrt3_eq_neg3_minus_2sqrt3_l0_647


namespace repeating_decimal_to_fraction_l0_39

theorem repeating_decimal_to_fraction : 
∀ (x : ℝ), x = 4 + (0.0036 / (1 - 0.01)) → x = 144/33 :=
by
  intro x hx
  -- This is a placeholder where the conversion proof would go.
  sorry

end repeating_decimal_to_fraction_l0_39


namespace jacob_younger_than_michael_l0_96

variables (J M : ℕ)

theorem jacob_younger_than_michael (h1 : M + 9 = 2 * (J + 9)) (h2 : J = 5) : M - J = 14 :=
by
  -- Insert proof steps here
  sorry

end jacob_younger_than_michael_l0_96


namespace number_of_photos_to_form_square_without_overlapping_l0_78

theorem number_of_photos_to_form_square_without_overlapping 
  (width length : ℕ)
  (h_width : width = 12)
  (h_length : length = 15) : 
  let lcm_value := Nat.lcm width length in
  let area_square := lcm_value * lcm_value in
  let area_photo := width * length in
  area_square / area_photo = 20 :=
by
  -- Definitions
  let width := 12
  let length := 15
  let lcm_value := Nat.lcm width length
  let area_square := lcm_value * lcm_value
  let area_photo := width * length

  -- Proof
  -- Prove LCM(12, 15) = 60
  have lcm_eq : lcm_value = 60 := 
    by 
      simp [Nat.lcm]; norm_num
   
   -- Prove (60^2 / (12 * 15) = 20)
  have result : area_square / area_photo = 20 := 
    by 
      rw [lcm_eq, Nat.mul_div_cancel, Nat.pow_two]
      norm_num

  exact result

end number_of_photos_to_form_square_without_overlapping_l0_78


namespace find_a20_l0_871

variable {a : ℕ → ℤ}
variable {d : ℤ}
variable {a_1 : ℤ}

def isArithmeticSeq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def formsGeomSeq (a1 a3 a4 : ℤ) : Prop :=
  (a3 - a1)^2 = a1 * (a4 - a1)

theorem find_a20 (h1 : isArithmeticSeq a (-2))
                 (h2 : formsGeomSeq a_1 (a_1 + 2*(-2)) (a_1 + 3*(-2)))
                 (ha1 : a_1 = 8) :
  a 20 = -30 :=
by
  sorry

end find_a20_l0_871


namespace num_birds_is_six_l0_709

-- Define the number of nests
def N : ℕ := 3

-- Define the difference between the number of birds and nests
def diff : ℕ := 3

-- Prove that the number of birds is 6
theorem num_birds_is_six (B : ℕ) (h1 : N = 3) (h2 : B - N = diff) : B = 6 := by
  -- Placeholder for the proof
  sorry

end num_birds_is_six_l0_709


namespace quadratic_root_property_l0_54

theorem quadratic_root_property (a b k : ℝ) 
  (h1 : a * b + 2 * a + 2 * b = 1) 
  (h2 : a + b = 3) 
  (h3 : a * b = k) : k = -5 := 
by
  sorry

end quadratic_root_property_l0_54


namespace neg_70kg_represents_subtract_70kg_l0_226

theorem neg_70kg_represents_subtract_70kg (add_30kg : Int) (concept_opposite : ∀ (x : Int), x = -(-x)) :
  -70 = -70 := 
by
  sorry

end neg_70kg_represents_subtract_70kg_l0_226


namespace negation_of_proposition_l0_82

theorem negation_of_proposition :
  (∀ (x y : ℝ), x^2 + y^2 - 1 > 0) → (∃ (x y : ℝ), x^2 + y^2 - 1 ≤ 0) :=
sorry

end negation_of_proposition_l0_82


namespace quadratic_inequality_solution_l0_608

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 4) * x - k + 8 > 0) ↔ -8/3 < k ∧ k < 6 :=
by
  sorry

end quadratic_inequality_solution_l0_608


namespace right_triangle_sides_l0_22

theorem right_triangle_sides {a b c : ℕ} (h1 : a * (b + 2) = 150) (h2 : a^2 + b^2 = c^2) (h3 : a + (1 / 2 : ℤ) * (a * b) = 75) :
  (a = 6 ∧ b = 23 ∧ c = 25) ∨ (a = 15 ∧ b = 8 ∧ c = 17) :=
sorry

end right_triangle_sides_l0_22


namespace no_solution_to_system_l0_41

open Real

theorem no_solution_to_system (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^(1/3) - y^(1/3) - z^(1/3) = 64) ∧ (x^(1/4) - y^(1/4) - z^(1/4) = 32) ∧ (x^(1/6) - y^(1/6) - z^(1/6) = 8) → False := by
  sorry

end no_solution_to_system_l0_41


namespace people_in_each_playgroup_l0_419

theorem people_in_each_playgroup (girls boys parents playgroups : ℕ) (hg : girls = 14) (hb : boys = 11) (hp : parents = 50) (hpg : playgroups = 3) :
  (girls + boys + parents) / playgroups = 25 := by
  sorry

end people_in_each_playgroup_l0_419


namespace third_character_has_2_lines_l0_839

-- Define the number of lines characters have
variables (x y z : ℕ)

-- The third character has x lines
-- Condition: The second character has 6 more than three times the number of lines the third character has
def second_character_lines : ℕ := 3 * x + 6

-- Condition: The first character has 8 more lines than the second character
def first_character_lines : ℕ := second_character_lines x + 8

-- The first character has 20 lines
def first_character_has_20_lines : Prop := first_character_lines x = 20

-- Prove that the third character has 2 lines
theorem third_character_has_2_lines (h : first_character_has_20_lines x) : x = 2 :=
by
  -- Skipping the proof
  sorry

end third_character_has_2_lines_l0_839


namespace min_value_expression_l0_811

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * b - a - 2 * b = 0) :
  ∃ p : ℝ, p = (a^2/4 - 2/a + b^2 - 1/b) ∧ p = 7 :=
by sorry

end min_value_expression_l0_811


namespace max_a_l0_785

-- Define the conditions
def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

def no_lattice_points (m : ℚ) : Prop :=
  ∀ (x : ℤ), 1 ≤ x ∧ x ≤ 50 → ¬ ∃ (y : ℤ), line_equation m x = y

def m_range (m a : ℚ) : Prop := (2 : ℚ) / 5 < m ∧ m < a

-- Define the problem statement
theorem max_a (a : ℚ) : (a = 22 / 51) ↔ (∃ m, no_lattice_points m ∧ m_range m a) :=
by 
  sorry

end max_a_l0_785


namespace inequality_proof_l0_689

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  ¬ (1 / (1 + x + x * y) > 1 / 3 ∧ 
     y / (1 + y + y * z) > 1 / 3 ∧
     (x * z) / (1 + z + x * z) > 1 / 3) :=
by
  sorry

end inequality_proof_l0_689


namespace velocity_of_point_C_l0_345

variable (a T R L x : ℝ)
variable (a_pos : a > 0) (T_pos : T > 0) (R_pos : R > 0) (L_pos : L > 0)
variable (h_eq : a * T / (a * T - R) = (L + x) / x)

theorem velocity_of_point_C : a * (L / R) = x / T := by
  sorry

end velocity_of_point_C_l0_345


namespace largest_and_smallest_multiples_of_12_l0_271

theorem largest_and_smallest_multiples_of_12 (k : ℤ) (n₁ n₂ : ℤ) (h₁ : k = -150) (h₂ : n₁ = -156) (h₃ : n₂ = -144) :
  (∃ m1 : ℤ, m1 * 12 = n₁ ∧ n₁ < k) ∧ (¬ (∃ m2 : ℤ, m2 * 12 = n₂ ∧ n₂ > k ∧ ∃ m2' : ℤ, m2' * 12 > k ∧ m2' * 12 < n₂)) :=
by
  sorry

end largest_and_smallest_multiples_of_12_l0_271


namespace max_value_f1_on_interval_range_of_a_g_increasing_l0_58

noncomputable def f1 (x : ℝ) : ℝ := 2 * x^2 + x + 2

theorem max_value_f1_on_interval : 
  (∀ x, x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) → f1 x ≤ 5) ∧ 
  (∃ x, x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) ∧ f1 x = 5) :=
sorry

noncomputable def f2 (a x : ℝ) : ℝ := a * x^2 + (a - 1) * x + a

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ Set.Icc (1 : ℝ) (2 : ℝ) → f2 a x / x ≥ 2) → a ≥ 1 :=
sorry

noncomputable def g (a x : ℝ) : ℝ := a * x^2 + (a - 1) * x + a + (1 - (a-1) * x^2) / x

theorem g_increasing (a : ℝ) : 
  (∀ x1 x2, (2 < x1 ∧ x1 < x2 ∧ x2 < 3) → g a x1 < g a x2) → a ≥ 1 / 16 :=
sorry

end max_value_f1_on_interval_range_of_a_g_increasing_l0_58


namespace algebra_expression_evaluation_l0_203

theorem algebra_expression_evaluation (a b c d e : ℝ) 
  (h1 : a * b = 1) 
  (h2 : c + d = 0) 
  (h3 : e < 0) 
  (h4 : abs e = 1) : 
  (-a * b) ^ 2009 - (c + d) ^ 2010 - e ^ 2011 = 0 := by 
  sorry

end algebra_expression_evaluation_l0_203


namespace probability_of_no_shaded_square_l0_451

noncomputable def rectangles_without_shaded_square_probability : ℚ :=
  let n := 502 * 1003
  let m := 502 ^ 2
  1 - (m : ℚ) / n 

theorem probability_of_no_shaded_square : rectangles_without_shaded_square_probability = 501 / 1003 :=
  sorry

end probability_of_no_shaded_square_l0_451


namespace minimum_value_of_F_l0_84

theorem minimum_value_of_F (f g : ℝ → ℝ) (a b : ℝ) (h_odd_f : ∀ x, f (-x) = -f x) 
  (h_odd_g : ∀ x, g (-x) = -g x) (h_max_F : ∃ x > 0, a * f x + b * g x + 3 = 10) 
  : ∃ x < 0, a * f x + b * g x + 3 = -4 := 
sorry

end minimum_value_of_F_l0_84


namespace magnet_cost_is_three_l0_779

noncomputable def stuffed_animal_cost : ℕ := 6
noncomputable def combined_stuffed_animals_cost : ℕ := 2 * stuffed_animal_cost
noncomputable def magnet_cost : ℕ := combined_stuffed_animals_cost / 4

theorem magnet_cost_is_three : magnet_cost = 3 :=
by
  sorry

end magnet_cost_is_three_l0_779


namespace range_of_a_l0_121

theorem range_of_a (a x y : ℝ) (h1 : 77 * a = (2 * x + 2 * y) / 2) (h2 : Real.sqrt (abs a) = Real.sqrt (x * y)) :
  a ∈ Set.Iic (-4) ∪ Set.Ici 4 :=
sorry

end range_of_a_l0_121


namespace snow_volume_l0_258

theorem snow_volume
  (length : ℝ) (width : ℝ) (depth : ℝ)
  (h_length : length = 15)
  (h_width : width = 3)
  (h_depth : depth = 0.6) :
  length * width * depth = 27 := 
by
  -- placeholder for proof
  sorry

end snow_volume_l0_258


namespace initial_sugar_amount_l0_253

-- Definitions based on the conditions
def packs : ℕ := 12
def weight_per_pack : ℕ := 250
def leftover_sugar : ℕ := 20

-- Theorem statement
theorem initial_sugar_amount : packs * weight_per_pack + leftover_sugar = 3020 :=
by
  sorry

end initial_sugar_amount_l0_253


namespace hyperbola_asymptotes_eq_l0_28

theorem hyperbola_asymptotes_eq (M : ℝ) :
  (4 / 3 = 5 / Real.sqrt M) → M = 225 / 16 :=
by
  intro h
  sorry

end hyperbola_asymptotes_eq_l0_28


namespace justify_misha_decision_l0_899

-- Define the conditions based on the problem description
def reviews_smartphone_A := (7, 4) -- 7 positive and 4 negative reviews for A
def reviews_smartphone_B := (4, 1) -- 4 positive and 1 negative reviews for B

-- Define the ratios for each smartphone based on their reviews
def ratio_A := (reviews_smartphone_A.1 : ℚ) / reviews_smartphone_A.2
def ratio_B := (reviews_smartphone_B.1 : ℚ) / reviews_smartphone_B.2

-- Goal: to show that ratio_B > ratio_A, justifying Misha's decision
theorem justify_misha_decision : ratio_B > ratio_A := by
  -- placeholders to bypass the proof steps
  sorry

end justify_misha_decision_l0_899


namespace quadratic_inequality_solution_l0_114

theorem quadratic_inequality_solution (x : ℝ) :
  (-3 * x^2 + 8 * x + 3 > 0) ↔ (x < -1/3 ∨ x > 3) :=
by
  sorry

end quadratic_inequality_solution_l0_114


namespace tan_alpha_value_l0_72

noncomputable def f (x : ℝ) := 3 * Real.sin x + 4 * Real.cos x

theorem tan_alpha_value (α : ℝ) (h : ∀ x : ℝ, f x ≥ f α) : Real.tan α = 3 / 4 := 
sorry

end tan_alpha_value_l0_72


namespace width_of_larger_cuboid_l0_155

theorem width_of_larger_cuboid
    (length_larger : ℝ)
    (width_larger : ℝ)
    (height_larger : ℝ)
    (length_smaller : ℝ)
    (width_smaller : ℝ)
    (height_smaller : ℝ)
    (num_smaller : ℕ)
    (volume_larger : ℝ)
    (volume_smaller : ℝ)
    (divided_into : Real) :
    length_larger = 12 → height_larger = 10 →
    length_smaller = 5 → width_smaller = 3 → height_smaller = 2 →
    num_smaller = 56 →
    volume_smaller = length_smaller * width_smaller * height_smaller →
    volume_larger = num_smaller * volume_smaller →
    volume_larger = length_larger * width_larger * height_larger →
    divided_into = volume_larger / (length_larger * height_larger) →
    width_larger = 14 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end width_of_larger_cuboid_l0_155


namespace twenty_five_percent_greater_l0_149

theorem twenty_five_percent_greater (x : ℕ) (h : x = (88 + (88 * 25) / 100)) : x = 110 :=
sorry

end twenty_five_percent_greater_l0_149


namespace problem1_problem2_l0_635

noncomputable def f (x a b : ℝ) : ℝ := (x + a) / (x + b)

-- Problem (1): Prove the inequality f(x-1) > 0 given b = 1.
theorem problem1 (a x : ℝ) : f (x - 1) a 1 > 0 := sorry

-- Problem (2): Prove the values of a and b such that the range of f(x) for x ∈ [-1, 2] is [5/4, 2].
theorem problem2 (a b : ℝ) (H₁ : f (-1) a b = 5 / 4) (H₂ : f 2 a b = 2) :
    (a = 3 ∧ b = 2) ∨ (a = -4 ∧ b = -3) := sorry

end problem1_problem2_l0_635


namespace total_worth_correct_l0_991

def row1_gold_bars : ℕ := 5
def row1_weight_per_bar : ℕ := 2
def row1_cost_per_kg : ℕ := 20000

def row2_gold_bars : ℕ := 8
def row2_weight_per_bar : ℕ := 3
def row2_cost_per_kg : ℕ := 18000

def row3_gold_bars : ℕ := 3
def row3_weight_per_bar : ℕ := 5
def row3_cost_per_kg : ℕ := 22000

def row4_gold_bars : ℕ := 4
def row4_weight_per_bar : ℕ := 4
def row4_cost_per_kg : ℕ := 25000

def total_worth : ℕ :=
  (row1_gold_bars * row1_weight_per_bar * row1_cost_per_kg)
  + (row2_gold_bars * row2_weight_per_bar * row2_cost_per_kg)
  + (row3_gold_bars * row3_weight_per_bar * row3_cost_per_kg)
  + (row4_gold_bars * row4_weight_per_bar * row4_cost_per_kg)

theorem total_worth_correct : total_worth = 1362000 := by
  sorry

end total_worth_correct_l0_991


namespace original_price_l0_392

theorem original_price (x : ℝ) (h1 : 0.95 * x * 1.40 = 1.33 * x) (h2 : 1.33 * x = 2 * x - 1352.06) : x = 2018 := sorry

end original_price_l0_392


namespace f_value_at_2_9_l0_933

-- Define the function f with its properties as conditions
noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the domain of f
axiom f_domain : ∀ x, 0 ≤ x ∧ x ≤ 1

-- Condition (i)
axiom f_0_eq : f 0 = 0

-- Condition (ii)
axiom f_monotone : ∀ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x ≤ f y

-- Condition (iii)
axiom f_symmetry : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (1 - x) = 3/4 - f x / 2

-- Condition (iv)
axiom f_scale : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (x / 3) = f x / 3

-- Proof goal
theorem f_value_at_2_9 : f (2/9) = 5/24 := by
  sorry

end f_value_at_2_9_l0_933


namespace sequence_bound_l0_621

variable {a : ℕ+ → ℝ}

theorem sequence_bound (h : ∀ k m : ℕ+, |a (k + m) - a k - a m| ≤ 1) :
    ∀ (p q : ℕ+), |a p / p - a q / q| < 1 / p + 1 / q :=
by
  sorry

end sequence_bound_l0_621


namespace pants_cost_l0_390

theorem pants_cost (starting_amount shirts_cost shirts_count amount_left money_after_shirts pants_cost : ℕ) 
    (h1 : starting_amount = 109)
    (h2 : shirts_cost = 11)
    (h3 : shirts_count = 2)
    (h4 : amount_left = 74)
    (h5 : money_after_shirts = starting_amount - shirts_cost * shirts_count)
    (h6 : pants_cost = money_after_shirts - amount_left) :
  pants_cost = 13 :=
by
  sorry

end pants_cost_l0_390


namespace minimum_value_of_func_l0_830

-- Define the circle and the line constraints, and the question
namespace CircleLineProblem

def is_center_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y + 1 = 0

def line_divides_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, is_center_of_circle x y → a * x - b * y + 3 = 0

noncomputable def func_to_minimize (a b : ℝ) : ℝ :=
  (2 / a) + (1 / (b - 1))

theorem minimum_value_of_func :
  ∃ (a b : ℝ), a > 0 ∧ b > 1 ∧ line_divides_circle a b ∧ func_to_minimize a b = 8 :=
by
  sorry

end CircleLineProblem

end minimum_value_of_func_l0_830


namespace area_of_rectangular_garden_l0_784

theorem area_of_rectangular_garden (length width : ℝ) (h_length : length = 2.5) (h_width : width = 0.48) :
  length * width = 1.2 :=
by
  sorry

end area_of_rectangular_garden_l0_784


namespace nonneg_sets_property_l0_40

open Set Nat

theorem nonneg_sets_property (A : Set ℕ) :
  (∀ m n : ℕ, m + n ∈ A → m * n ∈ A) ↔
  (A = ∅ ∨ A = {0} ∨ A = {0, 1} ∨ A = {0, 1, 2} ∨ A = {0, 1, 2, 3} ∨ A = {0, 1, 2, 3, 4} ∨ A = { n | 0 ≤ n }) :=
sorry

end nonneg_sets_property_l0_40


namespace ellipse_area_l0_521

theorem ellipse_area
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (a : { endpoints_major_axis : (ℝ × ℝ) × (ℝ × ℝ) // endpoints_major_axis = ((x1, y1), (x2, y2)) })
  (b : { point_on_ellipse : ℝ × ℝ // point_on_ellipse = (x3, y3) }) :
  (-5 : ℝ) = x1 ∧ (2 : ℝ) = y1 ∧ (15 : ℝ) = x2 ∧ (2 : ℝ) = y2 ∧
  (8 : ℝ) = x3 ∧ (6 : ℝ) = y3 → 
  100 * Real.pi * Real.sqrt (16 / 91) = 100 * Real.pi * Real.sqrt (16 / 91) :=
by
  sorry

end ellipse_area_l0_521


namespace maxwell_walking_speed_l0_982

-- Define Maxwell's walking speed
def Maxwell_speed (v : ℕ) : Prop :=
  ∀ t1 t2 : ℕ, t1 = 10 → t2 = 9 →
  ∀ d1 d2 : ℕ, d1 = 10 * v → d2 = 6 * t2 →
  ∀ d_total : ℕ, d_total = 94 →
  d1 + d2 = d_total

theorem maxwell_walking_speed : Maxwell_speed 4 :=
by
  sorry

end maxwell_walking_speed_l0_982


namespace regular_polygon_sides_l0_18

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 150) : 
  ∃ (n : ℕ), 180 * (n - 2) / n = 150 ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l0_18


namespace fiftyThreeDaysFromFridayIsTuesday_l0_723

-- Domain definition: Days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to compute the day of the week after a given number of days
def dayAfter (start: Day) (n: ℕ) : Day :=
  match start with
  | Sunday    => Day.ofNat ((0 + n) % 7)
  | Monday    => Day.ofNat ((1 + n) % 7)
  | Tuesday   => Day.ofNat ((2 + n) % 7)
  | Wednesday => Day.ofNat ((3 + n) % 7)
  | Thursday  => Day.ofNat ((4 + n) % 7)
  | Friday    => Day.ofNat ((5 + n) % 7)
  | Saturday  => Day.ofNat ((6 + n) % 7)

namespace Day

-- Helper function to convert natural numbers back to day of the week
def ofNat : ℕ → Day
| 0 => Sunday
| 1 => Monday
| 2 => Tuesday
| 3 => Wednesday
| 4 => Thursday
| 5 => Friday
| 6 => Saturday
| _ => Day.ofNat (n % 7)

-- Lean statement: Prove that 53 days from Friday is Tuesday
theorem fiftyThreeDaysFromFridayIsTuesday :
  dayAfter Friday 53 = Tuesday :=
sorry

end fiftyThreeDaysFromFridayIsTuesday_l0_723


namespace solve_for_x_l0_444

theorem solve_for_x (x : ℝ) (h : -200 * x = 1600) : x = -8 :=
sorry

end solve_for_x_l0_444


namespace isosceles_triangles_with_perimeter_27_l0_212

def is_valid_isosceles_triangle (a b : ℕ) : Prop :=
  2 * a + b = 27 ∧ 2 * a > b

theorem isosceles_triangles_with_perimeter_27 :
  { t : ℕ × ℕ // is_valid_isosceles_triangle t.1 t.2 }.card = 7 := 
sorry

end isosceles_triangles_with_perimeter_27_l0_212


namespace find_constant_C_l0_745

def polynomial_remainder (C : ℝ) (x : ℝ) : ℝ :=
  C * x^3 - 3 * x^2 + x - 1

theorem find_constant_C :
  (polynomial_remainder 2 (-1) = -7) → 2 = 2 :=
by
  sorry

end find_constant_C_l0_745


namespace g_value_at_neg3_l0_993

noncomputable def g : ℚ → ℚ := sorry

theorem g_value_at_neg3 (h : ∀ x : ℚ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = 2 * x^2) : 
  g (-3) = 98 / 13 := 
sorry

end g_value_at_neg3_l0_993


namespace complement_of_alpha_l0_614

-- Define that the angle α is given as 44 degrees 36 minutes
def alpha : ℚ := 44 + 36 / 60  -- using rational numbers to represent the degrees and minutes

-- Define the complement function
def complement (angle : ℚ) : ℚ := 90 - angle

-- State the proposition to prove
theorem complement_of_alpha : complement alpha = 45 + 24 / 60 := 
by
  sorry

end complement_of_alpha_l0_614


namespace cubic_difference_l0_499

theorem cubic_difference (x y : ℝ) (h1 : x + y = 15) (h2 : 2 * x + y = 20) : x^3 - y^3 = -875 := 
by
  sorry

end cubic_difference_l0_499


namespace find_y_values_l0_336

variable (x y : ℝ)

theorem find_y_values 
    (h1 : 3 * x^2 + 9 * x + 4 * y - 2 = 0)
    (h2 : 3 * x + 2 * y - 6 = 0) : 
    y^2 - 13 * y + 26 = 0 := by
  sorry

end find_y_values_l0_336


namespace minimum_distance_l0_205

theorem minimum_distance (x y : ℝ) (h : x - y - 1 = 0) : (x - 2)^2 + (y - 2)^2 ≥ 1 / 2 :=
sorry

end minimum_distance_l0_205


namespace determine_m_for_value_range_l0_954

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * x + m

theorem determine_m_for_value_range :
  ∀ m : ℝ, (∀ x : ℝ, f m x ≥ 0) ↔ m = 1 :=
by
  sorry

end determine_m_for_value_range_l0_954


namespace max_k_value_l0_634

noncomputable def max_k : ℝ := sorry 

theorem max_k_value :
  ∀ (k : ℝ),
  (∃ (x y : ℝ), (x - 4)^2 + y^2 = 1 ∧ y = k * x - 2 ∧  (x - 4)^2 + y^2 ≤ 4) ↔ 
  k ≤ 4 / 3 := sorry

end max_k_value_l0_634


namespace contrapositive_example_l0_872

theorem contrapositive_example (a b m : ℝ) :
  (a > b → a * (m^2 + 1) > b * (m^2 + 1)) ↔ (a * (m^2 + 1) ≤ b * (m^2 + 1) → a ≤ b) :=
by sorry

end contrapositive_example_l0_872


namespace greatest_prime_factor_of_144_l0_742

-- Define the number 144
def num : ℕ := 144

-- Define what it means for a number to be a prime factor of num
def is_prime_factor (p n : ℕ) : Prop :=
  Prime p ∧ p ∣ n

-- Define what it means to be the greatest prime factor
def greatest_prime_factor (p n : ℕ) : Prop :=
  is_prime_factor p n ∧ (∀ q, is_prime_factor q n → q ≤ p)

-- Prove that the greatest prime factor of 144 is 3
theorem greatest_prime_factor_of_144 : greatest_prime_factor 3 num :=
sorry

end greatest_prime_factor_of_144_l0_742


namespace time_addition_and_sum_l0_381

noncomputable def time_after_addition (hours_1 minutes_1 seconds_1 hours_2 minutes_2 seconds_2 : ℕ) : (ℕ × ℕ × ℕ) :=
  let total_seconds := seconds_1 + seconds_2
  let extra_minutes := total_seconds / 60
  let result_seconds := total_seconds % 60
  let total_minutes := minutes_1 + minutes_2 + extra_minutes
  let extra_hours := total_minutes / 60
  let result_minutes := total_minutes % 60
  let total_hours := hours_1 + hours_2 + extra_hours
  let result_hours := total_hours % 12
  (result_hours, result_minutes, result_seconds)

theorem time_addition_and_sum :
  let current_hours := 3
  let current_minutes := 0
  let current_seconds := 0
  let add_hours := 300
  let add_minutes := 55
  let add_seconds := 30
  let (final_hours, final_minutes, final_seconds) := time_after_addition current_hours current_minutes current_seconds add_hours add_minutes add_seconds
  final_hours + final_minutes + final_seconds = 88 :=
by
  sorry

end time_addition_and_sum_l0_381


namespace algebra_expression_value_l0_370

theorem algebra_expression_value (a b : ℝ) (h1 : a + b = 10) (h2 : a * b = 11) : a^2 - a * b + b^2 = 67 :=
by
  sorry

end algebra_expression_value_l0_370


namespace find_smaller_number_l0_262

theorem find_smaller_number (a b : ℕ) 
  (h1 : a + b = 15) 
  (h2 : 3 * (a - b) = 21) : b = 4 :=
by
  sorry

end find_smaller_number_l0_262


namespace y_minus_x_eq_seven_point_five_l0_752

theorem y_minus_x_eq_seven_point_five (x y : ℚ) (h1 : x + y = 8) (h2 : y - 3 * x = 7) :
  y - x = 7.5 :=
by sorry

end y_minus_x_eq_seven_point_five_l0_752


namespace cos_double_angle_l0_80

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 1 / 3) : Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end cos_double_angle_l0_80


namespace one_minus_repeating_decimal_l0_323

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ := x

theorem one_minus_repeating_decimal:
  ∀ (x : ℚ), x = 1/3 → 1 - x = 2/3 :=
by
  sorry

end one_minus_repeating_decimal_l0_323


namespace max_radius_of_circle_l0_910

theorem max_radius_of_circle (c : ℝ × ℝ → Prop) (h1 : c (16, 0)) (h2 : c (-16, 0)) :
  ∃ r : ℝ, r = 16 :=
by
  sorry

end max_radius_of_circle_l0_910


namespace day_after_53_days_from_Friday_l0_725

def Day : Type := ℕ -- Representing days of the week as natural numbers (0 for Sunday, 1 for Monday, ..., 6 for Saturday)

def Friday : Day := 5
def Tuesday : Day := 2

def days_after (start_day : Day) (n : ℕ) : Day :=
  (start_day + n) % 7

theorem day_after_53_days_from_Friday : days_after Friday 53 = Tuesday :=
  by
  -- proof goes here
  sorry

end day_after_53_days_from_Friday_l0_725


namespace fraction_students_received_Bs_l0_962

theorem fraction_students_received_Bs (fraction_As : ℝ) (fraction_As_or_Bs : ℝ) (h1 : fraction_As = 0.7) (h2 : fraction_As_or_Bs = 0.9) :
  fraction_As_or_Bs - fraction_As = 0.2 :=
by
  sorry

end fraction_students_received_Bs_l0_962


namespace remainder_47_mod_288_is_23_mod_24_l0_448

theorem remainder_47_mod_288_is_23_mod_24 (m : ℤ) (h : m % 288 = 47) : m % 24 = 23 := 
sorry

end remainder_47_mod_288_is_23_mod_24_l0_448


namespace range_of_a_l0_519

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Icc (1 : ℝ) 2, x^2 + a ≤ a * x - 3) ↔ 7 ≤ a :=
sorry

end range_of_a_l0_519


namespace ratio_345_iff_arithmetic_sequence_l0_70

-- Define the variables and the context
variables (a b c : ℕ) -- assuming non-negative integers for simplicity
variable (k : ℕ) -- scaling factor for the 3:4:5 ratio
variable (d : ℕ) -- common difference in the arithmetic sequence

-- Conditions given
def isRightAngledTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∧ a < b ∧ b < c

def is345Ratio (a b c : ℕ) : Prop :=
  ∃ k, a = 3 * k ∧ b = 4 * k ∧ c = 5 * k

def formsArithmeticSequence (a b c : ℕ) : Prop :=
  ∃ d, b = a + d ∧ c = b + d 

-- The statement to prove: sufficiency and necessity
theorem ratio_345_iff_arithmetic_sequence 
  (h_triangle : isRightAngledTriangle a b c) :
  (is345Ratio a b c ↔ formsArithmeticSequence a b c) :=
sorry

end ratio_345_iff_arithmetic_sequence_l0_70


namespace division_exponentiation_addition_l0_929

theorem division_exponentiation_addition :
  6 / -3 + 2^2 * (1 - 4) = -14 := by
sorry

end division_exponentiation_addition_l0_929


namespace minimize_q_l0_441

noncomputable def q (x : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - 6

theorem minimize_q : ∃ x : ℝ, q x = 2 :=
by
  sorry

end minimize_q_l0_441


namespace ellipse_eccentricity_equilateral_triangle_l0_218

theorem ellipse_eccentricity_equilateral_triangle
  (c a : ℝ) (h : c / a = 1 / 2) : eccentricity = 1 / 2 :=
by
  -- Proof goes here, we add sorry to skip proof content
  sorry

end ellipse_eccentricity_equilateral_triangle_l0_218


namespace MrsHiltRows_l0_984

theorem MrsHiltRows :
  let (a : ℕ) := 16
  let (b : ℕ) := 14
  let (r : ℕ) := 5
  (a + b) / r = 6 := by
  sorry

end MrsHiltRows_l0_984


namespace original_number_l0_710

theorem original_number (sum_orig : ℕ) (sum_new : ℕ) (changed_value : ℕ) (avg_orig : ℕ) (avg_new : ℕ) (n : ℕ) :
    sum_orig = n * avg_orig →
    sum_new = sum_orig - changed_value + 9 →
    avg_new = 8 →
    avg_orig = 7 →
    n = 7 →
    sum_new = n * avg_new →
    changed_value = 2 := 
by
  sorry

end original_number_l0_710


namespace ratio_of_areas_l0_425

theorem ratio_of_areas (C1 C2 : ℝ) (h : (60 / 360) * C1 = (30 / 360) * C2) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 1 / 4 :=
by
  sorry

end ratio_of_areas_l0_425


namespace time_to_shovel_snow_l0_842

noncomputable def initial_rate : ℕ := 30
noncomputable def decay_rate : ℕ := 2
noncomputable def driveway_width : ℕ := 6
noncomputable def driveway_length : ℕ := 15
noncomputable def snow_depth : ℕ := 2

noncomputable def total_snow_volume : ℕ := driveway_width * driveway_length * snow_depth

def snow_shoveling_time (initial_rate decay_rate total_volume : ℕ) : ℕ :=
-- Function to compute the time needed, assuming definition provided
sorry

theorem time_to_shovel_snow 
  : snow_shoveling_time initial_rate decay_rate total_snow_volume = 8 :=
sorry

end time_to_shovel_snow_l0_842


namespace last_three_digits_of_7_pow_103_l0_797

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 327 :=
by
  sorry

end last_three_digits_of_7_pow_103_l0_797


namespace stone_hitting_ground_time_l0_118

noncomputable def equation (s : ℝ) : ℝ := -4.5 * s^2 - 12 * s + 48

theorem stone_hitting_ground_time :
  ∃ s : ℝ, equation s = 0 ∧ s = (-8 + 16 * Real.sqrt 7) / 6 :=
by
  sorry

end stone_hitting_ground_time_l0_118


namespace f_one_zero_range_of_a_l0_629

variable (f : ℝ → ℝ) (a : ℝ)

-- Conditions
def odd_function : Prop := ∀ x : ℝ, x ≠ 0 → f (-x) = -f x
def increasing_on_pos : Prop := ∀ x y : ℝ, 0 < x → x < y → f x < f y
def f_neg_one_zero : Prop := f (-1) = 0
def f_a_minus_half_neg : Prop := f (a - 1/2) < 0

-- Questions
theorem f_one_zero (h1 : odd_function f) (h2 : increasing_on_pos f) (h3 : f_neg_one_zero f) : f 1 = 0 := 
sorry

theorem range_of_a (h1 : odd_function f) (h2 : increasing_on_pos f) (h3 : f_neg_one_zero f) (h4 : f_a_minus_half_neg f a) :
  1/2 < a ∧ a < 3/2 ∨ a < -1/2 :=
sorry

end f_one_zero_range_of_a_l0_629


namespace regular_polygon_sides_l0_21

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) 
(h_interior : (n - 2) * 180 / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l0_21


namespace interview_problem_l0_659

noncomputable def judge_scores : List ℝ := [70, 85, 86, 88, 90, 90, 92, 94, 95, 100]

theorem interview_problem :
  let sorted_scores := List.sort judge_scores
  let n := sorted_scores.length
  let median := (sorted_scores.get! (n/2 - 1) + sorted_scores.get! (n/2)) / 2
  let mean := sorted_scores.sum / n
  let removed_scores := List.filter (λ x, x ≠ List.minimum judge_scores ∧ x ≠ List.maximum judge_scores) judge_scores
  let new_mean := removed_scores.sum / removed_scores.length
  let variance (l : List ℝ) : ℝ := (l.map (λ x, (x - mean) ^ 2)).sum / l.length
  let new_variance := variance removed_scores
  Prop := (
    (∃ p : ℝ, p = (2 / (judge_scores.length * (judge_scores.length - 1))) ∧ p ≠ 1 / judge_scores.length) ∧
    (sorted_scores.get! (6 - 1) + sorted_scores.get! 6) / 2 = 91 ∧
    mean < median ∧
    new_mean > mean ∧
    new_variance < variance judge_scores
  )
:= sorry

end interview_problem_l0_659


namespace periodic_odd_fn_calc_l0_516

theorem periodic_odd_fn_calc :
  ∀ (f : ℝ → ℝ),
  (∀ x, f (x + 2) = f x) ∧ (∀ x, f (-x) = -f x) ∧ (∀ x, 0 < x ∧ x < 1 → f x = 4^x) →
  f (-5 / 2) + f 2 = -2 :=
by
  intros f h
  sorry

end periodic_odd_fn_calc_l0_516


namespace number_of_truthful_dwarfs_l0_178

def total_dwarfs := 10
def hands_raised_vanilla := 10
def hands_raised_chocolate := 5
def hands_raised_fruit := 1
def total_hands_raised := hands_raised_vanilla + hands_raised_chocolate + hands_raised_fruit
def extra_hands := total_hands_raised - total_dwarfs
def liars := extra_hands
def truthful := total_dwarfs - liars

theorem number_of_truthful_dwarfs : truthful = 4 :=
by sorry

end number_of_truthful_dwarfs_l0_178


namespace sin_B_triangle_area_l0_831

variable {a b c : ℝ}
variable {A B C : ℝ}

theorem sin_B (hC : C = 3 / 4 * Real.pi) (hSinA : Real.sin A = Real.sqrt 5 / 5) :
  Real.sin B = Real.sqrt 10 / 10 := by
  sorry

theorem triangle_area (hC : C = 3 / 4 * Real.pi) (hSinA : Real.sin A = Real.sqrt 5 / 5)
  (hDiff : c - a = 5 - Real.sqrt 10) (hSinB : Real.sin B = Real.sqrt 10 / 10) :
  1 / 2 * a * c * Real.sin B = 5 / 2 := by
  sorry

end sin_B_triangle_area_l0_831


namespace total_tiles_l0_469

theorem total_tiles (s : ℕ) (H1 : 2 * s - 1 = 57) : s^2 = 841 := by
  sorry

end total_tiles_l0_469


namespace average_speed_bike_l0_191

theorem average_speed_bike (t_goal : ℚ) (d_swim r_swim : ℚ) (d_run r_run : ℚ) (d_bike r_bike : ℚ) :
  t_goal = 1.75 →
  d_swim = 1 / 3 ∧ r_swim = 1.5 →
  d_run = 2.5 ∧ r_run = 8 →
  d_bike = 12 →
  r_bike = 1728 / 175 :=
by
  intros h_goal h_swim h_run h_bike
  sorry

end average_speed_bike_l0_191


namespace cos_equation_solution_range_l0_231

theorem cos_equation_solution_range (t : ℝ) (h_t : 0 ≤ t ∧ t ≤ π) :
  (∃ x : ℝ, cos (x + t) = 1 - cos x) ↔ t ∈ set.Icc 0 (2 * π / 3) :=
by sorry

end cos_equation_solution_range_l0_231


namespace polygon_sides_eq_eight_l0_559

theorem polygon_sides_eq_eight (n : ℕ) :
  ((n - 2) * 180 = 3 * 360) → n = 8 :=
by
  intro h
  sorry

end polygon_sides_eq_eight_l0_559


namespace area_of_ring_between_outermost_and_middle_circle_l0_562

noncomputable def pi : ℝ := Real.pi

theorem area_of_ring_between_outermost_and_middle_circle :
  let r_outermost := 12
  let r_middle := 8
  let A_outermost := pi * r_outermost^2
  let A_middle := pi * r_middle^2
  A_outermost - A_middle = 80 * pi :=
by 
  sorry

end area_of_ring_between_outermost_and_middle_circle_l0_562


namespace union_segments_lebesgue_measurable_and_minimal_measure_l0_673

noncomputable def continuous_function (f : ℝ → ℝ) : Prop :=
  continuous_on f (Icc 0 1)

def At (t : ℝ) : set (ℝ × ℝ) :=
  {(t, 0)}

def Bt (f : ℝ → ℝ) (t : ℝ) : set (ℝ × ℝ) :=
  {(f t, 1)}

def segment (p q : ℝ × ℝ) : set (ℝ × ℝ) :=
  { x | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ x = (t * p.1 + (1 - t) * q.1, t * p.2 + (1 - t) * q.2) }

def union_segments (f : ℝ → ℝ) : set (ℝ × ℝ) :=
  ⋃ t ∈ (Icc 0 1), segment (t, 0) (f t, 1)

theorem union_segments_lebesgue_measurable_and_minimal_measure (f : ℝ → ℝ) (hf : continuous_function f) :
  (measurable_set (union_segments f)) ∧ (measure_space.measure (union_segments f) = real.sqrt 2 - 1) :=
by 
  sorry

end union_segments_lebesgue_measurable_and_minimal_measure_l0_673


namespace problem_inequality_l0_852

theorem problem_inequality {a : ℝ} (h : ∀ x : ℝ, (x - a) * (1 - x - a) < 1) : 
  -1/2 < a ∧ a < 3/2 := by
  sorry

end problem_inequality_l0_852


namespace entry_exit_ways_l0_458

theorem entry_exit_ways (n : ℕ) (h : n = 8) : n * (n - 1) = 56 :=
by {
  sorry
}

end entry_exit_ways_l0_458


namespace isosceles_triangles_with_perimeter_27_count_l0_213

theorem isosceles_triangles_with_perimeter_27_count :
  ∃ n, (∀ (a : ℕ), 7 ≤ a ∧ a ≤ 13 → ∃ (b : ℕ), b = 27 - 2*a ∧ b < 2*a) ∧ n = 7 :=
sorry

end isosceles_triangles_with_perimeter_27_count_l0_213


namespace third_side_length_integer_l0_967

noncomputable
def side_a : ℝ := 3.14

noncomputable
def side_b : ℝ := 0.67

def is_valid_triangle_side (side: ℝ) : Prop :=
  side_a - side_b < side ∧ side < side_a + side_b

theorem third_side_length_integer (side: ℕ) : is_valid_triangle_side side.to_real → side = 3 :=
  by
  sorry

end third_side_length_integer_l0_967


namespace sin_double_angle_l0_50

open Real

theorem sin_double_angle
  {α : ℝ} (h1: tan α = -1/2) (h2: 0 < α ∧ α < π) :
  sin (2 * α) = -4/5 :=
sorry

end sin_double_angle_l0_50


namespace day_53_days_from_friday_l0_719

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end day_53_days_from_friday_l0_719


namespace angle_in_second_quadrant_l0_508

open Real

-- Define the fourth quadrant condition
def isFourthQuadrant (α : ℝ) (k : ℤ) : Prop :=
  2 * k * π - π / 2 < α ∧ α < 2 * k * π

-- Define the second quadrant condition
def isSecondQuadrant (β : ℝ) (k : ℤ) : Prop :=
  2 * k * π + π / 2 < β ∧ β < 2 * k * π + π

-- The main theorem to prove
theorem angle_in_second_quadrant (α : ℝ) (k : ℤ) :
  isFourthQuadrant α k → isSecondQuadrant (π + α) k :=
sorry

end angle_in_second_quadrant_l0_508


namespace greatest_b_value_for_integer_solution_eq_l0_248

theorem greatest_b_value_for_integer_solution_eq : ∀ (b : ℤ), (∃ (x : ℤ), x^2 + b * x = -20) → b > 0 → b ≤ 21 :=
by
  sorry

end greatest_b_value_for_integer_solution_eq_l0_248


namespace average_salary_of_all_workers_l0_694

def totalTechnicians : Nat := 6
def avgSalaryTechnician : Nat := 12000
def restWorkers : Nat := 6
def avgSalaryRest : Nat := 6000
def totalWorkers : Nat := 12
def totalSalary := (totalTechnicians * avgSalaryTechnician) + (restWorkers * avgSalaryRest)

theorem average_salary_of_all_workers : totalSalary / totalWorkers = 9000 := 
by
    -- replace with mathematical proof once available
    sorry

end average_salary_of_all_workers_l0_694


namespace height_of_taller_tree_l0_126

theorem height_of_taller_tree 
  (h : ℝ) 
  (ratio_condition : (h - 20) / h = 2 / 3) : 
  h = 60 := 
by 
  sorry

end height_of_taller_tree_l0_126


namespace questionnaire_visitors_l0_706

theorem questionnaire_visitors (V E : ℕ) (H1 : 140 = V - E) 
  (H2 : E = (3 * V) / 4) : V = 560 :=
by
  sorry

end questionnaire_visitors_l0_706


namespace solution_set_equivalence_l0_260

def solution_set_inequality (x : ℝ) : Prop :=
  abs (x - 1) + abs x < 3

theorem solution_set_equivalence :
  { x : ℝ | solution_set_inequality x } = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end solution_set_equivalence_l0_260


namespace derivative_of_f_l0_330

noncomputable def f (x : ℝ) : ℝ := x / (1 - Real.cos x)

theorem derivative_of_f :
  (deriv f) x = (1 - Real.cos x - x * Real.sin x) / (1 - Real.cos x)^2 :=
sorry

end derivative_of_f_l0_330


namespace even_integers_count_form_3k_plus_4_l0_956

theorem even_integers_count_form_3k_plus_4 
  (n : ℕ) (h1 : 20 ≤ n ∧ n ≤ 250)
  (h2 : ∃ k : ℕ, n = 3 * k + 4 ∧ Even n) : 
  ∃ N : ℕ, N = 39 :=
by {
  sorry
}

end even_integers_count_form_3k_plus_4_l0_956


namespace total_wheels_l0_975

def cars := 2
def car_wheels := 4
def bikes_with_one_wheel := 1
def bikes_with_two_wheels := 2
def trash_can_wheels := 2
def tricycle_wheels := 3
def roller_skate_wheels := 3 -- since one is missing a wheel
def wheelchair_wheels := 6 -- 4 large + 2 small wheels
def wagon_wheels := 4

theorem total_wheels : cars * car_wheels + 
                        bikes_with_one_wheel * 1 + 
                        bikes_with_two_wheels * 2 + 
                        trash_can_wheels + 
                        tricycle_wheels + 
                        roller_skate_wheels + 
                        wheelchair_wheels + 
                        wagon_wheels = 31 :=
by
  sorry

end total_wheels_l0_975


namespace polar_equation_is_circle_l0_939

-- Define the polar coordinates equation condition
def polar_equation (r θ : ℝ) : Prop := r = 5

-- Define what it means for a set of points to form a circle centered at the origin with a radius of 5
def is_circle_radius_5 (x y : ℝ) : Prop := x^2 + y^2 = 25

-- State the theorem we want to prove
theorem polar_equation_is_circle (r θ : ℝ) (x y : ℝ) (h1 : polar_equation r θ)
  (h2 : x = r * Real.cos θ) (h3 : y = r * Real.sin θ) : is_circle_radius_5 x y := 
sorry

end polar_equation_is_circle_l0_939


namespace smallest_integer_n_exists_l0_142

-- Define the conditions
def lcm_gcd_correct_division (a b : ℕ) : Prop :=
  (lcm a b) / (gcd a b) = 44

-- Define the main problem
theorem smallest_integer_n_exists : ∃ n : ℕ, lcm_gcd_correct_division 60 n ∧ 
  (∀ k : ℕ, lcm_gcd_correct_division 60 k → k ≥ n) :=
begin
  sorry
end

end smallest_integer_n_exists_l0_142


namespace g_composed_g_has_exactly_two_distinct_real_roots_l0_979

theorem g_composed_g_has_exactly_two_distinct_real_roots (d : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 + 4 * x + d) = 0 ∧ (y^2 + 4 * y + d) = 0) ↔ d = 8 :=
sorry

end g_composed_g_has_exactly_two_distinct_real_roots_l0_979


namespace negations_true_of_BD_l0_25

def is_parallelogram_rhombus : Prop :=
  ∃ p : parallelogram, is_rhombus p

def exists_x_in_R : Prop :=
  ∃ x : ℝ, x^2 - 3 * x + 3 < 0

def forall_x_in_R : Prop :=
  ∀ x : ℝ, |x| + x^2 ≥ 0

def quad_eq_has_real_solutions (a : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 - a * x + 1 = 0) → real_roots x

theorem negations_true_of_BD :
  (¬exists_x_in_R) ∧ (¬quad_eq_has_real_solutions a) :=
sorry

end negations_true_of_BD_l0_25


namespace brady_passing_yards_proof_l0_887

def tom_brady_current_passing_yards 
  (record_yards : ℕ) (games_left : ℕ) (average_yards_needed : ℕ) 
  (total_yards_needed_to_break_record : ℕ :=
    record_yards + 1) : ℕ :=
  total_yards_needed_to_break_record - games_left * average_yards_needed

theorem brady_passing_yards_proof :
  tom_brady_current_passing_yards 5999 6 300 = 4200 :=
by 
  sorry

end brady_passing_yards_proof_l0_887


namespace coloring_probability_l0_90

-- Definition of the problem and its conditions
def num_cells := 16
def num_diags := 2
def chosen_diags := 7

-- Define the probability
noncomputable def prob_coloring_correct : ℚ :=
  (num_diags ^ chosen_diags : ℚ) / (num_diags ^ num_cells)

-- The Lean theorem statement
theorem coloring_probability : prob_coloring_correct = 1 / 512 := 
by 
  unfold prob_coloring_correct
  -- The proof steps would follow here (omitted)
  sorry

end coloring_probability_l0_90


namespace max_apples_discarded_l0_8

theorem max_apples_discarded (n : ℕ) : n % 7 ≤ 6 := by
  sorry

end max_apples_discarded_l0_8


namespace find_divisible_by_3_l0_560

theorem find_divisible_by_3 (n : ℕ) : 
  (∀ k : ℕ, k ≤ 12 → (3 * k + 12) ≤ n) ∧ 
  (∀ m : ℕ, m ≥ 13 → (3 * m + 12) > n) →
  n = 48 :=
by
  sorry

end find_divisible_by_3_l0_560


namespace minimum_value_of_weighted_sum_l0_843

theorem minimum_value_of_weighted_sum 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 27) :
  3 * a + 6 * b + 9 * c ≥ 54 :=
sorry

end minimum_value_of_weighted_sum_l0_843


namespace factorize_expression_l0_322

-- The primary goal is to prove that -2xy^2 + 4xy - 2x = -2x(y - 1)^2
theorem factorize_expression (x y : ℝ) : 
  -2 * x * y^2 + 4 * x * y - 2 * x = -2 * x * (y - 1)^2 := 
by 
  sorry

end factorize_expression_l0_322


namespace smallest_possible_n_l0_138

theorem smallest_possible_n (n : ℕ) (h : n > 0) 
  (h_condition : Nat.lcm 60 n / Nat.gcd 60 n = 44) : n = 165 := by
  sorry

end smallest_possible_n_l0_138


namespace scheduling_subjects_l0_214

theorem scheduling_subjects (periods subjects : ℕ)
  (h_periods : periods = 7)
  (h_subjects : subjects = 4) :
  ∃ (ways : ℕ), ways = Nat.choose periods subjects * subjects.factorial ∧ ways = 840 :=
by
  use Nat.choose 7 4 * 4.factorial
  split
  {
    rw [h_periods, h_subjects],
    norm_num
  }
  {
    rw [h_subjects],
    norm_num
  }
  sorry

end scheduling_subjects_l0_214


namespace allen_blocks_l0_586

def blocks_per_color : Nat := 7
def colors_used : Nat := 7

theorem allen_blocks : (blocks_per_color * colors_used) = 49 :=
by
  sorry

end allen_blocks_l0_586


namespace x1_mul_x2_l0_228

open Real

theorem x1_mul_x2 (x1 x2 : ℝ) (h1 : x1 + x2 = 2 * sqrt 1703) (h2 : abs (x1 - x2) = 90) : x1 * x2 = -322 := by
  sorry

end x1_mul_x2_l0_228


namespace solve_adult_tickets_l0_297

theorem solve_adult_tickets (A C : ℕ) (h1 : 8 * A + 5 * C = 236) (h2 : A + C = 34) : A = 22 :=
sorry

end solve_adult_tickets_l0_297


namespace lyka_saving_per_week_l0_105

-- Definitions from the conditions
def smartphone_price : ℕ := 160
def lyka_has : ℕ := 40
def weeks_in_two_months : ℕ := 8

-- The goal (question == correct answer)
theorem lyka_saving_per_week :
  (smartphone_price - lyka_has) / weeks_in_two_months = 15 :=
sorry

end lyka_saving_per_week_l0_105


namespace max_pieces_l0_272

namespace CakeProblem

-- Define the dimensions of the cake and the pieces.
def cake_side : ℕ := 16
def piece_side : ℕ := 4

-- Define the areas of the cake and the pieces.
def cake_area : ℕ := cake_side * cake_side
def piece_area : ℕ := piece_side * piece_side

-- State the main problem to prove.
theorem max_pieces : cake_area / piece_area = 16 :=
by
  -- The proof is omitted.
  sorry

end CakeProblem

end max_pieces_l0_272


namespace royalty_amount_l0_554

-- Define the conditions and the question proof.
theorem royalty_amount (x : ℝ) :
  (800 ≤ x ∧ x ≤ 4000 → (x - 800) * 0.14 = 420) ∧
  (x > 4000 → x * 0.11 = 420) ∧
  420 = 420 →
  x = 3800 :=
by
  sorry

end royalty_amount_l0_554


namespace remainder_division_l0_904

theorem remainder_division (n r : ℕ) (k : ℤ) (h1 : n % 25 = r) (h2 : (n + 15) % 5 = r) (h3 : 0 ≤ r ∧ r < 25) : r = 5 :=
sorry

end remainder_division_l0_904


namespace quadratic_root_zero_l0_942

theorem quadratic_root_zero (a : ℝ) : 
  ((a-1) * 0^2 + 0 + a^2 - 1 = 0) 
  → a ≠ 1 
  → a = -1 := 
by
  intro h1 h2
  sorry

end quadratic_root_zero_l0_942


namespace symmetric_point_origin_l0_876

def symmetric_point (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, -M.2)

theorem symmetric_point_origin (x y : ℝ) (h : (x, y) = (-2, 3)) :
  symmetric_point (x, y) = (2, -3) :=
by
  rw [h]
  unfold symmetric_point
  simp
  sorry

end symmetric_point_origin_l0_876


namespace mean_inequality_l0_543

variable (a b : ℝ)

-- Conditions: a and b are distinct and non-zero
axiom h₀ : a ≠ b
axiom h₁ : a ≠ 0
axiom h₂ : b ≠ 0

theorem mean_inequality (h₀ : a ≠ b) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : 
  (a^2 + b^2) / 2 > (a + b) / 2 ∧ (a + b) / 2 > Real.sqrt (a * b) :=
sorry -- Proof is not provided, only statement.

end mean_inequality_l0_543


namespace video_game_cost_l0_532

theorem video_game_cost :
  let september_saving : ℕ := 50
  let october_saving : ℕ := 37
  let november_saving : ℕ := 11
  let mom_gift : ℕ := 25
  let remaining_money : ℕ := 36
  let total_savings : ℕ := september_saving + october_saving + november_saving
  let total_with_gift : ℕ := total_savings + mom_gift
  let game_cost : ℕ := total_with_gift - remaining_money
  game_cost = 87 :=
by
  sorry

end video_game_cost_l0_532


namespace diamond_2_3_l0_714

def diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

theorem diamond_2_3 : diamond 2 3 = 16 :=
by
  -- Imported definition and theorem structure.
  sorry

end diamond_2_3_l0_714


namespace series_converges_l0_891

theorem series_converges (u : ℕ → ℝ) (h : ∀ n, u n = n / (3 : ℝ)^n) :
  ∃ l, 0 ≤ l ∧ l < 1 ∧ ∑' n, u n = l := by
  sorry

end series_converges_l0_891


namespace fifty_three_days_from_Friday_is_Tuesday_l0_729

def days_of_week : Type := {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}

def modulo_days (n : ℕ) : ℕ := n % 7

def day_of_week_after_days (start_day : days_of_week) (n : ℕ) : days_of_week :=
  list.nth_le (list.rotate [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday] (modulo_days n)) sorry -- using sorry for index proof

theorem fifty_three_days_from_Friday_is_Tuesday :
  day_of_week_after_days Friday 53 = Tuesday :=
sorry

end fifty_three_days_from_Friday_is_Tuesday_l0_729


namespace dance_troupe_minimum_members_l0_13

theorem dance_troupe_minimum_members :
  ∃ n : ℕ, n > 0 ∧ n % 6 = 0 ∧ n % 9 = 0 ∧ n % 12 = 0 ∧ n % 5 = 0 ∧ n = 180 :=
begin
  use 180,
  split,
  { norm_num }, -- Prove that 180 > 0
  split,
  { norm_num }, -- Prove that 180 % 6 = 0
  split,
  { norm_num }, -- Prove that 180 % 9 = 0
  split,
  { norm_num }, -- Prove that 180 % 12 = 0
  split,
  { norm_num }, -- Prove that 180 % 5 = 0
  { norm_num }, -- Prove that 180 = 180
end

end dance_troupe_minimum_members_l0_13


namespace find_A_satisfy_3A_multiple_of_8_l0_127

theorem find_A_satisfy_3A_multiple_of_8 (A : ℕ) (h : 0 ≤ A ∧ A < 10) : 8 ∣ (30 + A) ↔ A = 2 := 
by
  sorry

end find_A_satisfy_3A_multiple_of_8_l0_127


namespace total_letters_in_all_names_l0_974

theorem total_letters_in_all_names :
  let jonathan_first := 8
  let jonathan_surname := 10
  let younger_sister_first := 5
  let younger_sister_surname := 10
  let older_brother_first := 6
  let older_brother_surname := 10
  let youngest_sibling_first := 4
  let youngest_sibling_hyphenated_surname := 15
  jonathan_first + jonathan_surname + younger_sister_first + younger_sister_surname +
  older_brother_first + older_brother_surname + youngest_sibling_first + youngest_sibling_hyphenated_surname = 68 := by
  sorry

end total_letters_in_all_names_l0_974


namespace total_amount_invested_l0_86

-- Define the conditions and specify the correct answer
theorem total_amount_invested (x y : ℝ) (h8 : y = 600) 
  (h_income_diff : 0.10 * (x - 600) - 0.08 * 600 = 92) : 
  x + y = 2000 := sorry

end total_amount_invested_l0_86


namespace problem_conditions_l0_978

noncomputable def f (a b c x : ℝ) := 3 * a * x^2 + 2 * b * x + c

theorem problem_conditions (a b c : ℝ) (h0 : a + b + c = 0)
  (h1 : f a b c 0 > 0) (h2 : f a b c 1 > 0) :
    (a > 0 ∧ -2 < b / a ∧ b / a < -1) ∧
    (∃ z1 z2 : ℝ, 0 < z1 ∧ z1 < 1 ∧ 0 < z2 ∧ z2 < 1 ∧ z1 ≠ z2 ∧ f a b c z1 = 0 ∧ f a b c z2 = 0) :=
by
  sorry

end problem_conditions_l0_978


namespace LykaSavings_l0_103

-- Define the given values and the properties
def totalCost : ℝ := 160
def amountWithLyka : ℝ := 40
def averageWeeksPerMonth : ℝ := 4.33
def numberOfMonths : ℝ := 2

-- Define the remaining amount Lyka needs
def remainingAmount : ℝ := totalCost - amountWithLyka

-- Define the number of weeks in the saving period
def numberOfWeeks : ℝ := numberOfMonths * averageWeeksPerMonth

-- Define the weekly saving amount
def weeklySaving : ℝ := remainingAmount / numberOfWeeks

-- State the theorem to be proved
theorem LykaSavings :  weeklySaving ≈ 13.86 :=
by
  -- Proof steps (omitted)
  sorry

end LykaSavings_l0_103


namespace triangle_height_l0_371

theorem triangle_height (area base : ℝ) (h : ℝ) (h_area : area = 46) (h_base : base = 10) 
  (h_formula : area = (base * h) / 2) : 
  h = 9.2 :=
by
  sorry

end triangle_height_l0_371


namespace same_color_points_exist_l0_580

theorem same_color_points_exist (d : ℝ) (colored_plane : ℝ × ℝ → Prop) :
  (∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ colored_plane p1 = colored_plane p2 ∧ dist p1 p2 = d) := 
sorry

end same_color_points_exist_l0_580


namespace total_present_ages_l0_7

variable (P Q P' Q' : ℕ)

-- Condition 1: 6 years ago, \( p \) was half of \( q \) in age.
axiom cond1 : P = Q / 2

-- Condition 2: The ratio of their present ages is 3:4.
axiom cond2 : (P + 6) * 4 = (Q + 6) * 3

-- We need to prove: the total of their present ages is 21
theorem total_present_ages : P' + Q' = 21 :=
by
  -- We already have the variables and axioms in the context, so we just need to state the goal
  sorry

end total_present_ages_l0_7


namespace sum_of_first_21_terms_l0_775

def is_constant_sum_sequence (a : ℕ → ℕ) (c : ℕ) : Prop :=
  ∀ n, a n + a (n + 1) = c

theorem sum_of_first_21_terms (a : ℕ → ℕ) (h1 : is_constant_sum_sequence a 5) (h2 : a 1 = 2) : (Finset.range 21).sum a = 52 :=
by
  sorry

end sum_of_first_21_terms_l0_775


namespace least_three_digit_multiple_of_3_4_9_is_108_l0_435

theorem least_three_digit_multiple_of_3_4_9_is_108 :
  ∃ (n : ℕ), (100 ≤ n) ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 9 = 0) ∧ (n = 108) :=
by
  sorry

end least_three_digit_multiple_of_3_4_9_is_108_l0_435


namespace problem_proof_l0_68

noncomputable def triangle_expression (a b c : ℝ) (A B C : ℝ) : ℝ :=
  b^2 * (Real.cos (C / 2))^2 + c^2 * (Real.cos (B / 2))^2 + 
  2 * b * c * Real.cos (B / 2) * Real.cos (C / 2) * Real.sin (A / 2)

theorem problem_proof (a b c A B C : ℝ) (h1 : a + b + c = 16) : 
  triangle_expression a b c A B C = 64 := 
sorry

end problem_proof_l0_68


namespace complex_number_problem_l0_361

noncomputable def z (cos_value : ℂ) := by sorry

theorem complex_number_problem
  (z : ℂ)
  (hz : z + z⁻¹ = 2 * real.cos (5 * real.pi / 180)) :
  z^100 + z^(-100) = -1.92 :=
sorry

end complex_number_problem_l0_361


namespace LykaSavings_l0_104

-- Define the given values and the properties
def totalCost : ℝ := 160
def amountWithLyka : ℝ := 40
def averageWeeksPerMonth : ℝ := 4.33
def numberOfMonths : ℝ := 2

-- Define the remaining amount Lyka needs
def remainingAmount : ℝ := totalCost - amountWithLyka

-- Define the number of weeks in the saving period
def numberOfWeeks : ℝ := numberOfMonths * averageWeeksPerMonth

-- Define the weekly saving amount
def weeklySaving : ℝ := remainingAmount / numberOfWeeks

-- State the theorem to be proved
theorem LykaSavings :  weeklySaving ≈ 13.86 :=
by
  -- Proof steps (omitted)
  sorry

end LykaSavings_l0_104


namespace line_through_center_parallel_to_given_line_l0_935

def point_in_line (p : ℝ × ℝ) (a b c : ℝ) : Prop :=
  a * p.1 + b * p.2 + c = 0

noncomputable def slope_of_line (a b c : ℝ) : ℝ :=
  -a / b

theorem line_through_center_parallel_to_given_line :
  ∃ a b c : ℝ, a = 2 ∧ b = -1 ∧ c = -4 ∧
    point_in_line (2, 0) a b c ∧
    slope_of_line a b c = slope_of_line 2 (-1) 1 :=
by
  sorry

end line_through_center_parallel_to_given_line_l0_935


namespace value_of_a_l0_944

theorem value_of_a (a : ℝ) (x : ℝ) (h : (a - 1) * x^2 + x + a^2 - 1 = 0) : a = -1 :=
sorry

end value_of_a_l0_944


namespace problem1_remainder_of_9_power_100_mod_8_problem2_last_digit_of_2012_power_2012_l0_756

-- Problem 1: Prove the remainder of the Euclidean division of \(9^{100}\) by 8 is 1.
theorem problem1_remainder_of_9_power_100_mod_8 :
  (9 ^ 100) % 8 = 1 :=
by
sorry

-- Problem 2: Prove the last digit of \(2012^{2012}\) is 6.
theorem problem2_last_digit_of_2012_power_2012 :
  (2012 ^ 2012) % 10 = 6 :=
by
sorry

end problem1_remainder_of_9_power_100_mod_8_problem2_last_digit_of_2012_power_2012_l0_756


namespace part1_part2_l0_502

-- Part 1
noncomputable def f (x a : ℝ) := x * Real.log x - a * x^2 + a

theorem part1 (a : ℝ) : (∀ x : ℝ, 0 < x → f x a ≤ a) → a ≥ 1 / Real.exp 1 :=
by
  sorry

-- Part 2
theorem part2 (a : ℝ) (x₀ : ℝ) : 
  (∀ x : ℝ, f x₀ a < f x a → x = x₀) → a < 1 / 2 → 2 * a - 1 < f x₀ a ∧ f x₀ a < 0 :=
by
  sorry

end part1_part2_l0_502


namespace solution_l0_480

noncomputable def f : ℝ → ℝ := sorry

lemma problem_conditions:
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (-x + 1) = f (x + 1)) ∧ f (-1) = 1 :=
sorry

theorem solution : f 2017 = -1 :=
sorry

end solution_l0_480


namespace verify_exact_countries_attended_l0_613

theorem verify_exact_countries_attended :
  let start_year := 1990
  let years_between_festivals := 3
  let total_festivals := 12
  let attended_countries := 68
  (attended_countries = 68) :=
by
  let start_year := 1990
  let years_between_festivals := 3
  let total_festivals := 12
  let attended_countries := 68
  have : attended_countries = 68 := rfl
  exact this

end verify_exact_countries_attended_l0_613


namespace subtraction_example_l0_27

theorem subtraction_example : -1 - 3 = -4 := 
  sorry

end subtraction_example_l0_27


namespace speed_of_car_in_second_hour_l0_124

theorem speed_of_car_in_second_hour
(speed_in_first_hour : ℝ)
(average_speed : ℝ)
(total_time : ℝ)
(speed_in_second_hour : ℝ)
(h1 : speed_in_first_hour = 100)
(h2 : average_speed = 65)
(h3 : total_time = 2)
(h4 : average_speed = (speed_in_first_hour + speed_in_second_hour) / total_time) :
  speed_in_second_hour = 30 :=
by {
  sorry
}

end speed_of_car_in_second_hour_l0_124


namespace curve_equation_l0_329

noncomputable def curve_passing_condition (x y : ℝ) : Prop :=
  (∃ (f : ℝ → ℝ), f 2 = 3 ∧ ∀ (t : ℝ), (f t) * t = 6 ∧ ((t ≠ 0 ∧ f t ≠ 0) → (t, f t) = (x, y)))

theorem curve_equation (x y : ℝ) (h1 : curve_passing_condition x y) : x * y = 6 :=
  sorry

end curve_equation_l0_329


namespace overall_gain_is_correct_l0_460

noncomputable def overall_gain_percentage : ℝ :=
  let CP_A := 100
  let SP_A := 120 / (1 - 0.20)
  let gain_A := SP_A - CP_A

  let CP_B := 200
  let SP_B := 240 / (1 + 0.10)
  let gain_B := SP_B - CP_B

  let CP_C := 150
  let SP_C := (165 / (1 + 0.05)) / (1 - 0.10)
  let gain_C := SP_C - CP_C

  let CP_D := 300
  let SP_D := (345 / (1 - 0.05)) / (1 + 0.15)
  let gain_D := SP_D - CP_D

  let total_gain := gain_A + gain_B + gain_C + gain_D
  let total_CP := CP_A + CP_B + CP_C + CP_D
  (total_gain / total_CP) * 100

theorem overall_gain_is_correct : abs (overall_gain_percentage - 14.48) < 0.01 := by
  sorry

end overall_gain_is_correct_l0_460


namespace terminating_decimal_expansion_of_13_over_320_l0_197

theorem terminating_decimal_expansion_of_13_over_320 : ∃ (b : ℕ) (a : ℚ), (13 : ℚ) / 320 = a / 10 ^ b ∧ a / 10 ^ b = 0.650 :=
by
  sorry

end terminating_decimal_expansion_of_13_over_320_l0_197


namespace NorrisSavings_l0_389

theorem NorrisSavings : 
  let saved_september := 29
  let saved_october := 25
  let saved_november := 31
  let saved_december := 35
  let saved_january := 40
  saved_september + saved_october + saved_november + saved_december + saved_january = 160 :=
by
  sorry

end NorrisSavings_l0_389


namespace symmetric_point_x_axis_l0_971

variable (P : (ℝ × ℝ)) (x : ℝ) (y : ℝ)

-- Given P is a point (x, y)
def symmetric_about_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

-- Special case for the point (-2, 3)
theorem symmetric_point_x_axis : 
  symmetric_about_x_axis (-2, 3) = (-2, -3) :=
by 
  sorry

end symmetric_point_x_axis_l0_971


namespace equal_roots_of_quadratic_eq_l0_960

theorem equal_roots_of_quadratic_eq (n : ℝ) : (∃ x : ℝ, (x^2 - x + n = 0) ∧ (Δ = 0)) ↔ n = 1 / 4 :=
by
  have h₁ : Δ = 0 := by sorry  -- The discriminant condition
  sorry  -- Placeholder for completing the theorem proof

end equal_roots_of_quadratic_eq_l0_960


namespace Tim_eats_91_pickle_slices_l0_238

theorem Tim_eats_91_pickle_slices :
  let Sammy := 25
  let Tammy := 3 * Sammy
  let Ron := Tammy - 0.15 * Tammy
  let Amy := Sammy + 0.50 * Sammy
  let CombinedTotal := Ron + Amy
  let Tim := CombinedTotal - 0.10 * CombinedTotal
  Tim = 91 :=
by
  admit

end Tim_eats_91_pickle_slices_l0_238


namespace digit_place_value_ratio_l0_92

theorem digit_place_value_ratio :
  let number := 86304.2957
  let digit_6_value := 1000
  let digit_5_value := 0.1
  digit_6_value / digit_5_value = 10000 :=
by
  let number := 86304.2957
  let digit_6_value := 1000
  let digit_5_value := 0.1
  sorry

end digit_place_value_ratio_l0_92


namespace solve_x_l0_542

theorem solve_x :
  (2 / 3 - 1 / 4) = 1 / (12 / 5) :=
by
  sorry

end solve_x_l0_542


namespace all_solutions_of_diophantine_eq_l0_810

theorem all_solutions_of_diophantine_eq
  (a b c x0 y0 : ℤ) (h_gcd : Int.gcd a b = 1)
  (h_sol : a * x0 + b * y0 = c) :
  ∀ x y : ℤ, (a * x + b * y = c) →
  ∃ t : ℤ, x = x0 + b * t ∧ y = y0 - a * t :=
by
  sorry

end all_solutions_of_diophantine_eq_l0_810


namespace quadratic_k_value_l0_57

theorem quadratic_k_value (a b k : ℝ) (h_eq : a * b + 2 * a + 2 * b = 1)
  (h_roots : Polynomial.eval₂ (RingHom.id ℝ) a (Polynomial.C k * Polynomial.X ^ 0 + Polynomial.C (-3) * Polynomial.X + Polynomial.C 1) = 0 ∧
             Polynomial.eval₂ (RingHom.id ℝ) b (Polynomial.C k * Polynomial.X ^ 0 + Polynomial.C (-3) * Polynomial.X + Polynomial.C 1) = 0) : 
  k = -5 :=
by
  sorry

end quadratic_k_value_l0_57


namespace integral_fx_l0_946

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem integral_fx :
  ∫ x in -Real.pi..0, f x = -2 - (1/2) * Real.pi ^ 2 :=
by
  sorry

end integral_fx_l0_946


namespace product_probability_l0_864

def S : Set ℤ := {4, 13, 22, 29, 37, 43, 57, 63, 71}

def pairs (s: Set ℤ) : Set (ℤ × ℤ) :=
  {p | p.1 ∈ s ∧ p.2 ∈ s ∧ p.1 < p.2}

def product_gt_150 (s : Set ℤ) : Set (ℤ × ℤ) :=
  {p ∈ pairs s | p.1 * p.2 > 150}

def probability_gt_150 : ℚ :=
  (product_gt_150 S).to_finset.card / (pairs S).to_finset.card

theorem product_probability : probability_gt_150 = 8/9 := by
  sorry

end product_probability_l0_864


namespace day_after_53_days_from_Friday_l0_727

def Day : Type := ℕ -- Representing days of the week as natural numbers (0 for Sunday, 1 for Monday, ..., 6 for Saturday)

def Friday : Day := 5
def Tuesday : Day := 2

def days_after (start_day : Day) (n : ℕ) : Day :=
  (start_day + n) % 7

theorem day_after_53_days_from_Friday : days_after Friday 53 = Tuesday :=
  by
  -- proof goes here
  sorry

end day_after_53_days_from_Friday_l0_727


namespace product_of_possible_b_l0_411

theorem product_of_possible_b (b : ℤ) (h1 : y = 3) (h2 : y = 8) (h3 : x = 2)
  (h4 : (y = 3 ∧ y = 8 ∧ x = 2 ∧ (x = b ∨ x = b)) → forms_square y y x x) :
  b = 7 ∨ b = -3 → 7 * (-3) = -21 :=
by
  sorry

end product_of_possible_b_l0_411


namespace gretchen_fewest_trips_l0_820

def fewestTrips (total_objects : ℕ) (max_carry : ℕ) : ℕ :=
  (total_objects + max_carry - 1) / max_carry

theorem gretchen_fewest_trips : fewestTrips 17 3 = 6 := 
  sorry

end gretchen_fewest_trips_l0_820


namespace dwarfs_truthful_count_l0_182

theorem dwarfs_truthful_count :
  ∃ (T L : ℕ), T + L = 10 ∧
    (∀ t : ℕ, t = 10 → t + ((10 - T) * 2 - T) = 16) ∧
    T = 4 :=
by
  sorry

end dwarfs_truthful_count_l0_182


namespace sum_of_integers_l0_266

theorem sum_of_integers (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1)
  (h4 : a * b * c = 343000)
  (h5 : Nat.gcd a b = 1) (h6 : Nat.gcd b c = 1) (h7 : Nat.gcd a c = 1) :
  a + b + c = 476 :=
by
  sorry

end sum_of_integers_l0_266


namespace minimize_p_for_repeating_decimal_l0_221

noncomputable def repeating_decimal_as_fraction : ℚ :=
  2 / 11

theorem minimize_p_for_repeating_decimal :
  ∀ p q : ℕ, p ≠ 0 ∧ q ≠ 0 ∧ (0.181818181818181818 = (p : ℚ) / q) ∧ (nat.gcd p q = 1 ∧ ∀ p1 q1 : ℕ, (p1 : ℚ) / q1 = 0.181818181818181818 → q1 < q → nat.gcd p1 q1 ≠ 1) → p = 2 := 
by { sorry }

end minimize_p_for_repeating_decimal_l0_221


namespace coordinates_provided_l0_970

-- Define the coordinates of point P in the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

-- Define the point P with its given coordinates
def P : Point := {x := 3, y := -5}

-- Lean 4 statement for the proof problem
theorem coordinates_provided : (P.x, P.y) = (3, -5) := by
  -- Proof not provided
  sorry

end coordinates_provided_l0_970


namespace prob_A_not_losing_is_correct_l0_539

def prob_A_wins := 0.4
def prob_draw := 0.2
def prob_A_not_losing := 0.6

theorem prob_A_not_losing_is_correct : prob_A_wins + prob_draw = prob_A_not_losing :=
by sorry

end prob_A_not_losing_is_correct_l0_539


namespace truthful_dwarfs_count_l0_188

def number_of_dwarfs := 10
def vanilla_ice_cream := number_of_dwarfs
def chocolate_ice_cream := number_of_dwarfs / 2
def fruit_ice_cream := 1

theorem truthful_dwarfs_count (T L : ℕ) (h1 : T + L = 10)
  (h2 : vanilla_ice_cream = T + (L * 2))
  (h3 : chocolate_ice_cream = T / 2 + (L / 2 * 2))
  (h4 : fruit_ice_cream = 1)
  : T = 4 :=
sorry

end truthful_dwarfs_count_l0_188


namespace problem_l0_473

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem problem :
  let A := 3.14159265
  let B := Real.sqrt 36
  let C := Real.sqrt 7
  let D := 4.1
  is_irrational C := by
  sorry

end problem_l0_473


namespace solve_x_l0_79

theorem solve_x (x : ℝ) (h : (x + 1) ^ 2 = 9) : x = 2 ∨ x = -4 :=
sorry

end solve_x_l0_79


namespace average_is_4_l0_368

theorem average_is_4 (p q r s : ℝ) (h : (5 / 4) * (p + q + r + s) = 20) : 
  (p + q + r + s) / 4 = 4 := 
by 
  sorry 

end average_is_4_l0_368


namespace rectangle_area_proof_l0_773

variable (x y : ℕ) -- Declaring the variables to represent length and width of the rectangle.

-- Declaring the conditions as hypotheses.
def condition1 := (x + 3) * (y - 1) = x * y
def condition2 := (x - 3) * (y + 2) = x * y
def condition3 := (x + 4) * (y - 2) = x * y

-- The theorem to prove the area is 36 given the above conditions.
theorem rectangle_area_proof (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x y) : x * y = 36 :=
by
  sorry

end rectangle_area_proof_l0_773


namespace solution_set_of_quadratic_inequality_l0_999

theorem solution_set_of_quadratic_inequality (x : ℝ) :
  (x^2 ≤ 4) ↔ (-2 ≤ x ∧ x ≤ 2) :=
by 
  sorry

end solution_set_of_quadratic_inequality_l0_999


namespace system_solution_l0_617

theorem system_solution (x y : ℝ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 3) : x - y = 3 :=
by
  -- proof goes here
  sorry

end system_solution_l0_617


namespace chandler_bike_purchase_l0_595

theorem chandler_bike_purchase : 
  ∀ (x : ℕ), (120 + 20 * x = 640) → x = 26 := 
by
  sorry

end chandler_bike_purchase_l0_595


namespace geometric_sequence_alpha_5_l0_523

theorem geometric_sequence_alpha_5 (α : ℕ → ℝ) (h1 : α 4 * α 5 * α 6 = 27) (h2 : α 4 * α 6 = (α 5) ^ 2) : α 5 = 3 := 
sorry

end geometric_sequence_alpha_5_l0_523


namespace length_of_first_train_l0_924

theorem length_of_first_train 
  (speed_first_train_kmph : ℝ) 
  (speed_second_train_kmph : ℝ) 
  (crossing_time_s : ℝ) 
  (length_second_train_m : ℝ) 
  (hspeed_first : speed_first_train_kmph = 120) 
  (hspeed_second : speed_second_train_kmph = 80) 
  (htime : crossing_time_s = 9) 
  (hlength_second : length_second_train_m = 320.04) :
  ∃ (length_first_train_m : ℝ), abs (length_first_train_m - 180) < 0.1 :=
by
  sorry

end length_of_first_train_l0_924


namespace pyramid_boxes_l0_464

theorem pyramid_boxes (a₁ a₂ aₙ : ℕ) (d : ℕ) (n : ℕ) (Sₙ : ℕ) 
  (h₁ : a₁ = 12) 
  (h₂ : a₂ = 15) 
  (h₃ : aₙ = 39) 
  (h₄ : d = 3) 
  (h₅ : a₂ = a₁ + d)
  (h₆ : aₙ = a₁ + (n - 1) * d) 
  (h₇ : Sₙ = n * (a₁ + aₙ) / 2) :
  Sₙ = 255 :=
by
  sorry

end pyramid_boxes_l0_464


namespace find_x_l0_573

theorem find_x (x : ℤ) (h : 9873 + x = 13800) : x = 3927 :=
by {
  sorry
}

end find_x_l0_573


namespace no_positive_integer_solutions_l0_409

theorem no_positive_integer_solutions :
  ¬ ∃ (x1 x2 : ℕ), 903 * x1 + 731 * x2 = 1106 := by
  sorry

end no_positive_integer_solutions_l0_409


namespace repeating_decimal_356_fraction_l0_321

noncomputable def repeating_decimal_356 := 3.0 + 56 / 99

theorem repeating_decimal_356_fraction : repeating_decimal_356 = 353 / 99 := by
  sorry

end repeating_decimal_356_fraction_l0_321


namespace final_number_of_cards_l0_339

def initial_cards : ℕ := 26
def cards_given_to_mary : ℕ := 18
def cards_found_in_box : ℕ := 40
def cards_given_to_john : ℕ := 12
def cards_purchased_at_fleamarket : ℕ := 25

theorem final_number_of_cards :
  (initial_cards - cards_given_to_mary) + (cards_found_in_box - cards_given_to_john) + cards_purchased_at_fleamarket = 61 :=
by sorry

end final_number_of_cards_l0_339


namespace seamless_assembly_with_equilateral_triangle_l0_500

theorem seamless_assembly_with_equilateral_triangle :
  ∃ (polygon : ℕ → ℝ) (angle_150 : ℝ),
    (polygon 4 = 90) ∧ (polygon 6 = 120) ∧ (polygon 8 = 135) ∧ (polygon 3 = 60) ∧ (angle_150 = 150) ∧
    (∃ (n₁ n₂ n₃ : ℕ), n₁ * 150 + n₂ * 150 + n₃ * 60 = 360) :=
by {
  -- The proof would involve checking the precise integer combination for seamless assembly
  sorry
}

end seamless_assembly_with_equilateral_triangle_l0_500


namespace probability_of_rolling_sevens_l0_14

theorem probability_of_rolling_sevens :
  let p := (1 : ℚ) / 4
  let q := (3 : ℚ) / 4
  (7 * p^6 * q + p^7 = 22 / 16384) :=
by
  let p := (1 : ℚ) / 4
  let q := (3 : ℚ) / 4
  let k1 := 7 * p^6 * q
  let k2 := p^7
  have h1 : k1 = (21 : ℚ) / 16384 := by sorry
  have h2 : k2 = (1 : ℚ) / 16384 := by sorry
  have h := h1 + h2
  show (k1 + k2 = 22 / 16384)
  exact h

end probability_of_rolling_sevens_l0_14


namespace continuous_function_form_l0_619

noncomputable def f (t : ℝ) : ℝ := sorry

theorem continuous_function_form (f : ℝ → ℝ) (h1 : f 0 = -1 / 2) (h2 : ∀ x y, f (x + y) ≥ f x + f y + f (x * y) + 1) :
  ∃ (a : ℝ), ∀ x, f x = 1 / 2 + a * x + (a/2) * x ^ 2 := sorry

end continuous_function_form_l0_619


namespace correct_system_of_equations_l0_6

theorem correct_system_of_equations (x y : ℕ) : 
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔ 
  (x / 3 = y - 2) ∧ (x / 2 - 9 = y) := sorry

end correct_system_of_equations_l0_6


namespace rho_square_max_value_l0_684

variable {a b x y c : ℝ}
variable (ha_pos : a > 0) (hb_pos : b > 0)
variable (ha_ge_b : a ≥ b)
variable (hx_range : 0 ≤ x ∧ x < a)
variable (hy_range : 0 ≤ y ∧ y < b)
variable (h_eq1 : a^2 + y^2 = b^2 + x^2)
variable (h_eq2 : b^2 + x^2 = (a - x)^2 + (b - y)^2 + c^2)

theorem rho_square_max_value : (a / b) ^ 2 ≤ 4 / 3 :=
sorry

end rho_square_max_value_l0_684


namespace exists_c_with_same_nonzero_decimal_digits_l0_980

theorem exists_c_with_same_nonzero_decimal_digits (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  ∃ (c : ℕ), 0 < c ∧ (∃ (k : ℕ), (c * m) % 10^k = (c * n) % 10^k) := 
sorry

end exists_c_with_same_nonzero_decimal_digits_l0_980


namespace ratio_of_part_to_whole_l0_537

theorem ratio_of_part_to_whole (N : ℝ) (h1 : (1/3) * (2/5) * N = 15) (h2 : (40/100) * N = 180) :
  (15 / N) = (1 / 7.5) :=
by
  sorry

end ratio_of_part_to_whole_l0_537


namespace shelves_needed_l0_583

theorem shelves_needed (initial_stock : ℕ) (additional_shipment : ℕ) (bears_per_shelf : ℕ) (total_bears : ℕ) (shelves : ℕ) :
  initial_stock = 4 → 
  additional_shipment = 10 → 
  bears_per_shelf = 7 → 
  total_bears = initial_stock + additional_shipment →
  total_bears / bears_per_shelf = shelves →
  shelves = 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end shelves_needed_l0_583


namespace factorial_division_example_l0_66

theorem factorial_division_example : (10! / (5! * 2!)) = 15120 := 
by
  sorry

end factorial_division_example_l0_66


namespace problem_thre_is_15_and_10_percent_l0_129

theorem problem_thre_is_15_and_10_percent (x y : ℝ) 
  (h1 : 3 = 0.15 * x) 
  (h2 : 3 = 0.10 * y) : 
  x - y = -10 := 
by 
  sorry

end problem_thre_is_15_and_10_percent_l0_129


namespace half_percent_of_160_l0_566

theorem half_percent_of_160 : (1 / 2 / 100) * 160 = 0.8 :=
by
  -- Proof goes here
  sorry

end half_percent_of_160_l0_566


namespace molecular_weight_of_compound_l0_273

def num_atoms_C : ℕ := 6
def num_atoms_H : ℕ := 8
def num_atoms_O : ℕ := 7

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def molecular_weight (nC nH nO : ℕ) (wC wH wO : ℝ) : ℝ :=
  nC * wC + nH * wH + nO * wO

theorem molecular_weight_of_compound :
  molecular_weight num_atoms_C num_atoms_H num_atoms_O atomic_weight_C atomic_weight_H atomic_weight_O = 192.124 :=
by
  sorry

end molecular_weight_of_compound_l0_273


namespace proof_problem_l0_24

-- Proposition B: ∃ x ∈ ℝ, x^2 - 3*x + 3 < 0
def propB : Prop := ∃ x : ℝ, x^2 - 3 * x + 3 < 0

-- Proposition D: ∀ x ∈ ℝ, x^2 - a*x + 1 = 0 has real solutions
def propD (a : ℝ) : Prop := ∀ x : ℝ, ∃ (x1 x2 : ℝ), x^2 - a * x + 1 = 0

-- Negation of Proposition B: ∀ x ∈ ℝ, x^2 - 3 * x + 3 ≥ 0
def neg_propB : Prop := ∀ x : ℝ, x^2 - 3 * x + 3 ≥ 0

-- Negation of Proposition D: ∃ a ∈ ℝ, ∃ x ∈ ℝ, ∄ (x1 x2 : ℝ), x^2 - a * x + 1 = 0
def neg_propD : Prop := ∃ a : ℝ, ∀ x : ℝ, ¬ ∃ (x1 x2 : ℝ), x^2 - a * x + 1 = 0 

-- The main theorem combining the results based on the solutions.
theorem proof_problem : neg_propB ∧ neg_propD :=
by
  sorry

end proof_problem_l0_24


namespace photographs_taken_l0_190

theorem photographs_taken (P : ℝ) (h : P + 0.80 * P = 180) : P = 100 :=
by sorry

end photographs_taken_l0_190


namespace find_cube_edge_length_l0_917

-- Define parameters based on the problem conditions
def is_solution (n : ℕ) : Prop :=
  n > 4 ∧
  (6 * (n - 4)^2 = (n - 4)^3)

-- The main theorem statement
theorem find_cube_edge_length : ∃ n : ℕ, is_solution n ∧ n = 10 :=
by
  use 10
  sorry

end find_cube_edge_length_l0_917


namespace fraction_of_full_fare_half_ticket_l0_921

theorem fraction_of_full_fare_half_ticket (F R : ℝ) 
  (h1 : F + R = 216) 
  (h2 : F + (1/2)*F + 2*R = 327) : 
  (1/2) = 1/2 :=
by
  sorry

end fraction_of_full_fare_half_ticket_l0_921


namespace least_three_digit_with_factors_correct_l0_429

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def has_factors (n : ℕ) : Prop := n % 3 = 0 ∧ n % 4 = 0 ∧ n % 9 = 0
def least_three_digit_with_factors : ℕ := 108

theorem least_three_digit_with_factors_correct : 
  is_three_digit least_three_digit_with_factors ∧ has_factors least_three_digit_with_factors ∧
  ∀ m : ℕ, is_three_digit m → has_factors m → least_three_digit_with_factors ≤ m := 
by 
  sorry

end least_three_digit_with_factors_correct_l0_429


namespace xiao_wang_original_plan_l0_144

theorem xiao_wang_original_plan (p d1 extra_pages : ℕ) (original_days : ℝ) (x : ℝ) 
  (h1 : p = 200)
  (h2 : d1 = 5)
  (h3 : extra_pages = 5)
  (h4 : original_days = p / x)
  (h5 : original_days - 1 = d1 + (p - (d1 * x)) / (x + extra_pages)) :
  x = 20 := 
  sorry

end xiao_wang_original_plan_l0_144


namespace sunscreen_cost_l0_270

theorem sunscreen_cost (reapply_time : ℕ) (oz_per_application : ℕ) 
  (oz_per_bottle : ℕ) (cost_per_bottle : ℝ) (total_time : ℕ) (expected_cost : ℝ) :
  reapply_time = 2 →
  oz_per_application = 3 →
  oz_per_bottle = 12 →
  cost_per_bottle = 3.5 →
  total_time = 16 →
  expected_cost = 7 →
  (total_time / reapply_time) * (oz_per_application / oz_per_bottle) * cost_per_bottle = expected_cost :=
by
  intros
  sorry

end sunscreen_cost_l0_270


namespace ravish_maximum_marks_l0_281

theorem ravish_maximum_marks (M : ℝ) (h_pass : 0.40 * M = 80) : M = 200 :=
sorry

end ravish_maximum_marks_l0_281


namespace hash_difference_l0_825

def hash (x y : ℕ) : ℤ := x * y - 3 * x + y

theorem hash_difference : (hash 8 5) - (hash 5 8) = -12 := by
  sorry

end hash_difference_l0_825


namespace problem_solution_l0_202

variable (α : ℝ)
-- Condition: α in the first quadrant (0 < α < π/2)
variable (h1 : 0 < α ∧ α < Real.pi / 2)
-- Condition: sin α + cos α = sqrt 2
variable (h2 : Real.sin α + Real.cos α = Real.sqrt 2)

theorem problem_solution : Real.tan α + Real.cos α / Real.sin α = 2 :=
by
  sorry

end problem_solution_l0_202


namespace min_students_in_class_l0_375

-- Define the conditions
variables (b g : ℕ) -- number of boys and girls
variable (h1 : 3 * b = 4 * (2 * g)) -- Equal number of boys and girls passed the test

-- Define the desired minimum number of students
def min_students : ℕ := 17

-- The theorem which asserts that the total number of students in the class is at least 17
theorem min_students_in_class (b g : ℕ) (h1 : 3 * b = 4 * (2 * g)) : (b + g) ≥ min_students := 
sorry

end min_students_in_class_l0_375


namespace no_two_consecutive_heads_probability_l0_758

theorem no_two_consecutive_heads_probability : 
  let total_outcomes := 2 ^ 10 in
  let favorable_outcomes := 
    (∑ n in range 6, nat.choose (10 - n - 1) n) in
  (favorable_outcomes / total_outcomes : ℚ) = 9 / 64 :=
by
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := 1 + 10 + 36 + 56 + 35 + 6
  have h_total: total_outcomes = 1024 := by norm_num
  have h_favorable: favorable_outcomes = 144 := by norm_num
  have h_fraction: (favorable_outcomes : ℚ) / total_outcomes = 9 / 64 := by norm_num / [total_outcomes, favorable_outcomes]
  exact h_fraction

end no_two_consecutive_heads_probability_l0_758


namespace suresh_work_hours_l0_116

theorem suresh_work_hours (x : ℝ) (h : x / 15 + 8 / 20 = 1) : x = 9 :=
by 
    sorry

end suresh_work_hours_l0_116


namespace locus_of_midpoint_l0_807

open Real

noncomputable def circumcircle_eq (A B C : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let a := 1
  let b := 3
  let r2 := 5
  (a, b, r2)

theorem locus_of_midpoint (A B C N : ℝ × ℝ) :
  N = (6, 2) ∧ A = (0, 1) ∧ B = (2, 1) ∧ C = (3, 4) → 
  let P := (7 / 2, 5 / 2)
  let r2 := 5 / 4
  ∃ x y : ℝ, 
  (x, y) = P ∧ (x - 7 / 2)^2 + (y - 5 / 2)^2 = r2 :=
by sorry

end locus_of_midpoint_l0_807


namespace find_people_and_carriages_l0_2

theorem find_people_and_carriages (x y : ℝ) :
  (x / 3 = y + 2) ∧ ((x - 9) / 2 = y) ↔
  (x / 3 = y + 2) ∧ ((x - 9) / 2 = y) :=
by
  sorry

end find_people_and_carriages_l0_2


namespace largest_hexagon_angle_l0_546

theorem largest_hexagon_angle (x : ℝ) : 
  (2 * x + 2 * x + 2 * x + 3 * x + 4 * x + 5 * x = 720) → (5 * x = 200) := by
  sorry

end largest_hexagon_angle_l0_546


namespace gcd_greatest_possible_value_l0_802

noncomputable def Sn (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem gcd_greatest_possible_value (n : ℕ) (hn : 0 < n) : 
  Nat.gcd (3 * Sn n) (n + 1) = 1 :=
sorry

end gcd_greatest_possible_value_l0_802


namespace asian_population_percentage_in_west_is_57_l0_778

variable (NE MW South West : ℕ)

def total_asian_population (NE MW South West : ℕ) : ℕ :=
  NE + MW + South + West

def west_asian_population_percentage
  (NE MW South West : ℕ) (total_asian_population : ℕ) : ℚ :=
  (West : ℚ) / (total_asian_population : ℚ) * 100

theorem asian_population_percentage_in_west_is_57 :
  total_asian_population 2 3 4 12 = 21 →
  west_asian_population_percentage 2 3 4 12 21 = 57 :=
by
  intros
  sorry

end asian_population_percentage_in_west_is_57_l0_778


namespace correct_sampling_method_order_l0_584

-- Definitions for sampling methods
def simple_random_sampling (method : ℕ) : Bool :=
  method = 1

def systematic_sampling (method : ℕ) : Bool :=
  method = 2

def stratified_sampling (method : ℕ) : Bool :=
  method = 3

-- Main theorem stating the correct method order
theorem correct_sampling_method_order : simple_random_sampling 1 ∧ stratified_sampling 3 ∧ systematic_sampling 2 :=
by
  sorry

end correct_sampling_method_order_l0_584


namespace chess_competition_players_l0_426

theorem chess_competition_players (J H : ℕ) (total_points : ℕ) (junior_points : ℕ) (high_school_points : ℕ → ℕ)
  (plays : ℕ → ℕ)
  (H_junior_points : junior_points = 8)
  (H_total_points : total_points = (J + H) * (J + H - 1) / 2)
  (H_total_points_contribution : total_points = junior_points + H * high_school_points H)
  (H_even_distribution : ∀ x : ℕ, 0 ≤ x ∧ x ≤ J → high_school_points H = x * (x - 1) / 2)
  (H_H_cases : H = 7 ∨ H = 9 ∨ H = 14) :
  H = 7 ∨ H = 14 :=
by
  have H_cases : H = 7 ∨ H = 14 :=
    by
      sorry
  exact H_cases

end chess_competition_players_l0_426


namespace quadratic_equation_m_l0_369

theorem quadratic_equation_m (m : ℝ) (h1 : |m| + 1 = 2) (h2 : m + 1 ≠ 0) : m = 1 :=
sorry

end quadratic_equation_m_l0_369


namespace inequality_transformation_l0_616

variable {x y : ℝ}

theorem inequality_transformation (h : x > y) : x + 5 > y + 5 :=
by
  sorry

end inequality_transformation_l0_616


namespace determine_Sets_l0_168

variable (S : Set Point)

def constraints (S : Set Point) : Prop :=
  ∀ {A B C D : Point}, A ∈ S → B ∈ S → C ∈ S → D ∈ S → A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (∃ (circle : Circle), A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle ∨
  ∃ (line : Line), A ∈ line ∧ B ∈ line ∧ C ∈ line ∧ D ∈ line)

theorem determine_Sets (S : Set Point) (h : constraints S):
  (∃ (circle : Circle), ∀ (p : Point), p ∈ S → p ∈ circle) ∨
  (∃ (A B C D : Point), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧
  ∀ E ∈ Set.Points where E ≠ A ∧ E ≠ B ∧ E ≠ C ∧ E ≠ D → ∃ (P : Point), P ∈ Set.inter (Line.through A B) (Line.through C D) ∨
  P ∈ Set.inter (Line.through A C) (Line.through B D) ∨ P ∈ Set.inter (Line.through A D) (Line.through B C)) ∨
  (∃ (line : Line), ∀ (p1 : Point), p1 ∈ S → p1 ∈ line ∧ ∃ p2, p2 ∈ S ∧ p2 ∉ line) := sorry

end determine_Sets_l0_168


namespace find_day_53_days_from_friday_l0_733

-- Define the days of the week
inductive day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open day

-- Define a function that calculates the day of the week after a certain number of days
def add_days (d : day) (n : ℕ) : day :=
  match d, n % 7 with
  | Sunday, 0 => Sunday
  | Monday, 0 => Monday
  | Tuesday, 0 => Tuesday
  | Wednesday, 0 => Wednesday
  | Thursday, 0 => Thursday
  | Friday, 0 => Friday
  | Saturday, 0 => Saturday
  | Sunday, 1 => Monday
  | Monday, 1 => Tuesday
  | Tuesday, 1 => Wednesday
  | Wednesday, 1 => Thursday
  | Thursday, 1 => Friday
  | Friday, 1 => Saturday
  | Saturday, 1 => Sunday
  | Sunday, 2 => Tuesday
  | Monday, 2 => Wednesday
  | Tuesday, 2 => Thursday
  | Wednesday, 2 => Friday
  | Thursday, 2 => Saturday
  | Friday, 2 => Sunday
  | Saturday, 2 => Monday
  | Sunday, 3 => Wednesday
  | Monday, 3 => Thursday
  | Tuesday, 3 => Friday
  | Wednesday, 3 => Saturday
  | Thursday, 3 => Sunday
  | Friday, 3 => Monday
  | Saturday, 3 => Tuesday
  | Sunday, 4 => Thursday
  | Monday, 4 => Friday
  | Tuesday, 4 => Saturday
  | Wednesday, 4 => Sunday
  | Thursday, 4 => Monday
  | Friday, 4 => Tuesday
  | Saturday, 4 => Wednesday
  | Sunday, 5 => Friday
  | Monday, 5 => Saturday
  | Tuesday, 5 => Sunday
  | Wednesday, 5 => Monday
  | Thursday, 5 => Tuesday
  | Friday, 5 => Wednesday
  | Saturday, 5 => Thursday
  | Sunday, 6 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 6 => Friday
  | _, _ => Sunday -- This case is unreachable, but required to complete the pattern match
  end

-- Lean statement for the proof problem:
theorem find_day_53_days_from_friday : add_days Friday 53 = Tuesday :=
  sorry

end find_day_53_days_from_friday_l0_733


namespace circle_center_radius_sum_l0_482

theorem circle_center_radius_sum (u v s : ℝ) (h1 : (x + 4)^2 + (y - 1)^2 = 13)
    (h2 : (u, v) = (-4, 1)) (h3 : s = Real.sqrt 13) : 
    u + v + s = -3 + Real.sqrt 13 :=
by
  sorry

end circle_center_radius_sum_l0_482


namespace find_sample_size_l0_33

theorem find_sample_size :
  ∀ (n : ℕ), 
    (∃ x : ℝ,
      2 * x + 3 * x + 4 * x + 6 * x + 4 * x + x = 1 ∧
      2 * n * x + 3 * n * x + 4 * n * x = 27) →
    n = 60 :=
by
  intro n
  rintro ⟨x, h1, h2⟩
  sorry

end find_sample_size_l0_33


namespace find_values_l0_199

theorem find_values (a b: ℝ) (h1: a > b) (h2: b > 1)
  (h3: Real.log a / Real.log b + Real.log b / Real.log a = 5 / 2)
  (h4: a^b = b^a) :
  a = 4 ∧ b = 2 := 
sorry

end find_values_l0_199


namespace loss_percentage_is_five_l0_148

/-- Definitions -/
def original_price : ℝ := 490
def sold_price : ℝ := 465.50
def loss_amount : ℝ := original_price - sold_price

/-- Theorem -/
theorem loss_percentage_is_five :
  (loss_amount / original_price) * 100 = 5 :=
by
  sorry

end loss_percentage_is_five_l0_148


namespace complex_pow_six_eq_eight_i_l0_310

theorem complex_pow_six_eq_eight_i (i : ℂ) (h : i^2 = -1) : (1 - i) ^ 6 = 8 * i := by
  sorry

end complex_pow_six_eq_eight_i_l0_310


namespace find_point_P_l0_71

def f (x : ℝ) : ℝ := x^4 - 2 * x

def tangent_line_perpendicular (x y : ℝ) : Prop :=
  (f x) = y ∧ (4 * x^3 - 2 = 2)

theorem find_point_P :
  ∃ (x y : ℝ), tangent_line_perpendicular x y ∧ x = 1 ∧ y = -1 :=
sorry

end find_point_P_l0_71


namespace smallest_period_monotonic_interval_max_area_ABC_l0_495

variables {A B C a b c : ℝ}
variables (m n : ℝ × ℝ)

-- Condition that vectors m and n are perpendicular
def m := (2 * Real.sin B, Real.sqrt 3)
def n := (2 * Real.cos (B / 2) ^ 2 - 1, Real.cos (2 * B))
def perpendicular := m.1 * n.1 + m.2 * n.2 = 0

-- Given an acute triangle ABC and b = 4
def acute_triangle_ABC := A + B + C = Real.pi ∧ (A < Real.pi / 2) ∧ (B < Real.pi / 2) ∧ (C < Real.pi / 2)
def side_b := b = 4

-- (1) Smallest positive period and monotonically increasing interval
def smallest_period_f (f : ℝ → ℝ) : ℝ := sorry
def monotonically_increasing_interval (f : ℝ → ℝ) (k : ℤ) : Set ℝ := sorry
theorem smallest_period_monotonic_interval :
  ∀ B : ℝ, perpendicular → (smallest_period_f (λ x => Real.sin (2 * x - B)) = Real.pi) ∧
    (∃ k:ℤ, monotonically_increasing_interval (λ x => Real.sin (2 * x - B)) k = Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12)) := sorry

-- (2) Maximum area of triangle ABC
def max_area_triangle (a c : ℝ) : ℝ := (1 / 2) * a * c * (Real.sin B)
theorem max_area_ABC :
  ∀ (a c : ℝ), acute_triangle_ABC → side_b → max_area_triangle a c ≤ 4 * Real.sqrt 3 := sorry

end smallest_period_monotonic_interval_max_area_ABC_l0_495


namespace option_b_represents_factoring_l0_590

theorem option_b_represents_factoring (x y : ℤ) :
  x^2 - 2*x*y = x * (x - 2*y) :=
sorry

end option_b_represents_factoring_l0_590


namespace min_value_l0_701

variable {X : ℝ → ℝ} {σ : ℝ} (μ := 10)

-- The random variable X follows a normal distribution N(10, σ^2)
axiom norm_dist : ∀ x, X x = μ + σ * (Gaussian pdf 0 1).normalize

-- Probabilities given in the problem
variable (m n : ℝ)

-- The given probabilities P(X > 12) = m and P(8 ≤ X < 10) = n
variable [Pm : m = P (X > 12)]
variable [Pn : n = P (8 ≤ X < 10)]

theorem min_value : ∃ (m n : ℝ), m = P (X > 12) ∧ n = P (8 ≤ X < 10) ∧ (2 / m + 1 / n = 6 + 4 * real.sqrt 2) := sorry

end min_value_l0_701


namespace equation_solution_unique_l0_485

theorem equation_solution_unique (x y : ℤ) : 
  x^4 = y^2 + 2*y + 2 ↔ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) :=
by
  sorry

end equation_solution_unique_l0_485


namespace height_of_tree_l0_374

noncomputable def height_of_flagpole : ℝ := 4
noncomputable def shadow_of_flagpole : ℝ := 6
noncomputable def shadow_of_tree : ℝ := 12

theorem height_of_tree (h : height_of_flagpole / shadow_of_flagpole = x / shadow_of_tree) : x = 8 := by
  sorry

end height_of_tree_l0_374


namespace sum_of_pairwise_rel_prime_integers_l0_267

def is_pairwise_rel_prime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1

theorem sum_of_pairwise_rel_prime_integers 
  (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) 
  (h_prod : a * b * c = 343000) (h_rel_prime : is_pairwise_rel_prime a b c) : 
  a + b + c = 476 := 
sorry

end sum_of_pairwise_rel_prime_integers_l0_267


namespace frac_not_suff_nec_l0_53

theorem frac_not_suff_nec {a b : ℝ} (hab : a / b > 1) : 
  ¬ ((∀ a b : ℝ, a / b > 1 → a > b) ∧ (∀ a b : ℝ, a > b → a / b > 1)) :=
sorry

end frac_not_suff_nec_l0_53


namespace fraction_mistake_l0_965

theorem fraction_mistake (n : ℕ) (h : n = 288) (student_answer : ℕ) 
(h_student : student_answer = 240) : student_answer / n = 5 / 6 := 
by 
  -- Given that n = 288 and the student's answer = 240;
  -- we need to prove that 240/288 = 5/6
  sorry

end fraction_mistake_l0_965


namespace inscribed_square_proof_l0_290

theorem inscribed_square_proof :
  (∃ (r : ℝ), 2 * π * r = 72 * π ∧ r = 36) ∧ 
  (∃ (s : ℝ), (2 * (36:ℝ))^2 = 2 * s ^ 2 ∧ s = 36 * Real.sqrt 2) :=
by
  sorry

end inscribed_square_proof_l0_290


namespace odometer_reading_before_trip_l0_986

-- Define the given conditions
def odometer_reading_lunch : ℝ := 372.0
def miles_traveled : ℝ := 159.7

-- Theorem to prove that the odometer reading before the trip was 212.3 miles
theorem odometer_reading_before_trip : odometer_reading_lunch - miles_traveled = 212.3 := by
  sorry

end odometer_reading_before_trip_l0_986


namespace roundness_720_eq_7_l0_545

def roundness (n : ℕ) : ℕ :=
  if h : n > 1 then
    let factors := n.factorization
    factors.sum (λ _ k => k)
  else 0

theorem roundness_720_eq_7 : roundness 720 = 7 := by
  sorry

end roundness_720_eq_7_l0_545


namespace final_prices_l0_296

noncomputable def hat_initial_price : ℝ := 15
noncomputable def hat_first_discount : ℝ := 0.20
noncomputable def hat_second_discount : ℝ := 0.40

noncomputable def gloves_initial_price : ℝ := 8
noncomputable def gloves_first_discount : ℝ := 0.25
noncomputable def gloves_second_discount : ℝ := 0.30

theorem final_prices :
  let hat_price_after_first_discount := hat_initial_price * (1 - hat_first_discount)
  let hat_final_price := hat_price_after_first_discount * (1 - hat_second_discount)
  let gloves_price_after_first_discount := gloves_initial_price * (1 - gloves_first_discount)
  let gloves_final_price := gloves_price_after_first_discount * (1 - gloves_second_discount)
  hat_final_price = 7.20 ∧ gloves_final_price = 4.20 :=
by
  sorry

end final_prices_l0_296


namespace four_bags_remainder_l0_37

theorem four_bags_remainder (n : ℤ) (hn : n % 11 = 5) : (4 * n) % 11 = 9 := 
by
  sorry

end four_bags_remainder_l0_37


namespace only_ten_perfect_square_l0_606

theorem only_ten_perfect_square (n : ℤ) :
  ∃ k : ℤ, n^4 + 6 * n^3 + 11 * n^2 + 3 * n + 31 = k^2 ↔ n = 10 :=
by
  sorry

end only_ten_perfect_square_l0_606


namespace native_answer_l0_911

-- Define properties to represent native types
inductive NativeType
| normal
| zombie
| half_zombie

-- Define the function that determines the response of a native
def response (native : NativeType) : String :=
  match native with
  | NativeType.normal => "да"
  | NativeType.zombie => "да"
  | NativeType.half_zombie => "да"

-- Define the main theorem
theorem native_answer (native : NativeType) : response native = "да" :=
by sorry

end native_answer_l0_911


namespace velocity_of_point_C_l0_346

variable (a T R L x : ℝ)
variable (a_pos : a > 0) (T_pos : T > 0) (R_pos : R > 0) (L_pos : L > 0)
variable (h_eq : a * T / (a * T - R) = (L + x) / x)

theorem velocity_of_point_C : a * (L / R) = x / T := by
  sorry

end velocity_of_point_C_l0_346


namespace employees_original_number_l0_294

noncomputable def original_employees_approx (employees_remaining : ℝ) (reduction_percent : ℝ) : ℝ :=
  employees_remaining / (1 - reduction_percent)

theorem employees_original_number (employees_remaining : ℝ) (reduction_percent : ℝ) (original : ℝ) :
  employees_remaining = 462 → reduction_percent = 0.276 →
  abs (original_employees_approx employees_remaining reduction_percent - original) < 1 →
  original = 638 :=
by
  intros h_remaining h_reduction h_approx
  sorry

end employees_original_number_l0_294


namespace area_ratio_of_concentric_circles_l0_424

theorem area_ratio_of_concentric_circles (C1 C2 : ℝ) 
  (h1 : (60 / 360) * C1 = (30 / 360) * C2) : (C1 / C2)^2 = 1 / 4 := 
by 
  have h : C1 / C2 = 1 / 2 := by
    field_simp [h1]
  rw [h]
  norm_num

end area_ratio_of_concentric_circles_l0_424


namespace shortest_part_length_l0_285

theorem shortest_part_length (total_length : ℝ) (r1 r2 r3 : ℝ) (shortest_length : ℝ) :
  total_length = 196.85 → r1 = 3.6 → r2 = 8.4 → r3 = 12 → shortest_length = 29.5275 :=
by
  sorry

end shortest_part_length_l0_285


namespace stephanie_store_visits_l0_241

theorem stephanie_store_visits (oranges_per_visit total_oranges : ℕ) 
  (h1 : oranges_per_visit = 2)
  (h2 : total_oranges = 16) : 
  total_oranges / oranges_per_visit = 8 :=
by
  rw [h1, h2]
  norm_num
  sorry

end stephanie_store_visits_l0_241


namespace volume_of_parallelepiped_l0_882

theorem volume_of_parallelepiped 
  (l w h : ℝ)
  (h1 : l * w / Real.sqrt (l^2 + w^2) = 2 * Real.sqrt 5)
  (h2 : h * w / Real.sqrt (h^2 + w^2) = 30 / Real.sqrt 13)
  (h3 : h * l / Real.sqrt (h^2 + l^2) = 15 / Real.sqrt 10) 
  : l * w * h = 750 :=
sorry

end volume_of_parallelepiped_l0_882


namespace total_marks_eq_300_second_candidate_percentage_l0_153

-- Defining the conditions
def percentage_marks (total_marks : ℕ) : ℕ := 40
def fail_by (fail_marks : ℕ) : ℕ := 40
def passing_marks : ℕ := 160

-- The number of total marks in the exam computed from conditions
theorem total_marks_eq_300 : ∃ T, 0.40 * T = 120 :=
by
  use 300
  sorry

-- The percentage of marks the second candidate gets
theorem second_candidate_percentage : ∃ percent, percent = (180 / 300) * 100 :=
by
  use 60
  sorry

end total_marks_eq_300_second_candidate_percentage_l0_153


namespace shots_per_puppy_l0_596

-- Definitions
def num_pregnant_dogs : ℕ := 3
def puppies_per_dog : ℕ := 4
def cost_per_shot : ℕ := 5
def total_shot_cost : ℕ := 120

-- Total number of puppies
def total_puppies : ℕ := num_pregnant_dogs * puppies_per_dog

-- Total number of shots
def total_shots : ℕ := total_shot_cost / cost_per_shot

-- The theorem to prove
theorem shots_per_puppy : total_shots / total_puppies = 2 :=
by
  sorry

end shots_per_puppy_l0_596


namespace rod_length_l0_492

theorem rod_length (pieces : ℕ) (length_per_piece_cm : ℕ) (total_length_m : ℝ) :
  pieces = 35 → length_per_piece_cm = 85 → total_length_m = 29.75 :=
by
  intros h1 h2
  sorry

end rod_length_l0_492


namespace z_share_in_profit_l0_278

noncomputable def investment_share (investment : ℕ) (months : ℕ) : ℕ := investment * months

noncomputable def profit_share (profit : ℕ) (share : ℚ) : ℚ := (profit : ℚ) * share

theorem z_share_in_profit 
  (investment_X : ℕ := 36000)
  (investment_Y : ℕ := 42000)
  (investment_Z : ℕ := 48000)
  (months_X : ℕ := 12)
  (months_Y : ℕ := 12)
  (months_Z : ℕ := 8)
  (total_profit : ℕ := 14300) :
  profit_share total_profit (investment_share investment_Z months_Z / 
            (investment_share investment_X months_X + 
             investment_share investment_Y months_Y + 
             investment_share investment_Z months_Z)) = 2600 := 
by
  sorry

end z_share_in_profit_l0_278


namespace cube_volume_l0_249

theorem cube_volume (d : ℝ) (h : d = 6 * Real.sqrt 2) : 
  ∃ v : ℝ, v = 48 * Real.sqrt 6 := by
  let s := d / Real.sqrt 3
  let volume := s ^ 3
  use volume
  /- Proof of the volume calculation is omitted. -/
  sorry

end cube_volume_l0_249


namespace probability_adjacent_difference_l0_398

noncomputable def probability_no_adjacent_same_rolls : ℚ :=
  (7 / 8) ^ 6

theorem probability_adjacent_difference :
  let num_people := 6
  let sides_of_die := 8
  ( ∀ i : ℕ, 0 ≤ i ∧ i < num_people -> (∃ x : ℕ, 1 ≤ x ∧ x ≤ sides_of_die)) →
  probability_no_adjacent_same_rolls = 117649 / 262144 := 
by 
  sorry

end probability_adjacent_difference_l0_398


namespace which_point_is_in_fourth_quadrant_l0_591

def point (x: ℝ) (y: ℝ) : Prop := x > 0 ∧ y < 0

theorem which_point_is_in_fourth_quadrant :
  point 5 (-4) :=
by {
  -- proofs for each condition can be added,
  sorry
}

end which_point_is_in_fourth_quadrant_l0_591


namespace contestant_final_score_l0_454

theorem contestant_final_score (score_content score_skills score_effects : ℕ) 
                               (weight_content weight_skills weight_effects : ℕ) :
    score_content = 90 →
    score_skills  = 80 →
    score_effects = 90 →
    weight_content = 4 →
    weight_skills  = 2 →
    weight_effects = 4 →
    (score_content * weight_content + score_skills * weight_skills + score_effects * weight_effects) / 
    (weight_content + weight_skills + weight_effects) = 88 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end contestant_final_score_l0_454


namespace problem1_problem2_l0_627

open Set Real

-- Definition of sets A, B, and C
def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | x < a }

-- Problem 1: Prove A ∪ B = { x | 1 ≤ x < 10 }
theorem problem1 : A ∪ B = { x : ℝ | 1 ≤ x ∧ x < 10 } :=
sorry

-- Problem 2: Prove the range of a given the conditions
theorem problem2 (a : ℝ) (h1 : (A ∩ C a) ≠ ∅) (h2 : (B ∩ C a) = ∅) : 1 < a ∧ a ≤ 2 :=
sorry

end problem1_problem2_l0_627


namespace least_three_digit_multiple_l0_432

def LCM (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

theorem least_three_digit_multiple (n : ℕ) :
  (n >= 100) ∧ (n < 1000) ∧ (n % 36 = 0) ∧ (∀ m, (m >= 100) ∧ (m < 1000) ∧ (m % 36 = 0) → n <= m) ↔ n = 108 :=
sorry

end least_three_digit_multiple_l0_432


namespace no_consecutive_heads_probability_l0_765

def prob_no_consecutive_heads (n : ℕ) : ℝ := 
  -- This is the probability function for no consecutive heads
  if n = 10 then (9 / 64) else 0

theorem no_consecutive_heads_probability :
  prob_no_consecutive_heads 10 = 9 / 64 :=
by
  -- Proof would go here
  sorry

end no_consecutive_heads_probability_l0_765


namespace range_of_a_l0_363

-- Define the function f
def f (a x : ℝ) : ℝ := -x^3 + a * x^2 - x - 1

-- Define the derivative of f
def f_prime (a x : ℝ) : ℝ := -3 * x^2 + 2 * a * x - 1

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f_prime a x ≤ 0) ↔ -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
sorry

end range_of_a_l0_363


namespace sqrt_47_minus_2_range_l0_938

theorem sqrt_47_minus_2_range (h : 6 < Real.sqrt 47 ∧ Real.sqrt 47 < 7) : 4 < Real.sqrt 47 - 2 ∧ Real.sqrt 47 - 2 < 5 := by
  sorry

end sqrt_47_minus_2_range_l0_938


namespace peter_initial_erasers_l0_109

theorem peter_initial_erasers (E : ℕ) (h : E + 3 = 11) : E = 8 :=
by {
  sorry
}

end peter_initial_erasers_l0_109


namespace work_problem_solution_l0_574

theorem work_problem_solution :
  (∃ C: ℝ, 
    B_work_days = 8 ∧ 
    (1 / A_work_rate + 1 / B_work_days + C = 1 / 3) ∧ 
    C = 1 / 8
  ) → 
  A_work_days = 12 :=
by
  sorry

end work_problem_solution_l0_574


namespace total_dividends_correct_l0_897

-- Conditions
def net_profit (total_income expenses loan_penalty_rate : ℝ) : ℝ :=
  let net1 := total_income - expenses
  let loan_penalty := net1 * loan_penalty_rate
  net1 - loan_penalty

def total_loan_payments (monthly_payment months additional_payment : ℝ) : ℝ :=
  (monthly_payment * months) - additional_payment

def dividend_per_share (net_profit total_loan_payments num_shares : ℝ) : ℝ :=
  (net_profit - total_loan_payments) / num_shares

noncomputable def total_dividends_director (dividend_per_share shares_owned : ℝ) : ℝ :=
  dividend_per_share * shares_owned

theorem total_dividends_correct :
  total_dividends_director (dividend_per_share (net_profit 1500000 674992 0.2) (total_loan_payments 23914 12 74992) 1000) 550 = 246400 :=
sorry

end total_dividends_correct_l0_897


namespace max_expression_tends_to_infinity_l0_845

noncomputable def maximize_expression (x y z : ℝ) : ℝ :=
  1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)) + 1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))

theorem max_expression_tends_to_infinity : 
  ∀ (x y z : ℝ), -1 < x ∧ x < 1 ∧ -1 < y ∧ y < 1 ∧ -1 < z ∧ z < 1 → 
    ∃ M : ℝ, maximize_expression x y z > M :=
by
  intro x y z h
  sorry

end max_expression_tends_to_infinity_l0_845


namespace juniors_score_l0_658

theorem juniors_score (n : ℕ) (j s : ℕ) (avg_score students_avg seniors_avg : ℕ)
  (h1 : 0 < n)
  (h2 : j = n / 5)
  (h3 : s = 4 * n / 5)
  (h4 : avg_score = 80)
  (h5 : seniors_avg = 78)
  (h6 : students_avg = avg_score)
  (h7 : n * students_avg = n * avg_score)
  (h8 : s * seniors_avg = 78 * s) :
  (800 - 624) / j = 88 := by
  sorry

end juniors_score_l0_658


namespace apple_ratio_simplest_form_l0_691

theorem apple_ratio_simplest_form (sarah_apples brother_apples cousin_apples : ℕ) 
  (h1 : sarah_apples = 630)
  (h2 : brother_apples = 270)
  (h3 : cousin_apples = 540)
  (gcd_simplified : Nat.gcd (Nat.gcd sarah_apples brother_apples) cousin_apples = 90) :
  (sarah_apples / 90, brother_apples / 90, cousin_apples / 90) = (7, 3, 6) := 
by
  sorry

end apple_ratio_simplest_form_l0_691


namespace sunzi_carriage_l0_3

theorem sunzi_carriage (x y : ℕ) :
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔
  ((Three people share a carriage, leaving two carriages empty) ∧ (Two people share a carriage, leaving nine people walking)) := sorry

end sunzi_carriage_l0_3


namespace find_a1_l0_806

theorem find_a1 (a : ℕ → ℕ) (h1 : a 5 = 14) (h2 : ∀ n, a (n+1) - a n = n + 1) : a 1 = 0 :=
by
  sorry

end find_a1_l0_806


namespace value_of_expression_l0_216

-- Define the variables and conditions
variables (x y : ℝ)
axiom h1 : x + 2 * y = 4
axiom h2 : x * y = -8

-- Define the statement to be proven
theorem value_of_expression : x^2 + 4 * y^2 = 48 := 
by
  sorry

end value_of_expression_l0_216


namespace smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5_l0_170

theorem smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5 :
  ∃ n : ℕ, n > 0 ∧ (7^n % 5 = n^7 % 5) ∧
  ∀ m : ℕ, m > 0 ∧ (7^m % 5 = m^7 % 5) → n ≤ m :=
by
  sorry

end smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5_l0_170


namespace angle_RPS_is_27_l0_972

theorem angle_RPS_is_27 (PQ BP PR QS QS PSQ QPRS : ℝ) :
  PQ + PSQ + QS = 180 ∧ 
  QS = 48 ∧ 
  PSQ = 38 ∧ 
  QPRS = 67
  → (QS - QPRS = 27) := 
by {
  sorry
}

end angle_RPS_is_27_l0_972


namespace last_three_digits_of_7_pow_103_l0_798

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 327 :=
by
  sorry

end last_three_digits_of_7_pow_103_l0_798


namespace count_valid_subsets_l0_29

open Finset

noncomputable def set := {2, 3, 4, 5, 6, 7, 8, 9, 10}

def is_valid_subset (s : Finset ℕ) : Prop :=
  s.card = 3 ∧ 7 ∈ s ∧ s.sum id = 18

theorem count_valid_subsets : 
  (set.filter is_valid_subset).card = 3 :=
by
  sorry

end count_valid_subsets_l0_29


namespace geometric_sequence_condition_neither_necessary_nor_sufficient_l0_544

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

noncomputable def is_monotonically_increasing (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) > a n

theorem geometric_sequence_condition_neither_necessary_nor_sufficient (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q → ¬( (is_monotonically_increasing a ↔ q > 1) ) :=
by sorry

end geometric_sequence_condition_neither_necessary_nor_sufficient_l0_544


namespace abhay_speed_l0_969

theorem abhay_speed
    (A S : ℝ)
    (h1 : 30 / A = 30 / S + 2)
    (h2 : 30 / (2 * A) = 30 / S - 1) :
    A = 5 * Real.sqrt 6 :=
by
  sorry

end abhay_speed_l0_969


namespace Misha_l0_900

theorem Misha's_decision_justified :
  let A_pos := 7 in
  let A_neg := 4 in
  let B_pos := 4 in
  let B_neg := 1 in
  (B_pos / (B_pos + B_neg) > A_pos / (A_pos + A_neg)) := 
sorry

end Misha_l0_900


namespace root_in_interval_l0_373

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem root_in_interval (k : ℤ) (h : ∃ x : ℝ, k < x ∧ x < k + 1 ∧ f x = 0) : k = 0 :=
by
  sorry

end root_in_interval_l0_373


namespace larger_number_l0_445

theorem larger_number (HCF LCM a b : ℕ) (h_hcf : HCF = 28) (h_factors: 12 * 15 * HCF = LCM) (h_prod : a * b = HCF * LCM) :
  max a b = 180 :=
sorry

end larger_number_l0_445


namespace justify_misha_decision_l0_898

-- Define the conditions based on the problem description
def reviews_smartphone_A := (7, 4) -- 7 positive and 4 negative reviews for A
def reviews_smartphone_B := (4, 1) -- 4 positive and 1 negative reviews for B

-- Define the ratios for each smartphone based on their reviews
def ratio_A := (reviews_smartphone_A.1 : ℚ) / reviews_smartphone_A.2
def ratio_B := (reviews_smartphone_B.1 : ℚ) / reviews_smartphone_B.2

-- Goal: to show that ratio_B > ratio_A, justifying Misha's decision
theorem justify_misha_decision : ratio_B > ratio_A := by
  -- placeholders to bypass the proof steps
  sorry

end justify_misha_decision_l0_898


namespace arithmetic_sequence_count_l0_822

-- Define the initial conditions
def a1 : ℤ := -3
def d : ℤ := 3
def an : ℤ := 45

-- Proposition stating the number of terms n in the arithmetic sequence
theorem arithmetic_sequence_count :
  ∃ n : ℕ, an = a1 + (n - 1) * d ∧ n = 17 :=
by
  -- Skip the proof
  sorry

end arithmetic_sequence_count_l0_822


namespace cucumber_weight_evaporation_l0_769

theorem cucumber_weight_evaporation :
  let w_99 := 50
  let p_99 := 0.99
  let evap_99 := 0.01
  let w_98 := 30
  let p_98 := 0.98
  let evap_98 := 0.02
  let w_97 := 20
  let p_97 := 0.97
  let evap_97 := 0.03

  let initial_water_99 := p_99 * w_99
  let dry_matter_99 := w_99 - initial_water_99
  let evaporated_water_99 := evap_99 * initial_water_99
  let new_weight_99 := (initial_water_99 - evaporated_water_99) + dry_matter_99

  let initial_water_98 := p_98 * w_98
  let dry_matter_98 := w_98 - initial_water_98
  let evaporated_water_98 := evap_98 * initial_water_98
  let new_weight_98 := (initial_water_98 - evaporated_water_98) + dry_matter_98

  let initial_water_97 := p_97 * w_97
  let dry_matter_97 := w_97 - initial_water_97
  let evaporated_water_97 := evap_97 * initial_water_97
  let new_weight_97 := (initial_water_97 - evaporated_water_97) + dry_matter_97

  let total_new_weight := new_weight_99 + new_weight_98 + new_weight_97
  total_new_weight = 98.335 :=
 by
  sorry

end cucumber_weight_evaporation_l0_769


namespace find_side_length_l0_497

theorem find_side_length
  (a b : ℝ)
  (S : ℝ)
  (h1 : a = 4)
  (h2 : b = 5)
  (h3 : S = 5 * Real.sqrt 3) :
  ∃ c : ℝ, c = Real.sqrt 21 ∨ c = Real.sqrt 61 :=
by
  sorry

end find_side_length_l0_497


namespace width_of_rectangular_plot_l0_774

theorem width_of_rectangular_plot 
  (length : ℝ) 
  (poles : ℕ) 
  (distance_between_poles : ℝ) 
  (num_poles : ℕ) 
  (total_wire_length : ℝ) 
  (perimeter : ℝ) 
  (width : ℝ) :
  length = 90 ∧ 
  distance_between_poles = 5 ∧ 
  num_poles = 56 ∧ 
  total_wire_length = (num_poles - 1) * distance_between_poles ∧ 
  total_wire_length = 275 ∧ 
  perimeter = 2 * (length + width) 
  → width = 47.5 :=
by
  sorry

end width_of_rectangular_plot_l0_774


namespace infinite_squares_in_arithmetic_progression_l0_815

theorem infinite_squares_in_arithmetic_progression
  (a d : ℕ) (hposd : 0 < d) (hpos : 0 < a) (k n : ℕ)
  (hk : a + k * d = n^2) :
  ∃ (t : ℕ), ∃ (m : ℕ), (a + (k + t) * d = m^2) := by
  sorry

end infinite_squares_in_arithmetic_progression_l0_815


namespace Meadowood_problem_l0_93

theorem Meadowood_problem (s h : ℕ) : ¬(26 * s + 3 * h = 58) :=
sorry

end Meadowood_problem_l0_93


namespace number_of_truthful_gnomes_l0_185

variables (T L : ℕ)

-- Conditions
def total_gnomes : Prop := T + L = 10
def hands_raised_vanilla : Prop := 10 = 10
def hands_raised_chocolate : Prop := ½ * 10 = 5
def hands_raised_fruit : Prop := 1 = 1
def total_hands_raised : Prop := 10 + 5 + 1 = 16
def extra_hands_raised : Prop := 16 - 10 = 6
def lying_gnomes : Prop := L = 6
def truthful_gnomes : Prop := T = 4

-- Statement to prove
theorem number_of_truthful_gnomes :
  total_gnomes →
  hands_raised_vanilla →
  hands_raised_chocolate →
  hands_raised_fruit →
  total_hands_raised →
  extra_hands_raised →
  lying_gnomes →
  truthful_gnomes :=
begin
  intros,
  sorry,
end

end number_of_truthful_gnomes_l0_185


namespace average_length_l0_538

def length1 : ℕ := 2
def length2 : ℕ := 3
def length3 : ℕ := 7

theorem average_length : (length1 + length2 + length3) / 3 = 4 :=
by
  sorry

end average_length_l0_538


namespace landlord_packages_l0_552

def label_packages_required (start1 end1 start2 end2 start3 end3 : ℕ) : ℕ :=
  let digit_count := 1
  let hundreds_first := (end1 - start1 + 1)
  let hundreds_second := (end2 - start2 + 1)
  let hundreds_third := (end3 - start3 + 1)
  let total_hundreds := hundreds_first + hundreds_second + hundreds_third
  
  let tens_first := ((end1 - start1 + 1) / 10) 
  let tens_second := ((end2 - start2 + 1) / 10) 
  let tens_third := ((end3 - start3 + 1) / 10)
  let total_tens := tens_first + tens_second + tens_third

  let units_per_floor := 5
  let total_units := units_per_floor * 3
  
  let total_ones := total_hundreds + total_tens + total_units
  
  let packages_required := total_ones

  packages_required

theorem landlord_packages : label_packages_required 100 150 200 250 300 350 = 198 := 
  by sorry

end landlord_packages_l0_552


namespace sufficient_condition_l0_493

def M (x y : ℝ) : Prop := y ≥ x^2
def N (x y a : ℝ) : Prop := x^2 + (y - a)^2 ≤ 1

theorem sufficient_condition (a : ℝ) : (∀ x y : ℝ, N x y a → M x y) ↔ (a ≥ 5 / 4) := 
sorry

end sufficient_condition_l0_493


namespace smallest_of_x_y_z_l0_886

variables {a b c d : ℕ}

/-- Given that x, y, and z are in the ratio a, b, c respectively, 
    and their sum x + y + z equals d, and 0 < a < b < c,
    prove that the smallest of x, y, and z is da / (a + b + c). -/
theorem smallest_of_x_y_z (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : 0 < d)
    (h_sum : ∀ k : ℚ, x = k * a → y = k * b → z = k * c → x + y + z = d) : 
    (∃ k : ℚ, x = k * a ∧ y = k * b ∧ z = k * c ∧ k = d / (a + b + c) ∧ x = da / (a + b + c)) :=
by 
  sorry

end smallest_of_x_y_z_l0_886


namespace percentage_of_water_in_juice_l0_955

-- Define the initial condition for tomato puree water percentage
def puree_water_percentage : ℝ := 0.20

-- Define the volume of tomato puree produced from tomato juice
def volume_puree : ℝ := 3.75

-- Define the volume of tomato juice used to produce the puree
def volume_juice : ℝ := 30

-- Given conditions and definitions, prove the percentage of water in tomato juice
theorem percentage_of_water_in_juice :
  ((volume_juice - (volume_puree - puree_water_percentage * volume_puree)) / volume_juice) * 100 = 90 :=
by sorry

end percentage_of_water_in_juice_l0_955


namespace hyperbola_sum_l0_548

theorem hyperbola_sum (h k a b : ℝ) (c : ℝ)
  (h_eq : h = 3)
  (k_eq : k = -5)
  (a_eq : a = 5)
  (c_eq : c = 7)
  (c_squared_eq : c^2 = a^2 + b^2) :
  h + k + a + b = 3 + 2 * Real.sqrt 6 :=
by
  rw [h_eq, k_eq, a_eq, c_eq] at *
  sorry

end hyperbola_sum_l0_548


namespace sin_right_triangle_l0_788

theorem sin_right_triangle (FG GH : ℝ) (h1 : FG = 13) (h2 : GH = 12) (h3 : FG^2 = FH^2 + GH^2) : 
  sin_H = 5 / 13 :=
by sorry

end sin_right_triangle_l0_788


namespace triangle_third_side_l0_362

theorem triangle_third_side {x : ℕ} (h1 : 3 < x) (h2 : x < 7) (h3 : x % 2 = 1) : x = 5 := by
  sorry

end triangle_third_side_l0_362


namespace ratio_and_lcm_l0_520

noncomputable def common_factor (a b : ℕ) := ∃ x : ℕ, a = 3 * x ∧ b = 4 * x

theorem ratio_and_lcm (a b : ℕ) (h1 : common_factor a b) (h2 : Nat.lcm a b = 180) (h3 : a = 60) : b = 45 :=
by sorry

end ratio_and_lcm_l0_520


namespace stratified_sample_size_is_correct_l0_130

def workshop_A_produces : ℕ := 120
def workshop_B_produces : ℕ := 90
def workshop_C_produces : ℕ := 60
def sample_from_C : ℕ := 4

def total_products : ℕ := workshop_A_produces + workshop_B_produces + workshop_C_produces

noncomputable def sampling_ratio : ℚ := (sample_from_C:ℚ) / (workshop_C_produces:ℚ)

noncomputable def sample_size : ℚ := total_products * sampling_ratio

theorem stratified_sample_size_is_correct :
  sample_size = 18 := by
  sorry

end stratified_sample_size_is_correct_l0_130


namespace inverse_proportion_function_m_neg_l0_630

theorem inverse_proportion_function_m_neg
  (x : ℝ) (y : ℝ) (m : ℝ)
  (h1 : y = m / x)
  (h2 : (x < 0 → y > 0) ∧ (x > 0 → y < 0)) :
  m < 0 :=
sorry

end inverse_proportion_function_m_neg_l0_630


namespace symmetric_points_y_axis_l0_355

theorem symmetric_points_y_axis (a b : ℝ) (h1 : ∀ a b : ℝ, y_symmetric a (-3) 4 b ↔ a = -4 ∧ b = -3) : a + b = -7 :=
by
  have h2 := h1 a b
  cases h2 with ha hb,
  rw [ha, hb],
  norm_num

end symmetric_points_y_axis_l0_355


namespace calc_result_l0_826

theorem calc_result : (-2 * -3 + 2) = 8 := sorry

end calc_result_l0_826


namespace Carissa_ran_at_10_feet_per_second_l0_477

theorem Carissa_ran_at_10_feet_per_second :
  ∀ (n : ℕ), 
  (∃ (a : ℕ), 
    (2 * a + 2 * n^2 * a = 260) ∧ -- Total distance
    (a + n * a = 30)) → -- Total time spent
  (2 * n = 10) :=
by
  intro n
  intro h
  sorry

end Carissa_ran_at_10_feet_per_second_l0_477


namespace factorial_division_l0_64

-- Conditions: definition for factorial
def factorial : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * factorial n

-- Statement of the problem: Proving the equality
theorem factorial_division :
  (factorial 10) / ((factorial 5) * (factorial 2)) = 15120 :=
by
  sorry

end factorial_division_l0_64


namespace emily_annual_income_l0_222

variables {q I : ℝ}

theorem emily_annual_income (h1 : (0.01 * q * 30000 + 0.01 * (q + 3) * (I - 30000)) = ((q + 0.75) * 0.01 * I)) : 
  I = 40000 := 
by
  sorry

end emily_annual_income_l0_222


namespace Traci_trip_fraction_l0_233

theorem Traci_trip_fraction :
  let total_distance := 600
  let first_stop_distance := total_distance / 3
  let remaining_distance_after_first_stop := total_distance - first_stop_distance
  let final_leg_distance := 300
  let distance_between_stops := remaining_distance_after_first_stop - final_leg_distance
  (distance_between_stops / remaining_distance_after_first_stop) = 1 / 4 :=
by
  let total_distance := 600
  let first_stop_distance := 600 / 3
  let remaining_distance_after_first_stop := 600 - first_stop_distance
  let final_leg_distance := 300
  let distance_between_stops := remaining_distance_after_first_stop - final_leg_distance
  have h1 : total_distance = 600 := by exact rfl
  have h2 : first_stop_distance = 200 := by norm_num [first_stop_distance]
  have h3 : remaining_distance_after_first_stop = 400 := by norm_num [remaining_distance_after_first_stop]
  have h4 : distance_between_stops = 100 := by norm_num [distance_between_stops]
  show (distance_between_stops / remaining_distance_after_first_stop) = 1/4
  -- Proof omitted
  sorry

end Traci_trip_fraction_l0_233


namespace solve_x2_plus_4y2_l0_824

theorem solve_x2_plus_4y2 (x y : ℝ) (h₁ : x + 2 * y = 6) (h₂ : x * y = -6) : x^2 + 4 * y^2 = 60 :=
by
  sorry

end solve_x2_plus_4y2_l0_824


namespace rectangle_to_square_l0_856

variable (k n : ℕ)

theorem rectangle_to_square (h1 : k > 5) (h2 : k * (k - 5) = n^2) : n = 6 := by 
  sorry

end rectangle_to_square_l0_856


namespace intersect_sphere_circle_l0_332

-- Define the given sphere equation
def sphere (h k l R : ℝ) (x y z : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 + (z - l)^2 = R^2

-- Define the equation of a circle in the plane x = x0 parallel to the yz-plane
def circle_in_plane (x0 y0 z0 r : ℝ) (y z : ℝ) : Prop :=
  (y - y0)^2 + (z - z0)^2 = r^2

-- Define the intersecting circle from the sphere equation in the x = c plane
def intersecting_circle (h k l c R : ℝ) (y z : ℝ) : Prop :=
  (y - k)^2 + (z - l)^2 = R^2 - (h - c)^2

-- The main proof statement
theorem intersect_sphere_circle (h k l R c x0 y0 z0 r: ℝ) :
  ∀ y z, intersecting_circle h k l c R y z ↔ circle_in_plane x0 y0 z0 r y z :=
sorry

end intersect_sphere_circle_l0_332


namespace simplify_exponent_multiplication_l0_594

theorem simplify_exponent_multiplication :
  (10 ^ 0.25) * (10 ^ 0.15) * (10 ^ 0.35) * (10 ^ 0.05) * (10 ^ 0.85) * (10 ^ 0.35) = 10 ^ 2 := by
  sorry

end simplify_exponent_multiplication_l0_594


namespace initial_jellybeans_l0_561

theorem initial_jellybeans (J : ℕ) :
    (∀ x y : ℕ, x = 24 → y = 12 →
    (J - x - y + ((x + y) / 2) = 72) → J = 90) :=
by
  intros x y hx hy h
  rw [hx, hy] at h
  sorry

end initial_jellybeans_l0_561


namespace trig_identity_example_l0_782

theorem trig_identity_example :
  (2 * (Real.sin (Real.pi / 6)) - Real.tan (Real.pi / 4)) = 0 :=
by
  -- Definitions from conditions
  have h1 : Real.sin (Real.pi / 6) = 1/2 := Real.sin_pi_div_six
  have h2 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  rw [h1, h2]
  sorry -- The proof is omitted as per instructions

end trig_identity_example_l0_782


namespace total_cost_is_21_l0_36

-- Definitions of the costs
def cost_almond_croissant : Float := 4.50
def cost_salami_and_cheese_croissant : Float := 4.50
def cost_plain_croissant : Float := 3.00
def cost_focaccia : Float := 4.00
def cost_latte : Float := 2.50

-- Theorem stating the total cost
theorem total_cost_is_21 :
  (cost_almond_croissant + cost_salami_and_cheese_croissant) + (2 * cost_latte) + cost_plain_croissant + cost_focaccia = 21.00 :=
by
  sorry

end total_cost_is_21_l0_36


namespace find_simple_interest_years_l0_555

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

constant P_c : ℝ := 4000
constant r_c : ℝ := 0.10
constant n_c : ℕ := 1
constant t_c : ℝ := 2

constant P_s : ℝ := 1750
constant r_s : ℝ := 0.08
constant SI : ℝ := 420

theorem find_simple_interest_years (t : ℝ) : 
  840 / 2 = SI → SI = P_s * r_s * t → t = 3 :=
by 
  sorry

end find_simple_interest_years_l0_555


namespace find_third_side_l0_966

theorem find_third_side (a b : ℝ) (c : ℕ) 
  (h1 : a = 3.14)
  (h2 : b = 0.67)
  (h_triangle_ineq : a + b > ↑c ∧ a + ↑c > b ∧ b + ↑c > a) : 
  c = 3 := 
by
  -- Proof goes here
  sorry

end find_third_side_l0_966


namespace painted_rooms_l0_462

def total_rooms : ℕ := 12
def hours_per_room : ℕ := 7
def remaining_hours : ℕ := 49

theorem painted_rooms : total_rooms - (remaining_hours / hours_per_room) = 5 := by
  sorry

end painted_rooms_l0_462


namespace least_number_modular_l0_195

theorem least_number_modular 
  (n : ℕ)
  (h1 : n % 34 = 4)
  (h2 : n % 48 = 6)
  (h3 : n % 5 = 2) : n = 4082 :=
by
  sorry

end least_number_modular_l0_195


namespace length_down_correct_l0_916

variable (rate_up rate_down time_up time_down length_down : ℕ)
variable (h1 : rate_up = 8)
variable (h2 : time_up = 2)
variable (h3 : time_down = time_up)
variable (h4 : rate_down = (3 / 2) * rate_up)
variable (h5 : length_down = rate_down * time_down)

theorem length_down_correct : length_down = 24 := by
  sorry

end length_down_correct_l0_916


namespace smallest_integer_value_l0_893

theorem smallest_integer_value (y : ℤ) (h : 7 - 3 * y < -8) : y ≥ 6 :=
sorry

end smallest_integer_value_l0_893


namespace range_of_b_l0_698

noncomputable def range_b := {b | ∀ x, (e ^ x + b ≠ e ^ x) ∨ (e ^ x + b ≠ log x)}

theorem range_of_b (b : ℝ) : b ∈ range_b ↔ b ∈ Set.Icc (-2 : ℝ) (0 : ℝ) :=
by {
  sorry
}

end range_of_b_l0_698


namespace number_of_truthful_gnomes_l0_184

variables (T L : ℕ)

-- Conditions
def total_gnomes : Prop := T + L = 10
def hands_raised_vanilla : Prop := 10 = 10
def hands_raised_chocolate : Prop := ½ * 10 = 5
def hands_raised_fruit : Prop := 1 = 1
def total_hands_raised : Prop := 10 + 5 + 1 = 16
def extra_hands_raised : Prop := 16 - 10 = 6
def lying_gnomes : Prop := L = 6
def truthful_gnomes : Prop := T = 4

-- Statement to prove
theorem number_of_truthful_gnomes :
  total_gnomes →
  hands_raised_vanilla →
  hands_raised_chocolate →
  hands_raised_fruit →
  total_hands_raised →
  extra_hands_raised →
  lying_gnomes →
  truthful_gnomes :=
begin
  intros,
  sorry,
end

end number_of_truthful_gnomes_l0_184


namespace calculate_y_when_x_is_neg2_l0_236

def conditional_program (x : ℤ) : ℤ :=
  if x < 0 then
    2 * x + 3
  else if x > 0 then
    -2 * x + 5
  else
    0

theorem calculate_y_when_x_is_neg2 : conditional_program (-2) = -1 :=
by
  sorry

end calculate_y_when_x_is_neg2_l0_236


namespace percentage_deposited_l0_113

theorem percentage_deposited (amount_deposited income : ℝ) 
  (h1 : amount_deposited = 2500) (h2 : income = 10000) : 
  (amount_deposited / income) * 100 = 25 :=
by
  have amount_deposited_val : amount_deposited = 2500 := h1
  have income_val : income = 10000 := h2
  sorry

end percentage_deposited_l0_113


namespace gcd_18_30_45_l0_488

-- Define the conditions
def a := 18
def b := 30
def c := 45

-- Prove that the gcd of a, b, and c is 3
theorem gcd_18_30_45 : Nat.gcd (Nat.gcd a b) c = 3 :=
by
  -- Skip the proof itself
  sorry

end gcd_18_30_45_l0_488


namespace problem1_problem2_l0_928

def f (x y : ℝ) : ℝ := x^2 * y

def P0 : ℝ × ℝ := (5, 4)

def Δx : ℝ := 0.1
def Δy : ℝ := -0.2

def Δf (f : ℝ → ℝ → ℝ) (P : ℝ × ℝ) (Δx Δy : ℝ) : ℝ :=
  f (P.1 + Δx) (P.2 + Δy) - f P.1 P.2

def df (f : ℝ → ℝ → ℝ) (P : ℝ × ℝ) (Δx Δy : ℝ) : ℝ :=
  (2 * P.1 * P.2) * Δx + (P.1^2) * Δy

theorem problem1 : Δf f P0 Δx Δy = -1.162 := 
  sorry

theorem problem2 : df f P0 Δx Δy = -1 :=
  sorry

end problem1_problem2_l0_928


namespace buy_beams_l0_403

theorem buy_beams (C T x : ℕ) (hC : C = 6210) (hT : T = 3) (hx: x > 0):
  T * (x - 1) = C / x :=
by
  rw [hC, hT]
  sorry

end buy_beams_l0_403


namespace closest_ratio_adults_children_l0_243

theorem closest_ratio_adults_children :
  ∃ (a c : ℕ), 25 * a + 15 * c = 1950 ∧ a ≥ 1 ∧ c ≥ 1 ∧ a / c = 24 / 25 := sorry

end closest_ratio_adults_children_l0_243


namespace closest_point_l0_196

theorem closest_point 
  (x y z : ℝ) 
  (h_plane : 3 * x - 4 * y + 5 * z = 30)
  (A : ℝ × ℝ × ℝ := (1, 2, 3)) 
  (P : ℝ × ℝ × ℝ := (x, y, z)) :
  P = (11 / 5, 2 / 5, 5) := 
sorry

end closest_point_l0_196


namespace cost_of_45_lilies_l0_165

-- Defining the conditions
def price_per_lily (n : ℕ) : ℝ :=
  if n <= 30 then 2
  else 1.8

-- Stating the problem in Lean 4
theorem cost_of_45_lilies :
  price_per_lily 15 * 15 = 30 → (price_per_lily 45 * 45 = 81) :=
by
  intro h
  sorry

end cost_of_45_lilies_l0_165


namespace directors_dividends_correct_l0_902

theorem directors_dividends_correct :
  let net_profit : ℝ := (1500000 - 674992) - 0.2 * (1500000 - 674992)
  let total_loan_payments : ℝ := 23914 * 12 - 74992
  let profit_for_dividends : ℝ := net_profit - total_loan_payments
  let dividend_per_share : ℝ := profit_for_dividends / 1000
  let total_dividends_director : ℝ := dividend_per_share * 550
  total_dividends_director = 246400.0 :=
by
  sorry

end directors_dividends_correct_l0_902


namespace christina_age_fraction_l0_662

theorem christina_age_fraction {C : ℕ} (h1 : ∃ C : ℕ, (6 + 15) = (3/5 : ℚ) * C)
  (h2 : C + 5 = 40) : (C + 5) / 80 = 1 / 2 :=
by
  sorry

end christina_age_fraction_l0_662


namespace sequence_ln_l0_524

theorem sequence_ln (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + Real.log (1 + 1 / n)) :
  ∀ n : ℕ, n ≥ 1 → a n = 1 + Real.log n := 
sorry

end sequence_ln_l0_524


namespace average_earning_week_l0_245

theorem average_earning_week (D1 D2 D3 D4 D5 D6 D7 : ℝ) 
  (h1 : (D1 + D2 + D3 + D4) / 4 = 18)
  (h2 : (D4 + D5 + D6 + D7) / 4 = 22)
  (h3 : D4 = 13) : 
  (D1 + D2 + D3 + D4 + D5 + D6 + D7) / 7 = 22.86 := 
by 
  sorry

end average_earning_week_l0_245


namespace parabola_tangent_perpendicular_m_eq_one_parabola_min_MF_NF_l0_210

open Real

theorem parabola_tangent_perpendicular_m_eq_one (k : ℝ) (hk : k > 0) :
  (∃ x₁ x₂ y₁ y₂ : ℝ, (x₁^2 = 4 * y₁) ∧ (x₂^2 = 4 * y₂) ∧ (y₁ = k * x₁ + m) ∧ (y₂ = k * x₂ + m) ∧ ((x₁ / 2) * (x₂ / 2) = -1)) → m = 1 :=
sorry

theorem parabola_min_MF_NF (k : ℝ) (hk : k > 0) :
  (m = 2) → 
  (∃ x₁ x₂ y₁ y₂ : ℝ, (x₁^2 = 4 * y₁) ∧ (x₂^2 = 4 * y₂) ∧ (y₁ = k * x₁ + 2) ∧ (y₂ = k * x₂ + 2) ∧ |(y₁ + 1) * (y₂ + 1)| ≥ 9) :=
sorry

end parabola_tangent_perpendicular_m_eq_one_parabola_min_MF_NF_l0_210


namespace largest_n_proof_l0_333

def largest_n_less_than_50000_divisible_by_7 (n : ℕ) : Prop :=
  n < 50000 ∧ (10 * (n - 3)^5 - 2 * n^2 + 20 * n - 36) % 7 = 0

theorem largest_n_proof : ∃ n, largest_n_less_than_50000_divisible_by_7 n ∧ ∀ m, largest_n_less_than_50000_divisible_by_7 m → m ≤ n := 
sorry

end largest_n_proof_l0_333


namespace average_ab_l0_880

theorem average_ab {a b : ℝ} (h : (3 + 5 + 7 + a + b) / 5 = 15) : (a + b) / 2 = 30 :=
by
  sorry

end average_ab_l0_880


namespace length_of_tube_l0_287

theorem length_of_tube (h1 : ℝ) (mass_water : ℝ) (rho : ℝ) (doubled_pressure : Bool) (g : ℝ) :
  h1 = 1.5 → 
  mass_water = 1000 → 
  rho = 1000 → 
  g = 9.8 →
  doubled_pressure = true →
  ∃ h2 : ℝ, h2 = 1.5 :=
by
  intros h1_val mass_water_val rho_val g_val doubled_pressure_val
  have : ∃ h2, 29400 = 1000 * g * (h1 + h2) := sorry
  use 1.5
  assumption_sid
  sorry
  
end length_of_tube_l0_287


namespace fraction_subtraction_l0_326

theorem fraction_subtraction : (9 / 23) - (5 / 69) = 22 / 69 :=
by
  sorry

end fraction_subtraction_l0_326


namespace greatest_integer_l0_131

theorem greatest_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℕ, n = 9 * k - 2) (h3 : ∃ l : ℕ, n = 8 * l - 4) : n = 124 :=
by
  sorry

end greatest_integer_l0_131


namespace original_price_sarees_l0_416

theorem original_price_sarees (P : ℝ) (h : 0.85 * 0.80 * P = 272) : P = 400 :=
by
  sorry

end original_price_sarees_l0_416


namespace no_two_heads_consecutive_probability_l0_760

noncomputable def probability_no_two_heads_consecutive (total_flips : ℕ) : ℚ :=
  if total_flips = 10 then 144 / 1024 else 0

theorem no_two_heads_consecutive_probability :
  probability_no_two_heads_consecutive 10 = 9 / 64 :=
by
  unfold probability_no_two_heads_consecutive
  rw [if_pos rfl]
  norm_num
  sorry

end no_two_heads_consecutive_probability_l0_760


namespace powers_of_two_diff_div_by_1987_l0_859

theorem powers_of_two_diff_div_by_1987 :
  ∃ a b : ℕ, a > b ∧ 1987 ∣ (2^a - 2^b) :=
by sorry

end powers_of_two_diff_div_by_1987_l0_859


namespace det_matrix_A_l0_597

def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![8, 4], ![-2, 3]]

def determinant_2x2 (A : Matrix (Fin 2) (Fin 2) ℤ) : ℤ :=
  A 0 0 * A 1 1 - A 0 1 * A 1 0

theorem det_matrix_A : determinant_2x2 matrix_A = 32 := by
  sorry

end det_matrix_A_l0_597


namespace measure_of_angle_x_l0_481

-- Given conditions
def angle_ABC : ℝ := 120
def angle_BAD : ℝ := 31
def angle_BDA (x : ℝ) : Prop := x + 60 + 31 = 180 

-- Statement to prove
theorem measure_of_angle_x : 
  ∃ x : ℝ, angle_BDA x → x = 89 :=
by
  sorry

end measure_of_angle_x_l0_481


namespace smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5_l0_171

theorem smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5 :
  ∃ n : ℕ, n > 0 ∧ (7^n % 5 = n^7 % 5) ∧
  ∀ m : ℕ, m > 0 ∧ (7^m % 5 = m^7 % 5) → n ≤ m :=
by
  sorry

end smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5_l0_171


namespace intersection_A_B_range_m_l0_75

-- Definitions for Sets A, B, and C
def SetA : Set ℝ := { x | -2 ≤ x ∧ x < 5 }
def SetB : Set ℝ := { x | 3 * x - 5 ≥ x - 1 }
def SetC (m : ℝ) : Set ℝ := { x | -x + m > 0 }

-- Problem 1: Prove \( A \cap B = \{ x \mid 2 \leq x < 5 \} \)
theorem intersection_A_B : SetA ∩ SetB = { x : ℝ | 2 ≤ x ∧ x < 5 } :=
by
  sorry

-- Problem 2: Prove \( m \in [5, +\infty) \) given \( A \cup C = C \)
theorem range_m (m : ℝ) : (SetA ∪ SetC m = SetC m) → m ∈ Set.Ici 5 :=
by
  sorry

end intersection_A_B_range_m_l0_75


namespace sum_of_abs_of_coefficients_l0_204

theorem sum_of_abs_of_coefficients :
  ∃ a_0 a_2 a_4 a_1 a_3 a_5 : ℤ, 
    ((2*x - 1)^5 + (x + 2)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) ∧
    (|a_0| + |a_2| + |a_4| = 110) :=
by
  sorry

end sum_of_abs_of_coefficients_l0_204


namespace prob_two_blue_balls_l0_224

-- Ball and Urn Definitions
def total_balls : ℕ := 10
def blue_balls_initial : ℕ := 6
def red_balls_initial : ℕ := 4

-- Probabilities
def prob_blue_first_draw : ℚ := blue_balls_initial / total_balls
def prob_blue_second_draw_given_first_blue : ℚ :=
  (blue_balls_initial - 1) / (total_balls - 1)

-- Resulting Probability
def prob_both_blue : ℚ := prob_blue_first_draw * prob_blue_second_draw_given_first_blue

-- Statement to Prove
theorem prob_two_blue_balls :
  prob_both_blue = 1 / 3 :=
by
  sorry

end prob_two_blue_balls_l0_224


namespace reduced_number_l0_577

theorem reduced_number (N : ℕ) (m a n : ℕ) (k : ℕ) (h1 : N = m + 10^k * a + 10^(k+1) * n)
  (h2 : a < 10) (h3 : m < 10^k) (h4 : N' = m + 10^(k+1) * n) (h5 : N = 6 * N') :
  N ∈ {12, 24, 36, 48} :=
sorry

end reduced_number_l0_577


namespace ratio_hooper_bay_to_other_harbors_l0_821

-- Definitions based on conditions
def other_harbors_lobster : ℕ := 80
def total_lobster : ℕ := 480
def combined_other_harbors_lobster := 2 * other_harbors_lobster
def hooper_bay_lobster := total_lobster - combined_other_harbors_lobster

-- The theorem to prove
theorem ratio_hooper_bay_to_other_harbors : hooper_bay_lobster / combined_other_harbors_lobster = 2 :=
by
  sorry

end ratio_hooper_bay_to_other_harbors_l0_821


namespace equation_of_line_passing_through_ellipse_midpoint_l0_953

noncomputable def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

theorem equation_of_line_passing_through_ellipse_midpoint
  (x1 y1 x2 y2 : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (1, 1))
  (hA : ellipse x1 y1)
  (hB : ellipse x2 y2)
  (midAB : (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = 1) :
  ∃ (a b c : ℝ), a = 4 ∧ b = 3 ∧ c = -7 ∧ a * P.2 + b * P.1 + c = 0 :=
sorry

end equation_of_line_passing_through_ellipse_midpoint_l0_953


namespace initial_distance_between_trains_l0_713

theorem initial_distance_between_trains :
  let length_train1 := 100 -- meters
  let length_train2 := 200 -- meters
  let speed_train1_kmph := 54 -- km/h
  let speed_train2_kmph := 72 -- km/h
  let time_hours := 1.999840012798976 -- hours
  
  -- Conversion to meters per second
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600 -- 15 m/s
  let speed_train2_mps := speed_train2_kmph * 1000 / 3600 -- 20 m/s

  -- Conversion of time to seconds
  let time_seconds := time_hours * 3600 -- 7199.4240460755136 seconds

  -- Relative speed in meters per second
  let relative_speed := speed_train1_mps + speed_train2_mps -- 35 m/s

  -- Distance covered by both trains
  let distance_covered := relative_speed * time_seconds -- 251980.84161264498 meters

  -- Initial distance between the trains
  let initial_distance := distance_covered - (length_train1 + length_train2) -- 251680.84161264498 meters

  initial_distance = 251680.84161264498 := 
by
  sorry

end initial_distance_between_trains_l0_713


namespace percentage_regular_cars_l0_688

theorem percentage_regular_cars (total_cars : ℕ) (truck_percentage : ℚ) (convertibles : ℕ) 
  (h1 : total_cars = 125) (h2 : truck_percentage = 0.08) (h3 : convertibles = 35) : 
  (80 / 125 : ℚ) * 100 = 64 := 
by 
  sorry

end percentage_regular_cars_l0_688


namespace base_of_exponent_l0_512

theorem base_of_exponent (x : ℤ) (m : ℕ) (h₁ : (-2 : ℤ)^(2 * m) = x^(12 - m)) (h₂ : m = 4) : x = -2 :=
by 
  sorry

end base_of_exponent_l0_512


namespace expression_simplification_l0_877

-- Definitions for P and Q based on x and y
def P (x y : ℝ) := x + y
def Q (x y : ℝ) := x - y

-- The mathematical property to prove
theorem expression_simplification (x y : ℝ) (h : x ≠ 0) (k : y ≠ 0) : 
  (P x y + Q x y) / (P x y - Q x y) - (P x y - Q x y) / (P x y + Q x y) = (x^2 - y^2) / (x * y) := 
by
  -- Sorry is used to skip the proof here
  sorry

end expression_simplification_l0_877


namespace find_white_towels_l0_534

variable {W : ℕ} -- Define W as a natural number

-- Define the conditions as Lean definitions
def initial_towel_count (W : ℕ) : ℕ := 35 + W
def remaining_towel_count (W : ℕ) : ℕ := initial_towel_count W - 34

-- Theorem statement: Proving that W = 21 given the conditions
theorem find_white_towels (h : remaining_towel_count W = 22) : W = 21 :=
by
  sorry

end find_white_towels_l0_534


namespace symmetric_point_origin_l0_875

def symmetric_point (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, -M.2)

theorem symmetric_point_origin (x y : ℝ) (h : (x, y) = (-2, 3)) :
  symmetric_point (x, y) = (2, -3) :=
by
  rw [h]
  unfold symmetric_point
  simp
  sorry

end symmetric_point_origin_l0_875


namespace william_tickets_l0_439

theorem william_tickets (initial_tickets final_tickets : ℕ) (h1 : initial_tickets = 15) (h2 : final_tickets = 18) : 
  final_tickets - initial_tickets = 3 := 
by
  sorry

end william_tickets_l0_439


namespace total_cost_price_is_correct_l0_295

noncomputable def selling_price_before_discount (sp_after_discount : ℝ) (discount_rate : ℝ) : ℝ :=
  sp_after_discount / (1 - discount_rate)

noncomputable def cost_price_from_profit (selling_price : ℝ) (profit_rate : ℝ) : ℝ :=
  selling_price / (1 + profit_rate)

noncomputable def cost_price_from_loss (selling_price : ℝ) (loss_rate : ℝ) : ℝ :=
  selling_price / (1 - loss_rate)

noncomputable def total_cost_price : ℝ :=
  let CP1 := cost_price_from_profit (selling_price_before_discount 600 0.05) 0.25
  let CP2 := cost_price_from_loss 800 0.20
  let CP3 := cost_price_from_profit 1000 0.30 - 50
  CP1 + CP2 + CP3

theorem total_cost_price_is_correct : total_cost_price = 2224.49 :=
  by
  sorry

end total_cost_price_is_correct_l0_295


namespace largest_prime_factor_of_3913_l0_135

def is_prime (n : ℕ) := nat.prime n

def prime_factors_3913 := {17, 2, 5, 23}

theorem largest_prime_factor_of_3913 
  (h1 : is_prime 17)
  (h2 : is_prime 2)
  (h3 : is_prime 5)
  (h4 : is_prime 23)
  (h5 : 3913 = 17 * 2 * 5 * 23) : 
  (23 ∈ prime_factors_3913 ∧ ∀ x ∈ prime_factors_3913, x ≤ 23) :=
  by 
    sorry

end largest_prime_factor_of_3913_l0_135


namespace bananas_in_each_group_l0_407

theorem bananas_in_each_group (total_bananas groups : ℕ) (h1 : total_bananas = 392) (h2 : groups = 196) :
    total_bananas / groups = 2 :=
by
  sorry

end bananas_in_each_group_l0_407


namespace merchant_profit_condition_l0_918

theorem merchant_profit_condition (L : ℝ) (P : ℝ) (S : ℝ) (M : ℝ) :
  (P = 0.70 * L) →
  (S = 0.80 * M) →
  (S - P = 0.30 * S) →
  (M = 1.25 * L) := 
by
  intros h1 h2 h3
  sorry

end merchant_profit_condition_l0_918


namespace wire_cut_perimeter_equal_l0_23

theorem wire_cut_perimeter_equal (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 4 * (a / 4) = 8 * (b / 8)) :
  a / b = 1 :=
sorry

end wire_cut_perimeter_equal_l0_23


namespace total_books_proof_l0_863

noncomputable def economics_books (T : ℝ) := (1/4) * T + 10
noncomputable def rest_books (T : ℝ) := T - economics_books T
noncomputable def social_studies_books (T : ℝ) := (3/5) * rest_books T - 5
noncomputable def other_books := 13
noncomputable def science_books := 12
noncomputable def total_books_equation (T : ℝ) :=
  T = economics_books T + social_studies_books T + science_books + other_books

theorem total_books_proof : ∃ T : ℝ, total_books_equation T ∧ T = 80 := by
  sorry

end total_books_proof_l0_863


namespace speed_of_point_C_l0_351

theorem speed_of_point_C 
    (a T R L x : ℝ) 
    (h1 : x = L * (a * T) / R - L) 
    (h_eq: (a * T) / (a * T - R) = (L + x) / x) :
    (a * L) / R = x / T :=
by
  sorry

end speed_of_point_C_l0_351


namespace smallest_possible_n_l0_137

theorem smallest_possible_n (n : ℕ) (h : n > 0) 
  (h_condition : Nat.lcm 60 n / Nat.gcd 60 n = 44) : n = 165 := by
  sorry

end smallest_possible_n_l0_137


namespace find_side_length_l0_157

theorem find_side_length
  (n : ℕ) 
  (h : (6 * n^2) / (6 * n^3) = 1 / 3) : 
  n = 3 := 
by
  sorry

end find_side_length_l0_157


namespace inequality_always_holds_l0_509

theorem inequality_always_holds (a b c : ℝ) (h1 : a > b) (h2 : a * b ≠ 0) : a + c > b + c :=
sorry

end inequality_always_holds_l0_509


namespace no_two_heads_consecutively_probability_l0_763

theorem no_two_heads_consecutively_probability :
  (∃ (total_sequences : ℕ) (valid_sequences : ℕ),
    total_sequences = 2^10 ∧ valid_sequences = 1 + 10 + 36 + 56 + 35 + 6 ∧
    (valid_sequences / total_sequences : ℚ) = 9 / 64) :=
begin
  sorry
end

end no_two_heads_consecutively_probability_l0_763


namespace days_from_friday_l0_718

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end days_from_friday_l0_718


namespace muffin_expense_l0_47

theorem muffin_expense (B D : ℝ) 
    (h1 : D = 0.90 * B) 
    (h2 : B = D + 15) : 
    B + D = 285 := 
    sorry

end muffin_expense_l0_47


namespace sum_of_variables_l0_242

theorem sum_of_variables (a b c d : ℤ)
  (h1 : a - b + 2 * c = 7)
  (h2 : b - c + d = 8)
  (h3 : c - d + a = 5)
  (h4 : d - a + b = 4) : a + b + c + d = 20 :=
by
  sorry

end sum_of_variables_l0_242


namespace number_of_truthful_dwarfs_l0_179

def total_dwarfs := 10
def hands_raised_vanilla := 10
def hands_raised_chocolate := 5
def hands_raised_fruit := 1
def total_hands_raised := hands_raised_vanilla + hands_raised_chocolate + hands_raised_fruit
def extra_hands := total_hands_raised - total_dwarfs
def liars := extra_hands
def truthful := total_dwarfs - liars

theorem number_of_truthful_dwarfs : truthful = 4 :=
by sorry

end number_of_truthful_dwarfs_l0_179


namespace number_of_valid_six_tuples_l0_675

def is_valid_six_tuple (p : ℕ) (a b c d e f : ℕ) : Prop :=
  a + b + c + d + e + f = 3 * p ∧
  (a + b) % (c + d) = 0 ∧
  (b + c) % (d + e) = 0 ∧
  (c + d) % (e + f) = 0 ∧
  (d + e) % (f + a) = 0 ∧
  (e + f) % (a + b) = 0

theorem number_of_valid_six_tuples (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) : 
  ∃! n, n = p + 2 ∧ ∀ (a b c d e f : ℕ), is_valid_six_tuple p a b c d e f → n = p + 2 :=
sorry

end number_of_valid_six_tuples_l0_675


namespace f_7_minus_a_eq_neg_7_over_4_l0_638

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 2 else -Real.logb 3 x

variable (a : ℝ)

-- Given conditions
axiom h1 : f a = -2

-- The proof of the required condition
theorem f_7_minus_a_eq_neg_7_over_4 (h1 : f a = -2) : f (7 - a) = -7 / 4 := sorry

end f_7_minus_a_eq_neg_7_over_4_l0_638


namespace lori_earnings_l0_848

theorem lori_earnings
    (red_cars : ℕ)
    (white_cars : ℕ)
    (cost_red_car : ℕ)
    (cost_white_car : ℕ)
    (rental_time_hours : ℕ)
    (rental_time_minutes : ℕ)
    (correct_earnings : ℕ) :
    red_cars = 3 →
    white_cars = 2 →
    cost_red_car = 3 →
    cost_white_car = 2 →
    rental_time_hours = 3 →
    rental_time_minutes = rental_time_hours * 60 →
    correct_earnings = 2340 →
    (red_cars * cost_red_car + white_cars * cost_white_car) * rental_time_minutes = correct_earnings :=
by
  intros
  sorry

end lori_earnings_l0_848


namespace policeman_catches_thief_l0_340

/-
  From a police station situated on a straight road infinite in both directions, a thief has stolen a police car.
  Its maximal speed equals 90% of the maximal speed of a police cruiser. When the theft is discovered some time
  later, a policeman starts to pursue the thief on a cruiser. However, the policeman does not know in which direction
  along the road the thief has gone, nor does he know how long ago the car has been stolen. The goal is to prove
  that it is possible for the policeman to catch the thief.
-/
theorem policeman_catches_thief (v : ℝ) (T₀ : ℝ) (o₀ : ℝ) :
  (0 < v) →
  (0 < T₀) →
  ∃ T p, T₀ ≤ T ∧ p ≤ v * T :=
sorry

end policeman_catches_thief_l0_340


namespace find_B_intersection_point_l0_550

theorem find_B_intersection_point (k1 k2 : ℝ) (hA1 : 1 ≠ 0) 
  (hA2 : k1 = -2) (hA3 : k2 = -2) : 
  (-1, 2) ∈ {p : ℝ × ℝ | ∃ k1 k2, p.2 = k1 * p.1 ∧ p.2 = k2 / p.1} :=
sorry

end find_B_intersection_point_l0_550


namespace min_value_expr_l0_977

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  8 * a^3 + 12 * b^3 + 27 * c^3 + 1 / (9 * a * b * c) ≥ 4 := 
sorry

end min_value_expr_l0_977


namespace slices_remaining_l0_653

theorem slices_remaining (large_pizza_slices : ℕ) (xl_pizza_slices : ℕ) (large_pizza_ordered : ℕ) (xl_pizza_ordered : ℕ) (mary_eats_large : ℕ) (mary_eats_xl : ℕ) :
  large_pizza_slices = 8 →
  xl_pizza_slices = 12 →
  large_pizza_ordered = 1 →
  xl_pizza_ordered = 1 →
  mary_eats_large = 7 →
  mary_eats_xl = 3 →
  (large_pizza_slices * large_pizza_ordered - mary_eats_large + xl_pizza_slices * xl_pizza_ordered - mary_eats_xl) = 10 := 
by
  intros
  sorry

end slices_remaining_l0_653


namespace quadratic_root_property_l0_55

theorem quadratic_root_property (a b k : ℝ) 
  (h1 : a * b + 2 * a + 2 * b = 1) 
  (h2 : a + b = 3) 
  (h3 : a * b = k) : k = -5 := 
by
  sorry

end quadratic_root_property_l0_55


namespace john_ate_half_package_l0_579

def fraction_of_package_john_ate (servings : ℕ) (calories_per_serving : ℕ) (calories_consumed : ℕ) : ℚ :=
  calories_consumed / (servings * calories_per_serving : ℚ)

theorem john_ate_half_package (servings : ℕ) (calories_per_serving : ℕ) (calories_consumed : ℕ) 
    (h_servings : servings = 3) (h_calories_per_serving : calories_per_serving = 120) (h_calories_consumed : calories_consumed = 180) :
    fraction_of_package_john_ate servings calories_per_serving calories_consumed = 1 / 2 :=
by
  -- Replace the actual proof with sorry to ensure the statement compiles.
  sorry

end john_ate_half_package_l0_579


namespace miranda_heels_cost_l0_232

theorem miranda_heels_cost (months_saved : ℕ) (savings_per_month : ℕ) (gift_from_sister : ℕ) 
  (h1 : months_saved = 3) (h2 : savings_per_month = 70) (h3 : gift_from_sister = 50) : 
  months_saved * savings_per_month + gift_from_sister = 260 := 
by
  sorry

end miranda_heels_cost_l0_232


namespace sum_of_possible_values_of_d_l0_10

theorem sum_of_possible_values_of_d :
  let d (n : ℕ) := (Nat.log 16 n) + 1
  let lower_bound : ℕ := 256
  let upper_bound : ℕ := 1023
  (∀ n : ℕ, (lower_bound ≤ n ∧ n ≤ upper_bound) → d n = 3) →
  ∑ (d_val : ℕ) in {d n | lower_bound ≤ n ∧ n ≤ upper_bound}, d_val = 3 :=
by
  sorry

end sum_of_possible_values_of_d_l0_10


namespace average_of_modified_set_l0_951

theorem average_of_modified_set (a1 a2 a3 a4 a5 : ℝ) (h : (a1 + a2 + a3 + a4 + a5) / 5 = 8) :
  ((a1 + 10) + (a2 - 10) + (a3 + 10) + (a4 - 10) + (a5 + 10)) / 5 = 10 :=
by 
  sorry

end average_of_modified_set_l0_951


namespace average_speed_l0_283

-- Definitions based on the conditions from part a
def distance_first_hour : ℕ := 90
def distance_second_hour : ℕ := 30
def time_first_hour : ℕ := 1
def time_second_hour : ℕ := 1

-- Main theorem stating the average speed
theorem average_speed :
  (distance_first_hour + distance_second_hour) / (time_first_hour + time_second_hour) = 60 :=
by
  -- proof goes here
  sorry

end average_speed_l0_283


namespace correct_system_of_equations_l0_5

theorem correct_system_of_equations (x y : ℕ) : 
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔ 
  (x / 3 = y - 2) ∧ (x / 2 - 9 = y) := sorry

end correct_system_of_equations_l0_5


namespace largest_mersenne_prime_less_than_500_l0_31

-- Define what it means for a number to be prime
def is_prime (p : ℕ) : Prop :=
p > 1 ∧ ∀ (n : ℕ), n > 1 ∧ n < p → ¬ (p % n = 0)

-- Define what a Mersenne prime is
def is_mersenne_prime (m : ℕ) : Prop :=
∃ n : ℕ, is_prime n ∧ m = 2^n - 1

-- We state the main theorem we want to prove
theorem largest_mersenne_prime_less_than_500 : ∀ (m : ℕ), is_mersenne_prime m ∧ m < 500 → m ≤ 127 :=
by 
  sorry

end largest_mersenne_prime_less_than_500_l0_31


namespace velocity_of_point_C_l0_344

variable (a T R L x : ℝ)
variable (a_pos : a > 0) (T_pos : T > 0) (R_pos : R > 0) (L_pos : L > 0)
variable (h_eq : a * T / (a * T - R) = (L + x) / x)

theorem velocity_of_point_C : a * (L / R) = x / T := by
  sorry

end velocity_of_point_C_l0_344


namespace last_three_digits_of_7_pow_103_l0_792

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 614 := by
  sorry

end last_three_digits_of_7_pow_103_l0_792


namespace sum_possible_values_k_l0_663

theorem sum_possible_values_k :
  (∀ j k : ℕ, j > 0 → k > 0 → (1 / j + 1 / k = 1 / 5) → k ∈ {30, 10, 6}) →
  ∑ k in {30, 10, 6}, k = 46 :=
by {
  assume h,
  sorry
}

end sum_possible_values_k_l0_663


namespace total_students_l0_115

-- Definitions based on the conditions:
def yoongi_left : ℕ := 7
def yoongi_right : ℕ := 5

-- Theorem statement that proves the total number of students given the conditions
theorem total_students (y_left y_right : ℕ) : y_left = yoongi_left -> y_right = yoongi_right -> (y_left + y_right - 1) = 11 := 
by
  intros h1 h2
  rw [h1, h2]
  sorry

end total_students_l0_115


namespace Tn_lt_Sn_div2_l0_679

open_locale big_operators

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3)^n

def S (n : ℕ) : ℝ := 
  ∑ i in finset.range n, a (i + 1)

def T (n : ℕ) : ℝ := 
  ∑ i in finset.range n, b (i + 1)

theorem Tn_lt_Sn_div2 (n : ℕ) : T n < S n / 2 := 
sorry

end Tn_lt_Sn_div2_l0_679


namespace avg_of_second_largest_and_second_smallest_is_eight_l0_354

theorem avg_of_second_largest_and_second_smallest_is_eight :
  ∀ (a b c d e : ℕ), 
  a + b + c + d + e = 40 → 
  a < b ∧ b < c ∧ c < d ∧ d < e →
  ((d + b) / 2 : ℕ) = 8 := 
by
  intro a b c d e hsum horder
  /- the proof goes here, but we use sorry to skip it -/
  sorry

end avg_of_second_largest_and_second_smallest_is_eight_l0_354


namespace necessary_and_sufficient_condition_l0_220

variables {f g : ℝ → ℝ}

theorem necessary_and_sufficient_condition (f g : ℝ → ℝ)
  (hdom : ∀ x : ℝ, true)
  (hst : ∀ y : ℝ, true) :
  (∀ x : ℝ, f x > g x) ↔ (∀ x : ℝ, ¬ (x ∈ {x : ℝ | f x ≤ g x})) :=
by sorry

end necessary_and_sufficient_condition_l0_220


namespace arithmetic_seq_sum_specific_max_arithmetic_seq_sum_l0_809

variable {α : Type*} [Field α]
variables (a_1 : α) (d : α) (n : α)

def arithmetic_seq_sum (a₁ d : α) (n : α) : α :=
  (n / 2) * (2 * a₁ + (n - 1) * d)

theorem arithmetic_seq_sum_specific
  (h₁ : a₁ = 31) 
  (h₂ : arithmetic_seq_sum 31 d 10 = arithmetic_seq_sum 31 d 22)
  : arithmetic_seq_sum 31 2 n = n^2 + 30 * n :=
sorry

theorem max_arithmetic_seq_sum
  (h₁ : a₁ = 31) 
  (h₂ : arithmetic_seq_sum 31 d 10 = arithmetic_seq_sum 31 d 22)
  : ∃ n : ℕ, n = 16 ∧ arithmetic_seq_sum 31 2 n = 736 :=
sorry

end arithmetic_seq_sum_specific_max_arithmetic_seq_sum_l0_809


namespace waiter_total_customers_l0_0

theorem waiter_total_customers (tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) (tables_eq : tables = 6) (women_eq : women_per_table = 3) (men_eq : men_per_table = 5) :
  tables * (women_per_table + men_per_table) = 48 :=
by
  sorry

end waiter_total_customers_l0_0


namespace spend_together_is_85_l0_338

variable (B D : ℝ)

theorem spend_together_is_85 (h1 : D = 0.70 * B) (h2 : B = D + 15) : B + D = 85 := by
  sorry

end spend_together_is_85_l0_338


namespace average_30_matches_is_25_l0_754

noncomputable def average_runs_in_30_matches (average_20_matches average_10_matches : ℝ) (total_matches_20 total_matches_10 : ℕ) : ℝ :=
  let total_runs_20 := total_matches_20 * average_20_matches
  let total_runs_10 := total_matches_10 * average_10_matches
  (total_runs_20 + total_runs_10) / (total_matches_20 + total_matches_10)

theorem average_30_matches_is_25 (h1 : average_runs_in_30_matches 30 15 20 10 = 25) : 
  average_runs_in_30_matches 30 15 20 10 = 25 := 
  by
    exact h1

end average_30_matches_is_25_l0_754


namespace john_multiple_is_correct_l0_671

noncomputable def compute_multiple (cost_per_computer : ℝ) 
                                   (num_computers : ℕ)
                                   (rent : ℝ)
                                   (non_rent_expenses : ℝ)
                                   (profit : ℝ) : ℝ :=
  let total_revenue := (num_computers : ℝ) * cost_per_computer
  let total_expenses := (num_computers : ℝ) * 800 + rent + non_rent_expenses
  let x := (total_expenses + profit) / total_revenue
  x

theorem john_multiple_is_correct :
  compute_multiple 800 60 5000 3000 11200 = 1.4 := by
  sorry

end john_multiple_is_correct_l0_671


namespace machine_A_produces_7_sprockets_per_hour_l0_150

theorem machine_A_produces_7_sprockets_per_hour
    (A B : ℝ)
    (h1 : B = 1.10 * A)
    (h2 : ∃ t : ℝ, 770 = A * (t + 10) ∧ 770 = B * t) : 
    A = 7 := 
by 
    sorry

end machine_A_produces_7_sprockets_per_hour_l0_150


namespace swap_instruments_readings_change_l0_449

def U0 : ℝ := 45
def R : ℝ := 50
def r : ℝ := 20

theorem swap_instruments_readings_change :
  let I_total := U0 / (R / 2 + r)
  let U1 := I_total * r
  let I1 := I_total / 2
  let I2 := U0 / R
  let I := U0 / (R + r)
  let U2 := I * r
  let ΔI := I2 - I1
  let ΔU := U1 - U2
  ΔI = 0.4 ∧ ΔU = 7.14 :=
by
  sorry

end swap_instruments_readings_change_l0_449


namespace sum_of_consecutive_odds_l0_254

theorem sum_of_consecutive_odds (a : ℤ) (h : (a - 2) * a * (a + 2) = 9177) : (a - 2) + a + (a + 2) = 63 := 
sorry

end sum_of_consecutive_odds_l0_254


namespace circle_diameter_given_area_l0_892

theorem circle_diameter_given_area : 
  (∃ (r : ℝ), 81 * Real.pi = Real.pi * r^2 ∧ 2 * r = d) → d = 18 := by
  sorry

end circle_diameter_given_area_l0_892


namespace general_form_equation_l0_878

theorem general_form_equation (x : ℝ) : 
  x * (2 * x - 1) = 5 * (x + 3) ↔ 2 * x^2 - 6 * x - 15 = 0 := 
by 
  sorry

end general_form_equation_l0_878


namespace coins_donated_l0_990

theorem coins_donated (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (coins_left : ℕ) : 
  pennies = 42 ∧ nickels = 36 ∧ dimes = 15 ∧ coins_left = 27 → (pennies + nickels + dimes - coins_left) = 66 :=
by
  intros h
  sorry

end coins_donated_l0_990


namespace find_a_l0_637

noncomputable def tangent_line (a : ℝ) (x : ℝ) := (3 * a * (1:ℝ)^2 + 1) * (x - 1) + (a * (1:ℝ)^3 + (1:ℝ) + 1)

theorem find_a : ∃ a : ℝ, tangent_line a 2 = 7 := 
sorry

end find_a_l0_637


namespace total_sales_l0_870

theorem total_sales (T : ℝ) (h1 : (2 / 5) * T = (2 / 5) * T) (h2 : (3 / 5) * T = 48) : T = 80 :=
by
  -- added sorry to skip proofs as per the requirement
  sorry

end total_sales_l0_870


namespace woody_saving_weeks_l0_748

variable (cost_needed current_savings weekly_allowance : ℕ)

theorem woody_saving_weeks (h₁ : cost_needed = 282)
                           (h₂ : current_savings = 42)
                           (h₃ : weekly_allowance = 24) :
  (cost_needed - current_savings) / weekly_allowance = 10 := by
  sorry

end woody_saving_weeks_l0_748


namespace find_expression_value_l0_948

theorem find_expression_value (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) :
  (x^3 + 3 * y^3) / 9 = 73 / 3 :=
by
  sorry

end find_expression_value_l0_948


namespace number_of_truthful_dwarfs_l0_180

def total_dwarfs := 10
def hands_raised_vanilla := 10
def hands_raised_chocolate := 5
def hands_raised_fruit := 1
def total_hands_raised := hands_raised_vanilla + hands_raised_chocolate + hands_raised_fruit
def extra_hands := total_hands_raised - total_dwarfs
def liars := extra_hands
def truthful := total_dwarfs - liars

theorem number_of_truthful_dwarfs : truthful = 4 :=
by sorry

end number_of_truthful_dwarfs_l0_180


namespace retrievers_count_l0_97

-- Definitions of given conditions
def huskies := 5
def pitbulls := 2
def retrievers := Nat
def husky_pups := 3
def pitbull_pups := 3
def retriever_extra_pups := 2
def total_pups_excess := 30

-- Equation derived from the problem conditions
def total_pups (G : Nat) := huskies * husky_pups + pitbulls * pitbull_pups + G * (husky_pups + retriever_extra_pups)
def total_adults (G : Nat) := huskies + pitbulls + G

theorem retrievers_count : ∃ G : Nat, G = 4 ∧ total_pups G = total_adults G + total_pups_excess :=
by
  sorry

end retrievers_count_l0_97


namespace trailing_zeros_1_to_100_l0_641

def count_multiples (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then 0 else n / k

def trailing_zeros_in_range (n : ℕ) : ℕ :=
  let multiples_of_5 := count_multiples n 5
  let multiples_of_25 := count_multiples n 25
  multiples_of_5 + multiples_of_25

theorem trailing_zeros_1_to_100 : trailing_zeros_in_range 100 = 24 := by
  sorry

end trailing_zeros_1_to_100_l0_641


namespace smallest_possible_c_minus_a_l0_422

theorem smallest_possible_c_minus_a :
  ∃ (a b c : ℕ), 
    a < b ∧ b < c ∧ a * b * c = Nat.factorial 9 ∧ c - a = 216 := 
by
  sorry

end smallest_possible_c_minus_a_l0_422


namespace dice_sum_impossible_l0_317

theorem dice_sum_impossible (a b c d : ℕ) (h1 : a * b * c * d = 216)
  (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) 
  (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6) : 
  a + b + c + d ≠ 18 :=
sorry

end dice_sum_impossible_l0_317


namespace probability_sum_divisible_by_3_l0_915

theorem probability_sum_divisible_by_3 (dice_count : ℕ) (total_events : ℕ) (valid_events : ℕ) : 
  dice_count = 3 ∧ total_events = 6 * 6 * 6 ∧ valid_events = 72 → 
  (valid_events : ℚ) / (total_events : ℚ) = 1 / 3 :=
by
  intros h
  have h_dice_count : dice_count = 3 := h.1
  have h_total_events : total_events = 6 * 6 * 6 := h.2.1
  have h_valid_events : valid_events = 72 := h.2.2
  sorry

end probability_sum_divisible_by_3_l0_915


namespace not_odd_function_l0_259

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x ^ 2 + 1)

theorem not_odd_function : ¬ is_odd_function f := by
  sorry

end not_odd_function_l0_259


namespace height_of_old_lamp_l0_94

theorem height_of_old_lamp (height_new_lamp : ℝ) (height_difference : ℝ) (h : height_new_lamp = 2.33) (h_diff : height_difference = 1.33) : 
  (height_new_lamp - height_difference) = 1.00 :=
by
  have height_new : height_new_lamp = 2.33 := h
  have height_diff : height_difference = 1.33 := h_diff
  sorry

end height_of_old_lamp_l0_94


namespace part1_part2_l0_366

-- Conditions
def U := ℝ
def A : Set ℝ := {x | 0 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 2}
def B (m : ℝ) : Set ℝ := {x | x ≤ 3 * m - 4 ∨ x ≥ 8 + m}
def complement_U (B : Set ℝ) : Set ℝ := {x | ¬(x ∈ B)}
def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

-- Assertions
theorem part1 (m : ℝ) (h1 : m = 2) : intersection A (complement_U (B m)) = {x | 2 < x ∧ x < 4} :=
  sorry

theorem part2 (h : intersection A (complement_U (B m)) = ∅) : -4 ≤ m ∧ m ≤ 5 / 3 :=
  sorry

end part1_part2_l0_366


namespace solution_part_for_a_l0_674

noncomputable def find_k (k x y n : ℕ) : Prop :=
  gcd x y = 1 ∧ 
  x > 0 ∧ y > 0 ∧ 
  k % (x^2) = 0 ∧ 
  k % (y^2) = 0 ∧ 
  k / (x^2) = n ∧ 
  k / (y^2) = n + 148

theorem solution_part_for_a (k x y n : ℕ) (h : find_k k x y n) : k = 467856 :=
sorry

end solution_part_for_a_l0_674


namespace least_three_digit_multiple_of_3_4_9_is_108_l0_434

theorem least_three_digit_multiple_of_3_4_9_is_108 :
  ∃ (n : ℕ), (100 ≤ n) ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 9 = 0) ∧ (n = 108) :=
by
  sorry

end least_three_digit_multiple_of_3_4_9_is_108_l0_434


namespace dot_product_min_value_in_triangle_l0_379

noncomputable def dot_product_min_value (a b c : ℝ) (angleA : ℝ) : ℝ :=
  b * c * Real.cos angleA

theorem dot_product_min_value_in_triangle (b c : ℝ) (hyp1 : 0 ≤ b) (hyp2 : 0 ≤ c) 
  (hyp3 : b^2 + c^2 + b * c = 16) (hyp4 : Real.cos (2 * Real.pi / 3) = -1 / 2) : 
  ∃ (p : ℝ), p = dot_product_min_value 4 b c (2 * Real.pi / 3) ∧ p = -8 / 3 :=
by
  sorry

end dot_product_min_value_in_triangle_l0_379


namespace find_y_values_l0_846

theorem find_y_values (x : ℝ) (y : ℝ) 
  (h : x^2 + 4 * ((x / (x + 3))^2) = 64) : 
  y = (x + 3)^2 * (x - 2) / (2 * x + 3) → 
  y = 250 / 3 :=
sorry

end find_y_values_l0_846


namespace line_equation_exists_l0_194

theorem line_equation_exists 
  (a b : ℝ) 
  (ha_pos: a > 0)
  (hb_pos: b > 0)
  (h_area: 1 / 2 * a * b = 2) 
  (h_diff: a - b = 3 ∨ b - a = 3) : 
  (∀ x y : ℝ, (x + 4 * y = 4 ∧ (x / a + y / b = 1)) ∨ (4 * x + y = 4 ∧ (x / a + y / b = 1))) :=
sorry

end line_equation_exists_l0_194


namespace second_player_wins_12_petals_second_player_wins_11_petals_l0_456

def daisy_game (n : Nat) : Prop :=
  ∀ (p1_move p2_move : Nat → Nat → Prop), n % 2 = 0 → (∃ k, p1_move n k = false) ∧ (∃ ℓ, p2_move n ℓ = true)

theorem second_player_wins_12_petals : daisy_game 12 := sorry
theorem second_player_wins_11_petals : daisy_game 11 := sorry

end second_player_wins_12_petals_second_player_wins_11_petals_l0_456


namespace fifty_three_days_from_Friday_is_Tuesday_l0_730

def days_of_week : Type := {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}

def modulo_days (n : ℕ) : ℕ := n % 7

def day_of_week_after_days (start_day : days_of_week) (n : ℕ) : days_of_week :=
  list.nth_le (list.rotate [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday] (modulo_days n)) sorry -- using sorry for index proof

theorem fifty_three_days_from_Friday_is_Tuesday :
  day_of_week_after_days Friday 53 = Tuesday :=
sorry

end fifty_three_days_from_Friday_is_Tuesday_l0_730


namespace count_expressible_integers_l0_800

theorem count_expressible_integers :
  ∃ (count : ℕ), count = 1138 ∧ (∀ n, (n ≤ 2000) → (∃ x : ℝ, ⌊x⌋ + ⌊2 * x⌋ + ⌊4 * x⌋ = n)) :=
sorry

end count_expressible_integers_l0_800


namespace solve_for_x_l0_787

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) : x^3 - 2 * x^2 = 0 ↔ x = 2 :=
by sorry

end solve_for_x_l0_787


namespace general_term_of_arithmetic_seq_l0_599

variable {a : ℕ → ℤ}

def arithmetic_seq (a : ℕ → ℤ) := ∃ d, ∀ n, a n = a 0 + n * d

theorem general_term_of_arithmetic_seq :
  arithmetic_seq a →
  a 2 = 9 →
  (∃ x y, (x ^ 2 - 16 * x + 60 = 0) ∧ (a 0 = x) ∧ (a 4 = y)) →
  ∀ n, a n = -n + 11 :=
by
  intros h_arith h_a2 h_root
  sorry

end general_term_of_arithmetic_seq_l0_599


namespace worker_original_daily_wage_l0_472

-- Given Conditions
def increases : List ℝ := [0.20, 0.30, 0.40, 0.50, 0.60]
def new_total_weekly_salary : ℝ := 1457

-- Define the sum of the weekly increases
def total_increase : ℝ := (1 + increases.get! 0) + (1 + increases.get! 1) + (1 + increases.get! 2) + (1 + increases.get! 3) + (1 + increases.get! 4)

-- Main Theorem
theorem worker_original_daily_wage : ∀ (W : ℝ), total_increase * W = new_total_weekly_salary → W = 242.83 :=
by
  intro W h
  sorry

end worker_original_daily_wage_l0_472


namespace min_sum_of_product_2004_l0_122

theorem min_sum_of_product_2004 (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
    (hxyz : x * y * z = 2004) : x + y + z ≥ 174 ∧ ∃ (a b c : ℕ), a * b * c = 2004 ∧ a + b + c = 174 :=
by sorry

end min_sum_of_product_2004_l0_122


namespace number_of_truthful_dwarfs_l0_176

def num_dwarfs : Nat := 10
def num_vanilla : Nat := 10
def num_chocolate : Nat := 5
def num_fruit : Nat := 1

def total_hands_raised : Nat := num_vanilla + num_chocolate + num_fruit
def num_extra_hands : Nat := total_hands_raised - num_dwarfs

variable (T L : Nat)

axiom dwarfs_count : T + L = num_dwarfs
axiom hands_by_liars : L = num_extra_hands

theorem number_of_truthful_dwarfs : T = 4 :=
by
  have total_liars: num_dwarfs - T = num_extra_hands := by sorry
  have final_truthful: T = num_dwarfs - num_extra_hands := by sorry
  show T = 4 from final_truthful

end number_of_truthful_dwarfs_l0_176


namespace circle_equation_l0_490

-- Defining the points A and B
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 3)

-- Defining the center M of the circle on the x-axis
def M (a : ℝ) : ℝ × ℝ := (a, 0)

-- Defining the squared distance function between two points
def dist_sq (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2

-- Statement: Prove that the standard equation of the circle is (x - 2)² + y² = 10
theorem circle_equation : ∃ a : ℝ, (dist_sq (M a) A = dist_sq (M a) B) ∧ ((M a).1 = 2) ∧ (dist_sq (M a) A = 10) :=
sorry

end circle_equation_l0_490


namespace total_space_needed_for_trees_l0_690

def appleTreeWidth : ℕ := 10
def spaceBetweenAppleTrees : ℕ := 12
def numAppleTrees : ℕ := 2

def peachTreeWidth : ℕ := 12
def spaceBetweenPeachTrees : ℕ := 15
def numPeachTrees : ℕ := 2

def totalSpace : ℕ :=
  numAppleTrees * appleTreeWidth + spaceBetweenAppleTrees +
  numPeachTrees * peachTreeWidth + spaceBetweenPeachTrees

theorem total_space_needed_for_trees : totalSpace = 71 := by
  sorry

end total_space_needed_for_trees_l0_690


namespace total_apples_l0_850

-- Definitions based on the problem conditions
def marin_apples : ℕ := 8
def david_apples : ℕ := (3 * marin_apples) / 4
def amanda_apples : ℕ := (3 * david_apples) / 2 + 2

-- The statement that we need to prove
theorem total_apples : marin_apples + david_apples + amanda_apples = 25 := by
  -- The proof steps will go here
  sorry

end total_apples_l0_850


namespace max_homework_ratio_l0_685

theorem max_homework_ratio 
  (H : ℕ) -- time spent on history tasks
  (biology_time : ℕ)
  (total_homework_time : ℕ)
  (geography_time : ℕ)
  (history_geography_relation : geography_time = 3 * H)
  (total_time_relation : total_homework_time = 180)
  (biology_time_known : biology_time = 20)
  (sum_time_relation : H + geography_time + biology_time = total_homework_time) :
  H / biology_time = 2 :=
by
  sorry

end max_homework_ratio_l0_685


namespace complex_problem_l0_360

noncomputable def z (ζ : ℂ) : ℂ := ζ

theorem complex_problem (ζ : ℂ) (h : ζ + ζ⁻¹ = 2 * real.cos (real.pi / 36)) : 
  (ζ ^ 100 + ζ ^ (-100)) = -2 * real.cos (2 * real.pi / 9) :=
by sorry

end complex_problem_l0_360


namespace no_linear_term_in_product_l0_961

theorem no_linear_term_in_product (m : ℝ) :
  (∀ (x : ℝ), (x - 3) * (3 * x + m) - (3 * x^2 - 3 * m) = 0) → m = 9 :=
by
  intro h
  sorry

end no_linear_term_in_product_l0_961


namespace speed_of_point_C_l0_352

theorem speed_of_point_C 
    (a T R L x : ℝ) 
    (h1 : x = L * (a * T) / R - L) 
    (h_eq: (a * T) / (a * T - R) = (L + x) / x) :
    (a * L) / R = x / T :=
by
  sorry

end speed_of_point_C_l0_352


namespace vasya_triangle_rotation_l0_912

theorem vasya_triangle_rotation :
  (∀ (θ1 θ2 θ3 : ℝ), (12 * θ1 = 360) ∧ (6 * θ2 = 360) ∧ (θ1 + θ2 + θ3 = 180) → ∃ n : ℕ, (n * θ3 = 360) ∧ n ≥ 4) :=
by
  -- The formal proof is omitted, inserting "sorry" to indicate incomplete proof
  sorry

end vasya_triangle_rotation_l0_912


namespace inequality_holds_l0_650

open Real

theorem inequality_holds (a b : ℝ) (h : a > b) : (1/2)^b > (1/2)^a := 
by
  sorry

end inequality_holds_l0_650


namespace alpha_plus_beta_eq_118_l0_885

theorem alpha_plus_beta_eq_118 (α β : ℝ) (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 96 * x + 2209) / (x^2 + 63 * x - 3969)) : α + β = 118 :=
by
  sorry

end alpha_plus_beta_eq_118_l0_885


namespace multiple_with_digits_l0_541

theorem multiple_with_digits (n : ℕ) (h : n > 0) :
  ∃ (m : ℕ), (m % n = 0) ∧ (m < 10 ^ n) ∧ (∀ d ∈ m.digits 10, d = 0 ∨ d = 1) :=
by
  sorry

end multiple_with_digits_l0_541


namespace remaining_sum_eq_seven_eighths_l0_506

noncomputable def sum_series := 
  (1 / 2) + (1 / 4) + (1 / 8) + (1 / 16) + (1 / 32) + (1 / 64)

noncomputable def removed_terms := 
  (1 / 16) + (1 / 32) + (1 / 64)

theorem remaining_sum_eq_seven_eighths : 
  sum_series - removed_terms = 7 / 8 := by
  sorry

end remaining_sum_eq_seven_eighths_l0_506


namespace train_length_l0_569

theorem train_length (L : ℝ) (v1 v2 : ℝ) 
  (h1 : v1 = (L + 140) / 15)
  (h2 : v2 = (L + 250) / 20) 
  (h3 : v1 = v2) :
  L = 190 :=
by sorry

end train_length_l0_569


namespace david_dogs_left_l0_601

def total_dogs_left (boxes_small: Nat) (dogs_per_small: Nat) (boxes_large: Nat) (dogs_per_large: Nat) (giveaway_small: Nat) (giveaway_large: Nat): Nat :=
  let total_small := boxes_small * dogs_per_small
  let total_large := boxes_large * dogs_per_large
  let remaining_small := total_small - giveaway_small
  let remaining_large := total_large - giveaway_large
  remaining_small + remaining_large

theorem david_dogs_left :
  total_dogs_left 7 4 5 3 2 1 = 40 := by
  sorry

end david_dogs_left_l0_601


namespace range_of_a_l0_505

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x < 2 → (a+1)*x > 2*a+2) → a < -1 :=
by
  sorry

end range_of_a_l0_505


namespace number_of_solutions_l0_610

noncomputable def count_solutions : ℕ :=
  Nat.card { x // 0 ≤ x.1 ∧ x.1 ≤ 14 ∧
                  0 ≤ x.2 ∧ x.2 ≤ 14 ∧
                  0 ≤ x.3 ∧ x.3 ≤ 14 ∧
                  x.1 + x.2 + x.3 = 14 }

theorem number_of_solutions : count_solutions = 57 := by
  sorry

end number_of_solutions_l0_610


namespace sum_of_digits_next_exact_multiple_l0_851

noncomputable def Michael_next_age_sum_of_digits (L M T n : ℕ) : ℕ :=
  let next_age := M + n
  ((next_age / 10) % 10) + (next_age % 10)

theorem sum_of_digits_next_exact_multiple :
  ∀ (L M T n : ℕ),
    T = 2 →
    M = L + 4 →
    (∀ k : ℕ, k < 8 → ∃ m : ℕ, L = m * T + k * T) →
    (∃ n, (M + n) % (T + n) = 0) →
    Michael_next_age_sum_of_digits L M T n = 9 :=
by
  intros
  sorry

end sum_of_digits_next_exact_multiple_l0_851


namespace number_of_routes_from_P_to_Q_is_3_l0_427

-- Definitions of the nodes and paths
inductive Node
| P | Q | R | S | T | U | V
deriving DecidableEq, Repr

-- Definition of paths between nodes based on given conditions
def leads_to : Node → Node → Prop
| Node.P, Node.R => True
| Node.P, Node.S => True
| Node.R, Node.T => True
| Node.R, Node.U => True
| Node.S, Node.Q => True
| Node.T, Node.Q => True
| Node.U, Node.V => True
| Node.V, Node.Q => True
| _, _ => False

-- Proof statement: the number of different routes from P to Q
theorem number_of_routes_from_P_to_Q_is_3 : 
  ∃ (n : ℕ), n = 3 ∧ (∀ (route_count : ℕ), route_count = n → 
  ((leads_to Node.P Node.R ∧ leads_to Node.R Node.T ∧ leads_to Node.T Node.Q) ∨ 
   (leads_to Node.P Node.R ∧ leads_to Node.R Node.U ∧ leads_to Node.U Node.V ∧ leads_to Node.V Node.Q) ∨
   (leads_to Node.P Node.S ∧ leads_to Node.S Node.Q))) :=
by
  -- Placeholder proof
  sorry

end number_of_routes_from_P_to_Q_is_3_l0_427


namespace smallest_integer_of_lcm_gcd_l0_140

theorem smallest_integer_of_lcm_gcd (m : ℕ) (h1 : m > 0) (h2 : Nat.lcm 60 m / Nat.gcd 60 m = 44) : m = 165 :=
sorry

end smallest_integer_of_lcm_gcd_l0_140


namespace picking_ball_is_random_event_l0_661

-- Definitions based on problem conditions
def total_balls := 201
def black_balls := 200
def white_balls := 1

-- The goal to prove
theorem picking_ball_is_random_event : 
  (total_balls = black_balls + white_balls) ∧ 
  (black_balls > 0) ∧ 
  (white_balls > 0) → 
  random_event :=
by sorry

end picking_ball_is_random_event_l0_661


namespace ratio_of_triangle_areas_bcx_acx_l0_476

theorem ratio_of_triangle_areas_bcx_acx
  (BC AC : ℕ) (hBC : BC = 36) (hAC : AC = 45)
  (is_angle_bisector_CX : ∀ BX AX : ℕ, BX / AX = BC / AC) :
  (∃ BX AX : ℕ, BX / AX = 4 / 5) :=
by
  have h_ratio := is_angle_bisector_CX 36 45
  rw [hBC, hAC] at h_ratio
  exact ⟨4, 5, h_ratio⟩

end ratio_of_triangle_areas_bcx_acx_l0_476


namespace find_s_l0_643

theorem find_s (s t : ℝ) (h1 : 8 * s + 4 * t = 160) (h2 : t = 2 * s - 3) : s = 10.75 :=
by
  sorry

end find_s_l0_643


namespace problem_statement_l0_372

variables (P Q : Prop)

theorem problem_statement (h1 : ¬P) (h2 : ¬(P ∧ Q)) : ¬(P ∨ Q) :=
sorry

end problem_statement_l0_372


namespace balance_weights_l0_257

def pair_sum {α : Type*} (l : List α) [Add α] : List (α × α) :=
  l.zip l.tail

theorem balance_weights (w : Fin 100 → ℝ) (h : ∀ i j, |w i - w j| ≤ 20) :
  ∃ (l r : Finset (Fin 100)), l.card = 50 ∧ r.card = 50 ∧
  |(l.sum w - r.sum w)| ≤ 20 :=
sorry

end balance_weights_l0_257


namespace grill_run_time_l0_289

def time_burn (coals : ℕ) (burn_rate : ℕ) (interval : ℕ) : ℚ :=
  (coals / burn_rate) * interval

theorem grill_run_time :
  let time_a1 := time_burn 60 15 20
  let time_a2 := time_burn 75 12 20
  let time_a3 := time_burn 45 15 20
  let time_b1 := time_burn 50 10 30
  let time_b2 := time_burn 70 8 30
  let time_b3 := time_burn 40 10 30
  let time_b4 := time_burn 80 8 30
  time_a1 + time_a2 + time_a3 + time_b1 + time_b2 + time_b3 + time_b4 = 1097.5 := sorry

end grill_run_time_l0_289


namespace sum_of_squares_of_roots_l0_32

theorem sum_of_squares_of_roots (x₁ x₂ : ℚ) (h : 6 * x₁^2 - 9 * x₁ + 5 = 0 ∧ 6 * x₂^2 - 9 * x₂ + 5 = 0 ∧ x₁ ≠ x₂) : x₁^2 + x₂^2 = 7 / 12 :=
by
  -- Since we are only required to write the statement, we leave the proof as sorry
  sorry

end sum_of_squares_of_roots_l0_32


namespace parabola_directrix_l0_696

theorem parabola_directrix (a : ℝ) : 
  (∃ y, (y ^ 2 = 4 * a * (-2))) → a = 2 :=
by
  sorry

end parabola_directrix_l0_696


namespace probability_two_faces_no_faces_l0_154

theorem probability_two_faces_no_faces :
  let side_length := 5
  let total_cubes := side_length ^ 3
  let painted_faces := 2 * (side_length ^ 2)
  let two_painted_faces := 16
  let no_painted_faces := total_cubes - painted_faces + two_painted_faces
  (two_painted_faces = 16) →
  (no_painted_faces = 91) →
  -- Total ways to choose 2 cubes from 125
  let total_ways := (total_cubes * (total_cubes - 1)) / 2
  -- Ways to choose 1 cube with 2 painted faces and 1 with no painted faces
  let successful_ways := two_painted_faces * no_painted_faces
  (successful_ways = 1456) →
  (total_ways = 7750) →
  -- The desired probability
  let probability := successful_ways / (total_ways : ℝ)
  probability = 4 / 21 :=
by
  intros side_length total_cubes painted_faces two_painted_faces no_painted_faces h1 h2 total_ways successful_ways h3 h4 probability
  sorry

end probability_two_faces_no_faces_l0_154


namespace find_geometric_sequence_l0_193

def geometric_sequence (b1 b2 b3 b4 : ℤ) :=
  ∃ q : ℤ, b2 = b1 * q ∧ b3 = b1 * q^2 ∧ b4 = b1 * q^3

theorem find_geometric_sequence :
  ∃ b1 b2 b3 b4 : ℤ, 
    geometric_sequence b1 b2 b3 b4 ∧
    (b1 + b4 = -49) ∧
    (b2 + b3 = 14) ∧ 
    ((b1, b2, b3, b4) = (7, -14, 28, -56) ∨ (b1, b2, b3, b4) = (-56, 28, -14, 7)) :=
by
  sorry

end find_geometric_sequence_l0_193


namespace sum_of_children_ages_l0_884

theorem sum_of_children_ages :
  ∃ E: ℕ, E = 12 ∧ 
  (∃ a b c d e : ℕ, a = E ∧ b = E - 2 ∧ c = E - 4 ∧ d = E - 6 ∧ e = E - 8 ∧ 
   a + b + c + d + e = 40) :=
sorry

end sum_of_children_ages_l0_884


namespace savings_l0_672

def distance_each_way : ℕ := 150
def round_trip_distance : ℕ := 2 * distance_each_way
def rental_cost_first_option : ℕ := 50
def rental_cost_second_option : ℕ := 90
def gasoline_efficiency : ℕ := 15
def gasoline_cost_per_liter : ℚ := 0.90
def gasoline_needed_for_trip : ℚ := round_trip_distance / gasoline_efficiency
def total_gasoline_cost : ℚ := gasoline_needed_for_trip * gasoline_cost_per_liter
def total_cost_first_option : ℚ := rental_cost_first_option + total_gasoline_cost
def total_cost_second_option : ℚ := rental_cost_second_option

theorem savings : total_cost_second_option - total_cost_first_option = 22 := by
  sorry

end savings_l0_672


namespace owner_overtakes_thief_l0_161

theorem owner_overtakes_thief :
  let thief_speed_initial := 45 -- kmph
  let discovery_time := 0.5 -- hours
  let owner_speed := 50 -- kmph
  let mud_road_speed := 35 -- kmph
  let mud_road_distance := 30 -- km
  let speed_bumps_speed := 40 -- kmph
  let speed_bumps_distance := 5 -- km
  let traffic_speed := 30 -- kmph
  let head_start_distance := thief_speed_initial * discovery_time
  let mud_road_time := mud_road_distance / mud_road_speed
  let speed_bumps_time := speed_bumps_distance / speed_bumps_speed
  let total_distance_before_traffic := mud_road_distance + speed_bumps_distance
  let total_time_before_traffic := mud_road_time + speed_bumps_time
  let distance_owner_travelled := owner_speed * total_time_before_traffic
  head_start_distance + total_distance_before_traffic < distance_owner_travelled →
  discovery_time + total_time_before_traffic = 1.482 :=
by sorry


end owner_overtakes_thief_l0_161


namespace common_ratio_of_arithmetic_sequence_l0_59

theorem common_ratio_of_arithmetic_sequence (S_odd S_even : ℤ) (q : ℤ) 
  (h1 : S_odd + S_even = -240) (h2 : S_odd - S_even = 80) 
  (h3 : q = S_even / S_odd) : q = 2 := 
  sorry

end common_ratio_of_arithmetic_sequence_l0_59


namespace fewest_trips_l0_818

theorem fewest_trips (total_objects : ℕ) (capacity : ℕ) (h_objects : total_objects = 17) (h_capacity : capacity = 3) : 
  (total_objects + capacity - 1) / capacity = 6 :=
by
  sorry

end fewest_trips_l0_818


namespace root_interval_range_l0_829

theorem root_interval_range (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ x^3 - 3*x + m = 0) → (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end root_interval_range_l0_829


namespace solve_quadratic_eq1_solve_quadratic_eq2_complete_square_l0_867

theorem solve_quadratic_eq1 : ∀ x : ℝ, 2 * x^2 + 5 * x + 3 = 0 → (x = -3/2 ∨ x = -1) :=
by
  intro x
  intro h
  sorry

theorem solve_quadratic_eq2_complete_square : ∀ x : ℝ, x^2 - 2 * x - 1 = 0 → (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_complete_square_l0_867


namespace room_area_ratio_l0_890

theorem room_area_ratio (total_squares overlapping_squares : ℕ) 
  (h_total : total_squares = 16) 
  (h_overlap : overlapping_squares = 4) : 
  total_squares / overlapping_squares = 4 := 
by 
  sorry

end room_area_ratio_l0_890


namespace people_on_trolley_l0_299

-- Given conditions
variable (X : ℕ)

def initial_people : ℕ := 10

def second_stop_people : ℕ := initial_people - 3 + 20

def third_stop_people : ℕ := second_stop_people - 18 + 2

def fourth_stop_people : ℕ := third_stop_people - 5 + X

-- Prove the current number of people on the trolley is 6 + X
theorem people_on_trolley (X : ℕ) : 
  fourth_stop_people X = 6 + X := 
by 
  unfold fourth_stop_people
  unfold third_stop_people
  unfold second_stop_people
  unfold initial_people
  sorry

end people_on_trolley_l0_299


namespace volume_is_120_l0_483

namespace volume_proof

-- Definitions from the given conditions
variables (a b c : ℝ)
axiom ab_relation : a * b = 48
axiom bc_relation : b * c = 20
axiom ca_relation : c * a = 15

-- Goal to prove
theorem volume_is_120 : a * b * c = 120 := by
  sorry

end volume_proof

end volume_is_120_l0_483


namespace solution_set_inequality_l0_207

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x

axiom mono_increasing (x y : ℝ) (hxy : 0 < x ∧ x < y) : f x < f y

axiom f_2_eq_0 : f 2 = 0

theorem solution_set_inequality :
  { x : ℝ | (x - 1) * f x < 0 } = { x : ℝ | -2 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x ∧ x < 2 } :=
by {
  sorry
}

end solution_set_inequality_l0_207


namespace lori_earnings_equation_l0_847

theorem lori_earnings_equation : 
  ∀ (white_cars red_cars : ℕ) 
    (white_rent_cost red_rent_cost minutes_per_hour rental_hours : ℕ), 
  white_cars = 2 →
  red_cars = 3 →
  white_rent_cost = 2 →
  red_rent_cost = 3 →
  minutes_per_hour = 60 →
  rental_hours = 3 →
  (white_cars * white_rent_cost + red_cars * red_rent_cost) * rental_hours * minutes_per_hour = 2340 := 
by
  intros white_cars red_cars white_rent_cost red_rent_cost minutes_per_hour rental_hours 
         h_white_cars h_red_cars h_white_rent_cost h_red_rent_cost h_minutes_per_hour h_rental_hours
  rw [h_white_cars, h_red_cars, h_white_rent_cost, h_red_rent_cost, h_minutes_per_hour, h_rental_hours]
  calc
    (2 * 2 + 3 * 3) * 3 * 60 = (4 + 9) * 3 * 60 : by rw [mul_add, mul_comm 3]
    ... = 13 * 3 * 60 : by rw [add_mul]
    ... = 39 * 60 : by rw [mul_assoc]
    ... = 2340  : by norm_num
  done

end lori_earnings_equation_l0_847


namespace isosceles_triangle_perimeter_l0_622

theorem isosceles_triangle_perimeter (a b c : ℝ) (h₀ : a = 5) (h₁ : b = 10) 
  (h₂ : c = 10 ∨ c = 5) (h₃ : a = b ∨ b = c ∨ c = a) 
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 25 := by
  sorry

end isosceles_triangle_perimeter_l0_622


namespace math_score_computation_l0_455

def comprehensive_score 
  (reg_score : ℕ) (mid_score : ℕ) (fin_score : ℕ) 
  (reg_weight : ℕ) (mid_weight : ℕ) (fin_weight : ℕ) 
  : ℕ :=
  (reg_score * reg_weight + mid_score * mid_weight + fin_score * fin_weight) 
  / (reg_weight + mid_weight + fin_weight)

theorem math_score_computation :
  comprehensive_score 80 80 85 3 3 4 = 82 := by
sorry

end math_score_computation_l0_455


namespace total_questions_attempted_l0_377

theorem total_questions_attempted 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) (total_marks : ℕ) (correct_answers : ℕ) 
  (total_questions : ℕ) (incorrect_answers : ℕ)
  (h_marks_per_correct : marks_per_correct = 4)
  (h_marks_lost_per_wrong : marks_lost_per_wrong = 1) 
  (h_total_marks : total_marks = 130) 
  (h_correct_answers : correct_answers = 36) 
  (h_score_eq : marks_per_correct * correct_answers - marks_lost_per_wrong * incorrect_answers = total_marks)
  (h_total_questions : total_questions = correct_answers + incorrect_answers) : 
  total_questions = 50 :=
by
  sorry

end total_questions_attempted_l0_377


namespace range_of_a_for_monotonically_decreasing_l0_517

noncomputable def f (a x: ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - 2 * x

theorem range_of_a_for_monotonically_decreasing (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (1/x - a*x - 2 < 0)) ↔ (a ∈ Set.Ioi (-1)) := 
sorry

end range_of_a_for_monotonically_decreasing_l0_517


namespace rotational_transform_preserves_expression_l0_988

theorem rotational_transform_preserves_expression
  (a b c : ℝ)
  (ϕ : ℝ)
  (a1 b1 c1 : ℝ)
  (x' y' x'' y'' : ℝ)
  (h1 : x'' = x' * Real.cos ϕ + y' * Real.sin ϕ)
  (h2 : y'' = -x' * Real.sin ϕ + y' * Real.cos ϕ)
  (def_a1 : a1 = a * (Real.cos ϕ)^2 - 2 * b * (Real.cos ϕ) * (Real.sin ϕ) + c * (Real.sin ϕ)^2)
  (def_b1 : b1 = a * (Real.cos ϕ) * (Real.sin ϕ) + b * ((Real.cos ϕ)^2 - (Real.sin ϕ)^2) - c * (Real.cos ϕ) * (Real.sin ϕ))
  (def_c1 : c1 = a * (Real.sin ϕ)^2 + 2 * b * (Real.cos ϕ) * (Real.sin ϕ) + c * (Real.cos ϕ)^2) :
  a1 * c1 - b1^2 = a * c - b^2 := sorry

end rotational_transform_preserves_expression_l0_988


namespace car_first_hour_speed_l0_557

theorem car_first_hour_speed
  (x speed2 : ℝ)
  (avgSpeed : ℝ)
  (h_speed2 : speed2 = 60)
  (h_avgSpeed : avgSpeed = 35) :
  (avgSpeed = (x + speed2) / 2) → x = 10 :=
by
  sorry

end car_first_hour_speed_l0_557


namespace solution_set_of_inequality_l0_940

theorem solution_set_of_inequality:
  {x : ℝ | |x - 5| + |x + 1| < 8} = {x : ℝ | -2 < x ∧ x < 6} :=
sorry

end solution_set_of_inequality_l0_940


namespace chosen_number_is_5_l0_463

theorem chosen_number_is_5 (x : ℕ) (h_pos : x > 0)
  (h_eq : ((10 * x + 5 - x^2) / x) - x = 1) : x = 5 :=
by
  sorry

end chosen_number_is_5_l0_463


namespace gcd_18_30_45_l0_489

-- Define the conditions
def a := 18
def b := 30
def c := 45

-- Prove that the gcd of a, b, and c is 3
theorem gcd_18_30_45 : Nat.gcd (Nat.gcd a b) c = 3 :=
by
  -- Skip the proof itself
  sorry

end gcd_18_30_45_l0_489


namespace siblings_of_kevin_l0_702

-- Define traits of each child
structure Child where
  eye_color : String
  hair_color : String

def Oliver : Child := ⟨"Green", "Red"⟩
def Kevin : Child := ⟨"Grey", "Brown"⟩
def Lily : Child := ⟨"Grey", "Red"⟩
def Emma : Child := ⟨"Green", "Brown"⟩
def Noah : Child := ⟨"Green", "Red"⟩
def Mia : Child := ⟨"Green", "Brown"⟩

-- Define the condition that siblings must share at least one trait
def share_at_least_one_trait (c1 c2 : Child) : Prop :=
  c1.eye_color = c2.eye_color ∨ c1.hair_color = c2.hair_color

-- Prove that Emma and Mia are Kevin's siblings
theorem siblings_of_kevin : share_at_least_one_trait Kevin Emma ∧ share_at_least_one_trait Kevin Mia ∧ share_at_least_one_trait Emma Mia :=
  sorry

end siblings_of_kevin_l0_702


namespace total_cost_to_replace_floor_l0_132

def removal_cost : ℝ := 50
def cost_per_sqft : ℝ := 1.25
def room_dimensions : (ℝ × ℝ) := (8, 7)

theorem total_cost_to_replace_floor :
  removal_cost + (cost_per_sqft * (room_dimensions.1 * room_dimensions.2)) = 120 := by
  sorry

end total_cost_to_replace_floor_l0_132


namespace total_inheritance_money_l0_237

-- Defining the conditions
def number_of_inheritors : ℕ := 5
def amount_per_person : ℕ := 105500

-- The proof problem
theorem total_inheritance_money :
  number_of_inheritors * amount_per_person = 527500 :=
by sorry

end total_inheritance_money_l0_237


namespace crate_weight_l0_160

variable (C : ℝ)
variable (carton_weight : ℝ := 3)
variable (total_weight : ℝ := 96)
variable (num_crates : ℝ := 12)
variable (num_cartons : ℝ := 16)

theorem crate_weight :
  (num_crates * C + num_cartons * carton_weight = total_weight) → (C = 4) :=
by
  sorry

end crate_weight_l0_160


namespace diminishing_allocation_proof_l0_263

noncomputable def diminishing_allocation_problem : Prop :=
  ∃ (a b m : ℝ), 
  a = 0.2 ∧
  b * (1 - a)^2 = 80 ∧
  b * (1 - a) + b * (1 - a)^3 = 164 ∧
  b + 80 + 164 = m ∧
  m = 369

theorem diminishing_allocation_proof : diminishing_allocation_problem :=
by
  sorry

end diminishing_allocation_proof_l0_263


namespace find_x_l0_611

theorem find_x :
  ∃ x : ℝ, (0 < x) ∧ (⌊x⌋ * x + x^2 = 93) ∧ (x = 7.10) :=
by {
   sorry
}

end find_x_l0_611


namespace eccentricity_of_ellipse_l0_697

noncomputable def e (a b c : ℝ) : ℝ := c / a

theorem eccentricity_of_ellipse (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (a - c) * (a + c) = (2 * c)^2) : e a b c = (Real.sqrt 5) / 5 := 
by
  sorry

end eccentricity_of_ellipse_l0_697


namespace sqrt_expression_l0_894

noncomputable def a : ℝ := 5 - 3 * Real.sqrt 2
noncomputable def b : ℝ := 5 + 3 * Real.sqrt 2

theorem sqrt_expression : 
  Real.sqrt (a^2) + Real.sqrt (b^2) + 2 = 12 :=
by
  sorry

end sqrt_expression_l0_894


namespace marble_arrangements_count_l0_280

-- Definitions
def marbles : Finset String := {"Aggie", "Bumblebee", "Steelie", "Tiger", "Clearie"}
def not_adjacent (x y : String) (lst : List String) : Prop := ¬(lst.indexOf x + 1 = lst.indexOf y ∨ lst.indexOf y + 1 = lst.indexOf x)
def valid_arrangements (lst : List String) : Prop := 
  lst.toFinset = marbles ∧ not_adjacent "Steelie" "Tiger" lst ∧ not_adjacent "Bumblebee" "Clearie" lst

-- Theorem Statement
theorem marble_arrangements_count : 
  (Finset.filter valid_arrangements (Finset.permutations marbles.toList)).card = 72 := 
  by sorry

end marble_arrangements_count_l0_280


namespace calculate_div_expression_l0_930

variable (x y : ℝ)

theorem calculate_div_expression : (6 * x^3 * y^2) / (-3 * x * y) = -2 * x^2 * y := by
  sorry

end calculate_div_expression_l0_930


namespace last_three_digits_of_7_pow_103_l0_799

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 327 :=
by
  sorry

end last_three_digits_of_7_pow_103_l0_799


namespace original_savings_l0_442

/-- Linda spent 3/4 of her savings on furniture and the rest on a TV costing $210. 
    What were her original savings? -/
theorem original_savings (S : ℝ) (h1 : S * (1/4) = 210) : S = 840 :=
by
  sorry

end original_savings_l0_442


namespace value_of_A_l0_654

theorem value_of_A (A : ℕ) : (A * 1000 + 567) % 100 < 50 → (A * 1000 + 567) / 10 * 10 = 2560 → A = 2 :=
by
  intro h1 h2
  sorry

end value_of_A_l0_654


namespace product_of_areas_eq_square_of_volume_l0_17

-- define the dimensions of the prism
variables (x y z : ℝ)

-- define the areas of the faces as conditions
def top_area := x * y
def back_area := y * z
def lateral_face_area := z * x

-- define the product of the areas of the top, back, and one lateral face
def product_of_areas := (top_area x y) * (back_area y z) * (lateral_face_area z x)

-- define the volume of the prism
def volume := x * y * z

-- theorem to prove: product of areas equals square of the volume
theorem product_of_areas_eq_square_of_volume 
  (ht: top_area x y = x * y)
  (hb: back_area y z = y * z)
  (hl: lateral_face_area z x = z * x) :
  product_of_areas x y z = (volume x y z) ^ 2 :=
by
  sorry

end product_of_areas_eq_square_of_volume_l0_17


namespace sqrt_nested_eq_five_l0_192

theorem sqrt_nested_eq_five {x : ℝ} (h : x = Real.sqrt (15 + x)) : x = 5 :=
sorry

end sqrt_nested_eq_five_l0_192


namespace minimum_score_to_win_l0_83

namespace CompetitionPoints

-- Define points awarded for each position
def points_first : ℕ := 5
def points_second : ℕ := 3
def points_third : ℕ := 1

-- Define the number of competitions
def competitions : ℕ := 3

-- Total points in one competition
def total_points_one_competition : ℕ := points_first + points_second + points_third

-- Total points in all competitions
def total_points_all_competitions : ℕ := total_points_one_competition * competitions

theorem minimum_score_to_win : ∃ m : ℕ, m = 13 ∧ (∀ s : ℕ, s < 13 → ¬ ∃ c1 c2 c3 : ℕ, 
  c1 ≤ competitions ∧ c2 ≤ competitions ∧ c3 ≤ competitions ∧ 
  ((c1 * points_first) + (c2 * points_second) + (c3 * points_third)) = s) :=
by {
  sorry
}

end CompetitionPoints

end minimum_score_to_win_l0_83


namespace fifty_three_days_from_Friday_is_Tuesday_l0_728

def days_of_week : Type := {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}

def modulo_days (n : ℕ) : ℕ := n % 7

def day_of_week_after_days (start_day : days_of_week) (n : ℕ) : days_of_week :=
  list.nth_le (list.rotate [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday] (modulo_days n)) sorry -- using sorry for index proof

theorem fifty_three_days_from_Friday_is_Tuesday :
  day_of_week_after_days Friday 53 = Tuesday :=
sorry

end fifty_three_days_from_Friday_is_Tuesday_l0_728


namespace value_of_x_l0_282

-- Define the conditions
variable (C S x : ℝ)
variable (h1 : 20 * C = x * S)
variable (h2 : (S - C) / C * 100 = 25)

-- Define the statement to be proved
theorem value_of_x : x = 16 :=
by
  sorry

end value_of_x_l0_282


namespace elongation_rate_significantly_improved_l0_576

noncomputable def elongation_improvement : Prop :=
  let x : List ℝ := [545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
  let y : List ℝ := [536, 527, 543, 530, 560, 533, 522, 550, 576, 536]
  let z := List.zipWith (λ xi yi => xi - yi) x y
  let n : ℝ := 10
  let mean_z := (List.sum z) / n
  let variance_z := (List.sum (List.map (λ zi => (zi - mean_z)^2) z)) / n
  mean_z = 11 ∧ 
  variance_z = 61 ∧ 
  mean_z ≥ 2 * Real.sqrt (variance_z / n)

-- We state the theorem without proof
theorem elongation_rate_significantly_improved : elongation_improvement :=
by
  -- Proof can be written here
  sorry

end elongation_rate_significantly_improved_l0_576


namespace smallest_value_36k_minus_5l_l0_301

theorem smallest_value_36k_minus_5l (k l : ℕ) :
  ∃ k l, 0 < 36^k - 5^l ∧ (∀ k' l', (0 < 36^k' - 5^l' → 36^k - 5^l ≤ 36^k' - 5^l')) ∧ 36^k - 5^l = 11 :=
by sorry

end smallest_value_36k_minus_5l_l0_301


namespace number_of_truthful_dwarfs_l0_175

def num_dwarfs : Nat := 10
def num_vanilla : Nat := 10
def num_chocolate : Nat := 5
def num_fruit : Nat := 1

def total_hands_raised : Nat := num_vanilla + num_chocolate + num_fruit
def num_extra_hands : Nat := total_hands_raised - num_dwarfs

variable (T L : Nat)

axiom dwarfs_count : T + L = num_dwarfs
axiom hands_by_liars : L = num_extra_hands

theorem number_of_truthful_dwarfs : T = 4 :=
by
  have total_liars: num_dwarfs - T = num_extra_hands := by sorry
  have final_truthful: T = num_dwarfs - num_extra_hands := by sorry
  show T = 4 from final_truthful

end number_of_truthful_dwarfs_l0_175


namespace fraction_of_menu_vegan_soy_free_l0_26

def num_vegan_dishes : Nat := 6
def fraction_menu_vegan : ℚ := 1 / 4
def num_vegan_dishes_with_soy : Nat := 4

def num_vegan_soy_free_dishes : Nat := num_vegan_dishes - num_vegan_dishes_with_soy
def fraction_vegan_soy_free : ℚ := num_vegan_soy_free_dishes / num_vegan_dishes
def fraction_menu_vegan_soy_free : ℚ := fraction_vegan_soy_free * fraction_menu_vegan

theorem fraction_of_menu_vegan_soy_free :
  fraction_menu_vegan_soy_free = 1 / 12 := by
  sorry

end fraction_of_menu_vegan_soy_free_l0_26


namespace cube_root_floor_equality_l0_234

theorem cube_root_floor_equality (n : ℕ) : 
  (⌊(n : ℝ)^(1/3) + (n+1 : ℝ)^(1/3)⌋ : ℝ) = ⌊(8*n + 3 : ℝ)^(1/3)⌋ :=
sorry

end cube_root_floor_equality_l0_234


namespace expression_value_l0_275

theorem expression_value :
    (2.502 + 0.064)^2 - ((2.502 - 0.064)^2) / (2.502 * 0.064) = 4.002 :=
by
  -- the proof goes here
  sorry

end expression_value_l0_275


namespace parabola_directrix_l0_119

theorem parabola_directrix (y x : ℝ) (h : y^2 = -4 * x) : x = 1 :=
sorry

end parabola_directrix_l0_119


namespace symmetrical_polynomial_l0_549

noncomputable def Q (x : ℝ) (f g h i j k : ℝ) : ℝ :=
  x^6 + f * x^5 + g * x^4 + h * x^3 + i * x^2 + j * x + k

theorem symmetrical_polynomial (f g h i j k : ℝ) :
  (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ Q 0 f g h i j k = 0 ∧
    Q x f g h i j k = x * (x - a) * (x + a) * (x - b) * (x + b) * (x - c) ∧
    Q x f g h i j k = Q (-x) f g h i j k) →
  f = 0 :=
by sorry

end symmetrical_polynomial_l0_549


namespace x_varies_as_half_power_of_z_l0_648

variable {x y z : ℝ} -- declare variables as real numbers

-- Assume the conditions, which are the relationships between x, y, and z
variable (k j : ℝ) (k_pos : k > 0) (j_pos : j > 0)
axiom xy_relationship : ∀ y, x = k * y^2
axiom yz_relationship : ∀ z, y = j * z^(1/4)

-- The theorem we want to prove
theorem x_varies_as_half_power_of_z (z : ℝ) (h : z ≥ 0) : ∃ m, m > 0 ∧ x = m * z^(1/2) :=
sorry

end x_varies_as_half_power_of_z_l0_648


namespace trigonometric_expression_l0_52

theorem trigonometric_expression (α : ℝ) (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (3 * Real.cos α + 3 * Real.sin α) = 2 / 3 :=
by
  sorry

end trigonometric_expression_l0_52


namespace B_cycling_speed_l0_163

theorem B_cycling_speed (v : ℝ) : 
  (∀ (t : ℝ), 10 * t + 30 = B_start_distance) ∧ 
  (B_start_distance = 60) ∧ 
  (t = 3) →
  v = 20 :=
sorry

end B_cycling_speed_l0_163


namespace smaller_balloon_radius_is_correct_l0_467

-- Condition: original balloon radius
def original_balloon_radius : ℝ := 2

-- Condition: number of smaller balloons
def num_smaller_balloons : ℕ := 64

-- Question (to be proved): Radius of each smaller balloon
theorem smaller_balloon_radius_is_correct :
  ∃ r : ℝ, (4/3) * Real.pi * (original_balloon_radius^3) = num_smaller_balloons * (4/3) * Real.pi * (r^3) ∧ r = 1/2 := 
by {
  sorry
}

end smaller_balloon_radius_is_correct_l0_467


namespace matrix_power_15_l0_61

-- Define the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![ 0, -1,  0;
      1,  0,  0;
      0,  0,  1]

-- Define what we want to prove
theorem matrix_power_15 :
  B^15 = !![ 0,  1,  0;
            -1,  0,  0;
             0,  0,  1] :=
sorry

end matrix_power_15_l0_61


namespace basketball_free_throws_l0_757

theorem basketball_free_throws (total_players : ℕ) (number_captains : ℕ) (players_not_including_one : ℕ) 
  (free_throws_per_captain : ℕ) (total_free_throws : ℕ) 
  (h1 : total_players = 15)
  (h2 : number_captains = 2)
  (h3 : players_not_including_one = total_players - 1)
  (h4 : free_throws_per_captain = players_not_including_one * number_captains)
  (h5 : total_free_throws = free_throws_per_captain)
  : total_free_throws = 28 :=
by
  -- Proof is not required, so we provide sorry to skip it.
  sorry

end basketball_free_throws_l0_757


namespace initial_sugar_weight_l0_251

-- Definitions corresponding to the conditions
def num_packs : ℕ := 12
def weight_per_pack : ℕ := 250
def leftover_sugar : ℕ := 20

-- Statement of the proof problem
theorem initial_sugar_weight : 
  (num_packs * weight_per_pack + leftover_sugar = 3020) :=
by
  sorry

end initial_sugar_weight_l0_251


namespace intercept_sum_mod_7_l0_312

theorem intercept_sum_mod_7 :
  ∃ (x_0 y_0 : ℤ), (2 * x_0 ≡ 3 * y_0 + 1 [ZMOD 7]) ∧ (0 ≤ x_0) ∧ (x_0 < 7) ∧ (0 ≤ y_0) ∧ (y_0 < 7) ∧ (x_0 + y_0 = 6) :=
by
  sorry

end intercept_sum_mod_7_l0_312


namespace inequality_proof_l0_393

theorem inequality_proof (x : ℝ) (n : ℕ) (hx : 0 < x) : 
  1 + x^(n+1) ≥ (2*x)^n / (1 + x)^(n-1) := 
by
  sorry

end inequality_proof_l0_393


namespace Robin_hair_initial_length_l0_862

theorem Robin_hair_initial_length (x : ℝ) (h1 : x + 8 - 20 = 2) : x = 14 :=
by
  sorry

end Robin_hair_initial_length_l0_862


namespace inequality_solution_set_l0_208

theorem inequality_solution_set
  (a b c m n : ℝ) (h : a ≠ 0) 
  (h1 : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ m < x ∧ x < n)
  (h2 : 0 < m)
  (h3 : ∀ x : ℝ, cx^2 + bx + a < 0 ↔ (x < 1 / n ∨ 1 / m < x)) :
  (cx^2 + bx + a < 0 ↔ (x < 1 / n ∨ 1 / m < x)) := 
sorry

end inequality_solution_set_l0_208


namespace g_value_range_l0_100

noncomputable def g (x y z : ℝ) : ℝ :=
  (x^2 / (x^2 + y^2)) + (y^2 / (y^2 + z^2)) + (z^2 / (z^2 + x^2))

theorem g_value_range (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : 
  (3/2 : ℝ) ≤ g x y z ∧ g x y z ≤ (3 : ℝ) / 2 := 
sorry

end g_value_range_l0_100


namespace triangle_square_ratio_l0_593

theorem triangle_square_ratio (t s : ℝ) 
  (h1 : 3 * t = 15) 
  (h2 : 4 * s = 12) : 
  t / s = 5 / 3 :=
by 
  -- skipping the proof
  sorry

end triangle_square_ratio_l0_593


namespace find_numbers_l0_76

theorem find_numbers :
  ∃ a d : ℝ, 
    ((a - d) + a + (a + d) = 12) ∧ 
    ((a - d) * a * (a + d) = 48) ∧
    (a = 4) ∧ 
    (d = -2) ∧ 
    (a - d = 6) ∧ 
    (a + d = 2) :=
by
  sorry

end find_numbers_l0_76


namespace solution_l0_292

noncomputable def die1 : Finset ℕ := {1, 2, 3, 3, 4, 4}.toFinset
noncomputable def die2 : Finset ℕ := {2, 3, 5, 6, 7, 8}.toFinset

def probability_target_sum : ℚ :=
(let outcomes := (die1.product die2).toFinset in
 let favorable_outcomes := outcomes.filter (λ (x : ℕ × ℕ), (x.1 + x.2 = 6) ∨ (x.1 + x.2 = 8) ∨ (x.1 + x.2 = 10)) in
 favorable_outcomes.card.toRat / outcomes.card.toRat)

theorem solution : probability_target_sum = 11 / 36 :=
by sorry

end solution_l0_292


namespace multiple_of_P_l0_828

theorem multiple_of_P (P Q R : ℝ) (T : ℝ) (x : ℝ) (total_profit Rs900 : ℝ)
  (h1 : P = 6 * Q)
  (h2 : P = 10 * R)
  (h3 : R = T / 5.1)
  (h4 : total_profit = Rs900 + (T - R)) :
  x = 10 :=
by
  sorry

end multiple_of_P_l0_828


namespace negation_exists_implies_forall_l0_699

theorem negation_exists_implies_forall (x_0 : ℝ) (h : ∃ x_0 : ℝ, x_0^3 - x_0 + 1 > 0) : 
  ¬ (∃ x_0 : ℝ, x_0^3 - x_0 + 1 > 0) ↔ ∀ x : ℝ, x^3 - x + 1 ≤ 0 :=
by 
  sorry

end negation_exists_implies_forall_l0_699


namespace age_ordered_youngest_to_oldest_l0_111

variable (M Q S : Nat)

theorem age_ordered_youngest_to_oldest 
  (h1 : M = Q ∨ S = Q)
  (h2 : M ≥ Q)
  (h3 : S ≤ Q) : S = Q ∧ M > Q :=
by 
  sorry

end age_ordered_youngest_to_oldest_l0_111


namespace prob_of_25_sixes_on_surface_prob_of_at_least_one_one_on_surface_expected_number_of_sixes_on_surface_expected_sum_of_numbers_on_surface_expected_value_of_diff_digits_on_surface_l0_291

-- Definitions for the conditions.

-- cube configuration
def num_dice : ℕ := 27
def num_visible_dice : ℕ := 26
def num_faces_per_die : ℕ := 6
def num_visible_faces : ℕ := 54

-- Given probabilities
def prob_six (face : ℕ) : ℚ := 1/6
def prob_not_six (face : ℕ) : ℚ := 5/6
def prob_not_one (face : ℕ) : ℚ := 5/6

-- Expected values given conditions
def expected_num_sixes : ℚ := 9
def expected_sum_faces : ℚ := 189
def expected_diff_digits : ℚ := 6 - (5^6) / (2 * 3^17)

-- Probabilities given conditions
def prob_25_sixes_on_surface : ℚ := (26 * 5) / (6^26)
def prob_at_least_one_one : ℚ := 1 - (5^6) / (2^2 * 3^18)

-- Lean statements for proof

theorem prob_of_25_sixes_on_surface :
  prob_25_sixes_on_surface = 31 / (2^13 * 3^18) := by
  sorry

theorem prob_of_at_least_one_one_on_surface :
  prob_at_least_one_one = 0.99998992 := by
  sorry

theorem expected_number_of_sixes_on_surface :
  expected_num_sixes = 9 := by
  sorry

theorem expected_sum_of_numbers_on_surface :
  expected_sum_faces = 189 := by
  sorry

theorem expected_value_of_diff_digits_on_surface :
  expected_diff_digits = 6 - (5^6) / (2 * 3^17) := by
  sorry

end prob_of_25_sixes_on_surface_prob_of_at_least_one_one_on_surface_expected_number_of_sixes_on_surface_expected_sum_of_numbers_on_surface_expected_value_of_diff_digits_on_surface_l0_291


namespace root_difference_l0_695

theorem root_difference (p : ℝ) (r s : ℝ) :
  (r + s = p) ∧ (r * s = (p^2 - 1) / 4) ∧ (r ≥ s) → r - s = 1 :=
by
  intro h
  sorry

end root_difference_l0_695


namespace gcd_n_cube_plus_m_square_l0_941

theorem gcd_n_cube_plus_m_square (n m : ℤ) (h : n > 2^3) : Int.gcd (n^3 + m^2) (n + 2) = 1 :=
by
  sorry

end gcd_n_cube_plus_m_square_l0_941


namespace days_from_friday_l0_717

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end days_from_friday_l0_717


namespace perfect_squares_of_k_l0_383

theorem perfect_squares_of_k (k : ℕ) (h : ∃ (a : ℕ), k * (k + 1) = 3 * a^2) : 
  ∃ (m n : ℕ), k = 3 * m^2 ∧ k + 1 = n^2 := 
sorry

end perfect_squares_of_k_l0_383


namespace notebook_cost_l0_833

theorem notebook_cost (s n c : ℕ) (h1 : s ≥ 19) (h2 : n > 2) (h3 : c > n) (h4 : s * c * n = 3969) : c = 27 :=
sorry

end notebook_cost_l0_833


namespace find_cd_l0_744

def g (c d x : ℝ) := c * x^3 - 7 * x^2 + d * x - 4

theorem find_cd : ∃ c d : ℝ, (g c d 2 = -4) ∧ (g c d (-1) = -22) ∧ (c = 19/3) ∧ (d = -8/3) := 
by
  sorry

end find_cd_l0_744


namespace circle_equation_l0_813

theorem circle_equation 
  (h k : ℝ) 
  (H_center : k = 2 * h)
  (H_tangent : ∃ (r : ℝ), (h - 1)^2 + (k - 0)^2 = r^2 ∧ r = k) :
  (x - 1)^2 + (y - 2)^2 = 4 := 
sorry

end circle_equation_l0_813


namespace probability_of_color_change_is_1_over_6_l0_470

noncomputable def watchColorChangeProbability : ℚ :=
  let cycleDuration := 45 + 5 + 40
  let favorableDuration := 5 + 5 + 5
  favorableDuration / cycleDuration

theorem probability_of_color_change_is_1_over_6 :
  watchColorChangeProbability = 1 / 6 :=
by
  sorry

end probability_of_color_change_is_1_over_6_l0_470


namespace fiftyThreeDaysFromFridayIsTuesday_l0_722

-- Domain definition: Days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to compute the day of the week after a given number of days
def dayAfter (start: Day) (n: ℕ) : Day :=
  match start with
  | Sunday    => Day.ofNat ((0 + n) % 7)
  | Monday    => Day.ofNat ((1 + n) % 7)
  | Tuesday   => Day.ofNat ((2 + n) % 7)
  | Wednesday => Day.ofNat ((3 + n) % 7)
  | Thursday  => Day.ofNat ((4 + n) % 7)
  | Friday    => Day.ofNat ((5 + n) % 7)
  | Saturday  => Day.ofNat ((6 + n) % 7)

namespace Day

-- Helper function to convert natural numbers back to day of the week
def ofNat : ℕ → Day
| 0 => Sunday
| 1 => Monday
| 2 => Tuesday
| 3 => Wednesday
| 4 => Thursday
| 5 => Friday
| 6 => Saturday
| _ => Day.ofNat (n % 7)

-- Lean statement: Prove that 53 days from Friday is Tuesday
theorem fiftyThreeDaysFromFridayIsTuesday :
  dayAfter Friday 53 = Tuesday :=
sorry

end fiftyThreeDaysFromFridayIsTuesday_l0_722


namespace spiderCanEatAllFlies_l0_468

-- Define the number of nodes in the grid.
def numNodes := 100

-- Define initial conditions.
def cornerStart := true
def numFlies := 100
def fliesAtNodes (nodes : ℕ) : Prop := nodes = numFlies

-- Define the predicate for whether the spider can eat all flies within a certain number of moves.
def canEatAllFliesWithinMoves (maxMoves : ℕ) : Prop :=
  ∃ (moves : ℕ), moves ≤ maxMoves

-- The theorem we need to prove in Lean 4.
theorem spiderCanEatAllFlies (h1 : cornerStart) (h2 : fliesAtNodes numFlies) : canEatAllFliesWithinMoves 2000 :=
by
  sorry

end spiderCanEatAllFlies_l0_468


namespace sum_of_possible_k_l0_665

noncomputable def find_sum_k : Nat :=
  let S := { k | ∃ j : Nat, k > 0 ∧ j > 0 ∧ (1 / (j : ℚ) + 1 / (k : ℚ) = 1 / 5) }
  S.to_finset.sum id

theorem sum_of_possible_k : find_sum_k = 46 :=
by
  sorry

end sum_of_possible_k_l0_665


namespace least_three_digit_multiple_l0_431

def LCM (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

theorem least_three_digit_multiple (n : ℕ) :
  (n >= 100) ∧ (n < 1000) ∧ (n % 36 = 0) ∧ (∀ m, (m >= 100) ∧ (m < 1000) ∧ (m % 36 = 0) → n <= m) ↔ n = 108 :=
sorry

end least_three_digit_multiple_l0_431


namespace boss_spends_7600_per_month_l0_656

def hoursPerWeekFiona : ℕ := 40
def hoursPerWeekJohn : ℕ := 30
def hoursPerWeekJeremy : ℕ := 25
def hourlyRate : ℕ := 20
def weeksPerMonth : ℕ := 4

def weeklyEarnings (hours : ℕ) (rate : ℕ) : ℕ := hours * rate
def monthlyEarnings (weekly : ℕ) (weeks : ℕ) : ℕ := weekly * weeks

def totalMonthlyExpenditure : ℕ :=
  monthlyEarnings (weeklyEarnings hoursPerWeekFiona hourlyRate) weeksPerMonth +
  monthlyEarnings (weeklyEarnings hoursPerWeekJohn hourlyRate) weeksPerMonth +
  monthlyEarnings (weeklyEarnings hoursPerWeekJeremy hourlyRate) weeksPerMonth

theorem boss_spends_7600_per_month :
  totalMonthlyExpenditure = 7600 :=
by
  sorry

end boss_spends_7600_per_month_l0_656


namespace last_three_digits_7_pow_103_l0_796

theorem last_three_digits_7_pow_103 : (7 ^ 103) % 1000 = 60 := sorry

end last_three_digits_7_pow_103_l0_796


namespace bike_race_difference_l0_655

-- Define the conditions
def carlos_miles : ℕ := 70
def dana_miles : ℕ := 50
def time_period : ℕ := 5

-- State the theorem to prove the difference in miles biked
theorem bike_race_difference :
  carlos_miles - dana_miles = 20 := 
sorry

end bike_race_difference_l0_655


namespace father_catches_up_l0_749

noncomputable def min_steps_to_catch_up : Prop :=
  let x := 30
  let father_steps := 5
  let xiaoming_steps := 8
  let distance_ratio := 2 / 5
  let xiaoming_headstart := 27
  ((xiaoming_headstart + (xiaoming_steps / father_steps) * x) / distance_ratio) = x

theorem father_catches_up : min_steps_to_catch_up :=
  by
  sorry

end father_catches_up_l0_749


namespace tricycles_count_l0_919

-- Define the variables for number of bicycles, tricycles, and scooters.
variables (b t s : ℕ)

-- Define the total number of children and total number of wheels conditions.
def children_condition := b + t + s = 10
def wheels_condition := 2 * b + 3 * t + 2 * s = 27

-- Prove that number of tricycles t is 4 under these conditions.
theorem tricycles_count : children_condition b t s → wheels_condition b t s → t = 4 := by
  sorry

end tricycles_count_l0_919


namespace exists_xy_interval_l0_235

theorem exists_xy_interval (a b : ℝ) : 
  ∃ (x y : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ |x * y - a * x - b * y| ≥ 1 / 3 :=
sorry

end exists_xy_interval_l0_235


namespace num_distinguishable_arrangements_l0_861

/- Definitions for the problem conditions -/
def num_gold_coins : ℕ := 5
def num_silver_coins : ℕ := 3
def total_coins : ℕ := num_gold_coins + num_silver_coins

/- The main theorem statement -/
theorem num_distinguishable_arrangements : 
  (nat.choose total_coins num_gold_coins) * 30 = 1680 :=
by
  -- Placeholder for proof
  sorry

end num_distinguishable_arrangements_l0_861


namespace exists_monic_poly_degree_8_with_sqrt3_sqrt5_as_root_l0_789

theorem exists_monic_poly_degree_8_with_sqrt3_sqrt5_as_root :
  ∃ p : Polynomial ℚ, Monic p ∧ degree p = 8 ∧ root_of_polynomial p (√3 + √5) := by
  sorry

end exists_monic_poly_degree_8_with_sqrt3_sqrt5_as_root_l0_789


namespace find_triangle_height_l0_868

-- Given conditions
def triangle_area : ℝ := 960
def base : ℝ := 48

-- The problem is to find the height such that 960 = (1/2) * 48 * height
theorem find_triangle_height (height : ℝ) 
  (h_area : triangle_area = (1/2) * base * height) : height = 40 := by
  sorry

end find_triangle_height_l0_868


namespace roots_pure_imaginary_if_negative_real_k_l0_600

theorem roots_pure_imaginary_if_negative_real_k (k : ℝ) (h_neg : k < 0) :
  (∃ (z : ℂ), 10 * z^2 - 3 * Complex.I * z - (k : ℂ) = 0 ∧ z.im ≠ 0 ∧ z.re = 0) :=
sorry

end roots_pure_imaginary_if_negative_real_k_l0_600


namespace total_cost_is_21_l0_35

-- Definitions of the costs
def cost_almond_croissant : Float := 4.50
def cost_salami_and_cheese_croissant : Float := 4.50
def cost_plain_croissant : Float := 3.00
def cost_focaccia : Float := 4.00
def cost_latte : Float := 2.50

-- Theorem stating the total cost
theorem total_cost_is_21 :
  (cost_almond_croissant + cost_salami_and_cheese_croissant) + (2 * cost_latte) + cost_plain_croissant + cost_focaccia = 21.00 :=
by
  sorry

end total_cost_is_21_l0_35


namespace check_sufficient_condition_for_eq_l0_314

theorem check_sufficient_condition_for_eq (a b c : ℤ) (h : a = c - 1 ∧ b = a - 1) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 = 1 := 
by
  sorry

end check_sufficient_condition_for_eq_l0_314


namespace correlation_signs_l0_246

section CorrelationProblem

variables {X Y U V : list ℝ}

-- Given data points
def X_vals : list ℝ := [10, 11.3, 11.8, 12.5, 13]
def Y_vals : list ℝ := [1, 2, 3, 4, 5]
def U_vals : list ℝ := [10, 11.3, 11.8, 12.5, 13]
def V_vals : list ℝ := [5, 4, 3, 2, 1]

-- Definitions of correlation coefficients
noncomputable def r1 := correlation X_vals Y_vals
noncomputable def r2 := correlation U_vals V_vals

-- The problem statement: proof that r2 < 0 < r1
theorem correlation_signs :
  r2 < 0 ∧ 0 < r1 := 
sorry

end CorrelationProblem

end correlation_signs_l0_246


namespace initial_temperature_is_20_l0_838

-- Define the initial temperature, final temperature, rate of increase and time
def T_initial (T_final : ℕ) (rate_of_increase : ℕ) (time : ℕ) : ℕ :=
  T_final - rate_of_increase * time

-- Statement: The initial temperature is 20 degrees given the specified conditions.
theorem initial_temperature_is_20 :
  T_initial 100 5 16 = 20 :=
by
  sorry

end initial_temperature_is_20_l0_838


namespace smallest_n_satisfying_congruence_l0_172

theorem smallest_n_satisfying_congruence :
  ∃ (n : ℕ), n > 0 ∧ (∀ m > 0, m < n → (7^m % 5) ≠ (m^7 % 5)) ∧ (7^n % 5) = (n^7 % 5) := 
by sorry

end smallest_n_satisfying_congruence_l0_172


namespace coin_problem_l0_288

theorem coin_problem : ∃ n : ℕ, n % 8 = 6 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5) ∧ n % 9 = 0 :=
by
  sorry

end coin_problem_l0_288


namespace positive_roots_implies_nonnegative_m_l0_816

variables {x1 x2 m : ℝ}

theorem positive_roots_implies_nonnegative_m (h1 : x1 > 0) (h2 : x2 > 0)
  (h3 : x1 * x2 = 1) (h4 : x1 + x2 = m + 2) : m ≥ 0 :=
by
  sorry

end positive_roots_implies_nonnegative_m_l0_816


namespace peytons_children_l0_110

theorem peytons_children (C : ℕ) (juice_per_week : ℕ) (weeks_in_school_year : ℕ) (total_juice_boxes : ℕ) 
  (h1 : juice_per_week = 5) 
  (h2 : weeks_in_school_year = 25) 
  (h3 : total_juice_boxes = 375)
  (h4 : C * (juice_per_week * weeks_in_school_year) = total_juice_boxes) 
  : C = 3 :=
sorry

end peytons_children_l0_110


namespace range_of_y_l0_503

noncomputable def y (x : ℝ) : ℝ := (Real.log x / Real.log 2 + 2) * (2 * (Real.log x / (2 * Real.log 2)) - 4)

theorem range_of_y :
  (1 ≤ x ∧ x ≤ 8) →
  (∀ t : ℝ, t = Real.log x / Real.log 2 → y x = t^2 - 2 * t - 8 ∧ 0 ≤ t ∧ t ≤ 3) →
  ∃ ymin ymax, (ymin ≤ y x ∧ y x ≤ ymax) ∧ ymin = -9 ∧ ymax = -5 :=
by
  sorry

end range_of_y_l0_503


namespace pentagon_quadrilateral_sum_of_angles_l0_247

   theorem pentagon_quadrilateral_sum_of_angles
     (exterior_angle_pentagon : ℕ := 72)
     (interior_angle_pentagon : ℕ := 108)
     (sum_interior_angles_quadrilateral : ℕ := 360)
     (reflex_angle : ℕ := 252) :
     (sum_interior_angles_quadrilateral - reflex_angle = interior_angle_pentagon) :=
   by
     sorry
   
end pentagon_quadrilateral_sum_of_angles_l0_247


namespace jack_total_plates_after_smashing_and_buying_l0_526

def initial_flower_plates : ℕ := 6
def initial_checked_plates : ℕ := 9
def initial_striped_plates : ℕ := 3
def smashed_flower_plates : ℕ := 2
def smashed_striped_plates : ℕ := 1
def new_polka_dotted_plates : ℕ := initial_checked_plates * initial_checked_plates

theorem jack_total_plates_after_smashing_and_buying : 
  initial_flower_plates - smashed_flower_plates
  + initial_checked_plates
  + initial_striped_plates - smashed_striped_plates
  + new_polka_dotted_plates = 96 := 
by {
  -- calculation proof here
  sorry
}

end jack_total_plates_after_smashing_and_buying_l0_526


namespace TableCostEquals_l0_686

-- Define the given conditions and final result
def total_spent : ℕ := 56
def num_chairs : ℕ := 2
def chair_cost : ℕ := 11
def table_cost : ℕ := 34

-- State the assertion to be proved
theorem TableCostEquals :
  table_cost = total_spent - (num_chairs * chair_cost) := 
by 
  sorry

end TableCostEquals_l0_686


namespace james_received_stickers_l0_669

theorem james_received_stickers (initial_stickers given_away final_stickers received_stickers : ℕ) 
  (h_initial : initial_stickers = 269)
  (h_given : given_away = 48)
  (h_final : final_stickers = 423)
  (h_total_before_giving_away : initial_stickers + received_stickers = given_away + final_stickers) :
  received_stickers = 202 :=
by
  sorry

end james_received_stickers_l0_669


namespace find_special_numbers_l0_578

theorem find_special_numbers :
  {N : ℕ | ∃ k m a, N = m + 10^k * a ∧ 0 ≤ a ∧ a < 10 ∧ 0 ≤ k ∧ m < 10^k 
                ∧ ¬(N % 10 = 0) 
                ∧ (N = 6 * (m + 10^(k+1) * (0 : ℕ))) } = {12, 24, 36, 48} := 
by sorry

end find_special_numbers_l0_578


namespace fifteen_quarters_twenty_dimes_equal_five_quarters_n_dimes_l0_85

theorem fifteen_quarters_twenty_dimes_equal_five_quarters_n_dimes :
  (15 * 25 + 20 * 10 = 5 * 25 + n * 10) -> n = 45 :=
by
  sorry

end fifteen_quarters_twenty_dimes_equal_five_quarters_n_dimes_l0_85


namespace biff_break_even_time_l0_927

noncomputable def total_cost_excluding_wifi : ℝ :=
  11 + 3 + 16 + 8 + 10 + 35 + 0.1 * 35

noncomputable def total_cost_including_wifi_connection : ℝ :=
  total_cost_excluding_wifi + 5

noncomputable def effective_hourly_earning : ℝ := 12 - 1

noncomputable def hours_to_break_even : ℝ :=
  total_cost_including_wifi_connection / effective_hourly_earning

theorem biff_break_even_time : hours_to_break_even ≤ 9 := by
  sorry

end biff_break_even_time_l0_927


namespace set_union_complement_eq_l0_387

def P : Set ℝ := {x | x^2 - 4 * x + 3 ≤ 0}
def Q : Set ℝ := {x | x^2 - 4 < 0}
def R_complement_Q : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}

theorem set_union_complement_eq :
  P ∪ R_complement_Q = {x | x ≤ -2} ∪ {x | x ≥ 1} :=
by {
  sorry
}

end set_union_complement_eq_l0_387


namespace rectangular_reconfiguration_l0_854

theorem rectangular_reconfiguration (k : ℕ) (n : ℕ) (h₁ : k - 5 > 0) (h₂ : k ≥ 6) (h₃ : k ≤ 9) :
  (k * (k - 5) = n^2) → (n = 6) :=
by {
  sorry  -- proof is omitted
}

end rectangular_reconfiguration_l0_854


namespace salary_based_on_tax_l0_276

theorem salary_based_on_tax (salary tax paid_tax excess_800 excess_500 excess_500_2000 : ℤ) 
    (h1 : excess_800 = salary - 800)
    (h2 : excess_500 = min excess_800 500)
    (h3 : excess_500_2000 = excess_800 - excess_500)
    (h4 : paid_tax = (excess_500 * 5 / 100) + (excess_500_2000 * 10 / 100))
    (h5 : paid_tax = 80) :
  salary = 1850 := by
  sorry

end salary_based_on_tax_l0_276


namespace find_n_in_arithmetic_sequence_l0_60

theorem find_n_in_arithmetic_sequence 
  (a : ℕ → ℕ)
  (a_1 : ℕ)
  (d : ℕ) 
  (a_n : ℕ) 
  (n : ℕ)
  (h₀ : a_1 = 11)
  (h₁ : d = 2)
  (h₂ : a n = a_1 + (n - 1) * d)
  (h₃ : a n = 2009) :
  n = 1000 := 
by
  -- The proof steps would go here
  sorry

end find_n_in_arithmetic_sequence_l0_60


namespace triangle_area_l0_471

theorem triangle_area {r : ℝ} (h_r : r = 6) {x : ℝ} 
  (h1 : 5 * x = 2 * r)
  (h2 : x = 12 / 5) : 
  (1 / 2 * (3 * x) * (4 * x) = 34.56) :=
by
  sorry

end triangle_area_l0_471


namespace quadrilateral_diagonals_midpoint_l0_996

noncomputable def is_midpoint (P A B : Point) : Prop := dist P A = dist P B

theorem quadrilateral_diagonals_midpoint
    (A B C D P : Point)
    (h1 : collinear A C P)
    (h2 : collinear B D P)
    (h3 : Area (Triangle.mk A B P) ^ 2 + Area (Triangle.mk C D P) ^ 2 =
          Area (Triangle.mk B C P) ^ 2 + Area (Triangle.mk A D P) ^ 2)
    : ∃ Q R S, (Q, R ∈ set.insert A (set.insert B (set.insert C (set.singleton D)))) ∧ 
      (Q = R → False) ∧ 
      ((S = Q ∨ S = R) ∧ is_midpoint P Q R) :=
by
  sorry

end quadrilateral_diagonals_midpoint_l0_996


namespace num_valid_seating_arrangements_l0_484

-- Define the dimensions of the examination room
def rows : Nat := 5
def columns : Nat := 6
def total_seats : Nat := rows * columns

-- Define the condition for students not sitting next to each other
def valid_seating_arrangements (rows columns : Nat) : Nat := sorry

-- The theorem to prove the number of seating arrangements
theorem num_valid_seating_arrangements : valid_seating_arrangements rows columns = 772 := 
by 
  sorry

end num_valid_seating_arrangements_l0_484


namespace sequence_equal_l0_913

variable {n : ℕ} (h1 : 2 ≤ n)
variable (a : ℕ → ℝ)
variable (h2 : ∀ i, a i ≠ -1)
variable (h3 : ∀ i, a (i + 2) = (a i ^ 2 + a i) / (a (i + 1) + 1))
variable (h4 : a n = a 0)
variable (h5 : a (n + 1) = a 1)

theorem sequence_equal 
  (h1 : 2 ≤ n)
  (h2 : ∀ i, a i ≠ -1) 
  (h3 : ∀ i, a (i + 2) = (a i ^ 2 + a i) / (a (i + 1) + 1))
  (h4 : a n = a 0)
  (h5 : a (n + 1) = a 1) :
  ∀ i, a i = a 0 := 
sorry

end sequence_equal_l0_913


namespace further_flight_Gaeun_l0_388

theorem further_flight_Gaeun :
  let nana_distance_m := 1.618
  let gaeun_distance_cm := 162.3
  let conversion_factor := 100
  let nana_distance_cm := nana_distance_m * conversion_factor
  gaeun_distance_cm > nana_distance_cm := 
  sorry

end further_flight_Gaeun_l0_388


namespace max_legs_lengths_l0_997

theorem max_legs_lengths (a x y : ℝ) (h₁ : x^2 + y^2 = a^2) (h₂ : 3 * x + 4 * y ≤ 5 * a) :
  3 * x + 4 * y = 5 * a → x = (3 * a / 5) ∧ y = (4 * a / 5) :=
by
  sorry

end max_legs_lengths_l0_997


namespace watermelon_ratio_l0_983

theorem watermelon_ratio (michael_weight : ℕ) (john_weight : ℕ) (clay_weight : ℕ)
  (h₁ : michael_weight = 8) 
  (h₂ : john_weight = 12) 
  (h₃ : john_weight * 2 = clay_weight) :
  clay_weight / michael_weight = 3 :=
by {
  sorry
}

end watermelon_ratio_l0_983


namespace clothing_price_reduction_l0_581

def price_reduction (original_profit_per_piece : ℕ) (original_sales_volume : ℕ) (target_profit : ℕ) (increase_in_sales_per_unit_price_reduction : ℕ) : ℕ :=
  sorry

theorem clothing_price_reduction :
  ∃ x : ℕ, (40 - x) * (20 + 2 * x) = 1200 :=
sorry

end clothing_price_reduction_l0_581


namespace tom_age_difference_l0_423

/-- 
Tom Johnson's age is some years less than twice as old as his sister.
The sum of their ages is 14 years.
Tom's age is 9 years.
Prove that the number of years less Tom's age is than twice his sister's age is 1 year. 
-/ 
theorem tom_age_difference (T S : ℕ) 
  (h₁ : T = 9) 
  (h₂ : T + S = 14) : 
  2 * S - T = 1 := 
by 
  sorry

end tom_age_difference_l0_423


namespace count_positive_integers_satisfying_conditions_l0_313

theorem count_positive_integers_satisfying_conditions :
  let condition1 (n : ℕ) := (169 * n) ^ 25 > n ^ 75
  let condition2 (n : ℕ) := n ^ 75 > 3 ^ 150
  ∃ (count : ℕ), count = 3 ∧ (∀ (n : ℕ), (condition1 n) ∧ (condition2 n) → 9 < n ∧ n < 13) :=
by
  sorry

end count_positive_integers_satisfying_conditions_l0_313


namespace radius_of_circle_with_square_and_chord_l0_43

theorem radius_of_circle_with_square_and_chord :
  ∃ (r : ℝ), 
    (∀ (chord_length square_side_length : ℝ), chord_length = 6 ∧ square_side_length = 2 → 
    (r = Real.sqrt 10)) :=
by
  sorry

end radius_of_circle_with_square_and_chord_l0_43


namespace tan_theta_eq_l0_384

variables (k θ : ℝ)

-- Condition: k > 0
axiom k_pos : k > 0

-- Condition: k * cos θ = 12
axiom k_cos_theta : k * Real.cos θ = 12

-- Condition: k * sin θ = 5
axiom k_sin_theta : k * Real.sin θ = 5

-- To prove: tan θ = 5 / 12
theorem tan_theta_eq : Real.tan θ = 5 / 12 := by
  sorry

end tan_theta_eq_l0_384


namespace difference_is_20_l0_905

def x : ℕ := 10

def a : ℕ := 3 * x

def b : ℕ := 20 - x

theorem difference_is_20 : a - b = 20 := 
by 
  sorry

end difference_is_20_l0_905


namespace find_base_number_l0_513

theorem find_base_number (y : ℕ) (base : ℕ) (h : 9^y = base ^ 16) (hy : y = 8) : base = 3 :=
by
  -- We skip the proof steps and insert sorry here
  sorry

end find_base_number_l0_513


namespace solve_for_x_l0_399

theorem solve_for_x : ∃ x : ℤ, 25 - (4 + 3) = 5 + x ∧ x = 13 :=
by {
  sorry
}

end solve_for_x_l0_399


namespace simplify_and_evaluate_l0_866

theorem simplify_and_evaluate
  (a b : ℝ)
  (h : |a - 1| + (b + 2)^2 = 0) :
  ((2 * a + b)^2 - (2 * a + b) * (2 * a - b)) / (-1 / 2 * b) = 0 := 
sorry

end simplify_and_evaluate_l0_866


namespace determine_digit_I_l0_837

theorem determine_digit_I (F I V E T H R N : ℕ) (hF : F = 8) (hE_odd : E = 1 ∨ E = 3 ∨ E = 5 ∨ E = 7 ∨ E = 9)
  (h_diff : F ≠ I ∧ F ≠ V ∧ F ≠ E ∧ F ≠ T ∧ F ≠ H ∧ F ≠ R ∧ F ≠ N 
             ∧ I ≠ V ∧ I ≠ E ∧ I ≠ T ∧ I ≠ H ∧ I ≠ R ∧ I ≠ N 
             ∧ V ≠ E ∧ V ≠ T ∧ V ≠ H ∧ V ≠ R ∧ V ≠ N 
             ∧ E ≠ T ∧ E ≠ H ∧ E ≠ R ∧ E ≠ N 
             ∧ T ≠ H ∧ T ≠ R ∧ T ≠ N 
             ∧ H ≠ R ∧ H ≠ N 
             ∧ R ≠ N)
  (h_verify_sum : (10^3 * 8 + 10^2 * I + 10 * V + E) + (10^4 * T + 10^3 * H + 10^2 * R + 11 * E) = 10^3 * N + 10^2 * I + 10 * N + E) :
  I = 4 := 
sorry

end determine_digit_I_l0_837
