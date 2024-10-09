import Mathlib

namespace simplify_expr_l249_24950

-- Define the expression
def expr (a : ℝ) := 4 * a ^ 2 * (3 * a - 1)

-- State the theorem
theorem simplify_expr (a : ℝ) : expr a = 12 * a ^ 3 - 4 * a ^ 2 := 
by 
  sorry

end simplify_expr_l249_24950


namespace inequality_solution_l249_24903

theorem inequality_solution (a : ℝ) : (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 :=
by
  sorry

end inequality_solution_l249_24903


namespace prime_divides_sum_diff_l249_24968

theorem prime_divides_sum_diff
  (a b c p : ℕ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hp : p.Prime) 
  (h1 : p ∣ (100 * a + 10 * b + c)) 
  (h2 : p ∣ (100 * c + 10 * b + a)) 
  : p ∣ (a + b + c) ∨ p ∣ (a - b + c) ∨ p ∣ (a - c) :=
by
  sorry

end prime_divides_sum_diff_l249_24968


namespace grade_point_average_l249_24964

theorem grade_point_average (X : ℝ) (GPA_rest : ℝ) (GPA_whole : ℝ) 
  (h1 : GPA_rest = 66) (h2 : GPA_whole = 64) 
  (h3 : (1 / 3) * X + (2 / 3) * GPA_rest = GPA_whole) : X = 60 :=
sorry

end grade_point_average_l249_24964


namespace train_length_correct_l249_24934

noncomputable def train_length (v_kmph : ℝ) (t_sec : ℝ) (bridge_length : ℝ) : ℝ :=
  let v_mps := v_kmph / 3.6
  let total_distance := v_mps * t_sec
  total_distance - bridge_length

theorem train_length_correct : train_length 72 12.099 132 = 109.98 :=
by
  sorry

end train_length_correct_l249_24934


namespace first_train_takes_4_hours_less_l249_24991

-- Definitions of conditions
def distance: ℝ := 425.80645161290323
def speed_first_train: ℝ := 75
def speed_second_train: ℝ := 44

-- Lean statement to prove the correct answer
theorem first_train_takes_4_hours_less:
  (distance / speed_second_train) - (distance / speed_first_train) = 4 := 
  by
    -- Skip the actual proof
    sorry

end first_train_takes_4_hours_less_l249_24991


namespace max_sum_of_arithmetic_sequence_l249_24955

theorem max_sum_of_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a3 : a 3 = 7)
  (h_a1_a7 : a 1 + a 7 = 10)
  (h_S : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 1 - a 0))) / 2) :
  ∃ n, S n = S 6 ∧ (∀ m, S m ≤ S 6) :=
sorry

end max_sum_of_arithmetic_sequence_l249_24955


namespace cos_alpha_in_fourth_quadrant_l249_24965

theorem cos_alpha_in_fourth_quadrant (α : ℝ) (P : ℝ × ℝ) (h_angle_quadrant : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi)
(h_point : P = (Real.sqrt 5, 2)) (h_sin : Real.sin α = (Real.sqrt 2 / 4) * 2) :
  Real.cos α = Real.sqrt 10 / 4 :=
sorry

end cos_alpha_in_fourth_quadrant_l249_24965


namespace correct_operation_l249_24982

theorem correct_operation : (a : ℕ) →
  (a^2 * a^3 = a^5) ∧
  (2 * a + 4 ≠ 6 * a) ∧
  ((2 * a)^2 ≠ 2 * a^2) ∧
  (a^3 / a^3 ≠ a) := sorry

end correct_operation_l249_24982


namespace combined_speed_in_still_water_l249_24975

theorem combined_speed_in_still_water 
  (U1 D1 U2 D2 : ℝ) 
  (hU1 : U1 = 30) 
  (hD1 : D1 = 60) 
  (hU2 : U2 = 40) 
  (hD2 : D2 = 80) 
  : (U1 + D1) / 2 + (U2 + D2) / 2 = 105 := 
by 
  sorry

end combined_speed_in_still_water_l249_24975


namespace parcel_cost_guangzhou_shanghai_l249_24999

theorem parcel_cost_guangzhou_shanghai (x y : ℕ) :
  (x + 2 * y = 10 ∧ x + 3 * (y + 3) + 2 = 23) →
  (x = 6 ∧ y = 2 ∧ (6 + 4 * 2 = 14)) := by
  sorry

end parcel_cost_guangzhou_shanghai_l249_24999


namespace forest_enclosure_l249_24932

theorem forest_enclosure
  (n : ℕ)
  (a : Fin n → ℝ)
  (h_a_lt_100 : ∀ i, a i < 100)
  (d : Fin n → Fin n → ℝ)
  (h_dist : ∀ i j, i < j → d i j ≤ (a i) - (a j)) :
  ∃ f : ℝ, f = 200 :=
by
  -- The proof goes here
  sorry

end forest_enclosure_l249_24932


namespace units_digit_seven_pow_ten_l249_24907

theorem units_digit_seven_pow_ten : ∃ u : ℕ, (7^10) % 10 = u ∧ u = 9 :=
by
  use 9
  sorry

end units_digit_seven_pow_ten_l249_24907


namespace min_distance_mn_l249_24984

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_distance_mn : ∃ m > 0, ∀ x > 0, |f x - g x| = 1/2 + 1/2 * Real.log 2 :=
by
  sorry

end min_distance_mn_l249_24984


namespace integer_solutions_to_quadratic_inequality_l249_24981

theorem integer_solutions_to_quadratic_inequality :
  {x : ℤ | (x^2 + 6 * x + 8) * (x^2 - 4 * x + 3) < 0} = {-3, 2} :=
by
  sorry

end integer_solutions_to_quadratic_inequality_l249_24981


namespace fraction_addition_l249_24977

def fraction_sum : ℚ := (2 : ℚ)/5 + (3 : ℚ)/8

theorem fraction_addition : fraction_sum = 31/40 := by
  sorry

end fraction_addition_l249_24977


namespace trigonometric_identity_l249_24921

theorem trigonometric_identity (α : ℝ) (h : (1 + Real.tan α) / (1 - Real.tan α) = 2012) : 
  (1 / Real.cos (2 * α)) + Real.tan (2 * α) = 2012 := 
by
  -- This will be the proof body which we omit with sorry
  sorry

end trigonometric_identity_l249_24921


namespace matrix_addition_correct_l249_24918

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, -1, 3], ![0, 4, -2], ![5, -3, 1]]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![ -3,  2, -4], ![ 1, -6,  3], ![-2,  4,  0]]

def expectedSum : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![-1,  1, -1], ![ 1, -2,  1], ![ 3,  1,  1]]

theorem matrix_addition_correct :
  A + B = expectedSum := by
  sorry

end matrix_addition_correct_l249_24918


namespace max_value_of_sum_l249_24920

theorem max_value_of_sum (x y z : ℝ) (h : x^2 + 4 * y^2 + 9 * z^2 = 3) : x + 2 * y + 3 * z ≤ 3 :=
sorry

end max_value_of_sum_l249_24920


namespace average_salary_of_technicians_l249_24973

theorem average_salary_of_technicians
  (total_workers : ℕ)
  (average_salary_all : ℕ)
  (average_salary_non_technicians : ℕ)
  (num_technicians : ℕ)
  (num_non_technicians : ℕ)
  (h1 : total_workers = 21)
  (h2 : average_salary_all = 8000)
  (h3 : average_salary_non_technicians = 6000)
  (h4 : num_technicians = 7)
  (h5 : num_non_technicians = 14) :
  (average_salary_all * total_workers - average_salary_non_technicians * num_non_technicians) / num_technicians = 12000 :=
by
  sorry

end average_salary_of_technicians_l249_24973


namespace problem_proof_l249_24971

-- Definition of the function f
def f (x : ℝ) : ℝ := 2 * x + 2 - x

-- Condition given in the problem
axiom h : ∃ a : ℝ, f a = 3

-- Theorem statement
theorem problem_proof : ∃ a : ℝ, f a = 3 → f (2 * a) = 7 :=
by
  sorry

end problem_proof_l249_24971


namespace num_white_balls_l249_24989

theorem num_white_balls (W : ℕ) (h : (W : ℝ) / (6 + W) = 0.45454545454545453) : W = 5 :=
by
  sorry

end num_white_balls_l249_24989


namespace smallest_arithmetic_mean_divisible_product_l249_24933

theorem smallest_arithmetic_mean_divisible_product :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → Nat.gcd ((n + i) * 1111) 1111 = 1111) ∧ (n + 4 = 97) := 
  sorry

end smallest_arithmetic_mean_divisible_product_l249_24933


namespace calculate_value_expression_l249_24938

theorem calculate_value_expression :
  3000 * (3000 ^ 3000 + 3000 ^ 2999) = 3001 * 3000 ^ 3000 := 
by
  sorry

end calculate_value_expression_l249_24938


namespace polynomial_identity_l249_24969

theorem polynomial_identity : 
  ∀ x : ℝ, 
    5 * x^3 - 32 * x^2 + 75 * x - 71 = 
    5 * (x - 2)^3 + (-2) * (x - 2)^2 + 7 * (x - 2) - 9 :=
by 
  sorry

end polynomial_identity_l249_24969


namespace arithmetic_sequence_30th_term_l249_24940

-- Definitions
def a₁ : ℤ := 8
def d : ℤ := -3
def n : ℕ := 30

-- The statement to be proved
theorem arithmetic_sequence_30th_term :
  a₁ + (n - 1) * d = -79 :=
by
  sorry

end arithmetic_sequence_30th_term_l249_24940


namespace geometric_sequence_n_value_l249_24956

theorem geometric_sequence_n_value (a : ℕ → ℝ) (q : ℝ) (n : ℕ) 
  (h1 : a 3 + a 6 = 36) 
  (h2 : a 4 + a 7 = 18)
  (h3 : a n = 1/2) :
  n = 9 :=
sorry

end geometric_sequence_n_value_l249_24956


namespace opposite_terminal_sides_l249_24951

theorem opposite_terminal_sides (α β : ℝ) (k : ℤ) (h : ∃ k : ℤ, α = β + 180 + k * 360) :
  α = β + 180 + k * 360 :=
by sorry

end opposite_terminal_sides_l249_24951


namespace person_speed_l249_24960

namespace EscalatorProblem

/-- The speed of the person v_p walking on the moving escalator is 3 ft/sec given the conditions -/
theorem person_speed (v_p : ℝ) 
  (escalator_speed : ℝ := 12) 
  (escalator_length : ℝ := 150) 
  (time_taken : ℝ := 10) :
  escalator_length = (v_p + escalator_speed) * time_taken → v_p = 3 := 
by sorry

end EscalatorProblem

end person_speed_l249_24960


namespace baseball_tickets_l249_24997

theorem baseball_tickets (B : ℕ) 
  (h1 : 25 = 2 * B + 6) : B = 9 :=
sorry

end baseball_tickets_l249_24997


namespace range_of_a_l249_24908

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬ (x^2 - 2 * x + 3 ≤ a^2 - 2 * a - 1)) ↔ (-1 < a ∧ a < 3) :=
sorry

end range_of_a_l249_24908


namespace equation_of_parallel_line_through_point_l249_24912

theorem equation_of_parallel_line_through_point :
  ∃ m b, (∀ x y, y = m * x + b → (∃ k, k = 3 ^ 2 - 9 * 2 + 1)) ∧ 
         (∀ x y, y = 3 * x + b → y - 0 = 3 * (x - (-2))) :=
sorry

end equation_of_parallel_line_through_point_l249_24912


namespace percentage_increase_is_50_l249_24905

def papaya_growth (P : ℝ) : Prop :=
  let growth1 := 2
  let growth2 := 2 * (1 + P / 100)
  let growth3 := 1.5 * growth2
  let growth4 := 2 * growth3
  let growth5 := 0.5 * growth4
  growth1 + growth2 + growth3 + growth4 + growth5 = 23

theorem percentage_increase_is_50 :
  ∃ (P : ℝ), papaya_growth P ∧ P = 50 := by
  sorry

end percentage_increase_is_50_l249_24905


namespace factor_polynomial_l249_24909

variable {R : Type*} [CommRing R]

theorem factor_polynomial (a b c : R) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (-(a + b + c) * (a^2 + b^2 + c^2 + ab + bc + ac)) :=
by
  sorry

end factor_polynomial_l249_24909


namespace maximize_expression_l249_24996

theorem maximize_expression
  (a b c : ℝ)
  (h1 : a ≥ 0)
  (h2 : b ≥ 0)
  (h3 : c ≥ 0)
  (h_sum : a + b + c = 3) :
  (a^2 - a * b + b^2) * (a^2 - a * c + c^2) * (b^2 - b * c + c^2) ≤ 729 / 432 := 
sorry

end maximize_expression_l249_24996


namespace initial_candles_count_l249_24957

section

variable (C : ℝ)
variable (h_Alyssa : C / 2 = C / 2)
variable (h_Chelsea : C / 2 - 0.7 * (C / 2) = 6)

theorem initial_candles_count : C = 40 := 
by sorry

end

end initial_candles_count_l249_24957


namespace number_of_neutrons_eq_l249_24901

variable (A n x : ℕ)

/-- The number of neutrons N in the nucleus of an atom R, given that:
  1. A is the atomic mass number of R.
  2. The ion RO3^(n-) contains x outer electrons. -/
theorem number_of_neutrons_eq (N : ℕ) (h : A - N + 24 + n = x) : N = A + n + 24 - x :=
by sorry

end number_of_neutrons_eq_l249_24901


namespace julian_notes_problem_l249_24986

theorem julian_notes_problem (x y : ℤ) (h1 : 3 * x + 4 * y = 151) (h2 : x = 19 ∨ y = 19) :
  x = 25 ∨ y = 25 := 
by
  sorry

end julian_notes_problem_l249_24986


namespace return_journey_time_l249_24978

-- Define the conditions
def walking_speed : ℕ := 100 -- meters per minute
def walking_time : ℕ := 36 -- minutes
def running_speed : ℕ := 3 -- meters per second

-- Define derived values from conditions
def distance_walked : ℕ := walking_speed * walking_time -- meters
def running_speed_minute : ℕ := running_speed * 60 -- meters per minute

-- Statement of the problem
theorem return_journey_time :
  (distance_walked / running_speed_minute) = 20 := by
  sorry

end return_journey_time_l249_24978


namespace stone_length_is_correct_l249_24985

variable (length_m width_m : ℕ)
variable (num_stones : ℕ)
variable (width_stone dm : ℕ)

def length_of_each_stone (length_m : ℕ) (width_m : ℕ) (num_stones : ℕ) (width_stone : ℕ) : ℕ :=
  let length_dm := length_m * 10
  let width_dm := width_m * 10
  let area_hall := length_dm * width_dm
  let area_stone := width_stone * 5
  (area_hall / num_stones) / width_stone

theorem stone_length_is_correct :
  length_of_each_stone 36 15 5400 5 = 2 := by
  sorry

end stone_length_is_correct_l249_24985


namespace person_speed_in_kmph_l249_24966

-- Define the distance in meters
def distance_meters : ℕ := 300

-- Define the time in minutes
def time_minutes : ℕ := 4

-- Function to convert distance from meters to kilometers
def meters_to_kilometers (m : ℕ) : ℕ := m / 1000

-- Function to convert time from minutes to hours
def minutes_to_hours (min : ℕ) : ℚ := min / 60

-- Define the expected speed in km/h
def expected_speed : ℚ := 4.5

-- Proof statement
theorem person_speed_in_kmph : 
  meters_to_kilometers distance_meters / minutes_to_hours time_minutes = expected_speed :=
by 
  -- This is where the steps to verify the theorem would be located, currently omitted for the sake of the statement.
  sorry

end person_speed_in_kmph_l249_24966


namespace find_x_l249_24904

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  ⟨Q.x - P.x, Q.y - P.y⟩

def parallel (v w : Point) : Prop :=
  v.x * w.y = v.y * w.x

theorem find_x (A B C : Point) (hA : A = ⟨0, -3⟩) (hB : B = ⟨3, 3⟩) (hC : C = ⟨x, -1⟩) (h_parallel : parallel (vector A B) (vector A C)) : x = 1 := 
by
  sorry

end find_x_l249_24904


namespace find_g_at_7_l249_24953

noncomputable def g (x : ℝ) (a b c : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 4

theorem find_g_at_7 (a b c : ℝ) (h_symm : ∀ x : ℝ, g x a b c + g (-x) a b c = -8) (h_neg7: g (-7) a b c = 12) :
  g 7 a b c = -20 :=
by
  sorry

end find_g_at_7_l249_24953


namespace MischiefConventionHandshakes_l249_24976

theorem MischiefConventionHandshakes :
  let gremlins := 30
  let imps := 25
  let reconciled_imps := 10
  let non_reconciled_imps := imps - reconciled_imps
  let handshakes_among_gremlins := (gremlins * (gremlins - 1)) / 2
  let handshakes_among_imps := (reconciled_imps * (reconciled_imps - 1)) / 2
  let handshakes_between_gremlins_and_imps := gremlins * imps
  handshakes_among_gremlins + handshakes_among_imps + handshakes_between_gremlins_and_imps = 1230 := by
  sorry

end MischiefConventionHandshakes_l249_24976


namespace selling_price_correct_l249_24967

def meters_of_cloth : ℕ := 45
def profit_per_meter : ℝ := 12
def cost_price_per_meter : ℝ := 88
def total_selling_price : ℝ := 4500

theorem selling_price_correct :
  (cost_price_per_meter * meters_of_cloth) + (profit_per_meter * meters_of_cloth) = total_selling_price :=
by
  sorry

end selling_price_correct_l249_24967


namespace string_length_l249_24947

def cylindrical_post_circumference : ℝ := 6
def cylindrical_post_height : ℝ := 15
def loops : ℝ := 3

theorem string_length :
  (cylindrical_post_height / loops)^2 + cylindrical_post_circumference^2 = 61 → 
  loops * Real.sqrt 61 = 3 * Real.sqrt 61 :=
by
  sorry

end string_length_l249_24947


namespace solve_inequalities_l249_24998

theorem solve_inequalities {x : ℝ} :
  (3 * x + 1) / 2 > x ∧ (4 * (x - 2) ≤ x - 5) ↔ (-1 < x ∧ x ≤ 1) :=
by sorry

end solve_inequalities_l249_24998


namespace fraction_of_sy_not_declared_major_l249_24943

-- Conditions
variables (T : ℝ) -- Total number of students
variables (first_year : ℝ) -- Fraction of first-year students
variables (second_year : ℝ) -- Fraction of second-year students
variables (decl_fy_major : ℝ) -- Fraction of first-year students who have declared a major
variables (decl_sy_major : ℝ) -- Fraction of second-year students who have declared a major

-- Definitions from conditions
def fraction_first_year_students := 1 / 2
def fraction_second_year_students := 1 / 2
def fraction_fy_declared_major := 1 / 5
def fraction_sy_declared_major := 4 * fraction_fy_declared_major

-- Hollow statement
theorem fraction_of_sy_not_declared_major :
  first_year = fraction_first_year_students →
  second_year = fraction_second_year_students →
  decl_fy_major = fraction_fy_declared_major →
  decl_sy_major = fraction_sy_declared_major →
  (1 - decl_sy_major) * second_year = 1 / 10 :=
by
  sorry

end fraction_of_sy_not_declared_major_l249_24943


namespace cecile_apples_l249_24911

theorem cecile_apples (C D : ℕ) (h1 : D = C + 20) (h2 : C + D = 50) : C = 15 :=
by
  -- Proof steps would go here
  sorry

end cecile_apples_l249_24911


namespace count_valid_three_digit_numbers_l249_24941

theorem count_valid_three_digit_numbers : 
  let is_valid (a b c : ℕ) := 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ b = (a + c) / 2 ∧ (a + c) % 2 = 0
  ∃ n : ℕ, (∀ a b c : ℕ, is_valid a b c → n = 45) :=
sorry

end count_valid_three_digit_numbers_l249_24941


namespace marco_might_need_at_least_n_tables_n_tables_are_sufficient_l249_24942
open Function

variables (n : ℕ) (friends_sticker_sets : Fin n → Finset (Fin n))

-- Each friend is missing exactly one unique sticker
def each_friend_missing_one_unique_sticker :=
  ∀ i : Fin n, ∃ j : Fin n, friends_sticker_sets i = (Finset.univ \ {j})

-- A pair of friends is wholesome if their combined collection has all stickers
def is_wholesome_pair (i j : Fin n) :=
  ∀ s : Fin n, s ∈ friends_sticker_sets i ∨ s ∈ friends_sticker_sets j

-- Main problem statements
-- Problem 1: Marco might need to reserve at least n different tables
theorem marco_might_need_at_least_n_tables 
  (h : each_friend_missing_one_unique_sticker n friends_sticker_sets) : 
  ∃ i j : Fin n, i ≠ j ∧ is_wholesome_pair n friends_sticker_sets i j :=
sorry

-- Problem 2: n tables will always be enough for Marco to achieve his goal
theorem n_tables_are_sufficient
  (h : each_friend_missing_one_unique_sticker n friends_sticker_sets) :
  ∃ arrangement : Fin n → Fin n, ∀ i j, i ≠ j → arrangement i ≠ arrangement j :=
sorry

end marco_might_need_at_least_n_tables_n_tables_are_sufficient_l249_24942


namespace sum_a_b_eq_4_l249_24995

-- Define the problem conditions
variables (a b : ℝ)

-- State the conditions
def condition1 : Prop := 2 * a = 8
def condition2 : Prop := a^2 - b = 16

-- State the theorem
theorem sum_a_b_eq_4 (h1 : condition1 a) (h2 : condition2 a b) : a + b = 4 :=
by sorry

end sum_a_b_eq_4_l249_24995


namespace carrie_savings_l249_24974

-- Define the original prices and discount rates
def deltaPrice : ℝ := 850
def deltaDiscount : ℝ := 0.20
def unitedPrice : ℝ := 1100
def unitedDiscount : ℝ := 0.30

-- Calculate discounted prices
def deltaDiscountAmount : ℝ := deltaPrice * deltaDiscount
def unitedDiscountAmount : ℝ := unitedPrice * unitedDiscount

def deltaDiscountedPrice : ℝ := deltaPrice - deltaDiscountAmount
def unitedDiscountedPrice : ℝ := unitedPrice - unitedDiscountAmount

-- Define the savings by choosing the cheaper flight
def savingsByChoosingCheaperFlight : ℝ := unitedDiscountedPrice - deltaDiscountedPrice

-- The theorem stating the amount saved
theorem carrie_savings : savingsByChoosingCheaperFlight = 90 :=
by
  sorry

end carrie_savings_l249_24974


namespace problem_statement_l249_24931

theorem problem_statement (m : ℝ) : 
  (¬ (∃ x : ℝ, x^2 + 2 * x + m ≤ 0)) → m > 1 :=
by
  sorry

end problem_statement_l249_24931


namespace gcd_of_set_B_is_five_l249_24952

def is_in_set_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = (x - 2) + (x - 1) + x + (x + 1) + (x + 2)

theorem gcd_of_set_B_is_five : ∃ d, (∀ n, is_in_set_B n → d ∣ n) ∧ d = 5 :=
by 
  sorry

end gcd_of_set_B_is_five_l249_24952


namespace solve_inequalities_l249_24958

theorem solve_inequalities (x : ℝ) : (x + 1 > 0 ∧ x - 3 < 2) ↔ (-1 < x ∧ x < 5) :=
by sorry

end solve_inequalities_l249_24958


namespace abs_neg_three_l249_24993

theorem abs_neg_three : abs (-3) = 3 := 
by
  sorry

end abs_neg_three_l249_24993


namespace quadratic_inequality_solution_l249_24959

theorem quadratic_inequality_solution (m : ℝ) :
  {x : ℝ | (x - m) * (x - (m + 1)) > 0} = {x | x < m ∨ x > m + 1} := sorry

end quadratic_inequality_solution_l249_24959


namespace real_roots_of_f_l249_24962

noncomputable def f (x : ℝ) : ℝ := x^4 - 3 * x^3 + 3 * x^2 - x - 6

theorem real_roots_of_f :
  {x | f x = 0} = {-1, 1, 2, 3} :=
sorry

end real_roots_of_f_l249_24962


namespace find_n_l249_24917

theorem find_n {x n : ℕ} (h1 : 3 * x - 4 = 8) (h2 : 7 * x - 15 = 13) (h3 : 4 * x + 2 = 18) 
  (h4 : n = 803) : 8 + (n - 1) * 5 = 4018 := by
  sorry

end find_n_l249_24917


namespace frac_sum_eq_l249_24900

theorem frac_sum_eq (a b : ℝ) (h1 : a^2 + a - 1 = 0) (h2 : b^2 + b - 1 = 0) : 
  (a / b + b / a = 2) ∨ (a / b + b / a = -3) := 
sorry

end frac_sum_eq_l249_24900


namespace arithmetic_sequence_sum_l249_24983

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 8 = 8)
  (h2 : ∀ n, S n = n * (a 1 + a n) / 2) (h3 : a 1 + a 15 = 2 * a 8) :
  S 15 = 120 := sorry

end arithmetic_sequence_sum_l249_24983


namespace smallest_b_l249_24935

theorem smallest_b
  (a b : ℕ)
  (h_pos : 0 < b)
  (h_diff : a - b = 8)
  (h_gcd : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) :
  b = 4 := sorry

end smallest_b_l249_24935


namespace exists_sequence_a_l249_24925

-- Define the sequence and properties
def sequence_a (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 = 2 ∧
  a 18 = 2019 ∧
  ∀ k, 3 ≤ k → k ≤ 18 → ∃ i j, 1 ≤ i → i < j → j < k → a k = a i + a j

-- The main theorem statement
theorem exists_sequence_a : ∃ (a : ℕ → ℤ), sequence_a a := 
sorry

end exists_sequence_a_l249_24925


namespace find_digits_l249_24954

def divisible_45z_by_8 (z : ℕ) : Prop :=
  45 * z % 8 = 0

def sum_digits_divisible_by_9 (x y z : ℕ) : Prop :=
  (1 + 3 + x + y + 4 + 5 + z) % 9 = 0

def alternating_sum_digits_divisible_by_11 (x y z : ℕ) : Prop :=
  (1 - 3 + x - y + 4 - 5 + z) % 11 = 0

theorem find_digits (x y z : ℕ) (h_div8 : divisible_45z_by_8 z) (h_div9 : sum_digits_divisible_by_9 x y z) (h_div11 : alternating_sum_digits_divisible_by_11 x y z) :
  x = 2 ∧ y = 3 ∧ z = 6 := 
sorry

end find_digits_l249_24954


namespace spinsters_count_l249_24939

theorem spinsters_count (S C : ℕ) (h1 : S / C = 2 / 9) (h2 : C = S + 42) : S = 12 := by
  sorry

end spinsters_count_l249_24939


namespace value_in_parentheses_l249_24926

theorem value_in_parentheses (x : ℝ) (h : x / Real.sqrt 18 = Real.sqrt 2) : x = 6 :=
sorry

end value_in_parentheses_l249_24926


namespace rectangle_length_l249_24936

theorem rectangle_length (P L W : ℕ) (h1 : P = 48) (h2 : L = 2 * W) (h3 : P = 2 * L + 2 * W) : L = 16 := by
  sorry

end rectangle_length_l249_24936


namespace kw_price_approx_4266_percent_l249_24988

noncomputable def kw_price_percentage (A B C D E : ℝ) (hA : KW = 1.5 * A) (hB : KW = 2 * B) (hC : KW = 2.5 * C) (hD : KW = 2.25 * D) (hE : KW = 3 * E) : ℝ :=
  let total_assets := A + B + C + D + E
  let price_kw := 1.5 * A
  (price_kw / total_assets) * 100

theorem kw_price_approx_4266_percent (A B C D E KW : ℝ)
  (hA : KW = 1.5 * A) (hB : KW = 2 * B) (hC : KW = 2.5 * C) (hD : KW = 2.25 * D) (hE : KW = 3 * E)
  (hB_from_A : B = 0.75 * A) (hC_from_A : C = 0.6 * A) (hD_from_A : D = 0.6667 * A) (hE_from_A : E = 0.5 * A) :
  abs ((kw_price_percentage A B C D E hA hB hC hD hE) - 42.66) < 1 :=
by sorry

end kw_price_approx_4266_percent_l249_24988


namespace employees_six_years_or_more_percentage_l249_24915

theorem employees_six_years_or_more_percentage 
  (Y : ℕ)
  (Total : ℝ := (3 * Y:ℝ) + (4 * Y:ℝ) + (7 * Y:ℝ) - (2 * Y:ℝ) + (6 * Y:ℝ) + (1 * Y:ℝ))
  (Employees_Six_Years : ℝ := (6 * Y:ℝ) + (1 * Y:ℝ))
  : Employees_Six_Years / Total * 100 = 36.84 :=
by
  sorry

end employees_six_years_or_more_percentage_l249_24915


namespace function_value_at_2018_l249_24910

theorem function_value_at_2018 (f : ℝ → ℝ)
  (h1 : f 4 = 2 - Real.sqrt 3)
  (h2 : ∀ x, f (x + 2) = 1 / (- f x)) :
  f 2018 = -2 - Real.sqrt 3 :=
by
  sorry

end function_value_at_2018_l249_24910


namespace y_pow_x_eq_x_pow_y_l249_24979

theorem y_pow_x_eq_x_pow_y (n : ℕ) (hn : 0 < n) :
    let x := (1 + 1 / (n : ℝ)) ^ n
    let y := (1 + 1 / (n : ℝ)) ^ (n + 1)
    y ^ x = x ^ y := 
    sorry

end y_pow_x_eq_x_pow_y_l249_24979


namespace train_length_is_360_l249_24963

-- Conditions from the problem
variable (speed_kmph : ℕ) (time_sec : ℕ) (platform_length_m : ℕ)

-- Definitions to be used for the conditions
def speed_ms (speed_kmph : ℕ) : ℤ := (speed_kmph * 1000) / 3600 -- Speed in m/s
def total_distance (speed_ms : ℤ) (time_sec : ℕ) : ℤ := speed_ms * (time_sec : ℤ) -- Total distance covered
def train_length (total_distance : ℤ) (platform_length : ℤ) : ℤ := total_distance - platform_length -- Length of the train

-- Assertion statement
theorem train_length_is_360 : train_length (total_distance (speed_ms speed_kmph) time_sec) platform_length_m = 360 := 
  by sorry

end train_length_is_360_l249_24963


namespace set_intersection_union_eq_complement_l249_24922

def A : Set ℝ := {x | 2 * x^2 + x - 3 = 0}
def B : Set ℝ := {i | i^2 ≥ 4}
def complement_C : Set ℝ := {-1, 1, 3/2}

theorem set_intersection_union_eq_complement :
  A ∩ B ∪ complement_C = complement_C :=
by
  sorry

end set_intersection_union_eq_complement_l249_24922


namespace max_value_of_function_neg_x_l249_24994

theorem max_value_of_function_neg_x (x : ℝ) (h : x < 0) : 
  ∃ y, (y = 2 * x + 2 / x) ∧ y ≤ -4 := sorry

end max_value_of_function_neg_x_l249_24994


namespace trip_time_l249_24945

theorem trip_time (distance half_dist speed1 speed2 : ℝ) 
  (h_distance : distance = 360) 
  (h_half_distance : half_dist = distance / 2) 
  (h_speed1 : speed1 = 50) 
  (h_speed2 : speed2 = 45) : 
  (half_dist / speed1 + half_dist / speed2) = 7.6 := 
by
  -- Simplify the expressions based on provided conditions
  sorry

end trip_time_l249_24945


namespace solution_set_inequality_l249_24919

theorem solution_set_inequality {a b c x : ℝ} (h1 : a < 0)
  (h2 : -b / a = 1 + 2) (h3 : c / a = 1 * 2) :
  a - c * (x^2 - x - 1) - b * x ≥ 0 ↔ x ≤ -3 / 2 ∨ x ≥ 1 := by
  sorry

end solution_set_inequality_l249_24919


namespace probability_of_green_ball_l249_24972

-- Definitions according to the conditions.
def containerA : ℕ × ℕ := (4, 6) -- 4 red balls, 6 green balls
def containerB : ℕ × ℕ := (6, 4) -- 6 red balls, 4 green balls
def containerC : ℕ × ℕ := (6, 4) -- 6 red balls, 4 green balls

-- Proving the probability of selecting a green ball.
theorem probability_of_green_ball :
  let pA := 1 / 3
  let pB := 1 / 3
  let pC := 1 / 3
  let pGreenA := (containerA.2 : ℚ) / (containerA.1 + containerA.2)
  let pGreenB := (containerB.2 : ℚ) / (containerB.1 + containerB.2)
  let pGreenC := (containerC.2 : ℚ) / (containerC.1 + containerC.2)
  pA * pGreenA + pB * pGreenB + pC * pGreenC = 7 / 15
  :=
by
  -- Formal proof will be filled in here.
  sorry

end probability_of_green_ball_l249_24972


namespace parabola_through_P_l249_24930

-- Define the point P
def P : ℝ × ℝ := (4, -2)

-- Define a condition function for equations y^2 = a*x
def satisfies_y_eq_ax (a : ℝ) : Prop := 
  ∃ x y, (x, y) = P ∧ y^2 = a * x

-- Define a condition function for equations x^2 = b*y
def satisfies_x_eq_by (b : ℝ) : Prop := 
  ∃ x y, (x, y) = P ∧ x^2 = b * y

-- Lean's theorem statement
theorem parabola_through_P : satisfies_y_eq_ax 1 ∨ satisfies_x_eq_by (-8) :=
sorry

end parabola_through_P_l249_24930


namespace combine_syllables_to_computer_l249_24928

/-- Conditions provided in the problem -/
def first_syllable : String := "ком" -- A big piece of a snowman
def second_syllable : String := "пьют" -- Something done by elephants at a watering hole
def third_syllable : String := "ер" -- The old name of the hard sign

/-- The result obtained by combining the three syllables should be "компьютер" -/
theorem combine_syllables_to_computer :
  (first_syllable ++ second_syllable ++ third_syllable) = "компьютер" :=
by
  -- Proof to be provided
  sorry

end combine_syllables_to_computer_l249_24928


namespace geometric_sequence_a4_l249_24914

theorem geometric_sequence_a4 {a : ℕ → ℝ} (q : ℝ) (h₁ : q > 0)
  (h₂ : ∀ n, a (n + 1) = a 1 * q ^ (n)) (h₃ : a 1 = 2) 
  (h₄ : a 2 + 4 = (a 1 + a 3) / 2) : a 4 = 54 := 
by
  sorry

end geometric_sequence_a4_l249_24914


namespace find_ray_solutions_l249_24924

noncomputable def polynomial (a x : ℝ) : ℝ :=
  x^3 - (a^2 + a + 1) * x^2 + (a^3 + a^2 + a) * x - a^3

theorem find_ray_solutions (a : ℝ) :
  (∀ x : ℝ, polynomial a x ≥ 0 → ∃ b : ℝ, ∀ y ≥ b, polynomial a y ≥ 0) ↔ a = 1 ∨ a = -1 :=
sorry

end find_ray_solutions_l249_24924


namespace value_of_x_l249_24990

theorem value_of_x (x y : ℝ) (h : x / (x - 1) = (y^2 + 2 * y + 3) / (y^2 + 2 * y + 2))  : 
  x = y^2 + 2 * y + 3 := 
by 
  sorry

end value_of_x_l249_24990


namespace scientific_notation_of_00000065_l249_24987

theorem scientific_notation_of_00000065:
  (6.5 * 10^(-7)) = 0.00000065 :=
by
  -- Proof goes here
  sorry

end scientific_notation_of_00000065_l249_24987


namespace solve_for_k_l249_24929

theorem solve_for_k (k : ℚ) : 
  (∃ x : ℚ, (3 * x - 6 = 0) ∧ (2 * x - 5 * k = 11)) → k = -7/5 :=
by 
  intro h
  cases' h with x hx
  have hx1 : x = 2 := by linarith
  have hx2 : x = 11 / 2 + 5 / 2 * k := by linarith
  linarith

end solve_for_k_l249_24929


namespace fraction_product_l249_24980

theorem fraction_product : (1 / 4 : ℚ) * (2 / 5) * (3 / 6) = (1 / 20) := by
  sorry

end fraction_product_l249_24980


namespace tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l249_24913

-- Conditions
variables {O : ℝ × ℝ} (A : ℝ × ℝ) (B : ℝ × ℝ)
          {P Q : ℝ × ℝ} (p : ℝ)
          (hp : 0 < p)
          (hA : A.1 ^ 2 = 2 * p * A.2)
          (hB : B = (0, -1))
          (hP : P.2 = P.1 ^ 2 / (2 * p))
          (hQ : Q.2 = Q.1 ^ 2 / (2 * p))

-- Proof problem statements
theorem tangent_line_AB
  (hAB_tangent : ∀ x : ℝ, x ^ 2 / (2 * p) = 2 * x - 1 → x = 1) : true :=
by sorry

theorem op_oq_leq_oa_squared 
  (h_op_oq_leq : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + (P.1 ^ 2 / (2 * p)) ^ 2) * (Q.1 ^ 2 + (Q.1 ^ 2 / (2 * p)) ^ 2) ≤ 2) : true :=
by sorry

theorem bp_bq_gt_ba_squared 
  ( h_bp_bq_gt : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + ((P.1 ^ 2 / (2 * p)) + 1) ^ 2) * (Q.1 ^ 2 + ((Q.1 ^ 2 / (2 * p)) +1 ) ^ 2) > 5 ) : true :=
by sorry

end tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l249_24913


namespace price_before_tax_l249_24992

theorem price_before_tax (P : ℝ) (h : 1.15 * P = 1955) : P = 1700 :=
by sorry

end price_before_tax_l249_24992


namespace complex_number_proof_l249_24906

open Complex

noncomputable def problem_complex (z : ℂ) (h1 : z ^ 7 = 1) (h2 : z ≠ 1) : ℂ :=
  (z - 1) * (z^2 - 1) * (z^3 - 1) * (z^4 - 1) * (z^5 - 1) * (z^6 - 1)

theorem complex_number_proof (z : ℂ) (h1 : z ^ 7 = 1) (h2 : z ≠ 1) :
  problem_complex z h1 h2 = 8 :=
  sorry

end complex_number_proof_l249_24906


namespace remainder_3_pow_89_plus_5_mod_7_l249_24949

theorem remainder_3_pow_89_plus_5_mod_7 :
  (3^1 % 7 = 3) ∧ (3^2 % 7 = 2) ∧ (3^3 % 7 = 6) ∧ (3^4 % 7 = 4) ∧ (3^5 % 7 = 5) ∧ (3^6 % 7 = 1) →
  ((3^89 + 5) % 7 = 3) :=
by
  intros h
  sorry

end remainder_3_pow_89_plus_5_mod_7_l249_24949


namespace k_value_if_perfect_square_l249_24923

theorem k_value_if_perfect_square (k : ℤ) (x : ℝ) (h : ∃ (a : ℝ), x^2 + k * x + 25 = a^2) : k = 10 ∨ k = -10 := by
  sorry

end k_value_if_perfect_square_l249_24923


namespace legs_total_l249_24946

def number_of_legs_bee := 6
def number_of_legs_spider := 8
def number_of_bees := 5
def number_of_spiders := 2
def total_legs := number_of_bees * number_of_legs_bee + number_of_spiders * number_of_legs_spider

theorem legs_total : total_legs = 46 := by
  sorry

end legs_total_l249_24946


namespace original_total_price_l249_24948

-- Definitions of the original prices
def original_price_candy_box : ℕ := 10
def original_price_soda : ℕ := 6
def original_price_chips : ℕ := 4
def original_price_chocolate_bar : ℕ := 2

-- Mathematical problem statement
theorem original_total_price :
  original_price_candy_box + original_price_soda + original_price_chips + original_price_chocolate_bar = 22 :=
by
  sorry

end original_total_price_l249_24948


namespace traders_fabric_sales_l249_24961

theorem traders_fabric_sales (x y : ℕ) : 
  x + y = 85 ∧
  x = y + 5 ∧
  60 = x * (60 / y) ∧
  30 = y * (30 / x) →
  (x, y) = (25, 20) :=
by {
  sorry
}

end traders_fabric_sales_l249_24961


namespace unique_positive_b_for_discriminant_zero_l249_24927

theorem unique_positive_b_for_discriminant_zero (c : ℝ) : 
  (∃! b : ℝ, b > 0 ∧ (b^2 + 1/b^2)^2 - 4 * c = 0) → c = 1 :=
by
  sorry

end unique_positive_b_for_discriminant_zero_l249_24927


namespace evaluate_expression_l249_24970

theorem evaluate_expression : 6^4 - 4 * 6^3 + 6^2 - 2 * 6 + 1 = 457 := 
  by
    sorry

end evaluate_expression_l249_24970


namespace num_integers_D_l249_24916

theorem num_integers_D :
  ∃ (D : ℝ) (n : ℕ), 
    (∀ (a b : ℝ), -1/4 < a → a < 1/4 → -1/4 < b → b < 1/4 → abs (a^2 - D * b^2) < 1) → n = 32 :=
sorry

end num_integers_D_l249_24916


namespace cyclic_quadrilateral_diameter_l249_24937

theorem cyclic_quadrilateral_diameter
  (AB BC CD DA : ℝ)
  (h1 : AB = 25)
  (h2 : BC = 39)
  (h3 : CD = 52)
  (h4 : DA = 60) : 
  ∃ D : ℝ, D = 65 :=
by
  sorry

end cyclic_quadrilateral_diameter_l249_24937


namespace new_car_travel_distance_l249_24902

-- Define the distance traveled by the older car
def distance_older_car : ℝ := 150

-- Define the percentage increase
def percentage_increase : ℝ := 0.30

-- Define the condition for the newer car's travel distance
def distance_newer_car (d_old : ℝ) (perc_inc : ℝ) : ℝ :=
  d_old * (1 + perc_inc)

-- Prove the main statement
theorem new_car_travel_distance :
  distance_newer_car distance_older_car percentage_increase = 195 := by
  -- Skip the proof body as instructed
  sorry

end new_car_travel_distance_l249_24902


namespace Bridget_Skittles_Final_l249_24944

def Bridget_Skittles_initial : Nat := 4
def Henry_Skittles_initial : Nat := 4

def Bridget_Receives_Henry_Skittles : Nat :=
  Bridget_Skittles_initial + Henry_Skittles_initial

theorem Bridget_Skittles_Final :
  Bridget_Receives_Henry_Skittles = 8 :=
by
  -- Proof steps here, assuming sorry for now
  sorry

end Bridget_Skittles_Final_l249_24944
