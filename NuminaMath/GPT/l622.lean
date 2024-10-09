import Mathlib

namespace anna_spent_more_on_lunch_l622_62237

def bagel_cost : ℝ := 0.95
def cream_cheese_cost : ℝ := 0.50
def orange_juice_cost : ℝ := 1.25
def orange_juice_discount : ℝ := 0.32
def sandwich_cost : ℝ := 4.65
def avocado_cost : ℝ := 0.75
def milk_cost : ℝ := 1.15
def milk_discount : ℝ := 0.10

-- Calculate total cost of breakfast.
def breakfast_cost : ℝ := 
  let bagel_with_cream_cheese := bagel_cost + cream_cheese_cost
  let discounted_orange_juice := orange_juice_cost - (orange_juice_cost * orange_juice_discount)
  bagel_with_cream_cheese + discounted_orange_juice

-- Calculate total cost of lunch.
def lunch_cost : ℝ :=
  let sandwich_with_avocado := sandwich_cost + avocado_cost
  let discounted_milk := milk_cost - (milk_cost * milk_discount)
  sandwich_with_avocado + discounted_milk

-- Calculate the difference between lunch and breakfast costs.
theorem anna_spent_more_on_lunch : lunch_cost - breakfast_cost = 4.14 := by
  sorry

end anna_spent_more_on_lunch_l622_62237


namespace find_parabola_focus_l622_62210

theorem find_parabola_focus : 
  ∀ (x y : ℝ), (y = 2 * x ^ 2 + 4 * x - 1) → (∃ p q : ℝ, p = -1 ∧ q = -(23:ℝ) / 8 ∧ (y = 2 * x ^ 2 + 4 * x - 1) → (x, y) = (p, q)) :=
by
  sorry

end find_parabola_focus_l622_62210


namespace trig_identity_solutions_l622_62207

open Real

theorem trig_identity_solutions (x : ℝ) (k n : ℤ) :
  (4 * sin x * cos (π / 2 - x) + 4 * sin (π + x) * cos x + 2 * sin (3 * π / 2 - x) * cos (π + x) = 1) ↔ 
  (∃ k : ℤ, x = arctan (1 / 3) + π * k) ∨ (∃ n : ℤ, x = π / 4 + π * n) := 
sorry

end trig_identity_solutions_l622_62207


namespace unique_shirt_and_tie_outfits_l622_62289

theorem unique_shirt_and_tie_outfits :
  let shirts := 10
  let ties := 8
  let choose n k := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose shirts 5 * choose ties 4 = 17640 :=
by
  sorry

end unique_shirt_and_tie_outfits_l622_62289


namespace canoe_kayak_problem_l622_62261

theorem canoe_kayak_problem (C K : ℕ) 
  (h1 : 9 * C + 12 * K = 432)
  (h2 : C = (4 * K) / 3) : 
  C - K = 6 := by
sorry

end canoe_kayak_problem_l622_62261


namespace selling_price_is_1260_l622_62239

-- Definitions based on conditions
def purchase_price : ℕ := 900
def repair_cost : ℕ := 300
def gain_percent : ℕ := 5 -- percentage as a natural number

-- Known variables
def total_cost : ℕ := purchase_price + repair_cost
def gain_amount : ℕ := (gain_percent * total_cost) / 100
def selling_price : ℕ := total_cost + gain_amount

-- The theorem we want to prove
theorem selling_price_is_1260 : selling_price = 1260 := by
  sorry

end selling_price_is_1260_l622_62239


namespace negation_of_universal_proposition_l622_62254

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) :=
by
  sorry

end negation_of_universal_proposition_l622_62254


namespace addition_example_l622_62236

theorem addition_example : 36 + 15 = 51 := 
by
  sorry

end addition_example_l622_62236


namespace intersection_P_Q_l622_62222

def P : Set ℝ := {x | Real.log x / Real.log 2 < -1}
def Q : Set ℝ := {x | abs x < 1}

theorem intersection_P_Q : P ∩ Q = {x | 0 < x ∧ x < 1 / 2} := by
  sorry

end intersection_P_Q_l622_62222


namespace find_number_l622_62284

noncomputable def question (x : ℝ) : Prop :=
  (2 * x^2 + Real.sqrt 6)^3 = 19683

theorem find_number : ∃ x : ℝ, question x ∧ (x = Real.sqrt ((27 - Real.sqrt 6) / 2) ∨ x = -Real.sqrt ((27 - Real.sqrt 6) / 2)) :=
  sorry

end find_number_l622_62284


namespace minimum_AP_BP_l622_62208

def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (7, 3)
def parabola (P : ℝ × ℝ) : Prop := P.2 * P.2 = 8 * P.1

noncomputable def distance (P Q : ℝ × ℝ) : ℝ := ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt

theorem minimum_AP_BP : 
  ∀ (P : ℝ × ℝ), parabola P → distance A P + distance B P ≥ 3 * Real.sqrt 10 :=
by 
  intros P hP
  sorry

end minimum_AP_BP_l622_62208


namespace part_a_part_b_l622_62248

noncomputable def probability_Peter_satisfied : ℚ :=
  let total_people := 100
  let men := 50
  let women := 50
  let P_both_men := (men - 1 : ℚ)/ (total_people - 1 : ℚ) * (men - 2 : ℚ)/ (total_people - 2 : ℚ)
  1 - P_both_men

theorem part_a : probability_Peter_satisfied = 25 / 33 := 
  sorry

noncomputable def expected_satisfied_men : ℚ :=
  let men := 50
  probability_Peter_satisfied * men

theorem part_b : expected_satisfied_men = 1250 / 33 := 
  sorry

end part_a_part_b_l622_62248


namespace tree_height_by_time_boy_is_36_inches_l622_62298

noncomputable def final_tree_height : ℕ :=
  let T₀ := 16
  let B₀ := 24
  let Bₓ := 36
  let boy_growth := Bₓ - B₀
  let tree_growth := 2 * boy_growth
  T₀ + tree_growth

theorem tree_height_by_time_boy_is_36_inches :
  final_tree_height = 40 :=
by
  sorry

end tree_height_by_time_boy_is_36_inches_l622_62298


namespace no_such_function_exists_l622_62256

noncomputable def func_a (a : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧ ∀ n : ℕ, a n = n - a (a n)

theorem no_such_function_exists : ¬ ∃ a : ℕ → ℕ, func_a a :=
by
  sorry

end no_such_function_exists_l622_62256


namespace correct_answer_l622_62295

-- Statement of the problem
theorem correct_answer :
  ∃ (answer : String),
    (answer = "long before" ∨ answer = "before long" ∨ answer = "soon after" ∨ answer = "shortly after") ∧
    answer = "long before" :=
by
  sorry

end correct_answer_l622_62295


namespace find_number_l622_62233

theorem find_number (x : ℝ) : 50 + (x * 12) / (180 / 3) = 51 ↔ x = 5 := by
  sorry

end find_number_l622_62233


namespace range_of_a_l622_62229

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (Real.exp x / x) - a * (x ^ 2)

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → (f a x1 / x2) - (f a x2 / x1) < 0) ↔ (a ≤ Real.exp 2 / 12) := by
  sorry

end range_of_a_l622_62229


namespace range_of_a_l622_62265

noncomputable def A : Set ℝ := {x : ℝ | ((x^2) - x - 2) ≤ 0}

theorem range_of_a (a : ℝ) : (∀ x ∈ A, (x^2 - a*x - a - 2) ≤ 0) → a ≥ (2/3) :=
by
  intro h
  sorry

end range_of_a_l622_62265


namespace votes_cast_l622_62266

theorem votes_cast (total_votes : ℕ) 
  (h1 : (3/8 : ℚ) * total_votes = 45)
  (h2 : (1/4 : ℚ) * total_votes = (1/4 : ℚ) * 120) : 
  total_votes = 120 := 
by
  sorry

end votes_cast_l622_62266


namespace lines_intersection_l622_62270

/-- Two lines are defined by the equations y = 2x + c and y = 4x + d.
These lines intersect at the point (8, 12).
Prove that c + d = -24. -/
theorem lines_intersection (c d : ℝ) (h1 : 12 = 2 * 8 + c) (h2 : 12 = 4 * 8 + d) :
    c + d = -24 :=
by
  sorry

end lines_intersection_l622_62270


namespace max_value_of_y_l622_62262

open Real

noncomputable def y (x : ℝ) : ℝ := 
  (sin (π / 4 + x) - sin (π / 4 - x)) * sin (π / 3 + x)

theorem max_value_of_y : 
  ∃ x : ℝ, (∀ x, y x ≤ 3 * sqrt 2 / 4) ∧ (∀ k : ℤ, x = k * π + π / 3 → y x = 3 * sqrt 2 / 4) :=
sorry

end max_value_of_y_l622_62262


namespace faster_speed_l622_62223

variable (v : ℝ)
variable (distance fasterDistance speed time : ℝ)
variable (h_distance : distance = 24)
variable (h_speed : speed = 4)
variable (h_fasterDistance : fasterDistance = distance + 6)
variable (h_time : time = distance / speed)

theorem faster_speed (h : 6 = fasterDistance / v) : v = 5 :=
by
  sorry

end faster_speed_l622_62223


namespace verify_a_eq_x0_verify_p_squared_ge_4x0q_l622_62240

theorem verify_a_eq_x0 (p q x0 a b : ℝ) (hx0_root : x0^3 + p * x0 + q = 0) 
  (h_eq : ∀ x : ℝ, x^3 + p * x + q = (x - x0) * (x^2 + a * x + b)) : 
  a = x0 :=
by
  sorry

theorem verify_p_squared_ge_4x0q (p q x0 b : ℝ) (hx0_root : x0^3 + p * x0 + q = 0) 
  (h_eq : ∀ x : ℝ, x^3 + p * x + q = (x - x0) * (x^2 + x0 * x + b)) : 
  p^2 ≥ 4 * x0 * q :=
by
  sorry

end verify_a_eq_x0_verify_p_squared_ge_4x0q_l622_62240


namespace neg_of_exists_a_l622_62250

theorem neg_of_exists_a (a : ℝ) : ¬ (∃ a : ℝ, a^2 + 1 < 2 * a) :=
by
  sorry

end neg_of_exists_a_l622_62250


namespace linear_func_3_5_l622_62234

def linear_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem linear_func_3_5 (f : ℝ → ℝ) (h_linear: linear_function f) 
  (h_diff: ∀ d : ℝ, f (d + 1) - f d = 3) : f 3 - f 5 = -6 :=
by
  sorry

end linear_func_3_5_l622_62234


namespace initial_population_l622_62259

theorem initial_population (P : ℝ) 
  (h1 : P * 0.90 * 0.95 * 0.85 * 1.08 = 6514) : P = 8300 :=
by
  -- Given conditions lead to the final population being 6514
  -- We need to show that the initial population P was 8300
  sorry

end initial_population_l622_62259


namespace percent_in_range_70_to_79_is_correct_l622_62264

-- Define the total number of students.
def total_students : Nat := 8 + 12 + 11 + 5 + 7

-- Define the number of students within the $70\%-79\%$ range.
def students_70_to_79 : Nat := 11

-- Define the percentage of the students within the $70\%-79\%$ range.
def percent_70_to_79 : ℚ := (students_70_to_79 : ℚ) / (total_students : ℚ) * 100

theorem percent_in_range_70_to_79_is_correct : percent_70_to_79 = 25.58 := by
  sorry

end percent_in_range_70_to_79_is_correct_l622_62264


namespace work_completion_time_l622_62228

noncomputable def work_done_by_woman_per_day : ℝ := 1 / 50
noncomputable def work_done_by_child_per_day : ℝ := 1 / 100
noncomputable def total_work_done_by_5_women_per_day : ℝ := 5 * work_done_by_woman_per_day
noncomputable def total_work_done_by_10_children_per_day : ℝ := 10 * work_done_by_child_per_day
noncomputable def combined_work_per_day : ℝ := total_work_done_by_5_women_per_day + total_work_done_by_10_children_per_day

theorem work_completion_time (h1 : 10 / 5 = 2) (h2 : 10 / 10 = 1) :
  1 / combined_work_per_day = 5 :=
by
  sorry

end work_completion_time_l622_62228


namespace selection_plans_count_l622_62246

-- Define the number of boys and girls
def num_boys : ℕ := 3
def num_girls : ℕ := 3
def total_students : ℕ := num_boys + num_girls

-- Define the number of subjects
def num_subjects : ℕ := 3

-- Prove that the number of selection plans is 120
theorem selection_plans_count :
  (Nat.choose total_students num_subjects) * (num_subjects.factorial) = 120 := 
by
  sorry

end selection_plans_count_l622_62246


namespace find_a_find_distance_l622_62201

-- Problem 1: Given conditions to find 'a'
theorem find_a (a : ℝ) :
  (∃ θ ρ, ρ = 2 * Real.cos θ ∧ 3 * ρ * Real.cos θ + 4 * ρ * Real.sin θ + a = 0) →
  (a = 2 ∨ a = -8) :=
sorry

-- Problem 2: Given point and line, find the distance
theorem find_distance : 
  ∃ (d : ℝ), d = Real.sqrt 3 + 5/2 ∧
  (∃ θ ρ, θ = 11 * Real.pi / 6 ∧ ρ = 2 ∧ 
   (ρ = Real.sqrt (3 * (Real.sin θ - Real.pi / 6)^2 + (ρ * Real.cos (θ - Real.pi / 6))^2) 
   → ρ * Real.sin (θ - Real.pi / 6) = 1)) :=
sorry

end find_a_find_distance_l622_62201


namespace solution_l622_62252

def p : Prop := ∀ x > 0, Real.log (x + 1) > 0
def q : Prop := ∀ a b : ℝ, a > b → a^2 > b^2

theorem solution : p ∧ ¬ q := by
  sorry

end solution_l622_62252


namespace fraction_greater_than_decimal_l622_62244

/-- 
  Prove that the fraction 1/3 is greater than the decimal 0.333 by the amount 1/(3 * 10^3)
-/
theorem fraction_greater_than_decimal :
  (1 / 3 : ℚ) = (333 / 1000 : ℚ) + (1 / (3 * 1000) : ℚ) :=
by
  sorry

end fraction_greater_than_decimal_l622_62244


namespace expression_parity_l622_62231

theorem expression_parity (a b c : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_a_odd : a % 2 = 1) (h_b_odd : b % 2 = 1) : (3^a + (b + 1)^2 * c) % 2 = 1 :=
by sorry

end expression_parity_l622_62231


namespace mikes_ride_is_46_miles_l622_62206

-- Define the conditions and the question in Lean 4
variable (M : ℕ)

-- Mike's cost formula
def mikes_cost (M : ℕ) : ℚ := 2.50 + 0.25 * M

-- Annie's total cost
def annies_miles : ℕ := 26
def annies_cost : ℚ := 2.50 + 5.00 + 0.25 * annies_miles

-- The proof statement
theorem mikes_ride_is_46_miles (h : mikes_cost M = annies_cost) : M = 46 :=
by sorry

end mikes_ride_is_46_miles_l622_62206


namespace existence_of_solution_values_continuous_solution_value_l622_62272

noncomputable def functional_equation_has_solution (a : ℝ) (f : ℝ → ℝ) : Prop :=
  f 0 = 0 ∧ f 1 = 1 ∧ ∀ x y, (x ≤ y → f ((x + y) / 2) = (1 - a) * f x + a * f y)

theorem existence_of_solution_values :
  {a : ℝ | ∃ f : ℝ → ℝ, functional_equation_has_solution a f} = {0, 1/2, 1} :=
sorry

theorem continuous_solution_value :
  {a : ℝ | ∃ (f : ℝ → ℝ) (hf : Continuous f), functional_equation_has_solution a f} = {1/2} :=
sorry

end existence_of_solution_values_continuous_solution_value_l622_62272


namespace relationship_between_a_and_b_l622_62268

def a : ℤ := (-12) * (-23) * (-34) * (-45)
def b : ℤ := (-123) * (-234) * (-345)

theorem relationship_between_a_and_b : a > b := by
  sorry

end relationship_between_a_and_b_l622_62268


namespace sum_of_roots_l622_62200

theorem sum_of_roots (x : ℝ) (h : x + 49 / x = 14) : x + x = 14 :=
sorry

end sum_of_roots_l622_62200


namespace shaded_shape_area_l622_62274

/-- Define the coordinates and the conditions for the central square and triangles in the grid -/
def grid_size := 10
def central_square_side := 2
def central_square_area := central_square_side * central_square_side

def triangle_base := 5
def triangle_height := 5
def triangle_area := (1 / 2) * triangle_base * triangle_height

def number_of_triangles := 4
def total_triangle_area := number_of_triangles * triangle_area

def total_shaded_area := total_triangle_area + central_square_area

theorem shaded_shape_area : total_shaded_area = 54 :=
by
  -- We have defined each area component and summed them to the total shaded area.
  -- The statement ensures that the area of the shaded shape is equal to 54.
  sorry

end shaded_shape_area_l622_62274


namespace cannot_divide_m_l622_62247

/-
  A proof that for the real number m = 2009^3 - 2009, 
  the number 2007 does not divide m.
-/

theorem cannot_divide_m (m : ℤ) (h : m = 2009^3 - 2009) : ¬ (2007 ∣ m) := 
by sorry

end cannot_divide_m_l622_62247


namespace valid_sequences_length_21_l622_62275

def valid_sequences (n : ℕ) : ℕ :=
  if n < 3 then 0
  else if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else valid_sequences (n - 3) + valid_sequences (n - 4)

theorem valid_sequences_length_21 : valid_sequences 21 = 38 :=
by
  sorry

end valid_sequences_length_21_l622_62275


namespace position_1011th_square_l622_62253

-- Define the initial position and transformations
inductive SquarePosition
| ABCD : SquarePosition
| DABC : SquarePosition
| BADC : SquarePosition
| DCBA : SquarePosition

open SquarePosition

def R1 (p : SquarePosition) : SquarePosition :=
  match p with
  | ABCD => DABC
  | DABC => BADC
  | BADC => DCBA
  | DCBA => ABCD

def R2 (p : SquarePosition) : SquarePosition :=
  match p with
  | ABCD => DCBA
  | DCBA => ABCD
  | DABC => BADC
  | BADC => DABC

def transform : ℕ → SquarePosition
| 0 => ABCD
| n + 1 => if n % 2 = 0 then R1 (transform n) else R2 (transform n)

theorem position_1011th_square : transform 1011 = DCBA :=
by {
  sorry
}

end position_1011th_square_l622_62253


namespace trader_gain_percentage_l622_62280

theorem trader_gain_percentage (C : ℝ) (h1 : 95 * C = (95 * C - cost_of_95_pens) + (19 * C)) :
  100 * (19 * C / (95 * C)) = 20 := 
by {
  sorry
}

end trader_gain_percentage_l622_62280


namespace part1_l622_62245

theorem part1 (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, x > 0 → f x < 0) :
  a > 1 :=
sorry

end part1_l622_62245


namespace M_sufficient_not_necessary_for_N_l622_62299

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x < 2}

theorem M_sufficient_not_necessary_for_N (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (a ∈ N → ¬ (a ∈ M)) :=
sorry

end M_sufficient_not_necessary_for_N_l622_62299


namespace spring_spending_l622_62277

theorem spring_spending (end_of_feb : ℝ) (end_of_may : ℝ) (h_end_of_feb : end_of_feb = 0.8) (h_end_of_may : end_of_may = 2.5)
  : (end_of_may - end_of_feb) = 1.7 :=
by
  have spending_end_of_feb : end_of_feb = 0.8 := h_end_of_feb
  have spending_end_of_may : end_of_may = 2.5 := h_end_of_may
  sorry

end spring_spending_l622_62277


namespace vector_sum_correct_l622_62211

def vec1 : Fin 3 → ℤ := ![-7, 3, 5]
def vec2 : Fin 3 → ℤ := ![4, -1, -6]
def vec3 : Fin 3 → ℤ := ![1, 8, 2]
def expectedSum : Fin 3 → ℤ := ![-2, 10, 1]

theorem vector_sum_correct :
  (fun i => vec1 i + vec2 i + vec3 i) = expectedSum := 
by
  sorry

end vector_sum_correct_l622_62211


namespace average_speed_l622_62242

theorem average_speed (v1 v2 t1 t2 total_time total_distance : ℝ)
  (h1 : v1 = 50)
  (h2 : t1 = 4)
  (h3 : v2 = 80)
  (h4 : t2 = 4)
  (h5 : total_time = t1 + t2)
  (h6 : total_distance = v1 * t1 + v2 * t2) :
  (total_distance / total_time = 65) :=
by
  sorry

end average_speed_l622_62242


namespace find_m_value_l622_62286

-- Definitions based on conditions
variables {a b m : ℝ} (ha : 2 ^ a = m) (hb : 5 ^ b = m) (h : 1 / a + 1 / b = 1)

-- Lean 4 statement of the problem
theorem find_m_value (ha : 2 ^ a = m) (hb : 5 ^ b = m) (h : 1 / a + 1 / b = 1) : m = 10 := sorry

end find_m_value_l622_62286


namespace turnip_difference_l622_62271

theorem turnip_difference :
  let melanie_turnips := 139
  let benny_turnips := 113
  let caroline_turnips := 172
  (melanie_turnips + benny_turnips) - caroline_turnips = 80 :=
by
  let melanie_turnips := 139
  let benny_turnips := 113
  let caroline_turnips := 172
  show (melanie_turnips + benny_turnips) - caroline_turnips = 80
  sorry

end turnip_difference_l622_62271


namespace yearly_savings_l622_62203

-- Define the various constants given in the problem
def weeks_in_year : ℕ := 52
def months_in_year : ℕ := 12
def non_peak_weeks : ℕ := 16
def peak_weeks : ℕ := weeks_in_year - non_peak_weeks
def non_peak_months : ℕ := 4
def peak_months : ℕ := months_in_year - non_peak_months

-- Rates
def weekly_cost_non_peak_large : ℕ := 10
def weekly_cost_peak_large : ℕ := 12
def monthly_cost_non_peak_large : ℕ := 42
def monthly_cost_peak_large : ℕ := 48

-- Additional surcharge
def holiday_weeks : ℕ := 6
def holiday_surcharge : ℕ := 2

-- Compute the yearly costs
def yearly_weekly_cost : ℕ :=
  (non_peak_weeks * weekly_cost_non_peak_large) +
  (peak_weeks * weekly_cost_peak_large) +
  (holiday_weeks * (holiday_surcharge + weekly_cost_peak_large))

def yearly_monthly_cost : ℕ :=
  (non_peak_months * monthly_cost_non_peak_large) +
  (peak_months * monthly_cost_peak_large)

theorem yearly_savings : yearly_weekly_cost - yearly_monthly_cost = 124 := by
  sorry

end yearly_savings_l622_62203


namespace sufficient_but_not_necessary_not_necessary_l622_62219

variable (x y : ℝ)

theorem sufficient_but_not_necessary (h1: x ≥ 2) (h2: y ≥ 2): x^2 + y^2 ≥ 4 :=
by
  sorry

theorem not_necessary (hx4 : x^2 + y^2 ≥ 4) : ¬ (x ≥ 2 ∧ y ≥ 2) → ∃ x y, (x^2 + y^2 ≥ 4) ∧ (¬ (x ≥ 2) ∨ ¬ (y ≥ 2)) :=
by
  sorry

end sufficient_but_not_necessary_not_necessary_l622_62219


namespace movie_ticket_ratio_l622_62230

-- Definitions based on the conditions
def monday_cost : ℕ := 5
def wednesday_cost : ℕ := 2 * monday_cost

theorem movie_ticket_ratio (S : ℕ) (h1 : wednesday_cost + S = 35) :
  S / monday_cost = 5 :=
by
  -- Placeholder for proof
  sorry

end movie_ticket_ratio_l622_62230


namespace necessary_but_not_sufficient_l622_62238

-- Define sets M and N
def M (x : ℝ) : Prop := x < 5
def N (x : ℝ) : Prop := x > 3

-- Define the union and intersection of M and N
def M_union_N (x : ℝ) : Prop := M x ∨ N x
def M_inter_N (x : ℝ) : Prop := M x ∧ N x

-- Theorem statement: Prove the necessity but not sufficiency
theorem necessary_but_not_sufficient (x : ℝ) :
  M_inter_N x → M_union_N x ∧ ¬(M_union_N x → M_inter_N x) := 
sorry

end necessary_but_not_sufficient_l622_62238


namespace part_a_part_b_l622_62243

-- Define the problem as described
noncomputable def can_transform_to_square (figure : Type) (parts : ℕ) (all_triangles : Bool) : Bool :=
sorry  -- This is a placeholder for the actual implementation

-- The figure satisfies the condition to cut into four parts and rearrange into a square
theorem part_a (figure : Type) : can_transform_to_square figure 4 false = true :=
sorry

-- The figure satisfies the condition to cut into five triangular parts and rearrange into a square
theorem part_b (figure : Type) : can_transform_to_square figure 5 true = true :=
sorry

end part_a_part_b_l622_62243


namespace molecular_weight_of_compound_l622_62235

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

end molecular_weight_of_compound_l622_62235


namespace arccos_pi_over_3_l622_62282

theorem arccos_pi_over_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_pi_over_3_l622_62282


namespace train_length_l622_62255

-- Defining the conditions
def speed_kmh : ℕ := 64
def speed_m_per_s : ℚ := (64 * 1000) / 3600 -- 64 km/h converted to m/s
def time_to_cross_seconds : ℕ := 9 

-- The theorem to prove the length of the train
theorem train_length : speed_m_per_s * time_to_cross_seconds = 160 := 
by 
  unfold speed_m_per_s 
  norm_num
  sorry -- Placeholder for actual proof

end train_length_l622_62255


namespace xyz_inequality_l622_62294

theorem xyz_inequality (x y z : ℝ) (h_condition : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := 
sorry

end xyz_inequality_l622_62294


namespace ten_person_round_robin_l622_62212

def number_of_matches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem ten_person_round_robin : number_of_matches 10 = 45 :=
by
  -- Proof steps would go here, but are omitted for this task
  sorry

end ten_person_round_robin_l622_62212


namespace true_proposition_l622_62214

noncomputable def prop_p (x : ℝ) : Prop := x > 0 → x^2 - 2*x + 1 > 0

noncomputable def prop_q (x₀ : ℝ) : Prop := x₀ > 0 ∧ x₀^2 - 2*x₀ + 1 ≤ 0

theorem true_proposition : ¬ (∀ x > 0, x^2 - 2*x + 1 > 0) ∧ (∃ x₀ > 0, x₀^2 - 2*x₀ + 1 ≤ 0) :=
by
  sorry

end true_proposition_l622_62214


namespace sum_of_angles_l622_62213

theorem sum_of_angles : 
    ∀ (angle1 angle3 angle5 angle2 angle4 angle6 angleA angleB angleC : ℝ),
    angle1 + angle3 + angle5 = 180 ∧
    angle2 + angle4 + angle6 = 180 ∧
    angleA + angleB + angleC = 180 →
    angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angleA + angleB + angleC = 540 :=
by
  intro angle1 angle3 angle5 angle2 angle4 angle6 angleA angleB angleC
  intro h
  sorry

end sum_of_angles_l622_62213


namespace jessica_journey_total_distance_l622_62216

theorem jessica_journey_total_distance
  (y : ℝ)
  (h1 : y = (y / 4) + 25 + (y / 4)) :
  y = 50 :=
by
  sorry

end jessica_journey_total_distance_l622_62216


namespace evaluate_expression_l622_62232

theorem evaluate_expression : 2^(3^2) + 3^(2^3) = 7073 := by
  sorry

end evaluate_expression_l622_62232


namespace sym_diff_A_B_l622_62220

def set_diff (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def sym_diff (M N : Set ℝ) : Set ℝ := set_diff M N ∪ set_diff N M

def A : Set ℝ := {x | -1 ≤ x ∧ x < 1}
def B : Set ℝ := {x | x < 0}

theorem sym_diff_A_B :
  sym_diff A B = {x | x < -1} ∪ {x | 0 ≤ x ∧ x < 1} := by
  sorry

end sym_diff_A_B_l622_62220


namespace distance_traveled_on_second_day_l622_62209

theorem distance_traveled_on_second_day 
  (a₁ : ℝ) 
  (h_sum : a₁ + a₁ / 2 + a₁ / 4 + a₁ / 8 + a₁ / 16 + a₁ / 32 = 189) 
  : a₁ / 2 = 48 :=
by
  sorry

end distance_traveled_on_second_day_l622_62209


namespace find_number_l622_62241

theorem find_number (x : ℤ) (h : x + x^2 = 342) : x = 18 ∨ x = -19 :=
sorry

end find_number_l622_62241


namespace solve_for_a_l622_62291

theorem solve_for_a (a : ℤ) : -2 - a = 0 → a = -2 :=
by
  sorry

end solve_for_a_l622_62291


namespace problem1_problem2_l622_62293

-- Definition of sets A and B
def A : Set ℝ := { x | x^2 - 2*x - 3 < 0 }
def B (p : ℝ) : Set ℝ := { x | abs (x - p) > 1 }

-- Statement for the first problem
theorem problem1 : B 0 ∩ A = { x | 1 < x ∧ x < 3 } := 
by
  sorry

-- Statement for the second problem
theorem problem2 (p : ℝ) (h : A ∪ B p = B p) : p ≤ -2 ∨ p ≥ 4 := 
by
  sorry

end problem1_problem2_l622_62293


namespace carly_shipping_cost_l622_62205

noncomputable def total_shipping_cost (flat_fee cost_per_pound weight : ℝ) : ℝ :=
flat_fee + cost_per_pound * weight

theorem carly_shipping_cost : 
  total_shipping_cost 5 0.80 5 = 9 :=
by 
  unfold total_shipping_cost
  norm_num

end carly_shipping_cost_l622_62205


namespace nesbitts_inequality_l622_62278

theorem nesbitts_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ (3 / 2) :=
sorry

end nesbitts_inequality_l622_62278


namespace sqrt_sum_simplification_l622_62226

theorem sqrt_sum_simplification :
  (Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3)) = 2 * Real.sqrt 6 :=
by
    sorry

end sqrt_sum_simplification_l622_62226


namespace originally_anticipated_profit_margin_l622_62224

theorem originally_anticipated_profit_margin (decrease_percent increase_percent : ℝ) (original_price current_price : ℝ) (selling_price : ℝ) :
  decrease_percent = 6.4 → 
  increase_percent = 8 → 
  original_price = 1 → 
  current_price = original_price - original_price * decrease_percent / 100 → 
  selling_price = original_price * (1 + x / 100) → 
  selling_price = current_price * (1 + (x + increase_percent) / 100) →
  x = 117 :=
by
  intros h_dec_perc h_inc_perc h_org_price h_cur_price h_selling_price_orig h_selling_price_cur
  sorry

end originally_anticipated_profit_margin_l622_62224


namespace sum_of_midpoints_l622_62283

theorem sum_of_midpoints (p q r : ℝ) (h : p + q + r = 15) :
  (p + q) / 2 + (p + r) / 2 + (q + r) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_l622_62283


namespace triangle_ABC_area_l622_62292

def point : Type := ℚ × ℚ

def triangle_area (A B C : point) : ℚ :=
  let (x1, y1) := A;
  let (x2, y2) := B;
  let (x3, y3) := C;
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_ABC_area :
  let A : point := (-5, 4)
  let B : point := (1, 7)
  let C : point := (4, -3)
  triangle_area A B C = 34.5 :=
by
  sorry

end triangle_ABC_area_l622_62292


namespace sam_drove_200_miles_l622_62297

-- Define the conditions
def marguerite_distance : ℕ := 150
def marguerite_time : ℕ := 3
def sam_time : ℕ := 4
def marguerite_speed := marguerite_distance / marguerite_time

-- Define the question
def sam_distance (speed : ℕ) (time : ℕ) : ℕ := speed * time

-- State the theorem to prove the answer
theorem sam_drove_200_miles :
  sam_distance marguerite_speed sam_time = 200 := by
  sorry

end sam_drove_200_miles_l622_62297


namespace machine_output_l622_62290

theorem machine_output (input : ℕ) (output : ℕ) (h : input = 26) (h_out : output = input + 15 - 6) : output = 35 := 
by 
  sorry

end machine_output_l622_62290


namespace find_x_l622_62260

variable (P T S : Point)
variable (angle_PTS angle_TSR x : ℝ)
variable (reflector : Point)

-- Given conditions
axiom angle_PTS_is_90 : angle_PTS = 90
axiom angle_TSR_is_26 : angle_TSR = 26

-- Proof problem
theorem find_x : x = 32 := by
  sorry

end find_x_l622_62260


namespace total_drawing_sheets_l622_62204

-- Definitions based on the conditions given
def brown_sheets := 28
def yellow_sheets := 27

-- The statement we need to prove
theorem total_drawing_sheets : brown_sheets + yellow_sheets = 55 := by
  sorry

end total_drawing_sheets_l622_62204


namespace tens_digit_of_large_power_l622_62288

theorem tens_digit_of_large_power : ∃ a : ℕ, a = 2 ∧ ∀ n ≥ 2, (5 ^ n) % 100 = 25 :=
by
  sorry

end tens_digit_of_large_power_l622_62288


namespace find_x_in_terms_of_N_l622_62276

theorem find_x_in_terms_of_N (N : ℤ) (x y : ℝ) 
(h1 : (⌊x⌋ : ℤ) + 2 * y = N + 2) 
(h2 : (⌊y⌋ : ℤ) + 2 * x = 3 - N) : 
x = (3 / 2) - N := 
by
  sorry

end find_x_in_terms_of_N_l622_62276


namespace intersection_of_A_and_B_l622_62221

def A : Set ℝ := { x | x^2 - 5 * x - 6 ≤ 0 }

def B : Set ℝ := { x | x < 4 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | -1 ≤ x ∧ x < 4 } :=
sorry

end intersection_of_A_and_B_l622_62221


namespace percentage_25_of_200_l622_62296

def percentage_of (percent : ℝ) (amount : ℝ) : ℝ := percent * amount

theorem percentage_25_of_200 :
  percentage_of 0.25 200 = 50 :=
by sorry

end percentage_25_of_200_l622_62296


namespace John_can_finish_work_alone_in_48_days_l622_62273

noncomputable def John_and_Roger_can_finish_together_in_24_days (J R: ℝ) : Prop :=
  1 / J + 1 / R = 1 / 24

noncomputable def John_finished_remaining_work (J: ℝ) : Prop :=
  (1 / 3) / (16 / J) = 1

theorem John_can_finish_work_alone_in_48_days (J R: ℝ) 
  (h1 : John_and_Roger_can_finish_together_in_24_days J R) 
  (h2 : John_finished_remaining_work J):
  J = 48 := 
sorry

end John_can_finish_work_alone_in_48_days_l622_62273


namespace total_students_l622_62258

-- Definitions based on conditions
variable (T M Z : ℕ)  -- T for Tina's students, M for Maura's students, Z for Zack's students

-- Conditions as hypotheses
axiom h1 : T = M  -- Tina's classroom has the same amount of students as Maura's
axiom h2 : Z = (T + M) / 2  -- Zack's classroom has half the amount of total students between Tina and Maura's classrooms
axiom h3 : Z = 23  -- There are 23 students in Zack's class when present

-- Proof statement
theorem total_students : T + M + Z = 69 :=
  sorry

end total_students_l622_62258


namespace center_of_circle_l622_62225

theorem center_of_circle : ∃ c : ℝ × ℝ, (∀ x y : ℝ, (x^2 + y^2 - 2*x + 4*y + 3 = 0 ↔ ((x - c.1)^2 + (y + c.2)^2 = 2))) ∧ (c = (1, -2)) :=
by
  -- Proof is omitted
  sorry

end center_of_circle_l622_62225


namespace crop_yield_growth_l622_62279

-- Definitions based on conditions
def initial_yield := 300
def final_yield := 363
def eqn (x : ℝ) : Prop := initial_yield * (1 + x)^2 = final_yield

-- The theorem we need to prove
theorem crop_yield_growth (x : ℝ) : eqn x :=
by
  sorry

end crop_yield_growth_l622_62279


namespace minor_axis_length_is_2sqrt3_l622_62251

-- Define the points given in the problem
def points : List (ℝ × ℝ) := [(1, 1), (0, 0), (0, 3), (4, 0), (4, 3)]

-- Define a function that checks if an ellipse with axes parallel to the coordinate axes
-- passes through given points, and returns the length of its minor axis if it does.
noncomputable def minor_axis_length (pts : List (ℝ × ℝ)) : ℝ :=
  if h : (0,0) ∈ pts ∧ (0,3) ∈ pts ∧ (4,0) ∈ pts ∧ (4,3) ∈ pts ∧ (1,1) ∈ pts then
    let a := (4 - 0) / 2 -- half the width of the rectangle
    let b_sq := 3 -- derived from solving the ellipse equation
    2 * Real.sqrt b_sq
  else 0

-- The theorem statement:
theorem minor_axis_length_is_2sqrt3 : minor_axis_length points = 2 * Real.sqrt 3 := by
  sorry

end minor_axis_length_is_2sqrt3_l622_62251


namespace barium_oxide_moles_l622_62281

noncomputable def moles_of_bao_needed (mass_H2O : ℝ) (molar_mass_H2O : ℝ) : ℝ :=
  mass_H2O / molar_mass_H2O

theorem barium_oxide_moles :
  moles_of_bao_needed 54 18.015 = 3 :=
by
  unfold moles_of_bao_needed
  norm_num
  sorry

end barium_oxide_moles_l622_62281


namespace base6_addition_l622_62285

/-- Adding two numbers in base 6 -/
theorem base6_addition : (3454 : ℕ) + (12345 : ℕ) = (142042 : ℕ) := by
  sorry

end base6_addition_l622_62285


namespace hungarian_math_olympiad_1927_l622_62269

-- Definitions
def is_coprime (a b : ℤ) : Prop :=
  Int.gcd a b = 1

-- The main statement
theorem hungarian_math_olympiad_1927
  (a b c d x y k m : ℤ) 
  (h_coprime : is_coprime a b)
  (h_m : m = a * d - b * c)
  (h_divides : m ∣ (a * x + b * y)) :
  m ∣ (c * x + d * y) :=
sorry

end hungarian_math_olympiad_1927_l622_62269


namespace sequence_a_n_l622_62227

theorem sequence_a_n (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n : ℕ, S n = 3 + 2^n) →
  (a 1 = 5) ∧ (∀ n : ℕ, n ≥ 2 → a n = 2^(n-1)) ↔ 
  (∀ n : ℕ, a n = if n = 1 then 5 else 2^(n-1)) :=
by
  sorry

end sequence_a_n_l622_62227


namespace sum_of_consecutive_numbers_mod_13_l622_62257

theorem sum_of_consecutive_numbers_mod_13 :
  ((8930 + 8931 + 8932 + 8933 + 8934) % 13) = 5 :=
by
  sorry

end sum_of_consecutive_numbers_mod_13_l622_62257


namespace determine_a_range_l622_62263

variable (a : ℝ)

-- Define proposition p as a function
def p : Prop := ∀ x : ℝ, x^2 + x > a

-- Negation of Proposition q
def not_q : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 2 - a ≠ 0

-- The main theorem to be stated, proving the range of 'a'
theorem determine_a_range (h₁ : p a) (h₂ : not_q a) : -2 < a ∧ a < -1 / 4 := sorry

end determine_a_range_l622_62263


namespace probability_of_yellow_or_green_l622_62218

def bag : List (String × Nat) := [("yellow", 4), ("green", 3), ("red", 2), ("blue", 1)]

def total_marbles (bag : List (String × Nat)) : Nat := bag.foldr (fun (_, n) acc => n + acc) 0

def favorable_outcomes (bag : List (String × Nat)) : Nat :=
  (bag.filter (fun (color, _) => color = "yellow" ∨ color = "green")).foldr (fun (_, n) acc => n + acc) 0

theorem probability_of_yellow_or_green :
  (favorable_outcomes bag : ℚ) / (total_marbles bag : ℚ) = 7 / 10 := by
  sorry

end probability_of_yellow_or_green_l622_62218


namespace average_salary_l622_62217

theorem average_salary (a b c d e : ℕ) (h1 : a = 8000) (h2 : b = 5000) (h3 : c = 16000) (h4 : d = 7000) (h5 : e = 9000) :
  (a + b + c + d + e) / 5 = 9000 :=
by
  sorry

end average_salary_l622_62217


namespace sum_of_midpoint_coordinates_l622_62215

theorem sum_of_midpoint_coordinates (x1 y1 x2 y2 : ℝ) (h1 : x1 = 8) (h2 : y1 = 16) (h3 : x2 = -2) (h4 : y2 = -8) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 7 := by
  sorry

end sum_of_midpoint_coordinates_l622_62215


namespace total_number_of_applications_l622_62267

def in_state_apps := 200
def out_state_apps := 2 * in_state_apps
def total_apps := in_state_apps + out_state_apps

theorem total_number_of_applications : total_apps = 600 := by
  sorry

end total_number_of_applications_l622_62267


namespace point_on_line_l622_62287

theorem point_on_line (m : ℝ) (P : ℝ × ℝ) (line_eq : ℝ × ℝ → Prop) (h : P = (2, m)) 
  (h_line : line_eq = fun P => 3 * P.1 + P.2 = 2) : 
  3 * 2 + m = 2 → m = -4 :=
by
  intro h1
  linarith

end point_on_line_l622_62287


namespace alcohol_solution_mixing_l622_62202

theorem alcohol_solution_mixing :
  ∀ (V_i C_i C_f C_a x : ℝ),
    V_i = 6 →
    C_i = 0.40 →
    C_f = 0.50 →
    C_a = 0.90 →
    x = 1.5 →
  0.50 * (V_i + x) = (C_i * V_i) + C_a * x →
  C_f * (V_i + x) = (C_i * V_i) + (C_a * x) := 
by
  intros V_i C_i C_f C_a x Vi_eq Ci_eq Cf_eq Ca_eq x_eq h
  sorry

end alcohol_solution_mixing_l622_62202


namespace lucy_final_balance_l622_62249

def initial_balance : ℝ := 65
def deposit : ℝ := 15
def withdrawal : ℝ := 4

theorem lucy_final_balance : initial_balance + deposit - withdrawal = 76 :=
by
  sorry

end lucy_final_balance_l622_62249
