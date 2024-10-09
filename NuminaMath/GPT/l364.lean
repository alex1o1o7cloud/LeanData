import Mathlib

namespace perpendicular_slopes_l364_36460

theorem perpendicular_slopes {m : ℝ} (h : (1 : ℝ) * -m = -1) : m = 1 :=
by sorry

end perpendicular_slopes_l364_36460


namespace sum_of_center_coordinates_l364_36404

theorem sum_of_center_coordinates (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 7) (h2 : y1 = -6) (h3 : x2 = -5) (h4 : y2 = 4) :
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  midpoint_x + midpoint_y = 0 := by
  -- Definitions and setup
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  sorry

end sum_of_center_coordinates_l364_36404


namespace num_zeros_of_g_l364_36441

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 - 2 * x
else -(x^2 - 2 * -x)

noncomputable def g (x : ℝ) : ℝ := f x + 1

theorem num_zeros_of_g : ∃! x : ℝ, g x = 0 := sorry

end num_zeros_of_g_l364_36441


namespace simplify_and_evaluate_l364_36498

-- Define the expression
def expr (x : ℝ) : ℝ := x^2 * (x + 1) - x * (x^2 - x + 1)

-- The main theorem stating the equivalence
theorem simplify_and_evaluate (x : ℝ) (h : x = 5) : expr x = 45 :=
by {
  sorry
}

end simplify_and_evaluate_l364_36498


namespace interior_angle_regular_octagon_l364_36413

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l364_36413


namespace problem_statement_l364_36402

theorem problem_statement (x y z : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : z ≠ 0) (h₃ : x + y + z = 0) (h₄ : xy + xz + yz ≠ 0) : 
  (x^7 + y^7 + z^7) / (x * y * z * (x * y + x * z + y * z)) = -7 :=
by
  sorry

end problem_statement_l364_36402


namespace max_lateral_surface_area_of_tetrahedron_l364_36475

open Real

theorem max_lateral_surface_area_of_tetrahedron :
  ∀ (PA PB PC : ℝ), (PA^2 + PB^2 + PC^2 = 36) → (PA * PB + PB * PC + PA * PC ≤ 36) →
  (1/2 * (PA * PB + PB * PC + PA * PC) ≤ 18) :=
by
  intro PA PB PC hsum hineq
  sorry

end max_lateral_surface_area_of_tetrahedron_l364_36475


namespace matilda_jellybeans_l364_36499

theorem matilda_jellybeans (steve_jellybeans : ℕ) (h_steve : steve_jellybeans = 84)
  (h_matt : ℕ) (h_matt_calc : h_matt = 10 * steve_jellybeans)
  (h_matilda : ℕ) (h_matilda_calc : h_matilda = h_matt / 2) :
  h_matilda = 420 := by
  sorry

end matilda_jellybeans_l364_36499


namespace milk_purchase_maximum_l364_36466

theorem milk_purchase_maximum :
  let num_1_liter_bottles := 6
  let num_half_liter_bottles := 6
  let value_per_1_liter_bottle := 20
  let value_per_half_liter_bottle := 15
  let price_per_liter := 22
  let total_value := num_1_liter_bottles * value_per_1_liter_bottle + num_half_liter_bottles * value_per_half_liter_bottle
  total_value / price_per_liter = 5 :=
by
  sorry

end milk_purchase_maximum_l364_36466


namespace palindrome_clock_count_l364_36406

-- Definitions based on conditions from the problem statement.
def is_valid_hour (h : ℕ) : Prop := h < 24
def is_valid_minute (m : ℕ) : Prop := m < 60
def is_palindrome (h m : ℕ) : Prop :=
  (h < 10 ∧ m / 10 = h ∧ m % 10 = h) ∨
  (h >= 10 ∧ (h / 10) = (m % 10) ∧ (h % 10) = (m / 10 % 10))

-- Main theorem statement
theorem palindrome_clock_count : 
  (∃ n : ℕ, n = 66 ∧ ∀ (h m : ℕ), is_valid_hour h → is_valid_minute m → is_palindrome h m) := 
sorry

end palindrome_clock_count_l364_36406


namespace original_average_l364_36470

theorem original_average (A : ℝ) (h : 5 * A = 130) : A = 26 :=
by
  have h1 : 5 * A / 5 = 130 / 5 := by sorry
  sorry

end original_average_l364_36470


namespace harry_average_sleep_l364_36435

-- Conditions
def sleep_time_monday : ℕ × ℕ := (8, 15)
def sleep_time_tuesday : ℕ × ℕ := (7, 45)
def sleep_time_wednesday : ℕ × ℕ := (8, 10)
def sleep_time_thursday : ℕ × ℕ := (10, 25)
def sleep_time_friday : ℕ × ℕ := (7, 50)

-- Total sleep time calculation
def total_sleep_time : ℕ × ℕ :=
  let (h1, m1) := sleep_time_monday
  let (h2, m2) := sleep_time_tuesday
  let (h3, m3) := sleep_time_wednesday
  let (h4, m4) := sleep_time_thursday
  let (h5, m5) := sleep_time_friday
  (h1 + h2 + h3 + h4 + h5, m1 + m2 + m3 + m4 + m5)

-- Convert minutes to hours and minutes
def convert_minutes (mins : ℕ) : ℕ × ℕ :=
  (mins / 60, mins % 60)

-- Final total sleep time
def final_total_time : ℕ × ℕ :=
  let (total_hours, total_minutes) := total_sleep_time
  let (extra_hours, remaining_minutes) := convert_minutes total_minutes
  (total_hours + extra_hours, remaining_minutes)

-- Average calculation
def average_sleep_time : ℕ × ℕ :=
  let (total_hours, total_minutes) := final_total_time
  (total_hours / 5, (total_hours % 5) * 60 / 5 + total_minutes / 5)

-- The proof statement
theorem harry_average_sleep :
  average_sleep_time = (8, 29) :=
  by
    sorry

end harry_average_sleep_l364_36435


namespace graph_does_not_pass_first_quadrant_l364_36427

variables {a b x : ℝ}

theorem graph_does_not_pass_first_quadrant 
  (h₁ : 0 < a ∧ a < 1) 
  (h₂ : b < -1) : 
  ¬ ∃ x : ℝ, 0 < x ∧ 0 < a^x + b :=
sorry

end graph_does_not_pass_first_quadrant_l364_36427


namespace machine_working_days_l364_36480

variable {V a b c x y z : ℝ} 

noncomputable def machine_individual_times_condition (a b c : ℝ) : Prop :=
  ∀ (x y z : ℝ), (x = a + (-(a + b) + Real.sqrt ((a - b) ^ 2 + 4 * a * b * c ^ 2)) / (2 * (c + 1))) ∧
                 (y = b + (-(a + b) + Real.sqrt ((a - b) ^ 2 + 4 * a * b * c ^ 2)) / (2 * (c + 1))) ∧
                 (z = (-(c * (a + b)) + c * Real.sqrt ((a - b) ^ 2 + 4 * a * b * c ^ 2)) / (2 * (c + 1))) ∧
                 (c > 1)

theorem machine_working_days (h1 : x = (z / c) + a) (h2 : y = (z / c) + b) (h3 : z = c * (z / c)) :
  machine_individual_times_condition a b c :=
by
  sorry

end machine_working_days_l364_36480


namespace travel_options_l364_36467

-- Define the conditions
def trains_from_A_to_B := 3
def ferries_from_B_to_C := 2

-- State the proof problem
theorem travel_options (t : ℕ) (f : ℕ) (h1 : t = trains_from_A_to_B) (h2 : f = ferries_from_B_to_C) : t * f = 6 :=
by
  rewrite [h1, h2]
  sorry

end travel_options_l364_36467


namespace paving_cost_l364_36428

-- Definitions based on conditions
def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sqm : ℝ := 600
def expected_cost : ℝ := 12375

-- The problem statement
theorem paving_cost :
  (length * width * rate_per_sqm = expected_cost) :=
sorry

end paving_cost_l364_36428


namespace probability_queen_of_diamonds_l364_36425

/-- 
A standard deck of 52 cards consists of 13 ranks and 4 suits.
We want to prove that the probability the top card is the Queen of Diamonds is 1/52.
-/
theorem probability_queen_of_diamonds 
  (total_cards : ℕ) 
  (queen_of_diamonds : ℕ)
  (h1 : total_cards = 52)
  (h2 : queen_of_diamonds = 1) : 
  (queen_of_diamonds : ℚ) / (total_cards : ℚ) = 1 / 52 := 
by 
  sorry

end probability_queen_of_diamonds_l364_36425


namespace pen_case_cost_l364_36417

noncomputable def case_cost (p i c : ℝ) : Prop :=
  p + i + c = 2.30 ∧
  p = 1.50 + i ∧
  c = 0.5 * i →
  c = 0.1335

theorem pen_case_cost (p i c : ℝ) : case_cost p i c :=
by
  sorry

end pen_case_cost_l364_36417


namespace Oshea_needs_30_small_planters_l364_36459

theorem Oshea_needs_30_small_planters 
  (total_seeds : ℕ) 
  (large_planters : ℕ) 
  (capacity_large : ℕ) 
  (capacity_small : ℕ)
  (h1: total_seeds = 200) 
  (h2: large_planters = 4) 
  (h3: capacity_large = 20) 
  (h4: capacity_small = 4) : 
  (total_seeds - large_planters * capacity_large) / capacity_small = 30 :=
by 
  sorry

end Oshea_needs_30_small_planters_l364_36459


namespace sum_of_coordinates_l364_36421

theorem sum_of_coordinates (x : ℚ) : (0, 0) = (0, 0) ∧ (x, -3) = (x, -3) ∧ ((-3 - 0) / (x - 0) = 4 / 5) → x - 3 = -27 / 4 := 
sorry

end sum_of_coordinates_l364_36421


namespace geometric_sequence_a4_l364_36457

noncomputable def a (n : ℕ) : ℝ := sorry -- placeholder for the geometric sequence

axiom a_2 : a 2 = -2
axiom a_6 : a 6 = -32
axiom geom_seq (n : ℕ) : a (n + 1) / a n = a (n + 2) / a (n + 1)

theorem geometric_sequence_a4 : a 4 = -8 := 
by
  sorry

end geometric_sequence_a4_l364_36457


namespace max_profit_l364_36477

-- Definition of the conditions
def production_requirements (tonAprodA tonAprodB tonBprodA tonBprodB: ℕ )
  := tonAprodA = 3 ∧ tonAprodB = 1 ∧ tonBprodA = 2 ∧ tonBprodB = 3

def profit_per_ton ( profitA profitB: ℕ )
  := profitA = 50000 ∧ profitB = 30000

def raw_material_limits ( rawA rawB: ℕ)
  := rawA = 13 ∧ rawB = 18

theorem max_profit 
  (production_requirements: production_requirements 3 1 2 3)
  (profit_per_ton: profit_per_ton 50000 30000)
  (raw_material_limits: raw_material_limits 13 18)
: ∃ (maxProfit: ℕ), maxProfit = 270000 := 
by 
  sorry

end max_profit_l364_36477


namespace total_children_is_11_l364_36452

noncomputable def num_of_children (b g : ℕ) := b + g

theorem total_children_is_11 (b g : ℕ) :
  (∃ c : ℕ, b * c + g * (c + 1) = 47) ∧
  (∃ m : ℕ, b * (m + 1) + g * m = 74) → 
  num_of_children b g = 11 :=
by
  -- The proof steps would go here to show that b + g = 11
  sorry

end total_children_is_11_l364_36452


namespace village_population_l364_36412

variable (Px : ℕ) (t : ℕ) (dX dY : ℕ)
variable (Py : ℕ := 42000) (rateX : ℕ := 1200) (rateY : ℕ := 800) (timeYears : ℕ := 15)

theorem village_population : (Px - rateX * timeYears = Py + rateY * timeYears) → Px = 72000 :=
by
  sorry

end village_population_l364_36412


namespace range_of_a_l364_36469

noncomputable def f (a x : ℝ) := a * Real.log x + x - 1

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x → f a x ≥ 0) : a ≥ -1 := by
  sorry

end range_of_a_l364_36469


namespace sally_quarters_l364_36474

noncomputable def initial_quarters : ℕ := 760
noncomputable def spent_quarters : ℕ := 418
noncomputable def remaining_quarters : ℕ := 342

theorem sally_quarters : initial_quarters - spent_quarters = remaining_quarters :=
by sorry

end sally_quarters_l364_36474


namespace inequality_proof_l364_36476

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c) ≥ 1 / (a + b) + 1 / (b + c) + 1 / (c + a) :=
by
  sorry

end inequality_proof_l364_36476


namespace pyramid_max_volume_height_l364_36430

-- Define the conditions and the theorem
theorem pyramid_max_volume_height
  (a h V : ℝ)
  (SA : ℝ := 2 * Real.sqrt 3)
  (h_eq : h = Real.sqrt (SA^2 - (Real.sqrt 2 * a / 2)^2))
  (V_eq : V = (1 / 3) * a^2 * h)
  (derivative_at_max : ∀ a, (48 * a^3 - 3 * a^5 = 0) → (a = 0 ∨ a = 4))
  (max_a_value : a = 4):
  h = 2 :=
by
  sorry

end pyramid_max_volume_height_l364_36430


namespace course_gender_relationship_expected_value_X_l364_36472

-- Define the data based on the problem statement
def total_students := 450
def total_boys := 250
def total_girls := 200
def boys_course_b := 150
def girls_course_a := 50
def boys_course_a := total_boys - boys_course_b -- 100
def girls_course_b := total_girls - girls_course_a -- 150

-- Test statistic for independence (calculated)
def chi_squared := 22.5
def critical_value := 10.828

-- Null hypothesis for independence
def H0 := "The choice of course is independent of gender"

-- part 1: proving independence rejection based on chi-squared value
theorem course_gender_relationship : chi_squared > critical_value :=
  by sorry

-- For part 2, stratified sampling and expected value
-- Define probabilities and expected value
def P_X_0 := 1/6
def P_X_1 := 1/2
def P_X_2 := 3/10
def P_X_3 := 1/30

def expected_X := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3

-- part 2: proving expected value E(X) calculation
theorem expected_value_X : expected_X = 6/5 :=
  by sorry

end course_gender_relationship_expected_value_X_l364_36472


namespace gcd_binom_is_integer_l364_36447

theorem gcd_binom_is_integer 
  (m n : ℤ) 
  (hm : m ≥ 1) 
  (hn : n ≥ m)
  (gcd_mn : ℤ := Int.gcd m n)
  (binom_nm : ℤ := Nat.choose n.toNat m.toNat) :
  (gcd_mn * binom_nm) % n.toNat = 0 := by
  sorry

end gcd_binom_is_integer_l364_36447


namespace three_digit_number_l364_36488

theorem three_digit_number (a b c : ℕ) (h1 : a + b + c = 10) (h2 : b = a + c) (h3 : 100 * c + 10 * b + a = 100 * a + 10 * b + c + 99) : (100 * a + 10 * b + c) = 253 := 
by
  sorry

end three_digit_number_l364_36488


namespace remainder_91_pow_91_mod_100_l364_36408

-- Definitions
def large_power_mod (a b n : ℕ) : ℕ :=
  (a^b) % n

-- Statement
theorem remainder_91_pow_91_mod_100 : large_power_mod 91 91 100 = 91 :=
by
  sorry

end remainder_91_pow_91_mod_100_l364_36408


namespace sequence_property_l364_36426

theorem sequence_property :
  ∃ (a_0 a_1 a_2 a_3 : ℕ),
    a_0 + a_1 + a_2 + a_3 = 4 ∧
    (a_0 = ([a_0, a_1, a_2, a_3].count 0)) ∧
    (a_1 = ([a_0, a_1, a_2, a_3].count 1)) ∧
    (a_2 = ([a_0, a_1, a_2, a_3].count 2)) ∧
    (a_3 = ([a_0, a_1, a_2, a_3].count 3)) :=
sorry

end sequence_property_l364_36426


namespace taxi_fare_ride_distance_l364_36462

theorem taxi_fare_ride_distance (fare_first: ℝ) (first_mile: ℝ) (additional_fare_rate: ℝ) (additional_distance: ℝ) (total_amount: ℝ) (tip: ℝ) (x: ℝ) :
  fare_first = 3.00 ∧ first_mile = 0.75 ∧ additional_fare_rate = 0.25 ∧ additional_distance = 0.1 ∧ total_amount = 15 ∧ tip = 3 ∧
  (total_amount - tip) = fare_first + additional_fare_rate * (x - first_mile) / additional_distance → x = 4.35 :=
by
  intros
  sorry

end taxi_fare_ride_distance_l364_36462


namespace number_of_girls_l364_36431

theorem number_of_girls (classes : ℕ) (students_per_class : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : classes = 4) 
  (h2 : students_per_class = 25) 
  (h3 : boys = 56) 
  (h4 : girls = (classes * students_per_class) - boys) : 
  girls = 44 :=
by
  sorry

end number_of_girls_l364_36431


namespace cone_volume_from_half_sector_l364_36442

theorem cone_volume_from_half_sector (R : ℝ) (V : ℝ) : 
  R = 6 →
  V = (1/3) * Real.pi * (R / 2)^2 * (R * Real.sqrt 3) →
  V = 9 * Real.pi * Real.sqrt 3 := by sorry

end cone_volume_from_half_sector_l364_36442


namespace num_integers_condition_l364_36420

theorem num_integers_condition : 
  (∃ (n1 n2 n3 : ℤ), 0 < n1 ∧ n1 < 30 ∧ (∃ k1 : ℤ, (30 - n1) / n1 = k1 ^ 2) ∧
                     0 < n2 ∧ n2 < 30 ∧ (∃ k2 : ℤ, (30 - n2) / n2 = k2 ^ 2) ∧
                     0 < n3 ∧ n3 < 30 ∧ (∃ k3 : ℤ, (30 - n3) / n3 = k3 ^ 2) ∧
                     ∀ n : ℤ, 0 < n ∧ n < 30 ∧ (∃ k : ℤ, (30 - n) / n = k ^ 2) → 
                              (n = n1 ∨ n = n2 ∨ n = n3)) :=
sorry

end num_integers_condition_l364_36420


namespace solve_for_m_l364_36485

theorem solve_for_m (x m : ℝ) (hx : 0 < x) (h_eq : m / (x^2 - 9) + 2 / (x + 3) = 1 / (x - 3)) : m = 6 :=
sorry

end solve_for_m_l364_36485


namespace days_before_reinforcement_l364_36407

/-- A garrison of 2000 men originally has provisions for 62 days.
    After some days, a reinforcement of 2700 men arrives.
    The provisions are found to last for only 20 days more after the reinforcement arrives.
    Prove that the number of days passed before the reinforcement arrived is 15. -/
theorem days_before_reinforcement 
  (x : ℕ) 
  (num_men_orig : ℕ := 2000) 
  (num_men_reinf : ℕ := 2700) 
  (days_orig : ℕ := 62) 
  (days_after_reinf : ℕ := 20) 
  (total_provisions : ℕ := num_men_orig * days_orig)
  (remaining_provisions : ℕ := num_men_orig * (days_orig - x))
  (consumption_after_reinf : ℕ := (num_men_orig + num_men_reinf) * days_after_reinf) 
  (provisions_eq : remaining_provisions = consumption_after_reinf) : 
  x = 15 := 
by 
  sorry

end days_before_reinforcement_l364_36407


namespace maximize_winning_probability_l364_36468

def ahmet_wins (n : ℕ) : Prop :=
  n = 13

theorem maximize_winning_probability :
  ∃ n ∈ {x : ℕ | x ≥ 1 ∧ x ≤ 25}, ahmet_wins n :=
by
  sorry

end maximize_winning_probability_l364_36468


namespace sum_of_coordinates_D_l364_36484

theorem sum_of_coordinates_D
    (C N D : ℝ × ℝ) 
    (hC : C = (10, 5))
    (hN : N = (4, 9))
    (h_midpoint : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) : 
    C.1 + D.1 + (C.2 + D.2) = 22 :=
  by sorry

end sum_of_coordinates_D_l364_36484


namespace chocolate_candy_cost_l364_36418

-- Define the constants and conditions
def cost_per_box : ℕ := 5
def candies_per_box : ℕ := 30
def discount_rate : ℝ := 0.1

-- Define the total number of candies to buy
def total_candies : ℕ := 450

-- Define the threshold for applying discount
def discount_threshold : ℕ := 300

-- Calculate the number of boxes needed
def boxes_needed (total_candies : ℕ) (candies_per_box : ℕ) : ℕ :=
  total_candies / candies_per_box

-- Calculate the total cost without discount
def total_cost (boxes_needed : ℕ) (cost_per_box : ℕ) : ℝ :=
  boxes_needed * cost_per_box

-- Calculate the discounted cost
def discounted_cost (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  if total_candies > discount_threshold then
    total_cost * (1 - discount_rate)
  else
    total_cost

-- Statement to be proved
theorem chocolate_candy_cost :
  discounted_cost 
    (total_cost (boxes_needed total_candies candies_per_box) cost_per_box) 
    discount_rate = 67.5 :=
by
  -- Proof is needed here, using the correct steps from the solution.
  sorry

end chocolate_candy_cost_l364_36418


namespace number_of_children_is_five_l364_36483

/-- The sum of the ages of children born at intervals of 2 years each is 50 years, 
    and the age of the youngest child is 6 years.
    Prove that the number of children is 5. -/
theorem number_of_children_is_five (n : ℕ) (h1 : (0 < n ∧ n / 2 * (8 + 2 * n) = 50)): n = 5 :=
sorry

end number_of_children_is_five_l364_36483


namespace households_subscribing_B_and_C_l364_36453

/-- Each household subscribes to 2 different newspapers.
Residents only subscribe to newspapers A, B, and C.
There are 30 subscriptions for newspaper A.
There are 34 subscriptions for newspaper B.
There are 40 subscriptions for newspaper C.
Thus, the number of households that subscribe to both
newspaper B and newspaper C is 22. -/
theorem households_subscribing_B_and_C (subs_A subs_B subs_C households : ℕ) 
    (hA : subs_A = 30) (hB : subs_B = 34) (hC : subs_C = 40) (h_total : households = (subs_A + subs_B + subs_C) / 2) :
  (households - subs_A) = 22 :=
by
  -- Substitute the values to demonstrate equality based on the given conditions.
  sorry

end households_subscribing_B_and_C_l364_36453


namespace mark_savings_l364_36497

-- Given conditions
def original_price : ℝ := 300
def discount_rate : ℝ := 0.20
def cheaper_lens_price : ℝ := 220

-- Definitions derived from conditions
def discount_amount : ℝ := original_price * discount_rate
def discounted_price : ℝ := original_price - discount_amount
def savings : ℝ := discounted_price - cheaper_lens_price

-- Statement to prove
theorem mark_savings : savings = 20 :=
by
  -- Definitions incorporated
  have h1 : discount_amount = 300 * 0.20 := rfl
  have h2 : discounted_price = 300 - discount_amount := rfl
  have h3 : cheaper_lens_price = 220 := rfl
  have h4 : savings = discounted_price - cheaper_lens_price := rfl
  sorry

end mark_savings_l364_36497


namespace no_perfect_square_l364_36424

theorem no_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, k^2 = 2 * 13^n + 5 * 7^n + 26 :=
sorry

end no_perfect_square_l364_36424


namespace line_equation_l364_36439

theorem line_equation (A : ℝ × ℝ) (hA : A = (1, 4))
  (sum_intercepts_zero : ∃ a b : ℝ, (a + b = 0) ∧ (A.1 * b + A.2 * a = a * b)) :
  (∀ x y : ℝ, x - A.1 = (y - A.2) * 4 → 4 * x - y = 0) ∨
  (∀ x y : ℝ, (x / (-3)) + (y / 3) = 1 → x - y + 3 = 0) :=
sorry

end line_equation_l364_36439


namespace problem_dorlir_ahmeti_equality_case_l364_36493

theorem problem_dorlir_ahmeti (x y z : ℝ)
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h : x^2 + y^2 + z^2 = x + y + z) : 
  (x + 1) / Real.sqrt (x^5 + x + 1) + 
  (y + 1) / Real.sqrt (y^5 + y + 1) + 
  (z + 1) / Real.sqrt (z^5 + z + 1) ≥ 3 :=
sorry
  
theorem equality_case (x y z : ℝ)
  (hx : x = 0) (hy : y = 0) (hz : z = 0) : 
  (x + 1) / Real.sqrt (x^5 + x + 1) + 
  (y + 1) / Real.sqrt (y^5 + y + 1) + 
  (z + 1) / Real.sqrt (z^5 + z + 1) = 3 :=
sorry

end problem_dorlir_ahmeti_equality_case_l364_36493


namespace minimum_cost_l364_36422

noncomputable def volume : ℝ := 4800
noncomputable def depth : ℝ := 3
noncomputable def base_cost_per_sqm : ℝ := 150
noncomputable def wall_cost_per_sqm : ℝ := 120
noncomputable def base_area (volume depth : ℝ) : ℝ := volume / depth
noncomputable def wall_surface_area (x : ℝ) : ℝ :=
  6 * x + (2 * (volume * depth / x))

noncomputable def construction_cost (x : ℝ) : ℝ :=
  wall_surface_area x * wall_cost_per_sqm + base_area volume depth * base_cost_per_sqm

theorem minimum_cost :
  ∃(x : ℝ), x = 40 ∧ construction_cost x = 297600 := by
  sorry

end minimum_cost_l364_36422


namespace cos_sin_value_l364_36409

theorem cos_sin_value (α : ℝ) (h : Real.tan α = Real.sqrt 2) : Real.cos α * Real.sin α = Real.sqrt 2 / 3 :=
sorry

end cos_sin_value_l364_36409


namespace joan_initial_books_l364_36486

variable (books_sold : ℕ)
variable (books_left : ℕ)

theorem joan_initial_books (h1 : books_sold = 26) (h2 : books_left = 7) : books_sold + books_left = 33 := by
  sorry

end joan_initial_books_l364_36486


namespace solve_for_x_l364_36411

def condition (x : ℝ) : Prop := (x - 5)^3 = (1 / 27)⁻¹

theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 8 := by
  use 8
  unfold condition
  sorry

end solve_for_x_l364_36411


namespace base8_subtraction_and_conversion_l364_36429

-- Define the base 8 numbers
def num1 : ℕ := 7463 -- 7463 in base 8
def num2 : ℕ := 3254 -- 3254 in base 8

-- Define the subtraction in base 8 and conversion to base 10
def result_base8 : ℕ := 4207 -- Expected result in base 8
def result_base10 : ℕ := 2183 -- Expected result in base 10

-- Helper function to convert from base 8 to base 10
def convert_base8_to_base10 (n : ℕ) : ℕ := 
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8 + (n % 10)
 
-- Main theorem statement
theorem base8_subtraction_and_conversion :
  (num1 - num2 = result_base8) ∧ (convert_base8_to_base10 result_base8 = result_base10) :=
by
  sorry

end base8_subtraction_and_conversion_l364_36429


namespace total_supermarkets_FGH_chain_l364_36410

def supermarkets_us : ℕ := 47
def supermarkets_difference : ℕ := 10
def supermarkets_canada : ℕ := supermarkets_us - supermarkets_difference
def total_supermarkets : ℕ := supermarkets_us + supermarkets_canada

theorem total_supermarkets_FGH_chain : total_supermarkets = 84 :=
by 
  sorry

end total_supermarkets_FGH_chain_l364_36410


namespace necessary_but_not_sufficient_condition_l364_36434

def p (x : ℝ) : Prop := |x + 1| ≤ 4
def q (x : ℝ) : Prop := x^2 - 5*x + 6 ≤ 0

theorem necessary_but_not_sufficient_condition (x : ℝ) : 
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) := 
by
  sorry

end necessary_but_not_sufficient_condition_l364_36434


namespace z_investment_correct_l364_36496

noncomputable def z_investment 
    (x_investment : ℕ) 
    (y_investment : ℕ) 
    (z_profit : ℕ) 
    (total_profit : ℕ)
    (profit_z : ℕ) : ℕ := 
  let x_time := 12
  let y_time := 12
  let z_time := 8
  let x_share := x_investment * x_time
  let y_share := y_investment * y_time
  let profit_ratio := total_profit - profit_z
  (x_share + y_share) * z_time / profit_ratio

theorem z_investment_correct : 
  z_investment 36000 42000 4032 13860 4032 = 52000 :=
by sorry

end z_investment_correct_l364_36496


namespace arithmetic_geometric_progression_l364_36405

theorem arithmetic_geometric_progression (a d : ℝ)
    (h1 : 2 * (a - d) * a * (a + d + 7) = 1000)
    (h2 : a^2 = 2 * (a - d) * (a + d + 7)) :
    d = 8 ∨ d = -8 := 
    sorry

end arithmetic_geometric_progression_l364_36405


namespace ratio_shorter_longer_l364_36473

theorem ratio_shorter_longer (total_length shorter_length longer_length : ℝ)
  (h1 : total_length = 21) 
  (h2 : shorter_length = 6) 
  (h3 : longer_length = total_length - shorter_length) 
  (h4 : shorter_length / longer_length = 2 / 5) : 
  shorter_length / longer_length = 2 / 5 :=
by sorry

end ratio_shorter_longer_l364_36473


namespace evaluate_expression_l364_36456

theorem evaluate_expression : 6 - 8 * (5 - 2^3) / 2 = 18 := by
  sorry

end evaluate_expression_l364_36456


namespace original_workers_l364_36432

theorem original_workers (x y : ℝ) (h : x = (65 / 100) * y) : y = (20 / 13) * x :=
by sorry

end original_workers_l364_36432


namespace max_point_diff_l364_36492

theorem max_point_diff (n : ℕ) : ∃ max_diff, max_diff = 2 :=
by
  -- Conditions from (a)
  -- - \( n \) teams participate in a football tournament.
  -- - Each team plays against every other team exactly once.
  -- - The winning team is awarded 2 points.
  -- - A draw gives -1 point to each team.
  -- - The losing team gets 0 points.
  -- Correct Answer from (b)
  -- - The maximum point difference between teams that are next to each other in the ranking is 2.
  sorry

end max_point_diff_l364_36492


namespace find_k_of_quadratic_polynomial_l364_36401

variable (k : ℝ)

theorem find_k_of_quadratic_polynomial (h1 : (k - 2) = 0) (h2 : k ≠ 0) : k = 2 :=
by
  -- proof omitted
  sorry

end find_k_of_quadratic_polynomial_l364_36401


namespace prove_angle_C_prove_max_area_l364_36438

open Real

variables {A B C : ℝ} {a b c : ℝ} (abc_is_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
variables (R : ℝ) (circumradius_is_sqrt2 : R = sqrt 2)
variables (H : 2 * sqrt 2 * (sin A ^ 2 - sin C ^ 2) = (a - b) * sin B)
variables (law_of_sines : a = 2 * R * sin A ∧ b = 2 * R * sin B ∧ c = 2 * R * sin C)

-- Part 1: Prove that angle C = π / 3
theorem prove_angle_C : C = π / 3 :=
sorry

-- Part 2: Prove that the maximum value of the area S of triangle ABC is (3 * sqrt 3) / 2
theorem prove_max_area : (1 / 2) * a * b * sin C ≤ (3 * sqrt 3) / 2 :=
sorry

end prove_angle_C_prove_max_area_l364_36438


namespace merry_boxes_on_sunday_l364_36479

theorem merry_boxes_on_sunday
  (num_boxes_saturday : ℕ := 50)
  (apples_per_box : ℕ := 10)
  (total_apples_sold : ℕ := 720)
  (remaining_boxes : ℕ := 3) :
  num_boxes_saturday * apples_per_box ≤ total_apples_sold →
  (total_apples_sold - num_boxes_saturday * apples_per_box) / apples_per_box + remaining_boxes = 25 := by
  intros
  sorry

end merry_boxes_on_sunday_l364_36479


namespace angle_bisector_slope_l364_36487

theorem angle_bisector_slope :
  let m₁ := 2
  let m₂ := 5
  let k := (7 - 2 * Real.sqrt 5) / 11
  True :=
by admit

end angle_bisector_slope_l364_36487


namespace closest_to_fraction_is_2000_l364_36403

-- Define the original fractions and their approximations
def numerator : ℝ := 410
def denominator : ℝ := 0.21
def approximated_numerator : ℝ := 400
def approximated_denominator : ℝ := 0.2

-- Define the options to choose from
def options : List ℝ := [100, 500, 1900, 2000, 2500]

-- Statement to prove that the closest value to numerator / denominator is 2000
theorem closest_to_fraction_is_2000 : 
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 100) ∧
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 500) ∧
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 1900) ∧
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 2500) :=
sorry

end closest_to_fraction_is_2000_l364_36403


namespace candy_pieces_given_l364_36461

theorem candy_pieces_given (initial total : ℕ) (h1 : initial = 68) (h2 : total = 93) :
  total - initial = 25 :=
by
  sorry

end candy_pieces_given_l364_36461


namespace spinning_class_frequency_l364_36481

/--
We define the conditions given in the problem:
- duration of each class in hours,
- calorie burn rate per minute,
- total calories burned per week.
We then state that the number of classes James attends per week is equal to 3.
-/
def class_duration_hours : ℝ := 1.5
def calories_per_minute : ℝ := 7
def total_calories_per_week : ℝ := 1890

theorem spinning_class_frequency :
  total_calories_per_week / (class_duration_hours * 60 * calories_per_minute) = 3 :=
by
  sorry

end spinning_class_frequency_l364_36481


namespace sum_digits_largest_N_l364_36437

-- Define the conditions
def is_multiple_of_six (N : ℕ) : Prop := N % 6 = 0

def P (N : ℕ) : ℚ := 
  let favorable_positions := (N + 1) *
    (⌊(1:ℚ) / 3 * N⌋ + 1 + (N - ⌈(2:ℚ) / 3 * N⌉ + 1))
  favorable_positions / (N + 1)

axiom P_6_equals_1 : P 6 = 1
axiom P_large_N : ∀ ε > 0, ∃ N > 0, is_multiple_of_six N ∧ P N ≥ (5/6) - ε

-- Main theorem statement
theorem sum_digits_largest_N : 
  ∃ N : ℕ, is_multiple_of_six N ∧ P N > 3/4 ∧ (N.digits 10).sum = 6 :=
sorry

end sum_digits_largest_N_l364_36437


namespace sin_minus_cos_value_complex_trig_value_l364_36445

noncomputable def sin_cos_equation (x : Real) :=
  -Real.pi / 2 < x ∧ x < Real.pi / 2 ∧ Real.sin x + Real.cos x = -1 / 5

theorem sin_minus_cos_value (x : Real) (h : sin_cos_equation x) :
  Real.sin x - Real.cos x = 7 / 5 :=
sorry

theorem complex_trig_value (x : Real) (h : sin_cos_equation x) :
  (Real.sin (Real.pi + x) + Real.sin (3 * Real.pi / 2 - x)) / 
  (Real.tan (Real.pi - x) + Real.sin (Real.pi / 2 - x)) = 3 / 11 :=
sorry

end sin_minus_cos_value_complex_trig_value_l364_36445


namespace g_h_of_2_eq_2340_l364_36419

def g (x : ℝ) : ℝ := 2 * x^2 + 5 * x - 3
def h (x : ℝ) : ℝ := 4 * x^3 + 1

theorem g_h_of_2_eq_2340 : g (h 2) = 2340 := 
  sorry

end g_h_of_2_eq_2340_l364_36419


namespace triangle_tangent_identity_l364_36465

theorem triangle_tangent_identity (A B C : ℝ) (h : A + B + C = Real.pi) : 
  (Real.tan (A / 2) * Real.tan (B / 2)) + (Real.tan (B / 2) * Real.tan (C / 2)) + (Real.tan (C / 2) * Real.tan (A / 2)) = 1 :=
by
  sorry

end triangle_tangent_identity_l364_36465


namespace percentage_error_divide_instead_of_multiply_l364_36494

theorem percentage_error_divide_instead_of_multiply (x : ℝ) : 
  let correct_result := 5 * x 
  let incorrect_result := x / 10 
  let error := correct_result - incorrect_result 
  let percentage_error := (error / correct_result) * 100 
  percentage_error = 98 :=
by
  sorry

end percentage_error_divide_instead_of_multiply_l364_36494


namespace value_of_f_neg_11_over_2_l364_36455

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodicity (x : ℝ) : f (x + 2) = - (f x)⁻¹
axiom interval_value (h : 2 ≤ 5 / 2 ∧ 5 / 2 ≤ 3) : f (5 / 2) = 5 / 2

theorem value_of_f_neg_11_over_2 : f (-11 / 2) = 5 / 2 :=
by
  sorry

end value_of_f_neg_11_over_2_l364_36455


namespace solve_quadratic_eqn_l364_36423

theorem solve_quadratic_eqn :
  ∃ x₁ x₂ : ℝ, (x - 6) * (x + 2) = 0 ↔ (x = x₁ ∨ x = x₂) ∧ x₁ = 6 ∧ x₂ = -2 :=
by
  sorry

end solve_quadratic_eqn_l364_36423


namespace find_x_l364_36415

theorem find_x (x : ℝ) (h : (40 / 80) = Real.sqrt (x / 80)) : x = 20 := 
by 
  sorry

end find_x_l364_36415


namespace find_physics_marks_l364_36448

theorem find_physics_marks (P C M : ℕ) (h1 : P + C + M = 210) (h2 : P + M = 180) (h3 : P + C = 140) : P = 110 :=
sorry

end find_physics_marks_l364_36448


namespace average_weasels_caught_per_week_l364_36491

-- Definitions based on the conditions
def initial_weasels : ℕ := 100
def initial_rabbits : ℕ := 50
def foxes : ℕ := 3
def rabbits_caught_per_week_per_fox : ℕ := 2
def weeks : ℕ := 3
def remaining_animals : ℕ := 96

-- Main theorem statement
theorem average_weasels_caught_per_week :
  (foxes * weeks * rabbits_caught_per_week_per_fox +
   foxes * weeks * W = initial_weasels + initial_rabbits - remaining_animals) →
  W = 4 :=
sorry

end average_weasels_caught_per_week_l364_36491


namespace number_of_cars_repaired_l364_36400

theorem number_of_cars_repaired
  (oil_change_cost repair_cost car_wash_cost : ℕ)
  (oil_changes repairs car_washes total_earnings : ℕ)
  (h₁ : oil_change_cost = 20)
  (h₂ : repair_cost = 30)
  (h₃ : car_wash_cost = 5)
  (h₄ : oil_changes = 5)
  (h₅ : car_washes = 15)
  (h₆ : total_earnings = 475)
  (h₇ : 5 * oil_change_cost + 15 * car_wash_cost + repairs * repair_cost = total_earnings) :
  repairs = 10 :=
by sorry

end number_of_cars_repaired_l364_36400


namespace original_number_of_friends_l364_36450

theorem original_number_of_friends (F : ℕ) (h₁ : 5000 / F - 125 = 5000 / (F + 8)) : F = 16 :=
sorry

end original_number_of_friends_l364_36450


namespace fraction_meaningful_condition_l364_36464

theorem fraction_meaningful_condition (m : ℝ) : (m + 3 ≠ 0) → (m ≠ -3) :=
by
  intro h
  sorry

end fraction_meaningful_condition_l364_36464


namespace squirrel_nuts_collection_l364_36482

theorem squirrel_nuts_collection (n : ℕ) (e u : ℕ → ℕ) :
  (∀ k, 1 ≤ k ∧ k ≤ n → e k = u k + k) ∧
  (∀ k, 1 ≤ k ∧ k ≤ n → u k = e (k + 1) + u k / 100) ∧
  e n = n →
  n = 99 → 
  (∃ S : ℕ, (∀ k, 1 ≤ k ∧ k ≤ n → e k = S)) ∧ 
  S = 9801 :=
sorry

end squirrel_nuts_collection_l364_36482


namespace apple_harvest_l364_36458

theorem apple_harvest (sacks_per_section : ℕ) (num_sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 45 →
  num_sections = 8 →
  total_sacks = sacks_per_section * num_sections →
  total_sacks = 360 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end apple_harvest_l364_36458


namespace sin_25_over_6_pi_l364_36446

noncomputable def sin_value : ℝ :=
  Real.sin (25 / 6 * Real.pi)

theorem sin_25_over_6_pi : sin_value = 1 / 2 := by
  sorry

end sin_25_over_6_pi_l364_36446


namespace novels_next_to_each_other_l364_36451

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem novels_next_to_each_other (n_essays n_novels : Nat) (condition_novels : n_novels = 2) (condition_essays : n_essays = 3) :
  let total_units := (n_novels - 1) + n_essays
  factorial total_units * factorial n_novels = 48 :=
by
  sorry

end novels_next_to_each_other_l364_36451


namespace probability_divisible_by_3_l364_36490

-- Define the set of numbers
def S : Set ℕ := {2, 3, 5, 6}

-- Define the pairs of numbers whose product is divisible by 3
def valid_pairs : Set (ℕ × ℕ) := {(2, 3), (2, 6), (3, 5), (3, 6), (5, 6)}

-- Define the total number of pairs
def total_pairs := 6

-- Define the number of valid pairs
def valid_pairs_count := 5

-- Prove that the probability of choosing two numbers whose product is divisible by 3 is 5/6
theorem probability_divisible_by_3 : (valid_pairs_count / total_pairs : ℚ) = 5 / 6 := by
  sorry

end probability_divisible_by_3_l364_36490


namespace valid_outfits_l364_36436

-- Let's define the conditions first:
variable (shirts colors pairs : ℕ)

-- Suppose we have the following constraints according to the given problem:
def totalShirts : ℕ := 6
def totalPants : ℕ := 6
def totalHats : ℕ := 6
def totalShoes : ℕ := 6
def numOfColors : ℕ := 6

-- We refuse to wear an outfit in which all 4 items are the same color, or in which the shoes match the color of any other item.
theorem valid_outfits : 
  (totalShirts * totalPants * totalHats * (totalShoes - 1) + (totalShirts * 5 - totalShoes)) = 1104 :=
by sorry

end valid_outfits_l364_36436


namespace N_is_composite_l364_36449

def N : ℕ := 2011 * 2012 * 2013 * 2014 + 1

theorem N_is_composite : ¬ Prime N := by
  sorry

end N_is_composite_l364_36449


namespace distinct_real_numbers_condition_l364_36414

theorem distinct_real_numbers_condition (a b c : ℝ) (h_abc_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : (a / (b - c)) + (b / (c - a)) + (c / (a - b)) = 1) :
  (a / (b - c)^2) + (b / (c - a)^2) + (c / (a - b)^2) = 1 := 
by sorry

end distinct_real_numbers_condition_l364_36414


namespace jane_dolls_l364_36416

theorem jane_dolls (jane_dolls jill_dolls : ℕ) (h1 : jane_dolls + jill_dolls = 32) (h2 : jill_dolls = jane_dolls + 6) : jane_dolls = 13 := 
by {
  sorry
}

end jane_dolls_l364_36416


namespace sqrt_of_square_is_identity_l364_36478

variable {a : ℝ} (h : a > 0)

theorem sqrt_of_square_is_identity (h : a > 0) : Real.sqrt (a^2) = a := 
  sorry

end sqrt_of_square_is_identity_l364_36478


namespace total_students_correct_l364_36433

-- Define the given conditions
variables (A B C : ℕ)

-- Number of students in class B
def B_def : ℕ := 25

-- Number of students in class A (B is 8 fewer than A)
def A_def : ℕ := B_def + 8

-- Number of students in class C (C is 5 times B)
def C_def : ℕ := 5 * B_def

-- The total number of students
def total_students : ℕ := A_def + B_def + C_def

-- The proof statement
theorem total_students_correct : total_students = 183 := by
  sorry

end total_students_correct_l364_36433


namespace symmetry_origin_points_l364_36440

theorem symmetry_origin_points (x y : ℝ) (h₁ : (x, -2) = (-3, -y)) : x + y = -1 :=
sorry

end symmetry_origin_points_l364_36440


namespace simplify_radicals_l364_36444

open Real

theorem simplify_radicals : sqrt 72 + sqrt 32 = 10 * sqrt 2 := by
  sorry

end simplify_radicals_l364_36444


namespace power_problem_l364_36454

theorem power_problem (k : ℕ) (h : 6 ^ k = 4) : 6 ^ (2 * k + 3) = 3456 := 
by 
  sorry

end power_problem_l364_36454


namespace min_value_2x_plus_y_l364_36489

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/(y + 1) = 2) : 2 * x + y = 3 :=
sorry

end min_value_2x_plus_y_l364_36489


namespace exclusive_movies_count_l364_36495

-- Define the conditions
def shared_movies : Nat := 15
def andrew_movies : Nat := 25
def john_movies_exclusive : Nat := 8

-- Define the result calculation
def exclusive_movies (andrew_movies shared_movies john_movies_exclusive : Nat) : Nat :=
  (andrew_movies - shared_movies) + john_movies_exclusive

-- Statement to prove
theorem exclusive_movies_count : exclusive_movies andrew_movies shared_movies john_movies_exclusive = 18 := by
  sorry

end exclusive_movies_count_l364_36495


namespace range_of_b_l364_36463

theorem range_of_b (b : ℝ) (hb : b > 0) : (∃ x : ℝ, |x - 5| + |x - 10| > b) ↔ (0 < b ∧ b < 5) :=
by
  sorry

end range_of_b_l364_36463


namespace purely_imaginary_complex_l364_36443

theorem purely_imaginary_complex (a : ℝ) 
  (h₁ : a^2 + 2 * a - 3 = 0)
  (h₂ : a + 3 ≠ 0) : a = 1 := by
  sorry

end purely_imaginary_complex_l364_36443


namespace total_candy_bars_correct_l364_36471

-- Define the number of each type of candy bar.
def snickers : Nat := 3
def marsBars : Nat := 2
def butterfingers : Nat := 7

-- Define the total number of candy bars.
def totalCandyBars : Nat := snickers + marsBars + butterfingers

-- Formulate the theorem about the total number of candy bars.
theorem total_candy_bars_correct : totalCandyBars = 12 :=
sorry

end total_candy_bars_correct_l364_36471
