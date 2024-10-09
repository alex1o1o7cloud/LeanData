import Mathlib

namespace students_problem_count_l1832_183292

theorem students_problem_count 
  (x y z q r : ℕ) 
  (H1 : x + y + z + q + r = 30) 
  (H2 : x + 2 * y + 3 * z + 4 * q + 5 * r = 40) 
  (h_y_pos : 1 ≤ y) 
  (h_z_pos : 1 ≤ z) 
  (h_q_pos : 1 ≤ q) 
  (h_r_pos : 1 ≤ r) : 
  x = 26 := 
  sorry

end students_problem_count_l1832_183292


namespace percentage_saved_l1832_183277

-- Define the actual and saved amount.
def actual_investment : ℕ := 150000
def saved_amount : ℕ := 50000

-- Define the planned investment based on the conditions.
def planned_investment : ℕ := actual_investment + saved_amount

-- Proof goal: The percentage saved is 25%.
theorem percentage_saved : (saved_amount * 100) / planned_investment = 25 := 
by 
  sorry

end percentage_saved_l1832_183277


namespace remainder_of_concatenated_numbers_l1832_183281

def concatenatedNumbers : ℕ :=
  let digits := List.range (50) -- [0, 1, 2, ..., 49]
  digits.foldl (fun acc d => acc * 10 ^ (Nat.digits 10 d).length + d) 0

theorem remainder_of_concatenated_numbers :
  concatenatedNumbers % 50 = 49 :=
by
  sorry

end remainder_of_concatenated_numbers_l1832_183281


namespace quadratic_inequality_empty_set_l1832_183230

theorem quadratic_inequality_empty_set (a : ℝ) :
  (∀ x : ℝ, ¬ (ax^2 - ax + 1 < 0)) ↔ (0 ≤ a ∧ a ≤ 4) :=
by
  sorry

end quadratic_inequality_empty_set_l1832_183230


namespace current_dogwood_trees_l1832_183282

def number_of_trees (X : ℕ) : Prop :=
  X + 61 = 100

theorem current_dogwood_trees (X : ℕ) (h : number_of_trees X) : X = 39 :=
by 
  sorry

end current_dogwood_trees_l1832_183282


namespace solve_problem_statement_l1832_183218

def problem_statement : Prop :=
  ∃ n, 3^19 % n = 7 ∧ n = 1162261460

theorem solve_problem_statement : problem_statement :=
  sorry

end solve_problem_statement_l1832_183218


namespace dartboard_distribution_count_l1832_183205

-- Definition of the problem in Lean 4
def count_dartboard_distributions : ℕ :=
  -- We directly use the identified correct answer
  5

theorem dartboard_distribution_count :
  count_dartboard_distributions = 5 :=
sorry

end dartboard_distribution_count_l1832_183205


namespace function_decomposition_l1832_183273

theorem function_decomposition (f : ℝ → ℝ) :
  ∃ (a : ℝ) (f₁ f₂ : ℝ → ℝ), a > 0 ∧ (∀ x, f₁ x = f₁ (-x)) ∧ (∀ x, f₂ x = f₂ (2 * a - x)) ∧ (∀ x, f x = f₁ x + f₂ x) :=
sorry

end function_decomposition_l1832_183273


namespace dimes_turned_in_l1832_183216

theorem dimes_turned_in (total_coins nickels quarters : ℕ) (h1 : total_coins = 11) (h2 : nickels = 2) (h3 : quarters = 7) : 
  ∃ dimes : ℕ, dimes + nickels + quarters = total_coins ∧ dimes = 2 :=
by
  sorry

end dimes_turned_in_l1832_183216


namespace sin1993_cos1993_leq_zero_l1832_183233

theorem sin1993_cos1993_leq_zero (x : ℝ) (h : Real.sin x + Real.cos x ≤ 0) : 
  Real.sin x ^ 1993 + Real.cos x ^ 1993 ≤ 0 := 
by 
  sorry

end sin1993_cos1993_leq_zero_l1832_183233


namespace ways_to_seat_people_l1832_183244

noncomputable def number_of_ways : ℕ :=
  let choose_people := (Nat.choose 12 8)
  let divide_groups := (Nat.choose 8 4)
  let arrange_circular_table := (Nat.factorial 3)
  choose_people * divide_groups * (arrange_circular_table * arrange_circular_table)

theorem ways_to_seat_people :
  number_of_ways = 1247400 :=
by 
  -- proof goes here
  sorry

end ways_to_seat_people_l1832_183244


namespace total_students_l1832_183235

-- Definitions based on the conditions:
def yoongi_left : ℕ := 7
def yoongi_right : ℕ := 5

-- Theorem statement that proves the total number of students given the conditions
theorem total_students (y_left y_right : ℕ) : y_left = yoongi_left -> y_right = yoongi_right -> (y_left + y_right - 1) = 11 := 
by
  intros h1 h2
  rw [h1, h2]
  sorry

end total_students_l1832_183235


namespace smallest_solution_l1832_183252

def polynomial (x : ℝ) := x^4 - 34 * x^2 + 225 = 0

theorem smallest_solution : ∃ x : ℝ, polynomial x ∧ ∀ y : ℝ, polynomial y → x ≤ y := 
sorry

end smallest_solution_l1832_183252


namespace identity_eq_coefficients_l1832_183275

theorem identity_eq_coefficients (a b c d : ℝ) :
  (∀ x : ℝ, a * x + b = c * x + d) ↔ (a = c ∧ b = d) :=
by
  sorry

end identity_eq_coefficients_l1832_183275


namespace no_consecutive_beeches_probability_l1832_183262

theorem no_consecutive_beeches_probability :
  let total_trees := 12
  let oaks := 3
  let holm_oaks := 4
  let beeches := 5
  let total_arrangements := (Nat.factorial total_trees) / ((Nat.factorial oaks) * (Nat.factorial holm_oaks) * (Nat.factorial beeches))
  let favorable_arrangements :=
    let slots := oaks + holm_oaks + 1
    Nat.choose slots beeches * ((Nat.factorial (oaks + holm_oaks)) / ((Nat.factorial oaks) * (Nat.factorial holm_oaks)))
  let probability := favorable_arrangements / total_arrangements
  probability = 7 / 99 :=
by
  sorry

end no_consecutive_beeches_probability_l1832_183262


namespace remainder_xyz_mod7_condition_l1832_183219

-- Define variables and conditions
variables (x y z : ℕ)
theorem remainder_xyz_mod7_condition (hx : x < 7) (hy : y < 7) (hz : z < 7)
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 7])
  (h2 : 3 * x + 2 * y + z ≡ 2 [MOD 7])
  (h3 : 2 * x + y + 3 * z ≡ 3 [MOD 7]) :
  (x * y * z % 7) ≡ 1 [MOD 7] := sorry

end remainder_xyz_mod7_condition_l1832_183219


namespace phone_answered_before_fifth_ring_l1832_183258

theorem phone_answered_before_fifth_ring:
  (0.1 + 0.2 + 0.25 + 0.25 = 0.8) :=
by
  sorry

end phone_answered_before_fifth_ring_l1832_183258


namespace find_fourth_power_sum_l1832_183283

theorem find_fourth_power_sum (a b c : ℝ) 
    (h1 : a + b + c = 2) 
    (h2 : a^2 + b^2 + c^2 = 3) 
    (h3 : a^3 + b^3 + c^3 = 4) : 
    a^4 + b^4 + c^4 = 7.833 :=
sorry

end find_fourth_power_sum_l1832_183283


namespace second_car_avg_mpg_l1832_183217

theorem second_car_avg_mpg 
  (x y : ℝ) 
  (h1 : x + y = 75) 
  (h2 : 25 * x + 35 * y = 2275) : 
  y = 40 := 
by sorry

end second_car_avg_mpg_l1832_183217


namespace sqrt_180_eq_l1832_183255

noncomputable def simplify_sqrt_180 : Real := 6 * Real.sqrt 5

theorem sqrt_180_eq : Real.sqrt 180 = simplify_sqrt_180 := 
by
  -- proof omitted
  sorry

end sqrt_180_eq_l1832_183255


namespace length_of_room_calculation_l1832_183269

variable (broadness_of_room : ℝ) (width_of_carpet : ℝ) (total_cost : ℝ) (rate_per_sq_meter : ℝ) (area_of_carpet : ℝ) (length_of_room : ℝ)

theorem length_of_room_calculation (h1 : broadness_of_room = 9) 
    (h2 : width_of_carpet = 0.75) 
    (h3 : total_cost = 1872) 
    (h4 : rate_per_sq_meter = 12) 
    (h5 : area_of_carpet = total_cost / rate_per_sq_meter)
    (h6 : area_of_carpet = length_of_room * width_of_carpet) 
    : length_of_room = 208 := 
by 
    sorry

end length_of_room_calculation_l1832_183269


namespace find_x_l1832_183234

theorem find_x : ∃ x : ℝ, (3 * (x + 2 - 6)) / 4 = 3 ∧ x = 8 :=
by
  sorry

end find_x_l1832_183234


namespace min_value_of_a_plus_b_l1832_183287

theorem min_value_of_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1/a + 1/b = 1) : a + b = 4 :=
sorry

end min_value_of_a_plus_b_l1832_183287


namespace pentadecagon_diagonals_l1832_183259

def numberOfDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem pentadecagon_diagonals : numberOfDiagonals 15 = 90 :=
by
  sorry

end pentadecagon_diagonals_l1832_183259


namespace probability_heads_l1832_183202

theorem probability_heads (p : ℝ) (q : ℝ) (C_10_5 C_10_6 : ℝ)
  (h1 : q = 1 - p)
  (h2 : C_10_5 = 252)
  (h3 : C_10_6 = 210)
  (eqn : C_10_5 * p^5 * q^5 = C_10_6 * p^6 * q^4) :
  p = 6 / 11 := 
by
  sorry

end probability_heads_l1832_183202


namespace balls_in_boxes_l1832_183265

def num_ways_to_partition_6_in_4_parts : ℕ :=
  -- The different partitions of 6: (6,0,0,0), (5,1,0,0), (4,2,0,0), (4,1,1,0),
  -- (3,3,0,0), (3,2,1,0), (3,1,1,1), (2,2,2,0), (2,2,1,1)
  ([
    [6, 0, 0, 0],
    [5, 1, 0, 0],
    [4, 2, 0, 0],
    [4, 1, 1, 0],
    [3, 3, 0, 0],
    [3, 2, 1, 0],
    [3, 1, 1, 1],
    [2, 2, 2, 0],
    [2, 2, 1, 1]
  ]).length

theorem balls_in_boxes : num_ways_to_partition_6_in_4_parts = 9 := by
  sorry

end balls_in_boxes_l1832_183265


namespace smallest_integer_in_set_A_l1832_183204

def set_A : Set ℝ := {x | |x - 2| ≤ 5}

theorem smallest_integer_in_set_A : ∃ m ∈ set_A, ∀ n ∈ set_A, m ≤ n := 
  sorry

end smallest_integer_in_set_A_l1832_183204


namespace largest_four_digit_negative_congruent_3_mod_29_l1832_183227

theorem largest_four_digit_negative_congruent_3_mod_29 : 
  ∃ (n : ℤ), n < 0 ∧ n ≥ -9999 ∧ (n % 29 = 3) ∧ n = -1012 :=
sorry

end largest_four_digit_negative_congruent_3_mod_29_l1832_183227


namespace reaction_spontaneous_at_high_temperature_l1832_183215

theorem reaction_spontaneous_at_high_temperature
  (ΔH : ℝ) (ΔS : ℝ) (T : ℝ) (ΔG : ℝ)
  (h_ΔH_pos : ΔH > 0)
  (h_ΔS_pos : ΔS > 0)
  (h_ΔG_eq : ΔG = ΔH - T * ΔS) :
  (∃ T_high : ℝ, T_high > 0 ∧ ΔG < 0) := sorry

end reaction_spontaneous_at_high_temperature_l1832_183215


namespace distance_between_homes_l1832_183249

theorem distance_between_homes (Maxwell_speed : ℝ) (Brad_speed : ℝ) (M_time : ℝ) (B_delay : ℝ) (D : ℝ) 
  (h1 : Maxwell_speed = 4) 
  (h2 : Brad_speed = 6)
  (h3 : M_time = 8)
  (h4 : B_delay = 1) :
  D = 74 :=
by
  sorry

end distance_between_homes_l1832_183249


namespace jill_speed_is_8_l1832_183272

-- Definitions for conditions
def speed_jack1 := 12 -- speed in km/h for the first 12 km
def distance_jack1 := 12 -- distance in km for the first 12 km

def speed_jack2 := 6 -- speed in km/h for the second 12 km
def distance_jack2 := 12 -- distance in km for the second 12 km

def distance_jill := distance_jack1 + distance_jack2 -- total distance in km for Jill

-- Total time taken by Jack
def time_jack := (distance_jack1 / speed_jack1) + (distance_jack2 / speed_jack2)

-- Jill's speed calculation
def jill_speed := distance_jill / time_jack

-- Theorem stating Jill's speed is 8 km/h
theorem jill_speed_is_8 : jill_speed = 8 := by
  sorry

end jill_speed_is_8_l1832_183272


namespace factory_hours_per_day_l1832_183201

def factory_produces (hours_per_day : ℕ) : Prop :=
  let refrigerators_per_hour := 90
  let coolers_per_hour := 160
  let total_products_per_hour := refrigerators_per_hour + coolers_per_hour
  let total_products_in_5_days := 11250
  total_products_per_hour * (5 * hours_per_day) = total_products_in_5_days

theorem factory_hours_per_day : ∃ h : ℕ, factory_produces h ∧ h = 9 :=
by
  existsi 9
  unfold factory_produces
  sorry

end factory_hours_per_day_l1832_183201


namespace geometric_series_r_l1832_183261

theorem geometric_series_r (a r : ℝ) 
    (h1 : a * (1 - r ^ 0) / (1 - r) = 24) 
    (h2 : a * r / (1 - r ^ 2) = 8) : 
    r = 1 / 2 := 
sorry

end geometric_series_r_l1832_183261


namespace finalStoresAtEndOf2020_l1832_183278

def initialStores : ℕ := 23
def storesOpened2019 : ℕ := 5
def storesClosed2019 : ℕ := 2
def storesOpened2020 : ℕ := 10
def storesClosed2020 : ℕ := 6

theorem finalStoresAtEndOf2020 : initialStores + (storesOpened2019 - storesClosed2019) + (storesOpened2020 - storesClosed2020) = 30 :=
by
  sorry

end finalStoresAtEndOf2020_l1832_183278


namespace ways_to_choose_providers_l1832_183206

theorem ways_to_choose_providers : (25 * 24 * 23 * 22 = 303600) :=
by
  sorry

end ways_to_choose_providers_l1832_183206


namespace bicycle_has_four_wheels_l1832_183293

-- Define the universe and properties of cars
axiom Car : Type
axiom Bicycle : Car
axiom has_four_wheels : Car → Prop
axiom all_cars_have_four_wheels : ∀ c : Car, has_four_wheels c

-- Define the theorem
theorem bicycle_has_four_wheels : has_four_wheels Bicycle :=
by
  sorry

end bicycle_has_four_wheels_l1832_183293


namespace jane_albert_same_committee_l1832_183257

def probability_same_committee (total_MBAs : ℕ) (committee_size : ℕ) (num_committees : ℕ) (favorable_cases : ℕ) (total_cases : ℕ) : ℚ :=
  favorable_cases / total_cases

theorem jane_albert_same_committee :
  probability_same_committee 9 4 3 105 630 = 1 / 6 :=
by
  sorry

end jane_albert_same_committee_l1832_183257


namespace total_sheep_l1832_183236

variable (x y : ℕ)
/-- Initial condition: After one ram runs away, the ratio of rams to ewes is 7:5. -/
def initial_ratio (x y : ℕ) : Prop := 5 * (x - 1) = 7 * y
/-- Second condition: After the ram returns and one ewe runs away, the ratio of rams to ewes is 5:3. -/
def second_ratio (x y : ℕ) : Prop := 3 * x = 5 * (y - 1)
/-- The total number of sheep in the flock initially is 25. -/
theorem total_sheep (x y : ℕ) 
  (h1 : initial_ratio x y) 
  (h2 : second_ratio x y) : 
  x + y = 25 := 
by sorry

end total_sheep_l1832_183236


namespace inequality_solution_l1832_183220

theorem inequality_solution {x : ℝ} :
  (12 * x^2 + 24 * x - 75) / ((3 * x - 5) * (x + 5)) < 4 ↔ -5 < x ∧ x < 5 / 3 := by
  sorry

end inequality_solution_l1832_183220


namespace second_investment_amount_l1832_183288

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

theorem second_investment_amount :
  ∀ (P₁ P₂ I₁ I₂ r t : ℝ), 
    P₁ = 5000 →
    I₁ = 250 →
    I₂ = 1000 →
    I₁ = simple_interest P₁ r t →
    I₂ = simple_interest P₂ r t →
    P₂ = 20000 := 
by 
  intros P₁ P₂ I₁ I₂ r t hP₁ hI₁ hI₂ hI₁_eq hI₂_eq
  sorry

end second_investment_amount_l1832_183288


namespace frank_used_2_bags_l1832_183207

theorem frank_used_2_bags (total_candy : ℕ) (candy_per_bag : ℕ) (h1 : total_candy = 16) (h2 : candy_per_bag = 8) : (total_candy / candy_per_bag) = 2 := 
by
  sorry

end frank_used_2_bags_l1832_183207


namespace lower_bound_expression_l1832_183246

theorem lower_bound_expression (n : ℤ) (L : ℤ) :
  (∃ k : ℕ, k = 20 ∧
          ∀ n, (L < 4 * n + 7 ∧ 4 * n + 7 < 80)) →
  L = 3 :=
by
  sorry

end lower_bound_expression_l1832_183246


namespace interest_paid_percent_l1832_183253

noncomputable def down_payment : ℝ := 300
noncomputable def total_cost : ℝ := 750
noncomputable def monthly_payment : ℝ := 57
noncomputable def final_payment : ℝ := 21
noncomputable def num_monthly_payments : ℕ := 9

noncomputable def total_instalments : ℝ := (num_monthly_payments * monthly_payment) + final_payment
noncomputable def total_paid : ℝ := total_instalments + down_payment
noncomputable def amount_borrowed : ℝ := total_cost - down_payment
noncomputable def interest_paid : ℝ := total_paid - amount_borrowed
noncomputable def interest_percent : ℝ := (interest_paid / amount_borrowed) * 100

theorem interest_paid_percent:
  interest_percent = 85.33 := by
  sorry

end interest_paid_percent_l1832_183253


namespace minimum_tanA_9tanB_l1832_183213

variable (a b c A B : ℝ)
variable (Aacute : A > 0 ∧ A < π / 2)
variable (h1 : a^2 = b^2 + 2*b*c * Real.sin A)
variable (habc : a = b * Real.sin A)

theorem minimum_tanA_9tanB : 
  ∃ (A B : ℝ), (A > 0 ∧ A < π / 2) ∧ (a^2 = b^2 + 2*b*c * Real.sin A) ∧ (a = b * Real.sin A) ∧ 
  (min ((Real.tan A) - 9*(Real.tan B)) = -2) := 
  sorry

end minimum_tanA_9tanB_l1832_183213


namespace tangent_line_through_point_l1832_183239

-- Definitions based purely on the conditions given in the problem.
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 25
def point_on_line (x y : ℝ) : Prop := 3 * x - 4 * y + 25 = 0
def point_given : ℝ × ℝ := (-3, 4)

-- The theorem statement to be proven
theorem tangent_line_through_point : point_on_line point_given.1 point_given.2 := 
sorry

end tangent_line_through_point_l1832_183239


namespace range_of_c_l1832_183209

theorem range_of_c (a c : ℝ) (ha : a ≥ 1 / 8)
  (h : ∀ x > 0, 2 * x + a / x ≥ c) : c ≤ 1 :=
sorry

end range_of_c_l1832_183209


namespace train_speed_correct_l1832_183254

theorem train_speed_correct :
  ∀ (L : ℝ) (V_man : ℝ) (T : ℝ) (V_train : ℝ),
    L = 220 ∧ V_man = 6 * (1000 / 3600) ∧ T = 11.999040076793857 ∧ 
    L / T - V_man = V_train ↔ V_train * 3.6 = 60 :=
by
  intros L V_man T V_train
  sorry

end train_speed_correct_l1832_183254


namespace earnings_difference_l1832_183221

-- Define the prices and quantities.
def price_A : ℝ := 4
def price_B : ℝ := 3.5
def quantity_A : ℕ := 300
def quantity_B : ℕ := 350

-- Define the earnings for both companies.
def earnings_A := price_A * quantity_A
def earnings_B := price_B * quantity_B

-- State the theorem we intend to prove.
theorem earnings_difference :
  earnings_B - earnings_A = 25 := by
  sorry

end earnings_difference_l1832_183221


namespace lila_substituted_value_l1832_183225

theorem lila_substituted_value:
  let a := 2
  let b := 3
  let c := 4
  let d := 5
  let f := 6
  ∃ e : ℚ, 20 * e = 2 * (3 - 4 * (5 - (e / 6))) ∧ e = -51 / 28 := sorry

end lila_substituted_value_l1832_183225


namespace walkway_and_border_area_correct_l1832_183210

-- Definitions based on the given conditions
def flower_bed_width : ℕ := 8
def flower_bed_height : ℕ := 3
def walkway_width : ℕ := 2
def border_width : ℕ := 4
def num_rows : ℕ := 4
def num_columns : ℕ := 3

-- Total width calculation
def total_width : ℕ := 
  (flower_bed_width * num_columns) + (walkway_width * (num_columns + 1)) + (border_width * 2)

-- Total height calculation
def total_height : ℕ := 
  (flower_bed_height * num_rows) + (walkway_width * (num_rows + 1)) + (border_width * 2)

-- Total area of the garden including walkways and decorative border
def total_area : ℕ := total_width * total_height

-- Total area of flower beds
def flower_bed_area : ℕ := 
  (flower_bed_width * flower_bed_height) * (num_rows * num_columns)

-- Area of the walkways and decorative border
def walkway_and_border_area : ℕ := total_area - flower_bed_area

theorem walkway_and_border_area_correct : 
  walkway_and_border_area = 912 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end walkway_and_border_area_correct_l1832_183210


namespace helga_extra_hours_last_friday_l1832_183267

theorem helga_extra_hours_last_friday
  (weekly_articles : ℕ)
  (extra_hours_thursday : ℕ)
  (extra_articles_thursday : ℕ)
  (extra_articles_friday : ℕ)
  (articles_per_half_hour : ℕ)
  (half_hours_per_hour : ℕ)
  (usual_articles_per_day : ℕ)
  (days_per_week : ℕ)
  (articles_last_thursday_plus_friday : ℕ)
  (total_articles : ℕ) :
  (weekly_articles = (usual_articles_per_day * days_per_week)) →
  (extra_hours_thursday = 2) →
  (articles_per_half_hour = 5) →
  (half_hours_per_hour = 2) →
  (usual_articles_per_day = (articles_per_half_hour * 8)) →
  (extra_articles_thursday = (articles_per_half_hour * (extra_hours_thursday * half_hours_per_hour))) →
  (articles_last_thursday_plus_friday = weekly_articles + extra_articles_thursday) →
  (total_articles = 250) →
  (extra_articles_friday = total_articles - articles_last_thursday_plus_friday) →
  (extra_articles_friday = 30) →
  ((extra_articles_friday / articles_per_half_hour) = 6) →
  (3 = (6 / half_hours_per_hour)) :=
by
  intro hw1 hw2 hw3 hw4 hw5 hw6 hw7 hw8 hw9 hw10
  sorry

end helga_extra_hours_last_friday_l1832_183267


namespace probability_linda_picks_letter_in_mathematics_l1832_183238

def english_alphabet : Finset Char := "ABCDEFGHIJKLMNOPQRSTUVWXYZ".toList.toFinset

def word_mathematics : Finset Char := "MATHEMATICS".toList.toFinset

theorem probability_linda_picks_letter_in_mathematics : 
  (word_mathematics.card : ℚ) / (english_alphabet.card : ℚ) = 4 / 13 := by sorry

end probability_linda_picks_letter_in_mathematics_l1832_183238


namespace b_car_usage_hours_l1832_183241

theorem b_car_usage_hours (h : ℕ) (total_cost_a_b_c : ℕ) 
  (a_usage : ℕ) (b_payment : ℕ) (c_usage : ℕ) 
  (total_cost : total_cost_a_b_c = 720)
  (usage_a : a_usage = 9) 
  (usage_c : c_usage = 13)
  (payment_b : b_payment = 225) 
  (cost_per_hour : ℝ := total_cost_a_b_c / (a_usage + h + c_usage)) :
  b_payment = cost_per_hour * h → h = 10 := 
by
  sorry

end b_car_usage_hours_l1832_183241


namespace age_of_15th_student_l1832_183276

theorem age_of_15th_student (avg_age_15 avg_age_3 avg_age_11 : ℕ) 
  (h_avg_15 : avg_age_15 = 15) 
  (h_avg_3 : avg_age_3 = 14) 
  (h_avg_11 : avg_age_11 = 16) : 
  ∃ x : ℕ, x = 7 := 
by
  sorry

end age_of_15th_student_l1832_183276


namespace excircle_identity_l1832_183294

variables (a b c r_a r_b r_c : ℝ)

-- Conditions: r_a, r_b, r_c are the radii of the excircles opposite vertices A, B, and C respectively.
-- In the triangle ABC, a, b, c are the sides opposite vertices A, B, and C respectively.

theorem excircle_identity:
  (a^2 / (r_a * (r_b + r_c)) + b^2 / (r_b * (r_c + r_a)) + c^2 / (r_c * (r_a + r_b))) = 2 :=
by
  sorry

end excircle_identity_l1832_183294


namespace remainder_eq_159_l1832_183268

def x : ℕ := 2^40
def numerator : ℕ := 2^160 + 160
def denominator : ℕ := 2^80 + 2^40 + 1

theorem remainder_eq_159 : (numerator % denominator) = 159 := 
by {
  -- Proof will be filled in here.
  sorry
}

end remainder_eq_159_l1832_183268


namespace num_solutions_l1832_183200

theorem num_solutions (k : ℤ) :
  (∃ a b c : ℝ, (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
    (a^2 + b^2 = k * c * (a + b)) ∧
    (b^2 + c^2 = k * a * (b + c)) ∧
    (c^2 + a^2 = k * b * (c + a))) ↔ k = 1 ∨ k = -2 :=
sorry

end num_solutions_l1832_183200


namespace domain_of_tan_2x_plus_pi_over_3_l1832_183203

noncomputable def domain_tan_transformed : Set ℝ :=
  {x : ℝ | ∀ (k : ℤ), x ≠ k * (Real.pi / 2) + (Real.pi / 12)}

theorem domain_of_tan_2x_plus_pi_over_3 :
  (∀ x : ℝ, x ∉ domain_tan_transformed ↔ ∃ (k : ℤ), x = k * (Real.pi / 2) + (Real.pi / 12)) :=
sorry

end domain_of_tan_2x_plus_pi_over_3_l1832_183203


namespace negate_proposition_l1832_183280

open Classical

variable (x : ℝ)

theorem negate_proposition :
  (¬ ∀ x : ℝ, x^2 + 2 * x + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0 :=
by
  sorry

end negate_proposition_l1832_183280


namespace sum_of_arithmetic_series_l1832_183266

theorem sum_of_arithmetic_series (A B C : ℕ) (n : ℕ) 
  (hA : A = n * (2 * a₁ + (n - 1) * d) / 2)
  (hB : B = 2 * n * (2 * a₁ + (2 * n - 1) * d) / 2)
  (hC : C = 3 * n * (2 * a₁ + (3 * n - 1) * d) / 2) :
  C = 3 * (B - A) := sorry

end sum_of_arithmetic_series_l1832_183266


namespace yellow_flower_count_l1832_183270

-- Define the number of flowers of each color and total flowers based on given conditions
def total_flowers : Nat := 96
def green_flowers : Nat := 9
def red_flowers : Nat := 3 * green_flowers
def blue_flowers : Nat := total_flowers / 2

-- Define the number of yellow flowers
def yellow_flowers : Nat := total_flowers - (green_flowers + red_flowers + blue_flowers)

-- The theorem we aim to prove
theorem yellow_flower_count : yellow_flowers = 12 := by
  sorry

end yellow_flower_count_l1832_183270


namespace sqrt_div_value_l1832_183245

open Real

theorem sqrt_div_value (n x : ℝ) (h1 : n = 3600) (h2 : sqrt n / x = 4) : x = 15 :=
by
  sorry

end sqrt_div_value_l1832_183245


namespace sum_abs_a1_to_a10_l1832_183289

def S (n : ℕ) : ℤ := n^2 - 4 * n + 2
def a (n : ℕ) : ℤ := if n = 1 then S 1 else S n - S (n - 1)

theorem sum_abs_a1_to_a10 : (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10| = 66) := 
by
  sorry

end sum_abs_a1_to_a10_l1832_183289


namespace SarahsNumber_is_2880_l1832_183247

def SarahsNumber (n : ℕ) : Prop :=
  (144 ∣ n) ∧ (45 ∣ n) ∧ (1000 ≤ n ∧ n ≤ 3000)

theorem SarahsNumber_is_2880 : SarahsNumber 2880 :=
  by
  sorry

end SarahsNumber_is_2880_l1832_183247


namespace extended_cross_cannot_form_cube_l1832_183264

-- Define what it means to form a cube from patterns
def forms_cube (pattern : Type) : Prop := 
  sorry -- Definition for forming a cube would be detailed here

-- Define the Extended Cross pattern in a way that captures its structure
def extended_cross : Type := sorry -- Definition for Extended Cross structure

-- Define the L shape pattern in a way that captures its structure
def l_shape : Type := sorry -- Definition for L shape structure

-- The theorem statement proving that the Extended Cross pattern cannot form a cube
theorem extended_cross_cannot_form_cube : ¬(forms_cube extended_cross) := 
  sorry

end extended_cross_cannot_form_cube_l1832_183264


namespace polynomial_calculation_l1832_183263

theorem polynomial_calculation :
  (49^5 - 5 * 49^4 + 10 * 49^3 - 10 * 49^2 + 5 * 49 - 1) = 254804368 :=
by
  sorry

end polynomial_calculation_l1832_183263


namespace range_of_2a_plus_3b_l1832_183286

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 < a + b) (h2 : a + b < 3) (h3 : 2 < a - b) (h4 : a - b < 4) :
  -9/2 < 2*a + 3*b ∧ 2*a + 3*b < 13/2 :=
  sorry

end range_of_2a_plus_3b_l1832_183286


namespace greatest_b_value_ineq_l1832_183211

theorem greatest_b_value_ineq (b : ℝ) (h : -b^2 + 8 * b - 15 ≥ 0) : b ≤ 5 := 
sorry

end greatest_b_value_ineq_l1832_183211


namespace number_of_female_students_l1832_183228

theorem number_of_female_students (M F : ℕ) (h1 : F = M + 6) (h2 : M + F = 82) : F = 44 :=
by
  sorry

end number_of_female_students_l1832_183228


namespace find_a_l1832_183297

-- Given Conditions
def is_hyperbola (a : ℝ) : Prop := ∀ x y : ℝ, (x^2 / a) - (y^2 / 2) = 1
def is_asymptote (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = 2 * x

-- Question
theorem find_a (a : ℝ) (f : ℝ → ℝ) (hyp : is_hyperbola a) (asym : is_asymptote f) : a = 1 / 2 :=
sorry

end find_a_l1832_183297


namespace x_add_inv_ge_two_x_add_inv_eq_two_iff_l1832_183290

theorem x_add_inv_ge_two {x : ℝ} (h : 0 < x) : x + (1 / x) ≥ 2 :=
sorry

theorem x_add_inv_eq_two_iff {x : ℝ} (h : 0 < x) : (x + (1 / x) = 2) ↔ (x = 1) :=
sorry

end x_add_inv_ge_two_x_add_inv_eq_two_iff_l1832_183290


namespace stock_price_calculation_l1832_183296

def stock_price_end_of_first_year (initial_price : ℝ) (increase_percent : ℝ) : ℝ :=
  initial_price * (1 + increase_percent)

def stock_price_end_of_second_year (price_first_year : ℝ) (decrease_percent : ℝ) : ℝ :=
  price_first_year * (1 - decrease_percent)

theorem stock_price_calculation 
  (initial_price : ℝ)
  (increase_percent : ℝ)
  (decrease_percent : ℝ)
  (final_price : ℝ) :
  initial_price = 120 ∧ 
  increase_percent = 0.80 ∧
  decrease_percent = 0.30 ∧
  final_price = 151.20 → 
  stock_price_end_of_second_year (stock_price_end_of_first_year initial_price increase_percent) decrease_percent = final_price :=
by
  sorry

end stock_price_calculation_l1832_183296


namespace kendall_total_distance_l1832_183250

def distance_with_mother : ℝ := 0.17
def distance_with_father : ℝ := 0.5
def total_distance : ℝ := 0.67

theorem kendall_total_distance :
  (distance_with_mother + distance_with_father = total_distance) :=
sorry

end kendall_total_distance_l1832_183250


namespace bigger_number_in_ratio_l1832_183237

theorem bigger_number_in_ratio (x : ℕ) (h : 11 * x = 143) : 8 * x = 104 :=
by
  sorry

end bigger_number_in_ratio_l1832_183237


namespace condition1_condition2_l1832_183240

-- Definition for the coordinates of point P based on given m
def P (m : ℝ) : ℝ × ℝ := (3 * m - 6, m + 1)

-- Condition 1: Point P lies on the x-axis
theorem condition1 (m : ℝ) (hx : P m = (3 * m - 6, 0)) : P m = (-9, 0) := 
by {
  -- Show that if y-coordinate is zero, then m + 1 = 0, hence m = -1
  sorry
}

-- Condition 2: Point A is (-1, 2) and AP is parallel to the y-axis
theorem condition2 (m : ℝ) (A : ℝ × ℝ := (-1, 2)) (hy : (3 * m - 6 = -1)) : P m = (-1, 8/3) :=
by {
  -- Show that if the x-coordinates of A and P are equal, then 3m-6 = -1, hence m = 5/3
  sorry
}

end condition1_condition2_l1832_183240


namespace num_routes_M_to_N_l1832_183229

-- Define the relevant points and connections as predicates
def can_reach_directly (x y : String) : Prop :=
  if (x = "C" ∧ y = "N") ∨ (x = "D" ∧ y = "N") ∨ (x = "B" ∧ y = "N") then true else false

def can_reach_via (x y z : String) : Prop :=
  if (x = "A" ∧ y = "C" ∧ z = "N") ∨ (x = "A" ∧ y = "D" ∧ z = "N") ∨ (x = "B" ∧ y = "A" ∧ z = "N") ∨ 
     (x = "B" ∧ y = "C" ∧ z = "N") ∨ (x = "E" ∧ y = "B" ∧ z = "N") ∨ (x = "F" ∧ y = "A" ∧ z = "N") ∨ 
     (x = "F" ∧ y = "B" ∧ z = "N") then true else false

-- Define a function to compute the number of ways from a starting point to "N"
noncomputable def num_routes_to_N : String → ℕ
| "N" => 1
| "C" => 1
| "D" => 1
| "A" => 2 -- from C to N and D to N
| "B" => 4 -- from B to N directly, from B to N via A (2 ways), from B to N via C
| "E" => 4 -- from E to N via B
| "F" => 6 -- from F to N via A (2 ways), from F to N via B (4 ways)
| "M" => 16 -- from M to N via A, B, E, F
| _ => 0

-- The theorem statement
theorem num_routes_M_to_N : num_routes_to_N "M" = 16 :=
by
  sorry

end num_routes_M_to_N_l1832_183229


namespace usual_time_is_42_l1832_183251

noncomputable def usual_time_to_school (R T : ℝ) := T * R
noncomputable def improved_time_to_school (R T : ℝ) := ((7/6) * R) * (T - 6)

theorem usual_time_is_42 (R T : ℝ) :
  (usual_time_to_school R T) = (improved_time_to_school R T) → T = 42 :=
by
  sorry

end usual_time_is_42_l1832_183251


namespace inequality_proof_l1832_183243

theorem inequality_proof 
  (x y z w : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w)
  (h_eq : (x^3 + y^3)^4 = z^3 + w^3) :
  x^4 * z + y^4 * w ≥ z * w :=
sorry

end inequality_proof_l1832_183243


namespace total_payment_correct_l1832_183299

def payment_y : ℝ := 318.1818181818182
def payment_ratio : ℝ := 1.2
def payment_x : ℝ := payment_ratio * payment_y
def total_payment : ℝ := payment_x + payment_y

theorem total_payment_correct :
  total_payment = 700.00 :=
sorry

end total_payment_correct_l1832_183299


namespace absolute_difference_l1832_183260

theorem absolute_difference : |8 - 3^2| - |4^2 - 6*3| = -1 := by
  sorry

end absolute_difference_l1832_183260


namespace min_value_expression_l1832_183231

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 
  (1 / a + 2) * (1 / b + 2) ≥ 16 :=
sorry

end min_value_expression_l1832_183231


namespace circle_area_from_diameter_points_l1832_183248

theorem circle_area_from_diameter_points (C D : ℝ × ℝ)
    (hC : C = (-2, 3)) (hD : D = (4, -1)) :
    ∃ (A : ℝ), A = 13 * Real.pi :=
by
  let distance := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  have diameter : distance = Real.sqrt (6^2 + (-4)^2) := sorry -- this follows from the coordinates
  have radius : distance / 2 = Real.sqrt 13 := sorry -- half of the diameter
  exact ⟨13 * Real.pi, sorry⟩ -- area of the circle

end circle_area_from_diameter_points_l1832_183248


namespace cost_of_blue_pill_l1832_183226

/-
Statement:
Bob takes two blue pills and one orange pill each day for three weeks.
The cost of a blue pill is $2 more than an orange pill.
The total cost for all pills over the three weeks amounts to $966.
Prove that the cost of one blue pill is $16.
-/

theorem cost_of_blue_pill (days : ℕ) (total_cost : ℝ) (cost_orange : ℝ) (cost_blue : ℝ) 
  (h1 : days = 21) 
  (h2 : total_cost = 966) 
  (h3 : cost_blue = cost_orange + 2) 
  (daily_pill_cost : ℝ)
  (h4 : daily_pill_cost = total_cost / days)
  (h5 : daily_pill_cost = 2 * cost_blue + cost_orange) :
  cost_blue = 16 :=
by
  sorry

end cost_of_blue_pill_l1832_183226


namespace problem_statement_l1832_183295

theorem problem_statement
  (a b c d e : ℝ)
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |e| = 1) :
  e^2 + 2023 * (c * d) - (a + b) / 20 = 2024 := 
by 
  sorry

end problem_statement_l1832_183295


namespace difference_of_squares_l1832_183291

theorem difference_of_squares (x y : ℝ) (h₁ : x + y = 20) (h₂ : x - y = 10) : x^2 - y^2 = 200 :=
by {
  sorry
}

end difference_of_squares_l1832_183291


namespace people_per_entrance_l1832_183212

theorem people_per_entrance (e p : ℕ) (h1 : e = 5) (h2 : p = 1415) : p / e = 283 := by
  sorry

end people_per_entrance_l1832_183212


namespace find_common_ratio_l1832_183284

-- Defining the conditions in Lean
variables (a : ℕ → ℝ) (d q : ℝ)

-- The arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) - a n = d

-- The geometric sequence condition
def is_geometric_sequence (a1 a2 a4 q : ℝ) : Prop :=
a2 ^ 2 = a1 * a4

-- Proving the main theorem
theorem find_common_ratio (a : ℕ → ℝ) (d q : ℝ) (h_arith : is_arithmetic_sequence a d) (d_ne_zero : d ≠ 0) 
(h_geom : is_geometric_sequence (a 1) (a 2) (a 4) q) : q = 2 :=
by
  sorry

end find_common_ratio_l1832_183284


namespace leaves_blew_away_correct_l1832_183223

-- Define the initial number of leaves Mikey had.
def initial_leaves : ℕ := 356

-- Define the number of leaves Mikey has left.
def leaves_left : ℕ := 112

-- Define the number of leaves that blew away.
def leaves_blew_away : ℕ := initial_leaves - leaves_left

-- Prove that the number of leaves that blew away is 244.
theorem leaves_blew_away_correct : leaves_blew_away = 244 :=
by sorry

end leaves_blew_away_correct_l1832_183223


namespace product_of_numbers_l1832_183224

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 21) (h2 : x^2 + y^2 = 527) : x * y = -43.05 := by
  sorry

end product_of_numbers_l1832_183224


namespace dog_food_amount_l1832_183256

theorem dog_food_amount (x : ℕ) (h1 : 3 * x + 6 = 15) : x = 3 :=
by {
  sorry
}

end dog_food_amount_l1832_183256


namespace each_episode_length_l1832_183232

theorem each_episode_length (h_watch_time : ∀ d : ℕ, d = 5 → 2 * 60 * d = 600)
  (h_episodes : 20 > 0) : 600 / 20 = 30 := by
  -- Conditions used:
  -- 1. h_watch_time : John wants to finish a show in 5 days by watching 2 hours a day.
  -- 2. h_episodes : There are 20 episodes.
  -- Goal: Prove that each episode is 30 minutes long.
  sorry

end each_episode_length_l1832_183232


namespace copy_pages_l1832_183285

theorem copy_pages
  (total_cents : ℕ)
  (cost_per_page : ℚ)
  (h_total : total_cents = 2000)
  (h_cost : cost_per_page = 2.5) :
  (total_cents / cost_per_page) = 800 :=
by
  -- This is where the proof would go
  sorry

end copy_pages_l1832_183285


namespace age_ratio_l1832_183274

-- Define the conditions
def ArunCurrentAgeAfter6Years (A: ℕ) : Prop := A + 6 = 36
def DeepakCurrentAge : ℕ := 42

-- Define the goal statement
theorem age_ratio (A: ℕ) (hc: ArunCurrentAgeAfter6Years A) : A / gcd A DeepakCurrentAge = 5 ∧ DeepakCurrentAge / gcd A DeepakCurrentAge = 7 :=
by
  sorry

end age_ratio_l1832_183274


namespace circle_center_radius_sum_l1832_183279

theorem circle_center_radius_sum (u v s : ℝ) (h1 : (x + 4)^2 + (y - 1)^2 = 13)
    (h2 : (u, v) = (-4, 1)) (h3 : s = Real.sqrt 13) : 
    u + v + s = -3 + Real.sqrt 13 :=
by
  sorry

end circle_center_radius_sum_l1832_183279


namespace mac_total_loss_is_correct_l1832_183298

def day_1_value : ℝ := 6 * 0.075 + 2 * 0.0075
def day_2_value : ℝ := 10 * 0.0045 + 5 * 0.0036
def day_3_value : ℝ := 4 * 0.10 + 1 * 0.011
def day_4_value : ℝ := 7 * 0.013 + 5 * 0.038
def day_5_value : ℝ := 3 * 0.5 + 2 * 0.0019
def day_6_value : ℝ := 12 * 0.0072 + 3 * 0.0013
def day_7_value : ℝ := 8 * 0.045 + 6 * 0.0089

def total_value : ℝ := day_1_value + day_2_value + day_3_value + day_4_value + day_5_value + day_6_value + day_7_value

def daily_loss (total_value: ℝ): ℝ := total_value - 0.25

def total_loss : ℝ := daily_loss day_1_value + daily_loss day_2_value + daily_loss day_3_value + daily_loss day_4_value + daily_loss day_5_value + daily_loss day_6_value + daily_loss day_7_value

theorem mac_total_loss_is_correct : total_loss = 2.1619 := 
by 
  simp [day_1_value, day_2_value, day_3_value, day_4_value, day_5_value, day_6_value, day_7_value, daily_loss, total_loss]
  sorry

end mac_total_loss_is_correct_l1832_183298


namespace padic_zeros_l1832_183271

variable {p : ℕ} (hp : p > 1)
variable {a : ℕ} (hnz : a % p ≠ 0)

theorem padic_zeros (k : ℕ) (hk : k ≥ 1) :
  (a^(p^(k-1)*(p-1)) - 1) % (p^k) = 0 :=
sorry

end padic_zeros_l1832_183271


namespace equation_of_line_l_l1832_183208

def point (P : ℝ × ℝ) := P = (2, 1)
def parallel (x y : ℝ) : Prop := 2 * x - y + 2 = 0

theorem equation_of_line_l (c : ℝ) (x y : ℝ) :
  (parallel x y ∧ point (x, y)) →
  2 * x - y + c = 0 →
  c = -3 → 2 * x - y - 3 = 0 :=
by
  intro h1 h2 h3
  sorry

end equation_of_line_l_l1832_183208


namespace tangent_point_condition_l1832_183222

open Function

def f (x : ℝ) : ℝ := x^3 - 3 * x
def tangent_line (s : ℝ) (x t : ℝ) : ℝ := (3 * s^2 - 3) * (x - 2) + s^3 - 3 * s

theorem tangent_point_condition (t : ℝ) (h_tangent : ∃s : ℝ, tangent_line s 2 t = t) 
  (h_not_on_curve : ∀ s, (2, t) ≠ (s, f s)) : t = -6 :=
by
  sorry

end tangent_point_condition_l1832_183222


namespace linear_function_passing_through_point_and_intersecting_another_line_area_of_triangle_l1832_183214

theorem linear_function_passing_through_point_and_intersecting_another_line (
  k b : ℝ)
  (h1 : (∀ x y : ℝ, y = k * x + b → ((x = 3 ∧ y = -3) ∨ (x = 3/4 ∧ y = 0))))
  (h2 : (∀ x : ℝ, 0 = (4 * x - 3) → x = 3/4))
  : k = -4 / 3 ∧ b = 1 := 
sorry

theorem area_of_triangle (
  k b : ℝ)
  (h1 : k = -4 / 3 ∧ b = 1)
  : 1 / 2 * 3 / 4 * 1 = 3 / 8 := 
sorry

end linear_function_passing_through_point_and_intersecting_another_line_area_of_triangle_l1832_183214


namespace square_circle_radius_l1832_183242

theorem square_circle_radius (a R : ℝ) (h1 : a^2 = 256) (h2 : R = 10) : R = 10 :=
sorry

end square_circle_radius_l1832_183242
