import Mathlib

namespace contrapositive_a_eq_b_imp_a_sq_eq_b_sq_l733_73371

theorem contrapositive_a_eq_b_imp_a_sq_eq_b_sq (a b : ℝ) :
  (a = b → a^2 = b^2) ↔ (a^2 ≠ b^2 → a ≠ b) :=
by
  sorry

end contrapositive_a_eq_b_imp_a_sq_eq_b_sq_l733_73371


namespace polar_eq_circle_l733_73322

-- Definition of the problem condition in polar coordinates
def polar_eq (ρ : ℝ) : Prop := ρ = 1

-- Definition of the assertion we want to prove: that it represents a circle
def represents_circle (ρ : ℝ) (θ : ℝ) : Prop := (ρ = 1) → ∃ (x y : ℝ), (ρ = 1) ∧ (x^2 + y^2 = 1)

theorem polar_eq_circle : ∀ (ρ θ : ℝ), polar_eq ρ → represents_circle ρ θ :=
by
  intros ρ θ hρ hs
  sorry

end polar_eq_circle_l733_73322


namespace max_value_y_interval_l733_73307

noncomputable def y (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem max_value_y_interval : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → y x ≤ 2) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y x = 2) 
:=
by
  sorry

end max_value_y_interval_l733_73307


namespace total_cost_l733_73319

-- Define the conditions
def dozen := 12
def cost_of_dozen_cupcakes := 10
def cost_of_dozen_cookies := 8
def cost_of_dozen_brownies := 12

def num_dozen_cupcakes := 4
def num_dozen_cookies := 3
def num_dozen_brownies := 2

-- Define the total cost for each type of treat
def total_cost_cupcakes := num_dozen_cupcakes * cost_of_dozen_cupcakes
def total_cost_cookies := num_dozen_cookies * cost_of_dozen_cookies
def total_cost_brownies := num_dozen_brownies * cost_of_dozen_brownies

-- The theorem to prove the total cost
theorem total_cost : total_cost_cupcakes + total_cost_cookies + total_cost_brownies = 88 := by
  -- Here would go the proof, but it's omitted as per the instructions
  sorry

end total_cost_l733_73319


namespace price_of_necklace_l733_73344

-- Define the necessary conditions.
def num_charms_per_necklace : ℕ := 10
def cost_per_charm : ℕ := 15
def num_necklaces_sold : ℕ := 30
def total_profit : ℕ := 1500

-- Calculation of selling price per necklace
def cost_per_necklace := num_charms_per_necklace * cost_per_charm
def total_cost := cost_per_necklace * num_necklaces_sold
def total_revenue := total_cost + total_profit
def selling_price_per_necklace := total_revenue / num_necklaces_sold

-- Statement of the problem in Lean 4
theorem price_of_necklace : selling_price_per_necklace = 200 := by
  sorry

end price_of_necklace_l733_73344


namespace circle_symmetry_l733_73324

theorem circle_symmetry (a : ℝ) : 
  (∀ x y : ℝ, (x^2 + y^2 - a*x + 2*y + 1 = 0 ↔ x^2 + y^2 = 1) ↔ a = 2) :=
sorry

end circle_symmetry_l733_73324


namespace max_mn_value_l733_73390

theorem max_mn_value (m n : ℝ) (hm : m > 0) (hn : n > 0)
  (hA1 : ∀ k : ℝ, k * (-2) - (-1) + 2 * k - 1 = 0)
  (hA2 : m * (-2) + n * (-1) + 2 = 0) :
  mn ≤ 1/2 := sorry

end max_mn_value_l733_73390


namespace compound_interest_calculation_l733_73339

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  let A := P * ((1 + r / (n : ℝ)) ^ (n * t))
  A - P

theorem compound_interest_calculation :
  compoundInterest 500 0.05 1 5 = 138.14 := by
  sorry

end compound_interest_calculation_l733_73339


namespace mutually_exclusive_event_3_l733_73345

def is_odd (n : ℕ) := n % 2 = 1
def is_even (n : ℕ) := n % 2 = 0

def event_1 (a b : ℕ) := 
(is_odd a ∧ is_even b) ∨ (is_even a ∧ is_odd b)

def event_2 (a b : ℕ) := 
is_odd a ∧ is_odd b

def event_3 (a b : ℕ) := 
is_odd a ∧ is_even a ∧ is_odd b ∧ is_even b

def event_4 (a b : ℕ) :=
(is_odd a ∧ is_even b) ∨ (is_even a ∧ is_odd b)

theorem mutually_exclusive_event_3 :
  ∀ a b : ℕ, event_3 a b → ¬ event_1 a b ∧ ¬ event_2 a b ∧ ¬ event_4 a b := by
sorry

end mutually_exclusive_event_3_l733_73345


namespace charlotte_overall_score_l733_73363

theorem charlotte_overall_score :
  (0.60 * 15 + 0.75 * 20 + 0.85 * 25).round / 60 = 0.75 :=
by
  sorry

end charlotte_overall_score_l733_73363


namespace minimum_value_fraction_l733_73368

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem minimum_value_fraction (a : ℕ → ℝ) (m n : ℕ) (q : ℝ) (h_geometric : geometric_sequence a q)
  (h_positive : ∀ k : ℕ, 0 < a k)
  (h_condition1 : a 7 = a 6 + 2 * a 5)
  (h_condition2 : ∃ r, r ^ 2 = a m * a n ∧ r = 2 * a 1) :
  (1 / m + 9 / n) ≥ 4 :=
  sorry

end minimum_value_fraction_l733_73368


namespace sugar_solution_sweeter_l733_73360

theorem sugar_solution_sweeter (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : 
    (b + m) / (a + m) > b / a :=
sorry

end sugar_solution_sweeter_l733_73360


namespace no_solutions_xyz_l733_73302

theorem no_solutions_xyz : ∀ (x y z : ℝ), x + y = 3 → xy - z^2 = 2 → false := by
  intros x y z h1 h2
  sorry

end no_solutions_xyz_l733_73302


namespace positive_n_for_one_solution_l733_73306

theorem positive_n_for_one_solution :
  ∀ (n : ℝ), (4 * (0 : ℝ)) ^ 2 + n * (0) + 16 = 0 → (n^2 - 256 = 0) → n = 16 :=
by
  intro n
  intro h
  intro discriminant_eq_zero
  sorry

end positive_n_for_one_solution_l733_73306


namespace blueberry_jelly_amount_l733_73374

theorem blueberry_jelly_amount (total_jelly : ℕ) (strawberry_jelly : ℕ) 
  (h_total : total_jelly = 6310) 
  (h_strawberry : strawberry_jelly = 1792) 
  : total_jelly - strawberry_jelly = 4518 := 
by 
  sorry

end blueberry_jelly_amount_l733_73374


namespace determine_a_l733_73313

-- Define the sets A and B
def A : Set ℕ := {1, 3}
def B (a : ℕ) : Set ℕ := {1, 2, a}

-- The proof statement
theorem determine_a (a : ℕ) (h : A ⊆ B a) : a = 3 :=
by 
  sorry

end determine_a_l733_73313


namespace total_votes_l733_73315

theorem total_votes (V : ℝ) (h : 0.60 * V - 0.40 * V = 1200) : V = 6000 :=
sorry

end total_votes_l733_73315


namespace age_difference_l733_73310

theorem age_difference
  (A B : ℕ)
  (hB : B = 48)
  (h_condition : A + 10 = 2 * (B - 10)) :
  A - B = 18 :=
by
  sorry

end age_difference_l733_73310


namespace hurricane_damage_in_GBP_l733_73357

def damage_in_AUD : ℤ := 45000000
def conversion_rate : ℚ := 1 / 2 -- 1 AUD = 1/2 GBP

theorem hurricane_damage_in_GBP : 
  (damage_in_AUD : ℚ) * conversion_rate = 22500000 := 
by
  sorry

end hurricane_damage_in_GBP_l733_73357


namespace line_has_equal_intercepts_find_a_l733_73327

theorem line_has_equal_intercepts (a : ℝ) :
  (∃ l : ℝ, (l = 0 → ax + y - 2 - a = 0) ∧ (l = 1 → (a = 1 ∨ a = -2))) := sorry

-- formalizing the problem
theorem find_a (a : ℝ) (h_eq_intercepts : ∀ x y : ℝ, (a * x + y - 2 - a = 0 ↔ (x = 2 + a ∧ y = -2 - a))) :
  a = 1 ∨ a = -2 := sorry

end line_has_equal_intercepts_find_a_l733_73327


namespace second_quadratic_roots_complex_iff_first_roots_real_distinct_l733_73347

theorem second_quadratic_roots_complex_iff_first_roots_real_distinct (q : ℝ) :
  q < 1 → (∀ x : ℂ, (3 - q) * x^2 + 2 * (1 + q) * x + (q^2 - q + 2) ≠ 0) :=
by
  -- Placeholder for the proof
  sorry

end second_quadratic_roots_complex_iff_first_roots_real_distinct_l733_73347


namespace nonzero_roots_ratio_l733_73329

theorem nonzero_roots_ratio (m : ℝ) (h : m ≠ 0) :
  (∃ (r s : ℝ), r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ r + s = 4 ∧ r * s = m) → m = 3 :=
by 
  intro h_exists
  obtain ⟨r, s, hr_ne_zero, hs_ne_zero, h_ratio, h_sum, h_prod⟩ := h_exists
  sorry

end nonzero_roots_ratio_l733_73329


namespace osborn_friday_time_l733_73381

-- Conditions
def time_monday : ℕ := 2
def time_tuesday : ℕ := 4
def time_wednesday : ℕ := 3
def time_thursday : ℕ := 4
def old_average_time_per_day : ℕ := 3
def school_days_per_week : ℕ := 5

-- Total time needed to match old average
def total_time_needed : ℕ := old_average_time_per_day * school_days_per_week

-- Total time spent from Monday to Thursday
def time_spent_mon_to_thu : ℕ := time_monday + time_tuesday + time_wednesday + time_thursday

-- Goal: Find time on Friday
def time_friday : ℕ := total_time_needed - time_spent_mon_to_thu

theorem osborn_friday_time : time_friday = 2 :=
by
  sorry

end osborn_friday_time_l733_73381


namespace julian_initial_owing_l733_73336

theorem julian_initial_owing (jenny_owing_initial: ℕ) (borrow: ℕ) (total_owing: ℕ):
    borrow = 8 → total_owing = 28 → jenny_owing_initial + borrow = total_owing → jenny_owing_initial = 20 :=
by intros;
   exact sorry

end julian_initial_owing_l733_73336


namespace number_of_cows_l733_73373

-- Define the total number of legs and number of legs per cow
def total_legs : ℕ := 460
def legs_per_cow : ℕ := 4

-- Mathematical proof problem as a Lean 4 statement
theorem number_of_cows : total_legs / legs_per_cow = 115 := by
  -- This is the proof statement place. We use 'sorry' as a placeholder for the actual proof.
  sorry

end number_of_cows_l733_73373


namespace point_three_units_away_from_A_is_negative_seven_or_negative_one_l733_73383

-- Defining the point A on the number line
def A : ℤ := -4

-- Definition of the condition where a point is 3 units away from A
def three_units_away (x : ℤ) : Prop := (x = A - 3) ∨ (x = A + 3)

-- The statement to be proved
theorem point_three_units_away_from_A_is_negative_seven_or_negative_one (x : ℤ) :
  three_units_away x → (x = -7 ∨ x = -1) :=
sorry

end point_three_units_away_from_A_is_negative_seven_or_negative_one_l733_73383


namespace find_ab_range_m_l733_73355

-- Part 1
theorem find_ab (a b: ℝ) (h1 : 3 - 6 * a + b = 0) (h2 : -1 + 3 * a - b + a^2 = 0) :
  a = 2 ∧ b = 9 := 
sorry

-- Part 2
theorem range_m (m: ℝ) (h: ∀ x ∈ (Set.Icc (-2) 1), x^3 + 3 * 2 * x^2 + 9 * x + 4 - m ≤ 0) :
  20 ≤ m :=
sorry

end find_ab_range_m_l733_73355


namespace minimum_vehicles_l733_73311

theorem minimum_vehicles (students adults : ℕ) (van_capacity minibus_capacity : ℕ)
    (severe_allergies_students : ℕ) (vehicle_requires_adult : Prop)
    (h_students : students = 24) (h_adults : adults = 3)
    (h_van_capacity : van_capacity = 8) (h_minibus_capacity : minibus_capacity = 14)
    (h_severe_allergies_students : severe_allergies_students = 2)
    (h_vehicle_requires_adult : vehicle_requires_adult)
    : ∃ (min_vehicles : ℕ), min_vehicles = 5 :=
by
  sorry

end minimum_vehicles_l733_73311


namespace abc_value_l733_73396

-- Define constants for the problem
variable (a b c k : ℕ)

-- Assumptions based on the given conditions
axiom h1 : a - b = 3
axiom h2 : a^2 + b^2 = 29
axiom h3 : a^2 + b^2 + c^2 = k
axiom pos_k : k > 0
axiom pos_a : a > 0

-- The goal is to prove that abc = 10
theorem abc_value : a * b * c = 10 :=
by
  sorry

end abc_value_l733_73396


namespace triangle_area_l733_73333

noncomputable def area_triangle (b c angle_C : ℝ) : ℝ :=
  (1 / 2) * b * c * Real.sin angle_C

theorem triangle_area :
  let b := 1
  let c := Real.sqrt 3
  let angle_C := 2 * Real.pi / 3
  area_triangle b c (Real.sin angle_C) = Real.sqrt 3 / 4 :=
by
  sorry

end triangle_area_l733_73333


namespace obtain_half_not_obtain_one_l733_73399

theorem obtain_half (x : ℕ) : (10 + x) / (97 + x) = 1 / 2 ↔ x = 77 := 
by
  sorry

theorem not_obtain_one (x k : ℕ) : ¬ ((10 + x) / (97 + x) = 1 ∨ (10 * k) / (97 * k) = 1) := 
by
  sorry

end obtain_half_not_obtain_one_l733_73399


namespace customer_savings_l733_73343

variables (P : ℝ) (reducedPrice negotiatedPrice savings : ℝ)

-- Conditions:
def initialReduction : reducedPrice = 0.95 * P := by sorry
def finalNegotiation : negotiatedPrice = 0.90 * reducedPrice := by sorry
def savingsCalculation : savings = P - negotiatedPrice := by sorry

-- Proof problem:
theorem customer_savings : savings = 0.145 * P :=
by {
  sorry
}

end customer_savings_l733_73343


namespace john_running_time_l733_73377

theorem john_running_time
  (x : ℚ)
  (h1 : 15 * x + 10 * (9 - x) = 100)
  (h2 : 0 ≤ x)
  (h3 : x ≤ 9) :
  x = 2 := by
  sorry

end john_running_time_l733_73377


namespace there_exists_triangle_part_two_l733_73366

noncomputable def exists_triangle (a b c : ℝ) : Prop :=
a > 0 ∧
4 * a - 8 * b + 4 * c ≥ 0 ∧
9 * a - 12 * b + 4 * c ≥ 0 ∧
2 * a ≤ 2 * b ∧
2 * b ≤ 3 * a ∧
b^2 ≥ a*c

theorem there_exists_triangle (a b c : ℝ) (h1 : a > 0)
  (h2 : 4 * a - 8 * b + 4 * c ≥ 0)
  (h3 : 9 * a - 12 * b + 4 * c ≥ 0)
  (h4 : 2 * a ≤ 2 * b)
  (h5 : 2 * b ≤ 3 * a)
  (h6 : b^2 ≥ a * c) : 
 a ≤ b ∧ b ≤ c ∧ a + b > c :=
sorry

theorem part_two (a b c : ℝ) (h1 : a > 0) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c < a + b) :
  ∃ h : a > 0, (a / (a + c) + b / (b + a) > c / (b + c)) :=
sorry

end there_exists_triangle_part_two_l733_73366


namespace smallest_number_l733_73394

theorem smallest_number (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : (a + b + c) = 90) (h4 : b = 28) (h5 : b = c - 6) : a = 28 :=
by 
  sorry

end smallest_number_l733_73394


namespace sandy_total_earnings_l733_73367

-- Define the conditions
def hourly_wage : ℕ := 15
def hours_friday : ℕ := 10
def hours_saturday : ℕ := 6
def hours_sunday : ℕ := 14

-- Define the total hours worked and total earnings
def total_hours := hours_friday + hours_saturday + hours_sunday
def total_earnings := total_hours * hourly_wage

-- State the theorem
theorem sandy_total_earnings : total_earnings = 450 := by
  sorry

end sandy_total_earnings_l733_73367


namespace area_between_curves_eq_nine_l733_73305

def f (x : ℝ) := 2 * x - x^2 + 3
def g (x : ℝ) := x^2 - 4 * x + 3

theorem area_between_curves_eq_nine :
  ∫ x in (0 : ℝ)..(3 : ℝ), (f x - g x) = 9 := by
  sorry

end area_between_curves_eq_nine_l733_73305


namespace sarah_shirts_l733_73361

theorem sarah_shirts (loads : ℕ) (pieces_per_load : ℕ) (sweaters : ℕ) 
  (total_pieces : ℕ) (shirts : ℕ) : 
  loads = 9 → pieces_per_load = 5 → sweaters = 2 →
  total_pieces = loads * pieces_per_load → shirts = total_pieces - sweaters → 
  shirts = 43 :=
by
  intros h_loads h_pieces_per_load h_sweaters h_total_pieces h_shirts
  sorry

end sarah_shirts_l733_73361


namespace magician_act_reappearance_l733_73354

-- Defining the conditions as given in the problem
def total_performances : ℕ := 100

def no_one_reappears (perf : ℕ) : ℕ := perf / 10
def two_reappear (perf : ℕ) : ℕ := perf / 5
def one_reappears (perf : ℕ) : ℕ := perf - no_one_reappears perf - two_reappear perf
def total_reappeared (perf : ℕ) : ℕ := one_reappears perf + 2 * two_reappear perf

-- The statement to be proved
theorem magician_act_reappearance : total_reappeared total_performances = 110 := by
  sorry

end magician_act_reappearance_l733_73354


namespace susan_age_in_5_years_l733_73387

-- Definitions of the given conditions
def james_age_in_15_years : ℕ := 37
def years_until_james_is_37 : ℕ := 15
def years_ago_james_twice_janet : ℕ := 8
def susan_born_when_janet_turned : ℕ := 3
def years_to_future_susan_age : ℕ := 5

-- Calculate the current age of people involved
def james_current_age : ℕ := james_age_in_15_years - years_until_james_is_37
def james_age_8_years_ago : ℕ := james_current_age - years_ago_james_twice_janet
def janet_age_8_years_ago : ℕ := james_age_8_years_ago / 2
def janet_current_age : ℕ := janet_age_8_years_ago + years_ago_james_twice_janet
def susan_current_age : ℕ := janet_current_age - susan_born_when_janet_turned

-- Prove that Susan will be 17 years old in 5 years
theorem susan_age_in_5_years (james_age_future : james_age_in_15_years = 37)
  (years_until_james_37 : years_until_james_is_37 = 15)
  (years_ago_twice_janet : years_ago_james_twice_janet = 8)
  (susan_born_janet : susan_born_when_janet_turned = 3)
  (years_future : years_to_future_susan_age = 5) :
  susan_current_age + years_to_future_susan_age = 17 := by
  -- The proof is omitted
  sorry

end susan_age_in_5_years_l733_73387


namespace cost_of_large_poster_is_correct_l733_73346

/-- Problem conditions -/
def posters_per_day : ℕ := 5
def large_posters_per_day : ℕ := 2
def large_poster_sale_price : ℝ := 10
def small_posters_per_day : ℕ := 3
def small_poster_sale_price : ℝ := 6
def small_poster_cost : ℝ := 3
def weekly_profit : ℝ := 95

/-- The cost to make a large poster -/
noncomputable def large_poster_cost : ℝ := 5

/-- Prove that the cost to make a large poster is $5 given the conditions -/
theorem cost_of_large_poster_is_correct :
    large_poster_cost = 5 :=
by
  -- (Condition translation into Lean)
  let daily_profit := weekly_profit / 5
  let daily_revenue := (large_posters_per_day * large_poster_sale_price) + (small_posters_per_day * small_poster_sale_price)
  let daily_cost_small_posters := small_posters_per_day * small_poster_cost
  
  -- Express the daily profit in terms of costs, including unknown large_poster_cost
  have calc_profit : daily_profit = daily_revenue - daily_cost_small_posters - (large_posters_per_day * (large_poster_cost)) :=
    sorry
  
  -- Setting the equation to solve for large_poster_cost
  have eqn : daily_profit = 19 := by
    sorry

  -- Solve for large_poster_cost
  have solve_large_poster_cost : 19 = daily_revenue - daily_cost_small_posters - (large_posters_per_day * 5) :=
    by sorry
  
  sorry

end cost_of_large_poster_is_correct_l733_73346


namespace problem_1_problem_2_problem_3_l733_73356

def pair_otimes (a b c d : ℚ) : ℚ := b * c - a * d

-- Problem (1)
theorem problem_1 : pair_otimes 5 3 (-2) 1 = -11 := 
by 
  unfold pair_otimes 
  sorry

-- Problem (2)
theorem problem_2 (x : ℚ) (h : pair_otimes 2 (3 * x - 1) 6 (x + 2) = 22) : x = 2 := 
by 
  unfold pair_otimes at h
  sorry

-- Problem (3)
theorem problem_3 (x k : ℤ) (h : pair_otimes 4 (k - 2) x (2 * x - 1) = 6) : 
  k = 8 ∨ k = 9 ∨ k = 11 ∨ k = 12 := 
by 
  unfold pair_otimes at h
  sorry

end problem_1_problem_2_problem_3_l733_73356


namespace excluded_numbers_range_l733_73382

theorem excluded_numbers_range (S S' E : ℕ) (h1 : S = 31 * 10) (h2 : S' = 28 * 8) (h3 : E = S - S') (h4 : E > 70) :
  ∀ (x y : ℕ), x + y = E → 1 ≤ x ∧ x ≤ 85 ∧ 1 ≤ y ∧ y ≤ 85 := by
  sorry

end excluded_numbers_range_l733_73382


namespace remainder_2_pow_224_plus_104_l733_73389

theorem remainder_2_pow_224_plus_104 (x : ℕ) (h1 : x = 2 ^ 56) : 
  (2 ^ 224 + 104) % (2 ^ 112 + 2 ^ 56 + 1) = 103 := 
by
  sorry

end remainder_2_pow_224_plus_104_l733_73389


namespace probability_of_sum_8_9_10_l733_73326

/-- The list of face values for the first die. -/
def first_die : List ℕ := [1, 1, 3, 3, 5, 6]

/-- The list of face values for the second die. -/
def second_die : List ℕ := [1, 2, 4, 5, 7, 9]

/-- The condition to verify if the sum is 8, 9, or 10. -/
def valid_sum (s : ℕ) : Bool := s = 8 ∨ s = 9 ∨ s = 10

/-- Calculate probability of the sum being 8, 9, or 10 for the two dice. -/
def calculate_probability : ℚ :=
  let total_rolls := first_die.length * second_die.length
  let valid_rolls := 
    first_die.foldl (fun acc d1 =>
      acc + second_die.foldl (fun acc' d2 => 
        if valid_sum (d1 + d2) then acc' + 1 else acc') 0) 0
  valid_rolls / total_rolls

/-- The required probability is 7/18. -/
theorem probability_of_sum_8_9_10 : calculate_probability = 7 / 18 := 
  sorry

end probability_of_sum_8_9_10_l733_73326


namespace evaluate_expression_l733_73349

theorem evaluate_expression (a b c : ℚ) 
  (h1 : c = b - 11) 
  (h2 : b = a + 3) 
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7) = 10 / 7 := 
sorry

end evaluate_expression_l733_73349


namespace problem_conditions_l733_73338

noncomputable def f (x : ℝ) : ℝ := (2 * x - x^2) * Real.exp x

theorem problem_conditions :
  (∀ x, f x > 0 ↔ 0 < x ∧ x < 2) ∧
  (∃ x_max, x_max = Real.sqrt 2 ∧ (∀ y, f y ≤ f x_max)) ∧
  ¬(∃ x_min, ∀ y, f x_min ≤ f y) :=
by sorry

end problem_conditions_l733_73338


namespace jess_height_l733_73370

variable (Jana_height Kelly_height Jess_height : ℕ)

-- Conditions
axiom Jana_height_eq : Jana_height = 74
axiom Jana_taller_than_Kelly : Jana_height = Kelly_height + 5
axiom Kelly_shorter_than_Jess : Kelly_height = Jess_height - 3

-- Prove Jess's height
theorem jess_height : Jess_height = 72 := by
  -- Proof goes here
  sorry

end jess_height_l733_73370


namespace b_finishes_remaining_work_correct_time_for_b_l733_73342

theorem b_finishes_remaining_work (a_days : ℝ) (b_days : ℝ) (work_together_days : ℝ) (remaining_work_after : ℝ) : ℝ :=
  let a_work_rate := 1 / a_days
  let b_work_rate := 1 / b_days
  let combined_work_per_day := a_work_rate + b_work_rate
  let work_done_together := combined_work_per_day * work_together_days
  let remaining_work := 1 - work_done_together
  let b_completion_time := remaining_work / b_work_rate
  b_completion_time

theorem correct_time_for_b : b_finishes_remaining_work 2 6 1 (1 - 2/3) = 2 := 
by sorry

end b_finishes_remaining_work_correct_time_for_b_l733_73342


namespace spherical_coordinates_equivalence_l733_73304

theorem spherical_coordinates_equivalence
  (ρ θ φ : ℝ)
  (h_ρ : ρ > 0)
  (h_θ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (h_φ : φ = 2 * Real.pi - (7 * Real.pi / 4)) :
  (ρ, θ, φ) = (4, 3 * Real.pi / 4, Real.pi / 4) :=
by 
  sorry

end spherical_coordinates_equivalence_l733_73304


namespace ferry_round_trip_time_increases_l733_73337

variable {S V a b : ℝ}

theorem ferry_round_trip_time_increases (h1 : V > 0) (h2 : a < b) (h3 : V > a) (h4 : V > b) :
  (S / (V + b) + S / (V - b)) > (S / (V + a) + S / (V - a)) :=
by sorry

end ferry_round_trip_time_increases_l733_73337


namespace cyclist_is_jean_l733_73395

theorem cyclist_is_jean (x x' y y' : ℝ) (hx : x' = 4 * x) (hy : y = 4 * y') : x < y :=
by
  sorry

end cyclist_is_jean_l733_73395


namespace sum_b_a1_a2_a3_a4_eq_60_l733_73362

def a_n (n : ℕ) : ℕ := n + 2
def b_n (n : ℕ) : ℕ := 2^(n-1)

theorem sum_b_a1_a2_a3_a4_eq_60 :
  b_n (a_n 1) + b_n (a_n 2) + b_n (a_n 3) + b_n (a_n 4) = 60 :=
by
  sorry

end sum_b_a1_a2_a3_a4_eq_60_l733_73362


namespace x_of_x35x_div_by_18_l733_73303

theorem x_of_x35x_div_by_18 (x : ℕ) (h₁ : 18 = 2 * 9) (h₂ : (2 * x + 8) % 9 = 0) (h₃ : ∃ k : ℕ, x = 2 * k) : x = 8 :=
sorry

end x_of_x35x_div_by_18_l733_73303


namespace value_range_of_a_l733_73331

variable (A B : Set ℝ)

noncomputable def A_def : Set ℝ := { x | 2 * x^2 - 3 * x + 1 ≤ 0 }
noncomputable def B_def (a : ℝ) : Set ℝ := { x | x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0 }

theorem value_range_of_a (a : ℝ) (hA : A = A_def) (hB : B = B_def a) :
    (Bᶜ ∩ A = ∅) → (0 ≤ a ∧ a ≤ 0.5) := 
sorry

end value_range_of_a_l733_73331


namespace smallest_positive_integer_rel_prime_180_l733_73365

theorem smallest_positive_integer_rel_prime_180 : 
  ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → y ≥ 7 := 
by 
  sorry

end smallest_positive_integer_rel_prime_180_l733_73365


namespace us2_eq_3958_div_125_l733_73393

-- Definitions based on conditions
def t (x : ℚ) : ℚ := 5 * x - 12
def s (t_x : ℚ) : ℚ := (2 : ℚ) ^ 2 + 3 * 2 - 2
def u (s_t_x : ℚ) : ℚ := (14 : ℚ) / 5 ^ 3 + 2 * (14 / 5) ^ 2 - 14 / 5 + 4

-- Prove that u(s(2)) = 3958 / 125
theorem us2_eq_3958_div_125 : u (s (2)) = 3958 / 125 := by
  sorry

end us2_eq_3958_div_125_l733_73393


namespace total_time_to_school_and_back_l733_73312

-- Definition of the conditions
def speed_to_school : ℝ := 3 -- in km/hr
def speed_back_home : ℝ := 2 -- in km/hr
def distance : ℝ := 6 -- in km

-- Proof statement
theorem total_time_to_school_and_back : 
  (distance / speed_to_school) + (distance / speed_back_home) = 5 := 
by
  sorry

end total_time_to_school_and_back_l733_73312


namespace compare_f_values_l733_73392

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.cos x

theorem compare_f_values :
  f 0.6 > f (-0.5) ∧ f (-0.5) > f 0 := by
  sorry

end compare_f_values_l733_73392


namespace cody_initial_marbles_l733_73386

theorem cody_initial_marbles (M : ℕ) (h1 : (2 / 3 : ℝ) * M - (1 / 4 : ℝ) * ((2 / 3 : ℝ) * M) - (2 * (1 / 4 : ℝ) * ((2 / 3 : ℝ) * M)) = 7) : M = 42 := 
  sorry

end cody_initial_marbles_l733_73386


namespace find_x_from_conditions_l733_73330

theorem find_x_from_conditions (a b x y s : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) :
  s = (4 * a)^(4 * b) ∧ s = a^b * y^b ∧ y = 4 * x → x = 64 * a^3 :=
by
  sorry

end find_x_from_conditions_l733_73330


namespace jon_monthly_earnings_l733_73397

def earnings_per_person : ℝ := 0.10
def visits_per_hour : ℕ := 50
def hours_per_day : ℕ := 24
def days_per_month : ℕ := 30

theorem jon_monthly_earnings : 
  (earnings_per_person * visits_per_hour * hours_per_day * days_per_month) = 3600 :=
by
  sorry

end jon_monthly_earnings_l733_73397


namespace robin_photo_count_l733_73351

theorem robin_photo_count (photos_per_page : ℕ) (full_pages : ℕ) 
  (h1 : photos_per_page = 6) (h2 : full_pages = 122) :
  photos_per_page * full_pages = 732 :=
by
  sorry

end robin_photo_count_l733_73351


namespace area_of_45_45_90_triangle_l733_73325

noncomputable def leg_length (hypotenuse : ℝ) : ℝ :=
  hypotenuse / Real.sqrt 2

theorem area_of_45_45_90_triangle (hypotenuse : ℝ) (h : hypotenuse = 13) : 
  (1 / 2) * (leg_length hypotenuse) * (leg_length hypotenuse) = 84.5 :=
by
  sorry

end area_of_45_45_90_triangle_l733_73325


namespace at_least_one_bigger_than_44_9_l733_73332

noncomputable def x : ℕ → ℝ := sorry
noncomputable def y : ℕ → ℝ := sorry

axiom x_positive (n : ℕ) : 0 < x n
axiom y_positive (n : ℕ) : 0 < y n
axiom recurrence_x (n : ℕ) : x (n + 1) = x n + 1 / (2 * y n)
axiom recurrence_y (n : ℕ) : y (n + 1) = y n + 1 / (2 * x n)

theorem at_least_one_bigger_than_44_9 : x 2018 > 44.9 ∨ y 2018 > 44.9 :=
sorry

end at_least_one_bigger_than_44_9_l733_73332


namespace binary_to_decimal_10101_l733_73321

theorem binary_to_decimal_10101 : (1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 21 :=
by
  sorry

end binary_to_decimal_10101_l733_73321


namespace quadrilateral_side_difference_l733_73335

variable (a b c d : ℝ)

theorem quadrilateral_side_difference :
  a + b + c + d = 120 →
  a + c = 50 →
  (a^2 + c^2 = 1600) →
  (b + d = 70 ∧ b * d = 450) →
  |b - d| = 2 * Real.sqrt 775 :=
by
  intros ha hb hc hd
  sorry

end quadrilateral_side_difference_l733_73335


namespace find_dividend_l733_73323

def quotient : ℝ := -427.86
def divisor : ℝ := 52.7
def remainder : ℝ := -14.5
def dividend : ℝ := (quotient * divisor) + remainder

theorem find_dividend : dividend = -22571.122 := by
  sorry

end find_dividend_l733_73323


namespace meters_of_cloth_l733_73352

variable (total_cost cost_per_meter : ℝ)
variable (h1 : total_cost = 434.75)
variable (h2 : cost_per_meter = 47)

theorem meters_of_cloth : 
  total_cost / cost_per_meter = 9.25 := 
by
  sorry

end meters_of_cloth_l733_73352


namespace speed_of_stream_l733_73353

theorem speed_of_stream (v : ℝ) (d : ℝ) :
  (∀ d : ℝ, d > 0 → (1 / (6 - v) = 2 * (1 / (6 + v)))) → v = 2 := by
  sorry

end speed_of_stream_l733_73353


namespace ratio_of_doctors_to_nurses_l733_73358

theorem ratio_of_doctors_to_nurses (total_staff doctors nurses : ℕ) (h1 : total_staff = 456) (h2 : nurses = 264) (h3 : doctors + nurses = total_staff) :
  doctors = 192 ∧ (doctors : ℚ) / nurses = 8 / 11 :=
by
  sorry

end ratio_of_doctors_to_nurses_l733_73358


namespace total_volume_of_four_cubes_is_500_l733_73334

-- Definitions for the problem assumptions
def edge_length := 5
def volume_of_cube (edge_length : ℕ) := edge_length ^ 3
def number_of_boxes := 4

-- Main statement to prove
theorem total_volume_of_four_cubes_is_500 :
  (volume_of_cube edge_length) * number_of_boxes = 500 :=
by
  -- Proof steps will go here
  sorry

end total_volume_of_four_cubes_is_500_l733_73334


namespace count_ordered_triples_l733_73379

def S := Finset.range 20

def succ (a b : ℕ) : Prop := 
  (0 < a - b ∧ a - b ≤ 10) ∨ (b - a > 10)

theorem count_ordered_triples 
  (h : ∃ n : ℕ, (S.card = 20) ∧
                (∀ x y z : ℕ, 
                   x ∈ S → y ∈ S → z ∈ S →
                   (succ x y) → (succ y z) → (succ z x) →
                   n = 1260)) : True := sorry

end count_ordered_triples_l733_73379


namespace line_intersects_x_axis_at_point_l733_73364

theorem line_intersects_x_axis_at_point :
  (∃ x, 5 * 0 - 2 * x = 10) ↔ (x = -5) ∧ (∃ x, 5 * y - 2 * x = 10 ∧ y = 0) :=
by
  sorry

end line_intersects_x_axis_at_point_l733_73364


namespace find_difference_l733_73391

-- Define the initial amounts each person paid.
def Alex_paid : ℕ := 95
def Tom_paid : ℕ := 140
def Dorothy_paid : ℕ := 110
def Sammy_paid : ℕ := 155

-- Define the total spent and the share per person.
def total_spent : ℕ := Alex_paid + Tom_paid + Dorothy_paid + Sammy_paid
def share : ℕ := total_spent / 4

-- Define how much each person needs to pay or should receive.
def Alex_balance : ℤ := share - Alex_paid
def Tom_balance : ℤ := Tom_paid - share
def Dorothy_balance : ℤ := share - Dorothy_paid
def Sammy_balance : ℤ := Sammy_paid - share

-- Define the values of t and d.
def t : ℤ := 0
def d : ℤ := 15

-- The proof goal
theorem find_difference : t - d = -15 := by
  sorry

end find_difference_l733_73391


namespace sqrt_four_ninths_l733_73385

theorem sqrt_four_ninths : 
  (∀ (x : ℚ), x * x = 4 / 9 → (x = 2 / 3 ∨ x = - (2 / 3))) :=
by sorry

end sqrt_four_ninths_l733_73385


namespace find_a_b_c_eq_32_l733_73314

variables {a b c : ℤ}

theorem find_a_b_c_eq_32
  (h1 : ∃ a b : ℤ, x^2 + 19 * x + 88 = (x + a) * (x + b))
  (h2 : ∃ b c : ℤ, x^2 - 21 * x + 108 = (x - b) * (x - c)) :
  a + b + c = 32 :=
sorry

end find_a_b_c_eq_32_l733_73314


namespace total_animal_sightings_l733_73384

def A_Jan := 26
def A_Feb := 3 * A_Jan
def A_Mar := A_Feb / 2

theorem total_animal_sightings : A_Jan + A_Feb + A_Mar = 143 := by
  sorry

end total_animal_sightings_l733_73384


namespace campers_afternoon_l733_73380

theorem campers_afternoon (total_campers morning_campers afternoon_campers : ℕ)
  (h1 : total_campers = 60)
  (h2 : morning_campers = 53)
  (h3 : afternoon_campers = total_campers - morning_campers) :
  afternoon_campers = 7 := by
  sorry

end campers_afternoon_l733_73380


namespace total_pairs_sold_l733_73359

theorem total_pairs_sold
  (H S : ℕ)
  (price_soft : ℕ := 150)
  (price_hard : ℕ := 85)
  (diff_lenses : S = H + 5)
  (total_sales_eq : price_soft * S + price_hard * H = 1455) :
  H + S = 11 := by
sorry

end total_pairs_sold_l733_73359


namespace DianasInitialSpeed_l733_73340

open Nat

theorem DianasInitialSpeed
  (total_distance : ℕ)
  (initial_time : ℕ)
  (tired_speed : ℕ)
  (total_time : ℕ)
  (distance_when_tired : ℕ)
  (initial_distance : ℕ)
  (initial_speed : ℕ)
  (initial_hours : ℕ) :
  total_distance = 10 →
  initial_time = 2 →
  tired_speed = 1 →
  total_time = 6 →
  distance_when_tired = tired_speed * (total_time - initial_time) →
  initial_distance = total_distance - distance_when_tired →
  initial_distance = initial_speed * initial_time →
  initial_speed = 3 := by
  sorry

end DianasInitialSpeed_l733_73340


namespace john_total_cost_l733_73308

-- Definitions based on given conditions
def yearly_cost_first_8_years : ℕ := 10000
def yearly_cost_next_10_years : ℕ := 20000
def university_tuition : ℕ := 250000
def years_first_phase : ℕ := 8
def years_second_phase : ℕ := 10

-- We need to prove the total cost John pays
theorem john_total_cost : 
  (years_first_phase * yearly_cost_first_8_years + years_second_phase * yearly_cost_next_10_years + university_tuition) / 2 = 265000 :=
by sorry

end john_total_cost_l733_73308


namespace min_value_expression_l733_73309

theorem min_value_expression (a b : ℝ) (h : a > b) (h0 : b > 0) :
  ∃ m : ℝ, m = (a^2 + 1 / (a * b) + 1 / (a * (a - b))) ∧ m = 4 :=
sorry

end min_value_expression_l733_73309


namespace speed_of_first_car_l733_73388

theorem speed_of_first_car (v : ℝ) (h1 : 2.5 * v + 2.5 * 45 = 175) : v = 25 :=
by
  sorry

end speed_of_first_car_l733_73388


namespace multiply_fractions_l733_73320

theorem multiply_fractions :
  (2/3) * (4/7) * (9/11) * (5/8) = 15/77 :=
by
  -- It is just a statement, no need for the proof steps here
  sorry

end multiply_fractions_l733_73320


namespace melanie_correct_coins_and_value_l733_73350

def melanie_coins_problem : Prop :=
let dimes_initial := 19
let dimes_dad := 39
let dimes_sister := 15
let dimes_mother := 25
let total_dimes := dimes_initial + dimes_dad + dimes_sister + dimes_mother

let nickels_initial := 12
let nickels_dad := 22
let nickels_sister := 7
let nickels_mother := 10
let nickels_grandmother := 30
let total_nickels := nickels_initial + nickels_dad + nickels_sister + nickels_mother + nickels_grandmother

let quarters_initial := 8
let quarters_dad := 15
let quarters_sister := 12
let quarters_grandmother := 3
let total_quarters := quarters_initial + quarters_dad + quarters_sister + quarters_grandmother

let dimes_value := total_dimes * 0.10
let nickels_value := total_nickels * 0.05
let quarters_value := total_quarters * 0.25
let total_value := dimes_value + nickels_value + quarters_value

total_dimes = 98 ∧ total_nickels = 81 ∧ total_quarters = 38 ∧ total_value = 23.35

theorem melanie_correct_coins_and_value : melanie_coins_problem :=
by sorry

end melanie_correct_coins_and_value_l733_73350


namespace interval_intersection_l733_73375

/--
  This statement asserts that the intersection of the intervals (2/4, 3/4) and (2/5, 3/5)
  results in the interval (1/2, 0.6), which is the solution to the problem.
-/
theorem interval_intersection :
  { x : ℝ | 2 < 4 * x ∧ 4 * x < 3 ∧ 2 < 5 * x ∧ 5 * x < 3 } = { x : ℝ | 0.5 < x ∧ x < 0.6 } :=
by
  sorry

end interval_intersection_l733_73375


namespace find_value_l733_73318

variable (x y z : ℕ)

-- Condition: x / 4 = y / 3 = z / 2
def ratio_condition := x / 4 = y / 3 ∧ y / 3 = z / 2

-- Theorem: Given the ratio condition, prove that (x - y + 3z) / x = 7 / 4.
theorem find_value (h : ratio_condition x y z) : (x - y + 3 * z) / x = 7 / 4 := 
  by sorry

end find_value_l733_73318


namespace coefficient_of_x_in_first_equation_is_one_l733_73341

theorem coefficient_of_x_in_first_equation_is_one
  (x y z : ℝ)
  (h1 : x - 5 * y + 3 * z = 22 / 6)
  (h2 : 4 * x + 8 * y - 11 * z = 7)
  (h3 : 5 * x - 6 * y + 2 * z = 12)
  (h4 : x + y + z = 10) :
  (1 : ℝ) = 1 := 
by 
  sorry

end coefficient_of_x_in_first_equation_is_one_l733_73341


namespace min_value_a_squared_ab_b_squared_l733_73376

theorem min_value_a_squared_ab_b_squared {a b t p : ℝ} (h1 : a + b = t) (h2 : ab = p) :
  a^2 + ab + b^2 ≥ 3 * t^2 / 4 := by
  sorry

end min_value_a_squared_ab_b_squared_l733_73376


namespace sum_of_possible_values_d_l733_73348

theorem sum_of_possible_values_d :
  let range_8 := (512, 4095)
  let digits_in_base_16 := 3
  (∀ n, n ∈ Set.Icc range_8.1 range_8.2 → (Nat.digits 16 n).length = digits_in_base_16)
  → digits_in_base_16 = 3 :=
by
  sorry

end sum_of_possible_values_d_l733_73348


namespace problem_inequality_l733_73328

theorem problem_inequality (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
    (y + z) / (2 * x) + (z + x) / (2 * y) + (x + y) / (2 * z) ≥
    2 * x / (y + z) + 2 * y / (z + x) + 2 * z / (x + y) :=
by
  sorry

end problem_inequality_l733_73328


namespace distance_between_homes_l733_73301

theorem distance_between_homes (Maxwell_distance : ℝ) (Maxwell_speed : ℝ) (Brad_speed : ℝ) (midpoint : ℝ) 
    (h1 : Maxwell_speed = 2) 
    (h2 : Brad_speed = 4) 
    (h3 : Maxwell_distance = 12) 
    (h4 : midpoint = Maxwell_distance * 2 * (Brad_speed / Maxwell_speed) + Maxwell_distance) :
midpoint = 36 :=
by
  sorry

end distance_between_homes_l733_73301


namespace polynomial_divisibility_l733_73372

theorem polynomial_divisibility (m : ℕ) (odd_m : m % 2 = 1) (x y z : ℤ) :
    ∃ k : ℤ, (x + y + z)^m - x^m - y^m - z^m = k * ((x + y + z)^3 - x^3 - y^3 - z^3) := 
by 
  sorry

end polynomial_divisibility_l733_73372


namespace range_of_x_l733_73300

theorem range_of_x (x m : ℝ) (h₁ : 1 ≤ m) (h₂ : m ≤ 3) (h₃ : x + 3 * m + 5 > 0) : x > -14 := 
sorry

end range_of_x_l733_73300


namespace triangle_sides_l733_73369
-- Import the entire library mainly used for geometry and algebraic proofs.

-- Define the main problem statement as a theorem.
theorem triangle_sides (a b c : ℕ) (r_incircle : ℕ)
  (r_excircle_a r_excircle_b r_excircle_c : ℕ) (s : ℕ)
  (area : ℕ) : 
  r_incircle = 1 → 
  area = s →
  r_excircle_a * r_excircle_b * r_excircle_c = (s * s * s) →
  s = (a + b + c) / 2 →
  r_excircle_a = s / (s - a) →
  r_excircle_b = s / (s - b) →
  r_excircle_c = s / (s - c) →
  a * b = 12 → 
  a = 3 ∧ b = 4 ∧ c = 5 :=
by {
  -- Placeholder for the proof.
  sorry
}

end triangle_sides_l733_73369


namespace ladder_slides_out_l733_73317

theorem ladder_slides_out (ladder_length foot_initial_dist ladder_slip_down foot_final_dist : ℝ) 
  (h_ladder_length : ladder_length = 25)
  (h_foot_initial_dist : foot_initial_dist = 7)
  (h_ladder_slip_down : ladder_slip_down = 4)
  (h_foot_final_dist : foot_final_dist = 15) :
  foot_final_dist - foot_initial_dist = 8 :=
  by
  simp [h_ladder_length, h_foot_initial_dist, h_ladder_slip_down, h_foot_final_dist]
  sorry

end ladder_slides_out_l733_73317


namespace first_discount_percentage_l733_73378

theorem first_discount_percentage 
  (original_price final_price : ℝ) 
  (successive_discount1 successive_discount2 : ℝ) 
  (h1 : original_price = 10000)
  (h2 : final_price = 6840)
  (h3 : successive_discount1 = 0.10)
  (h4 : successive_discount2 = 0.05)
  : ∃ x, (1 - x / 100) * (1 - successive_discount1) * (1 - successive_discount2) * original_price = final_price ∧ x = 20 :=
by
  sorry

end first_discount_percentage_l733_73378


namespace green_marbles_l733_73316

theorem green_marbles 
  (total_marbles : ℕ)
  (red_marbles : ℕ)
  (at_least_blue_marbles : ℕ)
  (h1 : total_marbles = 63) 
  (h2 : at_least_blue_marbles ≥ total_marbles / 3) 
  (h3 : red_marbles = 38) 
  : ∃ green_marbles : ℕ, total_marbles - red_marbles - at_least_blue_marbles = green_marbles ∧ green_marbles = 4 :=
by
  sorry

end green_marbles_l733_73316


namespace milk_left_after_third_operation_l733_73398

theorem milk_left_after_third_operation :
  ∀ (initial_milk : ℝ), initial_milk > 0 →
  (initial_milk * 0.8 * 0.8 * 0.8 / initial_milk) * 100 = 51.2 :=
by
  intros initial_milk h_initial_milk_pos
  sorry

end milk_left_after_third_operation_l733_73398
