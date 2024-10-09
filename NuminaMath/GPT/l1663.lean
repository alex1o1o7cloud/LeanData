import Mathlib

namespace lcm_5_6_10_15_l1663_166362

theorem lcm_5_6_10_15 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 10 15) = 30 := 
by
  sorry

end lcm_5_6_10_15_l1663_166362


namespace inequality_proof_l1663_166388

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 + x + 2 * x^2) * (2 + 3 * y + y^2) * (4 + z + z^2) ≥ 60 * x * y * z :=
by
  sorry

end inequality_proof_l1663_166388


namespace cultural_festival_recommendation_schemes_l1663_166381

theorem cultural_festival_recommendation_schemes :
  (∃ (females : Finset ℕ) (males : Finset ℕ),
    females.card = 3 ∧ males.card = 2 ∧
    ∃ (dance : Finset ℕ) (singing : Finset ℕ) (instruments : Finset ℕ),
      dance.card = 2 ∧ dance ⊆ females ∧
      singing.card = 2 ∧ singing ∩ females ≠ ∅ ∧
      instruments.card = 1 ∧ instruments ⊆ males ∧
      (females ∪ males).card = 5) → 
  ∃ (recommendation_schemes : ℕ), recommendation_schemes = 18 :=
by
  sorry

end cultural_festival_recommendation_schemes_l1663_166381


namespace sales_volume_relation_maximize_profit_l1663_166307

-- Definition of the conditions given in the problem
def cost_price : ℝ := 40
def min_selling_price : ℝ := 45
def initial_selling_price : ℝ := 45
def initial_sales_volume : ℝ := 700
def sales_decrease_rate : ℝ := 20

-- Lean statement for part 1
theorem sales_volume_relation (x : ℝ) : 
  (45 ≤ x) →
  (y = 700 - 20 * (x - 45)) → 
  y = -20 * x + 1600 := sorry

-- Lean statement for part 2
theorem maximize_profit (x : ℝ) :
  (45 ≤ x) →
  (P = (x - 40) * (-20 * x + 1600)) →
  ∃ max_x max_P, max_x = 60 ∧ max_P = 8000 := sorry

end sales_volume_relation_maximize_profit_l1663_166307


namespace sum_of_all_possible_values_of_x_l1663_166371

noncomputable def sum_of_roots_of_equation : ℚ :=
  let eq : Polynomial ℚ := 4 * Polynomial.X ^ 2 + 3 * Polynomial.X - 5
  let roots := eq.roots
  roots.sum

theorem sum_of_all_possible_values_of_x :
  sum_of_roots_of_equation = -3/4 := 
  sorry

end sum_of_all_possible_values_of_x_l1663_166371


namespace total_trips_correct_l1663_166321

-- Define Timothy's movie trips in 2009
def timothy_2009_trips : ℕ := 24

-- Define Timothy's movie trips in 2010
def timothy_2010_trips : ℕ := timothy_2009_trips + 7

-- Define Theresa's movie trips in 2009
def theresa_2009_trips : ℕ := timothy_2009_trips / 2

-- Define Theresa's movie trips in 2010
def theresa_2010_trips : ℕ := timothy_2010_trips * 2

-- Define the total number of trips for Timothy and Theresa in 2009 and 2010
def total_trips : ℕ := (timothy_2009_trips + timothy_2010_trips) + (theresa_2009_trips + theresa_2010_trips)

-- Prove the total number of trips is 129
theorem total_trips_correct : total_trips = 129 :=
by
  sorry

end total_trips_correct_l1663_166321


namespace sin_cos_value_sin_minus_cos_value_tan_value_l1663_166352

variable (x : ℝ)

theorem sin_cos_value 
  (h1 : - (Real.pi / 2) < x) 
  (h2 : x < 0) 
  (h3 : Real.sin x + Real.cos x = 1 / 5) : 
  Real.sin x * Real.cos x = - 12 / 25 := 
sorry

theorem sin_minus_cos_value 
  (h1 : - (Real.pi / 2) < x) 
  (h2 : x < 0) 
  (h3 : Real.sin x + Real.cos x = 1 / 5) : 
  Real.sin x - Real.cos x = - 7 / 5 := 
sorry

theorem tan_value 
  (h1 : - (Real.pi / 2) < x) 
  (h2 : x < 0) 
  (h3 : Real.sin x + Real.cos x = 1 / 5) : 
  Real.tan x = - 3 / 4 := 
sorry

end sin_cos_value_sin_minus_cos_value_tan_value_l1663_166352


namespace gcd_poly_l1663_166372

-- Defining the conditions as stated in part a:
def is_even_multiple_of_1171 (b : ℤ) : Prop :=
  ∃ k : ℤ, b = 1171 * k * 2

-- Stating the main theorem based on the conditions and required proof in part c:
theorem gcd_poly (b : ℤ) (h : is_even_multiple_of_1171 b) : Int.gcd (3 * b ^ 2 + 47 * b + 79) (b + 17) = 1 := by
  sorry

end gcd_poly_l1663_166372


namespace molly_age_condition_l1663_166319

-- Definitions
def S : ℕ := 38 - 6
def M : ℕ := 24

-- The proof problem
theorem molly_age_condition :
  (S / M = 4 / 3) → (S = 32) → (M = 24) :=
by
  intro h_ratio h_S
  sorry

end molly_age_condition_l1663_166319


namespace maximize_ab2c3_l1663_166306

def positive_numbers (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 

def sum_constant (a b c A : ℝ) : Prop :=
  a + b + c = A

noncomputable def maximize_expression (a b c : ℝ) : ℝ :=
  a * b^2 * c^3

theorem maximize_ab2c3 (a b c A : ℝ) (h1 : positive_numbers a b c)
  (h2 : sum_constant a b c A) : 
  maximize_expression a b c ≤ maximize_expression (A / 6) (A / 3) (A / 2) :=
sorry

end maximize_ab2c3_l1663_166306


namespace ratio_of_triangle_areas_bcx_acx_l1663_166317

theorem ratio_of_triangle_areas_bcx_acx
  (BC AC : ℕ) (hBC : BC = 36) (hAC : AC = 45)
  (is_angle_bisector_CX : ∀ BX AX : ℕ, BX / AX = BC / AC) :
  (∃ BX AX : ℕ, BX / AX = 4 / 5) :=
by
  have h_ratio := is_angle_bisector_CX 36 45
  rw [hBC, hAC] at h_ratio
  exact ⟨4, 5, h_ratio⟩

end ratio_of_triangle_areas_bcx_acx_l1663_166317


namespace negation_of_AllLinearAreMonotonic_is_SomeLinearAreNotMonotonic_l1663_166328

-- Definitions based on the conditions in the problem:
def LinearFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b
def MonotonicFunction (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- The proposition that 'All linear functions are monotonic functions'
def AllLinearAreMonotonic : Prop := ∀ (f : ℝ → ℝ), LinearFunction f → MonotonicFunction f

-- The correct answer to the question:
def SomeLinearAreNotMonotonic : Prop := ∃ (f : ℝ → ℝ), LinearFunction f ∧ ¬ MonotonicFunction f

-- The proof problem:
theorem negation_of_AllLinearAreMonotonic_is_SomeLinearAreNotMonotonic : 
  ¬ AllLinearAreMonotonic ↔ SomeLinearAreNotMonotonic :=
by
  sorry

end negation_of_AllLinearAreMonotonic_is_SomeLinearAreNotMonotonic_l1663_166328


namespace value_of_m_l1663_166356

theorem value_of_m (m : ℤ) : 
  (∃ f : ℤ → ℤ, ∀ x : ℤ, x^2 - (m+1)*x + 1 = (f x)^2) → (m = 1 ∨ m = -3) := 
by
  sorry

end value_of_m_l1663_166356


namespace sulfuric_acid_moles_l1663_166394

-- Definitions based on the conditions
def iron_moles := 2
def hydrogen_moles := 2

-- The reaction equation in the problem
def reaction (Fe H₂SO₄ : ℕ) : Prop :=
  Fe + H₂SO₄ = hydrogen_moles

-- Goal: prove the number of moles of sulfuric acid used is 2
theorem sulfuric_acid_moles (Fe : ℕ) (H₂SO₄ : ℕ) (h : reaction Fe H₂SO₄) :
  H₂SO₄ = 2 :=
sorry

end sulfuric_acid_moles_l1663_166394


namespace geometric_sequence_result_l1663_166396

-- Definitions representing the conditions
variables {a : ℕ → ℝ}

-- Conditions
axiom cond1 : a 7 * a 11 = 6
axiom cond2 : a 4 + a 14 = 5

theorem geometric_sequence_result :
  ∃ x, x = a 20 / a 10 ∧ (x = 2 / 3 ∨ x = 3 / 2) :=
by {
  sorry
}

end geometric_sequence_result_l1663_166396


namespace greatest_three_digit_number_l1663_166327

theorem greatest_three_digit_number 
  (n : ℕ)
  (h1 : n % 7 = 2)
  (h2 : n % 6 = 4)
  (h3 : n ≥ 100)
  (h4 : n < 1000) :
  n = 994 :=
sorry

end greatest_three_digit_number_l1663_166327


namespace integer_root_count_l1663_166308

theorem integer_root_count (b : ℝ) :
  (∃ r s : ℤ, r + s = b ∧ r * s = 8 * b) ↔
  b = -9 ∨ b = 0 ∨ b = 9 :=
sorry

end integer_root_count_l1663_166308


namespace max_of_three_numbers_l1663_166339

theorem max_of_three_numbers : ∀ (a b c : ℕ), a = 10 → b = 11 → c = 12 → max (max a b) c = 12 :=
by
  intros a b c h1 h2 h3
  rw [h1, h2, h3]
  sorry

end max_of_three_numbers_l1663_166339


namespace family_vacation_rain_days_l1663_166338

theorem family_vacation_rain_days (r_m r_a : ℕ) 
(h_rain_days : r_m + r_a = 13)
(clear_mornings : r_a = 11)
(clear_afternoons : r_m = 12) : 
r_m + r_a = 23 := 
by 
  sorry

end family_vacation_rain_days_l1663_166338


namespace solve_for_y_l1663_166392

def diamond (a b : ℕ) : ℕ := 2 * a + b

theorem solve_for_y (y : ℕ) (h : diamond 4 (diamond 3 y) = 17) : y = 3 :=
by sorry

end solve_for_y_l1663_166392


namespace find_square_l1663_166340

theorem find_square (y : ℝ) (h : (y + 5)^(1/3) = 3) : (y + 5)^2 = 729 := 
sorry

end find_square_l1663_166340


namespace sum_of_a_b_c_d_e_l1663_166301

theorem sum_of_a_b_c_d_e (a b c d e : ℤ) (h1 : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120)
  (h2 : a ≠ b) (h3 : a ≠ c) (h4 : a ≠ d) (h5 : a ≠ e) (h6 : b ≠ c) (h7 : b ≠ d) (h8 : b ≠ e) 
  (h9 : c ≠ d) (h10 : c ≠ e) (h11 : d ≠ e) : a + b + c + d + e = 33 := by
  sorry

end sum_of_a_b_c_d_e_l1663_166301


namespace tractor_planting_rate_l1663_166335

theorem tractor_planting_rate
  (A : ℕ) (D : ℕ)
  (T1_days : ℕ) (T1 : ℕ)
  (T2_days : ℕ) (T2 : ℕ)
  (total_acres : A = 1700)
  (total_days : D = 5)
  (crew1_tractors : T1 = 2)
  (crew1_days : T1_days = 2)
  (crew2_tractors : T2 = 7)
  (crew2_days : T2_days = 3)
  : (A / (T1 * T1_days + T2 * T2_days)) = 68 := 
sorry

end tractor_planting_rate_l1663_166335


namespace range_of_a_l1663_166343

theorem range_of_a (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 - 2 * a * x + 2 < 0) → a ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) := by
  sorry

end range_of_a_l1663_166343


namespace tenth_term_ar_sequence_l1663_166390

-- Variables for the first term and common difference
variables (a1 d : ℕ) (n : ℕ)

-- Specific given values
def a1_fixed := 3
def d_fixed := 2

-- Define the nth term of the arithmetic sequence
def a_n (n : ℕ) := a1 + (n - 1) * d

-- The statement to prove
theorem tenth_term_ar_sequence : a_n 10 = 21 := by
  -- Definitions for a1 and d
  let a1 := a1_fixed
  let d := d_fixed
  -- The rest of the proof
  sorry

end tenth_term_ar_sequence_l1663_166390


namespace dog_treats_cost_l1663_166312

theorem dog_treats_cost
  (treats_per_day : ℕ)
  (cost_per_treat : ℚ)
  (days_in_month : ℕ)
  (H1 : treats_per_day = 2)
  (H2 : cost_per_treat = 0.1)
  (H3 : days_in_month = 30) :
  treats_per_day * days_in_month * cost_per_treat = 6 :=
by sorry

end dog_treats_cost_l1663_166312


namespace num_ducks_l1663_166369

variable (D G : ℕ)

theorem num_ducks (h1 : D + G = 8) (h2 : 2 * D + 4 * G = 24) : D = 4 := by
  sorry

end num_ducks_l1663_166369


namespace rain_probability_l1663_166364

theorem rain_probability :
  let PM : ℝ := 0.62
  let PT : ℝ := 0.54
  let PMcTc : ℝ := 0.28
  let PMT : ℝ := PM + PT - (1 - PMcTc)
  PMT = 0.44 :=
by
  sorry

end rain_probability_l1663_166364


namespace battery_current_l1663_166353

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l1663_166353


namespace largest_k_inequality_l1663_166332

noncomputable def k : ℚ := 39 / 2

theorem largest_k_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a + b + c)^3 ≥ (5 / 2) * (a^3 + b^3 + c^3) + k * a * b * c := 
sorry

end largest_k_inequality_l1663_166332


namespace ajax_weight_after_two_weeks_l1663_166379

/-- Initial weight of Ajax in kilograms. -/
def initial_weight_kg : ℝ := 80

/-- Conversion factor from kilograms to pounds. -/
def kg_to_pounds : ℝ := 2.2

/-- Weight lost per hour of each exercise type. -/
def high_intensity_loss_per_hour : ℝ := 4
def moderate_intensity_loss_per_hour : ℝ := 2.5
def low_intensity_loss_per_hour : ℝ := 1.5

/-- Ajax's weekly exercise routine. -/
def weekly_high_intensity_hours : ℝ := 1 * 3 + 1.5 * 1
def weekly_moderate_intensity_hours : ℝ := 0.5 * 5
def weekly_low_intensity_hours : ℝ := 1 * 2 + 0.5 * 1

/-- Calculate the total weight loss in pounds per week. -/
def total_weekly_weight_loss_pounds : ℝ :=
  weekly_high_intensity_hours * high_intensity_loss_per_hour +
  weekly_moderate_intensity_hours * moderate_intensity_loss_per_hour +
  weekly_low_intensity_hours * low_intensity_loss_per_hour

/-- Calculate the total weight loss in pounds for two weeks. -/
def total_weight_loss_pounds_for_two_weeks : ℝ :=
  total_weekly_weight_loss_pounds * 2

/-- Calculate Ajax's initial weight in pounds. -/
def initial_weight_pounds : ℝ :=
  initial_weight_kg * kg_to_pounds

/-- Calculate Ajax's new weight after two weeks. -/
def new_weight_pounds : ℝ :=
  initial_weight_pounds - total_weight_loss_pounds_for_two_weeks

/-- Prove that Ajax's new weight in pounds is 120 after following the workout schedule for two weeks. -/
theorem ajax_weight_after_two_weeks :
  new_weight_pounds = 120 :=
by
  sorry

end ajax_weight_after_two_weeks_l1663_166379


namespace chickens_and_rabbits_l1663_166384

theorem chickens_and_rabbits (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : x = 23 ∧ y = 12 :=
by
  sorry

end chickens_and_rabbits_l1663_166384


namespace trapezoid_circumscribed_radius_l1663_166304

theorem trapezoid_circumscribed_radius 
  (a b : ℝ) 
  (height : ℝ)
  (ratio_ab : a / b = 5 / 12)
  (height_eq_midsegment : height = 17) :
  ∃ r : ℝ, r = 13 :=
by
  -- Assuming conditions directly as given
  have h1 : a / b = 5 / 12 := ratio_ab
  have h2 : height = 17 := height_eq_midsegment
  -- The rest of the proof goes here
  sorry

end trapezoid_circumscribed_radius_l1663_166304


namespace pet_store_problem_l1663_166347

theorem pet_store_problem 
  (initial_puppies : ℕ) 
  (sold_day1 : ℕ) 
  (sold_day2 : ℕ) 
  (sold_day3 : ℕ) 
  (sold_day4 : ℕ)
  (sold_day5 : ℕ) 
  (puppies_per_cage : ℕ)
  (initial_puppies_eq : initial_puppies = 120) 
  (sold_day1_eq : sold_day1 = 25) 
  (sold_day2_eq : sold_day2 = 10) 
  (sold_day3_eq : sold_day3 = 30) 
  (sold_day4_eq : sold_day4 = 15) 
  (sold_day5_eq : sold_day5 = 28) 
  (puppies_per_cage_eq : puppies_per_cage = 6) : 
  (initial_puppies - (sold_day1 + sold_day2 + sold_day3 + sold_day4 + sold_day5)) / puppies_per_cage = 2 := 
by 
  sorry

end pet_store_problem_l1663_166347


namespace smallest_c_for_inverse_l1663_166318

noncomputable def g (x : ℝ) : ℝ := (x + 3)^2 - 6

theorem smallest_c_for_inverse : 
  ∃ (c : ℝ), (∀ x1 x2, x1 ≥ c → x2 ≥ c → g x1 = g x2 → x1 = x2) ∧ 
            (∀ c', c' < c → ∃ x1 x2, x1 ≥ c' → x2 ≥ c' → g x1 = g x2 ∧ x1 ≠ x2) ∧ 
            c = -3 :=
by 
  sorry

end smallest_c_for_inverse_l1663_166318


namespace product_inequality_l1663_166397

variable (x1 x2 x3 x4 y1 y2 : ℝ)

theorem product_inequality (h1 : y2 ≥ y1) 
                          (h2 : y1 ≥ x1)
                          (h3 : x1 ≥ x3)
                          (h4 : x3 ≥ x2)
                          (h5 : x2 ≥ x1)
                          (h6 : x1 ≥ 2)
                          (h7 : x1 + x2 + x3 + x4 ≥ y1 + y2) : 
                          x1 * x2 * x3 * x4 ≥ y1 * y2 :=
  sorry

end product_inequality_l1663_166397


namespace vectors_projection_l1663_166334

noncomputable def p := (⟨-44 / 53, 154 / 53⟩ : ℝ × ℝ)

theorem vectors_projection :
  let u := (⟨-4, 2⟩ : ℝ × ℝ)
  let v := (⟨3, 4⟩ : ℝ × ℝ)
  let w := (⟨7, 2⟩ : ℝ × ℝ)
  (⟨(7 * (24 / 53)) - 4, (2 * (24 / 53)) + 2⟩ : ℝ × ℝ) = p :=
by {
  -- proof skipped
  sorry
}

end vectors_projection_l1663_166334


namespace totalPeoplePresent_l1663_166349

-- Defining the constants based on the problem conditions
def associateProfessors := 2
def assistantProfessors := 7

def totalPencils := 11
def totalCharts := 16

-- The main proof statement
theorem totalPeoplePresent :
  (∃ (A B : ℕ), (2 * A + B = totalPencils) ∧ (A + 2 * B = totalCharts)) →
  (associateProfessors + assistantProfessors = 9) :=
  by
  sorry

end totalPeoplePresent_l1663_166349


namespace max_abc_value_l1663_166373

variables (a b c : ℕ)

theorem max_abc_value : 
  (a > 0) → (b > 0) → (c > 0) → a + 2 * b + 3 * c = 100 → abc ≤ 6171 := 
by sorry

end max_abc_value_l1663_166373


namespace find_sum_of_squares_of_roots_l1663_166382

theorem find_sum_of_squares_of_roots (a b c : ℝ) (h_ab : a < b) (h_bc : b < c)
  (f : ℝ → ℝ) (hf : ∀ x, f x = x^3 - 2 * x^2 - 3 * x + 4)
  (h_eq : f a = f b ∧ f b = f c) :
  a^2 + b^2 + c^2 = 10 :=
sorry

end find_sum_of_squares_of_roots_l1663_166382


namespace marks_fathers_gift_l1663_166383

noncomputable def total_spent (books : ℕ) (cost_per_book : ℕ) : ℕ :=
  books * cost_per_book

noncomputable def total_money_given (spent : ℕ) (left_over : ℕ) : ℕ :=
  spent + left_over

theorem marks_fathers_gift :
  total_money_given (total_spent 10 5) 35 = 85 := by
  sorry

end marks_fathers_gift_l1663_166383


namespace question_l1663_166357

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end question_l1663_166357


namespace infinite_primes_divide_f_l1663_166303

def non_constant_function (f : ℕ → ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ f a ≠ f b

def divisibility_condition (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a ≠ b → (a - b) ∣ (f a - f b)

theorem infinite_primes_divide_f (f : ℕ → ℕ) 
  (h_non_const : non_constant_function f)
  (h_div : divisibility_condition f) :
  ∃ᶠ p in Filter.atTop, ∃ c : ℕ, p ∣ f c := sorry

end infinite_primes_divide_f_l1663_166303


namespace scientific_notation_of_300_million_l1663_166374

theorem scientific_notation_of_300_million : 
  300000000 = 3 * 10^8 := 
by
  sorry

end scientific_notation_of_300_million_l1663_166374


namespace janet_overtime_multiple_l1663_166376

theorem janet_overtime_multiple :
  let hourly_rate := 20
  let weekly_hours := 52
  let regular_hours := 40
  let car_price := 4640
  let weeks_needed := 4
  let normal_weekly_earning := regular_hours * hourly_rate
  let overtime_hours := weekly_hours - regular_hours
  let required_weekly_earning := car_price / weeks_needed
  let overtime_weekly_earning := required_weekly_earning - normal_weekly_earning
  let overtime_rate := overtime_weekly_earning / overtime_hours
  (overtime_rate / hourly_rate = 1.5) :=
by
  sorry

end janet_overtime_multiple_l1663_166376


namespace complex_expression_value_l1663_166367

theorem complex_expression_value {i : ℂ} (h : i^2 = -1) : i^3 * (1 + i)^2 = 2 := 
by
  sorry

end complex_expression_value_l1663_166367


namespace carolyn_sum_of_removed_numbers_eq_31_l1663_166395

theorem carolyn_sum_of_removed_numbers_eq_31 :
  let initial_list := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let carolyn_first_turn := 4
  let carolyn_numbers_removed := [4, 9, 10, 8]
  let sum := carolyn_numbers_removed.sum
  sum = 31 :=
by
  sorry

end carolyn_sum_of_removed_numbers_eq_31_l1663_166395


namespace range_of_a_l1663_166305

noncomputable def p (a : ℝ) := ∀ x : ℝ, x^2 + a ≥ 0
noncomputable def q (a : ℝ) := ∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0

theorem range_of_a (a : ℝ) : (p a ∧ q a) → (a ≥ 0) := by
  sorry

end range_of_a_l1663_166305


namespace positive_difference_eq_250_l1663_166393

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l1663_166393


namespace part_one_solution_set_part_two_m_range_l1663_166375

theorem part_one_solution_set (m : ℝ) (x : ℝ) (h : m = 0) : ((m - 1) * x ^ 2 + (m - 1) * x + 2 > 0) ↔ (-2 < x ∧ x < 1) :=
by
  sorry

theorem part_two_m_range (m : ℝ) : (∀ x : ℝ, (m - 1) * x ^ 2 + (m - 1) * x + 2 > 0) ↔ (1 ≤ m ∧ m < 9) :=
by
  sorry

end part_one_solution_set_part_two_m_range_l1663_166375


namespace snowfall_total_l1663_166309

theorem snowfall_total (snowfall_wed snowfall_thu snowfall_fri : ℝ)
  (h_wed : snowfall_wed = 0.33)
  (h_thu : snowfall_thu = 0.33)
  (h_fri : snowfall_fri = 0.22) :
  snowfall_wed + snowfall_thu + snowfall_fri = 0.88 :=
by
  rw [h_wed, h_thu, h_fri]
  norm_num

end snowfall_total_l1663_166309


namespace complement_intersection_l1663_166365

universe u

def U : Finset Int := {-3, -2, -1, 0, 1}
def A : Finset Int := {-2, -1}
def B : Finset Int := {-3, -1, 0}

def complement_U (A : Finset Int) (U : Finset Int) : Finset Int :=
  U.filter (λ x => x ∉ A)

theorem complement_intersection :
  (complement_U A U) ∩ B = {-3, 0} :=
by
  sorry

end complement_intersection_l1663_166365


namespace johns_average_speed_l1663_166322

def continuous_driving_duration (start_time end_time : ℝ) (distance : ℝ) : Prop :=
start_time = 10.5 ∧ end_time = 14.75 ∧ distance = 190

theorem johns_average_speed
  (start_time end_time : ℝ) 
  (distance : ℝ)
  (h : continuous_driving_duration start_time end_time distance) :
  (distance / (end_time - start_time) = 44.7) :=
by
  sorry

end johns_average_speed_l1663_166322


namespace pyramid_partition_volumes_l1663_166302

noncomputable def pyramid_partition_ratios (S A B C D P Q V1 V2 : ℝ) : Prop :=
  let P := ((S + B) / 2 : ℝ)
  let Q := ((S + D) / 2 : ℝ)
  (V1 < V2) → 
  (V2 / V1 = 5)

theorem pyramid_partition_volumes
  (S A B C D P Q : ℝ)
  (V1 V2 : ℝ)
  (hP : P = (S + B) / 2)
  (hQ : Q = (S + D) / 2)
  (hV1 : V1 < V2)
  : V2 / V1 = 5 := 
sorry

end pyramid_partition_volumes_l1663_166302


namespace large_pizza_slices_l1663_166330

variable (L : ℕ)

theorem large_pizza_slices :
  (2 * L + 2 * 8 = 48) → (L = 16) :=
by 
  sorry

end large_pizza_slices_l1663_166330


namespace find_ab_plus_a_plus_b_l1663_166359

-- Define the polynomial
def quartic_poly (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 - 6*x - 1

-- Define the roots conditions
def is_root (p : ℝ → ℝ) (r : ℝ) : Prop := p r = 0

-- State the proof problem
theorem find_ab_plus_a_plus_b :
  ∃ a b : ℝ,
    is_root quartic_poly a ∧
    is_root quartic_poly b ∧
    ab = a * b ∧
    a_plus_b = a + b ∧
    ab + a_plus_b = 4 :=
by sorry

end find_ab_plus_a_plus_b_l1663_166359


namespace oli_scoops_l1663_166363

theorem oli_scoops : ∃ x : ℤ, ∀ y : ℤ, y = 2 * x ∧ y = x + 4 → x = 4 :=
by
  sorry

end oli_scoops_l1663_166363


namespace problem_solution_l1663_166354

variable (a : ℝ)

theorem problem_solution (h : a^2 + a - 3 = 0) : a^2 * (a + 4) = 9 := by
  sorry

end problem_solution_l1663_166354


namespace exists_base_for_1994_no_base_for_1993_l1663_166344

-- Problem 1: Existence of a base for 1994 with identical digits
theorem exists_base_for_1994 :
  ∃ b : ℕ, 1 < b ∧ b < 1993 ∧ (∃ a : ℕ, ∀ n : ℕ, 1994 = a * ((b ^ n - 1) / (b - 1)) ∧ a = 2) :=
sorry

-- Problem 2: Non-existence of a base for 1993 with identical digits
theorem no_base_for_1993 :
  ¬∃ b : ℕ, 1 < b ∧ b < 1992 ∧ (∃ a : ℕ, ∀ n : ℕ, 1993 = a * ((b ^ n - 1) / (b - 1))) :=
sorry

end exists_base_for_1994_no_base_for_1993_l1663_166344


namespace find_y_when_x_is_7_l1663_166336

theorem find_y_when_x_is_7
  (x y : ℝ)
  (h1 : x * y = 384)
  (h2 : x + y = 40)
  (h3 : x - y = 8)
  (h4 : x = 7) :
  y = 384 / 7 :=
by
  sorry

end find_y_when_x_is_7_l1663_166336


namespace find_b_l1663_166358

def has_exactly_one_real_solution (f : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, f x = 0

theorem find_b (b : ℝ) :
  (∃! (x : ℝ), x^3 - b * x^2 - 3 * b * x + b^2 - 4 = 0) ↔ b < 2 :=
by
  sorry

end find_b_l1663_166358


namespace tamara_is_68_inch_l1663_166315

-- Defining the conditions
variables (K T : ℕ)

-- Condition 1: Tamara's height in terms of Kim's height
def tamara_height := T = 3 * K - 4

-- Condition 2: Combined height of Tamara and Kim
def combined_height := T + K = 92

-- Statement to prove: Tamara's height is 68 inches
theorem tamara_is_68_inch (h1 : tamara_height T K) (h2 : combined_height T K) : T = 68 :=
by
  sorry

end tamara_is_68_inch_l1663_166315


namespace min_value_expression_min_value_is_7_l1663_166313

theorem min_value_expression (x : ℝ) (hx : x > 0) : 
  6 * x + 1 / (x^6) ≥ 7 :=
sorry

theorem min_value_is_7 : 
  6 * 1 + 1 / (1^6) = 7 :=
by norm_num

end min_value_expression_min_value_is_7_l1663_166313


namespace number_of_Cl_atoms_l1663_166345

/-- 
Given a compound with 1 aluminum atom and a molecular weight of 132 g/mol,
prove that the number of chlorine atoms in the compound is 3.
--/
theorem number_of_Cl_atoms 
  (weight_Al : ℝ) 
  (weight_Cl : ℝ) 
  (molecular_weight : ℝ)
  (ha : weight_Al = 26.98)
  (hc : weight_Cl = 35.45)
  (hm : molecular_weight = 132) :
  ∃ n : ℕ, n = 3 :=
by
  sorry

end number_of_Cl_atoms_l1663_166345


namespace circle_center_coordinates_l1663_166386

-- Definition of the circle's equation
def circle_eq : Prop := ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 3

-- Proof of the circle's center coordinates
theorem circle_center_coordinates : ∃ h k : ℝ, (h, k) = (2, -1) := 
sorry

end circle_center_coordinates_l1663_166386


namespace divide_estate_l1663_166329

theorem divide_estate (total_estate : ℕ) (son_share : ℕ) (daughter_share : ℕ) (wife_share : ℕ) :
  total_estate = 210 →
  son_share = (4 / 7) * total_estate →
  daughter_share = (1 / 7) * total_estate →
  wife_share = (2 / 7) * total_estate →
  son_share + daughter_share + wife_share = total_estate :=
by
  intros
  sorry

end divide_estate_l1663_166329


namespace find_a_b_of_solution_set_l1663_166350

theorem find_a_b_of_solution_set :
  ∃ a b : ℝ, (∀ x : ℝ, x^2 + (a + 1) * x + a * b = 0 ↔ x = -1 ∨ x = 4) → a + b = -3 :=
by
  sorry

end find_a_b_of_solution_set_l1663_166350


namespace toms_final_stamp_count_l1663_166355

-- Definitions of the given conditions

def initial_stamps : ℕ := 3000
def mike_gift : ℕ := 17
def harry_gift : ℕ := 2 * mike_gift + 10
def sarah_gift : ℕ := 3 * mike_gift - 5
def damaged_stamps : ℕ := 37

-- Statement of the goal
theorem toms_final_stamp_count :
  initial_stamps + mike_gift + harry_gift + sarah_gift - damaged_stamps = 3070 :=
by
  sorry

end toms_final_stamp_count_l1663_166355


namespace final_result_is_110_l1663_166316

theorem final_result_is_110 (x : ℕ) (h1 : x = 155) : (x * 2 - 200) = 110 :=
by
  -- placeholder for the solution proof
  sorry

end final_result_is_110_l1663_166316


namespace distance_a_beats_b_l1663_166377

noncomputable def time_a : ℕ := 90 -- A's time in seconds 
noncomputable def time_b : ℕ := 180 -- B's time in seconds 
noncomputable def distance : ℝ := 4.5 -- distance in km

theorem distance_a_beats_b : distance = (distance / time_a) * (time_b - time_a) :=
by
  -- sorry placeholder for proof
  sorry

end distance_a_beats_b_l1663_166377


namespace average_rate_decrease_price_reduction_l1663_166324

-- Define the initial and final factory prices
def initial_price : ℝ := 200
def final_price : ℝ := 162

-- Define the function representing the average rate of decrease
def average_rate_of_decrease (x : ℝ) : Prop :=
  initial_price * (1 - x) * (1 - x) = final_price

-- Theorem stating the average rate of decrease (proving x = 0.1)
theorem average_rate_decrease : ∃ x : ℝ, average_rate_of_decrease x ∧ x = 0.1 :=
by
  use 0.1
  sorry

-- Define the selling price without reduction, sold without reduction, increase in pieces sold, and profit
def selling_price : ℝ := 200
def sold_without_reduction : ℕ := 20
def increase_pcs_per_5yuan_reduction : ℕ := 10
def profit : ℝ := 1150

-- Define the function representing the price reduction determination
def price_reduction_correct (m : ℝ) : Prop :=
  (38 - m) * (sold_without_reduction + 2 * m / 5) = profit

-- Theorem stating the price reduction (proving m = 15)
theorem price_reduction : ∃ m : ℝ, price_reduction_correct m ∧ m = 15 :=
by
  use 15
  sorry

end average_rate_decrease_price_reduction_l1663_166324


namespace find_a_l1663_166341

-- Define the conditions for the lines l1 and l2
def line1 (a x y : ℝ) : Prop := a * x + 4 * y + 6 = 0
def line2 (a x y : ℝ) : Prop := ((3/4) * a + 1) * x + a * y - (3/2) = 0

-- Define the condition for parallel lines
def parallel (a : ℝ) : Prop := a^2 - 4 * ((3/4) * a + 1) = 0 ∧ 4 * (-3/2) - 6 * a ≠ 0

-- Define the condition for perpendicular lines
def perpendicular (a : ℝ) : Prop := a * ((3/4) * a + 1) + 4 * a = 0

-- The theorem to prove values of a for which l1 is parallel or perpendicular to l2
theorem find_a (a : ℝ) :
  (parallel a → a = 4) ∧ (perpendicular a → a = 0 ∨ a = -20/3) :=
by
  sorry

end find_a_l1663_166341


namespace initial_bags_of_rice_l1663_166380

theorem initial_bags_of_rice (sold restocked final initial : Int) 
  (h1 : sold = 23)
  (h2 : restocked = 132)
  (h3 : final = 164) 
  : ((initial - sold) + restocked = final) ↔ initial = 55 :=
by 
  have eq1 : ((initial - sold) + restocked = final) ↔ initial - 23 + 132 = 164 := by rw [h1, h2, h3]
  simp [eq1]
  sorry

end initial_bags_of_rice_l1663_166380


namespace prudence_sleep_4_weeks_equals_200_l1663_166399

-- Conditions
def sunday_to_thursday_sleep := 6 
def friday_saturday_sleep := 9 
def nap := 1 

-- Number of days in the mentioned periods per week
def sunday_to_thursday_days := 5
def friday_saturday_days := 2
def nap_days := 2

-- Calculate total sleep per week
def total_sleep_per_week : Nat :=
  (sunday_to_thursday_days * sunday_to_thursday_sleep) +
  (friday_saturday_days * friday_saturday_sleep) +
  (nap_days * nap)

-- Calculate total sleep in 4 weeks
def total_sleep_in_4_weeks : Nat :=
  4 * total_sleep_per_week

theorem prudence_sleep_4_weeks_equals_200 : total_sleep_in_4_weeks = 200 := by
  sorry

end prudence_sleep_4_weeks_equals_200_l1663_166399


namespace arithmetic_sequence_ratio_l1663_166325

theorem arithmetic_sequence_ratio (x y a₁ a₂ a₃ b₁ b₂ b₃ b₄ : ℝ) (h₁ : x ≠ y)
    (h₂ : a₁ = x + d) (h₃ : a₂ = x + 2 * d) (h₄ : a₃ = x + 3 * d) (h₅ : y = x + 4 * d)
    (h₆ : b₁ = x - d') (h₇ : b₂ = x + d') (h₈ : b₃ = x + 2 * d') (h₉ : y = x + 3 * d') (h₁₀ : b₄ = x + 4 * d') :
    (b₄ - b₃) / (a₂ - a₁) = 8 / 3 :=
by
  sorry

end arithmetic_sequence_ratio_l1663_166325


namespace age_of_john_l1663_166337

theorem age_of_john (J S : ℕ) 
  (h1 : S = 2 * J)
  (h2 : S + (50 - J) = 60) :
  J = 10 :=
sorry

end age_of_john_l1663_166337


namespace parabolas_intersect_on_circle_l1663_166300

theorem parabolas_intersect_on_circle :
  let parabola1 (x y : ℝ) := y = (x - 2)^2
  let parabola2 (x y : ℝ) := x + 6 = (y + 1)^2
  ∃ (cx cy r : ℝ), ∀ (x y : ℝ), (parabola1 x y ∧ parabola2 x y) → (x - cx)^2 + (y - cy)^2 = r^2 ∧ r^2 = 33/2 :=
by
  sorry

end parabolas_intersect_on_circle_l1663_166300


namespace frood_least_throw_points_more_than_eat_l1663_166326

theorem frood_least_throw_points_more_than_eat (n : ℕ) : n^2 > 12 * n ↔ n ≥ 13 :=
sorry

end frood_least_throw_points_more_than_eat_l1663_166326


namespace probability_heart_and_face_card_club_l1663_166366

-- Conditions
def num_cards : ℕ := 52
def num_hearts : ℕ := 13
def num_face_card_clubs : ℕ := 3

-- Define the probabilities
def prob_heart_first : ℚ := num_hearts / num_cards
def prob_face_card_club_given_heart : ℚ := num_face_card_clubs / (num_cards - 1)

-- Proof statement
theorem probability_heart_and_face_card_club :
  prob_heart_first * prob_face_card_club_given_heart = 3 / 204 :=
by
  sorry

end probability_heart_and_face_card_club_l1663_166366


namespace kombucha_bottles_l1663_166314

theorem kombucha_bottles (b_m : ℕ) (c : ℝ) (r : ℝ) (m : ℕ)
  (hb : b_m = 15) (hc : c = 3.00) (hr : r = 0.10) (hm : m = 12) :
  (b_m * m * r) / c = 6 := by
  sorry

end kombucha_bottles_l1663_166314


namespace average_candies_correct_l1663_166368

def candy_counts : List ℕ := [16, 22, 30, 26, 18, 20]
def num_members : ℕ := 6
def total_candies : ℕ := List.sum candy_counts
def average_candies : ℕ := total_candies / num_members

theorem average_candies_correct : average_candies = 22 := by
  -- Proof is omitted, as per instructions
  sorry

end average_candies_correct_l1663_166368


namespace xy_value_l1663_166346

structure Point (R : Type) := (x : R) (y : R)

def A : Point ℝ := ⟨2, 7⟩ 
def C : Point ℝ := ⟨4, 3⟩ 

def is_midpoint (A B C : Point ℝ) : Prop :=
  (C.x = (A.x + B.x) / 2) ∧ (C.y = (A.y + B.y) / 2)

theorem xy_value (x y : ℝ) (B : Point ℝ := ⟨x, y⟩) (H : is_midpoint A B C) :
  x * y = -6 := 
sorry

end xy_value_l1663_166346


namespace area_of_square_A_l1663_166310

noncomputable def square_areas (a b : ℕ) : Prop :=
  (b ^ 2 = 81) ∧ (a = b + 4)

theorem area_of_square_A : ∃ a b : ℕ, square_areas a b → a ^ 2 = 169 :=
by
  sorry

end area_of_square_A_l1663_166310


namespace sum_of_squares_of_coeffs_l1663_166331

theorem sum_of_squares_of_coeffs :
  let p := 3 * (x^5 + 5 * x^3 + 2 * x + 1)
  let coeffs := [3, 15, 6, 3]
  coeffs.map (λ c => c^2) |>.sum = 279 := by
  sorry

end sum_of_squares_of_coeffs_l1663_166331


namespace point_on_line_l1663_166311

theorem point_on_line (m n k : ℝ) (h1 : m = 2 * n + 5) (h2 : m + 4 = 2 * (n + k) + 5) : k = 2 := by
  sorry

end point_on_line_l1663_166311


namespace question1_question2_l1663_166385

namespace MathProofs

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

-- Definitions based on conditions
def isA := ∀ x, A x ↔ (-3 < x ∧ x < 2)
def isB := ∀ x, B x ↔ (Real.exp (x - 1) ≥ 1)
def isCuA := ∀ x, (U \ A) x ↔ (x ≤ -3 ∨ x ≥ 2)

-- Proof of Question 1
theorem question1 : (∀ x, (A ∪ B) x ↔ (x > -3)) := by
  sorry

-- Proof of Question 2
theorem question2 : (∀ x, ((U \ A) ∩ B) x ↔ (x ≥ 2)) := by
  sorry

end MathProofs

end question1_question2_l1663_166385


namespace gain_percentage_l1663_166333

theorem gain_percentage (selling_price gain : ℝ) (h1 : selling_price = 225) (h2 : gain = 75) : 
  (gain / (selling_price - gain) * 100) = 50 :=
by
  sorry

end gain_percentage_l1663_166333


namespace total_amount_l1663_166351

-- Define the conditions in Lean
variables (X Y Z: ℝ)
variable (h1 : Y = 0.75 * X)
variable (h2 : Z = (2/3) * X)
variable (h3 : Y = 48)

-- The theorem stating that the total amount of money is Rs. 154.67
theorem total_amount (X Y Z : ℝ) (h1 : Y = 0.75 * X) (h2 : Z = (2/3) * X) (h3 : Y = 48) : 
  X + Y + Z = 154.67 := 
by
  sorry

end total_amount_l1663_166351


namespace contrapositive_of_given_condition_l1663_166320

-- Definitions
variable (P Q : Prop)

-- Given condition: If Jane answered all questions correctly, she will get a prize
axiom h : P → Q

-- Statement to be proven: If Jane did not get a prize, she answered at least one question incorrectly
theorem contrapositive_of_given_condition : ¬ Q → ¬ P := by
  sorry

end contrapositive_of_given_condition_l1663_166320


namespace average_value_l1663_166360

variable (z : ℝ)

theorem average_value : (0 + 2 * z^2 + 4 * z^2 + 8 * z^2 + 16 * z^2) / 5 = 6 * z^2 :=
by
  sorry

end average_value_l1663_166360


namespace quadratic_no_real_roots_l1663_166398

-- Given conditions
variables {p q a b c : ℝ}
variables (hp_pos : 0 < p) (hq_pos : 0 < q) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
variables (hp_neq_q : p ≠ q)

-- p, a, q form a geometric sequence
variables (h_geo : a^2 = p * q)

-- p, b, c, q form an arithmetic sequence
variables (h_arith1 : 2 * b = p + c)
variables (h_arith2 : 2 * c = b + q)

-- Proof statement
theorem quadratic_no_real_roots (hp_pos hq_pos ha_pos hb_pos hc_pos hp_neq_q h_geo h_arith1 h_arith2 : ℝ) :
    (b * (x : ℝ)^2 - 2 * a * x + c = 0) → false :=
sorry

end quadratic_no_real_roots_l1663_166398


namespace find_paintings_l1663_166378

noncomputable def cost_painting (P : ℕ) : ℝ := 40 * P
noncomputable def cost_toy : ℝ := 20 * 8
noncomputable def total_cost (P : ℕ) : ℝ := cost_painting P + cost_toy

noncomputable def sell_painting (P : ℕ) : ℝ := 36 * P
noncomputable def sell_toy : ℝ := 17 * 8
noncomputable def total_sell (P : ℕ) : ℝ := sell_painting P + sell_toy

noncomputable def total_loss (P : ℕ) : ℝ := total_cost P - total_sell P

theorem find_paintings : ∀ (P : ℕ), total_loss P = 64 → P = 10 :=
by
  intros P h
  sorry

end find_paintings_l1663_166378


namespace nilpotent_matrix_squared_zero_l1663_166342

variable {R : Type*} [Field R]
variable (A : Matrix (Fin 2) (Fin 2) R)

theorem nilpotent_matrix_squared_zero (h : A^4 = 0) : A^2 = 0 := 
sorry

end nilpotent_matrix_squared_zero_l1663_166342


namespace geom_series_sum_l1663_166348

def geom_sum (b1 : ℚ) (r : ℚ) (n : ℕ) : ℚ := 
  b1 * (1 - r^n) / (1 - r)

def b1 : ℚ := 3 / 4
def r : ℚ := 3 / 4
def n : ℕ := 15

theorem geom_series_sum :
  geom_sum b1 r n = 3177884751 / 1073741824 :=
by sorry

end geom_series_sum_l1663_166348


namespace agreed_upon_service_period_l1663_166389

theorem agreed_upon_service_period (x : ℕ) (hx : 900 + 100 = 1000) 
(assumed_service : x * 1000 = 9 * (650 + 100)) :
  x = 12 :=
by {
  sorry
}

end agreed_upon_service_period_l1663_166389


namespace two_candidates_solve_all_problems_l1663_166370

-- Definitions for the conditions and problem context
def candidates : Nat := 200
def problems : Nat := 6 
def solved_by (p : Nat) : Nat := 120 -- at least 120 participants solve each problem.

-- The main theorem representing the proof problem
theorem two_candidates_solve_all_problems :
  (∃ c1 c2 : Fin candidates, ∀ p : Fin problems, (solved_by p ≥ 120)) :=
by
  sorry

end two_candidates_solve_all_problems_l1663_166370


namespace domain_g_eq_l1663_166323

noncomputable def domain_f : Set ℝ := {x | -8 ≤ x ∧ x ≤ 4}

noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-2 * x)

theorem domain_g_eq (f : ℝ → ℝ) (h : ∀ x, x ∈ domain_f → f x ∈ domain_f) :
  {x | x ∈ [-2, 4]} = {x | -2 ≤ x ∧ x ≤ 4} :=
by {
  sorry
}

end domain_g_eq_l1663_166323


namespace max_cubes_submerged_l1663_166387

noncomputable def cylinder_radius (diameter: ℝ) : ℝ := diameter / 2

noncomputable def water_volume (radius height: ℝ) : ℝ := Real.pi * radius^2 * height

noncomputable def cube_volume (edge: ℝ) : ℝ := edge^3

noncomputable def height_of_cubes (edge n: ℝ) : ℝ := edge * n

theorem max_cubes_submerged (diameter height water_height edge: ℝ) 
  (h1: diameter = 2.9)
  (h2: water_height = 4)
  (h3: edge = 2):
  ∃ max_n: ℝ, max_n = 5 := 
  sorry

end max_cubes_submerged_l1663_166387


namespace both_students_given_correct_l1663_166391

open ProbabilityTheory

variables (P_A P_B : ℝ)

-- Define the conditions from part a)
def student_a_correct := P_A = 3 / 5
def student_b_correct := P_B = 1 / 3

-- Define the event that both students correctly answer
def both_students_correct := P_A * P_B

-- Define the event that the question is answered correctly
def question_answered_correctly := (P_A * (1 - P_B)) + ((1 - P_A) * P_B) + (P_A * P_B)

-- Define the conditional probability we need to prove
theorem both_students_given_correct (hA : student_a_correct P_A) (hB : student_b_correct P_B) :
  both_students_correct P_A P_B / question_answered_correctly P_A P_B = 3 / 11 := 
sorry

end both_students_given_correct_l1663_166391


namespace find_b_l1663_166361

-- Definitions from the conditions
variables (a b : ℝ)

-- Theorem statement using the conditions and the correct answer
theorem find_b (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 2) : b = 2 :=
by
  sorry

end find_b_l1663_166361
