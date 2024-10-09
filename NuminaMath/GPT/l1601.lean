import Mathlib

namespace sequence_a_10_value_l1601_160162

theorem sequence_a_10_value : 
  ∀ (a : ℕ → ℤ), a 1 = 3 → (∀ n : ℕ, 0 < n → a (n + 1) - a n = 2) → a 10 = 21 := 
by 
  intros a h1 hdiff
  sorry

end sequence_a_10_value_l1601_160162


namespace solution_to_quadratic_inequality_l1601_160146

def quadratic_inequality (x : ℝ) : Prop := 3 * x^2 - 5 * x > 9

theorem solution_to_quadratic_inequality (x : ℝ) : quadratic_inequality x ↔ x < -1 ∨ x > 3 :=
by
  sorry

end solution_to_quadratic_inequality_l1601_160146


namespace parabola_directrix_l1601_160154

theorem parabola_directrix (p : ℝ) (hp : p > 0) 
  (x1 x2 t : ℝ) 
  (h_intersect : ∃ y1 y2, y1 = x1 + t ∧ y2 = x2 + t ∧ x1^2 = 2 * p * y1 ∧ x2^2 = 2 * p * y2)
  (h_midpoint : (x1 + x2) / 2 = 2) :
  p = 2 → ∃ d : ℝ, d = -1 := 
by
  sorry

end parabola_directrix_l1601_160154


namespace no_such_arrangement_exists_l1601_160106

theorem no_such_arrangement_exists :
  ¬ ∃ (f : ℕ → ℕ) (c : ℕ), 
    (∀ n, 1 ≤ f n ∧ f n ≤ 1331) ∧
    (∀ x y z, f (x + 11 * y + 121 * z) = c → f ((x+1) + 11 * y + 121 * z) = c + 8) ∧
    (∀ x y z, f (x + 11 * y + 121 * z) = c → f (x + 11 * (y+1) + 121 * z) = c + 9) :=
sorry

end no_such_arrangement_exists_l1601_160106


namespace factor_tree_X_value_l1601_160163

-- Define the constants
def F : ℕ := 5 * 3
def G : ℕ := 7 * 3

-- Define the intermediate values
def Y : ℕ := 5 * F
def Z : ℕ := 7 * G

-- Final value of X
def X : ℕ := Y * Z

-- Prove the value of X
theorem factor_tree_X_value : X = 11025 := by
  sorry

end factor_tree_X_value_l1601_160163


namespace division_remainder_l1601_160170

def polynomial (x: ℤ) : ℤ := 3 * x^7 - x^6 - 7 * x^5 + 2 * x^3 + 4 * x^2 - 11
def divisor (x: ℤ) : ℤ := 2 * x - 4

theorem division_remainder : (polynomial 2) = 117 := 
  by 
  -- We state what needs to be proven here formally
  sorry

end division_remainder_l1601_160170


namespace find_c_l1601_160109

theorem find_c (c : ℝ) : 
  (∀ x : ℝ, x * (3 * x + 1) < c ↔ x ∈ Set.Ioo (-(7 / 3) : ℝ) (2 : ℝ)) → c = 14 :=
by
  intro h
  sorry

end find_c_l1601_160109


namespace max_correct_answers_l1601_160190

theorem max_correct_answers (a b c : ℕ) :
  a + b + c = 50 ∧ 4 * a - c = 99 ∧ b = 50 - a - c ∧ 50 - a - c ≥ 0 →
  a ≤ 29 := by
  sorry

end max_correct_answers_l1601_160190


namespace probability_same_outcomes_l1601_160107

-- Let us define the event space for a fair coin
inductive CoinTossOutcome
| H : CoinTossOutcome
| T : CoinTossOutcome

open CoinTossOutcome

-- Definition of an event where the outcomes are the same (HHH or TTT)
def same_outcomes (t1 t2 t3 : CoinTossOutcome) : Prop :=
  (t1 = H ∧ t2 = H ∧ t3 = H) ∨ (t1 = T ∧ t2 = T ∧ t3 = T)

-- Number of all possible outcomes for three coin tosses
def total_outcomes : ℕ := 2 ^ 3

-- Number of favorable outcomes where all outcomes are the same
def favorable_outcomes : ℕ := 2

-- Calculation of probability
def prob_same_outcomes : ℚ := favorable_outcomes / total_outcomes

-- The statement to be proved in Lean 4
theorem probability_same_outcomes : prob_same_outcomes = 1 / 4 := 
by sorry

end probability_same_outcomes_l1601_160107


namespace age_of_youngest_person_l1601_160117

theorem age_of_youngest_person :
  ∃ (a1 a2 a3 a4 : ℕ), 
  (a1 < a2) ∧ (a2 < a3) ∧ (a3 < a4) ∧ 
  (a4 = 50) ∧ 
  (a1 + a2 + a3 + a4 = 158) ∧ 
  (a2 - a1 = a3 - a2) ∧ (a3 - a2 = a4 - a3) ∧ 
  a1 = 29 :=
by
  sorry

end age_of_youngest_person_l1601_160117


namespace hayley_friends_l1601_160111

theorem hayley_friends (total_stickers : ℕ) (stickers_per_friend : ℕ) (h1 : total_stickers = 72) (h2 : stickers_per_friend = 8) : (total_stickers / stickers_per_friend) = 9 :=
by
  sorry

end hayley_friends_l1601_160111


namespace distance_per_trip_l1601_160130

--  Define the conditions as assumptions
variables (total_distance : ℝ) (num_trips : ℝ)
axiom h_total_distance : total_distance = 120
axiom h_num_trips : num_trips = 4

-- Define the question converted into a statement to be proven
theorem distance_per_trip : total_distance / num_trips = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end distance_per_trip_l1601_160130


namespace initial_HNO3_percentage_is_correct_l1601_160164

def initial_percentage_of_HNO3 (P : ℚ) : Prop :=
  let initial_volume := 60
  let added_volume := 18
  let final_volume := 78
  let final_percentage := 50
  (P / 100) * initial_volume + added_volume = (final_percentage / 100) * final_volume

theorem initial_HNO3_percentage_is_correct :
  initial_percentage_of_HNO3 35 :=
by
  sorry

end initial_HNO3_percentage_is_correct_l1601_160164


namespace minimum_value_h_at_a_eq_2_range_of_a_l1601_160149

noncomputable def f (a x : ℝ) : ℝ := a * x + (a - 1) / x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a x : ℝ) : ℝ := f a x - g x

theorem minimum_value_h_at_a_eq_2 : ∃ x, h 2 x = 3 := 
sorry

theorem range_of_a (a : ℝ) : (∀ x ≥ 1, h a x ≥ 1) ↔ a ≥ 1 :=
sorry

end minimum_value_h_at_a_eq_2_range_of_a_l1601_160149


namespace total_population_of_towns_l1601_160141

theorem total_population_of_towns :
  let num_towns := 25
  let avg_pop_min := 3600
  let avg_pop_max := 4000
  let estimated_avg_pop := (avg_pop_min + avg_pop_max) / 2
  num_towns * estimated_avg_pop = 95000 :=
by
  let num_towns := 25
  let avg_pop_min := 3600
  let avg_pop_max := 4000
  let estimated_avg_pop := (avg_pop_min + avg_pop_max) / 2
  show num_towns * estimated_avg_pop = 95000
  sorry

end total_population_of_towns_l1601_160141


namespace student_marks_l1601_160166

theorem student_marks (max_marks : ℕ) (passing_percentage : ℕ) (failed_by : ℕ) (student_marks : ℕ) : 
  (passing_percentage = 30) → (failed_by = 40) → (max_marks = 400) → 
  student_marks = (max_marks * passing_percentage / 100 - failed_by) → 
  student_marks = 80 :=
by {
  sorry
}

end student_marks_l1601_160166


namespace correct_statements_l1601_160158

namespace ProofProblem

def P1 : Prop := (-4) + (-5) = -9
def P2 : Prop := -5 - (-6) = 11
def P3 : Prop := -2 * (-10) = -20
def P4 : Prop := 4 / (-2) = -2

theorem correct_statements : P1 ∧ P4 ∧ ¬P2 ∧ ¬P3 := by
  -- proof to be filled in later
  sorry

end ProofProblem

end correct_statements_l1601_160158


namespace jack_birth_year_l1601_160198

theorem jack_birth_year 
  (first_amc8_year : ℕ) 
  (amc8_annual : ℕ → ℕ → ℕ) 
  (jack_age_ninth_amc8 : ℕ) 
  (ninth_amc8_year : amc8_annual first_amc8_year 9 = 1998) 
  (jack_age_in_ninth_amc8 : jack_age_ninth_amc8 = 15)
  : (1998 - jack_age_ninth_amc8 = 1983) := by
  sorry

end jack_birth_year_l1601_160198


namespace ratio_xyz_l1601_160168

theorem ratio_xyz (a x y z : ℝ) : 
  5 * x + 4 * y - 6 * z = a ∧
  4 * x - 5 * y + 7 * z = 27 * a ∧
  6 * x + 5 * y - 4 * z = 18 * a →
  (x :ℝ) / (y :ℝ) = 3 / 4 ∧
  (y :ℝ) / (z :ℝ) = 4 / 5 :=
by
  sorry

end ratio_xyz_l1601_160168


namespace line_through_two_points_l1601_160156

theorem line_through_two_points (x y : ℝ) (hA : (x, y) = (3, 0)) (hB : (x, y) = (0, 2)) :
  2 * x + 3 * y - 6 = 0 :=
sorry 

end line_through_two_points_l1601_160156


namespace tom_paths_avoiding_construction_l1601_160137

def tom_home : (ℕ × ℕ) := (0, 0)
def friend_home : (ℕ × ℕ) := (4, 3)
def construction_site : (ℕ × ℕ) := (2, 2)

def total_paths_without_restriction : ℕ := Nat.choose 7 4
def paths_via_construction_site : ℕ := (Nat.choose 4 2) * (Nat.choose 3 1)
def valid_paths : ℕ := total_paths_without_restriction - paths_via_construction_site

theorem tom_paths_avoiding_construction : valid_paths = 17 := by
  sorry

end tom_paths_avoiding_construction_l1601_160137


namespace problem_solved_l1601_160176

-- Define the function f with the given conditions
def satisfies_conditions(f : ℝ × ℝ × ℝ → ℝ) :=
  (∀ x y z t : ℝ, f (x + t, y + t, z + t) = t + f (x, y, z)) ∧
  (∀ x y z t : ℝ, f (t * x, t * y, t * z) = t * f (x, y, z)) ∧
  (∀ x y z : ℝ, f (x, y, z) = f (y, x, z)) ∧
  (∀ x y z : ℝ, f (x, y, z) = f (x, z, y))

-- We'll state the main result to be proven, without giving the proof
theorem problem_solved (f : ℝ × ℝ × ℝ → ℝ) (h : satisfies_conditions f) : f (2000, 2001, 2002) = 2001 :=
  sorry

end problem_solved_l1601_160176


namespace decompose_zero_l1601_160160

theorem decompose_zero (a : ℤ) : 0 = 0 * a := by
  sorry

end decompose_zero_l1601_160160


namespace cost_of_one_stamp_l1601_160159

-- Defining the conditions
def cost_of_four_stamps := 136
def number_of_stamps := 4

-- Prove that if 4 stamps cost 136 cents, then one stamp costs 34 cents
theorem cost_of_one_stamp : cost_of_four_stamps / number_of_stamps = 34 :=
by
  sorry

end cost_of_one_stamp_l1601_160159


namespace sum_of_inserted_numbers_eq_12_l1601_160161

theorem sum_of_inserted_numbers_eq_12 (a b : ℝ) (r d : ℝ) 
  (h1 : a = 2 * r) 
  (h2 : b = 2 * r^2) 
  (h3 : b = a + d) 
  (h4 : 12 = b + d) : 
  a + b = 12 :=
by
  sorry

end sum_of_inserted_numbers_eq_12_l1601_160161


namespace upper_limit_of_raise_l1601_160155

theorem upper_limit_of_raise (lower upper : ℝ) (h_lower : lower = 0.05)
  (h_upper : upper > 0.08) (h_inequality : ∀ r, lower < r → r < upper)
  : upper < 0.09 :=
sorry

end upper_limit_of_raise_l1601_160155


namespace micah_water_intake_l1601_160191

def morning : ℝ := 1.5
def early_afternoon : ℝ := 2 * morning
def late_afternoon : ℝ := 3 * morning
def evening : ℝ := late_afternoon - 0.25 * late_afternoon
def night : ℝ := 2 * evening
def total_water_intake : ℝ := morning + early_afternoon + late_afternoon + evening + night

theorem micah_water_intake :
  total_water_intake = 19.125 := by
  sorry

end micah_water_intake_l1601_160191


namespace sum_zero_of_cubic_identity_l1601_160113

theorem sum_zero_of_cubic_identity (a b c : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) (h4 : a^3 + b^3 + c^3 = 3 * a * b * c) : 
  a + b + c = 0 :=
by
  sorry

end sum_zero_of_cubic_identity_l1601_160113


namespace bill_profit_difference_l1601_160147

theorem bill_profit_difference 
  (SP : ℝ) 
  (hSP : SP = 1.10 * (SP / 1.10)) 
  (hSP_val : SP = 989.9999999999992) 
  (NP : ℝ) 
  (hNP : NP = 0.90 * (SP / 1.10)) 
  (NSP : ℝ) 
  (hNSP : NSP = 1.30 * NP) 
  : NSP - SP = 63.0000000000008 := 
by 
  sorry

end bill_profit_difference_l1601_160147


namespace mass_percentage_of_Ca_in_CaO_is_correct_l1601_160180

noncomputable def molarMass_Ca : ℝ := 40.08
noncomputable def molarMass_O : ℝ := 16.00
noncomputable def molarMass_CaO : ℝ := molarMass_Ca + molarMass_O
noncomputable def massPercentageCaInCaO : ℝ := (molarMass_Ca / molarMass_CaO) * 100

theorem mass_percentage_of_Ca_in_CaO_is_correct :
  massPercentageCaInCaO = 71.47 :=
by
  -- This is where the proof would go
  sorry

end mass_percentage_of_Ca_in_CaO_is_correct_l1601_160180


namespace sixth_graders_more_than_seventh_l1601_160186

def pencil_cost : ℕ := 13
def eighth_graders_total : ℕ := 208
def seventh_graders_total : ℕ := 181
def sixth_graders_total : ℕ := 234

-- Number of students in each grade who bought a pencil
def seventh_graders_count := seventh_graders_total / pencil_cost
def sixth_graders_count := sixth_graders_total / pencil_cost

-- The difference in the number of sixth graders than seventh graders who bought a pencil
theorem sixth_graders_more_than_seventh : sixth_graders_count - seventh_graders_count = 4 :=
by sorry

end sixth_graders_more_than_seventh_l1601_160186


namespace find_m_l1601_160150

noncomputable def curve (x : ℝ) : ℝ := (1 / 4) * x^2
noncomputable def line (x : ℝ) : ℝ := 1 - 2 * x

theorem find_m (m n : ℝ) (h_curve : curve m = n) (h_perpendicular : (1 / 2) * m * (-2) = -1) : m = 1 := 
  sorry

end find_m_l1601_160150


namespace problem_inequality_l1601_160131

theorem problem_inequality (a b c m n p : ℝ) (h1 : a + b + c = 1) (h2 : m + n + p = 1) :
  -1 ≤ a * m + b * n + c * p ∧ a * m + b * n + c * p ≤ 1 := by
  sorry

end problem_inequality_l1601_160131


namespace p_sufficient_not_necessary_for_q_l1601_160122

-- Define the conditions p and q
def p (x : ℝ) := x^2 < 5 * x - 6
def q (x : ℝ) := |x + 1| ≤ 4

-- The goal to prove
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬ (∀ x, q x → p x) :=
by 
  sorry

end p_sufficient_not_necessary_for_q_l1601_160122


namespace combined_weight_l1601_160145

-- Definition of conditions
def regular_dinosaur_weight := 800
def five_regular_dinosaurs_weight := 5 * regular_dinosaur_weight
def barney_weight := five_regular_dinosaurs_weight + 1500

-- Statement to prove
theorem combined_weight (h1: five_regular_dinosaurs_weight = 5 * regular_dinosaur_weight)
                        (h2: barney_weight = five_regular_dinosaurs_weight + 1500) : 
        (barney_weight + five_regular_dinosaurs_weight = 9500) :=
by
    sorry

end combined_weight_l1601_160145


namespace minimum_a_l1601_160195

theorem minimum_a (a : ℝ) : (∀ x y : ℝ, 0 < x → 0 < y → (x + y) * (a / x + 4 / y) ≥ 16) → a ≥ 4 :=
by
  intros h
  -- We would provide a detailed mathematical proof here, but we use sorry for now.
  sorry

end minimum_a_l1601_160195


namespace sum_abcd_value_l1601_160182

theorem sum_abcd_value (a b c d : ℚ) :
  (2 * a + 3 = 2 * b + 5) ∧ 
  (2 * b + 5 = 2 * c + 7) ∧ 
  (2 * c + 7 = 2 * d + 9) ∧ 
  (2 * d + 9 = 2 * (a + b + c + d) + 13) → 
  a + b + c + d = -14 / 3 := 
by
  sorry

end sum_abcd_value_l1601_160182


namespace jersey_cost_difference_l1601_160189

theorem jersey_cost_difference :
  let jersey_cost := 115
  let tshirt_cost := 25
  jersey_cost - tshirt_cost = 90 :=
by
  -- proof goes here
  sorry

end jersey_cost_difference_l1601_160189


namespace train_length_l1601_160105

theorem train_length (L S : ℝ) 
  (h1 : L = S * 40) 
  (h2 : L + 1800 = S * 120) : 
  L = 900 := 
by
  sorry

end train_length_l1601_160105


namespace one_thirds_in_nine_halves_l1601_160167

theorem one_thirds_in_nine_halves : (9 / 2) / (1 / 3) = 13 := by
  sorry

end one_thirds_in_nine_halves_l1601_160167


namespace tomatoes_picked_l1601_160172

theorem tomatoes_picked (original_tomatoes left_tomatoes picked_tomatoes : ℕ)
  (h1 : original_tomatoes = 97)
  (h2 : left_tomatoes = 14)
  (h3 : picked_tomatoes = original_tomatoes - left_tomatoes) :
  picked_tomatoes = 83 :=
by sorry

end tomatoes_picked_l1601_160172


namespace correct_mark_l1601_160128

theorem correct_mark 
  (avg_wrong : ℝ := 60)
  (wrong_mark : ℝ := 90)
  (num_students : ℕ := 30)
  (avg_correct : ℝ := 57.5) :
  (wrong_mark - (avg_wrong * num_students - avg_correct * num_students)) = 15 :=
by
  sorry

end correct_mark_l1601_160128


namespace eq_of_operation_l1601_160152

theorem eq_of_operation {x : ℝ} (h : 60 + 5 * 12 / (x / 3) = 61) : x = 180 :=
by
  sorry

end eq_of_operation_l1601_160152


namespace factorable_quadratic_l1601_160196

theorem factorable_quadratic (b : Int) : 
  (∃ m n p q : Int, 35 * m * p = 35 ∧ m * q + n * p = b ∧ n * q = 35) ↔ (∃ k : Int, b = 2 * k) :=
sorry

end factorable_quadratic_l1601_160196


namespace complex_quadrant_l1601_160103

theorem complex_quadrant (z : ℂ) (h : (2 - I) * z = 1 + I) : 
  0 < z.re ∧ 0 < z.im := 
by 
  -- Proof will be provided here 
  sorry

end complex_quadrant_l1601_160103


namespace domain_of_f_l1601_160115

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x + 5)

theorem domain_of_f :
  { x : ℝ | f x ≠ 0 } = { x : ℝ | x ≠ -5 }
:= sorry

end domain_of_f_l1601_160115


namespace roots_quadratic_l1601_160178

theorem roots_quadratic (d e : ℝ) (h1 : 3 * d ^ 2 + 5 * d - 2 = 0) (h2 : 3 * e ^ 2 + 5 * e - 2 = 0) :
  (d - 1) * (e - 1) = 2 :=
sorry

end roots_quadratic_l1601_160178


namespace white_ducks_count_l1601_160175

theorem white_ducks_count (W : ℕ) : 
  (5 * W + 10 * 7 + 12 * 6 = 157) → W = 3 :=
by
  sorry

end white_ducks_count_l1601_160175


namespace tax_on_other_items_l1601_160118

theorem tax_on_other_items (total_amount clothing_amount food_amount other_items_amount tax_on_clothing tax_on_food total_tax : ℝ) (tax_percent_other : ℝ) 
(h1 : clothing_amount = 0.5 * total_amount)
(h2 : food_amount = 0.2 * total_amount)
(h3 : other_items_amount = 0.3 * total_amount)
(h4 : tax_on_clothing = 0.04 * clothing_amount)
(h5 : tax_on_food = 0) 
(h6 : total_tax = 0.044 * total_amount)
: 
(tax_percent_other = 8) := 
by
  -- Definitions from the problem
  -- Define the total tax paid as the sum of taxes on clothing, food, and other items
  let tax_other_items : ℝ := tax_percent_other / 100 * other_items_amount
  
  -- Total tax equation
  have h7 : tax_on_clothing + tax_on_food + tax_other_items = total_tax
  sorry

  -- Substitution values into the given conditions and solving
  have h8 : tax_on_clothing + tax_percent_other / 100 * other_items_amount = total_tax
  sorry
  
  have h9 : 0.04 * 0.5 * total_amount + tax_percent_other / 100 * 0.3 * total_amount = 0.044 * total_amount
  sorry

  have h10 : 0.02 * total_amount + tax_percent_other / 100 * 0.3 * total_amount = 0.044 * total_amount
  sorry

  have h11 : tax_percent_other / 100 * 0.3 * total_amount = 0.024 * total_amount
  sorry

  have h12 : tax_percent_other / 100 * 0.3 = 0.024
  sorry

  have h13 : tax_percent_other / 100 = 0.08
  sorry

  have h14 : tax_percent_other = 8
  sorry

  exact h14

end tax_on_other_items_l1601_160118


namespace range_of_a_l1601_160102

theorem range_of_a (x a : ℝ) (p : 0 < x ∧ x < 1)
  (q : (x - a) * (x - (a + 2)) ≤ 0) (h : ∀ x, (0 < x ∧ x < 1) → (x - a) * (x - (a + 2)) ≤ 0) :
  -1 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l1601_160102


namespace distance_to_origin_eq_three_l1601_160121

theorem distance_to_origin_eq_three :
  let P := (1, 2, 2)
  let origin := (0, 0, 0)
  dist P origin = 3 := by
  sorry

end distance_to_origin_eq_three_l1601_160121


namespace relay_race_solution_l1601_160101

variable (Sadie_time : ℝ) (Sadie_speed : ℝ)
variable (Ariana_time : ℝ) (Ariana_speed : ℝ)
variable (Sarah_speed : ℝ)
variable (total_distance : ℝ)

def relay_race_time : Prop :=
  let Sadie_distance := Sadie_time * Sadie_speed
  let Ariana_distance := Ariana_time * Ariana_speed
  let Sarah_distance := total_distance - Sadie_distance - Ariana_distance
  let Sarah_time := Sarah_distance / Sarah_speed
  Sadie_time + Ariana_time + Sarah_time = 4.5

theorem relay_race_solution (h1: Sadie_time = 2) (h2: Sadie_speed = 3)
  (h3: Ariana_time = 0.5) (h4: Ariana_speed = 6)
  (h5: Sarah_speed = 4) (h6: total_distance = 17) :
  relay_race_time Sadie_time Sadie_speed Ariana_time Ariana_speed Sarah_speed total_distance :=
by
  sorry

end relay_race_solution_l1601_160101


namespace inequality_real_equation_positive_integers_solution_l1601_160127

-- Prove the inequality for real numbers a and b
theorem inequality_real (a b : ℝ) :
  (a^2 + 1) * (b^2 + 1) + 50 ≥ 2 * ((2 * a + 1) * (3 * b + 1)) :=
  sorry

-- Find all positive integers n and p such that the equation holds
theorem equation_positive_integers_solution :
  ∃ (n p : ℕ), 0 < n ∧ 0 < p ∧ (n^2 + 1) * (p^2 + 1) + 45 = 2 * ((2 * n + 1) * (3 * p + 1)) ∧ n = 2 ∧ p = 2 :=
  sorry

end inequality_real_equation_positive_integers_solution_l1601_160127


namespace garden_dimensions_l1601_160142

theorem garden_dimensions (w l : ℕ) (h₁ : l = w + 3) (h₂ : 2 * (l + w) = 26) : w = 5 ∧ l = 8 :=
by
  sorry

end garden_dimensions_l1601_160142


namespace highway_length_l1601_160136

theorem highway_length 
  (speed_car1 speed_car2 : ℕ) (time : ℕ)
  (h_speed_car1 : speed_car1 = 54)
  (h_speed_car2 : speed_car2 = 57)
  (h_time : time = 3) : 
  speed_car1 * time + speed_car2 * time = 333 := by
  sorry

end highway_length_l1601_160136


namespace select_monkey_l1601_160110

theorem select_monkey (consumption : ℕ → ℕ) (n bananas minutes : ℕ)
  (h1 : consumption 1 = 1) (h2 : consumption 2 = 2) (h3 : consumption 3 = 3)
  (h4 : consumption 4 = 4) (h5 : consumption 5 = 5) (h6 : consumption 6 = 6)
  (h_total_minutes : minutes = 18) (h_total_bananas : bananas = 18) :
  consumption 1 * minutes = bananas :=
by
  sorry

end select_monkey_l1601_160110


namespace quadratic_has_single_solution_l1601_160100

theorem quadratic_has_single_solution (k : ℚ) : 
  (∀ x : ℚ, 3 * x^2 - 7 * x + k = 0 → x = 7 / 6) ↔ k = 49 / 12 := 
by
  sorry

end quadratic_has_single_solution_l1601_160100


namespace atomic_weight_of_chlorine_l1601_160129

theorem atomic_weight_of_chlorine (molecular_weight_AlCl3 : ℝ) (atomic_weight_Al : ℝ) (atomic_weight_Cl : ℝ) :
  molecular_weight_AlCl3 = 132 ∧ atomic_weight_Al = 26.98 →
  132 = 26.98 + 3 * atomic_weight_Cl →
  atomic_weight_Cl = 35.007 :=
by
  intros h1 h2
  sorry

end atomic_weight_of_chlorine_l1601_160129


namespace parallel_lines_slope_eq_l1601_160126

theorem parallel_lines_slope_eq (m : ℝ) :
  (∀ x y : ℝ, 3 * x + 4 * y - 3 = 0 ↔ 6 * x + m * y + 11 = 0) → m = 8 :=
by
  sorry

end parallel_lines_slope_eq_l1601_160126


namespace value_of_star_l1601_160120

theorem value_of_star (a b : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : (a + b) % 4 = 0) : a^2 + 2*a*b + b^2 = 64 :=
by
  sorry

end value_of_star_l1601_160120


namespace unique_x1_sequence_l1601_160192

open Nat

theorem unique_x1_sequence (x1 : ℝ) (x : ℕ → ℝ)
  (h₀ : x 1 = x1)
  (h₁ : ∀ n, x (n + 1) = x n * (x n + 1 / (n + 1))) :
  (∃! x1, (0 < x1 ∧ x1 < 1) ∧ 
   (∀ n, 0 < x n ∧ x n < x (n + 1) ∧ x (n + 1) < 1)) := sorry

end unique_x1_sequence_l1601_160192


namespace inequality_bound_l1601_160187

theorem inequality_bound (a b c d e p q : ℝ) (hpq : 0 < p ∧ p ≤ q)
  (ha : p ≤ a ∧ a ≤ q) (hb : p ≤ b ∧ b ≤ q) (hc : p ≤ c ∧ c ≤ q) 
  (hd : p ≤ d ∧ d ≤ q) (he : p ≤ e ∧ e ≤ q) :
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) 
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p))^2 :=
sorry

end inequality_bound_l1601_160187


namespace julia_mile_time_l1601_160197

variable (x : ℝ)

theorem julia_mile_time
  (h1 : ∀ x, x > 0)
  (h2 : ∀ x, x <= 13)
  (h3 : 65 = 5 * 13)
  (h4 : 50 = 65 - 15)
  (h5 : 50 = 5 * x) :
  x = 10 := by
  sorry

end julia_mile_time_l1601_160197


namespace parallel_planes_sufficient_not_necessary_for_perpendicular_lines_l1601_160184

variables {Point Line Plane : Type}
variables (α β : Plane) (ℓ m : Line) (point_on_line_ℓ : Point) (point_on_line_m : Point)

-- Definitions of conditions
def line_perpendicular_to_plane (ℓ : Line) (α : Plane) : Prop := sorry
def line_contained_in_plane (m : Line) (β : Plane) : Prop := sorry
def planes_parallel (α β : Plane) : Prop := sorry
def line_perpendicular_to_line (ℓ m : Line) : Prop := sorry

axiom h1 : line_perpendicular_to_plane ℓ α
axiom h2 : line_contained_in_plane m β

-- Statement of the proof problem
theorem parallel_planes_sufficient_not_necessary_for_perpendicular_lines : 
  (planes_parallel α β → line_perpendicular_to_line ℓ m) ∧ 
  ¬ (line_perpendicular_to_line ℓ m → planes_parallel α β) :=
  sorry

end parallel_planes_sufficient_not_necessary_for_perpendicular_lines_l1601_160184


namespace third_offense_fraction_l1601_160157

-- Define the conditions
def sentence_assault : ℕ := 3
def sentence_poisoning : ℕ := 24
def total_sentence : ℕ := 36

-- The main theorem to prove
theorem third_offense_fraction :
  (total_sentence - (sentence_assault + sentence_poisoning)) / (sentence_assault + sentence_poisoning) = 1 / 3 := by
  sorry

end third_offense_fraction_l1601_160157


namespace calvin_total_insects_l1601_160169

def R : ℕ := 15
def S : ℕ := 2 * R - 8
def C : ℕ := 11 -- rounded from (1/2) * R + 3
def P : ℕ := 3 * S + 7
def B : ℕ := 4 * C - 2
def E : ℕ := 3 * (R + S + C + P + B)
def total_insects : ℕ := R + S + C + P + B + E

theorem calvin_total_insects : total_insects = 652 :=
by
  -- service the proof here.
  sorry

end calvin_total_insects_l1601_160169


namespace find_x_values_l1601_160112

theorem find_x_values (
  x : ℝ
) (h₁ : x ≠ 0) (h₂ : x ≠ 1) (h₃ : x ≠ 2) :
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 4) ↔ 
  (x < (1 - Real.sqrt 17) / 2 ∨ (0 < x ∧ x < 1) ∨ (2 < x ∧ x < (1 + Real.sqrt 17) / 2)) :=
by
  sorry

end find_x_values_l1601_160112


namespace remaining_sum_eq_seven_eighths_l1601_160185

noncomputable def sum_series := 
  (1 / 2) + (1 / 4) + (1 / 8) + (1 / 16) + (1 / 32) + (1 / 64)

noncomputable def removed_terms := 
  (1 / 16) + (1 / 32) + (1 / 64)

theorem remaining_sum_eq_seven_eighths : 
  sum_series - removed_terms = 7 / 8 := by
  sorry

end remaining_sum_eq_seven_eighths_l1601_160185


namespace solution_set_inequality_l1601_160125

theorem solution_set_inequality (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  {x : ℝ | (x - a) * (x - (1 / a)) < 0} = {x : ℝ | a < x ∧ x < 1 / a} := sorry

end solution_set_inequality_l1601_160125


namespace rhombus_area_600_l1601_160124

noncomputable def area_of_rhombus (x y : ℝ) : ℝ := (x * y) * 2

theorem rhombus_area_600 (x y : ℝ) (qx qy : ℝ)
  (hx : x = 15) (hy : y = 20)
  (hr1 : qx = 15) (hr2 : qy = 20)
  (h_ratio : qy / qx = 4 / 3) :
  area_of_rhombus (2 * (x + y - 2)) (x + y) = 600 :=
by
  rw [hx, hy]
  sorry

end rhombus_area_600_l1601_160124


namespace train_crossing_pole_time_l1601_160138

theorem train_crossing_pole_time :
  ∀ (speed_kmph length_m: ℝ), speed_kmph = 160 → length_m = 400.032 → 
  length_m / (speed_kmph * 1000 / 3600) = 9.00072 :=
by
  intros speed_kmph length_m h_speed h_length
  rw [h_speed, h_length]
  -- The proof is omitted as per instructions
  sorry

end train_crossing_pole_time_l1601_160138


namespace find_phi_increasing_intervals_l1601_160183

open Real

-- Defining the symmetry condition
noncomputable def symmetric_phi (x_sym : ℝ) (k : ℤ) (phi : ℝ): Prop :=
  2 * x_sym + phi = k * π + π / 2

-- Finding the value of phi given the conditions
theorem find_phi (x_sym : ℝ) (phi : ℝ) (k : ℤ) 
  (h_sym: symmetric_phi x_sym k phi) (h_phi_bound : -π < phi ∧ phi < 0)
  (h_xsym: x_sym = π / 8) :
  phi = -3 * π / 4 :=
by
  sorry

-- Defining the function and its increasing intervals
noncomputable def f (x : ℝ) (phi : ℝ) : ℝ := sin (2 * x + phi)

-- Finding the increasing intervals of f on the interval [0, π]
theorem increasing_intervals (phi : ℝ) 
  (h_phi: phi = -3 * π / 4) :
  ∀ x, (0 ≤ x ∧ x ≤ π) → 
    (π / 8 ≤ x ∧ x ≤ 5 * π / 8) :=
by
  sorry

end find_phi_increasing_intervals_l1601_160183


namespace sum_of_first_15_terms_l1601_160123

theorem sum_of_first_15_terms (a : ℕ → ℝ) (r : ℝ)
    (h_geom : ∀ n, a (n + 1) = a n * r)
    (h1 : a 1 + a 2 + a 3 = 1)
    (h2 : a 4 + a 5 + a 6 = -2) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 +
   a 10 + a 11 + a 12 + a 13 + a 14 + a 15) = 11 :=
sorry

end sum_of_first_15_terms_l1601_160123


namespace points_earned_l1601_160143

-- Definition of the conditions explicitly stated in the problem
def points_per_bag := 8
def total_bags := 4
def bags_not_recycled := 2

-- Calculation of bags recycled
def bags_recycled := total_bags - bags_not_recycled

-- The main theorem stating the proof equivalent
theorem points_earned : points_per_bag * bags_recycled = 16 := 
by
  sorry

end points_earned_l1601_160143


namespace simon_stamps_received_l1601_160173

theorem simon_stamps_received (initial_stamps total_stamps received_stamps : ℕ) (h1 : initial_stamps = 34) (h2 : total_stamps = 61) : received_stamps = 27 :=
by
  sorry

end simon_stamps_received_l1601_160173


namespace youngest_child_age_l1601_160179

theorem youngest_child_age (total_bill mother_cost twin_age_cost total_age : ℕ) (twin_age youngest_age : ℕ) 
  (h1 : total_bill = 1485) (h2 : mother_cost = 695) (h3 : twin_age_cost = 65) 
  (h4 : total_age = (total_bill - mother_cost) / twin_age_cost)
  (h5 : total_age = 2 * twin_age + youngest_age) :
  youngest_age = 2 :=
by
  -- sorry: Proof to be completed later
  sorry

end youngest_child_age_l1601_160179


namespace smallest_rel_prime_l1601_160104

theorem smallest_rel_prime (n : ℕ) (h : n > 1) (rel_prime : ∀ p ∈ [2, 3, 5, 7], ¬ p ∣ n) : n = 11 :=
by sorry

end smallest_rel_prime_l1601_160104


namespace max_int_solution_of_inequality_system_l1601_160133

theorem max_int_solution_of_inequality_system :
  ∃ (x : ℤ), (∀ (y : ℤ), (3 * y - 1 < y + 1) ∧ (2 * (2 * y - 1) ≤ 5 * y + 1) → y ≤ x) ∧
             (3 * x - 1 < x + 1) ∧ (2 * (2 * x - 1) ≤ 5 * x + 1) ∧
             x = 0 :=
by
  sorry

end max_int_solution_of_inequality_system_l1601_160133


namespace add_number_l1601_160177

theorem add_number (x : ℕ) (h : 43 + x = 81) : x + 25 = 63 :=
by {
  -- Since this is focusing on the structure and statement no proof steps are required
  sorry
}

end add_number_l1601_160177


namespace max_sum_of_lengths_l1601_160114

def length_of_integer (k : ℤ) (hk : k > 1) : ℤ := sorry

theorem max_sum_of_lengths (x y : ℤ) (hx : x > 1) (hy : y > 1) (h : x + 3 * y < 920) :
  length_of_integer x hx + length_of_integer y hy = 15 :=
sorry

end max_sum_of_lengths_l1601_160114


namespace angle_CDB_45_degrees_l1601_160132

theorem angle_CDB_45_degrees
  (α β γ δ : ℝ)
  (triangle_isosceles_right : α = β)
  (triangle_angle_BCD : γ = 90)
  (square_angle_DCE : δ = 90)
  (triangle_angle_ABC : α = β)
  (isosceles_triangle_angle : α + β + γ = 180)
  (isosceles_triangle_right : α = 45)
  (isosceles_triangle_sum : α + α + 90 = 180)
  (square_geometry : δ = 90) :
  γ + δ = 180 →  180 - (γ + α) = 45 :=
by
  sorry

end angle_CDB_45_degrees_l1601_160132


namespace determine_a_l1601_160134

theorem determine_a (a : ℝ) (x1 x2 : ℝ) :
  (x1 * x1 + (2 * a - 1) * x1 + a * a = 0) ∧
  (x2 * x2 + (2 * a - 1) * x2 + a * a = 0) ∧
  ((x1 + 2) * (x2 + 2) = 11) →
  a = -1 :=
by
  sorry

end determine_a_l1601_160134


namespace negation_example_l1601_160140

theorem negation_example :
  ¬ (∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 1 < 0 :=
sorry

end negation_example_l1601_160140


namespace joe_fish_times_sam_l1601_160181

-- Define the number of fish Sam has
def sam_fish : ℕ := 7

-- Define the number of fish Harry has
def harry_fish : ℕ := 224

-- Define the number of times Joe has as many fish as Sam
def joe_times_sam (x : ℕ) : Prop :=
  4 * (sam_fish * x) = harry_fish

-- The theorem to prove Joe has 8 times as many fish as Sam
theorem joe_fish_times_sam : ∃ x, joe_times_sam x ∧ x = 8 :=
by
  sorry

end joe_fish_times_sam_l1601_160181


namespace MrFletcher_paid_l1601_160135

noncomputable def total_payment (hours_day1 hours_day2 hours_day3 rate_per_hour men : ℕ) : ℕ :=
  let total_hours := hours_day1 + hours_day2 + hours_day3
  let total_man_hours := total_hours * men
  total_man_hours * rate_per_hour

theorem MrFletcher_paid
  (hours_day1 hours_day2 hours_day3 : ℕ)
  (rate_per_hour men : ℕ)
  (h1 : hours_day1 = 10)
  (h2 : hours_day2 = 8)
  (h3 : hours_day3 = 15)
  (h4 : rate_per_hour = 10)
  (h5 : men = 2) :
  total_payment hours_day1 hours_day2 hours_day3 rate_per_hour men = 660 := 
by {
  -- skipped proof details
  sorry
}

end MrFletcher_paid_l1601_160135


namespace jennifer_fruits_left_l1601_160116

theorem jennifer_fruits_left:
  (apples = 2 * pears) →
  (cherries = oranges / 2) →
  (grapes = 3 * apples) →
  pears = 15 →
  oranges = 30 →
  pears_given = 3 →
  oranges_given = 5 →
  apples_given = 5 →
  cherries_given = 7 →
  grapes_given = 3 →
  (remaining_fruits =
    (pears - pears_given) +
    (oranges - oranges_given) +
    (apples - apples_given) +
    (cherries - cherries_given) +
    (grapes - grapes_given)) →
  remaining_fruits = 157 :=
by
  intros
  sorry

end jennifer_fruits_left_l1601_160116


namespace Kelly_initial_games_l1601_160119

-- Condition definitions
variable (give_away : ℕ) (left_over : ℕ)
variable (initial_games : ℕ)

-- Given conditions
axiom h1 : give_away = 15
axiom h2 : left_over = 35

-- Proof statement
theorem Kelly_initial_games : initial_games = give_away + left_over :=
sorry

end Kelly_initial_games_l1601_160119


namespace max_ab_at_extremum_l1601_160193

noncomputable def f (a b x : ℝ) : ℝ := 4*x^3 - a*x^2 - 2*b*x + 2

theorem max_ab_at_extremum (a b : ℝ) (h0: a > 0) (h1 : b > 0) (h2 : ∃ x, f a b x = 4*x^3 - a*x^2 - 2*b*x + 2 ∧ x = 1 ∧ 12*x^2 - 2*a*x - 2*b = 0) :
  ab ≤ 9 := 
sorry  -- proof not required

end max_ab_at_extremum_l1601_160193


namespace total_hours_verification_l1601_160165

def total_hours_data_analytics : ℕ := 
  let weekly_class_homework_hours := (2 * 3 + 1 * 4 + 4) * 24 
  let lab_project_hours := 8 * 6 + (10 + 14 + 18)
  weekly_class_homework_hours + lab_project_hours

def total_hours_programming : ℕ :=
  let weekly_hours := (2 * 2 + 2 * 4 + 6) * 24
  weekly_hours

def total_hours_statistics : ℕ :=
  let weekly_class_lab_project_hours := (2 * 3 + 1 * 2 + 3) * 24
  let exam_study_hours := 9 * 5
  weekly_class_lab_project_hours + exam_study_hours

def total_hours_all_courses : ℕ :=
  total_hours_data_analytics + total_hours_programming + total_hours_statistics

theorem total_hours_verification : 
    total_hours_all_courses = 1167 := 
by 
    sorry

end total_hours_verification_l1601_160165


namespace sum_digits_2_2005_times_5_2007_times_3_l1601_160144

-- Define a function to calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem sum_digits_2_2005_times_5_2007_times_3 : 
  sum_of_digits (2^2005 * 5^2007 * 3) = 12 := 
by 
  sorry

end sum_digits_2_2005_times_5_2007_times_3_l1601_160144


namespace eleven_percent_greater_than_seventy_l1601_160108

theorem eleven_percent_greater_than_seventy : ∀ x : ℝ, (x = 70 * (1 + 11 / 100)) → (x = 77.7) :=
by
  intro x
  intro h
  sorry

end eleven_percent_greater_than_seventy_l1601_160108


namespace perpendicular_case_parallel_case_l1601_160199

variable (a b : ℝ)

-- Define the lines
def line1 (a b x y : ℝ) := a * x - b * y + 4 = 0
def line2 (a b x y : ℝ) := (a - 1) * x + y + b = 0

-- Define perpendicular condition
def perpendicular (a b : ℝ) := a * (a - 1) - b = 0

-- Define point condition
def passes_through (a b : ℝ) := -3 * a + b + 4 = 0

-- Define parallel condition
def parallel (a b : ℝ) := a * (a - 1) + b = 0

-- Define intercepts equal condition
def intercepts_equal (a b : ℝ) := b = -a

theorem perpendicular_case
    (h1 : perpendicular a b)
    (h2 : passes_through a b) :
    a = 2 ∧ b = 2 :=
sorry

theorem parallel_case
    (h1 : parallel a b)
    (h2 : intercepts_equal a b) :
    a = 2 ∧ b = -2 :=
sorry

end perpendicular_case_parallel_case_l1601_160199


namespace pages_left_l1601_160188

theorem pages_left (total_pages read_fraction : ℕ) (h_total_pages : total_pages = 396) (h_read_fraction : read_fraction = 1/3) : total_pages * (1 - read_fraction) = 264 := 
by
  sorry

end pages_left_l1601_160188


namespace f_zero_f_odd_solve_inequality_l1601_160139

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : f (x + y) = f x + f y
axiom increasing_on_nonneg : ∀ {x y : ℝ}, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem f_zero : f 0 = 0 :=
by sorry

theorem f_odd (x : ℝ) : f (-x) = -f x :=
by sorry

theorem solve_inequality {x : ℝ} (h : 0 < x) : f (Real.log x / Real.log 10 - 1) < 0 ↔ 0 < x ∧ x < 10 :=
by sorry

end f_zero_f_odd_solve_inequality_l1601_160139


namespace average_DE_l1601_160153

theorem average_DE 
  (a b c d e : ℝ) 
  (avg_all : (a + b + c + d + e) / 5 = 80) 
  (avg_abc : (a + b + c) / 3 = 78) : 
  (d + e) / 2 = 83 := 
sorry

end average_DE_l1601_160153


namespace max_area_of_rectangle_l1601_160174

-- Define the parameters and the problem
def perimeter := 150
def half_perimeter := perimeter / 2

theorem max_area_of_rectangle (x : ℕ) (y : ℕ) 
  (h1 : x + y = half_perimeter)
  (h2 : x > 0) (h3 : y > 0) :
  (∃ x y, x * y ≤ 1406) := 
sorry

end max_area_of_rectangle_l1601_160174


namespace algebraic_expression_value_l1601_160194

theorem algebraic_expression_value (a b : ℝ) (h : 4 * b = 3 + 4 * a) :
  a + (a - (a - (a - b) - b) - b) - b = -3 / 2 := by
  sorry

end algebraic_expression_value_l1601_160194


namespace tan_identity_l1601_160148

theorem tan_identity :
  let t5 := Real.tan (Real.pi / 36) -- 5 degrees in radians
  let t40 := Real.tan (Real.pi / 9)  -- 40 degrees in radians
  t5 + t40 + t5 * t40 = 1 :=
by
  sorry

end tan_identity_l1601_160148


namespace boat_speed_in_still_water_l1601_160151

theorem boat_speed_in_still_water (V_b : ℝ) : 
    (∀ (stream_speed : ℝ) (travel_time : ℝ) (distance : ℝ), 
        stream_speed = 5 ∧ 
        travel_time = 5 ∧ 
        distance = 105 →
        distance = (V_b + stream_speed) * travel_time) → 
    V_b = 16 := 
by 
    intro h
    specialize h 5 5 105 
    have h1 : 105 = (V_b + 5) * 5 := h ⟨rfl, ⟨rfl, rfl⟩⟩
    sorry

end boat_speed_in_still_water_l1601_160151


namespace profit_per_meter_correct_l1601_160171

-- Define the conditions
def total_meters := 40
def total_profit := 1400

-- Define the profit per meter calculation
def profit_per_meter := total_profit / total_meters

-- Theorem stating the profit per meter is Rs. 35
theorem profit_per_meter_correct : profit_per_meter = 35 := by
  sorry

end profit_per_meter_correct_l1601_160171
