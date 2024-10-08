import Mathlib

namespace lowest_point_graph_l133_133229

theorem lowest_point_graph (x : ℝ) (h : x > -1) : ∃ y, y = (x^2 + 2*x + 2) / (x + 1) ∧ y ≥ 2 ∧ (x = 0 → y = 2) :=
  sorry

end lowest_point_graph_l133_133229


namespace percentage_sum_l133_133061

theorem percentage_sum {A B : ℝ} 
  (hA : 0.40 * A = 160) 
  (hB : (2/3) * B = 160) : 
  0.60 * (A + B) = 384 :=
by
  sorry

end percentage_sum_l133_133061


namespace tangent_line_at_one_minimum_a_range_of_a_l133_133901

-- Definitions for the given functions
def g (a x : ℝ) := a * x^2 - (a + 2) * x
noncomputable def h (x : ℝ) := Real.log x
noncomputable def f (a x : ℝ) := g a x + h x

-- Part (1): Prove the tangent line equation at x = 1 for a = 1
theorem tangent_line_at_one (x y : ℝ) (h_x : x = 1) (h_a : 1 = (1 : ℝ)) :
  x + y + 1 = 0 := by
  sorry

-- Part (2): Prove the minimum value of a given certain conditions
theorem minimum_a (a : ℝ) (h_a_pos : 0 < a) (h_x : 1 ≤ x ∧ x ≤ Real.exp 1)
  (h_fmin : ∀ x, f a x ≥ -2) : 
  a = 1 := by
  sorry

-- Part (3): Prove the range of values for a given a condition
theorem range_of_a (a x₁ x₂ : ℝ) (h_x : 0 < x₁ ∧ x₁ < x₂) 
  (h_f : ∀ x₁ x₂, (f a x₁ - f a x₂) / (x₁ - x₂) > -2) :
  0 ≤ a ∧ a ≤ 8 := by
  sorry

end tangent_line_at_one_minimum_a_range_of_a_l133_133901


namespace odd_function_a_minus_b_l133_133268

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_a_minus_b
  (a b : ℝ)
  (h : is_odd_function (λ x => 2 * x ^ 3 + a * x ^ 2 + b - 1)) :
  a - b = -1 :=
sorry

end odd_function_a_minus_b_l133_133268


namespace debt_calculation_correct_l133_133611

-- Conditions
def initial_debt : ℤ := 40
def repayment : ℤ := initial_debt / 2
def additional_borrowing : ℤ := 10

-- Final Debt Calculation
def remaining_debt : ℤ := initial_debt - repayment
def final_debt : ℤ := remaining_debt + additional_borrowing

-- Proof Statement
theorem debt_calculation_correct : final_debt = 30 := 
by 
  -- Skipping the proof
  sorry

end debt_calculation_correct_l133_133611


namespace no_integer_cube_eq_3n_squared_plus_3n_plus_7_l133_133961

theorem no_integer_cube_eq_3n_squared_plus_3n_plus_7 :
  ¬ ∃ x n : ℤ, x^3 = 3 * n^2 + 3 * n + 7 := 
sorry

end no_integer_cube_eq_3n_squared_plus_3n_plus_7_l133_133961


namespace mrs_evans_class_l133_133063

def students_enrolled_in_class (S Q1 Q2 missing both: ℕ) : Prop :=
  25 = Q1 ∧ 22 = Q2 ∧ 5 = missing ∧ 22 = both → S = Q1 + Q2 - both + missing

theorem mrs_evans_class (S : ℕ) : students_enrolled_in_class S 25 22 5 22 :=
by
  sorry

end mrs_evans_class_l133_133063


namespace value_of_expression_l133_133272

theorem value_of_expression :
  (3 * (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) + 2) = 4373 :=
by
  sorry

end value_of_expression_l133_133272


namespace water_usage_l133_133766

noncomputable def litres_per_household_per_month (total_litres : ℕ) (number_of_households : ℕ) : ℕ :=
  total_litres / number_of_households

theorem water_usage : litres_per_household_per_month 2000 10 = 200 :=
by
  sorry

end water_usage_l133_133766


namespace sum_of_possible_values_of_x_l133_133651

theorem sum_of_possible_values_of_x :
  ∀ x : ℝ, (x + 2) * (x - 3) = 20 → ∃ s, s = 1 :=
by
  sorry

end sum_of_possible_values_of_x_l133_133651


namespace parallel_lines_iff_determinant_zero_l133_133839

theorem parallel_lines_iff_determinant_zero (a1 b1 c1 a2 b2 c2 : ℝ) :
  (a1 * b2 - a2 * b1 = 0) ↔ ((a1 * c2 - a2 * c1 = 0) → (b1 * c2 - b2 * c1 = 0)) := 
sorry

end parallel_lines_iff_determinant_zero_l133_133839


namespace product_of_square_roots_l133_133529

theorem product_of_square_roots (a b : ℝ) (h₁ : a^2 = 9) (h₂ : b^2 = 9) (h₃ : a ≠ b) : a * b = -9 :=
by
  -- Proof skipped
  sorry

end product_of_square_roots_l133_133529


namespace cards_given_l133_133128

-- Defining the conditions
def initial_cards : ℕ := 4
def final_cards : ℕ := 12

-- The theorem to be proved
theorem cards_given : final_cards - initial_cards = 8 := by
  -- Proof will go here
  sorry

end cards_given_l133_133128


namespace table_cost_l133_133202

variable (T : ℝ) -- Cost of the table
variable (C : ℝ) -- Cost of a chair

-- Conditions
axiom h1 : C = T / 7
axiom h2 : T + 4 * C = 220

theorem table_cost : T = 140 :=
by
  sorry

end table_cost_l133_133202


namespace intersection_of_complement_l133_133242

open Set

theorem intersection_of_complement (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6})
  (hA : A = {1, 3, 4}) (hB : B = {2, 3, 4, 5}) : A ∩ (U \ B) = {1} :=
by
  rw [hU, hA, hB]
  -- Proof steps go here
  sorry

end intersection_of_complement_l133_133242


namespace trajectory_of_midpoint_l133_133851

theorem trajectory_of_midpoint
  (M : ℝ × ℝ)
  (P : ℝ × ℝ) (Q : ℝ × ℝ)
  (hP : P = (4, 0))
  (hQ : Q.1^2 + Q.2^2 = 4)
  (M_is_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1 - 2)^2 + M.2^2 = 1 :=
sorry

end trajectory_of_midpoint_l133_133851


namespace range_of_a_l133_133545

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1 - a) / 2 * x^2 - x

theorem range_of_a (a : ℝ) (h : a ≠ 1) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ici 1 ∧ f a x₀ < a / (a - 1)) →
  a ∈ Set.Ioo (-Real.sqrt 2 - 1) (Real.sqrt 2 - 1) ∨ a ∈ Set.Ioi 1 :=
by sorry

end range_of_a_l133_133545


namespace point_three_units_away_from_A_is_negative_seven_or_negative_one_l133_133743

-- Defining the point A on the number line
def A : ℤ := -4

-- Definition of the condition where a point is 3 units away from A
def three_units_away (x : ℤ) : Prop := (x = A - 3) ∨ (x = A + 3)

-- The statement to be proved
theorem point_three_units_away_from_A_is_negative_seven_or_negative_one (x : ℤ) :
  three_units_away x → (x = -7 ∨ x = -1) :=
sorry

end point_three_units_away_from_A_is_negative_seven_or_negative_one_l133_133743


namespace combined_total_time_l133_133708

def jerry_time : ℕ := 3
def elaine_time : ℕ := 2 * jerry_time
def george_time : ℕ := elaine_time / 3
def kramer_time : ℕ := 0
def total_time : ℕ := jerry_time + elaine_time + george_time + kramer_time

theorem combined_total_time : total_time = 11 := by
  unfold total_time jerry_time elaine_time george_time kramer_time
  rfl

end combined_total_time_l133_133708


namespace long_sleeve_shirts_correct_l133_133259

def total_shirts : ℕ := 9
def short_sleeve_shirts : ℕ := 4
def long_sleeve_shirts : ℕ := total_shirts - short_sleeve_shirts

theorem long_sleeve_shirts_correct : long_sleeve_shirts = 5 := by
  sorry

end long_sleeve_shirts_correct_l133_133259


namespace cannot_invert_all_signs_l133_133938

structure RegularDecagon :=
  (vertices : Fin 10 → ℤ)
  (diagonals : Fin 45 → ℤ) -- Assume we encode the intersections as unique indices for simplicity.
  (all_positives : ∀ v, vertices v = 1 ∧ ∀ d, diagonals d = 1)

def isValidSignChange (t : List ℤ) : Prop :=
  t.length % 2 = 0

theorem cannot_invert_all_signs (D : RegularDecagon) :
  ¬ (∃ f : Fin 10 → ℤ → ℤ, ∀ (side : Fin 10) (val : ℤ), f side val = -val) :=
sorry

end cannot_invert_all_signs_l133_133938


namespace single_room_cost_l133_133937

theorem single_room_cost (total_rooms : ℕ) (single_rooms : ℕ) (double_room_cost : ℕ) 
  (total_revenue : ℤ) (x : ℤ) : 
  total_rooms = 260 → 
  single_rooms = 64 → 
  double_room_cost = 60 → 
  total_revenue = 14000 → 
  64 * x + (total_rooms - single_rooms) * double_room_cost = total_revenue → 
  x = 35 := 
by 
  intros h_total_rooms h_single_rooms h_double_room_cost h_total_revenue h_eqn 
  -- Add steps for proving if necessary
  sorry

end single_room_cost_l133_133937


namespace triangle_side_length_l133_133531

open Real

/-- Given a triangle ABC with the incircle touching side AB at point D,
where AD = 5 and DB = 3, and given that the angle A is 60 degrees,
prove that the length of side BC is 13. -/
theorem triangle_side_length
  (A B C D : Point)
  (AD DB : ℝ)
  (hAD : AD = 5)
  (hDB : DB = 3)
  (angleA : Real)
  (hangleA : angleA = π / 3) : 
  ∃ BC : ℝ, BC = 13 :=
sorry

end triangle_side_length_l133_133531


namespace average_snowfall_per_minute_l133_133838

def total_snowfall := 550
def days_in_december := 31
def hours_per_day := 24
def minutes_per_hour := 60

theorem average_snowfall_per_minute :
  (total_snowfall : ℝ) / (days_in_december * hours_per_day * minutes_per_hour) = 550 / (31 * 24 * 60) :=
by
  sorry

end average_snowfall_per_minute_l133_133838


namespace prob_both_correct_l133_133664

def prob_A : ℤ := 70
def prob_B : ℤ := 55
def prob_neither : ℤ := 20

theorem prob_both_correct : (prob_A + prob_B - (100 - prob_neither)) = 45 :=
by
  sorry

end prob_both_correct_l133_133664


namespace total_number_of_people_l133_133683

variables (A B : ℕ)

def pencils_brought_by_assoc_profs (A : ℕ) : ℕ := 2 * A
def pencils_brought_by_asst_profs (B : ℕ) : ℕ := B
def charts_brought_by_assoc_profs (A : ℕ) : ℕ := A
def charts_brought_by_asst_profs (B : ℕ) : ℕ := 2 * B

axiom pencils_total : pencils_brought_by_assoc_profs A + pencils_brought_by_asst_profs B = 10
axiom charts_total : charts_brought_by_assoc_profs A + charts_brought_by_asst_profs B = 11

theorem total_number_of_people : A + B = 7 :=
sorry

end total_number_of_people_l133_133683


namespace geom_series_ratio_l133_133773

noncomputable def geomSeries (a q : ℝ) (n : ℕ) : ℝ :=
a * ((1 - q ^ n) / (1 - q))

theorem geom_series_ratio (a1 q : ℝ) (h : 8 * a1 * q + a1 * q^4 = 0) :
  (geomSeries a1 q 5) / (geomSeries a1 q 2) = -11 :=
sorry

end geom_series_ratio_l133_133773


namespace length_of_AB_l133_133559

-- Given the conditions and the question to prove, we write:
theorem length_of_AB (AB CD : ℝ) (h : ℝ) 
  (area_ABC : ℝ := 0.5 * AB * h) 
  (area_ADC : ℝ := 0.5 * CD * h)
  (ratio_areas : area_ABC / area_ADC = 5 / 2)
  (sum_AB_CD : AB + CD = 280) :
  AB = 200 :=
by
  sorry

end length_of_AB_l133_133559


namespace smallest_possible_sum_l133_133884

theorem smallest_possible_sum (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hneq : a ≠ b) 
  (heq : (1 / a : ℚ) + (1 / b) = 1 / 12) : a + b = 49 :=
sorry

end smallest_possible_sum_l133_133884


namespace minimum_value_expression_l133_133979

theorem minimum_value_expression (x : ℝ) : ∃ y : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 = y ∧ ∀ z : ℝ, ((x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 ≥ z) ↔ (z = 2034) :=
by
  sorry

end minimum_value_expression_l133_133979


namespace simplify_and_evaluate_correct_l133_133551

noncomputable def simplify_and_evaluate (x y : ℚ) : ℚ :=
  3 * (x^2 - 2 * x * y) - (3 * x^2 - 2 * y + 2 * (x * y + y))

theorem simplify_and_evaluate_correct : 
  simplify_and_evaluate (-1 / 2 : ℚ) (-3 : ℚ) = -12 := by
  sorry

end simplify_and_evaluate_correct_l133_133551


namespace arrests_per_day_in_each_city_l133_133951

-- Define the known conditions
def daysOfProtest := 30
def numberOfCities := 21
def daysInJailBeforeTrial := 4
def daysInJailAfterTrial := 7 / 2 * 7 -- half of a 2-week sentence in days, converted from weeks to days
def combinedJailTimeInWeeks := 9900
def combinedJailTimeInDays := combinedJailTimeInWeeks * 7

-- Define the proof statement
theorem arrests_per_day_in_each_city :
  (combinedJailTimeInDays / (daysInJailBeforeTrial + daysInJailAfterTrial)) / daysOfProtest / numberOfCities = 10 := 
by
  sorry

end arrests_per_day_in_each_city_l133_133951


namespace addition_correctness_l133_133804

theorem addition_correctness : 1.25 + 47.863 = 49.113 :=
by 
  sorry

end addition_correctness_l133_133804


namespace scalene_triangle_height_ratio_l133_133237

theorem scalene_triangle_height_ratio {a b c : ℝ} (h1 : a > b ∧ b > c ∧ a > c)
  (h2 : a + c = 2 * b) : 
  1 / 3 < c / a ∧ c / a < 1 :=
by sorry

end scalene_triangle_height_ratio_l133_133237


namespace probability_correct_l133_133099

noncomputable def probability_all_players_have_5_after_2023_rings 
    (initial_money : ℕ)
    (num_rings : ℕ) 
    (target_money : ℕ)
    : ℝ := 
    if initial_money = 5 ∧ num_rings = 2023 ∧ target_money = 5 
    then 1 / 4 
    else 0

theorem probability_correct : 
        probability_all_players_have_5_after_2023_rings 5 2023 5 = 1 / 4 := 
by 
    sorry

end probability_correct_l133_133099


namespace sum_first_four_terms_of_arithmetic_sequence_l133_133322

theorem sum_first_four_terms_of_arithmetic_sequence (a₈ a₉ a₁₀ : ℤ) (d : ℤ) (a₁ a₂ a₃ a₄ : ℤ) : 
  (a₈ = 21) →
  (a₉ = 17) →
  (a₁₀ = 13) →
  (d = a₉ - a₈) →
  (a₁ = a₈ - 7 * d) →
  (a₂ = a₁ + d) →
  (a₃ = a₂ + d) →
  (a₄ = a₃ + d) →
  a₁ + a₂ + a₃ + a₄ = 172 :=
by 
  intros h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈
  sorry

end sum_first_four_terms_of_arithmetic_sequence_l133_133322


namespace friends_area_is_greater_by_14_point_4_times_l133_133829

theorem friends_area_is_greater_by_14_point_4_times :
  let tommy_length := 2 * 200
  let tommy_width := 3 * 150
  let tommy_area := tommy_length * tommy_width
  let friend_block_area := 180 * 180
  let friend_area := 80 * friend_block_area
  friend_area / tommy_area = 14.4 :=
by
  let tommy_length := 2 * 200
  let tommy_width := 3 * 150
  let tommy_area := tommy_length * tommy_width
  let friend_block_area := 180 * 180
  let friend_area := 80 * friend_block_area
  sorry

end friends_area_is_greater_by_14_point_4_times_l133_133829


namespace smallest_y_value_smallest_y_value_is_neg6_l133_133488

theorem smallest_y_value :
  ∀ y : ℝ, (3 * y^2 + 21 * y + 18 = y * (2 * y + 12)) → (y = -3 ∨ y = -6) :=
by
  sorry

theorem smallest_y_value_is_neg6 :
  ∃ y : ℝ, (3 * y^2 + 21 * y + 18 = y * (2 * y + 12)) ∧ (y = -6) :=
by
  have H := smallest_y_value
  sorry

end smallest_y_value_smallest_y_value_is_neg6_l133_133488


namespace books_borrowed_by_lunchtime_l133_133656

theorem books_borrowed_by_lunchtime (x : ℕ) :
  (∀ x : ℕ, 100 - x + 40 - 30 = 60) → (x = 50) :=
by
  intro h
  have eqn := h x
  sorry

end books_borrowed_by_lunchtime_l133_133656


namespace power_mod_remainder_l133_133990

theorem power_mod_remainder (a : ℕ) (n : ℕ) (h1 : 3^5 % 11 = 1) (h2 : 221 % 5 = 1) : 3^221 % 11 = 3 :=
by
  sorry

end power_mod_remainder_l133_133990


namespace gas_pipe_probability_l133_133041

-- Define the problem statement in Lean.
theorem gas_pipe_probability :
  let total_area := 400 * 400 / 2
  let usable_area := (300 - 100) * (300 - 100) / 2
  usable_area / total_area = 1 / 4 :=
by
  -- Sorry will be placeholder for the proof
  sorry

end gas_pipe_probability_l133_133041


namespace James_baked_muffins_l133_133112

theorem James_baked_muffins (arthur_muffins : Nat) (multiplier : Nat) (james_muffins : Nat) : 
  arthur_muffins = 115 → 
  multiplier = 12 → 
  james_muffins = arthur_muffins * multiplier → 
  james_muffins = 1380 :=
by
  intros haf ham hmul
  rw [haf, ham] at hmul
  simp at hmul
  exact hmul

end James_baked_muffins_l133_133112


namespace find_f_x_sq_minus_2_l133_133549

-- Define the polynomial and its given condition
def f (x : ℝ) : ℝ := sorry  -- f is some polynomial, we'll leave it unspecified for now

-- Assume the given condition
axiom f_condition : ∀ x : ℝ, f (x^2 + 2) = x^4 + 6 * x^2 + 4

-- Prove the desired result
theorem find_f_x_sq_minus_2 (x : ℝ) : f (x^2 - 2) = x^4 - 2 * x^2 - 4 :=
sorry

end find_f_x_sq_minus_2_l133_133549


namespace number_of_cars_l133_133857

theorem number_of_cars (C : ℕ) : 
  let bicycles := 3
  let pickup_trucks := 8
  let tricycles := 1
  let car_tires := 4
  let bicycle_tires := 2
  let pickup_truck_tires := 4
  let tricycle_tires := 3
  let total_tires := 101
  (4 * C + 3 * bicycle_tires + 8 * pickup_truck_tires + 1 * tricycle_tires = total_tires) → C = 15 := by
  intros h
  sorry

end number_of_cars_l133_133857


namespace hyperbola_focal_length_l133_133346

noncomputable def a : ℝ := Real.sqrt 10
noncomputable def b : ℝ := Real.sqrt 2
noncomputable def c : ℝ := Real.sqrt (a ^ 2 + b ^ 2)
noncomputable def focal_length : ℝ := 2 * c

theorem hyperbola_focal_length :
  focal_length = 4 * Real.sqrt 3 := by
  sorry

end hyperbola_focal_length_l133_133346


namespace initial_integer_value_l133_133871

theorem initial_integer_value (x : ℤ) (h : (x + 2) * (x + 2) = x * x - 2016) : x = -505 := 
sorry

end initial_integer_value_l133_133871


namespace no_valid_formation_l133_133352

-- Define the conditions related to the formation:
-- s : number of rows
-- t : number of musicians per row
-- Total musicians = s * t = 400
-- t is divisible by 4
-- 10 ≤ t ≤ 50
-- Additionally, the brass section needs to form a triangle in the first three rows
-- while maintaining equal distribution of musicians from each section in every row.

theorem no_valid_formation (s t : ℕ) (h_mul : s * t = 400) 
  (h_div : t % 4 = 0) 
  (h_range : 10 ≤ t ∧ t ≤ 50) 
  (h_triangle : ∀ (r1 r2 r3 : ℕ), r1 < r2 ∧ r2 < r3 → r1 + r2 + r3 = 100 → false) : 
  x = 0 := by
  sorry

end no_valid_formation_l133_133352


namespace toms_dad_gave_him_dimes_l133_133881

theorem toms_dad_gave_him_dimes (original_dimes final_dimes dimes_given : ℕ)
  (h1 : original_dimes = 15)
  (h2 : final_dimes = 48)
  (h3 : final_dimes = original_dimes + dimes_given) :
  dimes_given = 33 :=
by
  -- Since the main goal here is just the statement, proof is omitted with sorry
  sorry

end toms_dad_gave_him_dimes_l133_133881


namespace power_mod_five_l133_133761

theorem power_mod_five (n : ℕ) (hn : n ≡ 0 [MOD 4]): (3^2000 ≡ 1 [MOD 5]) :=
by 
  sorry

end power_mod_five_l133_133761


namespace find_coefficients_sum_l133_133198

theorem find_coefficients_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (2 * x - 3)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 10 :=
by
  intro h
  sorry

end find_coefficients_sum_l133_133198


namespace M_less_equal_fraction_M_M_greater_equal_fraction_M_M_less_equal_sum_M_l133_133020

noncomputable def M : ℕ → ℕ → ℕ → ℝ := sorry

theorem M_less_equal_fraction_M (n k h : ℕ) : 
  M n k h ≤ (n / h) * M (n-1) (k-1) (h-1) :=
sorry

theorem M_greater_equal_fraction_M (n k h : ℕ) : 
  M n k h ≥ (n / (n - h)) * M (n-1) k k :=
sorry

theorem M_less_equal_sum_M (n k h : ℕ) : 
  M n k h ≤ M (n-1) (k-1) (h-1) + M (n-1) k h :=
sorry

end M_less_equal_fraction_M_M_greater_equal_fraction_M_M_less_equal_sum_M_l133_133020


namespace hyperbola_asymptotes_l133_133240

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_asymptotes (a b : ℝ) (h : hyperbola_eccentricity a b = Real.sqrt 3) :
  (∀ x y : ℝ, (y = Real.sqrt 2 * x) ∨ (y = -Real.sqrt 2 * x)) :=
sorry

end hyperbola_asymptotes_l133_133240


namespace ratio_of_sums_l133_133528

open Nat

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 8 - 2 * a 3) / 7)

def arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem ratio_of_sums
    (a : ℕ → ℝ)
    (S : ℕ → ℝ)
    (a_arith : arithmetic_sequence_property a 1)
    (s_def : ∀ n, S n = sum_of_first_n_terms a n)
    (a8_eq_2a3 : a 8 = 2 * a 3) :
  S 15 / S 5 = 6 :=
sorry

end ratio_of_sums_l133_133528


namespace quotient_of_division_l133_133935

theorem quotient_of_division (S L Q : ℕ) (h1 : S = 270) (h2 : L - S = 1365) (h3 : L % S = 15) : Q = 6 :=
by
  sorry

end quotient_of_division_l133_133935


namespace binom_20_4_l133_133640

theorem binom_20_4 : Nat.choose 20 4 = 4845 := by
  sorry

end binom_20_4_l133_133640


namespace find_k_for_parallel_vectors_l133_133598

theorem find_k_for_parallel_vectors (k : ℝ) :
  let a := (1, k)
  let b := (9, k - 6)
  (1 * (k - 6) - 9 * k = 0) → k = -3 / 4 :=
by
  intros a b parallel_cond
  sorry

end find_k_for_parallel_vectors_l133_133598


namespace average_goals_is_92_l133_133809

-- Definitions based on conditions
def layla_goals : ℕ := 104
def kristin_fewer_goals : ℕ := 24
def kristin_goals : ℕ := layla_goals - kristin_fewer_goals
def combined_goals : ℕ := layla_goals + kristin_goals
def average_goals : ℕ := combined_goals / 2

-- Theorem
theorem average_goals_is_92 : average_goals = 92 := 
  sorry

end average_goals_is_92_l133_133809


namespace train_speed_is_85_kmh_l133_133005

noncomputable def speed_of_train_in_kmh (length_of_train : ℝ) (time_to_cross : ℝ) (speed_of_man_kmh : ℝ) : ℝ :=
  let speed_of_man_mps := speed_of_man_kmh * 1000 / 3600
  let relative_speed_mps := length_of_train / time_to_cross
  let speed_of_train_mps := relative_speed_mps - speed_of_man_mps
  speed_of_train_mps * 3600 / 1000

theorem train_speed_is_85_kmh
  (length_of_train : ℝ)
  (time_to_cross : ℝ)
  (speed_of_man_kmh : ℝ)
  (h1 : length_of_train = 150)
  (h2 : time_to_cross = 6)
  (h3 : speed_of_man_kmh = 5) :
  speed_of_train_in_kmh length_of_train time_to_cross speed_of_man_kmh = 85 :=
by
  sorry

end train_speed_is_85_kmh_l133_133005


namespace no_real_solutions_l133_133379

theorem no_real_solutions : ¬ ∃ (r s : ℝ),
  (r - 50) / 3 = (s - 2 * r) / 4 ∧
  r^2 + 3 * s = 50 :=
by {
  -- sorry, proof steps would go here
  sorry
}

end no_real_solutions_l133_133379


namespace box_height_l133_133401

theorem box_height (h : ℝ) :
  ∃ (h : ℝ), 
  let large_sphere_radius := 3
  let small_sphere_radius := 1.5
  let box_width := 6
  h = 12 := 
sorry

end box_height_l133_133401


namespace linear_inequality_solution_set_l133_133866

variable (x : ℝ)

theorem linear_inequality_solution_set :
  ∀ x : ℝ, (2 * x - 4 > 0) → (x > 2) := 
by
  sorry

end linear_inequality_solution_set_l133_133866


namespace part_I_part_II_l133_133340

noncomputable def f (x a : ℝ) : ℝ := x - 1 - a * Real.log x

theorem part_I (a : ℝ) (h1 : 0 < a) (h2 : ∀ x : ℝ, 0 < x → f x a ≥ 0) : a = 1 := 
sorry

theorem part_II (n : ℕ) (hn : 0 < n) : 
  let an := (1 + 1 / (n : ℝ)) ^ n
  let bn := (1 + 1 / (n : ℝ)) ^ (n + 1)
  an < Real.exp 1 ∧ Real.exp 1 < bn := 
sorry

end part_I_part_II_l133_133340


namespace salt_percentage_l133_133187

theorem salt_percentage :
  ∀ (salt water : ℝ), salt = 10 → water = 90 → 
  100 * (salt / (salt + water)) = 10 :=
by
  intros salt water h_salt h_water
  sorry

end salt_percentage_l133_133187


namespace delta_value_l133_133731

theorem delta_value (Δ : ℤ) (h : 4 * -3 = Δ - 3) : Δ = -9 :=
sorry

end delta_value_l133_133731


namespace f_of_3_l133_133744

def f (x : ℕ) : ℤ :=
  if x = 0 then sorry else 2 * (x - 1) - 1  -- Define an appropriate value for f(0) later

theorem f_of_3 : f 3 = 3 := by
  sorry

end f_of_3_l133_133744


namespace days_not_worked_correct_l133_133616

def total_days : ℕ := 20
def earnings_for_work (days_worked : ℕ) : ℤ := 80 * days_worked
def penalty_for_no_work (days_not_worked : ℕ) : ℤ := -40 * days_not_worked
def final_earnings (days_worked days_not_worked : ℕ) : ℤ := 
  (earnings_for_work days_worked) + (penalty_for_no_work days_not_worked)
def received_amount : ℤ := 880

theorem days_not_worked_correct {y x : ℕ} 
  (h1 : x + y = total_days) 
  (h2 : final_earnings x y = received_amount) :
  y = 6 :=
sorry

end days_not_worked_correct_l133_133616


namespace x_when_y_is_125_l133_133391

noncomputable def C : ℝ := (2^2) * (5^2)

theorem x_when_y_is_125 
  (x y : ℝ) 
  (h_pos : x > 0 ∧ y > 0) 
  (h_inv : x^2 * y^2 = C) 
  (h_initial : y = 5) 
  (h_x_initial : x = 2) 
  (h_y : y = 125) : 
  x = 2 / 25 :=
by
  sorry

end x_when_y_is_125_l133_133391


namespace base_b_conversion_l133_133600

theorem base_b_conversion (b : ℝ) (h₁ : 1 * 5^2 + 3 * 5^1 + 2 * 5^0 = 42) (h₂ : 2 * b^2 + 2 * b + 1 = 42) :
  b = (-1 + Real.sqrt 83) / 2 := 
  sorry

end base_b_conversion_l133_133600


namespace sharon_trip_distance_l133_133797

theorem sharon_trip_distance
  (x : ℝ)
  (usual_speed : ℝ := x / 180)
  (reduced_speed : ℝ := usual_speed - 1/3)
  (time_before_storm : ℝ := (x / 3) / usual_speed)
  (time_during_storm : ℝ := (2 * x / 3) / reduced_speed)
  (total_trip_time : ℝ := 276)
  (h : time_before_storm + time_during_storm = total_trip_time) :
  x = 135 :=
sorry

end sharon_trip_distance_l133_133797


namespace general_term_formula_l133_133842
-- Import the Mathlib library 

-- Define the conditions as given in the problem
/-- 
Define the sequence that represents the numerators. 
This is an arithmetic sequence of odd numbers starting from 1.
-/
def numerator (n : ℕ) : ℕ := 2 * n + 1

/-- 
Define the sequence that represents the denominators. 
This is a geometric sequence with the first term being 2 and common ratio being 2.
-/
def denominator (n : ℕ) : ℕ := 2^(n+1)

-- State the main theorem that we need to prove
theorem general_term_formula (n : ℕ) : (numerator n) / (denominator n) = (2 * n + 1) / 2^(n+1) :=
sorry

end general_term_formula_l133_133842


namespace completion_time_is_midnight_next_day_l133_133389

-- Define the initial start time
def start_time : ℕ := 9 -- 9:00 AM in hours

-- Define the completion time for 1/4th of the mosaic
def partial_completion_time : ℕ := 3 * 60 + 45  -- 3 hours and 45 minutes in minutes

-- Calculate total_time needed to complete the whole mosaic
def total_time : ℕ := 4 * partial_completion_time -- total time in minutes

-- Define the time at which the artist should finish the entire mosaic
def end_time : ℕ := start_time * 60 + total_time -- end time in minutes

-- Assuming 24 hours in a day, calculate 12:00 AM next day in minutes from midnight
def midnight_next_day : ℕ := 24 * 60

-- Theorem proving the artist will finish at 12:00 AM next day
theorem completion_time_is_midnight_next_day :
  end_time = midnight_next_day := by
    sorry -- proof not required

end completion_time_is_midnight_next_day_l133_133389


namespace prime_exists_solution_l133_133366

theorem prime_exists_solution (p : ℕ) [hp : Fact p.Prime] :
  ∃ n : ℕ, (6 * n^2 + 5 * n + 1) % p = 0 :=
by
  sorry

end prime_exists_solution_l133_133366


namespace battery_lasts_12_more_hours_l133_133355

-- Define initial conditions
def standby_battery_life : ℕ := 36
def active_battery_life : ℕ := 4
def total_time_on : ℕ := 12
def active_usage_time : ℕ := 90  -- in minutes

-- Conversion and calculation functions
def active_usage_hours : ℚ := active_usage_time / 60
def standby_consumption_rate : ℚ := 1 / standby_battery_life
def active_consumption_rate : ℚ := 1 / active_battery_life
def battery_used_standby : ℚ := (total_time_on - active_usage_hours) * standby_consumption_rate
def battery_used_active : ℚ := active_usage_hours * active_consumption_rate
def total_battery_used : ℚ := battery_used_standby + battery_used_active
def remaining_battery : ℚ := 1 - total_battery_used
def additional_hours_standby : ℚ := remaining_battery / standby_consumption_rate

-- Proof statement
theorem battery_lasts_12_more_hours : additional_hours_standby = 12 := by
  sorry

end battery_lasts_12_more_hours_l133_133355


namespace greatest_value_of_sum_l133_133786

theorem greatest_value_of_sum (x y : ℝ) (h₁ : x^2 + y^2 = 100) (h₂ : x * y = 40) :
  x + y = 6 * Real.sqrt 5 :=
by
  sorry

end greatest_value_of_sum_l133_133786


namespace max_branch_diameter_l133_133009

theorem max_branch_diameter (d : ℝ) (w : ℝ) (angle : ℝ) (H: w = 1 ∧ angle = 90) :
  d ≤ 2 * Real.sqrt 2 + 2 := 
sorry

end max_branch_diameter_l133_133009


namespace max_length_segment_l133_133313

theorem max_length_segment (p b : ℝ) (h : b = p / 2) : (b * (p - b)) / p = p / 4 :=
by
  sorry

end max_length_segment_l133_133313


namespace cos_sum_nonneg_one_l133_133114

theorem cos_sum_nonneg_one (x y z : ℝ) (h : x + y + z = 0) : abs (Real.cos x) + abs (Real.cos y) + abs (Real.cos z) ≥ 1 := 
by {
  sorry
}

end cos_sum_nonneg_one_l133_133114


namespace proof_problem_l133_133869

-- Define the operation table as a function in Lean 4
def op (a b : ℕ) : ℕ :=
  if a = 1 then
    if b = 1 then 2 else if b = 2 then 1 else if b = 3 then 4 else 3
  else if a = 2 then
    if b = 1 then 1 else if b = 2 then 3 else if b = 3 then 2 else 4
  else if a = 3 then
    if b = 1 then 4 else if b = 2 then 2 else if b = 3 then 1 else 3
  else
    if b = 1 then 3 else if b = 2 then 4 else if b = 3 then 3 else 2

-- State the theorem to prove
theorem proof_problem : op (op 3 1) (op 4 2) = 2 :=
by
  sorry

end proof_problem_l133_133869


namespace proof_problem_l133_133572

noncomputable def problem_statement : Prop :=
  ∃ (x1 x2 x3 x4 : ℕ), 
    x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0 ∧ 
    x1 + x2 + x3 + x4 = 8 ∧ 
    x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ 
    (x1 + x2) = 2 * 2 ∧ 
    (x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 - 4 * 2 * (x1 + x2 + x3 + x4) + 4 * 4) = 4 ∧ 
    (x1 = 1 ∧ x2 = 1 ∧ x3 = 3 ∧ x4 = 3)

theorem proof_problem : problem_statement :=
sorry

end proof_problem_l133_133572


namespace find_original_number_l133_133214

-- Variables and assumptions
variables (N y x : ℕ)

-- Conditions
def crossing_out_condition : Prop := N = 10 * y + x
def sum_condition : Prop := N + y = 54321
def remainder_condition : Prop := x = 54321 % 11
def y_value : Prop := y = 4938

-- The final proof problem statement
theorem find_original_number (h1 : crossing_out_condition N y x) (h2 : sum_condition N y) 
  (h3 : remainder_condition x) (h4 : y_value y) : N = 49383 :=
sorry

end find_original_number_l133_133214


namespace cubic_eq_one_real_root_l133_133419

-- Given a, b, c forming a geometric sequence
variables {a b c : ℝ}

-- Definition of a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Equation ax^3 + bx^2 + cx = 0
def cubic_eq (a b c x : ℝ) : Prop :=
  a * x^3 + b * x^2 + c * x = 0

-- Prove the number of real roots
theorem cubic_eq_one_real_root (h : geometric_sequence a b c) :
  ∃ x : ℝ, cubic_eq a b c x ∧ ¬∃ y ≠ x, cubic_eq a b c y :=
sorry

end cubic_eq_one_real_root_l133_133419


namespace blue_segments_count_l133_133563

def grid_size : ℕ := 16
def total_dots : ℕ := grid_size * grid_size
def red_dots : ℕ := 133
def boundary_red_dots : ℕ := 32
def corner_red_dots : ℕ := 2
def yellow_segments : ℕ := 196

-- Dummy hypotheses representing the given conditions
axiom red_dots_on_grid : red_dots <= total_dots
axiom boundary_red_dots_count : boundary_red_dots = 32
axiom corner_red_dots_count : corner_red_dots = 2
axiom total_yellow_segments : yellow_segments = 196

-- Proving the number of blue line segments
theorem blue_segments_count :  ∃ (blue_segments : ℕ), blue_segments = 134 := 
sorry

end blue_segments_count_l133_133563


namespace paula_cans_used_l133_133064

/-- 
  Paula originally had enough paint to cover 42 rooms. 
  Unfortunately, she lost 4 cans of paint on her way, 
  and now she can only paint 34 rooms. 
  Prove the number of cans she used for these 34 rooms is 17.
-/
theorem paula_cans_used (R L P C : ℕ) (hR : R = 42) (hL : L = 4) (hP : P = 34)
    (hRooms : R - ((R - P) / L) * L = P) :
  C = 17 :=
by
  sorry

end paula_cans_used_l133_133064


namespace computer_production_per_month_l133_133715

def days : ℕ := 28
def hours_per_day : ℕ := 24
def intervals_per_hour : ℕ := 2
def computers_per_interval : ℕ := 3

theorem computer_production_per_month : 
  (days * hours_per_day * intervals_per_hour * computers_per_interval = 4032) :=
by sorry

end computer_production_per_month_l133_133715


namespace parabola_intersects_x_axis_expression_l133_133642

theorem parabola_intersects_x_axis_expression (m : ℝ) (h : m^2 - m - 1 = 0) : m^2 - m + 2017 = 2018 := 
by 
  sorry

end parabola_intersects_x_axis_expression_l133_133642


namespace log_domain_eq_l133_133358

noncomputable def quadratic_expr (x : ℝ) : ℝ := x^2 - 2 * x - 3

def log_domain (x : ℝ) : Prop := quadratic_expr x > 0

theorem log_domain_eq :
  {x : ℝ | log_domain x} = 
  {x : ℝ | x < -1} ∪ {x : ℝ | x > 3} :=
by {
  sorry
}

end log_domain_eq_l133_133358


namespace find_nine_boxes_of_same_variety_l133_133251

theorem find_nine_boxes_of_same_variety (boxes : ℕ) (A B C : ℕ) (h_total : boxes = 25) (h_one_variety : boxes = A + B + C) 
  (hA : A ≤ 25) (hB : B ≤ 25) (hC : C ≤ 25) :
  (A ≥ 9) ∨ (B ≥ 9) ∨ (C ≥ 9) :=
sorry

end find_nine_boxes_of_same_variety_l133_133251


namespace wall_height_proof_l133_133827

-- The dimensions of the brick in meters
def brick_length : ℝ := 0.30
def brick_width : ℝ := 0.12
def brick_height : ℝ := 0.10

-- The dimensions of the wall in meters
def wall_length : ℝ := 6
def wall_width : ℝ := 4

-- The number of bricks needed
def number_of_bricks : ℝ := 1366.6666666666667

-- The height of the wall in meters
def wall_height : ℝ := 0.205

-- The volume of one brick
def volume_of_one_brick : ℝ := brick_length * brick_width * brick_height

-- The total volume of all bricks needed
def total_volume_of_bricks : ℝ := number_of_bricks * volume_of_one_brick

-- The volume of the wall
def volume_of_wall : ℝ := wall_length * wall_width * wall_height

-- Proof that the height of the wall is 0.205 meters
theorem wall_height_proof : volume_of_wall = total_volume_of_bricks :=
by
  -- use definitions to evaluate the equality
  sorry

end wall_height_proof_l133_133827


namespace james_initial_bars_l133_133082

def initial_chocolate_bars (sold_last_week sold_this_week needs_to_sell : ℕ) : ℕ :=
  sold_last_week + sold_this_week + needs_to_sell

theorem james_initial_bars : 
  initial_chocolate_bars 5 7 6 = 18 :=
by 
  sorry

end james_initial_bars_l133_133082


namespace dice_sum_not_11_l133_133643

/-- Jeremy rolls three standard six-sided dice, with each showing a different number and the product of the numbers on the upper faces is 72.
    Prove that the sum 11 is not possible. --/
theorem dice_sum_not_11 : 
  ∃ (a b c : ℕ), 
    (1 ≤ a ∧ a ≤ 6) ∧ 
    (1 ≤ b ∧ b ≤ 6) ∧ 
    (1 ≤ c ∧ c ≤ 6) ∧ 
    (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ 
    (a * b * c = 72) ∧ 
    (a > 4 ∨ b > 4 ∨ c > 4) → 
    a + b + c ≠ 11 := 
by
  sorry

end dice_sum_not_11_l133_133643


namespace linear_equation_in_x_l133_133338

theorem linear_equation_in_x (m : ℤ) (h : |m| = 1) (h₂ : m - 1 ≠ 0) : m = -1 :=
sorry

end linear_equation_in_x_l133_133338


namespace original_price_l133_133121

theorem original_price (x : ℝ) (h1 : 0.95 * x * 1.40 = 1.33 * x) (h2 : 1.33 * x = 2 * x - 1352.06) : x = 2018 := sorry

end original_price_l133_133121


namespace store_discount_percentage_l133_133067

theorem store_discount_percentage
  (total_without_discount : ℝ := 350)
  (final_price : ℝ := 252)
  (coupon_percentage : ℝ := 0.1) :
  ∃ (x : ℝ), total_without_discount * (1 - x / 100) * (1 - coupon_percentage) = final_price ∧ x = 20 :=
by
  use 20
  sorry

end store_discount_percentage_l133_133067


namespace line_passes_through_parabola_vertex_l133_133350

theorem line_passes_through_parabola_vertex : 
  ∃ (c : ℝ), (∀ (x : ℝ), y = 2 * x + c → ∃ (x0 : ℝ), (x0 = 0 ∧ y = c^2)) ∧ 
  (∀ (c1 c2 : ℝ), (y = 2 * x + c1 ∧ y = 2 * x + c2 → c1 = c2)) → 
  ∃ c : ℝ, c = 0 ∨ c = 1 :=
by 
  -- Proof should be inserted here
  sorry

end line_passes_through_parabola_vertex_l133_133350


namespace square_of_sum_opposite_l133_133661

theorem square_of_sum_opposite (a b : ℝ) : (-(a) + b)^2 = (-a + b)^2 :=
by
  sorry

end square_of_sum_opposite_l133_133661


namespace percentage_increase_of_gross_l133_133481

theorem percentage_increase_of_gross
  (P R : ℝ)
  (price_drop : ℝ := 0.20)
  (quantity_increase : ℝ := 0.60)
  (original_gross : ℝ := P * R)
  (new_price : ℝ := (1 - price_drop) * P)
  (new_quantity_sold : ℝ := (1 + quantity_increase) * R)
  (new_gross : ℝ := new_price * new_quantity_sold)
  (percentage_increase : ℝ := ((new_gross - original_gross) / original_gross) * 100) :
  percentage_increase = 28 :=
by
  sorry

end percentage_increase_of_gross_l133_133481


namespace simplify_expression_l133_133292

variable (a b c : ℝ) 

theorem simplify_expression (h1 : a ≠ 4) (h2 : b ≠ 5) (h3 : c ≠ 6) :
  (a - 4) / (6 - c) * (b - 5) / (4 - a) * (c - 6) / (5 - b) = -1 :=
by
  sorry

end simplify_expression_l133_133292


namespace half_angle_third_quadrant_l133_133213

theorem half_angle_third_quadrant (α : ℝ) (k : ℤ) (h1 : k * 360 + 180 < α) (h2 : α < k * 360 + 270) : 
  (∃ n : ℤ, n * 360 + 90 < (α / 2) ∧ (α / 2) < n * 360 + 135) ∨ 
  (∃ n : ℤ, n * 360 + 270 < (α / 2) ∧ (α / 2) < n * 360 + 315) := 
sorry

end half_angle_third_quadrant_l133_133213


namespace prob_factor_less_than_nine_l133_133510

theorem prob_factor_less_than_nine : 
  (∃ (n : ℕ), n = 72) ∧ (∃ (total_factors : ℕ), total_factors = 12) ∧ 
  (∃ (factors_lt_9 : ℕ), factors_lt_9 = 6) → 
  (6 / 12 : ℚ) = (1 / 2 : ℚ) := 
by
  sorry

end prob_factor_less_than_nine_l133_133510


namespace no_intersection_points_of_polar_graphs_l133_133790

theorem no_intersection_points_of_polar_graphs :
  let c1_center := (3 / 2, 0)
  let r1 := 3 / 2
  let c2_center := (0, 3)
  let r2 := 3
  let distance_between_centers := Real.sqrt ((3 / 2 - 0) ^ 2 + (0 - 3) ^ 2)
  distance_between_centers > r1 + r2 :=
by
  sorry

end no_intersection_points_of_polar_graphs_l133_133790


namespace chen_steps_recorded_correct_l133_133225

-- Define the standard for steps per day
def standard : ℕ := 5000

-- Define the steps walked by Xia
def xia_steps : ℕ := 6200

-- Define the recorded steps for Xia
def xia_recorded : ℤ := xia_steps - standard

-- Assert that Xia's recorded steps are +1200
lemma xia_steps_recorded_correct : xia_recorded = 1200 := by
  sorry

-- Define the steps walked by Chen
def chen_steps : ℕ := 4800

-- Define the recorded steps for Chen
def chen_recorded : ℤ := standard - chen_steps

-- State and prove that Chen's recorded steps are -200
theorem chen_steps_recorded_correct : chen_recorded = -200 :=
  sorry

end chen_steps_recorded_correct_l133_133225


namespace complete_the_square_b_l133_133100

theorem complete_the_square_b (x : ℝ) : (x ^ 2 - 6 * x + 7 = 0) → ∃ a b : ℝ, (x + a) ^ 2 = b ∧ b = 2 :=
by
sorry

end complete_the_square_b_l133_133100


namespace Sally_lost_20_Pokemon_cards_l133_133177

theorem Sally_lost_20_Pokemon_cards (original_cards : ℕ) (received_cards : ℕ) (final_cards : ℕ) (lost_cards : ℕ) 
  (h1 : original_cards = 27) 
  (h2 : received_cards = 41) 
  (h3 : final_cards = 48) 
  (h4 : original_cards + received_cards - lost_cards = final_cards) : 
  lost_cards = 20 := 
sorry

end Sally_lost_20_Pokemon_cards_l133_133177


namespace smallest_denominator_fraction_interval_exists_l133_133223

def interval (a b c d : ℕ) : Prop :=
a = 14 ∧ b = 73 ∧ c = 5 ∧ d = 26

theorem smallest_denominator_fraction_interval_exists :
  ∃ (a b c d : ℕ), 
    a / b < 19 / 99 ∧ b < 99 ∧
    19 / 99 < c / d ∧ d < 99 ∧
    interval a b c d :=
by
  sorry

end smallest_denominator_fraction_interval_exists_l133_133223


namespace sum_of_coefficients_eq_l133_133415

theorem sum_of_coefficients_eq :
  ∃ n : ℕ, (∀ a b : ℕ, (3 * a + 5 * b)^n = 2^15) → n = 5 :=
by
  sorry

end sum_of_coefficients_eq_l133_133415


namespace average_rate_of_trip_l133_133369

theorem average_rate_of_trip (d : ℝ) (r1 : ℝ) (t1 : ℝ) (r_total : ℝ) :
  d = 640 →
  r1 = 80 →
  t1 = (320 / r1) →
  t2 = 3 * t1 →
  r_total = d / (t1 + t2) →
  r_total = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_rate_of_trip_l133_133369


namespace total_days_to_finish_job_l133_133003

noncomputable def workers_job_completion
  (initial_workers : ℕ)
  (additional_workers : ℕ)
  (initial_days : ℕ)
  (total_days : ℕ)
  (work_completion_days : ℕ)
  (remaining_work : ℝ)
  (additional_days_needed : ℝ)
  : ℝ :=
  initial_days + additional_days_needed

theorem total_days_to_finish_job
  (initial_workers : ℕ := 6)
  (additional_workers : ℕ := 4)
  (initial_days : ℕ := 3)
  (total_days : ℕ := 8)
  (work_completion_days : ℕ := 8)
  : workers_job_completion initial_workers additional_workers initial_days total_days work_completion_days (1 - (initial_days : ℝ) / work_completion_days) (remaining_work / (((initial_workers + additional_workers) : ℝ) / work_completion_days)) = 3.5 :=
  sorry

end total_days_to_finish_job_l133_133003


namespace competition_end_time_l133_133146

def time := ℕ × ℕ -- Representing time as a pair of hours and minutes

def start_time : time := (15, 15) -- 3:15 PM is represented as 15:15 in 24-hour format
def duration := 1825 -- Duration in minutes
def end_time : time := (21, 40) -- 9:40 PM is represented as 21:40 in 24-hour format

def add_minutes (t : time) (m : ℕ) : time :=
  let (h, min) := t
  let total_minutes := h * 60 + min + m
  (total_minutes / 60 % 24, total_minutes % 60)

theorem competition_end_time :
  add_minutes start_time duration = end_time :=
by
  -- The proof would go here
  sorry

end competition_end_time_l133_133146


namespace april_revenue_l133_133399

def revenue_after_tax (initial_roses : ℕ) (initial_tulips : ℕ) (initial_daisies : ℕ)
                      (final_roses : ℕ) (final_tulips : ℕ) (final_daisies : ℕ)
                      (price_rose : ℝ) (price_tulip : ℝ) (price_daisy : ℝ) (tax_rate : ℝ) : ℝ :=
(price_rose * (initial_roses - final_roses) + price_tulip * (initial_tulips - final_tulips) + price_daisy * (initial_daisies - final_daisies)) * (1 + tax_rate)

theorem april_revenue :
  revenue_after_tax 13 10 8 4 3 1 4 3 2 0.10 = 78.10 := by
  sorry

end april_revenue_l133_133399


namespace find_x3_plus_y3_l133_133091

theorem find_x3_plus_y3 (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 167) : x^3 + y^3 = 2005 :=
sorry

end find_x3_plus_y3_l133_133091


namespace elder_twice_as_old_l133_133071

theorem elder_twice_as_old (Y E : ℕ) (hY : Y = 35) (hDiff : E - Y = 20) : ∃ (X : ℕ),  X = 15 ∧ E - X = 2 * (Y - X) := 
by
  sorry

end elder_twice_as_old_l133_133071


namespace findMultipleOfSamsMoney_l133_133170

-- Define the conditions specified in the problem
def SamMoney : ℕ := 75
def TotalMoney : ℕ := 200
def BillyHasLess (x : ℕ) : ℕ := x * SamMoney - 25

-- State the theorem to prove
theorem findMultipleOfSamsMoney (x : ℕ) 
  (h1 : SamMoney + BillyHasLess x = TotalMoney) : x = 2 :=
by
  -- Placeholder for the proof
  sorry

end findMultipleOfSamsMoney_l133_133170


namespace min_value_of_expr_l133_133231

theorem min_value_of_expr (n : ℕ) (hn : n > 0) : (n / 3) + (27 / n) = 6 :=
by
  sorry

end min_value_of_expr_l133_133231


namespace infinitesolutions_k_l133_133200

-- Define the system of equations as given in the problem
def system_of_equations (x y k : ℝ) : Prop :=
  (3 * x - 4 * y = 5) ∧ (9 * x - 12 * y = k)

-- State the theorem that describes the condition for infinitely many solutions
theorem infinitesolutions_k (k : ℝ) :
  (∀ (x y : ℝ), system_of_equations x y k) ↔ k = 15 :=
by
  sorry

end infinitesolutions_k_l133_133200


namespace goteborg_to_stockholm_distance_l133_133388

/-- 
Given that the distance from Goteborg to Jonkoping on a map is 100 cm 
and the distance from Jonkoping to Stockholm is 150 cm, with a map scale of 1 cm: 20 km,
prove that the total distance from Goteborg to Stockholm passing through Jonkoping is 5000 km.
-/
theorem goteborg_to_stockholm_distance :
  let distance_G_to_J := 100 -- distance from Goteborg to Jonkoping in cm
  let distance_J_to_S := 150 -- distance from Jonkoping to Stockholm in cm
  let scale := 20 -- scale of the map, 1 cm : 20 km
  distance_G_to_J * scale + distance_J_to_S * scale = 5000 := 
by 
  let distance_G_to_J := 100 -- defining the distance from Goteborg to Jonkoping in cm
  let distance_J_to_S := 150 -- defining the distance from Jonkoping to Stockholm in cm
  let scale := 20 -- defining the scale of the map, 1 cm : 20 km
  sorry

end goteborg_to_stockholm_distance_l133_133388


namespace find_pairs_l133_133244

theorem find_pairs (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) :
  y ∣ x^2 + 1 ∧ x^2 ∣ y^3 + 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = 3 ∧ y = 2) :=
by
  sorry

end find_pairs_l133_133244


namespace Tyrone_total_money_l133_133568

theorem Tyrone_total_money :
  let usd_bills := 4 * 1 + 1 * 10 + 2 * 5 + 30 * 0.25 + 5 * 0.5 + 48 * 0.1 + 12 * 0.05 + 4 * 1 + 64 * 0.01 + 3 * 2 + 5 * 0.5
  let euro_to_usd := 20 * 1.1
  let pound_to_usd := 15 * 1.32
  let cad_to_usd := 6 * 0.76
  let total_usd_currency := usd_bills
  let total_foreign_usd_currency := euro_to_usd + pound_to_usd + cad_to_usd
  let total_money := total_usd_currency + total_foreign_usd_currency
  total_money = 98.90 :=
by
  sorry

end Tyrone_total_money_l133_133568


namespace unique_reconstruction_l133_133883

theorem unique_reconstruction (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (a b c d : ℝ) (Ha : x + y = a) (Hb : x - y = b) (Hc : x * y = c) (Hd : x / y = d) :
  ∃! (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' + y' = a ∧ x' - y' = b ∧ x' * y' = c ∧ x' / y' = d := 
sorry

end unique_reconstruction_l133_133883


namespace geometric_sequence_common_ratio_l133_133553

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 = 2)
  (h3 : a 5 = 1/4) :
  q = 1/2 :=
sorry

end geometric_sequence_common_ratio_l133_133553


namespace illuminated_cube_surface_area_l133_133188

noncomputable def edge_length : ℝ := Real.sqrt (2 + Real.sqrt 3)
noncomputable def radius : ℝ := Real.sqrt 2
noncomputable def illuminated_area (a ρ : ℝ) : ℝ := Real.sqrt 3 * (Real.pi + 3)

theorem illuminated_cube_surface_area :
  illuminated_area edge_length radius = Real.sqrt 3 * (Real.pi + 3) := sorry

end illuminated_cube_surface_area_l133_133188


namespace cost_expression_A_cost_expression_B_cost_comparison_10_students_cost_comparison_4_students_l133_133068

-- Define the conditions
def ticket_full_price : ℕ := 240
def discount_A : ℕ := ticket_full_price / 2
def discount_B (x : ℕ) : ℕ := 144 * (x + 1)

-- Algebraic expressions provided in the answer
def cost_A (x : ℕ) : ℕ := discount_A * x + ticket_full_price
def cost_B (x : ℕ) : ℕ := 144 * (x + 1)

-- Proofs for the specific cases
theorem cost_expression_A (x : ℕ) : cost_A x = 120 * x + 240 := by
  sorry

theorem cost_expression_B (x : ℕ) : cost_B x = 144 * (x + 1) := by
  sorry

theorem cost_comparison_10_students : cost_A 10 < cost_B 10 := by
  sorry

theorem cost_comparison_4_students : cost_A 4 = cost_B 4 := by
  sorry

end cost_expression_A_cost_expression_B_cost_comparison_10_students_cost_comparison_4_students_l133_133068


namespace sin_alpha_cos_beta_value_l133_133574

variables {α β : ℝ}

theorem sin_alpha_cos_beta_value 
  (h1 : Real.sin (α + β) = 1/2) 
  (h2 : 2 * Real.sin (α - β) = 1/2) : 
  Real.sin α * Real.cos β = 3/8 := by
sorry

end sin_alpha_cos_beta_value_l133_133574


namespace invalid_inverse_statement_l133_133691

/- Define the statements and their inverses -/

/-- Statement A: Vertical angles are equal. -/
def statement_A : Prop := ∀ {α β : ℝ}, α ≠ β → α = β

/-- Inverse of Statement A: If two angles are equal, then they are vertical angles. -/
def inverse_A : Prop := ∀ {α β : ℝ}, α = β → α ≠ β

/-- Statement B: If |a| = |b|, then a = b. -/
def statement_B (a b : ℝ) : Prop := abs a = abs b → a = b

/-- Inverse of Statement B: If a = b, then |a| = |b|. -/
def inverse_B (a b : ℝ) : Prop := a = b → abs a = abs b

/-- Statement C: If two lines are parallel, then the alternate interior angles are equal. -/
def statement_C (l1 l2 : Prop) : Prop := l1 → l2

/-- Inverse of Statement C: If the alternate interior angles are equal, then the two lines are parallel. -/
def inverse_C (l1 l2 : Prop) : Prop := l2 → l1

/-- Statement D: If a^2 = b^2, then a = b. -/
def statement_D (a b : ℝ) : Prop := a^2 = b^2 → a = b

/-- Inverse of Statement D: If a = b, then a^2 = b^2. -/
def inverse_D (a b : ℝ) : Prop := a = b → a^2 = b^2

/-- The statement that does not have a valid inverse among A, B, C, and D is statement A. -/
theorem invalid_inverse_statement : ¬inverse_A :=
by
sorry

end invalid_inverse_statement_l133_133691


namespace algebraic_identity_l133_133763

theorem algebraic_identity (a : ℚ) (h : a + a⁻¹ = 3) : a^2 + a⁻¹^2 = 7 := 
  sorry

end algebraic_identity_l133_133763


namespace find_third_number_l133_133144

-- Definitions
def A : ℕ := 600
def B : ℕ := 840
def LCM : ℕ := 50400
def HCF : ℕ := 60

-- Theorem to be proven
theorem find_third_number (C : ℕ) (h_lcm : Nat.lcm (Nat.lcm A B) C = LCM) (h_hcf : Nat.gcd (Nat.gcd A B) C = HCF) : C = 6 :=
by -- proof
  sorry

end find_third_number_l133_133144


namespace ab_plus_cd_l133_133680

variable (a b c d : ℝ)

theorem ab_plus_cd (h1 : a + b + c = -4)
                  (h2 : a + b + d = 2)
                  (h3 : a + c + d = 15)
                  (h4 : b + c + d = 10) :
                  a * b + c * d = 485 / 9 :=
by
  sorry

end ab_plus_cd_l133_133680


namespace determine_values_of_abc_l133_133878

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def f_inv (a b c : ℝ) (x : ℝ) : ℝ := c * x^2 + b * x + a

theorem determine_values_of_abc 
  (a b c : ℝ) 
  (h_f : ∀ x : ℝ, f a b c (f_inv a b c x) = x)
  (h_f_inv : ∀ x : ℝ, f_inv a b c (f a b c x) = x) : 
  a = -1 ∧ b = 1 ∧ c = 0 :=
by
  sorry

end determine_values_of_abc_l133_133878


namespace scientific_notation_equivalence_l133_133787

/-- The scientific notation for 20.26 thousand hectares in square meters is equal to 2.026 × 10^9. -/
theorem scientific_notation_equivalence :
  (20.26 * 10^3 * 10^4) = 2.026 * 10^9 := 
sorry

end scientific_notation_equivalence_l133_133787


namespace initial_dimes_l133_133902

theorem initial_dimes (dimes_received_from_dad : ℕ) (dimes_received_from_mom : ℕ) (total_dimes_now : ℕ) : 
  dimes_received_from_dad = 8 → dimes_received_from_mom = 4 → total_dimes_now = 19 → 
  total_dimes_now - (dimes_received_from_dad + dimes_received_from_mom) = 7 :=
by
  intros
  sorry

end initial_dimes_l133_133902


namespace percentage_increase_l133_133546

variable (x r : ℝ)

theorem percentage_increase (h_x : x = 78.4) (h_r : x = 70 * (1 + r)) : r = 0.12 :=
by
  -- Proof goes here
  sorry

end percentage_increase_l133_133546


namespace total_price_paid_l133_133033

noncomputable def total_price
    (price_rose : ℝ) (qty_rose : ℕ) (discount_rose : ℝ)
    (price_lily : ℝ) (qty_lily : ℕ) (discount_lily : ℝ)
    (price_sunflower : ℝ) (qty_sunflower : ℕ)
    (store_discount : ℝ) (tax_rate : ℝ)
    : ℝ :=
  let total_rose := qty_rose * price_rose
  let total_lily := qty_lily * price_lily
  let total_sunflower := qty_sunflower * price_sunflower
  let total := total_rose + total_lily + total_sunflower
  let total_disc_rose := total_rose * discount_rose
  let total_disc_lily := total_lily * discount_lily
  let discounted_total := total - total_disc_rose - total_disc_lily
  let store_discount_amount := discounted_total * store_discount
  let after_store_discount := discounted_total - store_discount_amount
  let tax_amount := after_store_discount * tax_rate
  after_store_discount + tax_amount

theorem total_price_paid :
  total_price 20 3 0.15 15 5 0.10 10 2 0.05 0.07 = 140.79 :=
by
  apply sorry

end total_price_paid_l133_133033


namespace lines_forming_angle_bamboo_pole_longest_shadow_angle_l133_133915

-- Define the angle between sunlight and ground
def angle_sunlight_ground : ℝ := 60

-- Proof problem 1 statement
theorem lines_forming_angle (A : ℝ) : 
  (A > angle_sunlight_ground → ∃ l : ℕ, l = 0) ∧ (A < angle_sunlight_ground → ∃ l : ℕ, ∀ n : ℕ, n > l) :=
  sorry

-- Proof problem 2 statement
theorem bamboo_pole_longest_shadow_angle : 
  ∀ bamboo_pole_angle ground_angle : ℝ, 
  (ground_angle = 60 → bamboo_pole_angle = 30) :=
  sorry

end lines_forming_angle_bamboo_pole_longest_shadow_angle_l133_133915


namespace probability_all_same_color_l133_133324

theorem probability_all_same_color :
  let red_plates := 7
  let blue_plates := 5
  let total_plates := red_plates + blue_plates
  let total_combinations := Nat.choose total_plates 3
  let red_combinations := Nat.choose red_plates 3
  let blue_combinations := Nat.choose blue_plates 3
  let favorable_combinations := red_combinations + blue_combinations
  let probability := (favorable_combinations : ℚ) / total_combinations
  probability = 9 / 44 :=
by 
  sorry

end probability_all_same_color_l133_133324


namespace triangle_area_ratio_l133_133548

-- Define parabola and focus
def parabola (x y : ℝ) : Prop := y^2 = 8 * x
def focus : (ℝ × ℝ) := (2, 0)

-- Define the line passing through the focus and intersecting the parabola
def line_through_focus (f : ℝ × ℝ) (a b : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  l (f.1) = f.2 ∧ parabola a.1 a.2 ∧ parabola b.1 b.2 ∧   -- line passes through the focus and intersects parabola at a and b
  l a.1 = a.2 ∧ l b.1 = b.2 ∧ 
  |a.1 - f.1| + |a.2 - f.2| = 3 ∧ -- condition |AF| = 3
  (f = (2, 0))

-- The proof problem
theorem triangle_area_ratio (a b : ℝ × ℝ) (l : ℝ → ℝ) 
  (h_line : line_through_focus focus a b l) :
  ∃ r, r = (1 / 2) := 
sorry

end triangle_area_ratio_l133_133548


namespace integer_root_b_l133_133966

theorem integer_root_b (a1 a2 a3 a4 a5 b : ℤ)
  (h_diff : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a3 ≠ a4 ∧ a3 ≠ a5 ∧ a4 ≠ a5)
  (h_sum : a1 + a2 + a3 + a4 + a5 = 9)
  (h_prod : (b - a1) * (b - a2) * (b - a3) * (b - a4) * (b - a5) = 2009) :
  b = 10 :=
sorry

end integer_root_b_l133_133966


namespace building_height_is_74_l133_133543

theorem building_height_is_74
  (building_shadow : ℚ)
  (flagpole_height : ℚ)
  (flagpole_shadow : ℚ)
  (ratio_valid : building_shadow / flagpole_shadow = 21 / 8)
  (flagpole_height_value : flagpole_height = 28)
  (building_shadow_value : building_shadow = 84)
  (flagpole_shadow_value : flagpole_shadow = 32) :
  ∃ (h : ℚ), h = 74 := by
  sorry

end building_height_is_74_l133_133543


namespace emily_num_dresses_l133_133166

theorem emily_num_dresses (M : ℕ) (D : ℕ) (E : ℕ) 
  (h1 : D = M + 12) 
  (h2 : M = E / 2) 
  (h3 : M + D + E = 44) : 
  E = 16 := 
by 
  sorry

end emily_num_dresses_l133_133166


namespace log_inequality_l133_133392

theorem log_inequality : 
  ∀ (logπ2 log2π : ℝ), logπ2 = 1 / log2π → 0 < logπ2 → 0 < log2π → (1 / logπ2 + 1 / log2π > 2) :=
by
  intros logπ2 log2π h1 h2 h3
  have h4: logπ2 = 1 / log2π := h1
  have h5: 0 < logπ2 := h2
  have h6: 0 < log2π := h3
  -- To be completed with the actual proof steps if needed
  sorry

end log_inequality_l133_133392


namespace jellybeans_in_new_bag_l133_133891

theorem jellybeans_in_new_bag (average_per_bag : ℕ) (num_bags : ℕ) (additional_avg_increase : ℕ) (total_jellybeans_old : ℕ) (total_jellybeans_new : ℕ) (num_bags_new : ℕ) (new_bag_jellybeans : ℕ) : 
  average_per_bag = 117 → 
  num_bags = 34 → 
  additional_avg_increase = 7 → 
  total_jellybeans_old = num_bags * average_per_bag → 
  total_jellybeans_new = (num_bags + 1) * (average_per_bag + additional_avg_increase) → 
  new_bag_jellybeans = total_jellybeans_new - total_jellybeans_old → 
  new_bag_jellybeans = 362 := 
by 
  intros 
  sorry

end jellybeans_in_new_bag_l133_133891


namespace factorization_x3_minus_9xy2_l133_133086

theorem factorization_x3_minus_9xy2 (x y : ℝ) : x^3 - 9 * x * y^2 = x * (x + 3 * y) * (x - 3 * y) :=
by sorry

end factorization_x3_minus_9xy2_l133_133086


namespace find_positive_m_l133_133205

theorem find_positive_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → x = y) ↔ m = 16 :=
by
  sorry

end find_positive_m_l133_133205


namespace square_inscribed_in_hexagon_has_side_length_l133_133025

-- Definitions for the conditions given
noncomputable def side_length_square (AB EF : ℝ) : ℝ :=
  if AB = 30 ∧ EF = 19 * (Real.sqrt 3 - 1) then 10 * Real.sqrt 3 else 0

-- The theorem stating the specified equality
theorem square_inscribed_in_hexagon_has_side_length (AB EF : ℝ)
  (hAB : AB = 30) (hEF : EF = 19 * (Real.sqrt 3 - 1)) :
  side_length_square AB EF = 10 * Real.sqrt 3 := 
by 
  -- This is the proof placeholder
  sorry

end square_inscribed_in_hexagon_has_side_length_l133_133025


namespace divisible_by_12_l133_133913

theorem divisible_by_12 (n : ℤ) : 12 ∣ (n^4 - n^2) := sorry

end divisible_by_12_l133_133913


namespace mean_and_variance_of_y_l133_133564

noncomputable def mean (xs : List ℝ) : ℝ :=
  if h : xs.length > 0 then xs.sum / xs.length else 0

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  if h : xs.length > 0 then (xs.map (λ x => (x - m)^2)).sum / xs.length else 0

theorem mean_and_variance_of_y
  (x : List ℝ)
  (hx_len : x.length = 20)
  (hx_mean : mean x = 1)
  (hx_var : variance x = 8) :
  let y := x.map (λ xi => 2 * xi + 3)
  mean y = 5 ∧ variance y = 32 :=
by
  let y := x.map (λ xi => 2 * xi + 3)
  sorry

end mean_and_variance_of_y_l133_133564


namespace number_of_groups_l133_133882

theorem number_of_groups (max min c : ℕ) (h_max : max = 140) (h_min : min = 50) (h_c : c = 10) : 
  (max - min) / c + 1 = 10 := 
by
  sorry

end number_of_groups_l133_133882


namespace new_credit_card_balance_l133_133499

theorem new_credit_card_balance (i g x r n : ℝ)
    (h_i : i = 126)
    (h_g : g = 60)
    (h_x : x = g / 2)
    (h_r : r = 45)
    (h_n : n = (i + g + x) - r) :
    n = 171 :=
sorry

end new_credit_card_balance_l133_133499


namespace symmetric_points_origin_l133_133185

theorem symmetric_points_origin (a b : ℝ) (h1 : 1 = -b) (h2 : a = 2) : a + b = 1 := by
  sorry

end symmetric_points_origin_l133_133185


namespace functional_linear_solution_l133_133301

variable (f : ℝ → ℝ)

theorem functional_linear_solution (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) : 
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_linear_solution_l133_133301


namespace sine_product_inequality_l133_133840

theorem sine_product_inequality :
  (1 / 8 : ℝ) < (Real.sin (20 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) * Real.sin (70 * Real.pi / 180)) ∧
                (Real.sin (20 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) * Real.sin (70 * Real.pi / 180)) < (1 / 4 : ℝ) :=
sorry

end sine_product_inequality_l133_133840


namespace hours_of_rain_l133_133757

def totalHours : ℕ := 9
def noRainHours : ℕ := 5
def rainHours : ℕ := totalHours - noRainHours

theorem hours_of_rain : rainHours = 4 := by
  sorry

end hours_of_rain_l133_133757


namespace bounded_poly_constant_l133_133196

theorem bounded_poly_constant (P : Polynomial ℤ) (B : ℕ) (h_bounded : ∀ x : ℤ, abs (P.eval x) ≤ B) : 
  P.degree = 0 :=
sorry

end bounded_poly_constant_l133_133196


namespace count_multiples_of_12_l133_133320

theorem count_multiples_of_12 (a b : ℤ) (h1 : 15 < a) (h2 : b < 205) (h3 : ∃ k : ℤ, a = 12 * k) (h4 : ∃ k : ℤ, b = 12 * k) : 
  ∃ n : ℕ, n = 16 := 
by 
  sorry

end count_multiples_of_12_l133_133320


namespace find_track_circumference_l133_133311

noncomputable def track_circumference : ℝ := 720

theorem find_track_circumference
  (A B : ℝ)
  (uA uB : ℝ)
  (h1 : A = 0)
  (h2 : B = track_circumference / 2)
  (h3 : ∀ t : ℝ, A + uA * t = B + uB * t ↔ t = 150 / uB)
  (h4 : ∀ t : ℝ, A + uA * t = B + uB * t ↔ t = (track_circumference - 90) / uA)
  (h5 : ∀ t : ℝ, A + uA * t = B + uB * t ↔ t = 1.5 * track_circumference / uA) :
  track_circumference = 720 :=
by sorry

end find_track_circumference_l133_133311


namespace children_working_initially_l133_133023

theorem children_working_initially (W C : ℝ) (n : ℕ) 
  (h1 : 10 * W = 1 / 5) 
  (h2 : n * C = 1 / 10) 
  (h3 : 5 * W + 10 * C = 1 / 5) : 
  n = 10 :=
by
  sorry

end children_working_initially_l133_133023


namespace range_of_a_l133_133386

theorem range_of_a (a : ℝ) :
  (a + 1)^2 > (3 - 2 * a)^2 ↔ (2 / 3) < a ∧ a < 4 :=
sorry

end range_of_a_l133_133386


namespace ambiguous_dates_count_l133_133561

theorem ambiguous_dates_count : 
  ∃ n : ℕ, n = 132 ∧ ∀ d m : ℕ, 1 ≤ d ∧ d ≤ 31 ∧ 1 ≤ m ∧ m ≤ 12 →
  ((d ≥ 1 ∧ d ≤ 12 ∧ m ≥ 1 ∧ m ≤ 12) → n = 132)
  :=
by 
  let ambiguous_days := 12 * 12
  let non_ambiguous_days := 12
  let total_ambiguous := ambiguous_days - non_ambiguous_days
  use total_ambiguous
  sorry

end ambiguous_dates_count_l133_133561


namespace simplify_fraction_l133_133381

variable {a b c : ℝ} -- assuming a, b, c are real numbers

theorem simplify_fraction (hc : a + b + c ≠ 0) :
  (a^2 + b^2 - c^2 + 2 * a * b) / (a^2 + c^2 - b^2 + 2 * a * c) = (a + b - c) / (a - b + c) :=
sorry

end simplify_fraction_l133_133381


namespace third_circle_radius_l133_133176

theorem third_circle_radius (r1 r2 d : ℝ) (τ : ℝ) (h1: r1 = 1) (h2: r2 = 9) (h3: d = 17) : 
  τ = 225 / 64 :=
by
  sorry

end third_circle_radius_l133_133176


namespace no_triangle_satisfies_condition_l133_133243

theorem no_triangle_satisfies_condition (x y z : ℝ) (h_tri : x + y > z ∧ x + z > y ∧ y + z > x) :
  x^3 + y^3 + z^3 ≠ (x + y) * (y + z) * (z + x) :=
by
  sorry

end no_triangle_satisfies_condition_l133_133243


namespace probability_not_snowing_l133_133841

  -- Define the probability that it will snow tomorrow
  def P_snowing : ℚ := 2 / 5

  -- Define the probability that it will not snow tomorrow
  def P_not_snowing : ℚ := 1 - P_snowing

  -- Theorem stating the required proof
  theorem probability_not_snowing : P_not_snowing = 3 / 5 :=
  by 
    -- Proof would go here
    sorry
  
end probability_not_snowing_l133_133841


namespace total_vessels_l133_133371

open Nat

theorem total_vessels (x y z w : ℕ) (hx : x > 0) (hy : y > x) (hz : z > y) (hw : w > z) :
  ∃ total : ℕ, total = x * (2 * y + 1) + z * (1 + 1 / w) := sorry

end total_vessels_l133_133371


namespace value_of_f_f_2_l133_133069

def f (x : ℝ) : ℝ := 2 * x^3 - 4 * x^2 + 3 * x - 1

theorem value_of_f_f_2 : f (f 2) = 164 := by
  sorry

end value_of_f_f_2_l133_133069


namespace brian_gallons_usage_l133_133147

/-
Brian’s car gets 20 miles per gallon. 
On his last trip, he traveled 60 miles. 
How many gallons of gas did he use?
-/

theorem brian_gallons_usage (miles_per_gallon : ℝ) (total_miles : ℝ) (gallons_used : ℝ) 
    (h1 : miles_per_gallon = 20) 
    (h2 : total_miles = 60) 
    (h3 : gallons_used = total_miles / miles_per_gallon) : 
    gallons_used = 3 := 
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end brian_gallons_usage_l133_133147


namespace gcd_of_4410_and_10800_l133_133157

theorem gcd_of_4410_and_10800 : Nat.gcd 4410 10800 = 90 := 
by 
  sorry

end gcd_of_4410_and_10800_l133_133157


namespace problem1_problem2_l133_133887

theorem problem1 (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  abs ((a + b) / (a - b)) + abs ((b + c) / (b - c)) + abs ((c + a) / (c - a)) ≥ 2 :=
sorry

theorem problem2 (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  abs ((a + b) / (a - b)) + abs ((b + c) / (b - c)) + abs ((c + a) / (c - a)) > 3 :=
sorry

end problem1_problem2_l133_133887


namespace total_cost_is_225_l133_133400

def total_tickets : ℕ := 29
def cost_7_dollar_ticket : ℕ := 7
def cost_9_dollar_ticket : ℕ := 9
def number_of_9_dollar_tickets : ℕ := 11
def number_of_7_dollar_tickets : ℕ := total_tickets - number_of_9_dollar_tickets
def total_cost : ℕ := (number_of_9_dollar_tickets * cost_9_dollar_ticket) + (number_of_7_dollar_tickets * cost_7_dollar_ticket)

theorem total_cost_is_225 : total_cost = 225 := by
  sorry

end total_cost_is_225_l133_133400


namespace water_heater_ratio_l133_133556

variable (Wallace_capacity : ℕ) (Catherine_capacity : ℕ)
variable (Wallace_fullness : ℚ := 3/4) (Catherine_fullness : ℚ := 3/4)
variable (total_water : ℕ := 45)

theorem water_heater_ratio :
  Wallace_capacity = 40 →
  (Wallace_fullness * Wallace_capacity : ℚ) + (Catherine_fullness * Catherine_capacity : ℚ) = total_water →
  ((Wallace_capacity : ℚ) / (Catherine_capacity : ℚ)) = 2 :=
by
  sorry

end water_heater_ratio_l133_133556


namespace no_14_consecutive_divisible_by_2_to_11_l133_133999

theorem no_14_consecutive_divisible_by_2_to_11 :
  ¬ ∃ (a : ℕ), ∀ i, i < 14 → ∃ p, Nat.Prime p ∧ 2 ≤ p ∧ p ≤ 11 ∧ (a + i) % p = 0 :=
by sorry

end no_14_consecutive_divisible_by_2_to_11_l133_133999


namespace number_of_boys_in_class_l133_133457

theorem number_of_boys_in_class 
  (n : ℕ)
  (average_height : ℝ)
  (incorrect_height : ℝ)
  (correct_height : ℝ)
  (actual_average_height : ℝ)
  (initial_average_height : average_height = 185)
  (incorrect_record : incorrect_height = 166)
  (correct_record : correct_height = 106)
  (actual_avg : actual_average_height = 183) 
  (total_height_incorrect : ℝ) 
  (total_height_correct : ℝ) 
  (total_height_eq : total_height_incorrect = 185 * n)
  (correct_total_height_eq : total_height_correct = 185 * n - (incorrect_height - correct_height))
  (actual_total_height_eq : total_height_correct = actual_average_height * n) :
  n = 30 :=
by
  sorry

end number_of_boys_in_class_l133_133457


namespace area_square_given_diagonal_l133_133843

theorem area_square_given_diagonal (d : ℝ) (h : d = 16) : (∃ A : ℝ, A = 128) :=
by 
  sorry

end area_square_given_diagonal_l133_133843


namespace number_of_tires_slashed_l133_133312

-- Definitions based on conditions
def cost_per_tire : ℤ := 250
def cost_window : ℤ := 700
def total_cost : ℤ := 1450

-- Proof statement
theorem number_of_tires_slashed : ∃ T : ℤ, cost_per_tire * T + cost_window = total_cost ∧ T = 3 := 
sorry

end number_of_tires_slashed_l133_133312


namespace scaled_multiplication_l133_133095

theorem scaled_multiplication (h : 268 * 74 = 19832) : 2.68 * 0.74 = 1.9832 :=
by
  -- proof steps would go here
  sorry

end scaled_multiplication_l133_133095


namespace deepak_age_l133_133308

variable (A D : ℕ)

theorem deepak_age (h1 : A / D = 2 / 3) (h2 : A + 5 = 25) : D = 30 :=
sorry

end deepak_age_l133_133308


namespace shopping_problem_l133_133372

theorem shopping_problem
  (D S H N : ℝ)
  (h1 : (D - (D / 2 - 10)) + (S - 0.85 * S) + (H - (H - 30)) + (N - N) = 120)
  (T_sale : ℝ := (D / 2 - 10) + 0.85 * S + (H - 30) + N) :
  (120 + 0.10 * T_sale = 0.10 * 1200) →
  D + S + H + N = 1200 :=
by
  sorry

end shopping_problem_l133_133372


namespace remaining_soup_can_feed_adults_l133_133712

-- Define initial conditions
def cans_per_soup_for_children : ℕ := 6
def cans_per_soup_for_adults : ℕ := 4
def initial_cans : ℕ := 8
def children_to_feed : ℕ := 24

-- Define the problem statement in Lean
theorem remaining_soup_can_feed_adults :
  (initial_cans - (children_to_feed / cans_per_soup_for_children)) * cans_per_soup_for_adults = 16 := by
  sorry

end remaining_soup_can_feed_adults_l133_133712


namespace Cornelia_current_age_l133_133929

theorem Cornelia_current_age (K : ℕ) (C : ℕ) (h1 : K = 20) (h2 : C + 10 = 3 * (K + 10)) : C = 80 :=
by
  sorry

end Cornelia_current_age_l133_133929


namespace interval_increase_for_k_eq_2_range_of_k_if_f_leq_0_l133_133002

noncomputable def f (x k : ℝ) : ℝ := Real.log x - k * x + 1

theorem interval_increase_for_k_eq_2 :
  ∃ k : ℝ, k = 2 → 
  ∃ a b : ℝ, 0 < b ∧ b = 1 / 2 ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 / 2 → (Real.log x - 2 * x + 1 < Real.log x - 2 * x + 1)) := 
sorry

theorem range_of_k_if_f_leq_0 :
  ∀ (k : ℝ), (∀ x : ℝ, 0 < x → Real.log x - k * x + 1 ≤ 0) →
  ∃ k_min : ℝ, k_min = 1 ∧ k ≥ k_min :=
sorry

end interval_increase_for_k_eq_2_range_of_k_if_f_leq_0_l133_133002


namespace max_value_2ab_3bc_lemma_l133_133425

noncomputable def max_value_2ab_3bc (a b c : ℝ) : ℝ :=
  2 * a * b + 3 * b * c

theorem max_value_2ab_3bc_lemma
  (a b c : ℝ)
  (ha : 0 ≤ a)
  (hb : 0 ≤ b)
  (hc : 0 ≤ c)
  (h : a^2 + b^2 + c^2 = 2) :
  max_value_2ab_3bc a b c ≤ 3 :=
sorry

end max_value_2ab_3bc_lemma_l133_133425


namespace interior_diagonals_sum_l133_133920

theorem interior_diagonals_sum (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 52)
  (h2 : 2 * (a * b + b * c + c * a) = 118) :
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 51 := 
by
  sorry

end interior_diagonals_sum_l133_133920


namespace largest_three_digit_sum_fifteen_l133_133323

theorem largest_three_digit_sum_fifteen : ∃ (a b c : ℕ), (a = 9 ∧ b = 6 ∧ c = 0 ∧ 100 * a + 10 * b + c = 960 ∧ a + b + c = 15 ∧ a < 10 ∧ b < 10 ∧ c < 10) := by
  sorry

end largest_three_digit_sum_fifteen_l133_133323


namespace alfred_gain_percent_l133_133361

theorem alfred_gain_percent :
  let purchase_price := 4700
  let repair_costs := 800
  let selling_price := 5800
  let total_cost := purchase_price + repair_costs
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  gain_percent = 5.45 := 
by
  sorry

end alfred_gain_percent_l133_133361


namespace polygon_coloring_l133_133685

theorem polygon_coloring (n m : ℕ) (h1 : n ≥ 3) (h2 : m ≥ 3) :
    ∃ b_n : ℕ, b_n = (m - 1) * ((m - 1) ^ (n - 1) + (-1 : ℤ) ^ n) :=
sorry

end polygon_coloring_l133_133685


namespace find_x0_range_l133_133512

variable {x y x0 : ℝ}

def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

def angle_condition (x0 : ℝ) : Prop :=
  let OM := Real.sqrt (x0^2 + 3)
  OM ≤ 2

theorem find_x0_range (h1 : circle_eq x y) (h2 : angle_condition x0) :
  -1 ≤ x0 ∧ x0 ≤ 1 := 
sorry

end find_x0_range_l133_133512


namespace transformed_equation_correct_l133_133822
-- Import the necessary library

-- Define the original equation of the parabola
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translation functions for the transformations
def translate_right (x : ℝ) : ℝ := x - 1
def translate_down (y : ℝ) : ℝ := y - 3

-- Define the transformed parabola equation
def transformed_parabola (x : ℝ) : ℝ := -2 * (translate_right x)^2 |> translate_down

-- The theorem stating the transformed equation
theorem transformed_equation_correct :
  ∀ x, transformed_parabola x = -2 * (x - 1)^2 - 3 :=
by { sorry }

end transformed_equation_correct_l133_133822


namespace sqrt_mul_neg_eq_l133_133276

theorem sqrt_mul_neg_eq : - (Real.sqrt 2) * (Real.sqrt 7) = - (Real.sqrt 14) := sorry

end sqrt_mul_neg_eq_l133_133276


namespace abcd_value_l133_133127

noncomputable def abcd_eval (a b c d : ℂ) : ℂ := a * b * c * d

theorem abcd_value (a b c d : ℂ) 
  (h1 : a + b + c + d = 5)
  (h2 : (5 - a)^4 + (5 - b)^4 + (5 - c)^4 + (5 - d)^4 = 125)
  (h3 : (a + b)^4 + (b + c)^4 + (c + d)^4 + (d + a)^4 + (a + c)^4 + (b + d)^4 = 1205)
  (h4 : a^4 + b^4 + c^4 + d^4 = 25) : 
  abcd_eval a b c d = 70 := 
sorry

end abcd_value_l133_133127


namespace convert_rectangular_to_polar_l133_133805

theorem convert_rectangular_to_polar (x y : ℝ) (h₁ : x = -2) (h₂ : y = -2) : 
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ) = (2 * Real.sqrt 2, 5 * Real.pi / 4) := by
  sorry

end convert_rectangular_to_polar_l133_133805


namespace compute_fg_neg_2_l133_133571

def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x^2 + 4 * x + 4

theorem compute_fg_neg_2 : f (g (-2)) = -5 :=
by
-- sorry is used to skip the proof
sorry

end compute_fg_neg_2_l133_133571


namespace diana_owes_amount_l133_133341

def principal : ℝ := 60
def rate : ℝ := 0.06
def time : ℝ := 1
def interest := principal * rate * time
def original_amount := principal
def total_amount := original_amount + interest

theorem diana_owes_amount :
  total_amount = 63.60 :=
by
  -- Placeholder for actual proof
  sorry

end diana_owes_amount_l133_133341


namespace evaluate_number_l133_133162

theorem evaluate_number (n : ℝ) (h : 22 + Real.sqrt (-4 + 6 * 4 * n) = 24) : n = 1 / 3 :=
by
  sorry

end evaluate_number_l133_133162


namespace avg_median_max_k_m_r_s_t_l133_133703

theorem avg_median_max_k_m_r_s_t (
  k m r s t : ℕ 
) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t)
  (h5 : 5 * 16 = k + m + r + s + t)
  (h6 : r = 17) : 
  t = 42 :=
by
  sorry

end avg_median_max_k_m_r_s_t_l133_133703


namespace jenna_stamp_division_l133_133626

theorem jenna_stamp_division (a b c : ℕ) (h₁ : a = 945) (h₂ : b = 1260) (h₃ : c = 630) :
  Nat.gcd (Nat.gcd a b) c = 105 :=
by
  rw [h₁, h₂, h₃]
  -- Now we need to prove Nat.gcd (Nat.gcd 945 1260) 630 = 105
  sorry

end jenna_stamp_division_l133_133626


namespace angle_CBE_minimal_l133_133785

theorem angle_CBE_minimal
    (ABC ABD DBE: ℝ)
    (h1: ABC = 40)
    (h2: ABD = 28)
    (h3: DBE = 10) : 
    CBE = 2 :=
by
  sorry

end angle_CBE_minimal_l133_133785


namespace max_radius_of_inscribable_circle_l133_133171

theorem max_radius_of_inscribable_circle
  (AB BC CD DA : ℝ) (x y z w : ℝ)
  (h1 : AB = 10) (h2 : BC = 12) (h3 : CD = 8) (h4 : DA = 14)
  (h5 : x + y = 10) (h6 : y + z = 12)
  (h7 : z + w = 8) (h8 : w + x = 14)
  (h9 : x + z = y + w) :
  ∃ r : ℝ, r = Real.sqrt 24.75 :=
by
  sorry

end max_radius_of_inscribable_circle_l133_133171


namespace maximize_product_l133_133538

theorem maximize_product (x y z : ℝ) (h1 : x ≥ 20) (h2 : y ≥ 40) (h3 : z ≥ 1675) (h4 : x + y + z = 2015) :
  x * y * z ≤ 721480000 / 27 :=
by sorry

end maximize_product_l133_133538


namespace find_x_plus_y_l133_133040

theorem find_x_plus_y (x y : ℝ) (hx : |x| + x + y = 14) (hy : x + |y| - y = 16) : x + y = 26 / 5 := 
sorry

end find_x_plus_y_l133_133040


namespace expectation_fish_l133_133823

noncomputable def fish_distribution : ℕ → ℚ → ℚ → ℚ → ℚ :=
  fun N a b c => (a / b) * (1 - (c / (a + b + c) ^ N))

def x_distribution : ℚ := 0.18
def y_distribution : ℚ := 0.02
def other_distribution : ℚ := 0.80
def total_fish : ℕ := 10

theorem expectation_fish :
  fish_distribution total_fish x_distribution y_distribution other_distribution = 1.6461 :=
  by
    sorry

end expectation_fish_l133_133823


namespace initial_dog_cat_ratio_l133_133317

theorem initial_dog_cat_ratio (C : ℕ) :
  75 / (C + 20) = 15 / 11 →
  (75 / C) = 15 / 7 :=
by
  sorry

end initial_dog_cat_ratio_l133_133317


namespace inverse_of_5_mod_34_l133_133607

theorem inverse_of_5_mod_34 : ∃ x : ℕ, (5 * x) % 34 = 1 ∧ 0 ≤ x ∧ x < 34 :=
by
  use 7
  have h : (5 * 7) % 34 = 1 := by sorry
  exact ⟨h, by norm_num, by norm_num⟩

end inverse_of_5_mod_34_l133_133607


namespace school_A_original_students_l133_133818

theorem school_A_original_students 
  (x y : ℕ) 
  (h1 : x + y = 864) 
  (h2 : x - 32 = y + 80) : 
  x = 488 := 
by 
  sorry

end school_A_original_students_l133_133818


namespace polynomial_zero_pairs_l133_133969

theorem polynomial_zero_pairs (r s : ℝ) :
  (∀ x : ℝ, (x = 0 ∨ x = 0) ↔ x^2 - 2 * r * x + r = 0) ∧
  (∀ x : ℝ, (x = 0 ∨ x = 0 ∨ x = 0) ↔ 27 * x^3 - 27 * r * x^2 + s * x - r^6 = 0) → 
  (r, s) = (0, 0) ∨ (r, s) = (1, 9) :=
by
  sorry

end polynomial_zero_pairs_l133_133969


namespace new_sample_variance_l133_133718

-- Definitions based on conditions
def sample_size (original : Nat) : Prop := original = 7
def sample_average (original : ℝ) : Prop := original = 5
def sample_variance (original : ℝ) : Prop := original = 2
def new_data_point (point : ℝ) : Prop := point = 5

-- Statement to be proved
theorem new_sample_variance (original_size : Nat) (original_avg : ℝ) (original_var : ℝ) (new_point : ℝ) 
  (h₁ : sample_size original_size) 
  (h₂ : sample_average original_avg) 
  (h₃ : sample_variance original_var) 
  (h₄ : new_data_point new_point) : 
  (8 * original_var + 0) / 8 = 7 / 4 := 
by 
  sorry

end new_sample_variance_l133_133718


namespace prob_sum_seven_prob_two_fours_l133_133344

-- Definitions and conditions
def total_outcomes : ℕ := 36
def outcomes_sum_seven : ℕ := 6
def outcomes_two_fours : ℕ := 1

-- Proof problem for question 1
theorem prob_sum_seven : outcomes_sum_seven / total_outcomes = 1 / 6 :=
by
  sorry

-- Proof problem for question 2
theorem prob_two_fours : outcomes_two_fours / total_outcomes = 1 / 36 :=
by
  sorry

end prob_sum_seven_prob_two_fours_l133_133344


namespace find_value_of_expression_l133_133428

variable {a : ℕ → ℤ}

-- Define arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variable (h1 : a 1 + 3 * a 8 + a 15 = 120)
variable (h2 : is_arithmetic_sequence a)

-- Theorem to be proved
theorem find_value_of_expression : 2 * a 6 - a 4 = 24 :=
sorry

end find_value_of_expression_l133_133428


namespace learn_at_least_537_words_l133_133788

theorem learn_at_least_537_words (total_words : ℕ) (guess_percentage : ℝ) (required_percentage : ℝ) :
  total_words = 600 → guess_percentage = 0.05 → required_percentage = 0.90 → 
  ∀ (words_learned : ℕ), words_learned ≥ 537 → 
  (words_learned + guess_percentage * (total_words - words_learned)) / total_words ≥ required_percentage :=
by
  intros h_total_words h_guess_percentage h_required_percentage words_learned h_words_learned
  sorry

end learn_at_least_537_words_l133_133788


namespace solution_set_of_inequality_l133_133828

theorem solution_set_of_inequality (x : ℝ) : (x * |x - 1| > 0) ↔ (0 < x ∧ x < 1 ∨ 1 < x) := 
by
  sorry

end solution_set_of_inequality_l133_133828


namespace ratio_a_to_d_l133_133862

theorem ratio_a_to_d (a b c d : ℚ) 
  (h1 : a / b = 5 / 4) 
  (h2 : b / c = 2 / 3) 
  (h3 : c / d = 3 / 5) : 
  a / d = 1 / 2 :=
sorry

end ratio_a_to_d_l133_133862


namespace calculate_difference_l133_133970

theorem calculate_difference (x y : ℝ) (h1 : x + y = 520) (h2 : x / y = 0.75) : y - x = 74 :=
by
  sorry

end calculate_difference_l133_133970


namespace find_common_ratio_l133_133465

noncomputable def common_ratio_of_geometric_sequence (a : ℕ → ℝ) (d : ℝ) 
  (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : ∃ (r : ℝ), r ≠ 0 ∧ a 3 ^ 2 = a 1 * a 9) : ℝ :=
3

theorem find_common_ratio 
( a : ℕ → ℝ) 
( d : ℝ) 
(h1 : d ≠ 0)
(h2 : ∀ n, a (n + 1) = a n + d)
(h3 : ∃ (r : ℝ), r ≠ 0 ∧ a 3 ^ 2 = a 1 * a 9) :
common_ratio_of_geometric_sequence a d h1 h2 h3 = 3 :=
sorry

end find_common_ratio_l133_133465


namespace rectangle_in_right_triangle_dimensions_l133_133582

theorem rectangle_in_right_triangle_dimensions :
  ∀ (DE EF DF x y : ℝ),
  DE = 6 → EF = 8 → DF = 10 →
  -- Assuming isosceles right triangle (interchange sides for the proof)
  ∃ (G H I J : ℝ),
  (G = 0 ∧ H = 0 ∧ I = y ∧ J = x ∧ x * y = GH * GI) → -- Rectangle GH parallel to DE
  (x = 10 / 8 * y) →
  ∃ (GH GI : ℝ), 
  GH = 8 / 8.33 ∧ GI = 6.67 / 8.33 →
  (x = 25 / 3 ∧ y = 40 / 6) :=
by
  sorry

end rectangle_in_right_triangle_dimensions_l133_133582


namespace smallest_four_digit_number_divisible_by_40_l133_133660

theorem smallest_four_digit_number_divisible_by_40 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 40 = 0 ∧ ∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ m % 40 = 0 → n <= m :=
by
  use 1000
  sorry

end smallest_four_digit_number_divisible_by_40_l133_133660


namespace average_speed_l133_133755

theorem average_speed (d1 d2 d3 d4 d5 t: ℕ) 
  (h1: d1 = 120) 
  (h2: d2 = 70) 
  (h3: d3 = 90) 
  (h4: d4 = 110) 
  (h5: d5 = 80) 
  (total_time: t = 5): 
  (d1 + d2 + d3 + d4 + d5) / t = 94 := 
by 
  -- proof will go here
  sorry

end average_speed_l133_133755


namespace total_theme_parks_l133_133872

theorem total_theme_parks 
  (J V M N : ℕ) 
  (hJ : J = 35)
  (hV : V = J + 40)
  (hM : M = J + 60)
  (hN : N = 2 * M) 
  : J + V + M + N = 395 :=
sorry

end total_theme_parks_l133_133872


namespace eval_expression_l133_133273

-- Define the given expression
def given_expression : ℤ := -( (16 / 2) * 12 - 75 + 4 * (2 * 5) + 25 )

-- State the desired result in a theorem
theorem eval_expression : given_expression = -86 := by
  -- Skipping the proof as per instructions
  sorry

end eval_expression_l133_133273


namespace sum_and_ratio_l133_133589

theorem sum_and_ratio (x y : ℝ) (h1 : x + y = 480) (h2 : x / y = 0.8) : y - x = 53.34 :=
by
  sorry

end sum_and_ratio_l133_133589


namespace least_possible_sum_of_bases_l133_133595

theorem least_possible_sum_of_bases : 
  ∃ (c d : ℕ), (2 * c + 9 = 9 * d + 2) ∧ (c + d = 13) :=
by
  sorry

end least_possible_sum_of_bases_l133_133595


namespace A_star_B_eq_l133_133383

def A : Set ℝ := {x | ∃ y, y = 2 * x - x^2}
def B : Set ℝ := {y | ∃ x, y = 2^x ∧ x > 0}
def A_star_B : Set ℝ := {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

theorem A_star_B_eq : A_star_B = {x | x ≤ 1} :=
by {
  sorry
}

end A_star_B_eq_l133_133383


namespace megan_popsicles_consumed_l133_133606

noncomputable def popsicles_consumed_in_time_period (time: ℕ) (interval: ℕ) : ℕ :=
  (time / interval)

theorem megan_popsicles_consumed:
  popsicles_consumed_in_time_period 315 30 = 10 :=
by
  sorry

end megan_popsicles_consumed_l133_133606


namespace number_difference_l133_133834

theorem number_difference (x y : ℕ) (h₁ : x + y = 41402) (h₂ : ∃ k : ℕ, x = 100 * k) (h₃ : y = x / 100) : x - y = 40590 :=
sorry

end number_difference_l133_133834


namespace units_digit_first_four_composite_is_eight_l133_133289

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l133_133289


namespace total_sand_correct_l133_133193

-- Define the conditions as variables and equations:
variables (x : ℕ) -- original days scheduled to complete
variables (total_sand : ℕ) -- total amount of sand in tons

-- Define the conditions in the problem:
def original_daily_amount := 15  -- tons per day as scheduled
def actual_daily_amount := 20  -- tons per day in reality
def days_ahead := 3  -- days finished ahead of schedule

-- Equation representing the planned and actual transportation:
def planned_sand := original_daily_amount * x
def actual_sand := actual_daily_amount * (x - days_ahead)

-- The goal is to prove:
theorem total_sand_correct : planned_sand = actual_sand → total_sand = 180 :=
by
  sorry

end total_sand_correct_l133_133193


namespace quadratic_form_proof_l133_133817

theorem quadratic_form_proof (k : ℝ) (a b c : ℝ) (h1 : 8*k^2 - 16*k + 28 = a * (k + b)^2 + c) (h2 : a = 8) (h3 : b = -1) (h4 : c = 20) : c / b = -20 :=
by {
  sorry
}

end quadratic_form_proof_l133_133817


namespace smallest_multiple_of_37_smallest_multiple_of_37_verification_l133_133963

theorem smallest_multiple_of_37 (x : ℕ) (h : 37 * x % 97 = 3) :
  x = 15 := sorry

theorem smallest_multiple_of_37_verification :
  37 * 15 = 555 := rfl

end smallest_multiple_of_37_smallest_multiple_of_37_verification_l133_133963


namespace cats_to_dogs_ratio_l133_133967

theorem cats_to_dogs_ratio (cats dogs : ℕ) (h1 : 2 * dogs = 3 * cats) (h2 : cats = 14) : dogs = 21 :=
by
  sorry

end cats_to_dogs_ratio_l133_133967


namespace sequence_property_l133_133567

theorem sequence_property
  (b : ℝ) (h₀ : b > 0)
  (u : ℕ → ℝ)
  (h₁ : u 1 = b)
  (h₂ : ∀ n ≥ 1, u (n + 1) = 1 / (2 - u n)) :
  u 10 = (4 * b - 3) / (6 * b - 5) :=
by
  sorry

end sequence_property_l133_133567


namespace range_of_a_l133_133050

noncomputable def p (a : ℝ) : Prop := 
  (1 + a)^2 + (1 - a)^2 < 4

noncomputable def q (a : ℝ) : Prop := 
  ∀ x : ℝ, x^2 + a * x + 1 ≥ 0

theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) ∧ (p a ∨ q a) ↔ (-2 ≤ a ∧ a ≤ -1) ∨ (1 ≤ a ∧ a ≤ 2) := 
by
  sorry

end range_of_a_l133_133050


namespace monotonic_increasing_on_interval_min_value_on_interval_max_value_on_interval_l133_133260

noncomputable def f (x : ℝ) : ℝ := 1 - (3 / (x + 2))

theorem monotonic_increasing_on_interval :
  ∀ (x₁ x₂ : ℝ), 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5 → f x₁ < f x₂ := sorry

theorem min_value_on_interval :
  ∃ (x : ℝ), x = 3 ∧ f x = 2 / 5 := sorry

theorem max_value_on_interval :
  ∃ (x : ℝ), x = 5 ∧ f x = 4 / 7 := sorry

end monotonic_increasing_on_interval_min_value_on_interval_max_value_on_interval_l133_133260


namespace simplify_polynomial_l133_133120

def poly1 (x : ℝ) : ℝ := 5 * x^12 - 3 * x^9 + 6 * x^8 - 2 * x^7
def poly2 (x : ℝ) : ℝ := 7 * x^12 + 2 * x^11 - x^9 + 4 * x^7 + 2 * x^5 - x + 3
def expected (x : ℝ) : ℝ := 12 * x^12 + 2 * x^11 - 4 * x^9 + 6 * x^8 + 2 * x^7 + 2 * x^5 - x + 3

theorem simplify_polynomial (x : ℝ) : poly1 x + poly2 x = expected x :=
  by sorry

end simplify_polynomial_l133_133120


namespace line_intersects_y_axis_at_0_2_l133_133228

theorem line_intersects_y_axis_at_0_2 :
  ∃ y : ℝ, (2, 8) ≠ (4, 14) ∧ ∀ x: ℝ, (3 * x + y = 2) ∧ x = 0 → y = 2 :=
by
  sorry

end line_intersects_y_axis_at_0_2_l133_133228


namespace middle_number_of_consecutive_squares_l133_133197

theorem middle_number_of_consecutive_squares (x : ℕ ) (h : x^2 + (x+1)^2 + (x+2)^2 = 2030) : x + 1 = 26 :=
sorry

end middle_number_of_consecutive_squares_l133_133197


namespace right_triangle_perimeter_l133_133490

theorem right_triangle_perimeter 
  (a b c : ℕ) (h : a = 11) (h1 : a * a + b * b = c * c) (h2 : a < c) : a + b + c = 132 :=
  sorry

end right_triangle_perimeter_l133_133490


namespace particles_probability_computation_l133_133203

theorem particles_probability_computation : 
  let L0 := 32
  let R0 := 68
  let N := 100
  let a := 1
  let b := 2
  let P_all_on_left := (a:ℚ) / b
  100 * a + b = 102 := by
  sorry

end particles_probability_computation_l133_133203


namespace complete_the_square_l133_133577

theorem complete_the_square (x : ℝ) : (x^2 - 8*x + 15 = 0) → ((x - 4)^2 = 1) :=
by
  intro h
  have eq1 : x^2 - 8*x + 15 = 0 := h
  sorry

end complete_the_square_l133_133577


namespace total_attendance_l133_133668

theorem total_attendance (first_concert : ℕ) (second_concert : ℕ) (third_concert : ℕ) :
  first_concert = 65899 →
  second_concert = first_concert + 119 →
  third_concert = 2 * second_concert →
  first_concert + second_concert + third_concert = 263953 :=
by
  intros h_first h_second h_third
  rw [h_first, h_second, h_third]
  sorry

end total_attendance_l133_133668


namespace binom_20_10_eq_184756_l133_133373

theorem binom_20_10_eq_184756 (h1 : Nat.choose 18 8 = 43758)
                               (h2 : Nat.choose 18 9 = 48620)
                               (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 :=
by
  sorry

end binom_20_10_eq_184756_l133_133373


namespace calculate_expression_l133_133281

theorem calculate_expression : 
  2 - 1 / (2 - 1 / (2 + 2)) = 10 / 7 := 
by sorry

end calculate_expression_l133_133281


namespace curve_intersection_l133_133932

noncomputable def C1 (t : ℝ) (a : ℝ) : ℝ × ℝ :=
  (2 * t + 2 * a, -t)

noncomputable def C2 (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.sin θ, 1 + 2 * Real.cos θ)

theorem curve_intersection (a : ℝ) :
  (∃ t θ : ℝ, C1 t a = C2 θ) ↔ 1 - Real.sqrt 5 ≤ a ∧ a ≤ 1 + Real.sqrt 5 :=
sorry

end curve_intersection_l133_133932


namespace unique_solution_l133_133266

theorem unique_solution (a b : ℤ) (h : a > b ∧ b > 0) (hab : a * b - a - b = 1) : a = 3 ∧ b = 2 := by
  sorry

end unique_solution_l133_133266


namespace sum_of_common_ratios_l133_133504

variable {k p r : ℝ}

-- Condition 1: geometric sequences with distinct common ratios
-- Condition 2: a_3 - b_3 = 3(a_2 - b_2)
def geometric_sequences (k p r : ℝ) : Prop :=
  (k ≠ 0) ∧ (p ≠ r) ∧ (k * p^2 - k * r^2 = 3 * (k * p - k * r))

theorem sum_of_common_ratios (k p r : ℝ) (h : geometric_sequences k p r) : p + r = 3 :=
by
  sorry

end sum_of_common_ratios_l133_133504


namespace sin_cos_identity_l133_133476

theorem sin_cos_identity (θ : Real) (h1 : 0 < θ ∧ θ < π) (h2 : Real.sin θ * Real.cos θ = - (1/8)) :
  Real.sin (2 * Real.pi + θ) - Real.sin ((Real.pi / 2) - θ) = (Real.sqrt 5) / 2 := by
  sorry

end sin_cos_identity_l133_133476


namespace C_is_14_years_younger_than_A_l133_133409

variable (A B C D : ℕ)

-- Conditions
axiom cond1 : A + B = (B + C) + 14
axiom cond2 : B + D = (C + A) + 10
axiom cond3 : D = C + 6

-- To prove
theorem C_is_14_years_younger_than_A : A - C = 14 :=
by
  sorry

end C_is_14_years_younger_than_A_l133_133409


namespace calories_needed_l133_133771

def calories_per_orange : ℕ := 80
def cost_per_orange : ℝ := 1.2
def initial_amount : ℝ := 10
def remaining_amount : ℝ := 4

theorem calories_needed : calories_per_orange * (initial_amount - remaining_amount) / cost_per_orange = 400 := 
by 
  sorry

end calories_needed_l133_133771


namespace segment_EC_length_l133_133153

noncomputable def length_of_segment_EC (a b c : ℕ) (angle_A_deg BC : ℝ) (BD_perp_AC CE_perp_AB : Prop) (angle_DBC_eq_3_angle_ECB : Prop) : ℝ :=
  a * (Real.sqrt b + Real.sqrt c)

theorem segment_EC_length
  (a b c : ℕ)
  (angle_A_deg BC : ℝ)
  (BD_perp_AC CE_perp_AB : Prop)
  (angle_DBC_eq_3_angle_ECB : Prop)
  (h1 : angle_A_deg = 45)
  (h2 : BC = 10)
  (h3 : BD_perp_AC)
  (h4 : CE_perp_AB)
  (h5 : angle_DBC_eq_3_angle_ECB)
  (h6 : length_of_segment_EC a b c angle_A_deg BC BD_perp_AC CE_perp_AB angle_DBC_eq_3_angle_ECB = 5 * (Real.sqrt 3 + Real.sqrt 1)) :
  a + b + c = 9 :=
  by
    sorry

end segment_EC_length_l133_133153


namespace negation_proposition_l133_133094

theorem negation_proposition (x y : ℝ) :
  (¬ (x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔ (x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)) := 
sorry

end negation_proposition_l133_133094


namespace bretschneider_l133_133427

noncomputable def bretschneider_theorem 
  (a b c d m n : ℝ) 
  (A C : ℝ) : Prop :=
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * Real.cos (A + C)

theorem bretschneider (a b c d m n A C : ℝ) :
  bretschneider_theorem a b c d m n A C :=
sorry

end bretschneider_l133_133427


namespace total_age_of_wines_l133_133079

theorem total_age_of_wines (age_carlo_rosi : ℕ) (age_franzia : ℕ) (age_twin_valley : ℕ) 
    (h1 : age_carlo_rosi = 40) (h2 : age_franzia = 3 * age_carlo_rosi) (h3 : age_carlo_rosi = 4 * age_twin_valley) : 
    age_franzia + age_carlo_rosi + age_twin_valley = 170 := 
by
    sorry

end total_age_of_wines_l133_133079


namespace base_b_prime_digits_l133_133972

theorem base_b_prime_digits (b' : ℕ) (h1 : b'^4 ≤ 216) (h2 : 216 < b'^5) : b' = 3 :=
by {
  sorry
}

end base_b_prime_digits_l133_133972


namespace find_number_l133_133057

theorem find_number (x : ℝ) (h : 0.4 * x = 15) : x = 37.5 := by
  sorry

end find_number_l133_133057


namespace domain_of_function_l133_133413

def domain_of_f (x : ℝ) : Prop :=
  (x ≤ 2) ∧ (x ≠ 1)

theorem domain_of_function :
  ∀ x : ℝ, x ∈ { x | (x ≤ 2) ∧ (x ≠ 1) } ↔ domain_of_f x :=
by
  sorry

end domain_of_function_l133_133413


namespace gcd_of_polynomial_and_multiple_of_12600_l133_133782

theorem gcd_of_polynomial_and_multiple_of_12600 (x : ℕ) (h : 12600 ∣ x) : gcd ((5 * x + 7) * (11 * x + 3) * (17 * x + 8) * (4 * x + 5)) x = 840 := by
  sorry

end gcd_of_polynomial_and_multiple_of_12600_l133_133782


namespace Karen_wall_paint_area_l133_133487

theorem Karen_wall_paint_area :
  let height_wall := 10
  let width_wall := 15
  let height_window := 3
  let width_window := 5
  let height_door := 2
  let width_door := 6
  let area_wall := height_wall * width_wall
  let area_window := height_window * width_window
  let area_door := height_door * width_door
  let area_to_paint := area_wall - area_window - area_door
  area_to_paint = 123 := by
{
  sorry
}

end Karen_wall_paint_area_l133_133487


namespace remainder_when_divided_by_x_minus_2_l133_133850

-- Define the polynomial function f(x)
def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 20*x^3 + x^2 - 47*x + 15

-- State the theorem to be proved with the given conditions
theorem remainder_when_divided_by_x_minus_2 :
  f 2 = -11 :=
by 
  -- Proof goes here
  sorry

end remainder_when_divided_by_x_minus_2_l133_133850


namespace isosceles_triangle_side_length_l133_133634

theorem isosceles_triangle_side_length (total_length : ℝ) (one_side_length : ℝ) (remaining_wire : ℝ) (equal_side : ℝ) :
  total_length = 20 → one_side_length = 6 → remaining_wire = total_length - one_side_length → remaining_wire / 2 = equal_side →
  equal_side = 7 :=
by
  intros h_total h_one_side h_remaining h_equal_side
  sorry

end isosceles_triangle_side_length_l133_133634


namespace Doug_money_l133_133052

theorem Doug_money (B D : ℝ) (h1 : B + 2*B + D = 68) (h2 : 2*B = (3/4)*D) : D = 32 := by
  sorry

end Doug_money_l133_133052


namespace total_discount_is_15_l133_133794

structure Item :=
  (price : ℝ)      -- Regular price
  (discount_rate : ℝ) -- Discount rate in decimal form

def t_shirt : Item := {price := 25, discount_rate := 0.3}
def jeans : Item := {price := 75, discount_rate := 0.1}

def discount (item : Item) : ℝ :=
  item.discount_rate * item.price

def total_discount (items : List Item) : ℝ :=
  items.map discount |>.sum

theorem total_discount_is_15 :
  total_discount [t_shirt, jeans] = 15 := by
  sorry

end total_discount_is_15_l133_133794


namespace mb_range_l133_133591

theorem mb_range (m b : ℝ) (hm : m = 3 / 4) (hb : b = -2 / 3) :
  -1 < m * b ∧ m * b < 0 :=
by
  rw [hm, hb]
  sorry

end mb_range_l133_133591


namespace adult_dog_cost_is_100_l133_133155

-- Define the costs for cats, puppies, and dogs.
def cat_cost : ℕ := 50
def puppy_cost : ℕ := 150

-- Define the number of each type of animal.
def number_of_cats : ℕ := 2
def number_of_adult_dogs : ℕ := 3
def number_of_puppies : ℕ := 2

-- The total cost
def total_cost : ℕ := 700

-- Define what needs to be proven: the cost of getting each adult dog ready for adoption.
theorem adult_dog_cost_is_100 (D : ℕ) (h : number_of_cats * cat_cost + number_of_adult_dogs * D + number_of_puppies * puppy_cost = total_cost) : D = 100 :=
by 
  sorry

end adult_dog_cost_is_100_l133_133155


namespace area_of_quadrilateral_l133_133596

theorem area_of_quadrilateral (A B C : ℝ) (triangle1 triangle2 triangle3 quadrilateral : ℝ)
  (hA : A = 5) (hB : B = 9) (hC : C = 9)
  (h_sum : quadrilateral = triangle1 + triangle2 + triangle3)
  (h1 : triangle1 = A)
  (h2 : triangle2 = B)
  (h3 : triangle3 = C) :
  quadrilateral = 40 :=
by
  sorry

end area_of_quadrilateral_l133_133596


namespace least_number_remainder_l133_133653

theorem least_number_remainder (n : ℕ) :
  (n % 7 = 4) ∧ (n % 9 = 4) ∧ (n % 12 = 4) ∧ (n % 18 = 4) → n = 256 :=
by
  sorry

end least_number_remainder_l133_133653


namespace symmetric_point_coordinates_l133_133467

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_about_x_axis (P : Point3D) : Point3D :=
  { x := P.x, y := -P.y, z := -P.z }

theorem symmetric_point_coordinates :
  symmetric_about_x_axis {x := 1, y := 3, z := 6} = {x := 1, y := -3, z := -6} :=
by
  sorry

end symmetric_point_coordinates_l133_133467


namespace largest_possible_M_l133_133946

theorem largest_possible_M (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h_cond : x * y + y * z + z * x = 1) :
    ∃ M, ∀ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y + y * z + z * x = 1 → 
    (x / (1 + yz/x) + y / (1 + zx/y) + z / (1 + xy/z) ≥ M) → 
        M = 3 / (Real.sqrt 3 + 1) :=
by
  sorry        

end largest_possible_M_l133_133946


namespace Sandy_goal_water_l133_133853

-- Definitions based on the conditions in problem a)
def milliliters_per_interval := 500
def time_per_interval := 2
def total_time := 12
def milliliters_to_liters := 1000

-- The goal statement that proves the question == answer given conditions.
theorem Sandy_goal_water : (milliliters_per_interval * (total_time / time_per_interval)) / milliliters_to_liters = 3 := by
  sorry

end Sandy_goal_water_l133_133853


namespace total_num_of_cars_l133_133638

-- Define conditions
def row_from_front := 14
def row_from_left := 19
def row_from_back := 11
def row_from_right := 16

-- Compute total number of rows from front to back
def rows_front_to_back : ℕ := (row_from_front - 1) + 1 + (row_from_back - 1)

-- Compute total number of rows from left to right
def rows_left_to_right : ℕ := (row_from_left - 1) + 1 + (row_from_right - 1)

theorem total_num_of_cars :
  rows_front_to_back = 24 ∧
  rows_left_to_right = 34 ∧
  24 * 34 = 816 :=
by
  sorry

end total_num_of_cars_l133_133638


namespace notepad_days_last_l133_133129

def fold_paper (n : Nat) : Nat := 2 ^ n

def lettersize_paper_pieces : Nat := 5
def folds : Nat := 3
def notes_per_day : Nat := 10

def smaller_note_papers_per_piece : Nat := fold_paper folds
def total_smaller_note_papers : Nat := lettersize_paper_pieces * smaller_note_papers_per_piece
def total_days : Nat := total_smaller_note_papers / notes_per_day

theorem notepad_days_last : total_days = 4 := by
  sorry

end notepad_days_last_l133_133129


namespace find_single_digit_A_l133_133710

theorem find_single_digit_A (A : ℕ) (h1 : A < 10) (h2 : (11 * A)^2 = 5929) : A = 7 := 
sorry

end find_single_digit_A_l133_133710


namespace find_f_neg_2_l133_133819

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x: ℝ, f (-x) = -f x

-- Problem statement
theorem find_f_neg_2 (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_fx_pos : ∀ x : ℝ, x > 0 → f x = 2 * x ^ 2 - 7) : 
  f (-2) = -1 :=
by
  sorry

end find_f_neg_2_l133_133819


namespace common_ratio_geometric_series_l133_133666

theorem common_ratio_geometric_series
  (a₁ a₂ a₃ : ℚ)
  (h₁ : a₁ = 7 / 8)
  (h₂ : a₂ = -14 / 27)
  (h₃ : a₃ = 56 / 81) :
  (a₂ / a₁ = a₃ / a₂) ∧ (a₂ / a₁ = -2 / 3) :=
by
  -- The proof will follow here
  sorry

end common_ratio_geometric_series_l133_133666


namespace no_possible_numbering_for_equal_sidesum_l133_133748

theorem no_possible_numbering_for_equal_sidesum (O : Point) (A : Fin 10 → Point) 
  (side_numbers : (Fin 10) → ℕ) (segment_numbers : (Fin 10) → ℕ) : 
  ¬ ∃ (side_segment_sum_equal : Fin 10 → ℕ) (sum_equal : ℕ),
    (∀ i, side_segment_sum_equal i = side_numbers i + segment_numbers i) ∧ 
    (∀ i, side_segment_sum_equal i = sum_equal) := 
sorry

end no_possible_numbering_for_equal_sidesum_l133_133748


namespace inequality_x_y_z_l133_133269

-- Definitions for the variables
variables {x y z : ℝ} 
variable {n : ℕ}

-- Positive numbers and summation condition
axiom h1 : 0 < x ∧ 0 < y ∧ 0 < z
axiom h2 : x + y + z = 1

-- The theorem to be proven
theorem inequality_x_y_z (h1 : 0 < x ∧ 0 < y ∧ 0 < z) (h2 : x + y + z = 1) (hn : n > 0) : 
  x^n + y^n + z^n ≥ (1 : ℝ) / (3:ℝ)^(n-1) :=
sorry

end inequality_x_y_z_l133_133269


namespace john_age_is_24_l133_133518

noncomputable def john_age_condition (j d b : ℕ) : Prop :=
  j = d - 28 ∧
  j + d = 76 ∧
  j + 5 = 2 * (b + 5)

theorem john_age_is_24 (d b : ℕ) : ∃ j, john_age_condition j d b ∧ j = 24 :=
by
  use 24
  unfold john_age_condition
  sorry

end john_age_is_24_l133_133518


namespace quadratic_no_real_roots_iff_l133_133503

theorem quadratic_no_real_roots_iff (m : ℝ) : (∀ x : ℝ, x^2 + 3 * x + m ≠ 0) ↔ m > 9 / 4 :=
by
  sorry

end quadratic_no_real_roots_iff_l133_133503


namespace incorrect_reasoning_C_l133_133684

theorem incorrect_reasoning_C
  {Point : Type} {Line Plane : Type}
  (A B : Point) (l : Line) (α β : Plane)
  (in_line : Point → Line → Prop)
  (in_plane : Point → Plane → Prop)
  (line_in_plane : Line → Plane → Prop)
  (disjoint : Line → Plane → Prop) :

  ¬(line_in_plane l α) ∧ in_line A l ∧ in_plane A α :=
sorry

end incorrect_reasoning_C_l133_133684


namespace scientific_notation_of_86000000_l133_133689

theorem scientific_notation_of_86000000 :
  ∃ (x : ℝ) (y : ℤ), 86000000 = x * 10^y ∧ x = 8.6 ∧ y = 7 :=
by
  use 8.6
  use 7
  sorry

end scientific_notation_of_86000000_l133_133689


namespace log_bounds_l133_133123

-- Definitions and assumptions
def tenCubed : Nat := 1000
def tenFourth : Nat := 10000
def twoNine : Nat := 512
def twoFourteen : Nat := 16384

-- Statement that encapsulates the proof problem
theorem log_bounds (h1 : 10^3 = tenCubed) 
                   (h2 : 10^4 = tenFourth) 
                   (h3 : 2^9 = twoNine) 
                   (h4 : 2^14 = twoFourteen) : 
  (2 / 7 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (1 / 3 : ℝ) :=
sorry

end log_bounds_l133_133123


namespace probability_two_girls_l133_133854

theorem probability_two_girls (total_students girls boys : ℕ) (htotal : total_students = 6) (hg : girls = 4) (hb : boys = 2) :
  (Nat.choose girls 2 / Nat.choose total_students 2 : ℝ) = 2 / 5 := by
  sorry

end probability_two_girls_l133_133854


namespace calculate_constants_l133_133416

noncomputable def parabola_tangent_to_line (a b : ℝ) : Prop :=
  let discriminant := (b - 2) ^ 2 + 28 * a
  discriminant = 0

theorem calculate_constants
  (a b : ℝ)
  (h_tangent : parabola_tangent_to_line a b) :
  a = -((b - 2) ^ 2) / 28 ∧ b ≠ 2 :=
by
  sorry

end calculate_constants_l133_133416


namespace brendan_remaining_money_l133_133321

-- Definitions given in the conditions
def weekly_pay (total_monthly_earnings : ℕ) (weeks_in_month : ℕ) : ℕ := total_monthly_earnings / weeks_in_month
def weekly_recharge_amount (weekly_pay : ℕ) : ℕ := weekly_pay / 2
def total_recharge_amount (weekly_recharge_amount : ℕ) (weeks_in_month : ℕ) : ℕ := weekly_recharge_amount * weeks_in_month
def remaining_money_after_car_purchase (total_monthly_earnings : ℕ) (car_cost : ℕ) : ℕ := total_monthly_earnings - car_cost
def total_remaining_money (remaining_money_after_car_purchase : ℕ) (total_recharge_amount : ℕ) : ℕ := remaining_money_after_car_purchase - total_recharge_amount

-- The actual statement to prove
theorem brendan_remaining_money
  (total_monthly_earnings : ℕ := 5000)
  (weeks_in_month : ℕ := 4)
  (car_cost : ℕ := 1500)
  (weekly_pay := weekly_pay total_monthly_earnings weeks_in_month)
  (weekly_recharge_amount := weekly_recharge_amount weekly_pay)
  (total_recharge_amount := total_recharge_amount weekly_recharge_amount weeks_in_month)
  (remaining_money_after_car_purchase := remaining_money_after_car_purchase total_monthly_earnings car_cost)
  (total_remaining_money := total_remaining_money remaining_money_after_car_purchase total_recharge_amount) :
  total_remaining_money = 1000 :=
sorry

end brendan_remaining_money_l133_133321


namespace complex_solution_count_l133_133519

theorem complex_solution_count : 
  ∃ (s : Finset ℂ), (∀ z ∈ s, (z^3 - 8) / (z^2 - 3 * z + 2) = 0) ∧ s.card = 2 := 
by
  sorry

end complex_solution_count_l133_133519


namespace negation_of_proposition_l133_133898

open Classical

theorem negation_of_proposition : (¬ ∀ x : ℝ, 2 * x + 4 ≥ 0) ↔ (∃ x : ℝ, 2 * x + 4 < 0) :=
by
  sorry

end negation_of_proposition_l133_133898


namespace total_money_spent_l133_133019

theorem total_money_spent {s j : ℝ} (hs : s = 14.28) (hj : j = 4.74) : s + j = 19.02 :=
by
  sorry

end total_money_spent_l133_133019


namespace Emily_subtract_59_l133_133226

theorem Emily_subtract_59 : (30 - 1) ^ 2 = 30 ^ 2 - 59 := by
  sorry

end Emily_subtract_59_l133_133226


namespace train_overtakes_motorbike_in_80_seconds_l133_133217

-- Definitions of the given conditions
def speed_train_kmph : ℝ := 100
def speed_motorbike_kmph : ℝ := 64
def length_train_m : ℝ := 800.064

-- Definition to convert kmph to m/s
noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

-- Relative speed in m/s
noncomputable def relative_speed_mps : ℝ :=
  kmph_to_mps (speed_train_kmph - speed_motorbike_kmph)

-- Time taken for the train to overtake the motorbike
noncomputable def time_to_overtake (distance_m : ℝ) (speed_mps : ℝ) : ℝ :=
  distance_m / speed_mps

-- The statement to be proved
theorem train_overtakes_motorbike_in_80_seconds :
  time_to_overtake length_train_m relative_speed_mps = 80.0064 :=
by
  sorry

end train_overtakes_motorbike_in_80_seconds_l133_133217


namespace soccer_tournament_matches_l133_133974

theorem soccer_tournament_matches (n: ℕ):
  n = 20 → ∃ m: ℕ, m = 19 := sorry

end soccer_tournament_matches_l133_133974


namespace fraction_of_repeating_decimal_l133_133752

theorem fraction_of_repeating_decimal :
  ∃ (f : ℚ), f = 0.73 ∧ f = 73 / 99 := by
  sorry

end fraction_of_repeating_decimal_l133_133752


namespace proof_problem_l133_133765

-- Define the rates of P and Q
def P_rate : ℚ := 1/3
def Q_rate : ℚ := 1/18

-- Define the time they work together
def combined_time : ℚ := 2

-- Define the job completion rates
def combined_rate (P_rate Q_rate : ℚ) : ℚ := P_rate + Q_rate

-- Define the job completed together in given time
def job_completed_together (rate time : ℚ) : ℚ := rate * time

-- Define the remaining job
def remaining_job (total_job completed_job : ℚ) : ℚ := total_job - completed_job

-- Define the time required for P to complete the remaining job
def time_for_P (P_rate remaining_job : ℚ) : ℚ := remaining_job / P_rate

-- Define the total job as 1
def total_job : ℚ := 1

-- Correct answer in minutes
def correct_answer_in_minutes (time_in_hours : ℚ) : ℚ := time_in_hours * 60

-- Problem statement
theorem proof_problem : 
  correct_answer_in_minutes (time_for_P P_rate (remaining_job total_job 
    (job_completed_together (combined_rate P_rate Q_rate) combined_time))) = 40 := 
by
  sorry

end proof_problem_l133_133765


namespace candy_box_original_price_l133_133106

theorem candy_box_original_price (P : ℝ) (h₁ : 1.25 * P = 10) : P = 8 := 
sorry

end candy_box_original_price_l133_133106


namespace time_to_odd_floor_l133_133889

-- Define the number of even-numbered floors
def evenFloors : Nat := 5

-- Define the number of odd-numbered floors
def oddFloors : Nat := 5

-- Define the time to climb one even-numbered floor
def timeEvenFloor : Nat := 15

-- Define the total time to reach the 10th floor
def totalTime : Nat := 120

-- Define the desired time per odd-numbered floor
def timeOddFloor : Nat := 9

-- Formalize the proof statement
theorem time_to_odd_floor : 
  (oddFloors * timeOddFloor = totalTime - (evenFloors * timeEvenFloor)) :=
by
  sorry

end time_to_odd_floor_l133_133889


namespace faye_total_crayons_l133_133111

  def num_rows : ℕ := 16
  def crayons_per_row : ℕ := 6
  def total_crayons : ℕ := num_rows * crayons_per_row

  theorem faye_total_crayons : total_crayons = 96 :=
  by
  sorry
  
end faye_total_crayons_l133_133111


namespace height_of_shorter_pot_is_20_l133_133026

-- Define the conditions as given
def height_of_taller_pot := 40
def shadow_of_taller_pot := 20
def shadow_of_shorter_pot := 10

-- Define the height of the shorter pot to be determined
def height_of_shorter_pot (h : ℝ) := h

-- Define the relationship using the concept of similar triangles
theorem height_of_shorter_pot_is_20 (h : ℝ) :
  (height_of_taller_pot / shadow_of_taller_pot = height_of_shorter_pot h / shadow_of_shorter_pot) → h = 20 :=
by
  intros
  sorry

end height_of_shorter_pot_is_20_l133_133026


namespace problem_statement_l133_133126

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - x^2

theorem problem_statement (x0 x1 x2 m : ℝ) (h0 : f x0 = m) (h1 : 0 < x1) (h2 : x1 < x0) (h3 : x0 < x2) :
    f x1 > m ∧ f x2 < m :=
sorry

end problem_statement_l133_133126


namespace circle_center_is_21_l133_133522

theorem circle_center_is_21 : ∀ x y : ℝ, x^2 + y^2 - 4 * x - 2 * y - 5 = 0 →
                                      ∃ h k : ℝ, h = 2 ∧ k = 1 ∧ (x - h)^2 + (y - k)^2 = 10 :=
by
  intro x y h_eq
  sorry

end circle_center_is_21_l133_133522


namespace log_comparison_l133_133089

theorem log_comparison (a b c : ℝ) 
  (h₁ : a = Real.log 6 / Real.log 3)
  (h₂ : b = Real.log 10 / Real.log 5)
  (h₃ : c = Real.log 14 / Real.log 7) :
  a > b ∧ b > c :=
  sorry

end log_comparison_l133_133089


namespace necessary_but_not_sufficient_l133_133554

-- Definitions
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c

-- The condition we are given
axiom m : ℝ

-- The quadratic equation specific condition
axiom quadratic_condition : quadratic_eq 1 2 m = 0

-- The necessary but not sufficient condition for real solutions
theorem necessary_but_not_sufficient (h : m < 2) : 
  ∃ x : ℝ, quadratic_eq 1 2 m x = 0 ∧ quadratic_eq 1 2 m x = 0 → m ≤ 1 ∨ m > 1 :=
sorry

end necessary_but_not_sufficient_l133_133554


namespace valid_factorizations_of_1870_l133_133803

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_valid_factor1 (n : ℕ) : Prop := 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ n = p1 * p2

def is_valid_factor2 (n : ℕ) : Prop := 
  ∃ (p k : ℕ), is_prime p ∧ (k = 4 ∨ k = 6 ∨ k = 8 ∨ k = 9) ∧ n = p * k

theorem valid_factorizations_of_1870 : 
  ∃ a b : ℕ, a * b = 1870 ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ 
  ((is_valid_factor1 a ∧ is_valid_factor2 b) ∨ (is_valid_factor1 b ∧ is_valid_factor2 a)) ∧ 
  (a = 34 ∧ b = 55 ∨ a = 55 ∧ b = 34) ∧ 
  (¬∃ x y : ℕ, x * y = 1870 ∧ 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 ∧ 
  ((is_valid_factor1 x ∧ is_valid_factor2 y) ∨ (is_valid_factor1 y ∧ is_valid_factor2 x)) ∧ 
  (x ≠ 34 ∨ y ≠ 55 ∨ x ≠ 55 ∨ y ≠ 34)) :=
sorry

end valid_factorizations_of_1870_l133_133803


namespace simplify_complex_expr_l133_133991

theorem simplify_complex_expr : ∀ (i : ℂ), (4 - 2 * i) - (7 - 2 * i) + (6 - 3 * i) = 3 - 3 * i := by
  intro i
  sorry

end simplify_complex_expr_l133_133991


namespace line_through_intersection_perpendicular_l133_133042

theorem line_through_intersection_perpendicular (x y : ℝ) :
  (2 * x - 3 * y + 10 = 0) ∧ (3 * x + 4 * y - 2 = 0) →
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ (a = 2) ∧ (b = 3) ∧ (c = -2) ∧ (3 * a + 2 * b = 0)) :=
by
  sorry

end line_through_intersection_perpendicular_l133_133042


namespace smallest_n_for_2n_3n_5n_conditions_l133_133043

theorem smallest_n_for_2n_3n_5n_conditions : 
  ∃ n : ℕ, 
    (∀ k : ℕ, 2 * n ≠ k^2) ∧          -- 2n is a perfect square
    (∀ k : ℕ, 3 * n ≠ k^3) ∧          -- 3n is a perfect cube
    (∀ k : ℕ, 5 * n ≠ k^5) ∧          -- 5n is a perfect fifth power
    n = 11250 :=
sorry

end smallest_n_for_2n_3n_5n_conditions_l133_133043


namespace no_integer_solution_for_large_n_l133_133795

theorem no_integer_solution_for_large_n (n : ℕ) (m : ℤ) (h : n ≥ 11) : ¬(m^2 + 2 * 3^n = m * (2^(n+1) - 1)) :=
sorry

end no_integer_solution_for_large_n_l133_133795


namespace proportion_fourth_number_l133_133867

theorem proportion_fourth_number (x y : ℝ) (h_x : x = 0.6) (h_prop : 0.75 / x = 10 / y) : y = 8 :=
by
  sorry

end proportion_fourth_number_l133_133867


namespace fib_fact_last_two_sum_is_five_l133_133513

def fib_fact_last_two_sum (s : List (Fin 100)) : Fin 100 :=
  s.sum

theorem fib_fact_last_two_sum_is_five :
  fib_fact_last_two_sum [1, 1, 2, 6, 20, 20, 0] = 5 :=
by 
  sorry

end fib_fact_last_two_sum_is_five_l133_133513


namespace permutations_mississippi_l133_133347

theorem permutations_mississippi : 
  let total_letters := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  (Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count)) = 34650 := 
by
  sorry

end permutations_mississippi_l133_133347


namespace storks_more_than_birds_l133_133161

def initial_birds := 2
def additional_birds := 3
def total_birds := initial_birds + additional_birds
def storks := 6
def difference := storks - total_birds

theorem storks_more_than_birds : difference = 1 :=
by
  sorry

end storks_more_than_birds_l133_133161


namespace jason_has_21_toys_l133_133236

-- Definitions based on the conditions
def rachel_toys : ℕ := 1
def john_toys : ℕ := rachel_toys + 6
def jason_toys : ℕ := 3 * john_toys

-- The theorem to prove
theorem jason_has_21_toys : jason_toys = 21 := by
  -- Proof not needed, hence sorry
  sorry

end jason_has_21_toys_l133_133236


namespace pyramid_pattern_l133_133378

theorem pyramid_pattern
  (R : ℕ → ℕ)  -- a function representing the number of blocks in each row
  (R₁ : R 1 = 9)  -- the first row has 9 blocks
  (sum_eq : R 1 + R 2 + R 3 + R 4 + R 5 = 25)  -- the total number of blocks is 25
  (pattern : ∀ n, 1 ≤ n ∧ n < 5 → R (n + 1) = R n - 2) : ∃ d, d = 2 :=
by
  have pattern_valid : R 1 = 9 ∧ R 2 = 7 ∧ R 3 = 5 ∧ R 4 = 3 ∧ R 5 = 1 :=
    sorry  -- Proof omitted
  exact ⟨2, rfl⟩

end pyramid_pattern_l133_133378


namespace no_equilateral_integer_coords_l133_133860

theorem no_equilateral_integer_coords (x1 y1 x2 y2 x3 y3 : ℤ) : 
  ¬ ((x1 ≠ x2 ∨ y1 ≠ y2) ∧ 
     (x1 ≠ x3 ∨ y1 ≠ y3) ∧
     (x2 ≠ x3 ∨ y2 ≠ y3) ∧ 
     ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (x3 - x1) ^ 2 + (y3 - y1) ^ 2 ∧ 
      (x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (x3 - x2) ^ 2 + (y3 - y2) ^ 2)) :=
by
  sorry

end no_equilateral_integer_coords_l133_133860


namespace gilbert_parsley_count_l133_133302

variable (basil mint parsley : ℕ)
variable (initial_basil : ℕ := 3)
variable (extra_basil : ℕ := 1)
variable (initial_mint : ℕ := 2)
variable (herb_total : ℕ := 5)

def initial_parsley := herb_total - (initial_basil + extra_basil)

theorem gilbert_parsley_count : initial_parsley = 1 := by
  -- basil = initial_basil + extra_basil
  -- mint = 0 (since all mint plants eaten)
  -- herb_total = basil + parsley
  -- 5 = 4 + parsley
  -- parsley = 1
  sorry

end gilbert_parsley_count_l133_133302


namespace f_3_eq_4_l133_133030

noncomputable def f : ℝ → ℝ := sorry

theorem f_3_eq_4 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = x^2) : f 3 = 4 :=
by
  sorry

end f_3_eq_4_l133_133030


namespace fruit_seller_price_l133_133179

theorem fruit_seller_price (CP SP : ℝ) (h1 : SP = 0.90 * CP) (h2 : 1.10 * CP = 13.444444444444445) : 
  SP = 11 :=
sorry

end fruit_seller_price_l133_133179


namespace spending_Mar_Apr_May_l133_133394

-- Define the expenditures at given points
def e_Feb : ℝ := 0.7
def e_Mar : ℝ := 1.2
def e_May : ℝ := 4.4

-- Define the amount spent from March to May
def amount_spent_Mar_Apr_May := e_May - e_Feb

-- The main theorem to prove
theorem spending_Mar_Apr_May : amount_spent_Mar_Apr_May = 3.7 := by
  sorry

end spending_Mar_Apr_May_l133_133394


namespace find_a_when_lines_perpendicular_l133_133211

theorem find_a_when_lines_perpendicular (a : ℝ) : 
  (∃ x y : ℝ, ax + 3 * y - 1 = 0 ∧  2 * x + (a^2 - a) * y + 3 = 0) ∧ 
  (∃ m₁ m₂ : ℝ, m₁ = -a / 3 ∧ m₂ = -2 / (a^2 - a) ∧ m₁ * m₂ = -1)
  → a = 0 ∨ a = 5 / 3 :=
by {
  sorry
}

end find_a_when_lines_perpendicular_l133_133211


namespace max_mn_value_l133_133721

theorem max_mn_value (m n : ℝ) (hm : m > 0) (hn : n > 0)
  (hA1 : ∀ k : ℝ, k * (-2) - (-1) + 2 * k - 1 = 0)
  (hA2 : m * (-2) + n * (-1) + 2 = 0) :
  mn ≤ 1/2 := sorry

end max_mn_value_l133_133721


namespace mineral_age_possibilities_l133_133615

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def count_permutations_with_repeats (n : ℕ) (repeats : List ℕ) : ℕ :=
  factorial n / List.foldl (· * factorial ·) 1 repeats

theorem mineral_age_possibilities : 
  let digits := [2, 2, 4, 4, 7, 9]
  let odd_digits := [7, 9]
  let remaining_digits := [2, 2, 4, 4]
  2 * count_permutations_with_repeats 5 [2,2] = 60 :=
by
  sorry

end mineral_age_possibilities_l133_133615


namespace calculate_sum_of_squares_l133_133604

variables {a b : ℤ}
theorem calculate_sum_of_squares (h1 : (a + b)^2 = 17) (h2 : (a - b)^2 = 11) : a^2 + b^2 = 14 :=
by
  sorry

end calculate_sum_of_squares_l133_133604


namespace population_of_males_l133_133474

theorem population_of_males (total_population : ℕ) (num_parts : ℕ) (part_population : ℕ) 
  (male_population : ℕ) (female_population : ℕ) (children_population : ℕ) :
  total_population = 600 →
  num_parts = 4 →
  part_population = total_population / num_parts →
  children_population = 2 * male_population →
  male_population = part_population →
  male_population = 150 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end population_of_males_l133_133474


namespace log2_a_plus_log2_b_zero_l133_133192

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log2_a_plus_log2_b_zero 
    (a b : ℝ) 
    (h : (Nat.choose 6 3) * (a^3) * (b^3) = 20) 
    (hc : (a^2 + b / a)^(3) = 20 * x^(3)) :
  log2 a + log2 b = 0 :=
by
  sorry

end log2_a_plus_log2_b_zero_l133_133192


namespace find_a_l133_133142

open Set
open Real

def A : Set ℝ := {-1, 1}
def B (a : ℝ) : Set ℝ := {x | a * x ^ 2 = 1}

theorem find_a (a : ℝ) (h : (A ∩ (B a)) = (B a)) : a = 1 :=
sorry

end find_a_l133_133142


namespace distinct_diagonals_in_convex_nonagon_l133_133676

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end distinct_diagonals_in_convex_nonagon_l133_133676


namespace arith_seq_ratio_l133_133537

theorem arith_seq_ratio (a_2 a_3 S_4 S_5 : ℕ) 
  (arithmetic_seq : ∀ n : ℕ, ℕ)
  (sum_of_first_n_terms : ∀ n : ℕ, ℕ)
  (h1 : (a_2 : ℚ) / a_3 = 1 / 3) 
  (h2 : S_4 = 4 * (a_2 - (a_3 - a_2)) + ((4 * 3 * (a_3 - a_2)) / 2)) 
  (h3 : S_5 = 5 * (a_2 - (a_3 - a_2)) + ((5 * 4 * (a_3 - a_2)) / 2)) :
  (S_4 : ℚ) / S_5 = 8 / 15 :=
by sorry

end arith_seq_ratio_l133_133537


namespace return_trip_avg_speed_l133_133952

noncomputable def avg_speed_return_trip : ℝ := 
  let distance_ab_to_sy := 120
  let rate_ab_to_sy := 50
  let total_time := 5.5
  let time_ab_to_sy := distance_ab_to_sy / rate_ab_to_sy
  let time_return_trip := total_time - time_ab_to_sy
  distance_ab_to_sy / time_return_trip

theorem return_trip_avg_speed 
  (distance_ab_to_sy : ℝ := 120)
  (rate_ab_to_sy : ℝ := 50)
  (total_time : ℝ := 5.5) 
  : avg_speed_return_trip = 38.71 :=
by
  sorry

end return_trip_avg_speed_l133_133952


namespace ratio_after_addition_l133_133084

theorem ratio_after_addition (a b : ℕ) (h1 : a * 3 = b * 2) (h2 : b - a = 8) : (a + 4) * 7 = (b + 4) * 5 :=
by
  sorry

end ratio_after_addition_l133_133084


namespace minimum_overlap_l133_133279

variable (U : Finset ℕ) -- This is the set of all people surveyed
variable (B V : Finset ℕ) -- These are the sets of people who like Beethoven and Vivaldi respectively.

-- Given conditions:
axiom h_total : U.card = 120
axiom h_B : B.card = 95
axiom h_V : V.card = 80
axiom h_subset_B : B ⊆ U
axiom h_subset_V : V ⊆ U

-- Question to prove:
theorem minimum_overlap : (B ∩ V).card = 95 + 80 - 120 := by
  sorry

end minimum_overlap_l133_133279


namespace complex_problem_l133_133284

theorem complex_problem 
  (a : ℝ) 
  (ha : a^2 - 9 = 0) :
  (a + (Complex.I ^ 19)) / (1 + Complex.I) = 1 - 2 * Complex.I := by
  sorry

end complex_problem_l133_133284


namespace number_of_pairs_divisible_by_five_l133_133526

theorem number_of_pairs_divisible_by_five :
  (∃ n : ℕ, n = 864) ↔
  ∀ a b : ℕ, (1 ≤ a ∧ a ≤ 80) ∧ (1 ≤ b ∧ b ≤ 30) →
  (a * b) % 5 = 0 → (∃ n : ℕ, n = 864) := 
sorry

end number_of_pairs_divisible_by_five_l133_133526


namespace find_certain_number_l133_133113

theorem find_certain_number (x : ℕ) (certain_number : ℕ)
  (h1 : certain_number * x = 675)
  (h2 : x = 27) : certain_number = 25 :=
by
  -- Proof goes here
  sorry

end find_certain_number_l133_133113


namespace chimps_seen_l133_133305

-- Given conditions
def lions := 8
def lion_legs := 4
def lizards := 5
def lizard_legs := 4
def tarantulas := 125
def tarantula_legs := 8
def goal_legs := 1100

-- Required to be proved
def chimp_legs := 4

theorem chimps_seen : (goal_legs - ((lions * lion_legs) + (lizards * lizard_legs) + (tarantulas * tarantula_legs))) / chimp_legs = 25 :=
by
  -- placeholder for the proof
  sorry

end chimps_seen_l133_133305


namespace find_A_l133_133208

def spadesuit (A B : ℝ) : ℝ := 4 * A + 3 * B + 6

theorem find_A (A : ℝ) (h : spadesuit A 5 = 59) : A = 9.5 :=
by sorry

end find_A_l133_133208


namespace average_is_correct_l133_133558

def nums : List ℝ := [13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_is_correct :
  (nums.sum / nums.length) = 125830.8 :=
by sorry

end average_is_correct_l133_133558


namespace resulting_solid_faces_l133_133855

-- Define a cube structure with a given number of faces
structure Cube where
  faces : Nat

-- Define the problem conditions and prove the total faces of the resulting solid
def original_cube := Cube.mk 6

def new_faces_per_cube := 5

def total_new_faces := original_cube.faces * new_faces_per_cube

def total_faces_of_resulting_solid := total_new_faces + original_cube.faces

theorem resulting_solid_faces : total_faces_of_resulting_solid = 36 := by
  sorry

end resulting_solid_faces_l133_133855


namespace d_share_l133_133230

theorem d_share (T : ℝ) (A B C D E : ℝ) 
  (h1 : A = 5 / 15 * T) 
  (h2 : B = 2 / 15 * T) 
  (h3 : C = 4 / 15 * T)
  (h4 : D = 3 / 15 * T)
  (h5 : E = 1 / 15 * T)
  (combined_AC : A + C = 3 / 5 * T)
  (diff_BE : B - E = 250) : 
  D = 750 :=
by
  sorry

end d_share_l133_133230


namespace teal_total_sales_l133_133307

variable (pum_pie_slices_per_pie : ℕ) (cus_pie_slices_per_pie : ℕ)
variable (pum_pie_price_per_slice : ℕ) (cus_pie_price_per_slice : ℕ)
variable (pum_pies_sold : ℕ) (cus_pies_sold : ℕ)

def total_slices_sold (slices_per_pie pies_sold : ℕ) : ℕ :=
  slices_per_pie * pies_sold

def total_sales (slices_sold price_per_slice : ℕ) : ℕ :=
  slices_sold * price_per_slice

theorem teal_total_sales
  (h1 : pum_pie_slices_per_pie = 8)
  (h2 : cus_pie_slices_per_pie = 6)
  (h3 : pum_pie_price_per_slice = 5)
  (h4 : cus_pie_price_per_slice = 6)
  (h5 : pum_pies_sold = 4)
  (h6 : cus_pies_sold = 5) :
  (total_sales (total_slices_sold pum_pie_slices_per_pie pum_pies_sold) pum_pie_price_per_slice) +
  (total_sales (total_slices_sold cus_pie_slices_per_pie cus_pies_sold) cus_pie_price_per_slice) = 340 :=
by
  sorry

end teal_total_sales_l133_133307


namespace sum_xyz_l133_133316

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_xyz :
  (∀ x y z : ℝ,
  log_base 3 (log_base 4 (log_base 5 x)) = 0 ∧
  log_base 4 (log_base 5 (log_base 3 y)) = 0 ∧
  log_base 5 (log_base 3 (log_base 4 z)) = 0 →
  x + y + z = 932) := 
by
  sorry

end sum_xyz_l133_133316


namespace avg_age_new_students_l133_133039

theorem avg_age_new_students :
  ∀ (O A_old A_new_avg : ℕ) (A_new : ℕ),
    O = 12 ∧ A_old = 40 ∧ A_new_avg = (A_old - 4) ∧ A_new_avg = 36 →
    A_new * 12 = (24 * A_new_avg) - (O * A_old) →
    A_new = 32 :=
by
  intros O A_old A_new_avg A_new
  intro h
  rcases h with ⟨hO, hA_old, hA_new_avg, h36⟩
  sorry

end avg_age_new_students_l133_133039


namespace solve_equation_l133_133936

open Function

theorem solve_equation (m n : ℕ) (h_gcd : gcd m n = 2) (h_lcm : lcm m n = 4) :
  m * n = (gcd m n)^2 + lcm m n ↔ (m = 2 ∧ n = 4) ∨ (m = 4 ∧ n = 2) :=
by
  sorry

end solve_equation_l133_133936


namespace total_amount_paid_l133_133997

theorem total_amount_paid (g_p g_q m_p m_q : ℝ) (g_d g_t m_d m_t : ℝ) : 
    g_p = 70 -> g_q = 8 -> g_d = 0.05 -> g_t = 0.08 -> 
    m_p = 55 -> m_q = 9 -> m_d = 0.07 -> m_t = 0.11 -> 
    (g_p * g_q * (1 - g_d) * (1 + g_t) + m_p * m_q * (1 - m_d) * (1 + m_t)) = 1085.55 := by 
    sorry

end total_amount_paid_l133_133997


namespace spelling_bee_students_count_l133_133792

theorem spelling_bee_students_count (x : ℕ) (h1 : x / 2 * 1 / 4 * 2 = 30) : x = 240 :=
by
  sorry

end spelling_bee_students_count_l133_133792


namespace function_conditions_satisfied_l133_133632

noncomputable def function_satisfying_conditions : ℝ → ℝ := fun x => -2 * x^2 + 3 * x

theorem function_conditions_satisfied :
  (function_satisfying_conditions 1 = 1) ∧
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ function_satisfying_conditions x = y) ∧
  (∀ x y : ℝ, x > 1 ∧ y = function_satisfying_conditions x → ∃ ε > 0, ∀ δ > 0, (x + δ > 1 → function_satisfying_conditions (x + δ) < y)) :=
by
  sorry

end function_conditions_satisfied_l133_133632


namespace problem_statement_l133_133310

variable (a b : ℝ)

open Real

noncomputable def inequality_holds (a b : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ a + b < 2 → (1 / (1 + a^2)) + (1 / (1 + b^2)) ≤ 2 / (1 + a * b)

noncomputable def equality_condition (a b : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ a + b < 2 → ((1 / (1 + a^2)) + (1 / (1 + b^2)) = 2 / (1 + a * b) ↔ a = b)

theorem problem_statement (a b : ℝ) : inequality_holds a b ∧ equality_condition a b :=
by
  sorry

end problem_statement_l133_133310


namespace focus_with_greatest_y_coordinate_l133_133257

-- Define the conditions as hypotheses
def ellipse_major_axis : (ℝ × ℝ) := (0, 3)
def ellipse_minor_axis : (ℝ × ℝ) := (2, 0)
def ellipse_semi_major_axis : ℝ := 3
def ellipse_semi_minor_axis : ℝ := 2

-- Define the theorem to compute the coordinates of the focus with the greater y-coordinate
theorem focus_with_greatest_y_coordinate :
  let a := ellipse_semi_major_axis
  let b := ellipse_semi_minor_axis
  let c := Real.sqrt (a^2 - b^2)
  (0, c) = (0, (Real.sqrt 5) / 2) :=
by
  -- skipped proof
  sorry

end focus_with_greatest_y_coordinate_l133_133257


namespace combined_area_rectangle_triangle_l133_133116

/-- 
  Given a rectangle ABCD with vertices A = (10, -30), 
  B = (2010, 170), D = (12, -50), and a right triangle
  ADE with vertex E = (12, -30), prove that the combined
  area of the rectangle and the triangle is 
  40400 + 20√101.
-/
theorem combined_area_rectangle_triangle :
  let A := (10, -30)
  let B := (2010, 170)
  let D := (12, -50)
  let E := (12, -30)
  let length_AB := Real.sqrt ((2010 - 10)^2 + (170 + 30)^2)
  let length_AD := Real.sqrt ((12 - 10)^2 + (-50 + 30)^2)
  let area_rectangle := length_AB * length_AD
  let length_DE := Real.sqrt ((12 - 12)^2 + (-50 + 30)^2)
  let area_triangle := 1/2 * length_DE * length_AD
  area_rectangle + area_triangle = 40400 + 20 * Real.sqrt 101 :=
by
  sorry

end combined_area_rectangle_triangle_l133_133116


namespace Joe_time_from_home_to_school_l133_133633

-- Define the parameters
def walking_time := 4 -- minutes
def waiting_time := 2 -- minutes
def running_speed_ratio := 2 -- Joe's running speed is twice his walking speed

-- Define the walking and running times
def running_time (walking_time : ℕ) (running_speed_ratio : ℕ) : ℕ :=
  walking_time / running_speed_ratio

-- Total time it takes Joe to get from home to school
def total_time (walking_time waiting_time : ℕ) (running_speed_ratio : ℕ) : ℕ :=
  walking_time + waiting_time + running_time walking_time running_speed_ratio

-- Conjecture to be proved
theorem Joe_time_from_home_to_school :
  total_time walking_time waiting_time running_speed_ratio = 10 := by
  sorry

end Joe_time_from_home_to_school_l133_133633


namespace intersection_condition_l133_133294

-- Define the lines
def line1 (x y : ℝ) := 2*x - 2*y - 3 = 0
def line2 (x y : ℝ) := 3*x - 5*y + 1 = 0
def line (a b x y : ℝ) := a*x - y + b = 0

-- Define the condition
def condition (a b : ℝ) := 17*a + 4*b = 11

-- Prove that the line l passes through the intersection point of l1 and l2 if and only if the condition holds
theorem intersection_condition (a b : ℝ) :
  (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ line a b x y) ↔ condition a b :=
  sorry

end intersection_condition_l133_133294


namespace find_fourth_root_l133_133235

theorem find_fourth_root (b c α : ℝ)
  (h₁ : b * (-3)^4 + (b + 3 * c) * (-3)^3 + (c - 4 * b) * (-3)^2 + (19 - b) * (-3) - 2 = 0)
  (h₂ : b * 4^4 + (b + 3 * c) * 4^3 + (c - 4 * b) * 4^2 + (19 - b) * 4 - 2 = 0)
  (h₃ : b * 2^4 + (b + 3 * c) * 2^3 + (c - 4 * b) * 2^2 + (19 - b) * 2 - 2 = 0)
  (h₄ : (-3) + 4 + 2 + α = 2)
  : α = 1 :=
sorry

end find_fourth_root_l133_133235


namespace prime_sum_divisors_l133_133103

theorem prime_sum_divisors (p : ℕ) (s : ℕ) : 
  (2 ≤ s ∧ s ≤ 10) → 
  (p = 2^s - 1) → 
  (p = 3 ∨ p = 7 ∨ p = 31 ∨ p = 127) :=
by
  intros h1 h2
  sorry

end prime_sum_divisors_l133_133103


namespace factorize_poly1_l133_133758

variable (a : ℝ)

theorem factorize_poly1 : a^4 + 2 * a^3 + 1 = (a + 1) * (a^3 + a^2 - a + 1) := 
sorry

end factorize_poly1_l133_133758


namespace tom_candy_pieces_l133_133540

/-!
# Problem Statement
Tom bought 14 boxes of chocolate candy, 10 boxes of fruit candy, and 8 boxes of caramel candy. 
He gave 8 chocolate boxes and 5 fruit boxes to his little brother. 
If each chocolate box has 3 pieces inside, each fruit box has 4 pieces, and each caramel box has 5 pieces, 
prove that Tom still has 78 pieces of candy.
-/

theorem tom_candy_pieces 
  (chocolate_boxes : ℕ := 14)
  (fruit_boxes : ℕ := 10)
  (caramel_boxes : ℕ := 8)
  (gave_away_chocolate_boxes : ℕ := 8)
  (gave_away_fruit_boxes : ℕ := 5)
  (chocolate_pieces_per_box : ℕ := 3)
  (fruit_pieces_per_box : ℕ := 4)
  (caramel_pieces_per_box : ℕ := 5)
  : chocolate_boxes * chocolate_pieces_per_box + 
    fruit_boxes * fruit_pieces_per_box + 
    caramel_boxes * caramel_pieces_per_box - 
    (gave_away_chocolate_boxes * chocolate_pieces_per_box + 
     gave_away_fruit_boxes * fruit_pieces_per_box) = 78 :=
by
  sorry

end tom_candy_pieces_l133_133540


namespace range_of_a1_l133_133971

theorem range_of_a1 {a : ℕ → ℝ} (h_seq : ∀ n : ℕ, 2 * a (n + 1) * a n + a (n + 1) - 3 * a n = 0)
  (h_a1_positive : a 1 > 0) :
  (0 < a 1) ∧ (a 1 < 1) ↔ ∀ m n : ℕ, m < n → a m < a n := by
  sorry

end range_of_a1_l133_133971


namespace union_of_A_and_B_l133_133216

def setA : Set ℝ := {x : ℝ | x > 1 / 2}
def setB : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem union_of_A_and_B : setA ∪ setB = {x : ℝ | -1 < x} :=
by
  sorry

end union_of_A_and_B_l133_133216


namespace triangle_land_area_l133_133508

theorem triangle_land_area :
  let base_cm := 12
  let height_cm := 9
  let scale_cm_to_miles := 3
  let square_mile_to_acres := 640
  let area_cm2 := (1 / 2 : Float) * base_cm * height_cm
  let area_miles2 := area_cm2 * (scale_cm_to_miles ^ 2)
  let area_acres := area_miles2 * square_mile_to_acres
  area_acres = 311040 :=
by
  -- Skipped proofs
  sorry

end triangle_land_area_l133_133508


namespace tangent_line_eq_l133_133586

theorem tangent_line_eq
    (f : ℝ → ℝ) (f_def : ∀ x, f x = x ^ 2)
    (tangent_point : ℝ × ℝ) (tangent_point_def : tangent_point = (1, 1))
    (f' : ℝ → ℝ) (f'_def : ∀ x, f' x = 2 * x)
    (slope_at_1 : f' 1 = 2) :
    ∃ (a b : ℝ), a = 2 ∧ b = -1 ∧ ∀ x y, y = a * x + b ↔ (2 * x - y - 1 = 0) :=
sorry

end tangent_line_eq_l133_133586


namespace financing_amount_correct_l133_133679

-- Define the conditions
def monthly_payment : ℕ := 150
def years : ℕ := 5
def months_per_year : ℕ := 12

-- Define the total financed amount
def total_financed : ℕ := monthly_payment * years * months_per_year

-- The statement that we need to prove
theorem financing_amount_correct : total_financed = 9000 := 
by
  sorry

end financing_amount_correct_l133_133679


namespace compare_store_costs_l133_133246

-- Define the conditions mathematically
def StoreA_cost (x : ℕ) : ℝ := 5 * x + 125
def StoreB_cost (x : ℕ) : ℝ := 4.5 * x + 135

theorem compare_store_costs (x : ℕ) (h : x ≥ 5) : 
  5 * 15 + 125 = 200 ∧ 4.5 * 15 + 135 = 202.5 ∧ 200 < 202.5 := 
by
  -- Here the theorem states the claims to be proved.
  sorry

end compare_store_costs_l133_133246


namespace simplify_and_calculate_expression_l133_133445

variable (a b : ℤ)

theorem simplify_and_calculate_expression (h_a : a = -3) (h_b : b = -2) :
  (a + b) * (b - a) + (2 * a^2 * b - a^3) / (-a) = -8 :=
by
  -- We include the proof steps here to achieve the final result.
  sorry

end simplify_and_calculate_expression_l133_133445


namespace intersection_of_circle_and_line_in_polar_coordinates_l133_133464

noncomputable section

def circle_polar_eq (ρ θ : ℝ) : Prop := ρ = Real.cos θ + Real.sin θ
def line_polar_eq (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2 / 2

theorem intersection_of_circle_and_line_in_polar_coordinates :
  ∀ θ ρ, (0 < θ ∧ θ < Real.pi) →
  circle_polar_eq ρ θ →
  line_polar_eq ρ θ →
  ρ = 1 ∧ θ = Real.pi / 2 :=
by
  sorry

end intersection_of_circle_and_line_in_polar_coordinates_l133_133464


namespace product_of_primes_l133_133612

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end product_of_primes_l133_133612


namespace length_of_train_l133_133201

theorem length_of_train (speed_km_hr : ℝ) (platform_length_m : ℝ) (time_sec : ℝ) 
  (h1 : speed_km_hr = 72) (h2 : platform_length_m = 250) (h3 : time_sec = 30) : 
  ∃ (train_length : ℝ), train_length = 350 := 
by 
  -- Definitions of the given conditions
  let speed_m_per_s := speed_km_hr * (5 / 18)
  let total_distance := speed_m_per_s * time_sec
  let train_length := total_distance - platform_length_m
  -- Verifying the length of the train
  use train_length
  sorry

end length_of_train_l133_133201


namespace intersection_A_B_l133_133778

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x | x < 2 }

theorem intersection_A_B : A ∩ B = { x | -1 ≤ x ∧ x < 2 } := 
by 
  sorry

end intersection_A_B_l133_133778


namespace max_cos_a_l133_133098

theorem max_cos_a (a b : ℝ) (h : Real.cos (a + b) = Real.cos a - Real.cos b) : 
  Real.cos a ≤ 1 := 
sorry

end max_cos_a_l133_133098


namespace parallel_lines_iff_a_eq_2_l133_133779

-- Define line equations
def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - a + 1 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y - 2 = 0

-- Prove that a = 2 is necessary and sufficient for the lines to be parallel.
theorem parallel_lines_iff_a_eq_2 (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → ∃ u v : ℝ, l2 a u v → x = u ∧ y = v) ↔ (a = 2) :=
by {
  sorry
}

end parallel_lines_iff_a_eq_2_l133_133779


namespace find_f_neg4_l133_133619

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^2 - a * x + b

theorem find_f_neg4 (a b : ℝ) (h1 : f 1 a b = -1) (h2 : f 2 a b = 2) : 
  f (-4) a b = 14 :=
by
  sorry

end find_f_neg4_l133_133619


namespace longer_piece_length_l133_133584

theorem longer_piece_length (x : ℝ) (h1 : x + (x + 2) = 30) : x + 2 = 16 :=
by sorry

end longer_piece_length_l133_133584


namespace bicycle_cost_price_l133_133056

theorem bicycle_cost_price (CP_A : ℝ) 
    (h1 : ∀ SP_B, SP_B = 1.20 * CP_A)
    (h2 : ∀ CP_C SP_B, CP_C = 1.40 * SP_B ∧ SP_B = 1.20 * CP_A)
    (h3 : ∀ SP_D CP_C, SP_D = 1.30 * CP_C ∧ CP_C = 1.40 * 1.20 * CP_A)
    (h4 : ∀ SP_D', SP_D' = 350 / 0.90) :
    CP_A = 350 / 1.9626 :=
by
  sorry

end bicycle_cost_price_l133_133056


namespace speed_of_mother_minimum_running_time_l133_133521

namespace XiaotongTravel

def distance_to_binjiang : ℝ := 4320
def time_diff : ℝ := 12
def speed_rate : ℝ := 1.2

theorem speed_of_mother : 
  ∃ (x : ℝ), (distance_to_binjiang / x - distance_to_binjiang / (speed_rate * x) = time_diff) → (speed_rate * x = 72) :=
sorry

def distance_to_company : ℝ := 2940
def running_speed : ℝ := 150
def total_time : ℝ := 30

theorem minimum_running_time :
  ∃ (y : ℝ), ((distance_to_company - running_speed * y) / 72 + y ≤ total_time) → (y ≥ 10) :=
sorry

end XiaotongTravel

end speed_of_mother_minimum_running_time_l133_133521


namespace gail_has_two_ten_dollar_bills_l133_133006

-- Define the given conditions
def total_amount : ℕ := 100
def num_five_bills : ℕ := 4
def num_twenty_bills : ℕ := 3
def value_five_bill : ℕ := 5
def value_twenty_bill : ℕ := 20
def value_ten_bill : ℕ := 10

-- The function to determine the number of ten-dollar bills
noncomputable def num_ten_bills : ℕ := 
  (total_amount - (num_five_bills * value_five_bill + num_twenty_bills * value_twenty_bill)) / value_ten_bill

-- Proof statement
theorem gail_has_two_ten_dollar_bills : num_ten_bills = 2 := by
  sorry

end gail_has_two_ten_dollar_bills_l133_133006


namespace can_determine_number_of_spies_l133_133923

def determine_spies (V : Fin 15 → ℕ) (S : Fin 15 → ℕ) : Prop :=
  V 0 = S 0 + S 1 ∧ 
  ∀ i : Fin 13, V (Fin.succ (Fin.succ i)) = S i + S (Fin.succ i) + S (Fin.succ (Fin.succ i)) ∧
  V 14 = S 13 + S 14

theorem can_determine_number_of_spies :
  ∃ S : Fin 15 → ℕ, ∀ V : Fin 15 → ℕ, determine_spies V S :=
sorry

end can_determine_number_of_spies_l133_133923


namespace perimeter_of_C_l133_133609

theorem perimeter_of_C (x y : ℝ) 
  (h₁ : 6 * x + 2 * y = 56) 
  (h₂ : 4 * x + 6 * y = 56) : 
  2 * x + 6 * y = 40 :=
sorry

end perimeter_of_C_l133_133609


namespace marble_probability_l133_133031

theorem marble_probability :
  let total_marbles := 13
  let red_marbles := 5
  let white_marbles := 8
  let first_red_prob := (red_marbles:ℚ) / total_marbles
  let second_white_given_first_red_prob := (white_marbles:ℚ) / (total_marbles - 1)
  let third_red_given_first_red_and_second_white_prob := (red_marbles - 1:ℚ) / (total_marbles - 2)
  first_red_prob * second_white_given_first_red_prob * third_red_given_first_red_and_second_white_prob = (40 : ℚ) / 429 :=
by
  let total_marbles := 13
  let red_marbles := 5
  let white_marbles := 8
  let first_red_prob := (red_marbles:ℚ) / total_marbles
  let second_white_given_first_red_prob := (white_marbles:ℚ) / (total_marbles - 1)
  let third_red_given_first_red_and_second_white_prob := (red_marbles - 1:ℚ) / (total_marbles - 2)
  -- Adding sorry to skip the proof
  sorry

end marble_probability_l133_133031


namespace exists_range_of_real_numbers_l133_133672

theorem exists_range_of_real_numbers (x : ℝ) :
  (x^2 - 5 * x + 7 ≠ 1) ↔ (x ≠ 3 ∧ x ≠ 2) := 
sorry

end exists_range_of_real_numbers_l133_133672


namespace intersection_complement_l133_133440

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5}) (hA : A = {2, 3, 4}) (hB : B = {1, 2})

theorem intersection_complement :
  A ∩ (U \ B) = {3, 4} :=
by
  rw [hU, hA, hB]
  sorry

end intersection_complement_l133_133440


namespace solution_eq1_solution_eq2_l133_133570

theorem solution_eq1 (x : ℝ) : 
  2 * x^2 - 4 * x - 1 = 0 ↔ 
  (x = 1 + (Real.sqrt 6) / 2 ∨ x = 1 - (Real.sqrt 6) / 2) := by
sorry

theorem solution_eq2 (x : ℝ) :
  (x - 1) * (x + 2) = 28 ↔ 
  (x = -6 ∨ x = 5) := by
sorry

end solution_eq1_solution_eq2_l133_133570


namespace necessary_and_sufficient_condition_l133_133105

-- Define the first circle
def circle1 (m : ℝ) : Set (ℝ × ℝ) :=
  { p | (p.1 + m)^2 + p.2^2 = 1 }

-- Define the second circle
def circle2 : Set (ℝ × ℝ) :=
  { p | (p.1 - 2)^2 + p.2^2 = 4 }

-- Define the condition -1 ≤ m ≤ 1
def condition (m : ℝ) : Prop :=
  -1 ≤ m ∧ m ≤ 1

-- Define the property for circles having common points
def circlesHaveCommonPoints (m : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ circle1 m ∧ p ∈ circle2

-- The final statement
theorem necessary_and_sufficient_condition (m : ℝ) :
  condition m → circlesHaveCommonPoints m ↔ (-5 ≤ m ∧ m ≤ 1) :=
by
  sorry

end necessary_and_sufficient_condition_l133_133105


namespace y_intercept_of_line_l133_133933

theorem y_intercept_of_line (x y : ℝ) (eq : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 := 
by
  sorry

end y_intercept_of_line_l133_133933


namespace convert_mps_to_kmph_l133_133943

theorem convert_mps_to_kmph (v_mps : ℝ) (c : ℝ) (h_c : c = 3.6) (h_v_mps : v_mps = 20) : (v_mps * c = 72) :=
by
  rw [h_v_mps, h_c]
  sorry

end convert_mps_to_kmph_l133_133943


namespace solve_quadratic_eq_l133_133733

theorem solve_quadratic_eq (a b x : ℝ) :
  12 * a * b * x^2 - (16 * a^2 - 9 * b^2) * x - 12 * a * b = 0 ↔ (x = 4 * a / (3 * b)) ∨ (x = -3 * b / (4 * a)) :=
by
  sorry

end solve_quadratic_eq_l133_133733


namespace apples_in_each_basket_l133_133150

-- Definitions based on the conditions
def total_apples : ℕ := 64
def baskets : ℕ := 4
def apples_taken_per_basket : ℕ := 3

-- Theorem statement based on the question and correct answer
theorem apples_in_each_basket (h1 : total_apples = 64) 
                              (h2 : baskets = 4) 
                              (h3 : apples_taken_per_basket = 3) : 
    (total_apples / baskets - apples_taken_per_basket) = 13 := 
by
  sorry

end apples_in_each_basket_l133_133150


namespace original_class_strength_l133_133959

theorem original_class_strength (x : ℕ) 
    (avg_original : ℕ)
    (num_new : ℕ) 
    (avg_new : ℕ) 
    (decrease : ℕ)
    (h1 : avg_original = 40)
    (h2 : num_new = 17)
    (h3 : avg_new = 32)
    (h4 : decrease = 4)
    (h5 : (40 * x + 17 * avg_new) = (x + num_new) * (40 - decrease))
    : x = 17 := 
by {
  sorry
}

end original_class_strength_l133_133959


namespace sum_of_first_6033_terms_l133_133291

noncomputable def geometric_series_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_6033_terms
  (a r : ℝ)  
  (h1 : geometric_series_sum a r 2011 = 200)
  (h2 : geometric_series_sum a r 4022 = 380) :
  geometric_series_sum a r 6033 = 542 := 
sorry

end sum_of_first_6033_terms_l133_133291


namespace max_true_statements_maximum_true_conditions_l133_133874

theorem max_true_statements (x y : ℝ) (h1 : (1/x > 1/y)) (h2 : (x^2 < y^2)) (h3 : (x > y)) (h4 : (x > 0)) (h5 : (y > 0)) :
  false :=
  sorry

theorem maximum_true_conditions (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  ¬ ((1/x > 1/y) ∧ (x^2 < y^2)) :=
  sorry

#check max_true_statements
#check maximum_true_conditions

end max_true_statements_maximum_true_conditions_l133_133874


namespace divisibility_equivalence_distinct_positive_l133_133921

variable (a b c : ℕ)

theorem divisibility_equivalence_distinct_positive (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a + b + c) ∣ (a^3 * b + b^3 * c + c^3 * a)) ↔ ((a + b + c) ∣ (a * b^3 + b * c^3 + c * a^3)) :=
by sorry

end divisibility_equivalence_distinct_positive_l133_133921


namespace ice_cream_not_sold_total_l133_133108

theorem ice_cream_not_sold_total :
  let chocolate_initial := 50
  let mango_initial := 54
  let vanilla_initial := 80
  let strawberry_initial := 40
  let chocolate_sold := (3 / 5 : ℚ) * chocolate_initial
  let mango_sold := (2 / 3 : ℚ) * mango_initial
  let vanilla_sold := (75 / 100 : ℚ) * vanilla_initial
  let strawberry_sold := (5 / 8 : ℚ) * strawberry_initial
  let chocolate_not_sold := chocolate_initial - chocolate_sold
  let mango_not_sold := mango_initial - mango_sold
  let vanilla_not_sold := vanilla_initial - vanilla_sold
  let strawberry_not_sold := strawberry_initial - strawberry_sold
  chocolate_not_sold + mango_not_sold + vanilla_not_sold + strawberry_not_sold = 73 :=
by sorry

end ice_cream_not_sold_total_l133_133108


namespace tens_digit_8_pow_2023_l133_133791

theorem tens_digit_8_pow_2023 : (8 ^ 2023 % 100) / 10 % 10 = 1 := 
sorry

end tens_digit_8_pow_2023_l133_133791


namespace sum_remainders_l133_133169

theorem sum_remainders :
  ∀ (a b c d : ℕ),
  a % 53 = 31 →
  b % 53 = 44 →
  c % 53 = 6 →
  d % 53 = 2 →
  (a + b + c + d) % 53 = 30 :=
by
  intros a b c d ha hb hc hd
  sorry

end sum_remainders_l133_133169


namespace candies_indeterminable_l133_133421

theorem candies_indeterminable
  (num_bags : ℕ) (cookies_per_bag : ℕ) (total_cookies : ℕ) (known_candies : ℕ) :
  num_bags = 26 →
  cookies_per_bag = 2 →
  total_cookies = 52 →
  num_bags * cookies_per_bag = total_cookies →
  ∀ (candies : ℕ), candies = known_candies → false :=
by
  intros
  sorry

end candies_indeterminable_l133_133421


namespace snow_at_least_once_l133_133507

noncomputable def prob_snow_at_least_once (p1 p2 p3: ℚ) : ℚ :=
  1 - (1 - p1) * (1 - p2) * (1 - p3)

theorem snow_at_least_once : 
  prob_snow_at_least_once (1/2) (2/3) (3/4) = 23 / 24 := 
by
  sorry

end snow_at_least_once_l133_133507


namespace range_of_a_l133_133663

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_non_neg (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ y) → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  even_function f → increasing_on_non_neg f → f a ≤ f 2 → -2 ≤ a ∧ a ≤ 2 :=
by
  intro h_even h_increasing h_le
  sorry

end range_of_a_l133_133663


namespace quadratic_range_l133_133505

noncomputable def f : ℝ → ℝ := sorry -- Quadratic function with a positive coefficient for its quadratic term

axiom symmetry_condition : ∀ x : ℝ, f x = f (4 - x)

theorem quadratic_range (x : ℝ) (h1 : f (1 - 2 * x ^ 2) < f (1 + 2 * x - x ^ 2)) : -2 < x ∧ x < 0 :=
by sorry

end quadratic_range_l133_133505


namespace brick_width_correct_l133_133811

theorem brick_width_correct
  (courtyard_length_m : ℕ) (courtyard_width_m : ℕ) (brick_length_cm : ℕ) (num_bricks : ℕ)
  (total_area_cm : ℕ) (brick_width_cm : ℕ) :
  courtyard_length_m = 25 →
  courtyard_width_m = 16 →
  brick_length_cm = 20 →
  num_bricks = 20000 →
  total_area_cm = courtyard_length_m * 100 * courtyard_width_m * 100 →
  total_area_cm = num_bricks * brick_length_cm * brick_width_cm →
  brick_width_cm = 10 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end brick_width_correct_l133_133811


namespace necessary_and_sufficient_condition_l133_133925

-- Definitions for sides opposite angles A, B, and C in a triangle.
variables {A B C : Real} {a b c : Real}

-- Condition p: sides a, b related to angles A, B via cosine
def condition_p (a b : Real) (A B : Real) : Prop := a / Real.cos A = b / Real.cos B

-- Condition q: sides a and b are equal
def condition_q (a b : Real) : Prop := a = b

theorem necessary_and_sufficient_condition (h1 : condition_p a b A B) : condition_q a b ↔ condition_p a b A B :=
by
  sorry

end necessary_and_sufficient_condition_l133_133925


namespace robot_transport_max_robots_l133_133252

section
variable {A B : ℕ}   -- Define the variables A and B
variable {m : ℕ}     -- Define the variable m

-- Part 1
theorem robot_transport (h1 : A = B + 30) (h2 : 1500 * B = 1000 * (B + 30)) : A = 90 ∧ B = 60 :=
by
  sorry

-- Part 2
theorem max_robots (h3 : 50000 * m + 30000 * (12 - m) ≤ 450000) : m ≤ 4 :=
by
  sorry
end

end robot_transport_max_robots_l133_133252


namespace T_n_sum_general_term_b_b_n_comparison_l133_133093

noncomputable def sequence_a (n : ℕ) : ℕ := sorry  -- Placeholder for sequence {a_n}
noncomputable def S (n : ℕ) : ℕ := sorry  -- Placeholder for sum of first n terms S_n
noncomputable def sequence_b (n : ℕ) (q : ℝ) : ℝ := sorry  -- Placeholder for sequence {b_n}

axiom sequence_a_def : ∀ n : ℕ, 2 * sequence_a (n + 1) = sequence_a n + sequence_a (n + 2)
axiom sequence_a_5 : sequence_a 5 = 5
axiom S_7 : S 7 = 28

noncomputable def T (n : ℕ) : ℝ := (2 * n : ℝ) / (n + 1 : ℝ)

theorem T_n_sum : ∀ n : ℕ, T n = 2 * (1 - 1 / (n + 1)) := sorry

axiom b1 : ℝ
axiom b_def : ∀ (n : ℕ) (q : ℝ), q > 0 → sequence_b (n + 1) q = sequence_b n q + q ^ (sequence_a n)

theorem general_term_b (q : ℝ) (n : ℕ) (hq : q > 0) : 
  (if q = 1 then sequence_b n q = n else sequence_b n q = (1 - q ^ n) / (1 - q)) := sorry

theorem b_n_comparison (q : ℝ) (n : ℕ) (hq : q > 0) : 
  sequence_b n q * sequence_b (n + 2) q < (sequence_b (n + 1) q) ^ 2 := sorry

end T_n_sum_general_term_b_b_n_comparison_l133_133093


namespace unit_vector_perpendicular_l133_133578

theorem unit_vector_perpendicular (x y : ℝ) (h : 3 * x + 4 * y = 0) (m : x^2 + y^2 = 1) : 
  (x = -4/5 ∧ y = 3/5) ∨ (x = 4/5 ∧ y = -3/5) :=
by
  sorry

end unit_vector_perpendicular_l133_133578


namespace phase_shift_of_sine_l133_133941

theorem phase_shift_of_sine (b c : ℝ) (h_b : b = 4) (h_c : c = - (Real.pi / 2)) :
  (-c / b) = Real.pi / 8 :=
by
  rw [h_b, h_c]
  sorry

end phase_shift_of_sine_l133_133941


namespace sum_of_values_l133_133016

theorem sum_of_values (N : ℝ) (h : N * (N + 4) = 8) : N + (4 - N - 8 / N) = -4 := 
sorry

end sum_of_values_l133_133016


namespace max_distance_circle_to_line_l133_133956

open Real

-- Definitions of polar equations and transformations to Cartesian coordinates
def circle_eq (ρ θ : ℝ) : Prop := (ρ = 8 * sin θ)
def line_eq (θ : ℝ) : Prop := (θ = π / 3)

-- Cartesian coordinate transformations
def circle_cartesian (x y : ℝ) : Prop := (x^2 + (y - 4)^2 = 16)
def line_cartesian (x y : ℝ) : Prop := (y = sqrt 3 * x)

-- Maximum distance problem statement
theorem max_distance_circle_to_line : 
  ∀ (x y : ℝ), circle_cartesian x y → 
  (∀ x y, line_cartesian x y → 
  ∃ d : ℝ, d = 6) :=
by
  sorry

end max_distance_circle_to_line_l133_133956


namespace rectangle_perimeter_l133_133370

theorem rectangle_perimeter (a b : ℕ) : 
  (2 * a + b = 6 ∨ a + 2 * b = 6 ∨ 2 * a + b = 9 ∨ a + 2 * b = 9) → 
  2 * a + 2 * b = 10 :=
by 
  sorry

end rectangle_perimeter_l133_133370


namespace can_encode_number_l133_133918

theorem can_encode_number : ∃ (m n : ℕ), (0.07 = 1 / (m : ℝ) + 1 / (n : ℝ)) :=
by
  -- Proof omitted
  sorry

end can_encode_number_l133_133918


namespace yogurt_production_cost_l133_133617

-- Define the conditions
def milk_cost_per_liter : ℝ := 1.5
def fruit_cost_per_kg : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Define the theorem statement
theorem yogurt_production_cost : 
  (milk_cost_per_liter * milk_needed_per_batch + fruit_cost_per_kg * fruit_needed_per_batch) * batches = 63 := 
  by 
  sorry

end yogurt_production_cost_l133_133617


namespace vector_magnitude_positive_l133_133760

variable {V : Type} [NormedAddCommGroup V] [NormedSpace ℝ V]

variables (a b : V)

-- Given: 
-- a is any non-zero vector
-- b is a unit vector
theorem vector_magnitude_positive (ha : a ≠ 0) (hb : ‖b‖ = 1) : ‖a‖ > 0 := 
sorry

end vector_magnitude_positive_l133_133760


namespace find_number_l133_133227

theorem find_number (N x : ℝ) (h : x = 9) (h1 : N - (5 / x) = 4 + (4 / x)) : N = 5 :=
by
  sorry

end find_number_l133_133227


namespace initial_books_l133_133070

theorem initial_books (added_books : ℝ) (books_per_shelf : ℝ) (shelves : ℝ) 
  (total_books : ℝ) : total_books = shelves * books_per_shelf → 
  shelves = 14 → books_per_shelf = 4.0 → added_books = 10.0 → 
  total_books - added_books = 46.0 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_books_l133_133070


namespace ratio_spaghetti_to_fettuccine_l133_133630

def spg : Nat := 300
def fet : Nat := 80

theorem ratio_spaghetti_to_fettuccine : spg / gcd spg fet = 300 / 20 ∧ fet / gcd spg fet = 80 / 20 ∧ (spg / gcd spg fet) / (fet / gcd spg fet) = 15 / 4 := by
  sorry

end ratio_spaghetti_to_fettuccine_l133_133630


namespace find_least_multiple_of_50_l133_133032

def digits (n : ℕ) : List ℕ := n.digits 10

def product_of_digits (n : ℕ) : ℕ := (digits n).prod

theorem find_least_multiple_of_50 :
  ∃ n, (n % 50 = 0) ∧ ((product_of_digits n) % 50 = 0) ∧ (∀ m, (m % 50 = 0) ∧ ((product_of_digits m) % 50 = 0) → n ≤ m) ↔ n = 5550 :=
by sorry

end find_least_multiple_of_50_l133_133032


namespace interval_of_x_l133_133178

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) → (2 < 5 * x ∧ 5 * x < 3) → (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end interval_of_x_l133_133178


namespace sum_leq_six_of_quadratic_roots_l133_133345

theorem sum_leq_six_of_quadratic_roots (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
  (h3 : ∃ r1 r2 : ℤ, r1 ≠ r2 ∧ x^2 + ab * x + (a + b) = 0 ∧ 
         x = r1 ∧ x = r2) : a + b ≤ 6 :=
by
  sorry

end sum_leq_six_of_quadratic_roots_l133_133345


namespace cody_initial_marbles_l133_133696

theorem cody_initial_marbles (M : ℕ) (h1 : (2 / 3 : ℝ) * M - (1 / 4 : ℝ) * ((2 / 3 : ℝ) * M) - (2 * (1 / 4 : ℝ) * ((2 / 3 : ℝ) * M)) = 7) : M = 42 := 
  sorry

end cody_initial_marbles_l133_133696


namespace round_to_nearest_whole_l133_133277

theorem round_to_nearest_whole (x : ℝ) (hx : x = 7643.498201) : Int.floor (x + 0.5) = 7643 := 
by
  -- To prove
  sorry

end round_to_nearest_whole_l133_133277


namespace cost_of_car_l133_133848

theorem cost_of_car (initial_payment : ℕ) (num_installments : ℕ) (installment_amount : ℕ) : 
  initial_payment = 3000 →
  num_installments = 6 →
  installment_amount = 2500 →
  initial_payment + num_installments * installment_amount = 18000 :=
by
  intros h_initial h_num h_installment
  sorry

end cost_of_car_l133_133848


namespace spring_length_relationship_l133_133330

def spring_length (x : ℝ) : ℝ := 6 + 0.3 * x

theorem spring_length_relationship (x : ℝ) : spring_length x = 0.3 * x + 6 :=
by sorry

end spring_length_relationship_l133_133330


namespace cube_path_length_l133_133686

noncomputable def path_length_dot_cube : ℝ :=
  let edge_length := 2
  let radius1 := Real.sqrt 5
  let radius2 := 1
  (radius1 + radius2) * Real.pi

theorem cube_path_length :
  path_length_dot_cube = (Real.sqrt 5 + 1) * Real.pi :=
by
  sorry

end cube_path_length_l133_133686


namespace kibble_consumption_rate_l133_133471

-- Kira fills her cat's bowl with 3 pounds of kibble before going to work.
def initial_kibble : ℚ := 3

-- There is still 1 pound left when she returns.
def remaining_kibble : ℚ := 1

-- Kira was away from home for 8 hours.
def time_away : ℚ := 8

-- Calculate the amount of kibble eaten
def kibble_eaten : ℚ := initial_kibble - remaining_kibble

-- Calculate the rate of consumption (hours per pound)
def rate_of_consumption (time: ℚ) (kibble: ℚ) : ℚ := time / kibble

-- Theorem statement: It takes 4 hours for Kira's cat to eat a pound of kibble.
theorem kibble_consumption_rate : rate_of_consumption time_away kibble_eaten = 4 := by
  sorry

end kibble_consumption_rate_l133_133471


namespace num_pairs_divisible_7_l133_133255

theorem num_pairs_divisible_7 (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 1000) (hy : 1 ≤ y ∧ y ≤ 1000)
  (divisible : (x^2 + y^2) % 7 = 0) : 
  (∃ k : ℕ, k = 20164) :=
sorry

end num_pairs_divisible_7_l133_133255


namespace smallest_value_a1_l133_133641

def seq (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = 7 * a (n-1) - 2 * n

theorem smallest_value_a1 (a : ℕ → ℝ) (h1 : ∀ n, 0 < a n) (h2 : seq a) : 
  a 1 ≥ 13 / 18 :=
sorry

end smallest_value_a1_l133_133641


namespace inequality_proof_l133_133856

theorem inequality_proof (a b : ℝ) (h : a > b ∧ b > 0) : 
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧ (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := 
by 
  sorry

end inequality_proof_l133_133856


namespace max_x_plus_y_l133_133337

-- Define the conditions as hypotheses in a Lean statement
theorem max_x_plus_y (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^4 = (x - 1) * (y^3 - 23) - 1) :
  x + y ≤ 7 ∧ (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^4 = (x - 1) * (y^3 - 23) - 1 ∧ x + y = 7) :=
by
  sorry

end max_x_plus_y_l133_133337


namespace not_divisible_l133_133402

theorem not_divisible (n : ℕ) : ¬ ((4^n - 1) ∣ (5^n - 1)) :=
by
  sorry

end not_divisible_l133_133402


namespace ferry_journey_difference_l133_133704

theorem ferry_journey_difference
  (time_P : ℝ) (speed_P : ℝ) (mult_Q : ℝ) (speed_diff : ℝ)
  (dist_P : ℝ := time_P * speed_P)
  (dist_Q : ℝ := mult_Q * dist_P)
  (speed_Q : ℝ := speed_P + speed_diff)
  (time_Q : ℝ := dist_Q / speed_Q) :
  time_P = 3 ∧ speed_P = 6 ∧ mult_Q = 3 ∧ speed_diff = 3 → time_Q - time_P = 3 := by
  sorry

end ferry_journey_difference_l133_133704


namespace tree_planting_activity_l133_133453

variables (trees_first_group trees_second_group people_first_group people_second_group : ℕ)
variable (average_trees_per_person_first_group average_trees_per_person_second_group : ℕ)

theorem tree_planting_activity :
  trees_first_group = 12 →
  trees_second_group = 36 →
  people_second_group = people_first_group + 6 →
  average_trees_per_person_first_group = trees_first_group / people_first_group →
  average_trees_per_person_second_group = trees_second_group / people_second_group →
  average_trees_per_person_first_group = average_trees_per_person_second_group →
  people_first_group = 3 := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end tree_planting_activity_l133_133453


namespace speed_of_sound_l133_133013

theorem speed_of_sound (d₁ d₂ t : ℝ) (speed_car : ℝ) (speed_km_hr_to_m_s : ℝ) :
  d₁ = 1200 ∧ speed_car = 108 ∧ speed_km_hr_to_m_s = (speed_car * 1000 / 3600) ∧ t = 3.9669421487603307 →
  (d₁ + speed_km_hr_to_m_s * t) / t = 332.59 :=
by sorry

end speed_of_sound_l133_133013


namespace find_side_AB_l133_133749

theorem find_side_AB 
  (B C : ℝ) (BC : ℝ) (hB : B = 45) (hC : C = 45) (hBC : BC = 10) : 
  ∃ AB : ℝ, AB = 5 * Real.sqrt 2 :=
by
  -- We add 'sorry' here to indicate that the proof is not provided.
  sorry

end find_side_AB_l133_133749


namespace age_multiplier_l133_133375

theorem age_multiplier (S F M X : ℕ) (h1 : S = 27) (h2 : F = 48) (h3 : S + F = 75)
  (h4 : 27 - X = F - S) (h5 : F = M * X) : M = 8 :=
by
  -- Proof will be filled in here
  sorry

end age_multiplier_l133_133375


namespace must_divide_l133_133473

-- Proving 5 is a divisor of q

variables {p q r s : ℕ}

theorem must_divide (h1 : Nat.gcd p q = 30) (h2 : Nat.gcd q r = 42)
                   (h3 : Nat.gcd r s = 66) (h4 : 80 < Nat.gcd s p)
                   (h5 : Nat.gcd s p < 120) :
                   5 ∣ q :=
sorry

end must_divide_l133_133473


namespace geometric_sequence_a3_equals_4_l133_133975

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ i, a (i+1) = a i * r

theorem geometric_sequence_a3_equals_4 
    (a_seq : is_geometric_sequence a) 
    (a_6_eq : a 6 = 6)
    (a_9_eq : a 9 = 9) : 
    a 3 = 4 := 
sorry

end geometric_sequence_a3_equals_4_l133_133975


namespace smallest_c_no_real_root_l133_133532

theorem smallest_c_no_real_root (c : ℤ) :
  (∀ x : ℝ, x^2 + (c : ℝ) * x + 10 ≠ 5) ↔ c = -4 :=
by
  sorry

end smallest_c_no_real_root_l133_133532


namespace find_S16_l133_133807

-- Definitions
def geom_seq (a : ℕ → ℝ) : Prop := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def sum_of_geom_seq (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, S n = a 0 * (1 - (a 1 / a 0)^n) / (1 - (a 1 / a 0))

-- Problem conditions
variables {a : ℕ → ℝ} {S : ℕ → ℝ}

axiom geom_seq_a : geom_seq a
axiom S4_eq : S 4 = 4
axiom S8_eq : S 8 = 12

-- Theorem
theorem find_S16 : S 16 = 60 :=
  sorry

end find_S16_l133_133807


namespace set_roster_method_l133_133420

open Set

theorem set_roster_method :
  { m : ℤ | ∃ n : ℕ, 12 = n * (m + 1) } = {0, 1, 2, 3, 5, 11} :=
  sorry

end set_roster_method_l133_133420


namespace greatest_perimeter_triangle_l133_133693

theorem greatest_perimeter_triangle :
  ∃ (x : ℕ), (x > (16 / 5)) ∧ (x < (16 / 3)) ∧ ((x = 4 ∨ x = 5) → 4 * x + x + 16 = 41) :=
by
  sorry

end greatest_perimeter_triangle_l133_133693


namespace ball_hits_ground_at_2_72_l133_133437

-- Define the initial conditions
def initial_velocity (v₀ : ℝ) := v₀ = 30
def initial_height (h₀ : ℝ) := h₀ = 200
def ball_height (t : ℝ) : ℝ := -16 * t^2 - 30 * t + 200

-- Prove that the ball hits the ground at t = 2.72 seconds
theorem ball_hits_ground_at_2_72 (t : ℝ) (h : ℝ) 
  (v₀ : ℝ) (h₀ : ℝ) 
  (hv₀ : initial_velocity v₀) 
  (hh₀ : initial_height h₀)
  (h_eq: ball_height t = h) 
  (h₀_eq: ball_height 0 = h₀) : 
  h = 0 -> t = 2.72 :=
by
  sorry

end ball_hits_ground_at_2_72_l133_133437


namespace problem_i31_problem_i32_problem_i33_problem_i34_l133_133509

-- Problem I3.1
theorem problem_i31 (a : ℝ) :
  a = 1.8 * 5.0865 + 1 - 0.0865 * 1.8 → a = 10 :=
by sorry

-- Problem I3.2
theorem problem_i32 (a b : ℕ) (oh ok : ℕ) (OABC : Prop) :
  oh = ok ∧ oh = a ∧ ok = a ∧ OABC ∧ (b = AC) → b = 10 :=
by sorry

-- Problem I3.3
theorem problem_i33 (b c : ℕ) :
  b = 10 → c = (10 - 2) :=
by sorry

-- Problem I3.4
theorem problem_i34 (c d : ℕ) :
  c = 30 → d = 3 * c → d = 90 :=
by sorry

end problem_i31_problem_i32_problem_i33_problem_i34_l133_133509


namespace quadratic_function_proof_l133_133729

noncomputable def quadratic_function_condition (a b c : ℝ) :=
  ∀ x : ℝ, ((-3 ≤ x ∧ x ≤ 1) → (a * x^2 + b * x + c) ≤ 0) ∧
           ((x < -3 ∨ 1 < x) → (a * x^2 + b * x + c) > 0) ∧
           (a * 2^2 + b * 2 + c) = 5

theorem quadratic_function_proof (a b c : ℝ) (m : ℝ)
  (h : quadratic_function_condition a b c) :
  (a = 1 ∧ b = 2 ∧ c = -3) ∧ (m ≥ -7/9 ↔ ∃ x : ℝ, a * x^2 + b * x + c = 9 * m + 3) :=
by
  sorry

end quadratic_function_proof_l133_133729


namespace twenty_percent_greater_l133_133669

theorem twenty_percent_greater (x : ℕ) : 
  x = 80 + (20 * 80 / 100) → x = 96 :=
by
  sorry

end twenty_percent_greater_l133_133669


namespace probability_of_selection_l133_133681

-- Problem setup
def number_of_students : ℕ := 54
def number_of_students_eliminated : ℕ := 4
def number_of_remaining_students : ℕ := number_of_students - number_of_students_eliminated
def number_of_students_selected : ℕ := 5

-- Statement to be proved
theorem probability_of_selection :
  (number_of_students_selected : ℚ) / (number_of_students : ℚ) = 5 / 54 :=
sorry

end probability_of_selection_l133_133681


namespace positive_solution_l133_133954

variable {x y z : ℝ}

theorem positive_solution (h1 : x * y = 8 - 2 * x - 3 * y)
    (h2 : y * z = 8 - 4 * y - 2 * z)
    (h3 : x * z = 40 - 5 * x - 3 * z) :
    x = 10 := by
  sorry

end positive_solution_l133_133954


namespace fractionOf_Product_Of_Fractions_l133_133218

noncomputable def fractionOfProductOfFractions := 
  let a := (2 : ℚ) / 9 * (5 : ℚ) / 6 -- Define the product of the fractions
  let b := (3 : ℚ) / 4 -- Define another fraction
  a / b = 20 / 81 -- Statement to be proven

theorem fractionOf_Product_Of_Fractions: fractionOfProductOfFractions :=
by sorry

end fractionOf_Product_Of_Fractions_l133_133218


namespace profit_percentage_l133_133195

theorem profit_percentage (CP SP : ℝ) (h1 : CP = 500) (h2 : SP = 650) : 
  (SP - CP) / CP * 100 = 30 :=
by
  sorry

end profit_percentage_l133_133195


namespace range_of_c_over_a_l133_133769

theorem range_of_c_over_a (a b c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + b + c = 0) : -2 < c / a ∧ c / a < -1 :=
by {
  sorry
}

end range_of_c_over_a_l133_133769


namespace value_of_m_l133_133377

theorem value_of_m :
  ∀ m : ℝ, (x : ℝ) → (x^2 - 5 * x + m = (x - 3) * (x - 2)) → m = 6 :=
by
  sorry

end value_of_m_l133_133377


namespace calc_subtract_l133_133618

-- Define the repeating decimal
def repeating_decimal := (11 : ℚ) / 9

-- Define the problem statement
theorem calc_subtract : 3 - repeating_decimal = (16 : ℚ) / 9 := by
  sorry

end calc_subtract_l133_133618


namespace smallest_divisible_by_15_11_12_l133_133280

theorem smallest_divisible_by_15_11_12 : ∃ n : ℕ, (n > 0) ∧ (15 ∣ n) ∧ (11 ∣ n) ∧ (12 ∣ n) ∧ (∀ m : ℕ, (m > 0) ∧ (15 ∣ m) ∧ (11 ∣ m) ∧ (12 ∣ m) → n ≤ m) ∧ n = 660 :=
by
  sorry

end smallest_divisible_by_15_11_12_l133_133280


namespace man_and_son_work_together_l133_133384

-- Define the rates at which the man and his son can complete the work
def man_work_rate := 1 / 5
def son_work_rate := 1 / 20

-- Define the combined work rate when they work together
def combined_work_rate := man_work_rate + son_work_rate

-- Define the total time taken to complete the work together
def days_to_complete_together := 1 / combined_work_rate

-- The theorem stating that they will complete the work in 4 days
theorem man_and_son_work_together : days_to_complete_together = 4 := by
  sorry

end man_and_son_work_together_l133_133384


namespace triangle_similarity_length_RY_l133_133775

theorem triangle_similarity_length_RY
  (P Q R X Y Z : Type)
  [MetricSpace P] [MetricSpace Q] [MetricSpace R]
  [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (PQ : ℝ) (XY : ℝ) (RY_length : ℝ)
  (h1 : PQ = 10)
  (h2 : XY = 6)
  (h3 : ∀ (PR QR PX QX RZ : ℝ) (angle_PY_RZ : ℝ),
    PR + RY_length = PX ∧
    QR + RY_length = QX ∧ 
    angle_PY_RZ = 120 ∧
    PR > 0 ∧ QR > 0 ∧ RY_length > 0)
  (h4 : XY / PQ = RY_length / (PQ + RY_length)) :
  RY_length = 15 := by
  sorry

end triangle_similarity_length_RY_l133_133775


namespace tan_identity_proof_l133_133799

theorem tan_identity_proof :
  (1 - Real.tan (100 * Real.pi / 180)) * (1 - Real.tan (35 * Real.pi / 180)) = 2 :=
by
  have tan_135 : Real.tan (135 * Real.pi / 180) = -1 := by sorry -- This needs a separate proof.
  have tan_sum_formula : ∀ A B : ℝ, Real.tan (A + B) = (Real.tan A + Real.tan B) / (1 - Real.tan A * Real.tan B) := by sorry -- This needs a deeper exploration
  sorry -- Main proof to be filled

end tan_identity_proof_l133_133799


namespace number_of_pairs_satisfying_x_sq_minus_y_sq_eq_100_l133_133896

theorem number_of_pairs_satisfying_x_sq_minus_y_sq_eq_100 :
  ∃! (n : ℕ), n = 3 ∧ ∀ (x y : ℕ), x > 0 → y > 0 → x^2 - y^2 = 100 ↔ (x, y) = (26, 24) ∨ (x, y) = (15, 10) ∨ (x, y) = (15, 5) :=
by
  sorry

end number_of_pairs_satisfying_x_sq_minus_y_sq_eq_100_l133_133896


namespace arithmetic_geometric_sequence_l133_133329

theorem arithmetic_geometric_sequence :
  ∀ (a₁ a₂ b₂ : ℝ),
    -- Conditions for arithmetic sequence: -1, a₁, a₂, 8
    2 * a₁ = -1 + a₂ ∧
    2 * a₂ = a₁ + 8 →
    -- Conditions for geometric sequence: -1, b₁, b₂, b₃, -4
    (∃ (b₁ b₃ : ℝ), b₁^2 = b₂ ∧ b₁ != 0 ∧ -4 * b₁^4 = b₂ → -1 * b₁ = b₃) →
    -- Goal: Calculate and prove the value
    (a₁ * a₂ / b₂) = -5 :=
by {
  sorry
}

end arithmetic_geometric_sequence_l133_133329


namespace jeff_total_run_is_290_l133_133406

variables (monday_to_wednesday_run : ℕ)
variables (thursday_run : ℕ)
variables (friday_run : ℕ)

def jeff_weekly_run_total : ℕ :=
  monday_to_wednesday_run + thursday_run + friday_run

theorem jeff_total_run_is_290 :
  (60 * 3) + (60 - 20) + (60 + 10) = 290 :=
by
  sorry

end jeff_total_run_is_290_l133_133406


namespace guise_hot_dogs_l133_133780

theorem guise_hot_dogs (x : ℤ) (h1 : x + (x + 2) + (x + 4) = 36) : x = 10 :=
by
  sorry

end guise_hot_dogs_l133_133780


namespace apple_production_l133_133912

variable {S1 S2 S3 : ℝ}

theorem apple_production (h1 : S2 = 0.8 * S1) 
                         (h2 : S3 = 2 * S2) 
                         (h3 : S1 + S2 + S3 = 680) : 
                         S1 = 200 := 
by
  sorry

end apple_production_l133_133912


namespace prize_winners_l133_133118

theorem prize_winners (total_people : ℕ) (percent_envelope : ℝ) (percent_win : ℝ) 
  (h_total : total_people = 100) (h_percent_envelope : percent_envelope = 0.40) 
  (h_percent_win : percent_win = 0.20) : 
  (percent_win * (percent_envelope * total_people)) = 8 := by
  sorry

end prize_winners_l133_133118


namespace op_proof_l133_133454

-- Definition of the operation \(\oplus\)
def op (x y : ℝ) : ℝ := x^2 + y

-- Theorem statement for the given proof problem
theorem op_proof (h : ℝ) : op h (op h h) = 2 * h^2 + h :=
by 
  sorry

end op_proof_l133_133454


namespace locus_of_midpoint_l133_133387

theorem locus_of_midpoint
  (x y : ℝ)
  (h : ∃ (A : ℝ × ℝ), A = (2*x, 2*y) ∧ (A.1)^2 + (A.2)^2 - 8*A.1 = 0) :
  x^2 + y^2 - 4*x = 0 :=
by
  sorry

end locus_of_midpoint_l133_133387


namespace cyclist_is_jean_l133_133728

theorem cyclist_is_jean (x x' y y' : ℝ) (hx : x' = 4 * x) (hy : y = 4 * y') : x < y :=
by
  sorry

end cyclist_is_jean_l133_133728


namespace luncheon_cost_l133_133180

variables (s c p : ℝ)

def eq1 := 5 * s + 8 * c + 2 * p = 5.10
def eq2 := 6 * s + 11 * c + 2 * p = 6.45

theorem luncheon_cost (h₁ : 5 * s + 8 * c + 2 * p = 5.10) (h₂ : 6 * s + 11 * c + 2 * p = 6.45) : 
  s + c + p = 1.35 :=
  sorry

end luncheon_cost_l133_133180


namespace Danny_more_wrappers_than_caps_l133_133480

-- Define the conditions
def bottle_caps_park := 11
def wrappers_park := 28

-- State the theorem representing the problem
theorem Danny_more_wrappers_than_caps:
  wrappers_park - bottle_caps_park = 17 :=
by
  sorry

end Danny_more_wrappers_than_caps_l133_133480


namespace sum_inverse_one_minus_roots_eq_half_l133_133154

noncomputable def cubic_eq_roots (x : ℝ) : ℝ := 10 * x^3 - 25 * x^2 + 8 * x - 1

theorem sum_inverse_one_minus_roots_eq_half
  {p q s : ℝ} (hpqseq : cubic_eq_roots p = 0 ∧ cubic_eq_roots q = 0 ∧ cubic_eq_roots s = 0)
  (hpospq : 0 < p ∧ 0 < q ∧ 0 < s) (hlespq : p < 1 ∧ q < 1 ∧ s < 1) :
  (1 / (1 - p)) + (1 / (1 - q)) + (1 / (1 - s)) = 1 / 2 :=
sorry

end sum_inverse_one_minus_roots_eq_half_l133_133154


namespace solve_equation_1_solve_equation_2_l133_133496

theorem solve_equation_1 :
  ∀ x : ℝ, (2 * x - 1) ^ 2 = 9 ↔ (x = 2 ∨ x = -1) :=
by
  sorry

theorem solve_equation_2 :
  ∀ x : ℝ, x ^ 2 - 4 * x - 12 = 0 ↔ (x = 6 ∨ x = -2) :=
by
  sorry

end solve_equation_1_solve_equation_2_l133_133496


namespace necessary_condition_of_equilateral_triangle_l133_133034

variable {A B C: ℝ}
variable {a b c: ℝ}

theorem necessary_condition_of_equilateral_triangle
  (h1 : B + C = 2 * A)
  (h2 : b + c = 2 * a)
  : (A = B ∧ B = C ∧ a = b ∧ b = c) ↔ (B + C = 2 * A ∧ b + c = 2 * a) := 
by
  sorry

end necessary_condition_of_equilateral_triangle_l133_133034


namespace irreducible_fraction_l133_133622

theorem irreducible_fraction (n : ℤ) : Int.gcd (2 * n + 1) (3 * n + 1) = 1 :=
sorry

end irreducible_fraction_l133_133622


namespace compare_f_values_l133_133711

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.cos x

theorem compare_f_values :
  f 0.6 > f (-0.5) ∧ f (-0.5) > f 0 := by
  sorry

end compare_f_values_l133_133711


namespace cistern_fill_time_l133_133053

variable (C : ℝ) -- Volume of the cistern
variable (X Y Z : ℝ) -- Rates at which pipes X, Y, and Z fill the cistern

-- Pipes X and Y together, pipes X and Z together, and pipes Y and Z together conditions
def condition1 := X + Y = C / 3
def condition2 := X + Z = C / 4
def condition3 := Y + Z = C / 5

theorem cistern_fill_time (h1 : condition1 C X Y) (h2 : condition2 C X Z) (h3 : condition3 C Y Z) :
  1 / (X + Y + Z) = 120 / 47 :=
by
  sorry

end cistern_fill_time_l133_133053


namespace sum_of_ages_l133_133911

-- Definitions for conditions
def age_product (a b c : ℕ) : Prop := a * b * c = 72
def younger_than_10 (k : ℕ) : Prop := k < 10

-- Main statement
theorem sum_of_ages (a b k : ℕ) (h_product : age_product a b k) (h_twin : a = b) (h_kiana : younger_than_10 k) : 
  a + b + k = 14 := sorry

end sum_of_ages_l133_133911


namespace All_Yarns_are_Zorps_and_Xings_l133_133249

-- Define the basic properties
variables {α : Type}
variable (Zorp Xing Yarn Wit Vamp : α → Prop)

-- Given conditions
axiom all_Zorps_are_Xings : ∀ z, Zorp z → Xing z
axiom all_Yarns_are_Xings : ∀ y, Yarn y → Xing y
axiom all_Wits_are_Zorps : ∀ w, Wit w → Zorp w
axiom all_Yarns_are_Wits : ∀ y, Yarn y → Wit y
axiom all_Yarns_are_Vamps : ∀ y, Yarn y → Vamp y

-- Proof problem
theorem All_Yarns_are_Zorps_and_Xings : 
  ∀ y, Yarn y → (Zorp y ∧ Xing y) :=
sorry

end All_Yarns_are_Zorps_and_Xings_l133_133249


namespace range_of_a_l133_133239

def set1 : Set ℝ := {x | x ≤ 2}
def set2 (a : ℝ) : Set ℝ := {x | x > a}
variable (a : ℝ)

theorem range_of_a (h : set1 ∪ set2 a = Set.univ) : a ≤ 2 :=
by sorry

end range_of_a_l133_133239


namespace least_number_to_subtract_l133_133747

theorem least_number_to_subtract (x : ℕ) :
  1439 - x ≡ 3 [MOD 5] ∧ 
  1439 - x ≡ 3 [MOD 11] ∧ 
  1439 - x ≡ 3 [MOD 13] ↔ 
  x = 9 :=
by sorry

end least_number_to_subtract_l133_133747


namespace jason_flames_per_minute_l133_133139

theorem jason_flames_per_minute :
  (∀ (t : ℕ), t % 15 = 0 -> (5 * (t / 15) = 20)) :=
sorry

end jason_flames_per_minute_l133_133139


namespace lake_with_more_frogs_has_45_frogs_l133_133431

-- Definitions for the problem.
variable (F : ℝ) -- Number of frogs in the lake with more frogs.
variable (F_less : ℝ) -- Number of frogs in Lake Crystal (the lake with fewer frogs).

-- Conditions
axiom fewer_frogs_condition : F_less = 0.8 * F
axiom total_frogs_condition : F + F_less = 81

-- Theorem statement: Proving that the number of frogs in the lake with more frogs is 45.
theorem lake_with_more_frogs_has_45_frogs :
  F = 45 :=
by
  sorry

end lake_with_more_frogs_has_45_frogs_l133_133431


namespace solve_for_x_l133_133295

theorem solve_for_x (x : ℝ) (h : 9 / (1 + 4 / x) = 1) : x = 1 / 2 :=
by
  sorry

end solve_for_x_l133_133295


namespace sin_alpha_eq_three_fifths_l133_133182

theorem sin_alpha_eq_three_fifths (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.tan α = -3 / 4) 
  (h3 : Real.sin α > 0) 
  (h4 : Real.cos α < 0) 
  (h5 : Real.sin α ^ 2 + Real.cos α ^ 2 = 1) : 
  Real.sin α = 3 / 5 := 
sorry

end sin_alpha_eq_three_fifths_l133_133182


namespace max_value_expr_l133_133078

theorem max_value_expr (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 4) : 
  10 * x + 3 * y + 15 * z ≤ 9.455 :=
sorry

end max_value_expr_l133_133078


namespace find_x_l133_133038

theorem find_x (x y : ℕ) 
  (h1 : 3^x * 4^y = 59049) 
  (h2 : x - y = 10) : 
  x = 10 := 
by 
  sorry

end find_x_l133_133038


namespace wire_cut_example_l133_133812

theorem wire_cut_example (total_length piece_ratio : ℝ) (h1 : total_length = 28) (h2 : piece_ratio = 2.00001 / 5) :
  ∃ (shorter_piece : ℝ), shorter_piece + piece_ratio * shorter_piece = total_length ∧ shorter_piece = 20 :=
by
  sorry

end wire_cut_example_l133_133812


namespace advertisement_probability_l133_133008

theorem advertisement_probability
  (ads_time_hour : ℕ)
  (total_time_hour : ℕ)
  (h1 : ads_time_hour = 20)
  (h2 : total_time_hour = 60) :
  ads_time_hour / total_time_hour = 1 / 3 :=
by
  sorry

end advertisement_probability_l133_133008


namespace trisha_spending_l133_133629

theorem trisha_spending :
  let initial_amount := 167
  let spent_meat := 17
  let spent_veggies := 43
  let spent_eggs := 5
  let spent_dog_food := 45
  let remaining_amount := 35
  let total_spent := initial_amount - remaining_amount
  let other_spending := spent_meat + spent_veggies + spent_eggs + spent_dog_food
  total_spent - other_spending = 22 :=
by
  let initial_amount := 167
  let spent_meat := 17
  let spent_veggies := 43
  let spent_eggs := 5
  let spent_dog_food := 45
  let remaining_amount := 35
  -- Calculate total spent
  let total_spent := initial_amount - remaining_amount
  -- Calculate spending on other items
  let other_spending := spent_meat + spent_veggies + spent_eggs + spent_dog_food
  -- Statement to prove
  show total_spent - other_spending = 22
  sorry

end trisha_spending_l133_133629


namespace min_remainder_n_div_2005_l133_133288

theorem min_remainder_n_div_2005 (n : ℕ) (hn_pos : 0 < n) 
  (h1 : n % 902 = 602) (h2 : n % 802 = 502) (h3 : n % 702 = 402) :
  n % 2005 = 101 :=
sorry

end min_remainder_n_div_2005_l133_133288


namespace percent_is_50_l133_133588

variable (cats hogs percent : ℕ)
variable (hogs_eq_3cats : hogs = 3 * cats)
variable (hogs_eq_75 : hogs = 75)

theorem percent_is_50
  (cats_minus_5_percent_eq_10 : (cats - 5) * percent = 1000)
  (cats_eq_25 : cats = 25) :
  percent = 50 := by
  sorry

end percent_is_50_l133_133588


namespace sum_due_is_correct_l133_133593

theorem sum_due_is_correct (BD TD PV : ℝ) (h1 : BD = 80) (h2 : TD = 70) (h_relation : BD = TD + (TD^2) / PV) : PV = 490 :=
by sorry

end sum_due_is_correct_l133_133593


namespace triangular_angles_l133_133148

noncomputable def measure_of_B (A : ℝ) : ℝ :=
  Real.arcsin (Real.sqrt ((1 + Real.sin (2 * A)) / 3))

noncomputable def length_of_c (A : ℝ) : ℝ := 
  Real.sqrt (22 - 6 * Real.sqrt 13 * Real.cos (measure_of_B A))

noncomputable def area_of_triangle_ABC (A : ℝ) : ℝ := 
  (3 * Real.sqrt 13 / 2) * Real.sqrt ((1 + Real.sin (2 * A)) / 3)

theorem triangular_angles 
  (a b c : ℝ) (b_pos : b = Real.sqrt 13) (a_pos : a = 3) (h : b * Real.cos c = (2 * a - c) * Real.cos (measure_of_B c)) :
  c = length_of_c c ∧
  (3 * Real.sqrt 13 / 2) * Real.sqrt ((1 + Real.sin (2 * c)) / 3) = area_of_triangle_ABC c :=
by
  sorry

end triangular_angles_l133_133148


namespace two_digit_number_reverse_sum_eq_99_l133_133793

theorem two_digit_number_reverse_sum_eq_99 :
  ∃ a b : ℕ, (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ ((10 * a + b) - (10 * b + a) = 5 * (a + b))
  ∧ (10 * a + b) + (10 * b + a) = 99 := 
by
  sorry

end two_digit_number_reverse_sum_eq_99_l133_133793


namespace smallest_possible_x2_plus_y2_l133_133985

theorem smallest_possible_x2_plus_y2 (x y : ℝ) (h : (x + 3) * (y - 3) = 0) : x^2 + y^2 = 18 :=
sorry

end smallest_possible_x2_plus_y2_l133_133985


namespace trig_expression_value_l133_133863

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 3) :
  2 * Real.sin α ^ 2 + 4 * Real.sin α * Real.cos α - 9 * Real.cos α ^ 2 = 21 / 10 :=
by
  sorry

end trig_expression_value_l133_133863


namespace cube_volume_l133_133333

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l133_133333


namespace chess_match_duration_l133_133364

def time_per_move_polly := 28
def time_per_move_peter := 40
def total_moves := 30
def moves_per_player := total_moves / 2

def Polly_time := moves_per_player * time_per_move_polly
def Peter_time := moves_per_player * time_per_move_peter
def total_time_seconds := Polly_time + Peter_time
def total_time_minutes := total_time_seconds / 60

theorem chess_match_duration : total_time_minutes = 17 := by
  sorry

end chess_match_duration_l133_133364


namespace inequality_holds_l133_133137

variable (a b c : ℝ)

theorem inequality_holds (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + b*c) / (a * (b + c)) + 
  (b^2 + c*a) / (b * (c + a)) + 
  (c^2 + a*b) / (c * (a + b)) ≥ 3 :=
sorry

end inequality_holds_l133_133137


namespace factorial_div_eq_l133_133328
-- Import the entire math library

-- Define the entities involved in the problem
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the given conditions
def given_expression : ℕ := factorial 10 / (factorial 7 * factorial 3)

-- State the main theorem that corresponds to the given problem and its correct answer
theorem factorial_div_eq : given_expression = 120 :=
by 
  -- Proof is omitted
  sorry

end factorial_div_eq_l133_133328


namespace end_digit_of_number_l133_133452

theorem end_digit_of_number (n : ℕ) (h_n : n = 2022) (h_start : ∃ (f : ℕ → ℕ), f 0 = 4 ∧ 
    (∀ i < n - 1, (19 ∣ (10 * f i + f (i + 1))) ∨ (23 ∣ (10 * f i + f (i + 1))))) :
  ∃ (f : ℕ → ℕ), f (n - 1) = 8 :=
by {
  sorry
}

end end_digit_of_number_l133_133452


namespace math_problem_l133_133977

theorem math_problem (x y n : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
  (hy_reverse : ∃ a b, x = 10 * a + b ∧ y = 10 * b + a) 
  (h_xy_square_sum : x^2 + y^2 = n^2) : x + y + n = 132 :=
sorry

end math_problem_l133_133977


namespace product_of_16_and_21_point_3_l133_133624

theorem product_of_16_and_21_point_3 (h1 : 213 * 16 = 3408) : 16 * 21.3 = 340.8 :=
by sorry

end product_of_16_and_21_point_3_l133_133624


namespace certain_number_is_18_l133_133764

theorem certain_number_is_18 (p q : ℚ) (h₁ : 3 / p = 8) (h₂ : p - q = 0.20833333333333334) : 3 / q = 18 :=
sorry

end certain_number_is_18_l133_133764


namespace sum_of_consecutive_neg_ints_l133_133036

theorem sum_of_consecutive_neg_ints (n : ℤ) (h : n * (n + 1) = 2720) (hn : n < 0) (hn_plus1 : n + 1 < 0) :
  n + (n + 1) = -105 :=
sorry

end sum_of_consecutive_neg_ints_l133_133036


namespace cubic_polynomial_roots_l133_133739

theorem cubic_polynomial_roots (a : ℚ) :
  (x^3 - 6*x^2 + a*x - 6 = 0) ∧ (x = 3) → (x = 1 ∨ x = 2 ∨ x = 3) :=
by
  sorry

end cubic_polynomial_roots_l133_133739


namespace smallest_percent_increase_from_2_to_3_l133_133482

def percent_increase (initial final : ℕ) : ℚ := 
  ((final - initial : ℕ) : ℚ) / (initial : ℕ) * 100

def value_at_question : ℕ → ℕ
| 1 => 100
| 2 => 200
| 3 => 300
| 4 => 500
| 5 => 1000
| 6 => 2000
| 7 => 4000
| 8 => 8000
| 9 => 16000
| 10 => 32000
| 11 => 64000
| 12 => 125000
| 13 => 250000
| 14 => 500000
| 15 => 1000000
| _ => 0  -- Default case for questions out of range

theorem smallest_percent_increase_from_2_to_3 :
  let p1 := percent_increase (value_at_question 1) (value_at_question 2)
  let p2 := percent_increase (value_at_question 2) (value_at_question 3)
  let p3 := percent_increase (value_at_question 3) (value_at_question 4)
  let p11 := percent_increase (value_at_question 11) (value_at_question 12)
  let p14 := percent_increase (value_at_question 14) (value_at_question 15)
  p2 < p1 ∧ p2 < p3 ∧ p2 < p11 ∧ p2 < p14 :=
by
  sorry

end smallest_percent_increase_from_2_to_3_l133_133482


namespace Ronaldinho_age_2018_l133_133190

variable (X : ℕ)

theorem Ronaldinho_age_2018 (h : X^2 = 2025) : X - (2025 - 2018) = 38 := by
  sorry

end Ronaldinho_age_2018_l133_133190


namespace change_in_f_when_x_increased_by_2_change_in_f_when_x_decreased_by_2_l133_133136

-- Given f(x) = x^2 - 5x
def f (x : ℝ) : ℝ := x^2 - 5 * x

-- Prove the change in f(x) when x is increased by 2 is 4x - 6
theorem change_in_f_when_x_increased_by_2 (x : ℝ) : f (x + 2) - f x = 4 * x - 6 := by
  sorry

-- Prove the change in f(x) when x is decreased by 2 is -4x + 14
theorem change_in_f_when_x_decreased_by_2 (x : ℝ) : f (x - 2) - f x = -4 * x + 14 := by
  sorry

end change_in_f_when_x_increased_by_2_change_in_f_when_x_decreased_by_2_l133_133136


namespace shirt_final_price_is_correct_l133_133432

noncomputable def final_price_percentage (initial_price : ℝ) : ℝ :=
  let first_discount := initial_price * 0.80
  let second_discount := first_discount * 0.90
  let anniversary_addition := second_discount * 1.05
  let final_price := anniversary_addition * 1.15
  final_price / initial_price * 100

theorem shirt_final_price_is_correct (initial_price : ℝ) : final_price_percentage initial_price = 86.94 := by
  sorry

end shirt_final_price_is_correct_l133_133432


namespace correct_calculation_l133_133886

theorem correct_calculation (x y : ℝ) : (x^2 * y)^3 = x^6 * y^3 :=
  sorry

end correct_calculation_l133_133886


namespace monotonicity_of_f_range_of_k_for_three_zeros_l133_133992

noncomputable def f (x k : ℝ) : ℝ := x^3 - k * x + k^2

def f_derivative (x k : ℝ) : ℝ := 3 * x^2 - k

theorem monotonicity_of_f (k : ℝ) : 
  (∀ x : ℝ, 0 <= f_derivative x k) ↔ k <= 0 :=
by sorry

theorem range_of_k_for_three_zeros : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 k = 0 ∧ f x2 k = 0 ∧ f x3 k = 0) ↔ (0 < k ∧ k < 4 / 27) :=
by sorry

end monotonicity_of_f_range_of_k_for_three_zeros_l133_133992


namespace jennifer_cards_left_l133_133004

-- Define the initial number of cards and the number of cards eaten
def initial_cards : ℕ := 72
def eaten_cards : ℕ := 61

-- Define the final number of cards
def final_cards (initial_cards eaten_cards : ℕ) : ℕ :=
  initial_cards - eaten_cards

-- Proposition stating that Jennifer has 11 cards left
theorem jennifer_cards_left : final_cards initial_cards eaten_cards = 11 :=
by
  -- Proof here
  sorry

end jennifer_cards_left_l133_133004


namespace total_players_is_28_l133_133530

def total_players (A B C AB BC AC ABC : ℕ) : ℕ :=
  A + B + C - (AB + BC + AC) + ABC

theorem total_players_is_28 :
  total_players 10 15 18 8 6 4 3 = 28 :=
by
  -- as per inclusion-exclusion principle
  -- T = A + B + C - (AB + BC + AC) + ABC
  -- substituting given values we repeatedly perform steps until final answer
  -- take user inputs to build your final answer.
  sorry

end total_players_is_28_l133_133530


namespace temperature_below_zero_l133_133930

-- Assume the basic definitions and context needed
def above_zero (temp : Int) := temp > 0
def below_zero (temp : Int) := temp < 0

theorem temperature_below_zero (t1 t2 : Int) (h1 : above_zero t1) (h2 : t2 = -7) :
  below_zero t2 := by 
  -- This is where the proof would go
  sorry

end temperature_below_zero_l133_133930


namespace multiply_negatives_l133_133810

theorem multiply_negatives : (- (1 / 2)) * (- 2) = 1 :=
by
  sorry

end multiply_negatives_l133_133810


namespace heaviest_object_can_be_weighed_is_40_number_of_different_weights_is_40_l133_133502

def weights : List ℕ := [1, 3, 9, 27]

theorem heaviest_object_can_be_weighed_is_40 : 
  List.sum weights = 40 :=
by
  sorry

theorem number_of_different_weights_is_40 :
  List.range (List.sum weights) = List.range 40 :=
by
  sorry

end heaviest_object_can_be_weighed_is_40_number_of_different_weights_is_40_l133_133502


namespace problem1_problem2_l133_133077

-- Problem 1: Sequence "Seven six five four three two one" is a descending order
theorem problem1 : ∃ term: String, term = "Descending Order" ∧ "Seven six five four three two one" = "Descending Order" := sorry

-- Problem 2: Describing a computing tool that knows 0 and 1 and can calculate large numbers (computer)
theorem problem2 : ∃ tool: String, tool = "Computer" ∧ "I only know 0 and 1, can calculate millions and billions, available in both software and hardware" = "Computer" := sorry

end problem1_problem2_l133_133077


namespace distance_between_parallel_lines_l133_133713

theorem distance_between_parallel_lines
  (line1 : ∀ (x y : ℝ), 3*x - 2*y - 1 = 0)
  (line2 : ∀ (x y : ℝ), 3*x - 2*y + 1 = 0) :
  ∃ d : ℝ, d = (2 * Real.sqrt 13) / 13 :=
by
  sorry

end distance_between_parallel_lines_l133_133713


namespace final_temperature_is_100_l133_133555

-- Definitions based on conditions
def initial_temperature := 20  -- in degrees
def heating_rate := 5          -- in degrees per minute
def heating_time := 16         -- in minutes

-- The proof statement
theorem final_temperature_is_100 :
  initial_temperature + heating_rate * heating_time = 100 := by
  sorry

end final_temperature_is_100_l133_133555


namespace students_playing_both_football_and_cricket_l133_133477

theorem students_playing_both_football_and_cricket
  (total_students : ℕ)
  (students_playing_football : ℕ)
  (students_playing_cricket : ℕ)
  (students_neither_football_nor_cricket : ℕ) :
  total_students = 250 →
  students_playing_football = 160 →
  students_playing_cricket = 90 →
  students_neither_football_nor_cricket = 50 →
  (students_playing_football + students_playing_cricket - (total_students - students_neither_football_nor_cricket)) = 50 :=
by
  intros h_total h_football h_cricket h_neither
  sorry

end students_playing_both_football_and_cricket_l133_133477


namespace wheel_radius_correct_l133_133438
noncomputable def wheel_radius (total_distance : ℝ) (n_revolutions : ℕ) : ℝ :=
  total_distance / (n_revolutions * 2 * Real.pi)

theorem wheel_radius_correct :
  wheel_radius 450.56 320 = 0.224 :=
by
  sorry

end wheel_radius_correct_l133_133438


namespace area_percentage_increase_l133_133422

theorem area_percentage_increase (r1 r2 : ℝ) (π : ℝ) (area1 area2 : ℝ) (N : ℝ) :
  r1 = 6 → r2 = 4 → area1 = π * r1 ^ 2 → area2 = π * r2 ^ 2 →
  N = 125 →
  ((area1 - area2) / area2) * 100 = N :=
by {
  sorry
}

end area_percentage_increase_l133_133422


namespace earliest_meeting_time_l133_133742

theorem earliest_meeting_time
    (charlie_lap : ℕ := 5)
    (ben_lap : ℕ := 8)
    (laura_lap_effective : ℕ := 11) :
    lcm (lcm charlie_lap ben_lap) laura_lap_effective = 440 := by
  sorry

end earliest_meeting_time_l133_133742


namespace range_of_k_l133_133469

theorem range_of_k (x y k : ℝ) (h1 : x - y = k - 1) (h2 : 3 * x + 2 * y = 4 * k + 5) (hk : 2 * x + 3 * y > 7) : k > 1 / 3 := 
sorry

end range_of_k_l133_133469


namespace binomial_expansion_l133_133996

theorem binomial_expansion : 
  (102: ℕ)^4 - 4 * (102: ℕ)^3 + 6 * (102: ℕ)^2 - 4 * (102: ℕ) + 1 = (101: ℕ)^4 :=
by sorry

end binomial_expansion_l133_133996


namespace rationalize_denominator_l133_133433

theorem rationalize_denominator (A B C : ℤ) (h : A + B * Real.sqrt C = -(9) - 4 * Real.sqrt 5) : A * B * C = 180 :=
by
  have hA : A = -9 := by sorry
  have hB : B = -4 := by sorry
  have hC : C = 5 := by sorry
  rw [hA, hB, hC]
  norm_num

end rationalize_denominator_l133_133433


namespace evaluate_expression_l133_133988

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 :=
by sorry

end evaluate_expression_l133_133988


namespace coefficient_x5_in_product_l133_133826

noncomputable def P : Polynomial ℤ := 
  Polynomial.C 1 * Polynomial.X ^ 6 +
  Polynomial.C (-2) * Polynomial.X ^ 5 +
  Polynomial.C 3 * Polynomial.X ^ 4 +
  Polynomial.C (-4) * Polynomial.X ^ 3 +
  Polynomial.C 5 * Polynomial.X ^ 2 +
  Polynomial.C (-6) * Polynomial.X +
  Polynomial.C 7

noncomputable def Q : Polynomial ℤ := 
  Polynomial.C 3 * Polynomial.X ^ 4 +
  Polynomial.C (-4) * Polynomial.X ^ 3 +
  Polynomial.C 5 * Polynomial.X ^ 2 +
  Polynomial.C 6 * Polynomial.X +
  Polynomial.C (-8)

theorem coefficient_x5_in_product (p q : Polynomial ℤ) :
  (p * q).coeff 5 = 2 :=
by
  have P := 
    Polynomial.C 1 * Polynomial.X ^ 6 +
    Polynomial.C (-2) * Polynomial.X ^ 5 +
    Polynomial.C 3 * Polynomial.X ^ 4 +
    Polynomial.C (-4) * Polynomial.X ^ 3 +
    Polynomial.C 5 * Polynomial.X ^ 2 +
    Polynomial.C (-6) * Polynomial.X +
    Polynomial.C 7
  have Q := 
    Polynomial.C 3 * Polynomial.X ^ 4 +
    Polynomial.C (-4) * Polynomial.X ^ 3 +
    Polynomial.C 5 * Polynomial.X ^ 2 +
    Polynomial.C 6 * Polynomial.X +
    Polynomial.C (-8)

  sorry

end coefficient_x5_in_product_l133_133826


namespace math_problem_l133_133893

variable {a b c d e f : ℕ}
variable (h1 : f < a)
variable (h2 : (a * b * d + 1) % c = 0)
variable (h3 : (a * c * e + 1) % b = 0)
variable (h4 : (b * c * f + 1) % a = 0)

theorem math_problem
  (h5 : (d : ℚ) / c < 1 - (e : ℚ) / b) :
  (d : ℚ) / c < 1 - (f : ℚ) / a :=
by {
  skip -- Adding "by" ... "sorry" to make the statement complete since no proof is required.
  sorry
}

end math_problem_l133_133893


namespace polygon_sides_eq_n_l133_133493

theorem polygon_sides_eq_n
  (sum_except_two_angles : ℝ)
  (angle_equal : ℝ)
  (h1 : sum_except_two_angles = 2970)
  (h2 : angle_equal * 2 < 180)
  : ∃ n : ℕ, 180 * (n - 2) = 2970 + 2 * angle_equal ∧ n = 19 :=
by
  sorry

end polygon_sides_eq_n_l133_133493


namespace tires_in_parking_lot_l133_133533

theorem tires_in_parking_lot (n : ℕ) (m : ℕ) (h : 30 = n) (h' : m = 5) : n * m = 150 := by
  sorry

end tires_in_parking_lot_l133_133533


namespace total_albums_l133_133396

-- Defining the initial conditions
def albumsAdele : ℕ := 30
def albumsBridget : ℕ := albumsAdele - 15
def albumsKatrina : ℕ := 6 * albumsBridget
def albumsMiriam : ℕ := 7 * albumsKatrina
def albumsCarlos : ℕ := 3 * albumsMiriam
def albumsDiane : ℕ := 2 * albumsKatrina

-- Proving the total number of albums
theorem total_albums :
  albumsAdele + albumsBridget + albumsKatrina + albumsMiriam + albumsCarlos + albumsDiane = 2835 :=
by
  sorry

end total_albums_l133_133396


namespace largest_domain_of_f_l133_133354

theorem largest_domain_of_f (f : ℝ → ℝ) (dom : ℝ → Prop) :
  (∀ x : ℝ, dom x → dom (1 / x)) →
  (∀ x : ℝ, dom x → (f x + f (1 / x) = x)) →
  (∀ x : ℝ, dom x ↔ x = 1 ∨ x = -1) :=
by
  intro h1 h2
  sorry

end largest_domain_of_f_l133_133354


namespace xy_equals_twelve_l133_133934

theorem xy_equals_twelve (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 :=
by
  sorry

end xy_equals_twelve_l133_133934


namespace dot_product_zero_l133_133449

-- Define vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, 3)

-- Define the dot product operation for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the scalar multiplication and vector subtraction for 2D vectors
def scalar_mul_vec (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Now we state the theorem we want to prove
theorem dot_product_zero : dot_product a (vec_sub (scalar_mul_vec 2 a) b) = 0 := 
by
  sorry

end dot_product_zero_l133_133449


namespace total_bill_l133_133890

theorem total_bill (m : ℝ) (h1 : m = 10 * (m / 10 + 3) - 27) : m = 270 :=
by
  sorry

end total_bill_l133_133890


namespace part1_a_eq_neg1_inter_part1_a_eq_neg1_union_part2_a_range_l133_133953

def A (x : ℝ) : Prop := x^2 - 4 * x - 5 ≥ 0
def B (x a : ℝ) : Prop := 2 * a ≤ x ∧ x ≤ a + 2

theorem part1_a_eq_neg1_inter (a : ℝ) (h : a = -1) : 
  {x : ℝ | A x} ∩ {x : ℝ | B x a} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
by sorry

theorem part1_a_eq_neg1_union (a : ℝ) (h : a = -1) : 
  {x : ℝ | A x} ∪ {x : ℝ | B x a} = {x : ℝ | x ≤ 1 ∨ x ≥ 5} :=
by sorry

theorem part2_a_range (a : ℝ) : 
  ({x : ℝ | A x} ∩ {x : ℝ | B x a} = {x : ℝ | B x a}) → 
  a ∈ {a : ℝ | a > 2 ∨ a ≤ -3} :=
by sorry

end part1_a_eq_neg1_inter_part1_a_eq_neg1_union_part2_a_range_l133_133953


namespace prime_square_mod_12_l133_133152

theorem prime_square_mod_12 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) : p^2 % 12 = 1 := 
by
  sorry

end prime_square_mod_12_l133_133152


namespace largest_integer_sol_l133_133035

theorem largest_integer_sol (x : ℤ) : (3 * x + 4 < 5 * x - 2) -> x = 3 :=
by
  sorry

end largest_integer_sol_l133_133035


namespace train_speed_correct_l133_133734

def length_of_train := 280 -- in meters
def time_to_pass_tree := 16 -- in seconds
def speed_of_train := 63 -- in km/hr

theorem train_speed_correct :
  (length_of_train / time_to_pass_tree) * (3600 / 1000) = speed_of_train :=
sorry

end train_speed_correct_l133_133734


namespace determine_pairs_l133_133483

theorem determine_pairs (a b : ℕ) (h : 2017^a = b^6 - 32 * b + 1) : 
  (a = 0 ∧ b = 0) ∨ (a = 0 ∧ b = 2) :=
by sorry

end determine_pairs_l133_133483


namespace cos_180_eq_neg_one_l133_133104

/-- The cosine of 180 degrees is -1. -/
theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 :=
by sorry

end cos_180_eq_neg_one_l133_133104


namespace average_rainfall_l133_133199

theorem average_rainfall (rainfall_Tuesday : ℝ) (rainfall_others : ℝ) (days_in_week : ℝ)
  (h1 : rainfall_Tuesday = 10.5) 
  (h2 : rainfall_Tuesday = rainfall_others)
  (h3 : days_in_week = 7) : 
  (rainfall_Tuesday + rainfall_others) / days_in_week = 3 :=
by
  sorry

end average_rainfall_l133_133199


namespace ethanol_percentage_in_fuel_B_l133_133080

theorem ethanol_percentage_in_fuel_B 
  (tank_capacity : ℕ)
  (fuel_A_vol : ℕ)
  (ethanol_in_A_percentage : ℝ)
  (ethanol_total : ℝ)
  (ethanol_A_vol : ℝ)
  (fuel_B_vol : ℕ)
  (ethanol_B_vol : ℝ)
  (ethanol_B_percentage : ℝ) 
  (h1 : tank_capacity = 204)
  (h2 : fuel_A_vol = 66)
  (h3 : ethanol_in_A_percentage = 0.12)
  (h4 : ethanol_total = 30)
  (h5 : ethanol_A_vol = fuel_A_vol * ethanol_in_A_percentage)
  (h6 : ethanol_B_vol = ethanol_total - ethanol_A_vol)
  (h7 : fuel_B_vol = tank_capacity - fuel_A_vol)
  (h8 : ethanol_B_percentage = (ethanol_B_vol / fuel_B_vol) * 100) :
  ethanol_B_percentage = 16 :=
by sorry

end ethanol_percentage_in_fuel_B_l133_133080


namespace calculate_selling_prices_l133_133767

noncomputable def selling_prices
  (cost1 cost2 cost3 : ℝ) (profit1 profit2 profit3 : ℝ) : ℝ × ℝ × ℝ :=
  let selling_price1 := cost1 + (profit1 / 100) * cost1
  let selling_price2 := cost2 + (profit2 / 100) * cost2
  let selling_price3 := cost3 + (profit3 / 100) * cost3
  (selling_price1, selling_price2, selling_price3)

theorem calculate_selling_prices :
  selling_prices 500 750 1000 20 25 30 = (600, 937.5, 1300) :=
by
  sorry

end calculate_selling_prices_l133_133767


namespace dividend_percentage_shares_l133_133435

theorem dividend_percentage_shares :
  ∀ (purchase_price market_value : ℝ) (interest_rate : ℝ),
  purchase_price = 56 →
  market_value = 42 →
  interest_rate = 0.12 →
  ( (interest_rate * purchase_price) / market_value * 100 = 16) :=
by
  intros purchase_price market_value interest_rate h1 h2 h3
  rw [h1, h2, h3]
  -- Calculations were done in solution
  sorry

end dividend_percentage_shares_l133_133435


namespace savings_promotion_l133_133627

theorem savings_promotion (reg_price promo_price num_pizzas : ℕ) (h1 : reg_price = 18) (h2 : promo_price = 5) (h3 : num_pizzas = 3) :
  reg_price * num_pizzas - promo_price * num_pizzas = 39 := by
  sorry

end savings_promotion_l133_133627


namespace abc_value_l133_133730

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

end abc_value_l133_133730


namespace five_y_eq_45_over_7_l133_133220

theorem five_y_eq_45_over_7 (x y : ℚ) (h1 : 3 * x + 4 * y = 0) (h2 : x = y - 3) : 5 * y = 45 / 7 := by
  sorry

end five_y_eq_45_over_7_l133_133220


namespace calculate_initial_income_l133_133300

noncomputable def initial_income : Float := 151173.52

theorem calculate_initial_income :
  let I := initial_income
  let children_distribution := 0.30 * I
  let eldest_child_share := (children_distribution / 6) + 0.05 * I
  let remaining_for_wife := 0.40 * I
  let remaining_after_distribution := I - (children_distribution + remaining_for_wife)
  let donation_to_orphanage := 0.10 * remaining_after_distribution
  let remaining_after_donation := remaining_after_distribution - donation_to_orphanage
  let federal_tax := 0.02 * remaining_after_donation
  let final_amount := remaining_after_donation - federal_tax
  final_amount = 40000 :=
by
  sorry

end calculate_initial_income_l133_133300


namespace range_of_k_l133_133359

theorem range_of_k (k : ℝ) : (∀ (x : ℝ), k * x ^ 2 - k * x - 1 < 0) ↔ (-4 < k ∧ k ≤ 0) := 
by 
  sorry

end range_of_k_l133_133359


namespace abscissa_of_tangent_point_l133_133001

theorem abscissa_of_tangent_point (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x, f x = Real.exp x + a * Real.exp (-x))
  (h_odd : ∀ x, (D^[2] f x) = - (D^[2] f (-x)))
  (slope_cond : ∀ x, (D f x) = 3 / 2) : 
  ∃ x ∈ Set.Ioo (-Real.log 2) (Real.log 2), x = Real.log 2 :=
by
  sorry

end abscissa_of_tangent_point_l133_133001


namespace hostel_provisions_l133_133849

theorem hostel_provisions (x : ℕ) :
  (250 * x = 200 * 60) → x = 48 :=
by
  sorry

end hostel_provisions_l133_133849


namespace intersection_of_A_and_B_l133_133468

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := 
by
  sorry

end intersection_of_A_and_B_l133_133468


namespace jogging_time_l133_133621

theorem jogging_time (distance : ℝ) (speed : ℝ) (h1 : distance = 25) (h2 : speed = 5) : (distance / speed) = 5 :=
by
  rw [h1, h2]
  norm_num

end jogging_time_l133_133621


namespace amount_sharpened_off_l133_133665

-- Defining the initial length of the pencil
def initial_length : ℕ := 31

-- Defining the length of the pencil after sharpening
def after_sharpening_length : ℕ := 14

-- Proving the amount sharpened off the pencil
theorem amount_sharpened_off : initial_length - after_sharpening_length = 17 := 
by 
  -- Here we would insert the proof steps, 
  -- but as instructed we leave it as sorry.
  sorry

end amount_sharpened_off_l133_133665


namespace total_trees_after_planting_l133_133479

-- Define the initial counts of the trees
def initial_maple_trees : ℕ := 2
def initial_poplar_trees : ℕ := 5
def initial_oak_trees : ℕ := 4

-- Define the planting rules
def maple_trees_planted (initial_maple : ℕ) : ℕ := 3 * initial_maple
def poplar_trees_planted (initial_poplar : ℕ) : ℕ := 3 * initial_poplar

-- Calculate the total number of each type of tree after planting
def total_maple_trees (initial_maple : ℕ) : ℕ :=
  initial_maple + maple_trees_planted initial_maple

def total_poplar_trees (initial_poplar : ℕ) : ℕ :=
  initial_poplar + poplar_trees_planted initial_poplar

def total_oak_trees (initial_oak : ℕ) : ℕ := initial_oak

-- Calculate the total number of trees in the park
def total_trees (initial_maple initial_poplar initial_oak : ℕ) : ℕ :=
  total_maple_trees initial_maple + total_poplar_trees initial_poplar + total_oak_trees initial_oak

-- The proof statement
theorem total_trees_after_planting :
  total_trees initial_maple_trees initial_poplar_trees initial_oak_trees = 32 := 
by
  -- Proof placeholder
  sorry

end total_trees_after_planting_l133_133479


namespace distance_each_player_runs_l133_133746

-- Definitions based on conditions
def length : ℝ := 100
def width : ℝ := 50
def laps : ℝ := 6

def perimeter (l w : ℝ) : ℝ := 2 * (l + w)

def total_distance (l w laps : ℝ) : ℝ := laps * perimeter l w

-- Theorem statement
theorem distance_each_player_runs :
  total_distance length width laps = 1800 := 
by 
  sorry

end distance_each_player_runs_l133_133746


namespace range_of_a_l133_133125

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2*x + a ≥ 0) ↔ (1 ≤ a) :=
by sorry

end range_of_a_l133_133125


namespace intersection_M_N_l133_133516

def M : Set ℝ := { x | x ≤ 4 }
def N : Set ℝ := { x | 0 < x }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x ≤ 4 } := 
by 
  sorry

end intersection_M_N_l133_133516


namespace sum_of_octahedron_faces_l133_133796

theorem sum_of_octahedron_faces (n : ℕ) :
  n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) = 8 * n + 28 :=
by
  sorry

end sum_of_octahedron_faces_l133_133796


namespace find_primes_l133_133740

open Int

theorem find_primes (p x y : ℕ) (hp : Nat.Prime p) (hx : 0 < x) (hy : 0 < y) :
  p ^ x = y ^ 3 + 1 ↔ (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
by
  sorry

end find_primes_l133_133740


namespace ice_cream_vendor_l133_133914

theorem ice_cream_vendor (choco : ℕ) (mango : ℕ) (sold_choco : ℚ) (sold_mango : ℚ) 
  (h_choco : choco = 50) (h_mango : mango = 54) (h_sold_choco : sold_choco = 3/5) 
  (h_sold_mango : sold_mango = 2/3) : 
  choco - (choco * sold_choco) + mango - (mango * sold_mango) = 38 := 
by 
  sorry

end ice_cream_vendor_l133_133914


namespace donor_multiple_l133_133924

def cost_per_box (food_cost : ℕ) (supplies_cost : ℕ) : ℕ := food_cost + supplies_cost

def total_initial_cost (num_boxes : ℕ) (cost_per_box : ℕ) : ℕ := num_boxes * cost_per_box

def additional_boxes (total_boxes : ℕ) (initial_boxes : ℕ) : ℕ := total_boxes - initial_boxes

def donor_contribution (additional_boxes : ℕ) (cost_per_box : ℕ) : ℕ := additional_boxes * cost_per_box

def multiple (donor_contribution : ℕ) (initial_cost : ℕ) : ℕ := donor_contribution / initial_cost

theorem donor_multiple 
    (initial_boxes : ℕ) (box_cost : ℕ) (total_boxes : ℕ) (donor_multi : ℕ)
    (h1 : initial_boxes = 400) 
    (h2 : box_cost = 245) 
    (h3 : total_boxes = 2000)
    : donor_multi = 4 :=
by
    let initial_cost := total_initial_cost initial_boxes box_cost
    let additional_boxes := additional_boxes total_boxes initial_boxes
    let contribution := donor_contribution additional_boxes box_cost
    have h4 : contribution = 392000 := sorry
    have h5 : initial_cost = 98000 := sorry
    have h6 : donor_multi = contribution / initial_cost := sorry
    -- Therefore, the multiple should be 4
    exact sorry

end donor_multiple_l133_133924


namespace girls_picked_more_l133_133450

variable (N I A V : ℕ)

theorem girls_picked_more (h1 : N > A) (h2 : N > V) (h3 : N > I)
                         (h4 : I ≥ A) (h5 : I ≥ V) (h6 : A > V) :
  N + I > A + V := by
  sorry

end girls_picked_more_l133_133450


namespace common_tangents_l133_133835

theorem common_tangents (r1 r2 d : ℝ) (h_r1 : r1 = 10) (h_r2 : r2 = 4) : 
  ∀ (n : ℕ), (n = 1) → ¬ (∃ (d : ℝ), 
    (6 < d ∧ d < 14 ∧ n = 2) ∨ 
    (d = 14 ∧ n = 3) ∨ 
    (d < 6 ∧ n = 0) ∨ 
    (d > 14 ∧ n = 4)) :=
by
  intro n h
  sorry

end common_tangents_l133_133835


namespace prod_mod_6_l133_133919

theorem prod_mod_6 (h1 : 2015 % 6 = 3) (h2 : 2016 % 6 = 0) (h3 : 2017 % 6 = 1) (h4 : 2018 % 6 = 2) : 
  (2015 * 2016 * 2017 * 2018) % 6 = 0 := 
by 
  sorry

end prod_mod_6_l133_133919


namespace line_through_point_bisected_by_hyperbola_l133_133865

theorem line_through_point_bisected_by_hyperbola :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * 3 + b * (-1) + c = 0) ∧
  (∀ x y : ℝ, (x^2 / 4 - y^2 = 1) → (a * x + b * y + c = 0)) ↔ (a = 3 ∧ b = 4 ∧ c = -5) :=
by
  sorry

end line_through_point_bisected_by_hyperbola_l133_133865


namespace given_even_function_and_monotonic_increasing_l133_133603

-- Define f as an even function on ℝ
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

-- Define that f is monotonically increasing on (-∞, 0)
def is_monotonically_increasing_on_negatives (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → y < 0 → f (x) < f (y)

-- Theorem statement
theorem given_even_function_and_monotonic_increasing {
  f : ℝ → ℝ
} (h_even : is_even_function f)
  (h_monotonic : is_monotonically_increasing_on_negatives f) :
  f (1) > f (-2) :=
sorry

end given_even_function_and_monotonic_increasing_l133_133603


namespace sequence_equal_l133_133047

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

end sequence_equal_l133_133047


namespace find_a_l133_133134

theorem find_a (a : ℝ) (i : ℂ) (h : i * i = -1) (h1 : (1 + a * i) * i = -3 + i) : a = 3 :=
by
  sorry

end find_a_l133_133134


namespace find_integer_K_l133_133847

-- Definitions based on the conditions
def is_valid_K (K Z : ℤ) : Prop :=
  Z = K^4 ∧ 3000 < Z ∧ Z < 4000 ∧ K > 1 ∧ ∃ (z : ℤ), K^4 = z^3

theorem find_integer_K :
  ∃ (K : ℤ), is_valid_K K 2401 :=
by
  sorry

end find_integer_K_l133_133847


namespace nth_term_arithmetic_seq_l133_133931

variable (a_n : Nat → Int)
variable (S : Nat → Int)
variable (a_1 : Int)

-- Conditions
def is_arithmetic_sequence (a_n : Nat → Int) : Prop :=
  ∃ d : Int, ∀ n : Nat, a_n (n + 1) = a_n n + d

def first_term (a_1 : Int) : Prop :=
  a_1 = 1

def sum_first_three_terms (S : Nat → Int) : Prop :=
  S 3 = 9

theorem nth_term_arithmetic_seq :
  (is_arithmetic_sequence a_n) →
  (first_term 1) →
  (sum_first_three_terms S) →
  ∀ n : Nat, a_n n = 2 * n - 1 :=
  sorry

end nth_term_arithmetic_seq_l133_133931


namespace num_divisors_720_l133_133922

-- Define the number 720 and its prime factorization
def n : ℕ := 720
def pf : List (ℕ × ℕ) := [(2, 4), (3, 2), (5, 1)]

-- Define the function to calculate the number of divisors from prime factorization
def num_divisors (pf : List (ℕ × ℕ)) : ℕ :=
  pf.foldr (λ p acc => acc * (p.snd + 1)) 1

-- Statement to prove
theorem num_divisors_720 : num_divisors pf = 30 :=
  by
  -- Placeholder for the actual proof
  sorry

end num_divisors_720_l133_133922


namespace find_sticker_price_l133_133907

-- Define the conditions and the question
def storeA_price (x : ℝ) : ℝ := 0.80 * x - 80
def storeB_price (x : ℝ) : ℝ := 0.70 * x - 40
def heather_saves_30 (x : ℝ) : Prop := storeA_price x = storeB_price x + 30

-- Define the main theorem
theorem find_sticker_price : ∃ x : ℝ, heather_saves_30 x ∧ x = 700 :=
by
  sorry

end find_sticker_price_l133_133907


namespace central_angle_of_region_l133_133832

theorem central_angle_of_region (A : ℝ) (θ : ℝ) (h : (1:ℝ) / 8 = (θ / 360) * A / A) : θ = 45 :=
by
  sorry

end central_angle_of_region_l133_133832


namespace sonika_years_in_bank_l133_133646

variable (P A1 A2 : ℚ)
variables (r t : ℚ)

def simple_interest (P r t : ℚ) : ℚ := P * r * t / 100
def amount_with_interest (P r t : ℚ) : ℚ := P + simple_interest P r t

theorem sonika_years_in_bank :
  P = 9000 → A1 = 10200 → A2 = 10740 →
  amount_with_interest P r t = A1 →
  amount_with_interest P (r + 2) t = A2 →
  t = 3 :=
by
  intros hP hA1 hA2 hA1_eq hA2_eq
  sorry

end sonika_years_in_bank_l133_133646


namespace simplify_expression_l133_133248

theorem simplify_expression (x : ℝ) : 7 * x + 9 - 3 * x + 15 * 2 = 4 * x + 39 := 
by sorry

end simplify_expression_l133_133248


namespace ratio_rocks_eaten_to_collected_l133_133784

def rocks_collected : ℕ := 10
def rocks_left : ℕ := 7
def rocks_spit_out : ℕ := 2

theorem ratio_rocks_eaten_to_collected : 
  (rocks_collected - rocks_left + rocks_spit_out) * 2 = rocks_collected := 
by 
  sorry

end ratio_rocks_eaten_to_collected_l133_133784


namespace no_such_n_l133_133486

theorem no_such_n (n : ℕ) (h_positive : n > 0) : 
  ¬ ∃ k : ℕ, (n^2 + 1) = k * (Nat.floor (Real.sqrt n))^2 + 2 := by
  sorry

end no_such_n_l133_133486


namespace kamal_chemistry_marks_l133_133625

-- Definitions of the marks
def english_marks : ℕ := 76
def math_marks : ℕ := 60
def physics_marks : ℕ := 72
def biology_marks : ℕ := 82
def average_marks : ℕ := 71
def num_subjects : ℕ := 5

-- Statement to be proved
theorem kamal_chemistry_marks : ∃ (chemistry_marks : ℕ), 
  76 + 60 + 72 + 82 + chemistry_marks = 71 * 5 :=
by
sorry

end kamal_chemistry_marks_l133_133625


namespace range_of_function_l133_133414

noncomputable def function_y (x : ℝ) : ℝ := -x^2 - 2 * x + 3

theorem range_of_function : 
  ∃ (a b : ℝ), a = -12 ∧ b = 4 ∧ 
  (∀ y, (∃ x, -5 ≤ x ∧ x ≤ 0 ∧ y = function_y x) ↔ a ≤ y ∧ y ≤ b) :=
sorry

end range_of_function_l133_133414


namespace calculate_expression_l133_133701

theorem calculate_expression :
  57.6 * (8 / 5) + 28.8 * (184 / 5) - 14.4 * 80 + 10.5 = 10.5 :=
by
  sorry

end calculate_expression_l133_133701


namespace arrangement_of_accommodation_l133_133897

open Nat

noncomputable def num_arrangements_accommodation : ℕ :=
  (factorial 13) / ((factorial 2) * (factorial 2) * (factorial 2) * (factorial 2))

theorem arrangement_of_accommodation : num_arrangements_accommodation = 389188800 := by
  sorry

end arrangement_of_accommodation_l133_133897


namespace polynomial_solution_l133_133614

variable (P : ℝ → ℝ → ℝ)

theorem polynomial_solution :
  (∀ x y : ℝ, P (x + y) (x - y) = 2 * P x y) →
  (∃ b c d : ℝ, ∀ x y : ℝ, P x y = b * x^2 + c * x * y + d * y^2) :=
by
  sorry

end polynomial_solution_l133_133614


namespace reciprocal_of_neg_2023_l133_133460

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l133_133460


namespace medal_award_ways_l133_133573

open Nat

theorem medal_award_ways :
  let sprinters := 10
  let italians := 4
  let medals := 3
  let gold_medal_ways := choose italians 1
  let remaining_sprinters := sprinters - 1
  let non_italians := remaining_sprinters - (italians - 1)
  let silver_medal_ways := choose non_italians 1
  let new_remaining_sprinters := remaining_sprinters - 1
  let new_non_italians := new_remaining_sprinters - (italians - 1)
  let bronze_medal_ways := choose new_non_italians 1
  gold_medal_ways * silver_medal_ways * bronze_medal_ways = 120 := by
    sorry

end medal_award_ways_l133_133573


namespace eval_abc_l133_133017

theorem eval_abc (a b c : ℚ) (h1 : a = 1 / 2) (h2 : b = 3 / 4) (h3 : c = 8) :
  a^3 * b^2 * c = 9 / 16 :=
by
  sorry

end eval_abc_l133_133017


namespace max_gcd_of_15m_plus_4_and_14m_plus_3_l133_133456

theorem max_gcd_of_15m_plus_4_and_14m_plus_3 (m : ℕ) (hm : 0 < m) :
  ∃ k : ℕ, k = gcd (15 * m + 4) (14 * m + 3) ∧ k = 11 :=
by {
  sorry
}

end max_gcd_of_15m_plus_4_and_14m_plus_3_l133_133456


namespace term_in_census_is_population_l133_133290

def term_for_entire_set_of_objects : String :=
  "population"

theorem term_in_census_is_population :
  term_for_entire_set_of_objects = "population" :=
sorry

end term_in_census_is_population_l133_133290


namespace seventh_term_correct_l133_133426

noncomputable def seventh_term_geometric_sequence (a r : ℝ) (h1 : a = 5) (h2 : a * r = 1/5) : ℝ :=
  a * r ^ 6

theorem seventh_term_correct :
  seventh_term_geometric_sequence 5 (1/25) (by rfl) (by norm_num) = 1 / 48828125 :=
  by
    unfold seventh_term_geometric_sequence
    sorry

end seventh_term_correct_l133_133426


namespace cone_height_l133_133304

theorem cone_height (R : ℝ) (h : ℝ) (r : ℝ) : 
  R = 8 → r = 2 → h = 2 * Real.sqrt 15 :=
by
  intro hR hr
  sorry

end cone_height_l133_133304


namespace price_per_rose_is_2_l133_133418

-- Definitions from conditions
def has_amount (total_dollars : ℕ) : Prop := total_dollars = 300
def total_roses (R : ℕ) : Prop := ∃ (j : ℕ) (i : ℕ), R / 3 = j ∧ R / 2 = i ∧ j + i = 125

-- Theorem stating the price per rose
theorem price_per_rose_is_2 (R : ℕ) : 
  has_amount 300 → total_roses R → 300 / R = 2 :=
sorry

end price_per_rose_is_2_l133_133418


namespace intersection_A_B_l133_133397

def A : Set ℝ := { x | abs x < 3 }
def B : Set ℝ := { x | 2 - x > 0 }

theorem intersection_A_B : A ∩ B = { x : ℝ | -3 < x ∧ x < 2 } :=
by
  sorry

end intersection_A_B_l133_133397


namespace cos_double_angle_l133_133407

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 3 / 5) : Real.cos (2 * α) = 7 / 25 :=
sorry

end cos_double_angle_l133_133407


namespace gcd_lcm_find_other_number_l133_133697

theorem gcd_lcm_find_other_number {a b : ℕ} (h_gcd : Nat.gcd a b = 36) (h_lcm : Nat.lcm a b = 8820) (h_a : a = 360) : b = 882 :=
by
  sorry

end gcd_lcm_find_other_number_l133_133697


namespace speed_of_first_car_l133_133717

theorem speed_of_first_car (v : ℝ) (h1 : 2.5 * v + 2.5 * 45 = 175) : v = 25 :=
by
  sorry

end speed_of_first_car_l133_133717


namespace value_of_g_at_3_l133_133222

def g (x : ℝ) := x^2 - 2*x + 1

theorem value_of_g_at_3 : g 3 = 4 :=
by
  sorry

end value_of_g_at_3_l133_133222


namespace trig_identity_tan_solutions_l133_133073

open Real

theorem trig_identity_tan_solutions :
  ∃ α β : ℝ, (tan α) * (tan β) = -3 ∧ (tan α) + (tan β) = 3 ∧
  abs (sin (α + β) ^ 2 - 3 * sin (α + β) * cos (α + β) - 3 * cos (α + β) ^ 2) = 3 :=
by
  have: ∀ x : ℝ, x^2 - 3*x - 3 = 0 → x = (3 + sqrt 21) / 2 ∨ x = (3 - sqrt 21) / 2 := sorry
  sorry

end trig_identity_tan_solutions_l133_133073


namespace part_a_l133_133140

def is_tricubic (k : ℕ) : Prop :=
  ∃ a b c : ℕ, k = a^3 + b^3 + c^3

theorem part_a : ∃ (n : ℕ), is_tricubic n ∧ ¬ is_tricubic (n + 2) ∧ ¬ is_tricubic (n + 28) :=
by 
  let n := 3 * (3*1+1)^3
  exists n
  sorry

end part_a_l133_133140


namespace sandy_savings_l133_133405

theorem sandy_savings (S : ℝ) :
  let last_year_savings := 0.10 * S
  let this_year_salary := 1.10 * S
  let this_year_savings := 1.65 * last_year_savings
  let P := this_year_savings / this_year_salary
  P * 100 = 15 :=
by
  let last_year_savings := 0.10 * S
  let this_year_salary := 1.10 * S
  let this_year_savings := 1.65 * last_year_savings
  let P := this_year_savings / this_year_salary
  have hP : P = 0.165 / 1.10 := by sorry
  have hP_percent : P * 100 = 15 := by sorry
  exact hP_percent

end sandy_savings_l133_133405


namespace value_of_a_minus_b_l133_133434

theorem value_of_a_minus_b (a b : ℝ) (h : (a - 5)^2 + |b^3 - 27| = 0) : a - b = 2 :=
by
  sorry

end value_of_a_minus_b_l133_133434


namespace required_fraction_l133_133463

theorem required_fraction
  (total_members : ℝ)
  (top_10_lists : ℝ) :
  total_members = 775 →
  top_10_lists = 193.75 →
  top_10_lists / total_members = 0.25 :=
by
  sorry

end required_fraction_l133_133463


namespace total_stars_l133_133145

theorem total_stars (g s : ℕ) (hg : g = 10^11) (hs : s = 10^11) : g * s = 10^22 :=
by
  rw [hg, hs]
  sorry

end total_stars_l133_133145


namespace problem_statement_l133_133910

theorem problem_statement (r p q : ℝ) (h1 : r > 0) (h2 : p * q ≠ 0) (h3 : p^2 * r > q^2 * r) : p^2 > q^2 := 
sorry

end problem_statement_l133_133910


namespace birds_more_than_storks_l133_133306

theorem birds_more_than_storks :
  let birds := 6
  let initial_storks := 3
  let additional_storks := 2
  let total_storks := initial_storks + additional_storks
  birds - total_storks = 1 := by
  sorry

end birds_more_than_storks_l133_133306


namespace number_of_ways_to_assign_friends_to_teams_l133_133115

theorem number_of_ways_to_assign_friends_to_teams (n m : ℕ) (h_n : n = 7) (h_m : m = 4) : m ^ n = 16384 :=
by
  rw [h_n, h_m]
  exact pow_succ' 4 6

end number_of_ways_to_assign_friends_to_teams_l133_133115


namespace inequality_proof_l133_133846

theorem inequality_proof
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a^2 + b^2 + c^2 = 1) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ (2 * (a^3 + b^3 + c^3)) / (a * b * c) + 3 :=
by
  sorry

end inequality_proof_l133_133846


namespace carson_seed_l133_133639

variable (s f : ℕ) -- Define the variables s and f as nonnegative integers

-- Conditions given in the problem
axiom h1 : s = 3 * f
axiom h2 : s + f = 60

-- The theorem to prove
theorem carson_seed : s = 45 :=
by
  -- Proof would go here
  sorry

end carson_seed_l133_133639


namespace value_of_k_l133_133523

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 6
def g (x : ℝ) (k : ℝ) : ℝ := x^2 - k * x - 8

theorem value_of_k:
  (f 5) - (g 5 k) = 20 → k = -10.8 :=
by
  sorry

end value_of_k_l133_133523


namespace second_trial_temperatures_l133_133085

-- Definitions based on the conditions
def range_start : ℝ := 60
def range_end : ℝ := 70
def golden_ratio : ℝ := 0.618

-- Calculations for trial temperatures
def lower_trial_temp : ℝ := range_start + (range_end - range_start) * golden_ratio
def upper_trial_temp : ℝ := range_end - (range_end - range_start) * golden_ratio

-- Lean 4 statement to prove the trial temperatures
theorem second_trial_temperatures :
  lower_trial_temp = 66.18 ∧ upper_trial_temp = 63.82 :=
by
  sorry

end second_trial_temperatures_l133_133085


namespace max_items_per_cycle_l133_133293

theorem max_items_per_cycle (shirts : Nat) (pants : Nat) (sweaters : Nat) (jeans : Nat)
  (cycle_time : Nat) (total_time : Nat) 
  (h_shirts : shirts = 18)
  (h_pants : pants = 12)
  (h_sweaters : sweaters = 17)
  (h_jeans : jeans = 13)
  (h_cycle_time : cycle_time = 45)
  (h_total_time : total_time = 3 * 60) :
  (shirts + pants + sweaters + jeans) / (total_time / cycle_time) = 15 :=
by
  -- We will provide the proof here
  sorry

end max_items_per_cycle_l133_133293


namespace tom_bought_new_books_l133_133998

def original_books : ℕ := 5
def sold_books : ℕ := 4
def current_books : ℕ := 39

def new_books (original_books sold_books current_books : ℕ) : ℕ :=
  current_books - (original_books - sold_books)

theorem tom_bought_new_books :
  new_books original_books sold_books current_books = 38 :=
by
  sorry

end tom_bought_new_books_l133_133998


namespace p_sufficient_not_necessary_for_q_l133_133303

-- Definitions based on conditions
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬ (∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l133_133303


namespace percent_non_bikers_play_basketball_l133_133814

noncomputable def total_children (N : ℕ) : ℕ := N
def basketball_players (N : ℕ) : ℕ := 7 * N / 10
def bikers (N : ℕ) : ℕ := 4 * N / 10
def basketball_bikers (N : ℕ) : ℕ := 3 * basketball_players N / 10
def basketball_non_bikers (N : ℕ) : ℕ := basketball_players N - basketball_bikers N
def non_bikers (N : ℕ) : ℕ := N - bikers N

theorem percent_non_bikers_play_basketball (N : ℕ) :
  (basketball_non_bikers N * 100 / non_bikers N) = 82 :=
by sorry

end percent_non_bikers_play_basketball_l133_133814


namespace ping_pong_matches_l133_133876

noncomputable def f (n k : ℕ) : ℕ :=
  Nat.ceil ((n : ℚ) / Nat.ceil ((k : ℚ) / 2))

theorem ping_pong_matches (n k : ℕ) (hn_pos : 0 < n) (hk_le : k ≤ 2 * n - 1) :
  f n k = Nat.ceil ((n : ℚ) / Nat.ceil ((k : ℚ) / 2)) :=
by
  sorry

end ping_pong_matches_l133_133876


namespace little_john_initial_money_l133_133472

def sweets_cost : ℝ := 2.25
def friends_donation : ℝ := 2 * 2.20
def money_left : ℝ := 3.85

theorem little_john_initial_money :
  sweets_cost + friends_donation + money_left = 10.50 :=
by
  sorry

end little_john_initial_money_l133_133472


namespace distance_between_A_and_B_is_90_l133_133194

variable (A B : Type)
variables (v_A v_B v'_A v'_B : ℝ)
variable (d : ℝ)

-- Conditions
axiom starts_simultaneously : True
axiom speed_ratio : v_A / v_B = 4 / 5
axiom A_speed_decrease : v'_A = 0.75 * v_A
axiom B_speed_increase : v'_B = 1.2 * v_B
axiom distance_when_B_reaches_A : ∃ k : ℝ, k = 30 -- Person A is 30 km away from location B

-- Goal
theorem distance_between_A_and_B_is_90 : d = 90 := by 
  sorry

end distance_between_A_and_B_is_90_l133_133194


namespace find_first_5digits_of_M_l133_133408

def last6digits (n : ℕ) : ℕ := n % 1000000

def first5digits (n : ℕ) : ℕ := n / 10

theorem find_first_5digits_of_M (M : ℕ) (h1 : last6digits M = last6digits (M^2)) (h2 : M > 999999) : first5digits M = 60937 := 
by sorry

end find_first_5digits_of_M_l133_133408


namespace q_at_4_l133_133484

def q (x : ℝ) : ℝ := |x - 3|^(1/3) + 3 * |x - 3|^(1/5) + 2 

theorem q_at_4 : q 4 = 6 := by
  sorry

end q_at_4_l133_133484


namespace geometric_seq_value_l133_133659

variable (a : ℕ → ℝ)
variable (g : ∀ n m : ℕ, a n * a m = a ((n + m) / 2) ^ 2)

theorem geometric_seq_value (h1 : a 2 = 1 / 3) (h2 : a 8 = 27) : a 5 = 3 ∨ a 5 = -3 := by
  sorry

end geometric_seq_value_l133_133659


namespace terminating_decimal_expansion_7_over_625_l133_133557

theorem terminating_decimal_expansion_7_over_625 : (7 / 625 : ℚ) = 112 / 10000 := by
  sorry

end terminating_decimal_expansion_7_over_625_l133_133557


namespace girl_attendance_l133_133539

theorem girl_attendance (g b : ℕ) (h1 : g + b = 1500) (h2 : (3 / 4 : ℚ) * g + (1 / 3 : ℚ) * b = 900) :
  (3 / 4 : ℚ) * g = 720 :=
by
  sorry

end girl_attendance_l133_133539


namespace carl_gave_beth_35_coins_l133_133928

theorem carl_gave_beth_35_coins (x : ℕ) (h1 : ∃ n, n = 125) (h2 : ∃ m, m = (125 + x) / 2) (h3 : m = 80) : x = 35 :=
by
  sorry

end carl_gave_beth_35_coins_l133_133928


namespace minimum_chess_pieces_l133_133917

theorem minimum_chess_pieces (n : ℕ) : 
  (n % 3 = 1) ∧ (n % 5 = 3) ∧ (n % 7 = 5) → 
  n = 103 :=
by 
  sorry

end minimum_chess_pieces_l133_133917


namespace find_c_l133_133774

theorem find_c (c d : ℝ) (h1 : c < 0) (h2 : d > 0)
    (max_min_condition : ∀ x, c * Real.cos (d * x) ≤ 3 ∧ c * Real.cos (d * x) ≥ -3) :
    c = -3 :=
by
  -- The statement says if c < 0, d > 0, and given the cosine function hitting max 3 and min -3, then c = -3.
  sorry

end find_c_l133_133774


namespace triangle_solutions_l133_133597

theorem triangle_solutions :
  ∀ (a b c : ℝ) (A B C : ℝ),
  a = 7.012 ∧
  c - b = 1.753 ∧
  B = 38 + 12/60 + 48/3600 ∧
  A = 81 + 47/60 + 12.5/3600 ∧
  C = 60 ∧
  b = 4.3825 ∧
  c = 6.1355 :=
sorry -- Proof goes here

end triangle_solutions_l133_133597


namespace taxi_service_charge_l133_133592

theorem taxi_service_charge (initial_fee : ℝ) (additional_charge : ℝ) (increment : ℝ) (total_charge : ℝ) 
  (h_initial_fee : initial_fee = 2.25) 
  (h_additional_charge : additional_charge = 0.4) 
  (h_increment : increment = 2 / 5) 
  (h_total_charge : total_charge = 5.85) : 
  ∃ distance : ℝ, distance = 3.6 :=
by
  sorry

end taxi_service_charge_l133_133592


namespace sally_initial_cards_l133_133263

theorem sally_initial_cards (X : ℕ) (h1 : X + 41 + 20 = 88) : X = 27 :=
by
  -- Proof goes here
  sorry

end sally_initial_cards_l133_133263


namespace volume_of_region_l133_133675

-- Define the conditions
def condition1 (x y z : ℝ) := abs (x + y + 2 * z) + abs (x + y - 2 * z) ≤ 12
def condition2 (x : ℝ) := x ≥ 0
def condition3 (y : ℝ) := y ≥ 0
def condition4 (z : ℝ) := z ≥ 0

-- Define the volume function
def volume (x y z : ℝ) := 18 * 3

-- Proof statement
theorem volume_of_region : ∀ (x y z : ℝ),
  condition1 x y z →
  condition2 x →
  condition3 y →
  condition4 z →
  volume x y z = 54 := by
  sorry

end volume_of_region_l133_133675


namespace sphere_radius_proportional_l133_133353

theorem sphere_radius_proportional
  (k : ℝ)
  (r1 r2 : ℝ)
  (W1 W2 : ℝ)
  (h_weight_area : ∀ (r : ℝ), W1 = k * (4 * π * r^2))
  (h_given1: W2 = 32)
  (h_given2: r2 = 0.3)
  (h_given3: W1 = 8):
  r1 = 0.15 := 
by
  sorry

end sphere_radius_proportional_l133_133353


namespace giuseppe_can_cut_rectangles_l133_133045

theorem giuseppe_can_cut_rectangles : 
  let board_length := 22
  let board_width := 15
  let rectangle_length := 3
  let rectangle_width := 5
  (board_length * board_width) / (rectangle_length * rectangle_width) = 22 :=
by
  sorry

end giuseppe_can_cut_rectangles_l133_133045


namespace lines_perpendicular_to_same_plane_are_parallel_l133_133648

theorem lines_perpendicular_to_same_plane_are_parallel 
  (parallel_proj_parallel_lines : Prop)
  (planes_parallel_to_same_line : Prop)
  (planes_perpendicular_to_same_plane : Prop)
  (lines_perpendicular_to_same_plane : Prop) 
  (h1 : ¬ parallel_proj_parallel_lines)
  (h2 : ¬ planes_parallel_to_same_line)
  (h3 : ¬ planes_perpendicular_to_same_plane) :
  lines_perpendicular_to_same_plane := 
sorry

end lines_perpendicular_to_same_plane_are_parallel_l133_133648


namespace rational_solution_quadratic_l133_133028

theorem rational_solution_quadratic (m : ℕ) (h_pos : m > 0) : 
  (∃ (x : ℚ), x * x * m + 25 * x + m = 0) ↔ m = 10 ∨ m = 12 :=
by sorry

end rational_solution_quadratic_l133_133028


namespace number_of_fours_is_even_l133_133620

theorem number_of_fours_is_even (n3 n4 n5 : ℕ) 
  (h1 : n3 + n4 + n5 = 80)
  (h2 : 3 * n3 + 4 * n4 + 5 * n5 = 276) : Even n4 := 
sorry

end number_of_fours_is_even_l133_133620


namespace dominic_domino_problem_l133_133903

theorem dominic_domino_problem 
  (num_dominoes : ℕ)
  (pips_pairs : ℕ → ℕ)
  (hexagonal_ring : ℕ → ℕ → Prop) : 
  ∀ (adj : ℕ → ℕ → Prop), 
  num_dominoes = 6 → 
  (∀ i j, hexagonal_ring i j → pips_pairs i = pips_pairs j) →
  ∃ k, k = 2 :=
by {
  sorry
}

end dominic_domino_problem_l133_133903


namespace sum_of_squares_five_consecutive_ints_not_perfect_square_l133_133944

theorem sum_of_squares_five_consecutive_ints_not_perfect_square (n : ℤ) :
  ∀ k : ℤ, k^2 ≠ 5 * (n^2 + 2) := 
sorry

end sum_of_squares_five_consecutive_ints_not_perfect_square_l133_133944


namespace range_of_m_l133_133836

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x > 0 → 9^x - m * 3^x + m + 1 > 0) → m < 2 + 2 * Real.sqrt 2 :=
sorry

end range_of_m_l133_133836


namespace max_d_minus_r_proof_l133_133059

noncomputable def max_d_minus_r : ℕ := 35

theorem max_d_minus_r_proof (d r : ℕ) (h1 : 2017 % d = r) (h2 : 1029 % d = r) (h3 : 725 % d = r) :
  d - r ≤ max_d_minus_r :=
  sorry

end max_d_minus_r_proof_l133_133059


namespace gcd_sub_12_eq_36_l133_133982

theorem gcd_sub_12_eq_36 :
  Nat.gcd 7344 48 - 12 = 36 := 
by 
  sorry

end gcd_sub_12_eq_36_l133_133982


namespace problem_statement_l133_133058

theorem problem_statement (a b : ℕ) (m n : ℕ)
  (h1 : 32 + (2 / 7 : ℝ) = 3 * (2 / 7 : ℝ))
  (h2 : 33 + (3 / 26 : ℝ) = 3 * (3 / 26 : ℝ))
  (h3 : 34 + (4 / 63 : ℝ) = 3 * (4 / 63 : ℝ))
  (h4 : 32014 + (m / n : ℝ) = 2014 * 3 * (m / n : ℝ))
  (h5 : 32016 + (a / b : ℝ) = 2016 * 3 * (a / b : ℝ)) :
  (b + 1) / (a * a) = 2016 :=
sorry

end problem_statement_l133_133058


namespace sum_of_abc_is_33_l133_133107

theorem sum_of_abc_is_33 (a b c N : ℕ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hN1 : N = 5 * a + 3 * b + 5 * c)
    (hN2 : N = 4 * a + 5 * b + 4 * c) (hN_range : 131 < N ∧ N < 150) :
  a + b + c = 33 := 
sorry

end sum_of_abc_is_33_l133_133107


namespace percentage_more_research_l133_133566

-- Defining the various times spent
def acclimation_period : ℝ := 1
def learning_basics_period : ℝ := 2
def dissertation_fraction : ℝ := 0.5
def total_time : ℝ := 7

-- Defining the time spent on each activity
def dissertation_period := dissertation_fraction * acclimation_period
def research_period := total_time - acclimation_period - learning_basics_period - dissertation_period

-- The main theorem to prove
theorem percentage_more_research : 
  ((research_period - learning_basics_period) / learning_basics_period) * 100 = 75 :=
by
  -- Placeholder for the proof
  sorry

end percentage_more_research_l133_133566


namespace regular_polygon_properties_l133_133015

theorem regular_polygon_properties
  (exterior_angle : ℝ := 18) :
  (∃ (n : ℕ), n = 20) ∧ (∃ (interior_angle : ℝ), interior_angle = 162) := 
by
  sorry

end regular_polygon_properties_l133_133015


namespace det_of_commuting_matrices_l133_133097

theorem det_of_commuting_matrices (n : ℕ) (hn : n ≥ 2) (A B : Matrix (Fin n) (Fin n) ℝ)
  (hA : A * A = -1) (hAB : A * B = B * A) : 
  0 ≤ B.det := 
sorry

end det_of_commuting_matrices_l133_133097


namespace domain_of_h_l133_133940

noncomputable def h (x : ℝ) : ℝ :=
  (x^2 - 9) / (abs (x - 4) + x^2 - 1)

theorem domain_of_h :
  ∀ (x : ℝ), x ≠ (1 + Real.sqrt 13) / 2 → (abs (x - 4) + x^2 - 1) ≠ 0 :=
sorry

end domain_of_h_l133_133940


namespace determine_m_with_opposite_roots_l133_133376

theorem determine_m_with_opposite_roots (c d k : ℝ) (h : c + d ≠ 0):
  (∃ m : ℝ, ∀ x : ℝ, (x^2 - d * x) / (c * x - k) = (m - 2) / (m + 2) ∧ 
            (x = -y ∧ y = -x)) ↔ m = 2 * (c - d) / (c + d) :=
sorry

end determine_m_with_opposite_roots_l133_133376


namespace monkeys_bananas_minimum_l133_133565

theorem monkeys_bananas_minimum (b1 b2 b3 : ℕ) (x y z : ℕ) : 
  (x = 2 * y) ∧ (z = (2 * y) / 3) ∧ 
  (x = (2 * b1) / 3 + (b2 / 3) + (5 * b3) / 12) ∧ 
  (y = (b1 / 6) + (b2 / 3) + (5 * b3) / 12) ∧ 
  (z = (b1 / 6) + (b2 / 3) + (b3 / 6)) →
  b1 = 324 ∧ b2 = 162 ∧ b3 = 72 ∧ (b1 + b2 + b3 = 558) :=
sorry

end monkeys_bananas_minimum_l133_133565


namespace sufficient_condition_for_inequality_l133_133852

theorem sufficient_condition_for_inequality (a : ℝ) (h : 0 < a ∧ a < 4) :
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0 :=
by
  sorry

end sufficient_condition_for_inequality_l133_133852


namespace sum_of_integers_is_24_l133_133212

theorem sum_of_integers_is_24 (x y : ℕ) (hx : x > y) (h1 : x - y = 4) (h2 : x * y = 132) : x + y = 24 :=
by
  sorry

end sum_of_integers_is_24_l133_133212


namespace rhombus_diagonal_l133_133117

theorem rhombus_diagonal (d1 d2 area : ℝ) (h1 : d1 = 20) (h2 : area = 160) (h3 : area = (d1 * d2) / 2) :
  d2 = 16 :=
by
  rw [h1, h2] at h3
  linarith

end rhombus_diagonal_l133_133117


namespace miguel_run_time_before_ariana_catches_up_l133_133789

theorem miguel_run_time_before_ariana_catches_up
  (head_start : ℕ := 20)
  (ariana_speed : ℕ := 6)
  (miguel_speed : ℕ := 4)
  (head_start_distance : ℕ := miguel_speed * head_start)
  (t_catchup : ℕ := (head_start_distance) / (ariana_speed - miguel_speed))
  (total_time : ℕ := t_catchup + head_start) :
  total_time = 60 := sorry

end miguel_run_time_before_ariana_catches_up_l133_133789


namespace triangle_angle_sum_l133_133296

theorem triangle_angle_sum {A B C : Type} 
  (angle_ABC : ℝ) (angle_BAC : ℝ) (angle_BCA : ℝ) (x : ℝ) 
  (h1: angle_ABC = 90) 
  (h2: angle_BAC = 3 * x) 
  (h3: angle_BCA = x + 10)
  : x = 20 :=
by
  sorry

end triangle_angle_sum_l133_133296


namespace fly_travel_distance_l133_133286

theorem fly_travel_distance
  (carA_speed : ℕ)
  (carB_speed : ℕ)
  (initial_distance : ℕ)
  (fly_speed : ℕ)
  (relative_speed : ℕ := carB_speed - carA_speed)
  (catchup_time : ℚ := initial_distance / relative_speed)
  (fly_travel : ℚ := fly_speed * catchup_time) :
  carA_speed = 20 → carB_speed = 30 → initial_distance = 1 → fly_speed = 40 → fly_travel = 4 :=
by
  sorry

end fly_travel_distance_l133_133286


namespace arithmetic_geometric_sum_l133_133605

noncomputable def a_n (n : ℕ) := 3 * n - 2
noncomputable def b_n (n : ℕ) := 4 ^ (n - 1)

theorem arithmetic_geometric_sum (n : ℕ) :
    a_n 1 = 1 ∧ a_n 2 = b_n 2 ∧ a_n 6 = b_n 3 ∧ S_n = 1 + (n - 1) * 4 ^ n :=
by sorry

end arithmetic_geometric_sum_l133_133605


namespace average_production_l133_133478

theorem average_production (n : ℕ) :
  let total_past_production := 50 * n
  let total_production_including_today := 100 + total_past_production
  let average_production := total_production_including_today / (n + 1)
  average_production = 55
  -> n = 9 :=
by
  sorry

end average_production_l133_133478


namespace distance_from_A_to_C_correct_total_distance_traveled_correct_l133_133048

-- Define the conditions
def distance_to_A : ℕ := 30
def distance_to_B : ℕ := 20
def distance_to_C : ℤ := -15
def times_to_C : ℕ := 3

-- Define the resulting calculated distances based on the conditions
def distance_A_to_C : ℕ := distance_to_A + distance_to_C.natAbs
def total_distance_traveled : ℕ := (distance_to_A + distance_to_B) * 2 + distance_to_C.natAbs * (times_to_C * 2)

-- The proof problems (statements) based on the problem's questions
theorem distance_from_A_to_C_correct : distance_A_to_C = 45 := by
  sorry

theorem total_distance_traveled_correct : total_distance_traveled = 190 := by
  sorry

end distance_from_A_to_C_correct_total_distance_traveled_correct_l133_133048


namespace bookseller_fiction_books_count_l133_133904

theorem bookseller_fiction_books_count (n : ℕ) (h1 : n.factorial * 6 = 36) : n = 3 :=
sorry

end bookseller_fiction_books_count_l133_133904


namespace math_expression_equivalent_l133_133151

theorem math_expression_equivalent :
  ((π - 1)^0 + 4 * (Real.sin (Real.pi / 4)) - Real.sqrt 8 + abs (-3) = 4) :=
by
  sorry

end math_expression_equivalent_l133_133151


namespace polyhedron_edges_l133_133580

theorem polyhedron_edges (F V E : ℕ) (h1 : F = 12) (h2 : V = 20) (h3 : F + V = E + 2) : E = 30 :=
by
  -- Additional details would go here, proof omitted as instructed.
  sorry

end polyhedron_edges_l133_133580


namespace find_amplitude_l133_133232

noncomputable def amplitude_of_cosine (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  a

theorem find_amplitude (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_max : amplitude_of_cosine a b h_a h_b = 3) :
  a = 3 :=
sorry

end find_amplitude_l133_133232


namespace nate_walks_past_per_minute_l133_133447

-- Define the conditions as constants
def rows_G := 15
def cars_per_row_G := 10
def rows_H := 20
def cars_per_row_H := 9
def total_minutes := 30

-- Define the problem statement
theorem nate_walks_past_per_minute :
  ((rows_G * cars_per_row_G) + (rows_H * cars_per_row_H)) / total_minutes = 11 := 
sorry

end nate_walks_past_per_minute_l133_133447


namespace find_sum_l133_133010

def f (x : ℝ) : ℝ := sorry

axiom f_non_decreasing : ∀ {x1 x2 : ℝ}, 0 ≤ x1 → x1 ≤ 1 → 0 ≤ x2 → x2 ≤ 1 → x1 < x2 → f x1 ≤ f x2
axiom f_at_0 : f 0 = 0
axiom f_scaling : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → f (x / 3) = (1 / 2) * f x
axiom f_symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → f (1 - x) = 1 - f x

theorem find_sum :
  f (1 / 3) + f (2 / 3) + f (1 / 9) + f (1 / 6) + f (1 / 8) = 7 / 4 :=
by
  sorry

end find_sum_l133_133010


namespace problem1_problem2_l133_133343

-- Problem 1
theorem problem1 : ((2 / 3 - 1 / 12 - 1 / 15) * -60) = -31 := by
  sorry

-- Problem 2
theorem problem2 : ((-7 / 8) / ((7 / 4) - 7 / 8 - 7 / 12)) = -3 := by
  sorry

end problem1_problem2_l133_133343


namespace car_cleaning_ratio_l133_133783

theorem car_cleaning_ratio
    (outside_cleaning_time : ℕ)
    (total_cleaning_time : ℕ)
    (h1 : outside_cleaning_time = 80)
    (h2 : total_cleaning_time = 100) :
    (total_cleaning_time - outside_cleaning_time) / outside_cleaning_time = 1 / 4 :=
by
  sorry

end car_cleaning_ratio_l133_133783


namespace largest_digit_divisible_by_6_l133_133908

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem largest_digit_divisible_by_6 (N : ℕ) (hN : N ≤ 9) :
  (∃ m : ℕ, 56780 + N = m * 6) ∧ is_even N ∧ is_divisible_by_3 (26 + N) → N = 4 := by
  sorry

end largest_digit_divisible_by_6_l133_133908


namespace total_oranges_picked_l133_133716

theorem total_oranges_picked (mary_oranges : Nat) (jason_oranges : Nat) (hmary : mary_oranges = 122) (hjason : jason_oranges = 105) : mary_oranges + jason_oranges = 227 := by
  sorry

end total_oranges_picked_l133_133716


namespace sum_of_digits_is_32_l133_133976

/-- 
Prove that the sum of digits \( A, B, C, D, E \) is 32 given the constraints
1. \( A, B, C, D, E \) are single digits.
2. The sum of the units column 3E results in 1 (units place of 2011).
3. The sum of the hundreds column 3A and carry equals 20 (hundreds place of 2011).
-/
theorem sum_of_digits_is_32
  (A B C D E : ℕ)
  (h1 : A < 10)
  (h2 : B < 10)
  (h3 : C < 10)
  (h4 : D < 10)
  (h5 : E < 10)
  (units_condition : 3 * E % 10 = 1)
  (hundreds_condition : ∃ carry: ℕ, carry < 10 ∧ 3 * A + carry = 20) :
  A + B + C + D + E = 32 := 
sorry

end sum_of_digits_is_32_l133_133976


namespace area_of_region_bounded_by_circle_l133_133062

theorem area_of_region_bounded_by_circle :
  (∃ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 9 = 0) →
  ∃ (area : ℝ), area = 4 * Real.pi :=
by
  sorry

end area_of_region_bounded_by_circle_l133_133062


namespace tom_coins_worth_l133_133167

-- Definitions based on conditions:
def total_coins : ℕ := 30
def value_difference_cents : ℕ := 90
def nickel_value_cents : ℕ := 5
def dime_value_cents : ℕ := 10

-- Main theorem statement:
theorem tom_coins_worth (n d : ℕ) (h1 : d = total_coins - n) 
    (h2 : (nickel_value_cents * n + dime_value_cents * d) - (dime_value_cents * n + nickel_value_cents * d) = value_difference_cents) : 
    (nickel_value_cents * n + dime_value_cents * d) = 180 :=
by
  sorry -- Proof omitted.

end tom_coins_worth_l133_133167


namespace combined_cost_l133_133124

variable (bench_cost : ℝ) (table_cost : ℝ)

-- Conditions
axiom bench_cost_def : bench_cost = 250.0
axiom table_cost_def : table_cost = 2 * bench_cost

-- Goal
theorem combined_cost (bench_cost : ℝ) (table_cost : ℝ) 
  (h1 : bench_cost = 250.0) (h2 : table_cost = 2 * bench_cost) : 
  table_cost + bench_cost = 750.0 :=
by
  sorry

end combined_cost_l133_133124


namespace jessica_threw_away_4_roses_l133_133241

def roses_thrown_away (a b c d : ℕ) : Prop :=
  (a + b) - d = c

theorem jessica_threw_away_4_roses :
  roses_thrown_away 2 25 23 4 :=
by
  -- This is where the proof would go
  sorry

end jessica_threw_away_4_roses_l133_133241


namespace max_product_distance_l133_133994

-- Definitions for the conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1
def is_focus (F : ℝ × ℝ) : Prop := F = (3, 0) ∨ F = (-3, 0)

-- The theorem statement
theorem max_product_distance (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) 
  (h1 : ellipse M.1 M.2) 
  (h2 : is_focus F1) 
  (h3 : is_focus F2) : 
  (∃ x y, M = (x, y) ∧ ellipse x y) → 
  |(M.1 - F1.1)^2 + (M.2 - F1.2)^2| * |(M.1 - F2.1)^2 + (M.2 - F2.2)^2| ≤ 81 := 
sorry

end max_product_distance_l133_133994


namespace purely_imaginary_complex_number_l133_133741

theorem purely_imaginary_complex_number (m : ℝ) :
  (m^2 - 2 * m - 3 = 0) ∧ (m^2 - 4 * m + 3 ≠ 0) → m = -1 :=
by
  sorry

end purely_imaginary_complex_number_l133_133741


namespace range_of_m_l133_133973

theorem range_of_m (m : ℝ) (h : ∃ x : ℝ, x < 0 ∧ mx^2 + 2*x + 1 = 0) : m ∈ Set.Iic 1 :=
sorry

end range_of_m_l133_133973


namespace jon_monthly_earnings_l133_133714

def earnings_per_person : ℝ := 0.10
def visits_per_hour : ℕ := 50
def hours_per_day : ℕ := 24
def days_per_month : ℕ := 30

theorem jon_monthly_earnings : 
  (earnings_per_person * visits_per_hour * hours_per_day * days_per_month) = 3600 :=
by
  sorry

end jon_monthly_earnings_l133_133714


namespace spinner_probabilities_l133_133754

theorem spinner_probabilities (pA pB pC pD : ℚ) (h1 : pA = 1/4) (h2 : pB = 1/3) (h3 : pA + pB + pC + pD = 1) :
  pC + pD = 5/12 :=
by
  -- Here you would construct the proof (left as sorry for this example)
  sorry

end spinner_probabilities_l133_133754


namespace cake_cost_is_20_l133_133550

-- Define the given conditions
def total_budget : ℕ := 50
def additional_needed : ℕ := 11
def bouquet_cost : ℕ := 36
def balloons_cost : ℕ := 5

-- Define the derived conditions
def total_cost : ℕ := total_budget + additional_needed
def combined_bouquet_balloons_cost : ℕ := bouquet_cost + balloons_cost
def cake_cost : ℕ := total_cost - combined_bouquet_balloons_cost

-- The theorem to be proved
theorem cake_cost_is_20 : cake_cost = 20 :=
by
  -- proof steps are not required
  sorry

end cake_cost_is_20_l133_133550


namespace coprime_divides_product_l133_133815

theorem coprime_divides_product {a b n : ℕ} (h1 : Nat.gcd a b = 1) (h2 : a ∣ n) (h3 : b ∣ n) : ab ∣ n :=
by
  sorry

end coprime_divides_product_l133_133815


namespace x_value_l133_133772

theorem x_value (x : ℤ) (h : x = (2009^2 - 2009) / 2009) : x = 2008 := by
  sorry

end x_value_l133_133772


namespace total_number_of_pipes_l133_133590

theorem total_number_of_pipes (bottom_layer top_layer layers : ℕ) 
  (h_bottom_layer : bottom_layer = 13) 
  (h_top_layer : top_layer = 3) 
  (h_layers : layers = 11) : 
  bottom_layer + top_layer = 16 → 
  (bottom_layer + top_layer) * layers / 2 = 88 := 
by
  intro h_sum
  sorry

end total_number_of_pipes_l133_133590


namespace boys_in_other_communities_l133_133534

theorem boys_in_other_communities (total_boys : ℕ) (muslim_percent hindu_percent sikh_percent : ℕ)
  (H_total : total_boys = 400)
  (H_muslim : muslim_percent = 44)
  (H_hindu : hindu_percent = 28)
  (H_sikh : sikh_percent = 10) :
  total_boys * (1 - (muslim_percent + hindu_percent + sikh_percent) / 100) = 72 :=
by
  sorry

end boys_in_other_communities_l133_133534


namespace shortest_path_length_l133_133076

theorem shortest_path_length (x y z : ℕ) (h1 : x + y = z + 1) (h2 : x + z = y + 5) (h3 : y + z = x + 7) : 
  min (min x y) z = 3 :=
by sorry

end shortest_path_length_l133_133076


namespace circle_circumference_difference_l133_133993

theorem circle_circumference_difference (d_inner : ℝ) (h_inner : d_inner = 100) 
  (d_outer : ℝ) (h_outer : d_outer = d_inner + 30) :
  ((π * d_outer) - (π * d_inner)) = 30 * π :=
by 
  sorry

end circle_circumference_difference_l133_133993


namespace total_amount_shared_l133_133506

theorem total_amount_shared (x y z : ℝ) (h1 : x = 1.25 * y) (h2 : y = 1.20 * z) (hz : z = 100) :
  x + y + z = 370 := by
  sorry

end total_amount_shared_l133_133506


namespace simplify_fraction_l133_133282

theorem simplify_fraction (b : ℕ) (hb : b = 5) : (15 * b^4) / (90 * b^3 * b) = 1 / 6 := by
  sorry

end simplify_fraction_l133_133282


namespace num_combinations_two_dresses_l133_133613

def num_colors : ℕ := 4
def num_patterns : ℕ := 5

def combinations_first_dress : ℕ := num_colors * num_patterns
def combinations_second_dress : ℕ := (num_colors - 1) * (num_patterns - 1)

theorem num_combinations_two_dresses :
  (combinations_first_dress * combinations_second_dress) = 240 := by
  sorry

end num_combinations_two_dresses_l133_133613


namespace smallest_digit_divisible_by_9_l133_133206

theorem smallest_digit_divisible_by_9 :
  ∃ (d : ℕ), (25 + d) % 9 = 0 ∧ (∀ e : ℕ, (25 + e) % 9 = 0 → e ≥ d) :=
by
  sorry

end smallest_digit_divisible_by_9_l133_133206


namespace correct_sample_size_l133_133024

-- Definitions based on conditions:
def population_size : ℕ := 1800
def sample_size : ℕ := 1000
def surveyed_parents : ℕ := 1000

-- The proof statement we need: 
-- Prove that the sample size is 1000, given the surveyed parents are 1000
theorem correct_sample_size (ps : ℕ) (sp : ℕ) (ss : ℕ) (h1 : ps = population_size) (h2 : sp = surveyed_parents) : ss = sample_size :=
  sorry

end correct_sample_size_l133_133024


namespace penelope_min_games_l133_133424

theorem penelope_min_games (m w l: ℕ) (h1: 25 * w - 13 * l = 2007) (h2: m = w + l) : m = 87 := by
  sorry

end penelope_min_games_l133_133424


namespace total_pumpkins_l133_133581

-- Define the number of pumpkins grown by Sandy and Mike
def pumpkinsSandy : ℕ := 51
def pumpkinsMike : ℕ := 23

-- Prove that their total is 74
theorem total_pumpkins : pumpkinsSandy + pumpkinsMike = 74 := by
  sorry

end total_pumpkins_l133_133581


namespace find_n_l133_133587

theorem find_n (a b : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h1 : ∃ n : ℕ, n - 76 = a^3) (h2 : ∃ n : ℕ, n + 76 = b^3) : ∃ n : ℕ, n = 140 :=
by 
  sorry

end find_n_l133_133587


namespace combined_area_is_256_l133_133980

-- Define the conditions
def side_length : ℝ := 16
def area_square : ℝ := side_length ^ 2

-- Define the property of the sides r and s
def r_s_property (r s : ℝ) : Prop :=
  (r + s)^2 + (r - s)^2 = side_length^2

-- The combined area of the four triangles
def combined_area_of_triangles (r s : ℝ) : ℝ :=
  2 * (r ^ 2 + s ^ 2)

-- Prove the final statement
theorem combined_area_is_256 (r s : ℝ) (h : r_s_property r s) :
  combined_area_of_triangles r s = 256 := by
  sorry

end combined_area_is_256_l133_133980


namespace inequality_three_var_l133_133955

theorem inequality_three_var
  (a b c : ℝ)
  (ha : 0 ≤ a)
  (hb : 0 ≤ b)
  (hc : 0 ≤ c) :
  2 * (a^3 + b^3 + c^3) ≥ a^2 * b + a * b^2 + a^2 * c + a * c^2 + b^2 * c + b * c^2 :=
by sorry

end inequality_three_var_l133_133955


namespace find_a_2016_l133_133132

-- Define the sequence a_n and its sum S_n
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
axiom S_n_eq : ∀ n : ℕ, S n + (1 + (2 / n)) * a n = 4
axiom a_1_eq : a 1 = 1
axiom a_rec : ∀ n : ℕ, n ≥ 2 → a n = (n / (2 * (n - 1))) * a (n - 1)

-- The theorem to prove
theorem find_a_2016 : a 2016 = 2016 / 2^2015 := by
  sorry

end find_a_2016_l133_133132


namespace maximum_area_of_inscribed_rectangle_l133_133287

theorem maximum_area_of_inscribed_rectangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (A : ℝ), A = (a * b) / 4 :=
by
  sorry -- placeholder for the proof

end maximum_area_of_inscribed_rectangle_l133_133287


namespace eggs_in_seven_boxes_l133_133576

-- define the conditions
def eggs_per_box : Nat := 15
def number_of_boxes : Nat := 7

-- state the main theorem to prove
theorem eggs_in_seven_boxes : eggs_per_box * number_of_boxes = 105 := by
  sorry

end eggs_in_seven_boxes_l133_133576


namespace perimeter_of_smaller_rectangle_l133_133667

theorem perimeter_of_smaller_rectangle :
  ∀ (L W n : ℕ), 
  L = 16 → W = 20 → n = 10 →
  (∃ (x y : ℕ), L % 2 = 0 ∧ W % 5 = 0 ∧ 2 * y = L ∧ 5 * x = W ∧ (L * W) / n = x * y ∧ 2 * (x + y) = 24) :=
by
  intros L W n H1 H2 H3
  use 4, 8
  sorry

end perimeter_of_smaller_rectangle_l133_133667


namespace smallest_number_of_seats_required_l133_133262

theorem smallest_number_of_seats_required (total_chairs : ℕ) (condition : ∀ (N : ℕ), ∀ (seating : Finset ℕ),
  seating.card = N → (∀ x ∈ seating, (x + 1) % total_chairs ∈ seating ∨ (x + total_chairs - 1) % total_chairs ∈ seating)) :
  total_chairs = 100 → ∃ N : ℕ, N = 20 :=
by
  intros
  sorry

end smallest_number_of_seats_required_l133_133262


namespace fraction_addition_l133_133297

theorem fraction_addition (d : ℤ) :
  (6 + 4 * d) / 9 + 3 / 2 = (39 + 8 * d) / 18 := sorry

end fraction_addition_l133_133297


namespace time_to_carry_backpack_l133_133049

/-- 
Given:
1. Lara takes 73 seconds to crank open the door to the obstacle course.
2. Lara traverses the obstacle course the second time in 5 minutes and 58 seconds.
3. The total time to complete the obstacle course is 874 seconds.

Prove:
The time it took Lara to carry the backpack through the obstacle course the first time is 443 seconds.
-/
theorem time_to_carry_backpack (door_time : ℕ) (second_traversal_time : ℕ) (total_time : ℕ) : 
  (door_time + second_traversal_time + 443 = total_time) :=
by
  -- Given conditions
  let door_time := 73
  let second_traversal_time := 5 * 60 + 58 -- Convert 5 minutes 58 seconds to seconds
  let total_time := 874
  -- Calculate the time to carry the backpack
  sorry

end time_to_carry_backpack_l133_133049


namespace sqrt_four_ninths_l133_133695

theorem sqrt_four_ninths : 
  (∀ (x : ℚ), x * x = 4 / 9 → (x = 2 / 3 ∨ x = - (2 / 3))) :=
by sorry

end sqrt_four_ninths_l133_133695


namespace photo_arrangement_l133_133894

noncomputable def valid_arrangements (teacher boys girls : ℕ) : ℕ :=
  if girls = 2 ∧ teacher = 1 ∧ boys = 2 then 24 else 0

theorem photo_arrangement :
  valid_arrangements 1 2 2 = 24 :=
by {
  -- The proof goes here.
  sorry
}

end photo_arrangement_l133_133894


namespace apples_in_each_box_l133_133298

theorem apples_in_each_box (x : ℕ) :
  (5 * x - (60 * 5)) = (2 * x) -> x = 100 :=
by
  sorry

end apples_in_each_box_l133_133298


namespace number_of_x_for_P_eq_zero_l133_133569

noncomputable def P (x : ℝ) : ℂ :=
  1 + Complex.exp (Complex.I * x) - Complex.exp (2 * Complex.I * x) + Complex.exp (3 * Complex.I * x) - Complex.exp (4 * Complex.I * x)

theorem number_of_x_for_P_eq_zero : 
  ∃ (n : ℕ), n = 4 ∧ ∃ (xs : Fin n → ℝ), (∀ i, 0 ≤ xs i ∧ xs i < 2 * Real.pi ∧ P (xs i) = 0) ∧ Function.Injective xs := 
sorry

end number_of_x_for_P_eq_zero_l133_133569


namespace weight_of_bowling_ball_l133_133204

variable (b c : ℝ)

axiom h1 : 5 * b = 2 * c
axiom h2 : 3 * c = 84

theorem weight_of_bowling_ball : b = 11.2 :=
by
  sorry

end weight_of_bowling_ball_l133_133204


namespace smallest_number_l133_133727

theorem smallest_number (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : (a + b + c) = 90) (h4 : b = 28) (h5 : b = c - 6) : a = 28 :=
by 
  sorry

end smallest_number_l133_133727


namespace last_three_digits_of_7_pow_103_l133_133671

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 327 :=
by
  sorry

end last_three_digits_of_7_pow_103_l133_133671


namespace profit_percentage_is_33_point_33_l133_133439

variable (C S : ℝ)

-- Initial condition based on the problem statement
axiom cost_eq_sell : 20 * C = 15 * S

-- Statement to prove
theorem profit_percentage_is_33_point_33 (h : 20 * C = 15 * S) : (S - C) / C * 100 = 33.33 := 
sorry

end profit_percentage_is_33_point_33_l133_133439


namespace weight_of_each_bar_l133_133879

theorem weight_of_each_bar 
  (num_bars : ℕ) 
  (cost_per_pound : ℝ) 
  (total_cost : ℝ) 
  (total_weight : ℝ) 
  (weight_per_bar : ℝ)
  (h1 : num_bars = 20)
  (h2 : cost_per_pound = 0.5)
  (h3 : total_cost = 15)
  (h4 : total_weight = total_cost / cost_per_pound)
  (h5 : weight_per_bar = total_weight / num_bars)
  : weight_per_bar = 1.5 := 
by
  sorry

end weight_of_each_bar_l133_133879


namespace transform_parabola_l133_133253

theorem transform_parabola (a b c : ℝ) (h : a ≠ 0) :
  ∃ (f : ℝ → ℝ), (∀ x, f (a * x^2 + b * x + c) = x^2) :=
sorry

end transform_parabola_l133_133253


namespace largest_prime_divisor_25_sq_plus_72_sq_l133_133141

theorem largest_prime_divisor_25_sq_plus_72_sq : ∃ p : ℕ, Nat.Prime p ∧ p ∣ (25^2 + 72^2) ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (25^2 + 72^2) → q ≤ p :=
sorry

end largest_prime_divisor_25_sq_plus_72_sq_l133_133141


namespace jordan_purchase_total_rounded_l133_133119

theorem jordan_purchase_total_rounded :
  let p1 := 2.49
  let p2 := 6.51
  let p3 := 11.49
  let r1 := 2 -- rounded value of p1
  let r2 := 7 -- rounded value of p2
  let r3 := 11 -- rounded value of p3
  r1 + r2 + r3 = 20 :=
by
  let p1 := 2.49
  let p2 := 6.51
  let p3 := 11.49
  let r1 := 2
  let r2 := 7
  let r3 := 11
  show r1 + r2 + r3 = 20
  sorry

end jordan_purchase_total_rounded_l133_133119


namespace trig_identity_eq_one_l133_133175

theorem trig_identity_eq_one :
  (Real.sin (160 * Real.pi / 180) + Real.sin (40 * Real.pi / 180)) *
  (Real.sin (140 * Real.pi / 180) + Real.sin (20 * Real.pi / 180)) +
  (Real.sin (50 * Real.pi / 180) - Real.sin (70 * Real.pi / 180)) *
  (Real.sin (130 * Real.pi / 180) - Real.sin (110 * Real.pi / 180)) =
  1 :=
sorry

end trig_identity_eq_one_l133_133175


namespace largest_decimal_of_four_digit_binary_l133_133986

theorem largest_decimal_of_four_digit_binary : ∀ n : ℕ, (n < 16) → n ≤ 15 :=
by {
  -- conditions: a four-digit binary number implies \( n \) must be less than \( 2^4 = 16 \)
  sorry
}

end largest_decimal_of_four_digit_binary_l133_133986


namespace net_loss_is_1_percent_l133_133552

noncomputable def net_loss_percent (CP SP1 SP2 SP3 SP4 : ℝ) : ℝ :=
  let TCP := 4 * CP
  let TSP := SP1 + SP2 + SP3 + SP4
  ((TCP - TSP) / TCP) * 100

theorem net_loss_is_1_percent
  (CP : ℝ)
  (HCP : CP = 1000)
  (SP1 : ℝ)
  (HSP1 : SP1 = CP * 1.1 * 0.95)
  (SP2 : ℝ)
  (HSP2 : SP2 = (CP * 0.9) * 1.02)
  (SP3 : ℝ)
  (HSP3 : SP3 = (CP * 1.2) * 1.03)
  (SP4 : ℝ)
  (HSP4 : SP4 = (CP * 0.75) * 1.01) :
  abs (net_loss_percent CP SP1 SP2 SP3 SP4 + 1.09) < 0.01 :=
by
  -- Proof omitted
  sorry

end net_loss_is_1_percent_l133_133552


namespace inequality_negatives_l133_133723

theorem inequality_negatives (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) : a^2 > b^2 :=
sorry

end inequality_negatives_l133_133723


namespace deaths_during_operation_l133_133636

noncomputable def initial_count : ℕ := 1000
noncomputable def first_day_remaining (n : ℕ) := 5 * n / 6
noncomputable def second_day_remaining (n : ℕ) := (35 * n / 48) - 1
noncomputable def third_day_remaining (n : ℕ) := (105 * n / 192) - 3 / 4

theorem deaths_during_operation : ∃ n : ℕ, initial_count - n = 472 ∧ n = 528 :=
  by sorry

end deaths_during_operation_l133_133636


namespace complex_is_purely_imaginary_iff_a_eq_2_l133_133870

theorem complex_is_purely_imaginary_iff_a_eq_2 (a : ℝ) :
  (a = 2) ↔ ((a^2 - 4 = 0) ∧ (a + 2 ≠ 0)) :=
by sorry

end complex_is_purely_imaginary_iff_a_eq_2_l133_133870


namespace decreasing_on_interval_l133_133831

variable {x m n : ℝ}

def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := |x^2 - 2 * m * x + n|

theorem decreasing_on_interval
  (h : ∀ x, f x m n = |x^2 - 2 * m * x + n|)
  (h_cond : m^2 - n ≤ 0) :
  ∀ x y, x ≤ y → y ≤ m → f y m n ≤ f x m n :=
sorry

end decreasing_on_interval_l133_133831


namespace Bryce_received_raisins_l133_133821

theorem Bryce_received_raisins :
  ∃ x : ℕ, (∀ y : ℕ, x = y + 6) ∧ (∀ z : ℕ, z = x / 2) → x = 12 :=
by
  sorry

end Bryce_received_raisins_l133_133821


namespace part_a_part_b_part_c_part_d_l133_133709

-- define the partitions function
def P (k l n : ℕ) : ℕ := sorry

-- Part (a) statement
theorem part_a (k l n : ℕ) :
  P k l n - P k (l - 1) n = P (k - 1) l (n - l) :=
sorry

-- Part (b) statement
theorem part_b (k l n : ℕ) :
  P k l n - P (k - 1) l n = P k (l - 1) (n - k) :=
sorry

-- Part (c) statement
theorem part_c (k l n : ℕ) :
  P k l n = P l k n :=
sorry

-- Part (d) statement
theorem part_d (k l n : ℕ) :
  P k l n = P k l (k * l - n) :=
sorry

end part_a_part_b_part_c_part_d_l133_133709


namespace club_additional_members_l133_133165

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) 
  (h1 : current_members = 10) 
  (h2 : additional_members = 5 + 2 * current_members - current_members) : 
  additional_members = 15 :=
by 
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l133_133165


namespace question1_question2_question3_question4_l133_133960

theorem question1 : (2 * 3) ^ 2 = 2 ^ 2 * 3 ^ 2 := by admit

theorem question2 : (-1 / 2 * 2) ^ 3 = (-1 / 2) ^ 3 * 2 ^ 3 := by admit

theorem question3 : (3 / 2) ^ 2019 * (-2 / 3) ^ 2019 = -1 := by admit

theorem question4 (a b : ℝ) (n : ℕ) (h : 0 < n): (a * b) ^ n = a ^ n * b ^ n := by admit

end question1_question2_question3_question4_l133_133960


namespace domain_of_function_l133_133650

def domain_condition_1 (x : ℝ) : Prop := 1 - x > 0
def domain_condition_2 (x : ℝ) : Prop := x + 3 ≥ 0

def in_domain (x : ℝ) : Prop := domain_condition_1 x ∧ domain_condition_2 x

theorem domain_of_function : ∀ x : ℝ, in_domain x ↔ (-3 : ℝ) ≤ x ∧ x < 1 := 
by sorry

end domain_of_function_l133_133650


namespace fred_spent_18_42_l133_133163

variable (football_price : ℝ) (pokemon_price : ℝ) (baseball_price : ℝ)
variable (football_packs : ℕ) (pokemon_packs : ℕ) (baseball_decks : ℕ)

def total_cost (football_price : ℝ) (football_packs : ℕ) (pokemon_price : ℝ) (pokemon_packs : ℕ) (baseball_price : ℝ) (baseball_decks : ℕ) : ℝ :=
  football_packs * football_price + pokemon_packs * pokemon_price + baseball_decks * baseball_price

theorem fred_spent_18_42 :
  total_cost 2.73 2 4.01 1 8.95 1 = 18.42 :=
by
  sorry

end fred_spent_18_42_l133_133163


namespace no_base_6_digit_divisible_by_7_l133_133412

theorem no_base_6_digit_divisible_by_7 :
  ∀ (d : ℕ), d < 6 → ¬ (7 ∣ (652 + 42 * d)) :=
by
  intros d hd
  sorry

end no_base_6_digit_divisible_by_7_l133_133412


namespace third_place_prize_correct_l133_133037

-- Define the conditions and formulate the problem
def total_amount_in_pot : ℝ := 210
def third_place_percentage : ℝ := 0.15
def third_place_prize (P : ℝ) : ℝ := third_place_percentage * P

-- The theorem to be proved
theorem third_place_prize_correct : 
  third_place_prize total_amount_in_pot = 31.5 := 
by
  sorry

end third_place_prize_correct_l133_133037


namespace number_of_lawns_mowed_l133_133524

noncomputable def ChargePerLawn : ℕ := 33
noncomputable def TotalTips : ℕ := 30
noncomputable def TotalEarnings : ℕ := 558

theorem number_of_lawns_mowed (L : ℕ) 
  (h1 : ChargePerLawn * L + TotalTips = TotalEarnings) : L = 16 := 
by
  sorry

end number_of_lawns_mowed_l133_133524


namespace directrix_of_parabola_l133_133351

-- Define the given condition
def parabola_eq (x y : ℝ) : Prop := y = -4 * x^2

-- The problem we need to prove
theorem directrix_of_parabola :
  ∃ y : ℝ, (∀ x : ℝ, parabola_eq x y) ↔ y = 1 / 16 :=
by
  sorry

end directrix_of_parabola_l133_133351


namespace inv_matrix_A_l133_133687

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![ ![ -2, 1 ],
     ![ (3/2 : ℚ), -1/2 ] ]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![ ![ 1, 2 ],
     ![ 3, 4 ] ]

theorem inv_matrix_A : A⁻¹ = A_inv := by
  sorry

end inv_matrix_A_l133_133687


namespace max_positive_root_satisfies_range_l133_133802

noncomputable def max_positive_root_in_range (b c d : ℝ) (hb : |b| ≤ 1) (hc : |c| ≤ 1.5) (hd : |d| ≤ 1) : Prop :=
  ∃ s : ℝ, 2.5 ≤ s ∧ s < 3 ∧ ∃ x : ℝ, x > 0 ∧ x^3 + b * x^2 + c * x + d = 0

theorem max_positive_root_satisfies_range (b c d : ℝ) (hb : |b| ≤ 1) (hc : |c| ≤ 1.5) (hd : |d| ≤ 1) :
  max_positive_root_in_range b c d hb hc hd := sorry

end max_positive_root_satisfies_range_l133_133802


namespace four_digit_number_is_2561_l133_133436

-- Define the problem domain based on given conditions
def unique_in_snowflake_and_directions (grid : Matrix (Fin 3) (Fin 6) ℕ) : Prop :=
  ∀ (i j : Fin 3), -- across all directions
    ∀ (x y : Fin 6), 
      (x ≠ y) → 
      (grid i x ≠ grid i y) -- uniqueness in i-direction
      ∧ (grid y x ≠ grid y y) -- uniqueness in j-direction

-- Assignment of numbers in the grid fulfilling the conditions
def grid : Matrix (Fin 3) (Fin 6) ℕ :=
![ ![2, 5, 2, 5, 1, 6], ![4, 3, 2, 6, 1, 1], ![6, 1, 4, 5, 3, 2] ]

-- Definition of the four-digit number
def ABCD : ℕ := grid 0 1 * 1000 + grid 0 2 * 100 + grid 0 3 * 10 + grid 0 4

-- The theorem to be proved
theorem four_digit_number_is_2561 :
  unique_in_snowflake_and_directions grid →
  ABCD = 2561 :=
sorry

end four_digit_number_is_2561_l133_133436


namespace average_chemistry_mathematics_l133_133210

variable {P C M : ℝ}

theorem average_chemistry_mathematics (h : P + C + M = P + 150) : (C + M) / 2 = 75 :=
by
  sorry

end average_chemistry_mathematics_l133_133210


namespace initial_total_cards_l133_133173

theorem initial_total_cards (x y : ℕ) (h1 : x / (x + y) = 1 / 3) (h2 : x / (x + y + 4) = 1 / 4) : x + y = 12 := 
sorry

end initial_total_cards_l133_133173


namespace coins_in_second_stack_l133_133800

theorem coins_in_second_stack (total_coins : ℕ) (stack1_coins : ℕ) (stack2_coins : ℕ) 
  (H1 : total_coins = 12) (H2 : stack1_coins = 4) : stack2_coins = 8 :=
by
  -- The proof is omitted.
  sorry

end coins_in_second_stack_l133_133800


namespace guests_equal_cost_l133_133429

-- Rental costs and meal costs
def rental_caesars_palace : ℕ := 800
def deluxe_meal_cost : ℕ := 30
def premium_meal_cost : ℕ := 40
def rental_venus_hall : ℕ := 500
def venus_special_cost : ℕ := 35
def venus_platter_cost : ℕ := 45

-- Meal distribution percentages
def deluxe_meal_percentage : ℚ := 0.60
def premium_meal_percentage : ℚ := 0.40
def venus_special_percentage : ℚ := 0.60
def venus_platter_percentage : ℚ := 0.40

-- Total costs calculation
noncomputable def total_cost_caesars (G : ℕ) : ℚ :=
  rental_caesars_palace + deluxe_meal_cost * deluxe_meal_percentage * G + premium_meal_cost * premium_meal_percentage * G

noncomputable def total_cost_venus (G : ℕ) : ℚ :=
  rental_venus_hall + venus_special_cost * venus_special_percentage * G + venus_platter_cost * venus_platter_percentage * G

-- Statement to show the equivalence of guest count
theorem guests_equal_cost (G : ℕ) : total_cost_caesars G = total_cost_venus G → G = 60 :=
by
  sorry

end guests_equal_cost_l133_133429


namespace smallest_integer_k_l133_133466

theorem smallest_integer_k (k : ℕ) : 
  (k > 1 ∧ 
   k % 13 = 1 ∧ 
   k % 7 = 1 ∧ 
   k % 5 = 1 ∧ 
   k % 3 = 1) ↔ k = 1366 := 
sorry

end smallest_integer_k_l133_133466


namespace largest_a_for_integer_solution_l133_133285

theorem largest_a_for_integer_solution :
  ∃ a : ℝ, (∀ x y : ℤ, x - 4 * y = 1 ∧ a * x + 3 * y = 1) ∧ (∀ a' : ℝ, (∀ x y : ℤ, x - 4 * y = 1 ∧ a' * x + 3 * y = 1) → a' ≤ a) ∧ a = 1 :=
sorry

end largest_a_for_integer_solution_l133_133285


namespace arithmetic_sequence_a7_l133_133989

theorem arithmetic_sequence_a7 :
  ∀ (a : ℕ → ℕ) (d : ℕ),
  (∀ n, a (n + 1) = a n + d) →
  a 1 = 2 →
  a 3 + a 5 = 10 →
  a 7 = 8 :=
by
  intros a d h_seq h_a1 h_sum
  sorry

end arithmetic_sequence_a7_l133_133989


namespace remainder_7459_div_9_l133_133261

theorem remainder_7459_div_9 : 7459 % 9 = 7 := 
by
  sorry

end remainder_7459_div_9_l133_133261


namespace original_divisor_in_terms_of_Y_l133_133184

variables (N D Y : ℤ)
variables (h1 : N = 45 * D + 13) (h2 : N = 6 * Y + 4)

theorem original_divisor_in_terms_of_Y (h1 : N = 45 * D + 13) (h2 : N = 6 * Y + 4) : 
  D = (2 * Y - 3) / 15 :=
sorry

end original_divisor_in_terms_of_Y_l133_133184


namespace possible_dimensions_of_plot_l133_133065

theorem possible_dimensions_of_plot (x : ℕ) :
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ 1000 * a + 100 * a + 10 * b + b = x * (x + 1)) →
  x = 33 ∨ x = 66 ∨ x = 99 :=
sorry

end possible_dimensions_of_plot_l133_133065


namespace daisies_bought_l133_133299

theorem daisies_bought (cost_per_flower roses total_cost : ℕ) 
  (h1 : cost_per_flower = 3) 
  (h2 : roses = 8) 
  (h3 : total_cost = 30) : 
  (total_cost - (roses * cost_per_flower)) / cost_per_flower = 2 :=
by
  sorry

end daisies_bought_l133_133299


namespace total_days_needed_l133_133274

-- Define the conditions
def project1_questions : ℕ := 518
def project2_questions : ℕ := 476
def questions_per_day : ℕ := 142

-- Define the statement to prove
theorem total_days_needed :
  (project1_questions + project2_questions) / questions_per_day = 7 := by
  sorry

end total_days_needed_l133_133274


namespace decimal_to_fraction_l133_133448

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l133_133448


namespace find_difference_l133_133722

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

end find_difference_l133_133722


namespace average_decrease_l133_133645

theorem average_decrease (avg_6 : ℝ) (obs_7 : ℝ) (new_avg : ℝ) (decrease : ℝ) :
  avg_6 = 11 → obs_7 = 4 → (6 * avg_6 + obs_7) / 7 = new_avg → avg_6 - new_avg = decrease → decrease = 1 :=
  by
    intros h1 h2 h3 h4
    rw [h1, h2] at *
    sorry

end average_decrease_l133_133645


namespace smallest_sum_of_digits_l133_133339

noncomputable def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem smallest_sum_of_digits (n : ℕ) (h : sum_of_digits n = 2017) : sum_of_digits (n + 1) = 2 := 
sorry

end smallest_sum_of_digits_l133_133339


namespace final_sign_is_minus_l133_133949

theorem final_sign_is_minus 
  (plus_count : ℕ) 
  (minus_count : ℕ) 
  (h_plus : plus_count = 2004) 
  (h_minus : minus_count = 2005) 
  (transform : (ℕ → ℕ → ℕ × ℕ) → Prop) :
  transform (fun plus minus =>
    if plus >= 2 then (plus - 1, minus)
    else if minus >= 2 then (plus, minus - 1)
    else if plus > 0 && minus > 0 then (plus - 1, minus - 1)
    else (0, 0)) →
  (plus_count = 0 ∧ minus_count = 1) := sorry

end final_sign_is_minus_l133_133949


namespace find_input_values_f_l133_133514

theorem find_input_values_f (f : ℤ → ℤ) 
  (h_def : ∀ x, f (2 * x + 3) = (x - 3) * (x + 4))
  (h_val : ∃ y, f y = 170) : 
  ∃ (a b : ℤ), (a = -25 ∧ b = 29) ∧ (f a = 170 ∧ f b = 170) :=
by
  sorry

end find_input_values_f_l133_133514


namespace candles_ratio_l133_133191

-- Conditions
def kalani_bedroom_candles : ℕ := 20
def donovan_candles : ℕ := 20
def total_candles_house : ℕ := 50

-- Definitions for the number of candles in the living room and the ratio
def living_room_candles : ℕ := total_candles_house - kalani_bedroom_candles - donovan_candles
def ratio_of_candles : ℚ := kalani_bedroom_candles / living_room_candles

theorem candles_ratio : ratio_of_candles = 2 :=
by
  sorry

end candles_ratio_l133_133191


namespace find_total_kids_l133_133183

-- Given conditions
def total_kids_in_camp (X : ℕ) : Prop :=
  let soccer_kids := X / 2
  let morning_soccer_kids := soccer_kids / 4
  let afternoon_soccer_kids := soccer_kids - morning_soccer_kids
  afternoon_soccer_kids = 750

-- Theorem statement
theorem find_total_kids (X : ℕ) (h : total_kids_in_camp X) : X = 2000 :=
by
  sorry

end find_total_kids_l133_133183


namespace reading_comprehension_application_method_1_application_method_2_l133_133088

-- Reading Comprehension Problem in Lean 4
theorem reading_comprehension (x : ℝ) (h : x^2 + x + 5 = 8) : 2 * x^2 + 2 * x - 4 = 2 :=
by sorry

-- Application of Methods Problem (1) in Lean 4
theorem application_method_1 (x : ℝ) (h : x^2 + x + 2 = 9) : -2 * x^2 - 2 * x + 3 = -11 :=
by sorry

-- Application of Methods Problem (2) in Lean 4
theorem application_method_2 (a b : ℝ) (h : 8 * a + 2 * b = 5) : a * (-2)^3 + b * (-2) + 3 = -2 :=
by sorry

end reading_comprehension_application_method_1_application_method_2_l133_133088


namespace total_books_correct_l133_133430

-- Definitions based on the conditions
def num_books_bottom_shelf (T : ℕ) := T / 3
def num_books_middle_shelf (T : ℕ) := T / 4
def num_books_top_shelf : ℕ := 30
def total_books (T : ℕ) := num_books_bottom_shelf T + num_books_middle_shelf T + num_books_top_shelf

theorem total_books_correct : total_books 72 = 72 :=
by
  sorry

end total_books_correct_l133_133430


namespace unique_two_scoop_sundaes_l133_133547

theorem unique_two_scoop_sundaes (n : ℕ) (hn : n = 8) : ∃ k, k = Nat.choose 8 2 :=
by
  use 28
  sorry

end unique_two_scoop_sundaes_l133_133547


namespace prob1_prob2_l133_133759

theorem prob1:
  (6 * (Real.tan (30 * Real.pi / 180))^2 - Real.sqrt 3 * Real.sin (60 * Real.pi / 180) - 2 * Real.sin (45 * Real.pi / 180)) = (1 / 2 - Real.sqrt 2) :=
sorry

theorem prob2:
  ((Real.sqrt 2 / 2) * Real.cos (45 * Real.pi / 180) - (Real.tan (40 * Real.pi / 180) + 1)^0 + Real.sqrt (1 / 4) + Real.sin (30 * Real.pi / 180)) = (1 / 2) :=
sorry

end prob1_prob2_l133_133759


namespace smaller_number_is_5_l133_133844

theorem smaller_number_is_5 (x y : ℤ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 := by
  sorry

end smaller_number_is_5_l133_133844


namespace store_loss_90_l133_133623

theorem store_loss_90 (x y : ℝ) (h1 : x * (1 + 0.12) = 3080) (h2 : y * (1 - 0.12) = 3080) :
  2 * 3080 - x - y = -90 :=
by
  sorry

end store_loss_90_l133_133623


namespace perpendicular_line_l133_133264

theorem perpendicular_line (x y : ℝ) (h : 2 * x + y - 10 = 0) : 
    (∃ k : ℝ, (x = 1 ∧ y = 2) → (k * (-2) = -1)) → 
    (∃ m b : ℝ, b = 3 ∧ m = 1/2) → 
    (x - 2 * y + 3 = 0) := 
sorry

end perpendicular_line_l133_133264


namespace cos_pi_minus_2alpha_eq_seven_over_twentyfive_l133_133055

variable (α : ℝ)

theorem cos_pi_minus_2alpha_eq_seven_over_twentyfive 
  (h : Real.sin (π / 2 - α) = 3 / 5) :
  Real.cos (π - 2 * α) = 7 / 25 := 
by
  sorry

end cos_pi_minus_2alpha_eq_seven_over_twentyfive_l133_133055


namespace smallest_positive_angle_l133_133652

theorem smallest_positive_angle (k : ℤ) : ∃ α, α = 400 + k * 360 ∧ α > 0 ∧ α = 40 :=
by
  use 40
  sorry

end smallest_positive_angle_l133_133652


namespace jason_and_lisa_cards_l133_133637

-- Define the number of cards Jason originally had
def jason_original_cards (remaining: ℕ) (given_away: ℕ) : ℕ :=
  remaining + given_away

-- Define the number of cards Lisa originally had
def lisa_original_cards (remaining: ℕ) (given_away: ℕ) : ℕ :=
  remaining + given_away

-- State the main theorem to be proved
theorem jason_and_lisa_cards :
  jason_original_cards 4 9 + lisa_original_cards 7 15 = 35 :=
by
  sorry

end jason_and_lisa_cards_l133_133637


namespace usual_walk_time_l133_133256

theorem usual_walk_time (S T : ℝ)
  (h : S / (2/3 * S) = (T + 15) / T) : T = 30 :=
by
  sorry

end usual_walk_time_l133_133256


namespace equation_1_solve_equation_2_solve_l133_133130

-- The first equation
theorem equation_1_solve (x : ℝ) (h : 4 * (x - 2) = 2 * x) : x = 4 :=
by
  sorry

-- The second equation
theorem equation_2_solve (x : ℝ) (h : (x + 1) / 4 = 1 - (1 - x) / 3) : x = -5 :=
by
  sorry

end equation_1_solve_equation_2_solve_l133_133130


namespace pencil_notebook_cost_l133_133542

theorem pencil_notebook_cost (p n : ℝ)
  (h1 : 9 * p + 10 * n = 5.35)
  (h2 : 6 * p + 4 * n = 2.50) :
  24 * 0.9 * p + 15 * n = 9.24 :=
by 
  sorry

end pencil_notebook_cost_l133_133542


namespace juanita_loss_l133_133635

theorem juanita_loss
  (entry_fee : ℝ) (hit_threshold : ℕ) (drum_payment_per_hit : ℝ) (drums_hit : ℕ) :
  entry_fee = 10 →
  hit_threshold = 200 →
  drum_payment_per_hit = 0.025 →
  drums_hit = 300 →
  - (entry_fee - ((drums_hit - hit_threshold) * drum_payment_per_hit)) = 7.50 :=
by
  intros h1 h2 h3 h4
  sorry

end juanita_loss_l133_133635


namespace ratio_of_sums_l133_133706

open Nat

def sum_multiples_of_3 (n : Nat) : Nat :=
  let m := n / 3
  m * (3 + 3 * m) / 2

def sum_first_n_integers (n : Nat) : Nat :=
  n * (n + 1) / 2

theorem ratio_of_sums :
  (sum_multiples_of_3 600) / (sum_first_n_integers 300) = 4 / 3 :=
by
  sorry

end ratio_of_sums_l133_133706


namespace difference_is_1365_l133_133983

-- Define the conditions as hypotheses
def difference_between_numbers (L S : ℕ) : Prop :=
  L = 1637 ∧ L = 6 * S + 5

-- State the theorem to prove the difference is 1365
theorem difference_is_1365 {L S : ℕ} (h₁ : L = 1637) (h₂ : L = 6 * S + 5) :
  L - S = 1365 :=
by
  sorry

end difference_is_1365_l133_133983


namespace income_percentage_l133_133599

theorem income_percentage (J T M : ℝ) 
  (h1 : T = 0.5 * J) 
  (h2 : M = 1.6 * T) : 
  M = 0.8 * J :=
by 
  sorry

end income_percentage_l133_133599


namespace number_of_ordered_pairs_l133_133927

theorem number_of_ordered_pairs (x y : ℕ) : (x * y = 1716) → 
  (∃! n : ℕ, n = 18) :=
by
  sorry

end number_of_ordered_pairs_l133_133927


namespace John_reads_50_pages_per_hour_l133_133732

noncomputable def pages_per_hour (reads_daily hours : ℕ) (total_pages total_weeks : ℕ) : ℕ :=
  let days := total_weeks * 7
  let pages_per_day := total_pages / days
  pages_per_day / reads_daily

theorem John_reads_50_pages_per_hour :
  pages_per_hour 2 2800 4 = 50 := by
  sorry

end John_reads_50_pages_per_hour_l133_133732


namespace billy_scores_two_points_each_round_l133_133090

def billy_old_score := 725
def billy_rounds := 363
def billy_target_score := billy_old_score + 1
def billy_points_per_round := billy_target_score / billy_rounds

theorem billy_scores_two_points_each_round :
  billy_points_per_round = 2 := by
  sorry

end billy_scores_two_points_each_round_l133_133090


namespace factorize_x_squared_minus_four_l133_133682

theorem factorize_x_squared_minus_four (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) :=
by
  sorry

end factorize_x_squared_minus_four_l133_133682


namespace scientific_notation_35_million_l133_133441

theorem scientific_notation_35_million :
  35000000 = 3.5 * (10 : Float) ^ 7 := 
by
  sorry

end scientific_notation_35_million_l133_133441


namespace range_of_a_l133_133858

variables (a b c : ℝ)

theorem range_of_a (h₁ : a^2 - b * c - 8 * a + 7 = 0)
                   (h₂ : b^2 + c^2 + b * c - 6 * a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end range_of_a_l133_133858


namespace sum_of_997_lemons_l133_133442

-- Define x and y as functions of k
def x (k : ℕ) := 1 + 9 * k
def y (k : ℕ) := 110 - 7 * k

-- The theorem we need to prove
theorem sum_of_997_lemons :
  ∃ (k : ℕ), 0 ≤ k ∧ k ≤ 15 ∧ 7 * (x k) + 9 * (y k) = 997 := 
by
  sorry -- Proof to be filled in

end sum_of_997_lemons_l133_133442


namespace train_speed_l133_133444

noncomputable def speed_in_kmh (distance : ℕ) (time : ℕ) : ℚ :=
  (distance : ℚ) / (time : ℚ) * 3600 / 1000

theorem train_speed
  (distance : ℕ) (time : ℕ)
  (h_dist : distance = 150)
  (h_time : time = 9) :
  speed_in_kmh distance time = 60 :=
by
  rw [h_dist, h_time]
  sorry

end train_speed_l133_133444


namespace area_of_right_triangle_ABC_l133_133000

variable {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

noncomputable def area_triangle_ABC (AB BC : ℝ) (angleB : ℕ) (hangle : angleB = 90) (hAB : AB = 30) (hBC : BC = 40) : ℝ :=
  1 / 2 * AB * BC

theorem area_of_right_triangle_ABC (AB BC : ℝ) (angleB : ℕ) (hangle : angleB = 90) 
  (hAB : AB = 30) (hBC : BC = 40) : 
  area_triangle_ABC AB BC angleB hangle hAB hBC = 600 :=
by
  sorry

end area_of_right_triangle_ABC_l133_133000


namespace percentage_discount_l133_133143

-- Define the given conditions
def equal_contribution (total: ℕ) (num_people: ℕ) := total / num_people

def original_contribution (amount_paid: ℕ) (discount: ℕ) := amount_paid + discount

def total_original_cost (individual_original: ℕ) (num_people: ℕ) := individual_original * num_people

def discount_amount (original_cost: ℕ) (discounted_cost: ℕ) := original_cost - discounted_cost

def discount_percentage (discount: ℕ) (original_cost: ℕ) := (discount * 100) / original_cost

-- Given conditions
def given_total := 48
def given_num_people := 3
def amount_paid_each := equal_contribution given_total given_num_people
def discount_each := 4
def original_payment_each := original_contribution amount_paid_each discount_each
def original_total_cost := total_original_cost original_payment_each given_num_people
def paid_total := 48

-- Question: What is the percentage discount
theorem percentage_discount :
  discount_percentage (discount_amount original_total_cost paid_total) original_total_cost = 20 :=
by
  sorry

end percentage_discount_l133_133143


namespace original_percentage_alcohol_l133_133916

-- Definitions of the conditions
def original_mixture_volume : ℝ := 15
def additional_water_volume : ℝ := 3
def final_percentage_alcohol : ℝ := 20.833333333333336
def final_mixture_volume : ℝ := original_mixture_volume + additional_water_volume

-- Lean statement to prove
theorem original_percentage_alcohol (A : ℝ) :
  (A / 100 * original_mixture_volume) = (final_percentage_alcohol / 100 * final_mixture_volume) →
  A = 25 :=
by
  sorry

end original_percentage_alcohol_l133_133916


namespace least_positive_int_satisfies_congruence_l133_133138

theorem least_positive_int_satisfies_congruence :
  ∃ x : ℕ, (x + 3001) % 15 = 1723 % 15 ∧ x = 12 :=
by
  sorry

end least_positive_int_satisfies_congruence_l133_133138


namespace identify_ATM_mistakes_additional_security_measures_l133_133647

-- Define the conditions as Boolean variables representing different mistakes and measures
variables (writing_PIN_on_card : Prop)
variables (using_ATM_despite_difficulty : Prop)
variables (believing_stranger : Prop)
variables (walking_away_without_card : Prop)
variables (use_trustworthy_locations : Prop)
variables (presence_during_transactions : Prop)
variables (enable_SMS_notifications : Prop)
variables (call_bank_for_suspicious_activities : Prop)
variables (be_cautious_of_fake_SMS_alerts : Prop)
variables (store_transaction_receipts : Prop)
variables (shield_PIN : Prop)
variables (use_chipped_cards : Prop)
variables (avoid_high_risk_ATMs : Prop)

-- Prove that the identified mistakes occur given the conditions
theorem identify_ATM_mistakes :
  writing_PIN_on_card ∧ using_ATM_despite_difficulty ∧ 
  believing_stranger ∧ walking_away_without_card := sorry

-- Prove that the additional security measures should be followed
theorem additional_security_measures :
  use_trustworthy_locations ∧ presence_during_transactions ∧ 
  enable_SMS_notifications ∧ call_bank_for_suspicious_activities ∧ 
  be_cautious_of_fake_SMS_alerts ∧ store_transaction_receipts ∧ 
  shield_PIN ∧ use_chipped_cards ∧ avoid_high_risk_ATMs := sorry

end identify_ATM_mistakes_additional_security_measures_l133_133647


namespace tan_ratio_l133_133968

open Real

variables (p q : ℝ)

-- Conditions
def cond1 := (sin p / cos q + sin q / cos p = 2)
def cond2 := (cos p / sin q + cos q / sin p = 3)

-- Proof statement
theorem tan_ratio (hpq : cond1 p q) (hq : cond2 p q) :
  (tan p / tan q + tan q / tan p = 8 / 5) :=
sorry

end tan_ratio_l133_133968


namespace sqrt_expression_eq_twelve_l133_133753

theorem sqrt_expression_eq_twelve : Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt 27) = 12 := 
sorry

end sqrt_expression_eq_twelve_l133_133753


namespace calculate_neg_three_minus_one_l133_133544

theorem calculate_neg_three_minus_one : -3 - 1 = -4 := by
  sorry

end calculate_neg_three_minus_one_l133_133544


namespace calculate_expression_l133_133109

def x : Float := 3.241
def y : Float := 14
def z : Float := 100
def expected_result : Float := 0.45374

theorem calculate_expression : (x * y) / z = expected_result := by
  sorry

end calculate_expression_l133_133109


namespace no_intersection_range_k_l133_133046

def problem_statement (k : ℝ) : Prop :=
  ∀ (x : ℝ),
    ¬(x > 1 ∧ x + 1 = k * x + 2) ∧ ¬(x < 1 ∧ -x - 1 = k * x + 2) ∧ 
    (x = 1 → (x + 1 ≠ k * x + 2 ∧ -x - 1 ≠ k * x + 2))

theorem no_intersection_range_k :
  ∀ (k : ℝ), problem_statement k ↔ -4 ≤ k ∧ k < -1 :=
sorry

end no_intersection_range_k_l133_133046


namespace expected_value_of_girls_left_of_boys_l133_133271

def num_girls_to_left_of_all_boys (boys girls : ℕ) : ℚ :=
  (boys + girls : ℚ) / (boys + 1)

theorem expected_value_of_girls_left_of_boys :
  num_girls_to_left_of_all_boys 10 7 = 7 / 11 := by
  sorry

end expected_value_of_girls_left_of_boys_l133_133271


namespace two_presses_printing_time_l133_133594

def printing_time (presses newspapers hours : ℕ) : ℕ := sorry

theorem two_presses_printing_time :
  ∀ (presses newspapers hours : ℕ),
    (presses = 4) →
    (newspapers = 8000) →
    (hours = 6) →
    printing_time 2 6000 hours = 9 := sorry

end two_presses_printing_time_l133_133594


namespace mr_c_gain_1000_l133_133957

-- Define the initial conditions
def initial_mr_c_cash := 15000
def initial_mr_c_house := 12000
def initial_mrs_d_cash := 16000

-- Define the changes in the house value
def house_value_appreciated := 13000
def house_value_depreciated := 11000

-- Define the cash changes after transactions
def mr_c_cash_after_first_sale := initial_mr_c_cash + house_value_appreciated
def mrs_d_cash_after_first_sale := initial_mrs_d_cash - house_value_appreciated
def mrs_d_cash_after_second_sale := mrs_d_cash_after_first_sale + house_value_depreciated
def mr_c_cash_after_second_sale := mr_c_cash_after_first_sale - house_value_depreciated

-- Define the final net worth for Mr. C
def final_mr_c_cash := mr_c_cash_after_second_sale
def final_mr_c_house := house_value_depreciated
def final_mr_c_net_worth := final_mr_c_cash + final_mr_c_house
def initial_mr_c_net_worth := initial_mr_c_cash + initial_mr_c_house

-- Statement to prove
theorem mr_c_gain_1000 : final_mr_c_net_worth = initial_mr_c_net_worth + 1000 := by
  sorry

end mr_c_gain_1000_l133_133957


namespace tree_height_when_planted_l133_133830

def initial_height (current_height : ℕ) (growth_rate : ℕ) (current_age : ℕ) (initial_age : ℕ) : ℕ :=
  current_height - (current_age - initial_age) * growth_rate

theorem tree_height_when_planted :
  initial_height 23 3 7 1 = 5 :=
by
  sorry

end tree_height_when_planted_l133_133830


namespace wire_cut_l133_133267

theorem wire_cut (total_length : ℝ) (ratio : ℝ) (shorter longer : ℝ) (h_total : total_length = 21) (h_ratio : ratio = 2/5)
  (h_shorter : longer = (5/2) * shorter) (h_sum : total_length = shorter + longer) : shorter = 6 := 
by
  -- total_length = 21, ratio = 2/5, longer = (5/2) * shorter, total_length = shorter + longer, prove shorter = 6
  sorry

end wire_cut_l133_133267


namespace fermat_prime_sum_not_possible_l133_133579

-- Definitions of the conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, (m ∣ p) → (m = 1 ∨ m = p)

-- The Lean statement
theorem fermat_prime_sum_not_possible 
  (n : ℕ) (x y z : ℤ) (p : ℕ) 
  (h_odd : is_odd n) 
  (h_gt_one : n > 1) 
  (h_prime : is_prime p)
  (h_sum: x + y = ↑p) :
  ¬ (x ^ n + y ^ n = z ^ n) :=
by
  sorry


end fermat_prime_sum_not_possible_l133_133579


namespace baseball_wins_l133_133494

-- Define the constants and conditions
def total_games : ℕ := 130
def won_more_than_lost (L W : ℕ) : Prop := W = 3 * L + 14
def total_games_played (L W : ℕ) : Prop := W + L = total_games

-- Define the theorem statement
theorem baseball_wins (L W : ℕ) 
  (h1 : won_more_than_lost L W)
  (h2 : total_games_played L W) : 
  W = 101 :=
  sorry

end baseball_wins_l133_133494


namespace susan_age_in_5_years_l133_133707

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

end susan_age_in_5_years_l133_133707


namespace possible_values_of_ratio_l133_133470

theorem possible_values_of_ratio (a d : ℝ) (h : a ≠ 0) (h_eq : a^2 - 6 * a * d + 8 * d^2 = 0) : 
  ∃ x : ℝ, (x = 1/2 ∨ x = 1/4) ∧ x = d/a :=
by
  sorry

end possible_values_of_ratio_l133_133470


namespace probability_point_between_C_and_D_l133_133735

theorem probability_point_between_C_and_D :
  ∀ (A B C D E : ℝ), A < B ∧ C < D ∧
  (B - A = 4 * (D - A)) ∧ (B - A = 4 * (B - E)) ∧
  (D - A = C - D) ∧ (C - D = E - C) ∧ (E - C = B - E) →
  (B - A ≠ 0) → 
  (C - D) / (B - A) = 1 / 4 :=
by
  intros A B C D E hAB hNonZero
  sorry

end probability_point_between_C_and_D_l133_133735


namespace problem_statement_l133_133677

theorem problem_statement (x1 x2 x3 : ℝ) 
  (h1 : x1 < x2)
  (h2 : x2 < x3)
  (h3 : (45*x1^3 - 4050*x1^2 - 4 = 0) ∧ 
        (45*x2^3 - 4050*x2^2 - 4 = 0) ∧ 
        (45*x3^3 - 4050*x3^2 - 4 = 0)) :
  x2 * (x1 + x3) = 0 :=
by
  sorry

end problem_statement_l133_133677


namespace soccer_balls_donated_l133_133083

def num_classes_per_school (elem_classes mid_classes : ℕ) : ℕ :=
  elem_classes + mid_classes

def total_classes (num_schools : ℕ) (classes_per_school : ℕ) : ℕ :=
  num_schools * classes_per_school

def total_soccer_balls (num_classes : ℕ) (balls_per_class : ℕ) : ℕ :=
  num_classes * balls_per_class

theorem soccer_balls_donated 
  (elem_classes mid_classes num_schools balls_per_class : ℕ) 
  (h_elem_classes : elem_classes = 4) 
  (h_mid_classes : mid_classes = 5) 
  (h_num_schools : num_schools = 2) 
  (h_balls_per_class : balls_per_class = 5) :
  total_soccer_balls (total_classes num_schools (num_classes_per_school elem_classes mid_classes)) balls_per_class = 90 :=
by
  sorry

end soccer_balls_donated_l133_133083


namespace green_bows_count_l133_133900

noncomputable def total_bows : ℕ := 36 * 4

def fraction_green : ℚ := 1/6

theorem green_bows_count (red blue green total yellow : ℕ) (h_red : red = total / 4)
  (h_blue : blue = total / 3) (h_green : green = total / 6)
  (h_yellow : yellow = total - red - blue - green)
  (h_yellow_count : yellow = 36) : green = 24 := by
  sorry

end green_bows_count_l133_133900


namespace last_digit_expr_is_4_l133_133021

-- Definitions for last digits.
def last_digit (n : ℕ) : ℕ := n % 10

def a : ℕ := 287
def b : ℕ := 269

def expr := (a * a) + (b * b) - (2 * a * b)

-- Conjecture stating that the last digit of the given expression is 4.
theorem last_digit_expr_is_4 : last_digit expr = 4 := 
by sorry

end last_digit_expr_is_4_l133_133021


namespace sum_of_tangents_l133_133673

noncomputable def function_f (x : ℝ) : ℝ :=
  max (max (4 * x + 20) (-x + 2)) (5 * x - 3)

theorem sum_of_tangents (q : ℝ → ℝ) (a b c : ℝ) (h1 : ∀ x, q x - (4 * x + 20) = q x - function_f x)
  (h2 : ∀ x, q x - (-x + 2) = q x - function_f x)
  (h3 : ∀ x, q x - (5 * x - 3) = q x - function_f x) :
  a + b + c = -83 / 10 :=
sorry

end sum_of_tangents_l133_133673


namespace find_x_plus_one_over_x_l133_133327

variable (x : ℝ)

theorem find_x_plus_one_over_x
  (h1 : x^3 + (1/x)^3 = 110)
  (h2 : (x + 1/x)^2 - 2*x - 2*(1/x) = 38) :
  x + 1/x = 5 :=
sorry

end find_x_plus_one_over_x_l133_133327


namespace oil_amount_correct_l133_133363

-- Definitions based on the conditions in the problem
def initial_amount : ℝ := 0.16666666666666666
def additional_amount : ℝ := 0.6666666666666666
def final_amount : ℝ := 0.8333333333333333

-- Lean 4 statement to prove the given problem
theorem oil_amount_correct :
  initial_amount + additional_amount = final_amount :=
by
  sorry

end oil_amount_correct_l133_133363


namespace A_inter_B_is_empty_l133_133500

def A : Set (ℤ × ℤ) := {p | ∃ x : ℤ, p = (x, x + 1)}
def B : Set ℤ := {y | ∃ x : ℤ, y = 2 * x}

theorem A_inter_B_is_empty : A ∩ (fun p => p.2 ∈ B) = ∅ :=
by {
  sorry
}

end A_inter_B_is_empty_l133_133500


namespace smallest_x_for_multiple_l133_133221

theorem smallest_x_for_multiple (x : ℕ) (h₁: 450 = 2 * 3^2 * 5^2) (h₂: 800 = 2^6 * 5^2) : 
  ((450 * x) % 800 = 0) ↔ x ≥ 32 :=
by
  sorry

end smallest_x_for_multiple_l133_133221


namespace remainder_when_a_squared_times_b_divided_by_n_l133_133511

theorem remainder_when_a_squared_times_b_divided_by_n (n : ℕ) (a : ℤ) (h1 : a * 3 ≡ 1 [ZMOD n]) : 
  (a^2 * 3) % n = a % n := 
by
  sorry

end remainder_when_a_squared_times_b_divided_by_n_l133_133511


namespace two_digit_numbers_div_quotient_remainder_l133_133348

theorem two_digit_numbers_div_quotient_remainder (x y : ℕ) (N : ℕ) (h1 : N = 10 * x + y) (h2 : N = 7 * (x + y) + 6) (hx_range : 1 ≤ x ∧ x ≤ 9) (hy_range : 0 ≤ y ∧ y ≤ 9) :
  N = 62 ∨ N = 83 := sorry

end two_digit_numbers_div_quotient_remainder_l133_133348


namespace sound_frequency_and_speed_glass_proof_l133_133536

def length_rod : ℝ := 1.10 -- Length of the glass rod, l in meters
def nodal_distance_air : ℝ := 0.12 -- Distance between nodal points in air, l' in meters
def speed_sound_air : ℝ := 340 -- Speed of sound in air, V in meters per second

-- Frequency of the sound produced
def frequency_sound_produced : ℝ := 1416.67

-- Speed of longitudinal waves in the glass
def speed_longitudinal_glass : ℝ := 3116.67

theorem sound_frequency_and_speed_glass_proof :
  (2 * nodal_distance_air = 0.24) ∧
  (frequency_sound_produced * (2 * length_rod) = speed_longitudinal_glass) :=
by
  -- Here we will include real equivalent math proof in the future
  sorry

end sound_frequency_and_speed_glass_proof_l133_133536


namespace cos_of_vector_dot_product_l133_133398

open Real

noncomputable def cos_value (x : ℝ) : ℝ := cos (x + π / 4)

theorem cos_of_vector_dot_product (x : ℝ)
  (h1 : π / 4 < x)
  (h2 : x < π / 2)
  (h3 : (sqrt 2) * cos x + (sqrt 2) * sin x = 8 / 5) :
  cos_value x = - 3 / 5 :=
by
  sorry

end cos_of_vector_dot_product_l133_133398


namespace intersection_x_sum_l133_133905

theorem intersection_x_sum :
  ∃ x : ℤ, (0 ≤ x ∧ x < 17) ∧ (4 * x + 3 ≡ 13 * x + 14 [ZMOD 17]) ∧ x = 5 :=
by
  sorry

end intersection_x_sum_l133_133905


namespace A_days_to_complete_alone_l133_133688

theorem A_days_to_complete_alone
  (work_left : ℝ := 0.41666666666666663)
  (B_days : ℝ := 20)
  (combined_days : ℝ := 5)
  : ∃ (A_days : ℝ), A_days = 15 := 
by
  sorry

end A_days_to_complete_alone_l133_133688


namespace percentage_increase_in_rectangle_area_l133_133022

theorem percentage_increase_in_rectangle_area (L W : ℝ) :
  (1.35 * 1.35 * L * W - L * W) / (L * W) * 100 = 82.25 :=
by sorry

end percentage_increase_in_rectangle_area_l133_133022


namespace focal_length_of_lens_l133_133357

-- Define the conditions
def initial_screen_distance : ℝ := 80
def moved_screen_distance : ℝ := 40
def lens_formula (f v u : ℝ) : Prop := (1 / f) = (1 / v) + (1 / u)

-- Define the proof goal
theorem focal_length_of_lens :
  ∃ f : ℝ, (f = 100 ∨ f = 60) ∧
  lens_formula f f (1 / 0) ∧  -- parallel beam implies object at infinity u = 1/0
  initial_screen_distance = 80 ∧
  moved_screen_distance = 40 :=
sorry

end focal_length_of_lens_l133_133357


namespace cos_double_angle_identity_l133_133525

open Real

theorem cos_double_angle_identity (α : ℝ) 
  (h : tan (α + π / 4) = 1 / 3) : cos (2 * α) = 3 / 5 :=
sorry

end cos_double_angle_identity_l133_133525


namespace smallest_prime_divides_sum_l133_133027

theorem smallest_prime_divides_sum :
  ∃ a, Prime a ∧ a ∣ (3 ^ 11 + 5 ^ 13) ∧
       ∀ b, Prime b → b ∣ (3 ^ 11 + 5 ^ 13) → a ≤ b :=
sorry

end smallest_prime_divides_sum_l133_133027


namespace min_value_expression_l133_133756

theorem min_value_expression (x : ℝ) (h : x > 3) : x + 4 / (x - 3) ≥ 7 :=
sorry

end min_value_expression_l133_133756


namespace problem_solution_l133_133258

noncomputable def greatest_integer_not_exceeding (z : ℝ) : ℤ := Int.floor z

theorem problem_solution (x : ℝ) (y : ℝ) 
  (h1 : y = 4 * greatest_integer_not_exceeding x + 4)
  (h2 : y = 5 * greatest_integer_not_exceeding (x - 3) + 7)
  (h3 : x > 3 ∧ ¬ ∃ (n : ℤ), x = ↑n) :
  64 < x + y ∧ x + y < 65 :=
by
  sorry

end problem_solution_l133_133258


namespace sumata_family_miles_driven_per_day_l133_133331

theorem sumata_family_miles_driven_per_day :
  let total_miles := 1837.5
  let number_of_days := 13.5
  let miles_per_day := total_miles / number_of_days
  (miles_per_day : Real) = 136.1111 :=
by
  sorry

end sumata_family_miles_driven_per_day_l133_133331


namespace us2_eq_3958_div_125_l133_133738

-- Definitions based on conditions
def t (x : ℚ) : ℚ := 5 * x - 12
def s (t_x : ℚ) : ℚ := (2 : ℚ) ^ 2 + 3 * 2 - 2
def u (s_t_x : ℚ) : ℚ := (14 : ℚ) / 5 ^ 3 + 2 * (14 / 5) ^ 2 - 14 / 5 + 4

-- Prove that u(s(2)) = 3958 / 125
theorem us2_eq_3958_div_125 : u (s (2)) = 3958 / 125 := by
  sorry

end us2_eq_3958_div_125_l133_133738


namespace convert_to_spherical_l133_133562

noncomputable def spherical_coordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let φ := Real.arccos (z / ρ)
  let θ := if y / x < 0 then Real.arctan (-y / x) + 2 * Real.pi else Real.arctan (y / x)
  (ρ, θ, φ)

theorem convert_to_spherical :
  let x := 1
  let y := -4 * Real.sqrt 3
  let z := 4
  spherical_coordinates x y z = (Real.sqrt 65, Real.arctan (-4 * Real.sqrt 3) + 2 * Real.pi, Real.arccos (4 / (Real.sqrt 65))) :=
by
  sorry

end convert_to_spherical_l133_133562


namespace sum_of_midpoint_coords_l133_133018

theorem sum_of_midpoint_coords (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 3) (hy1 : y1 = 5) (hx2 : x2 = 11) (hy2 : y2 = 21) :
  ((x1 + x2) / 2 + (y1 + y2) / 2) = 20 :=
by
  sorry

end sum_of_midpoint_coords_l133_133018


namespace quotient_division_l133_133319

/-- Definition of the condition that when 14 is divided by 3, the remainder is 2 --/
def division_property : Prop :=
  14 = 3 * (14 / 3) + 2

/-- Statement for finding the quotient when 14 is divided by 3 --/
theorem quotient_division (A : ℕ) (h : 14 = 3 * A + 2) : A = 4 :=
by
  have rem_2 := division_property
  sorry

end quotient_division_l133_133319


namespace percentage_in_quarters_l133_133265

theorem percentage_in_quarters (dimes quarters nickels : ℕ) (value_dime value_quarter value_nickel : ℕ)
  (h_dimes : dimes = 40)
  (h_quarters : quarters = 30)
  (h_nickels : nickels = 10)
  (h_value_dime : value_dime = 10)
  (h_value_quarter : value_quarter = 25)
  (h_value_nickel : value_nickel = 5) :
  (quarters * value_quarter : ℚ) / ((dimes * value_dime + quarters * value_quarter + nickels * value_nickel) : ℚ) * 100 = 62.5 := 
  sorry

end percentage_in_quarters_l133_133265


namespace solution_of_system_l133_133451

theorem solution_of_system :
  (∀ x : ℝ,
    (2 + x < 6 - 3 * x) ∧ (x ≤ (4 + x) / 2)
    → x < 1) :=
by
  sorry

end solution_of_system_l133_133451


namespace factorization_correct_l133_133044

-- Define noncomputable to deal with the natural arithmetic operations
noncomputable def a : ℕ := 66
noncomputable def b : ℕ := 231

-- Define the given expressions
noncomputable def lhs (x : ℕ) : ℤ := ((a : ℤ) * x^6) - ((b : ℤ) * x^12)
noncomputable def rhs (x : ℕ) : ℤ := (33 : ℤ) * x^6 * (2 - 7 * x^6)

-- The theorem to prove the equality
theorem factorization_correct (x : ℕ) : lhs x = rhs x :=
by sorry

end factorization_correct_l133_133044


namespace line_in_slope_intercept_form_l133_133964

def vec1 : ℝ × ℝ := (3, -7)
def point : ℝ × ℝ := (-2, 4)
def line_eq (x y : ℝ) : Prop := vec1.1 * (x - point.1) + vec1.2 * (y - point.2) = 0

theorem line_in_slope_intercept_form (x y : ℝ) : line_eq x y → y = (3 / 7) * x - (34 / 7) :=
by
  sorry

end line_in_slope_intercept_form_l133_133964


namespace tangent_line_equation_at_point_l133_133102

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1) - x

theorem tangent_line_equation_at_point :
  ∃ a b c : ℝ, (∀ x y : ℝ, a * x + b * y + c = 0 ↔ (x = 1 → y = -1 → f x = y)) ∧ (a * 1 + b * (-1) + c = 0) :=
by
  sorry

end tangent_line_equation_at_point_l133_133102


namespace inequality_proof_l133_133497

theorem inequality_proof (x : ℝ) : 3 * x - 6 > 5 * (x - 2) → x < 2 :=
by
  sorry

end inequality_proof_l133_133497


namespace tea_mixture_price_l133_133075

theorem tea_mixture_price :
  ∀ (price_A price_B : ℝ) (ratio_A ratio_B : ℝ),
  price_A = 65 →
  price_B = 70 →
  ratio_A = 1 →
  ratio_B = 1 →
  (price_A * ratio_A + price_B * ratio_B) / (ratio_A + ratio_B) = 67.5 :=
by
  intros price_A price_B ratio_A ratio_B h1 h2 h3 h4
  sorry

end tea_mixture_price_l133_133075


namespace grace_mowing_hours_l133_133768

-- Definitions for conditions
def earnings_mowing (x : ℕ) : ℕ := 6 * x
def earnings_weeds : ℕ := 11 * 9
def earnings_mulch : ℕ := 9 * 10
def total_september_earnings (x : ℕ) : ℕ := earnings_mowing x + earnings_weeds + earnings_mulch

-- Proof statement (with the total earnings of 567 specified)
theorem grace_mowing_hours (x : ℕ) (h : total_september_earnings x = 567) : x = 63 := by
  sorry

end grace_mowing_hours_l133_133768


namespace abes_age_l133_133491

theorem abes_age (A : ℕ) (h : A + (A - 7) = 29) : A = 18 :=
by
  sorry

end abes_age_l133_133491


namespace combined_weight_of_parcels_l133_133995

variable (x y z : ℕ)

theorem combined_weight_of_parcels : 
  (x + y = 132) ∧ (y + z = 135) ∧ (z + x = 140) → x + y + z = 204 :=
by 
  intros
  sorry

end combined_weight_of_parcels_l133_133995


namespace average_loss_l133_133051

theorem average_loss (cost_per_lootbox : ℝ) (average_value_per_lootbox : ℝ) (total_spent : ℝ)
                      (h1 : cost_per_lootbox = 5)
                      (h2 : average_value_per_lootbox = 3.5)
                      (h3 : total_spent = 40) :
  (total_spent - (average_value_per_lootbox * (total_spent / cost_per_lootbox))) = 12 :=
by
  sorry

end average_loss_l133_133051


namespace intersection_A_B_is_1_and_2_l133_133700

def A : Set ℝ := {x | x ^ 2 - 3 * x - 4 < 0}
def B : Set ℝ := {-2, -1, 1, 2, 4}

theorem intersection_A_B_is_1_and_2 : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_is_1_and_2_l133_133700


namespace percent_of_dollar_in_pocket_l133_133888

def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def half_dollar_value : ℕ := 50

theorem percent_of_dollar_in_pocket :
  let total_cents := penny_value + nickel_value + dime_value + quarter_value + half_dollar_value
  total_cents = 91 := by
  sorry

end percent_of_dollar_in_pocket_l133_133888


namespace exists_three_points_l133_133278

theorem exists_three_points (n : ℕ) (h : 3 ≤ n) (points : Fin n → EuclideanSpace ℝ (Fin 2))
  (distinct : ∀ i j : Fin n, i ≠ j → points i ≠ points j) :
  ∃ (A B C : Fin n),
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    1 ≤ dist (points A) (points B) / dist (points A) (points C) ∧ 
    dist (points A) (points B) / dist (points A) (points C) < (n + 1) / (n - 1) := 
sorry

end exists_three_points_l133_133278


namespace printer_to_enhanced_ratio_l133_133806

def B : ℕ := 2125
def P : ℕ := 2500 - B
def E : ℕ := B + 500
def total_price := E + P

theorem printer_to_enhanced_ratio :
  (P : ℚ) / total_price = 1 / 8 := 
by {
  -- skipping the proof
  sorry
}

end printer_to_enhanced_ratio_l133_133806


namespace f_decreasing_on_0_1_l133_133877

noncomputable def f (x : ℝ) : ℝ := x + 1 / x

theorem f_decreasing_on_0_1 : ∀ (x1 x2 : ℝ), (x1 ∈ Set.Ioo 0 1) → (x2 ∈ Set.Ioo 0 1) → (x1 < x2) → (f x1 < f x2) := by
  sorry

end f_decreasing_on_0_1_l133_133877


namespace complex_abs_sum_eq_1_or_3_l133_133736

open Complex

theorem complex_abs_sum_eq_1_or_3 (a b c : ℂ) (ha : abs a = 1) (hb : abs b = 1) (hc : abs c = 1) 
  (h : a^3/(b^2 * c) + b^3/(a^2 * c) + c^3/(a^2 * b) = 1) : abs (a + b + c) = 1 ∨ abs (a + b + c) = 3 := 
by
  sorry

end complex_abs_sum_eq_1_or_3_l133_133736


namespace opposite_of_negative_2020_is_2020_l133_133631

theorem opposite_of_negative_2020_is_2020 :
  ∃ x : ℤ, -2020 + x = 0 :=
by
  use 2020
  sorry

end opposite_of_negative_2020_is_2020_l133_133631


namespace primes_with_prime_remainders_l133_133122

namespace PrimePuzzle

open Nat

def primes_between (a b : Nat) : List Nat :=
  (List.range' (a + 1) (b - a)).filter Nat.Prime

def prime_remainders (lst : List Nat) (m : Nat) : List Nat :=
  (lst.map (λ n => n % m)).filter Nat.Prime

theorem primes_with_prime_remainders : 
  primes_between 40 85 = [41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83] ∧ 
  prime_remainders [41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83] 12 = [5, 7, 7, 11, 11, 7, 11] ∧ 
  (prime_remainders [41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83] 12).toFinset.card = 9 := 
by 
  sorry

end PrimePuzzle

end primes_with_prime_remainders_l133_133122


namespace center_of_circle_l133_133654

noncomputable def center_is_correct (x y : ℚ) : Prop :=
  (5 * x - 2 * y = -10) ∧ (3 * x + y = 0)

theorem center_of_circle : center_is_correct (-10 / 11) (30 / 11) :=
by
  sorry

end center_of_circle_l133_133654


namespace rectangle_dimensions_l133_133156

theorem rectangle_dimensions (x : ℝ) (h : 3 * x * x = 8 * x) : (x = 8 / 3 ∧ 3 * x = 8) :=
by {
  sorry
}

end rectangle_dimensions_l133_133156


namespace pond_contains_total_money_correct_l133_133149

def value_of_dime := 10
def value_of_quarter := 25
def value_of_nickel := 5
def value_of_penny := 1

def cindy_dimes := 5
def eric_quarters := 3
def garrick_nickels := 8
def ivy_pennies := 60

def total_money : ℕ := 
  cindy_dimes * value_of_dime + 
  eric_quarters * value_of_quarter + 
  garrick_nickels * value_of_nickel + 
  ivy_pennies * value_of_penny

theorem pond_contains_total_money_correct:
  total_money = 225 := by
  sorry

end pond_contains_total_money_correct_l133_133149


namespace tommy_initial_balloons_l133_133455

theorem tommy_initial_balloons :
  ∃ x : ℝ, x + 78.5 = 132.25 ∧ x = 53.75 := by
  sorry

end tommy_initial_balloons_l133_133455


namespace digits_interchanged_l133_133461

theorem digits_interchanged (a b k : ℤ) (h : 10 * a + b = k * (a + b) + 2) :
  10 * b + a = (k + 9) * (a + b) + 2 :=
by
  sorry

end digits_interchanged_l133_133461


namespace geometric_series_sum_l133_133459

-- Definition of the geometric sum function in Lean
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * ((1 - r^n) / (1 - r))

-- Specific terms for the problem
def a : ℚ := 2
def r : ℚ := 2 / 5
def n : ℕ := 5

-- The target sum we aim to prove
def target_sum : ℚ := 10310 / 3125

-- The theorem stating that the calculated sum equals the target sum
theorem geometric_series_sum : geometric_sum a r n = target_sum :=
by sorry

end geometric_series_sum_l133_133459


namespace correct_statements_identification_l133_133909

-- Definitions based on given conditions
def syntheticMethodCauseToEffect := True
def syntheticMethodForward := True
def analyticMethodEffectToCause := True
def analyticMethodIndirect := False
def analyticMethodBackward := True

-- The main statement to be proved
theorem correct_statements_identification :
  (syntheticMethodCauseToEffect = True) ∧ 
  (syntheticMethodForward = True) ∧ 
  (analyticMethodEffectToCause = True) ∧ 
  (analyticMethodBackward = True) ∧ 
  (analyticMethodIndirect = False) :=
by
  sorry

end correct_statements_identification_l133_133909


namespace find_a2_geometric_sequence_l133_133335

theorem find_a2_geometric_sequence (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = a n * r) 
  (h_a1 : a 1 = 1 / 4) (h_eq : a 3 * a 5 = 4 * (a 4 - 1)) : a 2 = 1 / 8 :=
by
  sorry

end find_a2_geometric_sequence_l133_133335


namespace inequality_for_positive_reals_l133_133705

variable (a b c : ℝ)

theorem inequality_for_positive_reals (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := by
  sorry

end inequality_for_positive_reals_l133_133705


namespace table_price_l133_133517

theorem table_price (C T : ℝ) (h1 : 2 * C + T = 0.6 * (C + 2 * T)) (h2 : C + T = 96) : T = 84 := by
  sorry

end table_price_l133_133517


namespace permutations_sum_divisible_by_37_l133_133462

theorem permutations_sum_divisible_by_37 (a b c : ℕ) (ha : 0 ≤ a ∧ a ≤ 9)
  (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) :
    ∃ k, (100 * a + 10 * b + c) + (100 * a + 10 * c + b) + (100 * b + 10 * a + c) + (100 * b + 10 * c + a) + (100 * c + 10 * a + b) + (100 * c + 10 * b + a) = 37 * k := 
by
  sorry

end permutations_sum_divisible_by_37_l133_133462


namespace range_of_m_if_neg_proposition_false_l133_133674

theorem range_of_m_if_neg_proposition_false :
  (¬ ∃ x_0 : ℝ, x_0^2 + m * x_0 + 2 * m - 3 < 0) ↔ (2 ≤ m ∧ m ≤ 6) :=
by
  sorry

end range_of_m_if_neg_proposition_false_l133_133674


namespace part1_part2_l133_133054

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l133_133054


namespace jordan_has_11_oreos_l133_133610

-- Define the conditions
def jamesOreos (x : ℕ) : ℕ := 3 + 2 * x
def totalOreos (jordanOreos : ℕ) : ℕ := 36

-- Theorem stating the problem that Jordan has 11 Oreos given the conditions
theorem jordan_has_11_oreos (x : ℕ) (h1 : jamesOreos x + x = totalOreos x) : x = 11 :=
by
  sorry

end jordan_has_11_oreos_l133_133610


namespace solution_exists_l133_133958

def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

def f' (x a b : ℝ) : ℝ := 3 * x^2 - 2 * a * x - b

theorem solution_exists (a b : ℝ) :
    f 1 a b = 10 ∧ f' 1 a b = 0 ↔ (a = -4 ∧ b = 11) :=
by 
  sorry

end solution_exists_l133_133958


namespace harry_worked_total_hours_l133_133174

theorem harry_worked_total_hours (x : ℝ) (H : ℝ) (H_total : ℝ) :
  (24 * x + 1.5 * x * H = 42 * x) → (H_total = 24 + H) → H_total = 36 :=
by
sorry

end harry_worked_total_hours_l133_133174


namespace cat_weights_ratio_l133_133770

variable (meg_cat_weight : ℕ) (anne_extra_weight : ℕ) (meg_cat_weight := 20) (anne_extra_weight := 8)

/-- The ratio of the weight of Meg's cat to the weight of Anne's cat -/
theorem cat_weights_ratio : (meg_cat_weight / Nat.gcd meg_cat_weight (meg_cat_weight + anne_extra_weight)) 
                            = 5 ∧ ((meg_cat_weight + anne_extra_weight) / Nat.gcd meg_cat_weight (meg_cat_weight + anne_extra_weight)) 
                            = 7 := by
  sorry

end cat_weights_ratio_l133_133770


namespace train_passes_man_in_4_4_seconds_l133_133314

noncomputable def train_speed_kmph : ℝ := 84
noncomputable def man_speed_kmph : ℝ := 6
noncomputable def train_length_m : ℝ := 110

noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

noncomputable def train_speed_mps : ℝ :=
  kmph_to_mps train_speed_kmph

noncomputable def man_speed_mps : ℝ :=
  kmph_to_mps man_speed_kmph

noncomputable def relative_speed_mps : ℝ :=
  train_speed_mps + man_speed_mps

noncomputable def passing_time : ℝ :=
  train_length_m / relative_speed_mps

theorem train_passes_man_in_4_4_seconds :
  passing_time = 4.4 :=
by
  sorry -- Proof not required, skipping the proof logic

end train_passes_man_in_4_4_seconds_l133_133314


namespace victor_initial_books_l133_133390

theorem victor_initial_books (x : ℕ) : (x + 3 = 12) → (x = 9) :=
by
  sorry

end victor_initial_books_l133_133390


namespace find_LN_l133_133012

noncomputable def LM : ℝ := 9
noncomputable def sin_N : ℝ := 3 / 5
noncomputable def LN : ℝ := 15

theorem find_LN (h₁ : sin_N = 3 / 5) (h₂ : LM = 9) (h₃ : sin_N = LM / LN) : LN = 15 :=
by
  sorry

end find_LN_l133_133012


namespace soldiers_height_order_l133_133560

theorem soldiers_height_order {n : ℕ} (a b : Fin n → ℝ) 
  (ha : ∀ i j, i ≤ j → a i ≥ a j) 
  (hb : ∀ i j, i ≤ j → b i ≥ b j) 
  (h : ∀ i, a i ≤ b i) :
  ∀ i, a i ≤ b i :=
  by sorry

end soldiers_height_order_l133_133560


namespace boys_and_girls_l133_133159

theorem boys_and_girls (B G : ℕ) (h1 : B + G = 30)
  (h2 : ∀ (i j : ℕ), i < B → j < B → i ≠ j → ∃ k, k < G ∧ ∀ l < B, l ≠ i → k ≠ l)
  (h3 : ∀ (i j : ℕ), i < G → j < G → i ≠ j → ∃ k, k < B ∧ ∀ l < G, l ≠ i → k ≠ l) :
  B = 15 ∧ G = 15 :=
by
  have hB : B ≤ G := sorry
  have hG : G ≤ B := sorry
  exact ⟨by linarith, by linarith⟩

end boys_and_girls_l133_133159


namespace cone_lateral_area_l133_133873

theorem cone_lateral_area (r l : ℝ) (h_r : r = 3) (h_l : l = 5) : 
  π * r * l = 15 * π := by
  sorry

end cone_lateral_area_l133_133873


namespace remainder_2_pow_224_plus_104_l133_133726

theorem remainder_2_pow_224_plus_104 (x : ℕ) (h1 : x = 2 ^ 56) : 
  (2 ^ 224 + 104) % (2 ^ 112 + 2 ^ 56 + 1) = 103 := 
by
  sorry

end remainder_2_pow_224_plus_104_l133_133726


namespace obtain_half_not_obtain_one_l133_133720

theorem obtain_half (x : ℕ) : (10 + x) / (97 + x) = 1 / 2 ↔ x = 77 := 
by
  sorry

theorem not_obtain_one (x k : ℕ) : ¬ ((10 + x) / (97 + x) = 1 ∨ (10 * k) / (97 * k) = 1) := 
by
  sorry

end obtain_half_not_obtain_one_l133_133720


namespace incorrect_intersections_l133_133608

theorem incorrect_intersections :
  (∃ x, (x = x ∧ x = Real.sqrt (x + 2)) ↔ x = 1 ∨ x = 2) →
  (∃ x, (x^2 - 3 * x + 2 = 2 ∧ x = 2) ↔ x = 1 ∨ x = 2) →
  (∃ x, (Real.sin x = 3 * x - 4 ∧ x = 2) ↔ x = 1 ∨ x = 2) → False :=
by {
  sorry
}

end incorrect_intersections_l133_133608


namespace problem_ab_cd_l133_133776

theorem problem_ab_cd
    (a b c d : ℝ)
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
    (habcd : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h1 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2012)
    (h2 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2012) :
  (ab)^2012 - (cd)^2012 = -2012 := 
sorry

end problem_ab_cd_l133_133776


namespace fraction_of_shaded_area_is_11_by_12_l133_133942

noncomputable def shaded_fraction_of_square : ℚ :=
  let s : ℚ := 1 -- Assume the side length of the square is 1 for simplicity.
  let P := (0, s / 2)
  let Q := (s / 3, s)
  let V := (0, s)
  let base := s / 2
  let height := s / 3
  let triangle_area := (1 / 2) * base * height
  let square_area := s * s
  let shaded_area := square_area - triangle_area
  shaded_area / square_area

theorem fraction_of_shaded_area_is_11_by_12 : shaded_fraction_of_square = 11 / 12 :=
  sorry

end fraction_of_shaded_area_is_11_by_12_l133_133942


namespace measure_of_angle_x_in_triangle_l133_133699

theorem measure_of_angle_x_in_triangle
  (x : ℝ)
  (h1 : x + 2 * x + 45 = 180) :
  x = 45 :=
sorry

end measure_of_angle_x_in_triangle_l133_133699


namespace montoya_family_budget_on_food_l133_133492

def spending_on_groceries : ℝ := 0.6
def spending_on_eating_out : ℝ := 0.2

theorem montoya_family_budget_on_food :
  spending_on_groceries + spending_on_eating_out = 0.8 :=
  by
  sorry

end montoya_family_budget_on_food_l133_133492


namespace area_S3_l133_133380

theorem area_S3 {s1 s2 s3 : ℝ} (h1 : s1^2 = 25)
  (h2 : s2 = s1 / Real.sqrt 2)
  (h3 : s3 = s2 / Real.sqrt 2)
  : s3^2 = 6.25 :=
by
  sorry

end area_S3_l133_133380


namespace range_of_f_l133_133186

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem range_of_f : Set.Icc 0 3 → (Set.Ico 1 5) :=
by
  sorry
  -- Here the proof steps would go, which are omitted based on your guidelines.

end range_of_f_l133_133186


namespace initial_cost_of_article_correct_l133_133209

noncomputable def initial_cost_of_article (final_cost : ℝ) : ℝ :=
  final_cost / (0.75 * 0.85 * 1.10 * 1.05)

theorem initial_cost_of_article_correct (final_cost : ℝ) (h : final_cost = 1226.25) :
  initial_cost_of_article final_cost = 1843.75 :=
by
  rw [h]
  norm_num
  rw [initial_cost_of_article]
  simp [initial_cost_of_article]
  norm_num
  sorry

end initial_cost_of_article_correct_l133_133209


namespace max_volume_cuboid_l133_133820

theorem max_volume_cuboid (x y z : ℕ) (h : 2 * (x * y + x * z + y * z) = 150) : x * y * z ≤ 125 :=
sorry

end max_volume_cuboid_l133_133820


namespace percent_decrease_l133_133356

theorem percent_decrease (P S : ℝ) (h₀ : P = 100) (h₁ : S = 70) :
  ((P - S) / P) * 100 = 30 :=
by
  sorry

end percent_decrease_l133_133356


namespace ratio_shiny_igneous_to_total_l133_133965

-- Define the conditions
variable (S I SI : ℕ)
variable (SS : ℕ)
variable (h1 : I = S / 2)
variable (h2 : SI = 40)
variable (h3 : S + I = 180)
variable (h4 : SS = S / 5)

-- Statement to prove
theorem ratio_shiny_igneous_to_total (S I SI SS : ℕ) 
  (h1 : I = S / 2) 
  (h2 : SI = 40) 
  (h3 : S + I = 180) 
  (h4 : SS = S / 5) : 
  SI / I = 2 / 3 := 
sorry

end ratio_shiny_igneous_to_total_l133_133965


namespace sin_double_angle_l133_133602

variable (θ : ℝ)

-- Given condition: tan(θ) = -3/5
def tan_theta : Prop := Real.tan θ = -3/5

-- Target to prove: sin(2θ) = -15/17
theorem sin_double_angle : tan_theta θ → Real.sin (2*θ) = -15/17 :=
by
  sorry

end sin_double_angle_l133_133602


namespace smallest_number_is_27_l133_133250

theorem smallest_number_is_27 (a b c : ℕ) (h_mean : (a + b + c) / 3 = 30) (h_median : b = 28) (h_largest : c = b + 7) : a = 27 :=
by {
  sorry
}

end smallest_number_is_27_l133_133250


namespace hyperbola_distance_property_l133_133692

theorem hyperbola_distance_property (P : ℝ × ℝ)
  (hP_on_hyperbola : (P.1 ^ 2 / 16) - (P.2 ^ 2 / 9) = 1)
  (h_dist_15 : dist P (5, 0) = 15) :
  dist P (-5, 0) = 7 ∨ dist P (-5, 0) = 23 := 
sorry

end hyperbola_distance_property_l133_133692


namespace total_apples_l133_133189

def packs : ℕ := 2
def apples_per_pack : ℕ := 4

theorem total_apples : packs * apples_per_pack = 8 := by
  sorry

end total_apples_l133_133189


namespace jillian_max_apartment_size_l133_133895

theorem jillian_max_apartment_size :
  ∀ s : ℝ, (1.10 * s = 880) → s = 800 :=
by
  intros s h
  sorry

end jillian_max_apartment_size_l133_133895


namespace area_of_fourth_rectangle_l133_133133

variable (x y z w : ℝ)
variable (Area_EFGH Area_EIKJ Area_KLMN Perimeter : ℝ)

def conditions :=
  (Area_EFGH = x * y ∧ Area_EFGH = 20 ∧
   Area_EIKJ = x * w ∧ Area_EIKJ = 25 ∧
   Area_KLMN = z * w ∧ Area_KLMN = 15 ∧
   Perimeter = 2 * (x + z + y + w) ∧ Perimeter = 40)

theorem area_of_fourth_rectangle (h : conditions x y z w Area_EFGH Area_EIKJ Area_KLMN Perimeter) :
  (y * w = 340) :=
by
  sorry

end area_of_fourth_rectangle_l133_133133


namespace sally_purchased_20_fifty_cent_items_l133_133245

noncomputable def num_fifty_cent_items (x y z : ℕ) (h1 : x + y + z = 30) (h2 : 50 * x + 500 * y + 1000 * z = 10000) : ℕ :=
x

theorem sally_purchased_20_fifty_cent_items
  (x y z : ℕ)
  (h1 : x + y + z = 30)
  (h2 : 50 * x + 500 * y + 1000 * z = 10000)
  : num_fifty_cent_items x y z h1 h2 = 20 :=
sorry

end sally_purchased_20_fifty_cent_items_l133_133245


namespace areas_of_triangle_and_parallelogram_are_equal_l133_133978

theorem areas_of_triangle_and_parallelogram_are_equal (b : ℝ) :
  let parallelogram_height := 100
  let triangle_height := 200
  let area_parallelogram := b * parallelogram_height
  let area_triangle := (1/2) * b * triangle_height
  area_parallelogram = area_triangle :=
by
  -- conditions
  let parallelogram_height := 100
  let triangle_height := 200
  let area_parallelogram := b * parallelogram_height
  let area_triangle := (1 / 2) * b * triangle_height
  -- relationship
  show area_parallelogram = area_triangle
  sorry

end areas_of_triangle_and_parallelogram_are_equal_l133_133978


namespace greatest_divisor_l133_133864

theorem greatest_divisor (d : ℕ) :
  (690 % d = 10) ∧ (875 % d = 25) ∧ ∀ e : ℕ, (690 % e = 10) ∧ (875 % e = 25) → (e ≤ d) :=
  sorry

end greatest_divisor_l133_133864


namespace unique_solution_l133_133374

def system_of_equations (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ) (x1 x2 x3 : ℝ) :=
  a11 * x1 + a12 * x2 + a13 * x3 = 0 ∧
  a21 * x1 + a22 * x2 + a23 * x3 = 0 ∧
  a31 * x1 + a32 * x2 + a33 * x3 = 0

theorem unique_solution
  (x1 x2 x3 : ℝ)
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ)
  (h_pos: 0 < a11 ∧ 0 < a22 ∧ 0 < a33)
  (h_neg: a12 < 0 ∧ a13 < 0 ∧ a21 < 0 ∧ a23 < 0 ∧ a31 < 0 ∧ a32 < 0)
  (h_sum_pos: 0 < a11 + a12 + a13 ∧ 0 < a21 + a22 + a23 ∧ 0 < a31 + a32 + a33)
  (h_system: system_of_equations a11 a12 a13 a21 a22 a23 a31 a32 a33 x1 x2 x3):
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 := sorry

end unique_solution_l133_133374


namespace angle_C_measure_ratio_inequality_l133_133501

open Real

variables (A B C a b c : ℝ)

-- Assumptions
variable (ABC_is_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b)
variable (sin_condition : sin (2 * C - π / 2) = 1/2)
variable (inequality_condition : a^2 + b^2 < c^2)

theorem angle_C_measure :
  0 < C ∧ C < π ∧ C = 2 * π / 3 := sorry

theorem ratio_inequality :
  1 < (a + b) / c ∧ (a + b) / c ≤ 2 * sqrt 3 / 3 := sorry

end angle_C_measure_ratio_inequality_l133_133501


namespace alex_received_12_cookies_l133_133583

theorem alex_received_12_cookies :
  ∃ y: ℕ, (∀ s: ℕ, y = s + 8 ∧ s = y / 3) → y = 12 := by
  sorry

end alex_received_12_cookies_l133_133583


namespace smallest_multiplier_to_perfect_square_l133_133498

theorem smallest_multiplier_to_perfect_square : ∃ k : ℕ, k > 0 ∧ ∀ m : ℕ, (2010 * m = k * k) → m = 2010 :=
by
  sorry

end smallest_multiplier_to_perfect_square_l133_133498


namespace ratio_of_b_to_a_is_4_l133_133801

theorem ratio_of_b_to_a_is_4 (b a : ℚ) (h1 : b = 4 * a) (h2 : b = 15 - 4 * a) : a = 15 / 8 := by
  sorry

end ratio_of_b_to_a_is_4_l133_133801


namespace number_of_hens_l133_133411

-- Let H be the number of hens and C be the number of cows
def hens_and_cows (H C : Nat) : Prop :=
  H + C = 50 ∧ 2 * H + 4 * C = 144

theorem number_of_hens : ∃ H C : Nat, hens_and_cows H C ∧ H = 28 :=
by
  -- The proof is omitted
  sorry

end number_of_hens_l133_133411


namespace temperature_in_quebec_city_is_negative_8_l133_133649

def temperature_vancouver : ℝ := 22
def temperature_calgary (temperature_vancouver : ℝ) : ℝ := temperature_vancouver - 19
def temperature_quebec_city (temperature_calgary : ℝ) : ℝ := temperature_calgary - 11

theorem temperature_in_quebec_city_is_negative_8 :
  temperature_quebec_city (temperature_calgary temperature_vancouver) = -8 := by
  sorry

end temperature_in_quebec_city_is_negative_8_l133_133649


namespace actual_distance_is_correct_l133_133485

noncomputable def actual_distance_in_meters (scale : ℕ) (map_distance_cm : ℝ) : ℝ :=
  (map_distance_cm * scale) / 100

theorem actual_distance_is_correct
  (scale : ℕ)
  (map_distance_cm : ℝ)
  (h_scale : scale = 3000000)
  (h_map_distance : map_distance_cm = 4) :
  actual_distance_in_meters scale map_distance_cm = 1.2 * 10^5 :=
by
  sorry

end actual_distance_is_correct_l133_133485


namespace probability_red_or_green_is_two_thirds_l133_133362

-- Define the conditions
def total_balls := 2 + 3 + 4
def favorable_outcomes := 2 + 4

-- Define the probability calculation
def probability_red_or_green := (favorable_outcomes : ℚ) / total_balls

-- The theorem statement
theorem probability_red_or_green_is_two_thirds : probability_red_or_green = 2 / 3 := by
  -- This part will contain the proof using Lean, but we skip it with "sorry" for now.
  sorry

end probability_red_or_green_is_two_thirds_l133_133362


namespace bob_start_time_l133_133270

-- Define constants for the problem conditions
def yolandaRate : ℝ := 3 -- Yolanda's walking rate in miles per hour
def bobRate : ℝ := 4 -- Bob's walking rate in miles per hour
def distanceXY : ℝ := 10 -- Distance between point X and Y in miles
def bobDistanceWhenMet : ℝ := 4 -- Distance Bob had walked when they met in miles

-- Define the theorem statement
theorem bob_start_time : 
  ∃ T : ℝ, (yolandaRate * T + bobDistanceWhenMet = distanceXY) →
  (T = 2) →
  ∃ tB : ℝ, T - tB = 1 :=
by
  -- Insert proof here
  sorry

end bob_start_time_l133_133270


namespace friends_picked_strawberries_with_Lilibeth_l133_133224

-- Define the conditions
def Lilibeth_baskets : ℕ := 6
def strawberries_per_basket : ℕ := 50
def total_strawberries : ℕ := 1200

-- Define the calculation of strawberries picked by Lilibeth
def Lilibeth_strawberries : ℕ := Lilibeth_baskets * strawberries_per_basket

-- Define the calculation of strawberries picked by friends
def friends_strawberries : ℕ := total_strawberries - Lilibeth_strawberries

-- Define the number of friends who picked strawberries
def friends_picked_with_Lilibeth : ℕ := friends_strawberries / Lilibeth_strawberries

-- The theorem we need to prove
theorem friends_picked_strawberries_with_Lilibeth : friends_picked_with_Lilibeth = 3 :=
by
  -- Proof goes here
  sorry

end friends_picked_strawberries_with_Lilibeth_l133_133224


namespace elevator_translation_l133_133135

-- Definitions based on conditions
def turning_of_steering_wheel : Prop := False
def rotation_of_bicycle_wheels : Prop := False
def motion_of_pendulum : Prop := False
def movement_of_elevator : Prop := True

-- Theorem statement
theorem elevator_translation :
  movement_of_elevator := by
  exact True.intro

end elevator_translation_l133_133135


namespace man_speed_in_still_water_l133_133081

noncomputable def speed_of_man_in_still_water (vm vs : ℝ) : Prop :=
  -- Condition 1: v_m + v_s = 8
  vm + vs = 8 ∧
  -- Condition 2: v_m - v_s = 5
  vm - vs = 5

-- Proving the speed of the man in still water is 6.5 km/h
theorem man_speed_in_still_water : ∃ (v_m : ℝ), (∃ v_s : ℝ, speed_of_man_in_still_water v_m v_s) ∧ v_m = 6.5 :=
by
  sorry

end man_speed_in_still_water_l133_133081


namespace kath_total_cost_l133_133410

def admission_cost : ℝ := 8
def discount_percentage_pre6pm : ℝ := 0.25
def discount_percentage_student : ℝ := 0.10
def time_of_movie : ℝ := 4
def num_people : ℕ := 6
def num_students : ℕ := 2

theorem kath_total_cost :
  let discounted_price := admission_cost * (1 - discount_percentage_pre6pm)
  let student_price := discounted_price * (1 - discount_percentage_student)
  let num_non_students := num_people - num_students - 1 -- remaining people (total - 2 students - Kath)
  let kath_and_siblings_cost := 3 * discounted_price
  let student_friends_cost := num_students * student_price
  let non_student_friend_cost := num_non_students * discounted_price
  let total_cost := kath_and_siblings_cost + student_friends_cost + non_student_friend_cost
  total_cost = 34.80 := by
  let discounted_price := admission_cost * (1 - discount_percentage_pre6pm)
  let student_price := discounted_price * (1 - discount_percentage_student)
  let num_non_students := num_people - num_students - 1
  let kath_and_siblings_cost := 3 * discounted_price
  let student_friends_cost := num_students * student_price
  let non_student_friend_cost := num_non_students * discounted_price
  let total_cost := kath_and_siblings_cost + student_friends_cost + non_student_friend_cost
  sorry

end kath_total_cost_l133_133410


namespace max_value_under_constraint_l133_133892

noncomputable def max_value_expression (a b c : ℝ) : ℝ :=
3 * a * b - 3 * b * c + 2 * c^2

theorem max_value_under_constraint
  (a b c : ℝ)
  (h : a^2 + b^2 + c^2 = 1) :
  max_value_expression a b c ≤ 3 :=
sorry

end max_value_under_constraint_l133_133892


namespace gcd_9011_4403_l133_133283

theorem gcd_9011_4403 : Nat.gcd 9011 4403 = 1 := 
by sorry

end gcd_9011_4403_l133_133283


namespace total_pencils_l133_133110

-- Defining the number of pencils each person has.
def jessica_pencils : ℕ := 8
def sandy_pencils : ℕ := 8
def jason_pencils : ℕ := 8

-- Theorem stating the total number of pencils
theorem total_pencils : jessica_pencils + sandy_pencils + jason_pencils = 24 := by
  sorry

end total_pencils_l133_133110


namespace Rhett_rent_expense_l133_133164

-- Define the problem statement using given conditions
theorem Rhett_rent_expense
  (late_payments : ℕ := 2)
  (no_late_fees : Bool := true)
  (fraction_of_salary : ℝ := 3 / 5)
  (monthly_salary : ℝ := 5000)
  (tax_rate : ℝ := 0.1) :
  let salary_after_taxes := monthly_salary * (1 - tax_rate)
  let total_late_rent := fraction_of_salary * salary_after_taxes
  let monthly_rent_expense := total_late_rent / late_payments
  monthly_rent_expense = 1350 := by
  sorry

end Rhett_rent_expense_l133_133164


namespace estimated_value_of_n_l133_133207

-- Definitions from the conditions of the problem
def total_balls (n : ℕ) : ℕ := n + 18 + 9
def probability_of_yellow (n : ℕ) : ℚ := 18 / total_balls n

-- The theorem stating what we need to prove
theorem estimated_value_of_n : ∃ n : ℕ, probability_of_yellow n = 0.30 ∧ n = 42 :=
by {
  sorry
}

end estimated_value_of_n_l133_133207


namespace board_partition_possible_l133_133014

variable (m n : ℕ)

theorem board_partition_possible (hm : m > 15) (hn : n > 15) :
  ((∃ k1, m = 5 * k1 ∧ ∃ k2, n = 4 * k2) ∨ (∃ k3, m = 4 * k3 ∧ ∃ k4, n = 5 * k4)) :=
sorry

end board_partition_possible_l133_133014


namespace P_inequality_l133_133215

def P (n : ℕ) (x : ℝ) : ℝ := (Finset.range (n + 1)).sum (λ k => x^k)

theorem P_inequality (x : ℝ) (hx : 0 < x) :
  P 20 x * P 21 (x^2) ≤ P 20 (x^2) * P 22 x :=
by
  sorry

end P_inequality_l133_133215


namespace keith_bought_cards_l133_133131

theorem keith_bought_cards (orig : ℕ) (now : ℕ) (bought : ℕ) 
  (h1 : orig = 40) (h2 : now = 18) (h3 : bought = orig - now) : bought = 22 := by
  sorry

end keith_bought_cards_l133_133131


namespace parabola_directrix_l133_133181

theorem parabola_directrix (a : ℝ) (h : -1 / (4 * a) = 2) : a = -1 / 8 :=
by
  sorry

end parabola_directrix_l133_133181


namespace sub_two_three_l133_133833

theorem sub_two_three : 2 - 3 = -1 := 
by 
  sorry

end sub_two_three_l133_133833


namespace range_of_independent_variable_l133_133762

theorem range_of_independent_variable (x : ℝ) : (1 - x > 0) → x < 1 :=
by
  sorry

end range_of_independent_variable_l133_133762


namespace product_of_four_consecutive_odd_numbers_is_perfect_square_l133_133360

theorem product_of_four_consecutive_odd_numbers_is_perfect_square (n : ℤ) :
    (n + 0) * (n + 2) * (n + 4) * (n + 6) = 9 :=
sorry

end product_of_four_consecutive_odd_numbers_is_perfect_square_l133_133360


namespace A_lt_B_l133_133158

variable (x y : ℝ)

def A (x y : ℝ) : ℝ := - y^2 + 4 * x - 3
def B (x y : ℝ) : ℝ := x^2 + 2 * x + 2 * y

theorem A_lt_B (x y : ℝ) : A x y < B x y := 
by
  sorry

end A_lt_B_l133_133158


namespace range_of_a_l133_133318

noncomputable def f (x : ℝ) : ℝ := x + Real.log x
noncomputable def g (a x : ℝ) : ℝ := a * x - 2 * Real.sin x

theorem range_of_a (a : ℝ) :
  (∀ x₁ > 0, ∃ x₂, (1 + 1 / x₁) * (a - 2 * Real.cos x₂) = -1) →
  -2 ≤ a ∧ a ≤ 1 :=
by {
  sorry
}

end range_of_a_l133_133318


namespace line_equation_through_point_l133_133824

theorem line_equation_through_point 
  (x y : ℝ)
  (h1 : (5, 2) ∈ {p : ℝ × ℝ | p.2 = p.1 * (2 / 5)})
  (h2 : (5, 2) ∈ {p : ℝ × ℝ | p.1 / 6 + p.2 / 12 = 1}) 
  (h3 : (5,2) ∈ {p : ℝ × ℝ | 2 * p.1 = p.2 }) :
  (2 * x + y - 12 = 0 ∨ 
   2 * x - 5 * y = 0) := 
sorry

end line_equation_through_point_l133_133824


namespace five_level_pyramid_has_80_pieces_l133_133096

-- Definitions based on problem conditions
def rods_per_level (level : ℕ) : ℕ :=
  if level = 1 then 4
  else if level = 2 then 8
  else if level = 3 then 12
  else if level = 4 then 16
  else if level = 5 then 20
  else 0

def connectors_per_level_transition : ℕ := 4

-- The total rods used for a five-level pyramid
def total_rods_five_levels : ℕ :=
  rods_per_level 1 + rods_per_level 2 + rods_per_level 3 + rods_per_level 4 + rods_per_level 5

-- The total connectors used for a five-level pyramid
def total_connectors_five_levels : ℕ :=
  connectors_per_level_transition * 5

-- The total pieces required for a five-level pyramid
def total_pieces_five_levels : ℕ :=
  total_rods_five_levels + total_connectors_five_levels

-- Main theorem statement for the proof problem
theorem five_level_pyramid_has_80_pieces : 
  total_pieces_five_levels = 80 :=
by
  -- We expect the total_pieces_five_levels to be equal to 80
  sorry

end five_level_pyramid_has_80_pieces_l133_133096


namespace selection_ways_l133_133160

-- Define the problem parameters
def male_students : ℕ := 4
def female_students : ℕ := 3
def total_selected : ℕ := 3

-- Define the binomial coefficient function for combinatorial calculations
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define conditions
def both_genders_must_be_represented : Prop :=
  total_selected = 3 ∧ male_students >= 1 ∧ female_students >= 1

-- Problem statement: proof that the total ways to select 3 students is 30
theorem selection_ways : both_genders_must_be_represented → 
  (binomial male_students 2 * binomial female_students 1 +
   binomial male_students 1 * binomial female_students 2) = 30 :=
by
  sorry

end selection_ways_l133_133160


namespace most_probable_sellable_samples_l133_133029

/-- Prove that the most probable number k of sellable samples out of 24,
given each has a 0.6 probability of being sellable, is either 14 or 15. -/
theorem most_probable_sellable_samples (n : ℕ) (p : ℝ) (q : ℝ) (k₀ k₁ : ℕ) 
  (h₁ : n = 24) (h₂ : p = 0.6) (h₃ : q = 1 - p)
  (h₄ : 24 * p - q < k₀) (h₅ : k₀ < 24 * p + p) 
  (h₆ : k₀ = 14) (h₇ : k₁ = 15) :
  (k₀ = 14 ∨ k₀ = 15) :=
  sorry

end most_probable_sellable_samples_l133_133029


namespace workman_B_days_l133_133575

theorem workman_B_days (A B : ℝ) (hA : A = (1 / 2) * B) (hTogether : (A + B) * 14 = 1) :
  1 / B = 21 :=
sorry

end workman_B_days_l133_133575


namespace original_weight_of_beef_l133_133219

theorem original_weight_of_beef (weight_after_processing : ℝ) (loss_percentage : ℝ) :
  loss_percentage = 0.5 → weight_after_processing = 750 → 
  (750 : ℝ) / (1 - 0.5) = 1500 :=
by
  intros h_loss_percent h_weight_after
  sorry

end original_weight_of_beef_l133_133219


namespace fruit_basket_ratio_l133_133315

theorem fruit_basket_ratio (total_fruits : ℕ) (oranges : ℕ) (apples : ℕ) (h1 : total_fruits = 40) (h2 : oranges = 10) (h3 : apples = total_fruits - oranges) :
  (apples / oranges) = 3 := by
  sorry

end fruit_basket_ratio_l133_133315


namespace simplify_frac_l133_133385

variable (b c : ℕ)
variable (b_val : b = 2)
variable (c_val : c = 3)

theorem simplify_frac : (15 * b ^ 4 * c ^ 2) / (45 * b ^ 3 * c) = 2 :=
by
  rw [b_val, c_val]
  sorry

end simplify_frac_l133_133385


namespace base_b_representation_l133_133238

theorem base_b_representation (b : ℕ) (h₁ : 1 * b + 5 = n) (h₂ : n^2 = 4 * b^2 + 3 * b + 3) : b = 7 :=
by {
  sorry
}

end base_b_representation_l133_133238


namespace point_in_third_quadrant_l133_133234

theorem point_in_third_quadrant
  (a b : ℝ)
  (hne : a ≠ 0)
  (y_increase : ∀ x1 x2, x1 < x2 → -5 * a * x1 + b < -5 * a * x2 + b)
  (ab_pos : a * b > 0) : 
  a < 0 ∧ b < 0 :=
by
  sorry

end point_in_third_quadrant_l133_133234


namespace locus_of_centers_of_circles_l133_133899

structure Point (α : Type _) :=
(x : α)
(y : α)

noncomputable def perpendicular_bisector {α : Type _} [LinearOrderedField α] (A B : Point α) : Set (Point α) :=
  {C | ∃ m b : α, C.y = m * C.x + b ∧ A.y = m * A.x + b ∧ B.y = m * B.x + b ∧
                 (A.x - B.x) * C.x + (A.y - B.y) * C.y = (A.x^2 + A.y^2 - B.x^2 - B.y^2) / 2}

theorem locus_of_centers_of_circles {α : Type _} [LinearOrderedField α] (A B : Point α) :
  (∀ (C : Point α), (∃ r : α, r > 0 ∧ ∃ k: α, (C.x - A.x)^2 + (C.y - A.y)^2 = r^2 ∧ (C.x - B.x)^2 + (C.y - B.y)^2 = r^2) 
  → C ∈ perpendicular_bisector A B) :=
by
  sorry

end locus_of_centers_of_circles_l133_133899


namespace angela_problems_l133_133326

theorem angela_problems (M J S K A : ℕ) :
  M = 3 →
  J = (M * M - 5) + ((M * M - 5) / 3) →
  S = 50 / 10 →
  K = (J + S) / 2 →
  A = 50 - (M + J + S + K) →
  A = 32 :=
by
  intros hM hJ hS hK hA
  sorry

end angela_problems_l133_133326


namespace ellipse_closer_to_circle_l133_133403

variables (a : ℝ)

-- Conditions: 1 < a < 2 + sqrt 5
def in_range_a (a : ℝ) : Prop := 1 < a ∧ a < 2 + Real.sqrt 5

-- Ellipse eccentricity should decrease as 'a' increases for the given range 1 < a < 2 + sqrt 5
theorem ellipse_closer_to_circle (h_range : in_range_a a) :
    ∃ b : ℝ, b = Real.sqrt (1 - (a^2 - 1) / (4 * a)) ∧ ∀ a', (1 < a' ∧ a' < 2 + Real.sqrt 5 ∧ a < a') → b > Real.sqrt (1 - (a'^2 - 1) / (4 * a')) := 
sorry

end ellipse_closer_to_circle_l133_133403


namespace bird_families_flew_away_to_Africa_l133_133875

theorem bird_families_flew_away_to_Africa 
  (B : ℕ) (n : ℕ) (hB94 : B = 94) (hB_A_plus_n : B = n + 47) : n = 47 :=
by
  sorry

end bird_families_flew_away_to_Africa_l133_133875


namespace arithmetic_sequence_2023rd_term_l133_133309

theorem arithmetic_sequence_2023rd_term 
  (p q : ℤ)
  (h1 : 3 * p - q + 9 = 9)
  (h2 : 3 * (3 * p - q + 9) - q + 9 = 3 * p + q) :
  p + (2023 - 1) * (3 * p - q + 9) = 18189 := by
  sorry

end arithmetic_sequence_2023rd_term_l133_133309


namespace correct_operation_among_given_ones_l133_133172

theorem correct_operation_among_given_ones
  (a : ℝ) :
  (a^2)^3 = a^6 :=
by {
  sorry
}

-- Auxiliary lemmas if needed (based on conditions):
lemma mul_powers_add_exponents (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by sorry

lemma power_of_a_power (a : ℝ) (m n : ℕ) : (a^m)^n = a^(m * n) := by sorry

lemma div_powers_subtract_exponents (a : ℝ) (m n : ℕ) : a^m / a^n = a^(m - n) := by sorry

lemma square_of_product (x y : ℝ) : (x * y)^2 = x^2 * y^2 := by sorry

end correct_operation_among_given_ones_l133_133172


namespace rhombus_area_is_160_l133_133984

-- Define the values of the diagonals
def d1 : ℝ := 16
def d2 : ℝ := 20

-- Define the formula for the area of the rhombus
noncomputable def area_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

-- State the theorem to be proved
theorem rhombus_area_is_160 :
  area_rhombus d1 d2 = 160 :=
by
  sorry

end rhombus_area_is_160_l133_133984


namespace animal_legs_count_l133_133906

-- Let's define the conditions first.
def total_animals : ℕ := 12
def chickens : ℕ := 5
def chicken_legs : ℕ := 2
def sheep_legs : ℕ := 4

-- Define the statement that we need to prove.
theorem animal_legs_count :
  ∃ (total_legs : ℕ), total_legs = 38 :=
by
  -- Adding the condition for total number of legs
  let sheep := total_animals - chickens
  let total_legs := (chickens * chicken_legs) + (sheep * sheep_legs)
  existsi total_legs
  -- Question proves the correct answer
  sorry

end animal_legs_count_l133_133906


namespace estimate_yellow_balls_l133_133737

theorem estimate_yellow_balls (m : ℕ) (h1: (5 : ℝ) / (5 + m) = 0.2) : m = 20 :=
  sorry

end estimate_yellow_balls_l133_133737


namespace sum_of_remainders_l133_133446

theorem sum_of_remainders (a b c d e : ℕ)
  (h1 : a % 13 = 3)
  (h2 : b % 13 = 5)
  (h3 : c % 13 = 7)
  (h4 : d % 13 = 9)
  (h5 : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 :=
by {
  sorry
}

end sum_of_remainders_l133_133446


namespace fill_bucket_time_l133_133859

theorem fill_bucket_time (time_full_bucket : ℕ) (fraction : ℚ) (time_two_thirds_bucket : ℕ) 
  (h1 : time_full_bucket = 150) (h2 : fraction = 2 / 3) : time_two_thirds_bucket = 100 :=
sorry

end fill_bucket_time_l133_133859


namespace systematic_sampling_removal_count_l133_133475

theorem systematic_sampling_removal_count :
  ∀ (N n : ℕ), N = 3204 ∧ n = 80 → N % n = 4 := 
by
  sorry

end systematic_sampling_removal_count_l133_133475


namespace average_age_constant_l133_133275

theorem average_age_constant 
  (average_age_3_years_ago : ℕ) 
  (number_of_members_3_years_ago : ℕ) 
  (baby_age_today : ℕ) 
  (number_of_members_today : ℕ) 
  (H1 : average_age_3_years_ago = 17) 
  (H2 : number_of_members_3_years_ago = 5) 
  (H3 : baby_age_today = 2) 
  (H4 : number_of_members_today = 6) : 
  average_age_3_years_ago = (average_age_3_years_ago * number_of_members_3_years_ago + baby_age_today + 3 * number_of_members_3_years_ago) / number_of_members_today := 
by sorry

end average_age_constant_l133_133275


namespace product_of_ninth_and_tenth_l133_133325

def scores_first_8 := [7, 4, 3, 6, 8, 3, 1, 5]
def total_points_first_8 := scores_first_8.sum

def condition1 (ninth_game_points tenth_game_points : ℕ) : Prop :=
  ninth_game_points < 10 ∧ tenth_game_points < 10

def condition2 (ninth_game_points : ℕ) : Prop :=
  (total_points_first_8 + ninth_game_points) % 9 = 0

def condition3 (ninth_game_points tenth_game_points : ℕ) : Prop :=
  (total_points_first_8 + ninth_game_points + tenth_game_points) % 10 = 0

theorem product_of_ninth_and_tenth (ninth_game_points : ℕ) (tenth_game_points : ℕ) 
  (h1 : condition1 ninth_game_points tenth_game_points)
  (h2 : condition2 ninth_game_points)
  (h3 : condition3 ninth_game_points tenth_game_points) : 
  ninth_game_points * tenth_game_points = 40 :=
sorry

end product_of_ninth_and_tenth_l133_133325


namespace value_expression_at_5_l133_133670

theorem value_expression_at_5 (x : ℕ) (hx : x = 5) : 2 * x^2 + 4 = 54 :=
by
  -- Adding sorry to skip the proof.
  sorry

end value_expression_at_5_l133_133670


namespace cos_double_angle_l133_133443

theorem cos_double_angle {α : ℝ} (h1 : 0 < α ∧ α < 2 * Real.pi ∧ α > 3 * Real.pi / 2) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.cos (2 * α) = Real.sqrt 5 / 3 := 
by
  sorry

end cos_double_angle_l133_133443


namespace triangle_angle_sum_l133_133837

theorem triangle_angle_sum (a b : ℝ) (ha : a = 40) (hb : b = 60) : ∃ x : ℝ, x = 180 - (a + b) :=
by
  use 80
  sorry

end triangle_angle_sum_l133_133837


namespace system_infinite_solutions_a_eq_neg2_l133_133541

theorem system_infinite_solutions_a_eq_neg2 
  (x y a : ℝ)
  (h1 : 2 * x + 2 * y = -1)
  (h2 : 4 * x + a^2 * y = a) 
  (infinitely_many_solutions : ∃ (a : ℝ), ∀ (c : ℝ), 4 * x + a^2 * y = c) :
  a = -2 :=
by
  sorry

end system_infinite_solutions_a_eq_neg2_l133_133541


namespace problem_l133_133007

theorem problem (m : ℝ) (h : m + 1/m = 6) : m^2 + 1/m^2 + 3 = 37 :=
by
  sorry

end problem_l133_133007


namespace prove_monotonic_increasing_range_l133_133698

open Real

noncomputable def problem_statement : Prop :=
  ∃ a : ℝ, (0 < a ∧ a < 1) ∧ (∀ x > 0, (a^x + (1 + a)^x) ≤ (a^(x+1) + (1 + a)^(x+1))) ∧
  (a ≥ (sqrt 5 - 1) / 2 ∧ a < 1)

theorem prove_monotonic_increasing_range : problem_statement := sorry

end prove_monotonic_increasing_range_l133_133698


namespace min_time_to_cross_river_l133_133527

-- Definitions for the time it takes each horse to cross the river
def timeA : ℕ := 2
def timeB : ℕ := 3
def timeC : ℕ := 7
def timeD : ℕ := 6

-- Definition for the minimum time required for all horses to cross the river
def min_crossing_time : ℕ := 18

-- Theorem stating the problem: 
theorem min_time_to_cross_river :
  ∀ (timeA timeB timeC timeD : ℕ), timeA = 2 → timeB = 3 → timeC = 7 → timeD = 6 →
  min_crossing_time = 18 :=
sorry

end min_time_to_cross_river_l133_133527


namespace total_age_difference_l133_133254

noncomputable def ages_difference (A B C : ℕ) : ℕ :=
  (A + B) - (B + C)

theorem total_age_difference (A B C : ℕ) (h₁ : A + B > B + C) (h₂ : C = A - 11) : ages_difference A B C = 11 :=
by
  sorry

end total_age_difference_l133_133254


namespace find_m_l133_133939

variable {a b c m : ℝ}

theorem find_m (h1 : a + b = 4)
               (h2 : a * b = m)
               (h3 : b + c = 8)
               (h4 : b * c = 5 * m) : m = 0 ∨ m = 3 :=
by {
  sorry
}

end find_m_l133_133939


namespace division_of_composite_products_l133_133367

noncomputable def product_of_first_seven_composites : ℕ :=
  4 * 6 * 8 * 9 * 10 * 12 * 14

noncomputable def product_of_next_seven_composites : ℕ :=
  15 * 16 * 18 * 20 * 21 * 22 * 24

noncomputable def divided_product_composites : ℚ :=
  product_of_first_seven_composites / product_of_next_seven_composites

theorem division_of_composite_products : divided_product_composites = 1 / 176 := by
  sorry

end division_of_composite_products_l133_133367


namespace second_assistant_smoked_pipes_l133_133092

theorem second_assistant_smoked_pipes
    (x y z : ℚ)
    (H1 : (2 / 3) * x = (4 / 9) * y)
    (H2 : x + y = 1)
    (H3 : (x + z) / (y - z) = y / x) :
    z = 1 / 5 → x = 2 / 5 ∧ y = 3 / 5 →
    ∀ n : ℕ, n = 5 :=
by
  sorry

end second_assistant_smoked_pipes_l133_133092


namespace wait_at_least_15_seconds_probability_l133_133087

-- Define the duration of the red light
def red_light_duration : ℕ := 40

-- Define the minimum waiting time for the green light
def min_wait_time : ℕ := 15

-- Define the duration after which pedestrian does not need to wait 15 seconds
def max_arrival_time : ℕ := red_light_duration - min_wait_time

-- Lean statement to prove the required probability
theorem wait_at_least_15_seconds_probability :
  (max_arrival_time : ℝ) / red_light_duration = 5 / 8 :=
by
  -- Proof omitted with sorry
  sorry

end wait_at_least_15_seconds_probability_l133_133087


namespace percentage_decrease_of_larger_angle_l133_133336

noncomputable def complementary_angles_decrease_percentage : Real :=
let total_degrees := 90
let ratio_sum := 3 + 7
let part := total_degrees / ratio_sum
let smaller_angle := 3 * part
let larger_angle := 7 * part
let increased_smaller_angle := smaller_angle * 1.2
let new_larger_angle := total_degrees - increased_smaller_angle
let decrease_amount := larger_angle - new_larger_angle
(decrease_amount / larger_angle) * 100

theorem percentage_decrease_of_larger_angle
  (smaller_increased_percentage : Real := 20)
  (ratio_three : Real := 3)
  (ratio_seven : Real := 7)
  (total_degrees : Real := 90)
  (expected_decrease : Real := 8.57):
  complementary_angles_decrease_percentage = expected_decrease := 
sorry

end percentage_decrease_of_larger_angle_l133_133336


namespace negation_of_universal_statement_l133_133585

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ^ 2 ≠ x) ↔ ∃ x : ℝ, x ^ 2 = x :=
by
  sorry

end negation_of_universal_statement_l133_133585


namespace area_difference_depends_only_on_bw_l133_133745

variable (b w n : ℕ)
variable (hb : b ≥ 2)
variable (hw : w ≥ 2)
variable (hn : n = b + w)

/-- Given conditions: 
1. \(b \geq 2\) 
2. \(w \geq 2\) 
3. \(n = b + w\)
4. There are \(2b\) identical black rods and \(2w\) identical white rods, each of side length 1. 
5. These rods form a regular \(2n\)-gon with parallel sides of the same color.
6. A convex \(2b\)-gon \(B\) is formed by translating the black rods. 
7. A convex \(2w\) A convex \(2w\)-gon \(W\) is formed by translating the white rods. 
Prove that the difference of the areas of \(B\) and \(W\) depends only on the numbers \(b\) and \(w\). -/
theorem area_difference_depends_only_on_bw :
  ∀ (A B W : ℝ), A - B = 2 * (b - w) :=
sorry

end area_difference_depends_only_on_bw_l133_133745


namespace sum_of_integers_with_even_product_l133_133750

theorem sum_of_integers_with_even_product (a b : ℤ) (h : ∃ k, a * b = 2 * k) : 
∃ k1 k2, a = 2 * k1 ∨ a = 2 * k1 + 1 ∧ (a + b = 2 * k2 ∨ a + b = 2 * k2 + 1) :=
by
  sorry

end sum_of_integers_with_even_product_l133_133750


namespace gold_distribution_l133_133168

theorem gold_distribution :
  ∃ (d : ℚ), 
    (4 * (a1: ℚ) + 6 * d = 3) ∧ 
    (3 * (a1: ℚ) + 24 * d = 4) ∧
    d = 7 / 78 :=
by {
  sorry
}

end gold_distribution_l133_133168


namespace no_positive_sequence_exists_l133_133868

theorem no_positive_sequence_exists:
  ¬ (∃ (b : ℕ → ℝ), (∀ n, b n > 0) ∧ (∀ m : ℕ, (∑' k, b ((k + 1) * m)) = (1 / m))) :=
by
  sorry

end no_positive_sequence_exists_l133_133868


namespace intersection_three_points_l133_133947

def circle_eq (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = a^2
def parabola_eq (a : ℝ) (x y : ℝ) : Prop := y = x^2 - 3 * a

theorem intersection_three_points (a : ℝ) :
  (∃ x1 y1 x2 y2 x3 y3 : ℝ, 
    circle_eq a x1 y1 ∧ parabola_eq a x1 y1 ∧
    circle_eq a x2 y2 ∧ parabola_eq a x2 y2 ∧
    circle_eq a x3 y3 ∧ parabola_eq a x3 y3 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧
    (x2 ≠ x3 ∨ y2 ≠ y3)) ↔ 
  a = 1/3 := by
  sorry

end intersection_three_points_l133_133947


namespace systematic_sampling_example_l133_133798

theorem systematic_sampling_example : 
  ∃ (a : ℕ → ℕ), (∀ i : ℕ, 5 ≤ i ∧ i ≤ 5 → a i = 5 + 10 * (i - 1)) ∧ 
  ∀ i : ℕ, 1 ≤ i ∧ i < 6 → a i - a (i - 1) = a (i + 1) - a i :=
sorry

end systematic_sampling_example_l133_133798


namespace lucas_initial_pet_beds_l133_133368

-- Definitions from the problem conditions
def additional_beds := 8
def beds_per_pet := 2
def pets := 10

-- Statement to prove
theorem lucas_initial_pet_beds :
  (pets * beds_per_pet) - additional_beds = 12 := 
by
  sorry

end lucas_initial_pet_beds_l133_133368


namespace profit_450_l133_133074

-- Define the conditions
def cost_per_garment : ℕ := 40
def wholesale_price : ℕ := 60

-- Define the piecewise function for wholesale price P
noncomputable def P (x : ℕ) : ℕ :=
  if h : 0 < x ∧ x ≤ 100 then wholesale_price
  else if h : 100 < x ∧ x ≤ 500 then 62 - x / 50
  else 0

-- Define the profit function L
noncomputable def L (x : ℕ) : ℕ :=
  if h : 0 < x ∧ x ≤ 100 then (P x - cost_per_garment) * x
  else if h : 100 < x ∧ x ≤ 500 then (22 * x - x^2 / 50)
  else 0

-- State the theorem
theorem profit_450 : L 450 = 5850 :=
by
  sorry

end profit_450_l133_133074


namespace initial_puppies_l133_133332

-- Definitions based on the conditions in the problem
def sold : ℕ := 21
def puppies_per_cage : ℕ := 9
def number_of_cages : ℕ := 9

-- The statement to prove
theorem initial_puppies : sold + (puppies_per_cage * number_of_cages) = 102 := by
  sorry

end initial_puppies_l133_133332


namespace sum_even_then_diff_even_sum_odd_then_diff_odd_l133_133011

theorem sum_even_then_diff_even (a b : ℤ) (h : (a + b) % 2 = 0) : (a - b) % 2 = 0 := by
  sorry

theorem sum_odd_then_diff_odd (a b : ℤ) (h : (a + b) % 2 = 1) : (a - b) % 2 = 1 := by
  sorry

end sum_even_then_diff_even_sum_odd_then_diff_odd_l133_133011


namespace profit_is_correct_l133_133724

-- Define the constants for expenses
def cost_of_lemons : ℕ := 10
def cost_of_sugar : ℕ := 5
def cost_of_cups : ℕ := 3

-- Define the cost per cup of lemonade
def price_per_cup : ℕ := 4

-- Define the number of cups sold
def cups_sold : ℕ := 21

-- Define the total revenue
def total_revenue : ℕ := cups_sold * price_per_cup

-- Define the total expenses
def total_expenses : ℕ := cost_of_lemons + cost_of_sugar + cost_of_cups

-- Define the profit
def profit : ℕ := total_revenue - total_expenses

-- The theorem stating the profit
theorem profit_is_correct : profit = 66 := by
  sorry

end profit_is_correct_l133_133724


namespace find_cows_l133_133060

theorem find_cows :
  ∃ (D C : ℕ), (2 * D + 4 * C = 2 * (D + C) + 30) → C = 15 := 
sorry

end find_cows_l133_133060


namespace find_set_T_l133_133885

namespace MathProof 

theorem find_set_T (S : Finset ℕ) (hS : ∀ x ∈ S, x > 0) :
  ∃ T : Finset ℕ, S ⊆ T ∧ ∀ x ∈ T, x ∣ (T.sum id) :=
by
  sorry

end MathProof 

end find_set_T_l133_133885


namespace nine_digit_positive_integers_l133_133945

theorem nine_digit_positive_integers :
  (∃ n : Nat, 10^8 * 9 = n ∧ n = 900000000) :=
sorry

end nine_digit_positive_integers_l133_133945


namespace warriors_wins_count_l133_133382

variable {wins : ℕ → ℕ}
variable (raptors hawks warriors spurs lakers : ℕ)

def conditions (wins : ℕ → ℕ) (raptors hawks warriors spurs lakers : ℕ) : Prop :=
  wins raptors > wins hawks ∧
  wins warriors > wins spurs ∧ wins warriors < wins lakers ∧
  wins spurs > 25

theorem warriors_wins_count
  (wins : ℕ → ℕ)
  (raptors hawks warriors spurs lakers : ℕ)
  (h : conditions wins raptors hawks warriors spurs lakers) :
  wins warriors = 37 := sorry

end warriors_wins_count_l133_133382


namespace inequality_holds_for_all_l133_133342

theorem inequality_holds_for_all (m : ℝ) 
  (h : ∀ x : ℝ, (x^2 - 8 * x + 20) / (m * x^2 - m * x - 1) < 0) : -4 < m ∧ m ≤ 0 := 
sorry

end inequality_holds_for_all_l133_133342


namespace minimum_surface_area_of_cube_l133_133845

noncomputable def brick_length := 25
noncomputable def brick_width := 15
noncomputable def brick_height := 5
noncomputable def side_length := Nat.lcm brick_width brick_length
noncomputable def surface_area := 6 * side_length * side_length

theorem minimum_surface_area_of_cube : surface_area = 33750 := 
by
  sorry

end minimum_surface_area_of_cube_l133_133845


namespace sum_of_n_and_k_l133_133349

theorem sum_of_n_and_k (n k : ℕ) 
  (h1 : (n.choose k) * 3 = (n.choose (k + 1)))
  (h2 : (n.choose (k + 1)) * 2 = (n.choose (k + 2))) :
  n + k = 13 :=
by
  sorry

end sum_of_n_and_k_l133_133349


namespace ratio_of_ian_to_jessica_l133_133601

/-- 
Rodney has 35 dollars more than Ian. 
Jessica has 100 dollars. 
Jessica has 15 dollars more than Rodney. 
Prove that the ratio of Ian's money to Jessica's money is 1/2.
-/
theorem ratio_of_ian_to_jessica (I R J : ℕ) (h1 : R = I + 35) (h2 : J = 100) (h3 : J = R + 15) :
  I / J = 1 / 2 :=
by
  sorry

end ratio_of_ian_to_jessica_l133_133601


namespace total_animal_sightings_l133_133725

def A_Jan := 26
def A_Feb := 3 * A_Jan
def A_Mar := A_Feb / 2

theorem total_animal_sightings : A_Jan + A_Feb + A_Mar = 143 := by
  sorry

end total_animal_sightings_l133_133725


namespace find_x_eq_3_l133_133962

noncomputable def f (x : ℝ) : ℝ := 4 * x - 9
noncomputable def f_inv (x : ℝ) : ℝ := (x + 9) / 4

theorem find_x_eq_3 : ∃ x : ℝ, f x = f_inv x ∧ x = 3 :=
by
  sorry

end find_x_eq_3_l133_133962


namespace milk_left_after_third_operation_l133_133719

theorem milk_left_after_third_operation :
  ∀ (initial_milk : ℝ), initial_milk > 0 →
  (initial_milk * 0.8 * 0.8 * 0.8 / initial_milk) * 100 = 51.2 :=
by
  intros initial_milk h_initial_milk_pos
  sorry

end milk_left_after_third_operation_l133_133719


namespace calculate_expression_l133_133395

variables (a b : ℝ) -- declaring variables a and b to be real numbers

theorem calculate_expression :
  (-a * b^2) ^ 3 + (a * b^2) * (a * b) ^ 2 * (-2 * b) ^ 2 = 3 * a^3 * b^6 :=
by
  sorry

end calculate_expression_l133_133395


namespace total_tablets_l133_133417

-- Variables for the numbers of Lenovo, Samsung, and Huawei tablets
variables (n x y : ℕ)

-- Conditions based on problem statement
def condition1 : Prop := 2 * x + 6 + y < n / 3

def condition2 : Prop := (n - 2 * x - y - 6 = 3 * y)

def condition3 : Prop := (n - 6 * x - y - 6 = 59)

-- The statement to prove that the total number of tablets is 94
theorem total_tablets (h1 : condition1 n x y) (h2 : condition2 n x y) (h3 : condition3 n x y) : n = 94 :=
by
  sorry

end total_tablets_l133_133417


namespace kevin_hopped_distance_after_four_hops_l133_133987

noncomputable def kevin_total_hopped_distance : ℚ :=
  let hop1 := 1
  let hop2 := 1 / 2
  let hop3 := 1 / 4
  let hop4 := 1 / 8
  hop1 + hop2 + hop3 + hop4

theorem kevin_hopped_distance_after_four_hops :
  kevin_total_hopped_distance = 15 / 8 :=
by
  sorry

end kevin_hopped_distance_after_four_hops_l133_133987


namespace marbles_left_calculation_l133_133751

/-- A magician starts with 20 red marbles and 30 blue marbles.
    He removes 3 red marbles and 12 blue marbles. We need to 
    prove that he has 35 marbles left in total. -/
theorem marbles_left_calculation (initial_red : ℕ) (initial_blue : ℕ) (removed_red : ℕ) 
    (removed_blue : ℕ) (H1 : initial_red = 20) (H2 : initial_blue = 30) 
    (H3 : removed_red = 3) (H4 : removed_blue = 4 * removed_red) :
    (initial_red - removed_red) + (initial_blue - removed_blue) = 35 :=
by
   -- sorry to skip the proof
   sorry

end marbles_left_calculation_l133_133751


namespace cart_distance_traveled_l133_133233

-- Define the problem parameters/conditions
def circumference_front : ℕ := 30
def circumference_back : ℕ := 33
def revolutions_difference : ℕ := 5

-- Define the question and the expected correct answer
theorem cart_distance_traveled :
  ∀ (R : ℕ), ((R + revolutions_difference) * circumference_front = R * circumference_back) → (R * circumference_back) = 1650 :=
by
  intro R h
  sorry

end cart_distance_traveled_l133_133233


namespace cube_root_of_8_l133_133334

theorem cube_root_of_8 : (∃ x : ℝ, x * x * x = 8) ∧ (∃ y : ℝ, y * y * y = 8 → y = 2) :=
by
  sorry

end cube_root_of_8_l133_133334


namespace decimal_to_fraction_l133_133948

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end decimal_to_fraction_l133_133948


namespace compute_expression_l133_133458

theorem compute_expression (p q : ℝ) (h1 : p + q = 6) (h2 : p * q = 10) : 
  p^3 + p^4 * q^2 + p^2 * q^4 + p * q^3 + p^5 * q^3 = 38676 := by
  -- Proof goes here
  sorry

end compute_expression_l133_133458


namespace funct_eq_x_l133_133247

theorem funct_eq_x (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x^4 + 4 * y^4) = f (x^2)^2 + 4 * y^3 * f y) : ∀ x : ℝ, f x = x := 
by 
  sorry

end funct_eq_x_l133_133247


namespace problem_solution_l133_133880

theorem problem_solution :
  3 * 995 + 4 * 996 + 5 * 997 + 6 * 998 + 7 * 999 - 4985 * 3 = 9980 := 
  by
  sorry

end problem_solution_l133_133880


namespace ice_cream_sandwiches_each_l133_133365

theorem ice_cream_sandwiches_each (total_ice_cream_sandwiches : ℕ) (number_of_nieces : ℕ) 
  (h1 : total_ice_cream_sandwiches = 143) (h2 : number_of_nieces = 11) : 
  total_ice_cream_sandwiches / number_of_nieces = 13 :=
by
  sorry

end ice_cream_sandwiches_each_l133_133365


namespace terminal_side_of_minus_330_in_first_quadrant_l133_133393

def angle_quadrant (angle : ℤ) : ℕ :=
  let reduced_angle := ((angle % 360) + 360) % 360
  if reduced_angle < 90 then 1
  else if reduced_angle < 180 then 2
  else if reduced_angle < 270 then 3
  else 4

theorem terminal_side_of_minus_330_in_first_quadrant :
  angle_quadrant (-330) = 1 :=
by
  -- We need a proof to justify the theorem, so we leave it with 'sorry' as instructed.
  sorry

end terminal_side_of_minus_330_in_first_quadrant_l133_133393


namespace find_angle_l133_133066

theorem find_angle (a b c d e : ℝ) (sum_of_hexagon_angles : ℝ) (h_sum : a = 135 ∧ b = 120 ∧ c = 105 ∧ d = 150 ∧ e = 110 ∧ sum_of_hexagon_angles = 720) : 
  ∃ P : ℝ, a + b + c + d + e + P = sum_of_hexagon_angles ∧ P = 100 :=
by
  sorry

end find_angle_l133_133066


namespace company_needs_86_workers_l133_133861

def profit_condition (n : ℕ) : Prop :=
  147 * n > 600 + 140 * n

theorem company_needs_86_workers (n : ℕ) : profit_condition n → n ≥ 86 :=
by
  intro h
  sorry

end company_needs_86_workers_l133_133861


namespace smallest_int_cond_l133_133678

theorem smallest_int_cond (b : ℕ) :
  (b % 9 = 5) ∧ (b % 11 = 7) → b = 95 :=
by
  intro h
  sorry

end smallest_int_cond_l133_133678


namespace f_3_2_plus_f_5_1_l133_133655

def f (a b : ℤ) : ℚ :=
  if a - b ≤ 2 then (a * b - a - 1) / (3 * a)
  else (a * b + b - 1) / (-3 * b)

theorem f_3_2_plus_f_5_1 :
  f 3 2 + f 5 1 = -13 / 9 :=
by
  sorry

end f_3_2_plus_f_5_1_l133_133655


namespace solve_positive_integer_x_l133_133825

theorem solve_positive_integer_x : ∃ (x : ℕ), 4 * x^2 - 16 * x - 60 = 0 ∧ x = 6 :=
by
  sorry

end solve_positive_integer_x_l133_133825


namespace carpenters_time_l133_133658

theorem carpenters_time (t1 t2 t3 t4 : ℝ) (ht1 : t1 = 1) (ht2 : t2 = 2)
  (ht3 : t3 = 3) (ht4 : t4 = 4) : (1 / (1 / t1 + 1 / t2 + 1 / t3 + 1 / t4)) = 12 / 25 := by
  sorry

end carpenters_time_l133_133658


namespace find_correct_result_l133_133657

noncomputable def correct_result : Prop :=
  ∃ (x : ℝ), (-1.25 * x - 0.25 = 1.25 * x) ∧ (-1.25 * x = 0.125)

theorem find_correct_result : correct_result :=
  sorry

end find_correct_result_l133_133657


namespace sum_of_given_numbers_l133_133813

theorem sum_of_given_numbers : 30 + 80000 + 700 + 60 = 80790 :=
  by
    sorry

end sum_of_given_numbers_l133_133813


namespace abs_value_equation_l133_133950

-- Define the main proof problem
theorem abs_value_equation (a b c d : ℝ)
  (h : ∀ x : ℝ, |2 * x + 4| + |a * x + b| = |c * x + d|) :
  d = 2 * c :=
sorry -- Proof skipped for this exercise

end abs_value_equation_l133_133950


namespace smallest_four_digit_mod_8_l133_133489

theorem smallest_four_digit_mod_8 :
  ∃ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 8 = 5 ∧ (∀ y : ℕ, y >= 1000 ∧ y < 10000 ∧ y % 8 = 5 → x ≤ y) :=
sorry

end smallest_four_digit_mod_8_l133_133489


namespace robert_coin_arrangement_l133_133816

noncomputable def num_arrangements (gold : ℕ) (silver : ℕ) : ℕ :=
  if gold + silver = 8 ∧ gold = 5 ∧ silver = 3 then 504 else 0

theorem robert_coin_arrangement :
  num_arrangements 5 3 = 504 := 
sorry

end robert_coin_arrangement_l133_133816


namespace tangent_line_at_origin_l133_133520

-- Define the function f(x) = x^3 + ax with an extremum at x = 1
def f (x a : ℝ) : ℝ := x^3 + a * x

-- Define the condition for a local extremum at x = 1: f'(1) = 0
def extremum_condition (a : ℝ) : Prop := (3 * 1^2 + a = 0)

-- Define the derivative of f at x = 0
def derivative_at_origin (a : ℝ) : ℝ := 3 * 0^2 + a

-- Define the value of function at x = 0
def value_at_origin (a : ℝ) : ℝ := f 0 a

-- The main theorem to prove
theorem tangent_line_at_origin (a : ℝ) (ha : extremum_condition a) :
    (value_at_origin a = 0) ∧ (derivative_at_origin a = -3) → ∀ x, (3 * x + (f x a - f 0 a) / (x - 0) = 0) := by
  sorry

end tangent_line_at_origin_l133_133520


namespace trig_identity_l133_133926

theorem trig_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (Real.sin θ + Real.cos θ) / Real.sin θ + Real.sin θ * Real.sin θ = 23 / 10 :=
sorry

end trig_identity_l133_133926


namespace angle_in_third_quadrant_l133_133694

theorem angle_in_third_quadrant
  (α : ℝ)
  (k : ℤ)
  (h : (π / 2) + 2 * (↑k) * π < α ∧ α < π + 2 * (↑k) * π) :
  π + 2 * (↑k) * π < (π / 2) + α ∧ (π / 2) + α < (3 * π / 2) + 2 * (↑k) * π :=
by
  sorry

end angle_in_third_quadrant_l133_133694


namespace union_A_B_subset_B_A_l133_133101

-- Condition definitions
def A : Set ℝ := {x | 2 * x - 8 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * (m + 1) * x + m^2 = 0}

-- Problem 1: If m = 4, prove A ∪ B = {2, 4, 8}
theorem union_A_B (m : ℝ) (h : m = 4) : A ∪ B m = {2, 4, 8} :=
sorry

-- Problem 2: If B ⊆ A, find the range for m
theorem subset_B_A (m : ℝ) (h : B m ⊆ A) : 
  m = 4 + 2 * Real.sqrt 2 ∨ m = 4 - 2 * Real.sqrt 2 ∨ m < -1 / 2 :=
sorry

end union_A_B_subset_B_A_l133_133101


namespace rods_in_one_mile_l133_133981

/-- Definitions based on given conditions -/
def miles_to_furlongs := 8
def furlongs_to_rods := 40

/-- The theorem stating the number of rods in one mile -/
theorem rods_in_one_mile : (miles_to_furlongs * furlongs_to_rods) = 320 := 
  sorry

end rods_in_one_mile_l133_133981


namespace monotonically_increasing_interval_l133_133628

noncomputable def f (x : ℝ) : ℝ := Real.log (-3 * x^2 + 4 * x + 4)

theorem monotonically_increasing_interval :
  ∀ x, x ∈ Set.Ioc (-2/3 : ℝ) (2/3 : ℝ) → MonotoneOn f (Set.Ioc (-2/3) (2/3)) :=
sorry

end monotonically_increasing_interval_l133_133628


namespace amount_of_H2O_formed_l133_133702

-- Define the balanced chemical equation as a relation
def balanced_equation : Prop :=
  ∀ (naoh hcl nacl h2o : ℕ), 
    (naoh + hcl = nacl + h2o)

-- Define the reaction of 2 moles of NaOH and 2 moles of HCl
def reaction (naoh hcl : ℕ) : ℕ :=
  if (naoh = 2) ∧ (hcl = 2) then 2 else 0

theorem amount_of_H2O_formed :
  balanced_equation →
  reaction 2 2 = 2 :=
by
  sorry

end amount_of_H2O_formed_l133_133702


namespace factorize_expr1_factorize_expr2_l133_133777

-- Define the expressions
def expr1 (m x y : ℝ) : ℝ := 3 * m * x - 6 * m * y
def expr2 (x : ℝ) : ℝ := 1 - 25 * x^2

-- Define the factorized forms
def factorized_expr1 (m x y : ℝ) : ℝ := 3 * m * (x - 2 * y)
def factorized_expr2 (x : ℝ) : ℝ := (1 + 5 * x) * (1 - 5 * x)

-- Proof problems
theorem factorize_expr1 (m x y : ℝ) : expr1 m x y = factorized_expr1 m x y := sorry
theorem factorize_expr2 (x : ℝ) : expr2 x = factorized_expr2 x := sorry

end factorize_expr1_factorize_expr2_l133_133777


namespace problem_a_proof_l133_133423

variables {A B C D M K : Point}
variables {triangle_ABC : Triangle A B C}
variables {incircle : Circle} (ht : touches incircle AC D) 
variables (hdm : diameter incircle D M) 
variables (bm_line : Line B M) (intersect_bm_ac : intersects bm_line AC K)

theorem problem_a_proof : 
  AK = DC :=
sorry

end problem_a_proof_l133_133423


namespace cube_truncation_edges_l133_133781

-- Define the initial condition: a cube
def initial_cube_edges : ℕ := 12

-- Define the condition of each corner being cut off
def corners_cut (corners : ℕ) (edges_added : ℕ) : ℕ :=
  corners * edges_added

-- Define the proof problem
theorem cube_truncation_edges : initial_cube_edges + corners_cut 8 3 = 36 := by
  sorry

end cube_truncation_edges_l133_133781


namespace fraction_identity_l133_133690

theorem fraction_identity (N F : ℝ) (hN : N = 8) (h : 0.5 * N = F * N + 2) : F = 1 / 4 :=
by {
  -- proof will go here
  sorry
}

end fraction_identity_l133_133690


namespace exists_sequences_l133_133404

theorem exists_sequences (m n : Nat → Nat) (h₁ : ∀ k, m k = 2 * k) (h₂ : ∀ k, n k = 5 * k * k)
  (h₃ : ∀ (i j : Nat), (i ≠ j) → (m i ≠ m j) ∧ (n i ≠ n j)) :
  (∀ k, Nat.sqrt (n k + (m k) * (m k)) = 3 * k) ∧
  (∀ k, Nat.sqrt (n k - (m k) * (m k)) = k) :=
by 
  sorry

end exists_sequences_l133_133404


namespace quadratic_root_m_eq_neg_fourteen_l133_133662

theorem quadratic_root_m_eq_neg_fourteen : ∀ (m : ℝ), (∃ x : ℝ, x = 2 ∧ x^2 + 5 * x + m = 0) → m = -14 :=
by
  sorry

end quadratic_root_m_eq_neg_fourteen_l133_133662


namespace janet_roses_l133_133644

def total_flowers (used_flowers extra_flowers : Nat) : Nat :=
  used_flowers + extra_flowers

def number_of_roses (total tulips : Nat) : Nat :=
  total - tulips

theorem janet_roses :
  ∀ (used_flowers extra_flowers tulips : Nat),
  used_flowers = 11 → extra_flowers = 4 → tulips = 4 →
  number_of_roses (total_flowers used_flowers extra_flowers) tulips = 11 :=
by
  intros used_flowers extra_flowers tulips h_used h_extra h_tulips
  rw [h_used, h_extra, h_tulips]
  -- proof steps skipped
  sorry

end janet_roses_l133_133644


namespace probability_of_winning_l133_133808

open Nat

theorem probability_of_winning (h : True) : 
  let num_cards := 3
  let num_books := 5
  (1 - (Nat.choose num_cards 2 * 2^num_books - num_cards) / num_cards^num_books) = 50 / 81 := sorry

end probability_of_winning_l133_133808


namespace gear_p_revolutions_per_minute_l133_133535

theorem gear_p_revolutions_per_minute (r : ℝ) 
  (cond2 : ℝ := 40) 
  (cond3 : 1.5 * r + 45 = 1.5 * 40) :
  r = 10 :=
by
  sorry

end gear_p_revolutions_per_minute_l133_133535


namespace inequality_with_sum_of_one_l133_133072

theorem inequality_with_sum_of_one
  (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_sum: a + b + c + d = 1) :
  (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) >= 1 / 2) :=
sorry

end inequality_with_sum_of_one_l133_133072


namespace general_formula_sum_first_n_terms_l133_133495

-- Definitions for arithmetic sequence, geometric aspects and sum conditions 
variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {d : ℕ}
variable {b_n : ℕ → ℕ}
variable {T_n : ℕ → ℕ}

-- Given conditions
axiom sum_condition (S3 S5 : ℕ) : S3 + S5 = 50
axiom common_difference : d ≠ 0
axiom first_term (a1 : ℕ) : a_n 1 = a1
axiom geometric_conditions (a1 a4 a13 : ℕ)
  (h1 : a_n 1 = a1) (h4 : a_n 4 = a4) (h13 : a_n 13 = a13) :
  a4 = a1 + 3 * d ∧ a13 = a1 + 12 * d ∧ (a4 ^ 2 = a1 * a13)

-- Proving the general formula for a_n
theorem general_formula (a_n : ℕ → ℕ)
  (h : ∀ (n : ℕ), a_n n = 2 * n + 1) : 
  a_n n = 2 * n + 1 := 
sorry

-- Proving the sum of the first n terms of sequence {b_n}
theorem sum_first_n_terms (a_n b_n : ℕ → ℕ) (T_n : ℕ → ℕ)
  (h_bn : ∀ (n : ℕ), b_n n = (2 * n + 1) * 2 ^ (n - 1))
  (h_Tn: ∀ (n : ℕ), T_n n = 1 + (2 * n - 1) * 2^n) :
  T_n n = 1 + (2 * n - 1) * 2^n :=
sorry

end general_formula_sum_first_n_terms_l133_133495


namespace find_pairs_satisfying_conditions_l133_133515

theorem find_pairs_satisfying_conditions :
  ∀ (m n : ℕ), (0 < m ∧ 0 < n) →
               (∃ k : ℤ, m^2 - 4 * n = k^2) →
               (∃ l : ℤ, n^2 - 4 * m = l^2) →
               (m = 4 ∧ n = 4) ∨ (m = 5 ∧ n = 6) ∨ (m = 6 ∧ n = 5) :=
by
  intros m n hmn h1 h2
  sorry

end find_pairs_satisfying_conditions_l133_133515
