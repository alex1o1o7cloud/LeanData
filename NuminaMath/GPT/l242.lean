import Mathlib

namespace smaller_angle_at_10_oclock_l242_242202

def degreeMeasureSmallerAngleAt10 := 
  let totalDegrees := 360
  let numHours := 12
  let degreesPerHour := totalDegrees / numHours
  let hourHandPosition := 10
  let minuteHandPosition := 12
  let divisionsBetween := if hourHandPosition < minuteHandPosition then minuteHandPosition - hourHandPosition else hourHandPosition - minuteHandPosition
  degreesPerHour * divisionsBetween

theorem smaller_angle_at_10_oclock : degreeMeasureSmallerAngleAt10 = 60 :=
  by 
    let totalDegrees := 360
    let numHours := 12
    let degreesPerHour := totalDegrees / numHours
    have h1 : degreesPerHour = 30 := by norm_num
    let hourHandPosition := 10
    let minuteHandPosition := 12
    let divisionsBetween := minuteHandPosition - hourHandPosition
    have h2 : divisionsBetween = 2 := by norm_num
    show 30 * divisionsBetween = 60
    calc 
      30 * 2 = 60 := by norm_num

end smaller_angle_at_10_oclock_l242_242202


namespace bullet_speed_difference_l242_242655

def speed_horse : ℕ := 20  -- feet per second
def speed_bullet : ℕ := 400  -- feet per second

def speed_forward : ℕ := speed_bullet + speed_horse
def speed_backward : ℕ := speed_bullet - speed_horse

theorem bullet_speed_difference : speed_forward - speed_backward = 40 :=
by
  sorry

end bullet_speed_difference_l242_242655


namespace artist_paint_usage_l242_242130

def ounces_of_paint_used (extra_large: ℕ) (large: ℕ) (medium: ℕ) (small: ℕ) : ℕ :=
  4 * extra_large + 3 * large + 2 * medium + 1 * small

theorem artist_paint_usage : ounces_of_paint_used 3 5 6 8 = 47 := by
  sorry

end artist_paint_usage_l242_242130


namespace volume_of_right_square_prism_l242_242538

theorem volume_of_right_square_prism (length width : ℕ) (H1 : length = 12) (H2 : width = 8) :
    ∃ V, (V = 72 ∨ V = 48) :=
by
  sorry

end volume_of_right_square_prism_l242_242538


namespace current_value_l242_242071

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l242_242071


namespace ratio_of_triangle_areas_l242_242804

noncomputable def area (a b c : ℕ) : ℚ := if a * a + b * b = c * c then (a * b : ℚ) / 2 else 0

theorem ratio_of_triangle_areas :
  let PQR := (7, 24, 25)
  let STU := (9, 40, 41)
  area PQR.1 PQR.2 PQR.3 / area STU.1 STU.2 STU.3 = (7 / 15 : ℚ) :=
by
  sorry

end ratio_of_triangle_areas_l242_242804


namespace battery_current_at_given_resistance_l242_242097

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l242_242097


namespace current_when_resistance_12_l242_242100

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l242_242100


namespace quadratic_root_value_l242_242576

theorem quadratic_root_value
  (a : ℝ) 
  (h : a^2 + 3 * a - 1010 = 0) :
  2 * a^2 + 6 * a + 4 = 2024 :=
by
  sorry

end quadratic_root_value_l242_242576


namespace probability_correct_l242_242329

variable (new_balls old_balls total_balls : ℕ)

-- Define initial conditions
def initial_conditions (new_balls old_balls : ℕ) : Prop :=
  new_balls = 4 ∧ old_balls = 2

-- Define total number of balls in the box
def total_balls_condition (new_balls old_balls total_balls : ℕ) : Prop :=
  total_balls = new_balls + old_balls ∧ total_balls = 6

-- Define the combination function
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability of picking one new ball and one old ball
def probability_one_new_one_old (new_balls old_balls total_balls : ℕ) : ℚ :=
  (combination new_balls 1 * combination old_balls 1) / (combination total_balls 2)

-- The theorem to prove the probability
theorem probability_correct (new_balls old_balls total_balls : ℕ)
  (h_initial : initial_conditions new_balls old_balls)
  (h_total : total_balls_condition new_balls old_balls total_balls) :
  probability_one_new_one_old new_balls old_balls total_balls = 8 / 15 := by
  sorry

end probability_correct_l242_242329


namespace binom_60_3_eq_34220_l242_242912

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l242_242912


namespace intersection_M_N_l242_242744

def M : Set ℝ := { x | Real.exp (x - 1) > 1 }
def N : Set ℝ := { x | x^2 - 2*x - 3 < 0 }

theorem intersection_M_N :
  (M ∩ N : Set ℝ) = { x | 1 < x ∧ x < 3 } := 
by
  sorry

end intersection_M_N_l242_242744


namespace train_crossing_time_l242_242171

noncomputable def train_length : ℝ := 385
noncomputable def train_speed_kmph : ℝ := 90
noncomputable def bridge_length : ℝ := 1250

noncomputable def convert_speed_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

noncomputable def time_to_cross_bridge (train_length bridge_length train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_mps := convert_speed_to_mps train_speed_kmph
  total_distance / speed_mps

theorem train_crossing_time :
  time_to_cross_bridge train_length bridge_length train_speed_kmph = 65.4 :=
by
  sorry

end train_crossing_time_l242_242171


namespace gamma_max_success_ratio_l242_242453

theorem gamma_max_success_ratio :
  ∀ (x y z w : ℕ),
    x > 0 → z > 0 →
    (5 * x < 3 * y) →
    (5 * z < 3 * w) →
    (y + w = 600) →
    (x + z ≤ 359) :=
by
  intros x y z w hx hz hxy hzw hyw
  sorry

end gamma_max_success_ratio_l242_242453


namespace current_value_l242_242066

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l242_242066


namespace sets_are_equal_l242_242345

def setA : Set ℤ := {x | ∃ a b : ℤ, x = 12 * a + 8 * b}
def setB : Set ℤ := {y | ∃ c d : ℤ, y = 20 * c + 16 * d}

theorem sets_are_equal : setA = setB := 
by
  sorry

end sets_are_equal_l242_242345


namespace greatest_integer_value_l242_242934

theorem greatest_integer_value (x : ℤ) : 3 * |x - 2| + 9 ≤ 24 → x ≤ 7 :=
by sorry

end greatest_integer_value_l242_242934


namespace sequence_3001_values_l242_242137

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then x else
  if n = 2 then 3000 else
  (sequence x (n - 1) + 1) / sequence x (n - 2)

theorem sequence_3001_values : 
  ∃ n : ℕ, ∀ x : ℝ, (∃ n, sequence x n = 3001) → 
    x = 3001 ∨ x = 1 ∨ x = 3001 / 9002999 ∨ x = 9002999 :=
sorry

end sequence_3001_values_l242_242137


namespace company_employees_count_l242_242230

theorem company_employees_count :
  ∃ E : ℕ, E = 80 + 100 - 30 + 20 := 
sorry

end company_employees_count_l242_242230


namespace binom_30_3_l242_242699

def binomial_coefficient (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binom_30_3 : binomial_coefficient 30 3 = 4060 := by
  sorry

end binom_30_3_l242_242699


namespace binom_60_3_l242_242900

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l242_242900


namespace true_propositions_l242_242277

-- Definitions for the propositions
def proposition1 (a b : ℝ) : Prop := a > b → a^2 > b^2
def proposition2 (a b : ℝ) : Prop := a^2 > b^2 → |a| > |b|
def proposition3 (a b c : ℝ) : Prop := (a > b ↔ a + c > b + c)

-- Theorem to state the true propositions
theorem true_propositions (a b c : ℝ) :
  -- Proposition 3 is true
  (proposition3 a b c) →
  -- Assert that the serial number of the true propositions is 3
  {3} = { i | (i = 1 ∧ proposition1 a b) ∨ (i = 2 ∧ proposition2 a b) ∨ (i = 3 ∧ proposition3 a b c)} :=
by
  sorry

end true_propositions_l242_242277


namespace total_revenue_correct_l242_242211

-- Define the costs of different types of returns
def cost_federal : ℕ := 50
def cost_state : ℕ := 30
def cost_quarterly : ℕ := 80

-- Define the quantities sold for different types of returns
def qty_federal : ℕ := 60
def qty_state : ℕ := 20
def qty_quarterly : ℕ := 10

-- Calculate the total revenue for the day
def total_revenue : ℕ := (cost_federal * qty_federal) + (cost_state * qty_state) + (cost_quarterly * qty_quarterly)

-- The theorem stating the total revenue calculation
theorem total_revenue_correct : total_revenue = 4400 := by
  sorry

end total_revenue_correct_l242_242211


namespace binomial_coefficient_30_3_l242_242708

theorem binomial_coefficient_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_coefficient_30_3_l242_242708


namespace max_value_a_l242_242163

theorem max_value_a (a b c d : ℝ) 
  (h1 : a ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h2 : b ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h3 : c ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h4 : d ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h5 : Real.sin a + Real.sin b + Real.sin c + Real.sin d = 1)
  (h6 : Real.cos (2 * a) + Real.cos (2 * b) + Real.cos (2 * c) + Real.cos (2 * d) ≥ 10 / 3) : 
  a ≤ Real.arcsin (1 / 2) := 
sorry

end max_value_a_l242_242163


namespace current_value_l242_242015

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l242_242015


namespace remaining_days_to_complete_job_l242_242386

-- Define the given conditions
def in_10_days (part_of_job_done : ℝ) (days : ℕ) : Prop :=
  part_of_job_done = 1 / 8 ∧ days = 10

-- Define the complete job condition
def complete_job (total_days : ℕ) : Prop :=
  total_days = 80

-- Define the remaining days to finish the job
def remaining_days (total_days_worked : ℕ) (days_worked : ℕ) (remaining : ℕ) : Prop :=
  total_days_worked = 80 ∧ days_worked = 10 ∧ remaining = 70

-- The theorem statement
theorem remaining_days_to_complete_job (part_of_job_done : ℝ) (days : ℕ) (total_days : ℕ) (total_days_worked : ℕ) (days_worked : ℕ) (remaining : ℕ) :
  in_10_days part_of_job_done days → complete_job total_days → remaining_days total_days_worked days_worked remaining :=
sorry

end remaining_days_to_complete_job_l242_242386


namespace current_value_l242_242087

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l242_242087


namespace original_loaf_slices_l242_242714

-- Define the given conditions
def andy_slices_1 := 3
def andy_slices_2 := 3
def toast_slices_per_piece := 2
def pieces_of_toast := 10
def slices_left_over := 1

-- Define the variables
def total_andy_slices := andy_slices_1 + andy_slices_2
def total_toast_slices := toast_slices_per_piece * pieces_of_toast

-- State the theorem
theorem original_loaf_slices : 
  ∃ S : ℕ, S = total_andy_slices + total_toast_slices + slices_left_over := 
by {
  sorry
}

end original_loaf_slices_l242_242714


namespace binom_30_3_eq_4060_l242_242692

theorem binom_30_3_eq_4060 : nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l242_242692


namespace expected_value_of_win_is_3_5_l242_242860

noncomputable def expected_value_win : ℝ :=
  (1/8) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

theorem expected_value_of_win_is_3_5 :
  expected_value_win = 3.5 :=
by
  sorry

end expected_value_of_win_is_3_5_l242_242860


namespace wade_final_profit_l242_242235

theorem wade_final_profit :
  let tips_per_customer_friday := 2.00
  let customers_friday := 28
  let tips_per_customer_saturday := 2.50
  let customers_saturday := 3 * customers_friday
  let tips_per_customer_sunday := 1.50
  let customers_sunday := 36
  let cost_ingredients_per_hotdog := 1.25
  let price_per_hotdog := 4.00
  let truck_maintenance_daily_cost := 50.00
  let total_taxes := 150.00
  let revenue_tips_friday := tips_per_customer_friday * customers_friday
  let revenue_hotdogs_friday := customers_friday * price_per_hotdog
  let cost_ingredients_friday := customers_friday * cost_ingredients_per_hotdog
  let revenue_friday := revenue_tips_friday + revenue_hotdogs_friday
  let total_costs_friday := cost_ingredients_friday + truck_maintenance_daily_cost
  let profit_friday := revenue_friday - total_costs_friday
  let revenue_tips_saturday := tips_per_customer_saturday * customers_saturday
  let revenue_hotdogs_saturday := customers_saturday * price_per_hotdog
  let cost_ingredients_saturday := customers_saturday * cost_ingredients_per_hotdog
  let revenue_saturday := revenue_tips_saturday + revenue_hotdogs_saturday
  let total_costs_saturday := cost_ingredients_saturday + truck_maintenance_daily_cost
  let profit_saturday := revenue_saturday - total_costs_saturday
  let revenue_tips_sunday := tips_per_customer_sunday * customers_sunday
  let revenue_hotdogs_sunday := customers_sunday * price_per_hotdog
  let cost_ingredients_sunday := customers_sunday * cost_ingredients_per_hotdog
  let revenue_sunday := revenue_tips_sunday + revenue_hotdogs_sunday
  let total_costs_sunday := cost_ingredients_sunday + truck_maintenance_daily_cost
  let profit_sunday := revenue_sunday - total_costs_sunday
  let total_profit := profit_friday + profit_saturday + profit_sunday
  let final_profit := total_profit - total_taxes
  final_profit = 427.00 :=
by
  sorry

end wade_final_profit_l242_242235


namespace tom_gas_spending_l242_242372

-- Defining the conditions given in the problem
def miles_per_gallon := 50
def miles_per_day := 75
def gas_price := 3
def number_of_days := 10

-- Defining the main theorem to be proven
theorem tom_gas_spending : 
  (miles_per_day * number_of_days) / miles_per_gallon * gas_price = 45 := 
by 
  sorry

end tom_gas_spending_l242_242372


namespace prob_X_leq_2_l242_242946

-- Define a discrete random variable X with given distributions
def X_distribution (i : ℕ) : ℚ :=
  if i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 then 1 / 4 else 0

theorem prob_X_leq_2 :
  (X_distribution 1 + X_distribution 2 = 1 / 2) :=
by
  unfold X_distribution
  simp
  norm_num
  sorry

end prob_X_leq_2_l242_242946


namespace total_spent_l242_242411

def price_almond_croissant : ℝ := 4.50
def price_salami_cheese_croissant : ℝ := 4.50
def price_plain_croissant : ℝ := 3.00
def price_focaccia : ℝ := 4.00
def price_latte : ℝ := 2.50
def num_lattes : ℕ := 2

theorem total_spent :
  price_almond_croissant + price_salami_cheese_croissant + price_plain_croissant +
  price_focaccia + (num_lattes * price_latte) = 21.00 := by
  sorry

end total_spent_l242_242411


namespace alexis_sew_skirt_time_l242_242682

theorem alexis_sew_skirt_time : 
  ∀ (S : ℝ), 
  (∀ (C : ℝ), C = 7) → 
  (6 * S + 4 * 7 = 40) → 
  S = 2 := 
by
  intros S _ h
  sorry

end alexis_sew_skirt_time_l242_242682


namespace battery_current_l242_242112

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l242_242112


namespace minimum_value_l242_242812

variable {x : ℝ}

theorem minimum_value (x : ℝ) : ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
by
  sorry

end minimum_value_l242_242812


namespace total_blocks_needed_l242_242528

theorem total_blocks_needed (length height : ℕ) (block_height : ℕ) (block1_length block2_length : ℕ)
                            (height_blocks : height = 8) (length_blocks : length = 102)
                            (block_height_cond : block_height = 1)
                            (block_lengths : block1_length = 2 ∧ block2_length = 1)
                            (staggered_cond : True) (even_ends : True) :
  ∃ total_blocks, total_blocks = 416 := 
  sorry

end total_blocks_needed_l242_242528


namespace expected_value_is_350_l242_242853

noncomputable def expected_value_of_winnings : ℚ :=
  ((1 / 8) * (8 - 1) + (1 / 8) * (8 - 2) + (1 / 8) * (8 - 3) + (1 / 8) * (8 - 4) +
  (1 / 8) * (8 - 5) + (1 / 8) * (8 - 6) + (1 / 8) * (8 - 7) + (1 / 8) * (8 - 8))

theorem expected_value_is_350 :
  expected_value_of_winnings = 3.50 := by
  sorry

end expected_value_is_350_l242_242853


namespace oranges_in_bin_after_changes_l242_242400

def initial_oranges := 31
def thrown_away_oranges := 9
def new_oranges := 38

theorem oranges_in_bin_after_changes : 
  initial_oranges - thrown_away_oranges + new_oranges = 60 := by
  sorry

end oranges_in_bin_after_changes_l242_242400


namespace expected_win_l242_242866

-- Definitions of conditions
def sides := fin 8 -- Finite type representing the 8 sides of the die

-- Function to calculate the win amount given a roll
def win_amount (n : sides) : ℝ := 8 - n.val

-- Probability of each side for a fair die
def probability : ℝ := 1 / 8

-- Definition of expected value calculation
def expected_value : ℝ := ∑ n in (finset.univ : finset sides), probability * (win_amount n)

-- Theorem statement
theorem expected_win : expected_value = 3.5 :=
by sorry

end expected_win_l242_242866


namespace current_value_l242_242073

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l242_242073


namespace find_n_l242_242434

-- Define the vectors \overrightarrow {AB}, \overrightarrow {BC}, and \overrightarrow {AC}
def vectorAB : ℝ × ℝ := (2, 4)
def vectorBC (n : ℝ) : ℝ × ℝ := (-2, 2 * n)
def vectorAC : ℝ × ℝ := (0, 2)

-- State the theorem and prove the value of n
theorem find_n (n : ℝ) (h : vectorAC = (vectorAB.1 + (vectorBC n).1, vectorAB.2 + (vectorBC n).2)) : n = -1 :=
by
  sorry

end find_n_l242_242434


namespace arithmetic_sequence_condition_l242_242333

theorem arithmetic_sequence_condition (a : ℕ → ℕ) 
(h1 : a 4 = 4) 
(h2 : a 3 + a 8 = 5) : 
a 7 = 1 := 
sorry

end arithmetic_sequence_condition_l242_242333


namespace volume_solid_correct_l242_242793

noncomputable def volume_of_solid : ℝ := 
  let area_rhombus := 1250 -- Area of the rhombus calculated from the bounded region
  let height := 10 -- Given height of the solid
  area_rhombus * height -- Volume of the solid

theorem volume_solid_correct (height: ℝ := 10) :
  volume_of_solid = 12500 := by
  sorry

end volume_solid_correct_l242_242793


namespace distinct_real_roots_l242_242732

theorem distinct_real_roots (a : ℝ) :
  let A := a - 3 in
  let B := -4 in
  let C := -1 in
  (a ≠ 3 ∧ 4 * a + 4 > 0) ↔ (a > -1 ∧ a ≠ 3) :=
by simp [A, B, C]; rink sorry

end distinct_real_roots_l242_242732


namespace additional_time_due_to_leak_is_six_l242_242531

open Real

noncomputable def filling_time_with_leak (R L : ℝ) : ℝ := 1 / (R - L)
noncomputable def filling_time_without_leak (R : ℝ) : ℝ := 1 / R
noncomputable def additional_filling_time (R L : ℝ) : ℝ :=
  filling_time_with_leak R L - filling_time_without_leak R

theorem additional_time_due_to_leak_is_six :
  additional_filling_time 0.25 (3 / 20) = 6 := by
  sorry

end additional_time_due_to_leak_is_six_l242_242531


namespace binomial_30_3_l242_242691

theorem binomial_30_3 : nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l242_242691


namespace binom_60_3_eq_34220_l242_242909

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l242_242909


namespace area_of_circle_l242_242486
open Real

-- Define the circumference condition
def circumference (r : ℝ) : ℝ :=
  2 * π * r

-- Define the area formula
def area (r : ℝ) : ℝ :=
  π * r * r

-- The given radius derived from the circumference
def radius_given_circumference (C : ℝ) : ℝ :=
  C / (2 * π)

-- The target proof statement
theorem area_of_circle (C : ℝ) (h : C = 36) : (area (radius_given_circumference C)) = 324 / π :=
by
  sorry

end area_of_circle_l242_242486


namespace cannot_be_arithmetic_progression_can_be_geometric_progression_l242_242686

theorem cannot_be_arithmetic_progression (a b c : ℝ) (ha : a = 2) (hb : b = real.sqrt 6) (hc : c = 4.5) : 
  ¬ ∃ d : ℝ, b = a + d ∧ c = a + 2 * d := by
  sorry

theorem can_be_geometric_progression (a b c : ℝ) (ha : a = 2) (hb : b = real.sqrt 6) (hc : c = 4.5) : 
  ∃ q : ℝ, q = real.sqrt (3 / 2) ∧ b = a * q ∧ c = a * q^2 := by
  sorry

end cannot_be_arithmetic_progression_can_be_geometric_progression_l242_242686


namespace right_triangle_hypotenuse_l242_242801

-- Define the right triangle conditions and hypotenuse calculation
theorem right_triangle_hypotenuse (a b c : ℝ) (h1 : b = a + 3) (h2 : 1 / 2 * a * b = 120) :
  c^2 = 425 :=
by
  sorry

end right_triangle_hypotenuse_l242_242801


namespace twentieth_term_is_78_l242_242710

-- Define the arithmetic sequence parameters
def first_term : ℤ := 2
def common_difference : ℤ := 4

-- Define the function to compute the n-th term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ := first_term + (n - 1) * common_difference

-- Formulate the theorem to prove
theorem twentieth_term_is_78 : nth_term 20 = 78 :=
by
  sorry

end twentieth_term_is_78_l242_242710


namespace largest_angle_of_triangle_l242_242363

theorem largest_angle_of_triangle
  (a b y : ℝ)
  (h1 : a = 60)
  (h2 : b = 70)
  (h3 : a + b + y = 180) :
  max a (max b y) = b :=
by
  sorry

end largest_angle_of_triangle_l242_242363


namespace max_red_socks_l242_242998

theorem max_red_socks (x y : ℕ) 
  (h1 : x + y ≤ 2017) 
  (h2 : (x * (x - 1) + y * (y - 1)) = (x + y) * (x + y - 1) / 2) : 
  x ≤ 990 := 
sorry

end max_red_socks_l242_242998


namespace playerA_winning_moves_l242_242455

-- Definitions of the game
-- Circles are labeled from 1 to 9
inductive Circle
| A | B | C1 | C2 | C3 | C4 | C5 | C6 | C7

inductive Player
| A | B

def StraightLine (c1 c2 c3 : Circle) : Prop := sorry
-- The straight line property between circles is specified by the game rules

-- Initial conditions
def initial_conditions (playerA_move playerB_move : Circle) : Prop :=
  playerA_move = Circle.A ∧ playerB_move = Circle.B

-- Winning condition
def winning_move (move : Circle) : Prop := sorry
-- This will check if a move leads to a win for Player A

-- Equivalent proof problem
theorem playerA_winning_moves : ∀ (move : Circle), initial_conditions Circle.A Circle.B → 
  (move = Circle.C2 ∨ move = Circle.C3 ∨ move = Circle.C4) → winning_move move :=
by
  sorry

end playerA_winning_moves_l242_242455


namespace speed_of_goods_train_l242_242869

open Real

theorem speed_of_goods_train
  (V_girl : ℝ := 100) -- The speed of the girl's train in km/h
  (t : ℝ := 6/3600)  -- The passing time in hours
  (L : ℝ := 560/1000) -- The length of the goods train in km
  (V_g : ℝ) -- The speed of the goods train in km/h
  : V_g = 236 := sorry

end speed_of_goods_train_l242_242869


namespace battery_current_l242_242022

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l242_242022


namespace rationalize_expression_l242_242556

theorem rationalize_expression :
  (2 * Real.sqrt 3) / (Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5) = 
  (Real.sqrt 6 + 3 - Real.sqrt 15) / 2 :=
sorry

end rationalize_expression_l242_242556


namespace arithmetic_progression_probability_l242_242646

def is_arithmetic_progression (a b c : ℕ) (d : ℕ) : Prop :=
  b - a = d ∧ c - b = d

noncomputable def probability_arithmetic_progression_diff_two : ℚ :=
  have total_outcomes : ℚ := 6 * 6 * 6
  have favorable_outcomes : ℚ := 12
  favorable_outcomes / total_outcomes

theorem arithmetic_progression_probability (d : ℕ) (h : d = 2) :
  probability_arithmetic_progression_diff_two = 1 / 18 :=
by 
  sorry

end arithmetic_progression_probability_l242_242646


namespace find_square_side_length_l242_242279

theorem find_square_side_length
  (a CF AE : ℝ)
  (h_CF : CF = 2 * a)
  (h_AE : AE = 3.5 * a)
  (h_sum : CF + AE = 91) :
  a = 26 := by
  sorry

end find_square_side_length_l242_242279


namespace total_amount_saved_l242_242250

def priceX : ℝ := 575
def surcharge_rateX : ℝ := 0.04
def installation_chargeX : ℝ := 82.50
def total_chargeX : ℝ := priceX + surcharge_rateX * priceX + installation_chargeX

def priceY : ℝ := 530
def surcharge_rateY : ℝ := 0.03
def installation_chargeY : ℝ := 93.00
def total_chargeY : ℝ := priceY + surcharge_rateY * priceY + installation_chargeY

def savings : ℝ := total_chargeX - total_chargeY

theorem total_amount_saved : savings = 41.60 :=
by
  sorry

end total_amount_saved_l242_242250


namespace scientific_notation_of_0_0000003_l242_242403

theorem scientific_notation_of_0_0000003 :
  0.0000003 = 3 * 10^(-7) :=
sorry

end scientific_notation_of_0_0000003_l242_242403


namespace total_cost_of_apples_l242_242887

def original_price_per_pound : ℝ := 1.6
def price_increase_percentage : ℝ := 0.25
def number_of_family_members : ℕ := 4
def pounds_per_person : ℝ := 2

theorem total_cost_of_apples : 
  let new_price_per_pound := original_price_per_pound * (1 + price_increase_percentage)
  let total_pounds := pounds_per_person * number_of_family_members
  let total_cost := total_pounds * new_price_per_pound
  total_cost = 16 := by
  sorry

end total_cost_of_apples_l242_242887


namespace interest_for_20000_l242_242759

-- Definition of simple interest
def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

variables (P1 P2 I1 I2 r : ℝ)
-- Given conditions
def h1 := (P1 = 5000)
def h2 := (I1 = 250)
def h3 := (r = I1 / P1)
-- Question condition
def h4 := (P2 = 20000)
def t := 1

theorem interest_for_20000 :
  P1 = 5000 →
  I1 = 250 →
  P2 = 20000 →
  r = I1 / P1 →
  simple_interest P2 r t = 1000 :=
by
  intros
  -- Proof goes here
  sorry

end interest_for_20000_l242_242759


namespace apple_cost_calculation_l242_242885

theorem apple_cost_calculation
    (original_price : ℝ)
    (price_raise : ℝ)
    (amount_per_person : ℝ)
    (num_people : ℝ) :
  original_price = 1.6 →
  price_raise = 0.25 →
  amount_per_person = 2 →
  num_people = 4 →
  (num_people * amount_per_person * (original_price * (1 + price_raise))) = 16 :=
by
  -- insert the mathematical proof steps/cardinality here
  sorry

end apple_cost_calculation_l242_242885


namespace intersection_A_B_l242_242312

def A : Set ℝ := { x | 2 < x ∧ x ≤ 4 }
def B : Set ℝ := { x | -1 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = { x | 2 < x ∧ x < 3 } :=
sorry

end intersection_A_B_l242_242312


namespace jane_doe_investment_l242_242339

theorem jane_doe_investment (total_investment mutual_funds real_estate : ℝ)
  (h1 : total_investment = 250000)
  (h2 : real_estate = 3 * mutual_funds)
  (h3 : total_investment = mutual_funds + real_estate) :
  real_estate = 187500 :=
by
  sorry

end jane_doe_investment_l242_242339


namespace quadratic_at_most_two_roots_l242_242352

theorem quadratic_at_most_two_roots (a b c x1 x2 x3 : ℝ) (ha : a ≠ 0) 
(h1 : a * x1^2 + b * x1 + c = 0)
(h2 : a * x2^2 + b * x2 + c = 0)
(h3 : a * x3^2 + b * x3 + c = 0)
(h_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) : 
false :=
sorry

end quadratic_at_most_two_roots_l242_242352


namespace number_of_balls_l242_242254

theorem number_of_balls (x : ℕ) (h : x - 20 = 30 - x) : x = 25 :=
sorry

end number_of_balls_l242_242254


namespace circle_area_from_circumference_l242_242487

theorem circle_area_from_circumference
  (c : ℝ)    -- the circumference
  (hc : c = 36)    -- condition: circumference is 36 cm
  : 
  ∃ A : ℝ,   -- there exists an area A
    A = 324 / π :=   -- conclusion: area is 324/π
by
  sorry   -- proof goes here

end circle_area_from_circumference_l242_242487


namespace simplify_fraction_l242_242407

-- We start by defining the problem in Lean.
theorem simplify_fraction :
  (1722^2 - 1715^2) / (1730^2 - 1705^2) = 7 / 25 :=
by
  -- Begin proof sketch (proof is not required, so we put sorry here)
  sorry

end simplify_fraction_l242_242407


namespace probability_AB_together_l242_242223

theorem probability_AB_together : 
  let total_events := 6
  let ab_together_events := 4
  let probability := ab_together_events / total_events
  probability = 2 / 3 :=
by
  sorry

end probability_AB_together_l242_242223


namespace evaluate_f_l242_242940

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1 else if x = 0 then Real.pi else 0

theorem evaluate_f : f (f (f (-1))) = Real.pi + 1 :=
by
  -- Proof goes here
  sorry

end evaluate_f_l242_242940


namespace equal_roots_of_quadratic_l242_242449

theorem equal_roots_of_quadratic (k : ℝ) : 
  (∃ x, (x^2 + 2 * x + k = 0) ∧ (x^2 + 2 * x + k) = 0) → k = 1 :=
by
  sorry

end equal_roots_of_quadratic_l242_242449


namespace circle_area_from_circumference_l242_242488

theorem circle_area_from_circumference
  (c : ℝ)    -- the circumference
  (hc : c = 36)    -- condition: circumference is 36 cm
  : 
  ∃ A : ℝ,   -- there exists an area A
    A = 324 / π :=   -- conclusion: area is 324/π
by
  sorry   -- proof goes here

end circle_area_from_circumference_l242_242488


namespace smallest_difference_of_factors_l242_242587

theorem smallest_difference_of_factors (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 2268) : 
  (a = 42 ∧ b = 54) ∨ (a = 54 ∧ b = 42) := sorry

end smallest_difference_of_factors_l242_242587


namespace routes_A_to_B_in_grid_l242_242172

theorem routes_A_to_B_in_grid : 
  let m := 3 in 
  let n := 3 in 
  let total_moves := m + n in 
  let move_to_right := m in 
  let move_down := n in 
  Nat.choose total_moves move_to_right = 20 := 
by
  let m := 3
  let n := 3
  let total_moves := m + n
  let move_to_right := m
  let move_down := n
  show Nat.choose total_moves move_to_right = 20
  sorry

end routes_A_to_B_in_grid_l242_242172


namespace remainder_5_pow_2023_mod_11_l242_242419

theorem remainder_5_pow_2023_mod_11 : (5^2023) % 11 = 4 :=
by
  have h1 : 5^2 % 11 = 25 % 11 := sorry
  have h2 : 25 % 11 = 3 := sorry
  have h3 : (3^5) % 11 = 1 := sorry
  have h4 : 3^1011 % 11 = ((3^5)^202 * 3) % 11 := sorry
  have h5 : ((3^5)^202 * 3) % 11 = (1^202 * 3) % 11 := sorry
  have h6 : (1^202 * 3) % 11 = 3 % 11 := sorry
  have h7 : (5^2023) % 11 = (3 * 5) % 11 := sorry
  have h8 : (3 * 5) % 11 = 15 % 11 := sorry
  have h9 : 15 % 11 = 4 := sorry
  exact h9

end remainder_5_pow_2023_mod_11_l242_242419


namespace middle_number_is_45_l242_242008

open Real

noncomputable def middle_number (l : List ℝ) (h_len : l.length = 13) 
  (h1 : (l.sum / 13) = 9) 
  (h2 : (l.take 6).sum = 30) 
  (h3 : (l.drop 7).sum = 42): ℝ := 
  l.nthLe 6 sorry  -- middle element (index 6 in 0-based index)

theorem middle_number_is_45 (l : List ℝ) (h_len : l.length = 13) 
  (h1 : (l.sum / 13) = 9) 
  (h2 : (l.take 6).sum = 30) 
  (h3 : (l.drop 7).sum = 42) : 
  middle_number l h_len h1 h2 h3 = 45 := 
sorry

end middle_number_is_45_l242_242008


namespace sqrt_9_eq_pm3_l242_242381

theorem sqrt_9_eq_pm3 : ∃ x : ℤ, x^2 = 9 ∧ (x = 3 ∨ x = -3) :=
by sorry

end sqrt_9_eq_pm3_l242_242381


namespace laps_run_l242_242782

theorem laps_run (x : ℕ) (total_distance required_distance lap_length extra_laps : ℕ) (h1 : total_distance = 2400) (h2 : lap_length = 150) (h3 : extra_laps = 4) (h4 : total_distance = lap_length * (x + extra_laps)) : x = 12 :=
by {
  sorry
}

end laps_run_l242_242782


namespace broken_glass_pieces_l242_242509

theorem broken_glass_pieces (x : ℕ) 
    (total_pieces : ℕ := 100) 
    (safe_fee : ℕ := 3) 
    (compensation : ℕ := 5) 
    (total_fee : ℕ := 260) 
    (h : safe_fee * (total_pieces - x) - compensation * x = total_fee) : x = 5 := by
  sorry

end broken_glass_pieces_l242_242509


namespace base9_addition_correct_l242_242680

-- Definition of base 9 addition problem.
def add_base9 (a b c : ℕ) : ℕ :=
  let sum := a + b + c -- Sum in base 10
  let d0 := sum % 9 -- Least significant digit in base 9
  let carry1 := sum / 9
  (carry1 + carry1 / 9 * 9 + carry1 % 9) + d0 -- Sum in base 9 considering carry

-- The specific values converted to base 9 integers
def n1 := 3 * 9^2 + 4 * 9 + 6
def n2 := 8 * 9^2 + 0 * 9 + 2
def n3 := 1 * 9^2 + 5 * 9 + 7

-- The expected result converted to base 9 integer
def expected_sum := 1 * 9^3 + 4 * 9^2 + 1 * 9 + 6

theorem base9_addition_correct : add_base9 n1 n2 n3 = expected_sum := by
  -- Proof will be provided here
  sorry

end base9_addition_correct_l242_242680


namespace find_r_for_two_roots_greater_than_neg_one_l242_242560

theorem find_r_for_two_roots_greater_than_neg_one :
  ∀ r : ℝ, (3.5 < r ∧ r < 4.5) ↔
  (let a := r - 4,
       b := -2 * (r - 3),
       c := r,
       discriminant := b^2 - 4 * a * c in
   discriminant > 0 ∧ 
   let vertex := -b / (2 * a) in
   vertex > -1 ∧ 
   a * (-1)^2 + b * (-1) + c > 0) :=
sorry

end find_r_for_two_roots_greater_than_neg_one_l242_242560


namespace complex_number_imaginary_l242_242593

theorem complex_number_imaginary (x : ℝ) 
  (h1 : x^2 - 2*x - 3 = 0)
  (h2 : x + 1 ≠ 0) : x = 3 := sorry

end complex_number_imaginary_l242_242593


namespace sequence_is_decreasing_l242_242740

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n

theorem sequence_is_decreasing (a : ℕ → ℝ) (h1 : a 1 < 0) (h2 : is_geometric_sequence a (1/3)) :
  ∀ n, a (n + 1) < a n :=
by
  -- Here should be the proof
  sorry

end sequence_is_decreasing_l242_242740


namespace inverse_proportion_symmetric_l242_242152

theorem inverse_proportion_symmetric (a b : ℝ) (h : a ≠ 0) (h_ab : b = -6 / -a) : (-b) = -6 / a :=
by
  -- the proof goes here
  sorry

end inverse_proportion_symmetric_l242_242152


namespace probability_all_white_balls_l242_242827

theorem probability_all_white_balls :
  let P := 7
  let Q := 8
  let total := P + Q
  (nat.choose P 4 * nat.choose 1 0) / nat.choose total 4 = (1 : ℚ) / 39 :=
by {
  have h1 : nat.choose total 4 = 1365 := by norm_num,
  have h2 : nat.choose P 4 = 35 := by norm_num,
  field_simp,
  rw [h1, h2],
  norm_num,
  sorry -- Proof steps are omitted.
}

end probability_all_white_balls_l242_242827


namespace base_s_is_8_l242_242330

-- Define a noncomputable field (necessary for algebraic manipulations)
noncomputable theory

-- Here is the Lean statement
theorem base_s_is_8 (s : ℕ) (h₁ : 5 * s^2 + 3 * s + s^3 + 2 * s^2 + 3 * s = 2 * s^3) : s = 8 :=
by sorry

end base_s_is_8_l242_242330


namespace sample_size_l242_242256

-- Definitions for the conditions
def ratio_A : Nat := 2
def ratio_B : Nat := 3
def ratio_C : Nat := 4
def stratified_sample_size : Nat := 9 -- Total parts in the ratio sum
def products_A_sample : Nat := 18 -- Sample contains 18 Type A products

-- We need to tie these conditions together and prove the size of the sample n
theorem sample_size (n : Nat) (ratio_A ratio_B ratio_C stratified_sample_size products_A_sample : Nat) :
  ratio_A = 2 → ratio_B = 3 → ratio_C = 4 → stratified_sample_size = 9 → products_A_sample = 18 → n = 81 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof body here
  sorry -- Placeholder for the proof

end sample_size_l242_242256


namespace remainder_of_171_divided_by_21_l242_242784

theorem remainder_of_171_divided_by_21 : 
  ∃ r, 171 = (21 * 8) + r ∧ r = 3 := 
by
  sorry

end remainder_of_171_divided_by_21_l242_242784


namespace gcd_of_terms_l242_242816

theorem gcd_of_terms (m n : ℕ) : gcd (4 * m^3 * n) (9 * m * n^3) = m * n := 
sorry

end gcd_of_terms_l242_242816


namespace expected_value_of_win_is_correct_l242_242832

noncomputable def expected_value_of_win : ℝ :=
  (1 / 8) * (8 - 1) + (1 / 8) * (8 - 2) + (1 / 8) * (8 - 3) +
  (1 / 8) * (8 - 4) + (1 / 8) * (8 - 5) + (1 / 8) * (8 - 6) +
  (1 / 8) * (8 - 7) + (1 / 8) * (8 - 8)

theorem expected_value_of_win_is_correct :
  expected_value_of_win = 3.5 :=
by
  sorry

end expected_value_of_win_is_correct_l242_242832


namespace binomial_coefficient_30_3_l242_242709

theorem binomial_coefficient_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_coefficient_30_3_l242_242709


namespace current_at_R_12_l242_242046

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l242_242046


namespace janine_total_pages_l242_242460

-- Define the conditions
def books_last_month : ℕ := 5
def books_this_month : ℕ := 2 * books_last_month
def books_per_page : ℕ := 10

-- Define the total number of pages she read in two months
def total_pages : ℕ :=
  let total_books := books_last_month + books_this_month
  total_books * books_per_page

-- State the theorem to be proven
theorem janine_total_pages : total_pages = 150 :=
by
  sorry

end janine_total_pages_l242_242460


namespace current_value_l242_242090

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l242_242090


namespace work_together_days_l242_242255

theorem work_together_days (hA : ∃ d : ℝ, d > 0 ∧ d = 15)
                          (hB : ∃ d : ℝ, d > 0 ∧ d = 20)
                          (hfrac : ∃ f : ℝ, f = (23 / 30)) :
  ∃ d : ℝ, d = 2 := by
  sorry

end work_together_days_l242_242255


namespace current_at_R_12_l242_242049

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l242_242049


namespace camera_guarantee_l242_242994

def battery_trials (b : Fin 22 → Bool) : Prop :=
  let charged := Finset.filter (λ i => b i) (Finset.univ : Finset (Fin 22))
  -- Ensuring there are exactly 15 charged batteries
  (charged.card = 15) ∧
  -- The camera works if any set of three batteries are charged
  (∀ (trials : Finset (Finset (Fin 22))),
   trials.card = 10 →
   ∃ t ∈ trials, (t.card = 3 ∧ t ⊆ charged))

theorem camera_guarantee :
  ∃ (b : Fin 22 → Bool), battery_trials b := by
  sorry

end camera_guarantee_l242_242994


namespace ELMO_value_l242_242772

def digits := {n : ℕ // n < 10}

variables (L E T M O : digits)

-- Conditions
axiom h1 : L.val ≠ 0
axiom h2 : O.val = 0
axiom h3 : (1000 * L.val + 100 * E.val + 10 * E.val + T.val) + (100 * L.val + 10 * M.val + T.val) = 1000 * T.val + L.val

-- Conclusion
theorem ELMO_value : E.val * 1000 + L.val * 100 + M.val * 10 + O.val = 1880 :=
sorry

end ELMO_value_l242_242772


namespace max_abs_sum_value_l242_242590

noncomputable def max_abs_sum (x y : ℝ) : ℝ := |x| + |y|

theorem max_abs_sum_value (x y : ℝ) (h : x^2 + y^2 = 4) : max_abs_sum x y ≤ 2 * Real.sqrt 2 :=
by {
  sorry
}

end max_abs_sum_value_l242_242590


namespace roxy_total_plants_remaining_l242_242353

def initial_flowering_plants : Nat := 7
def initial_fruiting_plants : Nat := 2 * initial_flowering_plants
def flowering_plants_bought : Nat := 3
def fruiting_plants_bought : Nat := 2
def flowering_plants_given_away : Nat := 1
def fruiting_plants_given_away : Nat := 4

def total_remaining_plants : Nat :=
  let flowering_plants_now := initial_flowering_plants + flowering_plants_bought - flowering_plants_given_away
  let fruiting_plants_now := initial_fruiting_plants + fruiting_plants_bought - fruiting_plants_given_away
  flowering_plants_now + fruiting_plants_now

theorem roxy_total_plants_remaining
  : total_remaining_plants = 21 := by
  sorry

end roxy_total_plants_remaining_l242_242353


namespace current_when_resistance_12_l242_242104

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l242_242104


namespace shared_bill_per_person_l242_242228

noncomputable def totalBill : ℝ := 139.00
noncomputable def tipPercentage : ℝ := 0.10
noncomputable def totalPeople : ℕ := 5

theorem shared_bill_per_person :
  let tipAmount := totalBill * tipPercentage
  let totalBillWithTip := totalBill + tipAmount
  let amountPerPerson := totalBillWithTip / totalPeople
  amountPerPerson = 30.58 :=
by
  let tipAmount := totalBill * tipPercentage
  let totalBillWithTip := totalBill + tipAmount
  let amountPerPerson := totalBillWithTip / totalPeople
  have h1 : tipAmount = 13.90 := by sorry
  have h2 : totalBillWithTip = 152.90 := by sorry
  have h3 : amountPerPerson = 30.58 := by sorry
  exact h3

end shared_bill_per_person_l242_242228


namespace find_current_when_resistance_is_12_l242_242040

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l242_242040


namespace heaps_never_empty_l242_242231

-- Define initial conditions
def initial_heaps := (1993, 199, 19)

-- Allowed operations
def add_stones (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
if a = 1993 then (a + b + c, b, c)
else if b = 199 then (a, b + a + c, c)
else (a, b, c + a + b)

def remove_stones (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
if a = 1993 then (a - (b + c), b, c)
else if b = 199 then (a, b - (a + c), c)
else (a, b, c - (a + b))

-- The proof statement
theorem heaps_never_empty :
  ∀ a b c : ℕ, a = 1993 ∧ b = 199 ∧ c = 19 ∧ (∀ n : ℕ, (a + b + c) % 2 = 1) ∧ (a - (b + c) % 2 = 1) → ¬(a = 0 ∨ b = 0 ∨ c = 0) := 
by {
  sorry
}

end heaps_never_empty_l242_242231


namespace math_problem_l242_242735

variable (a : ℝ)
noncomputable def problem := a = Real.sqrt 11 - 1
noncomputable def target := a^2 + 2 * a + 1 = 11

theorem math_problem (h : problem a) : target a :=
  sorry

end math_problem_l242_242735


namespace percentage_reduction_of_faculty_l242_242125

noncomputable def percentage_reduction (original reduced : ℝ) : ℝ :=
  ((original - reduced) / original) * 100

theorem percentage_reduction_of_faculty :
  percentage_reduction 226.74 195 = 13.99 :=
by sorry

end percentage_reduction_of_faculty_l242_242125


namespace binomial_sum_to_220_l242_242927

open Nat

def binomial_coeff (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binomial_sum_to_220 :
  binomial_coeff 2 2 + binomial_coeff 3 2 + binomial_coeff 4 2 + binomial_coeff 5 2 +
  binomial_coeff 6 2 + binomial_coeff 7 2 + binomial_coeff 8 2 + binomial_coeff 9 2 +
  binomial_coeff 10 2 + binomial_coeff 11 2 = 220 :=
by
  /- Proof goes here, use the computed value of combinations -/
  sorry

end binomial_sum_to_220_l242_242927


namespace current_at_resistance_12_l242_242037

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l242_242037


namespace circumradius_relationship_l242_242518

theorem circumradius_relationship 
  (a b c a' b' c' R : ℝ)
  (S S' p p' : ℝ)
  (h₁ : R = (a * b * c) / (4 * S))
  (h₂ : R = (a' * b' * c') / (4 * S'))
  (h₃ : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h₄ : S' = Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c')))
  (h₅ : p = (a + b + c) / 2)
  (h₆ : p' = (a' + b' + c') / 2) :
  (a * b * c) / Real.sqrt (p * (p - a) * (p - b) * (p - c)) = 
  (a' * b' * c') / Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c')) :=
by 
  sorry

end circumradius_relationship_l242_242518


namespace expression_equals_66069_l242_242813

-- Definitions based on the conditions
def numerator : Nat := 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10
def denominator : Nat := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10
def expression : Rat := numerator / denominator

-- The main theorem to be proven
theorem expression_equals_66069 : expression = 66069 := by
  sorry

end expression_equals_66069_l242_242813


namespace sin_cos_inequality_l242_242623

open Real

theorem sin_cos_inequality 
  (x : ℝ) (hx : 0 < x ∧ x < π / 2) 
  (m n : ℕ) (hmn : n > m)
  : 2 * abs (sin x ^ n - cos x ^ n) ≤ 3 * abs (sin x ^ m - cos x ^ m) :=
sorry

end sin_cos_inequality_l242_242623


namespace slope_of_line_AB_is_pm_4_3_l242_242431

noncomputable def slope_of_line_AB : ℝ := sorry

theorem slope_of_line_AB_is_pm_4_3 (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : y₁^2 = 4 * x₁)
  (h₂ : y₂^2 = 4 * x₂)
  (h₃ : (x₁, y₁) ≠ (x₂, y₂))
  (h₄ : (x₁ - 1, y₁) = -4 * (x₂ - 1, y₂)) :
  slope_of_line_AB = 4 / 3 ∨ slope_of_line_AB = -4 / 3 :=
sorry

end slope_of_line_AB_is_pm_4_3_l242_242431


namespace peter_savings_l242_242797

noncomputable def calc_discounted_price (original_price : ℝ) (discount_percentage : ℝ) : ℝ :=
    original_price * (1 - discount_percentage / 100)

noncomputable def calc_savings (original_price : ℝ) (external_price : ℝ) : ℝ :=
    original_price - external_price

noncomputable def total_savings : ℝ :=
    let math_original := 45.0
    let math_discount := 20.0
    let science_original := 60.0
    let science_discount := 25.0
    let literature_original := 35.0
    let literature_discount := 15.0
    let math_external := calc_discounted_price math_original math_discount
    let science_external := calc_discounted_price science_original science_discount
    let literature_external := calc_discounted_price literature_original literature_discount
    let math_savings := calc_savings math_original math_external
    let science_savings := calc_savings science_original science_external
    let literature_savings := calc_savings literature_original literature_external
    math_savings + science_savings + literature_savings

theorem peter_savings :
  total_savings = 29.25 :=
by
    sorry

end peter_savings_l242_242797


namespace intersection_of_P_and_Q_l242_242609

def P : Set ℝ := {x | Real.log x > 0}
def Q : Set ℝ := {x | -1 < x ∧ x < 2}
def R : Set ℝ := {x | 1 < x ∧ x < 2}

theorem intersection_of_P_and_Q : P ∩ Q = R := by
  sorry

end intersection_of_P_and_Q_l242_242609


namespace battery_current_l242_242113

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l242_242113


namespace solve_quadratic_l242_242355

theorem solve_quadratic : ∀ x : ℝ, 2 * x^2 - 3 * x + 1 = 0 ↔ x = 1 / 2 ∨ x = 1 :=
by
  intro x
  construct sorry

end solve_quadratic_l242_242355


namespace remainder_when_divided_by_6_l242_242238

theorem remainder_when_divided_by_6 (n : ℕ) (h1 : n % 12 = 8) : n % 6 = 2 :=
sorry

end remainder_when_divided_by_6_l242_242238


namespace problem_statement_l242_242176

theorem problem_statement (a b c : ℕ) (h1 : a < 12) (h2 : b < 12) (h3 : c < 12) (h4 : b + c = 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b + c :=
by
  sorry

end problem_statement_l242_242176


namespace expected_value_of_win_is_3_point_5_l242_242851

noncomputable def expected_value_win : ℚ :=
  let win (n : ℕ) : ℚ := 8 - n
  let probabilities := List.repeat (1/8 : ℚ) 8
  (List.range 8).map (λ n => probabilities.head! * win (n + 1)).sum

theorem expected_value_of_win_is_3_point_5 : expected_value_win = 3.5 := by
  sorry

end expected_value_of_win_is_3_point_5_l242_242851


namespace total_is_twenty_l242_242391

def num_blue := 5
def num_red := 7
def prob_red_or_white : ℚ := 0.75

noncomputable def total_marbles (T : ℕ) (W : ℕ) :=
  5 + 7 + W = T ∧ (7 + W) / T = prob_red_or_white

theorem total_is_twenty : ∃ (T : ℕ) (W : ℕ), total_marbles T W ∧ T = 20 :=
by
  sorry

end total_is_twenty_l242_242391


namespace dot_product_result_l242_242315

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, 1)

theorem dot_product_result : (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end dot_product_result_l242_242315


namespace janine_read_pages_in_two_months_l242_242462

theorem janine_read_pages_in_two_months :
  (let books_last_month := 5
   let books_this_month := 2 * books_last_month
   let total_books := books_last_month + books_this_month
   let pages_per_book := 10
   total_books * pages_per_book = 150) := by
   sorry

end janine_read_pages_in_two_months_l242_242462


namespace smallest_abc_sum_l242_242722

theorem smallest_abc_sum : 
  ∃ (a b c : ℕ), (a * c + 2 * b * c + a + 2 * b = c^2 + c + 6) ∧ (∀ (a' b' c' : ℕ), (a' * c' + 2 * b' * c' + a' + 2 * b' = c'^2 + c' + 6) → (a' + b' + c' ≥ a + b + c)) → (a, b, c) = (2, 1, 1) := 
by
  sorry

end smallest_abc_sum_l242_242722


namespace problem_l242_242739

-- Definitions and conditions
variable {a : ℕ → ℝ} -- sequence definition
variable {S : ℕ → ℝ} -- sum of first n terms

-- Condition: a_n ≠ 0 for all n ∈ ℕ^*
axiom h1 : ∀ n : ℕ, n > 0 → a n ≠ 0

-- Condition: a_n * a_{n+1} = S_n
axiom h2 : ∀ n : ℕ, n > 0 → a n * a (n + 1) = S n

-- Given: S_1 = a_1
axiom h3 : S 1 = a 1

-- Given: S_2 = a_1 + a_2
axiom h4 : S 2 = a 1 + a 2

-- Prove: a_3 - a_1 = 1
theorem problem : a 3 - a 1 = 1 := by
  sorry

end problem_l242_242739


namespace brenda_total_erasers_l242_242685

theorem brenda_total_erasers (number_of_groups : ℕ) (erasers_per_group : ℕ) (h1 : number_of_groups = 3) (h2 : erasers_per_group = 90) : number_of_groups * erasers_per_group = 270 := 
by
  sorry

end brenda_total_erasers_l242_242685


namespace tim_total_points_l242_242273

theorem tim_total_points :
  let single_points := 1000
  let tetris_points := 8 * single_points
  let singles := 6
  let tetrises := 4
  let total_points := singles * single_points + tetrises * tetris_points
  total_points = 38000 :=
by
  sorry

end tim_total_points_l242_242273


namespace binom_60_3_l242_242919

theorem binom_60_3 : nat.choose 60 3 = 34220 :=
by
  -- Proof goes here.
  sorry

end binom_60_3_l242_242919


namespace t_range_exists_f_monotonic_intervals_l242_242582

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp (-x)
noncomputable def f' (x : ℝ) : ℝ := -x * Real.exp (-x)
noncomputable def φ (x t : ℝ) : ℝ := x * f x + t * f' x + Real.exp (-x)

theorem t_range_exists (t : ℝ) : ∃ (x_1 x_2 : ℝ), x_1 ∈ Icc 0 1 ∧ x_2 ∈ Icc 0 1 ∧ 2 * φ x_1 t < φ x_2 t
    ↔ t ∈ Iio (3 - 2 * Real.exp 1) ∪ Ioi (3 - Real.exp 1 / 2) := 
  sorry

theorem f_monotonic_intervals : 
  let intervals := ((Iio 0), (Ioi 0)) in
  ∀ x : ℝ, (x ∈ intervals.1 → MonotoneOn f intervals.1) ∧ (x ∈ intervals.2 → MonotoneOn f intervals.2) :=
  sorry

end t_range_exists_f_monotonic_intervals_l242_242582


namespace adding_2_to_odd_integer_can_be_prime_l242_242554

def is_odd (n : ℤ) : Prop := n % 2 ≠ 0
def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m ∣ n → m = 1 ∨ m = n

theorem adding_2_to_odd_integer_can_be_prime :
  ∃ n : ℤ, is_odd n ∧ is_prime (n + 2) :=
by
  sorry

end adding_2_to_odd_integer_can_be_prime_l242_242554


namespace card_trick_l242_242586

/-- A magician is able to determine the fifth card from a 52-card deck using a prearranged 
    communication system between the magician and the assistant, thus no supernatural 
    abilities are required. -/
theorem card_trick (deck : Finset ℕ) (h_deck : deck.card = 52) (chosen_cards : Finset ℕ)
  (h_chosen : chosen_cards.card = 5) (shown_cards : Finset ℕ) (h_shown : shown_cards.card = 4)
  (fifth_card : ℕ) (h_fifth_card : fifth_card ∈ chosen_cards \ shown_cards) :
  ∃ (prearranged_system : (Finset ℕ) → (Finset ℕ) → ℕ),
    ∀ (remaining : Finset ℕ), remaining.card = 1 → 
    prearranged_system shown_cards remaining = fifth_card := 
sorry

end card_trick_l242_242586


namespace sequence_exists_l242_242613

theorem sequence_exists
  {a_0 b_0 c_0 a b c : ℤ}
  (gcd1 : Int.gcd (Int.gcd a_0 b_0) c_0 = 1)
  (gcd2 : Int.gcd (Int.gcd a b) c = 1) :
  ∃ (n : ℕ) (a_seq b_seq c_seq : Fin (n + 1) → ℤ),
    a_seq 0 = a_0 ∧ b_seq 0 = b_0 ∧ c_seq 0 = c_0 ∧ 
    a_seq n = a ∧ b_seq n = b ∧ c_seq n = c ∧
    ∀ (i : Fin n), (a_seq i) * (a_seq i.succ) + (b_seq i) * (b_seq i.succ) + (c_seq i) * (c_seq i.succ) = 1 :=
sorry

end sequence_exists_l242_242613


namespace remainder_8_pow_2023_div_5_l242_242000

-- Definition for modulo operation
def mod_five (a : Nat) : Nat := a % 5

-- Key theorem to prove
theorem remainder_8_pow_2023_div_5 : mod_five (8 ^ 2023) = 2 :=
by
  sorry -- This is where the proof would go, but it's not required per the instructions

end remainder_8_pow_2023_div_5_l242_242000


namespace medians_concurrent_l242_242982

/--
For any triangle ABC, there exists a point G, known as the centroid, such that
the sum of the vectors from G to each of the vertices A, B, and C is the zero vector.
-/
theorem medians_concurrent 
  (A B C : ℝ×ℝ) : 
  ∃ G : ℝ×ℝ, (G -ᵥ A) + (G -ᵥ B) + (G -ᵥ C) = (0, 0) := 
by 
  -- proof will go here
  sorry 

end medians_concurrent_l242_242982


namespace find_positive_real_solution_l242_242147

theorem find_positive_real_solution :
∃ x : ℝ, 0 < x ∧ (1/3 * (7 * x^2 - 3) = (x^2 - 70 * x - 20) * (x^2 + 35 * x + 7)) :=
sorry

end find_positive_real_solution_l242_242147


namespace matrix_addition_l242_242893

def M1 : Matrix (Fin 3) (Fin 3) ℤ :=
![![4, 1, -3],
  ![0, -2, 5],
  ![7, 0, 1]]

def M2 : Matrix (Fin 3) (Fin 3) ℤ :=
![![ -6,  9, 2],
  ![  3, -4, -8],
  ![  0,  5, -3]]

def M3 : Matrix (Fin 3) (Fin 3) ℤ :=
![![ -2, 10, -1],
  ![  3, -6, -3],
  ![  7,  5, -2]]

theorem matrix_addition : M1 + M2 = M3 := by
  sorry

end matrix_addition_l242_242893


namespace intersection_ab_correct_l242_242568

noncomputable def set_A : Set ℝ := { x : ℝ | x > 1/3 }
def set_B : Set ℝ := { x : ℝ | ∃ y : ℝ, x^2 + y^2 = 4 ∧ y ≥ -2 ∧ y ≤ 2 }
def intersection_AB : Set ℝ := { x : ℝ | 1/3 < x ∧ x ≤ 2 }

theorem intersection_ab_correct : set_A ∩ set_B = intersection_AB := 
by 
  -- proof omitted
  sorry

end intersection_ab_correct_l242_242568


namespace coconut_trees_per_sqm_l242_242199

def farm_area : ℕ := 20
def harvests : ℕ := 2
def total_earnings : ℝ := 240
def coconut_price : ℝ := 0.50
def coconuts_per_tree : ℕ := 6

theorem coconut_trees_per_sqm : 
  let total_coconuts := total_earnings / coconut_price / harvests
  let total_trees := total_coconuts / coconuts_per_tree 
  let trees_per_sqm := total_trees / farm_area 
  trees_per_sqm = 2 :=
by
  sorry

end coconut_trees_per_sqm_l242_242199


namespace balls_into_boxes_l242_242957

-- Define the conditions
def balls : ℕ := 7
def boxes : ℕ := 4

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Prove the equivalent proof problem
theorem balls_into_boxes :
    (binom (balls - 1) (boxes - 1) = 20) ∧ (binom (balls + (boxes - 1)) (boxes - 1) = 120) := by
  sorry

end balls_into_boxes_l242_242957


namespace average_age_of_cricket_team_l242_242819

theorem average_age_of_cricket_team 
  (num_members : ℕ)
  (avg_age : ℕ)
  (wicket_keeper_age : ℕ)
  (remaining_avg : ℕ)
  (cond1 : num_members = 11)
  (cond2 : avg_age = 29)
  (cond3 : wicket_keeper_age = avg_age + 3)
  (cond4 : remaining_avg = avg_age - 1) : 
  avg_age = 29 := 
by 
  have h1 : num_members = 11 := cond1
  have h2 : avg_age = 29 := cond2
  have h3 : wicket_keeper_age = avg_age + 3 := cond3
  have h4 : remaining_avg = avg_age - 1 := cond4
  -- proof steps will go here
  sorry

end average_age_of_cricket_team_l242_242819


namespace baker_weekend_hours_l242_242665

noncomputable def loaves_per_hour : ℕ := 5
noncomputable def ovens : ℕ := 4
noncomputable def weekday_hours : ℕ := 5
noncomputable def total_loaves : ℕ := 1740
noncomputable def weeks : ℕ := 3
noncomputable def weekday_days : ℕ := 5
noncomputable def weekend_days : ℕ := 2

theorem baker_weekend_hours :
  ((total_loaves - (weeks * weekday_days * weekday_hours * (loaves_per_hour * ovens))) / (weeks * (loaves_per_hour * ovens))) / weekend_days = 4 := by
  sorry

end baker_weekend_hours_l242_242665


namespace binom_60_3_l242_242899

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l242_242899


namespace line_intersection_l242_242413

theorem line_intersection : 
  ∃ (x y : ℚ), 
    8 * x - 5 * y = 10 ∧ 
    3 * x + 2 * y = 16 ∧ 
    x = 100 / 31 ∧ 
    y = 98 / 31 :=
by
  use 100 / 31
  use 98 / 31
  sorry

end line_intersection_l242_242413


namespace binom_30_3_l242_242704

theorem binom_30_3 : Nat.choose 30 3 = 4060 := 
by 
  sorry

end binom_30_3_l242_242704


namespace cricket_jumps_to_100m_l242_242831

theorem cricket_jumps_to_100m (x y : ℕ) (h : 9 * x + 8 * y = 100) : x + y = 12 :=
sorry

end cricket_jumps_to_100m_l242_242831


namespace true_propositions_l242_242742

-- Definitions according to conditions:
def p (x y : ℝ) : Prop := x > y → -x < -y
def q (x y : ℝ) : Prop := x > y → x^2 > y^2

-- Given that p is true and q is false.
axiom p_true {x y : ℝ} : p x y
axiom q_false {x y : ℝ} : ¬ q x y

-- Proving the actual propositions that are true:
theorem true_propositions (x y : ℝ) : 
  (p x y ∨ q x y) ∧ (p x y ∧ ¬ q x y) :=
by
  have h1 : p x y := p_true
  have h2 : ¬ q x y := q_false
  constructor
  · left; exact h1
  · constructor; assumption; assumption

end true_propositions_l242_242742


namespace fraction_calculation_l242_242405

theorem fraction_calculation :
  let a := (1 / 2) + (1 / 3)
  let b := (2 / 7) + (1 / 4)
  ((a / b) * (3 / 5)) = (14 / 15) :=
by
  sorry

end fraction_calculation_l242_242405


namespace ab_squared_ab_cubed_ab_power_n_l242_242896

-- Definitions of a and b as real numbers, and n as a natural number
variables (a b : ℝ) (n : ℕ)

theorem ab_squared (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by 
  sorry

theorem ab_cubed (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by 
  sorry

theorem ab_power_n (a b : ℝ) (n : ℕ) : (a * b) ^ n = a ^ n * b ^ n := by 
  sorry

end ab_squared_ab_cubed_ab_power_n_l242_242896


namespace running_time_square_field_l242_242007

theorem running_time_square_field
  (side : ℕ)
  (running_speed_kmh : ℕ)
  (perimeter : ℕ := 4 * side)
  (running_speed_ms : ℕ := (running_speed_kmh * 1000) / 3600)
  (time : ℕ := perimeter / running_speed_ms) 
  (h_side : side = 35)
  (h_speed : running_speed_kmh = 9) :
  time = 56 := 
by
  sorry

end running_time_square_field_l242_242007


namespace median_length_of_pieces_is_198_l242_242252

   -- Define the conditions
   variables (A B C D E : ℕ)
   variables (h_order : A ≤ B ∧ B ≤ C ∧ C ≤ D ∧ D ≤ E)
   variables (avg_length : (A + B + C + D + E) = 640)
   variables (h_A_max : A ≤ 110)

   -- Statement of the problem (proof stub)
   theorem median_length_of_pieces_is_198 :
     C = 198 :=
   by
   sorry
   
end median_length_of_pieces_is_198_l242_242252


namespace find_blotted_digits_l242_242156

theorem find_blotted_digits :
  ∃ x y : ℕ, (∃ n : ℕ, n * 3600 = 234 * 1000 + x * 100 + y * 10) ∧ x = 7 ∧ y = 0 :=
begin
  sorry
end

end find_blotted_digits_l242_242156


namespace binom_30_3_eq_4060_l242_242695

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := 
  by sorry

end binom_30_3_eq_4060_l242_242695


namespace binomial_60_3_eq_34220_l242_242915

open Nat

theorem binomial_60_3_eq_34220 : choose 60 3 = 34220 := 
by
  sorry

end binomial_60_3_eq_34220_l242_242915


namespace union_M_N_equals_0_1_5_l242_242615

def M : Set ℝ := { x | x^2 - 6 * x + 5 = 0 }
def N : Set ℝ := { x | x^2 - 5 * x = 0 }

theorem union_M_N_equals_0_1_5 : M ∪ N = {0, 1, 5} := by
  sorry

end union_M_N_equals_0_1_5_l242_242615


namespace solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0_l242_242433

variable (a b : ℝ)

theorem solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0 :
  (∀ x : ℝ, (|x - 2| > 1 ↔ x^2 + a * x + b > 0)) → a + b = -1 :=
by
  sorry

end solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0_l242_242433


namespace minimum_single_discount_l242_242939

theorem minimum_single_discount (n : ℕ) :
  (∀ x : ℝ, 0 < x → 
    ((1 - n / 100) * x < (1 - 0.18) * (1 - 0.18) * x) ∧
    ((1 - n / 100) * x < (1 - 0.12) * (1 - 0.12) * (1 - 0.12) * x) ∧
    ((1 - n / 100) * x < (1 - 0.28) * (1 - 0.07) * x))
  ↔ n = 34 :=
by
  sorry

end minimum_single_discount_l242_242939


namespace square_triangle_ratios_l242_242136

theorem square_triangle_ratios (s t : ℝ) 
  (P_s := 4 * s) 
  (R_s := s * Real.sqrt 2 / 2)
  (P_t := 3 * t) 
  (R_t := t * Real.sqrt 3 / 3) 
  (h : s = t) : 
  (P_s / P_t = 4 / 3) ∧ (R_s / R_t = Real.sqrt 6 / 2) := 
by
  sorry

end square_triangle_ratios_l242_242136


namespace lizard_eyes_l242_242338

theorem lizard_eyes (E W S : Nat) 
  (h1 : W = 3 * E) 
  (h2 : S = 7 * W) 
  (h3 : E = S + W - 69) : 
  E = 3 := 
by
  sorry

end lizard_eyes_l242_242338


namespace binom_60_3_l242_242920

theorem binom_60_3 : nat.choose 60 3 = 34220 :=
by
  -- Proof goes here.
  sorry

end binom_60_3_l242_242920


namespace distinct_arith_prog_triangles_l242_242584

theorem distinct_arith_prog_triangles (n : ℕ) (h10 : n % 10 = 0) : 
  (3 * n = 180 → ∃ d : ℕ, ∀ a b c, a = n - d ∧ b = n ∧ c = n + d 
  →  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d < 60) :=
by
  sorry

end distinct_arith_prog_triangles_l242_242584


namespace willie_stickers_l242_242660

def num_stickers_start_with (given_away remaining initial : Nat) : Prop :=
  remaining + given_away = initial

theorem willie_stickers : ∃ initial, num_stickers_start_with 7 29 initial ∧ initial = 36 :=
by {
  use 36,
  split,
  { -- prove that remaining + given_away = initial
    show 29 + 7 = 36,
    exact rfl,
  },
  { -- prove that initial = 36
    show 36 = 36,
    exact rfl,
  }
}

end willie_stickers_l242_242660


namespace volume_of_rectangular_solid_l242_242270

theorem volume_of_rectangular_solid 
  (a b c : ℝ)
  (h1 : a * b = 15)
  (h2 : b * c = 10)
  (h3 : c * a = 6) :
  a * b * c = 30 := 
by
  -- sorry placeholder for the proof
  sorry

end volume_of_rectangular_solid_l242_242270


namespace math_problem_l242_242736

variable (a : ℝ)
noncomputable def problem := a = Real.sqrt 11 - 1
noncomputable def target := a^2 + 2 * a + 1 = 11

theorem math_problem (h : problem a) : target a :=
  sorry

end math_problem_l242_242736


namespace gail_has_two_ten_dollar_bills_l242_242425

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

end gail_has_two_ten_dollar_bills_l242_242425


namespace sum_of_last_two_digits_l242_242547

-- Definitions based on given conditions
def six_power_twenty_five := 6^25
def fourteen_power_twenty_five := 14^25
def expression := six_power_twenty_five + fourteen_power_twenty_five
def modulo := 100

-- The statement we need to prove
theorem sum_of_last_two_digits : expression % modulo = 0 := by
  sorry

end sum_of_last_two_digits_l242_242547


namespace square_cookie_cutters_count_l242_242928

def triangles_sides : ℕ := 6 * 3
def hexagons_sides : ℕ := 2 * 6
def total_sides : ℕ := 46
def sides_from_squares (S : ℕ) : ℕ := S * 4

theorem square_cookie_cutters_count (S : ℕ) :
  triangles_sides + hexagons_sides + sides_from_squares S = total_sides → S = 4 :=
by
  sorry

end square_cookie_cutters_count_l242_242928


namespace expected_value_of_win_is_3point5_l242_242837

noncomputable def expected_value_win : ℝ := 
  ∑ n in Finset.range 8, (1 / 8 : ℝ) * (8 - n)

theorem expected_value_of_win_is_3point5 : expected_value_win = 3.5 := 
by 
  sorry

end expected_value_of_win_is_3point5_l242_242837


namespace pizza_problem_l242_242253

noncomputable def pizza_slices (total_slices pepperoni_slices mushroom_slices : ℕ) : ℕ := 
  let slices_with_both := total_slices - (pepperoni_slices + mushroom_slices - total_slices)
  slices_with_both

theorem pizza_problem 
  (total_slices pepperoni_slices mushroom_slices : ℕ)
  (h_total: total_slices = 16)
  (h_pepperoni: pepperoni_slices = 8)
  (h_mushrooms: mushroom_slices = 12)
  (h_at_least_one: pepperoni_slices + mushroom_slices - total_slices ≥ 0)
  (h_no_three_toppings: total_slices = pepperoni_slices + mushroom_slices - 
   (total_slices - (pepperoni_slices + mushroom_slices - total_slices))) : 
  pizza_slices total_slices pepperoni_slices mushroom_slices = 4 :=
by 
  rw [h_total, h_pepperoni, h_mushrooms]
  sorry

end pizza_problem_l242_242253


namespace solutionY_materialB_correct_l242_242478

open Real

-- Definitions and conditions from step a
def solutionX_materialA : ℝ := 0.20
def solutionX_materialB : ℝ := 0.80
def solutionY_materialA : ℝ := 0.30
def mixture_materialA : ℝ := 0.22
def solutionX_in_mixture : ℝ := 0.80
def solutionY_in_mixture : ℝ := 0.20

-- The conjecture to prove
theorem solutionY_materialB_correct (B_Y : ℝ) 
  (h1 : solutionX_materialA = 0.20)
  (h2 : solutionX_materialB = 0.80) 
  (h3 : solutionY_materialA = 0.30) 
  (h4 : mixture_materialA = 0.22)
  (h5 : solutionX_in_mixture = 0.80)
  (h6 : solutionY_in_mixture = 0.20) :
  B_Y = 1 - solutionY_materialA := by 
  sorry

end solutionY_materialB_correct_l242_242478


namespace equation1_solution_equation2_solution_l242_242206

theorem equation1_solution (x : ℚ) : 2 * (x - 3) = 1 - 3 * (x + 1) → x = 4 / 5 :=
by sorry

theorem equation2_solution (x : ℚ) : 3 * x + (x - 1) / 2 = 3 - (x - 1) / 3 → x = 1 :=
by sorry

end equation1_solution_equation2_solution_l242_242206


namespace denomination_of_second_note_l242_242120

theorem denomination_of_second_note
  (x : ℕ)
  (y : ℕ)
  (z : ℕ)
  (h1 : x = y)
  (h2 : y = z)
  (h3 : x + y + z = 75)
  (h4 : 1 * x + y * x + 10 * x = 400):
  y = 5 := by
  sorry

end denomination_of_second_note_l242_242120


namespace battery_current_at_given_resistance_l242_242095

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l242_242095


namespace binomial_30_3_l242_242702

-- Defining the binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Statement of the problem in Lean 4
theorem binomial_30_3 : binomial 30 3 = 12180 :=
by
  sorry

end binomial_30_3_l242_242702


namespace rank_from_last_l242_242325

theorem rank_from_last (total_students : ℕ) (rank_from_top : ℕ) (rank_from_last : ℕ) : 
  total_students = 35 → 
  rank_from_top = 14 → 
  rank_from_last = (total_students - rank_from_top + 1) → 
  rank_from_last = 22 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end rank_from_last_l242_242325


namespace simplify_polynomial_l242_242553

theorem simplify_polynomial (x : ℝ) (A B C D : ℝ) :
  (y = (x^3 + 12 * x^2 + 47 * x + 60) / (x + 3)) →
  (y = A * x^2 + B * x + C) →
  x ≠ D →
  A = 1 ∧ B = 9 ∧ C = 20 ∧ D = -3 :=
by
  sorry

end simplify_polynomial_l242_242553


namespace cos2_add_2sin2_eq_64_over_25_l242_242324

theorem cos2_add_2sin2_eq_64_over_25 (α : ℝ) (h : Real.tan α = 3 / 4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
sorry

end cos2_add_2sin2_eq_64_over_25_l242_242324


namespace number_of_solutions_l242_242494

open Real

-- Define the main equation in terms of absolute values 
def equation (x : ℝ) : Prop := abs (x - abs (2 * x + 1)) = 3

-- Prove that there are exactly 2 distinct solutions to the equation
theorem number_of_solutions : 
  ∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ ∧ equation x₂ :=
sorry

end number_of_solutions_l242_242494


namespace least_possible_integer_for_friends_statements_l242_242259

theorem least_possible_integer_for_friends_statements 
    (M : Nat)
    (statement_divisible_by : Nat → Prop)
    (h1 : ∀ n, 1 ≤ n ∧ n ≤ 30 → statement_divisible_by n = (M % n = 0))
    (h2 : ∃ m, 1 ≤ m ∧ m < 30 ∧ (statement_divisible_by m = false ∧ 
                                    statement_divisible_by (m + 1) = false)) :
    M = 12252240 :=
by
  sorry

end least_possible_integer_for_friends_statements_l242_242259


namespace workman_problem_l242_242535

theorem workman_problem
    (total_work : ℝ)
    (B_rate : ℝ)
    (A_rate : ℝ)
    (days_together : ℝ)
    (W : total_work = 8 * (A_rate + B_rate))
    (A_2B : A_rate = 2 * B_rate) :
    total_work = 24 * B_rate :=
by
  sorry

end workman_problem_l242_242535


namespace horror_movie_more_than_triple_romance_l242_242221

-- Definitions and Conditions
def tickets_sold_romance : ℕ := 25
def tickets_sold_horror : ℕ := 93
def triple_tickets_romance := 3 * tickets_sold_romance

-- Theorem Statement
theorem horror_movie_more_than_triple_romance :
  (tickets_sold_horror - triple_tickets_romance) = 18 :=
by
  sorry

end horror_movie_more_than_triple_romance_l242_242221


namespace generalized_inequality_l242_242161

theorem generalized_inequality (x : ℝ) (n : ℕ) (h1 : x > 0) : x^n + (n : ℝ) / x > n + 1 := 
sorry

end generalized_inequality_l242_242161


namespace Jack_has_18_dimes_l242_242986

theorem Jack_has_18_dimes :
  ∃ d q : ℕ, (d = q + 3 ∧ 10 * d + 25 * q = 555) ∧ d = 18 :=
by
  sorry

end Jack_has_18_dimes_l242_242986


namespace evaluate_expression_l242_242720

noncomputable def log_4_8 : ℝ := Real.log 8 / Real.log 4
noncomputable def log_8_16 : ℝ := Real.log 16 / Real.log 8

theorem evaluate_expression : Real.sqrt (log_4_8 * log_8_16) = Real.sqrt 2 :=
by
  sorry

end evaluate_expression_l242_242720


namespace oleg_can_find_adjacent_cells_divisible_by_4_l242_242185

theorem oleg_can_find_adjacent_cells_divisible_by_4 :
  ∀ (grid : Fin 22 → Fin 22 → ℕ),
  (∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 22 * 22) →
  ∃ i j k l, ((i = k ∧ (j = l + 1 ∨ j = l - 1)) ∨ ((i = k + 1 ∨ i = k - 1) ∧ j = l)) ∧ ((grid i j + grid k l) % 4 = 0) :=
by
  sorry

end oleg_can_find_adjacent_cells_divisible_by_4_l242_242185


namespace triangle_side_b_value_l242_242775

theorem triangle_side_b_value (a b c : ℝ) (A B C : ℝ) (h1 : a = Real.sqrt 3) (h2 : A = 60) (h3 : C = 75) : b = Real.sqrt 2 :=
sorry

end triangle_side_b_value_l242_242775


namespace find_a_range_l242_242310

theorem find_a_range (a : ℝ) (x : ℝ) (h1 : a * x < 6) (h2 : (3 * x - 6 * a) / 2 > a / 3 - 1) :
  a ≤ -3 / 2 :=
sorry

end find_a_range_l242_242310


namespace max_students_l242_242220

def num_pens : Nat := 1204
def num_pencils : Nat := 840

theorem max_students (n_pens n_pencils : Nat) (h_pens : n_pens = num_pens) (h_pencils : n_pencils = num_pencils) :
  Nat.gcd n_pens n_pencils = 16 := by
  sorry

end max_students_l242_242220


namespace minimize_y_l242_242777

noncomputable def y (x a b c : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + (x - c)^2

theorem minimize_y (a b c : ℝ) : ∃ x : ℝ, (∀ x0 : ℝ, y x a b c ≤ y x0 a b c) ∧ x = (a + b + c) / 3 :=
by
  sorry

end minimize_y_l242_242777


namespace battery_current_l242_242115

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l242_242115


namespace monotonic_intervals_max_min_values_on_interval_l242_242583

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem monotonic_intervals :
  (∀ x > -2, 0 < (x + 2) * Real.exp x) ∧ (∀ x < -2, (x + 2) * Real.exp x < 0) :=
by
  sorry

theorem max_min_values_on_interval :
  let a := -4
  let b := 0
  let f_a := (-4 + 1) * Real.exp (-4)
  let f_b := (0 + 1) * Real.exp 0
  let f_c := (-2 + 1) * Real.exp (-2)
  (f b = 1) ∧ (f_c = -1 / Real.exp 2) ∧ (f_a < f_b) ∧ (f_a < f_c) ∧ (f_c < f_b) :=
by
  sorry

end monotonic_intervals_max_min_values_on_interval_l242_242583


namespace focus_of_parabola_l242_242213

theorem focus_of_parabola (x y : ℝ) (h : x^2 = -y) : (0, -1/4) = (0, -1/4) :=
by sorry

end focus_of_parabola_l242_242213


namespace find_current_l242_242109

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l242_242109


namespace initial_bees_l242_242369

theorem initial_bees (B : ℕ) (h : B + 7 = 23) : B = 16 :=
by {
  sorry
}

end initial_bees_l242_242369


namespace handshakes_at_gathering_l242_242889

noncomputable def total_handshakes : Nat :=
  let twins := 16
  let triplets := 15
  let handshakes_among_twins := twins * 14 / 2
  let handshakes_among_triplets := 0
  let cross_handshakes := twins * triplets
  handshakes_among_twins + handshakes_among_triplets + cross_handshakes

theorem handshakes_at_gathering : total_handshakes = 352 := 
by
  -- By substituting the values, we can solve and show that the total handshakes equal to 352.
  sorry

end handshakes_at_gathering_l242_242889


namespace part_a_part_b_l242_242245

-- Define the tower of exponents function for convenience
def tower (base : ℕ) (height : ℕ) : ℕ :=
  if height = 0 then 1 else base^(tower base (height - 1))

-- Part a: Tower of 3s with height 99 is greater than Tower of 2s with height 100
theorem part_a : tower 3 99 > tower 2 100 := sorry

-- Part b: Tower of 3s with height 100 is greater than Tower of 3s with height 99
theorem part_b : tower 3 100 > tower 3 99 := sorry

end part_a_part_b_l242_242245


namespace audit_sampling_is_systematic_l242_242648

def is_systematic_sampling (population_size : Nat) (step : Nat) (initial_index : Nat) : Prop :=
  ∃ (k : Nat), ∀ (n : Nat), n ≠ 0 → initial_index + step * (n - 1) ≤ population_size

theorem audit_sampling_is_systematic :
  ∀ (population_size : Nat) (random_index : Nat),
  population_size = 50 * 50 →  -- This represents the total number of invoices (50% of a larger population segment)
  random_index < 50 →         -- Randomly selected index from the first 50 invoices
  is_systematic_sampling population_size 50 random_index := 
by
  intros
  sorry

end audit_sampling_is_systematic_l242_242648


namespace range_of_a_l242_242993

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x + 1 > 2 * x - 2) → (x < a)) → (a ≥ 3) :=
by
  sorry

end range_of_a_l242_242993


namespace rectangle_area_l242_242966

-- Define the conditions as hypotheses in Lean 4
variable (x : ℤ)
variable (area : ℤ := 864)
variable (width : ℤ := x - 12)

-- State the theorem to prove the relation between length and area
theorem rectangle_area (h : x * width = area) : x * (x - 12) = 864 :=
by 
  sorry

end rectangle_area_l242_242966


namespace remainder_of_power_mod_l242_242414

theorem remainder_of_power_mod :
  (5^2023) % 11 = 4 :=
  by
    sorry

end remainder_of_power_mod_l242_242414


namespace smallest_b_l242_242140

theorem smallest_b (b : ℕ) : 
  (b % 4 = 3) → 
  (b % 6 = 5) → 
  (b = 11) := 
by 
  intros h1 h2
  sorry

end smallest_b_l242_242140


namespace max_value_of_a_l242_242713

variable {R : Type*} [LinearOrderedField R]

def det (a b c d : R) : R := a * d - b * c

theorem max_value_of_a (a : R) :
  (∀ x : R, det (x - 1) (a - 2) (a + 1) x ≥ 1) → a ≤ (3 / 2 : R) :=
by
  sorry

end max_value_of_a_l242_242713


namespace binomial_coefficient_30_3_l242_242707

theorem binomial_coefficient_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_coefficient_30_3_l242_242707


namespace days_to_clear_land_l242_242513

-- Definitions of all the conditions
def length_of_land := 200
def width_of_land := 900
def area_cleared_by_one_rabbit_per_day_square_yards := 10
def number_of_rabbits := 100
def conversion_square_yards_to_square_feet := 9
def total_area_of_land := length_of_land * width_of_land
def area_cleared_by_one_rabbit_per_day_square_feet := area_cleared_by_one_rabbit_per_day_square_yards * conversion_square_yards_to_square_feet
def area_cleared_by_all_rabbits_per_day := number_of_rabbits * area_cleared_by_one_rabbit_per_day_square_feet

-- Theorem to prove the number of days required to clear the land
theorem days_to_clear_land :
  total_area_of_land / area_cleared_by_all_rabbits_per_day = 20 := by
  sorry

end days_to_clear_land_l242_242513


namespace find_x_l242_242441

noncomputable def e_squared := Real.exp 2

theorem find_x (x : ℝ) (h : Real.log (x^2 - 5*x + 10) = 2) :
  x = 4.4 ∨ x = 0.6 :=
sorry

end find_x_l242_242441


namespace total_tickets_sold_l242_242676

def price_adult_ticket : ℕ := 7
def price_child_ticket : ℕ := 4
def total_revenue : ℕ := 5100
def adult_tickets_sold : ℕ := 500

theorem total_tickets_sold : 
  ∃ (child_tickets_sold : ℕ), 
    price_adult_ticket * adult_tickets_sold + price_child_ticket * child_tickets_sold = total_revenue ∧
    adult_tickets_sold + child_tickets_sold = 900 :=
by
  sorry

end total_tickets_sold_l242_242676


namespace y_is_multiple_of_12_y_is_multiple_of_3_y_is_multiple_of_4_y_is_multiple_of_6_l242_242188

def y : ℕ := 36 + 48 + 72 + 144 + 216 + 432 + 1296

theorem y_is_multiple_of_12 : y % 12 = 0 := by
  sorry

theorem y_is_multiple_of_3 : y % 3 = 0 := by
  have h := y_is_multiple_of_12
  sorry

theorem y_is_multiple_of_4 : y % 4 = 0 := by
  have h := y_is_multiple_of_12
  sorry

theorem y_is_multiple_of_6 : y % 6 = 0 := by
  have h := y_is_multiple_of_12
  sorry

end y_is_multiple_of_12_y_is_multiple_of_3_y_is_multiple_of_4_y_is_multiple_of_6_l242_242188


namespace right_angled_triangle_area_l242_242942

theorem right_angled_triangle_area 
  (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a + b + c = 18) (h3 : a^2 + b^2 + c^2 = 128) : 
  (1/2) * a * b = 9 :=
by
  -- Proof will be added here
  sorry

end right_angled_triangle_area_l242_242942


namespace triangle_height_l242_242790

theorem triangle_height (A : ℝ) (b : ℝ) (h : ℝ) 
  (hA : A = 615) 
  (hb : b = 123)
  (h_area : A = 0.5 * b * h) : 
  h = 10 :=
by 
  -- Placeholder for the proof
  sorry

end triangle_height_l242_242790


namespace remaining_money_after_purchases_l242_242003

def initial_amount : ℝ := 100
def bread_cost : ℝ := 4
def candy_cost : ℝ := 3
def cereal_cost : ℝ := 6
def fruit_percentage : ℝ := 0.2
def milk_cost_each : ℝ := 4.50
def turkey_fraction : ℝ := 0.25

-- Calculate total spent on initial purchases
def initial_spent : ℝ := bread_cost + (2 * candy_cost) + cereal_cost

-- Remaining amount after initial purchases
def remaining_after_initial : ℝ := initial_amount - initial_spent

-- Spend 20% on fruits
def spent_on_fruits : ℝ := fruit_percentage * remaining_after_initial
def remaining_after_fruits : ℝ := remaining_after_initial - spent_on_fruits

-- Spend on two gallons of milk
def spent_on_milk : ℝ := 2 * milk_cost_each
def remaining_after_milk : ℝ := remaining_after_fruits - spent_on_milk

-- Spend 1/4 on turkey
def spent_on_turkey : ℝ := turkey_fraction * remaining_after_milk
def final_remaining : ℝ := remaining_after_milk - spent_on_turkey

theorem remaining_money_after_purchases : final_remaining = 43.65 := by
  sorry

end remaining_money_after_purchases_l242_242003


namespace num_routes_A_to_B_in_3x3_grid_l242_242173

theorem num_routes_A_to_B_in_3x3_grid : 
  let total_moves := 10
  let right_moves := 5
  let down_moves := 5
  let routes_count := Nat.choose total_moves right_moves
  routes_count = 252 := 
by
  let total_moves := 10
  let right_moves := 5
  let down_moves := 5
  let routes_count := Nat.choose total_moves right_moves
  have fact_10 : 10! = 3628800 := by sorry
  have fact_5  : 5!  = 120 := by sorry
  have comb_10_5: Nat.choose 10 5 = 252 := by sorry
  exact comb_10_5

end num_routes_A_to_B_in_3x3_grid_l242_242173


namespace system_solution_b_l242_242314

theorem system_solution_b (x y b : ℚ) 
  (h1 : 4 * x + 2 * y = b) 
  (h2 : 3 * x + 7 * y = 3 * b) 
  (hy : y = 3) : 
  b = 22 / 3 := 
by
  sorry

end system_solution_b_l242_242314


namespace fraction_value_l242_242118

theorem fraction_value (a b : ℚ) (h₁ : b / (a - 2) = 3 / 4) (h₂ : b / (a + 9) = 5 / 7) : b / a = 165 / 222 := 
by sorry

end fraction_value_l242_242118


namespace current_when_resistance_12_l242_242103

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l242_242103


namespace relay_race_arrangements_l242_242474

noncomputable def number_of_arrangements (athletes : Finset ℕ) (a b : ℕ) : ℕ :=
  (athletes.erase a).card.factorial * ((athletes.erase b).card.factorial - 2) * (athletes.card.factorial / ((athletes.card - 4).factorial)) / 4

theorem relay_race_arrangements :
  let athletes := {0, 1, 2, 3, 4, 5}
  number_of_arrangements athletes 0 1 = 252 := 
by
  sorry

end relay_race_arrangements_l242_242474


namespace total_protest_days_l242_242970

-- Definitions for the problem conditions
def first_protest_days : ℕ := 4
def second_protest_days : ℕ := first_protest_days + (first_protest_days / 4)

-- The proof statement
theorem total_protest_days : first_protest_days + second_protest_days = 9 := sorry

end total_protest_days_l242_242970


namespace rectangle_area_correct_l242_242360

noncomputable def rectangle_area (x: ℚ) : ℚ :=
  let length := 5 * x - 18
  let width := 25 - 4 * x
  length * width

theorem rectangle_area_correct (x: ℚ) (h1: 3.6 < x) (h2: x < 6.25) :
  rectangle_area (43 / 9) = (2809 / 81) := 
  by
    sorry

end rectangle_area_correct_l242_242360


namespace num_pairs_sold_l242_242005

theorem num_pairs_sold : 
  let total_amount : ℤ := 735
  let avg_price : ℝ := 9.8
  let num_pairs : ℝ := total_amount / avg_price
  num_pairs = 75 :=
by
  let total_amount : ℤ := 735
  let avg_price : ℝ := 9.8
  let num_pairs : ℝ := total_amount / avg_price
  exact sorry

end num_pairs_sold_l242_242005


namespace solution_set_f_l242_242581

noncomputable def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem solution_set_f (x : ℝ)  : (f(x) ≥ x^2 - 8x + 15) ↔ (5 - real.sqrt 3 ≤ x ∧ x ≤ 6) :=
sorry

end solution_set_f_l242_242581


namespace walking_speed_l242_242872

noncomputable def bridge_length : ℝ := 2500  -- length of the bridge in meters
noncomputable def crossing_time_minutes : ℝ := 15  -- time to cross the bridge in minutes
noncomputable def conversion_factor_time : ℝ := 1 / 60  -- factor to convert minutes to hours
noncomputable def conversion_factor_distance : ℝ := 1 / 1000  -- factor to convert meters to kilometers

theorem walking_speed (bridge_length crossing_time_minutes conversion_factor_time conversion_factor_distance : ℝ) : 
  bridge_length = 2500 → 
  crossing_time_minutes = 15 → 
  conversion_factor_time = 1 / 60 → 
  conversion_factor_distance = 1 / 1000 → 
  (bridge_length * conversion_factor_distance) / (crossing_time_minutes * conversion_factor_time) = 10 := 
by
  sorry

end walking_speed_l242_242872


namespace smallest_three_digit_divisible_by_4_and_5_l242_242521

theorem smallest_three_digit_divisible_by_4_and_5 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (m % 4 = 0) ∧ (m % 5 = 0) → m ≥ n →
n = 100 :=
sorry

end smallest_three_digit_divisible_by_4_and_5_l242_242521


namespace range_of_x_l242_242566

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 + (a - 4) * x + 4 - 2 * a

theorem range_of_x (a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  ∀ x : ℝ, (f x a > 0) ↔ (x < 1 ∨ x > 3) :=
by
  intro x
  sorry

end range_of_x_l242_242566


namespace steven_more_peaches_than_apples_l242_242458

-- Definitions
def apples_steven := 11
def peaches_steven := 18

-- Theorem statement
theorem steven_more_peaches_than_apples : (peaches_steven - apples_steven) = 7 := by 
  sorry

end steven_more_peaches_than_apples_l242_242458


namespace g_at_2_l242_242634

-- Assuming g is a function from ℝ to ℝ such that it satisfies the given condition.
def g : ℝ → ℝ := sorry

-- Condition of the problem
axiom g_condition : ∀ x : ℝ, g (2 ^ x) + x * g (2 ^ (-x)) = 2

-- The statement we want to prove
theorem g_at_2 : g (2) = 0 :=
by
  sorry

end g_at_2_l242_242634


namespace smallest_perimeter_l242_242811

noncomputable def smallest_possible_perimeter : ℕ :=
  let n := 3
  n + (n + 1) + (n + 2)

theorem smallest_perimeter (n : ℕ) (h : n > 2) (ineq1 : n + (n + 1) > (n + 2)) 
  (ineq2 : n + (n + 2) > (n + 1)) (ineq3 : (n + 1) + (n + 2) > n) : 
  smallest_possible_perimeter = 12 :=
by
  sorry

end smallest_perimeter_l242_242811


namespace current_when_resistance_12_l242_242102

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l242_242102


namespace terminating_decimal_l242_242930

theorem terminating_decimal : (45 / (2^2 * 5^3) : ℚ) = 0.090 :=
by
  sorry

end terminating_decimal_l242_242930


namespace battery_current_l242_242025

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l242_242025


namespace part1_part2_l242_242786

theorem part1 (a b h3 : ℝ) (C : ℝ) (h : 1 / h3 = 1 / a + 1 / b) : C ≤ 120 :=
sorry

theorem part2 (a b m3 : ℝ) (C : ℝ) (h : 1 / m3 = 1 / a + 1 / b) : C ≥ 120 :=
sorry

end part1_part2_l242_242786


namespace prob_diff_colors_correct_l242_242947

noncomputable def total_outcomes : ℕ :=
  let balls_pocket1 := 2 + 3 + 5
  let balls_pocket2 := 2 + 4 + 4
  balls_pocket1 * balls_pocket2

noncomputable def favorable_outcomes_same_color : ℕ :=
  let white_balls := 2 * 2
  let red_balls := 3 * 4
  let yellow_balls := 5 * 4
  white_balls + red_balls + yellow_balls

noncomputable def prob_same_color : ℚ :=
  favorable_outcomes_same_color / total_outcomes

noncomputable def prob_different_color : ℚ :=
  1 - prob_same_color

theorem prob_diff_colors_correct :
  prob_different_color = 16 / 25 :=
by sorry

end prob_diff_colors_correct_l242_242947


namespace inequality_for_average_daily_work_l242_242669

-- Given
def total_earthwork : ℕ := 300
def completed_earthwork_first_day : ℕ := 60
def scheduled_days : ℕ := 6
def days_ahead : ℕ := 2

-- To Prove
theorem inequality_for_average_daily_work (x : ℕ) :
  scheduled_days - days_ahead - 1 > 0 →
  (total_earthwork - completed_earthwork_first_day) ≤ x * (scheduled_days - days_ahead - 1) :=
by
  sorry

end inequality_for_average_daily_work_l242_242669


namespace ball_bounces_below_2_feet_l242_242392

theorem ball_bounces_below_2_feet :
  ∃ k : ℕ, 500 * (2 / 3 : ℝ) ^ k < 2 ∧ ∀ n < k, 500 * (2 / 3 : ℝ) ^ n ≥ 2 :=
by
  sorry

end ball_bounces_below_2_feet_l242_242392


namespace battery_current_l242_242084

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l242_242084


namespace paper_folding_ratio_l242_242126

theorem paper_folding_ratio :
  ∃ (side length small_perim large_perim : ℕ), 
    side_length = 6 ∧ 
    small_perim = 2 * (3 + 3) ∧ 
    large_perim = 2 * (6 + 3) ∧ 
    small_perim / large_perim = 2 / 3 :=
by sorry

end paper_folding_ratio_l242_242126


namespace binom_30_3_l242_242698

def binomial_coefficient (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binom_30_3 : binomial_coefficient 30 3 = 4060 := by
  sorry

end binom_30_3_l242_242698


namespace trigonometric_identity_l242_242953

theorem trigonometric_identity (α : ℝ) (h : Real.cos α + Real.sin α = 2 / 3) :
  (Real.sqrt 2 * Real.sin (2 * α - Real.pi / 4) + 1) / (1 + Real.tan α) = - 5 / 9 :=
sorry

end trigonometric_identity_l242_242953


namespace binom_60_3_l242_242902

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l242_242902


namespace cylinder_volume_increase_l242_242240

theorem cylinder_volume_increase (r h : ℝ) (V : ℝ) : 
  V = π * r^2 * h → 
  (3 * h) * (2 * r)^2 * π = 12 * V := by
    sorry

end cylinder_volume_increase_l242_242240


namespace range_of_m_l242_242570

noncomputable def f (x m : ℝ) : ℝ :=
  x^2 - 2 * m * x + m + 2

theorem range_of_m
  (m : ℝ)
  (h1 : ∃ a b : ℝ, f a m = 0 ∧ f b m = 0 ∧ a ≠ b)
  (h2 : ∀ x : ℝ, x ≥ 1 → 2*x - 2*m ≥ 0) :
  m < -1 :=
sorry

end range_of_m_l242_242570


namespace sqrt_9_eq_pm3_l242_242382

theorem sqrt_9_eq_pm3 : ∃ x : ℤ, x^2 = 9 ∧ (x = 3 ∨ x = -3) :=
by sorry

end sqrt_9_eq_pm3_l242_242382


namespace inlet_pipe_rate_l242_242543

theorem inlet_pipe_rate (capacity : ℕ) (t_empty : ℕ) (t_with_inlet : ℕ) (R_out : ℕ) :
  capacity = 6400 →
  t_empty = 10 →
  t_with_inlet = 16 →
  R_out = capacity / t_empty →
  (R_out - (capacity / t_with_inlet)) / 60 = 4 :=
by
  intros h1 h2 h3 h4 
  sorry

end inlet_pipe_rate_l242_242543


namespace possible_arrangements_count_l242_242257

-- Define students as a type
inductive Student
| A | B | C | D | E | F

open Student

-- Define Club as a type
inductive Club
| A | B | C

open Club

-- Define the arrangement constraints
structure Arrangement :=
(assignment : Student → Club)
(club_size : Club → Nat)
(A_and_B_same_club : assignment A = assignment B)
(C_and_D_diff_clubs : assignment C ≠ assignment D)
(club_A_size : club_size A = 3)
(all_clubs_nonempty : ∀ c : Club, club_size c > 0)

-- Define the possible number of arrangements
def arrangement_count (a : Arrangement) : Nat := sorry

-- Theorem stating the number of valid arrangements
theorem possible_arrangements_count : ∃ a : Arrangement, arrangement_count a = 24 := sorry

end possible_arrangements_count_l242_242257


namespace Duke_three_pointers_impossible_l242_242336

theorem Duke_three_pointers_impossible (old_record : ℤ)
  (points_needed_to_tie : ℤ)
  (points_broken_record : ℤ)
  (free_throws : ℕ)
  (regular_baskets : ℕ)
  (three_pointers : ℕ)
  (normal_three_pointers_per_game : ℕ)
  (max_attempts : ℕ)
  (last_minutes : ℕ)
  (points_per_free_throw : ℤ)
  (points_per_regular_basket : ℤ)
  (points_per_three_pointer : ℤ) :
  free_throws = 5 → regular_baskets = 4 → normal_three_pointers_per_game = 2 → max_attempts = 10 → 
  points_per_free_throw = 1 → points_per_regular_basket = 2 → points_per_three_pointer = 3 →
  old_record = 257 → points_needed_to_tie = 17 → points_broken_record = 5 →
  (free_throws + regular_baskets + three_pointers ≤ max_attempts) →
  last_minutes = 6 → 
  ¬(free_throws + regular_baskets + (points_needed_to_tie + points_broken_record - 
  (free_throws * points_per_free_throw + regular_baskets * points_per_regular_basket)) / points_per_three_pointer ≤ max_attempts) := sorry

end Duke_three_pointers_impossible_l242_242336


namespace min_value_m_l242_242158

open Finset

def Ai_sets (A : Finset ℕ) : Finset (Finset ℕ) :=
  { B : Finset ℕ | B.card = 5 }

theorem min_value_m 
    (A : Finset ℕ)
    (Ai : Fin n → Finset ℕ)
    (hAi : ∀ i, Ai i ∈ Ai_sets A)
    (hInter : ∀ i j, i ≠ j → (Ai i ∩ Ai j).card ≥ 2)
    (hUnion : A = Finset.bUnion Finset.univ Ai)
    (ki : A → ℕ)
    (hki : ∀ x, ki x = (Finset.filter (λ i, x ∈ Ai i) Finset.univ).card)
    (sum_ki : (A.sum ki) = 50)
    (sum_Cki2 : A.sum (λ x, (ki x).choose 2) ≥ 90) : 
  ∃ m, m = (A.image ki).max' sorry ∧ m ≥ 5 :=
by {
  sorry
}

end min_value_m_l242_242158


namespace child_ticket_cost_l242_242232

noncomputable def cost_of_child_ticket : ℝ := 3.50

theorem child_ticket_cost
  (adult_ticket_price : ℝ)
  (total_tickets : ℕ)
  (total_cost : ℝ)
  (adult_tickets_bought : ℕ)
  (adult_ticket_price_eq : adult_ticket_price = 5.50)
  (total_tickets_bought_eq : total_tickets = 21)
  (total_cost_eq : total_cost = 83.50)
  (adult_tickets_count : adult_tickets_bought = 5) :
  cost_of_child_ticket = 3.50 :=
by
  sorry

end child_ticket_cost_l242_242232


namespace rectangle_to_square_l242_242642

theorem rectangle_to_square (length width : ℕ) (h1 : 2 * (length + width) = 40) (h2 : length - 8 = width + 2) :
  width + 2 = 7 :=
by {
  -- Proof goes here
  sorry
}

end rectangle_to_square_l242_242642


namespace battery_current_l242_242053

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l242_242053


namespace trains_clear_in_correct_time_l242_242234

noncomputable def time_to_clear (length1 length2 : ℝ) (speed1_kmph speed2_kmph : ℝ) : ℝ :=
  let speed1_mps := speed1_kmph * 1000 / 3600
  let speed2_mps := speed2_kmph * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := length1 + length2
  total_distance / relative_speed

-- The lengths of the trains
def length1 : ℝ := 151
def length2 : ℝ := 165

-- The speeds of the trains in km/h
def speed1_kmph : ℝ := 80
def speed2_kmph : ℝ := 65

-- The correct answer
def correct_time : ℝ := 7.844

theorem trains_clear_in_correct_time :
  time_to_clear length1 length2 speed1_kmph speed2_kmph = correct_time :=
by
  -- Skipping proof
  sorry

end trains_clear_in_correct_time_l242_242234


namespace ab_cd_zero_l242_242344

theorem ab_cd_zero (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : ac + bd = 0) : 
  ab + cd = 0 := 
sorry

end ab_cd_zero_l242_242344


namespace sum_of_five_consecutive_odd_numbers_l242_242226

theorem sum_of_five_consecutive_odd_numbers (x : ℤ) : 
  (x - 4) + (x - 2) + x + (x + 2) + (x + 4) = 5 * x :=
by
  sorry

end sum_of_five_consecutive_odd_numbers_l242_242226


namespace binom_60_3_eq_34220_l242_242911

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l242_242911


namespace range_of_x_l242_242326

theorem range_of_x (x : ℝ) : (∀ t : ℝ, -1 ≤ t ∧ t ≤ 3 → x^2 - (t^2 + t - 3) * x + t^2 * (t - 3) > 0) ↔ (x < -4 ∨ x > 9) :=
by
  sorry

end range_of_x_l242_242326


namespace bucket_full_weight_l242_242001

theorem bucket_full_weight (x y c d : ℝ)
  (h1 : x + 3 / 4 * y = c)
  (h2 : x + 1 / 3 * y = d) :
  x + y = (8 / 5) * c - (7 / 5) * d :=
by
  sorry

end bucket_full_weight_l242_242001


namespace total_pints_l242_242143

-- Define the given conditions as constants
def annie_picked : Int := 8
def kathryn_picked : Int := annie_picked + 2
def ben_picked : Int := kathryn_picked - 3

-- State the main theorem to prove
theorem total_pints : annie_picked + kathryn_picked + ben_picked = 25 := by
  sorry

end total_pints_l242_242143


namespace cricket_team_matches_l242_242180

theorem cricket_team_matches 
  (M : ℕ) (W : ℕ) 
  (h1 : W = 20 * M / 100) 
  (h2 : (W + 80) * 100 = 52 * M) : 
  M = 250 :=
by
  sorry

end cricket_team_matches_l242_242180


namespace total_leaves_l242_242341

theorem total_leaves (ferns fronds leaves : ℕ) (h1 : ferns = 12) (h2 : fronds = 15) (h3 : leaves = 45) :
  ferns * fronds * leaves = 8100 :=
by
  sorry

end total_leaves_l242_242341


namespace locus_is_semicircle_l242_242229

noncomputable def locus_of_point_C (O A B C : Point) (k : ℝ) 
  (h_right_angle : is_right_angle O A B)
  (h_const_sum : (dist O A) + (dist O B) = k)
  (h_diameter : is_diameter O A B C) 
  (OC_parallel_AB : parallel (line_through O C) (line_through A B)) : Set Point :=
  { C : Point | exists A B, (dist O A + dist O B = k) ∧ (is_on_circle_with_diameter A B C) ∧ parallel (line_through O C) (line_through A B)}

theorem locus_is_semicircle {O A B C : Point} (k : ℝ)
  (h_right_angle : is_right_angle O A B)
  (h_const_sum : (dist O A) + (dist O B) = k)
  (h_diameter : is_diameter O A B C) 
  (OC_parallel_AB : parallel (line_through O C) (line_through A B)) :
  locus_of_point_C O A B C k h_right_angle h_const_sum h_diameter OC_parallel_AB = 
  { C : Point | is_on_semicircle_above_line {
                      center := midpoint A B,
                      radius := dist (midpoint A B) A,
                      endpoints := (A, B),
                      passes_through := O 
                                                    } } :=
sorry

end locus_is_semicircle_l242_242229


namespace large_pile_toys_l242_242595

theorem large_pile_toys (x y : ℕ) (h1 : x + y = 120) (h2 : y = 2 * x) : y = 80 := by
  sorry

end large_pile_toys_l242_242595


namespace find_y_l242_242564

def v := λ (y : ℝ), ![2, y]
def w := ![5, -1]
def proj_w_v (y : ℝ) := (inner (v y) w / inner w w) • w

theorem find_y (y : ℝ) (h : proj_w_v y = ![3, -0.6]) : y = -5.6 :=
by
  have h1 : inner (v y) w = 10 - y := by sorry
  have h2 : inner w w = 26 := by sorry
  have proj_formula : proj_w_v y = (10 - y) / 26 • w := by sorry
  rw [proj_formula] at h
  have eq1 : (10 - y) / 26 * 5 = 3 := by sorry
  have eq2 : y = -5.6 := by sorry
  exact eq2

end find_y_l242_242564


namespace marbles_each_friend_is_16_l242_242319

-- Define the initial condition
def initial_marbles : ℕ := 100

-- Define the marbles Harold kept for himself
def kept_marbles : ℕ := 20

-- Define the number of friends Harold shared the marbles with
def num_friends : ℕ := 5

-- Define the marbles each friend receives
def marbles_per_friend (initial kept : ℕ) (friends : ℕ) : ℕ :=
  (initial - kept) / friends

-- Prove that each friend gets 16 marbles
theorem marbles_each_friend_is_16 : marbles_per_friend initial_marbles kept_marbles num_friends = 16 :=
by
  unfold initial_marbles kept_marbles num_friends marbles_per_friend
  exact Nat.mk_eq Nat.zero 16 sorry

end marbles_each_friend_is_16_l242_242319


namespace moles_ethane_and_hexachloroethane_l242_242440

-- Define the conditions
def balanced_eq (a b c d : ℕ) : Prop :=
  a * 6 = b ∧ d * 6 = c

-- The main theorem statement
theorem moles_ethane_and_hexachloroethane (moles_Cl2 : ℕ) :
  moles_Cl2 = 18 → balanced_eq 1 1 18 3 :=
by
  sorry

end moles_ethane_and_hexachloroethane_l242_242440


namespace problem_part1_problem_part2_l242_242435

noncomputable def f (a : ℝ) (x : ℝ) := 2 * Real.log x + a / x
noncomputable def g (a : ℝ) (x : ℝ) := (x / 2) * f a x - a * x^2 - x

theorem problem_part1 (a : ℝ) :
  (∀ x : ℝ, f a x = 0 → x > 0) ↔ 0 < a ∧ a < 2/Real.exp 1 := sorry

theorem problem_part2 (a : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : g a x₁ = 0) (h₃ : g a x₂ = 0) :
  0 < a ∧ a < 2/Real.exp 1 → Real.log x₁ + 2 * Real.log x₂ > 3 := sorry

end problem_part1_problem_part2_l242_242435


namespace certain_number_divided_by_10_l242_242757
-- Broad import to bring in necessary libraries

-- Define the constants and hypotheses
variable (x : ℝ)
axiom condition : 5 * x = 100

-- Theorem to prove the required equality
theorem certain_number_divided_by_10 : (x / 10) = 2 :=
by
  -- The proof is skipped by sorry
  sorry

end certain_number_divided_by_10_l242_242757


namespace seashells_total_correct_l242_242476

def total_seashells (red_shells green_shells other_shells : ℕ) : ℕ :=
  red_shells + green_shells + other_shells

theorem seashells_total_correct :
  total_seashells 76 49 166 = 291 :=
by
  sorry

end seashells_total_correct_l242_242476


namespace current_at_R_12_l242_242045

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l242_242045


namespace current_at_resistance_12_l242_242060

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l242_242060


namespace sin_neg_270_eq_one_l242_242290

theorem sin_neg_270_eq_one : Real.sin (-(270 : ℝ) * (Real.pi / 180)) = 1 := by
  sorry

end sin_neg_270_eq_one_l242_242290


namespace current_value_l242_242017

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l242_242017


namespace martha_total_butterflies_l242_242470

variable (Yellow Blue Black : ℕ)

def butterfly_equations (Yellow Blue Black : ℕ) : Prop :=
  (Blue = 2 * Yellow) ∧ (Blue = 6) ∧ (Black = 10)

theorem martha_total_butterflies 
  (h : butterfly_equations Yellow Blue Black) : 
  (Yellow + Blue + Black = 19) :=
by
  sorry

end martha_total_butterflies_l242_242470


namespace sum_p_q_l242_242958

theorem sum_p_q (p q : ℚ) (g : ℚ → ℚ) (h : g = λ x => (x + 2) / (x^2 + p * x + q))
  (h_asymp1 : ∀ {x}, x = -1 → (x^2 + p * x + q) = 0)
  (h_asymp2 : ∀ {x}, x = 3 → (x^2 + p * x + q) = 0) :
  p + q = -5 := by
  sorry

end sum_p_q_l242_242958


namespace guests_not_eating_brownies_ala_mode_l242_242806

theorem guests_not_eating_brownies_ala_mode (total_brownies : ℕ) (eaten_brownies : ℕ) (eaten_scoops : ℕ)
    (scoops_per_serving : ℕ) (scoops_per_tub : ℕ) (tubs_eaten : ℕ) : 
    total_brownies = 32 → eaten_brownies = 28 → eaten_scoops = 48 → scoops_per_serving = 2 → scoops_per_tub = 8 → tubs_eaten = 6 → (eaten_scoops - eaten_brownies * scoops_per_serving) / scoops_per_serving = 4 :=
by
  intros
  sorry

end guests_not_eating_brownies_ala_mode_l242_242806


namespace cost_of_iphone_l242_242134

def trade_in_value : ℕ := 240
def weekly_earnings : ℕ := 80
def weeks_worked : ℕ := 7
def total_earnings := weekly_earnings * weeks_worked
def total_money := total_earnings + trade_in_value
def new_iphone_cost : ℕ := 800

theorem cost_of_iphone :
  total_money = new_iphone_cost := by
  sorry

end cost_of_iphone_l242_242134


namespace num_possible_measures_of_A_l242_242639

-- Given conditions
variables (A B : ℕ)
variables (k : ℕ) (hk : k ≥ 1)
variables (hab : A + B = 180)
variables (ha : A = k * B)

-- The proof statement
theorem num_possible_measures_of_A : 
  ∃ (n : ℕ), n = 17 ∧ ∀ k, (k + 1) ∣ 180 ∧ k ≥ 1 → n = 17 := 
begin
  sorry
end

end num_possible_measures_of_A_l242_242639


namespace current_at_R_12_l242_242048

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l242_242048


namespace solve_quadratic_1_solve_quadratic_2_l242_242479

theorem solve_quadratic_1 (x : ℝ) : 2 * x^2 - 7 * x - 1 = 0 ↔ 
  (x = (7 + Real.sqrt 57) / 4 ∨ x = (7 - Real.sqrt 57) / 4) := 
by 
  sorry

theorem solve_quadratic_2 (x : ℝ) : (2 * x - 3)^2 = 10 * x - 15 ↔ 
  (x = 3 / 2 ∨ x = 4) := 
by 
  sorry

end solve_quadratic_1_solve_quadratic_2_l242_242479


namespace book_has_50_pages_l242_242954

noncomputable def sentences_per_hour : ℕ := 200
noncomputable def hours_to_read : ℕ := 50
noncomputable def sentences_per_paragraph : ℕ := 10
noncomputable def paragraphs_per_page : ℕ := 20

theorem book_has_50_pages :
  (sentences_per_hour * hours_to_read) / sentences_per_paragraph / paragraphs_per_page = 50 :=
by
  sorry

end book_has_50_pages_l242_242954


namespace total_number_of_songs_is_30_l242_242822

-- Define the number of country albums and pop albums
def country_albums : ℕ := 2
def pop_albums : ℕ := 3

-- Define the number of songs per album
def songs_per_album : ℕ := 6

-- Define the total number of albums
def total_albums : ℕ := country_albums + pop_albums

-- Define the total number of songs
def total_songs : ℕ := total_albums * songs_per_album

-- Prove that the total number of songs is 30
theorem total_number_of_songs_is_30 : total_songs = 30 := 
sorry

end total_number_of_songs_is_30_l242_242822


namespace profit_of_150_cents_requires_120_oranges_l242_242533

def cost_price_per_orange := 15 / 4  -- cost price per orange in cents
def selling_price_per_orange := 30 / 6  -- selling price per orange in cents
def profit_per_orange := selling_price_per_orange - cost_price_per_orange  -- profit per orange in cents
def required_oranges_to_make_profit := 150 / profit_per_orange  -- number of oranges to get 150 cents of profit

theorem profit_of_150_cents_requires_120_oranges :
  required_oranges_to_make_profit = 120 :=
by
  -- the actual proof will follow here
  sorry

end profit_of_150_cents_requires_120_oranges_l242_242533


namespace current_at_resistance_12_l242_242034

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l242_242034


namespace original_slices_proof_l242_242719

def original_slices (andy_consumption toast_slices leftover_slice: ℕ) : ℕ :=
  andy_consumption + toast_slices + leftover_slice

theorem original_slices_proof :
  original_slices (3 * 2) (10 * 2) 1 = 27 :=
by
  sorry

end original_slices_proof_l242_242719


namespace binomial_60_3_eq_34220_l242_242916

open Nat

theorem binomial_60_3_eq_34220 : choose 60 3 = 34220 := 
by
  sorry

end binomial_60_3_eq_34220_l242_242916


namespace find_current_when_resistance_is_12_l242_242043

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l242_242043


namespace battery_current_when_resistance_12_l242_242075

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l242_242075


namespace finite_non_friends_iff_l242_242938

def isFriend (u n : ℕ) : Prop :=
  ∃ N : ℕ, N % n = 0 ∧ (N.digits 10).sum = u

theorem finite_non_friends_iff (n : ℕ) : (∃ᶠ u in at_top, ¬ isFriend u n) ↔ ¬ (3 ∣ n) := 
by
  sorry

end finite_non_friends_iff_l242_242938


namespace selling_price_correct_l242_242539

namespace Shopkeeper

def costPrice : ℝ := 1500
def profitPercentage : ℝ := 20
def expectedSellingPrice : ℝ := 1800

theorem selling_price_correct
  (cp : ℝ := costPrice)
  (pp : ℝ := profitPercentage) :
  cp * (1 + pp / 100) = expectedSellingPrice :=
by
  sorry

end Shopkeeper

end selling_price_correct_l242_242539


namespace staffing_ways_l242_242283

def total_resumes : ℕ := 30
def unsuitable_resumes : ℕ := 10
def suitable_resumes : ℕ := total_resumes - unsuitable_resumes
def position_count : ℕ := 5

theorem staffing_ways :
  20 * 19 * 18 * 17 * 16 = 1860480 := by
  sorry

end staffing_ways_l242_242283


namespace quadratic_inequality_l242_242178

theorem quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * a * x + 4 > 0) ↔ -2 < a ∧ a < 2 :=
by
  sorry

end quadratic_inequality_l242_242178


namespace binomial_30_3_l242_242703

-- Defining the binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Statement of the problem in Lean 4
theorem binomial_30_3 : binomial 30 3 = 12180 :=
by
  sorry

end binomial_30_3_l242_242703


namespace cannot_be_right_angle_triangle_l242_242683

-- Definition of the converse of the Pythagorean theorem
def is_right_angle_triangle (a b c : ℕ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2

-- Definition to check if a given set of sides cannot form a right-angled triangle
def cannot_form_right_angle_triangle (a b c : ℕ) : Prop :=
  ¬ is_right_angle_triangle a b c

-- Given sides of the triangle option D
theorem cannot_be_right_angle_triangle : cannot_form_right_angle_triangle 3 4 6 :=
  by sorry

end cannot_be_right_angle_triangle_l242_242683


namespace problem_220_l242_242004

variables (x y : ℝ)

theorem problem_220 (h1 : x + y = 10) (h2 : (x * y) / (x^2) = -3 / 2) :
  x = -20 ∧ y = 30 :=
by
  sorry

end problem_220_l242_242004


namespace expected_value_of_win_is_3point5_l242_242839

noncomputable def expected_value_win : ℝ := 
  ∑ n in Finset.range 8, (1 / 8 : ℝ) * (8 - n)

theorem expected_value_of_win_is_3point5 : expected_value_win = 3.5 := 
by 
  sorry

end expected_value_of_win_is_3point5_l242_242839


namespace binom_60_3_l242_242922

theorem binom_60_3 : nat.choose 60 3 = 34220 :=
by
  -- Proof goes here.
  sorry

end binom_60_3_l242_242922


namespace modulus_z_eq_sqrt_10_l242_242941

noncomputable def z := (10 * Complex.I) / (3 + Complex.I)

theorem modulus_z_eq_sqrt_10 : Complex.abs z = Real.sqrt 10 := by
  sorry

end modulus_z_eq_sqrt_10_l242_242941


namespace machine_bottle_caps_l242_242997

variable (A_rate : ℕ)
variable (A_time : ℕ)
variable (B_rate : ℕ)
variable (B_time : ℕ)
variable (C_rate : ℕ)
variable (C_time : ℕ)
variable (D_rate : ℕ)
variable (D_time : ℕ)
variable (E_rate : ℕ)
variable (E_time : ℕ)

def A_bottles := A_rate * A_time
def B_bottles := B_rate * B_time
def C_bottles := C_rate * C_time
def D_bottles := D_rate * D_time
def E_bottles := E_rate * E_time

theorem machine_bottle_caps (hA_rate : A_rate = 24)
                            (hA_time : A_time = 10)
                            (hB_rate : B_rate = A_rate - 3)
                            (hB_time : B_time = 12)
                            (hC_rate : C_rate = B_rate + 6)
                            (hC_time : C_time = 15)
                            (hD_rate : D_rate = C_rate - 4)
                            (hD_time : D_time = 8)
                            (hE_rate : E_rate = D_rate + 5)
                            (hE_time : E_time = 5) :
  A_bottles A_rate A_time = 240 ∧ 
  B_bottles B_rate B_time = 252 ∧ 
  C_bottles C_rate C_time = 405 ∧ 
  D_bottles D_rate D_time = 184 ∧ 
  E_bottles E_rate E_time = 140 := by
    sorry

end machine_bottle_caps_l242_242997


namespace current_value_l242_242068

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l242_242068


namespace range_of_m_l242_242814

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x / Real.log 2

theorem range_of_m :
  (∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → 2^x - Real.log x / Real.log (1/2) + m ≤ 0) →
  m ≤ -5 :=
sorry

end range_of_m_l242_242814


namespace circle_area_l242_242484

theorem circle_area (C : ℝ) (hC : C = 24) : ∃ (A : ℝ), A = 144 / π :=
by
  sorry

end circle_area_l242_242484


namespace tom_age_ratio_l242_242999

variable (T M : ℕ)
variable (h1 : T = T) -- Tom's age is equal to the sum of the ages of his four children
variable (h2 : T - M = 3 * (T - 4 * M)) -- M years ago, Tom's age was three times the sum of his children's ages then

theorem tom_age_ratio : (T / M) = 11 / 2 := 
by
  sorry

end tom_age_ratio_l242_242999


namespace solutions_of_quadratic_l242_242323

theorem solutions_of_quadratic (c : ℝ) (h : ∀ α β : ℝ, 
  (α^2 - 3*α + c = 0 ∧ β^2 - 3*β + c = 0) → 
  ( (-α)^2 + 3*(-α) - c = 0 ∨ (-β)^2 + 3*(-β) - c = 0 ) ) :
  ∃ α β : ℝ, (α = 0 ∧ β = 3) ∨ (α = 3 ∧ β = 0) :=
by
  sorry

end solutions_of_quadratic_l242_242323


namespace rational_root_even_denominator_l242_242963

theorem rational_root_even_denominator
  (a b c : ℤ)
  (sum_ab_even : (a + b) % 2 = 0)
  (c_odd : c % 2 = 1) :
  ∀ (p q : ℤ), (q ≠ 0) → (IsRationalRoot : a * (p * p) + b * p * q + c * (q * q) = 0) →
    gcd p q = 1 → q % 2 = 0 :=
by
  sorry

end rational_root_even_denominator_l242_242963


namespace area_ratio_none_of_these_l242_242989

theorem area_ratio_none_of_these (h r a : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) (a_pos : 0 < a) (h_square_a_square : h^2 > a^2) :
  ¬ (∃ ratio, ratio = (π * r / (h + r)) ∨
               ratio = (π * r^2 / (a + h)) ∨
               ratio = (π * a * r / (h + 2 * r)) ∨
               ratio = (π * r / (a + r))) :=
by sorry

end area_ratio_none_of_these_l242_242989


namespace quadratic_complete_square_l242_242225

theorem quadratic_complete_square :
  ∃ a b c : ℤ, (8 * x^2 - 48 * x - 320 = a * (x + b)^2 + c) ∧ (a + b + c = -387) :=
sorry

end quadratic_complete_square_l242_242225


namespace total_revenue_l242_242208

def price_federal := 50
def price_state := 30
def price_quarterly := 80
def num_federal := 60
def num_state := 20
def num_quarterly := 10

theorem total_revenue : (num_federal * price_federal + num_state * price_state + num_quarterly * price_quarterly) = 4400 := by
  sorry

end total_revenue_l242_242208


namespace expected_value_is_350_l242_242852

noncomputable def expected_value_of_winnings : ℚ :=
  ((1 / 8) * (8 - 1) + (1 / 8) * (8 - 2) + (1 / 8) * (8 - 3) + (1 / 8) * (8 - 4) +
  (1 / 8) * (8 - 5) + (1 / 8) * (8 - 6) + (1 / 8) * (8 - 7) + (1 / 8) * (8 - 8))

theorem expected_value_is_350 :
  expected_value_of_winnings = 3.50 := by
  sorry

end expected_value_is_350_l242_242852


namespace arithmetic_geometric_mean_l242_242626

theorem arithmetic_geometric_mean (a b : ℝ) 
  (h1 : (a + b) / 2 = 20) 
  (h2 : Real.sqrt (a * b) = Real.sqrt 135) : 
  a^2 + b^2 = 1330 :=
by
  sorry

end arithmetic_geometric_mean_l242_242626


namespace current_value_l242_242065

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l242_242065


namespace treasure_chest_age_l242_242394

theorem treasure_chest_age (n : ℕ) (h : n = 3 * 8^2 + 4 * 8^1 + 7 * 8^0) : n = 231 :=
by
  sorry

end treasure_chest_age_l242_242394


namespace avg_five_probability_l242_242296

/- Define the set of natural numbers from 1 to 9. -/
def S : Finset ℕ := Finset.range 10 \ {0}

/- Define the binomial coefficient for choosing 7 out of 9. -/
def choose_7_9 : ℕ := Nat.choose 9 7

/- Define the condition for the sum of chosen numbers to be 35. -/
def sum_is_35 (s : Finset ℕ) : Prop := s.sum id = 35

/- Number of ways to choose 3 pairs that sum to 10 and include number 5 - means sum should be 35-/
def ways_3_pairs_and_5 : ℕ := 4

/- Probability calculation. -/
def prob_sum_is_35 : ℚ := (ways_3_pairs_and_5: ℚ) / (choose_7_9: ℚ)

theorem avg_five_probability : prob_sum_is_35 = 1 / 9 := by
  sorry

end avg_five_probability_l242_242296


namespace probability_sum_odd_correct_l242_242201

noncomputable def probability_sum_odd : ℚ :=
  let total_ways := 10
  let ways_sum_odd := 6
  ways_sum_odd / total_ways

theorem probability_sum_odd_correct :
  probability_sum_odd = 3 / 5 :=
by
  unfold probability_sum_odd
  rfl

end probability_sum_odd_correct_l242_242201


namespace find_n_l242_242932

theorem find_n (n : ℕ) :
  (2^n - 1) % 3 = 0 ∧ (∃ m : ℤ, (2^n - 1) / 3 ∣ 4 * m^2 + 1) →
  ∃ j : ℕ, n = 2^j :=
by
  sorry

end find_n_l242_242932


namespace current_at_R_12_l242_242050

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l242_242050


namespace max_not_sum_S_l242_242773

def S : Set ℕ := {n | ∃ k : ℕ, n = 10^k + 1000}

theorem max_not_sum_S : ∀ x : ℕ, (∀ y ∈ S, ∃ m : ℕ, x ≠ m * y) ↔ x = 34999 := by
  sorry

end max_not_sum_S_l242_242773


namespace least_number_remainder_4_l242_242516

theorem least_number_remainder_4 (n : ℕ) :
  (n % 6 = 4) ∧ (n % 7 = 4) ∧ (n % 9 = 4) ∧ (n % 18 = 4) ↔ n = 130 :=
by
  sorry

end least_number_remainder_4_l242_242516


namespace battery_current_l242_242054

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l242_242054


namespace decreasing_interval_implies_a_ge_two_l242_242359

-- The function f is given
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x - 3

-- Defining the condition for f(x) being decreasing in the interval (-8, 2)
def is_decreasing_in_interval (a : ℝ) : Prop :=
  ∀ x y : ℝ, (-8 < x ∧ x < y ∧ y < 2) → f x a > f y a

-- The proof statement
theorem decreasing_interval_implies_a_ge_two (a : ℝ) (h : is_decreasing_in_interval a) : a ≥ 2 :=
sorry

end decreasing_interval_implies_a_ge_two_l242_242359


namespace focus_of_parabola_l242_242357

-- Definitions for the problem
def parabola_eq (x y : ℝ) : Prop := y = 2 * x^2

def general_parabola_form (x y h k p : ℝ) : Prop :=
  4 * p * (y - k) = (x - h)^2

def vertex_origin (h k : ℝ) : Prop := h = 0 ∧ k = 0

-- Lean statement asserting that the focus of the given parabola is (0, 1/8)
theorem focus_of_parabola : ∃ p : ℝ, parabola_eq x y → general_parabola_form x y 0 0 p ∧ p = 1/8 := by
  sorry

end focus_of_parabola_l242_242357


namespace shells_in_afternoon_l242_242619

-- Conditions: Lino picked up 292 shells in the morning and 616 shells in total.
def shells_in_morning : ℕ := 292
def total_shells : ℕ := 616

-- Theorem: The number of shells Lino picked up in the afternoon is 324.
theorem shells_in_afternoon : (total_shells - shells_in_morning) = 324 := 
by sorry

end shells_in_afternoon_l242_242619


namespace correct_articles_l242_242012

-- Define the given conditions
def specific_experience : Prop := true
def countable_noun : Prop := true

-- Problem statement: given the conditions, choose the correct articles to fill in the blanks
theorem correct_articles (h1 : specific_experience) (h2 : countable_noun) : 
  "the; a" = "the; a" :=
by
  sorry

end correct_articles_l242_242012


namespace prime_power_implies_one_l242_242186

theorem prime_power_implies_one (p : ℕ) (a : ℤ) (n : ℕ) (h_prime : Nat.Prime p) (h_eq : 2^p + 3^p = a^n) :
  n = 1 :=
sorry

end prime_power_implies_one_l242_242186


namespace lassis_from_12_mangoes_l242_242284

-- Conditions as definitions in Lean 4
def total_mangoes : ℕ := 12
def damaged_mango_ratio : ℕ := 1 / 6
def lassis_per_pair_mango : ℕ := 11

-- Equation to calculate the lassis
theorem lassis_from_12_mangoes : (total_mangoes - total_mangoes / 6) / 2 * lassis_per_pair_mango = 55 :=
by
  -- calculation steps should go here, but are omitted as per instructions
  sorry

end lassis_from_12_mangoes_l242_242284


namespace expected_value_of_win_is_3_5_l242_242862

noncomputable def expected_value_win : ℝ :=
  (1/8) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

theorem expected_value_of_win_is_3_5 :
  expected_value_win = 3.5 :=
by
  sorry

end expected_value_of_win_is_3_5_l242_242862


namespace triangle_area_l242_242456

variables {A B C a b c : ℝ}

/-- In triangle ABC, the sides opposite to angles A, B, and C are denoted as a, b, and c, respectively.
It is given that b * sin C + c * sin B = 4 * a * sin B * sin C and b^2 + c^2 - a^2 = 8.
Prove that the area of triangle ABC is 4 * sqrt 3 / 3. -/
theorem triangle_area (h1 : b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C)
  (h2 : b^2 + c^2 - a^2 = 8) :
  (1 / 2) * b * c * Real.sin A = 4 * Real.sqrt 3 / 3 :=
sorry

end triangle_area_l242_242456


namespace ball_reaches_less_than_2_feet_after_14_bounces_l242_242393

theorem ball_reaches_less_than_2_feet_after_14_bounces :
  ∀ (h₀ : ℝ) (r : ℝ), h₀ = 500 → r = 2 / 3 →
  ∃ (k : ℕ), k = 14 ∧ h₀ * r^k < 2 := by
  intros h₀ r h₀_eq r_eq
  use 14
  rw [h₀_eq, r_eq]
  norm_num
  apply lt_trans
    (norm_num [500 * (2 / 3)^14])
  norm_num [2]
  sorry -- Proof for the exact value comparison

end ball_reaches_less_than_2_feet_after_14_bounces_l242_242393


namespace number_of_citroens_submerged_is_zero_l242_242280

-- Definitions based on the conditions
variables (x y : ℕ) -- Define x as the number of Citroen and y as the number of Renault submerged
variables (r p c vr vp : ℕ) -- Define r as the number of Renault, p as the number of Peugeot, c as the number of Citroën

-- Given conditions translated
-- Condition 1: There were twice as many Renault cars as there were Peugeot cars
def condition1 (r p : ℕ) : Prop := r = 2 * p
-- Condition 2: There were twice as many Peugeot cars as there were Citroens
def condition2 (p c : ℕ) : Prop := p = 2 * c
-- Condition 3: As many Citroens as Renaults were submerged in the water
def condition3 (x y : ℕ) : Prop := y = x
-- Condition 4: Three times as many Renaults were in the water as there were Peugeots
def condition4 (r y : ℕ) : Prop := r = 3 * y
-- Condition 5: As many Peugeots visible in the water as there were Citroens
def condition5 (vp c : ℕ) : Prop := vp = c

-- The question to prove: The number of Citroen cars submerged is 0
theorem number_of_citroens_submerged_is_zero
  (h1 : condition1 r p) 
  (h2 : condition2 p c)
  (h3 : condition3 x y)
  (h4 : condition4 r y)
  (h5 : condition5 vp c) :
  x = 0 :=
sorry

end number_of_citroens_submerged_is_zero_l242_242280


namespace total_pears_picked_l242_242471

def mikes_pears : Nat := 8
def jasons_pears : Nat := 7
def freds_apples : Nat := 6

theorem total_pears_picked : (mikes_pears + jasons_pears) = 15 :=
by
  sorry

end total_pears_picked_l242_242471


namespace expected_value_of_win_is_3_point_5_l242_242849

noncomputable def expected_value_win : ℚ :=
  let win (n : ℕ) : ℚ := 8 - n
  let probabilities := List.repeat (1/8 : ℚ) 8
  (List.range 8).map (λ n => probabilities.head! * win (n + 1)).sum

theorem expected_value_of_win_is_3_point_5 : expected_value_win = 3.5 := by
  sorry

end expected_value_of_win_is_3_point_5_l242_242849


namespace total_salaries_l242_242505

theorem total_salaries (A_salary B_salary : ℝ)
  (hA : A_salary = 1500)
  (hsavings : 0.05 * A_salary = 0.15 * B_salary) :
  A_salary + B_salary = 2000 :=
by {
  sorry
}

end total_salaries_l242_242505


namespace math_problem_l242_242299

theorem math_problem (x : ℤ) :
  let a := 1990 * x + 1989
  let b := 1990 * x + 1990
  let c := 1990 * x + 1991
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  sorry

end math_problem_l242_242299


namespace current_when_resistance_12_l242_242099

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l242_242099


namespace compare_numbers_l242_242649

theorem compare_numbers :
  2^27 < 10^9 ∧ 10^9 < 5^13 :=
by {
  sorry
}

end compare_numbers_l242_242649


namespace arithmetic_progression_impossible_geometric_progression_possible_l242_242687

theorem arithmetic_progression_impossible (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 6) (h3 : c = 4.5) : 
  2 * b ≠ a + c :=
by {
    sorry
}

theorem geometric_progression_possible (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 6) (h3 : c = 4.5) : 
  ∃ r m : ℤ, (b / a)^r = (c / a)^m :=
by {
    sorry
}

end arithmetic_progression_impossible_geometric_progression_possible_l242_242687


namespace pizza_toppings_count_l242_242264

theorem pizza_toppings_count :
  let toppings := 8 in
  let one_topping_pizzas := toppings in
  let two_topping_pizzas := (toppings.choose 2) in
  let three_topping_pizzas := (toppings.choose 3) in
  one_topping_pizzas + two_topping_pizzas + three_topping_pizzas = 92 :=
by sorry

end pizza_toppings_count_l242_242264


namespace vector_perpendicular_to_plane_l242_242991

theorem vector_perpendicular_to_plane
  (a b c d : ℝ)
  (x1 y1 z1 x2 y2 z2 : ℝ)
  (h1 : a * x1 + b * y1 + c * z1 + d = 0)
  (h2 : a * x2 + b * y2 + c * z2 + d = 0) :
  a * (x1 - x2) + b * (y1 - y2) + c * (z1 - z2) = 0 :=
sorry

end vector_perpendicular_to_plane_l242_242991


namespace total_pints_l242_242144

-- Define the given conditions as constants
def annie_picked : Int := 8
def kathryn_picked : Int := annie_picked + 2
def ben_picked : Int := kathryn_picked - 3

-- State the main theorem to prove
theorem total_pints : annie_picked + kathryn_picked + ben_picked = 25 := by
  sorry

end total_pints_l242_242144


namespace sheetrock_width_l242_242536

theorem sheetrock_width (l A w : ℕ) (h_length : l = 6) (h_area : A = 30) (h_formula : A = l * w) : w = 5 :=
by
  -- Placeholder for the proof
  sorry

end sheetrock_width_l242_242536


namespace find_first_number_l242_242121

theorem find_first_number (x : ℝ) : (x + 16 + 8 + 22) / 4 = 13 ↔ x = 6 :=
by 
  sorry

end find_first_number_l242_242121


namespace binom_60_3_l242_242921

theorem binom_60_3 : nat.choose 60 3 = 34220 :=
by
  -- Proof goes here.
  sorry

end binom_60_3_l242_242921


namespace upper_limit_of_x_l242_242327

theorem upper_limit_of_x :
  ∀ x : ℤ, (0 < x ∧ x < 7) ∧ (0 < x ∧ x < some_upper_limit) ∧ (5 > x ∧ x > -1) ∧ (3 > x ∧ x > 0) ∧ (x + 2 < 4) →
  some_upper_limit = 2 :=
by
  intros x h
  sorry

end upper_limit_of_x_l242_242327


namespace geo_seq_sum_neg_six_l242_242766

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ (a₁ q : ℝ), q ≠ 0 ∧ ∀ n, a n = a₁ * q^n

theorem geo_seq_sum_neg_six
  (a : ℕ → ℝ)
  (hgeom : geometric_sequence a)
  (ha_neg : a 1 < 0)
  (h_condition : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) :
  a 3 + a 5 = -6 :=
  sorry

end geo_seq_sum_neg_six_l242_242766


namespace best_approximation_of_x_squared_l242_242334

theorem best_approximation_of_x_squared
  (x : ℝ) (A B C D E : ℝ)
  (h1 : -2 < -1)
  (h2 : -1 < 0)
  (h3 : 0 < 1)
  (h4 : 1 < 2)
  (hx : -1 < x ∧ x < 0)
  (hC : 0 < C ∧ C < 1) :
  x^2 = C :=
sorry

end best_approximation_of_x_squared_l242_242334


namespace circle_area_l242_242489

-- Let r be the radius of the circle
-- The circumference of the circle is given by 2 * π * r, which is 36 cm
-- We need to prove that given this condition, the area of the circle is 324/π square centimeters

theorem circle_area (r : Real) (h : 2 * Real.pi * r = 36) : Real.pi * r^2 = 324 / Real.pi :=
by
  sorry

end circle_area_l242_242489


namespace janine_read_pages_in_two_months_l242_242461

theorem janine_read_pages_in_two_months :
  (let books_last_month := 5
   let books_this_month := 2 * books_last_month
   let total_books := books_last_month + books_this_month
   let pages_per_book := 10
   total_books * pages_per_book = 150) := by
   sorry

end janine_read_pages_in_two_months_l242_242461


namespace ellipse_minor_axis_length_l242_242300

noncomputable def minor_axis_length (a b : ℝ) (eccentricity : ℝ) (sum_distances : ℝ) :=
  if (a > b ∧ b > 0 ∧ eccentricity = (Real.sqrt 5) / 3 ∧ sum_distances = 12) then
    2 * b
  else
    0

theorem ellipse_minor_axis_length (a b : ℝ) (eccentricity : ℝ) (sum_distances : ℝ)
  (h1 : a > b) (h2 : b > 0) (h3 : eccentricity = (Real.sqrt 5) / 3) (h4 : sum_distances = 12) :
  minor_axis_length a b eccentricity sum_distances = 8 :=
sorry

end ellipse_minor_axis_length_l242_242300


namespace appropriate_sampling_method_l242_242271

theorem appropriate_sampling_method
  (total_students : ℕ)
  (male_students : ℕ)
  (female_students : ℕ)
  (survey_size : ℕ)
  (diff_interests : Prop)
  (h1 : total_students = 1000)
  (h2 : male_students = 500)
  (h3 : female_students = 500)
  (h4 : survey_size = 100)
  (h5 : diff_interests) : 
  sampling_method = "stratified sampling" :=
by
  sorry

end appropriate_sampling_method_l242_242271


namespace find_range_a_l242_242579

noncomputable def f (a x : ℝ) : ℝ := x^2 + (a^2 - 1) * x + (a - 2)

theorem find_range_a (a : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ x > 1 ∧ y < 1 ) :
  -2 < a ∧ a < 1 := sorry

end find_range_a_l242_242579


namespace problem_l242_242442

theorem problem (a b : ℝ) (h : a > b) (k : b > 0) : b * (a - b) > 0 := 
by
  sorry

end problem_l242_242442


namespace choose_president_and_secretary_same_gender_l242_242195

theorem choose_president_and_secretary_same_gender :
  let total_members := 25
  let boys := 15
  let girls := 10
  ∃ (total_ways : ℕ), total_ways = (boys * (boys - 1)) + (girls * (girls - 1)) := sorry

end choose_president_and_secretary_same_gender_l242_242195


namespace packaging_combinations_l242_242529

theorem packaging_combinations :
  let wraps := 10
  let ribbons := 4
  let cards := 5
  let stickers := 6
  wraps * ribbons * cards * stickers = 1200 :=
by
  rfl

end packaging_combinations_l242_242529


namespace triangle_area_range_l242_242768

theorem triangle_area_range (A B C : ℝ) (a b c : ℝ) 
  (h1 : a * Real.sin B = Real.sqrt 3 * b * Real.cos A)
  (h2 : a = 3) :
  0 < (1 / 2) * b * c * Real.sin A ∧ 
  (1 / 2) * b * c * Real.sin A ≤ (9 * Real.sqrt 3) / 4 := 
  sorry

end triangle_area_range_l242_242768


namespace find_x_l242_242401

theorem find_x (x : ℝ) (h : 40 * x - 138 = 102) : x = 6 :=
by 
  sorry

end find_x_l242_242401


namespace parallel_vectors_sum_is_six_l242_242302

theorem parallel_vectors_sum_is_six (x y : ℝ) :
  let a := (4, -1, 1)
  let b := (x, y, 2)
  (x / 4 = 2) ∧ (y / -1 = 2) →
  x + y = 6 :=
by
  intros
  sorry

end parallel_vectors_sum_is_six_l242_242302


namespace power_neg8_equality_l242_242375

theorem power_neg8_equality :
  (1 / ((-8 : ℤ) ^ 2)^3) * (-8 : ℤ)^7 = 8 :=
by
  sorry

end power_neg8_equality_l242_242375


namespace x_y_difference_is_perfect_square_l242_242187

theorem x_y_difference_is_perfect_square (x y : ℕ) (h : 3 * x^2 + x = 4 * y^2 + y) : ∃ k : ℕ, k^2 = x - y :=
by {sorry}

end x_y_difference_is_perfect_square_l242_242187


namespace binomial_30_3_l242_242690

theorem binomial_30_3 : nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l242_242690


namespace binom_60_3_eq_34220_l242_242908

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l242_242908


namespace current_at_resistance_12_l242_242057

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l242_242057


namespace probability_largest_6_l242_242828

/-- A box contains seven cards numbered from 1 to 7. Four cards are selected randomly without replacement.
    The probability that 6 is the largest number selected is 2/7. -/
theorem probability_largest_6 (s : Finset ℕ) (h : s = {1, 2, 3, 4, 5, 6, 7}) :
  (Finset.card (Finset.filter (λ t : Finset ℕ, t.card = 4 ∧ t.max = 6) (s.powerset))) /
  (Finset.card (Finset.filter (λ t : Finset ℕ, t.card = 4) (s.powerset))) = 2 / 7 :=
sorry

end probability_largest_6_l242_242828


namespace total_cups_needed_l242_242399

theorem total_cups_needed (cereal_servings : ℝ) (milk_servings : ℝ) (nuts_servings : ℝ) 
  (cereal_cups_per_serving : ℝ) (milk_cups_per_serving : ℝ) (nuts_cups_per_serving : ℝ) : 
  cereal_servings = 18.0 ∧ milk_servings = 12.0 ∧ nuts_servings = 6.0 ∧ 
  cereal_cups_per_serving = 2.0 ∧ milk_cups_per_serving = 1.5 ∧ nuts_cups_per_serving = 0.5 → 
  (cereal_servings * cereal_cups_per_serving + milk_servings * milk_cups_per_serving + 
   nuts_servings * nuts_cups_per_serving) = 57.0 :=
by
  sorry

end total_cups_needed_l242_242399


namespace total_revenue_correct_l242_242210

-- Define the costs of different types of returns
def cost_federal : ℕ := 50
def cost_state : ℕ := 30
def cost_quarterly : ℕ := 80

-- Define the quantities sold for different types of returns
def qty_federal : ℕ := 60
def qty_state : ℕ := 20
def qty_quarterly : ℕ := 10

-- Calculate the total revenue for the day
def total_revenue : ℕ := (cost_federal * qty_federal) + (cost_state * qty_state) + (cost_quarterly * qty_quarterly)

-- The theorem stating the total revenue calculation
theorem total_revenue_correct : total_revenue = 4400 := by
  sorry

end total_revenue_correct_l242_242210


namespace solution_set_of_inequality_l242_242617

variable {f : ℝ → ℝ}

noncomputable def F (x : ℝ) : ℝ := x^2 * f x

theorem solution_set_of_inequality
  (h_diff : ∀ x < 0, DifferentiableAt ℝ f x) 
  (h_cond : ∀ x < 0, 2 * f x + x * (deriv f x) > x^2) :
  ∀ x, ((x + 2016)^2 * f (x + 2016) - 9 * f (-3) < 0) ↔ (-2019 < x ∧ x < -2016) :=
by
  sorry

end solution_set_of_inequality_l242_242617


namespace intersection_points_count_l242_242711

-- Definition of the two equations as conditions
def eq1 (x y : ℝ) : Prop := y = 3 * x^2
def eq2 (x y : ℝ) : Prop := y^2 - 6 * y + 8 = x^2

-- The theorem stating that the number of intersection points of the two graphs is exactly 4
theorem intersection_points_count : 
  ∃ (points : Finset (ℝ × ℝ)), (∀ p : ℝ × ℝ, p ∈ points ↔ eq1 p.1 p.2 ∧ eq2 p.1 p.2) ∧ points.card = 4 :=
by
  sorry

end intersection_points_count_l242_242711


namespace find_x_l242_242761

variable (x : ℝ)
def vector_a : ℝ × ℝ := (x, 2)
def vector_b : ℝ × ℝ := (x - 1, 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_x (h1 : dot_product (vector_a x + vector_b x) (vector_a x - vector_b x) = 0) : x = -1 := by 
  sorry

end find_x_l242_242761


namespace battery_current_l242_242023

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l242_242023


namespace fraction_savings_spent_on_furniture_l242_242780

theorem fraction_savings_spent_on_furniture (savings : ℝ) (tv_cost : ℝ) (F : ℝ) 
  (h1 : savings = 840) (h2 : tv_cost = 210) 
  (h3 : F * savings + tv_cost = savings) : F = 3 / 4 :=
sorry

end fraction_savings_spent_on_furniture_l242_242780


namespace smaller_number_l242_242961

theorem smaller_number {a b : ℕ} (h_ratio : b = 5 * a / 2) (h_lcm : Nat.lcm a b = 160) : a = 64 := 
by
  sorry

end smaller_number_l242_242961


namespace ratio_expenditure_l242_242675

variable (I : ℝ) -- Assume the income in the first year is I.

-- Conditions
def savings_first_year := 0.25 * I
def expenditure_first_year := 0.75 * I
def income_second_year := 1.25 * I
def savings_second_year := 2 * savings_first_year
def expenditure_second_year := income_second_year - savings_second_year
def total_expenditure_two_years := expenditure_first_year + expenditure_second_year

-- Statement to be proved
theorem ratio_expenditure 
  (savings_first_year : ℝ := 0.25 * I)
  (expenditure_first_year : ℝ := 0.75 * I)
  (income_second_year : ℝ := 1.25 * I)
  (savings_second_year : ℝ := 2 * savings_first_year)
  (expenditure_second_year : ℝ := income_second_year - savings_second_year)
  (total_expenditure_two_years : ℝ := expenditure_first_year + expenditure_second_year) :
  (total_expenditure_two_years / expenditure_first_year) = 2 := by
    sorry

end ratio_expenditure_l242_242675


namespace jill_speed_downhill_l242_242605

theorem jill_speed_downhill 
  (up_speed : ℕ) (total_time : ℕ) (hill_distance : ℕ) 
  (up_time : ℕ) (down_time : ℕ) (down_speed : ℕ) 
  (h1 : up_speed = 9)
  (h2 : total_time = 175)
  (h3 : hill_distance = 900)
  (h4 : up_time = hill_distance / up_speed)
  (h5 : down_time = total_time - up_time)
  (h6 : down_speed = hill_distance / down_time) :
  down_speed = 12 := 
  by
    sorry

end jill_speed_downhill_l242_242605


namespace correct_statements_l242_242972

-- Definitions
noncomputable def f (x b c : ℝ) : ℝ := abs x * x + b * x + c

-- Proof statements
theorem correct_statements (b c : ℝ) :
  (b > 0 → ∀ x y : ℝ, x ≤ y → f x b c ≤ f y b c) ∧
  (b < 0 → ¬ (∀ x : ℝ, ∃ m : ℝ, f x b c = m)) ∧
  (b = 0 → ∀ x : ℝ, f (x) b c = f (-x) b c) ∧
  (∃ x1 x2 x3 : ℝ, f x1 b c = 0 ∧ f x2 b c = 0 ∧ f x3 b c = 0) :=
sorry

end correct_statements_l242_242972


namespace vector_subtraction_l242_242751

def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def vec_smul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

def vec_a : ℝ × ℝ := (3, 5)
def vec_b : ℝ × ℝ := (-2, 1)

theorem vector_subtraction : vec_sub vec_a (vec_smul 2 vec_b) = (7, 3) :=
by
  sorry

end vector_subtraction_l242_242751


namespace jack_weight_l242_242457

-- Define weights and conditions
def weight_of_rocks : ℕ := 5 * 4
def weight_of_anna : ℕ := 40
def weight_of_jack : ℕ := weight_of_anna - weight_of_rocks

-- Prove that Jack's weight is 20 pounds
theorem jack_weight : weight_of_jack = 20 := by
  sorry

end jack_weight_l242_242457


namespace find_number_in_parentheses_l242_242563

theorem find_number_in_parentheses :
  ∃ x : ℝ, 3 + 2 * (x - 3) = 24.16 ∧ x = 13.58 :=
by
  sorry

end find_number_in_parentheses_l242_242563


namespace binom_60_3_l242_242918

theorem binom_60_3 : nat.choose 60 3 = 34220 :=
by
  -- Proof goes here.
  sorry

end binom_60_3_l242_242918


namespace find_current_l242_242105

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l242_242105


namespace expected_value_of_win_is_3_point_5_l242_242850

noncomputable def expected_value_win : ℚ :=
  let win (n : ℕ) : ℚ := 8 - n
  let probabilities := List.repeat (1/8 : ℚ) 8
  (List.range 8).map (λ n => probabilities.head! * win (n + 1)).sum

theorem expected_value_of_win_is_3_point_5 : expected_value_win = 3.5 := by
  sorry

end expected_value_of_win_is_3_point_5_l242_242850


namespace alexander_total_payment_l242_242881

variable (initialFee : ℝ) (dailyRent : ℝ) (costPerMile : ℝ) (daysRented : ℕ) (milesDriven : ℝ)

def totalCost (initialFee dailyRent costPerMile : ℝ) (daysRented : ℕ) (milesDriven : ℝ) : ℝ :=
  initialFee + (dailyRent * daysRented) + (costPerMile * milesDriven)

theorem alexander_total_payment :
  totalCost 15 30 0.25 3 350 = 192.5 :=
by
  unfold totalCost
  norm_num

end alexander_total_payment_l242_242881


namespace least_common_multiple_prime_numbers_l242_242217

theorem least_common_multiple_prime_numbers (x y : ℕ) (hx_prime : Prime x) (hy_prime : Prime y)
  (hxy : y < x) (h_eq : 2 * x + y = 12) : Nat.lcm x y = 10 :=
by
  sorry

end least_common_multiple_prime_numbers_l242_242217


namespace problem_I_problem_II_l242_242160

variable (t a : ℝ)

-- Problem (I)
theorem problem_I (h1 : a = 1) (h2 : t^2 - 5 * a * t + 4 * a^2 < 0) (h3 : (t - 2) * (t - 6) < 0) : 2 < t ∧ t < 4 := 
by 
  sorry   -- Proof omitted as per instructions

-- Problem (II)
theorem problem_II (h1 : (t - 2) * (t - 6) < 0 → t^2 - 5 * a * t + 4 * a^2 < 0) : 3 / 2 ≤ a ∧ a ≤ 2 :=
by 
  sorry   -- Proof omitted as per instructions

end problem_I_problem_II_l242_242160


namespace integer_pairs_satisfy_equation_l242_242552

theorem integer_pairs_satisfy_equation :
  ∀ (a b : ℤ), b + 1 ≠ 0 → b + 2 ≠ 0 → a + b + 1 ≠ 0 →
    ( (a + 2)/(b + 1) + (a + 1)/(b + 2) = 1 + 6/(a + b + 1) ↔ 
      (a = 1 ∧ b = 0) ∨ (∃ t : ℤ, t ≠ 0 ∧ t ≠ -1 ∧ a = -3 - t ∧ b = t) ) :=
by
  intros a b h1 h2 h3
  sorry

end integer_pairs_satisfy_equation_l242_242552


namespace circumradius_relationship_l242_242517

theorem circumradius_relationship 
  (a b c a' b' c' R : ℝ)
  (S S' p p' : ℝ)
  (h₁ : R = (a * b * c) / (4 * S))
  (h₂ : R = (a' * b' * c') / (4 * S'))
  (h₃ : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h₄ : S' = Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c')))
  (h₅ : p = (a + b + c) / 2)
  (h₆ : p' = (a' + b' + c') / 2) :
  (a * b * c) / Real.sqrt (p * (p - a) * (p - b) * (p - c)) = 
  (a' * b' * c') / Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c')) :=
by 
  sorry

end circumradius_relationship_l242_242517


namespace current_when_resistance_is_12_l242_242027

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l242_242027


namespace evaluate_expression_l242_242737

variable (a : ℝ)

def a_definition : Prop := a = Real.sqrt 11 - 1

theorem evaluate_expression (h : a_definition a) : a^2 + 2*a + 1 = 11 := by
  sorry

end evaluate_expression_l242_242737


namespace man_l242_242871

theorem man's_rowing_speed_in_still_water
  (river_speed : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (H_river_speed : river_speed = 2)
  (H_total_time : total_time = 1)
  (H_total_distance : total_distance = 5.333333333333333) :
  ∃ (v : ℝ), 
    v = 7.333333333333333 ∧
    ∀ d,
    d = total_distance / 2 →
    d = (v - river_speed) * (total_time / 2) ∧
    d = (v + river_speed) * (total_time / 2) := 
by
  sorry

end man_l242_242871


namespace max_at_zero_l242_242215

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem max_at_zero : ∀ x : ℝ, f x ≤ f 0 :=
by
  sorry

end max_at_zero_l242_242215


namespace ben_weekly_eggs_l242_242348

-- Definitions for the conditions
def weekly_saly_eggs : ℕ := 10
def weekly_ben_eggs (B : ℕ) : ℕ := B
def weekly_ked_eggs (B : ℕ) : ℕ := B / 2

def weekly_production (B : ℕ) : ℕ :=
  weekly_saly_eggs + weekly_ben_eggs B + weekly_ked_eggs B

def monthly_production (B : ℕ) : ℕ := 4 * weekly_production B

-- Theorem for the proof
theorem ben_weekly_eggs (B : ℕ) (h : monthly_production B = 124) : B = 14 :=
sorry

end ben_weekly_eggs_l242_242348


namespace find_R_when_S_7_l242_242190

-- Define the variables and equations in Lean
variables (R S g : ℕ)

-- The theorem statement based on the given conditions and desired conclusion
theorem find_R_when_S_7 (h1 : R = 2 * g * S + 3) (h2: R = 23) (h3 : S = 5) : (∃ g : ℕ, R = 2 * g * 7 + 3) :=
by {
  -- This part enforces the proof will be handled later
  sorry
}

end find_R_when_S_7_l242_242190


namespace bullet_speed_difference_l242_242651

theorem bullet_speed_difference (speed_horse speed_bullet : ℕ) 
    (h_horse : speed_horse = 20) (h_bullet : speed_bullet = 400) :
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    speed_same_direction - speed_opposite_direction = 40 :=
    by
    -- Define the speeds in terms of the given conditions.
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    -- State the equality to prove.
    show speed_same_direction - speed_opposite_direction = 40;
    -- Proof (skipped here).
    -- sorry is used to denote where the formal proof steps would go.
    sorry

end bullet_speed_difference_l242_242651


namespace evaluate_expression_l242_242404

theorem evaluate_expression : 3 + 5 * 2^3 - 4 / 2 + 7 * 3 = 62 := 
  by sorry

end evaluate_expression_l242_242404


namespace percentage_above_wholesale_correct_l242_242123

variable (wholesale_cost retail_cost employee_payment : ℝ)
variable (employee_discount percentage_above_wholesale : ℝ)

theorem percentage_above_wholesale_correct :
  wholesale_cost = 200 → 
  employee_discount = 0.25 → 
  employee_payment = 180 → 
  retail_cost = wholesale_cost + (percentage_above_wholesale / 100) * wholesale_cost →
  employee_payment = (1 - employee_discount) * retail_cost →
  percentage_above_wholesale = 20 :=
by
  intros
  sorry

end percentage_above_wholesale_correct_l242_242123


namespace pizza_topping_count_l242_242267

theorem pizza_topping_count (n : ℕ) (h : n = 8) : 
  (nat.choose n 1) + (nat.choose n 2) + (nat.choose n 3) = 92 :=
by {
  rw h,
  simp [nat.choose],
  sorry,
}

end pizza_topping_count_l242_242267


namespace cyclist_distance_l242_242671

theorem cyclist_distance
  (v t d : ℝ)
  (h1 : d = v * t)
  (h2 : d = (v + 1) * (t - 0.5))
  (h3 : d = (v - 1) * (t + 1)) :
  d = 6 :=
by
  sorry

end cyclist_distance_l242_242671


namespace problem_solution_l242_242975

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem problem_solution : a^10 + b^10 = 123 := sorry

end problem_solution_l242_242975


namespace each_wolf_needs_one_deer_l242_242995

-- Definitions used directly from conditions
variable (w_hunting : ℕ) -- wolves out hunting
variable (w_pack : ℕ) -- additional wolves in the pack
variable (m_per_wolf_per_day : ℕ) -- meat requirement per wolf per day
variable (d : ℕ) -- days until next hunt
variable (m_per_deer : ℕ) -- meat provided by one deer

-- Setting up conditions
def total_wolves : ℕ := w_hunting + w_pack
def daily_meat_requirement : ℕ := total_wolves * m_per_wolf_per_day
def total_meat_needed : ℕ := d * daily_meat_requirement
def deer_needed : ℕ := total_meat_needed / m_per_deer
def deer_per_wolf : ℕ := deer_needed / w_hunting

-- The statement to be proved
theorem each_wolf_needs_one_deer
  (h_w_hunting: w_hunting = 4)
  (h_w_pack: w_pack = 16)
  (h_m_per_wolf_per_day: m_per_wolf_per_day = 8)
  (h_d: d = 5)
  (h_m_per_deer: m_per_deer = 200) :
  deer_per_wolf = 1 :=
by
  unfold total_wolves daily_meat_requirement total_meat_needed deer_needed deer_per_wolf
  rw [h_w_hunting, h_w_pack, h_m_per_wolf_per_day, h_d, h_m_per_deer] -- Rewrite with given conditions
  sorry -- Proof steps are omitted

-- Assign actual values to the variables to ensure correctness
#eval deer_per_wolf 4 16 8 5 200 -- Expected evaluation result: 1

end each_wolf_needs_one_deer_l242_242995


namespace expected_value_of_8_sided_die_l242_242857

open ProbabilityTheory

-- Definitions based on conditions

-- Define the 8-sided die outcomes
def outcomes := ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ)

-- Define the probability of each outcome
def probability (n : ℕ) : ℝ := if n ∈ outcomes then (1 / 8) else 0

-- Define the payout function based on the roll outcome
def payout (n : ℕ) : ℝ := if n ∈ outcomes then 8 - n else 0

-- Define the expected value calculation
def expected_value : ℝ :=
  ∑ n in outcomes, (probability n) * (payout n)

-- Main theorem to prove the expected value is 3.5 dollars
theorem expected_value_of_8_sided_die : expected_value = 3.5 :=
by
  -- Expected value calculation skipped; to be completed
  sorry

end expected_value_of_8_sided_die_l242_242857


namespace polynomial_factor_l242_242796

def factorization_condition (p q : ℤ) : Prop :=
  ∃ r s : ℤ, 
    p = 4 * r ∧ 
    q = -3 * r + 4 * s ∧ 
    40 = 2 * r - 3 * s + 16 ∧ 
    -20 = s - 12

theorem polynomial_factor (p q : ℤ) (hpq : factorization_condition p q) : (p, q) = (0, -32) :=
by sorry

end polynomial_factor_l242_242796


namespace binom_30_3_eq_4060_l242_242697

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := 
  by sorry

end binom_30_3_eq_4060_l242_242697


namespace find_multiple_of_savings_l242_242883

variable (A K m : ℝ)

-- Conditions
def condition1 : Prop := A - 150 = (1 / 3) * K
def condition2 : Prop := A + K = 750

-- Question
def question : Prop := m * K = 3 * A

-- Proof Problem Statement
theorem find_multiple_of_savings (h1 : condition1 A K) (h2 : condition2 A K) : 
  question A K 2 :=
sorry

end find_multiple_of_savings_l242_242883


namespace probability_greater_than_4_l242_242307

open MeasureTheory

-- Define the probability density function of a normal distribution
def f (x : ℝ) : ℝ := (1 / real.sqrt (2 * real.pi)) * real.exp (-(x - 2)^2 / 2)

-- Define the integral condition
def integral_condition : Prop :=
  ∫ x in 0..2, f x = 1 / 3

-- Theorem statement to be proved
theorem probability_greater_than_4 (h : integral_condition) : 
  ∫ x in 4..real.infinity, f x = 1 / 6 :=
sorry

end probability_greater_than_4_l242_242307


namespace range_of_y_l242_242010

theorem range_of_y (x : ℝ) : 
  - (Real.sqrt 3) / 3 ≤ (Real.sin x) / (2 - Real.cos x) ∧ (Real.sin x) / (2 - Real.cos x) ≤ (Real.sqrt 3) / 3 :=
sorry

end range_of_y_l242_242010


namespace robotics_club_neither_l242_242350

theorem robotics_club_neither (total_students cs_students e_students both_students : ℕ)
  (h1 : total_students = 80)
  (h2 : cs_students = 52)
  (h3 : e_students = 45)
  (h4 : both_students = 32) :
  total_students - (cs_students - both_students + e_students - both_students + both_students) = 15 :=
by
  sorry

end robotics_club_neither_l242_242350


namespace large_pile_toys_l242_242594

theorem large_pile_toys (x y : ℕ) (h1 : x + y = 120) (h2 : y = 2 * x) : y = 80 := by
  sorry

end large_pile_toys_l242_242594


namespace system_of_equations_m_value_l242_242567

theorem system_of_equations_m_value {x y m : ℝ} 
  (h1 : 2 * x + y = 4)
  (h2 : x + 2 * y = m)
  (h3 : x + y = 1) : m = -1 := 
sorry

end system_of_equations_m_value_l242_242567


namespace solve_for_g2_l242_242635

-- Let g : ℝ → ℝ be a function satisfying the given condition
variable (g : ℝ → ℝ)

-- The given condition
def condition (x : ℝ) : Prop :=
  g (2 ^ x) + x * g (2 ^ (-x)) = 2

-- The main theorem we aim to prove
theorem solve_for_g2 (h : ∀ x, condition g x) : g 2 = 0 :=
by
  sorry

end solve_for_g2_l242_242635


namespace ratio_of_areas_l242_242219

noncomputable def side_length_S : ℝ := sorry
noncomputable def side_length_longer_R : ℝ := 1.2 * side_length_S
noncomputable def side_length_shorter_R : ℝ := 0.8 * side_length_S
noncomputable def area_S : ℝ := side_length_S ^ 2
noncomputable def area_R : ℝ := side_length_longer_R * side_length_shorter_R

theorem ratio_of_areas :
  (area_R / area_S) = (24 / 25) :=
by
  sorry

end ratio_of_areas_l242_242219


namespace m_greater_than_p_l242_242469

theorem m_greater_than_p (p m n : ℕ) (hp : Nat.Prime p) (hm : 0 < m) (hn : 0 < n) (eq : p^2 + m^2 = n^2) : m > p :=
sorry

end m_greater_than_p_l242_242469


namespace polar_to_cartesian_max_and_min_x_plus_y_l242_242752

-- Define the given polar equation and convert it to Cartesian equations
def polar_equation (rho θ : ℝ) : Prop :=
  rho^2 - 4 * (Real.sqrt 2) * rho * Real.cos (θ - Real.pi / 4) + 6 = 0

-- Cartesian equation derived from the polar equation
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 4 * y + 6 = 0

-- Prove equivalence of the given polar equation and its equivalent Cartesian form for all ρ and \theta
theorem polar_to_cartesian (rho θ : ℝ) : 
  (∃ (x y : ℝ), polar_equation rho θ ∧ x = rho * Real.cos θ ∧ y = rho * Real.sin θ ∧ cartesian_equation x y) :=
by
  sorry

-- Property of points (x, y) on the circle defined by the Cartesian equation
def lies_on_circle (x y : ℝ) : Prop :=
  cartesian_equation x y

-- Given a point (x, y) on the circle defined by cartesian_equation, show bounds for x + y
theorem max_and_min_x_plus_y (x y : ℝ) (h : lies_on_circle x y) : 
  2 ≤ x + y ∧ x + y ≤ 6 :=
by
  sorry

end polar_to_cartesian_max_and_min_x_plus_y_l242_242752


namespace mean_score_for_exam_l242_242730

variable (M SD : ℝ)

-- Define the conditions
def condition1 : Prop := 58 = M - 2 * SD
def condition2 : Prop := 98 = M + 3 * SD

-- The problem statement
theorem mean_score_for_exam (h1 : condition1 M SD) (h2 : condition2 M SD) : M = 74 :=
sorry

end mean_score_for_exam_l242_242730


namespace leo_third_part_time_l242_242343

-- Definitions to represent the conditions
def total_time : ℕ := 120
def first_part_time : ℕ := 25
def second_part_time : ℕ := 2 * first_part_time

-- Proposition to prove
theorem leo_third_part_time :
  total_time - (first_part_time + second_part_time) = 45 :=
by
  sorry

end leo_third_part_time_l242_242343


namespace smallest_m_n_sum_l242_242971

theorem smallest_m_n_sum (m n : ℕ) (hmn : m > n) (div_condition : 4900 ∣ (2023 ^ m - 2023 ^ n)) : m + n = 24 :=
by
  sorry

end smallest_m_n_sum_l242_242971


namespace marie_distance_biked_l242_242346

def biking_speed := 12.0 -- Speed in miles per hour
def biking_time := 2.583333333 -- Time in hours

theorem marie_distance_biked : biking_speed * biking_time = 31 := 
by 
  -- The proof steps go here
  sorry

end marie_distance_biked_l242_242346


namespace total_apples_picked_l242_242891

def number_of_children : Nat := 33
def apples_per_child : Nat := 10
def number_of_adults : Nat := 40
def apples_per_adult : Nat := 3

theorem total_apples_picked :
  (number_of_children * apples_per_child) + (number_of_adults * apples_per_adult) = 450 := by
  -- You need to provide proof here
  sorry

end total_apples_picked_l242_242891


namespace max_x_satisfies_inequality_l242_242306

theorem max_x_satisfies_inequality (k : ℝ) :
    (∀ x : ℝ, |x^2 - 4 * x + k| + |x - 3| ≤ 5 → x ≤ 3) → k = 8 :=
by
  intros h
  /- The proof goes here. -/
  sorry

end max_x_satisfies_inequality_l242_242306


namespace current_at_resistance_12_l242_242059

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l242_242059


namespace sufficient_but_not_necessary_condition_l242_242337

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h : a > b ∧ b > 0) : (1 / a < 1 / b) ∧ ¬ (1 / a < 1 / b → a > b ∧ b > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l242_242337


namespace wrapping_paper_area_l242_242269

variable (w h : ℝ)

theorem wrapping_paper_area : ∃ A, A = 4 * (w + h) ^ 2 :=
by
  sorry

end wrapping_paper_area_l242_242269


namespace find_current_l242_242107

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l242_242107


namespace n_squared_plus_3n_is_perfect_square_iff_l242_242412

theorem n_squared_plus_3n_is_perfect_square_iff (n : ℕ) : 
  ∃ k : ℕ, n^2 + 3 * n = k^2 ↔ n = 1 :=
by 
  sorry

end n_squared_plus_3n_is_perfect_square_iff_l242_242412


namespace m_perp_beta_l242_242774

variable {Point Line Plane : Type}
variable {belongs : Point → Line → Prop}
variable {perp : Line → Plane → Prop}
variable {intersect : Plane → Plane → Line}

variable (α β γ : Plane)
variable (m n l : Line)

-- Conditions for the problem
axiom n_perp_α : perp n α
axiom n_perp_β : perp n β
axiom m_perp_α : perp m α

-- Proof goal: proving m is perpendicular to β
theorem m_perp_beta : perp m β :=
by
  sorry

end m_perp_beta_l242_242774


namespace total_amount_paid_is_correct_l242_242684

def rate_per_kg_grapes := 98
def quantity_grapes := 15
def rate_per_kg_mangoes := 120
def quantity_mangoes := 8
def rate_per_kg_pineapples := 75
def quantity_pineapples := 5
def rate_per_kg_oranges := 60
def quantity_oranges := 10

def cost_grapes := rate_per_kg_grapes * quantity_grapes
def cost_mangoes := rate_per_kg_mangoes * quantity_mangoes
def cost_pineapples := rate_per_kg_pineapples * quantity_pineapples
def cost_oranges := rate_per_kg_oranges * quantity_oranges

def total_amount_paid := cost_grapes + cost_mangoes + cost_pineapples + cost_oranges

theorem total_amount_paid_is_correct : total_amount_paid = 3405 := by
  sorry

end total_amount_paid_is_correct_l242_242684


namespace expected_value_is_350_l242_242854

noncomputable def expected_value_of_winnings : ℚ :=
  ((1 / 8) * (8 - 1) + (1 / 8) * (8 - 2) + (1 / 8) * (8 - 3) + (1 / 8) * (8 - 4) +
  (1 / 8) * (8 - 5) + (1 / 8) * (8 - 6) + (1 / 8) * (8 - 7) + (1 / 8) * (8 - 8))

theorem expected_value_is_350 :
  expected_value_of_winnings = 3.50 := by
  sorry

end expected_value_is_350_l242_242854


namespace units_digit_sum_factorials_l242_242383

theorem units_digit_sum_factorials : 
  let units_digit (n : Nat) := n % 10 in
  (units_digit (1!) + units_digit (2!) + units_digit (3!) + units_digit (4!) + units_digit (Sum (List.init (500-4) (fun n => (n+5) !)))) % 10 = 3 :=
by
  let units_digit (n : Nat) := n % 10
  sorry

end units_digit_sum_factorials_l242_242383


namespace john_bought_3_croissants_l242_242291

variable (c k : ℕ)

theorem john_bought_3_croissants
  (h1 : c + k = 5)
  (h2 : ∃ n : ℕ, 88 * c + 44 * k = 100 * n) :
  c = 3 :=
by
-- Proof omitted
sorry

end john_bought_3_croissants_l242_242291


namespace bullet_speed_difference_l242_242659

def bullet_speed_in_same_direction (v_h v_b : ℝ) : ℝ :=
  v_b + v_h

def bullet_speed_in_opposite_direction (v_h v_b : ℝ) : ℝ :=
  v_b - v_h

theorem bullet_speed_difference (v_h v_b : ℝ) (h_h : v_h = 20) (h_b : v_b = 400) :
  bullet_speed_in_same_direction v_h v_b - bullet_speed_in_opposite_direction v_h v_b = 40 :=
by
  rw [h_h, h_b]
  sorry

end bullet_speed_difference_l242_242659


namespace quadratic_distinct_roots_l242_242733

theorem quadratic_distinct_roots (a : ℝ) : 
  (a > -1 ∧ a ≠ 3) ↔ 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (a - 3) * x₁^2 - 4 * x₁ - 1 = 0 ∧ 
    (a - 3) * x₂^2 - 4 * x₂ - 1 = 0 :=
by
  sorry

end quadratic_distinct_roots_l242_242733


namespace ribbons_jane_uses_l242_242603

-- Given conditions
def dresses_sewn_first_period (dresses_per_day : ℕ) (days : ℕ) : ℕ :=
  dresses_per_day * days

def dresses_sewn_second_period (dresses_per_day : ℕ) (days : ℕ) : ℕ :=
  dresses_per_day * days

def total_dresses_sewn (dresses_first_period : ℕ) (dresses_second_period : ℕ) : ℕ :=
  dresses_first_period + dresses_second_period

def total_ribbons_used (total_dresses : ℕ) (ribbons_per_dress : ℕ) : ℕ :=
  total_dresses * ribbons_per_dress

-- Theorem to prove
theorem ribbons_jane_uses :
  total_ribbons_used (total_dresses_sewn (dresses_sewn_first_period 2 7) (dresses_sewn_second_period 3 2)) 2 = 40 :=
  sorry

end ribbons_jane_uses_l242_242603


namespace current_at_resistance_12_l242_242061

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l242_242061


namespace shopkeeper_loss_percent_l242_242818

theorem shopkeeper_loss_percent (I : ℝ) (h1 : I > 0) : 
  (0.1 * (I - 0.4 * I)) = 0.4 * (1.1 * I) :=
by
  -- proof goes here
  sorry

end shopkeeper_loss_percent_l242_242818


namespace ribbons_jane_uses_l242_242604

-- Given conditions
def dresses_sewn_first_period (dresses_per_day : ℕ) (days : ℕ) : ℕ :=
  dresses_per_day * days

def dresses_sewn_second_period (dresses_per_day : ℕ) (days : ℕ) : ℕ :=
  dresses_per_day * days

def total_dresses_sewn (dresses_first_period : ℕ) (dresses_second_period : ℕ) : ℕ :=
  dresses_first_period + dresses_second_period

def total_ribbons_used (total_dresses : ℕ) (ribbons_per_dress : ℕ) : ℕ :=
  total_dresses * ribbons_per_dress

-- Theorem to prove
theorem ribbons_jane_uses :
  total_ribbons_used (total_dresses_sewn (dresses_sewn_first_period 2 7) (dresses_sewn_second_period 3 2)) 2 = 40 :=
  sorry

end ribbons_jane_uses_l242_242604


namespace cost_of_apples_l242_242884

theorem cost_of_apples (price_per_six_pounds : ℕ) (pounds_to_buy : ℕ) (expected_cost : ℕ) :
  price_per_six_pounds = 5 → pounds_to_buy = 18 → (expected_cost = 15) → 
  (price_per_six_pounds / 6) * pounds_to_buy = expected_cost :=
by
  intro price_per_six_pounds_eq pounds_to_buy_eq expected_cost_eq
  rw [price_per_six_pounds_eq, pounds_to_buy_eq, expected_cost_eq]
  -- the actual proof would follow, using math steps similar to the solution but skipped here
  sorry

end cost_of_apples_l242_242884


namespace intersection_A_B_l242_242743

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | 0 < x ∧ x ≤ 2}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_l242_242743


namespace john_payment_correct_l242_242340

noncomputable def camera_value : ℝ := 5000
noncomputable def base_rental_fee_per_week : ℝ := 0.10 * camera_value
noncomputable def high_demand_fee_per_week : ℝ := base_rental_fee_per_week + 0.03 * camera_value
noncomputable def low_demand_fee_per_week : ℝ := base_rental_fee_per_week - 0.02 * camera_value
noncomputable def total_rental_fee : ℝ :=
  high_demand_fee_per_week + low_demand_fee_per_week + high_demand_fee_per_week + low_demand_fee_per_week
noncomputable def insurance_fee : ℝ := 0.05 * camera_value
noncomputable def pre_tax_total_cost : ℝ := total_rental_fee + insurance_fee
noncomputable def tax : ℝ := 0.08 * pre_tax_total_cost
noncomputable def total_cost : ℝ := pre_tax_total_cost + tax

noncomputable def mike_contribution : ℝ := 0.20 * total_cost
noncomputable def sarah_contribution : ℝ := min (0.30 * total_cost) 1000
noncomputable def alex_contribution : ℝ := min (0.10 * total_cost) 700
noncomputable def total_friends_contributions : ℝ := mike_contribution + sarah_contribution + alex_contribution

noncomputable def john_final_payment : ℝ := total_cost - total_friends_contributions

theorem john_payment_correct : john_final_payment = 1015.20 :=
by
  sorry

end john_payment_correct_l242_242340


namespace john_speed_l242_242607

def johns_speed (race_distance_miles next_fastest_guy_time_min won_by_min : ℕ) : ℕ :=
    let john_time_min := next_fastest_guy_time_min - won_by_min
    let john_time_hr := john_time_min / 60
    race_distance_miles / john_time_hr

theorem john_speed (race_distance_miles next_fastest_guy_time_min won_by_min : ℕ)
    (h1 : race_distance_miles = 5) (h2 : next_fastest_guy_time_min = 23) (h3 : won_by_min = 3) : 
    johns_speed race_distance_miles next_fastest_guy_time_min won_by_min = 15 := 
by
    sorry

end john_speed_l242_242607


namespace minimum_cans_needed_l242_242527

theorem minimum_cans_needed (h : ∀ c, c * 10 ≥ 120) : ∃ c, c = 12 :=
by
  sorry

end minimum_cans_needed_l242_242527


namespace bullet_speed_difference_l242_242654

def speed_horse : ℕ := 20  -- feet per second
def speed_bullet : ℕ := 400  -- feet per second

def speed_forward : ℕ := speed_bullet + speed_horse
def speed_backward : ℕ := speed_bullet - speed_horse

theorem bullet_speed_difference : speed_forward - speed_backward = 40 :=
by
  sorry

end bullet_speed_difference_l242_242654


namespace transform_sin_to_cos_l242_242233

theorem transform_sin_to_cos :
  ∀ x : ℝ, y = sin (2 * x + π / 4) ↔ y = cos (2 * x - π / 4) := sorry

end transform_sin_to_cos_l242_242233


namespace smallest_number_of_brownies_l242_242317

noncomputable def total_brownies (m n : ℕ) : ℕ := m * n
def perimeter_brownies (m n : ℕ) : ℕ := 2 * m + 2 * n - 4
def interior_brownies (m n : ℕ) : ℕ := (m - 2) * (n - 2)

theorem smallest_number_of_brownies : 
  ∃ (m n : ℕ), 2 * interior_brownies m n = perimeter_brownies m n ∧ total_brownies m n = 36 :=
by
  sorry

end smallest_number_of_brownies_l242_242317


namespace train_speed_40_l242_242006

-- Definitions for the conditions
def passes_pole (L V : ℝ) := V = L / 8
def passes_stationary_train (L V : ℝ) := V = (L + 400) / 18

-- The theorem we want to prove
theorem train_speed_40 (L V : ℝ) (h1 : passes_pole L V) (h2 : passes_stationary_train L V) : V = 40 := 
sorry

end train_speed_40_l242_242006


namespace natural_numbers_pq_equal_l242_242222

theorem natural_numbers_pq_equal (p q : ℕ) (h : p^p + q^q = p^q + q^p) : p = q :=
sorry

end natural_numbers_pq_equal_l242_242222


namespace simplify_expression_evaluate_l242_242628

theorem simplify_expression_evaluate : 
  let x := 1
  let y := 2
  (2 * x - y) * (y + 2 * x) - (2 * y + x) * (2 * y - x) = -15 :=
by
  sorry

end simplify_expression_evaluate_l242_242628


namespace current_value_l242_242088

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l242_242088


namespace f_2015_value_l242_242577

noncomputable def f : ℝ → ℝ := sorry -- Define f with appropriate conditions

theorem f_2015_value :
  (∀ x, f x = -f (-x)) ∧
  (∀ x, f (x + 4) = f x) ∧
  (∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) →
  f 2015 = -2 :=
by
  sorry -- Proof to be provided

end f_2015_value_l242_242577


namespace minimum_value_of_eccentricity_sum_l242_242949

variable {a b m n c : ℝ} (ha : a > b) (hb : b > 0) (hm : m > 0) (hn : n > 0)
variable {e1 e2 : ℝ}

theorem minimum_value_of_eccentricity_sum 
  (h_equiv : a^2 + m^2 = 2 * c^2) 
  (e1_def : e1 = c / a) 
  (e2_def : e2 = c / m) : 
  (2 * e1^2 + (e2^2) / 2) = (9 / 4) :=
sorry

end minimum_value_of_eccentricity_sum_l242_242949


namespace quadratic_function_even_l242_242753

theorem quadratic_function_even (a b : ℝ) (h1 : ∀ x : ℝ, x^2 + (a-1)*x + a + b = x^2 - (a-1)*x + a + b) (h2 : 4 + (a-1)*2 + a + b = 0) : a + b = -4 := 
sorry

end quadratic_function_even_l242_242753


namespace triangle_inscribed_relation_l242_242519

noncomputable def herons_area (p a b c : ℝ) : ℝ := (p * (p - a) * (p - b) * (p - c)).sqrt

theorem triangle_inscribed_relation
  (S S' p p' : ℝ)
  (a b c a' b' c' r : ℝ)
  (h1 : r = S / p)
  (h2 : r = S' / p')
  (h3 : S = herons_area p a b c)
  (h4 : S' = herons_area p' a' b' c') :
  (p - a) * (p - b) * (p - c) / p = (p' - a') * (p' - b') * (p' - c') / p' :=
by sorry

end triangle_inscribed_relation_l242_242519


namespace factorize_polynomial_l242_242504

variable (a x y : ℝ)

theorem factorize_polynomial (a x y : ℝ) :
  3 * a * x ^ 2 - 3 * a * y ^ 2 = 3 * a * (x + y) * (x - y) := by
  sorry

end factorize_polynomial_l242_242504


namespace two_om_2om5_l242_242641

def om (a b : ℕ) : ℕ := a^b - b^a

theorem two_om_2om5 : om 2 (om 2 5) = 79 := by
  sorry

end two_om_2om5_l242_242641


namespace remainder_of_power_mod_l242_242416

theorem remainder_of_power_mod :
  (5^2023) % 11 = 4 :=
  by
    sorry

end remainder_of_power_mod_l242_242416


namespace trapezoid_area_l242_242877

theorem trapezoid_area (x : ℝ) :
  let base1 := 5 * x
  let base2 := 4 * x
  let height := x
  let area := height * (base1 + base2) / 2
  area = 9 * x^2 / 2 :=
by
  -- Definitions based on conditions
  let base1 := 5 * x
  let base2 := 4 * x
  let height := x
  let area := height * (base1 + base2) / 2
  -- Proof of the theorem, currently omitted
  sorry

end trapezoid_area_l242_242877


namespace seven_by_seven_grid_partition_l242_242390

theorem seven_by_seven_grid_partition : 
  ∀ (x y : ℕ), 4 * x + 3 * y = 49 ∧ x + y ≥ 16 → x = 1 :=
by sorry

end seven_by_seven_grid_partition_l242_242390


namespace sector_area_max_sector_area_l242_242948

-- Definitions based on the given conditions
def perimeter : ℝ := 8
def central_angle (α : ℝ) : Prop := α = 2

-- Question 1: Find the area of the sector given the central angle is 2 rad
theorem sector_area (r l : ℝ) (h1 : 2 * r + l = perimeter) (h2 : l = 2 * r) : 
  (1/2) * r * l = 4 := 
by sorry

-- Question 2: Find the maximum area of the sector and the corresponding central angle
theorem max_sector_area (r l : ℝ) (h1 : 2 * r + l = perimeter) : 
  ∃ r, 0 < r ∧ r < 4 ∧ l = 8 - 2 * r ∧ 
  (1/2) * r * l = 4 ∧ l = 2 * r := 
by sorry

end sector_area_max_sector_area_l242_242948


namespace equation_solution_unique_l242_242559

theorem equation_solution_unique (m a b : ℕ) (hm : 1 < m) (ha : 1 < a) (hb : 1 < b) :
  ((m + 1) * a = m * b + 1) ↔ m = 2 :=
sorry

end equation_solution_unique_l242_242559


namespace find_x_l242_242610

variables (a b x : ℝ)
variables (pos_a : a > 0) (pos_b : b > 0) (pos_x : x > 0)

theorem find_x : ((2 * a) ^ (2 * b) = (a^2) ^ b * x ^ b) → (x = 4) := by
  sorry

end find_x_l242_242610


namespace question1_solution_question2_solution_l242_242170

noncomputable def f (x m : ℝ) : ℝ := x^2 - m * x + m - 1

theorem question1_solution (x : ℝ) :
  ∀ x, f x 3 ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2 :=
sorry

theorem question2_solution (m : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f x m ≥ -1) ↔ m ≤ 4 :=
sorry

end question1_solution_question2_solution_l242_242170


namespace rahul_deepak_present_ages_l242_242502

theorem rahul_deepak_present_ages (R D : ℕ) 
  (h1 : R / D = 4 / 3)
  (h2 : R + 6 = 26)
  (h3 : D + 6 = 1/2 * (R + (R + 6)))
  (h4 : (R + 11) + (D + 11) = 59) 
  : R = 20 ∧ D = 17 :=
sorry

end rahul_deepak_present_ages_l242_242502


namespace expected_value_of_8_sided_die_l242_242840

-- Define the expected value function for the given win calculation rule.
def expected_value := (1/8 : ℚ) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

-- Formal statement of the proof problem.
theorem expected_value_of_8_sided_die : 
  expected_value = 3.50 :=
by
  sorry

end expected_value_of_8_sided_die_l242_242840


namespace number_of_students_l242_242464

-- Define John's total winnings
def john_total_winnings : ℤ := 155250

-- Define the proportion of winnings given to each student
def proportion_per_student : ℚ := 1 / 1000

-- Define the total amount received by students
def total_received_by_students : ℚ := 15525

-- Calculate the amount each student received
def amount_per_student : ℚ := john_total_winnings * proportion_per_student

-- Theorem to prove the number of students
theorem number_of_students : total_received_by_students / amount_per_student = 100 :=
by
  -- Lean will be expected to fill in this proof
  sorry

end number_of_students_l242_242464


namespace converse_statement_l242_242491

theorem converse_statement (a : ℝ) : (a > 2018 → a > 2017) ↔ (a > 2017 → a > 2018) :=
by
  sorry

end converse_statement_l242_242491


namespace points_lie_on_parabola_l242_242148

theorem points_lie_on_parabola (u : ℝ) :
  ∃ (x y : ℝ), x = 3^u - 4 ∧ y = 9^u - 7 * 3^u - 2 ∧ y = x^2 + x - 14 :=
by
  sorry

end points_lie_on_parabola_l242_242148


namespace spherical_to_rect_coords_l242_242712

open Real

noncomputable def spherical_to_rect (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * sin phi * cos theta, rho * sin phi * sin theta, rho * cos phi)

theorem spherical_to_rect_coords :
  spherical_to_rect 3 (π / 4) (π / 6) = (3 * sqrt 2 / 4, 3 * sqrt 2 / 4, 3 * sqrt 3 / 2) :=
by
  sorry

end spherical_to_rect_coords_l242_242712


namespace minimum_value_l242_242612

open Real

-- Given the conditions
variables (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k)

-- The theorem
theorem minimum_value (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < k) : 
  ∃ x, x = (3 : ℝ) / k ∧ ∀ y, y = (a / (k * b) + b / (k * c) + c / (k * a)) → y ≥ x :=
sorry

end minimum_value_l242_242612


namespace determine_common_ratio_l242_242465

-- Definition of geometric sequence and sum of first n terms
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def sum_geometric_sequence (a : ℕ → ℝ) : ℕ → ℝ
  | 0       => a 0
  | (n + 1) => a (n + 1) + sum_geometric_sequence a n

-- Main theorem
theorem determine_common_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : is_geometric_sequence a q)
  (h3 : ∀ n, S n = sum_geometric_sequence a n)
  (h4 : 3 * (S 2 + a 2 + a 1 * q^2) = 8 * a 1 * q + 5 * a 1) :
  q = 2 :=
by 
  sorry

end determine_common_ratio_l242_242465


namespace missing_fraction_l242_242929

theorem missing_fraction (x : ℕ) (h1 : x > 0) :
  let lost := (1 / 3 : ℚ) * x
  let found := (2 / 3 : ℚ) * lost
  let remaining := x - lost + found
  (x - remaining) / x = 1 / 9 :=
by
  sorry

end missing_fraction_l242_242929


namespace simplify_expression_l242_242204

theorem simplify_expression : 
  (3.875 * (1 / 5) + (38 + 3 / 4) * 0.09 - 0.155 / 0.4) / 
  (2 + 1 / 6 + (((4.32 - 1.68 - (1 + 8 / 25)) * (5 / 11) - 2 / 7) / (1 + 9 / 35)) + (1 + 11 / 24))
  = 1 := sorry

end simplify_expression_l242_242204


namespace battery_current_l242_242083

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l242_242083


namespace land_remaining_is_correct_l242_242402

def lizzie_covered : ℕ := 250
def other_covered : ℕ := 265
def total_land : ℕ := 900
def land_remaining : ℕ := total_land - (lizzie_covered + other_covered)

theorem land_remaining_is_correct : land_remaining = 385 := 
by
  sorry

end land_remaining_is_correct_l242_242402


namespace binomial_60_3_eq_34220_l242_242917

open Nat

theorem binomial_60_3_eq_34220 : choose 60 3 = 34220 := 
by
  sorry

end binomial_60_3_eq_34220_l242_242917


namespace area_of_circle_l242_242485
open Real

-- Define the circumference condition
def circumference (r : ℝ) : ℝ :=
  2 * π * r

-- Define the area formula
def area (r : ℝ) : ℝ :=
  π * r * r

-- The given radius derived from the circumference
def radius_given_circumference (C : ℝ) : ℝ :=
  C / (2 * π)

-- The target proof statement
theorem area_of_circle (C : ℝ) (h : C = 36) : (area (radius_given_circumference C)) = 324 / π :=
by
  sorry

end area_of_circle_l242_242485


namespace product_of_two_numbers_l242_242216

theorem product_of_two_numbers (a b : ℤ) (h1 : lcm a b = 72) (h2 : gcd a b = 8) :
  a * b = 576 :=
sorry

end product_of_two_numbers_l242_242216


namespace average_temperature_Robertson_l242_242133

def temperatures : List ℝ := [18, 21, 19, 22, 20]

noncomputable def average (temps : List ℝ) : ℝ :=
  (temps.sum) / (temps.length)

theorem average_temperature_Robertson :
  average temperatures = 20.0 :=
by
  sorry

end average_temperature_Robertson_l242_242133


namespace expand_product_l242_242292

theorem expand_product (x : ℝ) : (x^3 + 3) * (x^3 + 4) = x^6 + 7 * x^3 + 12 := 
  sorry

end expand_product_l242_242292


namespace customer_pays_correct_amount_l242_242534

def wholesale_price : ℝ := 4
def markup : ℝ := 0.25
def discount : ℝ := 0.05

def retail_price : ℝ := wholesale_price * (1 + markup)
def discount_amount : ℝ := retail_price * discount
def customer_price : ℝ := retail_price - discount_amount

theorem customer_pays_correct_amount : customer_price = 4.75 := by
  -- proof steps would go here, but we are skipping them as instructed
  sorry

end customer_pays_correct_amount_l242_242534


namespace battery_current_l242_242052

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l242_242052


namespace negation_of_universal_statement_l242_242640

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, x > Real.sin x)) ↔ (∃ x : ℝ, x ≤ Real.sin x) :=
by sorry

end negation_of_universal_statement_l242_242640


namespace max_special_pairs_l242_242423

open Finset

theorem max_special_pairs (n : ℕ) (hn_odd : Odd n) (hn_gt_one : n > 1) :
  ∃ p : Equiv.Perm (Fin n), (∃ S : ℕ, S = (n + 1) * (n + 3) / 8) :=
by
  sorry

end max_special_pairs_l242_242423


namespace half_abs_sum_diff_squares_cubes_l242_242376

theorem half_abs_sum_diff_squares_cubes (a b : ℤ) (h1 : a = 21) (h2 : b = 15) :
  (|a^2 - b^2| + |a^3 - b^3|) / 2 = 3051 := by
  sorry

end half_abs_sum_diff_squares_cubes_l242_242376


namespace total_expenditure_correct_l242_242895

-- Define the weekly costs based on the conditions
def cost_white_bread : Float := 2 * 3.50
def cost_baguette : Float := 1.50
def cost_sourdough_bread : Float := 2 * 4.50
def cost_croissant : Float := 2.00

-- Total weekly cost calculation
def weekly_cost : Float := cost_white_bread + cost_baguette + cost_sourdough_bread + cost_croissant

-- Total cost over 4 weeks
def total_cost_4_weeks (weeks : Float) : Float := weekly_cost * weeks

-- The assertion that needs to be proved
theorem total_expenditure_correct :
  total_cost_4_weeks 4 = 78.00 := by
  sorry

end total_expenditure_correct_l242_242895


namespace mr_arevalo_change_l242_242481

-- Definitions for the costs of the food items
def cost_smoky_salmon : ℤ := 40
def cost_black_burger : ℤ := 15
def cost_chicken_katsu : ℤ := 25

-- Definitions for the service charge and tip percentages
def service_charge_percent : ℝ := 0.10
def tip_percent : ℝ := 0.05

-- Definition for the amount Mr. Arevalo pays
def amount_paid : ℤ := 100

-- Calculation for total food cost
def total_food_cost : ℤ := cost_smoky_salmon + cost_black_burger + cost_chicken_katsu

-- Calculation for service charge
def service_charge : ℝ := service_charge_percent * total_food_cost

-- Calculation for tip
def tip : ℝ := tip_percent * total_food_cost

-- Calculation for the final bill amount
def final_bill_amount : ℝ := total_food_cost + service_charge + tip

-- Calculation for the change
def change : ℝ := amount_paid - final_bill_amount

-- Proof statement
theorem mr_arevalo_change : change = 8 := by
  sorry

end mr_arevalo_change_l242_242481


namespace ratio_of_intercepts_l242_242807

theorem ratio_of_intercepts
  (u v : ℚ)
  (h1 : 2 = 5 * u)
  (h2 : 3 = -7 * v) :
  u / v = -14 / 15 :=
by
  sorry

end ratio_of_intercepts_l242_242807


namespace problem_l242_242678

-- Definitions of the function g and its values at specific points
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

-- Conditions given in the problem
theorem problem (d e f : ℝ)
  (h0 : g d e f 0 = 8)
  (h1 : g d e f 1 = 5) :
  d + e + 2 * f = 13 :=
by
  sorry

end problem_l242_242678


namespace current_value_l242_242018

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l242_242018


namespace trainB_reaches_in_3_hours_l242_242373

variable (trainA_speed trainB_speed : ℕ) (x t : ℝ)

-- Given conditions
axiom h1 : trainA_speed = 70
axiom h2 : trainB_speed = 105
axiom h3 : ∀ x t, 70 * x + 70 * 9 = 105 * x + 105 * t

-- Prove that train B takes 3 hours to reach destination after meeting
theorem trainB_reaches_in_3_hours : t = 3 :=
by
  sorry

end trainB_reaches_in_3_hours_l242_242373


namespace total_weight_loss_l242_242475

def seth_loss : ℝ := 17.53
def jerome_loss : ℝ := 3 * seth_loss
def veronica_loss : ℝ := seth_loss + 1.56
def seth_veronica_loss : ℝ := seth_loss + veronica_loss
def maya_loss : ℝ := seth_veronica_loss - 0.25 * seth_veronica_loss
def total_loss : ℝ := seth_loss + jerome_loss + veronica_loss + maya_loss

theorem total_weight_loss : total_loss = 116.675 := by
  sorry

end total_weight_loss_l242_242475


namespace find_range_of_a_l242_242467

-- Definitions
def is_decreasing_function (a : ℝ) : Prop :=
  0 < a ∧ a < 1

def no_real_roots_of_poly (a : ℝ) : Prop :=
  4 * a < 1

def problem_statement (a : ℝ) : Prop :=
  (is_decreasing_function a ∨ no_real_roots_of_poly a) ∧ ¬ (is_decreasing_function a ∧ no_real_roots_of_poly a)

-- Main theorem
theorem find_range_of_a (a : ℝ) : problem_statement a ↔ (0 < a ∧ a ≤ 1 / 4) ∨ (a ≥ 1) :=
by
  -- Proof omitted
  sorry

end find_range_of_a_l242_242467


namespace units_digit_sum_factorials_500_l242_242384

-- Define the unit digit computation function
def unit_digit (n : ℕ) : ℕ := n % 10

-- Define the factorial function
def fact : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * fact n

-- Define the sum of factorials from 1 to n
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ i => fact i)

-- Define the problem statement
theorem units_digit_sum_factorials_500 : unit_digit (sum_factorials 500) = 3 :=
sorry

end units_digit_sum_factorials_500_l242_242384


namespace proof_cos_2x_cos_2y_l242_242745

variable {θ x y : ℝ}

-- Conditions
def is_arith_seq (a b c : ℝ) := b = (a + c) / 2
def is_geom_seq (a b c : ℝ) := b^2 = a * c

-- Proving the given statement with the provided conditions
theorem proof_cos_2x_cos_2y (h_arith : is_arith_seq (Real.sin θ) (Real.sin x) (Real.cos θ))
                            (h_geom : is_geom_seq (Real.sin θ) (Real.sin y) (Real.cos θ)) :
  2 * Real.cos (2 * x) = Real.cos (2 * y) :=
sorry

end proof_cos_2x_cos_2y_l242_242745


namespace find_current_when_resistance_is_12_l242_242042

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l242_242042


namespace battery_current_at_given_resistance_l242_242093

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l242_242093


namespace original_slices_proof_l242_242718

def original_slices (andy_consumption toast_slices leftover_slice: ℕ) : ℕ :=
  andy_consumption + toast_slices + leftover_slice

theorem original_slices_proof :
  original_slices (3 * 2) (10 * 2) 1 = 27 :=
by
  sorry

end original_slices_proof_l242_242718


namespace range_of_a_l242_242950

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 1 ∧ 2 * a * x + 4 = 0) ↔ (-2 ≤ a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l242_242950


namespace solution_set_inequality_l242_242926

theorem solution_set_inequality (x : ℝ) : 
  (abs (x + 3) - abs (x - 2) ≥ 3) ↔ (x ≥ 1) := 
by {
  sorry
}

end solution_set_inequality_l242_242926


namespace sequence_general_formula_l242_242181

theorem sequence_general_formula {a : ℕ → ℕ} 
  (h₁ : a 1 = 2) 
  (h₂ : ∀ n : ℕ, a (n + 1) = 2 * a n + 3 * 5 ^ n) 
  : ∀ n : ℕ, a n = 5 ^ n - 3 * 2 ^ (n - 1) :=
sorry

end sequence_general_formula_l242_242181


namespace expected_value_of_winnings_l242_242845

theorem expected_value_of_winnings (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 8) :
  ((∑ i in finset.range 8, (8 - i) * (1 / 8)) = 3.5) :=
by
  sorry

end expected_value_of_winnings_l242_242845


namespace lily_profit_is_correct_l242_242779

-- Define the conditions
def first_ticket_price : ℕ := 1
def price_increment : ℕ := 1
def number_of_tickets : ℕ := 5
def prize_amount : ℕ := 11

-- Define the sum of arithmetic series formula
def total_amount_collected (n : ℕ) (a : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Calculate the total amount collected
def total : ℕ := total_amount_collected number_of_tickets first_ticket_price price_increment

-- Define the profit calculation
def profit : ℕ := total - prize_amount

-- The statement we need to prove
theorem lily_profit_is_correct : profit = 4 := by
  sorry

end lily_profit_is_correct_l242_242779


namespace larger_pile_toys_l242_242596

-- Define the conditions
def total_toys (small_pile large_pile : ℕ) : Prop := small_pile + large_pile = 120
def larger_pile (small_pile large_pile : ℕ) : Prop := large_pile = 2 * small_pile

-- Define the proof problem
theorem larger_pile_toys (small_pile large_pile : ℕ) (h1 : total_toys small_pile large_pile) (h2 : larger_pile small_pile large_pile) : 
  large_pile = 80 := by
  sorry

end larger_pile_toys_l242_242596


namespace current_when_resistance_is_12_l242_242032

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l242_242032


namespace min_distance_from_P_to_line_l242_242600

noncomputable def min_distance_to_line : ℝ :=
2 * Real.sqrt 2 - 0.5 * Real.sqrt 10

theorem min_distance_from_P_to_line :
  ∀ (m : ℝ) (x y : ℝ),
  (x + m * y = 0) →
  (m * x - y - m + 3 = 0) →
  let P := (x, y) in
  ∃ d, d = min_distance_to_line ∧ 
  d = Real.abs (x + y - 8) / Real.sqrt (1^2 + 1^2) - 0.5 * Real.sqrt 10 :=
begin
  intros m x y,
  intros h1 h2,
  use min_distance_to_line,
  split,
  { refl },
  { sorry }
end

end min_distance_from_P_to_line_l242_242600


namespace quadratic_solution_l242_242354

theorem quadratic_solution (x : ℝ) : 2 * x^2 - 3 * x + 1 = 0 → (x = 1 / 2 ∨ x = 1) :=
by sorry

end quadratic_solution_l242_242354


namespace expected_value_of_win_is_correct_l242_242835

noncomputable def expected_value_of_win : ℝ :=
  (1 / 8) * (8 - 1) + (1 / 8) * (8 - 2) + (1 / 8) * (8 - 3) +
  (1 / 8) * (8 - 4) + (1 / 8) * (8 - 5) + (1 / 8) * (8 - 6) +
  (1 / 8) * (8 - 7) + (1 / 8) * (8 - 8)

theorem expected_value_of_win_is_correct :
  expected_value_of_win = 3.5 :=
by
  sorry

end expected_value_of_win_is_correct_l242_242835


namespace sally_purchased_20_fifty_cent_items_l242_242788

noncomputable def num_fifty_cent_items (x y z : ℕ) (h1 : x + y + z = 30) (h2 : 50 * x + 500 * y + 1000 * z = 10000) : ℕ :=
x

theorem sally_purchased_20_fifty_cent_items
  (x y z : ℕ)
  (h1 : x + y + z = 30)
  (h2 : 50 * x + 500 * y + 1000 * z = 10000)
  : num_fifty_cent_items x y z h1 h2 = 20 :=
sorry

end sally_purchased_20_fifty_cent_items_l242_242788


namespace battery_current_l242_242051

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l242_242051


namespace candice_bakery_expense_l242_242894

def weekly_expense (white_bread_price : ℕ → ℚ) (baguette_price : ℚ) (sourdough_bread_price : ℕ → ℚ) (croissant_price : ℚ) : ℚ :=
  white_bread_price 2 + baguette_price + sourdough_bread_price 2 + croissant_price

def four_weeks_expense (weekly_expense : ℚ) : ℚ :=
  weekly_expense * 4

theorem candice_bakery_expense :
  weekly_expense (λ n, 3.50 * n) 1.50 (λ n, 4.50 * n) 2.00 * 4 = 78.00 := by
  sorry

end candice_bakery_expense_l242_242894


namespace koala_fiber_intake_l242_242771

theorem koala_fiber_intake (absorption_percentage : ℝ) (absorbed_fiber : ℝ) (total_fiber : ℝ) :
  absorption_percentage = 0.30 → absorbed_fiber = 12 → absorbed_fiber = absorption_percentage * total_fiber → total_fiber = 40 :=
by
  intros h1 h2 h3
  sorry

end koala_fiber_intake_l242_242771


namespace estimate_white_balls_l242_242764

theorem estimate_white_balls
  (total_balls : ℕ)
  (trials : ℕ)
  (white_draws : ℕ)
  (proportion_white : ℚ)
  (hw : total_balls = 10)
  (ht : trials = 400)
  (hd : white_draws = 240)
  (hprop : proportion_white = 0.6) :
  ∃ x : ℕ, x = 6 :=
by
  sorry

end estimate_white_balls_l242_242764


namespace ratio_arithmetic_sequence_last_digit_l242_242924

def is_ratio_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n : ℕ, n > 0 → (a (n + 2) * a n) = (a (n + 1) ^ 2) * d

theorem ratio_arithmetic_sequence_last_digit :
  ∃ a : ℕ → ℕ, is_ratio_arithmetic_sequence a 1 ∧ a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 2 ∧
  (a 2009 / a 2006) % 10 = 6 :=
sorry

end ratio_arithmetic_sequence_last_digit_l242_242924


namespace simplify_expression_l242_242203

theorem simplify_expression (x : ℝ) : 
  (4 * x + 6 * x^3 + 8 - (3 - 6 * x^3 - 4 * x)) = 12 * x^3 + 8 * x + 5 := 
by
  sorry

end simplify_expression_l242_242203


namespace solve_system_of_equations_l242_242984

theorem solve_system_of_equations :
  ∃ x y : ℝ, (3 * x - 5 * y = -1.5) ∧ (7 * x + 2 * y = 4.7) ∧ x = 0.5 ∧ y = 0.6 :=
by
  sorry -- Proof to be completed

end solve_system_of_equations_l242_242984


namespace find_t_l242_242444

variables (s t : ℚ)

theorem find_t (h1 : 12 * s + 7 * t = 154) (h2 : s = 2 * t - 3) : t = 190 / 31 :=
by
  sorry

end find_t_l242_242444


namespace no_adjacent_same_roll_probability_l242_242154

noncomputable def probability_no_adjacent_same_roll : ℚ :=
  (1331 / 1728)

theorem no_adjacent_same_roll_probability :
  (probability_no_adjacent_same_roll = (1331 / 1728)) :=
by
  sorry

end no_adjacent_same_roll_probability_l242_242154


namespace area_of_octagon_l242_242540

theorem area_of_octagon (a b : ℝ) (hsquare : a ^ 2 = 16)
  (hperimeter : 4 * a = 8 * b) :
  2 * (1 + Real.sqrt 2) * b ^ 2 = 8 + 8 * Real.sqrt 2 :=
by
  sorry

end area_of_octagon_l242_242540


namespace abs_inequality_solution_set_l242_242506

theorem abs_inequality_solution_set (x : ℝ) : |x - 1| > 2 ↔ x > 3 ∨ x < -1 :=
by
  sorry

end abs_inequality_solution_set_l242_242506


namespace expected_value_of_8_sided_die_l242_242843

-- Define the expected value function for the given win calculation rule.
def expected_value := (1/8 : ℚ) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

-- Formal statement of the proof problem.
theorem expected_value_of_8_sided_die : 
  expected_value = 3.50 :=
by
  sorry

end expected_value_of_8_sided_die_l242_242843


namespace function_form_l242_242558

noncomputable def f : ℕ → ℕ := sorry

theorem function_form (c d a : ℕ) (h1 : c > 1) (h2 : a - c > 1)
  (hf : ∀ n : ℕ, f n + f (n + 1) = f (n + 2) + f (n + 3) - 168) :
  (∀ n : ℕ, f (2 * n) = c + n * d) ∧ (∀ n : ℕ, f (2 * n + 1) = (168 - d) * n + a - c) :=
sorry

end function_form_l242_242558


namespace joan_books_l242_242969

theorem joan_books : 
  (33 - 26 = 7) :=
by
  sorry

end joan_books_l242_242969


namespace triangle_is_isosceles_l242_242962

theorem triangle_is_isosceles 
  (A B C : ℝ)
  (h : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h_triangle : A + B + C = π)
  (h_condition : (Real.sin B) * (Real.sin C) = (Real.cos (A / 2)) ^ 2) :
  (B = C) :=
sorry

end triangle_is_isosceles_l242_242962


namespace fibonacci_units_digit_l242_242795

def fibonacci (n : ℕ) : ℕ :=
match n with
| 0     => 4
| 1     => 3
| (n+2) => fibonacci (n+1) + fibonacci n

def units_digit (n : ℕ) : ℕ :=
n % 10

theorem fibonacci_units_digit : units_digit (fibonacci (fibonacci 10)) = 3 := by
  sorry

end fibonacci_units_digit_l242_242795


namespace cost_of_book_l242_242968

-- Definitions based on the conditions
def cost_pen : ℕ := 4
def cost_ruler : ℕ := 1
def fifty_dollar_bill : ℕ := 50
def change_received : ℕ := 20
def total_spent : ℕ := fifty_dollar_bill - change_received

-- Problem Statement: Prove the cost of the book
theorem cost_of_book : ∀ (cost_pen cost_ruler total_spent : ℕ), 
  total_spent = 50 - 20 → cost_pen = 4 → cost_ruler = 1 →
  (total_spent - (cost_pen + cost_ruler) = 25) :=
by
  intros cost_pen cost_ruler total_spent h1 h2 h3
  sorry

end cost_of_book_l242_242968


namespace binom_60_3_eq_34220_l242_242910

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l242_242910


namespace reciprocal_of_mixed_number_l242_242800

def mixed_number := -1 - (4 / 5)

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem reciprocal_of_mixed_number : reciprocal mixed_number = -5 / 9 := 
by
  sorry

end reciprocal_of_mixed_number_l242_242800


namespace original_loaf_had_27_slices_l242_242717

def original_slices : ℕ :=
  let slices_andy_ate := 3 * 2
  let slices_for_toast := 2 * 10
  let slices_left := 1
  slices_andy_ate + slices_for_toast + slices_left

theorem original_loaf_had_27_slices (n : ℕ) (slices_andy_ate : ℕ) (slices_for_toast : ℕ) (slices_left : ℕ) :
  slices_andy_ate = 6 → slices_for_toast = 20 → slices_left = 1 → n = slices_andy_ate + slices_for_toast + slices_left → n = 27 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

-- Verifying the statement
example : original_slices = 27 := by
  have h1 : 3 * 2 = 6 := rfl
  have h2 : 2 * 10 = 20 := rfl
  have h3 : 1 = 1 := rfl
  exact original_loaf_had_27_slices original_slices 6 20 1 h1 h2 h3 rfl

end original_loaf_had_27_slices_l242_242717


namespace battery_current_l242_242086

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l242_242086


namespace thomas_total_blocks_l242_242371

theorem thomas_total_blocks :
  let stack1 := 7 in
  let stack2 := stack1 + 3 in
  let stack3 := stack2 - 6 in
  let stack4 := stack3 + 10 in
  let stack5 := stack2 * 2 in
  stack1 + stack2 + stack3 + stack4 + stack5 = 55 :=
by
  let stack1 := 7
  let stack2 := stack1 + 3
  let stack3 := stack2 - 6
  let stack4 := stack3 + 10
  let stack5 := stack2 * 2
  have : stack1 + stack2 + stack3 + stack4 + stack5 = 7 + 10 + 4 + 14 + 20 := by rfl
  rw [this]
  norm_num
  sorry

end thomas_total_blocks_l242_242371


namespace value_of_b_minus_a_l242_242436

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x / 2)

theorem value_of_b_minus_a (a b : ℝ) (h1 : ∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1 : ℝ) 2) (h2 : ∀ x, f x = 2 * Real.sin (x / 2)) : 
  b - a ≠ 14 * Real.pi / 3 :=
sorry

end value_of_b_minus_a_l242_242436


namespace perfect_square_expression_l242_242424

theorem perfect_square_expression (n : ℕ) (h : 7 ≤ n) : ∃ k : ℤ, (n + 2) ^ 2 = k ^ 2 :=
by 
  sorry

end perfect_square_expression_l242_242424


namespace expected_win_l242_242867

-- Definitions of conditions
def sides := fin 8 -- Finite type representing the 8 sides of the die

-- Function to calculate the win amount given a roll
def win_amount (n : sides) : ℝ := 8 - n.val

-- Probability of each side for a fair die
def probability : ℝ := 1 / 8

-- Definition of expected value calculation
def expected_value : ℝ := ∑ n in (finset.univ : finset sides), probability * (win_amount n)

-- Theorem statement
theorem expected_win : expected_value = 3.5 :=
by sorry

end expected_win_l242_242867


namespace triangle_area_hypotenuse_l242_242967

-- Definitions of the conditions
def DE : ℝ := 40
def DF : ℝ := 30
def angleD : ℝ := 90

-- Proof statement
theorem triangle_area_hypotenuse :
  let Area : ℝ := 1 / 2 * DE * DF
  let EF : ℝ := Real.sqrt (DE^2 + DF^2)
  Area = 600 ∧ EF = 50 := by
  sorry

end triangle_area_hypotenuse_l242_242967


namespace sally_last_10_shots_made_l242_242681

def sally_initial_shots : ℕ := 30
def sally_initial_success_rate : ℝ := 0.60
def sally_additional_shots : ℕ := 10
def sally_final_success_rate : ℝ := 0.65

theorem sally_last_10_shots_made (x : ℕ) 
  (h1 : sally_initial_success_rate * sally_initial_shots = 18)
  (h2 : sally_final_success_rate * (sally_initial_shots + sally_additional_shots) = 26) :
  x = 8 :=
by
  sorry

end sally_last_10_shots_made_l242_242681


namespace find_a_plus_b_l242_242432

theorem find_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, |x - 2| > 1 ↔ x^2 + a * x + b > 0) →
  a + b = -1 :=
by {
  assume h : ∀ x : ℝ, |x - 2| > 1 ↔ x^2 + a * x + b > 0,
  sorry
}

end find_a_plus_b_l242_242432


namespace expected_value_of_winnings_l242_242844

theorem expected_value_of_winnings (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 8) :
  ((∑ i in finset.range 8, (8 - i) * (1 / 8)) = 3.5) :=
by
  sorry

end expected_value_of_winnings_l242_242844


namespace households_used_both_brands_l242_242014

theorem households_used_both_brands (X : ℕ) : 
  (80 + 60 + X + 3 * X = 260) → X = 30 :=
by
  sorry

end households_used_both_brands_l242_242014


namespace students_neither_art_nor_music_l242_242332

def total_students := 75
def art_students := 45
def music_students := 50
def both_art_and_music := 30

theorem students_neither_art_nor_music : 
  total_students - (art_students - both_art_and_music + music_students - both_art_and_music + both_art_and_music) = 10 :=
by 
  sorry

end students_neither_art_nor_music_l242_242332


namespace binomial_60_3_l242_242906

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l242_242906


namespace bullet_speed_difference_l242_242657

def bullet_speed_in_same_direction (v_h v_b : ℝ) : ℝ :=
  v_b + v_h

def bullet_speed_in_opposite_direction (v_h v_b : ℝ) : ℝ :=
  v_b - v_h

theorem bullet_speed_difference (v_h v_b : ℝ) (h_h : v_h = 20) (h_b : v_b = 400) :
  bullet_speed_in_same_direction v_h v_b - bullet_speed_in_opposite_direction v_h v_b = 40 :=
by
  rw [h_h, h_b]
  sorry

end bullet_speed_difference_l242_242657


namespace jose_profit_share_l242_242523

theorem jose_profit_share :
  ∀ (Tom_investment Jose_investment total_profit month_investment_tom month_investment_jose total_month_investment: ℝ),
    Tom_investment = 30000 →
    ∃ (months_tom months_jose : ℝ), months_tom = 12 ∧ months_jose = 10 →
      Jose_investment = 45000 →
      total_profit = 72000 →
      month_investment_tom = Tom_investment * months_tom →
      month_investment_jose = Jose_investment * months_jose →
      total_month_investment = month_investment_tom + month_investment_jose →
      (Jose_investment * months_jose / total_month_investment) * total_profit = 40000 :=
by
  sorry

end jose_profit_share_l242_242523


namespace evaluate_h_j_l242_242776

def h (x : ℝ) : ℝ := 3 * x - 4
def j (x : ℝ) : ℝ := x - 2

theorem evaluate_h_j : h (2 + j 3) = 5 := by
  sorry

end evaluate_h_j_l242_242776


namespace larger_pile_toys_l242_242597

-- Define the conditions
def total_toys (small_pile large_pile : ℕ) : Prop := small_pile + large_pile = 120
def larger_pile (small_pile large_pile : ℕ) : Prop := large_pile = 2 * small_pile

-- Define the proof problem
theorem larger_pile_toys (small_pile large_pile : ℕ) (h1 : total_toys small_pile large_pile) (h2 : larger_pile small_pile large_pile) : 
  large_pile = 80 := by
  sorry

end larger_pile_toys_l242_242597


namespace blend_pieces_eq_two_l242_242973

variable (n_silk n_cashmere total_pieces : ℕ)

def luther_line := n_silk = 10 ∧ n_cashmere = n_silk / 2 ∧ total_pieces = 13

theorem blend_pieces_eq_two : luther_line n_silk n_cashmere total_pieces → (n_cashmere - (total_pieces - n_silk) = 2) :=
by
  intros
  sorry

end blend_pieces_eq_two_l242_242973


namespace ammonium_chloride_reacts_with_potassium_hydroxide_l242_242146

/-- Prove that 1 mole of ammonium chloride is required to react with 
    1 mole of potassium hydroxide to form 1 mole of ammonia, 
    1 mole of water, and 1 mole of potassium chloride, 
    given the balanced chemical equation:
    NH₄Cl + KOH → NH₃ + H₂O + KCl
-/
theorem ammonium_chloride_reacts_with_potassium_hydroxide :
    ∀ (NH₄Cl KOH NH₃ H₂O KCl : ℕ), 
    (NH₄Cl + KOH = NH₃ + H₂O + KCl) → 
    (NH₄Cl = 1) → 
    (KOH = 1) → 
    (NH₃ = 1) → 
    (H₂O = 1) → 
    (KCl = 1) → 
    NH₄Cl = 1 :=
by
  intros
  sorry

end ammonium_chloride_reacts_with_potassium_hydroxide_l242_242146


namespace smallest_square_value_l242_242611

theorem smallest_square_value (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h₁ : ∃ r : ℕ, 15 * a + 16 * b = r^2) (h₂ : ∃ s : ℕ, 16 * a - 15 * b = s^2) :
  ∃ (m : ℕ), m = 481^2 ∧ (15 * a + 16 * b = m ∨ 16 * a - 15 * b = m) :=
  sorry

end smallest_square_value_l242_242611


namespace five_equal_size_right_triangles_l242_242537

-- Definitions to represent the problem
structure Point2D :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A : Point2D)
(B : Point2D)
(C : Point2D)

structure Pentagon :=
(A : Point2D)
(B : Point2D)
(C : Point2D)
(D : Point2D)
(E : Point2D)
(O : Point2D)

-- Check if all triangles are right triangles.
def isRightTriangle (T : Triangle) : Prop :=
  ∃ (A B C : ℝ), T.A = ⟨A, 0⟩ ∧ T.B = ⟨0, B⟩ ∧ T.C = ⟨0, 0⟩ ∧ 
  (A ≠ 0 ∧ B ≠ 0)

def pentagonWithInnerTrianglesRight (P : Pentagon) : Prop :=
  ∀ (i j : ℕ) (hi : (0 ≤ i ∧ i < j) ∧ (j < 5)),
  let vertex := [P.A, P.B, P.C, P.D, P.E] in
  let triangles := [
    Triangle.mk P.A P.B P.O,
    Triangle.mk P.B P.C P.O,
    Triangle.mk P.C P.D P.O,
    Triangle.mk P.D P.E P.O,
    Triangle.mk P.E P.A P.O
  ] in
  isRightTriangle (triangles !! i) ∧ isRightTriangle (triangles !! j)

theorem five_equal_size_right_triangles (P : Pentagon) :
  pentagonWithInnerTrianglesRight P :=
by
  sorry

end five_equal_size_right_triangles_l242_242537


namespace three_tenths_of_number_l242_242446

theorem three_tenths_of_number (N : ℝ) (h : (1/3) * (1/4) * N = 15) : (3/10) * N = 54 :=
sorry

end three_tenths_of_number_l242_242446


namespace current_when_resistance_is_12_l242_242030

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l242_242030


namespace problem_1_1_eval_l242_242013

noncomputable def E (a b c : ℝ) : ℝ :=
  let A := (1/a - 1/(b+c))/(1/a + 1/(b+c))
  let B := 1 + (b^2 + c^2 - a^2)/(2*b*c)
  let C := (a - b - c)/(a * b * c)
  (A * B) / C

theorem problem_1_1_eval :
  E 0.02 (-11.05) 1.07 = 0.1 :=
by
  -- Proof goes here
  sorry

end problem_1_1_eval_l242_242013


namespace battery_current_l242_242116

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l242_242116


namespace smallest_even_natural_number_l242_242821

theorem smallest_even_natural_number (a : ℕ) :
  ( ∃ a, a % 2 = 0 ∧
    (a + 1) % 3 = 0 ∧
    (a + 2) % 5 = 0 ∧
    (a + 3) % 7 = 0 ∧
    (a + 4) % 11 = 0 ∧
    (a + 5) % 13 = 0 ) → 
  a = 788 := by
  sorry

end smallest_even_natural_number_l242_242821


namespace binomial_30_3_l242_242701

-- Defining the binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Statement of the problem in Lean 4
theorem binomial_30_3 : binomial 30 3 = 12180 :=
by
  sorry

end binomial_30_3_l242_242701


namespace jane_uses_40_ribbons_l242_242602

theorem jane_uses_40_ribbons :
  (∀ dresses1 dresses2 ribbons_per_dress, 
  dresses1 = 2 * 7 ∧ 
  dresses2 = 3 * 2 → 
  ribbons_per_dress = 2 → 
  (dresses1 + dresses2) * ribbons_per_dress = 40)
:= 
by 
  sorry

end jane_uses_40_ribbons_l242_242602


namespace cubic_ineq_l242_242196

theorem cubic_ineq (x p q : ℝ) (h : x^3 + p * x + q = 0) : 4 * q * x ≤ p^2 := 
  sorry

end cubic_ineq_l242_242196


namespace max_square_test_plots_l242_242868

theorem max_square_test_plots (h_field_dims : (24 : ℝ) = 24 ∧ (52 : ℝ) = 52)
    (h_total_fencing : 1994 = 1994)
    (h_partitioning : ∀ (n : ℤ), n % 6 = 0 → n ≤ 19 → 
      (104 * n - 76 ≤ 1994) → (n / 6 * 13)^2 = 702) :
    ∃ n : ℤ, (n / 6 * 13)^2 = 702 := sorry

end max_square_test_plots_l242_242868


namespace num_possible_values_for_n_l242_242637

open Real

noncomputable def count_possible_values_for_n : ℕ :=
  let log2 := log 2
  let log2_9 := log 9 / log2
  let log2_50 := log 50 / log2
  let range_n := ((6 : ℕ), 450)
  let count := range_n.2 - range_n.1 + 1
  count

theorem num_possible_values_for_n :
  count_possible_values_for_n = 445 :=
by
  sorry

end num_possible_values_for_n_l242_242637


namespace battery_current_l242_242021

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l242_242021


namespace ratio_of_length_to_perimeter_is_one_over_four_l242_242875

-- We define the conditions as given in the problem.
def room_length_1 : ℕ := 23 -- length of the rectangle in feet
def room_width_1 : ℕ := 15  -- width of the rectangle in feet
def room_width_2 : ℕ := 8   -- side of the square in feet

-- Total dimensions after including the square
def total_length : ℕ := room_length_1  -- total length remains the same
def total_width : ℕ := room_width_1 + room_width_2  -- width is sum of widths

-- Defining the perimeter
def perimeter (length width : ℕ) : ℕ := 2 * length + 2 * width

-- Calculate the ratio
def length_to_perimeter_ratio (length perimeter : ℕ) : ℚ := length / perimeter

-- Theorem to prove the desired ratio is 1:4
theorem ratio_of_length_to_perimeter_is_one_over_four : 
  length_to_perimeter_ratio total_length (perimeter total_length total_width) = 1 / 4 :=
by
  -- Proof code would go here
  sorry

end ratio_of_length_to_perimeter_is_one_over_four_l242_242875


namespace current_when_resistance_is_12_l242_242028

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l242_242028


namespace integer_triplets_prime_l242_242192

theorem integer_triplets_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ sol : ℕ, ((∃ (x y z : ℤ), (3 * x + y + z) * (x + 2 * y + z) * (x + y + z) = p) ∧
  if p = 2 then sol = 4 else sol = 12) :=
by
  sorry

end integer_triplets_prime_l242_242192


namespace battery_current_l242_242026

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l242_242026


namespace solve_equation_l242_242983

theorem solve_equation (n : ℝ) :
  (3 - 2 * n) / (n + 2) + (3 * n - 9) / (3 - 2 * n) = 2 ↔ 
  n = (25 + Real.sqrt 13) / 18 ∨ n = (25 - Real.sqrt 13) / 18 :=
by
  sorry

end solve_equation_l242_242983


namespace dad_additional_money_l242_242281

-- Define the conditions in Lean
def daily_savings : ℕ := 35
def days : ℕ := 7
def total_savings_before_doubling := daily_savings * days
def doubled_savings := 2 * total_savings_before_doubling
def total_amount_after_7_days : ℕ := 500

-- Define the theorem to prove
theorem dad_additional_money : (total_amount_after_7_days - doubled_savings) = 10 := by
  sorry

end dad_additional_money_l242_242281


namespace current_value_l242_242074

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l242_242074


namespace current_at_R_12_l242_242047

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l242_242047


namespace max_sum_of_positive_integers_with_product_144_l242_242992

theorem max_sum_of_positive_integers_with_product_144 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 144 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 75 := 
by
  sorry

end max_sum_of_positive_integers_with_product_144_l242_242992


namespace hyperbola_standard_equation_equation_of_line_L_l242_242428

open Real

noncomputable def hyperbola (x y : ℝ) : Prop :=
  y^2 - x^2 / 3 = 1

noncomputable def focus_on_y_axis := ∃ c : ℝ, c = 2

noncomputable def asymptote (x y : ℝ) : Prop := 
  y = sqrt 3 / 3 * x ∨ y = - sqrt 3 / 3 * x

noncomputable def point_A := (1, 1 / 2)

noncomputable def line_L (x y : ℝ) : Prop :=
  4 * x - 6 * y - 1 = 0

theorem hyperbola_standard_equation :
  ∃ (x y: ℝ), hyperbola x y :=
sorry

theorem equation_of_line_L :
  ∀ (x y : ℝ), point_A = (1, 1 / 2) ∧ line_L x y :=
sorry

end hyperbola_standard_equation_equation_of_line_L_l242_242428


namespace cannot_achieve_80_cents_with_six_coins_l242_242205

theorem cannot_achieve_80_cents_with_six_coins:
  ¬ (∃ (p n d : ℕ), p + n + d = 6 ∧ p + 5 * n + 10 * d = 80) :=
by
  sorry

end cannot_achieve_80_cents_with_six_coins_l242_242205


namespace heartsuit_fraction_l242_242177

-- Define the operation heartsuit
def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

-- Define the proof statement
theorem heartsuit_fraction :
  (heartsuit 2 4) / (heartsuit 4 2) = 2 :=
by
  -- We use 'sorry' to skip the actual proof steps
  sorry

end heartsuit_fraction_l242_242177


namespace closest_point_on_line_l242_242725

theorem closest_point_on_line (x y: ℚ) (h1: y = -4 * x + 3) (h2: ∀ p q: ℚ, y = -4 * p + 3 ∧ y = q * (-4 * p) - q * (-4 * 1 + 0)): (x, y) = (-1 / 17, 55 / 17) :=
sorry

end closest_point_on_line_l242_242725


namespace total_students_l242_242823

variables (B G : ℕ)
variables (two_thirds_boys : 2 * B = 3 * 400)
variables (three_fourths_girls : 3 * G = 4 * 150)
variables (total_participants : B + G = 800)

theorem total_students (B G : ℕ)
  (two_thirds_boys : 2 * B = 3 * 400)
  (three_fourths_girls : 3 * G = 4 * 150)
  (total_participants : B + G = 800) :
  B + G = 800 :=
by
  sorry

end total_students_l242_242823


namespace binom_30_3_eq_4060_l242_242696

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := 
  by sorry

end binom_30_3_eq_4060_l242_242696


namespace solution_set_l242_242580

def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem solution_set :
  {x | f x ≥ x^2 - 8 * x + 15} = {2} ∪ {x | x > 6} :=
by
  sorry

end solution_set_l242_242580


namespace min_possible_value_of_x_l242_242473

theorem min_possible_value_of_x :
  ∀ (x y : ℝ),
  (69 + 53 + 69 + 71 + 78 + x + y) / 7 = 66 →
  (∀ y ≤ 100, x ≥ 0) →
  x ≥ 22 :=
by
  intros x y h_avg h_y 
  -- proof steps go here
  sorry

end min_possible_value_of_x_l242_242473


namespace cylinder_volume_increase_l242_242239

theorem cylinder_volume_increase (r h : ℝ) (V : ℝ) : 
  V = π * r^2 * h → 
  (3 * h) * (2 * r)^2 * π = 12 * V := by
    sorry

end cylinder_volume_increase_l242_242239


namespace hyperbola_asymptotes_l242_242179

theorem hyperbola_asymptotes (x y : ℝ) (E : x^2 / 4 - y^2 = 1) :
  y = (1 / 2) * x ∨ y = -(1 / 2) * x :=
sorry

end hyperbola_asymptotes_l242_242179


namespace pow_modulus_l242_242420

theorem pow_modulus : (5 ^ 2023) % 11 = 3 := by
  sorry

end pow_modulus_l242_242420


namespace find_s_l242_242500

theorem find_s (x y : Real -> Real) : 
  (x 2 = 2 ∧ y 2 = 5) ∧ 
  (x 6 = 6 ∧ y 6 = 17) ∧ 
  (x 10 = 10 ∧ y 10 = 29) ∧ 
  (∀ x, y x = 3 * x - 1) -> 
  (y 34 = 101) := 
by 
  sorry

end find_s_l242_242500


namespace transfer_balls_l242_242546

theorem transfer_balls (X Y q p b : ℕ) (h : p + b = q) :
  b = q - p :=
by
  sorry

end transfer_balls_l242_242546


namespace intersection_of_sets_l242_242756

noncomputable def setM : Set ℝ := { x | x + 1 > 0 }
noncomputable def setN : Set ℝ := { x | 2 * x - 1 < 0 }

theorem intersection_of_sets : setM ∩ setN = { x : ℝ | -1 < x ∧ x < 1 / 2 } := by
  sorry

end intersection_of_sets_l242_242756


namespace transformation_correct_l242_242298

theorem transformation_correct (a b : ℝ) (h₁ : 3 * a = 2 * b) (h₂ : a ≠ 0) (h₃ : b ≠ 0) :
  a / 2 = b / 3 :=
sorry

end transformation_correct_l242_242298


namespace number_of_white_dogs_l242_242368

noncomputable def number_of_brown_dogs : ℕ := 20
noncomputable def number_of_black_dogs : ℕ := 15
noncomputable def total_number_of_dogs : ℕ := 45

theorem number_of_white_dogs : total_number_of_dogs - (number_of_brown_dogs + number_of_black_dogs) = 10 := by
  sorry

end number_of_white_dogs_l242_242368


namespace pow_modulus_l242_242421

theorem pow_modulus : (5 ^ 2023) % 11 = 3 := by
  sorry

end pow_modulus_l242_242421


namespace sin_cos_105_l242_242802

theorem sin_cos_105 (h1 : ∀ x : ℝ, Real.sin x * Real.cos x = 1 / 2 * Real.sin (2 * x))
                    (h2 : ∀ x : ℝ, Real.sin (180 * Real.pi / 180 + x) = - Real.sin x)
                    (h3 : Real.sin (30 * Real.pi / 180) = 1 / 2) :
  Real.sin (105 * Real.pi / 180) * Real.cos (105 * Real.pi / 180) = - 1 / 4 :=
by
  sorry

end sin_cos_105_l242_242802


namespace find_x_given_conditions_l242_242760

variable (x y z : ℝ)

theorem find_x_given_conditions
  (h1: x * y / (x + y) = 4)
  (h2: x * z / (x + z) = 9)
  (h3: y * z / (y + z) = 16)
  (h_pos: 0 < x ∧ 0 < y ∧ 0 < z)
  (h_distinct: x ≠ y ∧ x ≠ z ∧ y ≠ z) :
  x = 384/21 :=
sorry

end find_x_given_conditions_l242_242760


namespace find_nat_numbers_for_divisibility_l242_242933

theorem find_nat_numbers_for_divisibility :
  ∃ (a b : ℕ), (7^3 ∣ a^2 + a * b + b^2) ∧ (¬ 7 ∣ a) ∧ (¬ 7 ∣ b) ∧ (a = 1) ∧ (b = 18) := by
  sorry

end find_nat_numbers_for_divisibility_l242_242933


namespace hyperbola_real_axis_length_l242_242503

theorem hyperbola_real_axis_length : 
  (∃ (x y : ℝ), (x^2 / 2) - (y^2 / 4) = 1) → real_axis_length = 2 * Real.sqrt 2 :=
by
  -- Proof is omitted
  sorry

end hyperbola_real_axis_length_l242_242503


namespace joan_video_game_spending_l242_242182

theorem joan_video_game_spending:
  let basketball_game := 5.20
  let racing_game := 4.23
  basketball_game + racing_game = 9.43 := 
by
  sorry

end joan_video_game_spending_l242_242182


namespace distinct_values_count_l242_242138

-- Declare the sequence
noncomputable def sequence (x : ℝ) : ℕ → ℝ
| 0       := x
| 1       := 3000
| (n + 2) := (sequence n * sequence (n + 1) - 1)

def appears_3001 (x : ℝ) : Prop := ∃ n : ℕ, sequence x n = 3001

-- Main statement
theorem distinct_values_count :
  {x : ℝ | appears_3001 x}.card  = 4 :=
sorry

end distinct_values_count_l242_242138


namespace smallest_positive_period_range_on_interval_l242_242951

noncomputable def f (x : ℝ) : ℝ :=
  sqrt 2 * sin (x / 2) * cos (x / 2) - sqrt 2 * sin (x / 2) ^ 2

theorem smallest_positive_period
  : ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

theorem range_on_interval : 
  ∀ x ∈ set.Icc (-π) 0, f x ∈ set.Icc (-1 - sqrt 2 / 2) 0 :=
sorry

end smallest_positive_period_range_on_interval_l242_242951


namespace change_occurs_in_3_years_l242_242200

theorem change_occurs_in_3_years (P A1 A2 : ℝ) (R T : ℝ) (h1 : P = 825) (h2 : A1 = 956) (h3 : A2 = 1055)
    (h4 : A1 = P + (P * R * T) / 100)
    (h5 : A2 = P + (P * (R + 4) * T) / 100) : T = 3 :=
by
  sorry

end change_occurs_in_3_years_l242_242200


namespace binomial_30_3_l242_242689

theorem binomial_30_3 : nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l242_242689


namespace expected_win_l242_242865

-- Definitions of conditions
def sides := fin 8 -- Finite type representing the 8 sides of the die

-- Function to calculate the win amount given a roll
def win_amount (n : sides) : ℝ := 8 - n.val

-- Probability of each side for a fair die
def probability : ℝ := 1 / 8

-- Definition of expected value calculation
def expected_value : ℝ := ∑ n in (finset.univ : finset sides), probability * (win_amount n)

-- Theorem statement
theorem expected_win : expected_value = 3.5 :=
by sorry

end expected_win_l242_242865


namespace current_value_l242_242064

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l242_242064


namespace binom_60_3_l242_242901

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l242_242901


namespace coefficient_x3_l242_242335

theorem coefficient_x3 (x : ℕ) : 
  (nat.choose 5 3 - nat.choose 6 3) = -10 :=
by sorry

end coefficient_x3_l242_242335


namespace xy_system_solution_l242_242305

theorem xy_system_solution (x y : ℝ) (h₁ : x + 5 * y = 6) (h₂ : 3 * x - y = 2) : x + y = 2 := 
by 
  sorry

end xy_system_solution_l242_242305


namespace new_average_is_21_l242_242830

def initial_number_of_students : ℕ := 30
def late_students : ℕ := 4
def initial_jumping_students : ℕ := initial_number_of_students - late_students
def initial_average_score : ℕ := 20
def late_student_scores : List ℕ := [26, 27, 28, 29]
def total_jumps_initial_students : ℕ := initial_jumping_students * initial_average_score
def total_jumps_late_students : ℕ := late_student_scores.sum
def total_jumps_all_students : ℕ := total_jumps_initial_students + total_jumps_late_students
def new_average_score : ℕ := total_jumps_all_students / initial_number_of_students

theorem new_average_is_21 :
  new_average_score = 21 :=
sorry

end new_average_is_21_l242_242830


namespace triangle_inscribed_relation_l242_242520

noncomputable def herons_area (p a b c : ℝ) : ℝ := (p * (p - a) * (p - b) * (p - c)).sqrt

theorem triangle_inscribed_relation
  (S S' p p' : ℝ)
  (a b c a' b' c' r : ℝ)
  (h1 : r = S / p)
  (h2 : r = S' / p')
  (h3 : S = herons_area p a b c)
  (h4 : S' = herons_area p' a' b' c') :
  (p - a) * (p - b) * (p - c) / p = (p' - a') * (p' - b') * (p' - c') / p' :=
by sorry

end triangle_inscribed_relation_l242_242520


namespace part1_part2_l242_242437

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) (b : ℝ) := 0.5 * x^2 - b * x
noncomputable def h (x : ℝ) (b : ℝ) := f x + g x b

theorem part1 (b : ℝ) :
  (∃ (tangent_point : ℝ),
    tangent_point = 1 ∧
    deriv f tangent_point = 1 ∧
    f tangent_point = 0 ∧
    ∃ (y_tangent : ℝ → ℝ), (∀ (x : ℝ), y_tangent x = x - 1) ∧
    ∃ (tangent_for_g : ℝ), (∀ (x : ℝ), y_tangent x = g x b)
  ) → false :=
sorry 

theorem part2 (b : ℝ) :
  ¬ (∀ (x : ℝ) (hx : 0 < x), deriv (h x) b = 0 → deriv (h x) b < 0) →
  2 < b :=
sorry

end part1_part2_l242_242437


namespace probability_of_selecting_double_l242_242117

-- Define the conditions and the question
def total_integers : ℕ := 13

def number_of_doubles : ℕ := total_integers

def total_pairings : ℕ := 
  (total_integers * (total_integers + 1)) / 2

def probability_double : ℚ := 
  number_of_doubles / total_pairings

-- Statement to be proved 
theorem probability_of_selecting_double : 
  probability_double = 1/7 := 
sorry

end probability_of_selecting_double_l242_242117


namespace stack_crates_height_l242_242510

theorem stack_crates_height :
  ∀ a b c : ℕ, (3 * a + 4 * b + 5 * c = 50) ∧ (a + b + c = 12) → false :=
by
  sorry

end stack_crates_height_l242_242510


namespace battery_current_when_resistance_12_l242_242076

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l242_242076


namespace geometric_sequence_formula_l242_242571

noncomputable def a_n (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a_1 * q^n

theorem geometric_sequence_formula
  (a_1 q : ℝ)
  (h_pos : ∀ n : ℕ, a_n a_1 q n > 0)
  (h_4_eq : a_n a_1 q 4 = (a_n a_1 q 2)^2)
  (h_2_4_sum : a_n a_1 q 2 + a_n a_1 q 4 = 5 / 16) :
  ∀ n : ℕ, a_n a_1 q n = ((1 : ℝ) / 2) ^ n :=
sorry

end geometric_sequence_formula_l242_242571


namespace rectangle_distances_sum_l242_242892

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem rectangle_distances_sum :
  let A : (ℝ × ℝ) := (0, 0)
  let B : (ℝ × ℝ) := (3, 0)
  let C : (ℝ × ℝ) := (3, 4)
  let D : (ℝ × ℝ) := (0, 4)

  let M : (ℝ × ℝ) := ((B.1 + A.1) / 2, (B.2 + A.2) / 2)
  let N : (ℝ × ℝ) := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let O : (ℝ × ℝ) := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let P : (ℝ × ℝ) := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)

  distance A.1 A.2 M.1 M.2 + distance A.1 A.2 N.1 N.2 + distance A.1 A.2 O.1 O.2 + distance A.1 A.2 P.1 P.2 = 7.77 + Real.sqrt 13 :=
sorry

end rectangle_distances_sum_l242_242892


namespace arithmetic_mean_of_set_l242_242746

theorem arithmetic_mean_of_set {x : ℝ} (mean_eq_12 : (8 + 16 + 20 + x + 12) / 5 = 12) : x = 4 :=
by
  sorry

end arithmetic_mean_of_set_l242_242746


namespace binomial_60_3_eq_34220_l242_242914

open Nat

theorem binomial_60_3_eq_34220 : choose 60 3 = 34220 := 
by
  sorry

end binomial_60_3_eq_34220_l242_242914


namespace expected_value_of_winnings_l242_242846

theorem expected_value_of_winnings (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 8) :
  ((∑ i in finset.range 8, (8 - i) * (1 / 8)) = 3.5) :=
by
  sorry

end expected_value_of_winnings_l242_242846


namespace largest_expression_is_D_l242_242002

-- Define each expression
def exprA : ℤ := 3 - 1 + 4 + 6
def exprB : ℤ := 3 - 1 * 4 + 6
def exprC : ℤ := 3 - (1 + 4) * 6
def exprD : ℤ := 3 - 1 + 4 * 6
def exprE : ℤ := 3 * (1 - 4) + 6

-- The theorem stating that exprD is the largest value among the given expressions.
theorem largest_expression_is_D : 
  exprD = 26 ∧ 
  exprD > exprA ∧ 
  exprD > exprB ∧ 
  exprD > exprC ∧ 
  exprD > exprE := 
by {
  sorry
}

end largest_expression_is_D_l242_242002


namespace remainder_5_pow_2023_mod_11_l242_242418

theorem remainder_5_pow_2023_mod_11 : (5^2023) % 11 = 4 :=
by
  have h1 : 5^2 % 11 = 25 % 11 := sorry
  have h2 : 25 % 11 = 3 := sorry
  have h3 : (3^5) % 11 = 1 := sorry
  have h4 : 3^1011 % 11 = ((3^5)^202 * 3) % 11 := sorry
  have h5 : ((3^5)^202 * 3) % 11 = (1^202 * 3) % 11 := sorry
  have h6 : (1^202 * 3) % 11 = 3 % 11 := sorry
  have h7 : (5^2023) % 11 = (3 * 5) % 11 := sorry
  have h8 : (3 * 5) % 11 = 15 % 11 := sorry
  have h9 : 15 % 11 = 4 := sorry
  exact h9

end remainder_5_pow_2023_mod_11_l242_242418


namespace total_cost_of_apples_l242_242888

def original_price_per_pound : ℝ := 1.6
def price_increase_percentage : ℝ := 0.25
def number_of_family_members : ℕ := 4
def pounds_per_person : ℝ := 2

theorem total_cost_of_apples : 
  let new_price_per_pound := original_price_per_pound * (1 + price_increase_percentage)
  let total_pounds := pounds_per_person * number_of_family_members
  let total_cost := total_pounds * new_price_per_pound
  total_cost = 16 := by
  sorry

end total_cost_of_apples_l242_242888


namespace ratio_ac_l242_242799

variable {a b c d : ℝ}

-- Given the conditions
axiom ratio_ab : a / b = 5 / 4
axiom ratio_cd : c / d = 4 / 3
axiom ratio_db : d / b = 1 / 5

-- The statement to prove
theorem ratio_ac : a / c = 75 / 16 :=
  by sorry

end ratio_ac_l242_242799


namespace find_current_l242_242108

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l242_242108


namespace expected_value_of_win_is_3point5_l242_242838

noncomputable def expected_value_win : ℝ := 
  ∑ n in Finset.range 8, (1 / 8 : ℝ) * (8 - n)

theorem expected_value_of_win_is_3point5 : expected_value_win = 3.5 := 
by 
  sorry

end expected_value_of_win_is_3point5_l242_242838


namespace contradiction_proof_l242_242808

theorem contradiction_proof (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) : ¬ (¬ (a > 0) ∨ ¬ (b > 0) ∨ ¬ (c > 0)) → false :=
by sorry

end contradiction_proof_l242_242808


namespace train_crossing_time_l242_242664

variable (length_train : ℝ) (time_pole : ℝ) (length_platform : ℝ) (time_platform : ℝ)

-- Given conditions
def train_conditions := 
  length_train = 300 ∧
  time_pole = 14 ∧
  length_platform = 535.7142857142857

-- Theorem statement
theorem train_crossing_time (h : train_conditions length_train time_pole length_platform) :
  time_platform = 39 := sorry

end train_crossing_time_l242_242664


namespace general_term_formula_l242_242497

-- Conditions: sequence \(\frac{1}{2}\), \(\frac{1}{3}\), \(\frac{1}{4}\), \(\frac{1}{5}, \ldots\)
-- Let seq be the sequence in question.

def seq (n : ℕ) : ℚ := 1 / (n + 1)

-- Question: prove the general term formula is \(\frac{1}{n+1}\)
theorem general_term_formula (n : ℕ) : seq n = 1 / (n + 1) :=
by
  -- Proof goes here
  sorry

end general_term_formula_l242_242497


namespace problem1_problem2_l242_242303

variable (a b : ℝ)

-- (1) Prove a + b = 2 given the conditions
theorem problem1 (h1 : a > 0) (h2 : b > 0) (h3 : ∀ x : ℝ, abs (x - a) + abs (x + b) ≥ 2) : a + b = 2 :=
sorry

-- (2) Prove it is not possible for both a^2 + a > 2 and b^2 + b > 2 to hold simultaneously
theorem problem2 (h1: a + b = 2) (h2 : a^2 + a > 2) (h3 : b^2 + b > 2) : False :=
sorry

end problem1_problem2_l242_242303


namespace current_value_l242_242020

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l242_242020


namespace find_ordered_pair_l242_242561

theorem find_ordered_pair :
  ∃ x y : ℚ, 
  (x + 2 * y = (7 - x) + (7 - 2 * y)) ∧
  (3 * x - 2 * y = (x + 2) - (2 * y + 2)) ∧
  x = 0 ∧ 
  y = 7 / 2 :=
by
  sorry

end find_ordered_pair_l242_242561


namespace g_at_pi_over_4_l242_242167

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2) / 2 * Real.sin (2 * x) + (Real.sqrt 6) / 2 * Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 4)

theorem g_at_pi_over_4 : g (Real.pi / 4) = (Real.sqrt 6) / 2 := by
  sorry

end g_at_pi_over_4_l242_242167


namespace expected_value_of_win_is_correct_l242_242833

noncomputable def expected_value_of_win : ℝ :=
  (1 / 8) * (8 - 1) + (1 / 8) * (8 - 2) + (1 / 8) * (8 - 3) +
  (1 / 8) * (8 - 4) + (1 / 8) * (8 - 5) + (1 / 8) * (8 - 6) +
  (1 / 8) * (8 - 7) + (1 / 8) * (8 - 8)

theorem expected_value_of_win_is_correct :
  expected_value_of_win = 3.5 :=
by
  sorry

end expected_value_of_win_is_correct_l242_242833


namespace find_extrema_l242_242294

noncomputable def y (x : ℝ) := (Real.sin (3 * x))^2

theorem find_extrema : 
  ∃ (x : ℝ), (0 < x ∧ x < 0.6) ∧ (∀ ε > 0, ε < 0.6 - x → y (x + ε) ≤ y x ∧ y (x - ε) ≤ y x) ∧ x = Real.pi / 6 :=
by
  sorry

end find_extrema_l242_242294


namespace sandy_paid_cost_shop2_l242_242981

-- Define the conditions
def books_shop1 : ℕ := 65
def cost_shop1 : ℕ := 1380
def books_shop2 : ℕ := 55
def avg_price_per_book : ℕ := 19

-- Calculation of the total amount Sandy paid for the books from the second shop
def cost_shop2 (total_books: ℕ) (avg_price: ℕ) (cost1: ℕ) : ℕ :=
  (total_books * avg_price) - cost1

-- Define the theorem we want to prove
theorem sandy_paid_cost_shop2 : cost_shop2 (books_shop1 + books_shop2) avg_price_per_book cost_shop1 = 900 :=
sorry

end sandy_paid_cost_shop2_l242_242981


namespace transmit_data_time_l242_242287

def total_chunks (blocks: ℕ) (chunks_per_block: ℕ) : ℕ := blocks * chunks_per_block

def transmit_time (total_chunks: ℕ) (chunks_per_second: ℕ) : ℕ := total_chunks / chunks_per_second

def time_in_minutes (transmit_time_seconds: ℕ) : ℕ := transmit_time_seconds / 60

theorem transmit_data_time :
  ∀ (blocks chunks_per_block chunks_per_second : ℕ),
    blocks = 150 →
    chunks_per_block = 256 →
    chunks_per_second = 200 →
    time_in_minutes (transmit_time (total_chunks blocks chunks_per_block) chunks_per_second) = 3 := by
  intros
  sorry

end transmit_data_time_l242_242287


namespace part1_part2_l242_242316

variable (x : ℝ)
def A : ℝ := 2 * x^2 - 3 * x + 2
def B : ℝ := x^2 - 3 * x - 2

theorem part1 : A x - B x = x^2 + 4 := sorry

theorem part2 (h : x = -2) : A x - B x = 8 := sorry

end part1_part2_l242_242316


namespace semicircle_radius_approx_l242_242272

noncomputable def semicircle_radius (P : ℝ) : ℝ := P / (Real.pi + 2)

theorem semicircle_radius_approx (h : 46.27433388230814 = (Real.pi * r + 2 * r)) : 
  semicircle_radius 46.27433388230814 ≈ 8.998883928 :=
by
  sorry

end semicircle_radius_approx_l242_242272


namespace subproblem1_l242_242328

theorem subproblem1 (a b c q : ℝ) (h1 : c = b * q) (h2 : c = a * q^2) : 
  (Real.sqrt 5 - 1) / 2 < q ∧ q < (Real.sqrt 5 + 1) / 2 := 
sorry

end subproblem1_l242_242328


namespace reality_show_duration_l242_242620

variable (x : ℕ)

theorem reality_show_duration :
  (5 * x + 10 = 150) → (x = 28) :=
by
  intro h
  sorry

end reality_show_duration_l242_242620


namespace problem_condition_implies_statement_l242_242614

variable {a b c : ℝ}

theorem problem_condition_implies_statement :
  a^3 + a * b + a * c < 0 → b^5 - 4 * a * c > 0 :=
by
  intros h
  sorry

end problem_condition_implies_statement_l242_242614


namespace geometric_seq_arith_mean_l242_242747

theorem geometric_seq_arith_mean 
  (b : ℕ → ℝ) 
  (r : ℝ) 
  (b_geom : ∀ n, b (n + 1) = r * b n)
  (h_arith_mean : b 9 = (3 + 5) / 2) :
  b 1 * b 17 = 16 :=
by
  sorry

end geometric_seq_arith_mean_l242_242747


namespace battery_current_l242_242114

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l242_242114


namespace maximum_value_of_m_l242_242575

theorem maximum_value_of_m (x y : ℝ) (hx : x > 1 / 2) (hy : y > 1) : 
    (4 * x^2 / (y - 1) + y^2 / (2 * x - 1)) ≥ 8 := 
sorry

end maximum_value_of_m_l242_242575


namespace tangent_points_l242_242492

noncomputable def curve (x : ℝ) : ℝ := x^3 - x - 1

theorem tangent_points (x y : ℝ) (h : y = curve x) (slope_line : ℝ) (h_slope : slope_line = -1/2)
  (tangent_perpendicular : (3 * x^2 - 1) = 2) :
  (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) := sorry

end tangent_points_l242_242492


namespace find_x2_y2_l242_242262

variable (x y : ℝ)

-- Given conditions
def average_commute_time (x y : ℝ) := (x + y + 10 + 11 + 9) / 5 = 10
def variance_commute_time (x y : ℝ) := ( (x - 10) ^ 2 + (y - 10) ^ 2 + (10 - 10) ^ 2 + (11 - 10) ^ 2 + (9 - 10) ^ 2 ) / 5 = 2

-- The theorem to prove
theorem find_x2_y2 (hx_avg : average_commute_time x y) (hx_var : variance_commute_time x y) : 
  x^2 + y^2 = 208 :=
sorry

end find_x2_y2_l242_242262


namespace maximum_sum_of_distances_l242_242197

open Set
open Function
open EuclideanGeometry

variable {P : Type*} [MetricSpace P] [NormedGroup P] [NormedSpace ℝ P] [EuclideanSpace ℝ P]

theorem maximum_sum_of_distances (A B C M : P) (l : AffineSubspace ℝ P) 
  (hM : midpoint ℝ B C M) (hAM : convex ℝ ({A, M} : Set P)) 
  (h_perp : ⊥ (l.direction ⊓ (lineThrough ℝ A M).direction)) :
  ∀ l', (l'.direction ⊓ (lineThrough ℝ A M).direction = ⊥) → (sum_distances_to_line A B C l ≤ sum_distances_to_line A B C l') :=
sorry

end maximum_sum_of_distances_l242_242197


namespace balls_per_pack_l242_242783

theorem balls_per_pack (total_packs total_cost cost_per_ball total_balls balls_per_pack : ℕ)
  (h1 : total_packs = 4)
  (h2 : total_cost = 24)
  (h3 : cost_per_ball = 2)
  (h4 : total_balls = total_cost / cost_per_ball)
  (h5 : total_balls = 12)
  (h6 : balls_per_pack = total_balls / total_packs) :
  balls_per_pack = 3 := by 
  sorry

end balls_per_pack_l242_242783


namespace value_of_Y_l242_242598

/- Define the conditions given in the problem -/
def first_row_arithmetic_seq (a1 d1 : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d1
def fourth_row_arithmetic_seq (a4 d4 : ℕ) (n : ℕ) : ℕ := a4 + (n - 1) * d4

/- Constants given by the problem -/
def a1 : ℕ := 3
def fourth_term_first_row : ℕ := 27
def a4 : ℕ := 6
def fourth_term_fourth_row : ℕ := 66

/- Calculating common differences for first and fourth rows -/
def d1 : ℕ := (fourth_term_first_row - a1) / 3
def d4 : ℕ := (fourth_term_fourth_row - a4) / 3

/- Note that we are given that Y is at position (2, 2)
   Express Y in definition forms -/
def Y_row := first_row_arithmetic_seq (a1 + d1) d4 2
def Y_column := fourth_row_arithmetic_seq (a4 + d4) d1 2

/- Problem statement in Lean 4 -/
theorem value_of_Y : Y_row = 35 ∧ Y_column = 35 := by
  sorry

end value_of_Y_l242_242598


namespace difference_between_largest_and_smallest_l242_242247

def largest_number := 9765310
def smallest_number := 1035679
def expected_difference := 8729631
def digits := [3, 9, 6, 0, 5, 1, 7]

theorem difference_between_largest_and_smallest :
  (largest_number - smallest_number) = expected_difference :=
sorry

end difference_between_largest_and_smallest_l242_242247


namespace largest_angle_in_triangle_l242_242364

theorem largest_angle_in_triangle (y : ℝ) (h : 60 + 70 + y = 180) :
    70 > 60 ∧ 70 > y :=
by {
  sorry
}

end largest_angle_in_triangle_l242_242364


namespace mean_score_for_exam_l242_242729

variable (M SD : ℝ)

-- Define the conditions
def condition1 : Prop := 58 = M - 2 * SD
def condition2 : Prop := 98 = M + 3 * SD

-- The problem statement
theorem mean_score_for_exam (h1 : condition1 M SD) (h2 : condition2 M SD) : M = 74 :=
sorry

end mean_score_for_exam_l242_242729


namespace path_count_from_E_to_G_passing_through_F_l242_242956

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem path_count_from_E_to_G_passing_through_F :
  let E := (0, 0)
  let F := (5, 2)
  let G := (6, 5)
  ∃ (paths_EF paths_FG total_paths : ℕ),
  paths_EF = binom (5 + 2) 5 ∧
  paths_FG = binom (1 + 3) 1 ∧
  total_paths = paths_EF * paths_FG ∧
  total_paths = 84 := 
by
  sorry

end path_count_from_E_to_G_passing_through_F_l242_242956


namespace battery_current_l242_242024

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l242_242024


namespace conditional_probability_l242_242643

variable {Ω : Type*} {P : MeasureTheory.ProbabilityMeasure Ω}
variable (A B : MeasureTheory.MeasurableSet Ω)

-- Given conditions
axiom prob_A : P A = 0.8
axiom prob_B : P B = 0.4
axiom prob_A_and_B : P (A ∩ B) = 0.4

-- To prove the conditional probability
theorem conditional_probability : MeasureTheory.condProb P B A = 0.5 :=
by
  have h : P (A ∩ B) / P A = 0.5,
  from sorry,
  exact MeasureTheory.condProb_eq h

end conditional_probability_l242_242643


namespace jane_uses_40_ribbons_l242_242601

theorem jane_uses_40_ribbons :
  (∀ dresses1 dresses2 ribbons_per_dress, 
  dresses1 = 2 * 7 ∧ 
  dresses2 = 3 * 2 → 
  ribbons_per_dress = 2 → 
  (dresses1 + dresses2) * ribbons_per_dress = 40)
:= 
by 
  sorry

end jane_uses_40_ribbons_l242_242601


namespace find_g_3_l242_242443

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g_3 (h : ∀ x : ℝ, g (2 * x - 5) = 3 * x + 9) : g 3 = 21 :=
by
  sorry

end find_g_3_l242_242443


namespace ellipse_foci_coordinates_l242_242631

theorem ellipse_foci_coordinates :
  ∀ x y : ℝ,
  25 * x^2 + 16 * y^2 = 1 →
  (x, y) = (0, 3/20) ∨ (x, y) = (0, -3/20) :=
by
  intro x y h
  sorry

end ellipse_foci_coordinates_l242_242631


namespace ratio_of_pieces_l242_242525

-- Definitions from the conditions
def total_length : ℝ := 28
def shorter_piece_length : ℝ := 8.000028571387755

-- Derived definition
def longer_piece_length : ℝ := total_length - shorter_piece_length

-- Statement to prove the ratio
theorem ratio_of_pieces : 
  (shorter_piece_length / longer_piece_length) = 0.400000571428571 :=
by
  -- Use sorry to skip the proof
  sorry

end ratio_of_pieces_l242_242525


namespace quadratic_range_l242_242731

noncomputable def quadratic_condition (a m : ℝ) : Prop :=
  (a > 0) ∧ (a ≠ 1) ∧ (- (1 + 1 / m) > 0) ∧
  (3 * m^2 - 2 * m - 1 ≤ 0)

theorem quadratic_range (a m : ℝ) :
  quadratic_condition a m → - (1 / 3) ≤ m ∧ m < 0 :=
by sorry

end quadratic_range_l242_242731


namespace brian_final_cards_l242_242132

-- Definitions of initial conditions
def initial_cards : ℕ := 76
def cards_taken : ℕ := 59
def packs_bought : ℕ := 3
def cards_per_pack : ℕ := 15

-- The proof problem: Prove that the final number of cards is 62
theorem brian_final_cards : initial_cards - cards_taken + packs_bought * cards_per_pack = 62 :=
by
  -- Proof goes here, 'sorry' used to skip actual proof
  sorry

end brian_final_cards_l242_242132


namespace rabbits_clear_land_in_21_days_l242_242515

theorem rabbits_clear_land_in_21_days (length_feet width_feet : ℝ) (rabbits : ℕ) (clear_per_rabbit_per_day : ℝ) : 
  length_feet = 900 → width_feet = 200 → rabbits = 100 → clear_per_rabbit_per_day = 10 →
  (⌈ (length_feet / 3 * width_feet / 3) / (rabbits * clear_per_rabbit_per_day) ⌉ = 21) := 
by
  intros
  sorry

end rabbits_clear_land_in_21_days_l242_242515


namespace intersection_condition_sufficient_but_not_necessary_l242_242159

theorem intersection_condition_sufficient_but_not_necessary (k : ℝ) :
  (-Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3) →
  ((∃ x : ℝ, (k^2 + 1) * x^2 + (2 * k^2 - 2) * x + k^2 = 0) ∧ 
   ¬ (∃ k, (∃ x : ℝ, (k^2 + 1) * x^2 + (2 * k^2 - 2) * x + k^2 = 0) → 
   (¬ (-Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3)))) :=
sorry

end intersection_condition_sufficient_but_not_necessary_l242_242159


namespace battery_current_at_given_resistance_l242_242094

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l242_242094


namespace probability_ln_ineq_l242_242374

noncomputable def is_in_interval (a : ℝ) : Prop := 0 < a ∧ a < 1

noncomputable def ln_ineq (a : ℝ) : Prop := log (3 * a - 1) < 0

theorem probability_ln_ineq :
  ∀ (μ : measure_theory.measure ℝ),
    (∀ (a : ℝ), 0 < a ∧ a < 1 → μ {a} = 1) →
    (μ (set.Ioc (1/3 : ℝ) (2/3)) = 1/3) →
      μ {a | 0 < a ∧ a < 1 ∧ log (3 * a - 1) < 0} = 1/3 :=
begin
  intros μ hμ1 hμ2,
  -- Proof steps would go here
  sorry

end probability_ln_ineq_l242_242374


namespace binom_30_3_l242_242706

theorem binom_30_3 : Nat.choose 30 3 = 4060 := 
by 
  sorry

end binom_30_3_l242_242706


namespace time_to_cover_escalator_l242_242249

theorem time_to_cover_escalator (escalator_speed person_speed length : ℕ) (h1 : escalator_speed = 11) (h2 : person_speed = 3) (h3 : length = 126) : 
  length / (escalator_speed + person_speed) = 9 := by
  sorry

end time_to_cover_escalator_l242_242249


namespace circle_area_l242_242490

-- Let r be the radius of the circle
-- The circumference of the circle is given by 2 * π * r, which is 36 cm
-- We need to prove that given this condition, the area of the circle is 324/π square centimeters

theorem circle_area (r : Real) (h : 2 * Real.pi * r = 36) : Real.pi * r^2 = 324 / Real.pi :=
by
  sorry

end circle_area_l242_242490


namespace find_vertex_C_l242_242361

def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 4)
def euler_line (x y : ℝ) : Prop := x - y + 2 = 0

theorem find_vertex_C 
  (C : ℝ × ℝ)
  (h_centroid : (2 + C.1) / 3 = (4 + C.2) / 3)
  (h_euler_line : euler_line ((2 + C.1) / 3) ((4 + C.2) / 3))
  (h_circumcenter : (C.1 + 1)^2 + (C.2 - 1)^2 = 10) :
  C = (-4, 0) :=
sorry

end find_vertex_C_l242_242361


namespace no_two_birch_adjacent_l242_242395

noncomputable def calculate_probability_no_birch_adjacent : ℚ := 
  let total_ways := fact 12
  let non_birch_ways := fact 7
  let birch_spaces := (finset.range 8).choose 5
  let birch_permutations := fact 5
  ((non_birch_ways * (birch_spaces * birch_permutations)) : ℚ) / total_ways

theorem no_two_birch_adjacent (m n : ℕ) (h : no_two_birch_adjacent_probability = m / n) :
  m + n = 106 :=
by
  -- provided that the probability is simplified to 7 / 99
  have : no_two_birch_adjacent_probability = 7 / 99 := sorry
  sorry

end no_two_birch_adjacent_l242_242395


namespace integer_remainder_18_l242_242278

theorem integer_remainder_18 (n : ℤ) (h : n ∈ ({14, 15, 16, 17, 18} : Set ℤ)) : n % 7 = 4 :=
by
  sorry

end integer_remainder_18_l242_242278


namespace expected_win_l242_242864

-- Definitions of conditions
def sides := fin 8 -- Finite type representing the 8 sides of the die

-- Function to calculate the win amount given a roll
def win_amount (n : sides) : ℝ := 8 - n.val

-- Probability of each side for a fair die
def probability : ℝ := 1 / 8

-- Definition of expected value calculation
def expected_value : ℝ := ∑ n in (finset.univ : finset sides), probability * (win_amount n)

-- Theorem statement
theorem expected_win : expected_value = 3.5 :=
by sorry

end expected_win_l242_242864


namespace solution_inequality_l242_242937

open Set

theorem solution_inequality (x : ℝ) : (x > 3 ∨ x < -3) ↔ (x > 9 / x) := by
  sorry

end solution_inequality_l242_242937


namespace current_value_l242_242091

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l242_242091


namespace battery_current_l242_242085

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l242_242085


namespace g_at_2_l242_242633

-- Assuming g is a function from ℝ to ℝ such that it satisfies the given condition.
def g : ℝ → ℝ := sorry

-- Condition of the problem
axiom g_condition : ∀ x : ℝ, g (2 ^ x) + x * g (2 ^ (-x)) = 2

-- The statement we want to prove
theorem g_at_2 : g (2) = 0 :=
by
  sorry

end g_at_2_l242_242633


namespace bridge_length_is_correct_l242_242820

-- Train length in meters
def train_length : ℕ := 130

-- Train speed in km/hr
def train_speed_kmh : ℕ := 45

-- Time to cross bridge in seconds
def time_to_cross_bridge : ℕ := 30

-- Conversion factor from km/hr to m/s
def kmh_to_mps (kmh : ℕ) : ℚ := (kmh * 1000) / 3600

-- Train speed in m/s
def train_speed_mps := kmh_to_mps train_speed_kmh

-- Total distance covered by the train in 30 seconds
def total_distance := train_speed_mps * time_to_cross_bridge

-- Length of the bridge
def bridge_length := total_distance - train_length

theorem bridge_length_is_correct : bridge_length = 245 := by
  sorry

end bridge_length_is_correct_l242_242820


namespace expression_evaluation_l242_242562

noncomputable def x : ℝ := (Real.sqrt 1.21) ^ 3
noncomputable def y : ℝ := (Real.sqrt 0.81) ^ 2
noncomputable def a : ℝ := 4 * Real.sqrt 0.81
noncomputable def b : ℝ := 2 * Real.sqrt 0.49
noncomputable def c : ℝ := 3 * Real.sqrt 1.21
noncomputable def d : ℝ := 2 * Real.sqrt 0.49
noncomputable def e : ℝ := (Real.sqrt 0.81) ^ 4

theorem expression_evaluation : ((x / Real.sqrt y) - (Real.sqrt a / b^2) + ((Real.sqrt c / Real.sqrt d) / (3 * e))) = 1.291343 := by 
  sorry

end expression_evaluation_l242_242562


namespace binom_30_3_l242_242705

theorem binom_30_3 : Nat.choose 30 3 = 4060 := 
by 
  sorry

end binom_30_3_l242_242705


namespace rectangle_length_l242_242447

theorem rectangle_length (side_length_square : ℝ) (width_rectangle : ℝ) (area_equal : ℝ) 
  (square_area : side_length_square * side_length_square = area_equal) 
  (rectangle_area : width_rectangle * (width_rectangle * length) = area_equal) : 
  length = 24 :=
by 
  sorry

end rectangle_length_l242_242447


namespace current_value_l242_242070

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l242_242070


namespace solve_equation_l242_242789

theorem solve_equation :
  ∀ x : ℝ, 4 * x * (6 * x - 1) = 1 - 6 * x ↔ (x = 1/6 ∨ x = -1/4) := 
by
  sorry

end solve_equation_l242_242789


namespace adjacent_side_length_l242_242122

-- Given the conditions
variables (a b : ℝ)
-- Area of the rectangular flower bed
def area := 6 * a * b - 2 * b
-- One side of the rectangular flower bed
def side1 := 2 * b

-- Prove the length of the adjacent side
theorem adjacent_side_length : 
  (6 * a * b - 2 * b) / (2 * b) = 3 * a - 1 :=
by sorry

end adjacent_side_length_l242_242122


namespace planted_fraction_correct_l242_242557

noncomputable def field_planted_fraction (leg1 leg2 : ℕ) (square_distance : ℕ) : ℚ :=
  let hypotenuse := Real.sqrt (leg1^2 + leg2^2)
  let total_area := (leg1 * leg2) / 2
  let square_side := square_distance
  let square_area := square_side^2
  let planted_area := total_area - square_area
  planted_area / total_area

theorem planted_fraction_correct :
  field_planted_fraction 5 12 4 = 367 / 375 :=
by
  sorry

end planted_fraction_correct_l242_242557


namespace certain_number_equation_l242_242829

theorem certain_number_equation (x : ℤ) (h : 16 * x + 17 * x + 20 * x + 11 = 170) : x = 3 :=
by {
  sorry
}

end certain_number_equation_l242_242829


namespace BANANA_distinct_arrangements_l242_242322

theorem BANANA_distinct_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 1) * (Nat.factorial 3) * (Nat.factorial 2)) = 60 := 
by
  sorry

end BANANA_distinct_arrangements_l242_242322


namespace hyperbola_eccentricity_l242_242309

theorem hyperbola_eccentricity (m : ℝ) (h : m > 0) 
(hyperbola_eq : ∀ (x y : ℝ), x^2 / 9 - y^2 / m = 1) 
(eccentricity : ∀ (e : ℝ), e = 2) 
: m = 27 :=
sorry

end hyperbola_eccentricity_l242_242309


namespace remaining_customers_is_13_l242_242878

-- Given conditions
def initial_customers : ℕ := 36
def half_left_customers : ℕ := initial_customers / 2  -- 50% of customers leaving
def remaining_customers_after_half_left : ℕ := initial_customers - half_left_customers

def thirty_percent_of_remaining : ℚ := remaining_customers_after_half_left * 0.30 
def thirty_percent_of_remaining_rounded : ℕ := thirty_percent_of_remaining.floor.toNat  -- rounding down

def final_remaining_customers : ℕ := remaining_customers_after_half_left - thirty_percent_of_remaining_rounded

-- Proof statement without proof
theorem remaining_customers_is_13 : final_remaining_customers = 13 := by
  sorry

end remaining_customers_is_13_l242_242878


namespace total_vessels_l242_242667

open Nat

theorem total_vessels (x y z w : ℕ) (hx : x > 0) (hy : y > x) (hz : z > y) (hw : w > z) :
  ∃ total : ℕ, total = x * (2 * y + 1) + z * (1 + 1 / w) := sorry

end total_vessels_l242_242667


namespace find_chemistry_marks_l242_242550

theorem find_chemistry_marks
  (english_marks : ℕ) (math_marks : ℕ) (physics_marks : ℕ) (biology_marks : ℕ) (average_marks : ℕ) (chemistry_marks : ℕ) :
  english_marks = 86 → math_marks = 89 → physics_marks = 82 → biology_marks = 81 → average_marks = 85 →
  chemistry_marks = 425 - (english_marks + math_marks + physics_marks + biology_marks) →
  chemistry_marks = 87 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h6
  have total_marks := 425 - (86 + 89 + 82 + 81)
  norm_num at total_marks
  exact h6

end find_chemistry_marks_l242_242550


namespace probability_correct_l242_242526

-- Define the conditions of the problem
def total_white_balls : ℕ := 6
def total_black_balls : ℕ := 5
def total_balls : ℕ := total_white_balls + total_black_balls
def total_ways_draw_two_balls : ℕ := Nat.choose total_balls 2
def ways_choose_one_white_ball : ℕ := Nat.choose total_white_balls 1
def ways_choose_one_black_ball : ℕ := Nat.choose total_black_balls 1
def total_successful_outcomes : ℕ := ways_choose_one_white_ball * ways_choose_one_black_ball

-- Define the probability calculation
def probability_drawing_one_white_one_black : ℚ := total_successful_outcomes / total_ways_draw_two_balls

-- State the theorem
theorem probability_correct :
  probability_drawing_one_white_one_black = 6 / 11 :=
by
  sorry

end probability_correct_l242_242526


namespace carriage_and_people_l242_242765

variable {x y : ℕ}

theorem carriage_and_people :
  (3 * (x - 2) = y) ∧ (2 * x + 9 = y) :=
sorry

end carriage_and_people_l242_242765


namespace current_value_l242_242067

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l242_242067


namespace trade_in_value_of_old_phone_l242_242548

-- Define the given conditions
def cost_of_iphone : ℕ := 800
def earnings_per_week : ℕ := 80
def weeks_worked : ℕ := 7

-- Define the total earnings from babysitting
def total_earnings : ℕ := earnings_per_week * weeks_worked

-- Define the final proof statement
theorem trade_in_value_of_old_phone : cost_of_iphone - total_earnings = 240 :=
by
  unfold cost_of_iphone
  unfold total_earnings
  -- Substitute in the values
  have h1 : 800 - (80 * 7) = 240 := sorry
  exact h1

end trade_in_value_of_old_phone_l242_242548


namespace current_when_resistance_is_12_l242_242029

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l242_242029


namespace binom_30_3_eq_4060_l242_242693

theorem binom_30_3_eq_4060 : nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l242_242693


namespace family_travel_time_l242_242672

theorem family_travel_time (D : ℕ) (v1 v2 : ℕ) (d1 d2 : ℕ) (t1 t2 : ℕ) :
  D = 560 → 
  v1 = 35 → 
  v2 = 40 → 
  d1 = D / 2 →
  d2 = D / 2 →
  t1 = d1 / v1 →
  t2 = d2 / v2 → 
  t1 + t2 = 15 :=
by
  sorry

end family_travel_time_l242_242672


namespace smallest_positive_integer_k_l242_242873

theorem smallest_positive_integer_k:
  ∀ T : ℕ, ∀ n : ℕ, (T = n * (n + 1) / 2) → ∃ m : ℕ, 81 * T + 10 = m * (m + 1) / 2 :=
by
  intro T n h
  sorry

end smallest_positive_integer_k_l242_242873


namespace isosceles_triangles_l242_242778

noncomputable def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem isosceles_triangles (a b c : ℝ) (h : a ≥ b ∧ b ≥ c ∧ c > 0)
    (H : ∀ n : ℕ, a ^ n + b ^ n > c ^ n ∧ b ^ n + c ^ n > a ^ n ∧ c ^ n + a ^ n > b ^ n) :
    is_isosceles_triangle a b c :=
  sorry

end isosceles_triangles_l242_242778


namespace whitewashing_cost_l242_242214

noncomputable def cost_of_whitewashing (l w h : ℝ) (c : ℝ) (door_area window_area : ℝ) (num_windows : ℝ) : ℝ :=
  let perimeter := 2 * (l + w)
  let total_wall_area := perimeter * h
  let total_window_area := num_windows * window_area
  let total_paintable_area := total_wall_area - (door_area + total_window_area)
  total_paintable_area * c

theorem whitewashing_cost:
  cost_of_whitewashing 25 15 12 6 (6 * 3) (4 * 3) 3 = 5436 := by
  sorry

end whitewashing_cost_l242_242214


namespace calculate_total_people_l242_242545

-- Definitions given in the problem
def cost_per_adult_meal := 3
def num_kids := 7
def total_cost := 15

-- The target property to prove
theorem calculate_total_people : 
  (total_cost / cost_per_adult_meal) + num_kids = 12 := 
by 
  sorry

end calculate_total_people_l242_242545


namespace range_of_k_l242_242168

open Real

noncomputable def h (x : ℝ) : ℝ := -2 * log x / x

theorem range_of_k :
  let f : ℝ → ℝ := λ x, k * x,
      g : ℝ → ℝ := λ x, (1 / exp 1) ^ (x / 2) in
  (∃ x₁ x₂, (1 / exp 1) ≤ x₁ ∧ x₁ ≤ exp 2 ∧ (1 / exp 1) ≤ x₂ ∧ x₂ ≤ exp 2
   ∧ f x₁ = y ∧ g x₂ = y ∧ y = y ∧ x₁ + x₂ = 1) -> 
   λ k, -2 / (exp 1 : ℝ) ≤ k ∧ k ≤ 2 * exp 1 :=
begin
  sorry
end

end range_of_k_l242_242168


namespace inverse_proportion_symmetry_l242_242150

theorem inverse_proportion_symmetry (a b : ℝ) :
  (b = - 6 / (-a)) → (-b = - 6 / a) :=
by
  intro h
  sorry

end inverse_proportion_symmetry_l242_242150


namespace total_pencils_l242_242410

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (hp : pencils_per_child = 2) (hc : children = 8) :
  pencils_per_child * children = 16 :=
by
  sorry

end total_pencils_l242_242410


namespace probability_all_quitters_same_tribe_l242_242366

-- Definitions of the problem conditions
def total_contestants : ℕ := 20
def tribe_size : ℕ := 10
def quitters : ℕ := 3

-- Definition of the binomial coefficient
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem probability_all_quitters_same_tribe :
  (choose tribe_size quitters + choose tribe_size quitters) * 
  (total_contestants.choose quitters) = 240 
  ∧ ((choose tribe_size quitters + choose tribe_size quitters) / (total_contestants.choose quitters)) = 20 / 95 :=
by
  sorry

end probability_all_quitters_same_tribe_l242_242366


namespace problem_proof_l242_242630

noncomputable def arithmetic_sequences (a b : ℕ → ℤ) (S T : ℕ → ℤ) :=
  ∀ n, S n = (n * (2 * a 0 + (n - 1) * (a 1 - a 0))) / 2 ∧
         T n = (n * (2 * b 0 + (n - 1) * (b 1 - b 0))) / 2

theorem problem_proof 
  (a b : ℕ → ℤ) 
  (S T : ℕ → ℤ)
  (h_seq : arithmetic_sequences a b S T)
  (h_relation : ∀ n, S n / T n = (7 * n : ℤ) / (n + 3)) :
  (a 5) / (b 5) = 21 / 4 :=
by 
  sorry

end problem_proof_l242_242630


namespace current_value_l242_242063

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l242_242063


namespace num_8tuples_satisfying_condition_l242_242295

theorem num_8tuples_satisfying_condition :
  (∃! (y : Fin 8 → ℝ),
    (2 - y 0)^2 + (y 0 - y 1)^2 + (y 1 - y 2)^2 + 
    (y 2 - y 3)^2 + (y 3 - y 4)^2 + (y 4 - y 5)^2 + 
    (y 5 - y 6)^2 + (y 6 - y 7)^2 + y 7^2 = 4 / 9) :=
sorry

end num_8tuples_satisfying_condition_l242_242295


namespace lawn_chair_original_price_l242_242870

theorem lawn_chair_original_price (sale_price : ℝ) (discount_percentage : ℝ) (original_price : ℝ) :
  sale_price = 59.95 →
  discount_percentage = 23.09 →
  original_price = sale_price / (1 - discount_percentage / 100) →
  original_price = 77.95 :=
by sorry

end lawn_chair_original_price_l242_242870


namespace find_current_when_resistance_is_12_l242_242039

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l242_242039


namespace name_tag_area_l242_242990

-- Define the side length of the square
def side_length : ℕ := 11

-- Define the area calculation for a square
def square_area (side : ℕ) : ℕ := side * side

-- State the theorem: the area of a square with side length of 11 cm is 121 cm²
theorem name_tag_area : square_area side_length = 121 :=
by
  sorry

end name_tag_area_l242_242990


namespace remainder_of_power_mod_l242_242415

theorem remainder_of_power_mod :
  (5^2023) % 11 = 4 :=
  by
    sorry

end remainder_of_power_mod_l242_242415


namespace total_insects_eaten_l242_242825

theorem total_insects_eaten : 
  (5 * 6) + (3 * (2 * 6)) = 66 :=
by
  /- We'll calculate the total number of insects eaten by combining the amounts eaten by the geckos and lizards -/
  sorry

end total_insects_eaten_l242_242825


namespace factorize_polynomial_l242_242293

theorem factorize_polynomial {x : ℝ} : x^3 + 2 * x^2 - 3 * x = x * (x + 3) * (x - 1) :=
by sorry

end factorize_polynomial_l242_242293


namespace geometric_sequence_sum_inequality_l242_242945

open Classical

variable (a_1 q : ℝ) (h1 : a_1 > 0) (h2 : q > 0) (h3 : q ≠ 1)

theorem geometric_sequence_sum_inequality :
  a_1 + a_1 * q^3 > a_1 * q + a_1 * q^2 :=
by
  sorry

end geometric_sequence_sum_inequality_l242_242945


namespace total_revenue_l242_242209

def price_federal := 50
def price_state := 30
def price_quarterly := 80
def num_federal := 60
def num_state := 20
def num_quarterly := 10

theorem total_revenue : (num_federal * price_federal + num_state * price_state + num_quarterly * price_quarterly) = 4400 := by
  sorry

end total_revenue_l242_242209


namespace pen_cost_is_2_25_l242_242261

variables (p i : ℝ)

def total_cost (p i : ℝ) : Prop := p + i = 2.50
def pen_more_expensive (p i : ℝ) : Prop := p = 2 + i

theorem pen_cost_is_2_25 (p i : ℝ) 
  (h1 : total_cost p i) 
  (h2 : pen_more_expensive p i) : 
  p = 2.25 := 
by
  sorry

end pen_cost_is_2_25_l242_242261


namespace max_value_of_x_sq_plus_y_sq_l242_242749

theorem max_value_of_x_sq_plus_y_sq (x y : ℝ) 
  (h : x^2 + y^2 + 4 * x - 2 * y - 4 = 0) : 
  ∃ M, M = 14 + 6 * Real.sqrt 5 ∧ ∀ (x y : ℝ), x^2 + y^2 + 4 * x - 2 * y - 4 = 0 → x^2 + y^2 ≤ M :=
sorry

end max_value_of_x_sq_plus_y_sq_l242_242749


namespace find_d_value_l242_242943

theorem find_d_value 
  (x y d : ℝ)
  (h1 : 7^(3 * x - 1) * 3^(4 * y - 3) = 49^x * d^y)
  (h2 : x + y = 4) :
  d = 27 :=
by 
  sorry

end find_d_value_l242_242943


namespace current_value_l242_242092

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l242_242092


namespace exists_square_no_visible_points_l242_242565

-- Define visibility from the origin
def visible_from_origin (x y : ℤ) : Prop :=
  Int.gcd x y = 1

-- Main theorem statement
theorem exists_square_no_visible_points (n : ℕ) (hn : 0 < n) :
  ∃ (a b : ℤ), 
    (∀ (x y : ℤ), a ≤ x ∧ x ≤ a + n ∧ b ≤ y ∧ y ≤ b + n ∧ (x ≠ 0 ∨ y ≠ 0) → ¬visible_from_origin x y) :=
sorry

end exists_square_no_visible_points_l242_242565


namespace range_of_f_l242_242750

noncomputable def f (x : Real) : Real :=
  if x ≤ 1 then 2 * x + 1 else Real.log x + 1

theorem range_of_f (x : Real) : f x + f (x + 1) > 1 ↔ (x > -(3 / 4)) :=
  sorry

end range_of_f_l242_242750


namespace expected_value_of_8_sided_die_l242_242842

-- Define the expected value function for the given win calculation rule.
def expected_value := (1/8 : ℚ) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

-- Formal statement of the proof problem.
theorem expected_value_of_8_sided_die : 
  expected_value = 3.50 :=
by
  sorry

end expected_value_of_8_sided_die_l242_242842


namespace find_current_when_resistance_is_12_l242_242041

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l242_242041


namespace derivative_at_0_l242_242493

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x * Real.sin x - 7 * x

theorem derivative_at_0 : deriv f 0 = -6 := 
by
  sorry

end derivative_at_0_l242_242493


namespace terminal_side_in_first_quadrant_l242_242588

noncomputable def theta := -5

def in_first_quadrant (θ : ℝ) : Prop :=
  by sorry

theorem terminal_side_in_first_quadrant : in_first_quadrant theta := 
  by sorry

end terminal_side_in_first_quadrant_l242_242588


namespace positive_number_percent_l242_242397

theorem positive_number_percent (x : ℝ) (h : 0.01 * x^2 = 9) (hx : 0 < x) : x = 30 :=
sorry

end positive_number_percent_l242_242397


namespace interest_rate_and_years_l242_242679

theorem interest_rate_and_years
    (P : ℝ)
    (n : ℕ)
    (e : ℝ)
    (h1 : P * (e ^ n) * e = P * (e ^ (n + 1)) + 4156.02)
    (h2 : P * (e ^ (n - 1)) = P * (e ^ n) - 3996.12) :
    (e = 1.04) ∧ (P = 60000) ∧ (E = 4/100) ∧ (n = 14) := by
  sorry

end interest_rate_and_years_l242_242679


namespace intersection_of_M_and_N_l242_242313

-- Definitions of the sets
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x : ℤ | -2 < x ∧ x < 2}

-- The theorem to prove
theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_of_M_and_N_l242_242313


namespace inverse_proportion_symmetric_l242_242153

theorem inverse_proportion_symmetric (a b : ℝ) (h : a ≠ 0) (h_ab : b = -6 / -a) : (-b) = -6 / a :=
by
  -- the proof goes here
  sorry

end inverse_proportion_symmetric_l242_242153


namespace bigger_part_of_sum_54_l242_242524

theorem bigger_part_of_sum_54 (x y : ℕ) (h₁ : x + y = 54) (h₂ : 10 * x + 22 * y = 780) : x = 34 :=
sorry

end bigger_part_of_sum_54_l242_242524


namespace nth_equation_pattern_l242_242349

theorem nth_equation_pattern (n : ℕ) : (n + 1) * (n^2 - n + 1) - 1 = n^3 :=
by
  sorry

end nth_equation_pattern_l242_242349


namespace terrell_lifting_l242_242207

theorem terrell_lifting :
  (3 * 25 * 10 = 3 * 20 * 12.5) :=
by
  sorry

end terrell_lifting_l242_242207


namespace a_parallel_b_l242_242301

variable {Line : Type} (a b c : Line)

-- Definition of parallel lines
def parallel (x y : Line) : Prop := sorry

-- Conditions
axiom a_parallel_c : parallel a c
axiom b_parallel_c : parallel b c

-- Theorem to prove a is parallel to b given the conditions
theorem a_parallel_b : parallel a b :=
by
  sorry

end a_parallel_b_l242_242301


namespace expected_value_of_8_sided_die_l242_242858

open ProbabilityTheory

-- Definitions based on conditions

-- Define the 8-sided die outcomes
def outcomes := ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ)

-- Define the probability of each outcome
def probability (n : ℕ) : ℝ := if n ∈ outcomes then (1 / 8) else 0

-- Define the payout function based on the roll outcome
def payout (n : ℕ) : ℝ := if n ∈ outcomes then 8 - n else 0

-- Define the expected value calculation
def expected_value : ℝ :=
  ∑ n in outcomes, (probability n) * (payout n)

-- Main theorem to prove the expected value is 3.5 dollars
theorem expected_value_of_8_sided_die : expected_value = 3.5 :=
by
  -- Expected value calculation skipped; to be completed
  sorry

end expected_value_of_8_sided_die_l242_242858


namespace least_common_multiple_135_195_l242_242377

def leastCommonMultiple (a b : ℕ) : ℕ :=
  Nat.lcm a b

theorem least_common_multiple_135_195 : leastCommonMultiple 135 195 = 1755 := by
  sorry

end least_common_multiple_135_195_l242_242377


namespace pizza_toppings_l242_242266

theorem pizza_toppings (toppings : Finset String) (h : toppings.card = 8) :
  (toppings.card.choose 1 + toppings.card.choose 2 + toppings.card.choose 3) = 92 := by
  have ht : toppings.card = 8 := h
  sorry

end pizza_toppings_l242_242266


namespace radio_cost_price_l242_242358

theorem radio_cost_price (SP : ℝ) (Loss : ℝ) (CP : ℝ) (h1 : SP = 1110) (h2 : Loss = 0.26) (h3 : SP = CP * (1 - Loss)) : CP = 1500 :=
  by
  sorry

end radio_cost_price_l242_242358


namespace octal_to_decimal_l242_242549

theorem octal_to_decimal (n_octal : ℕ) (h : n_octal = 123) : 
  let d0 := 3 * 8^0
  let d1 := 2 * 8^1
  let d2 := 1 * 8^2
  n_octal = 64 + 16 + 3 :=
by
  sorry

end octal_to_decimal_l242_242549


namespace binomial_60_3_l242_242907

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l242_242907


namespace cats_in_shelter_l242_242663

theorem cats_in_shelter (C D: ℕ) (h1 : 15 * D = 7 * C) 
                        (h2 : 15 * (D + 12) = 11 * C) :
    C = 45 := by
  sorry

end cats_in_shelter_l242_242663


namespace initial_shells_l242_242770

theorem initial_shells (x : ℕ) (h : x + 23 = 28) : x = 5 :=
by
  sorry

end initial_shells_l242_242770


namespace original_loaf_had_27_slices_l242_242716

def original_slices : ℕ :=
  let slices_andy_ate := 3 * 2
  let slices_for_toast := 2 * 10
  let slices_left := 1
  slices_andy_ate + slices_for_toast + slices_left

theorem original_loaf_had_27_slices (n : ℕ) (slices_andy_ate : ℕ) (slices_for_toast : ℕ) (slices_left : ℕ) :
  slices_andy_ate = 6 → slices_for_toast = 20 → slices_left = 1 → n = slices_andy_ate + slices_for_toast + slices_left → n = 27 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

-- Verifying the statement
example : original_slices = 27 := by
  have h1 : 3 * 2 = 6 := rfl
  have h2 : 2 * 10 = 20 := rfl
  have h3 : 1 = 1 := rfl
  exact original_loaf_had_27_slices original_slices 6 20 1 h1 h2 h3 rfl

end original_loaf_had_27_slices_l242_242716


namespace complex_equality_l242_242426

theorem complex_equality (a b : ℝ) (i : ℂ) (h : i^2 = -1) (h_eq : a - b * i = (1 + i) * i^3) : a = 1 ∧ b = -1 :=
by sorry

end complex_equality_l242_242426


namespace pow_modulus_l242_242422

theorem pow_modulus : (5 ^ 2023) % 11 = 3 := by
  sorry

end pow_modulus_l242_242422


namespace probability_at_least_one_unqualified_l242_242952

theorem probability_at_least_one_unqualified :
  let total_products := 6
  let qualified_products := 4
  let unqualified_products := 2
  let products_selected := 2
  (1 - (Nat.choose qualified_products 2 / Nat.choose total_products 2)) = 3/5 :=
by
  sorry

end probability_at_least_one_unqualified_l242_242952


namespace solve_for_g2_l242_242636

-- Let g : ℝ → ℝ be a function satisfying the given condition
variable (g : ℝ → ℝ)

-- The given condition
def condition (x : ℝ) : Prop :=
  g (2 ^ x) + x * g (2 ^ (-x)) = 2

-- The main theorem we aim to prove
theorem solve_for_g2 (h : ∀ x, condition g x) : g 2 = 0 :=
by
  sorry

end solve_for_g2_l242_242636


namespace binomial_60_3_eq_34220_l242_242913

open Nat

theorem binomial_60_3_eq_34220 : choose 60 3 = 34220 := 
by
  sorry

end binomial_60_3_eq_34220_l242_242913


namespace mean_score_74_l242_242727

theorem mean_score_74 
  (M SD : ℝ)
  (h1 : 58 = M - 2 * SD)
  (h2 : 98 = M + 3 * SD) : 
  M = 74 :=
by
  sorry

end mean_score_74_l242_242727


namespace expected_value_of_8_sided_die_l242_242856

open ProbabilityTheory

-- Definitions based on conditions

-- Define the 8-sided die outcomes
def outcomes := ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ)

-- Define the probability of each outcome
def probability (n : ℕ) : ℝ := if n ∈ outcomes then (1 / 8) else 0

-- Define the payout function based on the roll outcome
def payout (n : ℕ) : ℝ := if n ∈ outcomes then 8 - n else 0

-- Define the expected value calculation
def expected_value : ℝ :=
  ∑ n in outcomes, (probability n) * (payout n)

-- Main theorem to prove the expected value is 3.5 dollars
theorem expected_value_of_8_sided_die : expected_value = 3.5 :=
by
  -- Expected value calculation skipped; to be completed
  sorry

end expected_value_of_8_sided_die_l242_242856


namespace marsha_remainder_l242_242194

-- Definitions based on problem conditions
def a (n : ℤ) : ℤ := 90 * n + 84
def b (m : ℤ) : ℤ := 120 * m + 114
def c (p : ℤ) : ℤ := 150 * p + 144

-- Proof statement
theorem marsha_remainder (n m p : ℤ) : ((a n + b m + c p) % 30) = 12 :=
by 
  -- Notice we need to add the proof steps here
  sorry 

end marsha_remainder_l242_242194


namespace bisectAltitude_l242_242212

noncomputable def touchCircleParallelogram (A B C D K L M : Point) (P : Circle) : Prop :=
  (P.touches A B K) ∧ (P.touches B C L) ∧ (P.touches C D M) ∧ (Parallelogram A B C D)

theorem bisectAltitude (A B C D K L M : Point) (P : Circle) 
  (cond : touchCircleParallelogram A B C D K L M P) : 
  bisects (Line K L) (Altitude C (Line A B)) :=
sorry

end bisectAltitude_l242_242212


namespace original_loaf_slices_l242_242715

-- Define the given conditions
def andy_slices_1 := 3
def andy_slices_2 := 3
def toast_slices_per_piece := 2
def pieces_of_toast := 10
def slices_left_over := 1

-- Define the variables
def total_andy_slices := andy_slices_1 + andy_slices_2
def total_toast_slices := toast_slices_per_piece * pieces_of_toast

-- State the theorem
theorem original_loaf_slices : 
  ∃ S : ℕ, S = total_andy_slices + total_toast_slices + slices_left_over := 
by {
  sorry
}

end original_loaf_slices_l242_242715


namespace parabola_tangent_sequence_l242_242408

noncomputable def geom_seq_sum (a2 : ℕ) : ℕ :=
  a2 + a2 / 4 + a2 / 16

theorem parabola_tangent_sequence (a2 : ℕ) (h : a2 = 32) : geom_seq_sum a2 = 42 :=
by
  rw [h]
  norm_num
  sorry

end parabola_tangent_sequence_l242_242408


namespace remainder_5_pow_2023_mod_11_l242_242417

theorem remainder_5_pow_2023_mod_11 : (5^2023) % 11 = 4 :=
by
  have h1 : 5^2 % 11 = 25 % 11 := sorry
  have h2 : 25 % 11 = 3 := sorry
  have h3 : (3^5) % 11 = 1 := sorry
  have h4 : 3^1011 % 11 = ((3^5)^202 * 3) % 11 := sorry
  have h5 : ((3^5)^202 * 3) % 11 = (1^202 * 3) % 11 := sorry
  have h6 : (1^202 * 3) % 11 = 3 % 11 := sorry
  have h7 : (5^2023) % 11 = (3 * 5) % 11 := sorry
  have h8 : (3 * 5) % 11 = 15 % 11 := sorry
  have h9 : 15 % 11 = 4 := sorry
  exact h9

end remainder_5_pow_2023_mod_11_l242_242417


namespace gold_bars_lost_l242_242985

-- Define the problem constants
def initial_bars : ℕ := 100
def friends : ℕ := 4
def bars_per_friend : ℕ := 20

-- Define the total distributed gold bars
def total_distributed : ℕ := friends * bars_per_friend

-- Define the number of lost gold bars
def lost_bars : ℕ := initial_bars - total_distributed

-- Theorem: Prove that the number of lost gold bars is 20
theorem gold_bars_lost : lost_bars = 20 := by
  sorry

end gold_bars_lost_l242_242985


namespace determine_a_l242_242139

theorem determine_a (a : ℕ) (p1 p2 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : 2 * p1 * p2 = a) (h4 : p1 + p2 = 15) : 
  a = 52 :=
by
  sorry

end determine_a_l242_242139


namespace value_of_g_neg3_l242_242496

def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem value_of_g_neg3 : g (-3) = 4 :=
by
  sorry

end value_of_g_neg3_l242_242496


namespace proof_problem_l242_242430

variables {R : Type*} [CommRing R]

-- f is a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Variable definitions for the conditions
variables (h_odd : is_odd f)
(h_f1 : f 1 = 1)
(h_period : ∀ x, f (x + 6) = f x + f 3)

-- The proof problem statement
theorem proof_problem : f 2015 + f 2016 = -1 :=
by
  sorry

end proof_problem_l242_242430


namespace evaluate_expression_l242_242738

variable (a : ℝ)

def a_definition : Prop := a = Real.sqrt 11 - 1

theorem evaluate_expression (h : a_definition a) : a^2 + 2*a + 1 = 11 := by
  sorry

end evaluate_expression_l242_242738


namespace minimum_a_l242_242268

noncomputable def func (t a : ℝ) := 5 * (t + 1) ^ 2 + a / (t + 1) ^ 5

theorem minimum_a (a : ℝ) (h: ∀ t ≥ 0, func t a ≥ 24) :
  a = 2 * Real.sqrt ((24 / 7) ^ 7) :=
sorry

end minimum_a_l242_242268


namespace right_triangle_hypotenuse_l242_242874

theorem right_triangle_hypotenuse (a h : ℝ) (r : ℝ) (h1 : r = 8) (h2 : h = a * Real.sqrt 2)
  (h3 : r = (a - h) / 2) : h = 16 * (Real.sqrt 2 + 1) := 
by
  sorry

end right_triangle_hypotenuse_l242_242874


namespace current_at_resistance_12_l242_242062

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l242_242062


namespace battery_current_when_resistance_12_l242_242080

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l242_242080


namespace mean_squared_sum_l242_242987

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry
noncomputable def z : ℝ := sorry

theorem mean_squared_sum :
  (x + y + z = 30) ∧ 
  (xyz = 125) ∧ 
  ((1 / x + 1 / y + 1 / z) = 3 / 4) 
  → x^2 + y^2 + z^2 = 712.5 :=
by
  intros h
  have h₁ : x + y + z = 30 := h.1
  have h₂ : xyz = 125 := h.2.1
  have h₃ : (1 / x + 1 / y + 1 / z) = 3 / 4 := h.2.2
  sorry

end mean_squared_sum_l242_242987


namespace expected_value_of_8_sided_die_l242_242859

open ProbabilityTheory

-- Definitions based on conditions

-- Define the 8-sided die outcomes
def outcomes := ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ)

-- Define the probability of each outcome
def probability (n : ℕ) : ℝ := if n ∈ outcomes then (1 / 8) else 0

-- Define the payout function based on the roll outcome
def payout (n : ℕ) : ℝ := if n ∈ outcomes then 8 - n else 0

-- Define the expected value calculation
def expected_value : ℝ :=
  ∑ n in outcomes, (probability n) * (payout n)

-- Main theorem to prove the expected value is 3.5 dollars
theorem expected_value_of_8_sided_die : expected_value = 3.5 :=
by
  -- Expected value calculation skipped; to be completed
  sorry

end expected_value_of_8_sided_die_l242_242859


namespace factorization_correct_l242_242495

theorem factorization_correct:
  ∃ a b : ℤ, (25 * x^2 - 85 * x - 150 = (5 * x + a) * (5 * x + b)) ∧ (a + 2 * b = -24) :=
by
  sorry

end factorization_correct_l242_242495


namespace product_evaluation_l242_242385

theorem product_evaluation : 
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
  sorry

end product_evaluation_l242_242385


namespace least_xy_value_l242_242157

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : (1/x : ℚ) + 1/(2*y) = 1/8) :
  xy ≥ 128 :=
sorry

end least_xy_value_l242_242157


namespace sequence_a2002_l242_242311

theorem sequence_a2002 :
  ∀ (a : ℕ → ℕ), (a 1 = 1) → (a 2 = 2) → 
  (∀ n, 2 ≤ n → a (n + 1) = 3 * a n - 2 * a (n - 1)) → 
  a 2002 = 2 ^ 2001 :=
by
  intros a ha1 ha2 hrecur
  sorry

end sequence_a2002_l242_242311


namespace battery_current_when_resistance_12_l242_242077

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l242_242077


namespace min_positive_period_and_symmetry_axis_l242_242499

noncomputable def f (x : ℝ) := - (Real.sin (x + Real.pi / 6)) * (Real.sin (x - Real.pi / 3))

theorem min_positive_period_and_symmetry_axis :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧ (∃ k : ℤ, ∀ x : ℝ, f x = f (x + 1 / 2 * k * Real.pi + Real.pi / 12)) := by
  sorry

end min_positive_period_and_symmetry_axis_l242_242499


namespace intersection_A_B_l242_242592

def A : Set ℤ := { -2, -1, 0, 1, 2 }
def B : Set ℤ := { x : ℤ | x < 1 }

theorem intersection_A_B : A ∩ B = { -2, -1, 0 } :=
by sorry

end intersection_A_B_l242_242592


namespace min_value_x3_l242_242438

noncomputable def min_x3 (x1 x2 x3 : ℝ) : ℝ := -21 / 11

theorem min_value_x3 (x1 x2 x3 : ℝ) 
  (h1 : x1 + (1 / 2) * x2 + (1 / 3) * x3 = 1)
  (h2 : x1^2 + (1 / 2) * x2^2 + (1 / 3) * x3^2 = 3) 
  : x3 ≥ - (21 / 11) := 
by sorry

end min_value_x3_l242_242438


namespace sqrt_sum_odds_l242_242236

theorem sqrt_sum_odds : 
  (Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + Real.sqrt (1+3+5+7+9) + Real.sqrt (1+3+5+7+9+11)) = 21 := 
by
  sorry

end sqrt_sum_odds_l242_242236


namespace right_triangle_excircle_incircle_l242_242624

theorem right_triangle_excircle_incircle (a b c r r_a : ℝ) (h : a^2 + b^2 = c^2) :
  (r = (a + b - c) / 2) → (r_a = (b + c - a) / 2) → r_a = 2 * r :=
by
  intros hr hra
  sorry

end right_triangle_excircle_incircle_l242_242624


namespace cyclic_permutations_sum_41234_l242_242155

theorem cyclic_permutations_sum_41234 :
  let n1 := 41234
  let n2 := 34124
  let n3 := 23414
  let n4 := 12434
  3 * (n1 + n2 + n3 + n4) = 396618 :=
by
  let n1 := 41234
  let n2 := 34124
  let n3 := 23414
  let n4 := 12434
  show 3 * (n1 + n2 + n3 + n4) = 396618
  sorry

end cyclic_permutations_sum_41234_l242_242155


namespace sum_lt_2500_probability_l242_242406

open ProbabilityTheory

theorem sum_lt_2500_probability :
  let x : MeasureTheory.MeasureSpace ℝ := uniform (Icc 0 1000)
  let y : MeasureTheory.MeasureSpace ℝ := uniform (Icc 0 3000)
  P (λ (xy : ℝ × ℝ), xy.1 + xy.2 < 2500) = 1 / 4 :=
by
  -- uniform distribution properties and combination
  -- calculation will be required here
  sorry

end sum_lt_2500_probability_l242_242406


namespace expected_value_of_win_is_correct_l242_242834

noncomputable def expected_value_of_win : ℝ :=
  (1 / 8) * (8 - 1) + (1 / 8) * (8 - 2) + (1 / 8) * (8 - 3) +
  (1 / 8) * (8 - 4) + (1 / 8) * (8 - 5) + (1 / 8) * (8 - 6) +
  (1 / 8) * (8 - 7) + (1 / 8) * (8 - 8)

theorem expected_value_of_win_is_correct :
  expected_value_of_win = 3.5 :=
by
  sorry

end expected_value_of_win_is_correct_l242_242834


namespace current_value_l242_242072

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l242_242072


namespace number_of_ways_to_choose_l242_242826

-- Define the teachers and classes
def teachers : ℕ := 5
def classes : ℕ := 4
def choices (t : ℕ) : ℕ := classes

-- Formalize the problem statement
theorem number_of_ways_to_choose : (choices teachers) ^ teachers = 1024 :=
by
  -- We denote the computation of (4^5)
  sorry

end number_of_ways_to_choose_l242_242826


namespace smallest_possible_w_l242_242591

theorem smallest_possible_w 
  (h1 : 936 = 2^3 * 3 * 13)
  (h2 : 2^5 = 32)
  (h3 : 3^3 = 27)
  (h4 : 14^2 = 196) :
  ∃ w : ℕ, (w > 0) ∧ (936 * w) % 32 = 0 ∧ (936 * w) % 27 = 0 ∧ (936 * w) % 196 = 0 ∧ w = 1764 :=
sorry

end smallest_possible_w_l242_242591


namespace expected_value_of_winnings_eq_3_point_5_l242_242532

noncomputable def expected_winnings : ℝ :=
  ∑ i in finset.range 8, (8 - i) * (1 / 8 : ℝ)

theorem expected_value_of_winnings_eq_3_point_5 :
  expected_winnings = 3.5 :=
sorry

end expected_value_of_winnings_eq_3_point_5_l242_242532


namespace current_when_resistance_is_12_l242_242031

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l242_242031


namespace probability_at_least_one_black_eq_seven_tenth_l242_242803

noncomputable def probability_drawing_at_least_one_black_ball : ℚ :=
  let total_ways := Nat.choose 5 2
  let ways_no_black := Nat.choose 3 2
  1 - (ways_no_black / total_ways)

theorem probability_at_least_one_black_eq_seven_tenth :
  probability_drawing_at_least_one_black_ball = 7 / 10 :=
by
  sorry

end probability_at_least_one_black_eq_seven_tenth_l242_242803


namespace which_is_right_triangle_l242_242243

-- Definitions for each group of numbers
def sides_A := (1, 2, 3)
def sides_B := (3, 4, 5)
def sides_C := (4, 5, 6)
def sides_D := (7, 8, 9)

-- Definition of a condition for right triangle using the converse of the Pythagorean theorem
def is_right_triangle (a b c: ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem which_is_right_triangle :
    ¬is_right_triangle 1 2 3 ∧
    ¬is_right_triangle 4 5 6 ∧
    ¬is_right_triangle 7 8 9 ∧
    is_right_triangle 3 4 5 :=
by
  sorry

end which_is_right_triangle_l242_242243


namespace non_adjacent_placements_l242_242569

theorem non_adjacent_placements (n : ℕ) : 
  let total_ways := n^2 * (n^2 - 1)
  let adjacent_ways := 2 * n^2 - 2 * n
  (total_ways - adjacent_ways) = n^4 - 3 * n^2 + 2 * n :=
by
  -- Proof is sorted out
  sorry

end non_adjacent_placements_l242_242569


namespace circle_area_isosceles_triangle_l242_242530

theorem circle_area_isosceles_triangle (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 3) (h₃ : c = 2) :
  ∃ R : ℝ, R = (81 / 32) * Real.pi :=
by sorry

end circle_area_isosceles_triangle_l242_242530


namespace selection_methods_l242_242454

theorem selection_methods (students lectures : ℕ) (h_stu : students = 4) (h_lect : lectures = 3) : 
  (lectures ^ students) = 81 := 
by
  rw [h_stu, h_lect]
  rfl

end selection_methods_l242_242454


namespace circle_diameter_and_circumference_l242_242650

theorem circle_diameter_and_circumference (A : ℝ) (hA : A = 225 * π) : 
  ∃ r d C, r = 15 ∧ d = 2 * r ∧ C = 2 * π * r ∧ d = 30 ∧ C = 30 * π :=
by
  sorry

end circle_diameter_and_circumference_l242_242650


namespace sum_of_powers_sequence_l242_242978

theorem sum_of_powers_sequence (a b : ℝ) 
  (h₁ : a + b = 1)
  (h₂ : a^2 + b^2 = 3)
  (h₃ : a^3 + b^3 = 4)
  (h₄ : a^4 + b^4 = 7)
  (h₅ : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 :=
sorry

end sum_of_powers_sequence_l242_242978


namespace pizza_topping_count_l242_242265

theorem pizza_topping_count (n : ℕ) (hn : n = 8) : 
  (nat.choose n 1) + (nat.choose n 2) + (nat.choose n 3) = 92 :=
by
  sorry

end pizza_topping_count_l242_242265


namespace count_numbers_containing_5_or_7_l242_242321

theorem count_numbers_containing_5_or_7 :
  let N := (1 : ℕ) ..
  let numbers := List.range' 1 (701)
  let contains_digit_5_or_7 (n : ℕ) : Prop :=
    (n.digits 10).contains 5 ∨ (n.digits 10).contains 7
  ∃ nums_subset : Finset ℕ, nums_subset.card = 244 ∧ ∀ n ∈ nums_subset, contains_digit_5_or_7 n :=
begin
  sorry
end

end count_numbers_containing_5_or_7_l242_242321


namespace max_value_of_f_f_is_increasing_on_intervals_l242_242169

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos x ^ 2 + Real.sqrt 3 * Real.sin (2 * x)

theorem max_value_of_f :
  ∃ (k : ℤ), ∀ (x : ℝ), x = k * Real.pi + Real.pi / 6 → f x = 3 :=
sorry

theorem f_is_increasing_on_intervals :
  ∀ (k : ℤ), ∀ (x y : ℝ), k * Real.pi - Real.pi / 3 ≤ x →
                x ≤ y → y ≤ k * Real.pi + Real.pi / 6 →
                f x ≤ f y :=
sorry

end max_value_of_f_f_is_increasing_on_intervals_l242_242169


namespace largest_n_crates_same_number_oranges_l242_242879

theorem largest_n_crates_same_number_oranges (total_crates : ℕ) 
  (crate_min_oranges : ℕ) (crate_max_oranges : ℕ) 
  (h1 : total_crates = 200) (h2 : crate_min_oranges = 100) (h3 : crate_max_oranges = 130) 
  : ∃ n : ℕ, n = 7 ∧ ∀ orange_count, crate_min_oranges ≤ orange_count ∧ orange_count ≤ crate_max_oranges → ∃ k, k = n ∧ ∃ t, t ≤ total_crates ∧ t ≥ k := 
sorry

end largest_n_crates_same_number_oranges_l242_242879


namespace buddy_thursday_cards_l242_242621

-- Definitions from the given conditions
def monday_cards : ℕ := 30
def tuesday_cards : ℕ := monday_cards / 2
def wednesday_cards : ℕ := tuesday_cards + 12
def thursday_extra_cards : ℕ := tuesday_cards / 3
def thursday_cards : ℕ := wednesday_cards + thursday_extra_cards

-- Theorem to prove the total number of baseball cards on Thursday
theorem buddy_thursday_cards : thursday_cards = 32 :=
by
  -- Proof steps would go here, but we just provide the result for now
  sorry

end buddy_thursday_cards_l242_242621


namespace intermediate_root_exists_l242_242980

open Polynomial

theorem intermediate_root_exists
  (a b c x1 x2 : ℝ) 
  (h1 : a * x1^2 + b * x1 + c = 0) 
  (h2 : -a * x2^2 + b * x2 + c = 0) :
  ∃ x3 : ℝ, (a / 2) * x3^2 + b * x3 + c = 0 ∧ (x1 ≤ x3 ∧ x3 ≤ x2 ∨ x1 ≥ x3 ∧ x3 ≥ x2) :=
sorry

end intermediate_root_exists_l242_242980


namespace find_number_l242_242931

theorem find_number (x : ℝ) (h : 0.6667 * x + 1 = 0.75 * x) : x = 12 := 
by
  sorry

end find_number_l242_242931


namespace f_odd_and_minimum_period_pi_l242_242792

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x

theorem f_odd_and_minimum_period_pi :
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + π) = f x) :=
  sorry

end f_odd_and_minimum_period_pi_l242_242792


namespace find_x_that_satisfies_f_l242_242794

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem find_x_that_satisfies_f (α : ℝ) (x : ℝ) (h : power_function α (-2) = -1/8) : 
  power_function α x = 27 → x = 1/3 :=
  by
  sorry

end find_x_that_satisfies_f_l242_242794


namespace point_M_coordinates_l242_242578

open Real

theorem point_M_coordinates (θ : ℝ) (h_tan : tan θ = -4 / 3) (h_theta : π / 2 < θ ∧ θ < π) :
  let x := 5 * cos θ
  let y := 5 * sin θ
  (x, y) = (-3, 4) := 
by 
  sorry

end point_M_coordinates_l242_242578


namespace max_absolute_difference_l242_242009

theorem max_absolute_difference (a b c d e : ℤ) (p : ℤ) :
  0 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e ∧ e ≤ 100 ∧ p = (a + b + c + d + e) / 5 →
  (|p - c| ≤ 40) :=
by
  sorry

end max_absolute_difference_l242_242009


namespace no_maximal_radius_of_inscribed_cylinder_l242_242791

theorem no_maximal_radius_of_inscribed_cylinder
  (base_radius_cone : ℝ) (height_cone : ℝ)
  (h_base_radius : base_radius_cone = 5) (h_height : height_cone = 10) :
  ¬ ∃ r : ℝ, 0 < r ∧ r < 5 ∧
    ∀ t : ℝ, 0 < t ∧ t < 5 → 2 * Real.pi * (10 * r - r ^ 2) ≥ 2 * Real.pi * (10 * t - t ^ 2) :=
by
  sorry

end no_maximal_radius_of_inscribed_cylinder_l242_242791


namespace garden_wall_additional_courses_l242_242606

theorem garden_wall_additional_courses (initial_courses additional_courses : ℕ) (bricks_per_course total_bricks bricks_removed : ℕ) 
  (h1 : bricks_per_course = 400) 
  (h2 : initial_courses = 3) 
  (h3 : bricks_removed = bricks_per_course / 2) 
  (h4 : total_bricks = 1800) 
  (h5 : total_bricks = initial_courses * bricks_per_course + additional_courses * bricks_per_course - bricks_removed) : 
  additional_courses = 2 :=
by
  sorry

end garden_wall_additional_courses_l242_242606


namespace each_wolf_needs_to_kill_one_deer_l242_242996

-- Conditions
def wolves_out_hunting : ℕ := 4
def additional_wolves : ℕ := 16
def wolves_total : ℕ := wolves_out_hunting + additional_wolves
def meat_per_wolf_per_day : ℕ := 8
def days_no_hunt : ℕ := 5
def meat_per_deer : ℕ := 200

-- Calculate total meat needed for all wolves over five days.
def total_meat_needed : ℕ := wolves_total * meat_per_wolf_per_day * days_no_hunt
-- Calculate total number of deer needed to meet the meat requirement.
def deer_needed : ℕ := total_meat_needed / meat_per_deer
-- Calculate number of deer each hunting wolf needs to kill.
def deer_per_wolf : ℕ := deer_needed / wolves_out_hunting

-- The proof statement
theorem each_wolf_needs_to_kill_one_deer : deer_per_wolf = 1 := 
by { sorry }

end each_wolf_needs_to_kill_one_deer_l242_242996


namespace find_x1_l242_242944

theorem find_x1 (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3) (h2 : x3 ≤ x2) (h3 : x2 ≤ x1) (h4 : x1 ≤ 1) 
    (h5 : (1 - x1)^3 + (x1 - x2)^3 + (x2 - x3)^3 + x3^3 = 1 / 8) : x1 = 3 / 4 := 
by 
  sorry

end find_x1_l242_242944


namespace ellipse_solution_length_AB_l242_242741

noncomputable def ellipse_equation (a b : ℝ) (e : ℝ) (minor_axis : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ e = 3 / 4 ∧ 2 * b = minor_axis ∧ minor_axis = 2 * Real.sqrt 7

theorem ellipse_solution (a b : ℝ) (e : ℝ) (minor_axis : ℝ) :
  ellipse_equation a b e minor_axis →
  (a^2 = 16 ∧ b^2 = 7 ∧ (1 / a^2) = 1 / 16 ∧ (1 / b^2) = 1 / 7) :=
by 
  intros h
  sorry

noncomputable def area_ratio (S1 S2 : ℝ) : Prop :=
  S1 / S2 = 9 / 13

theorem length_AB (S1 S2 : ℝ) :
  area_ratio S1 S2 →
  |S1 / S2| = |(9 * Real.sqrt 105) / 26| :=
by
  intros h
  sorry

end ellipse_solution_length_AB_l242_242741


namespace largest_angle_in_triangle_l242_242365

theorem largest_angle_in_triangle (y : ℝ) (h : 60 + 70 + y = 180) :
    70 > 60 ∧ 70 > y :=
by {
  sorry
}

end largest_angle_in_triangle_l242_242365


namespace current_at_resistance_12_l242_242035

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l242_242035


namespace parallel_to_a_perpendicular_to_a_l242_242429

-- Definition of vectors a and b and conditions
def a : ℝ × ℝ := (3, 4)
def b (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Mathematical statement for Problem (1)
theorem parallel_to_a (x y : ℝ) (h : b x y) (h_parallel : 3 * y - 4 * x = 0) :
  (x = 3/5 ∧ y = 4/5) ∨ (x = -3/5 ∧ y = -4/5) := 
sorry

-- Mathematical statement for Problem (2)
theorem perpendicular_to_a (x y : ℝ) (h : b x y) (h_perpendicular : 3 * x + 4 * y = 0) :
  (x = -4/5 ∧ y = 3/5) ∨ (x = 4/5 ∧ y = -3/5) := 
sorry

end parallel_to_a_perpendicular_to_a_l242_242429


namespace battery_current_l242_242081

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l242_242081


namespace find_missing_digit_divisibility_by_4_l242_242935

theorem find_missing_digit_divisibility_by_4 (x : ℕ) (h : x < 10) :
  (3280 + x) % 4 = 0 ↔ x = 0 ∨ x = 2 ∨ x = 4 ∨ x = 6 ∨ x = 8 :=
by
  sorry

end find_missing_digit_divisibility_by_4_l242_242935


namespace circle_line_no_intersection_l242_242166

theorem circle_line_no_intersection (b : ℝ) :
  (∀ x y : ℝ, ¬ (x^2 + y^2 = 2 ∧ y = x + b)) ↔ (b > 2 ∨ b < -2) :=
by sorry

end circle_line_no_intersection_l242_242166


namespace joel_donated_22_toys_l242_242184

-- Given conditions
variables (T : ℕ)
variables (stuffed_animals action_figures board_games puzzles : ℕ)
variables (total_donated : ℕ)

-- Define the conditions
def conditions := 
  stuffed_animals = 18 ∧ 
  action_figures = 42 ∧ 
  board_games = 2 ∧ 
  puzzles = 13 ∧ 
  total_donated = 108

-- Calculating the total number of toys from friends
def friends_total := 
  stuffed_animals + action_figures + board_games + puzzles

-- The total number of toys Joel and his sister donated
def total_joel_sister := 
  T + 2 * T

-- The proof problem
theorem joel_donated_22_toys 
  (h : conditions) : 
  3 * T + friends_total = total_donated → 2 * T = 22 :=
by
  intros
  sorry

end joel_donated_22_toys_l242_242184


namespace solution_set_inequality_l242_242227

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) / (x + 2) < 0 ↔ -2 < x ∧ x < 3 :=
sorry

end solution_set_inequality_l242_242227


namespace triangle_side_lengths_l242_242955

variable {c z m : ℕ}

axiom condition1 : 3 * c + z + m = 43
axiom condition2 : c + z + 3 * m = 35
axiom condition3 : 2 * (c + z + m) = 46

theorem triangle_side_lengths : c = 10 ∧ z = 7 ∧ m = 6 := 
by 
  sorry

end triangle_side_lengths_l242_242955


namespace vertex_of_quadratic1_vertex_of_quadratic2_l242_242409

theorem vertex_of_quadratic1 :
  ∃ x y : ℝ, 
  (∀ x', 2 * x'^2 - 4 * x' - 1 = 2 * (x' - x)^2 + y) ∧ 
  (x = 1 ∧ y = -3) :=
by sorry

theorem vertex_of_quadratic2 :
  ∃ x y : ℝ, 
  (∀ x', -3 * x'^2 + 6 * x' - 2 = -3 * (x' - x)^2 + y) ∧ 
  (x = 1 ∧ y = 1) :=
by sorry

end vertex_of_quadratic1_vertex_of_quadratic2_l242_242409


namespace find_b_l242_242723

open Real

theorem find_b (b : ℝ) (h : b + ⌈b⌉ = 21.5) : b = 10.5 :=
sorry

end find_b_l242_242723


namespace sum_of_first_4_terms_arithmetic_sequence_l242_242482

variable {a : ℕ → ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d a1, (∀ n, a n = a1 + n * d) ∧ (a 3 - a 1 = 2) ∧ (a 5 = 5)

-- Define the sum S4 for the first 4 terms of the sequence
def sum_first_4_terms (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3

-- Define the Lean statement for the problem
theorem sum_of_first_4_terms_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a → sum_first_4_terms a = 10 :=
by
  sorry

end sum_of_first_4_terms_arithmetic_sequence_l242_242482


namespace expected_value_of_win_is_3_5_l242_242861

noncomputable def expected_value_win : ℝ :=
  (1/8) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

theorem expected_value_of_win_is_3_5 :
  expected_value_win = 3.5 :=
by
  sorry

end expected_value_of_win_is_3_5_l242_242861


namespace solve_eq_l242_242629

-- Defining the condition
def eq_condition (x : ℝ) : Prop := (x - 3) ^ 2 = x ^ 2 - 9

-- The statement we need to prove
theorem solve_eq (x : ℝ) (h : eq_condition x) : x = 3 :=
by
  sorry

end solve_eq_l242_242629


namespace sum_of_Ns_l242_242445

theorem sum_of_Ns (N R : ℝ) (hN_nonzero : N ≠ 0) (h_eq : N - 3 * N^2 = R) : 
  ∃ N1 N2 : ℝ, N1 ≠ 0 ∧ N2 ≠ 0 ∧ 3 * N1^2 - N1 + R = 0 ∧ 3 * N2^2 - N2 + R = 0 ∧ (N1 + N2) = 1 / 3 :=
sorry

end sum_of_Ns_l242_242445


namespace square_root_of_9_l242_242379

theorem square_root_of_9 : {x : ℝ // x^2 = 9} = {x : ℝ // x = 3 ∨ x = -3} :=
by
  sorry

end square_root_of_9_l242_242379


namespace chalkboard_area_l242_242347

def width : Float := 3.5
def length : Float := 2.3 * width
def area : Float := length * width

theorem chalkboard_area : area = 28.175 :=
by 
  sorry

end chalkboard_area_l242_242347


namespace arithmetic_sequence_a1_a9_l242_242149

variable (a : ℕ → ℝ)

-- This statement captures if given condition holds, prove a_1 + a_9 = 18.
theorem arithmetic_sequence_a1_a9 (h : a 4 + a 5 + a 6 = 27)
    (h_seq : ∀ (n : ℕ), a (n + 1) = a n + (a 2 - a 1)) :
    a 1 + a 9 = 18 :=
sorry

end arithmetic_sequence_a1_a9_l242_242149


namespace net_population_increase_per_day_l242_242522

def birth_rate : Nat := 4
def death_rate : Nat := 2
def seconds_per_day : Nat := 24 * 60 * 60

theorem net_population_increase_per_day : 
  (birth_rate - death_rate) * (seconds_per_day / 2) = 86400 := by
  sorry

end net_population_increase_per_day_l242_242522


namespace storybook_pages_l242_242248

def reading_start_date := 10
def reading_end_date := 20
def pages_per_day := 11
def number_of_days := reading_end_date - reading_start_date + 1
def total_pages := pages_per_day * number_of_days

theorem storybook_pages : total_pages = 121 := by
  sorry

end storybook_pages_l242_242248


namespace battery_current_at_given_resistance_l242_242096

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l242_242096


namespace enclosed_area_is_correct_l242_242258

noncomputable def area_between_curves : ℝ := 
  let circle (x y : ℝ) := x^2 + y^2 = 4
  let cubic_parabola (x : ℝ) := - 1 / 2 * x^3 + 2 * x
  let x1 : ℝ := -2
  let x2 : ℝ := Real.sqrt 2
  -- Properly calculate the area between the two curves
  sorry

theorem enclosed_area_is_correct :
  area_between_curves = 3 * ( Real.pi + 1 ) / 2 :=
sorry

end enclosed_area_is_correct_l242_242258


namespace compare_fractions_l242_242304

theorem compare_fractions (a : ℝ) : 
  (a = 0 → (1 / (1 - a)) = (1 + a)) ∧ 
  (0 < a ∧ a < 1 → (1 / (1 - a)) > (1 + a)) ∧ 
  (a > 1 → (1 / (1 - a)) < (1 + a)) := by
  sorry

end compare_fractions_l242_242304


namespace three_squares_not_divisible_by_three_l242_242198

theorem three_squares_not_divisible_by_three 
  (N : ℕ) (a b c : ℤ) 
  (h₁ : N = 9 * (a^2 + b^2 + c^2)) :
  ∃ x y z : ℤ, N = x^2 + y^2 + z^2 ∧ ¬ (3 ∣ x) ∧ ¬ (3 ∣ y) ∧ ¬ (3 ∣ z) := 
sorry

end three_squares_not_divisible_by_three_l242_242198


namespace min_moves_queens_switch_places_l242_242622

-- Assume a type representing the board positions
inductive Position where
| first_rank | last_rank 

-- Assume a type representing the queens
inductive Queen where
| black | white

-- Function to count minimum moves for switching places
def min_moves_to_switch_places : ℕ :=
  sorry

theorem min_moves_queens_switch_places :
  min_moves_to_switch_places = 23 :=
  sorry

end min_moves_queens_switch_places_l242_242622


namespace probability_both_counterfeits_given_one_is_counterfeit_l242_242979

open ProbabilityTheory

def total_banknotes : ℕ := 20
def counterfeit_banknotes : ℕ := 5
def total_pairs := Nat.choose total_banknotes 2
def counterfeit_pairs := Nat.choose counterfeit_banknotes 2
def one_counterfeit_one_real_pairs := counterfeit_banknotes * (total_banknotes - counterfeit_banknotes)
def prob_A := (counterfeit_pairs : ℚ) / (total_pairs : ℚ)
def prob_B := ((counterfeit_pairs : ℚ) + (one_counterfeit_one_real_pairs : ℚ)) / (total_pairs : ℚ)

theorem probability_both_counterfeits_given_one_is_counterfeit 
  (hA : A ⊆ B) :
  (prob_A / prob_B) = (2 / 17) := by
sorry

end probability_both_counterfeits_given_one_is_counterfeit_l242_242979


namespace not_enough_money_l242_242817

-- Define the prices of the books
def price_animal_world : Real := 21.8
def price_fairy_tale_stories : Real := 19.5

-- Define the total amount of money Xiao Ming has
def xiao_ming_money : Real := 40.0

-- Define the statement we want to prove
theorem not_enough_money : (price_animal_world + price_fairy_tale_stories) > xiao_ming_money := by
  sorry

end not_enough_money_l242_242817


namespace picnic_problem_l242_242677

variable (M W A C : ℕ)

theorem picnic_problem (h1 : M = 90)
  (h2 : M = W + 40)
  (h3 : M + W + C = 240) :
  A = M + W ∧ A - C = 40 := by
  sorry

end picnic_problem_l242_242677


namespace expected_value_of_win_is_3point5_l242_242836

noncomputable def expected_value_win : ℝ := 
  ∑ n in Finset.range 8, (1 / 8 : ℝ) * (8 - n)

theorem expected_value_of_win_is_3point5 : expected_value_win = 3.5 := 
by 
  sorry

end expected_value_of_win_is_3point5_l242_242836


namespace binomial_7_4_l242_242135

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_7_4 : binomial 7 4 = 35 := by
  sorry

end binomial_7_4_l242_242135


namespace cost_of_pen_l242_242785

theorem cost_of_pen 
  (total_amount_spent : ℕ)
  (total_items : ℕ)
  (number_of_pencils : ℕ)
  (cost_of_pencil : ℕ)
  (cost_of_pen : ℕ)
  (h1 : total_amount_spent = 2000)
  (h2 : total_items = 36)
  (h3 : number_of_pencils = 16)
  (h4 : cost_of_pencil = 25)
  (remaining_amount_spent : ℕ)
  (number_of_pens : ℕ)
  (h5 : remaining_amount_spent = total_amount_spent - (number_of_pencils * cost_of_pencil))
  (h6 : number_of_pens = total_items - number_of_pencils)
  (total_cost_of_pens : ℕ)
  (h7 : total_cost_of_pens = remaining_amount_spent)
  (h8 : total_cost_of_pens = number_of_pens * cost_of_pen)
  : cost_of_pen = 80 := by
  sorry

end cost_of_pen_l242_242785


namespace grade_assignment_ways_l242_242398

/-- Define the number of students and the number of grade choices -/
def num_students : ℕ := 15
def num_grades : ℕ := 4

/-- Define the total number of ways to assign grades -/
def total_ways : ℕ := num_grades ^ num_students

/-- Prove that the total number of ways to assign grades is 4^15 -/
theorem grade_assignment_ways : total_ways = 1073741824 := by
  -- proof here
  sorry

end grade_assignment_ways_l242_242398


namespace current_at_resistance_12_l242_242036

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l242_242036


namespace relationship_abc_l242_242427

noncomputable def a (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ := (Real.sin x) / x
noncomputable def b (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ := (Real.sin (x^3)) / (x^3)
noncomputable def c (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ := ((Real.sin x)^3) / (x^3)

theorem relationship_abc (x : ℝ) (hx : 0 < x ∧ x < 1) : b x hx > a x hx ∧ a x hx > c x hx :=
by
  sorry

end relationship_abc_l242_242427


namespace words_per_page_is_106_l242_242666

noncomputable def book_pages := 154
noncomputable def max_words_per_page := 120
noncomputable def total_words_mod := 221
noncomputable def mod_val := 217

def number_of_words_per_page (p : ℕ) : Prop :=
  (book_pages * p ≡ total_words_mod [MOD mod_val]) ∧ (p ≤ max_words_per_page)

theorem words_per_page_is_106 : number_of_words_per_page 106 :=
by
  sorry

end words_per_page_is_106_l242_242666


namespace parallelepiped_diagonal_inequality_l242_242466

theorem parallelepiped_diagonal_inequality 
  (a b c d : ℝ) 
  (h_d : d = Real.sqrt (a^2 + b^2 + c^2)) : 
  a^2 + b^2 + c^2 ≥ d^2 / 3 := 
by 
  sorry

end parallelepiped_diagonal_inequality_l242_242466


namespace maria_anna_ages_l242_242644

theorem maria_anna_ages : 
  ∃ (x y : ℝ), x + y = 44 ∧ x = 2 * (y - (- (1/2) * x + (3/2) * ((2/3) * y))) ∧ x = 27.5 ∧ y = 16.5 := by 
  sorry

end maria_anna_ages_l242_242644


namespace fisherman_bass_count_l242_242673

theorem fisherman_bass_count (B T G : ℕ) (h1 : T = B / 4) (h2 : G = 2 * B) (h3 : B + T + G = 104) : B = 32 :=
by
  sorry

end fisherman_bass_count_l242_242673


namespace cab_to_bus_ratio_l242_242608

noncomputable def train_distance : ℤ := 300
noncomputable def bus_distance : ℤ := train_distance / 2
noncomputable def total_distance : ℤ := 500
noncomputable def cab_distance : ℤ := total_distance - (train_distance + bus_distance)
noncomputable def ratio : ℚ := cab_distance / bus_distance

theorem cab_to_bus_ratio :
  ratio = 1 / 3 := by
  sorry

end cab_to_bus_ratio_l242_242608


namespace number_of_matching_parity_sequences_l242_242758

-- Definition of digits and parity property

def digits : Finset ℕ := Finset.range 10  -- {0, 1, ..., 9}

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
    
def matching_parity_sequence (n : ℕ) : Finset (Fin n → ℕ) :=
  {s | ∀ i, i < n - 1 → (is_even (s i) ↔ is_even (s (i + 1))) ∨ (is_odd (s i) ↔ is_odd (s (i + 1)))}
    
-- The Problem Statement
theorem number_of_matching_parity_sequences : 
  (matching_parity_sequence 7).card = 156250 :=
sorry

end number_of_matching_parity_sequences_l242_242758


namespace minimum_value_l242_242498

open Real

theorem minimum_value (a : ℝ) (m n : ℝ) (h_a : a > 0) (h_a_not_one : a ≠ 1) 
                      (h_mn : m * n > 0) (h_point : -m - n + 1 = 0) :
  (1 / m + 2 / n) = 3 + 2 * sqrt 2 :=
by
  -- proof should go here
  sorry

end minimum_value_l242_242498


namespace smallest_four_digit_congruent_one_mod_17_l242_242810

theorem smallest_four_digit_congruent_one_mod_17 :
  ∃ (n : ℕ), 1000 ≤ n ∧ n % 17 = 1 ∧ n = 1003 :=
by
sorry

end smallest_four_digit_congruent_one_mod_17_l242_242810


namespace range_of_a_l242_242959

theorem range_of_a (a : ℝ) : (∀ (x : ℝ), x > 0 → x / (x ^ 2 + 3 * x + 1) ≤ a) → a ≥ 1 / 5 :=
by
  sorry

end range_of_a_l242_242959


namespace cost_of_swim_trunks_is_14_l242_242882

noncomputable def cost_of_swim_trunks : Real :=
  let flat_rate_shipping := 5.00
  let shipping_rate := 0.20
  let price_shirt := 12.00
  let price_socks := 5.00
  let price_shorts := 15.00
  let cost_known_items := 3 * price_shirt + price_socks + 2 * price_shorts
  let total_bill := 102.00
  let x := (total_bill - 0.20 * cost_known_items - cost_known_items) / 1.20
  x

theorem cost_of_swim_trunks_is_14 : cost_of_swim_trunks = 14 := by
  -- sorry is used to skip the proof
  sorry

end cost_of_swim_trunks_is_14_l242_242882


namespace greg_total_earnings_correct_l242_242318

def charge_per_dog := 20
def charge_per_minute := 1

def earnings_one_dog := charge_per_dog + charge_per_minute * 10
def earnings_two_dogs := 2 * (charge_per_dog + charge_per_minute * 7)
def earnings_three_dogs := 3 * (charge_per_dog + charge_per_minute * 9)

def total_earnings := earnings_one_dog + earnings_two_dogs + earnings_three_dogs

theorem greg_total_earnings_correct : total_earnings = 171 := by
  sorry

end greg_total_earnings_correct_l242_242318


namespace expected_value_of_win_is_3_5_l242_242863

noncomputable def expected_value_win : ℝ :=
  (1/8) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

theorem expected_value_of_win_is_3_5 :
  expected_value_win = 3.5 :=
by
  sorry

end expected_value_of_win_is_3_5_l242_242863


namespace remaining_games_win_percent_l242_242128

variable (totalGames : ℕ) (firstGames : ℕ) (firstWinPercent : ℕ) (seasonWinPercent : ℕ)

-- Given conditions expressed as assumptions:
-- The total number of games played in a season is 40
axiom total_games_condition : totalGames = 40
-- The number of first games played is 30
axiom first_games_condition : firstGames = 30
-- The team won 40% of the first 30 games
axiom first_win_percent_condition : firstWinPercent = 40
-- The team won 50% of all its games in the season
axiom season_win_percent_condition : seasonWinPercent = 50

-- We need to prove that the percentage of the remaining games that the team won is 80%
theorem remaining_games_win_percent {remainingWinPercent : ℕ} :
  totalGames = 40 →
  firstGames = 30 →
  firstWinPercent = 40 →
  seasonWinPercent = 50 →
  remainingWinPercent = 80 :=
by
  intros
  sorry

end remaining_games_win_percent_l242_242128


namespace p_implies_q_and_not_converse_l242_242574

def p (a : ℝ) := a ≤ 1
def q (a : ℝ) := abs a ≤ 1

theorem p_implies_q_and_not_converse (a : ℝ) : (p a → q a) ∧ ¬(q a → p a) :=
by
  repeat { sorry }

end p_implies_q_and_not_converse_l242_242574


namespace find_common_ratio_find_sum_of_first_n_terms_l242_242573

open Real

-- Given conditions
variable {a : ℕ → ℝ}
variable {q : ℝ}
variable (hq : q > 1)
variable (h1 : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1))
variable (h2 : a 5 ^ 2 = a 10)

-- Correct answer statements (equivalently reformulated questions)
theorem find_common_ratio : q = 2 := sorry

theorem find_sum_of_first_n_terms {n : ℕ} :
  ∑ i in finset.range n, (a i / 3 ^ i) = 2 * (1 - (2 / 3) ^ n) := sorry

end find_common_ratio_find_sum_of_first_n_terms_l242_242573


namespace total_chocolate_bars_in_large_box_l242_242662

def large_box_contains_18_small_boxes : ℕ := 18
def small_box_contains_28_chocolate_bars : ℕ := 28

theorem total_chocolate_bars_in_large_box :
  (large_box_contains_18_small_boxes * small_box_contains_28_chocolate_bars) = 504 := 
by
  sorry

end total_chocolate_bars_in_large_box_l242_242662


namespace rabbits_clear_land_in_21_days_l242_242514

theorem rabbits_clear_land_in_21_days (length_feet width_feet : ℝ) (rabbits : ℕ) (clear_per_rabbit_per_day : ℝ) : 
  length_feet = 900 → width_feet = 200 → rabbits = 100 → clear_per_rabbit_per_day = 10 →
  (⌈ (length_feet / 3 * width_feet / 3) / (rabbits * clear_per_rabbit_per_day) ⌉ = 21) := 
by
  intros
  sorry

end rabbits_clear_land_in_21_days_l242_242514


namespace solve_inequality_l242_242936

theorem solve_inequality (x : ℝ) : (|x - 3| + |x - 5| ≥ 4) ↔ (x ≤ 2 ∨ x ≥ 6) :=
by
  sorry

end solve_inequality_l242_242936


namespace percentage_of_female_students_l242_242451

theorem percentage_of_female_students {F : ℝ} (h1 : 200 > 0): ((200 * (F / 100)) * 0.5 * 0.5 = 30) → (F = 60) :=
by
  sorry

end percentage_of_female_students_l242_242451


namespace coefficient_ratio_is_4_l242_242297

noncomputable def coefficient_x3 := 
  let a := 60 -- Coefficient of x^3 in the expansion
  let b := Nat.choose 6 2 -- Binomial coefficient \binom{6}{2}
  a / b

theorem coefficient_ratio_is_4 : coefficient_x3 = 4 := by
  sorry

end coefficient_ratio_is_4_l242_242297


namespace number_of_rallies_l242_242288

open Nat

def X_rallies : Nat := 10
def O_rallies : Nat := 100
def sequence_Os : Nat := 3
def sequence_Xs : Nat := 7

theorem number_of_rallies : 
  (sequence_Os * O_rallies + sequence_Xs * X_rallies ≤ 379) ∧ 
  (sequence_Os * O_rallies + sequence_Xs * X_rallies ≥ 370) := 
by
  sorry

end number_of_rallies_l242_242288


namespace expected_value_of_win_is_3_point_5_l242_242848

noncomputable def expected_value_win : ℚ :=
  let win (n : ℕ) : ℚ := 8 - n
  let probabilities := List.repeat (1/8 : ℚ) 8
  (List.range 8).map (λ n => probabilities.head! * win (n + 1)).sum

theorem expected_value_of_win_is_3_point_5 : expected_value_win = 3.5 := by
  sorry

end expected_value_of_win_is_3_point_5_l242_242848


namespace work_completed_in_5_days_l242_242661

-- Define the rates of work for A, B, and C
def rateA : ℚ := 1 / 15
def rateB : ℚ := 1 / 14
def rateC : ℚ := 1 / 16

-- Summing their rates to get the combined rate
def combined_rate : ℚ := rateA + rateB + rateC

-- This is the statement we need to prove, i.e., the time required for A, B, and C to finish the work together is 5 days.
theorem work_completed_in_5_days (hA : rateA = 1 / 15) (hB : rateB = 1 / 14) (hC : rateC = 1 / 16) :
  (1 / combined_rate) = 5 :=
by
  sorry

end work_completed_in_5_days_l242_242661


namespace square_root_of_9_l242_242380

theorem square_root_of_9 : {x : ℝ // x^2 = 9} = {x : ℝ // x = 3 ∨ x = -3} :=
by
  sorry

end square_root_of_9_l242_242380


namespace largest_angle_of_triangle_l242_242362

theorem largest_angle_of_triangle
  (a b y : ℝ)
  (h1 : a = 60)
  (h2 : b = 70)
  (h3 : a + b + y = 180) :
  max a (max b y) = b :=
by
  sorry

end largest_angle_of_triangle_l242_242362


namespace total_word_count_is_5000_l242_242897

def introduction : ℕ := 450
def conclusion : ℕ := 3 * introduction
def body_sections : ℕ := 4 * 800

def total_word_count : ℕ := introduction + conclusion + body_sections

theorem total_word_count_is_5000 : total_word_count = 5000 := 
by
  -- Lean proof code will go here.
  sorry

end total_word_count_is_5000_l242_242897


namespace largest_n_satisfying_expression_l242_242809

theorem largest_n_satisfying_expression :
  ∃ n < 100000, (n - 3)^5 - n^2 + 10 * n - 30 ≡ 0 [MOD 3] ∧ 
  (∀ m, m < 100000 → (m - 3)^5 - m^2 + 10 * m - 30 ≡ 0 [MOD 3] → m ≤ 99998) := sorry

end largest_n_satisfying_expression_l242_242809


namespace bricks_required_l242_242670

-- Courtyard dimensions in meters
def length_courtyard_m := 23
def width_courtyard_m := 15

-- Brick dimensions in centimeters
def length_brick_cm := 17
def width_brick_cm := 9

-- Conversion from meters to centimeters
def meter_to_cm (m : Int) : Int :=
  m * 100

-- Area of courtyard in square centimeters
def area_courtyard_cm2 : Int :=
  meter_to_cm length_courtyard_m * meter_to_cm width_courtyard_m

-- Area of a single brick in square centimeters
def area_brick_cm2 : Int :=
  length_brick_cm * width_brick_cm

-- Calculate the number of bricks needed, ensuring we round up to the nearest whole number
def total_bricks_needed : Int :=
  (area_courtyard_cm2 + area_brick_cm2 - 1) / area_brick_cm2

-- The theorem stating the total number of bricks needed
theorem bricks_required :
  total_bricks_needed = 22550 := by
  sorry

end bricks_required_l242_242670


namespace expected_value_of_winnings_l242_242847

theorem expected_value_of_winnings (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 8) :
  ((∑ i in finset.range 8, (8 - i) * (1 / 8)) = 3.5) :=
by
  sorry

end expected_value_of_winnings_l242_242847


namespace volume_increase_factor_l242_242241

variable (π : ℝ) (r h : ℝ)

def original_volume : ℝ := π * r^2 * h

def new_volume : ℝ := π * (2 * r)^2 * (3 * h)

theorem volume_increase_factor : new_volume π r h = 12 * original_volume π r h :=
by
  -- Here we would include the proof that new_volume = 12 * original_volume
  sorry

end volume_increase_factor_l242_242241


namespace exists_pow_two_sub_one_divisible_by_odd_l242_242351

theorem exists_pow_two_sub_one_divisible_by_odd {a : ℕ} (h_odd : a % 2 = 1) 
  : ∃ b : ℕ, (2^b - 1) % a = 0 :=
sorry

end exists_pow_two_sub_one_divisible_by_odd_l242_242351


namespace grading_options_count_l242_242331

theorem grading_options_count :
  (4 ^ 15) = 1073741824 :=
by
  sorry

end grading_options_count_l242_242331


namespace solve_for_phi_l242_242308

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x)
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.cos (2 * x - 2 * φ)

theorem solve_for_phi (φ : ℝ) (h₁ : 0 < φ) (h₂ : φ < π / 2)
    (h_min_diff : |x1 - x2| = π / 6)
    (h_condition : |f x1 - g x2 φ| = 4) :
    φ = π / 3 := 
    sorry

end solve_for_phi_l242_242308


namespace wall_height_proof_l242_242668

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

end wall_height_proof_l242_242668


namespace age_twice_in_two_years_l242_242396

-- conditions
def father_age (S : ℕ) : ℕ := S + 24
def present_son_age : ℕ := 22
def present_father_age : ℕ := father_age present_son_age

-- theorem statement
theorem age_twice_in_two_years (S M Y : ℕ) (h1 : S = present_son_age) (h2 : M = present_father_age) : 
  M + 2 = 2 * (S + 2) :=
by
  sorry

end age_twice_in_two_years_l242_242396


namespace maximize_fraction_l242_242618

theorem maximize_fraction (A B C D : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_digits : A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9)
  (h_nonneg : 0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ 0 ≤ D)
  (h_integer : (A + B) % (C + D) = 0) : A + B = 17 :=
sorry

end maximize_fraction_l242_242618


namespace pizza_combination_count_l242_242263

open_locale nat

-- Given condition: the number of different toppings
def num_toppings : ℕ := 8

-- Calculating the total number of one-topping, two-topping, and three-topping pizzas
theorem pizza_combination_count : ((nat.choose num_toppings 1) + (nat.choose num_toppings 2) + (nat.choose num_toppings 3)) = 92 :=
by
  sorry

end pizza_combination_count_l242_242263


namespace find_pairs_l242_242726

noncomputable theory

theorem find_pairs (a b : ℕ) (P Q : ℤ[X]) (h : ∀ (n : ℕ), π (a * n) * Q.eval n = π (b * n) * P.eval n) : a = b :=
sorry

end find_pairs_l242_242726


namespace jen_shooting_game_times_l242_242627

theorem jen_shooting_game_times (x : ℕ) (h1 : 5 * x + 9 = 19) : x = 2 := by
  sorry

end jen_shooting_game_times_l242_242627


namespace binomial_60_3_l242_242904

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l242_242904


namespace battery_current_l242_242111

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l242_242111


namespace inequality_2_pow_ge_n_sq_l242_242145

theorem inequality_2_pow_ge_n_sq (n : ℕ) (hn : n ≠ 3) : 2^n ≥ n^2 :=
sorry

end inequality_2_pow_ge_n_sq_l242_242145


namespace find_current_when_resistance_is_12_l242_242044

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l242_242044


namespace LemonadeCalories_l242_242193

noncomputable def total_calories (lemon_juice sugar water honey : ℕ) (cal_per_100g_lemon_juice cal_per_100g_sugar cal_per_100g_honey : ℕ) : ℝ :=
  (lemon_juice / 100) * cal_per_100g_lemon_juice +
  (sugar / 100) * cal_per_100g_sugar +
  (honey / 100) * cal_per_100g_honey

noncomputable def calories_in_250g (total_calories : ℝ) (total_weight : ℕ) : ℝ :=
  (total_calories / total_weight) * 250

theorem LemonadeCalories :
  let lemon_juice := 150
  let sugar := 200
  let water := 300
  let honey := 50
  let cal_per_100g_lemon_juice := 25
  let cal_per_100g_sugar := 386
  let cal_per_100g_honey := 64
  let total_weight := lemon_juice + sugar + water + honey
  let total_cal := total_calories lemon_juice sugar water honey cal_per_100g_lemon_juice cal_per_100g_sugar cal_per_100g_honey
  calories_in_250g total_cal total_weight = 301 :=
by
  sorry

end LemonadeCalories_l242_242193


namespace battery_current_l242_242056

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l242_242056


namespace feed_days_l242_242463

theorem feed_days (morning_food evening_food total_food : ℕ) (h1 : morning_food = 1) (h2 : evening_food = 1) (h3 : total_food = 32)
: (total_food / (morning_food + evening_food)) = 16 := by
  sorry

end feed_days_l242_242463


namespace problem_solution_l242_242976

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem problem_solution : a^10 + b^10 = 123 := sorry

end problem_solution_l242_242976


namespace battery_current_l242_242082

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l242_242082


namespace binom_30_3_eq_4060_l242_242694

theorem binom_30_3_eq_4060 : nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l242_242694


namespace find_current_l242_242106

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l242_242106


namespace current_when_resistance_12_l242_242101

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l242_242101


namespace current_at_resistance_12_l242_242038

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l242_242038


namespace gunny_bag_can_hold_packets_l242_242674

theorem gunny_bag_can_hold_packets :
  let ton_to_kg := 1000
  let max_capacity_tons := 13
  let pound_to_kg := 0.453592
  let ounce_to_g := 28.3495
  let kilo_to_g := 1000
  let wheat_packet_pounds := 16
  let wheat_packet_ounces := 4
  let max_capacity_kg := max_capacity_tons * ton_to_kg
  let wheat_packet_kg := wheat_packet_pounds * pound_to_kg + (wheat_packet_ounces * ounce_to_g) / kilo_to_g
  max_capacity_kg / wheat_packet_kg >= 1763 := 
by
  sorry

end gunny_bag_can_hold_packets_l242_242674


namespace percentage_increase_l242_242164

variable (x y p : ℝ)

theorem percentage_increase (h : x = y + (p / 100) * y) : p = 100 * ((x - y) / y) := 
by 
  sorry

end percentage_increase_l242_242164


namespace expected_value_of_8_sided_die_l242_242841

-- Define the expected value function for the given win calculation rule.
def expected_value := (1/8 : ℚ) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

-- Formal statement of the proof problem.
theorem expected_value_of_8_sided_die : 
  expected_value = 3.50 :=
by
  sorry

end expected_value_of_8_sided_die_l242_242841


namespace remainder_13_plus_x_l242_242468

theorem remainder_13_plus_x (x : ℕ) (h1 : 7 * x % 31 = 1) : (13 + x) % 31 = 22 := 
by
  sorry

end remainder_13_plus_x_l242_242468


namespace binomial_60_3_l242_242905

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l242_242905


namespace remainder_when_multiplied_by_three_and_divided_by_eighth_prime_l242_242378

def first_seven_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def sum_first_seven_primes : ℕ := first_seven_primes.sum

def eighth_prime : ℕ := 19

theorem remainder_when_multiplied_by_three_and_divided_by_eighth_prime :
  ((sum_first_seven_primes * 3) % eighth_prime = 3) :=
by
  sorry

end remainder_when_multiplied_by_three_and_divided_by_eighth_prime_l242_242378


namespace area_not_covered_by_small_squares_l242_242541

def large_square_side_length : ℕ := 10
def small_square_side_length : ℕ := 4
def large_square_area : ℕ := large_square_side_length ^ 2
def small_square_area : ℕ := small_square_side_length ^ 2
def uncovered_area : ℕ := large_square_area - small_square_area

theorem area_not_covered_by_small_squares :
  uncovered_area = 84 := by
  sorry

end area_not_covered_by_small_squares_l242_242541


namespace selling_price_40_percent_profit_l242_242224

variable (C L : ℝ)

-- Condition: the profit earned by selling at $832 is equal to the loss incurred when selling at some price "L".
axiom eq_profit_loss : 832 - C = C - L

-- Condition: the desired profit price for a 40% profit on the cost price is $896.
axiom forty_percent_profit : 1.40 * C = 896

-- Theorem: the selling price for making a 40% profit is $896.
theorem selling_price_40_percent_profit : 1.40 * C = 896 :=
by
  sorry

end selling_price_40_percent_profit_l242_242224


namespace inverse_proportion_symmetry_l242_242151

theorem inverse_proportion_symmetry (a b : ℝ) :
  (b = - 6 / (-a)) → (-b = - 6 / a) :=
by
  intro h
  sorry

end inverse_proportion_symmetry_l242_242151


namespace football_game_spectators_l242_242452

theorem football_game_spectators (total_wristbands wristbands_per_person : ℕ)
  (h1 : total_wristbands = 250) (h2 : wristbands_per_person = 2) : 
  total_wristbands / wristbands_per_person = 125 :=
by
  sorry

end football_game_spectators_l242_242452


namespace percent_increase_quarter_l242_242501

-- Define the profit changes over each month
def profit_march (P : ℝ) := P
def profit_april (P : ℝ) := 1.40 * P
def profit_may (P : ℝ) := 1.12 * P
def profit_june (P : ℝ) := 1.68 * P

-- Starting Lean theorem statement
theorem percent_increase_quarter (P : ℝ) (hP : P > 0) :
  ((profit_june P - profit_march P) / profit_march P) * 100 = 68 :=
  sorry

end percent_increase_quarter_l242_242501


namespace expression_evaluation_l242_242289

noncomputable def evaluate_expression (a b c : ℚ) : ℚ :=
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7)

theorem expression_evaluation : 
  ∀ (a b c : ℚ), c = b - 11 → b = a + 3 → a = 5 → 
  (a + 2) ≠ 0 → (b - 3) ≠ 0 → (c + 7) ≠ 0 → 
  evaluate_expression a b c = 72 / 35 :=
by
  intros a b c hc hb ha h1 h2 h3
  rw [ha, hb, hc, evaluate_expression]
  -- The proof is not required.
  sorry

end expression_evaluation_l242_242289


namespace trailing_zeros_of_9_pow_999_plus_1_l242_242175

theorem trailing_zeros_of_9_pow_999_plus_1 :
  ∃ n : ℕ, n = 999 ∧ (9^n + 1) % 10 = 0 ∧ (9^n + 1) % 100 ≠ 0 :=
by
  sorry

end trailing_zeros_of_9_pow_999_plus_1_l242_242175


namespace intersection_P_Q_l242_242285

def P : Set ℝ := { x | x > 1 }
def Q : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_P_Q : P ∩ Q = { x | 1 < x ∧ x < 2 } :=
by
  sorry

end intersection_P_Q_l242_242285


namespace inconsistent_intercepts_l242_242748

-- Define the ellipse equation
def ellipse (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / m + y^2 / 4 = 1

-- Define the line equations
def line1 (x k : ℝ) : ℝ := k * x + 1
def line2 (x : ℝ) (k : ℝ) : ℝ := - k * x - 2

-- Disc calculation for line1
def disc1 (m k : ℝ) : ℝ :=
  let a := 4 + m * k^2
  let b := 2 * m * k
  let c := -3 * m
  b^2 - 4 * a * c

-- Disc calculation for line2
def disc2 (m k : ℝ) : ℝ :=
  let bb := 4 * m * k
  bb^2

-- Statement of the problem
theorem inconsistent_intercepts (m k : ℝ) (hm_pos : 0 < m) :
  disc1 m k ≠ disc2 m k :=
by
  sorry

end inconsistent_intercepts_l242_242748


namespace sum_of_powers_sequence_l242_242977

theorem sum_of_powers_sequence (a b : ℝ) 
  (h₁ : a + b = 1)
  (h₂ : a^2 + b^2 = 3)
  (h₃ : a^3 + b^3 = 4)
  (h₄ : a^4 + b^4 = 7)
  (h₅ : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 :=
sorry

end sum_of_powers_sequence_l242_242977


namespace minimum_value_of_m_minus_n_l242_242165

def f (x : ℝ) : ℝ := (x - 1) ^ 2

theorem minimum_value_of_m_minus_n 
  (f_even : ∀ x : ℝ, f x = f (-x))
  (condition1 : n ≤ f (-2))
  (condition2 : n ≤ f (-1 / 2))
  (condition3 : f (-2) ≤ m)
  (condition4 : f (-1 / 2) ≤ m)
  : ∃ n m, m - n = 1 :=
by
  sorry

end minimum_value_of_m_minus_n_l242_242165


namespace Dongdong_test_score_l242_242251

theorem Dongdong_test_score (a b c : ℕ) (h1 : a + b + c = 280) : a ≥ 94 ∨ b ≥ 94 ∨ c ≥ 94 :=
by
  sorry

end Dongdong_test_score_l242_242251


namespace rectangle_square_area_ratio_l242_242218

theorem rectangle_square_area_ratio (s : ℝ) (hs : s > 0) :
  let area_square := s^2 in
  let area_rectangle := (1.2 * s) * (0.8 * s) in
  area_rectangle / area_square = 24 / 25 :=
by
  sorry

end rectangle_square_area_ratio_l242_242218


namespace mean_score_74_l242_242728

theorem mean_score_74 
  (M SD : ℝ)
  (h1 : 58 = M - 2 * SD)
  (h2 : 98 = M + 3 * SD) : 
  M = 74 :=
by
  sorry

end mean_score_74_l242_242728


namespace minimum_energy_H1_l242_242890

-- Define the given conditions
def energyEfficiencyMin : ℝ := 0.1
def energyRequiredH6 : ℝ := 10 -- Energy in KJ
def energyLevels : Nat := 5 -- Number of energy levels from H1 to H6

-- Define the theorem to prove the minimum energy required from H1
theorem minimum_energy_H1 : (10 ^ energyLevels : ℝ) = 1000000 :=
by
  -- Placeholder for actual proof
  sorry

end minimum_energy_H1_l242_242890


namespace number_of_possible_measures_l242_242638

theorem number_of_possible_measures (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
sorry

end number_of_possible_measures_l242_242638


namespace angle_D_is_20_degrees_l242_242964

theorem angle_D_is_20_degrees (A B C D : ℝ) 
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 40)
  (h4 : B + C = 160) : D = 20 :=
by
  sorry

end angle_D_is_20_degrees_l242_242964


namespace Caroline_lost_4_pairs_of_socks_l242_242688

theorem Caroline_lost_4_pairs_of_socks 
  (initial_pairs : ℕ) (pairs_donated_fraction : ℚ)
  (new_pairs_purchased : ℕ) (new_pairs_gifted : ℕ)
  (final_pairs : ℕ) (L : ℕ) :
  initial_pairs = 40 →
  pairs_donated_fraction = 2/3 →
  new_pairs_purchased = 10 →
  new_pairs_gifted = 3 →
  final_pairs = 25 →
  (initial_pairs - L) * (1 - pairs_donated_fraction) + new_pairs_purchased + new_pairs_gifted = final_pairs →
  L = 4 :=
by {
  sorry
}

end Caroline_lost_4_pairs_of_socks_l242_242688


namespace binomial_60_3_l242_242903

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l242_242903


namespace sum_lent_out_l242_242127

theorem sum_lent_out (P R : ℝ) (h1 : 720 = P + (P * R * 2) / 100) (h2 : 1020 = P + (P * R * 7) / 100) : P = 600 := by
  sorry

end sum_lent_out_l242_242127


namespace intersection_eq_l242_242755

def setA : Set ℝ := { x | abs (x - 3) < 2 }
def setB : Set ℝ := { x | (x - 4) / x ≥ 0 }

theorem intersection_eq : setA ∩ setB = { x | 4 ≤ x ∧ x < 5 } :=
by 
  sorry

end intersection_eq_l242_242755


namespace thomas_blocks_total_l242_242370

theorem thomas_blocks_total 
  (first_stack : ℕ)
  (second_stack : ℕ)
  (third_stack : ℕ)
  (fourth_stack : ℕ)
  (fifth_stack : ℕ) 
  (h1 : first_stack = 7)
  (h2 : second_stack = first_stack + 3)
  (h3 : third_stack = second_stack - 6)
  (h4 : fourth_stack = third_stack + 10)
  (h5 : fifth_stack = 2 * second_stack) :
  first_stack + second_stack + third_stack + fourth_stack + fifth_stack = 55 :=
by
  sorry

end thomas_blocks_total_l242_242370


namespace find_multiplier_l242_242824

theorem find_multiplier (n k : ℤ) (h1 : n + 4 = 15) (h2 : 3 * n = k * (n + 4) + 3) : k = 2 :=
  sorry

end find_multiplier_l242_242824


namespace squared_distance_focus_product_tangents_l242_242734

variable {a b : ℝ}
variable {x0 y0 : ℝ}
variable {P Q R F : ℝ × ℝ}

-- Conditions
def is_ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def outside_ellipse (x0 y0 : ℝ) (a b : ℝ) : Prop :=
  (x0^2 / a^2) + (y0^2 / b^2) > 1

-- Question (statement we need to prove)
theorem squared_distance_focus_product_tangents
  (h_ellipse : is_ellipse Q.1 Q.2 a b)
  (h_ellipse' : is_ellipse R.1 R.2 a b)
  (h_outside : outside_ellipse x0 y0 a b)
  (h_a_greater_b : a > b) :
  ‖P - F‖^2 > ‖Q - F‖ * ‖R - F‖ := sorry

end squared_distance_focus_product_tangents_l242_242734


namespace log_sum_l242_242282

theorem log_sum : Real.logb 2 1 + Real.logb 3 9 = 2 := by
  sorry

end log_sum_l242_242282


namespace nancy_packs_l242_242974

theorem nancy_packs (total_bars packs_bars : ℕ) (h_total : total_bars = 30) (h_packs : packs_bars = 5) :
  total_bars / packs_bars = 6 :=
by
  sorry

end nancy_packs_l242_242974


namespace simplify_and_evaluate_expression_l242_242477

theorem simplify_and_evaluate_expression (a : ℂ) (h: a^2 + 4 * a + 1 = 0) :
  ( ( (a + 2) / (a^2 - 2 * a) + 8 / (4 - a^2) ) / ( (a^2 - 4) / a ) ) = 1 / 3 := by
  sorry

end simplify_and_evaluate_expression_l242_242477


namespace current_value_l242_242089

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l242_242089


namespace algorithm_comparable_to_euclidean_l242_242632

-- Define the conditions
def ancient_mathematics_world_leading : Prop := 
  True -- Placeholder representing the historical condition

def song_yuan_algorithm : Prop :=
  True -- Placeholder representing the algorithmic condition

-- The main theorem representing the problem statement
theorem algorithm_comparable_to_euclidean :
  ancient_mathematics_world_leading → song_yuan_algorithm → 
  True :=  -- Placeholder representing that the algorithm is the method of successive subtraction
by 
  intro h1 h2 
  sorry

end algorithm_comparable_to_euclidean_l242_242632


namespace pool_ratio_three_to_one_l242_242472

theorem pool_ratio_three_to_one (P : ℕ) (B B' : ℕ) (k : ℕ) :
  (P = 5 * B + 2) → (k * P = 5 * B' + 1) → k = 3 :=
by
  intros h1 h2
  sorry

end pool_ratio_three_to_one_l242_242472


namespace num_perfect_squares_mul_36_lt_10pow8_l242_242585

theorem num_perfect_squares_mul_36_lt_10pow8 : 
  ∃(n : ℕ), n = 1666 ∧ 
  ∀ (N : ℕ), (1 ≤ N) → (N^2 < 10^8) → (N^2 % 36 = 0) → 
  (N ≤ 9996 ∧ N % 6 = 0) :=
by
  sorry

end num_perfect_squares_mul_36_lt_10pow8_l242_242585


namespace smallest_n_for_divisibility_by_ten_million_l242_242616

theorem smallest_n_for_divisibility_by_ten_million 
  (a₁ a₂ : ℝ) 
  (a₁_eq : a₁ = 5 / 6) 
  (a₂_eq : a₂ = 30) 
  (n : ℕ) 
  (T : ℕ → ℝ) 
  (T_def : ∀ (k : ℕ), T k = a₁ * (36 ^ (k - 1))) :
  (∃ n, T n = T 9 ∧ (∃ m : ℤ, T n = m * 10^7)) := 
sorry

end smallest_n_for_divisibility_by_ten_million_l242_242616


namespace probability_first_two_heads_l242_242237

-- The probability of getting heads in a single flip of a fair coin
def probability_heads_single_flip : ℚ := 1 / 2

-- Independence of coin flips
def independent_flips {α : Type} (p : α → Prop) := ∀ a b : α, a ≠ b → p a ∧ p b

-- The event of getting heads on a coin flip
def heads_event : Prop := true

-- Problem statement: The probability that the first two flips are both heads
theorem probability_first_two_heads : probability_heads_single_flip * probability_heads_single_flip = 1 / 4 :=
by
  sorry

end probability_first_two_heads_l242_242237


namespace apple_cost_calculation_l242_242886

theorem apple_cost_calculation
    (original_price : ℝ)
    (price_raise : ℝ)
    (amount_per_person : ℝ)
    (num_people : ℝ) :
  original_price = 1.6 →
  price_raise = 0.25 →
  amount_per_person = 2 →
  num_people = 4 →
  (num_people * amount_per_person * (original_price * (1 + price_raise))) = 16 :=
by
  -- insert the mathematical proof steps/cardinality here
  sorry

end apple_cost_calculation_l242_242886


namespace x_squared_y_cubed_eq_200_l242_242162

theorem x_squared_y_cubed_eq_200 (x y : ℕ) (h : 2^x * 9^y = 200) : x^2 * y^3 = 200 := by
  sorry

end x_squared_y_cubed_eq_200_l242_242162


namespace log_expression_is_zero_l242_242141

noncomputable def log_expr : ℝ := (Real.logb 2 3 + Real.logb 2 27) * (Real.logb 4 4 + Real.logb 4 (1/4))

theorem log_expression_is_zero : log_expr = 0 :=
by
  sorry

end log_expression_is_zero_l242_242141


namespace increase_by_50_percent_l242_242389

def original : ℕ := 350
def increase_percent : ℕ := 50
def increased_number : ℕ := original * increase_percent / 100
def final_number : ℕ := original + increased_number

theorem increase_by_50_percent : final_number = 525 := 
by
  sorry

end increase_by_50_percent_l242_242389


namespace train_length_l242_242542

theorem train_length (V L : ℝ) (h1 : L = V * 18) (h2 : L + 550 = V * 51) : L = 300 := sorry

end train_length_l242_242542


namespace number_of_correct_statements_l242_242923

open Real

variables {f : ℝ → ℝ} {a b : ℝ} (h1 : a < b) (h2 : continuous f) (h3 : ∀ x, continuous (deriv f x))
  (h4 : deriv f a > 0) (h5 : deriv f b < 0)

theorem number_of_correct_statements : 2 =
          (if ∃ x ∈ set.Icc a b, f x = 0 then 1 else 0) +
          (if ∃ x ∈ set.Icc a b, f x > f b then 1 else 0) +
          (if ∀ x ∈ set.Icc a b, f x ≥ f a then 1 else 0) +
          (if ∃ x ∈ set.Icc a b, f a - f b > deriv f x * (a - b) then 1 else 0) := sorry

end number_of_correct_statements_l242_242923


namespace tax_collection_amount_l242_242721

theorem tax_collection_amount (paid_tax : ℝ) (willam_percentage : ℝ) (total_collected : ℝ) (h_paid: paid_tax = 480) (h_percentage: willam_percentage = 0.3125) :
    total_collected = 1536 :=
by
  sorry

end tax_collection_amount_l242_242721


namespace marbles_each_friend_gets_l242_242320

-- Definitions for the conditions
def total_marbles : ℕ := 100
def marbles_kept : ℕ := 20
def number_of_friends : ℕ := 5

-- The math proof problem
theorem marbles_each_friend_gets :
  (total_marbles - marbles_kept) / number_of_friends = 16 :=
by
  -- We include the proof steps within a by notation but stop at sorry for automated completion skipping proof steps
  sorry

end marbles_each_friend_gets_l242_242320


namespace problem_statement_l242_242356

noncomputable def r (C: ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def A (r: ℝ) : ℝ := Real.pi * r^2

noncomputable def combined_area_difference (C1 C2 C3: ℝ) : ℝ :=
  let r1 := r C1
  let r2 := r C2
  let r3 := r C3
  let A1 := A r1
  let A2 := A r2
  let A3 := A r3
  (A3 - A1) - A2

theorem problem_statement : combined_area_difference 528 704 880 = -9.76 :=
by
  sorry

end problem_statement_l242_242356


namespace hiring_manager_acceptance_l242_242483

theorem hiring_manager_acceptance :
  let average_age := 31
  let std_dev := 9
  let max_diff_ages := 19
  let k := max_diff_ages / (2 * std_dev)
  k = 19 / 18 :=
by
  let average_age := 31
  let std_dev := 9
  let max_diff_ages := 19
  let k := max_diff_ages / (2 * std_dev)
  show k = 19 / 18
  sorry

end hiring_manager_acceptance_l242_242483


namespace problem_solution_l242_242480

variable {x y z : ℝ}

/-- Suppose that x, y, and z are three positive numbers that satisfy the given conditions.
    Prove that z + 1/y = 13/77. --/
theorem problem_solution (h1 : x * y * z = 1)
                         (h2 : x + 1 / z = 8)
                         (h3 : y + 1 / x = 29) :
  z + 1 / y = 13 / 77 := 
  sorry

end problem_solution_l242_242480


namespace bullet_speed_difference_l242_242652

theorem bullet_speed_difference (speed_horse speed_bullet : ℕ) 
    (h_horse : speed_horse = 20) (h_bullet : speed_bullet = 400) :
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    speed_same_direction - speed_opposite_direction = 40 :=
    by
    -- Define the speeds in terms of the given conditions.
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    -- State the equality to prove.
    show speed_same_direction - speed_opposite_direction = 40;
    -- Proof (skipped here).
    -- sorry is used to denote where the formal proof steps would go.
    sorry

end bullet_speed_difference_l242_242652


namespace happy_numbers_l242_242647

theorem happy_numbers (n : ℕ) (h1 : n < 1000) 
(h2 : 7 ∣ n^2) (h3 : 8 ∣ n^2) (h4 : 9 ∣ n^2) (h5 : 10 ∣ n^2) : 
n = 420 ∨ n = 840 :=
sorry

end happy_numbers_l242_242647


namespace dot_product_necessity_l242_242960

variables (a b : ℝ → ℝ → ℝ)

def dot_product (a b : ℝ → ℝ → ℝ) (x y : ℝ) : ℝ :=
  a x y * b x y

def angle_is_acute (a b : ℝ → ℝ → ℝ) (x y : ℝ) : Prop :=
  0 < a x y

theorem dot_product_necessity (a b : ℝ → ℝ → ℝ) (x y : ℝ) :
  dot_product a b x y > 0 ↔ angle_is_acute a b x y :=
sorry

end dot_product_necessity_l242_242960


namespace current_at_resistance_12_l242_242033

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l242_242033


namespace domain_correct_l242_242286

noncomputable def domain_function (x : ℝ) : Prop :=
  (4 * x - 3 > 0) ∧ (Real.log (4 * x - 3) / Real.log 0.5 > 0)

theorem domain_correct : {x : ℝ | domain_function x} = {x : ℝ | (3 / 4 : ℝ) < x ∧ x < 1} :=
by
  sorry

end domain_correct_l242_242286


namespace volume_increase_factor_l242_242242

variable (π : ℝ) (r h : ℝ)

def original_volume : ℝ := π * r^2 * h

def new_volume : ℝ := π * (2 * r)^2 * (3 * h)

theorem volume_increase_factor : new_volume π r h = 12 * original_volume π r h :=
by
  -- Here we would include the proof that new_volume = 12 * original_volume
  sorry

end volume_increase_factor_l242_242242


namespace distance_between_circle_centers_l242_242191

-- Define the given side lengths of the triangle
def DE : ℝ := 12
def DF : ℝ := 15
def EF : ℝ := 9

-- Define the problem and assertion
theorem distance_between_circle_centers :
  ∃ d : ℝ, d = 12 * Real.sqrt 13 :=
sorry

end distance_between_circle_centers_l242_242191


namespace geom_seq_decreasing_l242_242572

variable {a : ℕ → ℝ}
variable {a₁ q : ℝ}

theorem geom_seq_decreasing (h : ∀ n, a n = a₁ * q^n) (h₀ : a₁ * (q - 1) < 0) (h₁ : q > 0) :
  ∀ n, a (n + 1) < a n := 
sorry

end geom_seq_decreasing_l242_242572


namespace current_value_l242_242019

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l242_242019


namespace option_A_option_B_option_C_option_D_l242_242767

namespace Inequalities

theorem option_A (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  a + (1/a) > b + (1/b) :=
sorry

theorem option_B (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m > n) :
  (m + 1) / (n + 1) < m / n :=
sorry

theorem option_C (c a b : ℝ) (hc : c > 0) (ha : a > 0) (hb : b > 0) (hca : c > a) (hab : a > b) :
  a / (c - a) > b / (c - b) :=
sorry

theorem option_D (a b : ℝ) (ha : a > -1) (hb : b > -1) (hab : a ≥ b) :
  a / (a + 1) ≥ b / (b + 1) :=
sorry

end Inequalities

end option_A_option_B_option_C_option_D_l242_242767


namespace larry_substitution_l242_242781

theorem larry_substitution (a b c d e : ℤ)
  (h_a : a = 2)
  (h_b : b = 5)
  (h_c : c = 3)
  (h_d : d = 4)
  (h_expr1 : a + b - c - d * e = 4 - 4 * e)
  (h_expr2 : a + (b - (c - (d * e))) = 4 + 4 * e) :
  e = 0 :=
by
  sorry

end larry_substitution_l242_242781


namespace days_to_clear_land_l242_242512

-- Definitions of all the conditions
def length_of_land := 200
def width_of_land := 900
def area_cleared_by_one_rabbit_per_day_square_yards := 10
def number_of_rabbits := 100
def conversion_square_yards_to_square_feet := 9
def total_area_of_land := length_of_land * width_of_land
def area_cleared_by_one_rabbit_per_day_square_feet := area_cleared_by_one_rabbit_per_day_square_yards * conversion_square_yards_to_square_feet
def area_cleared_by_all_rabbits_per_day := number_of_rabbits * area_cleared_by_one_rabbit_per_day_square_feet

-- Theorem to prove the number of days required to clear the land
theorem days_to_clear_land :
  total_area_of_land / area_cleared_by_all_rabbits_per_day = 20 := by
  sorry

end days_to_clear_land_l242_242512


namespace battery_current_at_given_resistance_l242_242098

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l242_242098


namespace prime_implies_n_eq_3k_l242_242625

theorem prime_implies_n_eq_3k (n : ℕ) (p : ℕ) (k : ℕ) (h_pos : k > 0)
  (h_prime : Prime p) (h_eq : p = 1 + 2^n + 4^n) :
  ∃ k : ℕ, k > 0 ∧ n = 3^k :=
by
  sorry

end prime_implies_n_eq_3k_l242_242625


namespace sum_of_three_consecutive_odds_is_69_l242_242507

-- Definition for the smallest of three consecutive odd numbers
def smallest_consecutive_odd := 21

-- Define the three consecutive odd numbers based on the smallest one
def first_consecutive_odd := smallest_consecutive_odd
def second_consecutive_odd := smallest_consecutive_odd + 2
def third_consecutive_odd := smallest_consecutive_odd + 4

-- Calculate the sum of these three consecutive odd numbers
def sum_consecutive_odds := first_consecutive_odd + second_consecutive_odd + third_consecutive_odd

-- Theorem statement that the sum of these three consecutive odd numbers is 69
theorem sum_of_three_consecutive_odds_is_69 : 
  sum_consecutive_odds = 69 := by
    sorry

end sum_of_three_consecutive_odds_is_69_l242_242507


namespace find_x_l242_242589

theorem find_x (x : ℤ) (h : (2 * x + 7) / 5 = 22) : x = 103 / 2 :=
by
  sorry

end find_x_l242_242589


namespace Joel_contributed_22_toys_l242_242183

/-
Define the given conditions as separate variables and statements in Lean:
1. Toys collected from friends.
2. Total toys donated.
3. Relationship between Joel's and his sister's toys.
4. Prove that Joel donated 22 toys.
-/

theorem Joel_contributed_22_toys (S : ℕ) (toys_from_friends : ℕ) (total_toys : ℕ) (sisters_toys : ℕ) 
  (h1 : toys_from_friends = 18 + 42 + 2 + 13)
  (h2 : total_toys = 108)
  (h3 : S + 2 * S = total_toys - toys_from_friends)
  (h4 : sisters_toys = S) :
  2 * S = 22 :=
  sorry

end Joel_contributed_22_toys_l242_242183


namespace train_speed_l242_242129

theorem train_speed (len_train len_bridge time : ℝ)
  (h1 : len_train = 100)
  (h2 : len_bridge = 180)
  (h3 : time = 27.997760179185665) :
  (len_train + len_bridge) / time * 3.6 = 36 :=
by
  sorry

end train_speed_l242_242129


namespace initial_price_of_phone_l242_242276

theorem initial_price_of_phone (P : ℝ) (h : 0.20 * P = 480) : P = 2400 :=
sorry

end initial_price_of_phone_l242_242276


namespace bullet_speed_difference_l242_242653

theorem bullet_speed_difference (speed_horse speed_bullet : ℕ) 
    (h_horse : speed_horse = 20) (h_bullet : speed_bullet = 400) :
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    speed_same_direction - speed_opposite_direction = 40 :=
    by
    -- Define the speeds in terms of the given conditions.
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    -- State the equality to prove.
    show speed_same_direction - speed_opposite_direction = 40;
    -- Proof (skipped here).
    -- sorry is used to denote where the formal proof steps would go.
    sorry

end bullet_speed_difference_l242_242653


namespace same_graph_iff_same_function_D_l242_242244

theorem same_graph_iff_same_function_D :
  ∀ x : ℝ, (|x| = if x ≥ 0 then x else -x) :=
by
  intro x
  sorry

end same_graph_iff_same_function_D_l242_242244


namespace part_I_part_II_l242_242450

-- Define the triangle and sides
structure Triangle :=
  (A B C : ℝ)   -- angles in the triangle
  (a b c : ℝ)   -- sides opposite to respective angles

-- Express given conditions in the problem
def conditions (T: Triangle) : Prop :=
  2 * (1 / (Real.tan T.A) + 1 / (Real.tan T.C)) = 1 / (Real.sin T.A) + 1 / (Real.sin T.C)

-- First theorem statement
theorem part_I (T : Triangle) : conditions T → (T.a + T.c = 2 * T.b) :=
sorry

-- Second theorem statement
theorem part_II (T : Triangle) : conditions T → (T.B ≤ Real.pi / 3) :=
sorry

end part_I_part_II_l242_242450


namespace binom_60_3_l242_242898

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l242_242898


namespace expected_value_is_350_l242_242855

noncomputable def expected_value_of_winnings : ℚ :=
  ((1 / 8) * (8 - 1) + (1 / 8) * (8 - 2) + (1 / 8) * (8 - 3) + (1 / 8) * (8 - 4) +
  (1 / 8) * (8 - 5) + (1 / 8) * (8 - 6) + (1 / 8) * (8 - 7) + (1 / 8) * (8 - 8))

theorem expected_value_is_350 :
  expected_value_of_winnings = 3.50 := by
  sorry

end expected_value_is_350_l242_242855


namespace brad_must_make_5_trips_l242_242131

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r ^ 2 * h

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ :=
  Real.pi * r ^ 2 * h

theorem brad_must_make_5_trips (r_barrel h_barrel r_bucket h_bucket : ℝ)
    (h1 : r_barrel = 10) (h2 : h_barrel = 15) (h3 : r_bucket = 10) (h4 : h_bucket = 10) :
    let trips := volume_of_cylinder r_barrel h_barrel / volume_of_cone r_bucket h_bucket
    let trips_needed := Int.ceil trips
    trips_needed = 5 := 
by
  sorry

end brad_must_make_5_trips_l242_242131


namespace equal_donations_amount_l242_242119

def raffle_tickets_sold := 25
def cost_per_ticket := 2
def total_raised := 100
def single_donation := 20
def amount_equal_donations (D : ℕ) : Prop := 2 * D + single_donation = total_raised - (raffle_tickets_sold * cost_per_ticket)

theorem equal_donations_amount (D : ℕ) (h : amount_equal_donations D) : D = 15 :=
  sorry

end equal_donations_amount_l242_242119


namespace harold_monthly_income_l242_242439

variable (M : ℕ)

def rent : ℕ := 700
def car_payment : ℕ := 300
def utilities : ℕ := car_payment / 2
def groceries : ℕ := 50

def total_expenses : ℕ := rent + car_payment + utilities + groceries
def remaining_money_after_expenses : ℕ := M - total_expenses
def retirement_saving_target : ℕ := 650
def required_remaining_money_pre_saving : ℕ := 2 * retirement_saving_target

theorem harold_monthly_income :
  remaining_money_after_expenses = required_remaining_money_pre_saving → M = 2500 :=
by
  sorry

end harold_monthly_income_l242_242439


namespace janine_total_pages_l242_242459

-- Define the conditions
def books_last_month : ℕ := 5
def books_this_month : ℕ := 2 * books_last_month
def books_per_page : ℕ := 10

-- Define the total number of pages she read in two months
def total_pages : ℕ :=
  let total_books := books_last_month + books_this_month
  total_books * books_per_page

-- State the theorem to be proven
theorem janine_total_pages : total_pages = 150 :=
by
  sorry

end janine_total_pages_l242_242459


namespace min_students_same_score_l242_242965

open Nat

theorem min_students_same_score :
  ∃ s : ℕ, ∃ (H : s ∈ (Finset.range 25).image (λ n, 6 + 4 * n - 1 * (6 - n))),
    (Finset.filter (λ student_score, student_score = s) (Finset.range 51).image (λ student, 
    6 + 4 * (student % 7) - 1 * ((6 - student % 7)))) .card ≥ 3 := by
  sorry

end min_students_same_score_l242_242965


namespace current_value_l242_242069

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l242_242069


namespace battery_current_when_resistance_12_l242_242078

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l242_242078


namespace jacob_writing_speed_ratio_l242_242769

theorem jacob_writing_speed_ratio (N : ℕ) (J : ℕ) (hN : N = 25) (h1 : J + N = 75) : J / N = 2 :=
by {
  sorry
}

end jacob_writing_speed_ratio_l242_242769


namespace slices_per_pizza_l242_242342

-- Definitions based on the conditions
def num_pizzas : Nat := 3
def total_cost : Nat := 72
def cost_per_5_slices : Nat := 10

-- To find the number of slices per pizza
theorem slices_per_pizza (num_pizzas : Nat) (total_cost : Nat) (cost_per_5_slices : Nat): 
  (total_cost / num_pizzas) / (cost_per_5_slices / 5) = 12 :=
by
  sorry

end slices_per_pizza_l242_242342


namespace current_at_resistance_12_l242_242058

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l242_242058


namespace current_value_l242_242016

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l242_242016


namespace problem_1_problem_2_l242_242555
-- Import the entire Mathlib library.

-- Problem (1)
theorem problem_1 (x y : ℝ) (h1 : |x - 3 * y| < 1 / 2) (h2 : |x + 2 * y| < 1 / 6) : |x| < 3 / 10 :=
sorry

-- Problem (2)
theorem problem_2 (x y : ℝ) : x^4 + 16 * y^4 ≥ 2 * x^3 * y + 8 * x * y^3 :=
sorry

end problem_1_problem_2_l242_242555


namespace some_number_value_l242_242246

theorem some_number_value (some_number : ℝ) (h : (some_number * 14) / 100 = 0.045388) :
  some_number = 0.3242 :=
sorry

end some_number_value_l242_242246


namespace tim_score_l242_242274

theorem tim_score :
  let single_line_points := 1000
  let tetris_points := 8 * single_line_points
  let singles_scored := 6
  let tetrises_scored := 4
  in singles_scored * single_line_points + tetrises_scored * tetris_points = 38000 := by
  sorry

end tim_score_l242_242274


namespace parabola_equation_l242_242988

theorem parabola_equation (a b c d e f: ℤ) (ha: a = 2) (hb: b = 0) (hc: c = 0) (hd: d = -16) (he: e = -1) (hf: f = 32) :
  ∃ x y : ℝ, 2 * x ^ 2 - 16 * x + 32 - y = 0 ∧ gcd (abs a) (gcd (abs b) (gcd (abs c) (gcd (abs d) (gcd (abs e) (abs f))))) = 1 :=
by
  sorry

end parabola_equation_l242_242988


namespace bullet_speed_difference_l242_242658

def bullet_speed_in_same_direction (v_h v_b : ℝ) : ℝ :=
  v_b + v_h

def bullet_speed_in_opposite_direction (v_h v_b : ℝ) : ℝ :=
  v_b - v_h

theorem bullet_speed_difference (v_h v_b : ℝ) (h_h : v_h = 20) (h_b : v_b = 400) :
  bullet_speed_in_same_direction v_h v_b - bullet_speed_in_opposite_direction v_h v_b = 40 :=
by
  rw [h_h, h_b]
  sorry

end bullet_speed_difference_l242_242658


namespace geometric_seq_common_ratio_l242_242599

theorem geometric_seq_common_ratio 
  (a : ℕ → ℝ) -- a_n is the sequence
  (S : ℕ → ℝ) -- S_n is the partial sum of the sequence
  (h1 : a 3 = 2 * S 2 + 1) -- condition a_3 = 2S_2 + 1
  (h2 : a 4 = 2 * S 3 + 1) -- condition a_4 = 2S_3 + 1
  (h3 : S 2 = a 1 / (1 / q) * (1 - q^3) / (1 - q)) -- sum of first 2 terms
  (h4 : S 3 = a 1 / (1 / q) * (1 - q^4) / (1 - q)) -- sum of first 3 terms
  : q = 3 := -- conclusion
by sorry

end geometric_seq_common_ratio_l242_242599


namespace efficiency_ratio_l242_242260

theorem efficiency_ratio (r : ℚ) (work_B : ℚ) (work_AB : ℚ) (B_alone : ℚ) (AB_together : ℚ) (efficiency_A : ℚ) (B_efficiency : ℚ) :
  B_alone = 30 ∧ AB_together = 20 ∧ B_efficiency = (1/B_alone) ∧ efficiency_A = (r * B_efficiency) ∧ (efficiency_A + B_efficiency) = (1 / AB_together) → r = 1 / 2 :=
by
  sorry

end efficiency_ratio_l242_242260


namespace min_shift_value_l242_242448

theorem min_shift_value (φ : ℝ) (hφ : φ > 0) :
  (∃ k : ℤ, φ = -k * π / 3 + π / 6) →
  ∃ φ_min : ℝ, φ_min = π / 6 ∧ (∀ φ', φ' > 0 → ∃ k' : ℤ, φ' = -k' * π / 3 + π / 6 → φ_min ≤ φ') :=
by
  intro h
  use π / 6
  constructor
  . sorry
  . sorry

end min_shift_value_l242_242448


namespace parabola_equation_l242_242367

theorem parabola_equation (P : ℝ × ℝ) (hp : P = (4, -2)) : 
  ∃ m : ℝ, (∀ x y : ℝ, (y^2 = m * x) → (x, y) = P) ∧ (m = 1) :=
by
  have m_val : 1 = 1 := rfl
  sorry

end parabola_equation_l242_242367


namespace solution_set_of_inequality_l242_242011

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_deriv_neg : ∀ x : ℝ, 0 < x → (x^2 + 1) * deriv f x + 2 * x * f x < 0)
  (h_f_neg1_zero : f (-1) = 0) :
  { x : ℝ | f x > 0 } = { x | x < -1 } ∪ { x | 0 < x ∧ x < 1 } := by
  sorry

end solution_set_of_inequality_l242_242011


namespace find_dimes_l242_242511

-- Definitions for the conditions
def total_dollars : ℕ := 13
def dollar_bills_1 : ℕ := 2
def dollar_bills_5 : ℕ := 1
def quarters : ℕ := 13
def nickels : ℕ := 8
def pennies : ℕ := 35
def value_dollar_bill_1 : ℝ := 1.0
def value_dollar_bill_5 : ℝ := 5.0
def value_quarter : ℝ := 0.25
def value_nickel : ℝ := 0.05
def value_penny : ℝ := 0.01
def value_dime : ℝ := 0.10

-- Theorem statement
theorem find_dimes (total_dollars dollar_bills_1 dollar_bills_5 quarters nickels pennies : ℕ)
  (value_dollar_bill_1 value_dollar_bill_5 value_quarter value_nickel value_penny value_dime : ℝ) :
  (2 * value_dollar_bill_1 + 1 * value_dollar_bill_5 + 13 * value_quarter + 8 * value_nickel + 35 * value_penny) + 
  (20 * value_dime) = ↑total_dollars :=
sorry

end find_dimes_l242_242511


namespace first_year_payment_l242_242387

theorem first_year_payment (X : ℝ) (second_year : ℝ) (third_year : ℝ) (fourth_year : ℝ) 
    (total_payments : ℝ) 
    (h1 : second_year = X + 2)
    (h2 : third_year = X + 5)
    (h3 : fourth_year = X + 9)
    (h4 : total_payments = X + second_year + third_year + fourth_year) :
    total_payments = 96 → X = 20 :=
by
    sorry

end first_year_payment_l242_242387


namespace surface_area_of_sphere_given_cube_volume_8_l242_242508

theorem surface_area_of_sphere_given_cube_volume_8 
  (volume_of_cube : ℝ)
  (h₁ : volume_of_cube = 8) :
  ∃ (surface_area_of_sphere : ℝ), 
  surface_area_of_sphere = 12 * Real.pi :=
by
  sorry

end surface_area_of_sphere_given_cube_volume_8_l242_242508


namespace probability_slope_condition_l242_242189

theorem probability_slope_condition :
  let p := 63
  let q := 128
  let point_of_interest := (3/4, 1/4)
  let unit_square := { (x, y) | 0 <= x ∧ x <= 1 ∧ 0 <= y ∧ y <= 1 }
  let condition_met := { (x, y) ∈ unit_square | y ≥ (3/4) * x - 5/16 }
  let prob := (condition_met.measure / unit_square.measure)
  p + q = 191 := sorry

end probability_slope_condition_l242_242189


namespace product_of_solutions_eq_zero_l242_242798

theorem product_of_solutions_eq_zero : 
  (∀ x : ℝ, (x + 3) / (2 * x + 3) = (4 * x + 4) / (7 * x + 4)) → 
  ∃ (x1 x2 : ℝ), (x1 = 0 ∨ x1 = 5) ∧ (x2 = 0 ∨ x2 = 5) ∧ x1 * x2 = 0 :=
by
  sorry

end product_of_solutions_eq_zero_l242_242798


namespace adelaide_ducks_l242_242275

variable (A E K : ℕ)

theorem adelaide_ducks (h1 : A = 2 * E) (h2 : E = K - 45) (h3 : (A + E + K) / 3 = 35) :
  A = 30 := by
  sorry

end adelaide_ducks_l242_242275


namespace player_B_questions_l242_242388

theorem player_B_questions :
  ∀ (a b : ℕ → ℕ), (∀ i j, i ≠ j → a i + b j = a j + b i) →
  ∃ k, k = 11 := sorry

end player_B_questions_l242_242388


namespace sin_cos_solution_count_l242_242174

-- Statement of the problem
theorem sin_cos_solution_count : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin (3 * x) = Real.cos (x / 2)) ∧ s.card = 6 := by
  sorry

end sin_cos_solution_count_l242_242174


namespace greatest_ab_sum_l242_242925

theorem greatest_ab_sum (a b : ℝ) (h1 : a^2 + b^2 = 130) (h2 : a * b = 45) :
  a + b = Real.sqrt 220 ∨ a + b = -Real.sqrt 220 :=
sorry

end greatest_ab_sum_l242_242925


namespace initial_percentage_increase_l242_242787

variable (S : ℝ) (P : ℝ)

theorem initial_percentage_increase :
  (S + (P / 100) * S) - 0.10 * (S + (P / 100) * S) = S + 0.15 * S →
  P = 16.67 :=
by
  sorry

end initial_percentage_increase_l242_242787


namespace battery_current_l242_242055

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l242_242055


namespace number_of_boys_in_school_l242_242763

theorem number_of_boys_in_school (B : ℝ) (h1 : 542.0 = B + 155) : B = 387 :=
by
  sorry

end number_of_boys_in_school_l242_242763


namespace bus_probability_l242_242544

/-- Probability that the first bus is red, and exactly 4 blue buses (out of 6 non-red) in the lineup of 7 buses /-
theorem bus_probability (total_buses : ℕ) (red_buses : ℕ) (blue_buses : ℕ) (yellow_buses : ℕ) : 
  red_buses = 5 ∧ blue_buses = 6 ∧ yellow_buses = 5 ∧ total_buses = red_buses + blue_buses + yellow_buses → 
  ((red_buses / total_buses) * ((Nat.choose blue_buses 4) * (Nat.choose yellow_buses 2) * (Nat.choose (red_buses - 1) 0) / (Nat.choose (total_buses - 1) 6))) = 75 / 8080 :=
by
  sorry

end bus_probability_l242_242544


namespace cross_section_quadrilateral_is_cylinder_l242_242815

-- Definition of the solids
inductive Solid
| cone
| cylinder
| sphere

-- Predicate for the cross-section being a quadrilateral
def is_quadrilateral_cross_section (solid : Solid) : Prop :=
  match solid with
  | Solid.cylinder => true
  | Solid.cone     => false
  | Solid.sphere   => false

-- Main theorem statement
theorem cross_section_quadrilateral_is_cylinder (s : Solid) :
  is_quadrilateral_cross_section s → s = Solid.cylinder :=
by
  cases s
  . simp [is_quadrilateral_cross_section]
  . simp [is_quadrilateral_cross_section]
  . simp [is_quadrilateral_cross_section]

end cross_section_quadrilateral_is_cylinder_l242_242815


namespace cost_of_childrens_ticket_l242_242880

theorem cost_of_childrens_ticket (x : ℝ) 
  (h1 : ∀ A C : ℝ, A = 2 * C) 
  (h2 : 152 = 2 * 76)
  (h3 : ∀ A C : ℝ, 5.50 * A + x * C = 1026) 
  (h4 : 152 = 152) : 
  x = 2.50 :=
by
  sorry

end cost_of_childrens_ticket_l242_242880


namespace find_current_l242_242110

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l242_242110


namespace general_term_of_series_l242_242754

def gen_term (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, a n = if n = 1 then 2 else 6 * n - 5

def series_sum (S : ℕ → ℕ) : Prop :=
∀ n : ℕ, S n = 3 * n ^ 2 - 2 * n + 1

theorem general_term_of_series (a S : ℕ → ℕ) (h : series_sum S) :
  gen_term a ↔ (∀ n : ℕ, a n = if n = 1 then 2 else S n - S (n - 1)) :=
by sorry

end general_term_of_series_l242_242754


namespace pipes_fill_time_l242_242876

noncomputable def filling_time (P X Y Z : ℝ) : ℝ :=
  P / (X + Y + Z)

theorem pipes_fill_time (P : ℝ) (X Y Z : ℝ)
  (h1 : X + Y = P / 3) 
  (h2 : X + Z = P / 6) 
  (h3 : Y + Z = P / 4.5) :
  filling_time P X Y Z = 36 / 13 := by
  sorry

end pipes_fill_time_l242_242876


namespace smallest_prime_perimeter_l242_242124

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_scalene (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_triple_prime (a b c : ℕ) : Prop := is_prime a ∧ is_prime b ∧ is_prime c

def is_triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

theorem smallest_prime_perimeter :
  ∃ a b c : ℕ, is_scalene a b c ∧ is_triple_prime a b c ∧ is_prime (a + b + c) ∧ a + b + c = 23 :=
sorry

end smallest_prime_perimeter_l242_242124


namespace bullet_speed_difference_l242_242656

def speed_horse : ℕ := 20  -- feet per second
def speed_bullet : ℕ := 400  -- feet per second

def speed_forward : ℕ := speed_bullet + speed_horse
def speed_backward : ℕ := speed_bullet - speed_horse

theorem bullet_speed_difference : speed_forward - speed_backward = 40 :=
by
  sorry

end bullet_speed_difference_l242_242656


namespace number_of_girls_l242_242645

theorem number_of_girls (sections : ℕ) (boys_per_section : ℕ) (total_boys : ℕ) (total_sections : ℕ) (boys_sections girls : ℕ) :
  total_boys = 408 → 
  total_sections = 27 → 
  total_boys / total_sections = boys_per_section → 
  boys_sections = total_boys / boys_per_section → 
  total_sections - boys_sections = girls / boys_per_section → 
  girls = 324 :=
by sorry

end number_of_girls_l242_242645


namespace solve_Q1_l242_242551

noncomputable def Q1 (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y + y * f x) = f x + f y + x * f y

theorem solve_Q1 :
  ∀ f : ℝ → ℝ, Q1 f → f = (id : ℝ → ℝ) :=
  by sorry

end solve_Q1_l242_242551


namespace abs_inequality_solution_l242_242724

theorem abs_inequality_solution (x : ℝ) : 
  (|5 - 2*x| >= 3) ↔ (x ≤ 1 ∨ x ≥ 4) := sorry

end abs_inequality_solution_l242_242724


namespace determine_p_and_q_l242_242142

noncomputable def find_p_and_q (a : ℝ) (p q : ℝ) : Prop :=
  (∀ x : ℝ, x = 1 ∨ x = -1 → (x^4 + p * x^2 + q * x + a^2 = 0))

theorem determine_p_and_q (a p q : ℝ) (h : find_p_and_q a p q) : p = -(a^2 + 1) ∧ q = 0 :=
by
  -- The proof would go here.
  sorry

end determine_p_and_q_l242_242142


namespace ratio_of_areas_of_triangles_l242_242805

theorem ratio_of_areas_of_triangles 
  (a b c d e f : ℕ)
  (h1 : a = 7) (h2 : b = 24) (h3 : c = 25)
  (h4 : d = 9) (h5 : e = 40) (h6 : f = 41) : 
  (84 : ℚ) / (180 : ℚ) = 7 / 15 := by
  have hPQR : a^2 + b^2 = c^2 := by
    rw [h1, h2, h3]
    norm_num
  have hSTU : d^2 + e^2 = f^2 := by
    rw [h4, h5, h6]
    norm_num
  have areaPQR : (1/2 : ℚ) * a * b = 84 := by
    rw [h1, h2]
    norm_num
  have areaSTU : (1/2 : ℚ) * d * e = 180 := by
    rw [h4, h5]
    norm_num
  sorry

end ratio_of_areas_of_triangles_l242_242805


namespace binom_30_3_l242_242700

def binomial_coefficient (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem binom_30_3 : binomial_coefficient 30 3 = 4060 := by
  sorry

end binom_30_3_l242_242700


namespace battery_current_when_resistance_12_l242_242079

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l242_242079


namespace system_solution_a_l242_242762

theorem system_solution_a (x y a : ℝ) (h1 : 3 * x + y = a) (h2 : 2 * x + 5 * y = 2 * a) (hx : x = 3) : a = 13 :=
by
  sorry

end system_solution_a_l242_242762
