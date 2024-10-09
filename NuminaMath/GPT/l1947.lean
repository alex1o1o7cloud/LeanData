import Mathlib

namespace area_ratio_trapezoid_abm_abcd_l1947_194777

-- Definitions based on conditions
variables {A B C D M : Type} [Zero A] [Zero B] [Zero C] [Zero D] [Zero M]
variables (BC AD : ℝ)

-- Condition: ABCD is a trapezoid with BC parallel to AD and diagonals AC and BD intersect M
-- Given BC = b and AD = a

-- Theorem statement
theorem area_ratio_trapezoid_abm_abcd (a b : ℝ) (h1 : BC = b) (h2 : AD = a) : 
  ∃ S_ABM S_ABCD : ℝ,
  (S_ABM / S_ABCD = a * b / (a + b)^2) :=
sorry

end area_ratio_trapezoid_abm_abcd_l1947_194777


namespace subset_M_N_l1947_194707

-- Definitions of M and N as per the problem statement
def M : Set ℝ := {-1, 1}
def N : Set ℝ := {x | 1 / x < 2}

-- Lean statement for the proof problem: M ⊆ N
theorem subset_M_N : M ⊆ N := by
  -- Proof will be provided here
  sorry

end subset_M_N_l1947_194707


namespace relationship_between_k_and_a_l1947_194761

theorem relationship_between_k_and_a (a k : ℝ) (h_a : 0 < a ∧ a < 1) :
  (k^2 + 1) * a^2 ≥ 1 :=
sorry

end relationship_between_k_and_a_l1947_194761


namespace shortest_side_length_l1947_194718

theorem shortest_side_length (A B C : ℝ) (a b c : ℝ)
  (h_sinA : Real.sin A = 5 / 13)
  (h_cosB : Real.cos B = 3 / 5)
  (h_longest : c = 63)
  (h_angles : A < B ∧ C = π - (A + B)) :
  a = 25 := by
sorry

end shortest_side_length_l1947_194718


namespace perp_line_eq_l1947_194735

theorem perp_line_eq (x y : ℝ) (c : ℝ) (hx : x = 1) (hy : y = 2) (hline : 2 * x + y - 5 = 0) :
  x - 2 * y + c = 0 ↔ c = 3 := 
by
  sorry

end perp_line_eq_l1947_194735


namespace xyz_inequality_l1947_194729

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  (x - y) * (y - z) * (x - z) ≤ 1 / Real.sqrt 2 :=
by
  sorry

end xyz_inequality_l1947_194729


namespace inverse_proportion_inequality_l1947_194799

variable (x1 x2 k : ℝ)

theorem inverse_proportion_inequality (hA : 2 = k / x1) (hB : 4 = k / x2) (hk : 0 < k) : 
  x1 > x2 ∧ x1 > 0 ∧ x2 > 0 :=
sorry

end inverse_proportion_inequality_l1947_194799


namespace total_number_of_fish_l1947_194750

-- Define the number of each type of fish
def goldfish : ℕ := 23
def blue_fish : ℕ := 15
def angelfish : ℕ := 8
def neon_tetra : ℕ := 12

-- Theorem stating the total number of fish
theorem total_number_of_fish : goldfish + blue_fish + angelfish + neon_tetra = 58 := by
  sorry

end total_number_of_fish_l1947_194750


namespace Annie_cookies_sum_l1947_194720

theorem Annie_cookies_sum :
  let cookies_monday := 5
  let cookies_tuesday := 2 * cookies_monday
  let cookies_wednesday := cookies_tuesday + (40 / 100) * cookies_tuesday
  cookies_monday + cookies_tuesday + cookies_wednesday = 29 :=
by
  sorry

end Annie_cookies_sum_l1947_194720


namespace least_cost_planting_l1947_194734

theorem least_cost_planting :
  let region1_area := 3 * 1
  let region2_area := 4 * 4
  let region3_area := 7 * 2
  let region4_area := 5 * 4
  let region5_area := 5 * 6
  let easter_lilies_cost_per_sqft := 3.25
  let dahlias_cost_per_sqft := 2.75
  let cannas_cost_per_sqft := 2.25
  let begonias_cost_per_sqft := 1.75
  let asters_cost_per_sqft := 1.25
  region1_area * easter_lilies_cost_per_sqft +
  region2_area * dahlias_cost_per_sqft +
  region3_area * cannas_cost_per_sqft +
  region4_area * begonias_cost_per_sqft +
  region5_area * asters_cost_per_sqft =
  156.75 := 
sorry

end least_cost_planting_l1947_194734


namespace range_of_x_l1947_194733

theorem range_of_x (x : ℝ) : (∀ t : ℝ, -1 ≤ t ∧ t ≤ 3 → x^2 - (t^2 + t - 3) * x + t^2 * (t - 3) > 0) ↔ (x < -4 ∨ x > 9) :=
by
  sorry

end range_of_x_l1947_194733


namespace correct_equation_l1947_194749

-- Conditions:
def number_of_branches (x : ℕ) := x
def number_of_small_branches (x : ℕ) := x * x
def total_number (x : ℕ) := 1 + number_of_branches x + number_of_small_branches x

-- Proof Problem:
theorem correct_equation (x : ℕ) : total_number x = 43 → x^2 + x + 1 = 43 :=
by 
  sorry

end correct_equation_l1947_194749


namespace boat_speed_in_still_water_l1947_194784

theorem boat_speed_in_still_water
  (speed_of_stream : ℝ)
  (time_downstream : ℝ)
  (distance_downstream : ℝ)
  (effective_speed : ℝ)
  (boat_speed : ℝ)
  (h1: speed_of_stream = 5)
  (h2: time_downstream = 2)
  (h3: distance_downstream = 54)
  (h4: effective_speed = boat_speed + speed_of_stream)
  (h5: distance_downstream = effective_speed * time_downstream) :
  boat_speed = 22 := by
  sorry

end boat_speed_in_still_water_l1947_194784


namespace determine_function_l1947_194768

def satisfies_condition (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))

theorem determine_function (f : ℤ → ℤ) (h : satisfies_condition f) :
  ∀ n : ℤ, f n = 0 ∨ ∃ K : ℤ, f n = 2 * n + K :=
sorry

end determine_function_l1947_194768


namespace probability_of_non_perimeter_square_l1947_194723

-- Defining the total number of squares on a 10x10 board
def total_squares : ℕ := 10 * 10

-- Defining the number of perimeter squares
def perimeter_squares : ℕ := 10 + 10 + (10 - 2) * 2

-- Defining the number of non-perimeter squares
def non_perimeter_squares : ℕ := total_squares - perimeter_squares

-- Defining the probability of selecting a non-perimeter square
def probability_non_perimeter : ℚ := non_perimeter_squares / total_squares

-- The main theorem statement to be proved
theorem probability_of_non_perimeter_square:
  probability_non_perimeter = 16 / 25 := 
sorry

end probability_of_non_perimeter_square_l1947_194723


namespace mark_reads_1750_pages_per_week_l1947_194741

def initialReadingHoursPerDay := 2
def increasePercentage := 150
def initialPagesPerDay := 100

def readingHoursPerDayAfterIncrease : Nat := initialReadingHoursPerDay + (initialReadingHoursPerDay * increasePercentage) / 100
def readingSpeedPerHour := initialPagesPerDay / initialReadingHoursPerDay
def pagesPerDayNow := readingHoursPerDayAfterIncrease * readingSpeedPerHour
def pagesPerWeekNow : Nat := pagesPerDayNow * 7

theorem mark_reads_1750_pages_per_week :
  pagesPerWeekNow = 1750 :=
sorry -- Proof omitted

end mark_reads_1750_pages_per_week_l1947_194741


namespace total_number_of_bees_is_fifteen_l1947_194721

noncomputable def totalBees (B : ℝ) : Prop :=
  (1/5) * B + (1/3) * B + (2/5) * B + 1 = B

theorem total_number_of_bees_is_fifteen : ∃ B : ℝ, totalBees B ∧ B = 15 :=
by
  sorry

end total_number_of_bees_is_fifteen_l1947_194721


namespace forces_angle_result_l1947_194746

noncomputable def forces_angle_condition (p1 p2 p : ℝ) (α : ℝ) : Prop :=
  p^2 = p1 * p2

noncomputable def angle_condition_range (p1 p2 : ℝ) : Prop :=
  (3 - Real.sqrt 5) / 2 ≤ p1 / p2 ∧ p1 / p2 ≤ (3 + Real.sqrt 5) / 2

theorem forces_angle_result (p1 p2 p α : ℝ) (h : forces_angle_condition p1 p2 p α) :
  120 * π / 180 ≤ α ∧ α ≤ 120 * π / 180 ∧ (angle_condition_range p1 p2) := 
sorry

end forces_angle_result_l1947_194746


namespace lock_combination_correct_l1947_194700

noncomputable def lock_combination : ℤ := 812

theorem lock_combination_correct :
  ∀ (S T A R : ℕ), S ≠ T → S ≠ A → S ≠ R → T ≠ A → T ≠ R → A ≠ R →
  ((S * 9^4 + T * 9^3 + A * 9^2 + R * 9 + S) + 
   (T * 9^4 + A * 9^3 + R * 9^2 + T * 9 + S) + 
   (S * 9^4 + T * 9^3 + A * 9^2 + R * 9 + T)) % 9^5 = 
  S * 9^4 + T * 9^3 + A * 9^2 + R * 9 + S →
  (S * 9^2 + T * 9^1 + A) = lock_combination := 
by
  intros S T A R hST hSA hSR hTA hTR hAR h_eq
  sorry

end lock_combination_correct_l1947_194700


namespace geometric_sequence_value_l1947_194771

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_value
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geo : geometric_sequence a r)
  (h_pos : ∀ n, a n > 0)
  (h_roots : ∀ (a1 a19 : ℝ), a1 = a 1 → a19 = a 19 → a1 * a19 = 16 ∧ a1 + a19 = 10) :
  a 8 * a 10 * a 12 = 64 := 
sorry

end geometric_sequence_value_l1947_194771


namespace function_property_l1947_194786

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem function_property
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_property : ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) > 0)
  : f (-4) > f (-6) :=
sorry

end function_property_l1947_194786


namespace students_answered_both_questions_correctly_l1947_194710

theorem students_answered_both_questions_correctly (P_A P_B P_A'_B' : ℝ) (h_P_A : P_A = 0.75) (h_P_B : P_B = 0.7) (h_P_A'_B' : P_A'_B' = 0.2) :
  ∃ P_A_B : ℝ, P_A_B = 0.65 := 
by
  sorry

end students_answered_both_questions_correctly_l1947_194710


namespace employee_payment_l1947_194769

theorem employee_payment (X Y : ℝ) 
  (h1 : X + Y = 880) 
  (h2 : X = 1.2 * Y) : Y = 400 := by
  sorry

end employee_payment_l1947_194769


namespace smallest_n_l1947_194766

theorem smallest_n (n : ℕ) (h1 : n ≡ 1 [MOD 3]) (h2 : n ≡ 4 [MOD 5]) (h3 : n > 20) : n = 34 := 
sorry

end smallest_n_l1947_194766


namespace corresponding_angle_C1_of_similar_triangles_l1947_194765

theorem corresponding_angle_C1_of_similar_triangles
  (α β γ : ℝ)
  (ABC_sim_A1B1C1 : true)
  (angle_A : α = 50)
  (angle_B : β = 95) :
  γ = 35 :=
by
  sorry

end corresponding_angle_C1_of_similar_triangles_l1947_194765


namespace remainder_50_pow_50_mod_7_l1947_194728

theorem remainder_50_pow_50_mod_7 : (50^50) % 7 = 1 := by
  sorry

end remainder_50_pow_50_mod_7_l1947_194728


namespace probability_sum_8_twice_l1947_194732

-- Define a structure for the scenario: a 7-sided die.
structure Die7 :=
(sides : Fin 7)

-- Define a function to check if the sum of two dice equals 8.
def is_sum_8 (d1 d2 : Die7) : Prop :=
  (d1.sides.val + 1) + (d2.sides.val + 1) = 8

-- Define the probability of the event given the conditions.
def probability_event_twice (successes total_outcomes : ℕ) : ℚ :=
  (successes / total_outcomes) * (successes / total_outcomes)

-- The total number of outcomes when rolling two 7-sided dice.
def total_outcomes : ℕ := 7 * 7

-- The number of successful outcomes that yield a sum of 8 with two rolls.
def successful_outcomes : ℕ := 7

-- Main theorem statement to be proved.
theorem probability_sum_8_twice :
  probability_event_twice successful_outcomes total_outcomes = 1 / 49 :=
by
  -- Sorry to indicate that the proof is omitted.
  sorry

end probability_sum_8_twice_l1947_194732


namespace range_of_m_l1947_194783

theorem range_of_m (f : ℝ → ℝ) {m : ℝ} (h_dec : ∀ x y, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 ∧ x < y → f x ≥ f y)
  (h_ineq : f (m - 1) > f (2 * m - 1)) : 0 < m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l1947_194783


namespace johns_weekly_allowance_l1947_194787

variable (A : ℝ)

theorem johns_weekly_allowance 
  (h1 : ∃ A : ℝ, A > 0) 
  (h2 : (4/15) * A = 0.75) : 
  A = 2.8125 := 
by 
  -- Proof can be filled in here
  sorry

end johns_weekly_allowance_l1947_194787


namespace find_normal_price_l1947_194701

open Real

theorem find_normal_price (P : ℝ) (h1 : 0.612 * P = 108) : P = 176.47 := by
  sorry

end find_normal_price_l1947_194701


namespace sequence_product_l1947_194770

theorem sequence_product (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = q * a n) (h₄ : a 4 = 2) :
  a 2 * a 3 * a 5 * a 6 = 16 :=
sorry

end sequence_product_l1947_194770


namespace composite_numbers_characterization_l1947_194772

noncomputable def is_sum_and_product_seq (n : ℕ) (seq : List ℕ) : Prop :=
  seq.sum = n ∧ seq.prod = n ∧ 2 ≤ seq.length ∧ ∀ x ∈ seq, 1 ≤ x

theorem composite_numbers_characterization (n : ℕ) :
  (∃ seq : List ℕ, is_sum_and_product_seq n seq) ↔ ¬Nat.Prime n ∧ 1 < n :=
sorry

end composite_numbers_characterization_l1947_194772


namespace moon_radius_scientific_notation_l1947_194736

noncomputable def moon_radius : ℝ := 1738000

theorem moon_radius_scientific_notation :
  moon_radius = 1.738 * 10^6 :=
by
  sorry

end moon_radius_scientific_notation_l1947_194736


namespace literature_more_than_science_science_less_than_literature_percent_l1947_194702

theorem literature_more_than_science (l s : ℕ) (h : 8 * s = 5 * l) : (l - s) / s = 3 / 5 :=
by {
  -- definition and given condition will be provided
  sorry
}

theorem science_less_than_literature_percent (l s : ℕ) (h : 8 * s = 5 * l) : ((l - s : ℚ) / l) * 100 = 37.5 :=
by {
  -- definition and given condition will be provided
  sorry
}

end literature_more_than_science_science_less_than_literature_percent_l1947_194702


namespace altitude_inequality_l1947_194724

theorem altitude_inequality
  (a b m_a m_b : ℝ)
  (h1 : a > b)
  (h2 : a * m_a = b * m_b) :
  a^2010 + m_a^2010 ≥ b^2010 + m_b^2010 :=
sorry

end altitude_inequality_l1947_194724


namespace range_of_a_nonempty_intersection_range_of_a_subset_intersection_l1947_194763

-- Define set A
def A : Set ℝ := {x | (x + 1) * (4 - x) ≤ 0}

-- Define set B in terms of variable a
def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 2}

-- Statement 1: Proving the range of a when A ∩ B ≠ ∅
theorem range_of_a_nonempty_intersection (a : ℝ) : (A ∩ B a ≠ ∅) → (-1 / 2 ≤ a ∧ a ≤ 2) :=
by
  sorry

-- Statement 2: Proving the range of a when A ∩ B = B
theorem range_of_a_subset_intersection (a : ℝ) : (A ∩ B a = B a) → (a ≥ 2 ∨ a ≤ -3) :=
by
  sorry

end range_of_a_nonempty_intersection_range_of_a_subset_intersection_l1947_194763


namespace julia_played_tag_with_4_kids_on_tuesday_l1947_194719

variable (k_monday : ℕ) (k_diff : ℕ)

theorem julia_played_tag_with_4_kids_on_tuesday
  (h_monday : k_monday = 16)
  (h_diff : k_monday = k_tuesday + 12) :
  k_tuesday = 4 :=
by
  sorry

end julia_played_tag_with_4_kids_on_tuesday_l1947_194719


namespace cannot_form_right_triangle_setA_l1947_194740

def is_right_triangle (a b c : ℝ) : Prop :=
  (a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)

theorem cannot_form_right_triangle_setA (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  ¬ is_right_triangle a b c :=
by {
  sorry
}

end cannot_form_right_triangle_setA_l1947_194740


namespace sum_of_numbers_l1947_194764

theorem sum_of_numbers (a b c : ℝ) :
  a^2 + b^2 + c^2 = 138 → ab + bc + ca = 131 → a + b + c = 20 :=
by
  sorry

end sum_of_numbers_l1947_194764


namespace reinforcement_1600_l1947_194792

/-- A garrison of 2000 men has provisions for 54 days. After 18 days, a reinforcement arrives, and it is now found that the provisions will last only for 20 days more. We define the initial total provisions, remaining provisions after 18 days, and form equations to solve for the unknown reinforcement R.
We need to prove that R = 1600 given these conditions.
-/
theorem reinforcement_1600 (P : ℕ) (M1 M2 : ℕ) (D1 D2 : ℕ) (R : ℕ) :
  M1 = 2000 →
  D1 = 54 →
  D2 = 20 →
  M2 = 2000 + R →
  P = M1 * D1 →
  (M1 * (D1 - 18) = M2 * D2) →
  R = 1600 :=
by
  intros hM1 hD1 hD2 hM2 hP hEquiv
  sorry

end reinforcement_1600_l1947_194792


namespace find_m_l1947_194753

def setA (m : ℝ) : Set ℝ := {1, m - 2}
def setB : Set ℝ := {2}

theorem find_m (m : ℝ) (H : setA m ∩ setB = {2}) : m = 4 :=
by
  sorry

end find_m_l1947_194753


namespace correct_transformation_l1947_194788

theorem correct_transformation :
  (∀ a b c : ℝ, c ≠ 0 → (a / c = b / c ↔ a = b)) ∧
  (∀ x : ℝ, ¬ (x / 4 + x / 3 = 1 ∧ 3 * x + 4 * x = 1)) ∧
  (∀ a b c : ℝ, ¬ (a * b = b * c ∧ a ≠ c)) ∧
  (∀ x a : ℝ, ¬ (4 * x = a ∧ x = 4 * a)) := sorry

end correct_transformation_l1947_194788


namespace retailer_overhead_expenses_l1947_194706

theorem retailer_overhead_expenses (purchase_price selling_price profit_percent : ℝ) (overhead_expenses : ℝ) 
  (h1 : purchase_price = 225) 
  (h2 : selling_price = 300) 
  (h3 : profit_percent = 25) 
  (h4 : selling_price = (purchase_price + overhead_expenses) * (1 + profit_percent / 100)) : 
  overhead_expenses = 15 := 
by
  sorry

end retailer_overhead_expenses_l1947_194706


namespace highest_price_without_lowering_revenue_minimum_annual_sales_volume_and_price_l1947_194703

noncomputable def original_price : ℝ := 25
noncomputable def original_sales_volume : ℝ := 80000
noncomputable def sales_volume_decrease_per_yuan_increase : ℝ := 2000

-- Question 1
theorem highest_price_without_lowering_revenue :
  ∀ (x : ℝ), 
  25 ≤ x ∧ (8 - (x - original_price) * 0.2) * x ≥ 25 * 8 → 
  x ≤ 40 :=
sorry

-- Question 2
noncomputable def tech_reform_fee (x : ℝ) : ℝ := (1 / 6) * (x^2 - 600)
noncomputable def fixed_promotion_fee : ℝ := 50
noncomputable def variable_promotion_fee (x : ℝ) : ℝ := (1 / 5) * x

theorem minimum_annual_sales_volume_and_price (x : ℝ) (a : ℝ) :
  x > 25 →
  (a * x ≥ 25 * 8 + fixed_promotion_fee + tech_reform_fee x + variable_promotion_fee x) →
  (a ≥ 10.2 ∧ x = 30) :=
sorry

end highest_price_without_lowering_revenue_minimum_annual_sales_volume_and_price_l1947_194703


namespace map_scale_to_yards_l1947_194739

theorem map_scale_to_yards :
  (6.25 * 500) / 3 = 1041 + 2 / 3 := 
by sorry

end map_scale_to_yards_l1947_194739


namespace Tim_running_hours_per_week_l1947_194747

noncomputable def running_time_per_week : ℝ :=
  let MWF_morning : ℝ := (1 * 60 + 20 - 10) / 60 -- minutes to hours
  let MWF_evening : ℝ := (45 - 10) / 60 -- minutes to hours
  let TS_morning : ℝ := (1 * 60 + 5 - 10) / 60 -- minutes to hours
  let TS_evening : ℝ := (50 - 10) / 60 -- minutes to hours
  let MWF_total : ℝ := (MWF_morning + MWF_evening) * 3
  let TS_total : ℝ := (TS_morning + TS_evening) * 2
  MWF_total + TS_total

theorem Tim_running_hours_per_week : running_time_per_week = 8.42 := by
  -- Add the detailed proof here
  sorry

end Tim_running_hours_per_week_l1947_194747


namespace rectangle_length_is_4_l1947_194760

theorem rectangle_length_is_4 (a : ℕ) (s : a = 4) (area_square : ℕ) 
(area_square_eq : area_square = a * a) 
(area_rectangle_eq : area_square = a * 4) : 
4 = a := by
  sorry

end rectangle_length_is_4_l1947_194760


namespace num_chords_num_triangles_l1947_194789

noncomputable def num_points : ℕ := 10

theorem num_chords (n : ℕ) (h : n = num_points) : (n.choose 2) = 45 := by
  sorry

theorem num_triangles (n : ℕ) (h : n = num_points) : (n.choose 3) = 120 := by
  sorry

end num_chords_num_triangles_l1947_194789


namespace james_writing_time_l1947_194793

theorem james_writing_time (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (num_people : ℕ) (days_per_week : ℕ):
  pages_per_hour = 10 →
  pages_per_person_per_day = 5 →
  num_people = 2 →
  days_per_week = 7 →
  (5 * 2 * 7) / 10 = 7 :=
by
  intros
  sorry

end james_writing_time_l1947_194793


namespace minimum_value_l1947_194779

noncomputable def min_value_condition (a b : ℝ) : Prop :=
  a + b = 1 / 2

theorem minimum_value (a b : ℝ) (h : min_value_condition a b) :
  (4 / a) + (1 / b) ≥ 18 :=
by
  sorry

end minimum_value_l1947_194779


namespace line_through_points_l1947_194778

theorem line_through_points (x1 y1 x2 y2 : ℝ) (m b : ℝ) 
  (h1 : x1 = -3) (h2 : y1 = 1) (h3 : x2 = 1) (h4 : y2 = 3)
  (h5 : y1 = m * x1 + b) (h6 : y2 = m * x2 + b) :
  m + b = 3 := 
sorry

end line_through_points_l1947_194778


namespace right_triangle_sqrt_l1947_194755

noncomputable def sqrt_2 := Real.sqrt 2
noncomputable def sqrt_3 := Real.sqrt 3
noncomputable def sqrt_5 := Real.sqrt 5

theorem right_triangle_sqrt: 
  (sqrt_2 ^ 2 + sqrt_3 ^ 2 = sqrt_5 ^ 2) :=
by
  sorry

end right_triangle_sqrt_l1947_194755


namespace option_c_opp_numbers_l1947_194717

theorem option_c_opp_numbers : (- (2 ^ 2)) = - ((-2) ^ 2) :=
by
  sorry

end option_c_opp_numbers_l1947_194717


namespace image_of_center_after_transform_l1947_194737

structure Point where
  x : ℤ
  y : ℤ

def reflect_across_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

def translate_right (p : Point) (units : ℤ) : Point :=
  { x := p.x + units, y := p.y }

def transform_point (p : Point) : Point :=
  translate_right (reflect_across_x p) 5

theorem image_of_center_after_transform :
  transform_point {x := -3, y := 4} = {x := 2, y := -4} := by
  sorry

end image_of_center_after_transform_l1947_194737


namespace circle_x_intercept_of_given_diameter_l1947_194796

theorem circle_x_intercept_of_given_diameter (A B : ℝ × ℝ) (hA : A = (2, 2)) (hB : B = (10, 8)) : ∃ x : ℝ, ((A.1 + B.1) / 2, (A.2 + B.2) / 2).1 - 6 = 0 :=
by
  -- Sorry to skip the proof
  sorry

end circle_x_intercept_of_given_diameter_l1947_194796


namespace min_value_of_a_squared_plus_b_squared_l1947_194759

-- Problem definition and condition
def is_on_circle (a b : ℝ) : Prop :=
  (a^2 + b^2 - 2*a + 4*b - 20) = 0

-- Theorem statement
theorem min_value_of_a_squared_plus_b_squared (a b : ℝ) (h : is_on_circle a b) :
  a^2 + b^2 = 30 - 10 * Real.sqrt 5 :=
sorry

end min_value_of_a_squared_plus_b_squared_l1947_194759


namespace inequality_not_holds_l1947_194712

variable (x y : ℝ)

theorem inequality_not_holds (h1 : x > 1) (h2 : 1 > y) : x - 1 ≤ 1 - y :=
sorry

end inequality_not_holds_l1947_194712


namespace combined_weight_l1947_194791

theorem combined_weight (S R : ℝ) (h1 : S - 5 = 2 * R) (h2 : S = 75) : S + R = 110 :=
sorry

end combined_weight_l1947_194791


namespace inequality_solution_set_l1947_194752

theorem inequality_solution_set (x : ℝ) : 3 ≤ abs (5 - 2 * x) ∧ abs (5 - 2 * x) < 9 ↔ (x > -2 ∧ x ≤ 1) ∨ (x ≥ 4 ∧ x < 7) := sorry

end inequality_solution_set_l1947_194752


namespace banana_equivalence_l1947_194797

theorem banana_equivalence :
  (3 / 4 : ℚ) * 12 = 9 → (1 / 3 : ℚ) * 6 = 2 :=
by
  intro h1
  linarith

end banana_equivalence_l1947_194797


namespace ratio_of_areas_of_triangles_l1947_194705

noncomputable def area_ratio_triangle_GHI_JKL
  (a_GHI b_GHI c_GHI : ℕ) (a_JKL b_JKL c_JKL : ℕ) 
  (alt_ratio_GHI : ℕ × ℕ) (alt_ratio_JKL : ℕ × ℕ) : ℚ :=
  let area_GHI := (a_GHI * b_GHI) / 2
  let area_JKL := (a_JKL * b_JKL) / 2
  area_GHI / area_JKL

theorem ratio_of_areas_of_triangles :
  let GHI_sides := (7, 24, 25)
  let JKL_sides := (9, 40, 41)
  area_ratio_triangle_GHI_JKL 7 24 25 9 40 41 (2, 3) (4, 5) = 7 / 15 :=
by
  sorry

end ratio_of_areas_of_triangles_l1947_194705


namespace find_V_D_l1947_194798

noncomputable def V_A : ℚ := sorry
noncomputable def V_B : ℚ := sorry
noncomputable def V_C : ℚ := sorry
noncomputable def V_D : ℚ := sorry
noncomputable def V_E : ℚ := sorry

axiom condition1 : V_A + V_B + V_C + V_D + V_E = 1 / 7.5
axiom condition2 : V_A + V_C + V_E = 1 / 5
axiom condition3 : V_A + V_C + V_D = 1 / 6
axiom condition4 : V_B + V_D + V_E = 1 / 4

theorem find_V_D : V_D = 1 / 12 := 
  by
    sorry

end find_V_D_l1947_194798


namespace square_roots_equal_49_l1947_194794

theorem square_roots_equal_49 (x a : ℝ) (hx1 : (2 * x - 3)^2 = a) (hx2 : (5 - x)^2 = a) (ha_pos: a > 0) : a = 49 := 
by 
  sorry

end square_roots_equal_49_l1947_194794


namespace value_of_expression_l1947_194716

theorem value_of_expression (a : ℝ) (h : 10 * a^2 + 3 * a + 2 = 5) : 
  3 * a + 2 = (31 + 3 * Real.sqrt 129) / 20 :=
by sorry

end value_of_expression_l1947_194716


namespace greatest_integer_value_l1947_194790

theorem greatest_integer_value (x : ℤ) : 3 * |x - 2| + 9 ≤ 24 → x ≤ 7 :=
by sorry

end greatest_integer_value_l1947_194790


namespace length_of_second_square_l1947_194776

-- Define conditions as variables
def Area_flag := 135
def Area_square1 := 40
def Area_square3 := 25

-- Define the length variable for the second square
variable (L : ℕ)

-- Define the area of the second square in terms of L
def Area_square2 : ℕ := 7 * L

-- Lean statement to be proved
theorem length_of_second_square :
  Area_square1 + Area_square2 L + Area_square3 = Area_flag → L = 10 :=
by sorry

end length_of_second_square_l1947_194776


namespace evaluate_expression_l1947_194711

-- Define the operation * given by the table
def op (a b : ℕ) : ℕ :=
  match (a, b) with
  | (1,1) => 1 | (1,2) => 2 | (1,3) => 3 | (1,4) => 4
  | (2,1) => 2 | (2,2) => 4 | (2,3) => 1 | (2,4) => 3
  | (3,1) => 3 | (3,2) => 1 | (3,3) => 4 | (3,4) => 2
  | (4,1) => 4 | (4,2) => 3 | (4,3) => 2 | (4,4) => 1
  | _ => 0  -- default to handle cases outside the defined table

-- Define the theorem to prove $(2*4)*(1*3) = 4$
theorem evaluate_expression : op (op 2 4) (op 1 3) = 4 := by
  sorry

end evaluate_expression_l1947_194711


namespace real_value_of_m_pure_imaginary_value_of_m_l1947_194730

open Complex

-- Given condition
def z (m : ℝ) : ℂ := (m^2 - m : ℂ) - (m^2 - 1 : ℂ) * I

-- Part (I)
theorem real_value_of_m (m : ℝ) (h : im (z m) = 0) : m = 1 ∨ m = -1 := by
  sorry

-- Part (II)
theorem pure_imaginary_value_of_m (m : ℝ) (h1 : re (z m) = 0) (h2 : im (z m) ≠ 0) : m = 0 := by
  sorry

end real_value_of_m_pure_imaginary_value_of_m_l1947_194730


namespace exponentiated_value_l1947_194756

theorem exponentiated_value (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3 * a + b) = 24 := by
  sorry

end exponentiated_value_l1947_194756


namespace cost_of_basic_calculator_l1947_194744

variable (B S G : ℕ)

theorem cost_of_basic_calculator 
  (h₁ : S = 2 * B)
  (h₂ : G = 3 * S)
  (h₃ : B + S + G = 72) : 
  B = 8 :=
by
  sorry

end cost_of_basic_calculator_l1947_194744


namespace intersection_of_A_and_B_l1947_194725

def set_A : Set ℝ := {x | -x^2 - x + 6 > 0}
def set_B : Set ℝ := {x | 5 / (x - 3) ≤ -1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x | -2 ≤ x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l1947_194725


namespace ratio_of_neighborhood_to_gina_l1947_194773

variable (Gina_bags : ℕ) (Weight_per_bag : ℕ) (Total_weight_collected : ℕ)

def neighborhood_to_gina_ratio (Gina_bags : ℕ) (Weight_per_bag : ℕ) (Total_weight_collected : ℕ) := 
  (Total_weight_collected - Gina_bags * Weight_per_bag) / (Gina_bags * Weight_per_bag)

theorem ratio_of_neighborhood_to_gina 
  (h₁ : Gina_bags = 2) 
  (h₂ : Weight_per_bag = 4) 
  (h₃ : Total_weight_collected = 664) :
  neighborhood_to_gina_ratio Gina_bags Weight_per_bag Total_weight_collected = 82 := 
by 
  sorry

end ratio_of_neighborhood_to_gina_l1947_194773


namespace arithmetic_identity_l1947_194774

theorem arithmetic_identity : 45 * 27 + 73 * 45 = 4500 := by sorry

end arithmetic_identity_l1947_194774


namespace cost_effective_bus_choice_l1947_194715

theorem cost_effective_bus_choice (x y : ℕ) (h1 : y = x - 1) (h2 : 32 < 48 * x - 64 * y ∧ 48 * x - 64 * y < 64) : 
  64 * 300 < x * 2600 → True :=
by {
  sorry
}

end cost_effective_bus_choice_l1947_194715


namespace females_advanced_degrees_under_40_l1947_194785

-- Definitions derived from conditions
def total_employees : ℕ := 280
def female_employees : ℕ := 160
def male_employees : ℕ := 120
def advanced_degree_holders : ℕ := 120
def college_degree_holders : ℕ := 100
def high_school_diploma_holders : ℕ := 60
def male_advanced_degree_holders : ℕ := 50
def male_college_degree_holders : ℕ := 35
def male_high_school_diploma_holders : ℕ := 35
def percentage_females_under_40 : ℝ := 0.75

-- The mathematically equivalent proof problem
theorem females_advanced_degrees_under_40 : 
  (advanced_degree_holders - male_advanced_degree_holders) * percentage_females_under_40 = 52 :=
by
  sorry -- Proof to be provided

end females_advanced_degrees_under_40_l1947_194785


namespace sphere_radius_l1947_194782

theorem sphere_radius (x y z r : ℝ) (h1 : 2 * x * y + 2 * y * z + 2 * z * x = 384)
  (h2 : x + y + z = 28) (h3 : (2 * r)^2 = x^2 + y^2 + z^2) : r = 10 := sorry

end sphere_radius_l1947_194782


namespace rain_in_first_hour_l1947_194757

theorem rain_in_first_hour (x : ℝ) (h1 : ∀ y : ℝ, y = 2 * x + 7) (h2 : x + (2 * x + 7) = 22) : x = 5 :=
sorry

end rain_in_first_hour_l1947_194757


namespace alice_basketball_probability_l1947_194767

/-- Alice and Bob play a game with a basketball. On each turn, if Alice has the basketball,
 there is a 5/8 chance that she will toss it to Bob and a 3/8 chance that she will keep the basketball.
 If Bob has the basketball, there is a 1/4 chance that he will toss it to Alice, and if he doesn't toss it to Alice,
 he keeps it. Alice starts with the basketball. What is the probability that Alice has the basketball again after two turns? -/
theorem alice_basketball_probability :
  (5 / 8) * (1 / 4) + (3 / 8) * (3 / 8) = 19 / 64 := 
by
  sorry

end alice_basketball_probability_l1947_194767


namespace petya_cannot_win_l1947_194738

theorem petya_cannot_win (n : ℕ) (h : n ≥ 3) : ¬ ∃ strategy : ℕ → ℕ → Prop, 
  (∀ k, strategy k (k+1) ∧ strategy k (k-1))
  ∧ ∀ m, ¬ strategy n m :=
sorry

end petya_cannot_win_l1947_194738


namespace min_cookies_divisible_by_13_l1947_194708

theorem min_cookies_divisible_by_13 (a b : ℕ) : ∃ n : ℕ, n > 0 ∧ n % 13 = 0 ∧ (∃ a b : ℕ, n = 10 * a + 21 * b) ∧ n = 52 :=
by
  sorry

end min_cookies_divisible_by_13_l1947_194708


namespace union_sets_l1947_194748

def A : Set ℕ := {2, 1, 3}
def B : Set ℕ := {2, 3, 5}

theorem union_sets : A ∪ B = {1, 2, 3, 5} := 
by {
  sorry
}

end union_sets_l1947_194748


namespace mechanism_parts_l1947_194704

-- Definitions
def total_parts (S L : Nat) : Prop := S + L = 25
def condition1 (S L : Nat) : Prop := ∀ (A : Finset (Fin 25)), (A.card = 12) → ∃ i, i ∈ A ∧ i < S
def condition2 (S L : Nat) : Prop := ∀ (B : Finset (Fin 25)), (B.card = 15) → ∃ i, i ∈ B ∧ i >= S

-- Main statement
theorem mechanism_parts :
  ∃ (S L : Nat), 
  total_parts S L ∧ 
  condition1 S L ∧ 
  condition2 S L ∧ 
  S = 14 ∧ 
  L = 11 :=
sorry

end mechanism_parts_l1947_194704


namespace geometric_sequence_tenth_term_l1947_194713

theorem geometric_sequence_tenth_term :
  let a : ℚ := 4
  let r : ℚ := 5/3
  let n : ℕ := 10
  a * r^(n-1) = 7812500 / 19683 :=
by sorry

end geometric_sequence_tenth_term_l1947_194713


namespace algebraic_expression_value_l1947_194754

def algebraic_expression (a b : ℤ) :=
  a + 2 * b + 2 * (a + 2 * b) + 1

theorem algebraic_expression_value :
  algebraic_expression 1 (-1) = -2 :=
by
  -- Proof skipped
  sorry

end algebraic_expression_value_l1947_194754


namespace total_people_100_l1947_194731

noncomputable def total_people (P : ℕ) : Prop :=
  (2 / 5 : ℚ) * P = 40 ∧ (1 / 4 : ℚ) * P ≤ P ∧ P ≥ 40 

theorem total_people_100 {P : ℕ} (h : total_people P) : P = 100 := 
by 
  sorry -- proof would go here

end total_people_100_l1947_194731


namespace shaded_fraction_of_rectangle_l1947_194758

theorem shaded_fraction_of_rectangle (a b : ℕ) (h_dim : a = 15 ∧ b = 24) (h_shaded : ∃ s, s = (1/3 : ℚ)) :
  ∃ f, f = (1/9 : ℚ) := 
by
  sorry

end shaded_fraction_of_rectangle_l1947_194758


namespace scientific_notation_equivalence_l1947_194762

theorem scientific_notation_equivalence : 3 * 10^(-7) = 0.0000003 :=
by
  sorry

end scientific_notation_equivalence_l1947_194762


namespace sugar_cheaper_than_apples_l1947_194709

/-- Given conditions about the prices and quantities of items that Fabian wants to buy,
    prove the price difference between one pack of sugar and one kilogram of apples. --/
theorem sugar_cheaper_than_apples
  (price_kg_apples : ℝ)
  (price_kg_walnuts : ℝ)
  (total_cost : ℝ)
  (cost_diff : ℝ)
  (num_kg_apples : ℕ := 5)
  (num_packs_sugar : ℕ := 3)
  (num_kg_walnuts : ℝ := 0.5)
  (price_kg_apples_val : price_kg_apples = 2)
  (price_kg_walnuts_val : price_kg_walnuts = 6)
  (total_cost_val : total_cost = 16) :
  cost_diff = price_kg_apples - (total_cost - (num_kg_apples * price_kg_apples + num_kg_walnuts * price_kg_walnuts))/num_packs_sugar → 
  cost_diff = 1 :=
by
  sorry

end sugar_cheaper_than_apples_l1947_194709


namespace alpha_eq_two_thirds_l1947_194714

theorem alpha_eq_two_thirds (α : ℚ) (h1 : 0 < α) (h2 : α < 1) (h3 : Real.cos (3 * Real.pi * α) + 2 * Real.cos (2 * Real.pi * α) = 0) : α = 2 / 3 :=
sorry

end alpha_eq_two_thirds_l1947_194714


namespace xiaoqiang_average_score_l1947_194745

theorem xiaoqiang_average_score
    (x : ℕ)
    (prev_avg : ℝ)
    (next_score : ℝ)
    (target_avg : ℝ)
    (h_prev_avg : prev_avg = 84)
    (h_next_score : next_score = 100)
    (h_target_avg : target_avg = 86) :
    (86 * x - (84 * (x - 1)) = 100) → x = 8 := 
by
  intros h_eq
  sorry

end xiaoqiang_average_score_l1947_194745


namespace interest_rate_of_A_to_B_l1947_194795

theorem interest_rate_of_A_to_B :
  ∀ (principal gain interest_B_to_C : ℝ), 
  principal = 3500 →
  gain = 525 →
  interest_B_to_C = 0.15 →
  (principal * interest_B_to_C * 3 - gain) = principal * (10 / 100) * 3 :=
by
  intros principal gain interest_B_to_C h_principal h_gain h_interest_B_to_C
  sorry

end interest_rate_of_A_to_B_l1947_194795


namespace problem_l1947_194722

theorem problem (a : ℕ) (b : ℚ) (c : ℤ) 
  (h1 : a = 1) 
  (h2 : b = 0) 
  (h3 : abs (c) = 6) :
  (a - b + c = (7 : ℤ)) ∨ (a - b + c = (-5 : ℤ)) := by
  sorry

end problem_l1947_194722


namespace determine_A_l1947_194780

theorem determine_A (x y A : ℝ) 
  (h : (x + y) ^ 3 - x * y * (x + y) = (x + y) * A) : 
  A = x^2 + x * y + y^2 := 
by
  sorry

end determine_A_l1947_194780


namespace perimeter_difference_zero_l1947_194781

theorem perimeter_difference_zero :
  let shape1_length := 4
  let shape1_width := 3
  let shape2_length := 6
  let shape2_width := 1
  let perimeter (l w : ℕ) := 2 * (l + w)
  perimeter shape1_length shape1_width = perimeter shape2_length shape2_width :=
by
  sorry

end perimeter_difference_zero_l1947_194781


namespace minimum_number_of_different_numbers_l1947_194727

theorem minimum_number_of_different_numbers (total_numbers : ℕ) (frequent_count : ℕ) (frequent_occurrences : ℕ) (less_frequent_occurrences : ℕ) (h1 : total_numbers = 2019) (h2 : frequent_count = 10) (h3 : less_frequent_occurrences = 9) : ∃ k : ℕ, k ≥ 225 :=
by {
  sorry
}

end minimum_number_of_different_numbers_l1947_194727


namespace eggs_today_l1947_194742

-- Condition definitions
def eggs_yesterday : ℕ := 10
def difference : ℕ := 59

-- Statement of the problem
theorem eggs_today : eggs_yesterday + difference = 69 := by
  sorry

end eggs_today_l1947_194742


namespace students_with_equal_scores_l1947_194751

theorem students_with_equal_scores 
  (n : ℕ)
  (scores : Fin n → Fin (n - 1)): 
  ∃ i j : Fin n, i ≠ j ∧ scores i = scores j := 
by 
  sorry

end students_with_equal_scores_l1947_194751


namespace intersection_M_P_l1947_194726

variable {x a : ℝ}

def M (a : ℝ) : Set ℝ := { x | x > a ∧ a^2 - 12*a + 20 < 0 }
def P : Set ℝ := { x | x ≤ 10 }

theorem intersection_M_P (a : ℝ) (h : 2 < a ∧ a < 10) : 
  M a ∩ P = { x | a < x ∧ x ≤ 10 } :=
sorry

end intersection_M_P_l1947_194726


namespace sqrt_four_squared_l1947_194743

theorem sqrt_four_squared : (Real.sqrt 4) ^ 2 = 4 :=
  by
    sorry

end sqrt_four_squared_l1947_194743


namespace find_m_l1947_194775

namespace MathProof

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 8

-- State the problem
theorem find_m (m : ℝ) (h : f 5 - g 5 m = 15) : m = -15 := by
  sorry

end MathProof

end find_m_l1947_194775
