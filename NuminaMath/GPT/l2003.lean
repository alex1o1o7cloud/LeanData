import Mathlib

namespace NUMINAMATH_GPT_a_received_share_l2003_200383

variables (I_a I_b I_c b_share total_investment total_profit a_share : ℕ)
  (h1 : I_a = 11000)
  (h2 : I_b = 15000)
  (h3 : I_c = 23000)
  (h4 : b_share = 3315)
  (h5 : total_investment = I_a + I_b + I_c)
  (h6 : total_profit = b_share * total_investment / I_b)
  (h7 : a_share = I_a * total_profit / total_investment)

theorem a_received_share : a_share = 2662 := by
  sorry

end NUMINAMATH_GPT_a_received_share_l2003_200383


namespace NUMINAMATH_GPT_evaluate_expression_at_x_neg3_l2003_200373

theorem evaluate_expression_at_x_neg3 :
  (5 + (-3) * (5 + (-3)) - 5^2) / ((-3) - 5 + (-3)^2) = -26 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_x_neg3_l2003_200373


namespace NUMINAMATH_GPT_total_books_in_week_l2003_200300

def books_read (n : ℕ) : ℕ :=
  if n = 0 then 2 -- day 1 (indexed by 0)
  else if n = 1 then 2 -- day 2
  else 2 + n -- starting from day 3 (indexed by 2)

-- Summing the books read from day 1 to day 7 (indexed from 0 to 6)
theorem total_books_in_week : (List.sum (List.map books_read [0, 1, 2, 3, 4, 5, 6])) = 29 := by
  sorry

end NUMINAMATH_GPT_total_books_in_week_l2003_200300


namespace NUMINAMATH_GPT_problem_value_expression_l2003_200344

theorem problem_value_expression 
  (x y : ℝ)
  (h₁ : x + y = 4)
  (h₂ : x * y = -2) : 
  x + (x^3 / y^2) + (y^3 / x^2) + y = 440 := 
sorry

end NUMINAMATH_GPT_problem_value_expression_l2003_200344


namespace NUMINAMATH_GPT_Mrs_Early_speed_l2003_200324

noncomputable def speed_to_reach_on_time (distance : ℝ) (ideal_time : ℝ) : ℝ := distance / ideal_time

theorem Mrs_Early_speed:
  ∃ (d t : ℝ), 
    (d = 50 * (t + 5/60)) ∧ 
    (d = 80 * (t - 7/60)) ∧ 
    (speed_to_reach_on_time d t = 59) := sorry

end NUMINAMATH_GPT_Mrs_Early_speed_l2003_200324


namespace NUMINAMATH_GPT_more_blue_count_l2003_200397

-- Definitions based on the conditions given in the problem
def total_people : ℕ := 150
def more_green : ℕ := 95
def both_green_blue : ℕ := 35
def neither_green_blue : ℕ := 25

-- The Lean statement to prove the number of people who believe turquoise is "more blue"
theorem more_blue_count : 
  (total_people - neither_green_blue) - (more_green - both_green_blue) = 65 :=
by 
  sorry

end NUMINAMATH_GPT_more_blue_count_l2003_200397


namespace NUMINAMATH_GPT_infinite_n_perfect_squares_l2003_200320

-- Define the condition that k is a positive natural number and k >= 2
variable (k : ℕ) (hk : 2 ≤ k) 

-- Define the statement asserting the existence of infinitely many n such that both kn + 1 and (k+1)n + 1 are perfect squares
theorem infinite_n_perfect_squares : ∀ k : ℕ, (2 ≤ k) → ∃ n : ℕ, ∀ m : ℕ, (2 ≤ k) → k * n + 1 = m * m ∧ (k + 1) * n + 1 = (m + k) * (m + k) := 
by
  sorry

end NUMINAMATH_GPT_infinite_n_perfect_squares_l2003_200320


namespace NUMINAMATH_GPT_mitya_age_l2003_200301

-- Definitions of the ages
variables (M S : ℕ)

-- Conditions based on the problem statements
axiom condition1 : M = S + 11
axiom condition2 : S = 2 * (S - (M - S))

-- The theorem stating that Mitya is 33 years old
theorem mitya_age : M = 33 :=
by
  -- Outline the proof
  sorry

end NUMINAMATH_GPT_mitya_age_l2003_200301


namespace NUMINAMATH_GPT_quadrilateral_area_l2003_200326

theorem quadrilateral_area 
  (d : ℝ) (h₁ h₂ : ℝ) 
  (hd : d = 22) 
  (hh₁ : h₁ = 9) 
  (hh₂ : h₂ = 6) : 
  (1/2 * d * h₁ + 1/2 * d * h₂ = 165) :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l2003_200326


namespace NUMINAMATH_GPT_tom_pie_portion_l2003_200334

theorem tom_pie_portion :
  let pie_left := 5 / 8
  let friends := 4
  let portion_per_person := pie_left / friends
  portion_per_person = 5 / 32 := by
  sorry

end NUMINAMATH_GPT_tom_pie_portion_l2003_200334


namespace NUMINAMATH_GPT_zou_mei_competition_l2003_200328

theorem zou_mei_competition (n : ℕ) (h1 : 271 = n^2 + 15) (h2 : n^2 + 33 = (n + 1)^2) : 
  ∃ n, 271 = n^2 + 15 ∧ n^2 + 33 = (n + 1)^2 :=
by
  existsi n
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_zou_mei_competition_l2003_200328


namespace NUMINAMATH_GPT_boats_meeting_distance_l2003_200311

theorem boats_meeting_distance (X : ℝ) 
  (H1 : ∃ (X : ℝ), (1200 - X) + 900 = X + 1200 + 300) 
  (H2 : X + 1200 + 300 = 2100 + X): 
  X = 300 :=
by
  sorry

end NUMINAMATH_GPT_boats_meeting_distance_l2003_200311


namespace NUMINAMATH_GPT_sticker_price_of_smartphone_l2003_200359

theorem sticker_price_of_smartphone (p : ℝ)
  (h1 : 0.90 * p - 100 = 0.80 * p - 20) : p = 800 :=
sorry

end NUMINAMATH_GPT_sticker_price_of_smartphone_l2003_200359


namespace NUMINAMATH_GPT_solve_equation_l2003_200357

theorem solve_equation :
  ∀ x : ℝ, (x * (2 * x + 4) = 10 + 5 * x) ↔ (x = -2 ∨ x = 2.5) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2003_200357


namespace NUMINAMATH_GPT_valid_marble_arrangements_eq_48_l2003_200327

def ZaraMarbleArrangements (n : ℕ) : ℕ := sorry

theorem valid_marble_arrangements_eq_48 : ZaraMarbleArrangements 5 = 48 := sorry

end NUMINAMATH_GPT_valid_marble_arrangements_eq_48_l2003_200327


namespace NUMINAMATH_GPT_janessa_gives_dexter_cards_l2003_200304

def initial_cards : Nat := 4
def father_cards : Nat := 13
def ordered_cards : Nat := 36
def bad_cards : Nat := 4
def kept_cards : Nat := 20

theorem janessa_gives_dexter_cards :
  initial_cards + father_cards + ordered_cards - bad_cards - kept_cards = 29 := 
by
  sorry

end NUMINAMATH_GPT_janessa_gives_dexter_cards_l2003_200304


namespace NUMINAMATH_GPT_interval_of_monotonic_increase_l2003_200367

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.exp x
noncomputable def y' (x : ℝ) : ℝ := 2 * x * Real.exp x + x^2 * Real.exp x

theorem interval_of_monotonic_increase :
  ∀ x : ℝ, (y' x ≥ 0 ↔ (x ∈ Set.Ici 0 ∨ x ∈ Set.Iic (-2))) :=
by
  sorry

end NUMINAMATH_GPT_interval_of_monotonic_increase_l2003_200367


namespace NUMINAMATH_GPT_any_nat_as_fraction_form_l2003_200313

theorem any_nat_as_fraction_form (n : ℕ) : ∃ (x y : ℕ), x = n^3 ∧ y = n^2 ∧ (x^3 / y^4 : ℝ) = n :=
by
  sorry

end NUMINAMATH_GPT_any_nat_as_fraction_form_l2003_200313


namespace NUMINAMATH_GPT_unique_fraction_condition_l2003_200381

theorem unique_fraction_condition :
  ∃! (x y : ℕ), x.gcd y = 1 ∧ y = x * 6 / 5 ∧ (1.2 * (x : ℚ) / y = (x + 1 : ℚ) / (y + 1)) := by
  sorry

end NUMINAMATH_GPT_unique_fraction_condition_l2003_200381


namespace NUMINAMATH_GPT_andrew_grapes_purchase_l2003_200346

theorem andrew_grapes_purchase (G : ℕ) (rate_grape rate_mango total_paid total_mango_cost : ℕ)
  (h1 : rate_grape = 54)
  (h2 : rate_mango = 62)
  (h3 : total_paid = 1376)
  (h4 : total_mango_cost = 10 * rate_mango)
  (h5 : total_paid = rate_grape * G + total_mango_cost) : G = 14 := by
  sorry

end NUMINAMATH_GPT_andrew_grapes_purchase_l2003_200346


namespace NUMINAMATH_GPT_misread_weight_l2003_200303

theorem misread_weight (n : ℕ) (average_incorrect : ℚ) (average_correct : ℚ) (corrected_weight : ℚ) (incorrect_total correct_total diff : ℚ)
  (h1 : n = 20)
  (h2 : average_incorrect = 58.4)
  (h3 : average_correct = 59)
  (h4 : corrected_weight = 68)
  (h5 : incorrect_total = n * average_incorrect)
  (h6 : correct_total = n * average_correct)
  (h7 : diff = correct_total - incorrect_total)
  (h8 : diff = corrected_weight - x) : x = 56 := 
sorry

end NUMINAMATH_GPT_misread_weight_l2003_200303


namespace NUMINAMATH_GPT_sale_in_third_month_l2003_200366

theorem sale_in_third_month
  (sale1 sale2 sale4 sale5 sale6 avg : ℝ)
  (n : ℕ)
  (h_sale1 : sale1 = 6235)
  (h_sale2 : sale2 = 6927)
  (h_sale4 : sale4 = 7230)
  (h_sale5 : sale5 = 6562)
  (h_sale6 : sale6 = 5191)
  (h_avg : avg = 6500)
  (h_n : n = 6) :
  ∃ sale3 : ℝ, sale3 = 6855 := by
  sorry

end NUMINAMATH_GPT_sale_in_third_month_l2003_200366


namespace NUMINAMATH_GPT_max_value_k_l2003_200350

noncomputable def sqrt_minus (x : ℝ) : ℝ := Real.sqrt (x - 3)
noncomputable def sqrt_six_minus (x : ℝ) : ℝ := Real.sqrt (6 - x)

theorem max_value_k (k : ℝ) : (∃ x : ℝ, 3 ≤ x ∧ x ≤ 6 ∧ sqrt_minus x + sqrt_six_minus x ≥ k) ↔ k ≤ Real.sqrt 12 := by
  sorry

end NUMINAMATH_GPT_max_value_k_l2003_200350


namespace NUMINAMATH_GPT_find_m_and_other_root_l2003_200351

theorem find_m_and_other_root (m : ℝ) (r : ℝ) :
    (∃ x : ℝ, x^2 + m*x - 2 = 0) ∧ (x = -1) → (m = -1 ∧ r = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_m_and_other_root_l2003_200351


namespace NUMINAMATH_GPT_numerator_is_12_l2003_200310

theorem numerator_is_12 (x : ℕ) (h1 : (x : ℤ) / (2 * x + 4 : ℤ) = 3 / 7) : x = 12 := 
sorry

end NUMINAMATH_GPT_numerator_is_12_l2003_200310


namespace NUMINAMATH_GPT_amelia_wins_probability_l2003_200379

theorem amelia_wins_probability :
  let pA := 1 / 4
  let pB := 1 / 3
  let pC := 1 / 2
  let cycle_probability := (1 - pA) * (1 - pB) * (1 - pC)
  let infinite_series_sum := 1 / (1 - cycle_probability)
  let total_probability := pA * infinite_series_sum
  total_probability = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_amelia_wins_probability_l2003_200379


namespace NUMINAMATH_GPT_parallelogram_slope_l2003_200349

theorem parallelogram_slope (a b c d : ℚ) :
    a = 35 + c ∧ b = 125 - c ∧ 875 - 25 * c = 280 + 8 * c ∧ (a, 8) = (b, 25)
    → ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ (∃ h : 8 * 33 * a + 595 = 2350, (m, n) = (25, 4)) :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_slope_l2003_200349


namespace NUMINAMATH_GPT_sequence_difference_l2003_200398

theorem sequence_difference (a : ℕ → ℤ) (h_rec : ∀ n : ℕ, a (n + 1) + a n = n) (h_a1 : a 1 = 2) :
  a 4 - a 2 = 1 :=
sorry

end NUMINAMATH_GPT_sequence_difference_l2003_200398


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2003_200345

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3, 4}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2003_200345


namespace NUMINAMATH_GPT_variance_transformation_l2003_200336

theorem variance_transformation (a_1 a_2 a_3 : ℝ) (h : (1 / 3) * ((a_1 - ((a_1 + a_2 + a_3) / 3))^2 + (a_2 - ((a_1 + a_2 + a_3) / 3))^2 + (a_3 - ((a_1 + a_2 + a_3) / 3))^2) = 1) :
  (1 / 3) * ((3 * a_1 + 2 - (3 * (a_1 + a_2 + a_3) / 3 + 2))^2 + (3 * a_2 + 2 - (3 * (a_1 + a_2 + a_3) / 3 + 2))^2 + (3 * a_3 + 2 - (3 * (a_1 + a_2 + a_3) / 3 + 2))^2) = 9 := by 
  sorry

end NUMINAMATH_GPT_variance_transformation_l2003_200336


namespace NUMINAMATH_GPT_parallelogram_area_l2003_200375

theorem parallelogram_area (b h : ℕ) (hb : b = 20) (hh : h = 4) : b * h = 80 := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l2003_200375


namespace NUMINAMATH_GPT_determine_xyz_l2003_200372

-- Define the conditions for the variables x, y, and z
variables (x y z : ℝ)

-- State the problem as a theorem
theorem determine_xyz :
  (x + y + z) * (x * y + x * z + y * z) = 24 ∧
  x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 8 →
  x * y * z = 16 / 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_determine_xyz_l2003_200372


namespace NUMINAMATH_GPT_plan_b_cheaper_than_plan_a_l2003_200337

theorem plan_b_cheaper_than_plan_a (x : ℕ) (h : 401 ≤ x) :
  2000 + 5 * x < 10 * x :=
by
  sorry

end NUMINAMATH_GPT_plan_b_cheaper_than_plan_a_l2003_200337


namespace NUMINAMATH_GPT_percentage_slump_in_business_l2003_200322

theorem percentage_slump_in_business (X Y : ℝ) (h1 : 0.05 * Y = 0.04 * X) : (X > 0) → (Y > 0) → (X - Y) / X * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_percentage_slump_in_business_l2003_200322


namespace NUMINAMATH_GPT_selection_schemes_count_l2003_200312

theorem selection_schemes_count :
  let total_teachers := 9
  let select_from_total := Nat.choose 9 3
  let select_all_male := Nat.choose 5 3
  let select_all_female := Nat.choose 4 3
  select_from_total - (select_all_male + select_all_female) = 420 := by
    sorry

end NUMINAMATH_GPT_selection_schemes_count_l2003_200312


namespace NUMINAMATH_GPT_employed_male_percent_problem_l2003_200368

noncomputable def employed_percent_population (total_population_employed_percent : ℝ) (employed_females_percent : ℝ) : ℝ :=
  let employed_males_percent := (1 - employed_females_percent) * total_population_employed_percent
  employed_males_percent

theorem employed_male_percent_problem :
  employed_percent_population 0.72 0.50 = 0.36 := by
  sorry

end NUMINAMATH_GPT_employed_male_percent_problem_l2003_200368


namespace NUMINAMATH_GPT_solve_inequalities_l2003_200329

theorem solve_inequalities (x : ℝ) (h₁ : (x - 1) / 2 < 2 * x + 1) (h₂ : -3 * (1 - x) ≥ -4) : x ≥ -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequalities_l2003_200329


namespace NUMINAMATH_GPT_scientific_notation_of_508_billion_yuan_l2003_200370

-- Definition for a billion in the international system.
def billion : ℝ := 10^9

-- The amount of money given in the problem.
def amount_in_billion (n : ℝ) : ℝ := n * billion

-- The Lean theorem statement to prove.
theorem scientific_notation_of_508_billion_yuan :
  amount_in_billion 508 = 5.08 * 10^11 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_508_billion_yuan_l2003_200370


namespace NUMINAMATH_GPT_smallest_n_divisible_by_100_million_l2003_200333

noncomputable def common_ratio (a1 a2 : ℚ) : ℚ := a2 / a1

noncomputable def nth_term (a1 r : ℚ) (n : ℕ) : ℚ := a1 * r^(n - 1)

theorem smallest_n_divisible_by_100_million :
  ∀ (a1 a2 : ℚ), a1 = 5/6 → a2 = 25 → 
  ∃ n : ℕ, nth_term a1 (common_ratio a1 a2) n % 100000000 = 0 ∧ n = 9 :=
by
  intros a1 a2 h1 h2
  have r := common_ratio a1 a2
  have a9 := nth_term a1 r 9
  sorry

end NUMINAMATH_GPT_smallest_n_divisible_by_100_million_l2003_200333


namespace NUMINAMATH_GPT_annual_cost_l2003_200308

def monday_miles : ℕ := 50
def wednesday_miles : ℕ := 50
def friday_miles : ℕ := 50
def sunday_miles : ℕ := 50

def tuesday_miles : ℕ := 100
def thursday_miles : ℕ := 100
def saturday_miles : ℕ := 100

def cost_per_mile : ℝ := 0.1
def weekly_fee : ℝ := 100
def weeks_in_year : ℕ := 52

noncomputable def total_weekly_miles : ℕ := 
  (monday_miles + wednesday_miles + friday_miles + sunday_miles) * 1 +
  (tuesday_miles + thursday_miles + saturday_miles) * 1

noncomputable def weekly_mileage_cost : ℝ := total_weekly_miles * cost_per_mile

noncomputable def weekly_total_cost : ℝ := weekly_fee + weekly_mileage_cost

noncomputable def annual_total_cost : ℝ := weekly_total_cost * weeks_in_year

theorem annual_cost (monday_miles wednesday_miles friday_miles sunday_miles
                     tuesday_miles thursday_miles saturday_miles : ℕ)
                     (cost_per_mile weekly_fee : ℝ) 
                     (weeks_in_year : ℕ) :
  monday_miles = 50 → wednesday_miles = 50 → friday_miles = 50 → sunday_miles = 50 →
  tuesday_miles = 100 → thursday_miles = 100 → saturday_miles = 100 →
  cost_per_mile = 0.1 → weekly_fee = 100 → weeks_in_year = 52 →
  annual_total_cost = 7800 :=
by
  intros
  sorry

end NUMINAMATH_GPT_annual_cost_l2003_200308


namespace NUMINAMATH_GPT_no_solution_in_natural_numbers_l2003_200394

theorem no_solution_in_natural_numbers (x y z : ℕ) : 
  (x / y : ℚ) + (y / z : ℚ) + (z / x : ℚ) ≠ 1 := 
by sorry

end NUMINAMATH_GPT_no_solution_in_natural_numbers_l2003_200394


namespace NUMINAMATH_GPT_range_of_m_l2003_200318

noncomputable def f (x : ℝ) : ℝ := 1 + Real.sin (2 * x)

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + m

theorem range_of_m (m : ℝ) :
  (∃ x₀ ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x₀ ≥ g x₀ m) → m ≤ Real.sqrt 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_m_l2003_200318


namespace NUMINAMATH_GPT_find_rate_of_stream_l2003_200371

noncomputable def rate_of_stream (v : ℝ) : Prop :=
  let rowing_speed := 36
  let downstream_speed := rowing_speed + v
  let upstream_speed := rowing_speed - v
  (1 / upstream_speed) = 3 * (1 / downstream_speed)

theorem find_rate_of_stream : ∃ v : ℝ, rate_of_stream v ∧ v = 18 :=
by
  use 18
  unfold rate_of_stream
  sorry

end NUMINAMATH_GPT_find_rate_of_stream_l2003_200371


namespace NUMINAMATH_GPT_polygon_sides_l2003_200360

theorem polygon_sides (n : ℕ) 
  (h_interior : (n - 2) * 180 = 4 * 360) : 
  n = 10 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l2003_200360


namespace NUMINAMATH_GPT_complete_residue_system_mod_l2003_200369

open Nat

theorem complete_residue_system_mod (m : ℕ) (x : Fin m → ℕ)
  (h : ∀ i j : Fin m, i ≠ j → ¬ ((x i) % m = (x j) % m)) :
  (Finset.image (λ i => x i % m) (Finset.univ : Finset (Fin m))) = Finset.range m :=
by
  -- Skipping the proof steps.
  sorry

end NUMINAMATH_GPT_complete_residue_system_mod_l2003_200369


namespace NUMINAMATH_GPT_count_adjacent_pairs_sum_multiple_of_three_l2003_200317

def adjacent_digit_sum_multiple_of_three (n : ℕ) : ℕ :=
  -- A function to count the number of pairs with a sum multiple of 3
  sorry

-- Define the sequence from 100 to 999 as digits concatenation
def digit_sequence : List ℕ := List.join (List.map (fun x => x.digits 10) (List.range' 100 900))

theorem count_adjacent_pairs_sum_multiple_of_three :
  adjacent_digit_sum_multiple_of_three digit_sequence.length = 897 :=
sorry

end NUMINAMATH_GPT_count_adjacent_pairs_sum_multiple_of_three_l2003_200317


namespace NUMINAMATH_GPT_problem1_problem2_l2003_200355

-- Define the function f(x)
def f (m x : ℝ) : ℝ := |m * x + 1| + |2 * x - 3|

-- Problem 1: Prove the range of x for f(x) = 4 when m = 2
theorem problem1 (x : ℝ) : f 2 x = 4 ↔ -1 / 2 ≤ x ∧ x ≤ 3 / 2 :=
by
  sorry

-- Problem 2: Prove the range of m given f(1) ≤ (2a^2 + 8) / a for any positive a
theorem problem2 (m : ℝ) (h : ∀ a : ℝ, a > 0 → f m 1 ≤ (2 * a^2 + 8) / a) : -8 ≤ m ∧ m ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2003_200355


namespace NUMINAMATH_GPT_equal_divided_value_l2003_200338

def n : ℕ := 8^2022

theorem equal_divided_value : n / 4 = 4^3032 := 
by {
  -- We state the equivalence and details used in the proof.
  sorry
}

end NUMINAMATH_GPT_equal_divided_value_l2003_200338


namespace NUMINAMATH_GPT_tom_tickets_l2003_200391

theorem tom_tickets :
  (45 + 38 + 52) - (12 + 23) = 100 := by
sorry

end NUMINAMATH_GPT_tom_tickets_l2003_200391


namespace NUMINAMATH_GPT_range_of_dot_product_l2003_200363

theorem range_of_dot_product
  (x y : ℝ)
  (on_ellipse : x^2 / 2 + y^2 = 1) :
  ∃ m n : ℝ, (m = 0) ∧ (n = 1) ∧ m ≤ x^2 / 2 ∧ x^2 / 2 ≤ n :=
sorry

end NUMINAMATH_GPT_range_of_dot_product_l2003_200363


namespace NUMINAMATH_GPT_problem_l2003_200319

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem problem : nabla (nabla 1 3) 2 = 67 :=
by
  sorry

end NUMINAMATH_GPT_problem_l2003_200319


namespace NUMINAMATH_GPT_common_ratio_q_l2003_200323

variable {α : Type*} [LinearOrderedField α]

def geom_seq (a q : α) : ℕ → α
| 0 => a
| n+1 => geom_seq a q n * q

def sum_geom_seq (a q : α) : ℕ → α
| 0 => a
| n+1 => sum_geom_seq a q n + geom_seq a q (n + 1)

theorem common_ratio_q (a q : α) (hq : 0 < q) (h_inc : ∀ n, geom_seq a q n < geom_seq a q (n + 1))
  (h1 : geom_seq a q 1 = 2)
  (h2 : sum_geom_seq a q 2 = 7) :
  q = 2 :=
sorry

end NUMINAMATH_GPT_common_ratio_q_l2003_200323


namespace NUMINAMATH_GPT_minimum_value_of_function_l2003_200342

noncomputable def function_y (x : ℝ) : ℝ := 1 / (Real.sqrt (x - x^2))

theorem minimum_value_of_function : (∀ x : ℝ, 0 < x ∧ x < 1 → function_y x ≥ 2) ∧ (∃ x : ℝ, 0 < x ∧ x < 1 ∧ function_y x = 2) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_function_l2003_200342


namespace NUMINAMATH_GPT_tripod_max_height_l2003_200325

noncomputable def tripod_new_height (original_height : ℝ) (original_leg_length : ℝ) (broken_leg_length : ℝ) : ℝ :=
  (broken_leg_length / original_leg_length) * original_height

theorem tripod_max_height :
  let original_height := 5
  let original_leg_length := 6
  let broken_leg_length := 4
  let h := tripod_new_height original_height original_leg_length broken_leg_length
  h = (10 / 3) :=
by
  sorry

end NUMINAMATH_GPT_tripod_max_height_l2003_200325


namespace NUMINAMATH_GPT_range_of_m_l2003_200306

-- Definitions of the propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, |x| + |x + 1| > m
def q (m : ℝ) : Prop := ∀ x > 2, 2 * x - 2 * m > 0

-- The main theorem statement
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2003_200306


namespace NUMINAMATH_GPT_units_digit_35_pow_7_plus_93_pow_45_l2003_200315

-- Definitions of units digit calculations for the specific values
def units_digit (n : ℕ) : ℕ := n % 10

def units_digit_35_pow_7 : ℕ := units_digit (35 ^ 7)
def units_digit_93_pow_45 : ℕ := units_digit (93 ^ 45)

-- Statement to prove that the sum of the units digits is 8
theorem units_digit_35_pow_7_plus_93_pow_45 : 
  units_digit (35 ^ 7) + units_digit (93 ^ 45) = 8 :=
by 
  sorry -- proof omitted

end NUMINAMATH_GPT_units_digit_35_pow_7_plus_93_pow_45_l2003_200315


namespace NUMINAMATH_GPT_find_number_of_rabbits_l2003_200305

def total_heads (R P : ℕ) : ℕ := R + P
def total_legs (R P : ℕ) : ℕ := 4 * R + 2 * P

theorem find_number_of_rabbits (R P : ℕ)
  (h1 : total_heads R P = 60)
  (h2 : total_legs R P = 192) :
  R = 36 := by
  sorry

end NUMINAMATH_GPT_find_number_of_rabbits_l2003_200305


namespace NUMINAMATH_GPT_problem1_problem2_l2003_200399

-- Define the conditions and the target proofs based on identified questions and answers

-- Problem 1
theorem problem1 (x : ℚ) : 
  9 * (x - 2)^2 ≤ 25 ↔ x = 11 / 3 ∨ x = 1 / 3 :=
sorry

-- Problem 2
theorem problem2 (x y : ℚ) :
  (x + 1) / 3 = 2 * y ∧ 2 * (x + 1) - y = 11 ↔ x = 5 ∧ y = 1 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2003_200399


namespace NUMINAMATH_GPT_cricket_players_count_l2003_200362

theorem cricket_players_count (Hockey Football Softball Total Cricket : ℕ) 
    (hHockey : Hockey = 12)
    (hFootball : Football = 18)
    (hSoftball : Softball = 13)
    (hTotal : Total = 59)
    (hTotalCalculation : Total = Hockey + Football + Softball + Cricket) : 
    Cricket = 16 := by
  sorry

end NUMINAMATH_GPT_cricket_players_count_l2003_200362


namespace NUMINAMATH_GPT_envelope_of_family_of_lines_l2003_200335

theorem envelope_of_family_of_lines (a α : ℝ) (hα : α > 0) :
    ∀ (x y : ℝ), (∃ α > 0,
    (x = a * α / 2 ∧ y = a / (2 * α))) ↔ (x * y = a^2 / 4) := by
  sorry

end NUMINAMATH_GPT_envelope_of_family_of_lines_l2003_200335


namespace NUMINAMATH_GPT_smaller_factor_of_4851_l2003_200348

-- Define the condition
def product_lim (m n : ℕ) : Prop := m * n = 4851 ∧ 10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100

-- The lean theorem statement
theorem smaller_factor_of_4851 : ∃ m n : ℕ, product_lim m n ∧ m = 49 := 
by {
    sorry
}

end NUMINAMATH_GPT_smaller_factor_of_4851_l2003_200348


namespace NUMINAMATH_GPT_num_pairs_equals_one_l2003_200365

noncomputable def fractional_part (x : ℚ) : ℚ := x - x.floor

open BigOperators

theorem num_pairs_equals_one :
  ∃! (n : ℕ) (q : ℚ), 
    (0 < q ∧ q < 2000) ∧ 
    ¬ q.isInt ∧ 
    fractional_part (q^2) = fractional_part (n.choose 2000)
:= sorry

end NUMINAMATH_GPT_num_pairs_equals_one_l2003_200365


namespace NUMINAMATH_GPT_square_and_product_l2003_200384

theorem square_and_product (x : ℤ) (h : x^2 = 1764) : (x = 42) ∧ ((x + 2) * (x - 2) = 1760) :=
by
  sorry

end NUMINAMATH_GPT_square_and_product_l2003_200384


namespace NUMINAMATH_GPT_polynomial_remainder_l2003_200390

theorem polynomial_remainder (z : ℂ) :
  let dividend := 4*z^3 - 5*z^2 - 17*z + 4
  let divisor := 4*z + 6
  let quotient := z^2 - 4*z + (1/4 : ℝ)
  let remainder := 5*z^2 + 6*z + (5/2 : ℝ)
  dividend = divisor * quotient + remainder := sorry

end NUMINAMATH_GPT_polynomial_remainder_l2003_200390


namespace NUMINAMATH_GPT_part_one_part_two_l2003_200309

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem part_one (a : ℝ) (h_pos : 0 < a) :
  (∀ x > 0, (Real.log x + 1/x + 1 - a) ≥ 0) ↔ (0 < a ∧ a ≤ 2) :=
sorry

theorem part_two (a : ℝ) (h_pos : 0 < a) :
  (∀ x, (x - 1) * (f x a) ≥ 0) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l2003_200309


namespace NUMINAMATH_GPT_distance_from_desk_to_fountain_l2003_200353

-- Problem definitions with given conditions
def total_distance : ℕ := 120
def trips : ℕ := 4

-- Formulate the proof problem as a Lean theorem statement
theorem distance_from_desk_to_fountain :
  total_distance / trips = 30 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_desk_to_fountain_l2003_200353


namespace NUMINAMATH_GPT_complement_of_A_in_U_l2003_200385

theorem complement_of_A_in_U :
    ∀ (U A : Set ℕ),
    U = {1, 2, 3, 4} →
    A = {1, 3} →
    (U \ A) = {2, 4} :=
by
  intros U A hU hA
  rw [hU, hA]
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l2003_200385


namespace NUMINAMATH_GPT_white_line_longer_l2003_200389

-- Define the lengths of the white and blue lines
def white_line_length : ℝ := 7.678934
def blue_line_length : ℝ := 3.33457689

-- State the main theorem
theorem white_line_longer :
  white_line_length - blue_line_length = 4.34435711 :=
by
  sorry

end NUMINAMATH_GPT_white_line_longer_l2003_200389


namespace NUMINAMATH_GPT_problem_solution_l2003_200358

noncomputable def arithmetic_sequences
    (a : ℕ → ℚ) (b : ℕ → ℚ)
    (Sn : ℕ → ℚ) (Tn : ℕ → ℚ) : Prop :=
  (∀ n : ℕ, Sn n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) ∧
  (∀ n : ℕ, Tn n = n / 2 * (2 * b 1 + (n - 1) * (b 2 - b 1))) ∧
  (∀ n : ℕ, Sn n / Tn n = (2 * n - 3) / (4 * n - 3))

theorem problem_solution
    (a : ℕ → ℚ) (b : ℕ → ℚ) (Sn : ℕ → ℚ) (Tn : ℕ → ℚ)
    (h_arith : arithmetic_sequences a b Sn Tn) :
    (a 9 / (b 5 + b 7)) + (a 3 / (b 8 + b 4)) = 19 / 41 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2003_200358


namespace NUMINAMATH_GPT_weight_of_new_person_l2003_200396

-- Define the given conditions
variables (avg_increase : ℝ) (num_people : ℕ) (replaced_weight : ℝ)
variable (new_weight : ℝ)

-- These are the conditions directly from the problem
axiom avg_weight_increase : avg_increase = 4.5
axiom number_of_people : num_people = 6
axiom person_to_replace_weight : replaced_weight = 75

-- Mathematical equivalent of the proof problem
theorem weight_of_new_person :
  new_weight = replaced_weight + avg_increase * num_people := 
sorry

end NUMINAMATH_GPT_weight_of_new_person_l2003_200396


namespace NUMINAMATH_GPT_find_value_of_b_l2003_200354

theorem find_value_of_b (a b : ℕ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 2) : b = 2 :=
sorry

end NUMINAMATH_GPT_find_value_of_b_l2003_200354


namespace NUMINAMATH_GPT_car_production_l2003_200387

theorem car_production (mp : ℕ) (h1 : 1800 = (mp + 50) * 12) : mp = 100 :=
by
  sorry

end NUMINAMATH_GPT_car_production_l2003_200387


namespace NUMINAMATH_GPT_evaluate_g_f_l2003_200339

def f (a b : ℤ) : ℤ × ℤ := (-a, b)

def g (m n : ℤ) : ℤ × ℤ := (m, -n)

theorem evaluate_g_f : g (f 2 (-3)).1 (f 2 (-3)).2 = (-2, 3) := by
  sorry

end NUMINAMATH_GPT_evaluate_g_f_l2003_200339


namespace NUMINAMATH_GPT_negation_of_existence_l2003_200331

theorem negation_of_existence :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_l2003_200331


namespace NUMINAMATH_GPT_Sarah_pool_depth_l2003_200314

theorem Sarah_pool_depth (S J : ℝ) (h1 : J = 2 * S + 5) (h2 : J = 15) : S = 5 := by
  sorry

end NUMINAMATH_GPT_Sarah_pool_depth_l2003_200314


namespace NUMINAMATH_GPT_complement_correct_l2003_200382

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 5}
def complement_U (M: Set ℕ) (U: Set ℕ) := {x ∈ U | x ∉ M}

theorem complement_correct : complement_U M U = {3, 4, 6} :=
by 
  sorry

end NUMINAMATH_GPT_complement_correct_l2003_200382


namespace NUMINAMATH_GPT_time_brushing_each_cat_l2003_200343

theorem time_brushing_each_cat :
  ∀ (t_total_free_time t_vacuum t_dust t_mop t_free_left_after_cleaning t_cats : ℕ),
  t_total_free_time = 3 * 60 →
  t_vacuum = 45 →
  t_dust = 60 →
  t_mop = 30 →
  t_cats = 3 →
  t_free_left_after_cleaning = 30 →
  ((t_total_free_time - t_free_left_after_cleaning) - (t_vacuum + t_dust + t_mop)) / t_cats = 5
 := by
  intros t_total_free_time t_vacuum t_dust t_mop t_free_left_after_cleaning t_cats
  intros h_total_free_time h_vacuum h_dust h_mop h_cats h_free_left
  sorry

end NUMINAMATH_GPT_time_brushing_each_cat_l2003_200343


namespace NUMINAMATH_GPT_highest_elevation_l2003_200321

   noncomputable def elevation (t : ℝ) : ℝ := 240 * t - 24 * t^2

   theorem highest_elevation : ∃ t : ℝ, elevation t = 600 ∧ ∀ x : ℝ, elevation x ≤ 600 := 
   sorry
   
end NUMINAMATH_GPT_highest_elevation_l2003_200321


namespace NUMINAMATH_GPT_common_tangent_lines_count_l2003_200395

-- Define the first circle
def C1 (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 9

-- Define the second circle
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y - 9 = 0

-- Definition for the number of common tangent lines between two circles
def number_of_common_tangent_lines (C1 C2 : ℝ → ℝ → Prop) : ℕ := sorry

-- The theorem stating the number of common tangent lines between the given circles
theorem common_tangent_lines_count : number_of_common_tangent_lines C1 C2 = 2 := by
  sorry

end NUMINAMATH_GPT_common_tangent_lines_count_l2003_200395


namespace NUMINAMATH_GPT_second_friend_shells_l2003_200361

theorem second_friend_shells (initial_shells : ℕ) (first_friend_shells : ℕ) (total_shells : ℕ) (second_friend_shells : ℕ) :
  initial_shells = 5 → first_friend_shells = 15 → total_shells = 37 → initial_shells + first_friend_shells + second_friend_shells = total_shells → second_friend_shells = 17 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  linarith

end NUMINAMATH_GPT_second_friend_shells_l2003_200361


namespace NUMINAMATH_GPT_find_number_l2003_200316

theorem find_number (x : ℝ) (h : x / 3 = x - 4) : x = 6 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l2003_200316


namespace NUMINAMATH_GPT_sum_reciprocal_squares_l2003_200376

open Real

theorem sum_reciprocal_squares (a : ℝ) (A B C D E F : ℝ)
    (square_ABCD : A = 0 ∧ B = a ∧ D = a ∧ C = a)
    (line_intersects : A = 0 ∧ E ≥ 0 ∧ E ≤ a ∧ F ≥ 0 ∧ F ≤ a) 
    (phi : ℝ) : 
    (cos phi * (a/cos phi))^2 + (sin phi * (a/sin phi))^2 = (1/a^2) := 
sorry 

end NUMINAMATH_GPT_sum_reciprocal_squares_l2003_200376


namespace NUMINAMATH_GPT_transformation_composition_l2003_200352

def f (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def g (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.1 - p.2)

theorem transformation_composition :
  f (g (-1, 2)) = (1, -3) :=
by {
  sorry
}

end NUMINAMATH_GPT_transformation_composition_l2003_200352


namespace NUMINAMATH_GPT_shirts_total_cost_l2003_200356

def shirt_cost_problem : Prop :=
  ∃ (first_shirt_cost second_shirt_cost total_cost : ℕ),
    first_shirt_cost = 15 ∧
    first_shirt_cost = second_shirt_cost + 6 ∧
    total_cost = first_shirt_cost + second_shirt_cost ∧
    total_cost = 24

theorem shirts_total_cost : shirt_cost_problem := by
  sorry

end NUMINAMATH_GPT_shirts_total_cost_l2003_200356


namespace NUMINAMATH_GPT_license_plate_count_l2003_200330

theorem license_plate_count : (26^3 * 5 * 5 * 4) = 1757600 := 
by 
  sorry

end NUMINAMATH_GPT_license_plate_count_l2003_200330


namespace NUMINAMATH_GPT_number_of_men_l2003_200347

theorem number_of_men (M W : ℕ) (h1 : W = 2) (h2 : ∃k, k = 4) : M = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_men_l2003_200347


namespace NUMINAMATH_GPT_ellas_coins_worth_l2003_200364

theorem ellas_coins_worth :
  ∀ (n d : ℕ), n + d = 18 → n = d + 2 → 5 * n + 10 * d = 130 := by
  intros n d h1 h2
  sorry

end NUMINAMATH_GPT_ellas_coins_worth_l2003_200364


namespace NUMINAMATH_GPT_volleyball_club_members_l2003_200332

variables (B G : ℝ)

theorem volleyball_club_members (h1 : B + G = 30) (h2 : 1 / 3 * G + B = 18) : B = 12 := by
  -- Mathematical steps and transformations done here to show B = 12
  sorry

end NUMINAMATH_GPT_volleyball_club_members_l2003_200332


namespace NUMINAMATH_GPT_point_B_in_third_quadrant_l2003_200377

theorem point_B_in_third_quadrant (x y : ℝ) (hx : x < 0) (hy : y < 1) :
    (y - 1 < 0) ∧ (x < 0) :=
by
  sorry  -- proof to be filled

end NUMINAMATH_GPT_point_B_in_third_quadrant_l2003_200377


namespace NUMINAMATH_GPT_probability_of_region_C_l2003_200388

theorem probability_of_region_C (P_A P_B P_C : ℚ) (hA : P_A = 1/3) (hB : P_B = 1/2) (hTotal : P_A + P_B + P_C = 1) : P_C = 1/6 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_region_C_l2003_200388


namespace NUMINAMATH_GPT_frac_mul_eq_l2003_200380

theorem frac_mul_eq : (2/3) * (3/8) = 1/4 := 
by 
  sorry

end NUMINAMATH_GPT_frac_mul_eq_l2003_200380


namespace NUMINAMATH_GPT_cost_prices_l2003_200341

variable {C1 C2 : ℝ}

theorem cost_prices (h1 : 0.30 * C1 - 0.15 * C1 = 120) (h2 : 0.25 * C2 - 0.10 * C2 = 150) :
  C1 = 800 ∧ C2 = 1000 := 
by
  sorry

end NUMINAMATH_GPT_cost_prices_l2003_200341


namespace NUMINAMATH_GPT_rate_of_interest_l2003_200378

theorem rate_of_interest (P T SI CI : ℝ) (hP : P = 4000) (hT : T = 2) (hSI : SI = 400) (hCI : CI = 410) :
  ∃ r : ℝ, SI = (P * r * T) / 100 ∧ CI = P * ((1 + r / 100) ^ T - 1) ∧ r = 5 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_interest_l2003_200378


namespace NUMINAMATH_GPT_kendra_total_earnings_l2003_200393

-- Definitions of the conditions based on the problem statement
def kendra_earnings_2014 : ℕ := 30000 - 8000
def laurel_earnings_2014 : ℕ := 30000
def kendra_earnings_2015 : ℕ := laurel_earnings_2014 + (laurel_earnings_2014 / 5)

-- The statement to be proved
theorem kendra_total_earnings : kendra_earnings_2014 + kendra_earnings_2015 = 58000 :=
by
  -- Using Lean tactics for the proof
  sorry

end NUMINAMATH_GPT_kendra_total_earnings_l2003_200393


namespace NUMINAMATH_GPT_jogging_track_circumference_l2003_200386

theorem jogging_track_circumference
  (Deepak_speed : ℝ)
  (Wife_speed : ℝ)
  (meet_time_minutes : ℝ)
  (H_deepak_speed : Deepak_speed = 4.5)
  (H_wife_speed : Wife_speed = 3.75)
  (H_meet_time_minutes : meet_time_minutes = 3.84) :
  let meet_time_hours := meet_time_minutes / 60
  let distance_deepak := Deepak_speed * meet_time_hours
  let distance_wife := Wife_speed * meet_time_hours
  let total_distance := distance_deepak + distance_wife
  let circumference := 2 * total_distance
  circumference = 1.056 :=
by
  sorry

end NUMINAMATH_GPT_jogging_track_circumference_l2003_200386


namespace NUMINAMATH_GPT_peanuts_added_l2003_200392

theorem peanuts_added (initial final added : ℕ) (h1 : initial = 4) (h2 : final = 8) (h3 : final = initial + added) : added = 4 :=
by
  rw [h1] at h3
  rw [h2] at h3
  sorry

end NUMINAMATH_GPT_peanuts_added_l2003_200392


namespace NUMINAMATH_GPT_percent_of_srp_bob_paid_l2003_200340

theorem percent_of_srp_bob_paid (SRP MP PriceBobPaid : ℝ) 
  (h1 : MP = 0.60 * SRP)
  (h2 : PriceBobPaid = 0.60 * MP) :
  (PriceBobPaid / SRP) * 100 = 36 := by
  sorry

end NUMINAMATH_GPT_percent_of_srp_bob_paid_l2003_200340


namespace NUMINAMATH_GPT_find_q_l2003_200302

noncomputable def p (q : ℝ) : ℝ := 16 / (3 * q)

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 3/2) (h4 : p * q = 16/3) : q = 24 / 6 + 19.6 / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l2003_200302


namespace NUMINAMATH_GPT_harry_did_not_get_an_A_l2003_200374

theorem harry_did_not_get_an_A
  (emily_Imp_frank : Prop)
  (frank_Imp_gina : Prop)
  (gina_Imp_harry : Prop)
  (exactly_one_did_not_get_an_A : ¬ (emily_Imp_frank ∧ frank_Imp_gina ∧ gina_Imp_harry)) :
  ¬ harry_Imp_gina :=
  sorry

end NUMINAMATH_GPT_harry_did_not_get_an_A_l2003_200374


namespace NUMINAMATH_GPT_equivalent_proof_problem_l2003_200307

-- Define the conditions as Lean 4 definitions
variable (x₁ x₂ : ℝ)

-- The conditions given in the problem
def condition1 : Prop := x₁ * Real.logb 2 x₁ = 1008
def condition2 : Prop := x₂ * 2^x₂ = 1008

-- The problem to be proved
theorem equivalent_proof_problem (hx₁ : condition1 x₁) (hx₂ : condition2 x₂) : 
  x₁ * x₂ = 1008 := 
sorry

end NUMINAMATH_GPT_equivalent_proof_problem_l2003_200307
