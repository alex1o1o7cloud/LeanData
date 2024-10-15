import Mathlib

namespace NUMINAMATH_GPT_minimum_value_of_expression_l223_22360

theorem minimum_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l223_22360


namespace NUMINAMATH_GPT_visiting_plans_count_l223_22343

-- Let's define the exhibitions
inductive Exhibition
| OperaCultureExhibition
| MingDynastyImperialCellarPorcelainExhibition
| AncientGreenLandscapePaintingExhibition
| ZhaoMengfuCalligraphyAndPaintingExhibition

open Exhibition

-- The condition is that the student must visit at least one painting exhibition in the morning and another in the afternoon
-- Proof that the number of different visiting plans is 10.
theorem visiting_plans_count :
  let exhibitions := [OperaCultureExhibition, MingDynastyImperialCellarPorcelainExhibition, AncientGreenLandscapePaintingExhibition, ZhaoMengfuCalligraphyAndPaintingExhibition]
  let painting_exhibitions := [AncientGreenLandscapePaintingExhibition, ZhaoMengfuCalligraphyAndPaintingExhibition]
  ∃ visits : List (Exhibition × Exhibition), (∀ (m a : Exhibition), (m ∈ painting_exhibitions ∨ a ∈ painting_exhibitions)) → visits.length = 10 :=
sorry

end NUMINAMATH_GPT_visiting_plans_count_l223_22343


namespace NUMINAMATH_GPT_smallest_non_representable_number_l223_22348

theorem smallest_non_representable_number :
  ∀ n : ℕ, (∀ a b c d : ℕ, n = (2^a - 2^b) / (2^c - 2^d) → n < 11) ∧
           (∀ a b c d : ℕ, 11 ≠ (2^a - 2^b) / (2^c - 2^d)) :=
sorry

end NUMINAMATH_GPT_smallest_non_representable_number_l223_22348


namespace NUMINAMATH_GPT_remainder_div_1442_l223_22396

theorem remainder_div_1442 (x k l r : ℤ) (h1 : 1816 = k * x + 6) (h2 : 1442 = l * x + r) (h3 : x = Int.gcd 1810 374) : r = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_div_1442_l223_22396


namespace NUMINAMATH_GPT_expenditure_on_house_rent_l223_22365

variable (X : ℝ) -- Let X be Bhanu's total income in rupees

-- Condition 1: Bhanu spends 300 rupees on petrol, which is 30% of his income
def condition_on_petrol : Prop := 0.30 * X = 300

-- Definition of remaining income
def remaining_income : ℝ := X - 300

-- Definition of house rent expenditure: 10% of remaining income
def house_rent : ℝ := 0.10 * remaining_income X

-- Theorem: If the condition on petrol holds, then the house rent expenditure is 70 rupees
theorem expenditure_on_house_rent (h : condition_on_petrol X) : house_rent X = 70 :=
  sorry

end NUMINAMATH_GPT_expenditure_on_house_rent_l223_22365


namespace NUMINAMATH_GPT_B_oxen_count_l223_22336

/- 
  A puts 10 oxen for 7 months.
  B puts some oxen for 5 months.
  C puts 15 oxen for 3 months.
  The rent of the pasture is Rs. 175.
  C should pay Rs. 45 as his share of rent.
  We need to prove that B put 12 oxen for grazing.
-/

def oxen_months (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

def A_ox_months := oxen_months 10 7
def C_ox_months := oxen_months 15 3

def total_rent : ℕ := 175
def C_rent_share : ℕ := 45

theorem B_oxen_count (x : ℕ) : 
  (C_rent_share : ℝ) / total_rent = (C_ox_months : ℝ) / (A_ox_months + 5 * x + C_ox_months) →
  x = 12 := 
by
  sorry

end NUMINAMATH_GPT_B_oxen_count_l223_22336


namespace NUMINAMATH_GPT_matrix_eigenvalue_problem_l223_22319

theorem matrix_eigenvalue_problem (k : ℝ) (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  ((3*x + 4*y = k*x) ∧ (6*x + 3*y = k*y)) → k = 3 :=
by
  sorry

end NUMINAMATH_GPT_matrix_eigenvalue_problem_l223_22319


namespace NUMINAMATH_GPT_paper_folding_ratio_l223_22349

theorem paper_folding_ratio :
  ∃ (side length small_perim large_perim : ℕ), 
    side_length = 6 ∧ 
    small_perim = 2 * (3 + 3) ∧ 
    large_perim = 2 * (6 + 3) ∧ 
    small_perim / large_perim = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_paper_folding_ratio_l223_22349


namespace NUMINAMATH_GPT_book_original_selling_price_l223_22306

theorem book_original_selling_price (CP SP1 SP2 : ℝ) 
  (h1 : SP1 = 0.9 * CP)
  (h2 : SP2 = 1.1 * CP)
  (h3 : SP2 = 990) : 
  SP1 = 810 :=
by
  sorry

end NUMINAMATH_GPT_book_original_selling_price_l223_22306


namespace NUMINAMATH_GPT_total_points_other_7_members_is_15_l223_22379

variable (x y : ℕ)
variable (h1 : y ≤ 21)
variable (h2 : y = x * 7 / 15 - 18)
variable (h3 : (1 / 3) * x + (1 / 5) * x + 18 + y = x)

theorem total_points_other_7_members_is_15 (h : x * 7 % 15 = 0) : y = 15 :=
by
  sorry

end NUMINAMATH_GPT_total_points_other_7_members_is_15_l223_22379


namespace NUMINAMATH_GPT_sonya_fell_times_l223_22357

theorem sonya_fell_times (steven_falls : ℕ) (stephanie_falls : ℕ) (sonya_falls : ℕ) :
  steven_falls = 3 →
  stephanie_falls = steven_falls + 13 →
  sonya_falls = 6 →
  sonya_falls = (stephanie_falls / 2) - 2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at *
  sorry

end NUMINAMATH_GPT_sonya_fell_times_l223_22357


namespace NUMINAMATH_GPT_RobertAteNine_l223_22390

-- Define the number of chocolates Nickel ate
def chocolatesNickelAte : ℕ := 2

-- Define the additional chocolates Robert ate compared to Nickel
def additionalChocolates : ℕ := 7

-- Define the total chocolates Robert ate
def chocolatesRobertAte : ℕ := chocolatesNickelAte + additionalChocolates

-- State the theorem we want to prove
theorem RobertAteNine : chocolatesRobertAte = 9 := by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_RobertAteNine_l223_22390


namespace NUMINAMATH_GPT_product_of_integers_l223_22398

theorem product_of_integers (A B C D : ℚ)
  (h1 : A + B + C + D = 100)
  (h2 : A + 5 = B - 5)
  (h3 : A + 5 = 2 * C)
  (h4 : A + 5 = D / 2) :
  A * B * C * D = 1517000000 / 6561 := by
  sorry

end NUMINAMATH_GPT_product_of_integers_l223_22398


namespace NUMINAMATH_GPT_athenas_min_wins_l223_22362

theorem athenas_min_wins (total_games : ℕ) (games_played : ℕ) (wins_so_far : ℕ) (losses_so_far : ℕ) 
                          (win_percentage_threshold : ℝ) (remaining_games : ℕ) (additional_wins_needed : ℕ) :
  total_games = 44 ∧ games_played = wins_so_far + losses_so_far ∧ wins_so_far = 20 ∧ losses_so_far = 15 ∧ 
  win_percentage_threshold = 0.6 ∧ remaining_games = total_games - games_played ∧ additional_wins_needed = 27 - wins_so_far → 
  additional_wins_needed = 7 :=
by
  sorry

end NUMINAMATH_GPT_athenas_min_wins_l223_22362


namespace NUMINAMATH_GPT_set_problems_l223_22369

def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}

theorem set_problems :
  (A ∩ B = ({4} : Set ℤ)) ∧
  (A ∪ B = ({1, 2, 4, 5, 6, 7, 8, 9, 10} : Set ℤ)) ∧
  (U \ (A ∪ B) = ({3} : Set ℤ)) ∧
  ((U \ A) ∩ (U \ B) = ({3} : Set ℤ)) :=
by
  sorry

end NUMINAMATH_GPT_set_problems_l223_22369


namespace NUMINAMATH_GPT_exists_root_f_between_0_and_1_l223_22381

noncomputable def f (x : ℝ) : ℝ := 4 - 4 * x - Real.exp x

theorem exists_root_f_between_0_and_1 :
  ∃ x ∈ Set.Ioo 0 1, f x = 0 :=
sorry

end NUMINAMATH_GPT_exists_root_f_between_0_and_1_l223_22381


namespace NUMINAMATH_GPT_sqrt_mult_minus_two_l223_22326

theorem sqrt_mult_minus_two (x y : ℝ) (hx : x = Real.sqrt 3) (hy : y = Real.sqrt 6) : 
  2 < x * y - 2 ∧ x * y - 2 < 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_mult_minus_two_l223_22326


namespace NUMINAMATH_GPT_sample_size_calculation_l223_22356

theorem sample_size_calculation :
  let workshop_A := 120
  let workshop_B := 80
  let workshop_C := 60
  let sample_from_C := 3
  let sampling_fraction := sample_from_C / workshop_C
  let sample_A := workshop_A * sampling_fraction
  let sample_B := workshop_B * sampling_fraction
  let sample_C := workshop_C * sampling_fraction
  let n := sample_A + sample_B + sample_C
  n = 13 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_calculation_l223_22356


namespace NUMINAMATH_GPT_parallelogram_sides_l223_22397

theorem parallelogram_sides (a b : ℝ)
  (h1 : 2 * (a + b) = 32)
  (h2 : b - a = 8) :
  a = 4 ∧ b = 12 :=
by
  -- Proof is to be provided
  sorry

end NUMINAMATH_GPT_parallelogram_sides_l223_22397


namespace NUMINAMATH_GPT_mean_score_juniors_is_103_l223_22338

noncomputable def mean_score_juniors : Prop :=
  ∃ (students juniors non_juniors m_j m_nj : ℝ),
  students = 160 ∧
  (students * 82) = (juniors * m_j + non_juniors * m_nj) ∧
  juniors = 0.4 * non_juniors ∧
  m_j = 1.4 * m_nj ∧
  m_j = 103

theorem mean_score_juniors_is_103 : mean_score_juniors :=
by
  sorry

end NUMINAMATH_GPT_mean_score_juniors_is_103_l223_22338


namespace NUMINAMATH_GPT_quartic_two_real_roots_l223_22352

theorem quartic_two_real_roots
  (a b c d e : ℝ)
  (h : ∃ β : ℝ, β > 1 ∧ a * β^2 + (c - b) * β + e - d = 0)
  (ha : a ≠ 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^4 + b * x1^3 + c * x1^2 + d * x1 + e = 0) ∧ (a * x2^4 + b * x2^3 + c * x2^2 + d * x2 + e = 0) := 
  sorry

end NUMINAMATH_GPT_quartic_two_real_roots_l223_22352


namespace NUMINAMATH_GPT_no_such_quadratic_exists_l223_22375

theorem no_such_quadratic_exists : ¬ ∃ (b c : ℝ), 
  (∀ x : ℝ, 6 * x ≤ 3 * x^2 + 3 ∧ 3 * x^2 + 3 ≤ x^2 + b * x + c) ∧
  (x^2 + b * x + c = 1) :=
by
  sorry

end NUMINAMATH_GPT_no_such_quadratic_exists_l223_22375


namespace NUMINAMATH_GPT_virginia_taught_fewer_years_l223_22345

variable (V A : ℕ)

theorem virginia_taught_fewer_years (h1 : V + A + 40 = 93) (h2 : V = A + 9) : 40 - V = 9 := by
  sorry

end NUMINAMATH_GPT_virginia_taught_fewer_years_l223_22345


namespace NUMINAMATH_GPT_top_card_yellow_second_card_not_yellow_l223_22324

-- Definitions based on conditions
def total_cards : Nat := 65

def yellow_cards : Nat := 13

def non_yellow_cards : Nat := total_cards - yellow_cards

-- Total combinations of choosing two cards
def total_combinations : Nat := total_cards * (total_cards - 1)

-- Numerator for desired probability 
def desired_combinations : Nat := yellow_cards * non_yellow_cards

-- Target probability
def desired_probability : Rat := Rat.ofInt (desired_combinations) / Rat.ofInt (total_combinations)

-- Mathematical proof statement
theorem top_card_yellow_second_card_not_yellow :
  desired_probability = Rat.ofInt 169 / Rat.ofInt 1040 :=
by
  sorry

end NUMINAMATH_GPT_top_card_yellow_second_card_not_yellow_l223_22324


namespace NUMINAMATH_GPT_correct_insights_l223_22387

def insight1 := ∀ connections : Type, (∃ journey : connections → Prop, ∀ (x : connections), ¬journey x)
def insight2 := ∀ connections : Type, (∃ (beneficial : connections → Prop), ∀ (x : connections), beneficial x → True)
def insight3 := ∀ connections : Type, (∃ (accidental : connections → Prop), ∀ (x : connections), accidental x → False)
def insight4 := ∀ connections : Type, (∃ (conditional : connections → Prop), ∀ (x : connections), conditional x → True)

theorem correct_insights : ¬ insight1 ∧ insight2 ∧ ¬ insight3 ∧ insight4 :=
by sorry

end NUMINAMATH_GPT_correct_insights_l223_22387


namespace NUMINAMATH_GPT_number_with_29_proper_divisors_is_720_l223_22377

theorem number_with_29_proper_divisors_is_720
  (n : ℕ) (h1 : n < 1000)
  (h2 : ∀ d, 1 < d ∧ d < n -> ∃ m, n = d * m):
  n = 720 := by
  sorry

end NUMINAMATH_GPT_number_with_29_proper_divisors_is_720_l223_22377


namespace NUMINAMATH_GPT_mr_ray_customers_without_fish_l223_22305

def mr_ray_num_customers_without_fish
  (total_customers : ℕ)
  (total_tuna_weight : ℕ)
  (specific_customers_30lb : ℕ)
  (specific_weight_30lb : ℕ)
  (specific_customers_20lb : ℕ)
  (specific_weight_20lb : ℕ)
  (weight_per_customer : ℕ)
  (remaining_tuna_weight : ℕ)
  (num_customers_served_with_remaining_tuna : ℕ)
  (total_satisfied_customers : ℕ) : ℕ :=
  total_customers - total_satisfied_customers

theorem mr_ray_customers_without_fish :
  mr_ray_num_customers_without_fish 100 2000 10 30 15 20 25 1400 56 81 = 19 :=
by 
  sorry

end NUMINAMATH_GPT_mr_ray_customers_without_fish_l223_22305


namespace NUMINAMATH_GPT_arithmetic_sequence_third_term_l223_22303

theorem arithmetic_sequence_third_term :
  ∀ (a d : ℤ), (a + 4 * d = 2) ∧ (a + 5 * d = 5) → (a + 2 * d = -4) :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_third_term_l223_22303


namespace NUMINAMATH_GPT_Jack_gave_Mike_six_notebooks_l223_22366

theorem Jack_gave_Mike_six_notebooks :
  ∀ (Gerald_notebooks Jack_notebooks_left notebooks_given_to_Paula total_notebooks_initial jack_notebooks_after_Paula notebooks_given_to_Mike : ℕ),
  Gerald_notebooks = 8 →
  Jack_notebooks_left = 10 →
  notebooks_given_to_Paula = 5 →
  total_notebooks_initial = Gerald_notebooks + 13 →
  jack_notebooks_after_Paula = total_notebooks_initial - notebooks_given_to_Paula →
  notebooks_given_to_Mike = jack_notebooks_after_Paula - Jack_notebooks_left →
  notebooks_given_to_Mike = 6 :=
by
  intros Gerald_notebooks Jack_notebooks_left notebooks_given_to_Paula total_notebooks_initial jack_notebooks_after_Paula notebooks_given_to_Mike
  intros Gerald_notebooks_eq Jack_notebooks_left_eq notebooks_given_to_Paula_eq total_notebooks_initial_eq jack_notebooks_after_Paula_eq notebooks_given_to_Mike_eq
  sorry

end NUMINAMATH_GPT_Jack_gave_Mike_six_notebooks_l223_22366


namespace NUMINAMATH_GPT_min_y_squared_l223_22316

noncomputable def isosceles_trapezoid_bases (EF GH : ℝ) := EF = 102 ∧ GH = 26

noncomputable def trapezoid_sides (EG FH y : ℝ) := EG = y ∧ FH = y

noncomputable def tangent_circle (center_on_EF tangent_to_EG_FH : Prop) := 
  ∃ P : ℝ × ℝ, true -- center P exists somewhere and lies on EF

theorem min_y_squared (EF GH EG FH y : ℝ) (center_on_EF tangent_to_EG_FH : Prop) 
  (h1 : isosceles_trapezoid_bases EF GH)
  (h2 : trapezoid_sides EG FH y)
  (h3 : tangent_circle center_on_EF tangent_to_EG_FH) : 
  ∃ n : ℝ, n^2 = 1938 :=
sorry

end NUMINAMATH_GPT_min_y_squared_l223_22316


namespace NUMINAMATH_GPT_find_m_l223_22323

theorem find_m (m : ℝ) 
  (f g : ℝ → ℝ) 
  (x : ℝ) 
  (hf : f x = x^2 - 3 * x + m) 
  (hg : g x = x^2 - 3 * x + 5 * m) 
  (hx : x = 5) 
  (h_eq : 3 * f x = 2 * g x) :
  m = 10 / 7 := 
sorry

end NUMINAMATH_GPT_find_m_l223_22323


namespace NUMINAMATH_GPT_right_triangle_acute_angle_l223_22313

theorem right_triangle_acute_angle (A B : ℝ) (h₁ : A + B = 90) (h₂ : A = 40) : B = 50 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_acute_angle_l223_22313


namespace NUMINAMATH_GPT_number_of_five_digit_numbers_with_one_odd_digit_l223_22376

def odd_digits : List ℕ := [1, 3, 5, 7, 9]
def even_digits : List ℕ := [0, 2, 4, 6, 8]

def five_digit_numbers_with_one_odd_digit : ℕ :=
  let num_1st_position := odd_digits.length * even_digits.length ^ 4
  let num_other_positions := 4 * odd_digits.length * (even_digits.length - 1) * (even_digits.length ^ 3)
  num_1st_position + num_other_positions

theorem number_of_five_digit_numbers_with_one_odd_digit :
  five_digit_numbers_with_one_odd_digit = 10625 :=
by
  sorry

end NUMINAMATH_GPT_number_of_five_digit_numbers_with_one_odd_digit_l223_22376


namespace NUMINAMATH_GPT_men_in_group_initial_l223_22317

variable (M : ℕ)  -- Initial number of men in the group
variable (A : ℕ)  -- Initial average age of the group

theorem men_in_group_initial : (2 * 50 - (18 + 22) = 60) → ((M + 6) = 60 / 6) → (M = 10) :=
by
  sorry

end NUMINAMATH_GPT_men_in_group_initial_l223_22317


namespace NUMINAMATH_GPT_wanda_walks_days_per_week_l223_22312

theorem wanda_walks_days_per_week 
  (daily_distance : ℝ) (weekly_distance : ℝ) (weeks : ℕ) (total_distance : ℝ) 
  (h_daily_walk: daily_distance = 2) 
  (h_total_walk: total_distance = 40) 
  (h_weeks: weeks = 4) : 
  ∃ d : ℕ, (d * daily_distance * weeks = total_distance) ∧ (d = 5) := 
by 
  sorry

end NUMINAMATH_GPT_wanda_walks_days_per_week_l223_22312


namespace NUMINAMATH_GPT_find_angle_A_l223_22314
open Real

theorem find_angle_A
  (a b : ℝ)
  (A B : ℝ)
  (h1 : b = 2 * a)
  (h2 : B = A + 60) :
  A = 30 :=
by 
  sorry

end NUMINAMATH_GPT_find_angle_A_l223_22314


namespace NUMINAMATH_GPT_probability_bus_there_when_mark_arrives_l223_22391

noncomputable def isProbabilityBusThereWhenMarkArrives : Prop :=
  let busArrival : ℝ := 60 -- The bus can arrive from time 0 to 60 minutes (2:00 PM to 3:00 PM)
  let busWait : ℝ := 30 -- The bus waits for 30 minutes
  let markArrival : ℝ := 90 -- Mark can arrive from time 30 to 90 minutes (2:30 PM to 3:30 PM)
  let overlapArea : ℝ := 1350 -- Total shaded area where bus arrival overlaps with Mark's arrival
  let totalArea : ℝ := busArrival * (markArrival - 30)
  let probability := overlapArea / totalArea
  probability = 1 / 4

theorem probability_bus_there_when_mark_arrives : isProbabilityBusThereWhenMarkArrives :=
by
  sorry

end NUMINAMATH_GPT_probability_bus_there_when_mark_arrives_l223_22391


namespace NUMINAMATH_GPT_fraction_meaningful_condition_l223_22384

-- Define a variable x
variable (x : ℝ)

-- State the condition that makes the fraction meaningful
def fraction_meaningful (x : ℝ) : Prop := (x - 2) ≠ 0

-- State the theorem we want to prove
theorem fraction_meaningful_condition : fraction_meaningful x ↔ x ≠ 2 := sorry

end NUMINAMATH_GPT_fraction_meaningful_condition_l223_22384


namespace NUMINAMATH_GPT_almonds_weight_l223_22399

def nuts_mixture (almonds_ratio walnuts_ratio total_weight : ℚ) : ℚ :=
  let total_parts := almonds_ratio + walnuts_ratio
  let weight_per_part := total_weight / total_parts
  let weight_almonds := weight_per_part * almonds_ratio
  weight_almonds

theorem almonds_weight (total_weight : ℚ) (h1 : total_weight = 140) : nuts_mixture 5 1 total_weight = 116.67 :=
by
  sorry

end NUMINAMATH_GPT_almonds_weight_l223_22399


namespace NUMINAMATH_GPT_catch_two_salmon_l223_22320

def totalTroutWeight : ℕ := 8
def numBass : ℕ := 6
def weightPerBass : ℕ := 2
def totalBassWeight : ℕ := numBass * weightPerBass
def campers : ℕ := 22
def weightPerCamper : ℕ := 2
def totalFishWeightRequired : ℕ := campers * weightPerCamper
def totalTroutAndBassWeight : ℕ := totalTroutWeight + totalBassWeight
def additionalFishWeightRequired : ℕ := totalFishWeightRequired - totalTroutAndBassWeight
def weightPerSalmon : ℕ := 12
def numSalmon : ℕ := additionalFishWeightRequired / weightPerSalmon

theorem catch_two_salmon : numSalmon = 2 := by
  sorry

end NUMINAMATH_GPT_catch_two_salmon_l223_22320


namespace NUMINAMATH_GPT_wholesome_bakery_loaves_on_wednesday_l223_22378

theorem wholesome_bakery_loaves_on_wednesday :
  ∀ (L_wed L_thu L_fri L_sat L_sun L_mon : ℕ),
    L_thu = 7 →
    L_fri = 10 →
    L_sat = 14 →
    L_sun = 19 →
    L_mon = 25 →
    L_thu - L_wed = 2 →
    L_wed = 5 :=
by intros L_wed L_thu L_fri L_sat L_sun L_mon;
   intros H_thu H_fri H_sat H_sun H_mon H_diff;
   sorry

end NUMINAMATH_GPT_wholesome_bakery_loaves_on_wednesday_l223_22378


namespace NUMINAMATH_GPT_complementary_angles_decrease_86_percent_l223_22395

theorem complementary_angles_decrease_86_percent (x : ℝ) (h : 10 * x = 90) :
  let small_angle := 3 * x
  let increased_small_angle := small_angle * 1.2
  let large_angle := 7 * x
  let new_large_angle := 90 - increased_small_angle
  (new_large_angle / large_angle) * 100 = 91.4 :=
by
  sorry

end NUMINAMATH_GPT_complementary_angles_decrease_86_percent_l223_22395


namespace NUMINAMATH_GPT_min_coach_handshakes_l223_22358

-- Definitions based on the problem conditions
def total_gymnasts : ℕ := 26
def total_handshakes : ℕ := 325

/- 
  The main theorem stating that the fewest number of handshakes 
  the coaches could have participated in is 0.
-/
theorem min_coach_handshakes (n : ℕ) (h : 0 ≤ n ∧ n * (n - 1) / 2 = total_handshakes) : 
  n = total_gymnasts → (total_handshakes - n * (n - 1) / 2) = 0 :=
by 
  intros h_n_eq_26
  sorry

end NUMINAMATH_GPT_min_coach_handshakes_l223_22358


namespace NUMINAMATH_GPT_three_digit_number_l223_22374

-- Define the variables involved.
variables (a b c n : ℕ)

-- Condition 1: c = 3a
def condition1 (a c : ℕ) : Prop := c = 3 * a

-- Condition 2: n is three-digit number constructed from a, b, and c.
def is_three_digit (a b c n : ℕ) : Prop := n = 100 * a + 10 * b + c

-- Condition 3: n leaves a remainder of 4 when divided by 5.
def condition2 (n : ℕ) : Prop := n % 5 = 4

-- Condition 4: n leaves a remainder of 3 when divided by 11.
def condition3 (n : ℕ) : Prop := n % 11 = 3

-- Define the main theorem
theorem three_digit_number (a b c n : ℕ) 
(h1: condition1 a c) 
(h2: is_three_digit a b c n) 
(h3: condition2 n) 
(h4: condition3 n) : 
n = 359 := 
sorry

end NUMINAMATH_GPT_three_digit_number_l223_22374


namespace NUMINAMATH_GPT_coeff_abs_sum_eq_729_l223_22322

-- Given polynomial (2x - 1)^6 expansion
theorem coeff_abs_sum_eq_729 (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) :
  (2 * x - 1) ^ 6 = a_6 * x ^ 6 + a_5 * x ^ 5 + a_4 * x ^ 4 + a_3 * x ^ 3 + a_2 * x ^ 2 + a_1 * x + a_0 →
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| = 729 :=
by
  sorry

end NUMINAMATH_GPT_coeff_abs_sum_eq_729_l223_22322


namespace NUMINAMATH_GPT_crackers_eaten_by_Daniel_and_Elsie_l223_22355

theorem crackers_eaten_by_Daniel_and_Elsie :
  ∀ (initial_crackers remaining_crackers eaten_by_Ally eaten_by_Bob eaten_by_Clair: ℝ),
    initial_crackers = 27.5 →
    remaining_crackers = 10.5 →
    eaten_by_Ally = 3.5 →
    eaten_by_Bob = 4.0 →
    eaten_by_Clair = 5.5 →
    initial_crackers - remaining_crackers = (eaten_by_Ally + eaten_by_Bob + eaten_by_Clair) + (4 : ℝ) :=
by sorry

end NUMINAMATH_GPT_crackers_eaten_by_Daniel_and_Elsie_l223_22355


namespace NUMINAMATH_GPT_solve_a_l223_22342

theorem solve_a (a x : ℤ) (h₀ : x = 5) (h₁ : a * x - 8 = 20 + a) : a = 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_a_l223_22342


namespace NUMINAMATH_GPT_females_in_group_l223_22373

theorem females_in_group (n F M : ℕ) (Index_F Index_M : ℝ) 
  (h1 : n = 25) 
  (h2 : Index_F = (n - F) / n)
  (h3 : Index_M = (n - M) / n) 
  (h4 : Index_F - Index_M = 0.36) :
  F = 8 := 
by
  sorry

end NUMINAMATH_GPT_females_in_group_l223_22373


namespace NUMINAMATH_GPT_reciprocal_of_neg_2023_l223_22325

theorem reciprocal_of_neg_2023 : 1 / (-2023) = - (1 / 2023) :=
by 
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_2023_l223_22325


namespace NUMINAMATH_GPT_total_packs_equiv_117_l223_22370

theorem total_packs_equiv_117 
  (nancy_cards : ℕ)
  (melanie_cards : ℕ)
  (mary_cards : ℕ)
  (alyssa_cards : ℕ)
  (nancy_pack : ℝ)
  (melanie_pack : ℝ)
  (mary_pack : ℝ)
  (alyssa_pack : ℝ)
  (H_nancy : nancy_cards = 540)
  (H_melanie : melanie_cards = 620)
  (H_mary : mary_cards = 480)
  (H_alyssa : alyssa_cards = 720)
  (H_nancy_pack : nancy_pack = 18.5)
  (H_melanie_pack : melanie_pack = 22.5)
  (H_mary_pack : mary_pack = 15.3)
  (H_alyssa_pack : alyssa_pack = 24) :
  (⌊nancy_cards / nancy_pack⌋₊ + ⌊melanie_cards / melanie_pack⌋₊ + ⌊mary_cards / mary_pack⌋₊ + ⌊alyssa_cards / alyssa_pack⌋₊) = 117 :=
by
  sorry

end NUMINAMATH_GPT_total_packs_equiv_117_l223_22370


namespace NUMINAMATH_GPT_intersection_S_T_l223_22344

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end NUMINAMATH_GPT_intersection_S_T_l223_22344


namespace NUMINAMATH_GPT_nested_sqrt_eq_two_l223_22392

theorem nested_sqrt_eq_two (x : ℝ) (h : x = Real.sqrt (2 + x)) : x = 2 :=
sorry

end NUMINAMATH_GPT_nested_sqrt_eq_two_l223_22392


namespace NUMINAMATH_GPT_rectangle_area_invariant_l223_22394

theorem rectangle_area_invariant
    (x y : ℕ)
    (h1 : x * y = (x + 3) * (y - 1))
    (h2 : x * y = (x - 3) * (y + 2)) :
    x * y = 15 :=
by sorry

end NUMINAMATH_GPT_rectangle_area_invariant_l223_22394


namespace NUMINAMATH_GPT_bill_money_left_l223_22340

def bill_remaining_money (merchantA_qty : Int) (merchantA_rate : Int) 
                        (merchantB_qty : Int) (merchantB_rate : Int)
                        (fine : Int) (merchantC_qty : Int) (merchantC_rate : Int) 
                        (protection_costs : Int) (passerby_qty : Int) 
                        (passerby_rate : Int) : Int :=
let incomeA := merchantA_qty * merchantA_rate
let incomeB := merchantB_qty * merchantB_rate
let incomeC := merchantC_qty * merchantC_rate
let incomeD := passerby_qty * passerby_rate
let total_income := incomeA + incomeB + incomeC + incomeD
let total_expenses := fine + protection_costs
total_income - total_expenses

theorem bill_money_left 
    (merchantA_qty : Int := 8) 
    (merchantA_rate : Int := 9) 
    (merchantB_qty : Int := 15) 
    (merchantB_rate : Int := 11) 
    (fine : Int := 80)
    (merchantC_qty : Int := 25) 
    (merchantC_rate : Int := 8) 
    (protection_costs : Int := 30) 
    (passerby_qty : Int := 12) 
    (passerby_rate : Int := 7) : 
    bill_remaining_money merchantA_qty merchantA_rate 
                         merchantB_qty merchantB_rate 
                         fine merchantC_qty merchantC_rate 
                         protection_costs passerby_qty 
                         passerby_rate = 411 := by 
  sorry

end NUMINAMATH_GPT_bill_money_left_l223_22340


namespace NUMINAMATH_GPT_ratio_of_radii_of_touching_circles_l223_22339

theorem ratio_of_radii_of_touching_circles
  (r R : ℝ) (A B C D : ℝ) (h1 : A + B + C = D)
  (h2 : 3 * A = 7 * B)
  (h3 : 7 * B = 2 * C)
  (h4 : R = D / 2)
  (h5 : B = R - 3 * A)
  (h6 : C = R - 2 * A)
  (h7 : r = 4 * A)
  (h8 : R = 6 * A) :
  R / r = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_radii_of_touching_circles_l223_22339


namespace NUMINAMATH_GPT_max_xy_value_l223_22382

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 140) : xy ≤ 168 :=
by sorry

end NUMINAMATH_GPT_max_xy_value_l223_22382


namespace NUMINAMATH_GPT_expected_number_of_digits_is_1_55_l223_22346

/-- Brent rolls a fair icosahedral die with numbers 1 through 20 on its faces -/
noncomputable def expectedNumberOfDigits : ℚ :=
  let P_one_digit := 9 / 20
  let P_two_digit := 11 / 20
  (P_one_digit * 1) + (P_two_digit * 2)

/-- The expected number of digits Brent will roll is 1.55 -/
theorem expected_number_of_digits_is_1_55 : expectedNumberOfDigits = 1.55 := by
  sorry

end NUMINAMATH_GPT_expected_number_of_digits_is_1_55_l223_22346


namespace NUMINAMATH_GPT_negative_large_base_zero_exponent_l223_22386

-- Define the problem conditions: base number and exponent
def base_number : ℤ := -2023
def exponent : ℕ := 0

-- Prove that (-2023)^0 equals 1
theorem negative_large_base_zero_exponent : base_number ^ exponent = 1 := by
  sorry

end NUMINAMATH_GPT_negative_large_base_zero_exponent_l223_22386


namespace NUMINAMATH_GPT_square_area_divided_into_equal_rectangles_l223_22302

theorem square_area_divided_into_equal_rectangles (w : ℝ) (a : ℝ) (h : 5 = w) :
  (∃ s : ℝ, s * s = a ∧ s * s / 5 = a / 5) ↔ a = 400 :=
by
  sorry

end NUMINAMATH_GPT_square_area_divided_into_equal_rectangles_l223_22302


namespace NUMINAMATH_GPT_proof_problem_l223_22329

open Real

-- Definitions
noncomputable def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Conditions
def eccentricity (c a : ℝ) : Prop :=
  c / a = (sqrt 2) / 2

def min_distance_to_focus (a c : ℝ) : Prop :=
  a - c = sqrt 2 - 1

-- Proof problem statement
theorem proof_problem (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (b_lt_a : b < a)
  (ecc : eccentricity c a) (min_dist : min_distance_to_focus a c)
  (x y k m : ℝ) (line_condition : y = k * x + m) :
  ellipse_equation x y a b → ellipse_equation x y (sqrt 2) 1 ∧
  (parabola_equation x y → (y = sqrt 2 / 2 * x + sqrt 2 ∨ y = -sqrt 2 / 2 * x - sqrt 2)) :=
sorry

end NUMINAMATH_GPT_proof_problem_l223_22329


namespace NUMINAMATH_GPT_paint_more_expensive_than_wallpaper_l223_22388

variable (x y z : ℝ)
variable (h : 4 * x + 4 * y = 7 * x + 2 * y + z)

theorem paint_more_expensive_than_wallpaper : y > x :=
by
  sorry

end NUMINAMATH_GPT_paint_more_expensive_than_wallpaper_l223_22388


namespace NUMINAMATH_GPT_mike_notebooks_total_l223_22333

theorem mike_notebooks_total
  (red_notebooks : ℕ)
  (green_notebooks : ℕ)
  (blue_notebooks_cost : ℕ)
  (total_cost : ℕ)
  (red_cost : ℕ)
  (green_cost : ℕ)
  (blue_cost : ℕ)
  (h1 : red_notebooks = 3)
  (h2 : red_cost = 4)
  (h3 : green_notebooks = 2)
  (h4 : green_cost = 2)
  (h5 : total_cost = 37)
  (h6 : blue_cost = 3)
  (h7 : total_cost = red_notebooks * red_cost + green_notebooks * green_cost + blue_notebooks_cost) :
  (red_notebooks + green_notebooks + blue_notebooks_cost / blue_cost = 12) :=
by {
  sorry
}

end NUMINAMATH_GPT_mike_notebooks_total_l223_22333


namespace NUMINAMATH_GPT_range_m_distinct_roots_l223_22327

theorem range_m_distinct_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (4^x₁ - m * 2^(x₁+1) + 2 - m = 0) ∧ (4^x₂ - m * 2^(x₂+1) + 2 - m = 0)) ↔ 1 < m ∧ m < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_m_distinct_roots_l223_22327


namespace NUMINAMATH_GPT_vector_sum_is_zero_l223_22372

variables {V : Type*} [AddCommGroup V]

variables (AB CF BC FA : V)

-- Condition: Vectors form a closed polygon
def vectors_form_closed_polygon (AB CF BC FA : V) : Prop :=
  AB + BC + CF + FA = 0

theorem vector_sum_is_zero
  (h : vectors_form_closed_polygon AB CF BC FA) :
  AB + BC + CF + FA = 0 :=
  h

end NUMINAMATH_GPT_vector_sum_is_zero_l223_22372


namespace NUMINAMATH_GPT_employee_selection_l223_22318

theorem employee_selection
  (total_employees : ℕ)
  (under_35 : ℕ)
  (between_35_and_49 : ℕ)
  (over_50 : ℕ)
  (selected_employees : ℕ) :
  total_employees = 500 →
  under_35 = 125 →
  between_35_and_49 = 280 →
  over_50 = 95 →
  selected_employees = 100 →
  (under_35 * selected_employees / total_employees = 25) ∧
  (between_35_and_49 * selected_employees / total_employees = 56) ∧
  (over_50 * selected_employees / total_employees = 19) := by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_employee_selection_l223_22318


namespace NUMINAMATH_GPT_sport_formulation_water_l223_22330

theorem sport_formulation_water
  (f c w : ℕ)  -- flavoring, corn syrup, and water respectively in standard formulation
  (f_s c_s w_s : ℕ)  -- flavoring, corn syrup, and water respectively in sport formulation
  (corn_syrup_sport : ℤ) -- amount of corn syrup in sport formulation in ounces
  (h_std_ratio : f = 1 ∧ c = 12 ∧ w = 30) -- given standard formulation ratios
  (h_sport_fc_ratio : f_s * 4 = c_s) -- sport formulation flavoring to corn syrup ratio
  (h_sport_fw_ratio : f_s * 60 = w_s) -- sport formulation flavoring to water ratio
  (h_corn_syrup_sport : c_s = corn_syrup_sport) -- amount of corn syrup in sport formulation
  : w_s = 30 := 
by 
  sorry

end NUMINAMATH_GPT_sport_formulation_water_l223_22330


namespace NUMINAMATH_GPT_four_squares_cover_larger_square_l223_22393

structure Square :=
  (side : ℝ) (h_positive : side > 0)

theorem four_squares_cover_larger_square (large small : Square) 
  (h_side_relation: large.side = 2 * small.side) : 
  large.side^2 = 4 * small.side^2 :=
by
  sorry

end NUMINAMATH_GPT_four_squares_cover_larger_square_l223_22393


namespace NUMINAMATH_GPT_cost_per_person_l223_22334

def total_cost : ℕ := 30000  -- Cost in million dollars
def num_people : ℕ := 300    -- Number of people in million

theorem cost_per_person : total_cost / num_people = 100 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_person_l223_22334


namespace NUMINAMATH_GPT_orchard_problem_l223_22359

theorem orchard_problem (number_of_peach_trees number_of_apple_trees : ℕ) 
  (h1 : number_of_apple_trees = number_of_peach_trees + 1700)
  (h2 : number_of_apple_trees = 3 * number_of_peach_trees + 200) :
  number_of_peach_trees = 750 ∧ number_of_apple_trees = 2450 :=
by
  sorry

end NUMINAMATH_GPT_orchard_problem_l223_22359


namespace NUMINAMATH_GPT_range_of_a_l223_22301

theorem range_of_a (a : ℝ) : (∀ x, 1 ≤ x ∧ x ≤ 3 → x^2 - a * x - 3 ≤ 0) ↔ (2 ≤ a) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l223_22301


namespace NUMINAMATH_GPT_no_nonzero_real_solutions_l223_22309

theorem no_nonzero_real_solutions (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  ¬ (2 / x + 3 / y = 1 / (x + y)) :=
by sorry

end NUMINAMATH_GPT_no_nonzero_real_solutions_l223_22309


namespace NUMINAMATH_GPT_sum_of_coefficients_l223_22321

theorem sum_of_coefficients :
  ∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ,
  (∀ x : ℤ, (2 - x)^7 = a₀ + a₁ * (1 + x)^2 + a₂ * (1 + x)^3 + a₃ * (1 + x)^4 + a₄ * (1 + x)^5 + a₅ * (1 + x)^6 + a₆ * (1 + x)^7 + a₇ * (1 + x)^8) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 129 := by sorry

end NUMINAMATH_GPT_sum_of_coefficients_l223_22321


namespace NUMINAMATH_GPT_susan_average_speed_l223_22380

noncomputable def average_speed_trip (d1 d2 : ℝ) (v1 v2 : ℝ) : ℝ :=
  let total_distance := d1 + d2
  let time1 := d1 / v1
  let time2 := d2 / v2
  let total_time := time1 + time2
  total_distance / total_time

theorem susan_average_speed :
  average_speed_trip 60 30 30 60 = 36 := 
by
  -- The proof can be filled in here
  sorry

end NUMINAMATH_GPT_susan_average_speed_l223_22380


namespace NUMINAMATH_GPT_subset_A_B_l223_22363

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_A_B (a : ℝ) : A a ⊆ B a ↔ a = 1 := by
  sorry

end NUMINAMATH_GPT_subset_A_B_l223_22363


namespace NUMINAMATH_GPT_rabbit_calories_l223_22361

theorem rabbit_calories (C : ℕ) :
  (6 * 300 = 2 * C + 200) → C = 800 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_rabbit_calories_l223_22361


namespace NUMINAMATH_GPT_average_length_correct_l223_22310

-- Given lengths of the two pieces
def length1 : ℕ := 2
def length2 : ℕ := 6

-- Define the average length
def average_length (l1 l2 : ℕ) : ℕ := (l1 + l2) / 2

-- State the theorem to prove
theorem average_length_correct : average_length length1 length2 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_average_length_correct_l223_22310


namespace NUMINAMATH_GPT_qs_length_l223_22304

theorem qs_length
  (PQR : Triangle)
  (PQ QR PR : ℝ)
  (h1 : PQ = 7)
  (h2 : QR = 8)
  (h3 : PR = 9)
  (bugs_meet_half_perimeter : PQ + QR + PR = 24)
  (bugs_meet_distance : PQ + qs = 12) :
  qs = 5 :=
by
  sorry

end NUMINAMATH_GPT_qs_length_l223_22304


namespace NUMINAMATH_GPT_OilBillJanuary_l223_22341

theorem OilBillJanuary (J F : ℝ) (h1 : F / J = 5 / 4) (h2 : (F + 30) / J = 3 / 2) : J = 120 := by
  sorry

end NUMINAMATH_GPT_OilBillJanuary_l223_22341


namespace NUMINAMATH_GPT_smaller_odd_number_l223_22308

theorem smaller_odd_number (n : ℤ) (h : n + (n + 2) = 48) : n = 23 :=
by
  sorry

end NUMINAMATH_GPT_smaller_odd_number_l223_22308


namespace NUMINAMATH_GPT_rate_at_which_bowls_were_bought_l223_22385

theorem rate_at_which_bowls_were_bought 
    (total_bowls : ℕ) (sold_bowls : ℕ) (price_per_sold_bowl : ℝ) (remaining_bowls : ℕ) (percentage_gain : ℝ) 
    (total_bowls_eq : total_bowls = 115) 
    (sold_bowls_eq : sold_bowls = 104) 
    (price_per_sold_bowl_eq : price_per_sold_bowl = 20) 
    (remaining_bowls_eq : remaining_bowls = 11) 
    (percentage_gain_eq : percentage_gain = 0.4830917874396135) 
  : ∃ (R : ℝ), R = 18 :=
  sorry

end NUMINAMATH_GPT_rate_at_which_bowls_were_bought_l223_22385


namespace NUMINAMATH_GPT_eggs_leftover_l223_22328

theorem eggs_leftover (d e f : ℕ) (total_eggs_per_carton : ℕ) 
  (h_d : d = 53) (h_e : e = 65) (h_f : f = 26) (h_carton : total_eggs_per_carton = 15) : (d + e + f) % total_eggs_per_carton = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_eggs_leftover_l223_22328


namespace NUMINAMATH_GPT_unique_point_intersection_l223_22315

theorem unique_point_intersection (k : ℝ) :
  (∃ x y, y = k * x + 2 ∧ y ^ 2 = 8 * x) → 
  ((k = 0) ∨ (k = 1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_unique_point_intersection_l223_22315


namespace NUMINAMATH_GPT_total_balloons_are_48_l223_22307

theorem total_balloons_are_48 
  (brooke_initial : ℕ) (brooke_add : ℕ) (tracy_initial : ℕ) (tracy_add : ℕ)
  (brooke_half_given : ℕ) (tracy_third_popped : ℕ) : 
  brooke_initial = 20 →
  brooke_add = 15 →
  tracy_initial = 10 →
  tracy_add = 35 →
  brooke_half_given = (brooke_initial + brooke_add) / 2 →
  tracy_third_popped = (tracy_initial + tracy_add) / 3 →
  (brooke_initial + brooke_add - brooke_half_given) + (tracy_initial + tracy_add - tracy_third_popped) = 48 := 
by
  intros
  sorry

end NUMINAMATH_GPT_total_balloons_are_48_l223_22307


namespace NUMINAMATH_GPT_same_side_of_line_l223_22337

theorem same_side_of_line (a : ℝ) :
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) > 0 ↔ a < -7 ∨ a > 24 :=
by
  sorry

end NUMINAMATH_GPT_same_side_of_line_l223_22337


namespace NUMINAMATH_GPT_kerosene_cost_l223_22371

/-- A dozen eggs cost as much as a pound of rice, a half-liter of kerosene costs as much as 8 eggs,
and each pound of rice costs $0.33. Prove that a liter of kerosene costs 44 cents. -/
theorem kerosene_cost :
  let egg_cost := 0.33 / 12
  let rice_cost := 0.33
  let half_liter_kerosene_cost := 8 * egg_cost
  let liter_kerosene_cost := half_liter_kerosene_cost * 2
  liter_kerosene_cost * 100 = 44 := 
by
  sorry

end NUMINAMATH_GPT_kerosene_cost_l223_22371


namespace NUMINAMATH_GPT_rowing_distance_l223_22389
-- Lean 4 Statement

theorem rowing_distance (v_m v_t D : ℝ) 
  (h1 : D = v_m + v_t)
  (h2 : 30 = 10 * (v_m - v_t))
  (h3 : 30 = 6 * (v_m + v_t)) :
  D = 5 :=
by sorry

end NUMINAMATH_GPT_rowing_distance_l223_22389


namespace NUMINAMATH_GPT_quadratic_b_value_l223_22351

theorem quadratic_b_value {b m : ℝ} (h : ∀ x, x^2 + b * x + 44 = (x + m)^2 + 8) : b = 12 :=
by
  -- hint for proving: expand (x+m)^2 + 8 and equate it with x^2 + bx + 44 to solve for b 
  sorry

end NUMINAMATH_GPT_quadratic_b_value_l223_22351


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l223_22347

theorem hyperbola_asymptotes (a : ℝ) (x y : ℝ) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  (∃ M : ℝ × ℝ, M.1 ^ 2 / a ^ 2 - M.2 ^ 2 = 1 ∧ M.2 ^ 2 = 8 * M.1 ∧ abs (dist M F) = 5) →
  (F.1 = 2 ∧ F.2 = 0) →
  (a = 3 / 5) → 
  (∀ x y : ℝ, (5 * x + 3 * y = 0) ∨ (5 * x - 3 * y = 0)) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l223_22347


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l223_22335

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = a n * q)
  (h0 : a 1 = 2) (h1 : a 4 = 1 / 4) : q = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l223_22335


namespace NUMINAMATH_GPT_min_f_value_inequality_solution_l223_22354

theorem min_f_value (x : ℝ) : |x+7| + |x-1| ≥ 8 := by
  sorry

theorem inequality_solution (x : ℝ) (m : ℝ) (h : m = 8) : |x-3| - 2*x ≤ 2*m - 12 ↔ x ≥ -1/3 := by
  sorry

end NUMINAMATH_GPT_min_f_value_inequality_solution_l223_22354


namespace NUMINAMATH_GPT_platform_length_l223_22350

theorem platform_length
  (train_length : ℕ)
  (time_pole : ℕ)
  (time_platform : ℕ)
  (h_train_length : train_length = 300)
  (h_time_pole : time_pole = 18)
  (h_time_platform : time_platform = 39) :
  ∃ (platform_length : ℕ), platform_length = 350 :=
by
  sorry

end NUMINAMATH_GPT_platform_length_l223_22350


namespace NUMINAMATH_GPT_smallest_n_modulo_l223_22311

theorem smallest_n_modulo :
  ∃ (n : ℕ), 0 < n ∧ 1031 * n % 30 = 1067 * n % 30 ∧ ∀ (m : ℕ), 0 < m ∧ 1031 * m % 30 = 1067 * m % 30 → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_modulo_l223_22311


namespace NUMINAMATH_GPT_area_of_ring_between_concentric_circles_l223_22364

theorem area_of_ring_between_concentric_circles :
  let radius_large := 12
  let radius_small := 7
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  let area_ring := area_large - area_small
  area_ring = 95 * Real.pi :=
by
  let radius_large := 12
  let radius_small := 7
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  let area_ring := area_large - area_small
  show area_ring = 95 * Real.pi
  sorry

end NUMINAMATH_GPT_area_of_ring_between_concentric_circles_l223_22364


namespace NUMINAMATH_GPT_total_jury_duty_days_l223_22332

-- Conditions
def jury_selection_days : ℕ := 2
def trial_multiplier : ℕ := 4
def evidence_review_hours : ℕ := 2
def lunch_hours : ℕ := 1
def trial_session_hours : ℕ := 6
def hours_per_day : ℕ := evidence_review_hours + lunch_hours + trial_session_hours
def deliberation_hours_per_day : ℕ := 14 - 2

def deliberation_first_defendant_days : ℕ := 6
def deliberation_second_defendant_days : ℕ := 4
def deliberation_third_defendant_days : ℕ := 5

def deliberation_first_defendant_total_hours : ℕ := deliberation_first_defendant_days * deliberation_hours_per_day
def deliberation_second_defendant_total_hours : ℕ := deliberation_second_defendant_days * deliberation_hours_per_day
def deliberation_third_defendant_total_hours : ℕ := deliberation_third_defendant_days * deliberation_hours_per_day

def deliberation_days_conversion (total_hours: ℕ) : ℕ := (total_hours + deliberation_hours_per_day - 1) / deliberation_hours_per_day

-- Total days spent
def total_days_spent : ℕ :=
  let trial_days := jury_selection_days * trial_multiplier
  let deliberation_days := deliberation_days_conversion deliberation_first_defendant_total_hours + deliberation_days_conversion deliberation_second_defendant_total_hours + deliberation_days_conversion deliberation_third_defendant_total_hours
  jury_selection_days + trial_days + deliberation_days

#eval total_days_spent -- Expected: 25

theorem total_jury_duty_days : total_days_spent = 25 := by
  sorry

end NUMINAMATH_GPT_total_jury_duty_days_l223_22332


namespace NUMINAMATH_GPT_arrange_numbers_l223_22300

noncomputable def a := (10^100)^10
noncomputable def b := 10^(10^10)
noncomputable def c := Nat.factorial 1000000
noncomputable def d := (Nat.factorial 100)^10

theorem arrange_numbers :
  a < d ∧ d < c ∧ c < b := 
sorry

end NUMINAMATH_GPT_arrange_numbers_l223_22300


namespace NUMINAMATH_GPT_solve_for_m_l223_22353

theorem solve_for_m (n : ℝ) (m : ℝ) (h : 21 * (m + n) + 21 = 21 * (-m + n) + 21) : m = 1 / 2 := 
sorry

end NUMINAMATH_GPT_solve_for_m_l223_22353


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l223_22383

noncomputable def cyclic_sum (f : ℝ → ℝ → ℝ) (x y z : ℝ) : ℝ :=
  f x y + f y z + f z x

theorem cyclic_sum_inequality
  (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hx : x = a + (1 / b) - 1) 
  (hy : y = b + (1 / c) - 1) 
  (hz : z = c + (1 / a) - 1)
  (hpx : x > 0) (hpy : y > 0) (hpz : z > 0) :
  cyclic_sum (fun x y => (x * y) / (Real.sqrt (x * y) + 2)) x y z ≥ 1 :=
sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l223_22383


namespace NUMINAMATH_GPT_training_weeks_l223_22368

variable (adoption_fee training_per_week cert_cost insurance_coverage out_of_pocket : ℕ)
variable (x : ℕ)

def adoption_fee_value : ℕ := 150
def training_per_week_cost : ℕ := 250
def certification_cost_value : ℕ := 3000
def insurance_coverage_percentage : ℕ := 90
def total_out_of_pocket : ℕ := 3450

theorem training_weeks :
  adoption_fee = adoption_fee_value →
  training_per_week = training_per_week_cost →
  cert_cost = certification_cost_value →
  insurance_coverage = insurance_coverage_percentage →
  out_of_pocket = total_out_of_pocket →
  (out_of_pocket = adoption_fee + training_per_week * x + (cert_cost * (100 - insurance_coverage)) / 100) →
  x = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  sorry

end NUMINAMATH_GPT_training_weeks_l223_22368


namespace NUMINAMATH_GPT_wrapping_paper_area_l223_22367

theorem wrapping_paper_area (a : ℝ) (h : ℝ) : h = a ∧ 1 ≥ 0 → 4 * a^2 = 4 * a^2 :=
by sorry

end NUMINAMATH_GPT_wrapping_paper_area_l223_22367


namespace NUMINAMATH_GPT_scientific_notation_10200000_l223_22331

theorem scientific_notation_10200000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 10.2 * 10^7 = a * 10^n := 
sorry

end NUMINAMATH_GPT_scientific_notation_10200000_l223_22331
