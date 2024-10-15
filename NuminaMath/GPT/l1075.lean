import Mathlib

namespace NUMINAMATH_GPT_curved_surface_area_of_sphere_l1075_107519

theorem curved_surface_area_of_sphere (r : ℝ) (h : r = 4) : 4 * π * r^2 = 64 * π :=
by
  rw [h, sq]
  norm_num
  sorry

end NUMINAMATH_GPT_curved_surface_area_of_sphere_l1075_107519


namespace NUMINAMATH_GPT_molecular_weight_C7H6O2_l1075_107599

noncomputable def molecular_weight_one_mole (w_9moles : ℕ) (m_9moles : ℕ) : ℕ :=
  m_9moles / w_9moles

theorem molecular_weight_C7H6O2 :
  molecular_weight_one_mole 9 1098 = 122 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_C7H6O2_l1075_107599


namespace NUMINAMATH_GPT_intersection_A_B_l1075_107506

def A : Set ℝ := { x | |x - 1| < 2 }
def B : Set ℝ := { x | Real.log x / Real.log 2 ≤ 1 }

theorem intersection_A_B :
  A ∩ B = {x | 0 < x ∧ x ≤ 2} := 
sorry

end NUMINAMATH_GPT_intersection_A_B_l1075_107506


namespace NUMINAMATH_GPT_triangle_concurrency_l1075_107598

-- Define Triangle Structure
structure Triangle (α : Type*) :=
(A B C : α)

-- Define Medians, Angle Bisectors, and Altitudes Concurrency Conditions
noncomputable def medians_concurrent {α : Type*} [MetricSpace α] (T : Triangle α) : Prop := sorry
noncomputable def angle_bisectors_concurrent {α : Type*} [MetricSpace α] (T : Triangle α) : Prop := sorry
noncomputable def altitudes_concurrent {α : Type*} [MetricSpace α] (T : Triangle α) : Prop := sorry

-- Main Theorem Statement
theorem triangle_concurrency {α : Type*} [MetricSpace α] (T : Triangle α) :
  medians_concurrent T ∧ angle_bisectors_concurrent T ∧ altitudes_concurrent T :=
by 
  -- Proof outline: Prove each concurrency condition
  sorry

end NUMINAMATH_GPT_triangle_concurrency_l1075_107598


namespace NUMINAMATH_GPT_price_reduction_l1075_107575

theorem price_reduction (p0 p1 p2 : ℝ) (H0 : p0 = 1) (H1 : p1 = 1.25 * p0) (H2 : p2 = 1.1 * p0) :
  ∃ x : ℝ, p2 = p1 * (1 - x / 100) ∧ x = 12 :=
  sorry

end NUMINAMATH_GPT_price_reduction_l1075_107575


namespace NUMINAMATH_GPT_length_of_arc_l1075_107574

theorem length_of_arc (S : ℝ) (α : ℝ) (hS : S = 4) (hα : α = 2) : 
  ∃ l : ℝ, l = 4 :=
by
  sorry

end NUMINAMATH_GPT_length_of_arc_l1075_107574


namespace NUMINAMATH_GPT_candies_bought_friday_l1075_107528

-- Definitions based on the given conditions
def candies_bought_tuesday : ℕ := 3
def candies_bought_thursday : ℕ := 5
def candies_left (c : ℕ) : Prop := c = 4
def candies_eaten (c : ℕ) : Prop := c = 6

-- Theorem to prove the number of candies bought on Friday
theorem candies_bought_friday (c_left c_eaten : ℕ) (h_left : candies_left c_left) (h_eaten : candies_eaten c_eaten) : 
  (10 - (candies_bought_tuesday + candies_bought_thursday) = 2) :=
  by
    sorry

end NUMINAMATH_GPT_candies_bought_friday_l1075_107528


namespace NUMINAMATH_GPT_total_peaches_in_each_basket_l1075_107557

-- Define the given conditions
def red_peaches : ℕ := 7
def green_peaches : ℕ := 3

-- State the theorem
theorem total_peaches_in_each_basket : red_peaches + green_peaches = 10 :=
by
  -- Proof goes here, which we skip for now
  sorry

end NUMINAMATH_GPT_total_peaches_in_each_basket_l1075_107557


namespace NUMINAMATH_GPT_remainder_correct_l1075_107503

open Polynomial

noncomputable def polynomial_remainder (p q : Polynomial ℝ) : Polynomial ℝ :=
  p % q

theorem remainder_correct : polynomial_remainder (X^6 - 2*X^5 + X^4 - X^2 - 2*X + 1)
                                                  ((X^2 - 1)*(X - 2)*(X + 2))
                                                = 2*X^3 - 9*X^2 + 3*X + 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_correct_l1075_107503


namespace NUMINAMATH_GPT_square_side_length_properties_l1075_107525

theorem square_side_length_properties (a: ℝ) (h: a^2 = 10) :
  a = Real.sqrt 10 ∧ (a^2 - 10 = 0) ∧ (3 < a ∧ a < 4) :=
by
  sorry

end NUMINAMATH_GPT_square_side_length_properties_l1075_107525


namespace NUMINAMATH_GPT_problem_distribution_l1075_107579

theorem problem_distribution:
  let num_problems := 6
  let num_friends := 15
  (num_friends ^ num_problems) = 11390625 :=
by sorry

end NUMINAMATH_GPT_problem_distribution_l1075_107579


namespace NUMINAMATH_GPT_triangle_angles_21_equal_triangles_around_square_l1075_107595

theorem triangle_angles_21_equal_triangles_around_square
    (theta alpha beta gamma : ℝ)
    (h1 : 4 * theta + 90 = 360)
    (h2 : alpha + beta + 90 = 180)
    (h3 : alpha + beta + gamma = 180)
    (h4 : gamma + 90 = 180)
    : theta = 67.5 ∧ alpha = 67.5 ∧ beta = 22.5 ∧ gamma = 90 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angles_21_equal_triangles_around_square_l1075_107595


namespace NUMINAMATH_GPT_smallest_whole_number_inequality_l1075_107535

theorem smallest_whole_number_inequality (x : ℕ) (h : 3 * x + 4 > 11 - 2 * x) : x ≥ 2 :=
sorry

end NUMINAMATH_GPT_smallest_whole_number_inequality_l1075_107535


namespace NUMINAMATH_GPT_right_triangle_area_l1075_107560

theorem right_triangle_area (a b : ℕ) (h1 : a = 36) (h2 : b = 48) : (1 / 2 : ℚ) * (a * b) = 864 := 
by 
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1075_107560


namespace NUMINAMATH_GPT_trig_identity_example_l1075_107553

theorem trig_identity_example :
  (Real.sin (43 * Real.pi / 180) * Real.cos (13 * Real.pi / 180) - Real.sin (13 * Real.pi / 180) * Real.cos (43 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_example_l1075_107553


namespace NUMINAMATH_GPT_coefficients_of_quadratic_function_l1075_107582

-- Define the quadratic function.
def quadratic_function (x : ℝ) : ℝ :=
  2 * (x - 3) ^ 2 + 2

-- Define the expected expanded form.
def expanded_form (x : ℝ) : ℝ :=
  2 * x ^ 2 - 12 * x + 20

-- State the proof problem.
theorem coefficients_of_quadratic_function :
  ∀ (x : ℝ), quadratic_function x = expanded_form x := by
  sorry

end NUMINAMATH_GPT_coefficients_of_quadratic_function_l1075_107582


namespace NUMINAMATH_GPT_range_of_m_l1075_107561

theorem range_of_m (m : ℝ) (x : ℝ) : (∀ x, (1 - m) * x = 2 - 3 * x → x > 0) ↔ m < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1075_107561


namespace NUMINAMATH_GPT_consecutive_ints_square_l1075_107597

theorem consecutive_ints_square (a b : ℤ) (h : b = a + 1) : 
  a^2 + b^2 + (a * b)^2 = (a * b + 1)^2 := 
by sorry

end NUMINAMATH_GPT_consecutive_ints_square_l1075_107597


namespace NUMINAMATH_GPT_intersecting_lines_a_value_l1075_107542

theorem intersecting_lines_a_value :
  ∀ t a b : ℝ, (b = 12) ∧ (b = 2 * a + t) ∧ (t = 4) → a = 4 :=
by
  intros t a b h
  obtain ⟨hb1, hb2, ht⟩ := h
  sorry

end NUMINAMATH_GPT_intersecting_lines_a_value_l1075_107542


namespace NUMINAMATH_GPT_range_of_a_l1075_107522

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ax^2 + ax + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1075_107522


namespace NUMINAMATH_GPT_radio_advertiser_savings_l1075_107566

def total_store_price : ℚ := 299.99
def ad_payment : ℚ := 55.98
def payments_count : ℚ := 5
def shipping_handling : ℚ := 12.99

def total_ad_price : ℚ := payments_count * ad_payment + shipping_handling

def savings_in_dollars : ℚ := total_store_price - total_ad_price
def savings_in_cents : ℚ := savings_in_dollars * 100

theorem radio_advertiser_savings :
  savings_in_cents = 710 := by
  sorry

end NUMINAMATH_GPT_radio_advertiser_savings_l1075_107566


namespace NUMINAMATH_GPT_minimum_total_trips_l1075_107544

theorem minimum_total_trips :
  ∃ (x y : ℕ), (31 * x + 32 * y = 5000) ∧ (x + y = 157) :=
by
  sorry

end NUMINAMATH_GPT_minimum_total_trips_l1075_107544


namespace NUMINAMATH_GPT_number_of_children_l1075_107517
-- Import the entirety of the Mathlib library

-- Define the conditions and the theorem to be proven
theorem number_of_children (C n : ℕ) 
  (h1 : C = 8 * n + 4) 
  (h2 : C = 11 * (n - 1)) : 
  n = 5 :=
by sorry

end NUMINAMATH_GPT_number_of_children_l1075_107517


namespace NUMINAMATH_GPT_determine_b_l1075_107538

def imaginary_unit : Type := {i : ℂ // i^2 = -1}

theorem determine_b (i : imaginary_unit) (b : ℝ) : 
  (2 - i.val) * 4 * i.val = 4 - b * i.val → b = -8 :=
by
  sorry

end NUMINAMATH_GPT_determine_b_l1075_107538


namespace NUMINAMATH_GPT_Exponent_Equality_l1075_107571

theorem Exponent_Equality : 2^8 * 2^32 = 256^5 :=
by
  sorry

end NUMINAMATH_GPT_Exponent_Equality_l1075_107571


namespace NUMINAMATH_GPT_adoption_complete_in_7_days_l1075_107531

-- Define the initial number of puppies
def initial_puppies := 9

-- Define the number of puppies brought in later
def additional_puppies := 12

-- Define the number of puppies adopted per day
def adoption_rate := 3

-- Define the total number of puppies
def total_puppies : Nat := initial_puppies + additional_puppies

-- Define the number of days required to adopt all puppies
def adoption_days : Nat := total_puppies / adoption_rate

-- Prove that the number of days to adopt all puppies is 7
theorem adoption_complete_in_7_days : adoption_days = 7 := by
  -- The exact implementation of the proof is not necessary,
  -- so we use sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_adoption_complete_in_7_days_l1075_107531


namespace NUMINAMATH_GPT_red_peaches_each_basket_l1075_107540

variable (TotalGreenPeachesInABasket : Nat) (TotalPeachesInABasket : Nat)

theorem red_peaches_each_basket (h1 : TotalPeachesInABasket = 10) (h2 : TotalGreenPeachesInABasket = 3) :
  (TotalPeachesInABasket - TotalGreenPeachesInABasket) = 7 := by
  sorry

end NUMINAMATH_GPT_red_peaches_each_basket_l1075_107540


namespace NUMINAMATH_GPT_square_of_1005_l1075_107537

theorem square_of_1005 : (1005 : ℕ)^2 = 1010025 := 
  sorry

end NUMINAMATH_GPT_square_of_1005_l1075_107537


namespace NUMINAMATH_GPT_box_dimensions_l1075_107516

theorem box_dimensions (x : ℝ) (bow_length_top bow_length_side : ℝ)
  (h1 : bow_length_top = 156 - 6 * x)
  (h2 : bow_length_side = 178 - 7 * x)
  (h_eq : bow_length_top = bow_length_side) :
  x = 22 :=
by sorry

end NUMINAMATH_GPT_box_dimensions_l1075_107516


namespace NUMINAMATH_GPT_leibo_orange_price_l1075_107592

variable (x y m : ℝ)

theorem leibo_orange_price :
  (3 * x + 2 * y = 78) ∧ (2 * x + 3 * y = 72) ∧ (18 * m + 12 * (100 - m) ≤ 1440) → (x = 18) ∧ (y = 12) ∧ (m ≤ 40) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_leibo_orange_price_l1075_107592


namespace NUMINAMATH_GPT_garrett_bought_peanut_granola_bars_l1075_107552

def garrett_granola_bars (t o : ℕ) (h_t : t = 14) (h_o : o = 6) : ℕ :=
  t - o

theorem garrett_bought_peanut_granola_bars : garrett_granola_bars 14 6 rfl rfl = 8 :=
  by
    unfold garrett_granola_bars
    rw [Nat.sub_eq_of_eq_add]
    sorry

end NUMINAMATH_GPT_garrett_bought_peanut_granola_bars_l1075_107552


namespace NUMINAMATH_GPT_terminating_decimal_representation_l1075_107505

-- Definitions derived from conditions
def given_fraction : ℚ := 53 / (2^2 * 5^3)

-- The theorem we aim to state that expresses the question and correct answer
theorem terminating_decimal_representation : given_fraction = 0.106 :=
by
  sorry  -- proof goes here

end NUMINAMATH_GPT_terminating_decimal_representation_l1075_107505


namespace NUMINAMATH_GPT_total_ages_l1075_107512

-- Definitions of the conditions
variables (A B : ℕ) (x : ℕ)

-- Condition 1: 10 years ago, A was half of B in age.
def condition1 : Prop := A - 10 = 1/2 * (B - 10)

-- Condition 2: The ratio of their present ages is 3:4.
def condition2 : Prop := A = 3 * x ∧ B = 4 * x

-- Main theorem to prove
theorem total_ages (A B : ℕ) (x : ℕ) (h1 : condition1 A B) (h2 : condition2 A B x) : A + B = 35 := 
by
  sorry

end NUMINAMATH_GPT_total_ages_l1075_107512


namespace NUMINAMATH_GPT_hyperbola_center_is_correct_l1075_107551

theorem hyperbola_center_is_correct :
  ∃ h k : ℝ, (∀ x y : ℝ, ((4 * y + 8)^2 / 16^2) - ((5 * x - 15)^2 / 9^2) = 1 → x - h = 0 ∧ y + k = 0) ∧ h = 3 ∧ k = -2 :=
sorry

end NUMINAMATH_GPT_hyperbola_center_is_correct_l1075_107551


namespace NUMINAMATH_GPT_project_completion_time_l1075_107526

theorem project_completion_time (rate_a rate_b rate_c : ℝ) (total_work : ℝ) (quit_time : ℝ) 
  (ha : rate_a = 1 / 20) 
  (hb : rate_b = 1 / 30) 
  (hc : rate_c = 1 / 40) 
  (htotal : total_work = 1)
  (hquit : quit_time = 18) : 
  ∃ T : ℝ, T = 18 :=
by {
  sorry
}

end NUMINAMATH_GPT_project_completion_time_l1075_107526


namespace NUMINAMATH_GPT_last_digit_base5_89_l1075_107583

theorem last_digit_base5_89 : 
  ∃ (b : ℕ), (89 : ℕ) = b * 5 + 4 :=
by
  -- The theorem above states that there exists an integer b, such that when we compute 89 in base 5, 
  -- its last digit is 4.
  sorry

end NUMINAMATH_GPT_last_digit_base5_89_l1075_107583


namespace NUMINAMATH_GPT_compare_abc_l1075_107576

noncomputable def a : ℝ := (2 / 5) ^ (3 / 5)
noncomputable def b : ℝ := (2 / 5) ^ (2 / 5)
noncomputable def c : ℝ := (3 / 5) ^ (2 / 5)

theorem compare_abc : a < b ∧ b < c := sorry

end NUMINAMATH_GPT_compare_abc_l1075_107576


namespace NUMINAMATH_GPT_six_dice_not_same_probability_l1075_107564

theorem six_dice_not_same_probability :
  let total_outcomes := 6^6
  let all_same := 6
  let probability_all_same := all_same / total_outcomes
  let probability_not_all_same := 1 - probability_all_same
  probability_not_all_same = 7775 / 7776 :=
by
  sorry

end NUMINAMATH_GPT_six_dice_not_same_probability_l1075_107564


namespace NUMINAMATH_GPT_li_family_cinema_cost_l1075_107539

theorem li_family_cinema_cost :
  let standard_ticket_price := 10
  let child_discount := 0.4
  let senior_discount := 0.3
  let handling_fee := 5
  let num_adults := 2
  let num_children := 1
  let num_seniors := 1
  let child_ticket_price := (1 - child_discount) * standard_ticket_price
  let senior_ticket_price := (1 - senior_discount) * standard_ticket_price
  let total_ticket_cost := num_adults * standard_ticket_price + num_children * child_ticket_price + num_seniors * senior_ticket_price
  let final_cost := total_ticket_cost + handling_fee
  final_cost = 38 :=
by
  sorry

end NUMINAMATH_GPT_li_family_cinema_cost_l1075_107539


namespace NUMINAMATH_GPT_students_in_dexters_high_school_l1075_107556

variables (D S N : ℕ)

theorem students_in_dexters_high_school :
  (D = 4 * S) ∧
  (D + S + N = 3600) ∧
  (N = S - 400) →
  D = 8000 / 3 := 
sorry

end NUMINAMATH_GPT_students_in_dexters_high_school_l1075_107556


namespace NUMINAMATH_GPT_original_savings_calculation_l1075_107514

theorem original_savings_calculation (S : ℝ) (F : ℝ) (T : ℝ) 
  (h1 : 0.8 * F = (3 / 4) * S)
  (h2 : 1.1 * T = 150)
  (h3 : (1 / 4) * S = T) :
  S = 545.44 :=
by
  sorry

end NUMINAMATH_GPT_original_savings_calculation_l1075_107514


namespace NUMINAMATH_GPT_symmetry_condition_l1075_107558

theorem symmetry_condition (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - a| = |(2 - x) + 1| + |(2 - x) - a|) ↔ a = 3 :=
by
  sorry

end NUMINAMATH_GPT_symmetry_condition_l1075_107558


namespace NUMINAMATH_GPT_neg_prop_of_exists_x_gt_0_xsq_sub_x_leq_0_l1075_107573

theorem neg_prop_of_exists_x_gt_0_xsq_sub_x_leq_0 :
  ¬ (∃ x : ℝ, x > 0 ∧ x^2 - x ≤ 0) ↔ ∀ x : ℝ, x ≤ 0 → x^2 - x > 0 :=
by
    sorry

end NUMINAMATH_GPT_neg_prop_of_exists_x_gt_0_xsq_sub_x_leq_0_l1075_107573


namespace NUMINAMATH_GPT_round_trip_average_mileage_l1075_107585

theorem round_trip_average_mileage 
  (d1 d2 : ℝ) (m1 m2 : ℝ)
  (h1 : d1 = 150) (h2 : d2 = 150)
  (h3 : m1 = 40) (h4 : m2 = 25) :
  (d1 + d2) / ((d1 / m1) + (d2 / m2)) = 30.77 :=
by
  sorry

end NUMINAMATH_GPT_round_trip_average_mileage_l1075_107585


namespace NUMINAMATH_GPT_interval_is_correct_l1075_107563

def total_population : ℕ := 2000
def sample_size : ℕ := 40
def interval_between_segments (N : ℕ) (n : ℕ) : ℕ := N / n

theorem interval_is_correct : interval_between_segments total_population sample_size = 50 :=
by
  sorry

end NUMINAMATH_GPT_interval_is_correct_l1075_107563


namespace NUMINAMATH_GPT_area_of_rectangle_l1075_107555

noncomputable def area_proof : ℝ :=
  let a := 294
  let b := 147
  let c := 3
  a + b * Real.sqrt c

theorem area_of_rectangle (ABCD : ℝ × ℝ) (E : ℝ) (F : ℝ) (BE : ℝ) (AB' : ℝ) : 
  BE = 21 ∧ BE = 2 * CF → AB' = 7 → 
  (ABCD.1 * ABCD.2 = 294 + 147 * Real.sqrt 3 ∧ (294 + 147 + 3 = 444)) :=
sorry

end NUMINAMATH_GPT_area_of_rectangle_l1075_107555


namespace NUMINAMATH_GPT_nelly_earns_per_night_l1075_107523

/-- 
  Nelly wants to buy pizza for herself and her 14 friends. Each pizza costs $12 and can feed 3 
  people. Nelly has to babysit for 15 nights to afford the pizza. We need to prove that Nelly earns 
  $4 per night babysitting.
--/
theorem nelly_earns_per_night 
  (total_people : ℕ) (people_per_pizza : ℕ) 
  (cost_per_pizza : ℕ) (total_nights : ℕ) (total_cost : ℕ) 
  (total_pizzas : ℕ) (cost_per_night : ℕ)
  (h1 : total_people = 15)
  (h2 : people_per_pizza = 3)
  (h3 : cost_per_pizza = 12)
  (h4 : total_nights = 15)
  (h5 : total_pizzas = total_people / people_per_pizza)
  (h6 : total_cost = total_pizzas * cost_per_pizza)
  (h7 : cost_per_night = total_cost / total_nights) :
  cost_per_night = 4 := sorry

end NUMINAMATH_GPT_nelly_earns_per_night_l1075_107523


namespace NUMINAMATH_GPT_spider_paths_l1075_107591

theorem spider_paths : (Nat.choose (7 + 3) 3) = 210 := 
by
  sorry

end NUMINAMATH_GPT_spider_paths_l1075_107591


namespace NUMINAMATH_GPT_emails_in_afternoon_l1075_107529

variable (e_m e_t e_a : Nat)
variable (h1 : e_m = 3)
variable (h2 : e_t = 8)

theorem emails_in_afternoon : e_a = 5 :=
by
  -- (Proof steps would go here)
  sorry

end NUMINAMATH_GPT_emails_in_afternoon_l1075_107529


namespace NUMINAMATH_GPT_power_of_11_in_expression_l1075_107568

-- Define the mathematical context
def prime_factors_count (n : ℕ) (a b c : ℕ) : ℕ :=
  n + a + b

-- Given conditions
def count_factors_of_2 : ℕ := 22
def count_factors_of_7 : ℕ := 5
def total_prime_factors : ℕ := 29

-- Theorem stating that power of 11 in the expression is 2
theorem power_of_11_in_expression : 
  ∃ n : ℕ, prime_factors_count n count_factors_of_2 count_factors_of_7 = total_prime_factors ∧ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_power_of_11_in_expression_l1075_107568


namespace NUMINAMATH_GPT_problem_statement_l1075_107550

theorem problem_statement (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + 6 = 104 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1075_107550


namespace NUMINAMATH_GPT_Sarahs_score_l1075_107509

theorem Sarahs_score (x g : ℕ) (h1 : g = x - 50) (h2 : (x + g) / 2 = 110) : x = 135 := by 
  sorry

end NUMINAMATH_GPT_Sarahs_score_l1075_107509


namespace NUMINAMATH_GPT_students_failed_l1075_107508

theorem students_failed (Q : ℕ) (x : ℕ) (h1 : 4 * Q < 56) (h2 : x = Nat.lcm 3 (Nat.lcm 7 2)) (h3 : x < 56) :
  let R := x - (x / 3 + x / 7 + x / 2) 
  R = 1 := 
by
  sorry

end NUMINAMATH_GPT_students_failed_l1075_107508


namespace NUMINAMATH_GPT_decrease_angle_equilateral_l1075_107524

theorem decrease_angle_equilateral (D E F : ℝ) (h : D = 60) (h_equilateral : D = E ∧ E = F) (h_decrease : D' = D - 20) :
  ∃ max_angle : ℝ, max_angle = 70 :=
by
  sorry

end NUMINAMATH_GPT_decrease_angle_equilateral_l1075_107524


namespace NUMINAMATH_GPT_floor_problem_2020_l1075_107594

-- Define the problem statement
theorem floor_problem_2020:
  2020 ^ 2021 - (Int.floor ((2020 ^ 2021 : ℝ) / 2021) * 2021) = 2020 :=
sorry

end NUMINAMATH_GPT_floor_problem_2020_l1075_107594


namespace NUMINAMATH_GPT_trajectory_eq_ellipse_l1075_107578

theorem trajectory_eq_ellipse :
  (∀ M : ℝ × ℝ, (∀ r : ℝ, (M.1 - 4)^2 + M.2^2 = r^2 ∧ (M.1 + 4)^2 + M.2^2 = (10 - r)^2) → false) →
  ∀ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1) :=
by
  sorry

end NUMINAMATH_GPT_trajectory_eq_ellipse_l1075_107578


namespace NUMINAMATH_GPT_find_a_l1075_107534

theorem find_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a = 0 ↔ 3 * x^4 - 48 = 0) → a = 4 :=
  by
    intros h
    sorry

end NUMINAMATH_GPT_find_a_l1075_107534


namespace NUMINAMATH_GPT_winnie_keeps_lollipops_l1075_107543

theorem winnie_keeps_lollipops :
  let cherry := 36
  let wintergreen := 125
  let grape := 8
  let shrimp_cocktail := 241
  let total_lollipops := cherry + wintergreen + grape + shrimp_cocktail
  let friends := 13
  total_lollipops % friends = 7 :=
by
  sorry

end NUMINAMATH_GPT_winnie_keeps_lollipops_l1075_107543


namespace NUMINAMATH_GPT_sum_of_altitudes_l1075_107584

theorem sum_of_altitudes (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : a^2 + b^2 = c^2) : a + b = 21 :=
by
  -- Using the provided hypotheses, the proof would ensure a + b = 21.
  sorry

end NUMINAMATH_GPT_sum_of_altitudes_l1075_107584


namespace NUMINAMATH_GPT_product_12_3460_l1075_107549

theorem product_12_3460 : 12 * 3460 = 41520 :=
by
  sorry

end NUMINAMATH_GPT_product_12_3460_l1075_107549


namespace NUMINAMATH_GPT_sum_of_primes_less_than_20_is_77_l1075_107515

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_primes_less_than_20_is_77_l1075_107515


namespace NUMINAMATH_GPT_initial_investment_l1075_107562

variable (P1 P2 π1 π2 : ℝ)

-- Given conditions
axiom h1 : π1 = 100
axiom h2 : π2 = 120

-- Revenue relation after the first transaction
axiom h3 : P2 = P1 + π1

-- Consistent profit relationship across transactions
axiom h4 : π2 = 0.2 * P2

-- To be proved
theorem initial_investment (P1 : ℝ) (h1 : π1 = 100) (h2 : π2 = 120) (h3 : P2 = P1 + π1) (h4 : π2 = 0.2 * P2) :
  P1 = 500 :=
sorry

end NUMINAMATH_GPT_initial_investment_l1075_107562


namespace NUMINAMATH_GPT_problem_1_problem_2_l1075_107533

-- Definitions of the given probabilities
def prob_A : ℚ := 2/3
def prob_B : ℚ := 1/4
def prob_C : ℚ := 2/5

-- Independence implies that the probabilities of combined events are products of individual probabilities.
-- To avoid unnecessary complications, we assume independence holds true without proof.
axiom independence : ∀ A B C : Prop, (A ∧ B ∧ C) ↔ (A ∧ B) ∧ C

-- Problem statement for part (1)
theorem problem_1 : prob_A * prob_B * prob_C = 1/15 := by
  sorry

-- Helper definitions for probabilities of not visiting
def not_prob_A : ℚ := 1 - prob_A
def not_prob_B : ℚ := 1 - prob_B
def not_prob_C : ℚ := 1 - prob_C

-- Problem statement for part (2)
theorem problem_2 : (prob_A * not_prob_B * not_prob_C + not_prob_A * prob_B * not_prob_C + not_prob_A * not_prob_B * prob_C) = 9/20 := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1075_107533


namespace NUMINAMATH_GPT_simplify_expression_l1075_107518

theorem simplify_expression : (225 / 10125) * 45 = 1 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1075_107518


namespace NUMINAMATH_GPT_rectangle_perimeter_l1075_107513

noncomputable def perimeter_rectangle (x y : ℝ) : ℝ := 2 * (x + y)

theorem rectangle_perimeter
  (x y a b : ℝ)
  (H1 : x * y = 2006)
  (H2 : x + y = 2 * a)
  (H3 : x^2 + y^2 = 4 * (a^2 - b^2))
  (b_val : b = Real.sqrt 1003)
  (a_val : a = 2 * Real.sqrt 1003) :
  perimeter_rectangle x y = 8 * Real.sqrt 1003 := by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1075_107513


namespace NUMINAMATH_GPT_budget_left_equals_16_l1075_107527

def initial_budget : ℤ := 200
def expense_shirt : ℤ := 30
def expense_pants : ℤ := 46
def expense_coat : ℤ := 38
def expense_socks : ℤ := 11
def expense_belt : ℤ := 18
def expense_shoes : ℤ := 41

def total_expenses : ℤ := 
  expense_shirt + expense_pants + expense_coat + expense_socks + expense_belt + expense_shoes

def budget_left : ℤ := initial_budget - total_expenses

theorem budget_left_equals_16 : 
  budget_left = 16 := by
  sorry

end NUMINAMATH_GPT_budget_left_equals_16_l1075_107527


namespace NUMINAMATH_GPT_f_monotonic_intervals_f_extreme_values_l1075_107559

def f (x : ℝ) : ℝ := x^3 - 12 * x

-- Monotonicity intervals
theorem f_monotonic_intervals (x : ℝ) : 
  (x < -2 → deriv f x > 0) ∧ 
  (-2 < x ∧ x < 2 → deriv f x < 0) ∧ 
  (2 < x → deriv f x > 0) := 
sorry

-- Extreme values
theorem f_extreme_values :
  f (-2) = 16 ∧ f (2) = -16 :=
sorry

end NUMINAMATH_GPT_f_monotonic_intervals_f_extreme_values_l1075_107559


namespace NUMINAMATH_GPT_min_value_expression_l1075_107541

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  8 * a^3 + 6 * b^3 + 27 * c^3 + 9 / (8 * a * b * c) ≥ 18 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1075_107541


namespace NUMINAMATH_GPT_trigonometric_identity_l1075_107567

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ ∈ Set.Ico 0 Real.pi) (hθ2 : Real.cos θ * (Real.sin θ + Real.cos θ) = 1) :
  θ = 0 ∨ θ = Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1075_107567


namespace NUMINAMATH_GPT_solution_set_inequality_l1075_107536

theorem solution_set_inequality (x : ℝ) : ((x - 1) * (x + 2) < 0) ↔ (-2 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1075_107536


namespace NUMINAMATH_GPT_smallest_common_multiple_l1075_107588

theorem smallest_common_multiple : Nat.lcm 18 35 = 630 := by
  sorry

end NUMINAMATH_GPT_smallest_common_multiple_l1075_107588


namespace NUMINAMATH_GPT_balance_scale_with_blue_balls_l1075_107507

variables (G Y W B : ℝ)

-- Conditions
def green_to_blue := 4 * G = 8 * B
def yellow_to_blue := 3 * Y = 8 * B
def white_to_blue := 5 * B = 3 * W

-- Proof problem statement
theorem balance_scale_with_blue_balls (h1 : green_to_blue G B) (h2 : yellow_to_blue Y B) (h3 : white_to_blue W B) : 
  3 * G + 3 * Y + 3 * W = 19 * B :=
by sorry

end NUMINAMATH_GPT_balance_scale_with_blue_balls_l1075_107507


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1075_107580

variable {θ m : ℝ}
variable {h₀ : θ ∈ Ioo 0 (Real.pi / 2)}
variable {h₁ : Real.sin θ + Real.cos θ = (Real.sqrt 3 + 1) / 2}
variable {h₂ : Real.sin θ * Real.cos θ = m / 2}

theorem problem_part1 :
  (Real.sin θ / (1 - 1 / Real.tan θ) + Real.cos θ / (1 - Real.tan θ)) = (Real.sqrt 3 + 1) / 2 :=
sorry

theorem problem_part2 :
  m = Real.sqrt 3 / 2 ∧ (θ = Real.pi / 6 ∨ θ = Real.pi / 3) :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1075_107580


namespace NUMINAMATH_GPT_distinct_divisors_in_set_l1075_107502

theorem distinct_divisors_in_set (p : ℕ) (hp : Nat.Prime p) (hp5 : 5 < p) :
  ∃ (x y : ℕ), x ∈ {p - n^2 | n : ℕ} ∧ y ∈ {p - n^2 | n : ℕ} ∧ x ≠ y ∧ x ≠ 1 ∧ x ∣ y :=
by
  sorry

end NUMINAMATH_GPT_distinct_divisors_in_set_l1075_107502


namespace NUMINAMATH_GPT_cos_pi_minus_alpha_correct_l1075_107570

noncomputable def cos_pi_minus_alpha (α : ℝ) (P : ℝ × ℝ) : ℝ :=
  let x := P.1
  let y := P.2
  let h := Real.sqrt (x^2 + y^2)
  let cos_alpha := x / h
  let cos_pi_minus_alpha := -cos_alpha
  cos_pi_minus_alpha

theorem cos_pi_minus_alpha_correct :
  cos_pi_minus_alpha α (-1, 2) = Real.sqrt 5 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_pi_minus_alpha_correct_l1075_107570


namespace NUMINAMATH_GPT_Quentin_chickens_l1075_107511

variable (C S Q : ℕ)

theorem Quentin_chickens (h1 : C = 37)
    (h2 : S = 3 * C - 4)
    (h3 : Q + S + C = 383) :
    (Q = 2 * S + 32) :=
by
  sorry

end NUMINAMATH_GPT_Quentin_chickens_l1075_107511


namespace NUMINAMATH_GPT_range_of_a_l1075_107565

noncomputable def f (a x : ℝ) : ℝ := - (1 / 3) * x^3 + (1 / 2) * x^2 + 2 * a * x

theorem range_of_a (a : ℝ) :
  (∀ x > (2 / 3), (deriv (f a)) x > 0) → a > -(1 / 9) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1075_107565


namespace NUMINAMATH_GPT_fraction_playing_in_field_l1075_107545

def class_size : ℕ := 50
def students_painting : ℚ := 3/5
def students_left_in_classroom : ℕ := 10

theorem fraction_playing_in_field :
  (class_size - students_left_in_classroom - students_painting * class_size) / class_size = 1/5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_playing_in_field_l1075_107545


namespace NUMINAMATH_GPT_Tyler_CDs_after_giveaway_and_purchase_l1075_107572

theorem Tyler_CDs_after_giveaway_and_purchase :
  (∃ cds_initial cds_giveaway_fraction cds_bought cds_final, 
     cds_initial = 21 ∧ 
     cds_giveaway_fraction = 1 / 3 ∧ 
     cds_bought = 8 ∧ 
     cds_final = cds_initial - (cds_initial * cds_giveaway_fraction) + cds_bought ∧
     cds_final = 22) := 
sorry

end NUMINAMATH_GPT_Tyler_CDs_after_giveaway_and_purchase_l1075_107572


namespace NUMINAMATH_GPT_simplify_expression_l1075_107500

theorem simplify_expression : 5 * (18 / -9) * (24 / 36) = -(20 / 3) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1075_107500


namespace NUMINAMATH_GPT_shaded_area_of_rotated_semicircle_l1075_107521

noncomputable def area_of_shaded_region (R : ℝ) (α : ℝ) : ℝ :=
  (1 / 2) * (2 * R) ^ 2 * (α / (2 * Real.pi))

theorem shaded_area_of_rotated_semicircle (R : ℝ) (α : ℝ) (h : α = Real.pi / 9) :
  area_of_shaded_region R α = 2 * Real.pi * R ^ 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_of_rotated_semicircle_l1075_107521


namespace NUMINAMATH_GPT_complement_intersection_l1075_107546

theorem complement_intersection (A B U : Set ℕ) (hA : A = {4, 5, 7}) (hB : B = {3, 4, 7, 8}) (hU : U = A ∪ B) :
  U \ (A ∩ B) = {3, 5, 8} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1075_107546


namespace NUMINAMATH_GPT_beyonce_total_songs_l1075_107510

theorem beyonce_total_songs :
  let singles := 12
  let albums := 4
  let songs_per_album := 18 + 14
  let total_album_songs := albums * songs_per_album
  let total_songs := total_album_songs + singles
  total_songs = 140 := by
  let singles := 12
  let albums := 4
  let songs_per_album := 18 + 14
  let total_album_songs := albums * songs_per_album
  let total_songs := total_album_songs + singles
  sorry

end NUMINAMATH_GPT_beyonce_total_songs_l1075_107510


namespace NUMINAMATH_GPT_number_of_white_balls_l1075_107586

theorem number_of_white_balls (x : ℕ) : (3 : ℕ) + x = 12 → x = 9 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_number_of_white_balls_l1075_107586


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l1075_107532

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α * Real.cos α > 0) (h2 : Real.sin α * Real.tan α < 0) : 
  (π < α ∧ α < 3 * π / 2) :=
by
  sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l1075_107532


namespace NUMINAMATH_GPT_students_in_all_three_workshops_l1075_107596

-- Define the students counts and other conditions
def num_students : ℕ := 25
def num_dance : ℕ := 12
def num_chess : ℕ := 15
def num_robotics : ℕ := 11
def num_at_least_two : ℕ := 12

-- Define the proof statement
theorem students_in_all_three_workshops : 
  ∃ c : ℕ, c = 1 ∧ 
    (∃ a b d : ℕ, 
      a + b + c + d = num_at_least_two ∧
      num_students ≥ num_dance + num_chess + num_robotics - a - b - d - 2 * c
    ) := 
by
  sorry

end NUMINAMATH_GPT_students_in_all_three_workshops_l1075_107596


namespace NUMINAMATH_GPT_sin_2017pi_over_6_l1075_107554

theorem sin_2017pi_over_6 : Real.sin (2017 * Real.pi / 6) = 1 / 2 := 
by 
  -- Proof to be filled in later
  sorry

end NUMINAMATH_GPT_sin_2017pi_over_6_l1075_107554


namespace NUMINAMATH_GPT_family_children_count_l1075_107547

theorem family_children_count (x y : ℕ) 
  (sister_condition : x = y - 1) 
  (brother_condition : y = 2 * (x - 1)) : 
  x + y = 7 := 
sorry

end NUMINAMATH_GPT_family_children_count_l1075_107547


namespace NUMINAMATH_GPT_total_strings_needed_l1075_107530

def basses := 3
def strings_per_bass := 4
def guitars := 2 * basses
def strings_per_guitar := 6
def eight_string_guitars := guitars - 3
def strings_per_eight_string_guitar := 8

theorem total_strings_needed :
  (basses * strings_per_bass) + (guitars * strings_per_guitar) + (eight_string_guitars * strings_per_eight_string_guitar) = 72 := by
  sorry

end NUMINAMATH_GPT_total_strings_needed_l1075_107530


namespace NUMINAMATH_GPT_solve_system_of_equations_l1075_107589

theorem solve_system_of_equations : 
  ∃ (x y : ℚ), 4 * x - 3 * y = -2 ∧ 5 * x + 2 * y = 8 ∧ x = 20 / 23 ∧ y = 42 / 23 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1075_107589


namespace NUMINAMATH_GPT_merill_has_30_marbles_l1075_107593

variable (M E : ℕ)

-- Conditions
def merill_twice_as_many_as_elliot : Prop := M = 2 * E
def together_five_fewer_than_selma : Prop := M + E = 45

theorem merill_has_30_marbles (h1 : merill_twice_as_many_as_elliot M E) (h2 : together_five_fewer_than_selma M E) : M = 30 := 
by
  sorry

end NUMINAMATH_GPT_merill_has_30_marbles_l1075_107593


namespace NUMINAMATH_GPT_smallest_number_is_neg1_l1075_107587

-- Defining the list of numbers
def numbers := [0, -1, 1, 2]

-- Theorem statement to prove that the smallest number in the list is -1
theorem smallest_number_is_neg1 :
  ∀ x ∈ numbers, x ≥ -1 := 
sorry

end NUMINAMATH_GPT_smallest_number_is_neg1_l1075_107587


namespace NUMINAMATH_GPT_professor_D_error_l1075_107581

noncomputable def polynomial_calculation_error (n : ℕ) : Prop :=
  ∃ (f : ℝ → ℝ), (∀ i : ℕ, i ≤ n+1 → f i = 2^i) ∧ f (n+2) ≠ 2^(n+2) - n - 3

theorem professor_D_error (n : ℕ) : polynomial_calculation_error n :=
  sorry

end NUMINAMATH_GPT_professor_D_error_l1075_107581


namespace NUMINAMATH_GPT_multiples_six_or_eight_not_both_l1075_107590

def countMultiples (n m : ℕ) : ℕ := n / m

def LCM (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem multiples_six_or_eight_not_both : 
  let multiplesSix := countMultiples 200 6
  let multiplesEight := countMultiples 200 8
  let commonMultiple := countMultiples 200 (LCM 6 8)
  multiplesSix - commonMultiple + multiplesEight - commonMultiple = 42 := 
by
  sorry

end NUMINAMATH_GPT_multiples_six_or_eight_not_both_l1075_107590


namespace NUMINAMATH_GPT_value_of_a_for_perfect_square_trinomial_l1075_107504

theorem value_of_a_for_perfect_square_trinomial (a : ℝ) (x y : ℝ) :
  (∃ b : ℝ, (x + b * y) ^ 2 = x^2 + a * x * y + y^2) ↔ (a = 2 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_for_perfect_square_trinomial_l1075_107504


namespace NUMINAMATH_GPT_mia_socks_problem_l1075_107520

theorem mia_socks_problem (x y z w : ℕ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) (hw : 1 ≤ w)
  (h1 : x + y + z + w = 16) (h2 : x + 2*y + 3*z + 4*w = 36) : x = 3 :=
sorry

end NUMINAMATH_GPT_mia_socks_problem_l1075_107520


namespace NUMINAMATH_GPT_set_diff_N_M_l1075_107501

universe u

def set_difference {α : Type u} (A B : Set α) : Set α :=
  { x | x ∈ A ∧ x ∉ B }

def M : Set ℕ := { 1, 2, 3, 4, 5 }
def N : Set ℕ := { 1, 2, 3, 7 }

theorem set_diff_N_M : set_difference N M = { 7 } :=
  by
    sorry

end NUMINAMATH_GPT_set_diff_N_M_l1075_107501


namespace NUMINAMATH_GPT_copper_to_zinc_ratio_l1075_107548

theorem copper_to_zinc_ratio (total_weight_brass : ℝ) (weight_zinc : ℝ) (weight_copper : ℝ) 
  (h1 : total_weight_brass = 100) (h2 : weight_zinc = 70) (h3 : weight_copper = total_weight_brass - weight_zinc) : 
  weight_copper / weight_zinc = 3 / 7 :=
by
  sorry

end NUMINAMATH_GPT_copper_to_zinc_ratio_l1075_107548


namespace NUMINAMATH_GPT_shaded_region_area_l1075_107577

noncomputable def area_of_shaded_region (a b c d : ℝ) (area_rect : ℝ) : ℝ :=
  let dg : ℝ := (a * d) / (c + d)
  let area_triangle : ℝ := 0.5 * dg * b
  area_rect - area_triangle

theorem shaded_region_area :
  area_of_shaded_region 12 5 12 4 (4 * 5) = 85 / 8 :=
by
  simp [area_of_shaded_region]
  sorry

end NUMINAMATH_GPT_shaded_region_area_l1075_107577


namespace NUMINAMATH_GPT_geometric_progression_vertex_l1075_107569

theorem geometric_progression_vertex (a b c d : ℝ) (q : ℝ)
  (h1 : b = 1)
  (h2 : c = 2)
  (h3 : q = c / b)
  (h4 : a = b / q)
  (h5 : d = c * q) :
  a + d = 9 / 2 :=
sorry

end NUMINAMATH_GPT_geometric_progression_vertex_l1075_107569
