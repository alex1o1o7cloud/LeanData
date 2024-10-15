import Mathlib

namespace NUMINAMATH_GPT_find_t_l196_19651

noncomputable def f (x t k : ℝ): ℝ := (1/3) * x^3 - (t/2) * x^2 + k * x

theorem find_t (a b t k : ℝ) (h1 : t > 0) (h2 : k > 0) 
  (h3 : a + b = t) (h4 : a * b = k)
  (h5 : 2 * a = b - 2)
  (h6 : (-2) ^ 2 = a * b) : 
  t = 5 :=
by 
  sorry

end NUMINAMATH_GPT_find_t_l196_19651


namespace NUMINAMATH_GPT_daisies_per_bouquet_is_7_l196_19648

/-
Each bouquet of roses contains 12 roses.
Each bouquet of daisies contains an equal number of daisies.
The flower shop sells 20 bouquets today.
10 of the bouquets are rose bouquets and 10 are daisy bouquets.
The flower shop sold 190 flowers in total today.
-/

def num_daisies_per_bouquet (roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold : ℕ) : ℕ :=
  (total_flowers_sold - total_roses_sold) / bouquets_sold 

theorem daisies_per_bouquet_is_7 :
  ∀ (roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold : ℕ),
  (roses_per_bouquet = 12) →
  (bouquets_sold = 10) →
  (total_roses_sold = bouquets_sold * roses_per_bouquet) →
  (total_flowers_sold = 190) →
  num_daisies_per_bouquet roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold = 7 :=
by
  intros
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_daisies_per_bouquet_is_7_l196_19648


namespace NUMINAMATH_GPT_sun_city_population_greater_than_twice_roseville_l196_19671

-- Conditions
def willowdale_population : ℕ := 2000
def roseville_population : ℕ := 3 * willowdale_population - 500
def sun_city_population : ℕ := 12000

-- Theorem
theorem sun_city_population_greater_than_twice_roseville :
  sun_city_population = 2 * roseville_population + 1000 :=
by
  -- The proof is omitted as per the problem statement
  sorry

end NUMINAMATH_GPT_sun_city_population_greater_than_twice_roseville_l196_19671


namespace NUMINAMATH_GPT_kyle_origami_stars_l196_19656

/-- Kyle bought 2 glass bottles, each can hold 15 origami stars,
    then bought another 3 identical glass bottles.
    Prove that the total number of origami stars needed to fill them is 75. -/
theorem kyle_origami_stars : (2 * 15) + (3 * 15) = 75 := by
  sorry

end NUMINAMATH_GPT_kyle_origami_stars_l196_19656


namespace NUMINAMATH_GPT_derivative_at_pi_over_4_l196_19654

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_at_pi_over_4 : 
  deriv f (Real.pi / 4) = Real.sqrt 2 / 2 + Real.sqrt 2 * Real.pi / 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_derivative_at_pi_over_4_l196_19654


namespace NUMINAMATH_GPT_liters_conversion_hours_to_days_cubic_meters_to_cubic_cm_l196_19668

-- Define the conversions and the corresponding proofs
theorem liters_conversion : 8.32 = 8 + 320 / 1000 := sorry

theorem hours_to_days : 6 = 1 / 4 * 24 := sorry

theorem cubic_meters_to_cubic_cm : 0.75 * 10^6 = 750000 := sorry

end NUMINAMATH_GPT_liters_conversion_hours_to_days_cubic_meters_to_cubic_cm_l196_19668


namespace NUMINAMATH_GPT_two_distinct_solutions_diff_l196_19606

theorem two_distinct_solutions_diff (a b : ℝ) (h1 : a ≠ b) (h2 : a > b)
  (h3 : ∀ x, (x = a ∨ x = b) ↔ (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3) :
  a - b = 3 :=
by
  -- Proof will be provided here.
  sorry

end NUMINAMATH_GPT_two_distinct_solutions_diff_l196_19606


namespace NUMINAMATH_GPT_ratio_of_sums_eq_neg_sqrt_2_l196_19687

open Real

theorem ratio_of_sums_eq_neg_sqrt_2
    (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 6) :
    (x + y) / (x - y) = -Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_ratio_of_sums_eq_neg_sqrt_2_l196_19687


namespace NUMINAMATH_GPT_Cody_age_is_14_l196_19611

variable (CodyGrandmotherAge CodyAge : ℕ)

theorem Cody_age_is_14 (h1 : CodyGrandmotherAge = 6 * CodyAge) (h2 : CodyGrandmotherAge = 84) : CodyAge = 14 := by
  sorry

end NUMINAMATH_GPT_Cody_age_is_14_l196_19611


namespace NUMINAMATH_GPT_ratio_of_area_to_breadth_l196_19645

theorem ratio_of_area_to_breadth (b l A : ℝ) (h₁ : b = 10) (h₂ : l - b = 10) (h₃ : A = l * b) : A / b = 20 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_area_to_breadth_l196_19645


namespace NUMINAMATH_GPT_length_of_garden_side_l196_19625

theorem length_of_garden_side (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 112) (h2 : perimeter = 4 * side_length) : 
  side_length = 28 :=
by
  sorry

end NUMINAMATH_GPT_length_of_garden_side_l196_19625


namespace NUMINAMATH_GPT_congruence_equivalence_l196_19680

theorem congruence_equivalence (m n a b : ℤ) (h_coprime : Int.gcd m n = 1) :
  a ≡ b [ZMOD m * n] ↔ (a ≡ b [ZMOD m] ∧ a ≡ b [ZMOD n]) :=
sorry

end NUMINAMATH_GPT_congruence_equivalence_l196_19680


namespace NUMINAMATH_GPT_sufficient_condition_implication_l196_19612

theorem sufficient_condition_implication {A B : Prop}
  (h : (¬A → ¬B) ∧ (B → A)): (B → A) ∧ (A → ¬¬A ∧ ¬A → ¬B) :=
by
  -- Note: We would provide the proof here normally, but we skip it for now.
  sorry

end NUMINAMATH_GPT_sufficient_condition_implication_l196_19612


namespace NUMINAMATH_GPT_problem_inequality_l196_19640

variable {a b c : ℝ}

theorem problem_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_not_all_equal : ¬ (a = b ∧ b = c)) : 
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) :=
by sorry

end NUMINAMATH_GPT_problem_inequality_l196_19640


namespace NUMINAMATH_GPT_fraction_equals_repeating_decimal_l196_19623

noncomputable def repeating_decimal_fraction : ℚ :=
  let a : ℚ := 46 / 100
  let r : ℚ := 1 / 100
  (a / (1 - r))

theorem fraction_equals_repeating_decimal :
  repeating_decimal_fraction = 46 / 99 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equals_repeating_decimal_l196_19623


namespace NUMINAMATH_GPT_equation_has_one_solution_l196_19601

theorem equation_has_one_solution : ∀ x : ℝ, x - 6 / (x - 2) = 4 - 6 / (x - 2) ↔ x = 4 :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_equation_has_one_solution_l196_19601


namespace NUMINAMATH_GPT_probability_shaded_region_l196_19659

def triangle_game :=
  let total_regions := 6
  let shaded_regions := 3
  shaded_regions / total_regions

theorem probability_shaded_region:
  triangle_game = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_probability_shaded_region_l196_19659


namespace NUMINAMATH_GPT_function_domain_l196_19630

noncomputable def sqrt_domain : Set ℝ :=
  {x | x + 1 ≥ 0 ∧ 2 - x > 0 ∧ 2 - x ≠ 1}

theorem function_domain :
  sqrt_domain = {x | -1 ≤ x ∧ x < 1} ∪ {x | 1 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_function_domain_l196_19630


namespace NUMINAMATH_GPT_percentage_increase_l196_19692

variable (A B C : ℝ)
variable (h1 : A = 0.71 * C)
variable (h2 : A = 0.05 * B)

theorem percentage_increase (A B C : ℝ) (h1 : A = 0.71 * C) (h2 : A = 0.05 * B) : (B - C) / C = 13.2 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l196_19692


namespace NUMINAMATH_GPT_line_intersects_ellipse_l196_19675

theorem line_intersects_ellipse (k : ℝ) (m : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1 → (x^2 / 5) + (y^2 / m) = 1 → True) ↔ (1 < m ∧ m < 5) ∨ (5 < m) :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_ellipse_l196_19675


namespace NUMINAMATH_GPT_minimum_value_of_reciprocal_sum_l196_19609

theorem minimum_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 * a * (-1) - b * 2 + 2 = 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a * (-1) - b * 2 + 2 = 0 ∧ (a + b = 1) ∧ (a = 1/2 ∧ b = 1/2) ∧ (1/a + 1/b = 4) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_reciprocal_sum_l196_19609


namespace NUMINAMATH_GPT_incorrect_median_l196_19644

def data_set : List ℕ := [7, 11, 10, 11, 6, 14, 11, 10, 11, 9]

noncomputable def median (l : List ℕ) : ℚ := 
  let sorted := l.toArray.qsort (· ≤ ·) 
  if sorted.size % 2 = 0 then
    (sorted.get! (sorted.size / 2 - 1) + sorted.get! (sorted.size / 2)) / 2
  else
    sorted.get! (sorted.size / 2)

theorem incorrect_median :
  median data_set ≠ 10 := by
  sorry

end NUMINAMATH_GPT_incorrect_median_l196_19644


namespace NUMINAMATH_GPT_probability_at_least_three_aces_l196_19679

open Nat

noncomputable def combination (n k : ℕ) : ℕ :=
  n.choose k

theorem probability_at_least_three_aces :
  (combination 4 3 * combination 48 2 + combination 4 4 * combination 48 1) / combination 52 5 = (combination 4 3 * combination 48 2 + combination 4 4 * combination 48 1 : ℚ) / combination 52 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_three_aces_l196_19679


namespace NUMINAMATH_GPT_find_a_l196_19696

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : (a^2 + a = 6)) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l196_19696


namespace NUMINAMATH_GPT_loan_proof_l196_19636

-- Definition of the conditions
def interest_rate_year_1 : ℝ := 0.10
def interest_rate_year_2 : ℝ := 0.12
def interest_rate_year_3 : ℝ := 0.14
def total_interest_paid : ℝ := 5400

-- Theorem proving the results
theorem loan_proof (P : ℝ) 
                   (annual_repayment : ℝ)
                   (remaining_principal : ℝ) :
  (interest_rate_year_1 * P) + 
  (interest_rate_year_2 * P) + 
  (interest_rate_year_3 * P) = total_interest_paid →
  3 * annual_repayment = total_interest_paid →
  remaining_principal = P →
  P = 15000 ∧ 
  annual_repayment = 1800 ∧ 
  remaining_principal = 15000 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_loan_proof_l196_19636


namespace NUMINAMATH_GPT_nap_duration_is_two_hours_l196_19642

-- Conditions as definitions in Lean
def naps_per_week : ℕ := 3
def days : ℕ := 70
def total_nap_hours : ℕ := 60

-- Calculate the duration of each nap
theorem nap_duration_is_two_hours :
  ∃ (nap_duration : ℕ), nap_duration = 2 ∧
  (days / 7) * naps_per_week * nap_duration = total_nap_hours :=
by
  sorry

end NUMINAMATH_GPT_nap_duration_is_two_hours_l196_19642


namespace NUMINAMATH_GPT_max_area_difference_160_perimeter_rectangles_l196_19632

theorem max_area_difference_160_perimeter_rectangles : 
  ∃ (l1 w1 l2 w2 : ℕ), (2 * l1 + 2 * w1 = 160) ∧ (2 * l2 + 2 * w2 = 160) ∧ 
  (l1 * w1 - l2 * w2 = 1521) := sorry

end NUMINAMATH_GPT_max_area_difference_160_perimeter_rectangles_l196_19632


namespace NUMINAMATH_GPT_investor_wait_time_l196_19670

noncomputable def compound_interest_time (P A r : ℝ) (n : ℕ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem investor_wait_time :
  compound_interest_time 600 661.5 0.10 2 = 1 := 
sorry

end NUMINAMATH_GPT_investor_wait_time_l196_19670


namespace NUMINAMATH_GPT_pizza_order_l196_19626

theorem pizza_order (couple_want: ℕ) (child_want: ℕ) (num_couples: ℕ) (num_children: ℕ) (slices_per_pizza: ℕ)
  (hcouple: couple_want = 3) (hchild: child_want = 1) (hnumc: num_couples = 1) (hnumch: num_children = 6) (hsp: slices_per_pizza = 4) :
  (couple_want * 2 * num_couples + child_want * num_children) / slices_per_pizza = 3 := 
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_pizza_order_l196_19626


namespace NUMINAMATH_GPT_values_of_x_defined_l196_19634

noncomputable def problem_statement (x : ℝ) : Prop :=
  (2 * x - 3 > 0) ∧ (5 - 2 * x > 0)

theorem values_of_x_defined (x : ℝ) :
  problem_statement x ↔ (3 / 2 < x ∧ x < 5 / 2) :=
by sorry

end NUMINAMATH_GPT_values_of_x_defined_l196_19634


namespace NUMINAMATH_GPT_smallest_two_digit_multiple_of_17_smallest_four_digit_multiple_of_17_l196_19602

theorem smallest_two_digit_multiple_of_17 : ∃ m, 10 ≤ m ∧ m < 100 ∧ 17 ∣ m ∧ ∀ n, 10 ≤ n ∧ n < 100 ∧ 17 ∣ n → m ≤ n :=
by
  sorry

theorem smallest_four_digit_multiple_of_17 : ∃ m, 1000 ≤ m ∧ m < 10000 ∧ 17 ∣ m ∧ ∀ n, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n → m ≤ n :=
by
  sorry

end NUMINAMATH_GPT_smallest_two_digit_multiple_of_17_smallest_four_digit_multiple_of_17_l196_19602


namespace NUMINAMATH_GPT_min_value_of_m_n_squared_l196_19663

theorem min_value_of_m_n_squared 
  (a b c : ℝ)
  (triangle_cond : a^2 + b^2 = c^2)
  (m n : ℝ)
  (line_cond : a * m + b * n + 3 * c = 0) 
  : m^2 + n^2 = 9 := 
by
  sorry

end NUMINAMATH_GPT_min_value_of_m_n_squared_l196_19663


namespace NUMINAMATH_GPT_customers_in_other_countries_l196_19673

-- Define the given conditions

def total_customers : ℕ := 7422
def customers_us : ℕ := 723

theorem customers_in_other_countries : total_customers - customers_us = 6699 :=
by
  -- This part will contain the proof, which is not required for this task.
  sorry

end NUMINAMATH_GPT_customers_in_other_countries_l196_19673


namespace NUMINAMATH_GPT_polygon_sides_arithmetic_progression_l196_19699

theorem polygon_sides_arithmetic_progression
  (angles_in_arithmetic_progression : ∃ (a d : ℝ) (angles : ℕ → ℝ), ∀ (k : ℕ), angles k = a + k * d)
  (common_difference : ∃ (d : ℝ), d = 3)
  (largest_angle : ∃ (n : ℕ) (angles : ℕ → ℝ), angles n = 150) :
  ∃ (n : ℕ), n = 15 :=
sorry

end NUMINAMATH_GPT_polygon_sides_arithmetic_progression_l196_19699


namespace NUMINAMATH_GPT_final_expression_simplified_l196_19619

variable (a : ℝ)

theorem final_expression_simplified : 
  (2 * a + 6 - 3 * a) / 2 = -a / 2 + 3 := 
by 
sorry

end NUMINAMATH_GPT_final_expression_simplified_l196_19619


namespace NUMINAMATH_GPT_smallest_integer_N_l196_19622

theorem smallest_integer_N : ∃ (N : ℕ), 
  (∀ (a : ℕ → ℕ), ((∀ (i : ℕ), i < 125 -> a i > 0 ∧ a i ≤ N) ∧
  (∀ (i : ℕ), 1 ≤ i ∧ i < 124 → a i > (a (i - 1) + a (i + 1)) / 2) ∧
  (∀ (i j : ℕ), i < 125 ∧ j < 125 ∧ i ≠ j → a i ≠ a j)) → N = 2016) :=
sorry

end NUMINAMATH_GPT_smallest_integer_N_l196_19622


namespace NUMINAMATH_GPT_expression_as_polynomial_l196_19639

theorem expression_as_polynomial (x : ℝ) :
  (3 * x^3 + 2 * x^2 + 5 * x + 9) * (x - 2) -
  (x - 2) * (2 * x^3 + 5 * x^2 - 74) +
  (4 * x - 17) * (x - 2) * (x + 4) = 
  x^4 + 2 * x^3 - 5 * x^2 + 9 * x - 30 :=
sorry

end NUMINAMATH_GPT_expression_as_polynomial_l196_19639


namespace NUMINAMATH_GPT_max_value_proof_l196_19638

noncomputable def max_value (x y z : ℝ) : ℝ :=
  1 / x + 2 / y + 3 / z

theorem max_value_proof (x y z : ℝ) (h1 : 2 / 5 ≤ z ∧ z ≤ min x y)
    (h2 : x * z ≥ 4 / 15) (h3 : y * z ≥ 1 / 5) : max_value x y z ≤ 13 := 
by
  sorry

end NUMINAMATH_GPT_max_value_proof_l196_19638


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l196_19618

theorem arithmetic_sequence_problem 
  (a_n b_n : ℕ → ℕ) 
  (S_n T_n : ℕ → ℕ) 
  (h1: ∀ n, S_n n = (n * (a_n n + a_n (n-1))) / 2)
  (h2: ∀ n, T_n n = (n * (b_n n + b_n (n-1))) / 2)
  (h3: ∀ n, (S_n n) / (T_n n) = (7 * n + 2) / (n + 3)):
  (a_n 4) / (b_n 4) = 51 / 10 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l196_19618


namespace NUMINAMATH_GPT_range_of_m_l196_19650

open Set

variable {α : Type}

noncomputable def A (m : ℝ) : Set ℝ := {x | m+1 ≤ x ∧ x ≤ 2*m-1}
noncomputable def B : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}

theorem range_of_m (m : ℝ) (hA : A m ⊆ B) (hA_nonempty : A m ≠ ∅) : 1 ≤ m ∧ m ≤ 3 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l196_19650


namespace NUMINAMATH_GPT_price_increase_percentage_l196_19694

-- Define the problem conditions
def lowest_price := 12
def highest_price := 21

-- Formulate the goal as a theorem
theorem price_increase_percentage :
  ((highest_price - lowest_price) / lowest_price : ℚ) * 100 = 75 := by
  sorry

end NUMINAMATH_GPT_price_increase_percentage_l196_19694


namespace NUMINAMATH_GPT_gcd_204_85_l196_19669

theorem gcd_204_85: Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_GPT_gcd_204_85_l196_19669


namespace NUMINAMATH_GPT_remove_terms_yield_desired_sum_l196_19627

-- Define the original sum and the terms to be removed
def originalSum : ℚ := 1/3 + 1/6 + 1/9 + 1/12 + 1/15 + 1/18
def termsToRemove : List ℚ := [1/9, 1/12, 1/15, 1/18]

-- Definition of the desired remaining sum
def desiredSum : ℚ := 1/2

noncomputable def sumRemainingTerms : ℚ :=
originalSum - List.sum termsToRemove

-- Lean theorem to prove
theorem remove_terms_yield_desired_sum : sumRemainingTerms = desiredSum :=
by 
  sorry

end NUMINAMATH_GPT_remove_terms_yield_desired_sum_l196_19627


namespace NUMINAMATH_GPT_train_length_is_correct_l196_19690

variable (speed_km_hr : Float) (time_sec : Float)

def speed_m_s (speed_km_hr : Float) : Float := speed_km_hr * (1000 / 3600)

def length_of_train (speed_km_hr : Float) (time_sec : Float) : Float :=
  speed_m_s speed_km_hr * time_sec

theorem train_length_is_correct :
  length_of_train 60 12 = 200.04 := 
sorry

end NUMINAMATH_GPT_train_length_is_correct_l196_19690


namespace NUMINAMATH_GPT_arthur_walks_distance_l196_19665

variables (blocks_east blocks_north : ℕ) 
variable (distance_per_block : ℝ)
variable (total_blocks : ℕ)
def total_distance (blocks : ℕ) (distance_per_block : ℝ) : ℝ :=
  blocks * distance_per_block

theorem arthur_walks_distance (h_east : blocks_east = 8) (h_north : blocks_north = 10) 
    (h_total_blocks : total_blocks = blocks_east + blocks_north)
    (h_distance_per_block : distance_per_block = 1 / 4) :
  total_distance total_blocks distance_per_block = 4.5 :=
by {
  -- Here we specify the proof, but as required, we use sorry to skip it.
  sorry
}

end NUMINAMATH_GPT_arthur_walks_distance_l196_19665


namespace NUMINAMATH_GPT_two_distinct_solutions_exist_l196_19607

theorem two_distinct_solutions_exist :
  ∃ (a1 b1 c1 d1 e1 a2 b2 c2 d2 e2 : ℕ), 
    1 ≤ a1 ∧ a1 ≤ 9 ∧ 1 ≤ b1 ∧ b1 ≤ 9 ∧ 1 ≤ c1 ∧ c1 ≤ 9 ∧ 1 ≤ d1 ∧ d1 ≤ 9 ∧ 1 ≤ e1 ∧ e1 ≤ 9 ∧
    1 ≤ a2 ∧ a2 ≤ 9 ∧ 1 ≤ b2 ∧ b2 ≤ 9 ∧ 1 ≤ c2 ∧ c2 ≤ 9 ∧ 1 ≤ d2 ∧ d2 ≤ 9 ∧ 1 ≤ e2 ∧ e2 ≤ 9 ∧
    (b1 - d1 = 2) ∧ (d1 - a1 = 3) ∧ (a1 - c1 = 1) ∧
    (b2 - d2 = 2) ∧ (d2 - a2 = 3) ∧ (a2 - c2 = 1) ∧
    ¬ (a1 = a2 ∧ b1 = b2 ∧ c1 = c2 ∧ d1 = d2 ∧ e1 = e2) :=
by
  sorry

end NUMINAMATH_GPT_two_distinct_solutions_exist_l196_19607


namespace NUMINAMATH_GPT_sophomores_in_program_l196_19615

-- Define variables
variable (P S : ℕ)

-- Conditions for the problem
def total_students (P S : ℕ) : Prop := P + S = 36
def percent_sophomores_club (P S : ℕ) (x : ℕ) : Prop := x = 3 * P / 10
def percent_seniors_club (P S : ℕ) (y : ℕ) : Prop := y = S / 4
def equal_club_members (x y : ℕ) : Prop := x = y

-- Theorem stating the problem and proof goal
theorem sophomores_in_program
  (x y : ℕ)
  (h1 : total_students P S)
  (h2 : percent_sophomores_club P S x)
  (h3 : percent_seniors_club P S y)
  (h4 : equal_club_members x y) :
  P = 15 := 
sorry

end NUMINAMATH_GPT_sophomores_in_program_l196_19615


namespace NUMINAMATH_GPT_cos_alpha_third_quadrant_l196_19652

theorem cos_alpha_third_quadrant (α : ℝ) (h1 : Real.sin α = -5 / 13) (h2 : Real.tan α > 0) : Real.cos α = -12 / 13 := 
sorry

end NUMINAMATH_GPT_cos_alpha_third_quadrant_l196_19652


namespace NUMINAMATH_GPT_number_of_math_books_l196_19604

-- Definitions for conditions
variables (M H : ℕ)

-- Given conditions as a Lean proposition
def conditions : Prop :=
  M + H = 80 ∧ 4 * M + 5 * H = 368

-- The theorem to prove
theorem number_of_math_books (M H : ℕ) (h : conditions M H) : M = 32 :=
by sorry

end NUMINAMATH_GPT_number_of_math_books_l196_19604


namespace NUMINAMATH_GPT_options_necessarily_positive_l196_19637

variable (x y z : ℝ)

theorem options_necessarily_positive (h₁ : -1 < x) (h₂ : x < 0) (h₃ : 0 < y) (h₄ : y < 1) (h₅ : 2 < z) (h₆ : z < 3) :
  y + x^2 * z > 0 ∧
  y + x^2 > 0 ∧
  y + y^2 > 0 ∧
  y + 2 * z > 0 := 
  sorry

end NUMINAMATH_GPT_options_necessarily_positive_l196_19637


namespace NUMINAMATH_GPT_odd_and_symmetric_f_l196_19608

open Real

noncomputable def f (A ϕ : ℝ) (x : ℝ) := A * sin (x + ϕ)

theorem odd_and_symmetric_f (A ϕ : ℝ) (hA : A > 0) (hmin : f A ϕ (π / 4) = -1) : 
  ∃ g : ℝ → ℝ, g x = -A * sin x ∧ (∀ x, g (-x) = -g x) ∧ (∀ x, g (π / 2 - x) = g (π / 2 + x)) :=
sorry

end NUMINAMATH_GPT_odd_and_symmetric_f_l196_19608


namespace NUMINAMATH_GPT_div_by_19_l196_19635

theorem div_by_19 (n : ℕ) : 19 ∣ (26^n - 7^n) :=
sorry

end NUMINAMATH_GPT_div_by_19_l196_19635


namespace NUMINAMATH_GPT_math_problem_l196_19614

theorem math_problem 
  (a b c d : ℝ) 
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : c ≥ d) 
  (h4 : d > 0) 
  (h5 : a + b + c + d = 1) : 
  (a + 2*b + 3*c + 4*d) * a^a * b^b * c^c * d^d < 1 := 
sorry

end NUMINAMATH_GPT_math_problem_l196_19614


namespace NUMINAMATH_GPT_tan_390_correct_l196_19677

-- We assume basic trigonometric functions and their properties
noncomputable def tan_390_equals_sqrt3_div3 : Prop :=
  Real.tan (390 * Real.pi / 180) = Real.sqrt 3 / 3

theorem tan_390_correct : tan_390_equals_sqrt3_div3 :=
  by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_tan_390_correct_l196_19677


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l196_19686

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a n = a 0 + n * d) 
  (h2 : ∀ n, S_n n = n * (a 0 + a n) / 2) 
  (h3 : 2 * a 6 = 5 + a 8) :
  S_n 9 = 45 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l196_19686


namespace NUMINAMATH_GPT_proof_problem_l196_19633

theorem proof_problem (a b c d x : ℝ)
  (h1 : c = 6 * d)
  (h2 : 2 * a = 1 / (-b))
  (h3 : abs x = 9) :
  (2 * a * b - 6 * d + c - x / 3 = -4) ∨ (2 * a * b - 6 * d + c - x / 3 = 2) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l196_19633


namespace NUMINAMATH_GPT_jackson_money_proof_l196_19647

noncomputable def jackson_money (W : ℝ) := 7 * W
noncomputable def lucy_money (W : ℝ) := 3 * W
noncomputable def ethan_money (W : ℝ) := 3 * W + 20

theorem jackson_money_proof : ∀ (W : ℝ), (W + 7 * W + 3 * W + (3 * W + 20) = 600) → jackson_money W = 290.01 :=
by 
  intros W h
  have total_eq := h
  sorry

end NUMINAMATH_GPT_jackson_money_proof_l196_19647


namespace NUMINAMATH_GPT_division_then_multiplication_l196_19695

theorem division_then_multiplication : (180 / 6) * 3 = 90 := 
by
  have step1 : 180 / 6 = 30 := sorry
  have step2 : 30 * 3 = 90 := sorry
  sorry

end NUMINAMATH_GPT_division_then_multiplication_l196_19695


namespace NUMINAMATH_GPT_find_x_l196_19621

-- Define the problem conditions.
def workers := ℕ
def gadgets := ℕ
def gizmos := ℕ
def hours := ℕ

-- Given conditions
def condition1 (g h : ℝ) := (1 / g = 2) ∧ (1 / h = 3)
def condition2 (g h : ℝ) := (100 * 3 / g = 900) ∧ (100 * 3 / h = 600)
def condition3 (x : ℕ) (g h : ℝ) := (40 * 4 / g = x) ∧ (40 * 4 / h = 480)

-- Proof problem statement
theorem find_x (g h : ℝ) (x : ℕ) : 
  condition1 g h → condition2 g h → condition3 x g h → x = 320 :=
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_find_x_l196_19621


namespace NUMINAMATH_GPT_compute_difference_of_reciprocals_l196_19683

theorem compute_difference_of_reciprocals
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = x / y) :
  (1 / x) - (1 / y) = - (1 / y^2) :=
by
  sorry

end NUMINAMATH_GPT_compute_difference_of_reciprocals_l196_19683


namespace NUMINAMATH_GPT_find_n_l196_19658

theorem find_n (n : ℚ) (h : (1 / (n + 2) + 3 / (n + 2) + n / (n + 2) = 4)) : n = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l196_19658


namespace NUMINAMATH_GPT_age_of_17th_student_l196_19653

theorem age_of_17th_student (avg_age_17 : ℕ) (total_students : ℕ) (avg_age_5 : ℕ) (students_5 : ℕ) (avg_age_9 : ℕ) (students_9 : ℕ)
  (h1 : avg_age_17 = 17) (h2 : total_students = 17) (h3 : avg_age_5 = 14) (h4 : students_5 = 5) (h5 : avg_age_9 = 16) (h6 : students_9 = 9) :
  ∃ age_17th_student : ℕ, age_17th_student = 75 :=
by
  sorry

end NUMINAMATH_GPT_age_of_17th_student_l196_19653


namespace NUMINAMATH_GPT_saline_drip_duration_l196_19641

theorem saline_drip_duration (rate_drops_per_minute : ℕ) (drop_to_ml_rate : ℕ → ℕ → Prop)
  (ml_received : ℕ) (time_hours : ℕ) :
  rate_drops_per_minute = 20 ->
  drop_to_ml_rate 100 5 ->
  ml_received = 120 ->
  time_hours = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_saline_drip_duration_l196_19641


namespace NUMINAMATH_GPT_work_completion_l196_19664

variable (A B : Type)

/-- A can do half of the work in 70 days and B can do one third of the work in 35 days.
Together, A and B can complete the work in 60 days. -/
theorem work_completion (hA : (1 : ℚ) / 2 / 70 = (1 : ℚ) / a) 
                      (hB : (1 : ℚ) / 3 / 35 = (1 : ℚ) / b) :
                      (1 / 140 + 1 / 105) = 1 / 60 :=
  sorry

end NUMINAMATH_GPT_work_completion_l196_19664


namespace NUMINAMATH_GPT_sum_smallest_largest_consecutive_even_integers_l196_19616

theorem sum_smallest_largest_consecutive_even_integers
  (n : ℕ) (a y : ℤ) 
  (hn_even : Even n) 
  (h_mean : y = (a + (a + 2 * (n - 1))) / 2) :
  2 * y = (a + (a + 2 * (n - 1))) :=
by
  sorry

end NUMINAMATH_GPT_sum_smallest_largest_consecutive_even_integers_l196_19616


namespace NUMINAMATH_GPT_find_Y_value_l196_19693

theorem find_Y_value : ∃ Y : ℤ, 80 - (Y - (6 + 2 * (7 - 8 - 5))) = 89 ∧ Y = -15 := by
  sorry

end NUMINAMATH_GPT_find_Y_value_l196_19693


namespace NUMINAMATH_GPT_solve_trig_problem_l196_19672

-- Definition of the given problem for trigonometric identities
def problem_statement : Prop :=
  (1 - Real.tan (Real.pi / 12)) / (1 + Real.tan (Real.pi / 12)) = Real.sqrt 3 / 3

theorem solve_trig_problem : problem_statement :=
  by
  sorry -- No proof is needed here

end NUMINAMATH_GPT_solve_trig_problem_l196_19672


namespace NUMINAMATH_GPT_janice_remaining_hours_l196_19667

def homework_time : ℕ := 30
def clean_room_time : ℕ := homework_time / 2
def walk_dog_time : ℕ := homework_time + 5
def trash_time : ℕ := homework_time / 6
def total_task_time : ℕ := homework_time + clean_room_time + walk_dog_time + trash_time
def remaining_minutes : ℕ := 35

theorem janice_remaining_hours : (remaining_minutes : ℚ) / 60 = (7 / 12 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_janice_remaining_hours_l196_19667


namespace NUMINAMATH_GPT_profit_percent_l196_19688

theorem profit_percent (P C : ℝ) (h : (2 / 3) * P = 0.88 * C) : P - C = 0.32 * C → (P - C) / C * 100 = 32 := by
  sorry

end NUMINAMATH_GPT_profit_percent_l196_19688


namespace NUMINAMATH_GPT_radius_of_third_circle_l196_19657

noncomputable def circle_radius {r1 r2 : ℝ} (h1 : r1 = 15) (h2 : r2 = 25) : ℝ :=
  let A_shaded := (25^2 * Real.pi) - (15^2 * Real.pi)
  let r := Real.sqrt (A_shaded / Real.pi)
  r

theorem radius_of_third_circle (r1 r2 r3 : ℝ) (h1 : r1 = 15) (h2 : r2 = 25) :
  circle_radius h1 h2 = 20 :=
by 
  sorry

end NUMINAMATH_GPT_radius_of_third_circle_l196_19657


namespace NUMINAMATH_GPT_circle_condition_l196_19691

def represents_circle (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x + 1/2)^2 + (y + m)^2 = 5/4 - m

theorem circle_condition (m : ℝ) : represents_circle m ↔ m < 5/4 :=
by sorry

end NUMINAMATH_GPT_circle_condition_l196_19691


namespace NUMINAMATH_GPT_ratio_men_to_women_l196_19682

theorem ratio_men_to_women (M W : ℕ) (h1 : W = M + 4) (h2 : M + W = 18) : M = 7 ∧ W = 11 :=
by
  sorry

end NUMINAMATH_GPT_ratio_men_to_women_l196_19682


namespace NUMINAMATH_GPT_man_completion_time_l196_19610

theorem man_completion_time (w_time : ℕ) (efficiency_increase : ℚ) (m_time : ℕ) :
  w_time = 40 → efficiency_increase = 1.25 → m_time = (w_time : ℚ) / efficiency_increase → m_time = 32 :=
by
  sorry

end NUMINAMATH_GPT_man_completion_time_l196_19610


namespace NUMINAMATH_GPT_min_n_Sn_l196_19660

/--
Given an arithmetic sequence {a_n}, let S_n denote the sum of its first n terms.
If S_4 = -2, S_5 = 0, and S_6 = 3, then the minimum value of n * S_n is -9.
-/
theorem min_n_Sn (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h₁ : S 4 = -2)
  (h₂ : S 5 = 0)
  (h₃ : S 6 = 3)
  (h₄ : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2)
  : ∃ n : ℕ, n * S n = -9 := 
sorry

end NUMINAMATH_GPT_min_n_Sn_l196_19660


namespace NUMINAMATH_GPT_max_bottles_drunk_l196_19649

theorem max_bottles_drunk (e b : ℕ) (h1 : e = 16) (h2 : b = 4) : 
  ∃ n : ℕ, n = 5 :=
by
  sorry

end NUMINAMATH_GPT_max_bottles_drunk_l196_19649


namespace NUMINAMATH_GPT_distance_from_origin_to_point_l196_19643

def point : ℝ × ℝ := (12, -16)
def origin : ℝ × ℝ := (0, 0)

theorem distance_from_origin_to_point : 
  let (x1, y1) := origin
  let (x2, y2) := point 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 20 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_origin_to_point_l196_19643


namespace NUMINAMATH_GPT_large_green_curlers_l196_19678

-- Define the number of total curlers
def total_curlers : ℕ := 16

-- Define the fraction for pink curlers
def pink_fraction : ℕ := 1 / 4

-- Define the number of pink curlers
def pink_curlers : ℕ := pink_fraction * total_curlers

-- Define the number of blue curlers
def blue_curlers : ℕ := 2 * pink_curlers

-- Define the total number of pink and blue curlers
def pink_and_blue_curlers : ℕ := pink_curlers + blue_curlers

-- Define the number of green curlers
def green_curlers : ℕ := total_curlers - pink_and_blue_curlers

-- Theorem stating the number of green curlers is 4
theorem large_green_curlers : green_curlers = 4 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_large_green_curlers_l196_19678


namespace NUMINAMATH_GPT_largest_possible_perimeter_l196_19603

noncomputable def max_perimeter (a b c: ℕ) : ℕ := 2 * (a + b + c - 6)

theorem largest_possible_perimeter :
  ∃ (a b c : ℕ), (a = c) ∧ ((a - 2) * (b - 2) = 8) ∧ (max_perimeter a b c = 42) := by
  sorry

end NUMINAMATH_GPT_largest_possible_perimeter_l196_19603


namespace NUMINAMATH_GPT_number_of_ways_to_form_team_l196_19617

noncomputable def binomial : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binomial n k + binomial n (k + 1)

theorem number_of_ways_to_form_team :
  let total_selections := binomial 11 5
  let all_boys_selections := binomial 8 5
  total_selections - all_boys_selections = 406 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_ways_to_form_team_l196_19617


namespace NUMINAMATH_GPT_operation_4_3_is_5_l196_19646

def custom_operation (m n : ℕ) : ℕ := n ^ 2 - m

theorem operation_4_3_is_5 : custom_operation 4 3 = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_operation_4_3_is_5_l196_19646


namespace NUMINAMATH_GPT_find_value_of_x_l196_19655

theorem find_value_of_x (x y z : ℤ) (h1 : x > y) (h2 : y > z) (h3 : z = 3)
  (h4 : 2 * x + 3 * y + 3 * z = 5 * y + 11) (h5 : (x = y + 1) ∧ (y = z + 1)) :
  x = 5 := 
sorry

end NUMINAMATH_GPT_find_value_of_x_l196_19655


namespace NUMINAMATH_GPT_ferries_are_divisible_by_4_l196_19661

theorem ferries_are_divisible_by_4 (t T : ℕ) (H : ∃ n : ℕ, T = n * t) :
  ∃ N : ℕ, N = 4 * (T / t) ∧ N % 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_ferries_are_divisible_by_4_l196_19661


namespace NUMINAMATH_GPT_area_of_sector_one_radian_l196_19676

theorem area_of_sector_one_radian (r θ : ℝ) (hθ : θ = 1) (hr : r = 1) : 
  (1/2 * (r * θ) * r) = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_sector_one_radian_l196_19676


namespace NUMINAMATH_GPT_roots_of_equation_l196_19685

theorem roots_of_equation :
  ∀ x : ℝ, (21 / (x^2 - 9) - 3 / (x - 3) = 1) ↔ (x = 3 ∨ x = -7) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_equation_l196_19685


namespace NUMINAMATH_GPT_tangent_slope_at_1_0_l196_19697

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_slope_at_1_0 : (deriv f 1) = 3 := by
  sorry

end NUMINAMATH_GPT_tangent_slope_at_1_0_l196_19697


namespace NUMINAMATH_GPT_no_rational_xyz_satisfies_l196_19684

theorem no_rational_xyz_satisfies:
  ¬ ∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
  (1 / (x - y) ^ 2 + 1 / (y - z) ^ 2 + 1 / (z - x) ^ 2 = 2014) :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_no_rational_xyz_satisfies_l196_19684


namespace NUMINAMATH_GPT_negative_half_power_zero_l196_19698

theorem negative_half_power_zero : (- (1 / 2)) ^ 0 = 1 :=
by
  sorry

end NUMINAMATH_GPT_negative_half_power_zero_l196_19698


namespace NUMINAMATH_GPT_least_common_multiple_135_195_l196_19628

def leastCommonMultiple (a b : ℕ) : ℕ :=
  Nat.lcm a b

theorem least_common_multiple_135_195 : leastCommonMultiple 135 195 = 1755 := by
  sorry

end NUMINAMATH_GPT_least_common_multiple_135_195_l196_19628


namespace NUMINAMATH_GPT_female_students_proportion_and_count_l196_19624

noncomputable def num_students : ℕ := 30
noncomputable def num_male_students : ℕ := 8
noncomputable def overall_avg_score : ℚ := 90
noncomputable def male_avg_scores : (ℚ × ℚ × ℚ) := (87, 95, 89)
noncomputable def female_avg_scores : (ℚ × ℚ × ℚ) := (92, 94, 91)
noncomputable def avg_attendance_alg_geom : ℚ := 0.85
noncomputable def avg_attendance_calc : ℚ := 0.89

theorem female_students_proportion_and_count :
  ∃ (F : ℕ), F = num_students - num_male_students ∧ (F / num_students : ℚ) = 11 / 15 :=
by
  sorry

end NUMINAMATH_GPT_female_students_proportion_and_count_l196_19624


namespace NUMINAMATH_GPT_students_still_in_school_l196_19629

theorem students_still_in_school
  (total_students : ℕ)
  (half_trip : total_students / 2 > 0)
  (half_remaining_sent_home : (total_students / 2) / 2 > 0)
  (total_students_val : total_students = 1000)
  :
  let students_still_in_school := total_students - (total_students / 2) - ((total_students - (total_students / 2)) / 2)
  students_still_in_school = 250 :=
by
  sorry

end NUMINAMATH_GPT_students_still_in_school_l196_19629


namespace NUMINAMATH_GPT_multiplicative_inverse_l196_19662

theorem multiplicative_inverse (a b n : ℤ) (h₁ : a = 208) (h₂ : b = 240) (h₃ : n = 307) : 
  (a * b) % n = 1 :=
by
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end NUMINAMATH_GPT_multiplicative_inverse_l196_19662


namespace NUMINAMATH_GPT_find_k_l196_19674

def total_balls (k : ℕ) : ℕ := 7 + k

def probability_green (k : ℕ) : ℚ := 7 / (total_balls k)
def probability_purple (k : ℕ) : ℚ := k / (total_balls k)

def expected_value (k : ℕ) : ℚ :=
  (probability_green k) * 3 + (probability_purple k) * (-1)

theorem find_k (k : ℕ) (h_pos : k > 0) (h_exp_value : expected_value k = 1) : k = 7 :=
sorry

end NUMINAMATH_GPT_find_k_l196_19674


namespace NUMINAMATH_GPT_Jolene_cars_washed_proof_l196_19620

-- Definitions for conditions
def number_of_families : ℕ := 4
def babysitting_rate : ℕ := 30 -- in dollars
def car_wash_rate : ℕ := 12 -- in dollars
def total_money_raised : ℕ := 180 -- in dollars

-- Mathematical representation of the problem:
def babysitting_earnings : ℕ := number_of_families * babysitting_rate
def earnings_from_cars : ℕ := total_money_raised - babysitting_earnings
def number_of_cars_washed : ℕ := earnings_from_cars / car_wash_rate

-- The proof statement
theorem Jolene_cars_washed_proof : number_of_cars_washed = 5 := 
sorry

end NUMINAMATH_GPT_Jolene_cars_washed_proof_l196_19620


namespace NUMINAMATH_GPT_principal_amount_invested_l196_19689

noncomputable def calculate_principal : ℕ := sorry

theorem principal_amount_invested (P : ℝ) (y : ℝ) 
    (h1 : 300 = P * y * 2 / 100) -- Condition for simple interest
    (h2 : 307.50 = P * ((1 + y/100)^2 - 1)) -- Condition for compound interest
    : P = 73.53 := 
sorry

end NUMINAMATH_GPT_principal_amount_invested_l196_19689


namespace NUMINAMATH_GPT_domain_of_f_l196_19681

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

theorem domain_of_f :
  { x : ℝ | 2 * x - 1 > 0 } = { x : ℝ | x > 1 / 2 } :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l196_19681


namespace NUMINAMATH_GPT_greatest_prime_factor_of_221_l196_19605

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def greatest_prime_factor (n : ℕ) (p : ℕ) : Prop := 
  is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q → q ∣ n → q ≤ p

theorem greatest_prime_factor_of_221 : greatest_prime_factor 221 17 := by
  sorry

end NUMINAMATH_GPT_greatest_prime_factor_of_221_l196_19605


namespace NUMINAMATH_GPT_last_two_digits_of_7_pow_2017_l196_19666

theorem last_two_digits_of_7_pow_2017 :
  (7 ^ 2017) % 100 = 7 :=
sorry

end NUMINAMATH_GPT_last_two_digits_of_7_pow_2017_l196_19666


namespace NUMINAMATH_GPT_intersection_points_of_curve_with_axes_l196_19600

theorem intersection_points_of_curve_with_axes :
  (∃ t : ℝ, (-2 + 5 * t = 0) ∧ (1 - 2 * t = 1/5)) ∧
  (∃ t : ℝ, (1 - 2 * t = 0) ∧ (-2 + 5 * t = 1/2)) :=
by {
  -- Proving the intersection points with the coordinate axes
  sorry
}

end NUMINAMATH_GPT_intersection_points_of_curve_with_axes_l196_19600


namespace NUMINAMATH_GPT_system_of_equations_solution_exists_l196_19631

theorem system_of_equations_solution_exists :
  ∃ (x y z : ℤ), 
    (x + y - 2018 = (y - 2019) * x) ∧
    (y + z - 2017 = (y - 2019) * z) ∧
    (x + z + 5 = x * z) ∧
    (x = 3 ∧ y = 2021 ∧ z = 4 ∨ 
    x = -1 ∧ y = 2019 ∧ z = -2) := 
sorry

end NUMINAMATH_GPT_system_of_equations_solution_exists_l196_19631


namespace NUMINAMATH_GPT_foxes_hunt_duration_l196_19613

variable (initial_weasels : ℕ) (initial_rabbits : ℕ) (remaining_rodents : ℕ)
variable (foxes : ℕ) (weasels_per_week : ℕ) (rabbits_per_week : ℕ)

def total_rodents_per_week (weasels_per_week rabbits_per_week foxes : ℕ) : ℕ :=
  foxes * (weasels_per_week + rabbits_per_week)

def initial_rodents (initial_weasels initial_rabbits : ℕ) : ℕ :=
  initial_weasels + initial_rabbits

def total_rodents_caught (initial_rodents remaining_rodents : ℕ) : ℕ :=
  initial_rodents - remaining_rodents

def weeks_hunted (total_rodents_caught total_rodents_per_week : ℕ) : ℕ :=
  total_rodents_caught / total_rodents_per_week

theorem foxes_hunt_duration
  (initial_weasels := 100) (initial_rabbits := 50) (remaining_rodents := 96)
  (foxes := 3) (weasels_per_week := 4) (rabbits_per_week := 2) :
  weeks_hunted (total_rodents_caught (initial_rodents initial_weasels initial_rabbits) remaining_rodents) 
                 (total_rodents_per_week weasels_per_week rabbits_per_week foxes) = 3 :=
by
  sorry

end NUMINAMATH_GPT_foxes_hunt_duration_l196_19613
