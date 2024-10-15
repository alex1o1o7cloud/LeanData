import Mathlib

namespace NUMINAMATH_GPT_sector_central_angle_l1436_143671

theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : 1/2 * l * r = 1) : l / r = 2 := 
by
  sorry

end NUMINAMATH_GPT_sector_central_angle_l1436_143671


namespace NUMINAMATH_GPT_percentage_decrease_is_25_percent_l1436_143633

noncomputable def percentage_decrease_in_revenue
  (R : ℝ)
  (projected_revenue : ℝ)
  (actual_revenue : ℝ) : ℝ :=
  ((R - actual_revenue) / R) * 100

-- Conditions
def last_year_revenue (R : ℝ) := R
def projected_revenue (R : ℝ) := 1.20 * R
def actual_revenue (R : ℝ) := 0.625 * (1.20 * R)

-- Proof statement
theorem percentage_decrease_is_25_percent (R : ℝ) :
  percentage_decrease_in_revenue R (projected_revenue R) (actual_revenue R) = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_is_25_percent_l1436_143633


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1436_143611

def setA (x : Real) : Prop := -1 < x ∧ x < 3
def setB (x : Real) : Prop := -2 < x ∧ x < 2

theorem intersection_of_A_and_B : {x : Real | setA x} ∩ {x : Real | setB x} = {x : Real | -1 < x ∧ x < 2} := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1436_143611


namespace NUMINAMATH_GPT_min_colors_needed_is_3_l1436_143658

noncomputable def min_colors_needed (S : Finset (Fin 7)) : Nat :=
  -- function to determine the minimum number of colors needed
  if ∀ (f : Finset (Fin 7) → Fin 3), ∀ (A B : Finset (Fin 7)), A.card = 3 ∧ B.card = 3 →
    A ∩ B = ∅ → f A ≠ f B then
    3
  else
    sorry

theorem min_colors_needed_is_3 :
  ∀ S : Finset (Fin 7), min_colors_needed S = 3 :=
by
  sorry

end NUMINAMATH_GPT_min_colors_needed_is_3_l1436_143658


namespace NUMINAMATH_GPT_calculate_expression_l1436_143602

theorem calculate_expression:
  202.2 * 89.8 - 20.22 * 186 + 2.022 * 3570 - 0.2022 * 16900 = 18198 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1436_143602


namespace NUMINAMATH_GPT_Mr_Spacek_birds_l1436_143646

theorem Mr_Spacek_birds :
  ∃ N : ℕ, 50 < N ∧ N < 100 ∧ N % 9 = 0 ∧ N % 4 = 0 ∧ N = 72 :=
by
  sorry

end NUMINAMATH_GPT_Mr_Spacek_birds_l1436_143646


namespace NUMINAMATH_GPT_amy_biking_miles_l1436_143637

theorem amy_biking_miles (x : ℕ) (h1 : ∀ y : ℕ, y = 2 * x - 3) (h2 : ∀ y : ℕ, x + y = 33) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_amy_biking_miles_l1436_143637


namespace NUMINAMATH_GPT_find_multiple_l1436_143608

-- Defining the conditions
variables (A B k : ℕ)

-- Given conditions
def sum_condition : Prop := A + B = 77
def bigger_number_condition : Prop := A = 42

-- Using the conditions and aiming to prove that k = 5
theorem find_multiple
  (h1 : sum_condition A B)
  (h2 : bigger_number_condition A) :
  6 * B = k * A → k = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l1436_143608


namespace NUMINAMATH_GPT_tan_difference_l1436_143621

theorem tan_difference (x y : ℝ) (hx : Real.tan x = 3) (hy : Real.tan y = 2) :
  Real.tan (x - y) = 1 / 7 := 
  sorry

end NUMINAMATH_GPT_tan_difference_l1436_143621


namespace NUMINAMATH_GPT_arcade_fraction_spent_l1436_143665

noncomputable def weekly_allowance : ℚ := 2.25 
def y (x : ℚ) : ℚ := 1 - x
def remainding_after_toy (x : ℚ) : ℚ := y x - (1/3) * y x

theorem arcade_fraction_spent : 
  ∃ x : ℚ, remainding_after_toy x = 0.60 ∧ x = 3/5 :=
by
  sorry

end NUMINAMATH_GPT_arcade_fraction_spent_l1436_143665


namespace NUMINAMATH_GPT_ab_root_inequality_l1436_143642

theorem ab_root_inequality (a b : ℝ) (h1: ∀ x : ℝ, (x + a) * (x + b) = -9) (h2: a < 0) (h3: b < 0) :
  a + b < -6 :=
sorry

end NUMINAMATH_GPT_ab_root_inequality_l1436_143642


namespace NUMINAMATH_GPT_number_of_zeros_at_end_l1436_143678

def N (n : Nat) := 10^(n+1) + 1

theorem number_of_zeros_at_end (n : Nat) (h : n = 2017) : 
  (N n)^(n + 1) - 1 ≡ 0 [MOD 10^(n + 1)] :=
sorry

end NUMINAMATH_GPT_number_of_zeros_at_end_l1436_143678


namespace NUMINAMATH_GPT_repeating_seventy_two_exceeds_seventy_two_l1436_143610

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end NUMINAMATH_GPT_repeating_seventy_two_exceeds_seventy_two_l1436_143610


namespace NUMINAMATH_GPT_quad_eq_complete_square_l1436_143654

theorem quad_eq_complete_square (p q : ℝ) 
  (h : ∀ x : ℝ, (4 * x^2 - p * x + q = 0 ↔ (x - 1/4)^2 = 33/16)) : q / p = -4 := by
  sorry

end NUMINAMATH_GPT_quad_eq_complete_square_l1436_143654


namespace NUMINAMATH_GPT_simplify_expression_l1436_143601

theorem simplify_expression (b c : ℝ) : 
  (2 * 3 * b * 4 * b^2 * 5 * b^3 * 6 * b^4 * 7 * c^2 = 5040 * b^10 * c^2) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1436_143601


namespace NUMINAMATH_GPT_tangent_line_at_2_m_range_for_three_roots_l1436_143600

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 + 3

theorem tangent_line_at_2 :
  ∃ k b, k = 12 ∧ b = -17 ∧ (∀ x, 12 * x - (k * (x - 2) + f 2) = b) :=
by
  sorry

theorem m_range_for_three_roots :
  {m : ℝ | ∃ x₀ x₁ x₂, x₀ < x₁ ∧ x₁ < x₂ ∧ f x₀ + m = 0 ∧ f x₁ + m = 0 ∧ f x₂ + m = 0} = 
  {m : ℝ | -3 < m ∧ m < -2} :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_2_m_range_for_three_roots_l1436_143600


namespace NUMINAMATH_GPT_symmetry_center_of_f_l1436_143674

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.cos (2 * x) + Real.sqrt 3 * Real.sin x * Real.cos x

theorem symmetry_center_of_f :
  ∃ c : ℝ, ∀ x : ℝ, f (2 * x + π / 6) = Real.sin (2 * (-π / 12) + π / 6) :=
sorry

end NUMINAMATH_GPT_symmetry_center_of_f_l1436_143674


namespace NUMINAMATH_GPT_miranda_saves_half_of_salary_l1436_143670

noncomputable def hourly_wage := 10
noncomputable def daily_hours := 10
noncomputable def weekly_days := 5
noncomputable def weekly_salary := hourly_wage * daily_hours * weekly_days

noncomputable def robby_saving_fraction := 2 / 5
noncomputable def jaylen_saving_fraction := 3 / 5
noncomputable def total_savings := 3000
noncomputable def weeks := 4

noncomputable def robby_weekly_savings := robby_saving_fraction * weekly_salary
noncomputable def jaylen_weekly_savings := jaylen_saving_fraction * weekly_salary
noncomputable def robby_total_savings := robby_weekly_savings * weeks
noncomputable def jaylen_total_savings := jaylen_weekly_savings * weeks
noncomputable def combined_savings_rj := robby_total_savings + jaylen_total_savings
noncomputable def miranda_total_savings := total_savings - combined_savings_rj
noncomputable def miranda_weekly_savings := miranda_total_savings / weeks

noncomputable def miranda_saving_fraction := miranda_weekly_savings / weekly_salary

theorem miranda_saves_half_of_salary:
  miranda_saving_fraction = 1 / 2 := 
by sorry

end NUMINAMATH_GPT_miranda_saves_half_of_salary_l1436_143670


namespace NUMINAMATH_GPT_find_x_l1436_143640

-- We are given points
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 2)

-- Vector a is (2x + 3, x^2 - 4)
def vec_a (x : ℝ) : ℝ × ℝ := (2 * x + 3, x^2 - 4)

-- Vector AB is calculated as
def vec_AB : ℝ × ℝ := (3 - 1, 2 - 2)

-- Define the condition that vec_a and vec_AB form 0° angle
def forms_zero_angle (u v : ℝ × ℝ) : Prop := (u.1 * v.2 - u.2 * v.1) = 0 ∧ (u.1 = v.1 ∧ v.2 = 0)

-- The proof statement
theorem find_x (x : ℝ) (h₁ : forms_zero_angle (vec_a x) vec_AB) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1436_143640


namespace NUMINAMATH_GPT_inequality_proof_l1436_143622

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
(h : Real.sqrt a + Real.sqrt b + Real.sqrt c = 3) :
  (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) ≥ 3 / 2 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1436_143622


namespace NUMINAMATH_GPT_unanswered_questions_equal_nine_l1436_143643

theorem unanswered_questions_equal_nine
  (x y z : ℕ)
  (h1 : 5 * x + 2 * z = 93)
  (h2 : 4 * x - y = 54)
  (h3 : x + y + z = 30) : 
  z = 9 := by
  sorry

end NUMINAMATH_GPT_unanswered_questions_equal_nine_l1436_143643


namespace NUMINAMATH_GPT_integer_solutions_2x2_2xy_9x_y_eq_2_l1436_143657

theorem integer_solutions_2x2_2xy_9x_y_eq_2 : ∀ (x y : ℤ), 2 * x^2 - 2 * x * y + 9 * x + y = 2 → (x, y) = (1, 9) ∨ (x, y) = (2, 8) ∨ (x, y) = (0, 2) ∨ (x, y) = (-1, 3) := 
by 
  intros x y h
  sorry

end NUMINAMATH_GPT_integer_solutions_2x2_2xy_9x_y_eq_2_l1436_143657


namespace NUMINAMATH_GPT_sum_of_interior_angles_l1436_143693

theorem sum_of_interior_angles (h : ∀ (n : ℕ), 360 / 20 = n) : 
  ∃ (s : ℕ), s = 2880 :=
by
  have n := 360 / 20
  have sum := 180 * (n - 2)
  use sum
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_l1436_143693


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1436_143656

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = -2) :
  (1 - 1 / (1 - x)) / (x^2 / (x^2 - 1)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1436_143656


namespace NUMINAMATH_GPT_quotient_base_6_l1436_143632

noncomputable def base_6_to_base_10 (n : ℕ) : ℕ := 
  match n with
  | 2314 => 2 * 6^3 + 3 * 6^2 + 1 * 6^1 + 4
  | 14 => 1 * 6^1 + 4
  | _ => 0

noncomputable def base_10_to_base_6 (n : ℕ) : ℕ := 
  match n with
  | 55 => 1 * 6^2 + 3 * 6^1 + 5
  | _ => 0

theorem quotient_base_6 :
  base_10_to_base_6 ((base_6_to_base_10 2314) / (base_6_to_base_10 14)) = 135 :=
by
  sorry

end NUMINAMATH_GPT_quotient_base_6_l1436_143632


namespace NUMINAMATH_GPT_bicycle_price_l1436_143615

theorem bicycle_price (P : ℝ) (h : 0.2 * P = 200) : P = 1000 := 
by
  sorry

end NUMINAMATH_GPT_bicycle_price_l1436_143615


namespace NUMINAMATH_GPT_possible_ratios_of_distances_l1436_143620

theorem possible_ratios_of_distances (a b : ℝ) (h : a > b) (h1 : ∃ points : Fin 4 → ℝ × ℝ, 
  ∀ (i j : Fin 4), i ≠ j → 
  (dist (points i) (points j) = a ∨ dist (points i) (points j) = b )) :
  a / b = Real.sqrt 2 ∨ 
  a / b = (1 + Real.sqrt 5) / 2 ∨ 
  a / b = Real.sqrt 3 ∨ 
  a / b = Real.sqrt (2 + Real.sqrt 3) :=
by 
  sorry

end NUMINAMATH_GPT_possible_ratios_of_distances_l1436_143620


namespace NUMINAMATH_GPT_rearrange_infinite_decimal_l1436_143605

-- Define the set of digits
def Digit : Type := Fin 10

-- Define the classes of digits
def Class1 (d : Digit) (dec : ℕ → Digit) : Prop :=
  ∃ n : ℕ, ∀ m : ℕ, m > n → dec m ≠ d

def Class2 (d : Digit) (dec : ℕ → Digit) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ dec m = d

-- The statement to prove
theorem rearrange_infinite_decimal (dec : ℕ → Digit) (h : ∃ d : Digit, ¬ Class1 d dec) :
  ∃ rearranged : ℕ → Digit, (Class1 d rearranged ∧ Class2 d rearranged) →
  ∃ r : ℚ, ∃ n : ℕ, ∀ m ≥ n, rearranged m = rearranged (m + n) :=
sorry

end NUMINAMATH_GPT_rearrange_infinite_decimal_l1436_143605


namespace NUMINAMATH_GPT_craig_distance_ridden_farther_l1436_143636

/-- Given that Craig rode the bus for 3.83 miles and walked for 0.17 miles,
    prove that the distance he rode farther than he walked is 3.66 miles. -/
theorem craig_distance_ridden_farther :
  let distance_bus := 3.83
  let distance_walked := 0.17
  distance_bus - distance_walked = 3.66 :=
by
  let distance_bus := 3.83
  let distance_walked := 0.17
  show distance_bus - distance_walked = 3.66
  sorry

end NUMINAMATH_GPT_craig_distance_ridden_farther_l1436_143636


namespace NUMINAMATH_GPT_measure_angle_P_l1436_143641

theorem measure_angle_P (P Q R S : ℝ) (hP : P = 3 * Q) (hR : 4 * R = P) (hS : 6 * S = P) (sum_angles : P + Q + R + S = 360) :
  P = 206 :=
by
  sorry

end NUMINAMATH_GPT_measure_angle_P_l1436_143641


namespace NUMINAMATH_GPT_expand_fraction_product_l1436_143618

-- Define the variable x and the condition that x ≠ 0
variable (x : ℝ) (h : x ≠ 0)

-- State the theorem
theorem expand_fraction_product (h : x ≠ 0) :
  3 / 7 * (7 / x^2 + 7 * x - 7 / x) = 3 / x^2 + 3 * x - 3 / x :=
sorry

end NUMINAMATH_GPT_expand_fraction_product_l1436_143618


namespace NUMINAMATH_GPT_medium_stores_count_l1436_143695

-- Define the total number of stores
def total_stores : ℕ := 300

-- Define the number of medium stores
def medium_stores : ℕ := 75

-- Define the sample size
def sample_size : ℕ := 20

-- Define the expected number of medium stores in the sample
def expected_medium_stores : ℕ := 5

-- The theorem statement claiming that the number of medium stores in the sample is 5
theorem medium_stores_count : 
  (sample_size * medium_stores) / total_stores = expected_medium_stores :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_medium_stores_count_l1436_143695


namespace NUMINAMATH_GPT_tom_trout_count_l1436_143677

theorem tom_trout_count (M T : ℕ) (hM : M = 8) (hT : T = 2 * M) : T = 16 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_tom_trout_count_l1436_143677


namespace NUMINAMATH_GPT_brenda_age_l1436_143635

theorem brenda_age (A B J : ℝ)
  (h1 : A = 4 * B)
  (h2 : J = B + 8)
  (h3 : A = J + 2) :
  B = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_brenda_age_l1436_143635


namespace NUMINAMATH_GPT_man_monthly_salary_l1436_143609

theorem man_monthly_salary (S E : ℝ) (h1 : 0.20 * S = S - 1.20 * E) (h2 : E = 0.80 * S) :
  S = 6000 :=
by
  sorry

end NUMINAMATH_GPT_man_monthly_salary_l1436_143609


namespace NUMINAMATH_GPT_water_bottles_needed_l1436_143668

-- Definitions based on the conditions
def number_of_people: Nat := 4
def travel_hours_each_way: Nat := 8
def water_consumption_rate: ℝ := 0.5 -- bottles per hour per person

-- The total travel time
def total_travel_hours := 2 * travel_hours_each_way

-- The total water needed per person
def water_needed_per_person := water_consumption_rate * total_travel_hours

-- The total water bottles needed for the family
def total_water_bottles := water_needed_per_person * number_of_people

-- The proof statement:
theorem water_bottles_needed : total_water_bottles = 32 := sorry

end NUMINAMATH_GPT_water_bottles_needed_l1436_143668


namespace NUMINAMATH_GPT_winning_candidate_percentage_l1436_143687

theorem winning_candidate_percentage
  (votes1 votes2 votes3 : ℕ)
  (h1 : votes1 = 3000)
  (h2 : votes2 = 5000)
  (h3 : votes3 = 20000) :
  ((votes3 : ℝ) / (votes1 + votes2 + votes3) * 100) = 71.43 := by
  sorry

end NUMINAMATH_GPT_winning_candidate_percentage_l1436_143687


namespace NUMINAMATH_GPT_prob_no_distinct_roots_l1436_143697

-- Definition of integers a, b, c between -7 and 7
def valid_range (n : Int) : Prop := -7 ≤ n ∧ n ≤ 7

-- Definition of the discriminant condition for non-distinct real roots
def no_distinct_real_roots (a b c : Int) : Prop := b * b - 4 * a * c ≤ 0

-- Counting total triplets (a, b, c) with valid range
def total_triplets : Int := 15 * 15 * 15

-- Counting valid triplets with no distinct real roots
def valid_triplets : Int := 225 + (3150 / 2) -- 225 when a = 0 and estimation for a ≠ 0

theorem prob_no_distinct_roots : 
  let P := valid_triplets / total_triplets 
  P = (604 / 1125 : Rat) := 
by
  sorry

end NUMINAMATH_GPT_prob_no_distinct_roots_l1436_143697


namespace NUMINAMATH_GPT_georgia_makes_muffins_l1436_143669

/--
Georgia makes muffins and brings them to her students on the first day of every month.
Her muffin recipe only makes 6 muffins and she has 24 students. 
Prove that Georgia makes 36 batches of muffins in 9 months.
-/
theorem georgia_makes_muffins 
  (muffins_per_batch : ℕ)
  (students : ℕ)
  (months : ℕ) 
  (batches_per_day : ℕ) 
  (total_batches : ℕ)
  (h1 : muffins_per_batch = 6)
  (h2 : students = 24)
  (h3 : months = 9)
  (h4 : batches_per_day = students / muffins_per_batch) : 
  total_batches = months * batches_per_day :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_georgia_makes_muffins_l1436_143669


namespace NUMINAMATH_GPT_cube_of_number_l1436_143694

theorem cube_of_number (n : ℕ) (h1 : 40000 < n^3) (h2 : n^3 < 50000) (h3 : (n^3 % 10) = 6) : n = 36 := by
  sorry

end NUMINAMATH_GPT_cube_of_number_l1436_143694


namespace NUMINAMATH_GPT_third_shiny_penny_prob_l1436_143650

open Nat

def num_shiny : Nat := 4
def num_dull : Nat := 5
def total_pennies : Nat := num_shiny + num_dull

theorem third_shiny_penny_prob :
  let a := 5
  let b := 9
  a + b = 14 := 
by
  sorry

end NUMINAMATH_GPT_third_shiny_penny_prob_l1436_143650


namespace NUMINAMATH_GPT_polynomial_divisibility_l1436_143606

theorem polynomial_divisibility (a b c : ℝ) :
  (∀ x : ℝ, (x ^ 4 + a * x ^ 2 + b * x + c) = (x - 1) ^ 3 * (x + 1) →
  a = 0 ∧ b = 2 ∧ c = -1) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l1436_143606


namespace NUMINAMATH_GPT_pests_eaten_by_frogs_in_week_l1436_143607

-- Definitions
def pests_per_day_per_frog : ℕ := 80
def days_per_week : ℕ := 7
def number_of_frogs : ℕ := 5

-- Proposition to prove
theorem pests_eaten_by_frogs_in_week : (pests_per_day_per_frog * days_per_week * number_of_frogs) = 2800 := 
by sorry

end NUMINAMATH_GPT_pests_eaten_by_frogs_in_week_l1436_143607


namespace NUMINAMATH_GPT_michael_total_payment_correct_l1436_143645

variable (original_suit_price : ℕ := 430)
variable (suit_discount : ℕ := 100)
variable (suit_tax_rate : ℚ := 0.05)

variable (original_shoes_price : ℕ := 190)
variable (shoes_discount : ℕ := 30)
variable (shoes_tax_rate : ℚ := 0.07)

variable (original_dress_shirt_price : ℕ := 80)
variable (original_tie_price : ℕ := 50)
variable (combined_discount_rate : ℚ := 0.20)
variable (dress_shirt_tax_rate : ℚ := 0.06)
variable (tie_tax_rate : ℚ := 0.04)

def calculate_total_amount_paid : ℚ :=
  let discounted_suit_price := original_suit_price - suit_discount
  let suit_tax := discounted_suit_price * suit_tax_rate
  let discounted_shoes_price := original_shoes_price - shoes_discount
  let shoes_tax := discounted_shoes_price * shoes_tax_rate
  let combined_original_price := original_dress_shirt_price + original_tie_price
  let combined_discount := combined_discount_rate * combined_original_price
  let discounted_combined_price := combined_original_price - combined_discount
  let discounted_dress_shirt_price := (original_dress_shirt_price / combined_original_price) * discounted_combined_price
  let discounted_tie_price := (original_tie_price / combined_original_price) * discounted_combined_price
  let dress_shirt_tax := discounted_dress_shirt_price * dress_shirt_tax_rate
  let tie_tax := discounted_tie_price * tie_tax_rate
  discounted_suit_price + suit_tax + discounted_shoes_price + shoes_tax + discounted_dress_shirt_price + dress_shirt_tax + discounted_tie_price + tie_tax

theorem michael_total_payment_correct : calculate_total_amount_paid = 627.14 := by
  sorry

end NUMINAMATH_GPT_michael_total_payment_correct_l1436_143645


namespace NUMINAMATH_GPT_trapezium_other_side_length_l1436_143604

theorem trapezium_other_side_length 
  (side1 : ℝ) (perpendicular_distance : ℝ) (area : ℝ) (side1_val : side1 = 5) 
  (perpendicular_distance_val : perpendicular_distance = 6) (area_val : area = 27) : 
  ∃ other_side : ℝ, other_side = 4 :=
by
  sorry

end NUMINAMATH_GPT_trapezium_other_side_length_l1436_143604


namespace NUMINAMATH_GPT_two_digit_numbers_with_5_as_second_last_digit_l1436_143619

theorem two_digit_numbers_with_5_as_second_last_digit:
  ∀ N : ℕ, (10 ≤ N ∧ N ≤ 99) → (∃ k : ℤ, (N * k) % 100 / 10 = 5) ↔ ¬(N % 20 = 0) :=
by
  sorry

end NUMINAMATH_GPT_two_digit_numbers_with_5_as_second_last_digit_l1436_143619


namespace NUMINAMATH_GPT_sum_primes_less_than_20_l1436_143638

theorem sum_primes_less_than_20 : (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := sorry

end NUMINAMATH_GPT_sum_primes_less_than_20_l1436_143638


namespace NUMINAMATH_GPT_train_platform_length_equal_l1436_143681

theorem train_platform_length_equal 
  (v : ℝ) (t : ℝ) (L_train : ℝ)
  (h1 : v = 144 * (1000 / 3600))
  (h2 : t = 60)
  (h3 : L_train = 1200) :
  L_train = 2400 - L_train := 
sorry

end NUMINAMATH_GPT_train_platform_length_equal_l1436_143681


namespace NUMINAMATH_GPT_moles_CO2_formed_l1436_143625

-- Define the conditions based on the problem statement
def moles_HCl := 1
def moles_NaHCO3 := 1

-- Define the reaction equation in equivalence terms
def chemical_equation (hcl : Nat) (nahco3 : Nat) : Nat :=
  if hcl = 1 ∧ nahco3 = 1 then 1 else 0

-- State the proof problem
theorem moles_CO2_formed : chemical_equation moles_HCl moles_NaHCO3 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_moles_CO2_formed_l1436_143625


namespace NUMINAMATH_GPT_least_five_digit_whole_number_is_perfect_square_and_cube_l1436_143691

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end NUMINAMATH_GPT_least_five_digit_whole_number_is_perfect_square_and_cube_l1436_143691


namespace NUMINAMATH_GPT_convert_base_10_to_base_7_l1436_143684

theorem convert_base_10_to_base_7 (n : ℕ) (h : n = 784) : 
  ∃ a b c d : ℕ, n = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ a = 2 ∧ b = 2 ∧ c = 0 ∧ d = 0 :=
by
  sorry

end NUMINAMATH_GPT_convert_base_10_to_base_7_l1436_143684


namespace NUMINAMATH_GPT_response_rate_increase_approx_l1436_143613

theorem response_rate_increase_approx :
  let original_customers := 80
  let original_respondents := 7
  let redesigned_customers := 63
  let redesigned_respondents := 9
  let original_response_rate := (original_respondents : ℝ) / original_customers * 100
  let redesigned_response_rate := (redesigned_respondents : ℝ) / redesigned_customers * 100
  let percentage_increase := (redesigned_response_rate - original_response_rate) / original_response_rate * 100
  abs (percentage_increase - 63.24) < 0.01 := by
  sorry

end NUMINAMATH_GPT_response_rate_increase_approx_l1436_143613


namespace NUMINAMATH_GPT_single_elimination_tournament_games_23_teams_l1436_143639

noncomputable def single_elimination_tournament_games (num_teams : ℕ) : ℕ :=
  num_teams - 1

theorem single_elimination_tournament_games_23_teams :
  single_elimination_tournament_games 23 = 22 :=
by
  -- Proof has been intentionally omitted
  sorry

end NUMINAMATH_GPT_single_elimination_tournament_games_23_teams_l1436_143639


namespace NUMINAMATH_GPT_fraction_less_than_thirty_percent_l1436_143647

theorem fraction_less_than_thirty_percent (x : ℚ) (hx : x * 180 = 36) (hx_lt : x < 0.3) : x = 1 / 5 := 
by
  sorry

end NUMINAMATH_GPT_fraction_less_than_thirty_percent_l1436_143647


namespace NUMINAMATH_GPT_power_function_value_l1436_143683

theorem power_function_value (α : ℝ) (h₁ : (2 : ℝ) ^ α = (Real.sqrt 2) / 2) : (9 : ℝ) ^ α = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_power_function_value_l1436_143683


namespace NUMINAMATH_GPT_total_weekly_messages_l1436_143667

theorem total_weekly_messages (n r1 r2 r3 r4 r5 m1 m2 m3 m4 m5 : ℕ) 
(p1 p2 p3 p4 : ℕ) (h1 : n = 200) (h2 : r1 = 15) (h3 : r2 = 25) (h4 : r3 = 10) 
(h5 : r4 = 20) (h6 : r5 = 5) (h7 : m1 = 40) (h8 : m2 = 60) (h9 : m3 = 50) 
(h10 : m4 = 30) (h11 : m5 = 20) (h12 : p1 = 15) (h13 : p2 = 25) (h14 : p3 = 40) 
(h15 : p4 = 10) : 
  let total_members_removed := r1 + r2 + r3 + r4 + r5
  let remaining_members := n - total_members_removed
  let daily_messages :=
        (25 * remaining_members / 100 * p1) +
        (50 * remaining_members / 100 * p2) +
        (20 * remaining_members / 100 * p3) +
        (5 * remaining_members / 100 * p4)
  let weekly_messages := daily_messages * 7
  weekly_messages = 21663 :=
by
  sorry

end NUMINAMATH_GPT_total_weekly_messages_l1436_143667


namespace NUMINAMATH_GPT_functional_equation_solution_l1436_143675

theorem functional_equation_solution (f : ℤ → ℤ) :
  (∀ m n : ℤ, f (f (m + n)) = f m + f n) ↔
  (∃ a : ℤ, ∀ n : ℤ, f n = n + a ∨ f n = 0) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l1436_143675


namespace NUMINAMATH_GPT_correct_operation_B_l1436_143676

theorem correct_operation_B (a b : ℝ) : 2 * a * b * b^2 = 2 * a * b^3 :=
sorry

end NUMINAMATH_GPT_correct_operation_B_l1436_143676


namespace NUMINAMATH_GPT_giyoon_chocolates_l1436_143630

theorem giyoon_chocolates (C X : ℕ) (h1 : C = 8 * X) (h2 : C = 6 * (X + 1) + 4) : C = 40 :=
by sorry

end NUMINAMATH_GPT_giyoon_chocolates_l1436_143630


namespace NUMINAMATH_GPT_sum_fifth_powers_divisible_by_15_l1436_143698

theorem sum_fifth_powers_divisible_by_15
  (A B C D E : ℤ) 
  (h : A + B + C + D + E = 0) : 
  (A^5 + B^5 + C^5 + D^5 + E^5) % 15 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_sum_fifth_powers_divisible_by_15_l1436_143698


namespace NUMINAMATH_GPT_KaydenceAge_l1436_143644

-- Definitions for ages of family members based on the problem conditions
def fatherAge := 60
def motherAge := fatherAge - 2 
def brotherAge := fatherAge / 2 
def sisterAge := 40
def totalFamilyAge := 200

-- Lean statement to prove the age of Kaydence
theorem KaydenceAge : 
  fatherAge + motherAge + brotherAge + sisterAge + Kaydence = totalFamilyAge → 
  Kaydence = 12 := 
by
  sorry

end NUMINAMATH_GPT_KaydenceAge_l1436_143644


namespace NUMINAMATH_GPT_ducks_cows_problem_l1436_143629

theorem ducks_cows_problem (D C : ℕ) (h : 2 * D + 4 * C = 2 * (D + C) + 24) : C = 12 := 
  sorry

end NUMINAMATH_GPT_ducks_cows_problem_l1436_143629


namespace NUMINAMATH_GPT_part_a_l1436_143666

theorem part_a (α : ℝ) (n : ℕ) (hα : α > 0) (hn : n > 1) : (1 + α)^n > 1 + n * α :=
sorry

end NUMINAMATH_GPT_part_a_l1436_143666


namespace NUMINAMATH_GPT_h_h_neg1_l1436_143664

def h (x: ℝ) : ℝ := 3 * x^2 - x + 1

theorem h_h_neg1 : h (h (-1)) = 71 := by
  sorry

end NUMINAMATH_GPT_h_h_neg1_l1436_143664


namespace NUMINAMATH_GPT_age_of_person_A_l1436_143624

-- Definitions corresponding to the conditions
variables (x y z : ℕ)
axiom sum_of_ages : x + y = 70
axiom age_difference_A_B : x - z = y
axiom age_difference_B_A_half : y - z = x / 2

-- The proof statement that needs to be proved
theorem age_of_person_A : x = 42 := by 
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_age_of_person_A_l1436_143624


namespace NUMINAMATH_GPT_quadratic_vertex_on_x_axis_l1436_143652

theorem quadratic_vertex_on_x_axis (k : ℝ) :
  (∃ x : ℝ, (x^2 + 2 * x + k) = 0) → k = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_vertex_on_x_axis_l1436_143652


namespace NUMINAMATH_GPT_intersection_points_count_l1436_143653

noncomputable def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) : ℝ := x ^ 2 - 4 * x + 4

theorem intersection_points_count : ∃! x y : ℝ, 0 < x ∧ f x = g x ∧ y ≠ x ∧ f y = g y :=
sorry

end NUMINAMATH_GPT_intersection_points_count_l1436_143653


namespace NUMINAMATH_GPT_intersection_l1436_143617

def A : Set ℝ := { x | -2 < x ∧ x < 3 }
def B : Set ℝ := { x | x > -1 }

theorem intersection (x : ℝ) : x ∈ (A ∩ B) ↔ -1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_GPT_intersection_l1436_143617


namespace NUMINAMATH_GPT_alpha_beta_sum_l1436_143696

variable (α β : ℝ)

theorem alpha_beta_sum (h : ∀ x, (x - α) / (x + β) = (x^2 - 64 * x + 992) / (x^2 + 56 * x - 3168)) :
  α + β = 82 :=
sorry

end NUMINAMATH_GPT_alpha_beta_sum_l1436_143696


namespace NUMINAMATH_GPT_bugs_eat_flowers_l1436_143651

-- Define the problem conditions
def number_of_bugs : ℕ := 3
def flowers_per_bug : ℕ := 2

-- Define the expected outcome
def total_flowers_eaten : ℕ := 6

-- Prove that total flowers eaten is equal to the product of the number of bugs and flowers per bug
theorem bugs_eat_flowers : number_of_bugs * flowers_per_bug = total_flowers_eaten :=
by
  sorry

end NUMINAMATH_GPT_bugs_eat_flowers_l1436_143651


namespace NUMINAMATH_GPT_domain_of_f_l1436_143689

-- Define the function y = sqrt(x-1) + sqrt(x*(3-x))
noncomputable def f (x : ℝ) := Real.sqrt (x - 1) + Real.sqrt (x * (3 - x))

-- Proposition about the domain of the function
theorem domain_of_f (x : ℝ) : (∃ y : ℝ, y = f x) ↔ 1 ≤ x ∧ x ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1436_143689


namespace NUMINAMATH_GPT_solve_system_infinite_solutions_l1436_143673

theorem solve_system_infinite_solutions (m : ℝ) (h1 : ∀ x y : ℝ, x + m * y = 2) (h2 : ∀ x y : ℝ, m * x + 16 * y = 8) :
  m = 4 :=
sorry

end NUMINAMATH_GPT_solve_system_infinite_solutions_l1436_143673


namespace NUMINAMATH_GPT_parabola_tangents_min_area_l1436_143682

noncomputable def parabola_tangents (p : ℝ) : Prop :=
  ∃ (y₀ : ℝ), p > 0 ∧ (2 * Real.sqrt (y₀^2 + 2 * p) = 4)

theorem parabola_tangents_min_area (p : ℝ) : parabola_tangents 2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_tangents_min_area_l1436_143682


namespace NUMINAMATH_GPT_inequality_proof_l1436_143649

theorem inequality_proof (n : ℕ) (h : n > 1) : 
  1 / (2 * n * Real.exp 1) < 1 / Real.exp 1 - (1 - 1 / n) ^ n ∧ 
  1 / Real.exp 1 - (1 - 1 / n) ^ n < 1 / (n * Real.exp 1) := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1436_143649


namespace NUMINAMATH_GPT_fraction_of_smaller_part_l1436_143699

theorem fraction_of_smaller_part (A B : ℕ) (x : ℚ) (h1 : A + B = 66) (h2 : A = 50) (h3 : 0.40 * A = x * B + 10) : x = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_smaller_part_l1436_143699


namespace NUMINAMATH_GPT_alice_zoe_difference_l1436_143616

-- Definitions of the conditions
def AliceApples := 8
def ZoeApples := 2

-- Theorem statement to prove the difference in apples eaten
theorem alice_zoe_difference : AliceApples - ZoeApples = 6 := by
  -- Proof
  sorry

end NUMINAMATH_GPT_alice_zoe_difference_l1436_143616


namespace NUMINAMATH_GPT_function_has_one_zero_l1436_143623

-- Define the function f
def f (x m : ℝ) : ℝ := (m - 1) * x^2 + 2 * (m + 1) * x - 1

-- State the theorem
theorem function_has_one_zero (m : ℝ) :
  (∃! x : ℝ, f x m = 0) ↔ m = 0 ∨ m = -3 := 
sorry

end NUMINAMATH_GPT_function_has_one_zero_l1436_143623


namespace NUMINAMATH_GPT_crackers_initial_count_l1436_143614

theorem crackers_initial_count (friends : ℕ) (crackers_per_friend : ℕ) (total_crackers : ℕ) :
  (friends = 4) → (crackers_per_friend = 2) → (total_crackers = friends * crackers_per_friend) → total_crackers = 8 :=
by intros h_friends h_crackers_per_friend h_total_crackers
   rw [h_friends, h_crackers_per_friend] at h_total_crackers
   exact h_total_crackers

end NUMINAMATH_GPT_crackers_initial_count_l1436_143614


namespace NUMINAMATH_GPT_trajectory_of_midpoint_l1436_143627

theorem trajectory_of_midpoint {x y : ℝ} :
  (∃ Mx My : ℝ, (Mx + 3)^2 + My^2 = 4 ∧ (2 * x - 3 = Mx) ∧ (2 * y = My)) →
  x^2 + y^2 = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_trajectory_of_midpoint_l1436_143627


namespace NUMINAMATH_GPT_max_possible_value_l1436_143661

theorem max_possible_value (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  ∀ (z : ℝ), (z = (x + y + 1) / x) → z ≤ -0.2 :=
by sorry

end NUMINAMATH_GPT_max_possible_value_l1436_143661


namespace NUMINAMATH_GPT_part1_geometric_sequence_part2_sum_of_terms_l1436_143672

/- Part 1 -/
theorem part1_geometric_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h₀ : a 1 = 3) 
  (h₁ : ∀ n, a (n + 1) = a n ^ 2 + 2 * a n) 
  (h₂ : ∀ n, 2 ^ b n = a n + 1) :
  ∃ r, ∀ n, b (n + 1) = r * b n ∧ r = 2 :=
by 
  use 2 
  sorry

/- Part 2 -/
theorem part2_sum_of_terms (b : ℕ → ℝ) (c : ℕ → ℝ) (T : ℕ → ℝ) 
  (h₀ : ∀ n, b n = 2 ^ n)
  (h₁ : ∀ n, c n = n / b n + 1) :
  ∀ n, T n = n + 2 - (n + 2) / 2 ^ n :=
by
  sorry

end NUMINAMATH_GPT_part1_geometric_sequence_part2_sum_of_terms_l1436_143672


namespace NUMINAMATH_GPT_quadratic_transformation_concept_l1436_143663

theorem quadratic_transformation_concept :
  ∀ x : ℝ, (x-3)^2 - 4*(x-3) = 0 ↔ (x = 3 ∨ x = 7) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_quadratic_transformation_concept_l1436_143663


namespace NUMINAMATH_GPT_length_of_train_is_110_l1436_143603

-- Define the speeds and time as constants
def speed_train_kmh := 90
def speed_man_kmh := 9
def time_pass_seconds := 4

-- Define the conversion factor from km/h to m/s
def kmh_to_mps (kmh : ℕ) : ℚ := (kmh : ℚ) * (5 / 18)

-- Calculate relative speed in m/s
def relative_speed_mps : ℚ := kmh_to_mps (speed_train_kmh + speed_man_kmh)

-- Define the length of the train in meters
def length_of_train : ℚ := relative_speed_mps * time_pass_seconds

-- The theorem to prove: The length of the train is 110 meters
theorem length_of_train_is_110 : length_of_train = 110 := 
by sorry

end NUMINAMATH_GPT_length_of_train_is_110_l1436_143603


namespace NUMINAMATH_GPT_famous_figures_mathematicians_l1436_143612

-- List of figures encoded as integers for simplicity
def Bill_Gates := 1
def Gauss := 2
def Liu_Xiang := 3
def Nobel := 4
def Chen_Jingrun := 5
def Chen_Xingshen := 6
def Gorky := 7
def Einstein := 8

-- Set of mathematicians encoded as a set of integers
def mathematicians : Set ℕ := {2, 5, 6}

-- Correct answer set
def correct_answer_set : Set ℕ := {2, 5, 6}

-- The statement to prove
theorem famous_figures_mathematicians:
  mathematicians = correct_answer_set :=
by sorry

end NUMINAMATH_GPT_famous_figures_mathematicians_l1436_143612


namespace NUMINAMATH_GPT_win_game_A_win_game_C_l1436_143690

-- Define the probabilities for heads and tails
def prob_heads : ℚ := 3 / 4
def prob_tails : ℚ := 1 / 4

-- Define the probability of winning Game A
def prob_win_game_A : ℚ := (prob_heads ^ 3) + (prob_tails ^ 3)

-- Define the probability of winning Game C
def prob_win_game_C : ℚ := (prob_heads ^ 4) + (prob_tails ^ 4)

-- State the theorem for Game A
theorem win_game_A : prob_win_game_A = 7 / 16 :=
by 
  -- Lean will check this proof
  sorry

-- State the theorem for Game C
theorem win_game_C : prob_win_game_C = 41 / 128 :=
by 
  -- Lean will check this proof
  sorry

end NUMINAMATH_GPT_win_game_A_win_game_C_l1436_143690


namespace NUMINAMATH_GPT_roots_difference_l1436_143680

theorem roots_difference (a b c : ℝ) (h_eq : a = 1) (h_b : b = -11) (h_c : c = 24) :
    let r1 := (-b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
    let r2 := (-b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
    r1 - r2 = 5 := 
by
  sorry

end NUMINAMATH_GPT_roots_difference_l1436_143680


namespace NUMINAMATH_GPT_max_k_value_condition_l1436_143626

theorem max_k_value_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  ∃ k, k = 100 ∧ (∀ k < 100, ∃ (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c), 
   (k * a * b * c / (a + b + c) <= (a + b)^2 + (a + b + 4 * c)^2)) :=
sorry

end NUMINAMATH_GPT_max_k_value_condition_l1436_143626


namespace NUMINAMATH_GPT_bike_tire_fixing_charge_l1436_143679

theorem bike_tire_fixing_charge (total_profit rent_profit retail_profit: ℝ) (cost_per_tire_parts charge_per_complex_parts charge_per_complex: ℝ) (complex_repairs tire_repairs: ℕ) (charge_per_tire: ℝ) :
  total_profit  = 3000 → rent_profit = 4000 → retail_profit = 2000 →
  cost_per_tire_parts = 5 → charge_per_complex_parts = 50 → charge_per_complex = 300 →
  complex_repairs = 2 → tire_repairs = 300 →
  total_profit = (tire_repairs * charge_per_tire + complex_repairs * charge_per_complex + retail_profit - tire_repairs * cost_per_tire_parts - complex_repairs * charge_per_complex_parts - rent_profit) →
  charge_per_tire = 20 :=
by 
  sorry

end NUMINAMATH_GPT_bike_tire_fixing_charge_l1436_143679


namespace NUMINAMATH_GPT_green_pairs_count_l1436_143648

theorem green_pairs_count 
  (blue_students : ℕ)
  (green_students : ℕ)
  (total_students : ℕ)
  (total_pairs : ℕ)
  (blue_blue_pairs : ℕ) 
  (mixed_pairs_students : ℕ) 
  (green_green_pairs : ℕ) 
  (count_blue : blue_students = 65)
  (count_green : green_students = 67)
  (count_total_students : total_students = 132)
  (count_total_pairs : total_pairs = 66)
  (count_blue_blue_pairs : blue_blue_pairs = 29)
  (count_mixed_blue_students : mixed_pairs_students = 7)
  (count_green_green_pairs : green_green_pairs = 30) :
  green_green_pairs = 30 :=
sorry

end NUMINAMATH_GPT_green_pairs_count_l1436_143648


namespace NUMINAMATH_GPT_subgroups_of_integers_l1436_143662

theorem subgroups_of_integers (G : AddSubgroup ℤ) : ∃ (d : ℤ), G = AddSubgroup.zmultiples d := 
sorry

end NUMINAMATH_GPT_subgroups_of_integers_l1436_143662


namespace NUMINAMATH_GPT_final_notebooks_l1436_143634

def initial_notebooks : ℕ := 10
def ordered_notebooks : ℕ := 6
def lost_notebooks : ℕ := 2

theorem final_notebooks : initial_notebooks + ordered_notebooks - lost_notebooks = 14 :=
by
  sorry

end NUMINAMATH_GPT_final_notebooks_l1436_143634


namespace NUMINAMATH_GPT_minimum_amount_spent_on_boxes_l1436_143628

theorem minimum_amount_spent_on_boxes
  (box_length : ℕ) (box_width : ℕ) (box_height : ℕ) 
  (cost_per_box : ℝ) (total_volume_needed : ℕ) :
  box_length = 20 →
  box_width = 20 →
  box_height = 12 →
  cost_per_box = 0.50 →
  total_volume_needed = 2400000 →
  (total_volume_needed / (box_length * box_width * box_height) * cost_per_box) = 250 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end NUMINAMATH_GPT_minimum_amount_spent_on_boxes_l1436_143628


namespace NUMINAMATH_GPT_arc_length_of_sector_l1436_143655

theorem arc_length_of_sector : 
  ∀ (r : ℝ) (theta: ℝ), r = 1 ∧ theta = 30 * (Real.pi / 180) → (theta * r = Real.pi / 6) :=
by
  sorry

end NUMINAMATH_GPT_arc_length_of_sector_l1436_143655


namespace NUMINAMATH_GPT_lateral_surface_of_prism_is_parallelogram_l1436_143692

-- Definitions based on conditions
def is_right_prism (P : Type) : Prop := sorry
def is_oblique_prism (P : Type) : Prop := sorry
def is_rectangle (S : Type) : Prop := sorry
def is_parallelogram (S : Type) : Prop := sorry
def lateral_surface (P : Type) : Type := sorry

-- Condition 1: The lateral surface of a right prism is a rectangle
axiom right_prism_surface_is_rectangle (P : Type) (h : is_right_prism P) : is_rectangle (lateral_surface P)

-- Condition 2: The lateral surface of an oblique prism can either be a rectangle or a parallelogram
axiom oblique_prism_surface_is_rectangle_or_parallelogram (P : Type) (h : is_oblique_prism P) :
  is_rectangle (lateral_surface P) ∨ is_parallelogram (lateral_surface P)

-- Lean 4 statement for the proof problem
theorem lateral_surface_of_prism_is_parallelogram (P : Type) (p : is_right_prism P ∨ is_oblique_prism P) :
  is_parallelogram (lateral_surface P) :=
by
  sorry

end NUMINAMATH_GPT_lateral_surface_of_prism_is_parallelogram_l1436_143692


namespace NUMINAMATH_GPT_inequality_proof_l1436_143685

theorem inequality_proof (a b c : ℝ) (ha : a = 2 / 21) (hb : b = Real.log 1.1) (hc : c = 21 / 220) : a < b ∧ b < c :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1436_143685


namespace NUMINAMATH_GPT_initial_water_percentage_l1436_143660

noncomputable def S : ℝ := 4.0
noncomputable def V_initial : ℝ := 440
noncomputable def V_final : ℝ := 460
noncomputable def sugar_added : ℝ := 3.2
noncomputable def water_added : ℝ := 10
noncomputable def kola_added : ℝ := 6.8
noncomputable def kola_percentage : ℝ := 8.0 / 100.0
noncomputable def final_sugar_percentage : ℝ := 4.521739130434784 / 100.0

theorem initial_water_percentage : 
  ∀ (W S : ℝ),
  V_initial * (S / 100) + sugar_added = final_sugar_percentage * V_final →
  (W + 8.0 + S) = 100.0 →
  W = 88.0
:=
by
  intros W S h1 h2
  sorry

end NUMINAMATH_GPT_initial_water_percentage_l1436_143660


namespace NUMINAMATH_GPT_cucumbers_count_l1436_143659

theorem cucumbers_count (C T : ℕ) 
  (h1 : C + T = 280)
  (h2 : T = 3 * C) : C = 70 :=
by sorry

end NUMINAMATH_GPT_cucumbers_count_l1436_143659


namespace NUMINAMATH_GPT_total_divisors_7350_l1436_143686

def primeFactorization (n : ℕ) : List (ℕ × ℕ) :=
  if n = 7350 then [(2, 1), (3, 1), (5, 2), (7, 2)] else []

def totalDivisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc (p : ℕ × ℕ) => acc * (p.snd + 1)) 1

theorem total_divisors_7350 : totalDivisors (primeFactorization 7350) = 36 :=
by
  sorry

end NUMINAMATH_GPT_total_divisors_7350_l1436_143686


namespace NUMINAMATH_GPT_run_to_grocery_store_time_l1436_143688

theorem run_to_grocery_store_time
  (running_time: ℝ)
  (grocery_distance: ℝ)
  (friend_distance: ℝ)
  (half_way : friend_distance = grocery_distance / 2)
  (constant_pace : running_time / grocery_distance = (25 : ℝ) / 3)
  : (friend_distance * (25 / 3)) + (friend_distance * (25 / 3)) = 25 :=
by
  -- Given proofs for the conditions can be filled here
  sorry

end NUMINAMATH_GPT_run_to_grocery_store_time_l1436_143688


namespace NUMINAMATH_GPT_quadratic_root_signs_l1436_143631

-- Variables representation
variables {x m : ℝ}

-- Given: The quadratic equation with one positive root and one negative root
theorem quadratic_root_signs (h : ∃ a b : ℝ, 2*a*2*b + (m+1)*(a + b) + m = 0 ∧ a > 0 ∧ b < 0) : 
  m < 0 := 
sorry

end NUMINAMATH_GPT_quadratic_root_signs_l1436_143631
