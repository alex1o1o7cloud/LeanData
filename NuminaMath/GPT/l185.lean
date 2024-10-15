import Mathlib

namespace NUMINAMATH_GPT_triangle_ABC_is_right_triangle_l185_18542

-- Define the triangle and the given conditions
variable (a b c : ℝ)
variable (h1 : a + c = 2*b)
variable (h2 : c - a = 1/2*b)

-- State the problem
theorem triangle_ABC_is_right_triangle : c^2 = a^2 + b^2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_ABC_is_right_triangle_l185_18542


namespace NUMINAMATH_GPT_no_integer_solutions_l185_18510

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), 19 * x^3 - 17 * y^3 = 50 := 
by 
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l185_18510


namespace NUMINAMATH_GPT_total_candies_l185_18549

-- Define variables and conditions
variables (x y z : ℕ)
axiom h1 : x = y / 2
axiom h2 : x + z = 24
axiom h3 : y + z = 34

-- The statement to be proved
theorem total_candies : x + y + z = 44 :=
by
  sorry

end NUMINAMATH_GPT_total_candies_l185_18549


namespace NUMINAMATH_GPT_exists_multiple_digits_0_1_l185_18537

theorem exists_multiple_digits_0_1 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (k ≤ n) ∧ (∃ m : ℕ, m * n = k) ∧ (∀ d : ℕ, ∃ i : ℕ, i ≤ n ∧ d = 0 ∨ d = 1) :=
sorry

end NUMINAMATH_GPT_exists_multiple_digits_0_1_l185_18537


namespace NUMINAMATH_GPT_decorative_object_height_l185_18522

def diameter_fountain := 20 -- meters
def radius_fountain := diameter_fountain / 2 -- meters

def max_height := 8 -- meters
def distance_to_max_height := 2 -- meters

-- The initial height of the water jets at the decorative object
def initial_height := 7.5 -- meters

theorem decorative_object_height :
  initial_height = 7.5 :=
  sorry

end NUMINAMATH_GPT_decorative_object_height_l185_18522


namespace NUMINAMATH_GPT_book_cost_l185_18518

theorem book_cost (p : ℝ) (h1 : 14 * p < 25) (h2 : 16 * p > 28) : 1.75 < p ∧ p < 1.7857 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_book_cost_l185_18518


namespace NUMINAMATH_GPT_min_value_frac_sum_l185_18506

theorem min_value_frac_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 1) :
  ∃ (a b : ℝ), (a + 3 * b = 1) ∧ (a > 0) ∧ (b > 0) ∧ (∀ (a b : ℝ), (a + 3 * b = 1) → 0 < a → 0 < b → (1 / a + 3 / b) ≥ 16) :=
sorry

end NUMINAMATH_GPT_min_value_frac_sum_l185_18506


namespace NUMINAMATH_GPT_find_second_bank_account_balance_l185_18524

theorem find_second_bank_account_balance : 
  (exists (X : ℝ),  
    let raw_material_cost := 100
    let machinery_cost := 125
    let raw_material_tax := 0.05 * raw_material_cost
    let discounted_machinery_cost := machinery_cost - (0.1 * machinery_cost)
    let machinery_tax := 0.08 * discounted_machinery_cost
    let total_raw_material_cost := raw_material_cost + raw_material_tax
    let total_machinery_cost := discounted_machinery_cost + machinery_tax
    let total_spent := total_raw_material_cost + total_machinery_cost
    let total_cash := 900 + X
    let spent_proportion := 0.2 * total_cash
    total_spent = spent_proportion → X = 232.50) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_second_bank_account_balance_l185_18524


namespace NUMINAMATH_GPT_initial_men_count_l185_18571

theorem initial_men_count
  (M : ℕ)
  (h1 : ∀ T : ℕ, (M * 8 * 10 = T) → (5 * 16 * 12 = T)) :
  M = 12 :=
by
  sorry

end NUMINAMATH_GPT_initial_men_count_l185_18571


namespace NUMINAMATH_GPT_inequality_generalization_l185_18544

theorem inequality_generalization (x : ℝ) (n : ℕ) (hn : n > 0) (hx : x > 0) 
  (h1 : x + 1 / x ≥ 2) (h2 : x + 4 / (x ^ 2) = (x / 2) + (x / 2) + 4 / (x ^ 2) ∧ (x / 2) + (x / 2) + 4 / (x ^ 2) ≥ 3) : 
  x + n^n / x^n ≥ n + 1 := 
sorry

end NUMINAMATH_GPT_inequality_generalization_l185_18544


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l185_18525

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x < 1 → x < 2) ∧ ¬ (x < 2 → x < 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l185_18525


namespace NUMINAMATH_GPT_last_four_digits_pow_product_is_5856_l185_18592

noncomputable def product : ℕ := 301 * 402 * 503 * 604 * 646 * 547 * 448 * 349

theorem last_four_digits_pow_product_is_5856 :
  (product % 10000) ^ 4 % 10000 = 5856 := by
  sorry

end NUMINAMATH_GPT_last_four_digits_pow_product_is_5856_l185_18592


namespace NUMINAMATH_GPT_ryan_more_hours_english_than_spanish_l185_18546

-- Define the time spent on various languages as constants
def hoursEnglish : ℕ := 7
def hoursSpanish : ℕ := 4

-- State the problem as a theorem
theorem ryan_more_hours_english_than_spanish : hoursEnglish - hoursSpanish = 3 :=
by sorry

end NUMINAMATH_GPT_ryan_more_hours_english_than_spanish_l185_18546


namespace NUMINAMATH_GPT_enthalpy_change_l185_18508

def DeltaH_prods : Float := -286.0 - 297.0
def DeltaH_reacts : Float := -20.17
def HessLaw (DeltaH_prods DeltaH_reacts : Float) : Float := DeltaH_prods - DeltaH_reacts

theorem enthalpy_change : HessLaw DeltaH_prods DeltaH_reacts = -1125.66 := by
  -- Lean needs a proof, which is not needed per instructions
  sorry

end NUMINAMATH_GPT_enthalpy_change_l185_18508


namespace NUMINAMATH_GPT_speed_of_water_l185_18570

variable (v : ℝ) -- the speed of the water in km/h
variable (t : ℝ) -- time taken to swim back in hours
variable (d : ℝ) -- distance swum against the current in km
variable (s : ℝ) -- speed in still water

theorem speed_of_water :
  ∀ (v t d s : ℝ),
  s = 20 -> t = 5 -> d = 40 -> d = (s - v) * t -> v = 12 :=
by
  intros v t d s ht hs hd heq
  sorry

end NUMINAMATH_GPT_speed_of_water_l185_18570


namespace NUMINAMATH_GPT_smallest_positive_integer_l185_18529

theorem smallest_positive_integer (n : ℕ) (hn : 0 < n) (h : 19 * n ≡ 1456 [MOD 11]) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l185_18529


namespace NUMINAMATH_GPT_number_of_child_workers_l185_18579

-- Define the conditions
def number_of_male_workers : ℕ := 20
def number_of_female_workers : ℕ := 15
def wage_per_male : ℕ := 35
def wage_per_female : ℕ := 20
def wage_per_child : ℕ := 8
def average_wage : ℕ := 26

-- Define the proof goal
theorem number_of_child_workers (C : ℕ) : 
  ((number_of_male_workers * wage_per_male +
    number_of_female_workers * wage_per_female +
    C * wage_per_child) /
   (number_of_male_workers + number_of_female_workers + C) = average_wage) → 
  C = 5 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_child_workers_l185_18579


namespace NUMINAMATH_GPT_bill_experience_l185_18590

theorem bill_experience (j b : ℕ) (h1 : j - 5 = 3 * (b - 5)) (h2 : j = 2 * b) : b = 10 := 
by
  sorry

end NUMINAMATH_GPT_bill_experience_l185_18590


namespace NUMINAMATH_GPT_common_ratio_of_geometric_series_l185_18515

theorem common_ratio_of_geometric_series : ∃ r : ℝ, ∀ n : ℕ, 
  r = (if n = 0 then 2 / 3
       else if n = 1 then (2 / 3) * (2 / 3)
       else if n = 2 then (2 / 3) * (2 / 3) * (2 / 3)
       else sorry)
  ∧ r = 2 / 3 := sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_series_l185_18515


namespace NUMINAMATH_GPT_total_ladybugs_correct_l185_18588

noncomputable def total_ladybugs (with_spots : ℕ) (without_spots : ℕ) : ℕ :=
  with_spots + without_spots

theorem total_ladybugs_correct :
  total_ladybugs 12170 54912 = 67082 :=
by
  unfold total_ladybugs
  rfl

end NUMINAMATH_GPT_total_ladybugs_correct_l185_18588


namespace NUMINAMATH_GPT_simplify_expression_l185_18599

theorem simplify_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) =
  6 * Real.sqrt 6 + 6 * Real.sqrt 10 - 6 * Real.sqrt 14 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l185_18599


namespace NUMINAMATH_GPT_least_number_to_subtract_l185_18543

theorem least_number_to_subtract (n : ℕ) (h1 : n = 157632)
  (h2 : ∃ k : ℕ, k = 12 * 18 * 24 / (gcd 12 (gcd 18 24)) ∧ k ∣ n - 24) :
  n - 24 = 24 := 
sorry

end NUMINAMATH_GPT_least_number_to_subtract_l185_18543


namespace NUMINAMATH_GPT_man_l185_18591

noncomputable def speed_of_current : ℝ := 3 -- in kmph
noncomputable def time_to_cover_100_meters_downstream : ℝ := 19.99840012798976 -- in seconds
noncomputable def distance_covered : ℝ := 0.1 -- in kilometers (100 meters)

noncomputable def speed_in_still_water : ℝ :=
  (distance_covered / (time_to_cover_100_meters_downstream / 3600)) - speed_of_current

theorem man's_speed_in_still_water :
  speed_in_still_water = 14.9997120913593 :=
  by
    sorry

end NUMINAMATH_GPT_man_l185_18591


namespace NUMINAMATH_GPT_similar_triangles_same_heights_ratio_l185_18559

theorem similar_triangles_same_heights_ratio (h1 h2 : ℝ) 
  (sim_ratio : h1 / h2 = 1 / 4) : h1 / h2 = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_similar_triangles_same_heights_ratio_l185_18559


namespace NUMINAMATH_GPT_avg_speed_3x_km_l185_18512

-- Definitions based on the conditions
def distance1 (x : ℕ) : ℕ := x
def speed1 : ℕ := 90
def distance2 (x : ℕ) : ℕ := 2 * x
def speed2 : ℕ := 20

-- The total distance covered
def total_distance (x : ℕ) : ℕ := distance1 x + distance2 x

-- The time taken for each part of the journey
def time1 (x : ℕ) : ℚ := distance1 x / speed1
def time2 (x : ℕ) : ℚ := distance2 x / speed2

-- The total time taken
def total_time (x : ℕ) : ℚ := time1 x + time2 x

-- The average speed
def average_speed (x : ℕ) : ℚ := total_distance x / total_time x

-- The theorem we want to prove
theorem avg_speed_3x_km (x : ℕ) : average_speed x = 27 := by
  sorry

end NUMINAMATH_GPT_avg_speed_3x_km_l185_18512


namespace NUMINAMATH_GPT_problem_l185_18568

theorem problem (y : ℝ) (hy : 5 = y^2 + 4 / y^2) : y + 2 / y = 3 ∨ y + 2 / y = -3 :=
by
  sorry

end NUMINAMATH_GPT_problem_l185_18568


namespace NUMINAMATH_GPT_zion_dad_age_difference_in_10_years_l185_18583

/-
Given:
1. Zion's age is 8 years.
2. Zion's dad's age is 3 more than 4 times Zion's age.
Prove:
In 10 years, the difference in age between Zion's dad and Zion will be 27 years.
-/

theorem zion_dad_age_difference_in_10_years :
  let zion_age := 8
  let dad_age := 4 * zion_age + 3
  (dad_age + 10) - (zion_age + 10) = 27 := by
  sorry

end NUMINAMATH_GPT_zion_dad_age_difference_in_10_years_l185_18583


namespace NUMINAMATH_GPT_calculate_a5_l185_18553

variable {a1 : ℝ} -- geometric sequence first term
variable {a : ℕ → ℝ} -- geometric sequence
variable {n : ℕ} -- sequence index
variable {r : ℝ} -- common ratio

-- Definitions based on the given conditions
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a1 * r ^ n

-- Given conditions
axiom common_ratio_is_two : r = 2
axiom product_condition : a 2 * a 10 = 16 -- indices offset by 1, so a3 = a 2 and a11 = a 10
axiom positive_terms : ∀ n, a n > 0

-- Goal: calculate a 4
theorem calculate_a5 : a 4 = 1 :=
sorry

end NUMINAMATH_GPT_calculate_a5_l185_18553


namespace NUMINAMATH_GPT_even_of_form_4a_plus_2_not_diff_of_squares_l185_18504

theorem even_of_form_4a_plus_2_not_diff_of_squares (a x y : ℤ) : ¬ (4 * a + 2 = x^2 - y^2) :=
by sorry

end NUMINAMATH_GPT_even_of_form_4a_plus_2_not_diff_of_squares_l185_18504


namespace NUMINAMATH_GPT_eggs_in_each_basket_l185_18511

theorem eggs_in_each_basket
  (total_red_eggs : ℕ)
  (total_orange_eggs : ℕ)
  (h_red : total_red_eggs = 30)
  (h_orange : total_orange_eggs = 45)
  (eggs_in_each_basket : ℕ)
  (h_at_least : eggs_in_each_basket ≥ 5) :
  (total_red_eggs % eggs_in_each_basket = 0) ∧ 
  (total_orange_eggs % eggs_in_each_basket = 0) ∧
  eggs_in_each_basket = 15 := sorry

end NUMINAMATH_GPT_eggs_in_each_basket_l185_18511


namespace NUMINAMATH_GPT_final_result_is_106_l185_18513

def chosen_number : ℕ := 122
def multiplied_by_2 (x : ℕ) : ℕ := 2 * x
def subtract_138 (y : ℕ) : ℕ := y - 138

theorem final_result_is_106 : subtract_138 (multiplied_by_2 chosen_number) = 106 :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_final_result_is_106_l185_18513


namespace NUMINAMATH_GPT_ranking_most_economical_l185_18582

theorem ranking_most_economical (c_T c_R c_J q_T q_R q_J : ℝ)
  (hR_cost : c_R = 1.25 * c_T)
  (hR_quantity : q_R = 0.75 * q_J)
  (hJ_quantity : q_J = 2.5 * q_T)
  (hJ_cost : c_J = 1.2 * c_R) :
  ((c_J / q_J) ≤ (c_R / q_R)) ∧ ((c_R / q_R) ≤ (c_T / q_T)) :=
by {
  sorry
}

end NUMINAMATH_GPT_ranking_most_economical_l185_18582


namespace NUMINAMATH_GPT_parabola_line_intersect_at_one_point_l185_18598

theorem parabola_line_intersect_at_one_point (a : ℚ) :
  (∃ x : ℚ, ax^2 + 5 * x + 4 = 0) → a = 25 / 16 :=
by
  -- Conditions and computation here
  sorry

end NUMINAMATH_GPT_parabola_line_intersect_at_one_point_l185_18598


namespace NUMINAMATH_GPT_four_digit_solution_l185_18594

-- Definitions for the conditions.
def condition1 (u z x : ℕ) : Prop := u + z - 4 * x = 1
def condition2 (u z y : ℕ) : Prop := u + 10 * z - 2 * y = 14

-- The theorem to prove that the four-digit number xyz is either 1014, 2218, or 1932
theorem four_digit_solution (x y z u : ℕ) (h1 : condition1 u z x) (h2 : condition2 u z y) :
  (x = 1 ∧ y = 0 ∧ z = 1 ∧ u = 4) ∨
  (x = 2 ∧ y = 2 ∧ z = 1 ∧ u = 8) ∨
  (x = 1 ∧ y = 9 ∧ z = 3 ∧ u = 2) := 
sorry

end NUMINAMATH_GPT_four_digit_solution_l185_18594


namespace NUMINAMATH_GPT_total_weight_of_8_moles_of_BaCl2_l185_18530

-- Define atomic weights
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_Cl : ℝ := 35.45

-- Define the molecular weight of BaCl2
def molecular_weight_BaCl2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Cl

-- Define the number of moles
def moles : ℝ := 8

-- Define the total weight calculation
def total_weight : ℝ := molecular_weight_BaCl2 * moles

-- The theorem to prove
theorem total_weight_of_8_moles_of_BaCl2 : total_weight = 1665.84 :=
by sorry

end NUMINAMATH_GPT_total_weight_of_8_moles_of_BaCl2_l185_18530


namespace NUMINAMATH_GPT_a_10_eq_505_l185_18573

-- The sequence definition
def a (n : ℕ) : ℕ :=
  let start := (n * (n - 1)) / 2 + 1
  List.sum (List.range' start n)

-- Theorem that the 10th term of the sequence is 505
theorem a_10_eq_505 : a 10 = 505 := 
by
  sorry

end NUMINAMATH_GPT_a_10_eq_505_l185_18573


namespace NUMINAMATH_GPT_fraction_comparison_l185_18500

theorem fraction_comparison : 
  (15 / 11 : ℝ) > (17 / 13 : ℝ) ∧ (17 / 13 : ℝ) > (19 / 15 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_fraction_comparison_l185_18500


namespace NUMINAMATH_GPT_smallest_possible_n_l185_18502

theorem smallest_possible_n (n : ℕ) (h1 : n ≥ 100) (h2 : n < 1000)
  (h3 : n % 9 = 2) (h4 : n % 7 = 2) : n = 128 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_n_l185_18502


namespace NUMINAMATH_GPT_classes_Mr_Gates_has_l185_18561

theorem classes_Mr_Gates_has (buns_per_package packages_bought students_per_class buns_per_student : ℕ) :
  buns_per_package = 8 → 
  packages_bought = 30 → 
  students_per_class = 30 → 
  buns_per_student = 2 → 
  (packages_bought * buns_per_package) / (students_per_class * buns_per_student) = 4 := 
by
  sorry

end NUMINAMATH_GPT_classes_Mr_Gates_has_l185_18561


namespace NUMINAMATH_GPT_ratio_of_areas_l185_18567

theorem ratio_of_areas (r : ℝ) (A_triangle : ℝ) (A_circle : ℝ) 
  (h1 : ∀ r, A_triangle = (3 * r^2) / 4)
  (h2 : ∀ r, A_circle = π * r^2) 
  : (A_triangle / A_circle) = 3 / (4 * π) :=
sorry

end NUMINAMATH_GPT_ratio_of_areas_l185_18567


namespace NUMINAMATH_GPT_frequency_of_middle_rectangle_l185_18516

theorem frequency_of_middle_rectangle
    (n : ℕ)
    (A : ℕ)
    (h1 : A + (n - 1) * A = 160) :
    A = 32 :=
by
  sorry

end NUMINAMATH_GPT_frequency_of_middle_rectangle_l185_18516


namespace NUMINAMATH_GPT_contradiction_proof_l185_18584

theorem contradiction_proof (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by sorry

end NUMINAMATH_GPT_contradiction_proof_l185_18584


namespace NUMINAMATH_GPT_percentage_problem_l185_18566

theorem percentage_problem (P : ℝ) :
  (P / 100) * 600 = (40 / 100) * 1050 → P = 70 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_percentage_problem_l185_18566


namespace NUMINAMATH_GPT_each_child_apples_l185_18557

-- Define the given conditions
def total_apples : ℕ := 450
def num_adults : ℕ := 40
def num_adults_apples : ℕ := 3
def num_children : ℕ := 33

-- Define the theorem to prove
theorem each_child_apples : 
  let total_apples_eaten_by_adults := num_adults * num_adults_apples
  let total_apples_for_children := total_apples - total_apples_eaten_by_adults
  let apples_per_child := total_apples_for_children / num_children
  apples_per_child = 10 :=
by
  sorry

end NUMINAMATH_GPT_each_child_apples_l185_18557


namespace NUMINAMATH_GPT_movie_hours_sum_l185_18548

noncomputable def total_movie_hours 
  (Michael Joyce Nikki Ryn Sam : ℕ) 
  (h1 : Joyce = Michael + 2)
  (h2 : Nikki = 3 * Michael)
  (h3 : Ryn = (4 * Nikki) / 5)
  (h4 : Sam = (3 * Joyce) / 2)
  (h5 : Nikki = 30) : ℕ :=
  Joyce + Michael + Nikki + Ryn + Sam

theorem movie_hours_sum (Michael Joyce Nikki Ryn Sam : ℕ) 
  (h1 : Joyce = Michael + 2)
  (h2 : Nikki = 3 * Michael)
  (h3 : Ryn = (4 * Nikki) / 5)
  (h4 : Sam = (3 * Joyce) / 2)
  (h5 : Nikki = 30) : 
  total_movie_hours Michael Joyce Nikki Ryn Sam h1 h2 h3 h4 h5 = 94 :=
by 
  -- The actual proof will go here, to demonstrate the calculations resulting in 94 hours
  sorry

end NUMINAMATH_GPT_movie_hours_sum_l185_18548


namespace NUMINAMATH_GPT_percent_value_in_quarters_l185_18581

theorem percent_value_in_quarters (num_dimes num_quarters : ℕ) 
  (value_dime value_quarter total_value value_in_quarters : ℕ) 
  (h1 : num_dimes = 75)
  (h2 : num_quarters = 30)
  (h3 : value_dime = num_dimes * 10)
  (h4 : value_quarter = num_quarters * 25)
  (h5 : total_value = value_dime + value_quarter)
  (h6 : value_in_quarters = num_quarters * 25) :
  (value_in_quarters / total_value) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_percent_value_in_quarters_l185_18581


namespace NUMINAMATH_GPT_range_of_a_l185_18593

noncomputable def f (x : ℤ) (a : ℝ) := (3 * x^2 + a * x + 26) / (x + 1)

theorem range_of_a (a : ℝ) :
  (∃ x : ℕ+, f x a ≤ 2) → a ≤ -15 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l185_18593


namespace NUMINAMATH_GPT_S13_is_52_l185_18507

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {n : ℕ}

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a1 d : ℕ), ∀ n, a n = a1 + (n - 1) * d

-- Define the sum of the first n terms
def sum_of_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem S13_is_52 (h1 : is_arithmetic_sequence a)
                  (h2 : a 3 + a 7 + a 11 = 12)
                  (h3 : sum_of_first_n_terms S a) :
  S 13 = 52 :=
by sorry

end NUMINAMATH_GPT_S13_is_52_l185_18507


namespace NUMINAMATH_GPT_logic_problem_l185_18572

theorem logic_problem (p q : Prop) (h1 : ¬p) (h2 : ¬(p ∧ q)) : ¬ (p ∨ q) :=
sorry

end NUMINAMATH_GPT_logic_problem_l185_18572


namespace NUMINAMATH_GPT_projectile_time_l185_18578

theorem projectile_time : ∃ t : ℝ, (60 - 8 * t - 5 * t^2 = 30) ∧ t = 1.773 := by
  sorry

end NUMINAMATH_GPT_projectile_time_l185_18578


namespace NUMINAMATH_GPT_system_solution_l185_18501

theorem system_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : 0 < x₃) (h₄ : 0 < x₄) (h₅ : 0 < x₅)
  (h₆ : x₁ + x₂ = x₃^2) (h₇ : x₃ + x₄ = x₅^2) (h₈ : x₄ + x₅ = x₁^2) (h₉ : x₅ + x₁ = x₂^2) :
  x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 := 
by 
  sorry

end NUMINAMATH_GPT_system_solution_l185_18501


namespace NUMINAMATH_GPT_john_needs_to_sell_1200_pencils_to_make_120_dollars_profit_l185_18534

theorem john_needs_to_sell_1200_pencils_to_make_120_dollars_profit :
  ∀ (buy_rate_pencils : ℕ) (buy_rate_dollars : ℕ) (sell_rate_pencils : ℕ) (sell_rate_dollars : ℕ),
    buy_rate_pencils = 5 →
    buy_rate_dollars = 7 →
    sell_rate_pencils = 4 →
    sell_rate_dollars = 6 →
    ∃ (n_pencils : ℕ), n_pencils = 1200 ∧ 
                        (sell_rate_dollars / sell_rate_pencils - buy_rate_dollars / buy_rate_pencils) * n_pencils = 120 :=
by
  sorry

end NUMINAMATH_GPT_john_needs_to_sell_1200_pencils_to_make_120_dollars_profit_l185_18534


namespace NUMINAMATH_GPT_rectangular_prism_pairs_l185_18532

def total_pairs_of_edges_in_rect_prism_different_dimensions (length width height : ℝ) 
  (h1 : length ≠ width) 
  (h2 : width ≠ height) 
  (h3 : height ≠ length) 
  : ℕ :=
66

theorem rectangular_prism_pairs (length width height : ℝ) 
  (h1 : length ≠ width) 
  (h2 : width ≠ height) 
  (h3 : height ≠ length) 
  : total_pairs_of_edges_in_rect_prism_different_dimensions length width height h1 h2 h3 = 66 := 
sorry

end NUMINAMATH_GPT_rectangular_prism_pairs_l185_18532


namespace NUMINAMATH_GPT_initial_amount_l185_18555

theorem initial_amount (x : ℕ) (h1 : x - 3 + 14 = 22) : x = 11 :=
sorry

end NUMINAMATH_GPT_initial_amount_l185_18555


namespace NUMINAMATH_GPT_largest_possible_s_l185_18585

noncomputable def max_value_of_s (p q r s : ℝ) (h1 : p + q + r + s = 8) (h2 : pq + pr + ps + qr + qs + rs = 12) : ℝ :=
  2 + 3 * Real.sqrt 2

theorem largest_possible_s (p q r s : ℝ) (h1 : p + q + r + s = 8) (h2 : pq + pr + ps + qr + qs + rs = 12) :
  s ≤ max_value_of_s p q r s h1 h2 := 
sorry

end NUMINAMATH_GPT_largest_possible_s_l185_18585


namespace NUMINAMATH_GPT_cubic_inches_in_one_cubic_foot_l185_18587

-- Definition for the given conversion between foot and inches
def foot_to_inches : ℕ := 12

-- The theorem to prove the cubic conversion
theorem cubic_inches_in_one_cubic_foot : (foot_to_inches ^ 3) = 1728 := by
  -- Skipping the actual proof
  sorry

end NUMINAMATH_GPT_cubic_inches_in_one_cubic_foot_l185_18587


namespace NUMINAMATH_GPT_apple_tree_total_production_l185_18580

noncomputable def first_season_production : ℕ := 200
noncomputable def second_season_production : ℕ := 
  first_season_production - (first_season_production * 20 / 100)
noncomputable def third_season_production : ℕ := 
  second_season_production * 2
noncomputable def total_production : ℕ := 
  first_season_production + second_season_production + third_season_production

theorem apple_tree_total_production :
  total_production = 680 := by
  sorry

end NUMINAMATH_GPT_apple_tree_total_production_l185_18580


namespace NUMINAMATH_GPT_solve_fraction_equation_l185_18505

theorem solve_fraction_equation : ∀ (x : ℝ), (x + 2) / (2 * x - 1) = 1 → x = 3 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_solve_fraction_equation_l185_18505


namespace NUMINAMATH_GPT_smallest_integer_larger_than_expr_is_248_l185_18528

noncomputable def small_int_larger_than_expr : ℕ :=
  let expr := (Real.sqrt 5 + Real.sqrt 3)^4
  248

theorem smallest_integer_larger_than_expr_is_248 :
    ∃ (n : ℕ), n > (Real.sqrt 5 + Real.sqrt 3)^4 ∧ n = small_int_larger_than_expr := 
by
  -- We introduce the target integer 248
  use (248 : ℕ)
  -- The given conditions should lead us to 248 being greater than the expression.
  sorry

end NUMINAMATH_GPT_smallest_integer_larger_than_expr_is_248_l185_18528


namespace NUMINAMATH_GPT_fewest_reciprocal_keypresses_l185_18576

theorem fewest_reciprocal_keypresses (f : ℝ → ℝ) (x : ℝ) (hx : x ≠ 0) 
  (h1 : f 50 = 1 / 50) (h2 : f (1 / 50) = 50) : 
  ∃ n : ℕ, n = 2 ∧ (∀ m : ℕ, (m < n) → (f^[m] 50 ≠ 50)) :=
by
  sorry

end NUMINAMATH_GPT_fewest_reciprocal_keypresses_l185_18576


namespace NUMINAMATH_GPT_revenue_equation_l185_18535

theorem revenue_equation (x : ℝ) (r_j r_t : ℝ) (h1 : r_j = 90) (h2 : r_t = 144) :
  r_j + r_j * (1 + x) + r_j * (1 + x)^2 = r_t :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_revenue_equation_l185_18535


namespace NUMINAMATH_GPT_integer_value_l185_18514

theorem integer_value (x y z : ℕ) (h1 : 2 * x = 5 * y) (h2 : 5 * y = 6 * z) (h3 : x > 0) (h4 : y > 0) (h5 : z > 0) :
  ∃ a : ℕ, a + y + z = 26 ∧ a = 15 := by
  sorry

end NUMINAMATH_GPT_integer_value_l185_18514


namespace NUMINAMATH_GPT_remainder_of_7_pow_308_mod_11_l185_18574

theorem remainder_of_7_pow_308_mod_11 :
  (7 ^ 308) % 11 = 9 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_7_pow_308_mod_11_l185_18574


namespace NUMINAMATH_GPT_ab_value_l185_18556

theorem ab_value (a b : ℝ) :
  (A = { x : ℝ | x^2 - 8 * x + 15 = 0 }) ∧
  (B = { x : ℝ | x^2 - a * x + b = 0 }) ∧
  (A ∪ B = {2, 3, 5}) ∧
  (A ∩ B = {3}) →
  (a * b = 30) :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l185_18556


namespace NUMINAMATH_GPT_total_race_distance_l185_18533

theorem total_race_distance :
  let sadie_time := 2
  let sadie_speed := 3
  let ariana_time := 0.5
  let ariana_speed := 6
  let total_time := 4.5
  let sarah_speed := 4
  let sarah_time := total_time - sadie_time - ariana_time
  let sadie_distance := sadie_speed * sadie_time
  let ariana_distance := ariana_speed * ariana_time
  let sarah_distance := sarah_speed * sarah_time
  let total_distance := sadie_distance + ariana_distance + sarah_distance
  total_distance = 17 :=
by
  sorry

end NUMINAMATH_GPT_total_race_distance_l185_18533


namespace NUMINAMATH_GPT_min_distance_ants_l185_18552

open Real

theorem min_distance_ants (points : Fin 1390 → ℝ × ℝ) :
  (∀ i j : Fin 1390, i ≠ j → dist (points i) (points j) > 0.02) → 
  (∀ i : Fin 1390, |(points i).snd| < 0.01) → 
  ∃ i j : Fin 1390, i ≠ j ∧ dist (points i) (points j) > 10 :=
by
  sorry

end NUMINAMATH_GPT_min_distance_ants_l185_18552


namespace NUMINAMATH_GPT_calc_perimeter_l185_18569

noncomputable def width (w: ℝ) (h: ℝ) : Prop :=
  h = w + 10

noncomputable def cost (P: ℝ) (rate: ℝ) (total_cost: ℝ) : Prop :=
  total_cost = P * rate

noncomputable def perimeter (w: ℝ) (P: ℝ) : Prop :=
  P = 2 * (w + (w + 10))

theorem calc_perimeter {w P : ℝ} (h_rate : ℝ) (h_total_cost : ℝ)
  (h1 : width w (w + 10))
  (h2 : cost (2 * (w + (w + 10))) h_rate h_total_cost) :
  P = 2 * (w + (w + 10)) →
  h_total_cost = 910 →
  h_rate = 6.5 →
  w = 30 →
  P = 140 :=
sorry

end NUMINAMATH_GPT_calc_perimeter_l185_18569


namespace NUMINAMATH_GPT_math_problem_l185_18547

theorem math_problem :
  (-1 : ℝ)^(53) + 3^(2^3 + 5^2 - 7^2) = -1 + (1 / 3^(16)) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l185_18547


namespace NUMINAMATH_GPT_kernel_count_in_final_bag_l185_18577

namespace PopcornKernelProblem

def percentage_popped (popped total : ℕ) : ℤ := ((popped : ℤ) * 100) / (total : ℤ)

def first_bag_percentage := percentage_popped 60 75
def second_bag_percentage := percentage_popped 42 50
def final_bag_percentage (x : ℕ) : ℤ := percentage_popped 82 x

theorem kernel_count_in_final_bag :
  (first_bag_percentage + second_bag_percentage + final_bag_percentage 100) / 3 = 82 := 
sorry

end PopcornKernelProblem

end NUMINAMATH_GPT_kernel_count_in_final_bag_l185_18577


namespace NUMINAMATH_GPT_initial_points_l185_18538

theorem initial_points (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_points_l185_18538


namespace NUMINAMATH_GPT_total_art_cost_l185_18540

-- Definitions based on the conditions
def total_price_first_3_pieces (price_per_piece : ℤ) : ℤ :=
  price_per_piece * 3

def price_increase (price_per_piece : ℤ) : ℤ :=
  price_per_piece / 2

def total_price_all_arts (price_per_piece next_piece_price : ℤ) : ℤ :=
  (total_price_first_3_pieces price_per_piece) + next_piece_price

-- The proof problem statement
theorem total_art_cost : 
  ∀ (price_per_piece : ℤ),
  total_price_first_3_pieces price_per_piece = 45000 →
  next_piece_price = price_per_piece + price_increase price_per_piece →
  total_price_all_arts price_per_piece next_piece_price = 67500 :=
  by
    intros price_per_piece h1 h2
    sorry

end NUMINAMATH_GPT_total_art_cost_l185_18540


namespace NUMINAMATH_GPT_trapezoid_area_correct_l185_18596

noncomputable def calculate_trapezoid_area : ℕ :=
  let parallel_side_1 := 6
  let parallel_side_2 := 12
  let leg := 5
  let radius := 5
  let height := radius
  let area := (1 / 2) * (parallel_side_1 + parallel_side_2) * height
  area

theorem trapezoid_area_correct :
  calculate_trapezoid_area = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_trapezoid_area_correct_l185_18596


namespace NUMINAMATH_GPT_green_ball_probability_l185_18536

def containerA := (8, 2) -- 8 green, 2 red
def containerB := (6, 4) -- 6 green, 4 red
def containerC := (5, 5) -- 5 green, 5 red
def containerD := (8, 2) -- 8 green, 2 red

def probability_of_green : ℚ :=
  (1 / 4) * (8 / 10) + (1 / 4) * (6 / 10) + (1 / 4) * (5 / 10) + (1 / 4) * (8 / 10)
  
theorem green_ball_probability :
  probability_of_green = 43 / 160 :=
sorry

end NUMINAMATH_GPT_green_ball_probability_l185_18536


namespace NUMINAMATH_GPT_part1_part2_l185_18562

-- Definitions of propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 = 0 ∧ x < 0 ∧ (∃ y : ℝ, y ≠ x ∧ y^2 + m * y + 1 = 0 ∧ y < 0)
def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≠ 0

-- Lean statement for part 1
theorem part1 (m : ℝ) :
  ¬ ¬ p m → m > 2 :=
sorry

-- Lean statement for part 2
theorem part2 (m : ℝ) :
  (p m ∨ q m) ∧ (¬(p m ∧ q m)) → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
sorry

end NUMINAMATH_GPT_part1_part2_l185_18562


namespace NUMINAMATH_GPT_min_value_expression_l185_18545

open Real

theorem min_value_expression (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
    (h_abc : a * b * c = 1 / 2) : 
    a^2 + 8 * a * b + 32 * b^2 + 16 * b * c + 8 * c^2 ≥ 18 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l185_18545


namespace NUMINAMATH_GPT_number_solution_exists_l185_18527

theorem number_solution_exists (x : ℝ) (h : 0.80 * x = (4 / 5 * 15) + 20) : x = 40 :=
sorry

end NUMINAMATH_GPT_number_solution_exists_l185_18527


namespace NUMINAMATH_GPT_geometric_sequence_8th_term_l185_18503

theorem geometric_sequence_8th_term (a : ℚ) (r : ℚ) (n : ℕ) (h_a : a = 27) (h_r : r = 2/3) (h_n : n = 8) :
  a * r^(n-1) = 128 / 81 :=
by
  rw [h_a, h_r, h_n]
  sorry

end NUMINAMATH_GPT_geometric_sequence_8th_term_l185_18503


namespace NUMINAMATH_GPT_abs_neg_two_eq_two_neg_two_pow_zero_eq_one_l185_18539

theorem abs_neg_two_eq_two : |(-2)| = 2 :=
sorry

theorem neg_two_pow_zero_eq_one : (-2)^0 = 1 :=
sorry

end NUMINAMATH_GPT_abs_neg_two_eq_two_neg_two_pow_zero_eq_one_l185_18539


namespace NUMINAMATH_GPT_problem_1_problem_2_l185_18575

noncomputable def f (x p : ℝ) := p * x - p / x - 2 * Real.log x
noncomputable def g (x : ℝ) := 2 * Real.exp 1 / x

theorem problem_1 (p : ℝ) : 
  (∀ x : ℝ, 0 < x → p * x - p / x - 2 * Real.log x ≥ 0) ↔ p ≥ 1 := 
by sorry

theorem problem_2 (p : ℝ) : 
  (∃ x_0 : ℝ, 1 ≤ x_0 ∧ x_0 ≤ Real.exp 1 ∧ f x_0 p > g x_0) ↔ 
  p > 4 * Real.exp 1 / (Real.exp 2 - 1) :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_l185_18575


namespace NUMINAMATH_GPT_infinite_grid_rectangles_l185_18523

theorem infinite_grid_rectangles (m : ℕ) (hm : m > 12) : 
  ∃ (x y : ℕ), x * y > m ∧ x * (y - 1) < m := 
  sorry

end NUMINAMATH_GPT_infinite_grid_rectangles_l185_18523


namespace NUMINAMATH_GPT_cars_with_neither_features_l185_18517

-- Define the given conditions
def total_cars : ℕ := 65
def cars_with_power_steering : ℕ := 45
def cars_with_power_windows : ℕ := 25
def cars_with_both_features : ℕ := 17

-- Define the statement to be proved
theorem cars_with_neither_features : total_cars - (cars_with_power_steering + cars_with_power_windows - cars_with_both_features) = 12 :=
by
  sorry

end NUMINAMATH_GPT_cars_with_neither_features_l185_18517


namespace NUMINAMATH_GPT_boat_distance_downstream_l185_18526

theorem boat_distance_downstream (v_s : ℝ) (h : 8 - v_s = 5) :
  8 + v_s = 11 :=
by
  sorry

end NUMINAMATH_GPT_boat_distance_downstream_l185_18526


namespace NUMINAMATH_GPT_percentage_paid_X_vs_Y_l185_18597

theorem percentage_paid_X_vs_Y (X Y : ℝ) (h1 : X + Y = 528) (h2 : Y = 240) :
  ((X / Y) * 100) = 120 :=
by
  sorry

end NUMINAMATH_GPT_percentage_paid_X_vs_Y_l185_18597


namespace NUMINAMATH_GPT_sqrt_123400_l185_18558

theorem sqrt_123400 (h1: Real.sqrt 12.34 = 3.512) : Real.sqrt 123400 = 351.2 :=
by 
  sorry

end NUMINAMATH_GPT_sqrt_123400_l185_18558


namespace NUMINAMATH_GPT_xyz_inequality_l185_18521

theorem xyz_inequality : ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
  Real.sqrt (a * b * c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) + (a + b + c)^2 ≥ 
  4 * Real.sqrt (3 * a * b * c * (a + b + c)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_xyz_inequality_l185_18521


namespace NUMINAMATH_GPT_tan_ratio_of_triangle_sides_l185_18541

theorem tan_ratio_of_triangle_sides (a b c : ℝ) (α β γ : ℝ) 
  (h1 : a^2 + b^2 = 2023 * c^2)
  (h2 : α + β + γ = π)
  (h3 : c ≠ 0):
  ( (Real.tan γ) / (Real.tan α + Real.tan β) ) = (a * b) / (1011 * c^2) := 
sorry

end NUMINAMATH_GPT_tan_ratio_of_triangle_sides_l185_18541


namespace NUMINAMATH_GPT_find_m_l185_18509

variable (m : ℝ)

-- Definitions of the vectors
def AB : ℝ × ℝ := (m + 3, 2 * m + 1)
def CD : ℝ × ℝ := (m + 3, -5)

-- Definition of perpendicular vectors, dot product is zero
def perp (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem find_m (h : perp (AB m) (CD m)) : m = 2 := by
  sorry

end NUMINAMATH_GPT_find_m_l185_18509


namespace NUMINAMATH_GPT_water_balloon_packs_l185_18564

theorem water_balloon_packs (P : ℕ) : 
  (6 * P + 12 = 30) → P = 3 := by
  sorry

end NUMINAMATH_GPT_water_balloon_packs_l185_18564


namespace NUMINAMATH_GPT_child_ticket_cost_l185_18551

variable (A C : ℕ) -- A stands for the number of adults, C stands for the cost of one child's ticket

theorem child_ticket_cost 
  (number_of_adults : ℕ) 
  (number_of_children : ℕ) 
  (cost_concessions : ℕ) 
  (total_cost_trip : ℕ)
  (cost_adult_ticket : ℕ) 
  (ticket_costs : ℕ) 
  (total_adult_cost : ℕ) 
  (remaining_ticket_cost : ℕ) 
  (child_ticket : ℕ) :
  number_of_adults = 5 →
  number_of_children = 2 →
  cost_concessions = 12 →
  total_cost_trip = 76 →
  cost_adult_ticket = 10 →
  ticket_costs = total_cost_trip - cost_concessions →
  total_adult_cost = number_of_adults * cost_adult_ticket →
  remaining_ticket_cost = ticket_costs - total_adult_cost →
  child_ticket = remaining_ticket_cost / number_of_children →
  C = 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  -- Adding sorry since the proof is not required
  sorry

end NUMINAMATH_GPT_child_ticket_cost_l185_18551


namespace NUMINAMATH_GPT_find_number_l185_18520

def divisor : ℕ := 22
def quotient : ℕ := 12
def remainder : ℕ := 1
def number : ℕ := (divisor * quotient) + remainder

theorem find_number : number = 265 := by
  sorry

end NUMINAMATH_GPT_find_number_l185_18520


namespace NUMINAMATH_GPT_monotonic_exponential_decreasing_l185_18560

variable (a : ℝ) (f : ℝ → ℝ)

theorem monotonic_exponential_decreasing {m n : ℝ}
  (h0 : a = (Real.sqrt 5 - 1) / 2)
  (h1 : ∀ x, f x = a^x)
  (h2 : 0 < a ∧ a < 1)
  (h3 : f m > f n) :
  m < n :=
sorry

end NUMINAMATH_GPT_monotonic_exponential_decreasing_l185_18560


namespace NUMINAMATH_GPT_original_cost_price_40_l185_18531

theorem original_cost_price_40
  (selling_price : ℝ)
  (decrease_rate : ℝ)
  (profit_increase_rate : ℝ)
  (new_selling_price := selling_price)
  (original_cost_price : ℝ)
  (new_cost_price := (1 - decrease_rate) * original_cost_price)
  (original_profit_margin := (selling_price - original_cost_price) / original_cost_price)
  (new_profit_margin := (new_selling_price - new_cost_price) / new_cost_price)
  (profit_margin_increase := profit_increase_rate)
  (h1 : selling_price = 48)
  (h2 : decrease_rate = 0.04)
  (h3 : profit_increase_rate = 0.05)
  (h4 : new_profit_margin = original_profit_margin + profit_margin_increase) :
  original_cost_price = 40 := 
by 
  sorry

end NUMINAMATH_GPT_original_cost_price_40_l185_18531


namespace NUMINAMATH_GPT_largest_angle_is_90_degrees_l185_18550

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem largest_angle_is_90_degrees (u : ℝ) (a b c : ℝ) (v : ℝ) (h_v : v = 1)
  (h_a : a = Real.sqrt (2 * u - 1))
  (h_b : b = Real.sqrt (2 * u + 3))
  (h_c : c = 2 * Real.sqrt (u + v)) :
  is_right_triangle a b c :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_is_90_degrees_l185_18550


namespace NUMINAMATH_GPT_percentage_of_girls_l185_18554

def total_students : ℕ := 100
def boys : ℕ := 50
def girls : ℕ := total_students - boys

theorem percentage_of_girls :
  (girls / total_students) * 100 = 50 := sorry

end NUMINAMATH_GPT_percentage_of_girls_l185_18554


namespace NUMINAMATH_GPT_bisection_method_third_interval_l185_18563

theorem bisection_method_third_interval 
  (f : ℝ → ℝ) (a b : ℝ) (H1 : a = -2) (H2 : b = 4) 
  (H3 : f a * f b ≤ 0) : 
  ∃ c d : ℝ, c = -1/2 ∧ d = 1 ∧ f c * f d ≤ 0 :=
by 
  sorry

end NUMINAMATH_GPT_bisection_method_third_interval_l185_18563


namespace NUMINAMATH_GPT_tyler_total_puppies_l185_18589

/-- 
  Tyler has 15 dogs, and each dog has 5 puppies.
  We want to prove that the total number of puppies is 75.
-/
def tyler_dogs : Nat := 15
def puppies_per_dog : Nat := 5
def total_puppies_tyler_has : Nat := tyler_dogs * puppies_per_dog

theorem tyler_total_puppies : total_puppies_tyler_has = 75 := by
  sorry

end NUMINAMATH_GPT_tyler_total_puppies_l185_18589


namespace NUMINAMATH_GPT_height_of_fourth_person_l185_18586

/-- There are 4 people of different heights standing in order of increasing height.
    The difference is 2 inches between the first person and the second person,
    and also between the second person and the third person.
    The difference between the third person and the fourth person is 6 inches.
    The average height of the four people is 76 inches.
    Prove that the height of the fourth person is 82 inches. -/
theorem height_of_fourth_person 
  (h1 h2 h3 h4 : ℕ) 
  (h2_def : h2 = h1 + 2)
  (h3_def : h3 = h2 + 2)
  (h4_def : h4 = h3 + 6)
  (average_height : (h1 + h2 + h3 + h4) / 4 = 76) 
  : h4 = 82 :=
by sorry

end NUMINAMATH_GPT_height_of_fourth_person_l185_18586


namespace NUMINAMATH_GPT_num_sides_regular_polygon_l185_18595

-- Define the perimeter and side length of the polygon
def perimeter : ℝ := 160
def side_length : ℝ := 10

-- Theorem to prove the number of sides
theorem num_sides_regular_polygon : 
  (perimeter / side_length) = 16 := by
    sorry  -- Proof is omitted

end NUMINAMATH_GPT_num_sides_regular_polygon_l185_18595


namespace NUMINAMATH_GPT_tan_double_angle_l185_18565

theorem tan_double_angle (α : ℝ) (h : 3 * Real.cos α + Real.sin α = 0) : 
    Real.tan (2 * α) = 3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l185_18565


namespace NUMINAMATH_GPT_multiplication_factor_l185_18519

theorem multiplication_factor
  (n : ℕ) (avg_orig avg_new : ℝ) (F : ℝ)
  (H1 : n = 7)
  (H2 : avg_orig = 24)
  (H3 : avg_new = 120)
  (H4 : (n * avg_new) = F * (n * avg_orig)) :
  F = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_multiplication_factor_l185_18519
