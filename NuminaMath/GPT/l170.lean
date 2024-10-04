import Mathlib

namespace solution_exists_iff_divisor_form_l170_170709

theorem solution_exists_iff_divisor_form (n : ℕ) (hn_pos : 0 < n) (hn_odd : n % 2 = 1) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 4 * x * y = n * (x + y)) ↔
    (∃ k : ℕ, n % (4 * k + 3) = 0) :=
by
  sorry

end solution_exists_iff_divisor_form_l170_170709


namespace cloth_cost_l170_170338

theorem cloth_cost
  (L : ℕ)
  (C : ℚ)
  (hL : L = 10)
  (h_condition : L * C = (L + 4) * (C - 1)) :
  10 * C = 35 := by
  sorry

end cloth_cost_l170_170338


namespace left_handed_rock_lovers_l170_170290

theorem left_handed_rock_lovers (total_people left_handed rock_music right_dislike_rock x : ℕ) :
  total_people = 30 →
  left_handed = 14 →
  rock_music = 20 →
  right_dislike_rock = 5 →
  (x + (left_handed - x) + (rock_music - x) + right_dislike_rock = total_people) →
  x = 9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end left_handed_rock_lovers_l170_170290


namespace first_other_factor_of_lcm_l170_170582

theorem first_other_factor_of_lcm (A B hcf lcm : ℕ) (h1 : A = 368) (h2 : hcf = 23) (h3 : lcm = hcf * 16 * X) :
  X = 1 :=
by
  sorry

end first_other_factor_of_lcm_l170_170582


namespace range_of_k_for_one_solution_l170_170677

-- Definitions
def angle_B : ℝ := 60 -- Angle B in degrees
def side_b : ℝ := 12 -- Length of side b
def side_a (k : ℝ) : ℝ := k -- Length of side a (parameterized by k)

-- Theorem stating the range of k that makes the side_a have exactly one solution
theorem range_of_k_for_one_solution (k : ℝ) : (0 < k ∧ k <= 12) ∨ k = 8 * Real.sqrt 3 := 
sorry

end range_of_k_for_one_solution_l170_170677


namespace problem_statement_l170_170398

variable {x y : ℝ}

theorem problem_statement (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : y - 2 / x ≠ 0) :
  (2 * x - 3 / y) / (3 * y - 2 / x) = (2 * x * y - 3) / (3 * x * y - 2) :=
sorry

end problem_statement_l170_170398


namespace sum_of_x_values_l170_170971

theorem sum_of_x_values (x : ℂ) (h₁ : x ≠ -3) (h₂ : 3 = (x^3 - 3 * x^2 - 10 * x) / (x + 3)) : x + (5 - x) = 5 :=
sorry

end sum_of_x_values_l170_170971


namespace desired_alcohol_percentage_l170_170336

def initial_volume := 6.0
def initial_percentage := 35.0 / 100.0
def added_alcohol := 1.8
def final_volume := initial_volume + added_alcohol
def initial_alcohol := initial_volume * initial_percentage
def final_alcohol := initial_alcohol + added_alcohol
def desired_percentage := (final_alcohol / final_volume) * 100.0

theorem desired_alcohol_percentage : desired_percentage = 50.0 := 
by
  -- Proof would go here, but is omitted as per the instructions
  sorry

end desired_alcohol_percentage_l170_170336


namespace complement_intersection_subset_condition_l170_170369

-- Definition of sets A, B, and C
def A := { x : ℝ | 3 ≤ x ∧ x < 7 }
def B := { x : ℝ | 2 < x ∧ x < 10 }
def C (a : ℝ) := { x : ℝ | x < a }

-- Proof problem 1 statement
theorem complement_intersection :
  ( { x : ℝ | x < 3 ∨ x ≥ 7 } ∩ { x : ℝ | 2 < x ∧ x < 10 } ) = { x : ℝ | 2 < x ∧ x < 3 ∨ 7 ≤ x ∧ x < 10 } :=
by
  sorry

-- Proof problem 2 statement
theorem subset_condition (a : ℝ) :
  ( { x : ℝ | 3 ≤ x ∧ x < 7 } ⊆ { x : ℝ | x < a } ) → (a ≥ 7) :=
by
  sorry

end complement_intersection_subset_condition_l170_170369


namespace handshake_count_l170_170223

theorem handshake_count (n_twins: ℕ) (n_triplets: ℕ)
  (twin_pairs: ℕ) (triplet_groups: ℕ)
  (handshakes_twin : ∀ (x: ℕ), x = (n_twins - 2))
  (handshakes_triplet : ∀ (y: ℕ), y = (n_triplets - 3))
  (handshakes_cross_twins : ∀ (z: ℕ), z = 3*n_triplets / 4)
  (handshakes_cross_triplets : ∀ (w: ℕ), w = n_twins / 4) :
  2 * (n_twins * (n_twins -1 -1) / 2 + n_triplets * (n_triplets - 1 - 1) / 2 + n_twins * (3*n_triplets / 4) + n_triplets * (n_twins / 4)) / 2 = 804 := 
sorry

end handshake_count_l170_170223


namespace number_of_levels_l170_170807

-- Definitions of the conditions
def blocks_per_step : ℕ := 3
def steps_per_level : ℕ := 8
def total_blocks_climbed : ℕ := 96

-- The theorem to prove
theorem number_of_levels : (total_blocks_climbed / blocks_per_step) / steps_per_level = 4 := by
  sorry

end number_of_levels_l170_170807


namespace profit_increase_l170_170847

theorem profit_increase (x y : ℝ) (a : ℝ) (hx_pos : x > 0) (hy_pos : y > 0)
  (profit_eq : y - x = x * (a / 100))
  (new_profit_eq : y - 0.95 * x = 0.95 * x * (a / 100) + 0.95 * x * (15 / 100)) :
  a = 185 :=
by
  sorry

end profit_increase_l170_170847


namespace math_problem_l170_170372

open Real

variables (a b b1 : ℝ) (F1 F2 M : ℝ × ℝ)
variables (C1 C2 : Set (ℝ × ℝ))

-- Defining the foci and intersection point
def F1 := (-1, 0)
def F2 := (1, 0)
def M := (2 * sqrt 3 / 3, sqrt 3 / 3)

-- Ellipse C1 and Hyperbola C2
def is_ellipse (C : Set (ℝ × ℝ)) := ∃ a b, a > 0 ∧ b > 0 ∧ a > b ∧
  (∀ p ∈ C, (p.1^2 / a^2) + (p.2^2 / b^2) = 1)
def is_hyperbola (C : Set (ℝ × ℝ)) := ∃ b1, b1 > 0 ∧
  (∀ p ∈ C, (p.1^2) - (p.2^2 / b1^2) = 1)

-- Conditions of the problem
def intersection_condition (p : ℝ × ℝ) :=
  p ∈ C1 ∧ p ∈ C2 ∧ p = M

def hyperbola_vertices := F1 ∈ C2 ∧ F2 ∈ C2

-- Fixed point condition
def fixed_point_condition := 
  ∃ P Q : ℝ × ℝ, 
    (P.1^2 - P.2^2 = 1) ∧ 
    (Q.1 = 0) ∧ 
    (F1.2 ≠ Q.2) ∧ 
    (P.2 / (P.1 + 1) = - (Q.1 + 1) / (Q.2)) ∧ 
    (∃ K : ℝ × ℝ, line_through R P Q = line_through R (1, 0))

theorem math_problem :
  is_ellipse C1 ∧ is_hyperbola C2 ∧ intersection_condition M ∧ hyperbola_vertices → fixed_point_condition := 
by 
  sorry

end math_problem_l170_170372


namespace sqrt_inequality_l170_170944

theorem sqrt_inequality : 2 * Real.sqrt 2 - Real.sqrt 7 < Real.sqrt 6 - Real.sqrt 5 := by sorry

end sqrt_inequality_l170_170944


namespace A_minus_one_not_prime_l170_170663

theorem A_minus_one_not_prime (n : ℕ) (h : 0 < n) (m : ℕ) (h1 : 10^(m-1) < 14^n) (h2 : 14^n < 10^m) :
  ¬ (Nat.Prime (2^n * 10^m + 14^n - 1)) :=
by
  sorry

end A_minus_one_not_prime_l170_170663


namespace lattice_points_in_region_l170_170348

theorem lattice_points_in_region : ∃ n : ℕ, n = 1 ∧ ∀ p : ℤ × ℤ, 
  (p.snd = abs p.fst ∨ p.snd = -(p.fst ^ 3) + 6 * (p.fst)) → n = 1 :=
by
  sorry

end lattice_points_in_region_l170_170348


namespace problem_statement_l170_170547

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 :=
by
  sorry

end problem_statement_l170_170547


namespace plus_signs_count_l170_170028

theorem plus_signs_count (total_symbols : ℕ) (n : ℕ) (m : ℕ)
    (h1 : total_symbols = 23)
    (h2 : ∀ (s : Finset ℕ), s.card = 10 → (∃ x ∈ s, x = n))
    (h3 : ∀ (s : Finset ℕ), s.card = 15 → (∃ x ∈ s, x = m)) :
    n = 14 := by
  sorry

end plus_signs_count_l170_170028


namespace profit_function_l170_170925

def cost_per_unit : ℝ := 8

def daily_sales_quantity (x : ℝ) : ℝ := -x + 30

def profit_per_unit (x : ℝ) : ℝ := x - cost_per_unit

def total_profit (x : ℝ) : ℝ := (profit_per_unit x) * (daily_sales_quantity x)

theorem profit_function (x : ℝ) : total_profit x = -x^2 + 38*x - 240 :=
  sorry

end profit_function_l170_170925


namespace Mary_works_hours_on_Tuesday_and_Thursday_l170_170715

theorem Mary_works_hours_on_Tuesday_and_Thursday 
  (h_mon_wed_fri : ∀ (d : ℕ), d = 3 → 9 * d = 27)
  (weekly_earnings : ℕ)
  (hourly_rate : ℕ)
  (weekly_hours_mon_wed_fri : ℕ)
  (tue_thu_hours : ℕ) :
  weekly_earnings = 407 →
  hourly_rate = 11 →
  weekly_hours_mon_wed_fri = 9 * 3 →
  weekly_earnings - weekly_hours_mon_wed_fri * hourly_rate = tue_thu_hours * hourly_rate →
  tue_thu_hours = 10 :=
by
  intros hearnings hrate hweek hsub
  sorry

end Mary_works_hours_on_Tuesday_and_Thursday_l170_170715


namespace configuration_of_points_l170_170520

-- Define a type for points
structure Point :=
(x : ℝ)
(y : ℝ)

-- Assuming general position in the plane
def general_position (points : List Point) : Prop :=
  -- Add definition of general position, skipping exact implementation
  sorry

-- Define the congruence condition
def triangles_congruent (points : List Point) : Prop :=
  -- Add definition of the congruent triangles condition
  sorry

-- Define the vertices of two equilateral triangles inscribed in a circle
def two_equilateral_triangles (points : List Point) : Prop :=
  -- Add definition to check if points form two equilateral triangles in a circle
  sorry

theorem configuration_of_points (points : List Point) (h6 : points.length = 6) :
  general_position points →
  triangles_congruent points →
  two_equilateral_triangles points :=
by
  sorry

end configuration_of_points_l170_170520


namespace count_of_plus_signs_l170_170033

-- Definitions based on the given conditions
def total_symbols : ℕ := 23
def any_10_symbols_contains_plus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 10 → (\exists c ∈ s, c = '+')
def any_15_symbols_contains_minus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 15 → (\exists c ∈ s, c = '-')

-- Main theorem statement without the proof
theorem count_of_plus_signs (symbols : Finset char) (h1 : symbols.card = total_symbols) 
  (h2 : any_10_symbols_contains_plus symbols) 
  (h3 : any_15_symbols_contains_minus symbols) : 
  symbols.count '+' = 14 :=
sorry

end count_of_plus_signs_l170_170033


namespace find_number_l170_170113

theorem find_number (x : ℕ) (n : ℕ) (h1 : x = 4) (h2 : x + n = 5) : n = 1 :=
by
  sorry

end find_number_l170_170113


namespace find_FC_l170_170364

variable (DC : ℝ) (CB : ℝ) (AB AD ED : ℝ)
variable (FC : ℝ)
variable (h1 : DC = 9)
variable (h2 : CB = 6)
variable (h3 : AB = (1/3) * AD)
variable (h4 : ED = (2/3) * AD)

theorem find_FC : FC = 9 :=
by sorry

end find_FC_l170_170364


namespace Roger_first_bag_candies_is_11_l170_170315

-- Define the conditions
def Sandra_bags : ℕ := 2
def Sandra_candies_per_bag : ℕ := 6
def Roger_bags : ℕ := 2
def Roger_second_bag_candies : ℕ := 3
def Extra_candies_Roger_has_than_Sandra : ℕ := 2

-- Define the total candy for Sandra
def Sandra_total_candies : ℕ := Sandra_bags * Sandra_candies_per_bag

-- Using the conditions, we define the total candy for Roger
def Roger_total_candies : ℕ := Sandra_total_candies + Extra_candies_Roger_has_than_Sandra

-- Define the candy in Roger's first bag
def Roger_first_bag_candies : ℕ := Roger_total_candies - Roger_second_bag_candies

-- The proof statement we need to prove
theorem Roger_first_bag_candies_is_11 : Roger_first_bag_candies = 11 := by
  sorry

end Roger_first_bag_candies_is_11_l170_170315


namespace liquid_x_percentage_l170_170918

theorem liquid_x_percentage (a_weight b_weight : ℝ) (a_percentage b_percentage : ℝ)
  (result_weight : ℝ) (x_weight_result : ℝ) (x_percentage_result : ℝ) :
  a_weight = 500 → b_weight = 700 → a_percentage = 0.8 / 100 →
  b_percentage = 1.8 / 100 → result_weight = a_weight + b_weight →
  x_weight_result = a_weight * a_percentage + b_weight * b_percentage →
  x_percentage_result = (x_weight_result / result_weight) * 100 →
  x_percentage_result = 1.3833 :=
by sorry

end liquid_x_percentage_l170_170918


namespace line_through_diameter_l170_170247

theorem line_through_diameter (P : ℝ × ℝ) (hP : P = (2, 1)) (h_circle : ∀ x y : ℝ, (x - 1)^2 + y^2 = 4) :
  ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = -1 :=
by
  exists 1, -1, -1
  sorry

end line_through_diameter_l170_170247


namespace count_numbers_with_property_l170_170604

open Nat

theorem count_numbers_with_property : 
  let N := { n : ℕ | n < 10^6 ∧ ∃ k : ℕ, 1 ≤ k ∧ k ≤ 43 ∧ 2012 ∣ n^k - 1 }
  N.card = 1988 :=
by
  sorry

end count_numbers_with_property_l170_170604


namespace repeating_decimal_subtraction_l170_170465

noncomputable def x := (0.246 : Real)
noncomputable def y := (0.135 : Real)
noncomputable def z := (0.579 : Real)

theorem repeating_decimal_subtraction :
  x - y - z = (-156 : ℚ) / 333 :=
by
  sorry

end repeating_decimal_subtraction_l170_170465


namespace admission_cutoff_score_l170_170490

theorem admission_cutoff_score (n : ℕ) (x : ℚ) (admitted_average non_admitted_average total_average : ℚ)
    (h1 : admitted_average = x + 15)
    (h2 : non_admitted_average = x - 20)
    (h3 : total_average = 90)
    (h4 : (admitted_average * (2 / 5) + non_admitted_average * (3 / 5)) = total_average) : x = 96 := 
by
  sorry

end admission_cutoff_score_l170_170490


namespace expected_value_of_winnings_equals_3_l170_170072

noncomputable def expected_value_of_winnings (p : ℕ → ℚ) : ℚ :=
  ∑ k in [2, 4, 6, 8], p k * k + p 3 * 2 + p 5 * 2

theorem expected_value_of_winnings_equals_3 :
  let p : ℕ → ℚ := λ k, if k ∈ [1, 2, 3, 4, 5, 6, 7, 8] then 1/8 else 0 in
  expected_value_of_winnings p = 3 := by
  sorry

end expected_value_of_winnings_equals_3_l170_170072


namespace third_group_members_l170_170481

theorem third_group_members (total_members first_group second_group : ℕ) (h₁ : total_members = 70) (h₂ : first_group = 25) (h₃ : second_group = 30) : (total_members - (first_group + second_group)) = 15 :=
sorry

end third_group_members_l170_170481


namespace set_complement_union_l170_170980

namespace ProblemOne

def A : Set ℝ := {x | x ≤ -3 ∨ x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem set_complement_union :
  (Aᶜ ∪ B) = {x : ℝ | -3 < x ∧ x < 5} := sorry

end ProblemOne

end set_complement_union_l170_170980


namespace points_satisfy_equation_l170_170230

theorem points_satisfy_equation :
  ∀ (x y : ℝ), x^2 - y^4 = Real.sqrt (18 * x - x^2 - 81) ↔ 
               (x = 9 ∧ y = Real.sqrt 3) ∨ (x = 9 ∧ y = -Real.sqrt 3) := 
by 
  intros x y 
  sorry

end points_satisfy_equation_l170_170230


namespace least_subtract_divisible_l170_170470

theorem least_subtract_divisible:
  ∃ n : ℕ, n = 31 ∧ (13603 - n) % 87 = 0 :=
by
  sorry

end least_subtract_divisible_l170_170470


namespace find_a_l170_170399

theorem find_a (a x y : ℝ) (h1 : ax - 3y = 0) (h2 : x + y = 1) (h3 : 2x + y = 0) : a = -6 := 
by sorry

end find_a_l170_170399


namespace cross_section_prism_in_sphere_l170_170638

noncomputable def cross_section_area 
  (a R : ℝ) 
  (h1 : a > 0) 
  (h2 : R > 0) 
  (h3 : a < 2 * R) : ℝ :=
  a * Real.sqrt (4 * R^2 - a^2)

theorem cross_section_prism_in_sphere 
  (a R : ℝ) 
  (h1 : a > 0) 
  (h2 : R > 0) 
  (h3 : a < 2 * R) :
  cross_section_area a R h1 h2 h3 = a * Real.sqrt (4 * R^2 - a^2) := 
  by
    sorry

end cross_section_prism_in_sphere_l170_170638


namespace sin_sum_arcsin_arctan_l170_170951

-- Definitions matching the conditions
def a := Real.arcsin (4 / 5)
def b := Real.arctan (1 / 2)

-- Theorem stating the question and expected answer
theorem sin_sum_arcsin_arctan : 
  Real.sin (a + b) = (11 * Real.sqrt 5) / 25 := 
by 
  sorry

end sin_sum_arcsin_arctan_l170_170951


namespace maximum_value_of_M_l170_170530

noncomputable def M (x : ℝ) : ℝ :=
  (Real.sin x * (2 - Real.cos x)) / (5 - 4 * Real.cos x)

theorem maximum_value_of_M : 
  ∃ x : ℝ, M x = (Real.sqrt 3) / 4 :=
sorry

end maximum_value_of_M_l170_170530


namespace find_slope_of_parallel_line_l170_170733

-- Define the condition that line1 is parallel to line2.
def lines_parallel (k : ℝ) : Prop :=
  k = -3

-- The theorem that proves the condition given.
theorem find_slope_of_parallel_line (k : ℝ) (h : lines_parallel k) : k = -3 :=
by
  exact h

end find_slope_of_parallel_line_l170_170733


namespace scout_weekend_earnings_l170_170168

theorem scout_weekend_earnings : 
  let base_pay_per_hour := 10.00
  let tip_per_customer := 5.00
  let saturday_hours := 4
  let saturday_customers := 5
  let sunday_hours := 5
  let sunday_customers := 8
  in
  (saturday_hours * base_pay_per_hour + saturday_customers * tip_per_customer) +
  (sunday_hours * base_pay_per_hour + sunday_customers * tip_per_customer) = 155.00 := sorry

end scout_weekend_earnings_l170_170168


namespace time_after_9999_seconds_l170_170866

theorem time_after_9999_seconds :
  let initial_time := Time.mk 7 45 0 in
  let duration_sec := 9999 in
  let end_time := Time.mk 10 31 39 in
  add_seconds_to_time initial_time duration_sec = end_time :=
by
  sorry

end time_after_9999_seconds_l170_170866


namespace bus_stop_time_l170_170517

theorem bus_stop_time (speed_excl_stops speed_incl_stops : ℝ) (h1 : speed_excl_stops = 50) (h2 : speed_incl_stops = 45) : (60 * ((speed_excl_stops - speed_incl_stops) / speed_excl_stops)) = 6 := 
by
  sorry

end bus_stop_time_l170_170517


namespace average_class_weight_l170_170753

theorem average_class_weight :
  let students_A := 50
  let weight_A := 60
  let students_B := 60
  let weight_B := 80
  let students_C := 70
  let weight_C := 75
  let students_D := 80
  let weight_D := 85
  let total_students := students_A + students_B + students_C + students_D
  let total_weight := students_A * weight_A + students_B * weight_B + students_C * weight_C + students_D * weight_D
  (total_weight / total_students : ℝ) = 76.35 :=
by
  sorry

end average_class_weight_l170_170753


namespace original_portion_al_l170_170217

variable (a b c : ℕ)

theorem original_portion_al :
  a + b + c = 1200 ∧
  a - 150 + 3 * b + 3 * c = 1800 ∧
  c = 2 * b →
  a = 825 :=
by
  sorry

end original_portion_al_l170_170217


namespace boys_to_girls_ratio_l170_170856

theorem boys_to_girls_ratio (boys girls : ℕ) (h_boys : boys = 1500) (h_girls : girls = 1200) : 
  (boys / Nat.gcd boys girls) = 5 ∧ (girls / Nat.gcd boys girls) = 4 := 
by 
  sorry

end boys_to_girls_ratio_l170_170856


namespace least_five_digit_congruent_eight_mod_17_l170_170048

theorem least_five_digit_congruent_eight_mod_17 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 8 ∧ n = 10009 :=
by
  use 10009
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  sorry

end least_five_digit_congruent_eight_mod_17_l170_170048


namespace Douglas_weight_correct_l170_170499

def Anne_weight : ℕ := 67
def weight_diff : ℕ := 15
def Douglas_weight : ℕ := 52

theorem Douglas_weight_correct : Douglas_weight = Anne_weight - weight_diff := by
  sorry

end Douglas_weight_correct_l170_170499


namespace probability_same_color_l170_170260

-- Definitions with conditions
def totalPlates := 11
def redPlates := 6
def bluePlates := 5

-- Calculate combinations
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main statement
theorem probability_same_color (totalPlates redPlates bluePlates : ℕ) (h1 : totalPlates = 11) 
(h2 : redPlates = 6) (h3 : bluePlates = 5) : 
  (2 / 11 : ℚ) = ((choose redPlates 3 + choose bluePlates 3) : ℚ) / (choose totalPlates 3) :=
by
  -- Proof steps will be inserted here
  sorry

end probability_same_color_l170_170260


namespace number_of_children_l170_170730

theorem number_of_children 
  (A C : ℕ) 
  (h1 : A + C = 201) 
  (h2 : 8 * A + 4 * C = 964) : 
  C = 161 := 
sorry

end number_of_children_l170_170730


namespace sufficient_not_necessary_condition_l170_170529

theorem sufficient_not_necessary_condition (x : ℝ) : (|x - 1/2| < 1/2) → (x^3 < 1) ∧ ¬(x^3 < 1) → (|x - 1/2| < 1/2) :=
sorry

end sufficient_not_necessary_condition_l170_170529


namespace part1_a_range_part2_x_range_l170_170671
open Real

-- Definitions based on given conditions
def quad_func (a b x : ℝ) : ℝ :=
  a * x^2 + b * x + 2

def y_at_x1 (a b : ℝ) : Prop :=
  quad_func a b 1 = 1

def pos_on_interval (a b l r : ℝ) (x : ℝ) : Prop :=
  l < x ∧ x < r → 0 < quad_func a b x

-- Part 1 proof statement in Lean 4
theorem part1_a_range (a b : ℝ) (h1 : y_at_x1 a b) (h2 : ∀ x : ℝ, pos_on_interval a b 2 5 x) :
  a > 3 - 2 * sqrt 2 :=
sorry

-- Part 2 proof statement in Lean 4
theorem part2_x_range (a b : ℝ) (h1 : y_at_x1 a b) (h2 : ∀ a' : ℝ, -2 ≤ a' ∧ a' ≤ -1 → 0 < quad_func a' b x) :
  (1 - sqrt 17) / 4 < x ∧ x < (1 + sqrt 17) / 4 :=
sorry

end part1_a_range_part2_x_range_l170_170671


namespace andrea_living_room_area_l170_170625

/-- Given that 60% of Andrea's living room floor is covered by a carpet 
     which has dimensions 4 feet by 9 feet, prove that the area of 
     Andrea's living room floor is 60 square feet. -/
theorem andrea_living_room_area :
  ∃ A, (0.60 * A = 4 * 9) ∧ A = 60 :=
by
  sorry

end andrea_living_room_area_l170_170625


namespace find_price_of_pastry_l170_170628

-- Define the known values and conditions
variable (P : ℕ)  -- Price of a pastry
variable (usual_pastries : ℕ := 20)
variable (usual_bread : ℕ := 10)
variable (bread_price : ℕ := 4)
variable (today_pastries : ℕ := 14)
variable (today_bread : ℕ := 25)
variable (price_difference : ℕ := 48)

-- Define the usual daily total and today's total
def usual_total := usual_pastries * P + usual_bread * bread_price
def today_total := today_pastries * P + today_bread * bread_price

-- Define the problem statement
theorem find_price_of_pastry (h: usual_total - today_total = price_difference) : P = 18 :=
  by sorry

end find_price_of_pastry_l170_170628


namespace factor_expression_l170_170229

variable (x y : ℝ)

theorem factor_expression :
(3*x^3 + 28*(x^2)*y + 4*x) - (-4*x^3 + 5*(x^2)*y - 4*x) = x*(x + 8)*(7*x + 1) := sorry

end factor_expression_l170_170229


namespace probability_of_one_or_two_in_pascal_l170_170792

def pascal_triangle_element_probability : ℚ :=
  let total_elements := 210 -- sum of the elements in the first 20 rows
  let ones_count := 39      -- total count of 1s in the first 20 rows
  let twos_count := 36      -- total count of 2s in the first 20 rows
  let favorable_elements := ones_count + twos_count
  favorable_elements / total_elements

theorem probability_of_one_or_two_in_pascal (n : ℕ) (h : n = 20) :
  pascal_triangle_element_probability = 5 / 14 := by
  rw [h]
  dsimp [pascal_triangle_element_probability]
  sorry

end probability_of_one_or_two_in_pascal_l170_170792


namespace arithmetic_sequence_a9_l170_170145

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a9
  (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a 1)
  (h2 : a 2 + a 4 = 2)
  (h5 : a 5 = 3) :
  a 9 = 7 :=
by
  sorry

end arithmetic_sequence_a9_l170_170145


namespace min_value_of_sin_x_plus_sin_z_thm_l170_170114

noncomputable def min_value_of_sin_x_plus_sin_z 
    (x y z : ℝ) 
    (h1 : sqrt 3 * Real.cos x = Real.cot y) 
    (h2 : 2 * Real.cos y = Real.tan z) 
    (h3 : Real.cos z = 2 * Real.cot x) : ℝ :=
  min (sin x + sin z)

theorem min_value_of_sin_x_plus_sin_z_thm 
    (x y z : ℝ)
    (h1 : sqrt 3 * Real.cos x = Real.cot y)
    (h2 : 2 * Real.cos y = Real.tan z)
    (h3 : Real.cos z = 2 * Real.cot x) :
  min_value_of_sin_x_plus_sin_z x y z h1 h2 h3 = -7 * sqrt 2 / 6 :=
sorry

end min_value_of_sin_x_plus_sin_z_thm_l170_170114


namespace remaining_wallpaper_removal_time_l170_170515

theorem remaining_wallpaper_removal_time (dining_walls living_walls : ℕ) (time_per_wall: ℕ) (time_spent: ℕ) :
  dining_walls = 4 →
  living_walls = 4 →
  time_per_wall = 2 →
  time_spent = 2 →
  time_per_wall * dining_walls + time_per_wall * living_walls - time_spent = 14 :=
by
  intros hd hl ht hs
  rw [hd, hl, ht, hs]
  exact dec_trivial

end remaining_wallpaper_removal_time_l170_170515


namespace probability_closer_to_6_than_0_is_0_6_l170_170929

noncomputable def probability_closer_to_6_than_0 : ℝ :=
  let total_length := 7
  let segment_length_closer_to_6 := 4
  let probability := (segment_length_closer_to_6 : ℝ) / total_length
  probability

theorem probability_closer_to_6_than_0_is_0_6 :
  probability_closer_to_6_than_0 = 0.6 := by
  sorry

end probability_closer_to_6_than_0_is_0_6_l170_170929


namespace corn_height_after_three_weeks_l170_170501

theorem corn_height_after_three_weeks 
  (week1_growth : ℕ) (week2_growth : ℕ) (week3_growth : ℕ) 
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  week1_growth + week2_growth + week3_growth = 22 :=
by {
  sorry
}

end corn_height_after_three_weeks_l170_170501


namespace family_vacation_days_l170_170781

theorem family_vacation_days
  (rained_days : ℕ)
  (total_days : ℕ)
  (clear_mornings : ℕ)
  (H1 : rained_days = 13)
  (H2 : total_days = 18)
  (H3 : clear_mornings = 11) :
  total_days = 18 :=
by
  -- proof to be filled in here
  sorry

end family_vacation_days_l170_170781


namespace school_C_paintings_l170_170163

theorem school_C_paintings
  (A B C : ℕ)
  (h1 : B + C = 41)
  (h2 : A + C = 38)
  (h3 : A + B = 43) : 
  C = 18 :=
by
  sorry

end school_C_paintings_l170_170163


namespace cost_of_fencing_l170_170911

-- Define the conditions
def width_garden : ℕ := 12
def length_playground : ℕ := 16
def width_playground : ℕ := 12
def price_per_meter : ℕ := 15
def area_playground : ℕ := length_playground * width_playground
def area_garden : ℕ := area_playground
def length_garden : ℕ := area_garden / width_garden
def perimeter_garden : ℕ := 2 * (length_garden + width_garden)
def cost_fencing : ℕ := perimeter_garden * price_per_meter

-- State the theorem
theorem cost_of_fencing : cost_fencing = 840 := by
  sorry

end cost_of_fencing_l170_170911


namespace proof_problem_l170_170978

variable (γ θ α : ℝ)
variable (x y : ℝ)

def condition1 := x = γ * Real.sin ((θ - α) / 2)
def condition2 := y = γ * Real.sin ((θ + α) / 2)

theorem proof_problem
  (h1 : condition1 γ θ α x)
  (h2 : condition2 γ θ α y)
  : x^2 - 2*x*y*Real.cos α + y^2 = γ^2 * (Real.sin α)^2 :=
by
  sorry

end proof_problem_l170_170978


namespace complex_fraction_simplification_l170_170431

theorem complex_fraction_simplification (i : ℂ) (hi : i^2 = -1) : 
  ((2 - i) / (1 + 4 * i)) = (-2 / 17 - (9 / 17) * i) :=
  sorry

end complex_fraction_simplification_l170_170431


namespace additional_distance_if_faster_speed_l170_170844

-- Conditions
def speed_slow := 10 -- km/hr
def speed_fast := 15 -- km/hr
def actual_distance := 30 -- km

-- Question and answer
theorem additional_distance_if_faster_speed : (speed_fast * (actual_distance / speed_slow) - actual_distance) = 15 := by
  sorry

end additional_distance_if_faster_speed_l170_170844


namespace sum_of_remainders_correct_l170_170930

def sum_of_remainders : ℕ :=
  let remainders := [43210 % 37, 54321 % 37, 65432 % 37, 76543 % 37, 87654 % 37, 98765 % 37]
  remainders.sum

theorem sum_of_remainders_correct : sum_of_remainders = 36 :=
by sorry

end sum_of_remainders_correct_l170_170930


namespace find_t_l170_170351

variables (c o u n t s : ℕ)

theorem find_t (h1 : c + o = u) 
               (h2 : u + n = t)
               (h3 : t + c = s)
               (h4 : o + n + s = 18)
               (hz : c > 0) (ho : o > 0) (hu : u > 0) (hn : n > 0) (ht : t > 0) (hs : s > 0) : 
               t = 9 := 
by
  sorry

end find_t_l170_170351


namespace coefficient_x3_l170_170556

-- Define the binomial coefficient
def binomial_coefficient (n k : Nat) : Nat :=
  Nat.choose n k

noncomputable def coefficient_x3_term : Nat :=
  binomial_coefficient 25 3

theorem coefficient_x3 : coefficient_x3_term = 2300 :=
by
  unfold coefficient_x3_term
  unfold binomial_coefficient
  -- Here, one would normally provide the proof steps, but we're adding sorry to skip
  sorry

end coefficient_x3_l170_170556


namespace tangent_line_equation_l170_170987

theorem tangent_line_equation (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  ∃ (m b : ℝ), y = m * x + b ∧ y = 4 * x - 2 :=
by
  sorry

end tangent_line_equation_l170_170987


namespace find_triangle_base_l170_170898

theorem find_triangle_base (left_side : ℝ) (right_side : ℝ) (base : ℝ) 
  (h_left : left_side = 12) 
  (h_right : right_side = left_side + 2)
  (h_sum : left_side + right_side + base = 50) :
  base = 24 := 
sorry

end find_triangle_base_l170_170898


namespace sine_cos_suffices_sine_cos_necessary_l170_170391

theorem sine_cos_suffices
  (a b c : ℝ)
  (h : ∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) :
  c > Real.sqrt (a^2 + b^2) :=
sorry

theorem sine_cos_necessary
  (a b c : ℝ)
  (h : c > Real.sqrt (a^2 + b^2)) :
  ∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0 :=
sorry

end sine_cos_suffices_sine_cos_necessary_l170_170391


namespace retailer_profit_percentage_l170_170949

theorem retailer_profit_percentage (items_sold : ℕ) (profit_per_item : ℝ) (discount_rate : ℝ)
  (discounted_items_needed : ℝ) (total_profit : ℝ) (item_cost : ℝ) :
  items_sold = 100 → 
  profit_per_item = 30 →
  discount_rate = 0.05 →
  discounted_items_needed = 156.86274509803923 →
  total_profit = 3000 →
  (discounted_items_needed * ((item_cost + profit_per_item) * (1 - discount_rate) - item_cost) = total_profit) →
  ((profit_per_item / item_cost) * 100 = 16) :=
by {
  sorry 
}

end retailer_profit_percentage_l170_170949


namespace correct_proposition_l170_170339

theorem correct_proposition (a b : ℝ) (h : |a| < b) : a^2 < b^2 :=
sorry

end correct_proposition_l170_170339


namespace least_value_of_a_plus_b_l170_170711

def a_and_b (a b : ℕ) : Prop :=
  (Nat.gcd (a + b) 330 = 1) ∧ 
  (a^a % b^b = 0) ∧ 
  (¬ (a % b = 0))

theorem least_value_of_a_plus_b :
  ∃ (a b : ℕ), a_and_b a b ∧ a + b = 105 :=
sorry

end least_value_of_a_plus_b_l170_170711


namespace profit_maximization_problem_l170_170630

-- Step 1: Define the data points and linear function
def data_points : List (ℝ × ℝ) := [(65, 70), (70, 60), (75, 50), (80, 40)]

-- Step 2: Define the linear function between y and x
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Step 3: Define cost and profit function
def cost_per_kg : ℝ := 60
def profit_function (y x : ℝ) : ℝ := y * (x - cost_per_kg)

-- Step 4: The main problem statement
theorem profit_maximization_problem :
  ∃ (k b : ℝ), 
  (∀ (x₁ x₂ : ℝ), (x₁, y₁) ∈ data_points ∧ (x₂, y₂) ∈ data_points → linear_function k b x₁ = y₁ ∧ linear_function k b x₂ = y₂) ∧
  ∃ (x : ℝ), profit_function (linear_function k b x) x = 600 ∧
  ∀ x : ℝ, -2 * x^2 + 320 * x - 12000 ≤ -2 * 80^2 + 320 * 80 - 12000
  :=
sorry

end profit_maximization_problem_l170_170630


namespace reduced_price_of_oil_l170_170932

/-- 
Given:
1. The original price per kg of oil is P.
2. The reduced price per kg of oil is 0.65P.
3. Rs. 800 can buy 5 kgs more oil at the reduced price than at the original price.
4. The equation 5P - 5 * 0.65P = 800 holds true.

Prove that the reduced price per kg of oil is Rs. 297.14.
-/
theorem reduced_price_of_oil (P : ℝ) (h1 : 5 * P - 5 * 0.65 * P = 800) : 
        0.65 * P = 297.14 := 
    sorry

end reduced_price_of_oil_l170_170932


namespace joel_age_when_dad_twice_l170_170867

theorem joel_age_when_dad_twice (x : ℕ) (h₁ : x = 22) : 
  let Joel_age := 5 + x 
  in Joel_age = 27 :=
by
  unfold Joel_age
  rw [h₁]
  norm_num

end joel_age_when_dad_twice_l170_170867


namespace calculate_area_bounded_figure_l170_170727

noncomputable def area_of_bounded_figure (R : ℝ) : ℝ :=
  (R^2 / 9) * (3 * Real.sqrt 3 - 2 * Real.pi)

theorem calculate_area_bounded_figure (R : ℝ) :
  ∀ r, r = (R / 3) → area_of_bounded_figure R = (R^2 / 9) * (3 * Real.sqrt 3 - 2 * Real.pi) :=
by
  intros r hr
  subst hr
  exact rfl

end calculate_area_bounded_figure_l170_170727


namespace find_f_neg_2017_l170_170837

-- Define f as given in the problem
def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 2

-- State the given problem condition
def condition (a b : ℝ) : Prop :=
  f a b 2017 = 10

-- The main problem statement proving the solution
theorem find_f_neg_2017 (a b : ℝ) (h : condition a b) : f a b (-2017) = -14 :=
by
  -- We state this theorem and provide a sorry to skip the proof
  sorry

end find_f_neg_2017_l170_170837


namespace perpendicular_condition_sufficient_not_necessary_l170_170946

theorem perpendicular_condition_sufficient_not_necessary (m : ℝ) :
  (∀ x y : ℝ, m * x + (2 * m - 1) * y + 1 = 0) →
  (∀ x y : ℝ, 3 * x + m * y + 3 = 0) →
  (∀ a b : ℝ, m = -1 → (∃ c d : ℝ, 3 / a = 1 / b)) →
  (m = -1 → (m = -1 → (3 / (-m / (2 * m - 1)) * m) / 2 - (3 / m) = -1)) :=
by sorry

end perpendicular_condition_sufficient_not_necessary_l170_170946


namespace circles_point_distance_l170_170330

noncomputable section

-- Define the data for the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def CircleA (R : ℝ) (K : ℝ × ℝ) : Circle := 
  { center := K, radius := R }

def CircleB (R : ℝ) (K : ℝ × ℝ) : Circle := 
  { center := (K.1 + 2 * R, K.2), radius := R }

-- Define the condition that two circles touch each other at point K
def circles_touch (C1 C2 : Circle) (K : ℝ × ℝ) : Prop :=
  dist C1.center K = C1.radius ∧ dist C2.center K = C2.radius ∧ dist C1.center C2.center = C1.radius + C2.radius

-- Define the angle condition ∠AKB = 90°
def angle_AKB_is_right (A K B : ℝ × ℝ) : Prop :=
  -- Using the fact that a dot product being zero implies orthogonality
  let vec1 := (A.1 - K.1, A.2 - K.2)
  let vec2 := (B.1 - K.1, B.2 - K.2)
  vec1.1 * vec2.1 + vec1.2 * vec2.2 = 0

-- Define the points A and B being on their respective circles
def on_circle (A : ℝ × ℝ) (C : Circle) : Prop :=
  dist A C.center = C.radius

-- Define the theorem
theorem circles_point_distance 
  (R : ℝ) (K A B : ℝ × ℝ) 
  (C1 := CircleA R K) 
  (C2 := CircleB R K) 
  (h1 : circles_touch C1 C2 K) 
  (h2 : on_circle A C1) 
  (h3 : on_circle B C2) 
  (h4 : angle_AKB_is_right A K B) : 
  dist A B = 2 * R := 
sorry

end circles_point_distance_l170_170330


namespace least_five_digit_congruent_to_8_mod_17_l170_170051

theorem least_five_digit_congruent_to_8_mod_17 : 
  ∃ n : ℕ, n = 10004 ∧ n % 17 = 8 ∧ 10000 ≤ n ∧ n < 100000 ∧ (∀ m, 10000 ≤ m ∧ m < 10004 → m % 17 ≠ 8) :=
sorry

end least_five_digit_congruent_to_8_mod_17_l170_170051


namespace rowing_upstream_speed_l170_170483

def speed_in_still_water : ℝ := 31
def speed_downstream : ℝ := 37

def speed_stream : ℝ := speed_downstream - speed_in_still_water

def speed_upstream : ℝ := speed_in_still_water - speed_stream

theorem rowing_upstream_speed :
  speed_upstream = 25 := by
  sorry

end rowing_upstream_speed_l170_170483


namespace teaching_arrangements_l170_170716

-- Define the conditions
structure Conditions :=
  (teach_A : ℕ)
  (teach_B : ℕ)
  (teach_C : ℕ)
  (teach_D : ℕ)
  (max_teach_AB : ∀ t, t = teach_A ∨ t = teach_B → t ≤ 2)
  (max_teach_CD : ∀ t, t = teach_C ∨ t = teach_D → t ≤ 1)
  (total_periods : ℕ)
  (teachers_per_period : ℕ)

-- Constants and assumptions
def problem_conditions : Conditions := {
  teach_A := 2,
  teach_B := 2,
  teach_C := 1,
  teach_D := 1,
  max_teach_AB := by sorry,
  max_teach_CD := by sorry,
  total_periods := 2,
  teachers_per_period := 2
}

-- Define the proof goal
theorem teaching_arrangements (c : Conditions) :
  c = problem_conditions → ∃ arrangements, arrangements = 19 :=
by
  sorry

end teaching_arrangements_l170_170716


namespace meaningful_sqrt_l170_170138

theorem meaningful_sqrt (a : ℝ) (h : a - 4 ≥ 0) : a ≥ 4 :=
sorry

end meaningful_sqrt_l170_170138


namespace kaleb_initial_books_l170_170700

def initial_books (sold_books bought_books final_books : ℕ) : ℕ := 
  sold_books - bought_books + final_books

theorem kaleb_initial_books :
  initial_books 17 (-7) 24 = 34 := 
by 
  -- use the definition of initial_books
  sorry

end kaleb_initial_books_l170_170700


namespace distance_between_sets_is_zero_l170_170813

noncomputable def A (x : ℝ) : ℝ := 2 * x - 1
noncomputable def B (x : ℝ) : ℝ := x^2 + 1

theorem distance_between_sets_is_zero : 
  ∃ (a b : ℝ), (∃ x₀ : ℝ, a = A x₀) ∧ (∃ y₀ : ℝ, b = B y₀) ∧ abs (a - b) = 0 := 
sorry

end distance_between_sets_is_zero_l170_170813


namespace parabola_directrix_l170_170964

theorem parabola_directrix (x y : ℝ) : 
  (∀ x: ℝ, y = -4 * x ^ 2 + 4) → (y = 65 / 16) := 
by 
  sorry

end parabola_directrix_l170_170964


namespace third_group_members_l170_170478

-- Define the total number of members in the choir
def total_members : ℕ := 70

-- Define the number of members in the first group
def first_group_members : ℕ := 25

-- Define the number of members in the second group
def second_group_members : ℕ := 30

-- Prove that the number of members in the third group is 15
theorem third_group_members : total_members - first_group_members - second_group_members = 15 := 
by 
  sorry

end third_group_members_l170_170478


namespace storybooks_sciencebooks_correct_l170_170637

-- Given conditions
def total_books : ℕ := 144
def ratio_storybooks_sciencebooks := (7, 5)
def fraction_storybooks := 7 / (7 + 5)
def fraction_sciencebooks := 5 / (7 + 5)

-- Prove the number of storybooks and science books
def number_of_storybooks : ℕ := 84
def number_of_sciencebooks : ℕ := 60

theorem storybooks_sciencebooks_correct :
  (fraction_storybooks * total_books = number_of_storybooks) ∧
  (fraction_sciencebooks * total_books = number_of_sciencebooks) :=
by
  sorry

end storybooks_sciencebooks_correct_l170_170637


namespace product_of_divisors_of_18_l170_170453

theorem product_of_divisors_of_18 : 
  let divisors := [1, 2, 3, 6, 9, 18] in divisors.prod = 5832 := 
by
  let divisors := [1, 2, 3, 6, 9, 18]
  have h : divisors.prod = 18^3 := sorry
  have h_calc : 18^3 = 5832 := by norm_num
  exact Eq.trans h h_calc

end product_of_divisors_of_18_l170_170453


namespace smallest_n_l170_170201

theorem smallest_n (n : ℕ) (h1 : ∃ k1, 5 * n = k1^2) (h2 : ∃ k2, 7 * n = k2^3) : n = 245 :=
sorry

end smallest_n_l170_170201


namespace pair_exists_l170_170562

theorem pair_exists (x : Fin 670 → ℝ) (h_distinct : Function.Injective x) (h_bounds : ∀ i, 0 < x i ∧ x i < 1) :
  ∃ (i j : Fin 670), 0 < x i * x j * (x j - x i) ∧ x i * x j * (x j - x i) < 1 / 2007 := 
by
  sorry

end pair_exists_l170_170562


namespace probability_genuine_given_equal_weight_l170_170362

noncomputable def num_genuine : ℕ := 16
noncomputable def num_counterfeit : ℕ := 4
noncomputable def total_coins : ℕ := num_genuine + num_counterfeit
noncomputable def event_A : Event := {ω | all_four_selected_are_genuine ω}
noncomputable def event_B : Event := {ω | combined_weight_pairs_equal ω}

axiom coin_events_proba (A B : Event) : 
  (P(A ∩ B) = 1092 / 2907) ∧ (P(B) = 2907 / 5814)

theorem probability_genuine_given_equal_weight : 
  @Probability.event (coin ω) event_A ∩ event_B * (coin ω) event_B :=
begin
  rw [conditional_probability_def],
  rw [coin_events_proba],
  simp,
  norm_num [1092 / 2907, 2907 / 5814];
  sorry,
end

end probability_genuine_given_equal_weight_l170_170362


namespace system_of_equations_solution_l170_170184

theorem system_of_equations_solution (x y z : ℝ) (h1 : x + y = 1) (h2 : x + z = 0) (h3 : y + z = -1) : 
    x = 1 ∧ y = 0 ∧ z = -1 := 
by 
  sorry

end system_of_equations_solution_l170_170184


namespace parabola_vertex_l170_170816

theorem parabola_vertex (x y : ℝ) : 
  y^2 + 10 * y + 3 * x + 9 = 0 → 
  (∃ v_x v_y, v_x = 16/3 ∧ v_y = -5 ∧ ∀ (y' : ℝ), (x, y) = (v_x, v_y) ↔ (x, y) = (-1 / 3 * ((y' + 5)^2 - 16), y')) :=
by
  sorry

end parabola_vertex_l170_170816


namespace q_can_complete_work_in_25_days_l170_170062

-- Define work rates for p, q, and r
variables (W_p W_q W_r : ℝ)

-- Define total work
variable (W : ℝ)

-- Prove that q can complete the work in 25 days under given conditions
theorem q_can_complete_work_in_25_days
  (h1 : W_p = W_q + W_r)
  (h2 : W_p + W_q = W / 10)
  (h3 : W_r = W / 50) :
  W_q = W / 25 :=
by
  -- Given: W_p = W_q + W_r
  -- Given: W_p + W_q = W / 10
  -- Given: W_r = W / 50
  -- We need to prove: W_q = W / 25
  sorry

end q_can_complete_work_in_25_days_l170_170062


namespace rate_percent_simple_interest_l170_170208

theorem rate_percent_simple_interest (P SI T : ℝ) (hP : P = 720) (hSI : SI = 180) (hT : T = 4) :
  (SI = P * (R / 100) * T) → R = 6.25 :=
by
  sorry

end rate_percent_simple_interest_l170_170208


namespace kittens_weight_problem_l170_170599

theorem kittens_weight_problem
  (w_lightest : ℕ)
  (w_heaviest : ℕ)
  (w_total : ℕ)
  (total_lightest : w_lightest = 80)
  (total_heaviest : w_heaviest = 200)
  (total_weight : w_total = 500) :
  ∃ (n : ℕ), n = 11 :=
by sorry

end kittens_weight_problem_l170_170599


namespace black_squares_count_l170_170161

def checkerboard_size : Nat := 32
def total_squares : Nat := checkerboard_size * checkerboard_size
def black_squares (n : Nat) : Nat := n / 2

theorem black_squares_count : black_squares total_squares = 512 := by
  let n := total_squares
  show black_squares n = 512
  sorry

end black_squares_count_l170_170161


namespace point_B_in_fourth_quadrant_l170_170401

theorem point_B_in_fourth_quadrant (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : (b > 0 ∧ a < 0) :=
by {
    sorry
}

end point_B_in_fourth_quadrant_l170_170401


namespace problem_1_a_problem_1_b_problem_2_l170_170541

def set_A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def set_B : Set ℝ := {x | 2 < x ∧ x < 9}
def complement_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 9}
def set_union (s₁ s₂ : Set ℝ) := {x | x ∈ s₁ ∨ x ∈ s₂}
def set_inter (s₁ s₂ : Set ℝ) := {x | x ∈ s₁ ∧ x ∈ s₂}

theorem problem_1_a :
  set_inter set_A set_B = {x : ℝ | 3 ≤ x ∧ x < 6} :=
sorry

theorem problem_1_b :
  set_union complement_B set_A = {x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} :=
sorry

def set_C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem problem_2 (a : ℝ) :
  (set_C a ⊆ set_B) → (2 ≤ a ∧ a ≤ 8) :=
sorry

end problem_1_a_problem_1_b_problem_2_l170_170541


namespace corn_height_growth_l170_170506

theorem corn_height_growth (init_height week1_growth week2_growth week3_growth : ℕ)
  (h0 : init_height = 0)
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  init_height + week1_growth + week2_growth + week3_growth = 22 :=
by sorry

end corn_height_growth_l170_170506


namespace range_of_m_l170_170891

theorem range_of_m (m : ℝ) :
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ x - (m^2 - 2 * m + 4) * y + 6 > 0) →
  -1 < m ∧ m < 3 :=
by
  intros h
  rcases h with ⟨x, y, hx, hy, hineq⟩
  rw [hx, hy] at hineq
  sorry

end range_of_m_l170_170891


namespace weight_of_new_person_l170_170439

theorem weight_of_new_person 
  (avg_increase : Real)
  (num_persons : Nat)
  (old_weight : Real)
  (new_avg_increase : avg_increase = 2.2)
  (number_of_persons : num_persons = 15)
  (weight_of_old_person : old_weight = 75)
  : (new_weight : Real) = old_weight + avg_increase * num_persons := 
  by sorry

end weight_of_new_person_l170_170439


namespace minimize_sum_m_n_l170_170331

-- Definitions of the given conditions
def last_three_digits_equal (a b : ℕ) : Prop :=
  (a % 1000) = (b % 1000)

-- The main statement to prove
theorem minimize_sum_m_n (m n : ℕ) (h1 : n > m) (h2 : 1 ≤ m) 
  (h3 : last_three_digits_equal (1978^n) (1978^m)) : m + n = 106 :=
sorry

end minimize_sum_m_n_l170_170331


namespace inverse_geometric_sequence_l170_170895

-- Define that a, b, c form a geometric sequence
def geometric_sequence (a b c : ℝ) := b^2 = a * c

-- Define the theorem: if b^2 = a * c, then a, b, c form a geometric sequence
theorem inverse_geometric_sequence (a b c : ℝ) (h : b^2 = a * c) : geometric_sequence a b c :=
by
  sorry

end inverse_geometric_sequence_l170_170895


namespace percentage_decrease_is_17_point_14_l170_170182

-- Define the conditions given in the problem
variable (S : ℝ) -- original salary
variable (D : ℝ) -- percentage decrease

-- Given conditions
def given_conditions : Prop :=
  1.40 * S - (D / 100) * 1.40 * S = 1.16 * S

-- The required proof problem, where we assert D = 17.14
theorem percentage_decrease_is_17_point_14 (S : ℝ) (h : given_conditions S D) : D = 17.14 := 
  sorry

end percentage_decrease_is_17_point_14_l170_170182


namespace infinitely_many_primes_satisfying_condition_l170_170878

theorem infinitely_many_primes_satisfying_condition :
  ∀ k : Nat, ∃ p : Nat, Nat.Prime p ∧ ∃ n : Nat, n > 0 ∧ p ∣ (2014^(2^n) + 2014) := 
sorry

end infinitely_many_primes_satisfying_condition_l170_170878


namespace pasta_preferences_l170_170209

theorem pasta_preferences :
  ∀ (students_total students_spaghetti students_tortellini students_penne : ℕ),
  students_total = 800 →
  students_spaghetti = 260 →
  students_tortellini = 160 →
  (students_penne : ℚ) / students_tortellini = 3 / 4 →
  students_spaghetti - students_penne = 140 :=
begin
  intros students_total students_spaghetti students_tortellini students_penne,
  intros h_total h_spaghetti h_tortellini h_ratio,
  sorry
end

end pasta_preferences_l170_170209


namespace actual_cost_of_article_l170_170644

theorem actual_cost_of_article (x : ℝ) (h : 0.80 * x = 620) : x = 775 :=
sorry

end actual_cost_of_article_l170_170644


namespace equilateral_triangle_not_centrally_symmetric_l170_170341

-- Definitions for the shapes
def is_centrally_symmetric (shape : Type) : Prop := sorry
def Parallelogram : Type := sorry
def LineSegment : Type := sorry
def EquilateralTriangle : Type := sorry
def Rhombus : Type := sorry

-- Main theorem statement
theorem equilateral_triangle_not_centrally_symmetric :
  ¬ is_centrally_symmetric EquilateralTriangle ∧
  is_centrally_symmetric Parallelogram ∧
  is_centrally_symmetric LineSegment ∧
  is_centrally_symmetric Rhombus :=
sorry

end equilateral_triangle_not_centrally_symmetric_l170_170341


namespace sphere_surface_area_l170_170489

theorem sphere_surface_area (a b c : ℝ) (r : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) (h4 : r = (Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2)) / 2):
    4 * Real.pi * r ^ 2 = 50 * Real.pi :=
by
  sorry

end sphere_surface_area_l170_170489


namespace range_of_a_l170_170696

noncomputable def operation (x y : ℝ) := x * (1 - y)

theorem range_of_a
  (a : ℝ)
  (hx : ∀ x : ℝ, operation (x - a) (x + a) < 1) :
  -1/2 < a ∧ a < 3/2 := by
  sorry

end range_of_a_l170_170696


namespace problem1_problem2_l170_170171

-- Problem 1

def a : ℚ := -1 / 2
def b : ℚ := -1

theorem problem1 :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) + a * b^2 = -3 / 4 :=
by
  sorry

-- Problem 2

def x : ℚ := 1 / 2
def y : ℚ := -2 / 3
axiom condition2 : abs (2 * x - 1) + (3 * y + 2)^2 = 0

theorem problem2 :
  5 * x^2 - (2 * x * y - 3 * (x * y / 3 + 2) + 5 * x^2) = 19 / 3 :=
by
  have h : abs (2 * x - 1) + (3 * y + 2)^2 = 0 := condition2
  sorry

end problem1_problem2_l170_170171


namespace abs_ineq_range_m_l170_170665

theorem abs_ineq_range_m :
  ∀ m : ℝ, (∀ x : ℝ, |x - 1| + |x + 2| ≥ m) ↔ m ≤ 3 :=
by
  sorry

end abs_ineq_range_m_l170_170665


namespace plus_signs_count_l170_170020

theorem plus_signs_count (n p m : ℕ) (h_total : n = 23) 
  (h_plus_min : ∀ s, s.card = 10 → ∃ t ∈ s, is_plus t)
  (h_minus_min : ∀ s, s.card = 15 → ∃ t ∈ s, is_minus t)
  : p = 14 :=
sorry

end plus_signs_count_l170_170020


namespace value_of_y_l170_170651

theorem value_of_y (x y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) (h2 : y = 3 * x) : y = 9 / 2 :=
sorry

end value_of_y_l170_170651


namespace dino_dolls_count_l170_170934

theorem dino_dolls_count (T : ℝ) (H : 0.7 * T = 140) : T = 200 :=
sorry

end dino_dolls_count_l170_170934


namespace derivative_of_volume_is_surface_area_l170_170071

noncomputable def V_sphere (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

theorem derivative_of_volume_is_surface_area (R : ℝ) (h : 0 < R) : 
  (deriv V_sphere R) = 4 * Real.pi * R^2 :=
by sorry

end derivative_of_volume_is_surface_area_l170_170071


namespace percent_receiving_speeding_tickets_l170_170159

theorem percent_receiving_speeding_tickets
  (total_motorists : ℕ)
  (percent_exceeding_limit percent_exceeding_limit_without_ticket : ℚ)
  (h_exceeding_limit : percent_exceeding_limit = 0.5)
  (h_exceeding_limit_without_ticket : percent_exceeding_limit_without_ticket = 0.2) :
  let exceeding_limit := percent_exceeding_limit * total_motorists
  let without_tickets := percent_exceeding_limit_without_ticket * exceeding_limit
  let with_tickets := exceeding_limit - without_tickets
  (with_tickets / total_motorists) * 100 = 40 :=
by
  sorry

end percent_receiving_speeding_tickets_l170_170159


namespace max_value_a7_a14_l170_170834

noncomputable def arithmetic_sequence_max_product (a_1 d : ℝ) : ℝ :=
  let a_7 := a_1 + 6 * d
  let a_14 := a_1 + 13 * d
  a_7 * a_14

theorem max_value_a7_a14 {a_1 d : ℝ} 
  (h : 10 = 2 * a_1 + 19 * d)
  (sum_first_20 : 100 = (10) * (a_1 + a_1 + 19 * d)) :
  arithmetic_sequence_max_product a_1 d = 25 :=
by
  sorry

end max_value_a7_a14_l170_170834


namespace total_apples_correct_l170_170789

variable (X : ℕ)

def Sarah_apples : ℕ := X

def Jackie_apples : ℕ := 2 * Sarah_apples X

def Adam_apples : ℕ := Jackie_apples X + 5

def total_apples : ℕ := Sarah_apples X + Jackie_apples X + Adam_apples X

theorem total_apples_correct : total_apples X = 5 * X + 5 := by
  sorry

end total_apples_correct_l170_170789


namespace line_length_limit_l170_170926

theorem line_length_limit : 
  ∑' n : ℕ, 1 / ((3 : ℝ) ^ n) + (1 / (3 ^ (n + 1))) * (Real.sqrt 3) = (3 + Real.sqrt 3) / 2 :=
sorry

end line_length_limit_l170_170926


namespace cubic_sum_divisible_by_9_l170_170195

theorem cubic_sum_divisible_by_9 (n : ℕ) (hn : n > 0) : 
  ∃ k, n^3 + (n+1)^3 + (n+2)^3 = 9*k := by
  sorry

end cubic_sum_divisible_by_9_l170_170195


namespace parabola_directrix_l170_170965

theorem parabola_directrix (x y : ℝ) : 
  (∀ x: ℝ, y = -4 * x ^ 2 + 4) → (y = 65 / 16) := 
by 
  sorry

end parabola_directrix_l170_170965


namespace Tim_has_16_pencils_l170_170194

variable (T_Sarah T_Tyrah T_Tim : Nat)

-- Conditions
def condition1 : Prop := T_Tyrah = 6 * T_Sarah
def condition2 : Prop := T_Tim = 8 * T_Sarah
def condition3 : Prop := T_Tyrah = 12

-- Theorem to prove
theorem Tim_has_16_pencils (h1 : condition1 T_Sarah T_Tyrah) (h2 : condition2 T_Sarah T_Tim) (h3 : condition3 T_Tyrah) : T_Tim = 16 :=
by
  sorry

end Tim_has_16_pencils_l170_170194


namespace shaded_fraction_is_half_l170_170404

-- Define the number of rows and columns in the grid
def num_rows : ℕ := 8
def num_columns : ℕ := 8

-- Define the number of shaded triangles based on the pattern explained
def shaded_rows : List ℕ := [1, 3, 5, 7]
def num_shaded_rows : ℕ := 4
def triangles_per_row : ℕ := num_columns
def num_shaded_triangles : ℕ := num_shaded_rows * triangles_per_row

-- Define the total number of triangles
def total_triangles : ℕ := num_rows * num_columns

-- Define the fraction of shaded triangles
def shaded_fraction : ℚ := num_shaded_triangles / total_triangles

-- Prove the shaded fraction is 1/2
theorem shaded_fraction_is_half : shaded_fraction = 1 / 2 :=
by
  -- Provide the calculations
  sorry

end shaded_fraction_is_half_l170_170404


namespace value_at_1971_l170_170575

def sequence_x (x : ℕ → ℝ) :=
  ∀ n > 1, 3 * x n - x (n - 1) = n

theorem value_at_1971 (x : ℕ → ℝ) (hx : sequence_x x) (h_initial : abs (x 1) < 1971) :
  abs (x 1971 - 985.25) < 0.000001 :=
by sorry

end value_at_1971_l170_170575


namespace scout_weekend_earnings_l170_170166

theorem scout_weekend_earnings
  (base_pay_per_hour : ℕ)
  (tip_per_delivery : ℕ)
  (hours_worked_saturday : ℕ)
  (deliveries_saturday : ℕ)
  (hours_worked_sunday : ℕ)
  (deliveries_sunday : ℕ)
  (total_earnings : ℕ)
  (h_base_pay : base_pay_per_hour = 10)
  (h_tip : tip_per_delivery = 5)
  (h_hours_sat : hours_worked_saturday = 4)
  (h_deliveries_sat : deliveries_saturday = 5)
  (h_hours_sun : hours_worked_sunday = 5)
  (h_deliveries_sun : deliveries_sunday = 8) :
  total_earnings = 155 :=
by
  sorry

end scout_weekend_earnings_l170_170166


namespace simplify_fraction_l170_170433

/-
  Given the conditions that \(i^2 = -1\),
  prove that \(\displaystyle\frac{2-i}{1+4i} = -\frac{2}{17} - \frac{9}{17}i\).
-/
theorem simplify_fraction : 
  let i : ℂ := ⟨0, 1⟩ in
  i^2 = -1 → (2 - i) / (1 + 4 * i) = - (2 / 17) - (9 / 17) * i :=
by
  intro h
  sorry

end simplify_fraction_l170_170433


namespace hyperbola_intersection_l170_170368

variable (a b c : ℝ) -- positive constants
variables (F1 F2 : (ℝ × ℝ)) -- foci of the hyperbola

-- The positive constants a and b
axiom a_pos : a > 0
axiom b_pos : b > 0

-- The foci are at (-c, 0) and (c, 0)
axiom F1_def : F1 = (-c, 0)
axiom F2_def : F2 = (c, 0)

-- We want to prove that the points (-c, b^2 / a) and (-c, -b^2 / a) are on the hyperbola
theorem hyperbola_intersection :
  (F1 = (-c, 0) ∧ F2 = (c, 0) ∧ a > 0 ∧ b > 0) →
  ∀ y : ℝ, ∃ y1 y2 : ℝ, (y1 = b^2 / a ∧ y2 = -b^2 / a ∧ 
  ( ( (-c)^2 / a^2) - (y1^2 / b^2) = 1 ∧  (-c)^2 / a^2 - y2^2 / b^2 = 1 ) ) :=
by
  intros h
  sorry

end hyperbola_intersection_l170_170368


namespace legs_per_bee_l170_170778

def number_of_bees : ℕ := 8
def total_legs : ℕ := 48

theorem legs_per_bee : (total_legs / number_of_bees) = 6 := by
  sorry

end legs_per_bee_l170_170778


namespace smallest_x_plus_y_l170_170661

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : x^2 - 29 * y^2 = 1) : x + y = 11621 := 
sorry

end smallest_x_plus_y_l170_170661


namespace hair_growth_l170_170409

theorem hair_growth (initial final : ℝ) (h_init : initial = 18) (h_final : final = 24) : final - initial = 6 :=
by
  sorry

end hair_growth_l170_170409


namespace problem_X_plus_Y_l170_170563

def num_five_digit_even_numbers : Nat := 45000
def num_five_digit_multiples_of_7 : Nat := 12857
def X := num_five_digit_even_numbers
def Y := num_five_digit_multiples_of_7

theorem problem_X_plus_Y : X + Y = 57857 :=
by
  sorry

end problem_X_plus_Y_l170_170563


namespace cost_of_50_tulips_l170_170099

theorem cost_of_50_tulips (c : ℕ → ℝ) :
  (∀ n : ℕ, n ≤ 40 → c n = n * (36 / 18)) ∧
  (∀ n : ℕ, n > 40 → c n = (40 * (36 / 18) + (n - 40) * (36 / 18)) * 0.9) ∧
  (c 18 = 36) →
  c 50 = 90 := sorry

end cost_of_50_tulips_l170_170099


namespace chess_tournament_games_l170_170186

-- Define the problem
def total_chess_games (n_players games_per_player : ℕ) : ℕ :=
  (n_players * games_per_player) / 2

-- Conditions: 
-- 1. There are 6 chess amateurs.
-- 2. Each amateur plays exactly 4 games.

theorem chess_tournament_games :
  total_chess_games 6 4 = 10 :=
  sorry

end chess_tournament_games_l170_170186


namespace inequality_solution_l170_170112

-- Define the inequality
def inequality (x : ℝ) : Prop := (3 * x - 1) / (2 - x) ≥ 1

-- Define the solution set
def solution_set (x : ℝ) : Prop := 3/4 ≤ x ∧ x ≤ 2

-- Theorem statement to prove the equivalence
theorem inequality_solution :
  ∀ x : ℝ, inequality x ↔ solution_set x := by
  sorry

end inequality_solution_l170_170112


namespace sum_of_roots_l170_170139

theorem sum_of_roots : (x₁ x₂ : ℝ) → (h : 2 * x₁^2 + 6 * x₁ - 1 = 0) → (h₂ : 2 * x₂^2 + 6 * x₂ - 1 = 0) → x₁ + x₂ = -3 :=
by 
  sorry

end sum_of_roots_l170_170139


namespace sum_geometric_series_l170_170437

theorem sum_geometric_series (x : ℂ) (h₀ : x ≠ 1) (h₁ : x^10 - 3*x + 2 = 0) : 
  x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := 
by 
  sorry

end sum_geometric_series_l170_170437


namespace kittens_weight_problem_l170_170598

theorem kittens_weight_problem
  (w_lightest : ℕ)
  (w_heaviest : ℕ)
  (w_total : ℕ)
  (total_lightest : w_lightest = 80)
  (total_heaviest : w_heaviest = 200)
  (total_weight : w_total = 500) :
  ∃ (n : ℕ), n = 11 :=
by sorry

end kittens_weight_problem_l170_170598


namespace gcd_204_85_l170_170109

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l170_170109


namespace range_of_2x_plus_y_range_of_c_l170_170674

open Real

def point_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2 * y

theorem range_of_2x_plus_y (x y : ℝ) (h : point_on_circle x y) : 
  1 - sqrt 2 ≤ 2 * x + y ∧ 2 * x + y ≤ 1 + sqrt 2 :=
sorry

theorem range_of_c (c : ℝ) : 
  (∀ x y : ℝ, point_on_circle x y → x + y + c > 0) → c ≥ -1 :=
sorry

end range_of_2x_plus_y_range_of_c_l170_170674


namespace warehouseGoodsDecreased_initialTonnage_totalLoadingFees_l170_170213

noncomputable def netChange (tonnages : List Int) : Int :=
  List.sum tonnages

noncomputable def initialGoods (finalGoods : Int) (change : Int) : Int :=
  finalGoods + change

noncomputable def totalFees (tonnages : List Int) (feePerTon : Int) : Int :=
  feePerTon * List.sum (tonnages.map (Int.natAbs))

theorem warehouseGoodsDecreased 
  (tonnages : List Int) (finalGoods : Int) (feePerTon : Int)
  (h1 : tonnages = [21, -32, -16, 35, -38, -20]) 
  (h2 : finalGoods = 580)
  (h3 : feePerTon = 4) : 
  netChange tonnages < 0 := by
  sorry

theorem initialTonnage 
  (tonnages : List Int) (finalGoods : Int) (change : Int)
  (h1 : tonnages = [21, -32, -16, 35, -38, -20])
  (h2 : finalGoods = 580)
  (h3 : change = netChange tonnages) : 
  initialGoods finalGoods change = 630 := by
  sorry

theorem totalLoadingFees 
  (tonnages : List Int) (feePerTon : Int)
  (h1 : tonnages = [21, -32, -16, 35, -38, -20])
  (h2 : feePerTon = 4) : 
  totalFees tonnages feePerTon = 648 := by
  sorry

end warehouseGoodsDecreased_initialTonnage_totalLoadingFees_l170_170213


namespace solve_y_pos_in_arithmetic_seq_l170_170579

-- Define the first term as 4
def first_term : ℕ := 4

-- Define the third term as 36
def third_term : ℕ := 36

-- Basing on the properties of an arithmetic sequence, 
-- we solve for the positive second term (y) such that its square equals to 20
theorem solve_y_pos_in_arithmetic_seq : ∃ y : ℝ, y > 0 ∧ y ^ 2 = 20 := by
  sorry

end solve_y_pos_in_arithmetic_seq_l170_170579


namespace cost_of_bought_movie_l170_170573

theorem cost_of_bought_movie 
  (ticket_cost : ℝ)
  (ticket_count : ℕ)
  (rental_cost : ℝ)
  (total_spent : ℝ)
  (bought_movie_cost : ℝ) :
  ticket_cost = 10.62 →
  ticket_count = 2 →
  rental_cost = 1.59 →
  total_spent = 36.78 →
  bought_movie_cost = total_spent - (ticket_cost * ticket_count + rental_cost) →
  bought_movie_cost = 13.95 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cost_of_bought_movie_l170_170573


namespace residue_mod_13_l170_170100

theorem residue_mod_13 :
  (250 ≡ 3 [MOD 13]) → 
  (20 ≡ 7 [MOD 13]) → 
  (5^2 ≡ 12 [MOD 13]) → 
  ((250 * 11 - 20 * 6 + 5^2) % 13 = 3) :=
by 
  sorry

end residue_mod_13_l170_170100


namespace diamond_expression_calculation_l170_170104

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_expression_calculation :
  (diamond (diamond 2 3) 5) - (diamond 2 (diamond 3 5)) = -37 / 210 :=
by
  sorry

end diamond_expression_calculation_l170_170104


namespace remainder_29_times_171997_pow_2000_mod_7_l170_170181

theorem remainder_29_times_171997_pow_2000_mod_7 :
  (29 * 171997^2000) % 7 = 4 :=
by
  sorry

end remainder_29_times_171997_pow_2000_mod_7_l170_170181


namespace triangle_base_l170_170900

theorem triangle_base (sum_of_sides : ℕ) (left_side : ℕ) (right_side : ℕ) (base : ℕ)
  (h1 : sum_of_sides = 50)
  (h2 : right_side = left_side + 2)
  (h3 : left_side = 12) :
  base = 24 :=
by
  sorry

end triangle_base_l170_170900


namespace trajectory_line_or_hyperbola_l170_170244

theorem trajectory_line_or_hyperbola
  (a b : ℝ)
  (ab_pos : a * b > 0)
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = a * x^2 + b) :
  (∃ s t : ℝ, f (s-t) * f (s+t) = (f s)^2) →
  (∃ s t : ℝ, ((t = 0) ∨ (a * t^2 - 2 * a * s^2 + 2 * b = 0))) → true := sorry

end trajectory_line_or_hyperbola_l170_170244


namespace triangle_XYZ_median_l170_170558

theorem triangle_XYZ_median (XYZ : Triangle) (YZ : ℝ) (XM : ℝ) (XY2_add_XZ2 : ℝ) 
  (hYZ : YZ = 12) (hXM : XM = 7) : XY2_add_XZ2 = 170 → N - n = 0 := by
  sorry

end triangle_XYZ_median_l170_170558


namespace marks_per_correct_answer_l170_170405

-- Definitions based on the conditions
def total_questions : ℕ := 60
def total_marks : ℕ := 160
def correct_questions : ℕ := 44
def wrong_mark_loss : ℕ := 1

-- The number of correct answers multiplies the marks per correct answer,
-- minus the loss from wrong answers, equals the total marks.
theorem marks_per_correct_answer (x : ℕ) :
  correct_questions * x - (total_questions - correct_questions) * wrong_mark_loss = total_marks → x = 4 := by
sorry

end marks_per_correct_answer_l170_170405


namespace fraction_increase_by_two_times_l170_170999

theorem fraction_increase_by_two_times (x y : ℝ) : 
  let new_val := ((2 * x) * (2 * y)) / (2 * x + 2 * y)
  let original_val := (x * y) / (x + y)
  new_val = 2 * original_val := 
by
  sorry

end fraction_increase_by_two_times_l170_170999


namespace plus_signs_count_l170_170029

theorem plus_signs_count (total_symbols : ℕ) (n : ℕ) (m : ℕ)
    (h1 : total_symbols = 23)
    (h2 : ∀ (s : Finset ℕ), s.card = 10 → (∃ x ∈ s, x = n))
    (h3 : ∀ (s : Finset ℕ), s.card = 15 → (∃ x ∈ s, x = m)) :
    n = 14 := by
  sorry

end plus_signs_count_l170_170029


namespace solution_exists_l170_170751

noncomputable def find_p_q : Prop :=
  ∃ p q : ℕ, (p^q - q^p = 1927) ∧ (p = 2611) ∧ (q = 11)

theorem solution_exists : find_p_q :=
sorry

end solution_exists_l170_170751


namespace sin_sum_arcsin_arctan_l170_170952

-- Definitions matching the conditions
def a := Real.arcsin (4 / 5)
def b := Real.arctan (1 / 2)

-- Theorem stating the question and expected answer
theorem sin_sum_arcsin_arctan : 
  Real.sin (a + b) = (11 * Real.sqrt 5) / 25 := 
by 
  sorry

end sin_sum_arcsin_arctan_l170_170952


namespace frustum_volume_correct_l170_170931

noncomputable def base_length := 20 -- cm
noncomputable def base_width := 10 -- cm
noncomputable def original_altitude := 12 -- cm
noncomputable def cut_height := 6 -- cm
noncomputable def base_area := base_length * base_width -- cm^2
noncomputable def original_volume := (1 / 3 : ℚ) * base_area * original_altitude -- cm^3
noncomputable def top_area := base_area / 4 -- cm^2
noncomputable def smaller_pyramid_volume := (1 / 3 : ℚ) * top_area * cut_height -- cm^3
noncomputable def frustum_volume := original_volume - smaller_pyramid_volume -- cm^3

theorem frustum_volume_correct :
  frustum_volume = 700 :=
by
  sorry

end frustum_volume_correct_l170_170931


namespace calculate_expression_l170_170507

theorem calculate_expression : |(-5 : ℤ)| + (1 / 3 : ℝ)⁻¹ - (Real.pi - 2) ^ 0 = 7 := by
  sorry

end calculate_expression_l170_170507


namespace calculate_fraction_l170_170998

theorem calculate_fraction (x y : ℚ) (h1 : x = 5 / 6) (h2 : y = 6 / 5) : (1 / 3) * x^8 * y^9 = 2 / 5 := by
  sorry

end calculate_fraction_l170_170998


namespace part_one_part_two_part_three_l170_170343

theorem part_one : 12 - (-11) - 1 = 22 := 
by
  sorry

theorem part_two : -(1 ^ 4) / ((-3) ^ 2) / (9 / 5) = -5 / 81 := 
by
  sorry

theorem part_three : -8 * (1/2 - 3/4 + 5/8) = -3 := 
by
  sorry

end part_one_part_two_part_three_l170_170343


namespace larger_cross_section_distance_l170_170756

theorem larger_cross_section_distance
  (h_area1 : ℝ)
  (h_area2 : ℝ)
  (dist_planes : ℝ)
  (h_area1_val : h_area1 = 256 * Real.sqrt 2)
  (h_area2_val : h_area2 = 576 * Real.sqrt 2)
  (dist_planes_val : dist_planes = 10) :
  ∃ h : ℝ, h = 30 :=
by
  sorry

end larger_cross_section_distance_l170_170756


namespace chocolates_left_l170_170880

-- Definitions based on the conditions
def initially_bought := 3
def gave_away := 2
def additionally_bought := 3

-- Theorem statement to prove
theorem chocolates_left : initially_bought - gave_away + additionally_bought = 4 := by
  -- Proof skipped
  sorry

end chocolates_left_l170_170880


namespace gcd_multiple_less_than_120_l170_170764

theorem gcd_multiple_less_than_120 (n : ℕ) (h1 : n < 120) (h2 : n % 10 = 0) (h3 : n % 15 = 0) : n ≤ 90 :=
by {
  sorry
}

end gcd_multiple_less_than_120_l170_170764


namespace correct_investment_allocation_l170_170190

noncomputable def investment_division (x : ℤ) : Prop :=
  let s := 2000
  let w := 500
  let rogers_investment := 2500
  let total_initial_capital := (5 / 2 : ℚ) * x
  let new_total_capital := total_initial_capital + rogers_investment
  let equal_share := new_total_capital / 3
  s + w = rogers_investment ∧ 
  (3 / 2 : ℚ) * x + s = equal_share ∧ 
  x + w = equal_share

theorem correct_investment_allocation (x : ℤ) (hx : 3 * x % 2 = 0) :
  x > 0 ∧ investment_division x :=
by
  sorry

end correct_investment_allocation_l170_170190


namespace red_ants_count_l170_170188

def total_ants : ℕ := 900
def black_ants : ℕ := 487
def red_ants (r : ℕ) : Prop := r + black_ants = total_ants

theorem red_ants_count : ∃ r : ℕ, red_ants r ∧ r = 413 := 
sorry

end red_ants_count_l170_170188


namespace opposite_of_neg2_l170_170744

theorem opposite_of_neg2 : ∃ y : ℤ, -2 + y = 0 ∧ y = 2 :=
by
  use 2
  simp
  sorry

end opposite_of_neg2_l170_170744


namespace max_chord_length_l170_170538

noncomputable def family_of_curves (θ x y : ℝ) := 
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

def line (x y : ℝ) := 2 * x = y

theorem max_chord_length :
  (∀ (θ : ℝ), ∀ (x y : ℝ), family_of_curves θ x y → line x y) → 
  ∃ (L : ℝ), L = 8 * Real.sqrt 5 :=
by
  sorry

end max_chord_length_l170_170538


namespace factory_correct_decision_prob_l170_170633

theorem factory_correct_decision_prob:
  let p := 0.8 in
  let q := 1 - p in
  let n := 3 in
  let correct_two_consultants := (Nat.choose n 2) * p^2 * q in
  let correct_three_consultants := (Nat.choose n 3) * p^3 in
  let probability_correct_decision := correct_two_consultants + correct_three_consultants in
  probability_correct_decision = 0.896 :=
by 
  sorry

end factory_correct_decision_prob_l170_170633


namespace smallest_k_for_repeating_representation_l170_170512

theorem smallest_k_for_repeating_representation:
  ∃ k : ℕ, (k > 0) ∧ (∀ m : ℕ, m > 0 → m < k → ¬(97*(5*m + 6) = 11*(m^2 - 1))) ∧ 97*(5*k + 6) = 11*(k^2 - 1) := by
  sorry

end smallest_k_for_repeating_representation_l170_170512


namespace commission_percentage_is_4_l170_170495

-- Define the given conditions
def commission := 12.50
def total_sales := 312.5

-- The problem is to prove the commission percentage
theorem commission_percentage_is_4 :
  (commission / total_sales) * 100 = 4 := by
  sorry

end commission_percentage_is_4_l170_170495


namespace seq_increasing_l170_170386

theorem seq_increasing (n : ℕ) (h : n > 0) : (↑n / (↑n + 2): ℝ) < (↑n + 1) / (↑n + 3) :=
by 
-- Converting ℕ to ℝ to make definitions correct
let an := (↑n / (↑n + 2): ℝ)
let an1 := (↑n + 1) / (↑n + 3)
-- Proof would go here
sorry

end seq_increasing_l170_170386


namespace second_number_is_180_l170_170446

theorem second_number_is_180 
  (x : ℝ) 
  (first : ℝ := 2 * x) 
  (third : ℝ := (1/3) * first)
  (h : first + x + third = 660) : 
  x = 180 :=
sorry

end second_number_is_180_l170_170446


namespace lcm_of_6_8_10_l170_170912

theorem lcm_of_6_8_10 : Nat.lcm (Nat.lcm 6 8) 10 = 120 := 
  by sorry

end lcm_of_6_8_10_l170_170912


namespace B_initial_investment_l170_170627

-- Definitions for investments and conditions
def A_init_invest : Real := 3000
def A_later_invest := 2 * A_init_invest

def A_yearly_investment := (A_init_invest * 6) + (A_later_invest * 6)

-- The amount B needs to invest for the yearly investment to be equal in the profit ratio 1:1
def B_investment (x : Real) := x * 12 

-- Definition of the proof problem
theorem B_initial_investment (x : Real) : A_yearly_investment = B_investment x → x = 4500 := 
by 
  sorry

end B_initial_investment_l170_170627


namespace parabola_directrix_l170_170966

theorem parabola_directrix (x y : ℝ) : 
  (∀ x: ℝ, y = -4 * x ^ 2 + 4) → (y = 65 / 16) := 
by 
  sorry

end parabola_directrix_l170_170966


namespace camden_total_legs_l170_170345

theorem camden_total_legs 
  (num_justin_dogs : ℕ := 14)
  (num_rico_dogs := num_justin_dogs + 10)
  (num_camden_dogs := 3 * num_rico_dogs / 4)
  (camden_3_leg_dogs : ℕ := 5)
  (camden_4_leg_dogs : ℕ := 7)
  (camden_2_leg_dogs : ℕ := 2) : 
  3 * camden_3_leg_dogs + 4 * camden_4_leg_dogs + 2 * camden_2_leg_dogs = 47 :=
by sorry

end camden_total_legs_l170_170345


namespace smallest_n_l170_170055

theorem smallest_n (n : ℕ) (hn : n > 0) (h : 623 * n % 32 = 1319 * n % 32) : n = 4 :=
sorry

end smallest_n_l170_170055


namespace carlos_gold_quarters_l170_170224

theorem carlos_gold_quarters:
  (let quarter_weight := 1 / 5 in
   let melt_value_per_ounce := 100 in
   let store_value_per_quarter := 0.25 in
   let quarters_per_ounce := 1 / quarter_weight in
   let total_melt_value := melt_value_per_ounce * quarters_per_ounce in
   let total_store_value := store_value_per_quarter * quarters_per_ounce in
   total_melt_value / total_store_value = 80) :=
by
  let quarter_weight := 1 / 5
  let melt_value_per_ounce := 100
  let store_value_per_quarter := 0.25
  let quarters_per_ounce := 1 / quarter_weight
  let total_melt_value := melt_value_per_ounce * quarters_per_ounce
  let total_store_value := store_value_per_quarter * quarters_per_ounce
  have : total_melt_value / total_store_value = 80 := sorry
  exact this

end carlos_gold_quarters_l170_170224


namespace complex_exp1990_sum_theorem_l170_170568

noncomputable def complex_exp1990_sum (x y : Complex) (h : x ≠ 0 ∧ y ≠ 0 ∧ x^2 + x*y + y^2 = 0) : Prop :=
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1

theorem complex_exp1990_sum_theorem (x y : Complex) (h : x ≠ 0 ∧ y ≠ 0 ∧ x^2 + x*y + y^2 = 0) : complex_exp1990_sum x y h :=
  sorry

end complex_exp1990_sum_theorem_l170_170568


namespace opposite_of_neg_two_l170_170736

theorem opposite_of_neg_two : ∃ x : Int, (-2 + x = 0) ∧ x = 2 :=
by
  use 2
  constructor
  . simp
  . rfl

end opposite_of_neg_two_l170_170736


namespace range_of_a_l170_170989
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + 1 + a
noncomputable def g (x : ℝ) : ℝ := 3 * Real.log x

theorem range_of_a (h : ∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f x a = -g x) : 
  0 ≤ a ∧ a ≤ Real.exp 3 - 4 := 
sorry

end range_of_a_l170_170989


namespace meaningful_fraction_implies_neq_neg4_l170_170908

theorem meaningful_fraction_implies_neq_neg4 (x : ℝ) : (x + 4 ≠ 0) ↔ (x ≠ -4) := 
by
  sorry

end meaningful_fraction_implies_neq_neg4_l170_170908


namespace log10_sum_diff_l170_170940

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log10_sum_diff :
  log10 32 + log10 50 - log10 8 = 2.301 :=
by
  sorry

end log10_sum_diff_l170_170940


namespace ray_initial_cents_l170_170722

theorem ray_initial_cents :
  ∀ (initial_cents : ℕ), 
    (∃ (peter_cents : ℕ), 
      peter_cents = 30 ∧
      ∃ (randi_cents : ℕ),
        randi_cents = 2 * peter_cents ∧
        randi_cents = peter_cents + 60 ∧
        peter_cents + randi_cents = initial_cents
    ) →
    initial_cents = 90 := 
by
    intros initial_cents h
    obtain ⟨peter_cents, hp, ⟨randi_cents, hr1, hr2, hr3⟩⟩ := h
    sorry

end ray_initial_cents_l170_170722


namespace a_minus_b_value_l170_170373

theorem a_minus_b_value (a b : ℤ) :
  (∀ x : ℝ, 9 * x^3 + y^2 + a * x - b * x^3 + x + 5 = y^2 + 5) → a - b = -10 :=
by
  sorry

end a_minus_b_value_l170_170373


namespace greatest_lcm_less_than_120_l170_170767

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b
noncomputable def multiples (x limit : ℕ) : List ℕ := List.range (limit / x) |>.map (λ n => x * (n + 1))

theorem greatest_lcm_less_than_120 :  GCM_of_10_and_15_lt_120 = 90
  where
    GCM_of_10_and_15_lt_120 : ℕ := match (multiples (lcm 10 15) 120) with
                                     | [] => 0
                                     | xs => xs.maximum'.getD 0 :=
  by
  apply sorry

end greatest_lcm_less_than_120_l170_170767


namespace minimum_value_l170_170250

open Real

theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : log (2^x) + log (8^y) = log 2) :
  ∃ (v : ℝ), v = 4 ∧ ∀ u, (∀ x y, x > 0 ∧ y > 0 → log (2^x) + log (8^y) = log 2 → x + 3*y = 1 → u = 4) := sorry

end minimum_value_l170_170250


namespace asep_wins_in_at_most_n_minus_5_div_4_steps_l170_170486

theorem asep_wins_in_at_most_n_minus_5_div_4_steps (n : ℕ) (h : n ≥ 14) : 
  ∃ f : ℕ → ℕ, (∀ X d : ℕ, 0 < d → d ∣ X → (X' = X + d ∨ X' = X - d) → (f X' ≤ f X + 1)) ∧ f n ≤ (n - 5) / 4 := 
sorry

end asep_wins_in_at_most_n_minus_5_div_4_steps_l170_170486


namespace product_of_divisors_of_18_l170_170451

def n : ℕ := 18

theorem product_of_divisors_of_18 : (∏ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), d) = 5832 := 
by 
  -- Proof of the theorem will go here
  sorry

end product_of_divisors_of_18_l170_170451


namespace jackson_star_fish_count_l170_170410

def total_starfish_per_spiral_shell (hermit_crabs : ℕ) (shells_per_crab : ℕ) (total_souvenirs : ℕ) : ℕ :=
  (total_souvenirs - (hermit_crabs + hermit_crabs * shells_per_crab)) / (hermit_crabs * shells_per_crab)

theorem jackson_star_fish_count :
  total_starfish_per_spiral_shell 45 3 450 = 2 :=
by
  -- The proof will be filled in here
  sorry

end jackson_star_fish_count_l170_170410


namespace problem1_problem2_l170_170211

noncomputable section

-- Define the setup: A bag with red and white balls
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def draw_count : ℕ := 4

-- Define the combination function
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problem 1: Drawing 4 balls such that they must be of two colors
theorem problem1 : 
  combination red_balls 3 * combination white_balls 1 
  + combination red_balls 2 * combination white_balls 2
  + combination red_balls 1 * combination white_balls 3 = 194 := 
by
  sorry

-- Problem 2: Drawing 4 balls such that the number of red balls drawn is not less than the number of white balls
theorem problem2 :
  combination red_balls 4 
  + combination red_balls 3 * combination white_balls 1
  + combination red_balls 2 * combination white_balls 2 = 115 := 
by
  sorry

end problem1_problem2_l170_170211


namespace expression_value_l170_170120

theorem expression_value (x y : ℝ) (h : x - 2 * y = 3) : 1 - 2 * x + 4 * y = -5 :=
by
  sorry

end expression_value_l170_170120


namespace triangle_arithmetic_progression_l170_170785

theorem triangle_arithmetic_progression (a d : ℝ) 
(h1 : (a-2*d)^2 + a^2 = (a+2*d)^2) 
(h2 : ∃ x : ℝ, (a = x * d) ∨ (d = x * a))
: (6 ∣ 6*d) ∧ (12 ∣ 6*d) ∧ (18 ∣ 6*d) ∧ (24 ∣ 6*d) ∧ (30 ∣ 6*d)
:= by
  sorry

end triangle_arithmetic_progression_l170_170785


namespace infinite_series_sum_l170_170152

theorem infinite_series_sum (c d : ℝ) (h1 : 0 < d) (h2 : 0 < c) (h3 : c > d) :
  (∑' n, 1 / (((2 * n + 1) * c - n * d) * ((2 * (n+1) - 1) * c - (n + 1 - 1) * d))) = 1 / ((c - d) * d) :=
sorry

end infinite_series_sum_l170_170152


namespace sequence_of_arrows_from_425_to_427_l170_170402

theorem sequence_of_arrows_from_425_to_427 :
  ∀ (arrows : ℕ → ℕ), (∀ n, arrows (n + 4) = arrows n) →
  (arrows 425, arrows 426, arrows 427) = (arrows 1, arrows 2, arrows 3) :=
by
  intros arrows h_period
  have h1 : arrows 425 = arrows 1 := by 
    sorry
  have h2 : arrows 426 = arrows 2 := by 
    sorry
  have h3 : arrows 427 = arrows 3 := by 
    sorry
  sorry

end sequence_of_arrows_from_425_to_427_l170_170402


namespace lumber_cut_length_l170_170666

-- Define lengths of the pieces
def length_W : ℝ := 5
def length_X : ℝ := 3
def length_Y : ℝ := 5
def length_Z : ℝ := 4

-- Define distances from line M to the left end of the pieces
def distance_X : ℝ := 3
def distance_Y : ℝ := 2
def distance_Z : ℝ := 1.5

-- Define the total length of the pieces
def total_length : ℝ := 17

-- Define the length per side when cut by L
def length_per_side : ℝ := 8.5

theorem lumber_cut_length :
    (∃ (d : ℝ), 4 * d - 6.5 = 8.5 ∧ d = 3.75) :=
by
  sorry

end lumber_cut_length_l170_170666


namespace range_of_n_l170_170678

def hyperbola_equation (m n : ℝ) : Prop :=
  (m^2 + n) * (3 * m^2 - n) > 0

def foci_distance (m n : ℝ) : Prop :=
  (m^2 + n) + (3 * m^2 - n) = 4

theorem range_of_n (m n : ℝ) :
  hyperbola_equation m n ∧ foci_distance m n →
  -1 < n ∧ n < 3 :=
by
  intro h
  have hyperbola_condition := h.1
  have distance_condition := h.2
  sorry

end range_of_n_l170_170678


namespace triangle_BC_length_l170_170412

noncomputable def length_of_BC (ABC : Triangle) (incircle_radius : ℝ) (altitude_A_to_BC : ℝ) 
    (BD_squared_plus_CD_squared : ℝ) : ℝ :=
  if incircle_radius = 3 ∧ altitude_A_to_BC = 15 ∧ BD_squared_plus_CD_squared = 33 then
    3 * Real.sqrt 7
  else
    0 -- This value is arbitrary, as the conditions above are specific

theorem triangle_BC_length {ABC : Triangle}
    (incircle_radius : ℝ) (altitude_A_to_BC : ℝ) (BD_squared_plus_CD_squared : ℝ) :
    incircle_radius = 3 →
    altitude_A_to_BC = 15 →
    BD_squared_plus_CD_squared = 33 →
    length_of_BC ABC incircle_radius altitude_A_to_BC BD_squared_plus_CD_squared = 3 * Real.sqrt 7 :=
by intros; sorry

end triangle_BC_length_l170_170412


namespace plus_signs_count_l170_170009

theorem plus_signs_count (num_symbols : ℕ) (at_least_one_plus_in_10 : ∀ s : Finset ℕ, s.card = 10 → (∃ i ∈ s, i < 14)) (at_least_one_minus_in_15 : ∀ s : Finset ℕ, s.card = 15 → (∃ i ∈ s, i ≥ 14)) : 
    ∃ (p m : ℕ), p + m = 23 ∧ p = 14 ∧ m = 9 := by
  sorry

end plus_signs_count_l170_170009


namespace monomial_same_type_m_n_sum_l170_170288

theorem monomial_same_type_m_n_sum (m n : ℕ) (x y : ℤ) 
  (h1 : 2 * x ^ (m - 1) * y ^ 2 = 1/3 * x ^ 2 * y ^ (n + 1)) : 
  m + n = 4 := 
sorry

end monomial_same_type_m_n_sum_l170_170288


namespace charity_meaning_l170_170616

theorem charity_meaning (noun_charity : String) (h : noun_charity = "charity") : 
  (noun_charity = "charity" → "charity" = "charitable organization") :=
by
  sorry

end charity_meaning_l170_170616


namespace max_alpha_for_2_alpha_divides_3n_plus_1_l170_170154

theorem max_alpha_for_2_alpha_divides_3n_plus_1 (n : ℕ) (hn : n > 0) : ∃ α : ℕ, (2 ^ α ∣ (3 ^ n + 1)) ∧ ¬ (2 ^ (α + 1) ∣ (3 ^ n + 1)) ∧ α = 1 :=
by
  sorry

end max_alpha_for_2_alpha_divides_3n_plus_1_l170_170154


namespace express_in_scientific_notation_l170_170142

theorem express_in_scientific_notation : (250000 : ℝ) = 2.5 * 10^5 := 
by {
  -- proof
  sorry
}

end express_in_scientific_notation_l170_170142


namespace solve_quad_1_solve_quad_2_l170_170728

theorem solve_quad_1 :
  ∀ (x : ℝ), x^2 - 5 * x - 6 = 0 ↔ x = 6 ∨ x = -1 := by
  sorry

theorem solve_quad_2 :
  ∀ (x : ℝ), (x + 1) * (x - 1) + x * (x + 2) = 7 + 6 * x ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
  sorry

end solve_quad_1_solve_quad_2_l170_170728


namespace product_of_divisors_of_18_l170_170458

theorem product_of_divisors_of_18 : 
  ∏ i in (finset.filter (λ x : ℕ, x ∣ 18) (finset.range (18 + 1))), i = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l170_170458


namespace plus_signs_count_l170_170030

theorem plus_signs_count (total_symbols : ℕ) (n : ℕ) (m : ℕ)
    (h1 : total_symbols = 23)
    (h2 : ∀ (s : Finset ℕ), s.card = 10 → (∃ x ∈ s, x = n))
    (h3 : ∀ (s : Finset ℕ), s.card = 15 → (∃ x ∈ s, x = m)) :
    n = 14 := by
  sorry

end plus_signs_count_l170_170030


namespace pipe_length_difference_l170_170472

theorem pipe_length_difference (total_length shorter_piece : ℕ) (h1 : total_length = 68) (h2 : shorter_piece = 28) : 
  total_length - shorter_piece * 2 = 12 := 
sorry

end pipe_length_difference_l170_170472


namespace Mairead_triathlon_l170_170056

noncomputable def convert_km_to_miles (km: Float) : Float :=
  0.621371 * km

noncomputable def convert_yards_to_miles (yd: Float) : Float :=
  0.000568182 * yd

noncomputable def convert_feet_to_miles (ft: Float) : Float :=
  0.000189394 * ft

noncomputable def total_distance_in_miles := 
  let run_distance_km := 40.0
  let run_distance_miles := convert_km_to_miles run_distance_km
  let walk_distance_miles := 3.0/5.0 * run_distance_miles
  let jog_distance_yd := 5.0 * (walk_distance_miles * 1760.0)
  let jog_distance_miles := convert_yards_to_miles jog_distance_yd
  let bike_distance_ft := 3.0 * (jog_distance_miles * 5280.0)
  let bike_distance_miles := convert_feet_to_miles bike_distance_ft
  let swim_distance_miles := 2.5
  run_distance_miles + walk_distance_miles + jog_distance_miles + bike_distance_miles + swim_distance_miles

theorem Mairead_triathlon:
  total_distance_in_miles = 340.449562 ∧
  (convert_km_to_miles 40.0) / 10.0 = 2.485484 ∧
  (3.0/5.0 * (convert_km_to_miles 40.0)) / 10.0 = 1.4912904 ∧
  (convert_yards_to_miles (5.0 * (3.0/5.0 * (convert_km_to_miles 40.0) * 1760.0))) / 10.0 = 7.45454544 ∧
  (convert_feet_to_miles (3.0 * (convert_yards_to_miles (5.0 * (3.0/5.0 * (convert_km_to_miles 40.0) * 1760.0)) * 5280.0))) / 10.0 = 22.36363636 ∧
  2.5 / 10.0 = 0.25 := sorry

end Mairead_triathlon_l170_170056


namespace harry_galleons_l170_170471

/--
Harry, Hermione, and Ron go to Diagon Alley to buy chocolate frogs. 
If Harry and Hermione spent one-fourth of their own money, they would spend 3 galleons in total. 
If Harry and Ron spent one-fifth of their own money, they would spend 24 galleons in total. 
Everyone has a whole number of galleons, and the total number of galleons between the three of them is a multiple of 7. 
Prove that the only possible number of galleons that Harry can have is 6.
-/
theorem harry_galleons (H He R : ℕ) (k : ℕ) :
    H + He = 12 →
    H + R = 120 → 
    H + He + R = 7 * k → 
    (H = 6) :=
by sorry

end harry_galleons_l170_170471


namespace tail_wind_distance_l170_170636

-- Definitions based on conditions
def speed_still_air : ℝ := 262.5
def t1 : ℝ := 3
def t2 : ℝ := 4

def effective_speed_tail_wind (w : ℝ) : ℝ := speed_still_air + w
def effective_speed_against_wind (w : ℝ) : ℝ := speed_still_air - w

theorem tail_wind_distance (w : ℝ) (d : ℝ) :
  effective_speed_tail_wind w * t1 = effective_speed_against_wind w * t2 →
  d = t1 * effective_speed_tail_wind w →
  d = 900 :=
by
  sorry

end tail_wind_distance_l170_170636


namespace opposite_of_neg_two_is_two_l170_170740

theorem opposite_of_neg_two_is_two (x : ℤ) (h : -2 + x = 0) : x = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l170_170740


namespace minimum_candies_to_identify_coins_l170_170333

-- Set up the problem: define the relevant elements.
inductive Coin : Type
| C1 : Coin
| C2 : Coin
| C3 : Coin
| C4 : Coin
| C5 : Coin

def values : List ℕ := [1, 2, 5, 10, 20]

-- Statement of the problem in Lean 4, no means to identify which is which except through purchases and change from vending machine.
theorem minimum_candies_to_identify_coins : ∃ n : ℕ, n = 4 :=
by
  -- Skipping the proof
  sorry

end minimum_candies_to_identify_coins_l170_170333


namespace directrix_parabola_l170_170963

-- Given the equation of the parabola and required transformations:
theorem directrix_parabola (d : ℚ) : 
  (∀ x : ℚ, y = -4 * x^2 + 4) → d = 65 / 16 :=
by sorry

end directrix_parabola_l170_170963


namespace customers_stayed_behind_l170_170088

theorem customers_stayed_behind : ∃ x : ℕ, (x + (x + 5) = 11) ∧ x = 3 := by
  sorry

end customers_stayed_behind_l170_170088


namespace find_number_l170_170360

theorem find_number (x : ℕ) (h : 15 * x = x + 196) : 15 * x = 210 :=
by
  sorry

end find_number_l170_170360


namespace vector_arithmetic_l170_170102

-- Define the vectors
def v1 : ℝ × ℝ := (3, -5)
def v2 : ℝ × ℝ := (2, -6)
def v3 : ℝ × ℝ := (-1, 4)

-- Define scalar multiplications
def scalar_mult1 : ℝ × ℝ := (12, -20)  -- 4 * v1
def scalar_mult2 : ℝ × ℝ := (6, -18)   -- 3 * v2

-- Define intermediate vector operations
def intermediate_vector1 : ℝ × ℝ := (6, -2)  -- (12, -20) - (6, -18)

-- Final operation
def final_vector : ℝ × ℝ := (5, 2)  -- (6, -2) + (-1, 4)

-- Prove the main statement
theorem vector_arithmetic : 
  (4 : ℝ) • v1 - (3 : ℝ) • v2 + v3 = final_vector := by
  sorry  -- proof placeholder

end vector_arithmetic_l170_170102


namespace pascal_triangle_probability_l170_170799

theorem pascal_triangle_probability :
  let total_elements := 20 * 21 / 2,
      ones := 1 + 19 * 2,
      twos := 18 * 2,
      elements := ones + twos in
  (total_elements = 210) →
  (ones = 39) →
  (twos = 36) →
  (elements = 75) →
  (75 / 210) = 5 / 14 :=
by
  intros,
  sorry

end pascal_triangle_probability_l170_170799


namespace prob_lathe_parts_l170_170035

theorem prob_lathe_parts (
  P_A_B1 : ℝ,
  P_A_B2 : ℝ,
  P_A_B3 : ℝ,
  P_B1 : ℝ,
  P_B2 : ℝ,
  P_B3 : ℝ
) : 
  (P_A_B1 = 0.05) →
  (P_A_B2 = 0.03) →
  (P_A_B3 = 0.03) →
  (P_B1 = 0.15) →
  (P_B2 = 0.25) →
  (P_B3 = 0.60) →
  (let P_A := P_A_B1 * P_B1 + P_A_B2 * P_B2 + P_A_B3 * P_B3 in
  P_A = 0.033 ∧
  P_B1 + P_B2 + P_B3 = 1 ∧
  (P_B1 * P_A_B1 / P_A = P_B2 * P_A_B2 / P_A) ∧
  (P_B1 * P_A_B1 / P_A + P_B2 * P_A_B2 / P_A ≠ P_B3 * P_A_B3 / P_A)) :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end prob_lathe_parts_l170_170035


namespace alpha_value_l170_170564

open Complex

theorem alpha_value (α β : ℂ) (h1 : β = 2 + 3 * I) (h2 : (α + β).im = 0) (h3 : (I * (2 * α - β)).im = 0) : α = 6 + 4 * I :=
by
  sorry

end alpha_value_l170_170564


namespace find_missing_number_l170_170660

theorem find_missing_number (x : ℕ) (h : 10010 - 12 * 3 * x = 9938) : x = 2 :=
by {
  sorry
}

end find_missing_number_l170_170660


namespace closest_integer_to_cube_root_of_200_l170_170612

theorem closest_integer_to_cube_root_of_200 : 
  ∃ (n : ℤ), n = 6 ∧ (n^3 = 216 ∨ n^3 > 125 ∧ n^3 < 216) := 
by
  existsi 6
  split
  · refl
  · right
    split
    · norm_num
    · norm_num

end closest_integer_to_cube_root_of_200_l170_170612


namespace x_plus_q_in_terms_of_q_l170_170395

theorem x_plus_q_in_terms_of_q (x q : ℝ) (h1 : |x - 5| = q) (h2 : x > 5) : x + q = 2 * q + 5 :=
by
  sorry

end x_plus_q_in_terms_of_q_l170_170395


namespace winning_votes_l170_170754

theorem winning_votes (V : ℝ) (h1 : 0.62 * V - 0.38 * V = 312) : 0.62 * V = 806 :=
by
  -- The proof should be written here, but we'll skip it as per the instructions.
  sorry

end winning_votes_l170_170754


namespace choose_roles_from_8_l170_170293

-- Define the number of people
def num_people : ℕ := 8
-- Define the function to count the number of ways to choose different persons for the roles
def choose_roles (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

theorem choose_roles_from_8 : choose_roles num_people = 336 := by
  -- sorry acts as a placeholder for the proof
  sorry

end choose_roles_from_8_l170_170293


namespace pipe_length_difference_l170_170474

theorem pipe_length_difference 
  (total_length shorter_piece : ℝ)
  (h1: total_length = 68) 
  (h2: shorter_piece = 28) : 
  ∃ longer_piece diff : ℝ, longer_piece = total_length - shorter_piece ∧ diff = longer_piece - shorter_piece ∧ diff = 12 :=
by
  sorry

end pipe_length_difference_l170_170474


namespace upper_pyramid_volume_l170_170082

noncomputable def volume_of_upper_smaller_pyramid : ℝ :=
  let base_edge_length := 10 * Real.sqrt 2 in
  let slant_edge_length := 12 in
  let plane_height := 4 in
  let initial_height := Real.sqrt (slant_edge_length^2 - (base_edge_length / Real.sqrt 2)^2) in
  let small_pyramid_height := initial_height - plane_height in
  let similarity_ratio := small_pyramid_height / initial_height in
  (base_edge_length * similarity_ratio)^2 * small_pyramid_height * (1 / 3)

theorem upper_pyramid_volume :
  volume_of_upper_smaller_pyramid = (1000 / 22) * (2 * Real.sqrt 11 - 4)^3 :=
by
  sorry

end upper_pyramid_volume_l170_170082


namespace min_bottles_to_fill_large_bottle_l170_170214

theorem min_bottles_to_fill_large_bottle (large_bottle_ml : Nat) (small_bottle1_ml : Nat) (small_bottle2_ml : Nat) (total_bottles : Nat) :
  large_bottle_ml = 800 ∧ small_bottle1_ml = 45 ∧ small_bottle2_ml = 60 ∧ total_bottles = 14 →
  ∃ x y : Nat, x * small_bottle1_ml + y * small_bottle2_ml = large_bottle_ml ∧ x + y = total_bottles :=
by
  intro h
  sorry

end min_bottles_to_fill_large_bottle_l170_170214


namespace exact_fraction_difference_l170_170647

theorem exact_fraction_difference :
  let x := (8:ℚ) / 11
  let y := (18:ℚ) / 25 
  x - y = (2:ℚ) / 275 :=
by
  -- Definitions from conditions: x = 0.\overline{72} and y = 0.72
  let x := (8:ℚ) / 11
  let y := (18:ℚ) / 25 
  -- Goal is to prove the exact fraction difference
  show x - y = (2:ℚ) / 275
  sorry

end exact_fraction_difference_l170_170647


namespace john_average_speed_l170_170298

theorem john_average_speed
  (uphill_distance : ℝ)
  (uphill_time : ℝ)
  (downhill_distance : ℝ)
  (downhill_time : ℝ)
  (uphill_time_is_45_minutes : uphill_time = 45)
  (downhill_time_is_15_minutes : downhill_time = 15)
  (uphill_distance_is_3_km : uphill_distance = 3)
  (downhill_distance_is_3_km : downhill_distance = 3)
  : (uphill_distance + downhill_distance) / ((uphill_time + downhill_time) / 60) = 6 := 
by
  sorry

end john_average_speed_l170_170298


namespace count_valid_a_values_l170_170662

def is_perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k * k = n

def valid_a_values (a : ℕ) : Prop :=
1 ≤ a ∧ a ≤ 100 ∧ is_perfect_square (16 * a + 9)

theorem count_valid_a_values :
  ∃ N : ℕ, N = Nat.card {a : ℕ | valid_a_values a} := sorry

end count_valid_a_values_l170_170662


namespace admission_fee_for_children_l170_170888

theorem admission_fee_for_children (x : ℝ) :
  (∀ (admission_fee_adult : ℝ) (total_people : ℝ) (total_fees_collected : ℝ) (children_admitted : ℝ) (adults_admitted : ℝ),
    admission_fee_adult = 4 ∧
    total_people = 315 ∧
    total_fees_collected = 810 ∧
    children_admitted = 180 ∧
    adults_admitted = total_people - children_admitted ∧
    total_fees_collected = children_admitted * x + adults_admitted * admission_fee_adult
  ) → x = 1.5 := sorry

end admission_fee_for_children_l170_170888


namespace determine_M_l170_170590

theorem determine_M (x y z M : ℚ) 
  (h1 : x + y + z = 120)
  (h2 : x - 10 = M)
  (h3 : y + 10 = M)
  (h4 : 10 * z = M) : 
  M = 400 / 7 := 
sorry

end determine_M_l170_170590


namespace fraction_undefined_l170_170136

theorem fraction_undefined (x : ℝ) : (x + 1 = 0) ↔ (x = -1) := 
  sorry

end fraction_undefined_l170_170136


namespace angle_PQR_is_90_l170_170295

variable (R P Q S : Type) [EuclideanGeometry R P Q S]
variable (RSP_is_straight : straight_line R S P)
variable (angle_QSP : ∡Q S P = 70)

theorem angle_PQR_is_90 : ∡P Q R = 90 :=
by
  sorry

end angle_PQR_is_90_l170_170295


namespace find_constants_l170_170960

theorem find_constants :
  ∃ (A B C : ℝ), (∀ x : ℝ, x ≠ 3 → x ≠ 4 → 
  (6 * x / ((x - 4) * (x - 3) ^ 2)) = (A / (x - 4) + B / (x - 3) + C / (x - 3) ^ 2)) ∧
  A = 24 ∧
  B = - 162 / 7 ∧
  C = - 18 :=
by
  use 24, -162 / 7, -18
  sorry

end find_constants_l170_170960


namespace cost_of_article_l170_170132

theorem cost_of_article (C : ℝ) (G : ℝ)
    (h1 : G = 520 - C)
    (h2 : 1.08 * G = 580 - C) :
    C = 230 :=
by
    sorry

end cost_of_article_l170_170132


namespace cube_sum_181_5_l170_170175

theorem cube_sum_181_5
  (u v w : ℝ)
  (h : (u - real.cbrt 17) * (v - real.cbrt 67) * (w - real.cbrt 97) = 1/2)
  (huvw_distinct : u ≠ v ∧ u ≠ w ∧ v ≠ w):
  u^3 + v^3 + w^3 = 181.5 :=
sorry

end cube_sum_181_5_l170_170175


namespace find_q_l170_170519

theorem find_q (q : ℚ) (h_nonzero: q ≠ 0) :
  ∃ q, (qx^2 - 18 * x + 8 = 0) → (324 - 32*q = 0) :=
begin
  sorry
end

end find_q_l170_170519


namespace boy_usual_time_l170_170196

noncomputable def usual_rate (R : ℝ) := R
noncomputable def usual_time (T : ℝ) := T
noncomputable def faster_rate (R : ℝ) := (7 / 6) * R
noncomputable def faster_time (T : ℝ) := T - 5

theorem boy_usual_time
  (R : ℝ) (T : ℝ) 
  (h1 : usual_rate R * usual_time T = faster_rate R * faster_time T) :
  T = 35 :=
by 
  unfold usual_rate usual_time faster_rate faster_time at h1
  sorry

end boy_usual_time_l170_170196


namespace trig_identity_l170_170389

-- Define the angle alpha with the given condition tan(alpha) = 2
variables (α : ℝ) (h : Real.tan α = 2)

-- State the theorem
theorem trig_identity : (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 :=
by
  sorry

end trig_identity_l170_170389


namespace product_of_divisors_of_18_is_5832_l170_170461

theorem product_of_divisors_of_18_is_5832 :
  ∏ d in (finset.filter (λ d : ℕ, 18 % d = 0) (finset.range 19)), d = 5832 :=
sorry

end product_of_divisors_of_18_is_5832_l170_170461


namespace tangency_point_is_ln2_l170_170119

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem tangency_point_is_ln2 (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) →
  (∃ m : ℝ, Real.exp m - Real.exp (-m) = 3 / 2) →
  m = Real.log 2 :=
by
  intro h1 h2
  sorry

end tangency_point_is_ln2_l170_170119


namespace ln_quadratic_decreasing_interval_l170_170450

noncomputable def decreasing_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

theorem ln_quadratic_decreasing_interval :
  decreasing_interval (λ x, Real.log (-x^2 - 2 * x + 8)) (-1) 2 :=
sorry

end ln_quadratic_decreasing_interval_l170_170450


namespace problem_l170_170885

variable (x y : ℝ)

theorem problem
  (h : (3 * x + 1) ^ 2 + |y - 3| = 0) :
  (x + 2 * y) * (x - 2 * y) + (x + 2 * y) ^ 2 - x * (2 * x + 3 * y) = -1 :=
sorry

end problem_l170_170885


namespace solution_set_of_inequality_l170_170981

variable {a x : ℝ}

theorem solution_set_of_inequality (h : 2 * a + 1 < 0) : 
  {x : ℝ | x^2 - 4 * a * x - 5 * a^2 > 0} = {x | x < 5 * a ∨ x > -a} := by
  sorry

end solution_set_of_inequality_l170_170981


namespace sequence_item_l170_170377

theorem sequence_item (n : ℕ) (a_n : ℕ → Rat) (h : a_n n = 2 / (n^2 + n)) : a_n n = 1 / 15 → n = 5 := by
  sorry

end sequence_item_l170_170377


namespace max_sum_a_b_l170_170682

theorem max_sum_a_b (a b : ℝ) (h : a^2 - a*b + b^2 = 1) : a + b ≤ 2 := 
by sorry

end max_sum_a_b_l170_170682


namespace magician_guarantee_success_l170_170075

-- Definitions based on the conditions in part a).
def deck_size : ℕ := 52

def is_edge_position (position : ℕ) : Prop :=
  position = 0 ∨ position = deck_size - 1

-- Statement of the proof problem in part c).
theorem magician_guarantee_success (position : ℕ) : is_edge_position position ↔ 
  forall spectator_strategy : ℕ → ℕ, 
  exists magician_strategy : (ℕ → ℕ → ℕ), 
  forall t : ℕ, t = position →
  (∃ k : ℕ, t = magician_strategy k (spectator_strategy k)) :=
sorry

end magician_guarantee_success_l170_170075


namespace perimeter_of_resulting_figure_l170_170701

-- Define the perimeters of the squares
def perimeter_small_square : ℕ := 40
def perimeter_large_square : ℕ := 100

-- Define the side lengths of the squares
def side_length_small_square := perimeter_small_square / 4
def side_length_large_square := perimeter_large_square / 4

-- Define the total perimeter of the uncombined squares
def total_perimeter_uncombined := perimeter_small_square + perimeter_large_square

-- Define the shared side length
def shared_side_length := side_length_small_square

-- Define the perimeter after considering the shared side
def resulting_perimeter := total_perimeter_uncombined - 2 * shared_side_length

-- Prove that the resulting perimeter is 120 cm
theorem perimeter_of_resulting_figure : resulting_perimeter = 120 := by
  sorry

end perimeter_of_resulting_figure_l170_170701


namespace students_exceed_guinea_pigs_l170_170948

theorem students_exceed_guinea_pigs :
  let classrooms := 5
  let students_per_classroom := 20
  let guinea_pigs_per_classroom := 3
  let total_students := classrooms * students_per_classroom
  let total_guinea_pigs := classrooms * guinea_pigs_per_classroom
  total_students - total_guinea_pigs = 85 :=
by
  -- using the conditions and correct answer identified above
  let classrooms := 5
  let students_per_classroom := 20
  let guinea_pigs_per_classroom := 3
  let total_students := classrooms * students_per_classroom
  let total_guinea_pigs := classrooms * guinea_pigs_per_classroom
  show total_students - total_guinea_pigs = 85
  sorry

end students_exceed_guinea_pigs_l170_170948


namespace correct_system_of_equations_l170_170862

variable (x y : Real)

-- Conditions
def condition1 : Prop := y = x + 4.5
def condition2 : Prop := 0.5 * y = x - 1

-- Main statement representing the correct system of equations
theorem correct_system_of_equations : condition1 x y ∧ condition2 x y :=
  sorry

end correct_system_of_equations_l170_170862


namespace initial_shirts_count_l170_170584

theorem initial_shirts_count 
  (S T x : ℝ)
  (h1 : 2 * S + x * T = 1600)
  (h2 : S + 6 * T = 1600)
  (h3 : 12 * T = 2400) :
  x = 4 :=
by
  sorry

end initial_shirts_count_l170_170584


namespace plus_signs_count_l170_170023

theorem plus_signs_count (p m : ℕ) (h_sum : p + m = 23)
                         (h_max_minus : m ≤ 9) (h_max_plus : p ≤ 14)
                         (h_at_least_one_plus_in_10 : ∀ (s : Finset (Fin 23)), s.card = 10 → ∃ i ∈ s, i < p)
                         (h_at_least_one_minus_in_15 : ∀ (s : Finset (Fin 23)), s.card = 15 → ∃ i ∈ s, m ≤ i) :
  p = 14 :=
by sorry

end plus_signs_count_l170_170023


namespace cube_sum_div_by_nine_l170_170044

theorem cube_sum_div_by_nine (n : ℕ) (hn : 0 < n) : (n^3 + (n+1)^3 + (n+2)^3) % 9 = 0 := by sorry

end cube_sum_div_by_nine_l170_170044


namespace closest_integer_to_cuberoot_of_200_l170_170613

theorem closest_integer_to_cuberoot_of_200 :
  ∃ n : ℤ, (n = 6) ∧ ( |200 - n^3| ≤ |200 - m^3|  ) ∀ m : ℤ := sorry

end closest_integer_to_cuberoot_of_200_l170_170613


namespace condition_of_A_with_respect_to_D_l170_170307

variables {A B C D : Prop}

theorem condition_of_A_with_respect_to_D (h1 : A → B) (h2 : ¬ (B → A)) (h3 : B ↔ C) (h4 : C → D) (h5 : ¬ (D → C)) :
  (D → A) ∧ ¬ (A → D) :=
by
  sorry

end condition_of_A_with_respect_to_D_l170_170307


namespace probability_closer_to_6_than_0_l170_170928

-- Defining the probability problem
theorem probability_closer_to_6_than_0 :
  let interval := set.Icc 0 7 in
  (∀ point : ℝ, point ∈ interval → point > 3) → 
  (measure_theory.measure_space.measure (set.Icc 3 7) / 
   measure_theory.measure_space.measure interval = 4 / 7) :=
by
  sorry

end probability_closer_to_6_than_0_l170_170928


namespace no_outliers_in_dataset_l170_170809

theorem no_outliers_in_dataset :
  let D := [7, 20, 34, 34, 40, 42, 42, 44, 52, 58]
  let Q1 := 34
  let Q3 := 44
  let IQR := Q3 - Q1
  let lower_threshold := Q1 - 1.5 * IQR
  let upper_threshold := Q3 + 1.5 * IQR
  (∀ x ∈ D, x ≥ lower_threshold) ∧ (∀ x ∈ D, x ≤ upper_threshold) →
  ∀ x ∈ D, ¬(x < lower_threshold ∨ x > upper_threshold) :=
by 
  sorry

end no_outliers_in_dataset_l170_170809


namespace class_average_score_l170_170690

theorem class_average_score (n_boys n_girls : ℕ) (avg_score_boys avg_score_girls : ℕ) 
  (h_nb : n_boys = 12)
  (h_ng : n_girls = 4)
  (h_ab : avg_score_boys = 84)
  (h_ag : avg_score_girls = 92) : 
  (n_boys * avg_score_boys + n_girls * avg_score_girls) / (n_boys + n_girls) = 86 := 
by 
  sorry

end class_average_score_l170_170690


namespace math_problem_l170_170421

variable (a b c : ℝ)

theorem math_problem 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  (ab / c + bc / a + ca / b) ≥ Real.sqrt 3 := 
by
  sorry

end math_problem_l170_170421


namespace rice_weight_between_9_8_and_10_2_l170_170924

noncomputable def rice_weight_probability : ℝ :=
let μ := 10
let σ := 0.1
let Φ : ℝ → ℝ := normalCDF ⟨μ, σ⟩
Φ 2

theorem rice_weight_between_9_8_and_10_2 :
  rice_weight_probability = 0.9544 :=
sorry

end rice_weight_between_9_8_and_10_2_l170_170924


namespace find_a_b_l170_170850

noncomputable def curve (x a b : ℝ) : ℝ := x^2 + a * x + b

noncomputable def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem find_a_b (a b : ℝ) :
  (∃ (y : ℝ) (x : ℝ), (y = curve x a b) ∧ tangent_line 0 b ∧ (2 * 0 + a = -1) ∧ (0 - b + 1 = 0)) ->
  a = -1 ∧ b = 1 := 
by
  sorry

end find_a_b_l170_170850


namespace circle_divides_CD_in_ratio_l170_170864

variable (A B C D : Point)
variable (BC a : ℝ)
variable (AD : ℝ := (1 + Real.sqrt 15) * BC)
variable (radius : ℝ := (2 / 3) * BC)
variable (EF : ℝ := (Real.sqrt 7 / 3) * BC)
variable (is_isosceles_trapezoid : is_isosceles_trapezoid A B C D)
variable (circle_centered_at_C : circle_centered_at C radius)
variable (chord_EF : chord_intersects_base EF AD)

theorem circle_divides_CD_in_ratio (CD DK KC : ℝ) (H1 : CD = 2 * a)
  (H2 : DK + KC = CD) (H3 : KC = CD - DK) : DK / KC = 2 :=
sorry

end circle_divides_CD_in_ratio_l170_170864


namespace spiders_make_webs_l170_170397

theorem spiders_make_webs :
  (∀ (s d : ℕ), s = 7 ∧ d = 7 → (∃ w : ℕ, w = s)) ∧
  (∀ (d w : ℕ), w = 1 ∧ d = 7 → (∃ s : ℕ, s = w)) →
  (∀ (s : ℕ), s = 1) :=
by
  sorry

end spiders_make_webs_l170_170397


namespace hours_per_toy_l170_170787

-- Defining the conditions
def toys_produced (hours: ℕ) : ℕ := 40 
def hours_worked : ℕ := 80

-- Theorem: If a worker makes 40 toys in 80 hours, then it takes 2 hours to make one toy.
theorem hours_per_toy : (hours_worked / toys_produced hours_worked) = 2 :=
by
  sorry

end hours_per_toy_l170_170787


namespace roots_of_polynomial_l170_170233

noncomputable def polynomial : Polynomial ℝ := Polynomial.X^3 + Polynomial.X^2 - 6 * Polynomial.X - 6

theorem roots_of_polynomial :
  (Polynomial.rootSet polynomial ℝ) = {-1, 3, -2} := 
sorry

end roots_of_polynomial_l170_170233


namespace incorrect_statements_l170_170287

def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def monotonically_decreasing_in_pos (f : ℝ → ℝ) : Prop :=
∀ x y, 0 < x ∧ x < y → f y ≤ f x

theorem incorrect_statements
  (f : ℝ → ℝ)
  (hf_even : even_function f)
  (hf_decreasing : monotonically_decreasing_in_pos f) :
  ¬ (∀ a, f (2 * a) < f (-a)) ∧ ¬ (f π > f (-3)) ∧ ¬ (∀ a, f (a^2 + 1) < f 1) :=
by sorry

end incorrect_statements_l170_170287


namespace plus_signs_count_l170_170006

theorem plus_signs_count (total_symbols : ℕ) (n_plus n_minus : ℕ)
  (h1 : total_symbols = 23)
  (h2 : ∀ s : Finset ℕ, s.card = 10 → ∃ i ∈ s, i ≤ n_plus)
  (h3 : ∀ s : Finset ℕ, s.card = 15 → ∃ i ∈ s, i > n_plus) :
  n_plus = 14 :=
by
  -- the proof will go here
  sorry

end plus_signs_count_l170_170006


namespace dan_balloons_correct_l170_170577

-- Define the initial conditions
def sam_initial_balloons : Float := 46.0
def sam_given_fred_balloons : Float := 10.0
def total_balloons : Float := 52.0

-- Calculate Sam's remaining balloons
def sam_current_balloons : Float := sam_initial_balloons - sam_given_fred_balloons

-- Define the target: Dan's balloons
def dan_balloons := total_balloons - sam_current_balloons

-- Statement to prove
theorem dan_balloons_correct : dan_balloons = 16.0 := sorry

end dan_balloons_correct_l170_170577


namespace scout_weekend_earnings_l170_170164

-- Define the constants and conditions
def base_pay : ℝ := 10.0
def tip_per_delivery : ℝ := 5.0
def saturday_hours : ℝ := 4.0
def sunday_hours : ℝ := 5.0
def saturday_deliveries : ℝ := 5.0
def sunday_deliveries : ℝ := 8.0

-- Calculate total hours worked
def total_hours : ℝ := saturday_hours + sunday_hours

-- Calculate base pay for the weekend
def total_base_pay : ℝ := total_hours * base_pay

-- Calculate total number of deliveries
def total_deliveries : ℝ := saturday_deliveries + sunday_deliveries

-- Calculate total earnings from tips
def total_tips : ℝ := total_deliveries * tip_per_delivery

-- Calculate total earnings
def total_earnings : ℝ := total_base_pay + total_tips

-- Theorem to prove the total earnings is $155.00
theorem scout_weekend_earnings : total_earnings = 155.0 := by
  sorry

end scout_weekend_earnings_l170_170164


namespace first_worker_time_l170_170328

def productivity (x y z : ℝ) : Prop :=
  x + y + z = 20 ∧
  (20 / x) > 3 ∧
  (20 / x) + (60 / (y + z)) = 8

theorem first_worker_time (x y z : ℝ) (h : productivity x y z) : 
  (80 / x) = 16 :=
  sorry

end first_worker_time_l170_170328


namespace gary_initial_money_l170_170242

/-- The initial amount of money Gary had, given that he spent $55 and has $18 left. -/
theorem gary_initial_money (amount_spent : ℤ) (amount_left : ℤ) (initial_amount : ℤ) 
  (h1 : amount_spent = 55) 
  (h2 : amount_left = 18) 
  : initial_amount = amount_spent + amount_left :=
by
  sorry

end gary_initial_money_l170_170242


namespace probability_same_color_is_correct_l170_170276

noncomputable def total_ways_select_plates : ℕ := Nat.choose 11 3
def ways_select_red_plates : ℕ := Nat.choose 6 3
def ways_select_blue_plates : ℕ := Nat.choose 5 3
noncomputable def favorable_outcomes : ℕ := ways_select_red_plates + ways_select_blue_plates
noncomputable def probability_same_color : ℚ := favorable_outcomes / total_ways_select_plates

theorem probability_same_color_is_correct :
  probability_same_color = 2/11 := 
by
  sorry

end probability_same_color_is_correct_l170_170276


namespace unique_solution_condition_l170_170652

theorem unique_solution_condition (a b : ℝ) : (4 * x - 6 + a = (b + 1) * x + 2) → b ≠ 3 :=
by
  intro h
  -- Given the condition equation
  have eq1 : 4 * x - 6 + a = (b + 1) * x + 2 := h
  -- Simplify to the form (3 - b) * x = 8 - a
  sorry

end unique_solution_condition_l170_170652


namespace Abhay_takes_1_hour_less_than_Sameer_l170_170858

noncomputable def Sameer_speed := 42 / (6 - 2)
noncomputable def Abhay_time_doubled_speed := 42 / (2 * 7)
noncomputable def Sameer_time := 42 / Sameer_speed

theorem Abhay_takes_1_hour_less_than_Sameer
  (distance : ℝ := 42)
  (Abhay_speed : ℝ := 7)
  (Sameer_speed : ℝ := Sameer_speed)
  (time_Sameer : ℝ := distance / Sameer_speed)
  (time_Abhay_doubled_speed : ℝ := distance / (2 * Abhay_speed)) :
  time_Sameer - time_Abhay_doubled_speed = 1 :=
by
  sorry

end Abhay_takes_1_hour_less_than_Sameer_l170_170858


namespace probability_top_card_diamond_l170_170084

theorem probability_top_card_diamond
  (cards : Finset (ℕ × ℕ))
  (h1 : cards.card = 60)
  (ranks : Finset ℕ)
  (h2 : ranks.card = 15)
  (suits : Finset ℕ)
  (h3 : suits.card = 4)
  (h4 : ∀ (r : ℕ), r ∈ ranks →
        (λ s : ℕ, (r, s)) '' suits ⊆ cards)
  (random_deck : list (ℕ × ℕ)) :
  (random_deck.head ∈ (λ r : ℕ, (r, 1)) '' ranks →
    (random_deck.head ∈ (λ r : ℕ, (r, 2)) '' ranks →
      (random_deck.head ∈ (λ r : ℕ, (r, 3)) '' ranks →
        (random_deck.head ∈ (λ r : ℕ, (r, 4)) '' ranks →
          ∃ probability : ℚ,
            probability = 1 / 4))) :=
begin
  sorry
end

end probability_top_card_diamond_l170_170084


namespace average_age_population_l170_170855

theorem average_age_population 
  (k : ℕ) 
  (hwomen : ℕ := 7 * k)
  (hmen : ℕ := 5 * k)
  (avg_women_age : ℕ := 40)
  (avg_men_age : ℕ := 30)
  (h_age_women : ℕ := avg_women_age * hwomen)
  (h_age_men : ℕ := avg_men_age * hmen) : 
  (h_age_women + h_age_men) / (hwomen + hmen) = 35 + 5/6 :=
by
  sorry -- proof will fill in here

end average_age_population_l170_170855


namespace average_math_score_l170_170418

theorem average_math_score (scores : Fin 4 → ℕ) (other_avg : ℕ) (num_students : ℕ) (num_other_students : ℕ)
  (h1 : scores 0 = 90) (h2 : scores 1 = 85) (h3 : scores 2 = 88) (h4 : scores 3 = 80)
  (h5 : other_avg = 82) (h6 : num_students = 30) (h7 : num_other_students = 26) :
  (90 + 85 + 88 + 80 + 26 * 82) / 30 = 82.5 :=
by
  sorry

end average_math_score_l170_170418


namespace pascal_element_probability_l170_170794

open Nat

def num_elems_first_n_rows (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

def count_ones (n : ℕ) : ℕ :=
  if n = 0 then 1 else if n = 1 then 2 else 2 * (n - 1) + 1

def count_twos (n : ℕ) : ℕ :=
  if n < 2 then 0 else 2 * (n - 2)

def probability_one_or_two (n : ℕ) : ℚ :=
  let total_elems := num_elems_first_n_rows n in
  let ones := count_ones n in
  let twos := count_twos n in
  (ones + twos) / total_elems

theorem pascal_element_probability :
  probability_one_or_two 20 = 5 / 14 :=
by
  sorry

end pascal_element_probability_l170_170794


namespace kittens_total_number_l170_170596

theorem kittens_total_number (W L H R : ℕ) (k : ℕ) 
  (h1 : W = 500) 
  (h2 : L = 80) 
  (h3 : H = 200) 
  (h4 : L + H + R = W) 
  (h5 : 40 * k ≤ R) 
  (h6 : R ≤ 50 * k) 
  (h7 : ∀ m, m ≠ 4 → m ≠ 6 → m ≠ k →
        40 * m ≤ R → R ≤ 50 * m → False) : 
  k = 5 ∧ 2 + 4 + k = 11 := 
by {
  -- The proof would go here
  sorry 
}

end kittens_total_number_l170_170596


namespace base_of_triangle_is_24_l170_170904

def triangle_sides_sum := 50
def left_side : ℕ := 12
def right_side := left_side + 2
def base := triangle_sides_sum - left_side - right_side

theorem base_of_triangle_is_24 :
  base = 24 :=
by 
  have h : left_side = 12 := rfl
  have h2 : right_side = 14 := by simp [right_side, h]
  have h3 : base = 24 := by simp [base, triangle_sides_sum, h, h2]
  exact h3

end base_of_triangle_is_24_l170_170904


namespace parabola_equation_through_origin_point_l170_170326

-- Define the conditions
def vertex_origin := (0, 0)
def point_on_parabola := (-2, 4)

-- Define what it means to be a standard equation of a parabola passing through a point
def standard_equation_passing_through (p : ℝ) (x y : ℝ) : Prop :=
  (y^2 = -2 * p * x ∨ x^2 = 2 * p * y)

-- The theorem stating the conclusion
theorem parabola_equation_through_origin_point :
  ∃ p > 0, standard_equation_passing_through p (-2) 4 ∧
  (4^2 = -8 * (-2) ∨ (-2)^2 = 4) := 
sorry

end parabola_equation_through_origin_point_l170_170326


namespace plus_signs_count_l170_170004

theorem plus_signs_count (total_symbols : ℕ) (n_plus n_minus : ℕ)
  (h1 : total_symbols = 23)
  (h2 : ∀ s : Finset ℕ, s.card = 10 → ∃ i ∈ s, i ≤ n_plus)
  (h3 : ∀ s : Finset ℕ, s.card = 15 → ∃ i ∈ s, i > n_plus) :
  n_plus = 14 :=
by
  -- the proof will go here
  sorry

end plus_signs_count_l170_170004


namespace distance_is_one_l170_170177

noncomputable def distance_between_bisectors_and_centroid : ℝ :=
  let AB := 9
  let AC := 12
  let BC := Real.sqrt (AB^2 + AC^2)
  let CD := BC / 2
  let CE := (2/3) * CD
  let r := (AB * AC) / (2 * (AB + AC + BC) / 2)
  let K := CE - r
  K

theorem distance_is_one : distance_between_bisectors_and_centroid = 1 :=
  sorry

end distance_is_one_l170_170177


namespace cube_edge_length_l170_170448

theorem cube_edge_length (sum_of_edges : ℕ) (num_edges : ℕ) (h : sum_of_edges = 144) (num_edges_h : num_edges = 12) :
  sum_of_edges / num_edges = 12 :=
by
  -- The proof is skipped.
  sorry

end cube_edge_length_l170_170448


namespace exists_root_abs_leq_2_abs_c_div_b_l170_170221

theorem exists_root_abs_leq_2_abs_c_div_b (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h_real_roots : ∃ x1 x2 : ℝ, a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ |x| ≤ 2 * |c / b| :=
by
  sorry

end exists_root_abs_leq_2_abs_c_div_b_l170_170221


namespace find_a_plus_b_l170_170246

def f (x : ℝ) : ℝ := x^3 + 3*x - 1

theorem find_a_plus_b (a b : ℝ) (h1 : f (a - 3) = -3) (h2 : f (b - 3) = 1) :
  a + b = 6 :=
sorry

end find_a_plus_b_l170_170246


namespace probability_smallest_divides_product_l170_170187

open Finset
open Rat

def set := {1, 2, 3, 4, 5, 6}

def comb (s : Finset ℕ) : Finset (Finset ℕ) := s.powerset.filter (λ x, x.card = 3)

theorem probability_smallest_divides_product :
  let outcomes := comb set in
  let successful := outcomes.filter (λ x, let a := x.min' (by dec_trivial),
                                            b := (x.erase a).min' (by dec_trivial),
                                            c := (x.erase a).erase b in
                                        a ∣ b * c) in
  (successful.card : ℚ) / (outcomes.card : ℚ) = 1 / 2 :=
by
  sorry

end probability_smallest_divides_product_l170_170187


namespace rationalize_cube_root_sum_l170_170721

theorem rationalize_cube_root_sum :
  let a := (5 : ℝ)^(1/3)
  let b := (3 : ℝ)^(1/3)
  let numerator := a^2 + a * b + b^2
  let denom := a - b
  let fraction := 1 / denom * numerator
  let A := 25
  let B := 15
  let C := 9
  let D := 2
  A + B + C + D = 51 :=
by
  let a := (5 : ℝ)^(1/3)
  let b := (3 : ℝ)^(1/3)
  let numerator := a^2 + a * b + b^2
  let denom := a - b
  let fraction := 1 / denom * numerator
  let A := 25
  let B := 15
  let C := 9
  let D := 2
  have step1 : (a^3 = 5) := by sorry
  have step2 : (b^3 = 3) := by sorry
  have denom_eq : denom = 2 := by sorry
  have frac_simp : fraction = (A^(1/3) + B^(1/3) + C^(1/3)) / D := by sorry
  show A + B + C + D = 51
  sorry

end rationalize_cube_root_sum_l170_170721


namespace stefan_more_vail_l170_170435

/-- Aiguo had 20 seashells --/
def a : ℕ := 20

/-- Vail had 5 less seashells than Aiguo --/
def v : ℕ := a - 5

/-- The total number of seashells of Stefan, Vail, and Aiguo is 66 --/
def total_seashells (s v a : ℕ) : Prop := s + v + a = 66

theorem stefan_more_vail (s v a : ℕ)
  (h_a : a = 20)
  (h_v : v = a - 5)
  (h_total : total_seashells s v a) :
  s - v = 16 :=
by {
  -- proofs would go here
  sorry
}

end stefan_more_vail_l170_170435


namespace transformed_polynomial_roots_l170_170684

theorem transformed_polynomial_roots (a b c d : ℝ) 
  (h1 : a + b + c + d = 0)
  (h2 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h3 : a * b * c * d ≠ 0)
  (h4 : Polynomial.eval a (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0)
  (h5 : Polynomial.eval b (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0)
  (h6 : Polynomial.eval c (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0)
  (h7 : Polynomial.eval d (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0):
  Polynomial.eval (-2 / d^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 ∧
    Polynomial.eval (-2 / c^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 ∧
    Polynomial.eval (-2 / b^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 ∧
    Polynomial.eval (-2 / a^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 :=
sorry

end transformed_polynomial_roots_l170_170684


namespace chocolates_left_l170_170881

-- Definitions based on the conditions
def initially_bought := 3
def gave_away := 2
def additionally_bought := 3

-- Theorem statement to prove
theorem chocolates_left : initially_bought - gave_away + additionally_bought = 4 := by
  -- Proof skipped
  sorry

end chocolates_left_l170_170881


namespace find_e_l170_170304

-- Define values for a, b, c, d
def a := 2
def b := 3
def c := 4
def d := 5

-- State the problem
theorem find_e (e : ℚ) : a + b + c + d + e = a + (b + (c - (d * e))) → e = -5/6 :=
by
  sorry

end find_e_l170_170304


namespace greatest_common_multiple_of_10_and_15_lt_120_l170_170762

theorem greatest_common_multiple_of_10_and_15_lt_120 : 
  ∃ (m : ℕ), lcm 10 15 = 30 ∧ m ∈ {i | i < 120 ∧ ∃ (k : ℕ), i = k * 30} ∧ m = 90 := 
sorry

end greatest_common_multiple_of_10_and_15_lt_120_l170_170762


namespace cos_B_in_triangle_l170_170686

theorem cos_B_in_triangle (A B C : ℝ) (h1 : 2 * B = A + C) (h2 : A + B + C = Real.pi) : 
  Real.cos B = 1 / 2 :=
sorry

end cos_B_in_triangle_l170_170686


namespace emails_in_afternoon_l170_170149

theorem emails_in_afternoon (A : ℕ) 
  (morning_emails : A + 3 = 10) : A = 7 :=
by {
    sorry
}

end emails_in_afternoon_l170_170149


namespace solution_set_for_inequality_l170_170832

theorem solution_set_for_inequality 
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_mono_dec : ∀ x y, 0 < x → x < y → f y ≤ f x)
  (h_f2 : f 2 = 0) :
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_for_inequality_l170_170832


namespace trajectory_midpoints_l170_170257

variables (a b c x y : ℝ)

def arithmetic_sequence (a b c : ℝ) : Prop := c = 2 * b - a

def line_eq (b a c x y : ℝ) : Prop := b * x + a * y + c = 0

def parabola_eq (x y : ℝ) : Prop := y^2 = -0.5 * x

theorem trajectory_midpoints
  (hac : arithmetic_sequence a b c)
  (line_cond : line_eq b a c x y)
  (parabola_cond : parabola_eq x y) :
  (x + 1 = -(2 * y - 1)^2) ∧ (y ≠ 1) :=
sorry

end trajectory_midpoints_l170_170257


namespace sin_arcsin_plus_arctan_l170_170956

theorem sin_arcsin_plus_arctan (a b : ℝ) (ha : a = Real.arcsin (4/5)) (hb : b = Real.arctan (1/2)) :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_arcsin_plus_arctan_l170_170956


namespace employees_percentage_6_years_or_more_l170_170144

-- Definition of the marks for each year range
def marks : List ℕ := [3, 6, 5, 4, 2, 2, 3, 2, 1, 1]

-- Number of employees per mark (as a variable)
variable (y : ℕ)

-- Calculation of total employees
def total_employees := List.sum marks * y

-- Calculation of employees with 6 years or more
def employees_6_years_or_more := (marks[6]! + marks[7]! + marks[8]! + marks[9]!) * y

-- Statement of the theorem
theorem employees_percentage_6_years_or_more :
  (employees_6_years_or_more y : ℚ) / (total_employees y) * 100 = 24.14 :=
by sorry

end employees_percentage_6_years_or_more_l170_170144


namespace fill_tank_in_18_minutes_l170_170419

-- Define the conditions
def rate_pipe_A := 1 / 9  -- tanks per minute
def rate_pipe_B := - (1 / 18) -- tanks per minute (negative because it's emptying)

-- Define the net rate of both pipes working together
def net_rate := rate_pipe_A + rate_pipe_B

-- Define the time to fill the tank when both pipes are working
def time_to_fill_tank := 1 / net_rate

theorem fill_tank_in_18_minutes : time_to_fill_tank = 18 := 
    by
    -- Sorry to skip the actual proof
    sorry

end fill_tank_in_18_minutes_l170_170419


namespace absolute_value_inequality_l170_170970

theorem absolute_value_inequality (x : ℝ) : 
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 4) ↔ (-1 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 7) := 
by sorry

end absolute_value_inequality_l170_170970


namespace incorrect_statement_C_l170_170366

noncomputable def f (a x : ℝ) : ℝ := x^2 * (Real.log x - a) + a

theorem incorrect_statement_C :
  ¬ (∀ a : ℝ, a > 0 → ∀ x : ℝ, x > 0 → f a x ≥ 0) := sorry

end incorrect_statement_C_l170_170366


namespace power_of_power_eq_512_l170_170757

theorem power_of_power_eq_512 : (2^3)^3 = 512 := by
  sorry

end power_of_power_eq_512_l170_170757


namespace product_of_divisors_of_18_l170_170460

theorem product_of_divisors_of_18 : (finset.prod (finset.filter (λ n, 18 % n = 0) (finset.range 19)) id) = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l170_170460


namespace range_of_p_l170_170363

-- Definitions of A and B
def A (p : ℝ) := {x : ℝ | x^2 + (p + 2) * x + 1 = 0}
def B := {x : ℝ | x > 0}

-- Condition of the problem: A ∩ B = ∅
def condition (p : ℝ) := ∀ x ∈ A p, x ∉ B

-- The statement to prove: p > -4
theorem range_of_p (p : ℝ) : condition p → p > -4 :=
by
  intro h
  sorry

end range_of_p_l170_170363


namespace roots_of_polynomial_l170_170234

noncomputable def polynomial : Polynomial ℝ := Polynomial.X^3 + Polynomial.X^2 - 6 * Polynomial.X - 6

theorem roots_of_polynomial :
  (Polynomial.rootSet polynomial ℝ) = {-1, 3, -2} := 
sorry

end roots_of_polynomial_l170_170234


namespace abs_opposite_sign_eq_sum_l170_170394

theorem abs_opposite_sign_eq_sum (a b : ℤ) (h : (|a + 1| * |b + 2| < 0)) : a + b = -3 :=
sorry

end abs_opposite_sign_eq_sum_l170_170394


namespace find_C_l170_170917

theorem find_C (A B C : ℕ) 
  (h1 : A + B + C = 700) 
  (h2 : A + C = 300) 
  (h3 : B + C = 600) 
  : C = 200 := sorry

end find_C_l170_170917


namespace least_number_to_add_l170_170052

theorem least_number_to_add (k : ℕ) : (1076 + k) % 23 = 0 ↔ k = 5 :=
by
  split
  · intro h
    -- we'd have to prove k = 5 here
    sorry
  · intro hk
    -- we'd have to prove (1076 + 5) % 23 = 0 here
    sorry

end least_number_to_add_l170_170052


namespace number_of_levels_l170_170806

-- Definitions of the conditions
def blocks_per_step : ℕ := 3
def steps_per_level : ℕ := 8
def total_blocks_climbed : ℕ := 96

-- The theorem to prove
theorem number_of_levels : (total_blocks_climbed / blocks_per_step) / steps_per_level = 4 := by
  sorry

end number_of_levels_l170_170806


namespace composition_points_value_l170_170438

theorem composition_points_value (f g : ℕ → ℕ) (ab cd : ℕ) 
  (h₁ : f 2 = 6) 
  (h₂ : f 3 = 4) 
  (h₃ : f 4 = 2)
  (h₄ : g 2 = 4) 
  (h₅ : g 3 = 2) 
  (h₆ : g 5 = 6) :
  let (a, b) := (2, 6)
  let (c, d) := (3, 4)
  ab + cd = (a * b) + (c * d) :=
by {
  sorry
}

end composition_points_value_l170_170438


namespace plus_signs_count_l170_170012

noncomputable def symbols : Type := fin 23 -- The type of 23 symbols

def is_plus (x : symbols) : Prop
def is_minus (x : symbols) : Prop

axiom total_symbols : ∀ (x : symbols), is_plus x ∨ is_minus x
axiom unique_classification : ∀ (x : symbols), ¬ (is_plus x ∧ is_minus x)
axiom ten_symbols_plus : ∀ (S : finset symbols), S.card = 10 → ∃ x ∈ S, is_plus x
axiom fifteen_symbols_minus : ∀ (S : finset symbols), S.card = 15 → ∃ x ∈ S, is_minus x

def num_plus_symbols : ℕ := finset.card (finset.filter is_plus finset.univ)

theorem plus_signs_count : num_plus_symbols = 14 :=
sorry

end plus_signs_count_l170_170012


namespace plus_signs_count_l170_170026

theorem plus_signs_count (p m : ℕ) (h_sum : p + m = 23)
                         (h_max_minus : m ≤ 9) (h_max_plus : p ≤ 14)
                         (h_at_least_one_plus_in_10 : ∀ (s : Finset (Fin 23)), s.card = 10 → ∃ i ∈ s, i < p)
                         (h_at_least_one_minus_in_15 : ∀ (s : Finset (Fin 23)), s.card = 15 → ∃ i ∈ s, m ≤ i) :
  p = 14 :=
by sorry

end plus_signs_count_l170_170026


namespace product_of_divisors_of_18_l170_170462

theorem product_of_divisors_of_18 : ∏ d in (Finset.filter (λ d, 18 % d = 0) (Finset.range 19)), d = 104976 := by
    sorry

end product_of_divisors_of_18_l170_170462


namespace area_difference_of_squares_l170_170921

theorem area_difference_of_squares (d1 d2 : ℝ) (h1 : d1 = 19) (h2 : d2 = 17) : 
  let s1 := d1 / Real.sqrt 2
  let s2 := d2 / Real.sqrt 2
  let area1 := s1 * s1
  let area2 := s2 * s2
  (area1 - area2) = 36 :=
by
  sorry

end area_difference_of_squares_l170_170921


namespace probability_same_color_l170_170259

-- Definitions with conditions
def totalPlates := 11
def redPlates := 6
def bluePlates := 5

-- Calculate combinations
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main statement
theorem probability_same_color (totalPlates redPlates bluePlates : ℕ) (h1 : totalPlates = 11) 
(h2 : redPlates = 6) (h3 : bluePlates = 5) : 
  (2 / 11 : ℚ) = ((choose redPlates 3 + choose bluePlates 3) : ℚ) / (choose totalPlates 3) :=
by
  -- Proof steps will be inserted here
  sorry

end probability_same_color_l170_170259


namespace Sara_Jim_equal_savings_l170_170162

theorem Sara_Jim_equal_savings:
  ∃ (w : ℕ), (∃ (sara_saved jim_saved : ℕ),
  sara_saved = 4100 + 10 * w ∧
  jim_saved = 15 * w ∧
  sara_saved = jim_saved) → w = 820 :=
by
  sorry

end Sara_Jim_equal_savings_l170_170162


namespace least_five_digit_congruent_l170_170047

theorem least_five_digit_congruent (x : ℕ) (h1 : x ≥ 10000) (h2 : x < 100000) (h3 : x % 17 = 8) : x = 10004 :=
by {
  sorry
}

end least_five_digit_congruent_l170_170047


namespace crease_length_l170_170488

theorem crease_length (A B C : ℝ) (h1 : A = 5) (h2 : B = 12) (h3 : C = 13) : ∃ D, D = 6.5 :=
by
  sorry

end crease_length_l170_170488


namespace fraction_red_marbles_l170_170853

theorem fraction_red_marbles (x : ℕ) (h : x > 0) :
  let blue := (2/3 : ℚ) * x
  let red := (1/3 : ℚ) * x
  let new_red := 3 * red
  let new_total := blue + new_red
  new_red / new_total = (3/5 : ℚ) := by
  sorry

end fraction_red_marbles_l170_170853


namespace Tim_pencils_value_l170_170192

variable (Sarah_pencils : ℕ)
variable (Tyrah_pencils : ℕ)
variable (Tim_pencils : ℕ)

axiom Tyrah_condition : Tyrah_pencils = 6 * Sarah_pencils
axiom Tim_condition : Tim_pencils = 8 * Sarah_pencils
axiom Tyrah_pencils_value : Tyrah_pencils = 12

theorem Tim_pencils_value : Tim_pencils = 16 :=
by
  sorry

end Tim_pencils_value_l170_170192


namespace question1_question2_l170_170156

-- Definitions:
def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a - 1) * x + (a^2 - 5) = 0}

-- Question 1 Statement:
theorem question1 (a : ℝ) (h : A ∩ B a = {2}) : a = -5 ∨ a = 1 := by
  sorry

-- Question 2 Statement:
theorem question2 (a : ℝ) (h : A ∪ B a = A) : a > 3 := by
  sorry

end question1_question2_l170_170156


namespace perimeter_of_square_field_l170_170469

variable (s a p : ℕ)

-- Given conditions as definitions
def area_eq_side_squared (a s : ℕ) : Prop := a = s^2
def perimeter_eq_four_sides (p s : ℕ) : Prop := p = 4 * s
def given_equation (a p : ℕ) : Prop := 6 * a = 6 * (2 * p + 9)

-- The proof statement
theorem perimeter_of_square_field (s a p : ℕ) 
  (h1 : area_eq_side_squared a s)
  (h2 : perimeter_eq_four_sides p s)
  (h3 : given_equation a p) :
  p = 36 :=
by
  sorry

end perimeter_of_square_field_l170_170469


namespace min_value_of_f_l170_170238

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 9) / sqrt (x^2 + 5)

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 4 :=
by
  sorry

end min_value_of_f_l170_170238


namespace trig_identity_l170_170983

theorem trig_identity (α : ℝ) (h : Real.cos (75 * Real.pi / 180 + α) = 1 / 3) :
  Real.sin (α - 15 * Real.pi / 180) + Real.cos (105 * Real.pi / 180 - α) = -2 / 3 :=
sorry

end trig_identity_l170_170983


namespace balloons_initial_count_l170_170513

theorem balloons_initial_count (B : ℕ) (G : ℕ) : ∃ G : ℕ, B = 7 * G + 4 := sorry

end balloons_initial_count_l170_170513


namespace complex_fraction_simplification_l170_170430

theorem complex_fraction_simplification (i : ℂ) (hi : i^2 = -1) : 
  ((2 - i) / (1 + 4 * i)) = (-2 / 17 - (9 / 17) * i) :=
  sorry

end complex_fraction_simplification_l170_170430


namespace opposite_of_neg_two_l170_170735

theorem opposite_of_neg_two : ∃ x : Int, (-2 + x = 0) ∧ x = 2 :=
by
  use 2
  constructor
  . simp
  . rfl

end opposite_of_neg_two_l170_170735


namespace books_left_over_l170_170691

theorem books_left_over (boxes : ℕ) (books_per_box_initial : ℕ) (books_per_box_new: ℕ) (total_books : ℕ) :
  boxes = 1500 →
  books_per_box_initial = 45 →
  books_per_box_new = 47 →
  total_books = boxes * books_per_box_initial →
  (total_books % books_per_box_new) = 8 :=
by intros; sorry

end books_left_over_l170_170691


namespace subset_definition_l170_170826

variable {α : Type} {A B : Set α}

theorem subset_definition :
  A ⊆ B ↔ ∀ a ∈ A, a ∈ B :=
by sorry

end subset_definition_l170_170826


namespace union_M_N_l170_170869

open Set Classical

noncomputable def M : Set ℝ := {x | x^2 = x}
noncomputable def N : Set ℝ := {x | Real.log x ≤ 0}

theorem union_M_N : M ∪ N = Icc 0 1 := by
  sorry

end union_M_N_l170_170869


namespace total_emails_received_l170_170068

theorem total_emails_received :
  let e1 := 16
  let e2 := e1 / 2
  let e3 := e2 / 2
  let e4 := e3 / 2
  e1 + e2 + e3 + e4 = 30 :=
by
  sorry

end total_emails_received_l170_170068


namespace smallest_sum_divisible_by_3_l170_170510

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def is_consecutive_prime (p1 p2 p3 p4 : ℕ) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
  (p2 = p1 + 4 ∨ p2 = p1 + 6 ∨ p2 = p1 + 2) ∧
  (p3 = p2 + 2 ∨ p3 = p2 + 4) ∧
  (p4 = p3 + 2 ∨ p4 = p3 + 4)

def greater_than_5 (p : ℕ) : Prop := p > 5

theorem smallest_sum_divisible_by_3 :
  ∃ (p1 p2 p3 p4 : ℕ), is_consecutive_prime p1 p2 p3 p4 ∧
                      greater_than_5 p1 ∧
                      (p1 + p2 + p3 + p4) % 3 = 0 ∧
                      (p1 + p2 + p3 + p4) = 48 :=
by sorry

end smallest_sum_divisible_by_3_l170_170510


namespace smallest_n_conditions_l170_170054

theorem smallest_n_conditions (n : ℕ) : 
  (∃ k m : ℕ, 4 * n = k^2 ∧ 5 * n = m^5 ∧ ∀ n' : ℕ, (∃ k' m' : ℕ, 4 * n' = k'^2 ∧ 5 * n' = m'^5) → n ≤ n') → 
  n = 625 :=
by
  intro h
  sorry

end smallest_n_conditions_l170_170054


namespace probability_same_color_plates_l170_170264

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem probability_same_color_plates :
  (choose 6 3 : ℚ) / (choose 11 3 : ℚ) = 4 / 33 := by
  sorry

end probability_same_color_plates_l170_170264


namespace alexa_fractions_l170_170493

theorem alexa_fractions (alexa_days ethans_days : ℕ) 
  (h1 : alexa_days = 9) (h2 : ethans_days = 12) : 
  alexa_days / ethans_days = 3 / 4 := 
by 
  sorry

end alexa_fractions_l170_170493


namespace ellipse_equation_l170_170664

theorem ellipse_equation (a b c : ℝ) (h0 : a > b) (h1 : b > 0) (h2 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (h3 : dist (3, y) (5 - 5 / 2, 0) = 6.5) (h4 : dist (3, y) (5 + 5 / 2, 0) = 3.5) : 
  ( ∀ x y, (x^2 / 25) + (y^2 / (75 / 4)) = 1 ) :=
sorry

end ellipse_equation_l170_170664


namespace Tim_has_16_pencils_l170_170193

variable (T_Sarah T_Tyrah T_Tim : Nat)

-- Conditions
def condition1 : Prop := T_Tyrah = 6 * T_Sarah
def condition2 : Prop := T_Tim = 8 * T_Sarah
def condition3 : Prop := T_Tyrah = 12

-- Theorem to prove
theorem Tim_has_16_pencils (h1 : condition1 T_Sarah T_Tyrah) (h2 : condition2 T_Sarah T_Tim) (h3 : condition3 T_Tyrah) : T_Tim = 16 :=
by
  sorry

end Tim_has_16_pencils_l170_170193


namespace remaining_volume_l170_170623

-- Given
variables (a d : ℚ) 
-- Define the volumes of sections as arithmetic sequence terms
def volume (n : ℕ) := a + n*d

-- Define total volume of bottom three sections
def bottomThreeVolume := volume a 0 + volume a d + volume a (2 * d) = 4

-- Define total volume of top four sections
def topFourVolume := volume a (5 * d) + volume a (6 * d) + volume a (7 * d) + volume a (8 * d) = 3

-- Define the volumes of the two middle sections
def middleTwoVolume := volume a (3 * d) + volume a (4 * d) = 2 + 3 / 22

-- Prove that the total volume of the remaining two sections is 2 3/22
theorem remaining_volume : bottomThreeVolume a d ∧ topFourVolume a d → middleTwoVolume a d :=
sorry  -- Placeholder for the actual proof

end remaining_volume_l170_170623


namespace initial_markup_percentage_l170_170216

-- Conditions:
-- 1. Initial price of the coat is $76.
-- 2. Increasing the price by $4 results in a 100% markup.
-- 3. A 100% markup implies the selling price is double the wholesale price.

theorem initial_markup_percentage (W : ℝ) (h1 : W + (76 - W) = 76)
  (h2 : 2 * W = 76 + 4) : (36 / 40) * 100 = 90 :=
by
  -- Using the conditions directly from the problem, we need to prove the theorem statement.
  sorry

end initial_markup_percentage_l170_170216


namespace ara_final_height_is_59_l170_170098

noncomputable def initial_shea_height : ℝ := 51.2
noncomputable def initial_ara_height : ℝ := initial_shea_height + 4
noncomputable def final_shea_height : ℝ := 64
noncomputable def shea_growth : ℝ := final_shea_height - initial_shea_height
noncomputable def ara_growth : ℝ := shea_growth / 3
noncomputable def final_ara_height : ℝ := initial_ara_height + ara_growth

theorem ara_final_height_is_59 :
  final_ara_height = 59 := by
  sorry

end ara_final_height_is_59_l170_170098


namespace articles_selling_price_eq_cost_price_of_50_articles_l170_170685

theorem articles_selling_price_eq_cost_price_of_50_articles (C S : ℝ) (N : ℕ) 
  (h1 : 50 * C = N * S) (h2 : S = 2 * C) : N = 25 := by
  sorry

end articles_selling_price_eq_cost_price_of_50_articles_l170_170685


namespace sum_of_consecutive_integers_product_384_l170_170324

theorem sum_of_consecutive_integers_product_384 :
  ∃ (a : ℤ), a * (a + 1) * (a + 2) = 384 ∧ a + (a + 1) + (a + 2) = 24 :=
by
  sorry

end sum_of_consecutive_integers_product_384_l170_170324


namespace pilot_fish_final_speed_relative_to_ocean_l170_170561

-- Define conditions
def keanu_speed : ℝ := 20 -- Keanu's speed in mph
def wind_speed : ℝ := 5 -- Wind speed in mph
def shark_speed (initial_speed: ℝ) : ℝ := 2 * initial_speed -- Shark doubles its speed

-- The pilot fish increases its speed by half the shark's increase
def pilot_fish_speed (initial_pilot_fish_speed shark_initial_speed : ℝ) : ℝ := 
  initial_pilot_fish_speed + 0.5 * shark_initial_speed

-- Define the speed of the pilot fish relative to the ocean
def pilot_fish_speed_relative_to_ocean (initial_pilot_fish_speed shark_initial_speed : ℝ) : ℝ := 
  pilot_fish_speed initial_pilot_fish_speed shark_initial_speed - wind_speed

-- Initial assumptions
def initial_pilot_fish_speed : ℝ := keanu_speed -- Pilot fish initially swims at the same speed as Keanu
def initial_shark_speed : ℝ := keanu_speed -- Let us assume the shark initially swims at the same speed as Keanu for simplicity

-- Prove the final speed of the pilot fish relative to the ocean
theorem pilot_fish_final_speed_relative_to_ocean : 
  pilot_fish_speed_relative_to_ocean initial_pilot_fish_speed initial_shark_speed = 25 := 
by sorry

end pilot_fish_final_speed_relative_to_ocean_l170_170561


namespace carlos_gold_quarters_l170_170225

theorem carlos_gold_quarters (quarter_weight : ℚ) 
  (store_value_per_quarter : ℚ) 
  (melt_value_per_ounce : ℚ) 
  (quarters_per_ounce : ℚ := 1 / quarter_weight) 
  (spent_value : ℚ := quarters_per_ounce * store_value_per_quarter)
  (melted_value: ℚ := melt_value_per_ounce) :
  quarter_weight = 1/5 ∧ store_value_per_quarter = 0.25 ∧ melt_value_per_ounce = 100 → 
  melted_value / spent_value = 80 := 
by
  intros h
  sorry

end carlos_gold_quarters_l170_170225


namespace cos_180_eq_neg1_sin_180_eq_0_l170_170346

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 := sorry
theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 := sorry

end cos_180_eq_neg1_sin_180_eq_0_l170_170346


namespace intersection_M_N_l170_170679

open Set

def M := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}
def N := {x : ℝ | 0 < x}
def intersection := {x : ℝ | 0 < x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = intersection := by
  sorry

end intersection_M_N_l170_170679


namespace greatest_a_inequality_l170_170511

theorem greatest_a_inequality :
  ∃ a : ℝ, (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ a * (x₁ * x₂ + x₂ * x₃ + x₃ * x₄ + x₄ * x₅)) ∧
          (∀ b : ℝ, (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ b * (x₁ * x₂ + x₂ * x₃ + x₃ * x₄ + x₄ * x₅)) → b ≤ a) ∧
          a = 2 / Real.sqrt 3 :=
sorry

end greatest_a_inequality_l170_170511


namespace carrie_remaining_money_l170_170649

def initial_money : ℝ := 200
def sweater_cost : ℝ := 36
def tshirt_cost : ℝ := 12
def tshirt_discount : ℝ := 0.10
def shoes_cost : ℝ := 45
def jeans_cost : ℝ := 52
def scarf_cost : ℝ := 18
def sales_tax_rate : ℝ := 0.05

-- Calculate tshirt price after discount
def tshirt_final_price : ℝ := tshirt_cost * (1 - tshirt_discount)

-- Sum all the item costs before tax
def total_cost_before_tax : ℝ := sweater_cost + tshirt_final_price + shoes_cost + jeans_cost + scarf_cost

-- Calculate the total sales tax
def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate

-- Calculate total cost after tax
def total_cost_after_tax : ℝ := total_cost_before_tax + sales_tax

-- Calculate the remaining money
def remaining_money (initial : ℝ) (total : ℝ) : ℝ := initial - total

theorem carrie_remaining_money
  (initial_money : ℝ)
  (sweater_cost : ℝ)
  (tshirt_cost : ℝ)
  (tshirt_discount : ℝ)
  (shoes_cost : ℝ)
  (jeans_cost : ℝ)
  (scarf_cost : ℝ)
  (sales_tax_rate : ℝ)
  (h₁ : initial_money = 200)
  (h₂ : sweater_cost = 36)
  (h₃ : tshirt_cost = 12)
  (h₄ : tshirt_discount = 0.10)
  (h₅ : shoes_cost = 45)
  (h₆ : jeans_cost = 52)
  (h₇ : scarf_cost = 18)
  (h₈ : sales_tax_rate = 0.05) :
  remaining_money initial_money (total_cost_after_tax) = 30.11 := 
by 
  simp only [remaining_money, total_cost_after_tax, total_cost_before_tax, tshirt_final_price, sales_tax];
  sorry

end carrie_remaining_money_l170_170649


namespace graph_is_hyperbola_l170_170349

theorem graph_is_hyperbola : ∀ (x y : ℝ), x^2 - 18 * y^2 - 6 * x + 4 * y + 9 = 0 → ∃ a b c d : ℝ, a * (x - b)^2 - c * (y - d)^2 = 1 :=
by
  -- Proof is omitted
  sorry

end graph_is_hyperbola_l170_170349


namespace todd_money_left_l170_170040

-- Define the initial amount of money Todd has
def initial_amount : ℕ := 20

-- Define the number of candy bars Todd buys
def number_of_candy_bars : ℕ := 4

-- Define the cost per candy bar
def cost_per_candy_bar : ℕ := 2

-- Define the total cost of the candy bars
def total_cost : ℕ := number_of_candy_bars * cost_per_candy_bar

-- Define the final amount of money Todd has left
def final_amount : ℕ := initial_amount - total_cost

-- The statement to be proven in Lean
theorem todd_money_left : final_amount = 12 := by
  -- The proof is omitted
  sorry

end todd_money_left_l170_170040


namespace jose_work_time_l170_170303

-- Define the variables for days taken by Jose and Raju
variables (J R T : ℕ)

-- State the conditions:
-- 1. Raju completes work in 40 days
-- 2. Together, Jose and Raju complete work in 8 days
axiom ra_work : R = 40
axiom together_work : T = 8

-- State the theorem that needs to be proven:
theorem jose_work_time (J R T : ℕ) (h1 : R = 40) (h2 : T = 8) : J = 10 :=
sorry

end jose_work_time_l170_170303


namespace maximum_marks_l170_170060

theorem maximum_marks (M : ℝ)
  (pass_threshold_percentage : ℝ := 33)
  (marks_obtained : ℝ := 92)
  (marks_failed_by : ℝ := 40) :
  (marks_obtained + marks_failed_by) = (pass_threshold_percentage / 100) * M → M = 400 := by
  sorry

end maximum_marks_l170_170060


namespace difference_of_reciprocals_l170_170549

theorem difference_of_reciprocals (p q : ℝ) (hp : 3 / p = 6) (hq : 3 / q = 15) : p - q = 3 / 10 :=
by
  sorry

end difference_of_reciprocals_l170_170549


namespace calculate_square_difference_l170_170804

theorem calculate_square_difference : 2023^2 - 2022^2 = 4045 := by
  sorry

end calculate_square_difference_l170_170804


namespace number_of_ordered_pairs_l170_170323

theorem number_of_ordered_pairs (x y : ℕ) : (x * y = 1716) → 
  (∃! n : ℕ, n = 18) :=
by
  sorry

end number_of_ordered_pairs_l170_170323


namespace max_markers_with_20_dollars_l170_170569

theorem max_markers_with_20_dollars (single_marker_cost : ℕ) (four_pack_cost : ℕ) (eight_pack_cost : ℕ) :
  single_marker_cost = 2 → four_pack_cost = 6 → eight_pack_cost = 10 → (∃ n, n = 16) := by
    intros h1 h2 h3
    existsi 16
    sorry

end max_markers_with_20_dollars_l170_170569


namespace opposite_of_neg_two_l170_170738

theorem opposite_of_neg_two : ∀ x : ℤ, (-2 + x = 0) → (x = 2) :=
begin
  assume x hx,
  sorry

end opposite_of_neg_two_l170_170738


namespace solution_set_l170_170310

-- Define the conditions
variable (f : ℝ → ℝ)
variable (odd_func : ∀ x : ℝ, f (-x) = -f x)
variable (increasing_pos : ∀ a b : ℝ, 0 < a → 0 < b → a < b → f a < f b)
variable (f_neg3_zero : f (-3) = 0)

-- State the theorem
theorem solution_set (x : ℝ) : x * f x < 0 ↔ (-3 < x ∧ x < 0 ∨ 0 < x ∧ x < 3) :=
sorry

end solution_set_l170_170310


namespace area_to_paint_correct_l170_170578

-- Define the measurements used in the problem
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window1_height : ℕ := 3
def window1_length : ℕ := 5
def window2_height : ℕ := 2
def window2_length : ℕ := 2

-- Definition of areas
def wall_area : ℕ := wall_height * wall_length
def window1_area : ℕ := window1_height * window1_length
def window2_area : ℕ := window2_height * window2_length

-- Definition of total area to paint
def total_area_to_paint : ℕ := wall_area - (window1_area + window2_area)

-- Theorem statement to prove the total area to paint is 131 square feet
theorem area_to_paint_correct : total_area_to_paint = 131 := by
  sorry

end area_to_paint_correct_l170_170578


namespace inequality_solution_ge_11_l170_170811

theorem inequality_solution_ge_11
  (m n : ℝ)
  (h1 : m > 0)
  (h2 : n > 1)
  (h3 : (1/m) + (2/(n-1)) = 1) :
  m + 2 * n ≥ 11 :=
sorry

end inequality_solution_ge_11_l170_170811


namespace least_five_digit_congruent_eight_mod_17_l170_170049

theorem least_five_digit_congruent_eight_mod_17 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 8 ∧ n = 10009 :=
by
  use 10009
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  sorry

end least_five_digit_congruent_eight_mod_17_l170_170049


namespace sam_watermelons_second_batch_l170_170427

theorem sam_watermelons_second_batch
  (initial_watermelons : ℕ)
  (total_watermelons : ℕ)
  (second_batch_watermelons : ℕ) :
  initial_watermelons = 4 →
  total_watermelons = 7 →
  second_batch_watermelons = total_watermelons - initial_watermelons →
  second_batch_watermelons = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sam_watermelons_second_batch_l170_170427


namespace pipe_length_difference_l170_170475

theorem pipe_length_difference 
  (total_length shorter_piece : ℝ)
  (h1: total_length = 68) 
  (h2: shorter_piece = 28) : 
  ∃ longer_piece diff : ℝ, longer_piece = total_length - shorter_piece ∧ diff = longer_piece - shorter_piece ∧ diff = 12 :=
by
  sorry

end pipe_length_difference_l170_170475


namespace least_five_digit_congruent_l170_170046

theorem least_five_digit_congruent (x : ℕ) (h1 : x ≥ 10000) (h2 : x < 100000) (h3 : x % 17 = 8) : x = 10004 :=
by {
  sorry
}

end least_five_digit_congruent_l170_170046


namespace tangent_line_equation_l170_170521

noncomputable def curve := fun x : ℝ => Real.sin (x + Real.pi / 3)

def tangent_line (x y : ℝ) : Prop :=
  x - 2 * y + Real.sqrt 3 = 0

theorem tangent_line_equation :
  tangent_line 0 (curve 0) := by
  unfold curve tangent_line
  sorry

end tangent_line_equation_l170_170521


namespace congruence_theorem_l170_170466

def triangle_congruent_SSA (a b : ℝ) (gamma : ℝ) :=
  b * b = a * a + (-2 * a * 5 * Real.cos gamma) + 25

theorem congruence_theorem : triangle_congruent_SSA 3 5 (150 * Real.pi / 180) :=
by
  -- Proof is omitted, based on the problem's instruction.
  sorry

end congruence_theorem_l170_170466


namespace min_max_F_l170_170254

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

def F (x y : ℝ) : ℝ := x^2 + y^2

theorem min_max_F (x y : ℝ) (h1 : f (y^2 - 6 * y + 11) + f (x^2 - 8 * x + 10) ≤ 0) (h2 : y ≥ 3) :
  ∃ (min_val max_val : ℝ), min_val = 13 ∧ max_val = 49 ∧
    min_val ≤ F x y ∧ F x y ≤ max_val :=
sorry

end min_max_F_l170_170254


namespace pascal_triangle_probability_l170_170798

theorem pascal_triangle_probability :
  let total_elements := 20 * 21 / 2,
      ones := 1 + 19 * 2,
      twos := 18 * 2,
      elements := ones + twos in
  (total_elements = 210) →
  (ones = 39) →
  (twos = 36) →
  (elements = 75) →
  (75 / 210) = 5 / 14 :=
by
  intros,
  sorry

end pascal_triangle_probability_l170_170798


namespace complement_of_A_with_respect_to_U_l170_170381

def U : Set ℤ := {1, 2, 3, 4, 5}
def A : Set ℤ := {x | abs (x - 3) < 2}
def C_UA : Set ℤ := { x | x ∈ U ∧ x ∉ A }

theorem complement_of_A_with_respect_to_U :
  C_UA = {1, 5} :=
by
  sorry

end complement_of_A_with_respect_to_U_l170_170381


namespace terminal_side_in_third_quadrant_l170_170975

theorem terminal_side_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  (∃ k : ℤ, α = k * π + π / 2 + π) := sorry

end terminal_side_in_third_quadrant_l170_170975


namespace red_ball_probability_l170_170447

noncomputable def Urn1_blue : ℕ := 5
noncomputable def Urn1_red : ℕ := 3
noncomputable def Urn2_blue : ℕ := 4
noncomputable def Urn2_red : ℕ := 4
noncomputable def Urn3_blue : ℕ := 8
noncomputable def Urn3_red : ℕ := 0

noncomputable def P_urn (n : ℕ) : ℝ := 1 / 3
noncomputable def P_red_urn1 : ℝ := (Urn1_red : ℝ) / (Urn1_blue + Urn1_red)
noncomputable def P_red_urn2 : ℝ := (Urn2_red : ℝ) / (Urn2_blue + Urn2_red)
noncomputable def P_red_urn3 : ℝ := (Urn3_red : ℝ) / (Urn3_blue + Urn3_red)

theorem red_ball_probability : 
  (P_urn 1 * P_red_urn1 + P_urn 2 * P_red_urn2 + P_urn 3 * P_red_urn3) = 7 / 24 :=
  by sorry

end red_ball_probability_l170_170447


namespace quadratic_transformation_l170_170833

noncomputable def transform_roots (p q r : ℚ) (u v : ℚ) 
    (h1 : u + v = -q / p) 
    (h2 : u * v = r / p) : Prop :=
  ∃ y : ℚ, y^2 - q^2 + 4 * p * r = 0

theorem quadratic_transformation (p q r u v : ℚ) 
    (h1 : u + v = -q / p) 
    (h2 : u * v = r / p) :
  ∃ y : ℚ, (y - (2 * p * u + q)) * (y - (2 * p * v + q)) = y^2 - q^2 + 4 * p * r :=
by {
  sorry
}

end quadratic_transformation_l170_170833


namespace sum_of_consecutive_perfect_squares_l170_170361

theorem sum_of_consecutive_perfect_squares (k : ℕ) (h_pos : 0 < k)
  (h_eq : 2 * k^2 + 2 * k + 1 = 181) : k = 9 ∧ (k + 1) = 10 := by
  sorry

end sum_of_consecutive_perfect_squares_l170_170361


namespace ellipse_equation_y_intercept_range_l170_170122

noncomputable def a := 2 * Real.sqrt 2
noncomputable def b := Real.sqrt 2
noncomputable def e := Real.sqrt 3 / 2
noncomputable def c := Real.sqrt 6
def M : ℝ × ℝ := (2, 1)

-- Condition: The ellipse equation form
def ellipse (x y : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Question 1: Proof that the ellipse equation is as given
theorem ellipse_equation :
  ellipse x y ↔ (x^2) / 8 + (y^2) / 2 = 1 := sorry

-- Condition: Line l is parallel to OM
def slope_OM := 1 / 2
def line_l (m x y : ℝ) : Prop := y = slope_OM * x + m

-- Question 2: Proof of the range for y-intercept m given the conditions
theorem y_intercept_range (m : ℝ) :
  (-Real.sqrt 2 < m ∧ m < 0 ∨ 0 < m ∧ m < Real.sqrt 2) ↔
  ∃ x1 y1 x2 y2,
    line_l m x1 y1 ∧ 
    line_l m x2 y2 ∧ 
    x1 ≠ x2 ∧ 
    y1 ≠ y2 ∧
    x1 * x2 + y1 * y2 < 0 := sorry

end ellipse_equation_y_intercept_range_l170_170122


namespace canoe_trip_shorter_l170_170039

def lake_diameter : ℝ := 2
def pi_value : ℝ := 3.14

theorem canoe_trip_shorter : (2 * pi_value * (lake_diameter / 2) - lake_diameter) = 4.28 :=
by
  sorry

end canoe_trip_shorter_l170_170039


namespace perfect_apples_count_l170_170141

-- Definitions (conditions)
def total_apples := 30
def too_small_fraction := (1 : ℚ) / 6
def not_ripe_fraction := (1 : ℚ) / 3
def too_small_apples := (too_small_fraction * total_apples : ℚ)
def not_ripe_apples := (not_ripe_fraction * total_apples : ℚ)

-- Statement of the theorem (proof problem)
theorem perfect_apples_count : total_apples - too_small_apples - not_ripe_apples = 15 := by
  sorry

end perfect_apples_count_l170_170141


namespace number_wall_problem_l170_170148

theorem number_wall_problem (m : ℤ) : 
  ((m + 5) + 16 + 18 = 56) → (m = 17) :=
by
  sorry

end number_wall_problem_l170_170148


namespace arithmetic_sequence_general_term_l170_170585

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℤ)
  (h1 : ∀ n m, a (n+1) - a n = a (m+1) - a m)
  (h2 : (a 2 + a 6) / 2 = 5)
  (h3 : (a 3 + a 7) / 2 = 7) :
  ∀ n, a n = 2 * n - 3 :=
by 
  sorry

end arithmetic_sequence_general_term_l170_170585


namespace typing_cost_equation_l170_170180

def typing_cost (x : ℝ) : ℝ :=
  200 * x + 80 * 3 + 20 * 6

theorem typing_cost_equation (x : ℝ) (h : typing_cost x = 1360) : x = 5 :=
by
  sorry

end typing_cost_equation_l170_170180


namespace closest_integer_to_cuberoot_of_200_l170_170611

theorem closest_integer_to_cuberoot_of_200 : 
  let c := (200 : ℝ)^(1/3)
  ∃ (k : ℤ), abs (c - 6) < abs (c - 5) :=
by
  let c := (200 : ℝ)^(1/3)
  existsi (6 : ℤ)
  sorry

end closest_integer_to_cuberoot_of_200_l170_170611


namespace fourth_term_of_geometric_sequence_l170_170634

theorem fourth_term_of_geometric_sequence 
  (a r : ℕ) 
  (h₁ : a = 3)
  (h₂ : a * r^2 = 75) :
  a * r^3 = 375 := 
by
  sorry

end fourth_term_of_geometric_sequence_l170_170634


namespace inequality_solution_fractional_equation_solution_l170_170922

-- Proof Problem 1
theorem inequality_solution (x : ℝ) : (1 - x) / 3 - x < 3 - (x + 2) / 4 → x > -2 :=
by
  sorry

-- Proof Problem 2
theorem fractional_equation_solution (x : ℝ) : (x - 2) / (2 * x - 1) + 1 = 3 / (2 * (1 - 2 * x)) → false :=
by
  sorry

end inequality_solution_fractional_equation_solution_l170_170922


namespace find_A_l170_170095

/-- Given that the equation Ax + 10y = 100 has two distinct positive integer solutions, prove that A = 10. -/
theorem find_A (A x1 y1 x2 y2 : ℕ) (h1 : A > 0) (h2 : x1 > 0) (h3 : y1 > 0) 
  (h4 : x2 > 0) (h5 : y2 > 0) (distinct_solutions : x1 ≠ x2 ∧ y1 ≠ y2) 
  (eq1 : A * x1 + 10 * y1 = 100) (eq2 : A * x2 + 10 * y2 = 100) : 
  A = 10 := sorry

end find_A_l170_170095


namespace proof_problem_l170_170367

noncomputable def problem : Prop :=
  ∃ (m n l : Type) (α β : Type) 
    (is_line : ∀ x, x = m ∨ x = n ∨ x = l)
    (is_plane : ∀ x, x = α ∨ x = β)
    (perpendicular : ∀ (l α : Type), Prop)
    (parallel : ∀ (l α : Type), Prop)
    (belongs_to : ∀ (l α : Type), Prop),
    (parallel l α → ∃ l', parallel l' α ∧ parallel l l') ∧
    (perpendicular m α ∧ perpendicular m β → parallel α β)

theorem proof_problem : problem :=
sorry

end proof_problem_l170_170367


namespace greatest_lcm_less_than_120_l170_170766

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b
noncomputable def multiples (x limit : ℕ) : List ℕ := List.range (limit / x) |>.map (λ n => x * (n + 1))

theorem greatest_lcm_less_than_120 :  GCM_of_10_and_15_lt_120 = 90
  where
    GCM_of_10_and_15_lt_120 : ℕ := match (multiples (lcm 10 15) 120) with
                                     | [] => 0
                                     | xs => xs.maximum'.getD 0 :=
  by
  apply sorry

end greatest_lcm_less_than_120_l170_170766


namespace sales_worth_l170_170467

variables (S : ℝ)
variables (old_scheme_remuneration new_scheme_remuneration : ℝ)

def old_scheme := 0.05 * S
def new_scheme := 1300 + 0.025 * (S - 4000)

theorem sales_worth :
  new_scheme S = old_scheme S + 600 →
  S = 24000 :=
by
  intro h
  sorry

end sales_worth_l170_170467


namespace factorization_of_polynomial_l170_170657

theorem factorization_of_polynomial (x : ℝ) :
  x^6 - x^4 - x^2 + 1 = (x - 1) * (x + 1) * (x^2 + 1) := 
sorry

end factorization_of_polynomial_l170_170657


namespace plus_signs_count_l170_170007

theorem plus_signs_count (num_symbols : ℕ) (at_least_one_plus_in_10 : ∀ s : Finset ℕ, s.card = 10 → (∃ i ∈ s, i < 14)) (at_least_one_minus_in_15 : ∀ s : Finset ℕ, s.card = 15 → (∃ i ∈ s, i ≥ 14)) : 
    ∃ (p m : ℕ), p + m = 23 ∧ p = 14 ∧ m = 9 := by
  sorry

end plus_signs_count_l170_170007


namespace smallest_number_of_tins_needed_l170_170876

variable (A : ℤ) (C : ℚ)

-- Conditions
def wall_area_valid : Prop := 1915 ≤ A ∧ A < 1925
def coverage_per_tin_valid : Prop := 17.5 ≤ C ∧ C < 18.5
def tins_needed_to_cover_wall (A : ℤ) (C : ℚ) : ℚ := A / C
def smallest_tins_needed : ℚ := 111

-- Proof problem statement
theorem smallest_number_of_tins_needed (A : ℤ) (C : ℚ)
    (h1 : wall_area_valid A)
    (h2 : coverage_per_tin_valid C)
    (h3 : 1915 ≤ A)
    (h4 : A < 1925)
    (h5 : 17.5 ≤ C)
    (h6 : C < 18.5) : 
  tins_needed_to_cover_wall A C + 1 ≥ smallest_tins_needed := by
    sorry

end smallest_number_of_tins_needed_l170_170876


namespace problem_part1_problem_part2_l170_170673

theorem problem_part1 :
  ∀ m : ℝ, (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 :=
by
sorry

theorem problem_part2 :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    4 * a^2 + 9 * b^2 + c^2 ≥ 36 / 49 ∧
    (4 * a^2 + 9 * b^2 + c^2 = 36 / 49 ↔ a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49) :=
by
sorry

end problem_part1_problem_part2_l170_170673


namespace solve_system_l170_170426

/-- Given the system of equations:
    3 * (x + y) - 4 * (x - y) = 5
    (x + y) / 2 + (x - y) / 6 = 0
  Prove that the solution is x = -1/3 and y = 2/3 
-/
theorem solve_system (x y : ℚ) 
  (h1 : 3 * (x + y) - 4 * (x - y) = 5)
  (h2 : (x + y) / 2 + (x - y) / 6 = 0) : 
  x = -1 / 3 ∧ y = 2 / 3 := 
sorry

end solve_system_l170_170426


namespace smallest_expression_l170_170133

theorem smallest_expression (x y : ℝ) (hx : x = 4) (hy : y = 2) :
  (y / x = 1 / 2) ∧ (y / x < x + y) ∧ (y / x < x * y) ∧ (y / x < x - y) ∧ (y / x < x / y) :=
by
  -- The proof is to be filled by the user
  sorry

end smallest_expression_l170_170133


namespace abc_zero_l170_170321

theorem abc_zero
  (a b c : ℝ)
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = (a * b * c)^3) :
  a * b * c = 0 := 
sorry

end abc_zero_l170_170321


namespace remainder_2503_div_28_l170_170769

theorem remainder_2503_div_28 : 2503 % 28 = 11 := 
by
  -- The proof goes here
  sorry

end remainder_2503_div_28_l170_170769


namespace limit_of_arithmetic_sequence_l170_170533

open Real
open BigOperators

noncomputable def a (n : ℕ) := 2 * n - 1

noncomputable def S (n : ℕ) := n^2

theorem limit_of_arithmetic_sequence : 
  (tendsto (λ n : ℕ, (S n : ℝ) / (a n)^2) at_top (𝓝 (1 / 4))) :=
begin
  sorry
end

end limit_of_arithmetic_sequence_l170_170533


namespace points_per_enemy_l170_170693

-- Definitions: total enemies, enemies not destroyed, points earned
def total_enemies : ℕ := 11
def enemies_not_destroyed : ℕ := 3
def points_earned : ℕ := 72

-- To prove: points per enemy
theorem points_per_enemy : points_earned / (total_enemies - enemies_not_destroyed) = 9 := 
by
  sorry

end points_per_enemy_l170_170693


namespace inequality_proof_l170_170378

theorem inequality_proof (b c : ℝ) (hb : 0 < b) (hc : 0 < c) :
  (b - c) ^ 2011 * (b + c) ^ 2011 * (c - b) ^ 2011 ≥ 
  (b ^ 2011 - c ^ 2011) * (b ^ 2011 + c ^ 2011) * (c ^ 2011 - b ^ 2011) := 
by
  sorry

end inequality_proof_l170_170378


namespace douglas_weight_proof_l170_170496

theorem douglas_weight_proof : 
  ∀ (anne_weight douglas_weight : ℕ), 
  anne_weight = 67 →
  anne_weight = douglas_weight + 15 →
  douglas_weight = 52 :=
by 
  intros anne_weight douglas_weight h1 h2 
  sorry

end douglas_weight_proof_l170_170496


namespace max_volume_of_hollow_cube_l170_170605

/-- 
We have 1000 solid cubes with edge lengths of 1 unit each. 
The small cubes can be glued together but not cut. 
The cube to be created is hollow with a wall thickness of 1 unit.
Prove that the maximum external volume of the cube we can create is 2197 cubic units.
--/

theorem max_volume_of_hollow_cube :
  ∃ x : ℕ, 6 * x^2 - 12 * x + 8 ≤ 1000 ∧ x^3 = 2197 :=
sorry

end max_volume_of_hollow_cube_l170_170605


namespace plus_signs_count_l170_170019

theorem plus_signs_count (n p m : ℕ) (h_total : n = 23) 
  (h_plus_min : ∀ s, s.card = 10 → ∃ t ∈ s, is_plus t)
  (h_minus_min : ∀ s, s.card = 15 → ∃ t ∈ s, is_minus t)
  : p = 14 :=
sorry

end plus_signs_count_l170_170019


namespace correct_match_results_l170_170621

-- Define the teams in the league
inductive Team
| Scotland : Team
| England  : Team
| Wales    : Team
| Ireland  : Team

-- Define a match result for a pair of teams
structure MatchResult where
  team1 : Team
  team2 : Team
  goals1 : ℕ
  goals2 : ℕ

def scotland_vs_england : MatchResult := {
  team1 := Team.Scotland,
  team2 := Team.England,
  goals1 := 3,
  goals2 := 0
}

-- All possible match results
def england_vs_ireland : MatchResult := {
  team1 := Team.England,
  team2 := Team.Ireland,
  goals1 := 1,
  goals2 := 0
}

def wales_vs_england : MatchResult := {
  team1 := Team.Wales,
  team2 := Team.England,
  goals1 := 1,
  goals2 := 1
}

def wales_vs_ireland : MatchResult := {
  team1 := Team.Wales,
  team2 := Team.Ireland,
  goals1 := 2,
  goals2 := 1
}

def scotland_vs_ireland : MatchResult := {
  team1 := Team.Scotland,
  team2 := Team.Ireland,
  goals1 := 2,
  goals2 := 0
}

theorem correct_match_results : 
  (england_vs_ireland.goals1 = 1 ∧ england_vs_ireland.goals2 = 0) ∧
  (wales_vs_england.goals1 = 1 ∧ wales_vs_england.goals2 = 1) ∧
  (scotland_vs_england.goals1 = 3 ∧ scotland_vs_england.goals2 = 0) ∧
  (wales_vs_ireland.goals1 = 2 ∧ wales_vs_ireland.goals2 = 1) ∧
  (scotland_vs_ireland.goals1 = 2 ∧ scotland_vs_ireland.goals2 = 0) :=
by 
  sorry

end correct_match_results_l170_170621


namespace area_intersection_M_N_l170_170256

open Complex Real

def M : set ℂ := {z | abs (z - 1) ≤ 1}
def N : set ℂ := {z | arg z ≥ π / 4}

theorem area_intersection_M_N :
  let S := area (M ∩ N)
  in S = (3 / 4) * π - 1 / 2 :=
sorry

end area_intersection_M_N_l170_170256


namespace pipeA_fills_tank_in_56_minutes_l170_170719

-- Define the relevant variables and conditions.
variable (t : ℕ) -- Time for Pipe A to fill the tank in minutes

-- Condition: Pipe B fills the tank 7 times faster than Pipe A
def pipeB_time (t : ℕ) := t / 7

-- Combined rate of Pipe A and Pipe B filling the tank in 7 minutes
def combined_rate (t : ℕ) := (1 / t) + (1 / pipeB_time t)

-- Given the combined rate fills the tank in 7 minutes
def combined_rate_equals (t : ℕ) := combined_rate t = 1 / 7

-- The proof statement
theorem pipeA_fills_tank_in_56_minutes (t : ℕ) (h : combined_rate_equals t) : t = 56 :=
sorry

end pipeA_fills_tank_in_56_minutes_l170_170719


namespace probability_same_color_plates_l170_170265

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem probability_same_color_plates :
  (choose 6 3 : ℚ) / (choose 11 3 : ℚ) = 4 / 33 := by
  sorry

end probability_same_color_plates_l170_170265


namespace tickets_used_l170_170775

def total_rides (ferris_wheel_rides bumper_car_rides : ℕ) : ℕ :=
  ferris_wheel_rides + bumper_car_rides

def tickets_per_ride : ℕ := 3

def total_tickets (total_rides tickets_per_ride : ℕ) : ℕ :=
  total_rides * tickets_per_ride

theorem tickets_used :
  total_tickets (total_rides 7 3) tickets_per_ride = 30 := by
  sorry

end tickets_used_l170_170775


namespace Robie_chocolates_left_l170_170883

def initial_bags : ℕ := 3
def given_away : ℕ := 2
def additional_bags : ℕ := 3

theorem Robie_chocolates_left : (initial_bags - given_away) + additional_bags = 4 :=
by
  sorry

end Robie_chocolates_left_l170_170883


namespace find_dolls_l170_170204

namespace DollsProblem

variables (S D : ℕ) -- Define S and D as natural numbers

-- Conditions as per the problem
def cond1 : Prop := 4 * S + 3 = D
def cond2 : Prop := 5 * S = D + 6

-- Theorem stating the problem
theorem find_dolls (h1 : cond1 S D) (h2 : cond2 S D) : D = 39 :=
by
  sorry

end DollsProblem

end find_dolls_l170_170204


namespace problem1_solution_l170_170729

theorem problem1_solution (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 16)
  (h2 : 5 * x - 6 * y = 33) : 
  x = 6 ∧ y = -1 / 2 := 
  by
  sorry

end problem1_solution_l170_170729


namespace complement_intersection_l170_170681

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection :
  ((U \ A) ∩ B) = {0} :=
by
  sorry

end complement_intersection_l170_170681


namespace third_group_members_l170_170479

-- Define the total number of members in the choir
def total_members : ℕ := 70

-- Define the number of members in the first group
def first_group_members : ℕ := 25

-- Define the number of members in the second group
def second_group_members : ℕ := 30

-- Prove that the number of members in the third group is 15
theorem third_group_members : total_members - first_group_members - second_group_members = 15 := 
by 
  sorry

end third_group_members_l170_170479


namespace maddie_weekend_watch_time_l170_170311

-- Defining the conditions provided in the problem
def num_episodes : ℕ := 8
def duration_per_episode : ℕ := 44
def minutes_on_monday : ℕ := 138
def minutes_on_tuesday : ℕ := 0
def minutes_on_wednesday : ℕ := 0
def minutes_on_thursday : ℕ := 21
def episodes_on_friday : ℕ := 2

-- Define the total time watched from Monday to Friday
def total_minutes_week : ℕ := num_episodes * duration_per_episode
def total_minutes_mon_to_fri : ℕ := 
  minutes_on_monday + 
  minutes_on_tuesday + 
  minutes_on_wednesday + 
  minutes_on_thursday + 
  (episodes_on_friday * duration_per_episode)

-- Define the weekend watch time
def weekend_watch_time : ℕ := total_minutes_week - total_minutes_mon_to_fri

-- The theorem to prove the correct answer
theorem maddie_weekend_watch_time : weekend_watch_time = 105 := by
  sorry

end maddie_weekend_watch_time_l170_170311


namespace value_of_a_l170_170990

variable (a : ℝ)

/-- The given function -/
def f (x : ℝ) : ℝ := a * Real.log x + (1/2) * a * x^2 - 2 * x

/-- The derivative of the function -/
def f_prime (x : ℝ) : ℝ := a / x + a * x - 2

/-- The function g(x) as defined in the solution -/
def g (x : ℝ) : ℝ := 2 * x / (1 + x^2)

/-- The main problem restated in Lean -/
theorem value_of_a (h : ∀ x ∈ Ioo 1 2, f_prime a x ≤ 0) : a < 1 := sorry

end value_of_a_l170_170990


namespace true_propositions_l170_170125

theorem true_propositions : 
  (∀ x : ℝ, x^3 < 1 → x^2 + 1 > 0) ∧ (∀ x : ℚ, x^2 = 2 → false) ∧ 
  (∀ x : ℕ, x^3 > x^2 → false) ∧ (∀ x : ℝ, x^2 + 1 > 0) :=
by 
  -- proof goes here
  sorry

end true_propositions_l170_170125


namespace roots_of_P_l170_170236

-- Define the polynomial P(x) = x^3 + x^2 - 6x - 6
noncomputable def P (x : ℝ) : ℝ := x^3 + x^2 - 6 * x - 6

-- Define the statement that the roots of the polynomial P are -1, sqrt(6), and -sqrt(6)
theorem roots_of_P : ∀ x : ℝ, P x = 0 ↔ (x = -1) ∨ (x = sqrt 6) ∨ (x = -sqrt 6) :=
sorry

end roots_of_P_l170_170236


namespace range_of_m_l170_170037

theorem range_of_m (m : ℝ) 
  (h : ∀ x : ℝ, 0 < x → m * x^2 + 2 * x + m ≤ 0) : m ≤ -1 :=
sorry

end range_of_m_l170_170037


namespace add_expression_l170_170923

theorem add_expression {k : ℕ} :
  (2 * k + 2) + (2 * k + 3) = (2 * k + 2) + (2 * k + 3) := sorry

end add_expression_l170_170923


namespace boat_speed_in_still_water_equals_6_l170_170291

def river_flow_rate : ℝ := 2
def distance_upstream : ℝ := 40
def distance_downstream : ℝ := 40
def total_time : ℝ := 15

theorem boat_speed_in_still_water_equals_6 :
  ∃ b : ℝ, (40 / (b - river_flow_rate) + 40 / (b + river_flow_rate) = total_time) ∧ b = 6 :=
sorry

end boat_speed_in_still_water_equals_6_l170_170291


namespace set_union_example_l170_170680

theorem set_union_example (x : ℕ) (M N : Set ℕ) (h1 : M = {0, x}) (h2 : N = {1, 2}) (h3 : M ∩ N = {2}) :
  M ∪ N = {0, 1, 2} := by
  sorry

end set_union_example_l170_170680


namespace opposite_of_neg_two_l170_170739

theorem opposite_of_neg_two : ∀ x : ℤ, (-2 + x = 0) → (x = 2) :=
begin
  assume x hx,
  sorry

end opposite_of_neg_two_l170_170739


namespace multiply_by_5_l170_170436

theorem multiply_by_5 (x : ℤ) (h : x - 7 = 9) : x * 5 = 80 := by
  sorry

end multiply_by_5_l170_170436


namespace max_value_expression_l170_170286

theorem max_value_expression (x y : ℝ) (h : x * y > 0) : 
  ∃ (m : ℝ), (∀ x y : ℝ, x * y > 0 → 
  m ≥ (x / (x + y) + 2 * y / (x + 2 * y))) ∧ 
  m = 4 - 2 * Real.sqrt 2 := 
sorry

end max_value_expression_l170_170286


namespace problem1_problem2_l170_170101

-- Given conditions
variables (x y : ℝ)

-- Problem 1: Prove that ((xy + 2) * (xy - 2) - 2 * x^2 * y^2 + 4) / (xy) = -xy
theorem problem1 : ((x * y + 2) * (x * y - 2) - 2 * x^2 * y^2 + 4) / (x * y) = - (x * y) :=
sorry

-- Problem 2: Prove that (2 * x + y)^2 - (2 * x + 3 * y) * (2 * x - 3 * y) = 4 * x * y + 10 * y^2
theorem problem2 : (2 * x + y)^2 - (2 * x + 3 * y) * (2 * x - 3 * y) = 4 * x * y + 10 * y^2 :=
sorry

end problem1_problem2_l170_170101


namespace perpendicular_vectors_implies_k_eq_2_l170_170128

variable (k : ℝ)
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-1, k)

theorem perpendicular_vectors_implies_k_eq_2 (h : (2 : ℝ) * (-1 : ℝ) + (1 : ℝ) * k = 0) : k = 2 := by
  sorry

end perpendicular_vectors_implies_k_eq_2_l170_170128


namespace segment_AB_length_l170_170147

-- Define the problem conditions
variables (AB CD h : ℝ)
variables (x : ℝ)
variables (AreaRatio : ℝ)
variable (k : ℝ := 5 / 2)

-- The given conditions
def condition1 : Prop := AB = 5 * x ∧ CD = 2 * x
def condition2 : Prop := AB + CD = 280
def condition3 : Prop := h = AB - 20
def condition4 : Prop := AreaRatio = k

-- The statement to prove
theorem segment_AB_length (h k : ℝ) (x : ℝ) :
  (AB = 5 * x ∧ CD = 2 * x) ∧ (AB + CD = 280) ∧ (h = AB - 20) ∧ (AreaRatio = k) → AB = 200 :=
by 
  sorry

end segment_AB_length_l170_170147


namespace opposite_of_neg_two_l170_170734

theorem opposite_of_neg_two : ∃ x : Int, (-2 + x = 0) ∧ x = 2 :=
by
  use 2
  constructor
  . simp
  . rfl

end opposite_of_neg_two_l170_170734


namespace same_color_probability_l170_170269

open Nat

theorem same_color_probability :
  let total_plates := 11
  let red_plates := 6
  let blue_plates := 5
  let chosen_plates := 3
  let total_ways := choose total_plates chosen_plates
  let red_ways := choose red_plates chosen_plates
  let blue_ways := choose blue_plates chosen_plates
  let same_color_ways := red_ways + blue_ways
  let probability := (same_color_ways : ℚ) / (total_ways : ℚ)
  probability = 2 / 11 := by sorry

end same_color_probability_l170_170269


namespace correct_point_on_hyperbola_l170_170991

-- Given condition
def hyperbola_condition (x y : ℝ) : Prop := x * y = -4

-- Question (translated to a mathematically equivalent proof)
theorem correct_point_on_hyperbola :
  hyperbola_condition (-2) 2 :=
sorry

end correct_point_on_hyperbola_l170_170991


namespace probability_of_one_or_two_l170_170796

/-- Represents the number of elements in the first 20 rows of Pascal's Triangle. -/
noncomputable def total_elements : ℕ := 210

/-- Represents the number of ones in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_ones : ℕ := 39

/-- Represents the number of twos in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_twos : ℕ :=18

/-- Prove that the probability of randomly choosing an element which is either 1 or 2
from the first 20 rows of Pascal's Triangle is 57/210. -/
theorem probability_of_one_or_two (h1 : total_elements = 210)
                                  (h2 : number_of_ones = 39)
                                  (h3 : number_of_twos = 18) :
    39 + 18 = 57 ∧ (57 : ℚ) / 210 = 57 / 210 :=
by {
    sorry
}

end probability_of_one_or_two_l170_170796


namespace range_of_f_smallest_positive_period_of_f_intervals_of_monotonic_increase_of_f_l170_170822

open Real

def f (x : ℝ) := 2 * sin x * sin x + 2 * sqrt 3 * sin x * cos x + 1

theorem range_of_f : Set.Icc 0 4 = Set.range f :=
sorry

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + π) = f x :=
sorry

theorem intervals_of_monotonic_increase_of_f (k : ℤ) :
  Set.Icc (-π / 6 + k * π) (π / 3 + k * π) ⊆ {x : ℝ | ∃ (m : ℤ), deriv f x > 0} :=
sorry

end range_of_f_smallest_positive_period_of_f_intervals_of_monotonic_increase_of_f_l170_170822


namespace plus_signs_count_l170_170010

theorem plus_signs_count (num_symbols : ℕ) (at_least_one_plus_in_10 : ∀ s : Finset ℕ, s.card = 10 → (∃ i ∈ s, i < 14)) (at_least_one_minus_in_15 : ∀ s : Finset ℕ, s.card = 15 → (∃ i ∈ s, i ≥ 14)) : 
    ∃ (p m : ℕ), p + m = 23 ∧ p = 14 ∧ m = 9 := by
  sorry

end plus_signs_count_l170_170010


namespace find_constants_a_b_l170_170308

def M : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3, 1],
  ![2, -2]
]

theorem find_constants_a_b :
  ∃ (a b : ℚ), (M⁻¹ = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ)) ∧
  a = 1/8 ∧ b = -1/8 :=
by
  sorry

end find_constants_a_b_l170_170308


namespace find_cola_cost_l170_170854

variable (C : ℝ)
variable (juice_cost water_cost : ℝ)
variable (colas_sold waters_sold juices_sold : ℕ)
variable (total_earnings : ℝ)

theorem find_cola_cost
  (h1 : juice_cost = 1.5)
  (h2 : water_cost = 1)
  (h3 : colas_sold = 15)
  (h4 : waters_sold = 25)
  (h5 : juices_sold = 12)
  (h6 : total_earnings = 88) :
  15 * C + 25 * water_cost + 12 * juice_cost = total_earnings → C = 3 :=
by
  intro h
  have eqn : 15 * C + 25 * 1 + 12 * 1.5 = 88 := by rw [h1, h2]; exact h
  -- the proof steps solving for C would go here
  sorry

end find_cola_cost_l170_170854


namespace sum_of_sequence_l170_170890

theorem sum_of_sequence (avg : ℕ → ℕ → ℕ) (n : ℕ) (total_sum : ℕ) 
  (condition : avg 16 272 = 17) : 
  total_sum = 272 := 
by 
  sorry

end sum_of_sequence_l170_170890


namespace machine_production_in_10_seconds_l170_170134

def items_per_minute : ℕ := 150
def seconds_per_minute : ℕ := 60
def production_rate_per_second : ℚ := items_per_minute / seconds_per_minute
def production_time_in_seconds : ℕ := 10
def expected_production_in_ten_seconds : ℚ := 25

theorem machine_production_in_10_seconds :
  (production_rate_per_second * production_time_in_seconds) = expected_production_in_ten_seconds :=
sorry

end machine_production_in_10_seconds_l170_170134


namespace positive_expression_with_b_l170_170823

-- Defining the conditions and final statement
open Real

theorem positive_expression_with_b (a : ℝ) : (a + 2) * (a + 5) * (a + 8) * (a + 11) + 82 > 0 := 
sorry

end positive_expression_with_b_l170_170823


namespace correct_option_is_d_l170_170284

theorem correct_option_is_d (x : ℚ) : -x^3 = (-x)^3 :=
sorry

end correct_option_is_d_l170_170284


namespace circumscribed_quadrilateral_identity_l170_170668

variables 
  (α β γ θ : ℝ)
  (h_angle_sum : α + β + γ + θ = 180)
  (OA OB OC OD AB BC CD DA : ℝ)
  (h_OA : OA = 1 / Real.sin α)
  (h_OB : OB = 1 / Real.sin β)
  (h_OC : OC = 1 / Real.sin γ)
  (h_OD : OD = 1 / Real.sin θ)
  (h_AB : AB = Real.sin (α + β) / (Real.sin α * Real.sin β))
  (h_BC : BC = Real.sin (β + γ) / (Real.sin β * Real.sin γ))
  (h_CD : CD = Real.sin (γ + θ) / (Real.sin γ * Real.sin θ))
  (h_DA : DA = Real.sin (θ + α) / (Real.sin θ * Real.sin α))

theorem circumscribed_quadrilateral_identity :
  OA * OC + OB * OD = Real.sqrt (AB * BC * CD * DA) := 
sorry

end circumscribed_quadrilateral_identity_l170_170668


namespace Tim_pencils_value_l170_170191

variable (Sarah_pencils : ℕ)
variable (Tyrah_pencils : ℕ)
variable (Tim_pencils : ℕ)

axiom Tyrah_condition : Tyrah_pencils = 6 * Sarah_pencils
axiom Tim_condition : Tim_pencils = 8 * Sarah_pencils
axiom Tyrah_pencils_value : Tyrah_pencils = 12

theorem Tim_pencils_value : Tim_pencils = 16 :=
by
  sorry

end Tim_pencils_value_l170_170191


namespace sphere_volume_l170_170749

theorem sphere_volume (h : 4 * π * r^2 = 256 * π) : (4 / 3) * π * r^3 = (2048 / 3) * π :=
by
  sorry

end sphere_volume_l170_170749


namespace probability_same_color_is_correct_l170_170282

-- Definitions of the conditions
def num_red : ℕ := 6
def num_blue : ℕ := 5
def total_plates : ℕ := num_red + num_blue
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probability statement
def prob_three_same_color : ℚ :=
  let total_ways := choose total_plates 3
  let ways_red := choose num_red 3
  let ways_blue := choose num_blue 3
  let favorable_ways := ways_red
  favorable_ways / total_ways

theorem probability_same_color_is_correct : prob_three_same_color = (4 : ℚ) / 33 := sorry

end probability_same_color_is_correct_l170_170282


namespace complex_expression_equality_l170_170283

theorem complex_expression_equality (i : ℂ) (h : i^2 = -1) : (1 + i)^16 - (1 - i)^16 = 0 := by
  sorry

end complex_expression_equality_l170_170283


namespace pieces_missing_l170_170172

def total_pieces : ℕ := 32
def pieces_present : ℕ := 24

theorem pieces_missing : total_pieces - pieces_present = 8 := by
sorry

end pieces_missing_l170_170172


namespace selection_of_representatives_l170_170074

theorem selection_of_representatives 
  (females : ℕ) (males : ℕ)
  (h_females : females = 3) (h_males : males = 4) :
  (females ≥ 1 ∧ males ≥ 1) →
  (females * (males * (males - 1) / 2) + (females * (females - 1) / 2 * males) = 30) := 
by
  sorry

end selection_of_representatives_l170_170074


namespace product_of_divisors_18_l170_170454

theorem product_of_divisors_18 : (∏ d in (list.range 18).filter (λ n, 18 % n = 0), d) = 18 ^ (9 / 2) :=
begin
  sorry
end

end product_of_divisors_18_l170_170454


namespace min_sin_x_plus_sin_z_l170_170115

theorem min_sin_x_plus_sin_z
  (x y z : ℝ)
  (h1 : sqrt 3 * cos x = cot y)
  (h2 : 2 * cos y = tan z)
  (h3 : cos z = 2 * cot x) :
  sin x + sin z ≥ -7 * sqrt 2 / 6 := 
sorry

end min_sin_x_plus_sin_z_l170_170115


namespace ribbon_length_difference_l170_170892

theorem ribbon_length_difference (S : ℝ) : 
  let Seojun_ribbon := S 
  let Siwon_ribbon := S + 8.8 
  let Seojun_new := Seojun_ribbon - 4.3
  let Siwon_new := Siwon_ribbon + 4.3 
  Siwon_new - Seojun_new = 17.4 :=
by
  -- Definition of original ribbon lengths
  let Seojun_ribbon := S
  let Siwon_ribbon := S + 8.8
  -- Seojun cuts and gives 4.3 meters to Siwon
  let Seojun_new := Seojun_ribbon - 4.3
  let Siwon_new := Siwon_ribbon + 4.3
  -- Compute the difference
  have h1 : Siwon_new - Seojun_new = (S + 8.8 + 4.3) - (S - 4.3) := by sorry
  -- Prove the final answer
  have h2 : Siwon_new - Seojun_new = 17.4 := by sorry

  exact h2

end ribbon_length_difference_l170_170892


namespace bob_age_l170_170937

variable {b j : ℝ}

theorem bob_age (h1 : b = 3 * j - 20) (h2 : b + j = 75) : b = 51 := by
  sorry

end bob_age_l170_170937


namespace coefficient_x2_in_expansion_l170_170357

theorem coefficient_x2_in_expansion (n k : ℕ) (h : n = 4 ∧ k = 2) :
  (nat.choose n k = 6) :=
by
  have h1 : nat.choose 4 2 = 6, by sorry,
  exact h1

end coefficient_x2_in_expansion_l170_170357


namespace eggs_ordered_l170_170755

theorem eggs_ordered (E : ℕ) (h1 : E > 0) (h_crepes : E * 1 / 4 = E / 4)
                     (h_cupcakes : 2 / 3 * (3 / 4 * E) = 1 / 2 * E)
                     (h_left : (3 / 4 * E - 2 / 3 * (3 / 4 * E)) = 9) :
  E = 18 := by
  sorry

end eggs_ordered_l170_170755


namespace airline_passenger_capacity_l170_170092

def seats_per_row : Nat := 7
def rows_per_airplane : Nat := 20
def airplanes_owned : Nat := 5
def flights_per_day_per_airplane : Nat := 2

def seats_per_airplane : Nat := rows_per_airplane * seats_per_row
def total_seats : Nat := airplanes_owned * seats_per_airplane
def total_flights_per_day : Nat := airplanes_owned * flights_per_day_per_airplane
def total_passengers_per_day : Nat := total_flights_per_day * total_seats

theorem airline_passenger_capacity :
  total_passengers_per_day = 7000 := sorry

end airline_passenger_capacity_l170_170092


namespace negation_of_proposition_l170_170178

theorem negation_of_proposition (x : ℝ) : 
  ¬ (|x| < 2 → x < 2) ↔ (|x| ≥ 2 → x ≥ 2) :=
sorry

end negation_of_proposition_l170_170178


namespace triangle_area_l170_170687

theorem triangle_area (B : Real) (AB AC : Real) 
  (hB : B = Real.pi / 6) 
  (hAB : AB = 2 * Real.sqrt 3)
  (hAC : AC = 2) : 
  let area := 1 / 2 * AB * AC * Real.sin B
  area = 2 * Real.sqrt 3 := by
  sorry

end triangle_area_l170_170687


namespace find_x_when_y_is_20_l170_170251

-- Definition of the problem conditions.
def constant_ratio (x y : ℝ) : Prop := ∃ k, (3 * x - 4) = k * (y + 7)

-- Main theorem statement.
theorem find_x_when_y_is_20 :
  (constant_ratio x 5 → constant_ratio 3 5) → 
  (constant_ratio x 20 → x = 5.0833) :=
  by sorry

end find_x_when_y_is_20_l170_170251


namespace coefficient_x4_in_expansion_l170_170760

theorem coefficient_x4_in_expansion : 
  (∑ k in Finset.range (9), (Nat.choose 8 k) * (3 : ℤ)^k * (2 : ℤ)^(8-k) * (X : ℤ[X])^k).coeff 4 = 90720 :=
by
  sorry

end coefficient_x4_in_expansion_l170_170760


namespace directrix_of_parabola_l170_170968

def parabola_eq (x : ℝ) : ℝ := -4 * x^2 + 4

theorem directrix_of_parabola : 
  ∃ y : ℝ, y = 65 / 16 :=
by
  sorry

end directrix_of_parabola_l170_170968


namespace two_le_three_l170_170557

/-- Proof that the proposition "2 ≤ 3" is true given the logical connective. -/
theorem two_le_three : 2 ≤ 3 := 
by
  sorry

end two_le_three_l170_170557


namespace pascal_element_probability_l170_170795

open Nat

def num_elems_first_n_rows (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

def count_ones (n : ℕ) : ℕ :=
  if n = 0 then 1 else if n = 1 then 2 else 2 * (n - 1) + 1

def count_twos (n : ℕ) : ℕ :=
  if n < 2 then 0 else 2 * (n - 2)

def probability_one_or_two (n : ℕ) : ℚ :=
  let total_elems := num_elems_first_n_rows n in
  let ones := count_ones n in
  let twos := count_twos n in
  (ones + twos) / total_elems

theorem pascal_element_probability :
  probability_one_or_two 20 = 5 / 14 :=
by
  sorry

end pascal_element_probability_l170_170795


namespace solution_set_for_inequality_l170_170746

theorem solution_set_for_inequality : 
  { x : ℝ | x * (x - 1) < 2 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end solution_set_for_inequality_l170_170746


namespace find_a_for_square_binomial_l170_170231

theorem find_a_for_square_binomial (a : ℚ) (h: ∃ (b : ℚ), ∀ (x : ℚ), 9 * x^2 + 21 * x + a = (3 * x + b)^2) : a = 49 / 4 := 
by 
  sorry

end find_a_for_square_binomial_l170_170231


namespace avg_speed_train_l170_170492

theorem avg_speed_train {D V : ℝ} (h1 : D = 20 * (90 / 60)) (h2 : 360 = 6 * 60) : 
  V = D / (360 / 60) :=
  by sorry

end avg_speed_train_l170_170492


namespace cristine_final_lemons_l170_170105

def cristine_lemons_initial : ℕ := 12
def cristine_lemons_given_to_neighbor : ℕ := 1 / 4 * cristine_lemons_initial
def cristine_lemons_left_after_giving : ℕ := cristine_lemons_initial - cristine_lemons_given_to_neighbor
def cristine_lemons_exchanged_for_oranges : ℕ := 1 / 3 * cristine_lemons_left_after_giving
def cristine_lemons_left_after_exchange : ℕ := cristine_lemons_left_after_giving - cristine_lemons_exchanged_for_oranges

theorem cristine_final_lemons : cristine_lemons_left_after_exchange = 6 :=
by
  sorry

end cristine_final_lemons_l170_170105


namespace triangle_perimeter_l170_170103

variable (y : ℝ)

theorem triangle_perimeter (h₁ : 2 * y > y) (h₂ : y > 0) :
  ∃ (P : ℝ), P = 2 * y + y * Real.sqrt 2 :=
sorry

end triangle_perimeter_l170_170103


namespace inequality_neg_multiply_l170_170390

theorem inequality_neg_multiply {a b : ℝ} (h : a > b) : -2 * a < -2 * b :=
sorry

end inequality_neg_multiply_l170_170390


namespace square_side_length_l170_170086

theorem square_side_length (s : ℝ) (h1 : 4 * s = 12) (h2 : s^2 = 9) : s = 3 :=
sorry

end square_side_length_l170_170086


namespace fraction_spent_is_one_third_l170_170915

-- Define the initial conditions and money variables
def initial_money := 32
def cost_bread := 3
def cost_candy := 2
def remaining_money_after_all := 18

-- Define the calculation for the money left after buying bread and candy bar
def money_left_after_bread_candy := initial_money - cost_bread - cost_candy

-- Define the calculation for the money spent on turkey
def money_spent_on_turkey := money_left_after_bread_candy - remaining_money_after_all

-- The fraction of the remaining money spent on the Turkey
noncomputable def fraction_spent_on_turkey := (money_spent_on_turkey : ℚ) / money_left_after_bread_candy

-- State the theorem that verifies the fraction spent on turkey is 1/3
theorem fraction_spent_is_one_third : fraction_spent_on_turkey = 1 / 3 := by
  sorry

end fraction_spent_is_one_third_l170_170915


namespace stratified_sampling_correct_l170_170933

-- Defining the conditions
def first_grade_students : ℕ := 600
def second_grade_students : ℕ := 680
def third_grade_students : ℕ := 720
def total_sample_size : ℕ := 50
def total_students := first_grade_students + second_grade_students + third_grade_students

-- Expected number of students to be sampled from first, second, and third grades
def expected_first_grade_sample := total_sample_size * first_grade_students / total_students
def expected_second_grade_sample := total_sample_size * second_grade_students / total_students
def expected_third_grade_sample := total_sample_size * third_grade_students / total_students

-- Main theorem statement
theorem stratified_sampling_correct :
  expected_first_grade_sample = 15 ∧
  expected_second_grade_sample = 17 ∧
  expected_third_grade_sample = 18 := by
  sorry

end stratified_sampling_correct_l170_170933


namespace train_speed_l170_170640

theorem train_speed (length : ℝ) (time : ℝ) (length_eq : length = 400) (time_eq : time = 16) :
  (length / time) * (3600 / 1000) = 90 :=
by 
  rw [length_eq, time_eq]
  sorry

end train_speed_l170_170640


namespace resulting_perimeter_l170_170703

theorem resulting_perimeter (p1 p2 : ℕ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  let a := p1 / 4 in
  let b := p2 / 4 in
  p1 + p2 - 2 * a = 120 :=
by
  sorry

end resulting_perimeter_l170_170703


namespace time_to_send_data_in_minutes_l170_170950

def blocks := 100
def chunks_per_block := 256
def transmission_rate := 100 -- chunks per second
def seconds_per_minute := 60

theorem time_to_send_data_in_minutes :
    (blocks * chunks_per_block) / transmission_rate / seconds_per_minute = 4 := by
  sorry

end time_to_send_data_in_minutes_l170_170950


namespace y_equals_px_div_5x_p_l170_170392

variable (p x y : ℝ)

theorem y_equals_px_div_5x_p (h : p = 5 * x * y / (x - y)) : y = p * x / (5 * x + p) :=
sorry

end y_equals_px_div_5x_p_l170_170392


namespace bug_travel_distance_half_l170_170635

-- Define the conditions
def isHexagonalGrid (side_length : ℝ) : Prop :=
  side_length = 1

def shortest_path_length (path_length : ℝ) : Prop :=
  path_length = 100

-- Define a theorem that encapsulates the problem statement
theorem bug_travel_distance_half (side_length path_length : ℝ)
  (H1 : isHexagonalGrid side_length)
  (H2 : shortest_path_length path_length) :
  ∃ one_direction_distance : ℝ, one_direction_distance = path_length / 2 :=
sorry -- Proof to be provided.

end bug_travel_distance_half_l170_170635


namespace candy_from_sister_l170_170824

variable (f : ℕ) (e : ℕ) (t : ℕ)

theorem candy_from_sister (h₁ : f = 47) (h₂ : e = 25) (h₃ : t = 62) :
  ∃ x : ℕ, x = t - (f - e) ∧ x = 40 :=
by sorry

end candy_from_sister_l170_170824


namespace largest_visits_l170_170063

theorem largest_visits (stores : ℕ) (total_visits : ℕ) (unique_visitors : ℕ) 
  (visits_two_stores : ℕ) (remaining_visitors : ℕ) : 
  stores = 7 ∧ total_visits = 21 ∧ unique_visitors = 11 ∧ visits_two_stores = 7 ∧ remaining_visitors = (unique_visitors - visits_two_stores) →
  (remaining_visitors * 2 <= total_visits - visits_two_stores * 2) → (∀ v : ℕ, v * unique_visitors = total_visits) →
  (∃ v_max : ℕ, v_max = 4) :=
by
  sorry

end largest_visits_l170_170063


namespace sodium_acetate_formed_is_3_l170_170359

-- Definitions for chemicals involved in the reaction
def AceticAcid : Type := ℕ -- Number of moles of acetic acid
def SodiumHydroxide : Type := ℕ -- Number of moles of sodium hydroxide
def SodiumAcetate : Type := ℕ -- Number of moles of sodium acetate

-- Given conditions as definitions
def reaction (acetic_acid naoh : ℕ) : ℕ :=
  if acetic_acid = naoh then acetic_acid else min acetic_acid naoh

-- Lean theorem statement
theorem sodium_acetate_formed_is_3 
  (acetic_acid naoh : ℕ) 
  (h1 : acetic_acid = 3) 
  (h2 : naoh = 3) :
  reaction acetic_acid naoh = 3 :=
by
  -- Proof body (to be completed)
  sorry

end sodium_acetate_formed_is_3_l170_170359


namespace sum_of_reciprocals_of_squares_l170_170810

open Real

theorem sum_of_reciprocals_of_squares {a b c : ℝ} (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = -7) (h3 : a * b * c = -2) :
  1 / a^2 + 1 / b^2 + 1 / c^2 = 73 / 4 :=
by
  sorry

end sum_of_reciprocals_of_squares_l170_170810


namespace team_answer_prob_team_expected_score_l170_170087

theorem team_answer_prob (P1 P2 P3 : ℝ) (h1 : P1 = 3 / 4) (h2 : (1 - P1) * (1 - P3) = 1 / 12) (h3 : P2 * P3 = 1 / 4) :
  (1 - (1 - P1) * (1 - P2) * (1 - P3)) = 91 / 96 :=
by
  sorry

theorem team_expected_score (P1 P2 P3 : ℝ) (h1 : P1 = 3 / 4) (h2 : (1 - P1) * (1 - P3) = 1 / 12) (h3 : P2 * P3 = 1 / 4) :
  30 * (10 * (91 / 96)) - 100 = 1475 / 8 :=
by
  sorry

end team_answer_prob_team_expected_score_l170_170087


namespace pq_true_l170_170877

-- Proposition p: a^2 + b^2 < 0 is false
def p_false (a b : ℝ) : Prop := ¬ (a^2 + b^2 < 0)

-- Proposition q: (a-2)^2 + |b-3| ≥ 0 is true
def q_true (a b : ℝ) : Prop := (a - 2)^2 + |b - 3| ≥ 0

-- Theorem stating that "p ∨ q" is true
theorem pq_true (a b : ℝ) (h1 : p_false a b) (h2 : q_true a b) : (a^2 + b^2 < 0 ∨ (a - 2)^2 + |b - 3| ≥ 0) :=
by {
  sorry
}

end pq_true_l170_170877


namespace total_interval_length_l170_170972

noncomputable def interval_length : ℝ :=
  1 / (1 + 2^Real.pi)

theorem total_interval_length :
  ∀ x : ℝ, x < 1 ∧ Real.tan (Real.log x / Real.log 4) > 0 →
  (∃ y, interval_length = y) :=
by
  sorry

end total_interval_length_l170_170972


namespace work_completion_days_l170_170058

theorem work_completion_days
  (A B : ℝ)
  (h1 : A + B = 1 / 12)
  (h2 : A = 1 / 20)
  : 1 / (A + B / 2) = 15 :=
by 
  sorry

end work_completion_days_l170_170058


namespace john_new_weekly_earnings_l170_170699

theorem john_new_weekly_earnings :
  let original_earnings : ℝ := 40
  let percentage_increase : ℝ := 37.5 / 100
  let raise_amount : ℝ := original_earnings * percentage_increase
  let new_weekly_earnings : ℝ := original_earnings + raise_amount
  new_weekly_earnings = 55 := 
by
  sorry

end john_new_weekly_earnings_l170_170699


namespace solve_for_x_l170_170468

/-- Given condition that 0.75 : x :: 5 : 9 -/
def ratio_condition (x : ℝ) : Prop := 0.75 / x = 5 / 9

theorem solve_for_x (x : ℝ) (h : ratio_condition x) : x = 1.35 := by
  sorry

end solve_for_x_l170_170468


namespace probability_same_color_plates_l170_170263

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem probability_same_color_plates :
  (choose 6 3 : ℚ) / (choose 11 3 : ℚ) = 4 / 33 := by
  sorry

end probability_same_color_plates_l170_170263


namespace boxes_needed_to_sell_l170_170560

theorem boxes_needed_to_sell (total_bars : ℕ) (bars_per_box : ℕ) (target_boxes : ℕ) (h₁ : total_bars = 710) (h₂ : bars_per_box = 5) : target_boxes = 142 :=
by
  sorry

end boxes_needed_to_sell_l170_170560


namespace opposite_of_neg_two_is_two_l170_170741

theorem opposite_of_neg_two_is_two (x : ℤ) (h : -2 + x = 0) : x = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l170_170741


namespace set_d_pythagorean_triple_l170_170340

theorem set_d_pythagorean_triple : (9^2 + 40^2 = 41^2) :=
by sorry

end set_d_pythagorean_triple_l170_170340


namespace percentage_increase_l170_170173

variable (A B y : ℝ)

theorem percentage_increase (h1 : B > A) (h2 : A > 0) :
  B = A + y / 100 * A ↔ y = 100 * (B - A) / A :=
by
  sorry

end percentage_increase_l170_170173


namespace calculate_expression_l170_170342

theorem calculate_expression : 14 - (-12) + (-25) - 17 = -16 := by
  -- definitions from conditions are understood and used here implicitly
  sorry

end calculate_expression_l170_170342


namespace probability_same_color_is_correct_l170_170277

noncomputable def total_ways_select_plates : ℕ := Nat.choose 11 3
def ways_select_red_plates : ℕ := Nat.choose 6 3
def ways_select_blue_plates : ℕ := Nat.choose 5 3
noncomputable def favorable_outcomes : ℕ := ways_select_red_plates + ways_select_blue_plates
noncomputable def probability_same_color : ℚ := favorable_outcomes / total_ways_select_plates

theorem probability_same_color_is_correct :
  probability_same_color = 2/11 := 
by
  sorry

end probability_same_color_is_correct_l170_170277


namespace original_price_of_shoes_l170_170417

theorem original_price_of_shoes (P : ℝ) (h1 : 0.80 * P = 480) : P = 600 := 
by
  sorry

end original_price_of_shoes_l170_170417


namespace plus_signs_count_l170_170016

noncomputable def solveSymbols : Prop :=
  ∃ p m : ℕ, p + m = 23 ∧ p ≤ 14 ∧ m ≤ 9 ∧ (∀ s : Finset (Fin 23), s.card = 10 → (s.image (λ i => if i < p then true else false)).count true ≥ 1) ∧ (∀ s : Finset (Fin 23), s.card = 15 → (s.image (λ i => if i < p then false else true)).count false ≥ 1) 

theorem plus_signs_count : solveSymbols :=
sorry

end plus_signs_count_l170_170016


namespace option_D_correct_l170_170712

variables (Line : Type) (Plane : Type)
variables (parallel : Line → Plane → Prop)
variables (perpendicular : Line → Plane → Prop)
variables (perpendicular_planes : Plane → Plane → Prop)

theorem option_D_correct (c : Line) (α β : Plane) :
  parallel c α → perpendicular c β → perpendicular_planes α β :=
sorry

end option_D_correct_l170_170712


namespace evens_before_odd_prob_l170_170219

open Nat

-- Definition of the problem parameters
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def odd_faces : Finset ℕ := {1, 3, 5, 7}
def even_faces : Finset ℕ := {2, 4, 6, 8}

-- Roll probability conditions
def prob_odd (n : ℕ) : ℚ := if n ∈ odd_faces then 1/2 else 0
def prob_even (n : ℕ) : ℚ := if n ∈ even_faces then 1/2 else 0

-- Probability calculation of seeing all evens before any odds
def prob_all_evens_before_first_odd : ℚ :=
  (1 / 2) * series_sum (5) (λ n, (1 / 2)^n - ∑ k in (range 4).filter(λ k, 1 ≤ k ∧ k ≤ 3),
                         (-1)^(k-1) * (binom 4 k) * (k / 4)^(n - 1))

-- Final probability statement to be proved
theorem evens_before_odd_prob : prob_all_evens_before_first_odd = 1 / 70 :=
  sorry

end evens_before_odd_prob_l170_170219


namespace stratified_sampling_freshman_l170_170782

def total_students : ℕ := 1800 + 1500 + 1200
def sample_size : ℕ := 150
def freshman_students : ℕ := 1200

/-- if a sample of 150 students is drawn using stratified sampling, 40 students should be drawn from the freshman year -/
theorem stratified_sampling_freshman :
  (freshman_students * sample_size) / total_students = 40 :=
by
  sorry

end stratified_sampling_freshman_l170_170782


namespace find_triangle_base_l170_170897

theorem find_triangle_base (left_side : ℝ) (right_side : ℝ) (base : ℝ) 
  (h_left : left_side = 12) 
  (h_right : right_side = left_side + 2)
  (h_sum : left_side + right_side + base = 50) :
  base = 24 := 
sorry

end find_triangle_base_l170_170897


namespace find_k_for_given_prime_l170_170416

theorem find_k_for_given_prime (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) (k : ℕ) 
  (h : ∃ a : ℕ, k^2 - p * k = a^2) : 
  k = (p + 1)^2 / 4 :=
sorry

end find_k_for_given_prime_l170_170416


namespace corn_height_growth_l170_170504

theorem corn_height_growth (init_height week1_growth week2_growth week3_growth : ℕ)
  (h0 : init_height = 0)
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  init_height + week1_growth + week2_growth + week3_growth = 22 :=
by sorry

end corn_height_growth_l170_170504


namespace maplewood_total_population_l170_170408

-- Define the number of cities
def num_cities : ℕ := 25

-- Define the bounds for the average population
def lower_bound : ℕ := 5200
def upper_bound : ℕ := 5700

-- Define the average population, calculated as the midpoint of the bounds
def average_population : ℕ := (lower_bound + upper_bound) / 2

-- Define the total population as the product of the number of cities and the average population
def total_population : ℕ := num_cities * average_population

-- Theorem statement to prove the total population is 136,250
theorem maplewood_total_population : total_population = 136250 := by
  -- Insert formal proof here
  sorry

end maplewood_total_population_l170_170408


namespace markup_correct_l170_170179

theorem markup_correct (purchase_price : ℝ) (overhead_percentage : ℝ) (net_profit : ℝ) :
  purchase_price = 48 → overhead_percentage = 0.15 → net_profit = 12 →
  (purchase_price * (1 + overhead_percentage) + net_profit - purchase_price) = 19.2 :=
by
  intros
  sorry

end markup_correct_l170_170179


namespace adam_has_more_apples_l170_170790

-- Define the number of apples Jackie has
def JackiesApples : Nat := 9

-- Define the number of apples Adam has
def AdamsApples : Nat := 14

-- Statement of the problem: Prove that Adam has 5 more apples than Jackie
theorem adam_has_more_apples :
  AdamsApples - JackiesApples = 5 :=
by
  sorry

end adam_has_more_apples_l170_170790


namespace average_of_data_is_six_l170_170534

def data : List ℕ := [4, 6, 5, 8, 7, 6]

theorem average_of_data_is_six : 
  (data.sum / data.length : ℚ) = 6 := 
by sorry

end average_of_data_is_six_l170_170534


namespace base_of_triangle_is_24_l170_170903

def triangle_sides_sum := 50
def left_side : ℕ := 12
def right_side := left_side + 2
def base := triangle_sides_sum - left_side - right_side

theorem base_of_triangle_is_24 :
  base = 24 :=
by 
  have h : left_side = 12 := rfl
  have h2 : right_side = 14 := by simp [right_side, h]
  have h3 : base = 24 := by simp [base, triangle_sides_sum, h, h2]
  exact h3

end base_of_triangle_is_24_l170_170903


namespace number_of_students_exclusively_in_math_l170_170546

variable (T M F K : ℕ)
variable (students_in_math students_in_foreign_language students_only_music : ℕ)
variable (students_not_in_music total_students_only_non_music : ℕ)

theorem number_of_students_exclusively_in_math (hT: T = 120) (hM: M = 82)
    (hF: F = 71) (hK: K = 20) :
    T - K = 100 →
    (M + F - 53 = T - K) →
    M - 53 = 29 :=
by
  intros
  sorry

end number_of_students_exclusively_in_math_l170_170546


namespace sum_of_xy_l170_170992

theorem sum_of_xy (x y : ℝ) (h1 : x^3 - 6*x^2 + 12*x = 13) (h2 : y^3 + 3*y - 3*y^2 = -4) : x + y = 3 :=
by sorry

end sum_of_xy_l170_170992


namespace tom_age_ratio_l170_170041

theorem tom_age_ratio (T N : ℕ)
  (sum_children : T = T) 
  (age_condition : T - N = 3 * (T - 4 * N)) :
  T / N = 11 / 2 := 
sorry

end tom_age_ratio_l170_170041


namespace inscribed_sphere_radius_l170_170697

-- Define the distances from points X and Y to the faces of the tetrahedron
variable (X_AB X_AD X_AC X_BC : ℝ)
variable (Y_AB Y_AD Y_AC Y_BC : ℝ)

-- Setting the given distances in the problem
axiom dist_X_AB : X_AB = 14
axiom dist_X_AD : X_AD = 11
axiom dist_X_AC : X_AC = 29
axiom dist_X_BC : X_BC = 8

axiom dist_Y_AB : Y_AB = 15
axiom dist_Y_AD : Y_AD = 13
axiom dist_Y_AC : Y_AC = 25
axiom dist_Y_BC : Y_BC = 11

-- The theorem to prove that the radius of the inscribed sphere of the tetrahedron is 17
theorem inscribed_sphere_radius : 
  ∃ r : ℝ, r = 17 ∧ 
  (∀ (d_X_AB d_X_AD d_X_AC d_X_BC d_Y_AB d_Y_AD d_Y_AC d_Y_BC: ℝ),
    d_X_AB = 14 ∧ d_X_AD = 11 ∧ d_X_AC = 29 ∧ d_X_BC = 8 ∧
    d_Y_AB = 15 ∧ d_Y_AD = 13 ∧ d_Y_AC = 25 ∧ d_Y_BC = 11 → 
    r = 17) :=
sorry

end inscribed_sphere_radius_l170_170697


namespace plus_signs_count_l170_170018

noncomputable def solveSymbols : Prop :=
  ∃ p m : ℕ, p + m = 23 ∧ p ≤ 14 ∧ m ≤ 9 ∧ (∀ s : Finset (Fin 23), s.card = 10 → (s.image (λ i => if i < p then true else false)).count true ≥ 1) ∧ (∀ s : Finset (Fin 23), s.card = 15 → (s.image (λ i => if i < p then false else true)).count false ≥ 1) 

theorem plus_signs_count : solveSymbols :=
sorry

end plus_signs_count_l170_170018


namespace range_of_x_l170_170289

theorem range_of_x (x : ℝ) (h : x - 5 ≥ 0) : x ≥ 5 := 
  sorry

end range_of_x_l170_170289


namespace jenna_reading_pages_l170_170301

theorem jenna_reading_pages :
  ∀ (total_pages goal_pages flight_pages busy_days total_days reading_days : ℕ),
    total_days = 30 →
    busy_days = 4 →
    flight_pages = 100 →
    goal_pages = 600 →
    reading_days = total_days - busy_days - 1 →
    (goal_pages - flight_pages) / reading_days = 20 :=
by
  intros total_pages goal_pages flight_pages busy_days total_days reading_days
  sorry

end jenna_reading_pages_l170_170301


namespace system_no_solution_iff_n_eq_neg_half_l170_170656

theorem system_no_solution_iff_n_eq_neg_half (x y z n : ℝ) :
  (¬ ∃ x y z, 2 * n * x + y = 2 ∧ n * y + 2 * z = 2 ∧ x + 2 * n * z = 2) ↔ n = -1/2 := by
  sorry

end system_no_solution_iff_n_eq_neg_half_l170_170656


namespace cars_meet_time_l170_170206

-- Define the initial conditions as Lean definitions
def distance_car1 (t : ℝ) : ℝ := 15 * t
def distance_car2 (t : ℝ) : ℝ := 20 * t
def total_distance : ℝ := 105

-- Define the proposition we want to prove
theorem cars_meet_time : ∃ (t : ℝ), distance_car1 t + distance_car2 t = total_distance ∧ t = 3 :=
by
  sorry

end cars_meet_time_l170_170206


namespace coefficient_of_x4_in_expansion_l170_170758

noncomputable def problem_statement : ℕ :=
  let n := 8
  let a := 2
  let b := 3
  let k := 4
  binomial n k * (b ^ k) * (a ^ (n - k))

theorem coefficient_of_x4_in_expansion :
  problem_statement = 90720 :=
by
  sorry

end coefficient_of_x4_in_expansion_l170_170758


namespace friedas_probability_to_corner_l170_170825

-- Define the grid size and positions
def grid_size : Nat := 4
def start_position : ℕ × ℕ := (3, 3)
def corner_positions : List (ℕ × ℕ) := [(1, 1), (1, 4), (4, 1), (4, 4)]

-- Define the number of hops allowed
def max_hops : Nat := 4

-- Define a function to calculate the probability of reaching a corner square
-- within the given number of hops starting from the initial position.
noncomputable def prob_reach_corner (grid_size : ℕ) (start_position : ℕ × ℕ) 
                                     (corner_positions : List (ℕ × ℕ)) 
                                     (max_hops : ℕ) : ℚ :=
  -- Implementation details skipped
  sorry

-- Define the main theorem that states the desired probability
theorem friedas_probability_to_corner : 
  prob_reach_corner grid_size start_position corner_positions max_hops = 17 / 64 :=
sorry

end friedas_probability_to_corner_l170_170825


namespace words_added_to_removed_ratio_l170_170057

-- Conditions in the problem
def Yvonnes_words : ℕ := 400
def Jannas_extra_words : ℕ := 150
def words_removed : ℕ := 20
def words_needed : ℕ := 1000 - 930

-- Definitions derived from the conditions
def Jannas_words : ℕ := Yvonnes_words + Jannas_extra_words
def total_words_before_editing : ℕ := Yvonnes_words + Jannas_words
def total_words_after_removal : ℕ := total_words_before_editing - words_removed
def words_added : ℕ := words_needed

-- The theorem we need to prove
theorem words_added_to_removed_ratio :
  (words_added : ℚ) / words_removed = 7 / 2 :=
sorry

end words_added_to_removed_ratio_l170_170057


namespace sum_prime_odd_2009_l170_170537

-- Given a, b ∈ ℕ (natural numbers), where a is prime, b is odd, and a^2 + b = 2009,
-- prove that a + b = 2007.
theorem sum_prime_odd_2009 (a b : ℕ) (h1 : Nat.Prime a) (h2 : Nat.Odd b) (h3 : a^2 + b = 2009) :
  a + b = 2007 := by
  sorry

end sum_prime_odd_2009_l170_170537


namespace thm1_thm2_thm3_thm4_l170_170309

variables {Point Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Definitions relating lines and planes
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p q : Plane) : Prop := sorry
def perpendicular_planes (p q : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Theorem 1: This statement is false, so we negate its for proof.
theorem thm1 (h1 : parallel_line_plane m α) (h2 : parallel_line_plane n β) (h3 : parallel_planes α β) :
  ¬ parallel_lines m n :=
sorry

-- Theorem 2: This statement is true, we need to prove it.
theorem thm2 (h1 : perpendicular_line_plane m α) (h2 : perpendicular_line_plane n β) (h3 : perpendicular_planes α β) :
  perpendicular_lines m n :=
sorry

-- Theorem 3: This statement is true, we need to prove it.
theorem thm3 (h1 : perpendicular_line_plane m α) (h2 : parallel_line_plane n β) (h3 : parallel_planes α β) :
  perpendicular_lines m n :=
sorry

-- Theorem 4: This statement is false, so we negate its for proof.
theorem thm4 (h1 : parallel_line_plane m α) (h2 : perpendicular_line_plane n β) (h3 : perpendicular_planes α β) :
  ¬ parallel_lines m n :=
sorry

end thm1_thm2_thm3_thm4_l170_170309


namespace angle_in_second_quadrant_l170_170675

theorem angle_in_second_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : (2 * Real.tan (α / 2)) / (1 - (Real.tan (α / 2))^2) < 0) : 
  ∃ q, q = 2 ∧ α ∈ {α | 0 < α ∧ α < π} :=
by
  sorry

end angle_in_second_quadrant_l170_170675


namespace tan_alpha_eq_l170_170976

theorem tan_alpha_eq : ∀ (α : ℝ),
  (Real.tan (α - (5 * Real.pi / 4)) = 1 / 5) →
  Real.tan α = 3 / 2 :=
by
  intro α h
  sorry

end tan_alpha_eq_l170_170976


namespace third_group_members_l170_170480

theorem third_group_members (total_members first_group second_group : ℕ) (h₁ : total_members = 70) (h₂ : first_group = 25) (h₃ : second_group = 30) : (total_members - (first_group + second_group)) = 15 :=
sorry

end third_group_members_l170_170480


namespace cyclist_problem_l170_170189

theorem cyclist_problem (MP NP : ℝ) (h1 : NP = MP + 30) (h2 : ∀ t : ℝ, t*MP = 10*t) 
  (h3 : ∀ t : ℝ, t*NP = 10*t) 
  (h4 : ∀ t : ℝ, t*MP = 42 → t*(MP + 30) = t*42 - 1/3) : 
  MP = 180 := 
sorry

end cyclist_problem_l170_170189


namespace problem_statement_l170_170387

theorem problem_statement (x y : ℝ) (log2_3 log5_3 : ℝ)
  (h1 : log2_3 > 1)
  (h2 : 0 < log5_3)
  (h3 : log5_3 < 1)
  (h4 : log2_3^x - log5_3^x ≥ log2_3^(-y) - log5_3^(-y)) :
  x + y ≥ 0 := 
sorry

end problem_statement_l170_170387


namespace ratio_of_inradii_l170_170600

-- Given triangle XYZ with sides XZ=5, YZ=12, XY=13
-- Let W be on XY such that ZW bisects ∠ YZX
-- The inscribed circles of triangles ZWX and ZWY have radii r_x and r_y respectively
-- Prove the ratio r_x / r_y = 1/6

theorem ratio_of_inradii
  (XZ YZ XY : ℝ)
  (W : ℝ)
  (r_x r_y : ℝ)
  (h1 : XZ = 5)
  (h2 : YZ = 12)
  (h3 : XY = 13)
  (h4 : r_x / r_y = 1/6) :
  r_x / r_y = 1/6 :=
by sorry

end ratio_of_inradii_l170_170600


namespace sin_sum_arcsin_arctan_l170_170954

theorem sin_sum_arcsin_arctan :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (1 / 2)
  Real.sin (a + b) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_sum_arcsin_arctan_l170_170954


namespace find_original_revenue_l170_170641

variable (currentRevenue : ℝ) (percentageDecrease : ℝ)
noncomputable def originalRevenue (currentRevenue : ℝ) (percentageDecrease : ℝ) : ℝ :=
  currentRevenue / (1 - percentageDecrease)

theorem find_original_revenue (h1 : currentRevenue = 48.0) (h2 : percentageDecrease = 0.3333333333333333) :
  originalRevenue currentRevenue percentageDecrease = 72.0 := by
  rw [h1, h2]
  unfold originalRevenue
  norm_num
  sorry

end find_original_revenue_l170_170641


namespace harvest_unripe_oranges_l170_170906

theorem harvest_unripe_oranges (R T D U: ℕ) (h1: R = 28) (h2: T = 2080) (h3: D = 26)
  (h4: T = D * (R + U)) :
  U = 52 :=
by
  sorry

end harvest_unripe_oranges_l170_170906


namespace total_emails_received_l170_170069

theorem total_emails_received :
  let e1 := 16
  let e2 := e1 / 2
  let e3 := e2 / 2
  let e4 := e3 / 2
  e1 + e2 + e3 + e4 = 30 :=
by
  sorry

end total_emails_received_l170_170069


namespace range_of_m_l170_170551

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), ¬((x - m) < -3) ∧ (1 + 2*x)/3 ≥ x - 1) ∧ 
  (∀ (x1 x2 x3 : ℤ), 
    (¬((x1 - m) < -3) ∧ (1 + 2 * x1)/3 ≥ x1 - 1) ∧
    (¬((x2 - m) < -3) ∧ (1 + 2 * x2)/3 ≥ x2 - 1) ∧
    (¬((x3 - m) < -3) ∧ (1 + 2 * x3)/3 ≥ x3 - 1)) →
  (4 ≤ m ∧ m < 5) :=
by 
  sorry

end range_of_m_l170_170551


namespace a_plus_b_eq_2007_l170_170536

theorem a_plus_b_eq_2007 (a b : ℕ) (ha : Prime a) (hb : Odd b)
  (h : a^2 + b = 2009) : a + b = 2007 :=
by
  sorry

end a_plus_b_eq_2007_l170_170536


namespace pneumonia_chronic_disease_confidence_l170_170863

noncomputable def K_square :=
  let a := 40
  let b := 20
  let c := 60
  let d := 80
  let n := a + b + c + d
  (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem pneumonia_chronic_disease_confidence :
  K_square > 7.879 := by
  sorry

end pneumonia_chronic_disease_confidence_l170_170863


namespace abc_positive_l170_170365

theorem abc_positive (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) : a > 0 ∧ b > 0 ∧ c > 0 :=
by
  sorry

end abc_positive_l170_170365


namespace rulers_in_drawer_l170_170593

-- conditions
def initial_rulers : ℕ := 46
def additional_rulers : ℕ := 25

-- question: total rulers in the drawer
def total_rulers : ℕ := initial_rulers + additional_rulers

-- proof statement: prove that total_rulers is 71
theorem rulers_in_drawer : total_rulers = 71 := by
  sorry

end rulers_in_drawer_l170_170593


namespace an_squared_diff_consec_cubes_l170_170306

theorem an_squared_diff_consec_cubes (a b : ℕ → ℤ) (n : ℕ) :
  a 1 = 1 → b 1 = 0 →
  (∀ n ≥ 1, a (n + 1) = 7 * (a n) + 12 * (b n) + 6) →
  (∀ n ≥ 1, b (n + 1) = 4 * (a n) + 7 * (b n) + 3) →
  a n ^ 2 = (b n + 1) ^ 3 - (b n) ^ 3 :=
by
  sorry

end an_squared_diff_consec_cubes_l170_170306


namespace sum_and_num_of_factors_eq_1767_l170_170670

theorem sum_and_num_of_factors_eq_1767 (n : ℕ) (σ d : ℕ → ℕ) :
  (σ n + d n = 1767) → 
  ∃ m : ℕ, σ m + d m = 1767 :=
by 
  sorry

end sum_and_num_of_factors_eq_1767_l170_170670


namespace least_five_digit_congruent_to_8_mod_17_l170_170050

theorem least_five_digit_congruent_to_8_mod_17 : 
  ∃ n : ℕ, n = 10004 ∧ n % 17 = 8 ∧ 10000 ≤ n ∧ n < 100000 ∧ (∀ m, 10000 ≤ m ∧ m < 10004 → m % 17 ≠ 8) :=
sorry

end least_five_digit_congruent_to_8_mod_17_l170_170050


namespace inequality_proof_l170_170725

variables (x y z : ℝ)

theorem inequality_proof (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  2 ≤ (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ∧ (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ≤ (1 + x) * (1 + y) * (1 + z) :=
by
  sorry

end inequality_proof_l170_170725


namespace base_of_triangle_is_24_l170_170905

def triangle_sides_sum := 50
def left_side : ℕ := 12
def right_side := left_side + 2
def base := triangle_sides_sum - left_side - right_side

theorem base_of_triangle_is_24 :
  base = 24 :=
by 
  have h : left_side = 12 := rfl
  have h2 : right_side = 14 := by simp [right_side, h]
  have h3 : base = 24 := by simp [base, triangle_sides_sum, h, h2]
  exact h3

end base_of_triangle_is_24_l170_170905


namespace urn_problem_l170_170220

noncomputable def count_balls (initial_white : ℕ) (initial_black : ℕ) (operations : ℕ) : ℕ :=
initial_white + initial_black + operations

noncomputable def urn_probability (initial_white : ℕ) (initial_black : ℕ) (operations : ℕ) (final_white : ℕ) (final_black : ℕ) : ℚ :=
if final_white + final_black = count_balls initial_white initial_black operations &&
   final_white = (initial_white + (operations - (final_black - initial_black))) &&
   (final_white + final_black) = 8 then 3 / 5 else 0

theorem urn_problem :
  let initial_white := 2
  let initial_black := 1
  let operations := 4
  let final_white := 4
  let final_black := 4
  count_balls initial_white initial_black operations = 8 ∧ urn_probability initial_white initial_black operations final_white final_black = 3 / 5 :=
by
  sorry

end urn_problem_l170_170220


namespace unique_solution_quadratic_l170_170518

theorem unique_solution_quadratic (q : ℝ) (hq : q ≠ 0) :
  (∃ x, q * x^2 - 18 * x + 8 = 0 ∧ ∀ y, q * y^2 - 18 * y + 8 = 0 → y = x) →
  q = 81 / 8 :=
by
  sorry

end unique_solution_quadratic_l170_170518


namespace percentage_increase_is_50_l170_170078

def papaya_growth (P : ℝ) : Prop :=
  let growth1 := 2
  let growth2 := 2 * (1 + P / 100)
  let growth3 := 1.5 * growth2
  let growth4 := 2 * growth3
  let growth5 := 0.5 * growth4
  growth1 + growth2 + growth3 + growth4 + growth5 = 23

theorem percentage_increase_is_50 :
  ∃ (P : ℝ), papaya_growth P ∧ P = 50 := by
  sorry

end percentage_increase_is_50_l170_170078


namespace part1_part2_l170_170140

theorem part1 (a b c C : ℝ) (h : b - 1/2 * c = a * Real.cos C) (h1 : ∃ (A B : ℝ), Real.sin B - 1/2 * Real.sin C = Real.sin A * Real.cos C) :
  ∃ A : ℝ, A = 60 :=
sorry

theorem part2 (a b c : ℝ) (h1 : 4 * (b + c) = 3 * b * c) (h2 : a = 2 * Real.sqrt 3) (h3 : b - 1/2 * c = a * Real.cos 60)
  (h4 : ∀ (A : ℝ), A = 60) : ∃ S : ℝ, S = 2 * Real.sqrt 3 :=
sorry

end part1_part2_l170_170140


namespace sequence_behavior_l170_170425

theorem sequence_behavior (b : ℕ → ℕ) :
  (∀ n, b n = n) ∨ ∃ N, ∀ n, n ≥ N → b n = b N :=
sorry

end sequence_behavior_l170_170425


namespace lcm_of_40_90_150_l170_170110

-- Definition to calculate the Least Common Multiple of three numbers
def lcm3 (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

-- Definitions for the given numbers
def n1 : ℕ := 40
def n2 : ℕ := 90
def n3 : ℕ := 150

-- The statement of the proof problem
theorem lcm_of_40_90_150 : lcm3 n1 n2 n3 = 1800 := by
  sorry

end lcm_of_40_90_150_l170_170110


namespace num_five_letter_words_correct_l170_170788

noncomputable def num_five_letter_words : ℕ := 1889568

theorem num_five_letter_words_correct :
  let a := 3
  let e := 4
  let i := 2
  let o := 5
  let u := 4
  (a + e + i + o + u) ^ 5 = num_five_letter_words :=
by
  sorry

end num_five_letter_words_correct_l170_170788


namespace multiplication_vs_subtraction_difference_l170_170202

variable (x : ℕ)
variable (h : x = 10)

theorem multiplication_vs_subtraction_difference :
  3 * x - (26 - x) = 14 := by
  sorry

end multiplication_vs_subtraction_difference_l170_170202


namespace sequence_product_l170_170669

-- Definitions for the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + (n - 1) * d

-- Definitions for the geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b1 r : ℝ), ∀ n, b n = b1 * r ^ (n - 1)

-- Defining the main proposition
theorem sequence_product (a b : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_geom  : is_geometric_sequence b)
  (h_eq    : b 7 = a 7)
  (h_cond  : 2 * a 2 - (a 7) ^ 2 + 2 * a 12 = 0) :
  b 3 * b 11 = 16 :=
sorry

end sequence_product_l170_170669


namespace count_of_plus_signs_l170_170034

-- Definitions based on the given conditions
def total_symbols : ℕ := 23
def any_10_symbols_contains_plus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 10 → (\exists c ∈ s, c = '+')
def any_15_symbols_contains_minus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 15 → (\exists c ∈ s, c = '-')

-- Main theorem statement without the proof
theorem count_of_plus_signs (symbols : Finset char) (h1 : symbols.card = total_symbols) 
  (h2 : any_10_symbols_contains_plus symbols) 
  (h3 : any_15_symbols_contains_minus symbols) : 
  symbols.count '+' = 14 :=
sorry

end count_of_plus_signs_l170_170034


namespace initial_birds_count_l170_170038

theorem initial_birds_count (B : ℕ) :
  ∃ B, B + 4 = 5 + 2 → B = 3 :=
by
  sorry

end initial_birds_count_l170_170038


namespace candy_weight_probability_l170_170631

open Probability
open MeasureTheory 

theorem candy_weight_probability (X : MeasureTheory.Measure ξ) [isNormal_var : IsNormal X 500 σ] (p : ℝ) 
  (condition : P (|X - 500| > 5) = p) : 
  P (495 ≤ X ∧ X ≤ 500) = (1 - p) / 2 := 
sorry

end candy_weight_probability_l170_170631


namespace minimum_button_presses_to_exit_l170_170914

def arms_after (r y : ℕ) : ℕ := 3 + r - 2 * y
def doors_after (y g : ℕ) : ℕ := 3 + y - 2 * g

theorem minimum_button_presses_to_exit :
  ∃ r y g : ℕ, arms_after r y = 0 ∧ doors_after y g = 0 ∧ r + y + g = 9 :=
sorry

end minimum_button_presses_to_exit_l170_170914


namespace square_and_product_l170_170126

theorem square_and_product (x : ℤ) (h : x^2 = 1764) : (x = 42) ∧ ((x + 2) * (x - 2) = 1760) :=
by
  sorry

end square_and_product_l170_170126


namespace minimum_value_l170_170130

theorem minimum_value (x : ℝ) (hx : 0 < x) : ∃ y, (y = x + 4 / (x + 1)) ∧ (∀ z, (x > 0 → z = x + 4 / (x + 1)) → 3 ≤ z) := sorry

end minimum_value_l170_170130


namespace vasya_most_points_anya_least_possible_l170_170097

theorem vasya_most_points_anya_least_possible :
  ∃ (A B V : ℕ) (A_score B_score V_score : ℕ),
  A > B ∧ B > V ∧
  A_score = 9 ∧ B_score = 10 ∧ V_score = 11 ∧
  (∃ (words_common_AB words_common_AV words_only_B words_only_V : ℕ),
  words_common_AB = 6 ∧ words_common_AV = 3 ∧ words_only_B = 2 ∧ words_only_V = 4 ∧
  A = words_common_AB + words_common_AV ∧
  B = words_only_B + words_common_AB ∧
  V = words_only_V + words_common_AV ∧
  A_score = words_common_AB + words_common_AV ∧
  B_score = 2 * words_only_B + words_common_AB ∧
  V_score = 2 * words_only_V + words_common_AV) :=
sorry

end vasya_most_points_anya_least_possible_l170_170097


namespace solve_system_of_equations_l170_170886

theorem solve_system_of_equations
  (a b c d x y z u : ℝ)
  (h1 : a^3 * x + a^2 * y + a * z + u = 0)
  (h2 : b^3 * x + b^2 * y + b * z + u = 0)
  (h3 : c^3 * x + c^2 * y + c * z + u = 0)
  (h4 : d^3 * x + d^2 * y + d * z + u = 1) :
  x = 1 / ((d - a) * (d - b) * (d - c)) ∧
  y = -(a + b + c) / ((d - a) * (d - b) * (d - c)) ∧
  z = (a * b + b * c + c * a) / ((d - a) * (d - b) * (d - c)) ∧
  u = - (a * b * c) / ((d - a) * (d - b) * (d - c)) :=
sorry

end solve_system_of_equations_l170_170886


namespace proof_problem_l170_170695

-- Definitions of sequence terms and their properties
def geometric_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ a 4 = 16 ∧ ∀ n, a n = 2^n

-- Definition for the sum of the first n terms of the sequence
noncomputable def sum_of_sequence (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = 2^(n + 1) - 2

-- Definition for the transformed sequence b_n = log_2 a_n
def transformed_sequence (a b : ℕ → ℕ) : Prop :=
  ∀ n, b n = Nat.log2 (a n)

-- Definition for the sum T_n related to b_n
noncomputable def sum_of_transformed_sequence (T : ℕ → ℚ) (b : ℕ → ℕ) : Prop :=
  ∀ n, T n = 1 - 1 / (n + 1)

theorem proof_problem :
  (∃ a : ℕ → ℕ, geometric_sequence a) ∧
  (∃ S : ℕ → ℕ, sum_of_sequence S) ∧
  (∃ (a b : ℕ → ℕ), geometric_sequence a ∧ transformed_sequence a b ∧
   (∃ T : ℕ → ℚ, sum_of_transformed_sequence T b)) :=
by {
  -- Definitions and proofs will go here
  sorry
}

end proof_problem_l170_170695


namespace gordon_total_cost_l170_170329

noncomputable def DiscountA (price : ℝ) : ℝ :=
if price > 22.00 then price * 0.70 else price

noncomputable def DiscountB (price : ℝ) : ℝ :=
if 10.00 < price ∧ price <= 20.00 then price * 0.80 else price

noncomputable def DiscountC (price : ℝ) : ℝ :=
if price < 10.00 then price * 0.85 else price

noncomputable def apply_discount (price : ℝ) : ℝ :=
if price > 22.00 then DiscountA price
else if price > 10.00 then DiscountB price
else DiscountC price

noncomputable def total_price (prices : List ℝ) : ℝ :=
(prices.map apply_discount).sum

noncomputable def total_with_tax_and_fee (prices : List ℝ) (tax_rate extra_fee : ℝ) : ℝ :=
let total := total_price prices
let tax := total * tax_rate
total + tax + extra_fee

theorem gordon_total_cost :
  total_with_tax_and_fee
    [25.00, 18.00, 21.00, 35.00, 12.00, 10.00, 8.50, 23.00, 6.00, 15.50, 30.00, 9.50]
    0.05 2.00
  = 171.27 :=
  sorry

end gordon_total_cost_l170_170329


namespace find_value_of_a_l170_170042

theorem find_value_of_a (a : ℝ) :
  (∀ x y : ℝ, (ax + y - 1 = 0) → (x - y + 3 = 0) → (-a) * 1 = -1) → a = 1 :=
by
  sorry

end find_value_of_a_l170_170042


namespace clothing_probability_l170_170995

/-- I have a drawer with 6 shirts, 8 pairs of shorts, 7 pairs of socks, and 3 jackets in it.
    If I reach in and randomly remove four articles of clothing, what is the probability that 
    I get one shirt, one pair of shorts, one pair of socks, and one jacket? -/
theorem clothing_probability :
  let total_articles := 6 + 8 + 7 + 3
  let total_combinations := Nat.choose total_articles 4
  let favorable_combinations := 6 * 8 * 7 * 3
  (favorable_combinations : ℚ) / total_combinations = 144 / 1815 :=
by
  let total_articles := 6 + 8 + 7 + 3
  let total_combinations := Nat.choose total_articles 4
  let favorable_combinations := 6 * 8 * 7 * 3
  suffices (favorable_combinations : ℚ) / total_combinations = 144 / 1815
  by
    sorry
  sorry

end clothing_probability_l170_170995


namespace simplify_fraction_l170_170432

/-
  Given the conditions that \(i^2 = -1\),
  prove that \(\displaystyle\frac{2-i}{1+4i} = -\frac{2}{17} - \frac{9}{17}i\).
-/
theorem simplify_fraction : 
  let i : ℂ := ⟨0, 1⟩ in
  i^2 = -1 → (2 - i) / (1 + 4 * i) = - (2 / 17) - (9 / 17) * i :=
by
  intro h
  sorry

end simplify_fraction_l170_170432


namespace relationship_abc_d_l170_170827

theorem relationship_abc_d : 
  ∀ (a b c d : ℝ), 
  a < b → 
  d < c → 
  (c - a) * (c - b) < 0 → 
  (d - a) * (d - b) > 0 → 
  d < a ∧ a < c ∧ c < b :=
by
  intros a b c d a_lt_b d_lt_c h1 h2
  sorry

end relationship_abc_d_l170_170827


namespace composite_square_perimeter_l170_170705

theorem composite_square_perimeter (p1 p2 : ℝ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  let s1 := p1 / 4
  let s2 := p2 / 4
  (p1 + p2 - 2 * s1) = 120 := 
by
  -- proof goes here
  sorry

end composite_square_perimeter_l170_170705


namespace compute_expression_l170_170508

theorem compute_expression : 3 * 3^3 - 9^50 / 9^48 = 0 := by
  sorry

end compute_expression_l170_170508


namespace inverse_exists_l170_170358

noncomputable def f (x : ℝ) : ℝ := 7 * x^3 - 2 * x^2 + 5 * x - 9

theorem inverse_exists :
  ∃ x : ℝ, 7 * x^3 - 2 * x^2 + 5 * x - 5.5 = 0 :=
sorry

end inverse_exists_l170_170358


namespace Douglas_weight_correct_l170_170498

def Anne_weight : ℕ := 67
def weight_diff : ℕ := 15
def Douglas_weight : ℕ := 52

theorem Douglas_weight_correct : Douglas_weight = Anne_weight - weight_diff := by
  sorry

end Douglas_weight_correct_l170_170498


namespace max_value_of_sinx_over_2_minus_cosx_l170_170566

theorem max_value_of_sinx_over_2_minus_cosx (x : ℝ) : 
  ∃ y_max, y_max = (Real.sqrt 3) / 3 ∧ ∀ y, y = (Real.sin x) / (2 - Real.cos x) → y ≤ y_max :=
sorry

end max_value_of_sinx_over_2_minus_cosx_l170_170566


namespace area_of_region_l170_170449

theorem area_of_region : 
  ∀ (x y : ℝ), 
  (x^2 + y^2 + 6*x - 8*y = 16) → 
  (π * 41) = (π * 41) :=
by
  sorry

end area_of_region_l170_170449


namespace inequality_solution_l170_170241

theorem inequality_solution (x : ℝ) : 
  x^3 - 10 * x^2 + 28 * x > 0 ↔ (0 < x ∧ x < 4) ∨ (6 < x)
:= sorry

end inequality_solution_l170_170241


namespace modulus_of_complex_l170_170979

-- Some necessary imports for complex numbers and proofs in Lean
open Complex

theorem modulus_of_complex (x y : ℝ) (h : (1 + I) * x = 1 + y * I) : abs (x + y * I) = Real.sqrt 2 :=
by
  sorry

end modulus_of_complex_l170_170979


namespace max_perimeter_of_rectangle_with_area_36_l170_170081

theorem max_perimeter_of_rectangle_with_area_36 :
  ∃ l w : ℕ, l * w = 36 ∧ (∀ l' w' : ℕ, l' * w' = 36 → 2 * (l + w) ≥ 2 * (l' + w')) ∧ 2 * (l + w) = 74 := 
sorry

end max_perimeter_of_rectangle_with_area_36_l170_170081


namespace sequence_formula_l170_170532

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, a n = 2^n - 1 :=
by
  sorry

end sequence_formula_l170_170532


namespace magnitude_of_two_a_minus_b_l170_170994

namespace VectorMagnitude

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (3, -2)

-- Function to calculate the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Vector operation 2a - b
def two_a_minus_b : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

-- The theorem to prove
theorem magnitude_of_two_a_minus_b : magnitude two_a_minus_b = Real.sqrt 17 := by
  sorry

end VectorMagnitude

end magnitude_of_two_a_minus_b_l170_170994


namespace bamboo_volume_l170_170861

theorem bamboo_volume :
  ∃ (a₁ d a₅ : ℚ), 
  (4 * a₁ + 6 * d = 5) ∧ 
  (3 * a₁ + 21 * d = 4) ∧ 
  (a₅ = a₁ + 4 * d) ∧ 
  (a₅ = 85 / 66) :=
sorry

end bamboo_volume_l170_170861


namespace region_area_l170_170658

noncomputable def area_of_region := 
  let a := 0
  let b := Real.sqrt 2 / 2
  ∫ x in a..b, (Real.arccos x) - (Real.arcsin x)

theorem region_area : area_of_region = 2 - Real.sqrt 2 :=
by
  sorry

end region_area_l170_170658


namespace negation_of_statement_6_l170_170988

variable (Teenager Adult : Type)
variable (CanCookWell : Teenager → Prop)
variable (CanCookWell' : Adult → Prop)

-- Conditions from the problem
def all_teenagers_can_cook_well : Prop :=
  ∀ t : Teenager, CanCookWell t

def some_teenagers_can_cook_well : Prop :=
  ∃ t : Teenager, CanCookWell t

def no_adults_can_cook_well : Prop :=
  ∀ a : Adult, ¬CanCookWell' a

def all_adults_cannot_cook_well : Prop :=
  ∀ a : Adult, ¬CanCookWell' a

def at_least_one_adult_cannot_cook_well : Prop :=
  ∃ a : Adult, ¬CanCookWell' a

def all_adults_can_cook_well : Prop :=
  ∀ a : Adult, CanCookWell' a

-- Theorem to prove
theorem negation_of_statement_6 :
  at_least_one_adult_cannot_cook_well Adult CanCookWell' = ¬ all_adults_can_cook_well Adult CanCookWell' :=
sorry

end negation_of_statement_6_l170_170988


namespace find_b_l170_170347

-- Definitions for conditions
def eq1 (a : ℤ) : Prop := 2 * a + 1 = 1
def eq2 (a b : ℤ) : Prop := 2 * b - 3 * a = 2

-- The theorem statement
theorem find_b (a b : ℤ) (h1 : eq1 a) (h2 : eq2 a b) : b = 1 :=
  sorry  -- Proof to be filled in.

end find_b_l170_170347


namespace probability_same_color_is_correct_l170_170274

noncomputable def total_ways_select_plates : ℕ := Nat.choose 11 3
def ways_select_red_plates : ℕ := Nat.choose 6 3
def ways_select_blue_plates : ℕ := Nat.choose 5 3
noncomputable def favorable_outcomes : ℕ := ways_select_red_plates + ways_select_blue_plates
noncomputable def probability_same_color : ℚ := favorable_outcomes / total_ways_select_plates

theorem probability_same_color_is_correct :
  probability_same_color = 2/11 := 
by
  sorry

end probability_same_color_is_correct_l170_170274


namespace find_a_l170_170400

theorem find_a
  (x y a : ℝ)
  (h1 : x + y = 1)
  (h2 : 2 * x + y = 0)
  (h3 : a * x - 3 * y = 0) :
  a = -6 :=
sorry

end find_a_l170_170400


namespace focus_of_parabola_l170_170106

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the coordinates of the focus
def is_focus (x y : ℝ) : Prop := (x = 0) ∧ (y = 1)

-- The theorem statement
theorem focus_of_parabola : 
  (∃ x y : ℝ, parabola x y ∧ is_focus x y) :=
sorry

end focus_of_parabola_l170_170106


namespace overtaking_time_l170_170205

variable (a_speed b_speed k_speed : ℕ)
variable (b_delay : ℕ) 
variable (t : ℕ)
variable (t_k : ℕ)

theorem overtaking_time (h1 : a_speed = 30)
                        (h2 : b_speed = 40)
                        (h3 : k_speed = 60)
                        (h4 : b_delay = 5)
                        (h5 : 30 * t = 40 * (t - 5))
                        (h6 : 30 * t = 60 * t_k)
                         : k_speed / 3 = 10 :=
by sorry

end overtaking_time_l170_170205


namespace product_of_all_positive_divisors_of_18_l170_170456

def product_divisors_18 : ℕ :=
  ∏ d in (Multiset.to_finset ([1, 2, 3, 6, 9, 18] : Multiset ℕ)), d

theorem product_of_all_positive_divisors_of_18 : product_divisors_18 = 5832 := by
  sorry

end product_of_all_positive_divisors_of_18_l170_170456


namespace same_color_probability_l170_170268

open Nat

theorem same_color_probability :
  let total_plates := 11
  let red_plates := 6
  let blue_plates := 5
  let chosen_plates := 3
  let total_ways := choose total_plates chosen_plates
  let red_ways := choose red_plates chosen_plates
  let blue_ways := choose blue_plates chosen_plates
  let same_color_ways := red_ways + blue_ways
  let probability := (same_color_ways : ℚ) / (total_ways : ℚ)
  probability = 2 / 11 := by sorry

end same_color_probability_l170_170268


namespace num_balls_in_box_l170_170146

theorem num_balls_in_box (n : ℕ) (h1: 9 <= n) (h2: (9 : ℝ) / n = 0.30) : n = 30 :=
sorry

end num_balls_in_box_l170_170146


namespace ellipse_area_l170_170689

/-- 
In a certain ellipse, the endpoints of the major axis are (1, 6) and (21, 6). 
Also, the ellipse passes through the point (19, 9). Prove that the area of the ellipse is 50π. 
-/
theorem ellipse_area : 
  let a := 10
  let b := 5 
  let center := (11, 6)
  let endpoints_major := [(1, 6), (21, 6)]
  let point_on_ellipse := (19, 9)
  ∀ x y, ((x - 11)^2 / a^2) + ((y - 6)^2 / b^2) = 1 → 
    (x, y) = (19, 9) →  -- given point on the ellipse
    (endpoints_major = [(1, 6), (21, 6)]) →  -- given endpoints of the major axis
    50 * Real.pi = π * a * b := 
by
  sorry

end ellipse_area_l170_170689


namespace triplet_unique_solution_l170_170355

theorem triplet_unique_solution {x y z : ℝ} :
  x^2 - 2*x - 4*z = 3 →
  y^2 - 2*y - 2*x = -14 →
  z^2 - 4*y - 4*z = -18 →
  (x = 2 ∧ y = 3 ∧ z = 4) :=
by
  sorry

end triplet_unique_solution_l170_170355


namespace part1_part2_l170_170379

variables (q x : ℝ)
def f (x : ℝ) (q : ℝ) : ℝ := x^2 - 16*x + q + 3
def g (x : ℝ) (q : ℝ) : ℝ := f x q + 51

theorem part1 (h1 : ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x q = 0):
  (-20 : ℝ) ≤ q ∧ q ≤ 12 := 
  sorry

theorem part2 (h2 : ∀ x ∈ Set.Icc (q : ℝ) 10, g x q ≥ 0) : 
  9 ≤ q ∧ q < 10 := 
  sorry

end part1_part2_l170_170379


namespace find_four_digit_number_l170_170045

def is_four_digit_number (k : ℕ) : Prop :=
  1000 ≤ k ∧ k < 10000

def appended_number (k : ℕ) : ℕ :=
  4000000 + k

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem find_four_digit_number (k : ℕ) (hk : is_four_digit_number k) :
  is_perfect_square (appended_number k) ↔ k = 4001 ∨ k = 8004 :=
sorry

end find_four_digit_number_l170_170045


namespace solve_for_x_l170_170655

theorem solve_for_x : 
  ∀ x : ℚ, x + 5/6 = 7/18 - 2/9 → x = -2/3 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l170_170655


namespace incorrect_statement_l170_170127

noncomputable def a : ℝ × ℝ := (1, -2)
noncomputable def b : ℝ × ℝ := (2, 1)
noncomputable def c : ℝ × ℝ := (-4, -2)

-- Define the incorrect vector statement D
theorem incorrect_statement :
  ¬ ∀ (d : ℝ × ℝ), ∃ (k1 k2 : ℝ), d = (k1 * b.1 + k2 * c.1, k1 * b.2 + k2 * c.2) := sorry

end incorrect_statement_l170_170127


namespace min_b1_b2_sum_l170_170591

def sequence_relation (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (3 * b n + 4073) / (2 + b (n + 1))

theorem min_b1_b2_sum (b : ℕ → ℕ) (h_seq : sequence_relation b) 
  (h_b1_pos : b 1 > 0) (h_b2_pos : b 2 > 0) :
  b 1 + b 2 = 158 :=
sorry

end min_b1_b2_sum_l170_170591


namespace smallest_of_2_and_4_smallest_combined_of_2_and_4_l170_170607

def smallest_number_of_two (a b : ℕ) : ℕ :=
  if a <= b then a else b

theorem smallest_of_2_and_4 : smallest_number_of_two 2 4 = 2 :=
  by simp [smallest_number_of_two]

def form_combined_number (a b : ℕ) : ℕ :=
  10 * a + b

theorem smallest_combined_of_2_and_4 : form_combined_number 2 4 = 24 :=
  by simp [form_combined_number]

end smallest_of_2_and_4_smallest_combined_of_2_and_4_l170_170607


namespace bisecting_line_eq_l170_170654

theorem bisecting_line_eq : ∃ (a : ℝ), (∀ x y : ℝ, (y = a * x) ↔ y = -1 / 6 * x) ∧ 
  (∀ p : ℝ × ℝ, (3 * p.1 - 5 * p.2  = 6 → p.2 = a * p.1) ∧ 
                  (4 * p.1 + p.2 + 6 = 0 → p.2 = a * p.1)) :=
by
  use -1 / 6
  sorry

end bisecting_line_eq_l170_170654


namespace alien_takes_home_l170_170093

variable (abducted : ℕ) (returned_percentage : ℚ) (taken_to_another_planet : ℕ)

-- Conditions
def initial_abducted_people : abducted = 200 := rfl
def percentage_returned_people : returned_percentage = 0.8 := rfl
def people_taken_to_another_planet : taken_to_another_planet = 10 := rfl

-- The question to prove
def people_taken_home (abducted : ℕ) (returned_percentage : ℚ) (taken_to_another_planet : ℕ) : ℕ :=
  let returned := (returned_percentage * abducted) in
  let remaining := abducted - returned in
  remaining - taken_to_another_planet

theorem alien_takes_home :
  people_taken_home 200 0.8 10 = 30 :=
by
  -- calculations directly in Lean or use sorry to represent the correctness
  sorry

end alien_takes_home_l170_170093


namespace polygon_sides_l170_170588

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360) : 
  n = 8 := 
sorry

end polygon_sides_l170_170588


namespace resulting_perimeter_l170_170704

theorem resulting_perimeter (p1 p2 : ℕ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  let a := p1 / 4 in
  let b := p2 / 4 in
  p1 + p2 - 2 * a = 120 :=
by
  sorry

end resulting_perimeter_l170_170704


namespace find_directrix_l170_170380

-- Define the parabola equation
def parabola_eq (x y : ℝ) : Prop := x^2 = 8 * y

-- State the problem to find the directrix of the given parabola
theorem find_directrix (x y : ℝ) (h : parabola_eq x y) : y = -2 :=
sorry

end find_directrix_l170_170380


namespace functional_equation_solution_l170_170622

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (x + f y) + f (y + f x) = 2 * f (x * f y)) → (∀ x : ℝ, f x = 0) :=
by
  intro f h
  sorry

end functional_equation_solution_l170_170622


namespace probability_of_one_or_two_l170_170797

/-- Represents the number of elements in the first 20 rows of Pascal's Triangle. -/
noncomputable def total_elements : ℕ := 210

/-- Represents the number of ones in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_ones : ℕ := 39

/-- Represents the number of twos in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_twos : ℕ :=18

/-- Prove that the probability of randomly choosing an element which is either 1 or 2
from the first 20 rows of Pascal's Triangle is 57/210. -/
theorem probability_of_one_or_two (h1 : total_elements = 210)
                                  (h2 : number_of_ones = 39)
                                  (h3 : number_of_twos = 18) :
    39 + 18 = 57 ∧ (57 : ℚ) / 210 = 57 / 210 :=
by {
    sorry
}

end probability_of_one_or_two_l170_170797


namespace number_of_correct_statements_l170_170443

theorem number_of_correct_statements (stmt1: Prop) (stmt2: Prop) (stmt3: Prop) :
  stmt1 ∧ stmt2 ∧ stmt3 → (∀ n, n = 3) :=
by
  sorry

end number_of_correct_statements_l170_170443


namespace accounting_major_students_count_l170_170618

theorem accounting_major_students_count (p q r s: ℕ) (h1: p * q * r * s = 1365) (h2: 1 < p) (h3: p < q) (h4: q < r) (h5: r < s):
  p = 3 :=
sorry

end accounting_major_students_count_l170_170618


namespace proof_problem_l170_170842

theorem proof_problem (x : ℝ) (h : x < 1) : -2 * x + 2 > 0 :=
by
  sorry

end proof_problem_l170_170842


namespace import_tax_amount_in_excess_l170_170484

theorem import_tax_amount_in_excess (X : ℝ) 
  (h1 : 0.07 * (2590 - X) = 111.30) : 
  X = 1000 :=
by
  sorry

end import_tax_amount_in_excess_l170_170484


namespace correct_propositions_l170_170535

-- Definitions for the propositions
def prop1 (a M b : Prop) : Prop := (a ∧ M) ∧ (b ∧ M) → a ∧ b
def prop2 (a M b : Prop) : Prop := (a ∧ M) ∧ (b ∧ ¬M) → a ∧ ¬b
def prop3 (a b M : Prop) : Prop := (a ∧ b) ∧ (b ∧ M) → a ∧ M
def prop4 (a M N : Prop) : Prop := (a ∧ ¬M) ∧ (a ∧ N) → ¬M ∧ N

-- Proof problem statement
theorem correct_propositions : 
  ∀ (a b M N : Prop), 
    (prop1 a M b = true) ∨ (prop1 a M b = false) ∧ 
    (prop2 a M b = true) ∨ (prop2 a M b = false) ∧ 
    (prop3 a b M = true) ∨ (prop3 a b M = false) ∧ 
    (prop4 a M N = true) ∨ (prop4 a M N = false) → 
    3 = 3 :=
by
  sorry

end correct_propositions_l170_170535


namespace infinite_points_in_region_l170_170350

theorem infinite_points_in_region : 
  ∀ x y : ℚ, 0 < x → 0 < y → x + 2 * y ≤ 6 → ¬(∃ n : ℕ, ∀ x y : ℚ, 0 < x → 0 < y → x + 2 * y ≤ 6 → sorry) :=
sorry

end infinite_points_in_region_l170_170350


namespace same_color_probability_l170_170271

open Nat

theorem same_color_probability :
  let total_plates := 11
  let red_plates := 6
  let blue_plates := 5
  let chosen_plates := 3
  let total_ways := choose total_plates chosen_plates
  let red_ways := choose red_plates chosen_plates
  let blue_ways := choose blue_plates chosen_plates
  let same_color_ways := red_ways + blue_ways
  let probability := (same_color_ways : ℚ) / (total_ways : ℚ)
  probability = 2 / 11 := by sorry

end same_color_probability_l170_170271


namespace find_length_of_GH_l170_170852

variable {A B C F G H : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
          [MetricSpace F] [MetricSpace G] [MetricSpace H]

variables (AB BC Res : ℝ)
variables (ratio1 ratio2 : ℝ)
variable (similar : SimilarTriangles A B C F G H)

def length_of_GH (GH : ℝ) : Prop :=
  GH = 15

theorem find_length_of_GH (h1 : AB = 15) (h2 : BC = 25) (h3 : ratio1 = 5) (h4 : ratio2 = 3)
  (h5 : similar) : ∃ GH, length_of_GH GH :=
by
  have ratio : ratio2 / ratio1 = 3 / 5 := by assumption
  sorry

end find_length_of_GH_l170_170852


namespace plus_signs_count_l170_170027

theorem plus_signs_count (total_symbols : ℕ) (n : ℕ) (m : ℕ)
    (h1 : total_symbols = 23)
    (h2 : ∀ (s : Finset ℕ), s.card = 10 → (∃ x ∈ s, x = n))
    (h3 : ∀ (s : Finset ℕ), s.card = 15 → (∃ x ∈ s, x = m)) :
    n = 14 := by
  sorry

end plus_signs_count_l170_170027


namespace average_exp_Feb_to_Jul_l170_170218

theorem average_exp_Feb_to_Jul (x y z : ℝ) 
    (h1 : 1200 + x + 0.85 * x + z + 1.10 * z + 0.90 * (1.10 * z) = 6 * 4200) 
    (h2 : 0 ≤ x) 
    (h3 : 0 ≤ z) : 
    (x + 0.85 * x + z + 1.10 * z + 0.90 * (1.10 * z) + 1500) / 6 = 4250 :=
by
    sorry

end average_exp_Feb_to_Jul_l170_170218


namespace reynald_volleyballs_l170_170723

def total_balls : ℕ := 145
def soccer_balls : ℕ := 20
def basketballs : ℕ := soccer_balls + 5
def tennis_balls : ℕ := 2 * soccer_balls
def baseballs : ℕ := soccer_balls + 10
def volleyballs : ℕ := total_balls - (soccer_balls + basketballs + tennis_balls + baseballs)

theorem reynald_volleyballs : volleyballs = 30 :=
by
  sorry

end reynald_volleyballs_l170_170723


namespace roots_of_quadratic_equation_l170_170841

theorem roots_of_quadratic_equation (a b c r s : ℝ) 
  (hr : a ≠ 0)
  (h : a * r^2 + b * r - c = 0)
  (h' : a * s^2 + b * s - c = 0)
  :
  (1 / r^2) + (1 / s^2) = (b^2 + 2 * a * c) / c^2 :=
by
  sorry

end roots_of_quadratic_equation_l170_170841


namespace median_number_of_children_l170_170581

-- Define the given conditions
def number_of_data_points : Nat := 13
def median_position : Nat := (number_of_data_points + 1) / 2

-- We assert the median value based on information given in the problem
def median_value : Nat := 4

-- Statement to prove the problem
theorem median_number_of_children (h1: median_position = 7) (h2: median_value = 4) : median_value = 4 := 
by
  sorry

end median_number_of_children_l170_170581


namespace sum_digits_2_2005_times_5_2007_times_3_l170_170770

-- Define a function to calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem sum_digits_2_2005_times_5_2007_times_3 : 
  sum_of_digits (2^2005 * 5^2007 * 3) = 12 := 
by 
  sorry

end sum_digits_2_2005_times_5_2007_times_3_l170_170770


namespace pipe_length_difference_l170_170473

theorem pipe_length_difference (total_length shorter_piece : ℕ) (h1 : total_length = 68) (h2 : shorter_piece = 28) : 
  total_length - shorter_piece * 2 = 12 := 
sorry

end pipe_length_difference_l170_170473


namespace plus_signs_count_l170_170005

theorem plus_signs_count (total_symbols : ℕ) (n_plus n_minus : ℕ)
  (h1 : total_symbols = 23)
  (h2 : ∀ s : Finset ℕ, s.card = 10 → ∃ i ∈ s, i ≤ n_plus)
  (h3 : ∀ s : Finset ℕ, s.card = 15 → ∃ i ∈ s, i > n_plus) :
  n_plus = 14 :=
by
  -- the proof will go here
  sorry

end plus_signs_count_l170_170005


namespace fraction_of_state_quarters_is_two_fifths_l170_170572

variable (total_quarters state_quarters : ℕ)
variable (is_pennsylvania_percentage : ℚ)
variable (pennsylvania_state_quarters : ℕ)

theorem fraction_of_state_quarters_is_two_fifths
  (h1 : total_quarters = 35)
  (h2 : pennsylvania_state_quarters = 7)
  (h3 : is_pennsylvania_percentage = 1 / 2)
  (h4 : state_quarters = 2 * pennsylvania_state_quarters)
  : (state_quarters : ℚ) / (total_quarters : ℚ) = 2 / 5 :=
sorry

end fraction_of_state_quarters_is_two_fifths_l170_170572


namespace area_ratio_of_squares_l170_170322

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * a = 1 / 2 * (4 * b)) : (b^2 / a^2) = 4 :=
by
  -- Proof goes here
  sorry

end area_ratio_of_squares_l170_170322


namespace part1_part2_l170_170974

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | 1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}
def H (a : ℝ) : Set ℝ := {x | abs (x - a) <= 2}

def symdiff (A B : Set ℝ) : Set ℝ := A ∩ (U \ B)

theorem part1 :
  symdiff M N = {x | 1 < x ∧ x < 2} ∧
  symdiff N M = {x | 3 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem part2 (a : ℝ) :
  symdiff (symdiff N M) (H a) =
    if a ≥ 4 ∨ a ≤ -1 then {x | 1 < x ∧ x < 2}
    else if 3 < a ∧ a < 4 then {x | 1 < x ∧ x < a - 2}
    else if -1 < a ∧ a < 0 then {x | a + 2 < x ∧ x < 2}
    else ∅ :=
by
  sorry

end part1_part2_l170_170974


namespace dishwasher_spending_l170_170107

theorem dishwasher_spending (E : ℝ) (h1 : E > 0) 
    (rent : ℝ := 0.40 * E)
    (left_over : ℝ := 0.28 * E)
    (spent : ℝ := 0.72 * E)
    (dishwasher : ℝ := spent - rent)
    (difference : ℝ := rent - dishwasher) :
    ((difference / rent) * 100) = 20 := 
by
  sorry

end dishwasher_spending_l170_170107


namespace plus_signs_count_l170_170021

theorem plus_signs_count (n p m : ℕ) (h_total : n = 23) 
  (h_plus_min : ∀ s, s.card = 10 → ∃ t ∈ s, is_plus t)
  (h_minus_min : ∀ s, s.card = 15 → ∃ t ∈ s, is_minus t)
  : p = 14 :=
sorry

end plus_signs_count_l170_170021


namespace determine_k_if_even_function_l170_170137

noncomputable def f (x k : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem determine_k_if_even_function (k : ℝ) (h_even: ∀ x : ℝ, f x k = f (-x) k ) : k = 1 :=
by
  sorry

end determine_k_if_even_function_l170_170137


namespace smallest_n_perfect_square_and_fifth_power_l170_170053

theorem smallest_n_perfect_square_and_fifth_power : 
  ∃ (n : ℕ), (∀ k : ℕ, 4 * n = k^2 ∧ 5 * n = k^5) ↔ n = 3125 :=
begin
  sorry
end

end smallest_n_perfect_square_and_fifth_power_l170_170053


namespace normal_distribution_probability_l170_170080

noncomputable 
def normal_distribution (μ σ : ℝ) : probability_mass_function ℝ := sorry -- assuming a normal distribution function

variables {ξ : ℝ} {a : ℝ}

-- Given conditions
axiom xi_normal : normal_distribution 10 100 = ξ
axiom P_greater_than_11 : P(ξ > 11) = a

-- Prove the Lean theorem
theorem normal_distribution_probability :
  P(9 < ξ ∧ ξ ≤ 11) = 1 - 2 * a :=
    sorry

end normal_distribution_probability_l170_170080


namespace coefficient_x4_in_expansion_l170_170759

theorem coefficient_x4_in_expansion : 
  (∃ (c : ℤ), c = (choose 8 4) * 3^4 * 2^4 ∧ c = 90720) := 
by
  use (choose 8 4) * 3^4 * 2^4
  split
  sorry
  sorry

end coefficient_x4_in_expansion_l170_170759


namespace max_value_exponent_l170_170526

theorem max_value_exponent {a b : ℝ} (h : 0 < b ∧ b < a ∧ a < 1) :
  max (max (a^b) (b^a)) (max (a^a) (b^b)) = a^b :=
sorry

end max_value_exponent_l170_170526


namespace probability_boarding_251_l170_170938

theorem probability_boarding_251 :
  let interval_152 := 5
  let interval_251 := 7
  let total_events := interval_152 * interval_251
  let favorable_events := (interval_152 * interval_152) / 2
  (favorable_events / total_events : ℚ) = 5 / 14 :=
by 
  sorry

end probability_boarding_251_l170_170938


namespace pascal_triangle_probability_l170_170801

-- Define the probability problem in Lean 4
theorem pascal_triangle_probability :
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  (ones_count + twos_count) / total_elements = 5 / 14 :=
by
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  have h1 : total_elements = 210 := by sorry
  have h2 : ones_count = 39 := by sorry
  have h3 : twos_count = 36 := by sorry
  have h4 : (39 + 36) / 210 = 5 / 14 := by sorry
  exact h4

end pascal_triangle_probability_l170_170801


namespace steve_book_earning_l170_170887

theorem steve_book_earning
  (total_copies : ℕ)
  (advance_copies : ℕ)
  (total_kept : ℝ)
  (agent_cut_percentage : ℝ)
  (copies : ℕ)
  (money_kept : ℝ)
  (x : ℝ)
  (h1 : total_copies = 1000000)
  (h2 : advance_copies = 100000)
  (h3 : total_kept = 1620000)
  (h4 : agent_cut_percentage = 0.10)
  (h5 : copies = total_copies - advance_copies)
  (h6 : money_kept = copies * (1 - agent_cut_percentage) * x)
  (h7 : money_kept = total_kept) :
  x = 2 := 
by 
  sorry

end steve_book_earning_l170_170887


namespace smallest_n_l170_170199

theorem smallest_n :
∃ (n : ℕ), (0 < n) ∧ (∃ k1 : ℕ, 5 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 7 * n = k2 ^ 3) ∧ n = 1225 :=
sorry

end smallest_n_l170_170199


namespace required_bollards_l170_170642

theorem required_bollards 
  (bollards_per_side : ℕ)
  (sides : ℕ)
  (fraction_installed : ℚ)
  : bollards_per_side = 4000 → 
    sides = 2 → 
    fraction_installed = 3/4 → 
    let total_bollards := bollards_per_side * sides in 
    let installed_bollards := fraction_installed * total_bollards in 
    let remaining_bollards := total_bollards - installed_bollards in 
    remaining_bollards = 2000 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end required_bollards_l170_170642


namespace inequality_solution_l170_170434

theorem inequality_solution (x : ℝ) :
  -1 < (x^2 - 12 * x + 35) / (x^2 - 4 * x + 8) ∧
  (x^2 - 12 * x + 35) / (x^2 - 4 * x + 8) < 1 ↔
  x > (27 / 8) :=
by sorry

end inequality_solution_l170_170434


namespace possible_values_of_a₁_l170_170615

-- Define arithmetic progression with first term a₁ and common difference d
def arithmetic_progression (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

-- Define the sum of the first 7 terms of the arithmetic progression
def sum_first_7_terms (a₁ d : ℤ) : ℤ := 7 * a₁ + 21 * d

-- Define the conditions given
def condition1 (a₁ d : ℤ) : Prop := 
  (arithmetic_progression a₁ d 7) * (arithmetic_progression a₁ d 12) > (sum_first_7_terms a₁ d) + 20

def condition2 (a₁ d : ℤ) : Prop := 
  (arithmetic_progression a₁ d 9) * (arithmetic_progression a₁ d 10) < (sum_first_7_terms a₁ d) + 44

-- The main problem to prove
def problem (a₁ : ℤ) (d : ℤ) : Prop := 
  condition1 a₁ d ∧ condition2 a₁ d

-- The theorem statement to prove
theorem possible_values_of_a₁ (a₁ d : ℤ) : problem a₁ d → a₁ = -9 ∨ a₁ = -8 ∨ a₁ = -7 ∨ a₁ = -6 ∨ a₁ = -4 ∨ a₁ = -3 ∨ a₁ = -2 ∨ a₁ = -1 := 
by sorry

end possible_values_of_a₁_l170_170615


namespace log_identity_l170_170160

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_identity
    (a b c : ℝ)
    (h1 : a ^ 2 + b ^ 2 = c ^ 2)
    (h2 : a > 0)
    (h3 : c > 0)
    (h4 : b > 0)
    (h5 : c > b) :
    log_base (c + b) a + log_base (c - b) a = 2 * log_base (c + b) a * log_base (c - b) a :=
sorry

end log_identity_l170_170160


namespace range_of_a_l170_170851

open Real

theorem range_of_a {a : ℝ} :
  (∃ x : ℝ, sqrt (3 * x + 6) + sqrt (14 - x) > a) → a < 8 :=
by
  intro h
  sorry

end range_of_a_l170_170851


namespace triangle_base_l170_170902

theorem triangle_base (sum_of_sides : ℕ) (left_side : ℕ) (right_side : ℕ) (base : ℕ)
  (h1 : sum_of_sides = 50)
  (h2 : right_side = left_side + 2)
  (h3 : left_side = 12) :
  base = 24 :=
by
  sorry

end triangle_base_l170_170902


namespace perfect_square_trinomial_m_l170_170393

theorem perfect_square_trinomial_m (m : ℤ) (x : ℤ) : (∃ a : ℤ, x^2 - mx + 16 = (x - a)^2) ↔ (m = 8 ∨ m = -8) :=
by sorry

end perfect_square_trinomial_m_l170_170393


namespace sum_of_squares_and_product_l170_170589

theorem sum_of_squares_and_product (x y : ℤ) 
  (h1 : x^2 + y^2 = 290) 
  (h2 : x * y = 96) :
  x + y = 22 :=
sorry

end sum_of_squares_and_product_l170_170589


namespace gcd_solutions_l170_170957

theorem gcd_solutions (x m n p: ℤ) (h_eq: x * (4 * x - 5) = 7) (h_gcd: Int.gcd m (Int.gcd n p) = 1)
  (h_form: ∃ x1 x2: ℤ, x1 = (m + Int.sqrt n) / p ∧ x2 = (m - Int.sqrt n) / p) : m + n + p = 150 :=
by
  have disc_eq : 25 + 112 = 137 :=
    by norm_num
  sorry

end gcd_solutions_l170_170957


namespace problem_I_inequality_solution_problem_II_condition_on_b_l170_170514

-- Define the function f(x).
def f (x : ℝ) : ℝ := |x - 2|

-- Problem (I): Proving the solution set to the given inequality.
theorem problem_I_inequality_solution (x : ℝ) : 
  f x + f (x + 1) ≥ 5 ↔ (x ≥ 4 ∨ x ≤ -1) :=
sorry

-- Problem (II): Proving the condition on |b|.
theorem problem_II_condition_on_b (a b : ℝ) (ha : |a| > 1) (h : f (a * b) > |a| * f (b / a)) :
  |b| > 2 :=
sorry

end problem_I_inequality_solution_problem_II_condition_on_b_l170_170514


namespace triangle_base_l170_170901

theorem triangle_base (sum_of_sides : ℕ) (left_side : ℕ) (right_side : ℕ) (base : ℕ)
  (h1 : sum_of_sides = 50)
  (h2 : right_side = left_side + 2)
  (h3 : left_side = 12) :
  base = 24 :=
by
  sorry

end triangle_base_l170_170901


namespace tan_of_alpha_l170_170667

theorem tan_of_alpha 
  (α : ℝ)
  (h1 : Real.sin α = (3 / 5))
  (h2 : α ∈ Set.Ioo (π / 2) π) : Real.tan α = -3 / 4 :=
sorry

end tan_of_alpha_l170_170667


namespace corn_height_after_three_weeks_l170_170503

theorem corn_height_after_three_weeks 
  (week1_growth : ℕ) (week2_growth : ℕ) (week3_growth : ℕ) 
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  week1_growth + week2_growth + week3_growth = 22 :=
by {
  sorry
}

end corn_height_after_three_weeks_l170_170503


namespace length_segment_midpoints_diagonals_trapezoid_l170_170441

theorem length_segment_midpoints_diagonals_trapezoid
  (a b c d : ℝ)
  (h_side_lengths : (2 = a ∨ 2 = b ∨ 2 = c ∨ 2 = d) ∧ 
                    (10 = a ∨ 10 = b ∨ 10 = c ∨ 10 = d) ∧ 
                    (10 = a ∨ 10 = b ∨ 10 = c ∨ 10 = d) ∧ 
                    (20 = a ∨ 20 = b ∨ 20 = c ∨ 20 = d))
  (h_parallel_sides : (a = 20 ∧ b = 2) ∨ (a = 2 ∧ b = 20)) :
  (1/2) * |a - b| = 9 :=
by
  sorry

end length_segment_midpoints_diagonals_trapezoid_l170_170441


namespace multiple_of_7_l170_170913

theorem multiple_of_7 :
  ∃ k : ℤ, 77 = 7 * k :=
sorry

end multiple_of_7_l170_170913


namespace probability_same_color_plates_l170_170266

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem probability_same_color_plates :
  (choose 6 3 : ℚ) / (choose 11 3 : ℚ) = 4 / 33 := by
  sorry

end probability_same_color_plates_l170_170266


namespace largest_valid_number_l170_170606

/-
Problem: 
What is the largest number, all of whose digits are 3, 2, or 4 whose digits add up to 16?

We prove that 4432 is the largest such number.
-/

def digits := [3, 2, 4]

def sum_of_digits (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

def is_valid_digit (d : ℕ) : Prop :=
  d = 3 ∨ d = 2 ∨ d = 4

def generate_number (l : List ℕ) : ℕ :=
  l.foldl (λ acc d => acc * 10 + d) 0

theorem largest_valid_number : 
  ∃ l : List ℕ, (∀ d ∈ l, is_valid_digit d) ∧ sum_of_digits l = 16 ∧ generate_number l = 4432 :=
  sorry

end largest_valid_number_l170_170606


namespace value_of_expression_l170_170973

theorem value_of_expression (A B C D : ℝ) (h1 : A - B = 30) (h2 : C + D = 20) :
  (B + C) - (A - D) = -10 :=
by
  sorry

end value_of_expression_l170_170973


namespace sin_sum_arcsin_arctan_l170_170953

theorem sin_sum_arcsin_arctan :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (1 / 2)
  Real.sin (a + b) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_sum_arcsin_arctan_l170_170953


namespace bowling_average_decrease_l170_170059

theorem bowling_average_decrease
    (initial_average : ℝ) (wickets_last_match : ℝ) (runs_last_match : ℝ)
    (average_decrease : ℝ) (W : ℝ)
    (H_initial : initial_average = 12.4)
    (H_wickets_last_match : wickets_last_match = 6)
    (H_runs_last_match : runs_last_match = 26)
    (H_average_decrease : average_decrease = 0.4) :
    W = 115 :=
by
  sorry

end bowling_average_decrease_l170_170059


namespace total_emails_vacation_l170_170066

def day_1_emails : ℕ := 16
def day_2_emails : ℕ := day_1_emails / 2
def day_3_emails : ℕ := day_2_emails / 2
def day_4_emails : ℕ := day_3_emails / 2

def total_emails : ℕ := day_1_emails + day_2_emails + day_3_emails + day_4_emails

theorem total_emails_vacation : total_emails = 30 := by
  -- Use "sorry" to skip the proof as per instructions.
  sorry

end total_emails_vacation_l170_170066


namespace Lauryn_earnings_l170_170707

variables (L : ℝ)

theorem Lauryn_earnings (h1 : 0.70 * L + L = 3400) : L = 2000 :=
sorry

end Lauryn_earnings_l170_170707


namespace product_of_three_consecutive_integers_divisible_by_six_l170_170197

theorem product_of_three_consecutive_integers_divisible_by_six (n : ℕ) : 
  6 ∣ (n * (n + 1) * (n + 2)) :=
sorry

end product_of_three_consecutive_integers_divisible_by_six_l170_170197


namespace IMO1991Q1_l170_170820

theorem IMO1991Q1 (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
    (h4 : 3^x + 4^y = 5^z) : x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end IMO1991Q1_l170_170820


namespace opposite_of_neg_two_is_two_l170_170742

theorem opposite_of_neg_two_is_two (x : ℤ) (h : -2 + x = 0) : x = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l170_170742


namespace compute_expression_l170_170509

theorem compute_expression (x : ℝ) (h : x = 3) : 
  (x^8 + 18 * x^4 + 81) / (x^4 + 9) = 90 :=
by 
  sorry

end compute_expression_l170_170509


namespace min_accommodation_cost_l170_170786

theorem min_accommodation_cost :
  ∃ (x y z : ℕ), x + y + z = 20 ∧ 3 * x + 2 * y + z = 50 ∧ 100 * 3 * x + 150 * 2 * y + 200 * z = 5500 :=
by
  sorry

end min_accommodation_cost_l170_170786


namespace bigger_part_of_sum_and_linear_combination_l170_170210

theorem bigger_part_of_sum_and_linear_combination (x y : ℕ) 
  (h1 : x + y = 24) 
  (h2 : 7 * x + 5 * y = 146) : x = 13 :=
by 
  sorry

end bigger_part_of_sum_and_linear_combination_l170_170210


namespace carlos_gold_quarters_l170_170226

theorem carlos_gold_quarters :
  (let quarter_weight := 1 / 5
       quarter_value := 0.25
       value_per_ounce := 100
       quarters_per_ounce := 1 / quarter_weight
       melt_value := value_per_ounce
       spend_value := quarters_per_ounce * quarter_value
    in melt_value / spend_value = 80) :=
by
  -- Definitions
  let quarter_weight := 1 / 5
  let quarter_value := 0.25
  let value_per_ounce := 100
  let quarters_per_ounce := 1 / quarter_weight
  let melt_value := value_per_ounce
  let spend_value := quarters_per_ounce * quarter_value

  -- Conclusion to be proven
  have h1 : quarters_per_ounce = 5 := sorry
  have h2 : spend_value = 1.25 := sorry
  have h3 : melt_value / spend_value = 80 := sorry

  show melt_value / spend_value = 80 from h3

end carlos_gold_quarters_l170_170226


namespace probability_same_color_is_correct_l170_170279

-- Definitions of the conditions
def num_red : ℕ := 6
def num_blue : ℕ := 5
def total_plates : ℕ := num_red + num_blue
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probability statement
def prob_three_same_color : ℚ :=
  let total_ways := choose total_plates 3
  let ways_red := choose num_red 3
  let ways_blue := choose num_blue 3
  let favorable_ways := ways_red
  favorable_ways / total_ways

theorem probability_same_color_is_correct : prob_three_same_color = (4 : ℚ) / 33 := sorry

end probability_same_color_is_correct_l170_170279


namespace inverse_of_B_squared_l170_170370

theorem inverse_of_B_squared (B_inv : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B_inv = ![![3, -2], ![0, 5]]) : 
  (B_inv * B_inv) = ![![9, -16], ![0, 25]] :=
by
  sorry

end inverse_of_B_squared_l170_170370


namespace num_factors_m_l170_170565

noncomputable def m : ℕ := 2^5 * 3^6 * 5^7 * 6^8

theorem num_factors_m : ∃ (k : ℕ), k = 1680 ∧ ∀ d : ℕ, d ∣ m ↔ ∃ (a b c : ℕ), 0 ≤ a ∧ a ≤ 13 ∧ 0 ≤ b ∧ b ≤ 14 ∧ 0 ≤ c ∧ c ≤ 7 ∧ d = 2^a * 3^b * 5^c :=
by 
sorry

end num_factors_m_l170_170565


namespace division_expression_l170_170803

theorem division_expression :
  (240 : ℚ) / (12 + 12 * 2 - 3) = 240 / 33 := by
  sorry

end division_expression_l170_170803


namespace smallest_Y_l170_170413

-- Define the necessary conditions
def is_digits_0_1 (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 0 ∨ d = 1

def is_divisible_by_15 (n : ℕ) : Prop :=
  n % 15 = 0

-- Define the main problem statement
theorem smallest_Y (S Y : ℕ) (hS_pos : S > 0) (hS_digits : is_digits_0_1 S) (hS_div_15 : is_divisible_by_15 S) (hY : Y = S / 15) :
  Y = 74 :=
sorry

end smallest_Y_l170_170413


namespace vector_perpendicular_l170_170543

open Real

theorem vector_perpendicular (t : ℝ) (a b : ℝ × ℝ) (h_a : a = (1, 2)) (h_b : b = (4, 3)) :
  a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0 ↔ t = -2 := by
  sorry

end vector_perpendicular_l170_170543


namespace point_A_inside_circle_max_min_dist_square_on_circle_chord_through_origin_l170_170375

def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 4 * y - m = 0

def inside_circle (x y m : ℝ) : Prop :=
  (x-1)^2 + (y+2)^2 < 5 + m

theorem point_A_inside_circle (m : ℝ) : -1 < m ∧ m < 4 ↔ inside_circle m (-2) m :=
sorry

def circle_equation_m_4 (x y : ℝ) : Prop :=
  circle_equation x y 4

def dist_square_to_point_H (x y : ℝ) : ℝ :=
  (x - 4)^2 + (y - 2)^2

theorem max_min_dist_square_on_circle (P : ℝ × ℝ) :
  circle_equation_m_4 P.1 P.2 →
  4 ≤ dist_square_to_point_H P.1 P.2 ∧ dist_square_to_point_H P.1 P.2 ≤ 64 :=
sorry

def line_equation (m x y : ℝ) : Prop :=
  y = x + m

theorem chord_through_origin (m : ℝ) :
  ∃ m : ℝ, line_equation m (1 : ℝ) (-2 : ℝ) ∧ 
  (m = -4 ∨ m = 1) :=
sorry

end point_A_inside_circle_max_min_dist_square_on_circle_chord_through_origin_l170_170375


namespace corn_height_after_three_weeks_l170_170502

theorem corn_height_after_three_weeks 
  (week1_growth : ℕ) (week2_growth : ℕ) (week3_growth : ℕ) 
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  week1_growth + week2_growth + week3_growth = 22 :=
by {
  sorry
}

end corn_height_after_three_weeks_l170_170502


namespace wallpaper_removal_time_l170_170516

theorem wallpaper_removal_time (time_per_wall : ℕ) (dining_room_walls_remaining : ℕ) (living_room_walls : ℕ) :
  time_per_wall = 2 → dining_room_walls_remaining = 3 → living_room_walls = 4 → 
  time_per_wall * (dining_room_walls_remaining + living_room_walls) = 14 :=
by
  sorry

end wallpaper_removal_time_l170_170516


namespace employees_participating_in_game_l170_170222

theorem employees_participating_in_game 
  (managers players : ℕ)
  (teams people_per_team : ℕ)
  (h_teams : teams = 3)
  (h_people_per_team : people_per_team = 2)
  (h_managers : managers = 3)
  (h_total_players : players = teams * people_per_team) :
  players - managers = 3 :=
sorry

end employees_participating_in_game_l170_170222


namespace ratio_girls_total_members_l170_170143

theorem ratio_girls_total_members {p_boy p_girl : ℚ} (h_prob_ratio : p_girl = (3/5) * p_boy) (h_total_prob : p_boy + p_girl = 1) :
  p_girl / (p_boy + p_girl) = 3 / 8 :=
by
  sorry

end ratio_girls_total_members_l170_170143


namespace quadratic_solution_l170_170183

theorem quadratic_solution (x : ℝ) : x^2 - 2 * x = 0 ↔ (x = 0 ∨ x = 2) := by
  sorry

end quadratic_solution_l170_170183


namespace problem_statement_l170_170567

noncomputable def g : ℝ → ℝ
| x => if x < 0 then -x
            else if x < 5 then x + 3
            else 2 * x ^ 2

theorem problem_statement : g (-6) + g 3 + g 8 = 140 :=
by
  -- Proof goes here
  sorry

end problem_statement_l170_170567


namespace find_radius_squared_l170_170779

theorem find_radius_squared (r : ℝ) (AB_len CD_len BP : ℝ) (angle_APD : ℝ) (h1 : AB_len = 12)
    (h2 : CD_len = 9) (h3 : BP = 10) (h4 : angle_APD = 60) : r^2 = 111 := by
  have AB_len := h1
  have CD_len := h2
  have BP := h3
  have angle_APD := h4
  sorry

end find_radius_squared_l170_170779


namespace minimum_guests_l170_170440

-- Define the conditions as variables
def total_food : ℕ := 4875
def max_food_per_guest : ℕ := 3

-- Define the theorem we need to prove
theorem minimum_guests : ∃ g : ℕ, g * max_food_per_guest = total_food ∧ g >= 1625 := by
  sorry

end minimum_guests_l170_170440


namespace find_n_arithmetic_sequence_l170_170374

-- Given conditions
def a₁ : ℕ := 20
def aₙ : ℕ := 54
def Sₙ : ℕ := 999

-- Arithmetic sequence sum formula and proof statement of n = 27
theorem find_n_arithmetic_sequence
  (a₁ : ℕ)
  (aₙ : ℕ)
  (Sₙ : ℕ)
  (h₁ : a₁ = 20)
  (h₂ : aₙ = 54)
  (h₃ : Sₙ = 999) : ∃ n : ℕ, n = 27 := 
by
  sorry

end find_n_arithmetic_sequence_l170_170374


namespace range_f_period_f_monotonic_increase_intervals_l170_170821

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin x) ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 1 

theorem range_f : Set.Icc 0 4 = Set.range f := sorry

theorem period_f : ∀ x, f (x + Real.pi) = f x := sorry

theorem monotonic_increase_intervals (k : ℤ) :
  ∀ x, (-π / 6 + k * π : ℝ) ≤ x ∧ x ≤ (π / 3 + k * π : ℝ) → 
        ∀ y, f y ≤ f x → y ≤ x := sorry

end range_f_period_f_monotonic_increase_intervals_l170_170821


namespace simplify_complex_subtraction_l170_170429

-- Definition of the nested expression
def complex_subtraction (x : ℝ) : ℝ :=
  1 - (2 - (3 - (4 - (5 - (6 - x)))))

-- Statement of the theorem to be proven
theorem simplify_complex_subtraction (x : ℝ) : complex_subtraction x = x - 3 :=
by {
  -- This proof will need to be filled in to verify the statement
  sorry
}

end simplify_complex_subtraction_l170_170429


namespace probability_same_color_is_correct_l170_170281

-- Definitions of the conditions
def num_red : ℕ := 6
def num_blue : ℕ := 5
def total_plates : ℕ := num_red + num_blue
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probability statement
def prob_three_same_color : ℚ :=
  let total_ways := choose total_plates 3
  let ways_red := choose num_red 3
  let ways_blue := choose num_blue 3
  let favorable_ways := ways_red
  favorable_ways / total_ways

theorem probability_same_color_is_correct : prob_three_same_color = (4 : ℚ) / 33 := sorry

end probability_same_color_is_correct_l170_170281


namespace closest_integer_to_cube_root_of_200_l170_170614

theorem closest_integer_to_cube_root_of_200 : 
  let n := 200 in
  let a := 5 in 
  let b := 6 in 
  abs (b - real.cbrt n) < abs (a - real.cbrt n) := 
by sorry

end closest_integer_to_cube_root_of_200_l170_170614


namespace ted_age_proof_l170_170646

theorem ted_age_proof (s t : ℝ) (h1 : t = 3 * s - 20) (h2 : t + s = 78) : t = 53.5 :=
by
  sorry  -- Proof steps are not required, hence using sorry.

end ted_age_proof_l170_170646


namespace probability_same_color_l170_170261

-- Definitions with conditions
def totalPlates := 11
def redPlates := 6
def bluePlates := 5

-- Calculate combinations
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main statement
theorem probability_same_color (totalPlates redPlates bluePlates : ℕ) (h1 : totalPlates = 11) 
(h2 : redPlates = 6) (h3 : bluePlates = 5) : 
  (2 / 11 : ℚ) = ((choose redPlates 3 + choose bluePlates 3) : ℚ) / (choose totalPlates 3) :=
by
  -- Proof steps will be inserted here
  sorry

end probability_same_color_l170_170261


namespace gcd_multiple_less_than_120_l170_170765

theorem gcd_multiple_less_than_120 (n : ℕ) (h1 : n < 120) (h2 : n % 10 = 0) (h3 : n % 15 = 0) : n ≤ 90 :=
by {
  sorry
}

end gcd_multiple_less_than_120_l170_170765


namespace hexagon_arrangements_eq_144_l170_170879

def is_valid_arrangement (arr : (Fin 7 → ℕ)) : Prop :=
  ∀ (i j k : Fin 7),
    (i.val + j.val + k.val = 18) → -- 18 being a derived constant factor (since 3x = 28 + 2G where G ∈ {1, 4, 7} and hence x = 30,34,38/3 respectively make it divisible by 3 sum is 18 always)
    arr i + arr j + arr k = arr ⟨3, sorry⟩ -- arr[3] is the position of G

noncomputable def count_valid_arrangements : ℕ :=
  sorry -- Calculation of 3*48 goes here and respective pairing and permutations.

theorem hexagon_arrangements_eq_144 :
  count_valid_arrangements = 144 :=
sorry

end hexagon_arrangements_eq_144_l170_170879


namespace binomial_510_510_l170_170650

theorem binomial_510_510 : Nat.choose 510 510 = 1 :=
by
  sorry

end binomial_510_510_l170_170650


namespace lines_perpendicular_to_same_plane_are_parallel_l170_170860

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

end lines_perpendicular_to_same_plane_are_parallel_l170_170860


namespace smallest_n_l170_170200

theorem smallest_n (n : ℕ) (h1 : ∃ k1, 5 * n = k1^2) (h2 : ∃ k2, 7 * n = k2^3) : n = 245 :=
sorry

end smallest_n_l170_170200


namespace girls_together_arrangements_no_two_girls_adjacent_arrangements_exactly_three_between_arrangements_adjacent_not_next_to_arrangements_l170_170116

open Finset

section problem_conditions

def boys : Finset ℕ := range 4
def girls : Finset ℕ := range 3
def people : Finset ℕ := boys ∪ girls

-- Definitions for specific conditions
def girls_together (arrangement : List ℕ) : Prop :=
  ⟦arrangement ≃ filter (λ x, x ∈ girls) arrangement⟧

def no_two_girls_adjacent (arrangement : List ℕ) : Prop :=
  ∀ i ∈ range (arrangement.length - 1), arrangement[i] ∉ girls ∨ arrangement[i + 1] ∉ girls

def exactly_three_between (arrangement : List ℕ) (A B : ℕ) : Prop :=
  ∃ i j, arrangement[i] = A ∧ arrangement[j] = B ∧ abs (i - j) = 4

def adjacent_not_next_to (arrangement : List ℕ) (A B C : ℕ) : Prop :=
  ∃ i, arrangement[i] = A ∧ arrangement[i + 1] = B ∧ ∀ j, j ≠ i ∧ j ≠ i + 1 → arrangement[j] ≠ C

end problem_conditions

-- Theorems for the subproblems
theorem girls_together_arrangements :
  ∃ arrangements : Finset (List ℕ), girls_together arrangements ∧ arrangements.card = 720 := sorry

theorem no_two_girls_adjacent_arrangements :
  ∃ arrangements : Finset (List ℕ), no_two_girls_adjacent arrangements ∧ arrangements.card = 1440 := sorry

theorem exactly_three_between_arrangements (A B : ℕ) :
  ∃ arrangements : Finset (List ℕ), exactly_three_between arrangements A B ∧ arrangements.card = 720 := sorry

theorem adjacent_not_next_to_arrangements (A B C : ℕ) :
  ∃ arrangements : Finset (List ℕ), adjacent_not_next_to arrangements A B C ∧ arrangements.card = 960 := sorry

end girls_together_arrangements_no_two_girls_adjacent_arrangements_exactly_three_between_arrangements_adjacent_not_next_to_arrangements_l170_170116


namespace members_play_both_eq_21_l170_170619

-- Given definitions
def TotalMembers := 80
def MembersPlayBadminton := 48
def MembersPlayTennis := 46
def MembersPlayNeither := 7

-- Inclusion-Exclusion Principle application to solve the problem
def MembersPlayBoth : ℕ := MembersPlayBadminton + MembersPlayTennis - (TotalMembers - MembersPlayNeither)

-- The theorem we want to prove
theorem members_play_both_eq_21 : MembersPlayBoth = 21 :=
by
  -- skipping the proof
  sorry

end members_play_both_eq_21_l170_170619


namespace ticket_cost_before_rally_l170_170001

-- We define the variables and constants given in the problem
def total_attendance : ℕ := 750
def tickets_before_rally : ℕ := 475
def tickets_at_door : ℕ := total_attendance - tickets_before_rally
def cost_at_door : ℝ := 2.75
def total_receipts : ℝ := 1706.25

-- Problem statement: Prove that the cost of each ticket bought before the rally (x) is 2 dollars.
theorem ticket_cost_before_rally (x : ℝ) 
  (h₁ : tickets_before_rally * x + tickets_at_door * cost_at_door = total_receipts) :
  x = 2 :=
by
  sorry

end ticket_cost_before_rally_l170_170001


namespace force_on_dam_correct_l170_170064

noncomputable def compute_force_on_dam (ρ g a b h : ℝ) : ℝ :=
  let pressure_at_depth (x : ℝ) := ρ * g * x
  let width_at_depth (x : ℝ) := b - x * (b - a) / h
  ∫ x in 0..h, pressure_at_depth x * width_at_depth x

theorem force_on_dam_correct :
  compute_force_on_dam 1000 10 7.2 12.0 5.0 = 1100000 := by
  sorry

end force_on_dam_correct_l170_170064


namespace ab_equals_one_l170_170871

theorem ab_equals_one {a b : ℝ} (h : a ≠ b) (hf : |Real.log a| = |Real.log b|) : a * b = 1 :=
  sorry

end ab_equals_one_l170_170871


namespace juice_packs_in_box_l170_170595

theorem juice_packs_in_box 
  (W_box L_box H_box W_juice_pack L_juice_pack H_juice_pack : ℕ)
  (hW_box : W_box = 24) (hL_box : L_box = 15) (hH_box : H_box = 28)
  (hW_juice_pack : W_juice_pack = 4) (hL_juice_pack : L_juice_pack = 5) (hH_juice_pack : H_juice_pack = 7) : 
  (W_box * L_box * H_box) / (W_juice_pack * L_juice_pack * H_juice_pack) = 72 :=
by
  sorry

end juice_packs_in_box_l170_170595


namespace center_distance_correct_l170_170777

noncomputable def ball_diameter : ℝ := 6
noncomputable def ball_radius : ℝ := ball_diameter / 2
noncomputable def R₁ : ℝ := 150
noncomputable def R₂ : ℝ := 50
noncomputable def R₃ : ℝ := 90
noncomputable def R₄ : ℝ := 120
noncomputable def elevation : ℝ := 4

noncomputable def adjusted_R₁ : ℝ := R₁ - ball_radius
noncomputable def adjusted_R₂ : ℝ := R₂ + ball_radius + elevation
noncomputable def adjusted_R₃ : ℝ := R₃ - ball_radius
noncomputable def adjusted_R₄ : ℝ := R₄ - ball_radius

noncomputable def distance_R₁ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₁
noncomputable def distance_R₂ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₂
noncomputable def distance_R₃ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₃
noncomputable def distance_R₄ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₄

noncomputable def total_distance : ℝ := distance_R₁ + distance_R₂ + distance_R₃ + distance_R₄

theorem center_distance_correct : total_distance = 408 * Real.pi := 
  by
  sorry

end center_distance_correct_l170_170777


namespace kittens_total_number_l170_170597

theorem kittens_total_number (W L H R : ℕ) (k : ℕ) 
  (h1 : W = 500) 
  (h2 : L = 80) 
  (h3 : H = 200) 
  (h4 : L + H + R = W) 
  (h5 : 40 * k ≤ R) 
  (h6 : R ≤ 50 * k) 
  (h7 : ∀ m, m ≠ 4 → m ≠ 6 → m ≠ k →
        40 * m ≤ R → R ≤ 50 * m → False) : 
  k = 5 ∧ 2 + 4 + k = 11 := 
by {
  -- The proof would go here
  sorry 
}

end kittens_total_number_l170_170597


namespace darryl_had_8_cantaloupes_left_l170_170942

namespace MelonSales

variable {α : Type*} [LinearOrderedField α]

structure Darryl := 
(CantaloupePrice : α)
(HoneydewPrice : α)
(InitialCantaloupes : α)
(InitialHoneydews : α)
(DroppedCantaloupes : α)
(RottenHoneydews : α)
(FinalHoneydews : α)
(TotalRevenue : α)

def cantaloupes_left_at_end_of_day (d : Darryl) : α :=
  d.InitialCantaloupes - d.DroppedCantaloupes - (d.TotalRevenue - d.HoneydewPrice * (d.InitialHoneydews - d.RottenHoneydews - d.FinalHoneydews)) / d.CantaloupePrice

theorem darryl_had_8_cantaloupes_left (d : Darryl) : 
  (d.CantaloupePrice = 2) → 
  (d.HoneydewPrice = 3) → 
  (d.InitialCantaloupes = 30) →
  (d.InitialHoneydews = 27) →
  (d.DroppedCantaloupes = 2) →
  (d.RottenHoneydews = 3) →
  (d.FinalHoneydews = 9) →
  (d.TotalRevenue = 85) →
  cantaloupes_left_at_end_of_day d = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end MelonSales

end darryl_had_8_cantaloupes_left_l170_170942


namespace product_of_divisors_of_18_l170_170457

theorem product_of_divisors_of_18 : ∏ d in {1, 2, 3, 6, 9, 18}, d = 5832 := by
  sorry

end product_of_divisors_of_18_l170_170457


namespace gold_quarter_value_comparison_l170_170227

theorem gold_quarter_value_comparison:
  (worth_in_store per_quarter: ℕ → ℝ) 
  (weight_per_quarter in_ounce: ℝ) 
  (earning_per_ounce melted: ℝ) : 
  (worth_in_store 4  = 0.25) →
  (weight_per_quarter = 1/5) →
  (earning_per_ounce = 100) →
  (earning_per_ounce * weight_per_quarter / worth_in_store 4 = 80) :=
by
  -- The proof goes here
  sorry

end gold_quarter_value_comparison_l170_170227


namespace machine_production_time_l170_170609

theorem machine_production_time (x : ℝ) 
  (h1 : 60 / x + 2 = 12) : 
  x = 6 :=
sorry

end machine_production_time_l170_170609


namespace fisherman_caught_total_fish_l170_170073

noncomputable def number_of_boxes : ℕ := 15
noncomputable def fish_per_box : ℕ := 20
noncomputable def fish_outside_boxes : ℕ := 6

theorem fisherman_caught_total_fish :
  number_of_boxes * fish_per_box + fish_outside_boxes = 306 :=
by
  sorry

end fisherman_caught_total_fish_l170_170073


namespace inequality_holds_for_all_x_l170_170845

theorem inequality_holds_for_all_x (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3 / 5 < a ∧ a ≤ 1) :=
by
  sorry

end inequality_holds_for_all_x_l170_170845


namespace percentage_increase_in_gross_revenue_l170_170773

theorem percentage_increase_in_gross_revenue 
  (P R : ℝ) 
  (hP : P > 0) 
  (hR : R > 0) 
  (new_price : ℝ := 0.80 * P) 
  (new_quantity : ℝ := 1.60 * R) : 
  (new_price * new_quantity - P * R) / (P * R) * 100 = 28 := 
by
  sorry

end percentage_increase_in_gross_revenue_l170_170773


namespace problem_complement_intersection_l170_170382

open Set

-- Define the universal set U
def U : Set ℕ := {0, 2, 4, 6, 8, 10}

-- Define set A
def A : Set ℕ := {0, 2, 4, 6}

-- Define set B based on A
def B : Set ℕ := {x | x ∈ A ∧ x < 4}

-- Define the complement of set A within U
def complement_A_U : Set ℕ := U \ A

-- Define the complement of set B within U
def complement_B_U : Set ℕ := U \ B

-- Prove the given equations
theorem problem_complement_intersection :
  (complement_A_U = {8, 10}) ∧ (A ∩ complement_B_U = {4, 6}) := 
by
  sorry

end problem_complement_intersection_l170_170382


namespace roots_of_P_l170_170235

-- Define the polynomial P(x) = x^3 + x^2 - 6x - 6
noncomputable def P (x : ℝ) : ℝ := x^3 + x^2 - 6 * x - 6

-- Define the statement that the roots of the polynomial P are -1, sqrt(6), and -sqrt(6)
theorem roots_of_P : ∀ x : ℝ, P x = 0 ↔ (x = -1) ∨ (x = sqrt 6) ∨ (x = -sqrt 6) :=
sorry

end roots_of_P_l170_170235


namespace composite_square_perimeter_l170_170706

theorem composite_square_perimeter (p1 p2 : ℝ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  let s1 := p1 / 4
  let s2 := p2 / 4
  (p1 + p2 - 2 * s1) = 120 := 
by
  -- proof goes here
  sorry

end composite_square_perimeter_l170_170706


namespace range_of_a_l170_170996

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (ax^2 - ax + 1 ≤ 0)) ↔ 0 ≤ a ∧ a < 4 :=
by
  sorry

end range_of_a_l170_170996


namespace jenna_reading_goal_l170_170300

theorem jenna_reading_goal (total_days : ℕ) (total_pages : ℕ) (unread_days : ℕ) (pages_on_23rd : ℕ) :
  total_days = 30 → total_pages = 600 → unread_days = 4 → pages_on_23rd = 100 →
  ∃ (pages_per_day : ℕ), 
  let days_to_read := total_days - unread_days - 1 in
  let pages_to_read_on_other_days := total_pages - pages_on_23rd in
  days_to_read ≠ 0 →
  pages_per_day * days_to_read = pages_to_read_on_other_days ∧ pages_per_day = 20 :=
by
  intros h1 h2 h3 h4
  use 20
  simp_all
  sorry

end jenna_reading_goal_l170_170300


namespace count_of_plus_signs_l170_170032

-- Definitions based on the given conditions
def total_symbols : ℕ := 23
def any_10_symbols_contains_plus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 10 → (\exists c ∈ s, c = '+')
def any_15_symbols_contains_minus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 15 → (\exists c ∈ s, c = '-')

-- Main theorem statement without the proof
theorem count_of_plus_signs (symbols : Finset char) (h1 : symbols.card = total_symbols) 
  (h2 : any_10_symbols_contains_plus symbols) 
  (h3 : any_15_symbols_contains_minus symbols) : 
  symbols.count '+' = 14 :=
sorry

end count_of_plus_signs_l170_170032


namespace smallest_n_l170_170198

theorem smallest_n :
∃ (n : ℕ), (0 < n) ∧ (∃ k1 : ℕ, 5 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 7 * n = k2 ^ 3) ∧ n = 1225 :=
sorry

end smallest_n_l170_170198


namespace problem_statement_l170_170548

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 :=
by
  sorry

end problem_statement_l170_170548


namespace reflection_points_line_l170_170318

theorem reflection_points_line (m b : ℝ)
  (h1 : (10 : ℝ) = 2 * (6 - m * (6 : ℝ) + b)) -- Reflecting the point (6, (m * 6 + b)) to (10, 7)
  (h2 : (6 : ℝ) * m + b = 5) -- Midpoint condition
  (h3 : (6 : ℝ) = (2 + 10) / 2) -- Calculating midpoint x-coordinate
  (h4 : (5 : ℝ) = (3 + 7) / 2) -- Calculating midpoint y-coordinate
  : m + b = 15 :=
sorry

end reflection_points_line_l170_170318


namespace gcd_1734_816_l170_170910

theorem gcd_1734_816 : Nat.gcd 1734 816 = 102 := by
  sorry

end gcd_1734_816_l170_170910


namespace magician_card_trick_l170_170482

-- Definitions and proof goal
theorem magician_card_trick :
  let n := 12
  let total_cards := n ^ 2
  let duplicate_cards := n
  let non_duplicate_cards := total_cards - duplicate_cards - (n - 1) - (n - 1)
  let total_ways_with_two_duplicates := Nat.choose duplicate_cards 2
  let total_ways_with_one_duplicate :=
    duplicate_cards * non_duplicate_cards
  (total_ways_with_two_duplicates + total_ways_with_one_duplicate) = 1386 :=
by
  let n := 12
  let total_cards := n ^ 2
  let duplicate_cards := n
  let non_duplicate_cards := total_cards - duplicate_cards - (n - 1) - (n - 1)
  let total_ways_with_two_duplicates := Nat.choose duplicate_cards 2
  let total_ways_with_one_duplicate :=
    duplicate_cards * non_duplicate_cards
  have h : (total_ways_with_two_duplicates + total_ways_with_one_duplicate) = 1386 := sorry
  exact h

end magician_card_trick_l170_170482


namespace union_of_A_and_B_at_m_equals_3_range_of_m_if_A_union_B_equals_A_l170_170542

def set_A : Set ℝ := { x | x^2 - x - 12 ≤ 0 }
def set_B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

-- Statement 1: Prove that when \( m = 3 \), \( A \cup B \) = \( \{ x \mid -3 \leq x \leq 5 \} \).
theorem union_of_A_and_B_at_m_equals_3 : set_A ∪ set_B 3 = { x | -3 ≤ x ∧ x ≤ 5 } :=
sorry

-- Statement 2: Prove that if \( A ∪ B = A \), then the range of \( m \) is \( (-\infty, \frac{5}{2}] \).
theorem range_of_m_if_A_union_B_equals_A (m : ℝ) : (set_A ∪ set_B m = set_A) → m ≤ 5 / 2 :=
sorry

end union_of_A_and_B_at_m_equals_3_range_of_m_if_A_union_B_equals_A_l170_170542


namespace angle_PQR_correct_l170_170294

-- Define the points and angles
variables {R P Q S : Type*}
variables (angle_RSQ angle_QSP angle_RQS angle_PQS : ℝ)

-- Define the conditions
def condition1 : Prop := true  -- RSP is a straight line implicitly means angle_RSQ + angle_QSP = 180
def condition2 : Prop := angle_QSP = 70
def condition3 (RS SQ : Type*) : Prop := true  -- Triangle RSQ is isosceles with RS = SQ
def condition4 (PS SQ : Type*) : Prop := true  -- Triangle PSQ is isosceles with PS = SQ

-- Define the isosceles triangle properties
def angle_RSQ_def : ℝ := 180 - angle_QSP
def angle_RQS_def : ℝ := 0.5 * (180 - angle_RSQ)
def angle_PQS_def : ℝ := 0.5 * (180 - angle_QSP)

-- Prove the main statement
theorem angle_PQR_correct : 
  (angle_RSQ = 110) →
  (angle_RQS = 35) →
  (angle_PQS = 55) →
  (angle_PQR : ℝ) = angle_PQS + angle_RQS :=
sorry

end angle_PQR_correct_l170_170294


namespace problem_l170_170984

variable {a b c : ℝ} -- Introducing variables a, b, c as real numbers

-- Conditions:
-- a, b, c are distinct positive real numbers
def distinct_pos (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a 

theorem problem (h : distinct_pos a b c) : 
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 :=
sorry 

end problem_l170_170984


namespace sum_of_primes_final_sum_l170_170815

theorem sum_of_primes (p : ℕ) (hp : Nat.Prime p) :
  (¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 6 [ZMOD p]) →
  p = 2 ∨ p = 5 :=
sorry

theorem final_sum :
  (∀ p : ℕ, Nat.Prime p → (¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 6 [ZMOD p]) → p = 2 ∨ p = 5) →
  (2 + 5 = 7) :=
sorry

end sum_of_primes_final_sum_l170_170815


namespace plus_signs_count_l170_170015

noncomputable def solveSymbols : Prop :=
  ∃ p m : ℕ, p + m = 23 ∧ p ≤ 14 ∧ m ≤ 9 ∧ (∀ s : Finset (Fin 23), s.card = 10 → (s.image (λ i => if i < p then true else false)).count true ≥ 1) ∧ (∀ s : Finset (Fin 23), s.card = 15 → (s.image (λ i => if i < p then false else true)).count false ≥ 1) 

theorem plus_signs_count : solveSymbols :=
sorry

end plus_signs_count_l170_170015


namespace moles_Cl2_combined_l170_170523

-- Condition Definitions
def moles_C2H6 := 2
def moles_HCl_formed := 2
def balanced_reaction (C2H6 Cl2 C2H4Cl2 HCl : ℝ) : Prop :=
  C2H6 + Cl2 = C2H4Cl2 + 2 * HCl

-- Mathematical Equivalent Proof Problem Statement
theorem moles_Cl2_combined (C2H6 Cl2 HCl C2H4Cl2 : ℝ) (h1 : C2H6 = 2) 
(h2 : HCl = 2) (h3 : balanced_reaction C2H6 Cl2 C2H4Cl2 HCl) :
  Cl2 = 1 :=
by
  -- The proof is stated here.
  sorry

end moles_Cl2_combined_l170_170523


namespace distance_between_city_A_and_city_B_l170_170232

noncomputable def eddyTravelTime : ℝ := 3  -- hours
noncomputable def freddyTravelTime : ℝ := 4  -- hours
noncomputable def constantDistance : ℝ := 300  -- km
noncomputable def speedRatio : ℝ := 2  -- Eddy:Freddy

theorem distance_between_city_A_and_city_B (D_B D_C : ℝ) (h1 : D_B = (3 / 2) * D_C) (h2 : D_C = 300) :
  D_B = 450 :=
by
  sorry

end distance_between_city_A_and_city_B_l170_170232


namespace douglas_weight_proof_l170_170497

theorem douglas_weight_proof : 
  ∀ (anne_weight douglas_weight : ℕ), 
  anne_weight = 67 →
  anne_weight = douglas_weight + 15 →
  douglas_weight = 52 :=
by 
  intros anne_weight douglas_weight h1 h2 
  sorry

end douglas_weight_proof_l170_170497


namespace combination_15_5_l170_170752

theorem combination_15_5 : 
  ∀ (n r : ℕ), n = 15 → r = 5 → n.choose r = 3003 :=
by
  intro n r h1 h2
  rw [h1, h2]
  exact Nat.choose_eq_factorial_div_factorial (by norm_num)

end combination_15_5_l170_170752


namespace monotonicity_and_extremes_l170_170539

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 + x^2 - 3 * x + 1

theorem monotonicity_and_extremes :
  (∀ x, f x > f (-3) ∨ f x < f (-3)) ∧
  (∀ x, f x > f 1 ∨ f x < f 1) ∧
  (∀ x, (x < -3 → (∀ y, y < x → f y < f x)) ∧ (x > 1 → (∀ y, y > x → f y < f x))) ∧
  f (-3) = 10 ∧ f 1 = -(2 / 3) :=
sorry

end monotonicity_and_extremes_l170_170539


namespace painting_time_calculation_l170_170108

theorem painting_time_calculation :
  let doug_rate := (1 : ℚ) / 5
  let dave_rate := (1 : ℚ) / 7
  let ellen_rate := (1 : ℚ) / 9
  let combined_rate := doug_rate + dave_rate + ellen_rate

  (combined_rate * (t - 1) = 1) → t = 458 / 143 :=
by
  intros
  let doug_rate : ℚ := 1 / 5
  let dave_rate : ℚ := 1 / 7
  let ellen_rate : ℚ := 1 / 9
  let combined_rate : ℚ := doug_rate + dave_rate + ellen_rate
  let t := (458 : ℚ) / 143
  sorry

end painting_time_calculation_l170_170108


namespace opposite_of_neg2_l170_170743

theorem opposite_of_neg2 : ∃ y : ℤ, -2 + y = 0 ∧ y = 2 :=
by
  use 2
  simp
  sorry

end opposite_of_neg2_l170_170743


namespace perimeter_of_resulting_figure_l170_170702

-- Define the perimeters of the squares
def perimeter_small_square : ℕ := 40
def perimeter_large_square : ℕ := 100

-- Define the side lengths of the squares
def side_length_small_square := perimeter_small_square / 4
def side_length_large_square := perimeter_large_square / 4

-- Define the total perimeter of the uncombined squares
def total_perimeter_uncombined := perimeter_small_square + perimeter_large_square

-- Define the shared side length
def shared_side_length := side_length_small_square

-- Define the perimeter after considering the shared side
def resulting_perimeter := total_perimeter_uncombined - 2 * shared_side_length

-- Prove that the resulting perimeter is 120 cm
theorem perimeter_of_resulting_figure : resulting_perimeter = 120 := by
  sorry

end perimeter_of_resulting_figure_l170_170702


namespace intersection_A_B_l170_170151

-- Conditions
def A : Set (ℕ × ℕ) := { (1, 2), (2, 1) }
def B : Set (ℕ × ℕ) := { p | p.fst - p.snd = 1 }

-- Problem statement
theorem intersection_A_B : A ∩ B = { (2, 1) } :=
by
  sorry

end intersection_A_B_l170_170151


namespace opposite_of_neg2_l170_170745

theorem opposite_of_neg2 : ∃ y : ℤ, -2 + y = 0 ∧ y = 2 :=
by
  use 2
  simp
  sorry

end opposite_of_neg2_l170_170745


namespace number_of_terms_in_arithmetic_sequence_is_39_l170_170839

theorem number_of_terms_in_arithmetic_sequence_is_39 :
  ∀ (a d l : ℤ), 
  d ≠ 0 → 
  a = 128 → 
  d = -3 → 
  l = 14 → 
  ∃ n : ℕ, (a + (↑n - 1) * d = l) ∧ (n = 39) :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_is_39_l170_170839


namespace distinct_students_l170_170802

theorem distinct_students 
  (students_euler : ℕ) (students_gauss : ℕ) (students_fibonacci : ℕ) (overlap_euler_gauss : ℕ)
  (h_euler : students_euler = 15) 
  (h_gauss : students_gauss = 10) 
  (h_fibonacci : students_fibonacci = 12) 
  (h_overlap : overlap_euler_gauss = 3) 
  : students_euler + students_gauss + students_fibonacci - overlap_euler_gauss = 34 :=
by
  sorry

end distinct_students_l170_170802


namespace directrix_of_parabola_l170_170967

def parabola_eq (x : ℝ) : ℝ := -4 * x^2 + 4

theorem directrix_of_parabola : 
  ∃ y : ℝ, y = 65 / 16 :=
by
  sorry

end directrix_of_parabola_l170_170967


namespace jelly_bean_problem_l170_170002

theorem jelly_bean_problem 
  (x y : ℕ) 
  (h1 : x + y = 1200) 
  (h2 : x = 3 * y - 400) :
  x = 800 := 
sorry

end jelly_bean_problem_l170_170002


namespace find_p_l170_170774

theorem find_p (m n p : ℝ) :
  m = (n / 7) - (2 / 5) →
  m + p = ((n + 21) / 7) - (2 / 5) →
  p = 3 := by
  sorry

end find_p_l170_170774


namespace distinct_positive_values_count_l170_170385

theorem distinct_positive_values_count : 
  ∃ (n : ℕ), n = 33 ∧ ∀ (x : ℕ), 
    (20 ≤ x ∧ x ≤ 99 ∧ 20 ≤ 2 * x ∧ 2 * x < 200 ∧ 3 * x ≥ 200) 
    ↔ (67 ≤ x ∧ x < 100) :=
  sorry

end distinct_positive_values_count_l170_170385


namespace product_of_divisors_of_18_l170_170452

theorem product_of_divisors_of_18 : 
  ∏ d in (finset.filter (λ d, 18 % d = 0) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_of_18_l170_170452


namespace perimeter_of_smaller_rectangle_l170_170491

theorem perimeter_of_smaller_rectangle (s t u : ℝ) (h1 : 4 * s = 160) (h2 : t = s / 2) (h3 : u = t / 3) : 
    2 * (t + u) = 400 / 3 := by
  sorry

end perimeter_of_smaller_rectangle_l170_170491


namespace not_possible_identical_nonzero_remainders_l170_170776

theorem not_possible_identical_nonzero_remainders :
  ¬ ∃ (a : ℕ → ℕ) (r : ℕ), (r > 0) ∧ (∀ i : Fin 100, a i % (a ((i + 1) % 100)) = r) :=
by
  sorry

end not_possible_identical_nonzero_remainders_l170_170776


namespace depth_of_melted_ice_cream_l170_170085

theorem depth_of_melted_ice_cream (r_sphere r_cylinder : ℝ) (Vs : ℝ) (Vc : ℝ) :
  r_sphere = 3 →
  r_cylinder = 12 →
  Vs = (4 / 3) * Real.pi * r_sphere^3 →
  Vc = Real.pi * r_cylinder^2 * (1 / 4) →
  Vs = Vc →
  (1 / 4) = 1 / 4 := 
by
  intros hr_sphere hr_cylinder hVs hVc hVs_eq_Vc
  sorry

end depth_of_melted_ice_cream_l170_170085


namespace interest_rate_is_correct_l170_170522

variable (A P I : ℝ)
variable (T R : ℝ)

theorem interest_rate_is_correct
  (hA : A = 1232)
  (hP : P = 1100)
  (hT : T = 12 / 5)
  (hI : I = A - P) :
  R = I * 100 / (P * T) :=
by
  sorry

end interest_rate_is_correct_l170_170522


namespace closest_integer_to_cube_root_of_200_l170_170610

theorem closest_integer_to_cube_root_of_200 : 
  ∃ (n : ℤ), 
    (n = 6) ∧ (n^3 < 200) ∧ (200 < (n + 1)^3) ∧ 
    (∀ m : ℤ, (m^3 < 200) → (200 < (m + 1)^3) → (Int.abs (n - Int.ofNat (200 ^ (1/3 : ℝ)).round) < Int.abs (m - Int.ofNat (200 ^ (1/3 : ℝ)).round))) :=
begin
  sorry
end

end closest_integer_to_cube_root_of_200_l170_170610


namespace total_tea_consumption_l170_170327

variables (S O P : ℝ)

theorem total_tea_consumption : 
  S + O = 11 →
  P + O = 15 →
  P + S = 13 →
  S + O + P = 19.5 :=
by
  intros h1 h2 h3
  sorry

end total_tea_consumption_l170_170327


namespace accurate_mass_l170_170065

variable (m1 m2 a b x : Real) -- Declare the variables

theorem accurate_mass (h1 : a * x = b * m1) (h2 : b * x = a * m2) : x = Real.sqrt (m1 * m2) := by
  -- We will prove the statement later
  sorry

end accurate_mass_l170_170065


namespace circle_center_is_neg4_2_l170_170356

noncomputable def circle_center (x y : ℝ) : Prop :=
  x^2 + 8 * x + y^2 - 4 * y = 16

theorem circle_center_is_neg4_2 :
  ∃ (h k : ℝ), (h = -4 ∧ k = 2) ∧
  ∀ (x y : ℝ), circle_center x y ↔ (x + 4)^2 + (y - 2)^2 = 36 :=
by
  sorry

end circle_center_is_neg4_2_l170_170356


namespace sufficient_but_not_necessary_condition_l170_170325

variables (x y : ℝ)

theorem sufficient_but_not_necessary_condition :
  ((x - 1) ^ 2 + (y - 2) ^ 2 = 0) → ((x - 1) * (y - 2) = 0) ∧ (¬ ((x - 1) * (y-2) = 0 → (x - 1)^2 + (y - 2)^2 = 0)) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l170_170325


namespace ZacharysBusRideLength_l170_170602

theorem ZacharysBusRideLength (vince_ride zach_ride : ℝ) 
  (h1 : vince_ride = 0.625) 
  (h2 : vince_ride = zach_ride + 0.125) : 
  zach_ride = 0.500 := 
by
  sorry

end ZacharysBusRideLength_l170_170602


namespace smallest_x_mod_conditions_l170_170771

theorem smallest_x_mod_conditions :
  ∃ x : ℕ, x > 0 ∧ x % 5 = 4 ∧ x % 6 = 5 ∧ x % 7 = 6 ∧ x = 209 := by
  sorry

end smallest_x_mod_conditions_l170_170771


namespace find_genuine_coin_l170_170791

-- Definitions: parameterizing the setup of the problem
def num_coins : ℕ := 100
def num_fake_coins : ℕ := 4
def num_real_coins : ℕ := num_coins - num_fake_coins

-- Assume the weights of coins
parameter (weight_real : ℝ) (weight_fake : ℝ)
parameter (h1 : weight_fake < weight_real)

-- Major problem statement: proving the ability to find at least one genuine coin
theorem find_genuine_coin (coins : fin num_coins → ℝ) :
  (∃ g : fin num_coins, coins g = weight_real) →  -- There exists a genuine coin
  (∃ g : fin num_coins, coins g = weight_real) := begin
  intro h,
  sorry -- Proof to be filled in later
end

end find_genuine_coin_l170_170791


namespace harry_pencils_lost_l170_170096

-- Define the conditions
def anna_pencils : ℕ := 50
def harry_initial_pencils : ℕ := 2 * anna_pencils
def harry_current_pencils : ℕ := 81

-- Define the proof statement
theorem harry_pencils_lost :
  harry_initial_pencils - harry_current_pencils = 19 :=
by
  -- The proof is to be filled in
  sorry

end harry_pencils_lost_l170_170096


namespace fran_speed_calculation_l170_170302

theorem fran_speed_calculation:
  let Joann_speed := 15
  let Joann_time := 5
  let Fran_time := 4
  let Fran_speed := (Joann_speed * Joann_time) / Fran_time
  Fran_speed = 18.75 := by
  sorry

end fran_speed_calculation_l170_170302


namespace average_sales_l170_170317

theorem average_sales
  (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 90)
  (h2 : a2 = 50)
  (h3 : a3 = 70)
  (h4 : a4 = 110)
  (h5 : a5 = 80) :
  (a1 + a2 + a3 + a4 + a5) / 5 = 80 :=
by
  sorry

end average_sales_l170_170317


namespace plus_signs_count_l170_170011

noncomputable def symbols : Type := fin 23 -- The type of 23 symbols

def is_plus (x : symbols) : Prop
def is_minus (x : symbols) : Prop

axiom total_symbols : ∀ (x : symbols), is_plus x ∨ is_minus x
axiom unique_classification : ∀ (x : symbols), ¬ (is_plus x ∧ is_minus x)
axiom ten_symbols_plus : ∀ (S : finset symbols), S.card = 10 → ∃ x ∈ S, is_plus x
axiom fifteen_symbols_minus : ∀ (S : finset symbols), S.card = 15 → ∃ x ∈ S, is_minus x

def num_plus_symbols : ℕ := finset.card (finset.filter is_plus finset.univ)

theorem plus_signs_count : num_plus_symbols = 14 :=
sorry

end plus_signs_count_l170_170011


namespace plus_signs_count_l170_170022

theorem plus_signs_count (n p m : ℕ) (h_total : n = 23) 
  (h_plus_min : ∀ s, s.card = 10 → ∃ t ∈ s, is_plus t)
  (h_minus_min : ∀ s, s.card = 15 → ∃ t ∈ s, is_minus t)
  : p = 14 :=
sorry

end plus_signs_count_l170_170022


namespace inequality_sqrt_three_l170_170422

theorem inequality_sqrt_three (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h : a^2 + b^2 + c^2 = 1) : 
  (ab / c + bc / a + ca / b) ≥ √3 :=
by
  sorry

end inequality_sqrt_three_l170_170422


namespace exists_even_among_pythagorean_triplet_l170_170846

theorem exists_even_among_pythagorean_triplet (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ x, (x = a ∨ x = b ∨ x = c) ∧ x % 2 = 0 :=
sorry

end exists_even_among_pythagorean_triplet_l170_170846


namespace find_down_payment_l170_170872

noncomputable def purchasePrice : ℝ := 118
noncomputable def monthlyPayment : ℝ := 10
noncomputable def numberOfMonths : ℝ := 12
noncomputable def interestRate : ℝ := 0.15254237288135593
noncomputable def totalPayments : ℝ := numberOfMonths * monthlyPayment -- total amount paid through installments
noncomputable def interestPaid : ℝ := purchasePrice * interestRate -- total interest paid
noncomputable def totalPaid : ℝ := purchasePrice + interestPaid -- total amount paid including interest

theorem find_down_payment : ∃ D : ℝ, D + totalPayments = totalPaid ∧ D = 16 :=
by sorry

end find_down_payment_l170_170872


namespace length_AE_l170_170314

/-- Given points A, B, C, D, and E on a plane with distances:
  - CA = 12,
  - AB = 8,
  - BC = 4,
  - CD = 5,
  - DB = 3,
  - BE = 6,
  - ED = 3.
  Prove that AE = sqrt 113.
--/
theorem length_AE (A B C D E : ℝ × ℝ)
  (h1 : dist C A = 12)
  (h2 : dist A B = 8)
  (h3 : dist B C = 4)
  (h4 : dist C D = 5)
  (h5 : dist D B = 3)
  (h6 : dist B E = 6)
  (h7 : dist E D = 3) : 
  dist A E = Real.sqrt 113 := 
  by 
    sorry

end length_AE_l170_170314


namespace ratio_X_N_l170_170129

-- Given conditions as definitions
variables (P Q M N X : ℝ)
variables (hM : M = 0.40 * Q)
variables (hQ : Q = 0.30 * P)
variables (hN : N = 0.60 * P)
variables (hX : X = 0.25 * M)

-- Prove that X / N == 1 / 20
theorem ratio_X_N : X / N = 1 / 20 :=
by
  sorry

end ratio_X_N_l170_170129


namespace total_emails_vacation_l170_170067

def day_1_emails : ℕ := 16
def day_2_emails : ℕ := day_1_emails / 2
def day_3_emails : ℕ := day_2_emails / 2
def day_4_emails : ℕ := day_3_emails / 2

def total_emails : ℕ := day_1_emails + day_2_emails + day_3_emails + day_4_emails

theorem total_emails_vacation : total_emails = 30 := by
  -- Use "sorry" to skip the proof as per instructions.
  sorry

end total_emails_vacation_l170_170067


namespace probability_same_color_l170_170262

-- Definitions with conditions
def totalPlates := 11
def redPlates := 6
def bluePlates := 5

-- Calculate combinations
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main statement
theorem probability_same_color (totalPlates redPlates bluePlates : ℕ) (h1 : totalPlates = 11) 
(h2 : redPlates = 6) (h3 : bluePlates = 5) : 
  (2 / 11 : ℚ) = ((choose redPlates 3 + choose bluePlates 3) : ℚ) / (choose totalPlates 3) :=
by
  -- Proof steps will be inserted here
  sorry

end probability_same_color_l170_170262


namespace ellipse_foci_x_axis_l170_170135

theorem ellipse_foci_x_axis (a b : ℝ) (h : ∀ x y : ℝ, a * x^2 + b * y^2 = 1) : 0 < a ∧ a < b :=
sorry

end ellipse_foci_x_axis_l170_170135


namespace tangent_line_slope_l170_170849

theorem tangent_line_slope (k : ℝ) :
  (∃ m : ℝ, (m^3 - m^2 + m = k * m) ∧ (k = 3 * m^2 - 2 * m + 1)) →
  (k = 1 ∨ k = 3 / 4) :=
by
  -- Proof goes here
  sorry

end tangent_line_slope_l170_170849


namespace min_value_of_expression_l170_170985

theorem min_value_of_expression (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + y = 1) : 
  ∃ min_value, min_value = 9 / 2 ∧ ∀ z, z = (1 / (x + 1) + 4 / y) → z ≥ min_value :=
sorry

end min_value_of_expression_l170_170985


namespace number_of_kg_of_mangoes_l170_170909

variable {m : ℕ}
def cost_apples := 8 * 70
def cost_mangoes (m : ℕ) := 75 * m
def total_cost := 1235

theorem number_of_kg_of_mangoes (h : cost_apples + cost_mangoes m = total_cost) : m = 9 :=
by
  sorry

end number_of_kg_of_mangoes_l170_170909


namespace time_difference_leak_l170_170783

/-- 
The machine usually fills one barrel in 3 minutes. 
However, with a leak, it takes 5 minutes to fill one barrel. 
Given that it takes 24 minutes longer to fill 12 barrels with the leak, prove that it will take 2n minutes longer to fill n barrels with the leak.
-/
theorem time_difference_leak (n : ℕ) : 
  (3 * 12 + 24 = 5 * 12) →
  (5 * n) - (3 * n) = 2 * n :=
by
  intros h
  sorry

end time_difference_leak_l170_170783


namespace bread_cost_l170_170203

theorem bread_cost
  (B : ℝ)
  (cost_peanut_butter : ℝ := 2)
  (initial_money : ℝ := 14)
  (money_leftover : ℝ := 5.25) :
  3 * B + cost_peanut_butter = (initial_money - money_leftover) → B = 2.25 :=
by
  sorry

end bread_cost_l170_170203


namespace david_pushups_more_than_zachary_l170_170772

theorem david_pushups_more_than_zachary :
  ∀ (Z D J : ℕ), Z = 51 → J = 69 → J = D - 4 → D = Z + 22 :=
by
  intros Z D J hZ hJ hJD
  sorry

end david_pushups_more_than_zachary_l170_170772


namespace find_cos_value_l170_170248

open Real

noncomputable def cos_value (α : ℝ) : ℝ :=
  cos (2 * π / 3 + 2 * α)

theorem find_cos_value (α : ℝ) (h : sin (π / 6 - α) = 1 / 4) :
  cos_value α = -7 / 8 :=
sorry

end find_cos_value_l170_170248


namespace minimum_cost_l170_170555

noncomputable def total_cost (x : ℝ) : ℝ :=
  (1800 / (x + 5)) + 0.5 * x

theorem minimum_cost : 
  (∃ x : ℝ, x = 55 ∧ total_cost x = 57.5) :=
  sorry

end minimum_cost_l170_170555


namespace failing_percentage_exceeds_35_percent_l170_170407

theorem failing_percentage_exceeds_35_percent:
  ∃ (n D A B failD failA : ℕ), 
  n = 25 ∧
  D + A - B = n ∧
  (failD * 100) / D = 30 ∧
  (failA * 100) / A = 30 ∧
  ((failD + failA) * 100) / n > 35 := 
by
  sorry

end failing_percentage_exceeds_35_percent_l170_170407


namespace trapezoid_midsegment_l170_170857

-- Define the problem conditions and question
theorem trapezoid_midsegment (b h x : ℝ) (h_nonzero : h ≠ 0) (hx : x = b + 75)
  (equal_areas : (1 / 2) * (h / 2) * (b + (b + 75)) = (1 / 2) * (h / 2) * ((b + 75) + (b + 150))) :
  ∃ n : ℤ, n = ⌊x^2 / 120⌋ ∧ n = 3000 := 
by 
  sorry

end trapezoid_midsegment_l170_170857


namespace james_owns_145_l170_170411

theorem james_owns_145 (total : ℝ) (diff : ℝ) (james_and_ali : total = 250) (james_more_than_ali : diff = 40):
  ∃ (james ali : ℝ), ali + diff = james ∧ ali + james = total ∧ james = 145 :=
by
  sorry

end james_owns_145_l170_170411


namespace license_plate_increase_l170_170747

-- definitions from conditions
def old_plates_count : ℕ := 26 ^ 2 * 10 ^ 3
def new_plates_count : ℕ := 26 ^ 4 * 10 ^ 2

-- theorem stating the increase in the number of license plates
theorem license_plate_increase : 
  (new_plates_count : ℚ) / (old_plates_count : ℚ) = 26 ^ 2 / 10 :=
by
  sorry

end license_plate_increase_l170_170747


namespace unw_touchable_area_l170_170829

-- Define the conditions
def ball_radius : ℝ := 1
def container_edge_length : ℝ := 5

-- Define the surface area that the ball can never touch
theorem unw_touchable_area : (ball_radius = 1) ∧ (container_edge_length = 5) → 
  let total_unreachable_area := 120
  let overlapping_area := 24
  let unreachable_area := total_unreachable_area - overlapping_area
  unreachable_area = 96 :=
by
  intros
  sorry

end unw_touchable_area_l170_170829


namespace smallest_nonprime_greater_than_with_no_prime_factors_less_than_15_l170_170153

-- Definitions for the given conditions.
def is_prime (p : ℕ) : Prop := (p > 1) ∧ ∀ d : ℕ, d ∣ p → (d = 1 ∨ d = p)

def has_no_prime_factors_less_than (n : ℕ) (m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

def is_nonprime (n : ℕ) : Prop := ¬ is_prime n

-- The main theorem based on the proof problem.
theorem smallest_nonprime_greater_than_with_no_prime_factors_less_than_15 
  (n : ℕ) (h1 : n > 1) (h2 : has_no_prime_factors_less_than n 15) (h3 : is_nonprime n) : 
  280 < n ∧ n ≤ 290 :=
by
  sorry

end smallest_nonprime_greater_than_with_no_prime_factors_less_than_15_l170_170153


namespace find_m_abc_inequality_l170_170540

-- Define properties and the theorem for the first problem
def f (x m : ℝ) := m - |x - 2|

theorem find_m (m : ℝ) : (∀ x, f (x + 2) m ≥ 0 ↔ x ∈ Set.Icc (-1 : ℝ) 1) → m = 1 := by
  intros h
  sorry

-- Define properties and the theorem for the second problem
theorem abc_inequality (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  (1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) → (a + 2 * b + 3 * c ≥ 9) := by
  intros h
  sorry

end find_m_abc_inequality_l170_170540


namespace roundness_of_hundred_billion_l170_170653

def roundness (n : ℕ) : ℕ :=
  let pf := n.factorization
  pf 2 + pf 5

theorem roundness_of_hundred_billion : roundness 100000000000 = 22 := by
  sorry

end roundness_of_hundred_billion_l170_170653


namespace race_distance_l170_170553

theorem race_distance (d x y z : ℝ) 
  (h1 : d / x = (d - 25) / y)
  (h2 : d / y = (d - 15) / z)
  (h3 : d / x = (d - 35) / z) : 
  d = 75 := 
sorry

end race_distance_l170_170553


namespace total_tennis_balls_used_l170_170406

theorem total_tennis_balls_used :
  let rounds := [1028, 514, 257, 128, 64, 32, 16, 8, 4]
  let cans_per_game_A := 6
  let cans_per_game_B := 8
  let balls_per_can_A := 3
  let balls_per_can_B := 4
  let games_A_to_B := rounds.splitAt 4
  let total_A := games_A_to_B.1.sum * cans_per_game_A * balls_per_can_A
  let total_B := games_A_to_B.2.sum * cans_per_game_B * balls_per_can_B
  total_A + total_B = 37573 := 
by
  sorry

end total_tennis_balls_used_l170_170406


namespace total_number_of_edges_in_hexahedron_is_12_l170_170552

-- Define a hexahedron
structure Hexahedron where
  face_count : Nat
  edges_per_face : Nat
  edge_sharing : Nat

-- Total edges calculation function
def total_edges (h : Hexahedron) : Nat := (h.face_count * h.edges_per_face) / h.edge_sharing

-- The specific hexahedron (cube) in question
def cube : Hexahedron := {
  face_count := 6,
  edges_per_face := 4,
  edge_sharing := 2
}

-- The theorem to prove the number of edges in a hexahedron
theorem total_number_of_edges_in_hexahedron_is_12 : total_edges cube = 12 := by
  sorry

end total_number_of_edges_in_hexahedron_is_12_l170_170552


namespace injective_of_comp_injective_surjective_of_comp_surjective_l170_170713

section FunctionProperties

variables {X Y V : Type} (f : X → Y) (g : Y → V)

-- Proof for part (i) if g ∘ f is injective, then f is injective
theorem injective_of_comp_injective (h : Function.Injective (g ∘ f)) : Function.Injective f :=
  sorry

-- Proof for part (ii) if g ∘ f is surjective, then g is surjective
theorem surjective_of_comp_surjective (h : Function.Surjective (g ∘ f)) : Function.Surjective g :=
  sorry

end FunctionProperties

end injective_of_comp_injective_surjective_of_comp_surjective_l170_170713


namespace pascal_triangle_probability_l170_170800

-- Define the probability problem in Lean 4
theorem pascal_triangle_probability :
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  (ones_count + twos_count) / total_elements = 5 / 14 :=
by
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  have h1 : total_elements = 210 := by sorry
  have h2 : ones_count = 39 := by sorry
  have h3 : twos_count = 36 := by sorry
  have h4 : (39 + 36) / 210 = 5 / 14 := by sorry
  exact h4

end pascal_triangle_probability_l170_170800


namespace exists_point_on_graph_of_quadratic_l170_170423

-- Define the condition for the discriminant to be zero
def is_single_root (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c = 0

-- Define a function representing a quadratic polynomial
def quadratic_poly (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- The main statement
theorem exists_point_on_graph_of_quadratic (b c : ℝ) 
  (h : is_single_root 1 b c) :
  ∃ (p q : ℝ), q = (p^2) / 4 ∧ is_single_root 1 p q :=
sorry

end exists_point_on_graph_of_quadratic_l170_170423


namespace even_n_equals_identical_numbers_l170_170717

theorem even_n_equals_identical_numbers (n : ℕ) (h1 : n ≥ 2) : 
  (∃ f : ℕ → ℕ, (∀ a b, f a = f b + f b) ∧ n % 2 = 0) :=
sorry


end even_n_equals_identical_numbers_l170_170717


namespace time_to_fill_pool_l170_170580

-- Define the conditions given in the problem
def pool_volume_gallons : ℕ := 30000
def num_hoses : ℕ := 5
def hose_flow_rate_gpm : ℕ := 3

-- Define the total flow rate per minute
def total_flow_rate_gpm : ℕ := num_hoses * hose_flow_rate_gpm

-- Define the total flow rate per hour
def total_flow_rate_gph : ℕ := total_flow_rate_gpm * 60

-- Prove that the time to fill the pool is equal to 34 hours
theorem time_to_fill_pool : pool_volume_gallons / total_flow_rate_gph = 34 :=
by {
  -- Insert detailed proof steps here.
  sorry
}

end time_to_fill_pool_l170_170580


namespace total_trees_in_gray_areas_l170_170907

theorem total_trees_in_gray_areas (x y : ℕ) (h1 : 82 + x = 100) (h2 : 82 + y = 90) :
  x + y = 26 :=
by
  sorry

end total_trees_in_gray_areas_l170_170907


namespace parallelogram_side_lengths_l170_170079

theorem parallelogram_side_lengths (x y : ℚ) 
  (h1 : 12 * x - 2 = 10) 
  (h2 : 5 * y + 5 = 4) : 
  x + y = 4 / 5 := 
by 
  sorry

end parallelogram_side_lengths_l170_170079


namespace zhang_shan_sales_prediction_l170_170916

theorem zhang_shan_sales_prediction (x : ℝ) (y : ℝ) (h : x = 34) (reg_eq : y = 2 * x + 60) : y = 128 :=
by
  sorry

end zhang_shan_sales_prediction_l170_170916


namespace num_two_digit_multiples_5_and_7_l170_170545

/-- 
    Theorem: There are exactly 2 positive two-digit integers that are multiples of both 5 and 7.
-/
theorem num_two_digit_multiples_5_and_7 : 
  ∃ (count : ℕ), count = 2 ∧ ∀ (n : ℕ), (10 ≤ n ∧ n ≤ 99) → 
    (n % 5 = 0 ∧ n % 7 = 0) ↔ (n = 35 ∨ n = 70) := 
by
  sorry

end num_two_digit_multiples_5_and_7_l170_170545


namespace overall_average_score_l170_170731

variables (average_male average_female sum_male sum_female total_sum : ℕ)
variables (count_male count_female total_count : ℕ)

def average_score (sum : ℕ) (count : ℕ) : ℕ := sum / count

theorem overall_average_score
  (average_male : ℕ := 84)
  (count_male : ℕ := 8)
  (average_female : ℕ := 92)
  (count_female : ℕ := 24)
  (sum_male : ℕ := count_male * average_male)
  (sum_female : ℕ := count_female * average_female)
  (total_sum : ℕ := sum_male + sum_female)
  (total_count : ℕ := count_male + count_female) :
  average_score total_sum total_count = 90 := 
sorry

end overall_average_score_l170_170731


namespace number_that_divides_and_leaves_remainder_54_l170_170396

theorem number_that_divides_and_leaves_remainder_54 :
  ∃ n : ℕ, n > 0 ∧ (55 ^ 55 + 55) % n = 54 ∧ n = 56 :=
by
  sorry

end number_that_divides_and_leaves_remainder_54_l170_170396


namespace usual_time_to_cover_distance_l170_170603

theorem usual_time_to_cover_distance (S T : ℝ) (h1 : 0.75 * S = S / (T + 24)) (h2 : S * T = 0.75 * S * (T + 24)) : T = 72 :=
by
  sorry

end usual_time_to_cover_distance_l170_170603


namespace quadratic_function_solution_l170_170524

theorem quadratic_function_solution {a b : ℝ} :
  (∀ x : ℝ, (x^2 + (a + 1)*x + b)^2 + a*(x^2 + (a + 1)*x + b) + b = f(f x + x)) →
  (∀ x : ℝ, f(f x + x) = (f x) * (x^2 + 1776*x + 2010)) →
  a = 1774 ∧ b = 235
    → ∀ x : ℝ, f x = x^2 + 1774*x + 235 :=
begin
  -- sorry:
  sorry
end

end quadratic_function_solution_l170_170524


namespace total_sugar_in_all_candy_l170_170158

-- definitions based on the conditions
def chocolateBars : ℕ := 14
def sugarPerChocolateBar : ℕ := 10
def lollipopSugar : ℕ := 37

-- proof statement
theorem total_sugar_in_all_candy :
  (chocolateBars * sugarPerChocolateBar + lollipopSugar) = 177 := 
by
  sorry

end total_sugar_in_all_candy_l170_170158


namespace rectangle_error_percent_deficit_l170_170694

theorem rectangle_error_percent_deficit (L W : ℝ) (p : ℝ) 
    (h1 : L > 0) (h2 : W > 0)
    (h3 : 1.05 * (1 - p) = 1.008) :
    p = 0.04 :=
by
  sorry

end rectangle_error_percent_deficit_l170_170694


namespace sqrt_square_sub_sqrt2_l170_170805

theorem sqrt_square_sub_sqrt2 (h : 1 < Real.sqrt 2) : Real.sqrt ((1 - Real.sqrt 2) ^ 2) = Real.sqrt 2 - 1 :=
by 
  sorry

end sqrt_square_sub_sqrt2_l170_170805


namespace inequality_abc_l170_170424

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) := 
by 
  sorry

end inequality_abc_l170_170424


namespace sharing_watermelons_l170_170818

theorem sharing_watermelons (h : 8 = people_per_watermelon) : people_for_4_watermelons = 32 :=
by
  let people_per_watermelon := 8
  let watermelons := 4
  let people_for_4_watermelons := people_per_watermelon * watermelons
  sorry

end sharing_watermelons_l170_170818


namespace seating_arrangements_l170_170571

theorem seating_arrangements : 
  let family_members := 5 -- Total number of family members
  let driver_choices := 2 -- Choices for driver
  let front_passenger_choices := 4 -- Choices for the front passenger seat
  
  ∃ driver front_passenger backseat_arrangements,
    (driver_choices = 2) ∧
    (front_passenger_choices = 4) ∧
    (backseat_arrangements = 6) ∧
    (driver_choices * front_passenger_choices * backseat_arrangements = 48) :=
by
  -- These value assignments ensure conditions are acknowledged
  let family_members := 5
  let driver_choices := 2
  let front_passenger_choices := 4
  let backseat_arrangements := 3.choose 2 * 2.factorial
  use [driver_choices, front_passenger_choices, backseat_arrangements]
  sorry -- Proof is omitted

end seating_arrangements_l170_170571


namespace other_diagonal_length_l170_170313

theorem other_diagonal_length (d2 : ℝ) (A : ℝ) (d1 : ℝ) 
  (h1 : d2 = 120) 
  (h2 : A = 4800) 
  (h3 : A = (d1 * d2) / 2) : d1 = 80 :=
by
  sorry

end other_diagonal_length_l170_170313


namespace at_least_one_less_than_equal_one_l170_170155

theorem at_least_one_less_than_equal_one
  (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 := 
by 
  sorry

end at_least_one_less_than_equal_one_l170_170155


namespace opposite_of_neg_two_l170_170737

theorem opposite_of_neg_two : ∀ x : ℤ, (-2 + x = 0) → (x = 2) :=
begin
  assume x hx,
  sorry

end opposite_of_neg_two_l170_170737


namespace amit_work_days_l170_170645

variable (x : ℕ)

theorem amit_work_days
  (ananthu_rate : ℚ := 1/30) -- Ananthu's work rate is 1/30
  (amit_days : ℕ := 3) -- Amit worked for 3 days
  (ananthu_days : ℕ := 24) -- Ananthu worked for remaining 24 days
  (total_days : ℕ := 27) -- Total work completed in 27 days
  (amit_work: ℚ := amit_days * 1/x) -- Amit's work rate
  (ananthu_work: ℚ := ananthu_days * ananthu_rate) -- Ananthu's work rate
  (total_work : ℚ := 1) -- Total work completed  
  : 3 * (1/x) + 24 * (1/30) = 1 ↔ x = 15 := 
by
  sorry

end amit_work_days_l170_170645


namespace divisibility_problem_l170_170865

theorem divisibility_problem :
  (2^62 + 1) % (2^31 + 2^16 + 1) = 0 := 
sorry

end divisibility_problem_l170_170865


namespace polynomial_coefficient_a5_l170_170388

theorem polynomial_coefficient_a5 : 
  (∃ (a0 a1 a2 a3 a4 a5 a6 : ℝ), 
    (∀ (x : ℝ), ((2 * x - 1)^5 * (x + 2) = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6)) ∧ 
    a5 = 176) := sorry

end polynomial_coefficient_a5_l170_170388


namespace bus_full_problem_l170_170629

theorem bus_full_problem
      (cap : ℕ := 80)
      (first_pickup_ratio : ℚ := 3/5)
      (second_pickup_exit : ℕ := 15)
      (waiting_people : ℕ := 50) :
      waiting_people - (cap - (first_pickup_ratio * cap - second_pickup_exit)) = 3 := by
  sorry

end bus_full_problem_l170_170629


namespace shortest_distance_midpoint_parabola_chord_l170_170531

theorem shortest_distance_midpoint_parabola_chord
  (A B : ℝ × ℝ)
  (hA : A.1 ^ 2 = 4 * A.2)
  (hB : B.1 ^ 2 = 4 * B.2)
  (cord_length : dist A B = 6)
  : dist ((A.1 + B.1) / 2, (A.2 + B.2) / 2) (0, 0) = 2 :=
sorry

end shortest_distance_midpoint_parabola_chord_l170_170531


namespace prob_l170_170036

noncomputable def g (x : ℝ) : ℝ := 1 / (2 + 1 / (2 + 1 / x))

theorem prob (x1 x2 x3 : ℝ) (h1 : x1 = 0) 
  (h2 : 2 + 1 / x2 = 0) 
  (h3 : 2 + 1 / (2 + 1 / x3) = 0) : 
  x1 + x2 + x3 = -9 / 10 := 
sorry

end prob_l170_170036


namespace distinct_complex_numbers_no_solution_l170_170720

theorem distinct_complex_numbers_no_solution :
  ¬∃ (a b c d : ℂ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
  (a^3 - b * c * d = b^3 - c * d * a) ∧ 
  (b^3 - c * d * a = c^3 - d * a * b) ∧ 
  (c^3 - d * a * b = d^3 - a * b * c) := 
by {
  sorry
}

end distinct_complex_numbers_no_solution_l170_170720


namespace combinations_15_3_l170_170554

def num_combinations (n k : ℕ) : ℕ := n.choose k

theorem combinations_15_3 :
  num_combinations 15 3 = 455 :=
sorry

end combinations_15_3_l170_170554


namespace probability_same_color_is_correct_l170_170275

noncomputable def total_ways_select_plates : ℕ := Nat.choose 11 3
def ways_select_red_plates : ℕ := Nat.choose 6 3
def ways_select_blue_plates : ℕ := Nat.choose 5 3
noncomputable def favorable_outcomes : ℕ := ways_select_red_plates + ways_select_blue_plates
noncomputable def probability_same_color : ℚ := favorable_outcomes / total_ways_select_plates

theorem probability_same_color_is_correct :
  probability_same_color = 2/11 := 
by
  sorry

end probability_same_color_is_correct_l170_170275


namespace arithmetic_seq_proof_l170_170830

noncomputable def arithmetic_sequence : Type := ℕ → ℝ

variables (a : ℕ → ℝ) (d : ℝ)

def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

variables (a₁ a₂ a₃ a₄ : ℝ)
variables (h1 : a 1 + a 2 = 10)
variables (h2 : a 4 = a 3 + 2)
variables (h3 : is_arithmetic_seq a d)

theorem arithmetic_seq_proof :
  a 3 + a 4 = 18 :=
sorry

end arithmetic_seq_proof_l170_170830


namespace coats_leftover_l170_170337

theorem coats_leftover :
  ∀ (total_coats : ℝ) (num_boxes : ℝ),
  total_coats = 385.5 →
  num_boxes = 7.5 →
  ∃ extra_coats : ℕ, extra_coats = 3 :=
by
  intros total_coats num_boxes h1 h2
  sorry

end coats_leftover_l170_170337


namespace remainder_when_divided_by_9_l170_170620

theorem remainder_when_divided_by_9 (z : ℤ) (k : ℤ) (h : z + 3 = 9 * k) :
  z % 9 = 6 :=
sorry

end remainder_when_divided_by_9_l170_170620


namespace simplify_expression_l170_170170

theorem simplify_expression (w x : ℤ) :
  3 * w + 6 * w + 9 * w + 12 * w + 15 * w + 20 * x + 24 = 45 * w + 20 * x + 24 :=
by sorry

end simplify_expression_l170_170170


namespace find_f_at_6_5_l170_170676

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f (-x) = f x
axiom functional_equation (x : ℝ) : f (x + 2) = - (1 / f x)
axiom initial_condition (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) : f x = x - 2

theorem find_f_at_6_5 : f 6.5 = -0.5 := by
  sorry

end find_f_at_6_5_l170_170676


namespace total_length_of_river_is_80_l170_170893

-- Definitions based on problem conditions
def straight_part_length := 20
def crooked_part_length := 3 * straight_part_length
def total_length_of_river := straight_part_length + crooked_part_length

-- Theorem stating that the total length of the river is 80 miles
theorem total_length_of_river_is_80 :
  total_length_of_river = 80 := by
    -- The proof is omitted
    sorry

end total_length_of_river_is_80_l170_170893


namespace angle_PQR_eq_90_l170_170297

theorem angle_PQR_eq_90
  (R S P Q : Type)
  [IsStraightLine R S P]
  (angle_QSP : ℝ)
  (h : angle_QSP = 70) :
  ∠PQR = 90 :=
by
  sorry

end angle_PQR_eq_90_l170_170297


namespace find_ordered_pair_l170_170111

theorem find_ordered_pair (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  18 * m * n = 72 - 9 * m - 4 * n ↔ (m = 8 ∧ n = 36) := 
by 
  sorry

end find_ordered_pair_l170_170111


namespace factorize_expression_l170_170353

theorem factorize_expression (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3)^2 :=
by
  sorry

end factorize_expression_l170_170353


namespace probability_same_color_l170_170258

-- Definitions with conditions
def totalPlates := 11
def redPlates := 6
def bluePlates := 5

-- Calculate combinations
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main statement
theorem probability_same_color (totalPlates redPlates bluePlates : ℕ) (h1 : totalPlates = 11) 
(h2 : redPlates = 6) (h3 : bluePlates = 5) : 
  (2 / 11 : ℚ) = ((choose redPlates 3 + choose bluePlates 3) : ℚ) / (choose totalPlates 3) :=
by
  -- Proof steps will be inserted here
  sorry

end probability_same_color_l170_170258


namespace evaluate_expression_l170_170819

theorem evaluate_expression :
  ∀ (a b c : ℚ),
  c = b + 1 →
  b = a + 5 →
  a = 3 →
  (a + 2 ≠ 0) →
  (b - 3 ≠ 0) →
  (c + 7 ≠ 0) →
  (a + 3) * (b + 1) * (c + 9) / ((a + 2) * (b - 3) * (c + 7)) = 2.43 := 
by
  intros a b c hc hb ha h1 h2 h3
  sorry

end evaluate_expression_l170_170819


namespace alien_home_planet_people_count_l170_170094

noncomputable def alien_earth_abduction (total_abducted returned_percentage taken_to_other_planet : ℕ) : ℕ :=
  let returned := total_abducted * returned_percentage / 100
  let remaining := total_abducted - returned
  remaining - taken_to_other_planet

theorem alien_home_planet_people_count :
  alien_earth_abduction 200 80 10 = 30 :=
by
  sorry

end alien_home_planet_people_count_l170_170094


namespace same_color_probability_l170_170272

open Nat

theorem same_color_probability :
  let total_plates := 11
  let red_plates := 6
  let blue_plates := 5
  let chosen_plates := 3
  let total_ways := choose total_plates chosen_plates
  let red_ways := choose red_plates chosen_plates
  let blue_ways := choose blue_plates chosen_plates
  let same_color_ways := red_ways + blue_ways
  let probability := (same_color_ways : ℚ) / (total_ways : ℚ)
  probability = 2 / 11 := by sorry

end same_color_probability_l170_170272


namespace reciprocal_expression_equals_two_l170_170131

theorem reciprocal_expression_equals_two (x y : ℝ) (h : x * y = 1) : 
  (x + 1 / y) * (2 * y - 1 / x) = 2 := by
  sorry

end reciprocal_expression_equals_two_l170_170131


namespace simplify_and_evaluate_expression_l170_170726

variables (a b : ℚ)

theorem simplify_and_evaluate_expression : 
  (4 * (a^2 - 2 * a * b) - (3 * a^2 - 5 * a * b + 1)) = 5 :=
by
  let a := -2
  let b := (1 : ℚ) / 3
  sorry

end simplify_and_evaluate_expression_l170_170726


namespace coefficient_of_x4_in_expansion_l170_170761

theorem coefficient_of_x4_in_expansion (x : ℤ) :
  let a := 3
  let b := 2
  let n := 8
  let k := 4
  (finset.sum (finset.range (n + 1)) (λ r, binomial n r * a^r * b^(n-r) * x^r) = 
  ∑ r in finset.range (n + 1), binomial n r * a^r * b^(n - r) * x^r)

  ∑ r in finset.range (n + 1), 
    if r = k then 
      binomial n r * a^r * b^(n-r)
    else 
      0 = 90720
:= 
by
  sorry

end coefficient_of_x4_in_expansion_l170_170761


namespace corn_height_growth_l170_170505

theorem corn_height_growth (init_height week1_growth week2_growth week3_growth : ℕ)
  (h0 : init_height = 0)
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  init_height + week1_growth + week2_growth + week3_growth = 22 :=
by sorry

end corn_height_growth_l170_170505


namespace find_a_l170_170527

open Complex

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the main hypothesis: (ai / (1 - i)) = (-1 + i)
def hypothesis (a : ℂ) : Prop :=
  (a * i) / (1 - i) = -1 + i

-- Now, we state the theorem we need to prove
theorem find_a (a : ℝ) (ha : hypothesis a) : a = 2 := by
  sorry

end find_a_l170_170527


namespace total_ear_muffs_bought_l170_170939

-- Define the number of ear muffs bought before December
def ear_muffs_before_dec : ℕ := 1346

-- Define the number of ear muffs bought during December
def ear_muffs_during_dec : ℕ := 6444

-- The total number of ear muffs bought by customers
theorem total_ear_muffs_bought : ear_muffs_before_dec + ear_muffs_during_dec = 7790 :=
by
  sorry

end total_ear_muffs_bought_l170_170939


namespace find_p_l170_170920

theorem find_p (m n p : ℚ) 
  (h1 : m = n / 6 - 2 / 5)
  (h2 : m + p = (n + 18) / 6 - 2 / 5) : 
  p = 3 := 
by 
  sorry

end find_p_l170_170920


namespace round_robin_tournament_l170_170692

open Finset

noncomputable def calculateSets (n : ℕ) : ℕ :=
  2 * (n * (n - 1) * (n - 2) / 6) / 2

theorem round_robin_tournament :
  ∀ (n : ℕ), (∀ (A B C : ℤ), 1 ≤ A ∧ A < n ∧ 1 ≤ B ∧ B < n ∧ 1 ≤ C ∧ C < n ∧ A ≠ B ∧ B ≠ C ∧ C ≠ A) →
    ∀ (games_won games_lost : ℕ), 
    (∀ (team : ℕ), team < n → (games_won = 12 ∧ games_lost = 8)) →
    (n = 21) →
    (calculateSets n = 665) := by sorry

end round_robin_tournament_l170_170692


namespace hyperbola_inequality_l170_170124

-- Define point P on the hyperbola in terms of a and b
theorem hyperbola_inequality (a b : ℝ) (h : (3*a + 3*b)^2 / 9 - (a - b)^2 = 1) : |a + b| ≥ 1 :=
sorry

end hyperbola_inequality_l170_170124


namespace cubic_roots_reciprocal_sum_l170_170986

theorem cubic_roots_reciprocal_sum {α β γ : ℝ} 
  (h₁ : α + β + γ = 6)
  (h₂ : α * β + β * γ + γ * α = 11)
  (h₃ : α * β * γ = 6) :
  (1 / α^2) + (1 / β^2) + (1 / γ^2) = 49 / 36 := 
by 
  sorry

end cubic_roots_reciprocal_sum_l170_170986


namespace max_squares_covered_by_card_l170_170070

theorem max_squares_covered_by_card : 
  let checkerboard_square_size := 1
  let card_side := 2
  let card_diagonal := Real.sqrt (card_side ^ 2 + card_side ^ 2)
  ∃ n, n = 9 :=
by
  let checkerboard_square_size := 1
  let card_side := 2
  let card_diagonal := Real.sqrt (card_side ^ 2 + card_side ^ 2)
  existsi 9
  sorry

end max_squares_covered_by_card_l170_170070


namespace molecular_weight_l170_170768

noncomputable def molecular_weight_of_one_mole : ℕ → ℝ :=
  fun n => if n = 1 then 78 else n * 78

theorem molecular_weight (n: ℕ) (hn: n > 0) (condition: ∃ k: ℕ, k = 4 ∧ 312 = k * 78) :
  molecular_weight_of_one_mole n = 78 * n :=
by
  sorry

end molecular_weight_l170_170768


namespace area_intersection_l170_170255

noncomputable def set_M : Set ℂ := {z : ℂ | abs (z - 1) ≤ 1}
noncomputable def set_N : Set ℂ := {z : ℂ | complex.arg z ≥ π / 4}

theorem area_intersection (S : ℝ) :
  S = (3 / 4) * real.pi - 1 / 2 →
  ∃ z : ℂ, z ∈ set_M ∩ set_N := 
sorry

end area_intersection_l170_170255


namespace lateral_surface_area_of_prism_l170_170874

theorem lateral_surface_area_of_prism 
  (a : ℝ) (α β V : ℝ) :
  let sin (x : ℝ) := Real.sin x 
  ∃ S : ℝ,
    S = (2 * V * sin ((α + β) / 2)) / (a * sin (α / 2) * sin (β / 2)) := 
sorry

end lateral_surface_area_of_prism_l170_170874


namespace linda_savings_l170_170061

theorem linda_savings (S : ℝ) (h1 : 1 / 4 * S = 150) : S = 600 :=
sorry

end linda_savings_l170_170061


namespace max_value_function_l170_170894

theorem max_value_function (x : ℝ) (h : x < 0) : 
  ∃ y_max, (∀ x', x' < 0 → (x' + 4 / x') ≤ y_max) ∧ y_max = -4 := 
sorry

end max_value_function_l170_170894


namespace find_a_of_exponential_inverse_l170_170319

theorem find_a_of_exponential_inverse (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : ∀ x, a^x = 9 ↔ x = 2) : a = 3 := 
by
  sorry

end find_a_of_exponential_inverse_l170_170319


namespace find_f_of_2_l170_170977

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f_of_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := by
  sorry

end find_f_of_2_l170_170977


namespace distance_traveled_l170_170212

theorem distance_traveled :
  ∫ t in (3:ℝ)..(5:ℝ), (2 * t + 3 : ℝ) = 22 :=
by
  sorry

end distance_traveled_l170_170212


namespace count_distinct_m_values_l170_170870

theorem count_distinct_m_values : 
  ∃ m_values : Finset ℤ, 
  (∀ x1 x2 : ℤ, x1 * x2 = 30 → (m_values : Set ℤ) = { x1 + x2 }) ∧ 
  m_values.card = 8 :=
by
  sorry

end count_distinct_m_values_l170_170870


namespace max_popsicles_l170_170875

def popsicles : ℕ := 1
def box_3 : ℕ := 3
def box_5 : ℕ := 5
def box_10 : ℕ := 10
def cost_popsicle : ℕ := 1
def cost_box_3 : ℕ := 2
def cost_box_5 : ℕ := 3
def cost_box_10 : ℕ := 4
def budget : ℕ := 10

theorem max_popsicles : 
  ∀ (popsicle_count : ℕ) (b3_count : ℕ) (b5_count : ℕ) (b10_count : ℕ),
    popsicle_count * cost_popsicle + b3_count * cost_box_3 + b5_count * cost_box_5 + b10_count * cost_box_10 ≤ budget →
    popsicle_count * popsicles + b3_count * box_3 + b5_count * box_5 + b10_count * box_10 ≤ 23 →
    ∃ p b3 b5 b10, popsicle_count = p ∧ b3_count = b3 ∧ b5_count = b5 ∧ b10_count = b10 ∧
    (p * cost_popsicle + b3 * cost_box_3 + b5 * cost_box_5 + b10 * cost_box_10 ≤ budget) ∧
    (p * popsicles + b3 * box_3 + b5 * box_5 + b10 * box_10 = 23) :=
by sorry

end max_popsicles_l170_170875


namespace directrix_parabola_l170_170962

-- Given the equation of the parabola and required transformations:
theorem directrix_parabola (d : ℚ) : 
  (∀ x : ℚ, y = -4 * x^2 + 4) → d = 65 / 16 :=
by sorry

end directrix_parabola_l170_170962


namespace multiply_neg_reverse_inequality_l170_170228

theorem multiply_neg_reverse_inequality (a b : ℝ) (h : a < b) : -2 * a > -2 * b :=
sorry

end multiply_neg_reverse_inequality_l170_170228


namespace product_of_divisors_18_l170_170459

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l170_170459


namespace product_of_divisors_18_l170_170455

theorem product_of_divisors_18 : ∏ d in (finset.filter (∣ 18) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_18_l170_170455


namespace sequence_count_l170_170118

def num_sequences (n m : ℕ) : ℕ :=
  Nat.choose n m

theorem sequence_count :
  let x := 490
  let y := 510
  let a : (n : ℕ) → ℕ := fun n => if n = 0 then 0 else if n = 1000 then 2020 else sorry
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 1000 → (a (k + 1) - a k = 1 ∨ a (k + 1) - a k = 3)) →
  (∃ binomial_coeff, binomial_coeff = num_sequences 1000 490) :=
by sorry

end sequence_count_l170_170118


namespace problem1_problem2_l170_170808

-- Definition and conditions
def i := Complex.I

-- Problem 1
theorem problem1 : (2 + 2 * i) / (1 - i)^2 + (Real.sqrt 2 / (1 + i)) ^ 2010 = -1 := 
by
  sorry

-- Problem 2
theorem problem2 : (4 - i^5) * (6 + 2 * i^7) + (7 + i^11) * (4 - 3 * i) = 47 - 39 * i := 
by
  sorry

end problem1_problem2_l170_170808


namespace percentage_number_l170_170077

theorem percentage_number (b : ℕ) (h : b = 100) : (320 * b / 100) = 320 :=
by
  sorry

end percentage_number_l170_170077


namespace same_color_probability_l170_170270

open Nat

theorem same_color_probability :
  let total_plates := 11
  let red_plates := 6
  let blue_plates := 5
  let chosen_plates := 3
  let total_ways := choose total_plates chosen_plates
  let red_ways := choose red_plates chosen_plates
  let blue_ways := choose blue_plates chosen_plates
  let same_color_ways := red_ways + blue_ways
  let probability := (same_color_ways : ℚ) / (total_ways : ℚ)
  probability = 2 / 11 := by sorry

end same_color_probability_l170_170270


namespace math_problem_l170_170245

noncomputable def a (b : ℝ) : ℝ := 
  sorry -- to be derived from the conditions

noncomputable def b : ℝ := 
  sorry -- to be derived from the conditions

theorem math_problem (a b: ℝ) 
  (h1: a - b = 1)
  (h2: a^2 - b^2 = -1) : 
  a^2008 - b^2008 = -1 := 
sorry

end math_problem_l170_170245


namespace solution_set_of_inequality_l170_170659

theorem solution_set_of_inequality (x : ℝ) : (x + 3) * (x - 5) < 0 ↔ (-3 < x ∧ x < 5) :=
by
  sorry

end solution_set_of_inequality_l170_170659


namespace woman_work_completion_woman_days_to_complete_l170_170784

theorem woman_work_completion (M W B : ℝ) (h1 : M + W + B = 1/4) (h2 : M = 1/6) (h3 : B = 1/18) : W = 1/36 :=
by
  -- Substitute h2 and h3 into h1 and solve for W
  sorry

theorem woman_days_to_complete (W : ℝ) (h : W = 1/36) : 1 / W = 36 :=
by
  -- Calculate the reciprocal of h
  sorry

end woman_work_completion_woman_days_to_complete_l170_170784


namespace pounds_per_ton_l170_170718

theorem pounds_per_ton (packet_count : ℕ) (packet_weight_pounds : ℚ) (packet_weight_ounces : ℚ) (ounces_per_pound : ℚ) (total_weight_tons : ℚ) (total_weight_pounds : ℚ) :
  packet_count = 1760 →
  packet_weight_pounds = 16 →
  packet_weight_ounces = 4 →
  ounces_per_pound = 16 →
  total_weight_tons = 13 →
  total_weight_pounds = (packet_count * (packet_weight_pounds + (packet_weight_ounces / ounces_per_pound))) →
  total_weight_pounds / total_weight_tons = 2200 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end pounds_per_ton_l170_170718


namespace not_algebraic_expression_C_l170_170494

-- Define what it means for something to be an algebraic expression, as per given problem's conditions
def is_algebraic_expression (expr : String) : Prop :=
  expr = "A" ∨ expr = "B" ∨ expr = "D"
  
theorem not_algebraic_expression_C : ¬ (is_algebraic_expression "C") :=
by
  -- This is a placeholder; proof steps are not required per instructions
  sorry

end not_algebraic_expression_C_l170_170494


namespace car_distance_l170_170207

theorem car_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (distance : ℝ) 
  (h_speed : speed = 160) 
  (h_time : time = 5) 
  (h_dist_formula : distance = speed * time) : 
  distance = 800 :=
by sorry

end car_distance_l170_170207


namespace joel_age_when_dad_is_twice_l170_170868

-- Given Conditions
def joel_age_now : ℕ := 5
def dad_age_now : ℕ := 32
def age_difference : ℕ := dad_age_now - joel_age_now

-- Proof Problem Statement
theorem joel_age_when_dad_is_twice (x : ℕ) (hx : dad_age_now - joel_age_now = 27) : x = 27 :=
by
  sorry

end joel_age_when_dad_is_twice_l170_170868


namespace volume_of_remaining_solid_after_removing_tetrahedra_l170_170574

theorem volume_of_remaining_solid_after_removing_tetrahedra :
  let cube_volume := 1
  let tetrahedron_volume := (1/3) * (1/2) * (1/2) * (1/2) * (1/2)
  cube_volume - 8 * tetrahedron_volume = 5 / 6 := by
  let cube_volume := 1
  let tetrahedron_volume := (1/3) * (1/2) * (1/2) * (1/2) * (1/2)
  have h : cube_volume - 8 * tetrahedron_volume = 5 / 6 := sorry
  exact h

end volume_of_remaining_solid_after_removing_tetrahedra_l170_170574


namespace stratified_random_sampling_l170_170477

theorem stratified_random_sampling
 (junior_students senior_students total_sample_size : ℕ)
 (junior_high_count senior_high_count : ℕ) 
 (total_sample : junior_high_count + senior_high_count = total_sample_size)
 (junior_students_ratio senior_students_ratio : ℕ)
 (ratio : junior_students_ratio + senior_students_ratio = 1)
 (junior_condition : junior_students_ratio = 2 * senior_students_ratio)
 (students_distribution : junior_students = 400 ∧ senior_students = 200 ∧ total_sample_size = 60)
 (combination_junior : (nat.choose junior_students junior_high_count))
 (combination_senior : (nat.choose senior_students senior_high_count)) :
 combination_junior * combination_senior = nat.choose 400 40 * nat.choose 200 20 :=
by
  sorry

end stratified_random_sampling_l170_170477


namespace find_m_in_hyperbola_l170_170240

-- Define the problem in Lean 4
theorem find_m_in_hyperbola (m : ℝ) (x y : ℝ) (e : ℝ) (a_sq : ℝ := 9) (h_eq : e = 2) (h_hyperbola : x^2 / a_sq - y^2 / m = 1) : m = 27 :=
sorry

end find_m_in_hyperbola_l170_170240


namespace airline_passenger_capacity_l170_170091

def seats_per_row : Nat := 7
def rows_per_airplane : Nat := 20
def airplanes_owned : Nat := 5
def flights_per_day_per_airplane : Nat := 2

def seats_per_airplane : Nat := rows_per_airplane * seats_per_row
def total_seats : Nat := airplanes_owned * seats_per_airplane
def total_flights_per_day : Nat := airplanes_owned * flights_per_day_per_airplane
def total_passengers_per_day : Nat := total_flights_per_day * total_seats

theorem airline_passenger_capacity :
  total_passengers_per_day = 7000 := sorry

end airline_passenger_capacity_l170_170091


namespace sum_of_ratios_eq_four_l170_170383

theorem sum_of_ratios_eq_four 
  (A B C D E : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace E]
  (BD DC AE EB : ℝ)
  (h1 : BD = 2 * DC)
  (h2 : AE = 2 * EB) : 
  (BD / DC) + (AE / EB) = 4 :=
  sorry

end sum_of_ratios_eq_four_l170_170383


namespace probability_same_color_is_correct_l170_170273

noncomputable def total_ways_select_plates : ℕ := Nat.choose 11 3
def ways_select_red_plates : ℕ := Nat.choose 6 3
def ways_select_blue_plates : ℕ := Nat.choose 5 3
noncomputable def favorable_outcomes : ℕ := ways_select_red_plates + ways_select_blue_plates
noncomputable def probability_same_color : ℚ := favorable_outcomes / total_ways_select_plates

theorem probability_same_color_is_correct :
  probability_same_color = 2/11 := 
by
  sorry

end probability_same_color_is_correct_l170_170273


namespace polynomial_roots_arithmetic_progression_complex_root_l170_170958

theorem polynomial_roots_arithmetic_progression_complex_root :
  ∃ a : ℝ, (∀ (r d : ℂ), (r - d) + r + (r + d) = 9 → (r - d) * r + (r - d) * (r + d) + r * (r + d) = 30 → d^2 = -3 → 
  (r - d) * r * (r + d) = -a) → a = -12 :=
by sorry

end polynomial_roots_arithmetic_progression_complex_root_l170_170958


namespace evaluate_expression_l170_170750

theorem evaluate_expression : (1:ℤ)^10 + (-1:ℤ)^8 + (-1:ℤ)^7 + (1:ℤ)^5 = 2 := by
  sorry

end evaluate_expression_l170_170750


namespace mr_castiel_sausages_l170_170570

theorem mr_castiel_sausages (S : ℕ) :
  S * (3 / 5) * (1 / 2) * (1 / 4) * (3 / 4) = 45 → S = 600 :=
by
  sorry

end mr_castiel_sausages_l170_170570


namespace plus_signs_count_l170_170008

theorem plus_signs_count (num_symbols : ℕ) (at_least_one_plus_in_10 : ∀ s : Finset ℕ, s.card = 10 → (∃ i ∈ s, i < 14)) (at_least_one_minus_in_15 : ∀ s : Finset ℕ, s.card = 15 → (∃ i ∈ s, i ≥ 14)) : 
    ∃ (p m : ℕ), p + m = 23 ∧ p = 14 ∧ m = 9 := by
  sorry

end plus_signs_count_l170_170008


namespace simplify_pow_prod_eq_l170_170316

noncomputable def simplify_pow_prod : ℝ :=
  (256:ℝ)^(1/4) * (625:ℝ)^(1/2)

theorem simplify_pow_prod_eq :
  simplify_pow_prod = 100 := by
  sorry

end simplify_pow_prod_eq_l170_170316


namespace probability_calculation_l170_170334

def p_X := 1 / 5
def p_Y := 1 / 2
def p_Z := 5 / 8
def p_not_Z := 1 - p_Z

theorem probability_calculation : 
    (p_X * p_Y * p_not_Z) = (3 / 80) := by
    sorry

end probability_calculation_l170_170334


namespace solution_quad_ineq_l170_170445

noncomputable def quadratic_inequality_solution_set :=
  {x : ℝ | (x > -1) ∧ (x < 3) ∧ (x ≠ 2)}

theorem solution_quad_ineq (x : ℝ) :
  ((x^2 - 2*x - 3)*(x^2 - 4*x + 4) < 0) ↔ x ∈ quadratic_inequality_solution_set :=
by sorry

end solution_quad_ineq_l170_170445


namespace inequality_proof_l170_170710

theorem inequality_proof 
  {x₁ x₂ x₃ x₄ x₅ x₆ : ℝ} (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) (h₆ : x₆ > 0) :
  (x₂ / x₁)^5 + (x₄ / x₂)^5 + (x₆ / x₃)^5 + (x₁ / x₄)^5 + (x₃ / x₅)^5 + (x₅ / x₆)^5 ≥ 
  (x₁ / x₂) + (x₂ / x₄) + (x₃ / x₆) + (x₄ / x₁) + (x₅ / x₃) + (x₆ / x₅) := 
  sorry

end inequality_proof_l170_170710


namespace marbles_left_l170_170812

theorem marbles_left (initial_marbles : ℕ) (given_marbles : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 64 → given_marbles = 14 → remaining_marbles = (initial_marbles - given_marbles) → remaining_marbles = 50 :=
by
  intros h_initial h_given h_calculation
  rw [h_initial, h_given] at h_calculation
  exact h_calculation

end marbles_left_l170_170812


namespace directrix_of_parabola_l170_170969

def parabola_eq (x : ℝ) : ℝ := -4 * x^2 + 4

theorem directrix_of_parabola : 
  ∃ y : ℝ, y = 65 / 16 :=
by
  sorry

end directrix_of_parabola_l170_170969


namespace sin_arcsin_plus_arctan_l170_170955

theorem sin_arcsin_plus_arctan (a b : ℝ) (ha : a = Real.arcsin (4/5)) (hb : b = Real.arctan (1/2)) :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_arcsin_plus_arctan_l170_170955


namespace find_triangle_base_l170_170899

theorem find_triangle_base (left_side : ℝ) (right_side : ℝ) (base : ℝ) 
  (h_left : left_side = 12) 
  (h_right : right_side = left_side + 2)
  (h_sum : left_side + right_side + base = 50) :
  base = 24 := 
sorry

end find_triangle_base_l170_170899


namespace workshop_cost_l170_170525

theorem workshop_cost
  (x : ℝ)
  (h1 : 0 < x) -- Given the cost must be positive
  (h2 : (x / 4) - 15 = x / 7) :
  x = 140 :=
by
  sorry

end workshop_cost_l170_170525


namespace even_function_properties_l170_170848

theorem even_function_properties 
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_increasing : ∀ x y : ℝ, 5 ≤ x ∧ x ≤ y ∧ y ≤ 7 → f x ≤ f y)
  (h_min_value : ∀ x : ℝ, 5 ≤ x ∧ x ≤ 7 → 6 ≤ f x) :
  (∀ x y : ℝ, -7 ≤ x ∧ x ≤ y ∧ y ≤ -5 → f y ≤ f x) ∧ (∀ x : ℝ, -7 ≤ x ∧ x ≤ -5 → 6 ≤ f x) :=
by
  sorry

end even_function_properties_l170_170848


namespace magnitude_of_difference_is_3sqrt5_l170_170384

noncomputable def vector_a : ℝ × ℝ := (1, -2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 4)

def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem magnitude_of_difference_is_3sqrt5 (x : ℝ) (h_parallel : parallel vector_a (vector_b x)) :
  (Real.sqrt ((vector_a.1 - (vector_b x).1) ^ 2 + (vector_a.2 - (vector_b x).2) ^ 2)) = 3 * Real.sqrt 5 :=
sorry

end magnitude_of_difference_is_3sqrt5_l170_170384


namespace first_term_geometric_progression_l170_170748

theorem first_term_geometric_progression (S : ℝ) (sum_first_two_terms : ℝ) (a : ℝ) (r : ℝ) :
  S = 8 → sum_first_two_terms = 5 →
  (a = 8 * (1 - (Real.sqrt 6) / 4)) ∨ (a = 8 * (1 + (Real.sqrt 6) / 4)) :=
by
  sorry

end first_term_geometric_progression_l170_170748


namespace SammyFinishedProblems_l170_170428

def initial : ℕ := 9 -- number of initial math problems
def remaining : ℕ := 7 -- number of remaining math problems
def finished (init rem : ℕ) : ℕ := init - rem -- defining number of finished problems

theorem SammyFinishedProblems : finished initial remaining = 2 := by
  sorry -- placeholder for proof

end SammyFinishedProblems_l170_170428


namespace find_savings_l170_170919

-- Definitions and conditions from the problem
def income : ℕ := 36000
def ratio_income_to_expenditure : ℚ := 9 / 8
def expenditure : ℚ := 36000 * (8 / 9)
def savings : ℚ := income - expenditure

-- The theorem statement to prove
theorem find_savings : savings = 4000 := by
  sorry

end find_savings_l170_170919


namespace simplify_and_evaluate_l170_170884

theorem simplify_and_evaluate (a : ℝ) (h : a = 3) : ((2 * a / (a + 1) - 1) / ((a - 1)^2 / (a + 1))) = 1 / 2 := by
  sorry

end simplify_and_evaluate_l170_170884


namespace ellen_total_legos_l170_170352

-- Conditions
def ellen_original_legos : ℝ := 2080.0
def ellen_winning_legos : ℝ := 17.0

-- Theorem statement
theorem ellen_total_legos : ellen_original_legos + ellen_winning_legos = 2097.0 :=
by
  -- The proof would go here, but we will use sorry to indicate it is skipped.
  sorry

end ellen_total_legos_l170_170352


namespace inequality_abc_l170_170420

theorem inequality_abc (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c)
  (h₃ : a^2 + b^2 + c^2 = 1) : 
  (ab / c) + (bc / a) + (ca / b) ≥ real.sqrt 3 :=
begin
  sorry
end

end inequality_abc_l170_170420


namespace pyramid_volume_l170_170174

theorem pyramid_volume (b : ℝ) (h₀ : b > 0) :
  let base_area := (b * b * (Real.sqrt 3)) / 4
  let height := b / 2
  let volume := (1 / 3) * base_area * height
  volume = (b^3 * (Real.sqrt 3)) / 24 :=
sorry

end pyramid_volume_l170_170174


namespace problem_statement_l170_170253

-- Define the given conditions
def P : ℝ × ℝ := (4/5, -3/5)
def α : ℝ 

-- Lemmas and theorems to be proven
theorem problem_statement (h : P = (Real.cos α, Real.sin α)) :
  Real.cos α = 4/5 ∧
  Real.tan α = -3/4 ∧
  Real.sin (α + Real.pi) = 3/5 :=
by
  sorry

end problem_statement_l170_170253


namespace correct_regression_eq_l170_170444

-- Definitions related to the conditions
def negative_correlation (y x : ℝ) : Prop :=
  -- y is negatively correlated with x implies a negative slope in regression
  ∃ a b : ℝ, a < 0 ∧ ∀ x, y = a * x + b

-- The potential regression equations
def regression_eq1 (x : ℝ) : ℝ := -10 * x + 200
def regression_eq2 (x : ℝ) : ℝ := 10 * x + 200
def regression_eq3 (x : ℝ) : ℝ := -10 * x - 200
def regression_eq4 (x : ℝ) : ℝ := 10 * x - 200

-- Prove that the correct regression equation is selected given the conditions
theorem correct_regression_eq (y x : ℝ) (h : negative_correlation y x) : 
  (∀ x : ℝ, y = regression_eq1 x) ∨ (∀ x : ℝ, y = regression_eq2 x) ∨ 
  (∀ x : ℝ, y = regression_eq3 x) ∨ (∀ x : ℝ, y = regression_eq4 x) →
  ∀ x : ℝ, y = regression_eq1 x := by
  -- This theorem states that given negative correlation and the possible options, 
  -- the correct regression equation consistent with all conditions must be regression_eq1.
  sorry

end correct_regression_eq_l170_170444


namespace math_problem_l170_170714

noncomputable def proof_statement : Prop :=
  ∃ (a b m : ℝ),
    0 < a ∧ 0 < b ∧ 0 < m ∧
    (5 = m^2 * ((a^2 / b^2) + (b^2 / a^2)) + m * (a/b + b/a)) ∧
    m = (-1 + Real.sqrt 21) / 2

theorem math_problem : proof_statement :=
  sorry

end math_problem_l170_170714


namespace lindy_distance_l170_170698

theorem lindy_distance
  (d : ℝ) (v_j : ℝ) (v_c : ℝ) (v_l : ℝ) (t : ℝ)
  (h1 : d = 270)
  (h2 : v_j = 4)
  (h3 : v_c = 5)
  (h4 : v_l = 8)
  (h_time : t = d / (v_j + v_c)) :
  v_l * t = 240 := by
  sorry

end lindy_distance_l170_170698


namespace complex_number_properties_l170_170835

open Complex

noncomputable def z : ℂ := (1 - I) / I

theorem complex_number_properties :
  z ^ 2 = 2 * I ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_number_properties_l170_170835


namespace f_characterization_l170_170814

noncomputable def op (a b : ℝ) := a * b

noncomputable def ot (a b : ℝ) := a + b

noncomputable def f (x : ℝ) := ot x 2 - op 2 x

-- Prove that f(x) is neither odd nor even and is a decreasing function
theorem f_characterization :
  (∀ x : ℝ, f x = -x + 2) ∧
  (∀ x : ℝ, f (-x) ≠ f x ∧ f (-x) ≠ -f x) ∧
  (∀ x y : ℝ, x < y → f x > f y) := sorry

end f_characterization_l170_170814


namespace find_three_digit_number_l170_170959

def is_three_digit_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000

def digits_sum (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  a + b + c

theorem find_three_digit_number : 
  ∃ n : ℕ, is_three_digit_number n ∧ n^2 = (digits_sum n)^5 ∧ n = 243 :=
sorry

end find_three_digit_number_l170_170959


namespace find_x_l170_170617

theorem find_x (x : ℤ) (h : (1 + 2 + 4 + 5 + 6 + 9 + 9 + 10 + 12 + x) / 10 = 7) : x = 12 :=
by
  sorry

end find_x_l170_170617


namespace problem_statement_l170_170993

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -6 < x ∧ x < 1}

theorem problem_statement : M ∩ N = N := by
  ext x
  constructor
  · intro h
    exact h.2
  · intro h
    exact ⟨h.2, h⟩

end problem_statement_l170_170993


namespace band_section_student_count_l170_170335

theorem band_section_student_count :
  (0.5 * 500) + (0.12 * 500) + (0.23 * 500) + (0.08 * 500) = 465 :=
by 
  sorry

end band_section_student_count_l170_170335


namespace quintic_polynomial_p_l170_170285

theorem quintic_polynomial_p (p q : ℝ) (h : (∀ x : ℝ, x^p + 4*x^3 - q*x^2 - 2*x + 5 = (x^5 + 4*x^3 - q*x^2 - 2*x + 5))) : -p = -5 :=
by {
  sorry
}

end quintic_polynomial_p_l170_170285


namespace max_non_colored_cubes_l170_170601

open Nat

-- Define the conditions
def isRectangularPrism (length width height volume : ℕ) := length * width * height = volume

-- The theorem stating the equivalent math proof problem
theorem max_non_colored_cubes (length width height : ℕ) (h₁ : isRectangularPrism length width height 1024) :
(length > 2 ∧ width > 2 ∧ height > 2) → (length - 2) * (width - 2) * (height - 2) = 504 := by
  sorry

end max_non_colored_cubes_l170_170601


namespace find_function_expression_l170_170836

variable (f : ℝ → ℝ)
variable (P : ℝ → ℝ → ℝ)

-- conditions
axiom a1 : f 1 = 1
axiom a2 : ∀ (x y : ℝ), f (x + y) = f x + f y + 2 * y * (x + y) + 1

-- proof statement
theorem find_function_expression (x : ℕ) (h : x ≠ 0) : f x = x^2 + 3*x - 3 := sorry

end find_function_expression_l170_170836


namespace average_is_equal_l170_170889

theorem average_is_equal (x : ℝ) :
  (1 / 3) * (2 * x + 4 + 5 * x + 3 + 3 * x + 8) = 3 * x - 5 → 
  x = -30 :=
by
  sorry

end average_is_equal_l170_170889


namespace volume_surface_ratio_l170_170927

-- Define the structure of the shape
structure Shape where
  center_cube : unit
  surrounding_cubes : Fin 6 -> unit
  top_cube : unit

-- Define the properties for the calculation
def volume (s : Shape) : ℕ := 8
def surface_area (s : Shape) : ℕ := 28
def ratio_volume_surface_area (s : Shape) : ℚ := volume s / surface_area s

-- Main theorem statement
theorem volume_surface_ratio (s : Shape) : ratio_volume_surface_area s = 2 / 7 := sorry

end volume_surface_ratio_l170_170927


namespace inequality_proof_l170_170249

variable (m n : ℝ)

theorem inequality_proof (hm : m < 0) (hn : n > 0) (h_sum : m + n < 0) : m < -n ∧ -n < n ∧ n < -m :=
by
  -- introduction and proof commands would go here, but we use sorry to indicate the proof is omitted
  sorry

end inequality_proof_l170_170249


namespace tan2α_sin_β_l170_170982

open Real

variables {α β : ℝ}

axiom α_acute : 0 < α ∧ α < π / 2
axiom β_acute : 0 < β ∧ β < π / 2
axiom sin_α : sin α = 4 / 5
axiom cos_alpha_beta : cos (α + β) = 5 / 13

theorem tan2α : tan 2 * α = -24 / 7 :=
by sorry

theorem sin_β : sin β = 16 / 65 :=
by sorry

end tan2α_sin_β_l170_170982


namespace polygon_sides_l170_170587

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
sorry

end polygon_sides_l170_170587


namespace cost_of_article_l170_170683

theorem cost_of_article (C: ℝ) (G: ℝ) (h1: 380 = C + G) (h2: 420 = C + G + 0.05 * C) : C = 800 :=
by
  sorry

end cost_of_article_l170_170683


namespace email_difference_l170_170559

def morning_emails_early : ℕ := 10
def morning_emails_late : ℕ := 15
def afternoon_emails_early : ℕ := 7
def afternoon_emails_late : ℕ := 12

theorem email_difference :
  (morning_emails_early + morning_emails_late) - (afternoon_emails_early + afternoon_emails_late) = 6 :=
by
  sorry

end email_difference_l170_170559


namespace requiredSheetsOfPaper_l170_170935

-- Define the conditions
def englishAlphabetLetters : ℕ := 26
def timesWrittenPerLetter : ℕ := 3
def sheetsOfPaperPerLetter (letters : ℕ) (times : ℕ) : ℕ := letters * times

-- State the theorem equivalent to the original math problem
theorem requiredSheetsOfPaper : sheetsOfPaperPerLetter englishAlphabetLetters timesWrittenPerLetter = 78 := by
  sorry

end requiredSheetsOfPaper_l170_170935


namespace boat_capacity_per_trip_l170_170817

theorem boat_capacity_per_trip (trips_per_day : ℕ) (total_people : ℕ) (days : ℕ) :
  trips_per_day = 4 → total_people = 96 → days = 2 → (total_people / (trips_per_day * days)) = 12 :=
by
  intros
  sorry

end boat_capacity_per_trip_l170_170817


namespace Robie_chocolates_left_l170_170882

def initial_bags : ℕ := 3
def given_away : ℕ := 2
def additional_bags : ℕ := 3

theorem Robie_chocolates_left : (initial_bags - given_away) + additional_bags = 4 :=
by
  sorry

end Robie_chocolates_left_l170_170882


namespace total_passengers_per_day_l170_170090

-- Define the conditions
def airplanes : ℕ := 5
def rows_per_airplane : ℕ := 20
def seats_per_row : ℕ := 7
def flights_per_day : ℕ := 2

-- Define the proof problem
theorem total_passengers_per_day : 
  (airplanes * rows_per_airplane * seats_per_row * flights_per_day) = 1400 := 
by 
  sorry

end total_passengers_per_day_l170_170090


namespace sum_of_numbers_l170_170320

theorem sum_of_numbers (a b c : ℝ) 
  (h₁ : (a + b + c) / 3 = a + 20) 
  (h₂ : (a + b + c) / 3 = c - 30) 
  (h₃ : b = 10) : 
  a + b + c = 60 := 
by
  sorry

end sum_of_numbers_l170_170320


namespace minimum_x_plus_y_l170_170371

theorem minimum_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) 
    (h1 : x - y < 1) (h2 : 2 * x - y > 2) (h3 : x < 5) : 
    x + y ≥ 6 :=
sorry

end minimum_x_plus_y_l170_170371


namespace solid_ball_performance_l170_170487

theorem solid_ball_performance :
    ∃ (x : ℝ), y = - (1 / 12 : ℝ) * x^2 + (2 / 3 : ℝ) * x + (5 / 3 : ℝ) ∧ y = 0 ∧ x = 10 :=
by
  sorry

end solid_ball_performance_l170_170487


namespace kylie_coins_count_l170_170305

theorem kylie_coins_count 
  (P : ℕ) 
  (from_brother : ℕ) 
  (from_father : ℕ) 
  (given_to_Laura : ℕ) 
  (coins_left : ℕ) 
  (h1 : from_brother = 13) 
  (h2 : from_father = 8) 
  (h3 : given_to_Laura = 21) 
  (h4 : coins_left = 15) : (P + from_brother + from_father) - given_to_Laura = coins_left → P = 15 :=
by
  sorry

end kylie_coins_count_l170_170305


namespace total_points_l170_170332

theorem total_points (paul_points cousin_points : ℕ) 
  (h_paul : paul_points = 3103) 
  (h_cousin : cousin_points = 2713) : 
  paul_points + cousin_points = 5816 := by
sorry

end total_points_l170_170332


namespace scout_weekend_earnings_l170_170169

theorem scout_weekend_earnings : 
  let base_pay_per_hour := 10.00
  let tip_per_customer := 5.00
  let saturday_hours := 4
  let saturday_customers := 5
  let sunday_hours := 5
  let sunday_customers := 8
  in
  (saturday_hours * base_pay_per_hour + saturday_customers * tip_per_customer) +
  (sunday_hours * base_pay_per_hour + sunday_customers * tip_per_customer) = 155.00 := sorry

end scout_weekend_earnings_l170_170169


namespace inequality_proof_l170_170708

theorem inequality_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  a^a * b^b + a^b * b^a ≤ 1 :=
  sorry

end inequality_proof_l170_170708


namespace smallest_natural_number_l170_170442

theorem smallest_natural_number (a : ℕ) : 
  (∃ a, a % 3 = 0 ∧ (a - 1) % 4 = 0 ∧ (a - 2) % 5 = 0) → a = 57 :=
by
  sorry

end smallest_natural_number_l170_170442


namespace plus_signs_count_l170_170013

noncomputable def symbols : Type := fin 23 -- The type of 23 symbols

def is_plus (x : symbols) : Prop
def is_minus (x : symbols) : Prop

axiom total_symbols : ∀ (x : symbols), is_plus x ∨ is_minus x
axiom unique_classification : ∀ (x : symbols), ¬ (is_plus x ∧ is_minus x)
axiom ten_symbols_plus : ∀ (S : finset symbols), S.card = 10 → ∃ x ∈ S, is_plus x
axiom fifteen_symbols_minus : ∀ (S : finset symbols), S.card = 15 → ∃ x ∈ S, is_minus x

def num_plus_symbols : ℕ := finset.card (finset.filter is_plus finset.univ)

theorem plus_signs_count : num_plus_symbols = 14 :=
sorry

end plus_signs_count_l170_170013


namespace sum_gcd_lcm_l170_170608

theorem sum_gcd_lcm (a b : ℕ) (ha : a = 45) (hb : b = 4095) :
    Nat.gcd a b + Nat.lcm a b = 4140 :=
by
  sorry

end sum_gcd_lcm_l170_170608


namespace angle_PQR_is_90_l170_170296

theorem angle_PQR_is_90 {P Q R S : Type}
  (is_straight_line_RSP : ∃ P R S : Type, (angle R S P = 180)) 
  (angle_QSP : angle Q S P = 70)
  (isosceles_RS_SQ : ∃ (RS SQ : Type), RS = SQ)
  (isosceles_PS_SQ : ∃ (PS SQ : Type), PS = SQ) : angle P Q R = 90 :=
by 
  sorry

end angle_PQR_is_90_l170_170296


namespace number_of_possible_digits_to_make_divisible_by_4_l170_170632

def four_digit_number_divisible_by_4 (N : ℕ) : Prop :=
  let number := N * 1000 + 264
  number % 4 = 0

theorem number_of_possible_digits_to_make_divisible_by_4 :
  ∃ (count : ℕ), count = 10 ∧ (∀ (N : ℕ), N < 10 → four_digit_number_divisible_by_4 N) :=
by {
  sorry
}

end number_of_possible_digits_to_make_divisible_by_4_l170_170632


namespace volleyballs_count_l170_170724

theorem volleyballs_count 
  (total_balls soccer_balls : ℕ)
  (basketballs tennis_balls baseballs volleyballs : ℕ) 
  (h_total : total_balls = 145) 
  (h_soccer : soccer_balls = 20) 
  (h_basketballs : basketballs = soccer_balls + 5)
  (h_tennis : tennis_balls = 2 * soccer_balls) 
  (h_baseballs : baseballs = soccer_balls + 10) 
  (h_specific_total : soccer_balls + basketballs + tennis_balls + baseballs = 115): 
  volleyballs = 30 := 
by 
  have h_specific_balls : soccer_balls + basketballs + tennis_balls + baseballs = 115 :=
    h_specific_total
  have total_basketballs : basketballs = 25 :=
    by rw [h_basketballs, h_soccer]; refl
  have total_tennis_balls : tennis_balls = 40 :=
    by rw [h_tennis, h_soccer]; refl
  have total_baseballs : baseballs = 30 :=
    by rw [h_baseballs, h_soccer]; refl
  have total_specific_balls : 20 + 25 + 40 + 30 = 115 :=
    by norm_num
  have volleyballs_to_find : volleyballs = 145 - 115 :=
    by rw [h_total]; exact rfl
  sorry

end volleyballs_count_l170_170724


namespace smallest_element_in_M_l170_170639

def f : ℝ → ℝ := sorry
axiom f1 (x y : ℝ) (h1 : x ≥ 1) (h2 : y = 3 * x) : f y = 3 * f x
axiom f2 (x : ℝ) (h : 1 ≤ x ∧ x ≤ 3) : f x = 1 - abs (x - 2)
axiom f99_value : f 99 = 18

theorem smallest_element_in_M : ∃ x : ℝ, x = 45 ∧ f x = 18 := by
  -- proof will be provided later
  sorry

end smallest_element_in_M_l170_170639


namespace find_f_f_neg1_l170_170528

def f (x : Int) : Int :=
  if x >= 0 then x + 2 else 1

theorem find_f_f_neg1 : f (f (-1)) = 3 :=
by
  sorry

end find_f_f_neg1_l170_170528


namespace paper_plates_cost_l170_170185

theorem paper_plates_cost (P C x : ℝ) 
(h1 : 100 * P + 200 * C = 6.00) 
(h2 : x * P + 40 * C = 1.20) : 
x = 20 := 
sorry

end paper_plates_cost_l170_170185


namespace range_of_b_l170_170838

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := Real.exp x * (x*x - b*x)

theorem range_of_b (b : ℝ) : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, 0 < (Real.exp x * ((x*x + (2 - b) * x - b)))) →
  b < 8/3 := 
sorry

end range_of_b_l170_170838


namespace a1_divides_a2_and_a2_divides_a3_probability_l170_170415

noncomputable def probability_a1_divides_a2_and_a2_divides_a3 : ℚ :=
  let n := 21
  let favorable := Nat.choose (n + 1) 3
  let total := n ^ 6
  (favorable * favorable) / total

theorem a1_divides_a2_and_a2_divides_a3_probability :
  probability_a1_divides_a2_and_a2_divides_a3 = 2371600 / 85766121 :=
by
  -- Calculation and combination of results are directly stated
  sorry

end a1_divides_a2_and_a2_divides_a3_probability_l170_170415


namespace men_with_6_boys_work_l170_170624

theorem men_with_6_boys_work (m b : ℚ) (x : ℕ) :
  2 * m + 4 * b = 1 / 4 →
  x * m + 6 * b = 1 / 3 →
  2 * b = 5 * m →
  x = 1 :=
by
  intros h1 h2 h3
  sorry

end men_with_6_boys_work_l170_170624


namespace right_handed_players_total_l170_170312

def total_players : ℕ := 64
def throwers : ℕ := 37
def non_throwers : ℕ := total_players - throwers
def left_handed_non_throwers : ℕ := non_throwers / 3
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers
def total_right_handed : ℕ := throwers + right_handed_non_throwers

theorem right_handed_players_total : total_right_handed = 55 := by
  sorry

end right_handed_players_total_l170_170312


namespace plus_signs_count_l170_170017

noncomputable def solveSymbols : Prop :=
  ∃ p m : ℕ, p + m = 23 ∧ p ≤ 14 ∧ m ≤ 9 ∧ (∀ s : Finset (Fin 23), s.card = 10 → (s.image (λ i => if i < p then true else false)).count true ≥ 1) ∧ (∀ s : Finset (Fin 23), s.card = 15 → (s.image (λ i => if i < p then false else true)).count false ≥ 1) 

theorem plus_signs_count : solveSymbols :=
sorry

end plus_signs_count_l170_170017


namespace remaining_bollards_to_be_installed_l170_170643

-- Definitions based on the problem's conditions
def total_bollards_per_side := 4000
def sides := 2
def total_bollards := total_bollards_per_side * sides
def installed_fraction := 3 / 4
def installed_bollards := installed_fraction * total_bollards
def remaining_bollards := total_bollards - installed_bollards

-- The statement to be proved
theorem remaining_bollards_to_be_installed : remaining_bollards = 2000 :=
  by sorry

end remaining_bollards_to_be_installed_l170_170643


namespace probability_same_color_is_correct_l170_170280

-- Definitions of the conditions
def num_red : ℕ := 6
def num_blue : ℕ := 5
def total_plates : ℕ := num_red + num_blue
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probability statement
def prob_three_same_color : ℚ :=
  let total_ways := choose total_plates 3
  let ways_red := choose num_red 3
  let ways_blue := choose num_blue 3
  let favorable_ways := ways_red
  favorable_ways / total_ways

theorem probability_same_color_is_correct : prob_three_same_color = (4 : ℚ) / 33 := sorry

end probability_same_color_is_correct_l170_170280


namespace incorrect_judgment_D_l170_170117

theorem incorrect_judgment_D (p q : Prop) (hp : p = (2 + 3 = 5)) (hq : q = (5 < 4)) : 
  ¬((p ∧ q) ∧ (p ∨ q)) := by 
    sorry

end incorrect_judgment_D_l170_170117


namespace scout_weekend_earnings_l170_170167

theorem scout_weekend_earnings
  (base_pay_per_hour : ℕ)
  (tip_per_delivery : ℕ)
  (hours_worked_saturday : ℕ)
  (deliveries_saturday : ℕ)
  (hours_worked_sunday : ℕ)
  (deliveries_sunday : ℕ)
  (total_earnings : ℕ)
  (h_base_pay : base_pay_per_hour = 10)
  (h_tip : tip_per_delivery = 5)
  (h_hours_sat : hours_worked_saturday = 4)
  (h_deliveries_sat : deliveries_saturday = 5)
  (h_hours_sun : hours_worked_sunday = 5)
  (h_deliveries_sun : deliveries_sunday = 8) :
  total_earnings = 155 :=
by
  sorry

end scout_weekend_earnings_l170_170167


namespace probability_same_color_is_correct_l170_170278

-- Definitions of the conditions
def num_red : ℕ := 6
def num_blue : ℕ := 5
def total_plates : ℕ := num_red + num_blue
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probability statement
def prob_three_same_color : ℚ :=
  let total_ways := choose total_plates 3
  let ways_red := choose num_red 3
  let ways_blue := choose num_blue 3
  let favorable_ways := ways_red
  favorable_ways / total_ways

theorem probability_same_color_is_correct : prob_three_same_color = (4 : ℚ) / 33 := sorry

end probability_same_color_is_correct_l170_170278


namespace scout_weekend_earnings_l170_170165

-- Define the constants and conditions
def base_pay : ℝ := 10.0
def tip_per_delivery : ℝ := 5.0
def saturday_hours : ℝ := 4.0
def sunday_hours : ℝ := 5.0
def saturday_deliveries : ℝ := 5.0
def sunday_deliveries : ℝ := 8.0

-- Calculate total hours worked
def total_hours : ℝ := saturday_hours + sunday_hours

-- Calculate base pay for the weekend
def total_base_pay : ℝ := total_hours * base_pay

-- Calculate total number of deliveries
def total_deliveries : ℝ := saturday_deliveries + sunday_deliveries

-- Calculate total earnings from tips
def total_tips : ℝ := total_deliveries * tip_per_delivery

-- Calculate total earnings
def total_earnings : ℝ := total_base_pay + total_tips

-- Theorem to prove the total earnings is $155.00
theorem scout_weekend_earnings : total_earnings = 155.0 := by
  sorry

end scout_weekend_earnings_l170_170165


namespace original_contribution_amount_l170_170859

theorem original_contribution_amount (F : ℕ) (N : ℕ) (C : ℕ) (A : ℕ) 
  (hF : F = 14) (hN : N = 19) (hC : C = 4) : A = 90 :=
by 
  sorry

end original_contribution_amount_l170_170859


namespace max_value_of_square_diff_max_value_of_square_diff_achieved_l170_170828

theorem max_value_of_square_diff (a b : ℝ) (h : a^2 + b^2 = 4) : (a - b)^2 ≤ 8 :=
sorry

theorem max_value_of_square_diff_achieved (a b : ℝ) (h : a^2 + b^2 = 4) : ∃ a b : ℝ, (a - b)^2 = 8 :=
sorry

end max_value_of_square_diff_max_value_of_square_diff_achieved_l170_170828


namespace exponentiation_multiplication_l170_170648

theorem exponentiation_multiplication (a : ℝ) : a^6 * a^2 = a^8 :=
by sorry

end exponentiation_multiplication_l170_170648


namespace directrix_parabola_l170_170961

-- Given the equation of the parabola and required transformations:
theorem directrix_parabola (d : ℚ) : 
  (∀ x : ℚ, y = -4 * x^2 + 4) → d = 65 / 16 :=
by sorry

end directrix_parabola_l170_170961


namespace sqrt_of_neg_five_squared_l170_170586

theorem sqrt_of_neg_five_squared : Real.sqrt ((-5 : ℝ)^2) = 5 ∨ Real.sqrt ((-5 : ℝ)^2) = -5 :=
by
  sorry

end sqrt_of_neg_five_squared_l170_170586


namespace sin_c_eq_tan_b_find_side_length_c_l170_170688

-- (1) Prove that sinC = tanB
theorem sin_c_eq_tan_b {a b c : ℝ} {C : ℝ} (h1 : a / b = 1 + Real.cos C) : 
  Real.sin C = Real.tan B := by
  sorry

-- (2) If given conditions, find the value of c
theorem find_side_length_c {a b c : ℝ} {B C : ℝ} 
  (h1 : Real.cos B = 2 * Real.sqrt 7 / 7)
  (h2 : 0 < C ∧ C < Real.pi / 2)
  (h3 : 1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) 
  : c = Real.sqrt 7 := by
  sorry

end sin_c_eq_tan_b_find_side_length_c_l170_170688


namespace clara_cookies_l170_170941

theorem clara_cookies (x : ℕ) :
  50 * 12 + x * 20 + 70 * 16 = 3320 → x = 80 :=
by
  sorry

end clara_cookies_l170_170941


namespace tan_sum_property_l170_170239

theorem tan_sum_property (t23 t37 : ℝ) (h1 : 23 + 37 = 60) (h2 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3) :
  Real.tan (23 * Real.pi / 180) + Real.tan (37 * Real.pi / 180) + Real.sqrt 3 * Real.tan (23 * Real.pi / 180) * Real.tan (37 * Real.pi / 180) = Real.sqrt 3 :=
sorry

end tan_sum_property_l170_170239


namespace plus_signs_count_l170_170025

theorem plus_signs_count (p m : ℕ) (h_sum : p + m = 23)
                         (h_max_minus : m ≤ 9) (h_max_plus : p ≤ 14)
                         (h_at_least_one_plus_in_10 : ∀ (s : Finset (Fin 23)), s.card = 10 → ∃ i ∈ s, i < p)
                         (h_at_least_one_minus_in_15 : ∀ (s : Finset (Fin 23)), s.card = 15 → ∃ i ∈ s, m ≤ i) :
  p = 14 :=
by sorry

end plus_signs_count_l170_170025


namespace min_people_in_photographs_l170_170403

-- Definitions based on conditions
def photographs := (List (Nat × Nat × Nat))
def menInCenter (photos : photographs) := photos.map (fun (c, _, _) => c)

-- Condition: there are 10 photographs each with a distinct man in the center
def valid_photographs (photos: photographs) :=
  photos.length = 10 ∧ photos.map (fun (c, _, _) => c) = List.range 10

-- Theorem to be proved: The minimum number of different people in the photographs is at least 16
theorem min_people_in_photographs (photos: photographs) (h : valid_photographs photos) : 
  ∃ people : Finset Nat, people.card ≥ 16 := 
sorry

end min_people_in_photographs_l170_170403


namespace sector_area_l170_170732

theorem sector_area (α : ℝ) (l : ℝ) (r : ℝ) :
  α = 2 ∧ l = 4 ∧ l = α * r → (1 / 2) * α * r^2 = 1 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  rw [h1, h3] at h2
  have r_eq_1 : r = 1 := by linarith
  rw r_eq_1 at *
  simp
  exact h1

end sector_area_l170_170732


namespace monotonic_function_range_maximum_value_condition_function_conditions_l170_170123

-- Part (1): Monotonicity condition
theorem monotonic_function_range (m : ℝ) :
  (∀ x : ℝ, deriv (fun x => (m - 3) * x^3 + 9 * x) x ≥ 0) ↔ (m ≥ 3) := sorry

-- Part (2): Maximum value condition
theorem maximum_value_condition (m : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (m - 3) * 8 + 18 = 4) ↔ (m = -2) := sorry

-- Combined statement (optional if you want to show entire problem in one go)
theorem function_conditions (m : ℝ) :
  (∀ x : ℝ, deriv (fun x => (m - 3) * x^3 + 9 * x) x ≥ 0 ∧ 
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (m - 3) * 8 + 18 = 4)) ↔ (m = -2 ∨ m ≥ 3) := sorry

end monotonic_function_range_maximum_value_condition_function_conditions_l170_170123


namespace geometric_sequence_first_term_l170_170943

theorem geometric_sequence_first_term (a r : ℝ) (h1 : a * r^2 = 18) (h2 : a * r^4 = 72) : a = 4.5 := by
  sorry

end geometric_sequence_first_term_l170_170943


namespace cracker_calories_l170_170947

theorem cracker_calories (cc : ℕ) (hc1 : ∀ (n : ℕ), n = 50 → cc = 50) (hc2 : ∀ (n : ℕ), n = 7 → 7 * 50 = 350) (hc3 : ∀ (n : ℕ), n = 10 * cc → 10 * cc = 10 * cc) (hc4 : 350 + 10 * cc = 500) : cc = 15 :=
by
  sorry

end cracker_calories_l170_170947


namespace collinear_points_b_value_l170_170945

theorem collinear_points_b_value (b : ℝ)
    (h : let slope1 := (4 - (-6)) / ((2 * b + 1) - 4)
         let slope2 := (1 - (-6)) / ((-3 * b + 2) - 4)
         slope1 = slope2) :
    b = -1 / 44 :=
by
  have slope1 := (4 - (-6)) / ((2 * b + 1) - 4)
  have slope2 := (1 - (-6)) / ((-3 * b + 2) - 4)
  have := h
  sorry

end collinear_points_b_value_l170_170945


namespace ellipse_standard_equation_midpoint_trajectory_equation_l170_170672

theorem ellipse_standard_equation :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ (∀ x y, (x, y) = (2, 0) → x^2 / a^2 + y^2 / b^2 = 1) → (a = 2 ∧ b = 1) :=
sorry

theorem midpoint_trajectory_equation :
  ∀ x y : ℝ,
  (∃ x0 y0 : ℝ, x0 = 2 * x - 1 ∧ y0 = 2 * y - 1 / 2 ∧ (x0^2 / 4 + y0^2 = 1)) →
  (x - 1 / 2)^2 + 4 * (y - 1 / 4)^2 = 1 :=
sorry

end ellipse_standard_equation_midpoint_trajectory_equation_l170_170672


namespace plus_signs_count_l170_170003

theorem plus_signs_count (total_symbols : ℕ) (n_plus n_minus : ℕ)
  (h1 : total_symbols = 23)
  (h2 : ∀ s : Finset ℕ, s.card = 10 → ∃ i ∈ s, i ≤ n_plus)
  (h3 : ∀ s : Finset ℕ, s.card = 15 → ∃ i ∈ s, i > n_plus) :
  n_plus = 14 :=
by
  -- the proof will go here
  sorry

end plus_signs_count_l170_170003


namespace plus_signs_count_l170_170014

noncomputable def symbols : Type := fin 23 -- The type of 23 symbols

def is_plus (x : symbols) : Prop
def is_minus (x : symbols) : Prop

axiom total_symbols : ∀ (x : symbols), is_plus x ∨ is_minus x
axiom unique_classification : ∀ (x : symbols), ¬ (is_plus x ∧ is_minus x)
axiom ten_symbols_plus : ∀ (S : finset symbols), S.card = 10 → ∃ x ∈ S, is_plus x
axiom fifteen_symbols_minus : ∀ (S : finset symbols), S.card = 15 → ∃ x ∈ S, is_minus x

def num_plus_symbols : ℕ := finset.card (finset.filter is_plus finset.univ)

theorem plus_signs_count : num_plus_symbols = 14 :=
sorry

end plus_signs_count_l170_170014


namespace spots_allocation_l170_170083

theorem spots_allocation : ∃ (n: ℕ) (k: ℕ), n = 9 ∧ k = 7 ∧ nat.choose n k = 36 := 
by {
  have n := 9,
  have k := 2,
  refine ⟨n, k, rfl, rfl, nat.choose n k = 36⟩,
  -- proof omitted
  sorry
}

end spots_allocation_l170_170083


namespace count_special_sum_integers_eq_11_l170_170344

def is_special_fraction (a b : ℕ) : Prop := a + b = 19 ∧ a > 0 ∧ b > 0

noncomputable def special_fractions : Finset ℚ :=
  Finset.univ.filter_map (λ p : ℕ × ℕ, if is_special_fraction p.1 p.2 then some (p.1 / p.2 : ℚ) else none)

noncomputable def special_sum_integers : Finset ℤ :=
  (Finset.product special_fractions special_fractions).image (λ pq, (pq.1 + pq.2).floor)

theorem count_special_sum_integers_eq_11 : special_sum_integers.card = 11 := 
  sorry

end count_special_sum_integers_eq_11_l170_170344


namespace magician_guarantee_three_of_clubs_l170_170076

-- Definitions corresponding to the identified conditions
def deck_size : ℕ := 52
def num_discarded : ℕ := 51
def magician_choice (n : ℕ) (from_left : bool) : ℕ := if from_left then n else deck_size + 1 - n
def is_edge_position (position : ℕ) : Prop := position = 1 ∨ position = deck_size

-- Statement of the problem, translated to Lean
theorem magician_guarantee_three_of_clubs (initial_pos : ℕ) (H : is_edge_position initial_pos) :
    ∃ strategy, ∀ spectator_choice, 
                (∃ remaining_cards, 
                  remaining_cards = deck_size - num_discarded + 1 ∧ 
                  three_of_clubs ∈ remaining_cards) :=
begin
  -- proving strategy exists if the initial position is at the edge
  sorry
end

end magician_guarantee_three_of_clubs_l170_170076


namespace alyssa_puppies_l170_170936

theorem alyssa_puppies (initial now given : ℕ) (h1 : initial = 12) (h2 : now = 5) : given = 7 :=
by
  have h3 : given = initial - now := by sorry
  rw [h1, h2] at h3
  exact h3

end alyssa_puppies_l170_170936


namespace calculate_total_cost_l170_170500

noncomputable def sandwich_cost : ℕ := 4
noncomputable def soda_cost : ℕ := 3
noncomputable def num_sandwiches : ℕ := 7
noncomputable def num_sodas : ℕ := 8
noncomputable def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem calculate_total_cost : total_cost = 52 := by
  sorry

end calculate_total_cost_l170_170500


namespace total_passengers_per_day_l170_170089

-- Define the conditions
def airplanes : ℕ := 5
def rows_per_airplane : ℕ := 20
def seats_per_row : ℕ := 7
def flights_per_day : ℕ := 2

-- Define the proof problem
theorem total_passengers_per_day : 
  (airplanes * rows_per_airplane * seats_per_row * flights_per_day) = 1400 := 
by 
  sorry

end total_passengers_per_day_l170_170089


namespace plus_signs_count_l170_170024

theorem plus_signs_count (p m : ℕ) (h_sum : p + m = 23)
                         (h_max_minus : m ≤ 9) (h_max_plus : p ≤ 14)
                         (h_at_least_one_plus_in_10 : ∀ (s : Finset (Fin 23)), s.card = 10 → ∃ i ∈ s, i < p)
                         (h_at_least_one_minus_in_15 : ∀ (s : Finset (Fin 23)), s.card = 15 → ∃ i ∈ s, m ≤ i) :
  p = 14 :=
by sorry

end plus_signs_count_l170_170024


namespace correctness_of_propositions_l170_170000

-- Definitions of the conditions
def residual_is_random_error (e : ℝ) : Prop := ∃ (y : ℝ) (y_hat : ℝ), e = y - y_hat
def data_constraints (a b c d : ℕ) : Prop := a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 ∧ d ≥ 5
def histogram_judgement : Prop := ∀ (H : Type) (rel : H → H → Prop), ¬(H ≠ H) ∨ (∀ x y : H, rel x y ↔ true)

-- The mathematical equivalence proof problem
theorem correctness_of_propositions (e : ℝ) (a b c d : ℕ) : 
  (residual_is_random_error e → false) ∧
  (data_constraints a b c d → true) ∧
  (histogram_judgement → true) :=
by
  sorry

end correctness_of_propositions_l170_170000


namespace probability_same_color_plates_l170_170267

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem probability_same_color_plates :
  (choose 6 3 : ℚ) / (choose 11 3 : ℚ) = 4 / 33 := by
  sorry

end probability_same_color_plates_l170_170267


namespace count_fractions_single_digit_less_than_1_l170_170544

theorem count_fractions_single_digit_less_than_1 : 
  let single_digit_nat := {n : ℕ | n < 10}
  let valid_fractions := {f : ℚ | 
    ∃ (n d : ℕ), n < d ∧ d ∈ single_digit_nat ∧ n ∈ single_digit_nat ∧ f = (n : ℚ) / (d : ℚ)}
  is_number_of_fractions := valid_fractions.card = 36 :=
by
  sorry

end count_fractions_single_digit_less_than_1_l170_170544


namespace radius_of_sector_l170_170831

theorem radius_of_sector (l : ℝ) (α : ℝ) (R : ℝ) (h1 : l = 2 * π / 3) (h2 : α = π / 3) : R = 2 := by
  have : l = |α| * R := by sorry
  rw [h1, h2] at this
  sorry

end radius_of_sector_l170_170831


namespace smallest_number_groups_l170_170464

theorem smallest_number_groups (x : ℕ) (h₁ : x % 18 = 0) (h₂ : x % 45 = 0) : x = 90 :=
sorry

end smallest_number_groups_l170_170464


namespace stratified_sampling_total_results_l170_170476

theorem stratified_sampling_total_results :
  let junior_students := 400
  let senior_students := 200
  let total_students_to_sample := 60
  let junior_sample := 40
  let senior_sample := 20
  (Nat.choose junior_students junior_sample) * (Nat.choose senior_students senior_sample) = Nat.choose 400 40 * Nat.choose 200 20 :=
  sorry

end stratified_sampling_total_results_l170_170476


namespace quadratic_complete_square_l170_170896

theorem quadratic_complete_square (a b c : ℝ) :
  (8*x^2 - 48*x - 288) = a*(x + b)^2 + c → a + b + c = -355 := 
  by
  sorry

end quadratic_complete_square_l170_170896


namespace chess_tournament_l170_170594

-- Define the number of chess amateurs
def num_amateurs : ℕ := 5

-- Define the number of games each amateur plays
def games_per_amateur : ℕ := 4

-- Define the total number of chess games possible
def total_games : ℕ := num_amateurs * (num_amateurs - 1) / 2

-- The main statement to prove
theorem chess_tournament : total_games = 10 := 
by
  -- here should be the proof, but according to the task, we use sorry to skip
  sorry

end chess_tournament_l170_170594


namespace power_modulo_l170_170463

theorem power_modulo (k : ℕ) : 7^32 % 19 = 1 → 7^2050 % 19 = 11 :=
by {
  sorry
}

end power_modulo_l170_170463


namespace trigonometry_problem_l170_170252

theorem trigonometry_problem
  (α : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (4 / 5, -3 / 5))
  (h_unit : P.1^2 + P.2^2 = 1) :
  (cos α = 4 / 5) ∧
  (tan α = -3 / 4) ∧
  (sin (α + π) = 3 / 5) ∧
  (cos (α - π / 2) ≠ 3 / 5) := by
    sorry

end trigonometry_problem_l170_170252


namespace range_of_m_l170_170376

theorem range_of_m {x m : ℝ} (h : ∀ x, x^2 - 2*x + 2*m - 1 ≥ 0) : m ≥ 1 :=
sorry

end range_of_m_l170_170376


namespace largest_y_coordinate_of_degenerate_ellipse_l170_170176

theorem largest_y_coordinate_of_degenerate_ellipse :
  ∀ (x y : ℝ), (x^2 / 36 + (y + 5)^2 / 16 = 0) → y = -5 :=
by
  intros x y h
  sorry

end largest_y_coordinate_of_degenerate_ellipse_l170_170176


namespace min_value_expression_l170_170237

theorem min_value_expression (x : ℝ) : 
  (\frac{x^2 + 9}{real_sqrt (x^2 + 5)} ≥ \frac{9 * real_sqrt 5}{5}) :=
sorry

end min_value_expression_l170_170237


namespace stones_in_pile_l170_170592

theorem stones_in_pile (initial_stones : ℕ) (final_stones_A : ℕ) (final_stones_B_min final_stones_B_max final_stones_B : ℕ) (operations : ℕ) :
  initial_stones = 2006 ∧ final_stones_A = 1990 ∧ final_stones_B_min = 2080 ∧ final_stones_B_max = 2100 ∧ operations < 20 ∧ (final_stones_B_min ≤ final_stones_B ∧ final_stones_B ≤ final_stones_B_max) 
  → final_stones_B = 2090 :=
by
  sorry

end stones_in_pile_l170_170592


namespace problem_l170_170414

noncomputable def M (x y z : ℝ) : ℝ :=
  (Real.sqrt (x^2 + x * y + y^2) * Real.sqrt (y^2 + y * z + z^2)) +
  (Real.sqrt (y^2 + y * z + z^2) * Real.sqrt (z^2 + z * x + x^2)) +
  (Real.sqrt (z^2 + z * x + x^2) * Real.sqrt (x^2 + x * y + y^2))

theorem problem (x y z : ℝ) (α β : ℝ) 
  (h1 : ∀ x y z, α * (x * y + y * z + z * x) ≤ M x y z)
  (h2 : ∀ x y z, M x y z ≤ β * (x^2 + y^2 + z^2)) :
  (∀ α, α ≤ 3) ∧ (∀ β, β ≥ 3) :=
sorry

end problem_l170_170414


namespace negation_equiv_l170_170583

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x^2 + 1 ≥ 1

-- Negation of the original proposition
def negated_prop : Prop := ∃ x : ℝ, x^2 + 1 < 1

-- Main theorem stating the equivalence
theorem negation_equiv :
  (¬ (∀ x : ℝ, original_prop x)) ↔ negated_prop :=
by sorry

end negation_equiv_l170_170583


namespace count_of_plus_signs_l170_170031

-- Definitions based on the given conditions
def total_symbols : ℕ := 23
def any_10_symbols_contains_plus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 10 → (\exists c ∈ s, c = '+')
def any_15_symbols_contains_minus (symbols : Finset char) : Prop := ∀ s : Finset char, s.card = 15 → (\exists c ∈ s, c = '-')

-- Main theorem statement without the proof
theorem count_of_plus_signs (symbols : Finset char) (h1 : symbols.card = total_symbols) 
  (h2 : any_10_symbols_contains_plus symbols) 
  (h3 : any_15_symbols_contains_minus symbols) : 
  symbols.count '+' = 14 :=
sorry

end count_of_plus_signs_l170_170031


namespace max_teams_tie_for_most_wins_l170_170292

theorem max_teams_tie_for_most_wins (n : ℕ) (h : n = 7) :
  ∃ k, k = 6 ∧ ∀ t : Finset ℕ, t.card = n → ⟨t.filter (λ x, wins x = 3)⟩.card = k :=
by sorry

end max_teams_tie_for_most_wins_l170_170292


namespace original_volume_of_cube_l170_170873

theorem original_volume_of_cube (a : ℕ) 
  (h1 : (a + 2) * (a - 2) * (a + 3) = a^3 - 7) : 
  a = 3 :=
by sorry

end original_volume_of_cube_l170_170873


namespace tabs_in_all_browsers_l170_170299

-- Definitions based on conditions
def windows_per_browser := 3
def tabs_per_window := 10
def number_of_browsers := 2

-- Total tabs calculation
def total_tabs := number_of_browsers * (windows_per_browser * tabs_per_window)

-- Proving the total number of tabs is 60
theorem tabs_in_all_browsers : total_tabs = 60 := by
  sorry

end tabs_in_all_browsers_l170_170299


namespace probability_of_one_or_two_in_pascal_l170_170793

def pascal_triangle_element_probability : ℚ :=
  let total_elements := 210 -- sum of the elements in the first 20 rows
  let ones_count := 39      -- total count of 1s in the first 20 rows
  let twos_count := 36      -- total count of 2s in the first 20 rows
  let favorable_elements := ones_count + twos_count
  favorable_elements / total_elements

theorem probability_of_one_or_two_in_pascal (n : ℕ) (h : n = 20) :
  pascal_triangle_element_probability = 5 / 14 := by
  rw [h]
  dsimp [pascal_triangle_element_probability]
  sorry

end probability_of_one_or_two_in_pascal_l170_170793


namespace final_concentration_of_milk_l170_170626

variable (x : ℝ) (total_vol : ℝ) (initial_milk : ℝ)
axiom x_value : x = 33.333333333333336
axiom total_volume : total_vol = 100
axiom initial_milk_vol : initial_milk = 36

theorem final_concentration_of_milk :
  let first_removal := x / total_vol * initial_milk
  let remaining_milk_after_first := initial_milk - first_removal
  let second_removal := x / total_vol * remaining_milk_after_first
  let final_milk := remaining_milk_after_first - second_removal
  (final_milk / total_vol) * 100 = 16 :=
by {
  sorry
}

end final_concentration_of_milk_l170_170626


namespace greatest_common_multiple_of_10_and_15_lt_120_l170_170763

theorem greatest_common_multiple_of_10_and_15_lt_120 : 
  ∃ (m : ℕ), lcm 10 15 = 30 ∧ m ∈ {i | i < 120 ∧ ∃ (k : ℕ), i = k * 30} ∧ m = 90 := 
sorry

end greatest_common_multiple_of_10_and_15_lt_120_l170_170763


namespace lucas_can_afford_book_l170_170157

-- Definitions from the conditions
def book_cost : ℝ := 28.50
def two_ten_dollar_bills : ℝ := 2 * 10
def five_one_dollar_bills : ℝ := 5 * 1
def six_quarters : ℝ := 6 * 0.25
def nickel_value : ℝ := 0.05

-- Given the conditions, we need to prove that if Lucas has at least 40 nickels, he can afford the book.
theorem lucas_can_afford_book (m : ℝ) (h : m >= 40) : 
  (two_ten_dollar_bills + five_one_dollar_bills + six_quarters + m * nickel_value) >= book_cost :=
by {
  sorry
}

end lucas_can_afford_book_l170_170157


namespace shaded_area_eq_63_l170_170043

noncomputable def rect1_width : ℕ := 4
noncomputable def rect1_height : ℕ := 12
noncomputable def rect2_width : ℕ := 5
noncomputable def rect2_height : ℕ := 7
noncomputable def overlap_width : ℕ := 4
noncomputable def overlap_height : ℕ := 5

theorem shaded_area_eq_63 :
  (rect1_width * rect1_height) + (rect2_width * rect2_height) - (overlap_width * overlap_height) = 63 := by
  sorry

end shaded_area_eq_63_l170_170043


namespace child_to_grandmother_ratio_l170_170485

variable (G D C : ℝ)

axiom condition1 : G + D + C = 150
axiom condition2 : D + C = 60
axiom condition3 : D = 42

theorem child_to_grandmother_ratio : (C / G) = (1 / 5) :=
by
  sorry

end child_to_grandmother_ratio_l170_170485


namespace largest_possible_P10_l170_170150

noncomputable def P (x : ℤ) : ℤ := x^2 + 3*x + 3

theorem largest_possible_P10 : P 10 = 133 := by
  sorry

end largest_possible_P10_l170_170150


namespace range_of_m_l170_170121

noncomputable def point := (ℝ × ℝ)
noncomputable def P : point := (-1, 1)
noncomputable def Q : point := (2, 2)
noncomputable def M : point := (0, -1)
noncomputable def line_eq (m : ℝ) := ∀ p : point, p.1 + m * p.2 + m = 0

theorem range_of_m (m : ℝ) (l : line_eq m) : -3 < m ∧ m < -2/3 := 
by
  sorry

end range_of_m_l170_170121


namespace zero_function_is_uniq_l170_170354

theorem zero_function_is_uniq (f : ℝ → ℝ) :
  (∀ (x : ℝ) (hx : x ≠ 0) (y : ℝ), f (x^2 + y) ≥ (1/x + 1) * f y) → 
  (∀ x, f x = 0) :=
by
  sorry

end zero_function_is_uniq_l170_170354


namespace calculate_fraction_l170_170997

theorem calculate_fraction (x y : ℚ) (h1 : x = 5 / 6) (h2 : y = 6 / 5) : (1 / 3) * x^8 * y^9 = 2 / 5 := by
  sorry

end calculate_fraction_l170_170997


namespace a6_b6_gt_a4b2_ab4_l170_170840

theorem a6_b6_gt_a4b2_ab4 {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) :
  a^6 + b^6 > a^4 * b^2 + a^2 * b^4 :=
sorry

end a6_b6_gt_a4b2_ab4_l170_170840


namespace volume_of_orange_concentrate_l170_170780

theorem volume_of_orange_concentrate
  (h_jug : ℝ := 8) -- height of the jug in inches
  (d_jug : ℝ := 3) -- diameter of the jug in inches
  (fraction_full : ℝ := 3 / 4) -- jug is three-quarters full
  (ratio_concentrate_to_water : ℝ := 1 / 5) -- ratio of concentrate to water
  : abs ((fraction_full * π * ((d_jug / 2)^2) * h_jug * (1 / (1 + ratio_concentrate_to_water))) - 2.25) < 0.01 :=
by
  sorry

end volume_of_orange_concentrate_l170_170780


namespace total_balls_donated_l170_170215

def num_elem_classes_A := 4
def num_middle_classes_A := 5
def num_elem_classes_B := 5
def num_middle_classes_B := 3
def num_elem_classes_C := 6
def num_middle_classes_C := 4
def balls_per_class := 5

theorem total_balls_donated :
  (num_elem_classes_A + num_middle_classes_A) * balls_per_class +
  (num_elem_classes_B + num_middle_classes_B) * balls_per_class +
  (num_elem_classes_C + num_middle_classes_C) * balls_per_class =
  135 :=
by
  sorry

end total_balls_donated_l170_170215


namespace factorization_1_min_value_l170_170576

-- Problem 1: Prove that m² - 4mn + 3n² = (m - 3n)(m - n)
theorem factorization_1 (m n : ℤ) : m^2 - 4*m*n + 3*n^2 = (m - 3*n)*(m - n) :=
by
  sorry

-- Problem 2: Prove that the minimum value of m² - 3m + 2015 is 2012 3/4
theorem min_value (m : ℝ) : ∃ x : ℝ, x = m^2 - 3*m + 2015 ∧ x = 2012 + 3/4 :=
by
  sorry

end factorization_1_min_value_l170_170576


namespace certain_number_l170_170843

theorem certain_number (a x : ℝ) (h1 : a / x * 2 = 12) (h2 : x = 0.1) : a = 0.6 := 
by
  sorry

end certain_number_l170_170843


namespace find_x_minus_y_l170_170550

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end find_x_minus_y_l170_170550


namespace arithmetic_sequence_of_condition_l170_170243

variables {R : Type*} [LinearOrderedRing R]

theorem arithmetic_sequence_of_condition (x y z : R) (h : (z-x)^2 - 4*(x-y)*(y-z) = 0) : 2*y = x + z :=
sorry

end arithmetic_sequence_of_condition_l170_170243
