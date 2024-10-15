import Mathlib

namespace NUMINAMATH_GPT_find_a6_l856_85678

noncomputable def a (n : ℕ) : ℝ := sorry

axiom geom_seq_inc :
  ∀ n : ℕ, a n < a (n + 1)

axiom root_eqn_a2_a4 :
  ∃ a2 a4 : ℝ, (a 2 = a2) ∧ (a 4 = a4) ∧ (a2^2 - 6 * a2 + 5 = 0) ∧ (a4^2 - 6 * a4 + 5 = 0)

theorem find_a6 : a 6 = 25 := 
sorry

end NUMINAMATH_GPT_find_a6_l856_85678


namespace NUMINAMATH_GPT_misread_system_of_equations_solutions_l856_85648

theorem misread_system_of_equations_solutions (a b : ℤ) (x₁ y₁ x₂ y₂ : ℤ)
  (h1 : x₁ = -3) (h2 : y₁ = -1) (h3 : x₂ = 5) (h4 : y₂ = 4)
  (eq1 : a * x₂ + 5 * y₂ = 15)
  (eq2 : 4 * x₁ - b * y₁ = -2) :
  a = -1 ∧ b = 10 ∧ a ^ 2023 + (- (1 / 10 : ℚ) * b) ^ 2023 = -2 := by
  -- Translate misreading conditions into theorems we need to prove (note: skipping proof).
  have hb : b = 10 := by sorry
  have ha : a = -1 := by sorry
  exact ⟨ha, hb, by simp [ha, hb]; norm_num⟩

end NUMINAMATH_GPT_misread_system_of_equations_solutions_l856_85648


namespace NUMINAMATH_GPT_inequality_am_gm_l856_85622

variable {u v : ℝ}

theorem inequality_am_gm (hu : 0 < u) (hv : 0 < v) : u ^ 3 + v ^ 3 ≥ u ^ 2 * v + v ^ 2 * u := by
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l856_85622


namespace NUMINAMATH_GPT_cubes_with_even_red_faces_count_l856_85646

def block_dimensions : ℕ × ℕ × ℕ := (6, 4, 2)
def is_painted_red : Prop := true
def total_cubes : ℕ := 48
def cubes_with_even_red_faces : ℕ := 24

theorem cubes_with_even_red_faces_count :
  ∀ (dimensions : ℕ × ℕ × ℕ) (painted_red : Prop) (cubes_count : ℕ), 
  dimensions = block_dimensions → painted_red = is_painted_red → cubes_count = total_cubes → 
  (cubes_with_even_red_faces = 24) :=
by intros dimensions painted_red cubes_count h1 h2 h3; exact sorry

end NUMINAMATH_GPT_cubes_with_even_red_faces_count_l856_85646


namespace NUMINAMATH_GPT_combined_selling_price_correct_l856_85664

noncomputable def cost_A : ℝ := 500
noncomputable def cost_B : ℝ := 800
noncomputable def profit_A_perc : ℝ := 0.10
noncomputable def profit_B_perc : ℝ := 0.15
noncomputable def tax_perc : ℝ := 0.05
noncomputable def packaging_fee : ℝ := 50

-- Calculating selling prices before tax and fees
noncomputable def selling_price_A_before_tax_fees : ℝ := cost_A * (1 + profit_A_perc)
noncomputable def selling_price_B_before_tax_fees : ℝ := cost_B * (1 + profit_B_perc)

-- Calculating taxes
noncomputable def tax_A : ℝ := selling_price_A_before_tax_fees * tax_perc
noncomputable def tax_B : ℝ := selling_price_B_before_tax_fees * tax_perc

-- Adding tax to selling prices
noncomputable def selling_price_A_incl_tax : ℝ := selling_price_A_before_tax_fees + tax_A
noncomputable def selling_price_B_incl_tax : ℝ := selling_price_B_before_tax_fees + tax_B

-- Adding packaging and shipping fees
noncomputable def final_selling_price_A : ℝ := selling_price_A_incl_tax + packaging_fee
noncomputable def final_selling_price_B : ℝ := selling_price_B_incl_tax + packaging_fee

-- Combined selling price
noncomputable def combined_selling_price : ℝ := final_selling_price_A + final_selling_price_B

theorem combined_selling_price_correct : 
  combined_selling_price = 1643.5 := by
  sorry

end NUMINAMATH_GPT_combined_selling_price_correct_l856_85664


namespace NUMINAMATH_GPT_unique_B_for_A47B_divisible_by_7_l856_85643

-- Define the conditions
def A : ℕ := 4

-- Define the main proof problem statement
theorem unique_B_for_A47B_divisible_by_7 : 
  ∃! B : ℕ, B ≤ 9 ∧ (100 * A + 70 + B) % 7 = 0 :=
        sorry

end NUMINAMATH_GPT_unique_B_for_A47B_divisible_by_7_l856_85643


namespace NUMINAMATH_GPT_rational_abs_eq_l856_85693

theorem rational_abs_eq (a : ℚ) (h : |-3 - a| = 3 + |a|) : 0 ≤ a := 
by
  sorry

end NUMINAMATH_GPT_rational_abs_eq_l856_85693


namespace NUMINAMATH_GPT_find_value_l856_85642

theorem find_value (a b : ℝ) (h : a + b + 1 = -2) : (a + b - 1) * (1 - a - b) = -16 := by
  sorry

end NUMINAMATH_GPT_find_value_l856_85642


namespace NUMINAMATH_GPT_david_boxes_l856_85616

-- Conditions
def number_of_dogs_per_box : ℕ := 4
def total_number_of_dogs : ℕ := 28

-- Problem
theorem david_boxes : total_number_of_dogs / number_of_dogs_per_box = 7 :=
by
  sorry

end NUMINAMATH_GPT_david_boxes_l856_85616


namespace NUMINAMATH_GPT_vertical_asymptote_l856_85699

theorem vertical_asymptote (x : ℝ) : 
  (∃ x, 4 * x + 5 = 0) → x = -5/4 :=
by 
  sorry

end NUMINAMATH_GPT_vertical_asymptote_l856_85699


namespace NUMINAMATH_GPT_leigh_path_length_l856_85610

theorem leigh_path_length :
  let north := 10
  let south := 40
  let west := 60
  let east := 20
  let net_south := south - north
  let net_west := west - east
  let distance := Real.sqrt (net_south^2 + net_west^2)
  distance = 50 := 
by sorry

end NUMINAMATH_GPT_leigh_path_length_l856_85610


namespace NUMINAMATH_GPT_hiking_packing_weight_l856_85672

theorem hiking_packing_weight
  (miles_per_hour : ℝ)
  (hours_per_day : ℝ)
  (days : ℝ)
  (supply_per_mile : ℝ)
  (resupply_fraction : ℝ)
  (expected_first_pack_weight : ℝ)
  (hiking_hours : ℝ := hours_per_day * days)
  (total_miles : ℝ := miles_per_hour * hiking_hours)
  (total_weight_needed : ℝ := supply_per_mile * total_miles)
  (resupply_weight : ℝ := resupply_fraction * total_weight_needed)
  (first_pack_weight : ℝ := total_weight_needed - resupply_weight) :
  first_pack_weight = 37.5 :=
by
  -- The proof goes here, but is omitted since the proof is not required.
  sorry

end NUMINAMATH_GPT_hiking_packing_weight_l856_85672


namespace NUMINAMATH_GPT_hexagon_diagonals_l856_85632

theorem hexagon_diagonals : (6 * (6 - 3)) / 2 = 9 := 
by 
  sorry

end NUMINAMATH_GPT_hexagon_diagonals_l856_85632


namespace NUMINAMATH_GPT_range_of_k_l856_85631

-- Definitions for the condition
def inequality_holds (k : ℝ) : Prop :=
  ∀ x : ℝ, x^4 + (k-1)*x^2 + 1 ≥ 0

-- Theorem statement
theorem range_of_k (k : ℝ) : inequality_holds k → k ≥ 1 :=
sorry

end NUMINAMATH_GPT_range_of_k_l856_85631


namespace NUMINAMATH_GPT_max_value_of_a1_l856_85618

theorem max_value_of_a1 (a1 a2 a3 a4 a5 a6 a7 : ℕ) (h_distinct : ∀ i j, i ≠ j → (i ≠ a1 → i ≠ a2 → i ≠ a3 → i ≠ a4 → i ≠ a5 → i ≠ a6 → i ≠ a7)) 
  (h_sum : a1 + a2 + a3 + a4 + a5 + a6 + a7 = 159) : a1 ≤ 19 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_a1_l856_85618


namespace NUMINAMATH_GPT_molecular_weight_is_62_024_l856_85611

def atomic_weight_H : ℝ := 1.008
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

def num_atoms_H : ℕ := 2
def num_atoms_C : ℕ := 1
def num_atoms_O : ℕ := 3

def molecular_weight_compound : ℝ :=
  num_atoms_H * atomic_weight_H + num_atoms_C * atomic_weight_C + num_atoms_O * atomic_weight_O

theorem molecular_weight_is_62_024 :
  molecular_weight_compound = 62.024 :=
by
  have h_H := num_atoms_H * atomic_weight_H
  have h_C := num_atoms_C * atomic_weight_C
  have h_O := num_atoms_O * atomic_weight_O
  have h_sum := h_H + h_C + h_O
  show molecular_weight_compound = 62.024
  sorry

end NUMINAMATH_GPT_molecular_weight_is_62_024_l856_85611


namespace NUMINAMATH_GPT_min_value_reciprocal_sum_l856_85667

theorem min_value_reciprocal_sum (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) : 
  (∃ c, (∀ x y, x > 0 → y > 0 → x + y = 1 → (1/x + 1/y) ≥ c) ∧ (1/a + 1/b = c)) 
:= 
sorry

end NUMINAMATH_GPT_min_value_reciprocal_sum_l856_85667


namespace NUMINAMATH_GPT_number_of_meters_sold_l856_85660

-- Define the given conditions
def price_per_meter : ℕ := 436 -- in kopecks
def total_revenue_end : ℕ := 728 -- in kopecks
def max_total_revenue : ℕ := 50000 -- in kopecks

-- State the problem formally in Lean 4
theorem number_of_meters_sold (x : ℕ) :
  price_per_meter * x ≡ total_revenue_end [MOD 1000] ∧
  price_per_meter * x ≤ max_total_revenue →
  x = 98 :=
sorry

end NUMINAMATH_GPT_number_of_meters_sold_l856_85660


namespace NUMINAMATH_GPT_jordon_machine_number_l856_85619

theorem jordon_machine_number : 
  ∃ x : ℝ, (2 * x + 3 = 27) ∧ x = 12 :=
by
  sorry

end NUMINAMATH_GPT_jordon_machine_number_l856_85619


namespace NUMINAMATH_GPT_find_angle_A_l856_85652

theorem find_angle_A (A B C : ℝ) (a b c : ℝ) 
  (h : 1 + (Real.tan A / Real.tan B) = 2 * c / b) : 
  A = Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_find_angle_A_l856_85652


namespace NUMINAMATH_GPT_smallest_c_value_l856_85662

theorem smallest_c_value (c d : ℝ) (h_nonneg_c : 0 ≤ c) (h_nonneg_d : 0 ≤ d)
  (h_eq_cos : ∀ x : ℤ, Real.cos (c * x - d) = Real.cos (35 * x)) :
  c = 35 := by
  sorry

end NUMINAMATH_GPT_smallest_c_value_l856_85662


namespace NUMINAMATH_GPT_smallest_n_l856_85644

theorem smallest_n (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000)
  (h2 : n % 9 = 2)
  (h3 : n % 6 = 4) : n = 146 :=
sorry

end NUMINAMATH_GPT_smallest_n_l856_85644


namespace NUMINAMATH_GPT_no_valid_N_for_case1_valid_N_values_for_case2_l856_85613

variable (P R N : ℕ)
variable (N_less_than_40 : N < 40)
variable (avg_all : ℕ)
variable (avg_promoted : ℕ)
variable (avg_repeaters : ℕ)
variable (new_avg_promoted : ℕ)
variable (new_avg_repeaters : ℕ)

variables
  (promoted_condition : (71 * P + 56 * R) / N = 66)
  (increase_condition : (76 * P) / (P + R) = 75 ∧ (61 * R) / (P + R) = 59)
  (equation1 : 71 * P = 2 * R)
  (equation2: P + R = N)

-- Proof for part (a)
theorem no_valid_N_for_case1 
  (new_avg_promoted' : ℕ := 75) 
  (new_avg_repeaters' : ℕ := 59)
  : ∀ N, ¬ N < 40 ∨ ¬ ((76 * P) / (P + R) = new_avg_promoted' ∧ (61 * R) / (P + R) = new_avg_repeaters') := 
  sorry

-- Proof for part (b)
theorem valid_N_values_for_case2
  (possible_N_values : Finset ℕ := {6, 12, 18, 24, 30, 36})
  (new_avg_promoted'' : ℕ := 79)
  (new_avg_repeaters'' : ℕ := 47)
  : ∀ N, N ∈ possible_N_values ↔ (((76 * P) / (P + R) = new_avg_promoted'') ∧ (61 * R) / (P + R) = new_avg_repeaters'') := 
  sorry

end NUMINAMATH_GPT_no_valid_N_for_case1_valid_N_values_for_case2_l856_85613


namespace NUMINAMATH_GPT_average_weight_l856_85626

theorem average_weight (A B C : ℝ) (h1 : (A + B) / 2 = 40) (h2 : (B + C) / 2 = 47) (h3 : B = 39) : (A + B + C) / 3 = 45 := 
  sorry

end NUMINAMATH_GPT_average_weight_l856_85626


namespace NUMINAMATH_GPT_fedya_incorrect_l856_85612

theorem fedya_incorrect 
  (a b c d : ℕ) 
  (a_ends_in_9 : a % 10 = 9)
  (b_ends_in_7 : b % 10 = 7)
  (c_ends_in_3 : c % 10 = 3)
  (d_is_1 : d = 1) : 
  a ≠ b * c + d :=
by {
  sorry
}

end NUMINAMATH_GPT_fedya_incorrect_l856_85612


namespace NUMINAMATH_GPT_last_digit_of_power_sum_l856_85604

theorem last_digit_of_power_sum (m : ℕ) (hm : 0 < m) : (2^(m + 2006) + 2^m) % 10 = 0 := 
sorry

end NUMINAMATH_GPT_last_digit_of_power_sum_l856_85604


namespace NUMINAMATH_GPT_product_of_roots_l856_85697

theorem product_of_roots :
  ∀ α β : ℝ, (Polynomial.roots (Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 5 * Polynomial.X + Polynomial.C (-12))).prod = -6 :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_l856_85697


namespace NUMINAMATH_GPT_sum_of_consecutive_perfect_squares_l856_85627

theorem sum_of_consecutive_perfect_squares (k : ℕ) (h_pos : 0 < k)
  (h_eq : 2 * k^2 + 2 * k + 1 = 181) : k = 9 ∧ (k + 1) = 10 := by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_perfect_squares_l856_85627


namespace NUMINAMATH_GPT_total_campers_went_rowing_l856_85669

theorem total_campers_went_rowing (morning_campers afternoon_campers : ℕ) (h_morning : morning_campers = 35) (h_afternoon : afternoon_campers = 27) : morning_campers + afternoon_campers = 62 := by
  -- handle the proof
  sorry

end NUMINAMATH_GPT_total_campers_went_rowing_l856_85669


namespace NUMINAMATH_GPT_part1_part2_part3_l856_85670

-- Definitions for the given functions
def y1 (x : ℝ) : ℝ := -x + 1
def y2 (x : ℝ) : ℝ := -3 * x + 2

-- Part (1)
theorem part1 (a : ℝ) : (∃ x : ℝ, y1 x = a + y2 x ∧ x > 0) ↔ (a > -1) := sorry

-- Part (2)
theorem part2 (x y : ℝ) (h1 : y = y1 x) (h2 : y = y2 x) : 12*x^2 + 12*x*y + 3*y^2 = 27/4 := sorry

-- Part (3)
theorem part3 (A B : ℝ) (x : ℝ) (h : (4 - 2 * x) / ((3 * x - 2) * (x - 1)) = A / y1 x + B / y2 x) : (A / B + B / A) = -17 / 4 := sorry

end NUMINAMATH_GPT_part1_part2_part3_l856_85670


namespace NUMINAMATH_GPT_zoo_recovery_time_l856_85624

theorem zoo_recovery_time (lions rhinos recover_time : ℕ) (total_animals : ℕ) (total_time : ℕ)
    (h_lions : lions = 3) (h_rhinos : rhinos = 2) (h_recover_time : recover_time = 2)
    (h_total_animals : total_animals = lions + rhinos) (h_total_time : total_time = total_animals * recover_time) :
    total_time = 10 :=
by
  rw [h_lions, h_rhinos] at h_total_animals
  rw [h_total_animals, h_recover_time] at h_total_time
  exact h_total_time

end NUMINAMATH_GPT_zoo_recovery_time_l856_85624


namespace NUMINAMATH_GPT_six_digit_number_under_5_lakh_with_digit_sum_43_l856_85641

def is_6_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000
def under_500000 (n : ℕ) : Prop := n < 500000
def digit_sum (n : ℕ) : ℕ := (n / 100000) + (n / 10000 % 10) + (n / 1000 % 10) + (n / 100 % 10) + (n / 10 % 10) + (n % 10)

theorem six_digit_number_under_5_lakh_with_digit_sum_43 :
  is_6_digit 499993 ∧ under_500000 499993 ∧ digit_sum 499993 = 43 :=
by 
  sorry

end NUMINAMATH_GPT_six_digit_number_under_5_lakh_with_digit_sum_43_l856_85641


namespace NUMINAMATH_GPT_percentage_increase_14point4_from_12_l856_85692

theorem percentage_increase_14point4_from_12 (x : ℝ) (h : x = 14.4) : 
  ((x - 12) / 12) * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_percentage_increase_14point4_from_12_l856_85692


namespace NUMINAMATH_GPT_no_function_f_exists_l856_85691

theorem no_function_f_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2013 :=
by sorry

end NUMINAMATH_GPT_no_function_f_exists_l856_85691


namespace NUMINAMATH_GPT_decimal_equivalent_one_quarter_power_one_l856_85617

theorem decimal_equivalent_one_quarter_power_one : (1 / 4 : ℝ) ^ 1 = 0.25 := by
  sorry

end NUMINAMATH_GPT_decimal_equivalent_one_quarter_power_one_l856_85617


namespace NUMINAMATH_GPT_part_one_part_two_l856_85640

def f (x a : ℝ) : ℝ := |x - a| + |x + 2|

-- Question (1)
theorem part_one (x : ℝ) (a : ℝ) (h : a = 1) : f x a ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := 
sorry

-- Question (2)
theorem part_two (a : ℝ) (h : a ≥ 1) : ∀ (y : ℝ), (∃ x : ℝ, f x a = y) ↔ (∃ b : ℝ, y = b + 2 ∧ b ≥ a) := 
sorry

end NUMINAMATH_GPT_part_one_part_two_l856_85640


namespace NUMINAMATH_GPT_problem_statement_l856_85680

variables (x y : ℚ)

theorem problem_statement 
  (h1 : x + y = 8 / 15) 
  (h2 : x - y = 1 / 105) : 
  x^2 - y^2 = 8 / 1575 :=
sorry

end NUMINAMATH_GPT_problem_statement_l856_85680


namespace NUMINAMATH_GPT_four_person_apartments_l856_85602

theorem four_person_apartments : 
  ∃ x : ℕ, 
    (4 * (10 + 20 * 2 + 4 * x)) * 3 / 4 = 210 → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_four_person_apartments_l856_85602


namespace NUMINAMATH_GPT_combined_teaching_years_l856_85607

def Adrienne_Yrs : ℕ := 22
def Virginia_Yrs : ℕ := Adrienne_Yrs + 9
def Dennis_Yrs : ℕ := 40

theorem combined_teaching_years :
  Adrienne_Yrs + Virginia_Yrs + Dennis_Yrs = 93 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_combined_teaching_years_l856_85607


namespace NUMINAMATH_GPT_max_gross_profit_price_l856_85621

def purchase_price : ℝ := 20
def Q (P : ℝ) : ℝ := 8300 - 170 * P - P^2
def L (P : ℝ) : ℝ := (8300 - 170 * P - P^2) * (P - 20)

theorem max_gross_profit_price : ∃ P : ℝ, (∀ x : ℝ, L x ≤ L P) ∧ P = 30 :=
by
  sorry

end NUMINAMATH_GPT_max_gross_profit_price_l856_85621


namespace NUMINAMATH_GPT_greatest_whole_number_inequality_l856_85635

theorem greatest_whole_number_inequality :
  ∃ x : ℕ, (5 * x - 4 < 3 - 2 * x) ∧ ∀ y : ℕ, (5 * y - 4 < 3 - 2 * y → y ≤ x) :=
sorry

end NUMINAMATH_GPT_greatest_whole_number_inequality_l856_85635


namespace NUMINAMATH_GPT_cost_price_percentage_l856_85685

theorem cost_price_percentage (SP CP : ℝ) (hp : SP - CP = (1/3) * CP) : CP = 0.75 * SP :=
by
  sorry

end NUMINAMATH_GPT_cost_price_percentage_l856_85685


namespace NUMINAMATH_GPT_opposite_of_three_l856_85639

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end NUMINAMATH_GPT_opposite_of_three_l856_85639


namespace NUMINAMATH_GPT_bus_passengers_remaining_l856_85689

theorem bus_passengers_remaining (initial_passengers : ℕ := 22) 
                                 (boarding_alighting1 : (ℤ × ℤ) := (4, -8)) 
                                 (boarding_alighting2 : (ℤ × ℤ) := (6, -5)) : 
                                 (initial_passengers : ℤ) + 
                                 (boarding_alighting1.fst + boarding_alighting1.snd) + 
                                 (boarding_alighting2.fst + boarding_alighting2.snd) = 19 :=
by
  sorry

end NUMINAMATH_GPT_bus_passengers_remaining_l856_85689


namespace NUMINAMATH_GPT_meal_service_count_l856_85695

/-- Define the number of people -/
def people_count : ℕ := 10

/-- Define the number of people that order pasta -/
def pasta_count : ℕ := 5

/-- Define the number of people that order salad -/
def salad_count : ℕ := 5

/-- Combination function to choose 2 people from 10 -/
def choose_2_from_10 : ℕ := Nat.choose 10 2

/-- Number of derangements of 8 people where exactly 2 people receive their correct meals -/
def derangement_8 : ℕ := 21

/-- Number of ways to correctly serve the meals where exactly 2 people receive the correct meal -/
theorem meal_service_count :
  choose_2_from_10 * derangement_8 = 945 :=
  by sorry

end NUMINAMATH_GPT_meal_service_count_l856_85695


namespace NUMINAMATH_GPT_solution_set_f_less_x_plus_1_l856_85687

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_continuous : Continuous f
axiom f_at_1 : f 1 = 2
axiom f_derivative : ∀ x, deriv f x < 1

theorem solution_set_f_less_x_plus_1 : 
  ∀ x : ℝ, (f x < x + 1) ↔ (x > 1) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_f_less_x_plus_1_l856_85687


namespace NUMINAMATH_GPT_solve_abs_inequality_l856_85668

theorem solve_abs_inequality (x : ℝ) :
  (|x-2| ≥ |x|) → x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_abs_inequality_l856_85668


namespace NUMINAMATH_GPT_minimum_value_inequality_l856_85614

theorem minimum_value_inequality (x y : ℝ) (hx : x > 2) (hy : y > 2) :
    (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) ≥ 18 := by
  sorry

end NUMINAMATH_GPT_minimum_value_inequality_l856_85614


namespace NUMINAMATH_GPT_time_to_fill_pool_l856_85636

theorem time_to_fill_pool (V : ℕ) (n : ℕ) (r : ℕ) (fill_rate_per_hour : ℕ) :
  V = 24000 → 
  n = 4 →
  r = 25 → -- 2.5 gallons per minute expressed as 25/10 gallons
  fill_rate_per_hour = (n * r * 6) → -- since 6 * 10 = 60 (to convert per minute rate to per hour, we divide so r is 25 instead of 2.5)
  V / fill_rate_per_hour = 40 :=
by
  sorry

end NUMINAMATH_GPT_time_to_fill_pool_l856_85636


namespace NUMINAMATH_GPT_solution_unique_s_l856_85647

theorem solution_unique_s (s : ℝ) (hs : ⌊s⌋ + s = 22.7) : s = 11.7 :=
sorry

end NUMINAMATH_GPT_solution_unique_s_l856_85647


namespace NUMINAMATH_GPT_odd_function_f_neg_x_l856_85684

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 - 2 * x else -(x^2 + 2 * x)

theorem odd_function_f_neg_x (x : ℝ) (hx : x < 0) :
  f x = -x^2 - 2 * x :=
by
  sorry

end NUMINAMATH_GPT_odd_function_f_neg_x_l856_85684


namespace NUMINAMATH_GPT_find_cos_minus_sin_l856_85690

variable (θ : ℝ)
variable (h1 : θ ∈ Set.Ioo (3 * Real.pi / 4) Real.pi)
variable (h2 : Real.sin θ * Real.cos θ = -Real.sqrt 3 / 2)

theorem find_cos_minus_sin : Real.cos θ - Real.sin θ = -Real.sqrt (1 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_GPT_find_cos_minus_sin_l856_85690


namespace NUMINAMATH_GPT_square_area_l856_85609

theorem square_area (l w x : ℝ) (h1 : 2 * (l + w) = 20) (h2 : l = x / 2) (h3 : w = x / 4) :
  x^2 = 1600 / 9 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l856_85609


namespace NUMINAMATH_GPT_find_x_on_line_segment_l856_85649

theorem find_x_on_line_segment (x : ℚ) : 
    (∃ m : ℚ, m = (9 - (-1))/(1 - (-2)) ∧ (2 - 9 = m * (x - 1))) → x = -11/10 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_on_line_segment_l856_85649


namespace NUMINAMATH_GPT_hyperbola_eqn_l856_85656

-- Definitions of given conditions
def a := 4
def b := 3
def c := 5

def hyperbola (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Hypotheses derived from conditions
axiom asymptotes : b / a = 3 / 4
axiom right_focus : a^2 + b^2 = c^2

-- Main theorem statement
theorem hyperbola_eqn : (forall x y, hyperbola x y ↔ x^2 / 16 - y^2 / 9 = 1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_hyperbola_eqn_l856_85656


namespace NUMINAMATH_GPT_car_mileage_city_l856_85620

theorem car_mileage_city {h c t : ℝ} (H1: 448 = h * t) (H2: 336 = c * t) (H3: c = h - 6) : c = 18 :=
sorry

end NUMINAMATH_GPT_car_mileage_city_l856_85620


namespace NUMINAMATH_GPT_scientific_notation_of_3900000000_l856_85676

theorem scientific_notation_of_3900000000 : 3900000000 = 3.9 * 10^9 :=
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_of_3900000000_l856_85676


namespace NUMINAMATH_GPT_problem1_problem2_l856_85679

-- Definitions
def total_questions := 5
def multiple_choice := 3
def true_false := 2
def total_outcomes := total_questions * (total_questions - 1)

-- (1) Probability of A drawing a true/false question and B drawing a multiple-choice question
def favorable_outcomes_1 := true_false * multiple_choice

-- (2) Probability of at least one of A or B drawing a multiple-choice question
def unfavorable_outcomes_2 := true_false * (true_false - 1)

-- Statements to be proved
theorem problem1 : favorable_outcomes_1 / total_outcomes = 3 / 10 := by sorry

theorem problem2 : 1 - (unfavorable_outcomes_2 / total_outcomes) = 9 / 10 := by sorry

end NUMINAMATH_GPT_problem1_problem2_l856_85679


namespace NUMINAMATH_GPT_remainder_of_3_pow_21_mod_11_l856_85634

theorem remainder_of_3_pow_21_mod_11 : (3^21 % 11) = 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_of_3_pow_21_mod_11_l856_85634


namespace NUMINAMATH_GPT_trains_time_distance_l856_85694

-- Define the speeds of the two trains
def speed1 : ℕ := 11
def speed2 : ℕ := 31

-- Define the distance between the two trains after time t
def distance_between_trains (t : ℕ) : ℕ :=
  speed2 * t - speed1 * t

-- Define the condition that this distance is 160 miles
def condition (t : ℕ) : Prop :=
  distance_between_trains t = 160

-- State the theorem to prove
theorem trains_time_distance : ∃ t : ℕ, condition t ∧ t = 8 :=
by
  use 8
  unfold condition
  unfold distance_between_trains
  -- Verifying the calculated distance
  sorry

end NUMINAMATH_GPT_trains_time_distance_l856_85694


namespace NUMINAMATH_GPT_value_of_b_conditioned_l856_85600

theorem value_of_b_conditioned
  (b: ℝ) 
  (h0 : 0 < b ∧ b < 7)
  (h1 : (1 / 2) * (8 - b) * (b - 8) / ((1 / 2) * (b / 2) * b) = 4 / 9):
  b = 4 := 
sorry

end NUMINAMATH_GPT_value_of_b_conditioned_l856_85600


namespace NUMINAMATH_GPT_exists_231_four_digit_integers_l856_85698

theorem exists_231_four_digit_integers (n : ℕ) : 
  (∃ A B C D : ℕ, 
     A ≠ 0 ∧ 
     1 ≤ A ∧ A ≤ 9 ∧ 
     0 ≤ B ∧ B ≤ 9 ∧ 
     0 ≤ C ∧ C ≤ 9 ∧ 
     0 ≤ D ∧ D ≤ 9 ∧ 
     999 * (A - D) + 90 * (B - C) = n^3) ↔ n = 231 :=
by sorry

end NUMINAMATH_GPT_exists_231_four_digit_integers_l856_85698


namespace NUMINAMATH_GPT_angle_bao_proof_l856_85651

noncomputable def angle_bao : ℝ := sorry -- angle BAO in degrees

theorem angle_bao_proof 
    (CD_is_diameter : true)
    (A_on_extension_DC_beyond_C : true)
    (E_on_semicircle : true)
    (B_is_intersection_AE_semicircle : B ≠ E)
    (AB_eq_OE : AB = OE)
    (angle_EOD_30_degrees : EOD = 30) : 
    angle_bao = 7.5 :=
sorry

end NUMINAMATH_GPT_angle_bao_proof_l856_85651


namespace NUMINAMATH_GPT_plants_needed_correct_l856_85666

def total_plants_needed (ferns palms succulents total_desired : ℕ) : ℕ :=
 total_desired - (ferns + palms + succulents)

theorem plants_needed_correct : total_plants_needed 3 5 7 24 = 9 := by
  sorry

end NUMINAMATH_GPT_plants_needed_correct_l856_85666


namespace NUMINAMATH_GPT_probability_of_red_ball_l856_85628

-- Define the conditions
def num_balls : ℕ := 3
def red_balls : ℕ := 2
def white_balls : ℕ := 1

-- Calculate the probability
def probability_drawing_red_ball : ℚ := red_balls / num_balls

-- The theorem statement to be proven
theorem probability_of_red_ball : probability_drawing_red_ball = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_red_ball_l856_85628


namespace NUMINAMATH_GPT_calories_remaining_for_dinner_l856_85665

theorem calories_remaining_for_dinner :
  let daily_limit := 2200
  let breakfast := 353
  let lunch := 885
  let snack := 130
  let total_consumed := breakfast + lunch + snack
  let remaining_for_dinner := daily_limit - total_consumed
  remaining_for_dinner = 832 := by
  sorry

end NUMINAMATH_GPT_calories_remaining_for_dinner_l856_85665


namespace NUMINAMATH_GPT_number_of_revolutions_wheel_half_mile_l856_85663

theorem number_of_revolutions_wheel_half_mile :
  let diameter := 10 * (1 : ℝ)
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let half_mile_in_feet := 2640
  (half_mile_in_feet / circumference) = 264 / Real.pi :=
by
  let diameter := 10 * (1 : ℝ)
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let half_mile_in_feet := 2640
  have h : (half_mile_in_feet / circumference) = 264 / Real.pi := by
    sorry
  exact h

end NUMINAMATH_GPT_number_of_revolutions_wheel_half_mile_l856_85663


namespace NUMINAMATH_GPT_total_pieces_on_chessboard_l856_85601

-- Given conditions about initial chess pieces and lost pieces.
def initial_pieces_each : Nat := 16
def pieces_lost_arianna : Nat := 3
def pieces_lost_samantha : Nat := 9

-- The remaining pieces for each player.
def remaining_pieces_arianna : Nat := initial_pieces_each - pieces_lost_arianna
def remaining_pieces_samantha : Nat := initial_pieces_each - pieces_lost_samantha

-- The total remaining pieces on the chessboard.
def total_remaining_pieces : Nat := remaining_pieces_arianna + remaining_pieces_samantha

-- The theorem to prove
theorem total_pieces_on_chessboard : total_remaining_pieces = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_pieces_on_chessboard_l856_85601


namespace NUMINAMATH_GPT_total_flour_l856_85605

def bought_rye_flour := 5
def bought_bread_flour := 10
def bought_chickpea_flour := 3
def had_pantry_flour := 2

theorem total_flour : bought_rye_flour + bought_bread_flour + bought_chickpea_flour + had_pantry_flour = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_flour_l856_85605


namespace NUMINAMATH_GPT_number_of_pairs_l856_85603

noncomputable def number_of_ordered_pairs (n : ℕ) : ℕ :=
  if n = 5 then 8 else 0

theorem number_of_pairs (f m: ℕ) : f ≥ 0 ∧ m ≥ 0 → number_of_ordered_pairs 5 = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_pairs_l856_85603


namespace NUMINAMATH_GPT_sum_a5_a6_a7_l856_85629

variable (a : ℕ → ℝ) (q : ℝ)

-- Assumptions
axiom geometric_sequence : ∀ n, a (n + 1) = a n * q

axiom sum_a1_a2_a3 : a 1 + a 2 + a 3 = 1
axiom sum_a2_a3_a4 : a 2 + a 3 + a 4 = 2

-- The theorem we want to prove
theorem sum_a5_a6_a7 : a 5 + a 6 + a 7 = 16 := sorry

end NUMINAMATH_GPT_sum_a5_a6_a7_l856_85629


namespace NUMINAMATH_GPT_max_area_triangle_PAB_l856_85637

open Real

noncomputable def ellipse_eq (x y : ℝ) : Prop := 
  (x^2 / 16) + (y^2 / 9) = 1

def point_A : (ℝ × ℝ) := (4, 0)
def point_B : (ℝ × ℝ) := (0, 3)

theorem max_area_triangle_PAB (P : ℝ × ℝ) (hP : ellipse_eq P.1 P.2) : 
  ∃ S, S = 6 * (sqrt 2 + 1) := 
sorry

end NUMINAMATH_GPT_max_area_triangle_PAB_l856_85637


namespace NUMINAMATH_GPT_rose_share_correct_l856_85674

-- Define the conditions
def purity_share (P : ℝ) : ℝ := P
def sheila_share (P : ℝ) : ℝ := 5 * P
def rose_share (P : ℝ) : ℝ := 3 * P
def total_rent := 5400

-- The theorem to be proven
theorem rose_share_correct (P : ℝ) (h : purity_share P + sheila_share P + rose_share P = total_rent) : 
  rose_share P = 1800 :=
  sorry

end NUMINAMATH_GPT_rose_share_correct_l856_85674


namespace NUMINAMATH_GPT_pentagon_area_greater_than_square_third_l856_85645

theorem pentagon_area_greater_than_square_third (a b : ℝ) :
  a^2 + (a * b) / 4 + (Real.sqrt 3 / 4) * b^2 > ((a + b)^2) / 3 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_area_greater_than_square_third_l856_85645


namespace NUMINAMATH_GPT_pounds_per_pie_l856_85657

-- Define the conditions
def total_weight : ℕ := 120
def applesauce_weight := total_weight / 2
def pies_weight := total_weight - applesauce_weight
def number_of_pies := 15

-- Define the required proof for pounds per pie
theorem pounds_per_pie :
  pies_weight / number_of_pies = 4 := by
  sorry

end NUMINAMATH_GPT_pounds_per_pie_l856_85657


namespace NUMINAMATH_GPT_remainder_of_prime_powers_l856_85683

theorem remainder_of_prime_powers (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  (p^(q-1) + q^(p-1)) % (p * q) = 1 := 
sorry

end NUMINAMATH_GPT_remainder_of_prime_powers_l856_85683


namespace NUMINAMATH_GPT_incorrect_weight_conclusion_l856_85650

theorem incorrect_weight_conclusion (x y : ℝ) (h1 : y = 0.85 * x - 85.71) :
  ¬ (x = 160 → y = 50.29) :=
sorry

end NUMINAMATH_GPT_incorrect_weight_conclusion_l856_85650


namespace NUMINAMATH_GPT_probability_a_2b_3c_gt_5_l856_85671

def isInUnitCube (a b c : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1

theorem probability_a_2b_3c_gt_5 (a b c : ℝ) :
  isInUnitCube a b c → ¬(a + 2 * b + 3 * c > 5) :=
by
  intro h
  -- The proof goes here, currently using sorry as placeholder
  sorry

end NUMINAMATH_GPT_probability_a_2b_3c_gt_5_l856_85671


namespace NUMINAMATH_GPT_car_speed_first_hour_l856_85677

theorem car_speed_first_hour (x : ℝ) (h : (79 = (x + 60) / 2)) : x = 98 :=
by {
  sorry
}

end NUMINAMATH_GPT_car_speed_first_hour_l856_85677


namespace NUMINAMATH_GPT_inequality_proof_l856_85675

variable (a b : ℝ)

theorem inequality_proof (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) :
  (1 / a) + (1 / b) + (1 / (a * b)) ≥ 8 :=
  by
    sorry

end NUMINAMATH_GPT_inequality_proof_l856_85675


namespace NUMINAMATH_GPT_puppies_given_l856_85688

-- Definitions of the initial and left numbers of puppies
def initial_puppies : ℕ := 7
def left_puppies : ℕ := 2

-- Theorem stating that the number of puppies given to friends is the difference
theorem puppies_given : initial_puppies - left_puppies = 5 := by
  sorry -- Proof not required, so we use sorry

end NUMINAMATH_GPT_puppies_given_l856_85688


namespace NUMINAMATH_GPT_find_n_l856_85654

theorem find_n (n : ℕ) : (256 : ℝ) ^ (1 / 4 : ℝ) = 4 ^ n → 256 = (4 ^ 4 : ℝ) → n = 1 :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_find_n_l856_85654


namespace NUMINAMATH_GPT_perimeter_of_triangle_XYZ_l856_85661

/-- 
  Given the inscribed circle of triangle XYZ is tangent to XY at P,
  its radius is 15, XP = 30, and PY = 36, then the perimeter of 
  triangle XYZ is 83.4.
-/
theorem perimeter_of_triangle_XYZ :
  ∀ (XYZ : Type) (P : XYZ) (radius : ℝ) (XP PY perimeter : ℝ),
    radius = 15 → 
    XP = 30 → 
    PY = 36 →
    perimeter = 83.4 :=
by 
  intros XYZ P radius XP PY perimeter h_radius h_XP h_PY
  sorry

end NUMINAMATH_GPT_perimeter_of_triangle_XYZ_l856_85661


namespace NUMINAMATH_GPT_problem_statement_l856_85682

def operation (x y z : ℕ) : ℕ := (2 * x * y + y^2) * z^3

theorem problem_statement : operation 7 (operation 4 5 3) 2 = 24844760 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l856_85682


namespace NUMINAMATH_GPT_pens_sales_consistency_books_left_indeterminate_l856_85653

-- The initial conditions
def initial_pens : ℕ := 42
def initial_books : ℕ := 143
def pens_left : ℕ := 19
def pens_sold : ℕ := 23

-- Prove the consistency of the number of pens sold
theorem pens_sales_consistency : initial_pens - pens_left = pens_sold := by
  sorry

-- Assert that the number of books left is indeterminate based on provided conditions
theorem books_left_indeterminate : ∃ b_left : ℕ, b_left ≤ initial_books ∧
    ∀ n_books_sold : ℕ, n_books_sold > 0 → b_left = initial_books - n_books_sold := by
  sorry

end NUMINAMATH_GPT_pens_sales_consistency_books_left_indeterminate_l856_85653


namespace NUMINAMATH_GPT_total_time_is_correct_l856_85681

-- Definitions based on conditions
def timeouts_for_running : ℕ := 5
def timeouts_for_throwing_food : ℕ := 5 * timeouts_for_running - 1
def timeouts_for_swearing : ℕ := timeouts_for_throwing_food / 3

-- Definition for total time-outs
def total_timeouts : ℕ := timeouts_for_running + timeouts_for_throwing_food + timeouts_for_swearing
-- Each time-out is 5 minutes
def timeout_duration : ℕ := 5

-- Total time in minutes
def total_time_in_minutes : ℕ := total_timeouts * timeout_duration

-- The proof statement
theorem total_time_is_correct : total_time_in_minutes = 185 := by
  sorry

end NUMINAMATH_GPT_total_time_is_correct_l856_85681


namespace NUMINAMATH_GPT_polynomial_has_roots_l856_85658

theorem polynomial_has_roots :
  ∃ x : ℝ, x ∈ [-4, -3, -1, 2] ∧ (x^4 + 6 * x^3 + 7 * x^2 - 14 * x - 12 = 0) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_has_roots_l856_85658


namespace NUMINAMATH_GPT_angle_RPS_is_1_degree_l856_85659

-- Definitions of the given angles
def angle_QRS : ℝ := 150
def angle_PQS : ℝ := 60
def angle_PSQ : ℝ := 49
def angle_QPR : ℝ := 70

-- Definition for the calculated angle QPS
def angle_QPS : ℝ := 180 - angle_PQS - angle_PSQ

-- Definition for the target angle RPS
def angle_RPS : ℝ := angle_QPS - angle_QPR

-- The theorem we aim to prove
theorem angle_RPS_is_1_degree : angle_RPS = 1 := by
  sorry

end NUMINAMATH_GPT_angle_RPS_is_1_degree_l856_85659


namespace NUMINAMATH_GPT_alice_commute_distance_l856_85655

noncomputable def office_distance_commute (commute_time_regular commute_time_holiday : ℝ) (speed_increase : ℝ) : ℝ := 
  let v := commute_time_regular * ((commute_time_regular + speed_increase) / commute_time_holiday - speed_increase)
  commute_time_regular * v

theorem alice_commute_distance : 
  office_distance_commute 0.5 0.3 12 = 9 := 
sorry

end NUMINAMATH_GPT_alice_commute_distance_l856_85655


namespace NUMINAMATH_GPT_remainder_when_sum_of_first_six_primes_divided_by_seventh_l856_85606

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_remainder_when_sum_of_first_six_primes_divided_by_seventh_l856_85606


namespace NUMINAMATH_GPT_largest_common_term_l856_85625

theorem largest_common_term (b : ℕ) : 
  (b < 1000) ∧ (b % 5 = 4) ∧ (b % 11 = 7) → b = 964 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_largest_common_term_l856_85625


namespace NUMINAMATH_GPT_lowest_point_graph_of_y_l856_85623

theorem lowest_point_graph_of_y (x : ℝ) (h : x > -1) :
  (x, (x^2 + 2 * x + 2) / (x + 1)) = (0, 2) ∧ ∀ y > -1, ( (y^2 + 2 * y + 2) / (y + 1) >= 2) := 
sorry

end NUMINAMATH_GPT_lowest_point_graph_of_y_l856_85623


namespace NUMINAMATH_GPT_find_solutions_l856_85673

theorem find_solutions (a m n : ℕ) (h : a > 0) (h₁ : m > 0) (h₂ : n > 0) :
  (a^m + 1) ∣ (a + 1)^n → 
  ((a = 1 ∧ True) ∨ (True ∧ m = 1) ∨ (a = 2 ∧ m = 3 ∧ n ≥ 2)) :=
by sorry

end NUMINAMATH_GPT_find_solutions_l856_85673


namespace NUMINAMATH_GPT_compute_exponent_multiplication_l856_85615

theorem compute_exponent_multiplication : 8 * (2 / 7)^4 = 128 / 2401 := by
  sorry

end NUMINAMATH_GPT_compute_exponent_multiplication_l856_85615


namespace NUMINAMATH_GPT_max_sum_distinct_factors_2029_l856_85608

theorem max_sum_distinct_factors_2029 :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2029 ∧ A + B + C = 297 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_distinct_factors_2029_l856_85608


namespace NUMINAMATH_GPT_perfect_squares_solutions_l856_85696

noncomputable def isPerfectSquare (n : ℤ) : Prop :=
  ∃ m : ℤ, m * m = n

theorem perfect_squares_solutions :
  ∀ (a b : ℕ),
    0 < a → 0 < b →
    (isPerfectSquare (↑a * ↑a - 4 * ↑b)) →
    (isPerfectSquare (↑b * ↑b - 4 * ↑a)) →
      (a = 4 ∧ b = 4) ∨
      (a = 5 ∧ b = 6) ∨
      (a = 6 ∧ b = 5) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_perfect_squares_solutions_l856_85696


namespace NUMINAMATH_GPT_costume_processing_time_l856_85633

theorem costume_processing_time (x : ℕ) : 
  (300 - 60) / (2 * x) + 60 / x = 9 → (60 / x) + (240 / (2 * x)) = 9 :=
by
  sorry

end NUMINAMATH_GPT_costume_processing_time_l856_85633


namespace NUMINAMATH_GPT_sum_squares_inequality_l856_85686

theorem sum_squares_inequality (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
(h_sum : x + y + z = 3) : 
  1 / (x^2 + y + z) + 1 / (x + y^2 + z) + 1 / (x + y + z^2) ≤ 1 := 
by 
  sorry

end NUMINAMATH_GPT_sum_squares_inequality_l856_85686


namespace NUMINAMATH_GPT_nature_of_roots_l856_85638

noncomputable def P (x : ℝ) : ℝ := x^6 - 5 * x^5 + 3 * x^2 - 8 * x + 16

theorem nature_of_roots : (∀ x : ℝ, x < 0 → P x > 0) ∧ ∃ x : ℝ, 1 < x ∧ x < 2 ∧ P x = 0 := 
by
  sorry

end NUMINAMATH_GPT_nature_of_roots_l856_85638


namespace NUMINAMATH_GPT_max_distance_circle_to_line_l856_85630

open Real

theorem max_distance_circle_to_line :
  let circle_eq (x y : ℝ) := x^2 + y^2 - 2*x - 2*y + 1 = 0
  let line_eq (x y : ℝ) := x - y = 2
  ∃ (M : ℝ), (∀ x y, circle_eq x y → ∀ (d : ℝ), (line_eq x y → M ≤ d)) ∧ M = sqrt 2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_max_distance_circle_to_line_l856_85630
