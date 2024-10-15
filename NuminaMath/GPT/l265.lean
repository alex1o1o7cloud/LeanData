import Mathlib

namespace NUMINAMATH_GPT_math_problem_l265_26514

variables (x y : ℝ)

noncomputable def question_value (x y : ℝ) : ℝ := (2 * x - 5 * y) / (5 * x + 2 * y)

theorem math_problem 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (cond : (5 * x - 2 * y) / (2 * x + 3 * y) = 1) : 
  question_value x y = -5 / 31 :=
sorry

end NUMINAMATH_GPT_math_problem_l265_26514


namespace NUMINAMATH_GPT_blake_spending_on_oranges_l265_26510

theorem blake_spending_on_oranges (spending_on_oranges spending_on_apples spending_on_mangoes initial_amount change_amount: ℝ)
  (h1 : spending_on_apples = 50)
  (h2 : spending_on_mangoes = 60)
  (h3 : initial_amount = 300)
  (h4 : change_amount = 150)
  (h5 : initial_amount - change_amount = spending_on_oranges + spending_on_apples + spending_on_mangoes) :
  spending_on_oranges = 40 := by
  sorry

end NUMINAMATH_GPT_blake_spending_on_oranges_l265_26510


namespace NUMINAMATH_GPT_problem_l265_26535

theorem problem (f : ℕ → ℕ → ℕ) (h0 : f 1 1 = 1) (h1 : ∀ m n, f m n ∈ {x | x > 0}) 
  (h2 : ∀ m n, f m (n + 1) = f m n + 2) (h3 : ∀ m, f (m + 1) 1 = 2 * f m 1) : 
  f 1 5 = 9 ∧ f 5 1 = 16 ∧ f 5 6 = 26 :=
sorry

end NUMINAMATH_GPT_problem_l265_26535


namespace NUMINAMATH_GPT_point_on_line_is_sufficient_but_not_necessary_condition_for_arithmetic_sequence_l265_26532

def is_on_line (n : ℕ) (a_n : ℕ) : Prop := a_n = 2 * n + 1

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n m, a n - a m = d * (n - m)

theorem point_on_line_is_sufficient_but_not_necessary_condition_for_arithmetic_sequence (a : ℕ → ℕ) :
  (∀ n, is_on_line n (a n)) → is_arithmetic_sequence a ∧ 
  ¬ (is_arithmetic_sequence a → ∀ n, is_on_line n (a n)) :=
sorry

end NUMINAMATH_GPT_point_on_line_is_sufficient_but_not_necessary_condition_for_arithmetic_sequence_l265_26532


namespace NUMINAMATH_GPT_sin_minus_cos_eq_one_l265_26589

theorem sin_minus_cos_eq_one (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = 1) : x = Real.pi / 2 :=
by sorry

end NUMINAMATH_GPT_sin_minus_cos_eq_one_l265_26589


namespace NUMINAMATH_GPT_ratio_s_t_l265_26594

variable {b s t : ℝ}
variable (hb : b ≠ 0)
variable (h1 : s = -b / 8)
variable (h2 : t = -b / 4)

theorem ratio_s_t : s / t = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_s_t_l265_26594


namespace NUMINAMATH_GPT_composite_product_division_l265_26528

noncomputable def firstFiveCompositeProduct : ℕ := 4 * 6 * 8 * 9 * 10
noncomputable def nextFiveCompositeProduct : ℕ := 12 * 14 * 15 * 16 * 18

theorem composite_product_division : firstFiveCompositeProduct / nextFiveCompositeProduct = 1 / 42 := by
  sorry

end NUMINAMATH_GPT_composite_product_division_l265_26528


namespace NUMINAMATH_GPT_min_containers_needed_l265_26553

def container_capacity : ℕ := 500
def required_tea : ℕ := 5000

theorem min_containers_needed (n : ℕ) : n * container_capacity ≥ required_tea → n = 10 :=
sorry

end NUMINAMATH_GPT_min_containers_needed_l265_26553


namespace NUMINAMATH_GPT_semicircle_inequality_l265_26524

-- Define the points on the semicircle
variables (A B C D E : ℝ)
-- Define the length function
def length (X Y : ℝ) : ℝ := abs (X - Y)

-- This is the main theorem statement
theorem semicircle_inequality {A B C D E : ℝ} :
  length A B ^ 2 + length B C ^ 2 + length C D ^ 2 + length D E ^ 2 +
  length A B * length B C * length C D + length B C * length C D * length D E < 4 :=
sorry

end NUMINAMATH_GPT_semicircle_inequality_l265_26524


namespace NUMINAMATH_GPT_evaluate_sum_l265_26547

variable {a b c : ℝ}

theorem evaluate_sum 
  (h : a / (30 - a) + b / (75 - b) + c / (55 - c) = 8) :
  6 / (30 - a) + 15 / (75 - b) + 11 / (55 - c) = 187 / 30 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_sum_l265_26547


namespace NUMINAMATH_GPT_right_triangle_perimeter_l265_26564

theorem right_triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) (perimeter : ℝ)
  (h1 : area = 180) 
  (h2 : leg1 = 30) 
  (h3 : (1 / 2) * leg1 * leg2 = area)
  (h4 : hypotenuse^2 = leg1^2 + leg2^2)
  (h5 : leg2 = 12) 
  (h6 : hypotenuse = 2 * Real.sqrt 261) :
  perimeter = 42 + 2 * Real.sqrt 261 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_perimeter_l265_26564


namespace NUMINAMATH_GPT_shooting_game_system_l265_26521

theorem shooting_game_system :
  ∃ x y : ℕ, (x + y = 20 ∧ 3 * x = y) :=
by
  sorry

end NUMINAMATH_GPT_shooting_game_system_l265_26521


namespace NUMINAMATH_GPT_max_value_m_l265_26575

theorem max_value_m (a b : ℝ) (ha : a > 0) (hb : b > 0) (m : ℝ)
  (h : (2 / a) + (1 / b) ≥ m / (2 * a + b)) : m ≤ 9 :=
sorry

end NUMINAMATH_GPT_max_value_m_l265_26575


namespace NUMINAMATH_GPT_min_val_proof_l265_26525

noncomputable def minimum_value (x y z: ℝ) := 9 / x + 4 / y + 1 / z

theorem min_val_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + 2 * y + 3 * z = 12) :
  minimum_value x y z ≥ 49 / 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_val_proof_l265_26525


namespace NUMINAMATH_GPT_number_of_men_for_2km_road_l265_26531

noncomputable def men_for_1km_road : ℕ := 30
noncomputable def days_for_1km_road : ℕ := 12
noncomputable def hours_per_day_for_1km_road : ℕ := 8
noncomputable def length_of_1st_road : ℕ := 1
noncomputable def length_of_2nd_road : ℕ := 2
noncomputable def working_hours_per_day_2nd_road : ℕ := 14
noncomputable def days_for_2km_road : ℝ := 20.571428571428573

theorem number_of_men_for_2km_road (total_man_hours_1km : ℕ := men_for_1km_road * days_for_1km_road * hours_per_day_for_1km_road):
  (men_for_1km_road * length_of_2nd_road * days_for_1km_road * hours_per_day_for_1km_road = 5760) →
  ∃ (men_for_2nd_road : ℕ), men_for_1km_road * 2 * days_for_1km_road * hours_per_day_for_1km_road = 5760 ∧  men_for_2nd_road * days_for_2km_road * working_hours_per_day_2nd_road = 5760 ∧ men_for_2nd_road = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_men_for_2km_road_l265_26531


namespace NUMINAMATH_GPT_find_f_of_3_l265_26520

theorem find_f_of_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1/x + 2) = x) : f 3 = 1 := 
sorry

end NUMINAMATH_GPT_find_f_of_3_l265_26520


namespace NUMINAMATH_GPT_find_x1_plus_x2_l265_26519

def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

theorem find_x1_plus_x2 (x1 x2 : ℝ) (hneq : x1 ≠ x2) (h1 : f x1 = 101) (h2 : f x2 = 101) : x1 + x2 = 2 := 
by 
  -- proof or sorry can be used; let's assume we use sorry to skip proof
  sorry

end NUMINAMATH_GPT_find_x1_plus_x2_l265_26519


namespace NUMINAMATH_GPT_fishing_tomorrow_l265_26527

theorem fishing_tomorrow (seven_every_day eight_every_other_day three_every_three_days twelve_yesterday ten_today : ℕ)
  (h1 : seven_every_day = 7)
  (h2 : eight_every_other_day = 8)
  (h3 : three_every_three_days = 3)
  (h4 : twelve_yesterday = 12)
  (h5 : ten_today = 10) :
  (seven_every_day + (eight_every_other_day - (twelve_yesterday - seven_every_day)) + three_every_three_days) = 15 :=
by
  sorry

end NUMINAMATH_GPT_fishing_tomorrow_l265_26527


namespace NUMINAMATH_GPT_jeff_can_store_songs_l265_26561

def gbToMb (gb : ℕ) : ℕ := gb * 1000

def newAppsStorage : ℕ :=
  5 * 450 + 5 * 300 + 5 * 150

def newPhotosStorage : ℕ :=
  300 * 4 + 50 * 8

def newVideosStorage : ℕ :=
  15 * 400 + 30 * 200

def newPDFsStorage : ℕ :=
  25 * 20

def totalNewStorage : ℕ :=
  newAppsStorage + newPhotosStorage + newVideosStorage + newPDFsStorage

def existingStorage : ℕ :=
  gbToMb 7

def totalUsedStorage : ℕ :=
  existingStorage + totalNewStorage

def totalStorage : ℕ :=
  gbToMb 32

def remainingStorage : ℕ :=
  totalStorage - totalUsedStorage

def numSongs (storage : ℕ) (avgSongSize : ℕ) : ℕ :=
  storage / avgSongSize

theorem jeff_can_store_songs : 
  numSongs remainingStorage 20 = 320 :=
by
  sorry

end NUMINAMATH_GPT_jeff_can_store_songs_l265_26561


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l265_26584

theorem arithmetic_sequence_common_difference 
  (a l S : ℕ) (h1 : a = 5) (h2 : l = 50) (h3 : S = 495) :
  (∃ d n : ℕ, l = a + (n-1) * d ∧ S = n * (a + l) / 2 ∧ d = 45 / 17) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l265_26584


namespace NUMINAMATH_GPT_part_a_part_b_l265_26516

-- Part A: Proving the specific values of p and q
theorem part_a (p q : ℝ) : 
  (∀ x : ℝ, (x + 3) ^ 2 + (7 * x + p) ^ 2 = (kx + m) ^ 2) ∧
  (∀ x : ℝ, (3 * x + 5) ^ 2 + (p * x + q) ^ 2 = (cx + d) ^ 2) → 
  p = 21 ∧ q = 35 :=
sorry

-- Part B: Proving the new polynomial is a square of a linear polynomial
theorem part_b (a b c A B C : ℝ) (hab : a ≠ 0) (hA : A ≠ 0) (hb : b ≠ 0) (hB : B ≠ 0)
  (habc : (∀ x : ℝ, (a * x + b) ^ 2 + (A * x + B) ^ 2 = (kx + m) ^ 2) ∧
         (∀ x : ℝ, (b * x + c) ^ 2 + (B * x + C) ^ 2 = (cx + d) ^ 2)) :
  ∀ x : ℝ, (c * x + a) ^ 2 + (C * x + A) ^ 2 = (lx + n) ^ 2 :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l265_26516


namespace NUMINAMATH_GPT_sum_series_eq_3_div_4_l265_26541

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end NUMINAMATH_GPT_sum_series_eq_3_div_4_l265_26541


namespace NUMINAMATH_GPT_sales_price_calculation_l265_26563

variables (C S : ℝ)
def gross_profit := 1.25 * C
def gross_profit_value := 30

theorem sales_price_calculation 
  (h1: gross_profit C = 30) :
  S = 54 :=
sorry

end NUMINAMATH_GPT_sales_price_calculation_l265_26563


namespace NUMINAMATH_GPT_integral_value_l265_26557

theorem integral_value (a : ℝ) (h : -35 * a^3 = -280) : ∫ x in a..2 * Real.exp 1, 1 / x = 1 := by
  sorry

end NUMINAMATH_GPT_integral_value_l265_26557


namespace NUMINAMATH_GPT_exists_not_in_range_f_l265_26558

noncomputable def f : ℝ → ℕ :=
sorry

axiom functional_equation : ∀ (x y : ℝ), f (x + (1 / f y)) = f (y + (1 / f x))

theorem exists_not_in_range_f :
  ∃ n : ℕ, ∀ x : ℝ, f x ≠ n :=
sorry

end NUMINAMATH_GPT_exists_not_in_range_f_l265_26558


namespace NUMINAMATH_GPT_no_k_for_linear_function_not_in_second_quadrant_l265_26538

theorem no_k_for_linear_function_not_in_second_quadrant :
  ¬∃ k : ℝ, ∀ x < 0, (k-1)*x + k ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_no_k_for_linear_function_not_in_second_quadrant_l265_26538


namespace NUMINAMATH_GPT_problem_l265_26565

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

def p : Prop := ∀ x : ℝ, x ≠ 0 → f x ≥ 4 ∧ (∃ x : ℝ, x > 0 ∧ f x = 4)

def q : Prop := ∀ (A B C : ℝ) (a b c : ℝ),
  A > B ↔ a > b

theorem problem : (¬p) ∧ q :=
sorry

end NUMINAMATH_GPT_problem_l265_26565


namespace NUMINAMATH_GPT_complaint_online_prob_l265_26550

/-- Define the various probability conditions -/
def prob_online := 4 / 5
def prob_store := 1 / 5
def qual_rate_online := 17 / 20
def qual_rate_store := 9 / 10
def non_qual_rate_online := 1 - qual_rate_online
def non_qual_rate_store := 1 - qual_rate_store
def prob_complaint_online := prob_online * non_qual_rate_online
def prob_complaint_store := prob_store * non_qual_rate_store
def total_prob_complaint := prob_complaint_online + prob_complaint_store

/-- The theorem states that given the conditions, the probability of an online purchase given a complaint is 6/7 -/
theorem complaint_online_prob : 
    (prob_complaint_online / total_prob_complaint) = 6 / 7 := 
by
    sorry

end NUMINAMATH_GPT_complaint_online_prob_l265_26550


namespace NUMINAMATH_GPT_marble_distribution_correct_l265_26549

def num_ways_to_distribute_marbles : ℕ :=
  -- Given:
  -- Evan divides 100 marbles among three volunteers with each getting at least one marble
  -- Lewis selects a positive integer n > 1 and for each volunteer, steals exactly 1/n of marbles if possible.
  -- Prove that the number of ways to distribute the marbles such that Lewis cannot steal from all volunteers
  3540

theorem marble_distribution_correct :
  num_ways_to_distribute_marbles = 3540 :=
sorry

end NUMINAMATH_GPT_marble_distribution_correct_l265_26549


namespace NUMINAMATH_GPT_sqrt_sum_inequality_l265_26567

open Real

theorem sqrt_sum_inequality (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 2) : 
  sqrt (2 * x + 1) + sqrt (2 * y + 1) + sqrt (2 * z + 1) ≤ sqrt 21 :=
sorry

end NUMINAMATH_GPT_sqrt_sum_inequality_l265_26567


namespace NUMINAMATH_GPT_total_pennies_donated_l265_26537

def cassandra_pennies : ℕ := 5000
def james_pennies : ℕ := cassandra_pennies - 276
def total_pennies : ℕ := cassandra_pennies + james_pennies

theorem total_pennies_donated : total_pennies = 9724 := by
  sorry

end NUMINAMATH_GPT_total_pennies_donated_l265_26537


namespace NUMINAMATH_GPT_square_area_from_triangle_perimeter_l265_26523

noncomputable def perimeter_triangle (a b c : ℝ) : ℝ := a + b + c

noncomputable def side_length_square (perimeter : ℝ) : ℝ := perimeter / 4

noncomputable def area_square (side_length : ℝ) : ℝ := side_length * side_length

theorem square_area_from_triangle_perimeter 
  (a b c : ℝ) 
  (h₁ : a = 5.5) 
  (h₂ : b = 7.5) 
  (h₃ : c = 11) 
  (h₄ : perimeter_triangle a b c = 24) 
  : area_square (side_length_square (perimeter_triangle a b c)) = 36 := 
by 
  simp [perimeter_triangle, side_length_square, area_square, h₁, h₂, h₃, h₄]
  sorry

end NUMINAMATH_GPT_square_area_from_triangle_perimeter_l265_26523


namespace NUMINAMATH_GPT_purely_imaginary_iff_m_eq_1_l265_26502

theorem purely_imaginary_iff_m_eq_1 (m : ℝ) :
  (m^2 - 1 = 0 ∧ m + 1 ≠ 0) → m = 1 :=
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_iff_m_eq_1_l265_26502


namespace NUMINAMATH_GPT_polar_to_cartesian_l265_26596

theorem polar_to_cartesian (θ ρ : ℝ) (h : ρ = 2 * Real.cos θ) :
  ∃ x y : ℝ, (x=ρ*Real.cos θ ∧ y=ρ*Real.sin θ) ∧ (x-1)^2 + y^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_l265_26596


namespace NUMINAMATH_GPT_james_drive_time_to_canada_l265_26505

theorem james_drive_time_to_canada : 
  ∀ (distance speed stop_time : ℕ), 
    speed = 60 → 
    distance = 360 → 
    stop_time = 1 → 
    (distance / speed) + stop_time = 7 :=
by
  intros distance speed stop_time h1 h2 h3
  sorry

end NUMINAMATH_GPT_james_drive_time_to_canada_l265_26505


namespace NUMINAMATH_GPT_isosceles_right_triangle_area_l265_26548

-- Define the isosceles right triangle and its properties

theorem isosceles_right_triangle_area 
  (h : ℝ)
  (hyp : h = 6) :
  let l : ℝ := h / Real.sqrt 2
  let A : ℝ := (l^2) / 2
  A = 9 :=
by
  -- The proof steps are skipped with sorry
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_area_l265_26548


namespace NUMINAMATH_GPT_sum_of_operations_l265_26573

def operation (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem sum_of_operations : operation 12 5 + operation 8 3 = 174 := by
  sorry

end NUMINAMATH_GPT_sum_of_operations_l265_26573


namespace NUMINAMATH_GPT_smallest_value_of_y_l265_26592

theorem smallest_value_of_y : 
  (∃ y : ℝ, 6 * y^2 - 41 * y + 55 = 0 ∧ ∀ z : ℝ, 6 * z^2 - 41 * z + 55 = 0 → y ≤ z) →
  ∃ y : ℝ, y = 2.5 :=
by sorry

end NUMINAMATH_GPT_smallest_value_of_y_l265_26592


namespace NUMINAMATH_GPT_certain_percentage_l265_26599

variable {x p : ℝ}

theorem certain_percentage (h1 : 0.40 * x = 160) : p * x = 200 ↔ p = 0.5 := 
by
  sorry

end NUMINAMATH_GPT_certain_percentage_l265_26599


namespace NUMINAMATH_GPT_maximum_ab_ac_bc_l265_26544

theorem maximum_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 5) : 
  ab + ac + bc ≤ 25 / 6 :=
sorry

end NUMINAMATH_GPT_maximum_ab_ac_bc_l265_26544


namespace NUMINAMATH_GPT_area_of_triangle_MEF_correct_l265_26506

noncomputable def area_of_triangle_MEF : ℝ :=
  let r := 10
  let chord_length := 12
  let parallel_segment_length := 15
  let angle_MOA := 30.0
  (1 / 2) * chord_length * (2 * Real.sqrt 21)

theorem area_of_triangle_MEF_correct :
  area_of_triangle_MEF = 12 * Real.sqrt 21 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_area_of_triangle_MEF_correct_l265_26506


namespace NUMINAMATH_GPT_conic_section_is_hyperbola_l265_26583

theorem conic_section_is_hyperbola : 
  ∀ (x y : ℝ), x^2 + 2 * x - 8 * y^2 = 0 → (∃ a b h k : ℝ, (x + 1)^2 / a^2 - (y - 0)^2 / b^2 = 1) := 
by 
  intros x y h_eq;
  sorry

end NUMINAMATH_GPT_conic_section_is_hyperbola_l265_26583


namespace NUMINAMATH_GPT_smallest_solution_l265_26515

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) 
    (h_eq : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) : x = 4 - Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_solution_l265_26515


namespace NUMINAMATH_GPT_total_flag_distance_moved_l265_26507

def flagpole_length : ℕ := 60

def initial_raise_distance : ℕ := flagpole_length

def lower_to_half_mast_distance : ℕ := flagpole_length / 2

def raise_from_half_mast_distance : ℕ := flagpole_length / 2

def final_lower_distance : ℕ := flagpole_length

theorem total_flag_distance_moved :
  initial_raise_distance + lower_to_half_mast_distance + raise_from_half_mast_distance + final_lower_distance = 180 :=
by
  sorry

end NUMINAMATH_GPT_total_flag_distance_moved_l265_26507


namespace NUMINAMATH_GPT_final_solution_percentage_l265_26579

variable (initial_volume replaced_fraction : ℝ)
variable (initial_concentration replaced_concentration : ℝ)

noncomputable
def final_acid_percentage (initial_volume replaced_fraction initial_concentration replaced_concentration : ℝ) : ℝ :=
  let remaining_volume := initial_volume * (1 - replaced_fraction)
  let replaced_volume := initial_volume * replaced_fraction
  let remaining_acid := remaining_volume * initial_concentration
  let replaced_acid := replaced_volume * replaced_concentration
  let total_acid := remaining_acid + replaced_acid
  let final_volume := initial_volume
  (total_acid / final_volume) * 100

theorem final_solution_percentage :
  final_acid_percentage 100 0.5 0.5 0.3 = 40 :=
by
  sorry

end NUMINAMATH_GPT_final_solution_percentage_l265_26579


namespace NUMINAMATH_GPT_ratio_of_white_socks_l265_26585

theorem ratio_of_white_socks 
  (total_socks : ℕ) (blue_socks : ℕ)
  (h_total_socks : total_socks = 180)
  (h_blue_socks : blue_socks = 60) :
  (total_socks - blue_socks) * 3 = total_socks * 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_white_socks_l265_26585


namespace NUMINAMATH_GPT_min_eq_one_implies_x_eq_one_l265_26574

open Real

theorem min_eq_one_implies_x_eq_one (x : ℝ) (h : min (1/2 + x) (x^2) = 1) : x = 1 := 
sorry

end NUMINAMATH_GPT_min_eq_one_implies_x_eq_one_l265_26574


namespace NUMINAMATH_GPT_f_neg2_eq_neg4_l265_26560

noncomputable def f (x : ℝ) : ℝ :=
  if hx : x >= 0 then 3^x - 2*x - 1
  else - (3^(-x) - 2*(-x) - 1)

theorem f_neg2_eq_neg4
: f (-2) = -4 :=
by
  sorry

end NUMINAMATH_GPT_f_neg2_eq_neg4_l265_26560


namespace NUMINAMATH_GPT_lychee_harvest_l265_26518

theorem lychee_harvest : 
  let last_year_red := 350
  let last_year_yellow := 490
  let this_year_red := 500
  let this_year_yellow := 700
  let sold_red := 2/3 * this_year_red
  let sold_yellow := 3/7 * this_year_yellow
  let remaining_red_after_sale := this_year_red - sold_red
  let remaining_yellow_after_sale := this_year_yellow - sold_yellow
  let family_ate_red := 3/5 * remaining_red_after_sale
  let family_ate_yellow := 4/9 * remaining_yellow_after_sale
  let remaining_red := remaining_red_after_sale - family_ate_red
  let remaining_yellow := remaining_yellow_after_sale - family_ate_yellow
  (this_year_red - last_year_red) / last_year_red * 100 = 42.86
  ∧ (this_year_yellow - last_year_yellow) / last_year_yellow * 100 = 42.86
  ∧ remaining_red = 67
  ∧ remaining_yellow = 223 :=
by
    intros
    sorry

end NUMINAMATH_GPT_lychee_harvest_l265_26518


namespace NUMINAMATH_GPT_sum_a_b_is_nine_l265_26593

theorem sum_a_b_is_nine (a b : ℤ) (h1 : a > b) (h2 : b > 0) 
    (h3 : (b + 2 - a)^2 + (a - b)^2 + (b + 2 + a)^2 + (a + b)^2 = 324) 
    (h4 : ∃ a' b', a' = a ∧ b' = b ∧ (b + 2 - a) * 1 = -(b + 2 - a)) : 
  a + b = 9 :=
sorry

end NUMINAMATH_GPT_sum_a_b_is_nine_l265_26593


namespace NUMINAMATH_GPT_stratified_sampling_is_reasonable_l265_26529

-- Defining our conditions and stating our theorem
def flat_land := 150
def ditch_land := 30
def sloped_land := 90
def total_acres := 270
def sampled_acres := 18
def sampling_ratio := sampled_acres / total_acres

def flat_land_sampled := flat_land * sampling_ratio
def ditch_land_sampled := ditch_land * sampling_ratio
def sloped_land_sampled := sloped_land * sampling_ratio

theorem stratified_sampling_is_reasonable :
  flat_land_sampled = 10 ∧
  ditch_land_sampled = 2 ∧
  sloped_land_sampled = 6 := 
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_is_reasonable_l265_26529


namespace NUMINAMATH_GPT_find_x_from_expression_l265_26536

theorem find_x_from_expression
  (y : ℚ)
  (h1 : y = -3/2)
  (h2 : -2 * (x : ℚ) - y^2 = 0.25) : 
  x = -5/4 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_from_expression_l265_26536


namespace NUMINAMATH_GPT_expand_product_eq_l265_26577

theorem expand_product_eq :
  (∀ (x : ℤ), (x^3 - 3 * x^2 + 3 * x - 1) * (x^2 + 3 * x + 3) = x^5 - 3 * x^3 - x^2 + 3 * x) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_expand_product_eq_l265_26577


namespace NUMINAMATH_GPT_inequality_proof_l265_26534

theorem inequality_proof (a b : ℝ) : 
  (a^4 + a^2 * b^2 + b^4) / 3 ≥ (a^3 * b + b^3 * a) / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l265_26534


namespace NUMINAMATH_GPT_number_of_students_l265_26503

theorem number_of_students (N T : ℕ) (h1 : T = 80 * N)
  (h2 : 95 * (N - 5) = T - 100) : N = 25 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l265_26503


namespace NUMINAMATH_GPT_students_journals_l265_26517

theorem students_journals :
  ∃ u v : ℕ, 
    u + v = 75000 ∧ 
    (7 * u + 2 * v = 300000) ∧ 
    (∃ b g : ℕ, b = u * 7 / 300 ∧ g = v * 2 / 300 ∧ b = 700 ∧ g = 300) :=
by {
  -- The proving steps will go here
  sorry
}

end NUMINAMATH_GPT_students_journals_l265_26517


namespace NUMINAMATH_GPT_number_of_pupils_l265_26572

theorem number_of_pupils (n : ℕ) : (83 - 63) / n = 1 / 2 → n = 40 :=
by
  intro h
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_number_of_pupils_l265_26572


namespace NUMINAMATH_GPT_vector_addition_and_scalar_multiplication_l265_26540

-- Specify the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 1)

-- Define the theorem we want to prove
theorem vector_addition_and_scalar_multiplication :
  a + 2 • b = (-3, 4) :=
sorry

end NUMINAMATH_GPT_vector_addition_and_scalar_multiplication_l265_26540


namespace NUMINAMATH_GPT_prove_parabola_points_l265_26569

open Real

noncomputable def parabola_equation (x y : ℝ) : Prop := x^2 = 4 * y

noncomputable def dist_to_focus (x y focus_x focus_y : ℝ) : ℝ :=
  (sqrt ((x - focus_x)^2 + (y - focus_y)^2))

theorem prove_parabola_points :
  ∀ (x1 y1 x2 y2 : ℝ),
  parabola_equation x1 y1 →
  parabola_equation x2 y2 →
  dist_to_focus x1 y1 0 1 - dist_to_focus x2 y2 0 1 = 2 →
  (y1 + x1^2 - y2 - x2^2 = 10) :=
by
  intros x1 y1 x2 y2 h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_prove_parabola_points_l265_26569


namespace NUMINAMATH_GPT_matches_in_each_matchbook_l265_26581

-- Conditions given in the problem
def one_stamp_worth_matches (s : ℕ) : Prop := s = 12
def tonya_initial_stamps (t : ℕ) : Prop := t = 13
def tonya_final_stamps (t : ℕ) : Prop := t = 3
def jimmy_initial_matchbooks (j : ℕ) : Prop := j = 5

-- Goal: prove M = 24
theorem matches_in_each_matchbook (M : ℕ) (s t_initial t_final j : ℕ) 
  (h1 : one_stamp_worth_matches s) 
  (h2 : tonya_initial_stamps t_initial) 
  (h3 : tonya_final_stamps t_final) 
  (h4 : jimmy_initial_matchbooks j) : M = 24 := by
  sorry

end NUMINAMATH_GPT_matches_in_each_matchbook_l265_26581


namespace NUMINAMATH_GPT_find_prime_pairs_l265_26554

open Nat

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem find_prime_pairs :
  ∀ m n : ℕ,
  is_prime m → is_prime n → (m < n ∧ n < 5 * m) → is_prime (m + 3 * n) →
  (m = 2 ∧ (n = 3 ∨ n = 5 ∨ n = 7)) :=
by
  sorry

end NUMINAMATH_GPT_find_prime_pairs_l265_26554


namespace NUMINAMATH_GPT_first_term_of_geometric_sequence_l265_26504

theorem first_term_of_geometric_sequence
  (a r : ℚ) -- where a is the first term and r is the common ratio
  (h1 : a * r^4 = 45) -- fifth term condition
  (h2 : a * r^5 = 60) -- sixth term condition
  : a = 1215 / 256 := 
sorry

end NUMINAMATH_GPT_first_term_of_geometric_sequence_l265_26504


namespace NUMINAMATH_GPT_ab_sum_not_one_l265_26511

theorem ab_sum_not_one (a b : ℝ) : a^2 + 2*a*b + b^2 + a + b - 2 ≠ 0 → a + b ≠ 1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_ab_sum_not_one_l265_26511


namespace NUMINAMATH_GPT_evaluate_division_l265_26533

theorem evaluate_division : 64 / 0.08 = 800 := by
  sorry

end NUMINAMATH_GPT_evaluate_division_l265_26533


namespace NUMINAMATH_GPT_job_completion_time_l265_26512

theorem job_completion_time (A_rate D_rate Combined_rate : ℝ) (hA : A_rate = 1 / 3) (hD : D_rate = 1 / 6) (hCombined : Combined_rate = A_rate + D_rate) :
  (1 / Combined_rate) = 2 :=
by sorry

end NUMINAMATH_GPT_job_completion_time_l265_26512


namespace NUMINAMATH_GPT_train_passes_tree_in_28_seconds_l265_26571

def km_per_hour_to_meter_per_second (km_per_hour : ℕ) : ℕ :=
  km_per_hour * 1000 / 3600

def pass_tree_time (length : ℕ) (speed_kmh : ℕ) : ℕ :=
  length / (km_per_hour_to_meter_per_second speed_kmh)

theorem train_passes_tree_in_28_seconds :
  pass_tree_time 490 63 = 28 :=
by
  sorry

end NUMINAMATH_GPT_train_passes_tree_in_28_seconds_l265_26571


namespace NUMINAMATH_GPT_cos_100_eq_neg_sqrt_l265_26559

theorem cos_100_eq_neg_sqrt (a : ℝ) (h : Real.sin (80 * Real.pi / 180) = a) : 
  Real.cos (100 * Real.pi / 180) = -Real.sqrt (1 - a^2) := 
sorry

end NUMINAMATH_GPT_cos_100_eq_neg_sqrt_l265_26559


namespace NUMINAMATH_GPT_inverse_proportion_quadrants_l265_26546

theorem inverse_proportion_quadrants (k : ℝ) : (∀ x, x ≠ 0 → ((x < 0 → (2 - k) / x > 0) ∧ (x > 0 → (2 - k) / x < 0))) → k > 2 :=
by sorry

end NUMINAMATH_GPT_inverse_proportion_quadrants_l265_26546


namespace NUMINAMATH_GPT_recreation_percentage_correct_l265_26588

noncomputable def recreation_percentage (W : ℝ) : ℝ :=
  let recreation_two_weeks_ago := 0.25 * W
  let wages_last_week := 0.95 * W
  let recreation_last_week := 0.35 * (0.95 * W)
  let wages_this_week := 0.95 * W * 0.85
  let recreation_this_week := 0.45 * (0.95 * W * 0.85)
  (recreation_this_week / recreation_two_weeks_ago) * 100

theorem recreation_percentage_correct (W : ℝ) : recreation_percentage W = 145.35 :=
by
  sorry

end NUMINAMATH_GPT_recreation_percentage_correct_l265_26588


namespace NUMINAMATH_GPT_shadow_length_minor_fullness_l265_26509

/-
An arithmetic sequence {a_n} where the length of shadows a_i decreases by the same amount, the conditions are:
1. The sum of the shadows on the Winter Solstice (a_1), the Beginning of Spring (a_4), and the Vernal Equinox (a_7) is 315 cun.
2. The sum of the shadows on the first nine solar terms is 855 cun.

We need to prove that the shadow length on Minor Fullness day (a_11) is 35 cun (i.e., 3 chi and 5 cun).
-/
theorem shadow_length_minor_fullness 
  (a : ℕ → ℕ) 
  (d : ℤ)
  (h1 : a 1 + a 4 + a 7 = 315) 
  (h2 : 9 * a 1 + 36 * d = 855) 
  (seq : ∀ n : ℕ, a n = a 1 + (n - 1) * d) :
  a 11 = 35 := 
by 
  sorry

end NUMINAMATH_GPT_shadow_length_minor_fullness_l265_26509


namespace NUMINAMATH_GPT_average_height_l265_26500

def heights : List ℕ := [145, 142, 138, 136, 143, 146, 138, 144, 137, 141]

theorem average_height :
  (heights.sum : ℕ) / heights.length = 141 := by
  sorry

end NUMINAMATH_GPT_average_height_l265_26500


namespace NUMINAMATH_GPT_odd_prime_2wy_factors_l265_26501

theorem odd_prime_2wy_factors (w y : ℕ) (h1 : Nat.Prime w) (h2 : Nat.Prime y) (h3 : ¬ Even w) (h4 : ¬ Even y) (h5 : w < y) (h6 : Nat.totient (2 * w * y) = 8) :
  w = 3 :=
sorry

end NUMINAMATH_GPT_odd_prime_2wy_factors_l265_26501


namespace NUMINAMATH_GPT_distinct_real_roots_find_k_values_l265_26582

-- Question 1: Prove the equation has two distinct real roots
theorem distinct_real_roots (k : ℝ) : 
  (2 * k + 1) ^ 2 - 4 * (k ^ 2 + k) > 0 :=
  by sorry

-- Question 2: Find the values of k when triangle ABC is a right triangle
theorem find_k_values (k : ℝ) : 
  (k = 3 ∨ k = 12) ↔ 
  (∃ (AB AC : ℝ), 
    AB ≠ AC ∧ AB = k ∧ AC = k + 1 ∧ (AB^2 + AC^2 = 5^2 ∨ AC^2 + 5^2 = AB^2)) :=
  by sorry

end NUMINAMATH_GPT_distinct_real_roots_find_k_values_l265_26582


namespace NUMINAMATH_GPT_speed_of_first_car_l265_26543

-- Define the conditions
def t : ℝ := 3.5
def v : ℝ := sorry -- (To be solved in the proof)
def speed_second_car : ℝ := 58
def total_distance : ℝ := 385

-- The distance each car travels after t hours
def distance_first_car : ℝ := v * t
def distance_second_car : ℝ := speed_second_car * t

-- The equation representing the total distance between the two cars after 3.5 hours
def equation := distance_first_car + distance_second_car = total_distance

-- The main theorem stating the speed of the first car
theorem speed_of_first_car : v = 52 :=
by
  -- The important proof steps would go here solving the equation "equation".
  sorry

end NUMINAMATH_GPT_speed_of_first_car_l265_26543


namespace NUMINAMATH_GPT_decreasing_interval_range_of_a_l265_26597

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem decreasing_interval :
  (∀ x > 0, deriv f x = 1 + log x) →
  { x : ℝ | 0 < x ∧ x < 1/e } = { x | 0 < x ∧ deriv f x < 0 } :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x ≥ -x^2 + a * x - 6) →
  a ≤ 5 + log 2 :=
sorry

end NUMINAMATH_GPT_decreasing_interval_range_of_a_l265_26597


namespace NUMINAMATH_GPT_min_value_pq_l265_26508

theorem min_value_pq (p q : ℝ) (hp : 0 < p) (hq : 0 < q)
  (h1 : p^2 - 8 * q ≥ 0)
  (h2 : 4 * q^2 - 4 * p ≥ 0) :
  p + q ≥ 6 :=
sorry

end NUMINAMATH_GPT_min_value_pq_l265_26508


namespace NUMINAMATH_GPT_lara_sees_leo_for_six_minutes_l265_26576

-- Define constants for speeds and initial distances
def lara_speed : ℕ := 60
def leo_speed : ℕ := 40
def initial_distance : ℕ := 1
def time_to_minutes (t : ℚ) : ℚ := t * 60
-- Define the condition that proves Lara can see Leo for 6 minutes
theorem lara_sees_leo_for_six_minutes :
  lara_speed > leo_speed ∧
  initial_distance > 0 ∧
  (initial_distance : ℚ) / (lara_speed - leo_speed) * 2 = (6 : ℚ) / 60 :=
by
  sorry

end NUMINAMATH_GPT_lara_sees_leo_for_six_minutes_l265_26576


namespace NUMINAMATH_GPT_factorize_x4_minus_16y4_l265_26530

theorem factorize_x4_minus_16y4 (x y : ℚ) : 
  x^4 - 16 * y^4 = (x^2 + 4 * y^2) * (x + 2 * y) * (x - 2 * y) := 
by 
  sorry

end NUMINAMATH_GPT_factorize_x4_minus_16y4_l265_26530


namespace NUMINAMATH_GPT_largest_root_l265_26587

theorem largest_root (p q r : ℝ) (h1 : p + q + r = 3) (h2 : p * q + p * r + q * r = -6) (h3 : p * q * r = -8) :
  max (max p q) r = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end NUMINAMATH_GPT_largest_root_l265_26587


namespace NUMINAMATH_GPT_prove_non_negative_axbycz_l265_26598

variable {a b c x y z : ℝ}

theorem prove_non_negative_axbycz
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) :
  a * x + b * y + c * z ≥ 0 := 
sorry

end NUMINAMATH_GPT_prove_non_negative_axbycz_l265_26598


namespace NUMINAMATH_GPT_smallest_positive_e_for_polynomial_l265_26552

theorem smallest_positive_e_for_polynomial :
  ∃ a b c d e : ℤ, e = 168 ∧
  (a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e = 0) ∧
  (a * (x + 3) * (x - 7) * (x - 8) * (4 * x + 1) = a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e) := sorry

end NUMINAMATH_GPT_smallest_positive_e_for_polynomial_l265_26552


namespace NUMINAMATH_GPT_find_six_y_minus_four_squared_l265_26580

theorem find_six_y_minus_four_squared (y : ℝ) (h : 3 * y^2 + 6 = 5 * y + 15) :
  (6 * y - 4)^2 = 134 :=
by
  sorry

end NUMINAMATH_GPT_find_six_y_minus_four_squared_l265_26580


namespace NUMINAMATH_GPT_smallest_n_l265_26566

def power_tower (a : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => a
  | (n+1) => a ^ (power_tower a n)

def pow3_cubed : ℕ := 3 ^ (3 ^ (3 ^ 3))

theorem smallest_n : ∃ n, (∃ k : ℕ, (power_tower 2 n) = k ∧ k > pow3_cubed) ∧ ∀ m, (∃ k : ℕ, (power_tower 2 m) = k ∧ k > pow3_cubed) → m ≥ n :=
  by
  sorry

end NUMINAMATH_GPT_smallest_n_l265_26566


namespace NUMINAMATH_GPT_rotational_homothety_commutes_l265_26556

-- Definitions for our conditions
variable (H1 H2 : Point → Point)

-- Definition of rotational homothety. 
-- You would define it based on your bespoke library/formalization.
axiom is_rot_homothety : ∀ (H : Point → Point), Prop

-- Main theorem statement
theorem rotational_homothety_commutes (H1 H2 : Point → Point) (A : Point) 
    (h1_rot : is_rot_homothety H1) (h2_rot : is_rot_homothety H2) : 
    (H1 ∘ H2 = H2 ∘ H1) ↔ (H1 (H2 A) = H2 (H1 A)) :=
sorry

end NUMINAMATH_GPT_rotational_homothety_commutes_l265_26556


namespace NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l265_26513

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ),
  ρ = 5 → θ = π / 6 → φ = π / 3 →
  let x := ρ * (Real.sin φ * Real.cos θ)
  let y := ρ * (Real.sin φ * Real.sin θ)
  let z := ρ * Real.cos φ
  x = 15 / 4 ∧ y = 5 * Real.sqrt 3 / 4 ∧ z = 2.5 :=
by
  intros ρ θ φ hρ hθ hφ
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l265_26513


namespace NUMINAMATH_GPT_sum_of_29_12_23_is_64_sixtyfour_is_two_to_six_l265_26545

theorem sum_of_29_12_23_is_64: 29 + 12 + 23 = 64 := sorry

theorem sixtyfour_is_two_to_six:
  64 = 2^6 := sorry

end NUMINAMATH_GPT_sum_of_29_12_23_is_64_sixtyfour_is_two_to_six_l265_26545


namespace NUMINAMATH_GPT_solve_for_y_in_equation_l265_26570

theorem solve_for_y_in_equation : ∃ y : ℝ, 7 * (2 * y - 3) + 5 = -3 * (4 - 5 * y) ∧ y = -4 :=
by
  use -4
  sorry

end NUMINAMATH_GPT_solve_for_y_in_equation_l265_26570


namespace NUMINAMATH_GPT_correct_calculation_l265_26590

theorem correct_calculation (x : ℤ) (h : x - 32 = 33) : x + 32 = 97 := 
by 
  sorry

end NUMINAMATH_GPT_correct_calculation_l265_26590


namespace NUMINAMATH_GPT_domain_of_f_l265_26539

noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt (2 - x)) + Real.log (x+1)

theorem domain_of_f : {x : ℝ | (2 - x) > 0 ∧ (x + 1) > 0} = {x : ℝ | -1 < x ∧ x < 2} := 
by
  ext x
  simp
  sorry

end NUMINAMATH_GPT_domain_of_f_l265_26539


namespace NUMINAMATH_GPT_age_of_eldest_child_l265_26522

-- Define the conditions as hypotheses
def child_ages_sum_equals_50 (x : ℕ) : Prop :=
  x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 50

-- Define the main theorem to prove the age of the eldest child
theorem age_of_eldest_child (x : ℕ) (h : child_ages_sum_equals_50 x) : x + 8 = 14 :=
sorry

end NUMINAMATH_GPT_age_of_eldest_child_l265_26522


namespace NUMINAMATH_GPT_prove_a2_b2_c2_zero_l265_26578

theorem prove_a2_b2_c2_zero (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) : a^2 + b^2 + c^2 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_prove_a2_b2_c2_zero_l265_26578


namespace NUMINAMATH_GPT_garden_area_eq_450_l265_26551

theorem garden_area_eq_450
  (width length : ℝ)
  (fencing : ℝ := 60) 
  (length_eq_twice_width : length = 2 * width)
  (fencing_eq : 2 * width + length = fencing) :
  width * length = 450 := by
  sorry

end NUMINAMATH_GPT_garden_area_eq_450_l265_26551


namespace NUMINAMATH_GPT_ratio_of_red_to_total_l265_26568

def hanna_erasers : Nat := 4
def tanya_total_erasers : Nat := 20

def rachel_erasers (hanna_erasers : Nat) : Nat :=
  hanna_erasers / 2

def tanya_red_erasers (rachel_erasers : Nat) : Nat :=
  2 * (rachel_erasers + 3)

theorem ratio_of_red_to_total (hanna_erasers tanya_total_erasers : Nat)
  (hanna_has_4 : hanna_erasers = 4) 
  (tanya_total_is_20 : tanya_total_erasers = 20) 
  (twice_as_many : hanna_erasers = 2 * (rachel_erasers hanna_erasers)) 
  (three_less_than_half : rachel_erasers hanna_erasers = (1 / 2:Rat) * (tanya_red_erasers (rachel_erasers hanna_erasers)) - 3) :
  (tanya_red_erasers (rachel_erasers hanna_erasers)) / tanya_total_erasers = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_red_to_total_l265_26568


namespace NUMINAMATH_GPT_a_plus_b_eq_2007_l265_26526

theorem a_plus_b_eq_2007 (a b : ℕ) (ha : Prime a) (hb : Odd b)
  (h : a^2 + b = 2009) : a + b = 2007 :=
by
  sorry

end NUMINAMATH_GPT_a_plus_b_eq_2007_l265_26526


namespace NUMINAMATH_GPT_range_of_a_l265_26555

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a <= x ∧ x < y ∧ y <= b → f y <= f x

theorem range_of_a (f : ℝ → ℝ) :
  odd_function f →
  decreasing_on_interval f (-1) 1 →
  (∀ a : ℝ, 0 < a ∧ a < 1 → f (1 - a) + f (2 * a - 1) < 0) →
  (∀ a : ℝ, 0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_GPT_range_of_a_l265_26555


namespace NUMINAMATH_GPT_lindsey_integer_l265_26562

theorem lindsey_integer (n : ℕ) (a b c : ℤ) (h1 : n < 50)
                        (h2 : n = 6 * a - 1)
                        (h3 : n = 8 * b - 5)
                        (h4 : n = 3 * c + 2) :
  n = 41 := 
  by sorry

end NUMINAMATH_GPT_lindsey_integer_l265_26562


namespace NUMINAMATH_GPT_Shara_borrowed_6_months_ago_l265_26595

theorem Shara_borrowed_6_months_ago (X : ℝ) (h1 : ∃ n : ℕ, (X / 2 - 4 * 10 = 20) ∧ (X / 2 = n * 10)) :
  ∃ m : ℕ, m * 10 = X / 2 → m = 6 := 
sorry

end NUMINAMATH_GPT_Shara_borrowed_6_months_ago_l265_26595


namespace NUMINAMATH_GPT_karen_box_crayons_l265_26586

theorem karen_box_crayons (judah_crayons : ℕ) (gilbert_crayons : ℕ) (beatrice_crayons : ℕ) (karen_crayons : ℕ)
  (h1 : judah_crayons = 8)
  (h2 : gilbert_crayons = 4 * judah_crayons)
  (h3 : beatrice_crayons = 2 * gilbert_crayons)
  (h4 : karen_crayons = 2 * beatrice_crayons) :
  karen_crayons = 128 :=
by
  sorry

end NUMINAMATH_GPT_karen_box_crayons_l265_26586


namespace NUMINAMATH_GPT_f_value_2009_l265_26591

noncomputable def f : ℝ → ℝ := sorry

theorem f_value_2009
    (h1 : ∀ x y : ℝ, f (x * y) = f x * f y)
    (h2 : f 0 ≠ 0) :
    f 2009 = 1 :=
sorry

end NUMINAMATH_GPT_f_value_2009_l265_26591


namespace NUMINAMATH_GPT_child_growth_l265_26542

-- Define variables for heights
def current_height : ℝ := 41.5
def previous_height : ℝ := 38.5

-- Define the problem statement in Lean 4
theorem child_growth :
  current_height - previous_height = 3 :=
by 
  sorry

end NUMINAMATH_GPT_child_growth_l265_26542
