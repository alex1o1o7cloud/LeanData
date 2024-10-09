import Mathlib

namespace correct_microorganism_dilution_statement_l1068_106847

def microorganism_dilution_conditions (A B C D : Prop) : Prop :=
  (A ↔ ∀ (dilutions : ℕ) (n : ℕ), 1000 ≤ dilutions ∧ dilutions ≤ 10000000) ∧
  (B ↔ ∀ (dilutions : ℕ) (actinomycetes : ℕ), dilutions = 1000 ∨ dilutions = 10000 ∨ dilutions = 100000) ∧
  (C ↔ ∀ (dilutions : ℕ) (fungi : ℕ), dilutions = 1000 ∨ dilutions = 10000 ∨ dilutions = 100000) ∧
  (D ↔ ∀ (dilutions : ℕ) (bacteria_first_time : ℕ), 10 ≤ dilutions ∧ dilutions ≤ 10000000)

theorem correct_microorganism_dilution_statement (A B C D : Prop)
  (h : microorganism_dilution_conditions A B C D) : D :=
sorry

end correct_microorganism_dilution_statement_l1068_106847


namespace surface_area_of_solid_block_l1068_106895

theorem surface_area_of_solid_block :
  let unit_cube_surface_area := 6
  let top_bottom_area := 2 * (3 * 5)
  let front_back_area := 2 * (3 * 5)
  let left_right_area := 2 * (3 * 1)
  top_bottom_area + front_back_area + left_right_area = 66 :=
by
  let unit_cube_surface_area := 6
  let top_bottom_area := 2 * (3 * 5)
  let front_back_area := 2 * (3 * 5)
  let left_right_area := 2 * (3 * 1)
  sorry

end surface_area_of_solid_block_l1068_106895


namespace incorrect_rounding_statement_l1068_106842

def rounded_to_nearest (n : ℝ) (accuracy : ℝ) : Prop :=
  ∃ (k : ℤ), abs (n - k * accuracy) < accuracy / 2

theorem incorrect_rounding_statement :
  ¬ rounded_to_nearest 23.9 10 :=
sorry

end incorrect_rounding_statement_l1068_106842


namespace pencils_inequalities_l1068_106874

theorem pencils_inequalities (x y : ℕ) :
  (3 * x < 48 ∧ 48 < 4 * x) ∧ (4 * y < 48 ∧ 48 < 5 * y) :=
sorry

end pencils_inequalities_l1068_106874


namespace how_many_raisins_did_bryce_receive_l1068_106896

def raisins_problem : Prop :=
  ∃ (B C : ℕ), B = C + 8 ∧ C = B / 3 ∧ B + C = 44 ∧ B = 33

theorem how_many_raisins_did_bryce_receive : raisins_problem :=
sorry

end how_many_raisins_did_bryce_receive_l1068_106896


namespace more_time_running_than_skipping_l1068_106870

def time_running : ℚ := 17 / 20
def time_skipping_rope : ℚ := 83 / 100

theorem more_time_running_than_skipping :
  time_running > time_skipping_rope :=
by
  -- sorry skips the proof
  sorry

end more_time_running_than_skipping_l1068_106870


namespace evaluate_trig_expression_l1068_106890

theorem evaluate_trig_expression :
  (Real.tan (π / 18) - Real.sqrt 3) * Real.sin (2 * π / 9) = -1 :=
by
  sorry

end evaluate_trig_expression_l1068_106890


namespace smallest_k_with_properties_l1068_106854

noncomputable def exists_coloring_and_function (k : ℕ) : Prop :=
  ∃ (colors : ℤ → Fin k) (f : ℤ → ℤ),
    (∀ m n : ℤ, colors m = colors n → f (m + n) = f m + f n) ∧
    (∃ m n : ℤ, f (m + n) ≠ f m + f n)

theorem smallest_k_with_properties : ∃ (k : ℕ), k > 0 ∧ exists_coloring_and_function k ∧
                                         (∀ k' : ℕ, k' > 0 ∧ k' < k → ¬ exists_coloring_and_function k') :=
by
  sorry

end smallest_k_with_properties_l1068_106854


namespace locus_of_P_is_single_ray_l1068_106831
  
noncomputable def M : ℝ × ℝ := (1, 0)
noncomputable def N : ℝ × ℝ := (3, 0)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))

theorem locus_of_P_is_single_ray (P : ℝ × ℝ) (h : distance P M - distance P N = 2) : 
∃ α : ℝ, P = (3 + α * (P.1 - 3), α * P.2) :=
sorry

end locus_of_P_is_single_ray_l1068_106831


namespace first_three_digits_of_decimal_part_of_10_pow_1001_plus_1_pow_9_div_8_l1068_106853

noncomputable def first_three_digits_of_decimal_part (x : ℝ) : ℕ :=
  -- here we would have the actual definition
  sorry

theorem first_three_digits_of_decimal_part_of_10_pow_1001_plus_1_pow_9_div_8 :
  first_three_digits_of_decimal_part ((10^1001 + 1)^((9:ℝ) / 8)) = 125 :=
sorry

end first_three_digits_of_decimal_part_of_10_pow_1001_plus_1_pow_9_div_8_l1068_106853


namespace find_three_digit_number_l1068_106827

def digits_to_num (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

theorem find_three_digit_number (a b c : ℕ) (h1 : 8 * a + 5 * b + c = 100) (h2 : a + b + c = 20) :
  digits_to_num a b c = 866 :=
by 
  sorry

end find_three_digit_number_l1068_106827


namespace cost_per_liter_of_fuel_l1068_106826

-- Definitions and conditions
def fuel_capacity : ℕ := 150
def initial_fuel : ℕ := 38
def change_received : ℕ := 14
def initial_money : ℕ := 350

-- Proof problem
theorem cost_per_liter_of_fuel :
  (initial_money - change_received) / (fuel_capacity - initial_fuel) = 3 :=
by
  sorry

end cost_per_liter_of_fuel_l1068_106826


namespace smaller_rectangle_dimensions_l1068_106834

theorem smaller_rectangle_dimensions (side_length : ℝ) (L W : ℝ) 
  (h1 : side_length = 10) 
  (h2 : L + 2 * L = side_length) 
  (h3 : W = L) : 
  L = 10 / 3 ∧ W = 10 / 3 :=
by 
  sorry

end smaller_rectangle_dimensions_l1068_106834


namespace pennies_on_friday_l1068_106886

-- Define the initial number of pennies and the function for doubling
def initial_pennies : Nat := 3
def double (n : Nat) : Nat := 2 * n

-- Prove the number of pennies on Friday
theorem pennies_on_friday : double (double (double (double initial_pennies))) = 48 := by
  sorry

end pennies_on_friday_l1068_106886


namespace javiers_household_legs_l1068_106887

-- Definitions given the problem conditions
def humans : ℕ := 6
def human_legs : ℕ := 2

def dogs : ℕ := 2
def dog_legs : ℕ := 4

def cats : ℕ := 1
def cat_legs : ℕ := 4

def parrots : ℕ := 1
def parrot_legs : ℕ := 2

def lizards : ℕ := 1
def lizard_legs : ℕ := 4

def stool_legs : ℕ := 3
def table_legs : ℕ := 4
def cabinet_legs : ℕ := 6

-- Problem statement
theorem javiers_household_legs :
  (humans * human_legs) + (dogs * dog_legs) + (cats * cat_legs) + (parrots * parrot_legs) +
  (lizards * lizard_legs) + stool_legs + table_legs + cabinet_legs = 43 := by
  -- We leave the proof as an exercise for the reader
  sorry

end javiers_household_legs_l1068_106887


namespace conic_section_is_ellipse_l1068_106838

/-- Given two fixed points (0, 2) and (4, -1) and the equation 
    sqrt(x^2 + (y - 2)^2) + sqrt((x - 4)^2 + (y + 1)^2) = 12, 
    prove that the conic section is an ellipse. -/
theorem conic_section_is_ellipse 
  (x y : ℝ)
  (h : Real.sqrt (x^2 + (y - 2)^2) + Real.sqrt ((x - 4)^2 + (y + 1)^2) = 12) :
  ∃ (F1 F2 : ℝ × ℝ), 
    F1 = (0, 2) ∧ 
    F2 = (4, -1) ∧ 
    ∀ (P : ℝ × ℝ), P = (x, y) → 
      Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + 
      Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 12 := 
sorry

end conic_section_is_ellipse_l1068_106838


namespace projection_is_negative_sqrt_10_l1068_106803

noncomputable def projection_of_AB_in_direction_of_AC : ℝ :=
  let A := (1, 1)
  let B := (-3, 3)
  let C := (4, 2)
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let dot_product := AB.1 * AC.1 + AB.2 * AC.2
  let magnitude_AC := Real.sqrt (AC.1^2 + AC.2^2)
  dot_product / magnitude_AC

theorem projection_is_negative_sqrt_10 :
  projection_of_AB_in_direction_of_AC = -Real.sqrt 10 :=
by
  sorry

end projection_is_negative_sqrt_10_l1068_106803


namespace arithmetic_sequence_geometric_ratio_l1068_106802

theorem arithmetic_sequence_geometric_ratio
  (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ∀ n : ℕ, a (n+1) = a n + d)
  (h_nonzero_d : d ≠ 0)
  (h_geo : (a 2) * (a 9) = (a 3) ^ 2)
  : (a 4 + a 5 + a 6) / (a 2 + a 3 + a 4) = (8 / 3) :=
by
  sorry

end arithmetic_sequence_geometric_ratio_l1068_106802


namespace water_level_decrease_3m_l1068_106875

-- Definitions from conditions
def increase (amount : ℝ) : ℝ := amount
def decrease (amount : ℝ) : ℝ := -amount

-- The claim to be proven
theorem water_level_decrease_3m : decrease 3 = -3 :=
by
  sorry

end water_level_decrease_3m_l1068_106875


namespace container_weight_l1068_106812

-- Definition of the problem conditions
def weight_of_copper_bar : ℕ := 90
def weight_of_steel_bar := weight_of_copper_bar + 20
def weight_of_tin_bar := weight_of_steel_bar / 2

-- Formal statement to be proven
theorem container_weight (n : ℕ) (h1 : weight_of_steel_bar = 2 * weight_of_tin_bar)
  (h2 : weight_of_steel_bar = weight_of_copper_bar + 20)
  (h3 : weight_of_copper_bar = 90) :
  20 * (weight_of_copper_bar + weight_of_steel_bar + weight_of_tin_bar) = 5100 := 
by sorry

end container_weight_l1068_106812


namespace candies_per_person_l1068_106852

def clowns : ℕ := 4
def children : ℕ := 30
def initial_candies : ℕ := 700
def candies_left : ℕ := 20

def total_people : ℕ := clowns + children
def candies_sold : ℕ := initial_candies - candies_left

theorem candies_per_person : candies_sold / total_people = 20 := by
  sorry

end candies_per_person_l1068_106852


namespace closest_perfect_square_to_315_l1068_106864

theorem closest_perfect_square_to_315 : ∃ n : ℤ, n^2 = 324 ∧
  (∀ m : ℤ, m ≠ n → (abs (315 - m^2) > abs (315 - n^2))) := 
sorry

end closest_perfect_square_to_315_l1068_106864


namespace packages_of_gum_l1068_106868

-- Define the conditions
variables (P : Nat) -- Number of packages Robin has

-- State the theorem
theorem packages_of_gum (h1 : 7 * P + 6 = 41) : P = 5 :=
by
  sorry

end packages_of_gum_l1068_106868


namespace product_of_roots_eq_negative_forty_nine_l1068_106857

theorem product_of_roots_eq_negative_forty_nine (t : ℝ) (h : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end product_of_roots_eq_negative_forty_nine_l1068_106857


namespace glass_volume_correct_l1068_106869

-- Define the constants based on the problem conditions
def pessimist_empty_percent : ℝ := 0.60
def optimist_full_percent : ℝ := 0.60
def water_difference : ℝ := 46

-- Let V be the volume of the glass
def glass_volume (V : ℝ) : Prop :=
  let pessimist_full := (1 - pessimist_empty_percent) * V
  let optimist_full := optimist_full_percent * V
  optimist_full - pessimist_full = water_difference

-- The theorem to prove
theorem glass_volume_correct :
  ∃ V : ℝ, glass_volume V ∧ V = 230 :=
by
  sorry

end glass_volume_correct_l1068_106869


namespace trigonometric_identity_l1068_106894

theorem trigonometric_identity (α : Real) (h : Real.sin (Real.pi + α) = -1/3) : 
  (Real.sin (2 * α) / Real.cos α) = 2/3 :=
by
  sorry

end trigonometric_identity_l1068_106894


namespace harmonic_mean_1999_2001_is_2000_l1068_106865

def harmonic_mean (a b : ℕ) : ℚ := (2 * a * b) / (a + b)

theorem harmonic_mean_1999_2001_is_2000 :
  abs (harmonic_mean 1999 2001 - 2000 : ℚ) < 1 := by
  -- Actual proof omitted
  sorry

end harmonic_mean_1999_2001_is_2000_l1068_106865


namespace piggy_bank_dimes_l1068_106809

theorem piggy_bank_dimes (q d : ℕ) 
  (h1 : q + d = 100) 
  (h2 : 25 * q + 10 * d = 1975) : 
  d = 35 :=
by
  -- skipping the proof
  sorry

end piggy_bank_dimes_l1068_106809


namespace rows_with_exactly_10_people_l1068_106832

theorem rows_with_exactly_10_people (x : ℕ) (total_people : ℕ) (row_nine_seat : ℕ) (row_ten_seat : ℕ) 
    (H1 : row_nine_seat = 9) (H2 : row_ten_seat = 10) 
    (H3 : total_people = 55) 
    (H4 : total_people = x * row_ten_seat + (6 - x) * row_nine_seat) 
    : x = 1 :=
by
  sorry

end rows_with_exactly_10_people_l1068_106832


namespace complex_multiplication_l1068_106888

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2 * i :=
by
  sorry

end complex_multiplication_l1068_106888


namespace area_triangle_l1068_106837

theorem area_triangle (A B C: ℝ) (AB AC : ℝ) (h1 : Real.sin A = 4 / 5) (h2 : AB * AC * Real.cos A = 6) :
  (1 / 2) * AB * AC * Real.sin A = 4 :=
by
  sorry

end area_triangle_l1068_106837


namespace find_base_k_l1068_106881

theorem find_base_k (k : ℕ) (h1 : 1 + 3 * k + 2 * k^2 = 30) : k = 4 :=
by sorry

end find_base_k_l1068_106881


namespace triangle_area_l1068_106872

theorem triangle_area (a b c : ℝ) (ha : a = 6) (hb : b = 5) (hc : c = 5) (isosceles : a = 2 * b) :
  let s := (a + b + c) / 2
  let area := (s * (s - a) * (s - b) * (s - c)).sqrt
  area = 12 :=
by
  sorry

end triangle_area_l1068_106872


namespace integer_solutions_exist_l1068_106840

theorem integer_solutions_exist (m n : ℤ) :
  ∃ (w x y z : ℤ), 
  (w + x + 2 * y + 2 * z = m) ∧ 
  (2 * w - 2 * x + y - z = n) := sorry

end integer_solutions_exist_l1068_106840


namespace pool_min_cost_l1068_106820

noncomputable def CostMinimization (x : ℝ) : ℝ :=
  150 * 1600 + 720 * (x + 1600 / x)

theorem pool_min_cost :
  ∃ (x : ℝ), x = 40 ∧ CostMinimization x = 297600 :=
by
  sorry

end pool_min_cost_l1068_106820


namespace greater_number_l1068_106863

theorem greater_number (x y : ℕ) (h1 : x * y = 2048) (h2 : x + y - (x - y) = 64) : x = 64 :=
by
  sorry

end greater_number_l1068_106863


namespace trig_identity_l1068_106806

theorem trig_identity
  (x : ℝ)
  (h : Real.tan (π / 4 + x) = 2014) :
  1 / Real.cos (2 * x) + Real.tan (2 * x) = 2014 :=
by
  sorry

end trig_identity_l1068_106806


namespace max_ab_value_1_half_l1068_106817

theorem max_ab_value_1_half 
  (a b : ℝ) 
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_eq : a + 2 * b = 1) :
  a = 1 / 2 → ab = 1 / 8 :=
sorry

end max_ab_value_1_half_l1068_106817


namespace percent_change_is_minus_5_point_5_percent_l1068_106814

noncomputable def overall_percent_change (initial_value : ℝ) : ℝ :=
  let day1_value := initial_value * 0.75
  let day2_value := day1_value * 1.4
  let final_value := day2_value * 0.9
  ((final_value / initial_value) - 1) * 100

theorem percent_change_is_minus_5_point_5_percent :
  ∀ (initial_value : ℝ), overall_percent_change initial_value = -5.5 :=
sorry

end percent_change_is_minus_5_point_5_percent_l1068_106814


namespace regular_polygon_sides_l1068_106815

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l1068_106815


namespace find_A_l1068_106859

def hash_rel (A B : ℝ) := A^2 + B^2

theorem find_A (A : ℝ) (h : hash_rel A 7 = 196) : A = 7 * Real.sqrt 3 :=
by sorry

end find_A_l1068_106859


namespace derivative_of_ln_2x_l1068_106810

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x)

theorem derivative_of_ln_2x (x : ℝ) : deriv f x = 1 / x :=
  sorry

end derivative_of_ln_2x_l1068_106810


namespace no_solution_for_12k_plus_7_l1068_106844

theorem no_solution_for_12k_plus_7 (k : ℤ) :
  ∀ (a b c : ℕ), 12 * k + 7 ≠ 2^a + 3^b - 5^c := 
by sorry

end no_solution_for_12k_plus_7_l1068_106844


namespace neg_q_true_l1068_106871

theorem neg_q_true : (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end neg_q_true_l1068_106871


namespace sum_of_consecutive_pages_with_product_15300_l1068_106839

theorem sum_of_consecutive_pages_with_product_15300 : 
  ∃ n : ℕ, n * (n + 1) = 15300 ∧ n + (n + 1) = 247 :=
by
  sorry

end sum_of_consecutive_pages_with_product_15300_l1068_106839


namespace A_inter_complement_B_is_empty_l1068_106877

open Set Real

noncomputable def U : Set Real := univ

noncomputable def A : Set Real := { x : Real | ∃ (y : Real), y = sqrt (Real.log x) }

noncomputable def B : Set Real := { y : Real | ∃ (x : Real), y = sqrt x }

theorem A_inter_complement_B_is_empty :
  A ∩ (U \ B) = ∅ :=
by
    sorry

end A_inter_complement_B_is_empty_l1068_106877


namespace problem1_problem2_l1068_106883

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + a^2 / x
noncomputable def g (x : ℝ) : ℝ := x + Real.log x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x a + g x

-- Problem 1: Prove that a = sqrt(3) given that x = 1 is an extremum point for h(x, a)
theorem problem1 (a : ℝ) (h_extremum : ∀ x : ℝ, x = 1 → 0 = (2 - a^2 / x^2 + 1 / x)) : a = Real.sqrt 3 := sorry

-- Problem 2: Prove the range of a is [ (e + 1) / 2, +∞ ) such that for any x1, x2 ∈ [1, e], f(x1, a) ≥ g(x2)
theorem problem2 (a : ℝ) :
  (∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ Real.exp 1 ∧ 1 ≤ x2 ∧ x2 ≤ Real.exp 1 → f x1 a ≥ g x2) →
  (Real.exp 1 + 1) / 2 ≤ a :=
sorry

end problem1_problem2_l1068_106883


namespace six_people_acquaintance_or_strangers_l1068_106876

theorem six_people_acquaintance_or_strangers (p : Fin 6 → Prop) :
  ∃ (A B C : Fin 6), (p A ∧ p B ∧ p C) ∨ (¬p A ∧ ¬p B ∧ ¬p C) :=
sorry

end six_people_acquaintance_or_strangers_l1068_106876


namespace red_marbles_eq_14_l1068_106850

theorem red_marbles_eq_14 (total_marbles : ℕ) (yellow_marbles : ℕ) (R : ℕ) (B : ℕ)
  (h1 : total_marbles = 85)
  (h2 : yellow_marbles = 29)
  (h3 : B = 3 * R)
  (h4 : (total_marbles - yellow_marbles) = R + B) :
  R = 14 :=
by
  sorry

end red_marbles_eq_14_l1068_106850


namespace completing_the_square_step_l1068_106878

theorem completing_the_square_step (x : ℝ) : 
  x^2 + 4 * x + 2 = 0 → x^2 + 4 * x = -2 :=
by
  intro h
  sorry

end completing_the_square_step_l1068_106878


namespace quadratic_trinomial_unique_l1068_106829

theorem quadratic_trinomial_unique
  (a b c : ℝ)
  (h1 : b^2 - 4*(a+1)*c = 0)
  (h2 : (b+1)^2 - 4*a*c = 0)
  (h3 : b^2 - 4*a*(c+1) = 0) :
  a = 1/8 ∧ b = -3/4 ∧ c = 1/8 :=
by
  sorry

end quadratic_trinomial_unique_l1068_106829


namespace hector_gumballs_remaining_l1068_106860

def gumballs_remaining (gumballs : ℕ) (given_todd : ℕ) (given_alisha : ℕ) (given_bobby : ℕ) : ℕ :=
  gumballs - (given_todd + given_alisha + given_bobby)

theorem hector_gumballs_remaining :
  let gumballs := 45
  let given_todd := 4
  let given_alisha := 2 * given_todd
  let given_bobby := 4 * given_alisha - 5
  gumballs_remaining gumballs given_todd given_alisha given_bobby = 6 :=
by 
  let gumballs := 45
  let given_todd := 4
  let given_alisha := 2 * given_todd
  let given_bobby := 4 * given_alisha - 5
  show gumballs_remaining gumballs given_todd given_alisha given_bobby = 6
  sorry

end hector_gumballs_remaining_l1068_106860


namespace function_increasing_interval_l1068_106891

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem function_increasing_interval :
  ∀ x : ℝ, x > 0 → deriv f x > 0 := 
sorry

end function_increasing_interval_l1068_106891


namespace sum_of_two_numbers_l1068_106880

theorem sum_of_two_numbers (S : ℝ) (L : ℝ) (h1 : S = 3.5) (h2 : L = 3 * S) : S + L = 14 :=
by
  sorry

end sum_of_two_numbers_l1068_106880


namespace find_a_l1068_106825

noncomputable def geometric_sequence_solution (a : ℝ) : Prop :=
  (a + 1) ^ 2 = (1 / (a - 1)) * (a ^ 2 - 1)

theorem find_a (a : ℝ) : geometric_sequence_solution a → a = 0 :=
by
  intro h
  sorry

end find_a_l1068_106825


namespace discounted_price_correct_l1068_106830

def discounted_price (P : ℝ) : ℝ :=
  P * 0.80 * 0.90 * 0.95

theorem discounted_price_correct :
  discounted_price 9502.923976608186 = 6498.40 :=
by
  sorry

end discounted_price_correct_l1068_106830


namespace find_expression_max_value_min_value_l1068_106897

namespace MathProblem

-- Define the function f(x) with parameters a and b
def f (a b x : ℝ) : ℝ := a * x^2 + a^2 * x + 2 * b - a^3

-- Hypotheses based on problem conditions
lemma a_neg (a b : ℝ) : a < 0 := sorry
lemma root_neg2 (a b : ℝ) : f a b (-2) = 0 := sorry
lemma root_6 (a b : ℝ) : f a b 6 = 0 := sorry

-- Proving the explicit expression for f(x)
theorem find_expression (a b : ℝ) (x : ℝ) : 
  a = -4 → 
  b = -8 → 
  f a b x = -4 * x^2 + 16 * x + 48 :=
sorry

-- Maximum value of f(x) on the interval [1, 10]
theorem max_value (x : ℝ) (h : 1 ≤ x ∧ x ≤ 10) : 
  f (-4) (-8) 2 = 64 :=
sorry

-- Minimum value of f(x) on the interval [1, 10]
theorem min_value (x : ℝ) (h : 1 ≤ x ∧ x ≤ 10) : 
  f (-4) (-8) 10 = -192 :=
sorry

end MathProblem

end find_expression_max_value_min_value_l1068_106897


namespace colored_copies_count_l1068_106823

theorem colored_copies_count :
  ∃ C W : ℕ, (C + W = 400) ∧ (10 * C + 5 * W = 2250) ∧ (C = 50) :=
by
  sorry

end colored_copies_count_l1068_106823


namespace pies_sold_in_week_l1068_106882

def daily_pies := 8
def days_in_week := 7
def total_pies := 56

theorem pies_sold_in_week : daily_pies * days_in_week = total_pies :=
by
  sorry

end pies_sold_in_week_l1068_106882


namespace phone_price_is_correct_l1068_106856

-- Definition of the conditions
def monthly_cost := 7
def months := 4
def total_cost := 30

-- Definition to be proven
def phone_price := total_cost - (monthly_cost * months)

theorem phone_price_is_correct : phone_price = 2 :=
by
  sorry

end phone_price_is_correct_l1068_106856


namespace club_men_count_l1068_106841

theorem club_men_count (M W : ℕ) (h1 : M + W = 30) (h2 : M + (W / 3 : ℕ) = 20) : M = 15 := by
  -- proof omitted
  sorry

end club_men_count_l1068_106841


namespace find_b_value_l1068_106899

theorem find_b_value {b : ℚ} (h : -8 ^ 2 + b * -8 - 45 = 0) : b = 19 / 8 :=
sorry

end find_b_value_l1068_106899


namespace food_per_puppy_meal_l1068_106805

-- Definitions for conditions
def mom_daily_food : ℝ := 1.5 * 3
def num_puppies : ℕ := 5
def total_food_needed : ℝ := 57
def num_days : ℕ := 6

-- Total food for the mom dog over the given period
def total_mom_food : ℝ := mom_daily_food * num_days

-- Total food for the puppies over the given period
def total_puppy_food : ℝ := total_food_needed - total_mom_food

-- Total number of puppy meals over the given period
def total_puppy_meals : ℕ := (num_puppies * 2) * num_days

theorem food_per_puppy_meal :
  total_puppy_food / total_puppy_meals = 0.5 :=
  sorry

end food_per_puppy_meal_l1068_106805


namespace triangle_circle_property_l1068_106851

-- Let a, b, and c be the lengths of the sides of a right triangle, where c is the hypotenuse.
variables {a b c : ℝ}

-- Let varrho_b be the radius of the circle inscribed around the leg b of the triangle.
variable {varrho_b : ℝ}

-- Assume the relationship a^2 + b^2 = c^2 (Pythagorean theorem).
axiom right_triangle : a^2 + b^2 = c^2

-- Prove that b + c = a + 2 * varrho_b
theorem triangle_circle_property (h : a^2 + b^2 = c^2) (radius_condition : varrho_b = (a*b)/(a+c-b)) : 
  b + c = a + 2 * varrho_b :=
sorry

end triangle_circle_property_l1068_106851


namespace no_viable_schedule_l1068_106828

theorem no_viable_schedule :
  ∀ (studentsA studentsB : ℕ), 
    studentsA = 29 → 
    studentsB = 32 → 
    ¬ ∃ (a b : ℕ),
      (a = 29 ∧ b = 32 ∧
      (a * b = studentsA * studentsB) ∧
      (∀ (x : ℕ), x < studentsA * studentsB →
        ∃ (iA iB : ℕ), 
          iA < studentsA ∧ 
          iB < studentsB ∧ 
          -- The condition that each pair is unique within this period
          ((iA + iB) % (studentsA * studentsB) = x))) := by
  sorry

end no_viable_schedule_l1068_106828


namespace proportional_segments_l1068_106813

-- Define the tetrahedron and points
structure Tetrahedron :=
(A B C D O A1 B1 C1 : ℝ)

-- Define the conditions of the problem
variables {tetra : Tetrahedron}

-- Define the segments and their relationships
axiom segments_parallel (DA : ℝ) (DB : ℝ) (DC : ℝ)
  (OA1 : ℝ) (OB1 : ℝ) (OC1 : ℝ) :
  OA1 / DA + OB1 / DB + OC1 / DC = 1

-- The theorem to prove, which follows directly from the given axiom 
theorem proportional_segments (DA DB DC : ℝ)
  (OA1 : ℝ) (OB1 : ℝ) (OC1 : ℝ) :
  OA1 / DA + OB1 / DB + OC1 / DC = 1 :=
segments_parallel DA DB DC OA1 OB1 OC1

end proportional_segments_l1068_106813


namespace probabilities_inequalities_l1068_106811

variables (M N : Prop) (P : Prop → ℝ)

axiom P_pos_M : P M > 0
axiom P_pos_N : P N > 0
axiom P_cond_N_M : P (N ∧ M) / P M > P (N ∧ ¬M) / P (¬M)

theorem probabilities_inequalities :
    (P (N ∧ M) / P M > P (N ∧ ¬M) / P (¬M)) ∧
    (P (N ∧ M) > P N * P M) ∧
    (P (M ∧ N) / P N > P (M ∧ ¬N) / P (¬N)) :=
by
    sorry

end probabilities_inequalities_l1068_106811


namespace domain_of_h_l1068_106898

noncomputable def h (x : ℝ) : ℝ := (5 * x - 2) / (2 * x - 10)

theorem domain_of_h :
  {x : ℝ | 2 * x - 10 ≠ 0} = {x : ℝ | x ≠ 5} :=
by
  sorry

end domain_of_h_l1068_106898


namespace yogurt_amount_l1068_106893

-- Conditions
def total_ingredients : ℝ := 0.5
def strawberries : ℝ := 0.2
def orange_juice : ℝ := 0.2

-- Question and Answer (Proof Goal)
theorem yogurt_amount : total_ingredients - strawberries - orange_juice = 0.1 := by
  -- Since calculation involves specifics, we add sorry to indicate the proof is skipped
  sorry

end yogurt_amount_l1068_106893


namespace proposition_p_q_true_l1068_106822

def represents_hyperbola (m : ℝ) : Prop := (1 - m) * (m + 2) < 0

def represents_ellipse (m : ℝ) : Prop := (2 * m > 2 - m) ∧ (2 - m > 0)

theorem proposition_p_q_true (m : ℝ) :
  represents_hyperbola m ∧ represents_ellipse m → (1 < m ∧ m < 2) :=
by
  sorry

end proposition_p_q_true_l1068_106822


namespace arithmetic_seq_a7_l1068_106836

theorem arithmetic_seq_a7 (a : ℕ → ℕ) (h1 : a 2 = 2) (h2 : a 4 + a 5 = 12) : a 7 = 10 :=
by
  sorry

end arithmetic_seq_a7_l1068_106836


namespace ants_on_track_l1068_106885

/-- Given that ants move on a circular track of length 60 cm at a speed of 1 cm/s
and that there are 48 pairwise collisions in a minute, prove that the possible 
total number of ants on the track is 10, 11, 14, or 25. -/
theorem ants_on_track (x y : ℕ) (h : x * y = 24) : x + y = 10 ∨ x + y = 11 ∨ x + y = 14 ∨ x + y = 25 :=
by sorry

end ants_on_track_l1068_106885


namespace matching_pair_probability_correct_l1068_106819

-- Define the basic assumptions (conditions)
def black_pairs : Nat := 7
def brown_pairs : Nat := 4
def gray_pairs : Nat := 3
def red_pairs : Nat := 2

def total_pairs : Nat := black_pairs + brown_pairs + gray_pairs + red_pairs
def total_shoes : Nat := 2 * total_pairs

-- The probability calculation will be shown as the final proof requirement
def matching_color_probability : Rat :=  (14 * 7 + 8 * 4 + 6 * 3 + 4 * 2 : Int) / (32 * 31 : Int)

-- The target statement to be proven
theorem matching_pair_probability_correct :
  matching_color_probability = (39 / 248 : Rat) :=
by
  sorry

end matching_pair_probability_correct_l1068_106819


namespace solution_set_abs_inequality_l1068_106801

theorem solution_set_abs_inequality (x : ℝ) : (|x - 1| ≤ 2) ↔ (-1 ≤ x ∧ x ≤ 3) :=
by
  sorry

end solution_set_abs_inequality_l1068_106801


namespace no_such_number_exists_l1068_106867

-- Definitions for conditions
def base_5_digit_number (x : ℕ) : Prop := 
  ∀ n, 0 ≤ n ∧ n < 2023 → x / 5^n % 5 < 5

def odd_plus_one (n m : ℕ) : Prop :=
  (∀ k < 1012, (n / 5^(2*k) % 25 / 5 = m / 5^(2*k) % 25 / 5 + 1)) ∧
  (∀ k < 1011, (n / 5^(2*k+1) % 25 / 5 = m / 5^(2*k+1) % 25 / 5 - 1))

def has_two_prime_factors_that_differ_by_two (x : ℕ) : Prop :=
  ∃ u v, u * v = x ∧ Prime u ∧ Prime v ∧ v = u + 2

-- Combined conditions for the hypothesized number x
def hypothesized_number (x : ℕ) : Prop := 
  base_5_digit_number x ∧
  odd_plus_one x x ∧
  has_two_prime_factors_that_differ_by_two x

-- The proof statement that the hypothesized number cannot exist
theorem no_such_number_exists : ¬ ∃ x, hypothesized_number x :=
by
  sorry

end no_such_number_exists_l1068_106867


namespace determine_m_l1068_106800

noncomputable def f (m x : ℝ) := (m^2 - m - 1) * x^(-5 * m - 3)

theorem determine_m : ∃ m : ℝ, (∀ x > 0, f m x = (m^2 - m - 1) * x^(-5 * m - 3)) ∧ (∀ x > 0, (m^2 - m - 1) * x^(-(5 * m + 3)) = (m^2 - m - 1) * x^(-5 * m - 3) → -5 * m - 3 > 0) ∧ m = -1 :=
by
  sorry

end determine_m_l1068_106800


namespace problem1_problem2_l1068_106804

-- Problem 1: Prove that the solution to f(x) <= 0 for a = -2 is [1, +∞)
theorem problem1 (x : ℝ) : (|x + 2| - 2 * x - 1 ≤ 0) ↔ (1 ≤ x) := sorry

-- Problem 2: Prove that the range of m such that there exists x ∈ ℝ satisfying f(x) + |x + 2| ≤ m for a = 1 is m ≥ 0
theorem problem2 (m : ℝ) : (∃ x : ℝ, |x - 1| - 2 * x - 1 + |x + 2| ≤ m) ↔ (0 ≤ m) := sorry

end problem1_problem2_l1068_106804


namespace find_M_value_when_x_3_l1068_106861

-- Definitions based on the given conditions
def polynomial (a b c d x : ℝ) : ℝ := a*x^5 + b*x^3 + c*x + d

-- Given conditions
variables (a b c d : ℝ)
axiom h₀ : polynomial a b c d 0 = -5
axiom h₁ : polynomial a b c d (-3) = 7

-- Desired statement: Prove that the value of polynomial at x = 3 is -17
theorem find_M_value_when_x_3 : polynomial a b c d 3 = -17 :=
by sorry

end find_M_value_when_x_3_l1068_106861


namespace book_costs_l1068_106848

theorem book_costs (C1 C2 : ℝ) (h1 : C1 + C2 = 450) (h2 : 0.85 * C1 = 1.19 * C2) : C1 = 262.5 := 
sorry

end book_costs_l1068_106848


namespace train_length_l1068_106816

theorem train_length (speed_kmph : ℝ) (time_seconds : ℝ) (speed_mps : ℝ) (length_train : ℝ) : 
  speed_kmph = 90 → 
  time_seconds = 6 → 
  speed_mps = (speed_kmph * 1000 / 3600) →
  length_train = (speed_mps * time_seconds) → 
  length_train = 150 :=
by
  intros h_speed h_time h_speed_mps h_length
  sorry

end train_length_l1068_106816


namespace fred_initial_balloons_l1068_106845

def green_balloons_initial (given: Nat) (left: Nat) : Nat := 
  given + left

theorem fred_initial_balloons : green_balloons_initial 221 488 = 709 :=
by
  sorry

end fred_initial_balloons_l1068_106845


namespace judys_school_week_l1068_106835

theorem judys_school_week
  (pencils_used : ℕ)
  (packs_cost : ℕ)
  (total_cost : ℕ)
  (days_period : ℕ)
  (pencils_per_pack : ℕ)
  (pencils_in_school_days : ℕ)
  (total_pencil_use : ℕ) :
  (total_cost / packs_cost * pencils_per_pack = total_pencil_use) →
  (total_pencil_use / days_period = pencils_used) →
  (pencils_in_school_days / pencils_used = 5) :=
sorry

end judys_school_week_l1068_106835


namespace baseball_team_games_l1068_106824

theorem baseball_team_games (P Q : ℕ) (hP : P > 3 * Q) (hQ : Q > 3) (hTotal : 2 * P + 6 * Q = 78) :
  2 * P = 54 :=
by
  -- placeholder for the actual proof
  sorry

end baseball_team_games_l1068_106824


namespace fruit_problem_l1068_106833

theorem fruit_problem
  (A B C : ℕ)
  (hA : A = 4) 
  (hB : B = 6) 
  (hC : C = 12) :
  ∃ x : ℕ, 1 = x / 2 := 
by
  sorry

end fruit_problem_l1068_106833


namespace circle_radius_l1068_106818

theorem circle_radius (r : ℝ) (π : ℝ) (h1 : π > 0) (h2 : ∀ x, π * x^2 = 100*π → x = 10) : r = 10 :=
by
  have : π * r^2 = 100*π → r = 10 := h2 r
  exact sorry

end circle_radius_l1068_106818


namespace find_f_four_thirds_l1068_106892

def f (y: ℝ) : ℝ := sorry  -- Placeholder for the function definition

theorem find_f_four_thirds : f (4 / 3) = - (7 / 2) := sorry

end find_f_four_thirds_l1068_106892


namespace max_blocks_that_fit_l1068_106858

noncomputable def box_volume : ℕ :=
  3 * 4 * 2

noncomputable def block_volume : ℕ :=
  2 * 1 * 2

noncomputable def max_blocks (box_volume : ℕ) (block_volume : ℕ) : ℕ :=
  box_volume / block_volume

theorem max_blocks_that_fit : max_blocks box_volume block_volume = 6 :=
by
  sorry

end max_blocks_that_fit_l1068_106858


namespace insert_zeros_between_digits_is_cube_l1068_106862

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n

theorem insert_zeros_between_digits_is_cube (k b : ℕ) (h_b : b ≥ 4) 
  : is_perfect_cube (1 * b^(3*(1+k)) + 3 * b^(2*(1+k)) + 3 * b^(1+k) + 1) :=
sorry

end insert_zeros_between_digits_is_cube_l1068_106862


namespace abigail_money_left_l1068_106843

def initial_amount : ℕ := 11
def spent_in_store : ℕ := 2
def amount_lost : ℕ := 6

theorem abigail_money_left :
  initial_amount - spent_in_store - amount_lost = 3 := 
by {
  sorry
}

end abigail_money_left_l1068_106843


namespace analytical_expression_range_of_t_l1068_106855

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem analytical_expression (x : ℝ) :
  (f (x + 1) - f x = 2 * x - 2) ∧ (f 1 = -2) :=
by
  sorry

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x > 0 ∧ f (x + t) < 0 → x = 1) ↔ (-2 <= t ∧ t < -1) :=
by
  sorry

end analytical_expression_range_of_t_l1068_106855


namespace subset_relation_l1068_106873

variables (M N : Set ℕ) 

theorem subset_relation (hM : M = {1, 2, 3, 4}) (hN : N = {2, 3, 4}) : N ⊆ M :=
sorry

end subset_relation_l1068_106873


namespace compute_sum_l1068_106821
-- Import the necessary library to have access to the required definitions and theorems.

-- Define the integers involved based on the conditions.
def a : ℕ := 157
def b : ℕ := 43
def c : ℕ := 19
def d : ℕ := 81

-- State the theorem that computes the sum of these integers and equate it to 300.
theorem compute_sum : a + b + c + d = 300 := by
  sorry

end compute_sum_l1068_106821


namespace parabola_equation_l1068_106866

theorem parabola_equation (P : ℝ × ℝ) (hP : P = (-4, -2)) :
  (∃ p : ℝ, P.1^2 = -2 * p * P.2 ∧ p = -4 ∧ x^2 = -8*y) ∨ 
  (∃ p : ℝ, P.2^2 = -2 * p * P.1 ∧ p = -1/2 ∧ y^2 = -x) :=
by
  sorry

end parabola_equation_l1068_106866


namespace sum_of_interior_numbers_eighth_row_l1068_106849

def sum_of_interior_numbers (n : ℕ) : ℕ :=
  2^(n-1) - 2

theorem sum_of_interior_numbers_eighth_row : sum_of_interior_numbers 8 = 126 :=
by
  sorry

end sum_of_interior_numbers_eighth_row_l1068_106849


namespace range_of_m_l1068_106808

noncomputable def equation_has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, 25^(-|x+1|) - 4 * 5^(-|x+1|) - m = 0

theorem range_of_m : ∀ m : ℝ, equation_has_real_roots m ↔ (-3 ≤ m ∧ m < 0) :=
by
  -- Proof omitted
  sorry

end range_of_m_l1068_106808


namespace student_test_ratio_l1068_106846

theorem student_test_ratio :
  ∀ (total_questions correct_responses : ℕ),
  total_questions = 100 →
  correct_responses = 93 →
  (total_questions - correct_responses) / correct_responses = 7 / 93 :=
by
  intros total_questions correct_responses h_total_questions h_correct_responses
  sorry

end student_test_ratio_l1068_106846


namespace simplify_expression_l1068_106889

variable (a b : ℤ)

theorem simplify_expression : 
  (32 * a + 45 * b) + (15 * a + 36 * b) - (27 * a + 41 * b) = 20 * a + 40 * b := 
by sorry

end simplify_expression_l1068_106889


namespace no_adjacent_stand_up_probability_l1068_106884

noncomputable def coin_flip_prob_adjacent_people_stand_up : ℚ :=
  123 / 1024

theorem no_adjacent_stand_up_probability :
  let num_people := 10
  let total_outcomes := 2^num_people
  (123 : ℚ) / total_outcomes = coin_flip_prob_adjacent_people_stand_up :=
by
  sorry

end no_adjacent_stand_up_probability_l1068_106884


namespace remaining_water_l1068_106879

theorem remaining_water (initial_water : ℚ) (used_water : ℚ) (remaining_water : ℚ) 
  (h1 : initial_water = 3) (h2 : used_water = 5/4) : remaining_water = 7/4 :=
by
  -- The proof would go here, but we are skipping it as per the instructions.
  sorry

end remaining_water_l1068_106879


namespace fewest_number_of_students_l1068_106807

theorem fewest_number_of_students :
  ∃ n : ℕ, n ≡ 3 [MOD 6] ∧ n ≡ 5 [MOD 8] ∧ n ≡ 7 [MOD 9] ∧ ∀ m : ℕ, (m ≡ 3 [MOD 6] ∧ m ≡ 5 [MOD 8] ∧ m ≡ 7 [MOD 9]) → m ≥ n := by
  sorry

end fewest_number_of_students_l1068_106807
