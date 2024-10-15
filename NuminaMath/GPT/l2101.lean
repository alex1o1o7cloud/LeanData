import Mathlib

namespace NUMINAMATH_GPT_sum_difference_even_odd_l2101_210160

theorem sum_difference_even_odd :
  let x := (100 / 2) * (2 + 200)
  let y := (100 / 2) * (1 + 199)
  x - y = 100 :=
by
  sorry

end NUMINAMATH_GPT_sum_difference_even_odd_l2101_210160


namespace NUMINAMATH_GPT_Sara_pears_left_l2101_210106

def Sara_has_left (initial_pears : ℕ) (given_to_Dan : ℕ) (given_to_Monica : ℕ) (given_to_Jenny : ℕ) : ℕ :=
  initial_pears - given_to_Dan - given_to_Monica - given_to_Jenny

theorem Sara_pears_left :
  Sara_has_left 35 28 4 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_Sara_pears_left_l2101_210106


namespace NUMINAMATH_GPT_hazel_drank_one_cup_l2101_210139

theorem hazel_drank_one_cup (total_cups made_to_crew bike_sold friends_given remaining_cups : ℕ) 
  (H1 : total_cups = 56)
  (H2 : made_to_crew = total_cups / 2)
  (H3 : bike_sold = 18)
  (H4 : friends_given = bike_sold / 2)
  (H5 : remaining_cups = total_cups - (made_to_crew + bike_sold + friends_given)) :
  remaining_cups = 1 := 
sorry

end NUMINAMATH_GPT_hazel_drank_one_cup_l2101_210139


namespace NUMINAMATH_GPT_difference_not_divisible_by_1976_l2101_210135

theorem difference_not_divisible_by_1976 (A B : ℕ) (hA : 100 ≤ A) (hA' : A < 1000) (hB : 100 ≤ B) (hB' : B < 1000) (h : A ≠ B) :
  ¬ (1976 ∣ (1000 * A + B - (1000 * B + A))) :=
by
  sorry

end NUMINAMATH_GPT_difference_not_divisible_by_1976_l2101_210135


namespace NUMINAMATH_GPT_equilibrium_shift_if_K_changes_l2101_210149

-- Define the equilibrium constant and its relation to temperature
def equilibrium_constant (T : ℝ) : ℝ := sorry

-- Define the conditions
axiom K_related_to_temp (T₁ T₂ : ℝ) (K₁ K₂ : ℝ) :
  equilibrium_constant T₁ = K₁ ∧ equilibrium_constant T₂ = K₂ → T₁ = T₂ ↔ K₁ = K₂

axiom K_constant_with_concentration_change (T : ℝ) (K : ℝ) (c₁ c₂ : ℝ) :
  equilibrium_constant T = K → equilibrium_constant T = K

axiom K_squared_with_stoichiometric_double (T : ℝ) (K : ℝ) :
  equilibrium_constant (2 * T) = K * K

-- Define the problem to be proved
theorem equilibrium_shift_if_K_changes (T₁ T₂ : ℝ) (K₁ K₂ : ℝ) :
  equilibrium_constant T₁ = K₁ ∧ equilibrium_constant T₂ = K₂ → K₁ ≠ K₂ → T₁ ≠ T₂ := 
sorry

end NUMINAMATH_GPT_equilibrium_shift_if_K_changes_l2101_210149


namespace NUMINAMATH_GPT_negation_of_existence_l2101_210154

theorem negation_of_existence : 
  (¬ ∃ x_0 : ℝ, (x_0 + 1 < 0) ∨ (x_0^2 - x_0 > 0)) ↔ ∀ x : ℝ, (x + 1 ≥ 0) ∧ (x^2 - x ≤ 0) := 
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_l2101_210154


namespace NUMINAMATH_GPT_five_point_eight_one_million_in_scientific_notation_l2101_210138

theorem five_point_eight_one_million_in_scientific_notation :
  5.81 * 10^6 = 5.81e6 :=
sorry

end NUMINAMATH_GPT_five_point_eight_one_million_in_scientific_notation_l2101_210138


namespace NUMINAMATH_GPT_lewis_weekly_earning_l2101_210140

theorem lewis_weekly_earning
  (weeks : ℕ)
  (weekly_rent : ℤ)
  (total_savings : ℤ)
  (h1 : weeks = 1181)
  (h2 : weekly_rent = 216)
  (h3 : total_savings = 324775)
  : ∃ (E : ℤ), E = 49075 / 100 :=
by
  let E := 49075 / 100
  use E
  sorry -- The proof would go here

end NUMINAMATH_GPT_lewis_weekly_earning_l2101_210140


namespace NUMINAMATH_GPT_number_of_4_digit_numbers_divisible_by_9_l2101_210175

theorem number_of_4_digit_numbers_divisible_by_9 :
  ∃ n : ℕ, (∀ k : ℕ, k ∈ Finset.range n → 1008 + k * 9 ≤ 9999) ∧
           (1008 + (n - 1) * 9 = 9999) ∧
           n = 1000 :=
by
  sorry

end NUMINAMATH_GPT_number_of_4_digit_numbers_divisible_by_9_l2101_210175


namespace NUMINAMATH_GPT_strip_width_l2101_210180

theorem strip_width (w : ℝ) (h_floor : ℝ := 10) (b_floor : ℝ := 8) (area_rug : ℝ := 24) :
  (h_floor - 2 * w) * (b_floor - 2 * w) = area_rug → w = 2 := 
by 
  sorry

end NUMINAMATH_GPT_strip_width_l2101_210180


namespace NUMINAMATH_GPT_metallic_sheet_width_l2101_210131

-- Defining the conditions
def sheet_length := 48
def cut_square_side := 8
def box_volume := 5632

-- Main theorem statement
theorem metallic_sheet_width 
    (L : ℕ := sheet_length)
    (s : ℕ := cut_square_side)
    (V : ℕ := box_volume) :
    (32 * (w - 2 * s) * s = V) → (w = 38) := by
  intros h1
  sorry

end NUMINAMATH_GPT_metallic_sheet_width_l2101_210131


namespace NUMINAMATH_GPT_prime_1021_n_unique_l2101_210136

theorem prime_1021_n_unique :
  ∃! (n : ℕ), n ≥ 2 ∧ Prime (n^3 + 2 * n + 1) :=
sorry

end NUMINAMATH_GPT_prime_1021_n_unique_l2101_210136


namespace NUMINAMATH_GPT_alex_jellybeans_l2101_210133

theorem alex_jellybeans (n : ℕ) (h1 : n ≥ 200) (h2 : n % 17 = 15) : n = 202 :=
sorry

end NUMINAMATH_GPT_alex_jellybeans_l2101_210133


namespace NUMINAMATH_GPT_hours_spent_writing_l2101_210113

-- Define the rates at which Jacob and Nathan write
def Nathan_rate : ℕ := 25        -- Nathan writes 25 letters per hour
def Jacob_rate : ℕ := 2 * Nathan_rate  -- Jacob writes twice as fast as Nathan

-- Define the combined rate
def combined_rate : ℕ := Nathan_rate + Jacob_rate

-- Define the total letters written and the hours spent
def total_letters : ℕ := 750
def hours_spent : ℕ := total_letters / combined_rate

-- The theorem to prove
theorem hours_spent_writing : hours_spent = 10 :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_hours_spent_writing_l2101_210113


namespace NUMINAMATH_GPT_tangent_line_parallel_curve_l2101_210163

def curve (x : ℝ) : ℝ := x^4

def line_parallel_to_curve (l : ℝ → ℝ → Prop) : Prop :=
  ∃ x0 y0 : ℝ, l x0 y0 ∧ curve x0 = y0 ∧ ∀ (x : ℝ), l x (curve x)

theorem tangent_line_parallel_curve :
  ∃ (l : ℝ → ℝ → Prop), line_parallel_to_curve l ∧ ∀ x y, l x y ↔ 8 * x + 16 * y + 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_parallel_curve_l2101_210163


namespace NUMINAMATH_GPT_probability_of_sequence_123456_l2101_210171

theorem probability_of_sequence_123456 :
  let total_sequences := 66 * 45 * 28 * 15 * 6 * 1     -- Total number of sequences
  let specific_sequences := 1 * 3 * 5 * 7 * 9 * 11        -- Sequences leading to 123456
  specific_sequences / total_sequences = 1 / 720 := by
  let total_sequences := 74919600
  let specific_sequences := 10395
  sorry

end NUMINAMATH_GPT_probability_of_sequence_123456_l2101_210171


namespace NUMINAMATH_GPT_find_x_l2101_210151

theorem find_x :
  (x : ℝ) →
  (0.40 * 2 = 0.25 * (0.30 * 15 + x)) →
  x = -1.3 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_find_x_l2101_210151


namespace NUMINAMATH_GPT_biscuits_per_guest_correct_l2101_210173

def flour_per_batch : ℚ := 5 / 4
def biscuits_per_batch : ℕ := 9
def flour_needed : ℚ := 5
def guests : ℕ := 18

theorem biscuits_per_guest_correct :
  (flour_needed * biscuits_per_batch / flour_per_batch) / guests = 2 := by
  sorry

end NUMINAMATH_GPT_biscuits_per_guest_correct_l2101_210173


namespace NUMINAMATH_GPT_back_wheel_revolutions_l2101_210147

-- Defining relevant distances and conditions
def front_wheel_radius : ℝ := 3 -- radius in feet
def back_wheel_radius : ℝ := 0.5 -- radius in feet
def front_wheel_revolutions : ℕ := 120

-- The target theorem
theorem back_wheel_revolutions :
  let front_wheel_circumference := 2 * Real.pi * front_wheel_radius
  let total_distance := front_wheel_circumference * (front_wheel_revolutions : ℝ)
  let back_wheel_circumference := 2 * Real.pi * back_wheel_radius
  let back_wheel_revs := total_distance / back_wheel_circumference
  back_wheel_revs = 720 :=
by
  sorry

end NUMINAMATH_GPT_back_wheel_revolutions_l2101_210147


namespace NUMINAMATH_GPT_logan_gas_expense_l2101_210110

-- Definitions based on conditions:
def annual_salary := 65000
def rent_expense := 20000
def grocery_expense := 5000
def desired_savings := 42000
def new_income_target := annual_salary + 10000

-- The property to be proved:
theorem logan_gas_expense : 
  ∀ (gas_expense : ℕ), 
  new_income_target - desired_savings = rent_expense + grocery_expense + gas_expense → 
  gas_expense = 8000 := 
by 
  sorry

end NUMINAMATH_GPT_logan_gas_expense_l2101_210110


namespace NUMINAMATH_GPT_triangle_angle_sum_l2101_210176

theorem triangle_angle_sum (CD CB : ℝ) 
    (isosceles_triangle: CD = CB)
    (interior_pentagon_angle: 108 = 180 * (5 - 2) / 5)
    (interior_triangle_angle: 60 = 180 / 3)
    (triangle_angle_sum: ∀ (a b c : ℝ), a + b + c = 180) :
    mangle_CDB = 6 :=
by
  have x : ℝ := 6
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_l2101_210176


namespace NUMINAMATH_GPT_range_of_t_l2101_210142

theorem range_of_t (t : ℝ) (x : ℝ) : (1 < x ∧ x ≤ 4) → (|x - t| < 1 ↔ 2 ≤ t ∧ t ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_t_l2101_210142


namespace NUMINAMATH_GPT_mass_percentage_of_O_in_dichromate_l2101_210166

noncomputable def molar_mass_Cr : ℝ := 52.00
noncomputable def molar_mass_O : ℝ := 16.00
noncomputable def molar_mass_Cr2O7_2_minus : ℝ := (2 * molar_mass_Cr) + (7 * molar_mass_O)

theorem mass_percentage_of_O_in_dichromate :
  (7 * molar_mass_O / molar_mass_Cr2O7_2_minus) * 100 = 51.85 := 
by
  sorry

end NUMINAMATH_GPT_mass_percentage_of_O_in_dichromate_l2101_210166


namespace NUMINAMATH_GPT_volume_of_one_gram_l2101_210194

theorem volume_of_one_gram (mass_per_cubic_meter : ℕ)
  (kilo_to_grams : ℕ)
  (cubic_meter_to_cubic_centimeters : ℕ)
  (substance_mass : mass_per_cubic_meter = 300)
  (kilo_conv : kilo_to_grams = 1000)
  (cubic_conv : cubic_meter_to_cubic_centimeters = 1000000)
  :
  ∃ v : ℝ, v = cubic_meter_to_cubic_centimeters / (mass_per_cubic_meter * kilo_to_grams) ∧ v = 10 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_volume_of_one_gram_l2101_210194


namespace NUMINAMATH_GPT_die_face_never_touches_board_l2101_210132

theorem die_face_never_touches_board : 
  ∃ (cube : Type) (roll : cube → cube) (occupied : Fin 8 × Fin 8 → cube → Prop),
    (∀ p : Fin 8 × Fin 8, ∃ c : cube, occupied p c) ∧ 
    (∃ f : cube, ¬ (∃ p : Fin 8 × Fin 8, occupied p f)) :=
by sorry

end NUMINAMATH_GPT_die_face_never_touches_board_l2101_210132


namespace NUMINAMATH_GPT_acetic_acid_molecular_weight_is_correct_l2101_210116

def molecular_weight_acetic_acid : ℝ :=
  let carbon_weight := 12.01
  let hydrogen_weight := 1.008
  let oxygen_weight := 16.00
  let num_carbons := 2
  let num_hydrogens := 4
  let num_oxygens := 2
  num_carbons * carbon_weight + num_hydrogens * hydrogen_weight + num_oxygens * oxygen_weight

theorem acetic_acid_molecular_weight_is_correct : molecular_weight_acetic_acid = 60.052 :=
by 
  unfold molecular_weight_acetic_acid
  sorry

end NUMINAMATH_GPT_acetic_acid_molecular_weight_is_correct_l2101_210116


namespace NUMINAMATH_GPT_sum_of_two_numbers_l2101_210169

theorem sum_of_two_numbers (a b : ℝ) (h1 : a * b = 16) (h2 : (1 / a) = 3 * (1 / b)) (ha : 0 < a) (hb : 0 < b) :
  a + b = 16 * Real.sqrt 3 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_two_numbers_l2101_210169


namespace NUMINAMATH_GPT_angleina_speed_from_grocery_to_gym_l2101_210118

variable (v : ℝ) (h1 : 720 / v - 40 = 240 / v)

theorem angleina_speed_from_grocery_to_gym : 2 * v = 24 :=
by
  sorry

end NUMINAMATH_GPT_angleina_speed_from_grocery_to_gym_l2101_210118


namespace NUMINAMATH_GPT_days_y_worked_l2101_210172

theorem days_y_worked 
  (W : ℝ) 
  (x_days : ℝ) (h1 : x_days = 36)
  (y_days : ℝ) (h2 : y_days = 24)
  (x_remaining_days : ℝ) (h3 : x_remaining_days = 18)
  (d : ℝ) :
  d * (W / y_days) + x_remaining_days * (W / x_days) = W → d = 12 :=
by
  -- Mathematical proof goes here
  sorry

end NUMINAMATH_GPT_days_y_worked_l2101_210172


namespace NUMINAMATH_GPT_third_derivative_y_l2101_210161

noncomputable def y (x : ℝ) : ℝ := x * Real.cos (x^2)

theorem third_derivative_y (x : ℝ) :
  (deriv^[3] y) x = (8 * x^4 - 6) * Real.sin (x^2) - 24 * x^2 * Real.cos (x^2) :=
by
  sorry

end NUMINAMATH_GPT_third_derivative_y_l2101_210161


namespace NUMINAMATH_GPT_exists_n_for_pow_lt_e_l2101_210111

theorem exists_n_for_pow_lt_e {p e : ℝ} (hp : 0 < p ∧ p < 1) (he : 0 < e) :
  ∃ n : ℕ, (1 - p) ^ n < e :=
sorry

end NUMINAMATH_GPT_exists_n_for_pow_lt_e_l2101_210111


namespace NUMINAMATH_GPT_find_derivative_l2101_210150

theorem find_derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 1 + Real.log x) : f' 1 = -1 := 
by
  sorry

end NUMINAMATH_GPT_find_derivative_l2101_210150


namespace NUMINAMATH_GPT_pencils_count_l2101_210157

theorem pencils_count (P L : ℕ) 
  (h1 : P * 6 = L * 5) 
  (h2 : L = P + 7) : 
  L = 42 :=
by
  sorry

end NUMINAMATH_GPT_pencils_count_l2101_210157


namespace NUMINAMATH_GPT_income_expenditure_ratio_l2101_210195

theorem income_expenditure_ratio (I E S : ℝ) (hI : I = 10000) (hS : S = 2000) (hEq : S = I - E) : I / E = 5 / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_income_expenditure_ratio_l2101_210195


namespace NUMINAMATH_GPT_lines_perpendicular_l2101_210188

theorem lines_perpendicular (A1 B1 C1 A2 B2 C2 : ℝ) (h : A1 * A2 + B1 * B2 = 0) :
  ∃(x y : ℝ), A1 * x + B1 * y + C1 = 0 ∧ A2 * x + B2 * y + C2 = 0 → A1 * A2 + B1 * B2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_lines_perpendicular_l2101_210188


namespace NUMINAMATH_GPT_right_triangle_third_side_l2101_210177

theorem right_triangle_third_side (a b : ℝ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :
  c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2) :=
by 
  sorry

end NUMINAMATH_GPT_right_triangle_third_side_l2101_210177


namespace NUMINAMATH_GPT_angles_at_point_l2101_210112

theorem angles_at_point (x y : ℝ) 
  (h1 : x + y + 120 = 360) 
  (h2 : x = 2 * y) : 
  x = 160 ∧ y = 80 :=
by
  sorry

end NUMINAMATH_GPT_angles_at_point_l2101_210112


namespace NUMINAMATH_GPT_road_length_l2101_210125

theorem road_length (n : ℕ) (d : ℕ) (trees : ℕ) (intervals : ℕ) (L : ℕ) 
  (h1 : n = 10) 
  (h2 : d = 10) 
  (h3 : trees = 10) 
  (h4 : intervals = trees - 1) 
  (h5 : L = intervals * d) : 
  L = 90 :=
by
  sorry

end NUMINAMATH_GPT_road_length_l2101_210125


namespace NUMINAMATH_GPT_total_balls_in_box_l2101_210123

theorem total_balls_in_box (red blue yellow total : ℕ) 
  (h1 : 2 * blue = 3 * red)
  (h2 : 3 * yellow = 4 * red) 
  (h3 : yellow = 40)
  (h4 : red + blue + yellow = total) : total = 90 :=
sorry

end NUMINAMATH_GPT_total_balls_in_box_l2101_210123


namespace NUMINAMATH_GPT_student_solves_exactly_20_problems_l2101_210167

theorem student_solves_exactly_20_problems :
  (∀ n, 1 ≤ (a : ℕ → ℕ) n) ∧ (∀ k, a (k + 7) ≤ a k + 12) ∧ a 77 ≤ 132 →
  ∃ i j, i < j ∧ a j - a i = 20 := sorry

end NUMINAMATH_GPT_student_solves_exactly_20_problems_l2101_210167


namespace NUMINAMATH_GPT_triangle_acute_angle_sufficient_condition_triangle_acute_angle_not_necessary_condition_l2101_210183

theorem triangle_acute_angle_sufficient_condition
  (a b c : ℝ) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  a ≤ (b + c) / 2 → b^2 + c^2 > a^2 :=
sorry

theorem triangle_acute_angle_not_necessary_condition
  (a b c : ℝ) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  b^2 + c^2 > a^2 → ¬ (a ≤ (b + c) / 2) :=
sorry

end NUMINAMATH_GPT_triangle_acute_angle_sufficient_condition_triangle_acute_angle_not_necessary_condition_l2101_210183


namespace NUMINAMATH_GPT_value_of_trig_expr_l2101_210168

theorem value_of_trig_expr : 2 * Real.cos (Real.pi / 12) ^ 2 + 1 = 2 + Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_trig_expr_l2101_210168


namespace NUMINAMATH_GPT_ratio_of_speeds_is_two_l2101_210159

noncomputable def joe_speed : ℝ := 0.266666666667
noncomputable def time : ℝ := 40
noncomputable def total_distance : ℝ := 16

noncomputable def joe_distance : ℝ := joe_speed * time
noncomputable def pete_distance : ℝ := total_distance - joe_distance
noncomputable def pete_speed : ℝ := pete_distance / time

theorem ratio_of_speeds_is_two :
  joe_speed / pete_speed = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_is_two_l2101_210159


namespace NUMINAMATH_GPT_fraction_of_40_l2101_210182

theorem fraction_of_40 : (3 / 4) * 40 = 30 :=
by
  -- We'll add the 'sorry' here to indicate that this is the proof part which is not required.
  sorry

end NUMINAMATH_GPT_fraction_of_40_l2101_210182


namespace NUMINAMATH_GPT_cats_combined_weight_l2101_210134

theorem cats_combined_weight :
  let cat1 := 2
  let cat2 := 7
  let cat3 := 4
  cat1 + cat2 + cat3 = 13 := 
by
  let cat1 := 2
  let cat2 := 7
  let cat3 := 4
  sorry

end NUMINAMATH_GPT_cats_combined_weight_l2101_210134


namespace NUMINAMATH_GPT_complement_union_l2101_210121

theorem complement_union (U M N : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5, 6})
  (hM : M = {2, 3, 5})
  (hN : N = {4, 5}) :
  U \ (M ∪ N) = {1, 6} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l2101_210121


namespace NUMINAMATH_GPT_smaller_of_two_numbers_l2101_210162

theorem smaller_of_two_numbers 
  (a b d : ℝ) (h : 0 < a ∧ a < b) (u v : ℝ) 
  (huv : u / v = b / a) (sum_uv : u + v = d) : 
  min u v = (a * d) / (a + b) :=
by
  sorry

end NUMINAMATH_GPT_smaller_of_two_numbers_l2101_210162


namespace NUMINAMATH_GPT_num_supermarkets_us_l2101_210153

noncomputable def num_supermarkets_total : ℕ := 84

noncomputable def us_canada_relationship (C : ℕ) : Prop := C + (C + 10) = num_supermarkets_total

theorem num_supermarkets_us (C : ℕ) (h : us_canada_relationship C) : C + 10 = 47 :=
sorry

end NUMINAMATH_GPT_num_supermarkets_us_l2101_210153


namespace NUMINAMATH_GPT_trains_crossing_time_l2101_210104

theorem trains_crossing_time
  (L : ℕ) (t1 t2 : ℕ)
  (h_length : L = 120)
  (h_t1 : t1 = 10)
  (h_t2 : t2 = 15) :
  let V1 := L / t1
  let V2 := L / t2
  let V_relative := V1 + V2
  let D := L + L
  (D / V_relative) = 12 :=
by
  sorry

end NUMINAMATH_GPT_trains_crossing_time_l2101_210104


namespace NUMINAMATH_GPT_santana_brothers_birthday_l2101_210198

theorem santana_brothers_birthday (b : ℕ) (oct : ℕ) (nov : ℕ) (dec : ℕ) (c_presents_diff : ℕ) :
  b = 7 → oct = 1 → nov = 1 → dec = 2 → c_presents_diff = 8 → (∃ M : ℕ, M = 3) :=
by
  sorry

end NUMINAMATH_GPT_santana_brothers_birthday_l2101_210198


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l2101_210137

theorem solution_set_of_quadratic_inequality (x : ℝ) : 
  x^2 + 3*x - 4 < 0 ↔ -4 < x ∧ x < 1 :=
sorry

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l2101_210137


namespace NUMINAMATH_GPT_max_value_of_f_l2101_210114

open Real

noncomputable def f (x : ℝ) : ℝ := -x - 9 / x + 18

theorem max_value_of_f : ∀ x > 0, f x ≤ 12 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l2101_210114


namespace NUMINAMATH_GPT_trains_meet_at_distance_360_km_l2101_210100

-- Define the speeds of the trains
def speed_A : ℕ := 30 -- speed of train A in kmph
def speed_B : ℕ := 40 -- speed of train B in kmph
def speed_C : ℕ := 60 -- speed of train C in kmph

-- Define the head starts in hours for trains A and B
def head_start_A : ℕ := 9 -- head start for train A in hours
def head_start_B : ℕ := 3 -- head start for train B in hours

-- Define the distances traveled by trains A and B by the time train C starts at 6 p.m.
def distance_A_start : ℕ := speed_A * head_start_A -- distance traveled by train A by 6 p.m.
def distance_B_start : ℕ := speed_B * head_start_B -- distance traveled by train B by 6 p.m.

-- The formula to calculate the distance after t hours from 6 p.m. for each train
def distance_A (t : ℕ) : ℕ := distance_A_start + speed_A * t
def distance_B (t : ℕ) : ℕ := distance_B_start + speed_B * t
def distance_C (t : ℕ) : ℕ := speed_C * t

-- Problem statement to prove the point where all three trains meet
theorem trains_meet_at_distance_360_km : ∃ t : ℕ, distance_A t = 360 ∧ distance_B t = 360 ∧ distance_C t = 360 := by
  sorry

end NUMINAMATH_GPT_trains_meet_at_distance_360_km_l2101_210100


namespace NUMINAMATH_GPT_find_angle_B_l2101_210143

-- Given definitions and conditions
variables {a b c : ℝ}
variables {A B C : ℝ}
variable (h1 : (a + b + c) * (a - b + c) = a * c )

-- Statement of the proof problem
theorem find_angle_B (h1 : (a + b + c) * (a - b + c) = a * c) :
  B = 2 * π / 3 :=
sorry

end NUMINAMATH_GPT_find_angle_B_l2101_210143


namespace NUMINAMATH_GPT_vectors_parallel_l2101_210126

def are_parallel (a b : ℝ × ℝ × ℝ) : Prop := ∃ k : ℝ, b = k • a

theorem vectors_parallel :
  let a := (1, 2, -2)
  let b := (-2, -4, 4)
  are_parallel a b :=
by
  let a := (1, 2, -2)
  let b := (-2, -4, 4)
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_vectors_parallel_l2101_210126


namespace NUMINAMATH_GPT_largest_tile_side_length_l2101_210108

theorem largest_tile_side_length (w h : ℕ) (hw : w = 17) (hh : h = 23) : Nat.gcd w h = 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_largest_tile_side_length_l2101_210108


namespace NUMINAMATH_GPT_evaluate_expression_l2101_210103

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem evaluate_expression :
  (4 / log_base 5 (2500^3) + 2 / log_base 2 (2500^3) = 1 / 3) := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2101_210103


namespace NUMINAMATH_GPT_octahedron_common_sum_is_39_l2101_210101

-- Define the vertices of the regular octahedron with numbers from 1 to 12
def vertices : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the property that the sum of four numbers at the vertices of each triangle face is the same
def common_sum (faces : List (List ℕ)) (k : ℕ) : Prop :=
  ∀ face ∈ faces, face.sum = k

-- Define the faces of the regular octahedron
def faces : List (List ℕ) := [
  [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 5, 9, 6],
  [2, 6, 10, 7], [3, 7, 11, 8], [4, 8, 12, 5], [1, 9, 2, 10]
]

-- Prove that the common sum is 39
theorem octahedron_common_sum_is_39 : common_sum faces 39 :=
  sorry

end NUMINAMATH_GPT_octahedron_common_sum_is_39_l2101_210101


namespace NUMINAMATH_GPT_initial_average_mark_l2101_210141

theorem initial_average_mark (A : ℕ) (A_excluded : ℕ := 20) (A_remaining : ℕ := 90) (n_total : ℕ := 14) (n_excluded : ℕ := 5) :
    (n_total * A = n_excluded * A_excluded + (n_total - n_excluded) * A_remaining) → A = 65 :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_initial_average_mark_l2101_210141


namespace NUMINAMATH_GPT_sara_quarters_final_l2101_210190

def initial_quarters : ℕ := 21
def quarters_from_dad : ℕ := 49
def quarters_spent_at_arcade : ℕ := 15
def dollar_bills_from_mom : ℕ := 2
def quarters_per_dollar : ℕ := 4

theorem sara_quarters_final :
  (initial_quarters + quarters_from_dad - quarters_spent_at_arcade + dollar_bills_from_mom * quarters_per_dollar) = 63 :=
by
  sorry

end NUMINAMATH_GPT_sara_quarters_final_l2101_210190


namespace NUMINAMATH_GPT_area_of_quadrilateral_l2101_210174

theorem area_of_quadrilateral (d a b : ℝ) (h₀ : d = 28) (h₁ : a = 9) (h₂ : b = 6) :
  (1 / 2 * d * a) + (1 / 2 * d * b) = 210 :=
by
  -- Provided proof steps are skipped
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_l2101_210174


namespace NUMINAMATH_GPT_total_tickets_sold_l2101_210107

-- Definitions and conditions
def orchestra_ticket_price : ℕ := 12
def balcony_ticket_price : ℕ := 8
def total_revenue : ℕ := 3320
def ticket_difference : ℕ := 190

-- Variables
variables (x y : ℕ) -- x is the number of orchestra tickets, y is the number of balcony tickets

-- Statements of conditions
def revenue_eq : Prop := orchestra_ticket_price * x + balcony_ticket_price * y = total_revenue
def tickets_relation : Prop := y = x + ticket_difference

-- The proof problem statement
theorem total_tickets_sold (h1 : revenue_eq x y) (h2 : tickets_relation x y) : x + y = 370 :=
by
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l2101_210107


namespace NUMINAMATH_GPT_price_reduction_l2101_210179

theorem price_reduction (x y : ℕ) (h1 : (13 - x) * y = 781) (h2 : y ≤ 100) : x = 2 :=
sorry

end NUMINAMATH_GPT_price_reduction_l2101_210179


namespace NUMINAMATH_GPT_sum_xy_22_l2101_210129

theorem sum_xy_22 (x y : ℕ) (h1 : 0 < x) (h2 : x < 25) (h3 : 0 < y) (h4 : y < 25) 
  (h5 : x + y + x * y = 118) : x + y = 22 :=
sorry

end NUMINAMATH_GPT_sum_xy_22_l2101_210129


namespace NUMINAMATH_GPT_solve_quadratic_l2101_210130

theorem solve_quadratic (x : ℝ) : (x^2 + 2*x = 0) ↔ (x = 0 ∨ x = -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l2101_210130


namespace NUMINAMATH_GPT_distance_x_intercepts_correct_l2101_210187

noncomputable def distance_between_x_intercepts : ℝ :=
  let slope1 : ℝ := 4
  let slope2 : ℝ := -3
  let intercept_point : Prod ℝ ℝ := (8, 20)
  let line1 (x : ℝ) : ℝ := slope1 * (x - intercept_point.1) + intercept_point.2
  let line2 (x : ℝ) : ℝ := slope2 * (x - intercept_point.1) + intercept_point.2
  let x_intercept1 : ℝ := (0 - intercept_point.2) / slope1 + intercept_point.1
  let x_intercept2 : ℝ := (0 - intercept_point.2) / slope2 + intercept_point.1
  abs (x_intercept2 - x_intercept1)

theorem distance_x_intercepts_correct :
  distance_between_x_intercepts = 35 / 3 :=
sorry

end NUMINAMATH_GPT_distance_x_intercepts_correct_l2101_210187


namespace NUMINAMATH_GPT_jacob_additional_money_needed_l2101_210105

/-- Jacob's total trip cost -/
def trip_cost : ℕ := 5000

/-- Jacob's hourly wage -/
def hourly_wage : ℕ := 20

/-- Jacob's working hours -/
def working_hours : ℕ := 10

/-- Income from job -/
def job_income : ℕ := hourly_wage * working_hours

/-- Price per cookie -/
def cookie_price : ℕ := 4

/-- Number of cookies sold -/
def cookies_sold : ℕ := 24

/-- Income from cookies -/
def cookie_income : ℕ := cookie_price * cookies_sold

/-- Lottery ticket cost -/
def lottery_ticket_cost : ℕ := 10

/-- Lottery win amount -/
def lottery_win : ℕ := 500

/-- Money received from each sister -/
def sister_gift : ℕ := 500

/-- Total income from job and cookies -/
def income_without_expenses : ℕ := job_income + cookie_income

/-- Income after lottery ticket purchase -/
def income_after_ticket : ℕ := income_without_expenses - lottery_ticket_cost

/-- Total income after lottery win -/
def income_with_lottery : ℕ := income_after_ticket + lottery_win

/-- Total gift from sisters -/
def total_sisters_gift : ℕ := 2 * sister_gift

/-- Total money Jacob has -/
def total_money : ℕ := income_with_lottery + total_sisters_gift

/-- Additional amount needed by Jacob -/
def additional_needed : ℕ := trip_cost - total_money

theorem jacob_additional_money_needed : additional_needed = 3214 := by
  sorry

end NUMINAMATH_GPT_jacob_additional_money_needed_l2101_210105


namespace NUMINAMATH_GPT_evaluate_expression_l2101_210197

theorem evaluate_expression : 
  ∀ (x y z : ℝ), 
  x = 2 → 
  y = -3 → 
  z = 1 → 
  x^2 + y^2 + z^2 + 2 * x * y - z^3 = 1 := by
  intros x y z hx hy hz
  rw [hx, hy, hz]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2101_210197


namespace NUMINAMATH_GPT_percentage_of_fruits_in_good_condition_l2101_210119

theorem percentage_of_fruits_in_good_condition :
  let total_oranges := 600
  let total_bananas := 400
  let rotten_oranges := (15 / 100.0) * total_oranges
  let rotten_bananas := (8 / 100.0) * total_bananas
  let good_condition_oranges := total_oranges - rotten_oranges
  let good_condition_bananas := total_bananas - rotten_bananas
  let total_fruits := total_oranges + total_bananas
  let total_fruits_in_good_condition := good_condition_oranges + good_condition_bananas
  let percentage_fruits_in_good_condition := (total_fruits_in_good_condition / total_fruits) * 100
  percentage_fruits_in_good_condition = 87.8 := sorry

end NUMINAMATH_GPT_percentage_of_fruits_in_good_condition_l2101_210119


namespace NUMINAMATH_GPT_range_of_a_l2101_210178

theorem range_of_a (a : ℝ) : 
(∀ x : ℝ, |x - 1| + |x - 3| > a ^ 2 - 2 * a - 1) ↔ -1 < a ∧ a < 3 := 
sorry

end NUMINAMATH_GPT_range_of_a_l2101_210178


namespace NUMINAMATH_GPT_download_time_is_2_hours_l2101_210145

theorem download_time_is_2_hours (internet_speed : ℕ) (f1 f2 f3 : ℕ) (total_size : ℕ)
  (total_min : ℕ) (hours : ℕ) :
  internet_speed = 2 ∧ f1 = 80 ∧ f2 = 90 ∧ f3 = 70 ∧ total_size = f1 + f2 + f3
  ∧ total_min = total_size / internet_speed ∧ hours = total_min / 60 → hours = 2 :=
by
  sorry

end NUMINAMATH_GPT_download_time_is_2_hours_l2101_210145


namespace NUMINAMATH_GPT_divisibility_condition_of_exponents_l2101_210191

theorem divisibility_condition_of_exponents (n : ℕ) (h : n ≥ 1) :
  (∀ a b : ℕ, (11 ∣ a^n + b^n) → (11 ∣ a ∧ 11 ∣ b)) ↔ (n % 2 = 0) :=
sorry

end NUMINAMATH_GPT_divisibility_condition_of_exponents_l2101_210191


namespace NUMINAMATH_GPT_y_satisfies_quadratic_l2101_210127

theorem y_satisfies_quadratic (x y : ℝ) 
  (h1 : 2 * x^2 + 6 * x + 5 * y + 1 = 0)
  (h2 : 2 * x + y + 3 = 0) : y^2 + 10 * y - 7 = 0 := 
sorry

end NUMINAMATH_GPT_y_satisfies_quadratic_l2101_210127


namespace NUMINAMATH_GPT_suraya_picked_more_apples_l2101_210146

theorem suraya_picked_more_apples (k c s : ℕ)
  (h_kayla : k = 20)
  (h_caleb : c = k - 5)
  (h_suraya : s = k + 7) :
  s - c = 12 :=
by
  -- Mark this as a place where the proof can be provided
  sorry

end NUMINAMATH_GPT_suraya_picked_more_apples_l2101_210146


namespace NUMINAMATH_GPT_derivative_at_1_l2101_210199

def f (x : ℝ) : ℝ := x^3 + x^2 - 2 * x - 2

def f_derivative (x : ℝ) : ℝ := 3*x^2 + 2*x - 2

theorem derivative_at_1 : f_derivative 1 = 3 := by
  sorry

end NUMINAMATH_GPT_derivative_at_1_l2101_210199


namespace NUMINAMATH_GPT_full_price_ticket_revenue_l2101_210158

theorem full_price_ticket_revenue (f d : ℕ) (p : ℝ) : 
  f + d = 200 → 
  f * p + d * (p / 3) = 3000 → 
  d = 200 - f → 
  (f * p) = 1500 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_full_price_ticket_revenue_l2101_210158


namespace NUMINAMATH_GPT_root_in_interval_l2101_210152

noncomputable def f (x : ℝ) : ℝ := 3^x + 3*x - 8

theorem root_in_interval :
  f 1 < 0 ∧ f 1.5 > 0 ∧ f 1.25 < 0 → ∃ c, 1.25 < c ∧ c < 1.5 ∧ f c = 0 :=
by
  sorry

end NUMINAMATH_GPT_root_in_interval_l2101_210152


namespace NUMINAMATH_GPT_missing_angle_measure_l2101_210120

theorem missing_angle_measure (n : ℕ) (h : 180 * (n - 2) = 3240 + 2 * (180 * (n - 2)) / n) : 
  (180 * (n - 2)) / n = 166 := 
by 
  sorry

end NUMINAMATH_GPT_missing_angle_measure_l2101_210120


namespace NUMINAMATH_GPT_largest_divisor_of_difference_of_squares_l2101_210185

theorem largest_divisor_of_difference_of_squares (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) (h : n < m) :
  ∃ k, (∀ m n : ℤ, m % 2 = 1 → n % 2 = 1 → n < m → k ∣ (m^2 - n^2)) ∧ (∀ j : ℤ, (∀ m n : ℤ, m % 2 = 1 → n % 2 = 1 → n < m → j ∣ (m^2 - n^2)) → j ≤ k) ∧ k = 8 :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_difference_of_squares_l2101_210185


namespace NUMINAMATH_GPT_length_of_bridge_correct_l2101_210117

open Real

noncomputable def length_of_bridge (length_of_train : ℝ) (time_to_cross : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed := speed_kmph * (1000 / 3600)
  let total_distance := speed * time_to_cross
  total_distance - length_of_train

theorem length_of_bridge_correct :
  length_of_bridge 200 34.997200223982084 36 = 149.97200223982084 := by
  sorry

end NUMINAMATH_GPT_length_of_bridge_correct_l2101_210117


namespace NUMINAMATH_GPT_inequlity_for_k_one_smallest_k_l2101_210122

noncomputable def triangle_sides (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

theorem inequlity_for_k_one (a b c : ℝ) (h : triangle_sides a b c) :
  a^3 + b^3 + c^3 < (a + b + c) * (a * b + b * c + c * a) :=
sorry

theorem smallest_k (a b c k : ℝ) (h : triangle_sides a b c) (hk : k = 1) :
  a^3 + b^3 + c^3 < k * (a + b + c) * (a * b + b * c + c * a) :=
sorry

end NUMINAMATH_GPT_inequlity_for_k_one_smallest_k_l2101_210122


namespace NUMINAMATH_GPT_purchase_price_l2101_210156

-- Define the context and conditions 
variables (P S : ℝ)
-- Define the conditions
axiom cond1 : S = P + 0.5 * S
axiom cond2 : S - P = 100

-- Define the main theorem
theorem purchase_price : P = 100 :=
by sorry

end NUMINAMATH_GPT_purchase_price_l2101_210156


namespace NUMINAMATH_GPT_family_members_l2101_210189

theorem family_members (cost_purify : ℝ) (water_per_person : ℝ) (total_cost : ℝ) 
  (h1 : cost_purify = 1) (h2 : water_per_person = 1 / 2) (h3 : total_cost = 3) : 
  total_cost / (cost_purify * water_per_person) = 6 :=
by
  sorry

end NUMINAMATH_GPT_family_members_l2101_210189


namespace NUMINAMATH_GPT_complex_modulus_problem_l2101_210193

noncomputable def imaginary_unit : ℂ := Complex.I

theorem complex_modulus_problem (z : ℂ) (h : (1 + Real.sqrt 3 * imaginary_unit)^2 * z = 1 - imaginary_unit^3) :
  Complex.abs z = Real.sqrt 2 / 4 :=
by
  sorry

end NUMINAMATH_GPT_complex_modulus_problem_l2101_210193


namespace NUMINAMATH_GPT_abs_inequality_solution_set_l2101_210165

theorem abs_inequality_solution_set (x : ℝ) : (|x - 1| ≥ 5) ↔ (x ≥ 6 ∨ x ≤ -4) := 
by sorry

end NUMINAMATH_GPT_abs_inequality_solution_set_l2101_210165


namespace NUMINAMATH_GPT_negation_of_prop_p_l2101_210170

open Classical

variable (p : Prop)

def prop_p := ∀ x : ℝ, x^3 - x^2 + 1 < 0

theorem negation_of_prop_p : ¬prop_p ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 := by
  sorry

end NUMINAMATH_GPT_negation_of_prop_p_l2101_210170


namespace NUMINAMATH_GPT_probability_sunflower_seed_l2101_210196

theorem probability_sunflower_seed :
  ∀ (sunflower_seeds green_bean_seeds pumpkin_seeds : ℕ),
  sunflower_seeds = 2 →
  green_bean_seeds = 3 →
  pumpkin_seeds = 4 →
  (sunflower_seeds + green_bean_seeds + pumpkin_seeds = 9) →
  (sunflower_seeds : ℚ) / (sunflower_seeds + green_bean_seeds + pumpkin_seeds) = 2 / 9 := 
by 
  intros sunflower_seeds green_bean_seeds pumpkin_seeds h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  rw [h1, h2, h3]
  sorry -- Proof omitted as per instructions.

end NUMINAMATH_GPT_probability_sunflower_seed_l2101_210196


namespace NUMINAMATH_GPT_inequality_a4_b4_c4_l2101_210192

theorem inequality_a4_b4_c4 (a b c : Real) : a^4 + b^4 + c^4 ≥ abc * (a + b + c) := 
by
  sorry

end NUMINAMATH_GPT_inequality_a4_b4_c4_l2101_210192


namespace NUMINAMATH_GPT_problem1_problem2_l2101_210102

def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 1|

theorem problem1 (x : ℝ) : f x ≥ 4 ↔ x ≤ -4/3 ∨ x ≥ 4/3 := 
  sorry

theorem problem2 (a : ℝ) : (∀ x : ℝ, f x > a) ↔ a < 3/2 := 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2101_210102


namespace NUMINAMATH_GPT_frequency_group_5_l2101_210128

theorem frequency_group_5 (total_students : ℕ) (freq1 freq2 freq3 freq4 : ℕ)
  (h1 : total_students = 45)
  (h2 : freq1 = 12)
  (h3 : freq2 = 11)
  (h4 : freq3 = 9)
  (h5 : freq4 = 4) :
  ((total_students - (freq1 + freq2 + freq3 + freq4)) / total_students : ℚ) = 0.2 := 
sorry

end NUMINAMATH_GPT_frequency_group_5_l2101_210128


namespace NUMINAMATH_GPT_non_shaded_region_perimeter_l2101_210181

def outer_rectangle_length : ℕ := 12
def outer_rectangle_width : ℕ := 10
def inner_rectangle_length : ℕ := 6
def inner_rectangle_width : ℕ := 2
def shaded_area : ℕ := 116

theorem non_shaded_region_perimeter :
  let total_area := outer_rectangle_length * outer_rectangle_width
  let inner_area := inner_rectangle_length * inner_rectangle_width
  let non_shaded_area := total_area - shaded_area
  non_shaded_area = 4 →
  ∃ width height, width * height = non_shaded_area ∧ 2 * (width + height) = 10 :=
by intros
   sorry

end NUMINAMATH_GPT_non_shaded_region_perimeter_l2101_210181


namespace NUMINAMATH_GPT_diophantine_solution_l2101_210184

theorem diophantine_solution :
  ∃ (x y k : ℤ), 1990 * x - 173 * y = 11 ∧ x = -22 + 173 * k ∧ y = 253 - 1990 * k :=
by {
  sorry
}

end NUMINAMATH_GPT_diophantine_solution_l2101_210184


namespace NUMINAMATH_GPT_fraction_given_to_cousin_l2101_210186

theorem fraction_given_to_cousin
  (initial_candies : ℕ)
  (brother_share sister_share : ℕ)
  (eaten_candies left_candies : ℕ)
  (remaining_candies : ℕ)
  (given_to_cousin : ℕ)
  (fraction : ℚ)
  (h1 : initial_candies = 50)
  (h2 : brother_share = 5)
  (h3 : sister_share = 5)
  (h4 : eaten_candies = 12)
  (h5 : left_candies = 18)
  (h6 : initial_candies - brother_share - sister_share = remaining_candies)
  (h7 : remaining_candies - given_to_cousin - eaten_candies = left_candies)
  (h8 : fraction = (given_to_cousin : ℚ) / (remaining_candies : ℚ))
  : fraction = 1 / 4 := 
sorry

end NUMINAMATH_GPT_fraction_given_to_cousin_l2101_210186


namespace NUMINAMATH_GPT_abigail_initial_money_l2101_210155

variables (A : ℝ) -- Initial amount of money Abigail had.

-- Conditions
variables (food_rate : ℝ := 0.60) -- 60% spent on food
variables (phone_rate : ℝ := 0.25) -- 25% of the remainder spent on phone bill
variables (entertainment_spent : ℝ := 20) -- $20 spent on entertainment
variables (final_amount : ℝ := 40) -- $40 left after all expenditures

theorem abigail_initial_money :
  (A - (A * food_rate)) * (1 - phone_rate) - entertainment_spent = final_amount → A = 200 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_abigail_initial_money_l2101_210155


namespace NUMINAMATH_GPT_polygon_area_correct_l2101_210164

noncomputable def polygonArea : ℝ :=
  let x1 := 1
  let y1 := 1
  let x2 := 4
  let y2 := 3
  let x3 := 5
  let y3 := 1
  let x4 := 6
  let y4 := 4
  let x5 := 3
  let y5 := 6
  (1 / 2 : ℝ) * 
  abs ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y5 + x5 * y1) -
       (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x5 + y5 * x1))

theorem polygon_area_correct : polygonArea = 11.5 := by
  sorry

end NUMINAMATH_GPT_polygon_area_correct_l2101_210164


namespace NUMINAMATH_GPT_problem_statement_l2101_210144

open BigOperators

-- Defining the arithmetic sequence
def a (n : ℕ) : ℕ := n - 1

-- Defining the sequence b_n
def b (n : ℕ) : ℕ :=
if n % 2 = 1 then
  a n + 1
else
  2 ^ a n

-- Defining T_2n as the sum of the first 2n terms of b
def T (n : ℕ) : ℕ :=
(∑ i in Finset.range n, b (2 * i + 1)) +
(∑ i in Finset.range n, b (2 * i + 2))

-- The theorem to be proven
theorem problem_statement (n : ℕ) : 
  a 2 * (a 4 + 1) = a 3 ^ 2 ∧
  T n = n^2 + (2^(2*n+1) - 2) / 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2101_210144


namespace NUMINAMATH_GPT_smallest_triangle_perimeter_consecutive_even_l2101_210109

theorem smallest_triangle_perimeter_consecutive_even :
  ∃ (a b c : ℕ), a = 2 ∧ b = 4 ∧ c = 6 ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ (a + b + c = 12) :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_triangle_perimeter_consecutive_even_l2101_210109


namespace NUMINAMATH_GPT_cube_of_square_is_15625_l2101_210115

/-- The third smallest prime number is 5 --/
def third_smallest_prime := 5

/-- The square of 5 is 25 --/
def square_of_third_smallest_prime := third_smallest_prime ^ 2

/-- The cube of the square of the third smallest prime number is 15625 --/
def cube_of_square_of_third_smallest_prime := square_of_third_smallest_prime ^ 3

theorem cube_of_square_is_15625 : cube_of_square_of_third_smallest_prime = 15625 := by
  sorry

end NUMINAMATH_GPT_cube_of_square_is_15625_l2101_210115


namespace NUMINAMATH_GPT_degrees_to_radians_300_l2101_210148

theorem degrees_to_radians_300:
  (300 * (Real.pi / 180) = 5 * Real.pi / 3) := 
by
  repeat { sorry }

end NUMINAMATH_GPT_degrees_to_radians_300_l2101_210148


namespace NUMINAMATH_GPT_find_p_q_r_sum_l2101_210124

theorem find_p_q_r_sum (p q r : ℕ) (hpq_rel_prime : Nat.gcd p q = 1) (hq_nonzero : q ≠ 0) 
  (h1 : ∃ t, (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4) 
  (h2 : ∃ t, (1 - Real.sin t) * (1 - Real.cos t) = p / q - Real.sqrt r) : 
  p + q + r = 7 :=
sorry

end NUMINAMATH_GPT_find_p_q_r_sum_l2101_210124
