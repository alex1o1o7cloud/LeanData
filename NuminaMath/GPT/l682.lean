import Mathlib

namespace max_zeros_consecutive_two_digit_product_l682_68210

theorem max_zeros_consecutive_two_digit_product :
  ∃ a b : ℕ, 10 ≤ a ∧ a < 100 ∧ b = a + 1 ∧ 10 ≤ b ∧ b < 100 ∧
  (∀ c, (c * 10) ∣ a * b → c ≤ 2) := 
  by
    sorry

end max_zeros_consecutive_two_digit_product_l682_68210


namespace S7_value_l682_68273

def arithmetic_seq_sum (n : ℕ) (a_1 d : ℚ) : ℚ :=
  n * a_1 + (n * (n - 1) / 2) * d

def a_n (n : ℕ) (a_1 d : ℚ) : ℚ :=
  a_1 + (n - 1) * d

theorem S7_value (a_1 d : ℚ) (S_n : ℕ → ℚ)
  (hSn_def : ∀ n, S_n n = arithmetic_seq_sum n a_1 d)
  (h_sum_condition : S_n 7 + S_n 5 = 10)
  (h_a3_condition : a_n 3 a_1 d = 5) :
  S_n 7 = -15 :=
by
  sorry

end S7_value_l682_68273


namespace fraction_power_evaluation_l682_68274

theorem fraction_power_evaluation (x y : ℚ) (h1 : x = 2 / 3) (h2 : y = 3 / 2) : 
  (3 / 4) * x^8 * y^9 = 9 / 8 := 
by
  sorry

end fraction_power_evaluation_l682_68274


namespace triangle_pyramid_angle_l682_68283

theorem triangle_pyramid_angle (φ : ℝ) (vertex_angle : ∀ (A B C : ℝ), (A + B + C = φ)) :
  ∃ θ : ℝ, θ = φ :=
by
  sorry

end triangle_pyramid_angle_l682_68283


namespace value_of_k_l682_68200

theorem value_of_k (k : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : y = (k - 1) * x + k^2 - 1)
  (h2 : ∃ m : ℝ, y = m * x)
  (h3 : k ≠ 1) :
  k = -1 :=
by
  sorry

end value_of_k_l682_68200


namespace proof_parabola_statements_l682_68242

theorem proof_parabola_statements (b c : ℝ)
  (h1 : 1/2 - b + c < 0)
  (h2 : 2 - 2 * b + c < 0) :
  (b^2 > 2 * c) ∧
  (c > 1 → b > 3/2) ∧
  (∀ (m1 m2 : ℝ), m1 < m2 ∧ m2 < b → ∀ (y : ℝ), y = (1/2)*m1^2 - b*m1 + c → ∀ (y2 : ℝ), y2 = (1/2)*m2^2 - b*m2 + c → y > y2) ∧
  (¬(∃ x1 x2 : ℝ, (1/2) * x1^2 - b * x1 + c = 0 ∧ (1/2) * x2^2 - b * x2 + c = 0 ∧ x1 + x2 > 3)) :=
by sorry

end proof_parabola_statements_l682_68242


namespace viewing_spot_coordinate_correct_l682_68254

-- Define the coordinates of the landmarks
def first_landmark := 150
def second_landmark := 450

-- The expected coordinate of the viewing spot
def expected_viewing_spot := 350

-- The theorem that formalizes the problem
theorem viewing_spot_coordinate_correct :
  let distance := second_landmark - first_landmark
  let fractional_distance := (2 / 3) * distance
  let viewing_spot := first_landmark + fractional_distance
  viewing_spot = expected_viewing_spot := 
by
  -- This is where the proof would go
  sorry

end viewing_spot_coordinate_correct_l682_68254


namespace imaginary_unit_power_l682_68259

def i := Complex.I

theorem imaginary_unit_power :
  ∀ a : ℝ, (2 - i + a * i ^ 2011).im = 0 → i ^ 2011 = i :=
by
  intro a
  intro h
  sorry

end imaginary_unit_power_l682_68259


namespace binary_sum_in_base_10_l682_68282

theorem binary_sum_in_base_10 :
  (255 : ℕ) + (63 : ℕ) = 318 :=
sorry

end binary_sum_in_base_10_l682_68282


namespace distance_between_consecutive_trees_l682_68230

-- Definitions from the problem statement
def yard_length : ℕ := 414
def number_of_trees : ℕ := 24
def number_of_intervals : ℕ := number_of_trees - 1
def distance_between_trees : ℕ := yard_length / number_of_intervals

-- Main theorem we want to prove
theorem distance_between_consecutive_trees :
  distance_between_trees = 18 := by
  -- Proof would go here
  sorry

end distance_between_consecutive_trees_l682_68230


namespace max_value_of_expression_l682_68205

theorem max_value_of_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 29 := 
sorry

end max_value_of_expression_l682_68205


namespace prod_ge_27_eq_iff_equality_l682_68236

variable (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
          (h4 : a + b + c + 2 = a * b * c)

theorem prod_ge_27 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + b + c + 2 = a * b * c) : (a + 1) * (b + 1) * (c + 1) ≥ 27 :=
by sorry

theorem eq_iff_equality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + b + c + 2 = a * b * c) : 
  ((a + 1) * (b + 1) * (c + 1) = 27) ↔ (a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end prod_ge_27_eq_iff_equality_l682_68236


namespace initial_stickers_correct_l682_68289

-- Definitions based on the conditions
def initial_stickers (X : ℕ) : ℕ := X
def after_buying (X : ℕ) : ℕ := X + 26
def after_birthday (X : ℕ) : ℕ := after_buying X + 20
def after_giving (X : ℕ) : ℕ := after_birthday X - 6
def after_decorating (X : ℕ) : ℕ := after_giving X - 58

-- Theorem stating the problem and the expected answer
theorem initial_stickers_correct (X : ℕ) (h : after_decorating X = 2) : initial_stickers X = 26 :=
by {
  sorry
}

end initial_stickers_correct_l682_68289


namespace average_price_of_cow_l682_68219

variable (price_cow price_goat : ℝ)

theorem average_price_of_cow (h1 : 2 * price_cow + 8 * price_goat = 1400)
                             (h2 : price_goat = 60) :
                             price_cow = 460 := 
by
  -- The following line allows the Lean code to compile successfully without providing a proof.
  sorry

end average_price_of_cow_l682_68219


namespace debby_drink_days_l682_68293

theorem debby_drink_days :
  ∀ (total_bottles : ℕ) (bottles_per_day : ℕ) (remaining_bottles : ℕ),
  total_bottles = 301 →
  bottles_per_day = 144 →
  remaining_bottles = 157 →
  (total_bottles - remaining_bottles) / bottles_per_day = 1 :=
by
  intros total_bottles bottles_per_day remaining_bottles ht he hb
  sorry

end debby_drink_days_l682_68293


namespace find_2n_plus_m_l682_68294

theorem find_2n_plus_m (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 
  2 * n + m = 36 := 
sorry

end find_2n_plus_m_l682_68294


namespace rhombus_side_length_l682_68237

noncomputable def side_length_rhombus (AB BC AC : ℝ) (condition1 : AB = 12) (condition2 : BC = 12) (condition3 : AC = 6) : ℝ :=
  4

theorem rhombus_side_length (AB BC AC : ℝ) (condition1 : AB = 12) (condition2 : BC = 12) (condition3 : AC = 6) (x : ℝ) :
  side_length_rhombus AB BC AC condition1 condition2 condition3 = x ↔ x = 4 := by
  sorry

end rhombus_side_length_l682_68237


namespace money_bounds_l682_68207

variables (c d : ℝ)

theorem money_bounds :
  (7 * c + d > 84) ∧ (5 * c - d = 35) → (c > 9.92 ∧ d > 14.58) :=
by
  intro h
  sorry

end money_bounds_l682_68207


namespace factor_expression_l682_68222

-- Define variables s and m
variables (s m : ℤ)

-- State the theorem to be proven: If s = 5, then m^2 - sm - 24 can be factored as (m - 8)(m + 3)
theorem factor_expression (hs : s = 5) : m^2 - s * m - 24 = (m - 8) * (m + 3) :=
by {
  sorry
}

end factor_expression_l682_68222


namespace repeating_decimals_sum_l682_68288

-- Define the repeating decimals as rational numbers
def dec_0_3 : ℚ := 1 / 3
def dec_0_02 : ℚ := 2 / 99
def dec_0_0004 : ℚ := 4 / 9999

-- State the theorem that we need to prove
theorem repeating_decimals_sum :
  dec_0_3 + dec_0_02 + dec_0_0004 = 10581 / 29889 :=
by
  sorry

end repeating_decimals_sum_l682_68288


namespace geometric_sequences_l682_68246

theorem geometric_sequences :
  ∃ (a q : ℝ) (a1 a2 a3 : ℕ → ℝ), 
    (∀ n, a1 n = a * (q - 2) ^ n) ∧ 
    (∀ n, a2 n = 2 * a * (q - 1) ^ n) ∧ 
    (∀ n, a3 n = 4 * a * q ^ n) ∧
    a = 1 ∧ q = 4 ∨ a = 192 / 31 ∧ q = 9 / 8 ∧
    (a + 2 * a + 4 * a = 84) ∧
    (a * (q - 2) + 2 * a * (q - 1) + 4 * a * q = 24) :=
sorry

end geometric_sequences_l682_68246


namespace approximation_hundred_thousandth_place_l682_68231

theorem approximation_hundred_thousandth_place (n : ℕ) (h : n = 537400000) : 
  ∃ p : ℕ, p = 100000 := 
sorry

end approximation_hundred_thousandth_place_l682_68231


namespace quadratic_discriminant_l682_68249

noncomputable def discriminant (a b c : ℚ) : ℚ :=
  b^2 - 4 * a * c

theorem quadratic_discriminant :
  discriminant 6 (6 + 1/6) (1/6) = 1225 / 36 :=
by
  sorry

end quadratic_discriminant_l682_68249


namespace solve_for_x_l682_68229

theorem solve_for_x (x : ℝ) (h₀ : x > 0) (h₁ : 1 / 2 * x * (3 * x) = 96) : x = 8 :=
sorry

end solve_for_x_l682_68229


namespace relationship_between_a_and_b_l682_68204

theorem relationship_between_a_and_b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b)
    (h₃ : ∀ x : ℝ, |(3 * x + 1) - 4| < a → |x - 1| < b) : a ≥ 3 * b :=
by
  -- Applying the given conditions, we want to demonstrate that a ≥ 3b.
  sorry

end relationship_between_a_and_b_l682_68204


namespace ion_electronic_structure_l682_68278

theorem ion_electronic_structure (R M Z n m X : ℤ) (h1 : R + X = M - n) (h2 : M - n = Z - m) (h3 : n > m) : M > Z ∧ Z > R := 
by 
  sorry

end ion_electronic_structure_l682_68278


namespace volume_of_prism_l682_68238

noncomputable def prismVolume {x y z : ℝ} 
  (h1 : x * y = 20) 
  (h2 : y * z = 12) 
  (h3 : x * z = 8) : ℝ :=
  x * y * z

theorem volume_of_prism (x y z : ℝ)
  (h1 : x * y = 20)
  (h2 : y * z = 12)
  (h3 : x * z = 8) : prismVolume h1 h2 h3 = 8 * Real.sqrt 15 :=
by
  sorry

end volume_of_prism_l682_68238


namespace division_problem_l682_68262

theorem division_problem (D d q r : ℕ) 
  (h1 : D + d + q + r = 205)
  (h2 : q = d) :
  D = 174 ∧ d = 13 :=
by {
  sorry
}

end division_problem_l682_68262


namespace solve_for_x_l682_68298

def operation (a b : ℝ) : ℝ := a^2 - 3*a + b

theorem solve_for_x (x : ℝ) : operation x 2 = 6 → (x = -1 ∨ x = 4) :=
by
  sorry

end solve_for_x_l682_68298


namespace find_a_l682_68281

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x < 0 then -(Real.log (-x) / Real.log 2) + a else 0

theorem find_a (a : ℝ) :
  (f a (-2) + f a (-4) = 1) → a = 2 :=
by
  sorry

end find_a_l682_68281


namespace range_of_k_value_of_k_l682_68220

-- Defining the quadratic equation having two real roots condition
def has_real_roots (k : ℝ) : Prop :=
  let Δ := 9 - 4 * (k - 2)
  Δ ≥ 0

-- First part: range of k
theorem range_of_k (k : ℝ) : has_real_roots k ↔ k ≤ 17 / 4 :=
  sorry

-- Second part: specific value of k given additional condition
theorem value_of_k (x1 x2 k : ℝ) (h1 : (x1 + x2) = 3) (h2 : (x1 * x2) = k - 2) (h3 : (x1 + x2 - x1 * x2) = 1) : k = 4 :=
  sorry

end range_of_k_value_of_k_l682_68220


namespace new_ratio_first_term_less_than_implied_l682_68268

-- Define the original and new ratios
def original_ratio := (6, 7)
def subtracted_value := 3
def new_ratio := (original_ratio.1 - subtracted_value, original_ratio.2 - subtracted_value)

-- Prove the required property
theorem new_ratio_first_term_less_than_implied {r1 r2 : ℕ} (h : new_ratio = (3, 4))
  (h_less : r1 > 3) :
  new_ratio.1 < r1 := 
sorry

end new_ratio_first_term_less_than_implied_l682_68268


namespace islander_real_name_l682_68244

-- Definition of types of people on the island
inductive IslanderType
| Knight   -- Always tells the truth
| Liar     -- Always lies
| Normal   -- Can lie or tell the truth

-- The possible names of the islander
inductive Name
| Edwin
| Edward

-- Condition: You met the islander who can be Edwin or Edward
def possible_names : List Name := [Name.Edwin, Name.Edward]

-- Condition: The islander said their name is Edward
def islander_statement : Name := Name.Edward

-- Condition: The islander is a Liar (as per the solution interpretation)
def islander_type : IslanderType := IslanderType.Liar

-- The proof problem: Prove the islander's real name is Edwin
theorem islander_real_name : islander_type = IslanderType.Liar ∧ islander_statement = Name.Edward → ∃ n : Name, n = Name.Edwin :=
by
  sorry

end islander_real_name_l682_68244


namespace age_difference_l682_68235

variables (F S M B : ℕ)

theorem age_difference:
  (F - S = 38) → (M - B = 36) → (F - M = 6) → (S - B = 4) :=
by
  intros h1 h2 h3
  -- Use the conditions to derive that S - B = 4
  sorry

end age_difference_l682_68235


namespace p3_mp_odd_iff_m_even_l682_68295

theorem p3_mp_odd_iff_m_even (p m : ℕ) (hp : p % 2 = 1) : (p^3 + m * p) % 2 = 1 ↔ m % 2 = 0 := sorry

end p3_mp_odd_iff_m_even_l682_68295


namespace stock_value_sale_l682_68245

theorem stock_value_sale
  (X : ℝ)
  (h1 : 0.20 * X * 0.10 - 0.80 * X * 0.05 = -350) :
  X = 17500 := by
  -- Proof goes here
  sorry

end stock_value_sale_l682_68245


namespace area_enclosed_by_curve_l682_68270

theorem area_enclosed_by_curve :
  let arc_length := (3 * Real.pi) / 4
  let side_length := 3
  let radius := arc_length / ((3 * Real.pi) / 4)
  let sector_area := (radius ^ 2 * Real.pi * (3 * Real.pi) / (4 * 2 * Real.pi))
  let total_sector_area := 8 * sector_area
  let octagon_area := 2 * (1 + Real.sqrt 2) * (side_length ^ 2)
  total_sector_area + octagon_area = 54 + 54 * Real.sqrt 2 + 3 * Real.pi
:= sorry

end area_enclosed_by_curve_l682_68270


namespace geometric_sum_first_8_terms_eq_17_l682_68209

theorem geometric_sum_first_8_terms_eq_17 (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = 2 * a n)
  (h2 : a 0 + a 1 + a 2 + a 3 = 1) : 
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 17 :=
sorry

end geometric_sum_first_8_terms_eq_17_l682_68209


namespace sum_mod_15_l682_68243

theorem sum_mod_15 
  (d e f : ℕ) 
  (hd : d % 15 = 11)
  (he : e % 15 = 12)
  (hf : f % 15 = 13) : 
  (d + e + f) % 15 = 6 :=
by
  sorry

end sum_mod_15_l682_68243


namespace hall_reunion_attendance_l682_68279

/-- At the Taj Hotel, two family reunions are happening: the Oates reunion and the Hall reunion.
All 150 guests at the hotel attend at least one of the reunions.
70 people attend the Oates reunion.
28 people attend both reunions.
Prove that 108 people attend the Hall reunion. -/
theorem hall_reunion_attendance (total oates both : ℕ) (h_total : total = 150) (h_oates : oates = 70) (h_both : both = 28) :
  ∃ hall : ℕ, total = oates + hall - both ∧ hall = 108 :=
by
  -- Proof will be skipped and not considered for this task
  sorry

end hall_reunion_attendance_l682_68279


namespace randy_piggy_bank_final_amount_l682_68261

def initial_amount : ℕ := 200
def spending_per_trip : ℕ := 2
def trips_per_month : ℕ := 4
def months_per_year : ℕ := 12

theorem randy_piggy_bank_final_amount :
  initial_amount - (spending_per_trip * trips_per_month * months_per_year) = 104 :=
by
  -- proof to be filled in
  sorry

end randy_piggy_bank_final_amount_l682_68261


namespace ball_attendance_l682_68247

variable (n m : ℕ)

def ball_conditions (n m : ℕ) := 
  n + m < 50 ∧ 
  4 ∣ 3 * n ∧ 
  7 ∣ 5 * m

theorem ball_attendance (n m : ℕ) (h : ball_conditions n m) : 
  n + m = 41 :=
sorry

end ball_attendance_l682_68247


namespace total_handshakes_l682_68260

-- Definitions based on conditions
def num_wizards : ℕ := 25
def num_elves : ℕ := 18

-- Each wizard shakes hands with every other wizard
def wizard_handshakes : ℕ := num_wizards * (num_wizards - 1) / 2

-- Each elf shakes hands with every wizard
def elf_wizard_handshakes : ℕ := num_elves * num_wizards

-- Total handshakes is the sum of the above two
theorem total_handshakes : wizard_handshakes + elf_wizard_handshakes = 750 := by
  sorry

end total_handshakes_l682_68260


namespace supermarket_flour_import_l682_68225

theorem supermarket_flour_import :
  let long_grain_rice := (9 : ℚ) / 20
  let glutinous_rice := (7 : ℚ) / 20
  let combined_rice := long_grain_rice + glutinous_rice
  let less_amount := (3 : ℚ) / 20
  let flour : ℚ := combined_rice - less_amount
  flour = (13 : ℚ) / 20 :=
by
  sorry

end supermarket_flour_import_l682_68225


namespace roots_inverse_cubed_l682_68286

-- Define the conditions and the problem statement
theorem roots_inverse_cubed (p q m r s : ℝ) (h1 : r + s = -q / p) (h2 : r * s = m / p) 
  (h3 : ∀ x : ℝ, p * x^2 + q * x + m = 0 → x = r ∨ x = s) : 
  1 / r^3 + 1 / s^3 = (-q^3 + 3 * q * m) / m^3 := 
sorry

end roots_inverse_cubed_l682_68286


namespace sqrt_range_real_l682_68297

theorem sqrt_range_real (x : ℝ) (h : 1 - 3 * x ≥ 0) : x ≤ 1 / 3 :=
sorry

end sqrt_range_real_l682_68297


namespace molecular_weight_of_acetic_acid_l682_68202

-- Define the molecular weight of 7 moles of acetic acid
def molecular_weight_7_moles_acetic_acid := 420 

-- Define the number of moles of acetic acid
def moles_acetic_acid := 7

-- Define the molecular weight of 1 mole of acetic acid
def molecular_weight_1_mole_acetic_acid := molecular_weight_7_moles_acetic_acid / moles_acetic_acid

-- The theorem stating that given the molecular weight of 7 moles of acetic acid, we have the molecular weight of the acetic acid
theorem molecular_weight_of_acetic_acid : molecular_weight_1_mole_acetic_acid = 60 := by
  -- proof to be solved
  sorry

end molecular_weight_of_acetic_acid_l682_68202


namespace classroom_students_l682_68272

theorem classroom_students (n : ℕ) (h1 : 20 < n ∧ n < 30) 
  (h2 : ∃ n_y : ℕ, n = 3 * n_y + 1) 
  (h3 : ∃ n_y' : ℕ, n = (4 * (n - 1)) / 3 + 1) :
  n = 25 := 
by sorry

end classroom_students_l682_68272


namespace friends_total_sales_l682_68275

theorem friends_total_sales :
  (Ryan Jason Zachary : ℕ) →
  (H1 : Ryan = Jason + 50) →
  (H2 : Jason = Zachary + (3 * Zachary / 10)) →
  (H3 : Zachary = 40 * 5) →
  Ryan + Jason + Zachary = 770 :=
by
  sorry

end friends_total_sales_l682_68275


namespace num_even_3digit_nums_lt_700_l682_68269

theorem num_even_3digit_nums_lt_700 
  (digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}) 
  (even_digits : Finset ℕ := {2, 4, 6}) 
  (h1 : ∀ n ∈ digits, n < 10)
  (h2 : 0 ∉ digits) : 
  ∃ n, n = 126 ∧ ∀ d, d ∈ digits → 
  (d < 10) ∧ ∀ u, u ∈ even_digits → 
  (u < 10) 
:=
  sorry

end num_even_3digit_nums_lt_700_l682_68269


namespace pets_remaining_is_correct_l682_68241

-- Definitions for the initial conditions and actions taken
def initial_puppies : Nat := 7
def initial_kittens : Nat := 6
def puppies_sold : Nat := 2
def kittens_sold : Nat := 3

-- Definition that calculates the remaining number of pets
def remaining_pets : Nat := initial_puppies + initial_kittens - (puppies_sold + kittens_sold)

-- The theorem to prove
theorem pets_remaining_is_correct : remaining_pets = 8 := by sorry

end pets_remaining_is_correct_l682_68241


namespace regular_triangular_pyramid_volume_l682_68267

noncomputable def pyramid_volume (a h γ : ℝ) : ℝ :=
  (Real.sqrt 3 * a^2 * h) / 12

theorem regular_triangular_pyramid_volume
  (a h γ : ℝ) (h_nonneg : 0 ≤ h) (γ_nonneg : 0 ≤ γ) :
  pyramid_volume a h γ = (Real.sqrt 3 * a^2 * h) / 12 :=
by
  sorry

end regular_triangular_pyramid_volume_l682_68267


namespace solve_system_of_equations_l682_68239

theorem solve_system_of_equations (x y : ℝ) (h1 : x - y = -5) (h2 : 3 * x + 2 * y = 10) : x = 0 ∧ y = 5 := by
  sorry

end solve_system_of_equations_l682_68239


namespace exists_divisor_for_all_f_values_l682_68215

theorem exists_divisor_for_all_f_values (f : ℕ → ℕ) (h_f_range : ∀ n, 1 < f n) (h_f_div : ∀ m n, f (m + n) ∣ f m + f n) :
  ∃ c : ℕ, c > 1 ∧ ∀ n, c ∣ f n := 
sorry

end exists_divisor_for_all_f_values_l682_68215


namespace total_money_at_least_108_l682_68212

-- Definitions for the problem
def tram_ticket_cost : ℕ := 1
def passenger_coins (n : ℕ) : Prop := n = 2 ∨ n = 5

-- Condition that conductor had no change initially
def initial_conductor_money : ℕ := 0

-- Condition that each passenger can pay exactly 1 Ft and receive change
def can_pay_ticket_with_change (coins : List ℕ) : Prop := 
  ∀ c ∈ coins, passenger_coins c → 
    ∃ change : List ℕ, (change.sum = c - tram_ticket_cost) ∧ 
      (∀ x ∈ change, passenger_coins x)

-- Assume we have 20 passengers with only 2 Ft and 5 Ft coins
def passengers_coins : List (List ℕ) :=
  -- Simplified representation
  List.replicate 20 [2, 5]

noncomputable def total_passenger_money : ℕ :=
  (passengers_coins.map List.sum).sum

-- Lean statement for the proof problem
theorem total_money_at_least_108 : total_passenger_money ≥ 108 :=
sorry

end total_money_at_least_108_l682_68212


namespace most_likely_outcome_is_draw_l682_68226

noncomputable def prob_A_win : ℝ := 0.3
noncomputable def prob_A_not_lose : ℝ := 0.7
noncomputable def prob_draw : ℝ := prob_A_not_lose - prob_A_win

theorem most_likely_outcome_is_draw :
  prob_draw = 0.4 ∧ prob_draw > prob_A_win ∧ prob_draw > (1 - prob_A_not_lose) :=
by
  -- proof goes here
  sorry

end most_likely_outcome_is_draw_l682_68226


namespace math_proof_equivalent_l682_68213

theorem math_proof_equivalent :
  (60 + 5 * 12) / (Real.sqrt 180 / 3) ^ 2 = 6 := by
  sorry

end math_proof_equivalent_l682_68213


namespace minimum_a_condition_l682_68227

theorem minimum_a_condition (a : ℝ) (h₀ : 0 < a) 
  (h₁ : ∀ x : ℝ, 1 < x → x + a / (x - 1) ≥ 5) :
  4 ≤ a :=
sorry

end minimum_a_condition_l682_68227


namespace determine_points_on_line_l682_68292

def pointA : ℝ × ℝ := (2, 5)
def pointB : ℝ × ℝ := (1, 2.2)
def line_eq (x y : ℝ) : ℝ := 3 * x - 5 * y + 8

theorem determine_points_on_line :
  (line_eq pointA.1 pointA.2 ≠ 0) ∧ (line_eq pointB.1 pointB.2 = 0) :=
by
  sorry

end determine_points_on_line_l682_68292


namespace power_sum_ge_three_l682_68255

theorem power_sum_ge_three {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 3) :
  a ^ a + b ^ b + c ^ c ≥ 3 :=
by
  sorry

end power_sum_ge_three_l682_68255


namespace eight_hash_six_l682_68266

def op (r s : ℝ) : ℝ := sorry

axiom op_r_zero (r : ℝ): op r 0 = r + 1
axiom op_comm (r s : ℝ) : op r s = op s r
axiom op_r_add_one_s (r s : ℝ): op (r + 1) s = (op r s) + s + 2

theorem eight_hash_six : op 8 6 = 69 := 
by sorry

end eight_hash_six_l682_68266


namespace area_of_circle_l682_68240

theorem area_of_circle (r θ : ℝ) (h : r = 4 * Real.cos θ - 3 * Real.sin θ) :
  ∃ π : ℝ, π * (5/2)^2 = 25 * π / 4 :=
by 
  sorry

end area_of_circle_l682_68240


namespace aria_spent_on_cookies_l682_68201

def aria_spent : ℕ := 2356

theorem aria_spent_on_cookies :
  (let cookies_per_day := 4
  let cost_per_cookie := 19
  let days_in_march := 31
  let total_cookies := days_in_march * cookies_per_day
  let total_cost := total_cookies * cost_per_cookie
  total_cost = aria_spent) :=
  sorry

end aria_spent_on_cookies_l682_68201


namespace pascal_triangle_row10_sum_l682_68252

def pascal_triangle_row_sum (n : ℕ) : ℕ :=
  2 ^ n

theorem pascal_triangle_row10_sum : pascal_triangle_row_sum 10 = 1024 :=
by
  -- Proof will demonstrate that 2^10 = 1024
  sorry

end pascal_triangle_row10_sum_l682_68252


namespace sphere_volume_diameter_l682_68285

theorem sphere_volume_diameter {D : ℝ} : 
  (D^3/2 + (1/21) * (D^3/2)) = (π * D^3 / 6) ↔ π = 22 / 7 := 
sorry

end sphere_volume_diameter_l682_68285


namespace countMultiplesOf30Between900And27000_l682_68216

noncomputable def smallestPerfectSquareDivisibleBy30 : ℕ :=
  900

noncomputable def smallestPerfectCubeDivisibleBy30 : ℕ :=
  27000

theorem countMultiplesOf30Between900And27000 :
  let lower_bound := smallestPerfectSquareDivisibleBy30 / 30;
  let upper_bound := smallestPerfectCubeDivisibleBy30 / 30;
  upper_bound - lower_bound + 1 = 871 :=
  by
  let lower_bound := smallestPerfectSquareDivisibleBy30 / 30;
  let upper_bound := smallestPerfectCubeDivisibleBy30 / 30;
  show upper_bound - lower_bound + 1 = 871;
  sorry

end countMultiplesOf30Between900And27000_l682_68216


namespace max_number_of_rectangles_in_square_l682_68224

-- Definitions and conditions
def area_square (n : ℕ) : ℕ := 4 * n^2
def area_rectangle (n : ℕ) : ℕ := n + 1
def max_rectangles (n : ℕ) : ℕ := area_square n / area_rectangle n

-- Lean theorem statement for the proof problem
theorem max_number_of_rectangles_in_square (n : ℕ) (h : n ≥ 4) :
  max_rectangles n = 4 * (n - 1) :=
sorry

end max_number_of_rectangles_in_square_l682_68224


namespace max_a_value_l682_68251

theorem max_a_value (a : ℝ)
  (H : ∀ x : ℝ, (x - 1) * x - (a - 2) * (a + 1) ≥ 1) :
  a ≤ 3 / 2 := by
  sorry

end max_a_value_l682_68251


namespace mary_mac_download_time_l682_68228

theorem mary_mac_download_time (x : ℕ) (windows_download : ℕ) (total_glitch : ℕ) (time_without_glitches : ℕ) (total_time : ℕ) :
  windows_download = 3 * x ∧
  total_glitch = 14 ∧
  time_without_glitches = 2 * total_glitch ∧
  total_time = 82 ∧
  x + windows_download + total_glitch + time_without_glitches = total_time →
  x = 10 :=
by 
  sorry

end mary_mac_download_time_l682_68228


namespace eight_point_shots_count_is_nine_l682_68277

def num_8_point_shots (x y z : ℕ) := 8 * x + 9 * y + 10 * z = 100 ∧
                                      x + y + z > 11 ∧ 
                                      x + y + z ≤ 12 ∧ 
                                      x > 0 ∧ 
                                      y > 0 ∧ 
                                      z > 0

theorem eight_point_shots_count_is_nine : 
  ∃ x y z : ℕ, num_8_point_shots x y z ∧ x = 9 :=
by
  sorry

end eight_point_shots_count_is_nine_l682_68277


namespace femaleRainbowTroutCount_l682_68253

noncomputable def numFemaleRainbowTrout : ℕ := 
  let numSpeckledTrout := 645
  let numFemaleSpeckled := 200
  let numMaleSpeckled := 445
  let numMaleRainbow := 150
  let totalTrout := 1000
  let numRainbowTrout := totalTrout - numSpeckledTrout
  numRainbowTrout - numMaleRainbow

theorem femaleRainbowTroutCount : numFemaleRainbowTrout = 205 := by
  -- Conditions
  let numSpeckledTrout : ℕ := 645
  let numMaleSpeckled := 2 * 200 + 45
  let totalTrout := 645 + 355
  let numRainbowTrout := totalTrout - numSpeckledTrout
  let numFemaleRainbow := numRainbowTrout - 150
  
  -- The proof would proceed here
  sorry

end femaleRainbowTroutCount_l682_68253


namespace max_stamps_l682_68232

theorem max_stamps (n friends extra total: ℕ) (h1: friends = 15) (h2: extra = 5) (h3: total < 150) : total ≤ 140 :=
by
  sorry

end max_stamps_l682_68232


namespace find_exponent_l682_68296

theorem find_exponent 
  (h1 : (1 : ℝ) / 9 = 3 ^ (-2 : ℝ))
  (h2 : (3 ^ (20 : ℝ) : ℝ) / 9 = 3 ^ x) : 
  x = 18 :=
by sorry

end find_exponent_l682_68296


namespace range_of_a_l682_68271

noncomputable def f (x : ℝ) : ℝ := 4 * x + 3 * Real.sin x

theorem range_of_a (a : ℝ) (h : f (1 - a) + f (1 - a^2) < 0) : 1 < a ∧ a < Real.sqrt 2 := sorry

end range_of_a_l682_68271


namespace sum_of_first_2m_terms_l682_68280

variable (m : ℕ)
variable (S : ℕ → ℤ)

-- Conditions
axiom Sm : S m = 100
axiom S3m : S (3 * m) = -150

-- Theorem statement
theorem sum_of_first_2m_terms : S (2 * m) = 50 :=
by
  sorry

end sum_of_first_2m_terms_l682_68280


namespace age_of_15th_person_l682_68221

theorem age_of_15th_person (avg_16 : ℝ) (avg_5 : ℝ) (avg_9 : ℝ) (total_16 : ℝ) (total_5 : ℝ) (total_9 : ℝ) :
  avg_16 = 15 ∧ avg_5 = 14 ∧ avg_9 = 16 ∧
  total_16 = 16 * avg_16 ∧ total_5 = 5 * avg_5 ∧ total_9 = 9 * avg_9 →
  (total_16 - total_5 - total_9) = 26 :=
by
  sorry

end age_of_15th_person_l682_68221


namespace distance_traveled_l682_68206

-- Define constants for speed and time
def speed : ℝ := 60
def time : ℝ := 5

-- Define the expected distance
def expected_distance : ℝ := 300

-- Theorem statement
theorem distance_traveled : speed * time = expected_distance :=
by
  sorry

end distance_traveled_l682_68206


namespace max_area_of_rectangle_l682_68291

theorem max_area_of_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 36) : (x * y) ≤ 81 :=
sorry

end max_area_of_rectangle_l682_68291


namespace integer_solutions_eq_0_or_2_l682_68284

theorem integer_solutions_eq_0_or_2 (a : ℤ) (x : ℤ) : 
  (a * x^2 + 6 = 0) → (a = -6 ∧ (x = 1 ∨ x = -1)) ∨ (¬ (a = -6) ∧ (x ≠ 1) ∧ (x ≠ -1)) :=
by 
sorry

end integer_solutions_eq_0_or_2_l682_68284


namespace chocolates_bought_l682_68263

theorem chocolates_bought (C S N : ℕ) (h1 : 4 * C = 7 * (S - C)) (h2 : N * C = 77 * S) :
  N = 121 :=
by
  sorry

end chocolates_bought_l682_68263


namespace find_tangent_perpendicular_t_l682_68203

noncomputable def y (x : ℝ) : ℝ := x * Real.log x

theorem find_tangent_perpendicular_t (t : ℝ) (ht : 0 < t) (h_perpendicular : (1 : ℝ) * (1 + Real.log t) = -1) :
  t = Real.exp (-2) :=
by
  sorry

end find_tangent_perpendicular_t_l682_68203


namespace probability_no_shaded_rectangle_l682_68218

-- Definitions
def total_rectangles_per_row : ℕ := (2005 * 2004) / 2
def shaded_rectangles_per_row : ℕ := 1002 * 1002

-- Proposition to prove
theorem probability_no_shaded_rectangle : 
  (1 - (shaded_rectangles_per_row : ℝ) / (total_rectangles_per_row : ℝ)) = (0.25 / 1002.25) := 
sorry

end probability_no_shaded_rectangle_l682_68218


namespace max_fridays_in_year_l682_68256

theorem max_fridays_in_year (days_in_common_year days_in_leap_year : ℕ) 
  (h_common_year : days_in_common_year = 365)
  (h_leap_year : days_in_leap_year = 366) : 
  ∃ (max_fridays : ℕ), max_fridays = 53 := 
by
  existsi 53
  sorry

end max_fridays_in_year_l682_68256


namespace vasya_petya_distance_l682_68299

theorem vasya_petya_distance :
  ∀ (D : ℝ), 
    (3 : ℝ) ≠ 0 → (6 : ℝ) ≠ 0 →
    ((D / 3) + (D / 6) = 2.5) →
    ((D / 6) + (D / 3) = 3.5) →
    D = 12 := 
by
  intros D h3 h6 h1 h2
  sorry

end vasya_petya_distance_l682_68299


namespace average_salary_of_technicians_l682_68208

theorem average_salary_of_technicians
  (total_workers : ℕ)
  (avg_salary_all_workers : ℕ)
  (total_technicians : ℕ)
  (avg_salary_non_technicians : ℕ)
  (h1 : total_workers = 18)
  (h2 : avg_salary_all_workers = 8000)
  (h3 : total_technicians = 6)
  (h4 : avg_salary_non_technicians = 6000) :
  (72000 / total_technicians) = 12000 := 
  sorry

end average_salary_of_technicians_l682_68208


namespace total_cards_beginning_l682_68250

-- Define the initial conditions
def num_boxes_orig : ℕ := 2 + 5  -- Robie originally had 2 + 5 boxes
def cards_per_box : ℕ := 10      -- Each box contains 10 cards
def extra_cards : ℕ := 5         -- 5 cards were not placed in a box

-- Prove the total number of cards Robie had in the beginning
theorem total_cards_beginning : (num_boxes_orig * cards_per_box) + extra_cards = 75 :=
by sorry

end total_cards_beginning_l682_68250


namespace range_of_a_for_positive_f_l682_68257

-- Let the function \(f(x) = ax^2 - 2x + 2\)
def f (a x : ℝ) := a * x^2 - 2 * x + 2

-- Theorem: The range of the real number \( a \) such that \( f(x) > 0 \) for all \( x \) in \( 1 < x < 4 \) is \((\dfrac{1}{2}, +\infty)\)
theorem range_of_a_for_positive_f :
  { a : ℝ | ∀ x : ℝ, 1 < x ∧ x < 4 → f a x > 0 } = { a : ℝ | a > 1/2 } :=
sorry

end range_of_a_for_positive_f_l682_68257


namespace sanya_towels_count_l682_68233

-- Defining the conditions based on the problem
def towels_per_hour := 7
def hours_per_day := 2
def days_needed := 7

-- The main statement to prove
theorem sanya_towels_count : 
  (towels_per_hour * hours_per_day * days_needed = 98) :=
by
  sorry

end sanya_towels_count_l682_68233


namespace cubic_expression_l682_68248

theorem cubic_expression (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + ac + bc = 50) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 1125 :=
sorry

end cubic_expression_l682_68248


namespace total_selling_price_l682_68214

theorem total_selling_price (cost1 cost2 cost3 : ℕ) (profit1 profit2 profit3 : ℚ) 
  (h1 : cost1 = 280) (h2 : cost2 = 350) (h3 : cost3 = 500) 
  (h4 : profit1 = 30) (h5 : profit2 = 45) (h6 : profit3 = 25) : 
  (cost1 + (profit1 / 100) * cost1) + (cost2 + (profit2 / 100) * cost2) + (cost3 + (profit3 / 100) * cost3) = 1496.5 := by
  sorry

end total_selling_price_l682_68214


namespace geom_seq_identity_l682_68217

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∀ n, ∃ r, a (n+1) = r * a n

theorem geom_seq_identity (a : ℕ → ℝ) (r : ℝ) (h1 : geometric_sequence a) (h2 : a 2 + a 4 = 2) :
  a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 4 := 
  sorry

end geom_seq_identity_l682_68217


namespace lisa_total_miles_flown_l682_68258

variable (distance_per_trip : ℝ := 256.0)
variable (number_of_trips : ℝ := 32.0)

theorem lisa_total_miles_flown : distance_per_trip * number_of_trips = 8192.0 := by
  sorry

end lisa_total_miles_flown_l682_68258


namespace average_age_of_cricket_team_l682_68265

theorem average_age_of_cricket_team 
  (num_members : ℕ)
  (avg_age : ℕ)
  (wicket_keeper_age : ℕ)
  (remaining_avg : ℕ)
  (cond1 : num_members = 11)
  (cond2 : avg_age = 29)
  (cond3 : wicket_keeper_age = avg_age + 3)
  (cond4 : remaining_avg = avg_age - 1) : 
  avg_age = 29 := 
by 
  have h1 : num_members = 11 := cond1
  have h2 : avg_age = 29 := cond2
  have h3 : wicket_keeper_age = avg_age + 3 := cond3
  have h4 : remaining_avg = avg_age - 1 := cond4
  -- proof steps will go here
  sorry

end average_age_of_cricket_team_l682_68265


namespace triangle_perpendicular_division_l682_68287

variable (a b c : ℝ)
variable (b_gt_c : b > c)
variable (triangle : True)

theorem triangle_perpendicular_division (a b c : ℝ) (b_gt_c : b > c) :
  let CK := (1 / 2) * Real.sqrt (a^2 + b^2 - c^2)
  CK = (1 / 2) * Real.sqrt (a^2 + b^2 - c^2) :=
by
  sorry

end triangle_perpendicular_division_l682_68287


namespace common_factor_l682_68264

-- Definition of the polynomial
def polynomial (x y m n : ℝ) : ℝ := 4 * x * (m - n) + 2 * y * (m - n) ^ 2

-- The theorem statement
theorem common_factor (x y m n : ℝ) : ∃ k : ℝ, k * (m - n) = polynomial x y m n :=
sorry

end common_factor_l682_68264


namespace triangle_altitude_angle_l682_68223

noncomputable def angle_between_altitudes (α : ℝ) : ℝ :=
if α ≤ 90 then α else 180 - α

theorem triangle_altitude_angle (α : ℝ) (hα : 0 < α ∧ α < 180) : 
  (angle_between_altitudes α = α ↔ α ≤ 90) ∧ (angle_between_altitudes α = 180 - α ↔ α > 90) := 
by
  sorry

end triangle_altitude_angle_l682_68223


namespace sin_2alpha_pos_of_tan_alpha_pos_l682_68234

theorem sin_2alpha_pos_of_tan_alpha_pos (α : Real) (h : Real.tan α > 0) : Real.sin (2 * α) > 0 :=
sorry

end sin_2alpha_pos_of_tan_alpha_pos_l682_68234


namespace domain_of_tan_l682_68211

open Real

noncomputable def function_domain : Set ℝ :=
  {x | ∀ k : ℤ, x ≠ k * π + 3 * π / 4}

theorem domain_of_tan : ∀ x : ℝ,
  (∃ k : ℤ, x = k * π + 3 * π / 4) → ¬ (∃ y : ℝ, y = tan (π / 4 - x)) :=
by
  intros x hx
  obtain ⟨k, hk⟩ := hx
  sorry

end domain_of_tan_l682_68211


namespace verify_segment_lengths_l682_68290

noncomputable def segment_lengths_proof : Prop :=
  let a := 2
  let b := 3
  let alpha := Real.arccos (5 / 16)
  let segment1 := 4 / 3
  let segment2 := 2 / 3
  let segment3 := 2
  let segment4 := 1
  ∀ (s1 s2 s3 s4 : ℝ), 
    (s1 = segment1 ∧ s2 = segment2 ∧ s3 = segment3 ∧ s4 = segment4) ↔
    -- Parallelogram sides and angle constraints
    (s1 + s2 = a ∧ s3 + s4 = b ∧ 
     -- Mutually perpendicular lines divide into equal areas
     (s1 * s3 * Real.sin alpha / 2 = s2 * s4 * Real.sin alpha / 2) )

-- Placeholder for proof
theorem verify_segment_lengths : segment_lengths_proof :=
  sorry

end verify_segment_lengths_l682_68290


namespace twice_plus_eight_lt_five_times_x_l682_68276

theorem twice_plus_eight_lt_five_times_x (x : ℝ) : 2 * x + 8 < 5 * x := 
sorry

end twice_plus_eight_lt_five_times_x_l682_68276
