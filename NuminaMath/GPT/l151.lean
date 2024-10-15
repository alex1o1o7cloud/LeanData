import Mathlib

namespace NUMINAMATH_GPT_trapezoid_segment_AB_length_l151_15108

/-
In the trapezoid shown, the ratio of the area of triangle ABC to the area of triangle ADC is 5:2.
If AB + CD = 240 cm, prove that the length of segment AB is 171.42857 cm.
-/

theorem trapezoid_segment_AB_length
  (AB CD : ℝ)
  (ratio_areas : ℝ := 5 / 2)
  (area_ratio_condition : AB / CD = ratio_areas)
  (length_sum_condition : AB + CD = 240) :
  AB = 171.42857 :=
sorry

end NUMINAMATH_GPT_trapezoid_segment_AB_length_l151_15108


namespace NUMINAMATH_GPT_total_faces_is_198_l151_15182

-- Definitions for the number of dice and geometrical shapes brought by each person:
def TomDice : ℕ := 4
def TimDice : ℕ := 5
def TaraDice : ℕ := 3
def TinaDice : ℕ := 2
def TonyCubes : ℕ := 1
def TonyTetrahedrons : ℕ := 3
def TonyIcosahedrons : ℕ := 2

-- Definitions for the number of faces for each type of dice or shape:
def SixSidedFaces : ℕ := 6
def EightSidedFaces : ℕ := 8
def TwelveSidedFaces : ℕ := 12
def TwentySidedFaces : ℕ := 20
def CubeFaces : ℕ := 6
def TetrahedronFaces : ℕ := 4
def IcosahedronFaces : ℕ := 20

-- We want to prove that the total number of faces is 198:
theorem total_faces_is_198 : 
  (TomDice * SixSidedFaces) + 
  (TimDice * EightSidedFaces) + 
  (TaraDice * TwelveSidedFaces) + 
  (TinaDice * TwentySidedFaces) + 
  (TonyCubes * CubeFaces) + 
  (TonyTetrahedrons * TetrahedronFaces) + 
  (TonyIcosahedrons * IcosahedronFaces) 
  = 198 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_faces_is_198_l151_15182


namespace NUMINAMATH_GPT_sum_of_three_distinct_l151_15152

def S : Set ℤ := {2, 5, 8, 11, 14, 17, 20}

theorem sum_of_three_distinct (S : Set ℤ) (h : S = {2, 5, 8, 11, 14, 17, 20}) :
  (∃ n : ℕ, n = 13 ∧ ∀ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    ∃ k : ℕ, a + b + c = 3 * k) := 
by  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_sum_of_three_distinct_l151_15152


namespace NUMINAMATH_GPT_skeleton_ratio_l151_15166

theorem skeleton_ratio (W M C : ℕ) 
  (h1 : W + M + C = 20)
  (h2 : M = C)
  (h3 : 20 * W + 25 * M + 10 * C = 375) :
  (W : ℚ) / (W + M + C) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_skeleton_ratio_l151_15166


namespace NUMINAMATH_GPT_range_of_m_l151_15130

noncomputable def p (x : ℝ) : Prop := abs (1 - (x - 1) / 3) ≤ 2
noncomputable def q (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) : (∀ x : ℝ, ¬p x → ¬q x m) → (m ≥ 9) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l151_15130


namespace NUMINAMATH_GPT_time_for_machine_A_l151_15185

theorem time_for_machine_A (x : ℝ) (T : ℝ) (A B : ℝ) :
  (B = 2 * x / 5) → 
  (A + B = x / 2) → 
  (A = x / T) → 
  T = 10 := 
by 
  intros hB hAB hA
  sorry

end NUMINAMATH_GPT_time_for_machine_A_l151_15185


namespace NUMINAMATH_GPT_polynomial_divisibility_l151_15123

theorem polynomial_divisibility (m : ℕ) (hm : 0 < m) :
  ∀ x : ℝ, x * (x + 1) * (2 * x + 1) ∣ (x + 1) ^ (2 * m) - x ^ (2 * m) - 2 * x - 1 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l151_15123


namespace NUMINAMATH_GPT_max_third_side_of_triangle_l151_15174

theorem max_third_side_of_triangle (a b : ℕ) (h₁ : a = 7) (h₂ : b = 11) : 
  ∃ c : ℕ, c < a + b ∧ c = 17 :=
by 
  sorry

end NUMINAMATH_GPT_max_third_side_of_triangle_l151_15174


namespace NUMINAMATH_GPT_problem_l151_15104

theorem problem (m n : ℕ) 
  (m_pos : 0 < m) 
  (n_pos : 0 < n) 
  (h1 : m + 8 < n) 
  (h2 : (m + (m + 3) + (m + 8) + n + (n + 3) + (2 * n - 1)) / 6 = n + 1) 
  (h3 : (m + 8 + n) / 2 = n + 1) : m + n = 16 :=
  sorry

end NUMINAMATH_GPT_problem_l151_15104


namespace NUMINAMATH_GPT_profit_loss_balance_l151_15143

-- Defining variables
variables (C L : Real)

-- Profit and loss equations according to problem conditions
theorem profit_loss_balance (h1 : 832 - C = C - L) (h2 : 992 = 0.55 * C) : 
  (C + 992 = 2795.64) :=
by
  -- Statement of the theorem
  sorry

end NUMINAMATH_GPT_profit_loss_balance_l151_15143


namespace NUMINAMATH_GPT_tumblonian_words_count_l151_15117

def numTumblonianWords : ℕ :=
  let alphabet_size := 6
  let max_word_length := 4
  let num_words n := alphabet_size ^ n
  (num_words 1) + (num_words 2) + (num_words 3) + (num_words 4)

theorem tumblonian_words_count : numTumblonianWords = 1554 := by
  sorry

end NUMINAMATH_GPT_tumblonian_words_count_l151_15117


namespace NUMINAMATH_GPT_sum_of_values_satisfying_l151_15157

theorem sum_of_values_satisfying (x : ℝ) (h : Real.sqrt ((x - 2) ^ 2) = 8) :
  ∃ x1 x2 : ℝ, (Real.sqrt ((x1 - 2) ^ 2) = 8) ∧ (Real.sqrt ((x2 - 2) ^ 2) = 8) ∧ x1 + x2 = 4 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_values_satisfying_l151_15157


namespace NUMINAMATH_GPT_two_pow_15000_mod_1250_l151_15178

theorem two_pow_15000_mod_1250 (h : 2 ^ 500 ≡ 1 [MOD 1250]) :
  2 ^ 15000 ≡ 1 [MOD 1250] :=
sorry

end NUMINAMATH_GPT_two_pow_15000_mod_1250_l151_15178


namespace NUMINAMATH_GPT_find_ab_l151_15150

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) :
  a * b = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l151_15150


namespace NUMINAMATH_GPT_octagon_diagonals_l151_15102

def total_lines (n : ℕ) : ℕ := n * (n - 1) / 2

theorem octagon_diagonals : total_lines 8 - 8 = 20 := 
by
  -- Calculate the total number of lines between any two points in an octagon
  have h1 : total_lines 8 = 28 := by sorry
  -- Subtract the number of sides of the octagon
  have h2 : 28 - 8 = 20 := by norm_num
  
  -- Combine results to conclude the theorem
  exact h2

end NUMINAMATH_GPT_octagon_diagonals_l151_15102


namespace NUMINAMATH_GPT_expression_for_M_value_of_M_when_x_eq_negative_2_and_y_eq_1_l151_15158

noncomputable def A (x y : ℝ) := x^2 - 3 * x * y - y^2
noncomputable def B (x y : ℝ) := x^2 - 3 * x * y - 3 * y^2
noncomputable def M (x y : ℝ) := 2 * A x y - B x y

theorem expression_for_M (x y : ℝ) : M x y = x^2 - 3 * x * y + y^2 := by
  sorry

theorem value_of_M_when_x_eq_negative_2_and_y_eq_1 :
  M (-2) 1 = 11 := by
  sorry

end NUMINAMATH_GPT_expression_for_M_value_of_M_when_x_eq_negative_2_and_y_eq_1_l151_15158


namespace NUMINAMATH_GPT_sum_of_ten_numbers_in_circle_l151_15161

theorem sum_of_ten_numbers_in_circle : 
  ∀ (a b c d e f g h i j : ℕ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i ∧ 0 < j ∧
  a = Nat.gcd b j + 1 ∧ b = Nat.gcd a c + 1 ∧ c = Nat.gcd b d + 1 ∧ d = Nat.gcd c e + 1 ∧ 
  e = Nat.gcd d f + 1 ∧ f = Nat.gcd e g + 1 ∧ g = Nat.gcd f h + 1 ∧ 
  h = Nat.gcd g i + 1 ∧ i = Nat.gcd h j + 1 ∧ j = Nat.gcd i a + 1 → 
  a + b + c + d + e + f + g + h + i + j = 28 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_ten_numbers_in_circle_l151_15161


namespace NUMINAMATH_GPT_y_percentage_of_8950_l151_15138

noncomputable def x := 0.18 * 4750
noncomputable def y := 1.30 * x
theorem y_percentage_of_8950 : (y / 8950) * 100 = 12.42 := 
by 
  -- proof steps are omitted
  sorry

end NUMINAMATH_GPT_y_percentage_of_8950_l151_15138


namespace NUMINAMATH_GPT_european_stamps_cost_l151_15169

def prices : String → ℕ 
| "Italy"   => 7
| "Japan"   => 7
| "Germany" => 5
| "China"   => 5
| _ => 0

def stamps_1950s : String → ℕ 
| "Italy"   => 5
| "Germany" => 8
| "China"   => 10
| "Japan"   => 6
| _ => 0

def stamps_1960s : String → ℕ 
| "Italy"   => 9
| "Germany" => 12
| "China"   => 5
| "Japan"   => 10
| _ => 0

def total_cost (stamps : String → ℕ) (price : String → ℕ) : ℕ :=
  (stamps "Italy" * price "Italy" +
   stamps "Germany" * price "Germany") 

theorem european_stamps_cost : total_cost stamps_1950s prices + total_cost stamps_1960s prices = 198 :=
by
  sorry

end NUMINAMATH_GPT_european_stamps_cost_l151_15169


namespace NUMINAMATH_GPT_evaluate_expression_l151_15160

theorem evaluate_expression (A B : ℝ) (hA : A = 2^7) (hB : B = 3^6) : (A ^ (1 / 3)) * (B ^ (1 / 2)) = 108 * 2 ^ (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l151_15160


namespace NUMINAMATH_GPT_total_kids_played_tag_with_l151_15183

theorem total_kids_played_tag_with : 
  let kids_mon : Nat := 12
  let kids_tues : Nat := 7
  let kids_wed : Nat := 15
  let kids_thurs : Nat := 10
  let kids_fri : Nat := 18
  (kids_mon + kids_tues + kids_wed + kids_thurs + kids_fri) = 62 := by
  sorry

end NUMINAMATH_GPT_total_kids_played_tag_with_l151_15183


namespace NUMINAMATH_GPT_average_rate_of_change_interval_l151_15135

noncomputable def average_rate_of_change (f : ℝ → ℝ) (x₀ x₁ : ℝ) : ℝ :=
  (f x₁ - f x₀) / (x₁ - x₀)

theorem average_rate_of_change_interval (f : ℝ → ℝ) (x₀ x₁ : ℝ) :
  (f x₁ - f x₀) / (x₁ - x₀) = average_rate_of_change f x₀ x₁ := by
  sorry

end NUMINAMATH_GPT_average_rate_of_change_interval_l151_15135


namespace NUMINAMATH_GPT_bread_carriers_l151_15188

-- Definitions for the number of men, women, and children
variables (m w c : ℕ)

-- Conditions from the problem
def total_people := m + w + c = 12
def total_bread := 8 * m + 2 * w + c = 48

-- Theorem to prove the correct number of men, women, and children
theorem bread_carriers (h1 : total_people m w c) (h2 : total_bread m w c) : 
  m = 5 ∧ w = 1 ∧ c = 6 :=
sorry

end NUMINAMATH_GPT_bread_carriers_l151_15188


namespace NUMINAMATH_GPT_must_be_true_if_not_all_electric_l151_15175

variable (P : Type) (ElectricCar : P → Prop)

theorem must_be_true_if_not_all_electric (h : ¬ ∀ x : P, ElectricCar x) : 
  ∃ x : P, ¬ ElectricCar x :=
by 
sorry

end NUMINAMATH_GPT_must_be_true_if_not_all_electric_l151_15175


namespace NUMINAMATH_GPT_derivative_at_pi_over_4_l151_15134

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem derivative_at_pi_over_4 : (deriv f (Real.pi / 4)) = -2 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_pi_over_4_l151_15134


namespace NUMINAMATH_GPT_lori_beanie_babies_times_l151_15173

theorem lori_beanie_babies_times (l s : ℕ) (h1 : l = 300) (h2 : l + s = 320) : l = 15 * s :=
by
  sorry

end NUMINAMATH_GPT_lori_beanie_babies_times_l151_15173


namespace NUMINAMATH_GPT_max_consecutive_irreducible_l151_15167

-- Define what it means for a five-digit number to be irreducible
def is_irreducible (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ ¬∃ x y : ℕ, 100 ≤ x ∧ x < 1000 ∧ 100 ≤ y ∧ y < 1000 ∧ x * y = n

-- Prove the maximum number of consecutive irreducible five-digit numbers is 99
theorem max_consecutive_irreducible : ∃ m : ℕ, m = 99 ∧ 
  (∀ n : ℕ, (n ≤ 99901) → (∀ k : ℕ, (n ≤ k ∧ k < n + m) → is_irreducible k)) ∧
  (∀ x y : ℕ, x > 99 → ∀ n : ℕ, (n ≤ 99899) → (∀ k : ℕ, (n ≤ k ∧ k < n + x) → is_irreducible k) → x = 99) :=
by
  sorry

end NUMINAMATH_GPT_max_consecutive_irreducible_l151_15167


namespace NUMINAMATH_GPT_number_of_orange_ribbons_l151_15101

/-- Define the total number of ribbons -/
def total_ribbons (yellow purple orange black total : ℕ) : Prop :=
  yellow + purple + orange + black = total

/-- Define the fractions -/
def fractions (total_ribbons yellow purple orange black : ℕ) : Prop :=
  yellow = total_ribbons / 4 ∧ purple = total_ribbons / 3 ∧ orange = total_ribbons / 12 ∧ black = 40

/-- Define the black ribbons fraction -/
def black_fraction (total_ribbons : ℕ) : Prop :=
  40 = total_ribbons / 3

theorem number_of_orange_ribbons :
  ∃ (total : ℕ), total_ribbons (total / 4) (total / 3) (total / 12) 40 total ∧ black_fraction total ∧ (total / 12 = 10) :=
by
  sorry

end NUMINAMATH_GPT_number_of_orange_ribbons_l151_15101


namespace NUMINAMATH_GPT_proof_problem_l151_15131

noncomputable def problem_equivalent_proof (x y z : ℝ) : Prop :=
  (x + y + z = 3) ∧
  (z + 6 = 2 * y - z) ∧
  (x + 8 * z = y + 2) →
  (x^2 + y^2 + z^2 = 21)

theorem proof_problem (x y z : ℝ) : problem_equivalent_proof x y z :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l151_15131


namespace NUMINAMATH_GPT_female_managers_count_l151_15176

variable (E M F FM : ℕ)

-- Conditions
def female_employees : Prop := F = 750
def fraction_managers : Prop := (2 / 5 : ℚ) * E = FM + (2 / 5 : ℚ) * M
def total_employees : Prop := E = M + F

-- Proof goal
theorem female_managers_count (h1 : female_employees F) 
                              (h2 : fraction_managers E M FM) 
                              (h3 : total_employees E M F) : 
  FM = 300 := 
sorry

end NUMINAMATH_GPT_female_managers_count_l151_15176


namespace NUMINAMATH_GPT_opposite_of_neg_two_is_two_l151_15164

theorem opposite_of_neg_two_is_two (x : ℤ) (h : -2 + x = 0) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_two_is_two_l151_15164


namespace NUMINAMATH_GPT_problem1_problem2_l151_15103

theorem problem1 : (1 * (-9)) - (-7) + (-6) - 5 = -13 := 
by 
  -- problem1 proof
  sorry

theorem problem2 : ((-5 / 12) + (2 / 3) - (3 / 4)) * (-12) = 6 := 
by 
  -- problem2 proof
  sorry

end NUMINAMATH_GPT_problem1_problem2_l151_15103


namespace NUMINAMATH_GPT_michelle_total_payment_l151_15172
noncomputable def michelle_base_cost := 25
noncomputable def included_talk_time := 40 -- in hours
noncomputable def text_cost := 10 -- in cents per message
noncomputable def extra_talk_cost := 15 -- in cents per minute
noncomputable def february_texts_sent := 200
noncomputable def february_talk_time := 41 -- in hours

theorem michelle_total_payment : 
  25 + ((200 * 10) / 100) + (((41 - 40) * 60 * 15) / 100) = 54 := by
  sorry

end NUMINAMATH_GPT_michelle_total_payment_l151_15172


namespace NUMINAMATH_GPT_quadratic_function_expression_l151_15192

theorem quadratic_function_expression : 
  ∃ (a : ℝ), (a ≠ 0) ∧ (∀ x : ℝ, x = -1 → x * (a * (x + 1) * (x - 2)) = 0) ∧
  (∀ x : ℝ, x = 2 → x * (a * (x + 1) * (x - 2)) = 0) ∧
  (∀ y : ℝ, ∃ x : ℝ, x = 0 ∧ y = -2 → y = a * (x + 1) * (x - 2)) 
  → (∀ x : ℝ, ∃ y : ℝ, y = x^2 - x - 2) := 
sorry

end NUMINAMATH_GPT_quadratic_function_expression_l151_15192


namespace NUMINAMATH_GPT_quadratic_general_form_l151_15170

theorem quadratic_general_form (x : ℝ) :
  x * (x + 2) = 5 * (x - 2) → x^2 - 3 * x - 10 = 0 := by
  sorry

end NUMINAMATH_GPT_quadratic_general_form_l151_15170


namespace NUMINAMATH_GPT_find_cost_price_of_article_l151_15125

theorem find_cost_price_of_article 
  (C : ℝ) 
  (h1 : 1.05 * C - 2 = 1.045 * C) 
  (h2 : 0.005 * C = 2) 
: C = 400 := 
by 
  sorry

end NUMINAMATH_GPT_find_cost_price_of_article_l151_15125


namespace NUMINAMATH_GPT_unshaded_area_eq_20_l151_15146

-- Define the dimensions of the first rectangle
def rect1_width := 4
def rect1_length := 12

-- Define the dimensions of the second rectangle
def rect2_width := 5
def rect2_length := 10

-- Define the dimensions of the overlapping region
def overlap_width := 4
def overlap_length := 5

-- Calculate area functions
def area (width length : ℕ) := width * length

-- Calculate areas of the individual rectangles and the overlapping region
def area_rect1 := area rect1_width rect1_length
def area_rect2 := area rect2_width rect2_length
def overlap_area := area overlap_width overlap_length

-- Calculate the total shaded area
def total_shaded_area := area_rect1 + area_rect2 - overlap_area

-- The total area of the combined figure (assumed to be the union of both rectangles) minus shaded area gives the unshaded area
def total_area := rect1_width * rect1_length + rect2_width * rect2_length
def unshaded_area := total_area - total_shaded_area

theorem unshaded_area_eq_20 : unshaded_area = 20 := by
  sorry

end NUMINAMATH_GPT_unshaded_area_eq_20_l151_15146


namespace NUMINAMATH_GPT_kinetic_energy_reduction_collisions_l151_15148

theorem kinetic_energy_reduction_collisions (E_0 : ℝ) (n : ℕ) :
  (1 / 2)^n * E_0 = E_0 / 64 → n = 6 :=
by
  sorry

end NUMINAMATH_GPT_kinetic_energy_reduction_collisions_l151_15148


namespace NUMINAMATH_GPT_count_integers_between_bounds_l151_15184

theorem count_integers_between_bounds : 
  ∃ n : ℤ, n = 15 ∧ ∀ x : ℤ, 3 < Real.sqrt (x : ℝ) ∧ Real.sqrt (x : ℝ) < 5 → 10 ≤ x ∧ x ≤ 24 :=
by
  sorry

end NUMINAMATH_GPT_count_integers_between_bounds_l151_15184


namespace NUMINAMATH_GPT_negation_p_l151_15124

open Nat

def p : Prop := ∀ n : ℕ, n^2 ≤ 2^n

theorem negation_p : ¬p ↔ ∃ n : ℕ, n^2 > 2^n :=
by
  sorry

end NUMINAMATH_GPT_negation_p_l151_15124


namespace NUMINAMATH_GPT_incorrect_statement_c_l151_15110

open Real

theorem incorrect_statement_c (p q: ℝ) : ¬(∀ x: ℝ, (x * abs x + p * x + q = 0 ↔ p^2 - 4 * q ≥ 0)) :=
sorry

end NUMINAMATH_GPT_incorrect_statement_c_l151_15110


namespace NUMINAMATH_GPT_three_digit_number_divisibility_four_digit_number_divisibility_l151_15193

-- Definition of three-digit number
def is_three_digit_number (a : ℕ) : Prop := 100 ≤ a ∧ a ≤ 999

-- Definition of four-digit number
def is_four_digit_number (b : ℕ) : Prop := 1000 ≤ b ∧ b ≤ 9999

-- First proof problem
theorem three_digit_number_divisibility (a : ℕ) (h : is_three_digit_number a) : 
  (1001 * a) % 7 = 0 ∧ (1001 * a) % 11 = 0 ∧ (1001 * a) % 13 = 0 := 
sorry

-- Second proof problem
theorem four_digit_number_divisibility (b : ℕ) (h : is_four_digit_number b) : 
  (10001 * b) % 73 = 0 ∧ (10001 * b) % 137 = 0 := 
sorry

end NUMINAMATH_GPT_three_digit_number_divisibility_four_digit_number_divisibility_l151_15193


namespace NUMINAMATH_GPT_locus_centers_tangent_circles_l151_15147

theorem locus_centers_tangent_circles (a b : ℝ) :
  (∃ r : ℝ, a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (3 - r)^2) →
  a^2 - 12 * a + 4 * b^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_locus_centers_tangent_circles_l151_15147


namespace NUMINAMATH_GPT_Onum_Lake_more_trout_l151_15149

theorem Onum_Lake_more_trout (O B R : ℕ) (hB : B = 75) (hR : R = O / 2) (hAvg : (O + B + R) / 3 = 75) : O - B = 25 :=
by
  sorry

end NUMINAMATH_GPT_Onum_Lake_more_trout_l151_15149


namespace NUMINAMATH_GPT_jacob_age_in_X_years_l151_15151

-- Definitions of the conditions
variable (J M X : ℕ)

theorem jacob_age_in_X_years
  (h1 : J = M - 14)
  (h2 : M + 9 = 2 * (J + 9))
  (h3 : J = 5) :
  J + X = 5 + X :=
by
  sorry

end NUMINAMATH_GPT_jacob_age_in_X_years_l151_15151


namespace NUMINAMATH_GPT_part1_part2_l151_15114

def f (a x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + a + 1

-- Proof problem 1: Prove that if a = 2, then f(x) ≥ 0 is equivalent to x ≥ 3/2 or x ≤ 1.
theorem part1 (x : ℝ) : f 2 x ≥ 0 ↔ x ≥ (3 / 2 : ℝ) ∨ x ≤ 1 := sorry

-- Proof problem 2: Prove that for a∈[-2,2], if f(x) < 0 always holds, then x ∈ (1, 3/2).
theorem part2 (a x : ℝ) (ha : a ≥ -2 ∧ a ≤ 2) : (∀ x, f a x < 0) ↔ 1 < x ∧ x < (3 / 2 : ℝ) := sorry

end NUMINAMATH_GPT_part1_part2_l151_15114


namespace NUMINAMATH_GPT_simplify_expression_l151_15120

theorem simplify_expression (x y : ℝ) (hxy : x ≠ y) : 
  ((x - y) ^ 3 / (x - y) ^ 2) * (y - x) = -(x - y) ^ 2 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l151_15120


namespace NUMINAMATH_GPT_compare_negatives_l151_15115

theorem compare_negatives : -1 > -2 := 
by 
  sorry

end NUMINAMATH_GPT_compare_negatives_l151_15115


namespace NUMINAMATH_GPT_ratio_of_vegetables_to_beef_l151_15198

variable (amountBeefInitial : ℕ) (amountBeefUnused : ℕ) (amountVegetables : ℕ)

def amount_beef_used (initial unused : ℕ) : ℕ := initial - unused
def ratio_vegetables_beef (vegetables beef : ℕ) : ℚ := vegetables / beef

theorem ratio_of_vegetables_to_beef 
  (h1 : amountBeefInitial = 4)
  (h2 : amountBeefUnused = 1)
  (h3 : amountVegetables = 6) :
  ratio_vegetables_beef amountVegetables (amount_beef_used amountBeefInitial amountBeefUnused) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_vegetables_to_beef_l151_15198


namespace NUMINAMATH_GPT_unique_friendly_determination_l151_15194

def is_friendly (a b : ℕ → ℕ) : Prop :=
∀ n : ℕ, ∃ i j : ℕ, n = a i * b j ∧ ∀ (k l : ℕ), n = a k * b l → (i = k ∧ j = l)

theorem unique_friendly_determination {a b c : ℕ → ℕ} 
  (h_friend_a_b : is_friendly a b) 
  (h_friend_a_c : is_friendly a c) :
  b = c :=
sorry

end NUMINAMATH_GPT_unique_friendly_determination_l151_15194


namespace NUMINAMATH_GPT_compare_abc_l151_15163

variable (a b c : ℝ)

noncomputable def define_a : ℝ := (2/3)^(1/3)
noncomputable def define_b : ℝ := (2/3)^(1/2)
noncomputable def define_c : ℝ := (3/5)^(1/2)

theorem compare_abc (h₁ : a = define_a) (h₂ : b = define_b) (h₃ : c = define_c) :
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_compare_abc_l151_15163


namespace NUMINAMATH_GPT_coeff_div_binom_eq_4_l151_15118

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def coeff_x5_expansion : ℚ :=
  binomial 8 2 * (-2) ^ 2

def binomial_coeff : ℚ :=
  binomial 8 2

theorem coeff_div_binom_eq_4 : 
  (coeff_x5_expansion / binomial_coeff) = 4 := by
  sorry

end NUMINAMATH_GPT_coeff_div_binom_eq_4_l151_15118


namespace NUMINAMATH_GPT_find_shop_width_l151_15128

def shop_width (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_square_foot : ℕ) : ℕ :=
  let annual_rent := monthly_rent * 12
  let total_area := annual_rent / annual_rent_per_square_foot
  total_area / length

theorem find_shop_width :
  shop_width 3600 20 144 = 15 :=
by 
  -- Here would go the proof, but we add sorry to skip it
  sorry

end NUMINAMATH_GPT_find_shop_width_l151_15128


namespace NUMINAMATH_GPT_find_angle_D_l151_15199

theorem find_angle_D (A B C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = 2 * D)
  (h3 : A + B + C + D = 360) : D = 60 :=
sorry

end NUMINAMATH_GPT_find_angle_D_l151_15199


namespace NUMINAMATH_GPT_aisha_additional_miles_l151_15191

theorem aisha_additional_miles
  (D : ℕ) (d : ℕ) (v1 : ℕ) (v2 : ℕ) (v_avg : ℕ)
  (h1 : D = 18) (h2 : v1 = 36) (h3 : v2 = 60) (h4 : v_avg = 48)
  (h5 : d = 30) :
  (D + d) / ((D / v1) + (d / v2)) = v_avg :=
  sorry

end NUMINAMATH_GPT_aisha_additional_miles_l151_15191


namespace NUMINAMATH_GPT_students_suggesting_bacon_l151_15109

theorem students_suggesting_bacon (S : ℕ) (M : ℕ) (h1: S = 310) (h2: M = 185) : S - M = 125 := 
by
  -- proof here
  sorry

end NUMINAMATH_GPT_students_suggesting_bacon_l151_15109


namespace NUMINAMATH_GPT_smallest_integer_no_inverse_mod_77_66_l151_15137

theorem smallest_integer_no_inverse_mod_77_66 :
  ∃ a : ℕ, 0 < a ∧ a = 11 ∧ gcd a 77 > 1 ∧ gcd a 66 > 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_no_inverse_mod_77_66_l151_15137


namespace NUMINAMATH_GPT_expression_equals_6_l151_15165

-- Define the expression as a Lean definition.
def expression : ℤ := 1 - (-2) - 3 - (-4) - 5 - (-6) - 7 - (-8)

-- The statement to prove that the expression equals 6.
theorem expression_equals_6 : expression = 6 := 
by 
  -- This is a placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_expression_equals_6_l151_15165


namespace NUMINAMATH_GPT_compute_expression_l151_15127

theorem compute_expression (x y : ℝ) (hx : 1/x + 1/y = 4) (hy : x*y + x + y = 5) : 
  x^2 * y + x * y^2 + x^2 + y^2 = 18 := 
by 
  -- Proof goes here 
  sorry

end NUMINAMATH_GPT_compute_expression_l151_15127


namespace NUMINAMATH_GPT_total_stars_l151_15105

theorem total_stars (students stars_per_student : ℕ) (h_students : students = 124) (h_stars_per_student : stars_per_student = 3) : students * stars_per_student = 372 := by
  sorry

end NUMINAMATH_GPT_total_stars_l151_15105


namespace NUMINAMATH_GPT_expand_polynomial_l151_15154

noncomputable def polynomial_expansion : Prop :=
  ∀ (x : ℤ), (x + 3) * (x^2 + 4 * x + 6) = x^3 + 7 * x^2 + 18 * x + 18

theorem expand_polynomial : polynomial_expansion :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l151_15154


namespace NUMINAMATH_GPT_quadratic_axis_of_symmetry_is_one_l151_15155

noncomputable def quadratic_axis_of_symmetry (b c : ℝ) : ℝ :=
  (-b / (2 * 1))

theorem quadratic_axis_of_symmetry_is_one
  (b c : ℝ)
  (hA : (0:ℝ)^2 + b * 0 + c = 3)
  (hB : (2:ℝ)^2 + b * 2 + c = 3) :
  quadratic_axis_of_symmetry b c = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_axis_of_symmetry_is_one_l151_15155


namespace NUMINAMATH_GPT_P_sufficient_but_not_necessary_for_Q_l151_15162

def P (x : ℝ) : Prop := (x - 3) * (x + 1) > 0
def Q (x : ℝ) : Prop := x^2 - 2 * x + 1 > 0

theorem P_sufficient_but_not_necessary_for_Q : 
  (∀ x : ℝ, P x → Q x) ∧ ¬ (∀ x : ℝ, Q x → P x) :=
by 
  sorry

end NUMINAMATH_GPT_P_sufficient_but_not_necessary_for_Q_l151_15162


namespace NUMINAMATH_GPT_total_reading_materials_l151_15119

theorem total_reading_materials (magazines newspapers : ℕ) (h1 : magazines = 425) (h2 : newspapers = 275) : 
  magazines + newspapers = 700 :=
by 
  sorry

end NUMINAMATH_GPT_total_reading_materials_l151_15119


namespace NUMINAMATH_GPT_correct_calculation_B_l151_15126

theorem correct_calculation_B :
  (∀ (a : ℕ), 3 * a^3 * 2 * a^2 ≠ 6 * a^6) ∧
  (∀ (x : ℕ), 2 * x^2 * 3 * x^2 = 6 * x^4) ∧
  (∀ (x : ℕ), 3 * x^2 * 4 * x^2 ≠ 12 * x^2) ∧
  (∀ (y : ℕ), 5 * y^3 * 3 * y^5 ≠ 8 * y^8) →
  (∀ (x : ℕ), 2 * x^2 * 3 * x^2 = 6 * x^4) := 
by
  sorry

end NUMINAMATH_GPT_correct_calculation_B_l151_15126


namespace NUMINAMATH_GPT_cost_of_seven_CDs_l151_15186

theorem cost_of_seven_CDs (cost_per_two : ℝ) (h1 : cost_per_two = 32) : (7 * (cost_per_two / 2)) = 112 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_seven_CDs_l151_15186


namespace NUMINAMATH_GPT_triangle_properties_l151_15177

theorem triangle_properties
  (a b : ℝ)
  (C : ℝ)
  (ha : a = 2)
  (hb : b = 3)
  (hC : C = Real.pi / 3)
  :
  let c := Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C)
  let area := (1 / 2) * a * b * Real.sin C
  let sin2A := 2 * (a * Real.sin C / c) * Real.sqrt (1 - (a * Real.sin C / c)^2)
  c = Real.sqrt 7 
  ∧ area = (3 * Real.sqrt 3) / 2 
  ∧ sin2A = (4 * Real.sqrt 3) / 7 :=
by
  sorry

end NUMINAMATH_GPT_triangle_properties_l151_15177


namespace NUMINAMATH_GPT_expression_value_at_2_l151_15187

theorem expression_value_at_2 : (2^2 - 3 * 2 + 2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_at_2_l151_15187


namespace NUMINAMATH_GPT_washing_machine_heavy_wash_usage_l151_15179

-- Definition of variables and constants
variables (H : ℕ)                           -- Amount of water used for a heavy wash
def regular_wash : ℕ := 10                   -- Gallons used for a regular wash
def light_wash : ℕ := 2                      -- Gallons used for a light wash
def extra_light_wash : ℕ := light_wash       -- Extra light wash due to bleach

-- Number of each type of wash
def num_heavy_washes : ℕ := 2
def num_regular_washes : ℕ := 3
def num_light_washes : ℕ := 1
def num_bleached_washes : ℕ := 2

-- Total water usage
def total_water_usage : ℕ := 
  num_heavy_washes * H + 
  num_regular_washes * regular_wash + 
  num_light_washes * light_wash + 
  num_bleached_washes * extra_light_wash

-- Given total water usage
def given_total_water_usage : ℕ := 76

-- Lean statement to prove the amount of water used for a heavy wash
theorem washing_machine_heavy_wash_usage : total_water_usage H = given_total_water_usage → H = 20 :=
by
  sorry

end NUMINAMATH_GPT_washing_machine_heavy_wash_usage_l151_15179


namespace NUMINAMATH_GPT_average_of_first_6_numbers_l151_15197

-- Definitions extracted from conditions
def average_of_11_numbers := 60
def average_of_last_6_numbers := 65
def sixth_number := 258
def total_sum := 11 * average_of_11_numbers
def sum_of_last_6_numbers := 6 * average_of_last_6_numbers

-- Lean 4 statement for the proof problem
theorem average_of_first_6_numbers :
  (∃ A, 6 * A = (total_sum - (sum_of_last_6_numbers - sixth_number))) →
  (∃ A, 6 * A = 528) :=
by
  intro h
  exact h

end NUMINAMATH_GPT_average_of_first_6_numbers_l151_15197


namespace NUMINAMATH_GPT_investment_C_120000_l151_15190

noncomputable def investment_C (P_B P_A_difference : ℕ) (investment_A investment_B : ℕ) : ℕ :=
  let P_A := (P_B * investment_A) / investment_B
  let P_C := P_A + P_A_difference
  (P_C * investment_B) / P_B

theorem investment_C_120000
  (investment_A investment_B P_B P_A_difference : ℕ)
  (hA : investment_A = 8000)
  (hB : investment_B = 10000)
  (hPB : P_B = 1400)
  (hPA_difference : P_A_difference = 560) :
  investment_C P_B P_A_difference investment_A investment_B = 120000 :=
by
  sorry

end NUMINAMATH_GPT_investment_C_120000_l151_15190


namespace NUMINAMATH_GPT_jessica_current_age_l151_15129

-- Definitions and conditions from the problem
def J (M : ℕ) : ℕ := M / 2
def M : ℕ := 60

-- Lean statement for the proof problem
theorem jessica_current_age : J M + 10 = 40 :=
by
  sorry

end NUMINAMATH_GPT_jessica_current_age_l151_15129


namespace NUMINAMATH_GPT_exists_natural_numbers_with_digit_sum_condition_l151_15139

def digit_sum (x : ℕ) : ℕ :=
  x.digits 10 |>.sum

theorem exists_natural_numbers_with_digit_sum_condition :
  ∃ (a b c : ℕ), digit_sum (a + b) < 5 ∧ digit_sum (a + c) < 5 ∧ digit_sum (b + c) < 5 ∧ digit_sum (a + b + c) > 50 :=
by
  sorry

end NUMINAMATH_GPT_exists_natural_numbers_with_digit_sum_condition_l151_15139


namespace NUMINAMATH_GPT_probability_six_distinct_numbers_l151_15189

theorem probability_six_distinct_numbers :
  let total_outcomes := 6^6
  let distinct_outcomes := Nat.factorial 6
  let probability := (distinct_outcomes:ℚ) / (total_outcomes:ℚ)
  probability = 5 / 324 :=
sorry

end NUMINAMATH_GPT_probability_six_distinct_numbers_l151_15189


namespace NUMINAMATH_GPT_smallest_integer_sum_of_squares_and_cubes_infinite_integers_sum_of_squares_and_cubes_l151_15140

theorem smallest_integer_sum_of_squares_and_cubes :
  ∃ (n : ℕ) (a b c d : ℕ), n > 2 ∧ n = a^2 + b^2 ∧ n = c^3 + d^3 ∧
  ∀ (m : ℕ) (x y u v : ℕ), (m > 2 ∧ m = x^2 + y^2 ∧ m = u^3 + v^3) → n ≤ m := 
sorry

theorem infinite_integers_sum_of_squares_and_cubes :
  ∀ (k : ℕ), ∃ (n : ℕ) (a b c d : ℕ), n = 1 + 2^(6*k) ∧ n = a^2 + b^2 ∧ n = c^3 + d^3 :=
sorry

end NUMINAMATH_GPT_smallest_integer_sum_of_squares_and_cubes_infinite_integers_sum_of_squares_and_cubes_l151_15140


namespace NUMINAMATH_GPT_product_of_B_coords_l151_15112

structure Point where
  x : ℝ
  y : ℝ

def isMidpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

theorem product_of_B_coords :
  ∀ (M A B : Point), 
  isMidpoint M A B →
  M = ⟨3, 7⟩ →
  A = ⟨5, 3⟩ →
  (B.x * B.y) = 11 :=
by intro M A B hM hM_def hA_def; sorry

end NUMINAMATH_GPT_product_of_B_coords_l151_15112


namespace NUMINAMATH_GPT_sally_cards_final_count_l151_15171

def initial_cards : ℕ := 27
def cards_from_Dan : ℕ := 41
def cards_bought : ℕ := 20
def cards_traded : ℕ := 15
def cards_lost : ℕ := 7

def final_cards (initial : ℕ) (from_Dan : ℕ) (bought : ℕ) (traded : ℕ) (lost : ℕ) : ℕ :=
  initial + from_Dan + bought - traded - lost

theorem sally_cards_final_count :
  final_cards initial_cards cards_from_Dan cards_bought cards_traded cards_lost = 66 := by
  sorry

end NUMINAMATH_GPT_sally_cards_final_count_l151_15171


namespace NUMINAMATH_GPT_largest_sum_of_digits_in_display_l151_15159

-- Define the conditions
def is_valid_hour (h : Nat) : Prop := 0 <= h ∧ h < 24
def is_valid_minute (m : Nat) : Prop := 0 <= m ∧ m < 60

-- Define helper functions to convert numbers to their digit sums
def digit_sum (n : Nat) : Nat :=
  n.digits 10 |>.sum

-- Define the largest possible sum of the digits condition
def largest_possible_digit_sum : Prop :=
  ∀ (h m : Nat), is_valid_hour h → is_valid_minute m → 
    digit_sum (h.div 10 + h % 10) + digit_sum (m.div 10 + m % 10) ≤ 24 ∧
    ∃ (h m : Nat), is_valid_hour h ∧ is_valid_minute m ∧ digit_sum (h.div 10 + h % 10) + digit_sum (m.div 10 + m % 10) = 24

-- The statement to prove
theorem largest_sum_of_digits_in_display : largest_possible_digit_sum :=
by
  sorry

end NUMINAMATH_GPT_largest_sum_of_digits_in_display_l151_15159


namespace NUMINAMATH_GPT_part1_part2_l151_15195

open Real

noncomputable def curve_parametric (α : ℝ) : ℝ × ℝ :=
  (2 + sqrt 10 * cos α, sqrt 10 * sin α)

noncomputable def curve_polar (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * cos θ - 6 = 0

noncomputable def line_polar (ρ θ : ℝ) : Prop :=
  ρ * cos θ + 2 * ρ * sin θ - 12 = 0

theorem part1 (α : ℝ) : ∃ ρ θ : ℝ, curve_polar ρ θ :=
  sorry

theorem part2 : ∃ ρ1 ρ2 : ℝ, curve_polar ρ1 (π / 4) ∧ line_polar ρ2 (π / 4) ∧ abs (ρ1 - ρ2) = sqrt 2 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l151_15195


namespace NUMINAMATH_GPT_a_and_b_together_time_eq_4_over_3_l151_15100

noncomputable def work_together_time (a b c h : ℝ) :=
  (1 / a) + (1 / b) + (1 / c) = (1 / (a - 6)) ∧
  (1 / a) + (1 / b) = 1 / h ∧
  (1 / (a - 6)) = (1 / (b - 1)) ∧
  (1 / (a - 6)) = 2 / c

theorem a_and_b_together_time_eq_4_over_3 (a b c h : ℝ) (h_wt : work_together_time a b c h) : 
  h = 4 / 3 :=
  sorry

end NUMINAMATH_GPT_a_and_b_together_time_eq_4_over_3_l151_15100


namespace NUMINAMATH_GPT_annual_growth_rate_proof_l151_15106

-- Lean 4 statement for the given problem
theorem annual_growth_rate_proof (profit_2021 : ℝ) (profit_2023 : ℝ) (r : ℝ)
  (h1 : profit_2021 = 3000)
  (h2 : profit_2023 = 4320)
  (h3 : profit_2023 = profit_2021 * (1 + r) ^ 2) :
  r = 0.2 :=
by sorry

end NUMINAMATH_GPT_annual_growth_rate_proof_l151_15106


namespace NUMINAMATH_GPT_binomial_expansion_example_l151_15145

theorem binomial_expansion_example : 7^3 + 3 * (7^2) * 2 + 3 * 7 * (2^2) + 2^3 = 729 := by
  sorry

end NUMINAMATH_GPT_binomial_expansion_example_l151_15145


namespace NUMINAMATH_GPT_exponent_evaluation_l151_15136

theorem exponent_evaluation {a b : ℕ} (h₁ : 2 ^ a ∣ 200) (h₂ : ¬ (2 ^ (a + 1) ∣ 200))
                           (h₃ : 5 ^ b ∣ 200) (h₄ : ¬ (5 ^ (b + 1) ∣ 200)) :
  (1 / 3) ^ (b - a) = 3 :=
by sorry

end NUMINAMATH_GPT_exponent_evaluation_l151_15136


namespace NUMINAMATH_GPT_range_of_a_l151_15111

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + a * x + a + 3 > 0) : 0 ≤ a := sorry

end NUMINAMATH_GPT_range_of_a_l151_15111


namespace NUMINAMATH_GPT_right_triangle_ineq_l151_15144

variable (a b c : ℝ)
variable (h : c^2 = a^2 + b^2)

theorem right_triangle_ineq (a b c : ℝ) (h : c^2 = a^2 + b^2) : (a^3 + b^3 + c^3) / (a * b * (a + b + c)) ≥ Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_ineq_l151_15144


namespace NUMINAMATH_GPT_cubic_sum_l151_15141

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by {
  sorry
}

end NUMINAMATH_GPT_cubic_sum_l151_15141


namespace NUMINAMATH_GPT_husband_age_l151_15180

theorem husband_age (a b : ℕ) (w_age h_age : ℕ) (ha : a > 0) (hb : b > 0) 
  (hw_age : w_age = 10 * a + b) 
  (hh_age : h_age = 10 * b + a) 
  (h_older : h_age > w_age)
  (h_difference : 9 * (b - a) = a + b) :
  h_age = 54 :=
by
  sorry

end NUMINAMATH_GPT_husband_age_l151_15180


namespace NUMINAMATH_GPT_problem1_l151_15133

/-- Problem 1: Given the formula \( S = vt + \frac{1}{2}at^2 \) and the conditions
  when \( t=1, S=4 \) and \( t=2, S=10 \), prove that when \( t=3 \), \( S=18 \). -/
theorem problem1 (v a t S: ℝ) 
  (h₁ : t = 1 → S = 4 → S = v * t + 1 / 2 * a * t^2)
  (h₂ : t = 2 → S = 10 → S = v * t + 1 / 2 * a * t^2):
  t = 3 → S = v * t + 1 / 2 * a * t^2 → S = 18 := by
  sorry

end NUMINAMATH_GPT_problem1_l151_15133


namespace NUMINAMATH_GPT_calculation_result_l151_15116

theorem calculation_result : 50 + 50 / 50 + 50 = 101 := by
  sorry

end NUMINAMATH_GPT_calculation_result_l151_15116


namespace NUMINAMATH_GPT_range_of_set_l151_15181

theorem range_of_set (a b c : ℕ) (h1 : a = 2) (h2 : b = 6) (h3 : 2 ≤ c ∧ c ≤ 10) (h4 : (a + b + c) / 3 = 6) : (c - a) = 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_set_l151_15181


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l151_15156

theorem infinite_geometric_series_sum :
  let a : ℚ := 1
  let r : ℚ := 1 / 3
  ∑' (n : ℕ), a * r ^ n = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_sum_l151_15156


namespace NUMINAMATH_GPT_multiply_by_11_l151_15168

theorem multiply_by_11 (A B : ℕ) (h : A + B < 10) : 
  (10 * A + B) * 11 = 100 * A + 10 * (A + B) + B :=
by
  sorry

end NUMINAMATH_GPT_multiply_by_11_l151_15168


namespace NUMINAMATH_GPT_length_of_platform_l151_15142

def len_train : ℕ := 300 -- length of the train in meters
def time_platform : ℕ := 39 -- time to cross the platform in seconds
def time_pole : ℕ := 26 -- time to cross the signal pole in seconds

theorem length_of_platform (L : ℕ) (h1 : len_train / time_pole = (len_train + L) / time_platform) : L = 150 :=
  sorry

end NUMINAMATH_GPT_length_of_platform_l151_15142


namespace NUMINAMATH_GPT_Cameron_books_proof_l151_15196

noncomputable def Cameron_initial_books :=
  let B : ℕ := 24
  let B_donated := B / 4
  let B_left := B - B_donated
  let C_donated (C : ℕ) := C / 3
  let C_left (C : ℕ) := C - C_donated C
  ∃ C : ℕ, B_left + C_left C = 38 ∧ C = 30

-- Note that we use sorry to indicate the proof is omitted.
theorem Cameron_books_proof : Cameron_initial_books :=
by {
  sorry
}

end NUMINAMATH_GPT_Cameron_books_proof_l151_15196


namespace NUMINAMATH_GPT_uneaten_chips_correct_l151_15113

def cookies_per_dozen : Nat := 12
def dozens : Nat := 4
def chips_per_cookie : Nat := 7

def total_cookies : Nat := dozens * cookies_per_dozen
def total_chips : Nat := total_cookies * chips_per_cookie
def eaten_cookies : Nat := total_cookies / 2
def uneaten_cookies : Nat := total_cookies - eaten_cookies

def uneaten_chips : Nat := uneaten_cookies * chips_per_cookie

theorem uneaten_chips_correct : uneaten_chips = 168 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_uneaten_chips_correct_l151_15113


namespace NUMINAMATH_GPT_spider_final_position_l151_15107

def circle_points : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def next_position (current : ℕ) : ℕ :=
  if current % 2 = 0 
  then (current + 3 - 1) % 7 + 1 -- Clockwise modulo operation for even
  else (current + 1 - 1) % 7 + 1 -- Clockwise modulo operation for odd

def spider_position_after_jumps (start : ℕ) (jumps : ℕ) : ℕ :=
  (Nat.iterate next_position jumps start)

theorem spider_final_position : spider_position_after_jumps 6 2055 = 2 := 
  by
  sorry

end NUMINAMATH_GPT_spider_final_position_l151_15107


namespace NUMINAMATH_GPT_multiplication_result_l151_15153

theorem multiplication_result : 143 * 21 * 4 * 37 * 2 = 888888 := by
  sorry

end NUMINAMATH_GPT_multiplication_result_l151_15153


namespace NUMINAMATH_GPT_roots_negative_reciprocal_condition_l151_15122

theorem roots_negative_reciprocal_condition (a b c : ℝ) (h : a ≠ 0) :
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r * s = -1 ∧ a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0) → c = -a :=
by
  sorry

end NUMINAMATH_GPT_roots_negative_reciprocal_condition_l151_15122


namespace NUMINAMATH_GPT_children_left_birthday_l151_15132

theorem children_left_birthday 
  (total_guests : ℕ := 60)
  (women : ℕ := 30)
  (men : ℕ := 15)
  (remaining_guests : ℕ := 50)
  (initial_children : ℕ := total_guests - women - men)
  (men_left : ℕ := men / 3)
  (total_left : ℕ := total_guests - remaining_guests)
  (children_left : ℕ := total_left - men_left) :
  children_left = 5 :=
by
  sorry

end NUMINAMATH_GPT_children_left_birthday_l151_15132


namespace NUMINAMATH_GPT_min_value_x_plus_y_l151_15121

theorem min_value_x_plus_y (x y : ℤ) (det : 3 < x * y ∧ x * y < 5) : x + y = -5 :=
sorry

end NUMINAMATH_GPT_min_value_x_plus_y_l151_15121
