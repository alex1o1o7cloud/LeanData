import Mathlib

namespace NUMINAMATH_GPT_problem_solution_l1261_126148

theorem problem_solution (x y : ℚ) (h1 : |x| + x + y - 2 = 14) (h2 : x + |y| - y + 3 = 20) : 
  x + y = 31/5 := 
by
  -- It remains to prove
  sorry

end NUMINAMATH_GPT_problem_solution_l1261_126148


namespace NUMINAMATH_GPT_town_population_growth_l1261_126110

noncomputable def populationAfterYears (population : ℝ) (year1Increase : ℝ) (year2Increase : ℝ) : ℝ :=
  let populationAfterFirstYear := population * (1 + year1Increase)
  let populationAfterSecondYear := populationAfterFirstYear * (1 + year2Increase)
  populationAfterSecondYear

theorem town_population_growth :
  ∀ (initialPopulation : ℝ) (year1Increase : ℝ) (year2Increase : ℝ),
    initialPopulation = 1000 → year1Increase = 0.10 → year2Increase = 0.20 →
      populationAfterYears initialPopulation year1Increase year2Increase = 1320 :=
by
  intros initialPopulation year1Increase year2Increase h1 h2 h3
  rw [h1, h2, h3]
  have h4 : populationAfterYears 1000 0.10 0.20 = 1320 := sorry
  exact h4

end NUMINAMATH_GPT_town_population_growth_l1261_126110


namespace NUMINAMATH_GPT_moral_of_saying_l1261_126103

/-!
  Comrade Mao Zedong said: "If you want to know the taste of a pear, you must change the pear and taste it yourself." 
  Prove that this emphasizes "Practice is the source of knowledge" (option C) over the other options.
-/

def question := "What is the moral of Comrade Mao Zedong's famous saying about tasting a pear?"

def options := ["Knowledge is the driving force behind the development of practice", 
                "Knowledge guides practice", 
                "Practice is the source of knowledge", 
                "Practice has social and historical characteristics"]

def correct_answer := "Practice is the source of knowledge"

theorem moral_of_saying : (question, options[2]) ∈ [("What is the moral of Comrade Mao Zedong's famous saying about tasting a pear?", 
                                                      "Practice is the source of knowledge")] := by 
  sorry

end NUMINAMATH_GPT_moral_of_saying_l1261_126103


namespace NUMINAMATH_GPT_midpoint_trajectory_l1261_126199

theorem midpoint_trajectory (x y : ℝ) :
  (∃ B C : ℝ × ℝ, B ≠ C ∧ (B.1^2 + B.2^2 = 25) ∧ (C.1^2 + C.2^2 = 25) ∧ 
                   (x, y) = ((B.1 + C.1)/2, (B.2 + C.2)/2) ∧ 
                   (B.1 - C.1)^2 + (B.2 - C.2)^2 = 36) →
  x^2 + y^2 = 16 :=
sorry

end NUMINAMATH_GPT_midpoint_trajectory_l1261_126199


namespace NUMINAMATH_GPT_prime_divisors_of_n_congruent_to_1_mod_4_l1261_126108

theorem prime_divisors_of_n_congruent_to_1_mod_4
  (x y n : ℕ)
  (hx : x ≥ 3)
  (hn : n ≥ 2)
  (h_eq : x^2 + 5 = y^n) :
  ∀ p : ℕ, Prime p → p ∣ n → p ≡ 1 [MOD 4] :=
by
  sorry

end NUMINAMATH_GPT_prime_divisors_of_n_congruent_to_1_mod_4_l1261_126108


namespace NUMINAMATH_GPT_find_p_and_q_solution_set_l1261_126160

theorem find_p_and_q (p q : ℝ) (h : ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - p * x - q < 0) : 
  p = 5 ∧ q = -6 :=
sorry

theorem solution_set (p q : ℝ) (h_p : p = 5) (h_q : q = -6) : 
  ∀ x : ℝ, q * x^2 - p * x - 1 > 0 ↔ - (1 / 2) < x ∧ x < - (1 / 3) :=
sorry

end NUMINAMATH_GPT_find_p_and_q_solution_set_l1261_126160


namespace NUMINAMATH_GPT_simplify_expression_eq_l1261_126163

theorem simplify_expression_eq (a : ℝ) (h₀ : a ≠ 0) (h₁ : a ≠ 1) : 
  (a - 1/a) / ((a^2 - 2 * a + 1) / a) = (a + 1) / (a - 1) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_eq_l1261_126163


namespace NUMINAMATH_GPT_sequence_monotonic_decreasing_l1261_126168

theorem sequence_monotonic_decreasing (t : ℝ) :
  (∀ n : ℕ, n > 0 → (- (n + 1) ^ 2 + t * (n + 1)) - (- n ^ 2 + t * n) < 0) ↔ (t < 3) :=
by 
  sorry

end NUMINAMATH_GPT_sequence_monotonic_decreasing_l1261_126168


namespace NUMINAMATH_GPT_inscribed_sphere_to_cube_volume_ratio_l1261_126177

theorem inscribed_sphere_to_cube_volume_ratio :
  let s := 8
  let r := s / 2
  let V_sphere := (4/3) * Real.pi * r^3
  let V_cube := s^3
  (V_sphere / V_cube) = Real.pi / 6 :=
by
  let s := 8
  let r := s / 2
  let V_sphere := (4/3) * Real.pi * r^3
  let V_cube := s^3
  sorry

end NUMINAMATH_GPT_inscribed_sphere_to_cube_volume_ratio_l1261_126177


namespace NUMINAMATH_GPT_express_in_scientific_notation_l1261_126139

theorem express_in_scientific_notation (n : ℝ) (h : n = 456.87 * 10^6) : n = 4.5687 * 10^8 :=
by 
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_express_in_scientific_notation_l1261_126139


namespace NUMINAMATH_GPT_solve_quadratic_eq_l1261_126193

theorem solve_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l1261_126193


namespace NUMINAMATH_GPT_distance_traveled_by_car_l1261_126144

theorem distance_traveled_by_car :
  let total_distance := 90
  let distance_by_foot := (1 / 5 : ℝ) * total_distance
  let distance_by_bus := (2 / 3 : ℝ) * total_distance
  let distance_by_car := total_distance - (distance_by_foot + distance_by_bus)
  distance_by_car = 12 :=
by
  sorry

end NUMINAMATH_GPT_distance_traveled_by_car_l1261_126144


namespace NUMINAMATH_GPT_evaluate_neg_64_pow_two_thirds_l1261_126109

theorem evaluate_neg_64_pow_two_thirds 
  (h : -64 = (-4)^3) : (-64)^(2/3) = 16 := 
by 
  -- sorry added to skip the proof.
  sorry  

end NUMINAMATH_GPT_evaluate_neg_64_pow_two_thirds_l1261_126109


namespace NUMINAMATH_GPT_mark_buttons_l1261_126196

/-- Mark started the day with some buttons. His friend Shane gave him 3 times that amount of buttons.
    Then his other friend Sam asked if he could have half of Mark’s buttons. 
    Mark ended up with 28 buttons. How many buttons did Mark start the day with? --/
theorem mark_buttons (B : ℕ) (h1 : 2 * B = 28) : B = 14 := by
  sorry

end NUMINAMATH_GPT_mark_buttons_l1261_126196


namespace NUMINAMATH_GPT_avg_first_12_even_is_13_l1261_126151

-- Definition of the first 12 even numbers
def first_12_even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

-- The sum of the first 12 even numbers
def sum_first_12_even_numbers : ℕ := first_12_even_numbers.sum

-- Number of first 12 even numbers
def count_12_even_numbers : ℕ := first_12_even_numbers.length

-- The average of the first 12 even numbers
def average_12_even_numbers : ℕ := sum_first_12_even_numbers / count_12_even_numbers

-- Proof statement that the average of the first 12 even numbers is 13
theorem avg_first_12_even_is_13 : average_12_even_numbers = 13 := by
  sorry

end NUMINAMATH_GPT_avg_first_12_even_is_13_l1261_126151


namespace NUMINAMATH_GPT_h_plus_k_l1261_126165

theorem h_plus_k :
  ∀ h k : ℝ, (∀ x : ℝ, x^2 + 4 * x + 4 = (x + h) ^ 2 - k) → h + k = 2 :=
by
  intro h k H
  -- using sorry to indicate the proof is omitted
  sorry

end NUMINAMATH_GPT_h_plus_k_l1261_126165


namespace NUMINAMATH_GPT_sum_of_five_integers_l1261_126192

theorem sum_of_five_integers :
  ∃ (n m : ℕ), (n * (n + 1) = 336) ∧ ((m - 1) * m * (m + 1) = 336) ∧ ((n + (n + 1) + (m - 1) + m + (m + 1)) = 51) := 
sorry

end NUMINAMATH_GPT_sum_of_five_integers_l1261_126192


namespace NUMINAMATH_GPT_range_of_m_l1261_126185

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem range_of_m (m : ℝ) (h : second_quadrant (m-3) (m-2)) : 2 < m ∧ m < 3 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1261_126185


namespace NUMINAMATH_GPT_molecular_weight_al_fluoride_l1261_126115

/-- Proving the molecular weight of Aluminum fluoride calculation -/
theorem molecular_weight_al_fluoride (x : ℕ) (h : 10 * x = 840) : x = 84 :=
by sorry

end NUMINAMATH_GPT_molecular_weight_al_fluoride_l1261_126115


namespace NUMINAMATH_GPT_chord_square_length_l1261_126175

theorem chord_square_length
    (r1 r2 r3 L1 L2 L3 : ℝ)
    (h1 : r1 = 4) 
    (h2 : r2 = 8) 
    (h3 : r3 = 12) 
    (tangent1 : ∀ x, (L1 - x)^2 + (L2 - x)^2 = (r1 + r2)^2)
    (tangent2 : ∀ x, x^2 + (L3 - x)^2 = (r3 - r2)^2) 
    (tangent3 : ∀ x, x^2 + (L3 - x)^2 = (r3 - r1)^2) : L1^2 = 3584 / 9 :=
by
  sorry

end NUMINAMATH_GPT_chord_square_length_l1261_126175


namespace NUMINAMATH_GPT_minimum_value_l1261_126140

open Real

theorem minimum_value (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (h : x * y - 2 * x - y + 1 = 0) :
  ∃ z : ℝ, (z = (3 / 2) * x^2 + y^2) ∧ z = 15 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l1261_126140


namespace NUMINAMATH_GPT_min_pairs_l1261_126116

-- Define the types for knights and liars
inductive Residents
| Knight : Residents
| Liar : Residents

def total_residents : ℕ := 200
def knights : ℕ := 100
def liars : ℕ := 100

-- Additional conditions
def conditions (friend_claims_knights friend_claims_liars : ℕ) : Prop :=
  friend_claims_knights = 100 ∧
  friend_claims_liars = 100 ∧
  knights + liars = total_residents

-- Minimum number of knight-liar pairs to prove
def min_knight_liar_pairs : ℕ := 50

theorem min_pairs {friend_claims_knights friend_claims_liars : ℕ} (h : conditions friend_claims_knights friend_claims_liars) :
    min_knight_liar_pairs = 50 :=
sorry

end NUMINAMATH_GPT_min_pairs_l1261_126116


namespace NUMINAMATH_GPT_bowling_ball_weight_l1261_126124

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 9 * b = 6 * c) 
  (h2 : 4 * c = 120) : 
  b = 20 :=
by 
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l1261_126124


namespace NUMINAMATH_GPT_wuxi_GDP_scientific_notation_l1261_126132

theorem wuxi_GDP_scientific_notation :
  14800 = 1.48 * 10^4 :=
sorry

end NUMINAMATH_GPT_wuxi_GDP_scientific_notation_l1261_126132


namespace NUMINAMATH_GPT_vendor_has_1512_liters_of_sprite_l1261_126118

-- Define the conditions
def liters_of_maaza := 60
def liters_of_pepsi := 144
def least_number_of_cans := 143
def gcd_maaza_pepsi := Nat.gcd liters_of_maaza liters_of_pepsi --let Lean compute GCD

-- Define the liters per can as the GCD of Maaza and Pepsi
def liters_per_can := gcd_maaza_pepsi

-- Define the number of cans for Maaza and Pepsi respectively
def cans_of_maaza := liters_of_maaza / liters_per_can
def cans_of_pepsi := liters_of_pepsi / liters_per_can

-- Define total cans for Maaza and Pepsi
def total_cans_for_maaza_and_pepsi := cans_of_maaza + cans_of_pepsi

-- Define the number of cans for Sprite
def cans_of_sprite := least_number_of_cans - total_cans_for_maaza_and_pepsi

-- The total liters of Sprite the vendor has
def liters_of_sprite := cans_of_sprite * liters_per_can

-- Statement to prove
theorem vendor_has_1512_liters_of_sprite : 
  liters_of_sprite = 1512 :=
by
  -- solution omitted 
  sorry

end NUMINAMATH_GPT_vendor_has_1512_liters_of_sprite_l1261_126118


namespace NUMINAMATH_GPT_circle_center_line_condition_l1261_126197

theorem circle_center_line_condition (a : ℝ) :
    (∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 4 * y - 6 = 0 → (a, -2) = (x, y) → x + 2 * y + 1 = 0) → a = 3 :=
by
  sorry

end NUMINAMATH_GPT_circle_center_line_condition_l1261_126197


namespace NUMINAMATH_GPT_walter_zoo_time_l1261_126162

theorem walter_zoo_time (S: ℕ) (H1: S + 8 * S + 13 = 130) : S = 13 :=
by sorry

end NUMINAMATH_GPT_walter_zoo_time_l1261_126162


namespace NUMINAMATH_GPT_three_digit_minuends_count_l1261_126126

theorem three_digit_minuends_count :
  ∀ a b c : ℕ, a - c = 4 ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
  (∃ n : ℕ, n = 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c - 396 = 100 * c + 10 * b + a) →
  ∃ count : ℕ, count = 50 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_minuends_count_l1261_126126


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l1261_126123

theorem solve_eq1 (x : ℝ) : x^2 - 6*x - 7 = 0 → x = 7 ∨ x = -1 :=
by
  sorry

theorem solve_eq2 (x : ℝ) : 3*x^2 - 1 = 2*x → x = 1 ∨ x = -1/3 :=
by
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l1261_126123


namespace NUMINAMATH_GPT_number_of_buckets_after_reduction_l1261_126128

def initial_buckets : ℕ := 25
def reduction_factor : ℚ := 2 / 5

theorem number_of_buckets_after_reduction :
  (initial_buckets : ℚ) * (1 / reduction_factor) = 63 := by
  sorry

end NUMINAMATH_GPT_number_of_buckets_after_reduction_l1261_126128


namespace NUMINAMATH_GPT_evaluate_expression_l1261_126131

theorem evaluate_expression (c : ℕ) (h : c = 4) : (c^c - c * (c - 1)^(c - 1))^c = 148^4 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1261_126131


namespace NUMINAMATH_GPT_number_of_people_joining_group_l1261_126182

theorem number_of_people_joining_group (x : ℕ) (h1 : 180 / 18 = 10) 
  (h2 : 180 / (18 + x) = 9) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_joining_group_l1261_126182


namespace NUMINAMATH_GPT_unique_integer_solution_l1261_126195

theorem unique_integer_solution (x : ℤ) : x^3 + (x + 1)^3 + (x + 2)^3 = (x + 3)^3 ↔ x = 3 := by
  sorry

end NUMINAMATH_GPT_unique_integer_solution_l1261_126195


namespace NUMINAMATH_GPT_no_real_roots_of_quadratic_l1261_126186

theorem no_real_roots_of_quadratic (k : ℝ) (hk : k ≠ 0) : ¬ ∃ x : ℝ, x^2 + 2 * k * x + 3 * k^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_of_quadratic_l1261_126186


namespace NUMINAMATH_GPT_measure_AX_l1261_126135

-- Definitions based on conditions
def circle_radii (r_A r_B r_C : ℝ) : Prop :=
  r_A - r_B = 6 ∧
  r_A - r_C = 5 ∧
  r_B + r_C = 9

-- Theorem statement
theorem measure_AX (r_A r_B r_C : ℝ) (h : circle_radii r_A r_B r_C) : r_A = 10 :=
by
  sorry

end NUMINAMATH_GPT_measure_AX_l1261_126135


namespace NUMINAMATH_GPT_pieces_from_sister_calculation_l1261_126111

-- Definitions for the conditions
def pieces_from_neighbors : ℕ := 5
def pieces_per_day : ℕ := 9
def duration : ℕ := 2

-- Definition to calculate the total number of pieces Emily ate
def total_pieces : ℕ := pieces_per_day * duration

-- Proof Problem: Prove Emily received 13 pieces of candy from her older sister
theorem pieces_from_sister_calculation :
  ∃ (pieces_from_sister : ℕ), pieces_from_sister = total_pieces - pieces_from_neighbors ∧ pieces_from_sister = 13 :=
by
  sorry

end NUMINAMATH_GPT_pieces_from_sister_calculation_l1261_126111


namespace NUMINAMATH_GPT_prime_ratio_sum_l1261_126188

theorem prime_ratio_sum (p q m : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
(h_roots : ∀ x : ℝ, x^2 - 99 * x + m = 0 → x = p ∨ x = q) :
  (p : ℚ) / q + q / p = 9413 / 194 :=
sorry

end NUMINAMATH_GPT_prime_ratio_sum_l1261_126188


namespace NUMINAMATH_GPT_gcd_102_238_l1261_126143

theorem gcd_102_238 : Nat.gcd 102 238 = 34 :=
by
  sorry

end NUMINAMATH_GPT_gcd_102_238_l1261_126143


namespace NUMINAMATH_GPT_compute_expression_l1261_126121

theorem compute_expression : 85 * 1305 - 25 * 1305 + 100 = 78400 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l1261_126121


namespace NUMINAMATH_GPT_least_multiple_of_25_gt_475_l1261_126147

theorem least_multiple_of_25_gt_475 : ∃ n : ℕ, n > 475 ∧ n % 25 = 0 ∧ ∀ m : ℕ, (m > 475 ∧ m % 25 = 0) → n ≤ m := 
  sorry

end NUMINAMATH_GPT_least_multiple_of_25_gt_475_l1261_126147


namespace NUMINAMATH_GPT_simplify_expression_l1261_126129

theorem simplify_expression (x : ℤ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 + 24 = 45 * x + 42 := 
by 
  -- proof steps
  sorry

end NUMINAMATH_GPT_simplify_expression_l1261_126129


namespace NUMINAMATH_GPT_right_angle_vertex_trajectory_l1261_126138

theorem right_angle_vertex_trajectory (x y : ℝ) :
  let M := (-2, 0)
  let N := (2, 0)
  let P := (x, y)
  (∃ (x y : ℝ), (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16) →
  x ≠ 2 ∧ x ≠ -2 →
  x^2 + y^2 = 4 :=
by
  intro h₁ h₂
  sorry

end NUMINAMATH_GPT_right_angle_vertex_trajectory_l1261_126138


namespace NUMINAMATH_GPT_problem_l1261_126107

theorem problem (x : ℝ) (h : x + 1/x = 10) :
  (x^2 + 1/x^2 = 98) ∧ (x^3 + 1/x^3 = 970) :=
by
  sorry

end NUMINAMATH_GPT_problem_l1261_126107


namespace NUMINAMATH_GPT_distance_to_right_focus_l1261_126184

variable (F1 F2 P : ℝ × ℝ)
variable (a : ℝ)
variable (h_ellipse : ∀ P : ℝ × ℝ, P ∈ { P : ℝ × ℝ | (P.1^2 / 9) + (P.2^2 / 8) = 1 })
variable (h_foci_dist : (P : ℝ × ℝ) → (F1 : ℝ × ℝ) → (F2 : ℝ × ℝ) → (dist P F1) = 2)
variable (semi_major_axis : a = 3)

theorem distance_to_right_focus (h : dist F1 F2 = 2 * a) : dist P F2 = 4 := 
sorry

end NUMINAMATH_GPT_distance_to_right_focus_l1261_126184


namespace NUMINAMATH_GPT_repeating_decimal_product_l1261_126105

theorem repeating_decimal_product 
  (x : ℚ) 
  (h1 : x = (0.0126 : ℚ)) 
  (h2 : 9999 * x = 126) 
  (h3 : x = 14 / 1111) : 
  14 * 1111 = 15554 := 
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_product_l1261_126105


namespace NUMINAMATH_GPT_daisy_dog_toys_l1261_126145

theorem daisy_dog_toys (X : ℕ) (lost_toys : ℕ) (total_toys_after_found : ℕ) : 
    (X - lost_toys + (3 + 3) - lost_toys + 5 = total_toys_after_found) → total_toys_after_found = 13 → X = 5 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_daisy_dog_toys_l1261_126145


namespace NUMINAMATH_GPT_no_infinite_prime_sequence_l1261_126156

theorem no_infinite_prime_sequence (p : ℕ → ℕ)
  (h : ∀ k : ℕ, Nat.Prime (p k) ∧ p (k + 1) = 5 * p k + 4) :
  ¬ ∀ n : ℕ, Nat.Prime (p n) :=
by
  sorry

end NUMINAMATH_GPT_no_infinite_prime_sequence_l1261_126156


namespace NUMINAMATH_GPT_simplify_exponent_l1261_126173

theorem simplify_exponent :
  2000 * 2000^2000 = 2000^2001 :=
by
  sorry

end NUMINAMATH_GPT_simplify_exponent_l1261_126173


namespace NUMINAMATH_GPT_negation_of_p_l1261_126117

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 := by
  sorry

end NUMINAMATH_GPT_negation_of_p_l1261_126117


namespace NUMINAMATH_GPT_pipe_fill_time_without_leakage_l1261_126134

theorem pipe_fill_time_without_leakage (t : ℕ) (h1 : 7 * t * (1/t - 1/70) = 1) : t = 60 :=
by
  sorry

end NUMINAMATH_GPT_pipe_fill_time_without_leakage_l1261_126134


namespace NUMINAMATH_GPT_compute_cos_l1261_126106

noncomputable def angle1 (A C B : ℝ) : Prop := A + C = 2 * B
noncomputable def angle2 (A C B : ℝ) : Prop := 1 / Real.cos A + 1 / Real.cos C = -Real.sqrt 2 / Real.cos B

theorem compute_cos (A B C : ℝ) (h1 : angle1 A C B) (h2 : angle2 A C B) : 
  Real.cos ((A - C) / 2) = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_compute_cos_l1261_126106


namespace NUMINAMATH_GPT_unique_solutions_of_system_l1261_126141

theorem unique_solutions_of_system (a : ℝ) :
  (∃! (x y : ℝ), a^2 - 2 * a * x - 6 * y + x^2 + y^2 = 0 ∧ (|x| - 4)^2 + (|y| - 3)^2 = 25) ↔
  (a ∈ Set.union (Set.Ioo (-12) (-6)) (Set.union {0} (Set.Ioo 6 12))) :=
by
  sorry

end NUMINAMATH_GPT_unique_solutions_of_system_l1261_126141


namespace NUMINAMATH_GPT_solve_inequality_system_l1261_126150

theorem solve_inequality_system (x : ℝ) : 
  (5 * x - 1 > 3 * (x + 1)) →
  ((1 / 2) * x - 1 ≤ 7 - (3 / 2) * x) →
  (2 < x ∧ x ≤ 4) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l1261_126150


namespace NUMINAMATH_GPT_remainder_when_112222333_divided_by_37_l1261_126166

theorem remainder_when_112222333_divided_by_37 : 112222333 % 37 = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_112222333_divided_by_37_l1261_126166


namespace NUMINAMATH_GPT_peter_contains_five_l1261_126136

theorem peter_contains_five (N : ℕ) (hN : N > 0) :
  ∃ k : ℕ, ∀ m : ℕ, m ≥ k → ∃ i : ℕ, 5 ≤ 10^i * (N * 5^m / 10^i) % 10 :=
sorry

end NUMINAMATH_GPT_peter_contains_five_l1261_126136


namespace NUMINAMATH_GPT_min_value_M_proof_l1261_126152

noncomputable def min_value_M (a b c d e f g M : ℝ) : Prop :=
  (∀ (a b c d e f g : ℝ), 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ g ≥ 0 ∧ 
    a + b + c + d + e + f + g = 1 ∧ 
    M = max (max (max (max (a + b + c) (b + c + d)) (c + d + e)) (d + e + f)) (e + f + g)
  → M ≥ (1 / 3))

theorem min_value_M_proof : min_value_M a b c d e f g M :=
by
  sorry

end NUMINAMATH_GPT_min_value_M_proof_l1261_126152


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_for_parallel_lines_l1261_126179

theorem necessary_and_sufficient_condition_for_parallel_lines (a l : ℝ) :
  (a = -1) ↔ (∀ x y : ℝ, ax + 3 * y + 3 = 0 → x + (a - 2) * y + l = 0) := 
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_for_parallel_lines_l1261_126179


namespace NUMINAMATH_GPT_P_intersection_Q_is_singleton_l1261_126120

theorem P_intersection_Q_is_singleton :
  {p : ℝ × ℝ | p.1 + p.2 = 3} ∩ {p : ℝ × ℝ | p.1 - p.2 = 5} = { (4, -1) } :=
by
  -- The proof steps would go here.
  sorry

end NUMINAMATH_GPT_P_intersection_Q_is_singleton_l1261_126120


namespace NUMINAMATH_GPT_equation_of_line_passing_through_point_with_slope_l1261_126149

theorem equation_of_line_passing_through_point_with_slope :
  ∃ (l : ℝ → ℝ), l 0 = -1 ∧ ∀ (x y : ℝ), y = l x ↔ y + 1 = 2 * x :=
sorry

end NUMINAMATH_GPT_equation_of_line_passing_through_point_with_slope_l1261_126149


namespace NUMINAMATH_GPT_volume_of_circumscribed_polyhedron_l1261_126112

theorem volume_of_circumscribed_polyhedron (R : ℝ) (V : ℝ) (S_n : ℝ) (h : Π (F_i : ℝ), V = (1/3) * S_n * R) : V = (1/3) * S_n * R :=
sorry

end NUMINAMATH_GPT_volume_of_circumscribed_polyhedron_l1261_126112


namespace NUMINAMATH_GPT_find_m_l1261_126178

theorem find_m (m : ℝ) :
  (∀ x y : ℝ, (3 * x + (m + 1) * y - (m - 7) = 0) → 
              (m * x + 2 * y + 3 * m = 0)) →
  (m + 1 ≠ 0) →
  m = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1261_126178


namespace NUMINAMATH_GPT_sum_of_x_y_l1261_126154

theorem sum_of_x_y (x y : ℝ) (h1 : 3 * x + 2 * y = 10) (h2 : 2 * x + 3 * y = 5) : x + y = 3 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_x_y_l1261_126154


namespace NUMINAMATH_GPT_sum_of_digits_l1261_126122

theorem sum_of_digits (d : ℕ) (h1 : d % 5 = 0) (h2 : 3 * d - 75 = d) : 
  (d / 10 + d % 10) = 11 :=
by {
  -- Placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_sum_of_digits_l1261_126122


namespace NUMINAMATH_GPT_determine_words_per_page_l1261_126130

noncomputable def wordsPerPage (totalPages : ℕ) (wordsPerPage : ℕ) (totalWordsMod : ℕ) : ℕ :=
if totalPages * wordsPerPage % 250 = totalWordsMod ∧ wordsPerPage <= 200 then wordsPerPage else 0

theorem determine_words_per_page :
  wordsPerPage 150 198 137 = 198 :=
by 
  sorry

end NUMINAMATH_GPT_determine_words_per_page_l1261_126130


namespace NUMINAMATH_GPT_total_area_correct_l1261_126164

-- Define the conditions from the problem
def side_length_small : ℕ := 2
def side_length_medium : ℕ := 4
def side_length_large : ℕ := 8

-- Define the areas of individual squares
def area_small : ℕ := side_length_small * side_length_small
def area_medium : ℕ := side_length_medium * side_length_medium
def area_large : ℕ := side_length_large * side_length_large

-- Define the additional areas as suggested by vague steps in the solution
def area_term1 : ℕ := 4 * 4 / 2 * 2
def area_term2 : ℕ := 2 * 2 / 2
def area_term3 : ℕ := (8 + 2) * 2 / 2 * 2

-- Define the total area as the sum of all calculated parts
def total_area : ℕ := area_large + (area_medium * 3) + area_small + area_term1 + area_term2 + area_term3

-- The theorem to prove total area is 150 square centimeters
theorem total_area_correct : total_area = 150 :=
by
  -- Proof goes here (steps from the solution)...
  sorry

end NUMINAMATH_GPT_total_area_correct_l1261_126164


namespace NUMINAMATH_GPT_johns_weekly_allowance_l1261_126169

theorem johns_weekly_allowance (A : ℝ) (h1: A - (3/5) * A = (2/5) * A)
  (h2: (2/5) * A - (1/3) * (2/5) * A = (4/15) * A)
  (h3: (4/15) * A = 0.92) : A = 3.45 :=
by {
  sorry
}

end NUMINAMATH_GPT_johns_weekly_allowance_l1261_126169


namespace NUMINAMATH_GPT_total_handshakes_l1261_126187

variable (n : ℕ) (h : n = 12)

theorem total_handshakes (H : ∀ (b : ℕ), b = n → (n * (n - 1)) / 2 = 66) : 
  (12 * 11) / 2 = 66 := 
by
  sorry

end NUMINAMATH_GPT_total_handshakes_l1261_126187


namespace NUMINAMATH_GPT_abc_gt_16_abc_geq_3125_div_108_l1261_126113

variables {a b c α β : ℝ}

-- Define the conditions
def conditions (a b c α β : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ b > 0 ∧
  (a * α^2 + b * α - c = 0) ∧
  (a * β^2 + b * β - c = 0) ∧
  (α ≠ β) ∧
  (α^3 + b * α^2 + a * α - c = 0) ∧
  (β^3 + b * β^2 + a * β - c = 0)

-- State the first proof problem
theorem abc_gt_16 (h : conditions a b c α β) : a * b * c > 16 :=
sorry

-- State the second proof problem
theorem abc_geq_3125_div_108 (h : conditions a b c α β) : a * b * c ≥ 3125 / 108 :=
sorry

end NUMINAMATH_GPT_abc_gt_16_abc_geq_3125_div_108_l1261_126113


namespace NUMINAMATH_GPT_triangle_area_l1261_126133

theorem triangle_area : 
  let p1 := (0, 0)
  let p2 := (0, 6)
  let p3 := (8, 15)
  let base := 6
  let height := 8
  0.5 * base * height = 24.0 :=
by
  let p1 := (0, 0)
  let p2 := (0, 6)
  let p3 := (8, 15)
  let base := 6
  let height := 8
  sorry

end NUMINAMATH_GPT_triangle_area_l1261_126133


namespace NUMINAMATH_GPT_find_a4_b4_l1261_126176

theorem find_a4_b4
  (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (h1 : a1 * b1 + a2 * b3 = 1)
  (h2 : a1 * b2 + a2 * b4 = 0)
  (h3 : a3 * b1 + a4 * b3 = 0)
  (h4 : a3 * b2 + a4 * b4 = 1)
  (h5 : a2 * b3 = 7) : 
  a4 * b4 = -6 :=
sorry

end NUMINAMATH_GPT_find_a4_b4_l1261_126176


namespace NUMINAMATH_GPT_rectangle_area_l1261_126127

theorem rectangle_area (x : ℝ) (h : (2*x - 3) * (3*x + 4) = 20 * x - 12) : x = 7 / 2 :=
sorry

end NUMINAMATH_GPT_rectangle_area_l1261_126127


namespace NUMINAMATH_GPT_minimum_value_of_f_l1261_126183

noncomputable def f (x : ℝ) : ℝ := x + 1 / x + 1 / (x + 1 / x) + 1 / (x^2 + 1 / x^2)

theorem minimum_value_of_f :
  (∀ x > 0, f x ≥ 3) ∧ (f 1 = 3) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1261_126183


namespace NUMINAMATH_GPT_smallest_number_divisible_l1261_126190

theorem smallest_number_divisible
  (x : ℕ)
  (h : (x - 2) % 12 = 0 ∧ (x - 2) % 16 = 0 ∧ (x - 2) % 18 = 0 ∧ (x - 2) % 21 = 0 ∧ (x - 2) % 28 = 0) :
  x = 1010 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_l1261_126190


namespace NUMINAMATH_GPT_janina_spend_on_supplies_each_day_l1261_126100

theorem janina_spend_on_supplies_each_day 
  (rent : ℝ)
  (p : ℝ)
  (n : ℕ)
  (H1 : rent = 30)
  (H2 : p = 2)
  (H3 : n = 21) :
  (n : ℝ) * p - rent = 12 := 
by
  sorry

end NUMINAMATH_GPT_janina_spend_on_supplies_each_day_l1261_126100


namespace NUMINAMATH_GPT_single_elimination_games_l1261_126174

theorem single_elimination_games (n : ℕ) (h : n = 512) : 
  (n - 1) = 511 :=
by
  sorry

end NUMINAMATH_GPT_single_elimination_games_l1261_126174


namespace NUMINAMATH_GPT_mary_average_speed_l1261_126158

noncomputable def average_speed (d1 d2 : ℝ) (t1 t2 : ℝ) : ℝ :=
  (d1 + d2) / ((t1 + t2) / 60)

theorem mary_average_speed :
  average_speed 1.5 1.5 45 15 = 3 := by
  sorry

end NUMINAMATH_GPT_mary_average_speed_l1261_126158


namespace NUMINAMATH_GPT_smallest_unfound_digit_in_odd_units_l1261_126157

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end NUMINAMATH_GPT_smallest_unfound_digit_in_odd_units_l1261_126157


namespace NUMINAMATH_GPT_correct_option_l1261_126194

theorem correct_option : 
  (-(2:ℤ))^3 ≠ -6 ∧ 
  (-(1:ℤ))^10 ≠ -10 ∧ 
  (-(1:ℚ)/3)^3 ≠ -1/9 ∧ 
  -(2:ℤ)^2 = -4 :=
by 
  sorry

end NUMINAMATH_GPT_correct_option_l1261_126194


namespace NUMINAMATH_GPT_number_of_boys_in_class_l1261_126167

theorem number_of_boys_in_class
  (g_ratio : ℕ) (b_ratio : ℕ) (total_students : ℕ)
  (h_ratio : g_ratio / b_ratio = 4 / 3)
  (h_total_students : g_ratio + b_ratio = 7 * (total_students / 56)) :
  total_students = 56 → 3 * (total_students / (4 + 3)) = 24 :=
by
  intros total_students_56
  sorry

end NUMINAMATH_GPT_number_of_boys_in_class_l1261_126167


namespace NUMINAMATH_GPT_converse_of_implication_l1261_126159

-- Given propositions p and q
variables (p q : Prop)

-- Proving the converse of "if p then q" is "if q then p"

theorem converse_of_implication (h : p → q) : q → p :=
sorry

end NUMINAMATH_GPT_converse_of_implication_l1261_126159


namespace NUMINAMATH_GPT_apples_distribution_l1261_126161

theorem apples_distribution (total_apples : ℕ) (rotten_apples : ℕ) (boxes : ℕ) (remaining_apples : ℕ) (apples_per_box : ℕ) :
  total_apples = 40 →
  rotten_apples = 4 →
  boxes = 4 →
  remaining_apples = total_apples - rotten_apples →
  apples_per_box = remaining_apples / boxes →
  apples_per_box = 9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_apples_distribution_l1261_126161


namespace NUMINAMATH_GPT_arithmetic_expression_value_l1261_126119

def mixed_to_frac (a b c : ℕ) : ℚ := a + b / c

theorem arithmetic_expression_value :
  ( ( (mixed_to_frac 5 4 45 - mixed_to_frac 4 1 6) / mixed_to_frac 5 8 15 ) / 
    ( (mixed_to_frac 4 2 3 + 3 / 4) * mixed_to_frac 3 9 13 ) * mixed_to_frac 34 2 7 + 
    (3 / 10 / (1 / 100) / 70) + 2 / 7 ) = 1 :=
by
  -- We need to convert the mixed numbers to fractions using mixed_to_frac
  -- Then, we simplify step-by-step as in the problem solution, but for now we just use sorry
  sorry

end NUMINAMATH_GPT_arithmetic_expression_value_l1261_126119


namespace NUMINAMATH_GPT_probability_of_picking_dumpling_with_egg_l1261_126142

-- Definitions based on the conditions
def total_dumplings : ℕ := 10
def dumplings_with_eggs : ℕ := 3

-- The proof statement
theorem probability_of_picking_dumpling_with_egg :
  (dumplings_with_eggs : ℚ) / total_dumplings = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_picking_dumpling_with_egg_l1261_126142


namespace NUMINAMATH_GPT_line_intersects_circle_l1261_126172

noncomputable def diameter : ℝ := 8
noncomputable def radius : ℝ := diameter / 2
noncomputable def center_to_line_distance : ℝ := 3

theorem line_intersects_circle :
  center_to_line_distance < radius → True :=
by {
  /- The proof would go here, but for now, we use sorry. -/
  sorry
}

end NUMINAMATH_GPT_line_intersects_circle_l1261_126172


namespace NUMINAMATH_GPT_tree_height_at_end_of_2_years_l1261_126181

-- Conditions:
-- 1. The tree tripled its height every year.
-- 2. The tree reached a height of 243 feet at the end of 5 years.
theorem tree_height_at_end_of_2_years (h5 : ℕ) (H5 : h5 = 243) : 
  ∃ h2, h2 = 9 := 
by sorry

end NUMINAMATH_GPT_tree_height_at_end_of_2_years_l1261_126181


namespace NUMINAMATH_GPT_sum_of_S_values_l1261_126180

noncomputable def a : ℕ := 32
noncomputable def b1 : ℕ := 16 -- When M = 73
noncomputable def c : ℕ := 25
noncomputable def b2 : ℕ := 89 -- When M = 146
noncomputable def x1 : ℕ := 14 -- When M = 73
noncomputable def x2 : ℕ := 7 -- When M = 146
noncomputable def y1 : ℕ := 3 -- When M = 73
noncomputable def y2 : ℕ := 54 -- When M = 146
noncomputable def z1 : ℕ := 8 -- When M = 73
noncomputable def z2 : ℕ := 4 -- When M = 146

theorem sum_of_S_values :
  let M1 := a + b1 + c
  let M2 := a + b2 + c
  let S1 := M1 + x1 + y1 + z1
  let S2 := M2 + x2 + y2 + z2
  (S1 = 98) ∧ (S2 = 211) ∧ (S1 + S2 = 309) := by
  sorry

end NUMINAMATH_GPT_sum_of_S_values_l1261_126180


namespace NUMINAMATH_GPT_largest_is_three_l1261_126153

variable (p q r : ℝ)

def cond1 : Prop := p + q + r = 3
def cond2 : Prop := p * q + p * r + q * r = 1
def cond3 : Prop := p * q * r = -6

theorem largest_is_three
  (h1 : cond1 p q r)
  (h2 : cond2 p q r)
  (h3 : cond3 p q r) :
  p = 3 ∨ q = 3 ∨ r = 3 := sorry

end NUMINAMATH_GPT_largest_is_three_l1261_126153


namespace NUMINAMATH_GPT_permutations_without_HMMT_l1261_126171

noncomputable def factorial (n : ℕ) : ℕ :=
  Nat.factorial n

noncomputable def multinomial (n : ℕ) (k1 k2 k3 : ℕ) : ℕ :=
  factorial n / (factorial k1 * factorial k2 * factorial k3)

theorem permutations_without_HMMT :
  let total_permutations := multinomial 8 2 2 4
  let block_permutations := multinomial 5 1 1 2
  (total_permutations - block_permutations + 1) = 361 :=
by
  sorry

end NUMINAMATH_GPT_permutations_without_HMMT_l1261_126171


namespace NUMINAMATH_GPT_expression_eq_l1261_126198

variable {α β γ δ p q : ℝ}

-- Conditions from the problem
def roots_eq1 (α β p : ℝ) : Prop := ∀ x : ℝ, (x - α) * (x - β) = x^2 + p*x - 1
def roots_eq2 (γ δ q : ℝ) : Prop := ∀ x : ℝ, (x - γ) * (x - δ) = x^2 + q*x + 1

-- The proof statement where the expression is equated to p^2 - q^2
theorem expression_eq (h1: roots_eq1 α β p) (h2: roots_eq2 γ δ q) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = p^2 - q^2 := sorry

end NUMINAMATH_GPT_expression_eq_l1261_126198


namespace NUMINAMATH_GPT_eccentricity_is_two_l1261_126104

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 3) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem eccentricity_is_two (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 3) : 
  eccentricity_of_hyperbola a b h1 h2 h3 = 2 := 
  sorry

end NUMINAMATH_GPT_eccentricity_is_two_l1261_126104


namespace NUMINAMATH_GPT_domain_of_f_l1261_126137

def domain (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
∀ x, f x ∈ D

noncomputable def f (x : ℝ) : ℝ := 1 / (x + 2)

theorem domain_of_f :
  domain f {y | y ≠ -2} :=
by sorry

end NUMINAMATH_GPT_domain_of_f_l1261_126137


namespace NUMINAMATH_GPT_monthly_rent_calc_l1261_126101

def monthly_rent (length width annual_rent_per_sq_ft : ℕ) : ℕ :=
  (length * width * annual_rent_per_sq_ft) / 12

theorem monthly_rent_calc :
  monthly_rent 10 8 360 = 2400 := 
  sorry

end NUMINAMATH_GPT_monthly_rent_calc_l1261_126101


namespace NUMINAMATH_GPT_power_function_even_l1261_126170

-- Define the function and its properties
def f (x : ℝ) (α : ℤ) : ℝ := x ^ (Int.toNat α)

-- State the theorem with given conditions
theorem power_function_even (α : ℤ) 
    (h : f 1 α ^ 2 + f (-1) α ^ 2 = 2 * (f 1 α + f (-1) α - 1)) : 
    ∀ x : ℝ, f x α = f (-x) α :=
by
  sorry

end NUMINAMATH_GPT_power_function_even_l1261_126170


namespace NUMINAMATH_GPT_magic_triangle_max_sum_l1261_126155

theorem magic_triangle_max_sum :
  ∃ (a b c d e f : ℕ), ((a = 5 ∨ a = 6 ∨ a = 7 ∨ a = 8 ∨ a = 9 ∨ a = 10) ∧
                        (b = 5 ∨ b = 6 ∨ b = 7 ∨ b = 8 ∨ b = 9 ∨ b = 10) ∧
                        (c = 5 ∨ c = 6 ∨ c = 7 ∨ c = 8 ∨ c = 9 ∨ c = 10) ∧
                        (d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8 ∨ d = 9 ∨ d = 10) ∧
                        (e = 5 ∨ e = 6 ∨ e = 7 ∨ e = 8 ∨ e = 9 ∨ e = 10) ∧
                        (f = 5 ∨ f = 6 ∨ f = 7 ∨ f = 8 ∨ f = 9 ∨ f = 10) ∧
                        (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧
                        (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧
                        (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧
                        (d ≠ e) ∧ (d ≠ f) ∧
                        (e ≠ f) ∧
                        (a + b + c = 24) ∧ (c + d + e = 24) ∧ (e + f + a = 24)) :=
sorry

end NUMINAMATH_GPT_magic_triangle_max_sum_l1261_126155


namespace NUMINAMATH_GPT_length_PX_l1261_126102

theorem length_PX (CX DP PW PX : ℕ) (hCX : CX = 60) (hDP : DP = 20) (hPW : PW = 40)
  (parallel_CD_WX : true)  -- We use a boolean to denote the parallel condition for simplicity
  (h1 : DP + PW = CX)  -- The sum of the segments from point C through P to point X
  (h2 : DP * 2 = PX)  -- The ratio condition derived from the similarity of triangles
  : PX = 40 := 
by
  -- using the given conditions and h2 to solve for PX
  sorry

end NUMINAMATH_GPT_length_PX_l1261_126102


namespace NUMINAMATH_GPT_mod_inverse_sum_l1261_126191

theorem mod_inverse_sum :
  ∃ a b : ℕ, (5 * a ≡ 1 [MOD 21]) ∧ (b = (a * a) % 21) ∧ ((a + b) % 21 = 9) :=
by
  sorry

end NUMINAMATH_GPT_mod_inverse_sum_l1261_126191


namespace NUMINAMATH_GPT_non_participating_members_l1261_126146

noncomputable def members := 35
noncomputable def badminton_players := 15
noncomputable def tennis_players := 18
noncomputable def both_players := 3

theorem non_participating_members : 
  members - (badminton_players + tennis_players - both_players) = 5 := by
  sorry

end NUMINAMATH_GPT_non_participating_members_l1261_126146


namespace NUMINAMATH_GPT_coefficient_of_linear_term_l1261_126189

def polynomial (x : ℝ) := x^2 - 2 * x - 3

theorem coefficient_of_linear_term : (∀ x : ℝ, polynomial x = x^2 - 2 * x - 3) → -2 = -2 := by
  intro h
  sorry

end NUMINAMATH_GPT_coefficient_of_linear_term_l1261_126189


namespace NUMINAMATH_GPT_amount_paid_Y_l1261_126114

theorem amount_paid_Y (X Y : ℝ) (h1 : X + Y = 330) (h2 : X = 1.2 * Y) : Y = 150 := 
by
  sorry

end NUMINAMATH_GPT_amount_paid_Y_l1261_126114


namespace NUMINAMATH_GPT_n_minus_m_l1261_126125

theorem n_minus_m (m n : ℤ) (h_m : m - 2 = 3) (h_n : n + 1 = 2) : n - m = -4 := sorry

end NUMINAMATH_GPT_n_minus_m_l1261_126125
