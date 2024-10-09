import Mathlib

namespace units_digit_2_pow_2010_5_pow_1004_14_pow_1002_l1636_163698

theorem units_digit_2_pow_2010_5_pow_1004_14_pow_1002 :
  (2^2010 * 5^1004 * 14^1002) % 10 = 0 := by
sorry

end units_digit_2_pow_2010_5_pow_1004_14_pow_1002_l1636_163698


namespace seth_sold_candy_bars_l1636_163684

theorem seth_sold_candy_bars (max_sold : ℕ) (seth_sold : ℕ) 
  (h1 : max_sold = 24) 
  (h2 : seth_sold = 3 * max_sold + 6) : 
  seth_sold = 78 := 
by sorry

end seth_sold_candy_bars_l1636_163684


namespace max_difference_y_coords_l1636_163691

noncomputable def maximumDifference : ℝ :=
  (4 * Real.sqrt 6) / 9

theorem max_difference_y_coords :
  let f1 (x : ℝ) := 3 - 2 * x^2 + x^3
  let f2 (x : ℝ) := 1 + x^2 + x^3
  let x1 := Real.sqrt (2/3)
  let x2 := - Real.sqrt (2/3)
  let y1 := f1 x1
  let y2 := f1 x2
  |y1 - y2| = maximumDifference := sorry

end max_difference_y_coords_l1636_163691


namespace trig_identity_30deg_l1636_163604

theorem trig_identity_30deg :
  let t30 := Real.tan (Real.pi / 6)
  let s30 := Real.sin (Real.pi / 6)
  let c30 := Real.cos (Real.pi / 6)
  t30 = (Real.sqrt 3) / 3 ∧ s30 = 1 / 2 ∧ c30 = (Real.sqrt 3) / 2 →
  t30 + 4 * s30 + 2 * c30 = (2 * (Real.sqrt 3) + 3) / 3 := 
by
  intros
  sorry

end trig_identity_30deg_l1636_163604


namespace grounded_days_for_lying_l1636_163622

def extra_days_per_grade_below_b : ℕ := 3
def grades_below_b : ℕ := 4
def total_days_grounded : ℕ := 26

theorem grounded_days_for_lying : 
  (total_days_grounded - (grades_below_b * extra_days_per_grade_below_b) = 14) := 
by 
  sorry

end grounded_days_for_lying_l1636_163622


namespace lara_has_largest_answer_l1636_163686

/-- Define the final result for John, given his operations --/
def final_john (n : ℕ) : ℕ :=
  let add_three := n + 3
  let double := add_three * 2
  double - 4

/-- Define the final result for Lara, given her operations --/
def final_lara (n : ℕ) : ℕ :=
  let triple := n * 3
  let add_five := triple + 5
  add_five - 6

/-- Define the final result for Miguel, given his operations --/
def final_miguel (n : ℕ) : ℕ :=
  let double := n * 2
  let subtract_two := double - 2
  subtract_two + 2

/-- Main theorem to be proven --/
theorem lara_has_largest_answer :
  final_lara 12 > final_john 12 ∧ final_lara 12 > final_miguel 12 :=
by {
  sorry
}

end lara_has_largest_answer_l1636_163686


namespace two_thirds_of_5_times_9_l1636_163669

theorem two_thirds_of_5_times_9 : (2 / 3) * (5 * 9) = 30 :=
by
  sorry

end two_thirds_of_5_times_9_l1636_163669


namespace sum_of_squares_and_products_l1636_163621

theorem sum_of_squares_and_products
  (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x^2 + y^2 + z^2 = 52) (h5 : x * y + y * z + z * x = 24) :
  x + y + z = 10 := 
by
  sorry

end sum_of_squares_and_products_l1636_163621


namespace ratio_x_y_l1636_163652

theorem ratio_x_y (x y : ℚ) (h : (14 * x - 5 * y) / (17 * x - 3 * y) = 2 / 7) : x / y = 29 / 64 :=
by
  sorry

end ratio_x_y_l1636_163652


namespace certain_number_existence_l1636_163603

theorem certain_number_existence : ∃ x : ℝ, (102 * 102) + (x * x) = 19808 ∧ x = 97 := by
  sorry

end certain_number_existence_l1636_163603


namespace fraction_of_a_mile_additional_charge_l1636_163661

-- Define the conditions
def initial_fee : ℚ := 2.25
def charge_per_fraction : ℚ := 0.25
def total_charge : ℚ := 4.50
def total_distance : ℚ := 3.6

-- Define the problem statement to prove
theorem fraction_of_a_mile_additional_charge :
  initial_fee = 2.25 →
  charge_per_fraction = 0.25 →
  total_charge = 4.50 →
  total_distance = 3.6 →
  total_distance - (total_charge - initial_fee) = 1.35 :=
by
  intros
  sorry

end fraction_of_a_mile_additional_charge_l1636_163661


namespace total_toothpicks_in_grid_l1636_163655

theorem total_toothpicks_in_grid (l w : ℕ) (h₁ : l = 50) (h₂ : w = 20) : 
  (l + 1) * w + (w + 1) * l + 2 * (l * w) = 4070 :=
by
  sorry

end total_toothpicks_in_grid_l1636_163655


namespace percentage_increase_book_price_l1636_163632

theorem percentage_increase_book_price (OldP NewP : ℕ) (hOldP : OldP = 300) (hNewP : NewP = 330) :
  ((NewP - OldP : ℕ) / OldP : ℚ) * 100 = 10 := by
  sorry

end percentage_increase_book_price_l1636_163632


namespace area_ratio_of_circles_l1636_163664

theorem area_ratio_of_circles (R_A R_B : ℝ) 
  (h1 : (60 / 360) * (2 * Real.pi * R_A) = (40 / 360) * (2 * Real.pi * R_B)) :
  (Real.pi * R_A ^ 2) / (Real.pi * R_B ^ 2) = 9 / 4 := 
sorry

end area_ratio_of_circles_l1636_163664


namespace boxes_filled_l1636_163651

noncomputable def bags_per_box := 6
noncomputable def balls_per_bag := 8
noncomputable def total_balls := 720

theorem boxes_filled (h1 : balls_per_bag = 8) (h2 : bags_per_box = 6) (h3 : total_balls = 720) :
  (total_balls / balls_per_bag) / bags_per_box = 15 :=
by
  sorry

end boxes_filled_l1636_163651


namespace inequalities_not_hold_l1636_163675

theorem inequalities_not_hold (x y z a b c : ℝ) (h1 : x < a) (h2 : y < b) (h3 : z < c) : 
  ¬ (x * y + y * z + z * x < a * b + b * c + c * a) ∧ 
  ¬ (x^2 + y^2 + z^2 < a^2 + b^2 + c^2) ∧ 
  ¬ (x * y * z < a * b * c) := 
sorry

end inequalities_not_hold_l1636_163675


namespace casey_correct_result_l1636_163609

variable (x : ℕ)

def incorrect_divide (x : ℕ) := x / 7
def incorrect_subtract (x : ℕ) := x - 20
def incorrect_result := 19

def reverse_subtract (x : ℕ) := x + 20
def reverse_divide (x : ℕ) := x * 7

def correct_multiply (x : ℕ) := x * 7
def correct_add (x : ℕ) := x + 20

theorem casey_correct_result (x : ℕ) (h : reverse_divide (reverse_subtract incorrect_result) = x) : correct_add (correct_multiply x) = 1931 :=
by
  sorry

end casey_correct_result_l1636_163609


namespace isosceles_triangle_perimeter_l1636_163615

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b) ∨ (b = c) ∨ (a = c)

def is_valid_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem isosceles_triangle_perimeter {a b : ℕ} (h₁ : is_isosceles_triangle a b b) (h₂ : is_valid_triangle a b b) : a + b + b = 15 :=
  sorry

end isosceles_triangle_perimeter_l1636_163615


namespace B_contribution_l1636_163608

-- Define the conditions
def capitalA : ℝ := 3500
def monthsA : ℕ := 12
def monthsB : ℕ := 7
def profit_ratio_A : ℕ := 2
def profit_ratio_B : ℕ := 3

-- Statement: B's contribution to the capital
theorem B_contribution :
  (capitalA * monthsA * profit_ratio_B) / (monthsB * profit_ratio_A) = 4500 := by
  sorry

end B_contribution_l1636_163608


namespace Josanna_seventh_test_score_l1636_163646

theorem Josanna_seventh_test_score (scores : List ℕ) (h_scores : scores = [95, 85, 75, 65, 90, 70])
                                   (average_increase : ℕ) (h_average_increase : average_increase = 5) :
                                   ∃ x, (List.sum scores + x) / (List.length scores + 1) = (List.sum scores) / (List.length scores) + average_increase := 
by
  sorry

end Josanna_seventh_test_score_l1636_163646


namespace binary_101_eq_5_l1636_163660

theorem binary_101_eq_5 : 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 5 := 
by
  sorry

end binary_101_eq_5_l1636_163660


namespace inequality_proof_l1636_163633

theorem inequality_proof (x y z : ℝ) 
  (h₁ : x + y + z = 0) 
  (h₂ : |x| + |y| + |z| ≤ 1) : 
  x + y / 3 + z / 5 ≤ 2 / 5 :=
sorry

end inequality_proof_l1636_163633


namespace solve_equation_l1636_163673

theorem solve_equation : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end solve_equation_l1636_163673


namespace red_bowl_values_possible_l1636_163624

theorem red_bowl_values_possible (r b y : ℕ) 
(h1 : r + b + y = 27)
(h2 : 15 * r + 3 * b + 18 * y = 378) : 
  r = 11 ∨ r = 16 ∨ r = 21 := 
  sorry

end red_bowl_values_possible_l1636_163624


namespace sum_of_repeating_decimals_l1636_163685

-- Definitions of the repeating decimals as fractions
def x : ℚ := 1 / 9
def y : ℚ := 2 / 99
def z : ℚ := 3 / 999

-- Theorem stating the sum of these fractions is equal to the expected result
theorem sum_of_repeating_decimals : x + y + z = 164 / 1221 := 
  sorry

end sum_of_repeating_decimals_l1636_163685


namespace exponential_decreasing_range_l1636_163640

theorem exponential_decreasing_range (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x y : ℝ, x < y → a^y < a^x) : 0 < a ∧ a < 1 :=
by sorry

end exponential_decreasing_range_l1636_163640


namespace skiing_ratio_l1636_163628

theorem skiing_ratio (S : ℕ) (H1 : 4000 ≤ 12000) (H2 : S + 4000 = 12000) : S / 4000 = 2 :=
by {
  sorry
}

end skiing_ratio_l1636_163628


namespace definite_integral_ln_l1636_163645

open Real

theorem definite_integral_ln (a b : ℝ) (h₁ : a = 1) (h₂ : b = exp 1) :
  ∫ x in a..b, (1 + log x) = exp 1 := by
  sorry

end definite_integral_ln_l1636_163645


namespace red_pigment_contribution_l1636_163656

theorem red_pigment_contribution :
  ∀ (G : ℝ), (2 * G + G + 3 * G = 24) →
  (0.6 * (2 * G) + 0.5 * (3 * G) = 10.8) :=
by
  intro G
  intro h1
  sorry

end red_pigment_contribution_l1636_163656


namespace hexagon_coloring_count_l1636_163648

def num_possible_colorings : Nat :=
by
  /- There are 7 choices for first vertex A.
     Once A is chosen, there are 6 choices for the remaining vertices B, C, D, E, F considering the diagonal restrictions. -/
  let total_colorings := 7 * 6 ^ 5
  let restricted_colorings := 7 * 6 ^ 3
  let valid_colorings := total_colorings - restricted_colorings
  exact valid_colorings

theorem hexagon_coloring_count : num_possible_colorings = 52920 :=
  by
    /- Computation steps above show that the number of valid colorings is 52920 -/
    sorry   -- Proof computation already indicated

end hexagon_coloring_count_l1636_163648


namespace inequality_proof_l1636_163620

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  (1 + a) * (1 + b) * (1 + c) ≥ 8 * (1 - a) * (1 - b) * (1 - c) :=
by
  sorry

end inequality_proof_l1636_163620


namespace exists_line_through_ellipse_diameter_circle_origin_l1636_163676

theorem exists_line_through_ellipse_diameter_circle_origin :
  ∃ m : ℝ, (m = (4 * Real.sqrt 3) / 3 ∨ m = -(4 * Real.sqrt 3) / 3) ∧
  ∀ (x y : ℝ), (x^2 + 2 * y^2 = 8) → (y = x + m) → (x^2 + (x + m)^2 = 8) :=
by
  sorry

end exists_line_through_ellipse_diameter_circle_origin_l1636_163676


namespace complex_product_l1636_163614

theorem complex_product (z1 z2 : ℂ) (h1 : Complex.abs z1 = 1) (h2 : Complex.abs z2 = 1) 
(h3 : z1 + z2 = -7/5 + (1/5) * Complex.I) : 
  z1 * z2 = 24/25 - (7/25) * Complex.I :=
by
  sorry

end complex_product_l1636_163614


namespace total_juice_drunk_l1636_163600

noncomputable def juiceConsumption (samDrink benDrink : ℕ) (samConsRatio benConsRatio : ℚ) : ℚ :=
  let samConsumed := samConsRatio * samDrink
  let samRemaining := samDrink - samConsumed
  let benConsumed := benConsRatio * benDrink
  let benRemaining := benDrink - benConsumed
  let benToSam := (1 / 2) * benRemaining + 1
  let samTotal := samConsumed + benToSam
  let benTotal := benConsumed - benToSam
  samTotal + benTotal

theorem total_juice_drunk : juiceConsumption 12 20 (2 / 3 : ℚ) (2 / 3 : ℚ) = 32 :=
sorry

end total_juice_drunk_l1636_163600


namespace value_of_d_l1636_163666

theorem value_of_d (d : ℝ) (h : ∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) : d = 5 :=
by
  sorry

end value_of_d_l1636_163666


namespace haley_trees_grown_after_typhoon_l1636_163623

def original_trees := 9
def trees_died := 4
def current_trees := 10

theorem haley_trees_grown_after_typhoon (newly_grown_trees : ℕ) :
  (original_trees - trees_died) + newly_grown_trees = current_trees → newly_grown_trees = 5 :=
by
  sorry

end haley_trees_grown_after_typhoon_l1636_163623


namespace find_point_C_l1636_163667

noncomputable def point_on_z_axis (z : ℝ) : ℝ × ℝ × ℝ := (0, 0, z)
def point_A : ℝ × ℝ × ℝ := (1, 0, 2)
def point_B : ℝ × ℝ × ℝ := (1, 1, 1)

theorem find_point_C :
  ∃ C : ℝ × ℝ × ℝ, (C = point_on_z_axis 1) ∧ (dist C point_A = dist C point_B) :=
by
  sorry

end find_point_C_l1636_163667


namespace domain_ln_l1636_163613

theorem domain_ln (x : ℝ) : (1 - 2 * x > 0) ↔ x < (1 / 2) :=
by
  sorry

end domain_ln_l1636_163613


namespace factor_product_modulo_l1636_163647

theorem factor_product_modulo (h1 : 2021 % 23 = 21) (h2 : 2022 % 23 = 22) (h3 : 2023 % 23 = 0) (h4 : 2024 % 23 = 1) (h5 : 2025 % 23 = 2) :
  (2021 * 2022 * 2023 * 2024 * 2025) % 23 = 0 :=
by
  sorry

end factor_product_modulo_l1636_163647


namespace fraction_correct_l1636_163679

-- Define the total number of coins.
def total_coins : ℕ := 30

-- Define the number of states that joined the union in the decade 1800 through 1809.
def states_1800_1809 : ℕ := 4

-- Define the fraction of coins representing states joining in the decade 1800 through 1809.
def fraction_coins_1800_1809 : ℚ := states_1800_1809 / total_coins

-- The theorem statement that needs to be proved.
theorem fraction_correct : fraction_coins_1800_1809 = (2 / 15) := 
by
  sorry

end fraction_correct_l1636_163679


namespace cost_of_50_tulips_l1636_163619

theorem cost_of_50_tulips (c : ℕ → ℝ) :
  (∀ n : ℕ, n ≤ 40 → c n = n * (36 / 18)) ∧
  (∀ n : ℕ, n > 40 → c n = (40 * (36 / 18) + (n - 40) * (36 / 18)) * 0.9) ∧
  (c 18 = 36) →
  c 50 = 90 := sorry

end cost_of_50_tulips_l1636_163619


namespace wrestling_match_student_count_l1636_163674

theorem wrestling_match_student_count (n : ℕ) (h : n * (n - 1) / 2 = 91) : n = 14 := by
  sorry

end wrestling_match_student_count_l1636_163674


namespace decimal_to_fraction_l1636_163650

theorem decimal_to_fraction : (0.3 + (0.24 - 0.24 / 100)) = (19 / 33) :=
by
  sorry

end decimal_to_fraction_l1636_163650


namespace product_sum_divisibility_l1636_163602

theorem product_sum_divisibility (m n : ℕ) (h : (m + n) ∣ (m * n)) (hm : 0 < m) (hn : 0 < n) : m + n ≤ n^2 :=
sorry

end product_sum_divisibility_l1636_163602


namespace crackers_per_person_l1636_163626

variable (darrenA : Nat)
variable (darrenB : Nat)
variable (aCrackersPerBox : Nat)
variable (bCrackersPerBox : Nat)
variable (calvinA : Nat)
variable (calvinB : Nat)
variable (totalPeople : Nat)

-- Definitions based on the conditions
def totalDarrenCrackers := darrenA * aCrackersPerBox + darrenB * bCrackersPerBox
def totalCalvinA := 2 * darrenA - 1
def totalCalvinCrackers := totalCalvinA * aCrackersPerBox + darrenB * bCrackersPerBox
def totalCrackers := totalDarrenCrackers + totalCalvinCrackers
def crackersPerPerson := totalCrackers / totalPeople

-- The theorem to prove the question equals the answer given the conditions
theorem crackers_per_person :
  darrenA = 4 →
  darrenB = 2 →
  aCrackersPerBox = 24 →
  bCrackersPerBox = 30 →
  calvinA = 7 →
  calvinB = darrenB →
  totalPeople = 5 →
  crackersPerPerson = 76 :=
by
  intros
  sorry

end crackers_per_person_l1636_163626


namespace shoveling_hours_l1636_163625

def initial_rate := 25

def rate_decrease := 2

def snow_volume := 6 * 12 * 3

def shoveling_rate (hour : ℕ) : ℕ :=
  if hour = 0 then initial_rate
  else initial_rate - rate_decrease * hour

def cumulative_snow (hour : ℕ) : ℕ :=
  if hour = 0 then snow_volume - shoveling_rate 0
  else cumulative_snow (hour - 1) - shoveling_rate hour

theorem shoveling_hours : cumulative_snow 12 ≠ 0 ∧ cumulative_snow 13 = 47 := by
  sorry

end shoveling_hours_l1636_163625


namespace find_second_number_l1636_163682

-- Define the given number
def given_number := 220070

-- Define the constants in the problem
def constant_555 := 555
def remainder := 70

-- Define the second number (our unknown)
variable (x : ℕ)

-- Define the condition as an equation
def condition : Prop :=
  given_number = (constant_555 + x) * 2 * (x - constant_555) + remainder

-- The theorem to prove that the second number is 343
theorem find_second_number : ∃ x : ℕ, condition x ∧ x = 343 :=
sorry

end find_second_number_l1636_163682


namespace inequality_proof_l1636_163662

variable (x y : ℝ)
variable (hx : 0 < x) (hy : 0 < y)

theorem inequality_proof :
  x / Real.sqrt y + y / Real.sqrt x ≥ Real.sqrt x + Real.sqrt y :=
sorry

end inequality_proof_l1636_163662


namespace find_eighth_term_l1636_163611

noncomputable def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  a + n * d

theorem find_eighth_term (a d : ℕ) :
  (arithmetic_sequence a d 0) + 
  (arithmetic_sequence a d 1) + 
  (arithmetic_sequence a d 2) + 
  (arithmetic_sequence a d 3) + 
  (arithmetic_sequence a d 4) + 
  (arithmetic_sequence a d 5) = 21 ∧
  arithmetic_sequence a d 6 = 7 →
  arithmetic_sequence a d 7 = 8 :=
by
  sorry

end find_eighth_term_l1636_163611


namespace sandwich_total_calories_l1636_163601

-- Given conditions
def bacon_calories := 2 * 125
def bacon_percentage := 20 / 100

-- Statement to prove
theorem sandwich_total_calories :
  bacon_calories / bacon_percentage = 1250 := 
sorry

end sandwich_total_calories_l1636_163601


namespace expansion_contains_x4_l1636_163641

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def expansion_term (x : ℂ) (i : ℂ) : ℂ :=
  binomial_coeff 6 2 * x^4 * i^2

theorem expansion_contains_x4 (x i : ℂ) (hi : i = Complex.I) : 
  expansion_term x i = -15 * x^4 := by
  sorry

end expansion_contains_x4_l1636_163641


namespace number_of_tiles_l1636_163654

theorem number_of_tiles (floor_length : ℝ) (floor_width : ℝ) (tile_length : ℝ) (tile_width : ℝ) 
  (h1 : floor_length = 9) 
  (h2 : floor_width = 12) 
  (h3 : tile_length = 1 / 2) 
  (h4 : tile_width = 2 / 3) 
  : (floor_length * floor_width) / (tile_length * tile_width) = 324 := 
by
  sorry

end number_of_tiles_l1636_163654


namespace min_red_chips_l1636_163612

theorem min_red_chips (w b r : ℕ) 
  (h1 : b ≥ w / 3) 
  (h2 : b ≤ r / 4) 
  (h3 : w + b ≥ 72) :
  72 ≤ r :=
by
  sorry

end min_red_chips_l1636_163612


namespace family_raised_percentage_l1636_163692

theorem family_raised_percentage :
  ∀ (total_funds friends_percentage own_savings family_funds remaining_funds : ℝ),
    total_funds = 10000 →
    friends_percentage = 0.40 →
    own_savings = 4200 →
    remaining_funds = total_funds - (friends_percentage * total_funds) →
    family_funds = remaining_funds - own_savings →
    (family_funds / remaining_funds) * 100 = 30 :=
by
  intros total_funds friends_percentage own_savings family_funds remaining_funds
  intros h_total_funds h_friends_percentage h_own_savings h_remaining_funds h_family_funds
  sorry

end family_raised_percentage_l1636_163692


namespace hexahedron_octahedron_ratio_l1636_163631

open Real

theorem hexahedron_octahedron_ratio (a : ℝ) (h_a_pos : 0 < a) :
  let r1 := (sqrt 6 * a / 9)
  let r2 := (sqrt 6 * a / 6)
  let ratio := r1 / r2
  ∃ m n : ℕ, gcd m n = 1 ∧ (ratio = (m : ℝ) / (n : ℝ)) ∧ (m * n = 6) :=
by {
  sorry
}

end hexahedron_octahedron_ratio_l1636_163631


namespace at_least_26_equal_differences_l1636_163668

theorem at_least_26_equal_differences (x : Fin 102 → ℕ) (h : ∀ i j, i < j → x i < x j) (h' : ∀ i, x i < 255) :
  (∃ d : Fin 101 → ℕ, ∃ s : Finset ℕ, s.card ≥ 26 ∧ (∀ i, d i = x i.succ - x i) ∧ ∃ i j, i ≠ j ∧ (d i = d j)) :=
by {
  sorry
}

end at_least_26_equal_differences_l1636_163668


namespace james_meat_sales_l1636_163694

theorem james_meat_sales
  (beef_pounds : ℕ)
  (pork_pounds : ℕ)
  (meat_per_meal : ℝ)
  (meal_price : ℝ)
  (total_meat : ℝ)
  (number_of_meals : ℝ)
  (total_money : ℝ)
  (h1 : beef_pounds = 20)
  (h2 : pork_pounds = beef_pounds / 2)
  (h3 : meat_per_meal = 1.5)
  (h4 : meal_price = 20)
  (h5 : total_meat = beef_pounds + pork_pounds)
  (h6 : number_of_meals = total_meat / meat_per_meal)
  (h7 : total_money = number_of_meals * meal_price) :
  total_money = 400 := by
  sorry

end james_meat_sales_l1636_163694


namespace cheryl_used_total_material_correct_amount_l1636_163670

def material_used (initial leftover : ℚ) : ℚ := initial - leftover

def total_material_used 
  (initial_a initial_b initial_c leftover_a leftover_b leftover_c : ℚ) : ℚ :=
  material_used initial_a leftover_a + material_used initial_b leftover_b + material_used initial_c leftover_c

theorem cheryl_used_total_material_correct_amount :
  total_material_used (2/9) (1/8) (3/10) (4/18) (1/12) (3/15) = 17/120 :=
by
  sorry

end cheryl_used_total_material_correct_amount_l1636_163670


namespace black_pieces_more_than_white_l1636_163680

theorem black_pieces_more_than_white (B W : ℕ) 
  (h₁ : (B - 1) * 7 = 9 * W)
  (h₂ : B * 5 = 7 * (W - 1)) :
  B - W = 7 :=
sorry

end black_pieces_more_than_white_l1636_163680


namespace find_xy_l1636_163649

variable (x y : ℚ)

theorem find_xy (h1 : 1/x + 3/y = 1/2) (h2 : 1/y - 3/x = 1/3) : 
    x = -20 ∧ y = 60/11 := 
by
  sorry

end find_xy_l1636_163649


namespace length_of_XY_in_triangle_XYZ_l1636_163690

theorem length_of_XY_in_triangle_XYZ :
  ∀ (XYZ : Type) (X Y Z : XYZ) (angle : XYZ → XYZ → XYZ → ℝ) (length : XYZ → XYZ → ℝ),
  angle X Z Y = 30 ∧ angle Y X Z = 90 ∧ length X Z = 8 → length X Y = 16 :=
by sorry

end length_of_XY_in_triangle_XYZ_l1636_163690


namespace common_rational_root_neg_not_integer_l1636_163695

theorem common_rational_root_neg_not_integer : 
  ∃ (p : ℚ), (p < 0) ∧ (¬ ∃ (z : ℤ), p = z) ∧ 
  (50 * p^4 + a * p^3 + b * p^2 + c * p + 20 = 0) ∧ 
  (20 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 50 = 0) := 
sorry

end common_rational_root_neg_not_integer_l1636_163695


namespace find_ratio_b_over_a_l1636_163606

theorem find_ratio_b_over_a (a b : ℝ)
  (h1 : ∀ x, deriv (fun x => a * x^2 + b) x = 2 * a * x)
  (h2 : deriv (fun x => a * x^2 + b) 1 = 2)
  (h3 : a * 1^2 + b = 3) : b / a = 2 := 
sorry

end find_ratio_b_over_a_l1636_163606


namespace intersection_M_N_l1636_163638

-- Definitions of sets M and N
def M : Set ℕ := {1, 2, 5}
def N : Set ℕ := {x | x ≤ 2}

-- Lean statement to prove that the intersection of M and N is {1, 2}
theorem intersection_M_N : M ∩ N = {1, 2} :=
by
  sorry

end intersection_M_N_l1636_163638


namespace number_of_kids_stay_home_l1636_163636

def total_kids : ℕ := 313473
def kids_at_camp : ℕ := 38608
def kids_stay_home : ℕ := 274865

theorem number_of_kids_stay_home :
  total_kids - kids_at_camp = kids_stay_home := 
by
  -- Subtracting the number of kids who go to camp from the total number of kids
  sorry

end number_of_kids_stay_home_l1636_163636


namespace slices_needed_l1636_163699

def slices_per_sandwich : ℕ := 3
def number_of_sandwiches : ℕ := 5

theorem slices_needed : slices_per_sandwich * number_of_sandwiches = 15 :=
by {
  sorry
}

end slices_needed_l1636_163699


namespace largest_integer_modulo_l1636_163688

theorem largest_integer_modulo (a : ℤ) : a < 93 ∧ a % 7 = 4 ∧ (∀ b : ℤ, b < 93 ∧ b % 7 = 4 → b ≤ a) ↔ a = 88 :=
by
    sorry

end largest_integer_modulo_l1636_163688


namespace solve_for_x_l1636_163634

theorem solve_for_x (x : ℝ) (h : (x - 6)^4 = (1 / 16)⁻¹) : x = 8 := 
by 
  sorry

end solve_for_x_l1636_163634


namespace minimum_boxes_required_l1636_163618

theorem minimum_boxes_required 
  (total_brochures : ℕ)
  (small_box_capacity : ℕ) (small_boxes_available : ℕ)
  (medium_box_capacity : ℕ) (medium_boxes_available : ℕ)
  (large_box_capacity : ℕ) (large_boxes_available : ℕ)
  (complete_fill : ∀ (box_capacity brochures : ℕ), box_capacity ∣ brochures)
  (min_boxes_required : ℕ) :
  total_brochures = 10000 →
  small_box_capacity = 50 →
  small_boxes_available = 40 →
  medium_box_capacity = 200 →
  medium_boxes_available = 25 →
  large_box_capacity = 500 →
  large_boxes_available = 10 →
  min_boxes_required = 35 :=
by
  intros
  sorry

end minimum_boxes_required_l1636_163618


namespace tank_capacity_l1636_163672

theorem tank_capacity (fill_rate drain_rate1 drain_rate2 : ℝ)
  (initial_fullness : ℝ) (time_to_fill : ℝ) (capacity_in_liters : ℝ) :
  fill_rate = 1 / 2 ∧
  drain_rate1 = 1 / 4 ∧
  drain_rate2 = 1 / 6 ∧ 
  initial_fullness = 1 / 2 ∧ 
  time_to_fill = 60 →
  capacity_in_liters = 10000 :=
by {
  sorry
}

end tank_capacity_l1636_163672


namespace abs_sub_self_nonneg_l1636_163663

theorem abs_sub_self_nonneg (m : ℚ) : |m| - m ≥ 0 := 
sorry

end abs_sub_self_nonneg_l1636_163663


namespace jared_current_age_condition_l1636_163637

variable (t j: ℕ)

-- Conditions
def tom_current_age := 25
def tom_future_age_condition := t + 5 = 30
def jared_past_age_condition := j - 2 = 2 * (t - 2)

-- Question
theorem jared_current_age_condition : 
  (t + 5 = 30) ∧ (j - 2 = 2 * (t - 2)) → j = 48 :=
by
  sorry

end jared_current_age_condition_l1636_163637


namespace squirrel_can_catch_nut_l1636_163630

-- Define the initial distance between Gabriel and the squirrel.
def initial_distance : ℝ := 3.75

-- Define the speed of the nut.
def nut_speed : ℝ := 5.0

-- Define the jumping distance of the squirrel.
def squirrel_jump_distance : ℝ := 1.8

-- Define the acceleration due to gravity.
def gravity : ℝ := 10.0

-- Define the positions of the nut and the squirrel as functions of time.
def nut_position_x (t : ℝ) : ℝ := nut_speed * t
def squirrel_position_x : ℝ := initial_distance
def nut_position_y (t : ℝ) : ℝ := 0.5 * gravity * t^2

-- Define the squared distance between the nut and the squirrel.
def distance_squared (t : ℝ) : ℝ :=
  (nut_position_x t - squirrel_position_x)^2 + (nut_position_y t)^2

-- Prove that the minimum distance squared is less than or equal to the squirrel's jumping distance squared.
theorem squirrel_can_catch_nut : ∃ t : ℝ, distance_squared t ≤ squirrel_jump_distance^2 := by
  -- Sorry placeholder, as the proof is not required.
  sorry

end squirrel_can_catch_nut_l1636_163630


namespace intersection_A_B_union_B_C_eq_B_iff_l1636_163687

-- Definitions for the sets A, B, and C
def setA : Set ℝ := { x | x^2 - 3 * x < 0 }
def setB : Set ℝ := { x | (x + 2) * (4 - x) ≥ 0 }
def setC (a : ℝ) : Set ℝ := { x | a < x ∧ x ≤ a + 1 }

-- Proving that A ∩ B = { x | 0 < x < 3 }
theorem intersection_A_B : setA ∩ setB = { x : ℝ | 0 < x ∧ x < 3 } :=
sorry

-- Proving that B ∪ C = B implies the range of a is [-2, 3]
theorem union_B_C_eq_B_iff (a : ℝ) : (setB ∪ setC a = setB) ↔ (-2 ≤ a ∧ a ≤ 3) :=
sorry

end intersection_A_B_union_B_C_eq_B_iff_l1636_163687


namespace bobby_pancakes_left_l1636_163629

def total_pancakes : ℕ := 21
def pancakes_eaten_by_bobby : ℕ := 5
def pancakes_eaten_by_dog : ℕ := 7

theorem bobby_pancakes_left : total_pancakes - (pancakes_eaten_by_bobby + pancakes_eaten_by_dog) = 9 :=
  by
  sorry

end bobby_pancakes_left_l1636_163629


namespace Mary_current_age_l1636_163678

theorem Mary_current_age
  (M J : ℕ) 
  (h1 : J - 5 = (M - 5) + 7) 
  (h2 : J + 5 = 2 * (M + 5)) : 
  M = 2 :=
by
  /- We need to show that the current age of Mary (M) is 2
     given the conditions h1 and h2.-/
  sorry

end Mary_current_age_l1636_163678


namespace gcd_repeated_integer_l1636_163683

theorem gcd_repeated_integer (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000) :
  ∃ d, (∀ k : ℕ, k = 1001001001 * n → d = 1001001001 ∧ d ∣ k) :=
sorry

end gcd_repeated_integer_l1636_163683


namespace min_distance_l1636_163605

theorem min_distance (W : ℝ) (b : ℝ) (n : ℕ) (H_W : W = 42) (H_b : b = 3) (H_n : n = 8) : 
  ∃ d : ℝ, d = 2 ∧ (W - n * b = 9 * d) := 
by 
  -- Here should go the proof
  sorry

end min_distance_l1636_163605


namespace total_time_spent_in_hours_l1636_163671

/-- Miriam's time spent on each task in minutes. -/
def time_laundry := 30
def time_bathroom := 15
def time_room := 35
def time_homework := 40

/-- The function to convert minutes to hours. -/
def minutes_to_hours (minutes : ℕ) := minutes / 60

/-- The total time spent in minutes. -/
def total_time_minutes := time_laundry + time_bathroom + time_room + time_homework

/-- The total time spent in hours. -/
def total_time_hours := minutes_to_hours total_time_minutes

/-- The main statement to be proved: total_time_hours equals 2. -/
theorem total_time_spent_in_hours : total_time_hours = 2 := 
by
  sorry

end total_time_spent_in_hours_l1636_163671


namespace distance_focus_to_asymptote_l1636_163644

theorem distance_focus_to_asymptote (m : ℝ) (x y : ℝ) (h1 : (x^2) / 9 - (y^2) / m = 1) 
  (h2 : (Real.sqrt 14) / 3 = (Real.sqrt (9 + m)) / 3) : 
  ∃ d : ℝ, d = Real.sqrt 5 := 
by 
  sorry

end distance_focus_to_asymptote_l1636_163644


namespace find_coordinates_M_l1636_163657

open Real

theorem find_coordinates_M (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℝ) :
  ∃ (xM yM zM : ℝ), 
  xM = (x1 + x2 + x3 + x4) / 4 ∧
  yM = (y1 + y2 + y3 + y4) / 4 ∧
  zM = (z1 + z2 + z3 + z4) / 4 ∧
  (x1 - xM) + (x2 - xM) + (x3 - xM) + (x4 - xM) = 0 ∧
  (y1 - yM) + (y2 - yM) + (y3 - yM) + (y4 - yM) = 0 ∧
  (z1 - zM) + (z2 - zM) + (z3 - zM) + (z4 - zM) = 0 := by
  sorry

end find_coordinates_M_l1636_163657


namespace sum_of_squares_and_product_l1636_163658

theorem sum_of_squares_and_product
  (x y : ℕ) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end sum_of_squares_and_product_l1636_163658


namespace value_of_c_l1636_163677

-- Define the function f(x)
def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem value_of_c (a b c m : ℝ) (h₀ : ∀ x : ℝ, 0 ≤ f x a b)
  (h₁ : ∀ x : ℝ, f x a b < c ↔ m < x ∧ x < m + 6) :
  c = 9 :=
sorry

end value_of_c_l1636_163677


namespace gcf_72_108_l1636_163697

def gcf (a b : ℕ) : ℕ := 
  Nat.gcd a b

theorem gcf_72_108 : gcf 72 108 = 36 := by
  sorry

end gcf_72_108_l1636_163697


namespace negation_equiv_l1636_163642

theorem negation_equiv {x : ℝ} : 
  (¬ (x^2 < 1 → -1 < x ∧ x < 1)) ↔ (x^2 ≥ 1 → x ≥ 1 ∨ x ≤ -1) :=
by
  sorry

end negation_equiv_l1636_163642


namespace total_capsules_sold_in_2_weeks_l1636_163627

-- Define the conditions as constants
def Earnings100mgPerWeek := 80
def CostPer100mgCapsule := 5
def Earnings500mgPerWeek := 60
def CostPer500mgCapsule := 2

-- Theorem to prove the total number of capsules sold in 2 weeks
theorem total_capsules_sold_in_2_weeks : 
  (Earnings100mgPerWeek / CostPer100mgCapsule) * 2 + (Earnings500mgPerWeek / CostPer500mgCapsule) * 2 = 92 :=
by
  sorry

end total_capsules_sold_in_2_weeks_l1636_163627


namespace exists_number_added_to_sum_of_digits_gives_2014_l1636_163681

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem exists_number_added_to_sum_of_digits_gives_2014 : 
  ∃ (n : ℕ), n + sum_of_digits n = 2014 :=
sorry

end exists_number_added_to_sum_of_digits_gives_2014_l1636_163681


namespace intersection_M_N_l1636_163616

open Real

def M := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}
def N := {x : ℝ | 2 - abs x > 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by
sorry

end intersection_M_N_l1636_163616


namespace first_day_more_than_300_l1636_163643

def paperclips (n : ℕ) : ℕ := 4 * 3^n

theorem first_day_more_than_300 : ∃ n, paperclips n > 300 ∧ n = 4 := by
  sorry

end first_day_more_than_300_l1636_163643


namespace complex_pure_imaginary_l1636_163689

theorem complex_pure_imaginary (a : ℝ) 
  (h1 : a^2 + 2*a - 3 = 0) 
  (h2 : a + 3 ≠ 0) : 
  a = 1 := 
by
  sorry

end complex_pure_imaginary_l1636_163689


namespace intersection_unique_l1636_163659

noncomputable def f (x : ℝ) := 3 * Real.log x
noncomputable def g (x : ℝ) := Real.log (x + 4)

theorem intersection_unique : ∃! x, f x = g x :=
sorry

end intersection_unique_l1636_163659


namespace inequality_holds_for_unit_interval_l1636_163610

theorem inequality_holds_for_unit_interval (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
    5 * (x ^ 2 + y ^ 2) ^ 2 ≤ 4 + (x + y) ^ 4 :=
by
    sorry

end inequality_holds_for_unit_interval_l1636_163610


namespace floor_sqrt_23_squared_l1636_163639

theorem floor_sqrt_23_squared : (Nat.floor (Real.sqrt 23)) ^ 2 = 16 :=
by
  -- Proof is omitted
  sorry

end floor_sqrt_23_squared_l1636_163639


namespace one_non_congruent_triangle_with_perimeter_10_l1636_163607

def is_valid_triangle (a b c : ℕ) : Prop :=
  a < b + c ∧ b < a + c ∧ c < a + b

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 10

def are_non_congruent (a b c : ℕ) (x y z : ℕ) : Prop :=
  ¬ (a = x ∧ b = y ∧ c = z ∨ a = x ∧ b = z ∧ c = y ∨ a = y ∧ b = x ∧ c = z ∨ 
     a = y ∧ b = z ∧ c = x ∨ a = z ∧ b = x ∧ c = y ∨ a = z ∧ b = y ∧ c = x)

theorem one_non_congruent_triangle_with_perimeter_10 :
  ∃ a b c : ℕ, is_valid_triangle a b c ∧ perimeter a b c ∧
  ∀ x y z : ℕ, is_valid_triangle x y z ∧ perimeter x y z → are_non_congruent a b c x y z → false :=
sorry

end one_non_congruent_triangle_with_perimeter_10_l1636_163607


namespace max_sum_of_factors_l1636_163693

theorem max_sum_of_factors (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C)
  (h4 : A * B * C = 3003) : A + B + C ≤ 117 :=
sorry

end max_sum_of_factors_l1636_163693


namespace find_radius_of_semicircular_plot_l1636_163696

noncomputable def radius_of_semicircular_plot (π : ℝ) : ℝ :=
  let total_fence_length := 33
  let opening_length := 3
  let effective_fence_length := total_fence_length - opening_length
  let r := effective_fence_length / (π + 2)
  r

theorem find_radius_of_semicircular_plot 
  (π : ℝ) (Hπ : π = Real.pi) :
  radius_of_semicircular_plot π = 30 / (Real.pi + 2) :=
by
  unfold radius_of_semicircular_plot
  rw [Hπ]
  sorry

end find_radius_of_semicircular_plot_l1636_163696


namespace perimeter_of_one_rectangle_l1636_163635

-- Define the conditions
def is_divided_into_congruent_rectangles (s : ℕ) : Prop :=
  ∃ (height width : ℕ), height = s ∧ width = s / 4

-- Main proof statement
theorem perimeter_of_one_rectangle {s : ℕ} (h₁ : 4 * s = 144)
  (h₂ : is_divided_into_congruent_rectangles s) : 
  ∃ (perimeter : ℕ), perimeter = 90 :=
by 
  sorry

end perimeter_of_one_rectangle_l1636_163635


namespace width_of_first_sheet_paper_l1636_163653

theorem width_of_first_sheet_paper :
  ∀ (w : ℝ),
  2 * 11 * w = 2 * 4.5 * 11 + 100 → 
  w = 199 / 22 := 
by
  intro w
  intro h
  sorry

end width_of_first_sheet_paper_l1636_163653


namespace tom_teaching_years_l1636_163665

theorem tom_teaching_years (T D : ℝ) (h1 : T + D = 70) (h2 : D = (1/2) * T - 5) : T = 50 :=
by
  -- This is where the proof would normally go if it were required.
  sorry

end tom_teaching_years_l1636_163665


namespace problem1_problem2_l1636_163617

-- Define A and B as given
def A (x y : ℝ) : ℝ := 2 * x^2 - 3 * x * y - 5 * x - 1
def B (x y : ℝ) : ℝ := -x^2 + x * y - 1

-- Problem statement 1: Prove 3A + 6B simplifies as expected
theorem problem1 (x y : ℝ) : 3 * A x y + 6 * B x y = -3 * x * y - 15 * x - 9 :=
  by
    sorry

-- Problem statement 2: Prove that if 3A + 6B is independent of x, then y = -5
theorem problem2 (y : ℝ) (h : ∀ x : ℝ, 3 * A x y + 6 * B x y = -9) : y = -5 :=
  by
    sorry

end problem1_problem2_l1636_163617
