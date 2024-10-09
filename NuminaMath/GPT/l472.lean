import Mathlib

namespace minimum_total_length_of_removed_segments_l472_47212

-- Definitions based on conditions
def right_angled_triangle_sides : Nat × Nat × Nat := (3, 4, 5)

def large_square_side_length : Nat := 7

-- Statement of the problem to be proved
theorem minimum_total_length_of_removed_segments
  (triangles : Fin 4 → (Nat × Nat × Nat) := fun _ => right_angled_triangle_sides)
  (side_length_of_large_square : Nat := large_square_side_length) :
  ∃ (removed_length : Nat), removed_length = 7 :=
sorry

end minimum_total_length_of_removed_segments_l472_47212


namespace semicircle_area_in_quarter_circle_l472_47213

theorem semicircle_area_in_quarter_circle (r : ℝ) (A : ℝ) (π : ℝ) (one : ℝ) :
    r = 1 / (Real.sqrt (2) + 1) →
    A = π * r^2 →
    120 * A / π = 20 :=
sorry

end semicircle_area_in_quarter_circle_l472_47213


namespace part1_part2_l472_47242

-- Part 1: Determining the number of toys A and ornaments B wholesaled
theorem part1 (x y : ℕ) (h₁ : x + y = 100) (h₂ : 60 * x + 50 * y = 5650) : 
  x = 65 ∧ y = 35 := by
  sorry

-- Part 2: Determining the minimum number of toys A to wholesale for a 1400元 profit
theorem part2 (m : ℕ) (h₁ : m ≤ 100) (h₂ : (80 - 60) * m + (60 - 50) * (100 - m) ≥ 1400) : 
  m ≥ 40 := by
  sorry

end part1_part2_l472_47242


namespace positive_integer_expression_iff_l472_47268

theorem positive_integer_expression_iff (p : ℕ) : (0 < p) ∧ (∃ k : ℕ, 0 < k ∧ 4 * p + 35 = k * (3 * p - 8)) ↔ p = 3 :=
by
  sorry

end positive_integer_expression_iff_l472_47268


namespace sum_of_first_10_terms_is_350_l472_47263

-- Define the terms and conditions for the arithmetic sequence
variables (a d : ℤ)

-- Define the 4th and 8th terms of the sequence
def fourth_term := a + 3*d
def eighth_term := a + 7*d

-- Given conditions
axiom h1 : fourth_term a d = 23
axiom h2 : eighth_term a d = 55

-- Sum of the first 10 terms of the sequence
def sum_first_10_terms := 10 / 2 * (2*a + (10 - 1)*d)

-- Theorem to prove
theorem sum_of_first_10_terms_is_350 : sum_first_10_terms a d = 350 :=
by sorry

end sum_of_first_10_terms_is_350_l472_47263


namespace total_six_letter_words_l472_47280

def num_vowels := 6
def vowel_count := 5
def word_length := 6

theorem total_six_letter_words : (num_vowels ^ word_length) = 46656 :=
by sorry

end total_six_letter_words_l472_47280


namespace solve_eq1_solve_eq2_l472_47275

-- Define the first equation
def eq1 (x : ℝ) : Prop := x^2 - 2 * x - 1 = 0

-- Define the second equation
def eq2 (x : ℝ) : Prop := (x - 2)^2 = 2 * x - 4

-- State the first theorem
theorem solve_eq1 (x : ℝ) : eq1 x ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by sorry

-- State the second theorem
theorem solve_eq2 (x : ℝ) : eq2 x ↔ (x = 2 ∨ x = 4) :=
by sorry

end solve_eq1_solve_eq2_l472_47275


namespace prob1_prob2_l472_47293

-- Definition and theorems related to the calculations of the given problem.
theorem prob1 : ((-12) - 5 + (-14) - (-39)) = 8 := by 
  sorry

theorem prob2 : (-2^2 * 5 - (-12) / 4 - 4) = -21 := by
  sorry

end prob1_prob2_l472_47293


namespace lcm_of_a_c_l472_47269

theorem lcm_of_a_c (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 24) : Nat.lcm a c = 30 := by
  sorry

end lcm_of_a_c_l472_47269


namespace bianca_total_pictures_l472_47241

def album1_pictures : Nat := 27
def album2_3_4_pictures : Nat := 3 * 2

theorem bianca_total_pictures : album1_pictures + album2_3_4_pictures = 33 := by
  sorry

end bianca_total_pictures_l472_47241


namespace solution_set_of_f_neg_2x_l472_47246

def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

theorem solution_set_of_f_neg_2x (a b : ℝ) (hf_sol : ∀ x : ℝ, (a * x - 1) * (x + b) > 0 ↔ -1 < x ∧ x < 3) :
  ∀ x : ℝ, f a b (-2 * x) < 0 ↔ (x < -3/2 ∨ x > 1/2) :=
by
  sorry

end solution_set_of_f_neg_2x_l472_47246


namespace real_roots_condition_l472_47228

-- Definitions based on conditions
def polynomial (x : ℝ) : ℝ := x^4 - 6 * x - 1
def is_root (a : ℝ) : Prop := polynomial a = 0

-- The statement we want to prove
theorem real_roots_condition (a b : ℝ) (ha: is_root a) (hb: is_root b) : 
  (a * b + 2 * a + 2 * b = 1.5 + Real.sqrt 3) := 
sorry

end real_roots_condition_l472_47228


namespace females_with_advanced_degrees_l472_47231

theorem females_with_advanced_degrees 
  (total_employees : ℕ) 
  (total_females : ℕ) 
  (total_advanced_degrees : ℕ) 
  (total_college_degrees : ℕ) 
  (males_with_college_degree : ℕ) 
  (h1 : total_employees = 180) 
  (h2 : total_females = 110) 
  (h3 : total_advanced_degrees = 90) 
  (h4 : total_college_degrees = 90) 
  (h5 : males_with_college_degree = 35) : 
  ∃ (females_with_advanced_degrees : ℕ), females_with_advanced_degrees = 55 := 
by {
  sorry
}

end females_with_advanced_degrees_l472_47231


namespace stream_current_rate_l472_47282

theorem stream_current_rate (r w : ℝ) : 
  (15 / (r + w) + 5 = 15 / (r - w)) → 
  (15 / (2 * r + w) + 1 = 15 / (2 * r - w)) →
  w = 2 := 
by
  sorry

end stream_current_rate_l472_47282


namespace probability_of_same_color_is_correct_l472_47214

-- Definitions from the problem conditions
def red_marbles := 6
def white_marbles := 7
def blue_marbles := 8
def total_marbles := red_marbles + white_marbles + blue_marbles -- 21

-- Calculate the probability of drawing 4 red marbles
def P_all_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2)) * ((red_marbles - 3) / (total_marbles - 3))

-- Calculate the probability of drawing 4 white marbles
def P_all_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2)) * ((white_marbles - 3) / (total_marbles - 3))

-- Calculate the probability of drawing 4 blue marbles
def P_all_blue := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) * ((blue_marbles - 2) / (total_marbles - 2)) * ((blue_marbles - 3) / (total_marbles - 3))

-- Total probability of drawing 4 marbles of the same color
def P_all_same_color := P_all_red + P_all_white + P_all_blue

-- Proof that the total probability is equal to the given correct answer
theorem probability_of_same_color_is_correct : P_all_same_color = 240 / 11970 := by
  sorry

end probability_of_same_color_is_correct_l472_47214


namespace problem1_problem2_l472_47229

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l472_47229


namespace total_people_clean_city_l472_47294

-- Define the conditions
def lizzie_group : Nat := 54
def group_difference : Nat := 17
def other_group := lizzie_group - group_difference

-- State the theorem
theorem total_people_clean_city : lizzie_group + other_group = 91 := by
  -- Proof would go here
  sorry

end total_people_clean_city_l472_47294


namespace sum_of_reflected_midpoint_coords_l472_47235

theorem sum_of_reflected_midpoint_coords (P R : ℝ × ℝ) 
  (hP : P = (2, 1)) (hR : R = (12, 15)) :
  let M := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)
  let P' := (-P.1, P.2)
  let R' := (-R.1, R.2)
  let M' := ((P'.1 + R'.1) / 2, (P'.2 + R'.2) / 2)
  M'.1 + M'.2 = 1 :=
by
  sorry

end sum_of_reflected_midpoint_coords_l472_47235


namespace directrix_of_parabola_l472_47289

theorem directrix_of_parabola (p : ℝ) (y x : ℝ) :
  y = x^2 → x^2 = 4 * p * y → 4 * y + 1 = 0 :=
by
  intros hyp1 hyp2
  sorry

end directrix_of_parabola_l472_47289


namespace intersection_point_lines_distance_point_to_line_l472_47245

-- Problem 1
theorem intersection_point_lines :
  ∃ (x y : ℝ), (x - y + 2 = 0) ∧ (x - 2 * y + 3 = 0) ∧ (x = -1) ∧ (y = 1) :=
sorry

-- Problem 2
theorem distance_point_to_line :
  ∀ (x y : ℝ), (x = 1) ∧ (y = -2) → ∃ d : ℝ, d = 3 ∧ (d = abs (3 * x + 4 * y - 10) / (Real.sqrt (3^2 + 4^2))) :=
sorry

end intersection_point_lines_distance_point_to_line_l472_47245


namespace inclination_angle_of_line_l472_47266

theorem inclination_angle_of_line (α : ℝ) (h_eq : ∀ x y, x - y + 1 = 0 ↔ y = x + 1) (h_range : 0 < α ∧ α < 180) :
  α = 45 :=
by
  -- α is the inclination angle satisfying tan α = 1 and 0 < α < 180
  sorry

end inclination_angle_of_line_l472_47266


namespace sweets_neither_red_nor_green_l472_47227

theorem sweets_neither_red_nor_green (total_sweets red_sweets green_sweets : ℕ)
  (h1 : total_sweets = 285)
  (h2 : red_sweets = 49)
  (h3 : green_sweets = 59) : total_sweets - (red_sweets + green_sweets) = 177 :=
by
  sorry

end sweets_neither_red_nor_green_l472_47227


namespace solve_x_values_l472_47255

theorem solve_x_values (x : ℝ) :
  (5 + x) / (7 + x) = (2 + x^2) / (4 + x) ↔ x = 1 ∨ x = -2 ∨ x = -3 := 
sorry

end solve_x_values_l472_47255


namespace katie_travel_distance_l472_47205

theorem katie_travel_distance (d_train d_bus d_bike d_car d_total d1 d2 d3 : ℕ)
  (h1 : d_train = 162)
  (h2 : d_bus = 124)
  (h3 : d_bike = 88)
  (h4 : d_car = 224)
  (h_total : d_total = d_train + d_bus + d_bike + d_car)
  (h1_distance : d1 = 96)
  (h2_distance : d2 = 108)
  (h3_distance : d3 = 130)
  (h1_prob : 30 = 30)
  (h2_prob : 50 = 50)
  (h3_prob : 20 = 20) :
  (d_total + d1 = 694) ∧
  (d_total + d2 = 706) ∧
  (d_total + d3 = 728) :=
sorry

end katie_travel_distance_l472_47205


namespace solve_for_x_l472_47217

theorem solve_for_x (n m x : ℕ) (h1 : 5 / 7 = n / 91) (h2 : 5 / 7 = (m + n) / 105) (h3 : 5 / 7 = (x - m) / 140) :
    x = 110 :=
sorry

end solve_for_x_l472_47217


namespace sophia_fraction_of_book_finished_l472_47209

variable (x : ℕ)

theorem sophia_fraction_of_book_finished (h1 : x + (x + 90) = 270) : (x + 90) / 270 = 2 / 3 := by
  sorry

end sophia_fraction_of_book_finished_l472_47209


namespace intersection_of_M_and_N_l472_47247

noncomputable def M : Set ℝ := { y : ℝ | ∃ x : ℝ, y = x^2 }
noncomputable def N : Set ℝ := { y : ℝ | ∃ x : ℝ, x^2 + y^2 = 1 }

theorem intersection_of_M_and_N : M ∩ N = { y : ℝ | 0 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end intersection_of_M_and_N_l472_47247


namespace gum_pieces_per_package_l472_47216

theorem gum_pieces_per_package :
  (∀ (packages pieces each_package : ℕ), packages = 9 ∧ pieces = 135 → each_package = pieces / packages → each_package = 15) := 
by
  intros packages pieces each_package
  sorry

end gum_pieces_per_package_l472_47216


namespace saras_sister_ordered_notebooks_l472_47221

theorem saras_sister_ordered_notebooks (x : ℕ) 
  (initial_notebooks : ℕ := 4) 
  (lost_notebooks : ℕ := 2) 
  (current_notebooks : ℕ := 8) :
  initial_notebooks + x - lost_notebooks = current_notebooks → x = 6 :=
by
  intros h
  sorry

end saras_sister_ordered_notebooks_l472_47221


namespace ice_cream_melt_l472_47236

theorem ice_cream_melt (r_sphere r_cylinder : ℝ) (h : ℝ)
  (V_sphere : ℝ := (4 / 3) * Real.pi * r_sphere^3)
  (V_cylinder : ℝ := Real.pi * r_cylinder^2 * h)
  (H_equal_volumes : V_sphere = V_cylinder) :
  h = 4 / 9 := by
  sorry

end ice_cream_melt_l472_47236


namespace projectiles_initial_distance_l472_47211

theorem projectiles_initial_distance 
  (v₁ v₂ : ℝ) (t : ℝ) (d₁ d₂ d : ℝ) 
  (hv₁ : v₁ = 445 / 60) -- speed of first projectile in km/min
  (hv₂ : v₂ = 545 / 60) -- speed of second projectile in km/min
  (ht : t = 84) -- time to meet in minutes
  (hd₁ : d₁ = v₁ * t) -- distance traveled by the first projectile
  (hd₂ : d₂ = v₂ * t) -- distance traveled by the second projectile
  (hd : d = d₁ + d₂) -- total initial distance
  : d = 1385.6 :=
by 
  sorry

end projectiles_initial_distance_l472_47211


namespace F_2457_find_Q_l472_47249

-- Define the properties of a "rising number"
def is_rising_number (m : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    m = 1000 * a + 100 * b + 10 * c + d ∧
    a < b ∧ b < c ∧ c < d ∧
    a + d = b + c

-- Define F(m) as specified
def F (m : ℕ) : ℤ :=
  let a := m / 1000
  let b := (m / 100) % 10
  let c := (m / 10) % 10
  let d := m % 10
  let m' := 1000 * c + 100 * b + 10 * a + d
  (m' - m) / 99

-- Problem statement for F(2457)
theorem F_2457 : F 2457 = 30 := sorry

-- Properties given in the problem statement for P and Q
def is_specific_rising_number (P Q : ℕ) : Prop :=
  ∃ (x y z t : ℕ),
    P = 1000 + 100 * x + 10 * y + z ∧
    Q = 1000 * x + 100 * t + 60 + z ∧
    1 < x ∧ x < t ∧ t < 6 ∧ 6 < z ∧
    1 + z = x + y ∧
    x + z = t + 6 ∧
    F P + F Q % 7 = 0

-- Problem statement to find the value of Q
theorem find_Q (Q : ℕ) : 
  ∃ (P : ℕ), is_specific_rising_number P Q ∧ Q = 3467 := sorry

end F_2457_find_Q_l472_47249


namespace sum_of_coordinates_eq_nine_halves_l472_47259

theorem sum_of_coordinates_eq_nine_halves {f : ℝ → ℝ} 
  (h₁ : 2 = (f 1) / 2) :
  (4 + (1 / 2) = 9 / 2) :=
by 
  sorry

end sum_of_coordinates_eq_nine_halves_l472_47259


namespace contractor_engaged_days_l472_47279

theorem contractor_engaged_days
  (earnings_per_day : ℤ)
  (fine_per_day : ℤ)
  (total_earnings : ℤ)
  (absent_days : ℤ)
  (days_worked : ℤ) 
  (h1 : earnings_per_day = 25)
  (h2 : fine_per_day = 15 / 2)
  (h3 : total_earnings = 620)
  (h4 : absent_days = 4)
  (h5 : total_earnings = earnings_per_day * days_worked - fine_per_day * absent_days) :
  days_worked = 26 := 
by {
  -- Proof goes here
  sorry
}

end contractor_engaged_days_l472_47279


namespace shorter_leg_length_l472_47240

theorem shorter_leg_length (m h x : ℝ) (H1 : m = 15) (H2 : h = 3 * x) (H3 : m = 0.5 * h) : x = 10 :=
by
  sorry

end shorter_leg_length_l472_47240


namespace product_of_last_two_digits_l472_47299

theorem product_of_last_two_digits (A B : ℕ) (h1 : B = 0 ∨ B = 5) (h2 : A + B = 12) : A * B = 35 :=
by {
  -- proof omitted
  sorry
}

end product_of_last_two_digits_l472_47299


namespace cost_of_airplane_l472_47230

theorem cost_of_airplane (amount : ℝ) (change : ℝ) (h_amount : amount = 5) (h_change : change = 0.72) : 
  amount - change = 4.28 := 
by
  sorry

end cost_of_airplane_l472_47230


namespace sun_xing_zhe_problem_l472_47288

theorem sun_xing_zhe_problem (S X Z : ℕ) (h : S < 10 ∧ X < 10 ∧ Z < 10)
  (hprod : (100 * S + 10 * X + Z) * (100 * Z + 10 * X + S) = 78445) :
  (100 * S + 10 * X + Z) + (100 * Z + 10 * X + S) = 1372 := 
by
  sorry

end sun_xing_zhe_problem_l472_47288


namespace first_discount_correct_l472_47233

noncomputable def first_discount (x : ℝ) : Prop :=
  let initial_price := 600
  let first_discounted_price := initial_price * (1 - x / 100)
  let final_price := first_discounted_price * (1 - 0.05)
  final_price = 456

theorem first_discount_correct : ∃ x : ℝ, first_discount x ∧ abs (x - 57.29) < 0.01 :=
by
  sorry

end first_discount_correct_l472_47233


namespace find_pastries_made_l472_47225

variable (cakes_made pastries_made total_cakes_sold total_pastries_sold extra_cakes more_cakes_than_pastries : ℕ)

def baker_conditions := (cakes_made = 157) ∧ 
                        (total_cakes_sold = 158) ∧ 
                        (total_pastries_sold = 147) ∧ 
                        (more_cakes_than_pastries = 11) ∧ 
                        (extra_cakes = total_cakes_sold - cakes_made) ∧ 
                        (pastries_made = cakes_made - more_cakes_than_pastries)

theorem find_pastries_made : 
  baker_conditions cakes_made pastries_made total_cakes_sold total_pastries_sold extra_cakes more_cakes_than_pastries → 
  pastries_made = 146 :=
by
  sorry

end find_pastries_made_l472_47225


namespace sum_of_factors_coefficients_l472_47281

theorem sum_of_factors_coefficients (a b c d e f g h i j k l m n o p : ℤ) :
  (81 * x^8 - 256 * y^8 = (a * x + b * y) *
                        (c * x^2 + d * x * y + e * y^2) *
                        (f * x^3 + g * x * y^2 + h * y^3) *
                        (i * x + j * y) *
                        (k * x^2 + l * x * y + m * y^2) *
                        (n * x^3 + o * x * y^2 + p * y^3)) →
  a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p = 40 :=
by
  sorry

end sum_of_factors_coefficients_l472_47281


namespace max_value_of_a_l472_47253

theorem max_value_of_a 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 - a * x) 
  (h2 : ∀ x ≥ 1, ∀ y ≥ 1, x ≤ y → f x ≤ f y) : 
  a ≤ 3 :=
sorry

end max_value_of_a_l472_47253


namespace total_selling_amount_l472_47218

-- Defining the given conditions
def total_metres_of_cloth := 200
def loss_per_metre := 6
def cost_price_per_metre := 66

-- Theorem statement to prove the total selling amount
theorem total_selling_amount : 
    (cost_price_per_metre - loss_per_metre) * total_metres_of_cloth = 12000 := 
by 
    sorry

end total_selling_amount_l472_47218


namespace inheritance_amount_l472_47207

def federalTaxRate : ℝ := 0.25
def stateTaxRate : ℝ := 0.15
def totalTaxPaid : ℝ := 16500

theorem inheritance_amount :
  ∃ x : ℝ, (federalTaxRate * x + stateTaxRate * (1 - federalTaxRate) * x = totalTaxPaid) → x = 45500 := by
  sorry

end inheritance_amount_l472_47207


namespace distances_sum_in_triangle_l472_47206

variable (A B C O : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
variable (a b c P AO BO CO : ℝ)

def triangle_sides (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def triangle_perimeter (a b c : ℝ) (P : ℝ) : Prop :=
  P = a + b + c

def point_inside_triangle (O : Type) : Prop := 
  ∃ (A B C : Type), True -- Placeholder for the actual geometric condition

def distances_to_vertices (O : Type) (AO BO CO : ℝ) : Prop := 
  AO >= 0 ∧ BO >= 0 ∧ CO >= 0

theorem distances_sum_in_triangle
  (h1 : triangle_sides a b c)
  (h2 : triangle_perimeter a b c P)
  (h3 : point_inside_triangle O)
  (h4 : distances_to_vertices O AO BO CO) :
  P / 2 < AO + BO + CO ∧ AO + BO + CO < P :=
sorry

end distances_sum_in_triangle_l472_47206


namespace polynomial_function_value_l472_47290

theorem polynomial_function_value 
  (p q r s : ℝ) 
  (h : p - q + r - s = 4) : 
  2 * p + q - 3 * r + 2 * s = -8 := 
by 
  sorry

end polynomial_function_value_l472_47290


namespace series_sum_eq_4_over_9_l472_47250

noncomputable def sum_series : ℝ := ∑' (k : ℕ), (k+1) / 4^(k+1)

theorem series_sum_eq_4_over_9 : sum_series = 4 / 9 := 
sorry

end series_sum_eq_4_over_9_l472_47250


namespace cubic_root_identity_l472_47292

theorem cubic_root_identity (r : ℝ) (h : (r^(1/3)) - (1/(r^(1/3))) = 2) : r^3 - (1/r^3) = 14 := 
by 
  sorry

end cubic_root_identity_l472_47292


namespace natural_pairs_l472_47264

theorem natural_pairs (x y : ℕ) : 2^(2 * x + 1) + 2^x + 1 = y^2 ↔ (x = 0 ∧ y = 2) ∨ (x = 4 ∧ y = 23) :=
by sorry

end natural_pairs_l472_47264


namespace intersection_M_N_eq_02_l472_47220

open Set

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {y | ∃ x ∈ M, y = 2 * x}

theorem intersection_M_N_eq_02 : M ∩ N = {0, 2} := 
by sorry

end intersection_M_N_eq_02_l472_47220


namespace sin_inv_tan_eq_l472_47204

open Real

theorem sin_inv_tan_eq :
  let a := arcsin (4/5)
  let b := arctan 3
  sin (a + b) = (13 * sqrt 10) / 50 := 
by
  let a := arcsin (4/5)
  let b := arctan 3
  sorry

end sin_inv_tan_eq_l472_47204


namespace determine_g_l472_47296

theorem determine_g (g : ℝ → ℝ) : (∀ x : ℝ, 4 * x^4 + x^3 - 2 * x + 5 + g x = 2 * x^3 - 7 * x^2 + 4) →
  (∀ x : ℝ, g x = -4 * x^4 + x^3 - 7 * x^2 + 2 * x - 1) :=
by
  intro h
  sorry

end determine_g_l472_47296


namespace range_values_y_div_x_l472_47234

-- Given the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 6 * y + 12 = 0

-- Prove that the range of values for y / x is [ (6 - 2 * sqrt 3) / 3, (6 + 2 * sqrt 3) / 3 ]
theorem range_values_y_div_x :
  (∀ x y : ℝ, circle_eq x y → (∃ k : ℝ, y = k * x) → 
  ( (6 - 2 * Real.sqrt 3) / 3 ≤ y / x ∧ y / x ≤ (6 + 2 * Real.sqrt 3) / 3 )) :=
sorry

end range_values_y_div_x_l472_47234


namespace compare_fractions_l472_47243

-- Define the fractions
def frac1 : ℚ := -2/3
def frac2 : ℚ := -3/4

-- Prove that -2/3 > -3/4
theorem compare_fractions : frac1 > frac2 :=
by {
  sorry
}

end compare_fractions_l472_47243


namespace reciprocal_of_sum_frac_is_correct_l472_47244

/-- The reciprocal of the sum of the fractions 1/4 and 1/6 is 12/5. -/
theorem reciprocal_of_sum_frac_is_correct:
  (1 / (1 / 4 + 1 / 6)) = (12 / 5) :=
by 
  sorry

end reciprocal_of_sum_frac_is_correct_l472_47244


namespace exists_palindromic_product_l472_47285

open Nat

def is_palindrome (n : ℕ) : Prop :=
  let digits := toDigits 10 n
  digits = digits.reverse

theorem exists_palindromic_product (x : ℕ) (hx : ¬ (10 ∣ x)) : ∃ y : ℕ, is_palindrome (x * y) :=
by
  -- Prove that there exists a natural number y such that x * y is a palindromic number
  sorry

end exists_palindromic_product_l472_47285


namespace simplify_expression_l472_47262

variable (b : ℤ)

theorem simplify_expression :
  (3 * b + 6 - 6 * b) / 3 = -b + 2 :=
sorry

end simplify_expression_l472_47262


namespace find_y_l472_47248

theorem find_y (y : ℚ) : (3 / y - (3 / y) * (y / 5) = 1.2) → y = 5 / 3 :=
sorry

end find_y_l472_47248


namespace complete_square_transform_l472_47265

theorem complete_square_transform (x : ℝ) :
  x^2 - 2 * x - 5 = 0 → (x - 1)^2 = 6 :=
by
  intro h
  sorry

end complete_square_transform_l472_47265


namespace parabola_properties_l472_47251

open Real 

theorem parabola_properties 
  (a : ℝ) 
  (h₀ : a ≠ 0)
  (h₁ : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + (1 - 2 * a) * x₁ + a^2 = 0) ∧ (x₂^2 + (1 - 2 * a) * x₂ + a^2 = 0)) :
  (a < 1 / 4 ∧ ∀ x₁ x₂, (x₁^2 + (1 - 2 * a) * x₁ + a^2 = 0) ∧ (x₂^2 + (1 - 2 * a) * x₂ + a^2 = 0) → x₁ < 0 ∧ x₂ < 0) ∧
  (∀ (x₁ x₂ C : ℝ), (x₁^2 + (1 - 2 * a) * x₁ + a^2 = 0) ∧ (x₂^2 + (1 - 2 * a) * x₂ + a^2 = 0) 
   ∧ (C = a^2) ∧ (-x₁ - x₂ = C - 2) → a = -3) :=
by
  sorry

end parabola_properties_l472_47251


namespace additional_seasons_is_one_l472_47202

-- Definitions for conditions
def episodes_per_season : Nat := 22
def episodes_last_season : Nat := episodes_per_season + 4
def episodes_in_9_seasons : Nat := 9 * episodes_per_season
def hours_per_episode : Nat := 1 / 2 -- Stored as half units

-- Given conditions
def total_hours_to_watch_after_last_season: Nat := 112 * 2 -- converted to half-hours
def time_watched_in_9_seasons: Nat := episodes_in_9_seasons * hours_per_episode
def additional_hours: Nat := total_hours_to_watch_after_last_season - time_watched_in_9_seasons

-- Theorem to prove
theorem additional_seasons_is_one : additional_hours / hours_per_episode = episodes_last_season -> 
      additional_hours / hours_per_episode / episodes_per_season = 1 :=
by
  sorry

end additional_seasons_is_one_l472_47202


namespace hoseok_position_l472_47286

variable (total_people : ℕ) (pos_from_back : ℕ)

theorem hoseok_position (h₁ : total_people = 9) (h₂ : pos_from_back = 5) :
  (total_people - pos_from_back + 1) = 5 :=
by
  sorry

end hoseok_position_l472_47286


namespace integral_sin_pi_over_2_to_pi_l472_47271

theorem integral_sin_pi_over_2_to_pi : ∫ x in (Real.pi / 2)..Real.pi, Real.sin x = 1 := by
  sorry

end integral_sin_pi_over_2_to_pi_l472_47271


namespace symbols_invariance_l472_47257

def final_symbol_invariant (symbols : List Char) : Prop :=
  ∀ (erase : List Char → List Char), 
  (∀ (l : List Char), 
    (erase l = List.cons '+' (List.tail (List.tail l)) ∨ 
    erase l = List.cons '-' (List.tail (List.tail l))) → 
    erase (erase l) = List.cons '+' (List.tail (List.tail (erase l))) ∨ 
    erase (erase l) = List.cons '-' (List.tail (List.tail (erase l)))) →
  (symbols = []) ∨ (symbols = ['+']) ∨ (symbols = ['-'])

theorem symbols_invariance (symbols : List Char) (h : final_symbol_invariant symbols) : 
  ∃ (s : Char), s = '+' ∨ s = '-' :=
  sorry

end symbols_invariance_l472_47257


namespace remainder_of_2n_div_11_l472_47256

theorem remainder_of_2n_div_11 (n k : ℤ) (h : n = 22 * k + 12) : (2 * n) % 11 = 2 :=
by
  sorry

end remainder_of_2n_div_11_l472_47256


namespace option_B_is_linear_inequality_with_one_var_l472_47272

noncomputable def is_linear_inequality_with_one_var (in_eq : String) : Prop :=
  match in_eq with
  | "3x^2 > 45 - 9x" => false
  | "3x - 2 < 4" => true
  | "1 / x < 2" => false
  | "4x - 3 < 2y - 7" => false
  | _ => false

theorem option_B_is_linear_inequality_with_one_var :
  is_linear_inequality_with_one_var "3x - 2 < 4" = true :=
by
  -- Add proof steps here
  sorry

end option_B_is_linear_inequality_with_one_var_l472_47272


namespace smallest_four_digit_multiple_of_37_l472_47277

theorem smallest_four_digit_multiple_of_37 : ∃ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 ∧ 37 ∣ n ∧ (∀ m : ℕ, m ≥ 1000 ∧ m ≤ 9999 ∧ 37 ∣ m → n ≤ m) ∧ n = 1036 :=
by
  sorry

end smallest_four_digit_multiple_of_37_l472_47277


namespace total_wet_surface_area_is_62_l472_47203

-- Define the dimensions of the cistern
def length_cistern : ℝ := 8
def width_cistern : ℝ := 4
def depth_water : ℝ := 1.25

-- Define the calculation of the wet surface area
def bottom_surface_area : ℝ := length_cistern * width_cistern
def longer_side_surface_area : ℝ := length_cistern * depth_water * 2
def shorter_end_surface_area : ℝ := width_cistern * depth_water * 2

-- Sum up all wet surface areas
def total_wet_surface_area : ℝ := bottom_surface_area + longer_side_surface_area + shorter_end_surface_area

-- The theorem stating that the total wet surface area is 62 m²
theorem total_wet_surface_area_is_62 : total_wet_surface_area = 62 := by
  sorry

end total_wet_surface_area_is_62_l472_47203


namespace distinct_paths_to_B_and_C_l472_47226

def paths_to_red_arrows : ℕ × ℕ := (1, 2)
def paths_from_first_red_to_blue : ℕ := 3 * 2
def paths_from_second_red_to_blue : ℕ := 4 * 2
def total_paths_to_blue_arrows : ℕ := paths_from_first_red_to_blue + paths_from_second_red_to_blue

def paths_from_first_two_blue_to_green : ℕ := 5 * 4
def paths_from_third_and_fourth_blue_to_green : ℕ := 6 * 4
def total_paths_to_green_arrows : ℕ := paths_from_first_two_blue_to_green + paths_from_third_and_fourth_blue_to_green

def paths_to_B : ℕ := total_paths_to_green_arrows * 3
def paths_to_C : ℕ := total_paths_to_green_arrows * 4
def total_paths : ℕ := paths_to_B + paths_to_C

theorem distinct_paths_to_B_and_C :
  total_paths = 4312 := 
by
  -- all conditions can be used within this proof
  sorry

end distinct_paths_to_B_and_C_l472_47226


namespace solve_for_x_l472_47219

theorem solve_for_x (x y : ℝ) (h₁ : y = 1 / (4 * x + 2)) (h₂ : y = 1 / 2) : x = 0 :=
by
  -- Placeholder for the proof
  sorry

end solve_for_x_l472_47219


namespace annie_initial_money_l472_47273

theorem annie_initial_money
  (hamburger_price : ℕ := 4)
  (milkshake_price : ℕ := 3)
  (num_hamburgers : ℕ := 8)
  (num_milkshakes : ℕ := 6)
  (money_left : ℕ := 70)
  (total_cost_hamburgers : ℕ := num_hamburgers * hamburger_price)
  (total_cost_milkshakes : ℕ := num_milkshakes * milkshake_price)
  (total_cost : ℕ := total_cost_hamburgers + total_cost_milkshakes)
  : num_hamburgers * hamburger_price + num_milkshakes * milkshake_price + money_left = 120 :=
by
  -- proof part skipped
  sorry

end annie_initial_money_l472_47273


namespace cube_root_expression_l472_47201

variable (x : ℝ)

theorem cube_root_expression (h : x + 1 / x = 7) : x^3 + 1 / x^3 = 322 :=
  sorry

end cube_root_expression_l472_47201


namespace correct_exponent_operation_l472_47222

theorem correct_exponent_operation (a b : ℝ) : 
  a^2 * a^3 = a^5 := 
by sorry

end correct_exponent_operation_l472_47222


namespace monotonicity_of_f_f_greater_than_2_ln_a_plus_3_div_2_l472_47270

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_of_f (a : ℝ) :
  (a ≤ 0 → ∀ x y, x < y → f x a > f y a) ∧
  (a > 0 →
    (∀ x, x < Real.log (1 / a) → f x a > f (Real.log (1 / a)) a) ∧
    (∀ x, x > Real.log (1 / a) → f x a > f (Real.log (1 / a)) a)) :=
sorry

theorem f_greater_than_2_ln_a_plus_3_div_2 (a : ℝ) (h : a > 0) (x : ℝ) :
  f x a > 2 * Real.log a + 3 / 2 :=
sorry

end monotonicity_of_f_f_greater_than_2_ln_a_plus_3_div_2_l472_47270


namespace find_number_l472_47278

theorem find_number (x : ℝ) (h : 2994 / x = 173) : x = 17.3 := 
sorry

end find_number_l472_47278


namespace find_symmetric_point_l472_47239

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def M : Point := ⟨3, -3, -1⟩

def line (x y z : ℝ) : Prop := 
  (x - 6) / 5 = (y - 3.5) / 4 ∧ (x - 6) / 5 = (z + 0.5) / 0

theorem find_symmetric_point (M' : Point) :
  (line M.x M.y M.z) →
  M' = ⟨-1, 2, 0⟩ := by
  sorry

end find_symmetric_point_l472_47239


namespace simplify_tan_product_l472_47297

-- Mathematical Conditions
def tan_inv (x : ℝ) : ℝ := sorry
noncomputable def tan (θ : ℝ) : ℝ := sorry

-- Problem statement to be proven
theorem simplify_tan_product (x y : ℝ) (hx : tan_inv x = 10) (hy : tan_inv y = 35) :
  (1 + x) * (1 + y) = 2 :=
sorry

end simplify_tan_product_l472_47297


namespace rhombus_area_l472_47261

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 20) :
  (d1 * d2) / 2 = 120 :=
by
  sorry

end rhombus_area_l472_47261


namespace john_has_dollars_left_l472_47267

-- Definitions based on the conditions
def john_savings_octal : ℕ := 5273
def rental_car_cost_decimal : ℕ := 1500

-- Define the function to convert octal to decimal
def octal_to_decimal (n : ℕ) : ℕ := -- Conversion logic
sorry

-- Statements for the conversion and subtraction
def john_savings_decimal : ℕ := octal_to_decimal john_savings_octal
def amount_left_for_gas_and_accommodations : ℕ :=
  john_savings_decimal - rental_car_cost_decimal

-- Theorem statement equivalent to the correct answer
theorem john_has_dollars_left :
  amount_left_for_gas_and_accommodations = 1247 :=
by sorry

end john_has_dollars_left_l472_47267


namespace coffee_cost_per_week_l472_47291

theorem coffee_cost_per_week 
  (people_in_house : ℕ) 
  (drinks_per_person_per_day : ℕ) 
  (ounces_per_cup : ℝ) 
  (cost_per_ounce : ℝ) 
  (num_days_in_week : ℕ) 
  (h1 : people_in_house = 4) 
  (h2 : drinks_per_person_per_day = 2)
  (h3 : ounces_per_cup = 0.5)
  (h4 : cost_per_ounce = 1.25)
  (h5 : num_days_in_week = 7) :
  people_in_house * drinks_per_person_per_day * ounces_per_cup * cost_per_ounce * num_days_in_week = 35 := 
by
  sorry

end coffee_cost_per_week_l472_47291


namespace amoeba_count_after_ten_days_l472_47284

theorem amoeba_count_after_ten_days : 
  let initial_amoebas := 1
  let splits_per_day := 3
  let days := 10
  (initial_amoebas * splits_per_day ^ days) = 59049 := 
by 
  let initial_amoebas := 1
  let splits_per_day := 3
  let days := 10
  show (initial_amoebas * splits_per_day ^ days) = 59049
  sorry

end amoeba_count_after_ten_days_l472_47284


namespace initial_students_l472_47298

variable (x : ℕ) -- let x be the initial number of students

-- each condition defined as a function
def first_round_rem (x : ℕ) : ℕ := (40 * x) / 100
def second_round_rem (x : ℕ) : ℕ := first_round_rem x / 2
def third_round_rem (x : ℕ) : ℕ := second_round_rem x / 4

theorem initial_students (x : ℕ) (h : third_round_rem x = 15) : x = 300 := 
by sorry  -- proof will be inserted here

end initial_students_l472_47298


namespace thabo_books_l472_47295

variable (P F H : Nat)

theorem thabo_books :
  P > 55 ∧ F = 2 * P ∧ H = 55 ∧ H + P + F = 280 → P - H = 20 :=
by
  sorry

end thabo_books_l472_47295


namespace line_intersects_x_axis_at_point_l472_47254

theorem line_intersects_x_axis_at_point :
  ∃ x, (4 * x - 2 * 0 = 6) ∧ (2 - 0 = 2 * (0 - x)) → x = 2 := 
by
  sorry

end line_intersects_x_axis_at_point_l472_47254


namespace coordinate_difference_l472_47232

theorem coordinate_difference (m n : ℝ) (h : m = 4 * n + 5) :
  (4 * (n + 0.5) + 5) - m = 2 :=
by
  -- proof skipped
  sorry

end coordinate_difference_l472_47232


namespace number_of_possible_third_side_lengths_l472_47283

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end number_of_possible_third_side_lengths_l472_47283


namespace determinant_zero_l472_47208

theorem determinant_zero (α β : ℝ) :
  Matrix.det ![
    ![0, Real.sin α, -Real.cos α],
    ![-Real.sin α, 0, Real.sin β],
    ![Real.cos α, -Real.sin β, 0]
  ] = 0 :=
by sorry

end determinant_zero_l472_47208


namespace find_f_of_3_l472_47223

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^7 + a*x^5 + b*x - 5

theorem find_f_of_3 (a b : ℝ) (h : f (-3) a b = 5) : f 3 a b = -15 := by
  sorry

end find_f_of_3_l472_47223


namespace lesser_number_l472_47237

theorem lesser_number (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 10) : y = 25 :=
by
  have h3 : x = 35 := sorry
  exact sorry

end lesser_number_l472_47237


namespace mary_total_cards_l472_47200

def mary_initial_cards := 33
def torn_cards := 6
def cards_given_by_sam := 23

theorem mary_total_cards : mary_initial_cards - torn_cards + cards_given_by_sam = 50 :=
  by
    sorry

end mary_total_cards_l472_47200


namespace solution_set_l472_47238

variable (x : ℝ)

def condition_1 : Prop := 2 * x - 4 ≤ 0
def condition_2 : Prop := -x + 1 < 0

theorem solution_set : (condition_1 x ∧ condition_2 x) ↔ (1 < x ∧ x ≤ 2) := by
sorry

end solution_set_l472_47238


namespace rotation_150_positions_l472_47260

/-
Define the initial positions and the shapes involved.
-/
noncomputable def initial_positions := ["A", "B", "C", "D"]
noncomputable def initial_order := ["triangle", "smaller_circle", "square", "pentagon"]

def rotate_clockwise_150 (pos : List String) : List String :=
  -- 1 full position and two-thirds into the next position
  [pos.get! 1, pos.get! 2, pos.get! 3, pos.get! 0]

theorem rotation_150_positions :
  rotate_clockwise_150 initial_positions = ["Triangle between B and C", 
                                            "Smaller circle between C and D", 
                                            "Square between D and A", 
                                            "Pentagon between A and B"] :=
by sorry

end rotation_150_positions_l472_47260


namespace greatest_number_dividing_1642_and_1856_l472_47252

theorem greatest_number_dividing_1642_and_1856 (a b r1 r2 k : ℤ) (h_intro : a = 1642) (h_intro2 : b = 1856) 
    (h_r1 : r1 = 6) (h_r2 : r2 = 4) (h_k1 : k = Int.gcd (a - r1) (b - r2)) :
    k = 4 :=
by
  sorry

end greatest_number_dividing_1642_and_1856_l472_47252


namespace towel_area_decrease_l472_47224

theorem towel_area_decrease (L B : ℝ) (hL : 0 < L) (hB : 0 < B) :
  let A := L * B
  let L' := 0.80 * L
  let B' := 0.90 * B
  let A' := L' * B'
  A' = 0.72 * A →
  ((A - A') / A) * 100 = 28 :=
by
  intros _ _ _ _
  sorry

end towel_area_decrease_l472_47224


namespace sections_capacity_l472_47210

theorem sections_capacity (total_people sections : ℕ) 
  (h1 : total_people = 984) 
  (h2 : sections = 4) : 
  total_people / sections = 246 := 
by
  sorry

end sections_capacity_l472_47210


namespace sam_bought_nine_books_l472_47215

-- Definitions based on the conditions
def initial_money : ℕ := 79
def cost_per_book : ℕ := 7
def money_left : ℕ := 16

-- The amount spent on books
def money_spent_on_books : ℕ := initial_money - money_left

-- The number of books bought
def number_of_books (spent : ℕ) (cost : ℕ) : ℕ := spent / cost

-- Let x be the number of books bought and prove x = 9
theorem sam_bought_nine_books : number_of_books money_spent_on_books cost_per_book = 9 :=
by
  sorry

end sam_bought_nine_books_l472_47215


namespace episodes_first_season_l472_47287

theorem episodes_first_season :
  ∃ (E : ℕ), (100000 * E + 200000 * (3 / 2) * E + 200000 * (3 / 2)^2 * E + 200000 * (3 / 2)^3 * E + 200000 * 24 = 16800000) ∧ E = 8 := 
by {
  sorry
}

end episodes_first_season_l472_47287


namespace graph_passes_fixed_point_l472_47274

-- Mathematical conditions
variables (a : ℝ)

-- Real numbers and conditions
def is_fixed_point (a : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1 ∧ ∃ x y, (x, y) = (2, 2) ∧ y = a^(x-2) + 1

-- Lean statement for the problem
theorem graph_passes_fixed_point : is_fixed_point a :=
  sorry

end graph_passes_fixed_point_l472_47274


namespace find_m_range_l472_47258

theorem find_m_range (m : ℝ) (x : ℝ) (h : ∃ c d : ℝ, (c ≠ 0) ∧ (∀ x, (c * x + d)^2 = x^2 + (12 / 5) * x + (2 * m / 5))) : 3.5 ≤ m ∧ m ≤ 3.7 :=
by
  sorry

end find_m_range_l472_47258


namespace paco_ate_more_cookies_l472_47276

-- Define the number of cookies Paco originally had
def original_cookies : ℕ := 25

-- Define the number of cookies Paco ate
def eaten_cookies : ℕ := 5

-- Define the number of cookies Paco bought
def bought_cookies : ℕ := 3

-- Define the number of more cookies Paco ate than bought
def more_cookies_eaten_than_bought : ℕ := eaten_cookies - bought_cookies

-- Prove that Paco ate 2 more cookies than he bought
theorem paco_ate_more_cookies : more_cookies_eaten_than_bought = 2 := by
  sorry

end paco_ate_more_cookies_l472_47276
