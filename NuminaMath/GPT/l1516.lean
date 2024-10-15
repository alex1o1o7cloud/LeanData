import Mathlib

namespace NUMINAMATH_GPT_inequality_solution_l1516_151645

theorem inequality_solution :
  {x : ℝ | (x^2 + 5 * x) / ((x - 3) ^ 2) ≥ 0} = {x | x < -5} ∪ {x | 0 ≤ x ∧ x < 3} ∪ {x | x > 3} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1516_151645


namespace NUMINAMATH_GPT_line_through_point_l1516_151610

theorem line_through_point (k : ℝ) : (2 - k * 3 = -4 * (-2)) → k = -2 := by
  sorry

end NUMINAMATH_GPT_line_through_point_l1516_151610


namespace NUMINAMATH_GPT_binomial_expansion_fifth_term_constant_l1516_151661

open Classical -- Allows the use of classical logic

noncomputable def binomial_term (n r : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n r) * (x ^ (n - r) / (x ^ r * (2 ^ r / x ^ r)))

theorem binomial_expansion_fifth_term_constant (n : ℕ) :
  (binomial_term n 4 x = (x ^ (n - 3 * 4) * (-2) ^ 4)) → n = 12 := by
  intro h
  sorry

end NUMINAMATH_GPT_binomial_expansion_fifth_term_constant_l1516_151661


namespace NUMINAMATH_GPT_base_k_to_decimal_l1516_151671

theorem base_k_to_decimal (k : ℕ) (h : 0 < k ∧ k < 10) : 
  1 * k^2 + 7 * k + 5 = 125 → k = 8 := 
by
  sorry

end NUMINAMATH_GPT_base_k_to_decimal_l1516_151671


namespace NUMINAMATH_GPT_candy_count_l1516_151638

def initial_candy : ℕ := 47
def eaten_candy : ℕ := 25
def sister_candy : ℕ := 40
def final_candy : ℕ := 62

theorem candy_count : initial_candy - eaten_candy + sister_candy = final_candy := 
by
  sorry

end NUMINAMATH_GPT_candy_count_l1516_151638


namespace NUMINAMATH_GPT_quadratic_eq_positive_integer_roots_l1516_151630

theorem quadratic_eq_positive_integer_roots (k p : ℕ) 
  (h1 : k > 0)
  (h2 : ∃ x1 x2 : ℕ, x1 > 0 ∧ x2 > 0 ∧ (k-1) * x1^2 - p * x1 + k = 0 ∧ (k-1) * x2^2 - p * x2 + k = 0) :
  k ^ (k * p) * (p ^ p + k ^ k) + (p + k) = 1989 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_positive_integer_roots_l1516_151630


namespace NUMINAMATH_GPT_find_a_l1516_151623

-- Points A and B on the x-axis
def point_A (a : ℝ) : (ℝ × ℝ) := (a, 0)
def point_B : (ℝ × ℝ) := (-3, 0)

-- Distance condition
def distance_condition (a : ℝ) : Prop := abs (a + 3) = 5

-- The proof problem: find a such that distance condition holds
theorem find_a (a : ℝ) : distance_condition a ↔ (a = -8 ∨ a = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1516_151623


namespace NUMINAMATH_GPT_relationship_of_a_and_b_l1516_151682

theorem relationship_of_a_and_b (a b : ℝ) (h_b_nonzero: b ≠ 0)
  (m n : ℤ) (h_intersection : ∃ (m n : ℤ), n = m^3 - a * m^2 - b * m ∧ n = a * m + b) :
  2 * a - b + 8 = 0 :=
  sorry

end NUMINAMATH_GPT_relationship_of_a_and_b_l1516_151682


namespace NUMINAMATH_GPT_area_of_storm_eye_l1516_151602

theorem area_of_storm_eye : 
  let large_quarter_circle_area := (1 / 4) * π * 5^2
  let small_circle_area := π * 2^2
  let storm_eye_area := large_quarter_circle_area - small_circle_area
  storm_eye_area = (9 * π) / 4 :=
by
  sorry

end NUMINAMATH_GPT_area_of_storm_eye_l1516_151602


namespace NUMINAMATH_GPT_xiaohong_test_number_l1516_151629

theorem xiaohong_test_number (x : ℕ) :
  (88 * x - 85 * (x - 1) = 100) → x = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_xiaohong_test_number_l1516_151629


namespace NUMINAMATH_GPT_find_digit_B_l1516_151652

theorem find_digit_B (B : ℕ) (h1 : B < 10) : 3 ∣ (5 + 2 + B + 6) → B = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_digit_B_l1516_151652


namespace NUMINAMATH_GPT_value_range_of_f_in_interval_l1516_151653

noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

theorem value_range_of_f_in_interval : 
  ∀ x, (2 ≤ x ∧ x ≤ 4) → (1/2 ≤ f x ∧ f x ≤ 2/3) := 
by
  sorry

end NUMINAMATH_GPT_value_range_of_f_in_interval_l1516_151653


namespace NUMINAMATH_GPT_cost_of_gas_l1516_151687

def hoursDriven1 : ℕ := 2
def speed1 : ℕ := 60
def hoursDriven2 : ℕ := 3
def speed2 : ℕ := 50
def milesPerGallon : ℕ := 30
def costPerGallon : ℕ := 2

def totalDistance : ℕ := (hoursDriven1 * speed1) + (hoursDriven2 * speed2)
def gallonsUsed : ℕ := totalDistance / milesPerGallon
def totalCost : ℕ := gallonsUsed * costPerGallon

theorem cost_of_gas : totalCost = 18 := by
  -- You should fill in the proof steps here.
  sorry

end NUMINAMATH_GPT_cost_of_gas_l1516_151687


namespace NUMINAMATH_GPT_average_score_all_students_l1516_151685

theorem average_score_all_students 
  (n1 n2 : Nat) 
  (avg1 avg2 : Nat) 
  (h1 : n1 = 20) 
  (h2 : avg1 = 80) 
  (h3 : n2 = 30) 
  (h4 : avg2 = 70) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 74 := 
by
  sorry

end NUMINAMATH_GPT_average_score_all_students_l1516_151685


namespace NUMINAMATH_GPT_spherical_coordinates_equivalence_l1516_151643

theorem spherical_coordinates_equivalence :
  ∀ (ρ θ φ : ℝ), 
        ρ = 3 → θ = (2 * Real.pi / 7) → φ = (8 * Real.pi / 5) →
        (0 < ρ) → 
        (0 ≤ (2 * Real.pi / 7) ∧ (2 * Real.pi / 7) < 2 * Real.pi) →
        (0 ≤ (8 * Real.pi / 5) ∧ (8 * Real.pi / 5) ≤ Real.pi) →
      ∃ (ρ' θ' φ' : ℝ), 
        ρ' = ρ ∧ θ' = (9 * Real.pi / 7) ∧ φ' = (2 * Real.pi / 5) :=
by
    sorry

end NUMINAMATH_GPT_spherical_coordinates_equivalence_l1516_151643


namespace NUMINAMATH_GPT_third_number_sixth_row_l1516_151639

/-- Define the arithmetic sequence and related properties. -/
def sequence (n : ℕ) : ℕ := 2 * n - 1

/-- Define sum of first k terms in a series where each row length doubles the previous row length. -/
def sum_of_rows (k : ℕ) : ℕ :=
  2^k - 1

/-- Statement of the problem: Prove that the third number in the sixth row is 67. -/
theorem third_number_sixth_row : sequence (sum_of_rows 5 + 3) = 67 := by
  sorry

end NUMINAMATH_GPT_third_number_sixth_row_l1516_151639


namespace NUMINAMATH_GPT_rectangle_area_l1516_151611

theorem rectangle_area (x y : ℝ) (hx : x ≠ 0) (h : x * y = 10) : y = 10 / x :=
sorry

end NUMINAMATH_GPT_rectangle_area_l1516_151611


namespace NUMINAMATH_GPT_swans_after_10_years_l1516_151663

-- Defining the initial conditions
def initial_swans : ℕ := 15

-- Condition that the number of swans doubles every 2 years
def double_every_two_years (n t : ℕ) : ℕ := n * (2 ^ (t / 2))

-- Prove that after 10 years, the number of swans will be 480
theorem swans_after_10_years : double_every_two_years initial_swans 10 = 480 :=
by
  sorry

end NUMINAMATH_GPT_swans_after_10_years_l1516_151663


namespace NUMINAMATH_GPT_arithmetic_seq_15th_term_is_53_l1516_151655

-- Define an arithmetic sequence
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Original terms given
def a₁ : ℤ := -3
def d : ℤ := 4
def n : ℕ := 15

-- Prove that the 15th term is 53
theorem arithmetic_seq_15th_term_is_53 :
  arithmetic_seq a₁ d n = 53 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_15th_term_is_53_l1516_151655


namespace NUMINAMATH_GPT_closest_fraction_to_medals_won_l1516_151635

theorem closest_fraction_to_medals_won :
  let gamma_fraction := (13:ℚ) / 80
  let fraction_1_4 := (1:ℚ) / 4
  let fraction_1_5 := (1:ℚ) / 5
  let fraction_1_6 := (1:ℚ) / 6
  let fraction_1_7 := (1:ℚ) / 7
  let fraction_1_8 := (1:ℚ) / 8
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_4) ∧
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_5) ∧
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_7) ∧
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_8) := by
  sorry

end NUMINAMATH_GPT_closest_fraction_to_medals_won_l1516_151635


namespace NUMINAMATH_GPT_input_value_of_x_l1516_151612

theorem input_value_of_x (x y : ℤ) (h₁ : (x < 0 → y = (x + 1) * (x + 1)) ∧ (¬(x < 0) → y = (x - 1) * (x - 1)))
  (h₂ : y = 16) : x = 5 ∨ x = -5 :=
sorry

end NUMINAMATH_GPT_input_value_of_x_l1516_151612


namespace NUMINAMATH_GPT_question1_question2_question3_l1516_151692

open Set

-- Define sets A and B
def A := { x : ℝ | x^2 + 6 * x + 5 < 0 }
def B := { x : ℝ | -1 ≤ x ∧ x < 1 }

-- Universal set U is implicitly ℝ in Lean

-- Question 1: Prove A ∩ B = ∅
theorem question1 : A ∩ B = ∅ := 
sorry

-- Question 2: Prove complement of A ∪ B in ℝ is (-∞, -5] ∪ [1, ∞)
theorem question2 : compl (A ∪ B) = { x : ℝ | x ≤ -5 } ∪ { x : ℝ | x ≥ 1 } := 
sorry

-- Define set C which depends on parameter a
def C (a: ℝ) := { x : ℝ | x < a }

-- Question 3: Prove if B ∩ C = B, then a ≥ 1
theorem question3 (a : ℝ) (h : B ∩ C a = B) : a ≥ 1 := 
sorry

end NUMINAMATH_GPT_question1_question2_question3_l1516_151692


namespace NUMINAMATH_GPT_closest_weight_total_shortfall_total_selling_price_l1516_151601

-- Definitions
def standard_weight : ℝ := 25
def weights : List ℝ := [1.5, -3, 2, -0.5, 1, -2, -2.5, -2]
def price_per_kg : ℝ := 2.6

-- Assertions
theorem closest_weight : ∃ w ∈ weights, abs w = 0.5 ∧ 25 + w = 24.5 :=
by sorry

theorem total_shortfall : (weights.sum = -5.5) :=
by sorry

theorem total_selling_price : (8 * standard_weight + weights.sum) * price_per_kg = 505.7 :=
by sorry

end NUMINAMATH_GPT_closest_weight_total_shortfall_total_selling_price_l1516_151601


namespace NUMINAMATH_GPT_gcd_360_504_l1516_151632

theorem gcd_360_504 : Nat.gcd 360 504 = 72 :=
by sorry

end NUMINAMATH_GPT_gcd_360_504_l1516_151632


namespace NUMINAMATH_GPT_minimum_teachers_to_cover_all_subjects_l1516_151688

/- Define the problem conditions -/
def maths_teachers := 7
def physics_teachers := 6
def chemistry_teachers := 5
def max_subjects_per_teacher := 3

/- The proof statement -/
theorem minimum_teachers_to_cover_all_subjects : 
  (maths_teachers + physics_teachers + chemistry_teachers) / max_subjects_per_teacher = 7 :=
sorry

end NUMINAMATH_GPT_minimum_teachers_to_cover_all_subjects_l1516_151688


namespace NUMINAMATH_GPT_function_increment_l1516_151656

theorem function_increment (x₁ x₂ : ℝ) (f : ℝ → ℝ) (h₁ : x₁ = 2) 
                           (h₂ : x₂ = 2.5) (h₃ : ∀ x, f x = x ^ 2) :
  f x₂ - f x₁ = 2.25 :=
by
  sorry

end NUMINAMATH_GPT_function_increment_l1516_151656


namespace NUMINAMATH_GPT_probability_of_selecting_A_l1516_151605

noncomputable def total_students : ℕ := 4
noncomputable def selected_student_A : ℕ := 1

theorem probability_of_selecting_A : 
  (selected_student_A : ℝ) / (total_students : ℝ) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_A_l1516_151605


namespace NUMINAMATH_GPT_alice_numbers_l1516_151609

theorem alice_numbers (x y : ℝ) (h1 : x * y = 12) (h2 : x + y = 7) : (x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 3) :=
by
  sorry

end NUMINAMATH_GPT_alice_numbers_l1516_151609


namespace NUMINAMATH_GPT_number_of_points_l1516_151633

theorem number_of_points (x y : ℕ) (h : y = (2 * x + 2018) / (x - 1)) 
  (h2 : x > y) (h3 : 0 < x) (h4 : 0 < y) : 
  ∃! (x y : ℕ), y = (2 * x + 2018) / (x - 1) ∧ x > y ∧ 0 < x ∧ 0 < y :=
sorry

end NUMINAMATH_GPT_number_of_points_l1516_151633


namespace NUMINAMATH_GPT_mass_percentage_of_S_in_Al2S3_l1516_151628

theorem mass_percentage_of_S_in_Al2S3 :
  let molar_mass_Al : ℝ := 26.98
  let molar_mass_S : ℝ := 32.06
  let formula_of_Al2S3: (ℕ × ℕ) := (2, 3)
  let molar_mass_Al2S3 : ℝ := (2 * molar_mass_Al) + (3 * molar_mass_S)
  let total_mass_S_in_Al2S3 : ℝ := 3 * molar_mass_S
  (total_mass_S_in_Al2S3 / molar_mass_Al2S3) * 100 = 64.07 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_of_S_in_Al2S3_l1516_151628


namespace NUMINAMATH_GPT_expression_value_zero_l1516_151622

variables (a b c A B C : ℝ)

theorem expression_value_zero
  (h1 : a + b + c = 0)
  (h2 : A + B + C = 0)
  (h3 : a / A + b / B + c / C = 0) :
  a * A^2 + b * B^2 + c * C^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_zero_l1516_151622


namespace NUMINAMATH_GPT_math_problem_l1516_151680

noncomputable def ellipse_standard_equation (a b : ℝ) : Prop :=
  a = 2 ∧ b = Real.sqrt 3 ∧ (∀ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def constant_slope_sum (T R S : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  T = (4, 0) ∧ l (1, 0) ∧ 
  (∀ TR TS : ℝ, (TR = (R.2 / (R.1 - 4)) ∧ TS = (S.2 / (S.1 - 4)) ∧ 
  (TR + TS = 0)))

theorem math_problem 
  {a b : ℝ} {T R S : ℝ × ℝ} {l : ℝ × ℝ → Prop} : 
  ellipse_standard_equation a b ∧ constant_slope_sum T R S l :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1516_151680


namespace NUMINAMATH_GPT_smallest_sum_xyz_l1516_151690

theorem smallest_sum_xyz (x y z : ℕ) (h : x * y * z = 40320) : x + y + z ≥ 103 :=
sorry

end NUMINAMATH_GPT_smallest_sum_xyz_l1516_151690


namespace NUMINAMATH_GPT_units_digit_fraction_l1516_151646

theorem units_digit_fraction : (2^3 * 31 * 33 * 17 * 7) % 10 = 6 := by
  sorry

end NUMINAMATH_GPT_units_digit_fraction_l1516_151646


namespace NUMINAMATH_GPT_distance_fall_l1516_151604

-- Given conditions as definitions
def velocity (g : ℝ) (t : ℝ) := g * t

-- The theorem stating the relationship between time t0 and distance S
theorem distance_fall (g : ℝ) (t0 : ℝ) : 
  (∫ t in (0 : ℝ)..t0, velocity g t) = (1/2) * g * t0^2 :=
by 
  sorry

end NUMINAMATH_GPT_distance_fall_l1516_151604


namespace NUMINAMATH_GPT_total_time_outside_class_l1516_151679

-- Definitions based on given conditions
def first_recess : ℕ := 15
def second_recess : ℕ := 15
def lunch : ℕ := 30
def third_recess : ℕ := 20

-- Proof problem statement
theorem total_time_outside_class : first_recess + second_recess + lunch + third_recess = 80 := 
by sorry

end NUMINAMATH_GPT_total_time_outside_class_l1516_151679


namespace NUMINAMATH_GPT_max_area_of_rect_l1516_151694

theorem max_area_of_rect (x y : ℝ) (h1 : x + y = 10) : 
  x * y ≤ 25 :=
by 
  sorry

end NUMINAMATH_GPT_max_area_of_rect_l1516_151694


namespace NUMINAMATH_GPT_neg_alpha_quadrant_l1516_151614

theorem neg_alpha_quadrant (α : ℝ) (k : ℤ) 
    (h1 : k * 360 + 180 < α)
    (h2 : α < k * 360 + 270) :
    k * 360 + 90 < -α ∧ -α < k * 360 + 180 :=
by
  sorry

end NUMINAMATH_GPT_neg_alpha_quadrant_l1516_151614


namespace NUMINAMATH_GPT_hyperbola_asymptote_value_l1516_151640

theorem hyperbola_asymptote_value {b : ℝ} (h : b > 0) 
  (asymptote_eq : ∀ x : ℝ, y = x * (1 / 2) ∨ y = -x * (1 / 2)) :
  b = 1 :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptote_value_l1516_151640


namespace NUMINAMATH_GPT_prob_not_lose_when_A_plays_l1516_151607

def appearance_prob_center_forward : ℝ := 0.3
def appearance_prob_winger : ℝ := 0.5
def appearance_prob_attacking_midfielder : ℝ := 0.2

def lose_prob_center_forward : ℝ := 0.3
def lose_prob_winger : ℝ := 0.2
def lose_prob_attacking_midfielder : ℝ := 0.2

theorem prob_not_lose_when_A_plays : 
    (appearance_prob_center_forward * (1 - lose_prob_center_forward) + 
    appearance_prob_winger * (1 - lose_prob_winger) + 
    appearance_prob_attacking_midfielder * (1 - lose_prob_attacking_midfielder)) = 0.77 := 
by
  sorry

end NUMINAMATH_GPT_prob_not_lose_when_A_plays_l1516_151607


namespace NUMINAMATH_GPT_roots_quadratic_equation_l1516_151606

theorem roots_quadratic_equation (x1 x2 : ℝ) (h1 : x1^2 - x1 - 1 = 0) (h2 : x2^2 - x2 - 1 = 0) :
  (x2 / x1) + (x1 / x2) = -3 :=
by
  sorry

end NUMINAMATH_GPT_roots_quadratic_equation_l1516_151606


namespace NUMINAMATH_GPT_celia_time_correct_lexie_time_correct_nik_time_correct_l1516_151658

noncomputable def lexie_time_per_mile : ℝ := 20
noncomputable def celia_time_per_mile : ℝ := lexie_time_per_mile / 2
noncomputable def nik_time_per_mile : ℝ := lexie_time_per_mile / 1.5

noncomputable def total_distance : ℝ := 30

-- Calculate the baseline running time without obstacles
noncomputable def lexie_baseline_time : ℝ := lexie_time_per_mile * total_distance
noncomputable def celia_baseline_time : ℝ := celia_time_per_mile * total_distance
noncomputable def nik_baseline_time : ℝ := nik_time_per_mile * total_distance

-- Additional time due to obstacles
noncomputable def celia_muddy_extra_time : ℝ := 2 * (celia_time_per_mile * 1.25 - celia_time_per_mile)
noncomputable def lexie_bee_extra_time : ℝ := 2 * 10
noncomputable def nik_detour_extra_time : ℝ := 0.5 * nik_time_per_mile

-- Total time taken including obstacles
noncomputable def celia_total_time : ℝ := celia_baseline_time + celia_muddy_extra_time
noncomputable def lexie_total_time : ℝ := lexie_baseline_time + lexie_bee_extra_time
noncomputable def nik_total_time : ℝ := nik_baseline_time + nik_detour_extra_time

theorem celia_time_correct : celia_total_time = 305 := by sorry
theorem lexie_time_correct : lexie_total_time = 620 := by sorry
theorem nik_time_correct : nik_total_time = 406.565 := by sorry

end NUMINAMATH_GPT_celia_time_correct_lexie_time_correct_nik_time_correct_l1516_151658


namespace NUMINAMATH_GPT_pizza_promotion_savings_l1516_151696

theorem pizza_promotion_savings :
  let regular_price : ℕ := 18
  let promo_price : ℕ := 5
  let num_pizzas : ℕ := 3
  let total_regular_price := num_pizzas * regular_price
  let total_promo_price := num_pizzas * promo_price
  let total_savings := total_regular_price - total_promo_price
  total_savings = 39 :=
by
  sorry

end NUMINAMATH_GPT_pizza_promotion_savings_l1516_151696


namespace NUMINAMATH_GPT_ratio_20_to_10_exists_l1516_151660

theorem ratio_20_to_10_exists (x : ℕ) (h : x = 20 * 10) : x = 200 :=
by sorry

end NUMINAMATH_GPT_ratio_20_to_10_exists_l1516_151660


namespace NUMINAMATH_GPT_determine_a2016_l1516_151644

noncomputable def a_n (n : ℕ) : ℤ := sorry
noncomputable def S_n (n : ℕ) : ℤ := sorry

axiom S1 : S_n 1 = 6
axiom S2 : S_n 2 = 4
axiom S_pos (n : ℕ) : S_n n > 0
axiom geom_progression (n : ℕ) : (S_n (2 * n - 1))^2 = S_n (2 * n) * S_n (2 * n + 2)
axiom arith_progression (n : ℕ) : 2 * S_n (2 * n + 2) = S_n (2 * n - 1) + S_n (2 * n + 1)

theorem determine_a2016 : a_n 2016 = -1009 :=
by sorry

end NUMINAMATH_GPT_determine_a2016_l1516_151644


namespace NUMINAMATH_GPT_magnitude_of_z_l1516_151637

noncomputable def z : ℂ := Complex.I * (3 + 4 * Complex.I)

theorem magnitude_of_z : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_GPT_magnitude_of_z_l1516_151637


namespace NUMINAMATH_GPT_inequality_proof_l1516_151651

theorem inequality_proof (x y : ℝ) (h : x * y < 0) : abs (x + y) < abs (x - y) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1516_151651


namespace NUMINAMATH_GPT_age_of_15th_student_l1516_151613

theorem age_of_15th_student 
  (total_age_15_students : ℕ)
  (total_age_3_students : ℕ)
  (total_age_11_students : ℕ)
  (h1 : total_age_15_students = 225)
  (h2 : total_age_3_students = 42)
  (h3 : total_age_11_students = 176) :
  total_age_15_students - (total_age_3_students + total_age_11_students) = 7 :=
by
  sorry

end NUMINAMATH_GPT_age_of_15th_student_l1516_151613


namespace NUMINAMATH_GPT_largest_integer_solution_l1516_151634

theorem largest_integer_solution (m : ℤ) (h : 2 * m + 7 ≤ 3) : m ≤ -2 :=
sorry

end NUMINAMATH_GPT_largest_integer_solution_l1516_151634


namespace NUMINAMATH_GPT_certain_value_of_101n_squared_l1516_151684

theorem certain_value_of_101n_squared 
  (n : ℤ) 
  (h : ∀ (n : ℤ), 101 * n^2 ≤ 4979 → n ≤ 7) : 
  4979 = 101 * 7^2 :=
by {
  /- proof goes here -/
  sorry
}

end NUMINAMATH_GPT_certain_value_of_101n_squared_l1516_151684


namespace NUMINAMATH_GPT_find_d_l1516_151666

noncomputable def d : ℝ := 3.44

theorem find_d :
  (∃ x : ℝ, (3 * x^2 + 19 * x - 84 = 0) ∧ x = ⌊d⌋) ∧
  (∃ y : ℝ, (5 * y^2 - 26 * y + 12 = 0) ∧ y = d - ⌊d⌋) →
  d = 3.44 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l1516_151666


namespace NUMINAMATH_GPT_no_solution_iff_k_nonnegative_l1516_151600

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then k * x + 2 else (1 / 2) ^ x

theorem no_solution_iff_k_nonnegative (k : ℝ) :
  (¬ ∃ x : ℝ, f k (f k x) = 3 / 2) ↔ k ≥ 0 :=
  sorry

end NUMINAMATH_GPT_no_solution_iff_k_nonnegative_l1516_151600


namespace NUMINAMATH_GPT_maximize_area_minimize_length_l1516_151620

-- Problem 1: Prove maximum area of the enclosure
theorem maximize_area (x y : ℝ) (h : x + 2 * y = 36) : 18 * 9 = 162 :=
by
  sorry

-- Problem 2: Prove the minimum length of steel wire mesh
theorem minimize_length (x y : ℝ) (h1 : x * y = 32) : 8 + 2 * 4 = 16 :=
by
  sorry

end NUMINAMATH_GPT_maximize_area_minimize_length_l1516_151620


namespace NUMINAMATH_GPT_range_of_a_l1516_151617

theorem range_of_a (a : ℝ) : 
  (∀ x, (x > 2 ∨ x < -1) → ¬(x^2 + 4 * x + a < 0)) → a ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1516_151617


namespace NUMINAMATH_GPT_brittany_first_test_grade_l1516_151669

theorem brittany_first_test_grade (x : ℤ) (h1 : (x + 84) / 2 = 81) : x = 78 :=
by
  sorry

end NUMINAMATH_GPT_brittany_first_test_grade_l1516_151669


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l1516_151668

theorem arithmetic_expression_evaluation :
  2 + 8 * 3 - 4 + 7 * 6 / 3 = 36 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l1516_151668


namespace NUMINAMATH_GPT_solution_pairs_l1516_151631

open Int

theorem solution_pairs (a b : ℝ) (h : ∀ n : ℕ, n > 0 → a * ⌊b * n⌋ = b * ⌊a * n⌋) :
  a = 0 ∨ b = 0 ∨ a = b ∨ (∃ (a_int b_int : ℤ), a = a_int ∧ b = b_int) :=
by sorry

end NUMINAMATH_GPT_solution_pairs_l1516_151631


namespace NUMINAMATH_GPT_area_of_circular_platform_l1516_151697

theorem area_of_circular_platform (d : ℝ) (h : d = 2) : ∃ (A : ℝ), A = Real.pi ∧ A = π *(d / 2)^2 := by
  sorry

end NUMINAMATH_GPT_area_of_circular_platform_l1516_151697


namespace NUMINAMATH_GPT_tim_pays_300_l1516_151662

def mri_cost : ℕ := 1200
def doctor_rate_per_hour : ℕ := 300
def examination_time_in_hours : ℕ := 1 / 2
def consultation_fee : ℕ := 150
def insurance_coverage : ℚ := 0.8

def examination_cost : ℕ := doctor_rate_per_hour * examination_time_in_hours
def total_cost_before_insurance : ℕ := mri_cost + examination_cost + consultation_fee
def insurance_coverage_amount : ℚ := total_cost_before_insurance * insurance_coverage
def amount_tim_pays : ℚ := total_cost_before_insurance - insurance_coverage_amount

theorem tim_pays_300 : amount_tim_pays = 300 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_tim_pays_300_l1516_151662


namespace NUMINAMATH_GPT_red_light_at_A_prob_calc_l1516_151678

-- Defining the conditions
def count_total_permutations : ℕ := Nat.factorial 4 / Nat.factorial 1
def count_favorable_permutations : ℕ := Nat.factorial 3 / Nat.factorial 1

-- Calculating the probability
def probability_red_at_A : ℚ := count_favorable_permutations / count_total_permutations

-- Statement to be proved
theorem red_light_at_A_prob_calc : probability_red_at_A = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_red_light_at_A_prob_calc_l1516_151678


namespace NUMINAMATH_GPT_union_intersection_l1516_151665

variable (a : ℝ)

def setA (a : ℝ) : Set ℝ := { x | (x - 3) * (x - a) = 0 }
def setB : Set ℝ := {1, 4}

theorem union_intersection (a : ℝ) :
  (if a = 3 then setA a ∪ setB = {1, 3, 4} ∧ setA a ∩ setB = ∅ else 
   if a = 1 then setA a ∪ setB = {1, 3, 4} ∧ setA a ∩ setB = {1} else
   if a = 4 then setA a ∪ setB = {1, 3, 4} ∧ setA a ∩ setB = {4} else
   setA a ∪ setB = {1, 3, 4, a} ∧ setA a ∩ setB = ∅) := sorry

end NUMINAMATH_GPT_union_intersection_l1516_151665


namespace NUMINAMATH_GPT_coplanar_vectors_m_value_l1516_151695

variable (m : ℝ)
variable (α β : ℝ)
def a := (5, 9, m)
def b := (1, -1, 2)
def c := (2, 5, 1)

theorem coplanar_vectors_m_value :
  ∃ (α β : ℝ), (5 = α + 2 * β) ∧ (9 = -α + 5 * β) ∧ (m = 2 * α + β) → m = 4 :=
by
  sorry

end NUMINAMATH_GPT_coplanar_vectors_m_value_l1516_151695


namespace NUMINAMATH_GPT_allan_balloons_count_l1516_151667

-- Definition of the conditions
def Total_balloons : ℕ := 3
def Jake_balloons : ℕ := 1

-- The theorem that corresponds to the problem statement
theorem allan_balloons_count (Allan_balloons : ℕ) (h : Allan_balloons + Jake_balloons = Total_balloons) : Allan_balloons = 2 := 
by
  sorry

end NUMINAMATH_GPT_allan_balloons_count_l1516_151667


namespace NUMINAMATH_GPT_probability_statements_l1516_151693

-- Assigning probabilities
def p_hit := 0.9
def p_miss := 1 - p_hit

-- Definitions based on the problem conditions
def shoot_4_times (shots : List Bool) : Bool :=
  shots.length = 4 ∧ ∀ (s : Bool), s ∈ shots → (s = true → s ≠ false) ∧ (s = false → s ≠ true ∧ s ≠ 0)

-- Statements derived from the conditions
def prob_shot_3 := p_hit

def prob_exact_3_out_of_4 := 
  let binom_4_3 := 4
  binom_4_3 * (p_hit^3) * (p_miss^1)

def prob_at_least_1_out_of_4 := 1 - (p_miss^4)

-- The equivalence proof
theorem probability_statements : 
  (prob_shot_3 = 0.9) ∧ 
  (prob_exact_3_out_of_4 = 0.2916) ∧ 
  (prob_at_least_1_out_of_4 = 0.9999) := 
by 
  sorry

end NUMINAMATH_GPT_probability_statements_l1516_151693


namespace NUMINAMATH_GPT_inequality_implies_range_of_a_l1516_151681

theorem inequality_implies_range_of_a (a : ℝ) :
  (∀ x : ℝ, |2 - x| + |1 + x| ≥ a^2 - 2 * a) → (-1 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_GPT_inequality_implies_range_of_a_l1516_151681


namespace NUMINAMATH_GPT_abdul_largest_number_l1516_151642

theorem abdul_largest_number {a b c d : ℕ} 
  (h1 : a + (b + c + d) / 3 = 17)
  (h2 : b + (a + c + d) / 3 = 21)
  (h3 : c + (a + b + d) / 3 = 23)
  (h4 : d + (a + b + c) / 3 = 29) :
  d = 21 :=
by sorry

end NUMINAMATH_GPT_abdul_largest_number_l1516_151642


namespace NUMINAMATH_GPT_math_problem_l1516_151676

theorem math_problem 
  (a b c : ℝ) 
  (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c) 
  (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : a^2 + b^2 = c^2 + ab) : 
  c^2 + ab < a*c + b*c := 
sorry

end NUMINAMATH_GPT_math_problem_l1516_151676


namespace NUMINAMATH_GPT_gcd_problem_l1516_151625

theorem gcd_problem : 
  let a := 690
  let b := 875
  let r1 := 10
  let r2 := 25
  let n1 := a - r1
  let n2 := b - r2
  gcd n1 n2 = 170 :=
by
  sorry

end NUMINAMATH_GPT_gcd_problem_l1516_151625


namespace NUMINAMATH_GPT_kids_go_to_camp_l1516_151698

theorem kids_go_to_camp (total_kids : ℕ) (kids_stay_home : ℕ) (h1 : total_kids = 898051) (h2 : kids_stay_home = 268627) : total_kids - kids_stay_home = 629424 :=
by
  sorry

end NUMINAMATH_GPT_kids_go_to_camp_l1516_151698


namespace NUMINAMATH_GPT_arithmetic_series_first_term_l1516_151650

theorem arithmetic_series_first_term :
  ∃ a d : ℚ, 
    (30 * (2 * a + 59 * d) = 240) ∧
    (30 * (2 * a + 179 * d) = 3240) ∧
    a = - (247 / 12) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_series_first_term_l1516_151650


namespace NUMINAMATH_GPT_quadratic_equation_single_solution_l1516_151648

theorem quadratic_equation_single_solution (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + a * x + 1 = 0) ∧ (∀ x1 x2 : ℝ, a * x1^2 + a * x1 + 1 = 0 → a * x2^2 + a * x2 + 1 = 0 → x1 = x2) → a = 4 :=
by sorry

end NUMINAMATH_GPT_quadratic_equation_single_solution_l1516_151648


namespace NUMINAMATH_GPT_value_of_f_at_pi_over_12_l1516_151619

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 12)

theorem value_of_f_at_pi_over_12 : f (Real.pi / 12) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_at_pi_over_12_l1516_151619


namespace NUMINAMATH_GPT_kelly_initial_games_l1516_151636

theorem kelly_initial_games (games_given_away : ℕ) (games_left : ℕ)
  (h1 : games_given_away = 91) (h2 : games_left = 92) : 
  games_given_away + games_left = 183 :=
by {
  sorry
}

end NUMINAMATH_GPT_kelly_initial_games_l1516_151636


namespace NUMINAMATH_GPT_customers_who_didnt_tip_l1516_151616

def initial_customers : ℕ := 39
def added_customers : ℕ := 12
def customers_who_tipped : ℕ := 2

theorem customers_who_didnt_tip : initial_customers + added_customers - customers_who_tipped = 49 := by
  sorry

end NUMINAMATH_GPT_customers_who_didnt_tip_l1516_151616


namespace NUMINAMATH_GPT_trish_walks_l1516_151621

variable (n : ℕ) (M D : ℝ)
variable (d : ℕ → ℝ)
variable (H1 : d 1 = 1)
variable (H2 : ∀ k : ℕ, d (k + 1) = 2 * d k)
variable (H3 : d n > M)

theorem trish_walks (n : ℕ) (M : ℝ) (H1 : d 1 = 1) (H2 : ∀ k : ℕ, d (k + 1) = 2 * d k) (H3 : d n > M) : 2^(n-1) > M := by
  sorry

end NUMINAMATH_GPT_trish_walks_l1516_151621


namespace NUMINAMATH_GPT_exists_disjoint_subsets_for_prime_products_l1516_151699

theorem exists_disjoint_subsets_for_prime_products :
  ∃ (A : Fin 100 → Set ℕ), (∀ i j, i ≠ j → Disjoint (A i) (A j)) ∧
    (∀ S : Set ℕ, Infinite S → (∃ m : ℕ, ∃ (a : Fin 100 → ℕ),
      (∀ i, a i ∈ A i) ∧ (∀ i, ∃ p : Fin m → ℕ, (∀ k, p k ∈ S) ∧ a i = (List.prod (List.ofFn p))))) :=
sorry

end NUMINAMATH_GPT_exists_disjoint_subsets_for_prime_products_l1516_151699


namespace NUMINAMATH_GPT_manuscript_fee_l1516_151672

noncomputable def tax (x : ℝ) : ℝ :=
  if x ≤ 800 then 0
  else if x <= 4000 then 0.14 * (x - 800)
  else 0.11 * x

theorem manuscript_fee (x : ℝ) (h₁ : tax x = 420)
  (h₂ : 800 < x ∧ x ≤ 4000 ∨ x > 4000) :
  x = 3800 :=
sorry

end NUMINAMATH_GPT_manuscript_fee_l1516_151672


namespace NUMINAMATH_GPT_VasyaSlowerWalkingFullWayHome_l1516_151618

namespace FishingTrip

-- Define the variables involved
variables (x v S : ℝ)   -- x is the speed of Vasya and Petya, v is the speed of Kolya on the bicycle, S is the distance from the house to the lake

-- Conditions derived from the problem statement:
-- Condition 1: When Kolya meets Vasya then Petya starts
-- Condition 2: Given: Petya’s travel time is \( \frac{5}{4} \times \) Vasya's travel time.

theorem VasyaSlowerWalkingFullWayHome (h1 : v = 3 * x) :
  2 * (S / x + v) = (5 / 2) * (S / x) :=
sorry

end FishingTrip

end NUMINAMATH_GPT_VasyaSlowerWalkingFullWayHome_l1516_151618


namespace NUMINAMATH_GPT_max_intersections_arith_geo_seq_l1516_151670

def arithmetic_sequence (n : ℕ) (d : ℝ) : ℝ := 1 + (n - 1) * d

def geometric_sequence (n : ℕ) (q : ℝ) : ℝ := q ^ (n - 1)

theorem max_intersections_arith_geo_seq (d : ℝ) (q : ℝ) (h_d : d ≠ 0) (h_q_pos : q > 0) (h_q_neq1 : q ≠ 1) :
  (∃ n : ℕ, arithmetic_sequence n d = geometric_sequence n q) → ∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧ (arithmetic_sequence n₁ d = geometric_sequence n₁ q) ∧ (arithmetic_sequence n₂ d = geometric_sequence n₂ q) :=
sorry

end NUMINAMATH_GPT_max_intersections_arith_geo_seq_l1516_151670


namespace NUMINAMATH_GPT_find_purchase_price_l1516_151649

noncomputable def purchase_price (a : ℝ) : ℝ := a
def retail_price : ℝ := 1100
def discount_rate : ℝ := 0.8
def profit_rate : ℝ := 0.1

theorem find_purchase_price (a : ℝ) (h : purchase_price a * (1 + profit_rate) = retail_price * discount_rate) : a = 800 := by
  sorry

end NUMINAMATH_GPT_find_purchase_price_l1516_151649


namespace NUMINAMATH_GPT_student_arrangement_l1516_151627

theorem student_arrangement (students : Fin 6 → Prop)
  (A : (students 0) ∨ (students 5) → False)
  (females_adj : ∃ (i : Fin 6), i < 5 ∧ students i → students (i + 1))
  : ∃! n, n = 96 := by
  sorry

end NUMINAMATH_GPT_student_arrangement_l1516_151627


namespace NUMINAMATH_GPT_probability_product_zero_probability_product_negative_l1516_151603

def given_set : List ℤ := [-3, -2, -1, 0, 5, 6, 7]

def num_pairs : ℕ := 21

theorem probability_product_zero :
  (6 : ℚ) / num_pairs = 2 / 7 := sorry

theorem probability_product_negative :
  (9 : ℚ) / num_pairs = 3 / 7 := sorry

end NUMINAMATH_GPT_probability_product_zero_probability_product_negative_l1516_151603


namespace NUMINAMATH_GPT_csc_square_value_l1516_151641

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 ∨ x = 1 then 0 -- provision for the illegal inputs as defined in the question
else 1/(x / (x - 1))

theorem csc_square_value (t : ℝ) (ht : 0 ≤ t ∧ t ≤ π / 2) :
  f (1 / (Real.sin t)^2) = (Real.cos t)^2 :=
by
  sorry

end NUMINAMATH_GPT_csc_square_value_l1516_151641


namespace NUMINAMATH_GPT_diff_is_multiple_of_9_l1516_151686

-- Definitions
def orig_num (a b : ℕ) : ℕ := 10 * a + b
def new_num (a b : ℕ) : ℕ := 10 * b + a

-- Statement of the mathematical proof problem
theorem diff_is_multiple_of_9 (a b : ℕ) : 
  9 ∣ (new_num a b - orig_num a b) :=
by
  sorry

end NUMINAMATH_GPT_diff_is_multiple_of_9_l1516_151686


namespace NUMINAMATH_GPT_solve_inequality_a_eq_2_solve_inequality_a_in_R_l1516_151626

theorem solve_inequality_a_eq_2 :
  {x : ℝ | x > 2 ∨ x < 1} = {x : ℝ | x^2 - 3*x + 2 > 0} :=
sorry

theorem solve_inequality_a_in_R (a : ℝ) :
  {x : ℝ | 
    (a > 1 ∧ (x > a ∨ x < 1)) ∨ 
    (a = 1 ∧ x ≠ 1) ∨ 
    (a < 1 ∧ (x > 1 ∨ x < a))
  } = 
  {x : ℝ | x^2 - (1 + a)*x + a > 0} :=
sorry

end NUMINAMATH_GPT_solve_inequality_a_eq_2_solve_inequality_a_in_R_l1516_151626


namespace NUMINAMATH_GPT_girl_walked_distance_l1516_151674

-- Define the conditions
def speed : ℝ := 5 -- speed in kmph
def time : ℝ := 6 -- time in hours

-- Define the distance calculation
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- The proof statement that we need to show
theorem girl_walked_distance :
  distance speed time = 30 := by
  sorry

end NUMINAMATH_GPT_girl_walked_distance_l1516_151674


namespace NUMINAMATH_GPT_evaluate_neg_64_exp_4_over_3_l1516_151654

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_neg_64_exp_4_over_3_l1516_151654


namespace NUMINAMATH_GPT_total_games_l1516_151659

/-- Definition of the number of games Alyssa went to this year -/
def games_this_year : Nat := 11

/-- Definition of the number of games Alyssa went to last year -/
def games_last_year : Nat := 13

/-- Definition of the number of games Alyssa plans to go to next year -/
def games_next_year : Nat := 15

/-- Statement to prove the total number of games Alyssa will go to in all -/
theorem total_games : games_this_year + games_last_year + games_next_year = 39 := by
  -- A sorry placeholder to skip the proof
  sorry

end NUMINAMATH_GPT_total_games_l1516_151659


namespace NUMINAMATH_GPT_part1_part2_l1516_151677

variable {α : Type*} [LinearOrderedField α]

-- Definitions based on given problem conditions.
def arithmetic_seq(a_n : ℕ → α) := ∃ a1 d, ∀ n, a_n n = a1 + ↑(n - 1) * d

noncomputable def a10_seq := (30 : α)
noncomputable def a20_seq := (50 : α)

-- Theorem statements to prove:
theorem part1 {a_n : ℕ → α} (h : arithmetic_seq a_n) (h10 : a_n 10 = a10_seq) (h20 : a_n 20 = a20_seq) :
  ∀ n, a_n n = 2 * ↑n + 10 := sorry

theorem part2 {a_n : ℕ → α} (h : arithmetic_seq a_n) (h10 : a_n 10 = a10_seq) (h20 : a_n 20 = a20_seq)
  (Sn : α) (hSn : Sn = 242) :
  ∃ n, Sn = (↑n / 2) * (2 * 12 + (↑n - 1) * 2) ∧ n = 11 := sorry

end NUMINAMATH_GPT_part1_part2_l1516_151677


namespace NUMINAMATH_GPT_annual_interest_rate_l1516_151675

theorem annual_interest_rate (P A : ℝ) (n t : ℕ) (r : ℝ) 
  (hP : P = 700) 
  (hA : A = 771.75) 
  (hn : n = 2) 
  (ht : t = 1) 
  (h : A = P * (1 + r / n) ^ (n * t)) : 
  r = 0.10 := 
by 
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_annual_interest_rate_l1516_151675


namespace NUMINAMATH_GPT_jose_is_12_years_older_l1516_151689

theorem jose_is_12_years_older (J M : ℕ) (h1 : M = 14) (h2 : J + M = 40) : J - M = 12 :=
by
  sorry

end NUMINAMATH_GPT_jose_is_12_years_older_l1516_151689


namespace NUMINAMATH_GPT_simplify_fraction_l1516_151615

theorem simplify_fraction (n : Nat) : (2^(n+4) - 3 * 2^n) / (2 * 2^(n+3)) = 13 / 16 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1516_151615


namespace NUMINAMATH_GPT_seashells_remaining_l1516_151647

def initial_seashells : ℕ := 35
def given_seashells : ℕ := 18

theorem seashells_remaining : initial_seashells - given_seashells = 17 := by
  sorry

end NUMINAMATH_GPT_seashells_remaining_l1516_151647


namespace NUMINAMATH_GPT_find_number_l1516_151657

theorem find_number (n : ℝ) (h : 1 / 2 * n + 7 = 17) : n = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1516_151657


namespace NUMINAMATH_GPT_initial_average_is_100_l1516_151691

-- Definitions based on the conditions from step a)
def students : ℕ := 10
def wrong_mark : ℕ := 90
def correct_mark : ℕ := 10
def correct_average : ℝ := 92

-- Initial average marks before correcting the error
def initial_average_marks (A : ℝ) : Prop :=
  10 * A = (students * correct_average) + (wrong_mark - correct_mark)

theorem initial_average_is_100 :
  ∃ A : ℝ, initial_average_marks A ∧ A = 100 :=
by {
  -- We are defining the placeholder for the actual proof.
  sorry
}

end NUMINAMATH_GPT_initial_average_is_100_l1516_151691


namespace NUMINAMATH_GPT_nathaniel_tickets_l1516_151664

theorem nathaniel_tickets :
  ∀ (B S : ℕ),
  (7 * B + 4 * S + 11 = 128) →
  (B + S = 20) :=
by
  intros B S h
  sorry

end NUMINAMATH_GPT_nathaniel_tickets_l1516_151664


namespace NUMINAMATH_GPT_badminton_members_count_l1516_151673

-- Definitions of the conditions
def total_members : ℕ := 40
def tennis_players : ℕ := 18
def neither_sport : ℕ := 5
def both_sports : ℕ := 3
def badminton_players : ℕ := 20 -- The answer we need to prove

-- The proof statement
theorem badminton_members_count :
  total_members = (badminton_players + tennis_players - both_sports) + neither_sport :=
by
  -- The proof is outlined here
  sorry

end NUMINAMATH_GPT_badminton_members_count_l1516_151673


namespace NUMINAMATH_GPT_percentage_calculation_l1516_151608

theorem percentage_calculation
  (x : ℝ)
  (hx : x = 16)
  (h : 0.15 * 40 - (P * x) = 2) :
  P = 0.25 := by
  sorry

end NUMINAMATH_GPT_percentage_calculation_l1516_151608


namespace NUMINAMATH_GPT_number_of_solutions_depends_on_a_l1516_151624

theorem number_of_solutions_depends_on_a (a : ℝ) : 
  (∀ x : ℝ, 2^(3 * x) + 4 * a * 2^(2 * x) + a^2 * 2^x - 6 * a^3 = 0) → 
  (if a = 0 then 0 else if a > 0 then 1 else 2) = 
  (if a = 0 then 0 else if a > 0 then 1 else 2) :=
by 
  sorry

end NUMINAMATH_GPT_number_of_solutions_depends_on_a_l1516_151624


namespace NUMINAMATH_GPT_ratio_of_functions_l1516_151683

def f (x : ℕ) : ℕ := 3 * x + 4
def g (x : ℕ) : ℕ := 4 * x - 3

theorem ratio_of_functions :
  f (g (f 3)) * 121 = 151 * g (f (g 3)) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_functions_l1516_151683
