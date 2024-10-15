import Mathlib

namespace NUMINAMATH_GPT_gcd_221_195_l896_89673

-- Define the two numbers
def a := 221
def b := 195

-- Statement of the problem: the gcd of a and b is 13
theorem gcd_221_195 : Nat.gcd a b = 13 := 
by
  sorry

end NUMINAMATH_GPT_gcd_221_195_l896_89673


namespace NUMINAMATH_GPT_probability_single_trial_l896_89696

theorem probability_single_trial 
  (p : ℝ) 
  (h₁ : ∀ n : ℕ, 1 ≤ n → ∃ x : ℝ, x = (1 - (1 - p) ^ n)) 
  (h₂ : 1 - (1 - p) ^ 4 = 65 / 81) : 
  p = 1 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_probability_single_trial_l896_89696


namespace NUMINAMATH_GPT_y_completion_days_l896_89651

theorem y_completion_days (d : ℕ) (h : (12 : ℚ) / d + 1 / 4 = 1) : d = 16 :=
by
  sorry

end NUMINAMATH_GPT_y_completion_days_l896_89651


namespace NUMINAMATH_GPT_lottery_winning_situations_l896_89672

theorem lottery_winning_situations :
  let num_tickets := 8
  let first_prize := 1
  let second_prize := 1
  let third_prize := 1
  let non_winning := 5
  let customers := 4
  let tickets_per_customer := 2
  let total_ways := 24 + 36
  total_ways = 60 :=
by
  let num_tickets := 8
  let first_prize := 1
  let second_prize := 1
  let third_prize := 1
  let non_winning := 5
  let customers := 4
  let tickets_per_customer := 2
  let total_ways := 24 + 36

  -- Skipping proof steps
  sorry

end NUMINAMATH_GPT_lottery_winning_situations_l896_89672


namespace NUMINAMATH_GPT_rain_all_three_days_is_six_percent_l896_89655

-- Definitions based on conditions from step a)
def P_rain_friday : ℚ := 2 / 5
def P_rain_saturday : ℚ := 1 / 2
def P_rain_sunday : ℚ := 3 / 10

-- The probability it will rain on all three days
def P_rain_all_three_days : ℚ := P_rain_friday * P_rain_saturday * P_rain_sunday

-- The Lean 4 theorem statement
theorem rain_all_three_days_is_six_percent : P_rain_all_three_days * 100 = 6 := by
  sorry

end NUMINAMATH_GPT_rain_all_three_days_is_six_percent_l896_89655


namespace NUMINAMATH_GPT_cannot_form_right_triangle_l896_89665

theorem cannot_form_right_triangle (a b c : ℕ) (h_a : a = 3) (h_b : b = 5) (h_c : c = 7) : 
  a^2 + b^2 ≠ c^2 :=
by 
  rw [h_a, h_b, h_c]
  sorry

end NUMINAMATH_GPT_cannot_form_right_triangle_l896_89665


namespace NUMINAMATH_GPT_Jackie_has_more_apples_l896_89688

def Adam_apples : Nat := 9
def Jackie_apples : Nat := 10

theorem Jackie_has_more_apples : Jackie_apples - Adam_apples = 1 := by
  sorry

end NUMINAMATH_GPT_Jackie_has_more_apples_l896_89688


namespace NUMINAMATH_GPT_min_value_of_E_l896_89697

noncomputable def E : ℝ := sorry

theorem min_value_of_E :
  (∀ x : ℝ, |E| + |x + 7| + |x - 5| ≥ 12) →
  (∃ x : ℝ, |x + 7| + |x - 5| = 12 → |E| = 0) :=
sorry

end NUMINAMATH_GPT_min_value_of_E_l896_89697


namespace NUMINAMATH_GPT_max_value_of_expression_l896_89636

theorem max_value_of_expression :
  ∃ x : ℝ, ∀ y : ℝ, -x^2 + 4*x + 10 ≤ -y^2 + 4*y + 10 ∧ -x^2 + 4*x + 10 = 14 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l896_89636


namespace NUMINAMATH_GPT_complex_ab_value_l896_89684

theorem complex_ab_value (a b : ℝ) (i : ℂ) (h_i : i = Complex.I) (h_z : a + b * i = (4 + 3 * i) * i) : a * b = -12 :=
by {
  sorry
}

end NUMINAMATH_GPT_complex_ab_value_l896_89684


namespace NUMINAMATH_GPT_sum_of_roots_l896_89660

-- Defined the equation x^2 - 7x + 2 - 16 = 0 as x^2 - 7x - 14 = 0
def equation (x : ℝ) := x^2 - 7 * x - 14 = 0 

-- State the theorem leveraging the above condition
theorem sum_of_roots : 
  (∃ x1 x2 : ℝ, equation x1 ∧ equation x2 ∧ x1 ≠ x2) →
  (∃ sum : ℝ, sum = 7) := by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l896_89660


namespace NUMINAMATH_GPT_correctStatement_l896_89601

def isValidInput : String → Bool
| "INPUT a, b, c;" => true
| "INPUT x=3;" => false
| _ => false

def isValidOutput : String → Bool
| "PRINT 20,3*2." => true
| "PRINT A=4;" => false
| _ => false

def isValidStatement : String → Bool
| stmt => (isValidInput stmt ∨ isValidOutput stmt)

theorem correctStatement : isValidStatement "PRINT 20,3*2." = true ∧ 
                           ¬(isValidStatement "INPUT a; b; c;" = true) ∧ 
                           ¬(isValidStatement "INPUT x=3;" = true) ∧ 
                           ¬(isValidStatement "PRINT A=4;" = true) := 
by sorry

end NUMINAMATH_GPT_correctStatement_l896_89601


namespace NUMINAMATH_GPT_numerator_of_fraction_l896_89687

theorem numerator_of_fraction (x : ℤ) (h : (x : ℚ) / (4 * x - 5) = 3 / 7) : x = 3 := 
sorry

end NUMINAMATH_GPT_numerator_of_fraction_l896_89687


namespace NUMINAMATH_GPT_adjacent_number_in_grid_l896_89678

def adjacent_triangle_number (k n: ℕ) : ℕ :=
  if k % 2 = 1 then n - k else n + k

theorem adjacent_number_in_grid (n : ℕ) (bound: n = 350) :
  let k := Nat.ceil (Real.sqrt n)
  let m := (k * k) - n
  k = 19 ∧ m = 19 →
  adjacent_triangle_number k n = 314 :=
by
  sorry

end NUMINAMATH_GPT_adjacent_number_in_grid_l896_89678


namespace NUMINAMATH_GPT_daniela_total_spent_l896_89671

-- Step d) Rewrite the math proof problem
theorem daniela_total_spent
    (shoe_price : ℤ) (dress_price : ℤ) (shoe_discount : ℤ) (dress_discount : ℤ)
    (shoe_count : ℤ)
    (shoe_original_price : shoe_price = 50)
    (dress_original_price : dress_price = 100)
    (shoe_discount_rate : shoe_discount = 40)
    (dress_discount_rate : dress_discount = 20)
    (shoe_total_count : shoe_count = 2)
    : shoe_count * (shoe_price - (shoe_price * shoe_discount / 100)) + (dress_price - (dress_price * dress_discount / 100)) = 140 := by 
    sorry

end NUMINAMATH_GPT_daniela_total_spent_l896_89671


namespace NUMINAMATH_GPT_find_a_of_perpendicular_tangent_and_line_l896_89616

open Real

theorem find_a_of_perpendicular_tangent_and_line :
  let e := Real.exp 1
  let slope_tangent := 1 / e
  let slope_line (a : ℝ) := a
  let tangent_perpendicular := ∀ (a : ℝ), slope_tangent * slope_line a = -1
  tangent_perpendicular -> ∃ a : ℝ, a = -e :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_of_perpendicular_tangent_and_line_l896_89616


namespace NUMINAMATH_GPT_tails_and_die_1_or_2_l896_89661

noncomputable def fairCoinFlipProbability : ℚ := 1 / 2
noncomputable def fairDieRollProbability : ℚ := 1 / 6
noncomputable def combinedProbability : ℚ := fairCoinFlipProbability * (fairDieRollProbability + fairDieRollProbability)

theorem tails_and_die_1_or_2 :
  combinedProbability = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_tails_and_die_1_or_2_l896_89661


namespace NUMINAMATH_GPT_triangle_is_obtuse_l896_89630

-- Define the conditions of the problem
def angles (x : ℝ) : Prop :=
  2 * x + 3 * x + 6 * x = 180

def obtuse_angle (x : ℝ) : Prop :=
  6 * x > 90

-- State the theorem
theorem triangle_is_obtuse (x : ℝ) (hx : angles x) : obtuse_angle x :=
sorry

end NUMINAMATH_GPT_triangle_is_obtuse_l896_89630


namespace NUMINAMATH_GPT_patty_weighs_more_l896_89625

variable (R : ℝ) (P_0 : ℝ) (L : ℝ) (P : ℝ) (D : ℝ)

theorem patty_weighs_more :
  (R = 100) →
  (P_0 = 4.5 * R) →
  (L = 235) →
  (P = P_0 - L) →
  (D = P - R) →
  D = 115 := by
  sorry

end NUMINAMATH_GPT_patty_weighs_more_l896_89625


namespace NUMINAMATH_GPT_vinces_bus_ride_length_l896_89607

theorem vinces_bus_ride_length (zachary_ride : ℝ) (vince_extra : ℝ) (vince_ride : ℝ) :
  zachary_ride = 0.5 →
  vince_extra = 0.13 →
  vince_ride = zachary_ride + vince_extra →
  vince_ride = 0.63 :=
by
  intros hz hv he
  -- proof steps here
  sorry

end NUMINAMATH_GPT_vinces_bus_ride_length_l896_89607


namespace NUMINAMATH_GPT_reflect_point_value_l896_89623

theorem reflect_point_value (mx b : ℝ) 
  (start end_ : ℝ × ℝ)
  (Hstart : start = (2, 3))
  (Hend : end_ = (10, 7))
  (Hreflection : ∃ m b: ℝ, (end_.fst, end_.snd) = 
              (2 * ((5 / 2) - (1 / 2) * 3 * m - b), 2 * ((5 / 2) + (1 / 2) * 3)) ∧ m = -2)
  : m + b = 15 :=
sorry

end NUMINAMATH_GPT_reflect_point_value_l896_89623


namespace NUMINAMATH_GPT_zero_function_l896_89690

noncomputable def f : ℝ → ℝ := sorry -- Let it be a placeholder for now.

theorem zero_function (a b : ℝ) (h_cont : ContinuousOn f (Set.Icc a b))
  (h_int : ∀ n : ℕ, ∫ x in a..b, (x : ℝ)^n * f x = 0) :
  ∀ x ∈ Set.Icc a b, f x = 0 :=
by
  sorry -- placeholder for the proof

end NUMINAMATH_GPT_zero_function_l896_89690


namespace NUMINAMATH_GPT_sum_of_values_satisfying_equation_l896_89670

noncomputable def sum_of_roots_of_quadratic (a b c : ℝ) : ℝ := -b / a

theorem sum_of_values_satisfying_equation :
  (∃ x : ℝ, (x^2 - 5 * x + 7 = 9)) →
  sum_of_roots_of_quadratic 1 (-5) (-2) = 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_values_satisfying_equation_l896_89670


namespace NUMINAMATH_GPT_boys_in_choir_l896_89657

theorem boys_in_choir
  (h1 : 20 + 2 * 20 + 16 + b = 88)
  : b = 12 :=
by
  sorry

end NUMINAMATH_GPT_boys_in_choir_l896_89657


namespace NUMINAMATH_GPT_order_of_m_n_p_q_l896_89652

variable {m n p q : ℝ} -- Define the variables as real numbers

theorem order_of_m_n_p_q (h1 : m < n) 
                         (h2 : p < q) 
                         (h3 : (p - m) * (p - n) < 0) 
                         (h4 : (q - m) * (q - n) < 0) : 
    m < p ∧ p < q ∧ q < n := 
by
  sorry

end NUMINAMATH_GPT_order_of_m_n_p_q_l896_89652


namespace NUMINAMATH_GPT_total_computers_needed_l896_89691

theorem total_computers_needed
    (initial_students : ℕ)
    (students_per_computer : ℕ)
    (additional_students : ℕ)
    (initial_computers : ℕ := initial_students / students_per_computer)
    (total_computers : ℕ := initial_computers + (additional_students / students_per_computer))
    (h1 : initial_students = 82)
    (h2 : students_per_computer = 2)
    (h3 : additional_students = 16) :
    total_computers = 49 :=
by
  -- The proof would normally go here
  sorry

end NUMINAMATH_GPT_total_computers_needed_l896_89691


namespace NUMINAMATH_GPT_abs_ineq_solution_set_l896_89628

theorem abs_ineq_solution_set {x : ℝ} : |x + 1| - |x - 3| ≥ 2 ↔ x ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_abs_ineq_solution_set_l896_89628


namespace NUMINAMATH_GPT_find_x_modulo_l896_89680

theorem find_x_modulo (k : ℤ) : ∃ x : ℤ, x = 18 + 31 * k ∧ ((37 * x) % 31 = 15) := by
  sorry

end NUMINAMATH_GPT_find_x_modulo_l896_89680


namespace NUMINAMATH_GPT_meadow_trees_count_l896_89614

theorem meadow_trees_count (n : ℕ) (f s m : ℕ → ℕ) :
  (f 20 = s 7) ∧ (f 7 = s 94) ∧ (s 7 > f 20) → 
  n = 100 :=
by
  sorry

end NUMINAMATH_GPT_meadow_trees_count_l896_89614


namespace NUMINAMATH_GPT_brinley_animals_count_l896_89618

theorem brinley_animals_count :
  let snakes := 100
  let arctic_foxes := 80
  let leopards := 20
  let bee_eaters := 10 * ((snakes / 2) + (2 * leopards))
  let cheetahs := 4 * (arctic_foxes - leopards)
  let alligators := 3 * (snakes * arctic_foxes * leopards)
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators = 481340 := by
  sorry

end NUMINAMATH_GPT_brinley_animals_count_l896_89618


namespace NUMINAMATH_GPT_three_digit_numbers_mod_1000_l896_89693

theorem three_digit_numbers_mod_1000 (n : ℕ) (h_lower : 100 ≤ n) (h_upper : n ≤ 999) : 
  (n^2 ≡ n [MOD 1000]) ↔ (n = 376 ∨ n = 625) :=
by sorry

end NUMINAMATH_GPT_three_digit_numbers_mod_1000_l896_89693


namespace NUMINAMATH_GPT_volume_ratio_l896_89602

namespace Geometry

variables {Point : Type} [MetricSpace Point]

noncomputable def volume_pyramid (A B1 C1 D1 : Point) : ℝ := sorry

theorem volume_ratio 
  (A B1 B2 C1 C2 D1 D2 : Point) 
  (hA_B1: dist A B1 ≠ 0) (hA_B2: dist A B2 ≠ 0)
  (hA_C1: dist A C1 ≠ 0) (hA_C2: dist A C2 ≠ 0)
  (hA_D1: dist A D1 ≠ 0) (hA_D2: dist A D2 ≠ 0) :
  (volume_pyramid A B1 C1 D1 / volume_pyramid A B2 C2 D2) = 
    (dist A B1 * dist A C1 * dist A D1) / (dist A B2 * dist A C2 * dist A D2) := 
sorry

end Geometry

end NUMINAMATH_GPT_volume_ratio_l896_89602


namespace NUMINAMATH_GPT_f_neg_def_l896_89633

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f x

def f_pos_def (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f x = x * (1 - x)

theorem f_neg_def (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : f_pos_def f) :
  ∀ x : ℝ, x < 0 → f x = x * (1 + x) :=
by
  sorry

end NUMINAMATH_GPT_f_neg_def_l896_89633


namespace NUMINAMATH_GPT_anne_speed_ratio_l896_89698

variable (B A A' : ℝ)

theorem anne_speed_ratio (h1 : A = 1 / 12)
                        (h2 : B + A = 1 / 4)
                        (h3 : B + A' = 1 / 3) : 
                        A' / A = 2 := 
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_anne_speed_ratio_l896_89698


namespace NUMINAMATH_GPT_silk_diameter_scientific_notation_l896_89692

-- Definition of the given condition
def silk_diameter := 0.000014 

-- The goal to be proved
theorem silk_diameter_scientific_notation : silk_diameter = 1.4 * 10^(-5) := 
by 
  sorry

end NUMINAMATH_GPT_silk_diameter_scientific_notation_l896_89692


namespace NUMINAMATH_GPT_abs_sum_zero_implies_diff_eq_five_l896_89658

theorem abs_sum_zero_implies_diff_eq_five (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a - b = 5 :=
  sorry

end NUMINAMATH_GPT_abs_sum_zero_implies_diff_eq_five_l896_89658


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l896_89695

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h : a 3 + a 4 + a 5 = 12) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l896_89695


namespace NUMINAMATH_GPT_imaginary_part_of_1_minus_2i_l896_89619

def i := Complex.I

theorem imaginary_part_of_1_minus_2i : Complex.im (1 - 2 * i) = -2 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_1_minus_2i_l896_89619


namespace NUMINAMATH_GPT_quadrilateral_area_l896_89638

noncomputable def AB : ℝ := 3
noncomputable def BC : ℝ := 3
noncomputable def CD : ℝ := 4
noncomputable def DA : ℝ := 8
noncomputable def angle_DAB_add_angle_ABC : ℝ := 180

theorem quadrilateral_area :
  AB = 3 ∧ BC = 3 ∧ CD = 4 ∧ DA = 8 ∧ angle_DAB_add_angle_ABC = 180 →
  ∃ area : ℝ, area = 13.2 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadrilateral_area_l896_89638


namespace NUMINAMATH_GPT_last_digit_base5_of_M_l896_89683

theorem last_digit_base5_of_M (d e f : ℕ) (hd : d < 5) (he : e < 5) (hf : f < 5)
  (h : 25 * d + 5 * e + f = 64 * f + 8 * e + d) : f = 0 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_base5_of_M_l896_89683


namespace NUMINAMATH_GPT_teacher_age_l896_89626

theorem teacher_age (avg_age_students : ℕ) (num_students : ℕ) (new_avg_with_teacher : ℕ) (num_total : ℕ) 
  (total_age_students : ℕ)
  (h1 : avg_age_students = 10)
  (h2 : num_students = 15)
  (h3 : new_avg_with_teacher = 11)
  (h4 : num_total = 16)
  (h5 : total_age_students = num_students * avg_age_students) :
  num_total * new_avg_with_teacher - total_age_students = 26 :=
by sorry

end NUMINAMATH_GPT_teacher_age_l896_89626


namespace NUMINAMATH_GPT_value_of_expression_l896_89643

theorem value_of_expression
  (a b c : ℝ)
  (h1 : |a - b| = 1)
  (h2 : |b - c| = 1)
  (h3 : |c - a| = 2)
  (h4 : a * b * c = 60) :
  (a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c) = 1 / 10 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l896_89643


namespace NUMINAMATH_GPT_median_of_64_consecutive_integers_l896_89663

theorem median_of_64_consecutive_integers (n : ℕ) (S : ℕ) (h1 : n = 64) (h2 : S = 8^4) :
  S / n = 64 :=
by
  -- to skip the proof
  sorry

end NUMINAMATH_GPT_median_of_64_consecutive_integers_l896_89663


namespace NUMINAMATH_GPT_time_saved_1200_miles_l896_89675

theorem time_saved_1200_miles
  (distance : ℕ)
  (speed1 speed2 : ℕ)
  (h_distance : distance = 1200)
  (h_speed1 : speed1 = 60)
  (h_speed2 : speed2 = 50) :
  (distance / speed2) - (distance / speed1) = 4 :=
by
  sorry

end NUMINAMATH_GPT_time_saved_1200_miles_l896_89675


namespace NUMINAMATH_GPT_divides_polynomial_l896_89646

theorem divides_polynomial (n : ℕ) (x : ℤ) (hn : 0 < n) :
  (x^2 + x + 1) ∣ (x^(n+2) + (x+1)^(2*n+1)) :=
sorry

end NUMINAMATH_GPT_divides_polynomial_l896_89646


namespace NUMINAMATH_GPT_triangle_area_ab_l896_89634

theorem triangle_area_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0.5 * (12 / a) * (12 / b) = 12) : a * b = 6 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_ab_l896_89634


namespace NUMINAMATH_GPT_rectangle_ratio_l896_89639

theorem rectangle_ratio (L B : ℕ) (hL : L = 250) (hB : B = 160) : L / B = 25 / 16 := by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l896_89639


namespace NUMINAMATH_GPT_minimum_small_droppers_l896_89631

/-
Given:
1. A total volume to be filled: V = 265 milliliters.
2. Small droppers can hold: s = 19 milliliters each.
3. No large droppers are used.

Prove:
The minimum number of small droppers required to fill the container completely is 14.
-/

theorem minimum_small_droppers (V s: ℕ) (hV: V = 265) (hs: s = 19) : 
  ∃ n: ℕ, n = 14 ∧ n * s ≥ V ∧ (n - 1) * s < V :=
by
  sorry  -- proof to be provided

end NUMINAMATH_GPT_minimum_small_droppers_l896_89631


namespace NUMINAMATH_GPT_ny_mets_fans_count_l896_89653

theorem ny_mets_fans_count (Y M R : ℕ) (h1 : 3 * M = 2 * Y) (h2 : 4 * R = 5 * M) (h3 : Y + M + R = 390) : M = 104 := 
by
  sorry

end NUMINAMATH_GPT_ny_mets_fans_count_l896_89653


namespace NUMINAMATH_GPT_pure_imaginary_a_zero_l896_89679

theorem pure_imaginary_a_zero (a : ℝ) (i : ℂ) (hi : i^2 = -1) :
  (z = (1 - (a:ℝ)^2 * i) / i) ∧ (∀ (z : ℂ), z.re = 0 → z = (0 : ℂ)) → a = 0 :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_a_zero_l896_89679


namespace NUMINAMATH_GPT_participants_begin_competition_l896_89682

theorem participants_begin_competition (x : ℝ) 
  (h1 : 0.4 * x * (1 / 4) = 16) : 
  x = 160 := 
by
  sorry

end NUMINAMATH_GPT_participants_begin_competition_l896_89682


namespace NUMINAMATH_GPT_ceil_y_squared_possibilities_l896_89606

theorem ceil_y_squared_possibilities (y : ℝ) (h : ⌈y⌉ = 15) : 
  ∃ n : ℕ, (n = 29) ∧ (∀ z : ℕ, ⌈y^2⌉ = z → (197 ≤ z ∧ z ≤ 225)) :=
by
  sorry

end NUMINAMATH_GPT_ceil_y_squared_possibilities_l896_89606


namespace NUMINAMATH_GPT_ivanka_woody_total_months_l896_89647

theorem ivanka_woody_total_months
  (woody_years : ℝ)
  (months_per_year : ℝ)
  (additional_months : ℕ)
  (woody_months : ℝ)
  (ivanka_months : ℝ)
  (total_months : ℝ)
  (h1 : woody_years = 1.5)
  (h2 : months_per_year = 12)
  (h3 : additional_months = 3)
  (h4 : woody_months = woody_years * months_per_year)
  (h5 : ivanka_months = woody_months + additional_months)
  (h6 : total_months = woody_months + ivanka_months) :
  total_months = 39 := by
  sorry

end NUMINAMATH_GPT_ivanka_woody_total_months_l896_89647


namespace NUMINAMATH_GPT_roadRepairDays_l896_89645

-- Definitions from the conditions
def dailyRepairLength1 : ℕ := 6
def daysToFinish1 : ℕ := 8
def totalLengthOfRoad : ℕ := dailyRepairLength1 * daysToFinish1
def dailyRepairLength2 : ℕ := 8
def daysToFinish2 : ℕ := totalLengthOfRoad / dailyRepairLength2

-- Theorem to be proven
theorem roadRepairDays :
  daysToFinish2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_roadRepairDays_l896_89645


namespace NUMINAMATH_GPT_solution_set_for_inequality_l896_89640

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem solution_set_for_inequality
  (h1 : is_odd f)
  (h2 : f 2 = 0)
  (h3 : ∀ x > 0, x * deriv f x - f x < 0) :
  {x : ℝ | f x / x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_GPT_solution_set_for_inequality_l896_89640


namespace NUMINAMATH_GPT_children_tickets_sold_l896_89644

theorem children_tickets_sold (A C : ℝ) (h1 : A + C = 400) (h2 : 6 * A + 4.5 * C = 2100) : C = 200 :=
sorry

end NUMINAMATH_GPT_children_tickets_sold_l896_89644


namespace NUMINAMATH_GPT_total_students_l896_89612

theorem total_students (T : ℝ) (h1 : 0.3 * T =  0.7 * T - 616) : T = 880 :=
by sorry

end NUMINAMATH_GPT_total_students_l896_89612


namespace NUMINAMATH_GPT_tangency_point_of_parabolas_l896_89641

theorem tangency_point_of_parabolas :
  ∃ (x y : ℝ), y = x^2 + 17 * x + 40 ∧ x = y^2 + 51 * y + 650 ∧ x = -7 ∧ y = -25 :=
by
  sorry

end NUMINAMATH_GPT_tangency_point_of_parabolas_l896_89641


namespace NUMINAMATH_GPT_pine_cones_on_roof_l896_89610

theorem pine_cones_on_roof 
  (num_trees : ℕ) 
  (pine_cones_per_tree : ℕ) 
  (percent_on_roof : ℝ) 
  (weight_per_pine_cone : ℝ) 
  (h1 : num_trees = 8)
  (h2 : pine_cones_per_tree = 200)
  (h3 : percent_on_roof = 0.30)
  (h4 : weight_per_pine_cone = 4) : 
  (num_trees * pine_cones_per_tree * percent_on_roof * weight_per_pine_cone = 1920) :=
by
  sorry

end NUMINAMATH_GPT_pine_cones_on_roof_l896_89610


namespace NUMINAMATH_GPT_robotics_club_non_participants_l896_89669

theorem robotics_club_non_participants (club_students electronics_students programming_students both_students : ℕ) 
  (h1 : club_students = 80) 
  (h2 : electronics_students = 45) 
  (h3 : programming_students = 50) 
  (h4 : both_students = 30) : 
  club_students - (electronics_students - both_students + programming_students - both_students + both_students) = 15 :=
by
  -- The proof would be here
  sorry

end NUMINAMATH_GPT_robotics_club_non_participants_l896_89669


namespace NUMINAMATH_GPT_quadratic_roots_conditions_l896_89668

-- Definitions of the given conditions.
variables (a b c : ℝ)  -- Coefficients of the quadratic trinomial
variable (h : b^2 - 4 * a * c ≥ 0)  -- Given condition that the discriminant is non-negative

-- Statement to prove:
theorem quadratic_roots_conditions (a b c : ℝ) (h : b^2 - 4 * a * c ≥ 0) :
  ¬(∀ x : ℝ, a^2 * x^2 + b^2 * x + c^2 = 0) ∧ (∀ x : ℝ, a^3 * x^2 + b^3 * x + c^3 = 0 → b^6 - 4 * a^3 * c^3 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_conditions_l896_89668


namespace NUMINAMATH_GPT_isosceles_triangle_apex_angle_l896_89605

theorem isosceles_triangle_apex_angle (a b c : ℝ) (ha : a = 40) (hb : b = 40) (hc : b = c) :
  (a + b + c = 180) → (c = 100 ∨ a = 40) :=
by
-- We start the proof and provide the conditions.
  sorry  -- Lean expects the proof here.

end NUMINAMATH_GPT_isosceles_triangle_apex_angle_l896_89605


namespace NUMINAMATH_GPT_number_of_friends_l896_89624

def money_emma : ℕ := 8

def money_daya : ℕ := money_emma + (money_emma * 25 / 100)

def money_jeff : ℕ := (2 * money_daya) / 5

def money_brenda : ℕ := money_jeff + 4

def money_brenda_condition : Prop := money_brenda = 8

def friends_pooling_pizza : ℕ := 4

theorem number_of_friends (h : money_brenda_condition) : friends_pooling_pizza = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_friends_l896_89624


namespace NUMINAMATH_GPT_math_group_question_count_l896_89609

theorem math_group_question_count (m n : ℕ) (h : m * (m - 1) + m * n + n = 51) : m = 6 ∧ n = 3 := 
sorry

end NUMINAMATH_GPT_math_group_question_count_l896_89609


namespace NUMINAMATH_GPT_arithmetic_mean_difference_l896_89689

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 20) : 
  r - p = 20 := 
sorry

end NUMINAMATH_GPT_arithmetic_mean_difference_l896_89689


namespace NUMINAMATH_GPT_man_swims_distance_back_l896_89635

def swimming_speed_still_water : ℝ := 8
def speed_of_water : ℝ := 4
def time_taken_against_current : ℝ := 2
def distance_swum : ℝ := 8

theorem man_swims_distance_back :
  (distance_swum = (swimming_speed_still_water - speed_of_water) * time_taken_against_current) :=
by
  -- The proof will be filled in later.
  sorry

end NUMINAMATH_GPT_man_swims_distance_back_l896_89635


namespace NUMINAMATH_GPT_tangent_line_through_point_and_circle_l896_89648

noncomputable def tangent_line_equation : String :=
  "y - 1 = 0"

theorem tangent_line_through_point_and_circle :
  ∀ (line_eq: String), 
  (∀ (x y: ℝ), (x - 1) ^ 2 + y ^ 2 = 1 ∧ (x, y) = (1, 1) → y - 1 = 0) →
  line_eq = tangent_line_equation :=
by
  intro line_eq h
  sorry

end NUMINAMATH_GPT_tangent_line_through_point_and_circle_l896_89648


namespace NUMINAMATH_GPT_baker_cakes_l896_89659

theorem baker_cakes : (62.5 + 149.25 - 144.75 = 67) :=
by
  sorry

end NUMINAMATH_GPT_baker_cakes_l896_89659


namespace NUMINAMATH_GPT_find_b_minus_a_l896_89654

theorem find_b_minus_a (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a - 9 * b + 18 * a * b = 2018) : b - a = 223 :=
sorry

end NUMINAMATH_GPT_find_b_minus_a_l896_89654


namespace NUMINAMATH_GPT_k_plus_m_eq_27_l896_89664

theorem k_plus_m_eq_27 (k m : ℝ) (a b c : ℝ) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a > 0) (h5 : b > 0) (h6 : c > 0)
  (h7 : a + b + c = 8) 
  (h8 : k = a * b + a * c + b * c) 
  (h9 : m = a * b * c) :
  k + m = 27 :=
by
  sorry

end NUMINAMATH_GPT_k_plus_m_eq_27_l896_89664


namespace NUMINAMATH_GPT_min_value_expression_l896_89694

noncomputable 
def min_value_condition (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) : ℝ :=
  (a + 1) * (b + 1) * (c + 1)

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) : 
  min_value_condition a b c h_pos h_abc = 8 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l896_89694


namespace NUMINAMATH_GPT_intersection_is_as_expected_l896_89620

noncomputable def quadratic_inequality_solution : Set ℝ :=
  { x | 2 * x^2 - 3 * x - 2 ≤ 0 }

noncomputable def logarithmic_condition : Set ℝ :=
  { x | x > 0 ∧ x ≠ 1 }

noncomputable def intersection_of_sets : Set ℝ :=
  (quadratic_inequality_solution ∩ logarithmic_condition)

theorem intersection_is_as_expected :
  intersection_of_sets = { x | (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) } :=
by
  sorry

end NUMINAMATH_GPT_intersection_is_as_expected_l896_89620


namespace NUMINAMATH_GPT_number_of_triangles_l896_89656

-- Define a structure representing a triangle with integer angles.
structure Triangle :=
  (A B C : ℕ) -- angles in integer degrees
  (angle_sum : A + B + C = 180)
  (obtuse_A : A > 90)

-- Define a structure representing point D on side BC of triangle ABC such that triangle ABD is right-angled
-- and triangle ADC is isosceles.
structure PointOnBC (ABC : Triangle) :=
  (D : ℕ) -- angle at D in triangle ABC
  (right_ABD : ABC.A = 90 ∨ ABC.B = 90 ∨ ABC.C = 90)
  (isosceles_ADC : ABC.A = ABC.B ∨ ABC.A = ABC.C ∨ ABC.B = ABC.C)

-- Problem Statement:
theorem number_of_triangles (t : Triangle) (d : PointOnBC t): ∃ n : ℕ, n = 88 :=
by
  sorry

end NUMINAMATH_GPT_number_of_triangles_l896_89656


namespace NUMINAMATH_GPT_lion_cub_birth_rate_l896_89611

theorem lion_cub_birth_rate :
  ∀ (x : ℕ), 100 + 12 * (x - 1) = 148 → x = 5 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_lion_cub_birth_rate_l896_89611


namespace NUMINAMATH_GPT_expectation_variance_comparison_l896_89662

variable {p1 p2 : ℝ}
variable {ξ1 ξ2 : ℝ}

theorem expectation_variance_comparison
  (h_p1 : 0 < p1)
  (h_p2 : p1 < p2)
  (h_p3 : p2 < 1 / 2)
  (h_ξ1 : ξ1 = p1)
  (h_ξ2 : ξ2 = p2):
  (ξ1 < ξ2) ∧ (ξ1 * (1 - ξ1) < ξ2 * (1 - ξ2)) := by
  sorry

end NUMINAMATH_GPT_expectation_variance_comparison_l896_89662


namespace NUMINAMATH_GPT_projectile_reaches_height_at_first_l896_89681

noncomputable def reach_height (t : ℝ) : ℝ :=
-16 * t^2 + 80 * t

theorem projectile_reaches_height_at_first (t : ℝ) :
  reach_height t = 36 → t = 0.5 :=
by
  -- The proof can be provided here
  sorry

end NUMINAMATH_GPT_projectile_reaches_height_at_first_l896_89681


namespace NUMINAMATH_GPT_value_of_N_l896_89667

theorem value_of_N : ∃ N : ℕ, (32^5 * 16^4 / 8^7) = 2^N ∧ N = 20 := by
  use 20
  sorry

end NUMINAMATH_GPT_value_of_N_l896_89667


namespace NUMINAMATH_GPT_common_measure_of_segments_l896_89637

theorem common_measure_of_segments (a b : ℚ) (h₁ : a = 4 / 15) (h₂ : b = 8 / 21) : 
  (∃ (c : ℚ), c = 1 / 105 ∧ ∃ (n₁ n₂ : ℕ), a = n₁ * c ∧ b = n₂ * c) := 
by {
  sorry
}

end NUMINAMATH_GPT_common_measure_of_segments_l896_89637


namespace NUMINAMATH_GPT_sqrt_pi_decimal_expansion_l896_89666

-- Statement of the problem: Compute the first 23 digits of the decimal expansion of sqrt(pi)
theorem sqrt_pi_decimal_expansion : 
  ( ∀ n, n ≤ 22 → 
    (digits : List ℕ) = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23] →
      (d1 = 1 ∧ d2 = 7 ∧ d3 = 7 ∧ d4 = 2 ∧ d5 = 4 ∧ d6 = 5 ∧ d7 = 3 ∧ d8 = 8 ∧ d9 = 5 ∧ d10 = 0 ∧ d11 = 9 ∧ d12 = 0 ∧ d13 = 5 ∧ d14 = 5 ∧ d15 = 1 ∧ d16 = 6 ∧ d17 = 0 ∧ d18 = 2 ∧ d19 = 7 ∧ d20 = 2 ∧ d21 = 9 ∧ d22 = 8 ∧ d23 = 1)) → 
  True :=
by
  sorry
  -- Actual proof to be filled, this is just the statement showing that we expected the digits 
  -- of the decimal expansion of sqrt(pi) match the specified values up to the 23rd place.

end NUMINAMATH_GPT_sqrt_pi_decimal_expansion_l896_89666


namespace NUMINAMATH_GPT_no_real_quadruples_solutions_l896_89600

theorem no_real_quadruples_solutions :
  ¬ ∃ (a b c d : ℝ),
    a^3 + c^3 = 2 ∧
    a^2 * b + c^2 * d = 0 ∧
    b^3 + d^3 = 1 ∧
    a * b^2 + c * d^2 = -6 := 
sorry

end NUMINAMATH_GPT_no_real_quadruples_solutions_l896_89600


namespace NUMINAMATH_GPT_simplify_and_rationalize_l896_89685

theorem simplify_and_rationalize :
  let x := (Real.sqrt 5 / Real.sqrt 7) * (Real.sqrt 9 / Real.sqrt 11) * (Real.sqrt 13 / Real.sqrt 17)
  x = 3 * Real.sqrt 84885 / 1309 := sorry

end NUMINAMATH_GPT_simplify_and_rationalize_l896_89685


namespace NUMINAMATH_GPT_contribution_required_l896_89621

-- Definitions corresponding to the problem statement
def total_amount : ℝ := 2000
def number_of_friends : ℝ := 7
def your_contribution_factor : ℝ := 2

-- Prove that the amount each friend needs to raise is approximately 222.22
theorem contribution_required (x : ℝ) 
  (h : 9 * x = total_amount) :
  x = 2000 / 9 := 
  by sorry

end NUMINAMATH_GPT_contribution_required_l896_89621


namespace NUMINAMATH_GPT_total_bees_in_hive_at_end_of_7_days_l896_89632

-- Definitions of given conditions
def daily_hatch : Nat := 3000
def daily_loss : Nat := 900
def initial_bees : Nat := 12500
def days : Nat := 7
def queen_count : Nat := 1

-- Statement to prove
theorem total_bees_in_hive_at_end_of_7_days :
  initial_bees + daily_hatch * days - daily_loss * days + queen_count = 27201 := by
  sorry

end NUMINAMATH_GPT_total_bees_in_hive_at_end_of_7_days_l896_89632


namespace NUMINAMATH_GPT_intersect_A_B_when_a_1_subset_A_B_range_a_l896_89686

def poly_eqn (x : ℝ) : Prop := -x ^ 2 - 2 * x + 8 = 0

def sol_set_A : Set ℝ := {x | poly_eqn x}

def inequality (a x : ℝ) : Prop := a * x - 1 ≤ 0

def sol_set_B (a : ℝ) : Set ℝ := {x | inequality a x}

theorem intersect_A_B_when_a_1 :
  sol_set_A ∩ sol_set_B 1 = { -4 } :=
sorry

theorem subset_A_B_range_a (a : ℝ) :
  sol_set_A ⊆ sol_set_B a ↔ (-1 / 4 : ℝ) ≤ a ∧ a ≤ 1 / 2 :=
sorry
 
end NUMINAMATH_GPT_intersect_A_B_when_a_1_subset_A_B_range_a_l896_89686


namespace NUMINAMATH_GPT_find_x_l896_89622

theorem find_x (x : ℝ) (y : ℝ) : 
  (10 * x * y - 15 * y + 3 * x - (9 / 2) = 0) ↔ x = (3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l896_89622


namespace NUMINAMATH_GPT_george_slices_l896_89629

def num_small_pizzas := 3
def num_large_pizzas := 2
def slices_per_small_pizza := 4
def slices_per_large_pizza := 8
def slices_leftover := 10
def slices_per_person := 3
def total_pizza_slices := (num_small_pizzas * slices_per_small_pizza) + (num_large_pizzas * slices_per_large_pizza)
def slices_eaten := total_pizza_slices - slices_leftover
def G := 6 -- Slices George would like to eat

theorem george_slices :
  G + (G + 1) + ((G + 1) / 2) + (3 * slices_per_person) = slices_eaten :=
by
  sorry

end NUMINAMATH_GPT_george_slices_l896_89629


namespace NUMINAMATH_GPT_triangle_inequality_l896_89699

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l896_89699


namespace NUMINAMATH_GPT_line_y_intercept_l896_89642

theorem line_y_intercept (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, 9)) (h2 : (x2, y2) = (5, 21)) :
    ∃ b : ℝ, (∀ x : ℝ, y = 4 * x + b) ∧ (b = 1) :=
by
  use 1
  sorry

end NUMINAMATH_GPT_line_y_intercept_l896_89642


namespace NUMINAMATH_GPT_sum_of_fractions_l896_89603

-- Definitions of parameters and conditions
variables {x y : ℝ}
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

-- The statement of the proof problem
theorem sum_of_fractions (hx : x ≠ 0) (hy : y ≠ 0) : 
  (3 / x) + (2 / y) = (3 * y + 2 * x) / (x * y) :=
sorry

end NUMINAMATH_GPT_sum_of_fractions_l896_89603


namespace NUMINAMATH_GPT_midpoint_C_l896_89613

variables (A B C : ℝ × ℝ)
variables (x1 y1 x2 y2 : ℝ)
variables (AC CB : ℝ)

def segment_division (A B C : ℝ × ℝ) (m n : ℝ) : Prop :=
  C = ((m * B.1 + n * A.1) / (m + n), (m * B.2 + n * A.2) / (m + n))

theorem midpoint_C :
  A = (-2, 1) →
  B = (4, 9) →
  AC = 2 * CB →
  segment_division A B C 2 1 →
  C = (2, 19 / 3) :=
by
  sorry

end NUMINAMATH_GPT_midpoint_C_l896_89613


namespace NUMINAMATH_GPT_A_is_7056_l896_89608

-- Define the variables and conditions
def D : ℕ := 4 * 3
def E : ℕ := 7 * 3
def B : ℕ := 4 * D
def C : ℕ := 7 * E
def A : ℕ := B * C

-- Prove that A = 7056 given the conditions
theorem A_is_7056 : A = 7056 := by
  -- We will skip the proof steps with 'sorry'
  sorry

end NUMINAMATH_GPT_A_is_7056_l896_89608


namespace NUMINAMATH_GPT_log10_sum_diff_l896_89650

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log10_sum_diff :
  log10 32 + log10 50 - log10 8 = 2.301 :=
by
  sorry

end NUMINAMATH_GPT_log10_sum_diff_l896_89650


namespace NUMINAMATH_GPT_interval_of_y_l896_89617

theorem interval_of_y (y : ℝ) (h : y = (1 / y) * (-y) - 5) : -6 ≤ y ∧ y ≤ -4 :=
by sorry

end NUMINAMATH_GPT_interval_of_y_l896_89617


namespace NUMINAMATH_GPT_equivalent_fraction_l896_89649

theorem equivalent_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h : (4 * x + 2 * y) / (x - 4 * y) = -3) :
  (2 * x + 8 * y) / (4 * x - 2 * y) = 38 / 13 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_fraction_l896_89649


namespace NUMINAMATH_GPT_age_equivalence_l896_89676

variable (x : ℕ)

theorem age_equivalence : ∃ x : ℕ, 60 + x = 35 + x + 11 + x ∧ x = 14 :=
by
  sorry

end NUMINAMATH_GPT_age_equivalence_l896_89676


namespace NUMINAMATH_GPT_num_white_black_balls_prob_2_black_balls_dist_exp_black_balls_l896_89615

-- Problem 1: Number of white and black balls
theorem num_white_black_balls (n m : ℕ) (h1 : n + m = 10)
  (h2 : (10 - m) = 4) : n = 4 ∧ m = 6 :=
by sorry

-- Problem 2: Probability of drawing exactly 2 black balls with replacement
theorem prob_2_black_balls (p_black_draw : ℕ → ℕ → ℚ)
  (h1 : ∀ n m, p_black_draw n m = (6/10)^(n-m) * (4/10)^m)
  (h2 : p_black_draw 2 3 = 54/125) : p_black_draw 2 3 = 54 / 125 :=
by sorry

-- Problem 3: Distribution and Expectation of number of black balls drawn without replacement
theorem dist_exp_black_balls (prob_X : ℕ → ℚ) (expect_X : ℚ)
  (h1 : prob_X 0 = 2/15) (h2 : prob_X 1 = 8/15) (h3 : prob_X 2 = 1/3)
  (h4 : expect_X = 6 / 5) : ∀ k, prob_X k = match k with
    | 0 => 2/15
    | 1 => 8/15
    | 2 => 1/3
    | _ => 0 :=
by sorry

end NUMINAMATH_GPT_num_white_black_balls_prob_2_black_balls_dist_exp_black_balls_l896_89615


namespace NUMINAMATH_GPT_solve_inequality_l896_89674

-- Define the inequality problem.
noncomputable def inequality_problem (x : ℝ) : Prop :=
(x^2 + 2 * x - 15) / (x + 5) < 0

-- Define the solution set.
def solution_set (x : ℝ) : Prop :=
-5 < x ∧ x < 3

-- State the equivalence theorem.
theorem solve_inequality (x : ℝ) (h : x ≠ -5) : 
  inequality_problem x ↔ solution_set x :=
sorry

end NUMINAMATH_GPT_solve_inequality_l896_89674


namespace NUMINAMATH_GPT_largest_integer_of_four_l896_89677

theorem largest_integer_of_four (a b c d : ℤ) 
  (h1 : a + b + c = 160) 
  (h2 : a + b + d = 185) 
  (h3 : a + c + d = 205) 
  (h4 : b + c + d = 230) : 
  max (max a (max b c)) d = 100 := 
by
  sorry

end NUMINAMATH_GPT_largest_integer_of_four_l896_89677


namespace NUMINAMATH_GPT_fifteenth_entry_is_21_l896_89604

def r_9 (n : ℕ) : ℕ := n % 9

def condition (n : ℕ) : Prop := (7 * n) % 9 ≤ 5

def sequence_elements (k : ℕ) : ℕ := 
  if k = 0 then 0
  else if k = 1 then 2
  else if k = 2 then 3
  else if k = 3 then 4
  else if k = 4 then 7
  else if k = 5 then 8
  else if k = 6 then 9
  else if k = 7 then 11
  else if k = 8 then 12
  else if k = 9 then 13
  else if k = 10 then 16
  else if k = 11 then 17
  else if k = 12 then 18
  else if k = 13 then 20
  else if k = 14 then 21
  else 0 -- for the sake of ensuring completeness

theorem fifteenth_entry_is_21 : sequence_elements 14 = 21 :=
by
  -- Mathematical proof omitted.
  sorry

end NUMINAMATH_GPT_fifteenth_entry_is_21_l896_89604


namespace NUMINAMATH_GPT_simplify_expression_l896_89627

theorem simplify_expression : ((3 * 2 + 4 + 6) / 3 - 2 / 3) = 14 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l896_89627
