import Mathlib

namespace NUMINAMATH_GPT_second_person_avg_pages_per_day_l186_18686

def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def average_book_pages : ℕ := 320
def closest_person_percentage : ℝ := 0.75

theorem second_person_avg_pages_per_day :
  (deshaun_books * average_book_pages * closest_person_percentage) / summer_days = 180 := by
sorry

end NUMINAMATH_GPT_second_person_avg_pages_per_day_l186_18686


namespace NUMINAMATH_GPT_maximal_segment_number_l186_18619

theorem maximal_segment_number (n : ℕ) (h : n > 4) : 
  ∃ k, k = if n % 2 = 0 then 2 * n - 4 else 2 * n - 3 :=
sorry

end NUMINAMATH_GPT_maximal_segment_number_l186_18619


namespace NUMINAMATH_GPT_mike_picked_l186_18620

-- Define the number of pears picked by Jason, Keith, and the total number of pears picked.
def jason_picked : ℕ := 46
def keith_picked : ℕ := 47
def total_picked : ℕ := 105

-- Define the goal that we need to prove: the number of pears Mike picked.
theorem mike_picked (jason_picked keith_picked total_picked : ℕ) 
  (h1 : jason_picked = 46) 
  (h2 : keith_picked = 47) 
  (h3 : total_picked = 105) 
  : (total_picked - (jason_picked + keith_picked)) = 12 :=
by sorry

end NUMINAMATH_GPT_mike_picked_l186_18620


namespace NUMINAMATH_GPT_initial_percentage_increase_l186_18662

-- Given conditions
def S_original : ℝ := 4000.0000000000005
def S_final : ℝ := 4180
def reduction : ℝ := 5

-- Predicate to prove the initial percentage increase is 10%
theorem initial_percentage_increase (x : ℝ) 
  (hx : (95/100) * (S_original * (1 + x / 100)) = S_final) : 
  x = 10 :=
sorry

end NUMINAMATH_GPT_initial_percentage_increase_l186_18662


namespace NUMINAMATH_GPT_min_mn_value_l186_18644

theorem min_mn_value
  (a : ℝ) (m : ℝ) (n : ℝ)
  (ha_pos : a > 0) (ha_ne_one : a ≠ 1) (hm_pos : m > 0) (hn_pos : n > 0)
  (H : (1 : ℝ) / m + (1 : ℝ) / n = 4) :
  m + n ≥ 1 :=
sorry

end NUMINAMATH_GPT_min_mn_value_l186_18644


namespace NUMINAMATH_GPT_correct_proposition_l186_18649

def curve_is_ellipse (k : ℝ) : Prop :=
  9 < k ∧ k < 25

def curve_is_hyperbola_on_x_axis (k : ℝ) : Prop :=
  k < 9

theorem correct_proposition (k : ℝ) :
  (curve_is_ellipse k ∨ ¬ curve_is_ellipse k) ∧ 
  (curve_is_hyperbola_on_x_axis k ∨ ¬ curve_is_hyperbola_on_x_axis k) →
  (9 < k ∧ k < 25 → curve_is_ellipse k) ∧ 
  (curve_is_ellipse k ↔ (9 < k ∧ k < 25)) ∧ 
  (curve_is_hyperbola_on_x_axis k ↔ k < 9) → 
  (curve_is_ellipse k ∧ curve_is_hyperbola_on_x_axis k) :=
by
  sorry

end NUMINAMATH_GPT_correct_proposition_l186_18649


namespace NUMINAMATH_GPT_probability_at_least_one_black_eq_seven_tenth_l186_18635

noncomputable def probability_drawing_at_least_one_black_ball : ℚ :=
  let total_ways := Nat.choose 5 2
  let ways_no_black := Nat.choose 3 2
  1 - (ways_no_black / total_ways)

theorem probability_at_least_one_black_eq_seven_tenth :
  probability_drawing_at_least_one_black_ball = 7 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_black_eq_seven_tenth_l186_18635


namespace NUMINAMATH_GPT_time_to_cook_one_potato_l186_18609

-- Definitions for the conditions
def total_potatoes : ℕ := 16
def cooked_potatoes : ℕ := 7
def remaining_minutes : ℕ := 45

-- Lean theorem that asserts the equivalence of the problem statement to the correct answer
theorem time_to_cook_one_potato (total_potatoes cooked_potatoes remaining_minutes : ℕ) 
  (h_total : total_potatoes = 16) 
  (h_cooked : cooked_potatoes = 7) 
  (h_remaining : remaining_minutes = 45) :
  (remaining_minutes / (total_potatoes - cooked_potatoes) = 5) :=
by
  -- Using sorry to skip proof
  sorry

end NUMINAMATH_GPT_time_to_cook_one_potato_l186_18609


namespace NUMINAMATH_GPT_amount_coach_mike_gave_l186_18658

-- Definitions from conditions
def cost_of_lemonade : ℕ := 58
def change_received : ℕ := 17

-- Theorem stating the proof problem
theorem amount_coach_mike_gave : cost_of_lemonade + change_received = 75 := by
  sorry

end NUMINAMATH_GPT_amount_coach_mike_gave_l186_18658


namespace NUMINAMATH_GPT_sum_of_angles_is_360_l186_18696

-- Let's define the specific angles within our geometric figure
variables (A B C D F G : ℝ)

-- Define a condition stating that these angles form a quadrilateral inside a geometric figure, such that their sum is valid
def angles_form_quadrilateral (A B C D F G : ℝ) : Prop :=
  (A + B + C + D + F + G = 360)

-- Finally, we declare the theorem we want to prove
theorem sum_of_angles_is_360 (A B C D F G : ℝ) (h : angles_form_quadrilateral A B C D F G) : A + B + C + D + F + G = 360 :=
  h


end NUMINAMATH_GPT_sum_of_angles_is_360_l186_18696


namespace NUMINAMATH_GPT_range_of_m_l186_18697

theorem range_of_m (m : ℝ) :
  (∃ x y, y = x^2 + m * x + 2 ∧ x - y + 1 = 0 ∧ 0 ≤ x ∧ x ≤ 2) → m ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l186_18697


namespace NUMINAMATH_GPT_tn_range_l186_18611

noncomputable def a (n : ℕ) : ℚ :=
  (2 * n - 1) / 10

noncomputable def b (n : ℕ) : ℚ :=
  2^(n - 1)

noncomputable def c (n : ℕ) : ℚ :=
  (1 + a n) / (4 * b n)

noncomputable def T (n : ℕ) : ℚ :=
  (1 / 10) * (2 - (n + 2) / (2^n)) + (9 / 20) * (2 - 1 / (2^(n-1)))

theorem tn_range (n : ℕ) : (101 / 400 : ℚ) ≤ T n ∧ T n < (103 / 200 : ℚ) :=
sorry

end NUMINAMATH_GPT_tn_range_l186_18611


namespace NUMINAMATH_GPT_f_zero_f_positive_all_f_increasing_f_range_l186_18617

universe u

noncomputable def f : ℝ → ℝ := sorry

axiom f_nonzero : f 0 ≠ 0
axiom f_positive : ∀ x : ℝ, 0 < x → f x > 1
axiom f_add_prop : ∀ a b : ℝ, f (a + b) = f a * f b

-- Problem 1: Prove that f(0) = 1
theorem f_zero : f 0 = 1 := sorry

-- Problem 2: Prove that for any x in ℝ, f(x) > 0
theorem f_positive_all (x : ℝ) : f x > 0 := sorry

-- Problem 3: Prove that f(x) is an increasing function on ℝ
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := sorry

-- Problem 4: Given f(x) * f(2x - x²) > 1, find the range of x
theorem f_range (x : ℝ) (h : f x * f (2*x - x^2) > 1) : 0 < x ∧ x < 3 := sorry

end NUMINAMATH_GPT_f_zero_f_positive_all_f_increasing_f_range_l186_18617


namespace NUMINAMATH_GPT_find_natural_number_n_l186_18628

theorem find_natural_number_n : 
  ∃ (n : ℕ), (∃ k : ℕ, n + 15 = k^2) ∧ (∃ m : ℕ, n - 14 = m^2) ∧ n = 210 :=
by
  sorry

end NUMINAMATH_GPT_find_natural_number_n_l186_18628


namespace NUMINAMATH_GPT_cassidy_grades_below_B_l186_18608

theorem cassidy_grades_below_B (x : ℕ) (h1 : 26 = 14 + 3 * x) : x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_cassidy_grades_below_B_l186_18608


namespace NUMINAMATH_GPT_brad_reads_more_pages_l186_18654

-- Definitions based on conditions
def greg_pages_per_day : ℕ := 18
def brad_pages_per_day : ℕ := 26

-- Statement to prove
theorem brad_reads_more_pages : brad_pages_per_day - greg_pages_per_day = 8 :=
by
  -- sorry is used here to indicate the absence of a proof
  sorry

end NUMINAMATH_GPT_brad_reads_more_pages_l186_18654


namespace NUMINAMATH_GPT_find_value_of_S_l186_18613

theorem find_value_of_S (S : ℝ)
  (h1 : (1 / 3) * (1 / 8) * S = (1 / 4) * (1 / 6) * 180) :
  S = 180 :=
sorry

end NUMINAMATH_GPT_find_value_of_S_l186_18613


namespace NUMINAMATH_GPT_sum_of_reciprocal_roots_l186_18663

theorem sum_of_reciprocal_roots (r s α β : ℝ) (h1 : 7 * r^2 - 8 * r + 6 = 0) (h2 : 7 * s^2 - 8 * s + 6 = 0) (h3 : α = 1 / r) (h4 : β = 1 / s) :
  α + β = 4 / 3 := 
sorry

end NUMINAMATH_GPT_sum_of_reciprocal_roots_l186_18663


namespace NUMINAMATH_GPT_solution_set_l186_18647

def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 1|

theorem solution_set : { x : ℝ | f x > 1 } = Set.Ioo (2/3) 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l186_18647


namespace NUMINAMATH_GPT_exist_A_B_l186_18615

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exist_A_B : ∃ (A B : ℕ), A = 2016 * B ∧ sum_of_digits A + 2016 * sum_of_digits B < 0 := sorry

end NUMINAMATH_GPT_exist_A_B_l186_18615


namespace NUMINAMATH_GPT_x_intercept_is_3_l186_18638

-- Define the given points
def point1 : ℝ × ℝ := (2, -2)
def point2 : ℝ × ℝ := (6, 6)

-- Prove the x-intercept is 3
theorem x_intercept_is_3 (x : ℝ) :
  (∃ m b : ℝ, (∀ x1 y1 x2 y2 : ℝ, (y1 = m * x1 + b) ∧ (x1, y1) = point1 ∧ (x2, y2) = point2) ∧ y = 0 ∧ x = -b / m) → x = 3 :=
sorry

end NUMINAMATH_GPT_x_intercept_is_3_l186_18638


namespace NUMINAMATH_GPT_smallest_constant_inequality_l186_18665

theorem smallest_constant_inequality :
  ∀ (x y : ℝ), 1 + (x + y)^2 ≤ (4 / 3) * (1 + x^2) * (1 + y^2) :=
by
  intro x y
  sorry

end NUMINAMATH_GPT_smallest_constant_inequality_l186_18665


namespace NUMINAMATH_GPT_condition_on_a_b_l186_18640

theorem condition_on_a_b (a b : ℝ) (h : a^2 * b^2 + 5 > 2 * a * b - a^2 - 4 * a) : ab ≠ 1 ∨ a ≠ -2 :=
by
  sorry

end NUMINAMATH_GPT_condition_on_a_b_l186_18640


namespace NUMINAMATH_GPT_find_b_l186_18601

noncomputable def a : ℂ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℂ := sorry

-- Given conditions
axiom sum_eq : a + b + c = 4
axiom prod_pairs_eq : a * b + b * c + c * a = 5
axiom prod_triple_eq : a * b * c = 6

-- Prove that b = 1
theorem find_b : b = 1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_b_l186_18601


namespace NUMINAMATH_GPT_residue_7_pow_1234_l186_18630

theorem residue_7_pow_1234 : (7^1234) % 13 = 4 := by
  sorry

end NUMINAMATH_GPT_residue_7_pow_1234_l186_18630


namespace NUMINAMATH_GPT_no_solution_system_l186_18618

theorem no_solution_system (a : ℝ) :
  (∀ (x : ℝ), (a ≠ 0 → (a^2 * x + 2 * a) / (a * x - 2 + a^2) < 0 ∨ ax + a ≤ 5/4)) ∧ 
  (a = 0 → ¬ ∃ (x : ℝ), (a^2 * x + 2 * a) / (a * x - 2 + a^2) ≥ 0 ∧ ax + a > 5/4) ↔ 
  a ∈ Set.Iic (-1/2) ∪ {0} :=
by sorry

end NUMINAMATH_GPT_no_solution_system_l186_18618


namespace NUMINAMATH_GPT_problem_1_problem_2_l186_18627

noncomputable def A := Real.pi / 3
noncomputable def b := 5
noncomputable def c := 4 -- derived from the solution
noncomputable def S : ℝ := 5 * Real.sqrt 3

theorem problem_1 (A : ℝ) 
  (h : Real.cos (2 * A) - 3 * Real.cos (Real.pi - A) = 1) 
  : A = Real.pi / 3 :=
sorry

theorem problem_2 (a : ℝ) 
  (b : ℝ) 
  (S : ℝ) 
  (h_b : b = 5) 
  (h_S : S = 5 * Real.sqrt 3) 
  : a = Real.sqrt 21 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l186_18627


namespace NUMINAMATH_GPT_zero_in_M_l186_18661

-- Define the set M
def M : Set ℕ := {0, 1, 2}

-- State the theorem to be proved
theorem zero_in_M : 0 ∈ M := 
  sorry

end NUMINAMATH_GPT_zero_in_M_l186_18661


namespace NUMINAMATH_GPT_num_solutions_gcd_lcm_l186_18607

noncomputable def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

theorem num_solutions_gcd_lcm (x y : ℕ) :
  (Nat.gcd x y = factorial 20) ∧ (Nat.lcm x y = factorial 30) →
  2^10 = 1024 :=
  by
  intro h
  sorry

end NUMINAMATH_GPT_num_solutions_gcd_lcm_l186_18607


namespace NUMINAMATH_GPT_parabola_translation_l186_18614

theorem parabola_translation :
  (∀ x : ℝ, y = x^2 → y' = (x - 1)^2 + 3) :=
sorry

end NUMINAMATH_GPT_parabola_translation_l186_18614


namespace NUMINAMATH_GPT_regular_polygon_perimeter_l186_18631

theorem regular_polygon_perimeter (s : ℕ) (E : ℕ) (n : ℕ) (P : ℕ)
  (h1 : s = 6)
  (h2 : E = 90)
  (h3 : E = 360 / n)
  (h4 : P = n * s) :
  P = 24 :=
by sorry

end NUMINAMATH_GPT_regular_polygon_perimeter_l186_18631


namespace NUMINAMATH_GPT_complex_division_l186_18682

theorem complex_division (i : ℂ) (hi : i = Complex.I) : (1 + i) / (1 - i) = i :=
by
  sorry

end NUMINAMATH_GPT_complex_division_l186_18682


namespace NUMINAMATH_GPT_range_of_expression_l186_18622

theorem range_of_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 ≤ β ∧ β ≤ π / 2) :
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π :=
sorry

end NUMINAMATH_GPT_range_of_expression_l186_18622


namespace NUMINAMATH_GPT_loss_due_to_simple_interest_l186_18689

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r)^t

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem loss_due_to_simple_interest (P : ℝ) (r : ℝ) (t : ℝ)
  (hP : P = 2500) (hr : r = 0.04) (ht : t = 2) :
  let CI := compound_interest P r t
  let SI := simple_interest P r t
  ∃ loss : ℝ, loss = CI - SI ∧ loss = 4 :=
by
  sorry

end NUMINAMATH_GPT_loss_due_to_simple_interest_l186_18689


namespace NUMINAMATH_GPT_arithmetic_progression_product_difference_le_one_l186_18632

theorem arithmetic_progression_product_difference_le_one 
  (a b : ℝ) :
  ∃ (m n k l : ℤ), |(a + b * m) * (a + b * n) - (a + b * k) * (a + b * l)| ≤ 1 :=
sorry

end NUMINAMATH_GPT_arithmetic_progression_product_difference_le_one_l186_18632


namespace NUMINAMATH_GPT_prime_p_sum_of_squares_l186_18636

theorem prime_p_sum_of_squares (p : ℕ) (hp : p.Prime) 
  (h : ∃ (a : ℕ), 2 * p = a^2 + (a + 1)^2 + (a + 2)^2 + (a + 3)^2) : 
  36 ∣ (p - 7) :=
by 
  sorry

end NUMINAMATH_GPT_prime_p_sum_of_squares_l186_18636


namespace NUMINAMATH_GPT_contrapositive_of_implication_l186_18667

theorem contrapositive_of_implication (a : ℝ) (h : a > 0 → a > 1) : a ≤ 1 → a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_of_implication_l186_18667


namespace NUMINAMATH_GPT_arithmetic_mean_is_correct_l186_18666

open Nat

noncomputable def arithmetic_mean_of_two_digit_multiples_of_9 : ℝ :=
  let smallest := 18
  let largest := 99
  let n := 10
  let sum := (n / 2 : ℝ) * (smallest + largest)
  sum / n

theorem arithmetic_mean_is_correct :
  arithmetic_mean_of_two_digit_multiples_of_9 = 58.5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_arithmetic_mean_is_correct_l186_18666


namespace NUMINAMATH_GPT_pizza_diameter_increase_l186_18690

theorem pizza_diameter_increase :
  ∀ (d D : ℝ), 
    (D / d)^2 = 1.96 → D = 1.4 * d := by
  sorry

end NUMINAMATH_GPT_pizza_diameter_increase_l186_18690


namespace NUMINAMATH_GPT_remainder_of_3y_l186_18602

theorem remainder_of_3y (y : ℕ) (hy : y % 9 = 5) : (3 * y) % 9 = 6 :=
sorry

end NUMINAMATH_GPT_remainder_of_3y_l186_18602


namespace NUMINAMATH_GPT_union_of_sets_l186_18687

variable (x : ℝ)

def A : Set ℝ := {x | 0 < x ∧ x < 3}
def B : Set ℝ := {x | -1 < x ∧ x < 2}
def target : Set ℝ := {x | -1 < x ∧ x < 3}

theorem union_of_sets : (A ∪ B) = target :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l186_18687


namespace NUMINAMATH_GPT_valid_passwords_l186_18698

theorem valid_passwords (total_passwords restricted_passwords : Nat) 
  (h_total : total_passwords = 10^4)
  (h_restricted : restricted_passwords = 8) : 
  total_passwords - restricted_passwords = 9992 := by
  sorry

end NUMINAMATH_GPT_valid_passwords_l186_18698


namespace NUMINAMATH_GPT_least_subtraction_l186_18650

theorem least_subtraction (n : ℕ) (d : ℕ) (r : ℕ) (h1 : n = 45678) (h2 : d = 47) (h3 : n % d = r) : r = 35 :=
by {
  sorry
}

end NUMINAMATH_GPT_least_subtraction_l186_18650


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_l186_18610

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 17) (h2 : x * y = 6) : x^2 + y^2 = 301 :=
by sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_l186_18610


namespace NUMINAMATH_GPT_quotient_unchanged_l186_18674

-- Define the variables
variables (a b k : ℝ)

-- Condition: k ≠ 0
theorem quotient_unchanged (h : k ≠ 0) : (a * k) / (b * k) = a / b := by
  sorry

end NUMINAMATH_GPT_quotient_unchanged_l186_18674


namespace NUMINAMATH_GPT_pictures_per_album_l186_18699

-- Define the problem conditions
def picturesFromPhone : Nat := 35
def picturesFromCamera : Nat := 5
def totalAlbums : Nat := 5

-- Define the total number of pictures
def totalPictures : Nat := picturesFromPhone + picturesFromCamera

-- Define what we need to prove
theorem pictures_per_album :
  totalPictures / totalAlbums = 8 := by
  sorry

end NUMINAMATH_GPT_pictures_per_album_l186_18699


namespace NUMINAMATH_GPT_y_coordinate_of_A_l186_18621

theorem y_coordinate_of_A (a : ℝ) (y : ℝ) (h1 : y = a * 1) (h2 : y = (4 - a) / 1) : y = 2 :=
by
  sorry

end NUMINAMATH_GPT_y_coordinate_of_A_l186_18621


namespace NUMINAMATH_GPT_factor_difference_of_squares_l186_18646

theorem factor_difference_of_squares (y : ℝ) : 81 - 16 * y^2 = (9 - 4 * y) * (9 + 4 * y) :=
by
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l186_18646


namespace NUMINAMATH_GPT_find_constant_A_l186_18623

theorem find_constant_A :
  ∀ (x : ℝ)
  (A B C D : ℝ),
      (
        (1 : ℝ) / (x^4 - 20 * x^3 + 147 * x^2 - 490 * x + 588) = 
        (A / (x + 3)) + (B / (x - 4)) + (C / ((x - 4)^2)) + (D / (x - 7))
      ) →
      A = - (1 / 490) := 
by 
  intro x A B C D h
  sorry

end NUMINAMATH_GPT_find_constant_A_l186_18623


namespace NUMINAMATH_GPT_odd_product_probability_lt_one_eighth_l186_18679

theorem odd_product_probability_lt_one_eighth : 
  (∃ p : ℝ, p = (500 / 1000) * (499 / 999) * (498 / 998)) → p < 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_odd_product_probability_lt_one_eighth_l186_18679


namespace NUMINAMATH_GPT_value_two_std_dev_less_than_mean_l186_18657

-- Define the given conditions for the problem.
def mean : ℝ := 15
def std_dev : ℝ := 1.5

-- Define the target value that should be 2 standard deviations less than the mean.
def target_value := mean - 2 * std_dev

-- State the theorem that represents the proof problem.
theorem value_two_std_dev_less_than_mean : target_value = 12 := by
  sorry

end NUMINAMATH_GPT_value_two_std_dev_less_than_mean_l186_18657


namespace NUMINAMATH_GPT_tingting_solution_correct_l186_18655

noncomputable def product_of_square_roots : ℝ :=
  (Real.sqrt 8) * (Real.sqrt 18)

theorem tingting_solution_correct : product_of_square_roots = 12 := by
  sorry

end NUMINAMATH_GPT_tingting_solution_correct_l186_18655


namespace NUMINAMATH_GPT_xander_pages_left_to_read_l186_18681

theorem xander_pages_left_to_read :
  let total_pages := 500
  let read_first_night := 0.2 * 500
  let read_second_night := 0.2 * 500
  let read_third_night := 0.3 * 500
  total_pages - (read_first_night + read_second_night + read_third_night) = 150 :=
by 
  sorry

end NUMINAMATH_GPT_xander_pages_left_to_read_l186_18681


namespace NUMINAMATH_GPT_abs_neg_three_l186_18692

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_three_l186_18692


namespace NUMINAMATH_GPT_no_rain_four_days_l186_18670

-- Define the probability of rain on any given day
def prob_rain : ℚ := 2/3

-- Define the probability that it does not rain on any given day
def prob_no_rain : ℚ := 1 - prob_rain

-- Define the probability that it does not rain at all over four days
def prob_no_rain_four_days : ℚ := prob_no_rain^4

theorem no_rain_four_days : prob_no_rain_four_days = 1/81 := by
  sorry

end NUMINAMATH_GPT_no_rain_four_days_l186_18670


namespace NUMINAMATH_GPT_find_number_of_students_l186_18642

theorem find_number_of_students
    (S N : ℕ) 
    (h₁ : 4 * S + 3 = N)
    (h₂ : 5 * S = N + 6) : 
  S = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_students_l186_18642


namespace NUMINAMATH_GPT_cooper_needs_1043_bricks_l186_18643

def wall1_length := 15
def wall1_height := 6
def wall1_depth := 3

def wall2_length := 20
def wall2_height := 4
def wall2_depth := 2

def wall3_length := 25
def wall3_height := 5
def wall3_depth := 3

def wall4_length := 17
def wall4_height := 7
def wall4_depth := 2

def bricks_needed_for_wall (length height depth: Nat) : Nat :=
  length * height * depth

def total_bricks_needed : Nat :=
  bricks_needed_for_wall wall1_length wall1_height wall1_depth +
  bricks_needed_for_wall wall2_length wall2_height wall2_depth +
  bricks_needed_for_wall wall3_length wall3_height wall3_depth +
  bricks_needed_for_wall wall4_length wall4_height wall4_depth

theorem cooper_needs_1043_bricks : total_bricks_needed = 1043 := by
  sorry

end NUMINAMATH_GPT_cooper_needs_1043_bricks_l186_18643


namespace NUMINAMATH_GPT_mean_of_remaining_four_numbers_l186_18616

theorem mean_of_remaining_four_numbers (a b c d : ℝ) (h1 : (a + b + c + d + 106) / 5 = 92) : 
  (a + b + c + d) / 4 = 88.5 := 
sorry

end NUMINAMATH_GPT_mean_of_remaining_four_numbers_l186_18616


namespace NUMINAMATH_GPT_Tom_spends_375_dollars_l186_18659

noncomputable def totalCost (numBricks : ℕ) (halfDiscount : ℚ) (fullPrice : ℚ) : ℚ :=
  let halfBricks := numBricks / 2
  let discountedPrice := fullPrice * halfDiscount
  (halfBricks * discountedPrice) + (halfBricks * fullPrice)

theorem Tom_spends_375_dollars : 
  ∀ (numBricks : ℕ) (halfDiscount fullPrice : ℚ), 
  numBricks = 1000 → halfDiscount = 0.5 → fullPrice = 0.5 → totalCost numBricks halfDiscount fullPrice = 375 := 
by
  intros numBricks halfDiscount fullPrice hnumBricks hhalfDiscount hfullPrice
  rw [hnumBricks, hhalfDiscount, hfullPrice]
  sorry

end NUMINAMATH_GPT_Tom_spends_375_dollars_l186_18659


namespace NUMINAMATH_GPT_compute_fraction_power_l186_18693

theorem compute_fraction_power (a b : ℕ) (ha : a = 123456) (hb : b = 41152) : (a ^ 5 / b ^ 5) = 243 := by
  sorry

end NUMINAMATH_GPT_compute_fraction_power_l186_18693


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_l186_18637

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem neither_sufficient_nor_necessary (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  ¬ ((q > 1) ↔ is_increasing_sequence a) :=
sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_l186_18637


namespace NUMINAMATH_GPT_red_red_pairs_l186_18680

theorem red_red_pairs (green_shirts red_shirts total_students total_pairs green_green_pairs : ℕ)
    (hg1 : green_shirts = 64)
    (hr1 : red_shirts = 68)
    (htotal : total_students = 132)
    (htotal_pairs : total_pairs = 66)
    (hgreen_green_pairs : green_green_pairs = 28) :
    (total_students = green_shirts + red_shirts) ∧
    (green_green_pairs ≤ total_pairs) ∧
    (∃ red_red_pairs, red_red_pairs = 30) :=
by
  sorry

end NUMINAMATH_GPT_red_red_pairs_l186_18680


namespace NUMINAMATH_GPT_arithmetic_sequence_minimum_value_S_l186_18695

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_minimum_value_S_l186_18695


namespace NUMINAMATH_GPT_pole_intersection_height_l186_18625

theorem pole_intersection_height 
  (h1 h2 d : ℝ) 
  (h1pos : h1 = 30) 
  (h2pos : h2 = 90) 
  (dpos : d = 150) : 
  ∃ y, y = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_pole_intersection_height_l186_18625


namespace NUMINAMATH_GPT_range_3x_plus_2y_l186_18683

theorem range_3x_plus_2y (x y : ℝ) : -1 < x + y ∧ x + y < 4 → 2 < x - y ∧ x - y < 3 → 
  -3/2 < 3*x + 2*y ∧ 3*x + 2*y < 23/2 :=
by
  sorry

end NUMINAMATH_GPT_range_3x_plus_2y_l186_18683


namespace NUMINAMATH_GPT_prove_correct_y_l186_18634

noncomputable def find_larger_y (x y : ℕ) : Prop :=
  y - x = 1365 ∧ y = 6 * x + 15

noncomputable def correct_y : ℕ := 1635

theorem prove_correct_y (x y : ℕ) (h : find_larger_y x y) : y = correct_y :=
by
  sorry

end NUMINAMATH_GPT_prove_correct_y_l186_18634


namespace NUMINAMATH_GPT_total_cakes_served_l186_18694

def Cakes_Monday_Lunch : ℕ := 5
def Cakes_Monday_Dinner : ℕ := 6
def Cakes_Sunday : ℕ := 3
def cakes_served_twice (n : ℕ) : ℕ := 2 * n
def cakes_thrown_away : ℕ := 4

theorem total_cakes_served : 
  Cakes_Sunday + Cakes_Monday_Lunch + Cakes_Monday_Dinner + 
  (cakes_served_twice (Cakes_Monday_Lunch + Cakes_Monday_Dinner) - cakes_thrown_away) = 32 := 
by 
  sorry

end NUMINAMATH_GPT_total_cakes_served_l186_18694


namespace NUMINAMATH_GPT_soda_cost_is_2_l186_18671

noncomputable def cost_per_soda (total_bill : ℕ) (num_adults : ℕ) (num_children : ℕ) 
  (adult_meal_cost : ℕ) (child_meal_cost : ℕ) (num_sodas : ℕ) : ℕ :=
  (total_bill - (num_adults * adult_meal_cost + num_children * child_meal_cost)) / num_sodas

theorem soda_cost_is_2 :
  let total_bill := 60
  let num_adults := 6
  let num_children := 2
  let adult_meal_cost := 6
  let child_meal_cost := 4
  let num_sodas := num_adults + num_children
  cost_per_soda total_bill num_adults num_children adult_meal_cost child_meal_cost num_sodas = 2 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_soda_cost_is_2_l186_18671


namespace NUMINAMATH_GPT_students_that_do_not_like_either_sport_l186_18626

def total_students : ℕ := 30
def students_like_basketball : ℕ := 15
def students_like_table_tennis : ℕ := 10
def students_like_both : ℕ := 3

theorem students_that_do_not_like_either_sport : (total_students - (students_like_basketball + students_like_table_tennis - students_like_both)) = 8 := 
by
  sorry

end NUMINAMATH_GPT_students_that_do_not_like_either_sport_l186_18626


namespace NUMINAMATH_GPT_bank_teller_rolls_of_coins_l186_18676

theorem bank_teller_rolls_of_coins (tellers : ℕ) (coins_per_roll : ℕ) (total_coins : ℕ) (h_tellers : tellers = 4) (h_coins_per_roll : coins_per_roll = 25) (h_total_coins : total_coins = 1000) : 
  (total_coins / tellers) / coins_per_roll = 10 :=
by 
  sorry

end NUMINAMATH_GPT_bank_teller_rolls_of_coins_l186_18676


namespace NUMINAMATH_GPT_common_ratio_geometric_sequence_l186_18651

theorem common_ratio_geometric_sequence 
  (a1 : ℝ) 
  (q : ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = a1 * (1 - q^n) / (1 - q)) 
  (h2 : 8 * S 6 = 7 * S 3) 
  (hq : q ≠ 1) : 
  q = -1 / 2 := 
sorry

end NUMINAMATH_GPT_common_ratio_geometric_sequence_l186_18651


namespace NUMINAMATH_GPT_pump_out_time_l186_18604

theorem pump_out_time
  (length : ℝ)
  (width : ℝ)
  (depth : ℝ)
  (rate : ℝ)
  (H_length : length = 50)
  (H_width : width = 30)
  (H_depth : depth = 1.8)
  (H_rate : rate = 2.5) : 
  (length * width * depth) / rate / 60 = 18 :=
by
  sorry

end NUMINAMATH_GPT_pump_out_time_l186_18604


namespace NUMINAMATH_GPT_tank_salt_solution_l186_18684

theorem tank_salt_solution (x : ℝ) (h1 : (0.20 * x + 14) / ((3 / 4) * x + 21) = 1 / 3) : x = 140 :=
sorry

end NUMINAMATH_GPT_tank_salt_solution_l186_18684


namespace NUMINAMATH_GPT_correct_inequality_l186_18606

variable (a b : ℝ)

theorem correct_inequality (h : a > b) : a - 3 > b - 3 :=
by
  sorry

end NUMINAMATH_GPT_correct_inequality_l186_18606


namespace NUMINAMATH_GPT_factor_expression_l186_18664

theorem factor_expression (x : ℝ) : 
  (10 * x^3 + 45 * x^2 - 5 * x) - (-5 * x^3 + 10 * x^2 - 5 * x) = 5 * x^2 * (3 * x + 7) :=
by 
  sorry

end NUMINAMATH_GPT_factor_expression_l186_18664


namespace NUMINAMATH_GPT_part1_part2_l186_18669

variable (a b c : ℝ)

-- Conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom eq_sum_square : a^2 + b^2 + 4*c^2 = 3

-- Part 1
theorem part1 : a + b + 2 * c ≤ 3 := 
by
  sorry

-- Additional condition for Part 2
variable (hc : b = 2*c)

-- Part 2
theorem part2 : (1 / a) + (1 / c) ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l186_18669


namespace NUMINAMATH_GPT_determine_y_value_l186_18668

theorem determine_y_value {k y : ℕ} (h1 : k > 0) (h2 : y > 0) (hk : k < 10) (hy : y < 10) :
  (8 * 100 + k * 10 + 8) + (k * 100 + 8 * 10 + 8) - (1 * 100 + 6 * 10 + y * 1) = 8 * 100 + k * 10 + 8 → 
  y = 9 :=
by
  sorry

end NUMINAMATH_GPT_determine_y_value_l186_18668


namespace NUMINAMATH_GPT_largest_of_five_consecutive_ints_15120_l186_18645

theorem largest_of_five_consecutive_ints_15120 :
  ∃ (a b c d e : ℕ), 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
  a * b * c * d * e = 15120 ∧ 
  e = 10 := 
sorry

end NUMINAMATH_GPT_largest_of_five_consecutive_ints_15120_l186_18645


namespace NUMINAMATH_GPT_bus_ticket_probability_l186_18648

theorem bus_ticket_probability :
  let total_tickets := 10 ^ 6
  let choices := Nat.choose 10 6 * 2
  (choices : ℝ) / total_tickets = 0.00042 :=
by
  sorry

end NUMINAMATH_GPT_bus_ticket_probability_l186_18648


namespace NUMINAMATH_GPT_min_distance_to_line_l186_18603

-- Given that a point P(x, y) lies on the line x - y - 1 = 0
-- We need to prove that the minimum value of (x - 2)^2 + (y - 2)^2 is 1/2
theorem min_distance_to_line (x y: ℝ) (h: x - y - 1 = 0) :
  ∃ P : ℝ, P = (x - 2)^2 + (y - 2)^2 ∧ P = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_distance_to_line_l186_18603


namespace NUMINAMATH_GPT_grades_with_fewer_students_l186_18685

-- Definitions of the involved quantities
variables (G1 G2 G5 G1_2 : ℕ)
variables (Set_X : ℕ)

-- Conditions given in the problem
theorem grades_with_fewer_students (h1: G1_2 = Set_X + 30) (h2: G5 = G1 - 30) :
  exists Set_X, G1_2 - Set_X = 30 :=
by 
  sorry

end NUMINAMATH_GPT_grades_with_fewer_students_l186_18685


namespace NUMINAMATH_GPT_gcd_1729_1309_eq_7_l186_18612

theorem gcd_1729_1309_eq_7 : Nat.gcd 1729 1309 = 7 := by
  sorry

end NUMINAMATH_GPT_gcd_1729_1309_eq_7_l186_18612


namespace NUMINAMATH_GPT_integer_solutions_eq_l186_18600

theorem integer_solutions_eq (x y : ℤ) (h : y^2 = x^3 + (x + 1)^2) : (x, y) = (0, 1) ∨ (x, y) = (0, -1) :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_eq_l186_18600


namespace NUMINAMATH_GPT_find_rectangle_width_l186_18633

noncomputable def area_of_square_eq_5times_area_of_rectangle (s l : ℝ) (w : ℝ) :=
  s^2 = 5 * (l * w)

noncomputable def perimeter_of_square_eq_160 (s : ℝ) :=
  4 * s = 160

theorem find_rectangle_width : ∃ w : ℝ, ∀ l : ℝ, 
  area_of_square_eq_5times_area_of_rectangle 40 l w ∧
  perimeter_of_square_eq_160 40 → 
  w = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_rectangle_width_l186_18633


namespace NUMINAMATH_GPT_incorrect_statement_D_l186_18691

theorem incorrect_statement_D : ∃ a : ℝ, a > 0 ∧ (1 - 1 / (2 * a) < 0) := by
  sorry

end NUMINAMATH_GPT_incorrect_statement_D_l186_18691


namespace NUMINAMATH_GPT_katya_needs_at_least_ten_l186_18673

variable (x : ℕ)

def prob_katya : ℚ := 4 / 5
def prob_pen  : ℚ := 1 / 2

def expected_correct (x : ℕ) : ℚ := x * prob_katya + (20 - x) * prob_pen

theorem katya_needs_at_least_ten :
  expected_correct x ≥ 13 → x ≥ 10 :=
  by sorry

end NUMINAMATH_GPT_katya_needs_at_least_ten_l186_18673


namespace NUMINAMATH_GPT_expression_equivalence_l186_18675

theorem expression_equivalence (a b : ℝ) : 2 * a * b - a^2 - b^2 = -((a - b)^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_expression_equivalence_l186_18675


namespace NUMINAMATH_GPT_average_mileage_highway_l186_18639

theorem average_mileage_highway (H : Real) : 
  (∀ d : Real, (d / 7.6) > 23 → false) → 
  (280.6 / 23 = H) → 
  H = 12.2 := by
  sorry

end NUMINAMATH_GPT_average_mileage_highway_l186_18639


namespace NUMINAMATH_GPT_range_of_m_intersection_l186_18656

noncomputable def f (x m : ℝ) : ℝ := (1/x) - (m/(x^2)) - (x/3)

theorem range_of_m_intersection (m : ℝ) :
  (∃! x : ℝ, f x m = 0) ↔ m ∈ (Set.Iic 0 ∪ {2/3}) :=
sorry

end NUMINAMATH_GPT_range_of_m_intersection_l186_18656


namespace NUMINAMATH_GPT_quadratic_roots_l186_18653

theorem quadratic_roots (a b c : ℝ) :
  (∀ (x y : ℝ), ((x, y) = (-2, 12) ∨ (x, y) = (0, -8) ∨ (x, y) = (1, -12) ∨ (x, y) = (3, -8)) → y = a * x^2 + b * x + c) →
  (a * 0^2 + b * 0 + c + 8 = 0) ∧ (a * 3^2 + b * 3 + c + 8 = 0) :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_l186_18653


namespace NUMINAMATH_GPT_perpendicular_x_intercept_l186_18605

theorem perpendicular_x_intercept (x : ℝ) :
  (∃ y : ℝ, 2 * x + 3 * y = 9) ∧ (∃ y : ℝ, y = 5) → x = -10 / 3 :=
by sorry -- Proof omitted

end NUMINAMATH_GPT_perpendicular_x_intercept_l186_18605


namespace NUMINAMATH_GPT_find_w_l186_18652

theorem find_w (u v w : ℝ) (h1 : 10 * u + 8 * v + 5 * w = 160)
  (h2 : v = u + 3) (h3 : w = 2 * v) : w = 13.5714 := by
  -- The proof would go here, but we leave it empty as per instructions.
  sorry

end NUMINAMATH_GPT_find_w_l186_18652


namespace NUMINAMATH_GPT_students_in_class_l186_18641

theorem students_in_class (n : ℕ) (T : ℕ) 
  (average_age_students : T = 16 * n)
  (staff_age : ℕ)
  (increased_average_age : (T + staff_age) / (n + 1) = 17)
  (staff_age_val : staff_age = 49) : n = 32 := 
by
  sorry

end NUMINAMATH_GPT_students_in_class_l186_18641


namespace NUMINAMATH_GPT_triangle_DFG_area_l186_18660

theorem triangle_DFG_area (a b x y : ℝ) (h_ab : a * b = 20) (h_xy : x * y = 8) : 
  (a * b - x * y) / 2 = 6 := 
by
  sorry

end NUMINAMATH_GPT_triangle_DFG_area_l186_18660


namespace NUMINAMATH_GPT_pascal_triangle_count_30_rows_l186_18688

def pascal_row_count (n : Nat) := n + 1

def sum_arithmetic_sequence (a₁ an n : Nat) : Nat :=
  n * (a₁ + an) / 2

theorem pascal_triangle_count_30_rows :
  sum_arithmetic_sequence (pascal_row_count 0) (pascal_row_count 29) 30 = 465 :=
by
  sorry

end NUMINAMATH_GPT_pascal_triangle_count_30_rows_l186_18688


namespace NUMINAMATH_GPT_sports_club_membership_l186_18629

theorem sports_club_membership :
  ∀ (total T B_and_T neither : ℕ),
    total = 30 → 
    T = 19 →
    B_and_T = 9 →
    neither = 2 →
  ∃ (B : ℕ), B = 18 :=
by
  intros total T B_and_T neither ht hT hBandT hNeither
  let B := total - neither - T + B_and_T
  use B
  sorry

end NUMINAMATH_GPT_sports_club_membership_l186_18629


namespace NUMINAMATH_GPT_sum_of_fractions_l186_18624

theorem sum_of_fractions:
  (2 / 5) + (3 / 8) + (1 / 4) = 1 + (1 / 40) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l186_18624


namespace NUMINAMATH_GPT_bookstore_floor_l186_18677

theorem bookstore_floor (academy_floor reading_room_floor bookstore_floor : ℤ)
  (h1: academy_floor = 7)
  (h2: reading_room_floor = academy_floor + 4)
  (h3: bookstore_floor = reading_room_floor - 9) :
  bookstore_floor = 2 :=
by
  sorry

end NUMINAMATH_GPT_bookstore_floor_l186_18677


namespace NUMINAMATH_GPT_number_of_students_in_range_l186_18678

-- Define the basic variables and conditions
variable (a b : ℝ) -- Heights of the rectangles in the histogram

-- Define the total number of surveyed students
def total_students : ℝ := 1500

-- Define the width of each histogram group
def group_width : ℝ := 5

-- State the theorem with the conditions and the expected result
theorem number_of_students_in_range (a b : ℝ) :
    5 * (a + b) * total_students = 7500 * (a + b) :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_number_of_students_in_range_l186_18678


namespace NUMINAMATH_GPT_product_of_roots_of_quadratic_equation_l186_18672

theorem product_of_roots_of_quadratic_equation :
  ∀ (x : ℝ), (x^2 + 14 * x + 48 = -4) → (-6) * (-8) = 48 :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_of_quadratic_equation_l186_18672
