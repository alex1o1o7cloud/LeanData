import Mathlib

namespace NUMINAMATH_GPT_sarah_score_l1774_177497

variable (s g : ℕ)  -- Sarah's and Greg's scores are natural numbers

theorem sarah_score
  (h1 : s = g + 50)  -- Sarah's score is 50 points more than Greg's
  (h2 : (s + g) / 2 = 110)  -- Average of their scores is 110
  : s = 135 :=  -- Prove Sarah's score is 135
by
  sorry

end NUMINAMATH_GPT_sarah_score_l1774_177497


namespace NUMINAMATH_GPT_sum_of_series_eq_one_third_l1774_177402

theorem sum_of_series_eq_one_third :
  ∑' k : ℕ, (2^k / (8^k - 1)) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_sum_of_series_eq_one_third_l1774_177402


namespace NUMINAMATH_GPT_vacation_days_in_march_l1774_177471

theorem vacation_days_in_march 
  (days_worked : ℕ) 
  (days_worked_to_vacation_days : ℕ) 
  (vacation_days_left : ℕ) 
  (days_in_march : ℕ) 
  (days_in_september : ℕ)
  (h1 : days_worked = 300)
  (h2 : days_worked_to_vacation_days = 10)
  (h3 : vacation_days_left = 15)
  (h4 : days_in_september = 2 * days_in_march)
  (h5 : days_worked / days_worked_to_vacation_days - (days_in_march + days_in_september) = vacation_days_left) 
  : days_in_march = 5 := 
by
  sorry

end NUMINAMATH_GPT_vacation_days_in_march_l1774_177471


namespace NUMINAMATH_GPT_less_than_its_reciprocal_l1774_177498

-- Define the numbers as constants
def a := -1/3
def b := -3/2
def c := 1/4
def d := 3/4
def e := 4/3 

-- Define the proposition that needs to be proved
theorem less_than_its_reciprocal (n : ℚ) :
  (n = -3/2 ∨ n = 1/4) ↔ (n < 1/n) :=
by
  sorry

end NUMINAMATH_GPT_less_than_its_reciprocal_l1774_177498


namespace NUMINAMATH_GPT_solve_inequality_l1774_177430

def satisfies_inequality (x : ℝ) : Prop :=
  (3 * x - 4) * (x + 1) / x ≥ 0

theorem solve_inequality :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | -1 ≤ x ∧ x < 0 ∨ x ≥ 4 / 3} :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1774_177430


namespace NUMINAMATH_GPT_line_through_point_with_opposite_intercepts_l1774_177461

theorem line_through_point_with_opposite_intercepts :
  (∃ m : ℝ, (∀ x y : ℝ, y = m * x → (2,3) = (x, y)) ∧ ((∀ a : ℝ, a ≠ 0 → (x / a + y / (-a) = 1) → (2 - 3 = a ∧ a = -1)))) →
  ((∀ x y : ℝ, 3 * x - 2 * y = 0) ∨ (∀ x y : ℝ, x - y + 1 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_line_through_point_with_opposite_intercepts_l1774_177461


namespace NUMINAMATH_GPT_parabola_equation_l1774_177413

theorem parabola_equation (p : ℝ) (h1 : 0 < p) (h2 : p / 2 = 2) : ∀ y x : ℝ, y^2 = -8 * x :=
by
  sorry

end NUMINAMATH_GPT_parabola_equation_l1774_177413


namespace NUMINAMATH_GPT_m_is_perfect_square_l1774_177485

-- Given definitions and conditions
def is_odd (k : ℤ) : Prop := ∃ n : ℤ, k = 2 * n + 1

def is_perfect_square (m : ℕ) : Prop := ∃ a : ℕ, m = a * a

theorem m_is_perfect_square (k m n : ℕ) (h1 : (2 + Real.sqrt 3) ^ k = 1 + m + n * Real.sqrt 3)
  (h2 : 0 < m) (h3 : 0 < n) (h4 : 0 < k) (h5 : is_odd k) : is_perfect_square m := 
sorry

end NUMINAMATH_GPT_m_is_perfect_square_l1774_177485


namespace NUMINAMATH_GPT_initial_assessed_value_l1774_177404

theorem initial_assessed_value (V : ℝ) (tax_rate : ℝ) (new_value : ℝ) (tax_increase : ℝ) 
  (h1 : tax_rate = 0.10) 
  (h2 : new_value = 28000) 
  (h3 : tax_increase = 800) 
  (h4 : tax_rate * new_value = tax_rate * V + tax_increase) : 
  V = 20000 :=
by
  sorry

end NUMINAMATH_GPT_initial_assessed_value_l1774_177404


namespace NUMINAMATH_GPT_vertical_angles_eq_l1774_177486

theorem vertical_angles_eq (A B : Type) (are_vertical : A = B) :
  A = B := 
by
  exact are_vertical

end NUMINAMATH_GPT_vertical_angles_eq_l1774_177486


namespace NUMINAMATH_GPT_simplify_expression_l1774_177492

variables (x y z : ℝ)

theorem simplify_expression (h₁ : x ≠ 2) (h₂ : y ≠ 3) (h₃ : z ≠ 4) : 
  ((x - 2) / (4 - z)) * ((y - 3) / (2 - x)) * ((z - 4) / (3 - y)) = -1 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1774_177492


namespace NUMINAMATH_GPT_hyperbola_foci_l1774_177415

theorem hyperbola_foci :
  (∀ x y : ℝ, x^2 - 2 * y^2 = 1) →
  (∃ c : ℝ, c = (Real.sqrt 6) / 2 ∧ (x = c ∨ x = -c) ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_foci_l1774_177415


namespace NUMINAMATH_GPT_quadrants_contain_points_l1774_177480

def satisfy_inequalities (x y : ℝ) : Prop :=
  y > -3 * x ∧ y > x + 2

def in_quadrant_I (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def in_quadrant_II (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem quadrants_contain_points (x y : ℝ) :
  satisfy_inequalities x y → (in_quadrant_I x y ∨ in_quadrant_II x y) :=
sorry

end NUMINAMATH_GPT_quadrants_contain_points_l1774_177480


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1774_177465

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ) (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1774_177465


namespace NUMINAMATH_GPT_log_one_third_nine_l1774_177407

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_one_third_nine : log_base (1/3) 9 = -2 := by
  sorry

end NUMINAMATH_GPT_log_one_third_nine_l1774_177407


namespace NUMINAMATH_GPT_randy_biscuits_left_l1774_177464

-- Define the function biscuits_left
def biscuits_left (initial: ℚ) (father_gift: ℚ) (mother_gift: ℚ) (brother_eat_percent: ℚ) : ℚ :=
  let total_before_eat := initial + father_gift + mother_gift
  let brother_ate := brother_eat_percent * total_before_eat
  total_before_eat - brother_ate

-- Given conditions
def initial_biscuits : ℚ := 32
def father_gift : ℚ := 2 / 3
def mother_gift : ℚ := 15
def brother_eat_percent : ℚ := 0.3

-- Correct answer as an approximation since we're dealing with real-world numbers
def approx (x y : ℚ) := abs (x - y) < 0.01

-- The proof problem statement in Lean 4
theorem randy_biscuits_left :
  approx (biscuits_left initial_biscuits father_gift mother_gift brother_eat_percent) 33.37 :=
by
  sorry

end NUMINAMATH_GPT_randy_biscuits_left_l1774_177464


namespace NUMINAMATH_GPT_middle_group_frequency_l1774_177490

theorem middle_group_frequency (sample_size : ℕ) (num_rectangles : ℕ)
  (A_middle : ℝ) (other_area_sum : ℝ)
  (h1 : sample_size = 300)
  (h2 : num_rectangles = 9)
  (h3 : A_middle = 1 / 5 * other_area_sum)
  (h4 : other_area_sum + A_middle = 1) :
  sample_size * A_middle = 50 :=
by
  sorry

end NUMINAMATH_GPT_middle_group_frequency_l1774_177490


namespace NUMINAMATH_GPT_find_starting_number_l1774_177473

theorem find_starting_number (k m : ℕ) (hk : 67 = (m - k) / 3 + 1) (hm : m = 300) : k = 102 := by
  sorry

end NUMINAMATH_GPT_find_starting_number_l1774_177473


namespace NUMINAMATH_GPT_gcd_pow_minus_one_l1774_177412

theorem gcd_pow_minus_one {m n : ℕ} (hm : 0 < m) (hn : 0 < n) :
  Nat.gcd (2^m - 1) (2^n - 1) = 2^Nat.gcd m n - 1 :=
sorry

end NUMINAMATH_GPT_gcd_pow_minus_one_l1774_177412


namespace NUMINAMATH_GPT_determine_x_l1774_177427

theorem determine_x (x y : ℤ) (h1 : x + 2 * y = 20) (h2 : y = 5) : x = 10 := 
by 
  sorry

end NUMINAMATH_GPT_determine_x_l1774_177427


namespace NUMINAMATH_GPT_time_difference_l1774_177433

-- Definitions of speeds and distance
def distance : Nat := 12
def alice_speed : Nat := 7
def bob_speed : Nat := 9

-- Calculations of total times based on speeds and distance
def alice_time : Nat := alice_speed * distance
def bob_time : Nat := bob_speed * distance

-- Statement of the problem
theorem time_difference : bob_time - alice_time = 24 := by
  sorry

end NUMINAMATH_GPT_time_difference_l1774_177433


namespace NUMINAMATH_GPT_misread_signs_in_front_of_6_terms_l1774_177456

/-- Define the polynomial function --/
def poly (x : ℝ) : ℝ :=
  10 * x ^ 9 + 9 * x ^ 8 + 8 * x ^ 7 + 7 * x ^ 6 + 6 * x ^ 5 + 5 * x ^ 4 + 4 * x ^ 3 + 3 * x ^ 2 + 2 * x + 1

/-- Xiao Ming's mistaken result --/
def mistaken_result : ℝ := 7

/-- Correct value of the expression at x = -1 --/
def correct_value : ℝ := poly (-1)

/-- The difference due to misreading signs --/
def difference : ℝ := mistaken_result - correct_value

/-- Prove that Xiao Ming misread the signs in front of 6 terms --/
theorem misread_signs_in_front_of_6_terms :
  difference / 2 = 6 :=
by
  simp [difference, correct_value, poly]
  -- the proof steps would go here
  sorry

#eval poly (-1)  -- to validate the correct value
#eval mistaken_result - poly (-1)  -- to validate the difference

end NUMINAMATH_GPT_misread_signs_in_front_of_6_terms_l1774_177456


namespace NUMINAMATH_GPT_solve_for_y_l1774_177422

noncomputable def determinant3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h

noncomputable def determinant2x2 (a b c d : ℝ) : ℝ := 
  a*d - b*c

theorem solve_for_y (b y : ℝ) (h : b ≠ 0) :
  determinant3x3 (y + 2 * b) y y y (y + 2 * b) y y y (y + 2 * b) = 0 → 
  y = -b / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1774_177422


namespace NUMINAMATH_GPT_derivative_at_one_l1774_177470

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 1 + x^2) : f' 1 = -2 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_one_l1774_177470


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1774_177435

noncomputable def M : Set ℕ := { x | 1 < x ∧ x < 7 }
noncomputable def N : Set ℕ := { x | x % 3 ≠ 0 }

theorem intersection_of_M_and_N :
  M ∩ N = {2, 4, 5} := sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1774_177435


namespace NUMINAMATH_GPT_y_increase_for_x_increase_l1774_177496

theorem y_increase_for_x_increase (x y : ℝ) (h : 4 * y = 9) : 12 * y = 27 :=
by
  sorry

end NUMINAMATH_GPT_y_increase_for_x_increase_l1774_177496


namespace NUMINAMATH_GPT_triangle_area_l1774_177436

theorem triangle_area (base height : ℕ) (h_base : base = 35) (h_height : height = 12) :
  (1 / 2 : ℚ) * base * height = 210 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l1774_177436


namespace NUMINAMATH_GPT_smallest_term_of_bn_div_an_is_four_l1774_177400

theorem smallest_term_of_bn_div_an_is_four
  (a : ℕ → ℚ)
  (b : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a n * a (n + 1) = 2 * S n)
  (h3 : b 1 = 16)
  (h4 : ∀ n, b (n + 1) - b n = 2 * n) :
  ∃ n : ℕ, ∀ m : ℕ, (m ≠ 4 → b m / a m > b 4 / a 4) ∧ (n = 4) := sorry

end NUMINAMATH_GPT_smallest_term_of_bn_div_an_is_four_l1774_177400


namespace NUMINAMATH_GPT_binomial_expansion_const_term_l1774_177451

theorem binomial_expansion_const_term (a : ℝ) (h : a > 0) 
  (A : ℝ) (B : ℝ) :
  (A = (15 * a ^ 4)) ∧ (B = 15 * a ^ 2) ∧ (A = 4 * B) → B = 60 := 
by 
  -- The actual proof is omitted
  sorry

end NUMINAMATH_GPT_binomial_expansion_const_term_l1774_177451


namespace NUMINAMATH_GPT_curve_three_lines_intersect_at_origin_l1774_177410

theorem curve_three_lines_intersect_at_origin (a : ℝ) :
  ((∀ x y : ℝ, (x + 2 * y + a) * (x^2 - y^2) = 0 → 
    ((y = x ∨ y = -x ∨ y = - (1/2) * x - a/2) ∧ 
     (x = 0 ∧ y = 0)))) ↔ a = 0 :=
sorry

end NUMINAMATH_GPT_curve_three_lines_intersect_at_origin_l1774_177410


namespace NUMINAMATH_GPT_math_problem_l1774_177428

noncomputable def a : ℕ := 1265
noncomputable def b : ℕ := 168
noncomputable def c : ℕ := 21
noncomputable def d : ℕ := 6
noncomputable def e : ℕ := 3

theorem math_problem : 
  ( ( b / 100 : ℚ ) * (a ^ 2 / c) / (d - e ^ 2) : ℚ ) = -42646.27 :=
by sorry

end NUMINAMATH_GPT_math_problem_l1774_177428


namespace NUMINAMATH_GPT_square_side_length_same_area_l1774_177469

theorem square_side_length_same_area (length width : ℕ) (l_eq : length = 72) (w_eq : width = 18) : 
  ∃ side_length : ℕ, side_length * side_length = length * width ∧ side_length = 36 :=
by
  sorry

end NUMINAMATH_GPT_square_side_length_same_area_l1774_177469


namespace NUMINAMATH_GPT_simplify_expression_l1774_177484

variable (q : Int) -- condition that q is an integer

theorem simplify_expression (q : Int) : 
  ((7 * q + 3) - 3 * q * 2) * 4 + (5 - 2 / 4) * (8 * q - 12) = 40 * q - 42 :=
  by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1774_177484


namespace NUMINAMATH_GPT_average_age_of_new_students_l1774_177420

theorem average_age_of_new_students :
  ∀ (initial_group_avg_age new_group_avg_age : ℝ) (initial_students new_students total_students : ℕ),
  initial_group_avg_age = 14 →
  initial_students = 10 →
  new_group_avg_age = 15 →
  new_students = 5 →
  total_students = initial_students + new_students →
  (new_group_avg_age * total_students - initial_group_avg_age * initial_students) / new_students = 17 :=
by
  intros initial_group_avg_age new_group_avg_age initial_students new_students total_students
  sorry

end NUMINAMATH_GPT_average_age_of_new_students_l1774_177420


namespace NUMINAMATH_GPT_james_weight_gain_l1774_177446

def cheezits_calories (bags : ℕ) (oz_per_bag : ℕ) (cal_per_oz : ℕ) : ℕ :=
  bags * oz_per_bag * cal_per_oz

def chocolate_calories (bars : ℕ) (cal_per_bar : ℕ) : ℕ :=
  bars * cal_per_bar

def popcorn_calories (bags : ℕ) (cal_per_bag : ℕ) : ℕ :=
  bags * cal_per_bag

def run_calories (mins : ℕ) (cal_per_min : ℕ) : ℕ :=
  mins * cal_per_min

def swim_calories (mins : ℕ) (cal_per_min : ℕ) : ℕ :=
  mins * cal_per_min

def cycle_calories (mins : ℕ) (cal_per_min : ℕ) : ℕ :=
  mins * cal_per_min

def total_calories_consumed : ℕ :=
  cheezits_calories 3 2 150 + chocolate_calories 2 250 + popcorn_calories 1 500

def total_calories_burned : ℕ :=
  run_calories 40 12 + swim_calories 30 15 + cycle_calories 20 10

def excess_calories : ℕ :=
  total_calories_consumed - total_calories_burned

def weight_gain (excess_cal : ℕ) (cal_per_lb : ℕ) : ℚ :=
  excess_cal / cal_per_lb

theorem james_weight_gain :
  weight_gain excess_calories 3500 = 770 / 3500 :=
sorry

end NUMINAMATH_GPT_james_weight_gain_l1774_177446


namespace NUMINAMATH_GPT_linear_dependence_k_l1774_177457

theorem linear_dependence_k :
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ 
    (a * (2 : ℝ) + b * (5 : ℝ) = 0) ∧ 
    (a * (3 : ℝ) + b * k = 0) →
  k = 15 / 2 := by
  sorry

end NUMINAMATH_GPT_linear_dependence_k_l1774_177457


namespace NUMINAMATH_GPT_scientific_notation_correct_l1774_177447

/-- Given the weight of the "人" shaped gate of the Three Gorges ship lock -/
def weight_kg : ℝ := 867000

/-- The scientific notation representation of the given weight -/
def scientific_notation_weight_kg : ℝ := 8.67 * 10^5

theorem scientific_notation_correct :
  weight_kg = scientific_notation_weight_kg :=
sorry

end NUMINAMATH_GPT_scientific_notation_correct_l1774_177447


namespace NUMINAMATH_GPT_other_employee_number_l1774_177429

-- Define the conditions
variables (total_employees : ℕ) (sample_size : ℕ) (e1 e2 e3 : ℕ)

-- Define the systematic sampling interval
def sampling_interval (total : ℕ) (size : ℕ) : ℕ := total / size

-- The Lean statement for the proof problem
theorem other_employee_number
  (h1 : total_employees = 52)
  (h2 : sample_size = 4)
  (h3 : e1 = 6)
  (h4 : e2 = 32)
  (h5 : e3 = 45) :
  ∃ e4 : ℕ, e4 = 19 := 
sorry

end NUMINAMATH_GPT_other_employee_number_l1774_177429


namespace NUMINAMATH_GPT_heating_rate_l1774_177405

/-- 
 Andy is making fudge. He needs to raise the temperature of the candy mixture from 60 degrees to 240 degrees. 
 Then, he needs to cool it down to 170 degrees. The candy heats at a certain rate and cools at a rate of 7 degrees/minute.
 It takes 46 minutes for the candy to be done. Prove that the heating rate is 5 degrees per minute.
-/
theorem heating_rate (initial_temp heating_temp cooling_temp : ℝ) (cooling_rate total_time : ℝ) 
  (h1 : initial_temp = 60) (h2 : heating_temp = 240) (h3 : cooling_temp = 170) 
  (h4 : cooling_rate = 7) (h5 : total_time = 46) : 
  ∃ (H : ℝ), H = 5 :=
by 
  -- We declare here that the rate H exists and is 5 degrees per minute.
  let H : ℝ := 5
  existsi H
  sorry

end NUMINAMATH_GPT_heating_rate_l1774_177405


namespace NUMINAMATH_GPT_age_difference_l1774_177426

variables (X Y Z : ℕ)

theorem age_difference (h : X + Y = Y + Z + 12) : X - Z = 12 :=
sorry

end NUMINAMATH_GPT_age_difference_l1774_177426


namespace NUMINAMATH_GPT_expected_value_of_winning_is_2550_l1774_177421

-- Definitions based on the conditions
def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℚ := 1 / 8
def winnings (n : ℕ) : ℕ := n^2

-- Expected value calculation based on the conditions
noncomputable def expected_value : ℚ :=
  (outcomes.map (λ n => probability n * winnings n)).sum

-- Proposition stating that the expected value is 25.50
theorem expected_value_of_winning_is_2550 : expected_value = 25.50 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_of_winning_is_2550_l1774_177421


namespace NUMINAMATH_GPT_simple_interest_rate_l1774_177495

theorem simple_interest_rate
  (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ)
  (h1 : SI = 400)
  (h2 : P = 800)
  (h3 : T = 2) :
  R = 25 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l1774_177495


namespace NUMINAMATH_GPT_remainder_when_3x_7y_5z_div_31517_l1774_177425

theorem remainder_when_3x_7y_5z_div_31517
  (x y z : ℕ)
  (hx : x % 23 = 9)
  (hy : y % 29 = 15)
  (hz : z % 37 = 12) :
  (3 * x + 7 * y - 5 * z) % 31517 = ((69 * (x / 23) + 203 * (y / 29) - 185 * (z / 37) + 72) % 31517) := 
sorry

end NUMINAMATH_GPT_remainder_when_3x_7y_5z_div_31517_l1774_177425


namespace NUMINAMATH_GPT_perimeter_of_rectangle_l1774_177458

theorem perimeter_of_rectangle (area : ℝ) (num_squares : ℕ) (square_side : ℝ) (width : ℝ) (height : ℝ) 
  (h1 : area = 216) (h2 : num_squares = 6) (h3 : area / num_squares = square_side^2)
  (h4 : width = 3 * square_side) (h5 : height = 2 * square_side) : 
  2 * (width + height) = 60 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_rectangle_l1774_177458


namespace NUMINAMATH_GPT_optimal_bicycle_point_l1774_177463

noncomputable def distance_A_B : ℝ := 30  -- Distance between A and B is 30 km
noncomputable def midpoint_distance : ℝ := distance_A_B / 2  -- Distance between midpoint C to both A and B is 15 km
noncomputable def walking_speed : ℝ := 5  -- Walking speed is 5 km/h
noncomputable def biking_speed : ℝ := 20  -- Biking speed is 20 km/h

theorem optimal_bicycle_point : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ (30 - x + 4 * x = 60 - 3 * x) → x = 5 :=
by sorry

end NUMINAMATH_GPT_optimal_bicycle_point_l1774_177463


namespace NUMINAMATH_GPT_batting_average_drop_l1774_177443

theorem batting_average_drop 
    (avg : ℕ)
    (innings : ℕ)
    (high : ℕ)
    (high_low_diff : ℕ)
    (low : ℕ)
    (total_runs : ℕ)
    (new_avg : ℕ)

    (h1 : avg = 50)
    (h2 : innings = 40)
    (h3 : high = 174)
    (h4 : high = low + 172)
    (h5 : total_runs = avg * innings)
    (h6 : new_avg = (total_runs - high - low) / (innings - 2)) :

  avg - new_avg = 2 :=
by
  sorry

end NUMINAMATH_GPT_batting_average_drop_l1774_177443


namespace NUMINAMATH_GPT_binary1011_eq_11_l1774_177438

-- Define a function to convert a binary number represented as a list of bits to a decimal number.
def binaryToDecimal (bits : List (Fin 2)) : Nat :=
  bits.foldr (λ (bit : Fin 2) (acc : Nat) => acc * 2 + bit.val) 0

-- The binary number 1011 represented as a list of bits.
def binary1011 : List (Fin 2) := [1, 0, 1, 1]

-- The theorem stating that the decimal equivalent of binary 1011 is 11.
theorem binary1011_eq_11 : binaryToDecimal binary1011 = 11 :=
by
  sorry

end NUMINAMATH_GPT_binary1011_eq_11_l1774_177438


namespace NUMINAMATH_GPT_ceil_sum_sqrt_eval_l1774_177411

theorem ceil_sum_sqrt_eval : 
  (⌈Real.sqrt 2⌉ + ⌈Real.sqrt 22⌉ + ⌈Real.sqrt 222⌉) = 22 := 
by
  sorry

end NUMINAMATH_GPT_ceil_sum_sqrt_eval_l1774_177411


namespace NUMINAMATH_GPT_area_of_rectangular_field_l1774_177472

theorem area_of_rectangular_field (length width perimeter : ℕ) 
  (h_perimeter : perimeter = 2 * (length + width)) 
  (h_length : length = 15) 
  (h_perimeter_value : perimeter = 70) : 
  (length * width = 300) :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangular_field_l1774_177472


namespace NUMINAMATH_GPT_nearest_integer_to_expansion_l1774_177468

theorem nearest_integer_to_expansion : 
  let a := (3 + 2 * Real.sqrt 2)
  let b := (3 - 2 * Real.sqrt 2)
  abs (a^4 - 1090) < 1 :=
by
  let a := (3 + 2 * Real.sqrt 2)
  let b := (3 - 2 * Real.sqrt 2)
  sorry

end NUMINAMATH_GPT_nearest_integer_to_expansion_l1774_177468


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1774_177462

theorem boat_speed_in_still_water (D V_s t_down t_up : ℝ) (h_val : V_s = 3) (h_down : D = (15 + V_s) * t_down) (h_up : D = (15 - V_s) * t_up) : 15 = 15 :=
by
  have h1 : 15 = (D / 1 - V_s) := sorry
  have h2 : 15 = (D / 1.5 + V_s) := sorry
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1774_177462


namespace NUMINAMATH_GPT_sin_75_l1774_177439

theorem sin_75 :
  Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end NUMINAMATH_GPT_sin_75_l1774_177439


namespace NUMINAMATH_GPT_sin_double_angle_identity_l1774_177466

theorem sin_double_angle_identity (x : ℝ) (h : Real.sin (x + π/4) = -3/5) : Real.sin (2 * x) = -7/25 := 
by 
  sorry

end NUMINAMATH_GPT_sin_double_angle_identity_l1774_177466


namespace NUMINAMATH_GPT_c_S_power_of_2_l1774_177453

variables (m : ℕ) (S : String)

-- condition: m > 1
def is_valid_m (m : ℕ) : Prop := m > 1

-- function c(S)
def c (S : String) : ℕ := sorry  -- actual implementation is skipped

-- function to check if a number represented by a string is divisible by m
def is_divisible_by (n m : ℕ) : Prop := n % m = 0

-- Property that c(S) can take only powers of 2
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem c_S_power_of_2 (m : ℕ) (S : String) (h1 : is_valid_m m) :
  is_power_of_two (c S) :=
sorry

end NUMINAMATH_GPT_c_S_power_of_2_l1774_177453


namespace NUMINAMATH_GPT_total_profit_is_18900_l1774_177445

-- Defining the conditions
variable (x : ℕ)  -- A's initial investment
variable (A_share : ℕ := 6300)  -- A's share in rupees

-- Total profit calculation
def total_annual_gain : ℕ :=
  (x * 12) + (2 * x * 6) + (3 * x * 4)

-- The main statement
theorem total_profit_is_18900 (x : ℕ) (A_share : ℕ := 6300) :
  3 * A_share = total_annual_gain x :=
by sorry

end NUMINAMATH_GPT_total_profit_is_18900_l1774_177445


namespace NUMINAMATH_GPT_constructible_angles_l1774_177474

def is_constructible (θ : ℝ) : Prop :=
  -- Define that θ is constructible if it can be constructed using compass and straightedge.
  sorry

theorem constructible_angles (α : ℝ) (β : ℝ) (k n : ℤ) (hβ : is_constructible β) :
  is_constructible (k * α / 2^n + β) :=
sorry

end NUMINAMATH_GPT_constructible_angles_l1774_177474


namespace NUMINAMATH_GPT_smallest_sum_of_three_integers_l1774_177487

theorem smallest_sum_of_three_integers (a b c : ℕ) (h1: a ≠ b) (h2: b ≠ c) (h3: a ≠ c) (h4: a * b * c = 72) :
  a + b + c = 13 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_three_integers_l1774_177487


namespace NUMINAMATH_GPT_correct_option_D_l1774_177483

theorem correct_option_D (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by sorry

end NUMINAMATH_GPT_correct_option_D_l1774_177483


namespace NUMINAMATH_GPT_liquid_mixture_ratio_l1774_177437

theorem liquid_mixture_ratio (m1 m2 m3 : ℝ) (ρ1 ρ2 ρ3 : ℝ) (k : ℝ)
  (hρ1 : ρ1 = 6 * k) (hρ2 : ρ2 = 3 * k) (hρ3 : ρ3 = 2 * k)
  (h_condition : m1 ≥ 3.5 * m2)
  (h_arith_mean : (m1 + m2 + m3) / (m1 / ρ1 + m2 / ρ2 + m3 / ρ3) = (ρ1 + ρ2 + ρ3) / 3) :
    ∃ x y : ℝ, x ≤ 2/7 ∧ (4 * x + 15 * y = 7) := sorry

end NUMINAMATH_GPT_liquid_mixture_ratio_l1774_177437


namespace NUMINAMATH_GPT_find_age_l1774_177414

variable (x : ℤ)

def age_4_years_hence := x + 4
def age_4_years_ago := x - 4
def brothers_age := x - 6

theorem find_age (hx : x = 4 * (x + 4) - 4 * (x - 4) + 1/2 * (x - 6)) : x = 58 :=
sorry

end NUMINAMATH_GPT_find_age_l1774_177414


namespace NUMINAMATH_GPT_glasses_needed_l1774_177416

theorem glasses_needed (total_juice : ℕ) (juice_per_glass : ℕ) : Prop :=
  total_juice = 153 ∧ juice_per_glass = 30 → (total_juice + juice_per_glass - 1) / juice_per_glass = 6

-- This will state our theorem but we include sorry to omit the proof.

end NUMINAMATH_GPT_glasses_needed_l1774_177416


namespace NUMINAMATH_GPT_evaluate_expression_l1774_177403

theorem evaluate_expression :
  (⌈(19 / 7 : ℚ) - ⌈(35 / 19 : ℚ)⌉⌉ / ⌈(35 / 7 : ℚ) + ⌈((7 * 19) / 35 : ℚ)⌉⌉) = (1 / 9 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1774_177403


namespace NUMINAMATH_GPT_KarenParagraphCount_l1774_177493

theorem KarenParagraphCount :
  ∀ (num_essays num_short_ans num_paragraphs total_time essay_time short_ans_time paragraph_time : ℕ),
    (num_essays = 2) →
    (num_short_ans = 15) →
    (total_time = 240) →
    (essay_time = 60) →
    (short_ans_time = 3) →
    (paragraph_time = 15) →
    (total_time = num_essays * essay_time + num_short_ans * short_ans_time + num_paragraphs * paragraph_time) →
    num_paragraphs = 5 :=
by
  sorry

end NUMINAMATH_GPT_KarenParagraphCount_l1774_177493


namespace NUMINAMATH_GPT_part_a_part_b_l1774_177476

-- Part (a)
theorem part_a (ABC : Type) (M: ABC) (R_a R_b R_c r : ℝ):
  ∀ (ABC : Type) (A B C : ABC) (M : ABC), 
  R_a + R_b + R_c ≥ 6 * r := sorry

-- Part (b)
theorem part_b (ABC : Type) (M: ABC) (R_a R_b R_c r : ℝ):
  ∀ (ABC : Type) (A B C : ABC) (M : ABC), 
  R_a^2 + R_b^2 + R_c^2 ≥ 12 * r^2 := sorry

end NUMINAMATH_GPT_part_a_part_b_l1774_177476


namespace NUMINAMATH_GPT_sum_of_cubes_identity_l1774_177455

theorem sum_of_cubes_identity (a b : ℝ) (h : a / (1 + b) + b / (1 + a) = 1) : a^3 + b^3 = a + b := by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_identity_l1774_177455


namespace NUMINAMATH_GPT_marathon_fraction_l1774_177423

theorem marathon_fraction :
  ∃ (f : ℚ), (2 * 7) = (6 + (6 + 6 * f)) ∧ f = 1 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_marathon_fraction_l1774_177423


namespace NUMINAMATH_GPT_daffodil_stamps_count_l1774_177442

theorem daffodil_stamps_count (r d : ℕ) (h1 : r = 2) (h2 : r = d) : d = 2 := by
  sorry

end NUMINAMATH_GPT_daffodil_stamps_count_l1774_177442


namespace NUMINAMATH_GPT_number_of_chickens_l1774_177419

variables (C G Ch : ℕ)

theorem number_of_chickens (h1 : C = 9) (h2 : G = 4 * C) (h3 : G = 2 * Ch) : Ch = 18 :=
by
  sorry

end NUMINAMATH_GPT_number_of_chickens_l1774_177419


namespace NUMINAMATH_GPT_interior_box_surface_area_l1774_177418

-- Given conditions
def original_length : ℕ := 40
def original_width : ℕ := 60
def corner_side : ℕ := 8

-- Calculate the initial area
def area_original : ℕ := original_length * original_width

-- Calculate the area of one corner
def area_corner : ℕ := corner_side * corner_side

-- Calculate the total area removed by four corners
def total_area_removed : ℕ := 4 * area_corner

-- Theorem to state the final area remaining
theorem interior_box_surface_area : 
  area_original - total_area_removed = 2144 :=
by
  -- Place the proof here
  sorry

end NUMINAMATH_GPT_interior_box_surface_area_l1774_177418


namespace NUMINAMATH_GPT_animal_shelter_dogs_l1774_177459

theorem animal_shelter_dogs (D C R : ℕ) 
  (h₁ : 15 * C = 7 * D)
  (h₂ : 15 * R = 4 * D)
  (h₃ : 15 * (C + 20) = 11 * D)
  (h₄ : 15 * (R + 10) = 6 * D) : 
  D = 75 :=
by
  -- Proof part is omitted
  sorry

end NUMINAMATH_GPT_animal_shelter_dogs_l1774_177459


namespace NUMINAMATH_GPT_regular_price_of_fish_l1774_177406

theorem regular_price_of_fish (discounted_price_per_quarter_pound : ℝ)
  (discount : ℝ) (hp1 : discounted_price_per_quarter_pound = 2) (hp2 : discount = 0.4) :
  ∃ x : ℝ, x = (40 / 3) :=
by
  sorry

end NUMINAMATH_GPT_regular_price_of_fish_l1774_177406


namespace NUMINAMATH_GPT_decimal_equivalent_l1774_177441

theorem decimal_equivalent (x : ℚ) (h : x = 16 / 50) : x = 32 / 100 :=
by
  sorry

end NUMINAMATH_GPT_decimal_equivalent_l1774_177441


namespace NUMINAMATH_GPT_sin_identity_l1774_177450

theorem sin_identity (α : ℝ) (h : Real.sin (π * α) = 4 / 5) : 
  Real.sin (π / 2 + 2 * α) = -24 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_identity_l1774_177450


namespace NUMINAMATH_GPT_calculation_power_l1774_177499

theorem calculation_power :
  (0.125 : ℝ) ^ 2012 * (2 ^ 2012) ^ 3 = 1 :=
sorry

end NUMINAMATH_GPT_calculation_power_l1774_177499


namespace NUMINAMATH_GPT_greatest_drop_in_price_l1774_177479

def jan_change : ℝ := -0.75
def feb_change : ℝ := 1.50
def mar_change : ℝ := -3.00
def apr_change : ℝ := 2.50
def may_change : ℝ := -0.25
def jun_change : ℝ := 0.80
def jul_change : ℝ := -2.75
def aug_change : ℝ := -1.20

theorem greatest_drop_in_price : 
  mar_change = min (min (min (min (min (min jan_change jul_change) aug_change) may_change) feb_change) apr_change) jun_change :=
by
  -- This statement is where the proof would go.
  sorry

end NUMINAMATH_GPT_greatest_drop_in_price_l1774_177479


namespace NUMINAMATH_GPT_bananas_to_oranges_l1774_177440

theorem bananas_to_oranges :
  (3 / 4 : ℝ) * 16 = 12 →
  (2 / 3 : ℝ) * 9 = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_bananas_to_oranges_l1774_177440


namespace NUMINAMATH_GPT_greatest_of_given_numbers_l1774_177417

-- Defining the given conditions
def a := 1000 + 0.01
def b := 1000 * 0.01
def c := 1000 / 0.01
def d := 0.01 / 1000
def e := 1000 - 0.01

-- Prove that c is the greatest
theorem greatest_of_given_numbers : c = max a (max b (max d e)) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_greatest_of_given_numbers_l1774_177417


namespace NUMINAMATH_GPT_y_coordinate_of_second_point_l1774_177489

theorem y_coordinate_of_second_point
  (m n : ℝ)
  (h₁ : m = 2 * n + 3)
  (h₂ : m + 2 = 2 * (n + 1) + 3) :
  (n + 1) = n + 1 :=
by
  -- proof to be provided
  sorry

end NUMINAMATH_GPT_y_coordinate_of_second_point_l1774_177489


namespace NUMINAMATH_GPT_nonneg_sets_property_l1774_177494

open Set Nat

theorem nonneg_sets_property (A : Set ℕ) :
  (∀ m n : ℕ, m + n ∈ A → m * n ∈ A) ↔
  (A = ∅ ∨ A = {0} ∨ A = {0, 1} ∨ A = {0, 1, 2} ∨ A = {0, 1, 2, 3} ∨ A = {0, 1, 2, 3, 4} ∨ A = { n | 0 ≤ n }) :=
sorry

end NUMINAMATH_GPT_nonneg_sets_property_l1774_177494


namespace NUMINAMATH_GPT_no_three_distinct_nat_numbers_sum_prime_l1774_177401

theorem no_three_distinct_nat_numbers_sum_prime:
  ¬∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ 
  Nat.Prime (a + b) ∧ Nat.Prime (a + c) ∧ Nat.Prime (b + c) := 
sorry

end NUMINAMATH_GPT_no_three_distinct_nat_numbers_sum_prime_l1774_177401


namespace NUMINAMATH_GPT_max_license_plates_l1774_177432

noncomputable def max_distinct_plates (m n : ℕ) : ℕ :=
  m ^ (n - 1)

theorem max_license_plates :
  max_distinct_plates 10 6 = 100000 := by
  sorry

end NUMINAMATH_GPT_max_license_plates_l1774_177432


namespace NUMINAMATH_GPT_marathons_total_distance_l1774_177408

theorem marathons_total_distance :
  ∀ (m y : ℕ),
  (26 + 385 / 1760 : ℕ) = 26 ∧ 385 % 1760 = 385 →
  15 * 26 + 15 * 385 / 1760 = m + 495 / 1760 ∧
  15 * 385 % 1760 = 495 →
  0 ≤ 495 ∧ 495 < 1760 →
  y = 495 := by
  intros
  sorry

end NUMINAMATH_GPT_marathons_total_distance_l1774_177408


namespace NUMINAMATH_GPT_remainder_of_product_mod_7_l1774_177477

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_of_product_mod_7_l1774_177477


namespace NUMINAMATH_GPT_correct_statement_l1774_177434

-- Conditions as definitions
def deductive_reasoning (p q r : Prop) : Prop :=
  (p → q) → (q → r) → (p → r)

def correctness_of_conclusion := true  -- Indicates statement is defined to be correct

def pattern_of_reasoning (p q r : Prop) : Prop :=
  deductive_reasoning p q r

-- Statement to prove
theorem correct_statement (p q r : Prop) :
  pattern_of_reasoning p q r = deductive_reasoning p q r :=
by sorry

end NUMINAMATH_GPT_correct_statement_l1774_177434


namespace NUMINAMATH_GPT_calculate_kevin_training_time_l1774_177475

theorem calculate_kevin_training_time : 
  ∀ (laps : ℕ) 
    (track_length : ℕ) 
    (run1_distance : ℕ) 
    (run1_speed : ℕ) 
    (walk_distance : ℕ) 
    (walk_speed : Real) 
    (run2_distance : ℕ) 
    (run2_speed : ℕ) 
    (minutes : ℕ) 
    (seconds : Real),
    laps = 8 →
    track_length = 500 →
    run1_distance = 200 →
    run1_speed = 3 →
    walk_distance = 100 →
    walk_speed = 1.5 →
    run2_distance = 200 →
    run2_speed = 4 →
    minutes = 24 →
    seconds = 27 →
    (∀ (t1 t2 t3 t_total t_training : Real),
      t1 = run1_distance / run1_speed →
      t2 = walk_distance / walk_speed →
      t3 = run2_distance / run2_speed →
      t_total = t1 + t2 + t3 →
      t_training = laps * t_total →
      t_training = (minutes * 60 + seconds)) := 
by
  intros laps track_length run1_distance run1_speed walk_distance walk_speed run2_distance run2_speed minutes seconds
  intros h_laps h_track_length h_run1_distance h_run1_speed h_walk_distance h_walk_speed h_run2_distance h_run2_speed h_minutes h_seconds
  intros t1 t2 t3 t_total t_training
  intros h_t1 h_t2 h_t3 h_t_total h_t_training
  sorry

end NUMINAMATH_GPT_calculate_kevin_training_time_l1774_177475


namespace NUMINAMATH_GPT_solution_l1774_177491

theorem solution :
  ∀ (x : ℝ), x ≠ 0 → (9 * x) ^ 18 = (27 * x) ^ 9 → x = 1 / 3 :=
by
  intro x
  intro h
  intro h_eq
  sorry

end NUMINAMATH_GPT_solution_l1774_177491


namespace NUMINAMATH_GPT_not_possible_2018_people_in_2019_minutes_l1774_177409

-- Definitions based on conditions
def initial_people (t : ℕ) : ℕ := 0
def changed_people (x y : ℕ) : ℕ := 2 * x - y

theorem not_possible_2018_people_in_2019_minutes :
  ¬ ∃ (x y : ℕ), (x + y = 2019) ∧ (2 * x - y = 2018) :=
by
  sorry

end NUMINAMATH_GPT_not_possible_2018_people_in_2019_minutes_l1774_177409


namespace NUMINAMATH_GPT_heather_counts_209_l1774_177454

def alice_numbers (n : ℕ) : ℕ := 5 * n - 2
def general_skip_numbers (m : ℕ) : ℕ := 3 * m - 1
def heather_number := 209

theorem heather_counts_209 :
  (∀ n, alice_numbers n > 0 ∧ alice_numbers n ≤ 500 → ¬heather_number = alice_numbers n) ∧
  (∀ m, general_skip_numbers m > 0 ∧ general_skip_numbers m ≤ 500 → ¬heather_number = general_skip_numbers m) ∧
  (1 ≤ heather_number ∧ heather_number ≤ 500) :=
by
  sorry

end NUMINAMATH_GPT_heather_counts_209_l1774_177454


namespace NUMINAMATH_GPT_correct_quadratic_opens_upwards_l1774_177481

-- Define the quadratic functions
def A (x : ℝ) : ℝ := 1 - x - 6 * x^2
def B (x : ℝ) : ℝ := -8 * x + x^2 + 1
def C (x : ℝ) : ℝ := (1 - x) * (x + 5)
def D (x : ℝ) : ℝ := 2 - (5 - x)^2

-- The theorem stating that function B is the one that opens upwards
theorem correct_quadratic_opens_upwards :
  ∃ (f : ℝ → ℝ) (h : f = B), ∀ (a b c : ℝ), f x = a * x^2 + b * x + c → a > 0 :=
sorry

end NUMINAMATH_GPT_correct_quadratic_opens_upwards_l1774_177481


namespace NUMINAMATH_GPT_matt_new_average_commission_l1774_177444

noncomputable def new_average_commission (x : ℝ) : ℝ :=
  (5 * x + 1000) / 6

theorem matt_new_average_commission
  (x : ℝ)
  (h1 : (5 * x + 1000) / 6 = x + 150)
  (h2 : x = 100) :
  new_average_commission x = 250 :=
by
  sorry

end NUMINAMATH_GPT_matt_new_average_commission_l1774_177444


namespace NUMINAMATH_GPT_part_I_part_II_l1774_177452

def f (x a : ℝ) : ℝ := |2 * x + 1| + |2 * x - a| + a

theorem part_I (x : ℝ) (h₁ : f x 3 > 7) : sorry := sorry

theorem part_II (a : ℝ) (h₂ : ∀ (x : ℝ), f x a ≥ 3) : sorry := sorry

end NUMINAMATH_GPT_part_I_part_II_l1774_177452


namespace NUMINAMATH_GPT_find_d_l1774_177460

theorem find_d (d : ℤ) :
  (∀ x : ℤ, (4 * x^3 + 13 * x^2 + d * x + 18 = 0 ↔ x = -3)) →
  d = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l1774_177460


namespace NUMINAMATH_GPT_initial_value_calculation_l1774_177478

theorem initial_value_calculation (P : ℝ) (h1 : ∀ n : ℕ, 0 ≤ n →
                                (P:ℝ) * (1 + 1/8) ^ n = 78468.75 → n = 2) :
  P = 61952 :=
sorry

end NUMINAMATH_GPT_initial_value_calculation_l1774_177478


namespace NUMINAMATH_GPT_expected_value_is_correct_l1774_177467

def probability_of_rolling_one : ℚ := 1 / 4

def probability_of_other_numbers : ℚ := 3 / 4 / 5

def win_amount : ℚ := 8

def loss_amount : ℚ := -3

def expected_value : ℚ := (probability_of_rolling_one * win_amount) + 
                          (probability_of_other_numbers * 5 * loss_amount)

theorem expected_value_is_correct : expected_value = -0.25 :=
by 
  unfold expected_value probability_of_rolling_one probability_of_other_numbers win_amount loss_amount
  sorry

end NUMINAMATH_GPT_expected_value_is_correct_l1774_177467


namespace NUMINAMATH_GPT_problem_l1774_177424

noncomputable def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
noncomputable def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

variable (f g : ℝ → ℝ)
variable (h₁ : is_odd f)
variable (h₂ : is_even g)
variable (h₃ : ∀ x, f x - g x = 2 * x^3 + x^2 + 3)

theorem problem : f 2 + g 2 = 9 :=
by sorry

end NUMINAMATH_GPT_problem_l1774_177424


namespace NUMINAMATH_GPT_power_multiplication_same_base_l1774_177431

theorem power_multiplication_same_base :
  (10 ^ 655 * 10 ^ 650 = 10 ^ 1305) :=
by {
  sorry
}

end NUMINAMATH_GPT_power_multiplication_same_base_l1774_177431


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1774_177488

def A : Set ℝ := { x | x > 2 ∨ x < -1 }
def B : Set ℝ := { x | (x + 1) * (4 - x) < 4 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | x > 3 ∨ x < -1 } := sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1774_177488


namespace NUMINAMATH_GPT_two_sum_fourth_power_square_l1774_177482

-- Define the condition
def sum_zero (x y z : ℤ) : Prop := x + y + z = 0

-- The theorem to be proven
theorem two_sum_fourth_power_square (x y z : ℤ) (h : sum_zero x y z) : ∃ k : ℤ, 2 * (x^4 + y^4 + z^4) = k^2 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_two_sum_fourth_power_square_l1774_177482


namespace NUMINAMATH_GPT_pure_alcohol_addition_l1774_177448

variables (P : ℝ) (V : ℝ := 14.285714285714286 ) (initial_volume : ℝ := 100) (final_percent_alcohol : ℝ := 0.30)

theorem pure_alcohol_addition :
  P / 100 * initial_volume + V = final_percent_alcohol * (initial_volume + V) :=
by
  sorry

end NUMINAMATH_GPT_pure_alcohol_addition_l1774_177448


namespace NUMINAMATH_GPT_prove_k_in_terms_of_x_l1774_177449

variables {A B k x : ℝ}

-- given conditions
def positive_numbers (A B : ℝ) := A > 0 ∧ B > 0
def ratio_condition (A B k : ℝ) := A = k * B
def percentage_condition (A B x : ℝ) := A = B + (x / 100) * B

-- proof statement
theorem prove_k_in_terms_of_x (A B k x : ℝ) (h1 : positive_numbers A B) (h2 : ratio_condition A B k) (h3 : percentage_condition A B x) (h4 : k > 1) :
  k = 1 + x / 100 :=
sorry

end NUMINAMATH_GPT_prove_k_in_terms_of_x_l1774_177449
