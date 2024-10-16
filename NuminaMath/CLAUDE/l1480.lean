import Mathlib

namespace NUMINAMATH_CALUDE_dormitory_students_l1480_148030

theorem dormitory_students (T : ℝ) (h1 : T > 0) : 
  let first_year := T / 2
  let second_year := T / 2
  let first_year_undeclared := (4 / 5) * first_year
  let first_year_declared := first_year - first_year_undeclared
  let second_year_declared := 4 * first_year_declared
  let second_year_undeclared := second_year - second_year_declared
  second_year_undeclared / T = 1 / 10 := by
sorry


end NUMINAMATH_CALUDE_dormitory_students_l1480_148030


namespace NUMINAMATH_CALUDE_marbles_probability_l1480_148085

def total_marbles : ℕ := 13
def black_marbles : ℕ := 4
def red_marbles : ℕ := 3
def green_marbles : ℕ := 6
def drawn_marbles : ℕ := 2

def prob_same_color : ℚ := 
  (black_marbles * (black_marbles - 1) + 
   red_marbles * (red_marbles - 1) + 
   green_marbles * (green_marbles - 1)) / 
  (total_marbles * (total_marbles - 1))

theorem marbles_probability : 
  prob_same_color = 4 / 13 :=
sorry

end NUMINAMATH_CALUDE_marbles_probability_l1480_148085


namespace NUMINAMATH_CALUDE_triangle_side_bounds_l1480_148019

/-- 
For a triangle with integer side lengths a, b, c and perimeter k, 
where a ≤ b ≤ c, the following inequalities hold:
2 - (k - 2⌊k/2⌋) ≤ a ≤ ⌊k/3⌋
⌊(k+4)/4⌋ ≤ b ≤ ⌊(k-1)/2⌋
⌊(k+2)/3⌋ ≤ c ≤ ⌊(k-1)/2⌋
-/
theorem triangle_side_bounds (a b c k : ℕ) 
  (h1 : a + b + c = k) 
  (h2 : a ≤ b) 
  (h3 : b ≤ c) :
  (2 - (k - 2*(k/2)) ≤ a ∧ a ≤ k/3) ∧
  ((k+4)/4 ≤ b ∧ b ≤ (k-1)/2) ∧
  ((k+2)/3 ≤ c ∧ c ≤ (k-1)/2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_bounds_l1480_148019


namespace NUMINAMATH_CALUDE_fraction_simplification_l1480_148064

theorem fraction_simplification 
  (x y z : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (hyz : y - z / x ≠ 0) : 
  (x + z / y) / (y + z / x) = x / y :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1480_148064


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1480_148002

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- The theorem statement -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (-1, x)
  let b : ℝ × ℝ := (x, -4)
  parallel a b → x = 2 ∨ x = -2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1480_148002


namespace NUMINAMATH_CALUDE_cos_neg_135_degrees_l1480_148054

theorem cos_neg_135_degrees :
  Real.cos ((-135 : ℝ) * π / 180) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_neg_135_degrees_l1480_148054


namespace NUMINAMATH_CALUDE_rectangle_arrangement_exists_l1480_148040

theorem rectangle_arrangement_exists : ∃ (a b c d : ℕ+), 
  (a * b + c * d = 49) ∧ 
  ((2 * (a + b) = 4 * (c + d)) ∨ (2 * (c + d) = 4 * (a + b))) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_arrangement_exists_l1480_148040


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1480_148015

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (6 - 3*i) / (-2 + 5*i) = (-27 : ℚ) / 29 - (24 : ℚ) / 29 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1480_148015


namespace NUMINAMATH_CALUDE_solution_set_l1480_148049

theorem solution_set (x y z : ℝ) : 
  x = (4 * z^2) / (1 + 4 * z^2) ∧
  y = (4 * x^2) / (1 + 4 * x^2) ∧
  z = (4 * y^2) / (1 + 4 * y^2) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l1480_148049


namespace NUMINAMATH_CALUDE_correct_final_bill_amount_l1480_148012

/-- Calculates the final bill amount after applying two late fees -/
def final_bill_amount (original_bill : ℝ) (first_fee_rate : ℝ) (second_fee_rate : ℝ) : ℝ :=
  let after_first_fee := original_bill * (1 + first_fee_rate)
  after_first_fee * (1 + second_fee_rate)

/-- Theorem stating that the final bill amount is correct -/
theorem correct_final_bill_amount :
  final_bill_amount 250 0.02 0.03 = 262.65 := by
  sorry

#eval final_bill_amount 250 0.02 0.03

end NUMINAMATH_CALUDE_correct_final_bill_amount_l1480_148012


namespace NUMINAMATH_CALUDE_find_b_value_l1480_148022

theorem find_b_value (a b : ℝ) (eq1 : 3 * a + 2 = 2) (eq2 : b - a = 2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l1480_148022


namespace NUMINAMATH_CALUDE_opposite_of_negative_eight_l1480_148032

theorem opposite_of_negative_eight : 
  -((-8 : ℤ)) = (8 : ℤ) := by
sorry

end NUMINAMATH_CALUDE_opposite_of_negative_eight_l1480_148032


namespace NUMINAMATH_CALUDE_rooms_already_painted_l1480_148082

theorem rooms_already_painted
  (total_rooms : ℕ)
  (time_per_room : ℕ)
  (time_left : ℕ)
  (h1 : total_rooms = 10)
  (h2 : time_per_room = 8)
  (h3 : time_left = 16) :
  total_rooms - (time_left / time_per_room) = 8 :=
by sorry

end NUMINAMATH_CALUDE_rooms_already_painted_l1480_148082


namespace NUMINAMATH_CALUDE_range_of_m_l1480_148007

theorem range_of_m (f : ℝ → ℝ) (h : ∀ x ∈ Set.Icc 0 1, f x ≥ m) :
  Set.Iic (-3 : ℝ) = {m : ℝ | ∀ x ∈ Set.Icc 0 1, f x ≥ m} :=
by sorry

#check range_of_m (fun x ↦ x^2 - 4*x)

end NUMINAMATH_CALUDE_range_of_m_l1480_148007


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_4_10_l1480_148045

theorem gcf_lcm_sum_4_10 : Nat.gcd 4 10 + Nat.lcm 4 10 = 22 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_4_10_l1480_148045


namespace NUMINAMATH_CALUDE_brownie_pieces_l1480_148065

theorem brownie_pieces (pan_length pan_width piece_length piece_width : ℕ) 
  (h1 : pan_length = 24)
  (h2 : pan_width = 30)
  (h3 : piece_length = 3)
  (h4 : piece_width = 4) :
  (pan_length * pan_width) / (piece_length * piece_width) = 60 := by
  sorry

#check brownie_pieces

end NUMINAMATH_CALUDE_brownie_pieces_l1480_148065


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l1480_148073

theorem smallest_m_for_integral_solutions :
  let has_integral_solutions (m : ℤ) := ∃ x y : ℤ, 10 * x^2 - m * x + 780 = 0 ∧ 10 * y^2 - m * y + 780 = 0 ∧ x ≠ y
  ∀ m : ℤ, m > 0 → has_integral_solutions m → m ≥ 190 ∧
  has_integral_solutions 190 :=
by sorry


end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l1480_148073


namespace NUMINAMATH_CALUDE_five_letter_words_count_l1480_148095

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The number of consonants in the alphabet -/
def num_consonants : ℕ := 21

/-- The total number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of five-letter words starting with a vowel and ending with a consonant -/
def num_words : ℕ := num_vowels * num_letters * num_letters * num_letters * num_consonants

theorem five_letter_words_count : num_words = 1844760 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_count_l1480_148095


namespace NUMINAMATH_CALUDE_initial_number_proof_l1480_148009

theorem initial_number_proof (x : ℤ) : (x + 2)^2 = x^2 - 2016 → x = -505 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l1480_148009


namespace NUMINAMATH_CALUDE_chess_game_outcome_l1480_148090

theorem chess_game_outcome (prob_A_win prob_A_not_lose : ℝ) 
  (h1 : prob_A_win = 0.3)
  (h2 : prob_A_not_lose = 0.7) :
  let prob_draw := prob_A_not_lose - prob_A_win
  prob_draw > prob_A_win ∧ prob_draw > (1 - prob_A_not_lose) :=
by sorry

end NUMINAMATH_CALUDE_chess_game_outcome_l1480_148090


namespace NUMINAMATH_CALUDE_expand_and_simplify_product_l1480_148075

theorem expand_and_simplify_product (x : ℝ) : 
  (5 * x + 3) * (2 * x^2 - x + 4) = 10 * x^3 + x^2 + 17 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_product_l1480_148075


namespace NUMINAMATH_CALUDE_special_polynomial_max_value_l1480_148013

/-- A polynomial with integer coefficients satisfying certain conditions -/
def SpecialPolynomial (p : ℤ → ℤ) : Prop :=
  (∀ m n : ℤ, (p m - p n) ∣ (m^2 - n^2)) ∧ 
  p 0 = 1 ∧ 
  p 1 = 2

/-- The maximum value of p(100) for a SpecialPolynomial p is 10001 -/
theorem special_polynomial_max_value : 
  ∀ p : ℤ → ℤ, SpecialPolynomial p → p 100 ≤ 10001 :=
by sorry

end NUMINAMATH_CALUDE_special_polynomial_max_value_l1480_148013


namespace NUMINAMATH_CALUDE_arithmetic_cube_reciprocal_roots_l1480_148063

theorem arithmetic_cube_reciprocal_roots :
  (∀ x : ℝ, x ≥ 0 → Real.sqrt x = (abs x) ^ (1/2)) →
  (∀ x : ℝ, x > 0 → (x ^ (1/3)) ^ 3 = x) →
  (∀ x : ℝ, x ≠ 0 → x * (1/x) = 1) →
  (Real.sqrt ((-81)^2) = 9) ∧
  ((1/27) ^ (1/3) = 1/3) ∧
  (1 / Real.sqrt 2 = Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_cube_reciprocal_roots_l1480_148063


namespace NUMINAMATH_CALUDE_a_in_range_l1480_148066

/-- The function f(x) = -x^2 - 2ax -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 - 2*a*x

/-- The maximum value of f(x) on [0, 1] is a^2 -/
def max_value (a : ℝ) : Prop :=
  ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ f a x = a^2 ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 1 → f a y ≤ a^2

/-- If the maximum value of f(x) on [0, 1] is a^2, then a is in [-1, 0] -/
theorem a_in_range (a : ℝ) (h : max_value a) : a ∈ Set.Icc (-1) 0 := by
  sorry

end NUMINAMATH_CALUDE_a_in_range_l1480_148066


namespace NUMINAMATH_CALUDE_basketball_score_proof_l1480_148068

-- Define the scores for each quarter
def alpha_scores (a r : ℝ) : Fin 4 → ℝ
| 0 => a
| 1 => a * r
| 2 => a * r^2
| 3 => a * r^3

def beta_scores (b d : ℝ) : Fin 4 → ℝ
| 0 => b
| 1 => b + d
| 2 => b + 2*d
| 3 => b + 3*d

-- Define the theorem
theorem basketball_score_proof 
  (a r b d : ℝ) 
  (h1 : 0 < r) -- Ensure increasing geometric sequence
  (h2 : 0 < d) -- Ensure increasing arithmetic sequence
  (h3 : alpha_scores a r 0 + alpha_scores a r 1 = beta_scores b d 0 + beta_scores b d 1) -- Tied at second quarter
  (h4 : (alpha_scores a r 0 + alpha_scores a r 1 + alpha_scores a r 2 + alpha_scores a r 3) = 
        (beta_scores b d 0 + beta_scores b d 1 + beta_scores b d 2 + beta_scores b d 3) + 2) -- Alpha wins by 2
  (h5 : (alpha_scores a r 0 + alpha_scores a r 1 + alpha_scores a r 2 + alpha_scores a r 3) ≤ 100) -- Alpha's total ≤ 100
  (h6 : (beta_scores b d 0 + beta_scores b d 1 + beta_scores b d 2 + beta_scores b d 3) ≤ 100) -- Beta's total ≤ 100
  : (alpha_scores a r 0 + alpha_scores a r 1 + beta_scores b d 0 + beta_scores b d 1) = 24 :=
by sorry


end NUMINAMATH_CALUDE_basketball_score_proof_l1480_148068


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_ribbons_l1480_148046

/-- The lengths of Amanda's ribbons in inches -/
def ribbon_lengths : List ℕ := [8, 16, 20, 28]

/-- A function to check if a number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- The theorem stating that 2 is the largest prime that divides all ribbon lengths -/
theorem largest_prime_divisor_of_ribbons :
  ∃ (p : ℕ), is_prime p ∧ 
    (∀ (length : ℕ), length ∈ ribbon_lengths → p ∣ length) ∧
    (∀ (q : ℕ), is_prime q → 
      (∀ (length : ℕ), length ∈ ribbon_lengths → q ∣ length) → q ≤ p) ∧
    p = 2 :=
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_ribbons_l1480_148046


namespace NUMINAMATH_CALUDE_angle_measures_l1480_148041

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧ t.A + t.B + t.C = 180

-- Define the ratio condition
def ratio_condition (t : Triangle) : Prop :=
  t.A / t.B = 1 / 2 ∧ t.B / t.C = 2 / 3

-- Theorem statement
theorem angle_measures (t : Triangle) 
  (h1 : valid_triangle t) (h2 : ratio_condition t) : 
  t.A = 30 ∧ t.B = 60 ∧ t.C = 90 :=
sorry

end NUMINAMATH_CALUDE_angle_measures_l1480_148041


namespace NUMINAMATH_CALUDE_lawyer_percentage_l1480_148031

theorem lawyer_percentage (total_members : ℝ) (h1 : total_members > 0) :
  let women_percentage : ℝ := 0.80
  let woman_lawyer_prob : ℝ := 0.32
  let women_lawyers_percentage : ℝ := woman_lawyer_prob / women_percentage
  women_lawyers_percentage = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_lawyer_percentage_l1480_148031


namespace NUMINAMATH_CALUDE_conditions_necessary_not_sufficient_l1480_148057

theorem conditions_necessary_not_sufficient :
  (∀ x y : ℝ, (2 < x ∧ x < 3 ∧ 0 < y ∧ y < 1) → (2 < x + y ∧ x + y < 4 ∧ 0 < x * y ∧ x * y < 3)) ∧
  (∃ x y : ℝ, (2 < x + y ∧ x + y < 4 ∧ 0 < x * y ∧ x * y < 3) ∧ ¬(2 < x ∧ x < 3 ∧ 0 < y ∧ y < 1)) :=
by sorry

end NUMINAMATH_CALUDE_conditions_necessary_not_sufficient_l1480_148057


namespace NUMINAMATH_CALUDE_mod_seven_difference_l1480_148029

theorem mod_seven_difference (n : ℕ) : (47^824 - 25^824) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_seven_difference_l1480_148029


namespace NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l1480_148069

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := x = -3 * y^2 + 2 * y + 2

/-- An x-intercept is a point where the parabola crosses the x-axis (y = 0) -/
def is_x_intercept (x : ℝ) : Prop := parabola_equation x 0

/-- The theorem stating that the parabola has exactly one x-intercept -/
theorem parabola_has_one_x_intercept : ∃! x : ℝ, is_x_intercept x := by sorry

end NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l1480_148069


namespace NUMINAMATH_CALUDE_triangle_inequality_cube_root_l1480_148033

/-- Given a, b, c are side lengths of a triangle, 
    prove that ∛((a²+bc)(b²+ca)(c²+ab)) > (a²+b²+c²)/2 -/
theorem triangle_inequality_cube_root (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (((a^2 + b*c) * (b^2 + c*a) * (c^2 + a*b))^(1/3) : ℝ) > (a^2 + b^2 + c^2) / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_cube_root_l1480_148033


namespace NUMINAMATH_CALUDE_grape_juice_percentage_l1480_148084

/-- Calculates the percentage of grape juice in a mixture after adding pure grape juice --/
theorem grape_juice_percentage
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_grape_juice : ℝ)
  (h1 : initial_volume = 30)
  (h2 : initial_concentration = 0.1)
  (h3 : added_grape_juice = 10) :
  let final_volume := initial_volume + added_grape_juice
  let initial_grape_juice := initial_volume * initial_concentration
  let final_grape_juice := initial_grape_juice + added_grape_juice
  final_grape_juice / final_volume = 0.325 := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_percentage_l1480_148084


namespace NUMINAMATH_CALUDE_special_triangle_properties_l1480_148053

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Conditions
  side_relation : a = (1/2) * c + b * Real.cos C
  area : (1/2) * a * c * Real.sin ((1/3) * Real.pi) = Real.sqrt 3
  side_b : b = Real.sqrt 13

/-- Properties of the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) : 
  t.B = (1/3) * Real.pi ∧ t.a + t.c = 5 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l1480_148053


namespace NUMINAMATH_CALUDE_exists_four_digit_number_sum_12_div_5_l1480_148050

/-- A four-digit number is represented as a tuple of four natural numbers -/
def FourDigitNumber := (ℕ × ℕ × ℕ × ℕ)

/-- Check if a given four-digit number has digits that add up to 12 -/
def digits_sum_to_12 (n : FourDigitNumber) : Prop :=
  n.1 + n.2.1 + n.2.2.1 + n.2.2.2 = 12

/-- Check if a given four-digit number is divisible by 5 -/
def divisible_by_5 (n : FourDigitNumber) : Prop :=
  (n.1 * 1000 + n.2.1 * 100 + n.2.2.1 * 10 + n.2.2.2) % 5 = 0

/-- Check if a given number is a valid four-digit number (between 1000 and 9999) -/
def is_valid_four_digit (n : FourDigitNumber) : Prop :=
  n.1 ≠ 0 ∧ n.1 ≤ 9 ∧ n.2.1 ≤ 9 ∧ n.2.2.1 ≤ 9 ∧ n.2.2.2 ≤ 9

theorem exists_four_digit_number_sum_12_div_5 :
  ∃ (n : FourDigitNumber), is_valid_four_digit n ∧ digits_sum_to_12 n ∧ divisible_by_5 n :=
by
  sorry

end NUMINAMATH_CALUDE_exists_four_digit_number_sum_12_div_5_l1480_148050


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l1480_148021

/-- Represents the four types of crops -/
inductive Crop
| Corn
| Wheat
| Soybeans
| Potatoes

/-- Represents a position in the 3x3 grid -/
structure Position :=
  (row : Fin 3)
  (col : Fin 3)

/-- Checks if two positions are adjacent -/
def are_adjacent (p1 p2 : Position) : Bool :=
  (p1.row = p2.row ∧ (p1.col.val + 1 = p2.col.val ∨ p2.col.val + 1 = p1.col.val)) ∨
  (p1.col = p2.col ∧ (p1.row.val + 1 = p2.row.val ∨ p2.row.val + 1 = p1.row.val))

/-- Represents a planting arrangement -/
def Arrangement := Position → Crop

/-- Checks if an arrangement is valid according to the rules -/
def is_valid_arrangement (arr : Arrangement) : Prop :=
  ∀ p1 p2 : Position,
    are_adjacent p1 p2 →
      (arr p1 ≠ arr p2) ∧
      ¬(arr p1 = Crop.Corn ∧ arr p2 = Crop.Wheat) ∧
      ¬(arr p1 = Crop.Wheat ∧ arr p2 = Crop.Corn)

/-- The main theorem to be proved -/
theorem valid_arrangements_count :
  ∃ (arrangements : Finset Arrangement),
    (∀ arr ∈ arrangements, is_valid_arrangement arr) ∧
    (∀ arr, is_valid_arrangement arr → arr ∈ arrangements) ∧
    arrangements.card = 16 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l1480_148021


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1480_148047

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x + 2*a > 4 ∧ 2*x - b < 5) ↔ (0 < x ∧ x < 2)) →
  (a + b)^2023 = 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1480_148047


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_diff_l1480_148070

-- Define the polynomial
def f (x : ℝ) : ℝ := 20 * x^3 - 40 * x^2 + 24 * x - 2

-- State the theorem
theorem root_sum_reciprocal_diff (a b c : ℝ) :
  f a = 0 → f b = 0 → f c = 0 →  -- a, b, c are roots of f
  a ≠ b → b ≠ c → a ≠ c →        -- roots are distinct
  0 < a → a < 1 →                -- a is between 0 and 1
  0 < b → b < 1 →                -- b is between 0 and 1
  0 < c → c < 1 →                -- c is between 0 and 1
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_diff_l1480_148070


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1480_148037

/-- A circle with center (1,1) that is tangent to the line x + y = 4 -/
def TangentCircle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 2}

/-- The line x + y = 4 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 4}

theorem circle_tangent_to_line :
  (∃ (p : ℝ × ℝ), p ∈ TangentCircle ∧ p ∈ TangentLine) ∧
  (∀ (p : ℝ × ℝ), p ∈ TangentCircle → p ∈ TangentLine → 
    ∀ (q : ℝ × ℝ), q ∈ TangentCircle → q = p ∨ q ∉ TangentLine) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l1480_148037


namespace NUMINAMATH_CALUDE_inequality_proof_l1480_148055

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b) * (b + c) * (c + d) * (d + a) * (1 + (a * b * c * d) ^ (1/4)) ^ 4 ≥ 
  16 * a * b * c * d * (1 + a) * (1 + b) * (1 + c) * (1 + d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1480_148055


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1480_148020

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A a ∩ B a = {9} → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1480_148020


namespace NUMINAMATH_CALUDE_correct_number_of_selections_l1480_148001

/-- The number of volunteers who only speak Russian -/
def russian_only : ℕ := 3

/-- The number of volunteers who speak both Russian and English -/
def bilingual : ℕ := 4

/-- The total number of volunteers -/
def total_volunteers : ℕ := russian_only + bilingual

/-- The number of English translators to be selected -/
def english_translators : ℕ := 2

/-- The number of Russian translators to be selected -/
def russian_translators : ℕ := 2

/-- The total number of translators to be selected -/
def total_translators : ℕ := english_translators + russian_translators

/-- The function to calculate the number of ways to select translators -/
def num_ways_to_select_translators : ℕ := sorry

/-- Theorem stating that the number of ways to select translators is 60 -/
theorem correct_number_of_selections :
  num_ways_to_select_translators = 60 := by sorry

end NUMINAMATH_CALUDE_correct_number_of_selections_l1480_148001


namespace NUMINAMATH_CALUDE_upward_parabola_m_value_l1480_148039

/-- If y=(m-1)x^2-2mx+1 is an upward-opening parabola, then m = 2 -/
theorem upward_parabola_m_value (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 - 2 * m * x + 1 = 0 → (m - 1) > 0) → 
  m = 2 := by sorry

end NUMINAMATH_CALUDE_upward_parabola_m_value_l1480_148039


namespace NUMINAMATH_CALUDE_card_problem_l1480_148038

/-- Given the number of cards for Brenda, Janet, and Mara, calculate the certain number -/
def certainNumber (brenda : ℕ) (janet : ℕ) (mara : ℕ) : ℕ :=
  mara + 40

theorem card_problem (brenda : ℕ) :
  let janet := brenda + 9
  let mara := 2 * janet
  brenda + janet + mara = 211 →
  certainNumber brenda janet mara = 150 := by
  sorry

#check card_problem

end NUMINAMATH_CALUDE_card_problem_l1480_148038


namespace NUMINAMATH_CALUDE_three_geometric_sequences_l1480_148048

/-- An arithmetic sequence starting with 1 -/
structure ArithmeticSequence :=
  (d : ℝ)
  (a₁ : ℝ := 1 + d)
  (a₂ : ℝ := 1 + 2*d)
  (a₃ : ℝ := 1 + 3*d)
  (positive : 0 < a₁ ∧ 0 < a₂ ∧ 0 < a₃)

/-- A function that counts the number of geometric sequences that can be formed from 1 and the terms of an arithmetic sequence -/
def countGeometricSequences (seq : ArithmeticSequence) : ℕ :=
  sorry

/-- The main theorem stating that there are exactly 3 geometric sequences -/
theorem three_geometric_sequences (seq : ArithmeticSequence) : 
  countGeometricSequences seq = 3 :=
sorry

end NUMINAMATH_CALUDE_three_geometric_sequences_l1480_148048


namespace NUMINAMATH_CALUDE_gcd_324_243_135_l1480_148036

theorem gcd_324_243_135 : Nat.gcd 324 (Nat.gcd 243 135) = 27 := by sorry

end NUMINAMATH_CALUDE_gcd_324_243_135_l1480_148036


namespace NUMINAMATH_CALUDE_tommy_gum_pieces_l1480_148003

/-- 
Given:
- Maria initially had 25 pieces of gum
- Luis gave Maria 20 pieces of gum
- Maria now has 61 pieces of gum in total

Prove that Tommy gave Maria 16 pieces of gum
-/
theorem tommy_gum_pieces (initial : Nat) (from_luis : Nat) (total : Nat) :
  initial = 25 →
  from_luis = 20 →
  total = 61 →
  total - (initial + from_luis) = 16 := by
  sorry

end NUMINAMATH_CALUDE_tommy_gum_pieces_l1480_148003


namespace NUMINAMATH_CALUDE_digital_earth_data_source_is_high_speed_network_databases_l1480_148014

/-- Represents the possible sources of spatial data for the digital Earth -/
inductive SpatialDataSource
  | SatelliteRemoteSensing
  | HighSpeedNetworkDatabases
  | InformationHighway
  | GISExchangeData

/-- Represents the digital Earth -/
structure DigitalEarth where
  mainDataSource : SpatialDataSource

/-- Axiom: The main source of basic spatial data for the digital Earth is from high-speed network databases -/
axiom digital_earth_main_data_source :
  ∀ (de : DigitalEarth), de.mainDataSource = SpatialDataSource.HighSpeedNetworkDatabases

/-- Theorem: The main source of basic spatial data for the digital Earth is from high-speed network databases -/
theorem digital_earth_data_source_is_high_speed_network_databases (de : DigitalEarth) :
  de.mainDataSource = SpatialDataSource.HighSpeedNetworkDatabases :=
by sorry

end NUMINAMATH_CALUDE_digital_earth_data_source_is_high_speed_network_databases_l1480_148014


namespace NUMINAMATH_CALUDE_trivia_team_score_l1480_148093

theorem trivia_team_score (total_members : ℕ) (absent_members : ℕ) (total_points : ℕ) :
  total_members = 14 →
  absent_members = 7 →
  total_points = 35 →
  (total_points / (total_members - absent_members) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_score_l1480_148093


namespace NUMINAMATH_CALUDE_negative_reverses_inequality_l1480_148000

theorem negative_reverses_inequality (a b : ℝ) (h : a > b) : -a < -b := by
  sorry

end NUMINAMATH_CALUDE_negative_reverses_inequality_l1480_148000


namespace NUMINAMATH_CALUDE_solve_equation_l1480_148024

theorem solve_equation (x : ℚ) : (4/7 : ℚ) * (1/5 : ℚ) * x = 12 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1480_148024


namespace NUMINAMATH_CALUDE_die_product_divisibility_l1480_148051

theorem die_product_divisibility : 
  ∀ (S : Finset ℕ), 
  S ⊆ Finset.range 9 → 
  S.card = 7 → 
  48 ∣ S.prod id := by
sorry

end NUMINAMATH_CALUDE_die_product_divisibility_l1480_148051


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_17_mod_26_l1480_148006

theorem largest_five_digit_congruent_17_mod_26 : ∃ (n : ℕ), n = 99997 ∧ 
  n < 100000 ∧ 
  n % 26 = 17 ∧ 
  ∀ (m : ℕ), m < 100000 → m % 26 = 17 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_17_mod_26_l1480_148006


namespace NUMINAMATH_CALUDE_arrangements_eight_athletes_three_consecutive_l1480_148076

/-- The number of tracks and athletes -/
def n : ℕ := 8

/-- The number of specified athletes that must be in consecutive tracks -/
def k : ℕ := 3

/-- The number of ways to arrange n athletes on n tracks, 
    where k specified athletes must be in consecutive tracks -/
def arrangements (n k : ℕ) : ℕ := sorry

/-- Theorem stating the correct number of arrangements for the given problem -/
theorem arrangements_eight_athletes_three_consecutive : 
  arrangements n k = 4320 := by sorry

end NUMINAMATH_CALUDE_arrangements_eight_athletes_three_consecutive_l1480_148076


namespace NUMINAMATH_CALUDE_line_vector_at_negative_two_l1480_148056

def line_vector (s : ℝ) : ℝ × ℝ := sorry

theorem line_vector_at_negative_two :
  line_vector 1 = (2, 5) →
  line_vector 4 = (8, -7) →
  line_vector (-2) = (-4, 17) := by sorry

end NUMINAMATH_CALUDE_line_vector_at_negative_two_l1480_148056


namespace NUMINAMATH_CALUDE_remainder_of_n_squared_plus_2n_plus_3_l1480_148092

theorem remainder_of_n_squared_plus_2n_plus_3 (n : ℤ) (k : ℤ) (h : n = 100 * k - 1) :
  (n^2 + 2*n + 3) % 100 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_n_squared_plus_2n_plus_3_l1480_148092


namespace NUMINAMATH_CALUDE_monomial_properties_l1480_148059

-- Define the monomial structure
structure Monomial (α : Type*) [Ring α] where
  coeff : α
  vars : List (Nat × Nat)

-- Define the monomial -2x^2y
def monomial : Monomial ℤ :=
  { coeff := -2,
    vars := [(1, 2), (2, 1)] }  -- Representing x^2 and y^1

-- Theorem statement
theorem monomial_properties :
  (monomial.coeff = -2) ∧
  (List.sum (monomial.vars.map (λ (_, exp) => exp)) = 3) :=
by sorry

end NUMINAMATH_CALUDE_monomial_properties_l1480_148059


namespace NUMINAMATH_CALUDE_power_calculation_l1480_148071

theorem power_calculation : (8^5 / 8^3) * 3^6 = 46656 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1480_148071


namespace NUMINAMATH_CALUDE_mountain_climb_speed_l1480_148028

-- Define the parameters
def total_time : ℝ := 20
def total_distance : ℝ := 80

-- Define the variables
variable (v : ℝ)  -- Speed on the first day
variable (t : ℝ)  -- Time spent on the first day

-- Define the theorem
theorem mountain_climb_speed :
  -- Conditions
  (t + (t - 2) + (t + 1) = total_time) →
  (v * t + (v + 0.5) * (t - 2) + (v - 0.5) * (t + 1) = total_distance) →
  -- Conclusion
  (v + 0.5 = 4.575) :=
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_mountain_climb_speed_l1480_148028


namespace NUMINAMATH_CALUDE_chord_length_line_circle_intersection_l1480_148010

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length_line_circle_intersection : 
  ∃ (A B : ℝ × ℝ),
    (A.1 + A.2 = 2) ∧ 
    (B.1 + B.2 = 2) ∧ 
    (A.1^2 + A.2^2 = 4) ∧ 
    (B.1^2 + B.2^2 = 4) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_line_circle_intersection_l1480_148010


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l1480_148091

/-- Given a cubic function f and two points (a, f(a)) and (b, f(b)), prove that a + b = -2 --/
theorem sum_of_roots_cubic (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = x^3 + 3*x^2 + 6*x) →
  f a = 1 →
  f b = -9 →
  a + b = -2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l1480_148091


namespace NUMINAMATH_CALUDE_salary_change_percentage_l1480_148025

theorem salary_change_percentage (x : ℝ) : 
  (1 - (x / 100)^2) = 0.91 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l1480_148025


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1480_148043

theorem complex_fraction_simplification : 
  (((12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324)) / 
   ((6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324))) = 221 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1480_148043


namespace NUMINAMATH_CALUDE_exponent_division_l1480_148042

theorem exponent_division (a : ℝ) : a^7 / a^4 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1480_148042


namespace NUMINAMATH_CALUDE_pencils_added_l1480_148052

theorem pencils_added (initial_pencils final_pencils : ℕ) 
  (h1 : initial_pencils = 41)
  (h2 : final_pencils = 71) :
  final_pencils - initial_pencils = 30 := by
  sorry

end NUMINAMATH_CALUDE_pencils_added_l1480_148052


namespace NUMINAMATH_CALUDE_negation_of_existential_negation_of_sqrt_3_rational_l1480_148004

theorem negation_of_existential (P : ℚ → Prop) :
  (¬ ∃ x : ℚ, P x) ↔ (∀ x : ℚ, ¬ P x) :=
by sorry

theorem negation_of_sqrt_3_rational :
  (¬ ∃ x : ℚ, x^2 = 3) ↔ (∀ x : ℚ, x^2 ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_negation_of_sqrt_3_rational_l1480_148004


namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_l1480_148081

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- Define the relation for a line being contained in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (plane_intersection : Plane → Plane → Line)

-- Theorem statement
theorem perpendicular_line_to_plane 
  (α β : Plane) (l m : Line) 
  (h1 : perp_planes α β)
  (h2 : plane_intersection α β = l)
  (h3 : line_in_plane m α)
  (h4 : perp_lines m l) :
  perp_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_l1480_148081


namespace NUMINAMATH_CALUDE_probability_of_two_positive_roots_l1480_148094

-- Define the interval for a
def interval : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - 2*a*x + 4*a - 3

-- Define the condition for two positive roots
def has_two_positive_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
  quadratic a x₁ = 0 ∧ quadratic a x₂ = 0

-- Define the probability measure on the interval
noncomputable def probability_measure : MeasureTheory.Measure ℝ :=
  sorry

-- State the theorem
theorem probability_of_two_positive_roots :
  probability_measure {a ∈ interval | has_two_positive_roots a} = 3/8 :=
sorry

end NUMINAMATH_CALUDE_probability_of_two_positive_roots_l1480_148094


namespace NUMINAMATH_CALUDE_shaded_area_in_rectangle_with_circles_l1480_148026

/-- Given a rectangle containing two tangent circles, calculate the area not occupied by the circles. -/
theorem shaded_area_in_rectangle_with_circles 
  (rectangle_length : ℝ) 
  (rectangle_height : ℝ)
  (small_circle_radius : ℝ)
  (large_circle_radius : ℝ) :
  rectangle_length = 20 →
  rectangle_height = 10 →
  small_circle_radius = 3 →
  large_circle_radius = 5 →
  ∃ (shaded_area : ℝ), 
    shaded_area = rectangle_length * rectangle_height - π * (small_circle_radius^2 + large_circle_radius^2) ∧
    shaded_area = 200 - 34 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_in_rectangle_with_circles_l1480_148026


namespace NUMINAMATH_CALUDE_line_x_intercept_l1480_148061

/-- Given a line passing through points (10, 3) and (-4, -4), 
    prove that its x-intercept is 4 -/
theorem line_x_intercept : 
  let p1 : ℝ × ℝ := (10, 3)
  let p2 : ℝ × ℝ := (-4, -4)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  (0 : ℝ) = m * 4 + b :=
by sorry

end NUMINAMATH_CALUDE_line_x_intercept_l1480_148061


namespace NUMINAMATH_CALUDE_alice_needs_1615_stamps_l1480_148018

def stamps_problem (alice ernie peggy danny bert : ℕ) : Prop :=
  alice = 65 ∧
  danny = alice + 5 ∧
  peggy = 2 * danny ∧
  ernie = 3 * peggy ∧
  bert = 4 * ernie

theorem alice_needs_1615_stamps 
  (alice ernie peggy danny bert : ℕ) 
  (h : stamps_problem alice ernie peggy danny bert) : 
  bert - alice = 1615 := by
sorry

end NUMINAMATH_CALUDE_alice_needs_1615_stamps_l1480_148018


namespace NUMINAMATH_CALUDE_integral_approximation_l1480_148079

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (hf_continuous : ContinuousOn f (Set.Icc 0 1))
variable (hf_range : ∀ x ∈ Set.Icc 0 1, 0 ≤ f x ∧ f x ≤ 1)

-- Define N and N_1
variable (N N_1 : ℕ)

-- Define the theorem
theorem integral_approximation :
  ∃ ε > 0, |∫ x in Set.Icc 0 1, f x - (N_1 : ℝ) / N| < ε :=
sorry

end NUMINAMATH_CALUDE_integral_approximation_l1480_148079


namespace NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l1480_148060

theorem midpoint_sum_equals_vertex_sum (a b c d : ℝ) :
  let vertices_sum := a + b + c + d
  let midpoints_sum := (a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2
  midpoints_sum = vertices_sum :=
by sorry

end NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l1480_148060


namespace NUMINAMATH_CALUDE_sqrt_of_negative_six_squared_l1480_148034

theorem sqrt_of_negative_six_squared (x : ℝ) : Real.sqrt ((-6)^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_negative_six_squared_l1480_148034


namespace NUMINAMATH_CALUDE_complex_sum_power_l1480_148027

theorem complex_sum_power (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) :
  (x / (x + y))^2013 + (y / (x + y))^2013 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_power_l1480_148027


namespace NUMINAMATH_CALUDE_jasons_commute_distance_l1480_148080

/-- Jason's commute to work problem -/
theorem jasons_commute_distance : ∀ (d1 d2 d3 d4 d5 : ℝ),
  d1 = 6 →                           -- Distance between first and second store
  d2 = d1 + (2/3 * d1) →             -- Distance between second and third store
  d3 = 4 →                           -- Distance from house to first store
  d4 = 4 →                           -- Distance from last store to work
  d5 = d1 + d2 + d3 + d4 →           -- Total commute distance
  d5 = 24 := by sorry

end NUMINAMATH_CALUDE_jasons_commute_distance_l1480_148080


namespace NUMINAMATH_CALUDE_y₁_greater_than_y₂_l1480_148017

-- Define the line
def line (x : ℝ) (b : ℝ) : ℝ := -2023 * x + b

-- Define the points A and B
def point_A (y₁ : ℝ) : ℝ × ℝ := (-2, y₁)
def point_B (y₂ : ℝ) : ℝ × ℝ := (-1, y₂)

-- Theorem statement
theorem y₁_greater_than_y₂ (b y₁ y₂ : ℝ) 
  (h₁ : line (-2) b = y₁)
  (h₂ : line (-1) b = y₂) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_greater_than_y₂_l1480_148017


namespace NUMINAMATH_CALUDE_max_m_value_l1480_148011

def f (x : ℝ) : ℝ := |2*x + 1| + |3*x - 2|

theorem max_m_value (h : Set.Icc (-4/5 : ℝ) (6/5) = {x : ℝ | f x ≤ 5}) :
  ∃ m : ℝ, m = 2 ∧ 
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ m^2 - 3*m + 5) ∧
  (∀ m' : ℝ, m' > m → ∃ x : ℝ, |x - 1| + |x + 2| < m'^2 - 3*m' + 5) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l1480_148011


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_75_l1480_148088

theorem distinct_prime_factors_of_75 : Nat.card (Nat.factors 75).toFinset = 2 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_75_l1480_148088


namespace NUMINAMATH_CALUDE_oreo_cheesecake_problem_l1480_148089

theorem oreo_cheesecake_problem (graham_boxes_initial : ℕ) (graham_boxes_per_cake : ℕ) (oreo_packets_per_cake : ℕ) (graham_boxes_leftover : ℕ) :
  graham_boxes_initial = 14 →
  graham_boxes_per_cake = 2 →
  oreo_packets_per_cake = 3 →
  graham_boxes_leftover = 4 →
  let cakes_made := (graham_boxes_initial - graham_boxes_leftover) / graham_boxes_per_cake
  ∃ oreo_packets_bought : ℕ, oreo_packets_bought = cakes_made * oreo_packets_per_cake :=
by sorry

end NUMINAMATH_CALUDE_oreo_cheesecake_problem_l1480_148089


namespace NUMINAMATH_CALUDE_probability_no_player_wins_all_is_11_16_l1480_148058

def num_players : Nat := 5

def num_games : Nat := (num_players * (num_players - 1)) / 2

def probability_no_player_wins_all : Rat :=
  1 - (num_players * (1 / 2 ^ (num_players - 1))) / (2 ^ num_games)

theorem probability_no_player_wins_all_is_11_16 :
  probability_no_player_wins_all = 11 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_player_wins_all_is_11_16_l1480_148058


namespace NUMINAMATH_CALUDE_solution_set_absolute_value_equation_l1480_148062

theorem solution_set_absolute_value_equation (x : ℝ) :
  |1 - x| + |2*x - 1| = |3*x - 2| ↔ x ≤ 1/2 ∨ x ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_absolute_value_equation_l1480_148062


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l1480_148005

theorem initial_markup_percentage 
  (C : ℝ) 
  (M : ℝ) 
  (h1 : C > 0) 
  (h2 : (C * (1 + M) * 1.25 * 0.75) = (C * 1.125)) : 
  M = 0.2 := by
sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l1480_148005


namespace NUMINAMATH_CALUDE_rational_sqrt_one_minus_xy_l1480_148087

theorem rational_sqrt_one_minus_xy (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ (q : ℚ), q^2 = 1 - x*y := by
  sorry

end NUMINAMATH_CALUDE_rational_sqrt_one_minus_xy_l1480_148087


namespace NUMINAMATH_CALUDE_absolute_value_even_and_increasing_l1480_148016

def f (x : ℝ) := abs x

theorem absolute_value_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_even_and_increasing_l1480_148016


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l1480_148078

theorem rectangular_prism_width 
  (l h d : ℝ) 
  (hl : l = 4) 
  (hh : h = 10) 
  (hd : d = 14) : 
  ∃ w : ℝ, w = 4 * Real.sqrt 5 ∧ l^2 + w^2 + h^2 = d^2 :=
sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l1480_148078


namespace NUMINAMATH_CALUDE_second_class_average_l1480_148072

/-- Given two classes of students, this theorem proves the average mark of the second class. -/
theorem second_class_average (students1 : ℕ) (students2 : ℕ) (avg1 : ℝ) (combined_avg : ℝ) 
  (h1 : students1 = 58)
  (h2 : students2 = 52)
  (h3 : avg1 = 67)
  (h4 : combined_avg = 74.0909090909091) : 
  ∃ (avg2 : ℝ), abs (avg2 - 81.62) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_second_class_average_l1480_148072


namespace NUMINAMATH_CALUDE_no_perfect_square_pairs_l1480_148077

theorem no_perfect_square_pairs : ¬∃ (x y : ℕ+), ∃ (z : ℕ+), (x * y + 1) * (x * y + x + 2) = z ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_pairs_l1480_148077


namespace NUMINAMATH_CALUDE_solve_for_y_l1480_148008

theorem solve_for_y (x y : ℝ) (h : 3 * x + 2 * y = 1) : y = (1 - 3 * x) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1480_148008


namespace NUMINAMATH_CALUDE_paper_string_area_l1480_148098

/-- The area of a paper string made from overlapping square sheets -/
theorem paper_string_area
  (num_sheets : ℕ)
  (sheet_side : ℝ)
  (overlap : ℝ)
  (h_num_sheets : num_sheets = 6)
  (h_sheet_side : sheet_side = 30)
  (h_overlap : overlap = 7) :
  (sheet_side + (num_sheets - 1) * (sheet_side - overlap)) * sheet_side = 4350 :=
sorry

end NUMINAMATH_CALUDE_paper_string_area_l1480_148098


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_odd_l1480_148074

def isOdd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

def inRange (n : ℕ) : Prop := 5 ≤ n ∧ n ≤ 12

theorem sum_of_largest_and_smallest_odd : 
  ∃ (a b : ℕ), 
    isOdd a ∧ isOdd b ∧ 
    inRange a ∧ inRange b ∧
    (∀ x, isOdd x ∧ inRange x → a ≤ x ∧ x ≤ b) ∧
    a + b = 16 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_odd_l1480_148074


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1480_148086

/-- A line that always passes through a fixed point regardless of the parameter m -/
def always_passes_through (m : ℝ) : Prop :=
  (m - 1) * (-3) + (2 * m - 3) * 1 + m = 0

/-- The theorem stating that the line passes through (-3, 1) for all real m -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, always_passes_through m :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1480_148086


namespace NUMINAMATH_CALUDE_total_shaded_area_l1480_148067

/-- Represents the fraction of area shaded at each level of division -/
def shaded_fraction : ℚ := 1 / 4

/-- Represents the ratio between successive terms in the geometric series -/
def common_ratio : ℚ := 1 / 16

/-- Theorem stating that the total shaded area is 4/15 -/
theorem total_shaded_area :
  (shaded_fraction / (1 - common_ratio) : ℚ) = 4 / 15 := by sorry

end NUMINAMATH_CALUDE_total_shaded_area_l1480_148067


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1480_148099

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_3 : a + b + c = 3) :
  (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≥ (3 / 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1480_148099


namespace NUMINAMATH_CALUDE_stable_performance_lower_variance_l1480_148096

/-- Represents an athlete's shooting performance -/
structure Athlete where
  average_score : ℝ
  variance : ℝ
  sessions : ℕ

/-- Defines stability of performance based on variance -/
def more_stable (a b : Athlete) : Prop :=
  a.variance < b.variance

theorem stable_performance_lower_variance 
  (a b : Athlete) 
  (h1 : a.average_score = b.average_score) 
  (h2 : a.sessions = b.sessions) 
  (h3 : a.sessions > 0) 
  (h4 : a.variance < b.variance) : 
  more_stable a b :=
sorry

end NUMINAMATH_CALUDE_stable_performance_lower_variance_l1480_148096


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1480_148023

theorem simplify_sqrt_expression :
  (Real.sqrt 450 / Real.sqrt 288) + (Real.sqrt 245 / Real.sqrt 96) = (30 + 7 * Real.sqrt 30) / 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1480_148023


namespace NUMINAMATH_CALUDE_probability_is_correct_l1480_148044

def total_stickers : ℕ := 18
def selected_stickers : ℕ := 10
def needed_stickers : ℕ := 6
def collected_stickers : ℕ := total_stickers - needed_stickers

def probability_complete_collection : ℚ :=
  (Nat.choose needed_stickers needed_stickers * Nat.choose collected_stickers (selected_stickers - needed_stickers)) /
  Nat.choose total_stickers selected_stickers

theorem probability_is_correct : probability_complete_collection = 5 / 442 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_correct_l1480_148044


namespace NUMINAMATH_CALUDE_lottery_probability_l1480_148083

theorem lottery_probability (winning_rate : ℚ) (num_tickets : ℕ) : 
  winning_rate = 1/3 → num_tickets = 3 → 
  (1 - (1 - winning_rate) ^ num_tickets) = 19/27 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_l1480_148083


namespace NUMINAMATH_CALUDE_circle_diameter_theorem_l1480_148035

/-- A circle with two intersecting perpendicular chords -/
structure CircleWithChords where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The lengths of the segments of the first chord -/
  chord1_seg1 : ℝ
  chord1_seg2 : ℝ
  /-- The lengths of the segments of the second chord -/
  chord2_seg1 : ℝ
  chord2_seg2 : ℝ
  /-- The chords are perpendicular -/
  chords_perpendicular : True
  /-- The product of segments of each chord equals the square of the radius -/
  chord1_property : chord1_seg1 * chord1_seg2 = radius ^ 2
  chord2_property : chord2_seg1 * chord2_seg2 = radius ^ 2

/-- The theorem to be proved -/
theorem circle_diameter_theorem (c : CircleWithChords) 
  (h1 : c.chord1_seg1 = 3 ∧ c.chord1_seg2 = 4) 
  (h2 : c.chord2_seg1 = 6 ∧ c.chord2_seg2 = 2) : 
  2 * c.radius = 4 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_theorem_l1480_148035


namespace NUMINAMATH_CALUDE_baseball_league_games_l1480_148097

theorem baseball_league_games (N M : ℕ) : 
  (N > 2 * M) → 
  (M > 4) → 
  (4 * N + 5 * M = 94) → 
  (4 * N = 64) := by
sorry

end NUMINAMATH_CALUDE_baseball_league_games_l1480_148097
