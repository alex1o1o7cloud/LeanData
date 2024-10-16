import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_function_value_l329_32979

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem: If f(1) = 7, f(2) = 12, and c = 3, then f(3) = 18 -/
theorem quadratic_function_value (a b : ℝ) 
  (h1 : f a b 3 1 = 7)
  (h2 : f a b 3 2 = 12) :
  f a b 3 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l329_32979


namespace NUMINAMATH_CALUDE_logistics_personnel_in_sample_l329_32957

theorem logistics_personnel_in_sample
  (total_staff : ℕ)
  (logistics_staff : ℕ)
  (sample_size : ℕ)
  (h1 : total_staff = 160)
  (h2 : logistics_staff = 24)
  (h3 : sample_size = 20) :
  (logistics_staff : ℚ) / (total_staff : ℚ) * (sample_size : ℚ) = 3 :=
by sorry

end NUMINAMATH_CALUDE_logistics_personnel_in_sample_l329_32957


namespace NUMINAMATH_CALUDE_wallet_cost_l329_32978

theorem wallet_cost (wallet_cost purse_cost : ℝ) : 
  purse_cost = 4 * wallet_cost - 3 →
  wallet_cost + purse_cost = 107 →
  wallet_cost = 22 := by
sorry

end NUMINAMATH_CALUDE_wallet_cost_l329_32978


namespace NUMINAMATH_CALUDE_volleyball_match_probability_l329_32973

-- Define the probability of Team A winning a single game
def p_win_game : ℚ := 2/3

-- Define the probability of Team A winning the match
def p_win_match : ℚ := 20/27

-- Theorem statement
theorem volleyball_match_probability :
  (p_win_game = 2/3) →  -- Probability of Team A winning a single game
  (p_win_match = p_win_game * p_win_game + 2 * p_win_game * (1 - p_win_game) * p_win_game) :=
by
  sorry

#check volleyball_match_probability

end NUMINAMATH_CALUDE_volleyball_match_probability_l329_32973


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l329_32925

theorem complementary_angles_difference (a b : ℝ) (h1 : a + b = 90) (h2 : a / b = 5 / 4) :
  |a - b| = 10 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l329_32925


namespace NUMINAMATH_CALUDE_candy_bar_cost_l329_32923

theorem candy_bar_cost (candy_bars : ℕ) (lollipops : ℕ) (lollipop_cost : ℚ)
  (snow_shoveling_fraction : ℚ) (driveway_charge : ℚ) (driveways : ℕ) :
  candy_bars = 2 →
  lollipops = 4 →
  lollipop_cost = 1/4 →
  snow_shoveling_fraction = 1/6 →
  driveway_charge = 3/2 →
  driveways = 10 →
  (driveway_charge * driveways * snow_shoveling_fraction - lollipops * lollipop_cost) / candy_bars = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l329_32923


namespace NUMINAMATH_CALUDE_units_digit_of_p_plus_two_l329_32918

def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

def has_positive_units_digit (n : ℕ) : Prop := n % 10 > 0

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_p_plus_two (p : ℕ) 
  (h1 : is_positive_even p)
  (h2 : has_positive_units_digit p)
  (h3 : units_digit (p^3) - units_digit (p^2) = 0) :
  units_digit (p + 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_p_plus_two_l329_32918


namespace NUMINAMATH_CALUDE_intersection_one_element_l329_32959

theorem intersection_one_element (a : ℝ) : 
  let A : Set ℝ := {1, a, 5}
  let B : Set ℝ := {2, a^2 + 1}
  (∃! x, x ∈ A ∩ B) → a = 0 ∨ a = -2 := by
sorry

end NUMINAMATH_CALUDE_intersection_one_element_l329_32959


namespace NUMINAMATH_CALUDE_additional_cost_proof_l329_32928

/-- Additional cost per international letter --/
def additional_cost_per_letter : ℚ := 55 / 100

/-- Number of letters --/
def num_letters : ℕ := 4

/-- Number of domestic letters --/
def num_domestic : ℕ := 2

/-- Number of international letters --/
def num_international : ℕ := 2

/-- Domestic postage rate per letter --/
def domestic_rate : ℚ := 108 / 100

/-- Weight of first international letter (in grams) --/
def weight_letter1 : ℕ := 25

/-- Weight of second international letter (in grams) --/
def weight_letter2 : ℕ := 45

/-- Rate for Country A for letters below 50 grams (per gram) --/
def rate_A_below50 : ℚ := 5 / 100

/-- Rate for Country B for letters below 50 grams (per gram) --/
def rate_B_below50 : ℚ := 4 / 100

/-- Total postage paid --/
def total_paid : ℚ := 630 / 100

theorem additional_cost_proof :
  let domestic_cost := num_domestic * domestic_rate
  let international_cost1 := weight_letter1 * rate_A_below50
  let international_cost2 := weight_letter2 * rate_B_below50
  let total_calculated := domestic_cost + international_cost1 + international_cost2
  let additional_total := total_paid - total_calculated
  additional_total / num_international = additional_cost_per_letter := by
  sorry

end NUMINAMATH_CALUDE_additional_cost_proof_l329_32928


namespace NUMINAMATH_CALUDE_password_generation_l329_32996

def polynomial (x y : ℤ) : ℤ := 32 * x^3 - 8 * x * y^2

def factor1 (x : ℤ) : ℤ := 8 * x
def factor2 (x y : ℤ) : ℤ := 2 * x + y
def factor3 (x y : ℤ) : ℤ := 2 * x - y

def concatenate (a b c : ℤ) : ℤ := a * 100000 + b * 1000 + c

theorem password_generation (x y : ℤ) (h1 : x = 10) (h2 : y = 10) :
  concatenate (factor1 x) (factor2 x y) (factor3 x y) = 803010 :=
by sorry

end NUMINAMATH_CALUDE_password_generation_l329_32996


namespace NUMINAMATH_CALUDE_no_four_digit_number_equals_46_10X_plus_Y_l329_32991

theorem no_four_digit_number_equals_46_10X_plus_Y :
  ¬ ∃ (X Y : ℕ) (a b c d : ℕ),
    (a = 4 ∨ a = 6 ∨ a = X ∨ a = Y) ∧
    (b = 4 ∨ b = 6 ∨ b = X ∨ b = Y) ∧
    (c = 4 ∨ c = 6 ∨ c = X ∨ c = Y) ∧
    (d = 4 ∨ d = 6 ∨ d = X ∨ d = Y) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    1000 ≤ 1000 * a + 100 * b + 10 * c + d ∧
    1000 * a + 100 * b + 10 * c + d < 10000 ∧
    1000 * a + 100 * b + 10 * c + d = 46 * (10 * X + Y) :=
by sorry

end NUMINAMATH_CALUDE_no_four_digit_number_equals_46_10X_plus_Y_l329_32991


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l329_32931

/-- Given a geometric sequence with 7 terms, where the first term is 8 and the last term is 5832,
    prove that the fifth term is 648. -/
theorem fifth_term_of_geometric_sequence (a : Fin 7 → ℝ) :
  (∀ i j, a (i + 1) / a i = a (j + 1) / a j) →  -- geometric sequence condition
  a 0 = 8 →                                     -- first term is 8
  a 6 = 5832 →                                  -- last term is 5832
  a 4 = 648 := by                               -- fifth term (index 4) is 648
sorry


end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l329_32931


namespace NUMINAMATH_CALUDE_lcm_problem_l329_32951

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l329_32951


namespace NUMINAMATH_CALUDE_problem_statement_l329_32921

theorem problem_statement (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4*b^2 = 1/(a*b) + 3) :
  (ab ≤ 1) ∧ (b > a → 1/a^3 - 1/b^3 > 3*(1/a - 1/b)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l329_32921


namespace NUMINAMATH_CALUDE_whale_population_growth_l329_32988

/-- Proves that given the conditions of whale population growth, 
    the initial number of whales was 4000 -/
theorem whale_population_growth (w : ℕ) 
  (h1 : 2 * w = w + w)  -- The number of whales doubles each year
  (h2 : 2 * (2 * w) + 800 = 8800)  -- Prediction for third year
  : w = 4000 := by
  sorry

end NUMINAMATH_CALUDE_whale_population_growth_l329_32988


namespace NUMINAMATH_CALUDE_student_rank_from_right_l329_32995

/-- Given a student ranked 8th from the left in a group of 20 students, 
    their rank from the right is 13th. -/
theorem student_rank_from_right 
  (total_students : ℕ) 
  (rank_from_left : ℕ) 
  (h1 : total_students = 20) 
  (h2 : rank_from_left = 8) : 
  total_students - (rank_from_left - 1) = 13 := by
sorry

end NUMINAMATH_CALUDE_student_rank_from_right_l329_32995


namespace NUMINAMATH_CALUDE_one_fifth_of_seven_times_nine_l329_32945

theorem one_fifth_of_seven_times_nine : (1 / 5 : ℚ) * (7 * 9) = 12.6 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_seven_times_nine_l329_32945


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l329_32986

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  m : ℝ
  equation : (x : ℝ) → (y : ℝ) → Prop :=
    fun x y => x^2 / (m + 1) - y^2 / (3 - m) = 1

/-- Theorem statement for the hyperbola eccentricity problem -/
theorem hyperbola_eccentricity_range 
  (C : Hyperbola) 
  (F : Point) 
  (k : ℝ) 
  (A B P Q : Point) 
  (h1 : F.x < 0) -- F is the left focus
  (h2 : k ≥ Real.sqrt 3) -- Line slope condition
  (h3 : C.equation A.x A.y ∧ C.equation B.x B.y) -- A and B are on the hyperbola
  (h4 : P.x = (A.x + F.x) / 2 ∧ P.y = (A.y + F.y) / 2) -- P is midpoint of AF
  (h5 : Q.x = (B.x + F.x) / 2 ∧ Q.y = (B.y + F.y) / 2) -- Q is midpoint of BF
  (h6 : (P.y - 0) * (Q.y - 0) = -(P.x - 0) * (Q.x - 0)) -- OP ⊥ OQ
  : ∃ (e : ℝ), e ≥ Real.sqrt 3 + 1 ∧ 
    ∀ (e' : ℝ), e' ≥ Real.sqrt 3 + 1 → 
    ∃ (C' : Hyperbola), C'.m = C.m ∧ 
    (∃ (F' A' B' P' Q' : Point) (k' : ℝ), 
      F'.x < 0 ∧ 
      k' ≥ Real.sqrt 3 ∧
      C'.equation A'.x A'.y ∧ C'.equation B'.x B'.y ∧
      P'.x = (A'.x + F'.x) / 2 ∧ P'.y = (A'.y + F'.y) / 2 ∧
      Q'.x = (B'.x + F'.x) / 2 ∧ Q'.y = (B'.y + F'.y) / 2 ∧
      (P'.y - 0) * (Q'.y - 0) = -(P'.x - 0) * (Q'.x - 0) ∧
      e' = C'.m) := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l329_32986


namespace NUMINAMATH_CALUDE_trajectory_of_G_l329_32944

noncomputable section

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x + Real.sqrt 7)^2 + y^2 = 64

-- Define the fixed point N
def point_N : ℝ × ℝ := (Real.sqrt 7, 0)

-- Define a point P on the circle M
def point_P (x y : ℝ) : Prop := circle_M x y

-- Define point Q on line NP
def point_Q (x y : ℝ) : Prop := ∃ t : ℝ, (x, y) = ((1 - t) * point_N.1 + t * x, (1 - t) * point_N.2 + t * y)

-- Define point G on line segment MP
def point_G (x y : ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (x, y) = ((1 - t) * (-Real.sqrt 7) + t * x, t * y)

-- Define the condition NP = 2NQ
def condition_NP_2NQ (x_p y_p x_q y_q : ℝ) : Prop :=
  (x_p - point_N.1, y_p - point_N.2) = (2 * (x_q - point_N.1), 2 * (y_q - point_N.2))

-- Define the condition GQ ⋅ NP = 0
def condition_GQ_perp_NP (x_g y_g x_q y_q x_p y_p : ℝ) : Prop :=
  (x_g - x_q) * (x_p - point_N.1) + (y_g - y_q) * (y_p - point_N.2) = 0

theorem trajectory_of_G (x y : ℝ) :
  (∃ x_p y_p x_q y_q, 
    point_P x_p y_p ∧
    point_Q x_q y_q ∧
    point_G x y ∧
    condition_NP_2NQ x_p y_p x_q y_q ∧
    condition_GQ_perp_NP x y x_q y_q x_p y_p) →
  x^2/16 + y^2/9 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_G_l329_32944


namespace NUMINAMATH_CALUDE_xyz_value_l329_32998

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 168) (h2 : y * (z + x) = 180) (h3 : z * (x + y) = 192) :
  x * y * z = 842 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l329_32998


namespace NUMINAMATH_CALUDE_odd_binomials_count_l329_32920

/-- The number of 1's in the binary representation of a natural number -/
def numOnes (n : ℕ) : ℕ := sorry

/-- The number of odd binomial coefficients in the n-th row of Pascal's triangle -/
def numOddBinomials (n : ℕ) : ℕ := sorry

/-- Theorem: The number of odd binomial coefficients in the n-th row of Pascal's triangle
    is equal to 2^k, where k is the number of 1's in the binary representation of n -/
theorem odd_binomials_count (n : ℕ) : numOddBinomials n = 2^(numOnes n) := by sorry

end NUMINAMATH_CALUDE_odd_binomials_count_l329_32920


namespace NUMINAMATH_CALUDE_fraction_simplification_l329_32935

theorem fraction_simplification :
  (3 : ℝ) / (Real.sqrt 75 + Real.sqrt 48 + Real.sqrt 18) = Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l329_32935


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l329_32940

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^2 - 22 * X + 58 = (X - 6) * q + 34 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l329_32940


namespace NUMINAMATH_CALUDE_income_calculation_l329_32912

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 8 = expenditure * 9 →  -- income and expenditure ratio is 9:8
  income = expenditure + savings → -- income equals expenditure plus savings
  savings = 4000 → -- savings are 4000
  income = 36000 := by -- prove that income is 36000
sorry

end NUMINAMATH_CALUDE_income_calculation_l329_32912


namespace NUMINAMATH_CALUDE_dorothy_age_proof_l329_32910

/-- Given Dorothy's age relationships with her sister, prove Dorothy's current age --/
theorem dorothy_age_proof (dorothy_age sister_age : ℕ) : 
  sister_age = 5 →
  dorothy_age = 3 * sister_age →
  dorothy_age + 5 = 2 * (sister_age + 5) →
  dorothy_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_age_proof_l329_32910


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l329_32974

theorem fixed_point_on_line (a b : ℝ) : (2 * a + b) * (-2) + (a + b) * 3 + a - b = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l329_32974


namespace NUMINAMATH_CALUDE_find_b_l329_32939

/-- Definition of the '@' operator for positive integers -/
def at_operator (k : ℕ+) (j : ℕ+) : ℕ+ :=
  sorry

/-- Theorem: Given a = 2020 and r = a / b = 0.5, prove b = 4040 -/
theorem find_b (a : ℕ) (b : ℕ) (r : ℚ) 
  (h1 : a = 2020) 
  (h2 : r = a / b) 
  (h3 : r = 1/2) : 
  b = 4040 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l329_32939


namespace NUMINAMATH_CALUDE_subset_intersection_iff_bounds_l329_32989

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

-- State the theorem
theorem subset_intersection_iff_bounds (a : ℝ) :
  (A a).Nonempty → (A a ⊆ A a ∩ B ↔ 6 ≤ a ∧ a ≤ 9) := by
  sorry

#check subset_intersection_iff_bounds

end NUMINAMATH_CALUDE_subset_intersection_iff_bounds_l329_32989


namespace NUMINAMATH_CALUDE_sum_of_fractions_sum_equals_14_1_l329_32952

theorem sum_of_fractions : 
  (1 / 10 : ℚ) + (2 / 10 : ℚ) + (3 / 10 : ℚ) + (4 / 10 : ℚ) + (10 / 10 : ℚ) + 
  (11 / 10 : ℚ) + (15 / 10 : ℚ) + (20 / 10 : ℚ) + (25 / 10 : ℚ) + (50 / 10 : ℚ) = 
  (141 : ℚ) / 10 := by
  sorry

theorem sum_equals_14_1 : 
  (1 / 10 : ℚ) + (2 / 10 : ℚ) + (3 / 10 : ℚ) + (4 / 10 : ℚ) + (10 / 10 : ℚ) + 
  (11 / 10 : ℚ) + (15 / 10 : ℚ) + (20 / 10 : ℚ) + (25 / 10 : ℚ) + (50 / 10 : ℚ) = 
  14.1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_sum_equals_14_1_l329_32952


namespace NUMINAMATH_CALUDE_probability_multiple_of_four_l329_32937

/-- A set of digits from 1 to 5 -/
def DigitSet : Finset ℕ := {1, 2, 3, 4, 5}

/-- A function to check if a three-digit number is divisible by 4 -/
def isDivisibleByFour (a b c : ℕ) : Prop := (10 * b + c) % 4 = 0

/-- The total number of ways to draw three digits from five -/
def totalWays : ℕ := 5 * 4 * 3

/-- The number of ways to draw three digits that form a number divisible by 4 -/
def validWays : ℕ := 15

theorem probability_multiple_of_four :
  (validWays : ℚ) / totalWays = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_probability_multiple_of_four_l329_32937


namespace NUMINAMATH_CALUDE_four_point_segment_ratio_l329_32970

/-- Given four distinct points on a plane with segment lengths a, a, a, a, 2a, and b,
    prove that b = a√3 -/
theorem four_point_segment_ratio (a b : ℝ) :
  ∃ (A B C D : ℝ × ℝ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    ({dist A B, dist A C, dist A D, dist B C, dist B D, dist C D} : Finset ℝ) =
      {a, a, a, a, 2*a, b} →
    b = a * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_four_point_segment_ratio_l329_32970


namespace NUMINAMATH_CALUDE_equation_solution_l329_32990

theorem equation_solution :
  ∃ x : ℝ, -((1 : ℝ) / 3) * x - 5 = 4 ∧ x = -27 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l329_32990


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_l329_32955

/-- Given a positive real number S representing the area of a rectangle,
    prove that the square with side length √S has the smallest perimeter
    among all rectangles with area S, and this minimum perimeter is 4√S. -/
theorem min_perimeter_rectangle (S : ℝ) (hS : S > 0) :
  ∃ (x y : ℝ),
    x > 0 ∧ y > 0 ∧
    x * y = S ∧
    (∀ (a b : ℝ), a > 0 → b > 0 → a * b = S → 2*(x + y) ≤ 2*(a + b)) ∧
    x = Real.sqrt S ∧ y = Real.sqrt S ∧
    2*(x + y) = 4 * Real.sqrt S :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_rectangle_l329_32955


namespace NUMINAMATH_CALUDE_salary_change_percentage_l329_32922

theorem salary_change_percentage (initial_salary : ℝ) (h : initial_salary > 0) :
  let decreased_salary := initial_salary * (1 - 0.6)
  let final_salary := decreased_salary * (1 + 0.6)
  final_salary = initial_salary * 0.64 ∧ 
  (initial_salary - final_salary) / initial_salary = 0.36 :=
by sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l329_32922


namespace NUMINAMATH_CALUDE_equal_diagonals_bisect_implies_rectangle_all_sides_equal_implies_rhombus_perpendicular_diagonals_not_imply_rhombus_all_sides_equal_not_imply_square_l329_32904

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of quadrilaterals
def has_equal_diagonals (q : Quadrilateral) : Prop := sorry
def diagonals_bisect_each_other (q : Quadrilateral) : Prop := sorry
def has_perpendicular_diagonals (q : Quadrilateral) : Prop := sorry
def has_all_sides_equal (q : Quadrilateral) : Prop := sorry

-- Define special quadrilaterals
def is_rectangle (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- Theorem 1
theorem equal_diagonals_bisect_implies_rectangle (q : Quadrilateral) :
  has_equal_diagonals q ∧ diagonals_bisect_each_other q → is_rectangle q :=
sorry

-- Theorem 2
theorem all_sides_equal_implies_rhombus (q : Quadrilateral) :
  has_all_sides_equal q → is_rhombus q :=
sorry

-- Theorem 3
theorem perpendicular_diagonals_not_imply_rhombus :
  ∃ q : Quadrilateral, has_perpendicular_diagonals q ∧ ¬is_rhombus q :=
sorry

-- Theorem 4
theorem all_sides_equal_not_imply_square :
  ∃ q : Quadrilateral, has_all_sides_equal q ∧ ¬is_square q :=
sorry

end NUMINAMATH_CALUDE_equal_diagonals_bisect_implies_rectangle_all_sides_equal_implies_rhombus_perpendicular_diagonals_not_imply_rhombus_all_sides_equal_not_imply_square_l329_32904


namespace NUMINAMATH_CALUDE_root_sum_ratio_l329_32901

theorem root_sum_ratio (m₁ m₂ : ℝ) : 
  (∃ p q : ℝ, 
    (∀ m : ℝ, m * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m * (q^2 - 3*q) + 2*q + 7 = 0) ∧
    p / q + q / p = 2 ∧
    (m₁ * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m₁ * (q^2 - 3*q) + 2*q + 7 = 0) ∧
    (m₂ * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m₂ * (q^2 - 3*q) + 2*q + 7 = 0)) →
  m₁ / m₂ + m₂ / m₁ = 136 / 9 := by
sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l329_32901


namespace NUMINAMATH_CALUDE_square_last_digits_l329_32943

theorem square_last_digits (n : ℕ) :
  (n^2 % 10 % 2 = 1) → ((n^2 % 100) / 10 % 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_square_last_digits_l329_32943


namespace NUMINAMATH_CALUDE_equation_solution_l329_32907

theorem equation_solution : ∃ (x : ℝ), (3 / (x - 2) - 1 = 1 / (2 - x)) ∧ (x = 6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l329_32907


namespace NUMINAMATH_CALUDE_series_sum_equals_four_ninths_l329_32946

theorem series_sum_equals_four_ninths :
  (∑' n : ℕ, n / (4 : ℝ)^n) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_four_ninths_l329_32946


namespace NUMINAMATH_CALUDE_no_real_solutions_log_equation_l329_32926

theorem no_real_solutions_log_equation :
  ¬∃ (x : ℝ), (x + 3 > 0 ∧ x - 1 > 0 ∧ x^2 - 2*x - 3 > 0) ∧
  (Real.log (x + 3) + Real.log (x - 1) = Real.log (x^2 - 2*x - 3)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_log_equation_l329_32926


namespace NUMINAMATH_CALUDE_pet_store_gerbils_l329_32987

/-- The number of gerbils left in a pet store after some are sold -/
def gerbils_left (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

/-- Theorem: Given 85 initial gerbils and 69 sold, 16 gerbils are left -/
theorem pet_store_gerbils : gerbils_left 85 69 = 16 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_gerbils_l329_32987


namespace NUMINAMATH_CALUDE_point_on_hyperbola_l329_32933

/-- A point (x, y) lies on the hyperbola y = -6/x if and only if xy = -6 -/
def lies_on_hyperbola (x y : ℝ) : Prop := x * y = -6

/-- The point (3, -2) lies on the hyperbola y = -6/x -/
theorem point_on_hyperbola : lies_on_hyperbola 3 (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_on_hyperbola_l329_32933


namespace NUMINAMATH_CALUDE_graph_of_2x_plus_5_is_straight_line_l329_32967

-- Define what it means for a function to be linear
def is_linear_function (f : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ x, f x = a * x + b

-- Define what it means for a graph to be a straight line
def is_straight_line (f : ℝ → ℝ) : Prop := 
  ∃ m b : ℝ, ∀ x, f x = m * x + b

-- Define our specific function
def f : ℝ → ℝ := λ x => 2 * x + 5

-- State the theorem
theorem graph_of_2x_plus_5_is_straight_line :
  (∀ g : ℝ → ℝ, is_linear_function g → is_straight_line g) →
  is_linear_function f →
  is_straight_line f := by
  sorry

end NUMINAMATH_CALUDE_graph_of_2x_plus_5_is_straight_line_l329_32967


namespace NUMINAMATH_CALUDE_jeans_discount_percentage_l329_32909

/-- Calculate the discount percentage on jeans --/
theorem jeans_discount_percentage
  (original_price : ℝ)
  (discounted_price_for_three : ℝ)
  (h1 : original_price = 40)
  (h2 : discounted_price_for_three = 112) :
  (original_price * 3 - discounted_price_for_three) / (original_price * 2) = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_jeans_discount_percentage_l329_32909


namespace NUMINAMATH_CALUDE_average_first_16_even_numbers_l329_32961

theorem average_first_16_even_numbers : 
  let first_16_even : List ℕ := List.range 16 |>.map (fun n => 2 * (n + 1))
  (first_16_even.sum / first_16_even.length : ℚ) = 17 := by
sorry

end NUMINAMATH_CALUDE_average_first_16_even_numbers_l329_32961


namespace NUMINAMATH_CALUDE_coaches_average_age_l329_32927

theorem coaches_average_age 
  (total_members : ℕ) 
  (overall_average : ℕ) 
  (num_girls : ℕ) 
  (num_boys : ℕ) 
  (num_coaches : ℕ) 
  (girls_average : ℕ) 
  (boys_average : ℕ) 
  (h1 : total_members = 50)
  (h2 : overall_average = 18)
  (h3 : num_girls = 25)
  (h4 : num_boys = 20)
  (h5 : num_coaches = 5)
  (h6 : girls_average = 16)
  (h7 : boys_average = 17)
  (h8 : total_members = num_girls + num_boys + num_coaches) :
  (total_members * overall_average - num_girls * girls_average - num_boys * boys_average) / num_coaches = 32 := by
  sorry

end NUMINAMATH_CALUDE_coaches_average_age_l329_32927


namespace NUMINAMATH_CALUDE_average_existence_l329_32947

theorem average_existence : ∃ N : ℝ, 12 < N ∧ N < 18 ∧ (8 + 12 + N) / 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_existence_l329_32947


namespace NUMINAMATH_CALUDE_lines_coincide_by_rotation_l329_32962

/-- Two lines that intersect can coincide by rotation -/
theorem lines_coincide_by_rotation (α c : ℝ) :
  ∃ (P : ℝ × ℝ), P.1 * Real.sin α = P.2 ∧ 
  ∃ (θ : ℝ), ∀ (x y : ℝ), 
    y = x * Real.sin α ↔ 
    (x - P.1) * Real.cos θ - (y - P.2) * Real.sin θ = 
    ((x - P.1) * Real.sin θ + (y - P.2) * Real.cos θ) * 2 + c :=
sorry

end NUMINAMATH_CALUDE_lines_coincide_by_rotation_l329_32962


namespace NUMINAMATH_CALUDE_vector_operation_l329_32930

def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

theorem vector_operation :
  (2 : ℝ) • a - b = (5, 7) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l329_32930


namespace NUMINAMATH_CALUDE_joshua_and_justin_shared_money_l329_32934

/-- Given that Joshua's share is $30 and it is thrice as much as Justin's share,
    prove that the total amount of money shared by Joshua and Justin is $40. -/
theorem joshua_and_justin_shared_money (joshua_share : ℕ) (justin_share : ℕ) : 
  joshua_share = 30 → joshua_share = 3 * justin_share → joshua_share + justin_share = 40 := by
  sorry

end NUMINAMATH_CALUDE_joshua_and_justin_shared_money_l329_32934


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l329_32942

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: For a geometric sequence, if a_4 * a_6 = 10, then a_2 * a_8 = 10 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_prod : a 4 * a 6 = 10) : a 2 * a 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l329_32942


namespace NUMINAMATH_CALUDE_subtraction_of_negative_l329_32917

theorem subtraction_of_negative : 12.345 - (-3.256) = 15.601 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_negative_l329_32917


namespace NUMINAMATH_CALUDE_circle_equation_point_on_circle_l329_32976

/-- The standard equation of a circle with center (2, -1) passing through (-1, 3) -/
theorem circle_equation : 
  ∃ (x y : ℝ), (x - 2)^2 + (y + 1)^2 = 25 ∧ 
  ∀ (a b : ℝ), (a - 2)^2 + (b + 1)^2 = 25 ↔ (a, b) ∈ {(x, y) | (x - 2)^2 + (y + 1)^2 = 25} :=
by
  sorry

/-- The given point (-1, 3) satisfies the circle equation -/
theorem point_on_circle : (-1 - 2)^2 + (3 + 1)^2 = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_point_on_circle_l329_32976


namespace NUMINAMATH_CALUDE_quadratic_solution_average_l329_32903

theorem quadratic_solution_average (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 3 * a * x + b
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ (x₁ + x₂) / 2 = 3 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_average_l329_32903


namespace NUMINAMATH_CALUDE_abc_product_l329_32911

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 24 * Real.rpow 3 (1/3))
  (hac : a * c = 40 * Real.rpow 3 (1/3))
  (hbc : b * c = 15 * Real.rpow 3 (1/3)) :
  a * b * c = 120 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l329_32911


namespace NUMINAMATH_CALUDE_quadrilateral_area_l329_32980

/-- The area of a quadrilateral with vertices at (1, 1), (1, 5), (3, 5), and (2006, 2003) is 4014 square units. -/
theorem quadrilateral_area : ℝ := by
  -- Define the vertices of the quadrilateral
  let A : (ℝ × ℝ) := (1, 1)
  let B : (ℝ × ℝ) := (1, 5)
  let C : (ℝ × ℝ) := (3, 5)
  let D : (ℝ × ℝ) := (2006, 2003)

  -- Define the function to calculate the area of a triangle given three points
  let triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
    let (x1, y1) := p1
    let (x2, y2) := p2
    let (x3, y3) := p3
    (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

  -- Calculate the area of the quadrilateral
  let area := triangle_area A B C + triangle_area A C D

  -- Prove that the area is equal to 4014
  sorry

-- The theorem statement
#check quadrilateral_area

end NUMINAMATH_CALUDE_quadrilateral_area_l329_32980


namespace NUMINAMATH_CALUDE_smarties_remainder_l329_32902

theorem smarties_remainder (n : ℕ) (h : n % 11 = 6) : (4 * n) % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_smarties_remainder_l329_32902


namespace NUMINAMATH_CALUDE_otimes_neg_two_three_otimes_commutative_four_neg_two_l329_32936

-- Define the ⊗ operation for rational numbers
def otimes (a b : ℚ) : ℚ := a * b - a - b - 2

-- Theorem 1: (-2) ⊗ 3 = -9
theorem otimes_neg_two_three : otimes (-2) 3 = -9 := by sorry

-- Theorem 2: 4 ⊗ (-2) = (-2) ⊗ 4
theorem otimes_commutative_four_neg_two : otimes 4 (-2) = otimes (-2) 4 := by sorry

end NUMINAMATH_CALUDE_otimes_neg_two_three_otimes_commutative_four_neg_two_l329_32936


namespace NUMINAMATH_CALUDE_parabola_max_vertex_sum_l329_32929

theorem parabola_max_vertex_sum (a T : ℤ) (h_T : T ≠ 0) : 
  let parabola (x y : ℝ) := ∃ b c : ℝ, y = a * x^2 + b * x + c
  let passes_through (x y : ℝ) := parabola x y
  let vertex_sum := 
    let h : ℝ := T
    let k : ℝ := -a * T^2
    h + k
  (passes_through 0 0) ∧ 
  (passes_through (2 * T) 0) ∧ 
  (passes_through (T + 2) 32) →
  (∀ N : ℝ, N = vertex_sum → N ≤ 68) ∧ 
  (∃ N : ℝ, N = vertex_sum ∧ N = 68) :=
by sorry

end NUMINAMATH_CALUDE_parabola_max_vertex_sum_l329_32929


namespace NUMINAMATH_CALUDE_binomial_prob_half_l329_32997

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: If X ~ B(n, p) with E(X) = 6 and D(X) = 3, then p = 1/2 -/
theorem binomial_prob_half (X : BinomialRV) 
  (h_exp : expectation X = 6)
  (h_var : variance X = 3) : 
  X.p = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_prob_half_l329_32997


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l329_32919

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 2) = 7 → x = 47 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l329_32919


namespace NUMINAMATH_CALUDE_female_officers_count_l329_32981

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 144 →
  female_ratio = 1/2 →
  female_on_duty_ratio = 18/100 →
  ↑(total_on_duty : ℕ) * female_ratio * (1 / female_on_duty_ratio) = 400 :=
by sorry

end NUMINAMATH_CALUDE_female_officers_count_l329_32981


namespace NUMINAMATH_CALUDE_locus_is_circle_l329_32900

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  s : ℝ
  a : Point
  b : Point
  c : Point

/-- The sum of squares of distances from a point to the vertices of a triangle -/
def sumOfSquaredDistances (p : Point) (t : IsoscelesRightTriangle) : ℝ :=
  (p.x - t.a.x)^2 + (p.y - t.a.y)^2 +
  (p.x - t.b.x)^2 + (p.y - t.b.y)^2 +
  (p.x - t.c.x)^2 + (p.y - t.c.y)^2

/-- The locus of points P such that the sum of squares of distances from P to the vertices is less than 2s^2 -/
def locus (t : IsoscelesRightTriangle) : Set Point :=
  {p : Point | sumOfSquaredDistances p t < 2 * t.s^2}

theorem locus_is_circle (t : IsoscelesRightTriangle) :
  locus t = {p : Point | (p.x - t.s/3)^2 + (p.y - t.s/3)^2 < (2*t.s/3)^2} :=
sorry

end NUMINAMATH_CALUDE_locus_is_circle_l329_32900


namespace NUMINAMATH_CALUDE_intersection_condition_implies_range_l329_32948

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 5}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 2}

-- Define the range of a
def range_of_a : Set ℝ := {a | a ≤ -3 ∨ a > 2}

-- State the theorem
theorem intersection_condition_implies_range :
  ∀ a : ℝ, (A ∩ B a = B a) → a ∈ range_of_a :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_implies_range_l329_32948


namespace NUMINAMATH_CALUDE_f_increasing_implies_F_decreasing_l329_32938

/-- A function f is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Definition of F in terms of f -/
def F (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f (1 - x) - f (1 + x)

/-- A function f is decreasing on ℝ -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem f_increasing_implies_F_decreasing (f : ℝ → ℝ) (h : IsIncreasing f) : IsDecreasing (F f) := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_implies_F_decreasing_l329_32938


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l329_32966

theorem quadratic_no_real_roots 
  (p q a b c : ℝ) 
  (hp_pos : p > 0) (hq_pos : q > 0) (ha_pos : a > 0) (hb_pos : b > 0) (hc_pos : c > 0)
  (hp_neq_q : p ≠ q)
  (h_geom : a^2 = p * q)
  (h_arith : b = (2*p + q)/3 ∧ c = (p + 2*q)/3) :
  (2*a)^2 - 4*b*c < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l329_32966


namespace NUMINAMATH_CALUDE_numerica_base_l329_32964

/-- Convert a number from base r to base 10 -/
def to_base_10 (digits : List Nat) (r : Nat) : Nat :=
  digits.foldr (fun d acc => d + r * acc) 0

/-- The base r used in Numerica -/
def r : Nat := sorry

/-- The price of the gadget in base r -/
def price : List Nat := [5, 3, 0]

/-- The payment made in base r -/
def payment : List Nat := [1, 1, 0, 0]

/-- The change received in base r -/
def change : List Nat := [4, 6, 0]

theorem numerica_base :
  (to_base_10 price r + to_base_10 change r = to_base_10 payment r) ↔ r = 9 := by
  sorry

end NUMINAMATH_CALUDE_numerica_base_l329_32964


namespace NUMINAMATH_CALUDE_divisor_problem_l329_32983

theorem divisor_problem (k : ℕ) : 12^k ∣ 856736 → k = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l329_32983


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l329_32950

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 1}
def B : Set ℝ := {x | -1 ≤ x ∧ x < 2}

-- State the theorem
theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {x : ℝ | x < 1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l329_32950


namespace NUMINAMATH_CALUDE_coin_division_problem_l329_32924

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (n % 8 = 5) → 
  (n % 7 = 2) → 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 5 ∨ m % 7 ≠ 2)) →
  (n % 9 = 1) := by
sorry

end NUMINAMATH_CALUDE_coin_division_problem_l329_32924


namespace NUMINAMATH_CALUDE_both_shooters_hit_probability_l329_32985

theorem both_shooters_hit_probability
  (prob_A : ℝ)
  (prob_B : ℝ)
  (h_prob_A : prob_A = 0.9)
  (h_prob_B : prob_B = 0.8)
  (h_independent : True)  -- Assumption of independence
  : prob_A * prob_B = 0.72 :=
by sorry

end NUMINAMATH_CALUDE_both_shooters_hit_probability_l329_32985


namespace NUMINAMATH_CALUDE_complex_power_of_four_l329_32992

theorem complex_power_of_four :
  (3 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^4 =
  -40.5 + 40.5 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_of_four_l329_32992


namespace NUMINAMATH_CALUDE_sequence_a_bounds_l329_32915

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | n+1 => sequence_a n + (1 / (n+1)^2) * (sequence_a n)^2

theorem sequence_a_bounds (n : ℕ) : 1 - 1 / (n + 3) < sequence_a (n + 1) ∧ sequence_a (n + 1) < n + 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_bounds_l329_32915


namespace NUMINAMATH_CALUDE_max_inscribed_triangles_count_l329_32941

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) :=
  (h_pos : 0 < b ∧ b < a)

/-- A right-angled isosceles triangle inscribed in an ellipse -/
structure InscribedTriangle (e : Ellipse a b) :=
  (vertex : ℝ × ℝ)
  (h_on_ellipse : (vertex.1^2 / a^2) + (vertex.2^2 / b^2) = 1)
  (h_right_angled : True)  -- Placeholder for the right-angled condition
  (h_isosceles : True)     -- Placeholder for the isosceles condition
  (h_vertex_b : vertex.1 = 0 ∧ vertex.2 = b)

/-- The maximum number of right-angled isosceles triangles inscribed in an ellipse -/
def max_inscribed_triangles (e : Ellipse a b) : ℕ :=
  3

theorem max_inscribed_triangles_count (a b : ℝ) (e : Ellipse a b) :
  ∃ (n : ℕ), n ≤ max_inscribed_triangles e ∧
  ∀ (m : ℕ), (∃ (triangles : Fin m → InscribedTriangle e), 
    ∀ (i j : Fin m), i ≠ j → triangles i ≠ triangles j) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_inscribed_triangles_count_l329_32941


namespace NUMINAMATH_CALUDE_periodic_function_l329_32960

theorem periodic_function (f : ℝ → ℝ) (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - (f x)^2)) →
  ∀ x : ℝ, f x = f (x + 2*a) :=
by sorry

end NUMINAMATH_CALUDE_periodic_function_l329_32960


namespace NUMINAMATH_CALUDE_final_grasshoppers_count_l329_32953

/-- Represents the state of the cage --/
structure CageState where
  crickets : ℕ
  grasshoppers : ℕ

/-- Represents a magician's trick --/
inductive Trick
  | Red
  | Green

/-- Applies a single trick to the cage state --/
def applyTrick (state : CageState) (trick : Trick) : CageState :=
  match trick with
  | Trick.Red => CageState.mk (state.crickets + 1) (state.grasshoppers - 4)
  | Trick.Green => CageState.mk (state.crickets - 5) (state.grasshoppers + 2)

/-- Applies a sequence of tricks to the cage state --/
def applyTricks (state : CageState) (tricks : List Trick) : CageState :=
  tricks.foldl applyTrick state

theorem final_grasshoppers_count (tricks : List Trick) :
  tricks.length = 18 →
  (applyTricks (CageState.mk 30 30) tricks).crickets = 0 →
  (applyTricks (CageState.mk 30 30) tricks).grasshoppers = 6 :=
by sorry

end NUMINAMATH_CALUDE_final_grasshoppers_count_l329_32953


namespace NUMINAMATH_CALUDE_seryozha_healthy_eating_days_l329_32968

/-- Represents the daily cookie consumption pattern -/
structure DailyCookies where
  chocolate : ℕ
  sugarFree : ℕ

/-- Represents the total cookie consumption over a period -/
structure TotalCookies where
  chocolate : ℕ
  sugarFree : ℕ

/-- Calculates the total cookies consumed over a period given the initial and final daily consumption -/
def calculateTotalCookies (initial final : DailyCookies) (days : ℕ) : TotalCookies :=
  { chocolate := (initial.chocolate + final.chocolate) * days / 2,
    sugarFree := (initial.sugarFree + final.sugarFree) * days / 2 }

/-- Theorem stating the number of days in Seryozha's healthy eating regimen -/
theorem seryozha_healthy_eating_days : 
  ∃ (initial : DailyCookies) (days : ℕ),
    let final : DailyCookies := ⟨initial.chocolate - (days - 1), initial.sugarFree + (days - 1)⟩
    let total : TotalCookies := calculateTotalCookies initial final days
    total.chocolate = 264 ∧ total.sugarFree = 187 ∧ days = 11 := by
  sorry


end NUMINAMATH_CALUDE_seryozha_healthy_eating_days_l329_32968


namespace NUMINAMATH_CALUDE_range_of_a_l329_32982

-- Define the propositions p and q as functions of x and a
def p (x : ℝ) : Prop := |4*x - 3| ≤ 1

def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the condition that not p is necessary but not sufficient for not q
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, ¬(q x a) → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ q x a)

-- State the theorem
theorem range_of_a :
  {a : ℝ | necessary_not_sufficient a} = {a : ℝ | a < 0 ∨ a > 1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l329_32982


namespace NUMINAMATH_CALUDE_stingray_count_shark_stingray_relation_total_fish_count_l329_32908

/-- The number of stingrays in an aquarium -/
def num_stingrays : ℕ := 28

/-- The number of sharks in the aquarium -/
def num_sharks : ℕ := 2 * num_stingrays

/-- The total number of fish in the aquarium -/
def total_fish : ℕ := 84

/-- Theorem stating that the number of stingrays is 28 -/
theorem stingray_count : num_stingrays = 28 := by
  sorry

/-- Theorem verifying the relationship between sharks and stingrays -/
theorem shark_stingray_relation : num_sharks = 2 * num_stingrays := by
  sorry

/-- Theorem verifying the total number of fish -/
theorem total_fish_count : num_stingrays + num_sharks = total_fish := by
  sorry

end NUMINAMATH_CALUDE_stingray_count_shark_stingray_relation_total_fish_count_l329_32908


namespace NUMINAMATH_CALUDE_cos_pi_half_minus_two_alpha_l329_32984

theorem cos_pi_half_minus_two_alpha (α : ℝ) (h : Real.sin (π/4 + α) = 1/3) :
  Real.cos (π/2 - 2*α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_half_minus_two_alpha_l329_32984


namespace NUMINAMATH_CALUDE_kyro_are_fylol_and_glyk_l329_32972

-- Define the types
variable (U : Type) -- Universe of discourse
variable (Fylol Glyk Kyro Mylo : Set U)

-- State the given conditions
variable (h1 : Fylol ⊆ Glyk)
variable (h2 : Kyro ⊆ Glyk)
variable (h3 : Mylo ⊆ Fylol)
variable (h4 : Kyro ⊆ Mylo)

-- Theorem to prove
theorem kyro_are_fylol_and_glyk : Kyro ⊆ Fylol ∩ Glyk := by sorry

end NUMINAMATH_CALUDE_kyro_are_fylol_and_glyk_l329_32972


namespace NUMINAMATH_CALUDE_reading_time_calculation_l329_32971

theorem reading_time_calculation (total_time math_time spelling_time : ℕ) 
  (h1 : total_time = 60)
  (h2 : math_time = 15)
  (h3 : spelling_time = 18) :
  total_time - (math_time + spelling_time) = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l329_32971


namespace NUMINAMATH_CALUDE_greatest_integer_value_five_satisfies_condition_no_greater_integer_greatest_integer_is_five_l329_32958

theorem greatest_integer_value (x : ℤ) : (3 * Int.natAbs x + 4 ≤ 19) → x ≤ 5 :=
by sorry

theorem five_satisfies_condition : 3 * Int.natAbs 5 + 4 ≤ 19 :=
by sorry

theorem no_greater_integer (y : ℤ) : y > 5 → (3 * Int.natAbs y + 4 > 19) :=
by sorry

theorem greatest_integer_is_five : 
  ∃ (x : ℤ), (3 * Int.natAbs x + 4 ≤ 19) ∧ (∀ (y : ℤ), (3 * Int.natAbs y + 4 ≤ 19) → y ≤ x) ∧ x = 5 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_value_five_satisfies_condition_no_greater_integer_greatest_integer_is_five_l329_32958


namespace NUMINAMATH_CALUDE_equal_selection_probability_l329_32913

/-- Represents the selection process for a visiting group from a larger group of students. -/
structure SelectionProcess where
  total_students : ℕ
  selected_students : ℕ
  eliminated_students : ℕ

/-- The probability of a student being selected given the selection process. -/
def selection_probability (process : SelectionProcess) : ℚ :=
  (process.selected_students : ℚ) / (process.total_students : ℚ)

/-- Theorem stating that the selection probability is equal for all students. -/
theorem equal_selection_probability (process : SelectionProcess) 
  (h1 : process.total_students = 2006)
  (h2 : process.selected_students = 50)
  (h3 : process.eliminated_students = 6) :
  ∀ (student1 student2 : Fin process.total_students),
    selection_probability process = selection_probability process :=
by
  sorry

#check equal_selection_probability

end NUMINAMATH_CALUDE_equal_selection_probability_l329_32913


namespace NUMINAMATH_CALUDE_ellipse_and_line_theorem_l329_32993

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  ellipse_C P.1 P.2

-- Define the arithmetic sequence property
def arithmetic_sequence (P : ℝ × ℝ) : Prop :=
  ∃ (d : ℝ), Real.sqrt ((P.1 + 1)^2 + P.2^2) = 2 - d ∧
             Real.sqrt ((P.1 - 1)^2 + P.2^2) = 2 + d

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  y = (3 * Real.sqrt 7 / 7) * (x - 1) ∨
  y = -(3 * Real.sqrt 7 / 7) * (x - 1)

-- Define the perpendicular property
def perpendicular_property (P Q : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1) * (Q.1 - F₁.1) + (P.2 - F₁.2) * (Q.2 - F₁.2) = 0

theorem ellipse_and_line_theorem :
  ∀ (P : ℝ × ℝ),
    point_on_ellipse P →
    arithmetic_sequence P →
    ∀ (Q : ℝ × ℝ),
      point_on_ellipse Q →
      Q.1 = F₂.1 →
      perpendicular_property P Q →
      line_m P.1 P.2 ∧ line_m Q.1 Q.2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_theorem_l329_32993


namespace NUMINAMATH_CALUDE_inequality_equivalence_l329_32914

theorem inequality_equivalence (x : ℝ) : 
  (5 * x - 1 < (x + 1)^2 ∧ (x + 1)^2 < 7 * x - 3) ↔ (2 < x ∧ x < 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l329_32914


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l329_32956

/-- A random variable following a normal distribution with mean 2 and variance 4 -/
def X : Real → Real := sorry

/-- The probability density function of X -/
def pdf_X : Real → Real := sorry

/-- The cumulative distribution function of X -/
def cdf_X : Real → Real := sorry

/-- The value 'a' such that P(X < a) = 0.2 -/
def a : Real := sorry

theorem normal_distribution_symmetry (h1 : ∀ x, pdf_X x = pdf_X (4 - x))
  (h2 : cdf_X a = 0.2) : cdf_X (4 - a) = 0.8 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l329_32956


namespace NUMINAMATH_CALUDE_inequality_proof_l329_32905

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) +
  (a + 2*b + c)^2 / (2*b^2 + (c + a)^2) +
  (a + b + 2*c)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l329_32905


namespace NUMINAMATH_CALUDE_work_equivalence_first_group_size_l329_32977

/-- The number of days it takes the first group to complete the work -/
def days_group1 : ℕ := 40

/-- The number of men in the second group -/
def men_group2 : ℕ := 20

/-- The number of days it takes the second group to complete the work -/
def days_group2 : ℕ := 68

/-- The number of men in the first group -/
def men_group1 : ℕ := 34

theorem work_equivalence :
  men_group1 * days_group1 = men_group2 * days_group2 :=
sorry

theorem first_group_size :
  men_group1 = (men_group2 * days_group2) / days_group1 :=
sorry

end NUMINAMATH_CALUDE_work_equivalence_first_group_size_l329_32977


namespace NUMINAMATH_CALUDE_box_depth_proof_l329_32975

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents a cube -/
structure Cube where
  edgeLength : ℕ

/-- Theorem: Given a box with specific dimensions filled with cubes, prove its depth -/
theorem box_depth_proof (box : BoxDimensions) (cube : Cube) (numCubes : ℕ) :
  box.length = 36 →
  box.width = 45 →
  numCubes = 40 →
  (box.length * box.width * box.depth = numCubes * cube.edgeLength ^ 3) →
  (box.length % cube.edgeLength = 0) →
  (box.width % cube.edgeLength = 0) →
  (box.depth % cube.edgeLength = 0) →
  box.depth = 18 := by
  sorry


end NUMINAMATH_CALUDE_box_depth_proof_l329_32975


namespace NUMINAMATH_CALUDE_italian_sausage_length_l329_32932

/-- The length of an Italian sausage in inches -/
def sausage_length : ℚ := 12 * (2 / 3)

/-- Theorem: The length of the Italian sausage is 8 inches -/
theorem italian_sausage_length : sausage_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_italian_sausage_length_l329_32932


namespace NUMINAMATH_CALUDE_a_seven_minus_a_two_l329_32906

def S (n : ℕ) : ℤ := 2 * n^2 - 3 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem a_seven_minus_a_two : a 7 - a 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_a_seven_minus_a_two_l329_32906


namespace NUMINAMATH_CALUDE_linear_function_fixed_point_l329_32999

theorem linear_function_fixed_point :
  ∀ (k : ℝ), (2 * k - 3) * 2 + (k + 1) * (-3) - (k - 9) = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_function_fixed_point_l329_32999


namespace NUMINAMATH_CALUDE_square_area_is_400_l329_32965

/-- A square cut into five rectangles of equal area -/
structure CutSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The width of one of the rectangles -/
  rect_width : ℝ
  /-- The number of rectangles the square is cut into -/
  num_rectangles : ℕ
  /-- The rectangles have equal area -/
  equal_area : ℝ
  /-- The given width of one rectangle -/
  given_width : ℝ
  /-- Condition: The number of rectangles is 5 -/
  h1 : num_rectangles = 5
  /-- Condition: The given width is 5 -/
  h2 : given_width = 5
  /-- Condition: The area of each rectangle is the total area divided by the number of rectangles -/
  h3 : equal_area = side^2 / num_rectangles
  /-- Condition: One of the rectangles has the given width -/
  h4 : rect_width = given_width

/-- The area of the square is 400 -/
theorem square_area_is_400 (s : CutSquare) : s.side^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_400_l329_32965


namespace NUMINAMATH_CALUDE_survey_sample_size_l329_32994

/-- Represents a survey with its characteristics -/
structure Survey where
  surveyors : ℕ
  households : ℕ
  questionnaires : ℕ

/-- Definition of sample size for a survey -/
def sampleSize (s : Survey) : ℕ := s.questionnaires

/-- Theorem stating that the sample size is equal to the number of questionnaires -/
theorem survey_sample_size (s : Survey) : sampleSize s = s.questionnaires := by
  sorry

/-- The specific survey described in the problem -/
def cityCenterSurvey : Survey := {
  surveyors := 400,
  households := 10000,
  questionnaires := 30000
}

#eval sampleSize cityCenterSurvey

end NUMINAMATH_CALUDE_survey_sample_size_l329_32994


namespace NUMINAMATH_CALUDE_sin_cos_difference_l329_32954

theorem sin_cos_difference (x y : Real) : 
  Real.sin (75 * π / 180) * Real.cos (30 * π / 180) - 
  Real.sin (15 * π / 180) * Real.sin (150 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_difference_l329_32954


namespace NUMINAMATH_CALUDE_total_balloons_l329_32969

/-- Given a set of balloons divided into 7 equal groups with 5 balloons in each group,
    the total number of balloons is 35. -/
theorem total_balloons (num_groups : ℕ) (balloons_per_group : ℕ) 
  (h1 : num_groups = 7) (h2 : balloons_per_group = 5) : 
  num_groups * balloons_per_group = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_l329_32969


namespace NUMINAMATH_CALUDE_cloth_sale_meters_l329_32916

/-- Proves that the number of meters of cloth sold is 85 given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_meters (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
    (h1 : total_selling_price = 8925)
    (h2 : profit_per_meter = 20)
    (h3 : cost_price_per_meter = 85) :
    (total_selling_price : ℚ) / ((cost_price_per_meter : ℚ) + (profit_per_meter : ℚ)) = 85 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_meters_l329_32916


namespace NUMINAMATH_CALUDE_special_equation_result_l329_32949

theorem special_equation_result (x y : ℝ) (h1 : x ≠ y) 
  (h2 : Real.sqrt (x^2 + 1) + Real.sqrt (y^2 + 1) = 2021*x + 2021*y) : 
  (x + Real.sqrt (x^2 + 1)) * (y + Real.sqrt (y^2 + 1)) = 1011/1010 := by
  sorry

end NUMINAMATH_CALUDE_special_equation_result_l329_32949


namespace NUMINAMATH_CALUDE_daps_to_dips_l329_32963

/-- Representation of the currency conversion problem -/
structure Currency where
  daps : ℚ
  dops : ℚ
  dips : ℚ

/-- The conversion rates between currencies -/
def conversion_rates : Currency → Prop
  | c => c.daps * 4 = c.dops * 5 ∧ c.dops * 10 = c.dips * 4

/-- Theorem stating the equivalence of 125 daps to 50 dips -/
theorem daps_to_dips (c : Currency) (h : conversion_rates c) : 
  c.daps * 50 = c.dips * 125 := by
  sorry

end NUMINAMATH_CALUDE_daps_to_dips_l329_32963
