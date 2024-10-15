import Mathlib

namespace NUMINAMATH_CALUDE_caitlins_number_l3531_353161

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem caitlins_number (a b c : ℕ) 
  (h1 : is_two_digit_prime a)
  (h2 : is_two_digit_prime b)
  (h3 : is_two_digit_prime c)
  (h4 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h5 : 1 ≤ a + b ∧ a + b ≤ 31)
  (h6 : a + c < a + b)
  (h7 : b + c > a + b) :
  c = 11 := by
sorry

end NUMINAMATH_CALUDE_caitlins_number_l3531_353161


namespace NUMINAMATH_CALUDE_james_has_winning_strategy_l3531_353190

/-- Represents a player in the coin-choosing game -/
inductive Player : Type
| John : Player
| James : Player

/-- The state of the game at any point -/
structure GameState :=
  (coins_left : List ℕ)
  (john_kopeks : ℕ)
  (james_kopeks : ℕ)
  (current_chooser : Player)

/-- A strategy is a function that takes the current game state and returns the chosen coin -/
def Strategy := GameState → ℕ

/-- The result of the game -/
inductive GameResult
| JohnWins : GameResult
| JamesWins : GameResult
| Draw : GameResult

/-- Play the game given strategies for both players -/
def play_game (john_strategy : Strategy) (james_strategy : Strategy) : GameResult :=
  sorry

/-- A winning strategy for a player ensures they always win or draw -/
def is_winning_strategy (player : Player) (strategy : Strategy) : Prop :=
  match player with
  | Player.John => ∀ james_strategy, play_game strategy james_strategy ≠ GameResult.JamesWins
  | Player.James => ∀ john_strategy, play_game john_strategy strategy ≠ GameResult.JohnWins

/-- The main theorem: James has a winning strategy -/
theorem james_has_winning_strategy :
  ∃ (strategy : Strategy), is_winning_strategy Player.James strategy :=
sorry

end NUMINAMATH_CALUDE_james_has_winning_strategy_l3531_353190


namespace NUMINAMATH_CALUDE_f_max_at_seven_l3531_353126

/-- The quadratic function we're analyzing -/
def f (y : ℝ) : ℝ := y^2 - 14*y + 24

/-- The theorem stating that f achieves its maximum at y = 7 -/
theorem f_max_at_seven :
  ∀ y : ℝ, f y ≤ f 7 := by
  sorry

end NUMINAMATH_CALUDE_f_max_at_seven_l3531_353126


namespace NUMINAMATH_CALUDE_N_mod_500_l3531_353118

/-- A function that counts the number of 1s in the binary representation of a natural number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- The sequence of positive integers whose binary representation has exactly 7 ones -/
def S : List ℕ := sorry

/-- The 500th number in the sequence S -/
def N : ℕ := sorry

theorem N_mod_500 : N % 500 = 375 := by sorry

end NUMINAMATH_CALUDE_N_mod_500_l3531_353118


namespace NUMINAMATH_CALUDE_cube_sum_of_equal_ratios_l3531_353169

theorem cube_sum_of_equal_ratios (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_equal_ratios_l3531_353169


namespace NUMINAMATH_CALUDE_cube_sphere_volume_ratio_cube_inscribed_in_sphere_volume_ratio_l3531_353165

/-- The ratio of the volume of a cube inscribed in a sphere to the volume of the sphere. -/
theorem cube_sphere_volume_ratio : ℝ :=
  2 * Real.sqrt 3 / Real.pi

/-- Theorem: For a cube inscribed in a sphere, the ratio of the volume of the cube
    to the volume of the sphere is 2√3/π. -/
theorem cube_inscribed_in_sphere_volume_ratio :
  let s : ℝ := cube_side_length -- side length of the cube
  let r : ℝ := sphere_radius -- radius of the sphere
  let cube_volume : ℝ := s^3
  let sphere_volume : ℝ := (4/3) * Real.pi * r^3
  r = (Real.sqrt 3 / 2) * s → -- condition that the cube is inscribed in the sphere
  cube_volume / sphere_volume = cube_sphere_volume_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_cube_sphere_volume_ratio_cube_inscribed_in_sphere_volume_ratio_l3531_353165


namespace NUMINAMATH_CALUDE_xyz_value_l3531_353123

theorem xyz_value (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 100 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3531_353123


namespace NUMINAMATH_CALUDE_min_cos_sum_sin_triangle_angles_l3531_353132

theorem min_cos_sum_sin_triangle_angles (A B C : Real) : 
  A + B + C = π → 
  A > 0 → B > 0 → C > 0 →
  ∃ (m : Real), m = -2 * Real.sqrt 6 / 9 ∧ 
    ∀ (X Y Z : Real), X + Y + Z = π → X > 0 → Y > 0 → Z > 0 → 
      m ≤ Real.cos X * (Real.sin Y + Real.sin Z) :=
by sorry

end NUMINAMATH_CALUDE_min_cos_sum_sin_triangle_angles_l3531_353132


namespace NUMINAMATH_CALUDE_unique_prime_sum_and_difference_l3531_353131

theorem unique_prime_sum_and_difference : 
  ∃! p : ℕ, 
    Prime p ∧ 
    (∃ q₁ q₂ : ℕ, Prime q₁ ∧ Prime q₂ ∧ p = q₁ + q₂) ∧
    (∃ q₃ q₄ : ℕ, Prime q₃ ∧ Prime q₄ ∧ q₃ > q₄ ∧ p = q₃ - q₄) ∧
    p = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_sum_and_difference_l3531_353131


namespace NUMINAMATH_CALUDE_circle_condition_l3531_353104

theorem circle_condition (m : ℝ) : 
  (∃ (a b r : ℝ), ∀ (x y : ℝ), (x^2 + y^2 - x + y + m = 0) ↔ ((x - a)^2 + (y - b)^2 = r^2)) → 
  m < 1/2 := by
sorry

end NUMINAMATH_CALUDE_circle_condition_l3531_353104


namespace NUMINAMATH_CALUDE_power_seven_700_mod_100_l3531_353199

theorem power_seven_700_mod_100 : 7^700 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_700_mod_100_l3531_353199


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3531_353166

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r ^ (n - 1)

theorem tenth_term_of_sequence (a : ℚ) (r : ℚ) :
  a = 5 → r = 3/2 → geometric_sequence a r 10 = 98415/512 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3531_353166


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3531_353101

/-- Given that the solution set of ax² + bx + c > 0 is (-1/3, 2),
    prove that the solution set of cx² + bx + a < 0 is (-3, 1/2) -/
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : ∀ x : ℝ, ax^2 + b*x + c > 0 ↔ -1/3 < x ∧ x < 2) :
  ∀ x : ℝ, c*x^2 + b*x + a < 0 ↔ -3 < x ∧ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3531_353101


namespace NUMINAMATH_CALUDE_selection_schemes_correct_l3531_353185

/-- The number of ways to select 4 students from 4 boys and 2 girls, with at least 1 girl in the group -/
def selection_schemes (num_boys num_girls group_size : ℕ) : ℕ :=
  Nat.choose (num_boys + num_girls) group_size - Nat.choose num_boys group_size

theorem selection_schemes_correct :
  selection_schemes 4 2 4 = 14 := by
  sorry

#eval selection_schemes 4 2 4

end NUMINAMATH_CALUDE_selection_schemes_correct_l3531_353185


namespace NUMINAMATH_CALUDE_no_candies_to_remove_for_30_and_5_l3531_353110

/-- Given a number of candies and sisters, calculate the minimum number of candies to remove for even distribution -/
def min_candies_to_remove (candies : ℕ) (sisters : ℕ) : ℕ :=
  candies % sisters

/-- Prove that for 30 candies and 5 sisters, no candies need to be removed for even distribution -/
theorem no_candies_to_remove_for_30_and_5 :
  min_candies_to_remove 30 5 = 0 := by
  sorry

#eval min_candies_to_remove 30 5

end NUMINAMATH_CALUDE_no_candies_to_remove_for_30_and_5_l3531_353110


namespace NUMINAMATH_CALUDE_least_multiple_945_l3531_353195

-- Define a function to check if a number is a multiple of 45
def isMultipleOf45 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 45 * k

-- Define a function to get the digits of a number
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

-- Define a function to calculate the product of a list of numbers
def productOfList (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

-- Define the main theorem
theorem least_multiple_945 :
  (isMultipleOf45 945) ∧
  (isMultipleOf45 (productOfList (digits 945))) ∧
  (∀ n : ℕ, n > 0 ∧ n < 945 →
    ¬(isMultipleOf45 n ∧ isMultipleOf45 (productOfList (digits n)))) :=
sorry

end NUMINAMATH_CALUDE_least_multiple_945_l3531_353195


namespace NUMINAMATH_CALUDE_smallest_positive_integer_form_l3531_353194

theorem smallest_positive_integer_form (m n : ℤ) : ∃ (k : ℕ), k > 0 ∧ ∃ (a b : ℤ), k = 1237 * a + 78653 * b ∧ ∀ (l : ℕ), l > 0 → ∃ (c d : ℤ), l = 1237 * c + 78653 * d → k ≤ l :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_form_l3531_353194


namespace NUMINAMATH_CALUDE_log_negative_undefined_l3531_353142

-- Define the logarithm function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_negative_undefined (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 3) :
  ¬∃ y, f a (-2) = y := by
  sorry


end NUMINAMATH_CALUDE_log_negative_undefined_l3531_353142


namespace NUMINAMATH_CALUDE_estimate_pi_l3531_353137

theorem estimate_pi (total_points : ℕ) (circle_points : ℕ) 
  (h1 : total_points = 1000) 
  (h2 : circle_points = 780) : 
  (circle_points : ℚ) / total_points * 4 = 78 / 25 := by
  sorry

end NUMINAMATH_CALUDE_estimate_pi_l3531_353137


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3531_353100

theorem quadratic_equation_solution (x : ℝ) :
  x^2 + 4*x - 2 = 0 ↔ x = -2 + Real.sqrt 6 ∨ x = -2 - Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3531_353100


namespace NUMINAMATH_CALUDE_essay_competition_probability_l3531_353154

theorem essay_competition_probability (n : ℕ) (h : n = 6) :
  let total_outcomes := n * n
  let favorable_outcomes := n * (n - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 6 :=
by sorry

end NUMINAMATH_CALUDE_essay_competition_probability_l3531_353154


namespace NUMINAMATH_CALUDE_spinner_probability_l3531_353119

theorem spinner_probability (pA pB pC pD pE : ℚ) : 
  pA = 1/3 →
  pB = 1/6 →
  pC = 2*pE →
  pD = 2*pE →
  pA + pB + pC + pD + pE = 1 →
  pE = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3531_353119


namespace NUMINAMATH_CALUDE_new_teacher_student_ratio_l3531_353138

/-- Proves that given the initial conditions, the new ratio of teachers to students is 1:25 -/
theorem new_teacher_student_ratio
  (initial_ratio : ℚ)
  (initial_teachers : ℕ)
  (student_increase : ℕ)
  (teacher_increase : ℕ)
  (new_student_ratio : ℚ)
  (h1 : initial_ratio = 50 / 1)
  (h2 : initial_teachers = 3)
  (h3 : student_increase = 50)
  (h4 : teacher_increase = 5)
  (h5 : new_student_ratio = 25 / 1) :
  (initial_teachers + teacher_increase) / (initial_ratio * initial_teachers + student_increase) = 1 / 25 := by
  sorry


end NUMINAMATH_CALUDE_new_teacher_student_ratio_l3531_353138


namespace NUMINAMATH_CALUDE_parabola_vertex_l3531_353159

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * (x - 3)^2 + 4

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (3, 4)

/-- Theorem: The vertex of the parabola y = -2(x-3)^2 + 4 is (3, 4) -/
theorem parabola_vertex : 
  ∀ x y : ℝ, parabola x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3531_353159


namespace NUMINAMATH_CALUDE_hundred_power_ten_as_sum_of_tens_l3531_353197

theorem hundred_power_ten_as_sum_of_tens (n : ℕ) : (100 ^ 10) = n * 10 → n = 10 ^ 19 := by
  sorry

end NUMINAMATH_CALUDE_hundred_power_ten_as_sum_of_tens_l3531_353197


namespace NUMINAMATH_CALUDE_slower_train_speed_l3531_353120

theorem slower_train_speed 
  (train_length : ℝ) 
  (faster_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 80) 
  (h2 : faster_speed = 52) 
  (h3 : passing_time = 36) : 
  ∃ slower_speed : ℝ, 
    slower_speed = 36 ∧ 
    (faster_speed - slower_speed) * passing_time / 3600 * 1000 = 2 * train_length :=
by sorry

end NUMINAMATH_CALUDE_slower_train_speed_l3531_353120


namespace NUMINAMATH_CALUDE_nigella_base_salary_l3531_353125

def house_sale_income (base_salary : ℝ) (commission_rate : ℝ) (house_prices : List ℝ) : ℝ :=
  base_salary + (commission_rate * (house_prices.sum))

theorem nigella_base_salary :
  let commission_rate : ℝ := 0.02
  let house_a_price : ℝ := 60000
  let house_b_price : ℝ := 3 * house_a_price
  let house_c_price : ℝ := 2 * house_a_price - 110000
  let house_prices : List ℝ := [house_a_price, house_b_price, house_c_price]
  let total_income : ℝ := 8000
  ∃ (base_salary : ℝ), 
    house_sale_income base_salary commission_rate house_prices = total_income ∧
    base_salary = 3000 :=
by sorry

end NUMINAMATH_CALUDE_nigella_base_salary_l3531_353125


namespace NUMINAMATH_CALUDE_unanswered_questions_l3531_353176

/-- Represents the scoring system and results of a math contest --/
structure ContestScore where
  totalQuestions : ℕ
  oldScore : ℕ
  newScore : ℕ

/-- Proves that the number of unanswered questions is 10 given the contest conditions --/
theorem unanswered_questions (score : ContestScore)
  (h1 : score.totalQuestions = 40)
  (h2 : ∃ c w : ℕ, 25 + 3 * c - w = score.oldScore)
  (h3 : score.oldScore = 95)
  (h4 : ∃ c w u : ℕ, 6 * c - 2 * w + 3 * u = score.newScore)
  (h5 : score.newScore = 120)
  (h6 : ∃ c w u : ℕ, c + w + u = score.totalQuestions) :
  ∃ c w : ℕ, c + w + 10 = score.totalQuestions :=
sorry

end NUMINAMATH_CALUDE_unanswered_questions_l3531_353176


namespace NUMINAMATH_CALUDE_shirt_discount_percentage_l3531_353178

theorem shirt_discount_percentage (original_price discounted_price : ℝ) 
  (h1 : original_price = 80)
  (h2 : discounted_price = 68) :
  (original_price - discounted_price) / original_price * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_shirt_discount_percentage_l3531_353178


namespace NUMINAMATH_CALUDE_survey_result_l3531_353102

theorem survey_result (total : ℕ) (radio_dislike_percent : ℚ) (music_dislike_percent : ℚ)
  (h_total : total = 1500)
  (h_radio : radio_dislike_percent = 40 / 100)
  (h_music : music_dislike_percent = 15 / 100) :
  (total : ℚ) * radio_dislike_percent * music_dislike_percent = 90 :=
by sorry

end NUMINAMATH_CALUDE_survey_result_l3531_353102


namespace NUMINAMATH_CALUDE_sam_initial_puppies_l3531_353196

/-- The number of puppies Sam gave away -/
def puppies_given : ℝ := 2.0

/-- The number of puppies Sam has now -/
def puppies_remaining : ℕ := 4

/-- The initial number of puppies Sam had -/
def initial_puppies : ℝ := puppies_given + puppies_remaining

theorem sam_initial_puppies : initial_puppies = 6.0 := by
  sorry

end NUMINAMATH_CALUDE_sam_initial_puppies_l3531_353196


namespace NUMINAMATH_CALUDE_max_square_triangle_area_ratio_l3531_353115

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A square with vertices X, Y, Z, and V. -/
structure Square where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  V : ℝ × ℝ

/-- The area of a triangle. -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The area of a square. -/
def squareArea (s : Square) : ℝ := sorry

/-- Predicate to check if a point is on a line segment. -/
def isOnSegment (p : ℝ × ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if two line segments are parallel. -/
def areParallel (a1 : ℝ × ℝ) (b1 : ℝ × ℝ) (a2 : ℝ × ℝ) (b2 : ℝ × ℝ) : Prop := sorry

/-- The main theorem stating the maximum area ratio. -/
theorem max_square_triangle_area_ratio 
  (t : Triangle) 
  (s : Square) 
  (h1 : isOnSegment s.X t.A t.B)
  (h2 : isOnSegment s.Y t.B t.C)
  (h3 : isOnSegment s.Z t.C t.A)
  (h4 : isOnSegment s.V t.A t.C)
  (h5 : areParallel s.V s.Z t.A t.B) :
  squareArea s / triangleArea t ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_max_square_triangle_area_ratio_l3531_353115


namespace NUMINAMATH_CALUDE_log_z_m_value_l3531_353147

theorem log_z_m_value (x y z m : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1) (hm : m > 0)
  (hlogx : Real.log m / Real.log x = 24)
  (hlogy : Real.log m / Real.log y = 40)
  (hlogxyz : Real.log m / (Real.log x + Real.log y + Real.log z) = 12) :
  Real.log m / Real.log z = 60 := by
  sorry

end NUMINAMATH_CALUDE_log_z_m_value_l3531_353147


namespace NUMINAMATH_CALUDE_circle_equation_coefficients_l3531_353181

theorem circle_equation_coefficients (D E F : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + D*x + E*y + F = 0 ↔ (x + 2)^2 + (y - 3)^2 = 4^2) →
  D = 4 ∧ E = -6 ∧ F = -3 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_coefficients_l3531_353181


namespace NUMINAMATH_CALUDE_janous_inequality_janous_equality_l3531_353187

theorem janous_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 := by
  sorry

theorem janous_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ y = z ∧ x = 2 * y := by
  sorry

end NUMINAMATH_CALUDE_janous_inequality_janous_equality_l3531_353187


namespace NUMINAMATH_CALUDE_smallest_divisor_square_plus_divisor_square_l3531_353164

theorem smallest_divisor_square_plus_divisor_square (n : ℕ) :
  n ≥ 2 →
  (∃ k d : ℕ,
    k > 1 ∧
    k ∣ n ∧
    (∀ m : ℕ, m > 1 → m ∣ n → m ≥ k) ∧
    d ∣ n ∧
    n = k^2 + d^2) ↔
  n = 8 ∨ n = 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_divisor_square_plus_divisor_square_l3531_353164


namespace NUMINAMATH_CALUDE_regression_line_not_exact_l3531_353139

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 0.5 * x - 85

-- Define the specific x value
def x_value : ℝ := 200

-- Theorem statement
theorem regression_line_not_exact (ε : ℝ) (h : ε > 0) :
  ∃ y : ℝ, y ≠ 15 ∧ |y - regression_line x_value| < ε :=
sorry

end NUMINAMATH_CALUDE_regression_line_not_exact_l3531_353139


namespace NUMINAMATH_CALUDE_inequality_proof_l3531_353146

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*a*c)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3531_353146


namespace NUMINAMATH_CALUDE_butter_mixture_profit_percentage_l3531_353124

/-- Calculates the profit percentage for a butter mixture sale --/
theorem butter_mixture_profit_percentage 
  (butter1_weight : ℝ) 
  (butter1_price : ℝ) 
  (butter2_weight : ℝ) 
  (butter2_price : ℝ) 
  (selling_price : ℝ) :
  butter1_weight = 54 →
  butter1_price = 150 →
  butter2_weight = 36 →
  butter2_price = 125 →
  selling_price = 196 →
  let total_cost := butter1_weight * butter1_price + butter2_weight * butter2_price
  let total_weight := butter1_weight + butter2_weight
  let selling_amount := selling_price * total_weight
  let profit := selling_amount - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage = 40 := by
sorry

end NUMINAMATH_CALUDE_butter_mixture_profit_percentage_l3531_353124


namespace NUMINAMATH_CALUDE_brad_speed_is_6_l3531_353177

-- Define the given conditions
def maxwell_speed : ℝ := 4
def brad_delay : ℝ := 1
def total_distance : ℝ := 34
def meeting_time : ℝ := 4

-- Define Brad's speed as a variable
def brad_speed : ℝ := sorry

-- Theorem to prove
theorem brad_speed_is_6 : brad_speed = 6 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_brad_speed_is_6_l3531_353177


namespace NUMINAMATH_CALUDE_impossibleColoring_l3531_353108

def Color := Bool

def isRed (c : Color) : Prop := c = true
def isBlue (c : Color) : Prop := c = false

theorem impossibleColoring :
  ¬∃(f : ℕ → Color),
    (∀ n : ℕ, n > 1000 → (isRed (f n) ∨ isBlue (f n))) ∧
    (∀ m n : ℕ, m > 1000 → n > 1000 → m ≠ n → isRed (f m) → isRed (f n) → isBlue (f (m * n))) ∧
    (∀ m n : ℕ, m > 1000 → n > 1000 → m = n + 1 → ¬(isBlue (f m) ∧ isBlue (f n))) :=
by
  sorry

end NUMINAMATH_CALUDE_impossibleColoring_l3531_353108


namespace NUMINAMATH_CALUDE_red_cars_count_l3531_353112

theorem red_cars_count (black_cars : ℕ) (ratio_red : ℕ) (ratio_black : ℕ) : 
  black_cars = 75 → ratio_red = 3 → ratio_black = 8 → 
  ∃ (red_cars : ℕ), red_cars * ratio_black = black_cars * ratio_red ∧ red_cars = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_red_cars_count_l3531_353112


namespace NUMINAMATH_CALUDE_candidate_c_wins_l3531_353167

/-- Represents a candidate in the election --/
inductive Candidate
  | A
  | B
  | C
  | D
  | E

/-- Returns the vote count for a given candidate --/
def votes (c : Candidate) : Float :=
  match c with
  | Candidate.A => 4237.5
  | Candidate.B => 7298.25
  | Candidate.C => 12498.75
  | Candidate.D => 8157.5
  | Candidate.E => 3748.3

/-- Calculates the total number of votes --/
def totalVotes : Float :=
  votes Candidate.A + votes Candidate.B + votes Candidate.C + votes Candidate.D + votes Candidate.E

/-- Calculates the percentage of votes for a given candidate --/
def votePercentage (c : Candidate) : Float :=
  (votes c / totalVotes) * 100

/-- Theorem stating that Candidate C has the highest percentage of votes --/
theorem candidate_c_wins :
  ∀ c : Candidate, c ≠ Candidate.C → votePercentage Candidate.C > votePercentage c :=
by sorry

end NUMINAMATH_CALUDE_candidate_c_wins_l3531_353167


namespace NUMINAMATH_CALUDE_homework_problem_distribution_l3531_353183

theorem homework_problem_distribution (total : ℕ) (finished : ℕ) (pages : ℕ) 
  (h1 : total = 60) 
  (h2 : finished = 20) 
  (h3 : pages = 5) 
  (h4 : pages > 0) :
  (total - finished) / pages = 8 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_distribution_l3531_353183


namespace NUMINAMATH_CALUDE_diane_poker_debt_l3531_353148

/-- Calculates the amount owed in a poker game scenario -/
def amount_owed (initial_amount winnings total_loss : ℕ) : ℕ :=
  total_loss - (initial_amount + winnings)

/-- Theorem: In Diane's poker game scenario, she owes $50 to her friends -/
theorem diane_poker_debt : amount_owed 100 65 215 = 50 := by
  sorry

end NUMINAMATH_CALUDE_diane_poker_debt_l3531_353148


namespace NUMINAMATH_CALUDE_max_participants_l3531_353122

/-- Represents the outcome of a chess game -/
inductive GameResult
| Win
| Draw
| Loss

/-- Represents a chess tournament -/
structure ChessTournament where
  participants : Nat
  results : Fin participants → Fin participants → GameResult

/-- Calculates the score of a player against two other players -/
def score (t : ChessTournament) (p1 p2 p3 : Fin t.participants) : Rat :=
  let s1 := match t.results p1 p2 with
    | GameResult.Win => 1
    | GameResult.Draw => 1/2
    | GameResult.Loss => 0
  let s2 := match t.results p1 p3 with
    | GameResult.Win => 1
    | GameResult.Draw => 1/2
    | GameResult.Loss => 0
  s1 + s2

/-- The tournament satisfies the given conditions -/
def validTournament (t : ChessTournament) : Prop :=
  (∀ p1 p2 : Fin t.participants, p1 ≠ p2 → t.results p1 p2 ≠ t.results p2 p1) ∧
  (∀ p1 p2 p3 : Fin t.participants, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
    (score t p1 p2 p3 = 3/2 ∨ score t p2 p1 p3 = 3/2 ∨ score t p3 p1 p2 = 3/2))

/-- The maximum number of participants in a valid tournament is 5 -/
theorem max_participants : ∀ t : ChessTournament, validTournament t → t.participants ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_participants_l3531_353122


namespace NUMINAMATH_CALUDE_garden_perimeter_l3531_353163

/-- 
A rectangular garden has a diagonal of 34 meters and an area of 240 square meters.
This theorem proves that the perimeter of such a garden is 80 meters.
-/
theorem garden_perimeter : 
  ∀ (a b : ℝ), 
  a > 0 → b > 0 →  -- Ensure positive dimensions
  a * b = 240 →    -- Area condition
  a^2 + b^2 = 34^2 →  -- Diagonal condition
  2 * (a + b) = 80 :=  -- Perimeter calculation
by
  sorry

#check garden_perimeter

end NUMINAMATH_CALUDE_garden_perimeter_l3531_353163


namespace NUMINAMATH_CALUDE_tea_cost_price_l3531_353156

/-- The cost price per kg of the 80 kg of tea -/
def C : ℝ := sorry

/-- The theorem stating the conditions and the result to be proved -/
theorem tea_cost_price :
  -- 80 kg of tea at cost price C
  -- 20 kg of tea at $20 per kg
  -- Total selling price for 100 kg at $20.8 per kg
  -- 30% profit margin
  80 * C + 20 * 20 = (100 * 20.8) / 1.3 →
  C = 15 := by sorry

end NUMINAMATH_CALUDE_tea_cost_price_l3531_353156


namespace NUMINAMATH_CALUDE_congruence_problem_l3531_353170

theorem congruence_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3531_353170


namespace NUMINAMATH_CALUDE_jason_tom_blue_difference_l3531_353140

/-- Represents the number of marbles a person has -/
structure MarbleCount where
  blue : ℕ
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the difference in blue marbles between two MarbleCounts -/
def blueDifference (a b : MarbleCount) : ℕ :=
  if a.blue ≥ b.blue then a.blue - b.blue else b.blue - a.blue

theorem jason_tom_blue_difference :
  let jason : MarbleCount := { blue := 44, red := 16, green := 8, yellow := 0 }
  let tom : MarbleCount := { blue := 24, red := 0, green := 7, yellow := 10 }
  blueDifference jason tom = 20 := by
  sorry

end NUMINAMATH_CALUDE_jason_tom_blue_difference_l3531_353140


namespace NUMINAMATH_CALUDE_function_difference_l3531_353116

theorem function_difference (f : ℕ+ → ℕ+) 
  (h_mono : ∀ m n : ℕ+, m < n → f m < f n)
  (h_comp : ∀ n : ℕ+, f (f n) = 3 * n) :
  f 2202 - f 2022 = 510 := by
sorry

end NUMINAMATH_CALUDE_function_difference_l3531_353116


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3531_353145

theorem geometric_sequence_fourth_term
  (a : ℝ)  -- first term
  (a₆ : ℝ) -- sixth term
  (h₁ : a = 81)
  (h₂ : a₆ = 32)
  (h₃ : ∃ r : ℝ, r > 0 ∧ a₆ = a * r^5) :
  ∃ a₄ : ℝ, a₄ = 24 ∧ ∃ r : ℝ, r > 0 ∧ a₄ = a * r^3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3531_353145


namespace NUMINAMATH_CALUDE_dot_product_AB_AC_l3531_353184

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (3, 4)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem dot_product_AB_AC : dot_product vector_AB vector_AC = -2 := by sorry

end NUMINAMATH_CALUDE_dot_product_AB_AC_l3531_353184


namespace NUMINAMATH_CALUDE_total_necklaces_l3531_353127

def necklaces_problem (boudreaux rhonda latch cecilia : ℕ) : Prop :=
  boudreaux = 12 ∧
  rhonda = boudreaux / 2 ∧
  latch = 3 * rhonda - 4 ∧
  cecilia = latch + 3 ∧
  boudreaux + rhonda + latch + cecilia = 49

theorem total_necklaces : ∃ (boudreaux rhonda latch cecilia : ℕ), 
  necklaces_problem boudreaux rhonda latch cecilia :=
by
  sorry

end NUMINAMATH_CALUDE_total_necklaces_l3531_353127


namespace NUMINAMATH_CALUDE_euler_formula_quadrant_l3531_353180

theorem euler_formula_quadrant :
  let θ : ℝ := 2 * Real.pi / 3
  let z : ℂ := Complex.exp (Complex.I * θ)
  z.re < 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_euler_formula_quadrant_l3531_353180


namespace NUMINAMATH_CALUDE_sphere_in_cone_surface_area_ratio_l3531_353135

theorem sphere_in_cone_surface_area_ratio (r : ℝ) (h : r > 0) :
  let cone_height : ℝ := 3 * r
  let triangle_side : ℝ := 2 * Real.sqrt 3 * r
  let sphere_surface_area : ℝ := 4 * Real.pi * r^2
  let cone_base_radius : ℝ := Real.sqrt 3 * r
  let cone_lateral_area : ℝ := Real.pi * cone_base_radius * triangle_side
  let cone_base_area : ℝ := Real.pi * cone_base_radius^2
  let cone_total_surface_area : ℝ := cone_lateral_area + cone_base_area
  cone_total_surface_area / sphere_surface_area = 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_sphere_in_cone_surface_area_ratio_l3531_353135


namespace NUMINAMATH_CALUDE_max_arithmetic_mean_of_special_pairs_l3531_353136

theorem max_arithmetic_mean_of_special_pairs : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a > b ∧
  (a + b) / 2 = (25 / 24) * Real.sqrt (a * b) ∧
  ∀ (c d : ℕ), 10 ≤ c ∧ c < 100 ∧ 10 ≤ d ∧ d < 100 ∧ c > d ∧
    (c + d) / 2 = (25 / 24) * Real.sqrt (c * d) →
    (a + b) / 2 ≥ (c + d) / 2 ∧
  (a + b) / 2 = 75 :=
by sorry

end NUMINAMATH_CALUDE_max_arithmetic_mean_of_special_pairs_l3531_353136


namespace NUMINAMATH_CALUDE_power_equality_l3531_353179

theorem power_equality : 32^2 * 4^4 = 2^18 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3531_353179


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_one_l3531_353128

theorem trigonometric_product_equals_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_one_l3531_353128


namespace NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_divisor_l3531_353144

/-- A police emergency number is a positive integer that ends with 133 in decimal representation. -/
def is_police_emergency_number (n : ℕ) : Prop :=
  n > 0 ∧ n % 1000 = 133

/-- Every police emergency number has a prime divisor greater than 7. -/
theorem police_emergency_number_has_large_prime_divisor (n : ℕ) :
  is_police_emergency_number n → ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n :=
by sorry

end NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_divisor_l3531_353144


namespace NUMINAMATH_CALUDE_train_speed_l3531_353157

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 1600) (h2 : time = 40) :
  length / time = 40 := by
sorry

end NUMINAMATH_CALUDE_train_speed_l3531_353157


namespace NUMINAMATH_CALUDE_f_composition_equals_9184_l3531_353106

/-- The function f(x) = 3x^2 + 2x - 1 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1

/-- Theorem: f(f(f(1))) = 9184 -/
theorem f_composition_equals_9184 : f (f (f 1)) = 9184 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_9184_l3531_353106


namespace NUMINAMATH_CALUDE_linear_system_solution_l3531_353143

theorem linear_system_solution :
  ∃! (x y : ℝ), (x - y = 1) ∧ (3 * x + 2 * y = 8) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l3531_353143


namespace NUMINAMATH_CALUDE_total_defective_rate_is_correct_l3531_353168

/-- The defective rate of worker x -/
def worker_x_rate : ℝ := 0.005

/-- The defective rate of worker y -/
def worker_y_rate : ℝ := 0.008

/-- The fraction of products checked by worker y -/
def worker_y_fraction : ℝ := 0.8

/-- The fraction of products checked by worker x -/
def worker_x_fraction : ℝ := 1 - worker_y_fraction

/-- The total defective rate of all products -/
def total_defective_rate : ℝ := worker_x_rate * worker_x_fraction + worker_y_rate * worker_y_fraction

theorem total_defective_rate_is_correct :
  total_defective_rate = 0.0074 := by sorry

end NUMINAMATH_CALUDE_total_defective_rate_is_correct_l3531_353168


namespace NUMINAMATH_CALUDE_min_points_for_top_two_l3531_353186

/-- Represents a soccer tournament --/
structure Tournament :=
  (num_teams : Nat)
  (scoring_system : List Nat)

/-- Calculates the total number of matches in a round-robin tournament --/
def total_matches (t : Tournament) : Nat :=
  t.num_teams * (t.num_teams - 1) / 2

/-- Calculates the maximum total points possible in the tournament --/
def max_total_points (t : Tournament) : Nat :=
  (total_matches t) * (t.scoring_system.head!)

/-- Theorem: In a 4-team round-robin tournament with the given scoring system,
    a team needs at least 7 points to guarantee a top-two finish --/
theorem min_points_for_top_two (t : Tournament) 
  (h1 : t.num_teams = 4)
  (h2 : t.scoring_system = [3, 1, 0]) : 
  ∃ (min_points : Nat), 
    (min_points = 7) ∧ 
    (∀ (team_points : Nat), 
      team_points ≥ min_points → 
      (max_total_points t - team_points) / (t.num_teams - 1) < team_points) :=
by sorry

end NUMINAMATH_CALUDE_min_points_for_top_two_l3531_353186


namespace NUMINAMATH_CALUDE_lcm_gcd_48_180_l3531_353188

theorem lcm_gcd_48_180 : 
  (Nat.lcm 48 180 = 720) ∧ (Nat.gcd 48 180 = 12) := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_48_180_l3531_353188


namespace NUMINAMATH_CALUDE_complete_square_l3531_353153

theorem complete_square (x : ℝ) : x^2 - 6*x + 10 = (x - 3)^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_l3531_353153


namespace NUMINAMATH_CALUDE_right_triangle_trig_l3531_353158

theorem right_triangle_trig (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : C = Real.pi / 2) (h3 : Real.sin A = 2 / 3) : Real.cos B = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_trig_l3531_353158


namespace NUMINAMATH_CALUDE_cookie_store_spending_l3531_353172

theorem cookie_store_spending (ben david : ℝ) 
  (h1 : david = ben / 2)
  (h2 : ben = david + 20) : 
  ben + david = 60 := by
sorry

end NUMINAMATH_CALUDE_cookie_store_spending_l3531_353172


namespace NUMINAMATH_CALUDE_smallest_n_for_2007n_mod_1000_l3531_353141

theorem smallest_n_for_2007n_mod_1000 : 
  ∀ n : ℕ+, n < 691 → (2007 * n.val) % 1000 ≠ 837 ∧ (2007 * 691) % 1000 = 837 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_2007n_mod_1000_l3531_353141


namespace NUMINAMATH_CALUDE_race_track_circumference_difference_l3531_353129

/-- The difference in circumferences of two concentric circles, where the outer circle's radius is 8 feet more than the inner circle's radius of 15 feet, is equal to 16π feet. -/
theorem race_track_circumference_difference : 
  let inner_radius : ℝ := 15
  let outer_radius : ℝ := inner_radius + 8
  let inner_circumference : ℝ := 2 * Real.pi * inner_radius
  let outer_circumference : ℝ := 2 * Real.pi * outer_radius
  outer_circumference - inner_circumference = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_race_track_circumference_difference_l3531_353129


namespace NUMINAMATH_CALUDE_inverse_function_problem_l3531_353105

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 2

-- Define the function f
def f (c d x : ℝ) : ℝ := c * x + d

-- State the theorem
theorem inverse_function_problem (c d : ℝ) :
  (∀ x, g x = (Function.invFun (f c d) x) - 5) →
  (Function.invFun (f c d) = Function.invFun (f c d)) →
  7 * c + 3 * d = -14/3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_problem_l3531_353105


namespace NUMINAMATH_CALUDE_russian_players_pairing_probability_l3531_353113

/-- The probability of all Russian players being paired with each other in a tennis tournament -/
theorem russian_players_pairing_probability
  (total_players : ℕ)
  (russian_players : ℕ)
  (h1 : total_players = 10)
  (h2 : russian_players = 4)
  (h3 : russian_players ≤ total_players) :
  (russian_players.choose 2 : ℚ) / (total_players.choose 2) = 1 / 21 :=
sorry

end NUMINAMATH_CALUDE_russian_players_pairing_probability_l3531_353113


namespace NUMINAMATH_CALUDE_ceiling_product_equation_l3531_353174

theorem ceiling_product_equation : ∃! x : ℝ, ⌈x⌉ * x = 168 ∧ x = 168 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_l3531_353174


namespace NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_five_l3531_353192

def numbers : List Nat := [4825, 4835, 4845, 4855, 4865]

def is_divisible_by_five (n : Nat) : Prop :=
  n % 5 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem product_of_digits_not_divisible_by_five :
  ∃ n ∈ numbers,
    ¬is_divisible_by_five n ∧
    ∀ m ∈ numbers, m ≠ n → is_divisible_by_five m ∧
    units_digit n * tens_digit n = 30 :=
  sorry

end NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_five_l3531_353192


namespace NUMINAMATH_CALUDE_cubic_function_root_condition_l3531_353114

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

theorem cubic_function_root_condition (a : ℝ) :
  (∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ < 0) → a > 2 := by
  sorry


end NUMINAMATH_CALUDE_cubic_function_root_condition_l3531_353114


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l3531_353189

theorem quadratic_equation_result (x : ℝ) (h : x^2 - 3*x = 4) : 3*x^2 - 9*x + 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l3531_353189


namespace NUMINAMATH_CALUDE_walking_time_calculation_l3531_353162

/-- A person walks at a constant rate. They cover 36 yards in 18 minutes and have 120 feet left to walk. -/
theorem walking_time_calculation (distance_covered : ℝ) (time_taken : ℝ) (distance_left : ℝ) :
  distance_covered = 36 * 3 →
  time_taken = 18 →
  distance_left = 120 →
  distance_left / (distance_covered / time_taken) = 20 := by
  sorry

end NUMINAMATH_CALUDE_walking_time_calculation_l3531_353162


namespace NUMINAMATH_CALUDE_coefficient_of_x_term_l3531_353150

theorem coefficient_of_x_term (x : ℝ) : 
  let expansion := (x - x + 1)^3
  ∃ a b c d : ℝ, expansion = a*x^3 + b*x^2 + c*x + d ∧ c = -3 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_term_l3531_353150


namespace NUMINAMATH_CALUDE_log_one_fifth_25_l3531_353121

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_one_fifth_25 : log (1/5) 25 = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_one_fifth_25_l3531_353121


namespace NUMINAMATH_CALUDE_point_not_on_graph_l3531_353103

theorem point_not_on_graph : ¬(2 / (2 + 2) = 2 / 3) := by sorry

end NUMINAMATH_CALUDE_point_not_on_graph_l3531_353103


namespace NUMINAMATH_CALUDE_sophies_bakery_purchase_l3531_353160

/-- Sophie's bakery purchase problem -/
theorem sophies_bakery_purchase
  (cupcake_price : ℚ)
  (cupcake_quantity : ℕ)
  (doughnut_price : ℚ)
  (doughnut_quantity : ℕ)
  (cookie_price : ℚ)
  (cookie_quantity : ℕ)
  (pie_slice_price : ℚ)
  (total_spent : ℚ)
  (h1 : cupcake_price = 2)
  (h2 : cupcake_quantity = 5)
  (h3 : doughnut_price = 1)
  (h4 : doughnut_quantity = 6)
  (h5 : cookie_price = 0.6)
  (h6 : cookie_quantity = 15)
  (h7 : pie_slice_price = 2)
  (h8 : total_spent = 33)
  : (total_spent - (cupcake_price * cupcake_quantity + doughnut_price * doughnut_quantity + cookie_price * cookie_quantity)) / pie_slice_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_sophies_bakery_purchase_l3531_353160


namespace NUMINAMATH_CALUDE_pump_emptying_time_l3531_353155

/-- Given a pool and two pumps A and B:
    * Pump A can empty the pool in 4 hours alone
    * Pumps A and B together can empty the pool in 80 minutes
    Prove that pump B can empty the pool in 2 hours alone -/
theorem pump_emptying_time (pool : ℝ) (pump_a pump_b : ℝ → ℝ) :
  (pump_a pool = pool / 4) →  -- Pump A empties the pool in 4 hours
  (pump_a pool + pump_b pool = pool / (80 / 60)) →  -- A and B together empty the pool in 80 minutes
  (pump_b pool = pool / 2) :=  -- Pump B empties the pool in 2 hours
by sorry

end NUMINAMATH_CALUDE_pump_emptying_time_l3531_353155


namespace NUMINAMATH_CALUDE_circle_properties_l3531_353134

-- Define the circle C
def circle_C (D E F : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + D * p.1 + E * p.2 + F = 0}

-- Define the line l
def line_l (D E F : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | D * p.1 + E * p.2 + F = 0}

-- Define the circle M
def circle_M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

theorem circle_properties (D E F : ℝ) 
    (h1 : D^2 + E^2 = F^2) 
    (h2 : F > 0) : 
  F > 4 ∧ 
  (let d := |F - 2| / 2
   let r := Real.sqrt (F^2 - 4*F) / 2
   d^2 - r^2 = 1) ∧
  (∃ M : Set (ℝ × ℝ), M = circle_M ∧ 
    (∀ p ∈ M, p ∈ line_l D E F → False) ∧
    (∀ p ∈ M, p ∈ circle_C D E F → False)) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3531_353134


namespace NUMINAMATH_CALUDE_uncle_jerry_tomatoes_l3531_353152

def tomatoes_problem (yesterday today total : ℕ) : Prop :=
  (yesterday = 120) ∧
  (today = yesterday + 50) ∧
  (total = yesterday + today)

theorem uncle_jerry_tomatoes : ∃ yesterday today total : ℕ,
  tomatoes_problem yesterday today total ∧ total = 290 := by sorry

end NUMINAMATH_CALUDE_uncle_jerry_tomatoes_l3531_353152


namespace NUMINAMATH_CALUDE_not_divisible_by_61_l3531_353182

theorem not_divisible_by_61 (x y : ℕ) 
  (h1 : ¬(61 ∣ x))
  (h2 : ¬(61 ∣ y))
  (h3 : 61 ∣ (7*x + 34*y)) :
  ¬(61 ∣ (5*x + 16*y)) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_61_l3531_353182


namespace NUMINAMATH_CALUDE_max_candy_leftover_l3531_353107

theorem max_candy_leftover (x : ℕ) (h : x > 11) : 
  ∃ (q r : ℕ), x = 11 * q + r ∧ r > 0 ∧ r ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l3531_353107


namespace NUMINAMATH_CALUDE_andrews_age_l3531_353149

theorem andrews_age (grandfather_age andrew_age : ℝ) 
  (h1 : grandfather_age = 9 * andrew_age)
  (h2 : grandfather_age - andrew_age = 63) : 
  andrew_age = 7.875 := by
sorry

end NUMINAMATH_CALUDE_andrews_age_l3531_353149


namespace NUMINAMATH_CALUDE_unique_solution_l3531_353111

def satisfies_equation (x y : ℕ+) : Prop :=
  (x.val ^ 4) * (y.val ^ 4) - 16 * (x.val ^ 2) * (y.val ^ 2) + 15 = 0

theorem unique_solution : 
  ∃! p : ℕ+ × ℕ+, satisfies_equation p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3531_353111


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l3531_353130

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l3531_353130


namespace NUMINAMATH_CALUDE_divisibility_problem_l3531_353198

theorem divisibility_problem (a : ℤ) : 
  0 ≤ a ∧ a < 13 → (12^20 + a) % 13 = 0 → a = 12 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3531_353198


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l3531_353193

theorem wire_cut_ratio (p q : ℝ) (h : p > 0 ∧ q > 0) : 
  (p^2 / 16 = π * (q / (2 * π))^2) → p / q = 4 / Real.sqrt π := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l3531_353193


namespace NUMINAMATH_CALUDE_piravena_round_trip_cost_l3531_353191

/-- Represents the cost of a journey between two cities -/
structure JourneyCost where
  distance : ℝ
  rate : ℝ
  bookingFee : ℝ := 0

def totalCost (journey : JourneyCost) : ℝ :=
  journey.distance * journey.rate + journey.bookingFee

def roundTripCost (outbound outboundRate inbound inboundRate bookingFee : ℝ) : ℝ :=
  totalCost { distance := outbound, rate := outboundRate, bookingFee := bookingFee } +
  totalCost { distance := inbound, rate := inboundRate }

theorem piravena_round_trip_cost :
  let distanceAB : ℝ := 4000
  let distanceAC : ℝ := 3000
  let busRate : ℝ := 0.20
  let planeRate : ℝ := 0.12
  let planeBookingFee : ℝ := 120
  roundTripCost distanceAB planeRate distanceAB busRate planeBookingFee = 1400 := by
  sorry

end NUMINAMATH_CALUDE_piravena_round_trip_cost_l3531_353191


namespace NUMINAMATH_CALUDE_q_value_for_p_seven_l3531_353171

/-- Given the equation Q = 3rP - 6, where r is a constant, prove that if Q = 27 when P = 5, then Q = 40 when P = 7 -/
theorem q_value_for_p_seven (r : ℝ) : 
  (∃ Q : ℝ, Q = 3 * r * 5 - 6 ∧ Q = 27) →
  (∃ Q : ℝ, Q = 3 * r * 7 - 6 ∧ Q = 40) :=
by sorry

end NUMINAMATH_CALUDE_q_value_for_p_seven_l3531_353171


namespace NUMINAMATH_CALUDE_part_one_part_two_l3531_353133

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Part 1
theorem part_one : 
  A 2 ∩ (Set.univ \ B 2) = {x | 2 < x ∧ x ≤ 4 ∨ 5 ≤ x ∧ x < 7} :=
by sorry

-- Part 2
theorem part_two :
  {a : ℝ | a ≠ 1 ∧ A a ∪ B a = A a} = {a | 1 < a ∧ a ≤ 3 ∨ a = -1} :=
by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3531_353133


namespace NUMINAMATH_CALUDE_domain_of_sqrt_tan_plus_sqrt_neg_cos_l3531_353173

theorem domain_of_sqrt_tan_plus_sqrt_neg_cos (x : ℝ) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (∃ y, y = Real.sqrt (Real.tan x) + Real.sqrt (-Real.cos x)) ↔
  x ∈ Set.Ico Real.pi (3 * Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_domain_of_sqrt_tan_plus_sqrt_neg_cos_l3531_353173


namespace NUMINAMATH_CALUDE_ellipse_point_distance_to_y_axis_l3531_353109

/-- Given an ellipse with equation x²/4 + y² = 1 and foci at (-√3, 0) and (√3, 0),
    if a point M(x,y) on the ellipse satisfies the condition that the vectors from
    the foci to M are perpendicular, then the absolute value of x is 2√6/3. -/
theorem ellipse_point_distance_to_y_axis 
  (x y : ℝ) 
  (h_ellipse : x^2/4 + y^2 = 1) 
  (h_perpendicular : (x + Real.sqrt 3) * (x - Real.sqrt 3) + y * y = 0) : 
  |x| = 2 * Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_point_distance_to_y_axis_l3531_353109


namespace NUMINAMATH_CALUDE_fruit_arrangement_count_l3531_353175

-- Define the number of each type of fruit
def num_apples : ℕ := 4
def num_oranges : ℕ := 2
def num_bananas : ℕ := 3

-- Define the total number of fruits
def total_fruits : ℕ := num_apples + num_oranges + num_bananas

-- Theorem statement
theorem fruit_arrangement_count : 
  (Nat.factorial total_fruits) / (Nat.factorial num_apples * Nat.factorial num_oranges * Nat.factorial num_bananas) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_count_l3531_353175


namespace NUMINAMATH_CALUDE_original_profit_margin_is_15_percent_l3531_353117

/-- Represents the profit margin as a real number between 0 and 1 -/
def ProfitMargin : Type := { x : ℝ // 0 ≤ x ∧ x ≤ 1 }

/-- The decrease in purchase price -/
def price_decrease : ℝ := 0.08

/-- The increase in profit margin -/
def margin_increase : ℝ := 0.10

/-- The original profit margin -/
def original_margin : ProfitMargin := ⟨0.15, by sorry⟩

theorem original_profit_margin_is_15_percent :
  ∀ (initial_price : ℝ),
  initial_price > 0 →
  let new_price := initial_price * (1 - price_decrease)
  let new_margin := original_margin.val + margin_increase
  let original_profit := initial_price * original_margin.val
  let new_profit := new_price * new_margin
  original_profit = new_profit := by sorry

end NUMINAMATH_CALUDE_original_profit_margin_is_15_percent_l3531_353117


namespace NUMINAMATH_CALUDE_strawberry_sales_l3531_353151

/-- The number of pints of strawberries sold by a supermarket -/
def pints_sold : ℕ := 54

/-- The revenue from selling strawberries on sale -/
def sale_revenue : ℕ := 216

/-- The revenue that would have been made without the sale -/
def non_sale_revenue : ℕ := 324

/-- The price difference between non-sale and sale price per pint -/
def price_difference : ℕ := 2

theorem strawberry_sales :
  ∃ (sale_price : ℚ),
    sale_price > 0 ∧
    sale_price * pints_sold = sale_revenue ∧
    (sale_price + price_difference) * pints_sold = non_sale_revenue :=
by sorry

end NUMINAMATH_CALUDE_strawberry_sales_l3531_353151
