import Mathlib

namespace NUMINAMATH_CALUDE_divisors_of_18n_cubed_l327_32710

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_18n_cubed (n : ℕ) 
  (h_odd : Odd n) 
  (h_divisors : num_divisors n = 13) : 
  num_divisors (18 * n^3) = 222 := by sorry

end NUMINAMATH_CALUDE_divisors_of_18n_cubed_l327_32710


namespace NUMINAMATH_CALUDE_vector_ac_coordinates_l327_32771

/-- Given two points A and B in 3D space, and a vector AC that is one-third of vector AB,
    prove that AC has specific coordinates. -/
theorem vector_ac_coordinates (A B C : ℝ × ℝ × ℝ) : 
  A = (1, 2, 3) →
  B = (4, 5, 9) →
  C - A = (1/3 : ℝ) • (B - A) →
  C - A = (1, 1, 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_ac_coordinates_l327_32771


namespace NUMINAMATH_CALUDE_negation_P_necessary_not_sufficient_for_negation_Q_l327_32708

def P (x : ℝ) : Prop := |x - 2| ≥ 1

def Q (x : ℝ) : Prop := x^2 - 3*x + 2 ≥ 0

theorem negation_P_necessary_not_sufficient_for_negation_Q :
  (∀ x, ¬(Q x) → ¬(P x)) ∧ 
  (∃ x, ¬(P x) ∧ Q x) :=
by sorry

end NUMINAMATH_CALUDE_negation_P_necessary_not_sufficient_for_negation_Q_l327_32708


namespace NUMINAMATH_CALUDE_remainder_theorem_l327_32747

theorem remainder_theorem (n : ℕ) (h1 : n > 0) (h2 : (n + 1) % 6 = 4) : n % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l327_32747


namespace NUMINAMATH_CALUDE_no_central_ring_numbers_l327_32711

/-- Definition of a central ring number -/
def is_central_ring_number (n : ℕ) : Prop :=
  (1000 ≤ n) ∧ (n < 10000) ∧  -- four-digit number
  (n % 11 ≠ 0) ∧              -- not divisible by 11
  ((n / 1000) % 11 = 0) ∧     -- removing thousands digit
  ((n % 1000 + (n / 10000) * 100) % 11 = 0) ∧  -- removing hundreds digit
  ((n / 100 * 10 + n % 10) % 11 = 0) ∧         -- removing tens digit
  ((n / 10) % 11 = 0)         -- removing ones digit

/-- Theorem: There are no central ring numbers -/
theorem no_central_ring_numbers : ¬∃ n, is_central_ring_number n := by
  sorry

end NUMINAMATH_CALUDE_no_central_ring_numbers_l327_32711


namespace NUMINAMATH_CALUDE_same_row_twice_l327_32723

theorem same_row_twice (num_rows : Nat) (num_people : Nat) :
  num_rows = 7 →
  num_people = 50 →
  ∃ (p1 p2 : Nat) (r : Nat),
    p1 ≠ p2 ∧
    p1 < num_people ∧
    p2 < num_people ∧
    r < num_rows ∧
    (∃ (morning_seating afternoon_seating : Nat → Nat),
      morning_seating p1 = r ∧
      morning_seating p2 = r ∧
      afternoon_seating p1 = r ∧
      afternoon_seating p2 = r) :=
by sorry

end NUMINAMATH_CALUDE_same_row_twice_l327_32723


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocals_equals_sqrt1111_over_112_l327_32796

theorem sqrt_sum_reciprocals_equals_sqrt1111_over_112 :
  Real.sqrt (1 / 25 + 1 / 36 + 1 / 49) = Real.sqrt 1111 / 112 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocals_equals_sqrt1111_over_112_l327_32796


namespace NUMINAMATH_CALUDE_product_72_difference_equals_sum_l327_32774

theorem product_72_difference_equals_sum (P Q R S : ℕ+) : 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S →
  P * Q = 72 →
  R * S = 72 →
  P - Q = R + S →
  P = 12 := by
sorry

end NUMINAMATH_CALUDE_product_72_difference_equals_sum_l327_32774


namespace NUMINAMATH_CALUDE_good_set_closed_under_addition_l327_32725

-- Define a "good set"
def is_good_set (A : Set ℚ) : Prop :=
  (0 ∈ A) ∧ (1 ∈ A) ∧
  (∀ x y, x ∈ A → y ∈ A → (x - y) ∈ A) ∧
  (∀ x, x ∈ A → x ≠ 0 → (1 / x) ∈ A)

-- Theorem statement
theorem good_set_closed_under_addition (A : Set ℚ) (h : is_good_set A) :
  ∀ x y, x ∈ A → y ∈ A → (x + y) ∈ A :=
by sorry

end NUMINAMATH_CALUDE_good_set_closed_under_addition_l327_32725


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l327_32798

/-- The ratio of the area of a square composed of 5 half-squares to the area of a larger square divided into 25 equal parts is 1/10 -/
theorem shaded_area_ratio (large_square_area : ℝ) (small_square_area : ℝ) 
  (h1 : large_square_area > 0)
  (h2 : small_square_area > 0)
  (h3 : large_square_area = 25 * small_square_area)
  (shaded_area : ℝ)
  (h4 : shaded_area = 5 * (small_square_area / 2)) :
  shaded_area / large_square_area = 1 / 10 := by
sorry


end NUMINAMATH_CALUDE_shaded_area_ratio_l327_32798


namespace NUMINAMATH_CALUDE_obtuse_angle_in_second_quadrant_l327_32722

/-- An angle is obtuse if it's greater than 90 degrees and less than 180 degrees -/
def is_obtuse_angle (α : ℝ) : Prop := 90 < α ∧ α < 180

/-- An angle is in the second quadrant if it's greater than 90 degrees and less than or equal to 180 degrees -/
def is_in_second_quadrant (α : ℝ) : Prop := 90 < α ∧ α ≤ 180

/-- Theorem: An obtuse angle is an angle in the second quadrant -/
theorem obtuse_angle_in_second_quadrant (α : ℝ) :
  is_obtuse_angle α → is_in_second_quadrant α :=
by sorry

end NUMINAMATH_CALUDE_obtuse_angle_in_second_quadrant_l327_32722


namespace NUMINAMATH_CALUDE_equation_solution_l327_32755

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x * (x - 2) - 2 * (x + 1)
  ∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 6 ∧ x₂ = 2 - Real.sqrt 6 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l327_32755


namespace NUMINAMATH_CALUDE_incorrect_statement_l327_32717

theorem incorrect_statement : ¬(∀ (p q : Prop), (p ∧ q → False) → (p → False) ∧ (q → False)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l327_32717


namespace NUMINAMATH_CALUDE_complement_union_A_B_complement_A_inter_B_l327_32738

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Theorem for the first part
theorem complement_union_A_B : 
  (Aᶜ ∪ Bᶜ) = {x | x ≤ 2 ∨ x ≥ 10} := by sorry

-- Theorem for the second part
theorem complement_A_inter_B : 
  (Aᶜ ∩ B) = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_complement_A_inter_B_l327_32738


namespace NUMINAMATH_CALUDE_xy_and_sum_of_squares_l327_32748

theorem xy_and_sum_of_squares (x y : ℝ) 
  (sum_eq : x + y = 3) 
  (prod_eq : (x + 2) * (y + 2) = 12) : 
  (xy = 2) ∧ (x^2 + 3*x*y + y^2 = 11) := by
  sorry


end NUMINAMATH_CALUDE_xy_and_sum_of_squares_l327_32748


namespace NUMINAMATH_CALUDE_problem_4_l327_32737

theorem problem_4 (x y : ℝ) (hx : x = 1) (hy : y = 2^100) :
  (x + 2*y)^2 + (x + 2*y)*(x - 2*y) - 4*x*y = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_4_l327_32737


namespace NUMINAMATH_CALUDE_principal_is_2000_l327_32789

/-- Given an interest rate, time period, and total interest, 
    calculates the principal amount borrowed. -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem stating that given the specific conditions, 
    the principal amount borrowed is 2000. -/
theorem principal_is_2000 : 
  let rate : ℚ := 5
  let time : ℕ := 13
  let interest : ℚ := 1300
  calculate_principal rate time interest = 2000 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_2000_l327_32789


namespace NUMINAMATH_CALUDE_greatest_common_divisor_with_same_remainder_l327_32728

theorem greatest_common_divisor_with_same_remainder (a b c : ℕ) (h : a < b ∧ b < c) :
  ∃ (d : ℕ), d > 0 ∧ d = Nat.gcd (b - a) (c - b) ∧
  ∀ (k : ℕ), k > d → ¬(∃ (r : ℕ), a % k = r ∧ b % k = r ∧ c % k = r) := by
  sorry

#check greatest_common_divisor_with_same_remainder 25 57 105

end NUMINAMATH_CALUDE_greatest_common_divisor_with_same_remainder_l327_32728


namespace NUMINAMATH_CALUDE_xyz_value_is_ten_l327_32772

-- Define the variables
variable (a b c x y z : ℂ)

-- State the theorem
theorem xyz_value_is_ten
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : x ≠ 0)
  (h5 : y ≠ 0)
  (h6 : z ≠ 0)
  (h7 : a = (b + c) / (x - 3))
  (h8 : b = (a + c) / (y - 3))
  (h9 : c = (a + b) / (z - 3))
  (h10 : x * y + x * z + y * z = 9)
  (h11 : x + y + z = 6) :
  x * y * z = 10 := by
sorry


end NUMINAMATH_CALUDE_xyz_value_is_ten_l327_32772


namespace NUMINAMATH_CALUDE_tic_tac_toe_losses_l327_32700

theorem tic_tac_toe_losses (total_games wins draws : ℕ) (h1 : total_games = 14) (h2 : wins = 2) (h3 : draws = 10) :
  total_games = wins + (total_games - wins - draws) + draws :=
by sorry

#check tic_tac_toe_losses

end NUMINAMATH_CALUDE_tic_tac_toe_losses_l327_32700


namespace NUMINAMATH_CALUDE_probability_units_digit_less_than_3_l327_32759

/-- A five-digit even integer -/
def FiveDigitEven : Type := { n : ℕ // 10000 ≤ n ∧ n < 100000 ∧ n % 2 = 0 }

/-- The set of possible units digits for even numbers -/
def EvenUnitsDigits : Finset ℕ := {0, 2, 4, 6, 8}

/-- The set of units digits less than 3 -/
def UnitsDigitsLessThan3 : Finset ℕ := {0, 2}

/-- The probability of a randomly chosen five-digit even integer having a units digit less than 3 -/
theorem probability_units_digit_less_than_3 :
  (Finset.card UnitsDigitsLessThan3 : ℚ) / (Finset.card EvenUnitsDigits : ℚ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_units_digit_less_than_3_l327_32759


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l327_32727

theorem complete_square_quadratic :
  ∀ (c d : ℝ), (∀ x : ℝ, x^2 + 14*x + 24 = 0 ↔ (x + c)^2 = d) → d = 25 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l327_32727


namespace NUMINAMATH_CALUDE_f_equals_g_l327_32749

-- Define the functions
def f (x : ℝ) : ℝ := (76 * x^6)^7
def g (x : ℝ) : ℝ := |x|

-- State the theorem
theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l327_32749


namespace NUMINAMATH_CALUDE_binomial_ratio_equals_one_l327_32719

-- Define the binomial coefficient for real numbers
noncomputable def binomial (r : ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1
  else (r * binomial (r - 1) (k - 1)) / k

-- State the theorem
theorem binomial_ratio_equals_one :
  (binomial (1/2 : ℝ) 1000 * 4^1000) / binomial 2000 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_ratio_equals_one_l327_32719


namespace NUMINAMATH_CALUDE_f_of_2_eq_neg_2_l327_32757

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x

-- State the theorem
theorem f_of_2_eq_neg_2 : f 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_eq_neg_2_l327_32757


namespace NUMINAMATH_CALUDE_divides_n_squared_plus_2n_plus_27_l327_32793

theorem divides_n_squared_plus_2n_plus_27 (n : ℕ) :
  n ∣ (n^2 + 2*n + 27) ↔ n = 1 ∨ n = 3 ∨ n = 9 ∨ n = 27 := by
  sorry

end NUMINAMATH_CALUDE_divides_n_squared_plus_2n_plus_27_l327_32793


namespace NUMINAMATH_CALUDE_chord_with_midpoint_A_no_chord_with_midpoint_B_l327_32769

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define a chord of the hyperbola
def is_chord (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  hyperbola x₁ y₁ ∧ hyperbola x₂ y₂

-- Define the midpoint of a chord
def is_midpoint (x y x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2

-- Theorem 1: Chord with midpoint A(2,1)
theorem chord_with_midpoint_A :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_chord x₁ y₁ x₂ y₂ ∧
    is_midpoint 2 1 x₁ y₁ x₂ y₂ ∧
    ∀ (x y : ℝ), y = 6*x - 11 ↔ ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
sorry

-- Theorem 2: No chord with midpoint B(1,1)
theorem no_chord_with_midpoint_B :
  ¬∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_chord x₁ y₁ x₂ y₂ ∧
    is_midpoint 1 1 x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_chord_with_midpoint_A_no_chord_with_midpoint_B_l327_32769


namespace NUMINAMATH_CALUDE_inequality_solution_set_l327_32703

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | a * x + 1 < a^2 + x}
  (a > 1 → S = {x : ℝ | x < a + 1}) ∧
  (a < 1 → S = {x : ℝ | x > a + 1}) ∧
  (a = 1 → S = ∅) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l327_32703


namespace NUMINAMATH_CALUDE_single_circle_percentage_l327_32766

/-- The number of children participating in the game -/
def n : ℕ := 10

/-- Calculates the double factorial of a natural number -/
def double_factorial (k : ℕ) : ℕ :=
  if k ≤ 1 then 1 else k * double_factorial (k - 2)

/-- Calculates the number of configurations where n children form a single circle -/
def single_circle_configs (n : ℕ) : ℕ := double_factorial (2 * n - 2)

/-- Calculates the total number of possible configurations for n children -/
def total_configs (n : ℕ) : ℕ := 387099936  -- This is the precomputed value for n = 10

/-- The main theorem to be proved -/
theorem single_circle_percentage :
  (single_circle_configs n : ℚ) / (total_configs n) = 12 / 25 := by
  sorry

#eval (single_circle_configs n : ℚ) / (total_configs n)

end NUMINAMATH_CALUDE_single_circle_percentage_l327_32766


namespace NUMINAMATH_CALUDE_angle_cde_is_eleven_degrees_l327_32794

/-- Given a configuration in a rectangle where:
    - Angle ACB = 80°
    - Angle FEG = 64°
    - Angle DCE = 86°
    - Angle DEC = 83°
    Prove that angle CDE (θ) is equal to 11°. -/
theorem angle_cde_is_eleven_degrees 
  (angle_ACB : ℝ) (angle_FEG : ℝ) (angle_DCE : ℝ) (angle_DEC : ℝ)
  (h1 : angle_ACB = 80)
  (h2 : angle_FEG = 64)
  (h3 : angle_DCE = 86)
  (h4 : angle_DEC = 83) :
  180 - angle_DCE - angle_DEC = 11 := by
  sorry

end NUMINAMATH_CALUDE_angle_cde_is_eleven_degrees_l327_32794


namespace NUMINAMATH_CALUDE_smallest_multiple_thirty_six_satisfies_thirty_six_is_smallest_l327_32792

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 400 * x % 576 = 0 → x ≥ 36 :=
  sorry

theorem thirty_six_satisfies : 400 * 36 % 576 = 0 :=
  sorry

theorem thirty_six_is_smallest : ∃ (x : ℕ), x > 0 ∧ 400 * x % 576 = 0 ∧ x = 36 :=
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_thirty_six_satisfies_thirty_six_is_smallest_l327_32792


namespace NUMINAMATH_CALUDE_katy_summer_reading_l327_32743

/-- The number of books Katy read in June -/
def june_books : ℕ := 8

/-- The number of books Katy read in July -/
def july_books : ℕ := 2 * june_books

/-- The number of books Katy read in August -/
def august_books : ℕ := july_books - 3

/-- The total number of books Katy read during the summer -/
def total_summer_books : ℕ := june_books + july_books + august_books

/-- Theorem stating that Katy read 37 books during the summer -/
theorem katy_summer_reading : total_summer_books = 37 := by
  sorry

end NUMINAMATH_CALUDE_katy_summer_reading_l327_32743


namespace NUMINAMATH_CALUDE_sqrt_three_cubed_l327_32701

theorem sqrt_three_cubed : Real.sqrt 3 ^ 3 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_cubed_l327_32701


namespace NUMINAMATH_CALUDE_sin_two_phi_value_l327_32702

theorem sin_two_phi_value (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) :
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_phi_value_l327_32702


namespace NUMINAMATH_CALUDE_silver_division_problem_l327_32752

theorem silver_division_problem (x y : ℤ) : 
  y = 7 * x + 4 ∧ y = 9 * x - 8 → y = 46 := by
sorry

end NUMINAMATH_CALUDE_silver_division_problem_l327_32752


namespace NUMINAMATH_CALUDE_fraction_identity_l327_32768

theorem fraction_identity (m : ℕ) (hm : m > 0) :
  (1 : ℚ) / (m * (m + 1)) = 1 / m - 1 / (m + 1) ∧
  (1 : ℚ) / (6 * 7) = 1 / 6 - 1 / 7 ∧
  ∃ (x : ℚ), x = 4 ∧ 1 / ((x - 1) * (x - 2)) + 1 / (x * (x - 1)) = 1 / x :=
by sorry

end NUMINAMATH_CALUDE_fraction_identity_l327_32768


namespace NUMINAMATH_CALUDE_age_digits_product_l327_32730

/-- A function that returns the digits of a two-digit number -/
def digits (n : ℕ) : List ℕ :=
  [n / 10, n % 10]

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

/-- A function that checks if a number is a power of another number -/
def isPowerOf (n : ℕ) (base : ℕ) : Prop :=
  ∃ k : ℕ, n = base ^ k

/-- A function that calculates the sum of a list of numbers -/
def sum (l : List ℕ) : ℕ :=
  l.foldl (·+·) 0

/-- A function that calculates the product of a list of numbers -/
def product (l : List ℕ) : ℕ :=
  l.foldl (·*·) 1

theorem age_digits_product : 
  ∃ (x y : ℕ),
    isTwoDigit x ∧ 
    isTwoDigit y ∧ 
    isPowerOf x 5 ∧ 
    isPowerOf y 2 ∧ 
    Odd (sum (digits x ++ digits y)) → 
    product (digits x ++ digits y) = 240 := by
  sorry

end NUMINAMATH_CALUDE_age_digits_product_l327_32730


namespace NUMINAMATH_CALUDE_indian_teepee_proportion_l327_32735

/-- Represents the fraction of drawings with a specific combination of person and dwelling -/
structure DrawingFraction :=
  (eskimo_teepee : ℚ)
  (eskimo_igloo : ℚ)
  (indian_igloo : ℚ)
  (indian_teepee : ℚ)

/-- The conditions given in the problem -/
def problem_conditions (df : DrawingFraction) : Prop :=
  df.eskimo_teepee + df.eskimo_igloo + df.indian_igloo + df.indian_teepee = 1 ∧
  df.indian_teepee + df.indian_igloo = 2 * (df.eskimo_teepee + df.eskimo_igloo) ∧
  df.indian_igloo = df.eskimo_teepee ∧
  df.eskimo_igloo = 3 * df.eskimo_teepee

/-- The theorem to be proved -/
theorem indian_teepee_proportion (df : DrawingFraction) :
  problem_conditions df →
  df.indian_teepee / (df.indian_teepee + df.eskimo_teepee) = 7/8 :=
by sorry

end NUMINAMATH_CALUDE_indian_teepee_proportion_l327_32735


namespace NUMINAMATH_CALUDE_polynomial_inequality_implies_linear_l327_32770

/-- A polynomial function from ℝ to ℝ -/
def RealPolynomial := ℝ → ℝ

/-- The property that the polynomial satisfies the given inequality -/
def SatisfiesInequality (f : RealPolynomial) : Prop :=
  ∀ x y : ℝ, 2 * y * f (x + y) + (x - y) * (f x + f y) ≥ 0

/-- The theorem stating that if a real polynomial satisfies the inequality, 
    it must be of the form f(x) = cx for some non-negative c -/
theorem polynomial_inequality_implies_linear 
  (f : RealPolynomial) (h : SatisfiesInequality f) :
  ∃ c : ℝ, c ≥ 0 ∧ ∀ x : ℝ, f x = c * x :=
sorry

end NUMINAMATH_CALUDE_polynomial_inequality_implies_linear_l327_32770


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l327_32758

theorem cube_root_equation_solution :
  ∃! x : ℝ, (10 - x / 3) ^ (1/3 : ℝ) = -4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l327_32758


namespace NUMINAMATH_CALUDE_red_cars_count_l327_32781

theorem red_cars_count (black_cars : ℕ) (ratio_red : ℕ) (ratio_black : ℕ) : 
  black_cars = 90 → ratio_red = 3 → ratio_black = 8 → 
  ∃ red_cars : ℕ, red_cars * ratio_black = black_cars * ratio_red ∧ red_cars = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_red_cars_count_l327_32781


namespace NUMINAMATH_CALUDE_nonzero_real_solution_l327_32778

theorem nonzero_real_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 12) (h2 : y + 1 / x = 3 / 8) :
  x = 4 ∨ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_real_solution_l327_32778


namespace NUMINAMATH_CALUDE_min_value_product_l327_32740

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a/b + b/c + c/a + b/a + c/b + a/c = 10)
  (h2 : a^2 + b^2 + c^2 = 9) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 91/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l327_32740


namespace NUMINAMATH_CALUDE_painted_cubes_count_l327_32780

/-- Represents a cube with side length n --/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted cube --/
structure PaintedCube (n : ℕ) extends Cube n where
  painted_faces : ℕ := 6

/-- Represents a cube that has been cut into smaller cubes --/
structure CutCube (n m : ℕ) extends PaintedCube n where
  cut_size : ℕ := m

/-- The number of smaller cubes with at least two painted faces in a cut painted cube --/
def cubes_with_two_plus_painted_faces (c : CutCube 4 1) : ℕ := 32

/-- Theorem stating that a 4x4x4 painted cube cut into 1x1x1 cubes has 32 smaller cubes with at least two painted faces --/
theorem painted_cubes_count (c : CutCube 4 1) : 
  cubes_with_two_plus_painted_faces c = 32 := by sorry

end NUMINAMATH_CALUDE_painted_cubes_count_l327_32780


namespace NUMINAMATH_CALUDE_a_eq_zero_necessary_not_sufficient_l327_32704

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero -/
def IsPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem a_eq_zero_necessary_not_sufficient :
  ∀ (z : ℂ), 
  (IsPurelyImaginary z → z.re = 0) ∧ 
  ∃ (z : ℂ), z.re = 0 ∧ ¬IsPurelyImaginary z :=
by sorry

end NUMINAMATH_CALUDE_a_eq_zero_necessary_not_sufficient_l327_32704


namespace NUMINAMATH_CALUDE_ellipse_product_l327_32744

-- Define the ellipse C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 5 + p.2^2 = 1}

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (2, 0)
def F₂ : ℝ × ℝ := (-2, 0)

-- State the theorem
theorem ellipse_product (P : ℝ × ℝ) 
  (h_on_ellipse : P ∈ C) 
  (h_perpendicular : (P.1 - F₁.1, P.2 - F₁.2) • (P.1 - F₂.1, P.2 - F₂.2) = 0) :
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * 
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_product_l327_32744


namespace NUMINAMATH_CALUDE_population_growth_rate_l327_32714

/-- Given a population increase of 160 persons in 40 minutes, 
    proves that the time taken for one person to be added is 15 seconds. -/
theorem population_growth_rate (persons : ℕ) (minutes : ℕ) (seconds_per_person : ℕ) : 
  persons = 160 ∧ minutes = 40 → seconds_per_person = 15 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_rate_l327_32714


namespace NUMINAMATH_CALUDE_museum_visit_arrangements_l327_32713

theorem museum_visit_arrangements (n m : ℕ) (hn : n = 6) (hm : m = 6) : 
  (Nat.choose n 2) * (m - 1) ^ (n - 2) = 15 * 625 := by
  sorry

end NUMINAMATH_CALUDE_museum_visit_arrangements_l327_32713


namespace NUMINAMATH_CALUDE_ramen_bread_intersection_l327_32750

theorem ramen_bread_intersection (total : ℕ) (ramen : ℕ) (bread : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : ramen = 289)
  (h3 : bread = 337)
  (h4 : neither = 56) :
  ramen + bread - total + neither = 182 :=
by sorry

end NUMINAMATH_CALUDE_ramen_bread_intersection_l327_32750


namespace NUMINAMATH_CALUDE_probability_differ_by_2_l327_32715

/-- A standard 6-sided die -/
def Die : Type := Fin 6

/-- The set of all possible outcomes when rolling a die twice -/
def TwoRolls : Type := Die × Die

/-- The condition for two rolls to differ by 2 -/
def DifferBy2 (roll : TwoRolls) : Prop :=
  (roll.1.val + 1 = roll.2.val) ∨ (roll.1.val = roll.2.val + 1)

/-- The number of favorable outcomes -/
def FavorableOutcomes : ℕ := 8

/-- The total number of possible outcomes -/
def TotalOutcomes : ℕ := 36

theorem probability_differ_by_2 :
  (FavorableOutcomes : ℚ) / TotalOutcomes = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_differ_by_2_l327_32715


namespace NUMINAMATH_CALUDE_students_per_group_l327_32754

theorem students_per_group 
  (total_students : ℕ) 
  (students_not_picked : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 64) 
  (h2 : students_not_picked = 36) 
  (h3 : num_groups = 4) : 
  (total_students - students_not_picked) / num_groups = 7 := by
sorry

end NUMINAMATH_CALUDE_students_per_group_l327_32754


namespace NUMINAMATH_CALUDE_ethanol_mixture_optimization_l327_32785

theorem ethanol_mixture_optimization (initial_volume : ℝ) (initial_ethanol_percentage : ℝ)
  (added_ethanol : ℝ) (final_ethanol_percentage : ℝ) :
  initial_volume = 45 →
  initial_ethanol_percentage = 0.05 →
  added_ethanol = 2.5 →
  final_ethanol_percentage = 0.1 →
  (initial_volume * initial_ethanol_percentage + added_ethanol) /
    (initial_volume + added_ethanol) = final_ethanol_percentage :=
by sorry

end NUMINAMATH_CALUDE_ethanol_mixture_optimization_l327_32785


namespace NUMINAMATH_CALUDE_square_sum_eq_841_times_product_plus_one_l327_32706

theorem square_sum_eq_841_times_product_plus_one :
  ∀ a b : ℕ, a^2 + b^2 = 841 * (a * b + 1) ↔ (a = 0 ∧ b = 29) ∨ (a = 29 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_eq_841_times_product_plus_one_l327_32706


namespace NUMINAMATH_CALUDE_bananas_left_in_jar_l327_32746

theorem bananas_left_in_jar (original : ℕ) (removed : ℕ) (h1 : original = 46) (h2 : removed = 5) :
  original - removed = 41 := by
  sorry

end NUMINAMATH_CALUDE_bananas_left_in_jar_l327_32746


namespace NUMINAMATH_CALUDE_number_relationship_l327_32742

theorem number_relationship (a b c d : ℝ) 
  (h1 : a < b) 
  (h2 : d < c) 
  (h3 : (c - a) * (c - b) < 0) 
  (h4 : (d - a) * (d - b) > 0) : 
  d < a ∧ a < c ∧ c < b :=
by sorry

end NUMINAMATH_CALUDE_number_relationship_l327_32742


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l327_32775

theorem rectangular_garden_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 588 →
  width = 14 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l327_32775


namespace NUMINAMATH_CALUDE_third_term_is_16_l327_32732

/-- Geometric sequence with common ratio 2 and sum of first 4 terms equal to 60 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 2 * a n) ∧ 
  (a 1 + a 2 + a 3 + a 4 = 60)

/-- The third term of the geometric sequence is 16 -/
theorem third_term_is_16 (a : ℕ → ℝ) (h : geometric_sequence a) : a 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_16_l327_32732


namespace NUMINAMATH_CALUDE_inequality_solution_set_l327_32721

theorem inequality_solution_set (x : ℝ) (h : x ≠ 1) :
  (2 * x - 1) / (x - 1) ≥ 1 ↔ x ≤ 0 ∨ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l327_32721


namespace NUMINAMATH_CALUDE_complete_square_sum_l327_32797

theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_complete_square_sum_l327_32797


namespace NUMINAMATH_CALUDE_half_liar_day_determination_l327_32709

-- Define the days of the week
inductive Day : Type
  | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define the half-liar's statement type
structure Statement where
  yesterday : Day
  tomorrow : Day

-- Define the function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

-- Define the function to get the previous day
def prevDay (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Sunday
  | Day.Tuesday => Day.Monday
  | Day.Wednesday => Day.Tuesday
  | Day.Thursday => Day.Wednesday
  | Day.Friday => Day.Thursday
  | Day.Saturday => Day.Friday
  | Day.Sunday => Day.Saturday

-- Define the theorem
theorem half_liar_day_determination
  (statement_week_ago : Statement)
  (statement_today : Statement)
  (h1 : statement_week_ago.yesterday = Day.Wednesday ∧ statement_week_ago.tomorrow = Day.Thursday)
  (h2 : statement_today.yesterday = Day.Friday ∧ statement_today.tomorrow = Day.Sunday)
  (h3 : ∀ (d : Day), nextDay (nextDay (nextDay (nextDay (nextDay (nextDay (nextDay d)))))) = d)
  : ∃ (today : Day), today = Day.Saturday :=
by
  sorry


end NUMINAMATH_CALUDE_half_liar_day_determination_l327_32709


namespace NUMINAMATH_CALUDE_oblique_square_area_theorem_main_theorem_l327_32753

/-- Represents a square in an oblique projection --/
structure ObliqueSquare where
  side : ℝ
  projectedSide : ℝ

/-- The area of the original square given its oblique projection --/
def originalArea (s : ObliqueSquare) : ℝ := s.side * s.side

/-- Theorem stating that for a square with a projected side of 4 units,
    the area of the original square can be either 16 or 64 --/
theorem oblique_square_area_theorem (s : ObliqueSquare) 
  (h : s.projectedSide = 4) :
  originalArea s = 16 ∨ originalArea s = 64 := by
  sorry

/-- Main theorem combining the above results --/
theorem main_theorem : 
  ∃ (s1 s2 : ObliqueSquare), 
    s1.projectedSide = 4 ∧ 
    s2.projectedSide = 4 ∧ 
    originalArea s1 = 16 ∧ 
    originalArea s2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_oblique_square_area_theorem_main_theorem_l327_32753


namespace NUMINAMATH_CALUDE_hope_project_protractors_l327_32731

theorem hope_project_protractors :
  ∀ (x y z : ℕ),
  x > 31 ∧ z > 33 ∧
  10 * x + 15 * y + 20 * z = 1710 ∧
  8 * x + 2 * y + 8 * z = 664 →
  6 * x + 7 * y + 5 * z = 680 :=
by sorry

end NUMINAMATH_CALUDE_hope_project_protractors_l327_32731


namespace NUMINAMATH_CALUDE_gcd_g_10_g_13_l327_32724

def g (x : ℤ) : ℤ := x^3 - 3*x^2 + x + 2050

theorem gcd_g_10_g_13 : Int.gcd (g 10) (g 13) = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_g_10_g_13_l327_32724


namespace NUMINAMATH_CALUDE_days_until_birthday_l327_32762

/-- Proof of the number of days until Maria's birthday --/
theorem days_until_birthday (daily_savings : ℕ) (flower_cost : ℕ) (flowers_bought : ℕ) :
  daily_savings = 2 →
  flower_cost = 4 →
  flowers_bought = 11 →
  (flowers_bought * flower_cost) / daily_savings = 22 :=
by sorry

end NUMINAMATH_CALUDE_days_until_birthday_l327_32762


namespace NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l327_32784

theorem x_fourth_plus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 + 1/x^4 = 727 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l327_32784


namespace NUMINAMATH_CALUDE_circle_triangle_intersection_l327_32764

/-- Given an equilateral triangle intersected by a circle at six points, 
    this theorem proves the length of DE based on other given lengths. -/
theorem circle_triangle_intersection (AG GF FC HJ : ℝ) (h1 : AG = 2) (h2 : GF = 13) 
  (h3 : FC = 1) (h4 : HJ = 7) : ∃ (DE : ℝ), DE = 2 * Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_intersection_l327_32764


namespace NUMINAMATH_CALUDE_greatest_valid_number_l327_32716

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  (n / 10) * (n % 10) = 12 ∧
  (n / 10) < (n % 10)

theorem greatest_valid_number :
  ∀ m : ℕ, is_valid_number m → m ≤ 34 :=
by sorry

end NUMINAMATH_CALUDE_greatest_valid_number_l327_32716


namespace NUMINAMATH_CALUDE_pentagonal_prism_coloring_l327_32779

structure PentagonalPrism where
  vertices : Fin 10 → Point
  color : Fin 45 → Color

inductive Color
  | Red
  | Blue

def isEdge (i j : Fin 10) : Bool :=
  (i < j ∧ (i.val + 1 = j.val ∨ (i.val = 4 ∧ j.val = 0) ∨ (i.val = 9 ∧ j.val = 5))) ∨
  (j < i ∧ (j.val + 1 = i.val ∨ (j.val = 4 ∧ i.val = 0) ∨ (j.val = 9 ∧ i.val = 5)))

def isTopFaceEdge (i j : Fin 10) : Bool :=
  i < 5 ∧ j < 5 ∧ isEdge i j

def isBottomFaceEdge (i j : Fin 10) : Bool :=
  i ≥ 5 ∧ j ≥ 5 ∧ isEdge i j

def getEdgeColor (p : PentagonalPrism) (i j : Fin 10) : Color :=
  if i < j then p.color ⟨i.val * 9 + j.val - (i.val * (i.val + 1) / 2), sorry⟩
  else p.color ⟨j.val * 9 + i.val - (j.val * (j.val + 1) / 2), sorry⟩

def noMonochromaticTriangle (p : PentagonalPrism) : Prop :=
  ∀ i j k : Fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ¬(getEdgeColor p i j = getEdgeColor p j k ∧ getEdgeColor p j k = getEdgeColor p i k)

theorem pentagonal_prism_coloring (p : PentagonalPrism) 
  (h : noMonochromaticTriangle p) :
  (∀ i j : Fin 10, isTopFaceEdge i j → getEdgeColor p i j = getEdgeColor p 0 1) ∧
  (∀ i j : Fin 10, isBottomFaceEdge i j → getEdgeColor p i j = getEdgeColor p 5 6) :=
sorry

end NUMINAMATH_CALUDE_pentagonal_prism_coloring_l327_32779


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l327_32787

/-- The quadratic equation 2qx^2 - 20x + 5 = 0 has only one solution when q = 10 -/
theorem unique_solution_quadratic :
  ∃! (q : ℝ), q ≠ 0 ∧ (∃! x, 2 * q * x^2 - 20 * x + 5 = 0) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l327_32787


namespace NUMINAMATH_CALUDE_coefficient_of_x_l327_32705

theorem coefficient_of_x (a x y : ℝ) : 
  a * x + y = 19 →
  x + 3 * y = 1 →
  3 * x + 2 * y = 10 →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l327_32705


namespace NUMINAMATH_CALUDE_gcf_540_196_l327_32760

theorem gcf_540_196 : Nat.gcd 540 196 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcf_540_196_l327_32760


namespace NUMINAMATH_CALUDE_toucan_problem_l327_32783

theorem toucan_problem (initial_toucans : ℕ) : 
  (initial_toucans + 1 = 3) → initial_toucans = 2 := by
sorry

end NUMINAMATH_CALUDE_toucan_problem_l327_32783


namespace NUMINAMATH_CALUDE_pre_bought_tickets_l327_32763

/-- The number of people who pre-bought plane tickets -/
def num_pre_buyers : ℕ := 20

/-- The price of a pre-bought ticket -/
def pre_bought_price : ℕ := 155

/-- The price of a ticket bought at the gate -/
def gate_price : ℕ := 200

/-- The number of people who bought tickets at the gate -/
def num_gate_buyers : ℕ := 30

/-- The difference in total amount paid between gate buyers and pre-buyers -/
def price_difference : ℕ := 2900

theorem pre_bought_tickets : 
  num_pre_buyers * pre_bought_price + price_difference = num_gate_buyers * gate_price := by
  sorry

end NUMINAMATH_CALUDE_pre_bought_tickets_l327_32763


namespace NUMINAMATH_CALUDE_total_cost_for_six_people_l327_32799

/-- The total cost of buying soda and pizza for a group -/
def total_cost (num_people : ℕ) (soda_price pizza_price : ℚ) : ℚ :=
  num_people * (soda_price + pizza_price)

/-- Theorem: The total cost for 6 people is $9.00 -/
theorem total_cost_for_six_people :
  total_cost 6 (1/2) 1 = 9 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_for_six_people_l327_32799


namespace NUMINAMATH_CALUDE_smallest_winning_number_l327_32790

def ian_action (x : ℕ) : ℕ := 3 * x

def marcella_action (x : ℕ) : ℕ := x + 150

def game_sequence (m : ℕ) : ℕ := 
  ian_action (marcella_action (ian_action (marcella_action (ian_action m))))

theorem smallest_winning_number : 
  ∀ m : ℕ, 0 ≤ m ∧ m ≤ 1999 →
    (m < 112 → 
      game_sequence m ≤ 5000 ∧ 
      marcella_action (game_sequence m) ≤ 5000 ∧ 
      ian_action (marcella_action (game_sequence m)) > 5000) →
    (game_sequence 112 ≤ 5000 ∧ 
     marcella_action (game_sequence 112) ≤ 5000 ∧ 
     ian_action (marcella_action (game_sequence 112)) > 5000) :=
by sorry

#check smallest_winning_number

end NUMINAMATH_CALUDE_smallest_winning_number_l327_32790


namespace NUMINAMATH_CALUDE_total_degrees_theorem_l327_32765

/-- Represents the budget allocation percentages for different sectors -/
structure BudgetAllocation where
  microphotonics : Float
  homeElectronics : Float
  foodAdditives : Float
  geneticallyModifiedMicroorganisms : Float
  industrialLubricants : Float
  artificialIntelligence : Float
  nanotechnology : Float

/-- Calculates the degrees in a circle graph for a given percentage -/
def percentageToDegrees (percentage : Float) : Float :=
  percentage * 3.6

/-- Calculates the total degrees for basic astrophysics, artificial intelligence, and nanotechnology -/
def totalDegrees (allocation : BudgetAllocation) : Float :=
  let basicAstrophysics := 100 - (allocation.microphotonics + allocation.homeElectronics + 
    allocation.foodAdditives + allocation.geneticallyModifiedMicroorganisms + 
    allocation.industrialLubricants + allocation.artificialIntelligence + allocation.nanotechnology)
  percentageToDegrees basicAstrophysics + 
  percentageToDegrees allocation.artificialIntelligence + 
  percentageToDegrees allocation.nanotechnology

/-- Theorem: The total degrees for basic astrophysics, artificial intelligence, and nanotechnology is 117.36 -/
theorem total_degrees_theorem (allocation : BudgetAllocation) 
  (h1 : allocation.microphotonics = 12.3)
  (h2 : allocation.homeElectronics = 17.8)
  (h3 : allocation.foodAdditives = 9.4)
  (h4 : allocation.geneticallyModifiedMicroorganisms = 21.7)
  (h5 : allocation.industrialLubricants = 6.2)
  (h6 : allocation.artificialIntelligence = 4.1)
  (h7 : allocation.nanotechnology = 5.3) :
  totalDegrees allocation = 117.36 := by
  sorry

end NUMINAMATH_CALUDE_total_degrees_theorem_l327_32765


namespace NUMINAMATH_CALUDE_fish_added_l327_32739

theorem fish_added (initial_fish final_fish : ℕ) (h1 : initial_fish = 10) (h2 : final_fish = 13) :
  final_fish - initial_fish = 3 := by
  sorry

end NUMINAMATH_CALUDE_fish_added_l327_32739


namespace NUMINAMATH_CALUDE_inequality_proof_l327_32795

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : b + d < a + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l327_32795


namespace NUMINAMATH_CALUDE_sqrt_360000_equals_600_l327_32767

theorem sqrt_360000_equals_600 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_equals_600_l327_32767


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l327_32712

/-- Given a triangle DEF where the measure of angle D is three times the measure of angle F,
    and angle F measures 18°, prove that the measure of angle E is 108°. -/
theorem angle_measure_in_triangle (D E F : ℝ) (h1 : D = 3 * F) (h2 : F = 18) :
  E = 108 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l327_32712


namespace NUMINAMATH_CALUDE_max_product_sum_11_l327_32729

theorem max_product_sum_11 :
  ∃ (a b : ℕ), a + b = 11 ∧
  ∀ (x y : ℕ), x + y = 11 → x * y ≤ a * b ∧
  a * b = 30 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_11_l327_32729


namespace NUMINAMATH_CALUDE_tangent_line_slope_l327_32734

/-- A line passing through the origin and tangent to the circle (x - √3)² + (y - 1)² = 1 has a slope of either 0 or √3 -/
theorem tangent_line_slope :
  ∀ k : ℝ,
  (∃ x y : ℝ, y = k * x ∧ (x - Real.sqrt 3)^2 + (y - 1)^2 = 1) →
  (k = 0 ∨ k = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l327_32734


namespace NUMINAMATH_CALUDE_square_roots_problem_l327_32761

theorem square_roots_problem (x : ℝ) (h : x > x - 6) :
  (x ^ 2 = (x - 6) ^ 2) → x ^ 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l327_32761


namespace NUMINAMATH_CALUDE_unique_function_exists_l327_32736

/-- A function satisfying the given inequality for all real x, y, z and fixed positive integer k -/
def SatisfiesInequality (f : ℝ → ℝ) (k : ℕ+) : Prop :=
  ∀ x y z : ℝ, f (x * y) + f (x + z) + k * f x * f (y * z) ≥ k^2

/-- There exists only one function satisfying the inequality -/
theorem unique_function_exists (k : ℕ+) : ∃! f : ℝ → ℝ, SatisfiesInequality f k := by
  sorry

end NUMINAMATH_CALUDE_unique_function_exists_l327_32736


namespace NUMINAMATH_CALUDE_center_transformation_l327_32786

/-- Reflects a point across the y-axis -/
def reflectY (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Rotates a point 90 degrees clockwise around the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

/-- Translates a point up by a given amount -/
def translateUp (p : ℝ × ℝ) (amount : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + amount)

/-- Applies all transformations to a point -/
def applyTransformations (p : ℝ × ℝ) : ℝ × ℝ :=
  translateUp (rotate90Clockwise (reflectY p)) 4

theorem center_transformation :
  applyTransformations (3, -5) = (-5, 7) := by
  sorry

end NUMINAMATH_CALUDE_center_transformation_l327_32786


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l327_32791

/-- Given an ellipse defined by the equation 16(x+2)^2 + 4(y-3)^2 = 64,
    prove that the distance between an endpoint of its major axis
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance :
  ∀ (C D : ℝ × ℝ),
  (∀ (x y : ℝ), 16 * (x + 2)^2 + 4 * (y - 3)^2 = 64 →
    (C.1 + 2)^2 / 4 + (C.2 - 3)^2 / 16 = 1 ∧
    (D.1 + 2)^2 / 4 + (D.2 - 3)^2 / 16 = 1 ∧
    ((C.1 + 2)^2 / 4 = 1 ∨ (C.2 - 3)^2 / 16 = 1) ∧
    ((D.1 + 2)^2 / 4 = 1 ∨ (D.2 - 3)^2 / 16 = 1) ∧
    (C.1 + 2)^2 / 4 ≠ (D.1 + 2)^2 / 4) →
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l327_32791


namespace NUMINAMATH_CALUDE_parabola_segment_sum_l327_32776

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 12*y

-- Define the focus F (we don't know its exact coordinates, so we leave it abstract)
variable (F : ℝ × ℝ)

-- Define points A, B, and P
variable (A B : ℝ × ℝ)
def P : ℝ × ℝ := (2, 1)

-- State that A and B are on the parabola
axiom A_on_parabola : parabola A.1 A.2
axiom B_on_parabola : parabola B.1 B.2

-- State that P is the midpoint of AB
axiom P_is_midpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- State the theorem
theorem parabola_segment_sum : 
  ∀ (F A B : ℝ × ℝ), 
  parabola A.1 A.2 → 
  parabola B.1 B.2 → 
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  dist A F + dist B F = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_segment_sum_l327_32776


namespace NUMINAMATH_CALUDE_unique_solution_l327_32733

/-- The function g(y) defined in the original problem -/
def g (y : ℝ) : ℝ := (15 * y + (15 * y + 8) ^ (1/3)) ^ (1/3)

/-- The equation from the original problem -/
def equation (y : ℝ) : Prop := g y = 8

theorem unique_solution :
  ∃! y : ℝ, equation y ∧ y = 168/5 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l327_32733


namespace NUMINAMATH_CALUDE_circle_line_intersection_l327_32707

theorem circle_line_intersection :
  ∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 16 ∧ p.1 = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l327_32707


namespace NUMINAMATH_CALUDE_enterprise_tax_comparison_l327_32782

theorem enterprise_tax_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a < b) : 
  let x := (b - a) / 2
  let y := Real.sqrt (b / a) - 1
  b * (1 + y) > b + x := by sorry

end NUMINAMATH_CALUDE_enterprise_tax_comparison_l327_32782


namespace NUMINAMATH_CALUDE_john_beats_per_minute_l327_32777

/-- Calculates the number of beats per minute John can play given his playing schedule and total beats played. -/
def beats_per_minute (hours_per_day : ℕ) (days : ℕ) (total_beats : ℕ) : ℕ :=
  total_beats / (hours_per_day * days * 60)

/-- Theorem stating that John can play 200 beats per minute given the problem conditions. -/
theorem john_beats_per_minute :
  beats_per_minute 2 3 72000 = 200 := by
  sorry

end NUMINAMATH_CALUDE_john_beats_per_minute_l327_32777


namespace NUMINAMATH_CALUDE_nina_total_spent_l327_32726

/-- The total amount spent by Nina on toys, basketball cards, and shirts. -/
def total_spent (toy_quantity : ℕ) (toy_price : ℕ) (card_quantity : ℕ) (card_price : ℕ) (shirt_quantity : ℕ) (shirt_price : ℕ) : ℕ :=
  toy_quantity * toy_price + card_quantity * card_price + shirt_quantity * shirt_price

/-- Theorem stating that Nina's total spent is $70 -/
theorem nina_total_spent :
  total_spent 3 10 2 5 5 6 = 70 := by
  sorry

end NUMINAMATH_CALUDE_nina_total_spent_l327_32726


namespace NUMINAMATH_CALUDE_weight_equivalence_l327_32788

/-- The weight ratio between small and large circles -/
def weight_ratio : ℚ := 2 / 5

/-- The number of small circles -/
def num_small_circles : ℕ := 15

/-- Theorem stating the equivalence in weight between small and large circles -/
theorem weight_equivalence :
  (num_small_circles : ℚ) * weight_ratio = 6 := by sorry

end NUMINAMATH_CALUDE_weight_equivalence_l327_32788


namespace NUMINAMATH_CALUDE_prime_pair_divisibility_l327_32718

theorem prime_pair_divisibility (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q →
  (p^p + q^q + 1 ≡ 0 [MOD pq] ↔ (p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pair_divisibility_l327_32718


namespace NUMINAMATH_CALUDE_focus_coordinates_for_specific_ellipse_l327_32773

/-- Represents an ellipse with its center and axis endpoints -/
structure Ellipse where
  center : ℝ × ℝ
  major_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ)
  minor_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ)

/-- Calculates the coordinates of the focus with greater x-coordinate for a given ellipse -/
def focus_with_greater_x (e : Ellipse) : ℝ × ℝ :=
  sorry

/-- Theorem stating that for the given ellipse, the focus with greater x-coordinate
    has coordinates (3.5 + √6/2, 0) -/
theorem focus_coordinates_for_specific_ellipse :
  let e : Ellipse := {
    center := (3.5, 0),
    major_axis_endpoints := ((0, 0), (7, 0)),
    minor_axis_endpoints := ((3.5, 2.5), (3.5, -2.5))
  }
  focus_with_greater_x e = (3.5 + Real.sqrt 6 / 2, 0) := by
  sorry

end NUMINAMATH_CALUDE_focus_coordinates_for_specific_ellipse_l327_32773


namespace NUMINAMATH_CALUDE_trees_died_in_typhoon_l327_32741

theorem trees_died_in_typhoon (initial_trees left_trees : ℕ) : 
  initial_trees = 20 → left_trees = 4 → initial_trees - left_trees = 16 := by
  sorry

end NUMINAMATH_CALUDE_trees_died_in_typhoon_l327_32741


namespace NUMINAMATH_CALUDE_particle_probability_l327_32756

/-- The probability of a particle reaching point (2,3) after 5 moves -/
theorem particle_probability (n : ℕ) (k : ℕ) (p : ℝ) : 
  n = 5 → k = 2 → p = 1/2 → 
  Nat.choose n k * p^n = Nat.choose 5 2 * (1/2)^5 :=
by sorry

end NUMINAMATH_CALUDE_particle_probability_l327_32756


namespace NUMINAMATH_CALUDE_equation_solution_l327_32745

theorem equation_solution (x : ℝ) : 9 / (1 + 4 / x) = 1 → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l327_32745


namespace NUMINAMATH_CALUDE_perpendicular_lines_sum_l327_32720

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- A point (x, y) lies on a line ax + by + c = 0 -/
def lies_on (x y a b c : ℝ) : Prop := a * x + b * y + c = 0

theorem perpendicular_lines_sum (a b c : ℝ) :
  perpendicular (-a/4) (2/5) →
  lies_on 1 c a 4 (-2) →
  lies_on 1 c 2 (-5) b →
  a + b + c = -4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sum_l327_32720


namespace NUMINAMATH_CALUDE_parabola_line_intersection_sum_l327_32751

/-- Parabola P with equation y = x^2 + 4 -/
def P : ℝ → ℝ := λ x ↦ x^2 + 4

/-- Point Q -/
def Q : ℝ × ℝ := (10, 5)

/-- Line through Q with slope m -/
def line_through_Q (m : ℝ) : ℝ → ℝ := λ x ↦ m * (x - Q.1) + Q.2

/-- The line through Q with slope m does not intersect P -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line_through_Q m x

theorem parabola_line_intersection_sum (r s : ℝ) 
  (h : ∀ m, no_intersection m ↔ r < m ∧ m < s) : 
  r + s = 40 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_sum_l327_32751
