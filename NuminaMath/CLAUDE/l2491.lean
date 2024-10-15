import Mathlib

namespace NUMINAMATH_CALUDE_zoe_played_two_months_l2491_249124

/-- Calculates the number of months played given the initial cost, monthly cost, and total spent -/
def months_played (initial_cost monthly_cost total_spent : ℕ) : ℕ :=
  (total_spent - initial_cost) / monthly_cost

/-- Proves that Zoe played the game online for 2 months -/
theorem zoe_played_two_months (initial_cost monthly_cost total_spent : ℕ) 
  (h1 : initial_cost = 5)
  (h2 : monthly_cost = 8)
  (h3 : total_spent = 21) :
  months_played initial_cost monthly_cost total_spent = 2 := by
  sorry

#eval months_played 5 8 21

end NUMINAMATH_CALUDE_zoe_played_two_months_l2491_249124


namespace NUMINAMATH_CALUDE_complex_quadratic_modulus_l2491_249186

theorem complex_quadratic_modulus (z : ℂ) : z^2 - 8*z + 40 = 0 → Complex.abs z = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadratic_modulus_l2491_249186


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l2491_249132

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l2491_249132


namespace NUMINAMATH_CALUDE_biased_coin_probability_l2491_249174

theorem biased_coin_probability : ∀ h : ℝ,
  0 < h ∧ h < 1 →
  (Nat.choose 6 2 : ℝ) * h^2 * (1 - h)^4 = (Nat.choose 6 3 : ℝ) * h^3 * (1 - h)^3 →
  (Nat.choose 6 4 : ℝ) * h^4 * (1 - h)^2 = 19440 / 117649 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l2491_249174


namespace NUMINAMATH_CALUDE_wire_cutting_l2491_249179

theorem wire_cutting (total_length : ℝ) (difference : ℝ) (longer_part : ℝ) : 
  total_length = 180 ∧ difference = 32 → longer_part = 106 :=
by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l2491_249179


namespace NUMINAMATH_CALUDE_no_solution_exists_l2491_249183

theorem no_solution_exists : ¬∃ (a b c d : ℕ), 
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧ 
  71 * a + 72 * b + 73 * c + 74 * d = 2014 :=
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2491_249183


namespace NUMINAMATH_CALUDE_complex_power_approximation_l2491_249139

/-- The complex number (2 + i)/(2 - i) raised to the power of 600 is approximately equal to -0.982 - 0.189i -/
theorem complex_power_approximation :
  let z : ℂ := (2 + Complex.I) / (2 - Complex.I)
  ∃ (ε : ℝ) (hε : ε > 0), Complex.abs (z^600 - (-0.982 - 0.189 * Complex.I)) < ε :=
by sorry

end NUMINAMATH_CALUDE_complex_power_approximation_l2491_249139


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l2491_249194

/-- The inequality condition for a and b -/
def satisfies_inequality (a b : ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → a * x^3 + b * y^2 ≥ x * y - 1

/-- The main theorem statement -/
theorem minimum_value_theorem :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ satisfies_inequality a b ∧
    a^2 + b = 2 / (3 * Real.sqrt 3) ∧
    ∀ a' b' : ℝ, a' > 0 → b' > 0 → satisfies_inequality a' b' →
      a'^2 + b' ≥ 2 / (3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l2491_249194


namespace NUMINAMATH_CALUDE_joseph_investment_result_l2491_249149

/-- Calculates the final amount in an investment account after a given number of years,
    with an initial investment, yearly interest rate, and monthly additional deposits. -/
def investment_calculation (initial_investment : ℝ) (interest_rate : ℝ) 
                           (monthly_deposit : ℝ) (years : ℕ) : ℝ :=
  sorry

/-- Theorem stating that the investment calculation for Joseph's scenario
    results in $3982 after two years. -/
theorem joseph_investment_result :
  investment_calculation 1000 0.10 100 2 = 3982 := by
  sorry

end NUMINAMATH_CALUDE_joseph_investment_result_l2491_249149


namespace NUMINAMATH_CALUDE_spinner_probability_l2491_249175

-- Define the spinner regions
inductive Region
| A
| B1
| B2
| C

-- Define the probability function
def P : Region → ℚ
| Region.A  => 3/8
| Region.B1 => 1/8
| Region.B2 => 1/4
| Region.C  => 1/4  -- This is what we want to prove

-- State the theorem
theorem spinner_probability :
  P Region.C = 1/4 :=
by
  sorry

-- Additional lemmas to help with the proof
lemma total_probability :
  P Region.A + P Region.B1 + P Region.B2 + P Region.C = 1 :=
by
  sorry

lemma b_subregions :
  P Region.B1 + P Region.B2 = 3/8 :=
by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l2491_249175


namespace NUMINAMATH_CALUDE_subtraction_with_division_l2491_249192

theorem subtraction_with_division : 3034 - (1002 / 200.4) = 3029 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_with_division_l2491_249192


namespace NUMINAMATH_CALUDE_shrimp_price_theorem_l2491_249152

/-- The discounted price of a quarter-pound package of shrimp -/
def discounted_price : ℝ := 2.25

/-- The discount percentage as a decimal -/
def discount_rate : ℝ := 0.4

/-- The standard price per pound of shrimp -/
def standard_price : ℝ := 15

/-- Theorem stating that the standard price per pound of shrimp is $15 -/
theorem shrimp_price_theorem :
  standard_price = 15 ∧
  discounted_price = (1 - discount_rate) * (standard_price / 4) :=
by sorry

end NUMINAMATH_CALUDE_shrimp_price_theorem_l2491_249152


namespace NUMINAMATH_CALUDE_sum_of_angles_l2491_249173

-- Define the angles as real numbers
variable (A B C D F G EDC ECD : ℝ)

-- Define the conditions
variable (h1 : A + B + C + D = 360) -- ABCD is a quadrilateral
variable (h2 : G + F = EDC + ECD)   -- Given condition

-- Theorem statement
theorem sum_of_angles : A + B + C + D + F + G = 360 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_l2491_249173


namespace NUMINAMATH_CALUDE_parabola_latus_rectum_l2491_249136

/-- 
For a parabola with equation x^2 = ay and latus rectum y = 2, 
the value of a is -8.
-/
theorem parabola_latus_rectum (a : ℝ) : 
  (∀ x y : ℝ, x^2 = a*y) →  -- equation of parabola
  (∃ x : ℝ, x^2 = 2*a) →    -- latus rectum condition
  a = -8 := by
sorry

end NUMINAMATH_CALUDE_parabola_latus_rectum_l2491_249136


namespace NUMINAMATH_CALUDE_choose_four_from_fifteen_l2491_249165

theorem choose_four_from_fifteen (n : ℕ) (k : ℕ) : n = 15 ∧ k = 4 → Nat.choose n k = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_fifteen_l2491_249165


namespace NUMINAMATH_CALUDE_three_sqrt_two_bounds_l2491_249153

theorem three_sqrt_two_bounds : 4 < 3 * Real.sqrt 2 ∧ 3 * Real.sqrt 2 < 5 := by
  sorry

end NUMINAMATH_CALUDE_three_sqrt_two_bounds_l2491_249153


namespace NUMINAMATH_CALUDE_even_function_with_range_l2491_249100

/-- Given a function f(x) = (x + a)(bx - a) where a and b are real constants,
    if f is an even function and its range is [-4, +∞),
    then f(x) = x^2 - 4 -/
theorem even_function_with_range (a b : ℝ) :
  (∀ x, (x + a) * (b * x - a) = ((-(x : ℝ)) + a) * (b * (-x) - a)) →
  (∀ y ≥ -4, ∃ x, (x + a) * (b * x - a) = y) →
  (∀ x, (x + a) * (b * x - a) = x^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_even_function_with_range_l2491_249100


namespace NUMINAMATH_CALUDE_total_molecular_weight_l2491_249168

-- Define atomic weights
def atomic_weight_K : ℝ := 39.10
def atomic_weight_Cr : ℝ := 51.996
def atomic_weight_O : ℝ := 16.00
def atomic_weight_Fe : ℝ := 55.845
def atomic_weight_S : ℝ := 32.07
def atomic_weight_Mn : ℝ := 54.938

-- Define molecular weights
def molecular_weight_K2Cr2O7 : ℝ :=
  2 * atomic_weight_K + 2 * atomic_weight_Cr + 7 * atomic_weight_O

def molecular_weight_Fe2SO43 : ℝ :=
  2 * atomic_weight_Fe + 3 * (atomic_weight_S + 4 * atomic_weight_O)

def molecular_weight_KMnO4 : ℝ :=
  atomic_weight_K + atomic_weight_Mn + 4 * atomic_weight_O

-- Define the theorem
theorem total_molecular_weight :
  4 * molecular_weight_K2Cr2O7 +
  3 * molecular_weight_Fe2SO43 +
  5 * molecular_weight_KMnO4 = 3166.658 :=
by sorry

end NUMINAMATH_CALUDE_total_molecular_weight_l2491_249168


namespace NUMINAMATH_CALUDE_positive_operation_l2491_249199

theorem positive_operation : 
  ((-1 : ℝ)^2 > 0) ∧ 
  (-(|-2|) ≤ 0) ∧ 
  (0 * (-3) = 0) ∧ 
  (-(3^2) < 0) := by
sorry

end NUMINAMATH_CALUDE_positive_operation_l2491_249199


namespace NUMINAMATH_CALUDE_square_root_identity_polynomial_identity_square_root_polynomial_l2491_249117

theorem square_root_identity (n : ℕ) : 
  Real.sqrt ((n - 1) * (n + 1) + 1) = n :=
sorry

theorem polynomial_identity (n : ℕ) : 
  (n * (n + 3) + 1)^2 = n * (n + 1) * (n + 2) * (n + 3) + 1 :=
sorry

theorem square_root_polynomial (n : ℕ) : 
  Real.sqrt (n * (n + 1) * (n + 2) * (n + 3) + 1) = n * (n + 3) :=
sorry

end NUMINAMATH_CALUDE_square_root_identity_polynomial_identity_square_root_polynomial_l2491_249117


namespace NUMINAMATH_CALUDE_jesse_carpet_need_l2491_249159

/-- The additional carpet needed for Jesse's room -/
def additional_carpet_needed (room_length : ℝ) (room_width : ℝ) (existing_carpet : ℝ) : ℝ :=
  room_length * room_width - existing_carpet

/-- Theorem stating the additional carpet needed for Jesse's room -/
theorem jesse_carpet_need : 
  additional_carpet_needed 11 15 16 = 149 := by
  sorry

end NUMINAMATH_CALUDE_jesse_carpet_need_l2491_249159


namespace NUMINAMATH_CALUDE_inequality_proof_l2491_249158

theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  3 * a^3 + 2 * b^3 ≥ 3 * a^2 * b + 2 * a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2491_249158


namespace NUMINAMATH_CALUDE_remainder_sum_l2491_249177

theorem remainder_sum (a b c : ℤ) 
  (ha : a % 90 = 84) 
  (hb : b % 120 = 114) 
  (hc : c % 150 = 144) : 
  (a + b + c) % 30 = 12 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l2491_249177


namespace NUMINAMATH_CALUDE_expression_evaluation_l2491_249180

theorem expression_evaluation :
  let a : ℚ := -1
  let b : ℚ := 1/7
  (3*a^3 - 2*a*b + b^2) - 2*(-a^3 - a*b + 4*b^2) = -5 - 1/7 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2491_249180


namespace NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l2491_249138

theorem order_of_logarithmic_fractions :
  let a : ℝ := (Real.log 2) / 2
  let b : ℝ := (Real.log 3) / 3
  let c : ℝ := (Real.log 5) / 5
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l2491_249138


namespace NUMINAMATH_CALUDE_trig_expression_value_cos_2α_minus_π_4_l2491_249187

/- For the first problem -/
theorem trig_expression_value (α : Real) (h : Real.tan α = -3/4) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = -3/4 := by
  sorry

/- For the second problem -/
theorem cos_2α_minus_π_4 (α : Real) (h1 : Real.sin α + Real.cos α = 1/5) (h2 : 0 ≤ α ∧ α ≤ π) :
  Real.cos (2*α - π/4) = -31*Real.sqrt 2/50 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_cos_2α_minus_π_4_l2491_249187


namespace NUMINAMATH_CALUDE_square_root_range_l2491_249169

theorem square_root_range (x : ℝ) : x - 2 ≥ 0 ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_range_l2491_249169


namespace NUMINAMATH_CALUDE_pie_eating_problem_l2491_249154

theorem pie_eating_problem (initial_stock : ℕ) (daily_portion : ℕ) (day : ℕ) :
  initial_stock = 340 →
  daily_portion > 0 →
  day > 0 →
  initial_stock = day * daily_portion + daily_portion / 4 →
  (day = 5 ∨ day = 21) :=
sorry

end NUMINAMATH_CALUDE_pie_eating_problem_l2491_249154


namespace NUMINAMATH_CALUDE_largest_810_triple_l2491_249157

/-- Converts a base-10 number to its base-8 representation as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 8) :: aux (m / 8)
  aux n |>.reverse

/-- Converts a list of digits to its base-10 representation -/
def fromDigits (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/-- Checks if a number is an 8-10 triple -/
def is810Triple (n : ℕ) : Prop :=
  fromDigits (toBase8 n) = 3 * n

/-- Statement: 273 is the largest 8-10 triple -/
theorem largest_810_triple : 
  (∀ m : ℕ, m > 273 → ¬ is810Triple m) ∧ is810Triple 273 :=
sorry

end NUMINAMATH_CALUDE_largest_810_triple_l2491_249157


namespace NUMINAMATH_CALUDE_equation_equivalence_l2491_249126

theorem equation_equivalence :
  ∃ (m n p : ℤ), ∀ (a b x y : ℝ),
    (a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1)) ↔ 
    ((a^m*x - a^n) * (a^p*y - a^3) = a^5*b^5) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2491_249126


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2491_249125

theorem sphere_volume_equals_surface_area (r : ℝ) : 
  (4 / 3 * Real.pi * r^3 = 4 * Real.pi * r^2) → r = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2491_249125


namespace NUMINAMATH_CALUDE_motorbike_distance_theorem_l2491_249134

/-- Given two motorbikes traveling the same distance, with speeds of 60 km/h and 64 km/h
    respectively, and the slower bike taking 1 hour more than the faster bike,
    prove that the distance traveled is 960 kilometers. -/
theorem motorbike_distance_theorem (distance : ℝ) (time_slower : ℝ) (time_faster : ℝ) :
  (60 * time_slower = distance) →
  (64 * time_faster = distance) →
  (time_slower = time_faster + 1) →
  distance = 960 := by
  sorry

end NUMINAMATH_CALUDE_motorbike_distance_theorem_l2491_249134


namespace NUMINAMATH_CALUDE_existence_of_sequences_l2491_249151

theorem existence_of_sequences : ∃ (u v : ℕ → ℕ) (k : ℕ+),
  (∀ n m : ℕ, n < m → u n < u m) ∧
  (∀ n m : ℕ, n < m → v n < v m) ∧
  (∀ n : ℕ, k * (u n * (u n + 1)) = v n ^ 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_sequences_l2491_249151


namespace NUMINAMATH_CALUDE_blue_balls_count_l2491_249143

theorem blue_balls_count (black_balls : ℕ) (blue_balls : ℕ) : 
  (black_balls : ℚ) / blue_balls = 5 / 3 → 
  black_balls = 15 → 
  blue_balls = 9 := by
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l2491_249143


namespace NUMINAMATH_CALUDE_pizza_combinations_l2491_249145

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  1 + n + n.choose 2 = 37 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l2491_249145


namespace NUMINAMATH_CALUDE_tangent_line_right_triangle_l2491_249118

/-- Given a line ax + by + c = 0 (a, b, c ≠ 0) tangent to the circle x² + y² = 1,
    the triangle with side lengths |a|, |b|, and |c| is a right triangle. -/
theorem tangent_line_right_triangle (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
    (h_tangent : ∀ x y : ℝ, a * x + b * y + c = 0 → x^2 + y^2 = 1) :
  a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_right_triangle_l2491_249118


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2491_249137

/-- 
Given a selling price and a profit percentage, calculate the cost price.
-/
theorem cost_price_calculation 
  (selling_price : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : selling_price = 1800) 
  (h2 : profit_percentage = 20) :
  selling_price / (1 + profit_percentage / 100) = 1500 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2491_249137


namespace NUMINAMATH_CALUDE_part1_part2_part3_l2491_249109

-- Define the system of equations
def system (x y m : ℝ) : Prop :=
  2 * x + y = 4 * m ∧ x + 2 * y = 2 * m + 1

-- Part 1
theorem part1 (x y m : ℝ) :
  system x y m → x + y = 1 → m = 1/3 := by sorry

-- Part 2
theorem part2 (x y m : ℝ) :
  system x y m → -1 ≤ x - y ∧ x - y ≤ 5 → 0 ≤ m ∧ m ≤ 3 := by sorry

-- Part 3
theorem part3 (m : ℝ) :
  0 ≤ m ∧ m ≤ 3 →
  (0 ≤ m ∧ m ≤ 3/2 → |m+2| + |2*m-3| = 5 - m) ∧
  (3/2 < m ∧ m ≤ 3 → |m+2| + |2*m-3| = 3*m - 1) := by sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l2491_249109


namespace NUMINAMATH_CALUDE_implication_disjunction_equivalence_l2491_249127

theorem implication_disjunction_equivalence (A B : Prop) : (A → B) ↔ (¬A ∨ B) := by
  sorry

end NUMINAMATH_CALUDE_implication_disjunction_equivalence_l2491_249127


namespace NUMINAMATH_CALUDE_sin_cos_shift_l2491_249176

/-- Given two functions f and g defined on real numbers,
    prove that they are equivalent up to a horizontal shift. -/
theorem sin_cos_shift (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x + π / 3)
  let g : ℝ → ℝ := λ x ↦ Real.cos (2 * x)
  f x = g (x - π / 12) := by
  sorry


end NUMINAMATH_CALUDE_sin_cos_shift_l2491_249176


namespace NUMINAMATH_CALUDE_base8_4532_equals_2394_l2491_249108

-- Define a function to convert a base 8 number to base 10
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Define the base 8 number 4532
def base8Number : List Nat := [2, 3, 5, 4]

-- Theorem statement
theorem base8_4532_equals_2394 :
  base8ToBase10 base8Number = 2394 := by
  sorry

end NUMINAMATH_CALUDE_base8_4532_equals_2394_l2491_249108


namespace NUMINAMATH_CALUDE_certain_number_divisor_of_factorial_l2491_249119

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem certain_number_divisor_of_factorial :
  ∃! (n : ℕ), n > 0 ∧ (factorial 15) % (n^6) = 0 ∧ (factorial 15) % (n^7) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_divisor_of_factorial_l2491_249119


namespace NUMINAMATH_CALUDE_units_digit_of_8129_power_1351_l2491_249111

theorem units_digit_of_8129_power_1351 : 8129^1351 % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_8129_power_1351_l2491_249111


namespace NUMINAMATH_CALUDE_seven_correct_guesses_l2491_249101

/-- A guess is either a lower bound (not less than) or an upper bound (not more than) -/
inductive Guess
  | LowerBound (n : Nat)
  | UpperBound (n : Nat)

/-- The set of guesses made by the teachers -/
def teacherGuesses : List Guess := [
  Guess.LowerBound 1, Guess.UpperBound 2,
  Guess.LowerBound 3, Guess.UpperBound 4,
  Guess.LowerBound 5, Guess.UpperBound 6,
  Guess.LowerBound 7, Guess.UpperBound 8,
  Guess.LowerBound 9, Guess.UpperBound 10,
  Guess.LowerBound 11, Guess.UpperBound 12
]

/-- A guess is correct if it's satisfied by the given number -/
def isCorrectGuess (x : Nat) (g : Guess) : Bool :=
  match g with
  | Guess.LowerBound n => x ≥ n
  | Guess.UpperBound n => x ≤ n

/-- The number of correct guesses for a given number -/
def correctGuessCount (x : Nat) : Nat :=
  (teacherGuesses.filter (isCorrectGuess x)).length

/-- There exists a number for which exactly 7 guesses are correct -/
theorem seven_correct_guesses : ∃ x, correctGuessCount x = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_correct_guesses_l2491_249101


namespace NUMINAMATH_CALUDE_range_of_f_l2491_249193

def f (x : ℝ) : ℝ := x^2 + 1

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2491_249193


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2491_249161

/-- Given a geometric sequence with first term 5 and second term 1/5, 
    the seventh term of the sequence is 1/48828125. -/
theorem seventh_term_of_geometric_sequence :
  let a₁ : ℚ := 5
  let a₂ : ℚ := 1/5
  let r : ℚ := a₂ / a₁
  let n : ℕ := 7
  let a_n : ℚ := a₁ * r^(n-1)
  a_n = 1/48828125 := by sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2491_249161


namespace NUMINAMATH_CALUDE_remainder_problem_l2491_249162

theorem remainder_problem (d : ℤ) (r : ℤ) 
  (h1 : d > 1)
  (h2 : 1250 % d = r)
  (h3 : 1890 % d = r)
  (h4 : 2500 % d = r) :
  d - r = 10 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2491_249162


namespace NUMINAMATH_CALUDE_chess_draw_probability_l2491_249133

theorem chess_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.4)
  (h_not_lose : p_not_lose = 0.9) :
  p_not_lose - p_win = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chess_draw_probability_l2491_249133


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2491_249164

theorem least_addition_for_divisibility : 
  ∃ (n : ℕ), n = 15 ∧ 
  (∀ (m : ℕ), m < n → ¬(23 ∣ (4499 * 17 + m))) ∧
  (23 ∣ (4499 * 17 + n)) := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2491_249164


namespace NUMINAMATH_CALUDE_emma_running_time_l2491_249172

theorem emma_running_time (emma_time : ℝ) (fernando_time : ℝ) : 
  fernando_time = 2 * emma_time →
  emma_time + fernando_time = 60 →
  emma_time = 20 := by
sorry

end NUMINAMATH_CALUDE_emma_running_time_l2491_249172


namespace NUMINAMATH_CALUDE_mineral_worth_l2491_249188

/-- The worth of a mineral given its price per gram and weight -/
theorem mineral_worth (price_per_gram : ℝ) (weight_1 weight_2 : ℝ) :
  price_per_gram = 17.25 →
  weight_1 = 1000 →
  weight_2 = 10 →
  price_per_gram * (weight_1 + weight_2) = 17422.5 := by
  sorry

end NUMINAMATH_CALUDE_mineral_worth_l2491_249188


namespace NUMINAMATH_CALUDE_no_mn_divisibility_l2491_249102

theorem no_mn_divisibility : ¬∃ (m n : ℕ+), 
  (m.val * n.val ∣ 3^m.val + 1) ∧ (m.val * n.val ∣ 3^n.val + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_mn_divisibility_l2491_249102


namespace NUMINAMATH_CALUDE_polar_bear_trout_consumption_l2491_249122

/-- The daily fish consumption of the polar bear in buckets -/
def total_fish : ℝ := 0.6

/-- The daily salmon consumption of the polar bear in buckets -/
def salmon : ℝ := 0.4

/-- The daily trout consumption of the polar bear in buckets -/
def trout : ℝ := total_fish - salmon

theorem polar_bear_trout_consumption :
  trout = 0.2 := by sorry

end NUMINAMATH_CALUDE_polar_bear_trout_consumption_l2491_249122


namespace NUMINAMATH_CALUDE_algebraic_simplification_l2491_249130

/-- Proves the simplification of two algebraic expressions -/
theorem algebraic_simplification 
  (a b m n : ℝ) : 
  ((a - 2*b) - (2*b - 5*a) = 6*a - 4*b) ∧ 
  (-m^2*n + (4*m*n^2 - 3*m*n) - 2*(m*n^2 - 3*m^2*n) = 5*m^2*n + 2*m*n^2 - 3*m*n) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l2491_249130


namespace NUMINAMATH_CALUDE_solve_travel_problem_l2491_249121

def travel_problem (train_distance : ℝ) : Prop :=
  let bus_distance := train_distance / 2
  let cab_distance := bus_distance / 3
  let total_distance := train_distance + bus_distance + cab_distance
  (train_distance = 300) → (total_distance = 500)

theorem solve_travel_problem : travel_problem 300 := by
  sorry

end NUMINAMATH_CALUDE_solve_travel_problem_l2491_249121


namespace NUMINAMATH_CALUDE_question_one_l2491_249112

theorem question_one (a : ℝ) (h : a^2 + a = 3) : 2*a^2 + 2*a + 2023 = 2029 := by
  sorry


end NUMINAMATH_CALUDE_question_one_l2491_249112


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2491_249115

theorem complex_number_in_first_quadrant :
  let z : ℂ := (2 + I) / (1 - I)
  (z.re > 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2491_249115


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l2491_249110

noncomputable def f (x : ℝ) : ℝ := -(Real.sqrt 3 / 3) * x^3 + 2

theorem tangent_slope_angle (x : ℝ) : 
  let f' := deriv f
  let slope := f' 1
  let angle := Real.arctan (-slope)
  x = 1 → angle = 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l2491_249110


namespace NUMINAMATH_CALUDE_negative_double_negation_l2491_249184

theorem negative_double_negation (x : ℝ) (h : -x = 2) : -(-(-x)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_double_negation_l2491_249184


namespace NUMINAMATH_CALUDE_ball_max_height_l2491_249103

-- Define the height function
def h (t : ℝ) : ℝ := -20 * t^2 - 40 * t + 50

-- State the theorem
theorem ball_max_height :
  ∃ (max : ℝ), max = 70 ∧ ∀ (t : ℝ), h t ≤ max :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l2491_249103


namespace NUMINAMATH_CALUDE_sum_base7_and_base13_equals_1109_l2491_249189

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 13 to base 10, where 'C' represents 12 -/
def base13ToBase10 (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of 536 (base 7) and 4C5 (base 13) is 1109 in base 10 -/
theorem sum_base7_and_base13_equals_1109 : 
  base7ToBase10 536 + base13ToBase10 4125 = 1109 := by sorry

end NUMINAMATH_CALUDE_sum_base7_and_base13_equals_1109_l2491_249189


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2491_249191

-- Define the set containing a
def S (a : ℝ) : Set ℝ := {a^2 - 2*a + 2, a - 1, 0}

-- Theorem statement
theorem solution_set_inequality (a : ℝ) 
  (h : {1, a} ⊆ S a) : 
  {x : ℝ | a*x^2 - 5*x + a > 0} = 
  {x : ℝ | x < 1/2 ∨ x > 2} := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2491_249191


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2491_249104

theorem geometric_sequence_common_ratio 
  (a : ℝ) (term2 term3 term4 : ℝ) :
  a = 12 ∧ 
  term2 = -18 ∧ 
  term3 = 27 ∧ 
  term4 = -40.5 ∧ 
  term2 = a * r ∧ 
  term3 = a * r^2 ∧ 
  term4 = a * r^3 →
  r = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2491_249104


namespace NUMINAMATH_CALUDE_simplified_fourth_root_sum_l2491_249148

theorem simplified_fourth_root_sum (a b : ℕ+) :
  (2^6 * 5^2 : ℝ)^(1/4) = a * b^(1/4) → a + b = 102 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fourth_root_sum_l2491_249148


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_below_neg_85_l2491_249166

theorem largest_multiple_of_seven_below_neg_85 :
  ∀ n : ℤ, 7 ∣ n ∧ n < -85 → n ≤ -91 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_below_neg_85_l2491_249166


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2491_249105

theorem sum_of_fractions : (1 : ℚ) / 12 + (1 : ℚ) / 15 = (3 : ℚ) / 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2491_249105


namespace NUMINAMATH_CALUDE_teammates_average_points_l2491_249196

/-- Proves that the teammates' average points per game is 40, given Wade's average and team total -/
theorem teammates_average_points (wade_avg : ℝ) (team_total : ℝ) (num_games : ℕ) : 
  wade_avg = 20 →
  team_total = 300 →
  num_games = 5 →
  (team_total - wade_avg * num_games) / num_games = 40 := by
  sorry

end NUMINAMATH_CALUDE_teammates_average_points_l2491_249196


namespace NUMINAMATH_CALUDE_min_value_part1_min_value_part2_l2491_249140

-- Part 1
theorem min_value_part1 (x : ℝ) (h : x > -1) :
  x + 4 / (x + 1) + 6 ≥ 9 ∧
  (x + 4 / (x + 1) + 6 = 9 ↔ x = 1) :=
sorry

-- Part 2
theorem min_value_part2 (x : ℝ) (h : x > 1) :
  (x^2 + 8) / (x - 1) ≥ 8 ∧
  ((x^2 + 8) / (x - 1) = 8 ↔ x = 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_part1_min_value_part2_l2491_249140


namespace NUMINAMATH_CALUDE_point_on_number_line_l2491_249144

theorem point_on_number_line (B C : ℝ) : 
  B = 3 → abs (C - B) = 2 → (C = 1 ∨ C = 5) :=
by sorry

end NUMINAMATH_CALUDE_point_on_number_line_l2491_249144


namespace NUMINAMATH_CALUDE_woodburning_price_l2491_249128

/-- Represents the selling price of a woodburning -/
def selling_price : ℝ := 15

/-- Represents the number of woodburnings sold -/
def num_woodburnings : ℕ := 20

/-- Represents the cost of wood -/
def wood_cost : ℝ := 100

/-- Represents the total profit -/
def total_profit : ℝ := 200

/-- Theorem stating that the selling price of each woodburning is $15 -/
theorem woodburning_price : 
  selling_price * num_woodburnings - wood_cost = total_profit :=
by sorry

end NUMINAMATH_CALUDE_woodburning_price_l2491_249128


namespace NUMINAMATH_CALUDE_vector_subtraction_l2491_249142

/-- Given two planar vectors a and b, prove that a - 2b equals the expected result. -/
theorem vector_subtraction (a b : ℝ × ℝ) (h1 : a = (5, 3)) (h2 : b = (1, -2)) :
  a - 2 • b = (3, 7) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2491_249142


namespace NUMINAMATH_CALUDE_total_erasers_l2491_249160

theorem total_erasers (celine gabriel julian : ℕ) : 
  celine = 2 * gabriel → 
  julian = 2 * celine → 
  celine = 10 → 
  celine + gabriel + julian = 35 := by
sorry

end NUMINAMATH_CALUDE_total_erasers_l2491_249160


namespace NUMINAMATH_CALUDE_cos_squared_165_minus_sin_squared_15_l2491_249141

theorem cos_squared_165_minus_sin_squared_15 :
  Real.cos (165 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_165_minus_sin_squared_15_l2491_249141


namespace NUMINAMATH_CALUDE_parallel_line_length_l2491_249190

/-- A triangle with a parallel line dividing it into equal areas -/
structure DividedTriangle where
  base : ℝ
  height : ℝ
  parallel_line : ℝ
  h_base_positive : 0 < base
  h_height_positive : 0 < height
  h_parallel_positive : 0 < parallel_line
  h_parallel_less_than_base : parallel_line < base
  h_equal_areas : parallel_line^2 / base^2 = 1/4

/-- The theorem stating that for a triangle with base 20 and height 24,
    the parallel line dividing it into four equal areas has length 10 -/
theorem parallel_line_length (t : DividedTriangle)
    (h_base : t.base = 20)
    (h_height : t.height = 24) :
    t.parallel_line = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_length_l2491_249190


namespace NUMINAMATH_CALUDE_inequality_solutions_l2491_249181

theorem inequality_solutions :
  (∀ x : ℝ, (|x + 1| / |x + 2| ≥ 1) ↔ (x ≤ -3/2 ∧ x ≠ -2)) ∧
  (∀ a x : ℝ,
    (a * (x - 1) / (x - 2) > 1) ↔
    ((a > 1 ∧ (x > 2 ∨ x < (a - 2) / (a - 1))) ∨
     (a = 1 ∧ x > 2) ∨
     (0 < a ∧ a < 1 ∧ 2 < x ∧ x < (a - 2) / (a - 1)) ∨
     (a < 0 ∧ (a - 2) / (a - 1) < x ∧ x < 2))) ∧
  (∀ x : ℝ, ¬(0 * (x - 1) / (x - 2) > 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l2491_249181


namespace NUMINAMATH_CALUDE_train_cars_count_l2491_249182

/-- Calculates the number of cars in a train based on observed data -/
def train_cars (cars_observed : ℕ) (observation_time : ℕ) (total_time : ℕ) : ℕ :=
  (cars_observed * total_time) / observation_time

/-- Proves that the number of cars in the train is 112 given the observed data -/
theorem train_cars_count : train_cars 8 15 210 = 112 := by
  sorry

end NUMINAMATH_CALUDE_train_cars_count_l2491_249182


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l2491_249116

theorem ellipse_hyperbola_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e_ellipse := Real.sqrt 3 / 2
  let c := e_ellipse * a
  let e_hyperbola := Real.sqrt ((a^2 + b^2) / a^2)
  (a^2 = b^2 + c^2) → e_hyperbola = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l2491_249116


namespace NUMINAMATH_CALUDE_calculate_fraction_product_l2491_249170

theorem calculate_fraction_product : 
  let mixed_number : ℚ := 3 + 3/4
  let decimal_one : ℚ := 0.2
  let whole_number : ℕ := 135
  let decimal_two : ℚ := 5.4
  ((mixed_number * decimal_one) / whole_number) * decimal_two = 0.03 := by
sorry

end NUMINAMATH_CALUDE_calculate_fraction_product_l2491_249170


namespace NUMINAMATH_CALUDE_minimum_students_with_both_devices_l2491_249167

theorem minimum_students_with_both_devices (n : ℕ) (h1 : n % 7 = 0) (h2 : n % 6 = 0) : ∃ x : ℕ,
  x = n * 3 / 7 + n * 5 / 6 - n ∧
  x ≥ 11 ∧
  (∀ y : ℕ, y < x → ∃ m : ℕ, m > n ∧ m % 7 = 0 ∧ m % 6 = 0 ∧ y = m * 3 / 7 + m * 5 / 6 - m) :=
by sorry

#check minimum_students_with_both_devices

end NUMINAMATH_CALUDE_minimum_students_with_both_devices_l2491_249167


namespace NUMINAMATH_CALUDE_frog_jump_distance_l2491_249129

/-- The jumping contest problem -/
theorem frog_jump_distance 
  (grasshopper_jump : ℕ) 
  (frog_extra_jump : ℕ) 
  (mouse_less_jump : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_extra_jump = 39)
  (h3 : mouse_less_jump = 94) :
  grasshopper_jump + frog_extra_jump = 58 :=
by sorry

end NUMINAMATH_CALUDE_frog_jump_distance_l2491_249129


namespace NUMINAMATH_CALUDE_bottle_ratio_is_half_l2491_249106

/-- Represents the distribution of bottles in a delivery van -/
structure BottleDistribution where
  total : ℕ
  cider : ℕ
  beer : ℕ
  mixed : ℕ
  first_house : ℕ

/-- The ratio of bottles given to the first house to the total number of bottles -/
def bottle_ratio (d : BottleDistribution) : ℚ :=
  d.first_house / d.total

/-- Theorem stating the ratio of bottles given to the first house to the total number of bottles -/
theorem bottle_ratio_is_half (d : BottleDistribution) 
    (h1 : d.total = 180)
    (h2 : d.cider = 40)
    (h3 : d.beer = 80)
    (h4 : d.mixed = d.total - d.cider - d.beer)
    (h5 : d.first_house = 90) : 
  bottle_ratio d = 1/2 := by
  sorry

#eval bottle_ratio { total := 180, cider := 40, beer := 80, mixed := 60, first_house := 90 }

end NUMINAMATH_CALUDE_bottle_ratio_is_half_l2491_249106


namespace NUMINAMATH_CALUDE_difference_of_products_l2491_249178

theorem difference_of_products : 20132014 * 20142013 - 20132013 * 20142014 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_products_l2491_249178


namespace NUMINAMATH_CALUDE_best_method_for_pedestrian_phone_use_data_l2491_249163

/-- Represents a data collection method -/
structure DataCollectionMethod where
  name : String
  target_group : String
  is_random : Bool

/-- Represents the characteristics of a good data collection method -/
structure MethodCharacteristics where
  is_representative : Bool
  is_extensive : Bool

/-- Defines the criteria for evaluating a data collection method -/
def evaluate_method (method : DataCollectionMethod) : MethodCharacteristics :=
  { is_representative := method.is_random && method.target_group = "pedestrians on roadside",
    is_extensive := method.is_random && method.target_group = "pedestrians on roadside" }

/-- The theorem stating that randomly distributing questionnaires to pedestrians on the roadside
    is the most representative and extensive method for collecting data on pedestrians
    walking on the roadside while looking down at their phones -/
theorem best_method_for_pedestrian_phone_use_data :
  let method := { name := "Random questionnaires to roadside pedestrians",
                  target_group := "pedestrians on roadside",
                  is_random := true : DataCollectionMethod }
  let evaluation := evaluate_method method
  evaluation.is_representative ∧ evaluation.is_extensive :=
by
  sorry


end NUMINAMATH_CALUDE_best_method_for_pedestrian_phone_use_data_l2491_249163


namespace NUMINAMATH_CALUDE_volleyball_tournament_l2491_249171

theorem volleyball_tournament (n : ℕ) : n > 0 → 2 * (n.choose 2) = 56 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_tournament_l2491_249171


namespace NUMINAMATH_CALUDE_cube_roots_of_unity_sum_l2491_249107

theorem cube_roots_of_unity_sum (i : ℂ) :
  i^2 = -1 →
  let x : ℂ := (-1 + i * Real.sqrt 3) / 2
  let y : ℂ := (-1 - i * Real.sqrt 3) / 2
  x^6 + y^6 = 2 := by sorry

end NUMINAMATH_CALUDE_cube_roots_of_unity_sum_l2491_249107


namespace NUMINAMATH_CALUDE_sum_divided_non_negative_l2491_249120

theorem sum_divided_non_negative (x : ℝ) :
  ((x + 6) / 2 ≥ 0) ↔ (∃ y ≥ 0, y = (x + 6) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_divided_non_negative_l2491_249120


namespace NUMINAMATH_CALUDE_symmetry_of_point_l2491_249198

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Origin point (0,0) -/
def origin : Point := ⟨0, 0⟩

/-- Symmetry of a point about the origin -/
def symmetric_about_origin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem symmetry_of_point :
  let A : Point := ⟨2, -1⟩
  let B : Point := symmetric_about_origin A
  B = ⟨-2, 1⟩ := by sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l2491_249198


namespace NUMINAMATH_CALUDE_westward_notation_l2491_249150

/-- Represents the direction on the runway -/
inductive Direction
  | East
  | West

/-- Represents a distance walked on the runway -/
structure Walk where
  distance : ℝ
  direction : Direction

/-- Converts a walk to its signed representation -/
def Walk.toSigned (w : Walk) : ℝ :=
  match w.direction with
  | Direction.East => w.distance
  | Direction.West => -w.distance

theorem westward_notation (d : ℝ) (h : d > 0) :
  let eastward := Walk.toSigned { distance := 8, direction := Direction.East }
  let westward := Walk.toSigned { distance := d, direction := Direction.West }
  eastward = 8 → westward = -d :=
by sorry

end NUMINAMATH_CALUDE_westward_notation_l2491_249150


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l2491_249147

theorem complex_power_magnitude : Complex.abs ((2 + 2 * Complex.I * Real.sqrt 2) ^ 6) = 1728 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l2491_249147


namespace NUMINAMATH_CALUDE_triangle_area_is_eight_l2491_249123

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A triangle defined by three lines -/
structure Triangle where
  line1 : Line
  line2 : Line
  line3 : Line

/-- Calculate the area of a triangle given its three bounding lines -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- The specific triangle in the problem -/
def problemTriangle : Triangle :=
  { line1 := { slope := 2, intercept := 0 }
  , line2 := { slope := -2, intercept := 0 }
  , line3 := { slope := 0, intercept := 4 }
  }

theorem triangle_area_is_eight :
  triangleArea problemTriangle = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_eight_l2491_249123


namespace NUMINAMATH_CALUDE_square_root_problem_l2491_249114

theorem square_root_problem (a x : ℝ) 
  (h1 : Real.sqrt a = x + 3) 
  (h2 : Real.sqrt a = 3 * x - 11) : 
  2 * a - 1 = 199 := by
sorry

end NUMINAMATH_CALUDE_square_root_problem_l2491_249114


namespace NUMINAMATH_CALUDE_total_buses_is_816_l2491_249156

/-- Represents the bus schedule for different types of days -/
structure BusSchedule where
  weekday : Nat
  saturday : Nat
  sunday_holiday : Nat

/-- Calculates the total number of buses in a month -/
def total_buses_in_month (schedule : BusSchedule) (public_holidays : Nat) : Nat :=
  let weekdays := 20 - public_holidays
  let saturdays := 4
  let sundays_holidays := 4 + public_holidays
  weekdays * schedule.weekday + saturdays * schedule.saturday + sundays_holidays * schedule.sunday_holiday

/-- The bus schedule for the given problem -/
def problem_schedule : BusSchedule :=
  { weekday := 36
  , saturday := 24
  , sunday_holiday := 12 }

/-- Theorem stating that the total number of buses in the month is 816 -/
theorem total_buses_is_816 :
  total_buses_in_month problem_schedule 2 = 816 := by
  sorry

end NUMINAMATH_CALUDE_total_buses_is_816_l2491_249156


namespace NUMINAMATH_CALUDE_fourth_quadrant_trig_simplification_l2491_249135

/-- For an angle α in the fourth quadrant, 
    cos α √((1 - sin α) / (1 + sin α)) + sin α √((1 - cos α) / (1 + cos α)) = cos α - sin α -/
theorem fourth_quadrant_trig_simplification (α : Real) 
  (h_fourth_quadrant : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi) :
  Real.cos α * Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) + 
  Real.sin α * Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = 
  Real.cos α - Real.sin α := by
sorry

end NUMINAMATH_CALUDE_fourth_quadrant_trig_simplification_l2491_249135


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2491_249146

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i + 2) / i = 1 - 2 * i := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2491_249146


namespace NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_3_l2491_249195

theorem simplify_fraction_with_sqrt_3 :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_3_l2491_249195


namespace NUMINAMATH_CALUDE_sum_of_powers_mod_17_l2491_249185

theorem sum_of_powers_mod_17 :
  (∃ x : ℤ, x * 3 ≡ 1 [ZMOD 17]) →
  (∃ y : ℤ, y * 3^2 ≡ 1 [ZMOD 17]) →
  (∃ z : ℤ, z * 3^3 ≡ 1 [ZMOD 17]) →
  (∃ w : ℤ, w * 3^4 ≡ 1 [ZMOD 17]) →
  (∃ v : ℤ, v * 3^5 ≡ 1 [ZMOD 17]) →
  (∃ u : ℤ, u * 3^6 ≡ 1 [ZMOD 17]) →
  x + y + z + w + v + u ≡ 5 [ZMOD 17] :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_mod_17_l2491_249185


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l2491_249197

theorem min_value_squared_sum (a b t p : ℝ) (h1 : a + b = t) (h2 : a * b = p) :
  a^2 + a*b + b^2 ≥ (3/4) * t^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l2491_249197


namespace NUMINAMATH_CALUDE_systematic_sampling_methods_l2491_249155

/-- Represents a sampling method -/
inductive SamplingMethod
  | BallSelection
  | ProductInspection
  | MarketSurvey
  | CinemaAudienceSurvey

/-- Defines the characteristics of systematic sampling -/
def is_systematic_sampling (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.BallSelection => true
  | SamplingMethod.ProductInspection => true
  | SamplingMethod.MarketSurvey => false
  | SamplingMethod.CinemaAudienceSurvey => true

/-- Theorem stating which sampling methods are systematic -/
theorem systematic_sampling_methods :
  (is_systematic_sampling SamplingMethod.BallSelection) ∧
  (is_systematic_sampling SamplingMethod.ProductInspection) ∧
  (¬is_systematic_sampling SamplingMethod.MarketSurvey) ∧
  (is_systematic_sampling SamplingMethod.CinemaAudienceSurvey) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_methods_l2491_249155


namespace NUMINAMATH_CALUDE_compute_a_l2491_249113

-- Define the polynomial
def f (a b : ℚ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 48

-- State the theorem
theorem compute_a : 
  ∃ (a b : ℚ), f a b (-1 - 5 * Real.sqrt 3) = 0 ∧ a = 50/37 := by
  sorry

end NUMINAMATH_CALUDE_compute_a_l2491_249113


namespace NUMINAMATH_CALUDE_cos_four_theta_value_l2491_249131

theorem cos_four_theta_value (θ : ℝ) 
  (h : ∑' n, (Real.cos θ)^(2*n) = 9/2) : 
  Real.cos (4*θ) = -31/81 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_theta_value_l2491_249131
