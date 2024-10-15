import Mathlib

namespace NUMINAMATH_CALUDE_max_table_height_value_l623_62390

/-- Triangle DEF with sides a, b, c -/
structure Triangle (a b c : ℝ) : Type :=
  (side_a : a > 0)
  (side_b : b > 0)
  (side_c : c > 0)

/-- The maximum table height function -/
def maxTableHeight (t : Triangle 25 29 32) : ℝ := sorry

/-- Theorem stating the maximum table height -/
theorem max_table_height_value (t : Triangle 25 29 32) : 
  maxTableHeight t = 64 * Real.sqrt 29106 / 1425 := by sorry

end NUMINAMATH_CALUDE_max_table_height_value_l623_62390


namespace NUMINAMATH_CALUDE_num_choices_eq_ten_l623_62387

/-- The number of science subjects -/
def num_science : ℕ := 3

/-- The number of humanities subjects -/
def num_humanities : ℕ := 3

/-- The total number of subjects to choose from -/
def total_subjects : ℕ := num_science + num_humanities

/-- The number of subjects that must be chosen -/
def subjects_to_choose : ℕ := 3

/-- The minimum number of science subjects that must be chosen -/
def min_science : ℕ := 2

/-- The function that calculates the number of ways to choose subjects -/
def num_choices : ℕ := sorry

theorem num_choices_eq_ten : num_choices = 10 := by sorry

end NUMINAMATH_CALUDE_num_choices_eq_ten_l623_62387


namespace NUMINAMATH_CALUDE_repeating_decimal_calculation_l623_62386

/-- Represents a repeating decimal with a two-digit repeating part -/
def RepeatingDecimal (a b : ℕ) : ℚ := a * 10 + b / 99

/-- The main theorem to prove -/
theorem repeating_decimal_calculation :
  let x : ℚ := RepeatingDecimal 5 4
  let y : ℚ := RepeatingDecimal 1 8
  (x / y) * (1 / 2) = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_calculation_l623_62386


namespace NUMINAMATH_CALUDE_speed_conversion_l623_62367

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- Given speed in meters per second -/
def given_speed_mps : ℝ := 23.3352

/-- Calculated speed in kilometers per hour -/
def calculated_speed_kmph : ℝ := 84.00672

theorem speed_conversion : given_speed_mps * mps_to_kmph = calculated_speed_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l623_62367


namespace NUMINAMATH_CALUDE_consecutive_integers_with_properties_l623_62326

def sumOfDigits (n : ℕ) : ℕ := sorry

def isPrime (n : ℕ) : Prop := sorry

def isPerfect (n : ℕ) : Prop := sorry

def isSquareFree (n : ℕ) : Prop := sorry

def numberOfDivisors (n : ℕ) : ℕ := sorry

def hasOnePrimeDivisorLessThan10 (n : ℕ) : Prop := sorry

def atMostTwoDigitsEqualOne (n : ℕ) : Prop := sorry

theorem consecutive_integers_with_properties :
  ∃ (n : ℕ),
    (isPrime (sumOfDigits n) ∨ isPrime (sumOfDigits (n + 1)) ∨ isPrime (sumOfDigits (n + 2))) ∧
    (isPerfect (sumOfDigits n) ∨ isPerfect (sumOfDigits (n + 1)) ∨ isPerfect (sumOfDigits (n + 2))) ∧
    (sumOfDigits n = numberOfDivisors n ∨ sumOfDigits (n + 1) = numberOfDivisors (n + 1) ∨ sumOfDigits (n + 2) = numberOfDivisors (n + 2)) ∧
    (atMostTwoDigitsEqualOne n ∧ atMostTwoDigitsEqualOne (n + 1) ∧ atMostTwoDigitsEqualOne (n + 2)) ∧
    (∃ (m : ℕ), (n + 11 = m^2) ∨ (n + 12 = m^2) ∨ (n + 13 = m^2)) ∧
    (hasOnePrimeDivisorLessThan10 n ∧ hasOnePrimeDivisorLessThan10 (n + 1) ∧ hasOnePrimeDivisorLessThan10 (n + 2)) ∧
    (isSquareFree n ∧ isSquareFree (n + 1) ∧ isSquareFree (n + 2)) ∧
    n = 2013 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_with_properties_l623_62326


namespace NUMINAMATH_CALUDE_min_value_theorem_l623_62378

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b + a * c + b * c + 2 * Real.sqrt 5 = 6 - a ^ 2) :
  2 * a + b + c ≥ 2 * Real.sqrt 5 - 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l623_62378


namespace NUMINAMATH_CALUDE_total_weight_juvenile_female_muscovy_l623_62392

/-- Given a pond with ducks, calculate the total weight of juvenile female Muscovy ducks -/
theorem total_weight_juvenile_female_muscovy (total_ducks : ℕ) 
  (muscovy_percentage mallard_percentage : ℚ)
  (female_muscovy_percentage : ℚ) 
  (juvenile_female_muscovy_percentage : ℚ)
  (avg_weight_juvenile_female_muscovy : ℚ) :
  total_ducks = 120 →
  muscovy_percentage = 45/100 →
  mallard_percentage = 35/100 →
  female_muscovy_percentage = 60/100 →
  juvenile_female_muscovy_percentage = 30/100 →
  avg_weight_juvenile_female_muscovy = 7/2 →
  ∃ (weight : ℚ), weight = 63/2 ∧ 
    weight = (total_ducks : ℚ) * muscovy_percentage * female_muscovy_percentage * 
             juvenile_female_muscovy_percentage * avg_weight_juvenile_female_muscovy :=
by sorry

end NUMINAMATH_CALUDE_total_weight_juvenile_female_muscovy_l623_62392


namespace NUMINAMATH_CALUDE_directrix_of_symmetrical_parabola_l623_62391

-- Define the original parabola
def original_parabola (x y : ℝ) : Prop := y = 2 * x^2

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop := y = x

-- Define the symmetrical parabola
def symmetrical_parabola (x y : ℝ) : Prop := y^2 = (1/2) * x

-- Theorem statement
theorem directrix_of_symmetrical_parabola :
  ∀ (x : ℝ), (∃ (y : ℝ), symmetrical_parabola x y) → (x = -1/8) = 
  (∀ (p : ℝ), p ≠ 0 → (∃ (h k : ℝ), ∀ (x y : ℝ), 
    symmetrical_parabola x y ↔ (y - k)^2 = 4 * p * (x - h) ∧ x = h - p)) :=
by sorry

end NUMINAMATH_CALUDE_directrix_of_symmetrical_parabola_l623_62391


namespace NUMINAMATH_CALUDE_min_green_beads_exact_min_green_beads_l623_62361

/-- Represents a necklace with red, blue, and green beads. -/
structure Necklace where
  total : Nat
  red : Nat
  blue : Nat
  green : Nat
  sum_eq_total : red + blue + green = total
  red_between_blue : red ≥ blue
  green_between_red : green ≥ red

/-- The minimum number of green beads in a necklace of 80 beads satisfying the given conditions. -/
theorem min_green_beads (n : Necklace) (h : n.total = 80) : n.green ≥ 27 := by
  sorry

/-- The minimum number of green beads is exactly 27. -/
theorem exact_min_green_beads : ∃ n : Necklace, n.total = 80 ∧ n.green = 27 := by
  sorry

end NUMINAMATH_CALUDE_min_green_beads_exact_min_green_beads_l623_62361


namespace NUMINAMATH_CALUDE_square_sum_geq_double_product_l623_62399

theorem square_sum_geq_double_product (a b : ℝ) : a^2 + b^2 ≥ 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_double_product_l623_62399


namespace NUMINAMATH_CALUDE_angle_expression_value_l623_62389

/-- Given that point P(1,2) is on the terminal side of angle α, 
    prove that (6sinα + 8cosα) / (3sinα - 2cosα) = 5 -/
theorem angle_expression_value (α : Real) (h : Complex.exp (α * Complex.I) = ⟨1, 2⟩) :
  (6 * Real.sin α + 8 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_value_l623_62389


namespace NUMINAMATH_CALUDE_yuri_puppies_count_l623_62388

/-- The number of puppies Yuri adopted in the first week -/
def first_week : ℕ := 20

/-- The number of puppies Yuri adopted in the second week -/
def second_week : ℕ := (2 * first_week) / 5

/-- The number of puppies Yuri adopted in the third week -/
def third_week : ℕ := (3 * second_week) / 8

/-- The number of puppies Yuri adopted in the fourth week -/
def fourth_week : ℕ := 2 * second_week

/-- The number of puppies Yuri adopted in the fifth week -/
def fifth_week : ℕ := first_week + 10

/-- The number of puppies Yuri adopted in the sixth week -/
def sixth_week : ℕ := 2 * third_week - 5

/-- The number of puppies Yuri adopted in the seventh week -/
def seventh_week : ℕ := 2 * sixth_week

/-- The number of puppies Yuri adopted in half of the eighth week -/
def eighth_week_half : ℕ := (5 * seventh_week) / 6

/-- The total number of puppies Yuri adopted -/
def total_puppies : ℕ := first_week + second_week + third_week + fourth_week + 
                         fifth_week + sixth_week + seventh_week + eighth_week_half

theorem yuri_puppies_count : total_puppies = 81 := by
  sorry

end NUMINAMATH_CALUDE_yuri_puppies_count_l623_62388


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l623_62327

def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 4

def nonzero_coeff_product (P : ℝ → ℝ) : ℝ :=
  3 * (-5) * 4

def coeff_abs_sum (P : ℝ → ℝ) : ℝ :=
  |3| + |-5| + |4|

def Q (x : ℝ) : ℝ :=
  (nonzero_coeff_product P) * x^3 + (nonzero_coeff_product P) * x + (nonzero_coeff_product P)

def R (x : ℝ) : ℝ :=
  (coeff_abs_sum P) * x^3 - (coeff_abs_sum P) * x + (coeff_abs_sum P)

theorem polynomial_evaluation :
  Q 1 = -180 ∧ R 1 = 12 ∧ Q 1 ≠ R 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l623_62327


namespace NUMINAMATH_CALUDE_matrix_square_zero_implication_l623_62351

theorem matrix_square_zero_implication (n : ℕ) (M N : Matrix (Fin n) (Fin n) ℝ) 
  (h : (M * N)^2 = 0) :
  (n = 2 → (N * M)^2 = 0) ∧ 
  (n ≥ 3 → ∃ (M' N' : Matrix (Fin n) (Fin n) ℝ), (M' * N')^2 = 0 ∧ (N' * M')^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_matrix_square_zero_implication_l623_62351


namespace NUMINAMATH_CALUDE_arcade_spending_amount_l623_62328

def weekly_allowance : ℚ := 345/100

def arcade_spending (x : ℚ) : Prop :=
  let remaining_after_arcade := weekly_allowance - x
  let toy_store_spending := (1/3) * remaining_after_arcade
  let candy_store_spending := 92/100
  remaining_after_arcade - toy_store_spending = candy_store_spending

theorem arcade_spending_amount :
  ∃ (x : ℚ), arcade_spending x ∧ x = 207/100 := by sorry

end NUMINAMATH_CALUDE_arcade_spending_amount_l623_62328


namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l623_62396

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 2 * Real.log x

theorem monotonic_increase_interval :
  ∀ x : ℝ, x > 0 → (∀ y : ℝ, y > x → f y > f x) ↔ x > Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l623_62396


namespace NUMINAMATH_CALUDE_function_properties_l623_62374

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + x^2

theorem function_properties :
  ∀ a : ℝ,
  (∀ x : ℝ, x ∈ Set.Icc 1 (exp 1) → f a x ∈ Set.Icc (f a 1) (f a (exp 1))) ∧
  (a = -4 → ∀ x : ℝ, x ∈ Set.Icc 1 (exp 1) → f a x ≤ f a (exp 1)) ∧
  (a = -4 → f a (exp 1) = (exp 1)^2 - 4) ∧
  (∃ n : ℕ, n ≤ 2 ∧ (∃ S : Finset ℝ, S.card = n ∧ ∀ x ∈ S, x ∈ Set.Icc 1 (exp 1) ∧ f a x = 0)) ∧
  (a > 0 → ¬∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 (exp 1) → x₂ ∈ Set.Icc 1 (exp 1) →
    |f a x₁ - f a x₂| ≤ |1/x₁ - 1/x₂|) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l623_62374


namespace NUMINAMATH_CALUDE_min_value_a_l623_62379

theorem min_value_a : 
  (∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 2 ∧ ∃ a : ℝ, a * 3^x ≥ x - 1) → 
  (∃ a_min : ℝ, a_min = -6 ∧ ∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 2 ∧ a * 3^x ≥ x - 1) → a ≥ a_min) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l623_62379


namespace NUMINAMATH_CALUDE_reflection_of_A_l623_62317

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem reflection_of_A : reflect_x (-4, 3) = (-4, -3) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_A_l623_62317


namespace NUMINAMATH_CALUDE_smallest_number_l623_62373

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

def number_a : Nat := base_to_decimal [5, 8] 9
def number_b : Nat := base_to_decimal [0, 1, 2] 6
def number_c : Nat := base_to_decimal [0, 0, 0, 1] 4
def number_d : Nat := base_to_decimal [1, 1, 1, 1, 1, 1] 2

theorem smallest_number :
  number_d < number_a ∧ number_d < number_b ∧ number_d < number_c :=
sorry

end NUMINAMATH_CALUDE_smallest_number_l623_62373


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l623_62362

/-- Represents a square grid with side length n -/
structure Grid (n : ℕ) where
  size : ℕ
  size_eq : size = n * n

/-- The value of a cell in the grid given its row and column -/
def cellValue (g : Grid 9) (row col : ℕ) : ℕ :=
  (row - 1) * 9 + col

/-- The sum of the corner values in a 9x9 grid -/
def cornerSum (g : Grid 9) : ℕ :=
  cellValue g 1 1 + cellValue g 1 9 + cellValue g 9 1 + cellValue g 9 9

/-- Theorem: The sum of the corner values in a 9x9 grid is 164 -/
theorem corner_sum_is_164 (g : Grid 9) : cornerSum g = 164 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_164_l623_62362


namespace NUMINAMATH_CALUDE_power_of_two_between_powers_of_ten_l623_62347

theorem power_of_two_between_powers_of_ten (t : ℕ+) : 
  (10 ^ (t.val - 1 : ℕ) < 2 ^ 64) ∧ (2 ^ 64 < 10 ^ t.val) → t = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_between_powers_of_ten_l623_62347


namespace NUMINAMATH_CALUDE_food_consumption_reduction_l623_62370

/-- Calculates the required reduction in food consumption per student to maintain the same total cost, given a decrease in the number of students and an increase in food price. -/
theorem food_consumption_reduction 
  (student_decrease_rate : ℝ) 
  (food_price_increase_rate : ℝ) 
  (ε : ℝ) -- tolerance for approximation
  (h1 : student_decrease_rate = 0.05)
  (h2 : food_price_increase_rate = 0.20)
  (h3 : ε > 0)
  : ∃ (reduction_rate : ℝ), 
    abs (reduction_rate - (1 - 1 / ((1 - student_decrease_rate) * (1 + food_price_increase_rate)))) < ε ∧ 
    abs (reduction_rate - 0.1228) < ε := by
  sorry

end NUMINAMATH_CALUDE_food_consumption_reduction_l623_62370


namespace NUMINAMATH_CALUDE_expression_evaluation_l623_62353

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 11) + 2 = -x^4 + 3*x^3 - 5*x^2 + 11*x + 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l623_62353


namespace NUMINAMATH_CALUDE_not_even_not_odd_composition_l623_62385

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Statement of the theorem
theorem not_even_not_odd_composition (f : ℝ → ℝ) (c : ℝ) (h : OddFunction f) :
  ¬ (EvenFunction (fun x ↦ f (f (x + c)))) ∧ ¬ (OddFunction (fun x ↦ f (f (x + c)))) :=
sorry

end NUMINAMATH_CALUDE_not_even_not_odd_composition_l623_62385


namespace NUMINAMATH_CALUDE_range_of_a_l623_62310

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- Define the theorem
theorem range_of_a (a : ℝ) : A ∪ B a = B a → -1 < a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l623_62310


namespace NUMINAMATH_CALUDE_pure_gala_trees_l623_62337

/-- Represents the apple orchard problem --/
def apple_orchard_problem (T F G : ℕ) : Prop :=
  (F : ℚ) + 0.1 * T = 238 ∧
  F = (3/4 : ℚ) * T ∧
  G = T - F

/-- Theorem stating the number of pure Gala trees --/
theorem pure_gala_trees : ∃ T F G : ℕ, 
  apple_orchard_problem T F G ∧ G = 70 := by
  sorry

end NUMINAMATH_CALUDE_pure_gala_trees_l623_62337


namespace NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l623_62393

theorem arithmetic_sequence_term_count 
  (a : ℕ) (d : ℕ) (l : ℕ) (n : ℕ) 
  (h1 : a = 7) 
  (h2 : d = 2) 
  (h3 : l = 145) 
  (h4 : l = a + (n - 1) * d) : 
  n = 70 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l623_62393


namespace NUMINAMATH_CALUDE_matches_left_after_2022_l623_62336

/-- The number of matchsticks needed to form a digit --/
def matchsticks_for_digit (d : Nat) : Nat :=
  if d = 2 then 5
  else if d = 0 then 6
  else 0  -- We only care about 2 and 0 for this problem

/-- The number of matchsticks needed to form the year 2022 --/
def matchsticks_for_2022 : Nat :=
  matchsticks_for_digit 2 * 3 + matchsticks_for_digit 0

/-- The initial number of matches in the box --/
def initial_matches : Nat := 30

/-- Theorem: After forming 2022 with matchsticks, 9 matches will be left --/
theorem matches_left_after_2022 :
  initial_matches - matchsticks_for_2022 = 9 := by
  sorry


end NUMINAMATH_CALUDE_matches_left_after_2022_l623_62336


namespace NUMINAMATH_CALUDE_not_all_trihedral_angles_form_equilateral_triangles_l623_62372

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180

/-- Represents a plane intersecting a trihedral angle -/
structure Intersection where
  angle : TrihedralAngle
  plane : Unit  -- We don't need to define the plane explicitly for this problem

/-- Predicate to check if an intersection forms an equilateral triangle -/
def forms_equilateral_triangle (i : Intersection) : Prop :=
  -- This would involve complex geometric calculations in reality
  sorry

/-- Theorem stating that not all trihedral angles can be intersected to form equilateral triangles -/
theorem not_all_trihedral_angles_form_equilateral_triangles :
  ∃ (t : TrihedralAngle), ∀ (p : Unit), ¬(forms_equilateral_triangle ⟨t, p⟩) :=
sorry

end NUMINAMATH_CALUDE_not_all_trihedral_angles_form_equilateral_triangles_l623_62372


namespace NUMINAMATH_CALUDE_solve_equation_l623_62316

/-- Custom operation # -/
def hash (a b : ℝ) : ℝ := a * b - b + b^2

/-- Theorem statement -/
theorem solve_equation (x : ℝ) (h : hash x 7 = 63) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l623_62316


namespace NUMINAMATH_CALUDE_equations_solutions_l623_62342

-- Define the equations
def equation1 (x : ℝ) : Prop := 6 * x - 7 = 4 * x - 5
def equation2 (x : ℝ) : Prop := (3 * x - 1) / 4 - 1 = (5 * x - 7) / 6

-- State the theorem
theorem equations_solutions :
  (∃ x : ℝ, equation1 x ∧ x = 1) ∧
  (∃ x : ℝ, equation2 x ∧ x = -1) :=
sorry

end NUMINAMATH_CALUDE_equations_solutions_l623_62342


namespace NUMINAMATH_CALUDE_smallest_multiple_of_5_and_711_l623_62357

theorem smallest_multiple_of_5_and_711 : 
  ∀ n : ℕ, n > 0 ∧ 5 ∣ n ∧ 711 ∣ n → n ≥ 3555 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_5_and_711_l623_62357


namespace NUMINAMATH_CALUDE_count_valid_m_l623_62329

def is_valid (m : ℕ+) : Prop :=
  ∃ k : ℕ+, (2310 : ℚ) / ((m : ℚ)^2 - 2) = k

theorem count_valid_m :
  ∃! (s : Finset ℕ+), s.card = 3 ∧ ∀ m : ℕ+, m ∈ s ↔ is_valid m :=
sorry

end NUMINAMATH_CALUDE_count_valid_m_l623_62329


namespace NUMINAMATH_CALUDE_prime_square_product_theorem_l623_62311

theorem prime_square_product_theorem :
  ∀ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℕ),
    Prime x₁ ∧ Prime x₂ ∧ Prime x₃ ∧ Prime x₄ ∧
    Prime x₅ ∧ Prime x₆ ∧ Prime x₇ ∧ Prime x₈ →
    4 * (x₁ * x₂ * x₃ * x₄ * x₅ * x₆ * x₇ * x₈) -
    (x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 + x₆^2 + x₇^2 + x₈^2) = 992 →
    x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧
    x₅ = 2 ∧ x₆ = 2 ∧ x₇ = 2 ∧ x₈ = 2 :=
by sorry

end NUMINAMATH_CALUDE_prime_square_product_theorem_l623_62311


namespace NUMINAMATH_CALUDE_simple_interest_rate_interest_rate_problem_l623_62365

/-- Simple interest calculation --/
theorem simple_interest_rate (principal amount : ℚ) (time : ℕ) (rate : ℚ) : 
  principal * (1 + rate * time) = amount →
  rate = (amount - principal) / (principal * time) :=
by
  sorry

/-- Prove that the interest rate is 5% given the problem conditions --/
theorem interest_rate_problem :
  let principal : ℚ := 600
  let amount : ℚ := 720
  let time : ℕ := 4
  let rate : ℚ := (amount - principal) / (principal * time)
  rate = 5 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_interest_rate_problem_l623_62365


namespace NUMINAMATH_CALUDE_soccer_substitutions_remainder_l623_62352

/-- Number of ways to make n substitutions in a soccer game -/
def substitutions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => 11 * (12 - k) * substitutions k

/-- Total number of ways to make 0 to 4 substitutions -/
def totalSubstitutions : ℕ :=
  (List.range 5).map substitutions |>.sum

/-- The remainder when the total number of substitutions is divided by 1000 -/
theorem soccer_substitutions_remainder :
  totalSubstitutions % 1000 = 522 := by
  sorry

end NUMINAMATH_CALUDE_soccer_substitutions_remainder_l623_62352


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l623_62395

/-- Given an arithmetic sequence where the sum of the first and fifth terms is 14,
    prove that the third term is 7. -/
theorem arithmetic_sequence_third_term
  (a : ℝ)  -- First term of the sequence
  (d : ℝ)  -- Common difference of the sequence
  (h : a + (a + 4*d) = 14)  -- Sum of first and fifth terms is 14
  : a + 2*d = 7 :=  -- Third term is 7
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l623_62395


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l623_62350

theorem simplify_fraction_product : 5 * (21 / 8) * (32 / -63) = -20 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l623_62350


namespace NUMINAMATH_CALUDE_pumpkins_eaten_by_rabbits_l623_62394

/-- Represents the number of pumpkins Sara initially grew -/
def initial_pumpkins : ℕ := 43

/-- Represents the number of pumpkins Sara has left after rabbits ate some -/
def remaining_pumpkins : ℕ := 20

/-- Represents the number of pumpkins eaten by rabbits -/
def eaten_pumpkins : ℕ := initial_pumpkins - remaining_pumpkins

/-- Theorem stating that the number of pumpkins eaten by rabbits is the difference between
    the initial number of pumpkins and the remaining number of pumpkins -/
theorem pumpkins_eaten_by_rabbits :
  eaten_pumpkins = initial_pumpkins - remaining_pumpkins :=
by
  sorry

end NUMINAMATH_CALUDE_pumpkins_eaten_by_rabbits_l623_62394


namespace NUMINAMATH_CALUDE_prime_pairs_satisfying_equation_l623_62302

theorem prime_pairs_satisfying_equation :
  ∀ (p q : ℕ), Prime p → Prime q →
  (∃ (x y z : ℕ), p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_satisfying_equation_l623_62302


namespace NUMINAMATH_CALUDE_students_in_neither_clubs_l623_62359

/-- Represents the number of students in various categories in a class --/
structure ClassMembers where
  total : ℕ
  chinese : ℕ
  math : ℕ
  both : ℕ

/-- Calculates the number of students in neither the Chinese nor Math club --/
def studentsInNeither (c : ClassMembers) : ℕ :=
  c.total - (c.chinese + c.math - c.both)

/-- Theorem stating the number of students in neither club for the given scenario --/
theorem students_in_neither_clubs (c : ClassMembers) 
  (h_total : c.total = 55)
  (h_chinese : c.chinese = 32)
  (h_math : c.math = 36)
  (h_both : c.both = 18) :
  studentsInNeither c = 5 := by
  sorry

#eval studentsInNeither { total := 55, chinese := 32, math := 36, both := 18 }

end NUMINAMATH_CALUDE_students_in_neither_clubs_l623_62359


namespace NUMINAMATH_CALUDE_rajans_position_l623_62307

/-- Given a row of boys, this theorem proves Rajan's position from the left end. -/
theorem rajans_position 
  (total_boys : ℕ) 
  (vinays_position_from_right : ℕ) 
  (boys_between : ℕ) 
  (h1 : total_boys = 24) 
  (h2 : vinays_position_from_right = 10) 
  (h3 : boys_between = 8) : 
  total_boys - (vinays_position_from_right - 1 + boys_between + 1) = 6 := by
sorry

end NUMINAMATH_CALUDE_rajans_position_l623_62307


namespace NUMINAMATH_CALUDE_probability_two_teachers_in_A_l623_62348

def num_teachers : ℕ := 3
def num_places : ℕ := 2

def total_assignments : ℕ := num_places ^ num_teachers

def assignments_with_two_in_A : ℕ := (Nat.choose num_teachers 2)

theorem probability_two_teachers_in_A :
  (assignments_with_two_in_A : ℚ) / total_assignments = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_teachers_in_A_l623_62348


namespace NUMINAMATH_CALUDE_man_speed_calculation_man_speed_approximately_5_004_l623_62349

/-- Calculates the speed of a man walking opposite to a train, given the train's length, speed, and time to cross the man. -/
theorem man_speed_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / crossing_time
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * 3600 / 1000
  man_speed_kmh

/-- The speed of the man is approximately 5.004 km/h -/
theorem man_speed_approximately_5_004 :
  ∃ ε > 0, |man_speed_calculation 200 114.99 6 - 5.004| < ε :=
sorry

end NUMINAMATH_CALUDE_man_speed_calculation_man_speed_approximately_5_004_l623_62349


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l623_62325

/-- Represents the height of a tree that quadruples each year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (4 ^ years)

/-- The problem statement -/
theorem tree_height_after_two_years 
  (h : tree_height 1 4 = 256) : 
  tree_height 1 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l623_62325


namespace NUMINAMATH_CALUDE_wedge_volume_specific_case_l623_62333

/-- Represents a cylindrical log with a wedge cut out. -/
structure WedgedLog where
  diameter : ℝ
  firstCutAngle : ℝ
  secondCutAngle : ℝ
  intersectionPoint : ℕ

/-- Calculates the volume of the wedge cut from the log. -/
def wedgeVolume (log : WedgedLog) : ℝ :=
  sorry

/-- Theorem stating the volume of the wedge under specific conditions. -/
theorem wedge_volume_specific_case :
  let log : WedgedLog := {
    diameter := 16,
    firstCutAngle := 90,  -- perpendicular cut
    secondCutAngle := 60,
    intersectionPoint := 1
  }
  wedgeVolume log = 512 * Real.pi := by sorry

end NUMINAMATH_CALUDE_wedge_volume_specific_case_l623_62333


namespace NUMINAMATH_CALUDE_max_gel_pens_l623_62354

/-- Represents the number of pens of each type -/
structure PenCounts where
  ballpoint : ℕ
  gel : ℕ
  fountain : ℕ

/-- Checks if the given pen counts satisfy all conditions -/
def is_valid_count (counts : PenCounts) : Prop :=
  counts.ballpoint + counts.gel + counts.fountain = 20 ∧
  counts.ballpoint > 0 ∧ counts.gel > 0 ∧ counts.fountain > 0 ∧
  10 * counts.ballpoint + 50 * counts.gel + 80 * counts.fountain = 1000

/-- Theorem stating that the maximum number of gel pens is 13 -/
theorem max_gel_pens : 
  (∃ (counts : PenCounts), is_valid_count counts ∧ counts.gel = 13) ∧
  (∀ (counts : PenCounts), is_valid_count counts → counts.gel ≤ 13) :=
sorry

end NUMINAMATH_CALUDE_max_gel_pens_l623_62354


namespace NUMINAMATH_CALUDE_remainder_256_div_13_l623_62314

theorem remainder_256_div_13 : ∃ q r : ℤ, 256 = 13 * q + r ∧ 0 ≤ r ∧ r < 13 ∧ r = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_256_div_13_l623_62314


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l623_62300

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p, p < k → Prime p → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 221) ∧
  (has_no_prime_factors_less_than 221 12) ∧
  (∀ m : ℕ, m < 221 → ¬(is_composite m ∧ has_no_prime_factors_less_than m 12)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l623_62300


namespace NUMINAMATH_CALUDE_power_twentyseven_x_plus_one_l623_62380

theorem power_twentyseven_x_plus_one (x : ℝ) (h : (3 : ℝ) ^ (2 * x) = 5) : 
  (27 : ℝ) ^ (x + 1) = 135 := by sorry

end NUMINAMATH_CALUDE_power_twentyseven_x_plus_one_l623_62380


namespace NUMINAMATH_CALUDE_cube_root_sum_equals_five_l623_62384

theorem cube_root_sum_equals_five :
  (Real.rpow (25 + 10 * Real.sqrt 5) (1/3 : ℝ)) + (Real.rpow (25 - 10 * Real.sqrt 5) (1/3 : ℝ)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_sum_equals_five_l623_62384


namespace NUMINAMATH_CALUDE_books_read_second_week_l623_62375

theorem books_read_second_week :
  ∀ (total_books : ℕ) 
    (first_week_books : ℕ) 
    (later_weeks_books : ℕ) 
    (total_weeks : ℕ),
  total_books = 54 →
  first_week_books = 6 →
  later_weeks_books = 9 →
  total_weeks = 7 →
  total_books = first_week_books + (total_weeks - 2) * later_weeks_books + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_books_read_second_week_l623_62375


namespace NUMINAMATH_CALUDE_halloween_candy_distribution_l623_62381

theorem halloween_candy_distribution (initial_candy : ℕ) (eaten_candy : ℕ) (num_piles : ℕ) 
  (h1 : initial_candy = 108)
  (h2 : eaten_candy = 36)
  (h3 : num_piles = 8) :
  (initial_candy - eaten_candy) / num_piles = 9 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_distribution_l623_62381


namespace NUMINAMATH_CALUDE_player_time_on_field_l623_62324

/-- Proves that each player in a team of 10 will play for 36 minutes in a 45-minute match with 8 players always on the field. -/
theorem player_time_on_field
  (team_size : ℕ)
  (players_on_field : ℕ)
  (match_duration : ℕ)
  (h1 : team_size = 10)
  (h2 : players_on_field = 8)
  (h3 : match_duration = 45)
  : (players_on_field * match_duration) / team_size = 36 := by
  sorry

#eval (8 * 45) / 10  -- Should output 36

end NUMINAMATH_CALUDE_player_time_on_field_l623_62324


namespace NUMINAMATH_CALUDE_no_real_solutions_l623_62341

theorem no_real_solutions : ¬∃ (x : ℝ), -3 * x - 8 = 8 * x^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l623_62341


namespace NUMINAMATH_CALUDE_license_plate_combinations_l623_62346

def number_of_letter_combinations : ℕ :=
  (Nat.choose 26 2) * 24 * (5 * 4 * 3 / (2 * 2))

def number_of_digit_combinations : ℕ := 10 * 9 * 8

theorem license_plate_combinations :
  number_of_letter_combinations * number_of_digit_combinations = 5644800 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l623_62346


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l623_62368

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l623_62368


namespace NUMINAMATH_CALUDE_no_rational_roots_l623_62303

def polynomial (x : ℚ) : ℚ := 3 * x^4 - 7 * x^3 - 4 * x^2 + 8 * x + 3

theorem no_rational_roots :
  ∀ x : ℚ, polynomial x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_no_rational_roots_l623_62303


namespace NUMINAMATH_CALUDE_factoring_a_squared_minus_nine_l623_62369

theorem factoring_a_squared_minus_nine (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factoring_a_squared_minus_nine_l623_62369


namespace NUMINAMATH_CALUDE_quadratic_discriminant_equality_l623_62397

theorem quadratic_discriminant_equality (a b c x : ℝ) (h1 : a ≠ 0) (h2 : a * x^2 + b * x + c = 0) : 
  b^2 - 4*a*c = (2*a*x + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_equality_l623_62397


namespace NUMINAMATH_CALUDE_multiply_add_distribute_l623_62345

theorem multiply_add_distribute : 57 * 33 + 13 * 33 = 2310 := by
  sorry

end NUMINAMATH_CALUDE_multiply_add_distribute_l623_62345


namespace NUMINAMATH_CALUDE_binomial_congruence_l623_62376

theorem binomial_congruence (p n : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) (hn : n > 0) 
  (h_cong : n ≡ 1 [MOD p]) : 
  (Nat.choose (n * p) p) ≡ n [MOD p^4] := by sorry

end NUMINAMATH_CALUDE_binomial_congruence_l623_62376


namespace NUMINAMATH_CALUDE_halloween_candy_count_l623_62339

/-- Represents the number of candies each person has -/
structure CandyCount where
  bob : Nat
  mary : Nat
  john : Nat
  sue : Nat
  sam : Nat

/-- The total number of candies for all friends -/
def totalCandies (cc : CandyCount) : Nat :=
  cc.bob + cc.mary + cc.john + cc.sue + cc.sam

/-- Theorem stating that the total number of candies is 50 -/
theorem halloween_candy_count :
  ∃ (cc : CandyCount),
    cc.bob = 10 ∧
    cc.mary = 5 ∧
    cc.john = 5 ∧
    cc.sue = 20 ∧
    cc.sam = 10 ∧
    totalCandies cc = 50 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_count_l623_62339


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_angles_l623_62363

theorem isosceles_right_triangle_angles (α : ℝ) :
  α > 0 ∧ α < 90 →
  (α + α + 90 = 180) →
  α = 45 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_angles_l623_62363


namespace NUMINAMATH_CALUDE_triangle_third_side_existence_l623_62322

theorem triangle_third_side_existence (a b c : ℝ) : 
  a = 3 → b = 5 → c = 4 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_existence_l623_62322


namespace NUMINAMATH_CALUDE_triangle_existence_and_uniqueness_l623_62360

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the given points
variable (D E F : Point)

-- Define the conditions
def is_midpoint (M : Point) (A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def is_trisection_point (E B C : Point) : Prop :=
  E.x = B.x + (C.x - B.x) / 3 ∧ E.y = B.y + (C.y - B.y) / 3

def is_quarter_point (F C A : Point) : Prop :=
  F.x = C.x + 3 * (A.x - C.x) / 4 ∧ F.y = C.y + 3 * (A.y - C.y) / 4

-- State the theorem
theorem triangle_existence_and_uniqueness :
  ∃! (ABC : Triangle),
    is_midpoint D ABC.A ABC.B ∧
    is_trisection_point E ABC.B ABC.C ∧
    is_quarter_point F ABC.C ABC.A :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_and_uniqueness_l623_62360


namespace NUMINAMATH_CALUDE_shopping_trip_solution_l623_62331

/-- The exchange rate from USD to CAD -/
def exchange_rate : ℚ := 8 / 5

/-- The amount spent in CAD -/
def amount_spent : ℕ := 80

/-- The function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The theorem stating the solution to the problem -/
theorem shopping_trip_solution (d : ℕ) : 
  (exchange_rate * d - amount_spent = d) → sum_of_digits d = 7 := by
  sorry

#eval sum_of_digits 133  -- This should output 7

end NUMINAMATH_CALUDE_shopping_trip_solution_l623_62331


namespace NUMINAMATH_CALUDE_binary_to_base4_correct_l623_62335

/-- Converts a binary number to base 4 --/
def binary_to_base4 (b : ℕ) : ℕ := sorry

/-- The binary representation of the number --/
def binary_num : ℕ := 110110100

/-- The base 4 representation of the number --/
def base4_num : ℕ := 31220

/-- Theorem stating that the conversion of the binary number to base 4 is correct --/
theorem binary_to_base4_correct : binary_to_base4 binary_num = base4_num := by sorry

end NUMINAMATH_CALUDE_binary_to_base4_correct_l623_62335


namespace NUMINAMATH_CALUDE_sashas_initial_questions_l623_62321

/-- Proves that given Sasha's completion rate, work time, and remaining questions,
    the initial number of questions is 60. -/
theorem sashas_initial_questions
  (completion_rate : ℕ)
  (work_time : ℕ)
  (remaining_questions : ℕ)
  (h1 : completion_rate = 15)
  (h2 : work_time = 2)
  (h3 : remaining_questions = 30) :
  completion_rate * work_time + remaining_questions = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_sashas_initial_questions_l623_62321


namespace NUMINAMATH_CALUDE_range_of_a_l623_62334

def f (x : ℝ) := x^2 - 4*x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4) a, f x ∈ Set.Icc (-4) 32) →
  (∀ y ∈ Set.Icc (-4) 32, ∃ x ∈ Set.Icc (-4) a, f x = y) →
  a ∈ Set.Icc 2 8 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l623_62334


namespace NUMINAMATH_CALUDE_palindrome_percentage_l623_62323

/-- A palindrome between 1000 and 2000 -/
structure Palindrome :=
  (a b c : Fin 10)

/-- The set of all palindromes between 1000 and 2000 -/
def all_palindromes : Finset Palindrome :=
  sorry

/-- The set of palindromes containing at least one 3 or 5 (except in the first digit) -/
def palindromes_with_3_or_5 : Finset Palindrome :=
  sorry

/-- The percentage of palindromes with 3 or 5 -/
def percentage_with_3_or_5 : ℚ :=
  (palindromes_with_3_or_5.card : ℚ) / (all_palindromes.card : ℚ) * 100

theorem palindrome_percentage :
  percentage_with_3_or_5 = 36 :=
sorry

end NUMINAMATH_CALUDE_palindrome_percentage_l623_62323


namespace NUMINAMATH_CALUDE_gcd_f_x_l623_62377

def f (x : ℤ) : ℤ := (5*x+3)*(11*x+2)*(14*x+7)*(3*x+8)

theorem gcd_f_x (x : ℤ) (h : ∃ k : ℤ, x = 3456 * k) : 
  Nat.gcd (Int.natAbs (f x)) (Int.natAbs x) = 48 := by
  sorry

end NUMINAMATH_CALUDE_gcd_f_x_l623_62377


namespace NUMINAMATH_CALUDE_phil_coin_collection_l623_62340

def initial_coins : ℕ := 250
def years_tripling : ℕ := 3
def weeks_in_year : ℕ := 52
def days_in_year : ℕ := 365
def coins_per_week_4th_year : ℕ := 5
def coins_every_second_day_5th_year : ℕ := 2
def coins_per_day_6th_year : ℕ := 1
def loss_fraction : ℚ := 1/3

def coins_after_loss : ℕ := 1160

theorem phil_coin_collection :
  let coins_after_3_years := initial_coins * (2^years_tripling)
  let coins_4th_year := coins_after_3_years + coins_per_week_4th_year * weeks_in_year
  let coins_5th_year := coins_4th_year + coins_every_second_day_5th_year * (days_in_year / 2)
  let coins_6th_year := coins_5th_year + coins_per_day_6th_year * days_in_year
  let coins_before_loss := coins_6th_year
  coins_after_loss = coins_before_loss - ⌊coins_before_loss * loss_fraction⌋ :=
by sorry

end NUMINAMATH_CALUDE_phil_coin_collection_l623_62340


namespace NUMINAMATH_CALUDE_rug_profit_calculation_l623_62355

theorem rug_profit_calculation (buying_price selling_price num_rugs tax_rate transport_fee : ℚ) 
  (h1 : buying_price = 40)
  (h2 : selling_price = 60)
  (h3 : num_rugs = 20)
  (h4 : tax_rate = 1/10)
  (h5 : transport_fee = 5) :
  let total_cost := buying_price * num_rugs + transport_fee * num_rugs
  let total_revenue := selling_price * num_rugs * (1 + tax_rate)
  let profit := total_revenue - total_cost
  profit = 420 := by sorry

end NUMINAMATH_CALUDE_rug_profit_calculation_l623_62355


namespace NUMINAMATH_CALUDE_min_sum_squares_l623_62315

theorem min_sum_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) 
  (heq : a^2 - 2015*a = b^2 - 2015*b) : 
  ∀ x y : ℝ, x > 0 → y > 0 → x ≠ y → x^2 - 2015*x = y^2 - 2015*y → 
  a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = (2015^2) / 2 := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l623_62315


namespace NUMINAMATH_CALUDE_new_person_weight_l623_62332

/-- Represents a group of people with their weights -/
structure WeightGroup where
  size : Nat
  total_weight : ℝ
  avg_weight : ℝ

/-- Represents the change in the group when a person is replaced -/
structure WeightChange where
  old_weight : ℝ
  new_weight : ℝ
  avg_increase : ℝ

/-- Theorem stating the weight of the new person -/
theorem new_person_weight 
  (group : WeightGroup)
  (change : WeightChange)
  (h1 : group.size = 8)
  (h2 : change.old_weight = 65)
  (h3 : change.avg_increase = 3.5)
  (h4 : ∀ w E, (w * (1 + E / 100) - w) ≤ change.avg_increase) :
  change.new_weight = 93 := by
sorry


end NUMINAMATH_CALUDE_new_person_weight_l623_62332


namespace NUMINAMATH_CALUDE_mary_has_more_than_marco_l623_62318

/-- Proves that Mary has $10 more than Marco after transactions --/
theorem mary_has_more_than_marco (marco_initial : ℕ) (mary_initial : ℕ) 
  (marco_gives : ℕ) (mary_spends : ℕ) : ℕ :=
by
  -- Define initial amounts
  have h1 : marco_initial = 24 := by sorry
  have h2 : mary_initial = 15 := by sorry
  
  -- Define amount Marco gives to Mary
  have h3 : marco_gives = marco_initial / 2 := by sorry
  
  -- Define amount Mary spends
  have h4 : mary_spends = 5 := by sorry
  
  -- Calculate final amounts
  let marco_final := marco_initial - marco_gives
  let mary_final := mary_initial + marco_gives - mary_spends
  
  -- Prove Mary has $10 more than Marco
  have h5 : mary_final - marco_final = 10 := by sorry
  
  exact 10

end NUMINAMATH_CALUDE_mary_has_more_than_marco_l623_62318


namespace NUMINAMATH_CALUDE_distinct_equals_odd_partitions_l623_62304

/-- The number of partitions of n into distinct positive integers -/
def distinctPartitions (n : ℕ) : ℕ := sorry

/-- The number of partitions of n into positive odd integers -/
def oddPartitions (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of partitions of n into distinct positive integers
    equals the number of partitions of n into positive odd integers -/
theorem distinct_equals_odd_partitions (n : ℕ+) :
  distinctPartitions n = oddPartitions n := by sorry

end NUMINAMATH_CALUDE_distinct_equals_odd_partitions_l623_62304


namespace NUMINAMATH_CALUDE_final_value_16_l623_62356

/-- A function that simulates the loop behavior --/
def loop_iteration (b : ℕ) : ℕ := b + 3

/-- The loop condition --/
def loop_condition (b : ℕ) : Prop := b < 16

/-- The theorem statement --/
theorem final_value_16 :
  ∃ (n : ℕ), 
    let b := 10
    let final_b := (loop_iteration^[n] b)
    (∀ k < n, loop_condition ((loop_iteration^[k]) b)) ∧
    ¬(loop_condition final_b) ∧
    final_b = 16 :=
sorry

end NUMINAMATH_CALUDE_final_value_16_l623_62356


namespace NUMINAMATH_CALUDE_amusement_park_visitors_l623_62343

/-- Represents the amusement park ticket sales problem -/
theorem amusement_park_visitors 
  (ticket_price : ℕ) 
  (saturday_visitors : ℕ) 
  (sunday_visitors : ℕ) 
  (total_revenue : ℕ) 
  (h1 : ticket_price = 3)
  (h2 : saturday_visitors = 200)
  (h3 : sunday_visitors = 300)
  (h4 : total_revenue = 3000) :
  ∃ (daily_visitors : ℕ), 
    daily_visitors * 5 * ticket_price + (saturday_visitors + sunday_visitors) * ticket_price = total_revenue ∧ 
    daily_visitors = 100 := by
  sorry


end NUMINAMATH_CALUDE_amusement_park_visitors_l623_62343


namespace NUMINAMATH_CALUDE_simplify_expression_l623_62398

theorem simplify_expression (m : ℝ) : (-m^4)^5 / m^5 * m = -m^14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l623_62398


namespace NUMINAMATH_CALUDE_quotient_problem_l623_62366

theorem quotient_problem (dividend : ℕ) (k : ℕ) (divisor : ℕ) :
  dividend = 64 → k = 8 → divisor = k → dividend / divisor = 8 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l623_62366


namespace NUMINAMATH_CALUDE_same_color_probability_l623_62382

def total_balls : ℕ := 20
def blue_balls : ℕ := 8
def green_balls : ℕ := 5
def red_balls : ℕ := 7

theorem same_color_probability :
  let prob_blue := (blue_balls / total_balls) ^ 2
  let prob_green := (green_balls / total_balls) ^ 2
  let prob_red := (red_balls / total_balls) ^ 2
  prob_blue + prob_green + prob_red = 117 / 200 := by
sorry

end NUMINAMATH_CALUDE_same_color_probability_l623_62382


namespace NUMINAMATH_CALUDE_shirt_cost_calculation_l623_62312

theorem shirt_cost_calculation :
  let discounted_shirts := 3
  let discounted_shirt_price := 15
  let first_discount := 0.1
  let second_discount := 0.05
  let taxed_shirts := 2
  let taxed_shirt_price := 20
  let first_tax := 0.05
  let second_tax := 0.03

  let discounted_price := discounted_shirt_price * (1 - first_discount) * (1 - second_discount)
  let taxed_price := taxed_shirt_price * (1 + first_tax) * (1 + second_tax)

  let total_cost := discounted_shirts * discounted_price + taxed_shirts * taxed_price

  total_cost = 81.735 := by sorry

end NUMINAMATH_CALUDE_shirt_cost_calculation_l623_62312


namespace NUMINAMATH_CALUDE_add_12345_seconds_to_1045am_l623_62301

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time: 10:45:00 -/
def initialTime : Time :=
  { hours := 10, minutes := 45, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 12345

/-- The expected final time: 13:45:45 -/
def expectedFinalTime : Time :=
  { hours := 13, minutes := 45, seconds := 45 }

theorem add_12345_seconds_to_1045am :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end NUMINAMATH_CALUDE_add_12345_seconds_to_1045am_l623_62301


namespace NUMINAMATH_CALUDE_inequality_proof_l623_62313

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a^4 + b^4 + c^4 + d^4 - 4*a*b*c*d ≥ 4*(a - b)^2 * Real.sqrt (a*b*c*d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l623_62313


namespace NUMINAMATH_CALUDE_smallest_number_proof_l623_62320

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 24 →
  b = 25 →
  max a (max b c) = b + 6 →
  min a (min b c) = 16 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l623_62320


namespace NUMINAMATH_CALUDE_triangle_special_angle_l623_62309

theorem triangle_special_angle (D E F : ℝ) : 
  D + E + F = 180 →  -- sum of angles in a triangle is 180 degrees
  D = E →            -- angles D and E are equal
  F = 2 * D →        -- angle F is twice angle D
  F = 90 :=          -- prove that F is 90 degrees
by sorry

end NUMINAMATH_CALUDE_triangle_special_angle_l623_62309


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l623_62364

theorem polynomial_division_theorem (x : ℝ) : 
  x^5 - 25*x^3 + 14*x^2 - 20*x + 15 = (x - 3)*(x^4 + 3*x^3 - 16*x^2 - 34*x - 122) + (-291) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l623_62364


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l623_62344

theorem right_triangle_third_side (a b c : ℝ) : 
  a = 3 ∧ b = 4 ∧ (c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2) → c = 5 ∨ c = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l623_62344


namespace NUMINAMATH_CALUDE_credit_sales_ratio_l623_62383

theorem credit_sales_ratio (total_sales cash_sales : ℚ) 
  (h1 : total_sales = 80)
  (h2 : cash_sales = 48) :
  (total_sales - cash_sales) / total_sales = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_credit_sales_ratio_l623_62383


namespace NUMINAMATH_CALUDE_sequence_existence_l623_62319

theorem sequence_existence : ∃ (a : ℕ → ℕ) (M : ℕ), 
  (∀ n, a n ≤ a (n + 1)) ∧ 
  (∀ k, ∃ n, a n > k) ∧
  (∀ n ≥ M, ¬(Nat.Prime (n + 1)) → 
    ∀ p, Nat.Prime p → p ∣ (Nat.factorial n + 1) → p > n + a n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_existence_l623_62319


namespace NUMINAMATH_CALUDE_green_eyed_students_l623_62305

theorem green_eyed_students (total : ℕ) (brown_green : ℕ) (neither : ℕ) 
  (h1 : total = 45)
  (h2 : brown_green = 9)
  (h3 : neither = 5) :
  ∃ (green : ℕ), 
    green = 10 ∧ 
    total = green + 3 * green - brown_green + neither :=
by
  sorry

end NUMINAMATH_CALUDE_green_eyed_students_l623_62305


namespace NUMINAMATH_CALUDE_rectangular_prism_prime_edges_l623_62371

theorem rectangular_prism_prime_edges (a b c : ℕ) (k : ℕ) : 
  Prime a → Prime b → Prime c →
  ∃ p n : ℕ, Prime p ∧ 2 * (a * b + b * c + c * a) = p^n →
  (a = 2^k - 1 ∧ Prime (2^k - 1) ∧ b = 2 ∧ c = 2) ∨
  (b = 2^k - 1 ∧ Prime (2^k - 1) ∧ a = 2 ∧ c = 2) ∨
  (c = 2^k - 1 ∧ Prime (2^k - 1) ∧ a = 2 ∧ b = 2) :=
sorry

end NUMINAMATH_CALUDE_rectangular_prism_prime_edges_l623_62371


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l623_62358

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the non-collinearity condition
variable (h_non_collinear : ¬ ∃ (c : ℝ), e₁ = c • e₂)

-- Define vectors a and b
def a (e₁ e₂ : V) : V := 2 • e₁ + 3 • e₂
def b (k : ℝ) (e₁ e₂ : V) : V := k • e₁ - 4 • e₂

-- State the theorem
theorem parallel_vectors_k_value 
  (h_parallel : ∃ (m : ℝ), a e₁ e₂ = m • (b k e₁ e₂)) :
  k = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l623_62358


namespace NUMINAMATH_CALUDE_stock_price_change_l623_62330

theorem stock_price_change (total_stocks : ℕ) (higher_percentage : ℚ) :
  total_stocks = 1980 →
  higher_percentage = 120 / 100 →
  ∃ (higher lower : ℕ),
    higher + lower = total_stocks ∧
    higher = (higher_percentage * lower).num ∧
    higher = 1080 :=
by sorry

end NUMINAMATH_CALUDE_stock_price_change_l623_62330


namespace NUMINAMATH_CALUDE_actual_average_height_l623_62338

/-- The number of boys in the class -/
def num_boys : ℕ := 60

/-- The initial calculated average height in cm -/
def initial_avg : ℝ := 185

/-- The recorded heights of the three boys with errors -/
def recorded_heights : Fin 3 → ℝ := ![170, 195, 160]

/-- The actual heights of the three boys -/
def actual_heights : Fin 3 → ℝ := ![140, 165, 190]

/-- The actual average height of the boys in the class -/
def actual_avg : ℝ := 184.50

theorem actual_average_height :
  let total_initial := initial_avg * num_boys
  let total_difference := (recorded_heights 0 - actual_heights 0) +
                          (recorded_heights 1 - actual_heights 1) +
                          (recorded_heights 2 - actual_heights 2)
  let corrected_total := total_initial - total_difference
  corrected_total / num_boys = actual_avg := by sorry

end NUMINAMATH_CALUDE_actual_average_height_l623_62338


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_quadratic_range_l623_62308

theorem greatest_integer_b_for_quadratic_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 5 > -25) ↔ b ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_quadratic_range_l623_62308


namespace NUMINAMATH_CALUDE_function_with_bounded_difference_is_constant_l623_62306

/-- A function f: ℝ → ℝ that satisfies |f(x) - f(y)| ≤ (x - y)² for all x, y ∈ ℝ is constant. -/
theorem function_with_bounded_difference_is_constant (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, |f x - f y| ≤ (x - y)^2) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
by sorry

end NUMINAMATH_CALUDE_function_with_bounded_difference_is_constant_l623_62306
