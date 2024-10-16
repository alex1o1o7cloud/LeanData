import Mathlib

namespace NUMINAMATH_CALUDE_other_side_length_l676_67608

/-- Represents a right triangle with given side lengths -/
structure RightTriangle where
  hypotenuse : ℝ
  side1 : ℝ
  side2 : ℝ
  hypotenuse_positive : hypotenuse > 0
  side1_positive : side1 > 0
  side2_positive : side2 > 0
  pythagorean : hypotenuse^2 = side1^2 + side2^2

/-- The length of the other side in a right triangle with hypotenuse 10 and one side 6 is 8 -/
theorem other_side_length (t : RightTriangle) (h1 : t.hypotenuse = 10) (h2 : t.side1 = 6) :
  t.side2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_other_side_length_l676_67608


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_sum_of_b_values_l676_67603

theorem quadratic_one_solution_sum (b : ℝ) : 
  (∃! x, 3 * x^2 + b * x + 6 * x + 1 = 0) ↔ 
  (b = -6 + 2 * Real.sqrt 3 ∨ b = -6 - 2 * Real.sqrt 3) :=
by sorry

theorem sum_of_b_values : 
  (-6 + 2 * Real.sqrt 3) + (-6 - 2 * Real.sqrt 3) = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_sum_of_b_values_l676_67603


namespace NUMINAMATH_CALUDE_cube_sum_identity_l676_67623

theorem cube_sum_identity (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h_sum : x + y + z = 0) :
  (x^3 + y^3 + z^3 + 3*x*y*z) / (x*y*z) = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_identity_l676_67623


namespace NUMINAMATH_CALUDE_gcd_of_repeated_five_digit_integers_l676_67657

theorem gcd_of_repeated_five_digit_integers : 
  ∃ (g : ℕ), 
    (∀ (n : ℕ), 10000 ≤ n ∧ n < 100000 → g ∣ (n * 10000100001)) ∧
    (∀ (d : ℕ), (∀ (n : ℕ), 10000 ≤ n ∧ n < 100000 → d ∣ (n * 10000100001)) → d ∣ g) ∧
    g = 10000100001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_repeated_five_digit_integers_l676_67657


namespace NUMINAMATH_CALUDE_jars_to_fill_l676_67653

def stars_per_jar : ℕ := 85
def initial_stars : ℕ := 33
def additional_stars : ℕ := 307

theorem jars_to_fill :
  (initial_stars + additional_stars) / stars_per_jar = 4 :=
by sorry

end NUMINAMATH_CALUDE_jars_to_fill_l676_67653


namespace NUMINAMATH_CALUDE_gcd_special_powers_l676_67667

theorem gcd_special_powers : Nat.gcd (2^1001 - 1) (2^1012 - 1) = 2^11 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_special_powers_l676_67667


namespace NUMINAMATH_CALUDE_exists_valid_path_2020_l676_67607

/-- Represents a square grid with diagonals drawn in each cell. -/
structure DiagonalGrid (n : ℕ) where
  size : n > 0

/-- Represents a path on the diagonal grid. -/
structure DiagonalPath (n : ℕ) where
  grid : DiagonalGrid n
  is_closed : Bool
  visits_all_cells : Bool
  no_repeated_diagonals : Bool

/-- Theorem stating the existence of a valid path in a 2020x2020 grid. -/
theorem exists_valid_path_2020 :
  ∃ (path : DiagonalPath 2020),
    path.is_closed ∧
    path.visits_all_cells ∧
    path.no_repeated_diagonals :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_path_2020_l676_67607


namespace NUMINAMATH_CALUDE_sqrt_77_between_consecutive_integers_product_l676_67638

theorem sqrt_77_between_consecutive_integers_product (a b : ℕ) : 
  a + 1 = b → 
  (a : ℝ) < Real.sqrt 77 → 
  Real.sqrt 77 < (b : ℝ) → 
  a * b = 72 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_77_between_consecutive_integers_product_l676_67638


namespace NUMINAMATH_CALUDE_stratified_sample_size_l676_67673

-- Define the total number of male and female athletes
def total_male : ℕ := 42
def total_female : ℕ := 30

-- Define the number of female athletes in the sample
def sampled_female : ℕ := 5

-- Theorem statement
theorem stratified_sample_size :
  ∃ (n : ℕ), 
    -- The sample size is the sum of sampled males and females
    n = (total_male * sampled_female / total_female) + sampled_female ∧
    -- The sample maintains the same ratio as the population
    n * total_female = (total_male + total_female) * sampled_female ∧
    -- The sample size is 12
    n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l676_67673


namespace NUMINAMATH_CALUDE_quadratic_solution_fractional_no_solution_l676_67688

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 + 4*x - 4 = 0

-- Define the fractional equation
def fractional_equation (x : ℝ) : Prop := x / (x - 2) + 3 = (x - 4) / (2 - x)

-- Theorem for the quadratic equation
theorem quadratic_solution :
  ∃ x₁ x₂ : ℝ, 
    (x₁ = 2 * Real.sqrt 2 - 2 ∧ quadratic_equation x₁) ∧
    (x₂ = -2 * Real.sqrt 2 - 2 ∧ quadratic_equation x₂) ∧
    (∀ x : ℝ, quadratic_equation x → x = x₁ ∨ x = x₂) :=
sorry

-- Theorem for the fractional equation
theorem fractional_no_solution :
  ¬∃ x : ℝ, fractional_equation x :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_fractional_no_solution_l676_67688


namespace NUMINAMATH_CALUDE_book_reading_fraction_l676_67686

theorem book_reading_fraction (total_pages remaining_pages : ℕ) 
  (h1 : total_pages = 468)
  (h2 : remaining_pages = 96)
  (h3 : (7 : ℚ) / 13 * total_pages + remaining_pages < total_pages) :
  let pages_read_first_week := (7 : ℚ) / 13 * total_pages
  let pages_remaining_after_first_week := total_pages - pages_read_first_week
  let pages_read_second_week := pages_remaining_after_first_week - remaining_pages
  pages_read_second_week / pages_remaining_after_first_week = 5 / 9 := by
sorry

end NUMINAMATH_CALUDE_book_reading_fraction_l676_67686


namespace NUMINAMATH_CALUDE_combinatorial_equations_solutions_l676_67628

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Falling factorial -/
def falling_factorial (n k : ℕ) : ℕ := sorry

theorem combinatorial_equations_solutions :
  (∃ x : ℕ, (binomial 9 x = binomial 9 (2*x - 3)) ∧ (x = 3 ∨ x = 4)) ∧
  (∃ x : ℕ, x ≤ 8 ∧ falling_factorial 8 x = 6 * falling_factorial 8 (x - 2) ∧ x = 7) :=
sorry

end NUMINAMATH_CALUDE_combinatorial_equations_solutions_l676_67628


namespace NUMINAMATH_CALUDE_factorial_1200_trailing_zeroes_l676_67609

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- Theorem: 1200! has 298 trailing zeroes -/
theorem factorial_1200_trailing_zeroes :
  trailingZeroes 1200 = 298 := by
  sorry

end NUMINAMATH_CALUDE_factorial_1200_trailing_zeroes_l676_67609


namespace NUMINAMATH_CALUDE_third_smallest_four_digit_pascal_l676_67637

/-- Pascal's triangle coefficients -/
def pascal_coeff (n k : ℕ) : ℕ := sorry

/-- Predicate for a number being in Pascal's triangle -/
def in_pascal_triangle (x : ℕ) : Prop :=
  ∃ n k : ℕ, pascal_coeff n k = x

/-- Predicate for a number being a four-digit number -/
def is_four_digit (x : ℕ) : Prop :=
  1000 ≤ x ∧ x ≤ 9999

/-- The third smallest four-digit number in Pascal's triangle is 1002 -/
theorem third_smallest_four_digit_pascal : 
  ∃ (a b : ℕ), 
    (∀ x : ℕ, (is_four_digit x ∧ in_pascal_triangle x ∧ x < 1002) → (x = 1000 ∨ x = 1001)) ∧
    (is_four_digit 1002 ∧ in_pascal_triangle 1002) ∧
    (a = 1000 ∧ b = 1001 ∧ in_pascal_triangle a ∧ in_pascal_triangle b) :=
by sorry

end NUMINAMATH_CALUDE_third_smallest_four_digit_pascal_l676_67637


namespace NUMINAMATH_CALUDE_injective_function_equation_l676_67646

theorem injective_function_equation (f : ℝ → ℝ) (h_inj : Function.Injective f) :
  (∀ x y : ℝ, x ≠ y → f ((x + y) / (x - y)) = (f x + f y) / (f x - f y)) →
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_injective_function_equation_l676_67646


namespace NUMINAMATH_CALUDE_probability_penny_dime_same_different_dollar_l676_67689

-- Define the coin types
inductive Coin
| Penny
| Nickel
| Dime
| Quarter
| Dollar

-- Define the possible outcomes for a coin flip
inductive FlipResult
| Heads
| Tails

-- Define a function to represent the result of flipping all coins
def CoinFlips := Coin → FlipResult

-- Define the condition for a successful outcome
def SuccessfulOutcome (flips : CoinFlips) : Prop :=
  (flips Coin.Penny = flips Coin.Dime) ∧ (flips Coin.Penny ≠ flips Coin.Dollar)

-- Define the total number of possible outcomes
def TotalOutcomes : ℕ := 2^5

-- Define the number of successful outcomes
def SuccessfulOutcomes : ℕ := 8

-- Theorem statement
theorem probability_penny_dime_same_different_dollar :
  (SuccessfulOutcomes : ℚ) / TotalOutcomes = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_penny_dime_same_different_dollar_l676_67689


namespace NUMINAMATH_CALUDE_binomial_sum_one_l676_67640

theorem binomial_sum_one (a a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x, (a - x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₂ = 80 →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 1 := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_one_l676_67640


namespace NUMINAMATH_CALUDE_suitcase_electronics_weight_l676_67625

/-- Given a suitcase with books, clothes, and electronics, prove the weight of electronics -/
theorem suitcase_electronics_weight 
  (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive
  (h2 : 4 * x > 7) -- Ensure we can remove 7 pounds of clothing
  (h3 : 5 * x / (4 * x - 7) = 5 / 2) -- Ratio doubles after removing 7 pounds
  : 2 * x = 7 := by
  sorry

end NUMINAMATH_CALUDE_suitcase_electronics_weight_l676_67625


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l676_67616

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, (a + 1) * x - y + 2 = 0 ↔ x + (a - 1) * y - 1 = 0) → 
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l676_67616


namespace NUMINAMATH_CALUDE_min_value_of_inverse_squares_l676_67648

theorem min_value_of_inverse_squares (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (∃ (x y : ℝ), x^2 + y^2 + 4*a*x + 4*a^2 - 4 = 0 ∧ x^2 + y^2 - 2*b*y + b^2 - 1 = 0) →
  (∃! (l : ℝ → ℝ), ∀ (x y : ℝ), (x^2 + y^2 + 4*a*x + 4*a^2 - 4 = 0 ∨ x^2 + y^2 - 2*b*y + b^2 - 1 = 0) → 
    y = l x ∧ (∀ (x' y' : ℝ), y' = l x' → (x'^2 + y'^2 + 4*a*x' + 4*a^2 - 4 > 0 ∧ x'^2 + y'^2 - 2*b*y' + b^2 - 1 > 0) ∨
    (x'^2 + y'^2 + 4*a*x' + 4*a^2 - 4 < 0 ∧ x'^2 + y'^2 - 2*b*y' + b^2 - 1 < 0))) →
  (1 / a^2 + 1 / b^2 ≥ 9) ∧ (∃ (a' b' : ℝ), a' ≠ 0 ∧ b' ≠ 0 ∧ 1 / a'^2 + 1 / b'^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_inverse_squares_l676_67648


namespace NUMINAMATH_CALUDE_floor_ceiling_solution_l676_67661

theorem floor_ceiling_solution (c : ℝ) : 
  (∃ (n : ℤ), n = ⌊c⌋ ∧ 3 * (n : ℝ)^2 + 8 * (n : ℝ) - 35 = 0) ∧
  (let frac := c - ⌊c⌋; 4 * frac^2 - 12 * frac + 5 = 0 ∧ 0 ≤ frac ∧ frac < 1) →
  c = -9/2 := by
sorry

end NUMINAMATH_CALUDE_floor_ceiling_solution_l676_67661


namespace NUMINAMATH_CALUDE_max_value_of_expression_l676_67642

theorem max_value_of_expression (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hsum : a + b + c + d ≤ 4) :
  (2 * a^2 + a^2 * b)^(1/4) + (2 * b^2 + b^2 * c)^(1/4) + 
  (2 * c^2 + c^2 * d)^(1/4) + (2 * d^2 + d^2 * a)^(1/4) ≤ 4 * (3^(1/4)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l676_67642


namespace NUMINAMATH_CALUDE_symmetry_axes_count_cube_symmetry_axes_count_tetrahedron_symmetry_axes_count_l676_67658

/-- The number of axes of symmetry in a cube -/
def cube_symmetry_axes : ℕ := 13

/-- The number of axes of symmetry in a regular tetrahedron -/
def tetrahedron_symmetry_axes : ℕ := 7

/-- Theorem stating the number of axes of symmetry for a cube and a regular tetrahedron -/
theorem symmetry_axes_count :
  (cube_symmetry_axes = 13) ∧ (tetrahedron_symmetry_axes = 7) := by
  sorry

/-- Theorem for the number of axes of symmetry in a cube -/
theorem cube_symmetry_axes_count : cube_symmetry_axes = 13 := by
  sorry

/-- Theorem for the number of axes of symmetry in a regular tetrahedron -/
theorem tetrahedron_symmetry_axes_count : tetrahedron_symmetry_axes = 7 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_axes_count_cube_symmetry_axes_count_tetrahedron_symmetry_axes_count_l676_67658


namespace NUMINAMATH_CALUDE_remainder_theorem_l676_67680

theorem remainder_theorem : 
  10002000400080016003200640128025605121024204840968192 % 100020004000800160032 = 40968192 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l676_67680


namespace NUMINAMATH_CALUDE_f_properties_l676_67670

def f (x : ℝ) := x^3 - 3*x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (f 1 = -2) ∧
  (∀ x, x = -1 ∨ x = 1 → deriv f x = 0) ∧
  (∀ x, f x ≤ f (-1)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l676_67670


namespace NUMINAMATH_CALUDE_savings_account_growth_l676_67639

/-- Calculates the final amount in a savings account with compound interest. -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- The problem statement -/
theorem savings_account_growth :
  let principal : ℝ := 1500
  let rate : ℝ := 0.04
  let years : ℕ := 21
  let final_amount := compound_interest principal rate years
  ∃ ε > 0, |final_amount - 3046.28| < ε :=
by sorry

end NUMINAMATH_CALUDE_savings_account_growth_l676_67639


namespace NUMINAMATH_CALUDE_trig_identity_l676_67681

theorem trig_identity (a : ℝ) (h : Real.sin (π * Real.cos a) = Real.cos (π * Real.sin a)) :
  35 * (Real.sin (2 * a))^2 + 84 * (Real.cos (4 * a))^2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l676_67681


namespace NUMINAMATH_CALUDE_rainwater_farm_problem_l676_67654

/-- Mr. Rainwater's farm animals problem -/
theorem rainwater_farm_problem (goats cows chickens : ℕ) : 
  goats = 4 * cows →
  goats = 2 * chickens →
  chickens = 18 →
  cows = 9 := by
sorry

end NUMINAMATH_CALUDE_rainwater_farm_problem_l676_67654


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l676_67624

/-- Calculates the total profit given investments and one partner's profit share -/
def calculate_total_profit (investment_A investment_B investment_C profit_share_A : ℚ) : ℚ :=
  let total_investment := investment_A + investment_B + investment_C
  (profit_share_A * total_investment) / investment_A

theorem partnership_profit_calculation 
  (investment_A investment_B investment_C profit_share_A : ℚ) 
  (h1 : investment_A = 6300)
  (h2 : investment_B = 4200)
  (h3 : investment_C = 10500)
  (h4 : profit_share_A = 3630) :
  calculate_total_profit investment_A investment_B investment_C profit_share_A = 12100 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l676_67624


namespace NUMINAMATH_CALUDE_tangent_three_implications_l676_67649

theorem tangent_three_implications (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5/7 ∧
  1 - 4 * Real.sin α * Real.cos α + 2 * (Real.cos α)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_three_implications_l676_67649


namespace NUMINAMATH_CALUDE_reflect_c_twice_l676_67679

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Theorem: Reflecting point C(2,2) over x-axis then y-axis results in C''(-2,-2) -/
theorem reflect_c_twice :
  let c : ℝ × ℝ := (2, 2)
  reflect_y (reflect_x c) = (-2, -2) := by
sorry

end NUMINAMATH_CALUDE_reflect_c_twice_l676_67679


namespace NUMINAMATH_CALUDE_right_triangle_median_to_hypotenuse_l676_67687

theorem right_triangle_median_to_hypotenuse (DE DF EF : ℝ) :
  DE = 15 →
  DF = 9 →
  EF = 12 →
  DE^2 = DF^2 + EF^2 →
  (DE / 2 : ℝ) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_median_to_hypotenuse_l676_67687


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_l676_67683

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_l676_67683


namespace NUMINAMATH_CALUDE_min_a_value_l676_67620

theorem min_a_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 9 * x + y = x * y) :
  ∃ (a : ℝ), a > 0 ∧ (∀ (x y : ℝ), x > 0 → y > 0 → a * x + y ≥ 25) ∧
  (∀ (b : ℝ), b > 0 → (∀ (x y : ℝ), x > 0 → y > 0 → b * x + y ≥ 25) → b ≥ a) ∧
  a = 4 :=
sorry

end NUMINAMATH_CALUDE_min_a_value_l676_67620


namespace NUMINAMATH_CALUDE_download_speed_scientific_notation_l676_67610

/-- The download speed of a 5G network in KB per second -/
def download_speed : ℝ := 1300000

/-- Scientific notation representation of the download speed -/
def scientific_notation : ℝ := 1.3 * (10 ^ 6)

theorem download_speed_scientific_notation : 
  download_speed = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_download_speed_scientific_notation_l676_67610


namespace NUMINAMATH_CALUDE_pentagon_perimeter_calculation_l676_67663

/-- The perimeter of a pentagon with given side lengths -/
def pentagon_perimeter (FG GH HI IJ JF : ℝ) : ℝ := FG + GH + HI + IJ + JF

/-- Theorem: The perimeter of pentagon FGHIJ is 7 + 2√5 -/
theorem pentagon_perimeter_calculation :
  pentagon_perimeter 2 2 (Real.sqrt 5) (Real.sqrt 5) 3 = 7 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_calculation_l676_67663


namespace NUMINAMATH_CALUDE_tangent_product_special_angles_l676_67678

theorem tangent_product_special_angles :
  let A : Real := 30 * π / 180
  let B : Real := 40 * π / 180
  let C : Real := 5 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) * (1 + Real.tan C) = 3 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_special_angles_l676_67678


namespace NUMINAMATH_CALUDE_circle_center_on_line_l676_67617

theorem circle_center_on_line (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*a*x + 4*y - 6 = 0 → 
    ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 2*a*h + 4*k - 6) ∧ 
    h + 2*k + 1 = 0) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_circle_center_on_line_l676_67617


namespace NUMINAMATH_CALUDE_intersection_and_complement_l676_67684

def A : Set ℝ := {x : ℝ | -4 < x^2 - 5*x + 2 ∧ x^2 - 5*x + 2 < 26}
def B : Set ℝ := {x : ℝ | -x^2 + 4*x - 3 < 0}

theorem intersection_and_complement :
  (A ∩ B = {x : ℝ | (-3 < x ∧ x < 1) ∨ (3 < x ∧ x < 8)}) ∧
  (Set.compl (A ∩ B) = {x : ℝ | x ≤ -3 ∨ (1 ≤ x ∧ x ≤ 3) ∨ x ≥ 8}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_complement_l676_67684


namespace NUMINAMATH_CALUDE_gym_cost_is_twelve_l676_67651

/-- Calculates the monthly cost of a gym membership given the total cost for 3 years and the down payment. -/
def monthly_gym_cost (total_cost : ℚ) (down_payment : ℚ) : ℚ :=
  (total_cost - down_payment) / (3 * 12)

/-- Theorem stating that the monthly cost of the gym is $12 under given conditions. -/
theorem gym_cost_is_twelve :
  let total_cost : ℚ := 482
  let down_payment : ℚ := 50
  monthly_gym_cost total_cost down_payment = 12 := by
  sorry

end NUMINAMATH_CALUDE_gym_cost_is_twelve_l676_67651


namespace NUMINAMATH_CALUDE_expression_evaluation_l676_67636

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := 3
  let z : ℝ := 1
  x^2 + y^2 - z^2 + 2*x*y + 3*y*z = 33 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l676_67636


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l676_67619

/-- Given a right triangle with legs x and y, if rotating about one leg produces a cone of volume 1000π cm³
    and rotating about the other leg produces a cone of volume 2250π cm³, 
    then the hypotenuse is approximately 39.08 cm. -/
theorem right_triangle_hypotenuse (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) :
  (1/3 * π * y^2 * x = 1000 * π) →
  (1/3 * π * x^2 * y = 2250 * π) →
  abs (Real.sqrt (x^2 + y^2) - 39.08) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l676_67619


namespace NUMINAMATH_CALUDE_line_plane_intersection_l676_67697

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define the intersection relation for lines and planes
variable (intersect : Line → Line → Prop)
variable (intersect_planes : Plane → Plane → Prop)

-- Theorem statement
theorem line_plane_intersection 
  (a b : Line) (α β : Plane) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β) 
  (h3 : subset a α) 
  (h4 : subset b β) 
  (h5 : intersect a b) : 
  intersect_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_intersection_l676_67697


namespace NUMINAMATH_CALUDE_solution_f_greater_than_two_minimum_value_f_l676_67675

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for the solution of f(x) > 2
theorem solution_f_greater_than_two :
  ∀ x : ℝ, f x > 2 ↔ x < -7 ∨ x > 5/3 := by sorry

-- Theorem for the minimum value of f
theorem minimum_value_f :
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) ∧ m = -9/2 := by sorry

end NUMINAMATH_CALUDE_solution_f_greater_than_two_minimum_value_f_l676_67675


namespace NUMINAMATH_CALUDE_logo_shaded_area_l676_67691

/-- The shaded area of a logo design with a square containing four larger circles and one smaller circle -/
theorem logo_shaded_area (square_side : ℝ) (large_circle_radius : ℝ) (small_circle_radius : ℝ) : 
  square_side = 24 →
  large_circle_radius = 6 →
  small_circle_radius = 3 →
  (square_side ^ 2) - (4 * Real.pi * large_circle_radius ^ 2) - (Real.pi * small_circle_radius ^ 2) = 576 - 153 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_logo_shaded_area_l676_67691


namespace NUMINAMATH_CALUDE_final_number_can_be_zero_l676_67635

/-- Represents the operation of replacing two numbers with their absolute difference -/
def difference_operation (S : Finset ℕ) : Finset ℕ :=
  sorry

/-- The initial set of integers from 1 to 2013 -/
def initial_set : Finset ℕ :=
  Finset.range 2013

/-- Applies the difference operation n times to the given set -/
def apply_n_times (S : Finset ℕ) (n : ℕ) : Finset ℕ :=
  sorry

theorem final_number_can_be_zero :
  ∃ (result : Finset ℕ), apply_n_times initial_set 2012 = result ∧ 0 ∈ result :=
sorry

end NUMINAMATH_CALUDE_final_number_can_be_zero_l676_67635


namespace NUMINAMATH_CALUDE_distance_traveled_l676_67643

/-- Calculates the total distance traveled given two speeds and two durations -/
def total_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Theorem: The total distance traveled is 255 miles -/
theorem distance_traveled : total_distance 45 2 55 3 = 255 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l676_67643


namespace NUMINAMATH_CALUDE_corresponding_angles_equal_if_then_form_l676_67604

/-- Two angles are corresponding if they occupy the same relative position when a line intersects two other lines. -/
def are_corresponding (α β : Angle) : Prop := sorry

/-- Rewrite the statement "corresponding angles are equal" in if-then form -/
theorem corresponding_angles_equal_if_then_form :
  (∀ α β : Angle, are_corresponding α β → α = β) ↔
  (∀ α β : Angle, are_corresponding α β → α = β) :=
by sorry

end NUMINAMATH_CALUDE_corresponding_angles_equal_if_then_form_l676_67604


namespace NUMINAMATH_CALUDE_triangle_sinC_l676_67669

theorem triangle_sinC (A B C : ℝ) (hABC : A + B + C = π) 
  (hAC : 3 = 2 * Real.sqrt 3 * (Real.sin A / Real.sin B))
  (hA : A = 2 * B) : 
  Real.sin C = Real.sqrt 6 / 9 := by
sorry

end NUMINAMATH_CALUDE_triangle_sinC_l676_67669


namespace NUMINAMATH_CALUDE_sum_distances_focus_to_points_l676_67630

/-- The parabola defined by y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (2, 0)

/-- Theorem: Sum of distances from focus to three points on parabola -/
theorem sum_distances_focus_to_points
  (A B C : ℝ × ℝ)
  (hA : A ∈ Parabola)
  (hB : B ∈ Parabola)
  (hC : C ∈ Parabola)
  (h_sum : F.1 * 3 = A.1 + B.1 + C.1) :
  dist F A + dist F B + dist F C = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_distances_focus_to_points_l676_67630


namespace NUMINAMATH_CALUDE_min_value_of_sum_l676_67633

theorem min_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 6) : 
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 6 → 
    9/a + 4/b + 1/c ≤ 9/x + 4/y + 1/z) ∧ 
  (9/a + 4/b + 1/c = 6) := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l676_67633


namespace NUMINAMATH_CALUDE_polynomial_composite_l676_67613

def P (x : ℕ) : ℕ := 4*x^3 + 6*x^2 + 4*x + 1

theorem polynomial_composite : ∀ x : ℕ, ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ P x = a * b :=
sorry

end NUMINAMATH_CALUDE_polynomial_composite_l676_67613


namespace NUMINAMATH_CALUDE_greatest_abcba_div_11_is_greatest_greatest_abcba_div_11_is_divisible_by_11_l676_67634

/-- Represents a five-digit number in the form AB,CBA -/
structure ABCBA where
  a : Nat
  b : Nat
  c : Nat
  value : Nat
  h1 : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9
  h2 : a ≠ b ∧ a ≠ c ∧ b ≠ c
  h3 : value = a * 10000 + b * 1000 + c * 100 + b * 10 + a

/-- The greatest ABCBA number divisible by 11 -/
def greatest_abcba_div_11 : ABCBA :=
  { a := 9
  , b := 6
  , c := 5
  , value := 96569
  , h1 := by simp
  , h2 := by simp
  , h3 := by simp
  }

theorem greatest_abcba_div_11_is_greatest :
  ∀ n : ABCBA, n.value % 11 = 0 → n.value ≤ greatest_abcba_div_11.value :=
sorry

theorem greatest_abcba_div_11_is_divisible_by_11 :
  greatest_abcba_div_11.value % 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_greatest_abcba_div_11_is_greatest_greatest_abcba_div_11_is_divisible_by_11_l676_67634


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l676_67641

theorem quadratic_equation_solution :
  let a : ℝ := 1
  let b : ℝ := 5
  let c : ℝ := -4
  let x₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + 5*x₁ - 4 = 0 ∧ x₂^2 + 5*x₂ - 4 = 0 ∧ x₁ ≠ x₂ :=
by sorry

#check quadratic_equation_solution

end NUMINAMATH_CALUDE_quadratic_equation_solution_l676_67641


namespace NUMINAMATH_CALUDE_arctan_less_arcsin_iff_l676_67606

theorem arctan_less_arcsin_iff (x : ℝ) : Real.arctan x < Real.arcsin x ↔ -1 < x ∧ x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_arctan_less_arcsin_iff_l676_67606


namespace NUMINAMATH_CALUDE_math_problem_l676_67674

theorem math_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ (h : a * b - a - 2 * b = 0), a + 2 * b ≥ 8) ∧
  (a^2 / b + b^2 / a ≥ a + b) ∧
  (∀ (h : 1 / (a + 1) + 1 / (b + 2) = 1 / 3), a * b + a + b ≥ 14 + 6 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_math_problem_l676_67674


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l676_67693

/-- The coefficients of the quadratic equation in general form -/
def quadratic_coefficients (a b c : ℝ) : ℝ × ℝ := (a, b)

/-- The original quadratic equation -/
def original_equation (x : ℝ) : Prop := 3 * x^2 + 1 = 6 * x

/-- The general form of the quadratic equation -/
def general_form (x : ℝ) : Prop := 3 * x^2 - 6 * x + 1 = 0

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), (∀ x, original_equation x ↔ general_form x) ∧
  quadratic_coefficients a b c = (3, -6) := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l676_67693


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l676_67605

theorem arithmetic_sequence_count : ∀ (a₁ d aₙ : ℝ) (n : ℕ),
  a₁ = 1.5 ∧ d = 4 ∧ aₙ = 45.5 ∧ aₙ = a₁ + (n - 1) * d →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l676_67605


namespace NUMINAMATH_CALUDE_gracie_is_56_inches_tall_l676_67659

/-- Gracie's height in inches -/
def gracies_height : ℕ := 56

/-- Theorem stating Gracie's height is 56 inches -/
theorem gracie_is_56_inches_tall : gracies_height = 56 := by
  sorry

end NUMINAMATH_CALUDE_gracie_is_56_inches_tall_l676_67659


namespace NUMINAMATH_CALUDE_min_containers_to_fill_jumbo_l676_67682

/-- The volume of a regular size container in milliliters -/
def regular_container_volume : ℕ := 75

/-- The volume of a jumbo container in milliliters -/
def jumbo_container_volume : ℕ := 1800

/-- The minimum number of regular size containers needed to fill a jumbo container -/
def min_containers : ℕ := (jumbo_container_volume + regular_container_volume - 1) / regular_container_volume

theorem min_containers_to_fill_jumbo : min_containers = 24 := by
  sorry

end NUMINAMATH_CALUDE_min_containers_to_fill_jumbo_l676_67682


namespace NUMINAMATH_CALUDE_volume_of_intersected_prism_l676_67601

/-- The volume of a solid formed by the intersection of a plane with a prism -/
theorem volume_of_intersected_prism (a : ℝ) (h : ℝ) :
  let prism_base_area : ℝ := (a^2 * Real.sqrt 3) / 2
  let prism_volume : ℝ := prism_base_area * h
  let intersection_volume : ℝ := (77 * Real.sqrt 3) / 54
  (h = 2) →
  (intersection_volume < prism_volume) →
  (intersection_volume > 0) →
  intersection_volume = (77 * Real.sqrt 3) / 54 :=
by sorry

end NUMINAMATH_CALUDE_volume_of_intersected_prism_l676_67601


namespace NUMINAMATH_CALUDE_system_solution_l676_67699

theorem system_solution (x y : ℝ) :
  (4 * (Real.cos x)^2 - 4 * Real.cos x * (Real.cos (6 * x))^2 + (Real.cos (6 * x))^2 = 0) ∧
  (Real.sin x = Real.cos y) ↔
  (∃ (k n : ℤ),
    ((x = π / 3 + 2 * π * ↑k ∧ (y = π / 6 + 2 * π * ↑n ∨ y = -π / 6 + 2 * π * ↑n)) ∨
     (x = -π / 3 + 2 * π * ↑k ∧ (y = 5 * π / 6 + 2 * π * ↑n ∨ y = -5 * π / 6 + 2 * π * ↑n)))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l676_67699


namespace NUMINAMATH_CALUDE_exists_number_with_large_square_digit_sum_l676_67685

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number whose square's digit sum exceeds 1000 times its own digit sum -/
theorem exists_number_with_large_square_digit_sum :
  ∃ n : ℕ, sumOfDigits (n^2) > 1000 * sumOfDigits n := by
  sorry

end NUMINAMATH_CALUDE_exists_number_with_large_square_digit_sum_l676_67685


namespace NUMINAMATH_CALUDE_sequence_theorem_l676_67698

def sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) (r p : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → r * (n - p) * S (n + 1) = n^2 * a n + (n^2 - n - 2) * a 1

theorem sequence_theorem (a : ℕ → ℝ) (S : ℕ → ℝ) (r p : ℝ) 
  (h1 : |a 1| ≠ |a 2|)
  (h2 : r ≠ 0)
  (h3 : sequence_property a S r p) :
  (p = 1) ∧ 
  (¬ ∃ k : ℝ, k ≠ 1 ∧ k ≠ -1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = k * a n) ∧
  (r = 2 → ∃ d : ℝ, ∀ n : ℕ, n > 0 → a (n + 1) = a n + d) :=
by sorry

end NUMINAMATH_CALUDE_sequence_theorem_l676_67698


namespace NUMINAMATH_CALUDE_proportion_check_l676_67660

/-- A set of four line segments forms a proportion if the product of the means equals the product of the extremes. -/
def is_proportion (a b c d : ℝ) : Prop := b * c = a * d

/-- The given sets of line segments -/
def set_A : Fin 4 → ℝ := ![2, 3, 5, 6]
def set_B : Fin 4 → ℝ := ![1, 2, 3, 5]
def set_C : Fin 4 → ℝ := ![1, 3, 3, 7]
def set_D : Fin 4 → ℝ := ![3, 2, 4, 6]

theorem proportion_check :
  ¬ is_proportion (set_A 0) (set_A 1) (set_A 2) (set_A 3) ∧
  ¬ is_proportion (set_B 0) (set_B 1) (set_B 2) (set_B 3) ∧
  ¬ is_proportion (set_C 0) (set_C 1) (set_C 2) (set_C 3) ∧
  is_proportion (set_D 0) (set_D 1) (set_D 2) (set_D 3) := by
  sorry

end NUMINAMATH_CALUDE_proportion_check_l676_67660


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l676_67631

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x > 2}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l676_67631


namespace NUMINAMATH_CALUDE_vector_addition_l676_67627

/-- Given two vectors OA and AB in R², prove that OB = OA + AB -/
theorem vector_addition (OA AB : ℝ × ℝ) (h1 : OA = (-2, 3)) (h2 : AB = (-1, -4)) :
  OA + AB = (-3, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l676_67627


namespace NUMINAMATH_CALUDE_jakes_weight_l676_67671

/-- Proves Jake's current weight given the conditions of the problem -/
theorem jakes_weight (jake sister brother : ℝ) 
  (h1 : jake - 20 = 2 * sister)
  (h2 : brother = 0.5 * jake)
  (h3 : jake + sister + brother = 330) :
  jake = 170 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l676_67671


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l676_67676

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (lost_by : ℕ)
  (h_total : total_votes = 20000)
  (h_lost : lost_by = 16000) :
  (total_votes - lost_by) / total_votes * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l676_67676


namespace NUMINAMATH_CALUDE_least_y_solution_l676_67644

-- Define the function we're trying to solve
def f (y : ℝ) := y + y^2

-- State the theorem
theorem least_y_solution :
  ∃ y : ℝ, y > 2 ∧ f y = 360 ∧ ∃ ε > 0, |y - 18.79| < ε :=
sorry

end NUMINAMATH_CALUDE_least_y_solution_l676_67644


namespace NUMINAMATH_CALUDE_find_number_l676_67629

theorem find_number : ∃ x : ℝ, x - (3/5) * x = 56 ∧ x = 140 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l676_67629


namespace NUMINAMATH_CALUDE_horner_method_polynomial_evaluation_l676_67668

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

def polynomial (x : ℝ) : ℝ :=
  3 * x^4 - x^2 + 2 * x + 1

theorem horner_method_polynomial_evaluation :
  let coeffs := [3, 0, -1, 2, 1]
  let x := 2
  let v₃ := (horner_method (coeffs.take 4) x)
  v₃ = 22 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_polynomial_evaluation_l676_67668


namespace NUMINAMATH_CALUDE_hamburger_combinations_l676_67690

/-- Represents the number of available condiments -/
def num_condiments : Nat := 8

/-- Represents the number of choices for meat patties -/
def meat_patty_choices : Nat := 3

/-- Calculates the total number of hamburger combinations -/
def total_combinations : Nat := 2^num_condiments * meat_patty_choices

/-- Theorem: The total number of different hamburger combinations is 768 -/
theorem hamburger_combinations : total_combinations = 768 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l676_67690


namespace NUMINAMATH_CALUDE_longer_base_length_l676_67655

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithCircle where
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Length of the shorter base -/
  short_base : ℝ
  /-- Length of the longer base -/
  long_base : ℝ
  /-- The circle is inscribed in the trapezoid -/
  inscribed : r > 0
  /-- The trapezoid is a right trapezoid -/
  right_angled : True
  /-- The shorter base is positive -/
  short_base_positive : short_base > 0
  /-- The longer base is longer than the shorter base -/
  base_inequality : long_base > short_base

/-- Theorem: The longer base of the trapezoid is 12 units -/
theorem longer_base_length (t : RightTrapezoidWithCircle) 
  (h1 : t.r = 3) 
  (h2 : t.short_base = 4) : 
  t.long_base = 12 := by
  sorry

end NUMINAMATH_CALUDE_longer_base_length_l676_67655


namespace NUMINAMATH_CALUDE_composite_3p_squared_plus_15_l676_67632

theorem composite_3p_squared_plus_15 (p : ℕ) (h : Nat.Prime p) :
  ¬ Nat.Prime (3 * p^2 + 15) := by
  sorry

end NUMINAMATH_CALUDE_composite_3p_squared_plus_15_l676_67632


namespace NUMINAMATH_CALUDE_g_502_solutions_l676_67615

-- Define g₁
def g₁ (x : ℚ) : ℚ := 1/2 - 4/(4*x + 2)

-- Define gₙ recursively
def g (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | 1 => g₁ x
  | n+1 => g₁ (g n x)

-- Theorem statement
theorem g_502_solutions (x : ℚ) : 
  g 502 x = x - 2 ↔ x = 115/64 ∨ x = 51/64 := by sorry

end NUMINAMATH_CALUDE_g_502_solutions_l676_67615


namespace NUMINAMATH_CALUDE_quadratic_function_problem_l676_67650

theorem quadratic_function_problem (a b : ℝ) : 
  (1^2 + a*1 + b = 2) → 
  ((-2)^2 + a*(-2) + b = -1) → 
  ((-3)^2 + a*(-3) + b = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_problem_l676_67650


namespace NUMINAMATH_CALUDE_money_distribution_l676_67695

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (AC_sum : A + C = 200)
  (C_amount : C = 10) :
  B + C = 310 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l676_67695


namespace NUMINAMATH_CALUDE_total_money_l676_67696

theorem total_money (A B C : ℕ) 
  (h1 : A + C = 200)
  (h2 : B + C = 350)
  (h3 : C = 50) :
  A + B + C = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l676_67696


namespace NUMINAMATH_CALUDE_base6_two_distinct_primes_l676_67677

/-- Represents a number in base 6 formed by appending fives to 1200 -/
def base6Number (n : ℕ) : ℕ :=
  288 * 6^(10*n + 2) + (6^(10*n + 2) - 1)

/-- Counts the number of distinct prime factors of a natural number -/
noncomputable def countDistinctPrimeFactors (x : ℕ) : ℕ := sorry

/-- Theorem stating that the base 6 number has exactly two distinct prime factors iff n = 0 -/
theorem base6_two_distinct_primes (n : ℕ) : 
  countDistinctPrimeFactors (base6Number n) = 2 ↔ n = 0 := by sorry

end NUMINAMATH_CALUDE_base6_two_distinct_primes_l676_67677


namespace NUMINAMATH_CALUDE_student_arrangement_l676_67662

theorem student_arrangement (n : ℕ) (h : n = 5) :
  let total_arrangements := n.factorial
  let a_left_arrangements := 2 * (n - 1).factorial
  let a_left_b_right_arrangements := (n - 2).factorial
  let valid_arrangements := total_arrangements - a_left_arrangements + a_left_b_right_arrangements
  valid_arrangements = 78 :=
sorry

end NUMINAMATH_CALUDE_student_arrangement_l676_67662


namespace NUMINAMATH_CALUDE_difference_of_reciprocals_l676_67665

theorem difference_of_reciprocals (p q : ℚ) 
  (hp : 3 / p = 6) 
  (hq : 3 / q = 15) : 
  p - q = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_difference_of_reciprocals_l676_67665


namespace NUMINAMATH_CALUDE_biography_increase_l676_67600

theorem biography_increase (B : ℝ) (N : ℝ) (h1 : B > 0) (h2 : N > 0) : 
  (0.20 * B + N = 0.32 * (B + N)) → 
  ((N / (0.20 * B)) = 15 / 17) := by
sorry

end NUMINAMATH_CALUDE_biography_increase_l676_67600


namespace NUMINAMATH_CALUDE_inscribed_circles_area_l676_67602

theorem inscribed_circles_area (R : ℝ) (d : ℝ) : 
  R = 10 ∧ d = 6 → 
  let h := R - d / 2
  let r := R - d / 2
  2 * Real.pi * r^2 = 98 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circles_area_l676_67602


namespace NUMINAMATH_CALUDE_largest_three_digit_base7_decimal_l676_67622

/-- The largest three-digit number in base 7 -/
def largest_base7 : ℕ := 666

/-- Conversion from base 7 to decimal -/
def base7_to_decimal (n : ℕ) : ℕ :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

/-- Theorem stating that the largest decimal number represented by a three-digit base-7 number is 342 -/
theorem largest_three_digit_base7_decimal :
  base7_to_decimal largest_base7 = 342 := by sorry

end NUMINAMATH_CALUDE_largest_three_digit_base7_decimal_l676_67622


namespace NUMINAMATH_CALUDE_inscribed_circle_theorem_l676_67666

def equilateral_triangle_with_inscribed_circle 
  (radius : ℝ) (height : ℝ) (x : ℝ) : Prop :=
  radius = 3/16 ∧ 
  height = 3 * radius ∧ 
  x = height - 1/2

theorem inscribed_circle_theorem :
  ∀ (radius height x : ℝ),
    equilateral_triangle_with_inscribed_circle radius height x →
    x = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_theorem_l676_67666


namespace NUMINAMATH_CALUDE_boat_distance_along_stream_l676_67652

def boat_problem (boat_speed : ℝ) (against_stream_distance : ℝ) : ℝ :=
  let stream_speed := boat_speed - against_stream_distance
  boat_speed + stream_speed

theorem boat_distance_along_stream 
  (boat_speed : ℝ) 
  (against_stream_distance : ℝ) 
  (h1 : boat_speed = 15) 
  (h2 : against_stream_distance = 9) : 
  boat_problem boat_speed against_stream_distance = 21 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_along_stream_l676_67652


namespace NUMINAMATH_CALUDE_prob_diana_wins_is_half_l676_67618

/-- Diana's die has 8 sides -/
def diana_sides : ℕ := 8

/-- Apollo's die has 6 sides -/
def apollo_sides : ℕ := 6

/-- The set of possible outcomes for Diana -/
def diana_outcomes : Finset ℕ := Finset.range diana_sides

/-- The set of possible outcomes for Apollo -/
def apollo_outcomes : Finset ℕ := Finset.range apollo_sides

/-- The set of even outcomes for Apollo -/
def apollo_even_outcomes : Finset ℕ := Finset.filter (fun n => n % 2 = 0) apollo_outcomes

/-- The probability that Diana rolls a number larger than Apollo, given that Apollo's number is even -/
def prob_diana_wins_given_apollo_even : ℚ :=
  let total_outcomes := (apollo_even_outcomes.card * diana_outcomes.card : ℚ)
  let favorable_outcomes := (apollo_even_outcomes.sum fun a =>
    (diana_outcomes.filter (fun d => d > a)).card : ℚ)
  favorable_outcomes / total_outcomes

/-- The main theorem: The probability is 1/2 -/
theorem prob_diana_wins_is_half : prob_diana_wins_given_apollo_even = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_prob_diana_wins_is_half_l676_67618


namespace NUMINAMATH_CALUDE_square_roots_problem_l676_67694

theorem square_roots_problem (m : ℝ) (n : ℝ) (h1 : n > 0) (h2 : 2*m - 1 = (n ^ (1/2 : ℝ))) (h3 : 2 - m = (n ^ (1/2 : ℝ))) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l676_67694


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l676_67656

theorem quadratic_equation_root (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 3 * x - 119 = 0) → 
  (a * 7^2 + 3 * 7 - 119 = 0) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l676_67656


namespace NUMINAMATH_CALUDE_roses_in_vase_l676_67645

theorem roses_in_vase (initial_roses : ℕ) : initial_roses + 8 = 18 → initial_roses = 10 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l676_67645


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l676_67672

theorem sum_of_three_consecutive_cubes_divisible_by_nine (n : ℤ) :
  ∃ k : ℤ, (n - 1)^3 + n^3 + (n + 1)^3 = 9 * k :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l676_67672


namespace NUMINAMATH_CALUDE_min_value_expression_l676_67611

theorem min_value_expression (x y : ℝ) 
  (h1 : x * y + 3 * x = 3)
  (h2 : 0 < x)
  (h3 : x < 1/2) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (x' y' : ℝ), 
    x' * y' + 3 * x' = 3 → 
    0 < x' → 
    x' < 1/2 → 
    3 / x' + 1 / (y' - 3) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l676_67611


namespace NUMINAMATH_CALUDE_certain_number_problem_l676_67692

theorem certain_number_problem :
  ∃ x : ℝ, x ≥ 0 ∧ 5 * (Real.sqrt x + 3) = 19 ∧ x = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l676_67692


namespace NUMINAMATH_CALUDE_solution_set_inequality_l676_67621

theorem solution_set_inequality (x : ℝ) : (x - 1) / (x + 2) > 0 ↔ x > 1 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l676_67621


namespace NUMINAMATH_CALUDE_sample_volume_calculation_l676_67647

theorem sample_volume_calculation (m : ℝ) 
  (h1 : m > 0)  -- Ensure m is positive
  (h2 : 8 / m + 0.15 + 0.45 = 1) : m = 20 := by
  sorry

end NUMINAMATH_CALUDE_sample_volume_calculation_l676_67647


namespace NUMINAMATH_CALUDE_correct_bottles_calculation_l676_67614

/-- Given that B bottles of water can be purchased for P pennies,
    and 1 euro is worth 100 pennies, this function calculates
    the number of bottles that can be purchased for E euros. -/
def bottles_per_euro (B P E : ℚ) : ℚ :=
  (100 * E * B) / P

/-- Theorem stating that the number of bottles that can be purchased
    for E euros is (100 * E * B) / P, given the conditions. -/
theorem correct_bottles_calculation (B P E : ℚ) (hB : B > 0) (hP : P > 0) (hE : E > 0) :
  bottles_per_euro B P E = (100 * E * B) / P :=
by sorry

end NUMINAMATH_CALUDE_correct_bottles_calculation_l676_67614


namespace NUMINAMATH_CALUDE_boxes_with_no_items_l676_67664

/-- Given the following conditions:
  - There are 15 boxes in total
  - 8 boxes contain pencils
  - 5 boxes contain pens
  - 3 boxes contain markers
  - 4 boxes contain both pens and pencils
  - 1 box contains all three items (pencils, pens, and markers)
  Prove that the number of boxes containing neither pens, pencils, nor markers is 5. -/
theorem boxes_with_no_items (total : ℕ) (pencil : ℕ) (pen : ℕ) (marker : ℕ) 
  (pen_and_pencil : ℕ) (all_three : ℕ) :
  total = 15 →
  pencil = 8 →
  pen = 5 →
  marker = 3 →
  pen_and_pencil = 4 →
  all_three = 1 →
  total - (pen_and_pencil + (pencil - pen_and_pencil) + 
    (pen - pen_and_pencil) + (marker - all_three)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_boxes_with_no_items_l676_67664


namespace NUMINAMATH_CALUDE_line_equation_l676_67626

-- Define a line by its slope and y-intercept
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

-- Function to translate a line
def translate_line (l : Line) (dx : ℝ) (dy : ℝ) : Line :=
  { slope := l.slope,
    y_intercept := l.y_intercept - l.slope * dx + dy }

-- Theorem statement
theorem line_equation (l : Line) :
  point_on_line { x := 1, y := 1 } l ∧
  translate_line (translate_line l 2 0) 0 (-1) = l →
  l.slope = 1/2 ∧ l.y_intercept = 1/2 :=
sorry

end NUMINAMATH_CALUDE_line_equation_l676_67626


namespace NUMINAMATH_CALUDE_fry_all_cutlets_in_15_minutes_l676_67612

/-- Represents a cutlet that needs to be fried -/
structure Cutlet where
  sides_fried : Fin 2 → Bool
  deriving Repr

/-- Represents the state of frying cutlets -/
structure FryingState where
  time : ℕ
  cutlets : Fin 3 → Cutlet
  pan : Fin 2 → Option (Fin 3)
  deriving Repr

/-- Checks if all cutlets are fully fried -/
def all_fried (state : FryingState) : Prop :=
  ∀ i : Fin 3, (state.cutlets i).sides_fried 0 ∧ (state.cutlets i).sides_fried 1

/-- Represents a valid frying step -/
def valid_step (before after : FryingState) : Prop :=
  after.time = before.time + 5 ∧
  (∀ i : Fin 3, 
    (after.cutlets i).sides_fried 0 = (before.cutlets i).sides_fried 0 ∨
    (after.cutlets i).sides_fried 1 = (before.cutlets i).sides_fried 1) ∧
  (∀ i : Fin 2, after.pan i ≠ none → 
    (∃ j : Fin 3, after.pan i = some j ∧ 
      ((before.cutlets j).sides_fried 0 ≠ (after.cutlets j).sides_fried 0 ∨
       (before.cutlets j).sides_fried 1 ≠ (after.cutlets j).sides_fried 1)))

/-- The initial state of frying -/
def initial_state : FryingState := {
  time := 0,
  cutlets := λ _ ↦ { sides_fried := λ _ ↦ false },
  pan := λ _ ↦ none
}

/-- Theorem stating that it's possible to fry all cutlets in 15 minutes -/
theorem fry_all_cutlets_in_15_minutes : 
  ∃ (final_state : FryingState), 
    final_state.time ≤ 15 ∧ 
    all_fried final_state ∧
    ∃ (step1 step2 : FryingState), 
      valid_step initial_state step1 ∧
      valid_step step1 step2 ∧
      valid_step step2 final_state :=
sorry

end NUMINAMATH_CALUDE_fry_all_cutlets_in_15_minutes_l676_67612
