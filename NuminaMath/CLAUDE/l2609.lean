import Mathlib

namespace NUMINAMATH_CALUDE_calculate_monthly_income_l2609_260980

/-- Calculates the total monthly income given the specified distributions and remaining amount. -/
theorem calculate_monthly_income (children_percentage : Real) (investment_percentage : Real)
  (tax_percentage : Real) (fixed_expenses : Real) (donation_percentage : Real)
  (remaining_amount : Real) :
  let total_income := (remaining_amount + fixed_expenses) /
    (1 - 3 * children_percentage - investment_percentage - tax_percentage -
     donation_percentage * (1 - 3 * children_percentage - investment_percentage - tax_percentage))
  (total_income - 3 * (children_percentage * total_income) -
   investment_percentage * total_income - tax_percentage * total_income - fixed_expenses -
   donation_percentage * (total_income - 3 * (children_percentage * total_income) -
   investment_percentage * total_income - tax_percentage * total_income - fixed_expenses)) =
  remaining_amount :=
by sorry

end NUMINAMATH_CALUDE_calculate_monthly_income_l2609_260980


namespace NUMINAMATH_CALUDE_certain_number_value_l2609_260951

theorem certain_number_value (y : ℕ) :
  (2^14 : ℕ) - (2^y : ℕ) = 3 * (2^12 : ℕ) → y = 13 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l2609_260951


namespace NUMINAMATH_CALUDE_min_length_AB_l2609_260942

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define the line y = 2
def line_y_2 (x y : ℝ) : Prop := y = 2

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- State the theorem
theorem min_length_AB :
  ∀ (x_A y_A x_B y_B : ℝ),
  line_y_2 x_A y_A →
  ellipse_C x_B y_B →
  perpendicular x_A y_A x_B y_B →
  ∀ (x y : ℝ),
  line_y_2 x y →
  ellipse_C x y →
  perpendicular x y x_B y_B →
  (x_A - x_B)^2 + (y_A - y_B)^2 ≤ (x - x_B)^2 + (y - y_B)^2 :=
sorry

end NUMINAMATH_CALUDE_min_length_AB_l2609_260942


namespace NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_8_equals_sqrt_2_l2609_260950

theorem sqrt_18_minus_sqrt_8_equals_sqrt_2 : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_8_equals_sqrt_2_l2609_260950


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l2609_260992

/-- Given Elaine's rent spending patterns over two years, prove that this year's rent
    is 187.5% of last year's rent. -/
theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.20 * last_year_earnings
  let this_year_earnings := 1.25 * last_year_earnings
  let this_year_rent := 0.30 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 187.5 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l2609_260992


namespace NUMINAMATH_CALUDE_base5_of_89_l2609_260915

-- Define a function to convert a natural number to its base-5 representation
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

-- Theorem stating that 89 in base-5 is equivalent to [4, 2, 3]
theorem base5_of_89 : toBase5 89 = [4, 2, 3] := by sorry

end NUMINAMATH_CALUDE_base5_of_89_l2609_260915


namespace NUMINAMATH_CALUDE_easiest_to_pick_black_l2609_260953

structure Box where
  label : Char
  black_balls : ℕ
  white_balls : ℕ

def probability_black (b : Box) : ℚ :=
  b.black_balls / (b.black_balls + b.white_balls)

def boxes : List Box := [
  ⟨'A', 12, 4⟩,
  ⟨'B', 10, 10⟩,
  ⟨'C', 4, 2⟩,
  ⟨'D', 10, 5⟩
]

theorem easiest_to_pick_black (boxes : List Box) :
  ∃ b ∈ boxes, ∀ b' ∈ boxes, probability_black b ≥ probability_black b' :=
sorry

end NUMINAMATH_CALUDE_easiest_to_pick_black_l2609_260953


namespace NUMINAMATH_CALUDE_perfect_square_form_l2609_260913

theorem perfect_square_form (k : ℕ+) : ∃ (n : ℕ+) (a : ℤ), a^2 = n * 2^(k : ℕ) - 7 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_form_l2609_260913


namespace NUMINAMATH_CALUDE_min_value_xy_over_x2_plus_y2_l2609_260993

theorem min_value_xy_over_x2_plus_y2 (x y : ℝ) 
  (hx : 0.4 ≤ x ∧ x ≤ 0.6) (hy : 0.3 ≤ y ∧ y ≤ 0.5) : 
  x * y / (x^2 + y^2) ≥ 0.4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_over_x2_plus_y2_l2609_260993


namespace NUMINAMATH_CALUDE_min_max_values_l2609_260936

theorem min_max_values (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + 3*b = 5) :
  (((1 : ℝ) / (a - b)) + (4 / (b - 1)) ≥ 25) ∧ (a*b - b^2 - a + b ≤ (1 : ℝ) / 16) := by
  sorry

end NUMINAMATH_CALUDE_min_max_values_l2609_260936


namespace NUMINAMATH_CALUDE_dinner_bill_problem_l2609_260981

theorem dinner_bill_problem (P : ℝ) : 
  (P * 0.9 + P * 0.08 + P * 0.15) - (P * 0.85 + P * 0.06 + P * 0.85 * 0.15) = 1 → 
  P = 400 / 37 := by
sorry

end NUMINAMATH_CALUDE_dinner_bill_problem_l2609_260981


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2609_260943

theorem trigonometric_identity (α β : ℝ) :
  1 - Real.sin α ^ 2 - Real.sin β ^ 2 + 2 * Real.sin α * Real.sin β * Real.cos (α - β) = 
  Real.cos (α - β) ^ 2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2609_260943


namespace NUMINAMATH_CALUDE_time_to_school_gate_l2609_260962

/-- Proves that the time to arrive at the school gate is 15 minutes -/
theorem time_to_school_gate 
  (total_time : ℕ) 
  (gate_to_building : ℕ) 
  (building_to_room : ℕ) 
  (h1 : total_time = 30) 
  (h2 : gate_to_building = 6) 
  (h3 : building_to_room = 9) : 
  total_time - gate_to_building - building_to_room = 15 := by
sorry

end NUMINAMATH_CALUDE_time_to_school_gate_l2609_260962


namespace NUMINAMATH_CALUDE_greatest_x_implies_n_l2609_260923

theorem greatest_x_implies_n (x : ℤ) (n : ℝ) : 
  (∀ y : ℤ, 2.13 * (10 : ℝ) ^ y < n → y ≤ 2) →
  (2.13 * (10 : ℝ) ^ 2 < n) ∧
  (∀ m : ℝ, m < n → m ≤ 213) ∧
  (n ≥ 214) :=
sorry

end NUMINAMATH_CALUDE_greatest_x_implies_n_l2609_260923


namespace NUMINAMATH_CALUDE_johns_remaining_money_l2609_260971

theorem johns_remaining_money (initial_amount : ℚ) : 
  initial_amount = 200 → 
  initial_amount - (3/8 * initial_amount + 3/10 * initial_amount) = 65 := by
sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l2609_260971


namespace NUMINAMATH_CALUDE_socks_difference_l2609_260901

/-- Proves that after losing half of the white socks, the person still has 6 more white socks than black socks -/
theorem socks_difference (black_socks : ℕ) (white_socks : ℕ) : 
  black_socks = 6 →
  white_socks = 4 * black_socks →
  (white_socks / 2) - black_socks = 6 := by
sorry

end NUMINAMATH_CALUDE_socks_difference_l2609_260901


namespace NUMINAMATH_CALUDE_rotation_dilation_determinant_l2609_260941

theorem rotation_dilation_determinant :
  ∀ (E : Matrix (Fin 2) (Fin 2) ℝ),
  (∃ (R S : Matrix (Fin 2) (Fin 2) ℝ),
    R = !![0, -1; 1, 0] ∧
    S = !![5, 0; 0, 5] ∧
    E = S * R) →
  Matrix.det E = 25 := by
sorry

end NUMINAMATH_CALUDE_rotation_dilation_determinant_l2609_260941


namespace NUMINAMATH_CALUDE_profit_maximization_l2609_260904

noncomputable def y (x : ℝ) : ℝ := 20 * x - 3 * x^2 + 96 * Real.log x - 90

theorem profit_maximization :
  ∃ (x_max : ℝ), 
    (4 ≤ x_max ∧ x_max ≤ 12) ∧
    (∀ x, 4 ≤ x ∧ x ≤ 12 → y x ≤ y x_max) ∧
    x_max = 6 ∧
    y x_max = 96 * Real.log 6 - 78 :=
sorry

end NUMINAMATH_CALUDE_profit_maximization_l2609_260904


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l2609_260949

theorem sum_of_two_numbers (x y : ℝ) (h1 : y = 4 * x) (h2 : x + y = 45) : x + y = 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l2609_260949


namespace NUMINAMATH_CALUDE_quadratic_minimum_minimum_at_three_l2609_260933

theorem quadratic_minimum (x : ℝ) : x^2 - 6*x + 1 ≥ -8 ∧ ∃ x₀ : ℝ, x₀^2 - 6*x₀ + 1 = -8 := by
  sorry

theorem minimum_at_three : (3 : ℝ)^2 - 6*3 + 1 = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_minimum_at_three_l2609_260933


namespace NUMINAMATH_CALUDE_constant_term_product_l2609_260907

-- Define polynomials p, q, and r
variable (p q r : ℝ → ℝ)

-- State the theorem
theorem constant_term_product (h1 : ∀ x, r x = p x * q x) 
                               (h2 : p 0 = 5) 
                               (h3 : r 0 = -10) : 
  q 0 = -2 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_product_l2609_260907


namespace NUMINAMATH_CALUDE_not_in_range_of_g_l2609_260900

/-- The function g(x) defined as x^3 + x^2 + bx + 2 -/
def g (b : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + b*x + 2

/-- Theorem stating that for all real b ≠ 6, -2 is not in the range of g(x) -/
theorem not_in_range_of_g (b : ℝ) (h : b ≠ 6) :
  ¬∃ x, g b x = -2 := by sorry

end NUMINAMATH_CALUDE_not_in_range_of_g_l2609_260900


namespace NUMINAMATH_CALUDE_select_three_from_five_eq_ten_distribute_five_to_three_eq_onefifty_l2609_260959

def select_three_from_five : ℕ := Nat.choose 5 3

def distribute_five_to_three : ℕ :=
  let scenario1 := Nat.choose 5 3 * Nat.factorial 3
  let scenario2 := Nat.choose 5 1 * Nat.choose 4 2 * Nat.factorial 3 / 2
  scenario1 + scenario2

theorem select_three_from_five_eq_ten :
  select_three_from_five = 10 := by sorry

theorem distribute_five_to_three_eq_onefifty :
  distribute_five_to_three = 150 := by sorry

end NUMINAMATH_CALUDE_select_three_from_five_eq_ten_distribute_five_to_three_eq_onefifty_l2609_260959


namespace NUMINAMATH_CALUDE_inscribed_circles_radii_equal_l2609_260922

theorem inscribed_circles_radii_equal (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let r₁ := a * b / (a + b)
  let r₂ := a * b / (a + b)
  r₁ = r₂ := by sorry

end NUMINAMATH_CALUDE_inscribed_circles_radii_equal_l2609_260922


namespace NUMINAMATH_CALUDE_flag_arrangement_remainder_l2609_260903

/-- The number of distinguishable arrangements of flags on two poles -/
def N : ℕ :=
  let blue_flags := 10
  let green_flags := 9
  let total_flags := blue_flags + green_flags
  let poles := 2
  -- Definition of N based on the problem conditions
  -- (Actual calculation is omitted as it's part of the proof)
  2310

/-- Theorem stating that N mod 1000 = 310 -/
theorem flag_arrangement_remainder :
  N % 1000 = 310 := by
  sorry

end NUMINAMATH_CALUDE_flag_arrangement_remainder_l2609_260903


namespace NUMINAMATH_CALUDE_min_value_of_objective_function_l2609_260938

-- Define the constraint region
def ConstraintRegion (x y : ℝ) : Prop :=
  2 * x - y ≥ 0 ∧ y ≥ x ∧ y ≥ -x + 2

-- Define the objective function
def ObjectiveFunction (x y : ℝ) : ℝ := 2 * x + y

-- Theorem statement
theorem min_value_of_objective_function :
  ∃ (min_z : ℝ), min_z = 8/3 ∧
  (∀ (x y : ℝ), ConstraintRegion x y → ObjectiveFunction x y ≥ min_z) ∧
  (∃ (x y : ℝ), ConstraintRegion x y ∧ ObjectiveFunction x y = min_z) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_objective_function_l2609_260938


namespace NUMINAMATH_CALUDE_difference_of_roots_absolute_value_l2609_260974

theorem difference_of_roots_absolute_value (a b c : ℝ) (h : a ≠ 0) :
  let r₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 1 ∧ b = -7 ∧ c = 10 → |r₁ - r₂| = 3 :=
by sorry

end NUMINAMATH_CALUDE_difference_of_roots_absolute_value_l2609_260974


namespace NUMINAMATH_CALUDE_chord_intersection_ratio_l2609_260948

-- Define a circle
variable (circle : Type) [AddCommGroup circle] [Module ℝ circle]

-- Define points on the circle
variable (E F G H Q : circle)

-- Define the lengths
variable (EQ FQ GQ HQ : ℝ)

-- State the theorem
theorem chord_intersection_ratio 
  (h1 : EQ = 5) 
  (h2 : GQ = 12) 
  (h3 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_chord_intersection_ratio_l2609_260948


namespace NUMINAMATH_CALUDE_gecko_eats_15_bugs_l2609_260983

/-- The number of bugs eaten by various creatures in a garden --/
structure GardenBugs where
  gecko : ℕ
  lizard : ℕ
  frog : ℕ
  toad : ℕ

/-- The conditions of the bug-eating scenario in the garden --/
def validGardenBugs (bugs : GardenBugs) : Prop :=
  bugs.lizard = bugs.gecko / 2 ∧
  bugs.frog = 3 * bugs.lizard ∧
  bugs.toad = (3 * bugs.frog) / 2 ∧
  bugs.gecko + bugs.lizard + bugs.frog + bugs.toad = 63

/-- The theorem stating that the gecko eats 15 bugs --/
theorem gecko_eats_15_bugs :
  ∃ (bugs : GardenBugs), validGardenBugs bugs ∧ bugs.gecko = 15 := by
  sorry

end NUMINAMATH_CALUDE_gecko_eats_15_bugs_l2609_260983


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l2609_260964

theorem max_value_of_trig_function (a b : ℝ) :
  (∀ x : ℝ, a * Real.cos x + b ≤ 1) →
  (∀ x : ℝ, a * Real.cos x + b ≥ -7) →
  (∃ x : ℝ, a * Real.cos x + b = 1) →
  (∃ x : ℝ, a * Real.cos x + b = -7) →
  (∀ x : ℝ, a * Real.cos x + b * Real.sin x ≤ 5) ∧
  (∃ x : ℝ, a * Real.cos x + b * Real.sin x = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l2609_260964


namespace NUMINAMATH_CALUDE_fraction_between_main_theorem_l2609_260961

theorem fraction_between (a b c d m n : ℕ) (h1 : 0 < b) (h2 : 0 < d) (h3 : 0 < n) :
  a * d < c * b → c * n < m * d → a * n < m * b →
  (a : ℚ) / b < (m : ℚ) / n ∧ (m : ℚ) / n < (c : ℚ) / d :=
by sorry

theorem main_theorem :
  (5 : ℚ) / 14 < (8 : ℚ) / 21 ∧ (8 : ℚ) / 21 < (5 : ℚ) / 12 :=
by sorry

end NUMINAMATH_CALUDE_fraction_between_main_theorem_l2609_260961


namespace NUMINAMATH_CALUDE_unique_prime_solution_l2609_260988

theorem unique_prime_solution :
  ∀ p q r : ℕ,
    Prime p ∧ Prime q ∧ Prime r →
    p + q^2 = r^4 →
    p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l2609_260988


namespace NUMINAMATH_CALUDE_ngo_employee_count_l2609_260925

/-- The number of illiterate employees -/
def illiterate_employees : ℕ := 20

/-- The decrease in total wages of illiterate employees in Rupees -/
def total_wage_decrease : ℕ := 300

/-- The decrease in average salary for all employees in Rupees -/
def average_salary_decrease : ℕ := 10

/-- The number of educated employees in the NGO -/
def educated_employees : ℕ := 10

theorem ngo_employee_count :
  educated_employees = total_wage_decrease / average_salary_decrease - illiterate_employees :=
by sorry

end NUMINAMATH_CALUDE_ngo_employee_count_l2609_260925


namespace NUMINAMATH_CALUDE_inverse_307_mod_455_l2609_260967

theorem inverse_307_mod_455 : ∃ x : ℕ, x < 455 ∧ (307 * x) % 455 = 1 :=
by
  use 81
  sorry

end NUMINAMATH_CALUDE_inverse_307_mod_455_l2609_260967


namespace NUMINAMATH_CALUDE_solution_set_l2609_260994

theorem solution_set : 
  {x : ℝ | x * (x + 3)^2 * (5 - x) = 0 ∧ x^2 + 3*x + 2 > 0} = {-3, 0, 5} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l2609_260994


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l2609_260970

/-- Two lines in R² defined by their parametric equations -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Checks if two lines are parallel -/
def are_parallel (l1 l2 : Line2D) : Prop :=
  ∃ c : ℝ, l1.direction = (c * l2.direction.1, c * l2.direction.2)

/-- The problem statement -/
theorem parallel_lines_k_value :
  ∃! k : ℝ, are_parallel
    (Line2D.mk (2, 3) (6, -9))
    (Line2D.mk (-1, 0) (3, k))
  ∧ k = -4.5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l2609_260970


namespace NUMINAMATH_CALUDE_parabola_fixed_point_l2609_260905

/-- The parabola equation as a function of x and p -/
def parabola (x p : ℝ) : ℝ := 2 * x^2 - p * x + 4 * p + 1

/-- The fixed point through which the parabola passes -/
def fixed_point : ℝ × ℝ := (4, 33)

theorem parabola_fixed_point :
  ∀ p : ℝ, parabola (fixed_point.1) p = fixed_point.2 := by
  sorry

#check parabola_fixed_point

end NUMINAMATH_CALUDE_parabola_fixed_point_l2609_260905


namespace NUMINAMATH_CALUDE_at_least_one_not_in_area_l2609_260991

theorem at_least_one_not_in_area (p q : Prop) : 
  (¬p ∨ ¬q) ↔ (∃ x, x = p ∨ x = q) ∧ (x → False) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_not_in_area_l2609_260991


namespace NUMINAMATH_CALUDE_equation_and_inequalities_l2609_260947

theorem equation_and_inequalities (x a : ℝ) (hx : x ≠ 0) :
  (x⁻¹ + a * x = 1 ↔ a = (x - 1) / x^2) ∧
  (x⁻¹ + a * x > 1 ↔ (a > (x - 1) / x^2 ∧ x > 0) ∨ (a < (x - 1) / x^2 ∧ x < 0)) ∧
  (x⁻¹ + a * x < 1 ↔ (a < (x - 1) / x^2 ∧ x > 0) ∨ (a > (x - 1) / x^2 ∧ x < 0)) := by
  sorry

end NUMINAMATH_CALUDE_equation_and_inequalities_l2609_260947


namespace NUMINAMATH_CALUDE_purple_part_length_l2609_260906

/-- The length of the purple part of a pencil -/
def purple_length : ℝ := 1.5

/-- The length of the black part of a pencil -/
def black_length : ℝ := 0.5

/-- The length of the blue part of a pencil -/
def blue_length : ℝ := 2

/-- The total length of the pencil -/
def total_length : ℝ := 4

/-- Theorem stating that the length of the purple part of the pencil is 1.5 cm -/
theorem purple_part_length :
  purple_length = total_length - (black_length + blue_length) :=
by sorry

end NUMINAMATH_CALUDE_purple_part_length_l2609_260906


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l2609_260934

theorem least_positive_integer_with_remainder_one : ∃ n : ℕ,
  n > 1 ∧
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 10 → n % k = 1) ∧
  (∀ m : ℕ, m > 1 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ 10 → m % k = 1) → n ≤ m) ∧
  n = 2521 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l2609_260934


namespace NUMINAMATH_CALUDE_smallest_division_is_six_l2609_260932

/-- A typical rectangular parallelepiped has all dimensions different -/
structure TypicalParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  all_different : length ≠ width ∧ width ≠ height ∧ length ≠ height

/-- A cube has all sides equal -/
structure Cube where
  side : ℝ

/-- The division of a cube into typical parallelepipeds -/
def CubeDivision (c : Cube) := List TypicalParallelepiped

/-- Predicate to check if a division is valid (i.e., the parallelepipeds fill the cube exactly) -/
def IsValidDivision (c : Cube) (d : CubeDivision c) : Prop := sorry

/-- The smallest number of typical parallelepipeds into which a cube can be divided is 6 -/
theorem smallest_division_is_six (c : Cube) : 
  (∃ (d : CubeDivision c), IsValidDivision c d ∧ d.length = 6) ∧
  (∀ (d : CubeDivision c), IsValidDivision c d → d.length ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_smallest_division_is_six_l2609_260932


namespace NUMINAMATH_CALUDE_parallelogram_area_l2609_260935

def v : ℝ × ℝ := (7, 4)
def w : ℝ × ℝ := (2, -9)

theorem parallelogram_area : 
  let v2w := (2 * w.1, 2 * w.2)
  abs (v.1 * v2w.2 - v.2 * v2w.1) = 142 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2609_260935


namespace NUMINAMATH_CALUDE_factorization_x4_minus_4x2_l2609_260917

theorem factorization_x4_minus_4x2 (x : ℝ) : x^4 - 4*x^2 = x^2 * (x - 2) * (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_4x2_l2609_260917


namespace NUMINAMATH_CALUDE_f_image_is_closed_interval_l2609_260928

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 6*x + 7

-- Define the domain
def domain : Set ℝ := Set.Ioc 2 5

-- Theorem statement
theorem f_image_is_closed_interval :
  Set.image f domain = Set.Icc (-2) 2 := by sorry

end NUMINAMATH_CALUDE_f_image_is_closed_interval_l2609_260928


namespace NUMINAMATH_CALUDE_circumference_diameter_ratio_l2609_260921

/-- The ratio of circumference to diameter for a ring with radius 15 cm and circumference 90 cm is 3. -/
theorem circumference_diameter_ratio :
  let radius : ℝ := 15
  let circumference : ℝ := 90
  let diameter : ℝ := 2 * radius
  circumference / diameter = 3 := by
  sorry

end NUMINAMATH_CALUDE_circumference_diameter_ratio_l2609_260921


namespace NUMINAMATH_CALUDE_real_axis_length_of_hyperbola_l2609_260912

/-- The length of the real axis of a hyperbola with equation x^2 - y^2/9 = 1 is 2 -/
theorem real_axis_length_of_hyperbola :
  let hyperbola_equation := fun (x y : ℝ) => x^2 - y^2/9 = 1
  ∃ a : ℝ, a > 0 ∧ hyperbola_equation = fun (x y : ℝ) => x^2/a^2 - y^2/(9*a^2) = 1 →
  (real_axis_length : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_real_axis_length_of_hyperbola_l2609_260912


namespace NUMINAMATH_CALUDE_genuine_purses_and_handbags_l2609_260958

theorem genuine_purses_and_handbags (total_purses : ℕ) (total_handbags : ℕ)
  (h_purses : total_purses = 26)
  (h_handbags : total_handbags = 24)
  (fake_purses : ℕ → ℕ)
  (fake_handbags : ℕ → ℕ)
  (h_fake_purses : fake_purses total_purses = total_purses / 2)
  (h_fake_handbags : fake_handbags total_handbags = total_handbags / 4) :
  total_purses - fake_purses total_purses + total_handbags - fake_handbags total_handbags = 31 := by
  sorry

end NUMINAMATH_CALUDE_genuine_purses_and_handbags_l2609_260958


namespace NUMINAMATH_CALUDE_radical_sum_equals_eight_sqrt_three_l2609_260910

theorem radical_sum_equals_eight_sqrt_three :
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_radical_sum_equals_eight_sqrt_three_l2609_260910


namespace NUMINAMATH_CALUDE_town_population_problem_l2609_260930

theorem town_population_problem : ∃ (n : ℝ), 
  n > 0 ∧ 
  0.92 * (0.85 * (n + 2500)) = n + 49 ∧ 
  n = 8740 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l2609_260930


namespace NUMINAMATH_CALUDE_inverse_negation_equivalence_l2609_260945

-- Define a quadrilateral type
structure Quadrilateral where
  isParallelogram : Prop
  oppositeSidesEqual : Prop

-- Define the original proposition
def originalProposition (q : Quadrilateral) : Prop :=
  q.oppositeSidesEqual → q.isParallelogram

-- Define the inverse negation
def inverseNegation (q : Quadrilateral) : Prop :=
  ¬q.isParallelogram → ¬q.oppositeSidesEqual

-- Theorem stating the equivalence of the inverse negation
theorem inverse_negation_equivalence :
  ∀ q : Quadrilateral, inverseNegation q ↔ ¬(originalProposition q) :=
sorry

end NUMINAMATH_CALUDE_inverse_negation_equivalence_l2609_260945


namespace NUMINAMATH_CALUDE_polygon_angles_l2609_260926

theorem polygon_angles (n : ℕ) (sum_interior : ℝ) (sum_exterior : ℝ) : 
  sum_exterior = 180 → 
  sum_interior = 4 * sum_exterior → 
  sum_interior = (n - 2) * 180 → 
  n = 11 ∧ sum_interior = 1620 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angles_l2609_260926


namespace NUMINAMATH_CALUDE_tamara_cracker_count_l2609_260924

/-- The number of crackers each person has -/
structure CrackerCount where
  tamara : ℕ
  nicholas : ℕ
  marcus : ℕ
  mona : ℕ

/-- The conditions of the cracker problem -/
def CrackerProblem (c : CrackerCount) : Prop :=
  c.tamara = 2 * c.nicholas ∧
  c.marcus = 3 * c.mona ∧
  c.nicholas = c.mona + 6 ∧
  c.marcus = 27

theorem tamara_cracker_count (c : CrackerCount) (h : CrackerProblem c) : c.tamara = 30 := by
  sorry

end NUMINAMATH_CALUDE_tamara_cracker_count_l2609_260924


namespace NUMINAMATH_CALUDE_tea_box_duration_l2609_260927

-- Define the daily tea usage in ounces
def daily_usage : ℚ := 1 / 5

-- Define the box size in ounces
def box_size : ℚ := 28

-- Define the number of days in a week
def days_per_week : ℕ := 7

-- Theorem to prove
theorem tea_box_duration : 
  (box_size / daily_usage) / days_per_week = 20 := by
  sorry

end NUMINAMATH_CALUDE_tea_box_duration_l2609_260927


namespace NUMINAMATH_CALUDE_number_of_black_balls_is_random_variable_l2609_260975

-- Define the total number of balls
def total_balls : ℕ := 8

-- Define the number of black balls
def black_balls : ℕ := 5

-- Define the number of white balls
def white_balls : ℕ := 3

-- Define the number of balls drawn
def drawn_balls : ℕ := 2

-- Define the possible outcomes for the number of black balls drawn
def possible_outcomes : Set ℕ := {0, 1, 2}

-- Define a random variable as a function from the sample space to the set of real numbers
def is_random_variable (X : Set ℕ → ℝ) : Prop :=
  ∀ n ∈ possible_outcomes, X {n} ∈ Set.range X

-- State the theorem
theorem number_of_black_balls_is_random_variable :
  ∃ X : Set ℕ → ℝ, is_random_variable X ∧ 
  (∀ n, X {n} = n) ∧
  (∀ n ∉ possible_outcomes, X {n} = 0) :=
sorry

end NUMINAMATH_CALUDE_number_of_black_balls_is_random_variable_l2609_260975


namespace NUMINAMATH_CALUDE_bicycles_in_garage_l2609_260998

/-- The number of bicycles in Connor's garage --/
def num_bicycles : ℕ := 20

/-- The number of cars in Connor's garage --/
def num_cars : ℕ := 10

/-- The number of motorcycles in Connor's garage --/
def num_motorcycles : ℕ := 5

/-- The total number of wheels in Connor's garage --/
def total_wheels : ℕ := 90

/-- The number of wheels on a bicycle --/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a car --/
def wheels_per_car : ℕ := 4

/-- The number of wheels on a motorcycle --/
def wheels_per_motorcycle : ℕ := 2

theorem bicycles_in_garage :
  num_bicycles * wheels_per_bicycle +
  num_cars * wheels_per_car +
  num_motorcycles * wheels_per_motorcycle = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_bicycles_in_garage_l2609_260998


namespace NUMINAMATH_CALUDE_two_digit_number_equation_l2609_260914

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≥ 0 ∧ units ≤ 9

/-- The property that the unit digit is 3 greater than the tens digit -/
def unit_is_three_greater (n : TwoDigitNumber) : Prop :=
  n.units = n.tens + 3

/-- The property that the square of the unit digit equals the two-digit number -/
def square_of_unit_is_number (n : TwoDigitNumber) : Prop :=
  n.units ^ 2 = 10 * n.tens + n.units

/-- Theorem: For a two-digit number satisfying the given conditions, 
    the tens digit x satisfies the equation x^2 - 5x + 6 = 0 -/
theorem two_digit_number_equation (n : TwoDigitNumber) 
  (h1 : unit_is_three_greater n) 
  (h2 : square_of_unit_is_number n) : 
  n.tens ^ 2 - 5 * n.tens + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_equation_l2609_260914


namespace NUMINAMATH_CALUDE_expression_evaluation_l2609_260902

theorem expression_evaluation :
  let x : ℚ := 3/2
  (2 + x) * (2 - x) + (x - 1) * (x + 5) = 5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2609_260902


namespace NUMINAMATH_CALUDE_smallest_number_l2609_260944

/-- Converts a number from base b to decimal --/
def toDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

/-- The decimal representation of 85₍₉₎ --/
def num1 : Nat := toDecimal [5, 8] 9

/-- The decimal representation of 210₍₆₎ --/
def num2 : Nat := toDecimal [0, 1, 2] 6

/-- The decimal representation of 1000₍₄₎ --/
def num3 : Nat := toDecimal [0, 0, 0, 1] 4

/-- The decimal representation of 111111₍₂₎ --/
def num4 : Nat := toDecimal [1, 1, 1, 1, 1, 1] 2

/-- Theorem stating that 111111₍₂₎ is the smallest among the given numbers --/
theorem smallest_number : num4 ≤ num1 ∧ num4 ≤ num2 ∧ num4 ≤ num3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2609_260944


namespace NUMINAMATH_CALUDE_custom_mult_solution_l2609_260997

/-- Custom multiplication operation for integers -/
def customMult (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that if 21b = 160 under the custom multiplication, then b = 9 -/
theorem custom_mult_solution :
  ∀ b : ℤ, customMult 21 b = 160 → b = 9 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_solution_l2609_260997


namespace NUMINAMATH_CALUDE_exactly_one_two_digit_sum_with_reverse_is_cube_l2609_260939

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

theorem exactly_one_two_digit_sum_with_reverse_is_cube : 
  ∃! n : ℕ, is_two_digit n ∧ is_perfect_cube (n + reverse_digits n) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_two_digit_sum_with_reverse_is_cube_l2609_260939


namespace NUMINAMATH_CALUDE_power_of_negative_product_l2609_260955

theorem power_of_negative_product (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_negative_product_l2609_260955


namespace NUMINAMATH_CALUDE_value_of_expression_l2609_260989

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem value_of_expression (a b c d : ℝ) :
  f a b c d (-2) = -3 →
  8*a - 4*b + 2*c - d = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2609_260989


namespace NUMINAMATH_CALUDE_pizzas_served_today_l2609_260920

/-- The number of pizzas served during lunch -/
def lunch_pizzas : ℕ := 9

/-- The number of pizzas served during dinner -/
def dinner_pizzas : ℕ := 6

/-- The total number of pizzas served today -/
def total_pizzas : ℕ := lunch_pizzas + dinner_pizzas

theorem pizzas_served_today : total_pizzas = 15 := by
  sorry

end NUMINAMATH_CALUDE_pizzas_served_today_l2609_260920


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l2609_260946

-- Define a decreasing function on ℝ
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- Theorem statement
theorem decreasing_function_inequality (f : ℝ → ℝ) (h : DecreasingOn f) : f 3 > f 5 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l2609_260946


namespace NUMINAMATH_CALUDE_regular_octagon_perimeter_l2609_260977

/-- The perimeter of a regular octagon with side length 2 is 16 -/
theorem regular_octagon_perimeter : 
  ∀ (side_length : ℝ), 
  side_length = 2 → 
  (8 : ℝ) * side_length = 16 := by
sorry

end NUMINAMATH_CALUDE_regular_octagon_perimeter_l2609_260977


namespace NUMINAMATH_CALUDE_expression_equals_seventeen_l2609_260966

theorem expression_equals_seventeen : 1-(-2) * 2 - 3 - (-4) * 2 - 5 - (-6) * 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_seventeen_l2609_260966


namespace NUMINAMATH_CALUDE_latus_rectum_equation_l2609_260995

/-- The equation of the latus rectum of the parabola y = -1/4 * x^2 -/
theorem latus_rectum_equation (x y : ℝ) :
  y = -1/4 * x^2 → (∃ (p : ℝ), p = -1/2 ∧ y = p) :=
by sorry

end NUMINAMATH_CALUDE_latus_rectum_equation_l2609_260995


namespace NUMINAMATH_CALUDE_loan_amount_l2609_260982

/-- Proves that given the conditions of the loan, the sum lent must be 500 Rs. -/
theorem loan_amount (interest_rate : ℚ) (time : ℕ) (interest_difference : ℚ) : 
  interest_rate = 4/100 →
  time = 8 →
  interest_difference = 340 →
  ∃ (principal : ℚ), 
    principal * interest_rate * time = principal - interest_difference ∧
    principal = 500 := by
  sorry

end NUMINAMATH_CALUDE_loan_amount_l2609_260982


namespace NUMINAMATH_CALUDE_predecessor_in_binary_l2609_260957

def binary_to_nat (b : List Bool) : Nat :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

theorem predecessor_in_binary :
  let Q : List Bool := [true, true, false, true, false, true, false]
  let Q_nat : Nat := binary_to_nat Q
  let pred_Q : List Bool := nat_to_binary (Q_nat - 1)
  pred_Q = [true, true, false, true, false, false, true] := by
  sorry

end NUMINAMATH_CALUDE_predecessor_in_binary_l2609_260957


namespace NUMINAMATH_CALUDE_gwens_birthday_money_l2609_260960

/-- The amount of money Gwen received from her mom -/
def money_from_mom : ℕ := sorry

/-- The amount of money Gwen received from her dad -/
def money_from_dad : ℕ := 5

/-- The amount of money Gwen spent -/
def money_spent : ℕ := 4

/-- The difference between money from mom and dad after spending -/
def difference_after_spending : ℕ := 2

theorem gwens_birthday_money : 
  money_from_mom = 6 ∧
  money_from_mom + money_from_dad - money_spent = 
  money_from_dad + difference_after_spending :=
by sorry

end NUMINAMATH_CALUDE_gwens_birthday_money_l2609_260960


namespace NUMINAMATH_CALUDE_min_contribution_l2609_260972

/-- Proves that given 10 people contributing a total of $20.00, with a maximum individual contribution of $11, the minimum amount each person must have contributed is $2.00. -/
theorem min_contribution (num_people : ℕ) (total_contribution : ℚ) (max_individual : ℚ) :
  num_people = 10 ∧ 
  total_contribution = 20 ∧ 
  max_individual = 11 →
  ∃ (min_contribution : ℚ),
    min_contribution = 2 ∧
    num_people * min_contribution = total_contribution ∧
    ∀ (individual : ℚ),
      individual ≥ min_contribution ∧
      individual ≤ max_individual ∧
      (num_people - 1) * min_contribution + individual = total_contribution :=
by sorry

end NUMINAMATH_CALUDE_min_contribution_l2609_260972


namespace NUMINAMATH_CALUDE_oplus_five_two_l2609_260909

def oplus (a b : ℝ) : ℝ := 3 * a + 4 * b

theorem oplus_five_two : oplus 5 2 = 23 := by sorry

end NUMINAMATH_CALUDE_oplus_five_two_l2609_260909


namespace NUMINAMATH_CALUDE_expression_value_l2609_260987

theorem expression_value : (45 - 13)^2 - (45^2 + 13^2) = -1170 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2609_260987


namespace NUMINAMATH_CALUDE_soccer_boys_percentage_l2609_260911

theorem soccer_boys_percentage (total_students boys soccer_players girls_not_playing : ℕ) : 
  total_students = 420 →
  boys = 312 →
  soccer_players = 250 →
  girls_not_playing = 63 →
  (boys - (total_students - boys - girls_not_playing)) / soccer_players * 100 = 82 := by
sorry

end NUMINAMATH_CALUDE_soccer_boys_percentage_l2609_260911


namespace NUMINAMATH_CALUDE_password_probability_l2609_260979

/-- The number of digits in the password -/
def password_length : ℕ := 6

/-- The set of possible even digits for the last position -/
def even_digits : Set ℕ := {0, 2, 4, 6, 8}

/-- The probability of guessing the correct password in one attempt, given the last digit is even -/
def prob_correct_first_attempt : ℚ := 1 / 5

/-- The probability of guessing the correct password in exactly two attempts, given the last digit is even -/
def prob_correct_second_attempt : ℚ := 4 / 25

/-- The probability of guessing the correct password in no more than two attempts, given the last digit is even -/
def prob_correct_within_two_attempts : ℚ := prob_correct_first_attempt + prob_correct_second_attempt

theorem password_probability : prob_correct_within_two_attempts = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_password_probability_l2609_260979


namespace NUMINAMATH_CALUDE_arctan_sum_equation_l2609_260931

theorem arctan_sum_equation : ∃ (n : ℕ+), 
  Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/7) + Real.arctan (1/n) = π/3 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equation_l2609_260931


namespace NUMINAMATH_CALUDE_distance_AD_MN_l2609_260969

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents the pyramid structure described in the problem -/
structure Pyramid where
  a : ℝ
  b : ℝ
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  M : Point3D
  N : Point3D

/-- The distance between two skew lines in 3D space -/
def distanceBetweenSkewLines (l1 l2 : Line3D) : ℝ :=
  sorry

/-- The main theorem stating the distance between AD and MN -/
theorem distance_AD_MN (p : Pyramid) :
  let AD := Line3D.mk p.A (Point3D.mk p.a p.a 0)
  let MN := Line3D.mk p.M (Point3D.mk 0 (p.a / 2) p.b)
  distanceBetweenSkewLines AD MN = (p.b / (2 * p.a)) * Real.sqrt (4 * p.a^2 - p.b^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_AD_MN_l2609_260969


namespace NUMINAMATH_CALUDE_weight_replacement_l2609_260908

theorem weight_replacement (n : ℕ) (old_weight new_weight avg_increase : ℝ) :
  n = 8 →
  new_weight = 95 →
  avg_increase = 2.5 →
  old_weight = new_weight - n * avg_increase →
  old_weight = 75 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l2609_260908


namespace NUMINAMATH_CALUDE_trig_identity_l2609_260978

theorem trig_identity (θ : ℝ) (h : 2 * Real.sin θ + Real.cos θ = 0) :
  Real.sin θ * Real.cos θ - Real.cos θ ^ 2 = - 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2609_260978


namespace NUMINAMATH_CALUDE_hexagon_square_side_ratio_l2609_260999

theorem hexagon_square_side_ratio (s_h s_s : ℝ) 
  (h_positive : s_h > 0 ∧ s_s > 0)
  (h_perimeter : 6 * s_h = 4 * s_s) : 
  s_s / s_h = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_square_side_ratio_l2609_260999


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2609_260937

theorem complex_sum_problem (a b c d e f : ℂ) : 
  b = 4 →
  e = -a - c →
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) = 6 + 3 * Complex.I →
  d + f = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2609_260937


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l2609_260956

theorem quadratic_equation_result (a : ℝ) (h : a^2 - 4*a - 12 = 0) : 2*a^2 - 8*a - 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l2609_260956


namespace NUMINAMATH_CALUDE_power_sum_divisibility_and_quotient_units_digit_l2609_260963

theorem power_sum_divisibility_and_quotient_units_digit :
  (∃ k : ℕ, 4^1987 + 6^1987 = 10 * k) ∧
  (∃ m : ℕ, 4^1987 + 6^1987 = 5 * m) ∧
  (∃ n : ℕ, (4^1987 + 6^1987) / 5 = 10 * n + 0) :=
by sorry

end NUMINAMATH_CALUDE_power_sum_divisibility_and_quotient_units_digit_l2609_260963


namespace NUMINAMATH_CALUDE_stella_monthly_income_l2609_260965

def months_in_year : ℕ := 12
def unpaid_leave_months : ℕ := 2
def annual_income : ℕ := 49190

def monthly_income : ℕ := annual_income / (months_in_year - unpaid_leave_months)

theorem stella_monthly_income : monthly_income = 4919 := by
  sorry

end NUMINAMATH_CALUDE_stella_monthly_income_l2609_260965


namespace NUMINAMATH_CALUDE_investments_sum_to_22000_l2609_260929

/-- Represents the initial investment amounts of five individuals --/
structure Investments where
  raghu : ℝ
  trishul : ℝ
  vishal : ℝ
  alok : ℝ
  harshit : ℝ

/-- Calculates the total sum of investments --/
def total_investment (i : Investments) : ℝ :=
  i.raghu + i.trishul + i.vishal + i.alok + i.harshit

/-- Theorem stating that the investments satisfy the given conditions and sum to 22000 --/
theorem investments_sum_to_22000 :
  ∃ (i : Investments),
    i.trishul = 0.9 * i.raghu ∧
    i.vishal = 1.1 * i.trishul ∧
    i.alok = 1.15 * i.trishul ∧
    i.harshit = 0.95 * i.vishal ∧
    total_investment i = 22000 :=
  sorry

end NUMINAMATH_CALUDE_investments_sum_to_22000_l2609_260929


namespace NUMINAMATH_CALUDE_compound_composition_l2609_260918

/-- Prove that a compound with 2 I atoms and a molecular weight of 294 g/mol contains 1 Ca atom -/
theorem compound_composition (atomic_weight_Ca atomic_weight_I : ℝ) 
  (h1 : atomic_weight_Ca = 40.08)
  (h2 : atomic_weight_I = 126.90)
  (h3 : 2 * atomic_weight_I + atomic_weight_Ca = 294) : 
  ∃ (n : ℕ), n = 1 ∧ n * atomic_weight_Ca = 294 - 2 * atomic_weight_I :=
by sorry

end NUMINAMATH_CALUDE_compound_composition_l2609_260918


namespace NUMINAMATH_CALUDE_max_lateral_area_cylinder_l2609_260984

/-- The maximum lateral area of a cylinder with a rectangular cross-section of perimeter 4 is π. -/
theorem max_lateral_area_cylinder (r h : ℝ) : 
  r > 0 → h > 0 → 2 * (2 * r + h) = 4 → 2 * π * r * h ≤ π := by
  sorry

end NUMINAMATH_CALUDE_max_lateral_area_cylinder_l2609_260984


namespace NUMINAMATH_CALUDE_meryll_question_ratio_l2609_260916

theorem meryll_question_ratio : 
  ∀ (total_mc : ℕ) (total_ps : ℕ) (written_mc_fraction : ℚ) (remaining : ℕ),
    total_mc = 35 →
    total_ps = 15 →
    written_mc_fraction = 2/5 →
    remaining = 31 →
    (total_mc * written_mc_fraction).num.toNat + 
    (total_ps - (remaining - (total_mc - (total_mc * written_mc_fraction).num.toNat))) = 
    total_ps / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_meryll_question_ratio_l2609_260916


namespace NUMINAMATH_CALUDE_average_daily_low_temp_l2609_260996

def daily_low_temperatures : List ℝ := [40, 47, 45, 41, 39]

theorem average_daily_low_temp : 
  (daily_low_temperatures.sum / daily_low_temperatures.length : ℝ) = 42.4 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_low_temp_l2609_260996


namespace NUMINAMATH_CALUDE_expression_evaluation_l2609_260973

theorem expression_evaluation (x y z : ℚ) 
  (hz : z = y - 11)
  (hy : y = x + 3)
  (hx : x = 5)
  (hd1 : x + 2 ≠ 0)
  (hd2 : y - 3 ≠ 0)
  (hd3 : z + 7 ≠ 0) :
  (x + 3) / (x + 2) * (y - 2) / (y - 3) * (z + 9) / (z + 7) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2609_260973


namespace NUMINAMATH_CALUDE_max_page_number_with_fifteen_fives_l2609_260990

/-- Represents the count of a specific digit in a number -/
def digitCount (n : ℕ) (d : ℕ) : ℕ := sorry

/-- Represents the total count of a specific digit used in numbering pages from 1 to n -/
def totalDigitCount (n : ℕ) (d : ℕ) : ℕ := sorry

/-- The maximum page number that can be reached with a given number of a specific digit -/
def maxPageNumber (availableDigits : ℕ) (digit : ℕ) : ℕ := sorry

theorem max_page_number_with_fifteen_fives :
  maxPageNumber 15 5 = 59 := by sorry

end NUMINAMATH_CALUDE_max_page_number_with_fifteen_fives_l2609_260990


namespace NUMINAMATH_CALUDE_smallest_integer_negative_quadratic_l2609_260940

theorem smallest_integer_negative_quadratic :
  ∃ (x : ℤ), (∀ (y : ℤ), y^2 - 11*y + 24 < 0 → x ≤ y) ∧ x^2 - 11*x + 24 < 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_negative_quadratic_l2609_260940


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l2609_260968

theorem roots_sum_of_squares (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 3 * x₁ - 1 = 0) → 
  (2 * x₂^2 - 3 * x₂ - 1 = 0) → 
  x₁^2 + x₂^2 = 13/4 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l2609_260968


namespace NUMINAMATH_CALUDE_complex_equation_first_quadrant_l2609_260919

theorem complex_equation_first_quadrant (z : ℂ) (a : ℝ) : 
  (1 - I) * z = a * I + 1 → 
  (z.re > 0 ∧ z.im > 0) → 
  a = 0 := by sorry

end NUMINAMATH_CALUDE_complex_equation_first_quadrant_l2609_260919


namespace NUMINAMATH_CALUDE_max_value_x3y2z_l2609_260976

theorem max_value_x3y2z (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  x^3 * y^2 * z ≤ 1/432 := by
sorry

end NUMINAMATH_CALUDE_max_value_x3y2z_l2609_260976


namespace NUMINAMATH_CALUDE_perpendicular_planes_l2609_260985

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the perpendicular relation between two planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (m n : Line) (α β : Plane) 
  (h1 : perp_line_plane m α) 
  (h2 : perp_line_plane n β) 
  (h3 : perp_line_line m n) : 
  perp_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l2609_260985


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l2609_260954

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, is_three_digit n → n % 9 = 0 → digit_sum n = 27 → n ≤ 999 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l2609_260954


namespace NUMINAMATH_CALUDE_quadratic_root_property_l2609_260986

theorem quadratic_root_property (m : ℝ) : 
  m^2 + m - 1 = 0 → 2*m^2 + 2*m + 2025 = 2027 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l2609_260986


namespace NUMINAMATH_CALUDE_expression_simplification_l2609_260952

theorem expression_simplification (a : ℕ) (h : a = 2023) :
  (a^3 - 2*a^2*(a+1) + 3*a*(a+1)^2 - (a+1)^3 + 2) / (a*(a+1)) = a + 1 / (a*(a+1)) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2609_260952
