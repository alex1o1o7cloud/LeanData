import Mathlib

namespace NUMINAMATH_CALUDE_power_of_power_l3949_394990

theorem power_of_power : (3^2)^4 = 6561 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l3949_394990


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3949_394934

/-- Given three numbers A, B, and C, where A is a three-digit number and B and C are two-digit numbers,
    if the sum of numbers containing the digit seven is 208 and the sum of numbers containing the digit three is 76,
    then the sum of A, B, and C is 247. -/
theorem sum_of_numbers (A B C : ℕ) : 
  100 ≤ A ∧ A < 1000 ∧   -- A is a three-digit number
  10 ≤ B ∧ B < 100 ∧     -- B is a two-digit number
  10 ≤ C ∧ C < 100 ∧     -- C is a two-digit number
  ((A.repr.contains '7' ∨ B.repr.contains '7' ∨ C.repr.contains '7') → A + B + C = 208) ∧  -- Sum of numbers with 7
  (B.repr.contains '3' ∧ C.repr.contains '3' → B + C = 76)  -- Sum of numbers with 3
  → A + B + C = 247 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3949_394934


namespace NUMINAMATH_CALUDE_block_dimension_l3949_394942

/-- The number of positions to place a 2x1x1 block in a layer of 11x10 --/
def positions_in_layer : ℕ := 199

/-- The number of positions to place a 2x1x1 block across two adjacent layers --/
def positions_across_layers : ℕ := 110

/-- The total number of positions to place a 2x1x1 block in a nx11x10 block --/
def total_positions (n : ℕ) : ℕ := n * positions_in_layer + (n - 1) * positions_across_layers

theorem block_dimension (n : ℕ) :
  total_positions n = 2362 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_block_dimension_l3949_394942


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3949_394908

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) 
  (h1 : ∀ x, a * x^2 + b * x + c > 0 ↔ x < -2 ∨ x > 4) :
  f a b c 2 < f a b c (-1) ∧ f a b c (-1) < f a b c 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3949_394908


namespace NUMINAMATH_CALUDE_lcm_18_30_l3949_394933

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_30_l3949_394933


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l3949_394959

theorem abs_eq_sqrt_sq (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l3949_394959


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3949_394903

theorem quadratic_one_root (n : ℝ) : 
  (∃! x : ℝ, x^2 - 6*n*x - 9*n = 0) ∧ n ≥ 0 → n = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3949_394903


namespace NUMINAMATH_CALUDE_trig_expression_equality_l3949_394919

theorem trig_expression_equality : 
  2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l3949_394919


namespace NUMINAMATH_CALUDE_m_range_l3949_394958

def A : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}

theorem m_range (m : ℝ) (h : A ∪ B m = A) : 1/2 ≤ m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l3949_394958


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3949_394931

theorem equal_roots_quadratic (q : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + q = 0 ∧ 
   ∀ y : ℝ, y^2 - 3*y + q = 0 → y = x) ↔ 
  q = 9/4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3949_394931


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3949_394971

theorem geometric_sequence_third_term
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a = 3) -- first term is 3
  (h2 : a * r^4 = 243) -- fifth term is 243
  : a * r^2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3949_394971


namespace NUMINAMATH_CALUDE_max_candy_pieces_l3949_394902

theorem max_candy_pieces (n : ℕ) (mean : ℚ) (min_pieces : ℕ) 
  (h1 : n = 40)
  (h2 : mean = 4)
  (h3 : min_pieces = 2) :
  ∃ (max_pieces : ℕ), max_pieces = 82 ∧ 
  (∀ (student_pieces : List ℕ), 
    student_pieces.length = n ∧ 
    (∀ x ∈ student_pieces, x ≥ min_pieces) ∧
    (student_pieces.sum / n : ℚ) = mean →
    ∀ x ∈ student_pieces, x ≤ max_pieces) :=
by sorry

end NUMINAMATH_CALUDE_max_candy_pieces_l3949_394902


namespace NUMINAMATH_CALUDE_sector_arc_length_l3949_394964

/-- The length of an arc of a sector with given central angle and radius -/
def arcLength (centralAngle : Real) (radius : Real) : Real :=
  radius * centralAngle

theorem sector_arc_length :
  let centralAngle : Real := π / 5
  let radius : Real := 20
  arcLength centralAngle radius = 4 * π := by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l3949_394964


namespace NUMINAMATH_CALUDE_quadratic_prime_values_l3949_394921

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPolynomial (a b c : ℤ) : ℤ → ℤ := fun x ↦ a * x^2 + b * x + c

/-- Predicate to check if a number is prime -/
def IsPrime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, 1 < m → m < n → ¬(n % m = 0)

theorem quadratic_prime_values 
  (a b c : ℤ) (n : ℤ) :
  (IsPrime (QuadraticPolynomial a b c (n - 1))) →
  (IsPrime (QuadraticPolynomial a b c n)) →
  (IsPrime (QuadraticPolynomial a b c (n + 1))) →
  ∃ m : ℤ, m ≠ n - 1 ∧ m ≠ n ∧ m ≠ n + 1 ∧ IsPrime (QuadraticPolynomial a b c m) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_prime_values_l3949_394921


namespace NUMINAMATH_CALUDE_rohan_salary_l3949_394926

def food_expense : ℚ := 30 / 100
def rent_expense : ℚ := 20 / 100
def entertainment_expense : ℚ := 10 / 100
def conveyance_expense : ℚ := 5 / 100
def education_expense : ℚ := 10 / 100
def utilities_expense : ℚ := 10 / 100
def miscellaneous_expense : ℚ := 5 / 100
def savings_amount : ℕ := 2500

def total_expenses : ℚ :=
  food_expense + rent_expense + entertainment_expense + conveyance_expense +
  education_expense + utilities_expense + miscellaneous_expense

def savings_percentage : ℚ := 1 - total_expenses

theorem rohan_salary :
  ∃ (salary : ℕ), (↑savings_amount : ℚ) / (↑salary : ℚ) = savings_percentage ∧ salary = 25000 :=
by sorry

end NUMINAMATH_CALUDE_rohan_salary_l3949_394926


namespace NUMINAMATH_CALUDE_height_calculations_l3949_394938

-- Define the conversion rate
def inch_to_cm : ℝ := 2.54

-- Define heights in inches
def maria_height_inches : ℝ := 54
def samuel_height_inches : ℝ := 72

-- Define function to convert inches to centimeters
def inches_to_cm (inches : ℝ) : ℝ := inches * inch_to_cm

-- Theorem statement
theorem height_calculations :
  let maria_height_cm := inches_to_cm maria_height_inches
  let samuel_height_cm := inches_to_cm samuel_height_inches
  let height_difference := samuel_height_cm - maria_height_cm
  (maria_height_cm = 137.16) ∧
  (samuel_height_cm = 182.88) ∧
  (height_difference = 45.72) := by
  sorry

end NUMINAMATH_CALUDE_height_calculations_l3949_394938


namespace NUMINAMATH_CALUDE_alpha_plus_beta_is_75_degrees_l3949_394911

theorem alpha_plus_beta_is_75_degrees (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2)  -- α is acute
  (h2 : 0 < β ∧ β < π / 2)  -- β is acute
  (h3 : |Real.sin α - 1/2| + Real.sqrt (Real.tan β - 1) = 0) : 
  α + β = π / 2.4 := by  -- π/2.4 is equivalent to 75°
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_is_75_degrees_l3949_394911


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l3949_394999

theorem coefficient_x_cubed_in_expansion : 
  (Finset.range 37).sum (fun k => (Nat.choose 36 k) * (1 ^ (36 - k)) * (1 ^ k)) = 7140 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l3949_394999


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3949_394986

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3949_394986


namespace NUMINAMATH_CALUDE_tan_150_degrees_l3949_394930

theorem tan_150_degrees :
  Real.tan (150 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l3949_394930


namespace NUMINAMATH_CALUDE_carpool_commute_days_l3949_394979

/-- Proves that the number of commuting days per week is 5 given the carpool conditions --/
theorem carpool_commute_days : 
  let total_commute : ℝ := 21 -- miles one way
  let gas_cost : ℝ := 2.5 -- $/gallon
  let car_efficiency : ℝ := 30 -- miles/gallon
  let weeks_per_month : ℕ := 4
  let individual_payment : ℝ := 14 -- $ per month
  let num_friends : ℕ := 5
  
  -- Calculate the number of commuting days per week
  let commute_days : ℝ := 
    (individual_payment * num_friends) / 
    (gas_cost * (2 * total_commute / car_efficiency) * weeks_per_month)
  
  commute_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_carpool_commute_days_l3949_394979


namespace NUMINAMATH_CALUDE_solution_set_f_leq_x_plus_1_min_value_f_no_positive_a_b_l3949_394984

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem 1: Solution set of f(x) ≤ x + 1
theorem solution_set_f_leq_x_plus_1 :
  {x : ℝ | f x ≤ x + 1} = {x : ℝ | 2/3 ≤ x ∧ x ≤ 4} :=
sorry

-- Theorem 2: Minimum value of f(x)
theorem min_value_f :
  ∃ k : ℝ, k = 1 ∧ ∀ x : ℝ, f x ≥ k :=
sorry

-- Theorem 3: Non-existence of positive a, b satisfying conditions
theorem no_positive_a_b :
  ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*a + b = 1 ∧ 1/a + 2/b = 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_x_plus_1_min_value_f_no_positive_a_b_l3949_394984


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3949_394935

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + y^2) = f (x^2 - y^2) + f (2*x*y)

/-- The main theorem -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x, f x ≥ 0) → SatisfiesEquation f →
  ∃ a : ℝ, a ≥ 0 ∧ ∀ x, f x = a * x^2 :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3949_394935


namespace NUMINAMATH_CALUDE_impossible_11_difference_l3949_394945

/-- Represents an L-shaped piece -/
structure LPiece where
  cells : ℕ
  odd_cells : Odd cells

/-- Represents a partition of a square into L-shaped pieces -/
structure Partition where
  pieces : List LPiece
  total_cells : (pieces.map LPiece.cells).sum = 120 * 120

theorem impossible_11_difference (p1 p2 : Partition) : 
  p2.pieces.length ≠ p1.pieces.length + 11 := by
  sorry

end NUMINAMATH_CALUDE_impossible_11_difference_l3949_394945


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3949_394950

/-- A parabola with equation y = 2(x-3)^2 - 5 -/
def parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 - 5

/-- The axis of symmetry of the parabola -/
def axis_of_symmetry : ℝ := 3

/-- Theorem stating that the axis of symmetry of the given parabola is x = 3 -/
theorem parabola_axis_of_symmetry :
  ∀ x : ℝ, parabola (2 * axis_of_symmetry - x) = parabola x :=
sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3949_394950


namespace NUMINAMATH_CALUDE_num_distinct_lines_is_seven_l3949_394943

/-- A right triangle with two 45-degree angles at the base -/
structure RightIsoscelesTriangle where
  /-- The right angle of the triangle -/
  right_angle : Angle
  /-- One of the 45-degree angles at the base -/
  base_angle1 : Angle
  /-- The other 45-degree angle at the base -/
  base_angle2 : Angle
  /-- The right angle is 90 degrees -/
  right_angle_is_right : right_angle = 90
  /-- The base angles are each 45 degrees -/
  base_angles_are_45 : base_angle1 = 45 ∧ base_angle2 = 45

/-- The number of distinct lines formed by altitudes, medians, and angle bisectors -/
def num_distinct_lines (t : RightIsoscelesTriangle) : ℕ := sorry

/-- Theorem stating that the number of distinct lines is 7 -/
theorem num_distinct_lines_is_seven (t : RightIsoscelesTriangle) : 
  num_distinct_lines t = 7 := by sorry

end NUMINAMATH_CALUDE_num_distinct_lines_is_seven_l3949_394943


namespace NUMINAMATH_CALUDE_sum_of_digits_2008_5009_7_l3949_394991

theorem sum_of_digits_2008_5009_7 :
  ∃ (n : ℕ), n = 2^2008 * 5^2009 * 7 ∧ (List.sum (Nat.digits 10 n) = 5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_2008_5009_7_l3949_394991


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_reverse_composite_l3949_394946

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def reverse_digits (n : ℕ) : ℕ :=
  if n < 10 then n else
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

theorem smallest_two_digit_prime_reverse_composite :
  ∃ p : ℕ, 
    p ≥ 10 ∧ p < 100 ∧  -- two-digit number
    is_prime p ∧
    is_composite (reverse_digits p) ∧
    p / 10 ≤ 3 ∧  -- starts with a digit less than or equal to 3
    (∀ q : ℕ, q ≥ 10 ∧ q < p ∧ is_prime q ∧ q / 10 ≤ 3 → ¬(is_composite (reverse_digits q))) ∧
    p = 23 :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_reverse_composite_l3949_394946


namespace NUMINAMATH_CALUDE_square_sum_product_l3949_394988

theorem square_sum_product (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) :
  a^2 + b^2 + a * b = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_l3949_394988


namespace NUMINAMATH_CALUDE_fundraiser_theorem_l3949_394972

/-- Represents the fundraiser problem --/
def fundraiser_problem (goal : ℕ) 
  (chocolate_price oatmeal_price sugar_price : ℕ)
  (chocolate_sold oatmeal_sold sugar_sold : ℕ) : Prop :=
  let current_profit := 
    chocolate_price * chocolate_sold + 
    oatmeal_price * oatmeal_sold + 
    sugar_price * sugar_sold
  goal - current_profit = 110

/-- The fundraiser theorem --/
theorem fundraiser_theorem : 
  fundraiser_problem 250 6 5 4 5 10 15 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_theorem_l3949_394972


namespace NUMINAMATH_CALUDE_descending_order_l3949_394940

-- Define the numbers in their respective bases
def a : ℕ := 3 * 16 + 14
def b : ℕ := 2 * 6^2 + 1 * 6 + 0
def c : ℕ := 1 * 4^3 + 0 * 4^2 + 0 * 4 + 0
def d : ℕ := 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2 + 1

-- Theorem statement
theorem descending_order : b > c ∧ c > a ∧ a > d := by
  sorry

end NUMINAMATH_CALUDE_descending_order_l3949_394940


namespace NUMINAMATH_CALUDE_min_sum_xyz_l3949_394966

theorem min_sum_xyz (x y z : ℤ) (h : (x - 10) * (y - 5) * (z - 2) = 1000) :
  ∀ a b c : ℤ, (a - 10) * (b - 5) * (c - 2) = 1000 → x + y + z ≤ a + b + c ∧ x + y + z = 92 :=
sorry

end NUMINAMATH_CALUDE_min_sum_xyz_l3949_394966


namespace NUMINAMATH_CALUDE_prob_two_changing_yao_l3949_394989

/-- The number of "yao" in one "gua" -/
def num_yao : ℕ := 6

/-- The probability of a coin facing heads up -/
def p_heads : ℚ := 1/2

/-- The probability of a "changing yao" -/
def p_changing_yao : ℚ := 2 * p_heads^3

/-- The probability of a non-changing yao -/
def p_non_changing_yao : ℚ := 1 - p_changing_yao

/-- The probability of having exactly two "changing yao" in one "gua" -/
theorem prob_two_changing_yao : 
  (Nat.choose num_yao 2 : ℚ) * p_changing_yao^2 * p_non_changing_yao^(num_yao - 2) = 1215/4096 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_changing_yao_l3949_394989


namespace NUMINAMATH_CALUDE_children_catered_count_l3949_394929

/-- Represents the number of children that can be catered with remaining food --/
def children_catered (total_adults : ℕ) (total_children : ℕ) (adults_meal_capacity : ℕ) (children_meal_capacity : ℕ) (adults_eaten : ℕ) (adult_child_consumption_ratio : ℚ) (adult_diet_restriction_percent : ℚ) (child_diet_restriction_percent : ℚ) : ℕ :=
  sorry

/-- Theorem stating the number of children that can be catered under given conditions --/
theorem children_catered_count : 
  children_catered 55 70 70 90 21 (3/2) (1/5) (3/20) = 63 :=
sorry

end NUMINAMATH_CALUDE_children_catered_count_l3949_394929


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_l3949_394982

theorem max_consecutive_integers_sum (n : ℕ) : n = 31 ↔ 
  (n ≥ 3 ∧ 
   (∀ k : ℕ, k ≥ 3 → k ≤ n → (k * (k + 1)) / 2 - 3 ≤ 500) ∧
   (∀ m : ℕ, m > n → (m * (m + 1)) / 2 - 3 > 500)) :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_l3949_394982


namespace NUMINAMATH_CALUDE_bd_production_l3949_394957

/-- Represents the total production of all workshops -/
def total_production : ℕ := 2800

/-- Represents the total number of units sampled for quality inspection -/
def total_sampled : ℕ := 140

/-- Represents the number of units sampled from workshops A and C combined -/
def ac_sampled : ℕ := 60

/-- Theorem stating that the total production from workshops B and D is 1600 units -/
theorem bd_production : 
  total_production - (ac_sampled * (total_production / total_sampled)) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_bd_production_l3949_394957


namespace NUMINAMATH_CALUDE_money_left_over_correct_l3949_394905

/-- Calculates the money left over after purchases given the specified conditions --/
def money_left_over (
  video_game_cost : ℚ)
  (video_game_discount : ℚ)
  (candy_cost : ℚ)
  (sales_tax : ℚ)
  (shipping_fee : ℚ)
  (babysitting_rate : ℚ)
  (bonus_rate : ℚ)
  (hours_worked : ℕ)
  (bonus_threshold : ℕ) : ℚ :=
  let discounted_game_cost := video_game_cost * (1 - video_game_discount)
  let total_before_tax := discounted_game_cost + shipping_fee + candy_cost
  let total_cost := total_before_tax * (1 + sales_tax)
  let regular_hours := min hours_worked bonus_threshold
  let bonus_hours := hours_worked - regular_hours
  let total_earnings := babysitting_rate * hours_worked + bonus_rate * bonus_hours
  total_earnings - total_cost

theorem money_left_over_correct :
  money_left_over 60 0.15 5 0.10 3 8 2 9 5 = 151/10 := by
  sorry

end NUMINAMATH_CALUDE_money_left_over_correct_l3949_394905


namespace NUMINAMATH_CALUDE_xyz_inequalities_l3949_394927

theorem xyz_inequalities (x y z : ℝ) 
  (h1 : x < y) (h2 : y < z) 
  (h3 : x + y + z = 6) 
  (h4 : x*y + y*z + z*x = 9) : 
  0 < x ∧ x < 1 ∧ 1 < y ∧ y < 3 ∧ 3 < z ∧ z < 4 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequalities_l3949_394927


namespace NUMINAMATH_CALUDE_tournament_handshakes_correct_l3949_394916

/-- The number of handshakes in a tennis tournament with 3 teams of 2 players each --/
def tournament_handshakes : ℕ := 12

/-- The number of teams in the tournament --/
def num_teams : ℕ := 3

/-- The number of players per team --/
def players_per_team : ℕ := 2

/-- The total number of players in the tournament --/
def total_players : ℕ := num_teams * players_per_team

/-- The number of handshakes per player --/
def handshakes_per_player : ℕ := total_players - 2

theorem tournament_handshakes_correct :
  tournament_handshakes = (total_players * handshakes_per_player) / 2 :=
by sorry

end NUMINAMATH_CALUDE_tournament_handshakes_correct_l3949_394916


namespace NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_l3949_394901

theorem sum_of_integers_ending_in_3 :
  let first_term : ℕ := 103
  let last_term : ℕ := 493
  let common_difference : ℕ := 10
  let n : ℕ := (last_term - first_term) / common_difference + 1
  (n : ℤ) * (first_term + last_term) / 2 = 11920 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_l3949_394901


namespace NUMINAMATH_CALUDE_max_min_values_on_interval_l3949_394962

def f (x : ℝ) := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max = 5 ∧ min = -15 := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_on_interval_l3949_394962


namespace NUMINAMATH_CALUDE_total_gumballs_l3949_394954

def gumball_problem (total gumballs_todd gumballs_alisha gumballs_bobby remaining : ℕ) : Prop :=
  gumballs_todd = 4 ∧
  gumballs_alisha = 2 * gumballs_todd ∧
  gumballs_bobby = 4 * gumballs_alisha - 5 ∧
  total = gumballs_todd + gumballs_alisha + gumballs_bobby + remaining ∧
  remaining = 6

theorem total_gumballs : ∃ total : ℕ, gumball_problem total 4 8 27 6 ∧ total = 45 :=
  sorry

end NUMINAMATH_CALUDE_total_gumballs_l3949_394954


namespace NUMINAMATH_CALUDE_mn_sum_for_5000_l3949_394952

theorem mn_sum_for_5000 (m n : ℕ+) : 
  m * n = 5000 →
  ¬(10 ∣ m) →
  ¬(10 ∣ n) →
  m + n = 633 := by
sorry

end NUMINAMATH_CALUDE_mn_sum_for_5000_l3949_394952


namespace NUMINAMATH_CALUDE_two_year_compound_interest_l3949_394936

/-- Calculates the final amount after two years of compound interest with variable rates -/
def final_amount (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  initial * (1 + rate1) * (1 + rate2)

/-- Theorem stating that given the specific initial amount and interest rates, 
    the final amount after two years is 82432 -/
theorem two_year_compound_interest :
  final_amount 64000 0.12 0.15 = 82432 := by
  sorry

#eval final_amount 64000 0.12 0.15

end NUMINAMATH_CALUDE_two_year_compound_interest_l3949_394936


namespace NUMINAMATH_CALUDE_restaurant_order_combinations_l3949_394922

/-- The number of items on the menu -/
def menu_items : ℕ := 15

/-- The number of people ordering -/
def num_people : ℕ := 3

/-- The number of specialty dishes -/
def specialty_dishes : ℕ := 3

/-- The number of different meal combinations -/
def meal_combinations : ℕ := 1611

theorem restaurant_order_combinations :
  (menu_items ^ num_people) - (num_people * (specialty_dishes * (menu_items - specialty_dishes) ^ (num_people - 1))) = meal_combinations := by
  sorry

end NUMINAMATH_CALUDE_restaurant_order_combinations_l3949_394922


namespace NUMINAMATH_CALUDE_rhombus_square_diagonals_l3949_394996

-- Define a rhombus
structure Rhombus :=
  (sides_equal : ∀ s1 s2 : ℝ, s1 = s2)
  (diagonals_perpendicular : Bool)

-- Define a square as a special case of rhombus
structure Square extends Rhombus :=
  (all_angles_right : Bool)

-- Theorem statement
theorem rhombus_square_diagonals :
  ∃ (r : Rhombus), ¬(∀ d1 d2 : ℝ, d1 = d2) ∧
  ∀ (s : Square), ∀ d1 d2 : ℝ, d1 = d2 :=
sorry

end NUMINAMATH_CALUDE_rhombus_square_diagonals_l3949_394996


namespace NUMINAMATH_CALUDE_cubic_inequality_l3949_394918

theorem cubic_inequality (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  3 * a^3 + 2 * b^3 ≥ 3 * a^2 * b + 2 * a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3949_394918


namespace NUMINAMATH_CALUDE_f_increasing_l3949_394978

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x - Real.sin x else x^3 + 1

theorem f_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_f_increasing_l3949_394978


namespace NUMINAMATH_CALUDE_point_order_on_parabola_l3949_394949

/-- Parabola equation y = (x-1)^2 - 2 -/
def parabola (x y : ℝ) : Prop := y = (x - 1)^2 - 2

theorem point_order_on_parabola (a b c d : ℝ) :
  parabola a 2 →
  parabola b 6 →
  parabola c d →
  d < 1 →
  a < 0 →
  b > 0 →
  a < c ∧ c < b :=
sorry

end NUMINAMATH_CALUDE_point_order_on_parabola_l3949_394949


namespace NUMINAMATH_CALUDE_initial_bottles_count_l3949_394995

/-- The number of bottles Jason buys -/
def jason_bottles : ℕ := 5

/-- The number of bottles Harry buys -/
def harry_bottles : ℕ := 6

/-- The number of bottles left on the shelf after purchases -/
def remaining_bottles : ℕ := 24

/-- The initial number of bottles on the shelf -/
def initial_bottles : ℕ := jason_bottles + harry_bottles + remaining_bottles

theorem initial_bottles_count : initial_bottles = 35 := by
  sorry

end NUMINAMATH_CALUDE_initial_bottles_count_l3949_394995


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_c_eq_five_l3949_394924

/-- The equation has infinitely many solutions if and only if c = 5 -/
theorem infinite_solutions_iff_c_eq_five :
  (∃ c : ℝ, ∀ x : ℝ, 3 * (5 + c * x) = 15 * x + 15) ↔ c = 5 :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_c_eq_five_l3949_394924


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l3949_394955

theorem min_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 ∧ 
  (∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 10 → a + b ≤ c + d) ∧
  a + b = 45 :=
sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l3949_394955


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l3949_394983

theorem factorial_equation_solutions :
  ∀ x y z : ℕ+,
    (2 ^ x.val + 3 ^ y.val - 7 = Nat.factorial z.val) ↔ 
    ((x, y, z) = (2, 2, 3) ∨ (x, y, z) = (2, 3, 4)) := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l3949_394983


namespace NUMINAMATH_CALUDE_recurring_larger_than_finite_l3949_394980

def recurring_decimal : ℚ := 1 + 3/10 + 5/100 + 42/10000 + 5/1000 * (1/9)
def finite_decimal : ℚ := 1 + 3/10 + 5/100 + 4/1000 + 2/10000

theorem recurring_larger_than_finite : recurring_decimal > finite_decimal := by
  sorry

end NUMINAMATH_CALUDE_recurring_larger_than_finite_l3949_394980


namespace NUMINAMATH_CALUDE_product_of_x_and_y_l3949_394961

theorem product_of_x_and_y (x y : ℝ) (h1 : 3 * x + 4 * y = 60) (h2 : 6 * x - 4 * y = 12) :
  x * y = 72 := by
  sorry

end NUMINAMATH_CALUDE_product_of_x_and_y_l3949_394961


namespace NUMINAMATH_CALUDE_smallest_multiple_of_twelve_power_l3949_394900

theorem smallest_multiple_of_twelve_power (k : ℕ) : 
  (3^k - k^3 = 1) → (∀ n : ℕ, n > 0 ∧ 12^k ∣ n → n ≥ 144) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_twelve_power_l3949_394900


namespace NUMINAMATH_CALUDE_tens_digit_of_9_pow_2023_l3949_394969

theorem tens_digit_of_9_pow_2023 : ∃ n : ℕ, 9^2023 ≡ 80 + n [ZMOD 100] ∧ 0 ≤ n ∧ n < 10 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_9_pow_2023_l3949_394969


namespace NUMINAMATH_CALUDE_gcd_102_238_l3949_394992

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l3949_394992


namespace NUMINAMATH_CALUDE_find_divisor_find_divisor_proof_l3949_394953

theorem find_divisor (original : Nat) (subtracted : Nat) (divisor : Nat) : Prop :=
  let remaining := original - subtracted
  (original = 1387) →
  (subtracted = 7) →
  (remaining % divisor = 0) →
  (∀ d : Nat, d > divisor → remaining % d ≠ 0 ∨ (original - d) % d ≠ 0) →
  divisor = 23

-- The proof would go here
theorem find_divisor_proof : find_divisor 1387 7 23 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_find_divisor_proof_l3949_394953


namespace NUMINAMATH_CALUDE_lcm_of_eight_numbers_l3949_394973

theorem lcm_of_eight_numbers : Nat.lcm 8 (Nat.lcm 24 (Nat.lcm 36 (Nat.lcm 54 (Nat.lcm 42 (Nat.lcm 51 (Nat.lcm 64 87)))))) = 5963328 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_eight_numbers_l3949_394973


namespace NUMINAMATH_CALUDE_janessa_keeps_twenty_cards_l3949_394923

/-- The number of cards Janessa keeps for herself given the initial conditions -/
def janessas_kept_cards (initial_cards : ℕ) (father_cards : ℕ) (ebay_cards : ℕ) (bad_cards : ℕ) (given_cards : ℕ) : ℕ :=
  initial_cards + father_cards + ebay_cards - bad_cards - given_cards

/-- Theorem stating that Janessa keeps 20 cards for herself -/
theorem janessa_keeps_twenty_cards :
  janessas_kept_cards 4 13 36 4 29 = 20 := by sorry

end NUMINAMATH_CALUDE_janessa_keeps_twenty_cards_l3949_394923


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3949_394913

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (4 + Real.sqrt x) = 4 → x = 144 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3949_394913


namespace NUMINAMATH_CALUDE_james_quiz_bowl_points_l3949_394937

/-- Calculates the total points earned by a student in a quiz bowl game. -/
def quiz_bowl_points (total_rounds : ℕ) (questions_per_round : ℕ) (points_per_correct : ℕ) 
  (bonus_points : ℕ) (questions_missed : ℕ) : ℕ :=
  let total_questions := total_rounds * questions_per_round
  let correct_answers := total_questions - questions_missed
  let base_points := correct_answers * points_per_correct
  let full_rounds := total_rounds - (questions_missed + questions_per_round - 1) / questions_per_round
  let bonus_total := full_rounds * bonus_points
  base_points + bonus_total

/-- Theorem stating that James earned 64 points in the quiz bowl game. -/
theorem james_quiz_bowl_points : 
  quiz_bowl_points 5 5 2 4 1 = 64 := by
  sorry

end NUMINAMATH_CALUDE_james_quiz_bowl_points_l3949_394937


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3949_394997

theorem right_triangle_segment_ratio (a b c r s : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ s > 0 →
  a^2 + b^2 = c^2 →
  r + s = c →
  a^2 = r * c →
  b^2 = s * c →
  a / b = 2 / 5 →
  r / s = 4 / 25 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3949_394997


namespace NUMINAMATH_CALUDE_poly_arrangement_l3949_394920

/-- The original polynomial -/
def original_poly (x y : ℝ) : ℝ := -2 * x^3 * y + 4 * x * y^3 + 1 - 3 * x^2 * y^2

/-- The polynomial arranged in descending order of y -/
def arranged_poly (x y : ℝ) : ℝ := 4 * x * y^3 - 3 * x^2 * y^2 - 2 * x^3 * y + 1

/-- Theorem stating that the original polynomial is equal to the arranged polynomial -/
theorem poly_arrangement (x y : ℝ) : original_poly x y = arranged_poly x y := by
  sorry

end NUMINAMATH_CALUDE_poly_arrangement_l3949_394920


namespace NUMINAMATH_CALUDE_room_width_calculation_l3949_394947

/-- Proves that given a rectangular room with a length of 5.5 meters, where the cost of paving the floor at a rate of 800 Rs/m² is 17,600 Rs, the width of the room is 4 meters. -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) :
  length = 5.5 →
  total_cost = 17600 →
  cost_per_sqm = 800 →
  total_cost / cost_per_sqm / length = 4 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l3949_394947


namespace NUMINAMATH_CALUDE_disjoint_subsets_remainder_l3949_394994

def T : Finset Nat := Finset.range 15

def disjoint_subsets (S : Finset Nat) : Nat :=
  (3^S.card - 2 * 2^S.card + 1) / 2

theorem disjoint_subsets_remainder (S : Finset Nat) (h : S = T) : 
  disjoint_subsets S % 500 = 186 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subsets_remainder_l3949_394994


namespace NUMINAMATH_CALUDE_stating_max_tulips_in_bouquet_l3949_394925

/-- Represents the cost of a yellow tulip in rubles -/
def yellow_cost : ℕ := 50

/-- Represents the cost of a red tulip in rubles -/
def red_cost : ℕ := 31

/-- Represents the maximum budget in rubles -/
def max_budget : ℕ := 600

/-- 
Theorem stating that the maximum number of tulips in a bouquet is 15,
given the specified conditions
-/
theorem max_tulips_in_bouquet :
  ∃ (y r : ℕ),
    -- The total number of tulips is odd
    Odd (y + r) ∧
    -- The difference between yellow and red tulips is exactly 1
    (y = r + 1 ∨ r = y + 1) ∧
    -- The total cost does not exceed the budget
    y * yellow_cost + r * red_cost ≤ max_budget ∧
    -- The total number of tulips is 15
    y + r = 15 ∧
    -- This is the maximum possible number of tulips
    ∀ (y' r' : ℕ),
      Odd (y' + r') →
      (y' = r' + 1 ∨ r' = y' + 1) →
      y' * yellow_cost + r' * red_cost ≤ max_budget →
      y' + r' ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_stating_max_tulips_in_bouquet_l3949_394925


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l3949_394944

/-- Given 5 moles of a compound with a total molecular weight of 1170,
    prove that the molecular weight of 1 mole of the compound is 234. -/
theorem molecular_weight_calculation (total_weight : ℝ) (num_moles : ℝ) :
  total_weight = 1170 →
  num_moles = 5 →
  total_weight / num_moles = 234 := by
sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l3949_394944


namespace NUMINAMATH_CALUDE_incorrect_statement_l3949_394932

-- Define the concept of planes
variable (α β : Set (ℝ × ℝ × ℝ))

-- Define perpendicularity between planes
def perpendicular (p q : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the concept of a line
def Line : Type := Set (ℝ × ℝ × ℝ)

-- Define perpendicularity between a line and a plane
def line_perp_plane (l : Line) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the intersection line of two planes
def intersection_line (p q : Set (ℝ × ℝ × ℝ)) : Line := sorry

-- Define a function to create a perpendicular line from a point to a line
def perp_line_to_line (point : ℝ × ℝ × ℝ) (l : Line) : Line := sorry

-- Theorem to be disproved
theorem incorrect_statement 
  (h1 : perpendicular α β)
  (point : ℝ × ℝ × ℝ)
  (h2 : point ∈ α) :
  line_perp_plane (perp_line_to_line point (intersection_line α β)) β := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l3949_394932


namespace NUMINAMATH_CALUDE_hyperbola_tangent_angle_bisector_parabola_tangent_angle_bisector_l3949_394987

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  f1 : Point2D  -- First focus
  f2 : Point2D  -- Second focus
  a : ℝ         -- Distance from center to vertex

/-- Represents a parabola -/
structure Parabola where
  f : Point2D    -- Focus
  directrix : Line2D

/-- Returns the angle bisector of three points -/
def angleBisector (p1 p2 p3 : Point2D) : Line2D :=
  sorry

/-- Returns the tangent line to a hyperbola at a given point -/
def hyperbolaTangent (h : Hyperbola) (p : Point2D) : Line2D :=
  sorry

/-- Returns the tangent line to a parabola at a given point -/
def parabolaTangent (p : Parabola) (pt : Point2D) : Line2D :=
  sorry

/-- Theorem: The angle bisector property holds for hyperbola tangents -/
theorem hyperbola_tangent_angle_bisector (h : Hyperbola) (p : Point2D) :
  hyperbolaTangent h p = angleBisector h.f1 p h.f2 :=
sorry

/-- Theorem: The angle bisector property holds for parabola tangents -/
theorem parabola_tangent_angle_bisector (p : Parabola) (pt : Point2D) :
  parabolaTangent p pt = angleBisector p.f pt (Point2D.mk 0 0) :=  -- Assuming (0,0) is on the directrix
sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_angle_bisector_parabola_tangent_angle_bisector_l3949_394987


namespace NUMINAMATH_CALUDE_quadratic_solution_properties_l3949_394948

theorem quadratic_solution_properties :
  ∀ (y₁ y₂ : ℝ), y₁^2 - 1500*y₁ + 750 = 0 ∧ y₂^2 - 1500*y₂ + 750 = 0 →
  y₁ + y₂ = 1500 ∧ y₁ * y₂ = 750 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_properties_l3949_394948


namespace NUMINAMATH_CALUDE_julias_preferred_number_l3949_394977

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem julias_preferred_number :
  ∃! n : ℕ,
    100 < n ∧ n < 200 ∧
    is_multiple n 13 ∧
    ¬ is_multiple n 3 ∧
    is_multiple (digit_sum n) 5 ∧
    n = 104 := by
  sorry

end NUMINAMATH_CALUDE_julias_preferred_number_l3949_394977


namespace NUMINAMATH_CALUDE_point_M_coordinates_l3949_394912

-- Define the curve
def curve (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Theorem statement
theorem point_M_coordinates :
  ∀ x y : ℝ,
  y = curve x →
  curve_derivative x = -4 →
  (x = -1 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l3949_394912


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3949_394993

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3949_394993


namespace NUMINAMATH_CALUDE_water_tower_problem_l3949_394981

theorem water_tower_problem (total_capacity : ℕ) (n1 n2 n3 n4 n5 : ℕ) :
  total_capacity = 2700 →
  n1 = 300 →
  n2 = 2 * n1 →
  n3 = n2 + 100 →
  n4 = 3 * n1 →
  n5 = n3 / 2 →
  n1 + n2 + n3 + n4 + n5 > total_capacity :=
by sorry

end NUMINAMATH_CALUDE_water_tower_problem_l3949_394981


namespace NUMINAMATH_CALUDE_cubic_function_constraint_l3949_394976

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*(a+2)

/-- f has both a maximum and a minimum value -/
def has_max_and_min (a : ℝ) : Prop := ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f' a x₁ = 0 ∧ f' a x₂ = 0

theorem cubic_function_constraint (a : ℝ) : 
  has_max_and_min a → a < -1 ∨ a > 2 := by sorry

end NUMINAMATH_CALUDE_cubic_function_constraint_l3949_394976


namespace NUMINAMATH_CALUDE_original_price_calculation_l3949_394941

theorem original_price_calculation (initial_price : ℚ) : 
  (initial_price * (1 + 10/100) * (1 - 20/100) = 2) → initial_price = 25/11 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l3949_394941


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3949_394975

theorem possible_values_of_a : 
  {a : ℤ | ∃ b c : ℤ, ∀ x : ℝ, (x - a) * (x - 12) + 1 = (x + b) * (x + c)} = {10, 14} := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3949_394975


namespace NUMINAMATH_CALUDE_new_eurasian_bridge_length_scientific_notation_l3949_394967

theorem new_eurasian_bridge_length_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 10900 = a * (10 : ℝ) ^ n ∧ a = 1.09 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_new_eurasian_bridge_length_scientific_notation_l3949_394967


namespace NUMINAMATH_CALUDE_only_real_number_line_bijection_is_correct_l3949_394915

-- Define the property of having a square root
def has_square_root (x : ℝ) : Prop := ∃ y : ℝ, y * y = x

-- Define the property of being irrational
def is_irrational (x : ℝ) : Prop := ¬ (∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

-- Define the property of cube root being equal to itself
def cube_root_equals_self (x : ℝ) : Prop := x * x * x = x

-- Define the property of having no square root
def has_no_square_root (x : ℝ) : Prop := ¬ (∃ y : ℝ, y * y = x)

-- Define the one-to-one correspondence between real numbers and points on a line
def real_number_line_bijection : Prop := 
  ∃ f : ℝ → ℝ, Function.Bijective f ∧ (∀ x : ℝ, f x = x)

-- Define the property that the difference of two irrationals is irrational
def irrational_diff_is_irrational : Prop := 
  ∀ x y : ℝ, is_irrational x → is_irrational y → is_irrational (x - y)

theorem only_real_number_line_bijection_is_correct : 
  (¬ (∀ x : ℝ, has_square_root x → is_irrational x)) ∧
  (¬ (∀ x : ℝ, cube_root_equals_self x → (x = 0 ∨ x = 1))) ∧
  (¬ (∀ a : ℝ, has_no_square_root (-a))) ∧
  real_number_line_bijection ∧
  (¬ irrational_diff_is_irrational) :=
by sorry

end NUMINAMATH_CALUDE_only_real_number_line_bijection_is_correct_l3949_394915


namespace NUMINAMATH_CALUDE_absolute_value_equation_one_root_l3949_394917

theorem absolute_value_equation_one_root :
  ∃! x : ℝ, (abs x - 4 / x = 3 * abs x / x) ∧ (x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_one_root_l3949_394917


namespace NUMINAMATH_CALUDE_square_area_to_side_length_ratio_l3949_394998

theorem square_area_to_side_length_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (a^2 / b^2 = 72 / 98) → (a / b = 6 / 7) := by
  sorry

end NUMINAMATH_CALUDE_square_area_to_side_length_ratio_l3949_394998


namespace NUMINAMATH_CALUDE_sin_graph_shift_l3949_394963

open Real

theorem sin_graph_shift (f g : ℝ → ℝ) (ω : ℝ) (h : ω = 2) :
  (∀ x, f x = sin (ω * x + π / 6)) →
  (∀ x, g x = sin (ω * x)) →
  ∃ shift, ∀ x, f x = g (x - shift) ∧ shift = π / 12 :=
sorry

end NUMINAMATH_CALUDE_sin_graph_shift_l3949_394963


namespace NUMINAMATH_CALUDE_certain_number_problem_l3949_394910

theorem certain_number_problem (x : ℤ) : 34 + x - 53 = 28 → x = 47 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3949_394910


namespace NUMINAMATH_CALUDE_probability_defect_free_l3949_394909

/-- Represents the proportion of components from Company A in the warehouse -/
def proportion_A : ℝ := 0.60

/-- Represents the proportion of components from Company B in the warehouse -/
def proportion_B : ℝ := 0.40

/-- Represents the defect rate of components from Company A -/
def defect_rate_A : ℝ := 0.98

/-- Represents the defect rate of components from Company B -/
def defect_rate_B : ℝ := 0.95

/-- Theorem stating that the probability of a randomly selected component being defect-free is 0.032 -/
theorem probability_defect_free : 
  proportion_A * (1 - defect_rate_A) + proportion_B * (1 - defect_rate_B) = 0.032 := by
  sorry


end NUMINAMATH_CALUDE_probability_defect_free_l3949_394909


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3949_394939

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Icc (-1/2 : ℝ) (-1/3) = {x | a * x^2 - b * x - 1 ≥ 0}) :
  {x : ℝ | x^2 - b*x - a < 0} = Set.Ioo 2 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3949_394939


namespace NUMINAMATH_CALUDE_football_progress_l3949_394985

/-- Calculates the overall progress in meters for a football team given their yard changes and the yard-to-meter conversion rate. -/
theorem football_progress (yard_to_meter : ℝ) (play1 play2 penalty play3 play4 : ℝ) :
  yard_to_meter = 0.9144 →
  play1 = -15 →
  play2 = 20 →
  penalty = -10 →
  play3 = 25 →
  play4 = -5 →
  (play1 + play2 + penalty + play3 + play4) * yard_to_meter = 13.716 := by
  sorry

end NUMINAMATH_CALUDE_football_progress_l3949_394985


namespace NUMINAMATH_CALUDE_largest_x_and_fraction_l3949_394974

theorem largest_x_and_fraction (x : ℝ) (a b c d : ℤ) : 
  (7 * x / 5 - 2 = 4 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (∀ y : ℝ, (7 * y / 5 - 2 = 4 / y) → y ≤ x) →
  (x = (5 + 5 * Real.sqrt 66) / 7 ∧ a * c * d / b = 462) := by
  sorry

end NUMINAMATH_CALUDE_largest_x_and_fraction_l3949_394974


namespace NUMINAMATH_CALUDE_sequence_equality_l3949_394956

-- Define the sequence S
def S (n : ℕ) : ℕ := (4 * n - 3)^2

-- Define the proposed form of S
def S_proposed (n : ℕ) (a b : ℤ) : ℤ := (4 * n - 3) * (a * n + b)

-- Theorem statement
theorem sequence_equality (a b : ℤ) :
  (∀ n : ℕ, n > 0 → S n = S_proposed n a b) →
  a^2 + b^2 = 25 :=
sorry

end NUMINAMATH_CALUDE_sequence_equality_l3949_394956


namespace NUMINAMATH_CALUDE_no_balloons_remain_intact_l3949_394951

/-- Represents the state of balloons in a hot air balloon --/
structure BalloonState where
  total : ℕ
  intact : ℕ
  doubleDurable : ℕ

/-- Calculates the number of intact balloons after the first 30 minutes --/
def afterFirstHalfHour (initial : ℕ) : ℕ :=
  initial - (initial / 5)

/-- Calculates the number of intact balloons after the next hour --/
def afterNextHour (intact : ℕ) : ℕ :=
  intact - (intact * 3 / 10)

/-- Calculates the number of double durable balloons --/
def doubleDurableBalloons (intact : ℕ) : ℕ :=
  intact / 10

/-- Calculates the final number of intact balloons --/
def finalIntactBalloons (state : BalloonState) : ℕ :=
  let nonDurableBlownUp := state.total - state.intact
  let toBlowUp := min (2 * (nonDurableBlownUp - state.doubleDurable)) state.intact
  state.intact - toBlowUp

/-- Main theorem: After all events, no balloons remain intact --/
theorem no_balloons_remain_intact (initialBalloons : ℕ) 
    (h1 : initialBalloons = 200) : 
    finalIntactBalloons 
      { total := initialBalloons,
        intact := afterNextHour (afterFirstHalfHour initialBalloons),
        doubleDurable := doubleDurableBalloons (afterNextHour (afterFirstHalfHour initialBalloons)) } = 0 := by
  sorry

#eval finalIntactBalloons 
  { total := 200,
    intact := afterNextHour (afterFirstHalfHour 200),
    doubleDurable := doubleDurableBalloons (afterNextHour (afterFirstHalfHour 200)) }

end NUMINAMATH_CALUDE_no_balloons_remain_intact_l3949_394951


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l3949_394906

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 30)
  (area2 : w * h = 40)
  (area3 : l * h = 12) :
  l * w * h = 120 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l3949_394906


namespace NUMINAMATH_CALUDE_factorization_x4_minus_64_l3949_394970

theorem factorization_x4_minus_64 (x : ℝ) : 
  x^4 - 64 = (x^2 + 8) * (x + 2 * Real.sqrt 2) * (x - 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_64_l3949_394970


namespace NUMINAMATH_CALUDE_combination_count_l3949_394907

theorem combination_count (n k m : ℕ) :
  (∃ (s : Finset (Finset ℕ)),
    (∀ t ∈ s, t.card = k ∧
      (∀ j ∈ t, 1 ≤ j ∧ j ≤ n) ∧
      (∀ (i j : ℕ), i ∈ t → j ∈ t → i < j → m ≤ j - i) ∧
      (∀ (i j : ℕ), i ∈ t → j ∈ t → i ≠ j → i < j)) ∧
    s.card = Nat.choose (n - (k - 1) * (m - 1)) k) :=
by sorry

end NUMINAMATH_CALUDE_combination_count_l3949_394907


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3949_394928

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  2 / a + 3 / b ≥ 5 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3949_394928


namespace NUMINAMATH_CALUDE_problem_solution_l3949_394968

theorem problem_solution (a b c d : ℕ+) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : a * b + b * c + a * c = a * b * c)
  (h5 : a * b * c = d) : d = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3949_394968


namespace NUMINAMATH_CALUDE_admission_fees_proof_l3949_394960

-- Define the given conditions
def child_fee : ℚ := 1.5
def adult_fee : ℚ := 4
def total_people : ℕ := 315
def num_children : ℕ := 180

-- Define the function to calculate total admission fees
def total_admission_fees : ℚ :=
  (child_fee * num_children) + (adult_fee * (total_people - num_children))

-- Theorem to prove
theorem admission_fees_proof : total_admission_fees = 810 := by
  sorry

end NUMINAMATH_CALUDE_admission_fees_proof_l3949_394960


namespace NUMINAMATH_CALUDE_grade2_sample_count_l3949_394904

/-- Represents the number of students in a grade -/
def GradeCount := ℕ

/-- Represents a ratio of students across three grades -/
structure GradeRatio :=
  (grade1 : ℕ)
  (grade2 : ℕ)
  (grade3 : ℕ)

/-- Calculates the number of students in a stratified sample for a specific grade -/
def stratifiedSampleCount (totalSample : ℕ) (ratio : GradeRatio) (gradeRatio : ℕ) : ℕ :=
  (totalSample * gradeRatio) / (ratio.grade1 + ratio.grade2 + ratio.grade3)

/-- Theorem stating the number of Grade 2 students in the stratified sample -/
theorem grade2_sample_count 
  (totalSample : ℕ) 
  (ratio : GradeRatio) 
  (h1 : totalSample = 240) 
  (h2 : ratio = GradeRatio.mk 5 4 3) : 
  stratifiedSampleCount totalSample ratio ratio.grade2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_grade2_sample_count_l3949_394904


namespace NUMINAMATH_CALUDE_tetrahedron_non_coplanar_selections_l3949_394965

/-- The number of ways to select 4 non-coplanar points from a tetrahedron -/
def nonCoplanarSelections : ℕ := 141

/-- Total number of points on the tetrahedron -/
def totalPoints : ℕ := 10

/-- Number of vertices of the tetrahedron -/
def vertices : ℕ := 4

/-- Number of midpoints of edges -/
def midpoints : ℕ := 6

/-- Number of points to be selected -/
def selectPoints : ℕ := 4

/-- Theorem stating that the number of ways to select 4 non-coplanar points
    from 10 points on a tetrahedron (4 vertices and 6 midpoints of edges) is 141 -/
theorem tetrahedron_non_coplanar_selections :
  totalPoints = vertices + midpoints ∧
  nonCoplanarSelections = 141 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_non_coplanar_selections_l3949_394965


namespace NUMINAMATH_CALUDE_p_plus_q_values_l3949_394914

theorem p_plus_q_values (p q : ℝ) 
  (hp : p^3 - 18*p^2 + 81*p - 162 = 0)
  (hq : 4*q^3 - 24*q^2 + 45*q - 27 = 0) :
  (p + q = 8) ∨ (p + q = 8 + 6*Real.sqrt 3) ∨ (p + q = 8 - 6*Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_p_plus_q_values_l3949_394914
