import Mathlib

namespace NUMINAMATH_CALUDE_coach_grunts_lineups_l3380_338032

/-- Represents a basketball team with its players and lineup constraints -/
structure BasketballTeam where
  total_players : Nat
  all_stars : Nat
  lineup_size : Nat
  mandatory_all_stars : Nat
  min_non_all_stars : Nat

/-- Calculates the number of possible starting lineups for a given basketball team -/
def possible_lineups (team : BasketballTeam) : Nat :=
  Nat.choose (team.total_players - team.all_stars) (team.lineup_size - team.mandatory_all_stars)

/-- The specific basketball team described in the problem -/
def coach_grunts_team : BasketballTeam :=
  { total_players := 15
  , all_stars := 3
  , lineup_size := 5
  , mandatory_all_stars := 2
  , min_non_all_stars := 3 }

/-- Theorem stating that the number of possible starting lineups for Coach Grunt's team is 220 -/
theorem coach_grunts_lineups :
  possible_lineups coach_grunts_team = 220 := by
  sorry

end NUMINAMATH_CALUDE_coach_grunts_lineups_l3380_338032


namespace NUMINAMATH_CALUDE_max_product_of_three_numbers_l3380_338097

theorem max_product_of_three_numbers (n : ℕ+) :
  ∃ (a b c : ℕ), 
    a ∈ Finset.range (3*n + 2) ∧ 
    b ∈ Finset.range (3*n + 2) ∧ 
    c ∈ Finset.range (3*n + 2) ∧ 
    a + b + c = 3*n + 1 ∧
    a * b * c = n^3 + n^2 ∧
    ∀ (x y z : ℕ), 
      x ∈ Finset.range (3*n + 2) → 
      y ∈ Finset.range (3*n + 2) → 
      z ∈ Finset.range (3*n + 2) → 
      x + y + z = 3*n + 1 → 
      x * y * z ≤ n^3 + n^2 := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_three_numbers_l3380_338097


namespace NUMINAMATH_CALUDE_f_2009_equals_1_l3380_338006

def is_even_function (f : ℤ → ℤ) : Prop :=
  ∀ x, f x = f (-x)

theorem f_2009_equals_1
  (f : ℤ → ℤ)
  (h_even : is_even_function f)
  (h_f_1 : f 1 = 1)
  (h_f_2008 : f 2008 ≠ 1)
  (h_max : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b)) :
  f 2009 = 1 := by
sorry

end NUMINAMATH_CALUDE_f_2009_equals_1_l3380_338006


namespace NUMINAMATH_CALUDE_power_tower_at_three_l3380_338008

theorem power_tower_at_three : 
  let x : ℕ := 3
  (x^x)^(x^x) = 27^27 := by
  sorry

end NUMINAMATH_CALUDE_power_tower_at_three_l3380_338008


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l3380_338080

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the asymptote condition
def asymptote_condition (a b : ℝ) : Prop :=
  b / a = Real.sqrt 3

-- Define the focus condition
def focus_condition (c : ℝ) : Prop :=
  Real.sqrt ((1 + c)^2 + 3) = 2

-- Define the point condition
def point_condition (x y c : ℝ) : Prop :=
  Real.sqrt ((x + c)^2 + y^2) = 5/2

-- Main theorem
theorem hyperbola_theorem (a b c : ℝ) (x y : ℝ) :
  hyperbola a b x y →
  asymptote_condition a b →
  focus_condition c →
  point_condition x y c →
  Real.sqrt ((x - c)^2 + y^2) = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l3380_338080


namespace NUMINAMATH_CALUDE_modular_equation_solution_l3380_338060

theorem modular_equation_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n < 151 ∧ (150 * n + 3) % 151 = 45 % 151 ∧ n = 109 := by
  sorry

end NUMINAMATH_CALUDE_modular_equation_solution_l3380_338060


namespace NUMINAMATH_CALUDE_rogers_money_l3380_338019

theorem rogers_money (x : ℝ) : 
  (x + 28 - 25 = 19) → (x = 16) := by
  sorry

end NUMINAMATH_CALUDE_rogers_money_l3380_338019


namespace NUMINAMATH_CALUDE_derivative_at_one_l3380_338005

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = 2 * x * f' 1 + Real.log x) →
  (∀ x, HasDerivAt f (f' x) x) →
  f' 1 = -1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l3380_338005


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l3380_338002

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l3380_338002


namespace NUMINAMATH_CALUDE_triangle_packing_l3380_338093

/-- Represents an equilateral triangle with side length L -/
structure EquilateralTriangle (L : ℝ) where
  sideLength : L > 0

/-- Represents a configuration of unit equilateral triangles inside a larger triangle -/
structure TriangleConfiguration (L : ℝ) where
  largeTriangle : EquilateralTriangle L
  numUnitTriangles : ℕ
  nonOverlapping : Bool
  parallelSides : Bool
  oppositeOrientation : Bool

/-- The theorem statement -/
theorem triangle_packing (L : ℝ) (config : TriangleConfiguration L) :
  config.nonOverlapping ∧ config.parallelSides ∧ config.oppositeOrientation →
  (config.numUnitTriangles : ℝ) ≤ (2 / 3) * L^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_packing_l3380_338093


namespace NUMINAMATH_CALUDE_coin_problem_l3380_338091

/-- Represents the number of coins of each type in the bag -/
def num_coins : ℕ := sorry

/-- Represents the total value of coins in rupees -/
def total_value : ℚ := 140

/-- Theorem stating that if the bag contains an equal number of one rupee, 50 paise, and 25 paise coins, 
    and the total value is 140 rupees, then the number of coins of each type is 80 -/
theorem coin_problem : 
  (num_coins : ℚ) + (num_coins : ℚ) * (1/2) + (num_coins : ℚ) * (1/4) = total_value → 
  num_coins = 80 := by sorry

end NUMINAMATH_CALUDE_coin_problem_l3380_338091


namespace NUMINAMATH_CALUDE_number_line_steps_l3380_338016

theorem number_line_steps (total_distance : ℝ) (num_steps : ℕ) (step_to_x : ℕ) : 
  total_distance = 32 →
  num_steps = 8 →
  step_to_x = 6 →
  (total_distance / num_steps) * step_to_x = 24 := by
sorry

end NUMINAMATH_CALUDE_number_line_steps_l3380_338016


namespace NUMINAMATH_CALUDE_power_equation_solution_l3380_338037

theorem power_equation_solution : ∃ x : ℕ, 125^2 = 5^x ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3380_338037


namespace NUMINAMATH_CALUDE_sqrt_square_iff_abs_l3380_338007

theorem sqrt_square_iff_abs (f g : ℝ → ℝ) :
  (∀ x, Real.sqrt (f x ^ 2) ≥ Real.sqrt (g x ^ 2)) ↔ (∀ x, |f x| ≥ |g x|) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_iff_abs_l3380_338007


namespace NUMINAMATH_CALUDE_alternating_pair_sum_50_eq_2550_l3380_338022

def alternatingPairSum (n : Nat) : Int :=
  let f (k : Nat) : Int :=
    if k % 4 ≤ 1 then (n - k + 1)^2 else -(n - k + 1)^2
  (List.range n).map f |>.sum

theorem alternating_pair_sum_50_eq_2550 :
  alternatingPairSum 50 = 2550 := by
  sorry

end NUMINAMATH_CALUDE_alternating_pair_sum_50_eq_2550_l3380_338022


namespace NUMINAMATH_CALUDE_total_sales_revenue_marie_sales_revenue_l3380_338015

/-- Calculates the total sales revenue from selling magazines and newspapers -/
theorem total_sales_revenue 
  (magazines_sold : ℕ) 
  (newspapers_sold : ℕ) 
  (magazine_price : ℚ) 
  (newspaper_price : ℚ) : ℚ :=
  magazines_sold * magazine_price + newspapers_sold * newspaper_price

/-- Proves that the total sales revenue for the given quantities and prices is correct -/
theorem marie_sales_revenue : 
  total_sales_revenue 425 275 (35/10) (5/4) = 1831.25 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_revenue_marie_sales_revenue_l3380_338015


namespace NUMINAMATH_CALUDE_sufficient_condition_for_a_gt_b_l3380_338035

theorem sufficient_condition_for_a_gt_b (a b : ℝ) : 
  (1 / a < 1 / b) ∧ (1 / b < 0) → a > b := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_a_gt_b_l3380_338035


namespace NUMINAMATH_CALUDE_book_series_first_year_l3380_338050

/-- Represents the publication years of a book series -/
def BookSeries (a : ℕ) : List ℕ :=
  List.range 7 |>.map (fun i => a + 7 * i)

/-- The theorem stating the properties of the book series -/
theorem book_series_first_year :
  ∀ a : ℕ,
  (BookSeries a).length = 7 ∧
  (∀ i j, i < j → (BookSeries a).get i < (BookSeries a).get j) ∧
  (BookSeries a).sum = 13524 →
  a = 1911 := by
sorry


end NUMINAMATH_CALUDE_book_series_first_year_l3380_338050


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3380_338072

/-- A quadratic expression is a square of a binomial if and only if its coefficients satisfy certain conditions -/
theorem quadratic_is_square_of_binomial (b : ℝ) : 
  (∃ (t u : ℝ), ∀ x, b * x^2 + 8 * x + 4 = (t * x + u)^2) ↔ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3380_338072


namespace NUMINAMATH_CALUDE_prism_volume_l3380_338085

/-- 
A right rectangular prism with one side length of 4 inches, 
and two faces with areas of 24 and 16 square inches respectively, 
has a volume of 64 cubic inches.
-/
theorem prism_volume : 
  ∀ (x y z : ℝ), 
  x = 4 → 
  x * y = 24 → 
  y * z = 16 → 
  x * y * z = 64 := by
sorry

end NUMINAMATH_CALUDE_prism_volume_l3380_338085


namespace NUMINAMATH_CALUDE_problem_solution_l3380_338094

theorem problem_solution (x : ℚ) : x - 2/5 = 7/15 - 1/3 - 1/6 → x = 11/30 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3380_338094


namespace NUMINAMATH_CALUDE_bara_numbers_l3380_338081

theorem bara_numbers (a b : ℤ) (h1 : a ≠ b) 
  (h2 : (a + b) + (a - b) + a * b + a / b = -100)
  (h3 : (a - b) + a * b + a / b = -100) :
  (a = -9 ∧ b = 9) ∨ (a = 11 ∧ b = -11) := by
sorry

end NUMINAMATH_CALUDE_bara_numbers_l3380_338081


namespace NUMINAMATH_CALUDE_dice_probability_l3380_338011

/-- The number of possible outcomes when rolling 7 six-sided dice -/
def total_outcomes : ℕ := 6^7

/-- The number of ways to choose 2 numbers from 6 -/
def choose_two_from_six : ℕ := Nat.choose 6 2

/-- The number of ways to arrange 2 pairs in 7 positions -/
def arrange_two_pairs : ℕ := Nat.choose 7 2 * Nat.choose 5 2

/-- The number of ways to arrange remaining dice for two pairs case -/
def arrange_remaining_two_pairs : ℕ := 4 * 3 * 2

/-- The number of ways to arrange triplet and pair -/
def arrange_triplet_pair : ℕ := Nat.choose 7 3 * Nat.choose 4 2

/-- The number of ways to arrange remaining dice for triplet and pair case -/
def arrange_remaining_triplet_pair : ℕ := 4 * 3

/-- The total number of favorable outcomes -/
def favorable_outcomes : ℕ :=
  (choose_two_from_six * arrange_two_pairs * arrange_remaining_two_pairs) +
  (2 * choose_two_from_six * arrange_triplet_pair * arrange_remaining_triplet_pair)

theorem dice_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 525 / 972 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l3380_338011


namespace NUMINAMATH_CALUDE_quadratic_two_roots_k_range_l3380_338014

theorem quadratic_two_roots_k_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ - k = 0 ∧ x₂^2 + 2*x₂ - k = 0) → k > -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_k_range_l3380_338014


namespace NUMINAMATH_CALUDE_complex_power_difference_l3380_338028

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) :
  (1 + i)^20 - (1 - i)^20 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_power_difference_l3380_338028


namespace NUMINAMATH_CALUDE_unique_solution_sin_cos_equation_l3380_338044

theorem unique_solution_sin_cos_equation :
  ∃! (n : ℕ+), Real.sin (π / (2 * n.val)) * Real.cos (π / (2 * n.val)) = n.val / 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sin_cos_equation_l3380_338044


namespace NUMINAMATH_CALUDE_EF_length_l3380_338000

/-- Configuration of line segments AB, CD, and EF -/
structure Configuration where
  AB_length : ℝ
  CD_length : ℝ
  EF_start_x : ℝ
  EF_end_x : ℝ
  AB_height : ℝ
  CD_height : ℝ
  EF_height : ℝ

/-- Conditions for the configuration -/
def valid_configuration (c : Configuration) : Prop :=
  c.AB_length = 120 ∧
  c.CD_length = 80 ∧
  c.EF_start_x = c.CD_length / 2 ∧
  c.EF_end_x = c.CD_length ∧
  c.AB_height > c.EF_height ∧
  c.EF_height > c.CD_height ∧
  c.EF_height = (c.AB_height + c.CD_height) / 2

/-- Theorem: The length of EF is 40 cm -/
theorem EF_length (c : Configuration) (h : valid_configuration c) : 
  c.EF_end_x - c.EF_start_x = 40 := by
  sorry

end NUMINAMATH_CALUDE_EF_length_l3380_338000


namespace NUMINAMATH_CALUDE_log_8_y_value_l3380_338077

theorem log_8_y_value (y : ℝ) (h : Real.log y / Real.log 8 = 3.25) : y = 32 * Real.sqrt (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_log_8_y_value_l3380_338077


namespace NUMINAMATH_CALUDE_third_month_sale_l3380_338004

def sale_month1 : ℕ := 6435
def sale_month2 : ℕ := 6927
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6562
def sale_month6 : ℕ := 7391
def average_sale : ℕ := 6900
def num_months : ℕ := 6

theorem third_month_sale :
  ∃ (sale_month3 : ℕ),
    sale_month3 = num_months * average_sale - (sale_month1 + sale_month2 + sale_month4 + sale_month5 + sale_month6) ∧
    sale_month3 = 6855 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l3380_338004


namespace NUMINAMATH_CALUDE_part1_part2_l3380_338025

-- Define the functions f and h
def f (m : ℝ) (x : ℝ) : ℝ := |x - m|
def h (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem for part 1
theorem part1 (m : ℝ) : 
  (∀ x, f m x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → m = 2 :=
sorry

-- Theorem for part 2
theorem part2 (t : ℝ) :
  (∃ x, x^2 + 6*x + h t = 0) → -5 ≤ t ∧ t ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l3380_338025


namespace NUMINAMATH_CALUDE_g_of_5_equals_15_l3380_338086

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem g_of_5_equals_15 : g 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_equals_15_l3380_338086


namespace NUMINAMATH_CALUDE_one_plane_through_line_parallel_to_skew_line_l3380_338090

-- Define the concept of a line in 3D space
structure Line3D where
  -- You might define a line using a point and a direction vector
  -- But for simplicity, we'll just declare it as an opaque type
  dummy : Unit

-- Define the concept of a plane in 3D space
structure Plane3D where
  -- Similar to Line3D, we'll keep this as an opaque type for simplicity
  dummy : Unit

-- Define what it means for two lines to be skew
def are_skew (a b : Line3D) : Prop :=
  -- Two lines are skew if they are neither intersecting nor parallel
  sorry

-- Define what it means for a plane to contain a line
def plane_contains_line (p : Plane3D) (l : Line3D) : Prop :=
  sorry

-- Define what it means for a plane to be parallel to a line
def plane_parallel_to_line (p : Plane3D) (l : Line3D) : Prop :=
  sorry

-- The main theorem
theorem one_plane_through_line_parallel_to_skew_line 
  (a b : Line3D) (h : are_skew a b) : 
  ∃! p : Plane3D, plane_contains_line p a ∧ plane_parallel_to_line p b :=
sorry

end NUMINAMATH_CALUDE_one_plane_through_line_parallel_to_skew_line_l3380_338090


namespace NUMINAMATH_CALUDE_p_or_q_can_be_either_l3380_338079

theorem p_or_q_can_be_either (p q : Prop) (h : ¬(p ∧ q)) : 
  (∃ b : Bool, (p ∨ q) = b) ∧ (∃ b : Bool, (p ∨ q) ≠ b) := by
sorry

end NUMINAMATH_CALUDE_p_or_q_can_be_either_l3380_338079


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3380_338099

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (2*x - 1)^6 = a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 729 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3380_338099


namespace NUMINAMATH_CALUDE_final_combined_price_theorem_l3380_338068

/-- Calculates the final price of an item after applying discount and tax --/
def finalPrice (initialPrice : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  initialPrice * (1 - discount) * (1 + tax)

/-- Calculates the price of an accessory after applying tax --/
def accessoryPrice (price : ℝ) (tax : ℝ) : ℝ :=
  price * (1 + tax)

/-- Theorem stating the final combined price of iPhone and accessories --/
theorem final_combined_price_theorem 
  (iPhoneInitialPrice : ℝ) 
  (iPhoneDiscount1 iPhoneDiscount2 : ℝ)
  (iPhoneTax1 iPhoneTax2 : ℝ)
  (screenProtectorPrice casePrice : ℝ)
  (accessoriesTax : ℝ)
  (h1 : iPhoneInitialPrice = 1000)
  (h2 : iPhoneDiscount1 = 0.1)
  (h3 : iPhoneDiscount2 = 0.2)
  (h4 : iPhoneTax1 = 0.08)
  (h5 : iPhoneTax2 = 0.06)
  (h6 : screenProtectorPrice = 30)
  (h7 : casePrice = 50)
  (h8 : accessoriesTax = 0.05) :
  let iPhoneFinalPrice := finalPrice (finalPrice iPhoneInitialPrice iPhoneDiscount1 iPhoneTax1) iPhoneDiscount2 iPhoneTax2
  let totalAccessoriesPrice := accessoryPrice screenProtectorPrice accessoriesTax + accessoryPrice casePrice accessoriesTax
  iPhoneFinalPrice + totalAccessoriesPrice = 908.256 := by
    sorry


end NUMINAMATH_CALUDE_final_combined_price_theorem_l3380_338068


namespace NUMINAMATH_CALUDE_fencing_cost_approx_l3380_338041

-- Define the diameter of the circular field
def diameter : ℝ := 40

-- Define the cost per meter of fencing
def cost_per_meter : ℝ := 3

-- Define pi as a constant (approximation)
def π : ℝ := 3.14159

-- Define the function to calculate the circumference of a circle
def circumference (d : ℝ) : ℝ := π * d

-- Define the function to calculate the total cost of fencing
def total_cost (c : ℝ) (rate : ℝ) : ℝ := c * rate

-- Theorem stating that the total cost is approximately 377
theorem fencing_cost_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  abs (total_cost (circumference diameter) cost_per_meter - 377) < ε :=
sorry

end NUMINAMATH_CALUDE_fencing_cost_approx_l3380_338041


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l3380_338040

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + x^2

theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (y = m * (x - 1) + f 1) ↔ (y = m * x + b ∧ 4 * x - y - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l3380_338040


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_l3380_338084

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_constant
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_const : ∃ k : ℝ, a 2 + a 4 + a 15 = k) :
  ∃ c : ℝ, a 7 = c :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_l3380_338084


namespace NUMINAMATH_CALUDE_factorial_30_trailing_zeros_l3380_338087

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- 30! has 7 trailing zeros -/
theorem factorial_30_trailing_zeros : trailingZeros 30 = 7 := by sorry

end NUMINAMATH_CALUDE_factorial_30_trailing_zeros_l3380_338087


namespace NUMINAMATH_CALUDE_cone_lateral_area_l3380_338026

/-- Given a cone with a central angle of 120° in its unfolded diagram and a base circle radius of 2 cm,
    prove that its lateral area is 12π cm². -/
theorem cone_lateral_area (central_angle : Real) (base_radius : Real) (lateral_area : Real) :
  central_angle = 120 * (π / 180) →
  base_radius = 2 →
  lateral_area = 12 * π →
  lateral_area = (1 / 2) * (2 * π * base_radius) * ((2 * π * base_radius) / (2 * π * (central_angle / (2 * π)))) :=
by sorry


end NUMINAMATH_CALUDE_cone_lateral_area_l3380_338026


namespace NUMINAMATH_CALUDE_total_students_l3380_338063

theorem total_students (group1_count : ℕ) (group1_avg : ℚ)
                       (group2_count : ℕ) (group2_avg : ℚ)
                       (total_avg : ℚ) :
  group1_count = 15 →
  group1_avg = 80 / 100 →
  group2_count = 10 →
  group2_avg = 90 / 100 →
  total_avg = 84 / 100 →
  group1_count + group2_count = 25 := by
sorry

end NUMINAMATH_CALUDE_total_students_l3380_338063


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l3380_338092

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planesPerpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) (h_distinct : α ≠ β)
  (h_parallel : parallel l α) (h_perpendicular : perpendicular l β) :
  planesPerpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l3380_338092


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3380_338013

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r ∧ a n > 0

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36 →
  a 3 + a 5 = 6 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3380_338013


namespace NUMINAMATH_CALUDE_carol_initial_cupcakes_l3380_338062

/-- The number of cupcakes Carol initially made -/
def initial_cupcakes : ℕ := 30

/-- The number of cupcakes Carol sold -/
def sold_cupcakes : ℕ := 9

/-- The number of additional cupcakes Carol made -/
def additional_cupcakes : ℕ := 28

/-- The total number of cupcakes Carol had at the end -/
def total_cupcakes : ℕ := 49

/-- Theorem: Carol initially made 30 cupcakes -/
theorem carol_initial_cupcakes : 
  initial_cupcakes = total_cupcakes - additional_cupcakes + sold_cupcakes :=
by sorry

end NUMINAMATH_CALUDE_carol_initial_cupcakes_l3380_338062


namespace NUMINAMATH_CALUDE_area_of_bounded_region_l3380_338052

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop :=
  y^2 + 2*x*y + 50*abs x = 500

-- Define the bounded region
def bounded_region : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ graph_equation x y}

-- State the theorem
theorem area_of_bounded_region :
  MeasureTheory.volume bounded_region = 1250 := by sorry

end NUMINAMATH_CALUDE_area_of_bounded_region_l3380_338052


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3380_338036

def M : Set ℝ := {x : ℝ | x^2 + 2*x = 0}
def N : Set ℝ := {x : ℝ | x^2 - 2*x = 0}

theorem union_of_M_and_N : M ∪ N = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3380_338036


namespace NUMINAMATH_CALUDE_sequence_seventh_term_l3380_338064

theorem sequence_seventh_term (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, a 1 + a (2 * n - 1) = 4 * n - 6) : 
  a 7 = 11 := by
sorry

end NUMINAMATH_CALUDE_sequence_seventh_term_l3380_338064


namespace NUMINAMATH_CALUDE_workers_read_both_books_l3380_338070

/-- The number of workers who have read both Saramago's and Kureishi's latest books -/
def workers_read_both (total : ℕ) (saramago : ℕ) (kureishi : ℕ) (neither : ℕ) : ℕ :=
  saramago + kureishi - (total - neither)

theorem workers_read_both_books :
  let total := 42
  let saramago := total / 2
  let kureishi := total / 6
  let neither := saramago - kureishi - 1
  workers_read_both total saramago kureishi neither = 6 := by
  sorry

#eval workers_read_both 42 21 7 20

end NUMINAMATH_CALUDE_workers_read_both_books_l3380_338070


namespace NUMINAMATH_CALUDE_complex_number_problem_l3380_338029

theorem complex_number_problem (a : ℝ) (z₁ : ℂ) (h₁ : a > 0) (h₂ : z₁ = 1 + a * I) (h₃ : ∃ b : ℝ, z₁^2 = b * I) :
  z₁ = 1 + I ∧ Complex.abs (z₁ / (1 - I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3380_338029


namespace NUMINAMATH_CALUDE_bake_sale_group_composition_l3380_338057

theorem bake_sale_group_composition (total : ℕ) (girls : ℕ) : 
  girls = (60 : ℕ) * total / 100 →
  (girls - 3 : ℕ) * 2 = total →
  girls = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_bake_sale_group_composition_l3380_338057


namespace NUMINAMATH_CALUDE_power_of_power_three_cubed_fourth_l3380_338055

theorem power_of_power_three_cubed_fourth : (3^3)^4 = 531441 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_cubed_fourth_l3380_338055


namespace NUMINAMATH_CALUDE_two_points_at_distance_from_line_l3380_338048

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Distance between a point and a line in 3D space -/
def distance_point_to_line (p : ℝ × ℝ × ℝ) (l : Line3D) : ℝ :=
  sorry

/-- Check if a line segment is perpendicular to a line in 3D space -/
def is_perpendicular (p1 p2 : ℝ × ℝ × ℝ) (l : Line3D) : Prop :=
  sorry

theorem two_points_at_distance_from_line 
  (L : Line3D) (d : ℝ) (P : ℝ × ℝ × ℝ) :
  ∃ (Q1 Q2 : ℝ × ℝ × ℝ),
    distance_point_to_line Q1 L = d ∧
    distance_point_to_line Q2 L = d ∧
    is_perpendicular P Q1 L ∧
    is_perpendicular P Q2 L :=
  sorry

end NUMINAMATH_CALUDE_two_points_at_distance_from_line_l3380_338048


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3380_338059

theorem rectangle_perimeter (b l : ℝ) (h1 : l = 3 * b) (h2 : b * l = 192) :
  2 * (b + l) = 64 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3380_338059


namespace NUMINAMATH_CALUDE_news_report_probability_l3380_338075

/-- The duration of the "Midday News" program in minutes -/
def program_duration : ℕ := 30

/-- The duration of the news report in minutes -/
def news_report_duration : ℕ := 5

/-- The time Xiao Zhang starts watching, in minutes after the program start -/
def watch_start_time : ℕ := 20

/-- The probability of watching the entire news report -/
def watch_probability : ℚ := 1 / 6

theorem news_report_probability :
  let favorable_time := program_duration - watch_start_time - news_report_duration + 1
  watch_probability = favorable_time / program_duration :=
by sorry

end NUMINAMATH_CALUDE_news_report_probability_l3380_338075


namespace NUMINAMATH_CALUDE_correct_articles_for_newton_discovery_l3380_338027

/-- Represents the possible article choices for each blank --/
inductive Article
  | A : Article  -- represents "a"
  | The : Article  -- represents "the"
  | None : Article  -- represents no article

/-- Represents the context of the discovery --/
structure DiscoveryContext where
  is_specific : Bool
  is_previously_mentioned : Bool

/-- Represents the usage of "man" in the sentence --/
structure ManUsage where
  represents_mankind : Bool

/-- Determines the correct article choice given the context --/
def correct_article (context : DiscoveryContext) (man_usage : ManUsage) : Article × Article :=
  sorry

/-- Theorem stating the correct article choice for the given sentence --/
theorem correct_articles_for_newton_discovery 
  (context : DiscoveryContext)
  (man_usage : ManUsage)
  (h1 : context.is_specific = true)
  (h2 : context.is_previously_mentioned = false)
  (h3 : man_usage.represents_mankind = true) :
  correct_article context man_usage = (Article.A, Article.The) :=
sorry

end NUMINAMATH_CALUDE_correct_articles_for_newton_discovery_l3380_338027


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3380_338039

/-- A geometric sequence with all positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_roots : ∃ m : ℝ, 2 * (a 5)^2 - m * (a 5) + 2 * Real.exp 4 = 0 ∧
                      2 * (a 13)^2 - m * (a 13) + 2 * Real.exp 4 = 0) :
  a 7 * a 9 * a 11 = Real.exp 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3380_338039


namespace NUMINAMATH_CALUDE_percentage_problem_l3380_338076

theorem percentage_problem (P : ℝ) : 
  (0.5 * 456 = (P / 100) * 120 + 180) → P = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3380_338076


namespace NUMINAMATH_CALUDE_hyperbola_vertex_to_asymptote_distance_l3380_338046

/-- Given a hyperbola with equation x²/a² - y²/3 = 1 and eccentricity 2,
    the distance from its vertex to its asymptote is √3/2 -/
theorem hyperbola_vertex_to_asymptote_distance
  (a : ℝ) -- Semi-major axis
  (h1 : a > 0) -- a is positive
  (h2 : (a^2 + 3) / a^2 = 4) -- Eccentricity condition
  : Real.sqrt 3 / 2 = 
    abs (-Real.sqrt 3 * a) / Real.sqrt (3 + 1) := by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_to_asymptote_distance_l3380_338046


namespace NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l3380_338012

theorem divisibility_of_fifth_power_differences (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (x - y) * (y - z) * (z - x)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l3380_338012


namespace NUMINAMATH_CALUDE_fixed_fee_calculation_l3380_338096

/-- Represents a cable service bill -/
structure CableBill where
  fixed_fee : ℝ
  hourly_rate : ℝ
  usage_hours : ℝ

/-- Calculates the total bill amount -/
def bill_amount (b : CableBill) : ℝ :=
  b.fixed_fee + b.hourly_rate * b.usage_hours

theorem fixed_fee_calculation 
  (feb : CableBill) (mar : CableBill) 
  (h_feb_amount : bill_amount feb = 20.72)
  (h_mar_amount : bill_amount mar = 35.28)
  (h_same_fee : feb.fixed_fee = mar.fixed_fee)
  (h_same_rate : feb.hourly_rate = mar.hourly_rate)
  (h_triple_usage : mar.usage_hours = 3 * feb.usage_hours) :
  feb.fixed_fee = 13.44 := by
sorry

end NUMINAMATH_CALUDE_fixed_fee_calculation_l3380_338096


namespace NUMINAMATH_CALUDE_parabola_intersection_l3380_338047

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3*x^2 + 4*x - 5
def g (x : ℝ) : ℝ := x^2 + 11

-- Theorem stating the intersection points
theorem parabola_intersection :
  (∃ (x y : ℝ), f x = g x ∧ y = f x) ↔
  (∃ (x y : ℝ), (x = -4 ∧ y = 27) ∨ (x = 2 ∧ y = 15)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l3380_338047


namespace NUMINAMATH_CALUDE_local_politics_coverage_l3380_338098

theorem local_politics_coverage (total_reporters : ℕ) 
  (h1 : total_reporters > 0) 
  (politics_coverage : ℝ) 
  (h2 : politics_coverage = 0.25) 
  (local_politics_non_coverage : ℝ) 
  (h3 : local_politics_non_coverage = 0.2) : 
  (politics_coverage * (1 - local_politics_non_coverage)) * total_reporters / total_reporters = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_local_politics_coverage_l3380_338098


namespace NUMINAMATH_CALUDE_cyclist_rejoining_time_l3380_338051

/-- Prove that the time taken for a cyclist to break away from a group, travel 10 km ahead, 
    turn back, and rejoin the group is 1/4 hours. -/
theorem cyclist_rejoining_time 
  (group_speed : ℝ) 
  (cyclist_speed : ℝ) 
  (separation_distance : ℝ) 
  (h1 : group_speed = 35) 
  (h2 : cyclist_speed = 45) 
  (h3 : separation_distance = 20) : 
  (separation_distance / (cyclist_speed - group_speed) = 1/4) := by
sorry

end NUMINAMATH_CALUDE_cyclist_rejoining_time_l3380_338051


namespace NUMINAMATH_CALUDE_largest_square_tile_size_l3380_338033

/-- The length of the courtyard in centimeters -/
def courtyard_length : ℕ := 378

/-- The width of the courtyard in centimeters -/
def courtyard_width : ℕ := 525

/-- The size of the largest square tile in centimeters -/
def largest_tile_size : ℕ := 21

theorem largest_square_tile_size :
  (courtyard_length % largest_tile_size = 0) ∧
  (courtyard_width % largest_tile_size = 0) ∧
  ∀ (tile_size : ℕ), tile_size > largest_tile_size →
    (courtyard_length % tile_size ≠ 0) ∨ (courtyard_width % tile_size ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_square_tile_size_l3380_338033


namespace NUMINAMATH_CALUDE_players_bought_l3380_338031

/-- Calculates the number of players bought by a football club given their financial transactions -/
theorem players_bought (initial_balance : ℕ) (players_sold : ℕ) (selling_price : ℕ) (buying_price : ℕ) (final_balance : ℕ) : 
  initial_balance + players_sold * selling_price - final_balance = 4 * buying_price :=
by
  sorry

#check players_bought 100000000 2 10000000 15000000 60000000

end NUMINAMATH_CALUDE_players_bought_l3380_338031


namespace NUMINAMATH_CALUDE_saras_quarters_l3380_338066

/-- Sara's quarters problem -/
theorem saras_quarters (initial_quarters borrowed_quarters : ℕ) 
  (h1 : initial_quarters = 4937)
  (h2 : borrowed_quarters = 1743) :
  initial_quarters - borrowed_quarters = 3194 :=
by sorry

end NUMINAMATH_CALUDE_saras_quarters_l3380_338066


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l3380_338056

theorem smallest_addition_for_divisibility : ∃! x : ℕ, 
  (x ≤ y → ∀ y : ℕ, (758492136547 + y) % 17 = 0 ∧ (758492136547 + y) % 3 = 0) ∧
  (758492136547 + x) % 17 = 0 ∧ 
  (758492136547 + x) % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l3380_338056


namespace NUMINAMATH_CALUDE_rectangle_length_l3380_338020

/-- Given a rectangle with a length to width ratio of 6:5 and a width of 20 inches,
    prove that its length is 24 inches. -/
theorem rectangle_length (width : ℝ) (length : ℝ) : 
  width = 20 → length / width = 6 / 5 → length = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l3380_338020


namespace NUMINAMATH_CALUDE_b_completes_in_24_days_l3380_338067

/-- Worker represents a person who can complete a task -/
structure Worker where
  rate : ℚ  -- work rate in units of work per day

/-- Represents a work scenario with three workers -/
structure WorkScenario where
  a : Worker
  b : Worker
  c : Worker
  combined_time_ab : ℚ  -- time for a and b to complete work together
  time_a : ℚ           -- time for a to complete work alone
  time_c : ℚ           -- time for c to complete work alone

/-- Calculate the time for worker b to complete the work alone -/
def time_for_b_alone (w : WorkScenario) : ℚ :=
  1 / (1 / w.combined_time_ab - 1 / w.time_a)

/-- Theorem stating that given the conditions, b takes 24 days to complete the work alone -/
theorem b_completes_in_24_days (w : WorkScenario) 
  (h1 : w.combined_time_ab = 8)
  (h2 : w.time_a = 12)
  (h3 : w.time_c = 18) :
  time_for_b_alone w = 24 := by
  sorry

#eval time_for_b_alone { a := ⟨1/12⟩, b := ⟨1/24⟩, c := ⟨1/18⟩, combined_time_ab := 8, time_a := 12, time_c := 18 }

end NUMINAMATH_CALUDE_b_completes_in_24_days_l3380_338067


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l3380_338001

theorem min_value_of_expression (x : ℝ) : 
  (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by sorry

theorem min_value_attained : 
  ∃ x : ℝ, (15 - x) * (13 - x) * (15 + x) * (13 + x) = -784 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l3380_338001


namespace NUMINAMATH_CALUDE_equation_solutions_l3380_338065

theorem equation_solutions :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁^2 - 9*y₁^2 = 18 ∧
    x₂^2 - 9*y₂^2 = 18 ∧
    x₁ = 19/2 ∧ y₁ = 17/6 ∧
    x₂ = 11/2 ∧ y₂ = 7/6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3380_338065


namespace NUMINAMATH_CALUDE_orange_min_cost_l3380_338038

/-- Represents the cost and quantity of oranges in a package -/
structure Package where
  quantity : ℕ
  cost : ℕ

/-- Calculates the minimum cost to buy a given number of oranges -/
def minCost (bag : Package) (box : Package) (total : ℕ) : ℕ :=
  sorry

theorem orange_min_cost :
  let bag : Package := ⟨4, 12⟩
  let box : Package := ⟨6, 25⟩
  let total : ℕ := 20
  minCost bag box total = 60 := by
  sorry

end NUMINAMATH_CALUDE_orange_min_cost_l3380_338038


namespace NUMINAMATH_CALUDE_unique_number_l3380_338069

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def contains_digit_5 (n : ℕ) : Prop := ∃ a b, n = 10*a + 5 + b ∧ 0 ≤ b ∧ b < 10

def divisible_by_3 (n : ℕ) : Prop := ∃ k, n = 3*k

theorem unique_number : 
  ∃! n : ℕ, 
    144 < n ∧ 
    n < 169 ∧ 
    is_odd n ∧ 
    contains_digit_5 n ∧ 
    divisible_by_3 n ∧ 
    n = 165 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_l3380_338069


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l3380_338073

theorem nested_fraction_equality : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l3380_338073


namespace NUMINAMATH_CALUDE_focus_of_parabola_l3380_338071

/-- The parabola defined by the equation y^2 = 4x -/
def parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of a parabola with equation y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- Theorem: The focus of the parabola y^2 = 4x has coordinates (1, 0) -/
theorem focus_of_parabola :
  focus ∈ {p : ℝ × ℝ | p.1 > 0 ∧ ∀ q ∈ parabola, (p.1 - q.1)^2 + (p.2 - q.2)^2 = (p.1 + q.1)^2} :=
sorry

end NUMINAMATH_CALUDE_focus_of_parabola_l3380_338071


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3380_338049

theorem algebraic_expression_value (a b : ℝ) (h : 5*a + 3*b = -4) :
  -8 - 2*(a + b) - 4*(2*a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3380_338049


namespace NUMINAMATH_CALUDE_jason_money_unchanged_l3380_338045

/-- Represents the money situation of Fred and Jason -/
structure MoneySituation where
  fred_initial : ℕ
  jason_initial : ℕ
  fred_final : ℕ
  total_earned : ℕ

/-- The theorem stating that Jason's final money is equal to his initial money -/
theorem jason_money_unchanged (situation : MoneySituation) 
  (h1 : situation.fred_initial = 111)
  (h2 : situation.jason_initial = 40)
  (h3 : situation.fred_final = 115)
  (h4 : situation.total_earned = 4) :
  situation.jason_initial = 40 := by
  sorry

#check jason_money_unchanged

end NUMINAMATH_CALUDE_jason_money_unchanged_l3380_338045


namespace NUMINAMATH_CALUDE_expected_product_1000_flips_l3380_338043

/-- The expected value of the product of heads and tails for n fair coin flips -/
def expected_product (n : ℕ) : ℚ := n * (n - 1) / 4

/-- Theorem: The expected value of the product of heads and tails for 1000 fair coin flips is 249750 -/
theorem expected_product_1000_flips : 
  expected_product 1000 = 249750 := by sorry

end NUMINAMATH_CALUDE_expected_product_1000_flips_l3380_338043


namespace NUMINAMATH_CALUDE_shaded_area_is_110_l3380_338058

/-- Represents a triangle inscribed in a hexagon --/
inductive InscribedTriangle
  | Small
  | Medium
  | Large

/-- The area of an inscribed triangle in terms of the number of unit triangles it contains --/
def triangle_area (t : InscribedTriangle) : ℕ :=
  match t with
  | InscribedTriangle.Small => 1
  | InscribedTriangle.Medium => 3
  | InscribedTriangle.Large => 7

/-- The area of a unit equilateral triangle in the hexagon --/
def unit_triangle_area : ℕ := 10

/-- The total area of the shaded part --/
def shaded_area : ℕ :=
  (triangle_area InscribedTriangle.Small +
   triangle_area InscribedTriangle.Medium +
   triangle_area InscribedTriangle.Large) * unit_triangle_area

theorem shaded_area_is_110 : shaded_area = 110 := by
  sorry


end NUMINAMATH_CALUDE_shaded_area_is_110_l3380_338058


namespace NUMINAMATH_CALUDE_sam_has_148_balls_l3380_338088

-- Define the number of tennis balls for each person
def lily_balls : ℕ := 84

-- Define Frodo's tennis balls in terms of Lily's
def frodo_balls : ℕ := (lily_balls * 135 + 50) / 100

-- Define Brian's tennis balls in terms of Frodo's
def brian_balls : ℕ := (frodo_balls * 35 + 5) / 10

-- Define Sam's tennis balls
def sam_balls : ℕ := ((frodo_balls + lily_balls) * 3 + 2) / 4

-- Theorem statement
theorem sam_has_148_balls : sam_balls = 148 := by
  sorry

end NUMINAMATH_CALUDE_sam_has_148_balls_l3380_338088


namespace NUMINAMATH_CALUDE_divisor_for_5_pow_100_mod_13_l3380_338042

theorem divisor_for_5_pow_100_mod_13 (D : ℕ+) :
  (5^100 : ℕ) % D = 13 → D = 5^100 - 13 + 1 := by
sorry

end NUMINAMATH_CALUDE_divisor_for_5_pow_100_mod_13_l3380_338042


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3380_338061

open Real

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, 0 < x ∧ x < π / 2 → x < tan x)) ↔
  (∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ x ≥ tan x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3380_338061


namespace NUMINAMATH_CALUDE_chip_price_reduction_l3380_338095

theorem chip_price_reduction (a b : ℝ) : 
  (∃ (price_after_first_reduction : ℝ), 
    price_after_first_reduction = a * (1 - 0.1) ∧
    b = price_after_first_reduction * (1 - 0.2)) →
  b = a * (1 - 0.1) * (1 - 0.2) := by
sorry

end NUMINAMATH_CALUDE_chip_price_reduction_l3380_338095


namespace NUMINAMATH_CALUDE_rectangle_new_length_l3380_338089

/-- Given a rectangle with original length 18 cm and breadth 10 cm,
    if the breadth is changed to 7.2 cm while maintaining the same area,
    the new length will be 25 cm. -/
theorem rectangle_new_length (original_length original_breadth new_breadth new_length : ℝ) :
  original_length = 18 ∧
  original_breadth = 10 ∧
  new_breadth = 7.2 ∧
  original_length * original_breadth = new_length * new_breadth →
  new_length = 25 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_new_length_l3380_338089


namespace NUMINAMATH_CALUDE_duty_roster_arrangements_l3380_338078

def number_of_arrangements (n : ℕ) : ℕ := n.factorial

def adjacent_arrangements (n : ℕ) : ℕ := 2 * (n - 1).factorial

def double_adjacent_arrangements (n : ℕ) : ℕ := 2 * 2 * (n - 2).factorial

theorem duty_roster_arrangements :
  let total := number_of_arrangements 6
  let adjacent_ab := adjacent_arrangements 6
  let adjacent_cd := adjacent_arrangements 6
  let both_adjacent := double_adjacent_arrangements 6
  total - adjacent_ab - adjacent_cd + both_adjacent = 336 := by sorry

end NUMINAMATH_CALUDE_duty_roster_arrangements_l3380_338078


namespace NUMINAMATH_CALUDE_corrected_mean_l3380_338018

theorem corrected_mean (n : ℕ) (incorrect_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) :
  n = 50 ∧ incorrect_mean = 30 ∧ incorrect_value = 23 ∧ correct_value = 48 →
  (n : ℚ) * incorrect_mean - incorrect_value + correct_value = n * (30.5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l3380_338018


namespace NUMINAMATH_CALUDE_replaced_person_weight_l3380_338003

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℝ) (new_person_weight : ℝ) : ℝ :=
  new_person_weight - initial_count * average_increase

/-- Theorem stating that the weight of the replaced person is 60 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 2.5 80 = 60 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l3380_338003


namespace NUMINAMATH_CALUDE_coin_problem_l3380_338030

theorem coin_problem (total : ℕ) (difference : ℕ) (tails : ℕ) : 
  total = 1250 →
  difference = 124 →
  tails + (tails + difference) = total →
  tails = 563 := by
sorry

end NUMINAMATH_CALUDE_coin_problem_l3380_338030


namespace NUMINAMATH_CALUDE_fib_like_seq_a7_l3380_338054

/-- An increasing sequence of positive integers satisfying the Fibonacci-like recurrence -/
def FibLikeSeq (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, n ≥ 1 → a (n + 2) = a (n + 1) + a n)

theorem fib_like_seq_a7 (a : ℕ → ℕ) (h : FibLikeSeq a) (h6 : a 6 = 50) : 
  a 7 = 83 := by
sorry

end NUMINAMATH_CALUDE_fib_like_seq_a7_l3380_338054


namespace NUMINAMATH_CALUDE_last_guard_hours_l3380_338021

/-- Represents the number of hours in a night shift -/
def total_hours : ℕ := 9

/-- Represents the number of guards -/
def num_guards : ℕ := 4

/-- Represents the hours taken by the first guard -/
def first_guard_hours : ℕ := 3

/-- Represents the hours taken by each middle guard -/
def middle_guard_hours : ℕ := 2

/-- Represents the number of middle guards -/
def num_middle_guards : ℕ := 2

theorem last_guard_hours :
  total_hours - (first_guard_hours + num_middle_guards * middle_guard_hours) = 2 := by
  sorry

end NUMINAMATH_CALUDE_last_guard_hours_l3380_338021


namespace NUMINAMATH_CALUDE_total_weight_AlF3_is_839_8_l3380_338053

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of fluorine in g/mol -/
def atomic_weight_F : ℝ := 19.00

/-- The number of aluminum atoms in AlF3 -/
def num_Al : ℕ := 1

/-- The number of fluorine atoms in AlF3 -/
def num_F : ℕ := 3

/-- The number of moles of AlF3 -/
def num_moles : ℝ := 10

/-- The molecular weight of AlF3 in g/mol -/
def molecular_weight_AlF3 : ℝ := atomic_weight_Al * num_Al + atomic_weight_F * num_F

/-- The total weight of AlF3 in grams -/
def total_weight_AlF3 : ℝ := molecular_weight_AlF3 * num_moles

theorem total_weight_AlF3_is_839_8 : total_weight_AlF3 = 839.8 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_AlF3_is_839_8_l3380_338053


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3380_338017

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = 2 → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3380_338017


namespace NUMINAMATH_CALUDE_product_equals_fraction_l3380_338083

theorem product_equals_fraction : 12 * 0.5 * 3 * 0.2 = 18 / 5 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l3380_338083


namespace NUMINAMATH_CALUDE_sample_size_equals_selected_students_l3380_338074

/-- Represents a school with classes and students -/
structure School where
  num_classes : ℕ
  students_per_class : ℕ
  selected_students : ℕ

/-- The sample size of a school's "Student Congress" -/
def sample_size (school : School) : ℕ :=
  school.selected_students

theorem sample_size_equals_selected_students (school : School) 
  (h1 : school.num_classes = 40)
  (h2 : school.students_per_class = 50)
  (h3 : school.selected_students = 150) :
  sample_size school = 150 := by
  sorry

#check sample_size_equals_selected_students

end NUMINAMATH_CALUDE_sample_size_equals_selected_students_l3380_338074


namespace NUMINAMATH_CALUDE_fraction_simplification_l3380_338082

theorem fraction_simplification :
  (3 : ℝ) / (2 * Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = (3 * Real.sqrt 2) / 38 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3380_338082


namespace NUMINAMATH_CALUDE_green_peaches_count_l3380_338010

/-- Given a basket of fruits with the following properties:
  * There are p total fruits
  * There are r red peaches
  * The rest are green peaches
  * The sum of red peaches and twice the green peaches is 3 more than the total fruits
  Then the number of green peaches is always 3 -/
theorem green_peaches_count (p r : ℕ) (h1 : p = r + (p - r)) 
    (h2 : r + 2 * (p - r) = p + 3) : p - r = 3 := by
  sorry

#check green_peaches_count

end NUMINAMATH_CALUDE_green_peaches_count_l3380_338010


namespace NUMINAMATH_CALUDE_unique_solution_l3380_338009

theorem unique_solution : ∃! x : ℝ, 4 * x - 3 = 9 * (x - 7) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3380_338009


namespace NUMINAMATH_CALUDE_student_count_l3380_338023

theorem student_count (stars_per_student : ℕ) (total_stars : ℕ) (h1 : stars_per_student = 3) (h2 : total_stars = 372) :
  total_stars / stars_per_student = 124 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l3380_338023


namespace NUMINAMATH_CALUDE_spade_problem_l3380_338034

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_problem : spade 5 (spade 3 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_spade_problem_l3380_338034


namespace NUMINAMATH_CALUDE_total_cost_of_clothing_l3380_338024

/-- The total cost of a shirt, pants, and shoes given specific pricing conditions -/
theorem total_cost_of_clothing (pants_price : ℝ) : 
  pants_price = 120 →
  let shirt_price := (3/4) * pants_price
  let shoes_price := pants_price + 10
  shirt_price + pants_price + shoes_price = 340 := by
sorry

end NUMINAMATH_CALUDE_total_cost_of_clothing_l3380_338024
