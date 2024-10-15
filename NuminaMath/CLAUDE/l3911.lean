import Mathlib

namespace NUMINAMATH_CALUDE_most_suitable_sampling_plan_l3911_391166

/-- Represents a production line in the factory -/
structure ProductionLine where
  boxes_per_day : ℕ
  deriving Repr

/-- Represents the factory with its production lines -/
structure Factory where
  production_lines : List ProductionLine
  deriving Repr

/-- Represents a sampling plan -/
inductive SamplingPlan
  | RandomOneFromAll
  | LastFromEach
  | RandomOneFromEach
  | AllFromOne
  deriving Repr

/-- Defines what makes a sampling plan suitable -/
def is_suitable_plan (factory : Factory) (plan : SamplingPlan) : Prop :=
  plan = SamplingPlan.RandomOneFromEach

/-- The theorem stating that randomly selecting one box from each production line is the most suitable sampling plan -/
theorem most_suitable_sampling_plan (factory : Factory) 
  (h1 : factory.production_lines.length = 5)
  (h2 : ∀ line ∈ factory.production_lines, line.boxes_per_day = 20) :
  is_suitable_plan factory SamplingPlan.RandomOneFromEach :=
by
  sorry

#check most_suitable_sampling_plan

end NUMINAMATH_CALUDE_most_suitable_sampling_plan_l3911_391166


namespace NUMINAMATH_CALUDE_incorrect_average_calculation_l3911_391142

theorem incorrect_average_calculation (n : ℕ) (correct_num incorrect_num : ℚ) (correct_avg : ℚ) :
  n = 10 ∧ 
  correct_num = 86 ∧ 
  incorrect_num = 26 ∧ 
  correct_avg = 26 →
  (n * correct_avg - correct_num + incorrect_num) / n = 20 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_average_calculation_l3911_391142


namespace NUMINAMATH_CALUDE_units_digit_of_4_pow_8_cubed_l3911_391179

theorem units_digit_of_4_pow_8_cubed (n : ℕ) : n = 4^(8^3) → n % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_4_pow_8_cubed_l3911_391179


namespace NUMINAMATH_CALUDE_product_of_quarters_l3911_391180

theorem product_of_quarters : (0.25 : ℝ) * 0.75 = 0.1875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_quarters_l3911_391180


namespace NUMINAMATH_CALUDE_concatenated_not_palindromic_l3911_391105

/-- Represents the concatenation of integers from 1 to n as a natural number -/
def concatenatedNumber (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is palindromic -/
def isPalindromic (num : ℕ) : Prop := sorry

/-- Theorem stating that the concatenated number is not palindromic for n > 1 -/
theorem concatenated_not_palindromic (n : ℕ) (h : n > 1) : 
  ¬(isPalindromic (concatenatedNumber n)) := by sorry

end NUMINAMATH_CALUDE_concatenated_not_palindromic_l3911_391105


namespace NUMINAMATH_CALUDE_not_perfect_square_8p_plus_1_l3911_391120

theorem not_perfect_square_8p_plus_1 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ¬ ∃ n : ℕ, 8 * p + 1 = (2 * n + 1)^2 := by
sorry

end NUMINAMATH_CALUDE_not_perfect_square_8p_plus_1_l3911_391120


namespace NUMINAMATH_CALUDE_beth_has_winning_strategy_l3911_391100

/-- Represents the state of a wall of bricks -/
structure Wall :=
  (bricks : ℕ)

/-- Represents the game state with multiple walls -/
structure GameState :=
  (walls : List Wall)

/-- Calculates the nim-value of a single wall -/
def nimValue (w : Wall) : ℕ :=
  sorry

/-- Calculates the combined nim-value of a game state -/
def combinedNimValue (gs : GameState) : ℕ :=
  sorry

/-- Determines if a given game state is a winning position for the current player -/
def isWinningPosition (gs : GameState) : Prop :=
  combinedNimValue gs ≠ 0

/-- The initial game state -/
def initialState : GameState :=
  { walls := [{ bricks := 7 }, { bricks := 3 }, { bricks := 2 }] }

theorem beth_has_winning_strategy :
  ¬ isWinningPosition initialState :=
sorry

end NUMINAMATH_CALUDE_beth_has_winning_strategy_l3911_391100


namespace NUMINAMATH_CALUDE_penny_problem_l3911_391168

theorem penny_problem (initial_pennies : ℕ) (older_pennies : ℕ) (removal_percentage : ℚ) :
  initial_pennies = 200 →
  older_pennies = 30 →
  removal_percentage = 1/5 →
  initial_pennies - older_pennies - Int.floor ((initial_pennies - older_pennies : ℚ) * removal_percentage) = 136 :=
by sorry

end NUMINAMATH_CALUDE_penny_problem_l3911_391168


namespace NUMINAMATH_CALUDE_no_valid_rectangle_l3911_391136

theorem no_valid_rectangle (a b x y : ℝ) : 
  a < b → 
  x < a → 
  y < a → 
  2 * (x + y) = (2/3) * (a + b) → 
  x * y = (1/3) * a * b → 
  False := by
sorry

end NUMINAMATH_CALUDE_no_valid_rectangle_l3911_391136


namespace NUMINAMATH_CALUDE_orthogonal_vectors_x_value_l3911_391162

def vector_a (x : ℝ) : Fin 2 → ℝ := ![x, 2]
def vector_b : Fin 2 → ℝ := ![2, -1]

def orthogonal (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

theorem orthogonal_vectors_x_value :
  ∃ x : ℝ, orthogonal (vector_a x) vector_b ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_x_value_l3911_391162


namespace NUMINAMATH_CALUDE_variance_range_best_for_stability_l3911_391114

/-- Represents a set of exam scores -/
def ExamScores := List ℝ

/-- Calculates the variance of a list of numbers -/
def variance (scores : ExamScores) : ℝ := sorry

/-- Calculates the range of a list of numbers -/
def range (scores : ExamScores) : ℝ := sorry

/-- Calculates the mean of a list of numbers -/
def mean (scores : ExamScores) : ℝ := sorry

/-- Calculates the median of a list of numbers -/
def median (scores : ExamScores) : ℝ := sorry

/-- Calculates the mode of a list of numbers -/
def mode (scores : ExamScores) : ℝ := sorry

/-- Measures how well a statistic represents the stability of scores -/
def stabilityMeasure (f : ExamScores → ℝ) : ℝ := sorry

theorem variance_range_best_for_stability (scores : ExamScores) 
  (h : scores.length = 5) :
  (stabilityMeasure variance > stabilityMeasure mean) ∧
  (stabilityMeasure variance > stabilityMeasure median) ∧
  (stabilityMeasure variance > stabilityMeasure mode) ∧
  (stabilityMeasure range > stabilityMeasure mean) ∧
  (stabilityMeasure range > stabilityMeasure median) ∧
  (stabilityMeasure range > stabilityMeasure mode) := by
  sorry

end NUMINAMATH_CALUDE_variance_range_best_for_stability_l3911_391114


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3911_391109

theorem sin_2alpha_value (α : Real) 
  (h : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (2 * α) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3911_391109


namespace NUMINAMATH_CALUDE_arabella_dance_time_l3911_391185

/-- The time Arabella spends learning three dance steps -/
def dance_learning_time (first_step : ℕ) : ℕ :=
  let second_step := first_step / 2
  let third_step := first_step + second_step
  first_step + second_step + third_step

/-- Proof that Arabella spends 90 minutes learning three dance steps -/
theorem arabella_dance_time :
  dance_learning_time 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_arabella_dance_time_l3911_391185


namespace NUMINAMATH_CALUDE_algebraic_expression_symmetry_l3911_391175

theorem algebraic_expression_symmetry (a b : ℝ) : 
  (a * 3^3 + b * 3 - 5 = 20) → (a * (-3)^3 + b * (-3) - 5 = -30) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_symmetry_l3911_391175


namespace NUMINAMATH_CALUDE_triangle_problem_l3911_391198

theorem triangle_problem (a b c A B C : ℝ) 
  (h1 : Real.sqrt 3 * c * Real.sin A = a * Real.cos C)
  (h2 : c = 2 * a)
  (h3 : b = 2 * Real.sqrt 3)
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h5 : 0 < A ∧ A < π)
  (h6 : 0 < B ∧ B < π)
  (h7 : 0 < C ∧ C < π)
  (h8 : A + B + C = π) :
  C = π / 6 ∧ 
  (1/2 * a * b * Real.sin C = (Real.sqrt 15 - Real.sqrt 3) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3911_391198


namespace NUMINAMATH_CALUDE_carltons_outfits_l3911_391170

/-- The number of unique outfit combinations for Carlton -/
def unique_outfit_combinations (button_up_shirts : ℕ) : ℕ :=
  let sweater_vests := 3 * button_up_shirts
  let ties := 2 * sweater_vests
  let shoes := 4 * ties
  let socks := 6 * shoes
  button_up_shirts * sweater_vests * ties * shoes * socks

/-- Theorem stating that Carlton's unique outfit combinations equal 77,760,000 -/
theorem carltons_outfits :
  unique_outfit_combinations 5 = 77760000 := by
  sorry

end NUMINAMATH_CALUDE_carltons_outfits_l3911_391170


namespace NUMINAMATH_CALUDE_parabola_coefficients_l3911_391194

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- Whether a parabola has a vertical axis of symmetry -/
def has_vertical_axis_of_symmetry (p : Parabola) : Prop := sorry

/-- Whether a point (x, y) lies on the parabola -/
def contains_point (p : Parabola) (x y : ℝ) : Prop := 
  y = p.a * x^2 + p.b * x + p.c

theorem parabola_coefficients :
  ∀ p : Parabola,
  vertex p = (2, -1) →
  has_vertical_axis_of_symmetry p →
  contains_point p 0 7 →
  (p.a, p.b, p.c) = (2, -8, 7) := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l3911_391194


namespace NUMINAMATH_CALUDE_angle_difference_range_l3911_391102

/-- Given an acute angle and the absolute difference between this angle and its supplementary angle -/
theorem angle_difference_range (x α : Real) : 
  (0 < x) → (x < 90) → (α = |180 - 2*x|) → (0 < α ∧ α < 180) := by sorry

end NUMINAMATH_CALUDE_angle_difference_range_l3911_391102


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l3911_391159

/-- The quadratic function f(x) = ax^2 + bx -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

/-- The function g(x) = f(x) - x -/
def g (a b : ℝ) (x : ℝ) : ℝ := f a b x - x

theorem quadratic_function_theorem (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b (1 - x) = f a b (1 + x)) →
  (∃! x, g a b x = 0) →
  (f a b = fun x ↦ -1/2 * x^2 + x) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 0, f a b x ∈ Set.Icc (-12 : ℝ) 0) ∧
  (∀ y ∈ Set.Icc (-12 : ℝ) 0, ∃ x ∈ Set.Icc (-4 : ℝ) 0, f a b x = y) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l3911_391159


namespace NUMINAMATH_CALUDE_tooth_extraction_cost_l3911_391173

def cleaning_cost : ℕ := 70
def filling_cost : ℕ := 120
def root_canal_cost : ℕ := 400
def dental_crown_cost : ℕ := 600

def total_known_costs : ℕ := cleaning_cost + 3 * filling_cost + root_canal_cost + dental_crown_cost

def total_bill : ℕ := 9 * root_canal_cost

theorem tooth_extraction_cost : 
  total_bill - total_known_costs = 2170 := by sorry

end NUMINAMATH_CALUDE_tooth_extraction_cost_l3911_391173


namespace NUMINAMATH_CALUDE_quadratic_equation_relation_l3911_391140

theorem quadratic_equation_relation (x : ℝ) : 
  x^2 + 3*x + 5 = 7 → x^2 + 3*x - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_relation_l3911_391140


namespace NUMINAMATH_CALUDE_divisor_sum_product_2016_l3911_391196

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem divisor_sum_product_2016 (n : ℕ) :
  n % 2 = 0 →
  (sum_odd_divisors n) * (sum_even_divisors n) = 2016 ↔ n = 192 ∨ n = 88 := by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_product_2016_l3911_391196


namespace NUMINAMATH_CALUDE_grapes_purchased_l3911_391137

theorem grapes_purchased (grape_price : ℕ) (mango_quantity : ℕ) (mango_price : ℕ) (total_paid : ℕ) :
  grape_price = 70 →
  mango_quantity = 9 →
  mango_price = 55 →
  total_paid = 705 →
  ∃ grape_quantity : ℕ, grape_quantity * grape_price + mango_quantity * mango_price = total_paid ∧ grape_quantity = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_grapes_purchased_l3911_391137


namespace NUMINAMATH_CALUDE_grocery_theorem_l3911_391187

def grocery_problem (initial_budget : ℚ) (bread_cost : ℚ) (candy_cost : ℚ) (final_remaining : ℚ) : Prop :=
  let remaining_after_bread_candy := initial_budget - (bread_cost + candy_cost)
  let spent_on_turkey := remaining_after_bread_candy - final_remaining
  spent_on_turkey / remaining_after_bread_candy = 1 / 3

theorem grocery_theorem :
  grocery_problem 32 3 2 18 := by
  sorry

end NUMINAMATH_CALUDE_grocery_theorem_l3911_391187


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3911_391118

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a parabola y^2 = 8x and a line passing through P(1, -1) intersecting 
    the parabola at points A and B, where P is the midpoint of AB, 
    prove that the equation of line AB is 4x + y - 3 = 0 -/
theorem parabola_line_intersection 
  (para : Parabola) 
  (P : Point) 
  (A B : Point) 
  (line : Line) : 
  para.p = 4 → 
  P.x = 1 → 
  P.y = -1 → 
  (A.x + B.x) / 2 = P.x → 
  (A.y + B.y) / 2 = P.y → 
  A.y^2 = 8 * A.x → 
  B.y^2 = 8 * B.x → 
  line.a * A.x + line.b * A.y + line.c = 0 → 
  line.a * B.x + line.b * B.y + line.c = 0 → 
  line.a = 4 ∧ line.b = 1 ∧ line.c = -3 := by 
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3911_391118


namespace NUMINAMATH_CALUDE_green_space_equation_l3911_391177

/-- Represents a rectangular green space -/
structure GreenSpace where
  length : ℝ
  width : ℝ
  area : ℝ

/-- Theorem stating the properties of the green space and the resulting equation -/
theorem green_space_equation (g : GreenSpace) 
  (h1 : g.area = 1000)
  (h2 : g.length = g.width + 30)
  (h3 : g.area = g.length * g.width) :
  g.length * (g.length - 30) = 1000 := by
  sorry

#check green_space_equation

end NUMINAMATH_CALUDE_green_space_equation_l3911_391177


namespace NUMINAMATH_CALUDE_square_sum_equals_twenty_l3911_391181

theorem square_sum_equals_twenty (x y : ℝ) 
  (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_twenty_l3911_391181


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3911_391115

/-- A right triangle with perimeter 60 and area 120 has a hypotenuse of length 26. -/
theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a + b + c = 60 →
  (1/2) * a * b = 120 →
  c = 26 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3911_391115


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3911_391188

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ complement_B = Set.Icc 2 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3911_391188


namespace NUMINAMATH_CALUDE_tan_ratio_problem_l3911_391128

theorem tan_ratio_problem (x : Real) (h : Real.tan (x + π / 4) = 2) :
  (Real.tan x) / (Real.tan (2 * x)) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_problem_l3911_391128


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_two_l3911_391108

-- Define the base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_two :
  lg 4 + lg 9 + 2 * Real.sqrt ((lg 6)^2 - lg 36 + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_two_l3911_391108


namespace NUMINAMATH_CALUDE_fermatville_temperature_range_l3911_391121

/-- The temperature range in Fermatville on Monday -/
def temperature_range (min_temp max_temp : Int) : Int :=
  max_temp - min_temp

/-- Theorem: The temperature range in Fermatville on Monday was 25°C -/
theorem fermatville_temperature_range :
  let min_temp : Int := -11
  let max_temp : Int := 14
  temperature_range min_temp max_temp = 25 := by
  sorry

end NUMINAMATH_CALUDE_fermatville_temperature_range_l3911_391121


namespace NUMINAMATH_CALUDE_weights_division_l3911_391167

theorem weights_division (n : ℕ) (h : n ≥ 3) :
  (∃ (a b c : Finset ℕ), a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ b ∩ c = ∅ ∧
    a ∪ b ∪ c = Finset.range (n + 1) \ {0} ∧
    (a.sum id = b.sum id) ∧ (b.sum id = c.sum id)) ↔
  (∃ k : ℕ, (n = 3 * k + 2 ∨ n = 3 * k + 3) ∧ k > 0) :=
by sorry

end NUMINAMATH_CALUDE_weights_division_l3911_391167


namespace NUMINAMATH_CALUDE_equation_proof_l3911_391112

theorem equation_proof : 289 + 2 * 17 * 5 + 25 = 484 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3911_391112


namespace NUMINAMATH_CALUDE_spinner_probability_l3911_391111

theorem spinner_probability (p_A p_B p_C p_D p_E : ℚ) :
  p_A = 3/8 →
  p_B = 1/4 →
  p_C = p_D →
  p_C = p_E →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 1/8 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l3911_391111


namespace NUMINAMATH_CALUDE_candy_problem_l3911_391151

theorem candy_problem (given_away eaten remaining : ℕ) 
  (h1 : given_away = 18)
  (h2 : eaten = 7)
  (h3 : remaining = 16) :
  given_away + eaten + remaining = 41 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l3911_391151


namespace NUMINAMATH_CALUDE_root_range_theorem_l3911_391116

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m-1)*x + m^2 - 2

theorem root_range_theorem (m : ℝ) : 
  (∃ x y : ℝ, x < -1 ∧ y > 1 ∧ 
   f m x = 0 ∧ f m y = 0 ∧ 
   ∀ z : ℝ, f m z = 0 → z = x ∨ z = y) ↔ 
  m > 0 ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_root_range_theorem_l3911_391116


namespace NUMINAMATH_CALUDE_age_difference_l3911_391176

theorem age_difference (a b : ℕ) (h1 : b = 38) (h2 : a + 10 = 2 * (b - 10)) : a - b = 8 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3911_391176


namespace NUMINAMATH_CALUDE_ellipse_equation_l3911_391171

/-- Given an ellipse C with equation x²/a² + y²/b² = 1, where a > b > 0, and the major axis is √2
    times the minor axis, if the line y = -x + 1 intersects the ellipse at points A and B such that
    the length of chord AB is 4√5/3, then the equation of the ellipse is x²/4 + y²/2 = 1. -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (h3 : a^2 = 2 * b^2) 
    (h4 : ∃ (x1 y1 x2 y2 : ℝ), 
      x1^2/a^2 + y1^2/b^2 = 1 ∧ 
      x2^2/a^2 + y2^2/b^2 = 1 ∧
      y1 = -x1 + 1 ∧ 
      y2 = -x2 + 1 ∧ 
      (x2 - x1)^2 + (y2 - y1)^2 = (4*Real.sqrt 5/3)^2) :
  ∀ x y : ℝ, x^2/4 + y^2/2 = 1 ↔ x^2/a^2 + y^2/b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3911_391171


namespace NUMINAMATH_CALUDE_negative_64_to_four_thirds_equals_256_l3911_391103

theorem negative_64_to_four_thirds_equals_256 : (-64 : ℝ) ^ (4/3) = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_64_to_four_thirds_equals_256_l3911_391103


namespace NUMINAMATH_CALUDE_class_selection_ways_l3911_391143

def total_classes : ℕ := 10
def advanced_classes : ℕ := 6
def intro_classes : ℕ := 4
def classes_to_choose : ℕ := 5
def min_advanced : ℕ := 3

theorem class_selection_ways :
  (Nat.choose advanced_classes 3 * Nat.choose intro_classes 2) +
  (Nat.choose advanced_classes 4 * Nat.choose intro_classes 1) +
  (Nat.choose advanced_classes 5) = 186 := by
  sorry

end NUMINAMATH_CALUDE_class_selection_ways_l3911_391143


namespace NUMINAMATH_CALUDE_prob_select_copresidents_value_l3911_391144

/-- Represents a math club with a given number of students -/
structure MathClub where
  students : ℕ
  co_presidents : Fin 2
  vice_president : Fin 1

/-- The set of math clubs in the school district -/
def school_clubs : Finset MathClub := sorry

/-- The probability of selecting both co-presidents when randomly selecting 
    three members from a randomly selected club -/
def prob_select_copresidents (clubs : Finset MathClub) : ℚ := sorry

theorem prob_select_copresidents_value : 
  prob_select_copresidents school_clubs = 43 / 420 := by sorry

end NUMINAMATH_CALUDE_prob_select_copresidents_value_l3911_391144


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_169_l3911_391145

theorem factor_x_squared_minus_169 (x : ℝ) : x^2 - 169 = (x - 13) * (x + 13) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_169_l3911_391145


namespace NUMINAMATH_CALUDE_defeat_points_is_zero_l3911_391189

/-- Represents the point system for a football competition -/
structure PointSystem where
  victory_points : ℕ
  draw_points : ℕ
  defeat_points : ℕ

/-- Represents the state of a team's performance -/
structure TeamPerformance where
  total_matches : ℕ
  matches_played : ℕ
  points : ℕ
  victories : ℕ
  draws : ℕ
  defeats : ℕ

/-- Theorem stating that the number of points for a defeat must be 0 -/
theorem defeat_points_is_zero 
  (ps : PointSystem) 
  (tp : TeamPerformance) 
  (h1 : ps.victory_points = 3)
  (h2 : ps.draw_points = 1)
  (h3 : tp.total_matches = 20)
  (h4 : tp.matches_played = 5)
  (h5 : tp.points = 8)
  (h6 : ∀ (future_victories : ℕ), 
        future_victories ≥ 9 → 
        tp.points + future_victories * ps.victory_points + 
        (tp.total_matches - tp.matches_played - future_victories) * ps.defeat_points ≥ 40) :
  ps.defeat_points = 0 := by
sorry

end NUMINAMATH_CALUDE_defeat_points_is_zero_l3911_391189


namespace NUMINAMATH_CALUDE_parallel_transitivity_l3911_391107

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define a relation for parallel lines
def Parallel (l₁ l₂ : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitivity (l₁ l₂ l₃ : Line) :
  Parallel l₁ l₃ → Parallel l₂ l₃ → Parallel l₁ l₂ :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l3911_391107


namespace NUMINAMATH_CALUDE_circle_perimeter_l3911_391165

theorem circle_perimeter (r : ℝ) (h : r = 4 / Real.pi) : 
  2 * Real.pi * r = 8 := by sorry

end NUMINAMATH_CALUDE_circle_perimeter_l3911_391165


namespace NUMINAMATH_CALUDE_inequality_proof_l3911_391147

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hne : ¬(x = y ∧ y = z)) : 
  (x + y) * (y + z) * (z + x) > 8 * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3911_391147


namespace NUMINAMATH_CALUDE_intersection_and_complement_intersection_l3911_391125

def I : Set ℕ := Set.univ

def A : Set ℕ := {x | ∃ n : ℕ, x = 3 * n ∧ n % 2 = 0}

def B : Set ℕ := {y | 24 % y = 0}

theorem intersection_and_complement_intersection :
  (A ∩ B = {6, 12, 24}) ∧
  ((I \ A) ∩ B = {1, 2, 3, 4, 8}) := by sorry

end NUMINAMATH_CALUDE_intersection_and_complement_intersection_l3911_391125


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_xy_l3911_391127

theorem factorization_x_squared_minus_xy (x y : ℝ) : x^2 - x*y = x*(x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_xy_l3911_391127


namespace NUMINAMATH_CALUDE_exponential_simplification_l3911_391132

theorem exponential_simplification :
  (10 ^ 1.4) * (10 ^ 0.5) / ((10 ^ 0.4) * (10 ^ 0.1)) = 10 ^ 1.4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_simplification_l3911_391132


namespace NUMINAMATH_CALUDE_car_speed_increase_l3911_391178

/-- Proves that the percentage increase in car Y's average speed compared to car Q's speed is 50% -/
theorem car_speed_increase (distance : ℝ) (time_Q time_Y : ℝ) 
  (h1 : distance = 80)
  (h2 : time_Q = 2)
  (h3 : time_Y = 1.3333333333333333)
  (h4 : distance / time_Y > distance / time_Q) :
  (distance / time_Y - distance / time_Q) / (distance / time_Q) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_increase_l3911_391178


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l3911_391113

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 60 ∧ x - y = 10 → x * y = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l3911_391113


namespace NUMINAMATH_CALUDE_distance_between_foci_l3911_391191

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 25

-- Define the foci
def focus1 : ℝ × ℝ := (4, 5)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem: The distance between the foci of the ellipse is 2√29
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 2 * Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_foci_l3911_391191


namespace NUMINAMATH_CALUDE_parakeet_to_kitten_ratio_l3911_391155

-- Define the number of each type of pet
def num_puppies : ℕ := 2
def num_kittens : ℕ := 2
def num_parakeets : ℕ := 3

-- Define the cost of a parakeet
def parakeet_cost : ℕ := 10

-- Define the relationship between puppy and parakeet costs
def puppy_cost : ℕ := 3 * parakeet_cost

-- Define the total cost of all pets
def total_cost : ℕ := 130

-- Define the cost of a kitten (to be proved)
def kitten_cost : ℕ := (total_cost - num_puppies * puppy_cost - num_parakeets * parakeet_cost) / num_kittens

-- Theorem to prove the ratio of parakeet cost to kitten cost
theorem parakeet_to_kitten_ratio :
  parakeet_cost * 2 = kitten_cost :=
by sorry

end NUMINAMATH_CALUDE_parakeet_to_kitten_ratio_l3911_391155


namespace NUMINAMATH_CALUDE_largest_divisor_of_even_square_diff_l3911_391160

theorem largest_divisor_of_even_square_diff (m n : ℤ) : 
  Even m → Even n → n < m → 
  (∃ (k : ℤ), ∀ (a b : ℤ), (Even a ∧ Even b ∧ b < a) → 
    k ∣ (a^2 - b^2) ∧ 
    (∀ (l : ℤ), (∀ (x y : ℤ), (Even x ∧ Even y ∧ y < x) → l ∣ (x^2 - y^2)) → l ≤ k)) → 
  (∃ (k : ℤ), ∀ (a b : ℤ), (Even a ∧ Even b ∧ b < a) → 
    k ∣ (a^2 - b^2) ∧ 
    (∀ (l : ℤ), (∀ (x y : ℤ), (Even x ∧ Even y ∧ y < x) → l ∣ (x^2 - y^2)) → l ≤ k)) ∧ 
  (∃ (k : ℤ), ∀ (a b : ℤ), (Even a ∧ Even b ∧ b < a) → 
    k ∣ (a^2 - b^2) ∧ 
    (∀ (l : ℤ), (∀ (x y : ℤ), (Even x ∧ Even y ∧ y < x) → l ∣ (x^2 - y^2)) → l ≤ k)) → k = 4 :=
by sorry


end NUMINAMATH_CALUDE_largest_divisor_of_even_square_diff_l3911_391160


namespace NUMINAMATH_CALUDE_average_marks_math_chem_l3911_391130

theorem average_marks_math_chem (math physics chem : ℕ) : 
  math + physics = 60 →
  chem = physics + 20 →
  (math + chem) / 2 = 40 :=
by sorry

end NUMINAMATH_CALUDE_average_marks_math_chem_l3911_391130


namespace NUMINAMATH_CALUDE_least_number_of_trees_l3911_391150

theorem least_number_of_trees : ∃ n : ℕ, 
  (∀ m : ℕ, m < n → (m % 7 ≠ 0 ∨ m % 6 ≠ 0 ∨ m % 4 ≠ 0)) ∧
  n % 7 = 0 ∧ n % 6 = 0 ∧ n % 4 = 0 ∧
  n = 84 := by
sorry

end NUMINAMATH_CALUDE_least_number_of_trees_l3911_391150


namespace NUMINAMATH_CALUDE_binomial_floor_divisibility_l3911_391148

theorem binomial_floor_divisibility (p n : ℕ) (hp : Nat.Prime p) :
  p ∣ (Nat.choose n p - n / p) := by
  sorry

end NUMINAMATH_CALUDE_binomial_floor_divisibility_l3911_391148


namespace NUMINAMATH_CALUDE_average_weight_of_twenty_boys_l3911_391146

theorem average_weight_of_twenty_boys 
  (num_group1 : ℕ) 
  (num_group2 : ℕ) 
  (avg_weight_group2 : ℝ) 
  (avg_weight_all : ℝ) :
  num_group1 = 20 →
  num_group2 = 8 →
  avg_weight_group2 = 45.15 →
  avg_weight_all = 48.792857142857144 →
  (num_group1 * 50.25 + num_group2 * avg_weight_group2) / (num_group1 + num_group2) = avg_weight_all :=
by sorry

end NUMINAMATH_CALUDE_average_weight_of_twenty_boys_l3911_391146


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l3911_391129

theorem gcd_of_specific_numbers : 
  let m : ℕ := 55555555
  let n : ℕ := 111111111
  Nat.gcd m n = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l3911_391129


namespace NUMINAMATH_CALUDE_car_overtake_distance_l3911_391135

theorem car_overtake_distance (speed_a speed_b time_to_overtake distance_ahead : ℝ) 
  (h1 : speed_a = 58)
  (h2 : speed_b = 50)
  (h3 : time_to_overtake = 4)
  (h4 : distance_ahead = 8) :
  speed_a * time_to_overtake - speed_b * time_to_overtake - distance_ahead = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_overtake_distance_l3911_391135


namespace NUMINAMATH_CALUDE_new_cube_weight_l3911_391153

/-- Given a cube of weight 3 pounds and density D, prove that a new cube with sides twice as long
    and density 1.25D will weigh 30 pounds. -/
theorem new_cube_weight (D : ℝ) (D_pos : D > 0) : 
  let original_weight : ℝ := 3
  let original_volume : ℝ := original_weight / D
  let new_volume : ℝ := 8 * original_volume
  let new_density : ℝ := 1.25 * D
  new_density * new_volume = 30 := by
  sorry


end NUMINAMATH_CALUDE_new_cube_weight_l3911_391153


namespace NUMINAMATH_CALUDE_rectangle_area_l3911_391133

theorem rectangle_area (r : ℝ) (ratio : ℝ) : r = 6 ∧ ratio = 3 →
  ∃ (length width : ℝ),
    width = 2 * r ∧
    length = ratio * width ∧
    length * width = 432 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3911_391133


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3911_391152

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 15*a^2 + 13*a - 8 = 0 → 
  b^3 - 15*b^2 + 13*b - 8 = 0 → 
  c^3 - 15*c^2 + 13*c - 8 = 0 → 
  a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 199/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3911_391152


namespace NUMINAMATH_CALUDE_charge_account_interest_l3911_391169

/-- Calculates the total amount owed after one year with simple interest -/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that the total amount owed after one year is $38.15 -/
theorem charge_account_interest :
  let principal : ℝ := 35
  let rate : ℝ := 0.09
  let time : ℝ := 1
  total_amount_owed principal rate time = 38.15 := by
sorry

end NUMINAMATH_CALUDE_charge_account_interest_l3911_391169


namespace NUMINAMATH_CALUDE_min_value_theorem_l3911_391122

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b + a * c = 4) :
  (2 / a) + (2 / (b + c)) + (8 / (a + b + c)) ≥ 4 ∧
  ((2 / a) + (2 / (b + c)) + (8 / (a + b + c)) = 4 ↔ a = 2 ∧ b + c = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3911_391122


namespace NUMINAMATH_CALUDE_perfect_squares_mod_seven_l3911_391139

theorem perfect_squares_mod_seven :
  ∃! (S : Finset ℕ), (∀ n ∈ S, ∃ m : ℤ, (m ^ 2 : ℤ) % 7 = n) ∧
                     (∀ k : ℤ, ∃ n ∈ S, (k ^ 2 : ℤ) % 7 = n) ∧
                     S.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_mod_seven_l3911_391139


namespace NUMINAMATH_CALUDE_clock_strike_time_l3911_391124

theorem clock_strike_time (strike_three : ℕ) (time_three : ℝ) (strike_six : ℕ) : 
  strike_three = 3 → time_three = 12 → strike_six = 6 → 
  ∃ (time_six : ℝ), time_six = 30 := by
  sorry

end NUMINAMATH_CALUDE_clock_strike_time_l3911_391124


namespace NUMINAMATH_CALUDE_license_plate_combinations_eq_960_l3911_391117

/-- Represents the set of possible characters for each position in the license plate --/
def LicensePlateChoices : Fin 5 → Finset Char :=
  fun i => match i with
    | 0 => {'3', '5', '6', '8', '9'}
    | 1 => {'B', 'C', 'D'}
    | _ => {'1', '3', '6', '9'}

/-- The number of possible license plate combinations --/
def LicensePlateCombinations : ℕ :=
  (LicensePlateChoices 0).card *
  (LicensePlateChoices 1).card *
  (LicensePlateChoices 2).card *
  (LicensePlateChoices 3).card *
  (LicensePlateChoices 4).card

/-- Theorem stating that the number of possible license plate combinations is 960 --/
theorem license_plate_combinations_eq_960 :
  LicensePlateCombinations = 960 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_eq_960_l3911_391117


namespace NUMINAMATH_CALUDE_sin_fourth_power_decomposition_l3911_391183

theorem sin_fourth_power_decomposition :
  ∃ (b₁ b₂ b₃ b₄ : ℝ),
    (∀ θ : ℝ, Real.sin θ ^ 4 = b₁ * Real.sin θ + b₂ * Real.sin (2 * θ) + b₃ * Real.sin (3 * θ) + b₄ * Real.sin (4 * θ)) →
    b₁^2 + b₂^2 + b₃^2 + b₄^2 = 17 / 64 :=
by sorry

end NUMINAMATH_CALUDE_sin_fourth_power_decomposition_l3911_391183


namespace NUMINAMATH_CALUDE_pentagon_area_l3911_391119

/-- The area of a pentagon with sides 18, 25, 30, 28, and 25 units is 1020 square units. -/
theorem pentagon_area : ℝ := by
  -- Define the pentagon
  let side1 : ℝ := 18
  let side2 : ℝ := 25
  let side3 : ℝ := 30
  let side4 : ℝ := 28
  let side5 : ℝ := 25

  -- Define the area of the pentagon
  let pentagon_area : ℝ := 1020

  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_pentagon_area_l3911_391119


namespace NUMINAMATH_CALUDE_expression_evaluation_l3911_391174

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3911_391174


namespace NUMINAMATH_CALUDE_divisor_of_power_of_four_l3911_391193

theorem divisor_of_power_of_four (a : ℕ) (d : ℕ) (h1 : a > 0) (h2 : 2 ∣ a) :
  let p := 4^a
  (p % d = 6) → d = 22 := by
sorry

end NUMINAMATH_CALUDE_divisor_of_power_of_four_l3911_391193


namespace NUMINAMATH_CALUDE_sixth_root_square_equation_solution_l3911_391192

theorem sixth_root_square_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ (((x * (x^4)^(1/2))^(1/6))^2 = 4) ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_sixth_root_square_equation_solution_l3911_391192


namespace NUMINAMATH_CALUDE_angle_bisector_length_l3911_391182

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  dist A B = 5 ∧ dist B C = 7 ∧ dist A C = 8

-- Define the angle bisector BD
def is_angle_bisector (A B C D : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), x / y = 5 / 7 ∧ x + y = 8 ∧ 
  dist A D = x ∧ dist D C = y

-- Main theorem
theorem angle_bisector_length 
  (A B C D : ℝ × ℝ) 
  (h1 : triangle_ABC A B C) 
  (h2 : is_angle_bisector A B C D) 
  (h3 : ∃ (k : ℝ), dist B D = k * Real.sqrt 3) : 
  ∃ (k : ℝ), dist B D = k * Real.sqrt 3 ∧ k = 5 / 3 :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l3911_391182


namespace NUMINAMATH_CALUDE_pastry_price_is_18_l3911_391161

/-- The price of a pastry satisfies the given conditions -/
theorem pastry_price_is_18 
  (usual_pastries : ℕ) 
  (usual_bread : ℕ) 
  (today_pastries : ℕ) 
  (today_bread : ℕ) 
  (bread_price : ℕ) 
  (daily_difference : ℕ) 
  (h1 : usual_pastries = 20) 
  (h2 : usual_bread = 10) 
  (h3 : today_pastries = 14) 
  (h4 : today_bread = 25) 
  (h5 : bread_price = 4) 
  (h6 : daily_difference = 48) : 
  ∃ (pastry_price : ℕ), 
    pastry_price = 18 ∧ 
    usual_pastries * pastry_price + usual_bread * bread_price - 
    (today_pastries * pastry_price + today_bread * bread_price) = daily_difference :=
by sorry

end NUMINAMATH_CALUDE_pastry_price_is_18_l3911_391161


namespace NUMINAMATH_CALUDE_special_line_properties_l3911_391110

/-- A line passing through (5, 2) with y-intercept twice its x-intercept -/
def special_line (x y : ℝ) : Prop :=
  2 * x - 5 * y + 60 = 0

theorem special_line_properties :
  (special_line 5 2) ∧ 
  (∃ (b : ℝ), special_line 0 b ∧ special_line (b/2) 0 ∧ b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_special_line_properties_l3911_391110


namespace NUMINAMATH_CALUDE_cards_per_deck_l3911_391157

theorem cards_per_deck 
  (num_decks : ℕ) 
  (num_layers : ℕ) 
  (cards_per_layer : ℕ) 
  (h1 : num_decks = 16) 
  (h2 : num_layers = 32) 
  (h3 : cards_per_layer = 26) : 
  (num_layers * cards_per_layer) / num_decks = 52 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_deck_l3911_391157


namespace NUMINAMATH_CALUDE_remainder_97_pow_25_mod_50_l3911_391197

theorem remainder_97_pow_25_mod_50 : 97^25 % 50 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_97_pow_25_mod_50_l3911_391197


namespace NUMINAMATH_CALUDE_identity_element_is_negative_four_l3911_391163

-- Define the operation ⊕
def circplus (a b : ℝ) : ℝ := a + b + 4

-- Define the property of being an identity element for ⊕
def is_identity (e : ℝ) : Prop :=
  ∀ a : ℝ, circplus e a = a

-- Theorem statement
theorem identity_element_is_negative_four :
  ∃ e : ℝ, is_identity e ∧ e = -4 := by
  sorry

end NUMINAMATH_CALUDE_identity_element_is_negative_four_l3911_391163


namespace NUMINAMATH_CALUDE_number_of_divisors_3003_l3911_391126

theorem number_of_divisors_3003 : Nat.card (Nat.divisors 3003) = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_3003_l3911_391126


namespace NUMINAMATH_CALUDE_joes_height_l3911_391154

theorem joes_height (sara_height joe_height : ℕ) : 
  sara_height + joe_height = 120 →
  joe_height = 2 * sara_height + 6 →
  joe_height = 82 :=
by
  sorry

end NUMINAMATH_CALUDE_joes_height_l3911_391154


namespace NUMINAMATH_CALUDE_lara_likes_one_last_digit_l3911_391195

theorem lara_likes_one_last_digit :
  ∃! d : ℕ, d < 10 ∧ ∀ n : ℕ, (n % 3 = 0 ∧ n % 5 = 0) → n % 10 = d :=
sorry

end NUMINAMATH_CALUDE_lara_likes_one_last_digit_l3911_391195


namespace NUMINAMATH_CALUDE_regular_hexagon_area_l3911_391190

/-- The area of a regular hexagon with vertices A(0,0) and C(6,2) is 20√3 -/
theorem regular_hexagon_area : 
  let A : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (6, 2)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let s : ℝ := AC / 2
  let hexagon_area : ℝ := 3 * Real.sqrt 3 * s^2 / 2
  hexagon_area = 20 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_regular_hexagon_area_l3911_391190


namespace NUMINAMATH_CALUDE_andy_late_demerits_l3911_391104

/-- The maximum number of demerits Andy can get before being fired -/
def max_demerits : ℕ := 50

/-- The number of times Andy showed up late -/
def late_instances : ℕ := 6

/-- The number of demerits Andy got for making an inappropriate joke -/
def joke_demerits : ℕ := 15

/-- The number of additional demerits Andy can get this month before being fired -/
def remaining_demerits : ℕ := 23

/-- The number of demerits Andy gets per instance of being late -/
def demerits_per_late_instance : ℕ := 2

theorem andy_late_demerits :
  late_instances * demerits_per_late_instance + joke_demerits = max_demerits - remaining_demerits :=
sorry

end NUMINAMATH_CALUDE_andy_late_demerits_l3911_391104


namespace NUMINAMATH_CALUDE_backup_settings_count_l3911_391123

/-- Represents the weight of a single piece of silverware in ounces -/
def silverware_weight : ℕ := 4

/-- Represents the number of silverware pieces per setting -/
def silverware_per_setting : ℕ := 3

/-- Represents the weight of a single plate in ounces -/
def plate_weight : ℕ := 12

/-- Represents the number of plates per setting -/
def plates_per_setting : ℕ := 2

/-- Represents the number of tables -/
def num_tables : ℕ := 15

/-- Represents the number of settings per table -/
def settings_per_table : ℕ := 8

/-- Represents the total weight of all settings including backups in ounces -/
def total_weight : ℕ := 5040

/-- Calculates the number of backup settings needed -/
def backup_settings : ℕ := 
  let setting_weight := silverware_weight * silverware_per_setting + plate_weight * plates_per_setting
  let total_settings := num_tables * settings_per_table
  let regular_settings_weight := total_settings * setting_weight
  (total_weight - regular_settings_weight) / setting_weight

theorem backup_settings_count : backup_settings = 20 := by
  sorry

end NUMINAMATH_CALUDE_backup_settings_count_l3911_391123


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3911_391164

theorem max_value_sqrt_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 1) :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 1 →
    Real.sqrt (4*x + 1) + Real.sqrt (4*y + 1) + Real.sqrt (4*z + 1) ≤ Real.sqrt (4*a + 1) + Real.sqrt (4*b + 1) + Real.sqrt (4*c + 1)) ∧
  Real.sqrt (4*a + 1) + Real.sqrt (4*b + 1) + Real.sqrt (4*c + 1) = Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3911_391164


namespace NUMINAMATH_CALUDE_simplify_fraction_division_l3911_391138

theorem simplify_fraction_division (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 2) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 3*x + 2) / (x^2 - 4*x + 4)) = (x - 2) / (x - 3) :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_division_l3911_391138


namespace NUMINAMATH_CALUDE_jellybean_probability_l3911_391156

def total_jellybeans : ℕ := 15
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 3
def white_jellybeans : ℕ := 5
def green_jellybeans : ℕ := 2
def picked_jellybeans : ℕ := 4

theorem jellybean_probability : 
  (Nat.choose red_jellybeans 3 * Nat.choose (total_jellybeans - red_jellybeans) 1) / 
  Nat.choose total_jellybeans picked_jellybeans = 20 / 273 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_probability_l3911_391156


namespace NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l3911_391149

/-- Calculates the profit percentage of a middleman in a series of transactions -/
theorem cricket_bat_profit_percentage 
  (a_cost : ℝ) 
  (a_profit_percent : ℝ) 
  (c_price : ℝ) 
  (h1 : a_cost = 152)
  (h2 : a_profit_percent = 20)
  (h3 : c_price = 228) :
  let a_sell := a_cost * (1 + a_profit_percent / 100)
  let b_profit := c_price - a_sell
  let b_profit_percent := (b_profit / a_sell) * 100
  b_profit_percent = 25 := by
sorry


end NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l3911_391149


namespace NUMINAMATH_CALUDE_paco_sweet_cookies_left_l3911_391199

/-- The number of sweet cookies Paco has left after eating some -/
def sweet_cookies_left (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem: Paco has 19 sweet cookies left -/
theorem paco_sweet_cookies_left : 
  sweet_cookies_left 34 15 = 19 := by
  sorry

end NUMINAMATH_CALUDE_paco_sweet_cookies_left_l3911_391199


namespace NUMINAMATH_CALUDE_range_of_a_l3911_391172

theorem range_of_a (a : ℝ) : (∀ x > 0, 4 * a > x^2 - x^3) → a > 1/27 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3911_391172


namespace NUMINAMATH_CALUDE_unique_factorial_sum_l3911_391101

theorem unique_factorial_sum (n : ℕ) : 2 * n * n.factorial + n.factorial = 2520 ↔ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_factorial_sum_l3911_391101


namespace NUMINAMATH_CALUDE_sum_of_xyz_l3911_391184

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 30) (hxz : x * z = 60) (hyz : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l3911_391184


namespace NUMINAMATH_CALUDE_greatest_x_value_l3911_391131

theorem greatest_x_value (x : ℝ) : 
  (2 * x^2 + 7 * x + 3 = 5) → x ≤ (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3911_391131


namespace NUMINAMATH_CALUDE_difference_positive_inequality_l3911_391186

theorem difference_positive_inequality (x : ℝ) :
  (1 / 3 * x - x > 0) ↔ (-2 / 3 * x > 0) := by sorry

end NUMINAMATH_CALUDE_difference_positive_inequality_l3911_391186


namespace NUMINAMATH_CALUDE_brookes_social_studies_problems_l3911_391141

/-- Calculates the number of social studies problems in Brooke's homework -/
theorem brookes_social_studies_problems :
  ∀ (math_problems science_problems : ℕ)
    (math_time social_studies_time science_time total_time : ℚ),
  math_problems = 15 →
  science_problems = 10 →
  math_time = 2 →
  social_studies_time = 1/2 →
  science_time = 3/2 →
  total_time = 48 →
  ∃ (social_studies_problems : ℕ),
    social_studies_problems = 6 ∧
    (math_problems : ℚ) * math_time +
    (social_studies_problems : ℚ) * social_studies_time +
    (science_problems : ℚ) * science_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_brookes_social_studies_problems_l3911_391141


namespace NUMINAMATH_CALUDE_partner_p_investment_time_l3911_391134

/-- Represents the investment and profit data for two partners -/
structure PartnershipData where
  investment_ratio_p : ℚ
  investment_ratio_q : ℚ
  profit_ratio_p : ℚ
  profit_ratio_q : ℚ
  investment_time_q : ℚ

/-- Calculates the investment time for partner p given the partnership data -/
def calculate_investment_time_p (data : PartnershipData) : ℚ :=
  (data.investment_ratio_q * data.profit_ratio_p * data.investment_time_q) /
  (data.investment_ratio_p * data.profit_ratio_q)

/-- Theorem stating that given the specific partnership data, partner p's investment time is 5 months -/
theorem partner_p_investment_time :
  let data : PartnershipData := {
    investment_ratio_p := 7,
    investment_ratio_q := 5,
    profit_ratio_p := 7,
    profit_ratio_q := 12,
    investment_time_q := 12
  }
  calculate_investment_time_p data = 5 := by sorry

end NUMINAMATH_CALUDE_partner_p_investment_time_l3911_391134


namespace NUMINAMATH_CALUDE_cost_per_set_is_correct_l3911_391106

/-- The cost of each set of drill bits -/
def cost_per_set : ℝ := 6

/-- The number of sets bought -/
def num_sets : ℕ := 5

/-- The tax rate -/
def tax_rate : ℝ := 0.1

/-- The total amount paid -/
def total_paid : ℝ := 33

/-- Theorem stating that the cost per set is correct given the conditions -/
theorem cost_per_set_is_correct : 
  num_sets * cost_per_set * (1 + tax_rate) = total_paid :=
sorry

end NUMINAMATH_CALUDE_cost_per_set_is_correct_l3911_391106


namespace NUMINAMATH_CALUDE_intercept_sum_zero_line_equation_l3911_391158

/-- A line passing through a point with sum of intercepts equal to zero -/
structure InterceptSumZeroLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through the point (1,4) -/
  passes_through_point : slope + y_intercept = 4
  /-- The sum of x and y intercepts is zero -/
  sum_of_intercepts_zero : (- y_intercept / slope) + y_intercept = 0

/-- The equation of the line is either 4x-y=0 or x-y+3=0 -/
theorem intercept_sum_zero_line_equation (line : InterceptSumZeroLine) :
  (line.slope = 4 ∧ line.y_intercept = 0) ∨ (line.slope = 1 ∧ line.y_intercept = 3) :=
sorry

end NUMINAMATH_CALUDE_intercept_sum_zero_line_equation_l3911_391158
