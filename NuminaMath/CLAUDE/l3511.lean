import Mathlib

namespace NUMINAMATH_CALUDE_tan_210_degrees_l3511_351119

theorem tan_210_degrees : Real.tan (210 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_210_degrees_l3511_351119


namespace NUMINAMATH_CALUDE_shenile_score_theorem_l3511_351116

/-- Represents the number of points Shenille scored in a basketball game -/
def shenilesScore (threePointAttempts twoPointAttempts : ℕ) : ℝ :=
  0.6 * (threePointAttempts + twoPointAttempts)

theorem shenile_score_theorem :
  ∀ threePointAttempts twoPointAttempts : ℕ,
  threePointAttempts + twoPointAttempts = 30 →
  shenilesScore threePointAttempts twoPointAttempts = 18 :=
by
  sorry

#check shenile_score_theorem

end NUMINAMATH_CALUDE_shenile_score_theorem_l3511_351116


namespace NUMINAMATH_CALUDE_handshake_count_l3511_351143

theorem handshake_count (n : ℕ) (h : n = 6) : 
  n * 2 * (n * 2 - 2) / 2 = 60 := by
  sorry

#check handshake_count

end NUMINAMATH_CALUDE_handshake_count_l3511_351143


namespace NUMINAMATH_CALUDE_oak_trees_after_planting_l3511_351147

/-- The number of oak trees in a park after planting new trees -/
theorem oak_trees_after_planting (current : ℕ) (new : ℕ) : current = 5 → new = 4 → current + new = 9 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_after_planting_l3511_351147


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l3511_351174

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

def complement_N (S : Set ℕ) : Set ℕ := {n : ℕ | n ∉ S}

theorem intersection_complement_theorem :
  A ∩ (complement_N B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l3511_351174


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l3511_351139

theorem sin_cos_fourth_power_sum (θ : Real) (h : Real.cos (2 * θ) = 1/3) :
  Real.sin θ ^ 4 + Real.cos θ ^ 4 = 5/9 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l3511_351139


namespace NUMINAMATH_CALUDE_machine_A_production_rate_l3511_351122

-- Define the production rates and times for machines A, P, and Q
variable (A : ℝ) -- Production rate of Machine A (sprockets per hour)
variable (P : ℝ) -- Production rate of Machine P (sprockets per hour)
variable (Q : ℝ) -- Production rate of Machine Q (sprockets per hour)
variable (T_Q : ℝ) -- Time taken by Machine Q to produce 440 sprockets

-- State the conditions
axiom total_sprockets : 440 = Q * T_Q
axiom time_difference : 440 = P * (T_Q + 10)
axiom production_ratio : Q = 1.1 * A

-- State the theorem to be proved
theorem machine_A_production_rate : A = 4 := by
  sorry

end NUMINAMATH_CALUDE_machine_A_production_rate_l3511_351122


namespace NUMINAMATH_CALUDE_problem_statement_l3511_351190

noncomputable def f (x : ℝ) := Real.exp x + x - 2
noncomputable def g (x : ℝ) := Real.log x + x^2 - 3

theorem problem_statement (a b : ℝ) (h1 : f a = 0) (h2 : g b = 0) :
  g a < 0 ∧ 0 < f b := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3511_351190


namespace NUMINAMATH_CALUDE_parabola_hyperbola_coincident_foci_l3511_351132

/-- Given a parabola and a hyperbola whose foci coincide, we can determine the focal parameter of the parabola. -/
theorem parabola_hyperbola_coincident_foci (p : ℝ) : 
  p > 0 → -- The focal parameter is positive
  (∃ (x y : ℝ), y^2 = 2*p*x) → -- Equation of the parabola
  (∃ (x y : ℝ), x^2 - y^2/3 = 1) → -- Equation of the hyperbola
  (p/2 = 2) → -- The focus of the parabola coincides with the right focus of the hyperbola
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_coincident_foci_l3511_351132


namespace NUMINAMATH_CALUDE_remainder_problem_l3511_351127

theorem remainder_problem (n : ℤ) (h : n % 7 = 5) : (3 * n + 2) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3511_351127


namespace NUMINAMATH_CALUDE_evan_future_books_l3511_351117

/-- Calculates the number of books Evan will have in ten years given the initial conditions --/
def books_in_ten_years (initial_books : ℕ) (reduction : ℕ) (multiplier : ℕ) (addition : ℕ) : ℕ :=
  let current_books := initial_books - reduction
  let books_after_halving := current_books / 2
  multiplier * books_after_halving + addition

/-- Theorem stating that Evan will have 1080 books in ten years --/
theorem evan_future_books :
  books_in_ten_years 400 80 6 120 = 1080 := by
  sorry

#eval books_in_ten_years 400 80 6 120

end NUMINAMATH_CALUDE_evan_future_books_l3511_351117


namespace NUMINAMATH_CALUDE_sufficient_condition_product_greater_than_one_l3511_351118

theorem sufficient_condition_product_greater_than_one :
  ∀ (a b : ℝ), a > 1 → b > 1 → a * b > 1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_product_greater_than_one_l3511_351118


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3511_351191

theorem fraction_to_decimal : 58 / 200 = 1.16 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3511_351191


namespace NUMINAMATH_CALUDE_nested_triple_op_result_l3511_351145

def triple_op (a b c : ℚ) : ℚ := (2 * a + b) / c

def nested_triple_op (x y z : ℚ) : ℚ :=
  triple_op (triple_op 30 60 90) (triple_op 3 6 9) (triple_op 6 12 18)

theorem nested_triple_op_result : nested_triple_op 30 60 90 = 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_triple_op_result_l3511_351145


namespace NUMINAMATH_CALUDE_square_sum_value_l3511_351150

theorem square_sum_value (x y : ℝ) 
  (eq1 : y + 9 = 3 * (x - 1)^2)
  (eq2 : x + 9 = 3 * (y - 1)^2)
  (neq : x ≠ y) : 
  x^2 + y^2 = 71/9 := by
sorry

end NUMINAMATH_CALUDE_square_sum_value_l3511_351150


namespace NUMINAMATH_CALUDE_efficient_methods_l3511_351142

-- Define the types of calculation methods
inductive CalculationMethod
  | Mental
  | Written
  | Calculator

-- Define a function to determine the most efficient method for a given calculation
def most_efficient_method (calculation : ℕ → ℕ → ℕ) : CalculationMethod :=
  sorry

-- Define the specific calculations
def calc1 : ℕ → ℕ → ℕ := λ x y ↦ (x - y) / 5
def calc2 : ℕ → ℕ → ℕ := λ x _ ↦ x * x

-- State the theorem
theorem efficient_methods :
  (most_efficient_method calc1 = CalculationMethod.Calculator) ∧
  (most_efficient_method calc2 = CalculationMethod.Mental) :=
sorry

end NUMINAMATH_CALUDE_efficient_methods_l3511_351142


namespace NUMINAMATH_CALUDE_quadrilateral_is_parallelogram_l3511_351108

-- Define the points
variable (A B C D M N P : ℝ × ℝ)

-- Define the conditions
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def is_midpoint (M B C : ℝ × ℝ) : Prop := sorry

def lines_intersect (A M B N P : ℝ × ℝ) : Prop := sorry

def ratio_equals (P M A : ℝ × ℝ) (r : ℚ) : Prop := sorry

def is_parallelogram (A B C D : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem quadrilateral_is_parallelogram 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_midpoint M B C)
  (h3 : is_midpoint N C D)
  (h4 : lines_intersect A M B N P)
  (h5 : ratio_equals P M A (1/5))
  (h6 : ratio_equals B P N (2/5))
  : is_parallelogram A B C D := by sorry

end NUMINAMATH_CALUDE_quadrilateral_is_parallelogram_l3511_351108


namespace NUMINAMATH_CALUDE_two_digit_numbers_dividing_all_relatives_l3511_351181

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_relative (ab n : ℕ) : Prop :=
  is_two_digit ab ∧
  n % 10 = ab % 10 ∧
  ∀ d ∈ (n / 10).digits 10, d ≠ 0 ∧
  digit_sum (n / 10) = ab / 10

def divides_all_relatives (ab : ℕ) : Prop :=
  is_two_digit ab ∧
  ∀ n : ℕ, is_relative ab n → ab ∣ n

theorem two_digit_numbers_dividing_all_relatives :
  {ab : ℕ | divides_all_relatives ab} =
  {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 30, 45, 90} :=
sorry

end NUMINAMATH_CALUDE_two_digit_numbers_dividing_all_relatives_l3511_351181


namespace NUMINAMATH_CALUDE_point_on_axis_l3511_351115

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of a point being on the x-axis -/
def onXAxis (p : Point2D) : Prop := p.y = 0

/-- Definition of a point being on the y-axis -/
def onYAxis (p : Point2D) : Prop := p.x = 0

/-- Theorem: If xy = 0, then the point is on the x-axis or y-axis -/
theorem point_on_axis (p : Point2D) (h : p.x * p.y = 0) :
  onXAxis p ∨ onYAxis p := by
  sorry

end NUMINAMATH_CALUDE_point_on_axis_l3511_351115


namespace NUMINAMATH_CALUDE_coin_touch_black_probability_l3511_351162

/-- Represents the square layout with black regions -/
structure SquareLayout where
  side_length : ℝ
  corner_triangle_leg : ℝ
  center_circle_diameter : ℝ

/-- Represents a coin -/
structure Coin where
  diameter : ℝ

/-- Calculates the probability of the coin touching any black region -/
def probability_touch_black (layout : SquareLayout) (coin : Coin) : ℝ :=
  sorry

/-- Theorem statement for the probability problem -/
theorem coin_touch_black_probability
  (layout : SquareLayout)
  (coin : Coin)
  (h1 : layout.side_length = 6)
  (h2 : layout.corner_triangle_leg = 1)
  (h3 : layout.center_circle_diameter = 2)
  (h4 : coin.diameter = 2) :
  probability_touch_black layout coin = (2 + Real.pi) / 16 := by
  sorry

end NUMINAMATH_CALUDE_coin_touch_black_probability_l3511_351162


namespace NUMINAMATH_CALUDE_book_purchase_total_price_l3511_351113

/-- Given a total of 80 books, with 32 math books costing $4 each and the rest being history books costing $5 each, prove that the total price is $368. -/
theorem book_purchase_total_price :
  let total_books : ℕ := 80
  let math_books : ℕ := 32
  let math_book_price : ℕ := 4
  let history_book_price : ℕ := 5
  let history_books : ℕ := total_books - math_books
  let total_price : ℕ := math_books * math_book_price + history_books * history_book_price
  total_price = 368 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_total_price_l3511_351113


namespace NUMINAMATH_CALUDE_product_of_distances_l3511_351198

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point P on the ellipse
def P : ℝ × ℝ := sorry

-- State that P is on the ellipse C
axiom P_on_C : C P.1 P.2

-- Define the dot product of vectors PF₁ and PF₂
def PF₁_dot_PF₂ : ℝ := sorry

-- State that the dot product of PF₁ and PF₂ is zero
axiom PF₁_perp_PF₂ : PF₁_dot_PF₂ = 0

-- Define the distances |PF₁| and |PF₂|
def dist_PF₁ : ℝ := sorry
def dist_PF₂ : ℝ := sorry

-- Theorem to prove
theorem product_of_distances : dist_PF₁ * dist_PF₂ = 2 := by sorry

end NUMINAMATH_CALUDE_product_of_distances_l3511_351198


namespace NUMINAMATH_CALUDE_pages_per_notebook_l3511_351128

/-- Given that James buys 2 notebooks, pays $5 in total, and each page costs 5 cents,
    prove that the number of pages in each notebook is 50. -/
theorem pages_per_notebook :
  let notebooks : ℕ := 2
  let total_cost : ℕ := 500  -- in cents
  let cost_per_page : ℕ := 5 -- in cents
  let total_pages : ℕ := total_cost / cost_per_page
  let pages_per_notebook : ℕ := total_pages / notebooks
  pages_per_notebook = 50 := by
sorry

end NUMINAMATH_CALUDE_pages_per_notebook_l3511_351128


namespace NUMINAMATH_CALUDE_correct_sum_after_error_l3511_351144

/-- Given two positive integers a and b, where a is a two-digit number,
    if reversing the digits of a before multiplying by b and adding 35 results in 226,
    then the correct sum of ab + 35 is 54. -/
theorem correct_sum_after_error (a b : ℕ+) : 
  (a.val ≥ 10 ∧ a.val ≤ 99) →
  (((10 * (a.val % 10) + (a.val / 10)) * b.val) + 35 = 226) →
  (a.val * b.val + 35 = 54) :=
by sorry

end NUMINAMATH_CALUDE_correct_sum_after_error_l3511_351144


namespace NUMINAMATH_CALUDE_sticker_distribution_l3511_351129

/-- The number of ways to distribute n identical objects into k distinct bins -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The problem statement -/
theorem sticker_distribution :
  stars_and_bars 10 4 = Nat.choose 13 3 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l3511_351129


namespace NUMINAMATH_CALUDE_element_in_set_l3511_351110

def U : Set Nat := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set Nat) (h : Set.compl M = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l3511_351110


namespace NUMINAMATH_CALUDE_mike_ride_mileage_l3511_351130

/-- Represents the cost of a taxi ride -/
structure TaxiRide where
  startFee : ℝ
  tollFee : ℝ
  mileage : ℝ
  costPerMile : ℝ

/-- Calculates the total cost of a taxi ride -/
def totalCost (ride : TaxiRide) : ℝ :=
  ride.startFee + ride.tollFee + ride.mileage * ride.costPerMile

theorem mike_ride_mileage :
  let mikeRide : TaxiRide := {
    startFee := 2.5,
    tollFee := 0,
    mileage := m,
    costPerMile := 0.25
  }
  let annieRide : TaxiRide := {
    startFee := 2.5,
    tollFee := 5,
    mileage := 26,
    costPerMile := 0.25
  }
  totalCost mikeRide = totalCost annieRide → m = 36 := by
  sorry

#check mike_ride_mileage

end NUMINAMATH_CALUDE_mike_ride_mileage_l3511_351130


namespace NUMINAMATH_CALUDE_survey_satisfactory_percentage_l3511_351125

/-- Given a survey of parents about their children's online class experience, 
    prove that 20% of the respondents rated Satisfactory. -/
theorem survey_satisfactory_percentage :
  ∀ (total excellent very_satisfactory satisfactory needs_improvement : ℕ),
  total = 120 →
  excellent = (15 * total) / 100 →
  very_satisfactory = (60 * total) / 100 →
  needs_improvement = 6 →
  satisfactory = total - excellent - very_satisfactory - needs_improvement →
  (satisfactory : ℚ) / total * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_survey_satisfactory_percentage_l3511_351125


namespace NUMINAMATH_CALUDE_opposites_equation_l3511_351156

theorem opposites_equation (x : ℝ) : (2 * x - 1 = -(-x + 5)) → (2 * x - 1 = x - 5) := by
  sorry

end NUMINAMATH_CALUDE_opposites_equation_l3511_351156


namespace NUMINAMATH_CALUDE_even_function_comparison_l3511_351141

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f is increasing on (-∞, 0) if f(x) < f(y) for all x < y < 0 -/
def IncreasingOnNegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → y < 0 → f x < f y

theorem even_function_comparison (f : ℝ → ℝ) (x₁ x₂ : ℝ)
    (heven : IsEven f)
    (hincr : IncreasingOnNegatives f)
    (hx₁ : x₁ < 0)
    (hsum : x₁ + x₂ > 0) :
    f x₁ > f x₂ := by
  sorry


end NUMINAMATH_CALUDE_even_function_comparison_l3511_351141


namespace NUMINAMATH_CALUDE_total_amount_is_correct_l3511_351101

/-- Calculates the final price of a good after applying rebate and sales tax -/
def finalPrice (originalPrice : ℚ) (rebatePercentage : ℚ) (salesTaxPercentage : ℚ) : ℚ :=
  let reducedPrice := originalPrice * (1 - rebatePercentage / 100)
  reducedPrice * (1 + salesTaxPercentage / 100)

/-- The total amount John has to pay for all goods -/
def totalAmount : ℚ :=
  finalPrice 2500 6 10 + finalPrice 3150 8 12 + finalPrice 1000 5 7

/-- Theorem stating that the total amount John has to pay is equal to 6847.26 -/
theorem total_amount_is_correct : totalAmount = 6847.26 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_correct_l3511_351101


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3511_351123

theorem quadratic_inequality_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - (2*k - 6)*x + k - 3 > 0) ↔ (3 < k ∧ k < 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3511_351123


namespace NUMINAMATH_CALUDE_room_population_change_l3511_351168

theorem room_population_change (initial_men initial_women : ℕ) : 
  initial_men / initial_women = 4 / 5 →
  ∃ (current_women : ℕ),
    initial_men + 2 = 14 ∧
    current_women = 2 * (initial_women - 3) ∧
    current_women = 24 :=
by sorry

end NUMINAMATH_CALUDE_room_population_change_l3511_351168


namespace NUMINAMATH_CALUDE_move_point_l3511_351106

/-- Moving a point left decreases its x-coordinate --/
def move_left (x : ℝ) (units : ℝ) : ℝ := x - units

/-- Moving a point up increases its y-coordinate --/
def move_up (y : ℝ) (units : ℝ) : ℝ := y + units

/-- A 2D point --/
structure Point where
  x : ℝ
  y : ℝ

/-- The initial point P --/
def P : Point := ⟨-2, -3⟩

/-- Theorem: Moving P 1 unit left and 3 units up results in (-3, 0) --/
theorem move_point :
  let new_x := move_left P.x 1
  let new_y := move_up P.y 3
  (new_x, new_y) = (-3, 0) := by sorry

end NUMINAMATH_CALUDE_move_point_l3511_351106


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l3511_351153

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E F : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l3511_351153


namespace NUMINAMATH_CALUDE_wall_width_calculation_l3511_351197

theorem wall_width_calculation (mirror_side : ℝ) (wall_length : ℝ) :
  mirror_side = 21 →
  wall_length = 31.5 →
  (mirror_side * mirror_side) * 2 = wall_length * (882 / wall_length) := by
  sorry

#check wall_width_calculation

end NUMINAMATH_CALUDE_wall_width_calculation_l3511_351197


namespace NUMINAMATH_CALUDE_arithmetic_mean_increase_l3511_351164

theorem arithmetic_mean_increase (b₁ b₂ b₃ b₄ b₅ : ℝ) :
  let original_mean := (b₁ + b₂ + b₃ + b₄ + b₅) / 5
  let new_mean := ((b₁ + 30) + (b₂ + 30) + (b₃ + 30) + (b₄ + 30) + (b₅ + 30)) / 5
  new_mean = original_mean + 30 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_increase_l3511_351164


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3511_351165

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5)
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3511_351165


namespace NUMINAMATH_CALUDE_minimum_cost_is_correct_l3511_351138

/-- Represents the dimensions and cost of a box --/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ
  cost : ℚ

/-- Represents the capacity of a box for different painting sizes --/
structure BoxCapacity where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Represents the collection of paintings --/
structure PaintingCollection where
  small : ℕ
  medium : ℕ
  large : ℕ

def smallBox : Box := ⟨20, 20, 15, 4/5⟩
def mediumBox : Box := ⟨22, 22, 17, 11/10⟩
def largeBox : Box := ⟨24, 24, 20, 27/20⟩

def smallBoxCapacity : BoxCapacity := ⟨3, 2, 0⟩
def mediumBoxCapacity : BoxCapacity := ⟨5, 4, 3⟩
def largeBoxCapacity : BoxCapacity := ⟨8, 6, 5⟩

def collection : PaintingCollection := ⟨1350, 2700, 3150⟩

/-- Calculates the minimum cost to move the entire collection --/
def minimumCost (collection : PaintingCollection) (largeBox : Box) (largeBoxCapacity : BoxCapacity) : ℚ :=
  let smallBoxes := (collection.small + largeBoxCapacity.small - 1) / largeBoxCapacity.small
  let mediumBoxes := (collection.medium + largeBoxCapacity.medium - 1) / largeBoxCapacity.medium
  let largeBoxes := (collection.large + largeBoxCapacity.large - 1) / largeBoxCapacity.large
  (smallBoxes + mediumBoxes + largeBoxes) * largeBox.cost

theorem minimum_cost_is_correct :
  minimumCost collection largeBox largeBoxCapacity = 1686.15 := by
  sorry

end NUMINAMATH_CALUDE_minimum_cost_is_correct_l3511_351138


namespace NUMINAMATH_CALUDE_sand_pile_volume_l3511_351185

/-- Theorem: Volume of a conical sand pile --/
theorem sand_pile_volume (diameter : Real) (height_ratio : Real) :
  diameter = 8 →
  height_ratio = 0.75 →
  let height := height_ratio * diameter
  let radius := diameter / 2
  let volume := (1 / 3) * Real.pi * radius^2 * height
  volume = 32 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sand_pile_volume_l3511_351185


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l3511_351155

theorem last_three_digits_of_7_to_103 :
  7^103 ≡ 343 [ZMOD 1000] := by
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l3511_351155


namespace NUMINAMATH_CALUDE_nested_square_roots_simplification_l3511_351112

theorem nested_square_roots_simplification :
  Real.sqrt (36 * Real.sqrt (12 * Real.sqrt 9)) = 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_roots_simplification_l3511_351112


namespace NUMINAMATH_CALUDE_recurrence_relation_expected_value_after_50_centuries_l3511_351157

/-- The expected value after n centuries in the 50 Cent game -/
def expected_value (n : ℕ) : ℚ :=
  0.5 + 0.25 * n

/-- The initial amount in dollars -/
def initial_amount : ℚ := 0.5

/-- The number of centuries -/
def num_centuries : ℕ := 50

/-- The recurrence relation for the expected value -/
theorem recurrence_relation (n : ℕ) :
  expected_value (n + 1) = (expected_value n + 0.5) / 2 :=
sorry

/-- The main theorem: The expected value after 50 centuries is $13.00 -/
theorem expected_value_after_50_centuries :
  expected_value num_centuries = 13 :=
sorry

end NUMINAMATH_CALUDE_recurrence_relation_expected_value_after_50_centuries_l3511_351157


namespace NUMINAMATH_CALUDE_common_divisor_problem_l3511_351149

theorem common_divisor_problem (n : ℕ) (hn : n < 50) :
  (∃ d : ℕ, d > 1 ∧ d ∣ (3 * n + 5) ∧ d ∣ (5 * n + 4)) ↔ n ∈ ({7, 20, 33, 46} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_common_divisor_problem_l3511_351149


namespace NUMINAMATH_CALUDE_train_distance_l3511_351186

/-- Proves that a train traveling at a speed derived from covering 2 miles in 2 minutes will travel 180 miles in 3 hours. -/
theorem train_distance (distance : ℝ) (time : ℝ) (hours : ℝ) : 
  distance = 2 → time = 2 → hours = 3 → (distance / time) * (hours * 60) = 180 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l3511_351186


namespace NUMINAMATH_CALUDE_peanut_butter_candy_count_l3511_351152

/-- The number of candy pieces in the banana jar -/
def banana_candy : ℕ := 43

/-- The number of candy pieces in the grape jar -/
def grape_candy : ℕ := banana_candy + 5

/-- The number of candy pieces in the peanut butter jar -/
def peanut_butter_candy : ℕ := 4 * grape_candy

/-- Theorem: The peanut butter jar contains 192 pieces of candy -/
theorem peanut_butter_candy_count : peanut_butter_candy = 192 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_candy_count_l3511_351152


namespace NUMINAMATH_CALUDE_vector_operations_l3511_351182

/-- Given vectors in R² -/
def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-1, 2]
def c : Fin 2 → ℝ := ![4, 1]

/-- Theorem stating the results of the vector operations -/
theorem vector_operations :
  (3 • a + b - 2 • c = ![0, 6]) ∧
  (a = (5/9 : ℝ) • b + (8/9 : ℝ) • c) := by
  sorry

end NUMINAMATH_CALUDE_vector_operations_l3511_351182


namespace NUMINAMATH_CALUDE_coefficient_x4_eq_21_l3511_351161

/-- The coefficient of x^4 in the binomial expansion of (x+1/x-1)^6 -/
def coefficient_x4 : ℕ :=
  (Nat.choose 6 0) * (Nat.choose 6 1) + (Nat.choose 6 2) * (Nat.choose 4 0)

/-- Theorem stating that the coefficient of x^4 in the binomial expansion of (x+1/x-1)^6 is 21 -/
theorem coefficient_x4_eq_21 : coefficient_x4 = 21 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_eq_21_l3511_351161


namespace NUMINAMATH_CALUDE_jake_weight_loss_l3511_351188

theorem jake_weight_loss (jake_current sister_current total_current : ℕ) 
  (h1 : jake_current + sister_current = total_current)
  (h2 : jake_current = 156)
  (h3 : total_current = 224) :
  ∃ (weight_loss : ℕ), jake_current - weight_loss = 2 * (sister_current - weight_loss) ∧ weight_loss = 20 := by
sorry

end NUMINAMATH_CALUDE_jake_weight_loss_l3511_351188


namespace NUMINAMATH_CALUDE_min_turns_rook_path_l3511_351170

/-- Represents a chessboard --/
structure Chessboard :=
  (files : Nat)
  (ranks : Nat)

/-- Represents a rook's path on a chessboard --/
structure RookPath :=
  (board : Chessboard)
  (turns : Nat)
  (visitsAllSquares : Bool)

/-- Defines a valid rook path that visits all squares exactly once --/
def isValidRookPath (path : RookPath) : Prop :=
  path.board.files = 8 ∧
  path.board.ranks = 8 ∧
  path.visitsAllSquares = true

/-- Theorem: The minimum number of turns for a rook to visit all squares on an 8x8 chessboard is 14 --/
theorem min_turns_rook_path :
  ∀ (path : RookPath), isValidRookPath path → path.turns ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_min_turns_rook_path_l3511_351170


namespace NUMINAMATH_CALUDE_janet_has_five_dimes_l3511_351137

/-- Represents the number of coins of each type Janet has -/
structure CoinCount where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- The conditions of Janet's coin collection -/
def janet_coins (c : CoinCount) : Prop :=
  c.nickels + c.dimes + c.quarters = 10 ∧
  c.dimes + c.quarters = 7 ∧
  c.nickels + c.dimes = 8

/-- Theorem stating that Janet has 5 dimes -/
theorem janet_has_five_dimes :
  ∃ c : CoinCount, janet_coins c ∧ c.dimes = 5 := by
  sorry

end NUMINAMATH_CALUDE_janet_has_five_dimes_l3511_351137


namespace NUMINAMATH_CALUDE_geometric_progression_equality_l3511_351179

theorem geometric_progression_equality 
  (a b q : ℝ) 
  (n p : ℕ) 
  (h_q : q ≠ 1) 
  (h_sum : a * (1 - q^(n*p)) / (1 - q) = b * (1 - q^(n*p)) / (1 - q^p)) :
  b = a * (1 - q^p) / (1 - q) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_equality_l3511_351179


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l3511_351196

/-- Represents a trapezoid EFGH -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  EG : ℝ
  area : ℝ

/-- Theorem stating the length of EG in the given trapezoid -/
theorem trapezoid_side_length (t : Trapezoid) 
  (h1 : t.EF = 10)
  (h2 : t.GH = 14)
  (h3 : t.area = 72)
  (h4 : t.EG = (((t.GH - t.EF) / 2) ^ 2 + (2 * t.area / (t.EF + t.GH)) ^ 2).sqrt) :
  t.EG = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l3511_351196


namespace NUMINAMATH_CALUDE_elizas_height_l3511_351184

/-- Given the heights of Eliza's siblings and their total height, prove Eliza's height -/
theorem elizas_height (total_height : ℕ) (sibling1_height : ℕ) (sibling2_height : ℕ) (sibling3_height : ℕ) (sibling4_height : ℕ) :
  total_height = 330 ∧ 
  sibling1_height = 66 ∧ 
  sibling2_height = 66 ∧ 
  sibling3_height = 60 ∧ 
  sibling4_height = sibling1_height + 2 →
  ∃ (eliza_height : ℕ), eliza_height = 68 ∧ 
    total_height = sibling1_height + sibling2_height + sibling3_height + sibling4_height + eliza_height :=
by
  sorry

end NUMINAMATH_CALUDE_elizas_height_l3511_351184


namespace NUMINAMATH_CALUDE_train_crossing_time_l3511_351131

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 400 ∧ 
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) ∧
  crossing_time = 10 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3511_351131


namespace NUMINAMATH_CALUDE_exists_plane_parallel_to_skew_lines_l3511_351124

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

/-- A plane is parallel to a line if the line's direction is perpendicular to the plane's normal -/
def plane_parallel_to_line (p : Plane3D) (l : Line3D) : Prop := sorry

/-- There exists a plane parallel to both skew lines -/
theorem exists_plane_parallel_to_skew_lines (a b : Line3D) (h : are_skew a b) :
  ∃ (α : Plane3D), plane_parallel_to_line α a ∧ plane_parallel_to_line α b := by
  sorry

end NUMINAMATH_CALUDE_exists_plane_parallel_to_skew_lines_l3511_351124


namespace NUMINAMATH_CALUDE_three_consecutive_days_without_class_l3511_351199

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a day in the month -/
structure Day where
  dayNumber : Nat
  dayOfWeek : DayOfWeek

/-- Definition of a month with its properties -/
structure Month where
  days : List Day
  startDay : DayOfWeek
  totalDays : Nat
  classSchedule : List Nat

/-- Main theorem to prove -/
theorem three_consecutive_days_without_class 
  (november2017 : Month)
  (h1 : november2017.startDay = DayOfWeek.Wednesday)
  (h2 : november2017.totalDays = 30)
  (h3 : november2017.classSchedule.length = 11)
  (h4 : ∀ d ∈ november2017.days, 
    d.dayOfWeek = DayOfWeek.Saturday ∨ d.dayOfWeek = DayOfWeek.Sunday → 
    d.dayNumber ∉ november2017.classSchedule) :
  ∃ d1 d2 d3 : Day, 
    d1 ∈ november2017.days ∧ 
    d2 ∈ november2017.days ∧ 
    d3 ∈ november2017.days ∧ 
    d1.dayNumber + 1 = d2.dayNumber ∧ 
    d2.dayNumber + 1 = d3.dayNumber ∧ 
    d1.dayNumber ∉ november2017.classSchedule ∧ 
    d2.dayNumber ∉ november2017.classSchedule ∧ 
    d3.dayNumber ∉ november2017.classSchedule :=
by sorry

end NUMINAMATH_CALUDE_three_consecutive_days_without_class_l3511_351199


namespace NUMINAMATH_CALUDE_power_of_two_geq_double_l3511_351102

theorem power_of_two_geq_double (n : ℕ) : 2^n ≥ 2*n := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_geq_double_l3511_351102


namespace NUMINAMATH_CALUDE_total_turnips_l3511_351154

theorem total_turnips (melanie_turnips benny_turnips : ℕ) 
  (h1 : melanie_turnips = 139) 
  (h2 : benny_turnips = 113) : 
  melanie_turnips + benny_turnips = 252 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_l3511_351154


namespace NUMINAMATH_CALUDE_perfect_pink_paint_ratio_l3511_351104

/-- The ratio of white paint to red paint in perfect pink paint is 1:1 -/
theorem perfect_pink_paint_ratio :
  ∀ (total_paint red_paint white_paint : ℚ),
  total_paint = 30 →
  red_paint = 15 →
  total_paint = red_paint + white_paint →
  white_paint / red_paint = 1 := by
sorry

end NUMINAMATH_CALUDE_perfect_pink_paint_ratio_l3511_351104


namespace NUMINAMATH_CALUDE_circle_radius_proof_l3511_351135

theorem circle_radius_proof (r : ℝ) (x₁ y₁ x₂ y₂ : ℝ) 
  (h_r_pos : r > 0)
  (h_circle₁ : x₁^2 + y₁^2 = r^2)
  (h_circle₂ : x₂^2 + y₂^2 = r^2)
  (h_sum₁ : x₁ + y₁ = 3)
  (h_sum₂ : x₂ + y₂ = 3)
  (h_product : x₁ * x₂ + y₁ * y₂ = -1/2 * r^2) :
  r = 3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l3511_351135


namespace NUMINAMATH_CALUDE_ferris_wheel_cost_l3511_351163

/-- The cost of a Ferris wheel ride given the conditions of Zach's amusement park visit -/
theorem ferris_wheel_cost 
  (total_rides : Nat) 
  (roller_coaster_cost log_ride_cost : Nat) 
  (zach_initial_tickets zach_additional_tickets : Nat) :
  total_rides = 3 →
  roller_coaster_cost = 7 →
  log_ride_cost = 1 →
  zach_initial_tickets = 1 →
  zach_additional_tickets = 9 →
  roller_coaster_cost + log_ride_cost + 2 = zach_initial_tickets + zach_additional_tickets :=
by sorry

end NUMINAMATH_CALUDE_ferris_wheel_cost_l3511_351163


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3511_351160

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 3*x

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2) 2 ∧
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≤ f c) ∧
  f c = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3511_351160


namespace NUMINAMATH_CALUDE_stating_popsicle_count_l3511_351172

/-- The number of popsicles in a box with specific melting rate properties -/
def num_popsicles : ℕ := 6

/-- The melting rate factor between consecutive popsicles -/
def melting_rate_factor : ℕ := 2

/-- The relative melting rate of the last popsicle compared to the first -/
def last_to_first_rate : ℕ := 32

/-- 
Theorem stating that the number of popsicles in the box is 6, given the melting rate properties
-/
theorem popsicle_count :
  (melting_rate_factor ^ (num_popsicles - 1) = last_to_first_rate) →
  num_popsicles = 6 := by
sorry

end NUMINAMATH_CALUDE_stating_popsicle_count_l3511_351172


namespace NUMINAMATH_CALUDE_olivia_savings_account_l3511_351133

/-- The compound interest function -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem olivia_savings_account :
  let principal : ℝ := 5000
  let rate : ℝ := 0.07
  let time : ℕ := 15
  let final_amount := compound_interest principal rate time
  ∃ ε > 0, |final_amount - 13795.15| < ε :=
by sorry

end NUMINAMATH_CALUDE_olivia_savings_account_l3511_351133


namespace NUMINAMATH_CALUDE_arc_length_for_given_circle_l3511_351194

theorem arc_length_for_given_circle (r : ℝ) (θ : ℝ) (arc_length : ℝ) : 
  r = 2 → θ = π / 7 → arc_length = r * θ → arc_length = 2 * π / 7 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_for_given_circle_l3511_351194


namespace NUMINAMATH_CALUDE_hiker_distance_l3511_351175

theorem hiker_distance (s t : ℝ) 
  (h1 : (s + 1) * (2/3 * t) = s * t) 
  (h2 : (s - 1) * (t + 3) = s * t) : 
  s * t = 6 := by
  sorry

end NUMINAMATH_CALUDE_hiker_distance_l3511_351175


namespace NUMINAMATH_CALUDE_binary_conversion_l3511_351146

-- Define the binary number
def binary_num : List Bool := [true, false, true, true, false, true]

-- Define the function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define the function to convert decimal to base-7
def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

-- Theorem statement
theorem binary_conversion :
  binary_to_decimal binary_num = 45 ∧
  decimal_to_base7 (binary_to_decimal binary_num) = [6, 3] := by
  sorry

end NUMINAMATH_CALUDE_binary_conversion_l3511_351146


namespace NUMINAMATH_CALUDE_betty_pays_nothing_l3511_351100

-- Define the ages and cost
def doug_age : ℕ := 40
def alice_age : ℕ := doug_age / 2
def total_age_sum : ℕ := 130
def cost_decrease_per_year : ℕ := 5

-- Define Betty's age
def betty_age : ℕ := total_age_sum - doug_age - alice_age

-- Define the original cost of a pack of nuts
def original_nut_cost : ℕ := 2 * betty_age

-- Define the age difference between Betty and Alice
def age_difference : ℕ := betty_age - alice_age

-- Define the total cost decrease
def total_cost_decrease : ℕ := age_difference * cost_decrease_per_year

-- Define the new cost of a pack of nuts
def new_nut_cost : ℕ := max 0 (original_nut_cost - total_cost_decrease)

-- Theorem to prove
theorem betty_pays_nothing : new_nut_cost * 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_betty_pays_nothing_l3511_351100


namespace NUMINAMATH_CALUDE_lily_reads_28_books_l3511_351105

/-- Represents Lily's reading habits and goals over two months -/
structure LilyReading where
  last_month_weekday : Nat
  last_month_weekend : Nat
  this_month_weekday_factor : Nat
  this_month_weekend_factor : Nat

/-- Calculates the total number of books Lily reads in two months -/
def total_books_read (r : LilyReading) : Nat :=
  let last_month_total := r.last_month_weekday + r.last_month_weekend
  let this_month_weekday := r.last_month_weekday * r.this_month_weekday_factor
  let this_month_weekend := r.last_month_weekend * r.this_month_weekend_factor
  let this_month_total := this_month_weekday + this_month_weekend
  last_month_total + this_month_total

/-- Theorem stating that Lily reads 28 books in total over two months -/
theorem lily_reads_28_books :
  ∀ (r : LilyReading),
    r.last_month_weekday = 4 →
    r.last_month_weekend = 4 →
    r.this_month_weekday_factor = 2 →
    r.this_month_weekend_factor = 3 →
    total_books_read r = 28 :=
  sorry


end NUMINAMATH_CALUDE_lily_reads_28_books_l3511_351105


namespace NUMINAMATH_CALUDE_union_when_m_neg_two_intersection_equals_B_iff_l3511_351140

-- Define sets A and B
def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < 1 + m}

-- Part 1
theorem union_when_m_neg_two : 
  A ∪ B (-2) = {x : ℝ | -5 < x ∧ x ≤ 4} := by sorry

-- Part 2
theorem intersection_equals_B_iff : 
  ∀ m : ℝ, A ∩ B m = B m ↔ m ≥ -1 := by sorry

end NUMINAMATH_CALUDE_union_when_m_neg_two_intersection_equals_B_iff_l3511_351140


namespace NUMINAMATH_CALUDE_erased_number_proof_l3511_351180

theorem erased_number_proof (n : ℕ) (i : ℕ) :
  n > 0 ∧ i > 0 ∧ i ≤ n ∧
  (n * (n + 1) / 2 - i) / (n - 1) = 602 / 17 →
  i = 7 :=
by sorry

end NUMINAMATH_CALUDE_erased_number_proof_l3511_351180


namespace NUMINAMATH_CALUDE_our_circle_center_and_radius_l3511_351107

/-- A circle in the xy-plane --/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle --/
def center (c : Circle) : ℝ × ℝ := sorry

/-- The radius of a circle --/
def radius (c : Circle) : ℝ := sorry

/-- Our specific circle --/
def our_circle : Circle :=
  { equation := λ x y => x^2 + y^2 - 6*x = 0 }

theorem our_circle_center_and_radius :
  center our_circle = (3, 0) ∧ radius our_circle = 3 := by sorry

end NUMINAMATH_CALUDE_our_circle_center_and_radius_l3511_351107


namespace NUMINAMATH_CALUDE_probability_sum_5_is_one_ninth_l3511_351178

/-- The number of possible outcomes when rolling two fair dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (sum of 5) when rolling two fair dice -/
def favorable_outcomes : ℕ := 4

/-- The probability of rolling a sum of 5 with two fair dice -/
def probability_sum_5 : ℚ := favorable_outcomes / total_outcomes

/-- Theorem stating that the probability of rolling a sum of 5 with two fair dice is 1/9 -/
theorem probability_sum_5_is_one_ninth : probability_sum_5 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_5_is_one_ninth_l3511_351178


namespace NUMINAMATH_CALUDE_acute_angles_sum_l3511_351151

theorem acute_angles_sum (a b : Real) : 
  0 < a ∧ a < π/2 →
  0 < b ∧ b < π/2 →
  4 * Real.sin a ^ 2 + 3 * Real.sin b ^ 2 = 1 →
  4 * Real.sin (2 * a) + 3 * Real.sin (2 * b) = 0 →
  2 * a + 3 * b = π/2 := by
sorry

end NUMINAMATH_CALUDE_acute_angles_sum_l3511_351151


namespace NUMINAMATH_CALUDE_base7_5304_equals_1866_l3511_351159

def base7_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem base7_5304_equals_1866 :
  base7_to_decimal [5, 3, 0, 4] = 1866 := by
  sorry

end NUMINAMATH_CALUDE_base7_5304_equals_1866_l3511_351159


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l3511_351120

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem: A batsman's average after 12 innings is 64, given the conditions -/
theorem batsman_average_after_12th_innings
  (stats : BatsmanStats)
  (h1 : stats.innings = 11)
  (h2 : newAverage stats 75 = stats.average + 1) :
  newAverage stats 75 = 64 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l3511_351120


namespace NUMINAMATH_CALUDE_correct_system_is_valid_l3511_351134

/-- Represents the purchase of labor tools by a school -/
structure ToolPurchase where
  x : ℕ  -- number of type A tools
  y : ℕ  -- number of type B tools
  total_tools : x + y = 145
  total_cost : 10 * x + 12 * y = 1580

/-- The correct system of equations for the tool purchase -/
def correct_system (p : ToolPurchase) : Prop :=
  (p.x + p.y = 145) ∧ (10 * p.x + 12 * p.y = 1580)

/-- Theorem stating that the given system of equations is correct -/
theorem correct_system_is_valid (p : ToolPurchase) : correct_system p := by
  sorry

#check correct_system_is_valid

end NUMINAMATH_CALUDE_correct_system_is_valid_l3511_351134


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3511_351192

theorem polynomial_simplification (x : ℝ) :
  (3*x^2 + 5*x + 8)*(x - 2) - (x - 2)*(x^2 + 6*x - 72) + (2*x - 15)*(x - 2)*(x + 4) =
  4*x^3 - 17*x^2 + 38*x - 40 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3511_351192


namespace NUMINAMATH_CALUDE_intersection_of_AB_and_CD_l3511_351177

def A : ℝ × ℝ × ℝ := (2, -1, 2)
def B : ℝ × ℝ × ℝ := (12, -11, 7)
def C : ℝ × ℝ × ℝ := (1, 4, -7)
def D : ℝ × ℝ × ℝ := (4, -2, 13)

def line_intersection (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

theorem intersection_of_AB_and_CD :
  line_intersection A B C D = (8/3, -7/3, 7/3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_AB_and_CD_l3511_351177


namespace NUMINAMATH_CALUDE_function_identity_l3511_351166

theorem function_identity (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x ≤ x) 
  (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end NUMINAMATH_CALUDE_function_identity_l3511_351166


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3511_351126

-- Define the parabola P
def P (x : ℝ) : ℝ := x^2 + 5

-- Define the point Q
def Q : ℝ × ℝ := (10, 10)

-- Define the line through Q with slope m
def line_through_Q (m : ℝ) (x : ℝ) : ℝ := m * (x - Q.1) + Q.2

-- Define the condition for no intersection
def no_intersection (m : ℝ) : Prop :=
  ∀ x : ℝ, line_through_Q m x ≠ P x

-- Define r and s
noncomputable def r : ℝ := 20 - 10 * Real.sqrt 38
noncomputable def s : ℝ := 20 + 10 * Real.sqrt 38

-- Theorem statement
theorem parabola_line_intersection :
  (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) →
  r + s = 40 :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3511_351126


namespace NUMINAMATH_CALUDE_halfway_fraction_l3511_351109

theorem halfway_fraction (a b c d : ℚ) (h1 : a = 2/3) (h2 : b = 4/5) (h3 : c = (a + b) / 2) : c = 11/15 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l3511_351109


namespace NUMINAMATH_CALUDE_sqrt_twelve_times_sqrt_three_minus_five_equals_one_l3511_351114

theorem sqrt_twelve_times_sqrt_three_minus_five_equals_one :
  Real.sqrt 12 * Real.sqrt 3 - 5 = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_twelve_times_sqrt_three_minus_five_equals_one_l3511_351114


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3511_351136

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x > 0 ∧ 
    (⌊x^2⌋ - x * ⌊x⌋ = 6) ∧
    (∀ y : ℝ, y > 0 → (⌊y^2⌋ - y * ⌊y⌋ = 6) → x ≤ y) ∧
    x = 55 / 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l3511_351136


namespace NUMINAMATH_CALUDE_novel_pages_count_l3511_351195

theorem novel_pages_count (x : ℝ) : 
  let day1_read := x / 6 + 10
  let day1_remaining := x - day1_read
  let day2_read := day1_remaining / 5 + 14
  let day2_remaining := day1_remaining - day2_read
  let day3_read := day2_remaining / 4 + 16
  let day3_remaining := day2_remaining - day3_read
  day3_remaining = 48 → x = 161 := by
sorry

end NUMINAMATH_CALUDE_novel_pages_count_l3511_351195


namespace NUMINAMATH_CALUDE_quadratic_intersection_range_l3511_351148

/-- For a quadratic function y = 2mx^2 + (8m+1)x + 8m that intersects the x-axis, 
    the range of m is [m ≥ -1/16 and m ≠ 0] -/
theorem quadratic_intersection_range (m : ℝ) : 
  (∃ x, 2*m*x^2 + (8*m + 1)*x + 8*m = 0) → 
  (m ≥ -1/16 ∧ m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_range_l3511_351148


namespace NUMINAMATH_CALUDE_odd_function_value_l3511_351176

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_value (f : ℝ → ℝ) (h_odd : is_odd f) (h_pos : ∀ x > 0, f x = x * (x - 1)) :
  f (-3) = -6 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l3511_351176


namespace NUMINAMATH_CALUDE_internet_charge_proof_l3511_351103

/-- The daily charge for internet service -/
def daily_charge : ℚ := 48/100

/-- The initial balance -/
def initial_balance : ℚ := 0

/-- The payment made -/
def payment : ℚ := 7

/-- The number of days of service -/
def service_days : ℕ := 25

/-- The debt threshold for service discontinuation -/
def debt_threshold : ℚ := 5

theorem internet_charge_proof :
  (initial_balance + payment - service_days * daily_charge = -debt_threshold) ∧
  (∀ x : ℚ, x > daily_charge → initial_balance + payment - service_days * x < -debt_threshold) :=
sorry

end NUMINAMATH_CALUDE_internet_charge_proof_l3511_351103


namespace NUMINAMATH_CALUDE_slope_of_line_slope_of_specific_line_l3511_351169

/-- The slope of a line given by the equation y + ax - b = 0 is -a. -/
theorem slope_of_line (a b : ℝ) : 
  (fun x y : ℝ => y + a * x - b = 0) = (fun x y : ℝ => y = -a * x + b) := by
  sorry

/-- The slope of the line y + 3x - 1 = 0 is -3. -/
theorem slope_of_specific_line : 
  (fun x y : ℝ => y + 3 * x - 1 = 0) = (fun x y : ℝ => y = -3 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_slope_of_specific_line_l3511_351169


namespace NUMINAMATH_CALUDE_monomial_evaluation_l3511_351167

theorem monomial_evaluation : 0.007 * (-5)^7 * 2^9 = -280000 := by
  sorry

end NUMINAMATH_CALUDE_monomial_evaluation_l3511_351167


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3511_351121

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |1 + x + x^2 / 2| < 1} = {x : ℝ | -2 < x ∧ x < 0} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3511_351121


namespace NUMINAMATH_CALUDE_simplify_expression_l3511_351111

theorem simplify_expression : 
  (Real.sqrt (32^(1/5)) - Real.sqrt 7)^2 = 11 - 4 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3511_351111


namespace NUMINAMATH_CALUDE_teacher_student_arrangements_l3511_351189

/-- The number of arrangements for a teacher and students in a row --/
def arrangements (n : ℕ) : ℕ :=
  (n - 2) * n.factorial

/-- The problem statement --/
theorem teacher_student_arrangements :
  arrangements 6 = 480 := by
  sorry

end NUMINAMATH_CALUDE_teacher_student_arrangements_l3511_351189


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainder_l3511_351158

theorem greatest_divisor_with_remainder (a b c : ℕ) (h : a = 263 ∧ b = 935 ∧ c = 1383) :
  (∃ (d : ℕ), d > 0 ∧ 
    (a % d = 7 ∧ b % d = 7 ∧ c % d = 7) ∧
    (∀ (k : ℕ), k > d → (a % k ≠ 7 ∨ b % k ≠ 7 ∨ c % k ≠ 7))) →
  (∃ (d : ℕ), d = 16 ∧
    (a % d = 7 ∧ b % d = 7 ∧ c % d = 7) ∧
    (∀ (k : ℕ), k > d → (a % k ≠ 7 ∨ b % k ≠ 7 ∨ c % k ≠ 7))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainder_l3511_351158


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3511_351187

theorem divisibility_equivalence (n : ℕ) :
  5 ∣ (1^n + 2^n + 3^n + 4^n) ↔ n % 4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3511_351187


namespace NUMINAMATH_CALUDE_final_number_is_two_thirds_l3511_351173

def board_numbers : List ℚ := List.map (λ k => k / 2016) (List.range 2016)

def transform (a b : ℚ) : ℚ := 3 * a * b - 2 * a - 2 * b + 2

theorem final_number_is_two_thirds :
  ∃ (moves : List (ℚ × ℚ)),
    moves.length = 2015 ∧
    (moves.foldl
      (λ board (a, b) => (transform a b) :: (board.filter (λ x => x ≠ a ∧ x ≠ b)))
      board_numbers).head? = some (2/3) :=
sorry

end NUMINAMATH_CALUDE_final_number_is_two_thirds_l3511_351173


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l3511_351193

theorem quadratic_inequality_roots (b : ℝ) : 
  (∀ x : ℝ, -x^2 + b*x - 5 < 0 ↔ x < 1 ∨ x > 5) → b = 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l3511_351193


namespace NUMINAMATH_CALUDE_johns_age_l3511_351171

theorem johns_age (john dad : ℕ) 
  (h1 : john + 34 = dad) 
  (h2 : john + dad = 84) : 
  john = 25 := by
sorry

end NUMINAMATH_CALUDE_johns_age_l3511_351171


namespace NUMINAMATH_CALUDE_keyword_selection_theorem_l3511_351183

theorem keyword_selection_theorem (n m k : ℕ) (h1 : n = 12) (h2 : m = 4) (h3 : k = 2) : 
  (Nat.choose n k * Nat.choose m 1 + Nat.choose m k) + 
  (Nat.choose n (k + 1) * Nat.choose m 1 + Nat.choose n k * Nat.choose m 2 + Nat.choose m (k + 1)) = 202 := by
  sorry

end NUMINAMATH_CALUDE_keyword_selection_theorem_l3511_351183
