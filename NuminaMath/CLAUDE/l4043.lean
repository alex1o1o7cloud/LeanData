import Mathlib

namespace NUMINAMATH_CALUDE_sequence_sum_l4043_404316

def S (n : ℕ) : ℤ :=
  if n % 2 = 1 then (n + 1) / 2 else -(n / 2)

theorem sequence_sum : S 19 * S 31 + S 48 = 136 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l4043_404316


namespace NUMINAMATH_CALUDE_roses_cut_is_difference_l4043_404313

/-- The number of roses Mary cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem stating that the number of roses Mary cut is the difference between the final and initial number of roses -/
theorem roses_cut_is_difference (initial_roses final_roses : ℕ) 
  (h : final_roses ≥ initial_roses) : 
  roses_cut initial_roses final_roses = final_roses - initial_roses :=
by
  sorry

#eval roses_cut 6 16  -- Should evaluate to 10

end NUMINAMATH_CALUDE_roses_cut_is_difference_l4043_404313


namespace NUMINAMATH_CALUDE_smallest_rational_number_l4043_404306

theorem smallest_rational_number : ∀ (a b c d : ℚ), 
  a = 0 → b = -1/2 → c = -1/3 → d = 4 →
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_rational_number_l4043_404306


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4043_404362

theorem sufficient_not_necessary (y : ℝ) (h : y > 2) :
  (∀ x, x > 1 → x + y > 3) ∧ 
  (∃ x, x + y > 3 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4043_404362


namespace NUMINAMATH_CALUDE_sum_of_imaginary_parts_l4043_404315

/-- Given three complex numbers with specific conditions, prove that s+u = 1 -/
theorem sum_of_imaginary_parts (p q r s t u : ℝ) : 
  q = 5 → 
  p = -r - 2*t → 
  Complex.mk (p + r + t) (q + s + u) = Complex.I * 6 → 
  s + u = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_imaginary_parts_l4043_404315


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l4043_404339

theorem simplify_trig_expression :
  1 / Real.sqrt (1 + Real.tan (160 * π / 180) ^ 2) = -Real.cos (160 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l4043_404339


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l4043_404379

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_multiplier_for_perfect_square :
  ∃ (k : ℕ), k > 0 ∧ 
  (∃ (m : ℕ), k * y = m^2) ∧
  (∀ (j : ℕ), 0 < j ∧ j < k → ¬∃ (n : ℕ), j * y = n^2) ∧
  k = 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l4043_404379


namespace NUMINAMATH_CALUDE_triangle_theorem_l4043_404368

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*sin(A) = 4b*sin(B) and a*c = √5*(a^2 - b^2 - c^2),
    then cos(A) = -√5/5 and sin(2B - A) = -2√5/5 -/
theorem triangle_theorem (a b c A B C : ℝ) 
  (h1 : a * Real.sin A = 4 * b * Real.sin B)
  (h2 : a * c = Real.sqrt 5 * (a^2 - b^2 - c^2)) :
  Real.cos A = -(Real.sqrt 5 / 5) ∧ 
  Real.sin (2 * B - A) = -(2 * Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l4043_404368


namespace NUMINAMATH_CALUDE_sally_peach_cost_l4043_404318

def total_spent : ℝ := 23.86
def cherry_cost : ℝ := 11.54
def peach_cost : ℝ := total_spent - cherry_cost

theorem sally_peach_cost : peach_cost = 12.32 := by
  sorry

end NUMINAMATH_CALUDE_sally_peach_cost_l4043_404318


namespace NUMINAMATH_CALUDE_quadratic_real_solution_l4043_404324

theorem quadratic_real_solution (m : ℝ) : 
  (∃ z : ℝ, z^2 + Complex.I * z + m = 0) ↔ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solution_l4043_404324


namespace NUMINAMATH_CALUDE_rhombus_area_l4043_404312

/-- The area of a rhombus with sides of length 4 and an acute angle of 45 degrees is 16 square units -/
theorem rhombus_area (side_length : ℝ) (acute_angle : ℝ) : 
  side_length = 4 → 
  acute_angle = 45 * π / 180 →
  side_length * side_length * Real.sin acute_angle = 16 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l4043_404312


namespace NUMINAMATH_CALUDE_distance_between_points_l4043_404361

theorem distance_between_points : 
  let p1 : ℚ × ℚ := (-3/2, -1/2)
  let p2 : ℚ × ℚ := (9/2, 7/2)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 52 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l4043_404361


namespace NUMINAMATH_CALUDE_sin_difference_of_complex_exponentials_l4043_404381

theorem sin_difference_of_complex_exponentials (α β : ℝ) :
  Complex.exp (α * I) = 4/5 + 3/5 * I →
  Complex.exp (β * I) = 12/13 + 5/13 * I →
  Real.sin (α - β) = -16/65 := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_of_complex_exponentials_l4043_404381


namespace NUMINAMATH_CALUDE_tripod_height_after_break_l4043_404389

theorem tripod_height_after_break (original_leg_length original_height broken_leg_length : ℝ) 
  (h : ℝ) (m n : ℕ) :
  original_leg_length = 6 →
  original_height = 5 →
  broken_leg_length = 4 →
  h = 12 →
  h = m / Real.sqrt n →
  m = 168 →
  n = 169 →
  ⌊m + Real.sqrt n⌋ = 181 :=
by sorry

end NUMINAMATH_CALUDE_tripod_height_after_break_l4043_404389


namespace NUMINAMATH_CALUDE_right_triangle_area_l4043_404323

/-- The area of a right-angled triangle with base 12 cm and height 15 cm is 90 square centimeters -/
theorem right_triangle_area : 
  ∀ (base height area : ℝ), 
  base = 12 → 
  height = 15 → 
  area = (1/2) * base * height → 
  area = 90 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l4043_404323


namespace NUMINAMATH_CALUDE_marbles_ratio_l4043_404305

def total_marbles : ℕ := 63
def your_initial_marbles : ℕ := 16

def brother_marbles : ℕ → ℕ → ℕ
  | your_marbles, marbles_given => 
    (total_marbles - your_marbles - (3 * (your_marbles - marbles_given))) / 2 + marbles_given

def your_final_marbles : ℕ := your_initial_marbles - 2

theorem marbles_ratio : 
  ∃ (m : ℕ), m > 0 ∧ your_final_marbles = m * (brother_marbles your_initial_marbles 2) ∧
  (your_final_marbles : ℚ) / (brother_marbles your_initial_marbles 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_marbles_ratio_l4043_404305


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l4043_404391

theorem max_product_sum_2000 :
  (∃ (x y : ℤ), x + y = 2000 ∧ x * y = 1000000) ∧
  (∀ (a b : ℤ), a + b = 2000 → a * b ≤ 1000000) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l4043_404391


namespace NUMINAMATH_CALUDE_certain_number_solution_l4043_404326

theorem certain_number_solution : 
  ∃ x : ℝ, (3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5) = 800.0000000000001 ∧ x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_solution_l4043_404326


namespace NUMINAMATH_CALUDE_english_only_enrollment_l4043_404349

/-- Represents the enrollment data for a class with English and German courses -/
structure ClassEnrollment where
  total : Nat
  both : Nat
  german : Nat

/-- Calculates the number of students enrolled only in English -/
def studentsOnlyEnglish (c : ClassEnrollment) : Nat :=
  c.total - c.german

/-- Theorem stating that 28 students are enrolled only in English -/
theorem english_only_enrollment (c : ClassEnrollment) 
  (h1 : c.total = 50)
  (h2 : c.both = 12)
  (h3 : c.german = 22)
  (h4 : c.total = studentsOnlyEnglish c + c.german) :
  studentsOnlyEnglish c = 28 := by
  sorry

#eval studentsOnlyEnglish { total := 50, both := 12, german := 22 }

end NUMINAMATH_CALUDE_english_only_enrollment_l4043_404349


namespace NUMINAMATH_CALUDE_chocolate_difference_l4043_404348

theorem chocolate_difference (t : ℚ) : 
  let sarah := (1 : ℚ) / 3 * t
  let andrew := (3 : ℚ) / 8 * t
  let cecily := t - (sarah + andrew)
  sarah - cecily = (1 : ℚ) / 24 * t := by sorry

end NUMINAMATH_CALUDE_chocolate_difference_l4043_404348


namespace NUMINAMATH_CALUDE_rotate_point_A_about_C_l4043_404377

/-- Rotates a point 180 degrees about a center point -/
def rotate180 (point center : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - point.1, 2 * center.2 - point.2)

theorem rotate_point_A_about_C : 
  let A : ℝ × ℝ := (-4, 1)
  let C : ℝ × ℝ := (-1, 1)
  rotate180 A C = (2, 1) := by sorry

end NUMINAMATH_CALUDE_rotate_point_A_about_C_l4043_404377


namespace NUMINAMATH_CALUDE_coin_count_l4043_404383

theorem coin_count (total_value : ℕ) (two_dollar_coins : ℕ) : 
  total_value = 402 → two_dollar_coins = 148 → 
  ∃ (one_dollar_coins : ℕ), 
    total_value = 2 * two_dollar_coins + one_dollar_coins ∧
    one_dollar_coins + two_dollar_coins = 254 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_count_l4043_404383


namespace NUMINAMATH_CALUDE_car_travel_theorem_l4043_404375

/-- Represents the distance-time relationship for a car traveling between two points --/
def distance_function (initial_distance : ℝ) (speed : ℝ) (time : ℝ) : ℝ :=
  initial_distance - speed * time

theorem car_travel_theorem (initial_distance speed time : ℝ) 
  (h1 : initial_distance = 120)
  (h2 : speed = 80)
  (h3 : 0 ≤ time)
  (h4 : time ≤ 1.5) :
  let y := distance_function initial_distance speed time
  ∀ x, x = time → y = 120 - 80 * x ∧ 
  (x = 0.8 → y = 56) := by
  sorry

#check car_travel_theorem

end NUMINAMATH_CALUDE_car_travel_theorem_l4043_404375


namespace NUMINAMATH_CALUDE_monomial_count_l4043_404378

-- Define a type for algebraic expressions
inductive AlgebraicExpr
  | Constant (c : ℚ)
  | Variable (v : String)
  | Product (e1 e2 : AlgebraicExpr)
  | Sum (e1 e2 : AlgebraicExpr)
  | Fraction (num den : AlgebraicExpr)

-- Define what a monomial is
def isMonomial : AlgebraicExpr → Bool
  | AlgebraicExpr.Constant _ => true
  | AlgebraicExpr.Variable _ => true
  | AlgebraicExpr.Product e1 e2 => isMonomial e1 && isMonomial e2
  | _ => false

-- Define the list of given expressions
def givenExpressions : List AlgebraicExpr := [
  AlgebraicExpr.Product (AlgebraicExpr.Constant (-1/2)) (AlgebraicExpr.Product (AlgebraicExpr.Variable "m") (AlgebraicExpr.Variable "n")),
  AlgebraicExpr.Variable "m",
  AlgebraicExpr.Constant (1/2),
  AlgebraicExpr.Fraction (AlgebraicExpr.Variable "b") (AlgebraicExpr.Variable "a"),
  AlgebraicExpr.Sum (AlgebraicExpr.Product (AlgebraicExpr.Constant 2) (AlgebraicExpr.Variable "m")) (AlgebraicExpr.Constant 1),
  AlgebraicExpr.Fraction (AlgebraicExpr.Sum (AlgebraicExpr.Variable "x") (AlgebraicExpr.Product (AlgebraicExpr.Constant (-1)) (AlgebraicExpr.Variable "y"))) (AlgebraicExpr.Constant 5),
  AlgebraicExpr.Fraction 
    (AlgebraicExpr.Sum (AlgebraicExpr.Product (AlgebraicExpr.Constant 2) (AlgebraicExpr.Variable "x")) (AlgebraicExpr.Variable "y"))
    (AlgebraicExpr.Sum (AlgebraicExpr.Variable "x") (AlgebraicExpr.Product (AlgebraicExpr.Constant (-1)) (AlgebraicExpr.Variable "y"))),
  AlgebraicExpr.Sum 
    (AlgebraicExpr.Sum 
      (AlgebraicExpr.Product (AlgebraicExpr.Variable "x") (AlgebraicExpr.Variable "x")) 
      (AlgebraicExpr.Product (AlgebraicExpr.Constant 2) (AlgebraicExpr.Variable "x")))
    (AlgebraicExpr.Constant (3/2))
]

-- Theorem statement
theorem monomial_count : 
  (givenExpressions.filter isMonomial).length = 3 := by sorry

end NUMINAMATH_CALUDE_monomial_count_l4043_404378


namespace NUMINAMATH_CALUDE_optimal_selling_price_l4043_404390

/-- Represents the selling price of grapes in yuan per kilogram -/
def selling_price : ℝ := 21

/-- Represents the cost price of grapes in yuan per kilogram -/
def cost_price : ℝ := 16

/-- Represents the daily sales volume in kilograms when the price is 26 yuan -/
def base_sales : ℝ := 320

/-- Represents the increase in sales volume for each yuan decrease in price -/
def sales_increase_rate : ℝ := 80

/-- Represents the target daily profit in yuan -/
def target_profit : ℝ := 3600

/-- Calculates the daily sales volume based on the selling price -/
def sales_volume (x : ℝ) : ℝ := base_sales + sales_increase_rate * (26 - x)

/-- Calculates the daily profit based on the selling price -/
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * sales_volume x

/-- Theorem stating that the chosen selling price satisfies the profit goal and is optimal -/
theorem optimal_selling_price : 
  daily_profit selling_price = target_profit ∧ 
  (∀ y, y < selling_price → daily_profit y < target_profit) :=
sorry

end NUMINAMATH_CALUDE_optimal_selling_price_l4043_404390


namespace NUMINAMATH_CALUDE_table_satisfies_function_l4043_404307

def f (x : ℝ) : ℝ := 100 - 5*x - 5*x^2

theorem table_satisfies_function : 
  (f 0 = 100) ∧ 
  (f 1 = 90) ∧ 
  (f 2 = 70) ∧ 
  (f 3 = 40) ∧ 
  (f 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_table_satisfies_function_l4043_404307


namespace NUMINAMATH_CALUDE_problem_statement_l4043_404365

theorem problem_statement (a b m n c : ℝ) 
  (h1 : a + b = 0) 
  (h2 : m * n = 1) 
  (h3 : |c| = 3) : 
  a + b + m * n - |c| = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4043_404365


namespace NUMINAMATH_CALUDE_min_value_theorem_l4043_404360

theorem min_value_theorem (x y : ℝ) 
  (h1 : x > 1/6) 
  (h2 : y > 0) 
  (h3 : x + y = 1/3) : 
  (∀ a b : ℝ, a > 1/6 ∧ b > 0 ∧ a + b = 1/3 → 
    1/(6*a - 1) + 6/b ≥ 1/(6*x - 1) + 6/y) ∧ 
  1/(6*x - 1) + 6/y = 49 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4043_404360


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l4043_404341

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 18 → n * exterior_angle = 360 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l4043_404341


namespace NUMINAMATH_CALUDE_outer_circle_radius_l4043_404322

/-- Given a circular race track with an inner circumference of 440 meters and a width of 14 meters,
    the radius of the outer circle is equal to (440 / (2 * π)) + 14 meters. -/
theorem outer_circle_radius (inner_circumference : ℝ) (track_width : ℝ) 
    (h1 : inner_circumference = 440)
    (h2 : track_width = 14) : 
    (inner_circumference / (2 * Real.pi) + track_width) = (440 / (2 * Real.pi) + 14) := by
  sorry

#check outer_circle_radius

end NUMINAMATH_CALUDE_outer_circle_radius_l4043_404322


namespace NUMINAMATH_CALUDE_slope_of_solutions_l4043_404395

/-- The equation that defines the relationship between x and y -/
def equation (x y : ℝ) : Prop := 2 / x + 3 / y = 0

/-- Theorem: The slope of the line determined by any two distinct solutions to the equation is -3/2 -/
theorem slope_of_solutions (x₁ y₁ x₂ y₂ : ℝ) (h₁ : equation x₁ y₁) (h₂ : equation x₂ y₂) (h_dist : (x₁, y₁) ≠ (x₂, y₂)) :
  (y₂ - y₁) / (x₂ - x₁) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_solutions_l4043_404395


namespace NUMINAMATH_CALUDE_five_digit_divisibility_l4043_404308

theorem five_digit_divisibility (a b c d e : ℕ) (h1 : a ≥ 1) (h2 : a ≤ 9) 
  (h3 : b ≤ 9) (h4 : c ≤ 9) (h5 : d ≤ 9) (h6 : e ≤ 9) :
  let n := 10000 * a + 1000 * b + 100 * c + 10 * d + e
  let m := 1000 * a + 100 * b + 10 * d + e
  (∃ k : ℕ, n = k * m) →
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ 100 * c = (k - 1) * m :=
by sorry

end NUMINAMATH_CALUDE_five_digit_divisibility_l4043_404308


namespace NUMINAMATH_CALUDE_integer_root_values_l4043_404327

/-- The polynomial for which we're finding integer roots -/
def P (a : ℤ) (x : ℤ) : ℤ := x^3 + 2*x^2 + a*x + 10

/-- The set of possible values for a -/
def A : Set ℤ := {-1210, -185, -26, -13, -11, -10, 65, 790}

/-- Theorem stating that A contains exactly the values of a for which P has an integer root -/
theorem integer_root_values (a : ℤ) : 
  (∃ x : ℤ, P a x = 0) ↔ a ∈ A :=
sorry

end NUMINAMATH_CALUDE_integer_root_values_l4043_404327


namespace NUMINAMATH_CALUDE_probability_heart_then_king_l4043_404350

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Fin 52)

/-- Represents the suit of a card -/
inductive Suit
| Hearts | Diamonds | Clubs | Spades

/-- Represents the rank of a card -/
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A function to determine if a card is a heart -/
def is_heart (card : Fin 52) : Prop := sorry

/-- A function to determine if a card is a king -/
def is_king (card : Fin 52) : Prop := sorry

/-- The number of hearts in a standard deck -/
def num_hearts : Nat := 13

/-- The number of kings in a standard deck -/
def num_kings : Nat := 4

/-- The probability of drawing a heart as the first card and a king as the second card -/
theorem probability_heart_then_king (d : Deck) :
  (num_hearts / d.cards.val) * (num_kings / (d.cards.val - 1)) = 1 / d.cards.val :=
sorry

end NUMINAMATH_CALUDE_probability_heart_then_king_l4043_404350


namespace NUMINAMATH_CALUDE_sum_of_38_and_twice_43_l4043_404330

theorem sum_of_38_and_twice_43 : 38 + 2 * 43 = 124 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_38_and_twice_43_l4043_404330


namespace NUMINAMATH_CALUDE_missing_score_is_90_l4043_404340

def known_scores : List ℕ := [85, 90, 87, 93]

theorem missing_score_is_90 (x : ℕ) :
  (x :: known_scores).sum / (x :: known_scores).length = 89 →
  x = 90 := by
  sorry

end NUMINAMATH_CALUDE_missing_score_is_90_l4043_404340


namespace NUMINAMATH_CALUDE_space_geometry_statements_l4043_404336

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (lineParallelPlane : Line → Plane → Prop)
variable (linePerpendicularPlane : Line → Plane → Prop)
variable (planeParallelPlane : Plane → Plane → Prop)
variable (planePerpendicularPlane : Plane → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- Define the theorem
theorem space_geometry_statements 
  (m n : Line) (α β : Plane) (A : Point) :
  (∀ l₁ l₂ p, parallel l₁ l₂ → lineParallelPlane l₂ p → lineParallelPlane l₁ p) ∧
  (parallel m n → linePerpendicularPlane n β → lineParallelPlane m α → planePerpendicularPlane α β) ∧
  (intersect m n → lineParallelPlane m α → lineParallelPlane m β → 
   lineParallelPlane n α → lineParallelPlane n β → planeParallelPlane α β) :=
by sorry

end NUMINAMATH_CALUDE_space_geometry_statements_l4043_404336


namespace NUMINAMATH_CALUDE_blueprint_to_actual_length_l4043_404304

/-- Represents the scale of the blueprint in meters per inch -/
def scale : ℝ := 50

/-- Represents the length of the line segment on the blueprint in inches -/
def blueprint_length : ℝ := 7.5

/-- Represents the actual length in meters that the blueprint line segment represents -/
def actual_length : ℝ := blueprint_length * scale

theorem blueprint_to_actual_length : actual_length = 375 := by
  sorry

end NUMINAMATH_CALUDE_blueprint_to_actual_length_l4043_404304


namespace NUMINAMATH_CALUDE_point_translation_on_sine_curves_l4043_404363

theorem point_translation_on_sine_curves : ∃ (t s : ℝ),
  -- P(π/4, t) is on y = sin(x - π/12)
  t = Real.sin (π / 4 - π / 12) ∧
  -- s > 0
  s > 0 ∧
  -- P' is on y = sin(2x) after translation
  Real.sin (2 * (π / 4 - s)) = t ∧
  -- t = 1/2
  t = 1 / 2 ∧
  -- Minimum value of s = π/6
  s = π / 6 ∧
  -- s is the minimum positive value satisfying the conditions
  ∀ (s' : ℝ), s' > 0 → Real.sin (2 * (π / 4 - s')) = t → s ≤ s' := by
sorry

end NUMINAMATH_CALUDE_point_translation_on_sine_curves_l4043_404363


namespace NUMINAMATH_CALUDE_range_of_a_l4043_404310

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Icc 1 4 ∪ Set.Iic (-2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4043_404310


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_value_l4043_404338

def M (a : ℤ) : Set ℤ := {a, 0}

def N : Set ℤ := {x : ℤ | 2 * x^2 - 5 * x < 0}

theorem intersection_nonempty_implies_a_value (a : ℤ) :
  (M a ∩ N).Nonempty → a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_value_l4043_404338


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l4043_404347

theorem sum_of_roots_quadratic_equation :
  ∀ x₁ x₂ : ℝ, (x₁^2 - 3*x₁ + 2 = 0) ∧ (x₂^2 - 3*x₂ + 2 = 0) → x₁ + x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l4043_404347


namespace NUMINAMATH_CALUDE_canned_food_bins_l4043_404344

theorem canned_food_bins (soup : Real) (vegetables : Real) (pasta : Real)
  (h1 : soup = 0.12)
  (h2 : vegetables = 0.12)
  (h3 : pasta = 0.5) :
  soup + vegetables + pasta = 0.74 := by
  sorry

end NUMINAMATH_CALUDE_canned_food_bins_l4043_404344


namespace NUMINAMATH_CALUDE_intersection_equals_B_l4043_404366

/-- The set A of solutions to x^2 - 4x + 3 = 0 -/
def A : Set ℝ := {x | x^2 - 4*x + 3 = 0}

/-- The set B of solutions to mx + 1 = 0 for some real m -/
def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

/-- The theorem stating the set of values for m that satisfy A ∩ B = B -/
theorem intersection_equals_B : 
  {m : ℝ | A ∩ B m = B m} = {-1, -1/3, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_equals_B_l4043_404366


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_cube_l4043_404331

def y : Nat := 2^3 * 3^5 * 4^5 * 5^4 * 6^3 * 7^5 * 8^2

def is_perfect_cube (n : Nat) : Prop :=
  ∃ m : Nat, n = m^3

theorem smallest_multiplier_for_perfect_cube :
  (∀ z < 350, ¬ is_perfect_cube (y * z)) ∧ is_perfect_cube (y * 350) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_cube_l4043_404331


namespace NUMINAMATH_CALUDE_not_integer_proofs_l4043_404329

theorem not_integer_proofs (a b c d : ℝ) (n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  let E := (a/(a+b+d)) + (b/(b+c+a)) + (c/(c+d+b)) + (d/(d+a+c))
  (1 < E ∧ E < 2) ∧ (n < Real.sqrt (n^2 + n) ∧ Real.sqrt (n^2 + n) < n + 1) := by
  sorry

end NUMINAMATH_CALUDE_not_integer_proofs_l4043_404329


namespace NUMINAMATH_CALUDE_solve_system_l4043_404314

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 10) 
  (eq2 : 6 * p + 5 * q = 17) : 
  p = 52 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l4043_404314


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4043_404335

theorem arithmetic_sequence_problem (k : ℕ+) : 
  let a : ℕ → ℤ := λ n => 2*n + 2
  let S : ℕ → ℤ := λ n => n^2 + 3*n
  S k - a (k + 5) = 44 → k = 7 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4043_404335


namespace NUMINAMATH_CALUDE_pentagon_area_is_8_5_l4043_404309

-- Define the pentagon vertices
def pentagon_vertices : List (ℤ × ℤ) := [(0, 0), (1, 2), (3, 3), (4, 1), (2, 0)]

-- Define the function to calculate the area of the pentagon
def pentagon_area (vertices : List (ℤ × ℤ)) : ℚ :=
  sorry

-- Theorem statement
theorem pentagon_area_is_8_5 :
  pentagon_area pentagon_vertices = 17/2 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_is_8_5_l4043_404309


namespace NUMINAMATH_CALUDE_complex_quotient_pure_imaginary_l4043_404386

theorem complex_quotient_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 2*Complex.I
  let z₂ : ℂ := 3 - 4*Complex.I
  (∃ b : ℝ, z₁ / z₂ = b*Complex.I ∧ b ≠ 0) → a = 8/3 := by
sorry

end NUMINAMATH_CALUDE_complex_quotient_pure_imaginary_l4043_404386


namespace NUMINAMATH_CALUDE_equation_solution_l4043_404367

theorem equation_solution (a : ℝ) (h1 : a ≠ -2) (h2 : a ≠ -3) (h3 : a ≠ 1/2) :
  let x : ℝ := (2*a - 1) / (a + 3)
  (2 : ℝ) ^ ((a + 3) / (a + 2)) * (32 : ℝ) ^ (1 / (x * (a + 2))) = (4 : ℝ) ^ (1 / x) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4043_404367


namespace NUMINAMATH_CALUDE_karting_track_routes_l4043_404370

/-- Represents the number of distinct routes ending at point A after n minutes -/
def M : ℕ → ℕ
| 0 => 0
| 1 => 0
| (n+2) => M (n+1) + M n

/-- The karting track problem -/
theorem karting_track_routes : M 10 = 34 := by
  sorry

end NUMINAMATH_CALUDE_karting_track_routes_l4043_404370


namespace NUMINAMATH_CALUDE_integral_evaluation_l4043_404387

open Real

theorem integral_evaluation : 
  ∫ (x : ℝ) in Real.arccos (1 / Real.sqrt 10)..Real.arccos (1 / Real.sqrt 26), 
    12 / ((6 + 5 * tan x) * sin (2 * x)) = log (105 / 93) := by
  sorry

end NUMINAMATH_CALUDE_integral_evaluation_l4043_404387


namespace NUMINAMATH_CALUDE_odd_sum_of_squares_l4043_404337

theorem odd_sum_of_squares (n m : ℤ) (h : Odd (n^2 + m^2)) :
  ¬(Even n ∧ Even m) ∧ ¬(Even (n + m)) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_of_squares_l4043_404337


namespace NUMINAMATH_CALUDE_fourth_student_number_l4043_404392

def systematicSampling (totalStudents : Nat) (samplesToSelect : Nat) (selected : List Nat) : Nat :=
  sorry

theorem fourth_student_number
  (totalStudents : Nat)
  (samplesToSelect : Nat)
  (selected : List Nat)
  (h1 : totalStudents = 54)
  (h2 : samplesToSelect = 4)
  (h3 : selected = [3, 29, 42])
  : systematicSampling totalStudents samplesToSelect selected = 16 :=
by sorry

end NUMINAMATH_CALUDE_fourth_student_number_l4043_404392


namespace NUMINAMATH_CALUDE_class_average_problem_l4043_404354

theorem class_average_problem (class_size : ℝ) (h_positive : class_size > 0) :
  let group1_size := 0.2 * class_size
  let group2_size := 0.5 * class_size
  let group3_size := class_size - group1_size - group2_size
  let group1_avg := 80
  let group2_avg := 60
  let overall_avg := 58
  ∃ (group3_avg : ℝ),
    (group1_size * group1_avg + group2_size * group2_avg + group3_size * group3_avg) / class_size = overall_avg ∧
    group3_avg = 40 := by
sorry

end NUMINAMATH_CALUDE_class_average_problem_l4043_404354


namespace NUMINAMATH_CALUDE_oil_depths_in_elliptical_tank_l4043_404373

/-- Represents an elliptical oil tank lying horizontally -/
structure EllipticalTank where
  length : ℝ
  majorAxis : ℝ
  minorAxis : ℝ

/-- Calculates the possible oil depths in an elliptical tank -/
def calculateOilDepths (tank : EllipticalTank) (oilSurfaceArea : ℝ) : Set ℝ :=
  sorry

/-- The theorem stating the correct oil depths for the given tank and oil surface area -/
theorem oil_depths_in_elliptical_tank :
  let tank : EllipticalTank := { length := 10, majorAxis := 8, minorAxis := 6 }
  let oilSurfaceArea : ℝ := 48
  calculateOilDepths tank oilSurfaceArea = {1.2, 4.8} := by
  sorry

end NUMINAMATH_CALUDE_oil_depths_in_elliptical_tank_l4043_404373


namespace NUMINAMATH_CALUDE_largest_base_for_twelve_cubed_l4043_404372

/-- Given a natural number n and a base b, returns the sum of digits of n when represented in base b -/
def sumOfDigits (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Returns true if b is the largest base such that the sum of digits of 12^3 in base b is not 3^2 -/
def isLargestBase (b : ℕ) : Prop :=
  (sumOfDigits (12^3) b ≠ 3^2) ∧
  ∀ k > b, sumOfDigits (12^3) k = 3^2

theorem largest_base_for_twelve_cubed :
  isLargestBase 9 := by sorry

end NUMINAMATH_CALUDE_largest_base_for_twelve_cubed_l4043_404372


namespace NUMINAMATH_CALUDE_peanut_butter_servings_l4043_404364

/-- Represents the amount of peanut butter in tablespoons -/
def peanut_butter : ℚ := 37 + 2/3

/-- Represents the serving size in tablespoons -/
def serving_size : ℚ := 2 + 1/2

/-- Calculates the number of servings in the jar -/
def number_of_servings : ℚ := peanut_butter / serving_size

/-- Proves that the number of servings in the jar is equal to 15 1/15 -/
theorem peanut_butter_servings : number_of_servings = 15 + 1/15 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_servings_l4043_404364


namespace NUMINAMATH_CALUDE_oranges_thrown_away_l4043_404357

theorem oranges_thrown_away (initial_oranges new_oranges final_oranges : ℕ) : 
  initial_oranges = 31 → new_oranges = 38 → final_oranges = 60 → 
  ∃ thrown_away : ℕ, initial_oranges - thrown_away + new_oranges = final_oranges ∧ thrown_away = 9 :=
by sorry

end NUMINAMATH_CALUDE_oranges_thrown_away_l4043_404357


namespace NUMINAMATH_CALUDE_cow_count_l4043_404384

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (ac : AnimalCount) : ℕ :=
  2 * ac.ducks + 4 * ac.cows

/-- The total number of heads in the group -/
def totalHeads (ac : AnimalCount) : ℕ :=
  ac.ducks + ac.cows

/-- The problem statement -/
theorem cow_count (ac : AnimalCount) :
  totalLegs ac = 2 * totalHeads ac + 36 → ac.cows = 18 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_l4043_404384


namespace NUMINAMATH_CALUDE_frog_corner_probability_l4043_404342

/-- Represents a position on the 3x3 grid -/
inductive Position
| Center
| Edge
| Corner

/-- Represents the state of the frog's movement -/
structure State where
  position : Position
  hops : Nat

/-- Transition function for the frog's movement -/
def transition (s : State) : List State := sorry

/-- Probability of reaching a corner after exactly 4 hops -/
def probability_corner_4_hops : ℚ := sorry

/-- Main theorem stating the probability of reaching a corner after exactly 4 hops -/
theorem frog_corner_probability :
  probability_corner_4_hops = 217 / 256 := by sorry

end NUMINAMATH_CALUDE_frog_corner_probability_l4043_404342


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l4043_404359

theorem arithmetic_calculation : 10 - 9 * 8 + 7^2 - 6 / 3 * 2 + 1 = -16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l4043_404359


namespace NUMINAMATH_CALUDE_magnified_tissue_diameter_l4043_404352

/-- Calculates the diameter of a magnified image given the actual diameter and magnification factor. -/
def magnifiedDiameter (actualDiameter : ℝ) (magnificationFactor : ℝ) : ℝ :=
  actualDiameter * magnificationFactor

/-- Proves that for a tissue with actual diameter 0.0003 cm and a microscope with 1000x magnification,
    the magnified image diameter is 0.3 cm. -/
theorem magnified_tissue_diameter :
  magnifiedDiameter 0.0003 1000 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_magnified_tissue_diameter_l4043_404352


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l4043_404343

theorem polynomial_equality_implies_sum (b₁ b₂ b₃ b₄ c₁ c₂ c₃ c₄ : ℝ) :
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃) * (x^2 + b₄*x + c₄)) →
  b₁*c₁ + b₂*c₂ + b₃*c₃ + b₄*c₄ = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l4043_404343


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l4043_404300

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 - 8 = (x - 2) * (x^5 + 2*x^4 + 4*x^3 + 8*x^2 + 16*x + 32) + 56 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l4043_404300


namespace NUMINAMATH_CALUDE_average_of_c_and_d_l4043_404328

theorem average_of_c_and_d (c d e : ℝ) : 
  (4 + 6 + 9 + c + d + e) / 6 = 20 → 
  e = c + 6 → 
  (c + d) / 2 = 47.5 := by
sorry

end NUMINAMATH_CALUDE_average_of_c_and_d_l4043_404328


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l4043_404397

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_b_value :
  ∀ b : ℝ,
  let A : Point := ⟨3, 1⟩
  let B : Point := ⟨-2, b⟩
  let C : Point := ⟨8, 11⟩
  collinear A B C → b = -9 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l4043_404397


namespace NUMINAMATH_CALUDE_journey_time_ratio_and_sum_l4043_404380

/-- Represents the ratio of road segments --/
def road_ratio : Fin 3 → ℚ
| 0 => 1
| 1 => 2
| 2 => 1

/-- Represents the ratio of speeds on different road types --/
def speed_ratio : Fin 3 → ℚ
| 0 => 3
| 1 => 2
| 2 => 4

/-- Calculates the time taken for a journey --/
def journey_time (r : Fin 3 → ℚ) (s : Fin 3 → ℚ) : ℚ :=
  (r 0 / s 0) + (r 1 / s 1) + (r 2 / s 2)

/-- Theorem stating the ratio of journey times and the sum of m and n --/
theorem journey_time_ratio_and_sum :
  let to_school := journey_time road_ratio speed_ratio
  let return_home := journey_time road_ratio (fun i => speed_ratio (2 - i))
  let ratio := to_school / return_home
  ∃ (m n : ℕ), m.Coprime n ∧ ratio = n / m ∧ m + n = 35 := by
  sorry


end NUMINAMATH_CALUDE_journey_time_ratio_and_sum_l4043_404380


namespace NUMINAMATH_CALUDE_five_dozen_apple_cost_l4043_404301

/-- The cost of apples given the number of dozens and the price -/
def apple_cost (dozens : ℚ) (price : ℚ) : ℚ := dozens * (price / 4)

/-- Theorem: If 4 dozen apples cost $31.20, then 5 dozen apples at the same rate will cost $39.00 -/
theorem five_dozen_apple_cost :
  apple_cost 5 31.20 = 39.00 :=
sorry

end NUMINAMATH_CALUDE_five_dozen_apple_cost_l4043_404301


namespace NUMINAMATH_CALUDE_chessboard_division_impossible_l4043_404321

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Represents a line on the chessboard --/
structure Line

/-- Represents a division of the chessboard --/
def ChessboardDivision := List Line

/-- Function to check if a division is valid --/
def is_valid_division (board : Chessboard) (division : ChessboardDivision) : Prop :=
  sorry

/-- Theorem: It's impossible to divide an 8x8 chessboard with 13 lines
    such that each region contains at most one square center --/
theorem chessboard_division_impossible :
  ∀ (board : Chessboard) (division : ChessboardDivision),
    board.size = 8 →
    division.length = 13 →
    ¬(is_valid_division board division) :=
sorry

end NUMINAMATH_CALUDE_chessboard_division_impossible_l4043_404321


namespace NUMINAMATH_CALUDE_prime_factors_equation_l4043_404345

theorem prime_factors_equation (x : ℕ) : 22 + x + 2 = 29 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_equation_l4043_404345


namespace NUMINAMATH_CALUDE_club_officer_selection_l4043_404358

/-- The number of ways to choose officers of the same gender from a club -/
def choose_officers (total_members : ℕ) (boys : ℕ) (girls : ℕ) : ℕ :=
  2 * (boys * (boys - 1) * (boys - 2))

/-- Theorem: Choosing officers from a club with specific conditions -/
theorem club_officer_selection :
  let total_members : ℕ := 30
  let boys : ℕ := 15
  let girls : ℕ := 15
  choose_officers total_members boys girls = 5460 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l4043_404358


namespace NUMINAMATH_CALUDE_complex_number_equality_l4043_404303

theorem complex_number_equality (z : ℂ) : 
  Complex.abs (z - 1) = Complex.abs (z - Complex.I) →
  ∃ (r : ℝ), r > 0 ∧ z - (z - 6) / (z - 1) = r →
  z = 2 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_equality_l4043_404303


namespace NUMINAMATH_CALUDE_ad_greater_than_bc_l4043_404311

theorem ad_greater_than_bc (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq : a + d = b + c) 
  (abs_ineq : |a - d| < |b - c|) : 
  a * d > b * c := by
sorry

end NUMINAMATH_CALUDE_ad_greater_than_bc_l4043_404311


namespace NUMINAMATH_CALUDE_exist_integers_satisfying_equation_l4043_404382

theorem exist_integers_satisfying_equation : ∃ (a b : ℤ), a * b * (2 * a + b) = 2015 ∧ a = 13 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_exist_integers_satisfying_equation_l4043_404382


namespace NUMINAMATH_CALUDE_expression_simplification_l4043_404396

theorem expression_simplification (x y z : ℝ) 
  (h1 : x > y) (h2 : y > 1) (h3 : z > 0) : 
  (x^y * y^(x+z)) / (y^(y+z) * x^x) = (x/y)^(y-x) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4043_404396


namespace NUMINAMATH_CALUDE_max_b_letters_l4043_404320

/-- The maximum number of "B" letters that can be formed with 47 sticks -/
theorem max_b_letters (total_sticks : ℕ) (sticks_per_b : ℕ) (sticks_per_v : ℕ)
  (h_total : total_sticks = 47)
  (h_b : sticks_per_b = 4)
  (h_v : sticks_per_v = 5)
  (h_all_used : ∃ (b v : ℕ), total_sticks = b * sticks_per_b + v * sticks_per_v) :
  ∃ (max_b : ℕ), 
    (max_b * sticks_per_b ≤ total_sticks) ∧ 
    (∀ b : ℕ, b * sticks_per_b ≤ total_sticks → b ≤ max_b) ∧
    (∃ v : ℕ, total_sticks = max_b * sticks_per_b + v * sticks_per_v) ∧
    max_b = 8 :=
sorry

end NUMINAMATH_CALUDE_max_b_letters_l4043_404320


namespace NUMINAMATH_CALUDE_election_winner_percentage_l4043_404369

theorem election_winner_percentage (total_votes winner_votes margin : ℕ) : 
  winner_votes = 1044 →
  margin = 288 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 58 / 100 := by
sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l4043_404369


namespace NUMINAMATH_CALUDE_bobbo_river_crossing_l4043_404393

/-- Bobbo's river crossing problem -/
theorem bobbo_river_crossing 
  (river_width : ℝ)
  (initial_speed : ℝ)
  (current_speed : ℝ)
  (waterfall_distance : ℝ)
  (h_river_width : river_width = 100)
  (h_initial_speed : initial_speed = 2)
  (h_current_speed : current_speed = 5)
  (h_waterfall_distance : waterfall_distance = 175) :
  let midway_distance := river_width / 2
  let time_to_midway := midway_distance / initial_speed
  let downstream_distance := current_speed * time_to_midway
  let remaining_distance := waterfall_distance - downstream_distance
  let time_left := remaining_distance / current_speed
  let required_speed := midway_distance / time_left
  required_speed - initial_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_bobbo_river_crossing_l4043_404393


namespace NUMINAMATH_CALUDE_smallest_n_for_f_greater_than_15_l4043_404356

def digit_sum (x : ℚ) : ℕ :=
  sorry

def f (n : ℕ+) : ℕ :=
  digit_sum ((1 : ℚ) / (7 ^ (n : ℕ)))

theorem smallest_n_for_f_greater_than_15 :
  ∀ k : ℕ+, k < 7 → f k ≤ 15 ∧ f 7 > 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_f_greater_than_15_l4043_404356


namespace NUMINAMATH_CALUDE_bottle_production_l4043_404371

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 8 such machines will produce 1440 bottles in 4 minutes. -/
theorem bottle_production (machines_base : ℕ) (bottles_per_minute : ℕ) (machines_new : ℕ) (minutes : ℕ)
    (h1 : machines_base = 6)
    (h2 : bottles_per_minute = 270)
    (h3 : machines_new = 8)
    (h4 : minutes = 4) :
    (machines_new * (bottles_per_minute / machines_base) * minutes) = 1440 :=
by sorry

end NUMINAMATH_CALUDE_bottle_production_l4043_404371


namespace NUMINAMATH_CALUDE_optimal_strategy_l4043_404353

/-- Represents the price of bananas on each day of Marina's trip. -/
def banana_prices : List ℝ := [1, 5, 1, 6, 7, 8, 1, 8, 7, 2, 7, 8, 1, 9, 2, 8, 7, 1]

/-- Represents the optimal buying strategy for bananas. -/
def buying_strategy : List ℕ := [1, 1, 1, 4, 0, 0, 1, 0, 1, 4, 1, 0, 0, 0, 3, 0, 0, 2, 0]

/-- The number of days in Marina's trip. -/
def trip_length : ℕ := 18

/-- The maximum number of days a banana can be eaten after purchase. -/
def max_banana_freshness : ℕ := 4

/-- Calculates the total cost of bananas based on a given buying strategy. -/
def total_cost (strategy : List ℕ) : ℝ :=
  List.sum (List.zipWith (· * ·) strategy banana_prices)

/-- Checks if a given buying strategy is valid according to the problem constraints. -/
def is_valid_strategy (strategy : List ℕ) : Prop :=
  strategy.length = trip_length + 1 ∧
  List.sum strategy = trip_length ∧
  ∀ i, i < trip_length → List.sum (List.take (min max_banana_freshness (trip_length - i)) (List.drop i strategy)) ≥ 1

/-- Theorem stating that the given buying strategy is optimal. -/
theorem optimal_strategy :
  is_valid_strategy buying_strategy ∧
  ∀ other_strategy, is_valid_strategy other_strategy →
    total_cost buying_strategy ≤ total_cost other_strategy :=
sorry

end NUMINAMATH_CALUDE_optimal_strategy_l4043_404353


namespace NUMINAMATH_CALUDE_walker_children_puzzle_l4043_404398

def is_aabb (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a * 1000 + a * 100 + b * 10 + b ∧ 0 < a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

def divisible_by_nine_out_of_ten (n : ℕ) : Prop :=
  ∃ k : ℕ, k ∈ (Finset.range 10).filter (λ i => i ≠ 0) ∧
    ∀ i ∈ (Finset.range 10).filter (λ i => i ≠ 0), i ≠ k → n % i = 0

theorem walker_children_puzzle :
  ∀ n : ℕ, is_aabb n → divisible_by_nine_out_of_ten n →
    ∃ (x y : ℕ), x + y = n → 
      ∃ k : ℕ, k ∈ (Finset.range 10).filter (λ i => i ≠ 0) ∧ n % k ≠ 0 ∧ k = 9 :=
sorry

end NUMINAMATH_CALUDE_walker_children_puzzle_l4043_404398


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_n_l4043_404399

def n : ℕ := 15^4 - 9^4

theorem largest_power_of_two_dividing_n : 
  ∃ k : ℕ, k = 5 ∧ 2^k ∣ n ∧ ∀ m : ℕ, 2^m ∣ n → m ≤ k :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_n_l4043_404399


namespace NUMINAMATH_CALUDE_cosine_angle_POQ_l4043_404319

/-- Given points P, Q, and O, prove that the cosine of angle POQ is -√10/10 -/
theorem cosine_angle_POQ :
  let P : ℝ × ℝ := (1, 1)
  let Q : ℝ × ℝ := (-2, 1)
  let O : ℝ × ℝ := (0, 0)
  let OP : ℝ × ℝ := (P.1 - O.1, P.2 - O.2)
  let OQ : ℝ × ℝ := (Q.1 - O.1, Q.2 - O.2)
  let dot_product : ℝ := OP.1 * OQ.1 + OP.2 * OQ.2
  let magnitude_OP : ℝ := Real.sqrt (OP.1^2 + OP.2^2)
  let magnitude_OQ : ℝ := Real.sqrt (OQ.1^2 + OQ.2^2)
  dot_product / (magnitude_OP * magnitude_OQ) = -Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cosine_angle_POQ_l4043_404319


namespace NUMINAMATH_CALUDE_f_properties_l4043_404388

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x - 1) / x - a * x + a

theorem f_properties (a : ℝ) :
  (∀ x y, 0 < x ∧ x < y → a ≤ 0 → f a x < f a y) ∧ 
  (∃ x₀, 0 < x₀ ∧ x₀ < 1 ∧ f a x₀ = Real.exp 1 - 1 → a < 1) ∧
  ¬(a < 1 → ∃ x₀, 0 < x₀ ∧ x₀ < 1 ∧ f a x₀ = Real.exp 1 - 1 ∧ 
    ∀ x, 0 < x ∧ x < 1 ∧ x ≠ x₀ → f a x ≠ Real.exp 1 - 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4043_404388


namespace NUMINAMATH_CALUDE_doris_erasers_taken_out_l4043_404317

/-- The number of erasers Doris took out of a box -/
def erasers_taken_out (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

theorem doris_erasers_taken_out :
  let initial := 69
  let left := 15
  erasers_taken_out initial left = 54 := by sorry

end NUMINAMATH_CALUDE_doris_erasers_taken_out_l4043_404317


namespace NUMINAMATH_CALUDE_eighth_odd_multiple_of_five_l4043_404333

def arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

theorem eighth_odd_multiple_of_five : 
  ∀ (a d : ℕ),
    a = 5 → 
    d = 10 → 
    (∀ n : ℕ, n > 0 → arithmetic_sequence a d n % 2 = 1) →
    (∀ n : ℕ, n > 0 → arithmetic_sequence a d n % 5 = 0) →
    arithmetic_sequence a d 8 = 75 :=
by sorry

end NUMINAMATH_CALUDE_eighth_odd_multiple_of_five_l4043_404333


namespace NUMINAMATH_CALUDE_mathborough_rainfall_2005_l4043_404325

/-- Rainfall data for Mathborough from 2003 to 2005 -/
structure RainfallData where
  rainfall_2003 : ℝ
  increase_2004 : ℝ
  increase_2005 : ℝ

/-- Calculate the total rainfall in Mathborough for 2005 -/
def totalRainfall2005 (data : RainfallData) : ℝ :=
  12 * (data.rainfall_2003 + data.increase_2004 + data.increase_2005)

/-- Theorem stating the total rainfall in Mathborough for 2005 -/
theorem mathborough_rainfall_2005 (data : RainfallData)
  (h1 : data.rainfall_2003 = 50)
  (h2 : data.increase_2004 = 5)
  (h3 : data.increase_2005 = 3) :
  totalRainfall2005 data = 696 := by
  sorry

#eval totalRainfall2005 ⟨50, 5, 3⟩

end NUMINAMATH_CALUDE_mathborough_rainfall_2005_l4043_404325


namespace NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l4043_404302

def is_nonprime (n : ℕ) : Prop := ¬(Nat.Prime n) ∧ n > 1

def has_no_small_prime_factor (n : ℕ) : Prop := ∀ p, Nat.Prime p → p < 20 → ¬(p ∣ n)

theorem smallest_nonprime_with_large_factors : 
  ∃ n : ℕ, is_nonprime n ∧ has_no_small_prime_factor n ∧ 
  (∀ m : ℕ, m < n → ¬(is_nonprime m ∧ has_no_small_prime_factor m)) ∧
  n = 529 :=
sorry

end NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l4043_404302


namespace NUMINAMATH_CALUDE_equation_implications_l4043_404351

theorem equation_implications (x y : ℝ) (h : x^2 - 2*x*y + 2*y^2 = 1) :
  (abs x ≤ Real.sqrt 2) ∧ (x^2 + 2*y^2 > 1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_implications_l4043_404351


namespace NUMINAMATH_CALUDE_lcm_12_15_18_l4043_404374

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_15_18_l4043_404374


namespace NUMINAMATH_CALUDE_crossed_out_number_is_21_l4043_404394

def first_n_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem crossed_out_number_is_21 :
  ∀ a : ℕ, 
    a > 0 ∧ a ≤ 20 →
    (∃ k : ℕ, k > 0 ∧ k ≤ 20 ∧ k ≠ a ∧ 
      k = (first_n_sum 20 - a) / 19 ∧ 
      (first_n_sum 20 - a) % 19 = 0) →
    a = 21 :=
by sorry

end NUMINAMATH_CALUDE_crossed_out_number_is_21_l4043_404394


namespace NUMINAMATH_CALUDE_system_solution_l4043_404346

theorem system_solution (a b : ℤ) :
  (b * (-1) + 2 * 2 = 8) →
  (a * 1 + 3 * 4 = 5) →
  (a = -7 ∧ b = -4) ∧
  ((-7) * 7 + 3 * 18 = 5) ∧
  ((-4) * 7 + 2 * 18 = 8) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4043_404346


namespace NUMINAMATH_CALUDE_brown_ball_weight_l4043_404334

theorem brown_ball_weight (blue_weight : ℝ) (total_weight : ℝ) (brown_weight : ℝ) :
  blue_weight = 6 →
  total_weight = 9.12 →
  total_weight = blue_weight + brown_weight →
  brown_weight = 3.12 :=
by
  sorry

end NUMINAMATH_CALUDE_brown_ball_weight_l4043_404334


namespace NUMINAMATH_CALUDE_ellipse_parabola_equations_l4043_404385

/-- Given an ellipse and a parabola with specific properties, 
    prove their equations. -/
theorem ellipse_parabola_equations 
  (a b c p : ℝ) 
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : p > 0)
  (h4 : c / a = 1 / 2)  -- eccentricity
  (h5 : a - c = 1 / 2)  -- distance from left focus to directrix
  (h6 : a = p / 2)      -- right vertex is focus of parabola
  : (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 + 4 * y^2 / 3 = 1) ∧
    (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_parabola_equations_l4043_404385


namespace NUMINAMATH_CALUDE_min_photos_theorem_l4043_404355

/-- Represents the number of photographs for each grade --/
structure PhotoDistribution where
  total : ℕ
  fourth : ℕ
  fifth : ℕ
  sixth : ℕ
  seventh : ℕ
  first_to_third : ℕ

/-- The minimum number of photographs needed to ensure at least 15 from one grade (4th to 7th) --/
def min_photos_for_fifteen (d : PhotoDistribution) : ℕ := 
  d.first_to_third + 4 * 14 + 1

/-- The theorem stating the minimum number of photographs needed --/
theorem min_photos_theorem (d : PhotoDistribution) 
  (h_total : d.total = 130)
  (h_fourth : d.fourth = 35)
  (h_fifth : d.fifth = 30)
  (h_sixth : d.sixth = 25)
  (h_seventh : d.seventh = 20)
  (h_first_to_third : d.first_to_third = d.total - (d.fourth + d.fifth + d.sixth + d.seventh)) :
  min_photos_for_fifteen d = 77 := by
  sorry

#eval min_photos_for_fifteen ⟨130, 35, 30, 25, 20, 20⟩

end NUMINAMATH_CALUDE_min_photos_theorem_l4043_404355


namespace NUMINAMATH_CALUDE_function_expression_l4043_404332

theorem function_expression (f : ℝ → ℝ) (h : ∀ x, f (x + 2) = 2 * x + 3) :
  ∀ x, f x = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_expression_l4043_404332


namespace NUMINAMATH_CALUDE_triangle_properties_l4043_404376

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  cos B = 4/5 →
  b = 2 →
  (a = 5/3 → A = π/6) ∧
  (∀ a c, a > 0 → c > 0 → (1/2) * a * c * (3/5) ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l4043_404376
