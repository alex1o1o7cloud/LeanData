import Mathlib

namespace NUMINAMATH_CALUDE_max_value_fraction_l234_23410

theorem max_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y)^2 / (x^2 + 2*x*y + y^2) ≤ (a + b)^2 / (a^2 + 2*a*b + b^2)) →
  (a + b)^2 / (a^2 + 2*a*b + b^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l234_23410


namespace NUMINAMATH_CALUDE_range_of_m_l234_23463

theorem range_of_m (m : ℝ) : 
  (∃ x₀ ∈ Set.Icc 1 2, x₀^2 - m*x₀ + 4 > 0) ↔ m < 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l234_23463


namespace NUMINAMATH_CALUDE_complex_equality_implies_a_value_l234_23469

theorem complex_equality_implies_a_value (a : ℝ) : 
  (Complex.re ((1 + 2*I) * (2*a + I)) = Complex.im ((1 + 2*I) * (2*a + I))) → 
  a = -5/2 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_implies_a_value_l234_23469


namespace NUMINAMATH_CALUDE_arrange_six_books_three_identical_l234_23443

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (identical : ℕ) : ℕ :=
  (Nat.factorial total) / (Nat.factorial identical)

/-- Theorem: Arranging 6 books with 3 identical copies results in 120 ways -/
theorem arrange_six_books_three_identical :
  arrange_books 6 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_arrange_six_books_three_identical_l234_23443


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l234_23480

theorem complex_magnitude_equation (z : ℂ) : 
  (z + Complex.I) * (1 - Complex.I) = 1 → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l234_23480


namespace NUMINAMATH_CALUDE_unique_solution_x_zero_l234_23401

theorem unique_solution_x_zero (x y : ℝ) : 
  y = 2 * x → (3 * y^2 + y + 4 = 2 * (6 * x^2 + y + 2)) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_x_zero_l234_23401


namespace NUMINAMATH_CALUDE_nickel_quarter_problem_l234_23442

theorem nickel_quarter_problem (total : ℚ) (nickel_value : ℚ) (quarter_value : ℚ) :
  total = 12 →
  nickel_value = 0.05 →
  quarter_value = 0.25 →
  ∃ (n : ℕ), n * nickel_value + n * quarter_value = total ∧ n = 40 :=
by sorry

end NUMINAMATH_CALUDE_nickel_quarter_problem_l234_23442


namespace NUMINAMATH_CALUDE_trigonometric_identities_l234_23406

theorem trigonometric_identities (α : Real) (h : Real.tan (π + α) = -1/2) :
  (2 * Real.cos (π - α) - 3 * Real.sin (π + α)) / (4 * Real.cos (α - 2*π) + Real.sin (4*π - α)) = -7/9 ∧
  Real.sin (α - 7*π) * Real.cos (α + 5*π) = -2/5 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l234_23406


namespace NUMINAMATH_CALUDE_x_power_five_minus_twenty_seven_x_squared_l234_23461

theorem x_power_five_minus_twenty_seven_x_squared (x : ℝ) (h : x^3 - 3*x = 5) :
  x^5 - 27*x^2 = -22*x^2 + 9*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_x_power_five_minus_twenty_seven_x_squared_l234_23461


namespace NUMINAMATH_CALUDE_solve_equation_l234_23475

theorem solve_equation (x : ℝ) :
  Real.sqrt (3 / x + 3) = 5 / 3 → x = -27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l234_23475


namespace NUMINAMATH_CALUDE_cherry_tomato_jars_l234_23494

theorem cherry_tomato_jars (total_tomatoes : ℕ) (tomatoes_per_jar : ℕ) (h1 : total_tomatoes = 56) (h2 : tomatoes_per_jar = 8) :
  (total_tomatoes / tomatoes_per_jar : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_cherry_tomato_jars_l234_23494


namespace NUMINAMATH_CALUDE_sum_range_l234_23451

theorem sum_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) 
  (h : a^2 - a + b^2 - b + a*b = 0) : 
  1 < a + b ∧ a + b < 4/3 := by
sorry

end NUMINAMATH_CALUDE_sum_range_l234_23451


namespace NUMINAMATH_CALUDE_sum_of_base8_digits_of_888_l234_23498

/-- Given a natural number n and a base b, returns the list of digits of n in base b -/
def toDigits (n : ℕ) (b : ℕ) : List ℕ := sorry

/-- The sum of a list of natural numbers -/
def sum (l : List ℕ) : ℕ := sorry

theorem sum_of_base8_digits_of_888 :
  sum (toDigits 888 8) = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_base8_digits_of_888_l234_23498


namespace NUMINAMATH_CALUDE_traveler_distance_l234_23455

theorem traveler_distance (north south west east : ℝ) : 
  north = 25 → 
  south = 10 → 
  west = 15 → 
  east = 7 → 
  let net_north := north - south
  let net_west := west - east
  Real.sqrt (net_north ^ 2 + net_west ^ 2) = 17 :=
by sorry

end NUMINAMATH_CALUDE_traveler_distance_l234_23455


namespace NUMINAMATH_CALUDE_rectangular_equation_focus_directrix_distance_l234_23445

-- Define the polar coordinate equation of the conic section curve C
def polarEquation (ρ θ : ℝ) : Prop :=
  ρ = 8 * Real.sin θ / (1 + Real.cos (2 * θ))

-- Define the conversion from polar to rectangular coordinates
def polarToRectangular (x y ρ θ : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Theorem: The rectangular coordinate equation of curve C is x² = 4y
theorem rectangular_equation (x y : ℝ) :
  (∃ ρ θ, polarEquation ρ θ ∧ polarToRectangular x y ρ θ) →
  x^2 = 4*y :=
sorry

-- Theorem: The distance from the focus to the directrix is 2
theorem focus_directrix_distance :
  (∃ p : ℝ, ∀ x y : ℝ, (∃ ρ θ, polarEquation ρ θ ∧ polarToRectangular x y ρ θ) →
    y = (1 / (4 * p)) * x^2) →
  2 = 2 :=
sorry

end NUMINAMATH_CALUDE_rectangular_equation_focus_directrix_distance_l234_23445


namespace NUMINAMATH_CALUDE_expand_expression_l234_23477

theorem expand_expression (a : ℝ) : 4 * a^2 * (3*a - 1) = 12*a^3 - 4*a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l234_23477


namespace NUMINAMATH_CALUDE_power_sum_of_i_l234_23440

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23456 + i^23457 + i^23458 + i^23459 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l234_23440


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l234_23412

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 169
def C₂ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 9

-- Define the moving circle
structure MovingCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency conditions
def internallyTangent (c : MovingCircle) : Prop :=
  C₁ (c.center.1 - c.radius) (c.center.2)

def externallyTangent (c : MovingCircle) : Prop :=
  C₂ (c.center.1 + c.radius) (c.center.2)

-- Define the trajectory
def trajectory (x y : ℝ) : Prop := x^2 / 64 + y^2 / 48 = 1

-- The theorem to prove
theorem moving_circle_trajectory :
  ∀ (c : MovingCircle),
    internallyTangent c →
    externallyTangent c →
    trajectory c.center.1 c.center.2 :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l234_23412


namespace NUMINAMATH_CALUDE_parabola_c_value_l234_23432

/-- A parabola with equation x = ay² + by + c, vertex at (5, -3), and passing through (7, 1) has c = 49/8 -/
theorem parabola_c_value (a b c : ℝ) : 
  (∀ y : ℝ, 5 = a * (-3)^2 + b * (-3) + c) →  -- vertex condition
  (7 = a * 1^2 + b * 1 + c) →                 -- point condition
  (c = 49/8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l234_23432


namespace NUMINAMATH_CALUDE_pyramid_volume_l234_23449

theorem pyramid_volume (base_length : ℝ) (base_width : ℝ) (height : ℝ) :
  base_length = 1/2 →
  base_width = 1/4 →
  height = 1 →
  (1/3) * base_length * base_width * height = 1/24 := by
sorry

end NUMINAMATH_CALUDE_pyramid_volume_l234_23449


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l234_23468

theorem sum_of_three_numbers (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 10 →
  (a + b + c) / 3 = a + 15 →
  (a + b + c) / 3 = c - 20 →
  c = 2 * a →
  a + b + c = 115 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l234_23468


namespace NUMINAMATH_CALUDE_hyperbola_equation_l234_23466

-- Define the hyperbola C
def hyperbola_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  y = (Real.sqrt 5 / 2) * x

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 3 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∀ x y a b : ℝ,
  hyperbola_C x y a b →
  (∃ x₀ y₀, asymptote x₀ y₀) →
  (∃ x₁ y₁, ellipse x₁ y₁ ∧ 
    (x₁ - x)^2 + (y₁ - y)^2 = (x₁ + x)^2 + (y₁ + y)^2) →
  x^2 / 4 - y^2 / 5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l234_23466


namespace NUMINAMATH_CALUDE_power_of_i_sum_l234_23467

theorem power_of_i_sum (i : ℂ) : i^2 = -1 → i^44 + i^444 + 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_i_sum_l234_23467


namespace NUMINAMATH_CALUDE_chess_tournament_results_l234_23484

/-- Represents a chess tournament with given conditions -/
structure ChessTournament where
  n : ℕ  -- number of players
  total_score : ℕ
  one_player_score : ℕ
  h1 : total_score = 210
  h2 : one_player_score = 12
  h3 : n * (n - 1) = total_score

/-- Theorem stating the main results of the tournament analysis -/
theorem chess_tournament_results (t : ChessTournament) :
  (t.n = 15) ∧ 
  (∃ (max_squares : ℕ), max_squares = 33 ∧ 
    ∀ (squares : ℕ), (squares = number_of_squares_knight_can_reach_in_two_moves) → 
      squares ≤ max_squares) ∧
  (∃ (winner_score : ℕ), winner_score > t.one_player_score) :=
sorry

/-- Helper function to calculate the number of squares a knight can reach in two moves -/
def number_of_squares_knight_can_reach_in_two_moves : ℕ :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_results_l234_23484


namespace NUMINAMATH_CALUDE_amount_distribution_l234_23441

theorem amount_distribution (total : ℕ) : 
  (total / 14 = total / 18 + 80) → total = 5040 := by
  sorry

end NUMINAMATH_CALUDE_amount_distribution_l234_23441


namespace NUMINAMATH_CALUDE_rectangle_y_value_l234_23458

theorem rectangle_y_value (y : ℝ) : 
  let vertices : List (ℝ × ℝ) := [(-2, y), (6, y), (-2, 2), (6, 2)]
  let length : ℝ := 6 - (-2)
  let height : ℝ := y - 2
  let area : ℝ := length * height
  (vertices.length = 4 ∧ area = 64) → y = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l234_23458


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l234_23490

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l234_23490


namespace NUMINAMATH_CALUDE_ratio_equality_l234_23481

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc : a^2 + b^2 + c^2 = 49)
  (sum_xyz : x^2 + y^2 + z^2 = 16)
  (dot_product : a*x + b*y + c*z = 28) :
  (a + b + c) / (x + y + z) = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l234_23481


namespace NUMINAMATH_CALUDE_octal_74532_to_decimal_l234_23486

def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

theorem octal_74532_to_decimal :
  octal_to_decimal [2, 3, 5, 4, 7] = 31066 := by
  sorry

end NUMINAMATH_CALUDE_octal_74532_to_decimal_l234_23486


namespace NUMINAMATH_CALUDE_lucy_current_age_l234_23460

/-- Lucy's current age -/
def lucy_age : ℕ := sorry

/-- Lovely's current age -/
def lovely_age : ℕ := sorry

/-- Lucy's age was three times Lovely's age 5 years ago -/
axiom past_age_relation : lucy_age - 5 = 3 * (lovely_age - 5)

/-- Lucy's age will be twice Lovely's age 10 years from now -/
axiom future_age_relation : lucy_age + 10 = 2 * (lovely_age + 10)

/-- Lucy's current age is 50 -/
theorem lucy_current_age : lucy_age = 50 := by sorry

end NUMINAMATH_CALUDE_lucy_current_age_l234_23460


namespace NUMINAMATH_CALUDE_nut_is_composed_of_prism_and_cylinder_l234_23437

-- Define the types for geometric bodies
inductive GeometricBody
| RegularHexagonalPrism
| Cylinder
| Nut

-- Define the composition of a nut
def nut_composition : List GeometricBody := [GeometricBody.RegularHexagonalPrism, GeometricBody.Cylinder]

-- Theorem statement
theorem nut_is_composed_of_prism_and_cylinder :
  (GeometricBody.Nut ∈ [GeometricBody.RegularHexagonalPrism, GeometricBody.Cylinder]) →
  (nut_composition.length = 2) →
  (nut_composition = [GeometricBody.RegularHexagonalPrism, GeometricBody.Cylinder]) :=
by sorry

end NUMINAMATH_CALUDE_nut_is_composed_of_prism_and_cylinder_l234_23437


namespace NUMINAMATH_CALUDE_onion_price_per_pound_l234_23465

/-- Represents the price and quantity of ingredients --/
structure Ingredient where
  name : String
  quantity : ℝ
  price_per_unit : ℝ

/-- Represents the ratatouille recipe --/
def Recipe : List Ingredient := [
  ⟨"eggplants", 5, 2⟩,
  ⟨"zucchini", 4, 2⟩,
  ⟨"tomatoes", 4, 3.5⟩,
  ⟨"basil", 1, 5⟩  -- Price adjusted for 1 pound
]

def onion_quantity : ℝ := 3
def quart_yield : ℕ := 4
def price_per_quart : ℝ := 10

/-- Calculates the total cost of ingredients excluding onions --/
def total_cost_without_onions : ℝ :=
  Recipe.map (fun i => i.quantity * i.price_per_unit) |>.sum

/-- Calculates the target total cost --/
def target_total_cost : ℝ := quart_yield * price_per_quart

/-- Theorem: The price per pound of onions is $1.00 --/
theorem onion_price_per_pound :
  (target_total_cost - total_cost_without_onions) / onion_quantity = 1 := by
  sorry

end NUMINAMATH_CALUDE_onion_price_per_pound_l234_23465


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l234_23407

/-- The longest segment in a cylinder --/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 4) (hh : h = 10) :
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = 2 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l234_23407


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l234_23413

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The sum of specific terms in the sequence equals 20 -/
def SumEquals20 (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 + a 7 + a 9 + a 11 = 20

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumEquals20 a) : 
  a 1 + a 13 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l234_23413


namespace NUMINAMATH_CALUDE_sum_squares_ge_product_sum_l234_23487

theorem sum_squares_ge_product_sum (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) : 
  x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ x₁ * (x₂ + x₃ + x₄ + x₅) := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_ge_product_sum_l234_23487


namespace NUMINAMATH_CALUDE_sum_of_forbidden_digits_units_digit_not_in_forbidden_sum_forbidden_digits_correct_l234_23436

def S (n : ℕ+) : ℕ := n.val * (n.val + 1) / 2

def forbidden_digits : Finset ℕ := {2, 4, 7, 9}

theorem sum_of_forbidden_digits : (forbidden_digits.sum id) = 22 := by sorry

theorem units_digit_not_in_forbidden (n : ℕ+) :
  (S n) % 10 ∉ forbidden_digits := by sorry

theorem sum_forbidden_digits_correct :
  ∃ (digits : Finset ℕ), 
    (∀ (n : ℕ+), (S n) % 10 ∉ digits) ∧
    (digits.sum id = 22) ∧
    (∀ (d : ℕ), d ∉ digits → ∃ (n : ℕ+), (S n) % 10 = d) := by sorry

end NUMINAMATH_CALUDE_sum_of_forbidden_digits_units_digit_not_in_forbidden_sum_forbidden_digits_correct_l234_23436


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l234_23427

theorem right_triangle_hypotenuse (DE DF : ℝ) (P Q : ℝ × ℝ) :
  DE > 0 →
  DF > 0 →
  P.1 = DE / 4 →
  P.2 = 0 →
  Q.1 = 0 →
  Q.2 = DF / 4 →
  (DE - P.1)^2 + DF^2 = 18^2 →
  DE^2 + (DF - Q.2)^2 = 30^2 →
  DE^2 + DF^2 = (24 * Real.sqrt 3)^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l234_23427


namespace NUMINAMATH_CALUDE_triangle_foldable_to_2020_layers_l234_23479

/-- A triangle in a plane --/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- A folding method that transforms a triangle into a uniformly thick object --/
structure FoldingMethod where
  apply : Triangle → ℕ

/-- The theorem stating that any triangle can be folded into 2020 layers --/
theorem triangle_foldable_to_2020_layers :
  ∀ (t : Triangle), ∃ (f : FoldingMethod), f.apply t = 2020 :=
sorry

end NUMINAMATH_CALUDE_triangle_foldable_to_2020_layers_l234_23479


namespace NUMINAMATH_CALUDE_absolute_value_equation_l234_23439

theorem absolute_value_equation (x z : ℝ) (h : |2*x - Real.sqrt z| = 2*x + Real.sqrt z) :
  x ≥ 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l234_23439


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l234_23478

/-- Given a triangle with side lengths a, b, c, 
    prove that a^2(b+c-a) + b^2(c+a-b) + c^2(a+b-c) ≤ 3abc -/
theorem triangle_inequality_sum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l234_23478


namespace NUMINAMATH_CALUDE_remainder_2365947_div_8_l234_23409

theorem remainder_2365947_div_8 : 2365947 % 8 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_2365947_div_8_l234_23409


namespace NUMINAMATH_CALUDE_circular_plate_ratio_l234_23404

theorem circular_plate_ratio (radius : ℝ) (circumference : ℝ) 
  (h1 : radius = 15)
  (h2 : circumference = 90) :
  circumference / (2 * radius) = 3 := by
  sorry

end NUMINAMATH_CALUDE_circular_plate_ratio_l234_23404


namespace NUMINAMATH_CALUDE_music_school_tuition_cost_l234_23447

/-- The cost calculation for music school tuition with sibling discounts -/
theorem music_school_tuition_cost : 
  let base_tuition : ℕ := 45
  let first_sibling_discount : ℕ := 15
  let additional_sibling_discount : ℕ := 10
  let num_children : ℕ := 4
  
  base_tuition + 
  (base_tuition - first_sibling_discount) + 
  (base_tuition - additional_sibling_discount) + 
  (base_tuition - additional_sibling_discount) = 145 :=
by sorry

end NUMINAMATH_CALUDE_music_school_tuition_cost_l234_23447


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l234_23403

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  long_base : ℝ

/-- Calculates the area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem stating that the area of the given isosceles trapezoid is approximately 2457.2 -/
theorem isosceles_trapezoid_area :
  let t : IsoscelesTrapezoid := ⟨40, 50, 65⟩
  ∃ ε > 0, |area t - 2457.2| < ε :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l234_23403


namespace NUMINAMATH_CALUDE_inequality_system_solution_l234_23408

theorem inequality_system_solution (x : ℝ) :
  3 * x > x - 4 ∧ (4 + x) / 3 > x + 2 → -2 < x ∧ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l234_23408


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l234_23499

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose 8 n = Nat.choose 8 2) → (n = 2 ∨ n = 6) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l234_23499


namespace NUMINAMATH_CALUDE_orange_probability_is_two_sevenths_l234_23423

/-- Represents the contents of a fruit basket -/
structure FruitBasket where
  oranges : ℕ
  apples : ℕ
  bananas : ℕ

/-- The initial state of the fruit basket -/
def initialBasket : FruitBasket := sorry

/-- The state of the basket after removing some fruits -/
def updatedBasket : FruitBasket := sorry

/-- The total number of fruits in the initial basket -/
def totalFruits : ℕ := 28

/-- Assertion that the updated basket has 3 oranges and 7 apples -/
axiom updated_basket_state : updatedBasket.oranges = 3 ∧ updatedBasket.apples = 7

/-- Assertion that 5 oranges and 3 apples were removed -/
axiom fruits_removed : initialBasket.oranges = updatedBasket.oranges + 5 ∧
                       initialBasket.apples = updatedBasket.apples + 3

/-- Assertion that the total number of fruits in the initial basket is correct -/
axiom initial_total_correct : initialBasket.oranges + initialBasket.apples + initialBasket.bananas = totalFruits

/-- The probability of choosing an orange from the initial basket -/
def orangeProbability : ℚ := sorry

/-- Theorem stating that the probability of choosing an orange is 2/7 -/
theorem orange_probability_is_two_sevenths : orangeProbability = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_orange_probability_is_two_sevenths_l234_23423


namespace NUMINAMATH_CALUDE_cake_recipe_proof_l234_23488

def baking_problem (total_flour sugar_needed flour_added : ℕ) : Prop :=
  total_flour - flour_added - sugar_needed = 5

theorem cake_recipe_proof :
  baking_problem 10 3 2 := by sorry

end NUMINAMATH_CALUDE_cake_recipe_proof_l234_23488


namespace NUMINAMATH_CALUDE_triangle_inequality_l234_23497

theorem triangle_inequality (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  A + B + C = π ∧
  a = 1 ∧
  b * Real.cos A - Real.cos B = 1 →
  Real.sqrt 3 < Real.sin B + 2 * Real.sqrt 3 * Real.sin A * Real.sin A ∧
  Real.sin B + 2 * Real.sqrt 3 * Real.sin A * Real.sin A < 1 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l234_23497


namespace NUMINAMATH_CALUDE_connors_date_cost_is_36_l234_23476

/-- The cost of Connor's movie date --/
def connors_date_cost : ℝ :=
  let ticket_price : ℝ := 10
  let ticket_quantity : ℕ := 2
  let combo_meal_price : ℝ := 11
  let candy_price : ℝ := 2.5
  let candy_quantity : ℕ := 2
  ticket_price * ticket_quantity + combo_meal_price + candy_price * candy_quantity

/-- Theorem stating the total cost of Connor's date --/
theorem connors_date_cost_is_36 : connors_date_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_connors_date_cost_is_36_l234_23476


namespace NUMINAMATH_CALUDE_pattern_equality_l234_23493

/-- The product of consecutive integers from n+1 to n+n -/
def leftSide (n : ℕ) : ℕ := Finset.prod (Finset.range n) (fun i => n + i + 1)

/-- The product of odd numbers from 1 to 2n-1 -/
def oddProduct (n : ℕ) : ℕ := Finset.prod (Finset.range n) (fun i => 2 * i + 1)

/-- The theorem stating the equality of the observed pattern -/
theorem pattern_equality (n : ℕ) : leftSide n = 2^n * oddProduct n := by
  sorry

#check pattern_equality

end NUMINAMATH_CALUDE_pattern_equality_l234_23493


namespace NUMINAMATH_CALUDE_unique_prime_pair_for_equation_l234_23411

theorem unique_prime_pair_for_equation : 
  ∃! (p q : ℕ), Prime p ∧ Prime q ∧ 
  (∃ x : ℤ, x^4 + p * x^3 - q = 0) ∧ 
  p = 2 ∧ q = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_pair_for_equation_l234_23411


namespace NUMINAMATH_CALUDE_polynomial_division_l234_23420

theorem polynomial_division (x y : ℝ) (hx : x ≠ 0) :
  (15 * x^4 * y^2 - 12 * x^2 * y^3 - 3 * x^2) / (-3 * x^2) = -5 * x^2 * y^2 + 4 * y^3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l234_23420


namespace NUMINAMATH_CALUDE_certain_number_problem_l234_23446

theorem certain_number_problem (x : ℕ) (certain_number : ℕ) 
  (h1 : 9873 + x = certain_number) (h2 : x = 3327) : 
  certain_number = 13200 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l234_23446


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_fifty_l234_23435

theorem largest_multiple_of_seven_less_than_negative_fifty :
  ∀ n : ℤ, 7 ∣ n ∧ n < -50 → n ≤ -56 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_fifty_l234_23435


namespace NUMINAMATH_CALUDE_system_two_solutions_l234_23483

-- Define the system of equations
def system (a x y : ℝ) : Prop :=
  (x - a)^2 + y^2 = 64 ∧ (|x| - 8)^2 + (|y| - 15)^2 = 289

-- Define the set of values for parameter a
def valid_a_set : Set ℝ :=
  {-28} ∪ Set.Ioc (-24) (-8) ∪ Set.Ico 8 24 ∪ {28}

-- Theorem statement
theorem system_two_solutions (a : ℝ) :
  (∃! x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∨ y₁ ≠ y₂ ∧ system a x₁ y₁ ∧ system a x₂ y₂) ↔
  a ∈ valid_a_set :=
sorry

end NUMINAMATH_CALUDE_system_two_solutions_l234_23483


namespace NUMINAMATH_CALUDE_smallest_integer_solution_inequality_neg_two_satisfies_inequality_neg_two_is_smallest_integer_solution_l234_23426

theorem smallest_integer_solution_inequality :
  ∀ x : ℤ, (9*x + 8)/6 - x/3 ≥ -1 → x ≥ -2 :=
by
  sorry

theorem neg_two_satisfies_inequality :
  (9*(-2) + 8)/6 - (-2)/3 ≥ -1 :=
by
  sorry

theorem neg_two_is_smallest_integer_solution :
  ∀ y : ℤ, y < -2 → (9*y + 8)/6 - y/3 < -1 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_inequality_neg_two_satisfies_inequality_neg_two_is_smallest_integer_solution_l234_23426


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l234_23438

def total_players : ℕ := 16
def triplets : ℕ := 3
def team_size : ℕ := 7

def choose_with_triplets (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem volleyball_team_selection :
  (choose_with_triplets 3 1 * choose_with_triplets 13 6) +
  (choose_with_triplets 3 2 * choose_with_triplets 13 5) +
  (choose_with_triplets 3 3 * choose_with_triplets 13 4) = 9724 :=
by
  sorry

#check volleyball_team_selection

end NUMINAMATH_CALUDE_volleyball_team_selection_l234_23438


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l234_23462

theorem right_triangle_hypotenuse (a b h : ℝ) : 
  a = 15 ∧ b = 36 ∧ a^2 + b^2 = h^2 → h = 39 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l234_23462


namespace NUMINAMATH_CALUDE_quadratic_inequality_l234_23405

-- Define the set [1,3]
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + (a - 2) * x - 2

-- Define the solution set
def X : Set ℝ := {x : ℝ | x < -1 ∨ x > 2/3}

-- State the theorem
theorem quadratic_inequality (x : ℝ) :
  (∃ a ∈ A, f a x > 0) → x ∈ X :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l234_23405


namespace NUMINAMATH_CALUDE_no_extreme_points_iff_l234_23471

/-- A cubic function parameterized by a real number a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * a * x^2 + (a + 1) * x

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 4 * a * x + (a + 1)

/-- Theorem stating that f has no extreme points iff 0 ≤ a ≤ 3 -/
theorem no_extreme_points_iff (a : ℝ) : 
  (∀ x : ℝ, f_deriv a x ≠ 0) ↔ 0 ≤ a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_no_extreme_points_iff_l234_23471


namespace NUMINAMATH_CALUDE_dot_product_range_l234_23457

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, -1)

-- Define the curve y = √(1-x²)
def on_curve (P : ℝ × ℝ) : Prop :=
  P.2 = Real.sqrt (1 - P.1^2) ∧ 0 ≤ P.1 ∧ P.1 ≤ 1

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define the vector from B to A
def BA : ℝ × ℝ := (A.1 - B.1, A.2 - B.2)

-- Define the vector from B to P
def BP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 - B.1, P.2 - B.2)

-- Theorem statement
theorem dot_product_range :
  ∀ P : ℝ × ℝ, on_curve P →
    1 ≤ dot_product (BP P) BA ∧ dot_product (BP P) BA ≤ 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l234_23457


namespace NUMINAMATH_CALUDE_constant_in_toll_formula_l234_23495

/-- The toll formula for a truck using a certain bridge -/
def toll_formula (constant : ℝ) (x : ℕ) : ℝ :=
  constant + 1.50 * (x - 2)

/-- The number of axles on an 18-wheel truck with 2 wheels on the front axle and 4 wheels on each other axle -/
def axles_18_wheel_truck : ℕ := 5

/-- The toll for the 18-wheel truck -/
def toll_18_wheel_truck : ℝ := 6

theorem constant_in_toll_formula :
  ∃ (constant : ℝ), 
    toll_formula constant axles_18_wheel_truck = toll_18_wheel_truck ∧ 
    constant = 1.50 := by
  sorry

end NUMINAMATH_CALUDE_constant_in_toll_formula_l234_23495


namespace NUMINAMATH_CALUDE_solution_set_f_solution_set_g_l234_23491

-- Define the quadratic functions
def f (x : ℝ) := x^2 - x - 6
def g (x : ℝ) := -2*x^2 + x + 1

-- Theorem for the first inequality
theorem solution_set_f : 
  {x : ℝ | f x > 0} = {x : ℝ | x < -2 ∨ x > 3} := by sorry

-- Theorem for the second inequality
theorem solution_set_g :
  {x : ℝ | g x < 0} = {x : ℝ | x < -1/2 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_solution_set_g_l234_23491


namespace NUMINAMATH_CALUDE_hyperbola_minimum_value_l234_23485

theorem hyperbola_minimum_value (a b : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) :
  let eccentricity := (a^2 + b^2).sqrt / a
  eccentricity = 2 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (b^2 + 1) / (Real.sqrt 3 * a) ≥ 4 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_minimum_value_l234_23485


namespace NUMINAMATH_CALUDE_chef_eggs_proof_l234_23425

def initial_eggs (eggs_in_fridge : ℕ) (eggs_per_cake : ℕ) (num_cakes : ℕ) : ℕ :=
  eggs_in_fridge + eggs_per_cake * num_cakes

theorem chef_eggs_proof :
  initial_eggs 10 5 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_chef_eggs_proof_l234_23425


namespace NUMINAMATH_CALUDE_point_on_graph_l234_23419

/-- The function f(x) = -2x + 1 -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- The point we're checking -/
def point : ℝ × ℝ := (1, -1)

/-- Theorem: The point (1, -1) lies on the graph of f(x) = -2x + 1 -/
theorem point_on_graph : f point.1 = point.2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_l234_23419


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_eq_neg_eight_l234_23418

-- Define the polynomial expression
def poly (x m : ℝ) : ℝ := (x^2 - x + m) * (x - 8)

-- Theorem statement
theorem no_linear_term_implies_m_eq_neg_eight :
  (∀ x : ℝ, ∃ a b c : ℝ, poly x m = a * x^3 + b * x^2 + c) → m = -8 :=
by sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_eq_neg_eight_l234_23418


namespace NUMINAMATH_CALUDE_machine_value_theorem_l234_23474

/-- Calculates the machine's value after two years and a major overhaul -/
def machine_value_after_two_years_and_overhaul (initial_value : ℝ) : ℝ :=
  let year1_depreciation_rate := 0.10
  let year2_depreciation_rate := 0.12
  let repair_rate := 0.03
  let overhaul_rate := 0.15
  
  let value_after_year1 := initial_value * (1 - year1_depreciation_rate) * (1 + repair_rate)
  let value_after_year2 := value_after_year1 * (1 - year2_depreciation_rate) * (1 + repair_rate)
  let final_value := value_after_year2 * (1 - overhaul_rate)
  
  final_value

/-- Theorem stating that the machine's value after two years and a major overhaul 
    is approximately $863.23, given an initial value of $1200 -/
theorem machine_value_theorem :
  ∃ ε > 0, abs (machine_value_after_two_years_and_overhaul 1200 - 863.23) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_machine_value_theorem_l234_23474


namespace NUMINAMATH_CALUDE_volume_per_balloon_l234_23414

/-- Given the number of balloons, volume of each gas tank, and number of tanks needed,
    prove that the volume of air per balloon is 10 liters. -/
theorem volume_per_balloon
  (num_balloons : ℕ)
  (tank_volume : ℕ)
  (num_tanks : ℕ)
  (h1 : num_balloons = 1000)
  (h2 : tank_volume = 500)
  (h3 : num_tanks = 20) :
  (num_tanks * tank_volume) / num_balloons = 10 :=
by sorry

end NUMINAMATH_CALUDE_volume_per_balloon_l234_23414


namespace NUMINAMATH_CALUDE_fifty_third_number_is_61_l234_23452

def adjustedSequence (n : ℕ) : ℕ :=
  n + (n - 1) / 4

theorem fifty_third_number_is_61 :
  adjustedSequence 53 = 61 := by
  sorry

end NUMINAMATH_CALUDE_fifty_third_number_is_61_l234_23452


namespace NUMINAMATH_CALUDE_eldest_age_is_32_l234_23482

/-- Represents the ages of three people A, B, and C -/
structure Ages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The present ages are in the ratio 5:7:8 -/
def present_ratio (ages : Ages) : Prop :=
  7 * ages.a = 5 * ages.b ∧ 8 * ages.a = 5 * ages.c

/-- The sum of ages 7 years ago was 59 -/
def past_sum (ages : Ages) : Prop :=
  (ages.a - 7) + (ages.b - 7) + (ages.c - 7) = 59

/-- Theorem stating that given the conditions, the eldest person's age is 32 -/
theorem eldest_age_is_32 (ages : Ages) 
  (h1 : present_ratio ages) 
  (h2 : past_sum ages) : 
  ages.c = 32 := by
  sorry


end NUMINAMATH_CALUDE_eldest_age_is_32_l234_23482


namespace NUMINAMATH_CALUDE_positive_real_solutions_range_l234_23431

theorem positive_real_solutions_range (a : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.pi ^ x = (a + 1) / (2 - a)) ↔ 1/2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_solutions_range_l234_23431


namespace NUMINAMATH_CALUDE_darnel_distance_difference_l234_23453

theorem darnel_distance_difference :
  let sprint_distance : ℚ := 875 / 1000
  let jog_distance : ℚ := 75 / 100
  sprint_distance - jog_distance = 125 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_darnel_distance_difference_l234_23453


namespace NUMINAMATH_CALUDE_local_election_vote_count_l234_23400

theorem local_election_vote_count (candidate1_percent : ℝ) (candidate2_percent : ℝ) 
  (candidate3_percent : ℝ) (candidate4_percent : ℝ) (candidate2_votes : ℕ) :
  candidate1_percent = 0.45 →
  candidate2_percent = 0.25 →
  candidate3_percent = 0.20 →
  candidate4_percent = 0.10 →
  candidate2_votes = 600 →
  candidate1_percent + candidate2_percent + candidate3_percent + candidate4_percent = 1 →
  ∃ (total_votes : ℕ), total_votes = 2400 ∧ 
    (candidate2_percent : ℝ) * total_votes = candidate2_votes := by
  sorry

end NUMINAMATH_CALUDE_local_election_vote_count_l234_23400


namespace NUMINAMATH_CALUDE_alpha_monogram_count_l234_23421

/-- The number of letters in the alphabet excluding 'A' -/
def n : ℕ := 25

/-- The number of initials to choose (first and middle) -/
def k : ℕ := 2

/-- The number of possible monograms for baby Alpha -/
def num_monograms : ℕ := n.choose k

theorem alpha_monogram_count : num_monograms = 300 := by
  sorry

end NUMINAMATH_CALUDE_alpha_monogram_count_l234_23421


namespace NUMINAMATH_CALUDE_original_fraction_is_two_thirds_l234_23456

theorem original_fraction_is_two_thirds 
  (x y : ℚ) 
  (h1 : x / (y + 1) = 1/2) 
  (h2 : (x + 1) / y = 1) : 
  x / y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_is_two_thirds_l234_23456


namespace NUMINAMATH_CALUDE_max_boxes_in_lot_l234_23428

theorem max_boxes_in_lot (lot_width lot_length box_width box_length : ℕ) 
  (hw : lot_width = 36)
  (hl : lot_length = 72)
  (bw : box_width = 3)
  (bl : box_length = 4) :
  (lot_width / box_width) * (lot_length / box_length) = 216 := by
  sorry

end NUMINAMATH_CALUDE_max_boxes_in_lot_l234_23428


namespace NUMINAMATH_CALUDE_negation_equivalence_l234_23450

/-- The original proposition p -/
def p : Prop := ∀ x : ℝ, x^2 + x - 6 ≤ 0

/-- The proposed negation of p -/
def q : Prop := ∃ x : ℝ, x^2 + x - 6 > 0

/-- Theorem stating that q is the negation of p -/
theorem negation_equivalence : ¬p ↔ q := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l234_23450


namespace NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l234_23464

/-- Represents a triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  n : ℕ
  side1 : ℕ := 2 * n
  side2 : ℕ := 2 * n + 2
  side3 : ℕ := 2 * n + 4

/-- Checks if the given EvenTriangle satisfies the triangle inequality -/
def is_valid (t : EvenTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Calculates the perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- Theorem: The smallest possible perimeter of a valid EvenTriangle is 18 -/
theorem smallest_even_triangle_perimeter :
  ∃ (t : EvenTriangle), is_valid t ∧ perimeter t = 18 ∧
  ∀ (t' : EvenTriangle), is_valid t' → perimeter t' ≥ 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l234_23464


namespace NUMINAMATH_CALUDE_fish_corn_equivalence_l234_23496

theorem fish_corn_equivalence :
  ∀ (fish honey corn : ℚ),
  (5 * fish = 3 * honey) →
  (honey = 6 * corn) →
  (fish = 3.6 * corn) :=
by sorry

end NUMINAMATH_CALUDE_fish_corn_equivalence_l234_23496


namespace NUMINAMATH_CALUDE_total_baking_time_l234_23489

def bread_time_1 : ℕ := 375
def bread_time_2 : ℕ := 160
def bread_time_3 : ℕ := 320

theorem total_baking_time :
  max (max bread_time_1 bread_time_2) bread_time_3 = 375 := by
  sorry

end NUMINAMATH_CALUDE_total_baking_time_l234_23489


namespace NUMINAMATH_CALUDE_quadrilateral_area_l234_23424

/-- Represents a quadrilateral ABCD with specific properties -/
structure Quadrilateral :=
  (AB : ℝ) (BC : ℝ) (AD : ℝ) (DC : ℝ)
  (AB_perp_BC : AB = BC)
  (AD_perp_DC : AD = DC)
  (AB_eq_9 : AB = 9)
  (AD_eq_8 : AD = 8)

/-- The area of the quadrilateral ABCD is 82.5 square units -/
theorem quadrilateral_area (q : Quadrilateral) : Real.sqrt ((q.AB ^ 2 + q.BC ^ 2) * (q.AD ^ 2 + q.DC ^ 2)) / 2 = 82.5 := by
  sorry

#check quadrilateral_area

end NUMINAMATH_CALUDE_quadrilateral_area_l234_23424


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l234_23444

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- arithmetic sequence
  (p q : ℕ)    -- indices
  (h1 : a p = 4)
  (h2 : a q = 2)
  (h3 : p = 4 + q)
  (h4 : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) :  -- arithmetic sequence property
  ∃ d : ℝ, d = 1/2 ∧ ∀ n : ℕ, a (n + 1) - a n = d := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l234_23444


namespace NUMINAMATH_CALUDE_diamond_three_four_l234_23433

def diamond (a b : ℝ) : ℝ := 4 * a + 3 * b - 2 * a * b

theorem diamond_three_four : diamond 3 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_four_l234_23433


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_1419_l234_23448

def product_of_evens (n : Nat) : Nat :=
  (List.range ((n / 2) - 1)).foldl (fun acc i => acc * (2 * (i + 2))) 2

theorem smallest_n_divisible_by_1419 :
  (∀ m : Nat, m < 106 → m % 2 = 0 → ¬(product_of_evens m % 1419 = 0)) ∧
  (106 % 2 = 0 ∧ product_of_evens 106 % 1419 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_1419_l234_23448


namespace NUMINAMATH_CALUDE_sandwiches_problem_l234_23454

theorem sandwiches_problem (S : ℚ) :
  (S > 0) →
  (3/4 * S - 1/8 * S - 1/4 * S - 5 = 4) →
  S = 24 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_problem_l234_23454


namespace NUMINAMATH_CALUDE_remi_tomato_seedlings_l234_23430

theorem remi_tomato_seedlings (day1 : ℕ) (total : ℕ) : 
  day1 = 200 →
  total = 5000 →
  (day1 + 2 * day1 + 3 * (2 * day1) + 4 * (2 * day1) = total) →
  3 * (2 * day1) + 4 * (2 * day1) = 4400 :=
by
  sorry

end NUMINAMATH_CALUDE_remi_tomato_seedlings_l234_23430


namespace NUMINAMATH_CALUDE_simple_interest_rate_l234_23402

/-- Given a principal amount and a simple interest rate, if the sum of money
becomes 7/6 of itself in 2 years, then the rate is 1/12 -/
theorem simple_interest_rate (P : ℝ) (R : ℝ) (P_pos : P > 0) :
  P * (1 + 2 * R) = (7 / 6) * P → R = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l234_23402


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l234_23429

/-- The slope of the asymptotes of a hyperbola with specific properties -/
theorem hyperbola_asymptote_slope (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  let A₁ : ℝ × ℝ := (-a, 0)
  let A₂ : ℝ × ℝ := (a, 0)
  let B : ℝ × ℝ := (c, b^2/a)
  let C : ℝ × ℝ := (c, -b^2/a)
  (∀ x y, x^2/a^2 - y^2/b^2 = 1 → (x = c ∧ (y = b^2/a ∨ y = -b^2/a))) →
  ((B.2 - A₁.2) * (C.2 - A₂.2) = -(B.1 - A₁.1) * (C.1 - A₂.1)) →
  (∀ x, (x : ℝ) = x ∨ (x : ℝ) = -x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l234_23429


namespace NUMINAMATH_CALUDE_trig_equality_l234_23434

theorem trig_equality (θ : ℝ) (h : Real.sin (θ + π/3) = 2/3) : 
  Real.cos (θ - π/6) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_equality_l234_23434


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l234_23417

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 3) % 9 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 25 = 0 ∧ (n + 3) % 21 = 0

theorem smallest_number_divisible_by_all : 
  is_divisible_by_all 3147 ∧ ∀ m < 3147, ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l234_23417


namespace NUMINAMATH_CALUDE_no_integer_root_2016_l234_23472

/-- A cubic polynomial with integer coefficients -/
def cubic_poly (a b c d : ℤ) : ℤ → ℤ := fun x ↦ a * x^3 + b * x^2 + c * x + d

theorem no_integer_root_2016 (a b c d : ℤ) :
  cubic_poly a b c d 1 = 2015 →
  cubic_poly a b c d 2 = 2017 →
  ∀ k : ℤ, cubic_poly a b c d k ≠ 2016 := by
sorry

end NUMINAMATH_CALUDE_no_integer_root_2016_l234_23472


namespace NUMINAMATH_CALUDE_car_travel_time_l234_23492

/-- Given two cars A and B with specific speeds and distance ratios, 
    prove that Car A takes 6 hours to reach its destination. -/
theorem car_travel_time :
  ∀ (speed_A speed_B time_B distance_A distance_B : ℝ),
  speed_A = 50 →
  speed_B = 100 →
  time_B = 1 →
  distance_A / distance_B = 3 →
  distance_B = speed_B * time_B →
  distance_A = speed_A * (distance_A / speed_A) →
  distance_A / speed_A = 6 := by
sorry

end NUMINAMATH_CALUDE_car_travel_time_l234_23492


namespace NUMINAMATH_CALUDE_factorial_sum_quotient_l234_23415

theorem factorial_sum_quotient (n : ℕ) (h : n ≥ 2) :
  (Nat.factorial (n + 2) + Nat.factorial (n + 1)) / Nat.factorial (n + 1) = n + 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_quotient_l234_23415


namespace NUMINAMATH_CALUDE_inequality_solution_set_l234_23422

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 1) * (x + 2) < 0

-- Define the solution set
def solution_set : Set ℝ := { x | -2 < x ∧ x < 1 }

-- Theorem statement
theorem inequality_solution_set : 
  { x : ℝ | inequality x } = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l234_23422


namespace NUMINAMATH_CALUDE_min_value_theorem_l234_23473

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 5*x*y) :
  3*x + 2*y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 3*y₀ = 5*x₀*y₀ ∧ 3*x₀ + 2*y₀ = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l234_23473


namespace NUMINAMATH_CALUDE_min_value_zero_implies_t_l234_23459

/-- The function f(x) defined in the problem -/
def f (t : ℝ) (x : ℝ) : ℝ := 4 * x^4 - 6 * t * x^3 + (2 * t + 6) * x^2 - 3 * t * x + 1

/-- The theorem statement -/
theorem min_value_zero_implies_t (t : ℝ) :
  (∀ x > 0, f t x ≥ 0) ∧ 
  (∃ x > 0, f t x = 0) →
  t = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_zero_implies_t_l234_23459


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_specific_case_l234_23416

theorem simplify_and_evaluate (a b : ℝ) :
  (a - b)^2 + (a + 3*b)*(a - 3*b) - a*(a - 2*b) = a^2 - 8*b^2 :=
by sorry

theorem specific_case : 
  let a : ℝ := -1
  let b : ℝ := 2
  (a - b)^2 + (a + 3*b)*(a - 3*b) - a*(a - 2*b) = -31 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_specific_case_l234_23416


namespace NUMINAMATH_CALUDE_distance_to_black_planet_l234_23470

/-- The distance to a black planet given spaceship and light travel times -/
theorem distance_to_black_planet 
  (v_ship : ℝ) -- speed of spaceship
  (v_light : ℝ) -- speed of light
  (t_total : ℝ) -- total time of travel and light reflection
  (h_v_ship : v_ship = 100000) -- spaceship speed in km/s
  (h_v_light : v_light = 300000) -- light speed in km/s
  (h_t_total : t_total = 100) -- total time in seconds
  : ∃ d : ℝ, d = 1500 * 10000 ∧ t_total = (d + v_ship * t_total) / v_light + d / v_light :=
by sorry

end NUMINAMATH_CALUDE_distance_to_black_planet_l234_23470
