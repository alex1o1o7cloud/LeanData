import Mathlib

namespace divisibility_equivalence_l2106_210674

theorem divisibility_equivalence (n : ℕ+) :
  (n.val^5 + 5^n.val) % 11 = 0 ↔ (n.val^5 * 5^n.val + 1) % 11 = 0 := by
  sorry

end divisibility_equivalence_l2106_210674


namespace circle_equation_implies_a_eq_neg_one_l2106_210677

/-- A circle equation in the form x^2 + by^2 + cx + d = 0 --/
structure CircleEquation where
  b : ℝ
  c : ℝ
  d : ℝ

/-- Condition for an equation to represent a circle --/
def is_circle (eq : CircleEquation) : Prop :=
  eq.b = 1 ∧ eq.b ≠ 0

/-- The given equation x^2 + (a+2)y^2 + 2ax + a = 0 --/
def given_equation (a : ℝ) : CircleEquation :=
  { b := a + 2
  , c := 2 * a
  , d := a }

theorem circle_equation_implies_a_eq_neg_one :
  ∀ a : ℝ, is_circle (given_equation a) → a = -1 := by
  sorry

end circle_equation_implies_a_eq_neg_one_l2106_210677


namespace x_equals_y_l2106_210603

theorem x_equals_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) : x = y :=
by sorry

end x_equals_y_l2106_210603


namespace exponent_equality_l2106_210622

theorem exponent_equality : 8^5 * 3^5 * 8^3 * 3^7 = 8^8 * 3^12 := by
  sorry

end exponent_equality_l2106_210622


namespace buses_needed_l2106_210661

def fifth_graders : ℕ := 109
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def seats_per_bus : ℕ := 72

def total_students : ℕ := fifth_graders + sixth_graders + seventh_graders
def total_chaperones : ℕ := (teachers_per_grade + parents_per_grade) * 3
def total_people : ℕ := total_students + total_chaperones

theorem buses_needed : 
  ∃ n : ℕ, n * seats_per_bus ≥ total_people ∧ 
  ∀ m : ℕ, m * seats_per_bus ≥ total_people → n ≤ m := by
  sorry

end buses_needed_l2106_210661


namespace x_plus_y_value_l2106_210681

theorem x_plus_y_value (x y : ℝ) (h1 : x + Real.cos y = 3005) 
  (h2 : x + 3005 * Real.sin y = 3004) (h3 : 0 ≤ y) (h4 : y ≤ π) : 
  x + y = 3004 := by
  sorry

end x_plus_y_value_l2106_210681


namespace total_pencils_donna_marcia_l2106_210615

/-- The number of pencils Cindi bought -/
def cindi_pencils : ℕ := 60

/-- The number of pencils Marcia bought -/
def marcia_pencils : ℕ := 2 * cindi_pencils

/-- The number of pencils Donna bought -/
def donna_pencils : ℕ := 3 * marcia_pencils

/-- Theorem: The total number of pencils bought by Donna and Marcia is 480 -/
theorem total_pencils_donna_marcia : donna_pencils + marcia_pencils = 480 := by
  sorry

end total_pencils_donna_marcia_l2106_210615


namespace divisible_by_three_l2106_210679

theorem divisible_by_three (x y : ℤ) (h : 3 ∣ (x^2 + y^2)) : 3 ∣ x ∧ 3 ∣ y := by
  sorry

end divisible_by_three_l2106_210679


namespace consequences_of_only_some_A_are_B_l2106_210667

-- Define sets A and B
variable (A B : Set α)

-- Define the premise "Only some A are B"
def only_some_A_are_B : Prop := ∃ x ∈ A, x ∈ B ∧ ∃ y ∈ A, y ∉ B

-- Theorem stating the consequences
theorem consequences_of_only_some_A_are_B (h : only_some_A_are_B A B) :
  (¬ ∀ x ∈ A, x ∈ B) ∧
  (∃ x ∈ A, x ∉ B) ∧
  (∃ x ∈ B, x ∈ A) ∧
  (∃ x ∈ A, x ∈ B) ∧
  (∃ x ∈ A, x ∉ B) :=
by sorry

end consequences_of_only_some_A_are_B_l2106_210667


namespace isosceles_triangle_perimeter_l2106_210609

/-- An isosceles triangle with side lengths 5 and 6 has a perimeter of either 16 or 17 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 5 ∧ b = 6 ∧
  ((a = b ∧ c ≤ a + b) ∨ (a = c ∧ b ≤ a + c) ∨ (b = c ∧ a ≤ b + c)) →
  a + b + c = 16 ∨ a + b + c = 17 := by
sorry

end isosceles_triangle_perimeter_l2106_210609


namespace cube_volume_from_surface_area_l2106_210666

/-- Given a cube with surface area 6x^2, its volume is x^3 -/
theorem cube_volume_from_surface_area (x : ℝ) :
  let surface_area := 6 * x^2
  let side_length := Real.sqrt (surface_area / 6)
  let volume := side_length^3
  volume = x^3 := by
  sorry

end cube_volume_from_surface_area_l2106_210666


namespace comm_add_comm_mul_distrib_l2106_210607

-- Commutative law of addition
theorem comm_add (a b : ℝ) : a + b = b + a := by sorry

-- Commutative law of multiplication
theorem comm_mul (a b : ℝ) : a * b = b * a := by sorry

-- Distributive law of multiplication over addition
theorem distrib (a b c : ℝ) : (a + b) * c = a * c + b * c := by sorry

end comm_add_comm_mul_distrib_l2106_210607


namespace triangle_projection_shapes_l2106_210659

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- Possible projection shapes -/
inductive ProjectionShape
  | Angle
  | Strip
  | TwoAngles
  | Triangle
  | CompositeShape

/-- Function to project a triangle onto a plane from a point -/
def project (t : Triangle3D) (p : Plane3D) (o : Point3D) : ProjectionShape :=
  sorry

/-- Theorem stating the possible projection shapes -/
theorem triangle_projection_shapes (t : Triangle3D) (p : Plane3D) (o : Point3D) 
  (h : o ∉ {x : Point3D | t.a.z = t.b.z ∧ t.b.z = t.c.z}) :
  ∃ (shape : ProjectionShape), project t p o = shape :=
sorry

end triangle_projection_shapes_l2106_210659


namespace machine_input_l2106_210646

theorem machine_input (x : ℝ) : 
  1.2 * ((3 * (x + 15) - 6) / 2)^2 = 35 → x = -9.4 := by
  sorry

end machine_input_l2106_210646


namespace largest_eight_digit_with_all_even_digits_l2106_210694

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop :=
  100000000 > n ∧ n ≥ 10000000

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k, n / (10^k) % 10 = d

def largest_eight_digit_with_even_digits : Nat := 99986420

theorem largest_eight_digit_with_all_even_digits :
  is_eight_digit largest_eight_digit_with_even_digits ∧
  contains_all_even_digits largest_eight_digit_with_even_digits ∧
  ∀ n, is_eight_digit n → contains_all_even_digits n →
    n ≤ largest_eight_digit_with_even_digits :=
by sorry

end largest_eight_digit_with_all_even_digits_l2106_210694


namespace negation_of_proposition_negation_of_specific_proposition_l2106_210645

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬P x) :=
by sorry

theorem negation_of_specific_proposition :
  (¬∀ x : ℝ, x^2 + 2*x + 3 ≥ 0) ↔ (∃ x : ℝ, x^2 + 2*x + 3 < 0) :=
by sorry

end negation_of_proposition_negation_of_specific_proposition_l2106_210645


namespace outer_digits_swap_l2106_210671

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hundreds_range : hundreds ≥ 1 ∧ hundreds ≤ 9
  tens_range : tens ≥ 0 ∧ tens ≤ 9
  units_range : units ≥ 0 ∧ units ≤ 9

/-- Convert a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNum (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

theorem outer_digits_swap (n : ThreeDigitNumber) 
  (h1 : n.toNum + 45 = 100 * n.hundreds + 10 * n.units + n.tens)
  (h2 : n.toNum = 100 * n.tens + 10 * n.hundreds + n.units + 270) :
  100 * n.units + 10 * n.tens + n.hundreds = n.toNum + 198 := by
  sorry

#check outer_digits_swap

end outer_digits_swap_l2106_210671


namespace parabola_y_relationship_l2106_210669

/-- Given that points (-4, y₁), (-1, y₂), and (5/3, y₃) lie on the graph of y = -x² - 4x + 5,
    prove that y₂ > y₁ > y₃ -/
theorem parabola_y_relationship (y₁ y₂ y₃ : ℝ) : 
  y₁ = -(-4)^2 - 4*(-4) + 5 →
  y₂ = -(-1)^2 - 4*(-1) + 5 →
  y₃ = -(5/3)^2 - 4*(5/3) + 5 →
  y₂ > y₁ ∧ y₁ > y₃ := by
  sorry


end parabola_y_relationship_l2106_210669


namespace geometric_sequence_formula_l2106_210618

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = q * a n ∧ a n > 0

theorem geometric_sequence_formula 
  (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_first : a 1 = 1) 
  (h_second : ∃ x : ℝ, a 2 = x + 1) 
  (h_third : ∃ x : ℝ, a 3 = 2 * x + 5) : 
  ∀ n : ℕ, a n = 3^(n - 1) := by
sorry

end geometric_sequence_formula_l2106_210618


namespace range_of_a_l2106_210652

-- Define the condition
def always_positive (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - a*x + 2*a > 0

-- State the theorem
theorem range_of_a (a : ℝ) :
  always_positive a ↔ (0 < a ∧ a < 8) :=
by sorry

end range_of_a_l2106_210652


namespace exists_unreachable_positive_configuration_l2106_210692

/-- Represents a cell in the grid -/
inductive Cell
| Plus
| Minus

/-- Represents an 8x8 grid -/
def Grid := Fin 8 → Fin 8 → Cell

/-- Represents the allowed operations -/
inductive Operation
| Flip3x3 (row col : Fin 6)  -- Top-left corner of 3x3 square
| Flip4x4 (row col : Fin 5)  -- Top-left corner of 4x4 square

/-- Applies an operation to a grid -/
def applyOperation (g : Grid) (op : Operation) : Grid :=
  sorry

/-- Checks if a grid is all positive -/
def isAllPositive (g : Grid) : Prop :=
  ∀ i j, g i j = Cell.Plus

/-- Theorem: There exists an initial grid configuration that cannot be transformed to all positive -/
theorem exists_unreachable_positive_configuration :
  ∃ (initial : Grid), ¬∃ (ops : List Operation), isAllPositive (ops.foldl applyOperation initial) :=
sorry

end exists_unreachable_positive_configuration_l2106_210692


namespace china_gdp_surpass_us_correct_regression_equation_china_gdp_surpass_us_in_2028_l2106_210678

-- Define the data types
structure GDPData where
  year_code : ℕ
  gdp : ℝ

-- Define the given data
def china_gdp_data : List GDPData := [
  ⟨1, 8.5⟩, ⟨2, 9.6⟩, ⟨3, 10.4⟩, ⟨4, 11⟩, ⟨5, 11.1⟩, ⟨6, 12.1⟩, ⟨7, 13.6⟩
]

-- Define the sums given in the problem
def sum_y : ℝ := 76.3
def sum_xy : ℝ := 326.2

-- Define the US GDP in 2018
def us_gdp_2018 : ℝ := 20.5

-- Define the linear regression function
def linear_regression (x : ℝ) : ℝ := 0.75 * x + 7.9

-- Theorem statement
theorem china_gdp_surpass_us (n : ℕ) :
  (linear_regression (n + 7 : ℝ) ≥ us_gdp_2018) ↔ (n + 2021 ≥ 2028) := by
  sorry

-- Prove that the linear regression equation is correct
theorem correct_regression_equation :
  ∀ x, linear_regression x = 0.75 * x + 7.9 := by
  sorry

-- Prove that China's GDP will surpass US 2018 GDP in 2028
theorem china_gdp_surpass_us_in_2028 :
  ∃ n : ℕ, n + 2021 = 2028 ∧ linear_regression (n + 7 : ℝ) ≥ us_gdp_2018 := by
  sorry

end china_gdp_surpass_us_correct_regression_equation_china_gdp_surpass_us_in_2028_l2106_210678


namespace necessary_but_not_sufficient_l2106_210612

theorem necessary_but_not_sufficient :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ∧
  (∃ x y : ℝ, x + y > 2 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end necessary_but_not_sufficient_l2106_210612


namespace quadratic_solution_existence_l2106_210672

/-- A quadratic function f(x) = ax^2 + bx + c, where a ≠ 0 and a, b, c are constants. -/
def QuadraticFunction (a b c : ℝ) (h : a ≠ 0) := fun (x : ℝ) ↦ a * x^2 + b * x + c

theorem quadratic_solution_existence (a b c : ℝ) (h : a ≠ 0) :
  let f := QuadraticFunction a b c h
  (f 6.17 = -0.03) →
  (f 6.18 = -0.01) →
  (f 6.19 = 0.02) →
  (f 6.20 = 0.04) →
  ∃ x : ℝ, (f x = 0) ∧ (6.18 < x) ∧ (x < 6.19) := by
  sorry

end quadratic_solution_existence_l2106_210672


namespace find_a_and_b_l2106_210630

/-- Set A defined by the equation ax - y² + b = 0 -/
def A (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 - p.2^2 + b = 0}

/-- Set B defined by the equation x² - ay - b = 0 -/
def B (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - a * p.2 - b = 0}

/-- Theorem stating that a = -3 and b = 7 given the conditions -/
theorem find_a_and_b :
  ∃ (a b : ℝ), (1, 2) ∈ A a b ∩ B a b ∧ a = -3 ∧ b = 7 := by
  sorry

end find_a_and_b_l2106_210630


namespace pancake_price_l2106_210648

/-- Janina's pancake stand problem -/
theorem pancake_price (daily_rent : ℝ) (daily_supplies : ℝ) (pancakes_to_cover_expenses : ℕ) :
  daily_rent = 30 ∧ daily_supplies = 12 ∧ pancakes_to_cover_expenses = 21 →
  (daily_rent + daily_supplies) / pancakes_to_cover_expenses = 2 :=
by sorry

end pancake_price_l2106_210648


namespace new_york_to_new_england_ratio_l2106_210657

/-- The population of New England -/
def new_england_population : ℕ := 2100000

/-- The combined population of New York and New England -/
def combined_population : ℕ := 3500000

/-- The population of New York -/
def new_york_population : ℕ := combined_population - new_england_population

/-- The ratio of New York's population to New England's population -/
theorem new_york_to_new_england_ratio :
  (new_york_population : ℚ) / (new_england_population : ℚ) = 2 / 3 := by
  sorry

end new_york_to_new_england_ratio_l2106_210657


namespace problem_1_problem_2_problem_3_problem_4_l2106_210627

-- Problem 1
theorem problem_1 : (Real.sqrt 50 * Real.sqrt 32) / Real.sqrt 8 - 4 * Real.sqrt 2 = 6 * Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 : Real.sqrt 12 - 6 * Real.sqrt (1/3) + Real.sqrt 48 = 4 * Real.sqrt 3 := by sorry

-- Problem 3
theorem problem_3 : (Real.sqrt 5 + 3) * (3 - Real.sqrt 5) - (Real.sqrt 3 - 1)^2 = 2 * Real.sqrt 3 := by sorry

-- Problem 4
theorem problem_4 : (Real.sqrt 24 + Real.sqrt 50) / Real.sqrt 2 - 6 * Real.sqrt (1/3) = 5 := by sorry

end problem_1_problem_2_problem_3_problem_4_l2106_210627


namespace final_amount_calculation_l2106_210606

/-- Calculates the final amount paid after applying a discount based on complete hundreds spent. -/
theorem final_amount_calculation (purchase_amount : ℕ) (discount_per_hundred : ℕ) : 
  purchase_amount = 250 ∧ discount_per_hundred = 10 →
  purchase_amount - (purchase_amount / 100) * discount_per_hundred = 230 := by
  sorry

end final_amount_calculation_l2106_210606


namespace ned_video_game_earnings_l2106_210676

/-- Given the total number of games, non-working games, and price per working game,
    calculates the total earnings from selling the working games. -/
def calculate_earnings (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : ℕ :=
  (total_games - non_working_games) * price_per_game

/-- Proves that Ned's earnings from selling his working video games is $63. -/
theorem ned_video_game_earnings :
  calculate_earnings 15 6 7 = 63 := by
  sorry

#eval calculate_earnings 15 6 7

end ned_video_game_earnings_l2106_210676


namespace problem_solution_l2106_210640

def f (a x : ℝ) : ℝ := |x - 1| + |x + a^2|

theorem problem_solution :
  (∀ x : ℝ, f (Real.sqrt 2) x ≥ 6 ↔ x ≤ -7/2 ∨ x ≥ 5/2) ∧
  (∃ x₀ : ℝ, f a x₀ < 4*a ↔ 2 - Real.sqrt 3 < a ∧ a < 2 + Real.sqrt 3) :=
by sorry

end problem_solution_l2106_210640


namespace expression_evaluation_l2106_210632

/-- Proves that the given expression evaluates to 11 when x = -2 and y = -1 -/
theorem expression_evaluation :
  let x : ℝ := -2
  let y : ℝ := -1
  3 * (2 * x^2 + x*y + 1/3) - (3 * x^2 + 4*x*y - y^2) = 11 := by
  sorry

end expression_evaluation_l2106_210632


namespace cosine_inequality_solution_l2106_210638

theorem cosine_inequality_solution (y : Real) : 
  (y ∈ Set.Icc 0 Real.pi) ∧ 
  (∀ x ∈ Set.Icc 0 Real.pi, Real.cos (x + y) ≥ Real.cos x + Real.cos y) → 
  y = 0 := by
sorry

end cosine_inequality_solution_l2106_210638


namespace popcorn_per_serving_l2106_210697

/-- The number of pieces of popcorn Jared can eat -/
def jared_popcorn : ℕ := 90

/-- The number of pieces of popcorn each of Jared's friends can eat -/
def friend_popcorn : ℕ := 60

/-- The number of Jared's friends -/
def num_friends : ℕ := 3

/-- The number of servings Jared should order -/
def num_servings : ℕ := 9

/-- Theorem stating that the number of pieces of popcorn in a serving is 30 -/
theorem popcorn_per_serving : 
  (jared_popcorn + num_friends * friend_popcorn) / num_servings = 30 := by
  sorry

end popcorn_per_serving_l2106_210697


namespace chris_money_before_birthday_l2106_210644

/-- Represents the amount of money Chris had before his birthday -/
def money_before_birthday : ℕ := sorry

/-- Represents the amount of money Chris received from his grandmother -/
def grandmother_gift : ℕ := 25

/-- Represents the amount of money Chris received from his aunt and uncle -/
def aunt_uncle_gift : ℕ := 20

/-- Represents the amount of money Chris received from his parents -/
def parents_gift : ℕ := 75

/-- Represents the total amount of money Chris has now -/
def total_money_now : ℕ := 279

/-- Theorem stating that Chris had $159 before his birthday -/
theorem chris_money_before_birthday :
  money_before_birthday = 159 :=
by sorry

end chris_money_before_birthday_l2106_210644


namespace cube_opposite_color_l2106_210650

/-- Represents the colors of the squares --/
inductive Color
  | P | C | M | S | L | K

/-- Represents the faces of a cube --/
inductive Face
  | Top | Bottom | Front | Back | Left | Right

/-- Represents a cube formed by six hinged squares --/
structure Cube where
  faces : Face → Color

/-- Defines the opposite face relationship --/
def opposite_face : Face → Face
  | Face.Top    => Face.Bottom
  | Face.Bottom => Face.Top
  | Face.Front  => Face.Back
  | Face.Back   => Face.Front
  | Face.Left   => Face.Right
  | Face.Right  => Face.Left

theorem cube_opposite_color (c : Cube) :
  c.faces Face.Top = Color.M →
  c.faces Face.Front = Color.L →
  c.faces (opposite_face Face.Front) = Color.K :=
by sorry

end cube_opposite_color_l2106_210650


namespace total_handshakes_l2106_210662

/-- The total number of handshakes in a group of boys with specific conditions -/
theorem total_handshakes (n : ℕ) (l : ℕ) (f : ℕ) (h : ℕ) : 
  n = 15 → l = 5 → f = 3 → h = 2 → 
  (n * (n - 1)) / 2 - (l * (n - l)) - f * h = 49 := by
  sorry

end total_handshakes_l2106_210662


namespace problem_solution_l2106_210696

def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {1, 2-x}

theorem problem_solution (x : ℝ) (C : Set ℝ) 
  (h1 : B x ⊆ A x) 
  (h2 : B x ∪ C = A x) : 
  x = -2 ∧ C = {3} := by
  sorry

#check problem_solution

end problem_solution_l2106_210696


namespace find_k_l2106_210647

theorem find_k : ∃ k : ℕ, (1/2)^16 * (1/81)^k = 1/(18^16) → k = 8 := by
  sorry

end find_k_l2106_210647


namespace wen_family_theater_cost_l2106_210605

/-- Represents the cost of tickets for a family theater outing -/
def theater_cost (regular_price : ℚ) : ℚ :=
  let senior_price := regular_price * (1 - 0.2)
  let child_price := regular_price * (1 - 0.4)
  let total_before_discount := 2 * senior_price + 2 * regular_price + 2 * child_price
  total_before_discount * (1 - 0.1)

/-- Theorem stating the total cost for the Wen family's theater tickets -/
theorem wen_family_theater_cost :
  ∃ (regular_price : ℚ),
    (regular_price * (1 - 0.2) = 7.5) ∧
    (theater_cost regular_price = 40.5) := by
  sorry


end wen_family_theater_cost_l2106_210605


namespace thompson_purchase_cost_l2106_210693

/-- The total cost of chickens and potatoes -/
def total_cost (num_chickens : ℕ) (chicken_price : ℝ) (potato_price : ℝ) : ℝ :=
  (num_chickens : ℝ) * chicken_price + potato_price

/-- Theorem: The total cost of 3 chickens at $3 each and a bag of potatoes at $6 is $15 -/
theorem thompson_purchase_cost : total_cost 3 3 6 = 15 := by
  sorry

end thompson_purchase_cost_l2106_210693


namespace set_inclusion_l2106_210610

-- Define the sets M, N, and P
def M : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 < 1}

def N : Set (ℝ × ℝ) := {p | Real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + 
                             Real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * Real.sqrt 2}

def P : Set (ℝ × ℝ) := {p | abs (p.1 + p.2) < 1 ∧ abs p.1 < 1 ∧ abs p.2 < 1}

-- State the theorem
theorem set_inclusion : M ⊆ P ∧ P ⊆ N := by sorry

end set_inclusion_l2106_210610


namespace seashells_per_day_l2106_210613

/-- 
Given a 5-day beach trip where 35 seashells were found in total, 
and assuming an equal number of seashells were found each day, 
prove that the number of seashells found per day is 7.
-/
theorem seashells_per_day 
  (days : ℕ) 
  (total_seashells : ℕ) 
  (seashells_per_day : ℕ) 
  (h1 : days = 5) 
  (h2 : total_seashells = 35) 
  (h3 : seashells_per_day * days = total_seashells) : 
  seashells_per_day = 7 := by
sorry

end seashells_per_day_l2106_210613


namespace hyperbola_real_axis_length_l2106_210616

/-- The hyperbola C: x² - y² = a² intersects with the directrix of the parabola y² = 16x 
    at two points with distance 4√3 between them. 
    This theorem states that the length of the real axis of hyperbola C is 4. -/
theorem hyperbola_real_axis_length (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.1 = -4 ∧ A.1^2 - A.2^2 = a^2) ∧ 
    (B.1 = -4 ∧ B.1^2 - B.2^2 = a^2) ∧ 
    (A.2 - B.2)^2 = 48) →
  2 * a = 4 := by sorry

end hyperbola_real_axis_length_l2106_210616


namespace clothing_sale_price_l2106_210621

theorem clothing_sale_price (a : ℝ) : 
  (∃ x y : ℝ, 
    x * 1.25 = a ∧ 
    y * 0.75 = a ∧ 
    x + y - 2*a = -8) → 
  a = 60 := by
sorry

end clothing_sale_price_l2106_210621


namespace better_to_answer_B_first_l2106_210611

-- Define the probabilities and point values
def prob_correct_A : Real := 0.8
def prob_correct_B : Real := 0.6
def points_A : ℕ := 20
def points_B : ℕ := 80

-- Define the expected score functions
def expected_score_A_first : Real :=
  0 * (1 - prob_correct_A) +
  points_A * (prob_correct_A * (1 - prob_correct_B)) +
  (points_A + points_B) * (prob_correct_A * prob_correct_B)

def expected_score_B_first : Real :=
  0 * (1 - prob_correct_B) +
  points_B * (prob_correct_B * (1 - prob_correct_A)) +
  (points_A + points_B) * (prob_correct_B * prob_correct_A)

-- Theorem statement
theorem better_to_answer_B_first :
  expected_score_B_first > expected_score_A_first := by
  sorry


end better_to_answer_B_first_l2106_210611


namespace sum_of_number_and_its_square_l2106_210629

theorem sum_of_number_and_its_square (x : ℝ) : x = 4 → x + x^2 = 20 := by
  sorry

end sum_of_number_and_its_square_l2106_210629


namespace prime_power_cube_plus_one_l2106_210651

theorem prime_power_cube_plus_one (p : ℕ) (x y : ℕ+) (h_prime : Nat.Prime p) :
  p ^ (x : ℕ) = y ^ 3 + 1 ↔ (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
sorry

end prime_power_cube_plus_one_l2106_210651


namespace sum_30_45_base3_l2106_210634

/-- Converts a natural number to its base 3 representation as a list of digits -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec go (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc
    else go (m / 3) ((m % 3) :: acc)
  go n []

/-- Theorem: The sum of 30 and 45 in base 10 is equal to 22010 in base 3 -/
theorem sum_30_45_base3 : toBase3 (30 + 45) = [2, 2, 0, 1, 0] := by
  sorry

end sum_30_45_base3_l2106_210634


namespace two_painter_time_l2106_210668

/-- The time taken for two painters to complete a wall together, given their individual rates -/
theorem two_painter_time (harish_rate ganpat_rate : ℝ) (harish_time ganpat_time : ℝ) :
  harish_rate = 1 / harish_time →
  ganpat_rate = 1 / ganpat_time →
  harish_time = 3 →
  ganpat_time = 6 →
  1 / (harish_rate + ganpat_rate) = 2 := by
  sorry

#check two_painter_time

end two_painter_time_l2106_210668


namespace sum_of_even_indexed_coefficients_l2106_210601

theorem sum_of_even_indexed_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^10 = a + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3 + a₄*(1-x)^4 + 
            a₅*(1-x)^5 + a₆*(1-x)^6 + a₇*(1-x)^7 + a₈*(1-x)^8 + a₉*(1-x)^9 + a₁₀*(1-x)^10) →
  a + a₂ + a₄ + a₆ + a₈ + a₁₀ = 2^9 := by
sorry

end sum_of_even_indexed_coefficients_l2106_210601


namespace square_sum_geq_product_l2106_210691

theorem square_sum_geq_product (x y z : ℝ) (h : x + y + z ≥ x * y * z) : x^2 + y^2 + z^2 ≥ x * y * z := by
  sorry

end square_sum_geq_product_l2106_210691


namespace retail_price_calculation_l2106_210675

/-- The retail price of a machine given wholesale price, discount, and profit margin -/
theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  let selling_price := (1 - discount_rate) * retail_price
  let profit := profit_rate * wholesale_price
  wholesale_price = 81 ∧ discount_rate = 0.1 ∧ profit_rate = 0.2 →
  ∃ retail_price : ℝ, selling_price = wholesale_price + profit ∧ retail_price = 108 :=
by
  sorry

end retail_price_calculation_l2106_210675


namespace boat_speed_in_still_water_l2106_210628

/-- Given a boat traveling downstream with a current of 3 km/hr,
    prove that its speed in still water is 15 km/hr if it travels 3.6 km in 12 minutes. -/
theorem boat_speed_in_still_water : ∀ (b : ℝ),
  (b + 3) * (1 / 5) = 3.6 →
  b = 15 := by sorry

end boat_speed_in_still_water_l2106_210628


namespace uniform_transform_l2106_210655

/-- A uniform random number between 0 and 1 -/
def uniform_random_01 : Set ℝ := Set.Icc 0 1

/-- The transformation function -/
def transform (x : ℝ) : ℝ := x * 5 - 2

/-- The set of numbers between -2 and 3 -/
def target_set : Set ℝ := Set.Icc (-2) 3

theorem uniform_transform :
  ∀ (a₁ : ℝ), a₁ ∈ uniform_random_01 → transform a₁ ∈ target_set :=
sorry

end uniform_transform_l2106_210655


namespace binomial_1409_1_l2106_210620

theorem binomial_1409_1 : (1409 : ℕ).choose 1 = 1409 := by sorry

end binomial_1409_1_l2106_210620


namespace books_sold_l2106_210682

/-- Given Kaleb's initial and final book counts, along with the number of new books bought,
    prove the number of books he sold. -/
theorem books_sold (initial : ℕ) (new_bought : ℕ) (final : ℕ) 
    (h1 : initial = 34) 
    (h2 : new_bought = 7) 
    (h3 : final = 24) : 
  initial - final + new_bought = 17 := by
  sorry

end books_sold_l2106_210682


namespace visit_either_not_both_l2106_210670

/-- The probability of visiting either Chile or Madagascar, but not both -/
theorem visit_either_not_both (p_chile p_madagascar : ℝ) 
  (h_chile : p_chile = 0.30)
  (h_madagascar : p_madagascar = 0.50) :
  p_chile + p_madagascar - p_chile * p_madagascar = 0.65 := by
  sorry

end visit_either_not_both_l2106_210670


namespace square_rectangle_perimeter_sum_l2106_210600

theorem square_rectangle_perimeter_sum :
  ∀ (s l w : ℝ),
  s > 0 ∧ l > 0 ∧ w > 0 →
  s^2 + l * w = 130 →
  s^2 - l * w = 50 →
  l = 2 * w →
  4 * s + 2 * (l + w) = 12 * Real.sqrt 10 + 12 * Real.sqrt 5 :=
by sorry

end square_rectangle_perimeter_sum_l2106_210600


namespace consecutive_integers_fourth_power_sum_l2106_210656

theorem consecutive_integers_fourth_power_sum (x : ℤ) : 
  x * (x + 1) * (x + 2) = 12 * (3 * x + 3) - 24 →
  x^4 + (x + 1)^4 + (x + 2)^4 = 98 := by
sorry

end consecutive_integers_fourth_power_sum_l2106_210656


namespace cubic_factorization_l2106_210614

theorem cubic_factorization (x : ℝ) : 4 * x^3 - 4 * x^2 + x = x * (2 * x - 1)^2 := by
  sorry

end cubic_factorization_l2106_210614


namespace right_triangle_inradius_l2106_210690

/-- The inradius of a right triangle with side lengths 6, 8, and 10 is 2 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 6 ∧ b = 8 ∧ c = 10 →  -- Side lengths condition
  a^2 + b^2 = c^2 →         -- Right triangle condition
  (a + b + c) / 2 * r = (a * b) / 2 →  -- Area formula using inradius
  r = 2 := by
sorry

end right_triangle_inradius_l2106_210690


namespace subtraction_preserves_inequality_l2106_210683

theorem subtraction_preserves_inequality (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end subtraction_preserves_inequality_l2106_210683


namespace four_digit_perfect_square_l2106_210653

theorem four_digit_perfect_square : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧  -- 4-digit number
  (∃ m : ℕ, n = m^2) ∧      -- perfect square
  (n / 100 = n % 100 + 1)   -- first two digits are one more than last two digits
  := by
  use 8281
  sorry

end four_digit_perfect_square_l2106_210653


namespace triangle_theorem_l2106_210637

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition from the problem -/
def satisfies_condition (t : Triangle) : Prop :=
  t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C - t.b - t.c = 0

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) 
  (h : satisfies_condition t) : 
  t.A = π / 3 ∧ 
  (t.a = Real.sqrt 3 → 
    ∀ (area : ℝ), area ≤ 3 * Real.sqrt 3 / 4 → 
      ∃ (t' : Triangle), satisfies_condition t' ∧ t'.a = Real.sqrt 3 ∧ 
        area = 1 / 2 * t'.b * t'.c * Real.sin t'.A) :=
sorry

end triangle_theorem_l2106_210637


namespace plot_area_calculation_l2106_210631

/-- Represents the area of a rectangular plot of land in acres, given its dimensions in miles. -/
def plot_area (length width : ℝ) : ℝ :=
  length * width * 640

/-- Theorem stating that a rectangular plot of land with dimensions 20 miles by 30 miles has an area of 384000 acres. -/
theorem plot_area_calculation :
  plot_area 30 20 = 384000 := by
  sorry


end plot_area_calculation_l2106_210631


namespace danielle_spending_l2106_210684

/-- Represents the cost and yield of supplies for making popsicles. -/
structure PopsicleSupplies where
  mold_cost : ℕ
  stick_pack_cost : ℕ
  stick_pack_size : ℕ
  juice_bottle_cost : ℕ
  popsicles_per_bottle : ℕ
  remaining_sticks : ℕ

/-- Calculates the total cost of supplies for making popsicles. -/
def total_cost (supplies : PopsicleSupplies) : ℕ :=
  supplies.mold_cost + supplies.stick_pack_cost +
  (supplies.stick_pack_size - supplies.remaining_sticks) / supplies.popsicles_per_bottle * supplies.juice_bottle_cost

/-- Theorem stating that Danielle's total spending on supplies equals $10. -/
theorem danielle_spending (supplies : PopsicleSupplies)
  (h1 : supplies.mold_cost = 3)
  (h2 : supplies.stick_pack_cost = 1)
  (h3 : supplies.stick_pack_size = 100)
  (h4 : supplies.juice_bottle_cost = 2)
  (h5 : supplies.popsicles_per_bottle = 20)
  (h6 : supplies.remaining_sticks = 40) :
  total_cost supplies = 10 := by
    sorry

end danielle_spending_l2106_210684


namespace parabola_sum_l2106_210688

def parabola (a b c : ℝ) (y : ℝ) : ℝ := a * y^2 + b * y + c

theorem parabola_sum (a b c : ℝ) :
  parabola a b c (-6) = 7 →
  parabola a b c (-4) = 5 →
  a + b + c = -35/2 := by sorry

end parabola_sum_l2106_210688


namespace stratified_sample_theorem_l2106_210698

/-- Represents a school with a given number of students -/
structure School where
  students : ℕ

/-- Calculates the number of students to be sampled from a school in a stratified sample -/
def stratifiedSampleSize (school : School) (totalStudents : ℕ) (sampleSize : ℕ) : ℕ :=
  (school.students * sampleSize) / totalStudents

theorem stratified_sample_theorem (schoolA schoolB schoolC : School) 
    (h1 : schoolA.students = 3600)
    (h2 : schoolB.students = 5400)
    (h3 : schoolC.students = 1800)
    (totalSampleSize : ℕ)
    (h4 : totalSampleSize = 90) :
  let totalStudents := schoolA.students + schoolB.students + schoolC.students
  (stratifiedSampleSize schoolA totalStudents totalSampleSize = 30) ∧
  (stratifiedSampleSize schoolB totalStudents totalSampleSize = 45) ∧
  (stratifiedSampleSize schoolC totalStudents totalSampleSize = 15) := by
  sorry


end stratified_sample_theorem_l2106_210698


namespace angle_C_is_pi_third_area_of_triangle_l2106_210695

namespace TriangleProof

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The determinant condition from the problem -/
def determinant_condition (t : Triangle) : Prop :=
  2 * t.c * Real.sin t.C = (2 * t.a - t.b) * Real.sin t.A + (2 * t.b - t.a) * Real.sin t.B

/-- Theorem 1: If the determinant condition holds, then C = π/3 -/
theorem angle_C_is_pi_third (t : Triangle) 
  (h : determinant_condition t) : t.C = Real.pi / 3 := by
  sorry

/-- Theorem 2: Area of the triangle under given conditions -/
theorem area_of_triangle (t : Triangle) 
  (h1 : Real.sin t.A = 4/5)
  (h2 : t.C = 2 * Real.pi / 3)
  (h3 : t.c = Real.sqrt 3) : 
  (1/2) * t.a * t.c * Real.sin t.B = (18 - 8 * Real.sqrt 3) / 25 := by
  sorry

end TriangleProof

end angle_C_is_pi_third_area_of_triangle_l2106_210695


namespace max_volume_container_l2106_210608

/-- Represents the dimensions of a rectangular container --/
structure ContainerDimensions where
  length : Real
  width : Real
  height : Real

/-- Calculates the volume of a rectangular container --/
def volume (d : ContainerDimensions) : Real :=
  d.length * d.width * d.height

/-- Represents the constraints of the problem --/
def containerConstraints (d : ContainerDimensions) : Prop :=
  d.length + d.width + d.height = 7.4 ∧  -- Half of the total bar length
  d.length = d.width + 0.5

/-- The main theorem to prove --/
theorem max_volume_container :
  ∃ (d : ContainerDimensions),
    containerConstraints d ∧
    d.height = 1.2 ∧
    volume d = 1.8 ∧
    (∀ (d' : ContainerDimensions), containerConstraints d' → volume d' ≤ volume d) :=
sorry

end max_volume_container_l2106_210608


namespace three_card_selection_l2106_210617

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (face_cards_per_suit : Nat)

/-- Calculates the number of ways to choose 3 cards from a deck
    such that all three cards are of different suits and one is a face card -/
def choose_three_cards (d : Deck) : Nat :=
  d.suits * d.face_cards_per_suit * (d.suits - 1).choose 2 * (d.cards_per_suit ^ 2)

/-- Theorem stating the number of ways to choose 3 cards from a standard deck
    with the given conditions -/
theorem three_card_selection (d : Deck) 
  (h1 : d.cards = 52)
  (h2 : d.suits = 4)
  (h3 : d.cards_per_suit = 13)
  (h4 : d.face_cards_per_suit = 3) :
  choose_three_cards d = 6084 := by
  sorry

#eval choose_three_cards { cards := 52, suits := 4, cards_per_suit := 13, face_cards_per_suit := 3 }

end three_card_selection_l2106_210617


namespace hyperbola_eccentricity_l2106_210633

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where:
    - The distance from the focus to the asymptote is 2√3
    - The minimum distance from a point on the right branch to the right focus is 2
    Then the eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity (a b c : ℝ) : 
  (∀ x y, x^2/a^2 - y^2/b^2 = 1 → 
    b * c / Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 3 ∧ 
    c - a = 2) → 
  c / a = 2 := by sorry

end hyperbola_eccentricity_l2106_210633


namespace fraction_to_decimal_l2106_210689

theorem fraction_to_decimal : 13 / 243 = 0.00416 := by
  sorry

end fraction_to_decimal_l2106_210689


namespace milk_cartons_accepted_l2106_210604

theorem milk_cartons_accepted (total_cartons : ℕ) (num_customers : ℕ) (damaged_per_customer : ℕ) 
  (h1 : total_cartons = 400)
  (h2 : num_customers = 4)
  (h3 : damaged_per_customer = 60) :
  (total_cartons / num_customers - damaged_per_customer) * num_customers = 160 :=
by sorry

end milk_cartons_accepted_l2106_210604


namespace polynomial_simplification_l2106_210680

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + x^2 + 6 * x - 8) = x^3 + 3 * x^2 + 3 * x + 3 :=
by sorry

end polynomial_simplification_l2106_210680


namespace costume_material_cost_l2106_210623

/-- Calculates the total cost of material for Jenna's costume --/
theorem costume_material_cost : 
  let skirt_length : ℕ := 12
  let skirt_width : ℕ := 4
  let num_skirts : ℕ := 3
  let bodice_area : ℕ := 2
  let sleeve_area : ℕ := 5
  let num_sleeves : ℕ := 2
  let cost_per_sqft : ℕ := 3
  
  skirt_length * skirt_width * num_skirts + 
  bodice_area + 
  sleeve_area * num_sleeves * cost_per_sqft = 468 := by
  sorry

end costume_material_cost_l2106_210623


namespace bread_rising_times_l2106_210639

/-- Represents the bread-making process with given time constraints --/
def BreadMaking (total_time rising_time kneading_time baking_time : ℕ) :=
  {n : ℕ // n * rising_time + kneading_time + baking_time = total_time}

/-- Theorem stating that Mark lets the bread rise twice --/
theorem bread_rising_times :
  BreadMaking 280 120 10 30 = {n : ℕ // n = 2} :=
by sorry

end bread_rising_times_l2106_210639


namespace million_factorizations_l2106_210658

def million : ℕ := 1000000

/-- The number of ways to represent 1,000,000 as a product of three factors when order matters -/
def distinct_factorizations : ℕ := 784

/-- The number of ways to represent 1,000,000 as a product of three factors when order doesn't matter -/
def identical_factorizations : ℕ := 139

/-- Function to count the number of ways to represent a number as a product of three factors -/
def count_factorizations (n : ℕ) (order_matters : Bool) : ℕ := sorry

theorem million_factorizations :
  (count_factorizations million true = distinct_factorizations) ∧
  (count_factorizations million false = identical_factorizations) := by sorry

end million_factorizations_l2106_210658


namespace arithmetic_sequence_common_difference_l2106_210624

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 1 + a 9 = 10)
  (h_second : a 2 = -1) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l2106_210624


namespace bamboo_problem_l2106_210687

theorem bamboo_problem (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 1 + a 2 + a 3 + a 4 = 3 →   -- sum of first 4 terms
  a 7 + a 8 + a 9 = 4 →         -- sum of last 3 terms
  a 5 = 67 / 66 := by
sorry

end bamboo_problem_l2106_210687


namespace village_population_percentage_l2106_210642

theorem village_population_percentage : 
  let part : ℕ := 23040
  let total : ℕ := 38400
  let percentage : ℚ := (part : ℚ) / (total : ℚ) * 100
  percentage = 60 := by sorry

end village_population_percentage_l2106_210642


namespace quadratic_inequality_l2106_210626

theorem quadratic_inequality (a b c A B C : ℝ) 
  (ha : a ≠ 0) (hA : A ≠ 0)
  (h : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) :
  |b^2 - 4*a*c| ≤ |B^2 - 4*A*C| := by sorry

end quadratic_inequality_l2106_210626


namespace yangzhou_construction_area_scientific_notation_l2106_210641

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem yangzhou_construction_area_scientific_notation :
  toScientificNotation 330100000 = ScientificNotation.mk 3.301 8 (by norm_num) :=
sorry

end yangzhou_construction_area_scientific_notation_l2106_210641


namespace normal_force_wooden_blocks_l2106_210699

/-- The normal force from a system of wooden blocks to a table -/
theorem normal_force_wooden_blocks
  (M m : ℝ)  -- Masses of the larger block and smaller cubes
  (α β : ℝ)  -- Angles of the sides of the larger block
  (hM : M > 0)  -- Mass of larger block is positive
  (hm : m > 0)  -- Mass of smaller cubes is positive
  (hα : 0 < α ∧ α < π/2)  -- α is between 0 and π/2
  (hβ : 0 < β ∧ β < π/2)  -- β is between 0 and π/2
  (g : ℝ)  -- Gravitational acceleration
  (hg : g > 0)  -- Gravitational acceleration is positive
  : ℝ :=
  M * g + m * g * (Real.cos α ^ 2 + Real.cos β ^ 2)

#check normal_force_wooden_blocks

end normal_force_wooden_blocks_l2106_210699


namespace purely_imaginary_complex_number_l2106_210663

theorem purely_imaginary_complex_number (m : ℝ) : 
  (Complex.mk (m^2 - 3*m) (m^2 - 5*m + 6)).im ≠ 0 ∧ 
  (Complex.mk (m^2 - 3*m) (m^2 - 5*m + 6)).re = 0 → 
  m = 0 := by
sorry

end purely_imaginary_complex_number_l2106_210663


namespace brahmagupta_formula_l2106_210673

/-- Represents a convex quadrilateral ABCD with side lengths a, b, c, d and diagonal lengths m, n -/
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  m : ℝ
  n : ℝ
  A : ℝ
  C : ℝ
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  d_pos : d > 0
  m_pos : m > 0
  n_pos : n > 0

/-- The Brahmagupta's formula for a convex quadrilateral -/
theorem brahmagupta_formula (q : ConvexQuadrilateral) :
  q.m^2 * q.n^2 = q.a^2 * q.c^2 + q.b^2 * q.d^2 - 2 * q.a * q.b * q.c * q.d * Real.cos (q.A + q.C) :=
by sorry

end brahmagupta_formula_l2106_210673


namespace james_socks_l2106_210660

/-- The total number of socks James has -/
def total_socks (red_pairs black_pairs white_socks : ℕ) : ℕ :=
  2 * red_pairs + 2 * black_pairs + white_socks

/-- Theorem stating the total number of socks James has -/
theorem james_socks : 
  ∀ (red_pairs black_pairs white_socks : ℕ),
    red_pairs = 20 →
    black_pairs = red_pairs / 2 →
    white_socks = 2 * (2 * red_pairs + 2 * black_pairs) →
    total_socks red_pairs black_pairs white_socks = 180 := by
  sorry

end james_socks_l2106_210660


namespace fraction_equality_l2106_210619

theorem fraction_equality (x : ℝ) (h : x ≠ 1) : -2 / (2 * x - 2) = 1 / (1 - x) := by
  sorry

end fraction_equality_l2106_210619


namespace range_of_a_l2106_210636

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) ↔ -1 < a ∧ a < 0 := by
sorry

end range_of_a_l2106_210636


namespace largest_two_three_digit_multiples_sum_l2106_210602

theorem largest_two_three_digit_multiples_sum : ∃ (a b : ℕ), 
  (a > 0 ∧ a < 100 ∧ a % 5 = 0 ∧ ∀ x : ℕ, x > 0 ∧ x < 100 ∧ x % 5 = 0 → x ≤ a) ∧
  (b > 0 ∧ b < 1000 ∧ b % 7 = 0 ∧ ∀ y : ℕ, y > 0 ∧ y < 1000 ∧ y % 7 = 0 → y ≤ b) ∧
  a + b = 1089 := by
sorry

end largest_two_three_digit_multiples_sum_l2106_210602


namespace guitar_center_discount_l2106_210685

/-- The discount offered by Guitar Center for a guitar with a suggested retail price of $1000,
    given that Guitar Center has a $100 shipping fee, Sweetwater has a 10% discount with free shipping,
    and the difference in final price between the two stores is $50. -/
theorem guitar_center_discount (suggested_price : ℕ) (gc_shipping : ℕ) (sw_discount_percent : ℕ) (price_difference : ℕ) :
  suggested_price = 1000 →
  gc_shipping = 100 →
  sw_discount_percent = 10 →
  price_difference = 50 →
  ∃ (gc_discount : ℕ), gc_discount = 150 :=
by sorry

end guitar_center_discount_l2106_210685


namespace hours_worked_l2106_210643

def hourly_wage : ℝ := 3.25
def total_earned : ℝ := 26

theorem hours_worked : 
  (total_earned / hourly_wage : ℝ) = 8 := by sorry

end hours_worked_l2106_210643


namespace smaller_number_problem_l2106_210654

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 12) (h3 : x > y) :
  y = 14 := by
  sorry

end smaller_number_problem_l2106_210654


namespace not_p_and_q_l2106_210664

-- Define proposition p
def p : Prop := ∀ a b : ℝ, a > b → a > b^2

-- Define proposition q
def q : Prop := (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0 → x ≤ 1) ∧ 
                (∃ x : ℝ, x ≤ 1 ∧ x^2 + 2*x - 3 > 0)

-- Theorem to prove
theorem not_p_and_q : ¬p ∧ q := by
  sorry

end not_p_and_q_l2106_210664


namespace geometric_sequence_property_l2106_210635

/-- A sequence is geometric if the ratio between any two consecutive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geometric : IsGeometric a) 
    (h_sum : a 4 + a 6 = 10) : 
  a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 100 := by
  sorry

end geometric_sequence_property_l2106_210635


namespace child_height_end_of_year_l2106_210649

/-- Calculates the child's height at the end of the school year given initial height and growth rates -/
def final_height (initial_height : ℝ) (rate1 : ℝ) (rate2 : ℝ) (rate3 : ℝ) : ℝ :=
  initial_height + (3 * rate1) + (3 * rate2) + (6 * rate3)

/-- Theorem stating that the child's height at the end of the school year is 43.3 inches -/
theorem child_height_end_of_year :
  final_height 38.5 0.5 0.3 0.4 = 43.3 := by
  sorry

#eval final_height 38.5 0.5 0.3 0.4

end child_height_end_of_year_l2106_210649


namespace geometric_series_common_ratio_l2106_210686

/-- The common ratio of the geometric series 2/3 + 4/9 + 8/27 + ... is 2/3 -/
theorem geometric_series_common_ratio : 
  let a : ℕ → ℚ := fun n => (2 / 3) * (2 / 3)^n
  ∀ n : ℕ, a (n + 1) / a n = 2 / 3 :=
by sorry

end geometric_series_common_ratio_l2106_210686


namespace simplify_expression_l2106_210665

theorem simplify_expression (a : ℝ) (h : -1 < a ∧ a < 0) :
  Real.sqrt ((a + 1/a)^2 - 4) + Real.sqrt ((a - 1/a)^2 + 4) = -2/a := by
  sorry

end simplify_expression_l2106_210665


namespace polar_to_rectangular_l2106_210625

/-- The rectangular coordinate equation of a curve given its polar equation -/
theorem polar_to_rectangular (ρ θ : ℝ) (h : ρ * Real.cos θ = 2) : 
  ∃ x : ℝ, x = 2 := by sorry

end polar_to_rectangular_l2106_210625
