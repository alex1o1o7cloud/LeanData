import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l3441_344130

theorem inequality_proof (n : ℕ+) : (2*n+1)^(n : ℕ) ≥ (2*n)^(n : ℕ) + (2*n-1)^(n : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3441_344130


namespace NUMINAMATH_CALUDE_function_decomposition_l3441_344154

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem function_decomposition (f g : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even g) 
  (h_sum : ∀ x, f x + g x = 2^x + 2*x) :
  (∀ x, g x = (2^x + 2^(-x)) / 2) ∧ 
  (∀ x, f x = 2^(x-1) + 2*x - 2^(-x-1)) :=
sorry

end NUMINAMATH_CALUDE_function_decomposition_l3441_344154


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3441_344138

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3441_344138


namespace NUMINAMATH_CALUDE_triangle_side_length_l3441_344111

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 →  -- 60 degrees in radians
  B = π/4 →  -- 45 degrees in radians
  b = 2 → 
  (a / Real.sin A = b / Real.sin B) →  -- Law of sines
  a = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3441_344111


namespace NUMINAMATH_CALUDE_extreme_values_and_tangent_line_l3441_344171

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_and_tangent_line 
  (a b : ℝ) 
  (h1 : f' a b 1 = 0) 
  (h2 : f' a b 2 = 0) :
  (a = -3 ∧ b = 4) ∧ 
  (∃ (k m : ℝ), k = 12 ∧ m = 8 ∧ ∀ (x y : ℝ), y = k * x + m ↔ y = (f' (-3) 4 0) * x + f (-3) 4 0) := by
  sorry

end NUMINAMATH_CALUDE_extreme_values_and_tangent_line_l3441_344171


namespace NUMINAMATH_CALUDE_solid_color_non_yellow_purple_percentage_l3441_344103

/-- Represents the distribution of marble types and colors -/
structure MarbleDistribution where
  solid_colored : ℝ
  striped : ℝ
  dotted : ℝ
  swirl_patterned : ℝ
  red_solid : ℝ
  blue_solid : ℝ
  green_solid : ℝ
  yellow_solid : ℝ
  purple_solid : ℝ

/-- The given marble distribution -/
def given_distribution : MarbleDistribution :=
  { solid_colored := 0.70
    striped := 0.10
    dotted := 0.10
    swirl_patterned := 0.10
    red_solid := 0.25
    blue_solid := 0.25
    green_solid := 0.20
    yellow_solid := 0.15
    purple_solid := 0.15 }

/-- Theorem stating that 49% of all marbles are solid-colored and neither yellow nor purple -/
theorem solid_color_non_yellow_purple_percentage
  (d : MarbleDistribution)
  (h1 : d.solid_colored + d.striped + d.dotted + d.swirl_patterned = 1)
  (h2 : d.red_solid + d.blue_solid + d.green_solid + d.yellow_solid + d.purple_solid = 1)
  (h3 : d = given_distribution) :
  d.solid_colored * (d.red_solid + d.blue_solid + d.green_solid) = 0.49 := by
  sorry

end NUMINAMATH_CALUDE_solid_color_non_yellow_purple_percentage_l3441_344103


namespace NUMINAMATH_CALUDE_isosceles_triangle_circle_properties_main_theorem_l3441_344181

/-- An isosceles triangle inscribed in a circle -/
structure IsoscelesTriangleInCircle where
  /-- Length of the two equal sides of the isosceles triangle -/
  side : ℝ
  /-- Length of the base of the isosceles triangle -/
  base : ℝ
  /-- Radius of the circumscribed circle -/
  radius : ℝ

/-- Theorem about the radius and area of a circle circumscribing an isosceles triangle -/
theorem isosceles_triangle_circle_properties (t : IsoscelesTriangleInCircle)
  (h_side : t.side = 4)
  (h_base : t.base = 3) :
  t.radius = 3.5 ∧ t.radius^2 * π = 12.25 * π := by
  sorry

/-- Main theorem combining the properties -/
theorem main_theorem :
  ∃ t : IsoscelesTriangleInCircle,
    t.side = 4 ∧
    t.base = 3 ∧
    t.radius = 3.5 ∧
    t.radius^2 * π = 12.25 * π := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_circle_properties_main_theorem_l3441_344181


namespace NUMINAMATH_CALUDE_quadratic_solution_positive_l3441_344134

theorem quadratic_solution_positive (x : ℝ) : 
  x > 0 ∧ 4 * x^2 + 8 * x - 20 = 0 ↔ x = Real.sqrt 6 - 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_positive_l3441_344134


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3441_344198

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + b^3 + c^3) / ((a + b + c)^3 - 26*a*b*c) ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3441_344198


namespace NUMINAMATH_CALUDE_mod_fourteen_power_ninety_six_minus_eight_l3441_344174

theorem mod_fourteen_power_ninety_six_minus_eight :
  (5^96 - 8) % 14 = 7 := by
sorry

end NUMINAMATH_CALUDE_mod_fourteen_power_ninety_six_minus_eight_l3441_344174


namespace NUMINAMATH_CALUDE_value_in_numerator_l3441_344164

theorem value_in_numerator (N V : ℤ) : 
  N = 1280 → (N + 720) / 125 = V / 462 → V = 7392 := by sorry

end NUMINAMATH_CALUDE_value_in_numerator_l3441_344164


namespace NUMINAMATH_CALUDE_six_is_simplified_quadratic_radical_l3441_344129

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_simplified_quadratic_radical (n : ℕ) : Prop :=
  n ≠ 0 ∧ ¬ is_perfect_square n ∧ ∀ m : ℕ, m > 1 → is_perfect_square m → ¬ (m ∣ n)

theorem six_is_simplified_quadratic_radical :
  is_simplified_quadratic_radical 6 :=
sorry

end NUMINAMATH_CALUDE_six_is_simplified_quadratic_radical_l3441_344129


namespace NUMINAMATH_CALUDE_unbroken_seashells_l3441_344119

theorem unbroken_seashells (total : ℕ) (broken : ℕ) (h1 : total = 23) (h2 : broken = 11) :
  total - broken = 12 := by
  sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l3441_344119


namespace NUMINAMATH_CALUDE_choir_selection_l3441_344190

theorem choir_selection (boys girls : ℕ) (h1 : boys = 3) (h2 : girls = 5) :
  let total := boys + girls
  (Nat.choose boys 2 * Nat.choose girls 2 = 30) ∧
  (Nat.choose total 4 - Nat.choose girls 4 = 65) :=
by sorry

end NUMINAMATH_CALUDE_choir_selection_l3441_344190


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l3441_344117

theorem quadratic_integer_roots (b : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ (x^2 - b*x + 3*b = 0) ∧ (y^2 - b*y + 3*b = 0)) → 
  (b = 9 ∨ b = -6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l3441_344117


namespace NUMINAMATH_CALUDE_sequence_increasing_l3441_344185

def a (n : ℕ+) : ℚ := (2 * n) / (2 * n + 1)

theorem sequence_increasing (n : ℕ+) : a n < a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_increasing_l3441_344185


namespace NUMINAMATH_CALUDE_quadratic_root_k_range_l3441_344116

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - k*x - 2

-- Theorem statement
theorem quadratic_root_k_range :
  ∀ k : ℝ, (∃ x : ℝ, 2 < x ∧ x < 5 ∧ f k x = 0) → (1 < k ∧ k < 23/5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_k_range_l3441_344116


namespace NUMINAMATH_CALUDE_hotel_weekly_loss_l3441_344153

def weekly_profit_loss (operations_expenses taxes employee_salaries : ℚ) : ℚ :=
  let meetings_income := (5 / 8) * operations_expenses
  let events_income := (3 / 10) * operations_expenses
  let rooms_income := (11 / 20) * operations_expenses
  let total_income := meetings_income + events_income + rooms_income
  let total_expenses := operations_expenses + taxes + employee_salaries
  total_income - total_expenses

theorem hotel_weekly_loss :
  weekly_profit_loss 5000 1200 2500 = -1325 :=
by sorry

end NUMINAMATH_CALUDE_hotel_weekly_loss_l3441_344153


namespace NUMINAMATH_CALUDE_blue_notebook_cost_l3441_344156

/-- The cost of each blue notebook given Mike's purchase details -/
theorem blue_notebook_cost 
  (total_spent : ℕ)
  (total_notebooks : ℕ)
  (red_notebooks : ℕ)
  (red_price : ℕ)
  (green_notebooks : ℕ)
  (green_price : ℕ)
  (h1 : total_spent = 37)
  (h2 : total_notebooks = 12)
  (h3 : red_notebooks = 3)
  (h4 : red_price = 4)
  (h5 : green_notebooks = 2)
  (h6 : green_price = 2)
  (h7 : total_notebooks = red_notebooks + green_notebooks + (total_notebooks - red_notebooks - green_notebooks))
  : (total_spent - red_notebooks * red_price - green_notebooks * green_price) / (total_notebooks - red_notebooks - green_notebooks) = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_notebook_cost_l3441_344156


namespace NUMINAMATH_CALUDE_mike_has_one_unbroken_seashell_l3441_344191

/-- Represents the number of unbroken seashells Mike has left after his beach trip and giving away one shell. -/
def unbroken_seashells_left : ℕ :=
  let total_seashells := 6
  let cone_shells := 3
  let conch_shells := 3
  let broken_cone_shells := 2
  let broken_conch_shells := 2
  let unbroken_cone_shells := cone_shells - broken_cone_shells
  let unbroken_conch_shells := conch_shells - broken_conch_shells
  let given_away_shells := 1
  unbroken_cone_shells + (unbroken_conch_shells - given_away_shells)

/-- Theorem stating that Mike has 1 unbroken seashell left. -/
theorem mike_has_one_unbroken_seashell : unbroken_seashells_left = 1 := by
  sorry

end NUMINAMATH_CALUDE_mike_has_one_unbroken_seashell_l3441_344191


namespace NUMINAMATH_CALUDE_min_value_theorem_l3441_344132

/-- Given positive real numbers m and n, vectors a and b, and a parallel to b,
    prove that the minimum value of 1/m + 2/n is 3 + 2√2 -/
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (a b : Fin 2 → ℝ)
  (ha : a = λ i => if i = 0 then m else 1)
  (hb : b = λ i => if i = 0 then 1 - n else 1)
  (parallel : ∃ (k : ℝ), a = λ i => k * (b i)) :
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 2/y ≥ 3 + 2 * Real.sqrt 2) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 2/y = 3 + 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3441_344132


namespace NUMINAMATH_CALUDE_regular_pentagon_perimeter_l3441_344125

/-- The sum of sides of a regular pentagon with side length 15 cm is 75 cm. -/
theorem regular_pentagon_perimeter (side_length : ℝ) (n_sides : ℕ) : 
  side_length = 15 → n_sides = 5 → side_length * n_sides = 75 := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_perimeter_l3441_344125


namespace NUMINAMATH_CALUDE_carrot_count_l3441_344161

theorem carrot_count (initial_carrots thrown_out_carrots picked_next_day : ℕ) :
  initial_carrots = 48 →
  thrown_out_carrots = 11 →
  picked_next_day = 15 →
  initial_carrots - thrown_out_carrots + picked_next_day = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_carrot_count_l3441_344161


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l3441_344176

/-- Number of ways to arrange n distinct objects in r positions --/
def arrangement (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of boxes --/
def num_boxes : ℕ := 7

/-- The number of balls --/
def num_balls : ℕ := 4

/-- The number of ways to arrange the balls satisfying all conditions --/
def valid_arrangements : ℕ :=
  arrangement num_balls num_balls * arrangement (num_balls + 1) 2 -
  arrangement 2 2 * arrangement 3 3 * arrangement 4 2

theorem valid_arrangements_count :
  valid_arrangements = 336 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l3441_344176


namespace NUMINAMATH_CALUDE_greatest_number_of_bouquets_l3441_344162

theorem greatest_number_of_bouquets (white_tulips red_tulips : ℕ) 
  (h_white : white_tulips = 21) (h_red : red_tulips = 91) : 
  (∃ (bouquets_count : ℕ) (white_per_bouquet red_per_bouquet : ℕ), 
    bouquets_count * white_per_bouquet = white_tulips ∧ 
    bouquets_count * red_per_bouquet = red_tulips ∧ 
    ∀ (other_count : ℕ) (other_white other_red : ℕ), 
      other_count * other_white = white_tulips → 
      other_count * other_red = red_tulips → 
      other_count ≤ bouquets_count) ∧ 
  (∃ (max_bouquets : ℕ), max_bouquets = 3 ∧ 
    ∀ (bouquets_count : ℕ) (white_per_bouquet red_per_bouquet : ℕ), 
      bouquets_count * white_per_bouquet = white_tulips → 
      bouquets_count * red_per_bouquet = red_tulips → 
      bouquets_count ≤ max_bouquets) := by
sorry

end NUMINAMATH_CALUDE_greatest_number_of_bouquets_l3441_344162


namespace NUMINAMATH_CALUDE_complement_of_P_l3441_344179

def P : Set ℝ := {x | |x + 3| + |x + 6| = 3}

theorem complement_of_P : 
  {x : ℝ | x < -6 ∨ x > -3} = (Set.univ : Set ℝ) \ P := by sorry

end NUMINAMATH_CALUDE_complement_of_P_l3441_344179


namespace NUMINAMATH_CALUDE_impossible_cross_sections_l3441_344108

-- Define a cube
structure Cube where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define a plane
structure Plane where
  normal_vector : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

-- Define possible shapes of cross-sections
inductive CrossSectionShape
  | ObtuseTriangle
  | RightAngledTrapezoid
  | Rhombus
  | RegularPentagon
  | RegularHexagon

-- Function to determine if a shape is possible
def is_possible_cross_section (cube : Cube) (plane : Plane) (shape : CrossSectionShape) : Prop :=
  match shape with
  | CrossSectionShape.ObtuseTriangle => False
  | CrossSectionShape.RightAngledTrapezoid => False
  | CrossSectionShape.Rhombus => True
  | CrossSectionShape.RegularPentagon => False
  | CrossSectionShape.RegularHexagon => True

-- Theorem statement
theorem impossible_cross_sections (cube : Cube) (plane : Plane) :
  ¬(is_possible_cross_section cube plane CrossSectionShape.ObtuseTriangle) ∧
  ¬(is_possible_cross_section cube plane CrossSectionShape.RightAngledTrapezoid) ∧
  ¬(is_possible_cross_section cube plane CrossSectionShape.RegularPentagon) :=
sorry

end NUMINAMATH_CALUDE_impossible_cross_sections_l3441_344108


namespace NUMINAMATH_CALUDE_rate_squares_sum_l3441_344109

theorem rate_squares_sum : ∃ (b j s : ℕ),
  3 * b + 2 * j + 4 * s = 70 ∧
  4 * b + 3 * j + 2 * s = 88 ∧
  b^2 + j^2 + s^2 = 405 := by
sorry

end NUMINAMATH_CALUDE_rate_squares_sum_l3441_344109


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_180_l3441_344184

/-- A function that returns true if a number is a five-digit number -/
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

/-- A function that returns the product of digits of a natural number -/
def digit_product (n : ℕ) : ℕ :=
  sorry

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem largest_five_digit_with_product_180 :
  ∃ M : ℕ, is_five_digit M ∧
           digit_product M = 180 ∧
           (∀ n : ℕ, is_five_digit n → digit_product n = 180 → n ≤ M) ∧
           digit_sum M = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_180_l3441_344184


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3441_344196

/-- Given a distance of 88 miles and a time of 4 hours, prove that the average speed is 22 miles per hour. -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 88) (h2 : time = 4) :
  distance / time = 22 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3441_344196


namespace NUMINAMATH_CALUDE_bakery_inventory_theorem_l3441_344135

/-- Represents the inventory and sales of a bakery --/
structure BakeryInventory where
  cheesecakes_display : ℕ
  cheesecakes_fridge : ℕ
  cherry_pies_ready : ℕ
  cherry_pies_oven : ℕ
  chocolate_eclairs_counter : ℕ
  chocolate_eclairs_pantry : ℕ
  cheesecakes_sold : ℕ
  cherry_pies_sold : ℕ
  chocolate_eclairs_sold : ℕ

/-- Calculates the total number of desserts left to sell --/
def desserts_left_to_sell (inventory : BakeryInventory) : ℕ :=
  (inventory.cheesecakes_display + inventory.cheesecakes_fridge - inventory.cheesecakes_sold) +
  (inventory.cherry_pies_ready + inventory.cherry_pies_oven - inventory.cherry_pies_sold) +
  (inventory.chocolate_eclairs_counter + inventory.chocolate_eclairs_pantry - inventory.chocolate_eclairs_sold)

/-- Theorem stating that given the specific inventory and sales, there are 62 desserts left to sell --/
theorem bakery_inventory_theorem (inventory : BakeryInventory) 
  (h1 : inventory.cheesecakes_display = 10)
  (h2 : inventory.cheesecakes_fridge = 15)
  (h3 : inventory.cherry_pies_ready = 12)
  (h4 : inventory.cherry_pies_oven = 20)
  (h5 : inventory.chocolate_eclairs_counter = 20)
  (h6 : inventory.chocolate_eclairs_pantry = 10)
  (h7 : inventory.cheesecakes_sold = 7)
  (h8 : inventory.cherry_pies_sold = 8)
  (h9 : inventory.chocolate_eclairs_sold = 10) :
  desserts_left_to_sell inventory = 62 := by
  sorry

end NUMINAMATH_CALUDE_bakery_inventory_theorem_l3441_344135


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3441_344143

theorem square_plus_reciprocal_square (n : ℝ) (h : n + 1/n = 10) :
  n^2 + 1/n^2 + 6 = 104 := by sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3441_344143


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_five_l3441_344189

theorem sum_of_solutions_is_five : 
  ∃! (s : ℝ), ∀ (x : ℝ), (x + 25 / x = 10) → (s = x) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_five_l3441_344189


namespace NUMINAMATH_CALUDE_sum_of_ninth_powers_of_roots_l3441_344146

theorem sum_of_ninth_powers_of_roots (u v w : ℂ) : 
  (u^3 - 3*u - 1 = 0) → 
  (v^3 - 3*v - 1 = 0) → 
  (w^3 - 3*w - 1 = 0) → 
  u^9 + v^9 + w^9 = 246 := by sorry

end NUMINAMATH_CALUDE_sum_of_ninth_powers_of_roots_l3441_344146


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3441_344106

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + (2*m - 5) * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 + (2*m - 5) * y + 12 = 0 → y = x) ↔ 
  m = 8.5 ∨ m = -3.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3441_344106


namespace NUMINAMATH_CALUDE_interest_rate_problem_l3441_344175

/-- The interest rate problem --/
theorem interest_rate_problem (total_investment : ℝ) (total_interest : ℝ) 
  (amount_at_r : ℝ) (rate_known : ℝ) :
  total_investment = 6000 →
  total_interest = 624 →
  amount_at_r = 1800 →
  rate_known = 0.11 →
  ∃ (r : ℝ), 
    amount_at_r * r + (total_investment - amount_at_r) * rate_known = total_interest ∧
    r = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l3441_344175


namespace NUMINAMATH_CALUDE_increasing_function_condition_l3441_344160

/-- The function f(x) = x^2 + ax + 1/x is increasing on (1/3, +∞) if and only if a ≥ 25/3 -/
theorem increasing_function_condition (a : ℝ) :
  (∀ x > 1/3, Monotone (fun x => x^2 + a*x + 1/x)) ↔ a ≥ 25/3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l3441_344160


namespace NUMINAMATH_CALUDE_fractional_equation_m_range_l3441_344169

theorem fractional_equation_m_range :
  ∀ m x : ℝ,
  (m / (1 - x) - 2 / (x - 1) = 1) →
  (x ≥ 0) →
  (x ≠ 1) →
  (m ≤ -1 ∧ m ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_m_range_l3441_344169


namespace NUMINAMATH_CALUDE_triangle_area_fraction_l3441_344173

/-- The area of a triangle given the coordinates of its vertices -/
def triangleArea (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The theorem stating that the area of the given triangle divided by the area of the grid equals 5/28 -/
theorem triangle_area_fraction :
  let a := (2, 2)
  let b := (6, 3)
  let c := (3, 6)
  let gridArea := 7 * 6
  (triangleArea a.1 a.2 b.1 b.2 c.1 c.2) / gridArea = 5 / 28 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_fraction_l3441_344173


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_four_l3441_344166

/-- Given vectors a and b in ℝ², prove that if a + 3b is parallel to a - b, then the x-coordinate of b is 4. -/
theorem parallel_vectors_imply_x_equals_four (a b : ℝ × ℝ) 
  (ha : a = (2, 1)) 
  (hb : b.2 = 2) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a + 3 • b = k • (a - b)) : 
  b.1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_four_l3441_344166


namespace NUMINAMATH_CALUDE_expression_value_l3441_344140

theorem expression_value (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  ∃ ε > 0, |x + (2 * x^3 / y^2) + (2 * y^3 / x^2) + y - 338| < ε :=
sorry

end NUMINAMATH_CALUDE_expression_value_l3441_344140


namespace NUMINAMATH_CALUDE_sum_lent_l3441_344101

/-- Given a sum of money divided into two parts where:
    1) The interest on the first part for 8 years at 3% per annum
       equals the interest on the second part for 3 years at 5% per annum
    2) The second part is Rs. 1680
    Prove that the total sum lent is Rs. 2730 -/
theorem sum_lent (first_part second_part : ℝ) : 
  second_part = 1680 →
  (first_part * 8 * 3) / 100 = (second_part * 3 * 5) / 100 →
  first_part + second_part = 2730 := by
  sorry

#check sum_lent

end NUMINAMATH_CALUDE_sum_lent_l3441_344101


namespace NUMINAMATH_CALUDE_T_is_three_rays_l3441_344104

/-- The set T of points in the coordinate plane -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               ((4 = x + 3 ∧ y - 5 ≤ 4) ∨
                (4 = y - 5 ∧ x + 3 ≤ 4) ∨
                (x + 3 = y - 5 ∧ 4 ≤ x + 3))}

/-- Definition of a ray starting from a point -/
def Ray (start : ℝ × ℝ) (dir : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = (start.1 + t * dir.1, start.2 + t * dir.2)}

/-- The three rays that should compose T -/
def ThreeRays : Set (ℝ × ℝ) :=
  Ray (1, 9) (0, -1) ∪ Ray (1, 9) (-1, 0) ∪ Ray (1, 9) (1, 1)

theorem T_is_three_rays : T = ThreeRays := by sorry

end NUMINAMATH_CALUDE_T_is_three_rays_l3441_344104


namespace NUMINAMATH_CALUDE_inequality_proof_l3441_344148

theorem inequality_proof (a b : ℝ) (ha : a > 1/2) (hb : b > 1/2) :
  a + 2*b - 5*a*b < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3441_344148


namespace NUMINAMATH_CALUDE_simplify_fraction_l3441_344182

theorem simplify_fraction : (90 + 54) / (150 - 90) = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3441_344182


namespace NUMINAMATH_CALUDE_orange_harvest_existence_l3441_344133

theorem orange_harvest_existence :
  ∃ (A B C D : ℕ), A + B + C + D = 56 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_existence_l3441_344133


namespace NUMINAMATH_CALUDE_ellipse_foci_l3441_344159

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 25 = 1

-- Define the foci coordinates
def foci : Set (ℝ × ℝ) :=
  {(0, 3), (0, -3)}

-- Theorem statement
theorem ellipse_foci :
  ∀ (f : ℝ × ℝ), f ∈ foci ↔
    (∃ (x y : ℝ), ellipse_equation x y ∧
      (x - f.1)^2 + (y - f.2)^2 +
      (x + f.1)^2 + (y + f.2)^2 = 4 * (5^2 + 4^2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_l3441_344159


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3441_344120

theorem midpoint_coordinate_sum (a b c d e f : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : d + e + f = 9) : 
  (a + b) / 2 + (b + c) / 2 + (c + a) / 2 = 15 ∧ 
  (d + e) / 2 + (e + f) / 2 + (f + d) / 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3441_344120


namespace NUMINAMATH_CALUDE_parent_chaperones_count_l3441_344149

/-- The number of parent chaperones on a school field trip -/
def num_parent_chaperones (total_students : ℕ) (num_teachers : ℕ) (students_left : ℕ) (chaperones_left : ℕ) (remaining_individuals : ℕ) : ℕ :=
  (remaining_individuals + students_left + chaperones_left) - (total_students + num_teachers)

theorem parent_chaperones_count :
  num_parent_chaperones 20 2 10 2 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_parent_chaperones_count_l3441_344149


namespace NUMINAMATH_CALUDE_youtube_video_length_l3441_344195

theorem youtube_video_length (x : ℝ) 
  (h1 : 6 * x + 6 * (x / 2) = 900) : x = 100 := by
  sorry

end NUMINAMATH_CALUDE_youtube_video_length_l3441_344195


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3441_344113

theorem sum_of_a_and_b (a b : ℚ) : 5 - Real.sqrt 3 * a = 2 * b + Real.sqrt 3 - a → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3441_344113


namespace NUMINAMATH_CALUDE_star_count_l3441_344136

theorem star_count (east : ℕ) (west : ℕ) : 
  east = 120 → 
  west = 6 * east → 
  east + west = 840 := by
sorry

end NUMINAMATH_CALUDE_star_count_l3441_344136


namespace NUMINAMATH_CALUDE_fixed_point_of_power_function_l3441_344100

/-- For any real α, the function f(x) = (x-1)^α passes through the point (2,1) -/
theorem fixed_point_of_power_function (α : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ (x - 1) ^ α
  f 2 = 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_power_function_l3441_344100


namespace NUMINAMATH_CALUDE_square_of_linear_expression_l3441_344150

theorem square_of_linear_expression (p : ℝ) (m : ℝ) : p ≠ 0 →
  (∃ a b : ℝ, ∀ x : ℝ, (9 * x^2 + 21 * x + 4 * m) / 9 = (a * x + b)^2) ∧
  (∃ a b : ℝ, (9 * (p - 1)^2 + 21 * (p - 1) + 4 * m) / 9 = (a * (p - 1) + b)^2) →
  m = 49 / 16 := by
sorry

end NUMINAMATH_CALUDE_square_of_linear_expression_l3441_344150


namespace NUMINAMATH_CALUDE_parabola_no_y_intercepts_l3441_344177

/-- The parabola defined by x = 3y^2 - 5y + 12 has no y-intercepts -/
theorem parabola_no_y_intercepts :
  ∀ y : ℝ, 3 * y^2 - 5 * y + 12 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_no_y_intercepts_l3441_344177


namespace NUMINAMATH_CALUDE_casper_candies_l3441_344141

/-- The number of candies Casper originally had -/
def original_candies : ℕ := 176

/-- The number of candies Casper gave to his brother on the first day -/
def candies_to_brother : ℕ := 3

/-- The number of candies Casper gave to his sister on the second day -/
def candies_to_sister : ℕ := 5

/-- The number of candies Casper ate on the third day -/
def final_candies : ℕ := 10

theorem casper_candies :
  let remaining_day1 := original_candies * 3 / 4 - candies_to_brother
  let remaining_day2 := remaining_day1 / 2 - candies_to_sister
  remaining_day2 = final_candies := by sorry

end NUMINAMATH_CALUDE_casper_candies_l3441_344141


namespace NUMINAMATH_CALUDE_bounce_count_correct_l3441_344122

/-- The smallest positive integer k such that 800 * (0.4^k) < 5 -/
def bounce_count : ℕ := 6

/-- The initial height of the ball in feet -/
def initial_height : ℝ := 800

/-- The ratio of the height after each bounce to the previous height -/
def bounce_ratio : ℝ := 0.4

/-- The target height in feet -/
def target_height : ℝ := 5

theorem bounce_count_correct : 
  (∀ k : ℕ, k < bounce_count → initial_height * (bounce_ratio ^ k) ≥ target_height) ∧
  initial_height * (bounce_ratio ^ bounce_count) < target_height :=
sorry

end NUMINAMATH_CALUDE_bounce_count_correct_l3441_344122


namespace NUMINAMATH_CALUDE_water_added_third_hour_is_one_l3441_344180

/-- Calculates the amount of water added in the third hour -/
def water_added_third_hour (initial_water : ℝ) (loss_rate : ℝ) (fourth_hour_addition : ℝ) (final_water : ℝ) : ℝ :=
  final_water - (initial_water - 3 * loss_rate + fourth_hour_addition)

theorem water_added_third_hour_is_one :
  let initial_water : ℝ := 40
  let loss_rate : ℝ := 2
  let fourth_hour_addition : ℝ := 3
  let final_water : ℝ := 36
  water_added_third_hour initial_water loss_rate fourth_hour_addition final_water = 1 := by
  sorry

#eval water_added_third_hour 40 2 3 36

end NUMINAMATH_CALUDE_water_added_third_hour_is_one_l3441_344180


namespace NUMINAMATH_CALUDE_rhino_fold_swap_impossible_l3441_344137

/-- Represents the number of folds on a rhinoceros -/
structure FoldCount where
  vertical : ℕ
  horizontal : ℕ

/-- Represents the state of folds on both sides of a rhinoceros -/
structure RhinoState where
  left : FoldCount
  right : FoldCount

def total_folds (state : RhinoState) : ℕ :=
  state.left.vertical + state.left.horizontal + state.right.vertical + state.right.horizontal

/-- Represents a single scratch action -/
inductive ScratchAction
  | left_vertical
  | left_horizontal
  | right_vertical
  | right_horizontal

/-- Applies a scratch action to a RhinoState -/
def apply_scratch (state : RhinoState) (action : ScratchAction) : RhinoState :=
  match action with
  | ScratchAction.left_vertical => 
      { left := { vertical := state.left.vertical - 2, horizontal := state.left.horizontal },
        right := { vertical := state.right.vertical + 1, horizontal := state.right.horizontal + 1 } }
  | ScratchAction.left_horizontal => 
      { left := { vertical := state.left.vertical, horizontal := state.left.horizontal - 2 },
        right := { vertical := state.right.vertical + 1, horizontal := state.right.horizontal + 1 } }
  | ScratchAction.right_vertical => 
      { left := { vertical := state.left.vertical + 1, horizontal := state.left.horizontal + 1 },
        right := { vertical := state.right.vertical - 2, horizontal := state.right.horizontal } }
  | ScratchAction.right_horizontal => 
      { left := { vertical := state.left.vertical + 1, horizontal := state.left.horizontal + 1 },
        right := { vertical := state.right.vertical, horizontal := state.right.horizontal - 2 } }

theorem rhino_fold_swap_impossible (initial : RhinoState) 
    (h_total : total_folds initial = 17) :
    ¬∃ (actions : List ScratchAction), 
      let final := actions.foldl apply_scratch initial
      total_folds final = 17 ∧ 
      final.left.vertical = initial.left.horizontal ∧
      final.left.horizontal = initial.left.vertical ∧
      final.right.vertical = initial.right.horizontal ∧
      final.right.horizontal = initial.right.vertical :=
  sorry

end NUMINAMATH_CALUDE_rhino_fold_swap_impossible_l3441_344137


namespace NUMINAMATH_CALUDE_triple_angle_square_equal_to_circle_l3441_344142

-- Tripling an angle
theorem triple_angle (α : Real) : ∃ β, β = 3 * α := by sorry

-- Constructing a square equal in area to a given circle
theorem square_equal_to_circle (r : Real) : 
  ∃ s, s^2 = π * r^2 := by sorry

end NUMINAMATH_CALUDE_triple_angle_square_equal_to_circle_l3441_344142


namespace NUMINAMATH_CALUDE_conference_handshakes_theorem_l3441_344145

/-- The number of handshakes in a conference with special conditions -/
def conference_handshakes (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) - (k.choose 2)

/-- Theorem: In a conference of 30 people, where 3 specific people don't shake hands with each other,
    the total number of handshakes is 432 -/
theorem conference_handshakes_theorem :
  conference_handshakes 30 3 = 432 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_theorem_l3441_344145


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3441_344187

theorem gcd_of_three_numbers : Nat.gcd 279 (Nat.gcd 372 465) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3441_344187


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_pairs_of_roots_l3441_344157

/-- Given a quintic polynomial x^5 + 10x^4 + 20x^3 + 15x^2 + 6x + 3, 
    this theorem states that the sum of reciprocals of products of pairs of its roots is 20/3 -/
theorem sum_of_reciprocal_pairs_of_roots (p q r s t : ℂ) : 
  p^5 + 10*p^4 + 20*p^3 + 15*p^2 + 6*p + 3 = 0 →
  q^5 + 10*q^4 + 20*q^3 + 15*q^2 + 6*q + 3 = 0 →
  r^5 + 10*r^4 + 20*r^3 + 15*r^2 + 6*r + 3 = 0 →
  s^5 + 10*s^4 + 20*s^3 + 15*s^2 + 6*s + 3 = 0 →
  t^5 + 10*t^4 + 20*t^3 + 15*t^2 + 6*t + 3 = 0 →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(p*t) + 1/(q*r) + 1/(q*s) + 1/(q*t) + 1/(r*s) + 1/(r*t) + 1/(s*t) = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_pairs_of_roots_l3441_344157


namespace NUMINAMATH_CALUDE_final_lives_correct_l3441_344183

/-- Given a player's initial lives, lost lives, and gained lives (before bonus),
    calculate the final number of lives after a secret bonus is applied. -/
def final_lives (initial_lives lost_lives gained_lives : ℕ) : ℕ :=
  initial_lives - lost_lives + 3 * gained_lives

/-- Theorem stating that the final_lives function correctly calculates
    the number of lives after the secret bonus is applied. -/
theorem final_lives_correct (X Y Z : ℕ) (h : Y ≤ X) :
  final_lives X Y Z = X - Y + 3 * Z :=
by sorry

end NUMINAMATH_CALUDE_final_lives_correct_l3441_344183


namespace NUMINAMATH_CALUDE_library_visitors_average_l3441_344151

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (days_in_month : ℕ) (h1 : sunday_visitors = 140) (h2 : other_day_visitors = 80) 
  (h3 : days_in_month = 30) :
  let sundays : ℕ := (days_in_month + 6) / 7
  let other_days : ℕ := days_in_month - sundays
  let total_visitors : ℕ := sundays * sunday_visitors + other_days * other_day_visitors
  (total_visitors : ℚ) / days_in_month = 88 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l3441_344151


namespace NUMINAMATH_CALUDE_sector_area_l3441_344147

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 4) (h2 : θ = π / 4) :
  (1 / 2) * θ * r^2 = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3441_344147


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3441_344139

theorem complex_fraction_simplification :
  (I : ℂ) / (3 + 4 * I) = (4 : ℂ) / 25 + (3 : ℂ) / 25 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3441_344139


namespace NUMINAMATH_CALUDE_negative_difference_l3441_344155

theorem negative_difference (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_l3441_344155


namespace NUMINAMATH_CALUDE_sunflower_seed_distribution_l3441_344178

theorem sunflower_seed_distribution (total_seeds : ℝ) (num_cans : ℝ) (seeds_per_can : ℝ) 
  (h1 : total_seeds = 54.0)
  (h2 : num_cans = 9.0)
  (h3 : seeds_per_can = total_seeds / num_cans) :
  seeds_per_can = 6.0 := by
sorry

end NUMINAMATH_CALUDE_sunflower_seed_distribution_l3441_344178


namespace NUMINAMATH_CALUDE_distance_per_interval_l3441_344128

-- Define the total distance walked
def total_distance : ℝ := 3

-- Define the total time taken
def total_time : ℝ := 45

-- Define the interval time
def interval_time : ℝ := 15

-- Theorem to prove
theorem distance_per_interval : 
  (total_distance / (total_time / interval_time)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_per_interval_l3441_344128


namespace NUMINAMATH_CALUDE_triangle_area_is_two_l3441_344158

/-- The area of the triangle bounded by the y-axis and two lines -/
def triangle_area : ℝ := 2

/-- The first line equation: y - 2x = 1 -/
def line1 (x y : ℝ) : Prop := y - 2 * x = 1

/-- The second line equation: 4y + x = 16 -/
def line2 (x y : ℝ) : Prop := 4 * y + x = 16

/-- The theorem stating that the area of the triangle is 2 -/
theorem triangle_area_is_two :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ = 0 ∧ line1 x₁ y₁ ∧
    x₂ = 0 ∧ line2 x₂ y₂ ∧
    triangle_area = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_two_l3441_344158


namespace NUMINAMATH_CALUDE_min_price_reduction_l3441_344188

theorem min_price_reduction (price_2004 : ℝ) (h1 : price_2004 > 0) : 
  let price_2005 := price_2004 * (1 - 0.15)
  let min_reduction := (price_2005 - price_2004 * 0.75) / price_2005 * 100
  ∀ ε > 0, ∃ δ > 0, 
    abs (min_reduction - 11.8) < δ ∧ 
    price_2004 * (1 - 0.15) * (1 - (min_reduction + ε) / 100) < price_2004 * 0.75 ∧
    price_2004 * (1 - 0.15) * (1 - (min_reduction - ε) / 100) > price_2004 * 0.75 :=
by sorry

end NUMINAMATH_CALUDE_min_price_reduction_l3441_344188


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3441_344167

/-- The solution set of the inequality 3 - 2x - x^2 < 0 -/
def solution_set : Set ℝ := {x | x < -3 ∨ x > 1}

/-- The inequality function -/
def f (x : ℝ) := 3 - 2*x - x^2

theorem inequality_solution_set :
  ∀ x : ℝ, f x < 0 ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3441_344167


namespace NUMINAMATH_CALUDE_josh_doug_money_ratio_l3441_344114

/-- Proves that the ratio of Josh's money to Doug's money is 3:4 given the problem conditions -/
theorem josh_doug_money_ratio :
  ∀ (josh doug brad : ℕ),
  josh + doug + brad = 68 →
  josh = 2 * brad →
  doug = 32 →
  (josh : ℚ) / doug = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_josh_doug_money_ratio_l3441_344114


namespace NUMINAMATH_CALUDE_geometric_sequence_operations_l3441_344110

-- Define a geometric sequence
def IsGeometric (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

-- Define the problem statement
theorem geometric_sequence_operations
  (a b : ℕ → ℝ)
  (ha : IsGeometric a)
  (hb : IsGeometric b)
  (hb_nonzero : ∀ n, b n ≠ 0) :
  IsGeometric (fun n ↦ a n * b n) ∧
  IsGeometric (fun n ↦ a n / b n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_operations_l3441_344110


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_tangent_to_circle_l3441_344121

/-- The value of m for which the asymptotes of the hyperbola y² - x²/m² = 1 
    are tangent to the circle x² + y² - 4y + 3 = 0, given m > 0 -/
theorem hyperbola_asymptote_tangent_to_circle (m : ℝ) 
  (hm : m > 0)
  (h_hyperbola : ∀ x y : ℝ, y^2 - x^2/m^2 = 1 → 
    (∃ k : ℝ, y = k*x/m ∨ y = -k*x/m))
  (h_circle : ∀ x y : ℝ, x^2 + y^2 - 4*y + 3 = 0 → 
    (x - 0)^2 + (y - 2)^2 = 1)
  (h_tangent : ∀ x y : ℝ, (y = x/m ∨ y = -x/m) → 
    ((0 - x)^2 + (2 - y)^2 = 1)) :
  m = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_tangent_to_circle_l3441_344121


namespace NUMINAMATH_CALUDE_cost_per_box_is_three_fifty_l3441_344163

/-- The cost per box of wafer cookies -/
def cost_per_box (num_trays : ℕ) (cookies_per_tray : ℕ) (cookies_per_box : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (((num_trays * cookies_per_tray) + cookies_per_box - 1) / cookies_per_box)

/-- Theorem stating that the cost per box is $3.50 given the problem conditions -/
theorem cost_per_box_is_three_fifty :
  cost_per_box 3 80 60 14 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_box_is_three_fifty_l3441_344163


namespace NUMINAMATH_CALUDE_deposit_percentage_l3441_344144

/-- Proves that the percentage P of the initial amount used in the deposit calculation is 30% --/
theorem deposit_percentage (initial_amount deposit_amount : ℝ) 
  (h1 : initial_amount = 50000)
  (h2 : deposit_amount = 750)
  (h3 : ∃ P : ℝ, deposit_amount = 0.20 * 0.25 * (P / 100) * initial_amount) :
  ∃ P : ℝ, P = 30 ∧ deposit_amount = 0.20 * 0.25 * (P / 100) * initial_amount :=
by sorry

end NUMINAMATH_CALUDE_deposit_percentage_l3441_344144


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3441_344123

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * time

theorem simple_interest_problem (P : ℝ) : 
  simple_interest P 0.08 3 = 
  (1/2) * compound_interest 4000 0.1 2 → P = 1750 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3441_344123


namespace NUMINAMATH_CALUDE_projection_of_A_on_Oxz_l3441_344131

/-- The projection of a point (x, y, z) onto the Oxz plane is (x, 0, z) -/
def proj_oxz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, 0, p.2.2)

/-- Point A in 3D space -/
def A : ℝ × ℝ × ℝ := (2, 3, 6)

/-- Point B is the projection of A onto the Oxz plane -/
def B : ℝ × ℝ × ℝ := proj_oxz A

theorem projection_of_A_on_Oxz :
  B = (2, 0, 6) := by sorry

end NUMINAMATH_CALUDE_projection_of_A_on_Oxz_l3441_344131


namespace NUMINAMATH_CALUDE_intersection_range_l3441_344194

/-- The function f(x) = 3x - x^3 --/
def f (x : ℝ) : ℝ := 3*x - x^3

/-- The line y = m intersects the graph of f at three distinct points --/
def intersects_at_three_points (m : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m

theorem intersection_range (m : ℝ) :
  intersects_at_three_points m → -2 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l3441_344194


namespace NUMINAMATH_CALUDE_proportional_expression_l3441_344193

/-- Given that y is directly proportional to x-2 and y = -4 when x = 3,
    prove that the analytical expression of y with respect to x is y = -4x + 8 -/
theorem proportional_expression (x y : ℝ) :
  (∃ k : ℝ, ∀ x, y = k * (x - 2)) →  -- y is directly proportional to x-2
  (3 : ℝ) = x → (-4 : ℝ) = y →       -- when x = 3, y = -4
  y = -4 * x + 8 :=                   -- the analytical expression
by sorry

end NUMINAMATH_CALUDE_proportional_expression_l3441_344193


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3441_344197

theorem repeating_decimal_sum : 
  (1 : ℚ) / 3 + 7 / 99 + 1 / 111 = 499 / 1189 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3441_344197


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l3441_344170

theorem sum_of_four_numbers : 1256 + 2561 + 5612 + 6125 = 15554 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l3441_344170


namespace NUMINAMATH_CALUDE_oxford_high_school_population_is_349_l3441_344165

/-- The number of people in Oxford High School -/
def oxford_high_school_population : ℕ :=
  let teachers : ℕ := 48
  let principal : ℕ := 1
  let classes : ℕ := 15
  let students_per_class : ℕ := 20
  let total_students : ℕ := classes * students_per_class
  teachers + principal + total_students

/-- Theorem stating the total number of people in Oxford High School -/
theorem oxford_high_school_population_is_349 :
  oxford_high_school_population = 349 := by
  sorry

end NUMINAMATH_CALUDE_oxford_high_school_population_is_349_l3441_344165


namespace NUMINAMATH_CALUDE_dance_camp_rabbits_l3441_344152

theorem dance_camp_rabbits :
  ∀ (R S : ℕ),
  R + S = 50 →
  4 * R + 8 * S = 2 * R + 16 * S →
  R = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_dance_camp_rabbits_l3441_344152


namespace NUMINAMATH_CALUDE_remainder_sum_l3441_344118

theorem remainder_sum (n : ℤ) : n % 12 = 5 → (n % 3 + n % 4 = 3) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3441_344118


namespace NUMINAMATH_CALUDE_red_ball_probability_l3441_344186

theorem red_ball_probability (n : ℕ) (r : ℕ) (k : ℕ) (h1 : n = 10) (h2 : r = 3) (h3 : k = 3) :
  let total_balls := n
  let red_balls := r
  let last_children := k
  let prob_one_red := (last_children.choose 1 : ℚ) * (red_balls / total_balls) * ((total_balls - red_balls) / total_balls) ^ 2
  prob_one_red = 441 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_red_ball_probability_l3441_344186


namespace NUMINAMATH_CALUDE_lunch_cost_proof_l3441_344199

/-- The cost of Mike's additional items -/
def mike_additional : ℝ := 11.75

/-- The cost of John's additional items -/
def john_additional : ℝ := 5.25

/-- The ratio of Mike's bill to John's bill -/
def bill_ratio : ℝ := 1.5

/-- The combined total cost of Mike and John's lunch -/
def total_cost : ℝ := 58.75

theorem lunch_cost_proof :
  ∃ (taco_grande_price : ℝ),
    taco_grande_price > 0 ∧
    (taco_grande_price + mike_additional) = bill_ratio * (taco_grande_price + john_additional) ∧
    (taco_grande_price + mike_additional) + (taco_grande_price + john_additional) = total_cost := by
  sorry

end NUMINAMATH_CALUDE_lunch_cost_proof_l3441_344199


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l3441_344127

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 24) :
  let r := d / 4
  (4 / 3) * π * r^3 = 2304 * π := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l3441_344127


namespace NUMINAMATH_CALUDE_min_cost_to_buy_all_items_l3441_344168

def items : ℕ := 20

-- Define the set of prices
def prices : Finset ℕ := Finset.range items.succ

-- Define the promotion
def promotion_group_size : ℕ := 5
def free_items : ℕ := items / promotion_group_size

-- Define the minimum cost function
def min_cost : ℕ := (Finset.sum prices id) - (Finset.sum (Finset.filter (λ x => x > items - free_items) prices) id)

-- The theorem to prove
theorem min_cost_to_buy_all_items : min_cost = 136 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_to_buy_all_items_l3441_344168


namespace NUMINAMATH_CALUDE_invalid_prism_diagonals_l3441_344115

theorem invalid_prism_diagonals : ¬∃ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a^2 + b^2 = 5^2 ∨ a^2 + b^2 = 12^2 ∨ a^2 + b^2 = 13^2) ∧
  (b^2 + c^2 = 5^2 ∨ b^2 + c^2 = 12^2 ∨ b^2 + c^2 = 13^2) ∧
  (a^2 + c^2 = 5^2 ∨ a^2 + c^2 = 12^2 ∨ a^2 + c^2 = 13^2) ∧
  (a^2 + b^2 + c^2 = 14^2) :=
by sorry

end NUMINAMATH_CALUDE_invalid_prism_diagonals_l3441_344115


namespace NUMINAMATH_CALUDE_coffee_consumption_l3441_344126

theorem coffee_consumption (x : ℝ) : 
  x > 0 → -- Tom's coffee size is positive
  (2/3 * x + (5/48 * x + 3) = 5/4 * (2/3 * x) - (5/48 * x + 3)) → -- They drink the same amount
  x + 1.25 * x = 36 -- Total coffee consumed is 36 ounces
  := by sorry

end NUMINAMATH_CALUDE_coffee_consumption_l3441_344126


namespace NUMINAMATH_CALUDE_incorrect_survey_method_statement_l3441_344102

-- Define survey methods
inductive SurveyMethod
| Sampling
| Comprehensive

-- Define scenarios
inductive Scenario
| StudentInterests
| ParentWorkConditions
| PopulationCensus
| LakeWaterQuality

-- Define function to determine appropriate survey method
def appropriateSurveyMethod (scenario : Scenario) : SurveyMethod :=
  match scenario with
  | Scenario.StudentInterests => SurveyMethod.Sampling
  | Scenario.ParentWorkConditions => SurveyMethod.Comprehensive
  | Scenario.PopulationCensus => SurveyMethod.Comprehensive
  | Scenario.LakeWaterQuality => SurveyMethod.Sampling

-- Theorem to prove
theorem incorrect_survey_method_statement :
  appropriateSurveyMethod Scenario.ParentWorkConditions ≠ SurveyMethod.Sampling :=
by sorry

end NUMINAMATH_CALUDE_incorrect_survey_method_statement_l3441_344102


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3441_344192

theorem complex_number_in_second_quadrant : 
  let z : ℂ := 2 * I / (1 - I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3441_344192


namespace NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3441_344105

theorem log_equality_implies_golden_ratio (p q : ℝ) 
  (hp : p > 0) (hq : q > 0) 
  (h : Real.log p / Real.log 8 = Real.log q / Real.log 12 ∧ 
       Real.log p / Real.log 8 = Real.log (p - q) / Real.log 18) : 
  q / p = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3441_344105


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l3441_344107

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + 5 * b < 100) :
  ab * (100 - 4 * a - 5 * b) ≤ 50000 / 27 :=
by sorry

theorem max_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 4 * a + 5 * b < 100 ∧
  ab * (100 - 4 * a - 5 * b) > 50000 / 27 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l3441_344107


namespace NUMINAMATH_CALUDE_average_and_difference_l3441_344124

theorem average_and_difference (x : ℝ) : 
  (40 + x + 15) / 3 = 35 → |x - 40| = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l3441_344124


namespace NUMINAMATH_CALUDE_diagonals_to_sides_ratio_for_pentagon_l3441_344112

-- Define the number of diagonals function
def num_diagonals (n : ℕ) : ℚ := n * (n - 3) / 2

-- Theorem statement
theorem diagonals_to_sides_ratio_for_pentagon :
  let n : ℕ := 5
  (num_diagonals n) / n = 1 := by sorry

end NUMINAMATH_CALUDE_diagonals_to_sides_ratio_for_pentagon_l3441_344112


namespace NUMINAMATH_CALUDE_crayon_ratio_l3441_344172

def billies_crayons : ℕ := 18
def bobbies_crayons : ℕ := 3 * billies_crayons
def lizzies_crayons : ℕ := 27

theorem crayon_ratio : 
  (lizzies_crayons : ℚ) / (bobbies_crayons : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_crayon_ratio_l3441_344172
