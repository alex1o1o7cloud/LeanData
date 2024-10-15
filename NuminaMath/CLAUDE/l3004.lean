import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3004_300429

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 * a 19 = 16) →
  (a 1 + a 19 = 10) →
  a 8 * a 10 * a 12 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3004_300429


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l3004_300444

/-- Proves that tripling the height and increasing the radius by 150% results in a volume increase by a factor of 18.75 -/
theorem cylinder_volume_increase (r h : ℝ) (r_pos : 0 < r) (h_pos : 0 < h) : 
  let new_r := 2.5 * r
  let new_h := 3 * h
  π * new_r^2 * new_h = 18.75 * (π * r^2 * h) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_increase_l3004_300444


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l3004_300471

theorem complex_magnitude_equation (t : ℝ) : t > 2 ∧ 
  Complex.abs (t + 4 * Complex.I * Real.sqrt 3) * Complex.abs (7 - 2 * Complex.I) = 17 * Real.sqrt 13 ↔ 
  t = Real.sqrt (1213 / 53) := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l3004_300471


namespace NUMINAMATH_CALUDE_square_perimeter_equal_area_rectangle_l3004_300450

theorem square_perimeter_equal_area_rectangle (l w : ℝ) (h1 : l = 1024) (h2 : w = 1) :
  let rectangle_area := l * w
  let square_side := (rectangle_area).sqrt
  let square_perimeter := 4 * square_side
  square_perimeter = 128 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_equal_area_rectangle_l3004_300450


namespace NUMINAMATH_CALUDE_carries_profit_l3004_300424

/-- Calculates the profit for a cake maker after taxes and expenses -/
def cake_profit (hours_per_day : ℕ) (days_worked : ℕ) (hourly_rate : ℚ) 
                (supply_cost : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_hours := hours_per_day * days_worked
  let gross_earnings := hourly_rate * total_hours
  let tax_amount := gross_earnings * tax_rate
  let after_tax_earnings := gross_earnings - tax_amount
  after_tax_earnings - supply_cost

/-- Theorem stating that Carrie's profit is $631.20 given the problem conditions -/
theorem carries_profit :
  cake_profit 4 6 35 150 (7/100) = 631.2 := by
  sorry

end NUMINAMATH_CALUDE_carries_profit_l3004_300424


namespace NUMINAMATH_CALUDE_tall_mirror_passes_l3004_300467

/-- The number of times Sarah and Ellie passed through the room with tall mirrors -/
def T : ℕ := sorry

/-- Sarah's reflections in tall mirrors -/
def sarah_tall : ℕ := 10

/-- Sarah's reflections in wide mirrors -/
def sarah_wide : ℕ := 5

/-- Ellie's reflections in tall mirrors -/
def ellie_tall : ℕ := 6

/-- Ellie's reflections in wide mirrors -/
def ellie_wide : ℕ := 3

/-- Number of times they passed through the wide mirrors room -/
def wide_passes : ℕ := 5

/-- Total number of reflections seen by Sarah and Ellie -/
def total_reflections : ℕ := 88

theorem tall_mirror_passes :
  T * (sarah_tall + ellie_tall) + wide_passes * (sarah_wide + ellie_wide) = total_reflections ∧
  T = 3 := by sorry

end NUMINAMATH_CALUDE_tall_mirror_passes_l3004_300467


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l3004_300469

/-- Given two points A(-3, y₁) and B(2, y₂) on the graph of y = 6/x, prove that y₁ < y₂ -/
theorem inverse_proportion_y_relationship (y₁ y₂ : ℝ) : 
  y₁ = 6 / (-3) → y₂ = 6 / 2 → y₁ < y₂ := by
  sorry


end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l3004_300469


namespace NUMINAMATH_CALUDE_cameron_donation_ratio_l3004_300482

theorem cameron_donation_ratio :
  let boris_initial : ℕ := 24
  let boris_donation_fraction : ℚ := 1/4
  let cameron_initial : ℕ := 30
  let total_after_donation : ℕ := 38
  let boris_after := boris_initial - boris_initial * boris_donation_fraction
  let cameron_after := total_after_donation - boris_after
  let cameron_donated := cameron_initial - cameron_after
  cameron_donated / cameron_initial = 1/3 := by
sorry

end NUMINAMATH_CALUDE_cameron_donation_ratio_l3004_300482


namespace NUMINAMATH_CALUDE_inscribed_hexagon_radius_equation_l3004_300462

/-- A hexagon inscribed in a circle with specific side lengths -/
structure InscribedHexagon where
  r : ℝ  -- radius of the circumscribed circle
  side1 : ℝ  -- length of two sides
  side2 : ℝ  -- length of two other sides
  side3 : ℝ  -- length of the remaining two sides
  h1 : side1 = 1
  h2 : side2 = 2
  h3 : side3 = 3

/-- The radius of the circumscribed circle satisfies a specific equation -/
theorem inscribed_hexagon_radius_equation (h : InscribedHexagon) : 
  2 * h.r^3 - 7 * h.r - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_radius_equation_l3004_300462


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3004_300422

theorem solve_linear_equation (x : ℝ) (h : x + 1 = 2) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3004_300422


namespace NUMINAMATH_CALUDE_vectors_collinear_l3004_300415

/-- Two vectors in ℝ² are collinear if their cross product is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b in ℝ², prove they are collinear -/
theorem vectors_collinear :
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (1, -2)
  collinear a b := by
  sorry

end NUMINAMATH_CALUDE_vectors_collinear_l3004_300415


namespace NUMINAMATH_CALUDE_CH4_yield_is_zero_l3004_300439

-- Define the molecules and their amounts
structure Molecule :=
  (C : ℕ) (H : ℕ) (O : ℕ)

-- Define the reactions
def reaction_CH4 (m : Molecule) : Molecule :=
  {C := m.C - 1, H := m.H - 4, O := m.O}

def reaction_CO2 (m : Molecule) : Molecule :=
  {C := m.C - 1, H := m.H, O := m.O - 2}

def reaction_H2O (m : Molecule) : Molecule :=
  {C := m.C, H := m.H - 4, O := m.O - 2}

-- Define the initial amounts
def initial_amounts : Molecule :=
  {C := 3, H := 12, O := 8}  -- 3 moles C, 6 moles H2 (12 H atoms), 4 moles O2 (8 O atoms)

-- Define the theoretical yield of CH4
def theoretical_yield_CH4 (m : Molecule) : ℕ :=
  min m.C (m.H / 4)

-- Theorem statement
theorem CH4_yield_is_zero :
  theoretical_yield_CH4 (reaction_H2O (reaction_CO2 initial_amounts)) = 0 :=
sorry

end NUMINAMATH_CALUDE_CH4_yield_is_zero_l3004_300439


namespace NUMINAMATH_CALUDE_product_and_remainder_problem_l3004_300487

theorem product_and_remainder_problem :
  ∃ (a b c d : ℤ),
    d = a * b * c ∧
    1 < a ∧ a < b ∧ b < c ∧
    233 % d = 79 ∧
    a + c = 13 := by
  sorry

end NUMINAMATH_CALUDE_product_and_remainder_problem_l3004_300487


namespace NUMINAMATH_CALUDE_stripe_area_on_cylinder_l3004_300413

/-- The area of a stripe on a cylindrical silo -/
theorem stripe_area_on_cylinder 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h1 : diameter = 40)
  (h2 : stripe_width = 2)
  (h3 : revolutions = 3) :
  stripe_width * revolutions * π * diameter = 240 * π := by
sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylinder_l3004_300413


namespace NUMINAMATH_CALUDE_expression_value_l3004_300478

theorem expression_value (a b : ℝ) (h : 3 * (a - 2) = 2 * (2 * b - 4)) :
  9 * a^2 - 24 * a * b + 16 * b^2 + 25 = 29 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3004_300478


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3004_300426

theorem min_sum_of_squares (a b c d : ℝ) (h : a + 3*b + 5*c + 7*d = 14) :
  a^2 + b^2 + c^2 + d^2 ≥ 7/3 ∧
  (a^2 + b^2 + c^2 + d^2 = 7/3 ↔ a = 1/6 ∧ b = 1/2 ∧ c = 5/6 ∧ d = 7/6) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3004_300426


namespace NUMINAMATH_CALUDE_units_digit_sum_powers_l3004_300486

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of raising a number to a power, considering only the units digit -/
def powerUnitsDigit (base : ℕ) (exp : ℕ) : ℕ :=
  (unitsDigit base ^ exp) % 10

theorem units_digit_sum_powers : unitsDigit (powerUnitsDigit 53 107 + powerUnitsDigit 97 59) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_powers_l3004_300486


namespace NUMINAMATH_CALUDE_digit_79_is_2_l3004_300470

/-- The sequence of digits formed by writing consecutive integers from 65 to 1 in descending order -/
def descending_sequence : List Nat := sorry

/-- The 79th digit in the descending sequence -/
def digit_79 : Nat := sorry

/-- Theorem stating that the 79th digit in the descending sequence is 2 -/
theorem digit_79_is_2 : digit_79 = 2 := by sorry

end NUMINAMATH_CALUDE_digit_79_is_2_l3004_300470


namespace NUMINAMATH_CALUDE_vector_properties_l3004_300412

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_properties :
  (∀ (m : ℝ) (a b : V), m • (a - b) = m • a - m • b) ∧
  (∀ (m n : ℝ) (a : V), (m - n) • a = m • a - n • a) ∧
  (∃ (m : ℝ) (a b : V), m • a = m • b ∧ a ≠ b) ∧
  (∀ (m n : ℝ) (a : V), a ≠ 0 → m • a = n • a → m = n) :=
sorry

end NUMINAMATH_CALUDE_vector_properties_l3004_300412


namespace NUMINAMATH_CALUDE_ellipse_theorem_l3004_300410

/-- Ellipse E with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  right_focus_dist : ℝ → ℝ → ℝ
  h_focus : right_focus_dist 1 (-1) = 3
  h_point : -1^2 / a^2 + (-Real.sqrt 6 / 2)^2 / b^2 = 1

/-- Line l intersecting the ellipse -/
def Line (m t : ℝ) (x y : ℝ) : Prop :=
  x - m * y - t = 0

/-- Statement of the theorem -/
theorem ellipse_theorem (E : Ellipse) :
  E.a^2 = 4 ∧ E.b^2 = 2 ∧
  ∀ m t, ∃ M N : ℝ × ℝ,
    M ≠ N ∧
    M ≠ (-E.a, 0) ∧ N ≠ (-E.a, 0) ∧
    Line m t M.1 M.2 ∧ Line m t N.1 N.2 ∧
    M.1^2 / 4 + M.2^2 / 2 = 1 ∧
    N.1^2 / 4 + N.2^2 / 2 = 1 ∧
    ((M.1 + E.a)^2 + M.2^2) * ((N.1 + E.a)^2 + N.2^2) =
      ((M.1 - N.1)^2 + (M.2 - N.2)^2) * ((M.1 + N.1 + 2*E.a)^2 + (M.2 + N.2)^2) / 4 →
    t = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l3004_300410


namespace NUMINAMATH_CALUDE_extremum_and_tangent_imply_max_min_difference_l3004_300417

def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

theorem extremum_and_tangent_imply_max_min_difference
  (a b c : ℝ) :
  (∃ x, deriv (f a b c) x = 0 ∧ x = 2) →
  (deriv (f a b c) 1 = -3) →
  ∃ max min : ℝ, 
    (∀ x, f a b c x ≤ max) ∧
    (∀ x, f a b c x ≥ min) ∧
    (max - min = 4) :=
sorry

end NUMINAMATH_CALUDE_extremum_and_tangent_imply_max_min_difference_l3004_300417


namespace NUMINAMATH_CALUDE_power_of_product_l3004_300430

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3004_300430


namespace NUMINAMATH_CALUDE_total_dumbbell_weight_l3004_300411

/-- Represents the weight of a single dumbbell in a pair --/
def dumbbell_weights : List ℕ := [3, 5, 8, 12, 18, 27]

/-- Theorem: The total weight of the dumbbell system is 146 lb --/
theorem total_dumbbell_weight : 
  (dumbbell_weights.map (·*2)).sum = 146 := by sorry

end NUMINAMATH_CALUDE_total_dumbbell_weight_l3004_300411


namespace NUMINAMATH_CALUDE_ellipse_foci_product_range_l3004_300456

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the foci
def leftFocus : ℝ × ℝ := sorry
def rightFocus : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_foci_product_range (p : ℝ × ℝ) :
  ellipse p.1 p.2 →
  3 ≤ (distance p leftFocus) * (distance p rightFocus) ∧
  (distance p leftFocus) * (distance p rightFocus) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_product_range_l3004_300456


namespace NUMINAMATH_CALUDE_max_square_pen_area_l3004_300401

/-- Given 36 feet of fencing, the maximum area of a square pen is 81 square feet. -/
theorem max_square_pen_area (fencing : ℝ) (h : fencing = 36) : 
  (fencing / 4) ^ 2 = 81 :=
by sorry

end NUMINAMATH_CALUDE_max_square_pen_area_l3004_300401


namespace NUMINAMATH_CALUDE_apple_cost_is_75_cents_l3004_300427

/-- The cost of an apple given the amount paid and change received -/
def appleCost (amountPaid change : ℚ) : ℚ :=
  amountPaid - change

/-- Proof that the apple costs $0.75 given the conditions -/
theorem apple_cost_is_75_cents (amountPaid change : ℚ) 
  (h1 : amountPaid = 5)
  (h2 : change = 4.25) : 
  appleCost amountPaid change = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_is_75_cents_l3004_300427


namespace NUMINAMATH_CALUDE_projection_theorem_l3004_300445

def vector_a : Fin 2 → ℝ := ![2, 3]
def vector_b : Fin 2 → ℝ := ![-1, 2]

theorem projection_theorem :
  let dot_product := (vector_a 0) * (vector_b 0) + (vector_a 1) * (vector_b 1)
  let magnitude_b := Real.sqrt ((vector_b 0)^2 + (vector_b 1)^2)
  dot_product / magnitude_b = 4 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_projection_theorem_l3004_300445


namespace NUMINAMATH_CALUDE_modulo_congruence_unique_solution_l3004_300484

theorem modulo_congruence_unique_solution : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 27514 [MOD 16] ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_modulo_congruence_unique_solution_l3004_300484


namespace NUMINAMATH_CALUDE_jake_not_drop_coffee_l3004_300489

/-- The probability of Jake tripping over his dog on any given morning. -/
def prob_trip : ℝ := 0.40

/-- The probability of Jake dropping his coffee when he trips over his dog. -/
def prob_drop_when_trip : ℝ := 0.25

/-- The probability of Jake missing a step on the stairs on any given morning. -/
def prob_miss_step : ℝ := 0.30

/-- The probability of Jake spilling his coffee when he misses a step. -/
def prob_spill_when_miss : ℝ := 0.20

/-- Theorem: The probability of Jake not dropping his coffee on any given morning is 0.846. -/
theorem jake_not_drop_coffee :
  (1 - prob_trip * prob_drop_when_trip) * (1 - prob_miss_step * prob_spill_when_miss) = 0.846 := by
  sorry

end NUMINAMATH_CALUDE_jake_not_drop_coffee_l3004_300489


namespace NUMINAMATH_CALUDE_total_wheels_is_25_l3004_300402

/-- Calculates the total number of wheels in Jordan's driveway -/
def total_wheels : ℕ :=
  let num_cars : ℕ := 2
  let wheels_per_car : ℕ := 4
  let num_bikes : ℕ := 2
  let wheels_per_bike : ℕ := 2
  let num_trash_cans : ℕ := 1
  let wheels_per_trash_can : ℕ := 2
  let num_tricycles : ℕ := 1
  let wheels_per_tricycle : ℕ := 3
  let num_roller_skate_pairs : ℕ := 1
  let wheels_per_roller_skate : ℕ := 4
  let wheels_per_roller_skate_pair : ℕ := 2 * wheels_per_roller_skate

  num_cars * wheels_per_car +
  num_bikes * wheels_per_bike +
  num_trash_cans * wheels_per_trash_can +
  num_tricycles * wheels_per_tricycle +
  num_roller_skate_pairs * wheels_per_roller_skate_pair

theorem total_wheels_is_25 : total_wheels = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_25_l3004_300402


namespace NUMINAMATH_CALUDE_problem_statement_l3004_300485

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- State the theorem
theorem problem_statement (m : ℝ) (a b c : ℝ) :
  (∀ x, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-1) 1) →
  a > 0 → b > 0 → c > 0 →
  1 / a + 1 / (2 * b) + 1 / (3 * c) = m →
  (m = 1 ∧ a + 2 * b + 3 * c ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3004_300485


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l3004_300418

/-- A function that returns the tens digit of a natural number -/
def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- A function that returns the ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ :=
  n % 10

/-- A predicate that checks if a natural number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

/-- A predicate that checks if a natural number is even -/
def is_even (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * k

/-- A predicate that checks if a natural number is a multiple of 9 -/
def is_multiple_of_9 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 9 * k

/-- A predicate that checks if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem unique_two_digit_number : 
  ∀ n : ℕ, 
    is_two_digit n ∧ 
    is_even n ∧ 
    is_multiple_of_9 n ∧ 
    is_perfect_square (tens_digit n * ones_digit n) → 
    n = 90 :=
sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l3004_300418


namespace NUMINAMATH_CALUDE_basketball_surface_area_l3004_300492

/-- The surface area of a sphere with circumference 30 inches is 900/π square inches -/
theorem basketball_surface_area :
  let circumference : ℝ := 30
  let radius : ℝ := circumference / (2 * Real.pi)
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 900 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_basketball_surface_area_l3004_300492


namespace NUMINAMATH_CALUDE_Z_in_third_quadrant_implies_a_range_l3004_300437

def Z (a : ℝ) : ℂ := Complex.mk (a^2 - 2*a) (a^2 - a - 2)

def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

theorem Z_in_third_quadrant_implies_a_range (a : ℝ) :
  in_third_quadrant (Z a) → 0 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_Z_in_third_quadrant_implies_a_range_l3004_300437


namespace NUMINAMATH_CALUDE_even_integer_solution_l3004_300443

-- Define the function h for even integers
def h (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n ≥ 2 then
    (n / 2) * (2 + n) / 2
  else
    0

-- Theorem statement
theorem even_integer_solution :
  ∃ x : ℕ, x % 2 = 0 ∧ x ≥ 2 ∧ h 18 / h x = 3 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_even_integer_solution_l3004_300443


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l3004_300404

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, m < n →
    ¬((m ≥ 10000 ∧ m < 100000) ∧
      (∃ x : ℕ, m = x^2) ∧
      (∃ y : ℕ, m = y^3))) ∧
  n = 15625 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l3004_300404


namespace NUMINAMATH_CALUDE_probability_two_common_books_is_36_105_l3004_300468

def total_books : ℕ := 12
def books_to_choose : ℕ := 4

def probability_two_common_books : ℚ :=
  (Nat.choose total_books 2 * Nat.choose (total_books - 2) 2 * Nat.choose (total_books - 4) 2) /
  (Nat.choose total_books books_to_choose * Nat.choose total_books books_to_choose)

theorem probability_two_common_books_is_36_105 :
  probability_two_common_books = 36 / 105 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_common_books_is_36_105_l3004_300468


namespace NUMINAMATH_CALUDE_limit_one_minus_x_squared_over_sin_pi_x_l3004_300494

/-- The limit of (1 - x^2) / sin(πx) as x approaches 1 is 2/π -/
theorem limit_one_minus_x_squared_over_sin_pi_x (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |(1 - x^2) / Real.sin (π * x) - 2/π| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_one_minus_x_squared_over_sin_pi_x_l3004_300494


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l3004_300472

theorem intersection_complement_equals_set (U A B : Set Nat) : 
  U = {1, 2, 3, 4, 5, 6, 7} →
  A = {2, 3, 4, 5} →
  B = {2, 3, 6, 7} →
  B ∩ (U \ A) = {6, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l3004_300472


namespace NUMINAMATH_CALUDE_cylinder_section_volume_l3004_300451

/-- Represents a cylinder -/
structure Cylinder where
  base_area : ℝ
  height : ℝ

/-- Represents a plane cutting the cylinder -/
structure CuttingPlane where
  not_parallel_to_base : Bool
  not_intersect_base : Bool

/-- Represents the section of the cylinder cut by the plane -/
structure CylinderSection where
  cylinder : Cylinder
  cutting_plane : CuttingPlane
  segment_height : ℝ

/-- The volume of a cylinder section -/
def section_volume (s : CylinderSection) : ℝ :=
  s.cylinder.base_area * s.segment_height

theorem cylinder_section_volume 
  (s : CylinderSection) 
  (h1 : s.cutting_plane.not_parallel_to_base = true) 
  (h2 : s.cutting_plane.not_intersect_base = true) : 
  ∃ (v : ℝ), v = section_volume s :=
sorry

end NUMINAMATH_CALUDE_cylinder_section_volume_l3004_300451


namespace NUMINAMATH_CALUDE_prime_pairs_congruence_l3004_300446

theorem prime_pairs_congruence (p : ℕ) (hp : Nat.Prime p) : 
  (∃ S : Finset (ℕ × ℕ), S.card = p ∧ 
    (∀ (x y : ℕ), (x, y) ∈ S ↔ 
      (x ≤ p ∧ y ≤ p ∧ (y^2 : ZMod p) = (x^3 - x : ZMod p))))
  ↔ (p = 2 ∨ p % 4 = 3) :=
sorry

end NUMINAMATH_CALUDE_prime_pairs_congruence_l3004_300446


namespace NUMINAMATH_CALUDE_four_point_triangles_l3004_300421

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A set of four points in a plane -/
structure FourPoints :=
  (a b c d : Point)

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Predicate to check if no three points in a set of four points are collinear -/
def no_three_collinear (points : FourPoints) : Prop :=
  ¬(collinear points.a points.b points.c) ∧
  ¬(collinear points.a points.b points.d) ∧
  ¬(collinear points.a points.c points.d) ∧
  ¬(collinear points.b points.c points.d)

/-- The number of distinct triangles that can be formed from four points -/
def num_triangles (points : FourPoints) : ℕ := sorry

/-- Theorem: Given four points on a plane where no three points are collinear,
    the number of distinct triangles that can be formed is 4 -/
theorem four_point_triangles (points : FourPoints) 
  (h : no_three_collinear points) : num_triangles points = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_point_triangles_l3004_300421


namespace NUMINAMATH_CALUDE_odd_function_derivative_l3004_300458

theorem odd_function_derivative (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →
  (∀ x, HasDerivAt f (g x) x) →
  ∀ x, g (-x) = -g x := by
sorry

end NUMINAMATH_CALUDE_odd_function_derivative_l3004_300458


namespace NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l3004_300447

/-- Given a mixture of zinc and copper in the ratio 9:11, where 27 kg of zinc is used,
    the total weight of the mixture is 60 kg. -/
theorem zinc_copper_mixture_weight : 
  ∀ (zinc copper total : ℝ),
  zinc = 27 →
  zinc / copper = 9 / 11 →
  total = zinc + copper →
  total = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l3004_300447


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l3004_300452

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (loss_margin : ℕ) 
  (candidate_percentage : ℚ) :
  total_votes = 2000 →
  loss_margin = 640 →
  candidate_percentage * total_votes + (candidate_percentage * total_votes + loss_margin) = total_votes →
  candidate_percentage = 34 / 100 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l3004_300452


namespace NUMINAMATH_CALUDE_muffin_boxes_l3004_300442

theorem muffin_boxes (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) : 
  total_muffins = 95 →
  muffins_per_box = 5 →
  available_boxes = 10 →
  (total_muffins - available_boxes * muffins_per_box + muffins_per_box - 1) / muffins_per_box = 9 :=
by sorry

end NUMINAMATH_CALUDE_muffin_boxes_l3004_300442


namespace NUMINAMATH_CALUDE_wheel_radius_increase_l3004_300463

/-- Calculates the increase in wheel radius given original and new odometer readings -/
theorem wheel_radius_increase
  (original_radius : ℝ)
  (original_reading : ℝ)
  (new_reading : ℝ)
  (inches_per_mile : ℝ)
  (h1 : original_radius = 16)
  (h2 : original_reading = 1000)
  (h3 : new_reading = 980)
  (h4 : inches_per_mile = 62560) :
  ∃ (increase : ℝ), abs (increase - 0.33) < 0.005 :=
by sorry

end NUMINAMATH_CALUDE_wheel_radius_increase_l3004_300463


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3004_300459

theorem imaginary_part_of_complex_fraction :
  Complex.im ((5 : ℂ) + Complex.I) / ((1 : ℂ) + Complex.I) = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3004_300459


namespace NUMINAMATH_CALUDE_nonAttackingRooksPlacementCount_l3004_300466

/-- The size of the chessboard -/
def boardSize : Nat := 8

/-- The total number of squares on the chessboard -/
def totalSquares : Nat := boardSize * boardSize

/-- The number of squares a rook attacks in its row and column, excluding itself -/
def attackedSquares : Nat := 2 * (boardSize - 1)

/-- The number of ways to place two rooks on a chessboard so they don't attack each other -/
def nonAttackingRooksPlacement : Nat := totalSquares * (totalSquares - 1 - attackedSquares)

theorem nonAttackingRooksPlacementCount : nonAttackingRooksPlacement = 3136 := by
  sorry

end NUMINAMATH_CALUDE_nonAttackingRooksPlacementCount_l3004_300466


namespace NUMINAMATH_CALUDE_evaluate_expression_l3004_300483

theorem evaluate_expression : 9^6 * 3^4 / 27^5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3004_300483


namespace NUMINAMATH_CALUDE_intended_number_is_five_l3004_300436

theorem intended_number_is_five : ∃! x : ℚ, (((3 * x * 10 + 2) / 19) + 7) = 3 * x := by
  sorry

end NUMINAMATH_CALUDE_intended_number_is_five_l3004_300436


namespace NUMINAMATH_CALUDE_can_transport_machines_l3004_300420

/-- Given three machines with masses in kg and a truck's capacity in kg,
    prove that the truck can transport all machines at once. -/
theorem can_transport_machines (m1 m2 m3 truck_capacity : ℕ) 
  (h1 : m1 = 800)
  (h2 : m2 = 500)
  (h3 : m3 = 600)
  (h4 : truck_capacity = 2000) :
  m1 + m2 + m3 ≤ truck_capacity := by
  sorry

#check can_transport_machines

end NUMINAMATH_CALUDE_can_transport_machines_l3004_300420


namespace NUMINAMATH_CALUDE_expression_in_terms_of_k_l3004_300428

theorem expression_in_terms_of_k (x y k : ℝ) (h : x ≠ y) 
  (hk : (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = k) :
  (x^8 + y^8) / (x^8 - y^8) - (x^8 - y^8) / (x^8 + y^8) = 
    (k - 2)^2 * (k + 2)^2 / (4 * k * (k^2 + 4)) :=
by sorry

end NUMINAMATH_CALUDE_expression_in_terms_of_k_l3004_300428


namespace NUMINAMATH_CALUDE_intercepts_count_l3004_300407

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 3*x - 2

-- Define x-intercepts
def is_x_intercept (x : ℝ) : Prop := f x = 0

-- Define y-intercepts
def is_y_intercept (y : ℝ) : Prop := ∃ x, f x = y ∧ x = 0

-- Theorem statement
theorem intercepts_count :
  (∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, is_x_intercept x) ∧
  (∃! y, is_y_intercept y) :=
sorry

end NUMINAMATH_CALUDE_intercepts_count_l3004_300407


namespace NUMINAMATH_CALUDE_corveus_sleep_deficit_l3004_300414

/-- Calculates the total sleep deficit for Corveus in a week --/
def corveusWeeklySleepDeficit : ℤ :=
  let weekdaySleep : ℤ := 5 * 5  -- 4 hours night sleep + 1 hour nap, for 5 days
  let weekendSleep : ℤ := 5 * 2  -- 5 hours night sleep for 2 days
  let daylightSavingAdjustment : ℤ := 1  -- Extra hour due to daylight saving
  let midnightAwakenings : ℤ := 2  -- Loses 1 hour twice a week
  let actualSleep : ℤ := weekdaySleep + weekendSleep + daylightSavingAdjustment - midnightAwakenings
  let recommendedSleep : ℤ := 6 * 7  -- 6 hours per day for 7 days
  recommendedSleep - actualSleep

/-- Theorem stating that Corveus's weekly sleep deficit is 8 hours --/
theorem corveus_sleep_deficit : corveusWeeklySleepDeficit = 8 := by
  sorry

end NUMINAMATH_CALUDE_corveus_sleep_deficit_l3004_300414


namespace NUMINAMATH_CALUDE_goldbach_nine_l3004_300481

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

-- Define the theorem
theorem goldbach_nine : 
  ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧ p + q + r = 9 :=
sorry

end NUMINAMATH_CALUDE_goldbach_nine_l3004_300481


namespace NUMINAMATH_CALUDE_employee_age_when_hired_l3004_300497

/-- Represents the retirement eligibility rule where age plus years of employment must equal 70 -/
def retirement_rule (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed = 70

/-- Represents the fact that the employee worked for 19 years before retirement eligibility -/
def years_worked : ℕ := 19

/-- Proves that the employee's age when hired was 51 -/
theorem employee_age_when_hired :
  ∃ (age_when_hired : ℕ),
    retirement_rule (age_when_hired + years_worked) years_worked ∧
    age_when_hired = 51 := by
  sorry

end NUMINAMATH_CALUDE_employee_age_when_hired_l3004_300497


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3004_300403

theorem inequality_system_solution_set :
  ∀ x : ℝ, (2 * x ≤ -1 ∧ x > -1) ↔ (-1 < x ∧ x ≤ -1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3004_300403


namespace NUMINAMATH_CALUDE_base5_product_l3004_300476

/-- Converts a list of digits in base 5 to a natural number -/
def fromBase5 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 5 * acc + d) 0

/-- Converts a natural number to a list of digits in base 5 -/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The main theorem stating that the product of 1324₅ and 23₅ in base 5 is 42112₅ -/
theorem base5_product :
  toBase5 (fromBase5 [1, 3, 2, 4] * fromBase5 [2, 3]) = [4, 2, 1, 1, 2] := by
  sorry

end NUMINAMATH_CALUDE_base5_product_l3004_300476


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3004_300490

/-- An arithmetic sequence with integer common ratio -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  q : ℤ
  seq_def : ∀ n, a (n + 1) = a n + q

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a 1 * n + seq.q * (n * (n - 1) / 2)

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h1 : seq.a 2 - seq.a 3 = -2)
  (h2 : seq.a 1 + seq.a 3 = 10/3) :
  sum_n seq 4 = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3004_300490


namespace NUMINAMATH_CALUDE_square_gt_when_abs_lt_l3004_300409

theorem square_gt_when_abs_lt (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_gt_when_abs_lt_l3004_300409


namespace NUMINAMATH_CALUDE_percentage_problem_l3004_300455

theorem percentage_problem (x : ℝ) : (0.15 * 0.30 * 0.50 * x = 90) → x = 4000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3004_300455


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l3004_300498

-- Define the hexagon and its angles
structure ConvexHexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

-- Define the theorem
theorem hexagon_angle_measure (h : ConvexHexagon) :
  h.A = h.B ∧ h.B = h.C ∧  -- A, B, C are congruent
  h.D = h.E ∧ h.E = h.F ∧  -- D, E, F are congruent
  h.A + 20 = h.D ∧         -- A is 20° less than D
  h.A + h.B + h.C + h.D + h.E + h.F = 720 -- Sum of angles in a hexagon
  →
  h.D = 130 := by sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l3004_300498


namespace NUMINAMATH_CALUDE_diamond_calculation_l3004_300440

-- Define the diamond operation
def diamond (X Y : ℚ) : ℚ := (2 * X + 3 * Y) / 5

-- Theorem statement
theorem diamond_calculation : diamond (diamond 3 15) 6 = 192 / 25 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l3004_300440


namespace NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l3004_300454

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (a = 3) ∧ 
    (∀ (x : ℂ), a * x^2 + b * x + c = 0 ↔ x = 4 + 2*I ∨ x = 4 - 2*I) ∧
    (a * (4 + 2*I)^2 + b * (4 + 2*I) + c = 0) ∧
    (a = 3 ∧ b = -24 ∧ c = 36) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l3004_300454


namespace NUMINAMATH_CALUDE_arrangement_count_l3004_300419

theorem arrangement_count :
  let teachers : ℕ := 3
  let students : ℕ := 6
  let groups : ℕ := 3
  let teachers_per_group : ℕ := 1
  let students_per_group : ℕ := 2
  
  (teachers.factorial * (students.choose students_per_group) * 
   ((students - students_per_group).choose students_per_group) * 
   ((students - 2 * students_per_group).choose students_per_group)) = 540 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l3004_300419


namespace NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_range_of_a_l3004_300448

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem 1: A ∪ B = {x | -1 ≤ x}
theorem union_A_B : A ∪ B = {x : ℝ | -1 ≤ x} := by sorry

-- Theorem 2: A ∩ (Cᴿ B) = {x | -1 ≤ x < 2}
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

-- Theorem 3: If B ∪ C = C, then a ≤ 3
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_range_of_a_l3004_300448


namespace NUMINAMATH_CALUDE_cube_difference_implies_sum_of_squares_l3004_300434

theorem cube_difference_implies_sum_of_squares (n : ℕ) (hn : n > 0) :
  (∃ x : ℕ, x > 0 ∧ (x + 1)^3 - x^3 = n^2) →
  ∃ a b : ℕ, n = a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_implies_sum_of_squares_l3004_300434


namespace NUMINAMATH_CALUDE_rectangle_area_l3004_300493

/-- A rectangle with specific properties -/
structure Rectangle where
  width : ℝ
  length : ℝ
  length_eq : length = 3 * width + 15
  perimeter_eq : 2 * (width + length) = 800

/-- The area of a rectangle with the given properties is 29234.375 square feet -/
theorem rectangle_area (rect : Rectangle) : rect.width * rect.length = 29234.375 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3004_300493


namespace NUMINAMATH_CALUDE_triathlete_swimming_speed_l3004_300495

/-- Calculates the swimming speed of a triathlete given the conditions of the problem -/
theorem triathlete_swimming_speed
  (distance : ℝ)
  (running_speed : ℝ)
  (average_rate : ℝ)
  (h1 : distance = 2)
  (h2 : running_speed = 10)
  (h3 : average_rate = 0.1111111111111111)
  : ∃ (swimming_speed : ℝ), swimming_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_triathlete_swimming_speed_l3004_300495


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3004_300457

theorem inequality_solution_set (x : ℝ) : 
  (6 * x + 2 < (x + 2)^2 ∧ (x + 2)^2 < 8 * x + 4) ↔ (2 + Real.sqrt 2 < x ∧ x < 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3004_300457


namespace NUMINAMATH_CALUDE_remainder_theorem_l3004_300441

theorem remainder_theorem (y : ℤ) 
  (h1 : (2 + y) % (3^3) = 3^2 % (3^3))
  (h2 : (4 + y) % (5^3) = 2^3 % (5^3))
  (h3 : (6 + y) % (7^3) = 7^2 % (7^3)) :
  y % 105 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3004_300441


namespace NUMINAMATH_CALUDE_at_most_one_prime_between_factorial_and_factorial_plus_n_plus_one_l3004_300435

theorem at_most_one_prime_between_factorial_and_factorial_plus_n_plus_one (n : ℕ) (hn : n > 1) :
  ∃! p : ℕ, Prime p ∧ n! < p ∧ p < n! + n + 1 :=
by sorry

end NUMINAMATH_CALUDE_at_most_one_prime_between_factorial_and_factorial_plus_n_plus_one_l3004_300435


namespace NUMINAMATH_CALUDE_f_range_implies_a_value_l3004_300405

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then
    -x + 3
  else if 2 ≤ x ∧ x ≤ 8 then
    1 + Real.log (2 * x) / Real.log (a^2 - 1)
  else
    0  -- undefined for other x values

theorem f_range_implies_a_value (a : ℝ) :
  (∀ y ∈ Set.range (f a), 2 ≤ y ∧ y ≤ 5) →
  (a = Real.sqrt 3 ∨ a = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_f_range_implies_a_value_l3004_300405


namespace NUMINAMATH_CALUDE_dartboard_region_angle_l3004_300473

/-- Given a circular dartboard with a region where the probability of a dart landing is 1/4,
    prove that the central angle of this region is 90°. -/
theorem dartboard_region_angle (probability : ℝ) (angle : ℝ) :
  probability = 1/4 →
  angle = probability * 360 →
  angle = 90 :=
by sorry

end NUMINAMATH_CALUDE_dartboard_region_angle_l3004_300473


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_negative_one_l3004_300453

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_negative_one :
  log10 (5/2) + 2 * log10 2 - (1/2)⁻¹ = -1 :=
by
  -- Assume the given condition
  have h : log10 2 + log10 5 = 1 := by sorry
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_negative_one_l3004_300453


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_l3004_300425

def A : ℕ := Nat.gcd 9 (Nat.gcd 12 18)
def B : ℕ := Nat.lcm 9 (Nat.lcm 12 18)

theorem gcf_lcm_sum : A + B = 39 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_l3004_300425


namespace NUMINAMATH_CALUDE_february_greatest_difference_l3004_300474

/-- Sales data for trumpet and trombone players -/
structure SalesData where
  trumpet : ℕ
  trombone : ℕ

/-- Calculate percentage difference between two numbers -/
def percentDifference (a b : ℕ) : ℚ :=
  (max a b - min a b : ℚ) / (min a b : ℚ) * 100

/-- Months of the year -/
inductive Month
  | Jan | Feb | Mar | Apr | May

/-- Sales data for each month -/
def monthlySales : Month → SalesData
  | Month.Jan => ⟨6, 4⟩
  | Month.Feb => ⟨27, 5⟩  -- Trumpet sales tripled
  | Month.Mar => ⟨8, 5⟩
  | Month.Apr => ⟨7, 8⟩
  | Month.May => ⟨5, 6⟩

/-- February has the greatest percent difference in sales -/
theorem february_greatest_difference :
  ∀ m : Month, m ≠ Month.Feb →
    percentDifference (monthlySales Month.Feb).trumpet (monthlySales Month.Feb).trombone >
    percentDifference (monthlySales m).trumpet (monthlySales m).trombone :=
by sorry

end NUMINAMATH_CALUDE_february_greatest_difference_l3004_300474


namespace NUMINAMATH_CALUDE_students_left_is_30_percent_l3004_300431

/-- The percentage of students left in a classroom after some students leave for activities -/
def students_left_percentage (total : ℕ) (painting : ℚ) (playing : ℚ) (workshop : ℚ) : ℚ :=
  (1 - (painting + playing + workshop)) * 100

/-- Theorem: Given the conditions, the percentage of students left in the classroom is 30% -/
theorem students_left_is_30_percent :
  students_left_percentage 250 (3/10) (2/10) (1/5) = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_left_is_30_percent_l3004_300431


namespace NUMINAMATH_CALUDE_b_payment_l3004_300464

/-- Calculate the amount b should pay for renting a pasture -/
theorem b_payment (total_rent : ℕ) 
  (a_horses a_months a_rate : ℕ) 
  (b_horses b_months b_rate : ℕ)
  (c_horses c_months c_rate : ℕ)
  (d_horses d_months d_rate : ℕ)
  (h_total_rent : total_rent = 725)
  (h_a : a_horses = 12 ∧ a_months = 8 ∧ a_rate = 5)
  (h_b : b_horses = 16 ∧ b_months = 9 ∧ b_rate = 6)
  (h_c : c_horses = 18 ∧ c_months = 6 ∧ c_rate = 7)
  (h_d : d_horses = 20 ∧ d_months = 4 ∧ d_rate = 4) :
  ∃ (b_payment : ℕ), b_payment = 259 ∧ 
  b_payment = round ((b_horses * b_months * b_rate : ℚ) / 
    ((a_horses * a_months * a_rate + b_horses * b_months * b_rate + 
      c_horses * c_months * c_rate + d_horses * d_months * d_rate) : ℚ) * total_rent) :=
by sorry

#check b_payment

end NUMINAMATH_CALUDE_b_payment_l3004_300464


namespace NUMINAMATH_CALUDE_expression_value_l3004_300499

theorem expression_value : 
  let x : ℕ := 2
  2 + 2 * (2 * 2) = 10 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3004_300499


namespace NUMINAMATH_CALUDE_largest_valid_number_l3004_300423

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a : Fin 10 → Fin 10),
    n = a 9 * 10^9 + a 8 * 10^8 + a 7 * 10^7 + a 6 * 10^6 + a 5 * 10^5 + 
        a 4 * 10^4 + a 3 * 10^3 + a 2 * 10^2 + a 1 * 10 + a 0 ∧
    ∀ i : Fin 10, (List.count (a i) (List.map a (List.range 10)) = a (9 - i))

def is_largest_valid_number (n : ℕ) : Prop :=
  is_valid_number n ∧ 
  ∀ m : ℕ, is_valid_number m → m ≤ n

theorem largest_valid_number : 
  is_largest_valid_number 8888228888 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3004_300423


namespace NUMINAMATH_CALUDE_last_problem_number_l3004_300432

theorem last_problem_number 
  (start : ℕ) 
  (total : ℕ) 
  (h1 : start = 75) 
  (h2 : total = 51) : 
  start + total - 1 = 125 := by
sorry

end NUMINAMATH_CALUDE_last_problem_number_l3004_300432


namespace NUMINAMATH_CALUDE_final_price_calculation_l3004_300408

/-- Calculates the final price of a set containing coffee, cheesecake, and sandwich -/
theorem final_price_calculation (coffee_price cheesecake_price sandwich_price : ℝ)
  (coffee_discount : ℝ) (additional_discount : ℝ) :
  coffee_price = 6 →
  cheesecake_price = 10 →
  sandwich_price = 8 →
  coffee_discount = 0.25 * coffee_price →
  additional_discount = 3 →
  (coffee_price - coffee_discount + cheesecake_price + sandwich_price) - additional_discount = 19.5 :=
by sorry

end NUMINAMATH_CALUDE_final_price_calculation_l3004_300408


namespace NUMINAMATH_CALUDE_log_sawing_time_l3004_300496

theorem log_sawing_time (log_length : ℕ) (section_length : ℕ) (saw_time : ℕ) 
  (h1 : log_length = 10)
  (h2 : section_length = 1)
  (h3 : saw_time = 3) :
  (log_length - 1) * saw_time = 27 :=
by sorry

end NUMINAMATH_CALUDE_log_sawing_time_l3004_300496


namespace NUMINAMATH_CALUDE_equal_integers_from_cyclic_equation_l3004_300475

theorem equal_integers_from_cyclic_equation 
  (n : ℕ+) (p : ℕ) (a b c : ℤ) 
  (h_prime : Nat.Prime p)
  (h_eq1 : a^(n : ℕ) + p * b = b^(n : ℕ) + p * c)
  (h_eq2 : b^(n : ℕ) + p * c = c^(n : ℕ) + p * a) :
  a = b ∧ b = c := by
sorry

end NUMINAMATH_CALUDE_equal_integers_from_cyclic_equation_l3004_300475


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_two_l3004_300433

theorem alpha_plus_beta_equals_two (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α - 17 = 0)
  (hβ : β^3 - 3*β^2 + 5*β + 11 = 0) : 
  α + β = 2 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_two_l3004_300433


namespace NUMINAMATH_CALUDE_power_inequality_l3004_300491

theorem power_inequality (x t : ℝ) (hx : x ≥ 3) :
  (0 < t ∧ t < 1 → x^t - (x-1)^t < (x-2)^t - (x-3)^t) ∧
  (t > 1 → x^t - (x-1)^t > (x-2)^t - (x-3)^t) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3004_300491


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3004_300479

/-- A line in the form y - 2 = mx + m passes through the point (-1, 2) for all values of m -/
theorem line_passes_through_fixed_point (m : ℝ) :
  let line := fun (x y : ℝ) => y - 2 = m * x + m
  line (-1) 2 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3004_300479


namespace NUMINAMATH_CALUDE_specific_weekly_profit_l3004_300406

/-- Represents a business owner's financial situation --/
structure BusinessOwner where
  daily_earnings : ℕ
  weekly_rent : ℕ

/-- Calculates the weekly profit for a business owner --/
def weekly_profit (owner : BusinessOwner) : ℕ :=
  owner.daily_earnings * 7 - owner.weekly_rent

/-- Theorem stating that a business owner with specific earnings and rent has a weekly profit of $36 --/
theorem specific_weekly_profit :
  ∀ (owner : BusinessOwner),
    owner.daily_earnings = 8 →
    owner.weekly_rent = 20 →
    weekly_profit owner = 36 := by
  sorry

#eval weekly_profit { daily_earnings := 8, weekly_rent := 20 }

end NUMINAMATH_CALUDE_specific_weekly_profit_l3004_300406


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l3004_300480

/-- Given a paint mixture with ratio blue:green:white as 3:3:5, 
    prove that using 15 quarts of white paint requires 9 quarts of green paint -/
theorem paint_mixture_ratio (blue green white : ℚ) : 
  blue / green = 1 →
  green / white = 3 / 5 →
  white = 15 →
  green = 9 :=
by sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l3004_300480


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l3004_300465

theorem cubic_root_equation_solution :
  ∀ x : ℝ, (x^(1/3) = 15 / (8 - x^(1/3))) ↔ (x = 27 ∨ x = 125) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l3004_300465


namespace NUMINAMATH_CALUDE_assignment_methods_eq_twelve_l3004_300488

/-- The number of ways to assign doctors and nurses to schools. -/
def assignment_methods (num_doctors num_nurses num_schools : ℕ) : ℕ :=
  (num_doctors.choose 1) * (num_nurses.choose 2)

/-- Theorem stating the number of assignment methods for the given problem. -/
theorem assignment_methods_eq_twelve :
  assignment_methods 2 4 2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_assignment_methods_eq_twelve_l3004_300488


namespace NUMINAMATH_CALUDE_village_foods_lettuce_price_l3004_300461

/-- The price of a head of lettuce at Village Foods -/
def lettuce_price : ℝ := 1

/-- The number of customers per month -/
def customers_per_month : ℕ := 500

/-- The number of lettuce heads each customer buys -/
def lettuce_per_customer : ℕ := 2

/-- The number of tomatoes each customer buys -/
def tomatoes_per_customer : ℕ := 4

/-- The price of each tomato -/
def tomato_price : ℝ := 0.5

/-- The total monthly sales of lettuce and tomatoes -/
def total_monthly_sales : ℝ := 2000

theorem village_foods_lettuce_price :
  lettuce_price = 1 ∧
  customers_per_month * lettuce_per_customer * lettuce_price +
  customers_per_month * tomatoes_per_customer * tomato_price = total_monthly_sales :=
by sorry

end NUMINAMATH_CALUDE_village_foods_lettuce_price_l3004_300461


namespace NUMINAMATH_CALUDE_parabola_equation_l3004_300449

/-- A parabola perpendicular to the x-axis passing through (1, -√2) has the equation y² = 2x -/
theorem parabola_equation : ∃ (f : ℝ → ℝ),
  (∀ x y : ℝ, f x = y ↔ y^2 = 2*x) ∧ 
  (f 1 = -Real.sqrt 2) ∧
  (∀ x y : ℝ, f x = y → (x, y) ∈ {p : ℝ × ℝ | p.2^2 = 2*p.1}) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l3004_300449


namespace NUMINAMATH_CALUDE_union_of_sets_l3004_300416

def A (a : ℕ) : Set ℕ := {3, 2^a}
def B (a b : ℕ) : Set ℕ := {a, b}

theorem union_of_sets (a b : ℕ) (h : A a ∩ B a b = {2}) : A a ∪ B a b = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3004_300416


namespace NUMINAMATH_CALUDE_mangoes_sold_to_market_proof_l3004_300438

/-- Calculates the amount of mangoes sold to market given total harvest, mangoes per kilogram, and remaining mangoes -/
def mangoes_sold_to_market (total_harvest : ℕ) (mangoes_per_kg : ℕ) (remaining_mangoes : ℕ) : ℕ :=
  let total_mangoes := total_harvest * mangoes_per_kg
  let sold_mangoes := total_mangoes - remaining_mangoes
  sold_mangoes / 2 / mangoes_per_kg

/-- Theorem stating that given the problem conditions, 20 kilograms of mangoes were sold to market -/
theorem mangoes_sold_to_market_proof :
  mangoes_sold_to_market 60 8 160 = 20 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_sold_to_market_proof_l3004_300438


namespace NUMINAMATH_CALUDE_unique_p_q_sum_l3004_300400

theorem unique_p_q_sum (p q : ℤ) : 
  p > 1 → q > 1 → 
  ∃ (k₁ k₂ : ℤ), (2*p - 1 = k₁ * q) ∧ (2*q - 1 = k₂ * p) →
  p + q = 8 := by
sorry

end NUMINAMATH_CALUDE_unique_p_q_sum_l3004_300400


namespace NUMINAMATH_CALUDE_inequality_implies_k_range_l3004_300460

/-- The natural logarithm function -/
noncomputable def f (x : ℝ) : ℝ := Real.log x

/-- The exponential function -/
noncomputable def g (x : ℝ) : ℝ := Real.exp x

/-- The main theorem -/
theorem inequality_implies_k_range (k : ℝ) :
  (∀ x : ℝ, x ≥ 1 → x * f x - k * (x + 1) * f (g (x - 1)) ≤ 0) →
  k ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_inequality_implies_k_range_l3004_300460


namespace NUMINAMATH_CALUDE_no_integer_points_on_circle_l3004_300477

theorem no_integer_points_on_circle : 
  ¬ ∃ (x : ℤ), (x - 3)^2 + (x + 1 + 2)^2 ≤ 8^2 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_points_on_circle_l3004_300477
