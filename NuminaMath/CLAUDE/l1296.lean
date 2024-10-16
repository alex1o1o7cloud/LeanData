import Mathlib

namespace NUMINAMATH_CALUDE_workforce_reduction_l1296_129695

theorem workforce_reduction (initial_employees : ℕ) : 
  (initial_employees : ℝ) * 0.85 * 0.75 = 182 → 
  initial_employees = 285 :=
by
  sorry

end NUMINAMATH_CALUDE_workforce_reduction_l1296_129695


namespace NUMINAMATH_CALUDE_x_cube_x_x_square_l1296_129616

theorem x_cube_x_x_square (x : ℝ) (h : -1 < x ∧ x < 0) : x^3 < x ∧ x < x^2 := by
  sorry

end NUMINAMATH_CALUDE_x_cube_x_x_square_l1296_129616


namespace NUMINAMATH_CALUDE_clock_hands_coincidence_coincidence_time_in_hours_and_minutes_l1296_129639

/-- The time in minutes when the hour and minute hands of a clock coincide after midnight -/
def coincidence_time : ℚ :=
  720 / 11

theorem clock_hands_coincidence :
  let minute_speed : ℚ := 360 / 60  -- degrees per minute
  let hour_speed : ℚ := 360 / 720   -- degrees per minute
  ∀ t : ℚ,
    t > 0 →
    t < coincidence_time →
    minute_speed * t ≠ hour_speed * t + 360 * (t / 720).floor →
    minute_speed * coincidence_time = hour_speed * coincidence_time + 360 :=
by sorry

theorem coincidence_time_in_hours_and_minutes :
  (coincidence_time / 60).floor = 1 ∧
  (coincidence_time % 60 : ℚ) = 65 / 11 :=
by sorry

end NUMINAMATH_CALUDE_clock_hands_coincidence_coincidence_time_in_hours_and_minutes_l1296_129639


namespace NUMINAMATH_CALUDE_average_weight_problem_l1296_129615

theorem average_weight_problem (A B C : ℝ) 
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 43)
  (h3 : B = 31) :
  (A + B + C) / 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l1296_129615


namespace NUMINAMATH_CALUDE_divisibility_property_l1296_129699

theorem divisibility_property (x y a b S : ℤ) 
  (sum_eq : x + y = S) 
  (masha_divisible : S ∣ (a * x + b * y)) : 
  S ∣ (b * x + a * y) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l1296_129699


namespace NUMINAMATH_CALUDE_always_positive_l1296_129673

-- Define a monotonically increasing odd function
def MonoIncreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x)

-- Define an arithmetic sequence
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h_f : MonoIncreasingOddFunction f)
  (h_a : ArithmeticSequence a)
  (h_a3 : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_always_positive_l1296_129673


namespace NUMINAMATH_CALUDE_percentage_problem_l1296_129697

theorem percentage_problem (x : ℝ) (h : 0.05 * x = 8) : 0.25 * x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1296_129697


namespace NUMINAMATH_CALUDE_number_with_32_percent_equal_115_2_l1296_129609

theorem number_with_32_percent_equal_115_2 (x : ℝ) :
  (32 / 100) * x = 115.2 → x = 360 := by
  sorry

end NUMINAMATH_CALUDE_number_with_32_percent_equal_115_2_l1296_129609


namespace NUMINAMATH_CALUDE_ten_thousand_squared_l1296_129688

theorem ten_thousand_squared : (10000 : ℕ) * 10000 = 100000000 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_squared_l1296_129688


namespace NUMINAMATH_CALUDE_expense_increase_percentage_l1296_129608

def monthly_salary : ℝ := 6000
def initial_savings_rate : ℝ := 0.20
def new_savings : ℝ := 240

theorem expense_increase_percentage :
  let initial_savings := monthly_salary * initial_savings_rate
  let savings_reduction := initial_savings - new_savings
  let expense_increase_percentage := (savings_reduction / monthly_salary) * 100
  expense_increase_percentage = 16 := by sorry

end NUMINAMATH_CALUDE_expense_increase_percentage_l1296_129608


namespace NUMINAMATH_CALUDE_min_value_function_l1296_129668

theorem min_value_function (x : ℝ) (h : x > 2) : 
  (x^2 - 4*x + 8) / (x - 2) ≥ 4 ∧ ∃ y > 2, (y^2 - 4*y + 8) / (y - 2) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_function_l1296_129668


namespace NUMINAMATH_CALUDE_black_area_calculation_l1296_129634

theorem black_area_calculation (large_side : ℝ) (small_side : ℝ) :
  large_side = 12 →
  small_side = 5 →
  large_side^2 - 2 * small_side^2 = 94 := by
  sorry

end NUMINAMATH_CALUDE_black_area_calculation_l1296_129634


namespace NUMINAMATH_CALUDE_alpha_values_l1296_129651

theorem alpha_values (α : Real) 
  (h1 : 0 < α ∧ α < 2 * Real.pi)
  (h2 : Real.sin α = Real.cos α)
  (h3 : (Real.sin α > 0 ∧ Real.cos α > 0) ∨ (Real.sin α < 0 ∧ Real.cos α < 0)) :
  α = Real.pi / 4 ∨ α = 5 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_alpha_values_l1296_129651


namespace NUMINAMATH_CALUDE_BC_length_is_580_l1296_129676

-- Define a structure for a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def has_integer_lengths (q : Quadrilateral) : Prop := sorry

def right_angle_at_B_and_D (q : Quadrilateral) : Prop := sorry

def AB_equals_BD (q : Quadrilateral) : Prop := sorry

def CD_equals_41 (q : Quadrilateral) : Prop := sorry

-- Define the length of BC
def BC_length (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem BC_length_is_580 (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_integer : has_integer_lengths q)
  (h_right_angles : right_angle_at_B_and_D q)
  (h_AB_BD : AB_equals_BD q)
  (h_CD_41 : CD_equals_41 q) :
  BC_length q = 580 := by sorry

end NUMINAMATH_CALUDE_BC_length_is_580_l1296_129676


namespace NUMINAMATH_CALUDE_opposite_numbers_quotient_l1296_129694

theorem opposite_numbers_quotient (a b : ℝ) :
  a ≠ b → a = -b → a / b = -1 := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_quotient_l1296_129694


namespace NUMINAMATH_CALUDE_coefficients_of_given_equation_l1296_129664

/-- Represents a quadratic equation in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given quadratic equation 2x^2 + x - 5 = 0 -/
def givenEquation : QuadraticEquation := ⟨2, 1, -5⟩

theorem coefficients_of_given_equation :
  givenEquation.a = 2 ∧ givenEquation.b = 1 ∧ givenEquation.c = -5 := by
  sorry

end NUMINAMATH_CALUDE_coefficients_of_given_equation_l1296_129664


namespace NUMINAMATH_CALUDE_m_value_l1296_129690

theorem m_value (m : ℝ) (h1 : m ≠ 0) :
  (∀ x : ℝ, (x^2 - m) * (x + m) = x^3 + m * (x^2 - x - 12)) →
  m = 12 := by
  sorry

end NUMINAMATH_CALUDE_m_value_l1296_129690


namespace NUMINAMATH_CALUDE_probability_diamond_then_ace_or_king_l1296_129658

/-- The number of cards in a combined deck of two standard decks -/
def total_cards : ℕ := 104

/-- The number of diamond cards in a combined deck of two standard decks -/
def diamond_cards : ℕ := 26

/-- The number of ace or king cards in a combined deck of two standard decks -/
def ace_or_king_cards : ℕ := 16

/-- The number of diamond cards that are not ace or king -/
def non_ace_king_diamond : ℕ := 22

/-- The number of diamond cards that are ace or king -/
def ace_king_diamond : ℕ := 4

theorem probability_diamond_then_ace_or_king :
  (diamond_cards * ace_or_king_cards - ace_king_diamond) / (total_cards * (total_cards - 1)) = 103 / 2678 := by
  sorry

end NUMINAMATH_CALUDE_probability_diamond_then_ace_or_king_l1296_129658


namespace NUMINAMATH_CALUDE_find_Z_l1296_129684

theorem find_Z : ∃ Z : ℝ, (100 + 20 / Z) * Z = 9020 ∧ Z = 90 := by
  sorry

end NUMINAMATH_CALUDE_find_Z_l1296_129684


namespace NUMINAMATH_CALUDE_camel_cost_l1296_129692

/-- The cost of animals in an imaginary market. -/
structure AnimalCosts where
  camel : ℚ
  horse : ℚ
  goat : ℚ
  ox : ℚ
  elephant : ℚ

/-- The conditions of the animal costs problem. -/
def animal_costs_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  26 * costs.horse = 50 * costs.goat ∧
  20 * costs.goat = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 170000

/-- The theorem stating that under the given conditions, a camel costs 27200. -/
theorem camel_cost (costs : AnimalCosts) :
  animal_costs_conditions costs → costs.camel = 27200 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l1296_129692


namespace NUMINAMATH_CALUDE_count_leap_years_l1296_129614

def is_leap_year (year : ℕ) : Bool :=
  if year % 100 = 0 then year % 400 = 0 else year % 4 = 0

def years : List ℕ := [1964, 1978, 1995, 1996, 2001, 2100]

theorem count_leap_years : (years.filter is_leap_year).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_leap_years_l1296_129614


namespace NUMINAMATH_CALUDE_quadratic_coefficient_inequalities_l1296_129635

theorem quadratic_coefficient_inequalities
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_real_roots : ∃ x y : ℝ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y) :
  min a (min b c) ≤ (1/4) * (a + b + c) ∧
  max a (max b c) ≥ (4/9) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_inequalities_l1296_129635


namespace NUMINAMATH_CALUDE_equal_numbers_iff_odd_l1296_129643

/-- Represents a square table of numbers -/
def Table (n : ℕ) := Fin n → Fin n → ℕ

/-- Initial state of the table with ones on the diagonal and zeros elsewhere -/
def initialTable (n : ℕ) : Table n :=
  λ i j => if i = j then 1 else 0

/-- Represents a closed path of a rook on the table -/
def RookPath (n : ℕ) := List (Fin n × Fin n)

/-- Checks if a path is valid (closed and non-self-intersecting) -/
def isValidPath (n : ℕ) (path : RookPath n) : Prop := sorry

/-- Applies the transformation along a given path -/
def applyTransformation (n : ℕ) (table : Table n) (path : RookPath n) : Table n := sorry

/-- Checks if all numbers in the table are equal -/
def allEqual (n : ℕ) (table : Table n) : Prop := sorry

/-- Main theorem: It's possible to make all numbers equal if and only if n is odd -/
theorem equal_numbers_iff_odd (n : ℕ) :
  (∃ (transformations : List (RookPath n)), 
    (∀ path ∈ transformations, isValidPath n path) ∧ 
    allEqual n (transformations.foldl (applyTransformation n) (initialTable n))) 
  ↔ n % 2 = 1 := by sorry

end NUMINAMATH_CALUDE_equal_numbers_iff_odd_l1296_129643


namespace NUMINAMATH_CALUDE_proposition_four_l1296_129663

theorem proposition_four (p q : Prop) :
  (p → q) ∧ ¬(q → p) → (¬p → ¬q) ∧ ¬(¬q → ¬p) :=
sorry


end NUMINAMATH_CALUDE_proposition_four_l1296_129663


namespace NUMINAMATH_CALUDE_frac_5_23_150th_digit_l1296_129632

/-- The decimal expansion of 5/23 -/
def decimal_expansion : ℕ → ℕ := sorry

/-- The period of the decimal expansion of 5/23 -/
def period : ℕ := 23

theorem frac_5_23_150th_digit : 
  decimal_expansion ((150 - 1) % period + 1) = 1 := by sorry

end NUMINAMATH_CALUDE_frac_5_23_150th_digit_l1296_129632


namespace NUMINAMATH_CALUDE_birthday_presents_total_l1296_129607

def leonard_wallets : ℕ := 3
def leonard_wallet_price : ℕ := 35
def leonard_sneakers : ℕ := 2
def leonard_sneaker_price : ℕ := 120
def leonard_belt_price : ℕ := 45

def michael_backpack_price : ℕ := 90
def michael_jeans : ℕ := 3
def michael_jeans_price : ℕ := 55
def michael_tie_price : ℕ := 25

def emily_shirts : ℕ := 2
def emily_shirt_price : ℕ := 70
def emily_books : ℕ := 4
def emily_book_price : ℕ := 15

def total_spent : ℕ := 870

theorem birthday_presents_total :
  (leonard_wallets * leonard_wallet_price + 
   leonard_sneakers * leonard_sneaker_price + 
   leonard_belt_price) +
  (michael_backpack_price + 
   michael_jeans * michael_jeans_price + 
   michael_tie_price) +
  (emily_shirts * emily_shirt_price + 
   emily_books * emily_book_price) = total_spent := by
  sorry

end NUMINAMATH_CALUDE_birthday_presents_total_l1296_129607


namespace NUMINAMATH_CALUDE_arcsin_cos_arcsin_plus_arccos_sin_arccos_l1296_129619

theorem arcsin_cos_arcsin_plus_arccos_sin_arccos (x : ℝ) (h : x ∈ Set.Icc (-1) 1) :
  Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_cos_arcsin_plus_arccos_sin_arccos_l1296_129619


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1296_129678

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X : Polynomial ℝ)^5 + 1 = (X^2 - 4*X + 5) * q + 76 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1296_129678


namespace NUMINAMATH_CALUDE_third_smallest_prime_squared_cubed_l1296_129629

-- Define a function to get the nth smallest prime number
def nthSmallestPrime (n : ℕ) : ℕ := sorry

-- State the theorem
theorem third_smallest_prime_squared_cubed :
  (nthSmallestPrime 3) ^ 2 ^ 3 = 15625 := by sorry

end NUMINAMATH_CALUDE_third_smallest_prime_squared_cubed_l1296_129629


namespace NUMINAMATH_CALUDE_parabola_equation_with_sqrt3_distance_l1296_129636

/-- Represents a parabola opening upwards -/
structure UprightParabola where
  /-- The distance from the focus to the directrix -/
  focus_directrix_distance : ℝ
  /-- Condition that the parabola opens upwards -/
  opens_upward : focus_directrix_distance > 0

/-- The standard equation of an upright parabola -/
def standard_equation (p : UprightParabola) : Prop :=
  ∀ x y : ℝ, x^2 = 2 * p.focus_directrix_distance * y

/-- Theorem stating the standard equation of a parabola with focus-directrix distance √3 -/
theorem parabola_equation_with_sqrt3_distance :
  ∀ (p : UprightParabola),
    p.focus_directrix_distance = Real.sqrt 3 →
    standard_equation p
    := by sorry

end NUMINAMATH_CALUDE_parabola_equation_with_sqrt3_distance_l1296_129636


namespace NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l1296_129681

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  pentagonal_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  sorry

/-- Theorem stating the number of space diagonals in the specific polyhedron Q -/
theorem space_diagonals_of_specific_polyhedron :
  ∃ Q : ConvexPolyhedron,
    Q.vertices = 30 ∧
    Q.edges = 70 ∧
    Q.faces = 40 ∧
    Q.triangular_faces = 30 ∧
    Q.pentagonal_faces = 10 ∧
    space_diagonals Q = 315 :=
  sorry

end NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l1296_129681


namespace NUMINAMATH_CALUDE_abs_neg_2023_eq_2023_l1296_129625

theorem abs_neg_2023_eq_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_eq_2023_l1296_129625


namespace NUMINAMATH_CALUDE_sock_drawer_theorem_l1296_129621

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (purple : ℕ)

/-- The minimum number of socks needed to guarantee at least n pairs -/
def minSocksForPairs (drawer : SockDrawer) (n : ℕ) : ℕ :=
  sorry

theorem sock_drawer_theorem (drawer : SockDrawer) 
  (h1 : drawer.red ≥ 30)
  (h2 : drawer.green ≥ 30)
  (h3 : drawer.blue ≥ 30)
  (h4 : drawer.yellow ≥ 30)
  (h5 : drawer.purple ≥ 30) :
  minSocksForPairs drawer 15 = 118 :=
sorry

end NUMINAMATH_CALUDE_sock_drawer_theorem_l1296_129621


namespace NUMINAMATH_CALUDE_brick_surface_area_l1296_129622

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 2 cm brick is 136 cm² -/
theorem brick_surface_area :
  surface_area 10 4 2 = 136 := by
  sorry

end NUMINAMATH_CALUDE_brick_surface_area_l1296_129622


namespace NUMINAMATH_CALUDE_system_solution_l1296_129627

theorem system_solution (x y z : ℝ) : 
  x = 1 ∧ y = -1 ∧ z = -2 →
  (2 * x + y + z = -1) ∧
  (3 * y - z = -1) ∧
  (3 * x + 2 * y + 3 * z = -5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1296_129627


namespace NUMINAMATH_CALUDE_probability_unqualified_example_l1296_129687

/-- Represents the probability of selecting at least one unqualified can -/
def probability_unqualified (total_cans : ℕ) (qualified_cans : ℕ) (unqualified_cans : ℕ) (selected_cans : ℕ) : ℚ :=
  1 - (Nat.choose qualified_cans selected_cans : ℚ) / (Nat.choose total_cans selected_cans : ℚ)

/-- Theorem stating that the probability of selecting at least one unqualified can
    when randomly choosing 2 cans from a box containing 3 qualified cans and 2 unqualified cans
    is equal to 0.7 -/
theorem probability_unqualified_example : probability_unqualified 5 3 2 2 = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_probability_unqualified_example_l1296_129687


namespace NUMINAMATH_CALUDE_system_solution_l1296_129659

theorem system_solution (x y z : ℝ) :
  (x^3 = z/y - 2*y/z ∧ y^3 = x/z - 2*z/x ∧ z^3 = y/x - 2*x/y) →
  ((x = 1 ∧ y = 1 ∧ z = -1) ∨
   (x = 1 ∧ y = -1 ∧ z = 1) ∨
   (x = -1 ∧ y = 1 ∧ z = 1) ∨
   (x = -1 ∧ y = -1 ∧ z = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1296_129659


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l1296_129671

theorem mans_rate_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 6)
  (h2 : speed_against_stream = 2) : 
  (speed_with_stream + speed_against_stream) / 2 = 4 := by
  sorry

#check mans_rate_in_still_water

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l1296_129671


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l1296_129698

/-- The smallest integer b > 3 for which 34_b is a perfect square -/
theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 3 ∧ 
  (∀ (x : ℕ), x > 3 ∧ x < b → ¬∃ (y : ℕ), 3*x + 4 = y^2) ∧
  (∃ (y : ℕ), 3*b + 4 = y^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l1296_129698


namespace NUMINAMATH_CALUDE_probability_theorem_l1296_129661

/-- The probability of drawing one white ball and one black ball from an urn -/
def probability_one_white_one_black (a b : ℕ) : ℚ :=
  (2 * a * b : ℚ) / ((a + b) * (a + b - 1))

/-- Theorem stating the probability of drawing one white and one black ball -/
theorem probability_theorem (a b : ℕ) (h : a + b > 1) :
  probability_one_white_one_black a b =
    (2 * a * b : ℚ) / ((a + b) * (a + b - 1)) := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l1296_129661


namespace NUMINAMATH_CALUDE_no_prime_sided_integer_area_triangle_l1296_129646

theorem no_prime_sided_integer_area_triangle : 
  ¬ ∃ (a b c : ℕ) (S : ℝ), 
    (Prime a ∧ Prime b ∧ Prime c) ∧ 
    (S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) ∧ 
    (S ≠ 0) ∧ 
    (∃ (n : ℕ), S = n) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sided_integer_area_triangle_l1296_129646


namespace NUMINAMATH_CALUDE_min_hypotenuse_right_triangle_l1296_129693

theorem min_hypotenuse_right_triangle (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b = 10) 
  (h5 : c^2 = a^2 + b^2) : 
  c ≥ 5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_hypotenuse_right_triangle_l1296_129693


namespace NUMINAMATH_CALUDE_radio_price_proof_l1296_129666

theorem radio_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 465.50)
  (h2 : loss_percentage = 5) : 
  ∃ (original_price : ℝ), 
    original_price = 490 ∧ 
    selling_price = original_price * (1 - loss_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_radio_price_proof_l1296_129666


namespace NUMINAMATH_CALUDE_sequence_classification_l1296_129679

/-- Given a sequence {aₙ} where the sum of the first n terms Sₙ = aⁿ - 1 (a is a non-zero real number),
    the sequence {aₙ} is either an arithmetic sequence or a geometric sequence. -/
theorem sequence_classification (a : ℝ) (ha : a ≠ 0) :
  let S : ℕ → ℝ := λ n => a^n - 1
  let a_seq : ℕ → ℝ := λ n => S n - S (n-1)
  (∀ n : ℕ, n > 1 → a_seq (n+1) - a_seq n = 0) ∨
  (∀ n : ℕ, n > 2 → a_seq (n+1) / a_seq n = a) :=
by sorry

end NUMINAMATH_CALUDE_sequence_classification_l1296_129679


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1296_129618

/-- The distance between the vertices of the hyperbola y^2/45 - x^2/20 = 1 is 6√5 -/
theorem hyperbola_vertex_distance : 
  let a := Real.sqrt 45
  let vertex_distance := 2 * a
  vertex_distance = 6 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1296_129618


namespace NUMINAMATH_CALUDE_circle_area_radius_decrease_l1296_129656

theorem circle_area_radius_decrease (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let A' := 0.64 * A
  let r' := Real.sqrt (A' / π)
  r' / r = 0.8 := by sorry

end NUMINAMATH_CALUDE_circle_area_radius_decrease_l1296_129656


namespace NUMINAMATH_CALUDE_normal_carwash_cost_l1296_129667

/-- The normal cost of a single carwash, given a discounted package deal -/
theorem normal_carwash_cost (package_size : ℕ) (discount_rate : ℚ) (package_price : ℚ) : 
  package_size = 20 →
  discount_rate = 3/5 →
  package_price = 180 →
  package_price = discount_rate * (package_size * (15 : ℚ)) := by
sorry

end NUMINAMATH_CALUDE_normal_carwash_cost_l1296_129667


namespace NUMINAMATH_CALUDE_digit_150_of_one_thirteenth_l1296_129611

def decimal_representation (n : ℕ) : ℚ → List ℕ := sorry

def nth_digit (n : ℕ) (l : List ℕ) : ℕ := sorry

theorem digit_150_of_one_thirteenth :
  let rep := decimal_representation 13 (1/13)
  nth_digit 150 rep = 3 := by sorry

end NUMINAMATH_CALUDE_digit_150_of_one_thirteenth_l1296_129611


namespace NUMINAMATH_CALUDE_x_convergence_to_sqrt2_l1296_129670

-- Define the sequence x_n
def x : ℕ → ℚ
| 0 => 1
| (n+1) => 1 + 1 / (2 + 1 / (x n))

-- Define the bound function
def bound (n : ℕ) : ℚ := 1 / 2^(2^n - 1)

-- State the theorem
theorem x_convergence_to_sqrt2 (n : ℕ) :
  |x n - Real.sqrt 2| < bound n :=
sorry

end NUMINAMATH_CALUDE_x_convergence_to_sqrt2_l1296_129670


namespace NUMINAMATH_CALUDE_probability_of_specific_arrangement_l1296_129633

def total_tiles : ℕ := 8
def x_tiles : ℕ := 5
def o_tiles : ℕ := 3

theorem probability_of_specific_arrangement :
  let total_arrangements := Nat.choose total_tiles x_tiles
  let specific_arrangement := 1
  (specific_arrangement : ℚ) / total_arrangements = 1 / 56 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_arrangement_l1296_129633


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1296_129680

theorem negation_of_proposition (P : Prop) :
  (¬ (∃ x : ℝ, 2 * x + 1 ≤ 0)) ↔ (∀ x : ℝ, 2 * x + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1296_129680


namespace NUMINAMATH_CALUDE_intersection_sum_l1296_129604

/-- Two functions f and g that intersect at given points -/
def f (a b x : ℝ) : ℝ := -2 * abs (x - a) + b
def g (c d x : ℝ) : ℝ := 2 * abs (x - c) + d

/-- Theorem stating that for functions f and g intersecting at (1, 7) and (11, -1), a + c = 12 -/
theorem intersection_sum (a b c d : ℝ) 
  (h1 : f a b 1 = g c d 1 ∧ f a b 1 = 7)
  (h2 : f a b 11 = g c d 11 ∧ f a b 11 = -1) :
  a + c = 12 := by
  sorry


end NUMINAMATH_CALUDE_intersection_sum_l1296_129604


namespace NUMINAMATH_CALUDE_eq2_eq3_same_graph_eq1_different_graph_l1296_129648

-- Define the three equations
def eq1 (x y : ℝ) : Prop := y = x + 3
def eq2 (x y : ℝ) : Prop := y = (x^2 - 1) / (x - 1)
def eq3 (x y : ℝ) : Prop := (x - 1) * y = x^2 - 1

-- Define the concept of having the same graph
def same_graph (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, x ≠ 1 → (f x y ↔ g x y)

-- Theorem stating that eq2 and eq3 have the same graph
theorem eq2_eq3_same_graph : same_graph eq2 eq3 := by sorry

-- Theorem stating that eq1 has a different graph from eq2 and eq3
theorem eq1_different_graph :
  ¬(same_graph eq1 eq2) ∧ ¬(same_graph eq1 eq3) := by sorry

end NUMINAMATH_CALUDE_eq2_eq3_same_graph_eq1_different_graph_l1296_129648


namespace NUMINAMATH_CALUDE_division_problem_l1296_129662

theorem division_problem (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = 725 →
  quotient = 20 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  divisor = 36 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1296_129662


namespace NUMINAMATH_CALUDE_arrangement_count_is_72_l1296_129644

/-- Represents the number of ways to arrange 5 people with specific conditions -/
def arrangement_count : ℕ := 72

/-- The number of people in the arrangement -/
def total_people : ℕ := 5

/-- The number of ways to arrange C and D together -/
def cd_arrangements : ℕ := 2

/-- The number of ways to arrange 3 entities (C-D unit, another person, and a space) -/
def entity_arrangements : ℕ := 6

/-- The number of ways to place A and B not adjacent in the remaining spaces -/
def ab_placements : ℕ := 6

/-- Theorem stating that the number of arrangements satisfying the conditions is 72 -/
theorem arrangement_count_is_72 :
  arrangement_count = cd_arrangements * entity_arrangements * ab_placements :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_is_72_l1296_129644


namespace NUMINAMATH_CALUDE_potato_bag_weight_l1296_129637

theorem potato_bag_weight (morning_bags : ℕ) (afternoon_bags : ℕ) (total_weight : ℕ) :
  morning_bags = 29 →
  afternoon_bags = 17 →
  total_weight = 322 →
  total_weight / (morning_bags + afternoon_bags) = 7 :=
by sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l1296_129637


namespace NUMINAMATH_CALUDE_exists_special_number_l1296_129620

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number n such that the sum of its digits is 1000
    and the sum of digits of its square is 1000000 -/
theorem exists_special_number : 
  ∃ n : ℕ, sum_of_digits n = 1000 ∧ sum_of_digits (n^2) = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_exists_special_number_l1296_129620


namespace NUMINAMATH_CALUDE_chlorine_moles_l1296_129647

/-- Represents the chemical reaction between Methane and Chlorine to produce Hydrochloric acid -/
def chemical_reaction (methane : ℝ) (chlorine : ℝ) (hydrochloric_acid : ℝ) : Prop :=
  methane = 1 ∧ hydrochloric_acid = 2 ∧ chlorine = hydrochloric_acid

/-- Theorem stating that 2 moles of Chlorine are combined in the reaction -/
theorem chlorine_moles : ∃ (chlorine : ℝ), chemical_reaction 1 chlorine 2 ∧ chlorine = 2 := by
  sorry

end NUMINAMATH_CALUDE_chlorine_moles_l1296_129647


namespace NUMINAMATH_CALUDE_digit_sum_properties_l1296_129642

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem digit_sum_properties :
  (∀ N : ℕ, S N ≤ 8 * S (8 * N)) ∧
  (∀ r q : ℕ, ∃ c_k : ℚ, c_k > 0 ∧
    (∀ N : ℕ, S (2^r * 5^q * N) / S N ≥ c_k) ∧
    c_k = 1 / S (2^q * 5^r) ∧
    (∀ c : ℚ, c > c_k → ∃ N : ℕ, S (2^r * 5^q * N) / S N < c)) ∧
  (∀ k : ℕ, (∃ r q : ℕ, k = 2^r * 5^q) ∨
    (∀ c : ℚ, c > 0 → ∃ N : ℕ, S (k * N) / S N < c)) :=
sorry

end NUMINAMATH_CALUDE_digit_sum_properties_l1296_129642


namespace NUMINAMATH_CALUDE_quadratic_intercepts_l1296_129675

/-- A quadratic function. -/
structure QuadraticFunction where
  f : ℝ → ℝ

/-- The x-intercepts of two quadratic functions. -/
structure XIntercepts where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  x₄ : ℝ

/-- The problem statement. -/
theorem quadratic_intercepts 
  (f g : QuadraticFunction) 
  (x : XIntercepts) 
  (h1 : ∀ x, g.f x = -f.f (120 - x))
  (h2 : ∃ v, g.f v = f.f v ∧ ∀ x, f.f x ≤ f.f v)
  (h3 : x.x₁ < x.x₂ ∧ x.x₂ < x.x₃ ∧ x.x₃ < x.x₄)
  (h4 : x.x₃ - x.x₂ = 160) :
  x.x₄ - x.x₁ = 640 + 320 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_intercepts_l1296_129675


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_sum_of_roots_specific_cubic_l1296_129654

theorem sum_of_roots_cubic (a b c d : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x^2 + c * x + d
  (∃ x y z : ℝ, f x = 0 ∧ f y = 0 ∧ f z = 0) →
  x + y + z = -b / a :=
sorry

theorem sum_of_roots_specific_cubic :
  let f : ℝ → ℝ := λ x => 25 * x^3 - 50 * x^2 + 35 * x + 7
  (∃ x y z : ℝ, f x = 0 ∧ f y = 0 ∧ f z = 0) →
  x + y + z = 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_sum_of_roots_specific_cubic_l1296_129654


namespace NUMINAMATH_CALUDE_calculation_proof_l1296_129655

theorem calculation_proof :
  (1/5 - 2/3 - 3/10) * (-60) = 46 ∧
  (-1)^2024 + 24 / (-2)^3 - 15^2 * (1/15)^2 = -3 :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l1296_129655


namespace NUMINAMATH_CALUDE_bird_weights_solution_l1296_129674

theorem bird_weights_solution :
  ∃! (A B V G : ℕ+),
    A + B + V + G = 32 ∧
    V < G ∧
    V + G < B ∧
    A < V + B ∧
    G + B < A + V ∧
    A = 13 ∧ B = 10 ∧ V = 4 ∧ G = 5 :=
by sorry

end NUMINAMATH_CALUDE_bird_weights_solution_l1296_129674


namespace NUMINAMATH_CALUDE_ball_distribution_ratio_l1296_129628

/-- The number of balls -/
def n : ℕ := 25

/-- The number of bins -/
def m : ℕ := 5

/-- The probability of distributing n balls into m bins such that
    one bin has 3 balls, another has 7 balls, and the other three have 5 balls each -/
noncomputable def p : ℝ := sorry

/-- The probability of distributing n balls equally into m bins (5 balls each) -/
noncomputable def q : ℝ := sorry

/-- Theorem stating that the ratio of p to q is 12 -/
theorem ball_distribution_ratio : p / q = 12 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_ratio_l1296_129628


namespace NUMINAMATH_CALUDE_simplify_fraction_multiplication_l1296_129600

theorem simplify_fraction_multiplication : (405 : ℚ) / 1215 * 27 = 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_multiplication_l1296_129600


namespace NUMINAMATH_CALUDE_initial_distance_calculation_l1296_129623

/-- Represents the scenario of two trucks traveling on the same route --/
structure TruckScenario where
  initial_distance : ℝ
  speed_x : ℝ
  speed_y : ℝ
  overtake_time : ℝ
  final_distance : ℝ

/-- Theorem stating the initial distance between trucks given the scenario conditions --/
theorem initial_distance_calculation (scenario : TruckScenario)
  (h1 : scenario.speed_x = 57)
  (h2 : scenario.speed_y = 63)
  (h3 : scenario.overtake_time = 3)
  (h4 : scenario.final_distance = 4)
  (h5 : scenario.speed_y > scenario.speed_x) :
  scenario.initial_distance = 14 := by
  sorry


end NUMINAMATH_CALUDE_initial_distance_calculation_l1296_129623


namespace NUMINAMATH_CALUDE_courtyard_paving_l1296_129612

theorem courtyard_paving (courtyard_length : ℝ) (courtyard_width : ℝ)
  (brick_length : ℝ) (brick_width : ℝ) :
  courtyard_length = 25 →
  courtyard_width = 16 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  ⌈(courtyard_length * courtyard_width) / (brick_length * brick_width)⌉ = 20000 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_paving_l1296_129612


namespace NUMINAMATH_CALUDE_max_value_problem_l1296_129653

theorem max_value_problem (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) 
  (hb : 0 ≤ b ∧ b ≤ 2) 
  (hc : 0 ≤ c ∧ c ≤ 2) : 
  a^2 * b^2 * c^2 + (2 - a)^2 * (2 - b)^2 * (2 - c)^2 ≤ 64 := by
  sorry

end NUMINAMATH_CALUDE_max_value_problem_l1296_129653


namespace NUMINAMATH_CALUDE_range_of_a_l1296_129696

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, -1 ≤ x₀ ∧ x₀ ≤ 1 ∧ 2 * a * x₀^2 + 2 * x₀ - 3 - a = 0) → 
  (a ≥ 1 ∨ a ≤ (-3 - Real.sqrt 7) / 2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1296_129696


namespace NUMINAMATH_CALUDE_rectangle_perimeter_in_square_l1296_129683

theorem rectangle_perimeter_in_square (d : ℝ) (h : d = 6) : 
  ∃ (s : ℝ), s > 0 ∧ s * Real.sqrt 2 = d ∧
  ∃ (rect_side : ℝ), rect_side = s / Real.sqrt 2 ∧
  4 * rect_side = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_in_square_l1296_129683


namespace NUMINAMATH_CALUDE_gnuff_tutoring_cost_l1296_129631

/-- Calculates the total amount paid for a tutoring session -/
def tutoring_cost (flat_rate : ℕ) (per_minute_rate : ℕ) (minutes : ℕ) : ℕ :=
  flat_rate + per_minute_rate * minutes

/-- Theorem: The total amount paid for Gnuff's tutoring session is $146 -/
theorem gnuff_tutoring_cost :
  tutoring_cost 20 7 18 = 146 := by
  sorry

end NUMINAMATH_CALUDE_gnuff_tutoring_cost_l1296_129631


namespace NUMINAMATH_CALUDE_most_reasonable_estimate_l1296_129672

/-- Represents the total number of female students in the first year -/
def total_female : ℕ := 504

/-- Represents the total number of male students in the first year -/
def total_male : ℕ := 596

/-- Represents the total number of students in the first year -/
def total_students : ℕ := total_female + total_male

/-- Represents the average weight of sampled female students -/
def avg_weight_female : ℝ := 49

/-- Represents the average weight of sampled male students -/
def avg_weight_male : ℝ := 57

/-- Theorem stating that the most reasonable estimate for the average weight
    of all first-year students is (504/1100) * 49 + (596/1100) * 57 -/
theorem most_reasonable_estimate :
  (total_female : ℝ) / total_students * avg_weight_female +
  (total_male : ℝ) / total_students * avg_weight_male =
  (504 : ℝ) / 1100 * 49 + (596 : ℝ) / 1100 * 57 := by
  sorry

end NUMINAMATH_CALUDE_most_reasonable_estimate_l1296_129672


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l1296_129603

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 2| + |x + 3| < a)) → a ∈ Set.Iic 5 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l1296_129603


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l1296_129650

theorem quadratic_equation_problem (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*m*x + m^2 - m + 2 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  x₁ + x₂ + x₁ * x₂ = 2 →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l1296_129650


namespace NUMINAMATH_CALUDE_cube_diff_even_iff_sum_even_l1296_129640

theorem cube_diff_even_iff_sum_even (p q : ℕ) :
  Even (p^3 - q^3) ↔ Even (p + q) := by sorry

end NUMINAMATH_CALUDE_cube_diff_even_iff_sum_even_l1296_129640


namespace NUMINAMATH_CALUDE_sign_determination_l1296_129660

theorem sign_determination (a b : ℝ) (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_sign_determination_l1296_129660


namespace NUMINAMATH_CALUDE_woman_birth_year_l1296_129610

/-- A woman born in the second half of the 19th century was x years old in the year x^2. -/
theorem woman_birth_year :
  ∃ x : ℕ,
    (1850 ≤ x^2 - x) ∧
    (x^2 - x < 1900) ∧
    (x^2 = x + 1892) :=
by sorry

end NUMINAMATH_CALUDE_woman_birth_year_l1296_129610


namespace NUMINAMATH_CALUDE_tiles_needed_to_cover_floor_l1296_129630

-- Define the dimensions of the floor and tiles
def floor_length : ℚ := 10
def floor_width : ℚ := 14
def tile_length : ℚ := 1/2  -- 6 inches in feet
def tile_width : ℚ := 2/3   -- 8 inches in feet

-- Theorem statement
theorem tiles_needed_to_cover_floor :
  (floor_length * floor_width) / (tile_length * tile_width) = 420 := by
  sorry

end NUMINAMATH_CALUDE_tiles_needed_to_cover_floor_l1296_129630


namespace NUMINAMATH_CALUDE_parallel_x_implies_parallel_y_implies_on_bisector_implies_l1296_129652

-- Define the coordinates of points A and B
def A (a : ℝ) : ℝ × ℝ := (a - 1, 2)
def B (b : ℝ) : ℝ × ℝ := (-3, b + 1)

-- Define the conditions
def parallel_to_x_axis (a b : ℝ) : Prop := (A a).2 = (B b).2
def parallel_to_y_axis (a b : ℝ) : Prop := (A a).1 = (B b).1
def on_bisector (a b : ℝ) : Prop := (A a).1 = (A a).2 ∧ (B b).1 = (B b).2

-- Theorem statements
theorem parallel_x_implies (a b : ℝ) : parallel_to_x_axis a b → a ≠ -2 ∧ b = 1 := by sorry

theorem parallel_y_implies (a b : ℝ) : parallel_to_y_axis a b → a = -2 ∧ b ≠ 1 := by sorry

theorem on_bisector_implies (a b : ℝ) : on_bisector a b → a = 3 ∧ b = -4 := by sorry

end NUMINAMATH_CALUDE_parallel_x_implies_parallel_y_implies_on_bisector_implies_l1296_129652


namespace NUMINAMATH_CALUDE_ellipse_focal_chord_area_l1296_129638

/-- Given an ellipse with equation x²/4 + y²/m = 1 (m > 0), where the focal chord F₁F₂ is the diameter
    of a circle intersecting the ellipse at point P in the first quadrant, if the area of triangle PF₁F₂
    is 1, then m = 1. -/
theorem ellipse_focal_chord_area (m : ℝ) (x y : ℝ) (F₁ F₂ P : ℝ × ℝ) : 
  m > 0 → 
  x^2 / 4 + y^2 / m = 1 →
  (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 16 →  -- F₁F₂ is diameter of circle with radius 2
  P.1^2 / 4 + P.2^2 / m = 1 →  -- P is on the ellipse
  P.1 ≥ 0 ∧ P.2 ≥ 0 →  -- P is in the first quadrant
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = 16 →  -- P is on the circle
  abs ((P.1 - F₁.1) * (P.2 - F₂.2) - (P.2 - F₁.2) * (P.1 - F₂.1)) / 2 = 1 →  -- Area of triangle PF₁F₂ is 1
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focal_chord_area_l1296_129638


namespace NUMINAMATH_CALUDE_plot_perimeter_l1296_129624

def rectangular_plot (length width : ℝ) : Prop :=
  length > 0 ∧ width > 0

theorem plot_perimeter (length width : ℝ) :
  rectangular_plot length width →
  length / width = 7 / 5 →
  length * width = 5040 →
  2 * (length + width) = 288 := by
  sorry

end NUMINAMATH_CALUDE_plot_perimeter_l1296_129624


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l1296_129641

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a2 : a 2 = 2)
  (h_sum : a 4 + a 5 = 12) :
  a 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l1296_129641


namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l1296_129686

theorem tan_45_degrees_equals_one : Real.tan (π / 4) = 1 := by sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l1296_129686


namespace NUMINAMATH_CALUDE_jacqueline_initial_plums_l1296_129669

/-- The number of plums Jacqueline had initially -/
def initial_plums : ℕ := 16

/-- The number of guavas Jacqueline had initially -/
def initial_guavas : ℕ := 18

/-- The number of apples Jacqueline had initially -/
def initial_apples : ℕ := 21

/-- The number of fruits Jacqueline gave away -/
def fruits_given_away : ℕ := 40

/-- The number of fruits Jacqueline had left -/
def fruits_left : ℕ := 15

/-- Theorem stating that the initial number of plums is 16 -/
theorem jacqueline_initial_plums :
  initial_plums = 16 ∧
  initial_plums + initial_guavas + initial_apples = fruits_given_away + fruits_left :=
by sorry

end NUMINAMATH_CALUDE_jacqueline_initial_plums_l1296_129669


namespace NUMINAMATH_CALUDE_correct_factorization_l1296_129605

theorem correct_factorization (x : ℝ) : 10 * x^2 - 5 * x = 5 * x * (2 * x - 1) := by
  sorry

#check correct_factorization

end NUMINAMATH_CALUDE_correct_factorization_l1296_129605


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1296_129617

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 17*x + 8 = 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 17*x + 8 = (x - r₁) * (x - r₂) ∧ 
  (1 / r₁ + 1 / r₂ : ℝ) = 17 / 8 :=
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1296_129617


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l1296_129682

/-- The height of a tree after n years, given its initial height and growth factor -/
def tree_height (initial_height : ℝ) (growth_factor : ℝ) (n : ℕ) : ℝ :=
  initial_height * growth_factor ^ n

/-- Theorem: If a tree triples its height every year and reaches 81 feet after 4 years,
    then its height after 2 years is 9 feet -/
theorem tree_height_after_two_years
  (h : ∃ initial_height : ℝ, tree_height initial_height 3 4 = 81) :
  ∃ initial_height : ℝ, tree_height initial_height 3 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l1296_129682


namespace NUMINAMATH_CALUDE_trapezoid_long_side_is_correct_l1296_129649

/-- A rectangle with given dimensions divided into three equal-area shapes -/
structure DividedRectangle where
  length : ℝ
  width : ℝ
  trapezoid_long_side : ℝ
  is_valid : 
    length = 3 ∧ 
    width = 1 ∧
    0 < trapezoid_long_side ∧ 
    trapezoid_long_side < length

/-- The area of each shape is one-third of the rectangle's area -/
def equal_area_condition (r : DividedRectangle) : Prop :=
  let rectangle_area := r.length * r.width
  let trapezoid_area := (r.trapezoid_long_side + r.length / 2) * r.width / 2
  trapezoid_area = rectangle_area / 3

/-- The main theorem: the longer side of the trapezoid is 1.25 -/
theorem trapezoid_long_side_is_correct (r : DividedRectangle) 
  (h : equal_area_condition r) : r.trapezoid_long_side = 1.25 := by
  sorry

#check trapezoid_long_side_is_correct

end NUMINAMATH_CALUDE_trapezoid_long_side_is_correct_l1296_129649


namespace NUMINAMATH_CALUDE_hexagon_percentage_is_62_5_l1296_129613

/-- Represents the tiling pattern of the plane -/
structure TilingPattern where
  /-- The number of smaller squares in each large square -/
  total_squares : ℕ
  /-- The number of smaller squares used to form hexagons in each large square -/
  hexagon_squares : ℕ

/-- Calculates the percentage of the plane enclosed by hexagons -/
def hexagon_percentage (pattern : TilingPattern) : ℚ :=
  (pattern.hexagon_squares : ℚ) / (pattern.total_squares : ℚ) * 100

/-- The theorem stating that the percentage of the plane enclosed by hexagons is 62.5% -/
theorem hexagon_percentage_is_62_5 (pattern : TilingPattern) 
  (h1 : pattern.total_squares = 16)
  (h2 : pattern.hexagon_squares = 10) : 
  hexagon_percentage pattern = 62.5 := by
  sorry

#eval hexagon_percentage { total_squares := 16, hexagon_squares := 10 }

end NUMINAMATH_CALUDE_hexagon_percentage_is_62_5_l1296_129613


namespace NUMINAMATH_CALUDE_power_23_2005_mod_36_l1296_129691

theorem power_23_2005_mod_36 : 23^2005 % 36 = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_23_2005_mod_36_l1296_129691


namespace NUMINAMATH_CALUDE_inequality_proof_l1296_129602

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_one : a + b + c = 1) :
  a^3 + b^3 ≥ a^2*b + a*b^2 ∧ (1/a - 1)*(1/b - 1)*(1/c - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1296_129602


namespace NUMINAMATH_CALUDE_dividend_percentage_calculation_l1296_129657

/-- Calculate the dividend percentage of shares -/
theorem dividend_percentage_calculation 
  (cost_price : ℝ) 
  (desired_interest_rate : ℝ) 
  (market_value : ℝ) : 
  cost_price = 60 →
  desired_interest_rate = 12 / 100 →
  market_value = 45 →
  (market_value * desired_interest_rate) / cost_price * 100 = 9 := by
  sorry

end NUMINAMATH_CALUDE_dividend_percentage_calculation_l1296_129657


namespace NUMINAMATH_CALUDE_not_divisible_by_power_of_five_l1296_129685

theorem not_divisible_by_power_of_five (n : ℕ+) (k : ℕ+) 
  (h : k < 5^n.val - 5^(n.val - 1)) : 
  ¬ (5^n.val ∣ 2^k.val - 1) :=
sorry

end NUMINAMATH_CALUDE_not_divisible_by_power_of_five_l1296_129685


namespace NUMINAMATH_CALUDE_divisible_by_seven_last_digits_l1296_129677

theorem divisible_by_seven_last_digits :
  ∃ (S : Finset Nat), (∀ n : Nat, n % 10 ∈ S ↔ ∃ m : Nat, m % 7 = 0 ∧ m % 10 = n % 10) ∧ Finset.card S = 2 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_seven_last_digits_l1296_129677


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l1296_129626

theorem sum_of_roots_equation (x : ℝ) : 
  (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a + b = 6) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l1296_129626


namespace NUMINAMATH_CALUDE_range_of_a_l1296_129689

/-- The function f defined on positive real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.log x)^2 - Real.log x

/-- The function h defined on positive real numbers. -/
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := (f a x + 1 - a) * (Real.log x)⁻¹

/-- The theorem stating the range of a given the conditions. -/
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (Real.exp (-3)) (Real.exp (-1)) →
                x₂ ∈ Set.Icc (Real.exp (-3)) (Real.exp (-1)) →
                |h a x₁ - h a x₂| ≤ a + 1/3) →
  a ∈ Set.Icc (1/11) (3/5) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1296_129689


namespace NUMINAMATH_CALUDE_train_average_speed_l1296_129606

theorem train_average_speed :
  let distance1 : ℝ := 290
  let time1 : ℝ := 4.5
  let distance2 : ℝ := 400
  let time2 : ℝ := 5.5
  let total_distance : ℝ := distance1 + distance2
  let total_time : ℝ := time1 + time2
  let average_speed : ℝ := total_distance / total_time
  average_speed = 69 := by sorry

end NUMINAMATH_CALUDE_train_average_speed_l1296_129606


namespace NUMINAMATH_CALUDE_fraction_simplification_l1296_129645

theorem fraction_simplification (x y : ℚ) (hx : x = 2/3) (hy : y = 5/8) :
  (6*x + 8*y) / (48*x*y) = 9/20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1296_129645


namespace NUMINAMATH_CALUDE_profit_distribution_correct_l1296_129665

def total_profit : ℕ := 280000

def shekhar_percentage : ℚ := 28 / 100
def rajeev_percentage : ℚ := 22 / 100
def jatin_percentage : ℚ := 20 / 100
def simran_percentage : ℚ := 18 / 100
def ramesh_percentage : ℚ := 12 / 100

def shekhar_share : ℕ := (shekhar_percentage * total_profit).num.toNat
def rajeev_share : ℕ := (rajeev_percentage * total_profit).num.toNat
def jatin_share : ℕ := (jatin_percentage * total_profit).num.toNat
def simran_share : ℕ := (simran_percentage * total_profit).num.toNat
def ramesh_share : ℕ := (ramesh_percentage * total_profit).num.toNat

theorem profit_distribution_correct :
  shekhar_share + rajeev_share + jatin_share + simran_share + ramesh_share = total_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_distribution_correct_l1296_129665


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1296_129601

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  let n := 196713
  let d := 7
  let x := 6
  x < d ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1296_129601
