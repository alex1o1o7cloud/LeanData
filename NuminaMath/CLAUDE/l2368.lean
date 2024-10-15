import Mathlib

namespace NUMINAMATH_CALUDE_simplest_proper_fraction_with_7_numerator_simplest_improper_fraction_with_7_denominator_l2368_236806

-- Define a function to check if a fraction is in its simplest form
def isSimplestForm (n d : ℕ) : Prop :=
  n.gcd d = 1

-- Define a function to check if a fraction is proper
def isProper (n d : ℕ) : Prop :=
  n < d

-- Define a function to check if a fraction is improper
def isImproper (n d : ℕ) : Prop :=
  n ≥ d

-- Theorem for the simplest proper fraction with 7 as numerator
theorem simplest_proper_fraction_with_7_numerator :
  isSimplestForm 7 8 ∧ isProper 7 8 ∧
  ∀ d : ℕ, d > 7 → isSimplestForm 7 d → d ≥ 8 :=
sorry

-- Theorem for the simplest improper fraction with 7 as denominator
theorem simplest_improper_fraction_with_7_denominator :
  isSimplestForm 8 7 ∧ isImproper 8 7 ∧
  ∀ n : ℕ, n > 7 → isSimplestForm n 7 → n ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_simplest_proper_fraction_with_7_numerator_simplest_improper_fraction_with_7_denominator_l2368_236806


namespace NUMINAMATH_CALUDE_initial_walking_speed_l2368_236879

/-- Proves that given a specific distance and time difference between two speeds,
    the initial speed is 11.25 kmph. -/
theorem initial_walking_speed 
  (distance : ℝ) 
  (time_diff : ℝ) 
  (faster_speed : ℝ) :
  distance = 9.999999999999998 →
  time_diff = 1/3 →
  faster_speed = 15 →
  ∃ (initial_speed : ℝ),
    distance / initial_speed - distance / faster_speed = time_diff ∧
    initial_speed = 11.25 := by
  sorry

#check initial_walking_speed

end NUMINAMATH_CALUDE_initial_walking_speed_l2368_236879


namespace NUMINAMATH_CALUDE_sum_abcd_equals_negative_28_over_3_l2368_236813

theorem sum_abcd_equals_negative_28_over_3 
  (a b c d : ℚ) 
  (h : a + 3 = b + 7 ∧ a + 3 = c + 5 ∧ a + 3 = d + 9 ∧ a + 3 = a + b + c + d + 13) : 
  a + b + c + d = -28/3 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_negative_28_over_3_l2368_236813


namespace NUMINAMATH_CALUDE_andrew_winning_strategy_l2368_236841

/-- Represents the state of the game with two heaps of pebbles -/
structure GameState where
  a : ℕ
  b : ℕ

/-- Predicate to check if a number is of the form 2^x + 1 -/
def isPowerOfTwoPlusOne (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 2^x + 1

/-- Predicate to check if Andrew has a winning strategy -/
def andrewWins (state : GameState) : Prop :=
  state.a = 1 ∨ state.b = 1 ∨
  isPowerOfTwoPlusOne (state.a + state.b) ∨
  (isPowerOfTwoPlusOne state.a ∧ state.b < state.a) ∨
  (isPowerOfTwoPlusOne state.b ∧ state.a < state.b)

/-- The main theorem stating the winning condition for Andrew -/
theorem andrew_winning_strategy (state : GameState) :
  andrewWins state ↔ ∃ (strategy : GameState → ℕ → GameState),
    ∀ (move : ℕ), andrewWins (strategy state move) :=
  sorry

end NUMINAMATH_CALUDE_andrew_winning_strategy_l2368_236841


namespace NUMINAMATH_CALUDE_quadratic_one_positive_root_l2368_236880

theorem quadratic_one_positive_root (a : ℝ) : 
  (∃! x : ℝ, x > 0 ∧ x^2 - a*x + a - 2 = 0) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_positive_root_l2368_236880


namespace NUMINAMATH_CALUDE_area_not_covered_by_circles_l2368_236877

/-- The area of a square not covered by four inscribed circles -/
theorem area_not_covered_by_circles (square_side : ℝ) (circle_radius : ℝ) 
  (h1 : square_side = 10)
  (h2 : circle_radius = 5)
  (h3 : circle_radius * 2 = square_side) :
  square_side ^ 2 - 4 * Real.pi * circle_radius ^ 2 + 4 * Real.pi * circle_radius ^ 2 / 2 = 100 - 50 * Real.pi := by
  sorry

#check area_not_covered_by_circles

end NUMINAMATH_CALUDE_area_not_covered_by_circles_l2368_236877


namespace NUMINAMATH_CALUDE_f_properties_l2368_236828

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

-- Theorem statement
theorem f_properties :
  f (f 4) = 1/2 ∧ ∀ x, f x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2368_236828


namespace NUMINAMATH_CALUDE_square_difference_formula_expression_equivalence_l2368_236805

/-- The square difference formula -/
theorem square_difference_formula (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by sorry

/-- Proof that (x+y)(-x+y) is equivalent to y^2 - x^2 -/
theorem expression_equivalence (x y : ℝ) : (x + y) * (-x + y) = y^2 - x^2 := by sorry

end NUMINAMATH_CALUDE_square_difference_formula_expression_equivalence_l2368_236805


namespace NUMINAMATH_CALUDE_total_tables_is_40_l2368_236878

/-- Represents the number of tables and seating capacity in a restaurant --/
structure Restaurant where
  new_tables : ℕ
  original_tables : ℕ
  new_table_capacity : ℕ
  original_table_capacity : ℕ
  total_seating_capacity : ℕ

/-- The conditions of the restaurant problem --/
def restaurant_conditions (r : Restaurant) : Prop :=
  r.new_table_capacity = 6 ∧
  r.original_table_capacity = 4 ∧
  r.total_seating_capacity = 212 ∧
  r.new_tables = r.original_tables + 12 ∧
  r.new_tables * r.new_table_capacity + r.original_tables * r.original_table_capacity = r.total_seating_capacity

/-- The theorem stating that the total number of tables is 40 --/
theorem total_tables_is_40 (r : Restaurant) (h : restaurant_conditions r) : 
  r.new_tables + r.original_tables = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_tables_is_40_l2368_236878


namespace NUMINAMATH_CALUDE_largest_time_for_85_degrees_l2368_236866

/-- The temperature function in Denver, CO on a specific day -/
def temperature (t : ℝ) : ℝ := -t^2 + 10*t + 60

/-- The largest non-negative real solution to the equation temperature(t) = 85 is 15 -/
theorem largest_time_for_85_degrees :
  (∃ (t : ℝ), t ≥ 0 ∧ temperature t = 85) →
  (∀ (t : ℝ), t ≥ 0 ∧ temperature t = 85 → t ≤ 15) ∧
  (temperature 15 = 85) := by
sorry

end NUMINAMATH_CALUDE_largest_time_for_85_degrees_l2368_236866


namespace NUMINAMATH_CALUDE_negative_sqrt_geq_a_plus_sqrt_neg_two_l2368_236842

theorem negative_sqrt_geq_a_plus_sqrt_neg_two (a : ℝ) (h : a > 0) :
  -Real.sqrt a ≥ a + Real.sqrt (-2) :=
by sorry

end NUMINAMATH_CALUDE_negative_sqrt_geq_a_plus_sqrt_neg_two_l2368_236842


namespace NUMINAMATH_CALUDE_range_of_a_l2368_236851

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

-- Define the theorem
theorem range_of_a : 
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → -2 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2368_236851


namespace NUMINAMATH_CALUDE_polynomial_coefficient_B_l2368_236834

theorem polynomial_coefficient_B (A C D : ℤ) : 
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (r₁ : ℤ) + r₂ + r₃ + r₄ + r₅ + r₆ = 10 →
    ∀ (z : ℂ), z^6 - 10*z^5 + A*z^4 + (-108)*z^3 + C*z^2 + D*z + 16 = 
      (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_B_l2368_236834


namespace NUMINAMATH_CALUDE_calendar_sum_equality_l2368_236801

/-- A calendar with dates behind letters --/
structure Calendar where
  C : ℕ
  A : ℕ
  B : ℕ
  S : ℕ

/-- The calendar satisfies the given conditions --/
def valid_calendar (cal : Calendar) : Prop :=
  cal.A = cal.C + 3 ∧
  cal.B = cal.A + 10 ∧
  cal.S = cal.C + 16

theorem calendar_sum_equality (cal : Calendar) (h : valid_calendar cal) :
  cal.C + cal.S = cal.A + cal.B :=
by sorry

end NUMINAMATH_CALUDE_calendar_sum_equality_l2368_236801


namespace NUMINAMATH_CALUDE_farm_legs_count_l2368_236875

/-- The number of legs for a given animal type -/
def legs_per_animal (animal : String) : ℕ :=
  match animal with
  | "cow" => 4
  | "duck" => 2
  | _ => 0

/-- The total number of animals in the farm -/
def total_animals : ℕ := 15

/-- The number of cows in the farm -/
def num_cows : ℕ := 6

/-- The number of ducks in the farm -/
def num_ducks : ℕ := total_animals - num_cows

theorem farm_legs_count : 
  legs_per_animal "cow" * num_cows + legs_per_animal "duck" * num_ducks = 42 := by
sorry

end NUMINAMATH_CALUDE_farm_legs_count_l2368_236875


namespace NUMINAMATH_CALUDE_negation_of_all_squares_positive_l2368_236829

theorem negation_of_all_squares_positive :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, ¬(x^2 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_positive_l2368_236829


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2368_236881

theorem fractional_equation_solution :
  ∃! x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ (1 / (x - 1) + 1 = 2 / (x^2 - 1)) ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2368_236881


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_specific_intersection_l2368_236824

/-- The hyperbola C: x²/a² - y² = 1 (a > 0) -/
def C (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 = 1 ∧ a > 0

/-- The line l: x + y = 1 -/
def l (x y : ℝ) : Prop := x + y = 1

/-- P is the intersection point of line l and the y-axis -/
def P : ℝ × ℝ := (0, 1)

/-- A and B are distinct intersection points of C and l -/
def intersectionPoints (a : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ C a A.1 A.2 ∧ l A.1 A.2 ∧ C a B.1 B.2 ∧ l B.1 B.2

/-- PA = (5/12)PB -/
def vectorRelation (A B : ℝ × ℝ) : Prop :=
  (A.1 - P.1, A.2 - P.2) = (5/12 * (B.1 - P.1), 5/12 * (B.2 - P.2))

theorem hyperbola_line_intersection (a : ℝ) :
  intersectionPoints a ↔ (0 < a ∧ a < Real.sqrt 2 ∧ a ≠ 1) :=
sorry

theorem specific_intersection (a : ℝ) (A B : ℝ × ℝ) :
  C a A.1 A.2 ∧ l A.1 A.2 ∧ C a B.1 B.2 ∧ l B.1 B.2 ∧ vectorRelation A B →
  a = 17/13 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_specific_intersection_l2368_236824


namespace NUMINAMATH_CALUDE_two_correct_conclusions_l2368_236897

-- Define the type for analogical conclusions
inductive AnalogyConclusion
| ComplexRational
| VectorParallel
| PlanePlanar

-- Function to check if a conclusion is correct
def isCorrectConclusion (c : AnalogyConclusion) : Prop :=
  match c with
  | .ComplexRational => True
  | .VectorParallel => False
  | .PlanePlanar => True

-- Theorem statement
theorem two_correct_conclusions :
  (∃ (c1 c2 : AnalogyConclusion), c1 ≠ c2 ∧ 
    isCorrectConclusion c1 ∧ isCorrectConclusion c2 ∧
    (∀ (c3 : AnalogyConclusion), c3 ≠ c1 ∧ c3 ≠ c2 → ¬isCorrectConclusion c3)) :=
by sorry

end NUMINAMATH_CALUDE_two_correct_conclusions_l2368_236897


namespace NUMINAMATH_CALUDE_intersection_range_l2368_236886

-- Define the semicircle
def semicircle (x y : ℝ) : Prop := x^2 + y^2 = 9 ∧ y ≥ 0

-- Define the line
def line (k x y : ℝ) : Prop := y = k*(x-3) + 4

-- Define the condition for two distinct solutions
def has_two_distinct_solutions (k : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    semicircle x₁ y₁ ∧ semicircle x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂

-- Theorem statement
theorem intersection_range :
  ∀ k : ℝ, has_two_distinct_solutions k ↔ 7/24 < k ∧ k ≤ 2/3 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l2368_236886


namespace NUMINAMATH_CALUDE_stamp_cost_theorem_l2368_236827

/-- The total cost of stamps in cents -/
def total_cost (type_a_cost type_b_cost type_c_cost : ℕ) 
               (type_a_quantity type_b_quantity type_c_quantity : ℕ) : ℕ :=
  type_a_cost * type_a_quantity + 
  type_b_cost * type_b_quantity + 
  type_c_cost * type_c_quantity

/-- Theorem: The total cost of stamps is 594 cents -/
theorem stamp_cost_theorem : 
  total_cost 34 52 73 4 6 2 = 594 := by
  sorry

end NUMINAMATH_CALUDE_stamp_cost_theorem_l2368_236827


namespace NUMINAMATH_CALUDE_unique_two_digit_reverse_ratio_l2368_236810

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem unique_two_digit_reverse_ratio :
  ∃! n : ℕ, is_two_digit n ∧ (n : ℚ) / (reverse_digits n : ℚ) = 7 / 4 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_reverse_ratio_l2368_236810


namespace NUMINAMATH_CALUDE_dog_toy_discount_l2368_236882

/-- Proves that the discount on the second toy in each pair is $6.00 given the conditions --/
theorem dog_toy_discount (toy_price : ℝ) (num_toys : ℕ) (total_spent : ℝ) 
  (h1 : toy_price = 12)
  (h2 : num_toys = 4)
  (h3 : total_spent = 36) :
  (toy_price * num_toys - total_spent) / 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_dog_toy_discount_l2368_236882


namespace NUMINAMATH_CALUDE_count_monomials_l2368_236845

/-- A function that determines if an algebraic expression is a monomial -/
def isMonomial (expr : String) : Bool :=
  match expr with
  | "(m+n)/2" => false
  | "2x^2y" => true
  | "1/x" => false
  | "-5" => true
  | "a" => true
  | _ => false

/-- The set of given algebraic expressions -/
def expressions : List String := ["(m+n)/2", "2x^2y", "1/x", "-5", "a"]

/-- Theorem stating that the number of monomials in the given set of expressions is 3 -/
theorem count_monomials :
  (expressions.filter isMonomial).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_monomials_l2368_236845


namespace NUMINAMATH_CALUDE_fair_coin_probability_l2368_236887

def n : ℕ := 5
def k : ℕ := 2
def p : ℚ := 1/2

theorem fair_coin_probability : 
  (n.choose k) * p^k * (1 - p)^(n - k) = 10/32 := by sorry

end NUMINAMATH_CALUDE_fair_coin_probability_l2368_236887


namespace NUMINAMATH_CALUDE_minimum_seats_for_adjacent_seating_l2368_236839

/-- Represents a seating arrangement in a row of seats. -/
structure SeatingArrangement where
  total_seats : ℕ
  occupied_seats : ℕ
  max_gap : ℕ

/-- Checks if a seating arrangement is valid. -/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.occupied_seats ≤ s.total_seats ∧ 
  s.max_gap ≤ 2

/-- Checks if adding one more person would force them to sit next to someone. -/
def forces_adjacent_seating (s : SeatingArrangement) : Prop :=
  s.max_gap ≤ 1

/-- The main theorem to prove. -/
theorem minimum_seats_for_adjacent_seating :
  ∃ (s : SeatingArrangement),
    s.total_seats = 150 ∧
    s.occupied_seats = 30 ∧
    is_valid_arrangement s ∧
    forces_adjacent_seating s ∧
    (∀ (s' : SeatingArrangement),
      s'.total_seats = 150 →
      s'.occupied_seats < 30 →
      is_valid_arrangement s' →
      ¬forces_adjacent_seating s') :=
sorry

end NUMINAMATH_CALUDE_minimum_seats_for_adjacent_seating_l2368_236839


namespace NUMINAMATH_CALUDE_find_A_l2368_236847

theorem find_A (A B C D : ℝ) 
  (diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (eq1 : 2 * B + B = 12)
  (eq2 : C - B = 5)
  (eq3 : D + C = 12)
  (eq4 : A - D = 5) :
  A = 8 := by
sorry

end NUMINAMATH_CALUDE_find_A_l2368_236847


namespace NUMINAMATH_CALUDE_x_plus_y_equals_negative_27_l2368_236863

theorem x_plus_y_equals_negative_27 (x y : ℤ) 
  (h1 : x + 1 = y - 8) 
  (h2 : x = 2 * y) : 
  x + y = -27 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_negative_27_l2368_236863


namespace NUMINAMATH_CALUDE_path_area_and_cost_l2368_236895

/-- Calculates the area of a path surrounding a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per square meter -/
def construction_cost (path_area cost_per_sqm : ℝ) : ℝ :=
  path_area * cost_per_sqm

theorem path_area_and_cost (field_length field_width path_width cost_per_sqm : ℝ)
  (h1 : field_length = 65)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5)
  (h4 : cost_per_sqm = 2) :
  path_area field_length field_width path_width = 625 ∧
  construction_cost (path_area field_length field_width path_width) cost_per_sqm = 1250 := by
  sorry

end NUMINAMATH_CALUDE_path_area_and_cost_l2368_236895


namespace NUMINAMATH_CALUDE_sum_greater_than_three_l2368_236822

theorem sum_greater_than_three (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_ineq : a * b + b * c + c * a > a + b + c) : 
  a + b + c > 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_three_l2368_236822


namespace NUMINAMATH_CALUDE_chord_equation_of_ellipse_l2368_236872

/-- The equation of a line that forms a chord of the ellipse x^2/2 + y^2 = 1,
    bisected by the point (1/2, 1/2) -/
theorem chord_equation_of_ellipse (x y : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 / 2 + y₁^2 = 1) ∧
    (x₂^2 / 2 + y₂^2 = 1) ∧
    ((x₁ + x₂) / 2 = 1/2) ∧
    ((y₁ + y₂) / 2 = 1/2) ∧
    (y - y₁) = ((y₂ - y₁) / (x₂ - x₁)) * (x - x₁)) →
  2*x + 4*y - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_of_ellipse_l2368_236872


namespace NUMINAMATH_CALUDE_largest_class_size_l2368_236833

theorem largest_class_size (total_students : ℕ) (num_classes : ℕ) (diff : ℕ) : 
  total_students = 140 → num_classes = 5 → diff = 2 →
  ∃ x : ℕ, x = 32 ∧ 
    (x + (x - diff) + (x - 2*diff) + (x - 3*diff) + (x - 4*diff) = total_students) :=
by sorry

end NUMINAMATH_CALUDE_largest_class_size_l2368_236833


namespace NUMINAMATH_CALUDE_complex_sum_product_theorem_l2368_236835

theorem complex_sum_product_theorem (x y z : ℂ) 
  (hx : x = Complex.mk x.re x.im)
  (hy : y = Complex.mk y.re y.im)
  (hz : z = Complex.mk z.re z.im)
  (h_magnitude : Complex.abs x = Complex.abs y ∧ Complex.abs y = Complex.abs z)
  (h_sum : x + y + z = Complex.mk (-Real.sqrt 3 / 2) (-Real.sqrt 5))
  (h_product : x * y * z = Complex.mk (Real.sqrt 3) (Real.sqrt 5)) :
  (x.re * x.im + y.re * y.im + z.re * z.im)^2 = 15/1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_product_theorem_l2368_236835


namespace NUMINAMATH_CALUDE_manny_cookie_pies_l2368_236816

theorem manny_cookie_pies :
  ∀ (num_pies : ℕ) (num_classmates : ℕ) (num_teacher : ℕ) (slices_per_pie : ℕ) (slices_left : ℕ),
    num_classmates = 24 →
    num_teacher = 1 →
    slices_per_pie = 10 →
    slices_left = 4 →
    (num_pies * slices_per_pie = num_classmates + num_teacher + 1 + slices_left) →
    num_pies = 3 :=
by
  sorry

#check manny_cookie_pies

end NUMINAMATH_CALUDE_manny_cookie_pies_l2368_236816


namespace NUMINAMATH_CALUDE_max_intersections_math_city_l2368_236896

/-- Represents the number of streets in Math City -/
def total_streets : ℕ := 10

/-- Represents the number of parallel streets -/
def parallel_streets : ℕ := 2

/-- Represents the number of non-parallel streets -/
def non_parallel_streets : ℕ := total_streets - parallel_streets

/-- 
  Theorem: Maximum number of intersections in Math City
  Given:
  - There are 10 streets in total
  - Exactly 2 streets are parallel to each other
  - No other pair of streets is parallel
  - No three streets meet at a single point
  Prove: The maximum number of intersections is 44
-/
theorem max_intersections_math_city : 
  (non_parallel_streets.choose 2) + (parallel_streets * non_parallel_streets) = 44 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_math_city_l2368_236896


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2368_236838

/-- The complex number i -/
def i : ℂ := Complex.I

/-- Predicate for the condition (a + bi)^2 = 2i -/
def condition (a b : ℝ) : Prop := (Complex.mk a b)^2 = 2*i

/-- Statement: a=b=1 is sufficient but not necessary for (a + bi)^2 = 2i -/
theorem sufficient_not_necessary :
  (∀ a b : ℝ, a = 1 ∧ b = 1 → condition a b) ∧
  (∃ a b : ℝ, condition a b ∧ (a ≠ 1 ∨ b ≠ 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2368_236838


namespace NUMINAMATH_CALUDE_horner_method_v2_l2368_236891

def horner_polynomial (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

def horner_v0 : ℝ := 1

def horner_v1 (x : ℝ) : ℝ := horner_v0 * x + 5

def horner_v2 (x : ℝ) : ℝ := horner_v1 x * x + 10

theorem horner_method_v2 :
  horner_v2 2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v2_l2368_236891


namespace NUMINAMATH_CALUDE_quadratic_triple_root_relation_l2368_236859

/-- For a quadratic equation px^2 + qx + r = 0, if one root is triple the other, 
    then 3q^2 = 16pr -/
theorem quadratic_triple_root_relation (p q r : ℝ) (x₁ x₂ : ℝ) : 
  (p * x₁^2 + q * x₁ + r = 0) →
  (p * x₂^2 + q * x₂ + r = 0) →
  (x₂ = 3 * x₁) →
  (3 * q^2 = 16 * p * r) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_triple_root_relation_l2368_236859


namespace NUMINAMATH_CALUDE_chef_potato_usage_l2368_236809

/-- The number of potatoes used for lunch -/
def lunch_potatoes : ℕ := 5

/-- The number of potatoes used for dinner -/
def dinner_potatoes : ℕ := 2

/-- The total number of potatoes used -/
def total_potatoes : ℕ := lunch_potatoes + dinner_potatoes

theorem chef_potato_usage : total_potatoes = 7 := by
  sorry

end NUMINAMATH_CALUDE_chef_potato_usage_l2368_236809


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2368_236804

theorem rationalize_denominator :
  18 / (Real.sqrt 36 + Real.sqrt 2) = 54 / 17 - 9 * Real.sqrt 2 / 17 := by
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2368_236804


namespace NUMINAMATH_CALUDE_return_speed_calculation_l2368_236888

/-- Proves that given a round trip of 4 miles (2 miles each way), where the first half
    takes 1 hour and the average speed for the entire trip is 3 miles/hour,
    the speed for the second half of the trip is 6 miles/hour. -/
theorem return_speed_calculation (total_distance : ℝ) (outbound_distance : ℝ) 
    (outbound_time : ℝ) (average_speed : ℝ) :
  total_distance = 4 →
  outbound_distance = 2 →
  outbound_time = 1 →
  average_speed = 3 →
  ∃ (return_speed : ℝ), 
    return_speed = 6 ∧ 
    average_speed = total_distance / (outbound_time + outbound_distance / return_speed) := by
  sorry


end NUMINAMATH_CALUDE_return_speed_calculation_l2368_236888


namespace NUMINAMATH_CALUDE_dream_sequence_sum_l2368_236893

/-- A sequence is a "dream sequence" if it satisfies the given equation -/
def isDreamSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, 1 / a (n + 1) - 2 / a n = 0

theorem dream_sequence_sum (b : ℕ → ℝ) :
  (∀ n, b n > 0) →  -- b is a positive sequence
  isDreamSequence (λ n => 1 / b n) →  -- 1/b_n is a dream sequence
  b 1 + b 2 + b 3 = 2 →  -- sum of first three terms is 2
  b 6 + b 7 + b 8 = 64 :=  -- sum of 6th, 7th, and 8th terms is 64
by
  sorry

end NUMINAMATH_CALUDE_dream_sequence_sum_l2368_236893


namespace NUMINAMATH_CALUDE_inequality_proof_l2368_236812

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hln : Real.log a * Real.log b > 0) :
  a^(b - 1) < b^(a - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2368_236812


namespace NUMINAMATH_CALUDE_square_equality_solutions_l2368_236898

theorem square_equality_solutions (x : ℝ) : 
  (x + 1)^2 = (2*x - 1)^2 ↔ x = 0 ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_square_equality_solutions_l2368_236898


namespace NUMINAMATH_CALUDE_inconsistent_equations_l2368_236871

theorem inconsistent_equations : ¬∃ (x y S : ℝ), (x + y = S) ∧ (x + 3*y = 1) ∧ (x + 2*y = 10) := by
  sorry

end NUMINAMATH_CALUDE_inconsistent_equations_l2368_236871


namespace NUMINAMATH_CALUDE_polynomial_has_real_root_l2368_236855

/-- The polynomial in question -/
def polynomial (b x : ℝ) : ℝ := x^4 + b*x^3 - 2*x^2 + b*x + 2

/-- Theorem stating the condition for the polynomial to have at least one real root -/
theorem polynomial_has_real_root (b : ℝ) :
  (∃ x : ℝ, polynomial b x = 0) ↔ b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_has_real_root_l2368_236855


namespace NUMINAMATH_CALUDE_range_of_a_l2368_236849

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 > 0) → -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2368_236849


namespace NUMINAMATH_CALUDE_unique_solution_set_l2368_236830

def A (m : ℝ) : Set ℝ := {x : ℝ | m * x^2 + 2 * x + 3 = 0}

def M : Set ℝ := {m : ℝ | ∃! x : ℝ, m * x^2 + 2 * x + 3 = 0}

theorem unique_solution_set : M = {0, 1/3} := by sorry

end NUMINAMATH_CALUDE_unique_solution_set_l2368_236830


namespace NUMINAMATH_CALUDE_one_fourth_of_8_8_l2368_236889

theorem one_fourth_of_8_8 : 
  (8.8 : ℚ) / 4 = 11 / 5 := by sorry

end NUMINAMATH_CALUDE_one_fourth_of_8_8_l2368_236889


namespace NUMINAMATH_CALUDE_steven_apple_count_l2368_236817

/-- The number of apples Jake has -/
def jake_apples : ℕ := 11

/-- The difference between Jake's and Steven's apple count -/
def apple_difference : ℕ := 3

/-- Proves that Steven has 8 apples given the conditions -/
theorem steven_apple_count : ∃ (steven_apples : ℕ), steven_apples = jake_apples - apple_difference :=
  sorry

end NUMINAMATH_CALUDE_steven_apple_count_l2368_236817


namespace NUMINAMATH_CALUDE_quarter_difference_zero_l2368_236853

/-- Represents a coin collection with nickels, dimes, and quarters. -/
structure CoinCollection where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- The total number of coins in the collection. -/
def CoinCollection.total (c : CoinCollection) : ℕ :=
  c.nickels + c.dimes + c.quarters

/-- The total value of the collection in cents. -/
def CoinCollection.value (c : CoinCollection) : ℕ :=
  5 * c.nickels + 10 * c.dimes + 25 * c.quarters

/-- Predicate for a valid coin collection according to the problem conditions. -/
def isValidCollection (c : CoinCollection) : Prop :=
  c.total = 150 ∧ c.value = 2000

/-- The theorem to be proved. -/
theorem quarter_difference_zero :
  ∀ c₁ c₂ : CoinCollection, isValidCollection c₁ → isValidCollection c₂ →
  c₁.quarters = c₂.quarters :=
sorry

end NUMINAMATH_CALUDE_quarter_difference_zero_l2368_236853


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_9689_l2368_236843

theorem largest_prime_factor_of_9689 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 9689 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 9689 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_9689_l2368_236843


namespace NUMINAMATH_CALUDE_farm_animals_l2368_236867

theorem farm_animals (sheep ducks : ℕ) : 
  sheep + ducks = 15 → 
  4 * sheep + 2 * ducks = 22 + 2 * (sheep + ducks) → 
  sheep = 11 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l2368_236867


namespace NUMINAMATH_CALUDE_mia_wall_paint_area_l2368_236874

/-- The area to be painted on Mia's wall --/
def areaToBePainted (wallHeight wallLength unPaintedWidth unPaintedHeight : ℝ) : ℝ :=
  wallHeight * wallLength - unPaintedWidth * unPaintedHeight

/-- Theorem stating the area Mia needs to paint --/
theorem mia_wall_paint_area :
  areaToBePainted 10 15 3 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_mia_wall_paint_area_l2368_236874


namespace NUMINAMATH_CALUDE_solution_composition_l2368_236861

/-- Represents the initial percentage of liquid X in the solution -/
def initial_percentage : ℝ := 30

/-- The initial weight of the solution in kg -/
def initial_weight : ℝ := 10

/-- The weight of water that evaporates in kg -/
def evaporated_water : ℝ := 2

/-- The weight of the original solution added back in kg -/
def added_solution : ℝ := 2

/-- The final percentage of liquid X in the new solution -/
def final_percentage : ℝ := 36

theorem solution_composition :
  let remaining_weight := initial_weight - evaporated_water
  let new_total_weight := remaining_weight + added_solution
  let initial_liquid_x := initial_percentage / 100 * initial_weight
  let added_liquid_x := initial_percentage / 100 * added_solution
  let total_liquid_x := initial_liquid_x + added_liquid_x
  total_liquid_x / new_total_weight * 100 = final_percentage :=
by sorry

end NUMINAMATH_CALUDE_solution_composition_l2368_236861


namespace NUMINAMATH_CALUDE_inequality_proof_l2368_236890

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) :
  2 * a * b * Real.log (b / a) < b^2 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2368_236890


namespace NUMINAMATH_CALUDE_multiply_three_negative_two_l2368_236894

theorem multiply_three_negative_two : 3 * (-2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_negative_two_l2368_236894


namespace NUMINAMATH_CALUDE_num_biology_books_is_15_l2368_236864

def num_chemistry_books : ℕ := 8
def total_ways_to_pick : ℕ := 2940

-- Function to calculate combinations
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem to prove
theorem num_biology_books_is_15 : 
  ∃ (B : ℕ), choose B 2 * choose num_chemistry_books 2 = total_ways_to_pick ∧ B = 15 :=
by sorry

end NUMINAMATH_CALUDE_num_biology_books_is_15_l2368_236864


namespace NUMINAMATH_CALUDE_circle_on_parabola_tangent_to_directrix_and_yaxis_l2368_236802

/-- A circle centered on a parabola and tangent to its directrix and the y-axis -/
theorem circle_on_parabola_tangent_to_directrix_and_yaxis :
  ∀ (x₀ : ℝ) (y₀ : ℝ) (r : ℝ),
  x₀ = 1 ∨ x₀ = -1 →
  y₀ = (1/2) * x₀^2 →
  r = 1 →
  (∀ (x y : ℝ), (x - x₀)^2 + (y - y₀)^2 = r^2 →
    ∃ (t : ℝ), x = t ∧ y = (1/2) * t^2) ∧
  (∃ (x y : ℝ), (x - x₀)^2 + (y - y₀)^2 = r^2 ∧ y = -(1/2)) ∧
  (∃ (y : ℝ), x₀^2 + (y - y₀)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_on_parabola_tangent_to_directrix_and_yaxis_l2368_236802


namespace NUMINAMATH_CALUDE_red_balls_count_l2368_236823

/-- Given a bag with 2400 balls of red, green, and blue colors,
    where the ratio of red:green:blue is 15:13:17,
    prove that the number of red balls is 795. -/
theorem red_balls_count (total : ℕ) (red green blue : ℕ) :
  total = 2400 →
  red + green + blue = 45 →
  red = 15 →
  green = 13 →
  blue = 17 →
  red * (total / (red + green + blue)) = 795 := by
  sorry


end NUMINAMATH_CALUDE_red_balls_count_l2368_236823


namespace NUMINAMATH_CALUDE_grid_value_theorem_l2368_236821

/-- Represents a 7x2 grid of rational numbers -/
def Grid := Fin 7 → Fin 2 → ℚ

/-- The main column forms an arithmetic sequence -/
def is_main_column_arithmetic (g : Grid) : Prop :=
  ∃ d : ℚ, ∀ i : Fin 6, g (i + 1) 0 - g i 0 = d

/-- The first two rows form arithmetic sequences -/
def are_first_two_rows_arithmetic (g : Grid) : Prop :=
  ∃ d₁ d₂ : ℚ, (g 0 1 - g 0 0 = d₁) ∧ (g 1 1 - g 1 0 = d₂)

/-- The grid satisfies the given conditions -/
def satisfies_conditions (g : Grid) : Prop :=
  (g 0 0 = -9) ∧ (g 3 0 = 56) ∧ (g 6 1 = 16) ∧
  is_main_column_arithmetic g ∧
  are_first_two_rows_arithmetic g

theorem grid_value_theorem (g : Grid) (h : satisfies_conditions g) : g 4 1 = -851/3 := by
  sorry

end NUMINAMATH_CALUDE_grid_value_theorem_l2368_236821


namespace NUMINAMATH_CALUDE_builder_boards_count_l2368_236868

/-- The number of boards in each package -/
def boards_per_package : ℕ := 3

/-- The number of packages the builder needs to buy -/
def packages_needed : ℕ := 52

/-- The total number of boards needed -/
def total_boards : ℕ := boards_per_package * packages_needed

theorem builder_boards_count : total_boards = 156 := by
  sorry

end NUMINAMATH_CALUDE_builder_boards_count_l2368_236868


namespace NUMINAMATH_CALUDE_vector_at_minus_2_l2368_236858

/-- A line in a plane parametrized by s -/
def line (s : ℝ) : ℝ × ℝ := sorry

/-- The vector on the line at s = 1 is (2, 5) -/
axiom vector_at_1 : line 1 = (2, 5)

/-- The vector on the line at s = 4 is (8, -7) -/
axiom vector_at_4 : line 4 = (8, -7)

/-- The vector on the line at s = -2 is (-4, 17) -/
theorem vector_at_minus_2 : line (-2) = (-4, 17) := by sorry

end NUMINAMATH_CALUDE_vector_at_minus_2_l2368_236858


namespace NUMINAMATH_CALUDE_special_sequence_2016th_term_l2368_236865

/-- A sequence with specific properties -/
def special_sequence (a : ℕ → ℝ) : Prop :=
  a 4 = 1 ∧ 
  a 11 = 9 ∧ 
  ∀ n : ℕ, a n + a (n + 1) + a (n + 2) = 15

/-- The 2016th term of the special sequence is 5 -/
theorem special_sequence_2016th_term (a : ℕ → ℝ) 
  (h : special_sequence a) : a 2016 = 5 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_2016th_term_l2368_236865


namespace NUMINAMATH_CALUDE_abc_mod_9_l2368_236826

theorem abc_mod_9 (a b c : ℕ) (ha : a < 9) (hb : b < 9) (hc : c < 9)
  (h1 : (a + 3*b + 2*c) % 9 = 0)
  (h2 : (2*a + 2*b + 3*c) % 9 = 3)
  (h3 : (3*a + b + 2*c) % 9 = 6) :
  (a * b * c) % 9 = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_mod_9_l2368_236826


namespace NUMINAMATH_CALUDE_additional_donation_amount_l2368_236850

/-- A proof that the additional donation was $20.00 given the conditions of the raffle ticket sale --/
theorem additional_donation_amount (num_tickets : ℕ) (ticket_price : ℚ) (num_fixed_donations : ℕ) (fixed_donation_amount : ℚ) (total_raised : ℚ) : 
  num_tickets = 25 →
  ticket_price = 2 →
  num_fixed_donations = 2 →
  fixed_donation_amount = 15 →
  total_raised = 100 →
  total_raised - (↑num_tickets * ticket_price + ↑num_fixed_donations * fixed_donation_amount) = 20 :=
by
  sorry

#check additional_donation_amount

end NUMINAMATH_CALUDE_additional_donation_amount_l2368_236850


namespace NUMINAMATH_CALUDE_two_lines_forming_angle_with_skew_lines_l2368_236856

/-- Represents a line in 3D space -/
structure Line3D where
  -- We'll use a simplified representation of a line
  -- More details could be added if needed

/-- Represents a point in 3D space -/
structure Point3D where
  -- We'll use a simplified representation of a point
  -- More details could be added if needed

/-- The angle between two lines -/
def angle_between_lines (l1 l2 : Line3D) : ℝ :=
  sorry

/-- Whether two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- Whether a line passes through a point -/
def line_passes_through (l : Line3D) (p : Point3D) : Prop :=
  sorry

/-- The main theorem -/
theorem two_lines_forming_angle_with_skew_lines 
  (a b : Line3D) (P : Point3D) 
  (h_skew : are_skew a b) 
  (h_angle : angle_between_lines a b = 50) : 
  ∃! (s : Finset Line3D), 
    s.card = 2 ∧ 
    ∀ l ∈ s, line_passes_through l P ∧ 
              angle_between_lines l a = 30 ∧ 
              angle_between_lines l b = 30 :=
sorry

end NUMINAMATH_CALUDE_two_lines_forming_angle_with_skew_lines_l2368_236856


namespace NUMINAMATH_CALUDE_tangent_line_passes_through_fixed_point_l2368_236807

/-- The parabola Γ -/
def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 = 4 * p.2}

/-- The point P from which tangents are drawn -/
def P (m : ℝ) : ℝ × ℝ := (m, -4)

/-- The fixed point through which AB always passes -/
def fixedPoint : ℝ × ℝ := (0, 4)

/-- Theorem stating that AB always passes through the fixed point -/
theorem tangent_line_passes_through_fixed_point (m : ℝ) :
  ∀ A B : ℝ × ℝ,
  A ∈ Γ → B ∈ Γ →
  (∃ t : ℝ, A = (1 - t) • P m + t • B) →
  (∃ s : ℝ, B = (1 - s) • P m + s • A) →
  ∃ r : ℝ, fixedPoint = (1 - r) • A + r • B :=
sorry

end NUMINAMATH_CALUDE_tangent_line_passes_through_fixed_point_l2368_236807


namespace NUMINAMATH_CALUDE_M_mod_1000_l2368_236848

/-- The number of characters in the string -/
def n : ℕ := 15

/-- The number of A's in the string -/
def a : ℕ := 3

/-- The number of B's in the string -/
def b : ℕ := 5

/-- The number of C's in the string -/
def c : ℕ := 4

/-- The number of D's in the string -/
def d : ℕ := 3

/-- The length of the first section where A's are not allowed -/
def first_section : ℕ := 3

/-- The length of the middle section where B's are not allowed -/
def middle_section : ℕ := 5

/-- The length of the last section where C's are not allowed -/
def last_section : ℕ := 7

/-- The function that calculates the number of permutations -/
def M : ℕ := sorry

theorem M_mod_1000 : M % 1000 = 60 := by sorry

end NUMINAMATH_CALUDE_M_mod_1000_l2368_236848


namespace NUMINAMATH_CALUDE_negation_of_all_linear_functions_are_monotonic_l2368_236892

-- Define the type of functions from real numbers to real numbers
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be linear
def IsLinear (f : RealFunction) : Prop := ∀ x y : ℝ, ∀ c : ℝ, f (c * x + y) = c * f x + f y

-- Define what it means for a function to be monotonic
def IsMonotonic (f : RealFunction) : Prop := ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- State the theorem
theorem negation_of_all_linear_functions_are_monotonic :
  (¬ ∀ f : RealFunction, IsLinear f → IsMonotonic f) ↔
  (∃ f : RealFunction, IsLinear f ∧ ¬IsMonotonic f) :=
sorry

end NUMINAMATH_CALUDE_negation_of_all_linear_functions_are_monotonic_l2368_236892


namespace NUMINAMATH_CALUDE_distinct_reals_with_integer_differences_are_integers_l2368_236857

theorem distinct_reals_with_integer_differences_are_integers 
  (a b : ℝ) 
  (distinct : a ≠ b) 
  (int_diff : ∀ k : ℕ, ∃ n : ℤ, a^k - b^k = n) : 
  ∃ m n : ℤ, (a : ℝ) = m ∧ (b : ℝ) = n := by
  sorry

end NUMINAMATH_CALUDE_distinct_reals_with_integer_differences_are_integers_l2368_236857


namespace NUMINAMATH_CALUDE_stratified_sample_sum_l2368_236883

def total_population : Nat := 100
def sample_size : Nat := 20
def stratum1_size : Nat := 10
def stratum2_size : Nat := 20

theorem stratified_sample_sum :
  let stratum1_sample := sample_size * stratum1_size / total_population
  let stratum2_sample := sample_size * stratum2_size / total_population
  stratum1_sample + stratum2_sample = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_sum_l2368_236883


namespace NUMINAMATH_CALUDE_min_values_theorem_l2368_236852

theorem min_values_theorem :
  (∀ x > 1, x + 4 / (x - 1) ≥ 5) ∧
  (∀ a b, a > 0 → b > 0 → a + b = a * b → 9 * a + b ≥ 16) := by
sorry

end NUMINAMATH_CALUDE_min_values_theorem_l2368_236852


namespace NUMINAMATH_CALUDE_remaining_card_theorem_l2368_236844

/-- Definition of the operation sequence on a stack of cards -/
def operationSequence (n : ℕ) : List ℕ :=
  sorry

/-- L(n) is the number on the remaining card after performing the operation sequence -/
def L (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating the form of k for which L(3k) = k -/
theorem remaining_card_theorem (k : ℕ) :
  (L (3 * k) = k) ↔ 
  (∃ j : ℕ, (k = (2 * 3^(6*j) - 2) / 7) ∨ (k = (3^(6*j + 2) - 2) / 7)) :=
by sorry

end NUMINAMATH_CALUDE_remaining_card_theorem_l2368_236844


namespace NUMINAMATH_CALUDE_bankers_discount_calculation_l2368_236854

/-- Calculates the banker's discount for a given period and rate -/
def bankers_discount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Theorem: The banker's discount for the given conditions is 18900 -/
theorem bankers_discount_calculation (principal : ℝ) 
  (rate1 rate2 rate3 : ℝ) (time1 time2 time3 : ℝ) :
  principal = 180000 ∧ 
  rate1 = 0.12 ∧ rate2 = 0.14 ∧ rate3 = 0.16 ∧
  time1 = 0.25 ∧ time2 = 0.25 ∧ time3 = 0.25 →
  bankers_discount principal rate1 time1 + 
  bankers_discount principal rate2 time2 + 
  bankers_discount principal rate3 time3 = 18900 := by
  sorry

#eval bankers_discount 180000 0.12 0.25 + 
      bankers_discount 180000 0.14 0.25 + 
      bankers_discount 180000 0.16 0.25

end NUMINAMATH_CALUDE_bankers_discount_calculation_l2368_236854


namespace NUMINAMATH_CALUDE_product_sign_l2368_236876

theorem product_sign (a b c d e : ℝ) : ab^2*c^3*d^4*e^5 < 0 → ab^2*c*d^4*e < 0 := by
  sorry

end NUMINAMATH_CALUDE_product_sign_l2368_236876


namespace NUMINAMATH_CALUDE_largest_coeff_x5_implies_n10_l2368_236885

theorem largest_coeff_x5_implies_n10 (n : ℕ+) :
  (∀ k : ℕ, k ≠ 5 → Nat.choose n 5 ≥ Nat.choose n k) →
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_largest_coeff_x5_implies_n10_l2368_236885


namespace NUMINAMATH_CALUDE_cookies_in_bag_l2368_236836

/-- The number of cookies that can fit in one paper bag given a total number of cookies and bags -/
def cookies_per_bag (total_cookies : ℕ) (total_bags : ℕ) : ℕ :=
  (total_cookies / total_bags : ℕ)

/-- Theorem stating that given 292 cookies and 19 paper bags, one bag can hold at most 15 cookies -/
theorem cookies_in_bag : cookies_per_bag 292 19 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_bag_l2368_236836


namespace NUMINAMATH_CALUDE_brown_gumdrops_after_replacement_l2368_236815

/-- Theorem about the number of brown gumdrops after replacement in a jar --/
theorem brown_gumdrops_after_replacement (total : ℕ) (green blue brown red yellow : ℕ) :
  total = 200 →
  green = 40 →
  blue = 50 →
  brown = 60 →
  red = 20 →
  yellow = 30 →
  (brown + (red / 3 : ℕ)) = 67 := by
  sorry

#check brown_gumdrops_after_replacement

end NUMINAMATH_CALUDE_brown_gumdrops_after_replacement_l2368_236815


namespace NUMINAMATH_CALUDE_part1_part2_l2368_236811

-- Define the function f
def f (a x : ℝ) : ℝ := 2 * a * x^2 - (a^2 + 4) * x + 2 * a

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x, f a x > 0 ↔ -4 < x ∧ x < -1/4) → (a = -8 ∨ a = -1/2) :=
sorry

-- Part 2
theorem part2 (a : ℝ) (h : a > 0) :
  (∀ x, f a x ≤ 0 ↔ 
    ((0 < a ∧ a < 2 → a/2 ≤ x ∧ x ≤ 2/a) ∧
     (a > 2 → 2/a ≤ x ∧ x ≤ a/2) ∧
     (a = 2 → x = 1))) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l2368_236811


namespace NUMINAMATH_CALUDE_round_trip_with_car_percentage_l2368_236869

/-- The percentage of passengers with round-trip tickets who did not take their cars -/
def no_car_percentage : ℝ := 60

/-- The percentage of all passengers who held round-trip tickets -/
def round_trip_percentage : ℝ := 62.5

/-- The theorem to prove -/
theorem round_trip_with_car_percentage :
  (100 - no_car_percentage) * round_trip_percentage / 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_with_car_percentage_l2368_236869


namespace NUMINAMATH_CALUDE_sunglasses_cap_probability_l2368_236899

theorem sunglasses_cap_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (total_hats : ℕ) 
  (prob_cap_and_sunglasses : ℚ) 
  (h1 : total_sunglasses = 80) 
  (h2 : total_caps = 60) 
  (h3 : total_hats = 40) 
  (h4 : prob_cap_and_sunglasses = 1/3) :
  (total_caps * prob_cap_and_sunglasses) / total_sunglasses = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_cap_probability_l2368_236899


namespace NUMINAMATH_CALUDE_unique_solution_l2368_236803

def is_valid_digit (d : ℕ) : Prop := d > 0 ∧ d ≤ 9

def are_distinct (a b c d e f g : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g

def to_six_digit_number (a b c d e f : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f

theorem unique_solution :
  ∀ A B : ℕ,
    is_valid_digit A →
    is_valid_digit B →
    are_distinct 1 2 3 4 5 A B →
    (to_six_digit_number A 1 2 3 4 5) % B = 0 →
    (to_six_digit_number 1 2 3 4 5 A) % B = 0 →
    A = 9 ∧ B = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2368_236803


namespace NUMINAMATH_CALUDE_set_A_theorem_l2368_236860

def A (a : ℝ) := {x : ℝ | 2 * x + a > 0}

theorem set_A_theorem (a : ℝ) :
  (1 ∉ A a) → (2 ∈ A a) → -4 < a ∧ a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_set_A_theorem_l2368_236860


namespace NUMINAMATH_CALUDE_cow_calf_total_cost_l2368_236837

theorem cow_calf_total_cost (cow_cost calf_cost : ℕ) 
  (h1 : cow_cost = 880)
  (h2 : calf_cost = 110)
  (h3 : cow_cost = 8 * calf_cost) : 
  cow_cost + calf_cost = 990 := by
  sorry

end NUMINAMATH_CALUDE_cow_calf_total_cost_l2368_236837


namespace NUMINAMATH_CALUDE_twentieth_digit_sum_one_thirteenth_one_eleventh_l2368_236884

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sumDecimalRepresentations (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in a decimal representation -/
def nthDigitAfterDecimal (f : ℕ → ℕ) (n : ℕ) : ℕ := sorry

theorem twentieth_digit_sum_one_thirteenth_one_eleventh :
  nthDigitAfterDecimal (sumDecimalRepresentations (1/13) (1/11)) 20 = 6 := by sorry

end NUMINAMATH_CALUDE_twentieth_digit_sum_one_thirteenth_one_eleventh_l2368_236884


namespace NUMINAMATH_CALUDE_grade12_population_l2368_236870

/-- Represents the number of students in each grade (10, 11, 12) -/
structure GradePopulation where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- The ratio of students in grades 10, 11, and 12 -/
def gradeRatio : GradePopulation := ⟨10, 8, 7⟩

/-- The number of students sampled -/
def sampleSize : ℕ := 200

/-- The sampling probability for each student -/
def samplingProbability : ℚ := 1/5

theorem grade12_population (pop : GradePopulation) :
  pop.grade10 / gradeRatio.grade10 = pop.grade11 / gradeRatio.grade11 ∧
  pop.grade11 / gradeRatio.grade11 = pop.grade12 / gradeRatio.grade12 ∧
  pop.grade10 + pop.grade11 + pop.grade12 = sampleSize / samplingProbability →
  pop.grade12 = 280 := by
sorry

end NUMINAMATH_CALUDE_grade12_population_l2368_236870


namespace NUMINAMATH_CALUDE_system_solution_l2368_236814

theorem system_solution : ∃ (x y : ℝ), 
  x^2 + y * Real.sqrt (x * y) = 105 ∧
  y^2 + x * Real.sqrt (x * y) = 70 ∧
  x = 9 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2368_236814


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2368_236800

theorem quadratic_inequality_solution (a : ℝ) :
  let solution_set := {x : ℝ | a * x^2 - (a + 3) * x + 3 ≤ 0}
  (a < 0 → solution_set = {x : ℝ | x ≤ 3/a ∨ x ≥ 1}) ∧
  (a = 0 → solution_set = {x : ℝ | x ≥ 1}) ∧
  (0 < a ∧ a < 3 → solution_set = {x : ℝ | 1 ≤ x ∧ x ≤ 3/a}) ∧
  (a = 3 → solution_set = {1}) ∧
  (a > 3 → solution_set = {x : ℝ | 3/a ≤ x ∧ x ≤ 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2368_236800


namespace NUMINAMATH_CALUDE_friend_riding_area_l2368_236820

/-- Given a rectangular riding area of width 2 and length 3, 
    prove that another area 4 times larger is 24 square blocks. -/
theorem friend_riding_area (width : ℕ) (length : ℕ) (multiplier : ℕ) : 
  width = 2 → length = 3 → multiplier = 4 → 
  (width * length * multiplier : ℕ) = 24 := by
  sorry

end NUMINAMATH_CALUDE_friend_riding_area_l2368_236820


namespace NUMINAMATH_CALUDE_total_graduation_messages_l2368_236808

def number_of_students : ℕ := 40

theorem total_graduation_messages :
  (number_of_students * (number_of_students - 1)) / 2 = 1560 :=
by sorry

end NUMINAMATH_CALUDE_total_graduation_messages_l2368_236808


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2368_236840

-- Define the expression as a function of x
def f (x : ℝ) : ℝ := (x + 1) * (x - 1) + x * (2 - x) + (x - 1)^2

-- Theorem stating the simplification and evaluation
theorem simplify_and_evaluate :
  (∀ x : ℝ, f x = x^2) ∧ (f 100 = 10000) := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2368_236840


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2368_236831

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 3 / 2) : 
  let e := Real.sqrt (1 + (b / a) ^ 2)
  e = Real.sqrt 13 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2368_236831


namespace NUMINAMATH_CALUDE_max_distance_ellipse_point_l2368_236832

/-- 
Given an ellipse x²/a² + y²/b² = 1 with a > b > 0, and A(0, b),
the maximum value of |PA| for any point P on the ellipse is max(a²/√(a² - b²), 2b).
-/
theorem max_distance_ellipse_point (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let ellipse := {P : ℝ × ℝ | (P.1^2 / a^2) + (P.2^2 / b^2) = 1}
  let A := (0, b)
  let dist_PA (P : ℝ × ℝ) := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  (∀ P ∈ ellipse, dist_PA P ≤ max (a^2 / Real.sqrt (a^2 - b^2)) (2*b)) ∧
  (∃ P ∈ ellipse, dist_PA P = max (a^2 / Real.sqrt (a^2 - b^2)) (2*b))
:= by sorry

end NUMINAMATH_CALUDE_max_distance_ellipse_point_l2368_236832


namespace NUMINAMATH_CALUDE_special_numbers_characterization_l2368_236825

/-- A function that returns true if a natural number has all distinct digits -/
def has_distinct_digits (n : ℕ) : Bool :=
  sorry

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

/-- A function that returns the product of digits of a natural number -/
def product_of_digits (n : ℕ) : ℕ :=
  sorry

/-- The set of numbers that satisfy the conditions -/
def special_numbers : Finset ℕ :=
  {123, 132, 213, 231, 312, 321}

theorem special_numbers_characterization :
  ∀ n : ℕ, n ∈ special_numbers ↔
    n > 9 ∧
    has_distinct_digits n ∧
    sum_of_digits n = product_of_digits n :=
by sorry

end NUMINAMATH_CALUDE_special_numbers_characterization_l2368_236825


namespace NUMINAMATH_CALUDE_max_distance_squared_l2368_236818

theorem max_distance_squared (x y : ℝ) : 
  (x + 2)^2 + (y - 5)^2 = 9 → 
  ∃ (max : ℝ), max = 64 ∧ ∀ (x' y' : ℝ), (x' + 2)^2 + (y' - 5)^2 = 9 → (x' - 1)^2 + (y' - 1)^2 ≤ max := by
sorry

end NUMINAMATH_CALUDE_max_distance_squared_l2368_236818


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2368_236862

/-- 
Prove that an arithmetic sequence with the given properties has 15 terms.
-/
theorem arithmetic_sequence_length :
  ∀ (a l d : ℤ) (n : ℕ),
  a = -5 →  -- First term
  l = 65 →  -- Last term
  d = 5 →   -- Common difference
  l = a + (n - 1) * d →  -- Arithmetic sequence formula
  n = 15 :=  -- Number of terms
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2368_236862


namespace NUMINAMATH_CALUDE_meaningful_fraction_l2368_236819

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l2368_236819


namespace NUMINAMATH_CALUDE_min_a_squared_plus_b_squared_l2368_236873

theorem min_a_squared_plus_b_squared : ∀ a b : ℝ,
  (∀ x : ℝ, x^2 + a*x + b - 3 = 0 → x = 2) →
  a^2 + b^2 ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_a_squared_plus_b_squared_l2368_236873


namespace NUMINAMATH_CALUDE_and_false_necessary_not_sufficient_for_or_false_l2368_236846

theorem and_false_necessary_not_sufficient_for_or_false (p q : Prop) :
  (¬(p ∧ q) → ¬(p ∨ q)) ∧ ¬(¬(p ∧ q) ↔ ¬(p ∨ q)) := by
  sorry

end NUMINAMATH_CALUDE_and_false_necessary_not_sufficient_for_or_false_l2368_236846
