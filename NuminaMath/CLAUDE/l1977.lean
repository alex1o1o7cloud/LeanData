import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l1977_197796

theorem inequality_proof (x y z : ℝ) (hx : 0 < x ∧ x < π/2) (hy : 0 < y ∧ y < π/2) (hz : 0 < z ∧ z < π/2) :
  (x * Real.cos x + y * Real.cos y + z * Real.cos z) / (x + y + z) ≤ (Real.cos x + Real.cos y + Real.cos z) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1977_197796


namespace NUMINAMATH_CALUDE_junior_prom_attendance_l1977_197705

theorem junior_prom_attendance :
  ∀ (total_kids : ℕ),
    (total_kids / 4 : ℕ) = 25 + 10 →
    total_kids = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_junior_prom_attendance_l1977_197705


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l1977_197766

/-- An ellipse with given major axis and eccentricity -/
structure Ellipse where
  major_axis : ℝ
  eccentricity : ℝ

/-- The standard equation of an ellipse -/
inductive StandardEquation where
  | x_axis : StandardEquation
  | y_axis : StandardEquation

/-- Theorem: For an ellipse with major axis 8 and eccentricity 3/4, 
    its standard equation is either (x²/16) + (y²/7) = 1 or (x²/7) + (y²/16) = 1 -/
theorem ellipse_standard_equation (e : Ellipse) 
  (h1 : e.major_axis = 8) 
  (h2 : e.eccentricity = 3/4) :
  ∃ (eq : StandardEquation), 
    (eq = StandardEquation.x_axis → ∀ (x y : ℝ), x^2/16 + y^2/7 = 1) ∧ 
    (eq = StandardEquation.y_axis → ∀ (x y : ℝ), x^2/7 + y^2/16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l1977_197766


namespace NUMINAMATH_CALUDE_parentheses_value_l1977_197711

theorem parentheses_value : (6 : ℝ) / Real.sqrt 18 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_value_l1977_197711


namespace NUMINAMATH_CALUDE_janes_bagels_l1977_197706

theorem janes_bagels (b m : ℕ) : 
  b + m = 6 →
  (55 * b + 80 * m) % 100 = 0 →
  b = 0 := by
sorry

end NUMINAMATH_CALUDE_janes_bagels_l1977_197706


namespace NUMINAMATH_CALUDE_missing_element_is_loop_l1977_197775

-- Define the basic elements of a flowchart
inductive FlowchartElement
| Input
| Output
| Condition
| Loop

-- Define the program structures
inductive ProgramStructure
| Sequence
| Condition
| Loop

-- Define the known basic elements
def known_elements : List FlowchartElement := [FlowchartElement.Input, FlowchartElement.Output, FlowchartElement.Condition]

-- Define the program structures
def program_structures : List ProgramStructure := [ProgramStructure.Sequence, ProgramStructure.Condition, ProgramStructure.Loop]

-- Theorem: The missing basic element of a flowchart is Loop
theorem missing_element_is_loop : 
  ∃ (e : FlowchartElement), e ∉ known_elements ∧ e = FlowchartElement.Loop :=
sorry

end NUMINAMATH_CALUDE_missing_element_is_loop_l1977_197775


namespace NUMINAMATH_CALUDE_random_walk_properties_l1977_197753

/-- Represents a random walk on a line. -/
structure RandomWalk where
  a : ℕ  -- number of steps to the right
  b : ℕ  -- number of steps to the left
  h : a > b

/-- The maximum possible range of a random walk. -/
def max_range (w : RandomWalk) : ℕ := w.a

/-- The minimum possible range of a random walk. -/
def min_range (w : RandomWalk) : ℕ := w.a - w.b

/-- The number of sequences that achieve the maximum range. -/
def max_range_sequences (w : RandomWalk) : ℕ := w.b + 1

/-- Theorem stating the properties of the random walk. -/
theorem random_walk_properties (w : RandomWalk) :
  (max_range w = w.a) ∧
  (min_range w = w.a - w.b) ∧
  (max_range_sequences w = w.b + 1) := by
  sorry


end NUMINAMATH_CALUDE_random_walk_properties_l1977_197753


namespace NUMINAMATH_CALUDE_qin_jiushao_count_for_specific_polynomial_l1977_197761

/-- The "Qin Jiushao" algorithm for polynomial evaluation -/
def qin_jiushao_eval (coeffs : List ℝ) (x : ℝ) : ℝ := sorry

/-- Counts the number of multiplications and additions in the "Qin Jiushao" algorithm -/
def qin_jiushao_count (coeffs : List ℝ) : (ℕ × ℕ) := sorry

theorem qin_jiushao_count_for_specific_polynomial :
  let coeffs := [5, 4, 3, 2, 1, 1]
  qin_jiushao_count coeffs = (5, 5) := by sorry

end NUMINAMATH_CALUDE_qin_jiushao_count_for_specific_polynomial_l1977_197761


namespace NUMINAMATH_CALUDE_remove_parentheses_l1977_197788

theorem remove_parentheses (x y z : ℝ) : -(x - (y - z)) = -x + y - z := by
  sorry

end NUMINAMATH_CALUDE_remove_parentheses_l1977_197788


namespace NUMINAMATH_CALUDE_train_speed_l1977_197707

/-- Calculate the speed of a train given its length and time to pass an observer -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 180) (h2 : time = 9) :
  length / time = 20 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l1977_197707


namespace NUMINAMATH_CALUDE_peter_bird_count_l1977_197729

/-- The fraction of birds that are ducks -/
def duck_fraction : ℚ := 1/3

/-- The cost of chicken feed per bird in dollars -/
def chicken_feed_cost : ℚ := 2

/-- The total cost to feed all chickens in dollars -/
def total_chicken_feed_cost : ℚ := 20

/-- The total number of birds Peter has -/
def total_birds : ℕ := 15

theorem peter_bird_count :
  (1 - duck_fraction) * total_birds = total_chicken_feed_cost / chicken_feed_cost :=
by sorry

end NUMINAMATH_CALUDE_peter_bird_count_l1977_197729


namespace NUMINAMATH_CALUDE_z_less_than_y_percentage_l1977_197708

/-- Given w, x, y, z are real numbers satisfying certain conditions,
    prove that z is 46% less than y. -/
theorem z_less_than_y_percentage (w x y z : ℝ) 
  (hw : w = 0.6 * x)
  (hx : x = 0.6 * y)
  (hz : z = 1.5 * w) :
  z = 0.54 * y := by
  sorry

end NUMINAMATH_CALUDE_z_less_than_y_percentage_l1977_197708


namespace NUMINAMATH_CALUDE_light_bulb_probability_l1977_197702

theorem light_bulb_probability (qualification_rate : ℝ) 
  (h1 : qualification_rate = 0.99) : 
  ℝ :=
by
  -- The probability of selecting a qualified light bulb
  -- is equal to the qualification rate
  sorry

#check light_bulb_probability

end NUMINAMATH_CALUDE_light_bulb_probability_l1977_197702


namespace NUMINAMATH_CALUDE_final_cell_count_l1977_197795

/-- Calculates the number of cells after a given number of days, 
    where cells double every 3 days starting from an initial population. -/
def cell_count (initial_cells : ℕ) (days : ℕ) : ℕ :=
  initial_cells * 2^(days / 3)

/-- Theorem stating that given 4 initial cells and 9 days, 
    the final cell count is 32. -/
theorem final_cell_count : cell_count 4 9 = 32 := by
  sorry

end NUMINAMATH_CALUDE_final_cell_count_l1977_197795


namespace NUMINAMATH_CALUDE_circle_equation_solution_l1977_197791

theorem circle_equation_solution :
  ∃! (x y : ℝ), (x - 13)^2 + (y - 14)^2 + (x - y)^2 = 1/3 ∧ x = 40/3 ∧ y = 41/3 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_solution_l1977_197791


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1977_197786

-- Define the hyperbola
def Hyperbola (a b : ℝ) : (ℝ × ℝ) → Prop :=
  λ (x, y) ↦ y^2 / a^2 - x^2 / b^2 = 1

-- Theorem statement
theorem hyperbola_equation (a : ℝ) (h1 : a = 2 * Real.sqrt 5) :
  ∃ b : ℝ, Hyperbola (a^2) (b^2) (2, -5) ∧ b^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1977_197786


namespace NUMINAMATH_CALUDE_definite_integral_reciprocal_cosine_squared_l1977_197704

theorem definite_integral_reciprocal_cosine_squared (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∫ x in (0)..(2 * Real.pi), 1 / (a + b * Real.cos x)^2 = (2 * Real.pi * a) / (a^2 - b^2)^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_reciprocal_cosine_squared_l1977_197704


namespace NUMINAMATH_CALUDE_lucas_easter_eggs_problem_l1977_197770

theorem lucas_easter_eggs_problem (blue_eggs green_eggs min_eggs : ℕ) 
  (h1 : blue_eggs = 30)
  (h2 : green_eggs = 42)
  (h3 : min_eggs = 5) :
  ∃ (basket_eggs : ℕ), 
    basket_eggs ≥ min_eggs ∧ 
    basket_eggs ∣ blue_eggs ∧ 
    basket_eggs ∣ green_eggs ∧
    ∀ (n : ℕ), n > basket_eggs → ¬(n ∣ blue_eggs ∧ n ∣ green_eggs) :=
by sorry

end NUMINAMATH_CALUDE_lucas_easter_eggs_problem_l1977_197770


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1977_197794

theorem sqrt_equation_solution :
  ∀ a b : ℕ+,
    a < b →
    (Real.sqrt (1 + Real.sqrt (25 + 20 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b) ↔
    (a = 1 ∧ b = 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1977_197794


namespace NUMINAMATH_CALUDE_john_pills_per_week_l1977_197709

/-- The number of pills John takes in a week -/
def pills_per_week (hours_between_pills : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  (hours_per_day / hours_between_pills) * days_per_week

/-- Theorem: John takes 28 pills in a week -/
theorem john_pills_per_week : 
  pills_per_week 6 24 7 = 28 := by
  sorry

end NUMINAMATH_CALUDE_john_pills_per_week_l1977_197709


namespace NUMINAMATH_CALUDE_circle_equation_coefficients_l1977_197790

/-- Represents a circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the coefficients of the general circle equation -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a given equation represents a specific circle -/
def represents_circle (eq : CircleEquation) (circle : Circle) : Prop :=
  ∀ (x y : ℝ), 
    x^2 + y^2 + 2*eq.a*x - eq.b*y + eq.c = 0 ↔ 
    (x - circle.center.1)^2 + (y - circle.center.2)^2 = circle.radius^2

/-- The main theorem to prove -/
theorem circle_equation_coefficients 
  (circle : Circle) 
  (h_center : circle.center = (2, 3)) 
  (h_radius : circle.radius = 3) :
  ∃ (eq : CircleEquation),
    represents_circle eq circle ∧ 
    eq.a = -2 ∧ 
    eq.b = 6 ∧ 
    eq.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_coefficients_l1977_197790


namespace NUMINAMATH_CALUDE_tangent_line_cubic_curve_l1977_197792

theorem tangent_line_cubic_curve (m : ℝ) : 
  (∃ x y : ℝ, y = 12 * x + m ∧ y = x^3 - 2 ∧ 12 = 3 * x^2) → 
  (m = -18 ∨ m = 14) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_curve_l1977_197792


namespace NUMINAMATH_CALUDE_putnam_inequality_l1977_197778

theorem putnam_inequality (a x : ℝ) (h1 : 0 < x) (h2 : x < a) :
  (a - x)^6 - 3*a*(a - x)^5 + (5/2)*a^2*(a - x)^4 - (1/2)*a^4*(a - x)^2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_putnam_inequality_l1977_197778


namespace NUMINAMATH_CALUDE_distance_opposite_points_l1977_197760

-- Define a point in polar coordinates
structure PolarPoint where
  r : ℝ
  θ : ℝ

-- Define the distance function between two polar points
def polarDistance (A B : PolarPoint) : ℝ :=
  sorry

-- Theorem statement
theorem distance_opposite_points (A B : PolarPoint) 
    (h : abs (B.θ - A.θ) = Real.pi) : 
  polarDistance A B = A.r + B.r := by
  sorry

end NUMINAMATH_CALUDE_distance_opposite_points_l1977_197760


namespace NUMINAMATH_CALUDE_zachary_purchase_l1977_197747

/-- The cost of items at a store -/
structure StorePrices where
  pencil : ℕ
  notebook : ℕ
  eraser : ℕ

/-- The conditions of the problem -/
def store_conditions (p : StorePrices) : Prop :=
  p.pencil + p.notebook = 80 ∧
  p.notebook + p.eraser = 85 ∧
  3 * p.pencil + 3 * p.notebook + 3 * p.eraser = 315

/-- The theorem to prove -/
theorem zachary_purchase (p : StorePrices) (h : store_conditions p) : 
  p.pencil + p.eraser = 45 := by
  sorry

end NUMINAMATH_CALUDE_zachary_purchase_l1977_197747


namespace NUMINAMATH_CALUDE_sum_of_simplified_fraction_l1977_197720

-- Define the repeating decimal 0.̅4̅5̅
def repeating_decimal : ℚ := 45 / 99

-- Define the function to simplify a fraction
def simplify (q : ℚ) : ℚ := q

-- Define the function to sum numerator and denominator
def sum_num_denom (q : ℚ) : ℕ := q.num.natAbs + q.den

-- Theorem statement
theorem sum_of_simplified_fraction :
  sum_num_denom (simplify repeating_decimal) = 16 := by sorry

end NUMINAMATH_CALUDE_sum_of_simplified_fraction_l1977_197720


namespace NUMINAMATH_CALUDE_original_combined_cost_l1977_197773

/-- Represents the original prices of items --/
structure OriginalPrices where
  dress : ℝ
  shoes : ℝ
  handbag : ℝ
  necklace : ℝ

/-- Represents the discounted prices of items --/
structure DiscountedPrices where
  dress : ℝ
  shoes : ℝ
  handbag : ℝ
  necklace : ℝ

/-- Calculates the total savings before the coupon --/
def totalSavings (original : OriginalPrices) (discounted : DiscountedPrices) : ℝ :=
  (original.dress - discounted.dress) +
  (original.shoes - discounted.shoes) +
  (original.handbag - discounted.handbag) +
  (original.necklace - discounted.necklace)

/-- Calculates the total discounted price before the coupon --/
def totalDiscountedPrice (discounted : DiscountedPrices) : ℝ :=
  discounted.dress + discounted.shoes + discounted.handbag + discounted.necklace

/-- The main theorem --/
theorem original_combined_cost (original : OriginalPrices) (discounted : DiscountedPrices)
  (h1 : discounted.dress = original.dress / 2 - 10)
  (h2 : discounted.shoes = original.shoes * 0.85)
  (h3 : discounted.handbag = original.handbag - 30)
  (h4 : discounted.necklace = original.necklace)
  (h5 : discounted.necklace ≤ original.dress)
  (h6 : totalSavings original discounted = 120)
  (h7 : totalDiscountedPrice discounted * 0.9 = totalDiscountedPrice discounted - 120) :
  original.dress + original.shoes + original.handbag + original.necklace = 1200 := by
  sorry


end NUMINAMATH_CALUDE_original_combined_cost_l1977_197773


namespace NUMINAMATH_CALUDE_sequence_expression_l1977_197765

theorem sequence_expression (a : ℕ → ℝ) :
  a 1 = 2 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + Real.log (1 + 1 / n)) →
  ∀ n : ℕ, n ≥ 1 → a n = 2 + Real.log n :=
by sorry

end NUMINAMATH_CALUDE_sequence_expression_l1977_197765


namespace NUMINAMATH_CALUDE_total_interest_calculation_l1977_197757

/-- Calculate the total interest after 10 years given the following conditions:
    1. The simple interest on the initial principal for 10 years is 400.
    2. The principal is trebled after 5 years. -/
theorem total_interest_calculation (P R : ℝ) 
  (h1 : P * R * 10 / 100 = 400) 
  (h2 : P > 0) 
  (h3 : R > 0) : 
  P * R * 5 / 100 + 3 * P * R * 5 / 100 = 1000 := by
  sorry

#check total_interest_calculation

end NUMINAMATH_CALUDE_total_interest_calculation_l1977_197757


namespace NUMINAMATH_CALUDE_shoe_box_problem_l1977_197768

theorem shoe_box_problem (num_pairs : ℕ) (prob : ℚ) (total_shoes : ℕ) : 
  num_pairs = 12 → 
  prob = 1 / 23 → 
  prob = num_pairs / (total_shoes.choose 2) → 
  total_shoes = 24 := by
sorry

end NUMINAMATH_CALUDE_shoe_box_problem_l1977_197768


namespace NUMINAMATH_CALUDE_b2_a2_minus_a1_value_l1977_197769

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  a₂ - 4 = 4 - a₁ ∧ 1 - a₂ = a₂ - 4

-- Define the geometric sequence
def geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  4 / b₁ = b₂ / 4 ∧ b₂ / 4 = 1 / b₂ ∧ 1 / b₂ = b₃ / 1

theorem b2_a2_minus_a1_value (a₁ a₂ b₁ b₂ b₃ : ℝ) :
  arithmetic_sequence a₁ a₂ → geometric_sequence b₁ b₂ b₃ →
  (b₂ * (a₂ - a₁) = 6 ∨ b₂ * (a₂ - a₁) = -6) :=
by sorry

end NUMINAMATH_CALUDE_b2_a2_minus_a1_value_l1977_197769


namespace NUMINAMATH_CALUDE_smallest_positive_integer_modulo_l1977_197764

theorem smallest_positive_integer_modulo (y : ℕ) : y = 14 ↔ 
  (y > 0 ∧ (y + 3050) % 15 = 1234 % 15 ∧ ∀ z : ℕ, z > 0 → (z + 3050) % 15 = 1234 % 15 → y ≤ z) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_modulo_l1977_197764


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1977_197712

theorem quadratic_inequality (a b c A B C : ℝ) 
  (ha : a ≠ 0) (hA : A ≠ 0)
  (h : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) :
  |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1977_197712


namespace NUMINAMATH_CALUDE_pythagorean_triple_for_eleven_l1977_197727

theorem pythagorean_triple_for_eleven : ∃ (b c : ℕ), 11^2 + b^2 = c^2 ∧ c = 61 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_for_eleven_l1977_197727


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1977_197752

-- System 1
theorem system_one_solution (x y : ℝ) : 
  x - y = 1 ∧ 2*x + y = 5 → x = 2 ∧ y = 1 := by sorry

-- System 2
theorem system_two_solution (x y : ℝ) : 
  x/2 - (y+1)/3 = 1 ∧ x + y = 1 → x = 2 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1977_197752


namespace NUMINAMATH_CALUDE_ways_to_top_center_l1977_197780

/-- Number of ways to reach the center square of the topmost row in a grid -/
def numWaysToTopCenter (n : ℕ) : ℕ :=
  2^(n-1)

/-- Theorem: The number of ways to reach the center square of the topmost row
    in a rectangular grid with n rows and 3 columns, starting from the bottom
    left corner and moving either one square right or simultaneously one square
    left and one square up at each step, is equal to 2^(n-1). -/
theorem ways_to_top_center (n : ℕ) (h : n > 0) :
  numWaysToTopCenter n = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_ways_to_top_center_l1977_197780


namespace NUMINAMATH_CALUDE_afternoon_campers_l1977_197742

theorem afternoon_campers (morning_campers : ℕ) (afternoon_difference : ℕ) : 
  morning_campers = 52 → 
  afternoon_difference = 9 → 
  morning_campers + afternoon_difference = 61 :=
by
  sorry

end NUMINAMATH_CALUDE_afternoon_campers_l1977_197742


namespace NUMINAMATH_CALUDE_no_prime_between_100_110_congruent_3_mod_6_l1977_197746

theorem no_prime_between_100_110_congruent_3_mod_6 : ¬ ∃ n : ℕ, 
  Nat.Prime n ∧ 100 < n ∧ n < 110 ∧ n % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_between_100_110_congruent_3_mod_6_l1977_197746


namespace NUMINAMATH_CALUDE_second_quadrant_trig_identity_l1977_197789

theorem second_quadrant_trig_identity (α : Real) 
  (h1 : π/2 < α ∧ α < π) : 
  (Real.sin α / Real.cos α) * Real.sqrt (1 / Real.sin α^2 - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_trig_identity_l1977_197789


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1977_197728

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₁ + a₂ = -1 and a₃ = 4,
    prove that a₄ + a₅ = 17. -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) 
    (h_arith : is_arithmetic_sequence a)
    (h_sum : a 1 + a 2 = -1)
    (h_third : a 3 = 4) : 
  a 4 + a 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1977_197728


namespace NUMINAMATH_CALUDE_number_of_girls_in_school_l1977_197726

/-- The number of girls in a school, given certain sampling conditions -/
theorem number_of_girls_in_school :
  ∀ (total_students sample_size : ℕ) (girls_in_school : ℕ),
  total_students = 1600 →
  sample_size = 200 →
  girls_in_school ≤ total_students →
  (girls_in_school : ℚ) / (total_students - girls_in_school : ℚ) = 95 / 105 →
  girls_in_school = 760 := by
sorry

end NUMINAMATH_CALUDE_number_of_girls_in_school_l1977_197726


namespace NUMINAMATH_CALUDE_bottle_production_l1977_197740

/-- Given that 6 identical machines produce 300 bottles per minute at a constant rate,
    10 such machines will produce 2000 bottles in 4 minutes. -/
theorem bottle_production (machines : ℕ) (bottles_per_minute : ℕ) (time : ℕ) : 
  machines = 6 → bottles_per_minute = 300 → time = 4 →
  (10 : ℕ) * bottles_per_minute * time / machines = 2000 :=
by sorry

end NUMINAMATH_CALUDE_bottle_production_l1977_197740


namespace NUMINAMATH_CALUDE_quadrilateral_area_in_possible_areas_all_possible_areas_achievable_l1977_197701

/-- Represents a point on the side of a square --/
inductive SidePoint
| A1 | A2 | A3  -- Points on side AB
| B1 | B2 | B3  -- Points on side BC
| C1 | C2 | C3  -- Points on side CD
| D1 | D2 | D3  -- Points on side DA

/-- Represents a quadrilateral formed by choosing points from each side of a square --/
structure Quadrilateral :=
  (p1 : SidePoint)
  (p2 : SidePoint)
  (p3 : SidePoint)
  (p4 : SidePoint)

/-- Calculates the area of a quadrilateral formed by choosing points from each side of a square --/
def area (q : Quadrilateral) : ℝ :=
  sorry  -- The actual calculation would go here

/-- The set of possible areas for quadrilaterals formed in the given square --/
def possible_areas : Set ℝ := {6, 7, 7.5, 8, 8.5, 9, 10}

/-- Theorem stating that the area of any quadrilateral formed in the given square
    must be one of the values in the possible_areas set --/
theorem quadrilateral_area_in_possible_areas (q : Quadrilateral) :
  area q ∈ possible_areas :=
sorry

/-- Theorem stating that every value in the possible_areas set
    is achievable by some quadrilateral in the given square --/
theorem all_possible_areas_achievable :
  ∀ a ∈ possible_areas, ∃ q : Quadrilateral, area q = a :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_area_in_possible_areas_all_possible_areas_achievable_l1977_197701


namespace NUMINAMATH_CALUDE_speed_conversion_l1977_197755

-- Define the conversion factor
def meters_per_second_to_kmph : ℝ := 3.6

-- Define the given speed in meters per second
def speed_in_mps : ℝ := 16.668

-- State the theorem
theorem speed_conversion :
  speed_in_mps * meters_per_second_to_kmph = 60.0048 := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l1977_197755


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l1977_197748

/-- The y-coordinate of the point on the y-axis equidistant from A(3, 0) and B(4, -3) is -8/3 -/
theorem equidistant_point_y_coordinate : 
  ∃ y : ℝ, 
    (3 - 0)^2 + (0 - y)^2 = (4 - 0)^2 + (-3 - y)^2 ∧ 
    y = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l1977_197748


namespace NUMINAMATH_CALUDE_race_time_difference_l1977_197767

/-- Proves that given the speeds of A and B are in the ratio 3:4, and A takes 2 hours to reach the destination, A takes 30 minutes more than B to reach the destination. -/
theorem race_time_difference (speed_a speed_b : ℝ) (time_a : ℝ) : 
  speed_a / speed_b = 3 / 4 →
  time_a = 2 →
  (time_a - (speed_a * time_a / speed_b)) * 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l1977_197767


namespace NUMINAMATH_CALUDE_couch_price_after_changes_l1977_197717

theorem couch_price_after_changes (initial_price : ℝ) 
  (h_initial : initial_price = 62500) : 
  let increase_factor := 1.2
  let decrease_factor := 0.8
  let final_factor := (increase_factor ^ 3) * (decrease_factor ^ 3)
  initial_price * final_factor = 55296 := by sorry

end NUMINAMATH_CALUDE_couch_price_after_changes_l1977_197717


namespace NUMINAMATH_CALUDE_triangle_equality_condition_l1977_197763

/-- In a triangle ABC, the sum of squares of its sides is equal to 4√3 times its area 
    if and only if the triangle is equilateral. -/
theorem triangle_equality_condition (a b c : ℝ) (Δ : ℝ) :
  (a > 0) → (b > 0) → (c > 0) → (Δ > 0) →
  (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * Δ) ↔ (a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_triangle_equality_condition_l1977_197763


namespace NUMINAMATH_CALUDE_complex_magnitude_for_specific_quadratic_l1977_197744

theorem complex_magnitude_for_specific_quadratic : 
  ∀ z : ℂ, z^2 - 6*z + 20 = 0 → Complex.abs z = Real.sqrt 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_for_specific_quadratic_l1977_197744


namespace NUMINAMATH_CALUDE_division_problem_l1977_197787

theorem division_problem : (96 / 6) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1977_197787


namespace NUMINAMATH_CALUDE_plates_problem_l1977_197735

theorem plates_problem (total_days : ℕ) (plates_two_people : ℕ) (plates_four_people : ℕ) (total_plates : ℕ) :
  total_days = 7 →
  plates_two_people = 2 →
  plates_four_people = 8 →
  total_plates = 38 →
  ∃ (days_two_people : ℕ),
    days_two_people * plates_two_people + (total_days - days_two_people) * plates_four_people = total_plates ∧
    days_two_people = 3 :=
by sorry

end NUMINAMATH_CALUDE_plates_problem_l1977_197735


namespace NUMINAMATH_CALUDE_carl_watermelons_left_l1977_197749

/-- Calculates the number of watermelons left after a day of selling -/
def watermelons_left (price : ℕ) (profit : ℕ) (initial : ℕ) : ℕ :=
  initial - (profit / price)

/-- Theorem: Given the conditions, Carl has 18 watermelons left -/
theorem carl_watermelons_left :
  let price : ℕ := 3
  let profit : ℕ := 105
  let initial : ℕ := 53
  watermelons_left price profit initial = 18 := by
  sorry

end NUMINAMATH_CALUDE_carl_watermelons_left_l1977_197749


namespace NUMINAMATH_CALUDE_sum_of_algebra_values_l1977_197774

-- Define the function that assigns numeric values to letters based on their position
def letterValue (position : ℕ) : ℤ :=
  match position % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 0
  | 6 => -1
  | 7 => -2
  | 0 => -3
  | _ => 0  -- This case should never occur due to the modulo operation

-- Define the positions of letters in "ALGEBRA"
def algebraPositions : List ℕ := [1, 12, 7, 5, 2, 18, 1]

-- Theorem statement
theorem sum_of_algebra_values :
  (algebraPositions.map letterValue).sum = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_algebra_values_l1977_197774


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l1977_197762

/-- The focal length of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²) -/
theorem hyperbola_focal_length (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let focal_length := Real.sqrt (a^2 + b^2)
  ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → 
    ∃ (f₁ f₂ : ℝ × ℝ), 
      f₁.1 = focal_length ∧ f₁.2 = 0 ∧
      f₂.1 = -focal_length ∧ f₂.2 = 0 ∧
      ∀ p : ℝ × ℝ, p.1^2/a^2 - p.2^2/b^2 = 1 → 
        Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + 
        Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = 2*a :=
by
  sorry


end NUMINAMATH_CALUDE_hyperbola_focal_length_l1977_197762


namespace NUMINAMATH_CALUDE_average_of_one_eighth_and_one_sixth_l1977_197733

theorem average_of_one_eighth_and_one_sixth :
  (1 / 8 + 1 / 6) / 2 = 7 / 48 := by sorry

end NUMINAMATH_CALUDE_average_of_one_eighth_and_one_sixth_l1977_197733


namespace NUMINAMATH_CALUDE_mary_spends_five_l1977_197719

/-- Proves that Mary spends $5 given the initial conditions and final state -/
theorem mary_spends_five (marco_initial : ℕ) (mary_initial : ℕ) 
  (h1 : marco_initial = 24)
  (h2 : mary_initial = 15)
  (marco_gives : ℕ := marco_initial / 2)
  (marco_final : ℕ := marco_initial - marco_gives)
  (mary_after_receiving : ℕ := mary_initial + marco_gives)
  (mary_final : ℕ)
  (h3 : mary_final = marco_final + 10) :
  mary_after_receiving - mary_final = 5 := by
sorry

end NUMINAMATH_CALUDE_mary_spends_five_l1977_197719


namespace NUMINAMATH_CALUDE_ellipse_and_triangle_properties_l1977_197723

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line with slope 1 -/
def Line := { m : ℝ // ∀ x y, y = x + m }

theorem ellipse_and_triangle_properties
  (G : Ellipse)
  (e : ℝ)
  (F : Point)
  (l : Line)
  (P : Point)
  (he : e = Real.sqrt 6 / 3)
  (hF : F.x = 2 * Real.sqrt 2 ∧ F.y = 0)
  (hP : P.x = -3 ∧ P.y = 2)
  (h_isosceles : ∃ A B : Point, A ≠ B ∧
    (A.x - P.x)^2 + (A.y - P.y)^2 = (B.x - P.x)^2 + (B.y - P.y)^2 ∧
    ∃ t : ℝ, A.y = A.x + l.val ∧ B.y = B.x + l.val ∧
    A.x^2 / G.a^2 + A.y^2 / G.b^2 = 1 ∧
    B.x^2 / G.a^2 + B.y^2 / G.b^2 = 1) :
  G.a^2 = 12 ∧ G.b^2 = 4 ∧
  ∃ A B : Point, A ≠ B ∧
    (A.x - P.x)^2 + (A.y - P.y)^2 = (B.x - P.x)^2 + (B.y - P.y)^2 ∧
    ∃ t : ℝ, A.y = A.x + l.val ∧ B.y = B.x + l.val ∧
    A.x^2 / G.a^2 + A.y^2 / G.b^2 = 1 ∧
    B.x^2 / G.a^2 + B.y^2 / G.b^2 = 1 ∧
    (B.x - A.x) * (B.y - A.y) / 2 = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_triangle_properties_l1977_197723


namespace NUMINAMATH_CALUDE_twin_primes_divisibility_l1977_197785

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem twin_primes_divisibility (a : ℤ) 
  (h1 : is_prime (a - 1).natAbs) 
  (h2 : is_prime (a + 1).natAbs) 
  (h3 : (a - 1).natAbs > 10) 
  (h4 : (a + 1).natAbs > 10) : 
  120 ∣ (a^3 - 4*a) :=
sorry

end NUMINAMATH_CALUDE_twin_primes_divisibility_l1977_197785


namespace NUMINAMATH_CALUDE_pizza_area_increase_l1977_197777

/-- Theorem: Percent increase in pizza area
    If the radius of a large pizza is 60% larger than that of a medium pizza,
    then the percent increase in area between a medium and a large pizza is 156%. -/
theorem pizza_area_increase (r : ℝ) (h : r > 0) : 
  let large_radius := 1.6 * r
  let medium_area := π * r^2
  let large_area := π * large_radius^2
  (large_area - medium_area) / medium_area * 100 = 156 :=
by
  sorry

#check pizza_area_increase

end NUMINAMATH_CALUDE_pizza_area_increase_l1977_197777


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1977_197730

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (10 + 3 * z) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1977_197730


namespace NUMINAMATH_CALUDE_geese_migration_rate_ratio_l1977_197776

/-- Given a population of geese where 50% are male and 20% of migrating geese are male,
    the ratio of migration rates between male and female geese is 1:4. -/
theorem geese_migration_rate_ratio :
  ∀ (total_geese male_geese migrating_geese male_migrating : ℕ),
  male_geese = total_geese / 2 →
  male_migrating = migrating_geese / 5 →
  (male_migrating : ℚ) / male_geese = (migrating_geese - male_migrating : ℚ) / (total_geese - male_geese) / 4 :=
by sorry

end NUMINAMATH_CALUDE_geese_migration_rate_ratio_l1977_197776


namespace NUMINAMATH_CALUDE_cube_product_inequality_l1977_197710

theorem cube_product_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  x^3 * y^3 * (x^3 + y^3) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_product_inequality_l1977_197710


namespace NUMINAMATH_CALUDE_composition_equality_l1977_197716

-- Define the functions f and h
def f (m n x : ℝ) : ℝ := m * x + n
def h (p q r x : ℝ) : ℝ := p * x^2 + q * x + r

-- State the theorem
theorem composition_equality (m n p q r : ℝ) :
  (∀ x, f m n (h p q r x) = h p q r (f m n x)) ↔ (m = p ∧ n = 0) := by
  sorry

end NUMINAMATH_CALUDE_composition_equality_l1977_197716


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1977_197799

/-- Proves that given the average speed from y to x and the average speed for the whole journey,
    we can determine the average speed from x to y. -/
theorem average_speed_calculation (speed_y_to_x : ℝ) (speed_round_trip : ℝ) (speed_x_to_y : ℝ) :
  speed_y_to_x = 36 →
  speed_round_trip = 39.6 →
  speed_x_to_y = 44 :=
by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1977_197799


namespace NUMINAMATH_CALUDE_dima_grade_and_instrument_l1977_197721

-- Define the students
inductive Student : Type
| Vasya : Student
| Dima : Student
| Kolya : Student
| Sergey : Student

-- Define the grades
inductive Grade : Type
| Fifth : Grade
| Sixth : Grade
| Seventh : Grade
| Eighth : Grade

-- Define the instruments
inductive Instrument : Type
| Saxophone : Instrument
| Keyboard : Instrument
| Drums : Instrument
| Guitar : Instrument

-- Define the assignment of grades and instruments to students
def grade_assignment : Student → Grade := sorry
def instrument_assignment : Student → Instrument := sorry

-- State the theorem
theorem dima_grade_and_instrument :
  (instrument_assignment Student.Vasya = Instrument.Saxophone) ∧
  (grade_assignment Student.Vasya ≠ Grade.Eighth) ∧
  (∃ s, grade_assignment s = Grade.Sixth ∧ instrument_assignment s = Instrument.Keyboard) ∧
  (∀ s, instrument_assignment s = Instrument.Drums → s ≠ Student.Dima) ∧
  (instrument_assignment Student.Sergey ≠ Instrument.Keyboard) ∧
  (grade_assignment Student.Sergey ≠ Grade.Fifth) ∧
  (grade_assignment Student.Dima ≠ Grade.Sixth) ∧
  (∀ s, instrument_assignment s = Instrument.Drums → grade_assignment s ≠ Grade.Eighth) →
  (grade_assignment Student.Dima = Grade.Eighth ∧ instrument_assignment Student.Dima = Instrument.Guitar) :=
by sorry


end NUMINAMATH_CALUDE_dima_grade_and_instrument_l1977_197721


namespace NUMINAMATH_CALUDE_dot_product_example_l1977_197754

theorem dot_product_example : 
  let v1 : Fin 2 → ℝ := ![3, -2]
  let v2 : Fin 2 → ℝ := ![-5, 7]
  Finset.sum (Finset.range 2) (λ i => v1 i * v2 i) = -29 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_example_l1977_197754


namespace NUMINAMATH_CALUDE_friend_candy_purchase_l1977_197718

def feeding_allowance : ℚ := 4
def fraction_given : ℚ := 1/4
def candy_cost : ℚ := 1/5  -- 20 cents = 1/5 dollar

theorem friend_candy_purchase :
  (feeding_allowance * fraction_given) / candy_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_friend_candy_purchase_l1977_197718


namespace NUMINAMATH_CALUDE_cosine_sum_and_square_l1977_197715

theorem cosine_sum_and_square (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.cos (5 * π / 6 + α) + (Real.cos (4 * π / 3 + α))^2 = (2 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_and_square_l1977_197715


namespace NUMINAMATH_CALUDE_finance_club_probability_l1977_197724

theorem finance_club_probability (total_members : ℕ) (interested_ratio : ℚ) : 
  total_members = 20 →
  interested_ratio = 3 / 4 →
  let interested_members := (interested_ratio * total_members).num
  let not_interested_members := total_members - interested_members
  let prob_neither_interested := (not_interested_members / total_members) * ((not_interested_members - 1) / (total_members - 1))
  1 - prob_neither_interested = 18 / 19 := by
sorry

end NUMINAMATH_CALUDE_finance_club_probability_l1977_197724


namespace NUMINAMATH_CALUDE_total_students_is_59_l1977_197739

/-- Represents a group of students with subgroups taking history and statistics -/
structure StudentGroup where
  total : ℕ
  history : ℕ
  statistics : ℕ
  both : ℕ
  history_only : ℕ
  history_or_statistics : ℕ

/-- The properties of the student group as described in the problem -/
def problem_group : StudentGroup where
  history := 36
  statistics := 32
  history_or_statistics := 59
  history_only := 27
  both := 36 - 27  -- Derived from history - history_only
  total := 59  -- This is what we want to prove

/-- Theorem stating that the total number of students in the group is 59 -/
theorem total_students_is_59 (g : StudentGroup) 
  (h1 : g.history = problem_group.history)
  (h2 : g.statistics = problem_group.statistics)
  (h3 : g.history_or_statistics = problem_group.history_or_statistics)
  (h4 : g.history_only = problem_group.history_only)
  (h5 : g.both = g.history - g.history_only)
  (h6 : g.history_or_statistics = g.history + g.statistics - g.both) :
  g.total = problem_group.total := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_59_l1977_197739


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_83_l1977_197751

theorem last_three_digits_of_7_to_83 :
  7^83 ≡ 886 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_83_l1977_197751


namespace NUMINAMATH_CALUDE_age_difference_l1977_197731

theorem age_difference (a b c : ℕ) : 
  b = 2 * c → 
  a + b + c = 22 → 
  b = 8 → 
  a - b = 2 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l1977_197731


namespace NUMINAMATH_CALUDE_infinitely_many_increasing_largest_prime_factors_l1977_197736

/-- h(n) denotes the largest prime factor of the natural number n -/
def h (n : ℕ) : ℕ := sorry

/-- There exist infinitely many natural numbers n such that 
    the largest prime factor of n is less than the largest prime factor of n+1, 
    which is less than the largest prime factor of n+2 -/
theorem infinitely_many_increasing_largest_prime_factors :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, h n < h (n + 1) ∧ h (n + 1) < h (n + 2) := by sorry

end NUMINAMATH_CALUDE_infinitely_many_increasing_largest_prime_factors_l1977_197736


namespace NUMINAMATH_CALUDE_sara_oranges_l1977_197797

/-- Given that Joan picked 37 oranges, 47 oranges were picked in total, 
    and Alyssa picked 30 pears, prove that Sara picked 10 oranges. -/
theorem sara_oranges (joan_oranges : ℕ) (total_oranges : ℕ) (alyssa_pears : ℕ) 
    (h1 : joan_oranges = 37)
    (h2 : total_oranges = 47)
    (h3 : alyssa_pears = 30) : 
  total_oranges - joan_oranges = 10 := by
  sorry

end NUMINAMATH_CALUDE_sara_oranges_l1977_197797


namespace NUMINAMATH_CALUDE_range_of_n_over_m_l1977_197732

def A (m n : ℝ) := {z : ℂ | Complex.abs (z + n * Complex.I) + Complex.abs (z - m * Complex.I) = n}
def B (m n : ℝ) := {z : ℂ | Complex.abs (z + n * Complex.I) - Complex.abs (z - m * Complex.I) = -m}

theorem range_of_n_over_m (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) 
  (hA : Set.Nonempty (A m n)) (hB : Set.Nonempty (B m n)) : 
  n / m ≤ -2 ∧ ∀ k : ℝ, ∃ m n : ℝ, m ≠ 0 ∧ n ≠ 0 ∧ Set.Nonempty (A m n) ∧ Set.Nonempty (B m n) ∧ n / m < k :=
sorry

end NUMINAMATH_CALUDE_range_of_n_over_m_l1977_197732


namespace NUMINAMATH_CALUDE_dvd_price_percentage_l1977_197713

theorem dvd_price_percentage (srp : ℝ) (h1 : srp > 0) : 
  let marked_price := 0.6 * srp
  let bob_price := 0.4 * marked_price
  bob_price / srp = 0.24 := by
sorry

end NUMINAMATH_CALUDE_dvd_price_percentage_l1977_197713


namespace NUMINAMATH_CALUDE_range_of_c_l1977_197737

/-- A condition is sufficient but not necessary -/
def SufficientButNotNecessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

theorem range_of_c (a c : ℝ) :
  SufficientButNotNecessary (a ≥ 1/8) (∀ x > 0, 2*x + a/x ≥ c) →
  c ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l1977_197737


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_to_3198_for_divisibility_by_8_l1977_197784

theorem least_addition_for_divisibility (n : Nat) (d : Nat) : ∃ (x : Nat), x < d ∧ (n + x) % d = 0 :=
by
  -- The proof would go here
  sorry

theorem least_addition_to_3198_for_divisibility_by_8 :
  ∃ (x : Nat), x < 8 ∧ (3198 + x) % 8 = 0 ∧ ∀ (y : Nat), y < x → (3198 + y) % 8 ≠ 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_to_3198_for_divisibility_by_8_l1977_197784


namespace NUMINAMATH_CALUDE_gumballs_per_package_l1977_197734

theorem gumballs_per_package (total_gumballs : ℕ) (total_boxes : ℕ) 
  (h1 : total_gumballs = 20) 
  (h2 : total_boxes = 4) 
  (h3 : total_gumballs > 0) 
  (h4 : total_boxes > 0) : 
  (total_gumballs / total_boxes : ℕ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gumballs_per_package_l1977_197734


namespace NUMINAMATH_CALUDE_conference_handshakes_l1977_197793

theorem conference_handshakes (n : ℕ) (h : n = 25) : 
  (n * (n - 1)) / 2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1977_197793


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1977_197750

theorem polynomial_division_theorem (x : ℝ) : 
  ∃ (q r : ℝ), x^5 - 24*x^3 + 12*x^2 - x + 20 = (x - 3) * (x^4 + 3*x^3 - 15*x^2 - 33*x - 100) + (-280) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1977_197750


namespace NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l1977_197743

theorem range_of_a_for_always_positive_quadratic :
  {a : ℝ | ∀ x : ℝ, 2 * x^2 + (a - 1) * x + (1/2 : ℝ) > 0} = {a : ℝ | -1 < a ∧ a < 3} := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l1977_197743


namespace NUMINAMATH_CALUDE_coin_value_difference_l1977_197741

theorem coin_value_difference (p n d : ℕ) : 
  p + n + d = 3030 →
  p ≥ 1 →
  n ≥ 1 →
  d ≥ 1 →
  (∀ p' n' d' : ℕ, p' + n' + d' = 3030 ∧ p' ≥ 1 ∧ n' ≥ 1 ∧ d' ≥ 1 →
    p' + 5 * n' + 10 * d' ≤ 30286 ∧
    p' + 5 * n' + 10 * d' ≥ 3043) →
  30286 - 3043 = 27243 :=
by sorry

end NUMINAMATH_CALUDE_coin_value_difference_l1977_197741


namespace NUMINAMATH_CALUDE_delta_value_l1977_197779

theorem delta_value (Δ : ℤ) (h : 4 * (-3) = Δ + 3) : Δ = -15 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l1977_197779


namespace NUMINAMATH_CALUDE_cube_packing_percentage_l1977_197771

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  side : ℕ

/-- Calculates the number of cubes that fit along a given dimension -/
def cubesFitAlongDimension (boxDim : ℕ) (cubeSide : ℕ) : ℕ :=
  boxDim / cubeSide

/-- Calculates the total number of cubes that fit in the box -/
def totalCubesFit (box : BoxDimensions) (cube : CubeDimensions) : ℕ :=
  (cubesFitAlongDimension box.length cube.side) *
  (cubesFitAlongDimension box.width cube.side) *
  (cubesFitAlongDimension box.height cube.side)

/-- Calculates the volume of a rectangular box -/
def boxVolume (box : BoxDimensions) : ℕ :=
  box.length * box.width * box.height

/-- Calculates the volume of a cube -/
def cubeVolume (cube : CubeDimensions) : ℕ :=
  cube.side * cube.side * cube.side

/-- Calculates the percentage of box volume occupied by cubes -/
def percentageOccupied (box : BoxDimensions) (cube : CubeDimensions) : ℚ :=
  let totalCubes := totalCubesFit box cube
  let volumeOccupied := totalCubes * (cubeVolume cube)
  (volumeOccupied : ℚ) / (boxVolume box : ℚ) * 100

/-- Theorem stating that the percentage of volume occupied by 3-inch cubes
    in a 9x8x12 inch box is 75% -/
theorem cube_packing_percentage :
  let box := BoxDimensions.mk 9 8 12
  let cube := CubeDimensions.mk 3
  percentageOccupied box cube = 75 := by
  sorry

end NUMINAMATH_CALUDE_cube_packing_percentage_l1977_197771


namespace NUMINAMATH_CALUDE_largest_y_floor_div_l1977_197725

theorem largest_y_floor_div : 
  ∀ y : ℝ, (↑(Int.floor y) / y = 8 / 9) → y ≤ 63 / 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_y_floor_div_l1977_197725


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1977_197756

theorem sqrt_product_simplification (x : ℝ) (h : x > 0) :
  Real.sqrt (100 * x) * Real.sqrt (3 * x) * Real.sqrt (18 * x) = 30 * x * Real.sqrt (6 * x) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1977_197756


namespace NUMINAMATH_CALUDE_profit_ratio_l1977_197782

def investment_p : ℕ := 500000
def investment_q : ℕ := 1000000

theorem profit_ratio (p q : ℕ) (h : p = investment_p ∧ q = investment_q) :
  (p : ℚ) / (p + q : ℚ) = 1 / 3 ∧ (q : ℚ) / (p + q : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_profit_ratio_l1977_197782


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1977_197714

theorem right_triangle_side_length (A B C : ℝ × ℝ) (AB AC BC : ℝ) :
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = AB^2 →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = AC^2 →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = BC^2 →
  AB^2 + AC^2 = BC^2 →
  Real.cos (30 * π / 180) = (BC^2 + AC^2 - AB^2) / (2 * BC * AC) →
  AC = 18 →
  AB = 18 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1977_197714


namespace NUMINAMATH_CALUDE_min_cards_36_4suits_l1977_197798

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- The minimum number of cards to draw to guarantee all suits are represented -/
def min_cards_to_draw (d : Deck) : ℕ :=
  (d.num_suits - 1) * d.cards_per_suit + 1

/-- Theorem stating the minimum number of cards to draw for a 36-card deck with 4 suits -/
theorem min_cards_36_4suits :
  ∃ (d : Deck), d.total_cards = 36 ∧ d.num_suits = 4 ∧ min_cards_to_draw d = 28 :=
sorry

end NUMINAMATH_CALUDE_min_cards_36_4suits_l1977_197798


namespace NUMINAMATH_CALUDE_abigail_expenses_l1977_197783

def initial_amount : ℝ := 200

def food_expense_percentage : ℝ := 0.60

def phone_bill_percentage : ℝ := 0.25

def entertainment_expense : ℝ := 20

def remaining_amount (initial : ℝ) (food_percent : ℝ) (phone_percent : ℝ) (entertainment : ℝ) : ℝ :=
  let after_food := initial * (1 - food_percent)
  let after_phone := after_food * (1 - phone_percent)
  after_phone - entertainment

theorem abigail_expenses :
  remaining_amount initial_amount food_expense_percentage phone_bill_percentage entertainment_expense = 40 := by
  sorry

end NUMINAMATH_CALUDE_abigail_expenses_l1977_197783


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l1977_197772

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y : ℝ) :
  S.card = 50 →
  x ∈ S →
  y ∈ S →
  x = 45 →
  y = 55 →
  (S.sum id) / S.card = 38 →
  ((S.sum id - (x + y)) / (S.card - 2) : ℝ) = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l1977_197772


namespace NUMINAMATH_CALUDE_is_arithmetic_sequence_l1977_197700

def S (n : ℕ) : ℝ := 2 * n + 1

theorem is_arithmetic_sequence :
  ∀ n : ℕ, S (n + 1) - S n = S 1 - S 0 :=
by
  sorry

end NUMINAMATH_CALUDE_is_arithmetic_sequence_l1977_197700


namespace NUMINAMATH_CALUDE_sector_area_l1977_197722

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (h1 : arc_length = 6) (h2 : central_angle = 2) : 
  (1/2) * arc_length * (arc_length / central_angle) = 9 := by
sorry

end NUMINAMATH_CALUDE_sector_area_l1977_197722


namespace NUMINAMATH_CALUDE_math_contest_correct_answers_l1977_197703

theorem math_contest_correct_answers 
  (total_problems : ℕ)
  (correct_points : ℤ)
  (incorrect_points : ℤ)
  (total_score : ℤ)
  (min_guesses : ℕ)
  (h1 : total_problems = 15)
  (h2 : correct_points = 6)
  (h3 : incorrect_points = -3)
  (h4 : total_score = 45)
  (h5 : min_guesses ≥ 4)
  : ∃ (correct_answers : ℕ), 
    correct_answers * correct_points + (total_problems - correct_answers) * incorrect_points = total_score ∧
    correct_answers = 10 := by
  sorry

end NUMINAMATH_CALUDE_math_contest_correct_answers_l1977_197703


namespace NUMINAMATH_CALUDE_max_coins_distribution_l1977_197781

theorem max_coins_distribution (n : ℕ) (h1 : n < 150) 
  (h2 : ∃ k : ℕ, n = 8 * k + 4) : n ≤ 148 := by
  sorry

end NUMINAMATH_CALUDE_max_coins_distribution_l1977_197781


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_l1977_197738

theorem units_digit_of_quotient : ∃ n : ℕ, (7^1993 + 5^1993) / 6 = 10 * n + 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_l1977_197738


namespace NUMINAMATH_CALUDE_naza_market_averages_l1977_197758

/-- Represents an electronic shop with TV sets and models -/
structure Shop where
  name : Char
  tv_sets : ℕ
  tv_models : ℕ

/-- The list of shops in the Naza market -/
def naza_shops : List Shop := [
  ⟨'A', 20, 3⟩,
  ⟨'B', 30, 4⟩,
  ⟨'C', 60, 5⟩,
  ⟨'D', 80, 6⟩,
  ⟨'E', 50, 2⟩,
  ⟨'F', 40, 4⟩,
  ⟨'G', 70, 3⟩
]

/-- The total number of shops -/
def total_shops : ℕ := naza_shops.length

/-- Calculates the average of a list of natural numbers -/
def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

/-- Theorem stating the average number of TV sets and models in Naza market shops -/
theorem naza_market_averages :
  average (naza_shops.map Shop.tv_sets) = 50 ∧
  average (naza_shops.map Shop.tv_models) = 27 / 7 := by
  sorry

end NUMINAMATH_CALUDE_naza_market_averages_l1977_197758


namespace NUMINAMATH_CALUDE_monthly_salary_proof_l1977_197759

/-- Proves that a person's monthly salary is 1000 Rs, given the conditions -/
theorem monthly_salary_proof (salary : ℝ) : salary = 1000 :=
  let initial_savings_rate : ℝ := 0.25
  let initial_expense_rate : ℝ := 1 - initial_savings_rate
  let expense_increase_rate : ℝ := 0.10
  let new_savings_amount : ℝ := 175

  have h1 : initial_savings_rate * salary = 
            salary - initial_expense_rate * salary := by sorry

  have h2 : new_savings_amount = 
            salary - (initial_expense_rate * salary * (1 + expense_increase_rate)) := by sorry

  sorry

end NUMINAMATH_CALUDE_monthly_salary_proof_l1977_197759


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1977_197745

theorem repeating_decimal_sum : ∃ (a b : ℚ), 
  (∀ n : ℕ, a = 2 / 10^n + a / 10^n) ∧ 
  (∀ m : ℕ, b = 3 / 100^m + b / 100^m) ∧ 
  (a + b = 25 / 99) := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1977_197745
