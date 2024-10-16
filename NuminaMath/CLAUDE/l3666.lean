import Mathlib

namespace NUMINAMATH_CALUDE_complete_square_sum_l3666_366644

theorem complete_square_sum (x : ℝ) : 
  (x^2 - 10*x + 15 = 0) → 
  ∃ (a b : ℤ), ((x + a : ℝ)^2 = b) ∧ (a + b = 5) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_sum_l3666_366644


namespace NUMINAMATH_CALUDE_area_is_two_l3666_366619

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (A B C : Circle) : Prop :=
  A.radius = 1 ∧
  B.radius = 1 ∧
  C.radius = 1 ∧
  -- A and B are tangent
  dist A.center B.center = 2 ∧
  -- C is tangent to the midpoint of AB
  C.center.1 = (A.center.1 + B.center.1) / 2 ∧
  C.center.2 = (A.center.2 + B.center.2) / 2 + 1

-- Define the area function
def area_inside_C_outside_AB (A B C : Circle) : ℝ := sorry

-- Theorem statement
theorem area_is_two (A B C : Circle) :
  problem_setup A B C → area_inside_C_outside_AB A B C = 2 := by sorry

end NUMINAMATH_CALUDE_area_is_two_l3666_366619


namespace NUMINAMATH_CALUDE_stock_worth_calculation_l3666_366627

theorem stock_worth_calculation (W : ℝ) 
  (h1 : 0.02 * W - 0.024 * W = -400) : W = 100000 := by
  sorry

end NUMINAMATH_CALUDE_stock_worth_calculation_l3666_366627


namespace NUMINAMATH_CALUDE_nested_subtraction_simplification_l3666_366677

theorem nested_subtraction_simplification (y : ℝ) : 2 - (2 - (2 - (2 - (2 - y)))) = 4 - y := by
  sorry

end NUMINAMATH_CALUDE_nested_subtraction_simplification_l3666_366677


namespace NUMINAMATH_CALUDE_min_sum_squares_l3666_366621

theorem min_sum_squares (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x + 2*y + 3*z = 2) : 
  x^2 + y^2 + z^2 ≥ 2/7 ∧ 
  (x^2 + y^2 + z^2 = 2/7 ↔ x = 1/7 ∧ y = 2/7 ∧ z = 3/7) :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3666_366621


namespace NUMINAMATH_CALUDE_sqrt_x_fifth_power_eq_1024_l3666_366699

theorem sqrt_x_fifth_power_eq_1024 (x : ℝ) : (Real.sqrt x) ^ 5 = 1024 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_fifth_power_eq_1024_l3666_366699


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l3666_366623

theorem multiply_and_simplify (x y z : ℝ) :
  (3 * x^2 * z - 7 * y^3) * (9 * x^4 * z^2 + 21 * x^2 * y * z^3 + 49 * y^6) = 27 * x^6 * z^3 - 343 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l3666_366623


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3666_366639

theorem sufficient_not_necessary (a b : ℝ) :
  (((a - b) * a^2 < 0 → a < b) ∧
  (∃ a b : ℝ, a < b ∧ (a - b) * a^2 ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3666_366639


namespace NUMINAMATH_CALUDE_unique_function_divisibility_l3666_366636

theorem unique_function_divisibility 
  (f : ℕ+ → ℕ+) 
  (h : ∀ (m n : ℕ+), (m^2 + f n) ∣ (m * f m + n)) : 
  ∀ (n : ℕ+), f n = n :=
sorry

end NUMINAMATH_CALUDE_unique_function_divisibility_l3666_366636


namespace NUMINAMATH_CALUDE_isaac_ribbon_length_l3666_366673

theorem isaac_ribbon_length :
  ∀ (total_parts : ℕ) (used_parts : ℕ) (unused_length : ℝ),
    total_parts = 6 →
    used_parts = 4 →
    unused_length = 10 →
    (unused_length / (total_parts - used_parts : ℝ)) * total_parts = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_isaac_ribbon_length_l3666_366673


namespace NUMINAMATH_CALUDE_floor_equation_solution_l3666_366616

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3*x⌋ - 1/2⌋ = ⌊x + 3⌋ ↔ 5/3 ≤ x ∧ x < 7/3 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l3666_366616


namespace NUMINAMATH_CALUDE_min_value_expression_l3666_366675

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2) / (a * b * c) ≥ 48 * Real.sqrt 6 ∧
  (∀ a b c, a > 0 → b > 0 → c > 0 → 
    (a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2) / (a * b * c) = 48 * Real.sqrt 6 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3666_366675


namespace NUMINAMATH_CALUDE_golden_ratio_range_l3666_366692

theorem golden_ratio_range : 
  let φ := (Real.sqrt 5 - 1) / 2
  0.6 < φ ∧ φ < 0.7 := by sorry

end NUMINAMATH_CALUDE_golden_ratio_range_l3666_366692


namespace NUMINAMATH_CALUDE_curve_transformation_l3666_366629

theorem curve_transformation (x : ℝ) : 
  Real.sin (4 * x + π / 3) = Real.cos (2 * (x - π / 24)) := by
  sorry

end NUMINAMATH_CALUDE_curve_transformation_l3666_366629


namespace NUMINAMATH_CALUDE_equal_bills_at_80_minutes_l3666_366646

/-- United Telephone's base rate in dollars -/
def united_base : ℚ := 8

/-- United Telephone's per-minute rate in dollars -/
def united_per_minute : ℚ := 1/4

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℚ := 12

/-- Atlantic Call's per-minute rate in dollars -/
def atlantic_per_minute : ℚ := 1/5

/-- The number of minutes at which the bills are equal -/
def equal_bill_minutes : ℚ := 80

theorem equal_bills_at_80_minutes :
  united_base + united_per_minute * equal_bill_minutes =
  atlantic_base + atlantic_per_minute * equal_bill_minutes :=
by sorry

end NUMINAMATH_CALUDE_equal_bills_at_80_minutes_l3666_366646


namespace NUMINAMATH_CALUDE_rice_weight_per_container_l3666_366694

/-- 
Given a bag of rice weighing sqrt(50) pounds divided equally into 7 containers,
prove that the weight of rice in each container, in ounces, is (80 * sqrt(2)) / 7,
assuming 1 pound = 16 ounces.
-/
theorem rice_weight_per_container 
  (total_weight : ℝ) 
  (num_containers : ℕ) 
  (pounds_to_ounces : ℝ) 
  (h1 : total_weight = Real.sqrt 50)
  (h2 : num_containers = 7)
  (h3 : pounds_to_ounces = 16) :
  (total_weight / num_containers) * pounds_to_ounces = (80 * Real.sqrt 2) / 7 := by
  sorry

end NUMINAMATH_CALUDE_rice_weight_per_container_l3666_366694


namespace NUMINAMATH_CALUDE_fraction_sum_of_squares_is_integer_l3666_366615

theorem fraction_sum_of_squares_is_integer (a b : ℚ) 
  (h1 : ∃ k : ℤ, a + b = k) 
  (h2 : ∃ m : ℤ, a * b / (a + b) = m) : 
  ∃ n : ℤ, (a^2 + b^2) / (a + b) = n := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_of_squares_is_integer_l3666_366615


namespace NUMINAMATH_CALUDE_penelope_candy_count_l3666_366634

/-- Given a ratio of M&M candies to Starbursts candies and a number of M&M candies,
    calculate the number of Starbursts candies. -/
def calculate_starbursts (mm_ratio : ℕ) (starbursts_ratio : ℕ) (mm_count : ℕ) : ℕ :=
  (mm_count / mm_ratio) * starbursts_ratio

/-- Theorem stating that given 5 M&M candies for every 3 Starbursts candies,
    and 25 M&M candies, there are 15 Starbursts candies. -/
theorem penelope_candy_count :
  calculate_starbursts 5 3 25 = 15 := by
  sorry

end NUMINAMATH_CALUDE_penelope_candy_count_l3666_366634


namespace NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_is_five_thirds_l3666_366660

/-- Represents a square subdivided into smaller squares --/
structure SubdividedSquare where
  -- The side length of the largest square
  side_length : ℝ
  -- The number of subdivisions (levels of recursion)
  subdivisions : ℕ

/-- Calculates the ratio of shaded area to unshaded area in a subdivided square --/
def shaded_to_unshaded_ratio (square : SubdividedSquare) : ℚ :=
  5 / 3

/-- Theorem stating that the ratio of shaded to unshaded area is 5/3 --/
theorem shaded_to_unshaded_ratio_is_five_thirds (square : SubdividedSquare) :
  shaded_to_unshaded_ratio square = 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_is_five_thirds_l3666_366660


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l3666_366662

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 24 * x + k = 0) ↔ k = 12 := by
  sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l3666_366662


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3666_366683

theorem sum_of_roots_quadratic (b : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ + b = 0) → (x₂^2 - 2*x₂ + b = 0) → x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3666_366683


namespace NUMINAMATH_CALUDE_five_solutions_l3666_366678

/-- The number of distinct ordered pairs of positive integers satisfying the equation -/
def count_solutions : ℕ := 5

/-- The equation that the ordered pairs must satisfy -/
def satisfies_equation (x y : ℕ+) : Prop :=
  (x.val ^ 4 * y.val ^ 4) - (20 * x.val ^ 2 * y.val ^ 2) + 64 = 0

/-- The theorem stating that there are exactly 5 distinct ordered pairs satisfying the equation -/
theorem five_solutions :
  (∃! (s : Finset (ℕ+ × ℕ+)), s.card = count_solutions ∧
    ∀ p ∈ s, satisfies_equation p.1 p.2 ∧
    ∀ p : ℕ+ × ℕ+, satisfies_equation p.1 p.2 → p ∈ s) :=
  sorry

end NUMINAMATH_CALUDE_five_solutions_l3666_366678


namespace NUMINAMATH_CALUDE_inequality_chain_l3666_366696

theorem inequality_chain (m n : ℝ) 
  (hm : m < 0) 
  (hn : n > 0) 
  (hmn : m + n < 0) : 
  m < -n ∧ -n < n ∧ n < -m :=
by sorry

end NUMINAMATH_CALUDE_inequality_chain_l3666_366696


namespace NUMINAMATH_CALUDE_victoria_snack_money_l3666_366641

theorem victoria_snack_money (initial_amount : ℕ) 
  (pizza_cost : ℕ) (pizza_quantity : ℕ)
  (juice_cost : ℕ) (juice_quantity : ℕ) :
  initial_amount = 50 →
  pizza_cost = 12 →
  pizza_quantity = 2 →
  juice_cost = 2 →
  juice_quantity = 2 →
  initial_amount - (pizza_cost * pizza_quantity + juice_cost * juice_quantity) = 22 := by
sorry


end NUMINAMATH_CALUDE_victoria_snack_money_l3666_366641


namespace NUMINAMATH_CALUDE_f_range_l3666_366626

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x + 1) / (x + 2)

theorem f_range :
  Set.range f = {y : ℝ | y < 1 ∨ y > 1} :=
sorry

end NUMINAMATH_CALUDE_f_range_l3666_366626


namespace NUMINAMATH_CALUDE_function_f_at_zero_l3666_366642

/-- A function f: ℝ → ℝ satisfying f(x+y) = f(x) + f(y) + 1/2 for all real x and y -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + (1/2 : ℝ)

/-- Theorem: For a function f satisfying the given property, f(0) = -1/2 -/
theorem function_f_at_zero (f : ℝ → ℝ) (h : FunctionF f) : f 0 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_f_at_zero_l3666_366642


namespace NUMINAMATH_CALUDE_range_of_a_l3666_366697

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| ≤ 1}
def N (a : ℝ) : Set ℝ := {x | (x - a) * (x - a - 3) ≤ 0}

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, 
  (∀ x : ℝ, x ∈ M → x ∈ N a) ∧ 
  (∃ x : ℝ, x ∈ N a ∧ x ∉ M) → 
  a ∈ Set.Icc (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3666_366697


namespace NUMINAMATH_CALUDE_factors_of_x4_minus_4_l3666_366650

theorem factors_of_x4_minus_4 (x : ℝ) : 
  (x^4 - 4 = (x^2 + 2) * (x^2 - 2)) ∧ 
  (x^4 - 4 = (x^2 - 4) * (x^2 + 4)) ∧ 
  (x^4 - 4 ≠ (x + 1) * ((x^3 - x^2 - x + 5) / (x + 1))) ∧ 
  (x^4 - 4 ≠ (x^2 - 2*x + 2) * ((x^2 + 2*x + 2) / (x^2 - 2*x + 2))) :=
by sorry

end NUMINAMATH_CALUDE_factors_of_x4_minus_4_l3666_366650


namespace NUMINAMATH_CALUDE_two_balls_picked_l3666_366698

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  red : Nat
  blue : Nat
  green : Nat

/-- Calculates the total number of balls in the bag -/
def totalBalls (bag : BagContents) : Nat :=
  bag.red + bag.blue + bag.green

/-- Calculates the probability of picking two red balls -/
def probTwoRed (bag : BagContents) (picked : Nat) : Rat :=
  if picked ≠ 2 then 0
  else
    let total := totalBalls bag
    (bag.red : Rat) / total * ((bag.red - 1) : Rat) / (total - 1)

theorem two_balls_picked (bag : BagContents) (picked : Nat) :
  bag.red = 4 → bag.blue = 3 → bag.green = 2 →
  probTwoRed bag picked = 1/6 →
  picked = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_balls_picked_l3666_366698


namespace NUMINAMATH_CALUDE_julio_lime_cost_l3666_366658

/-- Represents the cost of limes for Julio's mocktails over 30 days -/
def lime_cost (mocktails_per_day : ℕ) (lime_juice_per_mocktail : ℚ) (juice_per_lime : ℚ) (days : ℕ) (limes_per_dollar : ℕ) : ℚ :=
  let limes_needed := (mocktails_per_day * lime_juice_per_mocktail * days) / juice_per_lime
  let lime_sets := (limes_needed / limes_per_dollar).ceil
  lime_sets

theorem julio_lime_cost :
  lime_cost 1 (1/2) 2 30 3 = 5 := by
  sorry

#eval lime_cost 1 (1/2) 2 30 3

end NUMINAMATH_CALUDE_julio_lime_cost_l3666_366658


namespace NUMINAMATH_CALUDE_u_general_term_l3666_366628

def u : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | 2 => 0
  | (n + 3) => 2 * u (n + 2) + u (n + 1) - 2 * u n

theorem u_general_term : ∀ n : ℕ, u n = 2 - (2/3) * (-1)^n - (1/3) * 2^n := by
  sorry

end NUMINAMATH_CALUDE_u_general_term_l3666_366628


namespace NUMINAMATH_CALUDE_books_given_to_friend_l3666_366635

/-- Given that Paul initially had 134 books, sold 27 books, and was left with 68 books
    after giving some to his friend and selling in the garage sale,
    prove that the number of books Paul gave to his friend is 39. -/
theorem books_given_to_friend :
  ∀ (initial_books sold_books remaining_books books_to_friend : ℕ),
    initial_books = 134 →
    sold_books = 27 →
    remaining_books = 68 →
    initial_books - sold_books - books_to_friend = remaining_books →
    books_to_friend = 39 := by
  sorry

end NUMINAMATH_CALUDE_books_given_to_friend_l3666_366635


namespace NUMINAMATH_CALUDE_complex_square_root_of_negative_two_l3666_366693

theorem complex_square_root_of_negative_two (z : ℂ) : z^2 + 2 = 0 → z = Complex.I * Real.sqrt 2 ∨ z = -Complex.I * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_of_negative_two_l3666_366693


namespace NUMINAMATH_CALUDE_square_area_proof_l3666_366668

theorem square_area_proof (x : ℝ) 
  (h1 : 4 * x - 15 = 20 - 3 * x) : 
  (4 * x - 15) ^ 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l3666_366668


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3666_366617

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3666_366617


namespace NUMINAMATH_CALUDE_triangle_area_angle_l3666_366656

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S = (√3/4)(a² + b² - c²), then C = π/3 -/
theorem triangle_area_angle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)
  S = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) →
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_angle_l3666_366656


namespace NUMINAMATH_CALUDE_bugs_eat_flowers_l3666_366670

/-- The number of flowers eaten by a group of bugs -/
def flowers_eaten (num_bugs : ℕ) (flowers_per_bug : ℕ) : ℕ :=
  num_bugs * flowers_per_bug

/-- Theorem: Given 3 bugs, each eating 2 flowers, the total number of flowers eaten is 6 -/
theorem bugs_eat_flowers :
  flowers_eaten 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bugs_eat_flowers_l3666_366670


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l3666_366631

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.log x / Real.log (1/2))^2 - 2 * (Real.log x / Real.log (1/2)) + 1

theorem f_increasing_on_interval :
  StrictMonoOn f { x : ℝ | x ≥ Real.sqrt 2 / 2 } :=
sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l3666_366631


namespace NUMINAMATH_CALUDE_final_value_calculation_l3666_366637

def initial_value : ℝ := 1500

def first_increase (x : ℝ) : ℝ := x * 1.20

def second_decrease (x : ℝ) : ℝ := x * 0.85

def third_increase (x : ℝ) : ℝ := x * 1.10

theorem final_value_calculation :
  third_increase (second_decrease (first_increase initial_value)) = 1683 := by
  sorry

end NUMINAMATH_CALUDE_final_value_calculation_l3666_366637


namespace NUMINAMATH_CALUDE_bread_theorem_l3666_366618

def bread_problem (slices_per_loaf : ℕ) (num_friends : ℕ) (num_loaves : ℕ) : ℕ :=
  (slices_per_loaf * num_loaves) / num_friends

theorem bread_theorem :
  bread_problem 15 10 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bread_theorem_l3666_366618


namespace NUMINAMATH_CALUDE_triangle_angle_relationships_l3666_366661

/-- Given two triangles ABC and UVW with the specified side relationships,
    prove that ABC is acute-angled and express angles of UVW in terms of ABC. -/
theorem triangle_angle_relationships
  (a b c u v w : ℝ)
  (ha : a^2 = u * (v + w - u))
  (hb : b^2 = v * (w + u - v))
  (hc : c^2 = w * (u + v - w))
  : (a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) ∧
    ∃ (A B C U V W : ℝ),
    (0 < A ∧ A < π / 2) ∧
    (0 < B ∧ B < π / 2) ∧
    (0 < C ∧ C < π / 2) ∧
    (A + B + C = π) ∧
    (U = π - 2 * A) ∧
    (V = π - 2 * B) ∧
    (W = π - 2 * C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relationships_l3666_366661


namespace NUMINAMATH_CALUDE_pond_depth_l3666_366647

/-- Proves that a rectangular pond with given dimensions has a depth of 5 meters -/
theorem pond_depth (length width volume : ℝ) (h1 : length = 20) (h2 : width = 10) (h3 : volume = 1000) :
  volume / (length * width) = 5 := by
  sorry

end NUMINAMATH_CALUDE_pond_depth_l3666_366647


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3666_366669

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_prod : a 2 * a 4 = 1/2) :
  a 1 * a 3^2 * a 5 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3666_366669


namespace NUMINAMATH_CALUDE_B_power_five_eq_scalar_multiple_l3666_366689

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 6]

theorem B_power_five_eq_scalar_multiple :
  B^5 = (4096 : ℝ) • B := by sorry

end NUMINAMATH_CALUDE_B_power_five_eq_scalar_multiple_l3666_366689


namespace NUMINAMATH_CALUDE_exists_m_n_for_k_l3666_366686

theorem exists_m_n_for_k (k : ℕ) : 
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_exists_m_n_for_k_l3666_366686


namespace NUMINAMATH_CALUDE_group_50_properties_l3666_366605

def last_number (n : ℕ) : ℕ := 2 * (n * (n + 1) / 2)

def first_number (n : ℕ) : ℕ := last_number n - 2 * (n - 1)

def sum_of_group (n : ℕ) : ℕ := n * (first_number n + last_number n) / 2

theorem group_50_properties :
  last_number 50 = 2550 ∧
  first_number 50 = 2452 ∧
  sum_of_group 50 = 50 * 2501 := by
  sorry

end NUMINAMATH_CALUDE_group_50_properties_l3666_366605


namespace NUMINAMATH_CALUDE_first_group_size_l3666_366695

/-- The number of students in the first group -/
def first_group_count : ℕ := sorry

/-- The number of students in the second group -/
def second_group_count : ℕ := 11

/-- The total number of students in both groups -/
def total_students : ℕ := 31

/-- The average height of students in centimeters -/
def average_height : ℝ := 20

theorem first_group_size :
  (first_group_count : ℝ) * average_height +
  (second_group_count : ℝ) * average_height =
  (total_students : ℝ) * average_height ∧
  first_group_count + second_group_count = total_students →
  first_group_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l3666_366695


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3666_366652

/-- 
Given x = γ sin((θ - α)/2) and y = γ sin((θ + α)/2), 
prove that x^2 - 2xy cos α + y^2 = γ^2 sin^2 α
-/
theorem trigonometric_identity 
  (γ θ α x y : ℝ) 
  (hx : x = γ * Real.sin ((θ - α) / 2))
  (hy : y = γ * Real.sin ((θ + α) / 2)) :
  x^2 - 2*x*y*Real.cos α + y^2 = γ^2 * Real.sin α^2 := by
  sorry


end NUMINAMATH_CALUDE_trigonometric_identity_l3666_366652


namespace NUMINAMATH_CALUDE_derrick_has_34_pictures_l3666_366609

/-- The number of wild animal pictures Ralph has -/
def ralph_pictures : ℕ := 26

/-- The additional number of pictures Derrick has compared to Ralph -/
def additional_pictures : ℕ := 8

/-- The number of wild animal pictures Derrick has -/
def derrick_pictures : ℕ := ralph_pictures + additional_pictures

/-- Theorem stating that Derrick has 34 wild animal pictures -/
theorem derrick_has_34_pictures : derrick_pictures = 34 := by sorry

end NUMINAMATH_CALUDE_derrick_has_34_pictures_l3666_366609


namespace NUMINAMATH_CALUDE_candy_distribution_l3666_366624

/-- Represents the number of candies eaten by each person -/
structure CandyEaten where
  andrey : ℕ
  boris : ℕ
  denis : ℕ

/-- Represents the relative eating rates of the three people -/
structure EatingRates where
  andrey_boris : ℚ  -- Ratio of Andrey's rate to Boris's rate
  andrey_denis : ℚ  -- Ratio of Andrey's rate to Denis's rate

/-- Given the eating rates and total candies eaten, calculate how many candies each person ate -/
def calculate_candy_eaten (rates : EatingRates) (total : ℕ) : CandyEaten :=
  sorry

/-- The main theorem to prove -/
theorem candy_distribution (rates : EatingRates) (total : ℕ) :
  rates.andrey_boris = 4/3 →
  rates.andrey_denis = 6/7 →
  total = 70 →
  let result := calculate_candy_eaten rates total
  result.andrey = 24 ∧ result.boris = 18 ∧ result.denis = 28 :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3666_366624


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3666_366601

theorem consecutive_integers_product_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3666_366601


namespace NUMINAMATH_CALUDE_first_term_range_l3666_366600

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 1 / (2 - a n)

/-- The property that each term is greater than the previous one -/
def StrictlyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

/-- The main theorem stating the range of the first term -/
theorem first_term_range
  (a : ℕ → ℝ)
  (h_recurrence : RecurrenceSequence a)
  (h_increasing : StrictlyIncreasing a) :
  a 1 < 1 :=
sorry

end NUMINAMATH_CALUDE_first_term_range_l3666_366600


namespace NUMINAMATH_CALUDE_rectangle_tiling_tiling_count_l3666_366630

/-- A piece is a shape that can be used to tile a rectangle -/
structure Piece where
  shape : Set (ℕ × ℕ)

/-- A tiling is a way to cover a rectangle with pieces -/
def Tiling (m n : ℕ) (pieces : Finset Piece) :=
  Set (ℕ × ℕ × Piece)

/-- The number of ways to tile a 5 × 2k rectangle with 2k pieces -/
def TilingCount (k : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem rectangle_tiling (n : ℕ) (pieces : Finset Piece) :
  (∃ (t : Tiling 5 n pieces), pieces.card = n) → Even n :=
sorry

/-- The counting theorem -/
theorem tiling_count (k : ℕ) :
  k ≥ 3 → TilingCount k > 2 * 3^(k - 1) :=
sorry

end NUMINAMATH_CALUDE_rectangle_tiling_tiling_count_l3666_366630


namespace NUMINAMATH_CALUDE_divisibility_problem_l3666_366608

theorem divisibility_problem (x y : ℕ) : 
  (∀ z : ℕ, z < x → ¬((1056 + z) % 28 = 0 ∧ (1056 + z) % 42 = 0)) ∧
  ((1056 + x) % 28 = 0 ∧ (1056 + x) % 42 = 0) ∧
  (∀ w : ℕ, w > y → ¬((1056 - w) % 28 = 0 ∧ (1056 - w) % 42 = 0)) ∧
  ((1056 - y) % 28 = 0 ∧ (1056 - y) % 42 = 0) →
  x = 36 ∧ y = 48 := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3666_366608


namespace NUMINAMATH_CALUDE_time_to_finish_problems_l3666_366667

/-- The time required to finish all problems given the number of math and spelling problems and the rate of problem-solving. -/
theorem time_to_finish_problems
  (math_problems : ℕ)
  (spelling_problems : ℕ)
  (problems_per_hour : ℕ)
  (h1 : math_problems = 18)
  (h2 : spelling_problems = 6)
  (h3 : problems_per_hour = 4) :
  (math_problems + spelling_problems) / problems_per_hour = 6 :=
by sorry

end NUMINAMATH_CALUDE_time_to_finish_problems_l3666_366667


namespace NUMINAMATH_CALUDE_parabola_points_distance_l3666_366613

/-- A parabola defined by y = 9x^2 - 3x + 2 -/
def parabola (x y : ℝ) : Prop := y = 9 * x^2 - 3 * x + 2

/-- The origin (0,0) is the midpoint of two points -/
def origin_is_midpoint (p q : ℝ × ℝ) : Prop :=
  (p.1 + q.1) / 2 = 0 ∧ (p.2 + q.2) / 2 = 0

/-- The square of the distance between two points -/
def square_distance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem parabola_points_distance (p q : ℝ × ℝ) :
  parabola p.1 p.2 ∧ parabola q.1 q.2 ∧ origin_is_midpoint p q →
  square_distance p q = 580 / 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_distance_l3666_366613


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3666_366603

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

/-- Given conditions for the geometric sequence -/
def GeometricSequenceConditions (a : ℕ → ℝ) : Prop :=
  GeometricSequence a ∧ (a 5 * a 11 = 4) ∧ (a 3 + a 13 = 5)

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h : GeometricSequenceConditions a) : 
  (a 14 / a 4 = 4) ∨ (a 14 / a 4 = 1/4) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3666_366603


namespace NUMINAMATH_CALUDE_probability_yellow_ball_l3666_366632

def total_balls : ℕ := 5
def white_balls : ℕ := 2
def yellow_balls : ℕ := 3

theorem probability_yellow_ball :
  (yellow_balls : ℚ) / total_balls = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_yellow_ball_l3666_366632


namespace NUMINAMATH_CALUDE_twenty_cent_items_count_l3666_366681

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents20 : ℕ
  dollars150 : ℕ
  dollars250 : ℕ

/-- Checks if the given item counts satisfy the problem conditions -/
def satisfiesConditions (counts : ItemCounts) : Prop :=
  counts.cents20 + counts.dollars150 + counts.dollars250 = 50 ∧
  20 * counts.cents20 + 150 * counts.dollars150 + 250 * counts.dollars250 = 5000

/-- Theorem stating that the number of 20-cent items is 31 -/
theorem twenty_cent_items_count :
  ∃ (counts : ItemCounts), satisfiesConditions counts ∧ counts.cents20 = 31 := by
  sorry

end NUMINAMATH_CALUDE_twenty_cent_items_count_l3666_366681


namespace NUMINAMATH_CALUDE_parabola_directrix_l3666_366638

/-- The equation of the directrix of the parabola y = 4x^2 -/
theorem parabola_directrix (x y : ℝ) :
  (y = 4 * x^2) →  -- Given parabola equation
  ∃ (d : ℝ), d = -1/16 ∧ (∀ (x₀ y₀ : ℝ), y₀ = 4 * x₀^2 → y₀ ≥ d) ∧
              (∀ ε > 0, ∃ (x₁ y₁ : ℝ), y₁ = 4 * x₁^2 ∧ y₁ < d + ε) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3666_366638


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3666_366651

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 + a 3 = 4) →
  (a 4 + a 5 = 16) →
  (a 8 + a 9 = 256) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3666_366651


namespace NUMINAMATH_CALUDE_inverse_contrapositive_l3666_366684

theorem inverse_contrapositive (x y : ℝ) : x = 0 ∧ y = 2 → x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_contrapositive_l3666_366684


namespace NUMINAMATH_CALUDE_triangle_inequality_with_square_roots_l3666_366682

/-- Given a triangle with sides a, b, and c, the sum of the square roots of the semiperimeter minus each side is less than or equal to the sum of the square roots of the sides. Equality holds if and only if the triangle is equilateral. -/
theorem triangle_inequality_with_square_roots (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) ≤ 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ∧
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) = 
   Real.sqrt a + Real.sqrt b + Real.sqrt c ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_square_roots_l3666_366682


namespace NUMINAMATH_CALUDE_jasons_hardcover_books_l3666_366672

/-- Proves that Jason has 70 hardcover books given the problem conditions --/
theorem jasons_hardcover_books :
  let bookcase_limit : ℕ := 80
  let hardcover_weight : ℚ := 1/2
  let textbook_count : ℕ := 30
  let textbook_weight : ℕ := 2
  let knickknack_count : ℕ := 3
  let knickknack_weight : ℕ := 6
  let over_limit : ℕ := 33
  
  let total_weight : ℕ := bookcase_limit + over_limit
  let textbook_total_weight : ℕ := textbook_count * textbook_weight
  let knickknack_total_weight : ℕ := knickknack_count * knickknack_weight
  let hardcover_total_weight : ℕ := total_weight - textbook_total_weight - knickknack_total_weight
  
  (hardcover_total_weight : ℚ) / hardcover_weight = 70 := by sorry

end NUMINAMATH_CALUDE_jasons_hardcover_books_l3666_366672


namespace NUMINAMATH_CALUDE_max_value_of_sum_l3666_366655

theorem max_value_of_sum (a b c : ℝ) (h : a^2 + b^2 + c^2 = 4) :
  ∃ (max : ℝ), max = 10 * Real.sqrt 2 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 = 4 → 3*x + 4*y + 5*z ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l3666_366655


namespace NUMINAMATH_CALUDE_plain_croissant_price_l3666_366679

/-- The price of Sean's Sunday pastry purchase --/
def sean_pastry_purchase : ℝ → Prop :=
  fun plain_croissant_price =>
    let almond_croissant_price : ℝ := 4.50
    let salami_cheese_croissant_price : ℝ := 4.50
    let focaccia_price : ℝ := 4.00
    let latte_price : ℝ := 2.50
    let total_spent : ℝ := 21.00
    
    almond_croissant_price +
    salami_cheese_croissant_price +
    plain_croissant_price +
    focaccia_price +
    2 * latte_price = total_spent

theorem plain_croissant_price : ∃ (price : ℝ), sean_pastry_purchase price ∧ price = 3.00 := by
  sorry

end NUMINAMATH_CALUDE_plain_croissant_price_l3666_366679


namespace NUMINAMATH_CALUDE_solution_of_equations_solution_of_inequalities_l3666_366607

-- Part 1: System of Equations
def system_of_equations (x y : ℝ) : Prop :=
  2 * x - y = 3 ∧ 3 * x + 2 * y = 22

theorem solution_of_equations : 
  ∃ x y : ℝ, system_of_equations x y ∧ x = 4 ∧ y = 5 := by sorry

-- Part 2: System of Inequalities
def system_of_inequalities (x : ℝ) : Prop :=
  (x - 2) / 2 + 1 < (x + 1) / 3 ∧ 5 * x + 1 ≥ 2 * (2 + x)

theorem solution_of_inequalities : 
  ∀ x : ℝ, system_of_inequalities x ↔ 1 ≤ x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_solution_of_equations_solution_of_inequalities_l3666_366607


namespace NUMINAMATH_CALUDE_face_masks_per_box_l3666_366653

/-- Proves the number of face masks in each box given the problem conditions --/
theorem face_masks_per_box :
  ∀ (num_boxes : ℕ) (sell_price : ℚ) (total_cost : ℚ) (total_profit : ℚ),
    num_boxes = 3 →
    sell_price = 1/2 →
    total_cost = 15 →
    total_profit = 15 →
    ∃ (masks_per_box : ℕ),
      masks_per_box = 20 ∧
      (num_boxes * masks_per_box : ℚ) * sell_price - total_cost = total_profit :=
by
  sorry


end NUMINAMATH_CALUDE_face_masks_per_box_l3666_366653


namespace NUMINAMATH_CALUDE_gcd_operation_result_l3666_366654

theorem gcd_operation_result : (Nat.gcd 7350 165 - 15) * 3 = 0 := by sorry

end NUMINAMATH_CALUDE_gcd_operation_result_l3666_366654


namespace NUMINAMATH_CALUDE_cubic_equation_with_double_root_l3666_366622

theorem cubic_equation_with_double_root (k : ℝ) : 
  (∃ a b : ℝ, (3 * a^3 - 9 * a^2 - 81 * a + k = 0) ∧ 
               (3 * (2*a)^3 - 9 * (2*a)^2 - 81 * (2*a) + k = 0) ∧ 
               (3 * b^3 - 9 * b^2 - 81 * b + k = 0) ∧ 
               (a ≠ b) ∧ (k > 0)) →
  k = -6 * ((9 + Real.sqrt 837) / 14)^2 * (3 - 3 * ((9 + Real.sqrt 837) / 14)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_with_double_root_l3666_366622


namespace NUMINAMATH_CALUDE_custom_multiplication_prove_specific_case_l3666_366671

theorem custom_multiplication (x y : ℤ) : x * y = x * y - 2 * (x + y) := by sorry

theorem prove_specific_case : 1 * (-3) = 1 := by sorry

end NUMINAMATH_CALUDE_custom_multiplication_prove_specific_case_l3666_366671


namespace NUMINAMATH_CALUDE_curve_cartesian_to_polar_l3666_366663

/-- Given a curve C in the Cartesian coordinate system described by the parametric equations
    x = cos α and y = sin α + 1, prove that its polar equation is ρ = 2 sin θ. -/
theorem curve_cartesian_to_polar (α θ : Real) (ρ : Real) (x y : Real) :
  (x = Real.cos α ∧ y = Real.sin α + 1) →
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  ρ = 2 * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_curve_cartesian_to_polar_l3666_366663


namespace NUMINAMATH_CALUDE_helen_amy_height_difference_l3666_366688

/-- Given the heights of Angela, Amy, and the height difference between Angela and Helen,
    prove that Helen is 3 cm taller than Amy. -/
theorem helen_amy_height_difference
  (angela_height : ℕ)
  (amy_height : ℕ)
  (angela_helen_diff : ℕ)
  (h1 : angela_height = 157)
  (h2 : amy_height = 150)
  (h3 : angela_height = angela_helen_diff + helen_height)
  (helen_height : ℕ) :
  helen_height - amy_height = 3 :=
sorry

end NUMINAMATH_CALUDE_helen_amy_height_difference_l3666_366688


namespace NUMINAMATH_CALUDE_general_term_formula_l3666_366633

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n - 2 * a (n + 1) + a (n + 2) = 0

/-- Theorem stating the general term formula for the sequence -/
theorem general_term_formula (a : ℕ+ → ℝ) 
    (h_recurrence : RecurrenceSequence a) 
    (h_initial1 : a 1 = 2) 
    (h_initial2 : a 2 = 4) : 
    ∀ n : ℕ+, a n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_general_term_formula_l3666_366633


namespace NUMINAMATH_CALUDE_line_contains_point_l3666_366657

/-- A line in the xy-plane is represented by the equation 2 - kx = -4y for some real number k. -/
def line (k : ℝ) (x y : ℝ) : Prop := 2 - k * x = -4 * y

/-- The point (2, -1) lies on the line. -/
def point_on_line (k : ℝ) : Prop := line k 2 (-1)

/-- The value of k for which the line contains the point (2, -1) is -1. -/
theorem line_contains_point : ∃! k : ℝ, point_on_line k ∧ k = -1 := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l3666_366657


namespace NUMINAMATH_CALUDE_sin_shift_l3666_366645

theorem sin_shift (x : ℝ) : Real.sin (2 * x + π / 4) = Real.sin (2 * (x - π / 8)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l3666_366645


namespace NUMINAMATH_CALUDE_mass_of_X_in_BaX_l3666_366666

/-- The molar mass of barium in g/mol -/
def molar_mass_Ba : ℝ := 137.33

/-- The mass percentage of barium in the compound -/
def mass_percentage_Ba : ℝ := 66.18

/-- The mass of the compound in grams -/
def total_mass : ℝ := 100

theorem mass_of_X_in_BaX : 
  let mass_Ba := total_mass * (mass_percentage_Ba / 100)
  let mass_X := total_mass - mass_Ba
  mass_X = 33.82 := by sorry

end NUMINAMATH_CALUDE_mass_of_X_in_BaX_l3666_366666


namespace NUMINAMATH_CALUDE_inverse_A_cubed_l3666_366680

def A_inv : Matrix (Fin 2) (Fin 2) ℤ := !![3, 8; -2, -5]

theorem inverse_A_cubed :
  let A := A_inv⁻¹
  (A^3)⁻¹ = !![5, 0; -66, -137] := by
  sorry

end NUMINAMATH_CALUDE_inverse_A_cubed_l3666_366680


namespace NUMINAMATH_CALUDE_find_y_l3666_366610

theorem find_y : ∃ y : ℚ, (12 : ℚ)^2 * (6 : ℚ)^3 / y = 72 → y = 432 := by sorry

end NUMINAMATH_CALUDE_find_y_l3666_366610


namespace NUMINAMATH_CALUDE_inequality_range_l3666_366691

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^4 + (a-2)*x^2 + a ≥ 0) ↔ a ≥ 4 - 2*Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l3666_366691


namespace NUMINAMATH_CALUDE_fib_even_iff_index_div_three_l3666_366676

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: A Fibonacci number is even if and only if its index is divisible by 3 -/
theorem fib_even_iff_index_div_three (n : ℕ) : Even (fib n) ↔ 3 ∣ n := by sorry

end NUMINAMATH_CALUDE_fib_even_iff_index_div_three_l3666_366676


namespace NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l3666_366625

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largest_perfect_square_divisor (n : ℕ) : ℕ :=
  sorry

def prime_factors (n : ℕ) : List ℕ :=
  sorry

def exponents_of_prime_factors (n : ℕ) : List ℕ :=
  sorry

theorem sum_of_exponents_15_factorial :
  let n := factorial 15
  let largest_square := largest_perfect_square_divisor n
  let square_root := Nat.sqrt largest_square
  let exponents := exponents_of_prime_factors square_root
  List.sum exponents = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l3666_366625


namespace NUMINAMATH_CALUDE_marble_selection_probability_l3666_366640

/-- The number of blue marbles -/
def blue_marbles : ℕ := 7

/-- The number of yellow marbles -/
def yellow_marbles : ℕ := 5

/-- The total number of selections -/
def total_selections : ℕ := 7

/-- The number of blue marbles we want to select after the first yellow -/
def target_blue : ℕ := 3

/-- The probability of the described event -/
def probability : ℚ := 214375 / 1492992

theorem marble_selection_probability :
  (yellow_marbles : ℚ) / (yellow_marbles + blue_marbles) *
  (Nat.choose (total_selections - 1) target_blue : ℚ) *
  (blue_marbles ^ target_blue * yellow_marbles ^ (total_selections - target_blue - 1) : ℚ) /
  ((yellow_marbles + blue_marbles) ^ (total_selections - 1)) = probability :=
sorry

end NUMINAMATH_CALUDE_marble_selection_probability_l3666_366640


namespace NUMINAMATH_CALUDE_probability_two_twos_in_five_rolls_probability_two_twos_in_five_rolls_proof_l3666_366664

/-- The probability of rolling a 2 exactly two times in five rolls of a fair eight-sided die -/
theorem probability_two_twos_in_five_rolls : ℝ :=
let p : ℝ := 1 / 8  -- probability of rolling a 2
let q : ℝ := 1 - p  -- probability of not rolling a 2
let n : ℕ := 5      -- number of rolls
let k : ℕ := 2      -- number of desired successes
3430 / 32768

/-- Proof that the probability is correct -/
theorem probability_two_twos_in_five_rolls_proof :
  probability_two_twos_in_five_rolls = 3430 / 32768 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_twos_in_five_rolls_probability_two_twos_in_five_rolls_proof_l3666_366664


namespace NUMINAMATH_CALUDE_middle_of_three_consecutive_sum_30_l3666_366611

theorem middle_of_three_consecutive_sum_30 (a b c : ℕ) :
  (a + 1 = b) ∧ (b + 1 = c) ∧ (a + b + c = 30) → b = 10 := by
  sorry

end NUMINAMATH_CALUDE_middle_of_three_consecutive_sum_30_l3666_366611


namespace NUMINAMATH_CALUDE_algebra_test_female_students_l3666_366643

theorem algebra_test_female_students 
  (total_average : ℝ) 
  (num_male : ℕ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (h1 : total_average = 88) 
  (h2 : num_male = 15) 
  (h3 : male_average = 80) 
  (h4 : female_average = 94) : 
  ∃ (num_female : ℕ), 
    (num_male * male_average + num_female * female_average) / (num_male + num_female) = total_average ∧ 
    num_female = 20 := by
sorry


end NUMINAMATH_CALUDE_algebra_test_female_students_l3666_366643


namespace NUMINAMATH_CALUDE_base_ten_arithmetic_l3666_366614

theorem base_ten_arithmetic : (456 + 123) - 579 = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_ten_arithmetic_l3666_366614


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l3666_366602

/-- The perimeter of a T shape formed by a vertical rectangle and a horizontal rectangle -/
def t_perimeter (v_width v_height h_width h_height : ℝ) : ℝ :=
  2 * v_height + 2 * h_width + h_height

/-- Theorem: The perimeter of the T shape is 22 inches -/
theorem t_shape_perimeter :
  t_perimeter 2 6 3 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l3666_366602


namespace NUMINAMATH_CALUDE_annie_brownies_left_l3666_366690

/-- Calculates the number of brownies Annie has left after sharing -/
def brownies_left (initial : ℕ) (to_simon : ℕ) : ℕ :=
  let to_admin := initial / 2
  let after_admin := initial - to_admin
  let to_carl := after_admin / 2
  let after_carl := after_admin - to_carl
  after_carl - to_simon

/-- Proves that Annie has 3 brownies left after sharing -/
theorem annie_brownies_left :
  brownies_left 20 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_annie_brownies_left_l3666_366690


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3666_366612

def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_properties (f : ℝ → ℝ) 
  (h_quadratic : quadratic_function f)
  (h_f_0 : f 0 = 1)
  (h_f_diff : ∀ x, f (x + 1) - f x = 2 * x) :
  (∀ x, f x = x^2 - x + 1) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≤ 3) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≥ 3/4) ∧
  (∃ x ∈ Set.Icc (-1) 1, f x = 3) ∧
  (∃ x ∈ Set.Icc (-1) 1, f x = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3666_366612


namespace NUMINAMATH_CALUDE_probability_sum_twenty_l3666_366685

/-- A dodecahedral die with faces labeled 1 through 12 -/
def DodecahedralDie : Finset ℕ := Finset.range 12 

/-- The sample space of rolling two dodecahedral dice -/
def TwoDiceRolls : Finset (ℕ × ℕ) :=
  DodecahedralDie.product DodecahedralDie

/-- The event of rolling a sum of 20 with two dodecahedral dice -/
def SumTwenty : Finset (ℕ × ℕ) :=
  TwoDiceRolls.filter (fun p => p.1 + p.2 = 20)

/-- The probability of an event in a finite sample space -/
def probability (event : Finset α) (sampleSpace : Finset α) : ℚ :=
  event.card / sampleSpace.card

theorem probability_sum_twenty :
  probability SumTwenty TwoDiceRolls = 5 / 144 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_twenty_l3666_366685


namespace NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l3666_366665

/-- The line y = kx is tangent to the curve y = ln x if and only if k = 1/e -/
theorem tangent_line_to_ln_curve (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ k = 1 / x) ↔ k = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l3666_366665


namespace NUMINAMATH_CALUDE_optimal_position_C_l3666_366659

/-- The optimal position of point C on segment AB to maximize the length of CD -/
theorem optimal_position_C (t : ℝ) : 
  (0 ≤ t) → (t < 1) → 
  (∀ s, (0 ≤ s ∧ s < 1) → (t * (1 - t^2) / 4 ≥ s * (1 - s^2) / 4)) → 
  t = 1 / Real.sqrt 3 := by
  sorry

#check optimal_position_C

end NUMINAMATH_CALUDE_optimal_position_C_l3666_366659


namespace NUMINAMATH_CALUDE_sixty_degrees_in_vlecs_l3666_366687

/-- Represents the number of vlecs in a full circle on Venus -/
def full_circle_vlecs : ℕ := 800

/-- Represents the number of degrees in a full circle on Earth -/
def full_circle_degrees : ℕ := 360

/-- Represents the angle in degrees we want to convert to vlecs -/
def angle_degrees : ℕ := 60

/-- Converts an angle from degrees to vlecs -/
def degrees_to_vlecs (degrees : ℕ) : ℕ :=
  (degrees * full_circle_vlecs + full_circle_degrees / 2) / full_circle_degrees

theorem sixty_degrees_in_vlecs :
  degrees_to_vlecs angle_degrees = 133 := by
  sorry

end NUMINAMATH_CALUDE_sixty_degrees_in_vlecs_l3666_366687


namespace NUMINAMATH_CALUDE_stone_order_calculation_l3666_366648

theorem stone_order_calculation (total material_ordered concrete_ordered bricks_ordered stone_ordered : ℝ) :
  total_material_ordered = 0.83 ∧
  concrete_ordered = 0.17 ∧
  bricks_ordered = 0.17 ∧
  total_material_ordered = concrete_ordered + bricks_ordered + stone_ordered →
  stone_ordered = 0.49 := by
sorry

end NUMINAMATH_CALUDE_stone_order_calculation_l3666_366648


namespace NUMINAMATH_CALUDE_fraction_sum_proof_l3666_366620

theorem fraction_sum_proof : 
  let a : ℚ := 12 / 15
  let b : ℚ := 7 / 9
  let c : ℚ := 1 + 1 / 6
  let sum : ℚ := a + b + c
  sum = 247 / 90 ∧ (∀ n d : ℕ, n ≠ 0 → d ≠ 0 → (n : ℚ) / d = sum → n ≥ 247 ∧ d ≥ 90) :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_proof_l3666_366620


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l3666_366606

/-- Theorem: When the sides of a rectangle are increased by 35%, the area increases by 82.25% -/
theorem rectangle_area_increase (L W : ℝ) (L_pos : L > 0) (W_pos : W > 0) :
  let original_area := L * W
  let new_length := L * 1.35
  let new_width := W * 1.35
  let new_area := new_length * new_width
  (new_area - original_area) / original_area * 100 = 82.25 := by
  sorry

#check rectangle_area_increase

end NUMINAMATH_CALUDE_rectangle_area_increase_l3666_366606


namespace NUMINAMATH_CALUDE_function_identity_l3666_366649

theorem function_identity (f : ℕ → ℕ) (h : ∀ n : ℕ, f (n + 1) > f (f n)) :
  ∀ n : ℕ, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l3666_366649


namespace NUMINAMATH_CALUDE_game_points_total_l3666_366604

theorem game_points_total (eric_points mark_points samanta_points : ℕ) : 
  eric_points = 6 →
  mark_points = eric_points + eric_points / 2 →
  samanta_points = mark_points + 8 →
  eric_points + mark_points + samanta_points = 32 := by
sorry

end NUMINAMATH_CALUDE_game_points_total_l3666_366604


namespace NUMINAMATH_CALUDE_cubic_root_relation_l3666_366674

theorem cubic_root_relation (x₀ : ℝ) (z : ℝ) : 
  x₀^3 - x₀ - 1 = 0 →
  z = x₀^2 + 3 * x₀ + 1 →
  z^3 - 5*z^2 - 10*z - 11 = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_relation_l3666_366674
