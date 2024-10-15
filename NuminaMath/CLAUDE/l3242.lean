import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_l3242_324291

/-- Represents a four-digit number as individual digits -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10
  d_digit : d < 10

/-- Converts a four-digit number to its numerical value -/
def to_nat (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Converts a two-digit number to its numerical value -/
def two_digit_to_nat (a b : Nat) : Nat :=
  10 * a + b

/-- States that A̅B² = A̅CDB -/
def condition1 (n : FourDigitNumber) : Prop :=
  (two_digit_to_nat n.a n.b)^2 = to_nat n

/-- States that C̅D³ = A̅CBD -/
def condition2 (n : FourDigitNumber) : Prop :=
  (two_digit_to_nat n.c n.d)^3 = 1000 * n.a + 100 * n.c + 10 * n.b + n.d

/-- The main theorem stating that the only solution is A = 9, B = 6, C = 2, D = 1 -/
theorem unique_solution :
  ∀ n : FourDigitNumber, condition1 n ∧ condition2 n →
  n.a = 9 ∧ n.b = 6 ∧ n.c = 2 ∧ n.d = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3242_324291


namespace NUMINAMATH_CALUDE_f_lower_bound_a_range_l3242_324276

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x + 2*a + 3|

-- Theorem 1: f(x) ≥ 2 for all x and a
theorem f_lower_bound (x a : ℝ) : f x a ≥ 2 := by sorry

-- Theorem 2: If f(-3/2) < 3, then -1 < a < 0
theorem a_range (a : ℝ) : f (-3/2) a < 3 → -1 < a ∧ a < 0 := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_a_range_l3242_324276


namespace NUMINAMATH_CALUDE_watermelon_problem_l3242_324280

theorem watermelon_problem (initial_watermelons : ℕ) (total_watermelons : ℕ) 
  (h1 : initial_watermelons = 4)
  (h2 : total_watermelons = 7) :
  total_watermelons - initial_watermelons = 3 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_problem_l3242_324280


namespace NUMINAMATH_CALUDE_min_value_ab_l3242_324220

/-- Given that ab > 0 and points A(a, 0), B(0, b), and C(-2, -2) are collinear,
    the minimum value of ab is 16. -/
theorem min_value_ab (a b : ℝ) (hab : a * b > 0)
  (h_collinear : (a - 0) * (-2 - b) = (-2 - a) * (b - 0)) :
  ∀ x y : ℝ, x * y > 0 → (x - 0) * (-2 - y) = (-2 - x) * (y - 0) → a * b ≤ x * y → a * b = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l3242_324220


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3242_324278

theorem complex_equation_solution (z : ℂ) : z = -Complex.I / 7 ↔ 3 + 2 * Complex.I * z = 4 - 5 * Complex.I * z := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3242_324278


namespace NUMINAMATH_CALUDE_quadratic_equation_determination_l3242_324274

theorem quadratic_equation_determination (b c : ℝ) :
  (∀ x : ℝ, x^2 + b*x + c = 0 → x = 5 ∨ x = 3 ∨ x = -6 ∨ x = -4) →
  (5 + 3 = -b) →
  ((-6) * (-4) = c) →
  (b = -8 ∧ c = 24) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_determination_l3242_324274


namespace NUMINAMATH_CALUDE_eccentricity_relation_l3242_324254

-- Define the eccentricities and point coordinates
variable (e₁ e₂ : ℝ)
variable (O F₁ F₂ P : ℝ × ℝ)

-- Define the conditions
def is_standard_ellipse_hyperbola : Prop :=
  0 < e₁ ∧ e₁ < 1 ∧ e₂ > 1

def foci_on_x_axis : Prop :=
  ∃ c : ℝ, F₁ = (c, 0) ∧ F₂ = (-c, 0)

def O_is_origin : Prop :=
  O = (0, 0)

def P_on_both_curves : Prop :=
  ∃ (x y : ℝ), P = (x, y)

def distance_condition : Prop :=
  2 * ‖P - O‖ = ‖F₁ - F₂‖

-- State the theorem
theorem eccentricity_relation
  (h₁ : is_standard_ellipse_hyperbola e₁ e₂)
  (h₂ : foci_on_x_axis F₁ F₂)
  (h₃ : O_is_origin O)
  (h₄ : P_on_both_curves P)
  (h₅ : distance_condition O F₁ F₂ P) :
  (e₁ * e₂) / Real.sqrt (e₁^2 + e₂^2) = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_eccentricity_relation_l3242_324254


namespace NUMINAMATH_CALUDE_complex_equality_l3242_324241

theorem complex_equality (a : ℝ) : 
  (1 + (a - 2) * Complex.I).im = 0 → (a + Complex.I) / Complex.I = 1 - 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equality_l3242_324241


namespace NUMINAMATH_CALUDE_division_by_negative_fraction_l3242_324295

theorem division_by_negative_fraction :
  5 / (-1/2 : ℚ) = -10 := by sorry

end NUMINAMATH_CALUDE_division_by_negative_fraction_l3242_324295


namespace NUMINAMATH_CALUDE_candy_count_l3242_324239

/-- The number of bags of candy -/
def num_bags : ℕ := 26

/-- The number of candy pieces in each bag -/
def pieces_per_bag : ℕ := 33

/-- The total number of candy pieces -/
def total_pieces : ℕ := num_bags * pieces_per_bag

theorem candy_count : total_pieces = 858 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l3242_324239


namespace NUMINAMATH_CALUDE_derivative_evaluation_l3242_324257

theorem derivative_evaluation (x : ℝ) (h : x > 0) :
  let F : ℝ → ℝ := λ x => (1 - Real.sqrt x)^2 / x
  let F' : ℝ → ℝ := λ x => -1/x^2 + 1/x^(3/2)
  F' 0.01 = -9000 := by sorry

end NUMINAMATH_CALUDE_derivative_evaluation_l3242_324257


namespace NUMINAMATH_CALUDE_box_surface_area_l3242_324267

/-- Calculates the surface area of the interior of an open box formed from a rectangular cardboard with square corners removed. -/
def interior_surface_area (length width corner_size : ℕ) : ℕ :=
  length * width - 4 * (corner_size * corner_size)

/-- Proves that the surface area of the interior of the specified open box is 731 square units. -/
theorem box_surface_area : interior_surface_area 25 35 6 = 731 := by
  sorry

#eval interior_surface_area 25 35 6

end NUMINAMATH_CALUDE_box_surface_area_l3242_324267


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l3242_324256

noncomputable def f (x : ℝ) : ℝ := Real.sin 1 - Real.cos x

theorem f_derivative_at_one : 
  deriv f 1 = Real.sin 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l3242_324256


namespace NUMINAMATH_CALUDE_jellybean_count_l3242_324271

/-- The number of jellybeans in a dozen -/
def dozen : ℕ := 12

/-- Caleb's number of jellybeans -/
def caleb_jellybeans : ℕ := 3 * dozen

/-- Sophie's number of jellybeans -/
def sophie_jellybeans : ℕ := caleb_jellybeans / 2

/-- The total number of jellybeans Caleb and Sophie have -/
def total_jellybeans : ℕ := caleb_jellybeans + sophie_jellybeans

theorem jellybean_count : total_jellybeans = 54 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l3242_324271


namespace NUMINAMATH_CALUDE_solution_difference_l3242_324264

theorem solution_difference (p q : ℝ) : 
  ((6 * p - 18) / (p^2 + 3*p - 18) = p + 3) →
  ((6 * q - 18) / (q^2 + 3*q - 18) = q + 3) →
  p ≠ q →
  p > q →
  p - q = 9 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l3242_324264


namespace NUMINAMATH_CALUDE_prime_combinations_theorem_l3242_324269

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def all_combinations_prime (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → is_prime (10^k * 7 + (10^n - 1) / 9 - 10^k)

theorem prime_combinations_theorem :
  ∀ n : ℕ, (all_combinations_prime n ↔ n = 1 ∨ n = 2) :=
sorry

end NUMINAMATH_CALUDE_prime_combinations_theorem_l3242_324269


namespace NUMINAMATH_CALUDE_root_sum_square_l3242_324282

theorem root_sum_square (a b : ℝ) : 
  a ≠ b →
  (a^2 + 2*a - 2022 = 0) → 
  (b^2 + 2*b - 2022 = 0) → 
  a^2 + 4*a + 2*b = 2018 := by
sorry

end NUMINAMATH_CALUDE_root_sum_square_l3242_324282


namespace NUMINAMATH_CALUDE_room_number_unit_digit_l3242_324289

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def divisible_by_seven (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

def contains_digit_nine (n : ℕ) : Prop := ∃ a b : ℕ, n = 10 * a + 9 * b ∧ b ≤ 1

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def satisfies_three_conditions (n : ℕ) : Prop :=
  (is_prime n ∧ is_even n ∧ divisible_by_seven n) ∨
  (is_prime n ∧ is_even n ∧ contains_digit_nine n) ∨
  (is_prime n ∧ divisible_by_seven n ∧ contains_digit_nine n) ∨
  (is_even n ∧ divisible_by_seven n ∧ contains_digit_nine n)

theorem room_number_unit_digit :
  ∃ n : ℕ, is_two_digit n ∧ satisfies_three_conditions n ∧ n % 10 = 8 :=
sorry

end NUMINAMATH_CALUDE_room_number_unit_digit_l3242_324289


namespace NUMINAMATH_CALUDE_spade_evaluation_l3242_324221

def spade (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

theorem spade_evaluation : spade 2 (spade 3 4) = 384 := by
  sorry

end NUMINAMATH_CALUDE_spade_evaluation_l3242_324221


namespace NUMINAMATH_CALUDE_log_power_equality_l3242_324227

theorem log_power_equality (a N m : ℝ) (ha : a > 0) (hN : N > 0) (hm : m ≠ 0) :
  Real.log N^m / Real.log (a^m) = Real.log N / Real.log a := by
  sorry

end NUMINAMATH_CALUDE_log_power_equality_l3242_324227


namespace NUMINAMATH_CALUDE_part_one_part_two_l3242_324209

-- Define the line l: y = k(x-n)
def line (k n x : ℝ) : ℝ := k * (x - n)

-- Define the parabola y^2 = 4x
def parabola (x : ℝ) : ℝ := 4 * x

-- Define the intersection points
structure Point where
  x : ℝ
  y : ℝ

-- Theorem for part (I)
theorem part_one (k n : ℝ) (A B : Point) (h1 : A.y = line k n A.x) (h2 : B.y = line k n B.x)
  (h3 : A.y^2 = parabola A.x) (h4 : B.y^2 = parabola B.x) (h5 : A.x * B.x ≠ 0)
  (h6 : line k n 1 = 0) : A.x * B.x = 1 := by sorry

-- Theorem for part (II)
theorem part_two (k n : ℝ) (A B : Point) (h1 : A.y = line k n A.x) (h2 : B.y = line k n B.x)
  (h3 : A.y^2 = parabola A.x) (h4 : B.y^2 = parabola B.x) (h5 : A.x * B.x ≠ 0)
  (h6 : A.x * B.x + A.y * B.y = 0) : n = 4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3242_324209


namespace NUMINAMATH_CALUDE_z_equals_3s_l3242_324255

theorem z_equals_3s (z s : ℝ) (hz : z ≠ 0) (heq : z = Real.sqrt (6 * z * s - 9 * s^2)) : z = 3 * s := by
  sorry

end NUMINAMATH_CALUDE_z_equals_3s_l3242_324255


namespace NUMINAMATH_CALUDE_ali_total_money_l3242_324263

def five_dollar_bills : ℕ := 7
def ten_dollar_bills : ℕ := 1
def five_dollar_value : ℕ := 5
def ten_dollar_value : ℕ := 10

theorem ali_total_money :
  five_dollar_bills * five_dollar_value + ten_dollar_bills * ten_dollar_value = 45 := by
sorry

end NUMINAMATH_CALUDE_ali_total_money_l3242_324263


namespace NUMINAMATH_CALUDE_subtract_3a_from_expression_l3242_324245

variable (a : ℝ)

theorem subtract_3a_from_expression : (9 * a^2 - 3 * a + 8) - 3 * a = 9 * a^2 + 8 := by
  sorry

end NUMINAMATH_CALUDE_subtract_3a_from_expression_l3242_324245


namespace NUMINAMATH_CALUDE_expression_factorization_l3242_324252

theorem expression_factorization (x : ℝ) :
  (12 * x^3 + 45 * x - 3) - (-3 * x^3 + 5 * x - 2) = 5 * x * (3 * x^2 + 8) - 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l3242_324252


namespace NUMINAMATH_CALUDE_star_two_three_l3242_324260

/-- The star operation defined as a * b = a * b^3 - 2 * b + 2 -/
def star (a b : ℝ) : ℝ := a * b^3 - 2 * b + 2

/-- Theorem: The value of 2 ★ 3 is 50 -/
theorem star_two_three : star 2 3 = 50 := by
  sorry

end NUMINAMATH_CALUDE_star_two_three_l3242_324260


namespace NUMINAMATH_CALUDE_storks_vs_birds_l3242_324297

theorem storks_vs_birds (initial_birds : ℕ) (additional_birds : ℕ) (storks : ℕ) : 
  initial_birds = 3 → additional_birds = 2 → storks = 6 → 
  storks - (initial_birds + additional_birds) = 1 := by
sorry

end NUMINAMATH_CALUDE_storks_vs_birds_l3242_324297


namespace NUMINAMATH_CALUDE_school_capacity_l3242_324226

theorem school_capacity (total_classrooms : ℕ) 
  (desks_type1 desks_type2 desks_type3 : ℕ) : 
  total_classrooms = 30 →
  desks_type1 = 40 →
  desks_type2 = 35 →
  desks_type3 = 28 →
  (total_classrooms / 5 * desks_type1 + 
   total_classrooms / 3 * desks_type2 + 
   (total_classrooms - total_classrooms / 5 - total_classrooms / 3) * desks_type3) = 982 :=
by
  sorry

#check school_capacity

end NUMINAMATH_CALUDE_school_capacity_l3242_324226


namespace NUMINAMATH_CALUDE_smallest_n_with_three_triples_l3242_324259

/-- Function that counts the number of distinct ordered triples (a, b, c) of positive integers
    such that a^2 + b^2 + c^2 = n -/
def g (n : ℕ) : ℕ := 
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0 ∧ 
    t.1^2 + t.2.1^2 + t.2.2^2 = n) (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card

/-- 11 is the smallest positive integer n for which g(n) = 3 -/
theorem smallest_n_with_three_triples : 
  (∀ m : ℕ, m > 0 ∧ m < 11 → g m ≠ 3) ∧ g 11 = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_three_triples_l3242_324259


namespace NUMINAMATH_CALUDE_dragon_poker_partitions_l3242_324204

/-- The number of suits in the deck -/
def num_suits : ℕ := 4

/-- The target score to achieve -/
def target_score : ℕ := 2018

/-- The number of ways to partition the target score into exactly num_suits non-negative integers -/
def num_partitions : ℕ := (target_score + num_suits - 1).choose (num_suits - 1)

theorem dragon_poker_partitions :
  num_partitions = 1373734330 := by sorry

end NUMINAMATH_CALUDE_dragon_poker_partitions_l3242_324204


namespace NUMINAMATH_CALUDE_ellen_smoothie_total_cups_l3242_324216

/-- Represents the ingredients used in Ellen's smoothie recipe -/
structure SmoothieIngredients where
  strawberries : Float
  yogurt : Float
  orange_juice : Float
  honey : Float
  chia_seeds : Float
  spinach : Float

/-- Conversion factors for measurements -/
def ounce_to_cup : Float := 0.125
def tablespoon_to_cup : Float := 0.0625

/-- Ellen's smoothie recipe -/
def ellen_smoothie : SmoothieIngredients := {
  strawberries := 0.2,
  yogurt := 0.1,
  orange_juice := 0.2,
  honey := 1 * ounce_to_cup,
  chia_seeds := 2 * tablespoon_to_cup,
  spinach := 0.5
}

/-- Theorem stating the total cups of ingredients in Ellen's smoothie -/
theorem ellen_smoothie_total_cups : 
  ellen_smoothie.strawberries + 
  ellen_smoothie.yogurt + 
  ellen_smoothie.orange_juice + 
  ellen_smoothie.honey + 
  ellen_smoothie.chia_seeds + 
  ellen_smoothie.spinach = 1.25 := by sorry

end NUMINAMATH_CALUDE_ellen_smoothie_total_cups_l3242_324216


namespace NUMINAMATH_CALUDE_white_balls_count_l3242_324248

theorem white_balls_count (red : ℕ) (yellow : ℕ) (white : ℕ) 
  (h_red : red = 3)
  (h_yellow : yellow = 2)
  (h_prob : (yellow : ℚ) / (red + yellow + white) = 1/4) :
  white = 3 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l3242_324248


namespace NUMINAMATH_CALUDE_power_of_product_l3242_324230

theorem power_of_product (a b : ℝ) : (b^2 * a)^3 = a^3 * b^6 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l3242_324230


namespace NUMINAMATH_CALUDE_quadratic_intersects_x_axis_symmetric_roots_l3242_324232

/-- The quadratic function f(x) = x^2 - 2kx - 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 2*k*x - 1

theorem quadratic_intersects_x_axis (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 :=
sorry

theorem symmetric_roots :
  f 0 1 = 0 ∧ f 0 (-1) = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersects_x_axis_symmetric_roots_l3242_324232


namespace NUMINAMATH_CALUDE_simple_random_for_ten_basketballs_l3242_324275

/-- Enumeration of sampling methods --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | WithReplacement

/-- Definition of a sampling scenario --/
structure SamplingScenario where
  population_size : ℕ
  sample_size : ℕ
  for_quality_testing : Bool

/-- Function to determine the appropriate sampling method --/
def appropriate_sampling_method (scenario : SamplingScenario) : SamplingMethod :=
  sorry

/-- Theorem stating that Simple Random Sampling is appropriate for the given scenario --/
theorem simple_random_for_ten_basketballs :
  let scenario : SamplingScenario := {
    population_size := 10,
    sample_size := 1,
    for_quality_testing := true
  }
  appropriate_sampling_method scenario = SamplingMethod.SimpleRandom :=
sorry

end NUMINAMATH_CALUDE_simple_random_for_ten_basketballs_l3242_324275


namespace NUMINAMATH_CALUDE_crust_vs_bread_expenditure_l3242_324294

/-- Represents the percentage increase in expenditure when buying crust instead of bread -/
def expenditure_increase : ℝ := 36

/-- The ratio of crust weight to bread weight -/
def crust_weight_ratio : ℝ := 0.75

/-- The ratio of crust price to bread price -/
def crust_price_ratio : ℝ := 1.2

/-- The ratio of bread that is actually consumed -/
def bread_consumption_ratio : ℝ := 0.85

/-- The ratio of crust that is actually consumed -/
def crust_consumption_ratio : ℝ := 1

theorem crust_vs_bread_expenditure :
  expenditure_increase = 
    ((crust_consumption_ratio / crust_weight_ratio) / 
     bread_consumption_ratio * crust_price_ratio - 1) * 100 := by
  sorry

#eval expenditure_increase

end NUMINAMATH_CALUDE_crust_vs_bread_expenditure_l3242_324294


namespace NUMINAMATH_CALUDE_ship_passengers_ship_passengers_proof_l3242_324229

theorem ship_passengers : ℕ → Prop :=
  fun total_passengers =>
    (total_passengers : ℚ) = (1 / 12 + 1 / 4 + 1 / 9 + 1 / 6) * total_passengers + 42 →
    total_passengers = 108

-- Proof
theorem ship_passengers_proof : ship_passengers 108 := by
  sorry

end NUMINAMATH_CALUDE_ship_passengers_ship_passengers_proof_l3242_324229


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3242_324215

theorem arithmetic_calculation : 8 / 4 - 3 - 9 + 3 * 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3242_324215


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l3242_324208

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane 
  (α β : Plane) (m : Line)
  (h1 : subset m α)
  (h2 : parallel α β) :
  line_parallel m β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l3242_324208


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3242_324290

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Represents a line -/
structure Line where
  k : ℝ

/-- The area of a triangle formed by two points on an ellipse and a fixed point -/
def triangleArea (e : Ellipse) (l : Line) (A : Point) : ℝ := sorry

/-- The main theorem -/
theorem ellipse_intersection_theorem (e : Ellipse) (l : Line) (A : Point) :
  e.a = 2 ∧ e.b = Real.sqrt 2 ∧ 
  A.x = 2 ∧ A.y = 0 ∧
  triangleArea e l A = Real.sqrt 10 / 3 →
  l.k = 1 ∨ l.k = -1 := by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3242_324290


namespace NUMINAMATH_CALUDE_max_value_of_f_l3242_324242

/-- The function f represents the quadratic equation y = -3x^2 + 12x + 4 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 12 * x + 4

/-- The theorem states that the maximum value of f is 16 and occurs at x = 2 -/
theorem max_value_of_f :
  (∃ (x_max : ℝ), f x_max = 16 ∧ ∀ (x : ℝ), f x ≤ f x_max) ∧
  (f 2 = 16 ∧ ∀ (x : ℝ), f x ≤ 16) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3242_324242


namespace NUMINAMATH_CALUDE_apple_distribution_l3242_324224

theorem apple_distribution (n : ℕ) (k : ℕ) (min_apples : ℕ) : 
  n = 24 → k = 3 → min_apples = 2 → 
  (Nat.choose (n - k * min_apples + k - 1) (k - 1)) = 190 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l3242_324224


namespace NUMINAMATH_CALUDE_jellybean_probability_l3242_324270

/-- Represents the number of ways to choose k items from n items --/
def binomial (n k : ℕ) : ℕ := sorry

/-- The probability of an event given the number of favorable outcomes and total outcomes --/
def probability (favorable total : ℕ) : ℚ := sorry

theorem jellybean_probability :
  let total_jellybeans : ℕ := 15
  let green_jellybeans : ℕ := 6
  let purple_jellybeans : ℕ := 2
  let yellow_jellybeans : ℕ := 7
  let picked_jellybeans : ℕ := 4

  let total_outcomes : ℕ := binomial total_jellybeans picked_jellybeans
  let yellow_combinations : ℕ := binomial yellow_jellybeans 2
  let non_yellow_combinations : ℕ := binomial (green_jellybeans + purple_jellybeans) 2
  let favorable_outcomes : ℕ := yellow_combinations * non_yellow_combinations

  probability favorable_outcomes total_outcomes = 4 / 9 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_probability_l3242_324270


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3242_324205

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- Theorem 1
theorem problem_1 (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, g a x ∈ Set.Icc 0 4) ∧
  (∃ x ∈ Set.Icc 1 3, g a x = 0) ∧
  (∃ x ∈ Set.Icc 1 3, g a x = 4) →
  a = 1 := by sorry

-- Theorem 2
theorem problem_2 (k : ℝ) :
  (∀ x ≥ 1, g 1 (2^x) - k * 4^x ≥ 0) →
  k ≤ 1/4 := by sorry

-- Theorem 3
theorem problem_3 (k : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (g 1 (|2^x₁ - 1|) / |2^x₁ - 1| + k * (2 / |2^x₁ - 1|) - 3*k = 0) ∧
    (g 1 (|2^x₂ - 1|) / |2^x₂ - 1| + k * (2 / |2^x₂ - 1|) - 3*k = 0) ∧
    (g 1 (|2^x₃ - 1|) / |2^x₃ - 1| + k * (2 / |2^x₃ - 1|) - 3*k = 0)) →
  k > 0 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3242_324205


namespace NUMINAMATH_CALUDE_partition_natural_numbers_l3242_324277

theorem partition_natural_numbers : 
  ∃ (partition : ℕ → Fin 100), 
    (∀ i : Fin 100, ∃ n : ℕ, partition n = i) ∧ 
    (∀ a b c : ℕ, a + 99 * b = c → 
      partition a = partition c ∨ 
      partition a = partition b ∨ 
      partition b = partition c) :=
sorry

end NUMINAMATH_CALUDE_partition_natural_numbers_l3242_324277


namespace NUMINAMATH_CALUDE_smallest_multiple_of_4_and_14_l3242_324219

theorem smallest_multiple_of_4_and_14 : ∀ a : ℕ, a > 0 ∧ 4 ∣ a ∧ 14 ∣ a → a ≥ 28 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_4_and_14_l3242_324219


namespace NUMINAMATH_CALUDE_ball_distribution_probability_ratio_l3242_324284

theorem ball_distribution_probability_ratio :
  let total_balls : ℕ := 25
  let total_bins : ℕ := 6
  let p := (Nat.choose total_bins 2 * Nat.choose total_balls 3 * Nat.choose (total_balls - 3) 3 *
            Nat.choose (total_balls - 6) 4 * Nat.choose (total_balls - 10) 4 *
            Nat.choose (total_balls - 14) 4 * Nat.choose (total_balls - 18) 4) / 
           (Nat.factorial 4 * Nat.pow total_bins total_balls)
  let q := (Nat.choose total_bins 1 * Nat.choose total_balls 5 *
            Nat.choose (total_balls - 5) 4 * Nat.choose (total_balls - 9) 4 *
            Nat.choose (total_balls - 13) 4 * Nat.choose (total_balls - 17) 4 *
            Nat.choose (total_balls - 21) 4) / 
           Nat.pow total_bins total_balls
  p / q = 8 := by
sorry

end NUMINAMATH_CALUDE_ball_distribution_probability_ratio_l3242_324284


namespace NUMINAMATH_CALUDE_jean_initial_stuffies_l3242_324233

/-- Proves that Jean initially had 60 stuffies given the problem conditions -/
theorem jean_initial_stuffies :
  ∀ (initial : ℕ),
  (initial : ℚ) * (2/3) * (1/4) = 10 →
  initial = 60 := by
sorry

end NUMINAMATH_CALUDE_jean_initial_stuffies_l3242_324233


namespace NUMINAMATH_CALUDE_tan_sin_30_identity_l3242_324201

theorem tan_sin_30_identity : 
  (Real.tan (30 * π / 180))^2 - (Real.sin (30 * π / 180))^2 = 
  (4 / 3) * (Real.tan (30 * π / 180))^2 * (Real.sin (30 * π / 180))^2 := by
sorry

end NUMINAMATH_CALUDE_tan_sin_30_identity_l3242_324201


namespace NUMINAMATH_CALUDE_parametric_line_point_at_zero_l3242_324218

/-- A parametric line in 2D space -/
structure ParametricLine where
  /-- The position vector for a given parameter t -/
  pos : ℝ → (ℝ × ℝ)

/-- Theorem: Given a parametric line with specific points, find the point at t = 0 -/
theorem parametric_line_point_at_zero
  (line : ParametricLine)
  (h1 : line.pos 1 = (2, 3))
  (h4 : line.pos 4 = (6, -12)) :
  line.pos 0 = (2/3, 8) := by
  sorry

end NUMINAMATH_CALUDE_parametric_line_point_at_zero_l3242_324218


namespace NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l3242_324243

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l3242_324243


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3242_324225

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  b * c / a + a * c / b + a * b / c ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3242_324225


namespace NUMINAMATH_CALUDE_B_is_top_leftmost_l3242_324262

/-- Represents a rectangle with four sides labeled w, x, y, z --/
structure Rectangle where
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

/-- The set of all rectangles in the arrangement --/
def rectangles : Finset Rectangle := sorry

/-- Rectangle A --/
def A : Rectangle := ⟨5, 2, 8, 11⟩

/-- Rectangle B --/
def B : Rectangle := ⟨2, 1, 4, 7⟩

/-- Rectangle C --/
def C : Rectangle := ⟨4, 9, 6, 3⟩

/-- Rectangle D --/
def D : Rectangle := ⟨8, 6, 5, 9⟩

/-- Rectangle E --/
def E : Rectangle := ⟨10, 3, 9, 1⟩

/-- Rectangle F --/
def F : Rectangle := ⟨11, 4, 10, 2⟩

/-- Predicate to check if a rectangle is in the leftmost position --/
def isLeftmost (r : Rectangle) : Prop :=
  ∀ s ∈ rectangles, r.w ≤ s.w

/-- Predicate to check if a rectangle is in the top row --/
def isTopRow (r : Rectangle) : Prop := sorry

/-- The main theorem stating that B is the top leftmost rectangle --/
theorem B_is_top_leftmost : isLeftmost B ∧ isTopRow B := by sorry

end NUMINAMATH_CALUDE_B_is_top_leftmost_l3242_324262


namespace NUMINAMATH_CALUDE_elevator_max_velocity_l3242_324223

/-- Represents the state of the elevator at a given time -/
structure ElevatorState where
  time : ℝ
  velocity : ℝ

/-- The elevator's motion profile -/
def elevatorMotion : ℝ → ElevatorState := sorry

/-- The acceleration period of the elevator -/
def accelerationPeriod : Set ℝ := {t | 2 ≤ t ∧ t ≤ 4}

/-- The deceleration period of the elevator -/
def decelerationPeriod : Set ℝ := {t | 22 ≤ t ∧ t ≤ 24}

/-- The constant speed period of the elevator -/
def constantSpeedPeriod : Set ℝ := {t | 4 < t ∧ t < 22}

/-- The maximum downward velocity of the elevator -/
def maxDownwardVelocity : ℝ := sorry

theorem elevator_max_velocity :
  ∀ t ∈ constantSpeedPeriod,
    (elevatorMotion t).velocity = maxDownwardVelocity ∧
    ∀ s, (elevatorMotion s).velocity ≤ maxDownwardVelocity := by
  sorry

#check elevator_max_velocity

end NUMINAMATH_CALUDE_elevator_max_velocity_l3242_324223


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3242_324217

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  X^4 + 1 = (X^2 - 4*X + 6) * q + r ∧
  r.degree < (X^2 - 4*X + 6).degree ∧
  r = 16*X - 59 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3242_324217


namespace NUMINAMATH_CALUDE_stone_game_ratio_bound_l3242_324261

/-- The stone game process -/
structure StoneGame where
  n : ℕ
  s : ℕ
  t : ℕ
  board : Multiset ℕ

/-- The rules of the stone game -/
def stone_game_step (game : StoneGame) (a b : ℕ) : StoneGame :=
  { n := game.n
  , s := game.s + 1
  , t := game.t + Nat.gcd a b
  , board := game.board - {a, b} + {1, a + b}
  }

/-- The theorem to prove -/
theorem stone_game_ratio_bound (game : StoneGame) (h_n : game.n ≥ 3) 
    (h_init : game.board = Multiset.replicate game.n 1) 
    (h_s_pos : game.s > 0) : 
    1 ≤ (game.t : ℚ) / game.s ∧ (game.t : ℚ) / game.s < game.n - 1 := by
  sorry

end NUMINAMATH_CALUDE_stone_game_ratio_bound_l3242_324261


namespace NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l3242_324273

/-- Calculates the cost per quart of ratatouille given ingredient quantities and prices -/
theorem ratatouille_cost_per_quart :
  let eggplant_oz : Real := 88
  let eggplant_price : Real := 0.22
  let zucchini_oz : Real := 60.8
  let zucchini_price : Real := 0.15
  let tomato_oz : Real := 73.6
  let tomato_price : Real := 0.25
  let onion_oz : Real := 43.2
  let onion_price : Real := 0.07
  let basil_oz : Real := 16
  let basil_price : Real := 2.70 / 4
  let bell_pepper_oz : Real := 12
  let bell_pepper_price : Real := 0.20
  let total_yield_quarts : Real := 4.5
  let total_cost : Real := 
    eggplant_oz * eggplant_price +
    zucchini_oz * zucchini_price +
    tomato_oz * tomato_price +
    onion_oz * onion_price +
    basil_oz * basil_price +
    bell_pepper_oz * bell_pepper_price
  let cost_per_quart : Real := total_cost / total_yield_quarts
  cost_per_quart = 14.02 := by
  sorry

end NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l3242_324273


namespace NUMINAMATH_CALUDE_number_of_correct_statements_l3242_324235

-- Define the properties
def is_rational (m : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ m = a / b
def is_real (m : ℝ) : Prop := True

def tan_equal (A B : ℝ) : Prop := Real.tan A = Real.tan B
def angle_equal (A B : ℝ) : Prop := A = B

def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 3 = 0
def x_equals_3 (x : ℝ) : Prop := x = 3

-- Define the statements
def statement1 : Prop := 
  (∀ m : ℝ, is_rational m → is_real m) ∧ 
  ¬(∀ m : ℝ, is_real m → is_rational m)

def statement2 : Prop := 
  (∀ A B : ℝ, tan_equal A B → angle_equal A B) ∧ 
  ¬(∀ A B : ℝ, angle_equal A B → tan_equal A B)

def statement3 : Prop := 
  (∀ x : ℝ, x_equals_3 x → quadratic_equation x) ∧ 
  ¬(∀ x : ℝ, quadratic_equation x → x_equals_3 x)

-- Theorem to prove
theorem number_of_correct_statements : 
  (statement1 ∧ ¬statement2 ∧ statement3) → 
  (Nat.card {s | s = statement1 ∨ s = statement2 ∨ s = statement3 ∧ s} = 2) :=
sorry

end NUMINAMATH_CALUDE_number_of_correct_statements_l3242_324235


namespace NUMINAMATH_CALUDE_sandwich_problem_l3242_324231

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 349/100

/-- The cost of a soda in dollars -/
def soda_cost : ℚ := 87/100

/-- The number of sodas bought -/
def num_sodas : ℕ := 4

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := 1046/100

/-- The number of sandwiches bought -/
def num_sandwiches : ℕ := 2

theorem sandwich_problem :
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = total_cost :=
sorry

end NUMINAMATH_CALUDE_sandwich_problem_l3242_324231


namespace NUMINAMATH_CALUDE_parallel_transitivity_l3242_324265

-- Define a type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define a relation for parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitivity (a b c : Line) :
  parallel a c → parallel b c → parallel a b := by sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l3242_324265


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3242_324279

/-- A geometric sequence of positive real numbers. -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n > 0

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : GeometricSequence a q)
  (h_prod : a 2 * a 6 = 16)
  (h_sum : a 4 + a 8 = 8) :
  q = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3242_324279


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3242_324292

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + m > 0) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3242_324292


namespace NUMINAMATH_CALUDE_composite_product_equals_twelve_over_pi_squared_l3242_324283

-- Define the sequence of composite numbers
def composite : ℕ → ℕ
  | 0 => 4  -- First composite number
  | n + 1 => sorry  -- Definition of subsequent composite numbers

-- Define the infinite product
def infinite_product : ℝ := sorry

-- Define the infinite sum of reciprocal squares
def reciprocal_squares_sum : ℝ := sorry

-- Theorem statement
theorem composite_product_equals_twelve_over_pi_squared :
  (reciprocal_squares_sum = Real.pi^2 / 6) →
  infinite_product = 12 / Real.pi^2 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_equals_twelve_over_pi_squared_l3242_324283


namespace NUMINAMATH_CALUDE_triangle_properties_l3242_324238

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  let m : ℝ × ℝ := (Real.cos B, Real.cos C)
  let n : ℝ × ℝ := (2*a + c, b)
  (m.1 * n.1 + m.2 * n.2 = 0) →  -- m ⟂ n
  (b = Real.sqrt 13) →
  (a + c = 4) →
  (B = 2 * Real.pi / 3) ∧
  (Real.sqrt 3 / 2 < Real.sin (2*A) + Real.sin (2*C)) ∧
  (Real.sin (2*A) + Real.sin (2*C) ≤ Real.sqrt 3) ∧
  (1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3242_324238


namespace NUMINAMATH_CALUDE_irreducible_fraction_l3242_324298

theorem irreducible_fraction (a b c d : ℤ) (h : a * d - b * c = 1) :
  ¬∃ (m : ℤ), m > 1 ∧ m ∣ (a^2 + b^2) ∧ m ∣ (a * c + b * d) := by
  sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l3242_324298


namespace NUMINAMATH_CALUDE_complex_product_equals_negative_25i_l3242_324200

theorem complex_product_equals_negative_25i :
  let Q : ℂ := 3 + 4*Complex.I
  let E : ℂ := -Complex.I
  let D : ℂ := 3 - 4*Complex.I
  Q * E * D = -25 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_negative_25i_l3242_324200


namespace NUMINAMATH_CALUDE_sheila_weekly_earnings_l3242_324268

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hoursMWF : ℕ  -- Hours worked on Monday, Wednesday, Friday
  hoursTT : ℕ   -- Hours worked on Tuesday, Thursday
  daysLong : ℕ  -- Number of days working long hours (MWF)
  daysShort : ℕ -- Number of days working short hours (TT)
  hourlyRate : ℕ -- Hourly rate in dollars

/-- Calculates weekly earnings based on work schedule --/
def weeklyEarnings (schedule : WorkSchedule) : ℕ :=
  (schedule.hoursMWF * schedule.daysLong + schedule.hoursTT * schedule.daysShort) * schedule.hourlyRate

/-- Theorem stating Sheila's weekly earnings --/
theorem sheila_weekly_earnings :
  let schedule : WorkSchedule := {
    hoursMWF := 8,
    hoursTT := 6,
    daysLong := 3,
    daysShort := 2,
    hourlyRate := 11
  }
  weeklyEarnings schedule = 396 := by sorry

end NUMINAMATH_CALUDE_sheila_weekly_earnings_l3242_324268


namespace NUMINAMATH_CALUDE_ferry_tourists_sum_l3242_324236

/-- The number of trips made by the ferry -/
def num_trips : ℕ := 7

/-- The initial number of tourists -/
def initial_tourists : ℕ := 100

/-- The decrease in tourists per trip -/
def tourist_decrease : ℕ := 2

/-- The sum of tourists over all trips -/
def total_tourists : ℕ := 658

/-- Theorem stating that the sum of the arithmetic sequence
    representing the number of tourists per trip equals the total number of tourists -/
theorem ferry_tourists_sum :
  (num_trips / 2 : ℚ) * (2 * initial_tourists - (num_trips - 1) * tourist_decrease) = total_tourists := by
  sorry

end NUMINAMATH_CALUDE_ferry_tourists_sum_l3242_324236


namespace NUMINAMATH_CALUDE_intersection_coordinate_sum_l3242_324237

/-- Given a triangle ABC with A(0,6), B(0,0), C(8,0), 
    D is the midpoint of AB, 
    E is on BC such that BE is one-third of BC,
    F is the intersection of AE and CD.
    Prove that the sum of x and y coordinates of F is 56/11. -/
theorem intersection_coordinate_sum (A B C D E F : ℝ × ℝ) : 
  A = (0, 6) →
  B = (0, 0) →
  C = (8, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E.1 = B.1 + (C.1 - B.1) / 3 →
  E.2 = B.2 + (C.2 - B.2) / 3 →
  (F.2 - A.2) / (F.1 - A.1) = (E.2 - A.2) / (E.1 - A.1) →
  (F.2 - C.2) / (F.1 - C.1) = (D.2 - C.2) / (D.1 - C.1) →
  F.1 + F.2 = 56 / 11 := by
  sorry


end NUMINAMATH_CALUDE_intersection_coordinate_sum_l3242_324237


namespace NUMINAMATH_CALUDE_pole_not_perpendicular_l3242_324213

theorem pole_not_perpendicular (h : Real) (d : Real) (c : Real) 
  (h_val : h = 1.4)
  (d_val : d = 2)
  (c_val : c = 2.5) : 
  h^2 + d^2 ≠ c^2 := by
  sorry

end NUMINAMATH_CALUDE_pole_not_perpendicular_l3242_324213


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l3242_324211

/-- The area of a triangle with side lengths 15, 36, and 39 is 270 -/
theorem triangle_area : ℝ → Prop :=
  fun a => ∀ s₁ s₂ s₃ : ℝ,
    s₁ = 15 ∧ s₂ = 36 ∧ s₃ = 39 →
    (∃ A : ℝ, A = a ∧ A = 270)

/-- Proof of the theorem -/
theorem triangle_area_proof : triangle_area 270 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l3242_324211


namespace NUMINAMATH_CALUDE_binomial_distribution_parameters_l3242_324206

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The mean of a binomial distribution -/
def mean (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_parameters :
  ∃ X : BinomialDistribution, mean X = 15 ∧ variance X = 12 ∧ X.n = 60 ∧ X.p = 0.25 := by sorry

end NUMINAMATH_CALUDE_binomial_distribution_parameters_l3242_324206


namespace NUMINAMATH_CALUDE_angle_1303_equiv_neg137_l3242_324246

-- Define a function to represent angles with the same terminal side
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ n : ℤ, β = α + n * 360

-- State the theorem
theorem angle_1303_equiv_neg137 :
  same_terminal_side 1303 (-137) :=
sorry

end NUMINAMATH_CALUDE_angle_1303_equiv_neg137_l3242_324246


namespace NUMINAMATH_CALUDE_square_perimeters_sum_l3242_324222

theorem square_perimeters_sum (x y : ℝ) 
  (h1 : x^2 + y^2 = 113) 
  (h2 : x^2 - y^2 = 47) 
  (h3 : x ≥ y) : 
  3 * (4 * x) + 4 * y = 48 * Real.sqrt 5 + 4 * Real.sqrt 33 := by
sorry

end NUMINAMATH_CALUDE_square_perimeters_sum_l3242_324222


namespace NUMINAMATH_CALUDE_undefined_expression_l3242_324281

theorem undefined_expression (x : ℝ) : 
  (x^2 - 22*x + 121 = 0) ↔ (x = 11) := by sorry

#check undefined_expression

end NUMINAMATH_CALUDE_undefined_expression_l3242_324281


namespace NUMINAMATH_CALUDE_expression_simplification_l3242_324207

theorem expression_simplification : (((2 + 3 + 6 + 7) / 3) + ((3 * 6 + 9) / 4)) = 12.75 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3242_324207


namespace NUMINAMATH_CALUDE_trivia_team_absence_l3242_324214

theorem trivia_team_absence (total_members : ℕ) (points_per_member : ℕ) (total_score : ℕ) 
  (h1 : total_members = 14)
  (h2 : points_per_member = 5)
  (h3 : total_score = 35) :
  total_members - (total_score / points_per_member) = 7 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_absence_l3242_324214


namespace NUMINAMATH_CALUDE_division_problem_l3242_324202

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 159 → quotient = 9 → remainder = 6 → 
  dividend = divisor * quotient + remainder → 
  divisor = 17 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3242_324202


namespace NUMINAMATH_CALUDE_magnet_area_theorem_l3242_324250

/-- Represents a rectangular magnet with length and width in centimeters. -/
structure Magnet where
  length : ℝ
  width : ℝ

/-- Calculates the area of a magnet in square centimeters. -/
def area (m : Magnet) : ℝ := m.length * m.width

/-- Calculates the circumference of two identical magnets attached horizontally. -/
def totalCircumference (m : Magnet) : ℝ := 2 * (2 * m.length + 2 * m.width)

/-- Theorem: Given two identical rectangular magnets with a total circumference of 70 cm
    and a total length of 15 cm when attached horizontally, the area of one magnet is 150 cm². -/
theorem magnet_area_theorem (m : Magnet) 
    (h1 : totalCircumference m = 70)
    (h2 : 2 * m.length = 15) : 
  area m = 150 := by
  sorry

end NUMINAMATH_CALUDE_magnet_area_theorem_l3242_324250


namespace NUMINAMATH_CALUDE_one_minus_repeating_third_equals_two_thirds_l3242_324286

-- Define the repeating decimal 0.3333...
def repeating_third : ℚ := 1/3

-- Theorem statement
theorem one_minus_repeating_third_equals_two_thirds :
  1 - repeating_third = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_third_equals_two_thirds_l3242_324286


namespace NUMINAMATH_CALUDE_lines_not_parallel_l3242_324203

-- Define the types for our objects
variable (Point Line Plane : Type)

-- Define the relationships
variable (contains : Plane → Line → Prop)
variable (not_contains : Plane → Line → Prop)
variable (on_line : Point → Line → Prop)
variable (in_plane : Point → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_not_parallel 
  (m n : Line) (α : Plane) (A : Point)
  (h1 : not_contains α m)
  (h2 : contains α n)
  (h3 : on_line A m)
  (h4 : in_plane A α) :
  ¬(parallel m n) :=
by sorry

end NUMINAMATH_CALUDE_lines_not_parallel_l3242_324203


namespace NUMINAMATH_CALUDE_collinear_points_ratio_l3242_324247

/-- Given four collinear points E, F, G, H in that order, with EF = 3, FG = 6, and EH = 20,
    prove that the ratio of EG to FH is 9/17. -/
theorem collinear_points_ratio (E F G H : ℝ) : 
  (F - E = 3) → (G - F = 6) → (H - E = 20) → 
  (E < F) → (F < G) → (G < H) →
  (G - E) / (H - F) = 9 / 17 := by
sorry

end NUMINAMATH_CALUDE_collinear_points_ratio_l3242_324247


namespace NUMINAMATH_CALUDE_root_shift_polynomial_l3242_324251

theorem root_shift_polynomial (a b c : ℂ) : 
  (a^3 - 5*a^2 + 7*a - 2 = 0) ∧ 
  (b^3 - 5*b^2 + 7*b - 2 = 0) ∧ 
  (c^3 - 5*c^2 + 7*c - 2 = 0) → 
  ((a - 3)^3 + 4*(a - 3)^2 + 4*(a - 3) + 1 = 0) ∧
  ((b - 3)^3 + 4*(b - 3)^2 + 4*(b - 3) + 1 = 0) ∧
  ((c - 3)^3 + 4*(c - 3)^2 + 4*(c - 3) + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_polynomial_l3242_324251


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3242_324249

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  let angle := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_2a_plus_b := Real.sqrt ((2*a.1 + b.1)^2 + (2*a.2 + b.2)^2)
  angle = π/4 ∧ magnitude_a = 1 ∧ magnitude_2a_plus_b = Real.sqrt 10 →
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_vector_magnitude_problem_l3242_324249


namespace NUMINAMATH_CALUDE_simplify_expression_l3242_324244

theorem simplify_expression (x : ℝ) : 2*x^3 - (7*x^2 - 9*x) - 2*(x^3 - 3*x^2 + 4*x) = -x^2 + x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3242_324244


namespace NUMINAMATH_CALUDE_sin_polar_complete_circle_l3242_324212

open Real

theorem sin_polar_complete_circle (t : ℝ) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ r : ℝ, r = sin θ) →
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → sin θ = sin (θ + t)) →
  t = 2 * π :=
sorry

end NUMINAMATH_CALUDE_sin_polar_complete_circle_l3242_324212


namespace NUMINAMATH_CALUDE_function_graphs_common_point_l3242_324299

/-- Given real numbers a, b, c, and d, if the graphs of y = 2a + 1/(x-b) and y = 2c + 1/(x-d) 
    have exactly one common point, then the graphs of y = 2b + 1/(x-a) and y = 2d + 1/(x-c) 
    also have exactly one common point. -/
theorem function_graphs_common_point (a b c d : ℝ) :
  (∃! x : ℝ, 2 * a + 1 / (x - b) = 2 * c + 1 / (x - d)) →
  (∃! x : ℝ, 2 * b + 1 / (x - a) = 2 * d + 1 / (x - c)) :=
by sorry

end NUMINAMATH_CALUDE_function_graphs_common_point_l3242_324299


namespace NUMINAMATH_CALUDE_complex_exponential_form_angle_l3242_324272

theorem complex_exponential_form_angle (z : ℂ) : 
  z = 2 - 2 * Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (4 * Real.pi / 3)) :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_form_angle_l3242_324272


namespace NUMINAMATH_CALUDE_fuel_station_problem_l3242_324210

/-- Represents the problem of calculating the number of mini-vans filled up at a fuel station. -/
theorem fuel_station_problem (service_cost : ℝ) (fuel_cost_per_liter : ℝ) (total_cost : ℝ) 
  (minivan_tank : ℝ) (truck_tank : ℝ) (num_trucks : ℕ) :
  service_cost = 2.20 →
  fuel_cost_per_liter = 0.70 →
  total_cost = 347.7 →
  minivan_tank = 65 →
  truck_tank = minivan_tank * 2.2 →
  num_trucks = 2 →
  ∃ (num_minivans : ℕ), 
    (num_minivans : ℝ) * (service_cost + fuel_cost_per_liter * minivan_tank) + 
    (num_trucks : ℝ) * (service_cost + fuel_cost_per_liter * truck_tank) = total_cost ∧
    num_minivans = 3 :=
by sorry

end NUMINAMATH_CALUDE_fuel_station_problem_l3242_324210


namespace NUMINAMATH_CALUDE_firefly_group_size_l3242_324293

theorem firefly_group_size (butterfly_group_size : ℕ) (min_butterflies : ℕ) 
  (h1 : butterfly_group_size = 44)
  (h2 : min_butterflies = 748) :
  ∃ (firefly_group_size : ℕ),
    firefly_group_size = 
      (((min_butterflies + butterfly_group_size - 1) / butterfly_group_size) * butterfly_group_size) :=
by
  sorry

end NUMINAMATH_CALUDE_firefly_group_size_l3242_324293


namespace NUMINAMATH_CALUDE_parabola_point_ordering_l3242_324296

/-- Theorem: For a parabola y = (x-2)² + k and two points on it, prove y₁ > y₂ > k -/
theorem parabola_point_ordering (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  y₁ = (x₁ - 2)^2 + k →
  y₂ = (x₂ - 2)^2 + k →
  x₂ > 2 →
  2 > x₁ →
  x₁ + x₂ < 4 →
  y₁ > y₂ ∧ y₂ > k :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_ordering_l3242_324296


namespace NUMINAMATH_CALUDE_floor_sum_of_squares_and_product_l3242_324266

theorem floor_sum_of_squares_and_product (p q r s : ℝ) : 
  0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s →
  p^2 + q^2 = 2500 →
  r^2 + s^2 = 2500 →
  p * q = 1152 →
  r * s = 1152 →
  ⌊p + q + r + s⌋ = 138 := by
sorry

end NUMINAMATH_CALUDE_floor_sum_of_squares_and_product_l3242_324266


namespace NUMINAMATH_CALUDE_at_least_three_prime_factors_l3242_324234

theorem at_least_three_prime_factors
  (p : Nat)
  (h_prime : Nat.Prime p)
  (h_div : p^2 ∣ 2^(p-1) - 1)
  (n : Nat) :
  ∃ (q₁ q₂ q₃ : Nat),
    Nat.Prime q₁ ∧ Nat.Prime q₂ ∧ Nat.Prime q₃ ∧
    q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₂ ≠ q₃ ∧
    (q₁ ∣ (p-1) * (Nat.factorial p + 2^n)) ∧
    (q₂ ∣ (p-1) * (Nat.factorial p + 2^n)) ∧
    (q₃ ∣ (p-1) * (Nat.factorial p + 2^n)) :=
by sorry

end NUMINAMATH_CALUDE_at_least_three_prime_factors_l3242_324234


namespace NUMINAMATH_CALUDE_star_three_five_l3242_324228

-- Define the star operation
def star (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

-- Theorem statement
theorem star_three_five : star 3 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_star_three_five_l3242_324228


namespace NUMINAMATH_CALUDE_alice_wins_iff_m_even_or_n_odd_l3242_324240

/-- The game state on an n×n grid where players can color an m×m subgrid or a single cell -/
structure GameState (m n : ℕ+) where
  grid : Fin n → Fin n → Bool

/-- The result of the game -/
inductive GameResult
  | AliceWins
  | BobWins

/-- An optimal strategy for the game -/
def OptimalStrategy (m n : ℕ+) : GameState m n → GameResult := sorry

/-- The main theorem: Alice wins with optimal play if and only if m is even or n is odd -/
theorem alice_wins_iff_m_even_or_n_odd (m n : ℕ+) :
  (∀ initial : GameState m n, OptimalStrategy m n initial = GameResult.AliceWins) ↔ 
  (Even m.val ∨ Odd n.val) := by sorry

end NUMINAMATH_CALUDE_alice_wins_iff_m_even_or_n_odd_l3242_324240


namespace NUMINAMATH_CALUDE_courtyard_width_is_16_meters_l3242_324258

def courtyard_length : ℝ := 25
def brick_length : ℝ := 0.2
def brick_width : ℝ := 0.1
def total_bricks : ℕ := 20000

theorem courtyard_width_is_16_meters :
  let brick_area : ℝ := brick_length * brick_width
  let total_area : ℝ := (total_bricks : ℝ) * brick_area
  let courtyard_width : ℝ := total_area / courtyard_length
  courtyard_width = 16 := by sorry

end NUMINAMATH_CALUDE_courtyard_width_is_16_meters_l3242_324258


namespace NUMINAMATH_CALUDE_hexagonal_prism_vertices_l3242_324285

/-- A hexagonal prism is a three-dimensional geometric shape with hexagonal bases -/
structure HexagonalPrism :=
  (base : Nat)
  (height : Nat)

/-- The number of vertices in a hexagonal prism -/
def num_vertices (prism : HexagonalPrism) : Nat :=
  12

/-- Theorem: A hexagonal prism has 12 vertices -/
theorem hexagonal_prism_vertices (prism : HexagonalPrism) :
  num_vertices prism = 12 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_prism_vertices_l3242_324285


namespace NUMINAMATH_CALUDE_train_length_calculation_l3242_324288

/-- The length of a train given its speed and time to cross a point -/
theorem train_length_calculation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 360 →
  time_s = 0.9999200063994881 →
  ∃ (length_m : ℝ), abs (length_m - 99.992) < 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3242_324288


namespace NUMINAMATH_CALUDE_binomial_max_remainder_l3242_324287

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

theorem binomial_max_remainder (k : ℕ) (h1 : 30 ≤ k) (h2 : k ≤ 70) :
  ∃ M : ℕ, 
    (∀ j : ℕ, 30 ≤ j → j ≤ 70 → 
      (binomial 100 j) / Nat.gcd (binomial 100 j) (binomial 100 (j+3)) ≤ M) ∧
    M % 1000 = 664 := by
  sorry

end NUMINAMATH_CALUDE_binomial_max_remainder_l3242_324287


namespace NUMINAMATH_CALUDE_equation_system_solution_l3242_324253

theorem equation_system_solution :
  ∃ (x₁ x₂ : ℝ),
    (∀ x y : ℝ, 5 * y^2 + 3 * y + 2 = 2 * (10 * x^2 + 3 * y + 3) ∧ y = 3 * x + 1 →
      x = x₁ ∨ x = x₂) ∧
    x₁ = (-21 + Real.sqrt 641) / 50 ∧
    x₂ = (-21 - Real.sqrt 641) / 50 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l3242_324253
