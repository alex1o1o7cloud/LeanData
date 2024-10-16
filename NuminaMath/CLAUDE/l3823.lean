import Mathlib

namespace NUMINAMATH_CALUDE_cost_for_haleighs_pets_l3823_382311

/-- Calculates the cost of leggings for Haleigh's pets -/
def cost_of_leggings (dogs cats spiders parrots chickens octopuses : ℕ)
  (dog_legs cat_legs spider_legs parrot_legs chicken_legs octopus_legs : ℕ)
  (bulk_price : ℚ) (bulk_quantity : ℕ) (regular_price : ℚ) : ℚ :=
  let total_legs := dogs * dog_legs + cats * cat_legs + spiders * spider_legs +
                    parrots * parrot_legs + chickens * chicken_legs + octopuses * octopus_legs
  let total_pairs := total_legs / 2
  let bulk_sets := total_pairs / bulk_quantity
  let remaining_pairs := total_pairs % bulk_quantity
  (bulk_sets * bulk_price) + (remaining_pairs * regular_price)

theorem cost_for_haleighs_pets :
  cost_of_leggings 4 3 2 1 5 3 4 4 8 2 2 8 18 12 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_cost_for_haleighs_pets_l3823_382311


namespace NUMINAMATH_CALUDE_polly_breakfast_time_l3823_382389

/-- The number of minutes Polly spends cooking breakfast every day -/
def breakfast_time : ℕ := sorry

/-- The number of minutes Polly spends cooking lunch every day -/
def lunch_time : ℕ := 5

/-- The number of days in a week Polly spends 10 minutes cooking dinner -/
def short_dinner_days : ℕ := 4

/-- The number of minutes Polly spends cooking dinner on short dinner days -/
def short_dinner_time : ℕ := 10

/-- The number of minutes Polly spends cooking dinner on long dinner days -/
def long_dinner_time : ℕ := 30

/-- The total number of minutes Polly spends cooking in a week -/
def total_cooking_time : ℕ := 305

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem polly_breakfast_time :
  breakfast_time * days_in_week +
  lunch_time * days_in_week +
  short_dinner_time * short_dinner_days +
  long_dinner_time * (days_in_week - short_dinner_days) =
  total_cooking_time ∧
  breakfast_time = 20 := by sorry

end NUMINAMATH_CALUDE_polly_breakfast_time_l3823_382389


namespace NUMINAMATH_CALUDE_list_number_fraction_l3823_382333

theorem list_number_fraction (S : ℝ) (n : ℝ) :
  n = 7 * (S / 50) →
  n / (S + n) = 7 / 57 := by
  sorry

end NUMINAMATH_CALUDE_list_number_fraction_l3823_382333


namespace NUMINAMATH_CALUDE_sector_area_for_unit_radian_l3823_382308

theorem sector_area_for_unit_radian (arc_length : Real) (h : arc_length = 6) :
  let radius := arc_length  -- From definition of radian: 1 = arc_length / radius
  let sector_area := (1 / 2) * radius * arc_length
  sector_area = 18 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_for_unit_radian_l3823_382308


namespace NUMINAMATH_CALUDE_first_term_of_special_arithmetic_sequence_l3823_382369

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of the first 60 terms is 500
  sum_first_60 : (60 : ℚ) / 2 * (2 * a + 59 * d) = 500
  -- Sum of the next 60 terms (61 to 120) is 2900
  sum_next_60 : (60 : ℚ) / 2 * (2 * (a + 60 * d) + 59 * d) = 2900

/-- The first term of the arithmetic sequence with given properties is -34/3 -/
theorem first_term_of_special_arithmetic_sequence (seq : ArithmeticSequence) : seq.a = -34/3 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_special_arithmetic_sequence_l3823_382369


namespace NUMINAMATH_CALUDE_equation_solutions_l3823_382323

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁^2 + 2*x₁ = 0 ∧ x₂^2 + 2*x₂ = 0) ∧ x₁ = 0 ∧ x₂ = -2) ∧
  (∃ y₁ y₂ : ℝ, (2*y₁^2 - 2*y₁ = 1 ∧ 2*y₂^2 - 2*y₂ = 1) ∧ 
    y₁ = (1 + Real.sqrt 3) / 2 ∧ y₂ = (1 - Real.sqrt 3) / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3823_382323


namespace NUMINAMATH_CALUDE_quadratic_range_for_x_less_than_neg_two_l3823_382395

/-- Represents a quadratic function y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-value of a quadratic function at a given x -/
def QuadraticFunction.yValue (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_range_for_x_less_than_neg_two
  (f : QuadraticFunction)
  (h_a_pos : f.a > 0)
  (h_vertex : f.yValue (-1) = -6)
  (h_y_at_neg_two : f.yValue (-2) = -5)
  (x : ℝ)
  (h_x : x < -2) :
  f.yValue x > -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_for_x_less_than_neg_two_l3823_382395


namespace NUMINAMATH_CALUDE_sparrow_swallow_system_l3823_382330

/-- Represents the weight of a sparrow in taels -/
def sparrow_weight : ℝ := sorry

/-- Represents the weight of a swallow in taels -/
def swallow_weight : ℝ := sorry

/-- The total weight of five sparrows and six swallows is 16 taels -/
axiom total_weight : 5 * sparrow_weight + 6 * swallow_weight = 16

/-- Exchanging one sparrow with one swallow results in equal weights for both groups -/
axiom exchange_equal : 4 * sparrow_weight + swallow_weight = 5 * swallow_weight + sparrow_weight

/-- The system of equations representing the sparrow and swallow weight problem -/
theorem sparrow_swallow_system :
  (5 * sparrow_weight + 6 * swallow_weight = 16) ∧
  (4 * sparrow_weight + swallow_weight = 5 * swallow_weight + sparrow_weight) :=
sorry

end NUMINAMATH_CALUDE_sparrow_swallow_system_l3823_382330


namespace NUMINAMATH_CALUDE_merchant_markup_theorem_l3823_382329

/-- Proves the required markup percentage for a merchant to achieve a specific profit --/
theorem merchant_markup_theorem (list_price : ℝ) (h_list_price_pos : 0 < list_price) :
  let cost_price := 0.7 * list_price
  let selling_price := list_price
  let marked_price := (5/4) * list_price
  (cost_price = 0.7 * selling_price) →
  (selling_price = 0.8 * marked_price) →
  (marked_price = 1.25 * list_price) :=
by
  sorry

#check merchant_markup_theorem

end NUMINAMATH_CALUDE_merchant_markup_theorem_l3823_382329


namespace NUMINAMATH_CALUDE_rotate_line_theorem_l3823_382393

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

/-- Rotates a line counterclockwise by π/2 around a given point -/
def rotateLine (l : Line) (px py : ℝ) : Line :=
  sorry

theorem rotate_line_theorem (l : Line) :
  l.a = 2 ∧ l.b = -1 ∧ l.c = -2 →
  let rotated := rotateLine l 0 (-2)
  rotated.a = 1 ∧ rotated.b = 2 ∧ rotated.c = 4 :=
sorry

end NUMINAMATH_CALUDE_rotate_line_theorem_l3823_382393


namespace NUMINAMATH_CALUDE_point_coordinates_l3823_382317

def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

def distance_to_x_axis (y : ℝ) : ℝ := |y|

def distance_to_y_axis (x : ℝ) : ℝ := |x|

theorem point_coordinates :
  ∀ x y : ℝ,
    third_quadrant x y →
    distance_to_x_axis y = 2 →
    distance_to_y_axis x = 5 →
    (x, y) = (-5, -2) :=
by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3823_382317


namespace NUMINAMATH_CALUDE_solution_set_a_1_no_a_for_all_reals_l3823_382342

-- Define the inequality function
def inequality (a : ℝ) (x : ℝ) : Prop :=
  |a*x - 1| + |a*x - a| ≥ 2

-- Part 1: Solution set when a = 1
theorem solution_set_a_1 :
  ∀ x : ℝ, inequality 1 x ↔ (x ≤ 0 ∨ x ≥ 2) :=
sorry

-- Part 2: No a > 0 makes the solution set ℝ
theorem no_a_for_all_reals :
  ¬ ∃ a : ℝ, a > 0 ∧ (∀ x : ℝ, inequality a x) :=
sorry

end NUMINAMATH_CALUDE_solution_set_a_1_no_a_for_all_reals_l3823_382342


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3823_382354

-- Define the universe set U
def U : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 10}

-- Define subset A
def A : Set ℝ := {x : ℝ | 2 < x ∧ x ≤ 4}

-- Define subset B
def B : Set ℝ := {x : ℝ | 3 < x ∧ x ≤ 5}

-- State the theorem
theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {x : ℝ | (0 ≤ x ∧ x ≤ 2) ∨ (5 < x ∧ x < 10)} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3823_382354


namespace NUMINAMATH_CALUDE_min_omega_for_symmetry_axis_l3823_382341

/-- The minimum positive value of ω for which f(x) = sin(ωx + π/6) has a symmetry axis at x = π/12 -/
theorem min_omega_for_symmetry_axis : ∃ (ω_min : ℝ), 
  (∀ (ω : ℝ), ω > 0 → (∃ (k : ℤ), ω = 12 * k + 4)) → 
  (∀ (ω : ℝ), ω > 0 → ω ≥ ω_min) → 
  ω_min = 4 := by
sorry

end NUMINAMATH_CALUDE_min_omega_for_symmetry_axis_l3823_382341


namespace NUMINAMATH_CALUDE_find_b_squared_l3823_382382

/-- A complex function satisfying certain properties -/
def f (a b : ℝ) (z : ℂ) : ℂ := (a + b * Complex.I) * z

/-- The main theorem -/
theorem find_b_squared (a b : ℝ) :
  (∀ z : ℂ, Complex.abs (f a b z - z) = Complex.abs (f a b z)) →
  a = 2 →
  Complex.abs (a + b * Complex.I) = 10 →
  b^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_find_b_squared_l3823_382382


namespace NUMINAMATH_CALUDE_steak_knife_cost_l3823_382373

/-- The number of steak knife sets -/
def num_sets : ℕ := 2

/-- The number of steak knives in each set -/
def knives_per_set : ℕ := 4

/-- The cost of each set in dollars -/
def cost_per_set : ℚ := 80

/-- The total number of steak knives -/
def total_knives : ℕ := num_sets * knives_per_set

/-- The total cost of all sets in dollars -/
def total_cost : ℚ := num_sets * cost_per_set

/-- The cost of each single steak knife in dollars -/
def cost_per_knife : ℚ := total_cost / total_knives

theorem steak_knife_cost : cost_per_knife = 20 := by
  sorry

end NUMINAMATH_CALUDE_steak_knife_cost_l3823_382373


namespace NUMINAMATH_CALUDE_number_problem_l3823_382368

/-- Given a number N, if N/p = 8, N/q = 18, and p - q = 0.2777777777777778, then N = 4 -/
theorem number_problem (N p q : ℝ) 
  (h1 : N / p = 8)
  (h2 : N / q = 18)
  (h3 : p - q = 0.2777777777777778) : 
  N = 4 := by sorry

end NUMINAMATH_CALUDE_number_problem_l3823_382368


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_l3823_382384

theorem circumscribed_circle_area (s : ℝ) (h : s = 12) :
  let triangle_side := s
  let triangle_height := (Real.sqrt 3 / 2) * triangle_side
  let circle_radius := (2 / 3) * triangle_height
  let circle_area := π * circle_radius ^ 2
  circle_area = 48 * π := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_area_l3823_382384


namespace NUMINAMATH_CALUDE_percentage_less_l3823_382345

theorem percentage_less (q y w z : ℝ) : 
  w = 0.6 * q →
  z = 0.54 * y →
  z = 1.5 * w →
  q = 0.6 * y :=
by sorry

end NUMINAMATH_CALUDE_percentage_less_l3823_382345


namespace NUMINAMATH_CALUDE_strawberry_weight_sum_l3823_382350

/-- The weight of Marco's strawberries in pounds -/
def marco_strawberries : ℕ := 3

/-- The weight of Marco's dad's strawberries in pounds -/
def dad_strawberries : ℕ := 17

/-- The total weight of Marco's and his dad's strawberries -/
def total_strawberries : ℕ := marco_strawberries + dad_strawberries

theorem strawberry_weight_sum :
  total_strawberries = 20 := by sorry

end NUMINAMATH_CALUDE_strawberry_weight_sum_l3823_382350


namespace NUMINAMATH_CALUDE_odd_function_and_monotone_increasing_l3823_382380

/-- An odd function f(x) = x^2 + mx -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x

/-- f is an odd function -/
def is_odd (m : ℝ) : Prop := ∀ x, f m (-x) = -(f m x)

/-- f is monotonically increasing on an interval -/
def is_monotone_increasing (m : ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f m x < f m y

theorem odd_function_and_monotone_increasing :
  ∃ m, is_odd m ∧ 
  ∃ a, 1 < a ∧ a ≤ 3 ∧ 
  is_monotone_increasing m (-1) (a - 2) ∧
  m = 2 := by sorry

end NUMINAMATH_CALUDE_odd_function_and_monotone_increasing_l3823_382380


namespace NUMINAMATH_CALUDE_total_pets_l3823_382391

def num_dogs : ℕ := 2
def num_cats : ℕ := 3
def num_fish : ℕ := 2 * (num_dogs + num_cats)

theorem total_pets : num_dogs + num_cats + num_fish = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_l3823_382391


namespace NUMINAMATH_CALUDE_only_vertical_angles_true_l3823_382340

-- Define the propositions
def proposition1 := "Non-intersecting lines are parallel lines"
def proposition2 := "Corresponding angles are equal"
def proposition3 := "If the squares of two real numbers are equal, then the two real numbers are also equal"
def proposition4 := "Vertical angles are equal"

-- Define a function to check if a proposition is true
def is_true (p : String) : Prop :=
  p = proposition4

-- Theorem statement
theorem only_vertical_angles_true :
  (is_true proposition1 = false) ∧
  (is_true proposition2 = false) ∧
  (is_true proposition3 = false) ∧
  (is_true proposition4 = true) :=
by
  sorry


end NUMINAMATH_CALUDE_only_vertical_angles_true_l3823_382340


namespace NUMINAMATH_CALUDE_intersection_points_slope_l3823_382388

/-- Theorem: The intersection points of the lines 2x - 3y = 8s + 6 and x + 2y = 3s - 1, 
    where s is a real parameter, all lie on a line with slope -2/25 -/
theorem intersection_points_slope (s : ℝ) :
  ∃ (x y : ℝ), 
    (2 * x - 3 * y = 8 * s + 6) ∧ 
    (x + 2 * y = 3 * s - 1) → 
    ∃ (m b : ℝ), m = -2/25 ∧ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_slope_l3823_382388


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_240_360_l3823_382316

theorem lcm_gcf_ratio_240_360 : 
  (Nat.lcm 240 360) / (Nat.gcd 240 360) = 6 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_240_360_l3823_382316


namespace NUMINAMATH_CALUDE_average_first_50_even_numbers_l3823_382378

theorem average_first_50_even_numbers : 
  let first_even : ℕ := 2
  let count : ℕ := 50
  let last_even : ℕ := first_even + 2 * (count - 1)
  (first_even + last_even) / 2 = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_average_first_50_even_numbers_l3823_382378


namespace NUMINAMATH_CALUDE_family_reunion_food_l3823_382302

/-- The total amount of food Peter buys for the family reunion -/
def total_food (chicken : ℝ) (hamburger_ratio : ℝ) (hotdog_difference : ℝ) (sides_ratio : ℝ) : ℝ :=
  let hamburger := chicken * hamburger_ratio
  let hotdog := hamburger + hotdog_difference
  let sides := hotdog * sides_ratio
  chicken + hamburger + hotdog + sides

/-- Theorem stating the total amount of food Peter will buy -/
theorem family_reunion_food :
  total_food 16 (1/2) 2 (1/2) = 39 := by
  sorry

end NUMINAMATH_CALUDE_family_reunion_food_l3823_382302


namespace NUMINAMATH_CALUDE_solution_set_nonempty_implies_a_range_l3823_382309

theorem solution_set_nonempty_implies_a_range 
  (h : ∃ x, |x - 3| + |x - a| < 4) : 
  -1 < a ∧ a < 7 := by
sorry

end NUMINAMATH_CALUDE_solution_set_nonempty_implies_a_range_l3823_382309


namespace NUMINAMATH_CALUDE_max_value_of_sum_max_value_achieved_l3823_382326

theorem max_value_of_sum (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 2 → x^2 + y^3 + z^4 ≤ 2 := by
  sorry

theorem max_value_achieved (x y z : ℝ) : 
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2 ∧ x^2 + y^3 + z^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_max_value_achieved_l3823_382326


namespace NUMINAMATH_CALUDE_johns_allowance_l3823_382381

/-- Calculates the amount of allowance John received given his initial amount, spending, and final amount -/
def calculate_allowance (initial : ℕ) (spent : ℕ) (final : ℕ) : ℕ :=
  final - (initial - spent)

/-- Proves that John's allowance was 26 dollars given the problem conditions -/
theorem johns_allowance :
  let initial := 5
  let spent := 2
  let final := 29
  calculate_allowance initial spent final = 26 := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_l3823_382381


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3823_382352

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0) → 
  -2 ≤ a ∧ a < 6/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3823_382352


namespace NUMINAMATH_CALUDE_triangle_altitude_l3823_382357

/-- Given a rectangle with length 3s and width s, and a triangle inside with one side
    along the diagonal and area half of the rectangle's area, the altitude of the
    triangle to the diagonal base is 3s√10/10 -/
theorem triangle_altitude (s : ℝ) (h : s > 0) :
  let l := 3 * s
  let w := s
  let diagonal := Real.sqrt (l^2 + w^2)
  let rectangle_area := l * w
  let triangle_area := rectangle_area / 2
  triangle_area = (1 / 2) * diagonal * (3 * s * Real.sqrt 10 / 10) :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_l3823_382357


namespace NUMINAMATH_CALUDE_sum_a_plus_d_l3823_382321

theorem sum_a_plus_d (a b c d : ℝ) 
  (eq1 : a + b = 16) 
  (eq2 : b + c = 9) 
  (eq3 : c + d = 3) : 
  a + d = 13 := by
sorry

end NUMINAMATH_CALUDE_sum_a_plus_d_l3823_382321


namespace NUMINAMATH_CALUDE_recipe_eggs_l3823_382313

theorem recipe_eggs (total_eggs : ℕ) (rotten_eggs : ℕ) (prob_all_rotten : ℝ) :
  total_eggs = 36 →
  rotten_eggs = 3 →
  prob_all_rotten = 0.0047619047619047615 →
  ∃ (n : ℕ), (rotten_eggs : ℝ) / (total_eggs : ℝ) ^ n = prob_all_rotten ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_recipe_eggs_l3823_382313


namespace NUMINAMATH_CALUDE_binomial_probability_l3823_382336

/-- A random variable following a binomial distribution with parameters n and p -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  X : ℝ → ℝ  -- The random variable

/-- The probability mass function for a binomial distribution -/
def pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem binomial_probability (X : BinomialDistribution 6 (1/2)) :
  pmf 6 (1/2) 3 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_l3823_382336


namespace NUMINAMATH_CALUDE_max_area_rectangular_garden_l3823_382314

/-- The maximum area of a rectangular garden with a perimeter of 168 feet and natural number side lengths --/
theorem max_area_rectangular_garden :
  ∃ (w h : ℕ), 
    w + h = 84 ∧ 
    (∀ (x y : ℕ), x + y = 84 → x * y ≤ w * h) ∧
    w * h = 1764 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangular_garden_l3823_382314


namespace NUMINAMATH_CALUDE_cube_root_of_negative_64_l3823_382359

theorem cube_root_of_negative_64 : ∃ b : ℝ, b^3 = -64 ∧ b = -4 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_64_l3823_382359


namespace NUMINAMATH_CALUDE_find_m_find_t_range_l3823_382325

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- Theorem 1: Find the value of m
theorem find_m :
  (∃ m : ℝ, ∀ x : ℝ, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-2) 2) →
  (∃ m : ℝ, m = 2 ∧ ∀ x : ℝ, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-2) 2) :=
sorry

-- Theorem 2: Find the range of t
theorem find_t_range (m : ℝ) (h : m = 2) :
  (∀ x t : ℝ, f m x ≥ -|x + 6| - t^2 + t) →
  (∀ t : ℝ, t ∈ Set.Iic (-2) ∪ Set.Ici 3) :=
sorry

end NUMINAMATH_CALUDE_find_m_find_t_range_l3823_382325


namespace NUMINAMATH_CALUDE_complete_square_equivalence_l3823_382361

theorem complete_square_equivalence : 
  ∀ x : ℝ, x^2 - 6*x - 7 = 0 ↔ (x - 3)^2 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_complete_square_equivalence_l3823_382361


namespace NUMINAMATH_CALUDE_walkway_time_when_stopped_l3823_382320

/-- The time it takes to walk a moving walkway when it's stopped -/
theorem walkway_time_when_stopped 
  (length : ℝ) 
  (time_with : ℝ) 
  (time_against : ℝ) 
  (h1 : length = 60) 
  (h2 : time_with = 30) 
  (h3 : time_against = 120) : 
  (2 * length) / (length / time_with + length / time_against) = 48 := by
  sorry

end NUMINAMATH_CALUDE_walkway_time_when_stopped_l3823_382320


namespace NUMINAMATH_CALUDE_compound_interest_rate_l3823_382363

theorem compound_interest_rate (P : ℝ) (r : ℝ) : 
  P > 0 →
  r > 0 →
  P * (1 + r)^2 - P = 492 →
  P * (1 + r)^2 = 5292 →
  r = 0.05 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l3823_382363


namespace NUMINAMATH_CALUDE_calculation_proof_system_of_equations_proof_l3823_382303

-- Part 1: Calculation proof
theorem calculation_proof :
  -2^2 - |2 - Real.sqrt 5| + (8 : ℝ)^(1/3) = -Real.sqrt 5 := by sorry

-- Part 2: System of equations proof
theorem system_of_equations_proof :
  ∃ (x y : ℝ), 2*x + y = 5 ∧ x - 3*y = 6 ∧ x = 3 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_calculation_proof_system_of_equations_proof_l3823_382303


namespace NUMINAMATH_CALUDE_correct_operation_l3823_382304

theorem correct_operation : ∀ x : ℝ, 
  (∃ y : ℝ, y ^ 2 = 4 ∧ y > 0) ∧ 
  (3 * x^3 + 2 * x^3 ≠ 5 * x^6) ∧ 
  ((x + 1)^2 ≠ x^2 + 1) ∧ 
  (x^8 / x^4 ≠ x^2) :=
by sorry

end NUMINAMATH_CALUDE_correct_operation_l3823_382304


namespace NUMINAMATH_CALUDE_factorization_equality_l3823_382315

theorem factorization_equality (x : ℝ) : 
  (x + 1) * (x + 2) * (x + 3) * (x + 4) - 120 = (x^2 + 5*x + 16) * (x + 6) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3823_382315


namespace NUMINAMATH_CALUDE_inequality_properties_l3823_382337

theorem inequality_properties (x y : ℝ) (h : x > y) : x^3 > y^3 ∧ Real.log x > Real.log y := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l3823_382337


namespace NUMINAMATH_CALUDE_mike_lawn_money_l3823_382339

/-- The amount of money Mike made mowing lawns -/
def lawn_money : ℝ := sorry

/-- The amount of money Mike made weed eating -/
def weed_eating_money : ℝ := 26

/-- The number of weeks the money lasted -/
def weeks : ℕ := 8

/-- The amount Mike spent per week -/
def weekly_spending : ℝ := 5

theorem mike_lawn_money :
  lawn_money = 14 :=
by
  have total_spent : ℝ := weekly_spending * weeks
  have total_money : ℝ := lawn_money + weed_eating_money
  have h1 : total_money = total_spent := by sorry
  sorry

end NUMINAMATH_CALUDE_mike_lawn_money_l3823_382339


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l3823_382379

/-- Circle A with equation x^2 + y^2 = 1 -/
def circle_A : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 1}

/-- Circle B with equation (x-3)^2 + (y+4)^2 = 10 -/
def circle_B : Set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + (p.2 + 4)^2 = 10}

/-- The point P satisfies the condition that its distances to the tangent points on circles A and B are equal -/
def point_P : Set (ℝ × ℝ) :=
  {p | ∃ d e : ℝ × ℝ, d ∈ circle_A ∧ e ∈ circle_B ∧ 
       (p.1 - d.1)^2 + (p.2 - d.2)^2 = (p.1 - e.1)^2 + (p.2 - e.2)^2}

/-- The minimum distance from point P to the origin is 8/5 -/
theorem min_distance_to_origin : 
  ∀ p ∈ point_P, (∀ q ∈ point_P, p.1^2 + p.2^2 ≤ q.1^2 + q.2^2) → 
  p.1^2 + p.2^2 = (8/5)^2 := by
sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l3823_382379


namespace NUMINAMATH_CALUDE_blue_fish_ratio_l3823_382335

/-- Given a fish tank with the following properties:
  - The total number of fish is 60.
  - Half of the blue fish have spots.
  - There are 10 blue, spotted fish.
  Prove that the ratio of blue fish to the total number of fish is 1/3. -/
theorem blue_fish_ratio (total_fish : ℕ) (blue_spotted_fish : ℕ) 
  (h1 : total_fish = 60)
  (h2 : blue_spotted_fish = 10)
  (h3 : blue_spotted_fish * 2 = blue_spotted_fish + (total_fish - blue_spotted_fish * 2)) :
  (blue_spotted_fish * 2 : ℚ) / total_fish = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_fish_ratio_l3823_382335


namespace NUMINAMATH_CALUDE_equation_solution_l3823_382348

theorem equation_solution : ∃ x : ℝ, (10 - 2 * x = 14) ∧ (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3823_382348


namespace NUMINAMATH_CALUDE_polynomial_has_three_distinct_integer_roots_l3823_382377

def polynomial (x : ℤ) : ℤ := x^5 + 3*x^4 - 4044118*x^3 - 12132362*x^2 - 12132363*x - 2011^2

theorem polynomial_has_three_distinct_integer_roots :
  ∃ (a b c : ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ x : ℤ, polynomial x = 0 ↔ x = a ∨ x = b ∨ x = c) :=
sorry

end NUMINAMATH_CALUDE_polynomial_has_three_distinct_integer_roots_l3823_382377


namespace NUMINAMATH_CALUDE_inequality_proof_l3823_382306

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  a * (b^2 + c^2) + b * (c^2 + a^2) ≥ 4 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3823_382306


namespace NUMINAMATH_CALUDE_equation_solution_l3823_382332

theorem equation_solution (x y : ℝ) : 
  x^2 + (1 - y)^2 + (x - y)^2 = 1/3 ↔ x = 1/3 ∧ y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3823_382332


namespace NUMINAMATH_CALUDE_election_combinations_l3823_382327

def number_of_students : ℕ := 6
def number_of_positions : ℕ := 3

theorem election_combinations :
  (number_of_students * (number_of_students - 1) * (number_of_students - 2) = 120) :=
by sorry

end NUMINAMATH_CALUDE_election_combinations_l3823_382327


namespace NUMINAMATH_CALUDE_daughters_age_l3823_382356

/-- Given a mother and daughter whose combined age is 60 years this year,
    and ten years ago the mother's age was seven times the daughter's age,
    prove that the daughter's age this year is 15 years. -/
theorem daughters_age (mother_age daughter_age : ℕ) : 
  mother_age + daughter_age = 60 →
  mother_age - 10 = 7 * (daughter_age - 10) →
  daughter_age = 15 := by
sorry

end NUMINAMATH_CALUDE_daughters_age_l3823_382356


namespace NUMINAMATH_CALUDE_fibonacci_inequality_l3823_382307

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_inequality (n : ℕ) (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (fibonacci n / fibonacci (n - 1) : ℚ) < (a / b : ℚ) →
  (a / b : ℚ) < (fibonacci (n + 1) / fibonacci n : ℚ) →
  b ≥ fibonacci (n + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_fibonacci_inequality_l3823_382307


namespace NUMINAMATH_CALUDE_complex_modulus_l3823_382349

theorem complex_modulus (z : ℂ) (h : (z - 2) * Complex.I = 1 + Complex.I) : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3823_382349


namespace NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_15_l3823_382387

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_first_year_after_2010_with_digit_sum_15 (year : ℕ) : Prop :=
  year > 2010 ∧ 
  digit_sum year = 15 ∧ 
  ∀ y, 2010 < y ∧ y < year → digit_sum y ≠ 15

theorem first_year_after_2010_with_digit_sum_15 :
  is_first_year_after_2010_with_digit_sum_15 2049 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_15_l3823_382387


namespace NUMINAMATH_CALUDE_battleship_theorem_l3823_382353

/-- Represents a rectangular grid -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a ship on the grid -/
structure Ship :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a set of connected cells -/
structure ConnectedCells :=
  (num_cells : ℕ)

/-- The minimum number of shots needed to guarantee hitting a ship on a grid -/
def min_shots_to_hit_ship (g : Grid) (s : Ship) : ℕ := sorry

/-- The minimum number of shots needed to guarantee hitting connected cells on a grid -/
def min_shots_to_hit_connected_cells (g : Grid) (c : ConnectedCells) : ℕ := sorry

/-- The main theorem for the Battleship problem -/
theorem battleship_theorem (g : Grid) (s : Ship) (c : ConnectedCells) :
  g.rows = 7 ∧ g.cols = 7 ∧ 
  ((s.length = 1 ∧ s.width = 4) ∨ (s.length = 4 ∧ s.width = 1)) ∧
  c.num_cells = 4 →
  (min_shots_to_hit_ship g s = 12) ∧
  (min_shots_to_hit_connected_cells g c = 20) := by sorry

end NUMINAMATH_CALUDE_battleship_theorem_l3823_382353


namespace NUMINAMATH_CALUDE_largest_multiple_of_11_below_negative_150_l3823_382398

theorem largest_multiple_of_11_below_negative_150 :
  ∀ n : ℤ, n * 11 < -150 → n * 11 ≤ -154 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_11_below_negative_150_l3823_382398


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l3823_382365

/-- If each edge of a cube increases by 20%, the surface area of the cube increases by 44%. -/
theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_area := 6 * L^2
  let new_edge := 1.2 * L
  let new_area := 6 * new_edge^2
  (new_area - original_area) / original_area = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l3823_382365


namespace NUMINAMATH_CALUDE_book_selection_l3823_382305

theorem book_selection (picture_books : ℕ) (sci_fi_books : ℕ) (total_selection : ℕ) : 
  picture_books = 4 → sci_fi_books = 2 → total_selection = 4 →
  (Nat.choose (picture_books + sci_fi_books) total_selection - 
   Nat.choose picture_books total_selection) = 14 :=
by sorry

end NUMINAMATH_CALUDE_book_selection_l3823_382305


namespace NUMINAMATH_CALUDE_cylinder_volume_height_relation_l3823_382301

theorem cylinder_volume_height_relation (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  let v := π * r^2 * h
  let r' := 2 * r
  let v' := 2 * v
  ∃ h', v' = π * r'^2 * h' ∧ h' = h / 4 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_height_relation_l3823_382301


namespace NUMINAMATH_CALUDE_woodburning_profit_l3823_382318

/-- Calculates the profit from selling woodburnings -/
theorem woodburning_profit
  (num_sold : ℕ)
  (price_per_item : ℝ)
  (cost : ℝ)
  (h1 : num_sold = 20)
  (h2 : price_per_item = 15)
  (h3 : cost = 100) :
  num_sold * price_per_item - cost = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_woodburning_profit_l3823_382318


namespace NUMINAMATH_CALUDE_vector_operation_result_l3823_382397

theorem vector_operation_result :
  let v₁ : Fin 3 → ℝ := ![(-3), 2, (-5)]
  let v₂ : Fin 3 → ℝ := ![1, 6, (-3)]
  2 • v₁ + v₂ = ![-5, 10, (-13)] := by
sorry

end NUMINAMATH_CALUDE_vector_operation_result_l3823_382397


namespace NUMINAMATH_CALUDE_four_points_with_given_distances_l3823_382355

theorem four_points_with_given_distances : 
  ∃! (points : Finset (ℝ × ℝ)), 
    Finset.card points = 4 ∧ 
    (∀ p ∈ points, 
      (abs p.2 = 2 ∧ abs p.1 = 4)) ∧
    (∀ p : ℝ × ℝ, 
      (abs p.2 = 2 ∧ abs p.1 = 4) → p ∈ points) :=
by sorry

end NUMINAMATH_CALUDE_four_points_with_given_distances_l3823_382355


namespace NUMINAMATH_CALUDE_slide_boys_count_l3823_382383

/-- The number of boys who initially went down the slide -/
def initial_boys : ℕ := 22

/-- The number of additional boys who went down the slide -/
def additional_boys : ℕ := 13

/-- The total number of boys who went down the slide -/
def total_boys : ℕ := initial_boys + additional_boys

theorem slide_boys_count : total_boys = 35 := by
  sorry

end NUMINAMATH_CALUDE_slide_boys_count_l3823_382383


namespace NUMINAMATH_CALUDE_set_M_equals_three_two_four_three_one_l3823_382358

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 = 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | (m - 1)*x - 1 = 0}

-- Define the set M
def M : Set ℝ := {m : ℝ | A ∩ B m = B m}

-- Theorem statement
theorem set_M_equals_three_two_four_three_one : M = {3/2, 4/3, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_M_equals_three_two_four_three_one_l3823_382358


namespace NUMINAMATH_CALUDE_angle_z_is_100_l3823_382312

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.X > 0 ∧ t.Y > 0 ∧ t.Z > 0 ∧ t.X + t.Y + t.Z = 180

-- Define the given conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.X + t.Y = 80 ∧ t.X = 2 * t.Y

-- Theorem statement
theorem angle_z_is_100 (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_conditions t) : 
  t.Z = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_z_is_100_l3823_382312


namespace NUMINAMATH_CALUDE_inequality_proof_l3823_382367

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  6 * a * b * c ≤ a * b * (a + b) + b * c * (b + c) + a * c * (a + c) ∧
  a * b * (a + b) + b * c * (b + c) + a * c * (a + c) ≤ 2 * (a^3 + b^3 + c^3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3823_382367


namespace NUMINAMATH_CALUDE_cream_fraction_is_three_tenths_l3823_382396

-- Define the initial contents of the cups
def initial_A : ℚ := 8
def initial_B : ℚ := 6
def initial_C : ℚ := 4

-- Define the transfer fractions
def transfer_A_to_B : ℚ := 1/3
def transfer_B_to_A : ℚ := 1/2
def transfer_A_to_C : ℚ := 1/4
def transfer_C_to_A : ℚ := 1/3

-- Define the function to calculate the final fraction of cream in Cup A
def final_cream_fraction (
  initial_A initial_B initial_C : ℚ
) (
  transfer_A_to_B transfer_B_to_A transfer_A_to_C transfer_C_to_A : ℚ
) : ℚ :=
  sorry -- The actual calculation would go here

-- Theorem statement
theorem cream_fraction_is_three_tenths :
  final_cream_fraction
    initial_A initial_B initial_C
    transfer_A_to_B transfer_B_to_A transfer_A_to_C transfer_C_to_A
  = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_cream_fraction_is_three_tenths_l3823_382396


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l3823_382319

/-- A regular polygon with side length 7 and exterior angle 72 degrees has a perimeter of 35 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧
  side_length = 7 ∧
  exterior_angle = 72 ∧
  n * exterior_angle = 360 →
  n * side_length = 35 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l3823_382319


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3823_382392

theorem inscribed_circle_radius 
  (r : ℝ) 
  (α γ : ℝ) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.sin α * Real.sin γ = 1/Real.sqrt 10) : 
  ∃ ρ : ℝ, ρ = ((2 * Real.sqrt 10 - 5) / 5) * r :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3823_382392


namespace NUMINAMATH_CALUDE_bake_sale_earnings_l3823_382364

theorem bake_sale_earnings (total : ℝ) (ingredients_cost shelter_donation : ℝ) 
  (h1 : ingredients_cost = 100)
  (h2 : shelter_donation = (total - ingredients_cost) / 2 + 10)
  (h3 : shelter_donation = 160) : 
  total = 400 := by
sorry

end NUMINAMATH_CALUDE_bake_sale_earnings_l3823_382364


namespace NUMINAMATH_CALUDE_no_function_exists_l3823_382328

theorem no_function_exists : ¬∃ (f : ℤ → ℤ), ∀ (x y z : ℤ), f (x * y) + f (x * z) - f x * f (y * z) ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_no_function_exists_l3823_382328


namespace NUMINAMATH_CALUDE_two_solutions_system_l3823_382386

theorem two_solutions_system (x y : ℝ) : 
  (x = 3 * x^2 + y^2 ∧ y = 3 * x * y) → 
  (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y ∧ p.1 = 3 * p.1^2 + p.2^2 ∧ p.2 = 3 * p.1 * p.2) ∧
  (∃! q : ℝ × ℝ, q ≠ p ∧ q.1 = x ∧ q.2 = y ∧ q.1 = 3 * q.1^2 + q.2^2 ∧ q.2 = 3 * q.1 * q.2) ∧
  (∀ r : ℝ × ℝ, r ≠ p ∧ r ≠ q → ¬(r.1 = 3 * r.1^2 + r.2^2 ∧ r.2 = 3 * r.1 * r.2)) :=
by sorry

end NUMINAMATH_CALUDE_two_solutions_system_l3823_382386


namespace NUMINAMATH_CALUDE_cube_less_than_three_times_square_l3823_382372

theorem cube_less_than_three_times_square (x : ℤ) :
  x^3 < 3*x^2 ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_less_than_three_times_square_l3823_382372


namespace NUMINAMATH_CALUDE_dubblefud_product_l3823_382375

/-- Represents the number of points for each chip color in the game of Dubblefud -/
structure ChipPoints where
  yellow : ℕ
  blue : ℕ
  green : ℕ

/-- Represents the number of chips for each color in a selection -/
structure ChipSelection where
  yellow : ℕ
  blue : ℕ
  green : ℕ

/-- The theorem statement for the Dubblefud game problem -/
theorem dubblefud_product (points : ChipPoints) (selection : ChipSelection) :
  points.yellow = 2 →
  points.blue = 4 →
  points.green = 5 →
  selection.blue = selection.green →
  selection.yellow = 4 →
  (points.yellow * selection.yellow) *
  (points.blue * selection.blue) *
  (points.green * selection.green) =
  72 * selection.blue :=
by sorry

end NUMINAMATH_CALUDE_dubblefud_product_l3823_382375


namespace NUMINAMATH_CALUDE_sin_alpha_on_ray_l3823_382346

/-- If the terminal side of angle α lies on the ray y = -√3x (x < 0), then sin α = √3/2 -/
theorem sin_alpha_on_ray (α : Real) : 
  (∃ (x y : Real), x < 0 ∧ y = -Real.sqrt 3 * x ∧ 
   (∃ (r : Real), x^2 + y^2 = r^2 ∧ Real.sin α = y / r)) →
  Real.sin α = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_on_ray_l3823_382346


namespace NUMINAMATH_CALUDE_bicycles_in_garage_l3823_382366

theorem bicycles_in_garage (cars : ℕ) (total_wheels : ℕ) (bicycle_wheels : ℕ) (car_wheels : ℕ) : 
  cars = 16 → 
  total_wheels = 82 → 
  bicycle_wheels = 2 → 
  car_wheels = 4 → 
  ∃ bicycles : ℕ, bicycles * bicycle_wheels + cars * car_wheels = total_wheels ∧ bicycles = 9 :=
by sorry

end NUMINAMATH_CALUDE_bicycles_in_garage_l3823_382366


namespace NUMINAMATH_CALUDE_constant_theta_is_plane_l3823_382385

-- Define spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the condition θ = c
def constant_theta (c : ℝ) (p : SphericalCoord) : Prop :=
  p.θ = c

-- Define a plane in 3D space
def is_plane (S : Set SphericalCoord) : Prop :=
  ∃ (a b d : ℝ), ∀ (p : SphericalCoord), p ∈ S ↔ 
    a * (p.ρ * Real.sin p.φ * Real.cos p.θ) + 
    b * (p.ρ * Real.sin p.φ * Real.sin p.θ) + 
    d * (p.ρ * Real.cos p.φ) = 0

-- Theorem statement
theorem constant_theta_is_plane (c : ℝ) :
  is_plane {p : SphericalCoord | constant_theta c p} :=
sorry

end NUMINAMATH_CALUDE_constant_theta_is_plane_l3823_382385


namespace NUMINAMATH_CALUDE_number_problem_l3823_382347

theorem number_problem : ∃ x : ℚ, x = 15 + (x * 9/64) + (x * 1/2) ∧ x = 960/23 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3823_382347


namespace NUMINAMATH_CALUDE_f_monotonicity_and_min_value_l3823_382338

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x - 2 / x - a * (Real.log x - 1 / x^2)

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := (x^2 + 2) * (x - a) / x^3

-- Define the minimum value function g
def g (a : ℝ) : ℝ := a - a * Real.log a - 1 / a

-- Theorem statement
theorem f_monotonicity_and_min_value (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Ioo 0 a, f_deriv a x < 0) ∧
  (∀ x ∈ Set.Ioi a, f_deriv a x > 0) ∧
  g a < 1 := by sorry

end

end NUMINAMATH_CALUDE_f_monotonicity_and_min_value_l3823_382338


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l3823_382343

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem derivative_f_at_one :
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l3823_382343


namespace NUMINAMATH_CALUDE_orthogonal_equal_magnitude_vectors_l3823_382322

/-- Given two vectors a and b in R^3, prove that if they are orthogonal and have equal magnitude,
    then their components satisfy specific values. -/
theorem orthogonal_equal_magnitude_vectors 
  (a b : ℝ × ℝ × ℝ) 
  (h_a : a.1 = 4 ∧ a.2.2 = -2) 
  (h_b : b.1 = 1 ∧ b.2.1 = 2) 
  (h_orthogonal : a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = 0) 
  (h_equal_magnitude : a.1^2 + a.2.1^2 + a.2.2^2 = b.1^2 + b.2.1^2 + b.2.2^2) :
  a.2.1 = 11/4 ∧ b.2.2 = 19/4 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_equal_magnitude_vectors_l3823_382322


namespace NUMINAMATH_CALUDE_digits_of_3_pow_15_times_5_pow_10_l3823_382300

theorem digits_of_3_pow_15_times_5_pow_10 : 
  (Nat.log 10 (3^15 * 5^10) + 1 : ℕ) = 18 :=
sorry

end NUMINAMATH_CALUDE_digits_of_3_pow_15_times_5_pow_10_l3823_382300


namespace NUMINAMATH_CALUDE_proposition_truth_count_l3823_382351

theorem proposition_truth_count (a b c : ℝ) : 
  (∃ (x y z : ℝ), x * z^2 > y * z^2 ∧ x ≤ y) ∨ 
  (∀ (x y z : ℝ), x > y → x * z^2 > y * z^2) ∨
  (∀ (x y z : ℝ), x ≤ y → x * z^2 ≤ y * z^2) :=
by sorry

end NUMINAMATH_CALUDE_proposition_truth_count_l3823_382351


namespace NUMINAMATH_CALUDE_soccer_players_count_l3823_382331

def total_students : ℕ := 400
def sports_proportion : ℚ := 52 / 100
def soccer_proportion : ℚ := 125 / 1000

theorem soccer_players_count :
  ⌊(total_students : ℚ) * sports_proportion * soccer_proportion⌋ = 26 := by
  sorry

end NUMINAMATH_CALUDE_soccer_players_count_l3823_382331


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l3823_382394

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 9 = 0 ∧ x₂^2 + m*x₂ + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l3823_382394


namespace NUMINAMATH_CALUDE_prism_volume_l3823_382362

/-- A right triangular prism with given base area and lateral face areas has volume 12 -/
theorem prism_volume (base_area : ℝ) (lateral_area1 lateral_area2 lateral_area3 : ℝ) 
  (h_base : base_area = 4)
  (h_lateral1 : lateral_area1 = 9)
  (h_lateral2 : lateral_area2 = 10)
  (h_lateral3 : lateral_area3 = 17) :
  base_area * (lateral_area1 / base_area.sqrt) = 12 :=
by sorry

end NUMINAMATH_CALUDE_prism_volume_l3823_382362


namespace NUMINAMATH_CALUDE_erins_launderette_machines_l3823_382334

/-- Represents the number of coins in a machine --/
structure CoinCount where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Calculates the total value of coins in dollars --/
def coinValue (c : CoinCount) : Rat :=
  (c.quarters * 25 + c.dimes * 10 + c.nickels * 5 + c.pennies) / 100

/-- Represents the launderette problem --/
structure LaunderetteProblem where
  machineCoins : CoinCount
  totalCashed : Rat
  minMachines : Nat
  maxMachines : Nat

/-- The specific launderette problem instance --/
def erinsProblem : LaunderetteProblem :=
  { machineCoins := { quarters := 80, dimes := 100, nickels := 50, pennies := 120 }
    totalCashed := 165
    minMachines := 3
    maxMachines := 5 }

theorem erins_launderette_machines (p : LaunderetteProblem) (h : p = erinsProblem) :
    ∃ n : Nat, n ≥ p.minMachines ∧ n ≤ p.maxMachines ∧ 
    n * coinValue p.machineCoins = p.totalCashed := by sorry

end NUMINAMATH_CALUDE_erins_launderette_machines_l3823_382334


namespace NUMINAMATH_CALUDE_trivia_team_group_size_l3823_382390

theorem trivia_team_group_size 
  (total_students : ℕ) 
  (unpicked_students : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 35) 
  (h2 : unpicked_students = 11) 
  (h3 : num_groups = 4) :
  (total_students - unpicked_students) / num_groups = 6 :=
by sorry

end NUMINAMATH_CALUDE_trivia_team_group_size_l3823_382390


namespace NUMINAMATH_CALUDE_worm_domino_division_l3823_382399

/-- A worm is represented by a list of directions (Up or Right) -/
inductive Direction
| Up
| Right

def Worm := List Direction

/-- Count the number of cells in a worm -/
def cellCount (w : Worm) : Nat :=
  w.length + 1

/-- Predicate to check if a worm can be divided into n dominoes -/
def canDivideIntoDominoes (w : Worm) (n : Nat) : Prop :=
  ∃ (division : List (Worm × Worm)), 
    division.length = n ∧
    (division.map (λ (p : Worm × Worm) => cellCount p.1 + cellCount p.2)).sum = cellCount w

/-- The main theorem -/
theorem worm_domino_division (w : Worm) (n : Nat) :
  n > 2 → (canDivideIntoDominoes w n ↔ cellCount w = 2 * n) :=
by sorry

end NUMINAMATH_CALUDE_worm_domino_division_l3823_382399


namespace NUMINAMATH_CALUDE_apartment_doors_count_l3823_382360

/-- Calculates the total number of doors needed for apartment buildings -/
def total_doors (num_buildings : ℕ) (floors_per_building : ℕ) (apartments_per_floor : ℕ) (doors_per_apartment : ℕ) : ℕ :=
  num_buildings * floors_per_building * apartments_per_floor * doors_per_apartment

/-- Proves that the total number of doors needed for the given apartment buildings is 1008 -/
theorem apartment_doors_count :
  total_doors 2 12 6 7 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_apartment_doors_count_l3823_382360


namespace NUMINAMATH_CALUDE_max_value_implies_m_l3823_382344

open Real

theorem max_value_implies_m (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = sin (x + π/2) + cos (x - π/2) + m) →
  (∃ x₀, ∀ x, f x ≤ f x₀) →
  (∃ x₁, f x₁ = 2 * sqrt 2) →
  m = sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_m_l3823_382344


namespace NUMINAMATH_CALUDE_zeros_in_square_of_nines_l3823_382371

/-- The number of zeros in the decimal expansion of (10^8 - 1)² is 7 -/
theorem zeros_in_square_of_nines : ∃ n : ℕ, n = 7 ∧ 
  (∃ m : ℕ, (10^8 - 1)^2 = m * 10^n + k ∧ k < 10^n ∧ k % 10 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_zeros_in_square_of_nines_l3823_382371


namespace NUMINAMATH_CALUDE_no_perfect_square_pair_l3823_382310

theorem no_perfect_square_pair : ¬∃ (a b : ℕ+), 
  (∃ (k : ℕ+), (a.val ^ 2 + b.val : ℕ) = k.val ^ 2) ∧ 
  (∃ (m : ℕ+), (b.val ^ 2 + a.val : ℕ) = m.val ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_pair_l3823_382310


namespace NUMINAMATH_CALUDE_greatest_b_value_l3823_382376

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 8*x - 15 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 8*5 - 15 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l3823_382376


namespace NUMINAMATH_CALUDE_parabola_intersection_distance_ellipse_parabola_intersection_distance_l3823_382324

/-- The distance between intersection points of a parabola and a vertical line -/
theorem parabola_intersection_distance 
  (a : ℝ) -- Parameter of the parabola
  (x_intersect : ℝ) -- x-coordinate of the vertical line
  (h1 : a > 0) -- Ensure parabola opens to the right
  (h2 : x_intersect > 0) -- Ensure vertical line is to the right of y-axis
  : 
  let y1 := Real.sqrt (4 * a * x_intersect)
  let y2 := -Real.sqrt (4 * a * x_intersect)
  abs (y1 - y2) = 2 * Real.sqrt (4 * a * x_intersect) :=
by sorry

/-- The main theorem about the specific ellipse and parabola -/
theorem ellipse_parabola_intersection_distance :
  let ellipse := fun (x y : ℝ) => x^2 / 25 + y^2 / 16 = 1
  let parabola := fun (x y : ℝ) => y^2 = (100 / 3) * x
  let x_intersect := 25 / 3
  abs ((Real.sqrt ((100 / 3) * x_intersect)) - (-Real.sqrt ((100 / 3) * x_intersect))) = 100 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_distance_ellipse_parabola_intersection_distance_l3823_382324


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l3823_382374

/-- Number of ways to distribute indistinguishable balls among distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 84 ways to distribute 6 indistinguishable balls among 4 distinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 84 := by sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l3823_382374


namespace NUMINAMATH_CALUDE_sum_of_squares_and_product_l3823_382370

theorem sum_of_squares_and_product (x y : ℝ) : 
  x = 2 / (Real.sqrt 3 + 1) →
  y = 2 / (Real.sqrt 3 - 1) →
  x^2 + x*y + y^2 = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_product_l3823_382370
