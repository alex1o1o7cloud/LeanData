import Mathlib

namespace NUMINAMATH_CALUDE_symmetry_yoz_proof_l1465_146556

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the yOz plane -/
def symmetryYOZ (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, p.z⟩

/-- The original point -/
def originalPoint : Point3D :=
  ⟨1, -2, 3⟩

theorem symmetry_yoz_proof :
  symmetryYOZ originalPoint = Point3D.mk (-1) (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_yoz_proof_l1465_146556


namespace NUMINAMATH_CALUDE_sum_divisibility_odd_sum_divisibility_l1465_146523

theorem sum_divisibility (n : ℕ) :
  (∃ k : ℕ, 2 * n ∣ (n * (n + 1) / 2)) ↔ (∃ k : ℕ, n = 4 * k - 1) :=
sorry

theorem odd_sum_divisibility (n : ℕ) :
  (∃ k : ℕ, (2 * n + 1) ∣ (n * (n + 1) / 2)) ↔
  ((2 * n + 1) % 4 = 1 ∨ (2 * n + 1) % 4 = 3) :=
sorry

end NUMINAMATH_CALUDE_sum_divisibility_odd_sum_divisibility_l1465_146523


namespace NUMINAMATH_CALUDE_ball_probabilities_l1465_146563

/-- The total number of balls in the box -/
def total_balls : ℕ := 12

/-- The number of red balls in the box -/
def red_balls : ℕ := 5

/-- The number of black balls in the box -/
def black_balls : ℕ := 4

/-- The number of white balls in the box -/
def white_balls : ℕ := 2

/-- The number of green balls in the box -/
def green_balls : ℕ := 1

/-- The probability of drawing a red or black ball -/
def prob_red_or_black : ℚ := (red_balls + black_balls) / total_balls

/-- The probability of drawing a red, black, or white ball -/
def prob_red_black_or_white : ℚ := (red_balls + black_balls + white_balls) / total_balls

theorem ball_probabilities :
  prob_red_or_black = 3/4 ∧ prob_red_black_or_white = 11/12 :=
by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l1465_146563


namespace NUMINAMATH_CALUDE_point_A_coordinates_l1465_146517

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally and vertically -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

theorem point_A_coordinates :
  ∀ (x y : ℝ),
  let A : Point := ⟨2*x + y, x - 2*y⟩
  let B : Point := translate A 1 (-4)
  B = ⟨x - y, y⟩ →
  A = ⟨1, 3⟩ :=
by
  sorry


end NUMINAMATH_CALUDE_point_A_coordinates_l1465_146517


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l1465_146510

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := a * x^2 + (1 - a) * x - 1 > 0

-- Define the solution set for a = 2
def solution_set_a2 : Set ℝ := {x | x < -1/2 ∨ x > 1}

-- Define the solution set for a > -1
def solution_set_a_gt_neg1 (a : ℝ) : Set ℝ :=
  if a = 0 then
    {x | x > 1}
  else if a > 0 then
    {x | x < -1/a ∨ x > 1}
  else
    {x | 1 < x ∧ x < -1/a}

theorem inequality_solution_sets :
  (∀ x, x ∈ solution_set_a2 ↔ inequality 2 x) ∧
  (∀ a, a > -1 → ∀ x, x ∈ solution_set_a_gt_neg1 a ↔ inequality a x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l1465_146510


namespace NUMINAMATH_CALUDE_catch_up_time_l1465_146512

-- Define the speeds of the girl, young man, and tram
def girl_speed : ℝ := 1
def young_man_speed : ℝ := 2 * girl_speed
def tram_speed : ℝ := 5 * young_man_speed

-- Define the time the young man waits before exiting the tram
def wait_time : ℝ := 8

-- Define the theorem
theorem catch_up_time : 
  ∀ (t : ℝ), 
  (girl_speed * wait_time + tram_speed * wait_time + girl_speed * t = young_man_speed * t) → 
  t = 88 := by
  sorry

end NUMINAMATH_CALUDE_catch_up_time_l1465_146512


namespace NUMINAMATH_CALUDE_fair_attendance_l1465_146535

theorem fair_attendance (projected_increase : Real) (actual_decrease : Real) :
  projected_increase = 0.25 →
  actual_decrease = 0.20 →
  (1 - actual_decrease) / (1 + projected_increase) * 100 = 64 := by
sorry

end NUMINAMATH_CALUDE_fair_attendance_l1465_146535


namespace NUMINAMATH_CALUDE_three_distinct_roots_l1465_146545

open Real

theorem three_distinct_roots (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    (abs (x₁^3 - a^3) = x₁ - a) ∧
    (abs (x₂^3 - a^3) = x₂ - a) ∧
    (abs (x₃^3 - a^3) = x₃ - a)) ↔ 
  (-2 / sqrt 3 < a ∧ a < -1 / sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_three_distinct_roots_l1465_146545


namespace NUMINAMATH_CALUDE_min_odd_counties_for_valid_island_l1465_146574

/-- A rectangular county with a diagonal road -/
structure County where
  has_diagonal_road : Bool

/-- A rectangular island composed of counties -/
structure Island where
  counties : List County
  is_rectangular : Bool
  has_closed_path : Bool
  no_self_intersections : Bool

/-- Predicate to check if an island satisfies all conditions -/
def satisfies_conditions (island : Island) : Prop :=
  island.is_rectangular ∧
  island.has_closed_path ∧
  island.no_self_intersections ∧
  island.counties.length % 2 = 1 ∧
  ∀ c ∈ island.counties, c.has_diagonal_road

theorem min_odd_counties_for_valid_island :
  ∀ island : Island,
    satisfies_conditions island →
    island.counties.length ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_odd_counties_for_valid_island_l1465_146574


namespace NUMINAMATH_CALUDE_complement_of_M_l1465_146595

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem complement_of_M : Set.compl M = {x : ℝ | x < 0 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l1465_146595


namespace NUMINAMATH_CALUDE_squares_sum_equality_l1465_146564

/-- Represents a 3-4-5 right triangle with squares on each side -/
structure Triangle345WithSquares where
  /-- Area of the square on the side of length 3 -/
  A : ℝ
  /-- Area of the square on the side of length 4 -/
  B : ℝ
  /-- Area of the square on the hypotenuse (side of length 5) -/
  C : ℝ
  /-- The area of the square on side 3 is 9 -/
  h_A : A = 9
  /-- The area of the square on side 4 is 16 -/
  h_B : B = 16
  /-- The area of the square on the hypotenuse is 25 -/
  h_C : C = 25

/-- 
For a 3-4-5 right triangle with squares constructed on each side, 
the sum of the areas of the squares on the two shorter sides 
equals the area of the square on the hypotenuse.
-/
theorem squares_sum_equality (t : Triangle345WithSquares) : t.A + t.B = t.C := by
  sorry

end NUMINAMATH_CALUDE_squares_sum_equality_l1465_146564


namespace NUMINAMATH_CALUDE_smallest_multiple_in_sequence_l1465_146557

theorem smallest_multiple_in_sequence (a : ℕ) : 
  (∀ i ∈ Finset.range 16, ∃ k : ℕ, a + 3 * i = 3 * k) →
  (6 * a + 3 * (0 + 1 + 2 + 3 + 4 + 5) = 5 * a + 3 * (11 + 12 + 13 + 14 + 15)) →
  a = 150 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_in_sequence_l1465_146557


namespace NUMINAMATH_CALUDE_log_inequality_range_l1465_146509

theorem log_inequality_range (a : ℝ) : 
  (a > 0 ∧ ∀ x : ℝ, 0 < x ∧ x ≤ 1 → 4 * x < Real.log x / Real.log a) ↔ 
  (0 < a ∧ a < 1) := by
sorry

end NUMINAMATH_CALUDE_log_inequality_range_l1465_146509


namespace NUMINAMATH_CALUDE_polynomial_root_product_l1465_146572

theorem polynomial_root_product (b c : ℤ) : 
  (∀ r : ℝ, r^2 - r - 2 = 0 → r^5 - b*r - c = 0) → b*c = 110 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_product_l1465_146572


namespace NUMINAMATH_CALUDE_freshmen_assignment_l1465_146504

/-- The number of ways to assign n freshmen to k classes with at least one freshman in each class -/
def assignFreshmen (n k : ℕ) : ℕ :=
  sorry

/-- The number of ways to arrange m groups into k classes -/
def arrangeGroups (m k : ℕ) : ℕ :=
  sorry

theorem freshmen_assignment :
  assignFreshmen 5 3 * arrangeGroups 3 3 = 150 :=
sorry

end NUMINAMATH_CALUDE_freshmen_assignment_l1465_146504


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1465_146543

theorem quadratic_equation_solution (x : ℝ) : 
  x^2 - 6*x + 8 = 0 → x ≠ 4 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1465_146543


namespace NUMINAMATH_CALUDE_negative_squared_greater_than_product_l1465_146579

theorem negative_squared_greater_than_product {a b : ℝ} (h1 : a < b) (h2 : b < 0) : a^2 > a*b := by
  sorry

end NUMINAMATH_CALUDE_negative_squared_greater_than_product_l1465_146579


namespace NUMINAMATH_CALUDE_b_plus_3c_positive_l1465_146576

theorem b_plus_3c_positive (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  b + 3 * c > 0 := by
  sorry

end NUMINAMATH_CALUDE_b_plus_3c_positive_l1465_146576


namespace NUMINAMATH_CALUDE_percentage_problem_l1465_146546

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 100) : 1.2 * x = 600 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1465_146546


namespace NUMINAMATH_CALUDE_fraction_2021_2019_position_l1465_146588

def sequence_position (m n : ℕ) : ℕ :=
  let k := m + n
  let previous_terms := (k - 1) * (k - 2) / 2
  let current_group_position := m
  previous_terms + current_group_position

theorem fraction_2021_2019_position :
  sequence_position 2021 2019 = 8159741 :=
by sorry

end NUMINAMATH_CALUDE_fraction_2021_2019_position_l1465_146588


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1465_146503

/-- Atomic weights of elements in g/mol -/
def atomic_weight : String → ℝ
  | "H" => 1
  | "N" => 14
  | "O" => 16
  | "S" => 32
  | "Fe" => 56
  | _ => 0

/-- Molecular weight of a compound given its chemical formula -/
def molecular_weight (formula : String) : ℝ := sorry

/-- The compound (NH4)2SO4·Fe2(SO4)3·6H2O -/
def compound : String := "(NH4)2SO4·Fe2(SO4)3·6H2O"

/-- Theorem stating that the molecular weight of (NH4)2SO4·Fe2(SO4)3·6H2O is 772 g/mol -/
theorem compound_molecular_weight :
  molecular_weight compound = 772 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1465_146503


namespace NUMINAMATH_CALUDE_triangle_properties_l1465_146568

def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi / 2 ∧
  Real.cos (2 * A) = -1 / 3 ∧
  c = Real.sqrt 3 ∧
  Real.sin A = Real.sqrt 6 * Real.sin C

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (h : triangle_ABC A B C a b c) : 
  a = 3 * Real.sqrt 2 ∧ 
  b = 5 ∧ 
  (1 / 2 : ℝ) * b * c * Real.sin A = (5 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1465_146568


namespace NUMINAMATH_CALUDE_expression_value_l1465_146575

theorem expression_value (a b c : ℝ) 
  (h1 : a - b = 2) 
  (h2 : a - c = Real.rpow 7 (1/3)) : 
  (c - b) * ((a - b)^2 + (a - b)*(a - c) + (a - c)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1465_146575


namespace NUMINAMATH_CALUDE_restaurant_gratuity_calculation_l1465_146586

theorem restaurant_gratuity_calculation (dish_prices : List ℝ) (tip_percentage : ℝ) : 
  dish_prices = [10, 13, 17, 15, 20] → 
  tip_percentage = 0.18 → 
  (dish_prices.sum * tip_percentage) = 13.50 := by
sorry

end NUMINAMATH_CALUDE_restaurant_gratuity_calculation_l1465_146586


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l1465_146500

-- First expression
theorem factorization_1 (a b : ℝ) :
  -6 * a * b + 3 * a^2 + 3 * b^2 = 3 * (a - b)^2 := by sorry

-- Second expression
theorem factorization_2 (x y m : ℝ) :
  y^2 * (2 - m) + x^2 * (m - 2) = (m - 2) * (x + y) * (x - y) := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l1465_146500


namespace NUMINAMATH_CALUDE_function_constant_l1465_146570

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 - x) - Real.log (1 + x) + a

theorem function_constant (a : ℝ) :
  (∃ (M N : ℝ), (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ), f a x ≤ M ∧ N ≤ f a x) ∧ M + N = 1) →
  a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_constant_l1465_146570


namespace NUMINAMATH_CALUDE_jerry_first_table_trays_l1465_146508

/-- The number of trays Jerry can carry at a time -/
def trays_per_trip : ℕ := 8

/-- The number of trips Jerry made -/
def number_of_trips : ℕ := 2

/-- The number of trays Jerry picked up from the second table -/
def trays_from_second_table : ℕ := 7

/-- The number of trays Jerry picked up from the first table -/
def trays_from_first_table : ℕ := trays_per_trip * number_of_trips - trays_from_second_table

theorem jerry_first_table_trays :
  trays_from_first_table = 9 :=
by sorry

end NUMINAMATH_CALUDE_jerry_first_table_trays_l1465_146508


namespace NUMINAMATH_CALUDE_f_of_3_eq_one_over_17_l1465_146592

/-- Given f(x) = (x-2)/(4x+5), prove that f(3) = 1/17 -/
theorem f_of_3_eq_one_over_17 (f : ℝ → ℝ) (h : ∀ x, f x = (x - 2) / (4 * x + 5)) : 
  f 3 = 1 / 17 := by
sorry

end NUMINAMATH_CALUDE_f_of_3_eq_one_over_17_l1465_146592


namespace NUMINAMATH_CALUDE_equation_b_is_quadratic_l1465_146522

/-- A quadratic equation in one variable is an equation that can be written in the form ax² + bx + c = 0, where a ≠ 0 and x is a variable. --/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(y) = 5y² - 5y represents the equation 5y = 5y². --/
def f (y : ℝ) : ℝ := 5 * y^2 - 5 * y

/-- Theorem: The equation 5y = 5y² is a quadratic equation. --/
theorem equation_b_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_equation_b_is_quadratic_l1465_146522


namespace NUMINAMATH_CALUDE_finite_divisor_property_l1465_146542

/-- A number is a finite decimal if it can be expressed as a/b where b is of the form 2^u * 5^v -/
def IsFiniteDecimal (q : ℚ) : Prop :=
  ∃ (a b : ℕ) (u v : ℕ), q = a / b ∧ b = 2^u * 5^v

/-- A natural number n has the property that all its divisors result in finite decimals -/
def HasFiniteDivisors (n : ℕ) : Prop :=
  ∀ k : ℕ, 0 < k → k < n → IsFiniteDecimal (n / k)

/-- The theorem stating that only 2, 3, and 6 have the finite divisor property -/
theorem finite_divisor_property :
  ∀ n : ℕ, HasFiniteDivisors n ↔ n = 2 ∨ n = 3 ∨ n = 6 :=
sorry

end NUMINAMATH_CALUDE_finite_divisor_property_l1465_146542


namespace NUMINAMATH_CALUDE_square_plus_one_eq_empty_l1465_146567

theorem square_plus_one_eq_empty : {x : ℝ | x^2 + 1 = 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_square_plus_one_eq_empty_l1465_146567


namespace NUMINAMATH_CALUDE_symmetric_center_of_f_l1465_146581

/-- The function f(x) = x³ - 6x² --/
def f (x : ℝ) : ℝ := x^3 - 6*x^2

/-- A function g is odd if g(-x) = -g(x) for all x --/
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

/-- The point (a, b) is a symmetric center of f if f(x+a) - b is an odd function --/
def is_symmetric_center (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  is_odd (fun x ↦ f (x + a) - b)

/-- The point (2, -16) is the symmetric center of f(x) = x³ - 6x² --/
theorem symmetric_center_of_f :
  is_symmetric_center f 2 (-16) :=
sorry

end NUMINAMATH_CALUDE_symmetric_center_of_f_l1465_146581


namespace NUMINAMATH_CALUDE_used_car_clients_l1465_146514

theorem used_car_clients (num_cars : ℕ) (selections_per_car : ℕ) (cars_per_client : ℕ)
  (h_num_cars : num_cars = 16)
  (h_selections_per_car : selections_per_car = 3)
  (h_cars_per_client : cars_per_client = 2) :
  (num_cars * selections_per_car) / cars_per_client = 24 := by
  sorry

end NUMINAMATH_CALUDE_used_car_clients_l1465_146514


namespace NUMINAMATH_CALUDE_new_recipe_water_amount_l1465_146539

/-- Represents a recipe ratio --/
structure RecipeRatio :=
  (flour : ℕ)
  (water : ℕ)
  (sugar : ℕ)

/-- The original recipe ratio --/
def original_ratio : RecipeRatio :=
  ⟨8, 4, 3⟩

/-- The new recipe ratio --/
def new_ratio : RecipeRatio :=
  ⟨4, 1, 3⟩

/-- Amount of sugar in the new recipe (in cups) --/
def new_sugar_amount : ℕ := 6

/-- Calculates the amount of water in the new recipe --/
def calculate_water_amount (r : RecipeRatio) (sugar_amount : ℕ) : ℚ :=
  (r.water : ℚ) * sugar_amount / r.sugar

/-- Theorem stating that the new recipe calls for 2 cups of water --/
theorem new_recipe_water_amount :
  calculate_water_amount new_ratio new_sugar_amount = 2 := by
  sorry

end NUMINAMATH_CALUDE_new_recipe_water_amount_l1465_146539


namespace NUMINAMATH_CALUDE_stock_z_shares_l1465_146530

/-- Represents the number of shares for each stock --/
structure ShareDistribution where
  v : ℕ
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the range of shares in a distribution --/
def calculateRange (shares : ShareDistribution) : ℕ :=
  max shares.v (max shares.w (max shares.x (max shares.y shares.z))) -
  min shares.v (min shares.w (min shares.x (min shares.y shares.z)))

/-- Theorem stating that the number of shares of stock z is 47 --/
theorem stock_z_shares : ∃ (initial : ShareDistribution),
  initial.v = 68 ∧
  initial.w = 112 ∧
  initial.x = 56 ∧
  initial.y = 94 ∧
  let final : ShareDistribution := {
    v := initial.v,
    w := initial.w,
    x := initial.x - 20,
    y := initial.y + 23,
    z := initial.z
  }
  calculateRange final - calculateRange initial = 14 →
  initial.z = 47 := by
  sorry

end NUMINAMATH_CALUDE_stock_z_shares_l1465_146530


namespace NUMINAMATH_CALUDE_eight_sided_die_product_l1465_146591

theorem eight_sided_die_product (x : ℕ) (h : 1 ≤ x ∧ x ≤ 8) : 
  192 ∣ (Nat.factorial 8 / x) := by sorry

end NUMINAMATH_CALUDE_eight_sided_die_product_l1465_146591


namespace NUMINAMATH_CALUDE_min_cuts_for_4x4x4_cube_l1465_146536

/-- Represents a cube with given dimensions -/
structure Cube where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cut operation on a cube -/
inductive Cut
  | X : Cut  -- Cut parallel to YZ plane
  | Y : Cut  -- Cut parallel to XZ plane
  | Z : Cut  -- Cut parallel to XY plane

/-- Function to calculate the minimum number of cuts required -/
def min_cuts_to_unit_cubes (c : Cube) : ℕ :=
  sorry

/-- Theorem stating the minimum number of cuts required for a 4x4x4 cube -/
theorem min_cuts_for_4x4x4_cube :
  let initial_cube : Cube := { length := 4, width := 4, height := 4 }
  min_cuts_to_unit_cubes initial_cube = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_cuts_for_4x4x4_cube_l1465_146536


namespace NUMINAMATH_CALUDE_lemonade_water_calculation_l1465_146511

/-- The amount of water needed to make lemonade with a given ratio and total volume -/
def water_needed (water_ratio : ℚ) (juice_ratio : ℚ) (total_gallons : ℚ) (liters_per_gallon : ℚ) : ℚ :=
  (water_ratio / (water_ratio + juice_ratio)) * (total_gallons * liters_per_gallon)

/-- Theorem stating the amount of water needed for the lemonade recipe -/
theorem lemonade_water_calculation :
  let water_ratio : ℚ := 8
  let juice_ratio : ℚ := 2
  let total_gallons : ℚ := 2
  let liters_per_gallon : ℚ := 3785/1000
  water_needed water_ratio juice_ratio total_gallons liters_per_gallon = 6056/1000 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_water_calculation_l1465_146511


namespace NUMINAMATH_CALUDE_expression_evaluation_l1465_146584

theorem expression_evaluation (a b : ℚ) (h1 : a = 1/2) (h2 : b = -2) :
  (2*a + b)^2 - (2*a - b)*(a + b) - 2*(a - 2*b)*(a + 2*b) = 37 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1465_146584


namespace NUMINAMATH_CALUDE_diagonals_of_25_sided_polygon_convex_polygon_25_sides_diagonals_l1465_146525

theorem diagonals_of_25_sided_polygon : ℕ → ℕ
  | n => (n * (n - 1)) / 2 - n

theorem convex_polygon_25_sides_diagonals :
  diagonals_of_25_sided_polygon 25 = 275 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_of_25_sided_polygon_convex_polygon_25_sides_diagonals_l1465_146525


namespace NUMINAMATH_CALUDE_adjacent_number_in_triangular_arrangement_l1465_146573

/-- Function to calculate the first number in the k-th row -/
def first_number_in_row (k : ℕ) : ℕ := (k - 1)^2 + 1

/-- Function to calculate the last number in the k-th row -/
def last_number_in_row (k : ℕ) : ℕ := k^2

/-- Function to determine if a number is in the k-th row -/
def is_in_row (n : ℕ) (k : ℕ) : Prop :=
  first_number_in_row k ≤ n ∧ n ≤ last_number_in_row k

/-- Function to calculate the number below a given number in the triangular arrangement -/
def number_below (n : ℕ) : ℕ :=
  let k := (n.sqrt + 1 : ℕ)
  let position := n - first_number_in_row k + 1
  first_number_in_row (k + 1) + position - 1

theorem adjacent_number_in_triangular_arrangement :
  is_in_row 267 17 → number_below 267 = 301 := by sorry

end NUMINAMATH_CALUDE_adjacent_number_in_triangular_arrangement_l1465_146573


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l1465_146551

theorem compare_negative_fractions : -3/4 > -4/5 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l1465_146551


namespace NUMINAMATH_CALUDE_bottles_sold_eq_60_l1465_146578

/-- Represents the sales data for Wal-Mart's thermometers and hot-water bottles --/
structure SalesData where
  thermometer_price : ℕ
  bottle_price : ℕ
  total_sales : ℕ
  thermometer_to_bottle_ratio : ℕ

/-- Calculates the number of hot-water bottles sold given the sales data --/
def bottles_sold (data : SalesData) : ℕ :=
  data.total_sales / (data.bottle_price + data.thermometer_price * data.thermometer_to_bottle_ratio)

/-- Theorem stating that given the specific sales data, 60 hot-water bottles were sold --/
theorem bottles_sold_eq_60 (data : SalesData) 
    (h1 : data.thermometer_price = 2)
    (h2 : data.bottle_price = 6)
    (h3 : data.total_sales = 1200)
    (h4 : data.thermometer_to_bottle_ratio = 7) : 
  bottles_sold data = 60 := by
  sorry

#eval bottles_sold { thermometer_price := 2, bottle_price := 6, total_sales := 1200, thermometer_to_bottle_ratio := 7 }

end NUMINAMATH_CALUDE_bottles_sold_eq_60_l1465_146578


namespace NUMINAMATH_CALUDE_amy_and_noah_total_l1465_146533

/-- The number of books each person has -/
structure BookCounts where
  maddie : ℕ
  luisa : ℕ
  amy : ℕ
  noah : ℕ

/-- The conditions of the problem -/
def book_problem (bc : BookCounts) : Prop :=
  bc.maddie = 2^4 - 1 ∧
  bc.luisa = 18 ∧
  bc.amy + bc.luisa = bc.maddie + 9 ∧
  bc.noah = Int.sqrt (bc.amy^2) + 2 ∧
  (Int.sqrt (bc.amy^2))^2 = bc.amy^2

/-- The theorem to prove -/
theorem amy_and_noah_total (bc : BookCounts) :
  book_problem bc → bc.amy + bc.noah = 14 := by
  sorry

end NUMINAMATH_CALUDE_amy_and_noah_total_l1465_146533


namespace NUMINAMATH_CALUDE_dot_product_MN_MO_l1465_146590

-- Define the circle O
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}

-- Define a line l (we don't need to specify its equation, just that it exists)
def line_l : Set (ℝ × ℝ) := sorry

-- Define points M and N as the intersection of line l and circle O
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry

-- Define point O as the center of the circle
def O : ℝ × ℝ := (0, 0)

-- State that M and N are on the circle
axiom M_on_circle : M ∈ circle_O
axiom N_on_circle : N ∈ circle_O

-- State that M and N are on the line l
axiom M_on_line : M ∈ line_l
axiom N_on_line : N ∈ line_l

-- Define the distance between M and N
axiom MN_distance : Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 4

-- Define vectors MN and MO
def vec_MN : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)
def vec_MO : ℝ × ℝ := (O.1 - M.1, O.2 - M.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem to prove
theorem dot_product_MN_MO : dot_product vec_MN vec_MO = 8 := by sorry

end NUMINAMATH_CALUDE_dot_product_MN_MO_l1465_146590


namespace NUMINAMATH_CALUDE_unique_integer_square_less_than_triple_l1465_146513

theorem unique_integer_square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_square_less_than_triple_l1465_146513


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_neg_three_m_range_when_subset_condition_holds_l1465_146501

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |2*x + m|

-- Part I
theorem solution_set_when_m_is_neg_three :
  {x : ℝ | f x (-3) ≤ 6} = {x : ℝ | -4/3 ≤ x ∧ x ≤ 8/3} := by sorry

-- Part II
theorem m_range_when_subset_condition_holds :
  (∀ x ∈ Set.Icc (-1 : ℝ) (1/2 : ℝ), f x m ≤ |2*x - 4|) →
  m ∈ Set.Icc (-5/2 : ℝ) (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_neg_three_m_range_when_subset_condition_holds_l1465_146501


namespace NUMINAMATH_CALUDE_abs_plus_power_minus_sqrt_inequality_system_solution_l1465_146534

-- Part 1
theorem abs_plus_power_minus_sqrt : |-2| + (1 + Real.sqrt 3)^0 - Real.sqrt 9 = 0 := by
  sorry

-- Part 2
theorem inequality_system_solution (x : ℝ) :
  (2 * x + 1 > 3 * (x - 1) ∧ x + (x - 1) / 3 < 1) ↔ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_plus_power_minus_sqrt_inequality_system_solution_l1465_146534


namespace NUMINAMATH_CALUDE_bobby_candy_theorem_l1465_146593

/-- The number of candy pieces Bobby ate -/
def pieces_eaten : ℕ := 23

/-- The number of candy pieces Bobby has left -/
def pieces_left : ℕ := 7

/-- The initial number of candy pieces Bobby had -/
def initial_pieces : ℕ := pieces_eaten + pieces_left

theorem bobby_candy_theorem : initial_pieces = 30 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_theorem_l1465_146593


namespace NUMINAMATH_CALUDE_increase_by_percentage_l1465_146541

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 200 →
  percentage = 25 →
  final = initial * (1 + percentage / 100) →
  final = 250 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l1465_146541


namespace NUMINAMATH_CALUDE_heximal_binary_equality_l1465_146506

/-- Converts a heximal (base-6) number to decimal --/
def heximal_to_decimal (a b c d : ℕ) : ℕ :=
  a * 6^3 + b * 6^2 + c * 6^1 + d * 6^0

/-- Converts a binary number to decimal --/
def binary_to_decimal (a b c d e f g h : ℕ) : ℕ :=
  a * 2^7 + b * 2^6 + c * 2^5 + d * 2^4 + e * 2^3 + f * 2^2 + g * 2^1 + h * 2^0

/-- The theorem stating that k = 3 is the unique solution --/
theorem heximal_binary_equality :
  ∃! k : ℕ, k > 0 ∧ heximal_to_decimal 1 0 k 5 = binary_to_decimal 1 1 1 0 1 1 1 1 :=
by sorry

end NUMINAMATH_CALUDE_heximal_binary_equality_l1465_146506


namespace NUMINAMATH_CALUDE_simplify_expression_l1465_146554

theorem simplify_expression (x : ℝ) : 8*x + 15 - 3*x + 27 = 5*x + 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1465_146554


namespace NUMINAMATH_CALUDE_sarahs_age_l1465_146558

theorem sarahs_age (ana mark billy sarah : ℝ) 
  (h1 : sarah = 3 * mark - 4)
  (h2 : mark = billy + 4)
  (h3 : billy = ana / 2)
  (h4 : ∃ (years : ℝ), ana + years = 15) :
  sarah = 30.5 := by
sorry

end NUMINAMATH_CALUDE_sarahs_age_l1465_146558


namespace NUMINAMATH_CALUDE_f_properties_l1465_146553

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 - 2

-- State the theorem
theorem f_properties :
  -- Function f is decreasing on (-∞, 0) and (2, +∞), and increasing on (0, 2)
  (∀ x y, x < y ∧ ((x < 0 ∧ y < 0) ∨ (x > 2 ∧ y > 2)) → f x > f y) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y) ∧
  -- Maximum value on [-2, 2] is 18
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≤ 18) ∧
  (∃ x, x ∈ Set.Icc (-2) 2 ∧ f x = 18) ∧
  -- Minimum value on [-2, 2] is -2
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≥ -2) ∧
  (∃ x, x ∈ Set.Icc (-2) 2 ∧ f x = -2) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l1465_146553


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1465_146537

/-- The perimeter of a rectangular field with length 7/5 times its width and width of 75 meters is 360 meters. -/
theorem rectangle_perimeter (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  width = 75 →
  length = (7/5) * width →
  perimeter = 2 * (length + width) →
  perimeter = 360 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1465_146537


namespace NUMINAMATH_CALUDE_log_equation_solution_l1465_146580

theorem log_equation_solution :
  ∃ x : ℝ, (Real.log (3 * x) - 4 * Real.log 9 = 3) ∧ (x = 2187000) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1465_146580


namespace NUMINAMATH_CALUDE_problem_solution_l1465_146598

theorem problem_solution (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 9) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1465_146598


namespace NUMINAMATH_CALUDE_difference_between_squares_l1465_146521

theorem difference_between_squares : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_squares_l1465_146521


namespace NUMINAMATH_CALUDE_fourth_child_age_l1465_146594

theorem fourth_child_age (ages : Fin 4 → ℕ) 
  (avg_age : (ages 0 + ages 1 + ages 2 + ages 3) / 4 = 9)
  (known_ages : ages 0 = 6 ∧ ages 1 = 8 ∧ ages 2 = 11) :
  ages 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_child_age_l1465_146594


namespace NUMINAMATH_CALUDE_edward_remaining_money_l1465_146549

/-- Given Edward's initial amount and the amount he spent, calculate how much he has left. -/
theorem edward_remaining_money (initial : ℕ) (spent : ℕ) (remaining : ℕ) 
  (h1 : initial = 19) 
  (h2 : spent = 13) 
  (h3 : remaining = initial - spent) : 
  remaining = 6 := by
  sorry

end NUMINAMATH_CALUDE_edward_remaining_money_l1465_146549


namespace NUMINAMATH_CALUDE_arithmetic_trapezoid_third_largest_angle_l1465_146589

/-- Represents a trapezoid with angles in arithmetic sequence -/
structure ArithmeticTrapezoid where
  /-- The smallest angle of the trapezoid -/
  smallest_angle : ℝ
  /-- The common difference between consecutive angles -/
  angle_diff : ℝ

/-- The theorem statement -/
theorem arithmetic_trapezoid_third_largest_angle
  (trap : ArithmeticTrapezoid)
  (sum_smallest_largest : trap.smallest_angle + (trap.smallest_angle + 3 * trap.angle_diff) = 200)
  (second_smallest : trap.smallest_angle + trap.angle_diff = 70) :
  trap.smallest_angle + 2 * trap.angle_diff = 130 := by
  sorry

#check arithmetic_trapezoid_third_largest_angle

end NUMINAMATH_CALUDE_arithmetic_trapezoid_third_largest_angle_l1465_146589


namespace NUMINAMATH_CALUDE_seashell_collection_problem_l1465_146582

/-- Calculates the total number of seashells after Leo gives away a quarter of his collection. -/
def final_seashell_count (henry_shells : ℕ) (paul_shells : ℕ) (initial_total : ℕ) : ℕ :=
  let leo_shells := initial_total - (henry_shells + paul_shells)
  let leo_remaining := leo_shells - (leo_shells / 4)
  henry_shells + paul_shells + leo_remaining

/-- Theorem stating that given the initial conditions, the final seashell count is 53. -/
theorem seashell_collection_problem :
  final_seashell_count 11 24 59 = 53 := by
  sorry

end NUMINAMATH_CALUDE_seashell_collection_problem_l1465_146582


namespace NUMINAMATH_CALUDE_point_coordinates_l1465_146552

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point M in the second quadrant, 5 units away from the x-axis
    and 3 units away from the y-axis, has coordinates (-3, 5) -/
theorem point_coordinates (M : Point) 
  (h1 : SecondQuadrant M) 
  (h2 : DistanceToXAxis M = 5) 
  (h3 : DistanceToYAxis M = 3) : 
  M.x = -3 ∧ M.y = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1465_146552


namespace NUMINAMATH_CALUDE_solution_set_equality_l1465_146561

-- Define the inequality function
def f (x : ℝ) := x^2 + 2*x - 3

-- State the theorem
theorem solution_set_equality :
  {x : ℝ | f x < 0} = Set.Ioo (-3 : ℝ) (1 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_solution_set_equality_l1465_146561


namespace NUMINAMATH_CALUDE_cookies_left_l1465_146555

-- Define the number of cookies in a dozen
def cookies_per_dozen : ℕ := 12

-- Define the number of dozens John buys
def dozens_bought : ℕ := 2

-- Define the number of cookies John eats
def cookies_eaten : ℕ := 3

-- Theorem statement
theorem cookies_left : 
  dozens_bought * cookies_per_dozen - cookies_eaten = 21 := by
sorry

end NUMINAMATH_CALUDE_cookies_left_l1465_146555


namespace NUMINAMATH_CALUDE_min_team_a_size_l1465_146585

theorem min_team_a_size : ∃ (a : ℕ), a > 0 ∧ 
  (∃ (b : ℕ), b > 0 ∧ b + 90 = 2 * (a - 90)) ∧
  (∃ (k : ℕ), a + k = 6 * (b - k)) ∧
  (∀ (a' : ℕ), a' > 0 → 
    (∃ (b' : ℕ), b' > 0 ∧ b' + 90 = 2 * (a' - 90)) →
    (∃ (k' : ℕ), a' + k' = 6 * (b' - k')) →
    a ≤ a') ∧
  a = 153 := by
sorry

end NUMINAMATH_CALUDE_min_team_a_size_l1465_146585


namespace NUMINAMATH_CALUDE_ratio_problem_l1465_146587

theorem ratio_problem (a b c : ℕ+) (x m : ℚ) :
  (∃ (k : ℕ+), a = 4 * k ∧ b = 5 * k ∧ c = 6 * k) →
  x = a + (25 / 100) * a →
  m = b - (40 / 100) * b →
  Even c →
  (∀ (a' b' c' : ℕ+), (∃ (k' : ℕ+), a' = 4 * k' ∧ b' = 5 * k' ∧ c' = 6 * k') → 
    a + b + c ≤ a' + b' + c') →
  m / x = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1465_146587


namespace NUMINAMATH_CALUDE_factor_sum_relation_l1465_146571

theorem factor_sum_relation (P Q R : ℝ) : 
  (∃ b c : ℝ, x^4 + P*x^2 + R*x + Q = (x^2 + 3*x + 7) * (x^2 + b*x + c)) →
  P + Q + R = 11*P - 1 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_relation_l1465_146571


namespace NUMINAMATH_CALUDE_game_lives_proof_l1465_146538

/-- Calculates the total number of lives for all players in a game --/
def totalLives (initialPlayers newPlayers livesPerPlayer : ℕ) : ℕ :=
  (initialPlayers + newPlayers) * livesPerPlayer

/-- Proves that the total number of lives is 24 given the specified conditions --/
theorem game_lives_proof :
  let initialPlayers : ℕ := 2
  let newPlayers : ℕ := 2
  let livesPerPlayer : ℕ := 6
  totalLives initialPlayers newPlayers livesPerPlayer = 24 := by
  sorry


end NUMINAMATH_CALUDE_game_lives_proof_l1465_146538


namespace NUMINAMATH_CALUDE_simplify_expression_l1465_146569

theorem simplify_expression (a b : ℝ) (h : a = -b) :
  (2 * a * b * (a^3 - b^3)) / (a^2 + a*b + b^2) - 
  ((a - b) * (a^4 - b^4)) / (a^2 - b^2) = -8 * a^3 := by
  sorry

#check simplify_expression

end NUMINAMATH_CALUDE_simplify_expression_l1465_146569


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1465_146544

theorem inequality_equivalence (x : ℝ) :
  (x - 3) / (2 - x) ≥ 0 ↔ Real.log (x - 2) ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1465_146544


namespace NUMINAMATH_CALUDE_intersection_A_B_l1465_146524

def A : Set ℝ := {x | x + 2 = 0}
def B : Set ℝ := {x | x^2 - 4 = 0}

theorem intersection_A_B : A ∩ B = {-2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1465_146524


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l1465_146565

theorem arithmetic_mean_geq_geometric_mean 
  (a b c : ℝ) 
  (ha : a ≥ 0) 
  (hb : b ≥ 0) 
  (hc : c ≥ 0) : 
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l1465_146565


namespace NUMINAMATH_CALUDE_min_b_value_l1465_146505

/-- Given positive integers x, y, z in ratio 3:4:7 and y = 15b - 5, 
    prove the minimum positive integer b is 3 -/
theorem min_b_value (x y z b : ℕ+) : 
  (∃ k : ℕ+, x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) →
  y = 15 * b - 5 →
  (∀ b' : ℕ+, b' < b → 
    ¬∃ x' y' z' : ℕ+, (∃ k : ℕ+, x' = 3 * k ∧ y' = 4 * k ∧ z' = 7 * k) ∧ 
    y' = 15 * b' - 5) →
  b = 3 := by
  sorry

#check min_b_value

end NUMINAMATH_CALUDE_min_b_value_l1465_146505


namespace NUMINAMATH_CALUDE_jennifer_run_time_l1465_146515

/-- 
Given:
- Jennifer ran 3 miles in 1/3 of the time it took Mark to run 5 miles
- Mark took 45 minutes to run 5 miles

Prove that Jennifer would take 35 minutes to run 7 miles at the same rate.
-/
theorem jennifer_run_time 
  (mark_distance : ℝ) 
  (mark_time : ℝ) 
  (jennifer_distance : ℝ) 
  (jennifer_time_ratio : ℝ) 
  (jennifer_new_distance : ℝ)
  (h1 : mark_distance = 5)
  (h2 : mark_time = 45)
  (h3 : jennifer_distance = 3)
  (h4 : jennifer_time_ratio = 1/3)
  (h5 : jennifer_new_distance = 7)
  : (jennifer_new_distance / jennifer_distance) * (jennifer_time_ratio * mark_time) = 35 := by
  sorry

#check jennifer_run_time

end NUMINAMATH_CALUDE_jennifer_run_time_l1465_146515


namespace NUMINAMATH_CALUDE_range_of_a_l1465_146562

/-- The function f(x) = x^2 + ax + b -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The set A = {x | f(x) ≤ 0} -/
def A (a b : ℝ) : Set ℝ := {x | f a b x ≤ 0}

/-- The set B = {x | f(f(x)) ≤ 5/4} -/
def B (a b : ℝ) : Set ℝ := {x | f a b (f a b x) ≤ 5/4}

/-- Theorem: Given f(x) = x^2 + ax + b, A = {x | f(x) ≤ 0}, B = {x | f(f(x)) ≤ 5/4},
    and A = B ≠ ∅, the range of a is [√5, 5] -/
theorem range_of_a (a b : ℝ) : 
  A a b = B a b ∧ A a b ≠ ∅ → a ∈ Set.Icc (Real.sqrt 5) 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1465_146562


namespace NUMINAMATH_CALUDE_sequence_value_l1465_146548

theorem sequence_value (a : ℕ → ℕ) :
  a 1 = 0 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n) →
  a 2012 = 2011 * 2012 :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_value_l1465_146548


namespace NUMINAMATH_CALUDE_square_construction_implies_parallel_l1465_146531

-- Define the triangle ABC
variable (A B C : Plane)

-- Define the squares constructed on the sides of triangle ABC
variable (A₂ B₁ B₂ C₁ : Plane)

-- Define the additional squares
variable (A₃ A₄ B₃ B₄ : Plane)

-- Define the property of being a square
def is_square (P Q R S : Plane) : Prop := sorry

-- Define the property of being external to a triangle
def is_external_to_triangle (S₁ S₂ S₃ S₄ P Q R : Plane) : Prop := sorry

-- Define the property of being parallel
def is_parallel (P₁ P₂ Q₁ Q₂ : Plane) : Prop := sorry

theorem square_construction_implies_parallel :
  is_square A B B₁ A₂ →
  is_square B C C₁ B₂ →
  is_square C A A₁ C₂ →
  is_external_to_triangle A B B₁ A₂ A B C →
  is_external_to_triangle B C C₁ B₂ B C A →
  is_external_to_triangle C A A₁ C₂ C A B →
  is_square A₁ A₂ A₃ A₄ →
  is_square B₁ B₂ B₃ B₄ →
  is_external_to_triangle A₁ A₂ A₃ A₄ A A₁ A₂ →
  is_external_to_triangle B₁ B₂ B₃ B₄ B B₁ B₂ →
  is_parallel A₃ B₄ A B := by sorry

end NUMINAMATH_CALUDE_square_construction_implies_parallel_l1465_146531


namespace NUMINAMATH_CALUDE_book_selection_probabilities_l1465_146597

def chinese_books : ℕ := 4
def math_books : ℕ := 3
def total_books : ℕ := chinese_books + math_books
def books_to_select : ℕ := 2

def total_combinations : ℕ := Nat.choose total_books books_to_select

theorem book_selection_probabilities :
  let prob_two_math : ℚ := (Nat.choose math_books books_to_select : ℚ) / total_combinations
  let prob_one_each : ℚ := (chinese_books * math_books : ℚ) / total_combinations
  prob_two_math = 1/7 ∧ prob_one_each = 4/7 := by sorry

end NUMINAMATH_CALUDE_book_selection_probabilities_l1465_146597


namespace NUMINAMATH_CALUDE_unpainted_cubes_4x4x4_l1465_146526

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_units : Nat
  painted_per_face : Nat

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpainted_cubes (cube : PaintedCube) : Nat :=
  cube.total_units - (cube.painted_per_face * 6)

/-- Theorem: A 4x4x4 cube with 4 unit squares painted on each face has 40 unpainted unit cubes -/
theorem unpainted_cubes_4x4x4 :
  let cube : PaintedCube := {
    size := 4,
    total_units := 64,
    painted_per_face := 4
  }
  unpainted_cubes cube = 40 := by
  sorry


end NUMINAMATH_CALUDE_unpainted_cubes_4x4x4_l1465_146526


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1465_146577

theorem solution_set_equivalence :
  {x : ℝ | x - 2 > 1 ∧ x < 4} = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1465_146577


namespace NUMINAMATH_CALUDE_problem_solution_l1465_146540

theorem problem_solution (a b c d e : ℝ) 
  (h1 : |2 + a| + |b - 3| = 0)
  (h2 : c ≠ 0)
  (h3 : 1 / c = -d)
  (h4 : e = -5) :
  -a^b + 1/c - e + d = 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1465_146540


namespace NUMINAMATH_CALUDE_inequalities_proof_l1465_146502

theorem inequalities_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : Real.sqrt a ^ 3 + Real.sqrt b ^ 3 + Real.sqrt c ^ 3 = 1) :
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l1465_146502


namespace NUMINAMATH_CALUDE_negative_integer_sum_with_square_is_six_l1465_146532

theorem negative_integer_sum_with_square_is_six (N : ℤ) : 
  N < 0 → N^2 + N = 6 → N = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_sum_with_square_is_six_l1465_146532


namespace NUMINAMATH_CALUDE_first_term_arithmetic_progression_l1465_146599

/-- 
For a decreasing arithmetic progression with first term a, sum S, 
number of terms n, and common difference d, the following equation holds:
a = S/n + (n-1)d/2
-/
theorem first_term_arithmetic_progression 
  (a : ℝ) (S : ℝ) (n : ℕ) (d : ℝ) 
  (h1 : n > 0) 
  (h2 : d < 0) -- Ensures it's a decreasing progression
  (h3 : S = n/2 * (2*a + (n-1)*d)) -- Sum formula for arithmetic progression
  : a = S/n + (n-1)*d/2 := by
  sorry

end NUMINAMATH_CALUDE_first_term_arithmetic_progression_l1465_146599


namespace NUMINAMATH_CALUDE_diamond_calculation_l1465_146583

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_calculation :
  (diamond (diamond 2 (1/2)) (-4)) - (diamond 2 (diamond (1/2) (-4))) = -5/12 :=
by sorry

end NUMINAMATH_CALUDE_diamond_calculation_l1465_146583


namespace NUMINAMATH_CALUDE_line_passes_through_other_lattice_points_l1465_146547

theorem line_passes_through_other_lattice_points :
  ∃ (x y : ℤ), x ≠ 0 ∧ x ≠ 5 ∧ y ≠ 0 ∧ y ≠ 3 ∧ 5 * y = 3 * x := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_other_lattice_points_l1465_146547


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1465_146559

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 12) :
  (1 / a + 1 / b) ≥ 1 / 3 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 12 ∧ 1 / a₀ + 1 / b₀ = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1465_146559


namespace NUMINAMATH_CALUDE_special_sequence_second_term_l1465_146516

/-- An arithmetic sequence with three terms -/
structure ArithmeticSequence3 where
  a : ℤ  -- First term
  b : ℤ  -- Second term
  c : ℤ  -- Third term
  is_arithmetic : b - a = c - b

/-- The second term of an arithmetic sequence with 3² as first term and 3⁴ as third term -/
def second_term_of_special_sequence : ℤ := 45

/-- Theorem stating that the second term of the special arithmetic sequence is 45 -/
theorem special_sequence_second_term :
  ∀ (seq : ArithmeticSequence3), 
  seq.a = 3^2 ∧ seq.c = 3^4 → seq.b = second_term_of_special_sequence :=
by sorry

end NUMINAMATH_CALUDE_special_sequence_second_term_l1465_146516


namespace NUMINAMATH_CALUDE_max_value_expression_l1465_146527

def S : Set ℕ := {0, 1, 2, 3}

theorem max_value_expression (a b c d : ℕ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S) :
  (∀ x y z w : ℕ, x ∈ S → y ∈ S → z ∈ S → w ∈ S →
    z * (x^y + 1) - w ≤ c * (a^b + 1) - d) →
  c * (a^b + 1) - d = 30 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l1465_146527


namespace NUMINAMATH_CALUDE_harmonic_point_3_m_harmonic_point_hyperbola_l1465_146550

-- Definition of a harmonic point
def is_harmonic_point (x y t : ℝ) : Prop :=
  x^2 = 4*y + t ∧ y^2 = 4*x + t ∧ x ≠ y

-- Theorem for part 1
theorem harmonic_point_3_m (m : ℝ) :
  is_harmonic_point 3 m (3^2 - 4*m) → m = -7 :=
sorry

-- Theorem for part 2
theorem harmonic_point_hyperbola (k : ℝ) :
  (∃ x : ℝ, -3 < x ∧ x < -1 ∧ is_harmonic_point x (k/x) (x^2 - 4*(k/x))) →
  3 < k ∧ k < 4 :=
sorry

end NUMINAMATH_CALUDE_harmonic_point_3_m_harmonic_point_hyperbola_l1465_146550


namespace NUMINAMATH_CALUDE_unique_xxyy_square_l1465_146596

/-- Represents a four-digit number in the form xxyy --/
def xxyy_number (x y : Nat) : Nat :=
  1100 * x + 11 * y

/-- Predicate to check if a number is a perfect square --/
def is_perfect_square (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

theorem unique_xxyy_square :
  ∀ x y : Nat, x < 10 → y < 10 →
    (is_perfect_square (xxyy_number x y) ↔ x = 7 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_unique_xxyy_square_l1465_146596


namespace NUMINAMATH_CALUDE_store_profit_theorem_l1465_146518

/-- Represents the store's inventory and pricing information -/
structure Store :=
  (total_items : ℕ)
  (purchase_price_A : ℝ)
  (selling_price_A : ℝ)
  (purchase_price_B : ℝ)
  (original_selling_price_B : ℝ)
  (original_daily_sales_B : ℝ)
  (sales_increase_per_yuan : ℝ)
  (target_daily_profit_B : ℝ)

/-- Calculates the total profit based on the number of type A items -/
def total_profit (s : Store) (x : ℝ) : ℝ :=
  (s.selling_price_A - s.purchase_price_A) * x +
  (s.original_selling_price_B - s.purchase_price_B) * (s.total_items - x)

/-- Calculates the daily profit for type B items based on the new selling price -/
def daily_profit_B (s : Store) (new_price : ℝ) : ℝ :=
  (new_price - s.purchase_price_B) *
  (s.original_daily_sales_B + s.sales_increase_per_yuan * (s.original_selling_price_B - new_price))

/-- The main theorem stating the properties of the store's profit calculations -/
theorem store_profit_theorem (s : Store) (x : ℝ) :
  (s.total_items = 80 ∧
   s.purchase_price_A = 40 ∧
   s.selling_price_A = 55 ∧
   s.purchase_price_B = 28 ∧
   s.original_selling_price_B = 40 ∧
   s.original_daily_sales_B = 4 ∧
   s.sales_increase_per_yuan = 2 ∧
   s.target_daily_profit_B = 96) →
  (total_profit s x = 3 * x + 960 ∧
   (daily_profit_B s 34 = 96 ∨ daily_profit_B s 36 = 96)) :=
by sorry


end NUMINAMATH_CALUDE_store_profit_theorem_l1465_146518


namespace NUMINAMATH_CALUDE_solution_set_all_reals_solution_set_interval_l1465_146507

-- Part 1
theorem solution_set_all_reals (x : ℝ) : 8 * x - 1 ≤ 16 * x^2 := by sorry

-- Part 2
theorem solution_set_interval (a x : ℝ) (h : a < 0) :
  x^2 - 2*a*x - 3*a^2 < 0 ↔ 3*a < x ∧ x < -a := by sorry

end NUMINAMATH_CALUDE_solution_set_all_reals_solution_set_interval_l1465_146507


namespace NUMINAMATH_CALUDE_train_speed_l1465_146528

/-- The speed of a train given its length, time to cross a moving person, and the person's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (person_speed : ℝ) :
  train_length = 500 →
  crossing_time = 29.997600191984642 →
  person_speed = 3 →
  ∃ (train_speed : ℝ), abs (train_speed - 63) < 0.1 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l1465_146528


namespace NUMINAMATH_CALUDE_mean_temperature_l1465_146519

def temperatures : List ℝ := [-8, -5, -3, -5, 2, 4, 3, -1]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : ℝ) = -1.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l1465_146519


namespace NUMINAMATH_CALUDE_num_valid_configs_eq_30_l1465_146560

/-- A valid grid configuration -/
structure GridConfig where
  numbers : Fin 9 → Bool
  positions : Fin 6 → Fin 9
  left_greater_right : ∀ i : Fin 2, positions (2*i) > positions (2*i + 1)
  top_smaller_bottom : ∀ i : Fin 3, positions i < positions (i + 3)
  all_different : ∀ i j : Fin 6, i ≠ j → positions i ≠ positions j
  used_numbers : ∀ i : Fin 9, numbers i = (∃ j : Fin 6, positions j = i)

/-- The number of valid grid configurations -/
def num_valid_configs : ℕ := sorry

/-- The main theorem: there are exactly 30 valid grid configurations -/
theorem num_valid_configs_eq_30 : num_valid_configs = 30 := by sorry

end NUMINAMATH_CALUDE_num_valid_configs_eq_30_l1465_146560


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l1465_146529

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Evaluation of a quadratic polynomial at a point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℚ) : ℚ :=
  p.a * x^2 + p.b * x + p.c

/-- Condition for divisibility by (x - 2)(x + 2)(x - 9) -/
def isDivisibleByFactors (p : QuadraticPolynomial) : Prop :=
  (p.eval 2)^3 = 2 ∧ (p.eval (-2))^3 = -2 ∧ (p.eval 9)^3 = 9

theorem quadratic_polynomial_property (p : QuadraticPolynomial) 
  (h : isDivisibleByFactors p) : p.eval 14 = -230/11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l1465_146529


namespace NUMINAMATH_CALUDE_dice_sum_repetition_l1465_146520

theorem dice_sum_repetition (n : ℕ) (m : ℕ) (h1 : n = 21) (h2 : m = 22) :
  m > n → ∀ f : ℕ → ℕ, ∃ i j, i < j ∧ j < m ∧ f i = f j :=
by sorry

end NUMINAMATH_CALUDE_dice_sum_repetition_l1465_146520


namespace NUMINAMATH_CALUDE_jack_emails_afternoon_l1465_146566

theorem jack_emails_afternoon (morning_emails : ℕ) (total_emails : ℕ) (afternoon_emails : ℕ) 
  (h1 : morning_emails = 3)
  (h2 : total_emails = 8)
  (h3 : afternoon_emails = total_emails - morning_emails) :
  afternoon_emails = 5 := by
  sorry

end NUMINAMATH_CALUDE_jack_emails_afternoon_l1465_146566
