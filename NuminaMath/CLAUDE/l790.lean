import Mathlib

namespace NUMINAMATH_CALUDE_inscribed_triangle_relation_l790_79038

-- Define a triangle inscribed in a unit circle
structure InscribedTriangle where
  a : Real
  b : Real
  c : Real
  α : Real
  β : Real
  γ : Real
  sum_angles : α + β + γ = Real.pi
  side_a : a = 2 * Real.sin (α / 2)
  side_b : b = 2 * Real.sin (β / 2)
  side_c : c = 2 * Real.sin (γ / 2)

-- Theorem statement
theorem inscribed_triangle_relation (t : InscribedTriangle) :
  t.a^2 + t.b^2 + t.c^2 = 8 + 4 * Real.cos t.α * Real.cos t.β * Real.cos t.γ := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_relation_l790_79038


namespace NUMINAMATH_CALUDE_sum_of_abc_l790_79063

theorem sum_of_abc (a b c : ℕ+) (h1 : a * b + c = 31)
                   (h2 : b * c + a = 31) (h3 : a * c + b = 31) :
  (a : ℕ) + b + c = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l790_79063


namespace NUMINAMATH_CALUDE_value_of_T_l790_79040

theorem value_of_T : ∃ T : ℚ, (1/2 : ℚ) * (1/7 : ℚ) * T = (1/3 : ℚ) * (1/5 : ℚ) * 90 ∧ T = 84 := by
  sorry

end NUMINAMATH_CALUDE_value_of_T_l790_79040


namespace NUMINAMATH_CALUDE_pentadecagon_triangles_l790_79087

/-- The number of sides in a regular pentadecagon -/
def n : ℕ := 15

/-- The total number of triangles that can be formed using any three vertices of a regular pentadecagon -/
def total_triangles : ℕ := n.choose 3

/-- The number of triangles formed by three consecutive vertices in a regular pentadecagon -/
def consecutive_triangles : ℕ := n

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon, 
    where no triangle is formed by three consecutive vertices -/
def valid_triangles : ℕ := total_triangles - consecutive_triangles

theorem pentadecagon_triangles : valid_triangles = 440 := by
  sorry

end NUMINAMATH_CALUDE_pentadecagon_triangles_l790_79087


namespace NUMINAMATH_CALUDE_cube_inequality_l790_79047

theorem cube_inequality (a b : ℝ) : a < b → a^3 < b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l790_79047


namespace NUMINAMATH_CALUDE_solutions_difference_squared_l790_79054

theorem solutions_difference_squared (α β : ℝ) : 
  α ≠ β ∧ α^2 = 2*α + 1 ∧ β^2 = 2*β + 1 → (α - β)^2 = 8 := by sorry

end NUMINAMATH_CALUDE_solutions_difference_squared_l790_79054


namespace NUMINAMATH_CALUDE_pipe_fill_time_l790_79056

/-- Given two pipes that can fill a pool, where one takes T hours and the other takes 12 hours,
    prove that if both pipes together take 4.8 hours to fill the pool, then T = 8. -/
theorem pipe_fill_time (T : ℝ) :
  T > 0 →
  1 / T + 1 / 12 = 1 / 4.8 →
  T = 8 :=
by sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l790_79056


namespace NUMINAMATH_CALUDE_symmetric_point_yoz_plane_l790_79027

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The yOz plane in 3D space -/
def yOzPlane : Set Point3D := {p : Point3D | p.x = 0}

/-- Symmetry with respect to the yOz plane -/
def symmetricPointYOz (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

/-- Theorem: The point (-1, -2, 3) is symmetric to (1, -2, 3) with respect to the yOz plane -/
theorem symmetric_point_yoz_plane :
  let p1 : Point3D := { x := 1, y := -2, z := 3 }
  let p2 : Point3D := { x := -1, y := -2, z := 3 }
  symmetricPointYOz p1 = p2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_yoz_plane_l790_79027


namespace NUMINAMATH_CALUDE_polynomial_value_l790_79093

theorem polynomial_value (a b : ℝ) : 
  (a * 2^3 + b * 2 + 3 = 5) → 
  (a * (-2)^2 - 1/2 * b * (-2) - 3 = -2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_value_l790_79093


namespace NUMINAMATH_CALUDE_max_quarters_sasha_l790_79094

/-- Represents the value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- Represents the value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- Represents the value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- Represents the total amount Sasha has in dollars -/
def total_amount : ℚ := 480 / 100

theorem max_quarters_sasha : 
  ∀ q : ℕ, 
    (q : ℚ) * quarter_value + 
    (2 * q : ℚ) * nickel_value + 
    (q : ℚ) * dime_value ≤ total_amount → 
    q ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_max_quarters_sasha_l790_79094


namespace NUMINAMATH_CALUDE_flowers_left_in_peters_garden_l790_79031

/-- The number of flowers in Amanda's garden -/
def amanda_flowers : ℕ := 20

/-- The number of flowers in Peter's garden before giving away -/
def peter_flowers : ℕ := 3 * amanda_flowers

/-- The number of flowers Peter gave away -/
def flowers_given_away : ℕ := 15

/-- Theorem: The number of flowers left in Peter's garden is 45 -/
theorem flowers_left_in_peters_garden :
  peter_flowers - flowers_given_away = 45 := by
  sorry

end NUMINAMATH_CALUDE_flowers_left_in_peters_garden_l790_79031


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l790_79039

theorem quadratic_equation_properties : ∃ (x y : ℝ),
  x^2 + 1984513*x + 3154891 = 0 ∧
  y^2 + 1984513*y + 3154891 = 0 ∧
  x ≠ y ∧
  (∀ z : ℤ, z^2 + 1984513*z + 3154891 ≠ 0) ∧
  x ≤ 0 ∧
  y ≤ 0 ∧
  1/x + 1/y ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l790_79039


namespace NUMINAMATH_CALUDE_minimum_cologne_drops_l790_79049

theorem minimum_cologne_drops (f : ℕ) (n : ℕ) : 
  f > 0 →  -- number of boys is positive
  n > 0 →  -- number of drops is positive
  (∀ g : ℕ, g ≤ 4 → (3 * n : ℝ) ≥ (f * (n / 2 + 15) : ℝ)) →  -- no girl receives more than 3 bottles worth
  (f * ((n / 2 : ℝ) - 15) > (3 * n : ℝ)) →  -- mother receives more than any girl
  n ≥ 53 :=
by sorry

end NUMINAMATH_CALUDE_minimum_cologne_drops_l790_79049


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l790_79004

-- Define the condition
def condition (A : ℝ × ℝ) : Prop :=
  ∃ k : ℤ, A = (k * Real.pi, 0)

-- Define the statement
def statement (A : ℝ × ℝ) : Prop :=
  ∀ x : ℝ, Real.tan (A.1 + x) = -Real.tan (A.1 - x)

-- Theorem stating the condition is sufficient but not necessary
theorem condition_sufficient_not_necessary :
  (∀ A : ℝ × ℝ, condition A → statement A) ∧
  ¬(∀ A : ℝ × ℝ, statement A → condition A) :=
sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l790_79004


namespace NUMINAMATH_CALUDE_trigonometric_identities_l790_79021

theorem trigonometric_identities :
  (∃ (x y : Real),
    x = Real.tan (20 * π / 180) ∧
    y = Real.tan (40 * π / 180) ∧
    x + y + Real.sqrt 3 * x * y = Real.sqrt 3) ∧
  (∃ (z w : Real),
    z = Real.sin (50 * π / 180) ∧
    w = Real.tan (10 * π / 180) ∧
    z * (1 + Real.sqrt 3 * w) = 1) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l790_79021


namespace NUMINAMATH_CALUDE_min_odd_integers_l790_79000

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum1 : a + b = 22)
  (sum2 : a + b + c + d = 36)
  (sum3 : a + b + c + d + e + f = 50) :
  ∃ (a' b' c' d' e' f' : ℤ), 
    a' % 2 = 0 ∧ b' % 2 = 0 ∧ c' % 2 = 0 ∧ d' % 2 = 0 ∧ e' % 2 = 0 ∧ f' % 2 = 0 ∧
    a' + b' = 22 ∧
    a' + b' + c' + d' = 36 ∧
    a' + b' + c' + d' + e' + f' = 50 :=
by sorry

end NUMINAMATH_CALUDE_min_odd_integers_l790_79000


namespace NUMINAMATH_CALUDE_emily_beads_count_l790_79003

/-- The number of necklaces Emily made -/
def num_necklaces : ℕ := 11

/-- The number of beads required for each necklace -/
def beads_per_necklace : ℕ := 28

/-- The total number of beads Emily had -/
def total_beads : ℕ := num_necklaces * beads_per_necklace

theorem emily_beads_count : total_beads = 308 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_count_l790_79003


namespace NUMINAMATH_CALUDE_percentage_difference_l790_79090

theorem percentage_difference (x y : ℝ) (h : x = 18 * y) :
  (x - y) / x * 100 = 94.44 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l790_79090


namespace NUMINAMATH_CALUDE_staircase_extension_l790_79023

/-- Calculates the number of toothpicks needed for a staircase of n steps -/
def toothpicks (n : ℕ) : ℕ := 
  if n = 0 then 0
  else if n = 1 then 4
  else 4 + (n - 1) * 3 + ((n - 1) * (n - 2)) / 2

/-- The number of additional toothpicks needed to extend an n-step staircase to an m-step staircase -/
def additional_toothpicks (n m : ℕ) : ℕ := toothpicks m - toothpicks n

theorem staircase_extension :
  additional_toothpicks 3 6 = 36 :=
sorry

end NUMINAMATH_CALUDE_staircase_extension_l790_79023


namespace NUMINAMATH_CALUDE_geese_flew_away_l790_79050

/-- Proves that the number of geese that flew away is equal to the difference
    between the initial number of geese and the number of geese left in the field. -/
theorem geese_flew_away (initial : ℕ) (left : ℕ) (flew_away : ℕ)
    (h1 : initial = 51)
    (h2 : left = 23)
    (h3 : initial ≥ left) :
  flew_away = initial - left :=
by sorry

end NUMINAMATH_CALUDE_geese_flew_away_l790_79050


namespace NUMINAMATH_CALUDE_logarithmic_equation_solutions_l790_79035

theorem logarithmic_equation_solutions (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, x > 0 → x ≠ 1 →
    ((3 * (Real.log x / Real.log a) - 2) * (Real.log a / Real.log x)^2 = Real.log x / (Real.log a / 2) - 3) ↔
    (x = 1/a ∨ x = Real.sqrt a ∨ x = a^2) :=
by sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solutions_l790_79035


namespace NUMINAMATH_CALUDE_raisin_problem_l790_79078

theorem raisin_problem (x : ℕ) : 
  (x / 3 : ℚ) + 4 + ((2 * x / 3 - 4) / 2 : ℚ) + 16 = x → x = 54 := by
  sorry

end NUMINAMATH_CALUDE_raisin_problem_l790_79078


namespace NUMINAMATH_CALUDE_bus_tour_sales_l790_79010

theorem bus_tour_sales (total_tickets : ℕ) (senior_price regular_price : ℕ) (regular_tickets : ℕ)
  (h1 : total_tickets = 65)
  (h2 : senior_price = 10)
  (h3 : regular_price = 15)
  (h4 : regular_tickets = 41) :
  (total_tickets - regular_tickets) * senior_price + regular_tickets * regular_price = 855 := by
  sorry

end NUMINAMATH_CALUDE_bus_tour_sales_l790_79010


namespace NUMINAMATH_CALUDE_factorial_not_ending_19760_l790_79076

theorem factorial_not_ending_19760 (n : ℕ+) : ¬ ∃ k : ℕ, (n!:ℕ) % (10^(k+5)) = 19760 * 10^k :=
sorry

end NUMINAMATH_CALUDE_factorial_not_ending_19760_l790_79076


namespace NUMINAMATH_CALUDE_bounded_sequence_with_recurrence_is_constant_two_l790_79019

def is_bounded_sequence (a : ℕ → ℕ) : Prop :=
  ∃ M : ℕ, ∀ n, a n ≤ M

def satisfies_recurrence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = (a (n - 1) + a (n - 2)) / Nat.gcd (a (n - 1)) (a (n - 2))

theorem bounded_sequence_with_recurrence_is_constant_two (a : ℕ → ℕ) 
  (h_bounded : is_bounded_sequence a)
  (h_recurrence : satisfies_recurrence a) :
  ∀ n, a n = 2 :=
by sorry

end NUMINAMATH_CALUDE_bounded_sequence_with_recurrence_is_constant_two_l790_79019


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l790_79097

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo 1 2 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | b * x^2 + a * x + c < 0} = Set.Ioo (-2/3) 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l790_79097


namespace NUMINAMATH_CALUDE_intersection_condition_l790_79074

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem intersection_condition (a : ℝ) : A ∩ B a = B a → a = 0 ∨ a = 1/2 ∨ a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l790_79074


namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l790_79036

theorem multiplication_table_odd_fraction :
  let n : ℕ := 16
  let total_products : ℕ := n * n
  let odd_numbers : ℕ := (n + 1) / 2
  let odd_products : ℕ := odd_numbers * odd_numbers
  (odd_products : ℚ) / total_products = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l790_79036


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l790_79086

/-- Given a man's rowing speeds with and against a stream, calculates his rate in still water. -/
theorem mans_rate_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 20) 
  (h2 : speed_against_stream = 8) : 
  (speed_with_stream + speed_against_stream) / 2 = 14 := by
  sorry

#check mans_rate_in_still_water

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l790_79086


namespace NUMINAMATH_CALUDE_quadratic_factorization_l790_79007

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l790_79007


namespace NUMINAMATH_CALUDE_not_divisible_and_only_prime_l790_79022

theorem not_divisible_and_only_prime (n : ℕ) : 
  (n > 1 → ¬(n ∣ (2^n - 1))) ∧ 
  (n.Prime ∧ n^2 ∣ (2^n + 1) ↔ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_and_only_prime_l790_79022


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_theorem_l790_79069

/-- The sum of an arithmetic sequence with first term 3, common difference 4, and last term not exceeding 47 -/
def arithmetic_sequence_sum : ℕ → ℕ := λ n => n * (3 + (4 * n - 1)) / 2

/-- The number of terms in the sequence -/
def n : ℕ := 12

theorem arithmetic_sequence_sum_theorem :
  (∀ k : ℕ, k ≤ n → 3 + 4 * (k - 1) ≤ 47) ∧ 
  3 + 4 * (n - 1) = 47 ∧
  arithmetic_sequence_sum n = 300 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_theorem_l790_79069


namespace NUMINAMATH_CALUDE_sequence_properties_l790_79055

def S (n : ℕ+) : ℤ := -n^2 + 24*n

def a (n : ℕ+) : ℤ := -2*n + 25

theorem sequence_properties :
  (∀ n : ℕ+, S n - S (n-1) = a n) ∧
  (∀ n : ℕ+, n ≤ 12 → S n ≤ S 12) ∧
  (S 12 = 144) := by sorry

end NUMINAMATH_CALUDE_sequence_properties_l790_79055


namespace NUMINAMATH_CALUDE_sphere_surface_area_l790_79070

/-- The surface area of a sphere with diameter 9 inches is 81π square inches. -/
theorem sphere_surface_area (π : ℝ) (h : π > 0) : 
  let diameter : ℝ := 9
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * π * radius^2
  surface_area = 81 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l790_79070


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l790_79084

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l790_79084


namespace NUMINAMATH_CALUDE_f_extrema_l790_79030

noncomputable def f (x : ℝ) : ℝ := 1 + 3*x - x^3

theorem f_extrema :
  (∃ x : ℝ, f x = -1 ∧ ∀ y : ℝ, f y ≥ -1) ∧
  (∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l790_79030


namespace NUMINAMATH_CALUDE_goods_train_speed_l790_79052

/-- The speed of a goods train passing a man in another train -/
theorem goods_train_speed 
  (man_train_speed : ℝ) 
  (passing_time : ℝ) 
  (goods_train_length : ℝ) 
  (h1 : man_train_speed = 50) 
  (h2 : passing_time = 9 / 3600) 
  (h3 : goods_train_length = 0.28) : 
  ∃ (goods_train_speed : ℝ), goods_train_speed = 62 := by
sorry

end NUMINAMATH_CALUDE_goods_train_speed_l790_79052


namespace NUMINAMATH_CALUDE_calculation_proof_l790_79079

theorem calculation_proof : (8 * 5.4 - 0.6 * 10 / 1.2)^2 = 1459.24 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l790_79079


namespace NUMINAMATH_CALUDE_double_counted_integer_l790_79034

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem double_counted_integer (n : ℕ) (x : ℕ) :
  sum_of_first_n n + x = 5053 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_double_counted_integer_l790_79034


namespace NUMINAMATH_CALUDE_domino_count_for_0_to_12_l790_79091

/-- The number of tiles in a standard set of dominoes -/
def standard_domino_count : ℕ := 28

/-- The lowest value on a domino tile -/
def min_value : ℕ := 0

/-- The highest value on a domino tile in the new set -/
def max_value : ℕ := 12

/-- The number of tiles in a domino set with values from min_value to max_value -/
def domino_count (min : ℕ) (max : ℕ) : ℕ :=
  let n := max - min + 1
  (n * (n + 1)) / 2

theorem domino_count_for_0_to_12 :
  domino_count min_value max_value = 91 :=
sorry

end NUMINAMATH_CALUDE_domino_count_for_0_to_12_l790_79091


namespace NUMINAMATH_CALUDE_parabola_translation_l790_79071

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let p := Parabola.mk 2 0 0  -- y = 2x^2
  let p_translated := translate p 1 3  -- Translate 1 right, 3 up
  y = 2 * x^2 → y = 2 * (x - 1)^2 + 3 :=
by
  sorry

#check parabola_translation

end NUMINAMATH_CALUDE_parabola_translation_l790_79071


namespace NUMINAMATH_CALUDE_impossible_average_weight_problem_l790_79048

theorem impossible_average_weight_problem :
  ¬ ∃ (n : ℕ), n > 0 ∧ (n * 55 + 50) / (n + 1) = 50 := by
  sorry

end NUMINAMATH_CALUDE_impossible_average_weight_problem_l790_79048


namespace NUMINAMATH_CALUDE_rectangle_dimension_l790_79083

/-- A rectangle with vertices at (0, 0), (0, 6), (x, 6), and (x, 0) has a perimeter of 40 units. -/
def rectangle_perimeter (x : ℝ) : Prop :=
  x > 0 ∧ 2 * (x + 6) = 40

/-- The value of x for which the rectangle has a perimeter of 40 units is 14. -/
theorem rectangle_dimension : ∃ x : ℝ, rectangle_perimeter x ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_l790_79083


namespace NUMINAMATH_CALUDE_unsold_books_percentage_l790_79044

-- Define the initial stock and daily sales
def initial_stock : ℕ := 620
def daily_sales : List ℕ := [50, 82, 60, 48, 40]

-- Define the theorem
theorem unsold_books_percentage :
  let total_sold := daily_sales.sum
  let unsold := initial_stock - total_sold
  let percentage_unsold := (unsold : ℚ) / (initial_stock : ℚ) * 100
  ∃ ε > 0, abs (percentage_unsold - 54.84) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_unsold_books_percentage_l790_79044


namespace NUMINAMATH_CALUDE_solution_sets_l790_79020

def f (a x : ℝ) : ℝ := x^2 - 3*a*x + 2*a^2

theorem solution_sets :
  (∀ x, f 1 x ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2) ∧
  (∀ a, (a = 0 → ∀ x, ¬(f a x < 0)) ∧
        (a > 0 → ∀ x, f a x < 0 ↔ a < x ∧ x < 2*a) ∧
        (a < 0 → ∀ x, f a x < 0 ↔ 2*a < x ∧ x < a)) :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_l790_79020


namespace NUMINAMATH_CALUDE_marble_ratio_l790_79096

/-- Proves the ratio of Michael's marbles to Wolfgang and Ludo's combined marbles -/
theorem marble_ratio :
  let wolfgang_marbles : ℕ := 16
  let ludo_marbles : ℕ := wolfgang_marbles + wolfgang_marbles / 4
  let total_marbles : ℕ := 20 * 3
  let michael_marbles : ℕ := total_marbles - wolfgang_marbles - ludo_marbles
  let wolfgang_ludo_marbles : ℕ := wolfgang_marbles + ludo_marbles
  (michael_marbles : ℚ) / wolfgang_ludo_marbles = 2 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_marble_ratio_l790_79096


namespace NUMINAMATH_CALUDE_magnitude_of_sum_l790_79072

/-- Given two vectors a and b in ℝ², prove that the magnitude of a + 3b is 5√5 when a is parallel to b -/
theorem magnitude_of_sum (a b : ℝ × ℝ) (h_parallel : ∃ (k : ℝ), b = k • a) : 
  a.1 = 1 → a.2 = 2 → b.1 = -2 → 
  ‖(a.1 + 3 * b.1, a.2 + 3 * b.2)‖ = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_sum_l790_79072


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l790_79037

theorem quadratic_roots_sum_and_product (x₁ x₂ : ℝ) :
  x₁^2 - 3*x₁ + 1 = 0 → x₂^2 - 3*x₂ + 1 = 0 → x₁ + x₂ + x₁*x₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l790_79037


namespace NUMINAMATH_CALUDE_negation_of_all_exponential_are_monotonic_l790_79099

-- Define exponential function
def ExponentialFunction (f : ℝ → ℝ) : Prop := sorry

-- Define monotonic function
def MonotonicFunction (f : ℝ → ℝ) : Prop := sorry

-- Theorem statement
theorem negation_of_all_exponential_are_monotonic :
  (¬ ∀ f : ℝ → ℝ, ExponentialFunction f → MonotonicFunction f) ↔
  (∃ f : ℝ → ℝ, ExponentialFunction f ∧ ¬MonotonicFunction f) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_exponential_are_monotonic_l790_79099


namespace NUMINAMATH_CALUDE_fourth_term_is_eight_l790_79016

/-- An arithmetic progression with the given property -/
def ArithmeticProgression (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = S n + 1

theorem fourth_term_is_eight
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h : ArithmeticProgression a S) :
  a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_eight_l790_79016


namespace NUMINAMATH_CALUDE_inequality_solution_l790_79015

def solution_set : Set ℤ := {-3, 2}

def inequality (x : ℤ) : Prop :=
  (x^2 + 6*x + 8) * (x^2 - 4*x + 3) < 0

theorem inequality_solution :
  ∀ x : ℤ, inequality x ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l790_79015


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_half_l790_79062

theorem arctan_sum_equals_pi_half (a b : ℝ) (h1 : a = 1/3) (h2 : (a+1)*(b+1) = 3) :
  Real.arctan a + Real.arctan b = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_half_l790_79062


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_M_l790_79098

def M : Set ℤ := {0, 1}
def N : Set ℤ := {x | ∃ y, y = Real.sqrt (1 - x)}

theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_M_l790_79098


namespace NUMINAMATH_CALUDE_inequality_proof_l790_79017

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y + y * z + z * x = 3) : 
  (x + 3) / (y + z) + (y + 3) / (z + x) + (z + 3) / (x + y) + 3 ≥ 
  27 * ((Real.sqrt x + Real.sqrt y + Real.sqrt z)^2) / ((x + y + z)^3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l790_79017


namespace NUMINAMATH_CALUDE_existence_of_special_multiple_l790_79008

theorem existence_of_special_multiple : ∃ n : ℕ,
  (n % 2020 = 0) ∧
  (∀ d : Fin 10, ∃! pos : ℕ, 
    (n / 10^pos % 10 : Fin 10) = d) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_multiple_l790_79008


namespace NUMINAMATH_CALUDE_gift_contribution_l790_79051

theorem gift_contribution (a b c d e : ℝ) : 
  a + b + c + d + e = 120 →
  a = 2 * b →
  b = (1/3) * (c + d) →
  c = 2 * e →
  e = 12 := by sorry

end NUMINAMATH_CALUDE_gift_contribution_l790_79051


namespace NUMINAMATH_CALUDE_probability_divisor_of_twelve_l790_79024

def divisors_of_twelve : Finset ℕ := {1, 2, 3, 4, 6, 12}

theorem probability_divisor_of_twelve (die : Finset ℕ) 
  (h1 : die = Finset.range 12) 
  (h2 : die.card = 12) : 
  (divisors_of_twelve.card : ℚ) / (die.card : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_divisor_of_twelve_l790_79024


namespace NUMINAMATH_CALUDE_restaurant_group_children_l790_79026

/-- The number of children in a restaurant group -/
def num_children (num_adults : ℕ) (meal_cost : ℕ) (total_bill : ℕ) : ℕ :=
  (total_bill - num_adults * meal_cost) / meal_cost

theorem restaurant_group_children :
  num_children 2 3 21 = 5 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_group_children_l790_79026


namespace NUMINAMATH_CALUDE_chess_swimming_enrollment_percentage_l790_79029

theorem chess_swimming_enrollment_percentage 
  (total_students : ℕ) 
  (chess_percentage : ℚ) 
  (swimming_students : ℕ) 
  (h1 : total_students = 2000)
  (h2 : chess_percentage = 1/10)
  (h3 : swimming_students = 100) :
  (swimming_students : ℚ) / ((chess_percentage * total_students) : ℚ) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_chess_swimming_enrollment_percentage_l790_79029


namespace NUMINAMATH_CALUDE_good_numbers_exist_exist_good_sum_not_good_l790_79005

/-- A number is "good" if it can be expressed as a^2 + 161b^2 for some integers a and b -/
def is_good_number (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + 161 * b^2

theorem good_numbers_exist : is_good_number 100 ∧ is_good_number 2010 := by sorry

theorem exist_good_sum_not_good :
  ∃ x y : ℕ+, is_good_number (x^161 + y^161) ∧ ¬is_good_number (x + y) := by sorry

end NUMINAMATH_CALUDE_good_numbers_exist_exist_good_sum_not_good_l790_79005


namespace NUMINAMATH_CALUDE_pairball_play_time_l790_79032

theorem pairball_play_time (total_duration : ℕ) (num_children : ℕ) (children_per_game : ℕ) :
  total_duration = 120 →
  num_children = 6 →
  children_per_game = 2 →
  (total_duration * children_per_game) / num_children = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_pairball_play_time_l790_79032


namespace NUMINAMATH_CALUDE_apple_bag_weight_l790_79053

/-- Given a bag of apples costing 3.50 dollars, and knowing that 7 pounds of apples
    at the same rate would cost 4.9 dollars, prove that the bag contains 5 pounds of apples. -/
theorem apple_bag_weight (bag_cost : ℝ) (rate_pounds : ℝ) (rate_cost : ℝ) :
  bag_cost = 3.50 →
  rate_pounds = 7 →
  rate_cost = 4.9 →
  (rate_cost / rate_pounds) * (bag_cost / (rate_cost / rate_pounds)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_apple_bag_weight_l790_79053


namespace NUMINAMATH_CALUDE_triangle_area_l790_79028

/-- The area of a triangle with vertices A(2, 2), B(8, 2), and C(5, 11) is 27 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (8, 2)
  let C : ℝ × ℝ := (5, 11)
  let area := (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))
  area = 27 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l790_79028


namespace NUMINAMATH_CALUDE_trinomial_binomial_product_l790_79080

theorem trinomial_binomial_product : 
  ∀ x : ℝ, (2 * x^2 + 3 * x + 1) * (x - 4) = 2 * x^3 - 5 * x^2 - 11 * x - 4 := by
  sorry

end NUMINAMATH_CALUDE_trinomial_binomial_product_l790_79080


namespace NUMINAMATH_CALUDE_range_of_t_l790_79067

noncomputable def f (x : ℝ) : ℝ := x^3 + 1

noncomputable def g (t : ℝ) (x : ℝ) : ℝ := 2 * (Real.log x / Real.log 2)^2 - 2 * (Real.log x / Real.log 2) + t - 4

noncomputable def F (t : ℝ) (x : ℝ) : ℝ := f (g t x) - 1

theorem range_of_t (t : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x ∈ Set.Icc 1 (2 * Real.sqrt 2) ∧ y ∈ Set.Icc 1 (2 * Real.sqrt 2) ∧
    F t x = 0 ∧ F t y = 0 ∧
    ∀ z ∈ Set.Icc 1 (2 * Real.sqrt 2), F t z = 0 → z = x ∨ z = y) →
  t ∈ Set.Icc 4 (9/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_t_l790_79067


namespace NUMINAMATH_CALUDE_cone_height_from_circular_sector_l790_79001

theorem cone_height_from_circular_sector (r : ℝ) (h : r = 8) :
  let sector_arc_length := 2 * π * r / 4
  let cone_base_radius := sector_arc_length / (2 * π)
  let cone_slant_height := r
  cone_slant_height ^ 2 - cone_base_radius ^ 2 = (2 * Real.sqrt 15) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_from_circular_sector_l790_79001


namespace NUMINAMATH_CALUDE_triangle_area_l790_79068

/-- Given a triangle ABC with sides a, b, and c, prove that its area is 3√2/4
    when sinA = √3/3 and b² + c² - a² = 6 -/
theorem triangle_area (a b c : ℝ) (h1 : Real.sin A = Real.sqrt 3 / 3) 
  (h2 : b^2 + c^2 - a^2 = 6) : 
  (1/2 : ℝ) * b * c * Real.sin A = 3 * Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l790_79068


namespace NUMINAMATH_CALUDE_min_three_digit_quotient_l790_79042

theorem min_three_digit_quotient :
  ∃ (a b c : ℕ), 
    a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
    (∀ (x y z : ℕ), x ≤ 9 → y ≤ 9 → z ≤ 9 →
      (100 * a + 10 * b + c : ℚ) / (a + b + c : ℚ) ≤ (100 * x + 10 * y + z : ℚ) / (x + y + z : ℚ)) ∧
    (100 * a + 10 * b + c : ℚ) / (a + b + c : ℚ) = 50.5 := by
  sorry

end NUMINAMATH_CALUDE_min_three_digit_quotient_l790_79042


namespace NUMINAMATH_CALUDE_recipe_flour_calculation_l790_79014

theorem recipe_flour_calculation :
  let full_recipe : ℚ := 7 + 3/4
  let one_third_recipe : ℚ := (1/3) * full_recipe
  one_third_recipe = 2 + 7/12 := by sorry

end NUMINAMATH_CALUDE_recipe_flour_calculation_l790_79014


namespace NUMINAMATH_CALUDE_same_color_shoe_probability_l790_79013

/-- The number of pairs of shoes -/
def num_pairs : ℕ := 7

/-- The total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes to be selected -/
def selected_shoes : ℕ := 2

/-- The probability of selecting two shoes of the same color -/
def same_color_prob : ℚ := 1 / 13

theorem same_color_shoe_probability :
  (num_pairs : ℚ) / (total_shoes.choose selected_shoes) = same_color_prob :=
sorry

end NUMINAMATH_CALUDE_same_color_shoe_probability_l790_79013


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l790_79089

theorem quadratic_form_equivalence (y : ℝ) : y^2 - 8*y = (y - 4)^2 - 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l790_79089


namespace NUMINAMATH_CALUDE_age_difference_l790_79081

def age_problem (a b c : ℕ) : Prop :=
  b = 2 * c ∧ a + b + c = 27 ∧ b = 10

theorem age_difference (a b c : ℕ) (h : age_problem a b c) : a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l790_79081


namespace NUMINAMATH_CALUDE_pell_like_equation_solution_l790_79012

theorem pell_like_equation_solution (n : ℤ) :
  let x := (1/4) * ((1+Real.sqrt 2)^(2*n+1) + (1-Real.sqrt 2)^(2*n+1) - 2)
  let y := (1/(2*Real.sqrt 2)) * ((1+Real.sqrt 2)^(2*n+1) - (1-Real.sqrt 2)^(2*n+1))
  (x^2 + (x+1)^2 = y^2) ∧
  (∀ (a b : ℝ), a^2 + (a+1)^2 = b^2 → ∃ (m : ℤ), 
    a = (1/4) * ((1+Real.sqrt 2)^(2*m+1) + (1-Real.sqrt 2)^(2*m+1) - 2) ∧
    b = (1/(2*Real.sqrt 2)) * ((1+Real.sqrt 2)^(2*m+1) - (1-Real.sqrt 2)^(2*m+1)))
  := by sorry

end NUMINAMATH_CALUDE_pell_like_equation_solution_l790_79012


namespace NUMINAMATH_CALUDE_symmetrical_function_is_two_minus_ln_l790_79045

/-- A function whose graph is symmetrical to y = e^(2-x) with respect to y = x -/
def SymmetricalToExp (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ f y = x

/-- The main theorem stating that f(x) = 2 - ln(x) -/
theorem symmetrical_function_is_two_minus_ln (f : ℝ → ℝ) 
    (h : SymmetricalToExp f) : 
    ∀ x > 0, f x = 2 - Real.log x := by
  sorry

end NUMINAMATH_CALUDE_symmetrical_function_is_two_minus_ln_l790_79045


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_46_l790_79006

theorem consecutive_integers_sum_46 :
  ∃ (x y z w : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
  y = x + 1 ∧ z = y + 1 ∧ w = z + 1 ∧
  x + y + z + w = 46 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_46_l790_79006


namespace NUMINAMATH_CALUDE_no_six_numbers_exist_l790_79057

/-- Represents a six-digit number composed of digits 1 to 6 without repetitions -/
def SixDigitNumber := Fin 6 → Fin 6

/-- Represents a three-digit number composed of digits 1 to 6 without repetitions -/
def ThreeDigitNumber := Fin 3 → Fin 6

/-- Checks if a ThreeDigitNumber can be obtained from a SixDigitNumber by deleting three digits -/
def canBeObtained (six : SixDigitNumber) (three : ThreeDigitNumber) : Prop :=
  ∃ (i j k : Fin 6), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
    (∀ m : Fin 3, three m = six (if m < i then m else if m < j then m + 1 else m + 2))

/-- The main theorem stating that the required set of six numbers does not exist -/
theorem no_six_numbers_exist : 
  ¬ ∃ (numbers : Fin 6 → SixDigitNumber),
    (∀ i : Fin 6, Function.Injective (numbers i)) ∧
    (∀ three : ThreeDigitNumber, Function.Injective three → 
      ∃ (i : Fin 6), canBeObtained (numbers i) three) :=
by sorry


end NUMINAMATH_CALUDE_no_six_numbers_exist_l790_79057


namespace NUMINAMATH_CALUDE_polynomial_factorization_l790_79075

theorem polynomial_factorization :
  ∀ (x y a b : ℝ),
    (12 * x^3 * y - 3 * x * y^2 = 3 * x * y * (4 * x^2 - y)) ∧
    (x - 9 * x^3 = x * (1 + 3 * x) * (1 - 3 * x)) ∧
    (3 * a^2 - 12 * a * b * (a - b) = 3 * (a - 2 * b)^2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l790_79075


namespace NUMINAMATH_CALUDE_students_playing_soccer_l790_79073

-- Define the total number of students
def total_students : ℕ := 450

-- Define the number of boys
def boys : ℕ := 320

-- Define the percentage of boys playing soccer
def boys_soccer_percentage : ℚ := 86 / 100

-- Define the number of girls not playing soccer
def girls_not_soccer : ℕ := 95

-- Theorem to prove
theorem students_playing_soccer : 
  ∃ (soccer_players : ℕ), 
    soccer_players = 250 ∧ 
    soccer_players ≤ total_students ∧
    (total_students - boys) - girls_not_soccer = 
      (1 - boys_soccer_percentage) * soccer_players :=
sorry

end NUMINAMATH_CALUDE_students_playing_soccer_l790_79073


namespace NUMINAMATH_CALUDE_john_skateboard_distance_l790_79002

/-- Represents the distance John traveled in miles -/
structure JohnTrip where
  skateboard_to_park : ℕ
  walk_to_park : ℕ

/-- Calculates the total distance John skateboarded -/
def total_skateboard_distance (trip : JohnTrip) : ℕ :=
  2 * trip.skateboard_to_park + trip.skateboard_to_park

theorem john_skateboard_distance :
  ∀ (trip : JohnTrip),
    trip.skateboard_to_park = 10 ∧ trip.walk_to_park = 4 →
    total_skateboard_distance trip = 24 :=
by
  sorry

#check john_skateboard_distance

end NUMINAMATH_CALUDE_john_skateboard_distance_l790_79002


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l790_79060

theorem right_triangle_hypotenuse : 
  ∀ (short_leg long_leg hypotenuse : ℝ),
  short_leg > 0 →
  long_leg = 2 * short_leg + 3 →
  (1 / 2) * short_leg * long_leg = 84 →
  short_leg ^ 2 + long_leg ^ 2 = hypotenuse ^ 2 →
  hypotenuse = Real.sqrt 261 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l790_79060


namespace NUMINAMATH_CALUDE_absolute_value_inequalities_l790_79092

theorem absolute_value_inequalities (x y : ℝ) : 
  (abs (x + y) ≤ abs x + abs y) ∧ 
  (abs (x - y) ≥ abs x - abs y) ∧ 
  (abs (x - y) ≥ abs (abs x - abs y)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequalities_l790_79092


namespace NUMINAMATH_CALUDE_bucket_weight_l790_79018

theorem bucket_weight (p q : ℝ) 
  (h1 : ∃ x y : ℝ, x + 3/4 * y = p ∧ x + 1/3 * y = q) : 
  ∃ w : ℝ, w = (5*q - p)/5 ∧ 
  ∀ x y : ℝ, (x + 3/4 * y = p ∧ x + 1/3 * y = q) → x + 1/4 * y = w :=
sorry

end NUMINAMATH_CALUDE_bucket_weight_l790_79018


namespace NUMINAMATH_CALUDE_swamp_ecosystem_l790_79064

/-- In a swamp ecosystem, prove that each gharial needs to eat 15 fish per day given the following conditions:
  * Each frog eats 30 flies per day
  * Each fish eats 8 frogs per day
  * There are 9 gharials in the swamp
  * 32,400 flies are eaten every day
-/
theorem swamp_ecosystem (flies_per_frog : ℕ) (frogs_per_fish : ℕ) (num_gharials : ℕ) (total_flies : ℕ)
  (h1 : flies_per_frog = 30)
  (h2 : frogs_per_fish = 8)
  (h3 : num_gharials = 9)
  (h4 : total_flies = 32400) :
  total_flies / (flies_per_frog * frogs_per_fish * num_gharials) = 15 := by
  sorry

end NUMINAMATH_CALUDE_swamp_ecosystem_l790_79064


namespace NUMINAMATH_CALUDE_sum_interior_angles_hexagon_l790_79059

/-- The sum of interior angles of a regular hexagon is 720 degrees. -/
theorem sum_interior_angles_hexagon :
  ∀ (n : ℕ) (sum : ℝ),
  n = 6 →
  sum = (n - 2) * 180 →
  sum = 720 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_hexagon_l790_79059


namespace NUMINAMATH_CALUDE_total_animals_is_100_l790_79058

/-- Given the number of rabbits, calculates the total number of chickens, ducks, and rabbits. -/
def total_animals (num_rabbits : ℕ) : ℕ :=
  let num_ducks := num_rabbits + 12
  let num_chickens := 5 * num_ducks
  num_chickens + num_ducks + num_rabbits

/-- Theorem stating that given 4 rabbits, the total number of animals is 100. -/
theorem total_animals_is_100 : total_animals 4 = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_is_100_l790_79058


namespace NUMINAMATH_CALUDE_fish_distribution_l790_79066

theorem fish_distribution (bodies_of_water : ℕ) (total_fish : ℕ) 
  (h1 : bodies_of_water = 6) 
  (h2 : total_fish = 1050) : 
  total_fish / bodies_of_water = 175 := by
sorry

end NUMINAMATH_CALUDE_fish_distribution_l790_79066


namespace NUMINAMATH_CALUDE_polynomial_property_l790_79043

def P (a b c : ℝ) (x : ℝ) : ℝ := 2*x^3 + a*x^2 + b*x + c

theorem polynomial_property (a b c : ℝ) :
  P a b c 0 = 8 →
  (∃ m : ℝ, m = (-(c / 2)) ∧ 
             m = -((a / 2) / 3) ∧ 
             m = 2 + a + b + c) →
  b = -38 := by sorry

end NUMINAMATH_CALUDE_polynomial_property_l790_79043


namespace NUMINAMATH_CALUDE_circle_diameter_l790_79065

/-- Given a circle with area M and circumference N, if M/N = 15, then the diameter is 60 -/
theorem circle_diameter (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 15) :
  let r := N / (2 * Real.pi)
  let d := 2 * r
  d = 60 := by sorry

end NUMINAMATH_CALUDE_circle_diameter_l790_79065


namespace NUMINAMATH_CALUDE_f_1_eq_0_f_increasing_f_inequality_solution_l790_79009

noncomputable section

variable (f : ℝ → ℝ)

axiom domain : ∀ x, x > 0 → f x ≠ 0 → True

axiom f_4 : f 4 = 1

axiom f_product : ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂

axiom f_neg_on_unit : ∀ x, 0 < x → x < 1 → f x < 0

theorem f_1_eq_0 : f 1 = 0 := by sorry

theorem f_increasing : ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by sorry

theorem f_inequality_solution : 
  ∀ x, x > 0 → (f (3 * x + 1) + f (2 * x - 6) ≤ 3 ↔ 3 < x ∧ x ≤ 5) := by sorry

end

end NUMINAMATH_CALUDE_f_1_eq_0_f_increasing_f_inequality_solution_l790_79009


namespace NUMINAMATH_CALUDE_initial_ratio_proof_l790_79088

/-- Represents the ratio of two quantities -/
structure Ratio :=
  (numerator : ℚ)
  (denominator : ℚ)

/-- Represents the contents of a bucket with two liquids -/
structure Bucket :=
  (liquidA : ℚ)
  (liquidB : ℚ)

def replace_mixture (b : Bucket) (amount : ℚ) : Bucket :=
  { liquidA := b.liquidA,
    liquidB := b.liquidB + amount }

def ratio (b : Bucket) : Ratio :=
  { numerator := b.liquidA,
    denominator := b.liquidB }

theorem initial_ratio_proof (initial : Bucket) 
  (h1 : initial.liquidA = 21)
  (h2 : ratio (replace_mixture initial 9) = Ratio.mk 7 9) :
  ratio initial = Ratio.mk 7 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_proof_l790_79088


namespace NUMINAMATH_CALUDE_total_points_l790_79033

def game1_mike : ℕ := 5
def game1_john : ℕ := game1_mike + 2

def game2_mike : ℕ := 7
def game2_john : ℕ := game2_mike - 3

def game3_mike : ℕ := 10
def game3_john : ℕ := game3_mike / 2

def game4_mike : ℕ := 12
def game4_john : ℕ := game4_mike * 2

def game5_mike : ℕ := 6
def game5_john : ℕ := game5_mike

def game6_john : ℕ := 8
def game6_mike : ℕ := game6_john + 4

def mike_total : ℕ := game1_mike + game2_mike + game3_mike + game4_mike + game5_mike + game6_mike
def john_total : ℕ := game1_john + game2_john + game3_john + game4_john + game5_john + game6_john

theorem total_points : mike_total + john_total = 106 := by
  sorry

end NUMINAMATH_CALUDE_total_points_l790_79033


namespace NUMINAMATH_CALUDE_parabola_vertex_l790_79085

/-- The parabola defined by the equation y = 2(x-3)^2 - 7 -/
def parabola (x y : ℝ) : Prop := y = 2 * (x - 3)^2 - 7

/-- The vertex of a parabola -/
structure Vertex where
  x : ℝ
  y : ℝ

/-- Theorem: The vertex of the parabola y = 2(x-3)^2 - 7 is at the point (3, -7) -/
theorem parabola_vertex : 
  ∃ (v : Vertex), v.x = 3 ∧ v.y = -7 ∧ 
  (∀ (x y : ℝ), parabola x y → (x - v.x)^2 ≤ (y - v.y) / 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l790_79085


namespace NUMINAMATH_CALUDE_product_sequence_sum_l790_79061

theorem product_sequence_sum (a b : ℕ) (h1 : a / 3 = 12) (h2 : b = a - 1) : a + b = 71 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_sum_l790_79061


namespace NUMINAMATH_CALUDE_salary_difference_l790_79077

theorem salary_difference (a b : ℝ) (h : b = 1.25 * a) :
  (b - a) / b * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_salary_difference_l790_79077


namespace NUMINAMATH_CALUDE_probability_ratio_l790_79011

def total_slips : ℕ := 50
def numbers_range : ℕ := 10
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 5

def probability_same_number (total : ℕ) (range : ℕ) (per_num : ℕ) (drawn : ℕ) : ℚ :=
  (range : ℚ) / Nat.choose total drawn

def probability_three_and_two (total : ℕ) (range : ℕ) (per_num : ℕ) (drawn : ℕ) : ℚ :=
  (Nat.choose range 2 * Nat.choose per_num 3 * Nat.choose per_num 2 : ℚ) / Nat.choose total drawn

theorem probability_ratio :
  (probability_three_and_two total_slips numbers_range slips_per_number drawn_slips) /
  (probability_same_number total_slips numbers_range slips_per_number drawn_slips) = 450 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l790_79011


namespace NUMINAMATH_CALUDE_linda_needs_two_more_batches_l790_79082

/-- The number of additional batches of cookies Linda needs to bake --/
def additional_batches (classmates : ℕ) (cookies_per_student : ℕ) (dozens_per_batch : ℕ) 
  (chocolate_chip_batches : ℕ) (oatmeal_raisin_batches : ℕ) : ℕ :=
  let total_cookies_needed := classmates * cookies_per_student
  let cookies_per_batch := dozens_per_batch * 12
  let cookies_already_made := (chocolate_chip_batches + oatmeal_raisin_batches) * cookies_per_batch
  let cookies_still_needed := total_cookies_needed - cookies_already_made
  (cookies_still_needed + cookies_per_batch - 1) / cookies_per_batch

/-- Theorem stating that Linda needs to bake 2 more batches of cookies --/
theorem linda_needs_two_more_batches :
  additional_batches 24 10 4 2 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_linda_needs_two_more_batches_l790_79082


namespace NUMINAMATH_CALUDE_square_mod_32_l790_79046

theorem square_mod_32 (n : ℕ) (h : n % 8 = 6) : n^2 % 32 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_mod_32_l790_79046


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l790_79041

theorem logarithm_expression_equality : 
  (Real.log (27^(1/2)) + Real.log 8 - 3 * Real.log (10^(1/2))) / Real.log 1.2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l790_79041


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l790_79095

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (1 - Complex.I) / (1 + 3 * Complex.I)
  Complex.im z = -2/5 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l790_79095


namespace NUMINAMATH_CALUDE_sum_of_dimensions_l790_79025

/-- A rectangular box with dimensions A, B, and C, where AB = 40, AC = 90, and BC = 360 -/
structure RectangularBox where
  A : ℝ
  B : ℝ
  C : ℝ
  ab_area : A * B = 40
  ac_area : A * C = 90
  bc_area : B * C = 360

/-- The sum of dimensions A, B, and C of the rectangular box is 45 -/
theorem sum_of_dimensions (box : RectangularBox) : box.A + box.B + box.C = 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_dimensions_l790_79025
