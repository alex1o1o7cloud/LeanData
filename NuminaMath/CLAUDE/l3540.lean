import Mathlib

namespace NUMINAMATH_CALUDE_unique_b_value_l3540_354070

/-- The configuration of a circle and parabola with specific intersection properties -/
structure CircleParabolaConfig where
  b : ℝ
  circle_center : ℝ × ℝ
  parabola : ℝ → ℝ
  line : ℝ → ℝ
  intersect_origin : Bool
  intersect_line : Bool

/-- The theorem stating the unique value of b for the given configuration -/
theorem unique_b_value (config : CircleParabolaConfig) : 
  config.parabola = (λ x => (12/5) * x^2) →
  config.line = (λ x => (12/5) * x + config.b) →
  config.circle_center.2 = config.b →
  config.intersect_origin = true →
  config.intersect_line = true →
  config.b = 169/60 := by
  sorry

#check unique_b_value

end NUMINAMATH_CALUDE_unique_b_value_l3540_354070


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3540_354011

def is_valid_number (n : ℕ) : Prop :=
  ∃ k : ℕ, 
    n = 5 * 10^(k-1) + (n % 10^(k-1)) ∧ 
    10 * (n % 10^(k-1)) + 5 = n / 4

theorem smallest_valid_number : 
  is_valid_number 512820 ∧ 
  ∀ m : ℕ, m < 512820 → ¬(is_valid_number m) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3540_354011


namespace NUMINAMATH_CALUDE_square_triangle_equal_perimeter_l3540_354020

theorem square_triangle_equal_perimeter (x : ℝ) : 
  4 * (x + 2) = 3 * (2 * x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_perimeter_l3540_354020


namespace NUMINAMATH_CALUDE_springfield_population_difference_l3540_354010

/-- The population difference between two cities given the population of one city and their total population -/
def population_difference (population_springfield : ℕ) (total_population : ℕ) : ℕ :=
  population_springfield - (total_population - population_springfield)

/-- Theorem stating that the population difference between Springfield and the other city is 119,666 -/
theorem springfield_population_difference :
  population_difference 482653 845640 = 119666 := by
  sorry

end NUMINAMATH_CALUDE_springfield_population_difference_l3540_354010


namespace NUMINAMATH_CALUDE_problem_solution_l3540_354039

theorem problem_solution : ∀ M N X : ℕ,
  M = 2098 / 2 →
  N = M * 2 →
  X = M + N →
  X = 3147 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3540_354039


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l3540_354054

theorem abs_inequality_equivalence (x : ℝ) : 
  (1 ≤ |x + 3| ∧ |x + 3| ≤ 4) ↔ ((-7 ≤ x ∧ x ≤ -4) ∨ (-2 ≤ x ∧ x ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l3540_354054


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3540_354082

-- Define the original expression
def original_expression := (4 : ℚ) / (3 * (7 : ℚ)^(1/4))

-- Define the rationalized expression
def rationalized_expression := (4 * (343 : ℚ)^(1/4)) / 21

-- State the theorem
theorem rationalize_denominator :
  original_expression = rationalized_expression ∧
  ¬ (∃ (p : ℕ), Prime p ∧ (343 : ℕ) % p^4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3540_354082


namespace NUMINAMATH_CALUDE_probability_is_one_twelfth_l3540_354096

/-- The number of vertices in a regular decagon -/
def decagon_vertices : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The total number of possible triangles formed by choosing 3 vertices from a decagon -/
def total_triangles : ℕ := Nat.choose decagon_vertices triangle_vertices

/-- The number of triangles with at least two sides that are also sides of the decagon -/
def favorable_triangles : ℕ := decagon_vertices

/-- The probability of forming a triangle with at least two sides that are also sides of the decagon -/
def probability : ℚ := favorable_triangles / total_triangles

/-- Theorem stating the probability is 1/12 -/
theorem probability_is_one_twelfth : probability = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_twelfth_l3540_354096


namespace NUMINAMATH_CALUDE_glued_cubes_faces_l3540_354004

/-- The number of faces of a cube -/
def cube_faces : ℕ := 6

/-- The number of new faces contributed by each glued cube -/
def new_faces_per_cube : ℕ := 5

/-- The number of faces in the resulting solid when a cube is glued to each face of an original cube -/
def resulting_solid_faces : ℕ := cube_faces + cube_faces * new_faces_per_cube

theorem glued_cubes_faces : resulting_solid_faces = 36 := by
  sorry

end NUMINAMATH_CALUDE_glued_cubes_faces_l3540_354004


namespace NUMINAMATH_CALUDE_total_selling_price_theorem_l3540_354005

def calculate_selling_price (cost : ℝ) (profit_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  let price_before_discount := cost * (1 + profit_percent)
  price_before_discount * (1 - discount_percent)

theorem total_selling_price_theorem :
  let item1 := calculate_selling_price 192 0.25 0.10
  let item2 := calculate_selling_price 350 0.15 0.05
  let item3 := calculate_selling_price 500 0.30 0.15
  item1 + item2 + item3 = 1150.875 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_theorem_l3540_354005


namespace NUMINAMATH_CALUDE_max_points_for_one_participant_l3540_354036

theorem max_points_for_one_participant 
  (n : ℕ) 
  (avg : ℚ) 
  (min_points : ℕ) 
  (h1 : n = 50) 
  (h2 : avg = 8) 
  (h3 : min_points = 2) 
  (h4 : ∀ p : ℕ, p ≤ n → p ≥ min_points) : 
  ∃ max_points : ℕ, max_points = 302 ∧ 
  ∀ p : ℕ, p ≤ n → p ≤ max_points := by
sorry


end NUMINAMATH_CALUDE_max_points_for_one_participant_l3540_354036


namespace NUMINAMATH_CALUDE_grid_paths_6_4_l3540_354087

/-- The number of paths on a grid from (0,0) to (m,n) using exactly m+n steps -/
def grid_paths (m n : ℕ) : ℕ := Nat.choose (m + n) m

theorem grid_paths_6_4 : grid_paths 6 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_grid_paths_6_4_l3540_354087


namespace NUMINAMATH_CALUDE_repeating_six_equals_two_thirds_l3540_354064

/-- The decimal representation of a number with infinitely repeating 6 after the decimal point -/
def repeating_six : ℚ := sorry

/-- Theorem stating that the repeating decimal 0.666... is equal to 2/3 -/
theorem repeating_six_equals_two_thirds : repeating_six = 2/3 := by sorry

end NUMINAMATH_CALUDE_repeating_six_equals_two_thirds_l3540_354064


namespace NUMINAMATH_CALUDE_sum_inequality_l3540_354089

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a * b) / (a + b) + (b * c) / (b + c) + (c * a) / (c + a) + 
    (1 / 2) * ((a * b) / c + (b * c) / a + (c * a) / b) := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3540_354089


namespace NUMINAMATH_CALUDE_parabola_b_value_l3540_354095

/-- Given a parabola y = ax^2 + bx + c with vertex (p, p) and y-intercept (0, -p), where p ≠ 0, 
    prove that b = 4. -/
theorem parabola_b_value (a b c p : ℝ) (hp : p ≠ 0) 
  (h_vertex : ∀ x, a * x^2 + b * x + c = a * (x - p)^2 + p)
  (h_y_intercept : c = -p) : b = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_b_value_l3540_354095


namespace NUMINAMATH_CALUDE_intersection_point_unique_l3540_354022

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-3/5, -4/5)

/-- First line equation: y = 3x + 1 -/
def line1 (x y : ℚ) : Prop := y = 3 * x + 1

/-- Second line equation: y + 5 = -7x -/
def line2 (x y : ℚ) : Prop := y + 5 = -7 * x

theorem intersection_point_unique :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → (x', y') = (x, y) := by sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l3540_354022


namespace NUMINAMATH_CALUDE_mary_anne_sparkling_water_cost_l3540_354008

/-- The annual cost of Mary Anne's sparkling water consumption -/
def annual_sparkling_water_cost (daily_consumption : ℚ) (bottle_cost : ℚ) : ℚ :=
  (365 : ℚ) * daily_consumption * bottle_cost

/-- Theorem: Mary Anne's annual sparkling water cost is $146.00 -/
theorem mary_anne_sparkling_water_cost :
  annual_sparkling_water_cost (1/5) 2 = 146 := by
  sorry

end NUMINAMATH_CALUDE_mary_anne_sparkling_water_cost_l3540_354008


namespace NUMINAMATH_CALUDE_no_universal_rational_compact_cover_l3540_354075

theorem no_universal_rational_compact_cover :
  ¬ (∃ (A : ℕ → Set ℚ), 
    (∀ n, IsCompact (A n)) ∧ 
    (∀ K : Set ℚ, IsCompact K → ∃ n, K ⊆ A n)) := by
  sorry

end NUMINAMATH_CALUDE_no_universal_rational_compact_cover_l3540_354075


namespace NUMINAMATH_CALUDE_unique_prime_divisor_l3540_354038

theorem unique_prime_divisor (n : ℕ) (hn : n > 1) :
  ∀ k ∈ Finset.range n,
    ∃ p : ℕ, Nat.Prime p ∧ 
      (p ∣ (n.factorial + k + 1)) ∧
      (∀ j ∈ Finset.range n, j ≠ k → ¬(p ∣ (n.factorial + j + 1))) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_divisor_l3540_354038


namespace NUMINAMATH_CALUDE_sum_fourth_power_ge_two_min_sum_cube_and_reciprocal_cube_min_sum_cube_and_reciprocal_cube_equality_l3540_354098

-- Part I
theorem sum_fourth_power_ge_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  a^4 + b^4 ≥ 2 := by sorry

-- Part II
theorem min_sum_cube_and_reciprocal_cube (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 + (1/a + 1/b + 1/c)^3 ≥ 18 := by sorry

theorem min_sum_cube_and_reciprocal_cube_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 + (1/a + 1/b + 1/c)^3 = 18 ↔ a = (3 : ℝ)^(1/3) ∧ b = (3 : ℝ)^(1/3) ∧ c = (3 : ℝ)^(1/3) := by sorry

end NUMINAMATH_CALUDE_sum_fourth_power_ge_two_min_sum_cube_and_reciprocal_cube_min_sum_cube_and_reciprocal_cube_equality_l3540_354098


namespace NUMINAMATH_CALUDE_track_length_is_50_l3540_354060

/-- Calculates the length of a running track given weekly distance, days per week, and loops per day -/
def track_length (weekly_distance : ℕ) (days_per_week : ℕ) (loops_per_day : ℕ) : ℕ :=
  weekly_distance / (days_per_week * loops_per_day)

/-- Proves that given the specified conditions, the track length is 50 meters -/
theorem track_length_is_50 : 
  track_length 3500 7 10 = 50 := by
  sorry

#eval track_length 3500 7 10

end NUMINAMATH_CALUDE_track_length_is_50_l3540_354060


namespace NUMINAMATH_CALUDE_even_sum_sufficient_not_necessary_l3540_354092

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Define the sum of two functions
def SumFunc (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f x + g x

theorem even_sum_sufficient_not_necessary :
  (∀ f g : ℝ → ℝ, IsEven f ∧ IsEven g → IsEven (SumFunc f g)) ∧ 
  (∃ f g : ℝ → ℝ, IsEven (SumFunc f g) ∧ ¬(IsEven f) ∧ ¬(IsEven g)) :=
sorry

end NUMINAMATH_CALUDE_even_sum_sufficient_not_necessary_l3540_354092


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_formula_l3540_354042

/-- The sum of the first k+1 terms of an arithmetic series with first term k^2 + 2 and common difference 2 -/
def arithmetic_series_sum (k : ℕ) : ℕ := sorry

/-- The first term of the arithmetic series -/
def first_term (k : ℕ) : ℕ := k^2 + 2

/-- The common difference of the arithmetic series -/
def common_difference : ℕ := 2

/-- The number of terms in the series -/
def num_terms (k : ℕ) : ℕ := k + 1

theorem arithmetic_series_sum_formula (k : ℕ) :
  arithmetic_series_sum k = k^3 + 2*k^2 + 3*k + 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_formula_l3540_354042


namespace NUMINAMATH_CALUDE_functional_equation_result_l3540_354033

theorem functional_equation_result (g : ℝ → ℝ) 
  (h₁ : ∀ c d : ℝ, c^2 * g d = d^2 * g c) 
  (h₂ : g 4 ≠ 0) : 
  (g 7 - g 3) / g 4 = 5/2 := by sorry

end NUMINAMATH_CALUDE_functional_equation_result_l3540_354033


namespace NUMINAMATH_CALUDE_all_xoons_are_zeefs_and_yamps_l3540_354057

-- Define the types for our sets
variable (U : Type) -- Universe set
variable (Zeef Yamp Xoon Woon : Set U)

-- Define the given conditions
variable (h1 : Zeef ⊆ Yamp)
variable (h2 : Xoon ⊆ Yamp)
variable (h3 : Woon ⊆ Zeef)
variable (h4 : Xoon ⊆ Woon)

-- Theorem to prove
theorem all_xoons_are_zeefs_and_yamps :
  Xoon ⊆ Zeef ∩ Yamp :=
by sorry

end NUMINAMATH_CALUDE_all_xoons_are_zeefs_and_yamps_l3540_354057


namespace NUMINAMATH_CALUDE_max_value_x_plus_2y_l3540_354078

theorem max_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2*x^2 + 8*y^2 + x*y = 2) : x + 2*y ≤ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2y_l3540_354078


namespace NUMINAMATH_CALUDE_max_value_of_sin_cos_combination_l3540_354000

theorem max_value_of_sin_cos_combination :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin x + 4 * Real.cos x
  ∃ M : ℝ, M = 5 ∧ ∀ x : ℝ, f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sin_cos_combination_l3540_354000


namespace NUMINAMATH_CALUDE_fraction_equality_l3540_354062

theorem fraction_equality (a b : ℝ) (h : (1/a + 1/b) / (1/a - 1/b) = 1001) :
  (a + b) / (a - b) = -1001 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3540_354062


namespace NUMINAMATH_CALUDE_fraction_simplification_l3540_354021

theorem fraction_simplification (a b : ℝ) (h : a ≠ 0) :
  (a^2 + 2*a*b + b^2) / (a^2 + a*b) = (a + b) / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3540_354021


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l3540_354049

theorem smallest_number_of_eggs (total_containers : ℕ) (filled_containers : ℕ) : 
  total_containers > 10 →
  filled_containers = total_containers - 3 →
  15 * filled_containers + 14 * 3 > 150 →
  15 * filled_containers + 14 * 3 ≤ 15 * (filled_containers + 1) + 14 * 3 - 3 →
  15 * filled_containers + 14 * 3 = 162 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l3540_354049


namespace NUMINAMATH_CALUDE_greatest_number_in_set_l3540_354045

/-- A set of consecutive multiples of 2 -/
def ConsecutiveMultiplesOf2 (s : Set ℕ) : Prop :=
  ∃ start : ℕ, ∀ n ∈ s, ∃ k : ℕ, n = start + 2 * k

theorem greatest_number_in_set (s : Set ℕ) 
  (h1 : ConsecutiveMultiplesOf2 s)
  (h2 : Fintype s)
  (h3 : Fintype.card s = 50)
  (h4 : 56 ∈ s)
  (h5 : ∀ n ∈ s, n ≥ 56) :
  ∃ m ∈ s, m = 154 ∧ ∀ n ∈ s, n ≤ m :=
sorry

end NUMINAMATH_CALUDE_greatest_number_in_set_l3540_354045


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l3540_354019

theorem easter_egg_hunt (bonnie george cheryl kevin : ℕ) : 
  bonnie = 13 →
  george = 9 →
  cheryl = 56 →
  cheryl = bonnie + george + kevin + 29 →
  kevin = 5 := by
sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l3540_354019


namespace NUMINAMATH_CALUDE_range_of_m_l3540_354091

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of f being increasing on [-2, 2]
def is_increasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ 2 → f x < f y

-- Define the theorem
theorem range_of_m (h1 : is_increasing_on_interval f) (h2 : ∀ m, f (1 - m) < f m) :
  ∀ m, m ∈ Set.Ioo (1/2) 2 ↔ -2 ≤ 1 - m ∧ 1 - m < m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3540_354091


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3540_354017

theorem max_value_of_expression (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (h_sum : x + y + z = 3) :
  (x * y / (x + y + 1) + x * z / (x + z + 1) + y * z / (y + z + 1)) ≤ 1 ∧
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 3 ∧
    x * y / (x + y + 1) + x * z / (x + z + 1) + y * z / (y + z + 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3540_354017


namespace NUMINAMATH_CALUDE_fred_money_last_week_l3540_354081

theorem fred_money_last_week 
  (fred_now : ℕ)
  (jason_now : ℕ)
  (jason_earned : ℕ)
  (jason_last_week : ℕ)
  (h1 : fred_now = 112)
  (h2 : jason_now = 63)
  (h3 : jason_earned = 60)
  (h4 : jason_last_week = 3)
  : fred_now - (jason_earned + jason_last_week) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fred_money_last_week_l3540_354081


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_sufficient_not_necessary_condition_l3540_354023

-- Define the sets M and P
def M : Set ℝ := {x | x < -3 ∨ x > 5}
def P (a : ℝ) : Set ℝ := {x | (x - a) * (x - 8) ≤ 0}

-- Theorem 1: Necessary and sufficient condition
theorem necessary_and_sufficient_condition (a : ℝ) :
  M ∩ P a = {x | 5 < x ∧ x ≤ 8} ↔ -3 ≤ a ∧ a ≤ 5 := by sorry

-- Theorem 2: Sufficient but not necessary condition
theorem sufficient_not_necessary_condition :
  ∃ a : ℝ, (M ∩ P a = {x | 5 < x ∧ x ≤ 8}) ∧
  ¬(∀ b : ℝ, M ∩ P b = {x | 5 < x ∧ x ≤ 8} → b = a) := by sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_sufficient_not_necessary_condition_l3540_354023


namespace NUMINAMATH_CALUDE_horner_v2_value_l3540_354065

def horner_step (v : ℝ) (x : ℝ) (a : ℝ) : ℝ := v * x + a

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (λ acc coeff => horner_step acc x coeff) 0

def polynomial : List ℝ := [5, 2, 3.5, -2.6, 1.7, -0.8]

theorem horner_v2_value :
  let x : ℝ := 5
  let v0 : ℝ := polynomial.head!
  let v1 : ℝ := horner_step v0 x (polynomial.get! 1)
  let v2 : ℝ := horner_step v1 x (polynomial.get! 2)
  v2 = 138.5 := by sorry

end NUMINAMATH_CALUDE_horner_v2_value_l3540_354065


namespace NUMINAMATH_CALUDE_gravitational_force_at_distance_l3540_354088

/-- Represents the gravitational force at a given distance -/
structure GravitationalForce where
  distance : ℝ
  force : ℝ

/-- The gravitational constant k = f * d^2 -/
def gravitational_constant (gf : GravitationalForce) : ℝ :=
  gf.force * gf.distance^2

theorem gravitational_force_at_distance 
  (surface_force : GravitationalForce) 
  (space_force : GravitationalForce) :
  surface_force.distance = 5000 →
  surface_force.force = 800 →
  space_force.distance = 300000 →
  gravitational_constant surface_force = gravitational_constant space_force →
  space_force.force = 1/45 := by
sorry

end NUMINAMATH_CALUDE_gravitational_force_at_distance_l3540_354088


namespace NUMINAMATH_CALUDE_choir_group_ratio_l3540_354052

theorem choir_group_ratio (total_sopranos total_altos num_groups : ℕ) 
  (h1 : total_sopranos = 10)
  (h2 : total_altos = 15)
  (h3 : num_groups = 5)
  (h4 : total_sopranos % num_groups = 0)
  (h5 : total_altos % num_groups = 0) :
  (total_sopranos / num_groups : ℚ) / (total_altos / num_groups : ℚ) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_choir_group_ratio_l3540_354052


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3540_354073

theorem perfect_square_condition (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + m*x + 1 = y^2) → (m = 2 ∨ m = -2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3540_354073


namespace NUMINAMATH_CALUDE_num_chosen_bulbs_is_two_l3540_354051

/-- The number of bulbs chosen at random from a box containing defective and non-defective bulbs. -/
def num_chosen_bulbs : ℕ :=
  -- The actual number will be defined in the proof
  sorry

/-- The total number of bulbs in the box. -/
def total_bulbs : ℕ := 21

/-- The number of defective bulbs in the box. -/
def defective_bulbs : ℕ := 4

/-- The probability of choosing at least one defective bulb. -/
def prob_at_least_one_defective : ℝ := 0.35238095238095235

theorem num_chosen_bulbs_is_two :
  num_chosen_bulbs = 2 ∧
  (1 : ℝ) - (total_bulbs - defective_bulbs : ℝ) / total_bulbs ^ num_chosen_bulbs = prob_at_least_one_defective :=
by sorry

end NUMINAMATH_CALUDE_num_chosen_bulbs_is_two_l3540_354051


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3540_354094

theorem unique_quadratic_solution (b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! x, b * x^2 + 25 * x + 9 = 0) :
  ∃ x, b * x^2 + 25 * x + 9 = 0 ∧ x = -18/25 := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3540_354094


namespace NUMINAMATH_CALUDE_t_shirt_jersey_cost_difference_l3540_354047

/-- The cost difference between a t-shirt and a jersey -/
def cost_difference (t_shirt_price jersey_price : ℕ) : ℕ :=
  t_shirt_price - jersey_price

/-- Theorem: The cost difference between a t-shirt and a jersey is $158 -/
theorem t_shirt_jersey_cost_difference :
  cost_difference 192 34 = 158 := by
  sorry

end NUMINAMATH_CALUDE_t_shirt_jersey_cost_difference_l3540_354047


namespace NUMINAMATH_CALUDE_aurelia_percentage_l3540_354027

/-- Given Lauryn's earnings and the total earnings of Lauryn and Aurelia,
    calculate the percentage of Lauryn's earnings that Aurelia made. -/
theorem aurelia_percentage (lauryn_earnings total_earnings : ℝ) : 
  lauryn_earnings = 2000 →
  total_earnings = 3400 →
  (100 * (total_earnings - lauryn_earnings)) / lauryn_earnings = 70 := by
sorry

end NUMINAMATH_CALUDE_aurelia_percentage_l3540_354027


namespace NUMINAMATH_CALUDE_smallest_distance_circle_ellipse_l3540_354015

/-- The smallest distance between a point on the unit circle and a point on a specific ellipse -/
theorem smallest_distance_circle_ellipse :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let ellipse := {p : ℝ × ℝ | ((p.1 - 2)^2 / 9) + (p.2^2 / 9) = 1}
  (∃ (A : ℝ × ℝ) (B : ℝ × ℝ), A ∈ circle ∧ B ∈ ellipse ∧
    ∀ (C : ℝ × ℝ) (D : ℝ × ℝ), C ∈ circle → D ∈ ellipse →
      Real.sqrt 2 - 1 ≤ Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_smallest_distance_circle_ellipse_l3540_354015


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3540_354083

/-- Given two parallel vectors a and b, prove that x = 1/2 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (2 * x + 1, 4)
  let b : ℝ × ℝ := (2 - x, 3)
  (∃ (k : ℝ), a = k • b) →
  x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3540_354083


namespace NUMINAMATH_CALUDE_power_function_quadrants_l3540_354053

/-- A function f(x) = (m^2 - 5m + 7)x^m is a power function with its graph
    distributed in the first and third quadrants if and only if m = 3 -/
theorem power_function_quadrants (m : ℝ) : 
  (∀ x ≠ 0, ∃ f : ℝ → ℝ, f x = (m^2 - 5*m + 7) * x^m) ∧ 
  (∀ x > 0, (m^2 - 5*m + 7) * x^m > 0) ∧
  (∀ x < 0, (m^2 - 5*m + 7) * x^m < 0) ∧
  (m^2 - 5*m + 7 = 1) ↔ 
  m = 3 := by sorry

end NUMINAMATH_CALUDE_power_function_quadrants_l3540_354053


namespace NUMINAMATH_CALUDE_five_dice_same_number_probability_l3540_354074

theorem five_dice_same_number_probability : 
  let number_of_dice : ℕ := 5
  let faces_per_die : ℕ := 6
  let total_outcomes : ℕ := faces_per_die ^ number_of_dice
  let favorable_outcomes : ℕ := faces_per_die
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 1296 := by
sorry

end NUMINAMATH_CALUDE_five_dice_same_number_probability_l3540_354074


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3540_354058

theorem simplify_trig_expression (α : ℝ) :
  (2 * Real.sin (π - α) + Real.sin (2 * α)) / (Real.cos (α / 2))^2 = 4 * Real.sin α :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3540_354058


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3540_354037

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (z : ℂ), (3 : ℂ) - 2 * i * z = -4 + 5 * i * z ∧ z = -i :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3540_354037


namespace NUMINAMATH_CALUDE_weight_gain_ratio_l3540_354080

/-- The weight gain problem at the family reunion --/
theorem weight_gain_ratio (orlando jose fernando : ℕ) : 
  orlando = 5 →
  jose = 2 * orlando + 2 →
  orlando + jose + fernando = 20 →
  fernando * 4 = jose := by
  sorry

end NUMINAMATH_CALUDE_weight_gain_ratio_l3540_354080


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3540_354079

theorem purely_imaginary_complex_number (a : ℝ) : 
  (∀ z : ℂ, z = (a^2 - 3*a + 2 : ℝ) + (a - 1 : ℝ)*I → z.re = 0 ∧ z.im ≠ 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3540_354079


namespace NUMINAMATH_CALUDE_sample_size_calculation_l3540_354071

/-- Given a factory producing three product models A, B, and C with quantities in the ratio 3:4:7,
    prove that a sample containing 15 units of product A has a total size of 70. -/
theorem sample_size_calculation (ratio_A ratio_B ratio_C : ℕ) (sample_A : ℕ) (n : ℕ) : 
  ratio_A = 3 → ratio_B = 4 → ratio_C = 7 → sample_A = 15 →
  n = (ratio_A + ratio_B + ratio_C) * sample_A / ratio_A → n = 70 := by
  sorry

#check sample_size_calculation

end NUMINAMATH_CALUDE_sample_size_calculation_l3540_354071


namespace NUMINAMATH_CALUDE_grid_flip_theorem_l3540_354014

/-- Represents a 4x4 binary grid -/
def Grid := Matrix (Fin 4) (Fin 4) Bool

/-- Represents a flip operation on the grid -/
inductive FlipOperation
| row : Fin 4 → FlipOperation
| column : Fin 4 → FlipOperation
| diagonal : Bool → FlipOperation  -- True for main diagonal, False for anti-diagonal

/-- Applies a flip operation to the grid -/
def applyFlip (g : Grid) (op : FlipOperation) : Grid :=
  sorry

/-- Checks if the grid is all zeros -/
def isAllZeros (g : Grid) : Prop :=
  ∀ i j, g i j = false

/-- Initial configurations -/
def initialGrid1 : Grid :=
  ![![false, true,  true,  false],
    ![true,  true,  false, true ],
    ![false, false, true,  true ],
    ![false, false, true,  true ]]

def initialGrid2 : Grid :=
  ![![false, true,  false, false],
    ![true,  true,  false, true ],
    ![false, false, false, true ],
    ![true,  false, true,  true ]]

def initialGrid3 : Grid :=
  ![![false, false, false, false],
    ![true,  true,  false, false],
    ![false, true,  false, true ],
    ![true,  false, false, true ]]

/-- Main theorem -/
theorem grid_flip_theorem :
  (¬ ∃ (ops : List FlipOperation), isAllZeros (ops.foldl applyFlip initialGrid1)) ∧
  (¬ ∃ (ops : List FlipOperation), isAllZeros (ops.foldl applyFlip initialGrid2)) ∧
  (∃ (ops : List FlipOperation), isAllZeros (ops.foldl applyFlip initialGrid3)) :=
sorry

end NUMINAMATH_CALUDE_grid_flip_theorem_l3540_354014


namespace NUMINAMATH_CALUDE_jerseys_sold_equals_tshirts_sold_l3540_354069

theorem jerseys_sold_equals_tshirts_sold (jersey_profit : ℕ) (tshirt_profit : ℕ) 
  (tshirts_sold : ℕ) (jersey_cost_difference : ℕ) :
  jersey_profit = 115 →
  tshirt_profit = 25 →
  tshirts_sold = 113 →
  jersey_cost_difference = 90 →
  jersey_profit = tshirt_profit + jersey_cost_difference →
  ∃ (jerseys_sold : ℕ), jerseys_sold = tshirts_sold :=
by sorry


end NUMINAMATH_CALUDE_jerseys_sold_equals_tshirts_sold_l3540_354069


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3540_354006

theorem quadratic_equation_roots (p : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + p * x = 2) ∧ 
  (3 * (-1)^2 + p * (-1) = 2) →
  (3 * (2/3)^2 + p * (2/3) = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3540_354006


namespace NUMINAMATH_CALUDE_moving_circle_properties_l3540_354077

/-- The trajectory of the center of a moving circle M that is externally tangent to O₁ and internally tangent to O₂ -/
def trajectory_equation (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 27 = 1

/-- The product of slopes of lines connecting M(x,y) with fixed points -/
def slope_product (x y : ℝ) : Prop :=
  y ≠ 0 → (y / (x + 6)) * (y / (x - 6)) = -3/4

/-- Circle O₁ equation -/
def circle_O₁ (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x + 5 = 0

/-- Circle O₂ equation -/
def circle_O₂ (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 91 = 0

theorem moving_circle_properties
  (x y : ℝ)
  (h₁ : ∃ (r : ℝ), r > 0 ∧ ∀ (x' y' : ℝ), (x' - x)^2 + (y' - y)^2 = r^2 →
    (circle_O₁ x' y' ∨ circle_O₂ x' y') ∧ ¬(circle_O₁ x' y' ∧ circle_O₂ x' y')) :
  trajectory_equation x y ∧ slope_product x y :=
sorry

end NUMINAMATH_CALUDE_moving_circle_properties_l3540_354077


namespace NUMINAMATH_CALUDE_girls_fraction_proof_l3540_354031

theorem girls_fraction_proof (T G B : ℕ) (x : ℚ) : 
  (x * G = (1 / 6) * T) →  -- Some fraction of girls is 1/6 of total
  (B = 2 * G) →            -- Ratio of boys to girls is 2
  (T = B + G) →            -- Total is sum of boys and girls
  (x = 1 / 2) :=           -- Fraction of girls is 1/2
by sorry

end NUMINAMATH_CALUDE_girls_fraction_proof_l3540_354031


namespace NUMINAMATH_CALUDE_no_nonperiodic_function_satisfies_equation_l3540_354085

theorem no_nonperiodic_function_satisfies_equation :
  ¬∃ f : ℝ → ℝ, (∀ x : ℝ, f (x + 1) = f x * (f x + 1)) ∧ (¬∃ p : ℝ, p ≠ 0 ∧ ∀ x : ℝ, f (x + p) = f x) :=
by sorry

end NUMINAMATH_CALUDE_no_nonperiodic_function_satisfies_equation_l3540_354085


namespace NUMINAMATH_CALUDE_digit_245_l3540_354030

/-- The decimal representation of 13/17 -/
def decimal_rep : ℚ := 13 / 17

/-- The length of the repeating sequence in the decimal representation of 13/17 -/
def repeat_length : ℕ := 16

/-- The nth digit in the decimal representation of 13/17 -/
def nth_digit (n : ℕ) : ℕ := sorry

theorem digit_245 : nth_digit 245 = 7 := by sorry

end NUMINAMATH_CALUDE_digit_245_l3540_354030


namespace NUMINAMATH_CALUDE_inequality_proof_l3540_354072

theorem inequality_proof (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 0 < z ∧ z < 1) : 
  x / (1 - x) + y / (1 - y) + z / (1 - z) ≥ 3 * (x * y * z)^(1/3) / (1 - (x * y * z)^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3540_354072


namespace NUMINAMATH_CALUDE_circle_reassembly_possible_l3540_354093

/-- A circle with a marked point -/
structure MarkedCircle where
  center : ℝ × ℝ
  radius : ℝ
  marked_point : ℝ × ℝ

/-- A piece of a circle -/
structure CirclePiece

/-- Represents the process of cutting a circle into pieces -/
def cut_circle (c : MarkedCircle) (n : ℕ) : List CirclePiece :=
  sorry

/-- Represents the process of assembling pieces into a new circle -/
def assemble_circle (pieces : List CirclePiece) : MarkedCircle :=
  sorry

/-- Theorem stating that it's possible to cut and reassemble the circle as required -/
theorem circle_reassembly_possible (c : MarkedCircle) :
  ∃ (pieces : List CirclePiece),
    (pieces.length = 3) ∧
    (∃ (new_circle : MarkedCircle),
      (assemble_circle pieces = new_circle) ∧
      (new_circle.marked_point = new_circle.center)) :=
  sorry

end NUMINAMATH_CALUDE_circle_reassembly_possible_l3540_354093


namespace NUMINAMATH_CALUDE_triangle_probability_ten_points_triangle_probability_ten_points_with_conditions_l3540_354043

/-- Given 10 points in a plane where no three are collinear, this function
    calculates the probability that three out of four randomly chosen
    distinct segments connecting pairs of these points will form a triangle. -/
def probability_triangle_from_segments (n : ℕ) : ℚ :=
  if n = 10 then 16 / 473
  else 0

/-- Theorem stating that the probability of forming a triangle
    from three out of four randomly chosen segments is 16/473
    when there are 10 points in the plane and no three are collinear. -/
theorem triangle_probability_ten_points :
  probability_triangle_from_segments 10 = 16 / 473 := by
  sorry

/-- Assumption that no three points are collinear in the given set of points. -/
axiom no_three_collinear (n : ℕ) : Prop

/-- Theorem stating that given 10 points in a plane where no three are collinear,
    the probability that three out of four randomly chosen distinct segments
    connecting pairs of these points will form a triangle is 16/473. -/
theorem triangle_probability_ten_points_with_conditions :
  no_three_collinear 10 →
  probability_triangle_from_segments 10 = 16 / 473 := by
  sorry

end NUMINAMATH_CALUDE_triangle_probability_ten_points_triangle_probability_ten_points_with_conditions_l3540_354043


namespace NUMINAMATH_CALUDE_simplify_expression_l3540_354041

theorem simplify_expression (x : ℝ) : 4 * x^2 - (2 * x^2 + x - 1) + (2 - x^2 + 3 * x) = x^2 + 2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3540_354041


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l3540_354061

theorem solution_satisfies_equations : ∃ x : ℚ, 8 * x^3 = 125 ∧ 4 * (x - 1)^2 = 9 := by
  use 5/2
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l3540_354061


namespace NUMINAMATH_CALUDE_five_dice_not_same_probability_l3540_354003

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The number of dice being rolled -/
def num_dice : ℕ := 5

/-- The probability that five fair 6-sided dice won't all show the same number -/
theorem five_dice_not_same_probability :
  (1 - (num_sides : ℚ) / (num_sides ^ num_dice)) = 1295 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_five_dice_not_same_probability_l3540_354003


namespace NUMINAMATH_CALUDE_sum_complex_exp_argument_l3540_354013

/-- The sum of five complex exponentials has an argument of 59π/120 -/
theorem sum_complex_exp_argument :
  let z₁ := Complex.exp (11 * Real.pi * Complex.I / 120)
  let z₂ := Complex.exp (31 * Real.pi * Complex.I / 120)
  let z₃ := Complex.exp (-13 * Real.pi * Complex.I / 120)
  let z₄ := Complex.exp (-53 * Real.pi * Complex.I / 120)
  let z₅ := Complex.exp (-73 * Real.pi * Complex.I / 120)
  let sum := z₁ + z₂ + z₃ + z₄ + z₅
  ∃ (r : ℝ), sum = r * Complex.exp (59 * Real.pi * Complex.I / 120) ∧ r > 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_complex_exp_argument_l3540_354013


namespace NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l3540_354009

theorem sqrt_six_div_sqrt_two_eq_sqrt_three :
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l3540_354009


namespace NUMINAMATH_CALUDE_rahim_average_book_price_l3540_354084

/-- The average price of books bought by Rahim -/
def average_price (books1 books2 : ℕ) (price1 price2 : ℚ) : ℚ :=
  (price1 + price2) / (books1 + books2 : ℚ)

/-- Theorem: The average price Rahim paid per book is 85 rupees -/
theorem rahim_average_book_price :
  average_price 65 35 6500 2000 = 85 := by
  sorry

end NUMINAMATH_CALUDE_rahim_average_book_price_l3540_354084


namespace NUMINAMATH_CALUDE_f_properties_l3540_354046

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4

-- State the theorem
theorem f_properties :
  (∀ x y : ℝ, f (x * y) + f (y - x) ≥ f (y + x)) ∧
  (∀ x : ℝ, f x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_f_properties_l3540_354046


namespace NUMINAMATH_CALUDE_f_min_value_l3540_354097

/-- The function f(x) = 9x - 4x^2 -/
def f (x : ℝ) := 9 * x - 4 * x^2

/-- The minimum value of f(x) is -81/16 -/
theorem f_min_value : ∀ x : ℝ, f x ≥ -81/16 := by sorry

end NUMINAMATH_CALUDE_f_min_value_l3540_354097


namespace NUMINAMATH_CALUDE_distance_CX_l3540_354035

/-- Given five points A, B, C, D, X on a plane with specific distances between them,
    prove that the distance between C and X is 3. -/
theorem distance_CX (A B C D X : EuclideanSpace ℝ (Fin 2)) 
  (h1 : dist A C = 2)
  (h2 : dist A X = 5)
  (h3 : dist A D = 11)
  (h4 : dist C D = 9)
  (h5 : dist C B = 10)
  (h6 : dist D B = 1)
  (h7 : dist X B = 7) :
  dist C X = 3 := by
  sorry


end NUMINAMATH_CALUDE_distance_CX_l3540_354035


namespace NUMINAMATH_CALUDE_angle_conversion_l3540_354059

theorem angle_conversion (angle_deg : ℝ) (k : ℤ) (α : ℝ) :
  angle_deg = -1125 →
  (k = -4 ∧ α = (7 * π) / 4) →
  (0 ≤ α ∧ α < 2 * π) →
  angle_deg * π / 180 = 2 * k * π + α := by
  sorry

end NUMINAMATH_CALUDE_angle_conversion_l3540_354059


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3540_354048

/-- Given an arithmetic sequence {aₙ} with sum Sₙ of its first n terms, 
    if S₉ = a₄ + a₅ + a₆ + 72, then a₃ + a₇ = 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = (n : ℝ) * (a 1 + a n) / 2) →  -- Definition of Sₙ for arithmetic sequence
  (∀ n k, a (n + k) - a n = k * (a 2 - a 1)) →  -- Definition of arithmetic sequence
  S 9 = a 4 + a 5 + a 6 + 72 →  -- Given condition
  a 3 + a 7 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3540_354048


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3540_354050

/-- The area of a square inscribed in the ellipse x^2/4 + y^2 = 1, 
    with sides parallel to the coordinate axes -/
theorem inscribed_square_area : 
  ∃ (s : ℝ), s > 0 ∧ 
  (∀ (x y : ℝ), x^2/4 + y^2 = 1 → 
    (x = s ∨ x = -s) ∧ (y = s ∨ y = -s)) →
  s^2 = 16/5 := by
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3540_354050


namespace NUMINAMATH_CALUDE_set_union_problem_l3540_354067

theorem set_union_problem (M N : Set ℕ) (x : ℕ) :
  M = {0, x} →
  N = {1, 2} →
  M ∩ N = {1} →
  M ∪ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l3540_354067


namespace NUMINAMATH_CALUDE_triangle_property_l3540_354028

theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given condition
  Real.sin A ^ 2 - Real.sin B ^ 2 - Real.sin C ^ 2 = Real.sin B * Real.sin C →
  -- BC = 3
  a = 3 →
  -- Prove A = 2π/3
  A = 2 * π / 3 ∧
  -- Prove maximum perimeter is 3 + 2√3
  (∀ b' c' : Real, 0 < b' ∧ 0 < c' → a + b' + c' ≤ 3 + 2 * Real.sqrt 3) ∧
  (∃ b' c' : Real, 0 < b' ∧ 0 < c' ∧ a + b' + c' = 3 + 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l3540_354028


namespace NUMINAMATH_CALUDE_spelling_bee_contestants_l3540_354026

theorem spelling_bee_contestants (total : ℕ) : 
  (total / 2 : ℚ) / 4 = 30 → total = 240 := by sorry

end NUMINAMATH_CALUDE_spelling_bee_contestants_l3540_354026


namespace NUMINAMATH_CALUDE_sum_of_factors_l3540_354016

theorem sum_of_factors (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e → 
  (8 - a) * (8 - b) * (8 - c) * (8 - d) * (8 - e) = 120 →
  a + b + c + d + e = 39 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l3540_354016


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l3540_354012

theorem smallest_lcm_with_gcd_5 (m n : ℕ) : 
  1000 ≤ m ∧ m < 10000 ∧ 
  1000 ≤ n ∧ n < 10000 ∧ 
  Nat.gcd m n = 5 →
  201000 ≤ Nat.lcm m n ∧ 
  ∃ (a b : ℕ), 1000 ≤ a ∧ a < 10000 ∧ 
               1000 ≤ b ∧ b < 10000 ∧ 
               Nat.gcd a b = 5 ∧ 
               Nat.lcm a b = 201000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l3540_354012


namespace NUMINAMATH_CALUDE_machines_working_first_scenario_l3540_354066

/-- The number of machines working in the first scenario -/
def num_machines : ℕ := 8

/-- The time taken in the first scenario (in hours) -/
def time_first_scenario : ℕ := 6

/-- The number of machines in the second scenario -/
def num_machines_second : ℕ := 6

/-- The time taken in the second scenario (in hours) -/
def time_second : ℕ := 8

/-- The total work done in one job lot -/
def total_work : ℕ := 1

theorem machines_working_first_scenario :
  num_machines * time_first_scenario = num_machines_second * time_second :=
by sorry

end NUMINAMATH_CALUDE_machines_working_first_scenario_l3540_354066


namespace NUMINAMATH_CALUDE_equation_solutions_l3540_354086

theorem equation_solutions : 
  let f (x : ℝ) := (x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 3) * (x - 5) * (x - 1)
  let g (x : ℝ) := (x - 3) * (x - 7) * (x - 3)
  let S := {x : ℝ | x ≠ 3 ∧ x ≠ 7 ∧ f x / g x = 1}
  S = {3 + Real.sqrt 3, 3 - Real.sqrt 3, 3 + Real.sqrt 5, 3 - Real.sqrt 5} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3540_354086


namespace NUMINAMATH_CALUDE_sqrt_nine_over_two_simplification_l3540_354001

theorem sqrt_nine_over_two_simplification :
  Real.sqrt (9 / 2) = (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_over_two_simplification_l3540_354001


namespace NUMINAMATH_CALUDE_unique_k_no_solution_l3540_354029

theorem unique_k_no_solution (k : ℕ+) : 
  (k = 2) ↔ 
  ∀ m n : ℕ+, m ≠ n → 
    ¬(Nat.lcm m.val n.val - Nat.gcd m.val n.val = k.val * (m.val - n.val)) :=
by sorry

end NUMINAMATH_CALUDE_unique_k_no_solution_l3540_354029


namespace NUMINAMATH_CALUDE_problem_1_l3540_354090

theorem problem_1 : Real.sqrt 12 + 3 - 2^2 + |1 - Real.sqrt 3| = 3 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3540_354090


namespace NUMINAMATH_CALUDE_line_equation_l3540_354044

/-- Given a line passing through (a, 0) and cutting a triangular region with area T from the second quadrant, 
    the equation of the line is -2Tx + a²y + 2aT = 0 -/
theorem line_equation (a T : ℝ) (h1 : a ≠ 0) (h2 : T > 0) : 
  ∃ (f : ℝ → ℝ), (∀ x y, f x = y ↔ -2 * T * x + a^2 * y + 2 * a * T = 0) ∧ 
                  (f a = 0) ∧
                  (∀ x y, x > 0 → y > 0 → f x = y → 
                    (1/2) * a * y = T) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l3540_354044


namespace NUMINAMATH_CALUDE_max_distinct_factors_max_additional_factors_l3540_354025

theorem max_distinct_factors (x : Finset ℕ) :
  (∀ y ∈ x, y > 0) →
  (Nat.lcm 1024 2016 = Finset.lcm x (Nat.lcm 1024 2016)) →
  x.card ≤ 66 :=
by sorry

theorem max_additional_factors :
  ∃ (x : Finset ℕ), x.card = 64 ∧
  (∀ y ∈ x, y > 0) ∧
  (Nat.lcm 1024 2016 = Finset.lcm x (Nat.lcm 1024 2016)) :=
by sorry

end NUMINAMATH_CALUDE_max_distinct_factors_max_additional_factors_l3540_354025


namespace NUMINAMATH_CALUDE_infection_model_properties_l3540_354007

/-- Represents the infection spread model -/
structure InfectionModel where
  initialInfected : ℕ := 1
  totalAfterTwoRounds : ℕ := 64
  averageInfectionRate : ℕ
  thirdRoundInfections : ℕ

/-- Theorem stating the properties of the infection model -/
theorem infection_model_properties (model : InfectionModel) :
  model.initialInfected = 1 ∧
  model.totalAfterTwoRounds = 64 →
  model.averageInfectionRate = 7 ∧
  model.thirdRoundInfections = 448 := by
  sorry

#check infection_model_properties

end NUMINAMATH_CALUDE_infection_model_properties_l3540_354007


namespace NUMINAMATH_CALUDE_meter_to_km_conversion_kg_to_g_conversion_cm_to_dm_conversion_time_to_minutes_conversion_l3540_354018

-- Define conversion rates
def meter_to_km : ℕ → ℕ := λ m => m / 1000
def kg_to_g : ℕ → ℕ := λ kg => kg * 1000
def cm_to_dm : ℕ → ℕ := λ cm => cm / 10
def hours_to_minutes : ℕ → ℕ := λ h => h * 60

-- Theorem statements
theorem meter_to_km_conversion : meter_to_km 6000 = 6 := by sorry

theorem kg_to_g_conversion : kg_to_g (5 + 2) = 7000 := by sorry

theorem cm_to_dm_conversion : cm_to_dm (58 + 32) = 9 := by sorry

theorem time_to_minutes_conversion : hours_to_minutes 3 + 30 = 210 := by sorry

end NUMINAMATH_CALUDE_meter_to_km_conversion_kg_to_g_conversion_cm_to_dm_conversion_time_to_minutes_conversion_l3540_354018


namespace NUMINAMATH_CALUDE_fraction_simplification_l3540_354076

theorem fraction_simplification (a : ℝ) (h : a ≠ 1) :
  (a + 1) / (1 - a) * (a^2 + a) / (a^2 + 2*a + 1) - 1 / (1 - a) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3540_354076


namespace NUMINAMATH_CALUDE_money_division_l3540_354068

/-- Represents the share of money for each person -/
structure Share :=
  (a : ℚ)
  (b : ℚ)
  (c : ℚ)

/-- The problem statement and proof -/
theorem money_division (s : Share) : 
  s.c = 64 ∧ 
  s.b = 0.65 * s.a ∧ 
  s.c = 0.40 * s.a → 
  s.a + s.b + s.c = 328 := by
sorry


end NUMINAMATH_CALUDE_money_division_l3540_354068


namespace NUMINAMATH_CALUDE_correct_operations_l3540_354034

theorem correct_operations (x : ℝ) : 
  (x / 9 - 20 = 8) → (x * 9 + 20 = 2288) := by
  sorry

end NUMINAMATH_CALUDE_correct_operations_l3540_354034


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3540_354099

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 - 5*x + (5/4)*a > 0) ↔ a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3540_354099


namespace NUMINAMATH_CALUDE_eventually_all_play_all_l3540_354056

/-- Represents a player in the tournament -/
inductive Player
  | Mathematician (id : ℕ)
  | Humanitarian (id : ℕ)

/-- Represents the state of the tournament -/
structure TournamentState where
  n : ℕ  -- number of humanities students
  m : ℕ  -- number of mathematicians
  queue : List Player
  table : Player × Player
  h_different_sizes : n ≠ m

/-- Represents a game played between two players -/
def Game := Player × Player

/-- Simulates the tournament for a given number of steps -/
def simulateTournament (initial : TournamentState) (steps : ℕ) : List Game := sorry

/-- Checks if all mathematicians have played with all humanitarians -/
def allPlayedAgainstAll (games : List Game) : Prop := sorry

/-- The main theorem stating that eventually all mathematicians will play against all humanitarians -/
theorem eventually_all_play_all (initial : TournamentState) :
  ∃ k : ℕ, allPlayedAgainstAll (simulateTournament initial k) := by
  sorry

end NUMINAMATH_CALUDE_eventually_all_play_all_l3540_354056


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3540_354063

theorem cone_lateral_surface_area (l r : ℝ) (h1 : l = 5) (h2 : r = 2) :
  π * r * l = 10 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3540_354063


namespace NUMINAMATH_CALUDE_dog_bird_time_difference_l3540_354002

def dogs : ℕ := 3
def dog_hours : ℕ := 7
def holes : ℕ := 9
def birds : ℕ := 5
def bird_minutes : ℕ := 40
def nests : ℕ := 2

def dog_dig_time : ℚ := (dog_hours * 60 : ℚ) * holes / dogs
def bird_build_time : ℚ := (bird_minutes : ℚ) * birds / nests

theorem dog_bird_time_difference :
  dog_dig_time - bird_build_time = 40 := by sorry

end NUMINAMATH_CALUDE_dog_bird_time_difference_l3540_354002


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3540_354032

-- Define the types for planes and lines
variable (Plane : Type) (Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel
  (α β γ : Plane) (m n : Line)
  (h₁ : α ≠ β) (h₂ : α ≠ γ) (h₃ : β ≠ γ) (h₄ : m ≠ n)
  (h₅ : perpendicular m α) (h₆ : perpendicular n α) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3540_354032


namespace NUMINAMATH_CALUDE_select_workers_count_l3540_354055

/-- The number of ways to select two workers from a group of three for day and night shifts -/
def select_workers : ℕ :=
  let workers := 3
  let day_shift_choices := workers
  let night_shift_choices := workers - 1
  day_shift_choices * night_shift_choices

/-- Theorem: The number of ways to select two workers from a group of three for day and night shifts is 6 -/
theorem select_workers_count : select_workers = 6 := by
  sorry

end NUMINAMATH_CALUDE_select_workers_count_l3540_354055


namespace NUMINAMATH_CALUDE_delta_curve_from_rotations_l3540_354040

/-- A curve in 2D space -/
structure Curve where
  -- Add necessary fields for a curve

/-- Rotation of a curve around a point by an angle -/
def rotate (c : Curve) (center : ℝ × ℝ) (angle : ℝ) : Curve :=
  sorry

/-- Sum of curves -/
def sum_curves (curves : List Curve) : Curve :=
  sorry

/-- Check if a curve is a circle with given radius -/
def is_circle (c : Curve) (radius : ℝ) : Prop :=
  sorry

/-- Check if a curve is convex -/
def is_convex (c : Curve) : Prop :=
  sorry

/-- Check if a curve is a Δ-curve -/
def is_delta_curve (c : Curve) : Prop :=
  sorry

/-- Main theorem -/
theorem delta_curve_from_rotations (K : Curve) (O : ℝ × ℝ) (h : ℝ) :
  is_convex K →
  let K' := rotate K O (2 * π / 3)
  let K'' := rotate K O (4 * π / 3)
  let M := sum_curves [K, K', K'']
  is_circle M h →
  is_delta_curve K :=
sorry

end NUMINAMATH_CALUDE_delta_curve_from_rotations_l3540_354040


namespace NUMINAMATH_CALUDE_smallest_number_with_18_factors_l3540_354024

def num_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_number_with_18_factors : 
  ∃ m : ℕ, m > 1 ∧ 
           num_factors m = 18 ∧ 
           num_factors m - 2 ≥ 16 ∧
           ∀ k : ℕ, k > 1 → num_factors k = 18 → num_factors k - 2 ≥ 16 → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_18_factors_l3540_354024
