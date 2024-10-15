import Mathlib

namespace NUMINAMATH_CALUDE_owen_work_hours_l2762_276279

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Represents the number of hours Owen spends on daily chores -/
def hours_on_chores : ℕ := 7

/-- Represents the number of hours Owen sleeps -/
def hours_sleeping : ℕ := 11

/-- Calculates the number of hours Owen spends at work -/
def hours_at_work : ℕ := hours_in_day - hours_on_chores - hours_sleeping

/-- Theorem stating that Owen spends 6 hours at work -/
theorem owen_work_hours : hours_at_work = 6 := by
  sorry

end NUMINAMATH_CALUDE_owen_work_hours_l2762_276279


namespace NUMINAMATH_CALUDE_business_profit_l2762_276215

theorem business_profit (total_profit : ℝ) : 
  (0.25 * total_profit) + 2 * (0.25 * (0.75 * total_profit)) = 50000 →
  total_profit = 80000 := by
sorry

end NUMINAMATH_CALUDE_business_profit_l2762_276215


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l2762_276249

theorem probability_of_red_ball (basketA_white basketA_red basketB_yellow basketB_red basketB_black : ℕ)
  (probA probB : ℝ) : 
  basketA_white = 10 →
  basketA_red = 5 →
  basketB_yellow = 4 →
  basketB_red = 6 →
  basketB_black = 5 →
  probA = 0.6 →
  probB = 0.4 →
  (basketA_red / (basketA_white + basketA_red : ℝ)) * probA +
  (basketB_red / (basketB_yellow + basketB_red + basketB_black : ℝ)) * probB = 0.36 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l2762_276249


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l2762_276242

theorem cube_volume_surface_area (x : ℝ) :
  (∃ s : ℝ, s > 0 ∧ s^3 = 27*x ∧ 6*s^2 = 3*x) → x = 5832 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l2762_276242


namespace NUMINAMATH_CALUDE_square_root_divided_by_two_l2762_276299

theorem square_root_divided_by_two : Real.sqrt 16 / 2 = 2 := by sorry

end NUMINAMATH_CALUDE_square_root_divided_by_two_l2762_276299


namespace NUMINAMATH_CALUDE_triangle_area_maximum_l2762_276252

/-- The area of a triangle with two fixed sides is maximized when the angle between these sides is 90°. -/
theorem triangle_area_maximum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ∃ θ : ℝ, θ ∈ Set.Icc 0 π ∧
    ∀ φ : ℝ, φ ∈ Set.Icc 0 π →
      (1 / 2) * a * b * Real.sin θ ≥ (1 / 2) * a * b * Real.sin φ :=
  sorry

end NUMINAMATH_CALUDE_triangle_area_maximum_l2762_276252


namespace NUMINAMATH_CALUDE_parabola_intersection_midpoint_l2762_276278

/-- Parabola defined by y² = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Condition that A and B are on the parabola and |AF| + |BF| = 10 -/
def IntersectionCondition (A B : ℝ × ℝ) : Prop :=
  A ∈ Parabola ∧ B ∈ Parabola ∧
  Real.sqrt ((A.1 - Focus.1)^2 + (A.2 - Focus.2)^2) +
  Real.sqrt ((B.1 - Focus.1)^2 + (B.2 - Focus.2)^2) = 10

/-- The theorem to be proved -/
theorem parabola_intersection_midpoint
  (A B : ℝ × ℝ) (h : IntersectionCondition A B) :
  (A.1 + B.1) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_midpoint_l2762_276278


namespace NUMINAMATH_CALUDE_min_segments_to_return_l2762_276261

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    and the measure of angle ABC is 80 degrees, prove that the minimum number of segments
    needed to return to the starting point is 18. -/
theorem min_segments_to_return (m_angle_ABC : ℝ) (n : ℕ) : 
  m_angle_ABC = 80 → 
  (∀ m : ℕ, 100 * n = 360 * m) → 
  n ≥ 18 ∧ 
  (∀ k < n, ¬(∀ m : ℕ, 100 * k = 360 * m)) := by
  sorry

end NUMINAMATH_CALUDE_min_segments_to_return_l2762_276261


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2762_276294

theorem simplify_and_evaluate (x y : ℝ) (h1 : x = -1) (h2 : y = 2) :
  ((x + y)^2 - (x + 2*y)*(x - 2*y)) / (2*y) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2762_276294


namespace NUMINAMATH_CALUDE_count_primes_with_digit_sum_10_l2762_276289

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ Nat.Prime n ∧ digit_sum n = 10

theorem count_primes_with_digit_sum_10 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfies_condition n) ∧ S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_primes_with_digit_sum_10_l2762_276289


namespace NUMINAMATH_CALUDE_M_mod_100_l2762_276243

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def trailingZeros (n : ℕ) : ℕ :=
  let rec count_fives (m : ℕ) (acc : ℕ) : ℕ :=
    if m < 5 then acc
    else count_fives (m / 5) (acc + m / 5)
  count_fives n 0

def M : ℕ := trailingZeros (factorial 50)

theorem M_mod_100 : M % 100 = 12 := by sorry

end NUMINAMATH_CALUDE_M_mod_100_l2762_276243


namespace NUMINAMATH_CALUDE_necessary_condition_inequality_l2762_276244

theorem necessary_condition_inequality (a b c : ℝ) (hc : c ≠ 0) :
  (∀ a b c, c ≠ 0 → (a * c^2 > b * c^2 → a > b)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_condition_inequality_l2762_276244


namespace NUMINAMATH_CALUDE_cargo_ship_unloading_time_l2762_276204

/-- Cargo ship transportation problem -/
theorem cargo_ship_unloading_time 
  (loading_speed : ℝ) 
  (loading_time : ℝ) 
  (unloading_speed : ℝ) 
  (unloading_time : ℝ) 
  (h1 : loading_speed = 30)
  (h2 : loading_time = 8)
  (h3 : unloading_speed > 0) :
  unloading_time = (loading_speed * loading_time) / unloading_speed :=
by
  sorry

#check cargo_ship_unloading_time

end NUMINAMATH_CALUDE_cargo_ship_unloading_time_l2762_276204


namespace NUMINAMATH_CALUDE_min_value_of_geometric_sequence_l2762_276270

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem min_value_of_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_condition : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) :
  ∃ min_value : ℝ, min_value = 54 ∧ 
  ∀ x : ℝ, (∃ a : ℕ → ℝ, geometric_sequence a ∧ 
    (∀ n : ℕ, a n > 0) ∧ 
    2 * a 4 + a 3 - 2 * a 2 - a 1 = 8 ∧
    2 * a 8 + a 7 = x) → x ≥ min_value :=
sorry

end NUMINAMATH_CALUDE_min_value_of_geometric_sequence_l2762_276270


namespace NUMINAMATH_CALUDE_spies_configuration_exists_l2762_276228

/-- Represents a position on the 6x6 board -/
structure Position where
  row : Fin 6
  col : Fin 6

/-- Represents the direction a spy is facing -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a spy on the board -/
structure Spy where
  pos : Position
  dir : Direction

/-- Determines if a spy can see a given position -/
def Spy.canSee (s : Spy) (p : Position) : Bool :=
  match s.dir with
  | Direction.Up => 
      (s.pos.row < p.row && p.row ≤ s.pos.row + 2 && s.pos.col - 1 ≤ p.col && p.col ≤ s.pos.col + 1) 
  | Direction.Down => 
      (s.pos.row > p.row && p.row ≥ s.pos.row - 2 && s.pos.col - 1 ≤ p.col && p.col ≤ s.pos.col + 1)
  | Direction.Left => 
      (s.pos.col > p.col && p.col ≥ s.pos.col - 2 && s.pos.row - 1 ≤ p.row && p.row ≤ s.pos.row + 1)
  | Direction.Right => 
      (s.pos.col < p.col && p.col ≤ s.pos.col + 2 && s.pos.row - 1 ≤ p.row && p.row ≤ s.pos.row + 1)

/-- A valid configuration of spies -/
def ValidConfiguration (spies : List Spy) : Prop :=
  spies.length = 18 ∧ 
  ∀ s1 s2, s1 ∈ spies → s2 ∈ spies → s1 ≠ s2 → ¬(s1.canSee s2.pos) ∧ ¬(s2.canSee s1.pos)

/-- There exists a valid configuration of 18 spies on a 6x6 board -/
theorem spies_configuration_exists : ∃ spies : List Spy, ValidConfiguration spies := by
  sorry

end NUMINAMATH_CALUDE_spies_configuration_exists_l2762_276228


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2762_276240

/-- The solution set of the inequality -x^2 - x + 6 > 0 is the open interval (-3, 2) -/
theorem solution_set_quadratic_inequality :
  {x : ℝ | -x^2 - x + 6 > 0} = Set.Ioo (-3) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2762_276240


namespace NUMINAMATH_CALUDE_grape_juice_percentage_l2762_276220

/-- Calculates the percentage of grape juice in a mixture after adding pure grape juice -/
theorem grape_juice_percentage
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_pure_juice : ℝ)
  (h1 : initial_volume = 40)
  (h2 : initial_concentration = 0.1)
  (h3 : added_pure_juice = 20)
  : (initial_volume * initial_concentration + added_pure_juice) / (initial_volume + added_pure_juice) = 0.4 := by
  sorry

#check grape_juice_percentage

end NUMINAMATH_CALUDE_grape_juice_percentage_l2762_276220


namespace NUMINAMATH_CALUDE_counterexample_absolute_value_inequality_l2762_276221

theorem counterexample_absolute_value_inequality : 
  ∃ (a b : ℝ), (abs a > abs b) ∧ (a ≤ b) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_absolute_value_inequality_l2762_276221


namespace NUMINAMATH_CALUDE_f_always_positive_l2762_276291

def f (x : ℝ) : ℝ := x^8 - x^5 + x^2 - x + 1

theorem f_always_positive : ∀ x : ℝ, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_always_positive_l2762_276291


namespace NUMINAMATH_CALUDE_absolute_value_calculation_l2762_276264

theorem absolute_value_calculation : |-6| - (-4) + (-7) = 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_calculation_l2762_276264


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l2762_276213

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (h_nonzero : x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ y₁ ≠ 0 ∧ y₂ ≠ 0) 
  (h_inverse : ∃ k : ℝ, x₁ * y₁ = k ∧ x₂ * y₂ = k) (h_ratio : x₁ / x₂ = 3 / 4) : 
  y₁ / y₂ = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l2762_276213


namespace NUMINAMATH_CALUDE_barycentric_coordinate_properties_l2762_276271

/-- Barycentric coordinates in a tetrahedron -/
structure BarycentricCoord :=
  (x₁ x₂ x₃ x₄ : ℝ)
  (sum_to_one : x₁ + x₂ + x₃ + x₄ = 1)

/-- The tetrahedron A₁A₂A₃A₄ -/
structure Tetrahedron :=
  (A₁ A₂ A₃ A₄ : BarycentricCoord)

/-- A point lies on line A₁A₂ iff x₃ = 0 and x₄ = 0 -/
def lies_on_line_A₁A₂ (t : Tetrahedron) (p : BarycentricCoord) : Prop :=
  p.x₃ = 0 ∧ p.x₄ = 0

/-- A point lies on plane A₁A₂A₃ iff x₄ = 0 -/
def lies_on_plane_A₁A₂A₃ (t : Tetrahedron) (p : BarycentricCoord) : Prop :=
  p.x₄ = 0

/-- A point lies on the plane through A₃A₄ parallel to A₁A₂ iff x₁ = -x₂ and x₃ + x₄ = 1 -/
def lies_on_plane_parallel_A₁A₂_through_A₃A₄ (t : Tetrahedron) (p : BarycentricCoord) : Prop :=
  p.x₁ = -p.x₂ ∧ p.x₃ + p.x₄ = 1

theorem barycentric_coordinate_properties (t : Tetrahedron) (p : BarycentricCoord) :
  (lies_on_line_A₁A₂ t p ↔ p.x₃ = 0 ∧ p.x₄ = 0) ∧
  (lies_on_plane_A₁A₂A₃ t p ↔ p.x₄ = 0) ∧
  (lies_on_plane_parallel_A₁A₂_through_A₃A₄ t p ↔ p.x₁ = -p.x₂ ∧ p.x₃ + p.x₄ = 1) := by
  sorry

end NUMINAMATH_CALUDE_barycentric_coordinate_properties_l2762_276271


namespace NUMINAMATH_CALUDE_determinant_equality_l2762_276260

theorem determinant_equality (p q r s : ℝ) : 
  Matrix.det !![p, q; r, s] = 7 → Matrix.det !![p - 3*r, q - 3*s; r, s] = 7 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l2762_276260


namespace NUMINAMATH_CALUDE_complex_equation_l2762_276229

theorem complex_equation (z : ℂ) : (Complex.I * z = 1 - 2 * Complex.I) → z = -2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_l2762_276229


namespace NUMINAMATH_CALUDE_min_basketballs_is_two_l2762_276241

/-- Represents the number of items sold for each type of sporting good. -/
structure ItemsSold where
  frisbees : ℕ
  baseballs : ℕ
  basketballs : ℕ

/-- Checks if the given ItemsSold satisfies all conditions of the problem. -/
def satisfiesConditions (items : ItemsSold) : Prop :=
  items.frisbees + items.baseballs + items.basketballs = 180 ∧
  3 * items.frisbees + 5 * items.baseballs + 10 * items.basketballs = 800 ∧
  items.frisbees > items.baseballs ∧
  items.baseballs > items.basketballs

/-- The minimum number of basketballs that could have been sold. -/
def minBasketballs : ℕ := 2

/-- Theorem stating that the minimum number of basketballs sold is 2. -/
theorem min_basketballs_is_two :
  ∀ items : ItemsSold,
    satisfiesConditions items →
    items.basketballs ≥ minBasketballs :=
by
  sorry

#check min_basketballs_is_two

end NUMINAMATH_CALUDE_min_basketballs_is_two_l2762_276241


namespace NUMINAMATH_CALUDE_range_of_g_range_of_a_l2762_276267

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 1| - |x - 2|

-- Theorem for the range of g(x)
theorem range_of_g : Set.range g = Set.Icc (-1 : ℝ) 1 := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, g x < a^2 + a + 1) ↔ (a < -1 ∨ a > 1) := by sorry

end NUMINAMATH_CALUDE_range_of_g_range_of_a_l2762_276267


namespace NUMINAMATH_CALUDE_legal_fee_participants_l2762_276210

/-- The number of participants paying legal fees -/
def num_participants : ℕ := 8

/-- The total legal costs in francs -/
def total_cost : ℕ := 800

/-- The number of participants who cannot pay -/
def non_paying_participants : ℕ := 3

/-- The additional amount each paying participant contributes in francs -/
def additional_payment : ℕ := 60

/-- Theorem stating that the number of participants satisfies the given conditions -/
theorem legal_fee_participants :
  (total_cost : ℚ) / num_participants + additional_payment = 
  total_cost / (num_participants - non_paying_participants) :=
by sorry

end NUMINAMATH_CALUDE_legal_fee_participants_l2762_276210


namespace NUMINAMATH_CALUDE_round_trip_speed_calculation_l2762_276224

/-- Proves that given specific conditions for a round trip, the return speed must be 45 mph -/
theorem round_trip_speed_calculation (distance : ℝ) (speed_there : ℝ) (avg_speed : ℝ) :
  distance = 180 →
  speed_there = 90 →
  avg_speed = 60 →
  (2 * distance) / (distance / speed_there + distance / (2 * avg_speed - speed_there)) = avg_speed →
  2 * avg_speed - speed_there = 45 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_speed_calculation_l2762_276224


namespace NUMINAMATH_CALUDE_cake_shop_work_duration_l2762_276285

/-- Calculates the number of months worked given the total hours worked by Cathy -/
def months_worked (total_hours : ℕ) : ℚ :=
  let hours_per_week : ℕ := 20
  let weeks_per_month : ℕ := 4
  let extra_hours : ℕ := 20
  let regular_hours : ℕ := total_hours - extra_hours
  let regular_weeks : ℚ := regular_hours / hours_per_week
  regular_weeks / weeks_per_month

theorem cake_shop_work_duration :
  months_worked 180 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cake_shop_work_duration_l2762_276285


namespace NUMINAMATH_CALUDE_smallest_inverse_domain_l2762_276233

-- Define the function f
def f (x : ℝ) : ℝ := (x + 2)^2 - 5

-- State the theorem
theorem smallest_inverse_domain (c : ℝ) : 
  (∀ x ≥ c, ∀ y ≥ c, f x = f y → x = y) ∧ 
  (∀ d < c, ∃ x y, d ≤ x ∧ d ≤ y ∧ x ≠ y ∧ f x = f y) ↔ 
  c = -2 :=
sorry

end NUMINAMATH_CALUDE_smallest_inverse_domain_l2762_276233


namespace NUMINAMATH_CALUDE_store_a_discount_proof_l2762_276281

/-- The additional discount percentage offered by Store A -/
def store_a_discount : ℝ := 8

/-- The full price of the smartphone at Store A -/
def store_a_full_price : ℝ := 125

/-- The full price of the smartphone at Store B -/
def store_b_full_price : ℝ := 130

/-- The additional discount percentage offered by Store B -/
def store_b_discount : ℝ := 10

/-- The price difference between Store A and Store B after discounts -/
def price_difference : ℝ := 2

theorem store_a_discount_proof :
  store_a_full_price * (1 - store_a_discount / 100) =
  store_b_full_price * (1 - store_b_discount / 100) - price_difference :=
by sorry


end NUMINAMATH_CALUDE_store_a_discount_proof_l2762_276281


namespace NUMINAMATH_CALUDE_problem_solution_l2762_276290

theorem problem_solution (A B : ℝ) 
  (h1 : 30 - (4 * A + 5) = 3 * B) 
  (h2 : B = 2 * A) : 
  A = 2.5 ∧ B = 5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2762_276290


namespace NUMINAMATH_CALUDE_pool_length_l2762_276201

theorem pool_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 3 → 
  area = 30 → 
  area = length * width → 
  length = 10 := by
sorry

end NUMINAMATH_CALUDE_pool_length_l2762_276201


namespace NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l2762_276263

-- Problem 1
theorem calculate_expression : (-3)^2 - 3^0 + (-2) = 6 := by sorry

-- Problem 2
theorem solve_system_of_equations :
  ∃ x y : ℝ, 2*x - y = 3 ∧ x + y = 6 ∧ x = 3 ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l2762_276263


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_l2762_276237

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Part 1
theorem solution_set_part1 :
  ∀ x : ℝ, f (-4) x ≥ 6 ↔ x ≤ 0 ∨ x ≥ 6 :=
sorry

-- Part 2
theorem range_of_a :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ |x - 3|) → a ∈ Set.Icc (-1) 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_l2762_276237


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l2762_276205

def isArithmeticProgression (a : Fin 2008 → ℕ) : Prop :=
  ∃ (start d : ℕ), ∀ i : Fin 10, a i = start + i.val * d

theorem exists_valid_coloring :
  ∃ (f : Fin 2008 → Fin 4),
    ∀ (a : Fin 10 → Fin 2008),
      isArithmeticProgression (λ i => (a i).val + 1) →
        ∃ (i j : Fin 10), f (a i) ≠ f (a j) :=
by sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l2762_276205


namespace NUMINAMATH_CALUDE_sequence_sum_properties_l2762_276275

/-- Defines the sequence where a_1 = 1 and between the k-th 1 and the (k+1)-th 1, there are 2^(k-1) terms of 2 -/
def a : ℕ → ℕ :=
  sorry

/-- Sum of the first n terms of the sequence -/
def S (n : ℕ) : ℕ :=
  sorry

theorem sequence_sum_properties :
  (S 1998 = 3985) ∧ (∀ n : ℕ, S n ≠ 2001) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_properties_l2762_276275


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l2762_276222

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l2762_276222


namespace NUMINAMATH_CALUDE_triangle_point_C_l2762_276265

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

def isMedian (t : Triangle) (M : Point) : Prop :=
  M.x = (t.A.x + t.B.x) / 2 ∧ M.y = (t.A.y + t.B.y) / 2

def isAngleBisector (t : Triangle) (L : Point) : Prop :=
  -- We can't define this precisely without more geometric functions,
  -- so we'll leave it as an axiom for now
  True

theorem triangle_point_C (t : Triangle) (M L : Point) :
  t.A = Point.mk 2 8 →
  M = Point.mk 4 11 →
  L = Point.mk 6 6 →
  isMedian t M →
  isAngleBisector t L →
  t.C = Point.mk 14 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_point_C_l2762_276265


namespace NUMINAMATH_CALUDE_sequence_general_term_l2762_276232

def sequence_a : ℕ → ℤ
  | 0 => 3
  | 1 => 9
  | (n + 2) => 4 * sequence_a (n + 1) - 3 * sequence_a n - 4 * (n + 2) + 2

theorem sequence_general_term (n : ℕ) : 
  sequence_a n = 3^n + n^2 + 3*n + 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2762_276232


namespace NUMINAMATH_CALUDE_elementary_symmetric_polynomials_l2762_276253

variable (x y z : ℝ)

/-- Elementary symmetric polynomial of degree 1 -/
def σ₁ (x y z : ℝ) : ℝ := x + y + z

/-- Elementary symmetric polynomial of degree 2 -/
def σ₂ (x y z : ℝ) : ℝ := x*y + y*z + z*x

/-- Elementary symmetric polynomial of degree 3 -/
def σ₃ (x y z : ℝ) : ℝ := x*y*z

theorem elementary_symmetric_polynomials (x y z : ℝ) :
  ((x + y) * (y + z) * (x + z) = σ₂ x y z * σ₁ x y z - σ₃ x y z) ∧
  (x^3 + y^3 + z^3 - 3*x*y*z = σ₁ x y z * (σ₁ x y z^2 - 3 * σ₂ x y z)) ∧
  (x^3 + y^3 = σ₁ x y 0^3 - 3 * σ₁ x y 0 * σ₂ x y 0) ∧
  ((x^2 + y^2) * (y^2 + z^2) * (x^2 + z^2) = 
    σ₁ x y z^2 * σ₂ x y z^2 + 4 * σ₁ x y z * σ₂ x y z * σ₃ x y z - 
    2 * σ₂ x y z^3 - 2 * σ₁ x y z^3 * σ₃ x y z - σ₃ x y z^2) ∧
  (x^4 + y^4 + z^4 = 
    σ₁ x y z^4 - 4 * σ₁ x y z^2 * σ₂ x y z + 2 * σ₂ x y z^2 + 4 * σ₁ x y z * σ₃ x y z) :=
by sorry

end NUMINAMATH_CALUDE_elementary_symmetric_polynomials_l2762_276253


namespace NUMINAMATH_CALUDE_other_asymptote_equation_l2762_276254

/-- A hyperbola with given asymptote and foci x-coordinate -/
structure Hyperbola where
  asymptote : ℝ → ℝ
  foci_x : ℝ
  asymptote_eq : asymptote = fun x ↦ 2 * x + 3
  foci_x_eq : foci_x = 7

/-- The other asymptote of the hyperbola -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ :=
  fun x ↦ -2 * x + 31

/-- Theorem stating that the other asymptote has the correct equation -/
theorem other_asymptote_equation (h : Hyperbola) :
  other_asymptote h = fun x ↦ -2 * x + 31 := by
  sorry


end NUMINAMATH_CALUDE_other_asymptote_equation_l2762_276254


namespace NUMINAMATH_CALUDE_coefficient_of_n_l2762_276248

theorem coefficient_of_n (n : ℤ) : 
  (∃ (values : Finset ℤ), 
    (∀ m ∈ values, 1 < 4 * m + 7 ∧ 4 * m + 7 < 40) ∧ 
    Finset.card values = 10) → 
  (∃ k : ℤ, ∀ m : ℤ, 4 * m + 7 = k * m + 7 → k = 4) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_n_l2762_276248


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l2762_276296

/-- Given 6 consecutive odd numbers whose product is 135135, prove their sum is 48 -/
theorem consecutive_odd_numbers_sum (a b c d e f : ℕ) : 
  (a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f) →  -- consecutive
  (∃ k, a = 2*k + 1) →  -- a is odd
  (b = a + 2) → (c = b + 2) → (d = c + 2) → (e = d + 2) → (f = e + 2) →  -- consecutive odd numbers
  (a * b * c * d * e * f = 135135) →  -- product is 135135
  (a + b + c + d + e + f = 48) :=  -- sum is 48
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l2762_276296


namespace NUMINAMATH_CALUDE_height_prediction_at_10_l2762_276216

/-- Represents a linear regression model for height vs age -/
structure HeightModel where
  slope : ℝ
  intercept : ℝ

/-- Calculates the predicted height for a given age using the model -/
def predictHeight (model : HeightModel) (age : ℝ) : ℝ :=
  model.slope * age + model.intercept

/-- Defines what it means for a prediction to be "around" a value -/
def isAround (predicted : ℝ) (target : ℝ) (tolerance : ℝ) : Prop :=
  abs (predicted - target) ≤ tolerance

theorem height_prediction_at_10 (model : HeightModel) 
  (h1 : model.slope = 7.19) 
  (h2 : model.intercept = 73.93) : 
  ∃ (tolerance : ℝ), tolerance > 0 ∧ isAround (predictHeight model 10) 145.83 tolerance :=
sorry

end NUMINAMATH_CALUDE_height_prediction_at_10_l2762_276216


namespace NUMINAMATH_CALUDE_bug_path_tiles_l2762_276206

-- Define the rectangle's dimensions
def width : ℕ := 12
def length : ℕ := 20

-- Define the function to calculate the number of tiles visited
def tilesVisited (w l : ℕ) : ℕ := w + l - Nat.gcd w l

-- Theorem statement
theorem bug_path_tiles : tilesVisited width length = 28 := by
  sorry

end NUMINAMATH_CALUDE_bug_path_tiles_l2762_276206


namespace NUMINAMATH_CALUDE_golf_strokes_over_par_l2762_276200

/-- Calculates the number of strokes over par for a golfer -/
def strokes_over_par (holes : ℕ) (avg_strokes_per_hole : ℕ) (par_per_hole : ℕ) : ℕ :=
  (holes * avg_strokes_per_hole) - (holes * par_per_hole)

theorem golf_strokes_over_par :
  strokes_over_par 9 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_golf_strokes_over_par_l2762_276200


namespace NUMINAMATH_CALUDE_five_line_configurations_l2762_276207

/-- Represents a configuration of five lines in a plane -/
structure LineConfiguration where
  /-- The number of intersection points -/
  intersections : ℕ
  /-- The number of sets of parallel lines -/
  parallel_sets : ℕ

/-- The total count is the sum of intersection points and parallel sets -/
def total_count (config : LineConfiguration) : ℕ :=
  config.intersections + config.parallel_sets

/-- Possible configurations of five lines in a plane -/
def possible_configurations : List LineConfiguration :=
  [
    ⟨0, 1⟩,  -- All 5 lines parallel
    ⟨4, 1⟩,  -- 4 parallel lines and 1 intersecting
    ⟨6, 2⟩,  -- Two sets of parallel lines (2 and 3)
    ⟨7, 1⟩,  -- 3 parallel lines and 2 intersecting
    ⟨8, 2⟩,  -- Two pairs of parallel lines
    ⟨9, 1⟩,  -- 1 pair of parallel lines
    ⟨10, 0⟩  -- No parallel lines
  ]

theorem five_line_configurations :
  (possible_configurations.map total_count).toFinset = {1, 5, 8, 10} := by sorry

end NUMINAMATH_CALUDE_five_line_configurations_l2762_276207


namespace NUMINAMATH_CALUDE_heaviest_person_l2762_276293

def weight_problem (A D T V M : ℕ) : Prop :=
  A + D = 82 ∧
  D + T = 74 ∧
  T + V = 75 ∧
  V + M = 65 ∧
  M + A = 62

theorem heaviest_person (A D T V M : ℕ) 
  (h : weight_problem A D T V M) : 
  V = 43 ∧ V ≥ A ∧ V ≥ D ∧ V ≥ T ∧ V ≥ M :=
by
  sorry

#check heaviest_person

end NUMINAMATH_CALUDE_heaviest_person_l2762_276293


namespace NUMINAMATH_CALUDE_paving_stones_required_l2762_276274

-- Define the dimensions of the courtyard and paving stone
def courtyard_length : ℝ := 158.5
def courtyard_width : ℝ := 35.4
def stone_length : ℝ := 3.2
def stone_width : ℝ := 2.7

-- Define the theorem
theorem paving_stones_required :
  ∃ (n : ℕ), n = 650 ∧ 
  (n : ℝ) * (stone_length * stone_width) ≥ courtyard_length * courtyard_width ∧
  ∀ (m : ℕ), (m : ℝ) * (stone_length * stone_width) ≥ courtyard_length * courtyard_width → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_paving_stones_required_l2762_276274


namespace NUMINAMATH_CALUDE_lisa_to_total_ratio_l2762_276247

def total_earnings : ℝ := 60

def lisa_earnings (l : ℝ) : Prop := 
  ∃ (j t : ℝ), l + j + t = total_earnings ∧ t = l / 2 ∧ l = t + 15

theorem lisa_to_total_ratio : 
  ∀ l : ℝ, lisa_earnings l → l / total_earnings = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_lisa_to_total_ratio_l2762_276247


namespace NUMINAMATH_CALUDE_sequence_problem_l2762_276211

/-- Arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) : ℕ → ℤ
  | 0 => a
  | n + 1 => arithmeticSequence a d n + d

/-- Geometric sequence with first term a and common ratio r -/
def geometricSequence (a r : ℤ) : ℕ → ℤ
  | 0 => a
  | n + 1 => geometricSequence a r n * r

theorem sequence_problem :
  (arithmeticSequence 12 4 3 = 24) ∧
  (arithmeticSequence 12 4 4 = 28) ∧
  (geometricSequence 2 2 3 = 16) ∧
  (geometricSequence 2 2 4 = 32) := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l2762_276211


namespace NUMINAMATH_CALUDE_jane_inspected_five_eighths_l2762_276227

/-- Represents the fraction of products inspected by Jane given the total rejection rate,
    John's rejection rate, and Jane's rejection rate. -/
def jane_inspection_fraction (total_rejection_rate john_rejection_rate jane_rejection_rate : ℚ) : ℚ :=
  5 / 8

/-- Theorem stating that given the specified rejection rates, Jane inspected 5/8 of the products. -/
theorem jane_inspected_five_eighths
  (total_rejection_rate : ℚ)
  (john_rejection_rate : ℚ)
  (jane_rejection_rate : ℚ)
  (h_total : total_rejection_rate = 75 / 10000)
  (h_john : john_rejection_rate = 5 / 1000)
  (h_jane : jane_rejection_rate = 9 / 1000) :
  jane_inspection_fraction total_rejection_rate john_rejection_rate jane_rejection_rate = 5 / 8 := by
  sorry

#eval jane_inspection_fraction (75/10000) (5/1000) (9/1000)

end NUMINAMATH_CALUDE_jane_inspected_five_eighths_l2762_276227


namespace NUMINAMATH_CALUDE_tan_C_when_a_neg_eight_min_tan_C_l2762_276203

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the condition for tan A and tan B
def roots_condition (t : Triangle) (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + a*x + 4 = 0 ∧ y^2 + a*y + 4 = 0 ∧ 
  x = Real.tan t.A ∧ y = Real.tan t.B

-- Theorem 1: When a = -8, tan C = 8/3
theorem tan_C_when_a_neg_eight (t : Triangle) (a : ℝ) 
  (h : roots_condition t a) (h_a : a = -8) : 
  Real.tan t.C = 8/3 := by sorry

-- Theorem 2: Minimum value of tan C is 4/3, occurring when tan A = tan B = 2
theorem min_tan_C (t : Triangle) (a : ℝ) 
  (h : roots_condition t a) : 
  ∃ (t_min : Triangle), 
    (∀ t' : Triangle, roots_condition t' a → Real.tan t_min.C ≤ Real.tan t'.C) ∧
    Real.tan t_min.C = 4/3 ∧ 
    Real.tan t_min.A = 2 ∧ 
    Real.tan t_min.B = 2 := by sorry

end NUMINAMATH_CALUDE_tan_C_when_a_neg_eight_min_tan_C_l2762_276203


namespace NUMINAMATH_CALUDE_total_ants_count_l2762_276266

/-- The total number of ants employed for all tasks in the construction site. -/
def total_ants : ℕ :=
  let red_carrying := 413
  let black_carrying := 487
  let yellow_carrying := 360
  let red_digging := 356
  let black_digging := 518
  let green_digging := 250
  let red_assembling := 298
  let black_assembling := 392
  let blue_assembling := 200
  let black_food := black_carrying / 4
  red_carrying + black_carrying + yellow_carrying +
  red_digging + black_digging + green_digging +
  red_assembling + black_assembling + blue_assembling -
  black_food

/-- Theorem stating that the total number of ants employed for all tasks is 3153. -/
theorem total_ants_count : total_ants = 3153 := by
  sorry

end NUMINAMATH_CALUDE_total_ants_count_l2762_276266


namespace NUMINAMATH_CALUDE_smallest_divisible_by_fractions_l2762_276231

def is_divisible_by_fraction (n : ℕ) (a b : ℕ) : Prop :=
  ∃ k : ℕ, n * b = k * a

theorem smallest_divisible_by_fractions :
  ∀ n : ℕ, n > 0 →
    (is_divisible_by_fraction n 8 33 ∧
     is_divisible_by_fraction n 7 22 ∧
     is_divisible_by_fraction n 15 26) →
    n ≥ 120 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_fractions_l2762_276231


namespace NUMINAMATH_CALUDE_sandys_shopping_money_l2762_276212

theorem sandys_shopping_money (watch_price : ℝ) (money_left : ℝ) (spent_percentage : ℝ) : 
  watch_price = 50 →
  money_left = 210 →
  spent_percentage = 0.3 →
  ∃ (total_money : ℝ), 
    total_money = watch_price + (money_left / (1 - spent_percentage)) ∧
    total_money = 350 :=
by sorry

end NUMINAMATH_CALUDE_sandys_shopping_money_l2762_276212


namespace NUMINAMATH_CALUDE_a_upper_bound_l2762_276255

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 5

-- State the theorem
theorem a_upper_bound (a : ℝ) :
  (∀ x y, 5/2 < x ∧ x < y → f a x < f a y) →
  a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_a_upper_bound_l2762_276255


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2762_276292

/-- In a Cartesian coordinate system, the coordinates of the point (11, 9) with respect to the origin are (11, 9). -/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (11, 9)
  P = P :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2762_276292


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2762_276239

theorem division_remainder_problem (L S : ℕ) : 
  L - S = 1325 → 
  L = 1650 → 
  ∃ (R : ℕ), L = 5 * S + R ∧ R < S → 
  R = 25 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2762_276239


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2762_276202

theorem simplify_sqrt_expression : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 98 / Real.sqrt 56) = (3 + Real.sqrt 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2762_276202


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l2762_276230

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1)^2 + 2

theorem extreme_values_of_f :
  ∃ (a b c : ℝ), 
    (a = 0 ∧ b = 1 ∧ c = -1) ∧
    (∀ x : ℝ, f x ≥ 2) ∧
    (f a = 3 ∧ f b = 2 ∧ f c = 2) ∧
    (∀ x : ℝ, x ≠ a ∧ x ≠ b ∧ x ≠ c → f x < 3) :=
by
  sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l2762_276230


namespace NUMINAMATH_CALUDE_triangle_xz_interval_l2762_276288

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the point W on YZ
def W (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := sorry

-- Define the angle bisector
def is_angle_bisector (t : Triangle) (w : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem triangle_xz_interval (t : Triangle) :
  length t.X t.Y = 8 →
  is_angle_bisector t (W t) →
  length (W t) t.Z = 5 →
  perimeter t = 24 →
  ∃ m n : ℝ, m < n ∧ 
    (∀ xz : ℝ, m < xz ∧ xz < n ↔ length t.X t.Z = xz) ∧
    m + n = 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_xz_interval_l2762_276288


namespace NUMINAMATH_CALUDE_vector_expression_simplification_l2762_276246

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

theorem vector_expression_simplification (a b : V) :
  (2/3 : ℝ) • ((4 • a - 3 • b) + (1/3 : ℝ) • b - (1/4 : ℝ) • (6 • a - 7 • b)) =
  (5/3 : ℝ) • a - (11/18 : ℝ) • b := by sorry

end NUMINAMATH_CALUDE_vector_expression_simplification_l2762_276246


namespace NUMINAMATH_CALUDE_sundae_cost_theorem_l2762_276277

/-- The cost of ice cream in dollars -/
def ice_cream_cost : ℚ := 2

/-- The cost of one topping in dollars -/
def topping_cost : ℚ := 1/2

/-- The number of toppings on the sundae -/
def num_toppings : ℕ := 10

/-- The total cost of a sundae with given number of toppings -/
def sundae_cost (ice_cream : ℚ) (topping : ℚ) (num : ℕ) : ℚ :=
  ice_cream + topping * num

theorem sundae_cost_theorem :
  sundae_cost ice_cream_cost topping_cost num_toppings = 7 := by
  sorry

end NUMINAMATH_CALUDE_sundae_cost_theorem_l2762_276277


namespace NUMINAMATH_CALUDE_finance_equation_solution_l2762_276297

/-- Given the equation fp - w = 20000, where f = 4 and w = 10 + 200i, prove that p = 5002.5 + 50i. -/
theorem finance_equation_solution (f w p : ℂ) : 
  f = 4 → w = 10 + 200 * Complex.I → f * p - w = 20000 → p = 5002.5 + 50 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_finance_equation_solution_l2762_276297


namespace NUMINAMATH_CALUDE_train_speed_l2762_276262

/-- The speed of a train given its length and time to cross a point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 800) (h2 : time = 10) :
  length / time = 80 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2762_276262


namespace NUMINAMATH_CALUDE_total_friends_l2762_276276

theorem total_friends (initial_friends additional_friends : ℕ) 
  (h1 : initial_friends = 4) 
  (h2 : additional_friends = 3) : 
  initial_friends + additional_friends = 7 := by
    sorry

end NUMINAMATH_CALUDE_total_friends_l2762_276276


namespace NUMINAMATH_CALUDE_circle_increase_l2762_276234

theorem circle_increase (r : ℝ) (hr : r > 0) :
  let new_radius := 2.5 * r
  let area_increase_percent := ((π * new_radius^2 - π * r^2) / (π * r^2)) * 100
  let circumference_increase_percent := ((2 * π * new_radius - 2 * π * r) / (2 * π * r)) * 100
  area_increase_percent = 525 ∧ circumference_increase_percent = 150 := by
sorry


end NUMINAMATH_CALUDE_circle_increase_l2762_276234


namespace NUMINAMATH_CALUDE_gummy_worms_problem_l2762_276225

theorem gummy_worms_problem (initial_amount : ℕ) : 
  (((initial_amount / 2) / 2) / 2) / 2 = 4 → initial_amount = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_gummy_worms_problem_l2762_276225


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l2762_276256

theorem unique_solution_to_equation :
  ∃! z : ℝ, (z + 2)^4 + (2 - z)^4 = 258 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l2762_276256


namespace NUMINAMATH_CALUDE_banana_arrangements_l2762_276236

def word : String := "BANANA"

def letter_count : Nat := word.length

def b_count : Nat := 1
def a_count : Nat := 3
def n_count : Nat := 2

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

theorem banana_arrangements :
  (factorial letter_count) / (factorial b_count * factorial a_count * factorial n_count) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l2762_276236


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_13_l2762_276250

theorem greatest_three_digit_multiple_of_13 : 
  ∃ n : ℕ, n = 988 ∧ 
  n % 13 = 0 ∧
  n ≥ 100 ∧ n < 1000 ∧
  ∀ m : ℕ, m % 13 = 0 → m ≥ 100 → m < 1000 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_13_l2762_276250


namespace NUMINAMATH_CALUDE_division_addition_equality_l2762_276219

theorem division_addition_equality : 0.2 / 0.005 + 0.1 = 40.1 := by
  sorry

end NUMINAMATH_CALUDE_division_addition_equality_l2762_276219


namespace NUMINAMATH_CALUDE_line_points_b_plus_one_l2762_276269

/-- Given a line y = 0.75x + 1 and three points on the line, prove that b + 1 = 5 -/
theorem line_points_b_plus_one (b a : ℝ) : 
  (b = 0.75 * 4 + 1) →  -- Point (4, b) on the line
  (5 = 0.75 * a + 1) →  -- Point (a, 5) on the line
  (b + 1 = 0.75 * a + 1) →  -- Point (a, b + 1) on the line
  b + 1 = 5 :=
by sorry

end NUMINAMATH_CALUDE_line_points_b_plus_one_l2762_276269


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2762_276251

theorem polynomial_expansion (z : ℝ) :
  (3 * z^3 + 2 * z^2 - 4 * z + 1) * (4 * z^4 - 3 * z^2 + 2) =
  12 * z^7 + 8 * z^6 - 25 * z^5 - 2 * z^4 + 18 * z^3 + z^2 - 8 * z + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2762_276251


namespace NUMINAMATH_CALUDE_prob_sector_1_eq_prob_sector_8_prob_consecutive_sectors_correct_l2762_276280

-- Define the number of sectors and the number of played sectors
def total_sectors : ℕ := 13
def played_sectors : ℕ := 6

-- Define a function to calculate the probability of a specific sector being played
def prob_sector_played (sector : ℕ) : ℚ :=
  played_sectors / total_sectors

-- Theorem for part (a)
theorem prob_sector_1_eq_prob_sector_8 :
  prob_sector_played 1 = prob_sector_played 8 :=
sorry

-- Define a function to calculate the probability of sectors 1 to 6 being played consecutively
def prob_consecutive_sectors : ℚ :=
  (7^5 : ℚ) / (13^6 : ℚ)

-- Theorem for part (b)
theorem prob_consecutive_sectors_correct :
  prob_consecutive_sectors = (7^5 : ℚ) / (13^6 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_prob_sector_1_eq_prob_sector_8_prob_consecutive_sectors_correct_l2762_276280


namespace NUMINAMATH_CALUDE_machine_retail_price_l2762_276283

/-- The retail price of a machine -/
def retail_price : ℝ := 132

/-- The wholesale price of the machine -/
def wholesale_price : ℝ := 99

/-- The discount rate applied to the retail price -/
def discount_rate : ℝ := 0.1

/-- The profit rate as a percentage of the wholesale price -/
def profit_rate : ℝ := 0.2

theorem machine_retail_price :
  retail_price = wholesale_price * (1 + profit_rate) / (1 - discount_rate) :=
by sorry

end NUMINAMATH_CALUDE_machine_retail_price_l2762_276283


namespace NUMINAMATH_CALUDE_bertolli_farm_tomatoes_bertolli_farm_tomatoes_proof_l2762_276273

theorem bertolli_farm_tomatoes : ℕ → Prop :=
  fun tomatoes =>
    let corn : ℕ := 4112
    let onions : ℕ := 985
    let onions_difference : ℕ := 5200
    onions = tomatoes + corn - onions_difference →
    tomatoes = 2073

-- The proof is omitted
theorem bertolli_farm_tomatoes_proof : bertolli_farm_tomatoes 2073 := by
  sorry

end NUMINAMATH_CALUDE_bertolli_farm_tomatoes_bertolli_farm_tomatoes_proof_l2762_276273


namespace NUMINAMATH_CALUDE_slope_of_line_l2762_276295

/-- The slope of a line given by the equation (x/4) - (y/3) = -2 is -3/4 -/
theorem slope_of_line (x y : ℝ) : (x / 4 - y / 3 = -2) → (y = (-3 / 4) * x - 6) := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l2762_276295


namespace NUMINAMATH_CALUDE_average_age_combined_rooms_l2762_276258

theorem average_age_combined_rooms (room_a_count room_b_count room_c_count : ℕ)
                                   (room_a_avg room_b_avg room_c_avg : ℝ)
                                   (h1 : room_a_count = 8)
                                   (h2 : room_b_count = 5)
                                   (h3 : room_c_count = 7)
                                   (h4 : room_a_avg = 35)
                                   (h5 : room_b_avg = 30)
                                   (h6 : room_c_avg = 50) :
  let total_count := room_a_count + room_b_count + room_c_count
  let total_age := room_a_count * room_a_avg + room_b_count * room_b_avg + room_c_count * room_c_avg
  total_age / total_count = 39 := by
sorry

end NUMINAMATH_CALUDE_average_age_combined_rooms_l2762_276258


namespace NUMINAMATH_CALUDE_newspaper_pieces_not_all_found_l2762_276208

theorem newspaper_pieces_not_all_found :
  ¬∃ (k p v : ℕ), 1988 = k + 4 * p + 8 * v ∧ k > 0 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_pieces_not_all_found_l2762_276208


namespace NUMINAMATH_CALUDE_solve_parking_lot_l2762_276257

def parking_lot (num_bikes : ℕ) (total_wheels : ℕ) (wheels_per_car : ℕ) (wheels_per_bike : ℕ) : Prop :=
  ∃ (num_cars : ℕ), 
    num_cars * wheels_per_car + num_bikes * wheels_per_bike = total_wheels

theorem solve_parking_lot : 
  parking_lot 5 66 4 2 → ∃ (num_cars : ℕ), num_cars = 14 := by
  sorry

end NUMINAMATH_CALUDE_solve_parking_lot_l2762_276257


namespace NUMINAMATH_CALUDE_unique_base_number_for_16_factorial_l2762_276298

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem unique_base_number_for_16_factorial :
  ∃! b : ℕ+, b > 1 ∧ (factorial 16 % (b : ℕ)^6 = 0) ∧ (factorial 16 % (b : ℕ)^7 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_base_number_for_16_factorial_l2762_276298


namespace NUMINAMATH_CALUDE_lending_amount_calculation_l2762_276245

theorem lending_amount_calculation (P : ℝ) 
  (h1 : (P * 0.115 * 3) - (P * 0.10 * 3) = 157.5) : P = 3500 := by
  sorry

end NUMINAMATH_CALUDE_lending_amount_calculation_l2762_276245


namespace NUMINAMATH_CALUDE_coins_on_side_for_36_circumference_l2762_276268

/-- The number of coins on one side of a square arrangement, given the total number of coins on the circumference. -/
def coins_on_one_side (circumference_coins : ℕ) : ℕ :=
  (circumference_coins + 4) / 4

/-- Theorem stating that for a square arrangement of coins with 36 coins on the circumference, there are 10 coins on one side. -/
theorem coins_on_side_for_36_circumference :
  coins_on_one_side 36 = 10 := by
  sorry

#eval coins_on_one_side 36  -- This should output 10

end NUMINAMATH_CALUDE_coins_on_side_for_36_circumference_l2762_276268


namespace NUMINAMATH_CALUDE_sons_age_l2762_276272

theorem sons_age (father_age son_age : ℕ) : 
  father_age = 3 * son_age →
  (father_age - 8) = 4 * (son_age - 8) →
  son_age = 24 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2762_276272


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2762_276218

-- Define the inequality function
def f (x : ℝ) : Prop := (3 * x + 5) / (x - 1) > x

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < -1 ∨ (1 < x ∧ x < 5)

-- Theorem statement
theorem inequality_solution_set : 
  ∀ x : ℝ, x ≠ 1 → (f x ↔ solution_set x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2762_276218


namespace NUMINAMATH_CALUDE_boxes_loaded_is_100_l2762_276284

/-- The number of boxes loaded on a truck given its capacity and other items --/
def boxes_loaded (truck_capacity : ℕ) (box_weight crate_weight sack_weight bag_weight : ℕ)
  (num_crates num_sacks num_bags : ℕ) : ℕ :=
  (truck_capacity - (crate_weight * num_crates + sack_weight * num_sacks + bag_weight * num_bags)) / box_weight

/-- Theorem stating that 100 boxes were loaded given the specific conditions --/
theorem boxes_loaded_is_100 :
  boxes_loaded 13500 100 60 50 40 10 50 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_boxes_loaded_is_100_l2762_276284


namespace NUMINAMATH_CALUDE_self_reciprocal_set_l2762_276286

def self_reciprocal (x : ℝ) : Prop := x ≠ 0 ∧ x = 1 / x

theorem self_reciprocal_set :
  ∃ (S : Set ℝ), (∀ x, x ∈ S ↔ self_reciprocal x) ∧ S = {1, -1} :=
sorry

end NUMINAMATH_CALUDE_self_reciprocal_set_l2762_276286


namespace NUMINAMATH_CALUDE_percentage_class_a_is_40_percent_l2762_276209

/-- Represents a school with three classes -/
structure School where
  total_students : ℕ
  class_a : ℕ
  class_b : ℕ
  class_c : ℕ

/-- Calculates the percentage of students in class A -/
def percentage_class_a (s : School) : ℚ :=
  (s.class_a : ℚ) / (s.total_students : ℚ) * 100

/-- Theorem stating the percentage of students in class A -/
theorem percentage_class_a_is_40_percent (s : School) 
  (h1 : s.total_students = 80)
  (h2 : s.class_b = s.class_a - 21)
  (h3 : s.class_c = 37)
  (h4 : s.total_students = s.class_a + s.class_b + s.class_c) :
  percentage_class_a s = 40 := by
  sorry

#eval percentage_class_a {
  total_students := 80,
  class_a := 32,
  class_b := 11,
  class_c := 37
}

end NUMINAMATH_CALUDE_percentage_class_a_is_40_percent_l2762_276209


namespace NUMINAMATH_CALUDE_y_percentage_more_than_z_l2762_276223

/-- Proves that given the conditions, y gets 20% more than z -/
theorem y_percentage_more_than_z (total : ℝ) (z_share : ℝ) (x_more_than_y : ℝ) :
  total = 1480 →
  z_share = 400 →
  x_more_than_y = 0.25 →
  (((total - z_share) / (2 + x_more_than_y) - z_share) / z_share) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_y_percentage_more_than_z_l2762_276223


namespace NUMINAMATH_CALUDE_seokjins_uncle_age_l2762_276235

/-- Seokjin's uncle's age when Seokjin is 12 years old -/
def uncles_age (mothers_age_at_birth : ℕ) (age_difference : ℕ) : ℕ :=
  mothers_age_at_birth + 12 - age_difference

/-- Theorem stating that Seokjin's uncle's age is 41 when Seokjin is 12 -/
theorem seokjins_uncle_age :
  uncles_age 32 3 = 41 := by
  sorry

end NUMINAMATH_CALUDE_seokjins_uncle_age_l2762_276235


namespace NUMINAMATH_CALUDE_daily_water_evaporation_l2762_276238

/-- Given a glass with initial water amount, evaporation period, and total evaporation percentage,
    calculate the amount of water that evaporates each day. -/
theorem daily_water_evaporation
  (initial_water : ℝ)
  (evaporation_period : ℕ)
  (total_evaporation_percentage : ℝ)
  (h1 : initial_water = 25)
  (h2 : evaporation_period = 10)
  (h3 : total_evaporation_percentage = 1.6)
  : (initial_water * total_evaporation_percentage / 100) / evaporation_period = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_daily_water_evaporation_l2762_276238


namespace NUMINAMATH_CALUDE_inequality_systems_solution_l2762_276287

theorem inequality_systems_solution :
  (∀ x : ℝ, (2 * x ≥ x - 1 ∧ 4 * x + 10 > x + 1) ↔ x ≥ -1) ∧
  (∀ x : ℝ, (2 * x - 7 < 5 - 2 * x ∧ x / 4 - 1 ≤ (x - 1) / 2) ↔ -2 ≤ x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_systems_solution_l2762_276287


namespace NUMINAMATH_CALUDE_negation_of_implication_l2762_276282

theorem negation_of_implication (x y : ℝ) :
  (¬(x = y → Real.sqrt x = Real.sqrt y)) ↔ (x ≠ y → Real.sqrt x ≠ Real.sqrt y) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2762_276282


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l2762_276217

theorem lcm_hcf_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 → 
  Nat.gcd a b = 83 → 
  a = 210 → 
  b = 913 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l2762_276217


namespace NUMINAMATH_CALUDE_range_of_a_l2762_276214

theorem range_of_a (p q : ℝ → Prop) (a : ℝ) 
  (h1 : ∀ x, p x → q x)
  (h2 : ∃ x, q x ∧ ¬(p x))
  (hp : ∀ x, p x ↔ x^2 - 2*x - 3 < 0)
  (hq : ∀ x, q x ↔ x > a) :
  a ≤ -1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2762_276214


namespace NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l2762_276259

def is_pure_imaginary (z : ℂ) : Prop := ∃ b : ℝ, z = (0 : ℝ) + b * Complex.I

theorem a_zero_necessary_not_sufficient :
  (∀ a b : ℝ, is_pure_imaginary (Complex.ofReal a + Complex.I * b) → a = 0) ∧
  ¬(∀ a b : ℝ, a = 0 → is_pure_imaginary (Complex.ofReal a + Complex.I * b)) :=
by sorry

end NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l2762_276259


namespace NUMINAMATH_CALUDE_power_equation_solution_l2762_276226

def solution_set : Set ℝ := {-3, 1, 2}

theorem power_equation_solution (x : ℝ) : 
  (2*x - 3)^(x + 3) = 1 ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2762_276226
