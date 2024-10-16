import Mathlib

namespace NUMINAMATH_CALUDE_g_is_two_x_l4185_418597

/-- A function g: ℝ → ℝ satisfying certain conditions -/
def g_function (g : ℝ → ℝ) : Prop :=
  g 1 = 2 ∧ ∀ x y : ℝ, g (x^2 - y^2) = (x - y) * (g x + g y)

/-- Theorem stating that g(x) = 2x for all real x -/
theorem g_is_two_x (g : ℝ → ℝ) (h : g_function g) : ∀ x : ℝ, g x = 2 * x :=
sorry

end NUMINAMATH_CALUDE_g_is_two_x_l4185_418597


namespace NUMINAMATH_CALUDE_starting_lineup_count_l4185_418552

/-- Represents a football team -/
structure FootballTeam where
  total_members : ℕ
  offensive_linemen : ℕ
  hm : offensive_linemen ≤ total_members

/-- Calculates the number of ways to choose a starting lineup -/
def starting_lineup_combinations (team : FootballTeam) : ℕ :=
  team.offensive_linemen * (team.total_members - 1) * (team.total_members - 2) * (team.total_members - 3)

/-- Theorem stating the number of ways to choose a starting lineup for the given team -/
theorem starting_lineup_count (team : FootballTeam) 
  (h1 : team.total_members = 12) 
  (h2 : team.offensive_linemen = 4) : 
  starting_lineup_combinations team = 3960 := by
  sorry

#eval starting_lineup_combinations ⟨12, 4, by norm_num⟩

end NUMINAMATH_CALUDE_starting_lineup_count_l4185_418552


namespace NUMINAMATH_CALUDE_parallelogram_area_specific_parallelogram_area_l4185_418505

/-- The area of a parallelogram with given base, side length, and included angle --/
theorem parallelogram_area (base : ℝ) (side : ℝ) (angle : ℝ) : 
  base > 0 → side > 0 → 0 < angle ∧ angle < π →
  abs (base * side * Real.sin angle - 498.465) < 0.001 := by
  sorry

/-- Specific instance of the parallelogram area theorem --/
theorem specific_parallelogram_area : 
  abs (22 * 25 * Real.sin (65 * π / 180) - 498.465) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_specific_parallelogram_area_l4185_418505


namespace NUMINAMATH_CALUDE_cash_realized_specific_case_l4185_418571

/-- Given a total amount including brokerage and a brokerage rate,
    calculates the cash realized without brokerage -/
def cash_realized (total : ℚ) (brokerage_rate : ℚ) : ℚ :=
  total / (1 + brokerage_rate)

/-- Theorem stating that given the specific conditions of the problem,
    the cash realized is equal to 43200/401 -/
theorem cash_realized_specific_case :
  cash_realized 108 (1/400) = 43200/401 := by
  sorry

end NUMINAMATH_CALUDE_cash_realized_specific_case_l4185_418571


namespace NUMINAMATH_CALUDE_total_miles_walked_l4185_418598

-- Define the number of islands
def num_islands : ℕ := 4

-- Define the number of days to explore each island
def days_per_island : ℚ := 3/2

-- Define the daily walking distances for each type of island
def miles_per_day_type1 : ℕ := 20
def miles_per_day_type2 : ℕ := 25

-- Define the number of islands for each type
def num_islands_type1 : ℕ := 2
def num_islands_type2 : ℕ := 2

-- Theorem to prove
theorem total_miles_walked :
  (num_islands_type1 * miles_per_day_type1 + num_islands_type2 * miles_per_day_type2) * days_per_island = 135 := by
  sorry


end NUMINAMATH_CALUDE_total_miles_walked_l4185_418598


namespace NUMINAMATH_CALUDE_factorization_equality_l4185_418582

theorem factorization_equality (x : ℝ) : x * (x - 2) + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4185_418582


namespace NUMINAMATH_CALUDE_function_and_inequality_l4185_418515

/-- Given a function f(x) = (ax+b)/(x-2) where f(x) - x + 12 = 0 has roots 3 and 4,
    prove the form of f(x) and the solution set of f(x) < k for k > 1 -/
theorem function_and_inequality (a b : ℝ) (h1 : ∀ x : ℝ, x ≠ 2 → (a * x + b) / (x - 2) - x + 12 = 0) 
    (h2 : (a * 3 + b) / (3 - 2) - 3 + 12 = 0) (h3 : (a * 4 + b) / (4 - 2) - 4 + 12 = 0) :
  (∀ x : ℝ, x ≠ 2 → (a * x + b) / (x - 2) = (-x + 2) / (x - 2)) ∧
  (∀ k : ℝ, k > 1 →
    (1 < k ∧ k < 2 → {x : ℝ | (-x + 2) / (x - 2) < k} = {x : ℝ | 1 < x ∧ x < k} ∪ {x : ℝ | x > 2}) ∧
    (k = 2 → {x : ℝ | (-x + 2) / (x - 2) < k} = {x : ℝ | 1 < x ∧ x < 2} ∪ {x : ℝ | x > 2}) ∧
    (k > 2 → {x : ℝ | (-x + 2) / (x - 2) < k} = {x : ℝ | 1 < x ∧ x < 2} ∪ {x : ℝ | x > k})) :=
by sorry

end NUMINAMATH_CALUDE_function_and_inequality_l4185_418515


namespace NUMINAMATH_CALUDE_bijective_if_injective_or_surjective_finite_sets_l4185_418557

theorem bijective_if_injective_or_surjective_finite_sets
  {X Y : Type} [Fintype X] [Fintype Y]
  (h_card_eq : Fintype.card X = Fintype.card Y)
  (f : X → Y)
  (h_inj_or_surj : Function.Injective f ∨ Function.Surjective f) :
  Function.Bijective f :=
sorry

end NUMINAMATH_CALUDE_bijective_if_injective_or_surjective_finite_sets_l4185_418557


namespace NUMINAMATH_CALUDE_quarter_count_proof_l4185_418548

/-- Represents the types of coins in the collection -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Represents a collection of coins -/
structure CoinCollection where
  coins : List Coin

def CoinCollection.averageValue (c : CoinCollection) : ℚ :=
  sorry

def CoinCollection.addDimes (c : CoinCollection) (n : ℕ) : CoinCollection :=
  sorry

def CoinCollection.countQuarters (c : CoinCollection) : ℕ :=
  sorry

theorem quarter_count_proof (c : CoinCollection) :
  c.averageValue = 15 / 100 →
  (c.addDimes 2).averageValue = 17 / 100 →
  c.countQuarters = 4 :=
sorry

end NUMINAMATH_CALUDE_quarter_count_proof_l4185_418548


namespace NUMINAMATH_CALUDE_bottle_cap_weight_l4185_418531

theorem bottle_cap_weight (caps_per_ounce : ℕ) (total_caps : ℕ) (total_weight : ℕ) :
  caps_per_ounce = 7 →
  total_caps = 2016 →
  total_weight = total_caps / caps_per_ounce →
  total_weight = 288 :=
by sorry

end NUMINAMATH_CALUDE_bottle_cap_weight_l4185_418531


namespace NUMINAMATH_CALUDE_quadratic_equation_from_roots_l4185_418585

theorem quadratic_equation_from_roots (r s : ℝ) 
  (sum_roots : r + s = 12)
  (product_roots : r * s = 27)
  (root_relation : s = 3 * r) : 
  ∀ x : ℝ, x^2 - 12*x + 27 = (x - r) * (x - s) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_roots_l4185_418585


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4185_418501

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  (i^2 * (1 + i)).im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4185_418501


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l4185_418554

theorem smallest_four_digit_multiple_of_18 : 
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 18 = 0 → n ≥ 1008 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l4185_418554


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_negation_l4185_418575

theorem quadratic_inequality_and_negation :
  (∀ x : ℝ, x^2 + 2*x + 3 > 0) ∧
  (¬(∀ x : ℝ, x^2 + 2*x + 3 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_negation_l4185_418575


namespace NUMINAMATH_CALUDE_brianna_cd_purchase_l4185_418529

theorem brianna_cd_purchase (total_money : ℚ) (total_cds : ℚ) (h : total_money > 0) (h' : total_cds > 0) :
  (1 / 4 : ℚ) * total_money = (1 / 4 : ℚ) * (total_cds * (total_money / total_cds)) →
  total_money - (total_cds * (total_money / total_cds)) = 0 := by
sorry

end NUMINAMATH_CALUDE_brianna_cd_purchase_l4185_418529


namespace NUMINAMATH_CALUDE_y_value_l4185_418556

theorem y_value (x z y : ℝ) (h1 : x = 2 * z) (h2 : y = 3 * z - 1) (h3 : x = 40) : y = 59 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l4185_418556


namespace NUMINAMATH_CALUDE_sum_of_cubes_l4185_418581

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) : x^3 + y^3 = 640 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l4185_418581


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l4185_418578

theorem min_value_expression (y : ℝ) (h : y > 2) :
  (y^2 + y + 1) / Real.sqrt (y - 2) ≥ 3 * Real.sqrt 35 :=
by sorry

theorem min_value_achievable :
  ∃ y : ℝ, y > 2 ∧ (y^2 + y + 1) / Real.sqrt (y - 2) = 3 * Real.sqrt 35 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l4185_418578


namespace NUMINAMATH_CALUDE_cereal_box_servings_l4185_418590

def cereal_box_problem (total_cups : ℕ) (cups_per_serving : ℕ) : ℕ :=
  total_cups / cups_per_serving

theorem cereal_box_servings :
  cereal_box_problem 18 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_servings_l4185_418590


namespace NUMINAMATH_CALUDE_groomer_problem_l4185_418508

/-- The number of full-haired dogs a groomer has to dry --/
def num_full_haired_dogs : ℕ := by sorry

theorem groomer_problem :
  let time_short_haired : ℕ := 10  -- minutes to dry a short-haired dog
  let time_full_haired : ℕ := 2 * time_short_haired  -- minutes to dry a full-haired dog
  let num_short_haired : ℕ := 6  -- number of short-haired dogs
  let total_time : ℕ := 4 * 60  -- total time in minutes (4 hours)
  
  num_full_haired_dogs = 
    (total_time - num_short_haired * time_short_haired) / time_full_haired :=
by sorry

end NUMINAMATH_CALUDE_groomer_problem_l4185_418508


namespace NUMINAMATH_CALUDE_triangle_min_angle_le_60_l4185_418568

theorem triangle_min_angle_le_60 (α β γ : ℝ) :
  α + β + γ = 180 ∧ α > 0 ∧ β > 0 ∧ γ > 0 → min α (min β γ) ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_min_angle_le_60_l4185_418568


namespace NUMINAMATH_CALUDE_product_of_even_or_odd_is_even_l4185_418588

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the concept of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the product of two functions
def FunctionProduct (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f x * g x

-- State the theorem
theorem product_of_even_or_odd_is_even 
  (f φ : ℝ → ℝ) 
  (h : (IsEven f ∧ IsEven φ) ∨ (IsOdd f ∧ IsOdd φ)) : 
  IsEven (FunctionProduct f φ) := by
  sorry

end NUMINAMATH_CALUDE_product_of_even_or_odd_is_even_l4185_418588


namespace NUMINAMATH_CALUDE_parallel_planes_condition_l4185_418535

structure GeometricSpace where
  Line : Type
  Plane : Type
  subset : Line → Plane → Prop
  parallel : Line → Plane → Prop
  plane_parallel : Plane → Plane → Prop

variable (S : GeometricSpace)

theorem parallel_planes_condition
  (a b : S.Line) (α β : S.Plane)
  (h1 : S.subset a α)
  (h2 : S.subset b β) :
  (∃ (α' β' : S.Plane), S.plane_parallel α' β' →
    (S.parallel a β' ∧ S.parallel b α')) ∧
  ¬(∀ (α' β' : S.Plane), S.parallel a β' ∧ S.parallel b α' →
    S.plane_parallel α' β') := by
  sorry

end NUMINAMATH_CALUDE_parallel_planes_condition_l4185_418535


namespace NUMINAMATH_CALUDE_polygon_sides_for_900_degrees_l4185_418564

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180° --/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- For a polygon with n sides and sum of interior angles equal to 900°, n = 7 --/
theorem polygon_sides_for_900_degrees (n : ℕ) :
  sum_interior_angles n = 900 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_for_900_degrees_l4185_418564


namespace NUMINAMATH_CALUDE_total_logs_cut_l4185_418511

/-- The number of logs produced by cutting different types of trees -/
theorem total_logs_cut (pine_logs maple_logs walnut_logs oak_logs birch_logs : ℕ)
  (pine_trees maple_trees walnut_trees oak_trees birch_trees : ℕ)
  (h1 : pine_logs = 80)
  (h2 : maple_logs = 60)
  (h3 : walnut_logs = 100)
  (h4 : oak_logs = 90)
  (h5 : birch_logs = 55)
  (h6 : pine_trees = 8)
  (h7 : maple_trees = 3)
  (h8 : walnut_trees = 4)
  (h9 : oak_trees = 7)
  (h10 : birch_trees = 5) :
  pine_logs * pine_trees + maple_logs * maple_trees + walnut_logs * walnut_trees +
  oak_logs * oak_trees + birch_logs * birch_trees = 2125 := by
  sorry

end NUMINAMATH_CALUDE_total_logs_cut_l4185_418511


namespace NUMINAMATH_CALUDE_f_monotonicity_and_max_k_l4185_418579

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 2

theorem f_monotonicity_and_max_k :
  (∀ a ≤ 0, ∀ x y, x < y → f a x < f a y) ∧
  (∀ a > 0, ∀ x y, x < y → 
    ((x < Real.log a ∧ y < Real.log a → f a x > f a y) ∧
     (x > Real.log a ∧ y > Real.log a → f a x < f a y))) ∧
  (∀ k : ℤ, (∀ x > 0, (x - ↑k) * (Real.exp x - 1) + x + 1 > 0) → k ≤ 2) ∧
  (∀ x > 0, (x - 2) * (Real.exp x - 1) + x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_max_k_l4185_418579


namespace NUMINAMATH_CALUDE_triangular_display_total_l4185_418574

/-- Represents a triangular display of cans -/
structure CanDisplay where
  bottom_layer : ℕ
  second_layer : ℕ
  top_layer : ℕ

/-- Calculates the total number of cans in the display -/
def total_cans (d : CanDisplay) : ℕ :=
  sorry

/-- Theorem stating that the specific triangular display contains 165 cans -/
theorem triangular_display_total (d : CanDisplay) 
  (h1 : d.bottom_layer = 30)
  (h2 : d.second_layer = 27)
  (h3 : d.top_layer = 3) :
  total_cans d = 165 := by
  sorry

end NUMINAMATH_CALUDE_triangular_display_total_l4185_418574


namespace NUMINAMATH_CALUDE_quadratic_function_m_values_l4185_418551

theorem quadratic_function_m_values (m : ℝ) :
  (∃ a b c : ℝ, ∀ x, (m^2 - m) * x^(m^2 - 2*m - 1) + (m - 3) * x + m^2 = a * x^2 + b * x + c) →
  (m = 3 ∨ m = -1) ∧
  ((m = 3 → ∀ x, (m^2 - m) * x^(m^2 - 2*m - 1) + (m - 3) * x + m^2 = 6 * x^2 + 9) ∧
   (m = -1 → ∀ x, (m^2 - m) * x^(m^2 - 2*m - 1) + (m - 3) * x + m^2 = 2 * x^2 - 4 * x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_values_l4185_418551


namespace NUMINAMATH_CALUDE_rectangle_partition_into_L_shapes_rectangle_1985_1987_not_partitionable_rectangle_1987_1989_partitionable_l4185_418543

/-- An L-shape is a figure composed of 3 unit squares -/
def LShape : Nat := 3

/-- Checks if a number is divisible by 3 -/
def isDivisibleBy3 (n : Nat) : Prop := n % 3 = 0

/-- Checks if a number leaves a remainder of 2 when divided by 3 -/
def hasRemainder2 (n : Nat) : Prop := n % 3 = 2

/-- Theorem: A rectangle can be partitioned into L-shapes iff
    1) Its area is divisible by 3, and
    2) At least one side is divisible by 3, or both sides have remainder 2 when divided by 3 -/
theorem rectangle_partition_into_L_shapes (m n : Nat) :
  (isDivisibleBy3 (m * n)) ∧ 
  (isDivisibleBy3 m ∨ isDivisibleBy3 n ∨ (hasRemainder2 m ∧ hasRemainder2 n)) ↔ 
  ∃ (k : Nat), m * n = k * LShape := by sorry

/-- Corollary: 1985 × 1987 rectangle cannot be partitioned into L-shapes -/
theorem rectangle_1985_1987_not_partitionable :
  ¬ ∃ (k : Nat), 1985 * 1987 = k * LShape := by sorry

/-- Corollary: 1987 × 1989 rectangle can be partitioned into L-shapes -/
theorem rectangle_1987_1989_partitionable :
  ∃ (k : Nat), 1987 * 1989 = k * LShape := by sorry

end NUMINAMATH_CALUDE_rectangle_partition_into_L_shapes_rectangle_1985_1987_not_partitionable_rectangle_1987_1989_partitionable_l4185_418543


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_ninety_nine_thousand_nine_hundred_ninety_nine_is_largest_l4185_418563

theorem largest_n_divisible_by_seven (n : ℕ) : 
  n < 100000 →
  (9 * (n - 3)^5 - 2 * n^3 + 17 * n - 33) % 7 = 0 →
  n ≤ 99999 :=
by sorry

theorem ninety_nine_thousand_nine_hundred_ninety_nine_is_largest :
  (9 * (99999 - 3)^5 - 2 * 99999^3 + 17 * 99999 - 33) % 7 = 0 ∧
  ∀ m : ℕ, m > 99999 → m < 100000 → (9 * (m - 3)^5 - 2 * m^3 + 17 * m - 33) % 7 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_ninety_nine_thousand_nine_hundred_ninety_nine_is_largest_l4185_418563


namespace NUMINAMATH_CALUDE_second_diagonal_unrestricted_l4185_418540

/-- Represents a convex quadrilateral with specific properties -/
structure ConvexQuadrilateral where
  /-- The area of the quadrilateral in cm² -/
  area : ℝ
  /-- The length of the first diagonal in cm -/
  diagonal1 : ℝ
  /-- The length of the second diagonal in cm -/
  diagonal2 : ℝ
  /-- The sum of two opposite sides in cm -/
  opposite_sides_sum : ℝ
  /-- The area is positive -/
  area_pos : area > 0
  /-- Both diagonals are positive -/
  diag1_pos : diagonal1 > 0
  diag2_pos : diagonal2 > 0
  /-- The sum of opposite sides is non-negative -/
  opp_sides_sum_nonneg : opposite_sides_sum ≥ 0
  /-- The area is 32 cm² -/
  area_is_32 : area = 32
  /-- The sum of one diagonal and two opposite sides is 16 cm -/
  sum_is_16 : diagonal1 + opposite_sides_sum = 16

/-- Theorem stating that the second diagonal can be any positive real number -/
theorem second_diagonal_unrestricted (q : ConvexQuadrilateral) : 
  ∀ x : ℝ, x > 0 → ∃ q' : ConvexQuadrilateral, q'.diagonal2 = x := by
  sorry

end NUMINAMATH_CALUDE_second_diagonal_unrestricted_l4185_418540


namespace NUMINAMATH_CALUDE_factors_of_1320_eq_24_l4185_418584

/-- The number of distinct positive factors of 1320 -/
def factors_of_1320 : ℕ :=
  (3 : ℕ) * 2 * 2 * 2

/-- Theorem stating that the number of distinct positive factors of 1320 is 24 -/
theorem factors_of_1320_eq_24 : factors_of_1320 = 24 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_1320_eq_24_l4185_418584


namespace NUMINAMATH_CALUDE_unique_n_less_than_180_l4185_418591

theorem unique_n_less_than_180 : ∃! n : ℕ, n < 180 ∧ n % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_less_than_180_l4185_418591


namespace NUMINAMATH_CALUDE_opposite_of_2023_l4185_418569

theorem opposite_of_2023 : -(2023 : ℝ) = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l4185_418569


namespace NUMINAMATH_CALUDE_zeros_in_intervals_l4185_418545

/-- A quadratic function -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem zeros_in_intervals (a b c m n p : ℝ) (h_a : a ≠ 0) (h_order : m < n ∧ n < p) :
  (∃ x y, m < x ∧ x < n ∧ n < y ∧ y < p ∧ 
    quadratic_function a b c x = 0 ∧ 
    quadratic_function a b c y = 0) ↔ 
  (quadratic_function a b c m) * (quadratic_function a b c n) < 0 ∧
  (quadratic_function a b c p) * (quadratic_function a b c n) < 0 :=
sorry

end NUMINAMATH_CALUDE_zeros_in_intervals_l4185_418545


namespace NUMINAMATH_CALUDE_sum_of_fractions_l4185_418512

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l4185_418512


namespace NUMINAMATH_CALUDE_sequence_property_l4185_418521

theorem sequence_property (a : ℕ → ℕ) 
  (h1 : a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 1)
  (h2 : ∀ n : ℕ, a n * a (n + 3) - a (n + 1) * a (n + 2) = 1) :
  ∀ n : ℕ, ∃ k : ℤ, (a n + a (n + 2) : ℤ) = k * (a (n + 1) : ℤ) :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l4185_418521


namespace NUMINAMATH_CALUDE_complex_simplification_l4185_418544

theorem complex_simplification :
  (-5 + 3 * Complex.I) - (2 - 7 * Complex.I) + (1 + 2 * Complex.I) = -6 + 12 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l4185_418544


namespace NUMINAMATH_CALUDE_tower_surface_area_calculation_l4185_418560

def cube_surface_area (s : ℕ) : ℕ := 6 * s^2

def tower_surface_area (edge_lengths : List ℕ) : ℕ :=
  let n := edge_lengths.length
  edge_lengths.enum.foldl (fun acc (i, s) => 
    if i = 0 
    then acc + cube_surface_area s
    else acc + cube_surface_area s - s^2
  ) 0

theorem tower_surface_area_calculation :
  tower_surface_area [4, 5, 6, 7, 8, 9, 10] = 1871 :=
sorry

end NUMINAMATH_CALUDE_tower_surface_area_calculation_l4185_418560


namespace NUMINAMATH_CALUDE_average_transformation_l4185_418500

theorem average_transformation (x₁ x₂ x₃ : ℝ) (h : (x₁ + x₂ + x₃) / 3 = 8) :
  ((2 * x₁ - 1) + (2 * x₂ - 1) + (2 * x₃ - 1)) / 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_transformation_l4185_418500


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l4185_418510

theorem sum_of_three_consecutive_integers : ∃ (n : ℤ),
  (n - 1) + n + (n + 1) = 21 ∧
  ¬(∃ (m : ℤ), (m - 1) + m + (m + 1) = 17) ∧
  ¬(∃ (m : ℤ), (m - 1) + m + (m + 1) = 11) ∧
  ¬(∃ (m : ℤ), (m - 1) + m + (m + 1) = 25) ∧
  ¬(∃ (m : ℤ), (m - 1) + m + (m + 1) = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l4185_418510


namespace NUMINAMATH_CALUDE_lemon_pie_degrees_l4185_418586

theorem lemon_pie_degrees (total_students : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ)
  (h1 : total_students = 40)
  (h2 : chocolate = 15)
  (h3 : apple = 9)
  (h4 : blueberry = 7)
  (h5 : (total_students - (chocolate + apple + blueberry)) % 2 = 0) :
  let remaining := total_students - (chocolate + apple + blueberry)
  let lemon := remaining / 2
  (lemon : ℚ) / total_students * 360 = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_lemon_pie_degrees_l4185_418586


namespace NUMINAMATH_CALUDE_min_cookies_eaten_l4185_418536

/-- Represents the number of cookies at each stage of the process -/
structure CookieCount where
  initial : ℕ
  after_first : ℕ
  after_second : ℕ
  after_third : ℕ
  evening : ℕ

/-- Defines the cookie distribution process -/
def distribute_cookies (c : CookieCount) : Prop :=
  c.after_first = (2 * (c.initial - 1)) / 3 ∧
  c.after_second = (2 * (c.after_first - 1)) / 3 ∧
  c.after_third = (2 * (c.after_second - 1)) / 3 ∧
  c.evening = c.after_third - 1

/-- Defines the evening distribution condition -/
def evening_distribution (c : CookieCount) (n : ℕ) : Prop :=
  c.evening = 3 * n

/-- Defines the condition that no cookies are broken -/
def no_broken_cookies (c : CookieCount) : Prop :=
  c.initial % 1 = 0 ∧
  c.after_first % 1 = 0 ∧
  c.after_second % 1 = 0 ∧
  c.after_third % 1 = 0 ∧
  c.evening % 1 = 0

/-- Theorem stating the minimum number of cookies Xiao Wang could have eaten -/
theorem min_cookies_eaten (c : CookieCount) (n : ℕ) :
  distribute_cookies c →
  evening_distribution c n →
  no_broken_cookies c →
  (c.initial - c.after_first) = 6 ∧ n = 7 := by
  sorry

#check min_cookies_eaten

end NUMINAMATH_CALUDE_min_cookies_eaten_l4185_418536


namespace NUMINAMATH_CALUDE_silk_per_dress_is_five_l4185_418513

/-- Calculates the amount of silk needed for each dress given the initial silk amount,
    number of friends, silk given to each friend, and number of dresses made. -/
def silk_per_dress (initial_silk : ℕ) (num_friends : ℕ) (silk_per_friend : ℕ) (num_dresses : ℕ) : ℕ :=
  (initial_silk - num_friends * silk_per_friend) / num_dresses

/-- Proves that given the specified conditions, the amount of silk needed for each dress is 5 meters. -/
theorem silk_per_dress_is_five :
  silk_per_dress 600 5 20 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_silk_per_dress_is_five_l4185_418513


namespace NUMINAMATH_CALUDE_solution_characterization_l4185_418566

/-- The set of solutions to the system of equations:
    a² + b = c²
    b² + c = a²
    c² + a = b²
-/
def SolutionSet : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (0, 1, -1), (-1, 0, 1), (1, -1, 0)}

/-- A triplet (a, b, c) satisfies the system of equations -/
def SatisfiesSystem (t : ℝ × ℝ × ℝ) : Prop :=
  let (a, b, c) := t
  a^2 + b = c^2 ∧ b^2 + c = a^2 ∧ c^2 + a = b^2

theorem solution_characterization :
  ∀ t : ℝ × ℝ × ℝ, SatisfiesSystem t ↔ t ∈ SolutionSet := by
  sorry


end NUMINAMATH_CALUDE_solution_characterization_l4185_418566


namespace NUMINAMATH_CALUDE_trigonometric_problem_l4185_418587

theorem trigonometric_problem (α β : Real) 
  (h1 : 2 * Real.sin α = 2 * Real.sin (α / 2) ^ 2 - 1)
  (h2 : α ∈ Set.Ioo 0 Real.pi)
  (h3 : β ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h4 : 3 * Real.tan β ^ 2 - 2 * Real.tan β = 1) :
  (Real.sin (2 * α) + Real.cos (2 * α) = -1/5) ∧ 
  (α + β = 7 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l4185_418587


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l4185_418546

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (dims : BoxDimensions) : ℕ :=
  dims.length * dims.width * dims.height

/-- Calculates the maximum number of smaller boxes that can fit in a larger box -/
def maxBoxesFit (largeBox : BoxDimensions) (smallBox : BoxDimensions) : ℕ :=
  (boxVolume largeBox) / (boxVolume smallBox)

/-- The dimensions of the carton -/
def cartonDims : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDims : BoxDimensions :=
  { length := 7, width := 6, height := 10 }

/-- Theorem stating that the maximum number of soap boxes that can fit in the carton is 150 -/
theorem max_soap_boxes_in_carton :
  maxBoxesFit cartonDims soapBoxDims = 150 := by
  sorry


end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l4185_418546


namespace NUMINAMATH_CALUDE_cost_of_gums_in_dollars_l4185_418541

-- Define the cost of one piece of gum in cents
def cost_of_one_gum : ℕ := 2

-- Define the number of pieces of gum
def number_of_gums : ℕ := 500

-- Define the conversion rate from cents to dollars
def cents_per_dollar : ℕ := 100

-- Theorem to prove
theorem cost_of_gums_in_dollars : 
  (number_of_gums * cost_of_one_gum) / cents_per_dollar = 10 := by
  sorry


end NUMINAMATH_CALUDE_cost_of_gums_in_dollars_l4185_418541


namespace NUMINAMATH_CALUDE_polaroid_photo_length_l4185_418519

/-- The circumference of a rectangle given its length and width -/
def rectangleCircumference (length width : ℝ) : ℝ :=
  2 * (length + width)

/-- Theorem: A rectangular Polaroid photo with circumference 40 cm and width 8 cm has a length of 12 cm -/
theorem polaroid_photo_length (circumference width : ℝ) 
    (h_circumference : circumference = 40)
    (h_width : width = 8)
    (h_rect : rectangleCircumference length width = circumference) :
    length = 12 := by
  sorry


end NUMINAMATH_CALUDE_polaroid_photo_length_l4185_418519


namespace NUMINAMATH_CALUDE_lunch_cost_theorem_l4185_418528

/-- The cost of a Taco Grande Plate -/
def taco_grande_cost : ℕ := 8

/-- The cost of Mike's additional items -/
def mike_additional_cost : ℕ := 2 + 4 + 2

/-- Mike's total bill -/
def mike_bill : ℕ := taco_grande_cost + mike_additional_cost

/-- John's total bill -/
def john_bill : ℕ := taco_grande_cost

/-- The combined total cost of Mike and John's lunch -/
def combined_total_cost : ℕ := mike_bill + john_bill

theorem lunch_cost_theorem :
  (mike_bill = 2 * john_bill) →
  (combined_total_cost = 24) :=
by sorry

end NUMINAMATH_CALUDE_lunch_cost_theorem_l4185_418528


namespace NUMINAMATH_CALUDE_remainder_problem_l4185_418504

theorem remainder_problem (n : ℤ) (h : n % 9 = 4) : (4 * n - 11) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l4185_418504


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l4185_418525

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_area := 6 * L^2
  let new_edge_length := 1.6 * L
  let new_area := 6 * new_edge_length^2
  (new_area - original_area) / original_area * 100 = 156 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l4185_418525


namespace NUMINAMATH_CALUDE_sara_meets_bus_probability_l4185_418524

/-- Represents the time in minutes after 3:30 pm -/
def TimeAfter330 := { t : ℝ // 0 ≤ t ∧ t ≤ 60 }

/-- The bus arrives at a random time between 3:30 pm and 4:30 pm -/
def bus_arrival : TimeAfter330 := sorry

/-- Sara arrives at a random time between 3:30 pm and 4:30 pm -/
def sara_arrival : TimeAfter330 := sorry

/-- The bus waits for 40 minutes after arrival -/
def bus_wait_time : ℝ := 40

/-- The probability that Sara arrives while the bus is still waiting -/
def probability_sara_meets_bus : ℝ := sorry

theorem sara_meets_bus_probability :
  probability_sara_meets_bus = 2/3 := by sorry

end NUMINAMATH_CALUDE_sara_meets_bus_probability_l4185_418524


namespace NUMINAMATH_CALUDE_smallest_base_for_fraction_l4185_418576

theorem smallest_base_for_fraction (k : ℕ) : k = 14 ↔ 
  (k > 0 ∧ 
   ∀ m : ℕ, m > 0 ∧ m < k → (5 : ℚ) / 27 ≠ (m + 4 : ℚ) / (m^2 - 1) ∧
   (5 : ℚ) / 27 = (k + 4 : ℚ) / (k^2 - 1)) := by sorry

end NUMINAMATH_CALUDE_smallest_base_for_fraction_l4185_418576


namespace NUMINAMATH_CALUDE_min_square_area_is_49_l4185_418518

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a circle with diameter -/
structure Circle where
  diameter : ℝ

/-- Calculates the minimum square side length to contain given shapes -/
def minSquareSideLength (rect1 rect2 : Rectangle) (circle : Circle) : ℝ :=
  sorry

/-- Theorem: The minimum area of the square containing the given shapes is 49 -/
theorem min_square_area_is_49 : 
  let rect1 : Rectangle := ⟨2, 4⟩
  let rect2 : Rectangle := ⟨3, 5⟩
  let circle : Circle := ⟨3⟩
  (minSquareSideLength rect1 rect2 circle) ^ 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_min_square_area_is_49_l4185_418518


namespace NUMINAMATH_CALUDE_opposite_face_of_four_l4185_418589

/-- Represents the six faces of a cube -/
inductive Face
| A | B | C | D | E | F

/-- Assigns numbers to the faces of the cube -/
def face_value : Face → ℕ
| Face.A => 3
| Face.B => 4
| Face.C => 5
| Face.D => 6
| Face.E => 7
| Face.F => 8

/-- Defines the opposite face relation -/
def opposite : Face → Face
| Face.A => Face.F
| Face.B => Face.E
| Face.C => Face.D
| Face.D => Face.C
| Face.E => Face.B
| Face.F => Face.A

theorem opposite_face_of_four (h : ∀ (f : Face), face_value f + face_value (opposite f) = 11) :
  face_value (opposite Face.B) = 7 := by
  sorry

end NUMINAMATH_CALUDE_opposite_face_of_four_l4185_418589


namespace NUMINAMATH_CALUDE_harvard_acceptance_rate_l4185_418533

/-- Proves that the percentage of accepted students is 5% given the conditions -/
theorem harvard_acceptance_rate 
  (total_applicants : ℕ) 
  (attendance_rate : ℚ) 
  (attending_students : ℕ) 
  (h1 : total_applicants = 20000)
  (h2 : attendance_rate = 9/10)
  (h3 : attending_students = 900) :
  (attending_students / attendance_rate) / total_applicants = 1/20 := by
  sorry

#check harvard_acceptance_rate

end NUMINAMATH_CALUDE_harvard_acceptance_rate_l4185_418533


namespace NUMINAMATH_CALUDE_complement_of_N_in_M_l4185_418572

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {0, 1, 3}

theorem complement_of_N_in_M :
  M \ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_in_M_l4185_418572


namespace NUMINAMATH_CALUDE_congruence_solution_l4185_418517

theorem congruence_solution (n : ℕ) : n ≡ 40 [ZMOD 43] ↔ 11 * n ≡ 10 [ZMOD 43] ∧ n ≤ 42 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l4185_418517


namespace NUMINAMATH_CALUDE_seeds_per_flowerbed_l4185_418523

theorem seeds_per_flowerbed (total_seeds : ℕ) (num_flowerbeds : ℕ) (seeds_per_bed : ℕ) :
  total_seeds = 32 →
  num_flowerbeds = 8 →
  total_seeds = num_flowerbeds * seeds_per_bed →
  seeds_per_bed = 4 :=
by sorry

end NUMINAMATH_CALUDE_seeds_per_flowerbed_l4185_418523


namespace NUMINAMATH_CALUDE_goose_egg_count_l4185_418502

/-- The number of goose eggs laid at a certain pond -/
def total_eggs : ℕ := 1000

/-- The fraction of eggs that hatched -/
def hatch_rate : ℚ := 1/4

/-- The fraction of hatched geese that survived the first month -/
def first_month_survival_rate : ℚ := 4/5

/-- The fraction of geese that survived the first month but did not survive the first year -/
def first_year_mortality_rate : ℚ := 2/5

/-- The number of geese that survived the first year -/
def survivors : ℕ := 120

theorem goose_egg_count :
  total_eggs * hatch_rate * first_month_survival_rate * (1 - first_year_mortality_rate) = survivors := by
  sorry

end NUMINAMATH_CALUDE_goose_egg_count_l4185_418502


namespace NUMINAMATH_CALUDE_no_alpha_exists_for_all_x_l4185_418526

theorem no_alpha_exists_for_all_x (α : ℝ) (h : α > 0) : 
  ∃ x : ℝ, |Real.cos x| + |Real.cos (α * x)| ≤ Real.sin x + Real.sin (α * x) := by
sorry

end NUMINAMATH_CALUDE_no_alpha_exists_for_all_x_l4185_418526


namespace NUMINAMATH_CALUDE_max_largest_integer_l4185_418577

theorem max_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℚ) / 5 = 50 →
  max a (max b (max c (max d e))) - min a (min b (min c (min d e))) = 10 →
  max a (max b (max c (max d e))) ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_max_largest_integer_l4185_418577


namespace NUMINAMATH_CALUDE_perfect_squares_between_100_and_500_l4185_418594

theorem perfect_squares_between_100_and_500 : 
  (Finset.filter (fun n => 100 < n^2 ∧ n^2 < 500) (Finset.range 23)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_between_100_and_500_l4185_418594


namespace NUMINAMATH_CALUDE_linear_function_problem_l4185_418550

/-- A linear function satisfying specific conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem linear_function_problem :
  (∀ x y : ℝ, ∃ a b : ℝ, f x = a * x + b) ∧ 
  (∀ x : ℝ, f x = 3 * (f⁻¹ x) + 9) ∧
  f 2 = 5 ∧
  f 3 = 9 →
  f 5 = 9 - 8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_linear_function_problem_l4185_418550


namespace NUMINAMATH_CALUDE_min_odd_sided_polygon_divisible_into_parallelograms_l4185_418534

/-- A polygon is a closed shape with straight sides. -/
structure Polygon where
  sides : ℕ
  is_closed : Bool

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
structure Parallelogram where
  is_quadrilateral : Bool
  opposite_sides_parallel : Bool

/-- A function that checks if a polygon can be divided into parallelograms. -/
def can_be_divided_into_parallelograms (p : Polygon) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ ∃ (parallelograms : Fin n → Parallelogram), True

/-- Theorem stating the minimum number of sides for an odd-sided polygon
    that can be divided into parallelograms is 7. -/
theorem min_odd_sided_polygon_divisible_into_parallelograms :
  ∀ (p : Polygon),
    p.sides % 2 = 1 →
    can_be_divided_into_parallelograms p →
    p.sides ≥ 7 ∧
    ∃ (q : Polygon), q.sides = 7 ∧ can_be_divided_into_parallelograms q :=
sorry

end NUMINAMATH_CALUDE_min_odd_sided_polygon_divisible_into_parallelograms_l4185_418534


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l4185_418561

theorem cube_sum_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  2 * (x^3 + y^3 + z^3) ≥ x^2 * y + x^2 * z + y^2 * z + y^2 * x + z^2 * x + z^2 * y :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l4185_418561


namespace NUMINAMATH_CALUDE_cake_supplies_cost_l4185_418565

/-- Proves that the cost of supplies for a cake is $54 given the specified conditions -/
theorem cake_supplies_cost (hours_per_day : ℕ) (days_worked : ℕ) (hourly_rate : ℕ) (profit : ℕ) : 
  hours_per_day = 2 →
  days_worked = 4 →
  hourly_rate = 22 →
  profit = 122 →
  (hours_per_day * days_worked * hourly_rate) - profit = 54 :=
by sorry

end NUMINAMATH_CALUDE_cake_supplies_cost_l4185_418565


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l4185_418583

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + a*x + 4*a| ≤ 3) ↔ (a = 8 + 2*Real.sqrt 13 ∨ a = 8 - 2*Real.sqrt 13) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l4185_418583


namespace NUMINAMATH_CALUDE_not_both_odd_with_equal_product_l4185_418537

/-- Represents a mapping of letters to digits -/
def DigitMapping := Char → Fin 10

/-- Represents a number as a string of letters -/
def NumberWord := String

/-- Calculate the product of digits in a number word given a digit mapping -/
def digitProduct (mapping : DigitMapping) (word : NumberWord) : ℕ :=
  word.foldl (λ acc c => acc * (mapping c).val.succ) 1

/-- Check if a number word represents an odd number given a digit mapping -/
def isOdd (mapping : DigitMapping) (word : NumberWord) : Prop :=
  (mapping word.back).val % 2 = 1

theorem not_both_odd_with_equal_product (mapping : DigitMapping) 
    (word1 word2 : NumberWord) 
    (h_distinct : ∀ (c1 c2 : Char), c1 ≠ c2 → mapping c1 ≠ mapping c2)
    (h_equal_product : digitProduct mapping word1 = digitProduct mapping word2) :
    ¬(isOdd mapping word1 ∧ isOdd mapping word2) := by
  sorry

end NUMINAMATH_CALUDE_not_both_odd_with_equal_product_l4185_418537


namespace NUMINAMATH_CALUDE_constant_zero_sequence_l4185_418514

def is_sum_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ k, S (k + 1) = S k + a (k + 1)

theorem constant_zero_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h : ∀ k, S (k + 1) + S k = a (k + 1)) :
  ∀ n, a n = 0 :=
by sorry

end NUMINAMATH_CALUDE_constant_zero_sequence_l4185_418514


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l4185_418559

theorem geometric_sequence_first_term (a b c : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ 16 = b * r ∧ c = 16 * r ∧ 128 = c * r) →
  a = 1/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l4185_418559


namespace NUMINAMATH_CALUDE_cookie_radius_l4185_418522

/-- Given a circle with equation x^2 + y^2 + x - 5y = 10, its radius is √(33/2) -/
theorem cookie_radius (x y : ℝ) : 
  x^2 + y^2 + x - 5*y = 10 → 
  ∃ (center : ℝ × ℝ), ∃ (radius : ℝ), 
    (x - center.1)^2 + (y - center.2)^2 = radius^2 ∧ 
    radius = Real.sqrt (33/2) := by
  sorry

end NUMINAMATH_CALUDE_cookie_radius_l4185_418522


namespace NUMINAMATH_CALUDE_units_digit_of_smallest_n_with_2016_digits_l4185_418555

theorem units_digit_of_smallest_n_with_2016_digits : ∃ n : ℕ,
  (∀ m : ℕ, 7 * m < 10^2015 → m < n) ∧
  7 * n ≥ 10^2015 ∧
  7 * n < 10^2016 ∧
  n % 10 = 6 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_smallest_n_with_2016_digits_l4185_418555


namespace NUMINAMATH_CALUDE_probability_king_then_heart_l4185_418562

/- Define a standard deck of cards -/
def StandardDeck : ℕ := 52

/- Define the number of Kings in a standard deck -/
def NumberOfKings : ℕ := 4

/- Define the number of hearts in a standard deck -/
def NumberOfHearts : ℕ := 13

/- Theorem statement -/
theorem probability_king_then_heart (deck : ℕ) (kings : ℕ) (hearts : ℕ) 
  (h1 : deck = StandardDeck) 
  (h2 : kings = NumberOfKings) 
  (h3 : hearts = NumberOfHearts) : 
  (kings : ℚ) / deck * hearts / (deck - 1) = 1 / 52 := by
  sorry


end NUMINAMATH_CALUDE_probability_king_then_heart_l4185_418562


namespace NUMINAMATH_CALUDE_cone_base_diameter_l4185_418558

theorem cone_base_diameter (sphere_radius : ℝ) (cone_height : ℝ) (waste_percentage : ℝ) :
  sphere_radius = 9 →
  cone_height = 9 →
  waste_percentage = 75 →
  let cone_volume := (1 - waste_percentage / 100) * (4 / 3) * Real.pi * sphere_radius ^ 3
  let cone_base_radius := Real.sqrt (3 * cone_volume / (Real.pi * cone_height))
  2 * cone_base_radius = 9 :=
by sorry

end NUMINAMATH_CALUDE_cone_base_diameter_l4185_418558


namespace NUMINAMATH_CALUDE_jiangxi_2013_problem_l4185_418596

open Real

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem jiangxi_2013_problem (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f (exp x) = x + exp x) : 
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_jiangxi_2013_problem_l4185_418596


namespace NUMINAMATH_CALUDE_f_neg_one_eq_neg_one_l4185_418532

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem f_neg_one_eq_neg_one
  (h_odd : IsOdd f)
  (h_f_one : f 1 = 1) :
  f (-1) = -1 :=
sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_neg_one_l4185_418532


namespace NUMINAMATH_CALUDE_dice_probability_l4185_418539

def num_dice : ℕ := 8
def num_faces : ℕ := 6
def num_pairs : ℕ := 3

def total_outcomes : ℕ := num_faces ^ num_dice

def favorable_outcomes : ℕ :=
  Nat.choose num_faces num_pairs *
  Nat.choose (num_faces - num_pairs) (num_dice - 2 * num_pairs) *
  Nat.factorial num_pairs *
  Nat.factorial (num_dice - 2 * num_pairs) *
  Nat.choose num_dice 2 *
  Nat.choose (num_dice - 2) 2 *
  Nat.choose (num_dice - 4) 2

theorem dice_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 525 / 972 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l4185_418539


namespace NUMINAMATH_CALUDE_square_root_property_l4185_418567

theorem square_root_property (x : ℝ) : 
  (Real.sqrt (2*x + 3) = 3) → (2*x + 3)^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_root_property_l4185_418567


namespace NUMINAMATH_CALUDE_parabola_y_relationship_l4185_418507

/-- A parabola defined by y = -x² + 6x + c -/
def parabola (x : ℝ) (c : ℝ) : ℝ := -x^2 + 6*x + c

/-- Three points on the parabola -/
structure PointsOnParabola (c : ℝ) where
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  h₁ : parabola 1 c = y₁
  h₂ : parabola 3 c = y₂
  h₃ : parabola 4 c = y₃

/-- Theorem stating the relationship between y₁, y₂, and y₃ -/
theorem parabola_y_relationship (c : ℝ) (p : PointsOnParabola c) :
  p.y₁ < p.y₃ ∧ p.y₃ < p.y₂ := by sorry

end NUMINAMATH_CALUDE_parabola_y_relationship_l4185_418507


namespace NUMINAMATH_CALUDE_day_crew_fraction_is_eight_elevenths_l4185_418503

/-- Represents the fraction of boxes loaded by the day crew given the relative productivity and size of the night crew -/
def day_crew_fraction (night_crew_productivity : ℚ) (night_crew_size : ℚ) : ℚ :=
  1 / (1 + night_crew_productivity * night_crew_size)

theorem day_crew_fraction_is_eight_elevenths :
  day_crew_fraction (3/4) (1/2) = 8/11 := by
  sorry

end NUMINAMATH_CALUDE_day_crew_fraction_is_eight_elevenths_l4185_418503


namespace NUMINAMATH_CALUDE_problem_solution_l4185_418547

-- Define the conditions p and q
def p (x : ℝ) : Prop := 4 * x^2 + 12 * x - 7 ≤ 0
def q (a x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

theorem problem_solution :
  (∃ x : ℝ, p x ∧ ¬(q 0 x) ∧ -7/2 ≤ x ∧ x < -3) ∧
  (∀ a : ℝ, (∀ x : ℝ, p x → q a x) ↔ -5/2 ≤ a ∧ a ≤ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4185_418547


namespace NUMINAMATH_CALUDE_total_tosses_equals_sum_of_heads_and_tails_l4185_418580

/-- Represents the number of times Head came up in the coin tosses -/
def head_count : ℕ := 9

/-- Represents the number of times Tail came up in the coin tosses -/
def tail_count : ℕ := 5

/-- Theorem stating that the total number of coin tosses is the sum of head_count and tail_count -/
theorem total_tosses_equals_sum_of_heads_and_tails :
  head_count + tail_count = 14 := by sorry

end NUMINAMATH_CALUDE_total_tosses_equals_sum_of_heads_and_tails_l4185_418580


namespace NUMINAMATH_CALUDE_min_white_surface_fraction_problem_cube_l4185_418573

/-- Represents a cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  total_small_cubes : ℕ
  red_cubes : ℕ
  white_cubes : ℕ

/-- Calculates the minimum fraction of white surface area for a given LargeCube -/
def min_white_surface_fraction (cube : LargeCube) : ℚ :=
  sorry

/-- The specific cube described in the problem -/
def problem_cube : LargeCube :=
  { edge_length := 4
  , total_small_cubes := 64
  , red_cubes := 52
  , white_cubes := 12 }

theorem min_white_surface_fraction_problem_cube :
  min_white_surface_fraction problem_cube = 11 / 96 :=
sorry

end NUMINAMATH_CALUDE_min_white_surface_fraction_problem_cube_l4185_418573


namespace NUMINAMATH_CALUDE_circle_and_distance_l4185_418570

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop :=
  (P.1 + 1)^2 + P.2^2 = 2 * ((P.1 - 1)^2 + P.2^2)

-- Define the circle C
def C : Set (ℝ × ℝ) :=
  {P | (P.1 - 3)^2 + P.2^2 = 8}

-- Define the parabola
def parabola : Set (ℝ × ℝ) :=
  {P | P.2^2 = P.1}

theorem circle_and_distance :
  (∀ P, P_condition P → P ∈ C) ∧
  (∃ Q ∈ parabola, ∀ R ∈ parabola, 
    dist (3, 0) Q ≤ dist (3, 0) R ∧ 
    dist (3, 0) Q = Real.sqrt 11 / 2) :=
sorry

end NUMINAMATH_CALUDE_circle_and_distance_l4185_418570


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l4185_418520

theorem inequality_holds_iff (m : ℝ) :
  (∀ x : ℝ, (x^2 - m*x - 2) / (x^2 - 3*x + 4) > -1) ↔ -7 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l4185_418520


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_45_4050_l4185_418538

theorem gcd_lcm_sum_45_4050 : Nat.gcd 45 4050 + Nat.lcm 45 4050 = 4095 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_45_4050_l4185_418538


namespace NUMINAMATH_CALUDE_triangle_altitude_proof_l4185_418592

def triangle_altitude (a b c : ℝ) : Prop :=
  let tan_BCA := 1
  let tan_BAC := 1 / 7
  let perimeter := 24 + 18 * Real.sqrt 2
  let h := 3
  -- The altitude from B to AC has length h
  (tan_BCA = 1 ∧ tan_BAC = 1 / 7 ∧ 
   a + b + c = perimeter) → 
  h = 3

theorem triangle_altitude_proof : 
  ∃ (a b c : ℝ), triangle_altitude a b c :=
sorry

end NUMINAMATH_CALUDE_triangle_altitude_proof_l4185_418592


namespace NUMINAMATH_CALUDE_museum_trip_l4185_418553

def bus_trip (first_bus : ℕ) : Prop :=
  let second_bus := 2 * first_bus
  let third_bus := second_bus - 6
  let fourth_bus := first_bus + 9
  let total_people := first_bus + second_bus + third_bus + fourth_bus
  (first_bus ≤ 45) ∧ 
  (second_bus ≤ 45) ∧ 
  (third_bus ≤ 45) ∧ 
  (fourth_bus ≤ 45) ∧ 
  (total_people = 75)

theorem museum_trip : bus_trip 12 := by
  sorry

end NUMINAMATH_CALUDE_museum_trip_l4185_418553


namespace NUMINAMATH_CALUDE_solution_check_unique_non_solution_l4185_418530

theorem solution_check : ℝ → ℝ → Prop :=
  fun x y => x + y = 5

theorem unique_non_solution :
  (solution_check 2 3 ∧ 
   solution_check (-2) 7 ∧ 
   solution_check 0 5) ∧ 
  ¬(solution_check 1 6) := by
  sorry

end NUMINAMATH_CALUDE_solution_check_unique_non_solution_l4185_418530


namespace NUMINAMATH_CALUDE_max_side_squared_acute_triangle_l4185_418527

theorem max_side_squared_acute_triangle (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  b^2 + 4 * c^2 = 8 →
  Real.sin B + 2 * Real.sin C = 6 * b * Real.sin A * Real.sin C →
  a^2 ≤ (15 - 8 * Real.sqrt 2) / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_side_squared_acute_triangle_l4185_418527


namespace NUMINAMATH_CALUDE_odd_function_sum_l4185_418506

def f (x a b : ℝ) : ℝ := (x - 1)^2 + a * x^2 + b

theorem odd_function_sum (a b : ℝ) :
  (∀ x, f x a b = -f (-x) a b) → a + b = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l4185_418506


namespace NUMINAMATH_CALUDE_cumulonimbus_cloud_count_l4185_418593

theorem cumulonimbus_cloud_count :
  ∀ (cirrus cumulus cumulonimbus : ℕ),
    cirrus = 4 * cumulus →
    cumulus = 12 * cumulonimbus →
    cumulonimbus > 0 →
    cirrus = 144 →
    cumulonimbus = 3 := by
  sorry

end NUMINAMATH_CALUDE_cumulonimbus_cloud_count_l4185_418593


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l4185_418595

theorem orthogonal_vectors (y : ℚ) : 
  ((-4 : ℚ) * 3 + 7 * y = 0) → y = 12/7 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l4185_418595


namespace NUMINAMATH_CALUDE_wilsons_theorem_l4185_418599

theorem wilsons_theorem (p : ℕ) (h : p > 1) :
  Nat.Prime p ↔ (Nat.factorial (p - 1) % p = p - 1) := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l4185_418599


namespace NUMINAMATH_CALUDE_power_of_three_mod_five_l4185_418549

theorem power_of_three_mod_five : 3^2040 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_five_l4185_418549


namespace NUMINAMATH_CALUDE_weight_of_larger_square_l4185_418516

/-- Represents the properties of a square metal piece -/
structure MetalSquare where
  side : ℝ  -- side length in inches
  weight : ℝ  -- weight in ounces

/-- Theorem stating the relationship between two metal squares -/
theorem weight_of_larger_square 
  (small : MetalSquare) 
  (large : MetalSquare) 
  (h1 : small.side = 4) 
  (h2 : small.weight = 16) 
  (h3 : large.side = 6) 
  (h_uniform : ∀ (s1 s2 : MetalSquare), s1.weight / (s1.side ^ 2) = s2.weight / (s2.side ^ 2)) :
  large.weight = 36 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_larger_square_l4185_418516


namespace NUMINAMATH_CALUDE_cricket_team_winning_percentage_l4185_418542

theorem cricket_team_winning_percentage 
  (total_matches : ℕ) 
  (august_matches : ℕ) 
  (total_wins : ℕ) 
  (new_win_rate : ℚ) :
  total_matches = 144 ∧ 
  august_matches = 120 ∧ 
  total_wins = 75 ∧ 
  new_win_rate = 52/100 →
  (total_wins - (total_matches - august_matches)) / august_matches = 51/120 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_winning_percentage_l4185_418542


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l4185_418509

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l4185_418509
