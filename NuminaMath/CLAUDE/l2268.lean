import Mathlib

namespace NUMINAMATH_CALUDE_dave_tickets_l2268_226814

theorem dave_tickets (initial_tickets : ℕ) : 
  (initial_tickets - 2 - 10 = 2) → initial_tickets = 14 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_l2268_226814


namespace NUMINAMATH_CALUDE_existence_of_abc_l2268_226866

def S (x : ℕ) : ℕ := (x.digits 10).sum

theorem existence_of_abc : ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  S (a + b) < 5 ∧ 
  S (b + c) < 5 ∧ 
  S (c + a) < 5 ∧ 
  S (a + b + c) > 50 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_abc_l2268_226866


namespace NUMINAMATH_CALUDE_min_sum_of_intercepts_l2268_226817

/-- A line with positive x-intercept and y-intercept passing through (1,4) -/
structure InterceptLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  passes_through : 1 / a + 4 / b = 1

/-- The minimum sum of intercepts for a line passing through (1,4) is 9 -/
theorem min_sum_of_intercepts (l : InterceptLine) : l.a + l.b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_intercepts_l2268_226817


namespace NUMINAMATH_CALUDE_product_of_solutions_l2268_226823

theorem product_of_solutions (x : ℝ) : 
  (12 = 2 * x^2 + 4 * x) → 
  (∃ x₁ x₂ : ℝ, (12 = 2 * x₁^2 + 4 * x₁) ∧ (12 = 2 * x₂^2 + 4 * x₂) ∧ (x₁ * x₂ = -6)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2268_226823


namespace NUMINAMATH_CALUDE_triangle_max_area_l2268_226828

theorem triangle_max_area (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  3 * a * b = 25 - c^2 →
  let angle_C := 60 * π / 180
  let area := (1 / 2) * a * b * Real.sin angle_C
  area ≤ 25 * Real.sqrt 3 / 16 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2268_226828


namespace NUMINAMATH_CALUDE_no_solutions_in_interval_l2268_226802

theorem no_solutions_in_interval (x : Real) : 
  0 ≤ x ∧ x ≤ 2 * Real.pi → 1 / Real.sin x + 1 / Real.cos x ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_in_interval_l2268_226802


namespace NUMINAMATH_CALUDE_game_attendance_l2268_226862

theorem game_attendance : ∃ (total : ℕ), 
  (total : ℚ) * (40 / 100) + (total : ℚ) * (34 / 100) + 3 = total ∧ total = 12 := by
  sorry

end NUMINAMATH_CALUDE_game_attendance_l2268_226862


namespace NUMINAMATH_CALUDE_tammy_haircuts_l2268_226813

/-- The number of paid haircuts required to get a free haircut -/
def haircuts_for_free : ℕ := 14

/-- The number of free haircuts Tammy has already received -/
def free_haircuts_received : ℕ := 5

/-- The number of haircuts Tammy needs for her next free one -/
def haircuts_until_next_free : ℕ := 5

/-- The total number of haircuts Tammy has gotten -/
def total_haircuts : ℕ := 79

theorem tammy_haircuts :
  total_haircuts = 
    (free_haircuts_received * haircuts_for_free) + 
    (haircuts_for_free - haircuts_until_next_free) :=
by sorry

end NUMINAMATH_CALUDE_tammy_haircuts_l2268_226813


namespace NUMINAMATH_CALUDE_gcf_of_90_135_225_l2268_226839

theorem gcf_of_90_135_225 : Nat.gcd 90 (Nat.gcd 135 225) = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_90_135_225_l2268_226839


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_zero_l2268_226864

theorem cos_sin_sum_equals_zero :
  Real.cos (5 * Real.pi / 8) * Real.cos (Real.pi / 8) + 
  Real.sin (5 * Real.pi / 8) * Real.sin (Real.pi / 8) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_zero_l2268_226864


namespace NUMINAMATH_CALUDE_vector_equation_solution_l2268_226893

theorem vector_equation_solution :
  let a₁ : ℚ := 181 / 136
  let a₂ : ℚ := 25 / 68
  let v₁ : Fin 2 → ℚ := ![4, -1]
  let v₂ : Fin 2 → ℚ := ![5, 3]
  let result : Fin 2 → ℚ := ![9, 4]
  (a₁ • v₁ + a₂ • v₂) = result := by sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l2268_226893


namespace NUMINAMATH_CALUDE_car_rental_cost_l2268_226815

/-- Calculates the car rental cost for a vacation given the number of people,
    Airbnb rental cost, and each person's share. -/
theorem car_rental_cost (num_people : ℕ) (airbnb_cost : ℕ) (person_share : ℕ) : 
  num_people = 8 → airbnb_cost = 3200 → person_share = 500 →
  num_people * person_share - airbnb_cost = 800 := by
sorry

end NUMINAMATH_CALUDE_car_rental_cost_l2268_226815


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l2268_226898

theorem fraction_zero_implies_x_negative_two (x : ℝ) :
  (abs x - 2) / (2 - x) = 0 → x = -2 :=
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l2268_226898


namespace NUMINAMATH_CALUDE_range_of_t_l2268_226854

theorem range_of_t (t α β : ℝ) 
  (h1 : t = Real.cos β ^ 3 + (α / 2) * Real.cos β)
  (h2 : α ≤ t)
  (h3 : t ≤ α - 5 * Real.cos β) :
  -2/3 ≤ t ∧ t ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_t_l2268_226854


namespace NUMINAMATH_CALUDE_nickels_count_l2268_226851

/-- Given 30 coins (nickels and dimes) with a total value of 240 cents, prove that the number of nickels is 12. -/
theorem nickels_count (n d : ℕ) : 
  n + d = 30 →  -- Total number of coins
  5 * n + 10 * d = 240 →  -- Total value in cents
  n = 12 :=  -- Number of nickels
by sorry

end NUMINAMATH_CALUDE_nickels_count_l2268_226851


namespace NUMINAMATH_CALUDE_virginia_egg_problem_l2268_226861

/-- Virginia's egg problem -/
theorem virginia_egg_problem (initial_eggs : ℕ) (taken_eggs : ℕ) : 
  initial_eggs = 96 → taken_eggs = 3 → initial_eggs - taken_eggs = 93 := by
sorry

end NUMINAMATH_CALUDE_virginia_egg_problem_l2268_226861


namespace NUMINAMATH_CALUDE_room_width_calculation_l2268_226829

/-- Given a room with specified length, total paving cost, and paving rate per square meter,
    calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  length = 5.5 →
  total_cost = 28875 →
  rate_per_sqm = 1400 →
  (total_cost / rate_per_sqm) / length = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l2268_226829


namespace NUMINAMATH_CALUDE_max_cables_cut_theorem_l2268_226896

/-- Represents a computer network -/
structure ComputerNetwork where
  num_computers : ℕ
  num_cables : ℕ
  num_clusters : ℕ

/-- Calculates the maximum number of cables that can be cut -/
def max_cables_cut (network : ComputerNetwork) : ℕ :=
  network.num_cables - (network.num_clusters - 1)

/-- Theorem stating the maximum number of cables that can be cut -/
theorem max_cables_cut_theorem (network : ComputerNetwork) 
  (h1 : network.num_computers = 200)
  (h2 : network.num_cables = 345)
  (h3 : network.num_clusters = 8) :
  max_cables_cut network = 153 := by
  sorry

#eval max_cables_cut ⟨200, 345, 8⟩

end NUMINAMATH_CALUDE_max_cables_cut_theorem_l2268_226896


namespace NUMINAMATH_CALUDE_monotonically_increasing_intervals_minimum_value_in_interval_f_properties_l2268_226822

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 4

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x

-- Theorem for monotonically increasing intervals
theorem monotonically_increasing_intervals :
  (∀ x < 0, f' x > 0) ∧ (∀ x > 1, f' x > 0) :=
sorry

-- Theorem for minimum value in the interval [-1, 2]
theorem minimum_value_in_interval :
  ∀ x ∈ Set.Icc (-1) 2, f x ≥ f (-1) :=
sorry

-- Main theorem combining both parts
theorem f_properties :
  (∀ x < 0, f' x > 0) ∧ (∀ x > 1, f' x > 0) ∧
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ f (-1)) :=
sorry

end NUMINAMATH_CALUDE_monotonically_increasing_intervals_minimum_value_in_interval_f_properties_l2268_226822


namespace NUMINAMATH_CALUDE_cuboid_volume_l2268_226880

/-- The volume of a cuboid that can be divided into 3 equal cubes, each with edges measuring 6 cm, is 648 cm³. -/
theorem cuboid_volume (cuboid : Real) (cube : Real) :
  (cuboid = 3 * cube) →  -- The cuboid is divided into 3 equal parts
  (cube = 6^3) →         -- Each part is a cube with edges measuring 6 cm
  (cuboid = 648) :=      -- The volume of the original cuboid is 648 cm³
by sorry

end NUMINAMATH_CALUDE_cuboid_volume_l2268_226880


namespace NUMINAMATH_CALUDE_arrangements_eq_24_l2268_226859

/-- The number of letter cards -/
def n : ℕ := 6

/-- The number of cards that can be freely arranged -/
def k : ℕ := n - 2

/-- The number of different arrangements of n letter cards where two cards are fixed at the ends -/
def num_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 2)

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_eq_24 : num_arrangements n = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_eq_24_l2268_226859


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2268_226816

theorem partial_fraction_decomposition :
  ∃! (A B C : ℚ),
    ∀ (x : ℚ), x ≠ 2 → x ≠ 3 → x ≠ 4 →
      (x^2 - 9) / ((x - 2) * (x - 3) * (x - 4)) =
      A / (x - 2) + B / (x - 3) + C / (x - 4) ∧
      A = -5/2 ∧ B = 0 ∧ C = 7/2 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2268_226816


namespace NUMINAMATH_CALUDE_volume_of_specific_solid_l2268_226858

/-- 
A solid with a square base and extended top edges.
s: side length of the square base
-/
structure ExtendedSolid where
  s : ℝ
  base_square : s > 0
  top_extended : ℝ × ℝ
  vertical_edge : ℝ

/-- The volume of the ExtendedSolid -/
noncomputable def volume (solid : ExtendedSolid) : ℝ :=
  solid.s^2 * solid.s

/-- Theorem: The volume of the specific ExtendedSolid is 128√2 -/
theorem volume_of_specific_solid :
  ∃ (solid : ExtendedSolid),
    solid.s = 4 * Real.sqrt 2 ∧
    solid.top_extended = (3 * solid.s, solid.s) ∧
    solid.vertical_edge = solid.s ∧
    volume solid = 128 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_solid_l2268_226858


namespace NUMINAMATH_CALUDE_socks_total_is_112_25_l2268_226840

/-- The total number of socks George and Maria have after receiving additional socks -/
def total_socks (george_initial : ℝ) (maria_initial : ℝ) 
                (george_bought : ℝ) (george_from_dad : ℝ) 
                (maria_from_mom : ℝ) (maria_from_aunt : ℝ) : ℝ :=
  (george_initial + george_bought + george_from_dad) + 
  (maria_initial + maria_from_mom + maria_from_aunt)

/-- Theorem stating that the total number of socks is 112.25 -/
theorem socks_total_is_112_25 : 
  total_socks 28.5 24.75 36.25 4.5 15.5 2.75 = 112.25 := by
  sorry

end NUMINAMATH_CALUDE_socks_total_is_112_25_l2268_226840


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l2268_226819

/-- Proves that for a rectangular roof where the length is 7 times the width
    and the area is 847 square feet, the difference between the length
    and the width is 66 feet. -/
theorem roof_dimension_difference (width : ℝ) (length : ℝ) : 
  length = 7 * width →
  length * width = 847 →
  length - width = 66 := by
sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l2268_226819


namespace NUMINAMATH_CALUDE_cos_210_degrees_l2268_226849

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l2268_226849


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2268_226884

/-- Represents a quadratic equation ax^2 + 4x + c = 0 with exactly one solution -/
structure UniqueQuadratic where
  a : ℝ
  c : ℝ
  has_unique_solution : (4^2 - 4*a*c) = 0
  sum_constraint : a + c = 5
  order_constraint : a < c

theorem unique_quadratic_solution (q : UniqueQuadratic) : (q.a, q.c) = (1, 4) := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2268_226884


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2268_226853

theorem simple_interest_rate_calculation (P A T : ℝ) (h1 : P = 750) (h2 : A = 900) (h3 : T = 5) :
  let R := (A - P) / (P * T) * 100
  R = 4 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2268_226853


namespace NUMINAMATH_CALUDE_position_of_2018_l2268_226809

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that returns the sequence of natural numbers whose digits sum to 11, in ascending order -/
def sequence_sum_11 : List ℕ := sorry

/-- The position of a number in a list, with 1-based indexing -/
def position_in_list (n : ℕ) (l : List ℕ) : ℕ := sorry

theorem position_of_2018 : 
  position_in_list 2018 sequence_sum_11 = 134 := by sorry

end NUMINAMATH_CALUDE_position_of_2018_l2268_226809


namespace NUMINAMATH_CALUDE_fishermen_catch_l2268_226820

theorem fishermen_catch (total_fish : ℕ) (carp_ratio : ℚ) (perch_ratio : ℚ) 
  (h_total : total_fish = 80)
  (h_carp : carp_ratio = 5 / 9)
  (h_perch : perch_ratio = 7 / 11) :
  ∃ (first_catch second_catch : ℕ),
    first_catch + second_catch = total_fish ∧
    first_catch = 36 ∧
    second_catch = 44 := by
  sorry

end NUMINAMATH_CALUDE_fishermen_catch_l2268_226820


namespace NUMINAMATH_CALUDE_smallest_initial_value_l2268_226870

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_initial_value : 
  ∀ n : ℕ, n ≥ 308 → 
  (is_perfect_square (n - 139) ∧ 
   ∀ m : ℕ, m < n → ¬ is_perfect_square (m - 139)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_initial_value_l2268_226870


namespace NUMINAMATH_CALUDE_thirty_five_operations_sufficient_l2268_226810

/-- A type representing a 10x10 grid of integers -/
def Grid := Fin 10 → Fin 10 → ℕ

/-- Predicate to check if a number is composite -/
def IsComposite (n : ℕ) : Prop := n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- Predicate to check if two cells are adjacent -/
def IsAdjacent (i j k l : Fin 10) : Prop :=
  (i = k ∧ (j.val + 1 = l.val ∨ j.val = l.val + 1)) ∨
  (j = l ∧ (i.val + 1 = k.val ∨ i.val = k.val + 1))

/-- Predicate to check if a grid satisfies the composite sum condition -/
def SatisfiesCompositeSumCondition (g : Grid) : Prop :=
  ∀ i j k l, IsAdjacent i j k l → IsComposite (g i j + g k l)

/-- Function to represent a swap operation -/
def Swap (g : Grid) (i j k l : Fin 10) : Grid :=
  fun x y => if (x = i ∧ y = j) ∨ (x = k ∧ y = l) then g k l else g x y

/-- Theorem stating that 35 operations are sufficient to achieve the goal -/
theorem thirty_five_operations_sufficient :
  ∃ (initial : Grid) (swaps : List (Fin 10 × Fin 10 × Fin 10 × Fin 10)),
    (∀ i j, initial i j ∈ Set.range (fun n : Fin 100 => n.val + 1)) ∧
    swaps.length ≤ 35 ∧
    SatisfiesCompositeSumCondition (swaps.foldl (fun g (i, j, k, l) => Swap g i j k l) initial) :=
  sorry


end NUMINAMATH_CALUDE_thirty_five_operations_sufficient_l2268_226810


namespace NUMINAMATH_CALUDE_distance_to_point_distance_from_origin_to_point_l2268_226883

theorem distance_to_point : ℝ → ℝ → ℝ
  | x, y => Real.sqrt (x^2 + y^2)

theorem distance_from_origin_to_point : distance_to_point (-8) 15 = 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_distance_from_origin_to_point_l2268_226883


namespace NUMINAMATH_CALUDE_fish_in_tank_l2268_226803

theorem fish_in_tank (total : ℕ) (blue : ℕ) (spotted : ℕ) : 
  (3 * blue = total) →  -- One third of the fish are blue
  (2 * spotted = blue) →  -- Half of the blue fish have spots
  (spotted = 10) →  -- There are 10 blue, spotted fish
  total = 60 := by
  sorry

end NUMINAMATH_CALUDE_fish_in_tank_l2268_226803


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2268_226887

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The roots of the equation x^2 - 2x - 3 = 0 -/
def roots_equation (x y : ℝ) : Prop :=
  x^2 - 2*x - 3 = 0 ∧ y^2 - 2*y - 3 = 0

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  roots_equation (a 1) (a 4) →
  a 2 * a 3 = -3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2268_226887


namespace NUMINAMATH_CALUDE_mikes_coins_value_l2268_226805

/-- Represents the number of coins Mike has -/
def total_coins : ℕ := 17

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Calculates the total value of Mike's coins in cents -/
def total_value (dimes quarters : ℕ) : ℕ :=
  dimes * dime_value + quarters * quarter_value

theorem mikes_coins_value :
  ∃ (dimes quarters : ℕ),
    dimes + quarters = total_coins ∧
    quarters + 3 = 2 * dimes ∧
    total_value dimes quarters = 345 := by
  sorry

end NUMINAMATH_CALUDE_mikes_coins_value_l2268_226805


namespace NUMINAMATH_CALUDE_solve_fraction_equation_l2268_226826

theorem solve_fraction_equation (x : ℝ) : (1 / 3 - 1 / 4 : ℝ) = 1 / x → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_fraction_equation_l2268_226826


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2268_226889

theorem arithmetic_computation : 2 + 5 * 3 - 4 + 6 * 2 / 3 = 17 := by sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2268_226889


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2268_226842

theorem quadratic_no_real_roots : ∀ x : ℝ, x^2 - 4 * Real.sqrt 2 * x + 9 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2268_226842


namespace NUMINAMATH_CALUDE_joan_rock_collection_l2268_226844

theorem joan_rock_collection (minerals_today minerals_yesterday gemstones : ℕ) : 
  gemstones = minerals_yesterday / 2 →
  minerals_today = minerals_yesterday + 6 →
  minerals_today = 48 →
  gemstones = 21 := by
sorry

end NUMINAMATH_CALUDE_joan_rock_collection_l2268_226844


namespace NUMINAMATH_CALUDE_expression_value_l2268_226881

theorem expression_value (x : ℝ) (h : 5 * x^2 - x - 2 = 0) :
  (2*x + 1) * (2*x - 1) + x * (x - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2268_226881


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_11_l2268_226899

theorem binomial_coefficient_19_11 :
  (Nat.choose 19 11 = 82654) ∧ (Nat.choose 17 9 = 24310) ∧ (Nat.choose 17 7 = 19448) → 
  Nat.choose 19 11 = 82654 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_11_l2268_226899


namespace NUMINAMATH_CALUDE_xiaoding_distance_l2268_226897

/-- Represents the distance to school for each student in meters -/
structure SchoolDistances where
  xiaoding : ℕ
  xiaowang : ℕ
  xiaocheng : ℕ
  xiaozhang : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (d : SchoolDistances) : Prop :=
  d.xiaowang + d.xiaoding + d.xiaocheng + d.xiaozhang = 705 ∧
  d.xiaowang = 4 * d.xiaoding ∧
  d.xiaocheng = d.xiaowang / 2 + 20 ∧
  d.xiaozhang = 2 * d.xiaocheng - 15

/-- The theorem to be proved -/
theorem xiaoding_distance (d : SchoolDistances) :
  satisfiesConditions d → d.xiaoding = 60 := by
  sorry


end NUMINAMATH_CALUDE_xiaoding_distance_l2268_226897


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l2268_226845

/-- For a cube with volume 8y cubic units and surface area 6y square units, y = 64 -/
theorem cube_volume_surface_area (y : ℝ) (h1 : y > 0) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*y ∧ 6*s^2 = 6*y) → y = 64 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l2268_226845


namespace NUMINAMATH_CALUDE_fraction_order_l2268_226886

theorem fraction_order : 
  let f₁ : ℚ := 16 / 12
  let f₂ : ℚ := 20 / 16
  let f₃ : ℚ := 18 / 14
  let f₄ : ℚ := 22 / 17
  f₂ < f₃ ∧ f₃ < f₄ ∧ f₄ < f₁ :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l2268_226886


namespace NUMINAMATH_CALUDE_smallest_square_area_l2268_226867

/-- The smallest area of a square containing non-overlapping 1x4 and 2x5 rectangles -/
theorem smallest_square_area (r1_width r1_height r2_width r2_height : ℕ) 
  (h1 : r1_width = 1 ∧ r1_height = 4)
  (h2 : r2_width = 2 ∧ r2_height = 5)
  (h_no_overlap : True)  -- Represents the non-overlapping condition
  (h_parallel : True)    -- Represents the parallel sides condition
  : ∃ (s : ℕ), s^2 = 81 ∧ ∀ (t : ℕ), (t ≥ r1_width ∧ t ≥ r1_height ∧ t ≥ r2_width ∧ t ≥ r2_height) → t^2 ≥ s^2 := by
  sorry

#check smallest_square_area

end NUMINAMATH_CALUDE_smallest_square_area_l2268_226867


namespace NUMINAMATH_CALUDE_inner_triangle_perimeter_l2268_226846

theorem inner_triangle_perimeter (a : ℝ) (h : a = 8) :
  let outer_leg := a
  let inner_leg := a - 1
  let inner_hypotenuse := inner_leg * Real.sqrt 2
  let inner_perimeter := 2 * inner_leg + inner_hypotenuse
  inner_perimeter = 14 + 7 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_perimeter_l2268_226846


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2268_226824

/-- Given a line L1 with equation x - 2y + 3 = 0, prove that the line L2 with equation 2x + y - 1 = 0
    passes through the point (-1, 3) and is perpendicular to L1. -/
theorem perpendicular_line_through_point :
  let L1 : ℝ × ℝ → Prop := fun (x, y) ↦ x - 2*y + 3 = 0
  let L2 : ℝ × ℝ → Prop := fun (x, y) ↦ 2*x + y - 1 = 0
  let point : ℝ × ℝ := (-1, 3)
  (L2 point) ∧ 
  (∀ (p q : ℝ × ℝ), L1 p ∧ L1 q ∧ p ≠ q → 
    let v1 := (p.1 - q.1, p.2 - q.2)
    let v2 := (1, 2)
    v1.1 * v2.1 + v1.2 * v2.2 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2268_226824


namespace NUMINAMATH_CALUDE_decagon_triangles_l2268_226890

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def num_triangles : ℕ := Nat.choose n k

theorem decagon_triangles : num_triangles = 120 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangles_l2268_226890


namespace NUMINAMATH_CALUDE_regular_decagon_area_l2268_226871

theorem regular_decagon_area (s : ℝ) (h : s = Real.sqrt 2) :
  let area := 5 * s^2 * (Real.sqrt (5 + 2 * Real.sqrt 5)) / 4
  area = 3.5 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_decagon_area_l2268_226871


namespace NUMINAMATH_CALUDE_passes_count_l2268_226860

/-- The number of times Griffin and Hailey pass each other during their run -/
def number_of_passes (
  run_time : ℝ)
  (griffin_speed : ℝ)
  (hailey_speed : ℝ)
  (griffin_radius : ℝ)
  (hailey_radius : ℝ) : ℕ :=
  sorry

theorem passes_count :
  number_of_passes 45 260 310 50 45 = 86 :=
sorry

end NUMINAMATH_CALUDE_passes_count_l2268_226860


namespace NUMINAMATH_CALUDE_max_ab_value_l2268_226848

/-- Given a line ax + by - 6 = 0 (a > 0, b > 0) intercepted by the circle x^2 + y^2 - 2x - 4y = 0
    to form a chord of length 2√5, the maximum value of ab is 9/2 -/
theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), a * x + b * y - 6 = 0 ∧ 
                x^2 + y^2 - 2*x - 4*y = 0 ∧ 
                ∃ (x1 y1 x2 y2 : ℝ), 
                  a * x1 + b * y1 - 6 = 0 ∧ 
                  x1^2 + y1^2 - 2*x1 - 4*y1 = 0 ∧
                  a * x2 + b * y2 - 6 = 0 ∧ 
                  x2^2 + y2^2 - 2*x2 - 4*y2 = 0 ∧
                  (x2 - x1)^2 + (y2 - y1)^2 = 20) →
  a * b ≤ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l2268_226848


namespace NUMINAMATH_CALUDE_school_journey_time_l2268_226812

/-- The time for a journey to school, given specific conditions about forgetting an item -/
theorem school_journey_time : ∃ (t : ℝ), 
  (t > 0) ∧ 
  (t - 6 > 0) ∧
  (∃ (x : ℝ), x > 0 ∧ x = t / 5) ∧
  ((9/5) * t = t + 2) ∧ 
  (t = 20) := by
  sorry

end NUMINAMATH_CALUDE_school_journey_time_l2268_226812


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2268_226837

/-- Given a line equation 3x + 5y + d = 0, proves that if the sum of x- and y-intercepts is 16, then d = -30 -/
theorem line_intercepts_sum (d : ℝ) : 
  (∃ x y : ℝ, 3 * x + 5 * y + d = 0 ∧ x + y = 16) → d = -30 :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2268_226837


namespace NUMINAMATH_CALUDE_hat_cost_calculation_l2268_226856

/-- The price of a wooden toy -/
def wooden_toy_price : ℕ := 20

/-- The number of wooden toys Kendra bought -/
def wooden_toys_bought : ℕ := 2

/-- The number of hats Kendra bought -/
def hats_bought : ℕ := 3

/-- The amount Kendra paid with -/
def amount_paid : ℕ := 100

/-- The change Kendra received -/
def change_received : ℕ := 30

/-- The cost of a hat -/
def hat_cost : ℕ := 10

theorem hat_cost_calculation :
  hat_cost = (amount_paid - change_received - wooden_toy_price * wooden_toys_bought) / hats_bought :=
by sorry

end NUMINAMATH_CALUDE_hat_cost_calculation_l2268_226856


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2268_226801

/-- The sum of the infinite series ∑(k=1 to ∞) k^2 / 2^k equals 6 -/
theorem infinite_series_sum : 
  (∑' k : ℕ, (k^2 : ℝ) / (2 : ℝ)^k) = 6 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2268_226801


namespace NUMINAMATH_CALUDE_sine_plus_abs_sine_integral_l2268_226841

open Set
open MeasureTheory
open Real

theorem sine_plus_abs_sine_integral : 
  ∫ x in (-π/2)..(π/2), (sin x + |sin x|) = 2 := by sorry

end NUMINAMATH_CALUDE_sine_plus_abs_sine_integral_l2268_226841


namespace NUMINAMATH_CALUDE_equation_and_inequalities_solution_l2268_226855

theorem equation_and_inequalities_solution :
  (∃! x : ℝ, (3 / (x - 1) = 1 / (2 * x + 3))) ∧
  (∀ x : ℝ, (3 * x - 1 ≥ x + 1 ∧ x + 3 > 4 * x - 2) ↔ (1 ≤ x ∧ x < 5 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_equation_and_inequalities_solution_l2268_226855


namespace NUMINAMATH_CALUDE_proposition_analysis_l2268_226832

theorem proposition_analysis (a b c : ℝ) : 
  (∀ x y z : ℝ, (x ≤ y → x*z^2 ≤ y*z^2)) ∧ 
  (∃ x y z : ℝ, (x > y ∧ x*z^2 ≤ y*z^2)) ∧
  (∀ x y z : ℝ, (x*z^2 > y*z^2 → x > y)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_analysis_l2268_226832


namespace NUMINAMATH_CALUDE_largest_circle_area_in_square_l2268_226825

/-- The area of the largest circle inside a square of side length 70 cm -/
theorem largest_circle_area_in_square : 
  let square_side : ℝ := 70
  let circle_area : ℝ := Real.pi * (square_side / 2)^2
  circle_area = 1225 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_largest_circle_area_in_square_l2268_226825


namespace NUMINAMATH_CALUDE_inequality_proof_l2268_226885

theorem inequality_proof (n : ℕ+) (x : ℝ) (hx : x > 0) :
  x + (n : ℝ)^(n : ℕ) / x^(n : ℕ) ≥ (n : ℝ) + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2268_226885


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l2268_226879

theorem triangle_angle_proof (a b c : ℝ) : 
  a = 60 → b = 40 → a + b + c = 180 → c = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l2268_226879


namespace NUMINAMATH_CALUDE_smallest_integer_l2268_226852

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 28) :
  ∀ c : ℕ, c > 0 ∧ Nat.lcm a c / Nat.gcd a c = 28 → b ≤ c → b = 105 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_l2268_226852


namespace NUMINAMATH_CALUDE_regression_for_related_variables_l2268_226830

/-- A type representing a statistical variable -/
structure StatVariable where
  name : String

/-- A type representing a statistical analysis method -/
inductive AnalysisMethod
  | ErrorAnalysis
  | RegressionAnalysis
  | IndependenceTest

/-- A relation indicating that two variables are related -/
def are_related (v1 v2 : StatVariable) : Prop := sorry

/-- The correct method to analyze related variables -/
def analyze_related_variables (v1 v2 : StatVariable) : AnalysisMethod :=
  AnalysisMethod.RegressionAnalysis

/-- Theorem stating that regression analysis is the correct method for analyzing related variables -/
theorem regression_for_related_variables (height weight : StatVariable) 
    (h : are_related height weight) : 
    analyze_related_variables height weight = AnalysisMethod.RegressionAnalysis := by
  sorry

end NUMINAMATH_CALUDE_regression_for_related_variables_l2268_226830


namespace NUMINAMATH_CALUDE_wedding_guests_count_l2268_226892

/-- The number of guests attending the wedding -/
def total_guests : ℕ := 240

/-- The proportion of female guests -/
def female_proportion : ℚ := 3/5

/-- The proportion of female guests from Jay's family -/
def jay_family_proportion : ℚ := 1/2

/-- The number of female guests from Jay's family -/
def jay_family_females : ℕ := 72

theorem wedding_guests_count :
  (jay_family_females : ℚ) = (total_guests : ℚ) * female_proportion * jay_family_proportion :=
by sorry

end NUMINAMATH_CALUDE_wedding_guests_count_l2268_226892


namespace NUMINAMATH_CALUDE_right_triangle_GHI_side_GH_l2268_226850

/-- Represents a right triangle GHI with angle G = 30°, angle H = 90°, and HI = 10 -/
structure RightTriangleGHI where
  G : Real
  H : Real
  I : Real
  angleG : G = 30 * π / 180
  angleH : H = π / 2
  rightAngle : H = π / 2
  sideHI : I = 10

/-- Theorem stating that in the given right triangle GHI, GH = 10√3 -/
theorem right_triangle_GHI_side_GH (t : RightTriangleGHI) : 
  Real.sqrt ((10 * Real.sqrt 3) ^ 2) = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_GHI_side_GH_l2268_226850


namespace NUMINAMATH_CALUDE_g_sum_equals_negative_two_l2268_226804

/-- Piecewise function g(x, y) -/
noncomputable def g (x y : ℝ) : ℝ :=
  if x - y ≤ 1 then (x^2 * y - x + 3) / (3 * x)
  else (x^2 * y - y - 3) / (-3 * y)

/-- Theorem stating that g(3,2) + g(4,1) = -2 -/
theorem g_sum_equals_negative_two : g 3 2 + g 4 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_equals_negative_two_l2268_226804


namespace NUMINAMATH_CALUDE_beginner_trig_probability_probability_calculation_l2268_226869

/-- Represents the number of students in each course -/
structure CourseEnrollment where
  BC : ℕ  -- Beginner Calculus
  AC : ℕ  -- Advanced Calculus
  IC : ℕ  -- Intermediate Calculus
  BT : ℕ  -- Beginner Trigonometry
  AT : ℕ  -- Advanced Trigonometry
  IT : ℕ  -- Intermediate Trigonometry

/-- Represents the enrollment conditions for the math department -/
def EnrollmentConditions (e : CourseEnrollment) (total : ℕ) : Prop :=
  e.BC + e.AC + e.IC = (60 * total) / 100 ∧
  e.BT + e.AT + e.IT = (40 * total) / 100 ∧
  e.BC + e.BT = (45 * total) / 100 ∧
  e.AC + e.AT = (35 * total) / 100 ∧
  e.IC + e.IT = (20 * total) / 100 ∧
  e.BC = (125 * e.BT) / 100 ∧
  e.IC + e.AC = (120 * (e.IT + e.AT)) / 100

theorem beginner_trig_probability (e : CourseEnrollment) (total : ℕ) :
  EnrollmentConditions e total → total = 5000 → e.BT = 1000 :=
by sorry

theorem probability_calculation (e : CourseEnrollment) (total : ℕ) :
  EnrollmentConditions e total → total = 5000 → e.BT = 1000 →
  (e.BT : ℚ) / total = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_beginner_trig_probability_probability_calculation_l2268_226869


namespace NUMINAMATH_CALUDE_first_group_size_l2268_226863

/-- The number of men in the first group -/
def M : ℕ := sorry

/-- The length of the wall built by the first group -/
def length1 : ℝ := 66

/-- The number of days taken by the first group -/
def days1 : ℕ := 8

/-- The number of men in the second group -/
def men2 : ℕ := 86

/-- The length of the wall built by the second group -/
def length2 : ℝ := 283.8

/-- The number of days taken by the second group -/
def days2 : ℕ := 8

/-- The work done is directly proportional to the number of men and the length of the wall -/
axiom work_proportional : ∀ (men : ℕ) (length : ℝ) (days : ℕ), 
  (men : ℝ) * length / days = (M : ℝ) * length1 / days1

theorem first_group_size : 
  ∃ (m : ℕ), (m : ℝ) ≥ 368.5 ∧ (m : ℝ) < 369.5 ∧ M = m :=
sorry

end NUMINAMATH_CALUDE_first_group_size_l2268_226863


namespace NUMINAMATH_CALUDE_andrews_grapes_l2268_226874

theorem andrews_grapes (price_grapes : ℕ) (quantity_mangoes : ℕ) (price_mangoes : ℕ) (total_paid : ℕ) :
  price_grapes = 74 →
  quantity_mangoes = 9 →
  price_mangoes = 59 →
  total_paid = 975 →
  ∃ (quantity_grapes : ℕ), 
    quantity_grapes * price_grapes + quantity_mangoes * price_mangoes = total_paid ∧
    quantity_grapes = 6 := by
  sorry

end NUMINAMATH_CALUDE_andrews_grapes_l2268_226874


namespace NUMINAMATH_CALUDE_emily_egg_collection_l2268_226834

/-- The number of baskets Emily used --/
def num_baskets : ℕ := 1525

/-- The average number of eggs per basket --/
def eggs_per_basket : ℚ := 37.5

/-- The total number of eggs collected --/
def total_eggs : ℚ := num_baskets * eggs_per_basket

/-- Theorem stating that the total number of eggs is 57,187.5 --/
theorem emily_egg_collection :
  total_eggs = 57187.5 := by sorry

end NUMINAMATH_CALUDE_emily_egg_collection_l2268_226834


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l2268_226800

theorem longest_side_of_triangle (a b c : ℝ) (perimeter : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a / b = 3 / 2 →
  a / c = 2 →
  b / c = 4 / 3 →
  a + b + c = perimeter →
  perimeter = 104 →
  a = 48 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l2268_226800


namespace NUMINAMATH_CALUDE_kendall_change_total_l2268_226811

/-- Represents the value of coins in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- Calculates the total value of a given number of coins -/
def coin_total (coin : String) (count : ℕ) : ℕ :=
  (coin_value coin) * count

/-- Theorem stating the total amount of money Kendall has in change -/
theorem kendall_change_total : 
  coin_total "quarter" 10 + coin_total "dime" 12 + coin_total "nickel" 6 = 400 := by
  sorry

end NUMINAMATH_CALUDE_kendall_change_total_l2268_226811


namespace NUMINAMATH_CALUDE_common_root_implies_zero_l2268_226833

theorem common_root_implies_zero (a b : ℝ) : 
  (∃ r : ℝ, r^2 + a*r + b^2 = 0 ∧ r^2 + b*r + a^2 = 0) → 
  ¬(a ≠ 0 ∧ b ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_common_root_implies_zero_l2268_226833


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l2268_226807

theorem roots_sum_of_squares (m : ℤ) (a b c : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) → 
  a^2 + b^2 + c^2 = 4046 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l2268_226807


namespace NUMINAMATH_CALUDE_problem_statement_l2268_226876

theorem problem_statement (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a ^ b = b ^ a) (h2 : b = 9 * a) : a = (3 : ℝ) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2268_226876


namespace NUMINAMATH_CALUDE_rose_pollen_diameter_scientific_notation_l2268_226882

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The diameter of the rose pollen in meters -/
def rose_pollen_diameter : ℝ := 0.0000028

/-- The scientific notation representation of the rose pollen diameter -/
def rose_pollen_scientific : ScientificNotation :=
  { coefficient := 2.8
  , exponent := -6
  , is_valid := by sorry }

/-- Theorem stating that the rose pollen diameter is correctly expressed in scientific notation -/
theorem rose_pollen_diameter_scientific_notation :
  rose_pollen_diameter = rose_pollen_scientific.coefficient * (10 : ℝ) ^ rose_pollen_scientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_rose_pollen_diameter_scientific_notation_l2268_226882


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2268_226821

theorem complex_fraction_equality : (3 - I) / (1 - I) = 2 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2268_226821


namespace NUMINAMATH_CALUDE_no_real_intersection_l2268_226875

theorem no_real_intersection : ¬∃ x : ℝ, 3 * x^2 - 6 * x + 5 = 0 := by sorry

end NUMINAMATH_CALUDE_no_real_intersection_l2268_226875


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2268_226865

theorem inequality_equivalence (x : ℝ) : 
  2 / (x + 2) + 4 / (x + 8) ≤ 7 / 3 ↔ -8 < x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2268_226865


namespace NUMINAMATH_CALUDE_point_in_region_t_range_l2268_226827

/-- Given a point (1, t) in the region represented by x - y + 1 > 0, 
    the range of values for t is t < 2 -/
theorem point_in_region_t_range (t : ℝ) : 
  (1 : ℝ) - t + 1 > 0 → t < 2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_region_t_range_l2268_226827


namespace NUMINAMATH_CALUDE_cos_equation_rational_solution_l2268_226878

theorem cos_equation_rational_solution (a : ℚ) 
  (h1 : 0 < a) (h2 : a < 1) 
  (h3 : Real.cos (3 * Real.pi * a) + 2 * Real.cos (2 * Real.pi * a) = 0) : 
  a = 2/3 := by
sorry

end NUMINAMATH_CALUDE_cos_equation_rational_solution_l2268_226878


namespace NUMINAMATH_CALUDE_total_books_on_shelves_l2268_226891

theorem total_books_on_shelves (num_shelves : ℕ) (books_per_shelf : ℕ) :
  num_shelves = 625 → books_per_shelf = 28 → num_shelves * books_per_shelf = 22500 := by
  sorry

end NUMINAMATH_CALUDE_total_books_on_shelves_l2268_226891


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2268_226847

theorem quadratic_one_solution (p : ℝ) : 
  (∃! x : ℝ, 3 * x^2 - 6 * x + p = 0) ↔ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2268_226847


namespace NUMINAMATH_CALUDE_union_of_sets_l2268_226857

open Set

theorem union_of_sets (U A B : Set ℕ) : 
  U = {1, 2, 3, 4, 5, 6} →
  (U \ A) = {1, 2, 4} →
  (U \ B) = {3, 4, 5} →
  A ∪ B = {1, 2, 3, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2268_226857


namespace NUMINAMATH_CALUDE_P_equals_Q_l2268_226835

-- Define a one-to-one, strictly increasing function f: R → R
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_injective : Function.Injective f
axiom f_strictly_increasing : ∀ x y, x < y → f x < f y

-- Define the sets P and Q
def P : Set ℝ := {x | x > f x}
def Q : Set ℝ := {x | x > f (f x)}

-- State the theorem
theorem P_equals_Q : P = Q := by sorry

end NUMINAMATH_CALUDE_P_equals_Q_l2268_226835


namespace NUMINAMATH_CALUDE_multiplicative_inverse_123_mod_456_l2268_226806

theorem multiplicative_inverse_123_mod_456 :
  ∃ (x : ℕ), x < 456 ∧ (123 * x) % 456 = 1 :=
by
  use 52
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_123_mod_456_l2268_226806


namespace NUMINAMATH_CALUDE_tan_angle_through_P_l2268_226808

/-- An angle in the coordinate plane -/
structure Angle :=
  (initial_side : Set (ℝ × ℝ))
  (terminal_side : Set (ℝ × ℝ))

/-- The tangent of an angle -/
def tan (α : Angle) : ℝ := sorry

/-- The non-negative half-axis of the x-axis -/
def non_negative_x_axis : Set (ℝ × ℝ) := sorry

/-- A point P(-2,1) in the coordinate plane -/
def point_P : ℝ × ℝ := (-2, 1)

/-- The line passing through the origin and point P -/
def line_through_origin_and_P : Set (ℝ × ℝ) := sorry

theorem tan_angle_through_P :
  ∀ α : Angle,
  α.initial_side = non_negative_x_axis →
  α.terminal_side = line_through_origin_and_P →
  tan α = -1/2 := by sorry

end NUMINAMATH_CALUDE_tan_angle_through_P_l2268_226808


namespace NUMINAMATH_CALUDE_tan_value_from_ratio_l2268_226895

theorem tan_value_from_ratio (α : Real) :
  (Real.sin α + 7 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = -5 →
  Real.tan α = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_ratio_l2268_226895


namespace NUMINAMATH_CALUDE_train_distance_l2268_226872

/-- Represents the efficiency of a coal-powered train in miles per pound of coal -/
def train_efficiency : ℚ := 5 / 2

/-- Represents the amount of coal remaining in pounds -/
def coal_remaining : ℕ := 160

/-- Calculates the distance a train can travel given its efficiency and remaining coal -/
def distance_traveled (efficiency : ℚ) (coal : ℕ) : ℚ :=
  efficiency * coal

/-- Theorem stating that the train can travel 400 miles before running out of fuel -/
theorem train_distance : distance_traveled train_efficiency coal_remaining = 400 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l2268_226872


namespace NUMINAMATH_CALUDE_number_puzzle_l2268_226831

theorem number_puzzle (x : ℤ) : x - 62 + 45 = 55 → 7 * x = 504 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2268_226831


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l2268_226888

theorem least_integer_satisfying_inequality :
  ∀ x : ℤ, (3 * |x| - 2 > 13) → x ≥ -6 :=
by
  sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l2268_226888


namespace NUMINAMATH_CALUDE_remaining_balance_calculation_l2268_226843

def cost_per_charge : ℚ := 3.5
def number_of_charges : ℕ := 4
def initial_budget : ℚ := 20

theorem remaining_balance_calculation :
  initial_budget - (cost_per_charge * number_of_charges) = 6 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balance_calculation_l2268_226843


namespace NUMINAMATH_CALUDE_slope_of_line_l2268_226873

/-- The slope of a line given by the equation 4y = -6x + 12 is -3/2 -/
theorem slope_of_line (x y : ℝ) : 4 * y = -6 * x + 12 → (y - 3) / x = -3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l2268_226873


namespace NUMINAMATH_CALUDE_solution_set_equation_l2268_226894

theorem solution_set_equation : 
  ∀ x : ℝ, ((x - 1) / x)^2 - (7/2) * ((x - 1) / x) + 3 = 0 ↔ x = -1 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equation_l2268_226894


namespace NUMINAMATH_CALUDE_jack_plates_left_l2268_226877

/-- Represents the number of plates Jack has with different patterns -/
structure PlateCollection where
  flower : ℕ
  checked : ℕ
  polkadot : ℕ

/-- Calculates the total number of plates after Jack's actions -/
def total_plates_after_actions (initial : PlateCollection) : ℕ :=
  (initial.flower - 1) + initial.checked + (2 * initial.checked)

/-- Theorem stating that Jack has 27 plates left after his actions -/
theorem jack_plates_left (initial : PlateCollection) 
  (h1 : initial.flower = 4)
  (h2 : initial.checked = 8)
  (h3 : initial.polkadot = 0) : 
  total_plates_after_actions initial = 27 := by
  sorry

#check jack_plates_left

end NUMINAMATH_CALUDE_jack_plates_left_l2268_226877


namespace NUMINAMATH_CALUDE_square_sum_of_special_integers_l2268_226836

theorem square_sum_of_special_integers (x y : ℕ+) 
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1870)
  (h3 : x < y) :
  x^2 + y^2 = 986 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_special_integers_l2268_226836


namespace NUMINAMATH_CALUDE_six_grade_sequences_l2268_226868

/-- Represents the number of ways to assign n grades under the given conditions -/
def gradeSequences (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 3
  | n + 2 => 2 * gradeSequences (n + 1) + 2 * gradeSequences n

/-- The number of ways to assign 6 grades under the given conditions is 448 -/
theorem six_grade_sequences : gradeSequences 6 = 448 := by
  sorry

/-- Helper lemma: The recurrence relation holds for all n ≥ 2 -/
lemma recurrence_relation (n : ℕ) (h : n ≥ 2) :
  gradeSequences n = 2 * gradeSequences (n - 1) + 2 * gradeSequences (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_six_grade_sequences_l2268_226868


namespace NUMINAMATH_CALUDE_farmer_theorem_l2268_226818

def farmer_problem (initial_tomatoes initial_potatoes remaining_total : ℕ) : ℕ :=
  (initial_tomatoes + initial_potatoes) - remaining_total

theorem farmer_theorem (initial_tomatoes initial_potatoes remaining_total : ℕ) :
  farmer_problem initial_tomatoes initial_potatoes remaining_total =
  (initial_tomatoes + initial_potatoes) - remaining_total :=
by sorry

#eval farmer_problem 175 77 80

end NUMINAMATH_CALUDE_farmer_theorem_l2268_226818


namespace NUMINAMATH_CALUDE_shaded_areas_sum_l2268_226838

theorem shaded_areas_sum (R : ℝ) (h1 : R > 0) (h2 : π * R^2 = 81 * π) : 
  (π * R^2) / 2 + (π * (R/2)^2) / 2 = 50.625 * π := by sorry

end NUMINAMATH_CALUDE_shaded_areas_sum_l2268_226838
