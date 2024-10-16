import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l3589_358970

theorem problem_solution (a b c : ℝ) 
  (h1 : (a + b + c)^2 = 3*(a^2 + b^2 + c^2)) 
  (h2 : a + b + c = 12) : 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3589_358970


namespace NUMINAMATH_CALUDE_g_monotone_decreasing_iff_a_in_range_l3589_358999

/-- The function g(x) defined as ax³ + 2(1-a)x² - 3ax -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * (1 - a) * x^2 - 3 * a * x

/-- g(x) is monotonically decreasing in the interval (-∞, a/3) if and only if -1 ≤ a ≤ 0 -/
theorem g_monotone_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x y, x < y → y < a/3 → g a x > g a y) ↔ -1 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_g_monotone_decreasing_iff_a_in_range_l3589_358999


namespace NUMINAMATH_CALUDE_last_released_theorem_l3589_358911

/-- The position of the last released captive's servant -/
def last_released_position (N : ℕ) (total_purses : ℕ) : Set ℕ :=
  if total_purses = N + (N - 1) * N / 2
  then {N}
  else if total_purses = N + (N - 1) * N / 2 - 1
  then {N - 1, N}
  else ∅

/-- The main theorem about the position of the last released captive's servant -/
theorem last_released_theorem (N : ℕ) (total_purses : ℕ) 
  (h1 : N > 0) 
  (h2 : total_purses ≥ N) 
  (h3 : total_purses ≤ N + (N - 1) * N / 2) :
  (last_released_position N total_purses).Nonempty := by
  sorry

end NUMINAMATH_CALUDE_last_released_theorem_l3589_358911


namespace NUMINAMATH_CALUDE_prob_sum_25_l3589_358904

/-- Represents a 20-faced die with specific numbering --/
structure Die :=
  (faces : Finset ℕ)
  (blank : Bool)
  (proper : faces.card + (if blank then 1 else 0) = 20)

/-- The first die with faces 1-19 and one blank --/
def die1 : Die :=
  { faces := Finset.range 20 \ {0},
    blank := true,
    proper := by sorry }

/-- The second die with faces 1-7, 9-19 and one blank --/
def die2 : Die :=
  { faces := (Finset.range 20 \ {0, 8}),
    blank := true,
    proper := by sorry }

/-- The probability of an event given the sample space --/
def probability (event : Finset (ℕ × ℕ)) (sample_space : Finset (ℕ × ℕ)) : ℚ :=
  event.card / sample_space.card

/-- The set of all possible outcomes when rolling both dice --/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product die1.faces die2.faces

/-- The set of outcomes where the sum is 25 --/
def sum_25_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun p => p.1 + p.2 = 25)

/-- The main theorem stating the probability of rolling a sum of 25 --/
theorem prob_sum_25 :
  probability sum_25_outcomes all_outcomes = 13 / 400 := by sorry

end NUMINAMATH_CALUDE_prob_sum_25_l3589_358904


namespace NUMINAMATH_CALUDE_square_difference_63_57_l3589_358946

theorem square_difference_63_57 : 63^2 - 57^2 = 720 := by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_square_difference_63_57_l3589_358946


namespace NUMINAMATH_CALUDE_triangle_area_approx_l3589_358930

/-- The area of a triangle with sides 30, 28, and 10 is approximately 139.94 -/
theorem triangle_area_approx : ∃ (area : ℝ), 
  let a : ℝ := 30
  let b : ℝ := 28
  let c : ℝ := 10
  let s : ℝ := (a + b + c) / 2
  area = Real.sqrt (s * (s - a) * (s - b) * (s - c)) ∧ 
  abs (area - 139.94) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_approx_l3589_358930


namespace NUMINAMATH_CALUDE_sqrt_combination_l3589_358983

theorem sqrt_combination (t : ℝ) : 
  (∃ k : ℝ, k * Real.sqrt 12 = Real.sqrt (2 * t - 1)) → 
  Real.sqrt 12 = 2 * Real.sqrt 3 → 
  t = 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_combination_l3589_358983


namespace NUMINAMATH_CALUDE_hash_toy_difference_l3589_358941

theorem hash_toy_difference (bill_toys : ℕ) (total_toys : ℕ) : 
  bill_toys = 60 →
  total_toys = 99 →
  total_toys - bill_toys - bill_toys / 2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_hash_toy_difference_l3589_358941


namespace NUMINAMATH_CALUDE_jeffreys_poultry_farm_l3589_358909

/-- The number of roosters for every 3 hens on Jeffrey's poultry farm -/
def roosters_per_three_hens : ℕ := by sorry

theorem jeffreys_poultry_farm :
  let total_hens : ℕ := 12
  let chicks_per_hen : ℕ := 5
  let total_chickens : ℕ := 76
  roosters_per_three_hens = 1 := by sorry

end NUMINAMATH_CALUDE_jeffreys_poultry_farm_l3589_358909


namespace NUMINAMATH_CALUDE_unique_prime_sum_10123_l3589_358969

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem unique_prime_sum_10123 :
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 10123 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_sum_10123_l3589_358969


namespace NUMINAMATH_CALUDE_defective_bulb_probability_l3589_358962

/-- The probability of selecting at least one defective bulb when choosing two bulbs at random from a box containing 24 bulbs, of which 4 are defective, is 43/138. -/
theorem defective_bulb_probability (total_bulbs : ℕ) (defective_bulbs : ℕ) 
  (h1 : total_bulbs = 24) (h2 : defective_bulbs = 4) :
  let non_defective : ℕ := total_bulbs - defective_bulbs
  let prob_both_non_defective : ℚ := (non_defective / total_bulbs) * ((non_defective - 1) / (total_bulbs - 1))
  1 - prob_both_non_defective = 43 / 138 := by
  sorry

end NUMINAMATH_CALUDE_defective_bulb_probability_l3589_358962


namespace NUMINAMATH_CALUDE_shaniqua_haircuts_l3589_358919

/-- Represents the pricing and earnings of a hairstylist --/
structure HairstylistEarnings where
  haircut_price : ℕ
  style_price : ℕ
  total_earnings : ℕ
  num_styles : ℕ

/-- Calculates the number of haircuts given the hairstylist's earnings information --/
def calculate_haircuts (e : HairstylistEarnings) : ℕ :=
  (e.total_earnings - e.style_price * e.num_styles) / e.haircut_price

/-- Theorem stating that given Shaniqua's earnings information, she gave 8 haircuts --/
theorem shaniqua_haircuts :
  let e : HairstylistEarnings := {
    haircut_price := 12,
    style_price := 25,
    total_earnings := 221,
    num_styles := 5
  }
  calculate_haircuts e = 8 := by sorry

end NUMINAMATH_CALUDE_shaniqua_haircuts_l3589_358919


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sequence_l3589_358900

theorem triangle_arithmetic_sequence (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- c cos A, b cos B, a cos C form an arithmetic sequence
  2 * b * Real.cos B = c * Real.cos A + a * Real.cos C →
  -- Given conditions
  a + c = 3 * Real.sqrt 3 / 2 →
  b = Real.sqrt 3 →
  -- Conclusions
  B = π / 3 ∧
  (1 / 2 * a * c * Real.sin B = 5 * Real.sqrt 3 / 16) :=
by sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sequence_l3589_358900


namespace NUMINAMATH_CALUDE_inequality_proof_l3589_358935

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y ≥ 1) :
  x^3 + y^3 + 4*x*y ≥ x^2 + y^2 + x + y + 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3589_358935


namespace NUMINAMATH_CALUDE_abs_plus_power_minus_sqrt_inequality_system_solution_l3589_358927

-- Part 1
theorem abs_plus_power_minus_sqrt : |-2| + (1 + Real.sqrt 3)^0 - Real.sqrt 9 = 0 := by
  sorry

-- Part 2
theorem inequality_system_solution (x : ℝ) :
  (2 * x + 1 > 3 * (x - 1) ∧ x + (x - 1) / 3 < 1) ↔ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_plus_power_minus_sqrt_inequality_system_solution_l3589_358927


namespace NUMINAMATH_CALUDE_potato_yield_increase_l3589_358998

theorem potato_yield_increase (initial_area initial_yield final_area : ℝ) 
  (h1 : initial_area = 27)
  (h2 : final_area = 24)
  (h3 : initial_area * initial_yield = final_area * (initial_yield * (1 + yield_increase_percentage / 100))) :
  yield_increase_percentage = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_potato_yield_increase_l3589_358998


namespace NUMINAMATH_CALUDE_negative_5643_mod_10_l3589_358994

theorem negative_5643_mod_10 :
  ∃! (n : ℤ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5643 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_negative_5643_mod_10_l3589_358994


namespace NUMINAMATH_CALUDE_no_linear_term_condition_l3589_358976

theorem no_linear_term_condition (p q : ℝ) : 
  (∀ x : ℝ, ∃ a b c : ℝ, (x^2 - p*x + q)*(x - 3) = a*x^3 + b*x^2 + c) → 
  q + 3*p = 0 := by
sorry

end NUMINAMATH_CALUDE_no_linear_term_condition_l3589_358976


namespace NUMINAMATH_CALUDE_no_solution_lcm_equation_l3589_358958

theorem no_solution_lcm_equation :
  ¬ ∃ (a b : ℕ), 2 * a + 3 * b = Nat.lcm a b := by
  sorry

end NUMINAMATH_CALUDE_no_solution_lcm_equation_l3589_358958


namespace NUMINAMATH_CALUDE_max_value_sine_cosine_sum_l3589_358992

theorem max_value_sine_cosine_sum (x : ℝ) : 6 * Real.sin x + 8 * Real.cos x ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sine_cosine_sum_l3589_358992


namespace NUMINAMATH_CALUDE_monomial_sum_condition_l3589_358954

/-- If the sum of the monomials $-2x^{4}y^{m-1}$ and $5x^{n-1}y^{2}$ is a monomial, then $m-2n = -7$. -/
theorem monomial_sum_condition (m n : ℤ) : 
  (∃ (a : ℚ) (b c : ℕ), -2 * X^4 * Y^(m-1) + 5 * X^(n-1) * Y^2 = a * X^b * Y^c) → 
  m - 2*n = -7 :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_condition_l3589_358954


namespace NUMINAMATH_CALUDE_hamburger_buns_cost_l3589_358951

/-- The cost of hamburger buns given Lauren's grocery purchase --/
theorem hamburger_buns_cost : 
  ∀ (meat_price meat_weight lettuce_price tomato_price tomato_weight
     pickle_price pickle_discount paid change bun_price : ℝ),
  meat_price = 3.5 →
  meat_weight = 2 →
  lettuce_price = 1 →
  tomato_price = 2 →
  tomato_weight = 1.5 →
  pickle_price = 2.5 →
  pickle_discount = 1 →
  paid = 20 →
  change = 6 →
  bun_price = paid - change - (meat_price * meat_weight + lettuce_price + 
    tomato_price * tomato_weight + pickle_price - pickle_discount) →
  bun_price = 1.5 := by
sorry

end NUMINAMATH_CALUDE_hamburger_buns_cost_l3589_358951


namespace NUMINAMATH_CALUDE_split_tree_sum_lower_bound_l3589_358923

/-- Represents a tree where each node splits into two children that sum to the parent -/
inductive SplitTree : Nat → Type
  | leaf : SplitTree 1
  | node : (n : Nat) → (left right : Nat) → left + right = n → 
           SplitTree left → SplitTree right → SplitTree n

/-- The sum of all numbers in a SplitTree -/
def treeSum : {n : Nat} → SplitTree n → Nat
  | _, SplitTree.leaf => 1
  | n, SplitTree.node _ left right _ leftTree rightTree => 
      n + treeSum leftTree + treeSum rightTree

/-- Theorem: The sum of all numbers in a SplitTree starting with 2^n is at least n * 2^n -/
theorem split_tree_sum_lower_bound (n : Nat) (tree : SplitTree (2^n)) :
  treeSum tree ≥ n * 2^n := by
  sorry

end NUMINAMATH_CALUDE_split_tree_sum_lower_bound_l3589_358923


namespace NUMINAMATH_CALUDE_calculation_proof_l3589_358966

theorem calculation_proof : (π - 1) ^ 0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3589_358966


namespace NUMINAMATH_CALUDE_point_coordinates_on_terminal_side_l3589_358915

/-- Given a point P on the terminal side of -π/4 with |OP| = 2, prove its coordinates are (√2, -√2) -/
theorem point_coordinates_on_terminal_side (P : ℝ × ℝ) :
  (P.1 = Real.sqrt 2 ∧ P.2 = -Real.sqrt 2) ↔
  (∃ (r : ℝ), r > 0 ∧ P.1 = r * Real.cos (-π/4) ∧ P.2 = r * Real.sin (-π/4) ∧ r^2 = P.1^2 + P.2^2 ∧ r = 2) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_on_terminal_side_l3589_358915


namespace NUMINAMATH_CALUDE_max_selection_ways_l3589_358932

/-- The total number of socks -/
def total_socks : ℕ := 2017

/-- The function to calculate the number of ways to select socks -/
def selection_ways (partition : List ℕ) : ℕ :=
  partition.prod

/-- The theorem stating the maximum number of ways to select socks -/
theorem max_selection_ways :
  ∃ (partition : List ℕ),
    partition.sum = total_socks ∧
    ∀ (other_partition : List ℕ),
      other_partition.sum = total_socks →
      selection_ways other_partition ≤ selection_ways partition ∧
      selection_ways partition = 3^671 * 4 :=
sorry

end NUMINAMATH_CALUDE_max_selection_ways_l3589_358932


namespace NUMINAMATH_CALUDE_sin_480_degrees_l3589_358936

theorem sin_480_degrees : Real.sin (480 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_480_degrees_l3589_358936


namespace NUMINAMATH_CALUDE_distance_inequality_l3589_358917

theorem distance_inequality (a : ℝ) : 
  (|a - 1| < 3) → (-2 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_distance_inequality_l3589_358917


namespace NUMINAMATH_CALUDE_extra_interest_proof_l3589_358937

/-- Calculates simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem extra_interest_proof (principal : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time : ℝ) :
  principal = 5000 ∧ rate1 = 0.18 ∧ rate2 = 0.12 ∧ time = 2 →
  simpleInterest principal rate1 time - simpleInterest principal rate2 time = 600 := by
  sorry

end NUMINAMATH_CALUDE_extra_interest_proof_l3589_358937


namespace NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l3589_358953

theorem cos_pi_third_minus_alpha (α : Real) 
  (h : Real.sin (α + π / 6) = 2 * Real.sqrt 5 / 5) : 
  Real.cos (π / 3 - α) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l3589_358953


namespace NUMINAMATH_CALUDE_passenger_difference_l3589_358950

structure BusRoute where
  initial_passengers : ℕ
  first_passengers : ℕ
  final_passengers : ℕ
  terminal_passengers : ℕ

def BusRoute.valid (route : BusRoute) : Prop :=
  route.initial_passengers = 30 ∧
  route.terminal_passengers = 14 ∧
  route.first_passengers * 3 = route.final_passengers

theorem passenger_difference (route : BusRoute) (h : route.valid) :
  ∃ y : ℕ, route.first_passengers + y = route.initial_passengers + 6 :=
by sorry

end NUMINAMATH_CALUDE_passenger_difference_l3589_358950


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l3589_358920

theorem smallest_k_no_real_roots : 
  ∃ (k : ℤ), k = 2 ∧ 
  (∀ x : ℝ, 2 * x * (k * x - 4) - x^2 + 6 ≠ 0) ∧
  (∀ m : ℤ, m < k → ∃ x : ℝ, 2 * x * (m * x - 4) - x^2 + 6 = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l3589_358920


namespace NUMINAMATH_CALUDE_increase_mode_effect_l3589_358952

def shoe_sizes : List ℕ := [35, 36, 37, 38, 39]
def sales_quantities : List ℕ := [2, 8, 10, 6, 2]

def mode (l : List ℕ) : ℕ := sorry

def mean (l : List ℕ) : ℚ := sorry

def median (l : List ℕ) : ℚ := sorry

def variance (l : List ℕ) : ℚ := sorry

theorem increase_mode_effect 
  (most_common : ℕ) 
  (h1 : most_common ∈ shoe_sizes) 
  (h2 : ∀ x ∈ shoe_sizes, (sales_quantities.count most_common) ≥ (sales_quantities.count x)) :
  ∃ n : ℕ, 
    (mode (sales_quantities.map (λ x => if x = most_common then x + n else x)) = mode sales_quantities) ∧
    (mean (sales_quantities.map (λ x => if x = most_common then x + n else x)) ≠ mean sales_quantities ∨
     median (sales_quantities.map (λ x => if x = most_common then x + n else x)) = median sales_quantities ∨
     variance (sales_quantities.map (λ x => if x = most_common then x + n else x)) ≠ variance sales_quantities) :=
by sorry

end NUMINAMATH_CALUDE_increase_mode_effect_l3589_358952


namespace NUMINAMATH_CALUDE_syrup_volume_in_tank_syrup_volume_specific_l3589_358928

/-- The volume of syrup in a partially filled cylindrical tank -/
theorem syrup_volume_in_tank (tank_height : ℝ) (tank_diameter : ℝ) 
  (fill_ratio : ℝ) (syrup_ratio : ℝ) : ℝ :=
  let tank_radius : ℝ := tank_diameter / 2
  let liquid_height : ℝ := fill_ratio * tank_height
  let liquid_volume : ℝ := Real.pi * tank_radius^2 * liquid_height
  let syrup_volume : ℝ := liquid_volume * syrup_ratio / (1 + 1/syrup_ratio)
  syrup_volume

/-- The volume of syrup in the specific tank described in the problem -/
theorem syrup_volume_specific : 
  ∃ (ε : ℝ), abs (syrup_volume_in_tank 9 4 (1/3) (1/5) - 6.28) < ε ∧ ε < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_syrup_volume_in_tank_syrup_volume_specific_l3589_358928


namespace NUMINAMATH_CALUDE_monitor_height_is_seven_l3589_358978

/-- Represents a rectangular monitor -/
structure RectangularMonitor where
  width : ℝ
  height : ℝ

/-- The circumference of a rectangular monitor -/
def circumference (m : RectangularMonitor) : ℝ :=
  2 * (m.width + m.height)

/-- Theorem: A rectangular monitor with width 12 cm and circumference 38 cm has a height of 7 cm -/
theorem monitor_height_is_seven :
  ∃ (m : RectangularMonitor), m.width = 12 ∧ circumference m = 38 → m.height = 7 :=
by sorry

end NUMINAMATH_CALUDE_monitor_height_is_seven_l3589_358978


namespace NUMINAMATH_CALUDE_min_value_xy_l3589_358940

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y + 6 = x*y) :
  x * y ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_l3589_358940


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3589_358995

theorem fraction_subtraction : (3 + 6 + 9) / (2 + 5 + 7) - (2 + 5 + 7) / (3 + 6 + 9) = 32 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3589_358995


namespace NUMINAMATH_CALUDE_fraction_simplification_l3589_358924

theorem fraction_simplification :
  (3/6 + 4/5) / (5/12 + 1/4) = 39/20 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3589_358924


namespace NUMINAMATH_CALUDE_subtraction_puzzle_l3589_358973

theorem subtraction_puzzle (X Y : ℕ) : 
  X ≤ 9 → Y ≤ 9 → 45 + 8 * Y = 100 + 10 * X + 2 → X + Y = 10 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_puzzle_l3589_358973


namespace NUMINAMATH_CALUDE_abcd_sum_proof_l3589_358925

/-- Given four different digits A, B, C, and D forming a four-digit number ABCD,
    prove that if ABCD + ABCD = 7314, then ABCD = 3657 -/
theorem abcd_sum_proof (A B C D : ℕ) (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
    (h2 : 1000 ≤ A * 1000 + B * 100 + C * 10 + D ∧ A * 1000 + B * 100 + C * 10 + D < 10000)
    (h3 : (A * 1000 + B * 100 + C * 10 + D) + (A * 1000 + B * 100 + C * 10 + D) = 7314) :
  A * 1000 + B * 100 + C * 10 + D = 3657 := by
  sorry

end NUMINAMATH_CALUDE_abcd_sum_proof_l3589_358925


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l3589_358979

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 24)
  (area2 : w * h = 16)
  (area3 : l * h = 6) :
  l * w * h = 48 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l3589_358979


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3589_358968

/-- An arithmetic sequence with common difference 1 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 1

/-- Three terms form a geometric sequence -/
def geometric_seq (x y z : ℝ) : Prop :=
  y^2 = x * z

/-- The main theorem -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a)
  (h_geom : geometric_seq (a 1) (a 3) (a 7)) :
  a 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3589_358968


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3589_358939

/-- Given a hyperbola E with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and eccentricity √7/2,
    prove that its asymptotes have the equation y = ±(√3/2)x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt 7 / 2
  let c := e * a
  (c^2 / a^2 = 1 + (b/a)^2) →
  (b/a = Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3589_358939


namespace NUMINAMATH_CALUDE_team_savings_is_36_dollars_l3589_358918

-- Define the prices and team size
def regular_shirt_price : ℝ := 7.50
def regular_pants_price : ℝ := 15.00
def regular_socks_price : ℝ := 4.50
def discounted_shirt_price : ℝ := 6.75
def discounted_pants_price : ℝ := 13.50
def discounted_socks_price : ℝ := 3.75
def team_size : ℕ := 12

-- Define the total savings function
def total_savings : ℝ :=
  let regular_uniform_price := regular_shirt_price + regular_pants_price + regular_socks_price
  let discounted_uniform_price := discounted_shirt_price + discounted_pants_price + discounted_socks_price
  let savings_per_uniform := regular_uniform_price - discounted_uniform_price
  savings_per_uniform * team_size

-- Theorem statement
theorem team_savings_is_36_dollars : total_savings = 36 := by
  sorry

end NUMINAMATH_CALUDE_team_savings_is_36_dollars_l3589_358918


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l3589_358905

/-- Given a quadratic equation x^2 + 2(a-1)x + 2a + 6 = 0 with one positive and one negative real root,
    prove that a < -3 --/
theorem quadratic_root_condition (a : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ 
    x^2 + 2*(a-1)*x + 2*a + 6 = 0 ∧
    y^2 + 2*(a-1)*y + 2*a + 6 = 0) →
  a < -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l3589_358905


namespace NUMINAMATH_CALUDE_exists_two_sum_of_squares_representations_l3589_358971

theorem exists_two_sum_of_squares_representations : 
  ∃ (n : ℕ) (a b c d : ℕ), 
    n < 100 ∧ 
    a ≠ b ∧ 
    c ≠ d ∧ 
    (a, b) ≠ (c, d) ∧
    (a, b) ≠ (d, c) ∧
    n = a^2 + b^2 ∧ 
    n = c^2 + d^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_two_sum_of_squares_representations_l3589_358971


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l3589_358993

theorem simplify_nested_roots (x : ℝ) :
  (((x ^ 16) ^ (1 / 8)) ^ (1 / 4)) ^ 2 + (((x ^ 16) ^ (1 / 4)) ^ (1 / 8)) ^ 2 = 2 * x :=
by sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l3589_358993


namespace NUMINAMATH_CALUDE_unique_solution_mod_151_l3589_358959

theorem unique_solution_mod_151 :
  ∃! n : ℤ, 0 ≤ n ∧ n < 151 ∧ (150 * n) % 151 = 93 % 151 ∧ n = 58 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_mod_151_l3589_358959


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3589_358903

theorem absolute_value_inequality_solution_set :
  ∀ x : ℝ, |x^2 - 4| ≤ x + 2 ↔ (1 ≤ x ∧ x ≤ 3) ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3589_358903


namespace NUMINAMATH_CALUDE_slope_of_line_l3589_358908

theorem slope_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : 
  (y - 4) / x = -4 / 7 := by
sorry

end NUMINAMATH_CALUDE_slope_of_line_l3589_358908


namespace NUMINAMATH_CALUDE_divisor_of_a_l3589_358949

theorem divisor_of_a (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 24)
  (h2 : Nat.gcd b c = 36)
  (h3 : Nat.gcd c d = 54)
  (h4 : 70 < Nat.gcd d a ∧ Nat.gcd d a < 100) :
  13 ∣ a := by
  sorry

end NUMINAMATH_CALUDE_divisor_of_a_l3589_358949


namespace NUMINAMATH_CALUDE_ancient_chinese_algorithm_is_successive_subtraction_l3589_358986

/-- An ancient Chinese mathematical algorithm developed during the Song and Yuan dynasties -/
structure AncientChineseAlgorithm where
  name : String
  period : String
  comparable_to_euclidean : Bool

/-- The method of successive subtraction -/
def successive_subtraction : AncientChineseAlgorithm :=
  { name := "Method of Successive Subtraction",
    period := "Song and Yuan dynasties",
    comparable_to_euclidean := true }

/-- Theorem stating that the ancient Chinese algorithm comparable to the Euclidean algorithm
    of division is the method of successive subtraction -/
theorem ancient_chinese_algorithm_is_successive_subtraction :
  ∃ (a : AncientChineseAlgorithm), 
    a.period = "Song and Yuan dynasties" ∧ 
    a.comparable_to_euclidean = true ∧ 
    a = successive_subtraction :=
by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_algorithm_is_successive_subtraction_l3589_358986


namespace NUMINAMATH_CALUDE_hypotenuse_length_triangle_area_l3589_358931

-- Define a right triangle with legs 30 and 40
def right_triangle (a b c : ℝ) : Prop :=
  a = 30 ∧ b = 40 ∧ c^2 = a^2 + b^2

-- Theorem for the hypotenuse
theorem hypotenuse_length (a b c : ℝ) (h : right_triangle a b c) : c = 50 := by
  sorry

-- Theorem for the area
theorem triangle_area (a b : ℝ) (h : a = 30 ∧ b = 40) : (1/2) * a * b = 600 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_triangle_area_l3589_358931


namespace NUMINAMATH_CALUDE_initial_deposit_calculation_l3589_358990

/-- Proves that the initial deposit is $1000 given the conditions of the problem -/
theorem initial_deposit_calculation (P : ℝ) : 
  (P + 100 = 1100) →                    -- First year balance
  ((P + 100) * 1.2 = P * 1.32) →         -- Second year growth equals total growth
  P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_initial_deposit_calculation_l3589_358990


namespace NUMINAMATH_CALUDE_harmonic_point_3_m_harmonic_point_hyperbola_l3589_358938

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

end NUMINAMATH_CALUDE_harmonic_point_3_m_harmonic_point_hyperbola_l3589_358938


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l3589_358944

theorem largest_multiple_of_8_under_100 : 
  ∀ n : ℕ, n % 8 = 0 ∧ n < 100 → n ≤ 96 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l3589_358944


namespace NUMINAMATH_CALUDE_penelope_savings_l3589_358902

/-- The amount of money Penelope saves daily, in dollars. -/
def daily_savings : ℕ := 24

/-- The number of days in a year (assuming it's not a leap year). -/
def days_in_year : ℕ := 365

/-- The total amount Penelope saves in a year. -/
def total_savings : ℕ := daily_savings * days_in_year

/-- Theorem: Penelope's total savings after one year is $8,760. -/
theorem penelope_savings : total_savings = 8760 := by
  sorry

end NUMINAMATH_CALUDE_penelope_savings_l3589_358902


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3589_358955

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3589_358955


namespace NUMINAMATH_CALUDE_max_projection_length_l3589_358913

noncomputable section

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 0)
def curve (x : ℝ) : ℝ × ℝ := (x, x^2 + 1)

def projection_length (P : ℝ × ℝ) : ℝ :=
  let OA := A - O
  let OP := P - O
  abs (OA.1 * OP.1 + OA.2 * OP.2) / Real.sqrt (OP.1^2 + OP.2^2)

theorem max_projection_length :
  ∃ (max_length : ℝ), max_length = Real.sqrt 5 / 5 ∧
    ∀ (x : ℝ), projection_length (curve x) ≤ max_length :=
sorry

end

end NUMINAMATH_CALUDE_max_projection_length_l3589_358913


namespace NUMINAMATH_CALUDE_perpendicular_bisector_c_l3589_358942

/-- The perpendicular bisector of a line segment connecting two points (x₁, y₁) and (x₂, y₂) 
    is defined by the equation x + 2y = c. -/
def is_perpendicular_bisector (x₁ y₁ x₂ y₂ c : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + 2 * midpoint_y = c

/-- The value of c for the perpendicular bisector of the line segment 
    connecting (2,4) and (8,16) is 25. -/
theorem perpendicular_bisector_c : 
  is_perpendicular_bisector 2 4 8 16 25 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_c_l3589_358942


namespace NUMINAMATH_CALUDE_no_formula_fits_all_data_l3589_358972

def data : List (ℕ × ℕ) := [(1, 2), (2, 6), (3, 12), (4, 20), (5, 30)]

def formula_a (x : ℕ) : ℕ := 4 * x - 2
def formula_b (x : ℕ) : ℕ := x^3 - x^2 + 2*x
def formula_c (x : ℕ) : ℕ := 2 * x^2
def formula_d (x : ℕ) : ℕ := x^2 + 2*x + 1

theorem no_formula_fits_all_data :
  ¬(∀ (x y : ℕ), (x, y) ∈ data → 
    (y = formula_a x ∨ y = formula_b x ∨ y = formula_c x ∨ y = formula_d x)) :=
by sorry

end NUMINAMATH_CALUDE_no_formula_fits_all_data_l3589_358972


namespace NUMINAMATH_CALUDE_all_polyhedra_l3589_358957

-- Define the properties of a polyhedron
structure Polyhedron :=
  (has_flat_faces : Bool)
  (has_straight_edges : Bool)
  (has_sharp_corners : Bool)

-- Define the geometric solids
inductive GeometricSolid
  | TriangularPrism
  | SquareFrustum
  | Cube
  | HexagonalPyramid

-- Function to check if a geometric solid is a polyhedron
def is_polyhedron (solid : GeometricSolid) : Polyhedron :=
  match solid with
  | GeometricSolid.TriangularPrism => ⟨true, true, true⟩
  | GeometricSolid.SquareFrustum => ⟨true, true, true⟩
  | GeometricSolid.Cube => ⟨true, true, true⟩
  | GeometricSolid.HexagonalPyramid => ⟨true, true, true⟩

-- Theorem stating that all the given solids are polyhedra
theorem all_polyhedra :
  (is_polyhedron GeometricSolid.TriangularPrism).has_flat_faces ∧
  (is_polyhedron GeometricSolid.TriangularPrism).has_straight_edges ∧
  (is_polyhedron GeometricSolid.TriangularPrism).has_sharp_corners ∧
  (is_polyhedron GeometricSolid.SquareFrustum).has_flat_faces ∧
  (is_polyhedron GeometricSolid.SquareFrustum).has_straight_edges ∧
  (is_polyhedron GeometricSolid.SquareFrustum).has_sharp_corners ∧
  (is_polyhedron GeometricSolid.Cube).has_flat_faces ∧
  (is_polyhedron GeometricSolid.Cube).has_straight_edges ∧
  (is_polyhedron GeometricSolid.Cube).has_sharp_corners ∧
  (is_polyhedron GeometricSolid.HexagonalPyramid).has_flat_faces ∧
  (is_polyhedron GeometricSolid.HexagonalPyramid).has_straight_edges ∧
  (is_polyhedron GeometricSolid.HexagonalPyramid).has_sharp_corners :=
by sorry

end NUMINAMATH_CALUDE_all_polyhedra_l3589_358957


namespace NUMINAMATH_CALUDE_can_reach_ten_white_marbles_l3589_358975

-- Define the state of the urn
structure UrnState :=
  (white : ℕ)
  (black : ℕ)

-- Define the possible operations
inductive Operation
  | op1 -- 4B -> 2B
  | op2 -- 3B + W -> B
  | op3 -- 2B + 2W -> W + B
  | op4 -- B + 3W -> 2W
  | op5 -- 4W -> B

-- Define a function to apply an operation to the urn state
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.op1 => ⟨state.white, state.black - 2⟩
  | Operation.op2 => ⟨state.white - 1, state.black - 2⟩
  | Operation.op3 => ⟨state.white - 1, state.black - 1⟩
  | Operation.op4 => ⟨state.white - 1, state.black - 1⟩
  | Operation.op5 => ⟨state.white - 4, state.black + 1⟩

-- Define the initial state
def initialState : UrnState := ⟨50, 150⟩

-- Theorem: It is possible to reach exactly 10 white marbles
theorem can_reach_ten_white_marbles :
  ∃ (operations : List Operation),
    (operations.foldl applyOperation initialState).white = 10 :=
sorry

end NUMINAMATH_CALUDE_can_reach_ten_white_marbles_l3589_358975


namespace NUMINAMATH_CALUDE_football_playtime_l3589_358967

/-- Given a total playtime of 1.5 hours and a basketball playtime of 30 minutes,
    prove that the football playtime is 60 minutes. -/
theorem football_playtime
  (total_time : ℝ)
  (basketball_time : ℕ)
  (h1 : total_time = 1.5)
  (h2 : basketball_time = 30)
  : ↑basketball_time + 60 = total_time * 60 := by
  sorry

end NUMINAMATH_CALUDE_football_playtime_l3589_358967


namespace NUMINAMATH_CALUDE_jason_egg_consumption_l3589_358945

/-- The number of eggs Jason consumes in two weeks -/
def eggs_consumed_in_two_weeks : ℕ :=
  let eggs_per_omelet : ℕ := 3
  let days_in_two_weeks : ℕ := 14
  eggs_per_omelet * days_in_two_weeks

/-- Theorem stating that Jason consumes 42 eggs in two weeks -/
theorem jason_egg_consumption :
  eggs_consumed_in_two_weeks = 42 := by
  sorry

end NUMINAMATH_CALUDE_jason_egg_consumption_l3589_358945


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_2x_l3589_358991

theorem factorization_x_squared_minus_2x (x : ℝ) : x^2 - 2*x = x*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_2x_l3589_358991


namespace NUMINAMATH_CALUDE_pyramid_volume_l3589_358997

theorem pyramid_volume (base_length : ℝ) (base_width : ℝ) (edge_length : ℝ) :
  base_length = 5 →
  base_width = 10 →
  edge_length = 15 →
  let base_area := base_length * base_width
  let diagonal := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (diagonal / 2)^2)
  let volume := (1 / 3) * base_area * height
  volume = 232 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3589_358997


namespace NUMINAMATH_CALUDE_number_of_jeans_to_wash_l3589_358947

/-- The number of shirts Alex has to wash -/
def shirts : ℕ := 18

/-- The number of pants Alex has to wash -/
def pants : ℕ := 12

/-- The number of sweaters Alex has to wash -/
def sweaters : ℕ := 17

/-- The maximum number of items the washing machine can wash per cycle -/
def items_per_cycle : ℕ := 15

/-- The time in minutes each washing cycle takes -/
def minutes_per_cycle : ℕ := 45

/-- The total time in hours it takes to wash all clothes -/
def total_wash_time : ℕ := 3

/-- Theorem stating the number of jeans Alex has to wash -/
theorem number_of_jeans_to_wash : 
  ∃ (jeans : ℕ), 
    (shirts + pants + sweaters + jeans) = 
      (total_wash_time * 60 / minutes_per_cycle) * items_per_cycle ∧
    jeans = 13 := by
  sorry

end NUMINAMATH_CALUDE_number_of_jeans_to_wash_l3589_358947


namespace NUMINAMATH_CALUDE_division_with_remainder_l3589_358934

theorem division_with_remainder (A : ℕ) : 
  (A / 7 = 5) ∧ (A % 7 = 3) → A = 38 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_l3589_358934


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3589_358948

theorem trigonometric_identity (α : ℝ) : 
  Real.cos (4 * α) + Real.cos (3 * α) = 2 * Real.cos ((7 * α) / 2) * Real.cos (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3589_358948


namespace NUMINAMATH_CALUDE_apple_orchard_composition_l3589_358996

/-- Represents the composition of an apple orchard -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  pure_gala : ℕ
  cross_pollinated : ℕ

/-- The number of pure Gala trees in an orchard with given conditions -/
def pure_gala_count (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧
  o.pure_fuji = 3 * o.total / 4 ∧
  o.pure_fuji + o.cross_pollinated = 204 ∧
  o.pure_gala = 36

theorem apple_orchard_composition :
  ∃ (o : Orchard), pure_gala_count o :=
sorry

end NUMINAMATH_CALUDE_apple_orchard_composition_l3589_358996


namespace NUMINAMATH_CALUDE_smallest_multiple_l3589_358906

theorem smallest_multiple (x : ℕ) : x = 16 ↔ (
  x > 0 ∧
  450 * x % 800 = 0 ∧
  ∀ y : ℕ, y > 0 → y < x → 450 * y % 800 ≠ 0
) := by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3589_358906


namespace NUMINAMATH_CALUDE_min_value_expression_l3589_358961

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 1) (hab : a + b = 1) :
  (((2 * a + b) / (a * b) - 3) * c + (Real.sqrt 2) / (c - 1)) ≥ 4 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3589_358961


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l3589_358921

theorem quadratic_solution_property (x₁ x₂ : ℝ) : 
  x₁^2 - 5*x₁ + 2 = 0 → x₂^2 - 5*x₂ + 2 = 0 → 2*x₁ - x₁*x₂ + 2*x₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l3589_358921


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_in_fraction_l3589_358964

theorem zeros_before_first_nonzero_digit_in_fraction :
  ∃ (n : ℕ) (d : ℚ), 
    (d = 7 / 800) ∧ 
    (∃ (m : ℕ), d * (10 ^ n) = m ∧ m ≥ 100 ∧ m < 1000) ∧
    n = 3 :=
by sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_in_fraction_l3589_358964


namespace NUMINAMATH_CALUDE_number_problem_l3589_358926

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N - (1/2 : ℝ) * (1/6 : ℝ) * N = 35 →
  (40/100 : ℝ) * N = -280 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l3589_358926


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3589_358987

theorem hyperbola_equation (a b : ℝ) (h1 : b > 0) (h2 : a > 0) (h3 : ∃ n : ℕ, a = n) 
  (h4 : (a^2 + b^2) / a^2 = 7/4) (h5 : a^2 + b^2 ≤ 20) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  ((x^2 - 4*y^2/3 = 1) ∨ (x^2/4 - y^2/3 = 1) ∨ (x^2/9 - 4*y^2/27 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3589_358987


namespace NUMINAMATH_CALUDE_fred_total_games_l3589_358912

/-- The number of basketball games Fred attended this year -/
def games_this_year : ℕ := 36

/-- The number of basketball games Fred attended last year -/
def games_last_year : ℕ := 11

/-- The total number of basketball games Fred attended -/
def total_games : ℕ := games_this_year + games_last_year

theorem fred_total_games : total_games = 47 := by
  sorry

end NUMINAMATH_CALUDE_fred_total_games_l3589_358912


namespace NUMINAMATH_CALUDE_six_digit_concatenation_divisibility_l3589_358980

theorem six_digit_concatenation_divisibility :
  ∀ a b : ℕ,
    100000 ≤ a ∧ a < 1000000 →
    100000 ≤ b ∧ b < 1000000 →
    (∃ k : ℕ, 1000000 * a + b = k * a * b) →
    (a = 166667 ∧ b = 333334) := by
  sorry

end NUMINAMATH_CALUDE_six_digit_concatenation_divisibility_l3589_358980


namespace NUMINAMATH_CALUDE_value_of_a_l3589_358977

theorem value_of_a (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) 
  (h : ∀ x : ℝ, x^2 + 2*x^10 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                 a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10) : 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3589_358977


namespace NUMINAMATH_CALUDE_local_extremum_cubic_l3589_358910

/-- Given a cubic function f(x) = ax³ + 3x² - 6ax + b with a local extremum of 9 at x = 2,
    prove that a + 2b = -24 -/
theorem local_extremum_cubic (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + 3 * x^2 - 6 * a * x + b
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≤ f 2) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≥ f 2) ∧
  f 2 = 9 →
  a + 2 * b = -24 := by
sorry

end NUMINAMATH_CALUDE_local_extremum_cubic_l3589_358910


namespace NUMINAMATH_CALUDE_x4_plus_y4_equals_135_point_5_l3589_358929

theorem x4_plus_y4_equals_135_point_5 (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 14) : 
  x^4 + y^4 = 135.5 := by
sorry

end NUMINAMATH_CALUDE_x4_plus_y4_equals_135_point_5_l3589_358929


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l3589_358984

theorem trig_expression_simplification :
  (Real.sin (15 * π / 180) + Real.sin (25 * π / 180) + Real.sin (35 * π / 180) + Real.sin (45 * π / 180) +
   Real.sin (55 * π / 180) + Real.sin (65 * π / 180) + Real.sin (75 * π / 180) + Real.sin (85 * π / 180)) /
  (Real.cos (10 * π / 180) * Real.cos (15 * π / 180) * Real.cos (25 * π / 180)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l3589_358984


namespace NUMINAMATH_CALUDE_meghan_coffee_order_cost_l3589_358965

/-- Represents the cost of a coffee order with given quantities and prices --/
def coffee_order_cost (drip_coffee_price : ℚ) (drip_coffee_qty : ℕ)
                      (espresso_price : ℚ) (espresso_qty : ℕ)
                      (latte_price : ℚ) (latte_qty : ℕ)
                      (vanilla_syrup_price : ℚ) (vanilla_syrup_qty : ℕ)
                      (cold_brew_price : ℚ) (cold_brew_qty : ℕ)
                      (cappuccino_price : ℚ) (cappuccino_qty : ℕ) : ℚ :=
  drip_coffee_price * drip_coffee_qty +
  espresso_price * espresso_qty +
  latte_price * latte_qty +
  vanilla_syrup_price * vanilla_syrup_qty +
  cold_brew_price * cold_brew_qty +
  cappuccino_price * cappuccino_qty

/-- The total cost of Meghan's coffee order is $25.00 --/
theorem meghan_coffee_order_cost :
  coffee_order_cost (25/10) 2
                    (35/10) 1
                    4 2
                    (1/2) 1
                    (25/10) 2
                    (35/10) 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_meghan_coffee_order_cost_l3589_358965


namespace NUMINAMATH_CALUDE_y_coordinate_is_1000_l3589_358933

/-- A straight line in the xy-plane with given properties -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- A point on a line -/
structure Point where
  x : ℝ
  y : ℝ

/-- The y-coordinate of a point on a line can be calculated using the line's equation -/
def y_coordinate (l : Line) (p : Point) : ℝ :=
  l.slope * p.x + l.y_intercept

/-- Theorem: The y-coordinate of the specified point on the given line is 1000 -/
theorem y_coordinate_is_1000 (l : Line) (p : Point)
  (h1 : l.slope = 9.9)
  (h2 : l.y_intercept = 10)
  (h3 : p.x = 100) :
  y_coordinate l p = 1000 := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_is_1000_l3589_358933


namespace NUMINAMATH_CALUDE_polar_to_cartesian_parabola_l3589_358963

/-- The curve defined by the polar equation r = 1 / (1 - sin θ) is a parabola -/
theorem polar_to_cartesian_parabola :
  ∃ (x y : ℝ), (∃ (r θ : ℝ), r = 1 / (1 - Real.sin θ) ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  x^2 = 2*y + 1 :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_parabola_l3589_358963


namespace NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l3589_358907

theorem greatest_whole_number_satisfying_inequality :
  ∀ (n : ℤ), n ≤ 0 ↔ (3 : ℝ) * n + 2 < 5 - 2 * n :=
by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l3589_358907


namespace NUMINAMATH_CALUDE_probability_four_students_same_vehicle_l3589_358974

/-- The probability that 4 students all ride in the same vehicle when there are 3 vehicles to choose from. -/
theorem probability_four_students_same_vehicle (num_vehicles : ℕ) (num_students : ℕ) : 
  num_vehicles = 3 → num_students = 4 → (num_vehicles : ℚ)/(num_vehicles^num_students : ℚ) = 1/27 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_students_same_vehicle_l3589_358974


namespace NUMINAMATH_CALUDE_cube_holes_surface_area_222_l3589_358943

/-- Calculates the surface area of a cube with square holes cut through each face. -/
def cube_with_holes_surface_area (cube_edge : ℝ) (hole_edge : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_edge^2
  let hole_area := 6 * hole_edge^2
  let new_exposed_area := 6 * 4 * hole_edge^2
  original_surface_area - hole_area + new_exposed_area

/-- Theorem stating that a cube with edge length 5 and square holes of side length 2
    has a total surface area of 222 square meters. -/
theorem cube_holes_surface_area_222 :
  cube_with_holes_surface_area 5 2 = 222 := by
  sorry

end NUMINAMATH_CALUDE_cube_holes_surface_area_222_l3589_358943


namespace NUMINAMATH_CALUDE_infinite_geometric_series_second_term_l3589_358914

theorem infinite_geometric_series_second_term 
  (r : ℝ) (S : ℝ) (h_r : r = 1/4) (h_S : S = 40) :
  let a := S * (1 - r)
  (a * r) = 15/2 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_second_term_l3589_358914


namespace NUMINAMATH_CALUDE_triangle_problem_l3589_358922

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  b * Real.cos A = (2 * c + a) * Real.cos (π - B) ∧
  b = Real.sqrt 21 ∧
  1 / 2 * a * c * Real.sin B = Real.sqrt 3 →
  B = 2 * π / 3 ∧ a + c = 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3589_358922


namespace NUMINAMATH_CALUDE_probability_both_selected_l3589_358982

theorem probability_both_selected (p_ram p_ravi : ℚ) 
  (h1 : p_ram = 6/7) (h2 : p_ravi = 1/5) : 
  p_ram * p_ravi = 6/35 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l3589_358982


namespace NUMINAMATH_CALUDE_solve_for_m_l3589_358916

theorem solve_for_m : ∃ m : ℤ, 5^2 + 7 = 4^3 + m ∧ m = -32 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l3589_358916


namespace NUMINAMATH_CALUDE_existence_of_four_integers_l3589_358901

theorem existence_of_four_integers (a : Fin 97 → ℕ+) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) : 
  ∃ w x y z : Fin 97, w ≠ x ∧ y ≠ z ∧ 1984 ∣ ((a w).val - (a x).val) * ((a y).val - (a z).val) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_four_integers_l3589_358901


namespace NUMINAMATH_CALUDE_student_927_selected_l3589_358989

/-- Represents a student number in the range [1, 1000] -/
def StudentNumber := Fin 1000

/-- The total number of students -/
def totalStudents : Nat := 1000

/-- The number of students to be sampled -/
def sampleSize : Nat := 200

/-- The sampling interval -/
def samplingInterval : Nat := totalStudents / sampleSize

/-- Predicate to check if a student number is selected in the systematic sampling -/
def isSelected (n : StudentNumber) : Prop :=
  n.val % samplingInterval = 122 % samplingInterval

/-- Theorem stating that if student 122 is selected, then student 927 is also selected -/
theorem student_927_selected :
  isSelected ⟨121, by norm_num⟩ → isSelected ⟨926, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_student_927_selected_l3589_358989


namespace NUMINAMATH_CALUDE_janet_cat_collars_l3589_358960

/-- The number of inches of nylon needed for a dog collar -/
def dog_collar_nylon : ℕ := 18

/-- The number of inches of nylon needed for a cat collar -/
def cat_collar_nylon : ℕ := 10

/-- The total number of inches of nylon Janet needs -/
def total_nylon : ℕ := 192

/-- The number of dog collars Janet needs to make -/
def num_dog_collars : ℕ := 9

/-- Theorem stating that Janet needs to make 3 cat collars -/
theorem janet_cat_collars : 
  (total_nylon - num_dog_collars * dog_collar_nylon) / cat_collar_nylon = 3 := by
  sorry

end NUMINAMATH_CALUDE_janet_cat_collars_l3589_358960


namespace NUMINAMATH_CALUDE_joan_games_this_year_l3589_358985

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := sorry

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := 9

/-- The total number of football games Joan went to -/
def total_games : ℕ := 13

/-- Theorem stating that the number of games Joan went to this year is 4 -/
theorem joan_games_this_year : games_this_year = 4 := by sorry

end NUMINAMATH_CALUDE_joan_games_this_year_l3589_358985


namespace NUMINAMATH_CALUDE_min_value_expression_equality_achieved_l3589_358988

theorem min_value_expression (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 1)^2) ≥ Real.sqrt 13 := by
  sorry

theorem equality_achieved : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 1)^2) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_achieved_l3589_358988


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3589_358956

theorem polynomial_remainder (x : ℤ) : (x^15 - 2) % (x + 2) = -32770 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3589_358956


namespace NUMINAMATH_CALUDE_julia_tag_game_l3589_358981

theorem julia_tag_game (tuesday_kids : ℕ) (monday_difference : ℕ) : 
  tuesday_kids = 5 → monday_difference = 1 → tuesday_kids + monday_difference = 6 :=
by sorry

end NUMINAMATH_CALUDE_julia_tag_game_l3589_358981
