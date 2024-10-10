import Mathlib

namespace unique_bezout_bounded_l350_35032

theorem unique_bezout_bounded (a b : ℕ) (ha : a > 1) (hb : b > 1) (hgcd : Nat.gcd a b = 1) :
  ∃! (r s : ℕ), a * r - b * s = 1 ∧ 0 < r ∧ r < b ∧ 0 < s ∧ s < a := by
  sorry

end unique_bezout_bounded_l350_35032


namespace slope_of_OP_l350_35088

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 2

-- Define the line
def is_on_line (x y k : ℝ) : Prop := x + y = k

-- Define the intersection points
def are_intersection_points (M N : ℝ × ℝ) (k : ℝ) : Prop :=
  is_on_ellipse M.1 M.2 ∧ is_on_ellipse N.1 N.2 ∧
  is_on_line M.1 M.2 k ∧ is_on_line N.1 N.2 k

-- Define the midpoint
def is_midpoint (P M N : ℝ × ℝ) : Prop :=
  P.1 = (M.1 + N.1) / 2 ∧ P.2 = (M.2 + N.2) / 2

-- Theorem statement
theorem slope_of_OP (k : ℝ) (M N P : ℝ × ℝ) :
  are_intersection_points M N k →
  is_midpoint P M N →
  P.2 / P.1 = 1 / 2 :=
sorry

end slope_of_OP_l350_35088


namespace double_shot_espresso_price_l350_35072

/-- Represents the cost of a coffee order -/
structure CoffeeOrder where
  drip_coffee : ℕ
  drip_coffee_price : ℚ
  latte : ℕ
  latte_price : ℚ
  vanilla_syrup : ℕ
  vanilla_syrup_price : ℚ
  cold_brew : ℕ
  cold_brew_price : ℚ
  cappuccino : ℕ
  cappuccino_price : ℚ
  double_shot_espresso : ℕ
  total_price : ℚ

/-- Calculates the cost of the double shot espresso -/
def double_shot_espresso_cost (order : CoffeeOrder) : ℚ :=
  order.total_price -
  (order.drip_coffee * order.drip_coffee_price +
   order.latte * order.latte_price +
   order.vanilla_syrup * order.vanilla_syrup_price +
   order.cold_brew * order.cold_brew_price +
   order.cappuccino * order.cappuccino_price)

/-- Theorem stating that the double shot espresso costs $3.50 -/
theorem double_shot_espresso_price (order : CoffeeOrder) 
  (h1 : order.drip_coffee = 2)
  (h2 : order.drip_coffee_price = 2.25)
  (h3 : order.latte = 2)
  (h4 : order.latte_price = 4)
  (h5 : order.vanilla_syrup = 1)
  (h6 : order.vanilla_syrup_price = 0.5)
  (h7 : order.cold_brew = 2)
  (h8 : order.cold_brew_price = 2.5)
  (h9 : order.cappuccino = 1)
  (h10 : order.cappuccino_price = 3.5)
  (h11 : order.double_shot_espresso = 1)
  (h12 : order.total_price = 25) :
  double_shot_espresso_cost order = 3.5 := by
  sorry


end double_shot_espresso_price_l350_35072


namespace garden_potato_yield_l350_35025

/-- Calculates the expected potato yield from a rectangular garden --/
theorem garden_potato_yield 
  (length_steps width_steps : ℕ) 
  (step_length : ℝ) 
  (planting_ratio : ℝ) 
  (yield_rate : ℝ) :
  length_steps = 10 →
  width_steps = 30 →
  step_length = 3 →
  planting_ratio = 0.9 →
  yield_rate = 3/4 →
  (length_steps : ℝ) * step_length * (width_steps : ℝ) * step_length * planting_ratio * yield_rate = 1822.5 := by
  sorry

end garden_potato_yield_l350_35025


namespace power_function_increasing_l350_35054

/-- A power function f(x) = (m^2 - m - 1)x^m is increasing on (0, +∞) if and only if m = 2 -/
theorem power_function_increasing (m : ℝ) : 
  (∀ x > 0, Monotone (fun x => (m^2 - m - 1) * x^m)) ↔ m = 2 := by
  sorry

end power_function_increasing_l350_35054


namespace max_gcd_13n_plus_4_8n_plus_3_l350_35000

theorem max_gcd_13n_plus_4_8n_plus_3 (n : ℕ+) : 
  (Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 7) ∧ 
  (∃ m : ℕ+, Nat.gcd (13 * m + 4) (8 * m + 3) = 7) := by
sorry

end max_gcd_13n_plus_4_8n_plus_3_l350_35000


namespace product_digit_range_l350_35087

theorem product_digit_range : 
  ∀ (a b : ℕ), 
    1 ≤ a ∧ a ≤ 9 → 
    100 ≤ b ∧ b ≤ 999 → 
    (100 ≤ a * b ∧ a * b ≤ 9999) := by
  sorry

end product_digit_range_l350_35087


namespace sequence_bound_l350_35006

def sequence_rule (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, ∃ d : ℕ, d < 10 ∧ 
    (a (2*n) = a (2*n - 1) - d) ∧
    (a (2*n + 1) = a (2*n) + d))

theorem sequence_bound (a : ℕ → ℕ) (h : sequence_rule a) : 
  ∀ n : ℕ, a n ≤ 10 * a 1 :=
sorry

end sequence_bound_l350_35006


namespace age_difference_l350_35058

theorem age_difference (sachin_age rahul_age : ℝ) : 
  sachin_age = 24.5 → 
  sachin_age / rahul_age = 7 / 9 → 
  rahul_age - sachin_age = 7 := by
sorry

end age_difference_l350_35058


namespace jerry_cut_eight_pine_trees_l350_35001

/-- The number of logs produced by one pine tree -/
def logs_per_pine : ℕ := 80

/-- The number of logs produced by one maple tree -/
def logs_per_maple : ℕ := 60

/-- The number of logs produced by one walnut tree -/
def logs_per_walnut : ℕ := 100

/-- The number of maple trees Jerry cut -/
def maple_trees : ℕ := 3

/-- The number of walnut trees Jerry cut -/
def walnut_trees : ℕ := 4

/-- The total number of logs Jerry got -/
def total_logs : ℕ := 1220

/-- Theorem stating that Jerry cut 8 pine trees -/
theorem jerry_cut_eight_pine_trees :
  ∃ (pine_trees : ℕ), pine_trees * logs_per_pine + 
                      maple_trees * logs_per_maple + 
                      walnut_trees * logs_per_walnut = total_logs ∧ 
                      pine_trees = 8 := by
  sorry

end jerry_cut_eight_pine_trees_l350_35001


namespace berry_picking_pattern_l350_35059

/-- A sequence of 5 numbers where the differences between consecutive terms
    form an arithmetic sequence with a common difference of 2 -/
def BerrySequence (a b c d e : ℕ) : Prop :=
  (c - b) - (b - a) = 2 ∧
  (d - c) - (c - b) = 2 ∧
  (e - d) - (d - c) = 2

theorem berry_picking_pattern (a b c d e : ℕ) :
  BerrySequence a b c d e →
  a = 3 →
  c = 7 →
  d = 12 →
  e = 19 →
  b = 6 := by
sorry

end berry_picking_pattern_l350_35059


namespace xy_value_l350_35010

theorem xy_value (x y : ℝ) (h : x^2 + y^2 - 22*x - 20*y + 221 = 0) : x * y = 110 := by
  sorry

end xy_value_l350_35010


namespace irrational_approximation_l350_35004

theorem irrational_approximation (ξ : ℝ) (h_irrational : Irrational ξ) :
  Set.Infinite {q : ℚ | ∃ (m : ℤ) (n : ℕ), q = m / n ∧ |ξ - (m / n)| < 1 / (Real.sqrt 5 * m^2)} := by
  sorry

end irrational_approximation_l350_35004


namespace sequence_with_constant_triple_sum_l350_35003

theorem sequence_with_constant_triple_sum :
  ∃! (a : Fin 8 → ℝ), 
    a 0 = 5 ∧ 
    a 7 = 8 ∧ 
    (∀ i : Fin 6, a i + a (i + 1) + a (i + 2) = 20) := by
  sorry

end sequence_with_constant_triple_sum_l350_35003


namespace function_equation_implies_identity_l350_35097

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f x + y) * (f (x - y) + 1) = f (f (x * f (x + 1)) - y * f (y - 1))) :
  ∀ x : ℝ, f x = x := by
  sorry

end function_equation_implies_identity_l350_35097


namespace onion_saute_time_l350_35013

def calzone_problem (onion_time : ℝ) : Prop :=
  let garlic_pepper_time := (1/4) * onion_time
  let knead_time := 30
  let rest_time := 2 * knead_time
  let assemble_time := (1/10) * (knead_time + rest_time)
  onion_time + garlic_pepper_time + knead_time + rest_time + assemble_time = 124

theorem onion_saute_time :
  ∃ (t : ℝ), calzone_problem t ∧ t = 20 := by
  sorry

end onion_saute_time_l350_35013


namespace f_composition_of_one_l350_35048

def f (x : ℤ) : ℤ :=
  if x % 2 = 0 then 3 * x / 2 else 2 * x + 1

theorem f_composition_of_one : f (f (f (f 1))) = 31 := by
  sorry

end f_composition_of_one_l350_35048


namespace time_A_is_120_l350_35051

/-- The time it takes for B to fill the tank alone (in minutes) -/
def time_B : ℝ := 40

/-- The total time to fill the tank when B is used for half the time and A and B fill it together for the other half (in minutes) -/
def total_time : ℝ := 29.999999999999993

/-- The time it takes for A to fill the tank alone (in minutes) -/
def time_A : ℝ := 120

/-- Theorem stating that the time for A to fill the tank alone is 120 minutes -/
theorem time_A_is_120 : time_A = 120 := by sorry

end time_A_is_120_l350_35051


namespace equidistant_point_x_coordinate_l350_35011

/-- The x-coordinate of a point P on the x-axis that is equidistant from A(-3, 0) and B(0, 5) is 8/3 -/
theorem equidistant_point_x_coordinate : 
  ∃ x : ℝ, 
    (x^2 + 6*x + 9 = x^2 + 25) ∧ 
    (∀ y : ℝ, ((-3 - x)^2 + y^2 = x^2 + (5 - y)^2) → y = 0) ∧
    x = 8/3 := by
  sorry

end equidistant_point_x_coordinate_l350_35011


namespace largest_prime_factor_of_4652_l350_35080

theorem largest_prime_factor_of_4652 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4652 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 4652 → q ≤ p :=
by sorry

end largest_prime_factor_of_4652_l350_35080


namespace sphere_volume_ratio_l350_35092

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 16 →
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 1 / 64 := by
  sorry

end sphere_volume_ratio_l350_35092


namespace oranges_per_group_l350_35061

theorem oranges_per_group (total_oranges : ℕ) (num_groups : ℕ) 
  (h1 : total_oranges = 384) (h2 : num_groups = 16) :
  total_oranges / num_groups = 24 := by
  sorry

end oranges_per_group_l350_35061


namespace expression_simplification_l350_35084

theorem expression_simplification (x y z : ℝ) : 
  (x - (2*y + z)) - ((x + 2*y) - 3*z) = -4*y + 2*z := by
  sorry

end expression_simplification_l350_35084


namespace total_blocks_l350_35016

theorem total_blocks (red : ℕ) (yellow : ℕ) (green : ℕ) (blue : ℕ) (orange : ℕ) (purple : ℕ)
  (h1 : red = 24)
  (h2 : yellow = red + 8)
  (h3 : green = yellow - 10)
  (h4 : blue = 2 * green)
  (h5 : orange = blue + 15)
  (h6 : purple = red + orange - 7) :
  red + yellow + green + blue + orange + purple = 257 := by
  sorry

end total_blocks_l350_35016


namespace escalator_travel_time_l350_35005

/-- Calculates the time taken for a person to cover the length of a moving escalator -/
theorem escalator_travel_time 
  (escalator_speed : ℝ) 
  (escalator_length : ℝ) 
  (person_speed : ℝ) : 
  escalator_speed = 12 →
  escalator_length = 210 →
  person_speed = 2 →
  escalator_length / (escalator_speed + person_speed) = 15 := by
sorry


end escalator_travel_time_l350_35005


namespace power_of_negative_product_l350_35093

theorem power_of_negative_product (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by sorry

end power_of_negative_product_l350_35093


namespace segment_length_proof_l350_35073

theorem segment_length_proof (C D R S : ℝ) : 
  C < R ∧ R < S ∧ S < D →  -- R and S are on the same side of midpoint
  (R - C) / (D - R) = 3 / 5 →  -- R divides CD in ratio 3:5
  (S - C) / (D - S) = 2 / 3 →  -- S divides CD in ratio 2:3
  S - R = 5 →  -- Length of RS is 5
  D - C = 200 := by  -- Length of CD is 200
sorry


end segment_length_proof_l350_35073


namespace brush_square_ratio_l350_35053

/-- Given a square with side length s and a brush width w, 
    if the brush covers exactly one-third of the square's area 
    when swept along both diagonals, then the ratio s/w is equal to 2√3 - 2. -/
theorem brush_square_ratio (s w : ℝ) (h : s > 0) (h' : w > 0) : 
  w^2 + ((s - w)^2) / 2 = (1/3) * s^2 → s / w = 2 * Real.sqrt 3 - 2 := by
  sorry

end brush_square_ratio_l350_35053


namespace bees_after_six_days_l350_35056

/-- The number of bees after n days in the hive process -/
def bees (n : ℕ) : ℕ := 6^n

/-- The process starts with 1 bee and continues for 6 days -/
def days : ℕ := 6

/-- The theorem stating the number of bees after 6 days -/
theorem bees_after_six_days : bees days = 46656 := by sorry

end bees_after_six_days_l350_35056


namespace squared_sum_equals_cube_root_l350_35044

theorem squared_sum_equals_cube_root (x y : ℝ) 
  (h1 : x^2 - 3*y^2 = 17/x) 
  (h2 : 3*x^2 - y^2 = 23/y) : 
  x^2 + y^2 = 818^(1/3) := by
  sorry

end squared_sum_equals_cube_root_l350_35044


namespace f_extrema_l350_35023

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 1) * (x + a)

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 1

theorem f_extrema (a : ℝ) (h : f_derivative a (-1) = 0) :
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (-3/2 : ℝ) 1, f a x ≤ max) ∧ 
    (∃ x ∈ Set.Icc (-3/2 : ℝ) 1, f a x = max) ∧
    (∀ x ∈ Set.Icc (-3/2 : ℝ) 1, min ≤ f a x) ∧ 
    (∃ x ∈ Set.Icc (-3/2 : ℝ) 1, f a x = min) ∧
    max = 6 ∧ min = 13/8 := by
  sorry

end f_extrema_l350_35023


namespace quadratic_roots_l350_35022

theorem quadratic_roots (a : ℝ) : 
  (3^2 - 2*3 + a = 0) → 
  ((-1)^2 - 2*(-1) + a = 0) := by
sorry

end quadratic_roots_l350_35022


namespace carlos_pesos_l350_35064

/-- The exchange rate from Mexican pesos to U.S. dollars -/
def exchange_rate : ℚ := 8 / 14

/-- The amount spent in U.S. dollars -/
def amount_spent : ℕ := 50

/-- The remaining amount is three times the spent amount -/
def remaining_ratio : ℕ := 3

/-- The number of Mexican pesos Carlos had -/
def p : ℕ := 350

theorem carlos_pesos :
  p * exchange_rate - amount_spent = remaining_ratio * amount_spent := by
  sorry

end carlos_pesos_l350_35064


namespace final_number_after_combinations_l350_35033

def combineNumbers (a b : ℕ) : ℕ := a * b + a + b

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem final_number_after_combinations : 
  ∀ (permutation : List ℕ), 
    permutation.length = 20 ∧ 
    (∀ n, n ∈ permutation ↔ 1 ≤ n ∧ n ≤ 20) →
    (permutation.foldl combineNumbers 0) = factorial 21 - 1 :=
by sorry

end final_number_after_combinations_l350_35033


namespace cosine_sine_shift_l350_35068

theorem cosine_sine_shift :
  let f (x : ℝ) := Real.cos (2 * x + π / 3)
  let g (x : ℝ) := Real.sin (2 * x)
  ∃ (shift : ℝ), shift = 5 * π / 6 ∧
    ∀ (x : ℝ), f x = g (x + shift) := by
  sorry

end cosine_sine_shift_l350_35068


namespace evening_emails_count_l350_35036

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 5

/-- The number of emails Jack received in the afternoon and evening combined -/
def afternoon_and_evening_emails : ℕ := 13

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := afternoon_and_evening_emails - afternoon_emails

theorem evening_emails_count : evening_emails = 8 := by
  sorry

end evening_emails_count_l350_35036


namespace hyperbola_ellipse_shared_foci_l350_35071

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the ellipse
def ellipse (a x y : ℝ) : Prop := x^2 / a^2 + y^2 / 16 = 1

-- Define the condition that a > 0
def a_positive (a : ℝ) : Prop := a > 0

-- Define the condition that the hyperbola and ellipse share the same foci
def same_foci (a : ℝ) : Prop := ∃ c : ℝ, c^2 = 9 ∧ 
  (∀ x y : ℝ, hyperbola x y ↔ x^2 / 4 - y^2 / 5 = 1) ∧
  (∀ x y : ℝ, ellipse a x y ↔ x^2 / a^2 + y^2 / 16 = 1)

-- Theorem statement
theorem hyperbola_ellipse_shared_foci (a : ℝ) :
  a_positive a → same_foci a → a = 5 := by sorry

end hyperbola_ellipse_shared_foci_l350_35071


namespace parametric_to_cartesian_l350_35046

-- Define the parametric equations
def x_param (t : ℝ) : ℝ := t + 1
def y_param (t : ℝ) : ℝ := 3 - t^2

-- State the theorem
theorem parametric_to_cartesian :
  ∀ (x y : ℝ), (∃ t : ℝ, x = x_param t ∧ y = y_param t) ↔ y = -x^2 + 2*x + 2 :=
by sorry

end parametric_to_cartesian_l350_35046


namespace max_sum_with_product_2665_l350_35074

theorem max_sum_with_product_2665 :
  ∀ A B C : ℕ+,
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2665 →
  A + B + C ≤ 539 :=
by sorry

end max_sum_with_product_2665_l350_35074


namespace isosceles_triangle_perimeter_l350_35024

theorem isosceles_triangle_perimeter (equilateral_perimeter : ℝ) (isosceles_base : ℝ) : 
  equilateral_perimeter = 45 → 
  isosceles_base = 10 → 
  ∃ (isosceles_side : ℝ), 
    isosceles_side = equilateral_perimeter / 3 ∧ 
    2 * isosceles_side + isosceles_base = 40 := by
  sorry

end isosceles_triangle_perimeter_l350_35024


namespace min_gumballs_for_five_colors_l350_35078

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- The minimum number of gumballs needed to guarantee five of the same color -/
def minGumballsForFive (machine : GumballMachine) : Nat :=
  16

/-- Theorem stating that for a machine with 10 red, 10 white, 10 blue, and 6 green gumballs,
    the minimum number of gumballs needed to guarantee five of the same color is 16 -/
theorem min_gumballs_for_five_colors (machine : GumballMachine)
    (h_red : machine.red = 10)
    (h_white : machine.white = 10)
    (h_blue : machine.blue = 10)
    (h_green : machine.green = 6) :
    minGumballsForFive machine = 16 := by
  sorry

end min_gumballs_for_five_colors_l350_35078


namespace remainder_theorem_l350_35045

theorem remainder_theorem (r : ℤ) : (r^15 - 1) % (r + 2) = -32769 := by
  sorry

end remainder_theorem_l350_35045


namespace inverse_functions_same_monotonicity_function_symmetry_origin_exists_odd_function_without_inverse_l350_35021

-- Define a type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define inverse functions
def IsInverse (f g : RealFunction) : Prop :=
  ∀ x, g (f x) = x ∧ f (g x) = x

-- Define monotonicity
def IsMonotoneIncreasing (f : RealFunction) : Prop :=
  ∀ x y, x < y → f x < f y

def IsMonotoneDecreasing (f : RealFunction) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define odd function
def IsOdd (f : RealFunction) : Prop :=
  ∀ x, f (-x) = -f x

-- Define symmetry with respect to the origin
def IsSymmetricToOrigin (f g : RealFunction) : Prop :=
  ∀ x, g x = -f (-x)

-- Theorem 1: Inverse functions have the same monotonicity
theorem inverse_functions_same_monotonicity (f g : RealFunction) 
  (h : IsInverse f g) : 
  (IsMonotoneIncreasing f ↔ IsMonotoneIncreasing g) ∧ 
  (IsMonotoneDecreasing f ↔ IsMonotoneDecreasing g) := by
  sorry

-- Theorem 2: Function symmetry with respect to the origin
theorem function_symmetry_origin (f : RealFunction) :
  IsSymmetricToOrigin f (λ x => -f (-x)) := by
  sorry

-- Theorem 3: Existence of an odd function without an inverse
theorem exists_odd_function_without_inverse :
  ∃ f : RealFunction, IsOdd f ∧ ¬(∃ g : RealFunction, IsInverse f g) := by
  sorry

end inverse_functions_same_monotonicity_function_symmetry_origin_exists_odd_function_without_inverse_l350_35021


namespace paige_team_total_points_l350_35096

def team_size : ℕ := 5
def paige_points : ℕ := 11
def other_player_points : ℕ := 6

theorem paige_team_total_points :
  (paige_points + (team_size - 1) * other_player_points) = 35 := by
  sorry

end paige_team_total_points_l350_35096


namespace fractional_equation_solution_l350_35012

theorem fractional_equation_solution (x a : ℝ) : 
  (2 * x + a) / (x + 1) = 1 → x < 0 → a > 1 ∧ a ≠ 2 := by
  sorry

end fractional_equation_solution_l350_35012


namespace coefficient_of_x_fourth_l350_35027

/-- The expression for which we need to find the coefficient of x^4 -/
def expression (x : ℝ) : ℝ := 5 * (x^4 - 2*x^3) + 3 * (2*x^2 - 3*x^4 + x^6) - (5*x^6 - 2*x^4)

/-- The coefficient of x^4 in the given expression is -2 -/
theorem coefficient_of_x_fourth : (deriv^[4] expression 0) / 24 = -2 := by
  sorry

end coefficient_of_x_fourth_l350_35027


namespace octagon_area_error_l350_35069

theorem octagon_area_error (L : ℝ) (h : L > 0) : 
  let measured_length := 1.1 * L
  let true_area := 2 * (1 + Real.sqrt 2) * L^2 / 4
  let estimated_area := 2 * (1 + Real.sqrt 2) * measured_length^2 / 4
  (estimated_area - true_area) / true_area * 100 = 21 := by sorry

end octagon_area_error_l350_35069


namespace peters_pizza_fraction_l350_35095

theorem peters_pizza_fraction (total_slices : ℕ) (peters_solo_slices : ℕ) 
  (shared_with_paul : ℕ) (shared_with_mary : ℕ) : 
  total_slices = 18 → 
  peters_solo_slices = 3 → 
  shared_with_paul = 2 → 
  shared_with_mary = 1 → 
  (peters_solo_slices : ℚ) / total_slices + 
  (shared_with_paul : ℚ) / (2 * total_slices) + 
  (shared_with_mary : ℚ) / (2 * total_slices) = 11 / 36 := by
sorry

end peters_pizza_fraction_l350_35095


namespace meeting_distance_l350_35029

theorem meeting_distance (initial_speed : ℝ) (speed_increase : ℝ) (initial_distance : ℝ) 
  (late_time : ℝ) (early_time : ℝ) :
  initial_speed = 45 ∧ 
  speed_increase = 20 ∧ 
  initial_distance = 45 ∧ 
  late_time = 0.75 ∧ 
  early_time = 0.25 → 
  ∃ (total_distance : ℝ),
    total_distance = initial_speed * (total_distance / initial_speed + late_time) ∧
    total_distance - initial_distance = (initial_speed + speed_increase) * 
      (total_distance / initial_speed - 1 - early_time) ∧
    total_distance = 191.25 := by
  sorry

end meeting_distance_l350_35029


namespace algorithm_steps_are_determinate_l350_35099

/-- Represents a step in an algorithm -/
structure AlgorithmStep where
  precise : Bool
  effective : Bool
  determinate : Bool

/-- Represents an algorithm -/
structure Algorithm where
  steps : List AlgorithmStep
  solvesProblem : Bool
  finite : Bool

/-- Theorem: Given an algorithm with finite, precise, and effective steps that solve a problem, 
    prove that all steps in the algorithm are determinate -/
theorem algorithm_steps_are_determinate (a : Algorithm) 
  (h1 : a.solvesProblem)
  (h2 : a.finite)
  (h3 : ∀ s ∈ a.steps, s.precise)
  (h4 : ∀ s ∈ a.steps, s.effective) :
  ∀ s ∈ a.steps, s.determinate := by
  sorry


end algorithm_steps_are_determinate_l350_35099


namespace total_snails_is_294_l350_35020

/-- The total number of snails found by a family of ducks -/
def total_snails : ℕ :=
  let total_ducklings : ℕ := 8
  let first_group_size : ℕ := 3
  let second_group_size : ℕ := 3
  let first_group_snails_per_duckling : ℕ := 5
  let second_group_snails_per_duckling : ℕ := 9
  let first_group_total : ℕ := first_group_size * first_group_snails_per_duckling
  let second_group_total : ℕ := second_group_size * second_group_snails_per_duckling
  let first_two_groups_total : ℕ := first_group_total + second_group_total
  let mother_duck_snails : ℕ := 3 * first_two_groups_total
  let remaining_ducklings : ℕ := total_ducklings - first_group_size - second_group_size
  let remaining_group_total : ℕ := remaining_ducklings * (mother_duck_snails / 2)
  first_group_total + second_group_total + mother_duck_snails + remaining_group_total

theorem total_snails_is_294 : total_snails = 294 := by
  sorry

end total_snails_is_294_l350_35020


namespace lewis_earnings_theorem_l350_35075

/-- Calculates Lewis's earnings per week without overtime during harvest season. -/
def lewis_earnings_without_overtime (weeks : ℕ) (overtime_pay : ℚ) (total_earnings : ℚ) : ℚ :=
  let total_overtime := overtime_pay * weeks
  let earnings_without_overtime := total_earnings - total_overtime
  earnings_without_overtime / weeks

/-- Proves that Lewis's earnings per week without overtime is approximately $27.61. -/
theorem lewis_earnings_theorem (weeks : ℕ) (overtime_pay : ℚ) (total_earnings : ℚ)
    (h1 : weeks = 1091)
    (h2 : overtime_pay = 939)
    (h3 : total_earnings = 1054997) :
    ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
    |lewis_earnings_without_overtime weeks overtime_pay total_earnings - 27.61| < ε :=
  sorry

end lewis_earnings_theorem_l350_35075


namespace largest_lcm_with_18_l350_35041

theorem largest_lcm_with_18 :
  max (Nat.lcm 18 3) (max (Nat.lcm 18 6) (max (Nat.lcm 18 9) (max (Nat.lcm 18 12) (max (Nat.lcm 18 15) (Nat.lcm 18 18))))) = 90 := by
  sorry

end largest_lcm_with_18_l350_35041


namespace machine_quality_comparison_l350_35047

/-- Data for machine production quality --/
structure MachineData where
  first_class : ℕ
  second_class : ℕ

/-- Calculate the frequency of first-class products --/
def frequency (data : MachineData) : ℚ :=
  data.first_class / (data.first_class + data.second_class)

/-- Calculate K² statistic --/
def k_squared (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem stating the frequencies and significance of difference --/
theorem machine_quality_comparison 
  (machine_a machine_b : MachineData)
  (h_a : machine_a = ⟨150, 50⟩)
  (h_b : machine_b = ⟨120, 80⟩) :
  (frequency machine_a = 3/4) ∧ 
  (frequency machine_b = 3/5) ∧ 
  (k_squared machine_a.first_class machine_a.second_class 
              machine_b.first_class machine_b.second_class > 6635/1000) := by
  sorry

#eval frequency ⟨150, 50⟩
#eval frequency ⟨120, 80⟩
#eval k_squared 150 50 120 80

end machine_quality_comparison_l350_35047


namespace matrix_power_result_l350_35085

theorem matrix_power_result (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B.mulVec (![7, -3]) = ![-14, 6]) :
  (B^4).mulVec (![7, -3]) = ![112, -48] := by
  sorry

end matrix_power_result_l350_35085


namespace isosceles_trapezoid_area_l350_35035

/-- The area of an isosceles trapezoid with bases 4x and 3x, and height x, is 7x²/2 -/
theorem isosceles_trapezoid_area (x : ℝ) : 
  let base1 : ℝ := 4 * x
  let base2 : ℝ := 3 * x
  let height : ℝ := x
  let area : ℝ := (base1 + base2) / 2 * height
  area = 7 * x^2 / 2 := by
sorry

end isosceles_trapezoid_area_l350_35035


namespace artworks_per_quarter_is_two_l350_35066

/-- The number of students in the art club -/
def num_students : ℕ := 15

/-- The number of quarters in a school year -/
def quarters_per_year : ℕ := 4

/-- The total number of artworks collected in two school years -/
def total_artworks : ℕ := 240

/-- The number of artworks each student makes by the end of each quarter -/
def artworks_per_student_per_quarter : ℕ := 2

/-- Theorem stating that the number of artworks each student makes by the end of each quarter is 2 -/
theorem artworks_per_quarter_is_two :
  artworks_per_student_per_quarter * num_students * quarters_per_year * 2 = total_artworks :=
by sorry

end artworks_per_quarter_is_two_l350_35066


namespace xiaoning_pe_score_l350_35034

/-- Calculates the comprehensive score for physical education based on midterm and final exam scores -/
def calculate_pe_score (midterm_score : ℝ) (final_score : ℝ) : ℝ :=
  0.3 * midterm_score + 0.7 * final_score

/-- Theorem: Xiaoning's physical education comprehensive score is 87 points -/
theorem xiaoning_pe_score :
  let max_score : ℝ := 100
  let midterm_weight : ℝ := 0.3
  let final_weight : ℝ := 0.7
  let xiaoning_midterm : ℝ := 80
  let xiaoning_final : ℝ := 90
  calculate_pe_score xiaoning_midterm xiaoning_final = 87 := by
  sorry

#eval calculate_pe_score 80 90

end xiaoning_pe_score_l350_35034


namespace investment_problem_l350_35081

def total_investment : ℝ := 1000
def silver_rate : ℝ := 0.04
def gold_rate : ℝ := 0.06
def years : ℕ := 3
def final_amount : ℝ := 1206.11

def silver_investment (x : ℝ) : ℝ := x * (1 + silver_rate) ^ years
def gold_investment (x : ℝ) : ℝ := (total_investment - x) * (1 + gold_rate) ^ years

theorem investment_problem (x : ℝ) :
  silver_investment x + gold_investment x = final_amount →
  x = 228.14 := by sorry

end investment_problem_l350_35081


namespace water_balloon_ratio_l350_35077

/-- Prove that the ratio of Randy's water balloons to Janice's water balloons is 1:2 -/
theorem water_balloon_ratio 
  (cynthia_balloons : ℕ) 
  (janice_balloons : ℕ) 
  (h1 : cynthia_balloons = 12)
  (h2 : janice_balloons = 6)
  (h3 : cynthia_balloons = 4 * (cynthia_balloons / 4)) :
  (cynthia_balloons / 4) / janice_balloons = 1 / 2 := by
  sorry

end water_balloon_ratio_l350_35077


namespace quadratic_real_roots_condition_l350_35014

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + m = 0) → m ≤ 9/4 := by
sorry

end quadratic_real_roots_condition_l350_35014


namespace max_candies_eaten_l350_35094

/-- The maximum sum of products of pairs from a set of n elements -/
def maxProductSum (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of initial elements on the board -/
def initialCount : ℕ := 30

/-- The theorem stating the maximum number of candies Karlson could eat -/
theorem max_candies_eaten :
  maxProductSum initialCount = 435 := by
  sorry

end max_candies_eaten_l350_35094


namespace sin_negative_three_pi_plus_alpha_l350_35039

theorem sin_negative_three_pi_plus_alpha (α : ℝ) (h : Real.sin (π + α) = 1/3) :
  Real.sin (-3*π + α) = 1/3 := by
  sorry

end sin_negative_three_pi_plus_alpha_l350_35039


namespace units_digit_of_sum_of_powers_l350_35083

theorem units_digit_of_sum_of_powers (a b : ℕ) (ha : a = 15) (hb : b = 220) :
  ∃ k : ℤ, (a + Real.sqrt b)^19 + (a - Real.sqrt b)^19 = 10 * k + 9 := by
  sorry

end units_digit_of_sum_of_powers_l350_35083


namespace vector_angle_cosine_l350_35067

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_angle_cosine (a b c : V) :
  a + b + c = 0 →
  ‖a‖ = 2 →
  ‖b‖ = 3 →
  ‖c‖ = 4 →
  ‖a‖ < ‖b‖ →
  inner a b / (‖a‖ * ‖b‖) = 1/4 := by sorry

end vector_angle_cosine_l350_35067


namespace sum_f_positive_l350_35040

def f (x : ℝ) : ℝ := x^5 + x

theorem sum_f_positive (a b c : ℝ) (hab : a + b > 0) (hbc : b + c > 0) (hca : c + a > 0) :
  f a + f b + f c > 0 := by
  sorry

end sum_f_positive_l350_35040


namespace complex_equation_solution_l350_35065

theorem complex_equation_solution :
  ∃ x : ℂ, (3 : ℂ) + 2 * Complex.I * x = 4 - 5 * Complex.I * x ∧ x = -Complex.I / 7 := by
  sorry

end complex_equation_solution_l350_35065


namespace sum_reciprocals_minus_products_l350_35082

theorem sum_reciprocals_minus_products (a b c : ℚ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a + b + c = a * b * c) : 
  a / b + a / c + b / a + b / c + c / a + c / b - a * b - b * c - c * a = -3 := by
sorry

end sum_reciprocals_minus_products_l350_35082


namespace distinct_values_of_c_l350_35031

theorem distinct_values_of_c (c : ℂ) (p q r : ℂ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (∀ z : ℂ, (z - p) * (z - q) * (z - r) + 1 = (z - c*p) * (z - c*q) * (z - c*r) + 1) →
  ∃ S : Finset ℂ, S.card = 4 ∧ c ∈ S ∧ ∀ x : ℂ, x ∈ S → 
    ∀ z : ℂ, (z - p) * (z - q) * (z - r) + 1 = (z - x*p) * (z - x*q) * (z - x*r) + 1 :=
by
  sorry

end distinct_values_of_c_l350_35031


namespace coordinates_wrt_origin_unchanged_l350_35098

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- The origin of the Cartesian coordinate system -/
def origin : CartesianPoint := ⟨0, 0⟩

/-- The coordinates of a point with respect to the origin -/
def coordinatesWrtOrigin (p : CartesianPoint) : CartesianPoint := p

theorem coordinates_wrt_origin_unchanged (p : CartesianPoint) :
  coordinatesWrtOrigin p = p := by sorry

end coordinates_wrt_origin_unchanged_l350_35098


namespace no_simultaneously_safe_numbers_l350_35018

def is_p_safe (n p : ℕ) : Prop :=
  ∀ k : ℤ, |n - k * p| ≥ 3

def simultaneously_safe (n : ℕ) : Prop :=
  is_p_safe n 5 ∧ is_p_safe n 7 ∧ is_p_safe n 11

theorem no_simultaneously_safe_numbers : 
  ¬ ∃ n : ℕ, n > 0 ∧ n ≤ 500 ∧ simultaneously_safe n := by
  sorry

end no_simultaneously_safe_numbers_l350_35018


namespace average_first_14_even_numbers_l350_35089

theorem average_first_14_even_numbers :
  let first_14_even : List ℕ := List.range 14 |>.map (fun n => 2 * (n + 1))
  (first_14_even.sum / first_14_even.length : ℚ) = 15 := by
  sorry

end average_first_14_even_numbers_l350_35089


namespace closest_fraction_l350_35050

def medals_won : ℚ := 17 / 100

def fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]

theorem closest_fraction :
  ∃ (f : ℚ), f ∈ fractions ∧
    ∀ (g : ℚ), g ∈ fractions → |medals_won - f| ≤ |medals_won - g| ∧
    f = 1/6 :=
  sorry

end closest_fraction_l350_35050


namespace no_valid_rectangle_l350_35043

theorem no_valid_rectangle (a b x y : ℝ) : 
  a < b →
  x < a →
  y < a →
  2 * (x + y) = (2 * (a + b)) / 3 →
  x * y = (a * b) / 3 →
  False :=
by sorry

end no_valid_rectangle_l350_35043


namespace slope_of_line_l350_35076

theorem slope_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → (y - 4) / (-x) = -4 / 7 := by
  sorry

end slope_of_line_l350_35076


namespace abc_is_50_l350_35015

def repeating_decimal (a b c : ℕ) : ℚ :=
  1 + (100 * a + 10 * b + c : ℚ) / 999

theorem abc_is_50 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  12 * (repeating_decimal a b c - (1 + (a : ℚ) / 10 + (b : ℚ) / 100 + (c : ℚ) / 1000)) = 0.6 →
  100 * a + 10 * b + c = 50 := by
sorry

end abc_is_50_l350_35015


namespace arithmetic_progression_not_power_l350_35008

theorem arithmetic_progression_not_power (n : ℕ) (k : ℕ) : 
  let a : ℕ → ℕ := λ i => 4 * i - 2
  ∀ i : ℕ, ∀ r : ℕ, 2 ≤ r → r ≤ n → ¬ ∃ m : ℕ, a i = m ^ r :=
by sorry

end arithmetic_progression_not_power_l350_35008


namespace f_two_equals_six_l350_35017

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 6

theorem f_two_equals_six (a b : ℝ) (h : f a b (-1) = f a b 3) : f a b 2 = 6 := by
  sorry

end f_two_equals_six_l350_35017


namespace complement_union_theorem_l350_35079

-- Define the universal set U
def U : Set ℕ := {x | 0 ≤ x ∧ x < 6}

-- Define set A
def A : Set ℕ := {1, 3, 5}

-- Define set B
def B : Set ℕ := {x ∈ U | x^2 + 4 = 5*x}

-- Theorem statement
theorem complement_union_theorem : 
  (U \ A) ∪ (U \ B) = {0, 2, 3, 4, 5} := by sorry

end complement_union_theorem_l350_35079


namespace laptop_price_calculation_l350_35026

def original_price : ℝ := 1200
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.12

def discounted_price : ℝ := original_price * (1 - discount_rate)
def total_price : ℝ := discounted_price * (1 + tax_rate)

theorem laptop_price_calculation :
  total_price = 940.8 := by sorry

end laptop_price_calculation_l350_35026


namespace equation_root_condition_l350_35052

/-- The equation has a root greater than zero if and only if a = -8 -/
theorem equation_root_condition (a : ℝ) : 
  (∃ x > 0, (3*x - 1) / (x - 3) = a / (3 - x) - 1) ↔ a = -8 := by
  sorry

end equation_root_condition_l350_35052


namespace circle_properties_l350_35063

/-- Theorem about a circle's properties given a specific sum of circumference, diameter, and radius -/
theorem circle_properties (r : ℝ) (h : 2 * Real.pi * r + 2 * r + r = 27.84) : 
  2 * r = 6 ∧ Real.pi * r^2 = 28.26 := by
  sorry

#check circle_properties

end circle_properties_l350_35063


namespace chloe_recycled_28_pounds_l350_35030

/-- Represents the recycling scenario with Chloe and her friends -/
structure RecyclingScenario where
  pounds_per_point : ℕ
  friends_recycled : ℕ
  total_points : ℕ

/-- Calculates the amount of paper Chloe recycled given the recycling scenario -/
def chloe_recycled (scenario : RecyclingScenario) : ℕ :=
  scenario.pounds_per_point * scenario.total_points - scenario.friends_recycled

/-- Theorem stating that Chloe recycled 28 pounds given the specific scenario -/
theorem chloe_recycled_28_pounds : 
  let scenario : RecyclingScenario := {
    pounds_per_point := 6,
    friends_recycled := 2,
    total_points := 5
  }
  chloe_recycled scenario = 28 := by
  sorry

end chloe_recycled_28_pounds_l350_35030


namespace sum_of_triangle_ops_l350_35009

-- Define the triangle operation
def triangle_op (a b c : ℤ) : ℤ := 2*a + 3*b - 4*c

-- State the theorem
theorem sum_of_triangle_ops : 
  triangle_op 2 3 5 + triangle_op 4 6 1 = 15 := by sorry

end sum_of_triangle_ops_l350_35009


namespace slope_product_is_four_l350_35007

-- Define the parabola and line
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0
def line (x y : ℝ) : Prop := y = 2*x - 4

-- Define points A, B, and C
def point_on_parabola_and_line (p : ℝ) (x y : ℝ) : Prop :=
  parabola p x y ∧ line x y

-- Define the vector relation
def vector_relation (p : ℝ) (xA yA xB yB xC yC : ℝ) : Prop :=
  xA + xB = (1/5) * xC ∧ yA + yB = (1/5) * yC

-- Define point M
def point_M (x y : ℝ) : Prop := x = 2 ∧ y = 2

-- Theorem statement
theorem slope_product_is_four (p : ℝ) (xA yA xB yB xC yC : ℝ) :
  point_on_parabola_and_line p xA yA →
  point_on_parabola_and_line p xB yB →
  parabola p xC yC →
  vector_relation p xA yA xB yB xC yC →
  point_M 2 2 →
  ((yA - 2) / (xA - 2)) * ((yB - 2) / (xB - 2)) = 4 :=
sorry

end slope_product_is_four_l350_35007


namespace total_amount_is_1800_l350_35028

/-- Calculates the total amount spent on courses for two semesters --/
def total_amount_spent (
  units_per_semester : ℕ
  ) (science_cost_per_unit : ℚ
  ) (humanities_cost_per_unit : ℚ
  ) (science_units_first : ℕ
  ) (humanities_units_first : ℕ
  ) (science_units_second : ℕ
  ) (humanities_units_second : ℕ
  ) (scholarship_percentage : ℚ
  ) : ℚ :=
  let first_semester_cost := 
    science_cost_per_unit * science_units_first + 
    humanities_cost_per_unit * humanities_units_first
  let second_semester_cost := 
    (1 - scholarship_percentage) * science_cost_per_unit * science_units_second + 
    humanities_cost_per_unit * humanities_units_second
  first_semester_cost + second_semester_cost

theorem total_amount_is_1800 :
  total_amount_spent 20 60 45 12 8 12 8 (1/2) = 1800 := by
  sorry

end total_amount_is_1800_l350_35028


namespace sum_interior_angles_pentagon_l350_35060

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: The sum of the interior angles of a pentagon is 540° -/
theorem sum_interior_angles_pentagon :
  sum_interior_angles 5 = 540 := by
  sorry

end sum_interior_angles_pentagon_l350_35060


namespace sum_equals_three_sqrt_fourteen_over_seven_l350_35055

theorem sum_equals_three_sqrt_fourteen_over_seven
  (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 1) 
  (h2 : x + 2*y + 3*z = Real.sqrt 14) : 
  x + y + z = (3 * Real.sqrt 14) / 7 := by
sorry

end sum_equals_three_sqrt_fourteen_over_seven_l350_35055


namespace x_value_l350_35086

theorem x_value : Real.sqrt (20 - 17 - 2 * 0 - 1 + 7) = 3 := by
  sorry

end x_value_l350_35086


namespace intersection_value_l350_35042

theorem intersection_value (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (3 / a = b) ∧ (a - 1 = b) → 1 / a - 1 / b = -1 / 3 := by
  sorry

end intersection_value_l350_35042


namespace temporary_employee_percentage_is_51_5_l350_35037

/-- Represents the composition of workers in a factory -/
structure WorkerComposition where
  technicians : Real
  skilled_laborers : Real
  unskilled_laborers : Real
  permanent_technicians : Real
  permanent_skilled : Real
  permanent_unskilled : Real

/-- Calculates the percentage of temporary employees in the factory -/
def temporary_employee_percentage (wc : WorkerComposition) : Real :=
  100 - (wc.technicians * wc.permanent_technicians + 
         wc.skilled_laborers * wc.permanent_skilled + 
         wc.unskilled_laborers * wc.permanent_unskilled)

/-- Theorem stating that given the conditions, the percentage of temporary employees is 51.5% -/
theorem temporary_employee_percentage_is_51_5 (wc : WorkerComposition) 
  (h1 : wc.technicians = 40)
  (h2 : wc.skilled_laborers = 35)
  (h3 : wc.unskilled_laborers = 25)
  (h4 : wc.permanent_technicians = 60)
  (h5 : wc.permanent_skilled = 45)
  (h6 : wc.permanent_unskilled = 35) :
  temporary_employee_percentage wc = 51.5 := by
  sorry

end temporary_employee_percentage_is_51_5_l350_35037


namespace ratio_cubes_equals_twentyseven_l350_35062

theorem ratio_cubes_equals_twentyseven : (81000 ^ 3) / (27000 ^ 3) = 27 := by
  sorry

end ratio_cubes_equals_twentyseven_l350_35062


namespace hyperbola_asymptote_implies_a_value_l350_35002

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y = 0

-- Theorem statement
theorem hyperbola_asymptote_implies_a_value :
  ∀ a : ℝ, (∃ x y : ℝ, hyperbola a x y ∧ asymptote x y) →
  a = Real.sqrt 3 / 3 :=
sorry

end hyperbola_asymptote_implies_a_value_l350_35002


namespace prime_divisibility_l350_35038

theorem prime_divisibility (p q : ℕ) : 
  Prime p → Prime q → q ∣ (3^p - 2^p) → p ∣ (q - 1) := by
  sorry

end prime_divisibility_l350_35038


namespace z_value_l350_35070

theorem z_value (a : ℕ) (z : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * z) : z = 49 := by
  sorry

end z_value_l350_35070


namespace eight_T_three_equals_fifty_l350_35049

-- Define the operation T
def T (a b : ℤ) : ℤ := 4*a + 6*b

-- Theorem statement
theorem eight_T_three_equals_fifty : T 8 3 = 50 := by
  sorry

end eight_T_three_equals_fifty_l350_35049


namespace abes_age_problem_l350_35057

theorem abes_age_problem (present_age : ℕ) (sum_ages : ℕ) (years_ago : ℕ) :
  present_age = 19 →
  sum_ages = 31 →
  sum_ages = present_age + (present_age - years_ago) →
  years_ago = 7 := by
sorry

end abes_age_problem_l350_35057


namespace arithmetic_sequence_unique_value_l350_35091

theorem arithmetic_sequence_unique_value (a : ℝ) (a_n : ℕ → ℝ) : 
  a > 0 ∧ 
  (∀ n : ℕ, a_n (n + 1) - a_n n = a_n (n + 2) - a_n (n + 1)) ∧ 
  a_n 1 = a ∧
  (∃! q : ℝ, a_n 2 - a_n 1 = q ∧ (a_n 2 + 2) - (a_n 1 + 1) = q ∧ (a_n 3 + 3) - (a_n 2 + 2) = q) →
  a = 1/3 := by
sorry

end arithmetic_sequence_unique_value_l350_35091


namespace triangle_shape_l350_35019

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) (h : a * Real.cos A = b * Real.cos B) :
  (a = b ∨ a = c ∨ b = c) ∨ (A = Real.pi / 2 ∨ B = Real.pi / 2 ∨ C = Real.pi / 2) :=
sorry

end triangle_shape_l350_35019


namespace power_functions_inequality_l350_35090

theorem power_functions_inequality (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂) :
  (x₁ + x₂)^2 / 4 < (x₁^2 + x₂^2) / 2 ∧
  2 / (x₁ + x₂) < (1 / x₁ + 1 / x₂) / 2 :=
by sorry

end power_functions_inequality_l350_35090
