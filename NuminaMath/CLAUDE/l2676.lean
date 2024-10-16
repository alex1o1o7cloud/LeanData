import Mathlib

namespace NUMINAMATH_CALUDE_certain_number_is_five_l2676_267636

theorem certain_number_is_five (n d : ℕ) (h1 : d > 0) (h2 : n % d = 3) (h3 : (n^2) % d = 4) : d = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_five_l2676_267636


namespace NUMINAMATH_CALUDE_expression_equals_one_l2676_267665

theorem expression_equals_one (x : ℝ) (h1 : x^3 ≠ 1) (h2 : x^3 ≠ -1) :
  ((x^2 + 2*x + 2)^2 * (x^4 - x^2 + 1)^2) / (x^3 + 1)^3 *
  ((x^2 - 2*x + 2)^2 * (x^4 + x^2 + 1)^2) / (x^3 - 1)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2676_267665


namespace NUMINAMATH_CALUDE_distinct_roots_condition_l2676_267649

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : Prop :=
  x^2 - 2*(k-1)*x + k^2 - 1 = 0

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ :=
  (-2*(k-1))^2 - 4*(k^2 - 1)

-- Theorem statement
theorem distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation x k ∧ quadratic_equation y k) ↔ k < 1 := by
  sorry

end NUMINAMATH_CALUDE_distinct_roots_condition_l2676_267649


namespace NUMINAMATH_CALUDE_dana_jayden_pencil_difference_l2676_267661

theorem dana_jayden_pencil_difference :
  ∀ (dana_pencils jayden_pencils marcus_pencils : ℕ),
    jayden_pencils = 20 →
    jayden_pencils = 2 * marcus_pencils →
    dana_pencils = marcus_pencils + 25 →
    dana_pencils - jayden_pencils = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_dana_jayden_pencil_difference_l2676_267661


namespace NUMINAMATH_CALUDE_factorial_fraction_is_integer_l2676_267687

theorem factorial_fraction_is_integer (m n : ℕ) : 
  ∃ k : ℤ, (↑((2 * m).factorial * (2 * n).factorial) : ℚ) / 
    (↑(m.factorial * n.factorial * (m + n).factorial) : ℚ) = ↑k :=
by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_is_integer_l2676_267687


namespace NUMINAMATH_CALUDE_congruence_problem_l2676_267656

theorem congruence_problem (x : ℤ) : 
  (5 * x + 8) % 14 = 3 → (3 * x + 10) % 14 = 7 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2676_267656


namespace NUMINAMATH_CALUDE_game_lives_distribution_l2676_267647

/-- Given a game with initial players, new players joining, and a total number of lives,
    calculate the number of lives per player. -/
def lives_per_player (initial_players : ℕ) (new_players : ℕ) (total_lives : ℕ) : ℕ :=
  total_lives / (initial_players + new_players)

/-- Theorem: In a game with 4 initial players, 5 new players joining, and a total of 27 lives,
    each player has 3 lives. -/
theorem game_lives_distribution :
  lives_per_player 4 5 27 = 3 := by
  sorry

end NUMINAMATH_CALUDE_game_lives_distribution_l2676_267647


namespace NUMINAMATH_CALUDE_sum_at_one_and_neg_one_l2676_267675

/-- A cubic polynomial Q satisfying specific conditions -/
structure CubicPolynomial (l : ℝ) where
  Q : ℝ → ℝ
  is_cubic : ∃ a b c : ℝ, ∀ x, Q x = a * x^3 + b * x^2 + c * x + l
  cond_0 : Q 0 = l
  cond_2 : Q 2 = 3 * l
  cond_neg_2 : Q (-2) = 5 * l

/-- Theorem stating the sum of Q(1) and Q(-1) -/
theorem sum_at_one_and_neg_one (l : ℝ) (poly : CubicPolynomial l) : 
  poly.Q 1 + poly.Q (-1) = (7/2) * l := by
  sorry

end NUMINAMATH_CALUDE_sum_at_one_and_neg_one_l2676_267675


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l2676_267612

theorem right_rectangular_prism_volume
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 24)
  (h_front : front_area = 18)
  (h_bottom : bottom_area = 12) :
  ∃ a b c : ℝ,
    a * b = side_area ∧
    b * c = front_area ∧
    c * a = bottom_area ∧
    a * b * c = 72 :=
by sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l2676_267612


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l2676_267694

theorem angle_sum_in_circle (x : ℝ) : 
  (3 * x + 7 * x + 4 * x + 2 * x + x = 360) → x = 360 / 17 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_circle_l2676_267694


namespace NUMINAMATH_CALUDE_bakery_ratio_l2676_267620

/-- Given the conditions of a bakery's storage room, prove the ratio of flour to baking soda --/
theorem bakery_ratio (sugar flour baking_soda : ℕ) : 
  sugar = 6000 ∧ 
  5 * flour = 2 * sugar ∧ 
  8 * (baking_soda + 60) = flour → 
  10 * baking_soda = flour := by sorry

end NUMINAMATH_CALUDE_bakery_ratio_l2676_267620


namespace NUMINAMATH_CALUDE_rose_orchid_difference_l2676_267688

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 5

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 3

/-- The number of roses finally in the vase -/
def final_roses : ℕ := 12

/-- The number of orchids finally in the vase -/
def final_orchids : ℕ := 2

/-- The difference between the final number of roses and orchids in the vase -/
theorem rose_orchid_difference : final_roses - final_orchids = 10 := by
  sorry

end NUMINAMATH_CALUDE_rose_orchid_difference_l2676_267688


namespace NUMINAMATH_CALUDE_unique_intersection_k_values_l2676_267638

-- Define the complex plane
variable (z : ℂ)

-- Define the equations
def equation1 (z : ℂ) : Prop := Complex.abs (z - 4) = 3 * Complex.abs (z + 4)
def equation2 (z : ℂ) (k : ℝ) : Prop := Complex.abs z = k

-- Define the theorem
theorem unique_intersection_k_values :
  ∃! z, equation1 z ∧ equation2 z k → k = 0.631 ∨ k = 25.369 :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_k_values_l2676_267638


namespace NUMINAMATH_CALUDE_triangle_abc_problem_l2676_267660

theorem triangle_abc_problem (A B C : Real) (a b c : Real) 
  (h1 : b * Real.sin A = 3 * c * Real.sin B)
  (h2 : a = 3)
  (h3 : Real.cos B = 2/3) : 
  b = Real.sqrt 6 ∧ Real.sin (2*B - π/3) = (4*Real.sqrt 5 + Real.sqrt 3) / 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_problem_l2676_267660


namespace NUMINAMATH_CALUDE_range_of_f_l2676_267652

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2, 3}

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2676_267652


namespace NUMINAMATH_CALUDE_power_multiplication_l2676_267679

theorem power_multiplication (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2676_267679


namespace NUMINAMATH_CALUDE_count_pairs_eq_15_l2676_267629

def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 ≤ 6) (Finset.product (Finset.range 6) (Finset.range 6))).card

theorem count_pairs_eq_15 : count_pairs = 15 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_15_l2676_267629


namespace NUMINAMATH_CALUDE_group_purchase_equation_l2676_267627

/-- Represents a group purchase scenario -/
structure GroupPurchase where
  price : ℝ  -- Price of the item
  contribution1 : ℝ  -- First contribution amount per person
  excess : ℝ  -- Excess amount for first contribution
  contribution2 : ℝ  -- Second contribution amount per person
  shortage : ℝ  -- Shortage amount for second contribution

/-- Theorem stating the equation for the group purchase scenario -/
theorem group_purchase_equation (gp : GroupPurchase) 
  (h1 : gp.contribution1 = 8) 
  (h2 : gp.excess = 3) 
  (h3 : gp.contribution2 = 7) 
  (h4 : gp.shortage = 4) :
  (gp.price + gp.excess) / gp.contribution1 = (gp.price - gp.shortage) / gp.contribution2 := by
  sorry

end NUMINAMATH_CALUDE_group_purchase_equation_l2676_267627


namespace NUMINAMATH_CALUDE_expected_no_allergies_is_75_l2676_267644

/-- The probability that an American does not suffer from allergies -/
def prob_no_allergies : ℚ := 1/4

/-- The size of the random sample of Americans -/
def sample_size : ℕ := 300

/-- The expected number of people in the sample who do not suffer from allergies -/
def expected_no_allergies : ℚ := prob_no_allergies * sample_size

theorem expected_no_allergies_is_75 : expected_no_allergies = 75 := by
  sorry

end NUMINAMATH_CALUDE_expected_no_allergies_is_75_l2676_267644


namespace NUMINAMATH_CALUDE_smallest_number_l2676_267609

/-- Converts a number from base k to decimal --/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

/-- The binary number 111111₍₂₎ --/
def binary_num : List Nat := [1, 1, 1, 1, 1, 1]

/-- The base-6 number 150₍₆₎ --/
def base6_num : List Nat := [0, 5, 1]

/-- The base-4 number 1000₍₄₎ --/
def base4_num : List Nat := [0, 0, 0, 1]

/-- The octal number 101₍₈₎ --/
def octal_num : List Nat := [1, 0, 1]

theorem smallest_number :
  to_decimal binary_num 2 < to_decimal base6_num 6 ∧
  to_decimal binary_num 2 < to_decimal base4_num 4 ∧
  to_decimal binary_num 2 < to_decimal octal_num 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2676_267609


namespace NUMINAMATH_CALUDE_digit_sum_properties_l2676_267653

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Permutation of digits relation -/
def is_digit_permutation (m k : ℕ) : Prop := sorry

theorem digit_sum_properties (M K : ℕ) (h : is_digit_permutation M K) : 
  (sum_of_digits (2 * M) = sum_of_digits (2 * K)) ∧ 
  (M % 2 = 0 → K % 2 = 0 → sum_of_digits (M / 2) = sum_of_digits (K / 2)) ∧
  (sum_of_digits (5 * M) = sum_of_digits (5 * K)) := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_properties_l2676_267653


namespace NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l2676_267614

theorem complex_exp_13pi_over_2 : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l2676_267614


namespace NUMINAMATH_CALUDE_heartsuit_ratio_l2676_267669

def heartsuit (n m : ℝ) : ℝ := n^2 * m^3

theorem heartsuit_ratio : (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_ratio_l2676_267669


namespace NUMINAMATH_CALUDE_coin_distribution_theorem_l2676_267674

/-- Represents the number of rounds in the coin distribution -/
def x : ℕ := sorry

/-- Pete's coins after distribution -/
def pete_coins (x : ℕ) : ℕ := x * (x + 1) / 2

/-- Paul's coins after distribution -/
def paul_coins (x : ℕ) : ℕ := x

/-- The condition that Pete has three times as many coins as Paul -/
axiom pete_triple_paul : pete_coins x = 3 * paul_coins x

/-- The total number of coins -/
def total_coins (x : ℕ) : ℕ := pete_coins x + paul_coins x

theorem coin_distribution_theorem : total_coins x = 20 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_theorem_l2676_267674


namespace NUMINAMATH_CALUDE_hollow_cube_side_length_l2676_267637

/-- Represents the number of cubes used to create a hollow cube -/
def hollow_cube_cubes (n : ℕ) : ℕ := 6 * n^2 - (n^2 + 4 * (n - 2))

/-- Theorem stating that if 98 cubes are used to make a hollow cube, its side length is 9 -/
theorem hollow_cube_side_length :
  ∃ (n : ℕ), hollow_cube_cubes n = 98 ∧ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_hollow_cube_side_length_l2676_267637


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2676_267673

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (1 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 68 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2676_267673


namespace NUMINAMATH_CALUDE_f_evaluation_l2676_267617

def f (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x - 7

theorem f_evaluation : 2 * f 2 + 3 * f (-2) = -107 := by
  sorry

end NUMINAMATH_CALUDE_f_evaluation_l2676_267617


namespace NUMINAMATH_CALUDE_office_network_connections_l2676_267623

/-- A network of switches with connections between them. -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ

/-- The total number of connections in a switch network. -/
def total_connections (network : SwitchNetwork) : ℕ :=
  (network.num_switches * network.connections_per_switch) / 2

/-- The theorem stating that a network of 40 switches, each connected to 4 others, has 80 connections. -/
theorem office_network_connections :
  let network : SwitchNetwork := { num_switches := 40, connections_per_switch := 4 }
  total_connections network = 80 := by
  sorry

end NUMINAMATH_CALUDE_office_network_connections_l2676_267623


namespace NUMINAMATH_CALUDE_triangle_circle_area_relation_l2676_267650

theorem triangle_circle_area_relation (A B C : ℝ) : 
  -- The triangle is inscribed in a circle
  -- The triangle has side lengths of 20, 21, and 29
  -- A, B, and C are the areas of the three parts outside the triangle
  -- C is the largest area among A, B, and C
  (20 : ℝ)^2 + 21^2 = 29^2 →  -- This ensures it's a right triangle
  A ≥ 0 → B ≥ 0 → C ≥ 0 →
  C ≥ A → C ≥ B →
  -- Prove the relation
  A + B + 210 = C := by
  sorry

end NUMINAMATH_CALUDE_triangle_circle_area_relation_l2676_267650


namespace NUMINAMATH_CALUDE_deposit_percentage_l2676_267634

def deposit : ℝ := 3800
def monthly_income : ℝ := 11875

theorem deposit_percentage : (deposit / monthly_income) * 100 = 32 := by
  sorry

end NUMINAMATH_CALUDE_deposit_percentage_l2676_267634


namespace NUMINAMATH_CALUDE_shaded_cubes_count_l2676_267630

/-- Represents a 4x4x4 cube with shaded faces -/
structure ShadedCube where
  /-- The number of smaller cubes along each edge of the large cube -/
  size : Nat
  /-- The number of shaded cubes in the central area of each face -/
  centralShaded : Nat
  /-- The number of shaded corner cubes per face -/
  cornerShaded : Nat

/-- Calculates the total number of uniquely shaded cubes -/
def totalShadedCubes (cube : ShadedCube) : Nat :=
  sorry

/-- Theorem stating that the total number of shaded cubes is 16 -/
theorem shaded_cubes_count (cube : ShadedCube) 
  (h1 : cube.size = 4)
  (h2 : cube.centralShaded = 4)
  (h3 : cube.cornerShaded = 1) : 
  totalShadedCubes cube = 16 := by
  sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_l2676_267630


namespace NUMINAMATH_CALUDE_saree_price_calculation_l2676_267606

/-- Calculates the final price after applying multiple discounts and a tax rate -/
def finalPrice (originalPrice : ℝ) (discounts : List ℝ) (taxRate : ℝ) : ℝ :=
  let discountedPrice := discounts.foldl (fun price discount => price * (1 - discount)) originalPrice
  discountedPrice * (1 + taxRate)

/-- Theorem: The final price of a 510 Rs item after specific discounts and tax is approximately 302.13 Rs -/
theorem saree_price_calculation :
  let originalPrice : ℝ := 510
  let discounts : List ℝ := [0.12, 0.15, 0.20, 0.10]
  let taxRate : ℝ := 0.10
  abs (finalPrice originalPrice discounts taxRate - 302.13) < 0.01 := by
  sorry

#eval finalPrice 510 [0.12, 0.15, 0.20, 0.10] 0.10

end NUMINAMATH_CALUDE_saree_price_calculation_l2676_267606


namespace NUMINAMATH_CALUDE_tv_price_increase_l2676_267666

theorem tv_price_increase (P : ℝ) (x : ℝ) (h : P > 0) :
  (0.80 * P + x / 100 * (0.80 * P) = 1.16 * P) → x = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_tv_price_increase_l2676_267666


namespace NUMINAMATH_CALUDE_modulus_of_z_l2676_267671

-- Define the imaginary unit
noncomputable def i : ℂ := Complex.I

-- Define z
noncomputable def z : ℂ := 4 / (1 + i)^4 - 3 * i

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2676_267671


namespace NUMINAMATH_CALUDE_max_profit_selling_price_l2676_267625

/-- Represents the profit function for a product sale --/
def profit_function (initial_cost initial_price initial_sales price_sensitivity : ℝ) (x : ℝ) : ℝ :=
  (x - initial_cost) * (initial_sales - (x - initial_price) * price_sensitivity)

/-- Theorem stating the maximum profit and optimal selling price --/
theorem max_profit_selling_price 
  (initial_cost : ℝ) 
  (initial_price : ℝ) 
  (initial_sales : ℝ) 
  (price_sensitivity : ℝ)
  (h_initial_cost : initial_cost = 8)
  (h_initial_price : initial_price = 10)
  (h_initial_sales : initial_sales = 60)
  (h_price_sensitivity : price_sensitivity = 10) :
  ∃ (max_profit optimal_price : ℝ),
    max_profit = 160 ∧ 
    optimal_price = 12 ∧
    ∀ x, profit_function initial_cost initial_price initial_sales price_sensitivity x ≤ max_profit :=
by sorry

end NUMINAMATH_CALUDE_max_profit_selling_price_l2676_267625


namespace NUMINAMATH_CALUDE_remainder_of_p_l2676_267632

-- Define the polynomial p(x)
def p (x : ℝ) (r : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (x + 1) * (x - 2)^2 * r x + a * x + b

-- State the theorem
theorem remainder_of_p (r : ℝ → ℝ) (a b : ℝ) :
  (p 2 r a b = 6) →
  (p (-1) r a b = 0) →
  ∃ q : ℝ → ℝ, ∀ x, p x r a b = (x + 1) * (x - 2)^2 * q x + 2 * x + 2 :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_p_l2676_267632


namespace NUMINAMATH_CALUDE_nth_prime_greater_than_3n_l2676_267639

-- Define the n-th prime number
def nth_prime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem nth_prime_greater_than_3n (n : ℕ) (h : n > 12) : nth_prime n > 3 * n := by
  sorry

end NUMINAMATH_CALUDE_nth_prime_greater_than_3n_l2676_267639


namespace NUMINAMATH_CALUDE_max_table_sum_l2676_267646

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def is_valid_arrangement (top : List ℕ) (left : List ℕ) : Prop :=
  top.length = 3 ∧ left.length = 4 ∧ 
  (∀ x ∈ top, x ∈ primes) ∧ 
  (∀ x ∈ left, x ∈ primes) ∧
  17 ∈ top ∧
  (∀ x ∈ primes, x ∈ top ∨ x ∈ left) ∧
  (∀ x ∈ top, ∀ y ∈ left, x ≠ y)

def table_sum (top : List ℕ) (left : List ℕ) : ℕ :=
  (top.sum) * (left.sum)

theorem max_table_sum :
  ∀ top left, is_valid_arrangement top left →
  table_sum top left ≤ 825 :=
sorry

end NUMINAMATH_CALUDE_max_table_sum_l2676_267646


namespace NUMINAMATH_CALUDE_department_age_analysis_l2676_267684

/-- Represents the age data for a department -/
def DepartmentData := List Nat

/-- Calculate the mode of a list of numbers -/
def mode (data : DepartmentData) : Nat :=
  sorry

/-- Calculate the median of a list of numbers -/
def median (data : DepartmentData) : Nat :=
  sorry

/-- Calculate the average of a list of numbers -/
def average (data : DepartmentData) : Rat :=
  sorry

theorem department_age_analysis 
  (dept_A dept_B : DepartmentData)
  (h1 : dept_A.length = 10)
  (h2 : dept_B.length = 10)
  (h3 : dept_A = [21, 23, 25, 26, 27, 28, 30, 32, 32, 32])
  (h4 : dept_B = [20, 22, 24, 24, 26, 28, 28, 30, 34, 40]) :
  (mode dept_A = 32) ∧ 
  (median dept_B = 26) ∧ 
  (average dept_A < average dept_B) :=
sorry

end NUMINAMATH_CALUDE_department_age_analysis_l2676_267684


namespace NUMINAMATH_CALUDE_dress_discount_percentage_l2676_267699

/-- Calculates the final discount percentage for a dress purchase with multiple discounts -/
theorem dress_discount_percentage (original_price : ℝ) (store_discount : ℝ) (member_discount : ℝ) :
  original_price = 350 →
  store_discount = 0.20 →
  member_discount = 0.10 →
  let price_after_store_discount := original_price * (1 - store_discount)
  let final_price := price_after_store_discount * (1 - member_discount)
  let total_discount := original_price - final_price
  let final_discount_percentage := (total_discount / original_price) * 100
  ∃ ε > 0, |final_discount_percentage - 28| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_dress_discount_percentage_l2676_267699


namespace NUMINAMATH_CALUDE_max_pieces_in_5x5_grid_l2676_267657

theorem max_pieces_in_5x5_grid : ∀ (n : ℕ),
  (∃ (areas : List ℕ), 
    areas.length = n ∧ 
    areas.sum = 25 ∧ 
    areas.Nodup ∧ 
    (∀ a ∈ areas, a > 0)) →
  n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_pieces_in_5x5_grid_l2676_267657


namespace NUMINAMATH_CALUDE_tennis_tournament_l2676_267613

theorem tennis_tournament (n : ℕ) : n > 0 → (
  let total_players := 4 * n
  let total_matches := (total_players * (total_players - 1)) / 2
  let women_wins := 3 * n * (3 * n)
  let men_wins := 3 * n * n
  women_wins + men_wins = total_matches ∧
  3 * men_wins = 2 * women_wins
) → n = 4 := by sorry

end NUMINAMATH_CALUDE_tennis_tournament_l2676_267613


namespace NUMINAMATH_CALUDE_custodian_jugs_theorem_l2676_267670

/-- The number of jugs needed to provide water for students -/
def jugs_needed (jug_capacity : ℕ) (num_students : ℕ) (cups_per_student : ℕ) : ℕ :=
  (num_students * cups_per_student + jug_capacity - 1) / jug_capacity

/-- Theorem: Given the conditions, 50 jugs are needed -/
theorem custodian_jugs_theorem :
  jugs_needed 40 200 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_custodian_jugs_theorem_l2676_267670


namespace NUMINAMATH_CALUDE_interesting_numbers_characterization_l2676_267677

def is_interesting (n : ℕ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧
    n = ⌊1/a⌋ + ⌊1/b⌋ + ⌊1/c⌋

theorem interesting_numbers_characterization :
  ∀ n : ℕ, is_interesting n ↔ n ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_interesting_numbers_characterization_l2676_267677


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2676_267668

theorem sqrt_equation_solution : 
  let x : ℝ := 12/5
  (Real.sqrt (6*x)) / (Real.sqrt (4*(x-2))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2676_267668


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l2676_267683

-- Define the set of real polynomials
def RealPolynomial := Polynomial ℝ

-- Define the condition for a, b, c
def SumProductZero (a b c : ℝ) : Prop := a * b + b * c + c * a = 0

-- Define the equation that P must satisfy
def SatisfiesEquation (P : RealPolynomial) : Prop :=
  ∀ a b c : ℝ, SumProductZero a b c →
    P.eval (a - b) + P.eval (b - c) + P.eval (c - a) = 2 * P.eval (a + b + c)

-- Define the form of the solution polynomial
def IsSolutionForm (P : RealPolynomial) : Prop :=
  ∃ u v : ℝ, P = Polynomial.monomial 4 u + Polynomial.monomial 2 v

-- State the theorem
theorem polynomial_equation_solution :
  ∀ P : RealPolynomial, SatisfiesEquation P → IsSolutionForm P :=
sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l2676_267683


namespace NUMINAMATH_CALUDE_circle_symmetry_implies_a_value_l2676_267690

/-- A circle C with equation x^2 + y^2 + 2x + ay - 10 = 0, where a is a real number -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + a*p.2 - 10 = 0}

/-- The line l with equation x - y + 2 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 2 = 0}

/-- A point is symmetric about a line if the line is the perpendicular bisector of the line segment
    joining the point and its reflection -/
def IsSymmetricAbout (p q : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  q ∈ l ∧ (p.1 + q.1) / 2 = q.1 ∧ (p.2 + q.2) / 2 = q.2

theorem circle_symmetry_implies_a_value (a : ℝ) :
  (∀ p ∈ Circle a, ∃ q, q ∈ Circle a ∧ IsSymmetricAbout p q Line) →
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetry_implies_a_value_l2676_267690


namespace NUMINAMATH_CALUDE_least_number_to_add_l2676_267604

theorem least_number_to_add (n : ℕ) : 
  (∀ m : ℕ, m < 234 → ¬(5 ∣ (2496 + m) ∧ 7 ∣ (2496 + m) ∧ 13 ∣ (2496 + m))) ∧ 
  (5 ∣ (2496 + 234) ∧ 7 ∣ (2496 + 234) ∧ 13 ∣ (2496 + 234)) := by
  sorry

end NUMINAMATH_CALUDE_least_number_to_add_l2676_267604


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l2676_267693

/-- Given two people moving in opposite directions, this theorem proves the speed of one person
    given the speed of the other and their final distance after a certain time. -/
theorem opposite_direction_speed
  (time : ℝ)
  (speed_person2 : ℝ)
  (final_distance : ℝ)
  (h1 : time > 0)
  (h2 : speed_person2 > 0)
  (h3 : final_distance > 0)
  (h4 : final_distance = (speed_person1 + speed_person2) * time)
  (h5 : time = 4)
  (h6 : speed_person2 = 3)
  (h7 : final_distance = 36) :
  speed_person1 = 6 :=
sorry

end NUMINAMATH_CALUDE_opposite_direction_speed_l2676_267693


namespace NUMINAMATH_CALUDE_train_crossing_time_l2676_267626

/-- Time for a train to cross a man moving in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 270 →
  train_speed = 25 →
  man_speed = 2 →
  (train_length / ((train_speed + man_speed) * (1000 / 3600))) = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2676_267626


namespace NUMINAMATH_CALUDE_problem_solution_l2676_267642

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- State the theorem
theorem problem_solution (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, ¬(p x a) → ¬(q x)) 
  (h3 : ∃ x, ¬(p x a) ∧ (q x)) :
  (a = 1 → ∃ x, x > 2 ∧ x < 3 ∧ p x a ∧ q x) ∧
  (a > 1 ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2676_267642


namespace NUMINAMATH_CALUDE_triangular_array_sum_of_digits_l2676_267615

/-- The sum of the first n positive integers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The number of rows in the triangular array -/
def N : ℕ := 77

theorem triangular_array_sum_of_digits :
  (triangular_number N = 3003) ∧ (sum_of_digits N = 14) := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_sum_of_digits_l2676_267615


namespace NUMINAMATH_CALUDE_pentagon_to_squares_area_ratio_l2676_267622

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  a : Point
  side : ℝ

/-- Calculate the area of a square -/
def squareArea (s : Square) : ℝ := s.side * s.side

/-- Calculate the area of a pentagon given its vertices -/
def pentagonArea (a b c d e : Point) : ℝ := sorry

/-- Main theorem: The ratio of the pentagon area to the sum of square areas is 5/12 -/
theorem pentagon_to_squares_area_ratio 
  (squareABCD squareEFGH squareKLMO : Square)
  (a b c d e f g h k l m o : Point) :
  squareABCD.side = 1 →
  squareEFGH.side = 2 →
  squareKLMO.side = 1 →
  b.x = h.x ∧ b.y = e.y → -- AB aligns with HE
  g.x = o.x ∧ m.y = k.y → -- GM aligns with OK
  d.x = (h.x + e.x) / 2 ∧ d.y = h.y → -- D is midpoint of HE
  c.x = h.x + (2/3) * (g.x - h.x) ∧ c.y = h.y → -- C is one-third along HG from H
  (pentagonArea a m k c b) / (squareArea squareABCD + squareArea squareEFGH + squareArea squareKLMO) = 5/12 := by
  sorry


end NUMINAMATH_CALUDE_pentagon_to_squares_area_ratio_l2676_267622


namespace NUMINAMATH_CALUDE_peanut_butter_probability_l2676_267692

def jenny_peanut_butter : ℕ := 40
def jenny_chocolate_chip : ℕ := 50
def marcus_peanut_butter : ℕ := 30
def marcus_lemon : ℕ := 20

def total_cookies : ℕ := jenny_peanut_butter + jenny_chocolate_chip + marcus_peanut_butter + marcus_lemon
def peanut_butter_cookies : ℕ := jenny_peanut_butter + marcus_peanut_butter

theorem peanut_butter_probability :
  (peanut_butter_cookies : ℚ) / (total_cookies : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_probability_l2676_267692


namespace NUMINAMATH_CALUDE_sequence_not_contains_010101_l2676_267689

/-- Represents a sequence where each term after the sixth is the last digit of the sum of the previous six terms -/
def Sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 0
  | 4 => 1
  | 5 => 0
  | n + 6 => (Sequence n + Sequence (n + 1) + Sequence (n + 2) + Sequence (n + 3) + Sequence (n + 4) + Sequence (n + 5)) % 10

/-- The weighted sum function used in the proof -/
def S (a b c d e f : ℕ) : ℕ := 2*a + 4*b + 6*c + 8*d + 10*e + 12*f

theorem sequence_not_contains_010101 :
  ∀ n : ℕ, ¬(Sequence n = 0 ∧ Sequence (n + 1) = 1 ∧ Sequence (n + 2) = 0 ∧
            Sequence (n + 3) = 1 ∧ Sequence (n + 4) = 0 ∧ Sequence (n + 5) = 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_not_contains_010101_l2676_267689


namespace NUMINAMATH_CALUDE_subtract_three_from_M_l2676_267643

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc
    else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

def M : List Bool := [false, false, false, false, true, true, false, true]

theorem subtract_three_from_M :
  decimal_to_binary (binary_to_decimal M - 3) = 
    [true, false, true, true, false, true, false, true] := by
  sorry

end NUMINAMATH_CALUDE_subtract_three_from_M_l2676_267643


namespace NUMINAMATH_CALUDE_fries_sold_total_l2676_267662

/-- Represents the number of fries sold -/
structure FriesSold where
  small : ℕ
  large : ℕ

/-- Calculates the total number of fries sold -/
def total_fries (f : FriesSold) : ℕ := f.small + f.large

/-- Theorem: If 4 small fries were sold and the ratio of large to small fries is 5:1, 
    then the total number of fries sold is 24 -/
theorem fries_sold_total (f : FriesSold) 
    (h1 : f.small = 4) 
    (h2 : f.large = 5 * f.small) : 
  total_fries f = 24 := by
  sorry


end NUMINAMATH_CALUDE_fries_sold_total_l2676_267662


namespace NUMINAMATH_CALUDE_davids_math_marks_l2676_267600

theorem davids_math_marks (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) 
  (h1 : english = 76)
  (h2 : physics = 82)
  (h3 : chemistry = 67)
  (h4 : biology = 85)
  (h5 : average = 75)
  (h6 : (english + physics + chemistry + biology + mathematics) / 5 = average) :
  mathematics = 65 := by
  sorry

#check davids_math_marks

end NUMINAMATH_CALUDE_davids_math_marks_l2676_267600


namespace NUMINAMATH_CALUDE_solve_for_B_l2676_267676

theorem solve_for_B : ∃ B : ℝ, (4 * B + 5 = 25) ∧ (B = 5) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_B_l2676_267676


namespace NUMINAMATH_CALUDE_parallelogram_area_l2676_267605

/-- The area of a parallelogram with base 20 meters and height 4 meters is 80 square meters. -/
theorem parallelogram_area :
  let base : ℝ := 20
  let height : ℝ := 4
  let area : ℝ := base * height
  area = 80 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2676_267605


namespace NUMINAMATH_CALUDE_repeating_ones_not_square_l2676_267682

/-- Defines a function that returns a number consisting of n repeating 1's -/
def repeatingOnes (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Theorem stating that for any positive natural number n, 
    the number consisting of n repeating 1's is not a perfect square -/
theorem repeating_ones_not_square (n : ℕ) (h : n > 0) : 
  ¬ ∃ m : ℕ, (repeatingOnes n) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_repeating_ones_not_square_l2676_267682


namespace NUMINAMATH_CALUDE_fraction_equality_l2676_267696

theorem fraction_equality : (1/4 - 1/5) / (1/3 - 1/6 + 1/12) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2676_267696


namespace NUMINAMATH_CALUDE_audiobook_listening_time_l2676_267631

theorem audiobook_listening_time 
  (num_books : ℕ) 
  (book_length : ℕ) 
  (daily_listening : ℕ) 
  (h1 : num_books = 6) 
  (h2 : book_length = 30) 
  (h3 : daily_listening = 2) : 
  (num_books * book_length) / daily_listening = 90 := by
sorry

end NUMINAMATH_CALUDE_audiobook_listening_time_l2676_267631


namespace NUMINAMATH_CALUDE_value_of_A_l2676_267641

/-- Given the values of words and letters, prove the value of A -/
theorem value_of_A (L LEAD DEAL DELL : ℤ) (h1 : L = 15) (h2 : LEAD = 50) (h3 : DEAL = 55) (h4 : DELL = 60) : ∃ A : ℤ, A = 25 := by
  sorry

end NUMINAMATH_CALUDE_value_of_A_l2676_267641


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l2676_267691

theorem cube_edge_ratio (a b : ℝ) (h : a^3 / b^3 = 27 / 1) : a / b = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l2676_267691


namespace NUMINAMATH_CALUDE_opposite_number_any_real_l2676_267695

theorem opposite_number_any_real (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x + y = 0) → 
  (∃ b : ℝ, a + b = 0 ∧ b = -a) → 
  True :=
by sorry

end NUMINAMATH_CALUDE_opposite_number_any_real_l2676_267695


namespace NUMINAMATH_CALUDE_unique_start_day_l2676_267621

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- A function that determines if a given day is the first day of a 30-day month with equal Saturdays and Sundays -/
def is_valid_start_day (d : DayOfWeek) : Prop :=
  ∃ (sat_count sun_count : ℕ),
    sat_count = sun_count ∧
    sat_count + sun_count ≤ 30 ∧
    (match d with
      | DayOfWeek.Monday    => sat_count = 4 ∧ sun_count = 4
      | DayOfWeek.Tuesday   => sat_count = 4 ∧ sun_count = 4
      | DayOfWeek.Wednesday => sat_count = 4 ∧ sun_count = 4
      | DayOfWeek.Thursday  => sat_count = 4 ∧ sun_count = 4
      | DayOfWeek.Friday    => sat_count = 4 ∧ sun_count = 4
      | DayOfWeek.Saturday  => sat_count = 5 ∧ sun_count = 5
      | DayOfWeek.Sunday    => sat_count = 5 ∧ sun_count = 4)

/-- Theorem stating that there is exactly one day of the week that can be the first day of a 30-day month with equal Saturdays and Sundays -/
theorem unique_start_day :
  ∃! (d : DayOfWeek), is_valid_start_day d :=
sorry

end NUMINAMATH_CALUDE_unique_start_day_l2676_267621


namespace NUMINAMATH_CALUDE_cassidy_poster_addition_l2676_267618

/-- The number of posters Cassidy had two years ago -/
def posters_two_years_ago : ℕ := 14

/-- The number of posters Cassidy has currently -/
def current_posters : ℕ := 22

/-- The number of posters Cassidy will have after this summer -/
def future_posters : ℕ := 2 * posters_two_years_ago

/-- The number of posters Cassidy will add this summer -/
def posters_to_add : ℕ := future_posters - current_posters

theorem cassidy_poster_addition : posters_to_add = 6 := by
  sorry

end NUMINAMATH_CALUDE_cassidy_poster_addition_l2676_267618


namespace NUMINAMATH_CALUDE_fraction_difference_2023_2022_l2676_267640

theorem fraction_difference_2023_2022 : 
  (2023 : ℚ) / 2022 - 2022 / 2023 = 4045 / (2022 * 2023) := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_2023_2022_l2676_267640


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l2676_267686

theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1)
  f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l2676_267686


namespace NUMINAMATH_CALUDE_moles_CH₄_required_l2676_267697

/-- Represents a chemical species in a reaction --/
inductive Species
| CH₄ : Species
| Cl₂ : Species
| CHCl₃ : Species
| HCl : Species

/-- Represents the stoichiometric coefficients in a chemical reaction --/
def reaction_coefficients : Species → ℚ
| Species.CH₄ => -1
| Species.Cl₂ => -3
| Species.CHCl₃ => 1
| Species.HCl => 3

/-- The number of moles of CHCl₃ formed --/
def moles_CHCl₃_formed : ℚ := 3

/-- Theorem stating that the number of moles of CH₄ required to form 3 moles of CHCl₃ is 3 moles --/
theorem moles_CH₄_required :
  -reaction_coefficients Species.CH₄ * moles_CHCl₃_formed = 3 := by sorry

end NUMINAMATH_CALUDE_moles_CH₄_required_l2676_267697


namespace NUMINAMATH_CALUDE_rock_song_requests_l2676_267655

/-- Represents the number of song requests for each genre --/
structure SongRequests where
  total : ℕ
  electropop : ℕ
  dance : ℕ
  rock : ℕ
  oldies : ℕ
  dj_choice : ℕ
  rap : ℕ

/-- Theorem stating the number of rock song requests given the conditions --/
theorem rock_song_requests (req : SongRequests) : req.rock = 5 :=
  by
  have h1 : req.total = 30 := by sorry
  have h2 : req.electropop = req.total / 2 := by sorry
  have h3 : req.dance = req.electropop / 3 := by sorry
  have h4 : req.oldies = req.rock - 3 := by sorry
  have h5 : req.dj_choice = req.oldies / 2 := by sorry
  have h6 : req.rap = 2 := by sorry
  have h7 : req.total = req.electropop + req.dance + req.rap + req.rock + req.oldies + req.dj_choice := by sorry
  sorry

end NUMINAMATH_CALUDE_rock_song_requests_l2676_267655


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l2676_267680

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ s : ℝ, s > 0 ∧ 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = s^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = s^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = s^2

-- Define midpoint
def Midpoint (M X Y : ℝ × ℝ) : Prop :=
  M.1 = (X.1 + Y.1) / 2 ∧ M.2 = (X.2 + Y.2) / 2

-- Define the theorem
theorem shaded_area_ratio 
  (A B C D E F G H : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : Midpoint D A B) 
  (h3 : Midpoint E B C) 
  (h4 : Midpoint F C A) 
  (h5 : Midpoint G D F) 
  (h6 : Midpoint H F E) :
  let shaded_area := 
    -- Area of triangle DEF + Area of three trapezoids
    (Real.sqrt 3 / 16 + 9 * Real.sqrt 3 / 32) * s^2
  let non_shaded_area := 
    -- Total area of triangle ABC - Shaded area
    (Real.sqrt 3 / 4 - 11 * Real.sqrt 3 / 32) * s^2
  shaded_area / non_shaded_area = 11 / 21 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l2676_267680


namespace NUMINAMATH_CALUDE_cube_root_of_x_sqrt_x_l2676_267664

theorem cube_root_of_x_sqrt_x (x : ℝ) (hx : x > 0) : 
  (x * Real.sqrt x) ^ (1/3 : ℝ) = x ^ (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_cube_root_of_x_sqrt_x_l2676_267664


namespace NUMINAMATH_CALUDE_ratio_transformation_l2676_267685

theorem ratio_transformation (a b c d x : ℚ) : 
  a = 4 ∧ b = 15 ∧ c = 3 ∧ d = 4 ∧ x = 29 →
  (a + x) / (b + x) = c / d := by
sorry

end NUMINAMATH_CALUDE_ratio_transformation_l2676_267685


namespace NUMINAMATH_CALUDE_always_two_real_roots_find_p_l2676_267610

-- Define the quadratic equation
def quadratic_equation (x p : ℝ) : Prop :=
  (x - 3) * (x - 2) = p * (p + 1)

-- Theorem 1: The equation always has two real roots
theorem always_two_real_roots (p : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ p ∧ quadratic_equation x₂ p :=
sorry

-- Theorem 2: If the roots satisfy the given condition, then p = -2
theorem find_p (p : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : quadratic_equation x₁ p)
  (h₂ : quadratic_equation x₂ p)
  (h₃ : x₁^2 + x₂^2 - x₁*x₂ = 3*p^2 + 1) :
  p = -2 :=
sorry

end NUMINAMATH_CALUDE_always_two_real_roots_find_p_l2676_267610


namespace NUMINAMATH_CALUDE_min_red_to_blue_l2676_267672

/-- Represents the colors of chameleons -/
inductive Color
  | Red
  | Blue
  | Green
  | Yellow
  | Purple

/-- Represents a chameleon -/
structure Chameleon where
  color : Color

/-- Represents the color change rule -/
def colorChangeRule (biter : Color) (bitten : Color) : Color :=
  sorry -- Specific implementation not provided in the problem

/-- Represents a sequence of bites -/
def BiteSequence := List (Nat × Nat)

/-- Function to apply a bite sequence to a list of chameleons -/
def applyBiteSequence (chameleons : List Chameleon) (sequence : BiteSequence) : List Chameleon :=
  sorry -- Implementation would depend on colorChangeRule

/-- Predicate to check if all chameleons in a list are blue -/
def allBlue (chameleons : List Chameleon) : Prop :=
  ∀ c ∈ chameleons, c.color = Color.Blue

/-- The main theorem to be proved -/
theorem min_red_to_blue :
  ∀ n : Nat,
    (n ≥ 5 →
      ∃ (sequence : BiteSequence),
        allBlue (applyBiteSequence (List.replicate n (Chameleon.mk Color.Red)) sequence)) ∧
    (n < 5 →
      ¬∃ (sequence : BiteSequence),
        allBlue (applyBiteSequence (List.replicate n (Chameleon.mk Color.Red)) sequence)) :=
  sorry


end NUMINAMATH_CALUDE_min_red_to_blue_l2676_267672


namespace NUMINAMATH_CALUDE_max_stores_visited_l2676_267616

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (double_visitors : ℕ) (total_shoppers : ℕ) 
  (h1 : total_stores = 8)
  (h2 : total_visits = 23)
  (h3 : double_visitors = 8)
  (h4 : total_shoppers = 12)
  (h5 : double_visitors * 2 ≤ total_visits)
  (h6 : ∀ n : ℕ, n ≤ total_shoppers → n > 0) :
  ∃ max_visits : ℕ, max_visits = 7 ∧ 
    (∀ person : ℕ, person ≤ total_shoppers → 
      ∃ visits : ℕ, visits ≤ max_visits ∧ 
        (double_visitors * 2 + (total_shoppers - double_visitors) * visits ≤ total_visits)) :=
by sorry

end NUMINAMATH_CALUDE_max_stores_visited_l2676_267616


namespace NUMINAMATH_CALUDE_quadratic_shift_l2676_267619

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := 2 * x^2

-- Define the vertical shift
def vertical_shift : ℝ := 1

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := original_function x + vertical_shift

-- Theorem statement
theorem quadratic_shift :
  ∀ x : ℝ, shifted_function x = 2 * x^2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_shift_l2676_267619


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2676_267603

theorem negation_of_proposition :
  ¬(∀ x : ℝ, x > 0 → x^2 - x ≤ 1) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2676_267603


namespace NUMINAMATH_CALUDE_max_apples_capacity_l2676_267654

theorem max_apples_capacity (num_boxes : ℕ) (trays_per_box : ℕ) (extra_trays : ℕ) (apples_per_tray : ℕ) : 
  num_boxes = 6 → trays_per_box = 12 → extra_trays = 4 → apples_per_tray = 8 →
  (num_boxes * trays_per_box + extra_trays) * apples_per_tray = 608 := by
  sorry

end NUMINAMATH_CALUDE_max_apples_capacity_l2676_267654


namespace NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l2676_267658

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 7*x

-- State the theorem
theorem f_difference_at_3_and_neg_3 : f 3 - f (-3) = 636 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l2676_267658


namespace NUMINAMATH_CALUDE_problem_statement_l2676_267601

theorem problem_statement (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h1 : x * (y + z) = 132)
  (h2 : z * (x + y) = 180)
  (h3 : x * y * z = 160) : 
  y * (z + x) = 160 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2676_267601


namespace NUMINAMATH_CALUDE_piggy_bank_value_l2676_267635

-- Define the number of pennies and dimes in one piggy bank
def pennies_per_bank : ℕ := 100
def dimes_per_bank : ℕ := 50

-- Define the value of pennies and dimes in cents
def penny_value : ℕ := 1
def dime_value : ℕ := 10

-- Define the number of piggy banks
def num_banks : ℕ := 2

-- Define the number of cents in a dollar
def cents_per_dollar : ℕ := 100

-- Theorem statement
theorem piggy_bank_value :
  (num_banks * (pennies_per_bank * penny_value + dimes_per_bank * dime_value)) / cents_per_dollar = 12 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_value_l2676_267635


namespace NUMINAMATH_CALUDE_machinery_cost_proof_l2676_267681

def total_amount : ℝ := 250
def raw_materials_cost : ℝ := 100
def cash_percentage : ℝ := 0.1

def machinery_cost : ℝ := total_amount - raw_materials_cost - (cash_percentage * total_amount)

theorem machinery_cost_proof : machinery_cost = 125 := by
  sorry

end NUMINAMATH_CALUDE_machinery_cost_proof_l2676_267681


namespace NUMINAMATH_CALUDE_equivalent_operations_l2676_267659

theorem equivalent_operations (x : ℝ) : 
  (x * (4/5)) / (2/7) = x * (14/5) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_operations_l2676_267659


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2676_267624

/-- Given a square with perimeter 80 units divided into two congruent rectangles
    by a horizontal line, prove that the perimeter of one rectangle is 60 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 80) :
  let square_side := square_perimeter / 4
  let rectangle_width := square_side
  let rectangle_height := square_side / 2
  2 * (rectangle_width + rectangle_height) = 60 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2676_267624


namespace NUMINAMATH_CALUDE_probability_in_pascal_triangle_l2676_267645

/-- The number of rows in Pascal's Triangle we're considering --/
def n : ℕ := 20

/-- The total number of elements in the first n rows of Pascal's Triangle --/
def total_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of ones in the first n rows of Pascal's Triangle --/
def ones_count (n : ℕ) : ℕ := 2 * (n - 1) + 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle --/
def probability_of_one (n : ℕ) : ℚ :=
  (ones_count n : ℚ) / (total_elements n : ℚ)

theorem probability_in_pascal_triangle :
  probability_of_one n = 13 / 70 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_pascal_triangle_l2676_267645


namespace NUMINAMATH_CALUDE_star_is_addition_l2676_267667

/-- A binary operation on real numbers satisfying (a ★ b) ★ c = a + b + c -/
def star_op (star : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, star (star a b) c = a + b + c

theorem star_is_addition (star : ℝ → ℝ → ℝ) (h : star_op star) :
  ∀ a b : ℝ, star a b = a + b :=
sorry

end NUMINAMATH_CALUDE_star_is_addition_l2676_267667


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l2676_267663

/-- If x^2 - kx + 25 is a perfect square polynomial, then k = ±10 -/
theorem perfect_square_polynomial (k : ℝ) : 
  (∃ (a : ℝ), ∀ x, x^2 - k*x + 25 = (x - a)^2) → (k = 10 ∨ k = -10) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l2676_267663


namespace NUMINAMATH_CALUDE_ancient_chinese_carriage_problem_l2676_267648

/-- Ancient Chinese carriage problem -/
theorem ancient_chinese_carriage_problem (x : ℕ) : 
  (∃ (people : ℕ), 
    (3 * (x - 2) = people) ∧ 
    (2 * x + 9 = people) ∧ 
    (x ≥ 2)) → 
  3 * (x - 2) = 2 * x + 9 :=
by sorry

end NUMINAMATH_CALUDE_ancient_chinese_carriage_problem_l2676_267648


namespace NUMINAMATH_CALUDE_reef_age_conversion_l2676_267607

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let ones := octal % 10
  let eights := (octal / 10) % 10
  let sixty_fours := octal / 100
  ones + 8 * eights + 64 * sixty_fours

theorem reef_age_conversion :
  octal_to_decimal 367 = 247 := by
  sorry

end NUMINAMATH_CALUDE_reef_age_conversion_l2676_267607


namespace NUMINAMATH_CALUDE_domain_of_f_half_x_l2676_267608

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(lg x)
def domain_f_lg_x : Set ℝ := { x | 0.1 ≤ x ∧ x ≤ 100 }

-- State the theorem
theorem domain_of_f_half_x (h : ∀ x ∈ domain_f_lg_x, f (Real.log x / Real.log 10) = f (Real.log x / Real.log 10)) :
  { x : ℝ | f (x / 2) = f (x / 2) } = { x : ℝ | -2 ≤ x ∧ x ≤ 4 } :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_half_x_l2676_267608


namespace NUMINAMATH_CALUDE_trapezoid_angle_bisector_area_ratio_l2676_267651

/-- The area ratio of the quadrilateral formed by angle bisector intersections to the trapezoid --/
def area_ratio (a b c d : ℝ) : Set ℝ :=
  {x | x = 1/45 ∨ x = 7/40}

/-- Theorem stating the area ratio for a trapezoid with given side lengths --/
theorem trapezoid_angle_bisector_area_ratio :
  ∀ (a b c d : ℝ),
  a = 5 ∧ b = 15 ∧ c = 15 ∧ d = 20 →
  ∃ (k : ℝ), k ∈ area_ratio a b c d :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_angle_bisector_area_ratio_l2676_267651


namespace NUMINAMATH_CALUDE_currency_notes_total_l2676_267633

/-- Proves that given the specified conditions, the total amount of currency notes is 5000 rupees. -/
theorem currency_notes_total (total_notes : ℕ) (amount_50 : ℕ) : 
  total_notes = 85 →
  amount_50 = 3500 →
  ∃ (notes_100 notes_50 : ℕ),
    notes_100 + notes_50 = total_notes ∧
    50 * notes_50 = amount_50 ∧
    100 * notes_100 + 50 * notes_50 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_currency_notes_total_l2676_267633


namespace NUMINAMATH_CALUDE_z_curve_not_simple_conic_l2676_267611

-- Define the complex number z
variable (z : ℂ)

-- Define the condition |z - 1/z| = 1
def condition (z : ℂ) : Prop := Complex.abs (z - z⁻¹) = 1

-- Define the curves we want to exclude
def is_ellipse (curve : ℂ → Prop) : Prop := sorry
def is_parabola (curve : ℂ → Prop) : Prop := sorry
def is_hyperbola (curve : ℂ → Prop) : Prop := sorry

-- Define the curve traced by z
def z_curve (z : ℂ) : Prop := condition z

-- State the theorem
theorem z_curve_not_simple_conic (z : ℂ) :
  condition z →
  ¬(is_ellipse z_curve ∨ is_parabola z_curve ∨ is_hyperbola z_curve) :=
sorry

end NUMINAMATH_CALUDE_z_curve_not_simple_conic_l2676_267611


namespace NUMINAMATH_CALUDE_solve_for_x_l2676_267628

theorem solve_for_x (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : y = 3) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2676_267628


namespace NUMINAMATH_CALUDE_little_john_money_distribution_l2676_267678

def problem (initial_amount : ℚ) (sweets_cost : ℚ) (num_friends : ℕ) (remaining_amount : ℚ) : Prop :=
  let total_spent : ℚ := initial_amount - remaining_amount
  let amount_given_away : ℚ := total_spent - sweets_cost
  let amount_per_friend : ℚ := amount_given_away / num_friends
  amount_per_friend = 1

theorem little_john_money_distribution :
  problem 7.1 1.05 2 4.05 := by
  sorry

end NUMINAMATH_CALUDE_little_john_money_distribution_l2676_267678


namespace NUMINAMATH_CALUDE_quadratic_function_conditions_l2676_267698

/-- A quadratic function passing through (1,-4) with vertex at (-1,0) -/
def f (x : ℝ) : ℝ := -x^2 - 2*x - 1

/-- Theorem stating that f satisfies the given conditions -/
theorem quadratic_function_conditions :
  (f 1 = -4) ∧ 
  (∃ a : ℝ, ∀ x : ℝ, f x = a * (x + 1)^2) := by
  sorry

#check quadratic_function_conditions

end NUMINAMATH_CALUDE_quadratic_function_conditions_l2676_267698


namespace NUMINAMATH_CALUDE_smallest_solution_comparison_l2676_267602

theorem smallest_solution_comparison (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0) :
  (∃ x y : ℝ, x < y ∧ p * x^2 + q = 0 ∧ p' * y^2 + q' = 0 ∧
    (∀ z : ℝ, p * z^2 + q = 0 → x ≤ z) ∧
    (∀ w : ℝ, p' * w^2 + q' = 0 → y ≤ w)) ↔
  Real.sqrt (q' / p') < Real.sqrt (q / p) :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_comparison_l2676_267602
