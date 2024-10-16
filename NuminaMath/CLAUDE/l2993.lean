import Mathlib

namespace NUMINAMATH_CALUDE_chocolate_bars_bought_l2993_299338

theorem chocolate_bars_bought (bar_cost : ℝ) (paid : ℝ) (max_change : ℝ) :
  bar_cost = 1.35 →
  paid = 10 →
  max_change = 1 →
  ∃ n : ℕ, n * bar_cost ≤ paid ∧
           paid - n * bar_cost < max_change ∧
           ∀ m : ℕ, m > n → m * bar_cost > paid :=
by
  sorry

#check chocolate_bars_bought

end NUMINAMATH_CALUDE_chocolate_bars_bought_l2993_299338


namespace NUMINAMATH_CALUDE_intersection_of_complex_circles_l2993_299374

theorem intersection_of_complex_circles (k : ℝ) :
  (∃! z : ℂ, Complex.abs (z - 3) = 2 * Complex.abs (z + 3) ∧ Complex.abs z = k) →
  k = 1 ∨ k = 9 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_complex_circles_l2993_299374


namespace NUMINAMATH_CALUDE_permutation_equation_solution_combination_equation_solution_l2993_299366

-- Define the factorial function
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the permutation function
def permutation (n k : ℕ) : ℕ := 
  if k ≤ n then factorial n / factorial (n - k) else 0

-- Define the combination function
def combination (n k : ℕ) : ℕ := 
  if k ≤ n then factorial n / (factorial k * factorial (n - k)) else 0

theorem permutation_equation_solution : 
  ∃! x : ℕ, permutation (2 * x) 4 = 60 * permutation x 3 ∧ x > 0 := by sorry

theorem combination_equation_solution : 
  ∃! n : ℕ, combination (n + 3) (n + 1) = 
    combination (n + 1) (n - 1) + combination (n + 1) n + combination n (n - 2) := by sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_combination_equation_solution_l2993_299366


namespace NUMINAMATH_CALUDE_smallest_multiple_one_to_five_l2993_299348

theorem smallest_multiple_one_to_five : ∃ n : ℕ+, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 5 → i ∣ n) ∧ (∀ m : ℕ+, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 5 → i ∣ m) → n ≤ m) ∧ n = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_one_to_five_l2993_299348


namespace NUMINAMATH_CALUDE_eliot_account_balance_l2993_299333

theorem eliot_account_balance 
  (al_balance : ℝ) 
  (eliot_balance : ℝ) 
  (al_more : al_balance > eliot_balance)
  (difference_sum : al_balance - eliot_balance = (1 / 12) * (al_balance + eliot_balance))
  (increased_difference : 1.1 * al_balance = 1.2 * eliot_balance + 20) :
  eliot_balance = 200 := by
sorry

end NUMINAMATH_CALUDE_eliot_account_balance_l2993_299333


namespace NUMINAMATH_CALUDE_function_inequality_l2993_299329

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_ineq : ∀ x : ℝ, f x > deriv f x) : 
  (Real.exp 2016 * f (-2016) > f 0) ∧ (f 2016 < Real.exp 2016 * f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2993_299329


namespace NUMINAMATH_CALUDE_store_a_cheaper_than_b_l2993_299304

/-- Represents the number of tennis rackets to be purchased -/
def num_rackets : ℕ := 30

/-- Represents the price of a tennis racket in yuan -/
def racket_price : ℕ := 100

/-- Represents the price of a can of tennis balls in yuan -/
def ball_price : ℕ := 20

/-- Represents the discount factor for Store B -/
def store_b_discount : ℚ := 9/10

/-- Theorem comparing costs of purchasing from Store A and Store B -/
theorem store_a_cheaper_than_b (x : ℕ) (h : x > num_rackets) :
  (20 : ℚ) * x + 2400 < (18 : ℚ) * x + 2700 ↔ x < 150 := by
  sorry

end NUMINAMATH_CALUDE_store_a_cheaper_than_b_l2993_299304


namespace NUMINAMATH_CALUDE_T_minus_n_is_even_l2993_299382

/-- The number of non-empty subsets with integer average -/
def T (n : ℕ) : ℕ := sorry

/-- Theorem: T_n - n is even for all n > 1 -/
theorem T_minus_n_is_even (n : ℕ) (h : n > 1) : Even (T n - n) := by
  sorry

end NUMINAMATH_CALUDE_T_minus_n_is_even_l2993_299382


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l2993_299321

theorem sine_cosine_inequality (x : ℝ) (n m : ℕ) 
  (h1 : 0 < x) (h2 : x < π / 2) (h3 : n > m) : 
  2 * |Real.sin x ^ n - Real.cos x ^ n| ≤ 3 * |Real.sin x ^ m - Real.cos x ^ m| := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l2993_299321


namespace NUMINAMATH_CALUDE_no_reverse_multiply_all_ones_l2993_299379

/-- Given a natural number, return the number with its digits reversed -/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- Check if a natural number is composed of only ones -/
def all_ones (n : ℕ) : Prop := sorry

theorem no_reverse_multiply_all_ones :
  ∀ n : ℕ, n > 1 → ¬(all_ones (n * reverse_digits n)) := by
  sorry

end NUMINAMATH_CALUDE_no_reverse_multiply_all_ones_l2993_299379


namespace NUMINAMATH_CALUDE_stephanies_age_to_jobs_age_ratio_l2993_299387

/-- Given the ages of Freddy, Stephanie, and Job, prove the ratio of Stephanie's age to Job's age -/
theorem stephanies_age_to_jobs_age_ratio :
  ∀ (freddy_age stephanie_age job_age : ℕ),
  freddy_age = 18 →
  stephanie_age = freddy_age + 2 →
  job_age = 5 →
  (stephanie_age : ℚ) / (job_age : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_stephanies_age_to_jobs_age_ratio_l2993_299387


namespace NUMINAMATH_CALUDE_solve_chocolate_problem_l2993_299305

def chocolate_problem (price_per_bar : ℕ) (total_bars : ℕ) (revenue : ℕ) : Prop :=
  let sold_bars : ℕ := revenue / price_per_bar
  let unsold_bars : ℕ := total_bars - sold_bars
  unsold_bars = 4

theorem solve_chocolate_problem :
  chocolate_problem 3 7 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_chocolate_problem_l2993_299305


namespace NUMINAMATH_CALUDE_min_value_theorem_l2993_299315

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + y = 2) :
  ∃ (min_val : ℝ), min_val = 9/4 ∧ ∀ (z : ℝ), z = 2/(x + 1) + 1/y → z ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2993_299315


namespace NUMINAMATH_CALUDE_sams_remaining_dimes_l2993_299361

/-- Given that Sam initially had 9 dimes and gave 7 dimes away, prove that he now has 2 dimes. -/
theorem sams_remaining_dimes (initial_dimes : ℕ) (dimes_given_away : ℕ) 
  (h1 : initial_dimes = 9)
  (h2 : dimes_given_away = 7) :
  initial_dimes - dimes_given_away = 2 := by
  sorry

end NUMINAMATH_CALUDE_sams_remaining_dimes_l2993_299361


namespace NUMINAMATH_CALUDE_elementary_school_coats_l2993_299314

theorem elementary_school_coats 
  (total_coats : ℕ) 
  (high_school_coats : ℕ) 
  (middle_school_coats : ℕ) 
  (h1 : total_coats = 9437)
  (h2 : high_school_coats = 6922)
  (h3 : middle_school_coats = 1825) :
  total_coats - (high_school_coats + middle_school_coats) = 690 := by
  sorry

end NUMINAMATH_CALUDE_elementary_school_coats_l2993_299314


namespace NUMINAMATH_CALUDE_intersection_equals_two_l2993_299317

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- State the theorem
theorem intersection_equals_two (a : ℝ) :
  A ∩ B a = {2} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_two_l2993_299317


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2993_299346

theorem negation_of_proposition (p : Prop) : 
  (¬(∀ a : ℝ, a > 0 ∧ a ≠ 1 → ∃ x : ℝ, a * x - x - a = 0)) ↔ 
  (∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, a * x - x - a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2993_299346


namespace NUMINAMATH_CALUDE_reinforcement_arrival_theorem_l2993_299380

/-- Represents the number of days after which the reinforcement arrived -/
def reinforcement_arrival_day : ℕ := 20

/-- The size of the initial garrison -/
def initial_garrison : ℕ := 2000

/-- The number of days the initial provisions would last -/
def initial_provision_days : ℕ := 40

/-- The size of the reinforcement -/
def reinforcement_size : ℕ := 2000

/-- The number of days the provisions last after reinforcement arrival -/
def remaining_days : ℕ := 10

theorem reinforcement_arrival_theorem :
  initial_garrison * initial_provision_days =
  initial_garrison * reinforcement_arrival_day +
  (initial_garrison + reinforcement_size) * remaining_days :=
by sorry

end NUMINAMATH_CALUDE_reinforcement_arrival_theorem_l2993_299380


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l2993_299309

-- Define the binary number as a list of bits (0 or 1)
def binary_number : List Nat := [1, 1, 0, 0, 1, 1]

-- Define the function to convert binary to decimal
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

-- Theorem to prove
theorem binary_110011_equals_51 :
  binary_to_decimal binary_number = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l2993_299309


namespace NUMINAMATH_CALUDE_unique_number_exists_l2993_299302

/-- A function that checks if a natural number consists only of digits 2 and 5 -/
def only_2_and_5 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 5

/-- The theorem to be proved -/
theorem unique_number_exists : ∃! n : ℕ,
  only_2_and_5 n ∧
  n.digits 10 = List.replicate 2005 0 ∧
  n % (2^2005) = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_number_exists_l2993_299302


namespace NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l2993_299336

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem distribute_5_balls_4_boxes : distribute_balls 5 4 = 6 := by sorry

end NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l2993_299336


namespace NUMINAMATH_CALUDE_mobile_phone_cost_l2993_299352

def refrigerator_cost : ℝ := 15000
def refrigerator_loss_percent : ℝ := 4
def mobile_profit_percent : ℝ := 9
def overall_profit : ℝ := 120

theorem mobile_phone_cost (mobile_cost : ℝ) : 
  (refrigerator_cost * (1 - refrigerator_loss_percent / 100) + 
   mobile_cost * (1 + mobile_profit_percent / 100)) - 
  (refrigerator_cost + mobile_cost) = overall_profit →
  mobile_cost = 8000 := by
sorry

end NUMINAMATH_CALUDE_mobile_phone_cost_l2993_299352


namespace NUMINAMATH_CALUDE_no_prime_arrangement_with_natural_expression_l2993_299392

theorem no_prime_arrangement_with_natural_expression :
  ¬ ∃ (p : ℕ → ℕ),
    (∀ n, Prime (p n)) ∧
    (∀ q : ℕ, Prime q → ∃ n, p n = q) ∧
    (∀ i : ℕ, ∃ k : ℕ, (p i * p (i + 1) - p (i + 2)^2) / (p i + p (i + 1)) = k) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_arrangement_with_natural_expression_l2993_299392


namespace NUMINAMATH_CALUDE_triangle_area_l2993_299339

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  -- a, b, c are sides opposite to angles A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- √3a = 2c*sin(A)
  Real.sqrt 3 * a = 2 * c * Real.sin A →
  -- c = √7
  c = Real.sqrt 7 →
  -- Perimeter of triangle ABC is 5 + √7
  a + b + c = 5 + Real.sqrt 7 →
  -- Area of triangle ABC is (3√3)/2
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2993_299339


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l2993_299381

def is_prime (n : ℕ) : Prop := sorry

def is_divisible_by (a b : ℕ) : Prop := sorry

theorem least_number_divisible_by_five_primes :
  ∃ (n : ℕ), n > 0 ∧
  (∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧ is_prime p₅ ∧
    is_divisible_by n p₁ ∧ is_divisible_by n p₂ ∧ is_divisible_by n p₃ ∧ 
    is_divisible_by n p₄ ∧ is_divisible_by n p₅) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ (q₁ q₂ q₃ q₄ q₅ : ℕ), 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧
      q₄ ≠ q₅ ∧
      is_prime q₁ ∧ is_prime q₂ ∧ is_prime q₃ ∧ is_prime q₄ ∧ is_prime q₅ ∧
      is_divisible_by m q₁ ∧ is_divisible_by m q₂ ∧ is_divisible_by m q₃ ∧ 
      is_divisible_by m q₄ ∧ is_divisible_by m q₅) → 
    m ≥ n) ∧
  n = 2310 :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l2993_299381


namespace NUMINAMATH_CALUDE_inner_hexagon_area_l2993_299370

/-- Given a hexagon ABCDEF with specific area properties, prove the area of the inner hexagon A₁B₁C₁D₁E₁F₁ -/
theorem inner_hexagon_area 
  (area_ABCDEF : ℝ) 
  (area_triangle : ℝ) 
  (area_shaded : ℝ) 
  (h1 : area_ABCDEF = 2010) 
  (h2 : area_triangle = 335) 
  (h3 : area_shaded = 670) : 
  area_ABCDEF - (6 * area_triangle + area_shaded) / 2 = 670 := by
sorry

end NUMINAMATH_CALUDE_inner_hexagon_area_l2993_299370


namespace NUMINAMATH_CALUDE_merchant_discount_percentage_l2993_299395

theorem merchant_discount_percentage 
  (markup_percentage : ℝ) 
  (profit_percentage : ℝ) 
  (discount_percentage : ℝ) : 
  markup_percentage = 75 → 
  profit_percentage = 5 → 
  discount_percentage = 40 → 
  let cost_price := 100
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := cost_price * (1 + profit_percentage / 100)
  let discount := marked_price - selling_price
  discount / marked_price * 100 = discount_percentage :=
by sorry

end NUMINAMATH_CALUDE_merchant_discount_percentage_l2993_299395


namespace NUMINAMATH_CALUDE_difference_le_two_l2993_299351

/-- Represents a right-angled triangle with integer sides -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h_right_angle : a ^ 2 + b ^ 2 = c ^ 2
  h_ordered : a < b ∧ b < c
  h_coprime : Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c

/-- The difference between the hypotenuse and the middle side -/
def difference (t : RightTriangle) : ℕ := t.c - t.b

/-- Theorem: For a right-angled triangle with integer sides a, b, c where
    a < b < c, a, b, c are pairwise co-prime, and (c - b) divides a,
    then (c - b) ≤ 2 -/
theorem difference_le_two (t : RightTriangle) (h_divides : t.a % (difference t) = 0) :
  difference t ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_difference_le_two_l2993_299351


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l2993_299355

/-- A color represented as a natural number -/
def Color := ℕ

/-- A point in the grid -/
structure GridPoint where
  x : ℕ
  y : ℕ
  h_x : x ≤ 12
  h_y : y ≤ 12

/-- A coloring of the grid -/
def GridColoring := GridPoint → Color

/-- A rectangle in the grid -/
structure Rectangle where
  x1 : ℕ
  y1 : ℕ
  x2 : ℕ
  y2 : ℕ
  h_x1 : x1 ≤ 12
  h_y1 : y1 ≤ 12
  h_x2 : x2 ≤ 12
  h_y2 : y2 ≤ 12
  h_distinct : (x1 ≠ x2 ∧ y1 ≠ y2) ∨ (x1 ≠ x2 ∧ y1 = y2) ∨ (x1 = x2 ∧ y1 ≠ y2)

/-- The theorem stating that there exists a monochromatic rectangle -/
theorem monochromatic_rectangle_exists (coloring : GridColoring) :
  ∃ (r : Rectangle) (c : Color),
    coloring ⟨r.x1, r.y1, r.h_x1, r.h_y1⟩ = c ∧
    coloring ⟨r.x1, r.y2, r.h_x1, r.h_y2⟩ = c ∧
    coloring ⟨r.x2, r.y1, r.h_x2, r.h_y1⟩ = c ∧
    coloring ⟨r.x2, r.y2, r.h_x2, r.h_y2⟩ = c :=
  sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l2993_299355


namespace NUMINAMATH_CALUDE_equation_roots_problem_l2993_299358

theorem equation_roots_problem (p q : ℝ) 
  (eq1 : ∃ x1 x2 : ℝ, x1^2 - p*x1 + 4 = 0 ∧ x2^2 - p*x2 + 4 = 0 ∧ x1 ≠ x2)
  (eq2 : ∃ x3 x4 : ℝ, 2*x3^2 - 9*x3 + q = 0 ∧ 2*x4^2 - 9*x4 + q = 0 ∧ x3 ≠ x4)
  (root_relation : ∃ x1 x2 x3 x4 : ℝ, 
    x1^2 - p*x1 + 4 = 0 ∧ x2^2 - p*x2 + 4 = 0 ∧ x1 < x2 ∧
    2*x3^2 - 9*x3 + q = 0 ∧ 2*x4^2 - 9*x4 + q = 0 ∧
    ((x3 = x2 + 2 ∧ x4 = x1 - 2) ∨ (x4 = x2 + 2 ∧ x3 = x1 - 2))) :
  q = -2 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_problem_l2993_299358


namespace NUMINAMATH_CALUDE_smaller_ladder_steps_l2993_299357

theorem smaller_ladder_steps 
  (full_ladder_steps : ℕ) 
  (full_ladder_climbs : ℕ) 
  (smaller_ladder_climbs : ℕ) 
  (total_steps : ℕ) 
  (h1 : full_ladder_steps = 11)
  (h2 : full_ladder_climbs = 10)
  (h3 : smaller_ladder_climbs = 7)
  (h4 : total_steps = 152)
  (h5 : full_ladder_steps * full_ladder_climbs + smaller_ladder_climbs * x = total_steps) :
  x = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_smaller_ladder_steps_l2993_299357


namespace NUMINAMATH_CALUDE_pawns_left_l2993_299372

/-- The number of pawns each player starts with in a standard chess game -/
def standard_pawns : ℕ := 8

/-- The number of pawns Sophia has lost -/
def sophia_lost : ℕ := 5

/-- The number of pawns Chloe has lost -/
def chloe_lost : ℕ := 1

/-- Theorem: The total number of pawns left in the game is 10 -/
theorem pawns_left : 
  (standard_pawns - sophia_lost) + (standard_pawns - chloe_lost) = 10 := by
  sorry

end NUMINAMATH_CALUDE_pawns_left_l2993_299372


namespace NUMINAMATH_CALUDE_mary_peaches_cost_l2993_299378

/-- The amount Mary paid for berries in dollars -/
def berries_cost : ℚ := 7.19

/-- The total amount Mary paid with in dollars -/
def total_paid : ℚ := 20

/-- The amount Mary received as change in dollars -/
def change_received : ℚ := 5.98

/-- The amount Mary paid for peaches in dollars -/
def peaches_cost : ℚ := total_paid - change_received - berries_cost

theorem mary_peaches_cost : peaches_cost = 6.83 := by sorry

end NUMINAMATH_CALUDE_mary_peaches_cost_l2993_299378


namespace NUMINAMATH_CALUDE_multiply_98_98_l2993_299326

theorem multiply_98_98 : 98 * 98 = 9604 := by
  sorry

end NUMINAMATH_CALUDE_multiply_98_98_l2993_299326


namespace NUMINAMATH_CALUDE_modified_array_sum_for_five_l2993_299383

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The modified 1/p-array sum -/
def modifiedArraySum (p : ℕ) : ℚ :=
  (3 * p^2) / ((9 * p^2 - 12 * p + 4) * (p - 1))

theorem modified_array_sum_for_five :
  modifiedArraySum 5 = 75 / 676 := by sorry

end NUMINAMATH_CALUDE_modified_array_sum_for_five_l2993_299383


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2993_299390

/-- An isosceles triangle with side lengths 6 and 9 -/
structure IsoscelesTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (is_isosceles : side1 = side2 ∨ side1 = 9 ∨ side2 = 9)
  (has_length_6 : side1 = 6 ∨ side2 = 6)
  (has_length_9 : side1 = 9 ∨ side2 = 9)

/-- The perimeter of an isosceles triangle with side lengths 6 and 9 is either 21 or 24 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : 
  t.side1 + t.side2 + (if t.side1 = t.side2 then t.side1 else 
    (if t.side1 = 9 ∨ t.side2 = 9 then 9 else 6)) = 21 ∨ 
  t.side1 + t.side2 + (if t.side1 = t.side2 then t.side1 else 
    (if t.side1 = 9 ∨ t.side2 = 9 then 9 else 6)) = 24 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2993_299390


namespace NUMINAMATH_CALUDE_water_in_final_mixture_l2993_299327

/-- Given a mixture where x liters of 10% acid solution is added to 5 liters of pure acid,
    resulting in a final mixture that is 40% water, prove that the amount of water
    in the final mixture is 3.6 liters. -/
theorem water_in_final_mixture :
  ∀ x : ℝ,
  x > 0 →
  0.4 * (5 + x) = 0.9 * x →
  0.9 * x = 3.6 :=
by
  sorry

end NUMINAMATH_CALUDE_water_in_final_mixture_l2993_299327


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2993_299323

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Proof that the sum of the given arithmetic sequence is 2772 -/
theorem arithmetic_sequence_sum :
  arithmetic_sum 7 147 4 = 2772 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2993_299323


namespace NUMINAMATH_CALUDE_two_numbers_product_sum_l2993_299344

theorem two_numbers_product_sum (x y : ℕ) : 
  x ∈ Finset.range 38 ∧ 
  y ∈ Finset.range 38 ∧ 
  x ≠ y ∧
  (Finset.sum (Finset.range 38) id) - x - y = x * y ∧
  y - x = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_product_sum_l2993_299344


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l2993_299300

theorem circle_diameter_from_area : 
  ∀ (A d : ℝ), A = 78.53981633974483 → d = 10 → A = π * (d / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l2993_299300


namespace NUMINAMATH_CALUDE_line_through_point_l2993_299345

theorem line_through_point (k : ℚ) : 
  (2 * k * 3 - 5 = 4 * (-4)) → k = -11/6 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2993_299345


namespace NUMINAMATH_CALUDE_production_plan_equation_l2993_299330

/-- Represents a factory's production plan -/
structure ProductionPlan where
  original_days : ℕ
  original_parts_per_day : ℕ
  new_days : ℕ
  additional_parts_per_day : ℕ
  extra_parts : ℕ

/-- The equation holds for the given production plan -/
def equation_holds (plan : ProductionPlan) : Prop :=
  plan.original_days * plan.original_parts_per_day = 
  plan.new_days * (plan.original_parts_per_day + plan.additional_parts_per_day) - plan.extra_parts

theorem production_plan_equation (plan : ProductionPlan) 
  (h1 : plan.original_days = 20)
  (h2 : plan.new_days = 15)
  (h3 : plan.additional_parts_per_day = 4)
  (h4 : plan.extra_parts = 10) :
  equation_holds plan := by
  sorry

#check production_plan_equation

end NUMINAMATH_CALUDE_production_plan_equation_l2993_299330


namespace NUMINAMATH_CALUDE_probability_one_tail_given_at_least_one_head_l2993_299335

def fair_coin_toss (n : ℕ) := 1 / 2 ^ n

def probability_at_least_one_head := 1 - fair_coin_toss 3

def probability_exactly_one_tail := 3 * (1 / 2) * (1 / 2)^2

theorem probability_one_tail_given_at_least_one_head :
  probability_exactly_one_tail / probability_at_least_one_head = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_tail_given_at_least_one_head_l2993_299335


namespace NUMINAMATH_CALUDE_club_member_ratio_l2993_299371

/-- 
Given a club with current members and additional members,
prove that the ratio of new total members to current members is 5:2.
-/
theorem club_member_ratio (current_members additional_members : ℕ) 
  (h1 : current_members = 10)
  (h2 : additional_members = 15) : 
  (current_members + additional_members) / current_members = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_club_member_ratio_l2993_299371


namespace NUMINAMATH_CALUDE_melissa_driving_hours_l2993_299306

/-- Calculates the total hours Melissa spends driving in a year -/
def total_driving_hours (trips_per_month : ℕ) (hours_per_trip : ℕ) (months_per_year : ℕ) : ℕ :=
  trips_per_month * hours_per_trip * months_per_year

/-- Proves that Melissa spends 72 hours driving in a year -/
theorem melissa_driving_hours :
  total_driving_hours 2 3 12 = 72 := by
  sorry

end NUMINAMATH_CALUDE_melissa_driving_hours_l2993_299306


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l2993_299399

theorem arctan_equation_solution (x : ℝ) :
  2 * Real.arctan (1/3) + Real.arctan (1/15) + Real.arctan (1/x) = π/4 → x = 53/4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l2993_299399


namespace NUMINAMATH_CALUDE_max_roses_for_1000_budget_l2993_299393

/-- Represents the price of roses for different quantities -/
structure RosePrices where
  individual : ℚ
  dozen : ℚ
  two_dozen : ℚ
  five_dozen : ℚ
  hundred : ℚ

/-- Calculates the maximum number of roses that can be purchased with a given budget -/
def maxRoses (prices : RosePrices) (budget : ℚ) : ℕ :=
  sorry

/-- The theorem stating that given the specific rose prices and a $1000 budget, 
    the maximum number of roses that can be purchased is 548 -/
theorem max_roses_for_1000_budget :
  let prices : RosePrices := {
    individual := 5.3,
    dozen := 36,
    two_dozen := 50,
    five_dozen := 110,
    hundred := 180
  }
  maxRoses prices 1000 = 548 := by
  sorry

end NUMINAMATH_CALUDE_max_roses_for_1000_budget_l2993_299393


namespace NUMINAMATH_CALUDE_chris_money_before_birthday_l2993_299342

/-- The amount of money Chris had before his birthday -/
def money_before_birthday : ℕ := sorry

/-- The amount Chris received from his grandmother -/
def grandmother_gift : ℕ := 25

/-- The amount Chris received from his aunt and uncle -/
def aunt_uncle_gift : ℕ := 20

/-- The amount Chris received from his parents -/
def parents_gift : ℕ := 75

/-- The total amount Chris has now -/
def total_money_now : ℕ := 279

/-- Theorem stating that Chris had $159 before his birthday -/
theorem chris_money_before_birthday :
  money_before_birthday = 159 :=
by sorry

end NUMINAMATH_CALUDE_chris_money_before_birthday_l2993_299342


namespace NUMINAMATH_CALUDE_no_zero_root_l2993_299376

theorem no_zero_root : 
  (∀ x : ℝ, 4 * x^2 - 3 = 49 → x ≠ 0) ∧
  (∀ x : ℝ, (3*x - 2)^2 = (x + 2)^2 → x ≠ 0) ∧
  (∀ x : ℝ, x^2 - x - 20 = 0 → x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_zero_root_l2993_299376


namespace NUMINAMATH_CALUDE_percentage_calculation_l2993_299319

theorem percentage_calculation (n : ℝ) (h : n = 5600) : 0.15 * (0.30 * (0.50 * n)) = 126 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2993_299319


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l2993_299340

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l2993_299340


namespace NUMINAMATH_CALUDE_average_fish_is_75_l2993_299308

/-- The number of fish in Boast Pool -/
def boast_pool : ℕ := 75

/-- The number of fish in Onum Lake -/
def onum_lake : ℕ := boast_pool + 25

/-- The number of fish in Riddle Pond -/
def riddle_pond : ℕ := onum_lake / 2

/-- The total number of fish in all three bodies of water -/
def total_fish : ℕ := boast_pool + onum_lake + riddle_pond

/-- The number of bodies of water -/
def num_bodies : ℕ := 3

/-- Theorem stating that the average number of fish in all three bodies of water is 75 -/
theorem average_fish_is_75 : total_fish / num_bodies = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_fish_is_75_l2993_299308


namespace NUMINAMATH_CALUDE_unique_prime_squared_plus_eleven_with_six_divisors_l2993_299364

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n has exactly 6 positive divisors -/
def has_six_divisors (n : ℕ) : Prop := num_divisors n = 6

theorem unique_prime_squared_plus_eleven_with_six_divisors :
  ∃! p : ℕ, Nat.Prime p ∧ has_six_divisors (p^2 + 11) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_squared_plus_eleven_with_six_divisors_l2993_299364


namespace NUMINAMATH_CALUDE_big_boxes_count_l2993_299389

theorem big_boxes_count (dolls_per_big_box : ℕ) (dolls_per_small_box : ℕ) 
  (small_box_count : ℕ) (total_dolls : ℕ) (h1 : dolls_per_big_box = 7) 
  (h2 : dolls_per_small_box = 4) (h3 : small_box_count = 9) (h4 : total_dolls = 71) :
  ∃ (big_box_count : ℕ), big_box_count * dolls_per_big_box + 
    small_box_count * dolls_per_small_box = total_dolls ∧ big_box_count = 5 :=
by sorry

end NUMINAMATH_CALUDE_big_boxes_count_l2993_299389


namespace NUMINAMATH_CALUDE_larger_integer_problem_l2993_299396

theorem larger_integer_problem (a b : ℕ+) : 
  (b : ℚ) / (a : ℚ) = 7 / 3 → 
  (a : ℕ) * b = 189 → 
  b = 21 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l2993_299396


namespace NUMINAMATH_CALUDE_project_selection_count_l2993_299368

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to select 3 additional projects --/
def selectProjects : ℕ :=
  choose 4 1 * choose 6 1 * choose 4 1 +
  choose 6 2 * choose 4 1 +
  choose 6 1 * choose 4 2

theorem project_selection_count :
  selectProjects = 192 := by sorry

end NUMINAMATH_CALUDE_project_selection_count_l2993_299368


namespace NUMINAMATH_CALUDE_paintable_area_four_bedrooms_l2993_299373

theorem paintable_area_four_bedrooms 
  (length : ℝ) (width : ℝ) (height : ℝ) (unpaintable_area : ℝ) (num_bedrooms : ℕ) :
  length = 15 →
  width = 11 →
  height = 9 →
  unpaintable_area = 80 →
  num_bedrooms = 4 →
  (2 * (length * height + width * height) - unpaintable_area) * num_bedrooms = 1552 := by
  sorry

end NUMINAMATH_CALUDE_paintable_area_four_bedrooms_l2993_299373


namespace NUMINAMATH_CALUDE_valid_liar_counts_l2993_299310

/-- Represents the number of people in the room -/
def total_people : ℕ := 30

/-- Represents the possible numbers of liars in the room -/
def possible_liar_counts : List ℕ := [2, 3, 5, 6, 10, 15, 30]

/-- Predicate to check if a number is a valid liar count -/
def is_valid_liar_count (x : ℕ) : Prop :=
  x > 1 ∧ (total_people % x = 0) ∧
  ∃ (n : ℕ), (n + 1) * x = total_people

/-- Theorem stating that the possible_liar_counts are the only valid liar counts -/
theorem valid_liar_counts :
  ∀ (x : ℕ), is_valid_liar_count x ↔ x ∈ possible_liar_counts :=
by sorry

end NUMINAMATH_CALUDE_valid_liar_counts_l2993_299310


namespace NUMINAMATH_CALUDE_solve_for_x_l2993_299385

theorem solve_for_x (x y : ℚ) (h1 : x / y = 5 / 2) (h2 : y = 30) : x = 75 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2993_299385


namespace NUMINAMATH_CALUDE_exam_students_count_l2993_299303

/-- The total number of students in an examination -/
def total_students : ℕ := 400

/-- The number of students who failed the examination -/
def failed_students : ℕ := 260

/-- The percentage of students who passed the examination -/
def pass_percentage : ℚ := 35 / 100

theorem exam_students_count :
  (1 - pass_percentage) * total_students = failed_students :=
sorry

end NUMINAMATH_CALUDE_exam_students_count_l2993_299303


namespace NUMINAMATH_CALUDE_system_a_solutions_system_b_solutions_l2993_299362

-- Part (a)
theorem system_a_solutions (x y z : ℝ) : 
  (2 * x = (y + z)^2 ∧ 2 * y = (z + x)^2 ∧ 2 * z = (x + y)^2) → 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2)) := by
  sorry

-- Part (b)
theorem system_b_solutions (x y z : ℝ) :
  (x^2 - x*y - x*z + z^2 = 0 ∧ 
   x^2 - x*z - y*z + 3*y^2 = 2 ∧ 
   y^2 + x*y + y*z - z^2 = 2) → 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 ∧ y = -1 ∧ z = -1)) := by
  sorry

end NUMINAMATH_CALUDE_system_a_solutions_system_b_solutions_l2993_299362


namespace NUMINAMATH_CALUDE_g_of_8_l2993_299350

theorem g_of_8 (g : ℝ → ℝ) (h : ∀ x, x ≠ 2 → g x = (7 * x + 3) / (x - 2)) :
  g 8 = 59 / 6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_8_l2993_299350


namespace NUMINAMATH_CALUDE_three_intersections_iff_a_in_open_interval_l2993_299377

/-- The function f(x) = x^3 - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The proposition that the line y = a intersects the graph of f(x) at three distinct points -/
def has_three_distinct_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ = a ∧ f x₂ = a ∧ f x₃ = a

/-- The theorem stating that the line y = a intersects the graph of f(x) at three distinct points
    if and only if a is in the open interval (-2, 2) -/
theorem three_intersections_iff_a_in_open_interval :
  ∀ a : ℝ, has_three_distinct_intersections a ↔ -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_three_intersections_iff_a_in_open_interval_l2993_299377


namespace NUMINAMATH_CALUDE_not_enough_ribbons_l2993_299375

/-- Represents the number of ribbons needed for a gift --/
structure RibbonRequirement where
  typeA : ℕ
  typeB : ℕ

/-- Represents the available ribbon supply --/
structure RibbonSupply where
  typeA : ℤ
  typeB : ℤ

def gift_count : ℕ := 8

def initial_supply : RibbonSupply := ⟨10, 12⟩

def requirement_gifts_1_to_4 : RibbonRequirement := ⟨2, 1⟩
def requirement_gifts_5_to_8 : RibbonRequirement := ⟨1, 3⟩

def total_ribbons_needed (req1 req2 : RibbonRequirement) : RibbonRequirement :=
  ⟨req1.typeA * 4 + req2.typeA * 4, req1.typeB * 4 + req2.typeB * 4⟩

def remaining_ribbons (supply : RibbonSupply) (needed : RibbonRequirement) : RibbonSupply :=
  ⟨supply.typeA - needed.typeA, supply.typeB - needed.typeB⟩

theorem not_enough_ribbons :
  let total_needed := total_ribbons_needed requirement_gifts_1_to_4 requirement_gifts_5_to_8
  let remaining := remaining_ribbons initial_supply total_needed
  remaining.typeA < 0 ∧ remaining.typeB < 0 ∧
  remaining.typeA = -2 ∧ remaining.typeB = -4 :=
by sorry

#check not_enough_ribbons

end NUMINAMATH_CALUDE_not_enough_ribbons_l2993_299375


namespace NUMINAMATH_CALUDE_color_assignment_count_l2993_299322

theorem color_assignment_count : ∀ (n m : ℕ), n = 5 ∧ m = 3 →
  (n * (n - 1) * (n - 2)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_color_assignment_count_l2993_299322


namespace NUMINAMATH_CALUDE_integers_less_than_four_abs_l2993_299316

theorem integers_less_than_four_abs : 
  {n : ℤ | |n| < 4} = {-3, -2, -1, 0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_integers_less_than_four_abs_l2993_299316


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l2993_299356

theorem semicircle_area_with_inscribed_rectangle (r : ℝ) :
  r > 0 →
  r * r = 2 →
  π * r * r = π :=
by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l2993_299356


namespace NUMINAMATH_CALUDE_triangle_inequality_l2993_299369

theorem triangle_inequality (A B C : ℝ) (h : A + B + C = π) :
  Real.tan (B / 2) * Real.tan (C / 2) ≤ ((1 - Real.sin (A / 2)) / Real.cos (A / 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2993_299369


namespace NUMINAMATH_CALUDE_smallest_number_l2993_299307

theorem smallest_number (S : Set ℤ) (hS : S = {0, 1, -5, -1}) :
  ∃ m ∈ S, ∀ x ∈ S, m ≤ x ∧ m = -5 := by sorry

end NUMINAMATH_CALUDE_smallest_number_l2993_299307


namespace NUMINAMATH_CALUDE_batsman_average_l2993_299332

theorem batsman_average (total_innings : ℕ) (last_score : ℕ) (average_increase : ℝ) : 
  total_innings = 20 →
  last_score = 90 →
  average_increase = 2 →
  (↑total_innings * (average_after_last_innings - average_increase) + ↑last_score) / ↑total_innings = average_after_last_innings →
  average_after_last_innings = 52 :=
by
  sorry

#check batsman_average

end NUMINAMATH_CALUDE_batsman_average_l2993_299332


namespace NUMINAMATH_CALUDE_sams_sitting_fee_is_correct_l2993_299343

/-- The one-time sitting fee for Sam's Picture Emporium -/
def sams_sitting_fee : ℝ := 140

/-- The price per sheet for John's Photo World -/
def johns_price_per_sheet : ℝ := 2.75

/-- The one-time sitting fee for John's Photo World -/
def johns_sitting_fee : ℝ := 125

/-- The price per sheet for Sam's Picture Emporium -/
def sams_price_per_sheet : ℝ := 1.50

/-- The number of sheets for which the total price is the same -/
def num_sheets : ℕ := 12

theorem sams_sitting_fee_is_correct :
  johns_price_per_sheet * num_sheets + johns_sitting_fee =
  sams_price_per_sheet * num_sheets + sams_sitting_fee :=
by
  sorry

#check sams_sitting_fee_is_correct

end NUMINAMATH_CALUDE_sams_sitting_fee_is_correct_l2993_299343


namespace NUMINAMATH_CALUDE_sum_of_digits_1197_l2993_299324

/-- Given a natural number, returns the sum of its digits -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The sum of digits of 1197 is 18 -/
theorem sum_of_digits_1197 : sum_of_digits 1197 = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_1197_l2993_299324


namespace NUMINAMATH_CALUDE_alternating_color_probability_alternating_color_probability_proof_l2993_299398

/-- The probability of drawing 10 balls with alternating colors from a box containing 5 white and 5 black balls -/
theorem alternating_color_probability : ℚ :=
  let total_balls : ℕ := 10
  let white_balls : ℕ := 5
  let black_balls : ℕ := 5
  let total_arrangements : ℕ := Nat.choose total_balls white_balls
  let successful_arrangements : ℕ := 2
  1 / 126

/-- Proof that the probability of drawing 10 balls with alternating colors from a box containing 5 white and 5 black balls is 1/126 -/
theorem alternating_color_probability_proof :
  alternating_color_probability = 1 / 126 := by
  sorry

end NUMINAMATH_CALUDE_alternating_color_probability_alternating_color_probability_proof_l2993_299398


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l2993_299341

/-- The function representing the curve y = x^3 + 2x - 1 -/
def f (x : ℝ) : ℝ := x^3 + 2*x - 1

/-- The derivative of the function f -/
def f' (x : ℝ) : ℝ := 3*x^2 + 2

/-- The point P on the curve -/
def P : ℝ × ℝ := (1, 2)

/-- The slope of the tangent line at point P -/
def k : ℝ := f' P.1

/-- The y-intercept of the tangent line -/
def b : ℝ := P.2 - k * P.1

theorem tangent_line_y_intercept :
  b = -3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l2993_299341


namespace NUMINAMATH_CALUDE_jeremy_age_l2993_299360

theorem jeremy_age (total_age : ℕ) (amy_age : ℚ) (chris_age : ℚ) (jeremy_age : ℚ) : 
  total_age = 132 →
  amy_age = (1 : ℚ) / 3 * jeremy_age →
  chris_age = 2 * amy_age →
  jeremy_age + amy_age + chris_age = total_age →
  jeremy_age = 66 :=
by sorry

end NUMINAMATH_CALUDE_jeremy_age_l2993_299360


namespace NUMINAMATH_CALUDE_apples_processed_equals_stems_l2993_299337

/-- A machine that processes apples and cuts stems -/
structure AppleProcessor where
  stems_after_2_hours : ℕ
  apples_processed : ℕ

/-- The number of stems after 2 hours is equal to the number of apples processed -/
axiom stems_equal_apples (m : AppleProcessor) : m.stems_after_2_hours = m.apples_processed

/-- Theorem: The number of apples processed is equal to the number of stems observed after 2 hours -/
theorem apples_processed_equals_stems (m : AppleProcessor) :
  m.apples_processed = m.stems_after_2_hours := by sorry

end NUMINAMATH_CALUDE_apples_processed_equals_stems_l2993_299337


namespace NUMINAMATH_CALUDE_french_students_count_l2993_299384

theorem french_students_count (total : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) (french : ℕ) : 
  total = 60 →
  german = 22 →
  both = 9 →
  neither = 6 →
  french + german - both = total - neither →
  french = 41 :=
by sorry

end NUMINAMATH_CALUDE_french_students_count_l2993_299384


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2993_299365

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2993_299365


namespace NUMINAMATH_CALUDE_no_real_roots_range_l2993_299353

theorem no_real_roots_range (p q : ℝ) : 
  (∀ x : ℝ, x^2 + 2*p*x - (q^2 - 2) ≠ 0) → p + q ∈ Set.Ioo (-2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_range_l2993_299353


namespace NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l2993_299347

theorem cubic_equation_integer_solutions :
  ∀ x y : ℤ, y^3 = x^3 + 8*x^2 - 6*x + 8 ↔ (x = 0 ∧ y = 2) ∨ (x = 9 ∧ y = 11) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l2993_299347


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2993_299318

theorem triangle_angle_calculation (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 2 →
  b = Real.sqrt 3 →
  B = π / 3 →  -- 60° in radians
  (a / Real.sin A = b / Real.sin B) →  -- Law of Sines
  A = π / 4  -- 45° in radians
:= by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2993_299318


namespace NUMINAMATH_CALUDE_min_value_cubic_expression_l2993_299354

theorem min_value_cubic_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x^3 + y^3 - 5*x*y ≥ -125/27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cubic_expression_l2993_299354


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l2993_299331

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, 7*x*y - 13*x + 15*y - 37 = 0 ↔ 
    ((x = -2 ∧ y = 11) ∨ (x = -1 ∧ y = 3) ∨ (x = 7 ∧ y = 2)) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l2993_299331


namespace NUMINAMATH_CALUDE_at_least_five_roots_l2993_299313

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the period T
variable (T : ℝ)

-- Assumptions
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_periodic : ∀ x, f (x + T) = f x)
variable (h_T_pos : T > 0)

-- Theorem statement
theorem at_least_five_roots :
  ∃ (x₁ x₂ x₃ x₄ x₅ : ℝ), 
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
     x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
     x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
     x₄ ≠ x₅) ∧
    (x₁ ∈ Set.Icc (-T) T ∧
     x₂ ∈ Set.Icc (-T) T ∧
     x₃ ∈ Set.Icc (-T) T ∧
     x₄ ∈ Set.Icc (-T) T ∧
     x₅ ∈ Set.Icc (-T) T) ∧
    (f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0) :=
by sorry

end NUMINAMATH_CALUDE_at_least_five_roots_l2993_299313


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l2993_299312

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Nat) (y : Nat)

/-- Represents the chessboard game -/
structure ChessboardGame :=
  (m : Nat) (n : Nat)

/-- Checks if a position is winning for the current player -/
def isWinningPosition (game : ChessboardGame) (pos : Position) : Prop :=
  pos.x ≠ pos.y

/-- Checks if the first player has a winning strategy -/
def firstPlayerWins (game : ChessboardGame) : Prop :=
  isWinningPosition game ⟨game.m - 1, game.n - 1⟩

/-- The main theorem: The first player wins iff m ≠ n -/
theorem first_player_winning_strategy (game : ChessboardGame) :
  firstPlayerWins game ↔ game.m ≠ game.n :=
sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l2993_299312


namespace NUMINAMATH_CALUDE_vegetables_amount_l2993_299363

def beef_initial : ℕ := 4
def beef_unused : ℕ := 1

def beef_used (initial unused : ℕ) : ℕ := initial - unused

def vegetables_used (beef : ℕ) : ℕ := 2 * beef

theorem vegetables_amount : vegetables_used (beef_used beef_initial beef_unused) = 6 := by
  sorry

end NUMINAMATH_CALUDE_vegetables_amount_l2993_299363


namespace NUMINAMATH_CALUDE_line_through_point_l2993_299311

theorem line_through_point (b : ℝ) : 
  (∀ x y : ℝ, b * x + (b + 2) * y = b - 1 → x = 3 ∧ y = -5) → b = -3 :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_l2993_299311


namespace NUMINAMATH_CALUDE_complex_number_location_l2993_299388

theorem complex_number_location (z : ℂ) (h : (2 + 3*I)*z = 1 + I) :
  (z.re > 0) ∧ (z.im < 0) :=
sorry

end NUMINAMATH_CALUDE_complex_number_location_l2993_299388


namespace NUMINAMATH_CALUDE_trig_identity_l2993_299328

theorem trig_identity (α : Real) (h : Real.sin (π / 8 + α) = 3 / 4) :
  Real.cos (3 * π / 8 - α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2993_299328


namespace NUMINAMATH_CALUDE_figurine_arrangement_l2993_299325

/-- The number of ways to arrange n uniquely sized figurines in a line -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n uniquely sized figurines in a line,
    with two specific figurines at opposite ends -/
def arrangementsWithEndsFixed (n : ℕ) : ℕ := 2 * arrangements (n - 2)

theorem figurine_arrangement :
  arrangementsWithEndsFixed 9 = 10080 := by
  sorry

end NUMINAMATH_CALUDE_figurine_arrangement_l2993_299325


namespace NUMINAMATH_CALUDE_coloring_books_problem_l2993_299386

theorem coloring_books_problem (initial_books : ℝ) 
  (first_giveaway : ℝ) (second_giveaway : ℝ) (remaining_books : ℝ) : 
  first_giveaway = 34.0 →
  second_giveaway = 3.0 →
  remaining_books = 11 →
  initial_books = first_giveaway + second_giveaway + remaining_books →
  initial_books = 48.0 := by
sorry

end NUMINAMATH_CALUDE_coloring_books_problem_l2993_299386


namespace NUMINAMATH_CALUDE_red_balls_count_l2993_299349

theorem red_balls_count (total_balls : ℕ) (black_balls : ℕ) (prob_black : ℚ) : 
  black_balls = 5 → 
  prob_black = 1/4 → 
  total_balls = black_balls + (total_balls - black_balls) →
  (total_balls - black_balls) = 15 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2993_299349


namespace NUMINAMATH_CALUDE_product_312_57_base7_units_digit_l2993_299334

theorem product_312_57_base7_units_digit : 
  (312 * 57) % 7 = 4 := by sorry

end NUMINAMATH_CALUDE_product_312_57_base7_units_digit_l2993_299334


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l2993_299359

theorem square_area_from_perimeter (perimeter : ℝ) (h : perimeter = 52) :
  let side_length := perimeter / 4
  let area := side_length ^ 2
  area = 169 := by sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l2993_299359


namespace NUMINAMATH_CALUDE_wire_ratio_square_octagon_l2993_299391

/-- The ratio of wire lengths for equal-area square and octagon -/
theorem wire_ratio_square_octagon (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a / 4)^2 = (1 + Real.sqrt 2) * (b / 8)^2 → a / b = Real.sqrt (2 * (1 + Real.sqrt 2)) / 2 := by
  sorry

#check wire_ratio_square_octagon

end NUMINAMATH_CALUDE_wire_ratio_square_octagon_l2993_299391


namespace NUMINAMATH_CALUDE_power_product_equality_l2993_299394

theorem power_product_equality (a b : ℕ) (h1 : a = 7^5) (h2 : b = 5^7) : a^7 * b^5 = 35^35 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2993_299394


namespace NUMINAMATH_CALUDE_balloon_count_l2993_299320

theorem balloon_count (my_balloons : ℕ) (friend_balloons : ℕ) 
  (h1 : friend_balloons = 5)
  (h2 : my_balloons - friend_balloons = 2) : 
  my_balloons = 7 := by
sorry

end NUMINAMATH_CALUDE_balloon_count_l2993_299320


namespace NUMINAMATH_CALUDE_inez_remaining_money_l2993_299367

def initial_amount : ℕ := 150
def pad_cost : ℕ := 50

theorem inez_remaining_money :
  let skate_cost : ℕ := initial_amount / 2
  let after_skates : ℕ := initial_amount - skate_cost
  let remaining : ℕ := after_skates - pad_cost
  remaining = 25 := by sorry

end NUMINAMATH_CALUDE_inez_remaining_money_l2993_299367


namespace NUMINAMATH_CALUDE_sum_a_b_equals_negative_two_l2993_299397

theorem sum_a_b_equals_negative_two (a b : ℝ) :
  |a - 1| + (b + 3)^2 = 0 → a + b = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_negative_two_l2993_299397


namespace NUMINAMATH_CALUDE_area_trace_proportionality_specific_area_trace_l2993_299301

/-- Given two concentric spheres and a smaller sphere tracing areas on both, 
    the areas traced are proportional to the square of the radii ratio. -/
theorem area_trace_proportionality 
  (R1 R2 r A1 : ℝ) 
  (h1 : 0 < r) 
  (h2 : r < R1) 
  (h3 : R1 < R2) 
  (h4 : 0 < A1) : 
  ∃ A2 : ℝ, A2 = A1 * (R2 / R1)^2 := by
  sorry

/-- The specific case with given values -/
theorem specific_area_trace 
  (R1 R2 r A1 : ℝ) 
  (h1 : r = 1) 
  (h2 : R1 = 4) 
  (h3 : R2 = 6) 
  (h4 : A1 = 17) : 
  ∃ A2 : ℝ, A2 = 38.25 := by
  sorry

end NUMINAMATH_CALUDE_area_trace_proportionality_specific_area_trace_l2993_299301
