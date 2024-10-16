import Mathlib

namespace NUMINAMATH_CALUDE_tan_equality_proof_l3335_333571

theorem tan_equality_proof (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1230 * π / 180) → 
  n = -30 ∨ n = 150 := by sorry

end NUMINAMATH_CALUDE_tan_equality_proof_l3335_333571


namespace NUMINAMATH_CALUDE_compound_interest_rate_l3335_333540

theorem compound_interest_rate (P : ℝ) (r : ℝ) 
  (h1 : P * (1 + r)^2 = 2420)
  (h2 : P * (1 + r)^3 = 3025) : 
  r = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l3335_333540


namespace NUMINAMATH_CALUDE_f_of_10_l3335_333544

/-- Given a function f(x) = 2x^2 + y where f(2) = 30, prove that f(10) = 222 -/
theorem f_of_10 (f : ℝ → ℝ) (y : ℝ) 
    (h1 : ∀ x, f x = 2 * x^2 + y) 
    (h2 : f 2 = 30) : 
  f 10 = 222 := by
sorry

end NUMINAMATH_CALUDE_f_of_10_l3335_333544


namespace NUMINAMATH_CALUDE_number_of_nests_l3335_333553

theorem number_of_nests (birds : ℕ) (nests : ℕ) : 
  birds = 6 → birds = nests + 3 → nests = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_nests_l3335_333553


namespace NUMINAMATH_CALUDE_kanga_lands_on_84_l3335_333517

def jump_sequence (n : ℕ) : ℕ :=
  9 * n

def kanga_position (n : ℕ) (extra_jumps : ℕ) : ℕ :=
  jump_sequence n + 
  if extra_jumps ≤ 2 then 3 * extra_jumps
  else 6 + (extra_jumps - 2)

theorem kanga_lands_on_84 : 
  ∃ (n : ℕ) (extra_jumps : ℕ), 
    kanga_position n extra_jumps = 84 ∧ 
    kanga_position n extra_jumps ≠ 82 ∧
    kanga_position n extra_jumps ≠ 83 ∧
    kanga_position n extra_jumps ≠ 85 ∧
    kanga_position n extra_jumps ≠ 86 :=
by sorry

end NUMINAMATH_CALUDE_kanga_lands_on_84_l3335_333517


namespace NUMINAMATH_CALUDE_marbles_on_desk_l3335_333599

theorem marbles_on_desk (desk_marbles : ℕ) : desk_marbles + 6 = 8 → desk_marbles = 2 := by
  sorry

end NUMINAMATH_CALUDE_marbles_on_desk_l3335_333599


namespace NUMINAMATH_CALUDE_quadratic_sum_l3335_333574

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (8 * x^2 + 40 * x + 160 = a * (x + b)^2 + c) ∧ (a + b + c = 120.5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3335_333574


namespace NUMINAMATH_CALUDE_sweets_problem_l3335_333560

theorem sweets_problem (num_children : ℕ) (sweets_per_child : ℕ) (remaining_fraction : ℚ) :
  num_children = 48 →
  sweets_per_child = 4 →
  remaining_fraction = 1/3 →
  ∃ total_sweets : ℕ,
    total_sweets = num_children * sweets_per_child / (1 - remaining_fraction) ∧
    total_sweets = 288 := by
  sorry

end NUMINAMATH_CALUDE_sweets_problem_l3335_333560


namespace NUMINAMATH_CALUDE_lighthouse_distance_l3335_333576

/-- Proves that in a triangle ABS with given side length and angles, BS = 72 km -/
theorem lighthouse_distance (AB : ℝ) (angle_A angle_B : ℝ) :
  AB = 36 * Real.sqrt 6 →
  angle_A = 45 * π / 180 →
  angle_B = 75 * π / 180 →
  let angle_S := π - (angle_A + angle_B)
  let BS := AB * Real.sin angle_A / Real.sin angle_S
  BS = 72 := by sorry

end NUMINAMATH_CALUDE_lighthouse_distance_l3335_333576


namespace NUMINAMATH_CALUDE_tray_height_proof_l3335_333526

/-- Given a square with side length 150 and cuts starting 8 units from each corner
    meeting at a 45° angle on the diagonal, the height of the resulting tray when folded
    is equal to the fourth root of 4096. -/
theorem tray_height_proof (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) :
  side_length = 150 →
  cut_distance = 8 →
  cut_angle = 45 →
  ∃ (h : ℝ), h = (8 * Real.sqrt 2 - 8) ∧ h^4 = 4096 :=
by sorry

end NUMINAMATH_CALUDE_tray_height_proof_l3335_333526


namespace NUMINAMATH_CALUDE_one_right_angled_triangle_l3335_333587

/-- A triangle with side lengths 15, 20, and x has exactly one right angle -/
def has_one_right_angle (x : ℤ) : Prop :=
  (x ^ 2 = 15 ^ 2 + 20 ^ 2) ∨ 
  (15 ^ 2 = x ^ 2 + 20 ^ 2) ∨ 
  (20 ^ 2 = 15 ^ 2 + x ^ 2)

/-- The triangle inequality is satisfied -/
def satisfies_triangle_inequality (x : ℤ) : Prop :=
  x > 0 ∧ 15 + 20 > x ∧ 15 + x > 20 ∧ 20 + x > 15

/-- There exists exactly one integer x that satisfies the conditions -/
theorem one_right_angled_triangle : 
  ∃! x : ℤ, satisfies_triangle_inequality x ∧ has_one_right_angle x :=
sorry

end NUMINAMATH_CALUDE_one_right_angled_triangle_l3335_333587


namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_11_l3335_333509

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def digit_sum_odd (n : ℕ) : ℕ :=
  (n / 10000000) + ((n / 100000) % 10) + ((n / 1000) % 10) + ((n / 10) % 10)

def digit_sum_even (n : ℕ) : ℕ :=
  ((n / 1000000) % 10) + ((n / 10000) % 10) + ((n / 100) % 10) + (n % 10)

theorem smallest_digit_divisible_by_11 :
  ∀ d : ℕ, d < 10 →
    (is_divisible_by_11 (85210000 + d * 1000 + 784) ↔ d = 1) ∧
    (∀ d' : ℕ, d' < d → ¬is_divisible_by_11 (85210000 + d' * 1000 + 784)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_11_l3335_333509


namespace NUMINAMATH_CALUDE_unique_solution_for_radical_equation_l3335_333521

theorem unique_solution_for_radical_equation (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (5 * x) * Real.sqrt (6 * x) * Real.sqrt (10 * x) = 10) : 
  x = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_radical_equation_l3335_333521


namespace NUMINAMATH_CALUDE_divisor_problem_l3335_333559

theorem divisor_problem (a b : ℕ) (divisor : ℕ) : 
  (10 ≤ a ∧ a ≤ 99) →  -- a is a two-digit number
  (a = 10 * (a / 10) + (a % 10)) →  -- a is represented in decimal form
  (divisor > 0) →  -- divisor is positive
  (a % divisor = 0) →  -- a is divisible by divisor
  (∀ x y : ℕ, (10 ≤ x ∧ x ≤ 99) → (x % divisor = 0) → (x / 10) * (x % 10) ≤ (a / 10) * (a % 10)) →  -- greatest possible value of b × a
  ((a / 10) * (a % 10) = 35) →  -- b × a = 35
  divisor = 3 :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l3335_333559


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_93_6_l3335_333585

theorem percentage_of_360_equals_93_6 : 
  (93.6 / 360) * 100 = 26 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_93_6_l3335_333585


namespace NUMINAMATH_CALUDE_ratio_equality_l3335_333556

theorem ratio_equality (x y : ℝ) (h : 1.5 * x = 0.04 * y) :
  (y - x) / (y + x) = 73 / 77 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3335_333556


namespace NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l3335_333565

def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

theorem monotonic_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y ∨ ∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l3335_333565


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l3335_333528

theorem sum_of_fractions_equals_seven :
  let S := 1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 14) - 
           1 / (Real.sqrt 14 - Real.sqrt 13) + 1 / (Real.sqrt 13 - 3)
  S = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l3335_333528


namespace NUMINAMATH_CALUDE_perimeter_ABCDEFG_l3335_333563

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point2D) : ℝ := sorry

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (m p1 p2 : Point2D) : Prop := sorry

/-- Calculate the perimeter of a polygon given by a list of points -/
def perimeter (points : List Point2D) : ℝ := sorry

/-- The main theorem -/
theorem perimeter_ABCDEFG :
  ∀ (A B C D E F G : Point2D),
    isEquilateral ⟨A, B, C⟩ →
    isEquilateral ⟨A, D, E⟩ →
    isEquilateral ⟨E, F, G⟩ →
    isMidpoint D A C →
    isMidpoint G A E →
    distance A B = 6 →
    perimeter [A, B, C, D, E, F, G] = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ABCDEFG_l3335_333563


namespace NUMINAMATH_CALUDE_letters_problem_l3335_333598

/-- The number of letters Greta's brother received -/
def brothers_letters : ℕ := sorry

/-- The number of letters Greta received -/
def gretas_letters : ℕ := sorry

/-- The number of letters Greta's mother received -/
def mothers_letters : ℕ := sorry

theorem letters_problem :
  (gretas_letters = brothers_letters + 10) ∧
  (mothers_letters = 2 * (gretas_letters + brothers_letters)) ∧
  (brothers_letters + gretas_letters + mothers_letters = 270) →
  brothers_letters = 40 := by
  sorry

end NUMINAMATH_CALUDE_letters_problem_l3335_333598


namespace NUMINAMATH_CALUDE_angle_with_seven_times_complement_supplement_l3335_333503

theorem angle_with_seven_times_complement_supplement (x : ℝ) : 
  (180 - x = 7 * (90 - x)) → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_seven_times_complement_supplement_l3335_333503


namespace NUMINAMATH_CALUDE_arc_length_30_degree_sector_l3335_333510

/-- The length of an arc in a circular sector with radius 1 cm and central angle 30° is π/6 cm. -/
theorem arc_length_30_degree_sector :
  let r : ℝ := 1  -- radius in cm
  let θ : ℝ := 30 * π / 180  -- central angle in radians
  let l : ℝ := r * θ  -- arc length formula
  l = π / 6 := by sorry

end NUMINAMATH_CALUDE_arc_length_30_degree_sector_l3335_333510


namespace NUMINAMATH_CALUDE_probability_is_one_half_l3335_333592

def total_balls : ℕ := 12
def white_balls : ℕ := 7
def black_balls : ℕ := 5
def drawn_balls : ℕ := 6

def probability_at_least_four_white : ℚ :=
  (Nat.choose white_balls 4 * Nat.choose black_balls 2 +
   Nat.choose white_balls 5 * Nat.choose black_balls 1 +
   Nat.choose white_balls 6 * Nat.choose black_balls 0) /
  Nat.choose total_balls drawn_balls

theorem probability_is_one_half :
  probability_at_least_four_white = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_half_l3335_333592


namespace NUMINAMATH_CALUDE_expression_simplification_l3335_333501

theorem expression_simplification (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m^2 - 1/n^2)^m * (n + 1/m)^(n-m) / ((n^2 - 1/m^2)^n * (m - 1/n)^(m-n)) = (m/n)^(m+n) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3335_333501


namespace NUMINAMATH_CALUDE_price_decrease_unit_increase_ratio_l3335_333595

theorem price_decrease_unit_increase_ratio (P U V : ℝ) 
  (h1 : P > 0) 
  (h2 : U > 0) 
  (h3 : V > U) 
  (h4 : P * U = 0.25 * P * V) : 
  ((V - U) / U) / 0.75 = 4 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_unit_increase_ratio_l3335_333595


namespace NUMINAMATH_CALUDE_smaller_mold_radius_l3335_333550

/-- The radius of a smaller hemisphere-shaped mold when a large hemisphere-shaped bowl
    with radius 1 foot is evenly distributed into 64 congruent smaller molds. -/
theorem smaller_mold_radius : ℝ → ℝ → ℝ → Prop :=
  fun (large_radius : ℝ) (num_molds : ℝ) (small_radius : ℝ) =>
    large_radius = 1 ∧
    num_molds = 64 ∧
    (2/3 * Real.pi * large_radius^3) = (num_molds * (2/3 * Real.pi * small_radius^3)) →
    small_radius = 1/4

/-- Proof of the smaller_mold_radius theorem. -/
lemma prove_smaller_mold_radius : smaller_mold_radius 1 64 (1/4) := by
  sorry

end NUMINAMATH_CALUDE_smaller_mold_radius_l3335_333550


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l3335_333568

/-- The height of a tree that triples every year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem: A tree that triples its height every year and reaches 243 feet after 5 years will be 9 feet tall after 2 years -/
theorem tree_height_after_two_years 
  (h : ∃ initial_height : ℝ, tree_height initial_height 5 = 243) :
  ∃ initial_height : ℝ, tree_height initial_height 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l3335_333568


namespace NUMINAMATH_CALUDE_mobile_phone_price_l3335_333580

/-- Represents the purchase and sale of two items with given profit/loss percentages and overall profit. -/
def ItemSale (refrigerator_price : ℝ) (refrigerator_loss : ℝ) (phone_profit : ℝ) (overall_profit : ℝ) (phone_price : ℝ) : Prop :=
  let refrigerator_sale := refrigerator_price * (1 - refrigerator_loss)
  let phone_sale := phone_price * (1 + phone_profit)
  refrigerator_sale + phone_sale - (refrigerator_price + phone_price) = overall_profit

/-- The purchase price of the mobile phone that satisfies the given conditions. -/
theorem mobile_phone_price :
  ∃ (phone_price : ℝ),
    ItemSale 15000 0.05 0.10 50 phone_price ∧
    phone_price = 8000 := by
  sorry

end NUMINAMATH_CALUDE_mobile_phone_price_l3335_333580


namespace NUMINAMATH_CALUDE_container_capacity_l3335_333586

theorem container_capacity (C : ℝ) : 0.40 * C + 14 = 0.75 * C → C = 40 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l3335_333586


namespace NUMINAMATH_CALUDE_managers_salary_l3335_333534

def employee_count : ℕ := 50
def initial_average_salary : ℚ := 2500
def average_increase : ℚ := 150

theorem managers_salary (manager_salary : ℚ) :
  (employee_count * initial_average_salary + manager_salary) / (employee_count + 1) =
  initial_average_salary + average_increase →
  manager_salary = 10150 := by
sorry

end NUMINAMATH_CALUDE_managers_salary_l3335_333534


namespace NUMINAMATH_CALUDE_calculate_dividend_l3335_333537

/-- Given a division with quotient, divisor, and remainder, calculate the dividend -/
theorem calculate_dividend (quotient divisor remainder : ℝ) :
  quotient = -415.2 →
  divisor = 2735 →
  remainder = 387.3 →
  (quotient * divisor) + remainder = -1135106.7 := by
  sorry

end NUMINAMATH_CALUDE_calculate_dividend_l3335_333537


namespace NUMINAMATH_CALUDE_max_l_value_l3335_333584

/-- The function f(x) = ax^2 + 8x + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 8 * x + 3

/-- The largest positive number l(a) such that |f(x)| ≤ 5 for all x ∈ [0, l(a)] -/
noncomputable def l (a : ℝ) : ℝ := sorry

/-- The theorem stating the maximum value of l(a) and the corresponding a -/
theorem max_l_value (a : ℝ) (h : a < 0) :
  (∃ (x : ℝ), x > 0 ∧ ∀ y ∈ Set.Icc 0 x, |f a y| ≤ 5) →
  (∀ b < 0, l b ≤ l a) →
  (a = -8 ∧ l a = (Real.sqrt 5 + 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_max_l_value_l3335_333584


namespace NUMINAMATH_CALUDE_inequality_solution_l3335_333569

theorem inequality_solution (x : ℝ) :
  (x^2 + 2*x - 15) / (x + 5) < 0 ↔ -5 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3335_333569


namespace NUMINAMATH_CALUDE_x_bijective_l3335_333543

def x : ℕ → ℤ
  | 0 => 0
  | n + 1 => 
    let r := (n + 1).log 3 + 1
    let k := (n + 1) / (3^(r-1)) - 1
    if (n + 1) = 3^(r-1) * (3*k + 1) then
      x n + (3^r - 1) / 2
    else if (n + 1) = 3^(r-1) * (3*k + 2) then
      x n - (3^r + 1) / 2
    else
      x n

theorem x_bijective : Function.Bijective x := by sorry

end NUMINAMATH_CALUDE_x_bijective_l3335_333543


namespace NUMINAMATH_CALUDE_inverse_proportion_points_l3335_333512

/-- Given that (-1,2) and (2,a) lie on the graph of y = k/x, prove that a = -1 -/
theorem inverse_proportion_points (k a : ℝ) : 
  (2 = k / (-1)) → (a = k / 2) → a = -1 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_points_l3335_333512


namespace NUMINAMATH_CALUDE_normal_distribution_two_std_below_mean_l3335_333541

theorem normal_distribution_two_std_below_mean :
  let μ : ℝ := 16.2  -- mean
  let σ : ℝ := 2.3   -- standard deviation
  let x : ℝ := μ - 2 * σ  -- value 2 standard deviations below mean
  x = 11.6 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_two_std_below_mean_l3335_333541


namespace NUMINAMATH_CALUDE_alices_favorite_number_l3335_333519

def is_multiple (a b : ℕ) : Prop := ∃ k, a = b * k

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem alices_favorite_number : ∃! n : ℕ, 
  90 < n ∧ n < 150 ∧ 
  is_multiple n 13 ∧ 
  ¬is_multiple n 3 ∧ 
  is_multiple (digit_sum n) 4 ∧
  n = 130 := by sorry

end NUMINAMATH_CALUDE_alices_favorite_number_l3335_333519


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3335_333520

-- Define the logarithm base 2
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- Define the logarithm base 10
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

theorem logarithm_expression_equality : 
  2^(log2 3) + lg (Real.sqrt 5) + lg (Real.sqrt 20) = 4 := by sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3335_333520


namespace NUMINAMATH_CALUDE_odd_function_property_l3335_333589

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h : is_odd_function f) :
  ∀ x, f x * f (-x) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3335_333589


namespace NUMINAMATH_CALUDE_ratio_problem_l3335_333588

theorem ratio_problem (a b c : ℚ) : 
  b / a = 4 → 
  b = 18 - 7 * a → 
  c = 2 * a - 6 → 
  a = 18 / 11 ∧ c = -30 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3335_333588


namespace NUMINAMATH_CALUDE_white_dandelions_on_saturday_l3335_333570

/-- Represents the state of dandelions in a meadow on a given day -/
structure DandelionState :=
  (yellow : ℕ)
  (white : ℕ)

/-- The lifecycle of a dandelion in days -/
def dandelionLifecycle : ℕ := 5

/-- The number of days a dandelion remains yellow -/
def yellowDays : ℕ := 3

/-- The state of dandelions on Monday -/
def mondayState : DandelionState :=
  { yellow := 20, white := 14 }

/-- The state of dandelions on Wednesday -/
def wednesdayState : DandelionState :=
  { yellow := 15, white := 11 }

/-- The number of days between Monday and Saturday -/
def daysToSaturday : ℕ := 5

theorem white_dandelions_on_saturday :
  (wednesdayState.yellow + wednesdayState.white) - mondayState.yellow =
    (mondayState.yellow + mondayState.white + daysToSaturday - dandelionLifecycle) :=
by sorry

end NUMINAMATH_CALUDE_white_dandelions_on_saturday_l3335_333570


namespace NUMINAMATH_CALUDE_no_valid_a_l3335_333542

theorem no_valid_a : ¬∃ a : ℕ+, (a ≤ 100) ∧ 
  (∃ x y : ℤ, x ≠ y ∧ 
    2 * x^2 + (3 * a.val + 1) * x + a.val^2 = 0 ∧
    2 * y^2 + (3 * a.val + 1) * y + a.val^2 = 0) :=
by
  sorry

#check no_valid_a

end NUMINAMATH_CALUDE_no_valid_a_l3335_333542


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_smallest_sum_is_negative_six_l3335_333511

def S : Finset Int := {0, 5, -2, 18, -4, 3}

theorem smallest_sum_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
  x ≠ y ∧ y ≠ z ∧ x ≠ z → 
  a + b + c ≤ x + y + z :=
by sorry

theorem smallest_sum_is_negative_six :
  ∃ a b c : Int, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b + c = -6 ∧
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
   x ≠ y ∧ y ≠ z ∧ x ≠ z → 
   a + b + c ≤ x + y + z) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_smallest_sum_is_negative_six_l3335_333511


namespace NUMINAMATH_CALUDE_stamps_per_page_l3335_333529

theorem stamps_per_page (a b c : ℕ) (ha : a = 945) (hb : b = 1260) (hc : c = 630) :
  Nat.gcd a (Nat.gcd b c) = 315 := by
  sorry

end NUMINAMATH_CALUDE_stamps_per_page_l3335_333529


namespace NUMINAMATH_CALUDE_problem_solution_l3335_333593

theorem problem_solution (a b c d : ℝ) : 
  2 * a^2 + 2 * b^2 + 2 * c^2 + 3 = 2 * d + Real.sqrt (2 * a + 2 * b + 2 * c - 3 * d) →
  d = 23 / 48 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3335_333593


namespace NUMINAMATH_CALUDE_refrigerator_transport_cost_l3335_333551

/-- Calculate the transport cost for a refrigerator purchase --/
theorem refrigerator_transport_cost 
  (purchase_price : ℕ) 
  (discount_rate : ℚ) 
  (installation_cost : ℕ) 
  (profit_rate : ℚ) 
  (selling_price : ℕ) 
  (h1 : purchase_price = 14500)
  (h2 : discount_rate = 1/5)
  (h3 : installation_cost = 250)
  (h4 : profit_rate = 1/10)
  (h5 : selling_price = 20350) : 
  ∃ (transport_cost : ℕ), transport_cost = 3375 :=
by
  sorry

#check refrigerator_transport_cost

end NUMINAMATH_CALUDE_refrigerator_transport_cost_l3335_333551


namespace NUMINAMATH_CALUDE_max_losses_in_tennis_tournament_l3335_333504

theorem max_losses_in_tennis_tournament (n : ℕ) (h : n = 12) :
  ∃ (max_losses : ℕ), max_losses = 6 ∧
  (∀ (player : Fin n), ∃ (losses : ℕ), losses ≤ max_losses) ∧
  (∃ (player : Fin n), ∃ (losses : ℕ), losses = max_losses) :=
by sorry

end NUMINAMATH_CALUDE_max_losses_in_tennis_tournament_l3335_333504


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l3335_333555

theorem triangle_ABC_properties (A B C : Real) (a b c : Real) 
  (m n : Fin 2 → Real) :
  m 0 = Real.cos A ∧ m 1 = Real.sin A ∧
  n 0 = Real.sqrt 2 - Real.sin A ∧ n 1 = Real.cos A ∧
  (m 0 * n 0 + m 1 * n 1 = 1) ∧
  b = 4 * Real.sqrt 2 ∧
  c = Real.sqrt 2 * a →
  A = π / 4 ∧ 
  (1/2 : Real) * b * c * Real.sin A = 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l3335_333555


namespace NUMINAMATH_CALUDE_height_difference_l3335_333597

/-- The height of the CN Tower in meters -/
def cn_tower_height : ℝ := 553

/-- The height of the Space Needle in meters -/
def space_needle_height : ℝ := 184

/-- Theorem stating the difference in height between the CN Tower and the Space Needle -/
theorem height_difference : cn_tower_height - space_needle_height = 369 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l3335_333597


namespace NUMINAMATH_CALUDE_nina_money_theorem_l3335_333516

theorem nina_money_theorem (x : ℝ) (h1 : 10 * x = 14 * (x - 3)) : 10 * x = 105 := by
  sorry

end NUMINAMATH_CALUDE_nina_money_theorem_l3335_333516


namespace NUMINAMATH_CALUDE_game_a_vs_game_b_l3335_333514

def coin_prob_heads : ℚ := 2/3
def coin_prob_tails : ℚ := 1/3

def game_a_win (p : ℚ) : ℚ := p^4 + (1-p)^4

def game_b_win (p : ℚ) : ℚ := p^3 * (1-p) + (1-p)^3 * p

theorem game_a_vs_game_b :
  game_a_win coin_prob_heads - game_b_win coin_prob_heads = 7/81 :=
by sorry

end NUMINAMATH_CALUDE_game_a_vs_game_b_l3335_333514


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l3335_333561

/-- Given two lines l₁ and l₂ in the plane, and a third line l₃,
    this theorem proves that the line ax + by + c = 0 passes through
    the intersection point of l₁ and l₂, and is perpendicular to l₃. -/
theorem intersection_and_perpendicular_line
  (l₁ : Real → Real → Prop) (l₂ : Real → Real → Prop) (l₃ : Real → Real → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ 2 * x - 3 * y + 10 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 3 * x + 4 * y - 2 = 0)
  (h₃ : ∀ x y, l₃ x y ↔ 3 * x - 2 * y + 5 = 0)
  : ∃ x y, l₁ x y ∧ l₂ x y ∧ 2 * x + 3 * y - 2 = 0 ∧
    (∀ x₁ y₁ x₂ y₂, l₃ x₁ y₁ → l₃ x₂ y₂ → (y₂ - y₁) * (3 * (x₂ - x₁)) = -2 * (y₂ - y₁)) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l3335_333561


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3335_333573

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 3 * x - 4 * y = -7 ∧ 4 * x - 3 * y = 5 ∧ x = 41 / 7 ∧ y = 43 / 7 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3335_333573


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3335_333577

theorem interest_rate_calculation (principal : ℝ) (time : ℝ) (lending_rate : ℝ) (gain_per_year : ℝ) 
  (h1 : principal = 20000)
  (h2 : time = 6)
  (h3 : lending_rate = 0.09)
  (h4 : gain_per_year = 200) :
  let interest_received := principal * lending_rate * time
  let total_gain := gain_per_year * time
  let interest_paid := interest_received - total_gain
  let borrowing_rate := interest_paid / (principal * time)
  borrowing_rate = 0.08 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3335_333577


namespace NUMINAMATH_CALUDE_remainder_2468135792_mod_101_l3335_333532

theorem remainder_2468135792_mod_101 : 2468135792 % 101 = 52 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2468135792_mod_101_l3335_333532


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3335_333575

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (4 * x + 9) = 12 → x = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3335_333575


namespace NUMINAMATH_CALUDE_range_of_m_l3335_333547

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), x^2 + 2*x - m > 0) ↔ 
  (1^2 + 2*1 - m ≤ 0 ∧ 2^2 + 2*2 - m > 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3335_333547


namespace NUMINAMATH_CALUDE_x_with_18_factors_l3335_333527

theorem x_with_18_factors (x : ℕ) : 
  (∃ (factors : Finset ℕ), factors.card = 18 ∧ (∀ f ∈ factors, f ∣ x)) → 
  18 ∣ x → 
  20 ∣ x → 
  x = 180 := by
sorry

end NUMINAMATH_CALUDE_x_with_18_factors_l3335_333527


namespace NUMINAMATH_CALUDE_f_below_tangent_and_inequality_l3335_333554

-- Define the function f(x) = (2-x)e^x
noncomputable def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

-- Define the tangent line l(x) = x + 2
def l (x : ℝ) : ℝ := x + 2

theorem f_below_tangent_and_inequality (n : ℕ) (hn : n > 0) :
  (∀ x : ℝ, x ≥ 0 → f x ≤ l x) ∧
  (f (1 / n - 1 / (n + 1)) + (1 / Real.exp 2) * f (2 - 1 / n) ≤ 2 + 1 / n) := by
  sorry

end NUMINAMATH_CALUDE_f_below_tangent_and_inequality_l3335_333554


namespace NUMINAMATH_CALUDE_xy_value_l3335_333533

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3335_333533


namespace NUMINAMATH_CALUDE_sum_of_roots_range_l3335_333524

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := if x ≤ 0 then x^2 else Real.exp x

-- Define the function F as [f(x)]^2
def F (x : ℝ) : ℝ := (f x)^2

-- Define the property that F(x) = a has exactly two roots
def has_two_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ F x₁ = a ∧ F x₂ = a ∧ ∀ x, F x = a → x = x₁ ∨ x = x₂

-- Theorem statement
theorem sum_of_roots_range (a : ℝ) (h : has_two_roots a) :
  ∃ x₁ x₂, F x₁ = a ∧ F x₂ = a ∧ x₁ + x₂ > -1 ∧ ∀ M, ∃ b > a, 
  ∃ y₁ y₂, F y₁ = b ∧ F y₂ = b ∧ y₁ + y₂ > M :=
sorry

end

end NUMINAMATH_CALUDE_sum_of_roots_range_l3335_333524


namespace NUMINAMATH_CALUDE_symmetric_about_x_axis_periodic_condition_symmetric_about_origin_l3335_333572

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Statement 1
theorem symmetric_about_x_axis (f : ℝ → ℝ) :
  (∀ x, f (-1 - x) = f (x - 1)) ↔ 
  (∀ x, f x = f (-x)) :=
sorry

-- Statement 2
theorem periodic_condition (f : ℝ → ℝ) :
  (∀ x, f (1 + x) = f (x - 1)) → 
  (∀ x, f (x + 2) = f x) :=
sorry

-- Statement 3
theorem symmetric_about_origin (f : ℝ → ℝ) :
  (∀ x, f (1 - x) = -f (x - 1)) → 
  (∀ x, f (-x) = -f x) :=
sorry

end NUMINAMATH_CALUDE_symmetric_about_x_axis_periodic_condition_symmetric_about_origin_l3335_333572


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_cube_l3335_333525

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_factor_for_perfect_cube (x y : ℕ) : 
  x = 5 * 30 * 60 →
  y > 0 →
  is_perfect_cube (x * y) →
  (∀ z : ℕ, z > 0 → z < y → ¬ is_perfect_cube (x * z)) →
  y = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_cube_l3335_333525


namespace NUMINAMATH_CALUDE_share_division_l3335_333582

theorem share_division (total : ℕ) (a b c : ℚ) 
  (h_total : total = 427)
  (h_sum : a + b + c = total)
  (h_ratio : 3 * a = 4 * b ∧ 4 * b = 7 * c) : 
  c = 84 := by
  sorry

end NUMINAMATH_CALUDE_share_division_l3335_333582


namespace NUMINAMATH_CALUDE_stock_sale_total_amount_l3335_333591

/-- Calculates the total amount including brokerage for a stock sale -/
theorem stock_sale_total_amount 
  (cash_realized : ℝ) 
  (brokerage_rate : ℝ) 
  (h1 : cash_realized = 106.25) 
  (h2 : brokerage_rate = 0.25) : 
  ∃ (total_amount : ℝ), total_amount = 106.52 ∧ 
  total_amount = cash_realized + (brokerage_rate / 100) * cash_realized :=
by sorry

end NUMINAMATH_CALUDE_stock_sale_total_amount_l3335_333591


namespace NUMINAMATH_CALUDE_harvest_duration_l3335_333531

def harvest_problem (weekly_earning : ℕ) (total_earning : ℕ) : Prop :=
  weekly_earning * 89 = total_earning

theorem harvest_duration : harvest_problem 2 178 := by
  sorry

end NUMINAMATH_CALUDE_harvest_duration_l3335_333531


namespace NUMINAMATH_CALUDE_circle_symmetric_point_theorem_l3335_333567

/-- A circle C in the xy-plane -/
structure Circle where
  a : ℝ
  b : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a circle -/
def Point.onCircle (p : Point) (c : Circle) : Prop :=
  p.x^2 + p.y^2 + 2*c.a*p.x - 4*p.y + c.b = 0

/-- Find the symmetric point of a given point about the line x + y - 3 = 0 -/
def symmetricPoint (p : Point) : Point :=
  { x := 2 - p.y, y := 2 - p.x }

/-- Main theorem -/
theorem circle_symmetric_point_theorem (c : Circle) : 
  let p : Point := { x := 1, y := 4 }
  (p.onCircle c ∧ (symmetricPoint p).onCircle c) → c.a = -1 ∧ c.b = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetric_point_theorem_l3335_333567


namespace NUMINAMATH_CALUDE_group_size_problem_l3335_333536

/-- Given a group where each member contributes as many paise as there are members,
    and the total collection is 5929 paise, prove that the number of members is 77. -/
theorem group_size_problem (n : ℕ) (h1 : n * n = 5929) : n = 77 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l3335_333536


namespace NUMINAMATH_CALUDE_total_cats_l3335_333581

theorem total_cats (original_cats female_kittens male_kittens : ℕ) 
  (h1 : original_cats = 2)
  (h2 : female_kittens = 3)
  (h3 : male_kittens = 2) :
  original_cats + female_kittens + male_kittens = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_l3335_333581


namespace NUMINAMATH_CALUDE_solution_set_for_m_eq_1_m_range_for_inequality_l3335_333513

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + m| + |2*x - 1|

-- Part 1
theorem solution_set_for_m_eq_1 :
  {x : ℝ | f 1 x ≥ 3} = {x : ℝ | x ≤ -1 ∨ x ≥ 1} :=
sorry

-- Part 2
theorem m_range_for_inequality (m : ℝ) :
  (0 < m ∧ m < 1/4) →
  (∀ x ∈ Set.Icc m (2*m), (1/2) * (f m x) ≤ |x + 1|) →
  m ∈ Set.Ioo 0 (1/4) :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_m_eq_1_m_range_for_inequality_l3335_333513


namespace NUMINAMATH_CALUDE_seokjin_pencils_used_l3335_333579

-- Define the initial number of pencils
def initial_pencils : ℕ := 12

-- Define the number of pencils Jungkook gave to Seokjin
def pencils_given : ℕ := 4

-- Define the final number of pencils each person has
def final_pencils : ℕ := 7

-- Define the number of pencils Seokjin used
def seokjin_used : ℕ := 9

-- Theorem statement
theorem seokjin_pencils_used :
  initial_pencils - seokjin_used + pencils_given = final_pencils ∧
  initial_pencils - (initial_pencils - final_pencils - pencils_given) - pencils_given = final_pencils :=
by sorry

end NUMINAMATH_CALUDE_seokjin_pencils_used_l3335_333579


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_factorial_squared_gt_power_of_two_l3335_333505

theorem negation_of_existence (P : ℕ → Prop) : 
  (¬ ∃ n, P n) ↔ (∀ n, ¬ P n) :=
by sorry

theorem negation_of_factorial_squared_gt_power_of_two : 
  (¬ ∃ n : ℕ, (n.factorial ^ 2 : ℝ) > 2^n) ↔ 
  (∀ n : ℕ, (n.factorial ^ 2 : ℝ) ≤ 2^n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_factorial_squared_gt_power_of_two_l3335_333505


namespace NUMINAMATH_CALUDE_hyperbola_imaginary_axis_length_l3335_333566

/-- Given a hyperbola x^2 + my^2 = 1 passing through the point (-√2, 2),
    the length of its imaginary axis is 4. -/
theorem hyperbola_imaginary_axis_length 
  (m : ℝ) 
  (h : (-Real.sqrt 2)^2 + m * 2^2 = 1) : 
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ↔ x^2 + m*y^2 = 1) ∧
    2*b = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_imaginary_axis_length_l3335_333566


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3335_333552

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def has_even_number_of_factors (n : ℕ) : Prop :=
  Even (Nat.card (Nat.divisors n))

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ, is_two_digit n ∧
    ((has_even_number_of_factors n ∨ n > 50) ∧
     ¬(has_even_number_of_factors n ∧ n > 50)) ∧
    ((Odd n ∨ n > 60) ∧ ¬(Odd n ∧ n > 60)) ∧
    ((Even n ∨ n > 70) ∧ ¬(Even n ∧ n > 70)) ∧
    n = 64 :=
by
  sorry

#check unique_number_satisfying_conditions

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3335_333552


namespace NUMINAMATH_CALUDE_star_inequality_l3335_333500

def star (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem star_inequality (x y : ℝ) : 3 * (star x y) ≠ star (3*x) (3*y) := by
  sorry

end NUMINAMATH_CALUDE_star_inequality_l3335_333500


namespace NUMINAMATH_CALUDE_common_material_choices_eq_120_l3335_333548

/-- The number of ways to choose r items from n items --/
def choose (n r : ℕ) : ℕ := Nat.choose n r

/-- The number of ways to arrange r items from n items --/
def arrange (n r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of ways two students can choose 2 out of 6 materials each, 
    such that they have exactly 1 material in common --/
def commonMaterialChoices : ℕ :=
  choose 6 1 * arrange 5 2

theorem common_material_choices_eq_120 : commonMaterialChoices = 120 := by
  sorry

end NUMINAMATH_CALUDE_common_material_choices_eq_120_l3335_333548


namespace NUMINAMATH_CALUDE_number_problem_l3335_333518

theorem number_problem : ∃ x : ℝ, 0.65 * x - 25 = 90 ∧ abs (x - 176.92) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3335_333518


namespace NUMINAMATH_CALUDE_irene_age_is_46_l3335_333596

-- Define the ages as natural numbers
def eddie_age : ℕ := 92
def becky_age : ℕ := eddie_age / 4
def irene_age : ℕ := 2 * becky_age

-- Theorem statement
theorem irene_age_is_46 : irene_age = 46 := by
  sorry

end NUMINAMATH_CALUDE_irene_age_is_46_l3335_333596


namespace NUMINAMATH_CALUDE_last_five_days_avg_l3335_333535

/-- Represents the TV production scenario in a factory --/
structure TVProduction where
  total_days : Nat
  first_period_days : Nat
  first_period_avg : Nat
  monthly_avg : Nat

/-- Calculates the average daily production for the last period --/
def last_period_avg (prod : TVProduction) : Rat :=
  let last_period_days := prod.total_days - prod.first_period_days
  let total_monthly_production := prod.monthly_avg * prod.total_days
  let first_period_production := prod.first_period_avg * prod.first_period_days
  let last_period_production := total_monthly_production - first_period_production
  last_period_production / last_period_days

/-- Theorem stating the average production for the last 5 days --/
theorem last_five_days_avg (prod : TVProduction) 
  (h1 : prod.total_days = 30)
  (h2 : prod.first_period_days = 25)
  (h3 : prod.first_period_avg = 50)
  (h4 : prod.monthly_avg = 45) :
  last_period_avg prod = 20 := by
  sorry

end NUMINAMATH_CALUDE_last_five_days_avg_l3335_333535


namespace NUMINAMATH_CALUDE_rth_term_is_8r_l3335_333557

-- Define the sum of n terms for the arithmetic progression
def S (n : ℕ) : ℕ := 5 * n + 4 * n^2 + 1

-- Define the r-th term of the arithmetic progression
def a (r : ℕ) : ℕ := S r - S (r - 1)

-- Theorem stating that the r-th term is equal to 8r
theorem rth_term_is_8r (r : ℕ) : a r = 8 * r := by
  sorry

end NUMINAMATH_CALUDE_rth_term_is_8r_l3335_333557


namespace NUMINAMATH_CALUDE_production_equation_proof_l3335_333508

/-- Represents a furniture production scenario -/
structure ProductionScenario where
  total : ℕ              -- Total sets to produce
  increase : ℕ           -- Daily production increase
  days_saved : ℕ         -- Days saved due to increase
  original_rate : ℕ      -- Original daily production rate

/-- Theorem stating the correct equation for the production scenario -/
theorem production_equation_proof (s : ProductionScenario) 
  (h1 : s.total = 540)
  (h2 : s.increase = 2)
  (h3 : s.days_saved = 3) :
  (s.total : ℝ) / s.original_rate - (s.total : ℝ) / (s.original_rate + s.increase) = s.days_saved := by
  sorry

#check production_equation_proof

end NUMINAMATH_CALUDE_production_equation_proof_l3335_333508


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l3335_333590

theorem cara_seating_arrangements (n : ℕ) (h : n = 8) :
  (n - 2 : ℕ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l3335_333590


namespace NUMINAMATH_CALUDE_triangle_side_sum_range_l3335_333502

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (a-b)(sin A + sin B) = (c-b)sin C and a = √3, then 5 < b² + c² ≤ 6. -/
theorem triangle_side_sum_range (a b c A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- Acute triangle
  A + B + C = π ∧ -- Sum of angles in a triangle
  (a - b) * (Real.sin A + Real.sin B) = (c - b) * Real.sin C ∧ -- Given condition
  a = Real.sqrt 3 → -- Given condition
  5 < b^2 + c^2 ∧ b^2 + c^2 ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_range_l3335_333502


namespace NUMINAMATH_CALUDE_ribbon_remaining_length_l3335_333546

/-- The length of the original ribbon in meters -/
def original_length : ℝ := 51

/-- The number of pieces cut from the ribbon -/
def num_pieces : ℕ := 100

/-- The length of each piece in centimeters -/
def piece_length_cm : ℝ := 15

/-- Conversion factor from centimeters to meters -/
def cm_to_m : ℝ := 0.01

/-- The remaining length of the ribbon after cutting the pieces -/
def remaining_length : ℝ := original_length - (num_pieces : ℝ) * piece_length_cm * cm_to_m

theorem ribbon_remaining_length :
  remaining_length = 36 := by sorry

end NUMINAMATH_CALUDE_ribbon_remaining_length_l3335_333546


namespace NUMINAMATH_CALUDE_function_property_l3335_333507

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x + f (1 - x) = 10)
  (h2 : ∀ x : ℝ, f (1 + x) = 3 + f x) :
  ∀ x : ℝ, f x + f (-x) = 7 := by
sorry

end NUMINAMATH_CALUDE_function_property_l3335_333507


namespace NUMINAMATH_CALUDE_circle_problem_l3335_333564

-- Define the circles and points
def larger_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 15^2}
def smaller_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 10^2}
def P : ℝ × ℝ := (9, 12)
def S (k : ℝ) : ℝ × ℝ := (0, k)

-- Define the theorem
theorem circle_problem (k : ℝ) :
  P ∈ larger_circle ∧
  S k ∈ smaller_circle ∧
  (∀ p ∈ larger_circle, ∃ q ∈ smaller_circle, ‖p - q‖ = 5) →
  k = 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_problem_l3335_333564


namespace NUMINAMATH_CALUDE_min_buses_for_field_trip_l3335_333558

/-- Represents the number of passengers that can be transported by a combination of buses. -/
def transport_capacity (small medium large : ℕ) : ℕ :=
  30 * small + 48 * medium + 72 * large

/-- Represents the total number of buses used. -/
def total_buses (small medium large : ℕ) : ℕ :=
  small + medium + large

theorem min_buses_for_field_trip :
  ∃ (small medium large : ℕ),
    small ≤ 10 ∧
    medium ≤ 15 ∧
    large ≤ 5 ∧
    transport_capacity small medium large ≥ 1230 ∧
    total_buses small medium large = 25 ∧
    (∀ (s m l : ℕ),
      s ≤ 10 →
      m ≤ 15 →
      l ≤ 5 →
      transport_capacity s m l ≥ 1230 →
      total_buses s m l ≥ 25) :=
by sorry

end NUMINAMATH_CALUDE_min_buses_for_field_trip_l3335_333558


namespace NUMINAMATH_CALUDE_ones_digit_of_8_to_47_l3335_333523

def ones_digit_cycle : List Nat := [8, 4, 2, 6]

theorem ones_digit_of_8_to_47 (h : ones_digit_cycle = [8, 4, 2, 6]) :
  (8^47 : ℕ) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_8_to_47_l3335_333523


namespace NUMINAMATH_CALUDE_eighth_term_of_specific_arithmetic_sequence_l3335_333578

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 8th term of the arithmetic sequence with first term 1 and common difference 3 is 22 -/
theorem eighth_term_of_specific_arithmetic_sequence :
  arithmeticSequenceTerm 1 3 8 = 22 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_specific_arithmetic_sequence_l3335_333578


namespace NUMINAMATH_CALUDE_yannas_baking_problem_l3335_333538

/-- Yanna's baking problem -/
theorem yannas_baking_problem (morning_butter_cookies morning_biscuits afternoon_butter_cookies afternoon_biscuits : ℕ) 
  (h1 : morning_butter_cookies = 20)
  (h2 : afternoon_butter_cookies = 10)
  (h3 : afternoon_biscuits = 20)
  (h4 : morning_biscuits + afternoon_biscuits = morning_butter_cookies + afternoon_butter_cookies + 30) :
  morning_biscuits = 40 := by
  sorry

end NUMINAMATH_CALUDE_yannas_baking_problem_l3335_333538


namespace NUMINAMATH_CALUDE_annette_caitlin_weight_l3335_333530

/-- The combined weight of Annette and Caitlin given the conditions -/
theorem annette_caitlin_weight :
  ∀ (annette caitlin sara : ℝ),
  caitlin + sara = 87 →
  annette = sara + 8 →
  annette + caitlin = 95 :=
by
  sorry

end NUMINAMATH_CALUDE_annette_caitlin_weight_l3335_333530


namespace NUMINAMATH_CALUDE_train_length_l3335_333562

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 60 * (1000 / 3600) →
  crossing_time = 18.598512119030477 →
  bridge_length = 200 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3335_333562


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l3335_333583

/-- The surface area of a cube with corner cubes removed -/
def surface_area_with_corners_removed (cube_side_length : ℝ) (corner_side_length : ℝ) : ℝ :=
  6 * cube_side_length^2

/-- The theorem stating that the surface area remains unchanged -/
theorem surface_area_unchanged (cube_side_length : ℝ) (corner_side_length : ℝ) 
  (h1 : cube_side_length = 5) 
  (h2 : corner_side_length = 2) : 
  surface_area_with_corners_removed cube_side_length corner_side_length = 150 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l3335_333583


namespace NUMINAMATH_CALUDE_train_passes_jogger_l3335_333545

/-- The time it takes for a train to pass a jogger given their speeds and initial positions -/
theorem train_passes_jogger (jogger_speed train_speed : ℝ) (train_length initial_distance : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  initial_distance = 230 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 35 :=
by sorry

end NUMINAMATH_CALUDE_train_passes_jogger_l3335_333545


namespace NUMINAMATH_CALUDE_sum_remainder_mod_nine_l3335_333594

theorem sum_remainder_mod_nine : ∃ k : ℕ, 
  88134 + 88135 + 88136 + 88137 + 88138 + 88139 = 9 * k + 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_nine_l3335_333594


namespace NUMINAMATH_CALUDE_sawmill_equivalence_l3335_333506

/-- Represents the number of cuts needed to divide a log into smaller logs -/
def cuts_needed (original_length : ℕ) (target_length : ℕ) : ℕ :=
  original_length / target_length - 1

/-- Represents the total number of cuts that can be made in one day -/
def cuts_per_day (logs_per_day : ℕ) (original_length : ℕ) (target_length : ℕ) : ℕ :=
  logs_per_day * cuts_needed original_length target_length

/-- Represents the time (in days) needed to cut a given number of logs -/
def time_needed (num_logs : ℕ) (original_length : ℕ) (target_length : ℕ) (cuts_per_day : ℕ) : ℚ :=
  (num_logs * cuts_needed original_length target_length : ℚ) / cuts_per_day

theorem sawmill_equivalence :
  let nine_meter_logs_per_day : ℕ := 600
  let twelve_meter_logs : ℕ := 400
  let cuts_per_day := cuts_per_day nine_meter_logs_per_day 9 3
  time_needed twelve_meter_logs 12 3 cuts_per_day = 1 := by
  sorry

end NUMINAMATH_CALUDE_sawmill_equivalence_l3335_333506


namespace NUMINAMATH_CALUDE_not_passed_implies_scored_less_than_90_percent_l3335_333549

-- Define the proposition for scoring at least 90% on the final exam
def scored_at_least_90_percent (student : Type) : Prop := sorry

-- Define the proposition for passing the course
def passed_course (student : Type) : Prop := sorry

-- State the given condition
axiom condition (student : Type) : passed_course student → scored_at_least_90_percent student

-- State the theorem to be proved
theorem not_passed_implies_scored_less_than_90_percent (student : Type) :
  ¬(passed_course student) → ¬(scored_at_least_90_percent student) := by sorry

end NUMINAMATH_CALUDE_not_passed_implies_scored_less_than_90_percent_l3335_333549


namespace NUMINAMATH_CALUDE_eight_digit_even_increasing_numbers_l3335_333539

theorem eight_digit_even_increasing_numbers (n : ℕ) (k : ℕ) : 
  n = 8 ∧ k = 4 → (n + k - 1).choose (k - 1) = 165 := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_even_increasing_numbers_l3335_333539


namespace NUMINAMATH_CALUDE_correct_assignment_plans_l3335_333522

/-- The number of students in the group -/
def total_students : ℕ := 6

/-- The number of tasks to be assigned -/
def total_tasks : ℕ := 4

/-- The number of students who cannot perform the first task -/
def restricted_students : ℕ := 2

/-- Calculates the number of distinct assignment plans -/
def assignment_plans : ℕ :=
  (total_students - restricted_students) * (total_students - 1) * (total_students - 2) * (total_students - 3)

theorem correct_assignment_plans :
  assignment_plans = 240 :=
sorry

end NUMINAMATH_CALUDE_correct_assignment_plans_l3335_333522


namespace NUMINAMATH_CALUDE_rose_price_is_seven_l3335_333515

/-- Calculates the price per rose given the initial number of roses,
    remaining number of roses, and total earnings. -/
def price_per_rose (initial : ℕ) (remaining : ℕ) (earnings : ℕ) : ℚ :=
  earnings / (initial - remaining)

/-- Proves that the price per rose is 7 dollars given the problem conditions. -/
theorem rose_price_is_seven :
  price_per_rose 9 4 35 = 7 := by
  sorry

end NUMINAMATH_CALUDE_rose_price_is_seven_l3335_333515
