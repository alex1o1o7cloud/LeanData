import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l165_16595

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_proof (k b : ℝ) :
  let a : ℕ → ℝ := λ n => k * n + b
  is_arithmetic_sequence a ∧ 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d) ∧ 
  (∀ n : ℕ, a (n + 1) - a n = k) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l165_16595


namespace NUMINAMATH_CALUDE_xyz_sum_reciprocal_l165_16512

theorem xyz_sum_reciprocal (x y z : ℝ) 
  (hpos_x : x > 0) (hpos_y : y > 0) (hpos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_sum_x : x + 1 / z = 7)
  (h_sum_y : y + 1 / x = 31) :
  z + 1 / y = 5 / 27 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_reciprocal_l165_16512


namespace NUMINAMATH_CALUDE_committee_formation_count_l165_16526

def total_members : ℕ := 12
def committee_size : ℕ := 5
def incompatible_members : ℕ := 2

theorem committee_formation_count :
  (Nat.choose total_members committee_size) -
  (Nat.choose (total_members - incompatible_members) (committee_size - incompatible_members)) = 672 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l165_16526


namespace NUMINAMATH_CALUDE_total_amount_cows_and_goats_l165_16577

/-- The total amount spent on cows and goats -/
def total_amount (num_cows num_goats cow_price goat_price : ℕ) : ℕ :=
  num_cows * cow_price + num_goats * goat_price

/-- Theorem: The total amount spent on 2 cows at Rs. 460 each and 8 goats at Rs. 60 each is Rs. 1400 -/
theorem total_amount_cows_and_goats :
  total_amount 2 8 460 60 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_cows_and_goats_l165_16577


namespace NUMINAMATH_CALUDE_inequality_proof_l165_16571

def f (x : ℝ) : ℝ := 2 * abs (x - 1) + x - 1

def g (x : ℝ) : ℝ := 16 * x^2 - 8 * x + 1

def M : Set ℝ := {x | f x ≤ 1}

def N : Set ℝ := {x | g x ≤ 4}

theorem inequality_proof (x : ℝ) (hx : x ∈ M ∩ N) : x^2 * f x + x * (f x)^2 ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l165_16571


namespace NUMINAMATH_CALUDE_problem_solution_l165_16520

theorem problem_solution (s P k : ℝ) (h : P = s / Real.sqrt ((1 + k) ^ n)) :
  n = (2 * Real.log (s / P)) / Real.log (1 + k) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l165_16520


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l165_16554

theorem recurring_decimal_to_fraction : 
  ∀ (x : ℚ), (∃ (n : ℕ), x = 3 + 7 / 9 * (1 / 10^n)) → x = 34 / 9 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l165_16554


namespace NUMINAMATH_CALUDE_chalkboard_area_l165_16502

theorem chalkboard_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 3.5 →
  length = 2.3 * width →
  area = length * width →
  area = 28.175 := by
sorry

end NUMINAMATH_CALUDE_chalkboard_area_l165_16502


namespace NUMINAMATH_CALUDE_interest_tax_rate_proof_l165_16511

/-- The tax rate for interest tax on savings deposits in China --/
def interest_tax_rate : ℝ := 0.20

theorem interest_tax_rate_proof (initial_deposit : ℝ) (interest_rate : ℝ) (total_received : ℝ)
  (h1 : initial_deposit = 10000)
  (h2 : interest_rate = 0.0225)
  (h3 : total_received = 10180) :
  initial_deposit + initial_deposit * interest_rate * (1 - interest_tax_rate) = total_received :=
by sorry

end NUMINAMATH_CALUDE_interest_tax_rate_proof_l165_16511


namespace NUMINAMATH_CALUDE_all_pies_have_ingredients_l165_16598

theorem all_pies_have_ingredients (total_pies : ℕ) 
  (blueberry_fraction : ℚ) (strawberry_fraction : ℚ) 
  (raspberry_fraction : ℚ) (almond_fraction : ℚ) : 
  total_pies = 48 →
  blueberry_fraction = 1/3 →
  strawberry_fraction = 3/8 →
  raspberry_fraction = 1/2 →
  almond_fraction = 1/4 →
  ∃ (blueberry strawberry raspberry almond : Finset (Fin total_pies)),
    (blueberry.card : ℚ) ≥ blueberry_fraction * total_pies ∧
    (strawberry.card : ℚ) ≥ strawberry_fraction * total_pies ∧
    (raspberry.card : ℚ) ≥ raspberry_fraction * total_pies ∧
    (almond.card : ℚ) ≥ almond_fraction * total_pies ∧
    (blueberry ∪ strawberry ∪ raspberry ∪ almond).card = total_pies :=
by sorry

end NUMINAMATH_CALUDE_all_pies_have_ingredients_l165_16598


namespace NUMINAMATH_CALUDE_equal_sets_implies_a_equals_one_l165_16529

theorem equal_sets_implies_a_equals_one (a : ℝ) : 
  ({2, -1} : Set ℝ) = {2, a^2 - 2*a} → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_sets_implies_a_equals_one_l165_16529


namespace NUMINAMATH_CALUDE_inverse_proportion_l165_16568

theorem inverse_proportion (x y : ℝ) (h : x ≠ 0) : 
  (3 * x * y = 1) ↔ ∃ k : ℝ, k ≠ 0 ∧ y = k / x := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l165_16568


namespace NUMINAMATH_CALUDE_number_of_pickers_l165_16510

theorem number_of_pickers (drums_per_day : ℕ) (total_days : ℕ) (total_drums : ℕ) :
  drums_per_day = 221 →
  total_days = 77 →
  total_drums = 17017 →
  drums_per_day * total_days = total_drums →
  drums_per_day = 221 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_pickers_l165_16510


namespace NUMINAMATH_CALUDE_max_pages_copied_l165_16506

/-- The cost in cents to copy 4 pages -/
def cost_per_4_pages : ℚ := 7

/-- The budget in dollars -/
def budget : ℚ := 15

/-- The number of pages that can be copied with the given budget -/
def pages_copied : ℕ := 857

/-- Theorem stating the maximum number of complete pages that can be copied -/
theorem max_pages_copied : 
  ⌊(budget * 100 / cost_per_4_pages) * 4⌋ = pages_copied :=
sorry

end NUMINAMATH_CALUDE_max_pages_copied_l165_16506


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l165_16585

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l165_16585


namespace NUMINAMATH_CALUDE_calculate_expression_l165_16560

theorem calculate_expression : (π - 1) ^ 0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + |(-3)| = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l165_16560


namespace NUMINAMATH_CALUDE_baseball_hits_theorem_l165_16507

theorem baseball_hits_theorem (total_hits home_runs triples doubles : ℕ) 
  (h1 : total_hits = 50)
  (h2 : home_runs = 2)
  (h3 : triples = 3)
  (h4 : doubles = 10) :
  let singles := total_hits - (home_runs + triples + doubles)
  let percentage := (singles : ℚ) / total_hits * 100
  singles = 35 ∧ percentage = 70 := by
sorry

end NUMINAMATH_CALUDE_baseball_hits_theorem_l165_16507


namespace NUMINAMATH_CALUDE_arrangement_satisfies_condition_l165_16508

def arrangement : List ℕ := [3, 1, 4, 1, 3, 0, 2, 4, 2, 0]

def count_between (list : List ℕ) (n : ℕ) : ℕ :=
  match list.indexOf? n, list.reverse.indexOf? n with
  | some i, some j => list.length - i - j - 2
  | _, _ => 0

def satisfies_condition (list : List ℕ) : Prop :=
  ∀ n ∈ list, count_between list n = n

theorem arrangement_satisfies_condition : 
  satisfies_condition arrangement :=
sorry

end NUMINAMATH_CALUDE_arrangement_satisfies_condition_l165_16508


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l165_16537

theorem complex_expression_simplification :
  (7 - 3 * Complex.I) - 3 * (2 + 4 * Complex.I) + (1 - Complex.I) * (3 + Complex.I) = 5 - 17 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l165_16537


namespace NUMINAMATH_CALUDE_mod_sum_powers_l165_16564

theorem mod_sum_powers (n : ℕ) : (44^1234 + 99^567) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mod_sum_powers_l165_16564


namespace NUMINAMATH_CALUDE_pancakes_needed_l165_16546

/-- Given a family of 8 people and 12 pancakes already made, prove that 4 more pancakes are needed for everyone to have a second pancake. -/
theorem pancakes_needed (family_size : ℕ) (pancakes_made : ℕ) : 
  family_size = 8 → pancakes_made = 12 → 
  (family_size * 2 - pancakes_made : ℕ) = 4 := by sorry

end NUMINAMATH_CALUDE_pancakes_needed_l165_16546


namespace NUMINAMATH_CALUDE_fabric_price_system_l165_16527

/-- Represents the price per foot of damask fabric in wen -/
def damask_price : ℝ := sorry

/-- Represents the price per foot of gauze fabric in wen -/
def gauze_price : ℝ := sorry

/-- The length of the damask fabric in feet -/
def damask_length : ℝ := 7

/-- The length of the gauze fabric in feet -/
def gauze_length : ℝ := 9

/-- The price difference per foot between damask and gauze fabrics in wen -/
def price_difference : ℝ := 36

theorem fabric_price_system :
  (damask_length * damask_price = gauze_length * gauze_price) ∧
  (damask_price - gauze_price = price_difference) := by sorry

end NUMINAMATH_CALUDE_fabric_price_system_l165_16527


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l165_16550

-- Define the slope of the given line
def m : ℚ := 1/2

-- Define the given line
def given_line (x : ℚ) : ℚ := m * x - 1

-- Define the point that the new line passes through
def point : ℚ × ℚ := (1, 0)

-- Define the equation of the new line
def new_line (x : ℚ) : ℚ := m * x - 1/2

theorem parallel_line_through_point :
  (∀ x, new_line x - new_line point.1 = m * (x - point.1)) ∧
  new_line point.1 = point.2 ∧
  ∀ x, new_line x - given_line x = new_line 0 - given_line 0 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l165_16550


namespace NUMINAMATH_CALUDE_exists_arithmetic_right_triangle_with_81_l165_16597

/-- A right triangle with integer side lengths forming an arithmetic sequence -/
structure ArithmeticRightTriangle where
  a : ℕ
  d : ℕ
  right_triangle : a^2 + (a + d)^2 = (a + 2*d)^2
  arithmetic_sequence : True

/-- The existence of an arithmetic right triangle with one side length equal to 81 -/
theorem exists_arithmetic_right_triangle_with_81 :
  ∃ (t : ArithmeticRightTriangle), t.a = 81 ∨ t.a + t.d = 81 ∨ t.a + 2*t.d = 81 := by
  sorry

end NUMINAMATH_CALUDE_exists_arithmetic_right_triangle_with_81_l165_16597


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l165_16509

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l m : Line) (α β : Plane)
  (h_diff_lines : l ≠ m)
  (h_diff_planes : α ≠ β)
  (h_parallel : parallel_line_plane l α)
  (h_perpendicular : perpendicular_line_plane l β) :
  perpendicular_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l165_16509


namespace NUMINAMATH_CALUDE_probability_five_digit_palindrome_div_11_l165_16587

-- Define a five-digit palindrome
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ ∃ a b c : ℕ, 
    a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = 10000 * a + 1000 * b + 100 * c + 10 * b + a

-- Define divisibility by 11
def divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

-- Count of five-digit palindromes
def count_five_digit_palindromes : ℕ := 900

-- Count of five-digit palindromes divisible by 11
def count_five_digit_palindromes_div_11 : ℕ := 90

-- Theorem statement
theorem probability_five_digit_palindrome_div_11 :
  (count_five_digit_palindromes_div_11 : ℚ) / count_five_digit_palindromes = 1 / 10 :=
sorry

end NUMINAMATH_CALUDE_probability_five_digit_palindrome_div_11_l165_16587


namespace NUMINAMATH_CALUDE_diagonal_cubes_180_270_360_l165_16525

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_on_diagonal (a b c : ℕ) : ℕ :=
  a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + Nat.gcd a (Nat.gcd b c)

/-- Theorem: The internal diagonal of a 180 × 270 × 360 rectangular solid passes through 540 cubes -/
theorem diagonal_cubes_180_270_360 :
  cubes_on_diagonal 180 270 360 = 540 := by sorry

end NUMINAMATH_CALUDE_diagonal_cubes_180_270_360_l165_16525


namespace NUMINAMATH_CALUDE_common_chord_length_l165_16556

theorem common_chord_length (r : ℝ) (d : ℝ) (h1 : r = 12) (h2 : d = 16) :
  let chord_length := 2 * Real.sqrt (r^2 - (d/2)^2)
  chord_length = 8 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_common_chord_length_l165_16556


namespace NUMINAMATH_CALUDE_fruit_store_problem_l165_16584

/-- The price of apples in yuan per kg -/
def apple_price : ℝ := 8

/-- The price of pears in yuan per kg -/
def pear_price : ℝ := 6

/-- The maximum number of kg of apples that can be purchased -/
def max_apple_kg : ℝ := 5

theorem fruit_store_problem :
  (∀ x y : ℝ, x + 3 * y = 26 ∧ 2 * x + y = 22 →
    x = apple_price ∧ y = pear_price) ∧
  (∀ m : ℝ, 8 * m + 6 * (15 - m) ≤ 100 → m ≤ max_apple_kg) :=
by sorry

end NUMINAMATH_CALUDE_fruit_store_problem_l165_16584


namespace NUMINAMATH_CALUDE_corn_increase_factor_l165_16542

theorem corn_increase_factor (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x < 1) 
  (h3 : 1 - x + x = 1/2) 
  (h4 : 1 - x + x/2 = 1/2) : 
  (3/2 * x) / (1/2 * x) = 3 := by sorry

end NUMINAMATH_CALUDE_corn_increase_factor_l165_16542


namespace NUMINAMATH_CALUDE_bill_denomination_l165_16588

-- Define the problem parameters
def total_bill : ℕ := 285
def coin_value : ℕ := 5
def total_items : ℕ := 24
def num_bills : ℕ := 11
def num_coins : ℕ := 11

-- Theorem to prove
theorem bill_denomination :
  ∃ (x : ℕ), 
    x * num_bills + coin_value * num_coins = total_bill ∧
    num_bills + num_coins = total_items ∧
    x = 20 := by
  sorry

end NUMINAMATH_CALUDE_bill_denomination_l165_16588


namespace NUMINAMATH_CALUDE_smallest_a_for_equation_l165_16572

theorem smallest_a_for_equation : ∃ (p : ℕ) (b : ℕ), 
  Nat.Prime p ∧ 
  b ≥ 2 ∧ 
  (9^p - 9) / p = b^2 ∧ 
  ∀ (a : ℕ) (q : ℕ) (c : ℕ), 
    a > 0 ∧ a < 9 → 
    Nat.Prime q → 
    c ≥ 2 → 
    (a^q - a) / q ≠ c^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_equation_l165_16572


namespace NUMINAMATH_CALUDE_sarah_copies_3600_pages_l165_16523

/-- The total number of pages Sarah will copy for two contracts -/
def total_pages (num_people : ℕ) (contract1_pages : ℕ) (contract1_copies : ℕ) 
                (contract2_pages : ℕ) (contract2_copies : ℕ) : ℕ :=
  num_people * (contract1_pages * contract1_copies + contract2_pages * contract2_copies)

/-- Theorem: Sarah will copy 3600 pages in total -/
theorem sarah_copies_3600_pages : 
  total_pages 20 30 3 45 2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_sarah_copies_3600_pages_l165_16523


namespace NUMINAMATH_CALUDE_sum_between_bounds_l165_16581

theorem sum_between_bounds : 
  (21/2 : ℚ) < (15/7 : ℚ) + (7/2 : ℚ) + (96/19 : ℚ) ∧ 
  (15/7 : ℚ) + (7/2 : ℚ) + (96/19 : ℚ) < 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_between_bounds_l165_16581


namespace NUMINAMATH_CALUDE_arrival_time_difference_l165_16539

/-- Represents the distance to the pool in miles -/
def distance_to_pool : ℝ := 3

/-- Represents Jill's speed in miles per hour -/
def jill_speed : ℝ := 12

/-- Represents Jack's speed in miles per hour -/
def jack_speed : ℝ := 3

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

/-- Calculates the time difference in minutes between Jill and Jack's arrival at the pool -/
theorem arrival_time_difference : 
  hours_to_minutes (distance_to_pool / jill_speed - distance_to_pool / jack_speed) = 45 := by
  sorry

end NUMINAMATH_CALUDE_arrival_time_difference_l165_16539


namespace NUMINAMATH_CALUDE_cube_surface_area_l165_16532

/-- Given three points A, B, and C as vertices of a cube, prove that its surface area is 294 -/
theorem cube_surface_area (A B C : ℝ × ℝ × ℝ) : 
  A = (1, 4, 2) → B = (2, 0, -7) → C = (5, -5, 1) → 
  (let surface_area := 6 * (dist A B)^2
   surface_area = 294) := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l165_16532


namespace NUMINAMATH_CALUDE_constant_function_l165_16596

theorem constant_function (α : ℝ) (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.cos x ^ 2 + Real.cos (x + α) ^ 2 - 2 * Real.cos α * Real.cos x * Real.cos (x + α)
  f x = (1 - Real.cos (2 * α)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_constant_function_l165_16596


namespace NUMINAMATH_CALUDE_stock_percentage_is_25_percent_l165_16538

/-- Calculates the percentage of a stock given the investment amount and income. -/
def stock_percentage (investment income : ℚ) : ℚ :=
  (income * 100) / investment

/-- Theorem stating that the stock percentage is 25% given the specified investment and income. -/
theorem stock_percentage_is_25_percent (investment : ℚ) (income : ℚ) 
  (h1 : investment = 15200)
  (h2 : income = 3800) :
  stock_percentage investment income = 25 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_is_25_percent_l165_16538


namespace NUMINAMATH_CALUDE_total_spent_proof_l165_16548

def jayda_stall1 : ℝ := 400
def jayda_stall2 : ℝ := 120
def jayda_stall3 : ℝ := 250
def aitana_multiplier : ℝ := 1.4 -- 1 + 2/5
def jayda_discount1 : ℝ := 0.05
def aitana_discount2 : ℝ := 0.10
def sales_tax : ℝ := 0.10
def exchange_rate : ℝ := 1.25

def total_spent_cad : ℝ :=
  ((jayda_stall1 * (1 - jayda_discount1) + jayda_stall2 + jayda_stall3) * (1 + sales_tax) +
   (jayda_stall1 * aitana_multiplier + 
    jayda_stall2 * aitana_multiplier * (1 - aitana_discount2) + 
    jayda_stall3 * aitana_multiplier) * (1 + sales_tax)) * exchange_rate

theorem total_spent_proof : total_spent_cad = 2490.40 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_proof_l165_16548


namespace NUMINAMATH_CALUDE_scaled_circle_equation_l165_16541

/-- Given a circle and a scaling transformation, prove the equation of the resulting curve -/
theorem scaled_circle_equation (x y x' y' : ℝ) :
  (x^2 + y^2 = 1) →  -- Circle equation
  (x' = 2*x) →       -- Scaling for x
  (y' = 3*y) →       -- Scaling for y
  (x'^2/4 + y'^2/9 = 1) -- Resulting curve equation
:= by sorry

end NUMINAMATH_CALUDE_scaled_circle_equation_l165_16541


namespace NUMINAMATH_CALUDE_system_solution_l165_16519

theorem system_solution :
  ∃! (x y : ℝ), 3 * x - 2 * y = 6 ∧ 2 * x + 3 * y = 17 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_l165_16519


namespace NUMINAMATH_CALUDE_juice_packs_fit_l165_16500

/-- The number of juice packs that can fit in a box without gaps -/
def juice_packs_in_box (box_width box_length box_height juice_width juice_length juice_height : ℕ) : ℕ :=
  (box_width * box_length * box_height) / (juice_width * juice_length * juice_height)

/-- Theorem stating that 72 juice packs fit in the given box -/
theorem juice_packs_fit :
  juice_packs_in_box 24 15 28 4 5 7 = 72 := by
  sorry

#eval juice_packs_in_box 24 15 28 4 5 7

end NUMINAMATH_CALUDE_juice_packs_fit_l165_16500


namespace NUMINAMATH_CALUDE_quadratic_solution_implies_sum_l165_16557

theorem quadratic_solution_implies_sum (a b : ℝ) : 
  (a * 2^2 - b * 2 + 2 = 0) → (2024 + 2*a - b = 2023) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implies_sum_l165_16557


namespace NUMINAMATH_CALUDE_translation_of_line_segment_l165_16545

/-- Given a line segment AB with endpoints A(1,0) and B(3,2), if it is translated to a new position
    where the new endpoints are A₁(a,1) and B₁(4,b), then a = 2 and b = 3. -/
theorem translation_of_line_segment (a b : ℝ) : 
  (∃ (dx dy : ℝ), (1 + dx = a ∧ 0 + dy = 1) ∧ (3 + dx = 4 ∧ 2 + dy = b)) → 
  (a = 2 ∧ b = 3) :=
by sorry

end NUMINAMATH_CALUDE_translation_of_line_segment_l165_16545


namespace NUMINAMATH_CALUDE_actual_distance_is_82_l165_16515

/-- Represents the distance between two towns on a map --/
def map_distance : ℝ := 9

/-- Represents the initial scale of the map --/
def initial_scale : ℝ := 10

/-- Represents the subsequent scale of the map --/
def subsequent_scale : ℝ := 8

/-- Represents the distance on the map where the initial scale applies --/
def initial_scale_distance : ℝ := 5

/-- Calculates the actual distance between two towns given the map distance and scales --/
def actual_distance : ℝ :=
  initial_scale * initial_scale_distance +
  subsequent_scale * (map_distance - initial_scale_distance)

/-- Theorem stating that the actual distance between the towns is 82 miles --/
theorem actual_distance_is_82 : actual_distance = 82 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_is_82_l165_16515


namespace NUMINAMATH_CALUDE_barbara_scrap_paper_heaps_l165_16552

/-- Represents the number of sheets in a bundle of paper -/
def sheets_per_bundle : ℕ := 2

/-- Represents the number of sheets in a bunch of paper -/
def sheets_per_bunch : ℕ := 4

/-- Represents the number of sheets in a heap of paper -/
def sheets_per_heap : ℕ := 20

/-- Represents the number of bundles of colored paper Barbara found -/
def colored_bundles : ℕ := 3

/-- Represents the number of bunches of white paper Barbara found -/
def white_bunches : ℕ := 2

/-- Represents the total number of sheets Barbara removed -/
def total_sheets_removed : ℕ := 114

/-- Theorem stating the number of heaps of scrap paper Barbara found -/
theorem barbara_scrap_paper_heaps :
  (total_sheets_removed - (colored_bundles * sheets_per_bundle + white_bunches * sheets_per_bunch)) / sheets_per_heap = 5 := by
  sorry

end NUMINAMATH_CALUDE_barbara_scrap_paper_heaps_l165_16552


namespace NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_l165_16569

/-- The nth positive integer that is both odd and a multiple of 5 -/
def oddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

/-- Prove that the 15th positive integer that is both odd and a multiple of 5 is 145 -/
theorem fifteenth_odd_multiple_of_5 : oddMultipleOf5 15 = 145 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_l165_16569


namespace NUMINAMATH_CALUDE_village_cats_l165_16516

theorem village_cats (total : ℕ) (spotted : ℕ) (fluffy_spotted : ℕ) : 
  spotted = total / 3 →
  fluffy_spotted = spotted / 4 →
  fluffy_spotted = 10 →
  total = 120 := by
sorry

end NUMINAMATH_CALUDE_village_cats_l165_16516


namespace NUMINAMATH_CALUDE_binomial_150_150_l165_16505

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_l165_16505


namespace NUMINAMATH_CALUDE_kara_forgotten_doses_l165_16504

/-- The number of times Kara takes medication per day -/
def doses_per_day : ℕ := 3

/-- The amount of water in ounces Kara drinks with each dose -/
def water_per_dose : ℕ := 4

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total amount of water in ounces Kara drank with her medication over two weeks -/
def total_water_drunk : ℕ := 160

/-- The number of times Kara forgot to take her medication on one day in the second week -/
def forgotten_doses : ℕ := 2

theorem kara_forgotten_doses :
  (doses_per_day * water_per_dose * days_in_week * 2) - total_water_drunk = forgotten_doses * water_per_dose :=
by sorry

end NUMINAMATH_CALUDE_kara_forgotten_doses_l165_16504


namespace NUMINAMATH_CALUDE_port_distance_l165_16553

/-- The distance between two ports given travel times and current speed -/
theorem port_distance (downstream_time upstream_time current_speed : ℝ) 
  (h_downstream : downstream_time = 3)
  (h_upstream : upstream_time = 4)
  (h_current : current_speed = 5) : 
  ∃ (distance boat_speed : ℝ),
    distance = downstream_time * (boat_speed + current_speed) ∧
    distance = upstream_time * (boat_speed - current_speed) ∧
    distance = 120 := by
  sorry

end NUMINAMATH_CALUDE_port_distance_l165_16553


namespace NUMINAMATH_CALUDE_prob_at_least_one_target_l165_16578

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards that are either hearts or kings -/
def target_cards : ℕ := 16

/-- The probability of drawing a card that is not a heart or king -/
def prob_not_target : ℚ := (deck_size - target_cards) / deck_size

/-- The number of draws -/
def num_draws : ℕ := 3

/-- The probability of drawing at least one heart or king in three draws with replacement -/
theorem prob_at_least_one_target :
  1 - prob_not_target ^ num_draws = 1468 / 2197 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_target_l165_16578


namespace NUMINAMATH_CALUDE_max_digits_product_3digit_2digit_l165_16517

theorem max_digits_product_3digit_2digit :
  ∀ (a b : ℕ), 100 ≤ a ∧ a ≤ 999 ∧ 10 ≤ b ∧ b ≤ 99 →
  a * b < 100000 :=
sorry

end NUMINAMATH_CALUDE_max_digits_product_3digit_2digit_l165_16517


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l165_16576

/-- Given a sphere and a right circular cone, where:
    - The radius of the cone's base is twice the radius of the sphere
    - The volume of the cone is one-third the volume of the sphere
    Prove that the ratio of the cone's altitude to its base radius is 1/6 -/
theorem cone_sphere_ratio (r : ℝ) (h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) : 
  (4 / 3 * Real.pi * r^2 * h = 1 / 3 * (4 / 3 * Real.pi * r^3)) → 
  (h / (2 * r) = 1 / 6) := by
sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l165_16576


namespace NUMINAMATH_CALUDE_methane_required_moles_l165_16547

/-- Represents a chemical species in a reaction -/
structure ChemicalSpecies where
  formula : String
  moles : ℚ

/-- Represents a chemical reaction -/
structure ChemicalReaction where
  reactants : List ChemicalSpecies
  products : List ChemicalSpecies

def methane_chlorine_reaction : ChemicalReaction :=
  { reactants := [
      { formula := "CH4", moles := 1 },
      { formula := "Cl2", moles := 1 }
    ],
    products := [
      { formula := "CH3Cl", moles := 1 },
      { formula := "HCl", moles := 1 }
    ]
  }

/-- Theorem stating that 2 moles of CH4 are required to react with 2 moles of Cl2 -/
theorem methane_required_moles 
  (reaction : ChemicalReaction)
  (h_reaction : reaction = methane_chlorine_reaction)
  (h_cl2_moles : ∃ cl2 ∈ reaction.reactants, cl2.formula = "Cl2" ∧ cl2.moles = 2)
  (h_hcl_moles : ∃ hcl ∈ reaction.products, hcl.formula = "HCl" ∧ hcl.moles = 2) :
  ∃ ch4 ∈ reaction.reactants, ch4.formula = "CH4" ∧ ch4.moles = 2 :=
sorry

end NUMINAMATH_CALUDE_methane_required_moles_l165_16547


namespace NUMINAMATH_CALUDE_x_value_l165_16524

theorem x_value : ∃ x : ℝ, x * 0.65 = 552.50 * 0.20 ∧ x = 170 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l165_16524


namespace NUMINAMATH_CALUDE_peach_tree_count_l165_16558

theorem peach_tree_count (almond_trees : ℕ) (peach_trees : ℕ) : 
  almond_trees = 300 →
  peach_trees = 2 * almond_trees - 30 →
  peach_trees = 570 := by
sorry

end NUMINAMATH_CALUDE_peach_tree_count_l165_16558


namespace NUMINAMATH_CALUDE_total_price_theorem_l165_16535

def refrigerator_price : ℝ := 4275
def washing_machine_price : ℝ := refrigerator_price - 1490
def sales_tax_rate : ℝ := 0.07

def total_price_with_tax : ℝ :=
  (refrigerator_price + washing_machine_price) * (1 + sales_tax_rate)

theorem total_price_theorem :
  total_price_with_tax = 7554.20 := by sorry

end NUMINAMATH_CALUDE_total_price_theorem_l165_16535


namespace NUMINAMATH_CALUDE_largest_integer_k_for_real_roots_l165_16562

theorem largest_integer_k_for_real_roots : ∃ (k : ℤ),
  (∀ (j : ℤ), (∀ (x : ℝ), ∃ (y : ℝ), x * (j * x + 1) - x^2 + 3 = 0) → j ≤ k) ∧
  (∀ (x : ℝ), ∃ (y : ℝ), x * (k * x + 1) - x^2 + 3 = 0) ∧
  k = 1 :=
sorry

end NUMINAMATH_CALUDE_largest_integer_k_for_real_roots_l165_16562


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l165_16518

theorem quadratic_form_sum (x : ℝ) : ∃ (d e : ℝ), x^2 - 18*x + 81 = (x + d)^2 + e ∧ d + e = -9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l165_16518


namespace NUMINAMATH_CALUDE_solve_equation_l165_16570

theorem solve_equation : ∃ x : ℝ, (3/2 : ℝ) * x - 3 = 15 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l165_16570


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l165_16574

theorem hyperbola_asymptote_slope (m : ℝ) (α : ℝ) :
  (∀ x y : ℝ, x^2 + y^2/m = 1) →
  (0 < α ∧ α < π/3) →
  (-3 < m ∧ m < 0) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l165_16574


namespace NUMINAMATH_CALUDE_ratio_equality_l165_16555

theorem ratio_equality (a b c d : ℝ) 
  (h1 : a = 5 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 6 * d) : 
  (a + b * c) / (c + d * b) = 3 * (5 + 6 * d) / (1 + 3 * d) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l165_16555


namespace NUMINAMATH_CALUDE_sprint_medal_awards_l165_16593

/-- The number of ways to award medals in the international sprint final -/
def medal_awards (total_sprinters : ℕ) (american_sprinters : ℕ) (medals : ℕ) : ℕ :=
  let non_american_sprinters := total_sprinters - american_sprinters
  let no_american_wins := non_american_sprinters.descFactorial medals
  let one_american_wins := american_sprinters * medals * (non_american_sprinters.descFactorial (medals - 1))
  no_american_wins + one_american_wins

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem sprint_medal_awards :
  medal_awards 10 4 3 = 480 := by
  sorry

end NUMINAMATH_CALUDE_sprint_medal_awards_l165_16593


namespace NUMINAMATH_CALUDE_kate_bouncy_balls_difference_l165_16528

/-- The number of bouncy balls in each pack -/
def balls_per_pack : ℕ := 18

/-- The number of packs of red bouncy balls Kate bought -/
def red_packs : ℕ := 7

/-- The number of packs of yellow bouncy balls Kate bought -/
def yellow_packs : ℕ := 6

/-- The total number of red bouncy balls Kate bought -/
def total_red_balls : ℕ := red_packs * balls_per_pack

/-- The total number of yellow bouncy balls Kate bought -/
def total_yellow_balls : ℕ := yellow_packs * balls_per_pack

/-- The difference between the number of red and yellow bouncy balls -/
def difference_in_balls : ℕ := total_red_balls - total_yellow_balls

theorem kate_bouncy_balls_difference :
  difference_in_balls = 18 := by sorry

end NUMINAMATH_CALUDE_kate_bouncy_balls_difference_l165_16528


namespace NUMINAMATH_CALUDE_ab_value_when_sqrt_and_abs_sum_zero_l165_16549

theorem ab_value_when_sqrt_and_abs_sum_zero (a b : ℝ) :
  Real.sqrt (a - 3) + |1 - b| = 0 → a * b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_when_sqrt_and_abs_sum_zero_l165_16549


namespace NUMINAMATH_CALUDE_expression_value_l165_16592

theorem expression_value (x y : ℝ) (h : (x - y) / y = 2) :
  ((1 / (x - y) + 1 / (x + y)) / (x / (x - y)^2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l165_16592


namespace NUMINAMATH_CALUDE_legendre_symbol_values_legendre_symbol_square_equivalence_minus_one_square_mod_p_eleven_power_sum_of_squares_l165_16513

-- Define the necessary variables and functions
variable (p : ℕ) (hp : Nat.Prime p) (hodd : p % 2 = 1)
variable (a : ℕ) (hcoprime : Nat.Coprime a p)

-- Theorem 1
theorem legendre_symbol_values :
  (a ^ ((p - 1) / 2)) % p = 1 ∨ (a ^ ((p - 1) / 2)) % p = p - 1 :=
sorry

-- Theorem 2
theorem legendre_symbol_square_equivalence :
  (a ^ ((p - 1) / 2)) % p = 1 ↔ ∃ x, (x * x) % p = a % p :=
sorry

-- Theorem 3
theorem minus_one_square_mod_p :
  (∃ x, (x * x) % p = p - 1) ↔ p % 4 = 1 :=
sorry

-- Theorem 4
theorem eleven_power_sum_of_squares (n : ℕ) :
  ∀ a b : ℕ, 11^n = a^2 + b^2 →
    ∃ k : ℕ, n = 2*k ∧ ((a = 11^k ∧ b = 0) ∨ (a = 0 ∧ b = 11^k)) :=
sorry

end NUMINAMATH_CALUDE_legendre_symbol_values_legendre_symbol_square_equivalence_minus_one_square_mod_p_eleven_power_sum_of_squares_l165_16513


namespace NUMINAMATH_CALUDE_centroid_trajectory_l165_16591

/-- Given a triangle ABC with vertices A(-3, 0), B(3, 0), and C(m, n) on the parabola y² = 6x,
    the centroid (x, y) of the triangle satisfies the equation y² = 2x for x ≠ 0. -/
theorem centroid_trajectory (m n x y : ℝ) : 
  n^2 = 6*m →                   -- C is on the parabola y² = 6x
  3*x = m →                     -- x-coordinate of centroid
  3*y = n →                     -- y-coordinate of centroid
  x ≠ 0 →                       -- x is non-zero
  y^2 = 2*x                     -- equation of centroid's trajectory
  := by sorry

end NUMINAMATH_CALUDE_centroid_trajectory_l165_16591


namespace NUMINAMATH_CALUDE_school_section_problem_l165_16563

/-- Calculates the maximum number of equal-sized mixed-gender sections
    that can be formed given the number of boys and girls and the required ratio. -/
def max_sections (boys girls : ℕ) (boy_ratio girl_ratio : ℕ) : ℕ :=
  min (boys / boy_ratio) (girls / girl_ratio)

/-- Theorem stating the solution to the school section problem -/
theorem school_section_problem :
  max_sections 2040 1728 3 2 = 680 := by
  sorry

end NUMINAMATH_CALUDE_school_section_problem_l165_16563


namespace NUMINAMATH_CALUDE_complex_square_root_of_negative_four_l165_16575

theorem complex_square_root_of_negative_four (z : ℂ) : 
  z^2 = -4 ∧ z.im > 0 → z = 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_root_of_negative_four_l165_16575


namespace NUMINAMATH_CALUDE_complex_sum_imaginary_l165_16514

theorem complex_sum_imaginary (a : ℝ) : 
  let z₁ : ℂ := a^2 - 2 - 3*a*Complex.I
  let z₂ : ℂ := a + (a^2 + 2)*Complex.I
  (z₁ + z₂).re = 0 ∧ (z₁ + z₂).im ≠ 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_imaginary_l165_16514


namespace NUMINAMATH_CALUDE_value_of_N_l165_16573

theorem value_of_N : 
  let N := (Real.sqrt (Real.sqrt 5 + 2) + Real.sqrt (Real.sqrt 5 - 2)) / 
           Real.sqrt (Real.sqrt 5 + 1) - 
           Real.sqrt (3 - 2 * Real.sqrt 2)
  N = 1 := by sorry

end NUMINAMATH_CALUDE_value_of_N_l165_16573


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l165_16533

/-- A color type representing red or blue --/
inductive Color
  | Red
  | Blue

/-- A type representing a complete graph with 6 vertices --/
structure CompleteGraph6 where
  /-- A function assigning a color to each pair of distinct vertices --/
  edgeColor : Fin 6 → Fin 6 → Color
  /-- Ensure the graph is undirected --/
  symm : ∀ (i j : Fin 6), i ≠ j → edgeColor i j = edgeColor j i

/-- Definition of a monochromatic triangle in the graph --/
def hasMonochromaticTriangle (g : CompleteGraph6) : Prop :=
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    g.edgeColor i j = g.edgeColor j k ∧ g.edgeColor j k = g.edgeColor k i

/-- Theorem stating that every complete graph with 6 vertices and edges colored red or blue
    contains a monochromatic triangle --/
theorem monochromatic_triangle_exists (g : CompleteGraph6) : hasMonochromaticTriangle g :=
  sorry


end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l165_16533


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l165_16503

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l165_16503


namespace NUMINAMATH_CALUDE_original_denominator_proof_l165_16594

theorem original_denominator_proof (d : ℚ) : 
  (2 : ℚ) / d ≠ (1 : ℚ) / 2 ∧ (2 + 5 : ℚ) / (d + 5) = (1 : ℚ) / 2 → d = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l165_16594


namespace NUMINAMATH_CALUDE_cistern_filling_time_l165_16567

theorem cistern_filling_time (t : ℝ) : t > 0 → 
  (4 * (1 / t + 1 / 18) + 8 / 18 = 1) → t = 12 := by sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l165_16567


namespace NUMINAMATH_CALUDE_brianna_cd_purchase_l165_16561

theorem brianna_cd_purchase (m : ℚ) (c : ℚ) (n : ℚ) (h1 : m > 0) (h2 : c > 0) (h3 : n > 0) :
  (1 / 4 : ℚ) * m = (1 / 4 : ℚ) * n * c →
  m - n * c = 0 :=
by sorry

end NUMINAMATH_CALUDE_brianna_cd_purchase_l165_16561


namespace NUMINAMATH_CALUDE_min_cost_to_1981_l165_16540

/-- Cost of multiplying by 3 -/
def mult_cost : ℕ := 5

/-- Cost of adding 4 -/
def add_cost : ℕ := 2

/-- The target number to reach -/
def target : ℕ := 1981

/-- A step in the calculation process -/
inductive Step
| Mult : Step  -- Multiply by 3
| Add : Step   -- Add 4

/-- A sequence of steps -/
def Sequence := List Step

/-- Calculate the result of applying a sequence of steps starting from 1 -/
def apply_sequence (s : Sequence) : ℕ :=
  s.foldl (λ n step => match step with
    | Step.Mult => n * 3
    | Step.Add => n + 4) 1

/-- Calculate the cost of a sequence of steps -/
def sequence_cost (s : Sequence) : ℕ :=
  s.foldl (λ cost step => cost + match step with
    | Step.Mult => mult_cost
    | Step.Add => add_cost) 0

/-- Theorem: The minimum cost to reach 1981 is 42 kopecks -/
theorem min_cost_to_1981 :
  ∃ (s : Sequence), apply_sequence s = target ∧
    sequence_cost s = 42 ∧
    ∀ (s' : Sequence), apply_sequence s' = target →
      sequence_cost s' ≥ sequence_cost s :=
by sorry

end NUMINAMATH_CALUDE_min_cost_to_1981_l165_16540


namespace NUMINAMATH_CALUDE_first_grade_students_l165_16599

theorem first_grade_students (total : ℕ) (difference : ℕ) (first_grade : ℕ) : 
  total = 1256 → 
  difference = 408 →
  first_grade + difference = total - first_grade →
  first_grade = 424 := by
sorry

end NUMINAMATH_CALUDE_first_grade_students_l165_16599


namespace NUMINAMATH_CALUDE_square_divides_power_plus_one_l165_16530

theorem square_divides_power_plus_one (n : ℕ) : 
  n ^ 2 ∣ 2 ^ n + 1 ↔ n = 1 ∨ n = 3 := by
sorry

end NUMINAMATH_CALUDE_square_divides_power_plus_one_l165_16530


namespace NUMINAMATH_CALUDE_valid_mixture_weight_l165_16543

/-- A cement mixture composed of sand, water, and gravel -/
structure CementMixture where
  total_weight : ℝ
  sand_fraction : ℝ
  water_fraction : ℝ
  gravel_weight : ℝ

/-- The cement mixture satisfies the given conditions -/
def is_valid_mixture (m : CementMixture) : Prop :=
  m.sand_fraction = 1/3 ∧
  m.water_fraction = 1/4 ∧
  m.gravel_weight = 10 ∧
  m.sand_fraction * m.total_weight + m.water_fraction * m.total_weight + m.gravel_weight = m.total_weight

/-- The theorem stating that a valid mixture has a total weight of 24 pounds -/
theorem valid_mixture_weight (m : CementMixture) (h : is_valid_mixture m) : m.total_weight = 24 := by
  sorry

end NUMINAMATH_CALUDE_valid_mixture_weight_l165_16543


namespace NUMINAMATH_CALUDE_fraction_2011_l165_16522

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The sequence of fractions as described in the problem -/
def fraction_sequence : ℕ → ℚ := sorry

/-- The sum of Euler's totient function up to n -/
def phi_sum (n : ℕ) : ℕ := sorry

theorem fraction_2011 : fraction_sequence 2011 = 49 / 111 := by sorry

end NUMINAMATH_CALUDE_fraction_2011_l165_16522


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l165_16551

/-- A hyperbola passing through the point (3, -√2) with eccentricity √5/2 has the standard equation x²/1 - y²/(1/4) = 1 -/
theorem hyperbola_standard_equation (x y a b : ℝ) : 
  (x = 3 ∧ y = -Real.sqrt 2) →  -- Point on the hyperbola
  (Real.sqrt 5 / 2 = Real.sqrt (a^2 + b^2) / a) →  -- Eccentricity
  (x^2 / a^2 - y^2 / b^2 = 1) →  -- General equation of hyperbola
  (a^2 = 1 ∧ b^2 = 1/4) :=  -- Standard equation coefficients
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l165_16551


namespace NUMINAMATH_CALUDE_existence_of_points_with_derivative_sum_zero_l165_16566

theorem existence_of_points_with_derivative_sum_zero
  {f : ℝ → ℝ} {a b : ℝ} (h_diff : DifferentiableOn ℝ f (Set.Icc a b))
  (h_eq : f a = f b) (h_lt : a < b) :
  ∃ x y, x ∈ Set.Icc a b ∧ y ∈ Set.Icc a b ∧ x ≠ y ∧
    (deriv f x) + 5 * (deriv f y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_points_with_derivative_sum_zero_l165_16566


namespace NUMINAMATH_CALUDE_biking_distance_l165_16583

/-- Calculates the distance traveled given a constant rate and time -/
def distance (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Proves that biking at 8 miles per hour for 2.5 hours results in a distance of 20 miles -/
theorem biking_distance :
  let rate : ℝ := 8
  let time : ℝ := 2.5
  distance rate time = 20 := by sorry

end NUMINAMATH_CALUDE_biking_distance_l165_16583


namespace NUMINAMATH_CALUDE_factorization_equality_l165_16579

theorem factorization_equality (a x : ℝ) : -a*x^2 + 2*a*x - a = -a*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l165_16579


namespace NUMINAMATH_CALUDE_vanessa_deleted_files_l165_16536

/-- Calculates the number of deleted files given the initial number of music and video files and the number of files left after deletion. -/
def deleted_files (music_files : ℕ) (video_files : ℕ) (files_left : ℕ) : ℕ :=
  music_files + video_files - files_left

/-- Proves that the number of deleted files is 10 given the specific conditions in the problem. -/
theorem vanessa_deleted_files :
  deleted_files 13 30 33 = 10 := by
  sorry

#eval deleted_files 13 30 33

end NUMINAMATH_CALUDE_vanessa_deleted_files_l165_16536


namespace NUMINAMATH_CALUDE_parallelogram_area_calculation_l165_16531

-- Define the parallelogram
def parallelogram_base : ℝ := 32
def parallelogram_height : ℝ := 14

-- Define the area formula for a parallelogram
def parallelogram_area (base height : ℝ) : ℝ := base * height

-- Theorem statement
theorem parallelogram_area_calculation :
  parallelogram_area parallelogram_base parallelogram_height = 448 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_calculation_l165_16531


namespace NUMINAMATH_CALUDE_larger_cuboid_length_l165_16582

/-- Proves that the length of a larger cuboid is 18m, given the specified conditions --/
theorem larger_cuboid_length : 
  ∀ (small_length small_width small_height : ℝ)
    (large_width large_height : ℝ)
    (num_small_cuboids : ℕ),
  small_length = 5 →
  small_width = 6 →
  small_height = 3 →
  large_width = 15 →
  large_height = 2 →
  num_small_cuboids = 6 →
  ∃ (large_length : ℝ),
    large_length * large_width * large_height = 
    num_small_cuboids * (small_length * small_width * small_height) ∧
    large_length = 18 := by
sorry


end NUMINAMATH_CALUDE_larger_cuboid_length_l165_16582


namespace NUMINAMATH_CALUDE_sodas_per_pack_james_sodas_problem_l165_16521

theorem sodas_per_pack (packs : ℕ) (initial_sodas : ℕ) (days_in_week : ℕ) (sodas_per_day : ℕ) : ℕ :=
  let total_sodas := sodas_per_day * days_in_week
  let new_sodas := total_sodas - initial_sodas
  new_sodas / packs

theorem james_sodas_problem : sodas_per_pack 5 10 7 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sodas_per_pack_james_sodas_problem_l165_16521


namespace NUMINAMATH_CALUDE_rationalize_cube_root_seven_l165_16534

def rationalize_denominator (a b : ℕ) : ℚ × ℕ × ℕ := sorry

theorem rationalize_cube_root_seven :
  let (frac, B, C) := rationalize_denominator 4 (3 * 7^(1/3))
  frac = 4 * (49^(1/3)) / 21 ∧ 
  B = 49 ∧ 
  C = 21 ∧
  4 + B + C = 74 := by sorry

end NUMINAMATH_CALUDE_rationalize_cube_root_seven_l165_16534


namespace NUMINAMATH_CALUDE_removed_triangles_area_l165_16565

/-- The combined area of four isosceles right triangles removed from the corners of a square
    with side length 20 units to form a regular octagon is 512 square units. -/
theorem removed_triangles_area (square_side : ℝ) (triangle_leg : ℝ) : 
  square_side = 20 →
  (square_side - 2 * triangle_leg)^2 + (triangle_leg - (square_side - 2 * triangle_leg))^2 = square_side^2 →
  4 * (1/2 * triangle_leg^2) = 512 :=
by sorry

end NUMINAMATH_CALUDE_removed_triangles_area_l165_16565


namespace NUMINAMATH_CALUDE_f_symmetry_l165_16559

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 1

-- State the theorem
theorem f_symmetry (a b : ℝ) : f a b 2 = -1 → f a b (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l165_16559


namespace NUMINAMATH_CALUDE_max_perpendicular_faces_theorem_l165_16590

/-- The maximum number of lateral faces of an n-sided pyramid that can be perpendicular to the base -/
def max_perpendicular_faces (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else (n + 1) / 2

/-- Theorem stating the maximum number of lateral faces of an n-sided pyramid that can be perpendicular to the base -/
theorem max_perpendicular_faces_theorem (n : ℕ) (h : n > 0) :
  max_perpendicular_faces n = 
    if n % 2 = 0 
    then n / 2 
    else (n + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_perpendicular_faces_theorem_l165_16590


namespace NUMINAMATH_CALUDE_triangle_properties_l165_16586

/-- Given a triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h_area : (1/2) * t.a * t.c * Real.sin t.B = (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2))
  (h_obtuse : t.C > π/2) :
  t.B = π/3 ∧ ∃ (k : ℝ), k > 2 ∧ t.c / t.a = k :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l165_16586


namespace NUMINAMATH_CALUDE_geometric_sum_first_seven_l165_16580

-- Define the geometric sequence
def geometric_sequence (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₀ * r^n

-- Define the sum of the first n terms of the geometric sequence
def geometric_sum (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then n * a₀ else a₀ * (1 - r^n) / (1 - r)

theorem geometric_sum_first_seven :
  let a₀ : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 7
  geometric_sum a₀ r n = 1093/2187 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sum_first_seven_l165_16580


namespace NUMINAMATH_CALUDE_distance_inequality_l165_16544

-- Define the types for planes, lines, and points
variable (Plane Line Point : Type)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the "in" relation for lines and planes
variable (in_plane : Line → Plane → Prop)

-- Define the "on" relation for points and lines
variable (on_line : Point → Line → Prop)

-- Define the distance function
variable (distance : Point → Point → ℝ)
variable (distance_point_to_line : Point → Line → ℝ)
variable (distance_line_to_line : Line → Line → ℝ)

-- Define the specific objects in our problem
variable (α β : Plane) (m n : Line) (A B : Point)

-- Define the theorem
theorem distance_inequality 
  (h_parallel : parallel α β)
  (h_m_in_α : in_plane m α)
  (h_n_in_β : in_plane n β)
  (h_A_on_m : on_line A m)
  (h_B_on_n : on_line B n)
  (h_a : distance A B = a)
  (h_b : distance_point_to_line A n = b)
  (h_c : distance_line_to_line m n = c)
  : c ≤ a ∧ a ≤ b :=
by sorry

end NUMINAMATH_CALUDE_distance_inequality_l165_16544


namespace NUMINAMATH_CALUDE_largest_a_for_integer_solution_l165_16501

theorem largest_a_for_integer_solution :
  ∃ (a : ℝ), ∀ (b : ℝ),
    (∃ (x y : ℤ), x - 4*y = 1 ∧ a*x + 3*y = 1) ∧
    (∀ (x y : ℤ), b > a → ¬(x - 4*y = 1 ∧ b*x + 3*y = 1)) →
    a = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_a_for_integer_solution_l165_16501


namespace NUMINAMATH_CALUDE_t_value_l165_16589

/-- Linear regression equation for the given data points -/
def linear_regression (x : ℝ) : ℝ := 1.04 * x + 1.9

/-- The value of t in the data set (4, t) -/
def t : ℝ := linear_regression 4

theorem t_value : t = 6.06 := by
  sorry

end NUMINAMATH_CALUDE_t_value_l165_16589
