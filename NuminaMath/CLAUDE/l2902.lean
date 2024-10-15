import Mathlib

namespace NUMINAMATH_CALUDE_division_value_problem_l2902_290232

theorem division_value_problem (x : ℝ) : (3 / x) * 12 = 9 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_value_problem_l2902_290232


namespace NUMINAMATH_CALUDE_train_passing_time_l2902_290283

/-- Time for a train to pass a man running in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 110 →
  train_speed = 90 * (1000 / 3600) →
  man_speed = 9 * (1000 / 3600) →
  (train_length / (train_speed + man_speed)) = 4 := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l2902_290283


namespace NUMINAMATH_CALUDE_base4_multiplication_division_l2902_290290

-- Define a function to convert base 4 to decimal
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4^i)) 0

-- Define a function to convert decimal to base 4
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

-- Define the numbers in base 4 as lists of digits
def num1 : List Nat := [1, 3, 2]  -- 231₄
def num2 : List Nat := [1, 2]     -- 21₄
def num3 : List Nat := [3]        -- 3₄
def result : List Nat := [3, 3, 0, 2]  -- 2033₄

-- State the theorem
theorem base4_multiplication_division :
  decimalToBase4 ((base4ToDecimal num1 * base4ToDecimal num2) / base4ToDecimal num3) = result := by
  sorry

end NUMINAMATH_CALUDE_base4_multiplication_division_l2902_290290


namespace NUMINAMATH_CALUDE_solve_equation_l2902_290270

theorem solve_equation (x : ℚ) : (3 * x - 7) / 4 = 15 → x = 67 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2902_290270


namespace NUMINAMATH_CALUDE_solve_system_for_q_l2902_290252

theorem solve_system_for_q (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 20) 
  (eq2 : 6 * p + 5 * q = 29) : 
  q = -25 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_for_q_l2902_290252


namespace NUMINAMATH_CALUDE_bin_draw_probability_l2902_290211

def bin_probability (black white drawn : ℕ) : ℚ :=
  let total := black + white
  let ways_3b1w := (black.choose 3) * (white.choose 1)
  let ways_1b3w := (black.choose 1) * (white.choose 3)
  let favorable := ways_3b1w + ways_1b3w
  let total_ways := total.choose drawn
  (favorable : ℚ) / total_ways

theorem bin_draw_probability : 
  bin_probability 10 8 4 = 19 / 38 := by
  sorry

end NUMINAMATH_CALUDE_bin_draw_probability_l2902_290211


namespace NUMINAMATH_CALUDE_identity_proof_l2902_290216

theorem identity_proof (a b c : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (b - c) / ((a - b) * (a - c)) + (c - a) / ((b - c) * (b - a)) + (a - b) / ((c - a) * (c - b)) =
  2 / (a - b) + 2 / (b - c) + 2 / (c - a) := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l2902_290216


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l2902_290273

theorem binomial_coefficient_problem (a : ℝ) : 
  (Nat.choose 7 6 : ℝ) * a = 7 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l2902_290273


namespace NUMINAMATH_CALUDE_function_properties_l2902_290226

theorem function_properties (f : ℝ → ℝ) (h1 : f (-2) > f (-1)) (h2 : f (-1) < f 0) :
  ¬ (
    (∀ x y, -2 ≤ x ∧ x ≤ y ∧ y ≤ -1 → f x ≥ f y) ∧
    (∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≤ f y)
  ) ∧
  ¬ (
    (∀ x y, -2 ≤ x ∧ x ≤ y ∧ y ≤ -1 → f x ≤ f y) ∧
    (∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≥ f y)
  ) ∧
  ¬ (∀ x, -2 ≤ x ∧ x ≤ 0 → f x ≥ f (-1)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2902_290226


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_two_thirds_l2902_290289

theorem negative_sixty_four_to_two_thirds : (-64 : ℝ) ^ (2/3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_two_thirds_l2902_290289


namespace NUMINAMATH_CALUDE_amp_eight_five_l2902_290203

-- Define the & operation
def amp (a b : ℤ) : ℤ := (a + b)^2 * (a - b)

-- State the theorem
theorem amp_eight_five : amp 8 5 = 507 := by
  sorry

end NUMINAMATH_CALUDE_amp_eight_five_l2902_290203


namespace NUMINAMATH_CALUDE_range_of_omega_l2902_290205

/-- Given vectors a and b, and a function f, prove the range of ω -/
theorem range_of_omega (ω : ℝ) (x : ℝ) : 
  ω > 0 →
  let a := (Real.sin (ω/2 * x), Real.sin (ω * x))
  let b := (Real.sin (ω/2 * x), (1/2 : ℝ))
  let f := λ x => (a.1 * b.1 + a.2 * b.2) - 1/2
  (∀ x ∈ Set.Ioo π (2*π), f x ≠ 0) →
  ω ∈ Set.Ioc 0 (1/8) ∪ Set.Icc (1/4) (5/8) :=
sorry

end NUMINAMATH_CALUDE_range_of_omega_l2902_290205


namespace NUMINAMATH_CALUDE_largest_subset_size_l2902_290295

/-- A function that returns the size of the largest subset of {1,2,...,n} where no two elements differ by 5 or 8 -/
def maxSubsetSize (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the largest subset of {1,2,3,...,2023} where no two elements differ by 5 or 8 has 780 elements -/
theorem largest_subset_size :
  maxSubsetSize 2023 = 780 :=
sorry

end NUMINAMATH_CALUDE_largest_subset_size_l2902_290295


namespace NUMINAMATH_CALUDE_tabitha_money_proof_l2902_290293

def calculate_remaining_money (initial_amount : ℚ) (given_away : ℚ) (investment_percentage : ℚ) (num_items : ℕ) (item_cost : ℚ) : ℚ :=
  let remaining_after_giving := initial_amount - given_away
  let investment_amount := (investment_percentage / 100) * remaining_after_giving
  let remaining_after_investment := remaining_after_giving - investment_amount
  let spent_on_items := (num_items : ℚ) * item_cost
  remaining_after_investment - spent_on_items

theorem tabitha_money_proof :
  calculate_remaining_money 45 10 60 12 0.75 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tabitha_money_proof_l2902_290293


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2902_290255

/-- Two arithmetic sequences and their sum sequences -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_ratio : 
  ∀ (a b : ℕ → ℚ) (S T : ℕ → ℚ),
  arithmetic_sequence a →
  arithmetic_sequence b →
  (∀ n : ℕ, S n = (n : ℚ) * a n - (n - 1 : ℚ) / 2 * (a n - a 1)) →
  (∀ n : ℕ, T n = (n : ℚ) * b n - (n - 1 : ℚ) / 2 * (b n - b 1)) →
  (∀ n : ℕ, n > 0 → S n / T n = (5 * n - 3 : ℚ) / (2 * n + 1)) →
  a 20 / b 7 = 64 / 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2902_290255


namespace NUMINAMATH_CALUDE_beth_twice_sister_age_l2902_290299

/-- 
Given:
- Beth is currently 18 years old
- Beth's sister is currently 5 years old

Prove that the number of years until Beth is twice her sister's age is 8.
-/
theorem beth_twice_sister_age (beth_age : ℕ) (sister_age : ℕ) : 
  beth_age = 18 → sister_age = 5 → (beth_age + 8 = 2 * (sister_age + 8)) := by
  sorry

end NUMINAMATH_CALUDE_beth_twice_sister_age_l2902_290299


namespace NUMINAMATH_CALUDE_air_conditioner_installation_rates_l2902_290256

theorem air_conditioner_installation_rates 
  (total_A : ℕ) (total_B : ℕ) (diff : ℕ) :
  total_A = 66 →
  total_B = 60 →
  diff = 2 →
  ∃ (days : ℕ) (rate_A : ℕ) (rate_B : ℕ),
    rate_A = rate_B + diff ∧
    rate_A * days = total_A ∧
    rate_B * days = total_B ∧
    rate_A = 22 ∧
    rate_B = 20 :=
by sorry

end NUMINAMATH_CALUDE_air_conditioner_installation_rates_l2902_290256


namespace NUMINAMATH_CALUDE_largest_digit_change_corrects_addition_l2902_290234

def original_sum : ℕ := 735 + 468 + 281
def given_result : ℕ := 1584
def correct_first_addend : ℕ := 835

theorem largest_digit_change_corrects_addition :
  (original_sum ≠ given_result) →
  (correct_first_addend + 468 + 281 = given_result) →
  ∀ (d : ℕ), d ≤ 9 →
    (d > 7 → 
      ¬∃ (a b c : ℕ), a ≤ 999 ∧ b ≤ 999 ∧ c ≤ 999 ∧
        (a + b + c = given_result) ∧
        (a = 735 + d * 100 - 700 ∨
         b = 468 + d * 100 - 400 ∨
         c = 281 + d * 100 - 200)) :=
sorry

end NUMINAMATH_CALUDE_largest_digit_change_corrects_addition_l2902_290234


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_p_and_q_l2902_290247

theorem not_p_sufficient_not_necessary_for_not_p_and_q
  (p q : Prop) :
  (∀ (h : ¬p), ¬(p ∧ q)) ∧
  ¬(∀ (h : ¬(p ∧ q)), ¬p) :=
by sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_p_and_q_l2902_290247


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l2902_290267

theorem inscribed_circle_area_ratio (a : ℝ) (ha : a > 0) :
  let square_area := a^2
  let circle_radius := a / 2
  let circle_area := π * circle_radius^2
  circle_area / square_area = π / 4 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l2902_290267


namespace NUMINAMATH_CALUDE_sqrt_15_factorial_simplification_l2902_290241

theorem sqrt_15_factorial_simplification :
  ∃ (a b : ℕ+) (q : ℚ),
    (a:ℝ) * Real.sqrt b = Real.sqrt (Nat.factorial 15) ∧
    q * (Nat.factorial 15 : ℚ) = (a * b : ℚ) ∧
    q = 1 / 30240 := by sorry

end NUMINAMATH_CALUDE_sqrt_15_factorial_simplification_l2902_290241


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2902_290208

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  (∃ a b c : ℕ, a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = 100 * a + 10 * b + c ∧
    n = 5 * a * b * c) ∧
  n = 175 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2902_290208


namespace NUMINAMATH_CALUDE_a_plus_b_equals_one_l2902_290244

theorem a_plus_b_equals_one (a b : ℝ) (h : |a^3 - 27| + (b + 2)^2 = 0) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_one_l2902_290244


namespace NUMINAMATH_CALUDE_square_difference_divided_l2902_290251

theorem square_difference_divided : (147^2 - 133^2) / 14 = 280 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_l2902_290251


namespace NUMINAMATH_CALUDE_gift_shop_combinations_l2902_290204

/-- The number of varieties of wrapping paper -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of ribbon colors -/
def ribbon_colors : ℕ := 5

/-- The number of gift card types -/
def gift_card_types : ℕ := 6

/-- The number of required ribbon colors (silver and gold) -/
def required_ribbon_colors : ℕ := 2

/-- The total number of possible combinations -/
def total_combinations : ℕ := wrapping_paper_varieties * required_ribbon_colors * gift_card_types

theorem gift_shop_combinations :
  total_combinations = 120 :=
by sorry

end NUMINAMATH_CALUDE_gift_shop_combinations_l2902_290204


namespace NUMINAMATH_CALUDE_cube_inequality_iff_l2902_290242

theorem cube_inequality_iff (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_inequality_iff_l2902_290242


namespace NUMINAMATH_CALUDE_three_blocks_selection_count_l2902_290201

-- Define the size of the grid
def grid_size : ℕ := 5

-- Define the number of blocks to select
def blocks_to_select : ℕ := 3

-- Theorem statement
theorem three_blocks_selection_count :
  (grid_size.choose blocks_to_select) * (grid_size.choose blocks_to_select) * (blocks_to_select.factorial) = 600 := by
  sorry

end NUMINAMATH_CALUDE_three_blocks_selection_count_l2902_290201


namespace NUMINAMATH_CALUDE_complex_equation_solution_existence_l2902_290210

theorem complex_equation_solution_existence :
  ∃ (z : ℂ), z * (z + 2*I) * (z + 4*I) = 4012*I ∧
  ∃ (a b : ℝ), z = a + b*I :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_existence_l2902_290210


namespace NUMINAMATH_CALUDE_exists_cheaper_a_l2902_290214

/-- Represents the charge for printing x copies from Company A -/
def company_a_charge (x : ℝ) : ℝ := 0.2 * x + 200

/-- Represents the charge for printing x copies from Company B -/
def company_b_charge (x : ℝ) : ℝ := 0.4 * x

/-- Theorem stating that there exists a number of copies where Company A is cheaper than Company B -/
theorem exists_cheaper_a : ∃ x : ℝ, company_a_charge x < company_b_charge x :=
sorry

end NUMINAMATH_CALUDE_exists_cheaper_a_l2902_290214


namespace NUMINAMATH_CALUDE_square_of_difference_l2902_290278

theorem square_of_difference (y : ℝ) (h : y^2 ≥ 49) :
  (7 - Real.sqrt (y^2 - 49))^2 = y^2 - 14 * Real.sqrt (y^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l2902_290278


namespace NUMINAMATH_CALUDE_set_equality_l2902_290224

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l2902_290224


namespace NUMINAMATH_CALUDE_negation_of_at_least_three_l2902_290243

-- Define a proposition for "at least n"
def at_least (n : ℕ) : Prop := sorry

-- Define a proposition for "at most n"
def at_most (n : ℕ) : Prop := sorry

-- State the given condition
axiom negation_rule : ∀ n : ℕ, ¬(at_least n) ↔ at_most (n - 1)

-- State the theorem to be proved
theorem negation_of_at_least_three : ¬(at_least 3) ↔ at_most 2 := by sorry

end NUMINAMATH_CALUDE_negation_of_at_least_three_l2902_290243


namespace NUMINAMATH_CALUDE_max_large_chips_l2902_290275

theorem max_large_chips (total : ℕ) (small large : ℕ → ℕ) (p : ℕ → ℕ) :
  total = 70 →
  (∀ n, total = small n + large n) →
  (∀ n, Prime (p n)) →
  (∀ n, small n = large n + p n) →
  (∀ n, large n ≤ 34) ∧ (∃ n, large n = 34) :=
by sorry

end NUMINAMATH_CALUDE_max_large_chips_l2902_290275


namespace NUMINAMATH_CALUDE_sum_of_digits_after_addition_l2902_290285

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Number of carries in addition -/
def carries_in_addition (a b : ℕ) : ℕ := sorry

theorem sum_of_digits_after_addition (A B : ℕ) 
  (hA : A > 0) 
  (hB : B > 0) 
  (hSumA : sum_of_digits A = 19) 
  (hSumB : sum_of_digits B = 20) 
  (hCarries : carries_in_addition A B = 2) : 
  sum_of_digits (A + B) = 21 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_after_addition_l2902_290285


namespace NUMINAMATH_CALUDE_parabola_equation_and_chord_length_l2902_290238

/-- Parabola with vertex at origin, focus on positive y-axis, and focus-directrix distance 2 -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  vertex_at_origin : equation 0 0
  focus_on_y_axis : ∃ y > 0, equation 0 y
  focus_directrix_distance : ℝ

/-- Line defined by y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := λ x y => y = m * x + b

theorem parabola_equation_and_chord_length 
  (p : Parabola) 
  (h_dist : p.focus_directrix_distance = 2) 
  (l : Line) 
  (h_line : l.m = 2 ∧ l.b = 1) :
  (∀ x y, p.equation x y ↔ x^2 = 4*y) ∧
  (∃ A B : ℝ × ℝ, 
    p.equation A.1 A.2 ∧ 
    p.equation B.1 B.2 ∧ 
    l.equation A.1 A.2 ∧ 
    l.equation B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 20) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_and_chord_length_l2902_290238


namespace NUMINAMATH_CALUDE_inequality_relationship_l2902_290264

theorem inequality_relationship (a b : ℝ) (ha : a > 0) (hb : -1 < b ∧ b < 0) :
  a * b < a * b^2 ∧ a * b^2 < a := by
  sorry

end NUMINAMATH_CALUDE_inequality_relationship_l2902_290264


namespace NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l2902_290240

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 15 sides has 90 diagonals -/
theorem fifteen_sided_polygon_diagonals :
  num_diagonals 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l2902_290240


namespace NUMINAMATH_CALUDE_trip_distance_proof_l2902_290271

/-- Represents the total length of the trip in miles. -/
def total_distance : ℝ := 150

/-- Represents the distance traveled on battery power in miles. -/
def battery_distance : ℝ := 50

/-- Represents the fuel consumption rate in gallons per mile. -/
def fuel_rate : ℝ := 0.03

/-- Represents the average fuel efficiency for the entire trip in miles per gallon. -/
def avg_efficiency : ℝ := 50

theorem trip_distance_proof :
  (total_distance / (fuel_rate * (total_distance - battery_distance)) = avg_efficiency) ∧
  (total_distance > battery_distance) :=
by sorry

end NUMINAMATH_CALUDE_trip_distance_proof_l2902_290271


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2902_290260

theorem rectangle_area_change (L W : ℝ) (h1 : L > 0) (h2 : W > 0) :
  let new_length := 1.4 * L
  let new_width := 0.5 * W
  let original_area := L * W
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = -0.3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2902_290260


namespace NUMINAMATH_CALUDE_wheel_radii_problem_l2902_290262

theorem wheel_radii_problem (x : ℝ) : 
  (2 * x > 0) →  -- Ensure positive radii
  (1500 / (2 * Real.pi * x + 5) = 1875 / (4 * Real.pi * x - 5)) → 
  (x = 15 / (2 * Real.pi) ∧ 2 * x = 15 / Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_wheel_radii_problem_l2902_290262


namespace NUMINAMATH_CALUDE_hallie_paintings_sold_l2902_290266

/-- The number of paintings Hallie sold -/
def paintings_sold (prize : ℕ) (painting_price : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings - prize) / painting_price

theorem hallie_paintings_sold :
  paintings_sold 150 50 300 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hallie_paintings_sold_l2902_290266


namespace NUMINAMATH_CALUDE_T_is_three_rays_with_common_endpoint_l2902_290250

/-- The set T of points in the coordinate plane satisfying the given conditions -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 2 ≤ 5) ∨
               (5 = y - 2 ∧ x + 3 ≤ 5) ∨
               (x + 3 = y - 2 ∧ 5 ≥ x + 3)}

/-- The common endpoint of the three rays -/
def common_endpoint : ℝ × ℝ := (2, 7)

/-- The three rays that form set T -/
def ray1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 ∧ p.2 ≤ 7}
def ray2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 7 ∧ p.1 ≤ 2}
def ray3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 5 ∧ p.1 ≤ 2}

/-- Theorem stating that T consists of three rays with a common endpoint -/
theorem T_is_three_rays_with_common_endpoint :
  T = ray1 ∪ ray2 ∪ ray3 ∧
  common_endpoint ∈ ray1 ∧
  common_endpoint ∈ ray2 ∧
  common_endpoint ∈ ray3 :=
sorry

end NUMINAMATH_CALUDE_T_is_three_rays_with_common_endpoint_l2902_290250


namespace NUMINAMATH_CALUDE_four_numbers_property_l2902_290218

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem four_numbers_property (a b c d : ℕ) : 
  a = 1 → b = 2 → c = 3 → d = 5 →
  is_prime (a * b + c * d) ∧ 
  is_prime (a * c + b * d) ∧ 
  is_prime (a * d + b * c) := by
sorry

end NUMINAMATH_CALUDE_four_numbers_property_l2902_290218


namespace NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l2902_290294

theorem tagged_fish_in_second_catch 
  (total_fish : ℕ) 
  (initial_tagged : ℕ) 
  (second_catch : ℕ) 
  (h1 : total_fish = 250) 
  (h2 : initial_tagged = 50) 
  (h3 : second_catch = 50) :
  (initial_tagged : ℚ) / total_fish = (initial_tagged : ℚ) / second_catch → 
  (initial_tagged : ℚ) * second_catch / total_fish = 10 :=
by sorry

end NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l2902_290294


namespace NUMINAMATH_CALUDE_number_division_l2902_290237

theorem number_division (x : ℝ) : x - 17 = 55 → x / 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_division_l2902_290237


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2902_290296

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, m^2}
def B : Set ℝ := {2, 4}

-- Define the intersection condition
def intersection_condition (m : ℝ) : Prop := A m ∩ B = {4}

-- Define sufficiency
def is_sufficient (m : ℝ) : Prop := m = -2 → intersection_condition m

-- Define non-necessity
def is_not_necessary (m : ℝ) : Prop := ∃ x : ℝ, x ≠ -2 ∧ intersection_condition x

-- Theorem statement
theorem sufficient_but_not_necessary :
  (∀ m : ℝ, is_sufficient m) ∧ (∃ m : ℝ, is_not_necessary m) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2902_290296


namespace NUMINAMATH_CALUDE_hanna_erasers_count_l2902_290206

/-- The number of erasers Tanya has -/
def tanya_erasers : ℕ := 20

/-- The number of red erasers Tanya has -/
def tanya_red_erasers : ℕ := tanya_erasers / 2

/-- The number of erasers Rachel has -/
def rachel_erasers : ℕ := tanya_red_erasers / 2 - 3

/-- The number of erasers Hanna has -/
def hanna_erasers : ℕ := rachel_erasers * 2

theorem hanna_erasers_count : hanna_erasers = 4 := by
  sorry

end NUMINAMATH_CALUDE_hanna_erasers_count_l2902_290206


namespace NUMINAMATH_CALUDE_other_asymptote_equation_l2902_290287

/-- A hyperbola with one known asymptote and foci on a vertical line -/
structure Hyperbola where
  /-- The slope of the known asymptote -/
  known_asymptote_slope : ℝ
  /-- The x-coordinate of the line containing the foci -/
  foci_x : ℝ

/-- The equation of the other asymptote of the hyperbola -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => y = (-h.known_asymptote_slope) * x + (h.known_asymptote_slope + 1) * h.foci_x * 2

theorem other_asymptote_equation (h : Hyperbola) 
    (h_slope : h.known_asymptote_slope = 4) 
    (h_foci : h.foci_x = 3) :
    other_asymptote h = fun x y => y = -4 * x + 24 := by
  sorry

end NUMINAMATH_CALUDE_other_asymptote_equation_l2902_290287


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2902_290254

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) (ha : a ≠ 0) :
  (a / (1 - r)) = 81 * (a * r^4 / (1 - r)) → r = 1/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2902_290254


namespace NUMINAMATH_CALUDE_square_floor_theorem_l2902_290228

/-- Represents a square floor tiled with congruent square tiles. -/
structure SquareFloor :=
  (side_length : ℕ)

/-- Calculates the number of black tiles on the diagonals of a square floor. -/
def black_tiles (floor : SquareFloor) : ℕ :=
  2 * floor.side_length - 1

/-- Calculates the total number of tiles on a square floor. -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length * floor.side_length

/-- Theorem stating that a square floor with 101 black diagonal tiles has 2601 total tiles. -/
theorem square_floor_theorem (floor : SquareFloor) :
  black_tiles floor = 101 → total_tiles floor = 2601 :=
by sorry

end NUMINAMATH_CALUDE_square_floor_theorem_l2902_290228


namespace NUMINAMATH_CALUDE_k_range_for_three_roots_l2902_290276

/-- The cubic function f(x) = x³ - x² - x + k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^3 - x^2 - x + k

/-- The derivative of f(x) -/
def f_deriv (x : ℝ) : ℝ := 3*x^2 - 2*x - 1

/-- Theorem stating the range of k for f(x) to have exactly three roots -/
theorem k_range_for_three_roots :
  ∀ k : ℝ, (∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f k x₁ = 0 ∧ f k x₂ = 0 ∧ f k x₃ = 0) ↔ 
  -5/27 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_k_range_for_three_roots_l2902_290276


namespace NUMINAMATH_CALUDE_exactly_one_integer_n_for_n_plus_i_sixth_power_integer_l2902_290279

theorem exactly_one_integer_n_for_n_plus_i_sixth_power_integer :
  ∃! (n : ℤ), ∃ (m : ℤ), (n + Complex.I) ^ 6 = m := by sorry

end NUMINAMATH_CALUDE_exactly_one_integer_n_for_n_plus_i_sixth_power_integer_l2902_290279


namespace NUMINAMATH_CALUDE_least_k_value_l2902_290222

theorem least_k_value (k : ℤ) : ∀ n : ℤ, n ≥ 7 ↔ (0.00010101 * (10 : ℝ)^n > 1000) :=
by sorry

end NUMINAMATH_CALUDE_least_k_value_l2902_290222


namespace NUMINAMATH_CALUDE_cosine_derivative_at_pi_sixth_l2902_290259

theorem cosine_derivative_at_pi_sixth :
  let f : ℝ → ℝ := λ x ↦ Real.cos x
  (deriv f) (π / 6) = - (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cosine_derivative_at_pi_sixth_l2902_290259


namespace NUMINAMATH_CALUDE_vincent_book_cost_l2902_290215

theorem vincent_book_cost (animal_books : ℕ) (space_books : ℕ) (train_books : ℕ) (cost_per_book : ℕ) :
  animal_books = 15 →
  space_books = 4 →
  train_books = 6 →
  cost_per_book = 26 →
  (animal_books + space_books + train_books) * cost_per_book = 650 :=
by
  sorry

end NUMINAMATH_CALUDE_vincent_book_cost_l2902_290215


namespace NUMINAMATH_CALUDE_arithmetic_sequence_forms_straight_line_l2902_290219

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_forms_straight_line
  (a : ℕ → ℝ) (h : isArithmeticSequence a) :
  ∃ m b : ℝ, ∀ n : ℕ, a n = m * n + b :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_forms_straight_line_l2902_290219


namespace NUMINAMATH_CALUDE_triangle_side_length_l2902_290200

theorem triangle_side_length
  (A B C : ℝ)  -- Angles of the triangle
  (AB BC AC : ℝ)  -- Sides of the triangle
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π)  -- Angle sum theorem
  (h5 : Real.cos (A + 2*C - B) + Real.sin (B + C - A) = 2)
  (h6 : AB = 2)
  : BC = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2902_290200


namespace NUMINAMATH_CALUDE_exists_trapezoid_in_selected_vertices_l2902_290284

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A set of selected vertices from a regular polygon -/
def SelectedVertices (n k : ℕ) (p : RegularPolygon n) :=
  {s : Finset (Fin n) // s.card = k}

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides -/
def IsTrapezoid (v1 v2 v3 v4 : ℝ × ℝ) : Prop :=
  (v1.1 - v2.1) * (v3.2 - v4.2) = (v1.2 - v2.2) * (v3.1 - v4.1) ∨
  (v1.1 - v3.1) * (v2.2 - v4.2) = (v1.2 - v3.2) * (v2.1 - v4.1) ∨
  (v1.1 - v4.1) * (v2.2 - v3.2) = (v1.2 - v4.2) * (v2.1 - v3.1)

/-- Main theorem: There exists a trapezoid among 64 selected vertices of a regular 1981-gon -/
theorem exists_trapezoid_in_selected_vertices 
  (p : RegularPolygon 1981) (s : SelectedVertices 1981 64 p) :
  ∃ (a b c d : Fin 1981), a ∈ s.val ∧ b ∈ s.val ∧ c ∈ s.val ∧ d ∈ s.val ∧
    IsTrapezoid (p.vertices a) (p.vertices b) (p.vertices c) (p.vertices d) :=
by
  sorry

end NUMINAMATH_CALUDE_exists_trapezoid_in_selected_vertices_l2902_290284


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2902_290291

theorem trigonometric_simplification (x : ℝ) :
  (1 + Real.sin x + Real.cos x + Real.sqrt 2 * Real.sin x * Real.cos x) /
  (1 - Real.sin x + Real.cos x - Real.sqrt 2 * Real.sin x * Real.cos x) =
  1 + (Real.sqrt 2 - 1) * Real.tan (x / 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2902_290291


namespace NUMINAMATH_CALUDE_function_is_identity_l2902_290213

/-- A function satisfying specific functional equations -/
def FunctionWithProperties (f : ℝ → ℝ) (c : ℝ) : Prop :=
  c ≠ 0 ∧ 
  (∀ x : ℝ, f (x + 1) = f x + c) ∧
  (∀ x : ℝ, f (x^2) = (f x)^2)

/-- Theorem stating that a function with the given properties is the identity function with c = 1 -/
theorem function_is_identity 
  {f : ℝ → ℝ} {c : ℝ} 
  (h : FunctionWithProperties f c) : 
  c = 1 ∧ ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_CALUDE_function_is_identity_l2902_290213


namespace NUMINAMATH_CALUDE_spearman_correlation_approx_l2902_290298

def scores_A : List ℝ := [95, 90, 86, 84, 75, 70, 62, 60, 57, 50]
def scores_B : List ℝ := [92, 93, 83, 80, 55, 60, 45, 72, 62, 70]

def spearman_rank_correlation (x y : List ℝ) : ℝ :=
  sorry

theorem spearman_correlation_approx :
  ∃ ε > 0, ε < 0.01 ∧ |spearman_rank_correlation scores_A scores_B - 0.64| < ε :=
by sorry

end NUMINAMATH_CALUDE_spearman_correlation_approx_l2902_290298


namespace NUMINAMATH_CALUDE_trapezoid_not_constructible_l2902_290207

/-- Represents a quadrilateral with sides a, b, c, d where a is parallel to c -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  a_parallel_c : True  -- We use True here as a placeholder for the parallel condition

/-- The triangle inequality: the sum of any two sides of a triangle must be greater than the third side -/
def triangle_inequality (x y z : ℝ) : Prop := x + y > z ∧ y + z > x ∧ z + x > y

/-- Theorem stating that a trapezoid with the given side lengths cannot be formed -/
theorem trapezoid_not_constructible : ¬ ∃ (t : Trapezoid), t.a = 16 ∧ t.b = 13 ∧ t.c = 10 ∧ t.d = 6 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_not_constructible_l2902_290207


namespace NUMINAMATH_CALUDE_brownie_solution_l2902_290253

/-- Represents the brownie distribution problem --/
def brownie_problem (total_brownies : ℕ) (total_cost : ℚ) (faculty_fraction : ℚ) 
  (faculty_price_increase : ℚ) (carl_fraction : ℚ) (simon_brownies : ℕ) 
  (friends_fraction : ℚ) (num_friends : ℕ) : Prop :=
  let original_price := total_cost / total_brownies
  let faculty_brownies := (faculty_fraction * total_brownies).floor
  let faculty_price := original_price + faculty_price_increase
  let remaining_after_faculty := total_brownies - faculty_brownies
  let carl_brownies := (carl_fraction * remaining_after_faculty).floor
  let remaining_after_carl := remaining_after_faculty - carl_brownies - simon_brownies
  let friends_brownies := (friends_fraction * remaining_after_carl).floor
  let annie_brownies := remaining_after_carl - friends_brownies
  let annie_cost := annie_brownies * original_price
  let faculty_cost := faculty_brownies * faculty_price
  annie_cost = 5.1 ∧ faculty_cost = 45

/-- Theorem stating the solution to the brownie problem --/
theorem brownie_solution : 
  brownie_problem 150 45 (3/5) 0.2 (1/4) 3 (2/3) 5 := by
  sorry

end NUMINAMATH_CALUDE_brownie_solution_l2902_290253


namespace NUMINAMATH_CALUDE_triangles_in_regular_decagon_l2902_290265

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def triangles_in_decagon : ℕ := 120

/-- A regular decagon has 10 vertices -/
def decagon_vertices : ℕ := 10

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

theorem triangles_in_regular_decagon : 
  triangles_in_decagon = Nat.choose decagon_vertices triangle_vertices := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_regular_decagon_l2902_290265


namespace NUMINAMATH_CALUDE_smallest_k_cosine_squared_l2902_290212

theorem smallest_k_cosine_squared (k : ℕ) : k = 53 ↔ 
  (k > 0 ∧ 
   ∀ m : ℕ, m > 0 → m < k → (Real.cos ((m^2 + 7^2 : ℝ) * Real.pi / 180))^2 ≠ 1) ∧
  (Real.cos ((k^2 + 7^2 : ℝ) * Real.pi / 180))^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_cosine_squared_l2902_290212


namespace NUMINAMATH_CALUDE_gear_system_rotation_l2902_290209

/-- Represents a circular arrangement of gears -/
structure GearSystem where
  n : ℕ  -- number of gears
  circular : Bool  -- true if the arrangement is circular

/-- Defines when a gear system can rotate -/
def can_rotate (gs : GearSystem) : Prop :=
  gs.circular ∧ Even gs.n

/-- Theorem: A circular gear system can rotate if and only if the number of gears is even -/
theorem gear_system_rotation (gs : GearSystem) (h : gs.circular = true) : 
  can_rotate gs ↔ Even gs.n :=
sorry

end NUMINAMATH_CALUDE_gear_system_rotation_l2902_290209


namespace NUMINAMATH_CALUDE_investment_return_calculation_l2902_290231

/-- Calculates the monthly return given the current value, duration, and growth factor of an investment. -/
def calculateMonthlyReturn (currentValue : ℚ) (months : ℕ) (growthFactor : ℚ) : ℚ :=
  (currentValue * (growthFactor - 1)) / months

/-- Theorem stating that an investment tripling over 5 months with a current value of $90 has a monthly return of $12. -/
theorem investment_return_calculation :
  let currentValue : ℚ := 90
  let months : ℕ := 5
  let growthFactor : ℚ := 3
  calculateMonthlyReturn currentValue months growthFactor = 12 := by
  sorry

end NUMINAMATH_CALUDE_investment_return_calculation_l2902_290231


namespace NUMINAMATH_CALUDE_petes_number_l2902_290225

theorem petes_number (x : ℚ) : 4 * (2 * x + 10) = 120 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l2902_290225


namespace NUMINAMATH_CALUDE_problem_solution_l2902_290288

def second_order_det (a b c d : ℝ) : ℝ := a * d - b * c

theorem problem_solution :
  (second_order_det 3 (-2) 4 (-3) = -1) ∧
  (∀ x : ℝ, second_order_det (2*x-3) (x+2) 2 4 = 6*x - 16) ∧
  (second_order_det 5 6 2 4 = 8) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2902_290288


namespace NUMINAMATH_CALUDE_fraction_simplification_l2902_290220

/-- 
For any integer n, the fraction (5n+3)/(7n+8) can be simplified by 5 
if and only if n is divisible by 5 or n is of the form 19k + 7 for some integer k.
-/
theorem fraction_simplification (n : ℤ) : 
  (∃ (m : ℤ), 5 * (7*n + 8) = 7 * (5*n + 3) * m) ↔ 
  (∃ (k : ℤ), n = 5*k ∨ n = 19*k + 7) := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2902_290220


namespace NUMINAMATH_CALUDE_cosine_period_l2902_290246

/-- The period of the cosine function with a modified argument -/
theorem cosine_period (f : ℝ → ℝ) (h : f = λ x => Real.cos ((3 * x) / 4 + π / 6)) :
  ∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧ p = 8 * π / 3 :=
sorry

end NUMINAMATH_CALUDE_cosine_period_l2902_290246


namespace NUMINAMATH_CALUDE_key_chain_profit_percentage_l2902_290292

theorem key_chain_profit_percentage 
  (P : ℝ) 
  (h1 : P = 100) 
  (h2 : P - 50 = 0.5 * P) 
  (h3 : 70 < P) : 
  (P - 70) / P = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_key_chain_profit_percentage_l2902_290292


namespace NUMINAMATH_CALUDE_prime_greater_than_five_form_l2902_290261

theorem prime_greater_than_five_form (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  ∃ k : ℕ, p = 6 * k + 1 := by
sorry

end NUMINAMATH_CALUDE_prime_greater_than_five_form_l2902_290261


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_and_perimeter_l2902_290268

/-- An isosceles right triangle with hypotenuse 6√2 -/
structure IsoscelesRightTriangle where
  hypotenuse : ℝ
  is_isosceles_right : hypotenuse = 6 * Real.sqrt 2

theorem isosceles_right_triangle_area_and_perimeter 
  (t : IsoscelesRightTriangle) : 
  ∃ (leg : ℝ), 
    leg^2 + leg^2 = t.hypotenuse^2 ∧ 
    (1/2 * leg * leg = 18) ∧ 
    (leg + leg + t.hypotenuse = 12 + 6 * Real.sqrt 2) := by
  sorry

#check isosceles_right_triangle_area_and_perimeter

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_and_perimeter_l2902_290268


namespace NUMINAMATH_CALUDE_quadratic_root_property_l2902_290227

theorem quadratic_root_property (m : ℝ) : 
  (∃ α β : ℝ, (3 * α^2 + m * α - 4 = 0) ∧ 
              (3 * β^2 + m * β - 4 = 0) ∧ 
              (α * β = 2 * (α^3 + β^3))) ↔ 
  (m = -1.5 ∨ m = 6 ∨ m = -2.4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l2902_290227


namespace NUMINAMATH_CALUDE_heather_bicycle_speed_l2902_290274

/-- Heather's bicycle problem -/
theorem heather_bicycle_speed (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 40 ∧ time = 5 ∧ speed = distance / time → speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_heather_bicycle_speed_l2902_290274


namespace NUMINAMATH_CALUDE_max_product_constrained_max_product_value_max_product_achieved_l2902_290280

theorem max_product_constrained (m n : ℝ) : 
  m = 8 - n → m > 0 → n > 0 → ∀ x y : ℝ, x = 8 - y → x > 0 → y > 0 → x * y ≤ m * n := by
  sorry

theorem max_product_value (m n : ℝ) :
  m = 8 - n → m > 0 → n > 0 → m * n ≤ 16 := by
  sorry

theorem max_product_achieved (m n : ℝ) :
  m = 8 - n → m > 0 → n > 0 → ∃ x y : ℝ, x = 8 - y ∧ x > 0 ∧ y > 0 ∧ x * y = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_max_product_value_max_product_achieved_l2902_290280


namespace NUMINAMATH_CALUDE_fixed_point_sum_l2902_290239

theorem fixed_point_sum (a : ℝ) (m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (fun x => a^(x - 1) + 2 : ℝ → ℝ) m = n → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_sum_l2902_290239


namespace NUMINAMATH_CALUDE_largest_square_area_l2902_290233

-- Define a right-angled triangle with squares on each side
structure RightTriangleWithSquares where
  xy : ℝ  -- Length of side XY
  yz : ℝ  -- Length of side YZ
  xz : ℝ  -- Length of hypotenuse XZ
  right_angle : xz^2 = xy^2 + yz^2  -- Pythagorean theorem

-- Theorem statement
theorem largest_square_area
  (t : RightTriangleWithSquares)
  (sum_of_squares : t.xy^2 + t.yz^2 + t.xz^2 = 450) :
  t.xz^2 = 225 := by
sorry

end NUMINAMATH_CALUDE_largest_square_area_l2902_290233


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2902_290221

def p (x : ℝ) : ℝ := 8*x^4 + 26*x^3 - 66*x^2 + 24*x

theorem roots_of_polynomial :
  (p 0 = 0) ∧ (p (1/2) = 0) ∧ (p (3/2) = 0) ∧ (p (-4) = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2902_290221


namespace NUMINAMATH_CALUDE_expression_simplification_l2902_290223

theorem expression_simplification (x : ℝ) : 
  (12 * x^12 - 3 * x^10 + 5 * x^9) + (-x^12 + 2 * x^10 + x^9 + 4 * x^4 + 6 * x^2 + 9) = 
  11 * x^12 - x^10 + 6 * x^9 + 4 * x^4 + 6 * x^2 + 9 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l2902_290223


namespace NUMINAMATH_CALUDE_two_digit_swap_difference_l2902_290297

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  tens_valid : tens < 10
  units_valid : units < 10

/-- Calculates the value of a two-digit number -/
def value (n : TwoDigitNumber) : ℕ := 10 * n.tens + n.units

/-- Swaps the digits of a two-digit number -/
def swap_digits (n : TwoDigitNumber) : TwoDigitNumber := {
  tens := n.units,
  units := n.tens,
  tens_valid := n.units_valid,
  units_valid := n.tens_valid
}

/-- 
Theorem: The difference between a two-digit number with its digits swapped
and the original number is equal to -9x + 9y, where x is the tens digit
and y is the units digit of the original number.
-/
theorem two_digit_swap_difference (n : TwoDigitNumber) :
  (value (swap_digits n) : ℤ) - (value n : ℤ) = -9 * (n.tens : ℤ) + 9 * (n.units : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_swap_difference_l2902_290297


namespace NUMINAMATH_CALUDE_alfred_maize_storage_l2902_290248

/-- Calculates the total amount of maize Alfred has after 2 years of storage, theft, and donation -/
theorem alfred_maize_storage (
  monthly_storage : ℕ)  -- Amount of maize stored each month
  (storage_period : ℕ)   -- Storage period in years
  (stolen : ℕ)           -- Amount of maize stolen
  (donation : ℕ)         -- Amount of maize donated
  (h1 : monthly_storage = 1)
  (h2 : storage_period = 2)
  (h3 : stolen = 5)
  (h4 : donation = 8) :
  monthly_storage * (storage_period * 12) - stolen + donation = 27 :=
by
  sorry


end NUMINAMATH_CALUDE_alfred_maize_storage_l2902_290248


namespace NUMINAMATH_CALUDE_tv_watching_time_conversion_l2902_290281

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Define the number of hours Logan watched TV
def hours_watched : ℕ := 5

-- Theorem to prove
theorem tv_watching_time_conversion :
  hours_watched * minutes_per_hour = 300 := by
  sorry

end NUMINAMATH_CALUDE_tv_watching_time_conversion_l2902_290281


namespace NUMINAMATH_CALUDE_rational_coefficient_terms_count_l2902_290249

theorem rational_coefficient_terms_count :
  let expansion := (fun (x y : ℝ) => (x * Real.rpow 2 (1/3) + y * Real.sqrt 3) ^ 500)
  let total_terms := 501
  let is_rational_coeff (k : ℕ) := (k % 3 = 0) ∧ ((500 - k) % 2 = 0)
  (Finset.filter is_rational_coeff (Finset.range total_terms)).card = 84 :=
sorry

end NUMINAMATH_CALUDE_rational_coefficient_terms_count_l2902_290249


namespace NUMINAMATH_CALUDE_cone_angle_cosine_l2902_290202

/-- Given a cone whose side surface unfolds into a sector with central angle 4π/3 and radius 18 cm,
    prove that the cosine of the angle between the slant height and the base is 2/3 -/
theorem cone_angle_cosine (θ : Real) (l r : Real) : 
  θ = 4 / 3 * π → 
  l = 18 → 
  θ = 2 * π * r / l → 
  r / l = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_cone_angle_cosine_l2902_290202


namespace NUMINAMATH_CALUDE_square_area_perimeter_relationship_l2902_290258

/-- The relationship between the area and perimeter of a square is quadratic -/
theorem square_area_perimeter_relationship (x y : ℝ) (h_pos : x > 0) :
  ∃ k : ℝ, y = k * x^2 ↔ 
  (∃ a : ℝ, a > 0 ∧ x = 4 * a ∧ y = a^2) :=
by sorry

end NUMINAMATH_CALUDE_square_area_perimeter_relationship_l2902_290258


namespace NUMINAMATH_CALUDE_q_coordinates_is_rectangle_l2902_290269

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle defined by four points -/
structure Rectangle where
  O : Point
  P : Point
  Q : Point
  R : Point

/-- Definition of our specific rectangle -/
def our_rectangle : Rectangle :=
  { O := { x := 0, y := 0 }
  , P := { x := 0, y := 3 }
  , R := { x := 5, y := 0 }
  , Q := { x := 5, y := 3 } }

/-- Theorem: The coordinates of Q in our_rectangle are (5,3) -/
theorem q_coordinates :
  our_rectangle.Q.x = 5 ∧ our_rectangle.Q.y = 3 := by
  sorry

/-- Theorem: our_rectangle is indeed a rectangle -/
theorem is_rectangle (rect : Rectangle) : 
  (rect.O.x = rect.P.x ∧ rect.O.y = rect.R.y) →
  (rect.Q.x = rect.R.x ∧ rect.Q.y = rect.P.y) →
  (rect.P.x - rect.O.x)^2 + (rect.P.y - rect.O.y)^2 =
  (rect.R.x - rect.O.x)^2 + (rect.R.y - rect.O.y)^2 →
  True := by
  sorry

end NUMINAMATH_CALUDE_q_coordinates_is_rectangle_l2902_290269


namespace NUMINAMATH_CALUDE_pr_qs_ratio_l2902_290245

-- Define the points and distances
def P : ℝ := 0
def Q : ℝ := 3
def R : ℝ := 10
def S : ℝ := 18

-- State the theorem
theorem pr_qs_ratio :
  (R - P) / (S - Q) = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_pr_qs_ratio_l2902_290245


namespace NUMINAMATH_CALUDE_camp_hair_colors_l2902_290217

theorem camp_hair_colors (total : ℕ) (brown green black : ℕ) : 
  brown = total / 2 →
  brown = 25 →
  green = 10 →
  black = 5 →
  total - (brown + green + black) = 10 := by
sorry

end NUMINAMATH_CALUDE_camp_hair_colors_l2902_290217


namespace NUMINAMATH_CALUDE_remainder_theorem_l2902_290257

theorem remainder_theorem : (7 * 10^15 + 3^15) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2902_290257


namespace NUMINAMATH_CALUDE_parallel_line_construction_l2902_290235

/-- A point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Predicate to check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Predicate to check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- Theorem: Given a line and a point not on the line, 
    it's possible to construct a parallel line through the point 
    using only compass and straightedge -/
theorem parallel_line_construction 
  (l : Line) (A : Point) (h : ¬A.onLine l) :
  ∃ (l' : Line), A.onLine l' ∧ l.parallel l' :=
sorry

end NUMINAMATH_CALUDE_parallel_line_construction_l2902_290235


namespace NUMINAMATH_CALUDE_canvas_cost_l2902_290263

/-- Proves that the cost of canvases is $40.00 given the specified conditions -/
theorem canvas_cost (total_spent easel_cost paintbrush_cost canvas_cost : ℚ) : 
  total_spent = 90 ∧ 
  easel_cost = 15 ∧ 
  paintbrush_cost = 15 ∧ 
  total_spent = canvas_cost + (1/2 * canvas_cost) + easel_cost + paintbrush_cost →
  canvas_cost = 40 := by
sorry

end NUMINAMATH_CALUDE_canvas_cost_l2902_290263


namespace NUMINAMATH_CALUDE_density_function_properties_l2902_290230

/-- A density function that satisfies specific integral properties --/
noncomputable def f (g f_ζ : ℝ → ℝ) (x : ℝ) : ℝ := (g (-x) + f_ζ x) / 2

/-- The theorem stating the properties of the density function --/
theorem density_function_properties
  (g f_ζ : ℝ → ℝ)
  (hg : ∀ x, g (-x) = -g x)  -- g is odd
  (hf_ζ : ∀ x, f_ζ (-x) = f_ζ x)  -- f_ζ is even
  (hf_density : ∀ x, f g f_ζ x ≥ 0 ∧ ∫ x, f g f_ζ x = 1)  -- f is a density function
  : (∃ x, f g f_ζ x ≠ f g f_ζ (-x))  -- f is not even
  ∧ (∀ n : ℕ, n ≥ 1 → ∫ x in Set.Ici 0, |x|^n * f g f_ζ x = ∫ x in Set.Iic 0, |x|^n * f g f_ζ x) :=
sorry

end NUMINAMATH_CALUDE_density_function_properties_l2902_290230


namespace NUMINAMATH_CALUDE_minimum_students_l2902_290286

theorem minimum_students (b g : ℕ) : 
  (3 * b = 5 * g) →  -- Same number of boys and girls passed
  (b ≥ 5) →          -- At least 5 boys (for 3/5 to be meaningful)
  (g ≥ 6) →          -- At least 6 girls (for 5/6 to be meaningful)
  (∀ b' g', (3 * b' = 5 * g') → (b' ≥ 5) → (g' ≥ 6) → (b' + g' ≥ b + g)) →
  b + g = 43 :=
by sorry

#check minimum_students

end NUMINAMATH_CALUDE_minimum_students_l2902_290286


namespace NUMINAMATH_CALUDE_existence_of_monochromatic_right_angled_pentagon_l2902_290277

-- Define a color type
inductive Color
| Red
| Yellow

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define a convex pentagon
def ConvexPentagon (p₁ p₂ p₃ p₄ p₅ : Point) : Prop := sorry

-- Define a right angle
def RightAngle (p₁ p₂ p₃ : Point) : Prop := sorry

-- Define the theorem
theorem existence_of_monochromatic_right_angled_pentagon :
  ∃ (p₁ p₂ p₃ p₄ p₅ : Point),
    ConvexPentagon p₁ p₂ p₃ p₄ p₅ ∧
    RightAngle p₁ p₂ p₃ ∧
    RightAngle p₂ p₃ p₄ ∧
    RightAngle p₃ p₄ p₅ ∧
    ((coloring p₁ = Color.Red ∧ coloring p₂ = Color.Red ∧ coloring p₃ = Color.Red ∧ 
      coloring p₄ = Color.Red ∧ coloring p₅ = Color.Red) ∨
     (coloring p₁ = Color.Yellow ∧ coloring p₂ = Color.Yellow ∧ coloring p₃ = Color.Yellow ∧ 
      coloring p₄ = Color.Yellow ∧ coloring p₅ = Color.Yellow)) :=
by sorry


end NUMINAMATH_CALUDE_existence_of_monochromatic_right_angled_pentagon_l2902_290277


namespace NUMINAMATH_CALUDE_max_b_plus_c_l2902_290229

theorem max_b_plus_c (a b c : ℕ) (h1 : a > b) (h2 : a + b = 18) (h3 : c = b + 2) :
  b + c ≤ 18 ∧ ∃ (b' c' : ℕ), b' + c' = 18 ∧ a > b' ∧ a + b' = 18 ∧ c' = b' + 2 := by
  sorry

end NUMINAMATH_CALUDE_max_b_plus_c_l2902_290229


namespace NUMINAMATH_CALUDE_fraction_division_simplify_fraction_division_l2902_290282

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem simplify_fraction_division :
  (5 : ℚ) / 6 / ((7 : ℚ) / 12) = 10 / 7 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_simplify_fraction_division_l2902_290282


namespace NUMINAMATH_CALUDE_fraction_equality_l2902_290272

theorem fraction_equality : (18 * 3 + 12) / (6 - 4) = 33 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2902_290272


namespace NUMINAMATH_CALUDE_work_day_meetings_percentage_l2902_290236

/-- Proves that given a 10-hour work day and two meetings, where the first meeting is 60 minutes long
    and the second is three times as long, the percentage of the work day spent in meetings is 40%. -/
theorem work_day_meetings_percentage (work_day_hours : ℕ) (first_meeting_minutes : ℕ) :
  work_day_hours = 10 →
  first_meeting_minutes = 60 →
  let work_day_minutes : ℕ := work_day_hours * 60
  let second_meeting_minutes : ℕ := 3 * first_meeting_minutes
  let total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes
  let meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (work_day_minutes : ℚ) * 100
  meeting_percentage = 40 := by
  sorry


end NUMINAMATH_CALUDE_work_day_meetings_percentage_l2902_290236
