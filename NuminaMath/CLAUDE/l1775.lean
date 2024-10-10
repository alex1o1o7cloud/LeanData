import Mathlib

namespace complex_equation_solution_l1775_177583

theorem complex_equation_solution :
  ∃ z : ℂ, (5 : ℂ) - 3 * Complex.I * z = (3 : ℂ) + 5 * Complex.I * z ∧ z = Complex.I / 4 := by
  sorry

end complex_equation_solution_l1775_177583


namespace present_ages_sum_l1775_177578

theorem present_ages_sum (a b : ℕ) (h1 : a - 10 = (b - 10) / 2) (h2 : 4 * a = 3 * b) : 
  a + b = 35 := by
  sorry

end present_ages_sum_l1775_177578


namespace sandy_marks_per_correct_sum_l1775_177546

theorem sandy_marks_per_correct_sum 
  (total_sums : ℕ) 
  (total_marks : ℕ) 
  (correct_sums : ℕ) 
  (penalty_per_incorrect : ℕ) 
  (h1 : total_sums = 30) 
  (h2 : total_marks = 65) 
  (h3 : correct_sums = 25) 
  (h4 : penalty_per_incorrect = 2) :
  (total_marks + penalty_per_incorrect * (total_sums - correct_sums)) / correct_sums = 3 := by
sorry

end sandy_marks_per_correct_sum_l1775_177546


namespace smallest_integer_below_sqrt5_plus_sqrt3_to_6th_l1775_177567

theorem smallest_integer_below_sqrt5_plus_sqrt3_to_6th :
  ∃ n : ℤ, n = 3322 ∧ n < (Real.sqrt 5 + Real.sqrt 3)^6 ∧ ∀ m : ℤ, m < (Real.sqrt 5 + Real.sqrt 3)^6 → m ≤ n :=
by sorry

end smallest_integer_below_sqrt5_plus_sqrt3_to_6th_l1775_177567


namespace simplify_fraction_l1775_177519

theorem simplify_fraction : (90 : ℚ) / 8100 = 1 / 90 := by
  sorry

end simplify_fraction_l1775_177519


namespace tax_rate_problem_l1775_177576

/-- The tax rate problem in Country X -/
theorem tax_rate_problem (income : ℝ) (total_tax : ℝ) (tax_rate_above_40k : ℝ) :
  income = 50000 →
  total_tax = 8000 →
  tax_rate_above_40k = 0.2 →
  ∃ (tax_rate_below_40k : ℝ),
    tax_rate_below_40k * 40000 + tax_rate_above_40k * (income - 40000) = total_tax ∧
    tax_rate_below_40k = 0.15 := by
  sorry

end tax_rate_problem_l1775_177576


namespace max_value_xy_8x_y_l1775_177557

theorem max_value_xy_8x_y (x y : ℝ) (h : x^2 + y^2 = 20) :
  ∃ (M : ℝ), M = 42 ∧ xy + 8*x + y ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 20 ∧ x₀*y₀ + 8*x₀ + y₀ = M :=
sorry

end max_value_xy_8x_y_l1775_177557


namespace abc_inequality_l1775_177500

theorem abc_inequality (a b c : ℝ) (h : a^2 * b * c + a * b^2 * c + a * b * c^2 = 1) :
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 := by
sorry

end abc_inequality_l1775_177500


namespace absolute_value_problem_l1775_177573

theorem absolute_value_problem (x y : ℝ) 
  (hx : |x| = 3) 
  (hy : |y| = 2) :
  (x < y → x - y = -5 ∨ x - y = -1) ∧
  (x * y > 0 → x + y = 5) := by
sorry

end absolute_value_problem_l1775_177573


namespace sequence_sum_l1775_177506

theorem sequence_sum (a b c d : ℕ) : 
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧
  b - a = c - b ∧
  c * c = b * d ∧
  d - a = 20 →
  a + b + c + d = 46 := by
sorry

end sequence_sum_l1775_177506


namespace total_original_cost_l1775_177562

theorem total_original_cost (x y z : ℝ) : 
  x * (1 + 0.3) = 351 →
  y * (1 + 0.25) = 275 →
  z * (1 + 0.2) = 96 →
  x + y + z = 570 := by
sorry

end total_original_cost_l1775_177562


namespace log_equation_solution_l1775_177534

theorem log_equation_solution (b x : ℝ) 
  (hb_pos : b > 0) 
  (hb_neq_one : b ≠ 1) 
  (hx_neq_one : x ≠ 1) 
  (h_eq : (Real.log x) / (Real.log (b^3)) + (Real.log b) / (Real.log (x^3)) + (Real.log x) / (Real.log b) = 2) : 
  x = b^((6 - 2 * Real.sqrt 5) / 8) :=
sorry

end log_equation_solution_l1775_177534


namespace abs_z_equals_sqrt_two_l1775_177535

-- Define the complex number z
def z : ℂ := 1 + 2 * Complex.I + Complex.I ^ 3

-- Theorem statement
theorem abs_z_equals_sqrt_two : Complex.abs z = Real.sqrt 2 := by
  sorry

end abs_z_equals_sqrt_two_l1775_177535


namespace equation_three_roots_l1775_177554

theorem equation_three_roots :
  ∃! (s : Finset ℝ), s.card = 3 ∧ 
  (∀ x : ℝ, x ∈ s ↔ Real.sqrt (9 - x) = x^2 * Real.sqrt (9 - x)) :=
by sorry

end equation_three_roots_l1775_177554


namespace profit_and_max_profit_l1775_177589

def cost_price : ℝ := 12
def initial_price : ℝ := 20
def initial_quantity : ℝ := 240
def quantity_increase_rate : ℝ := 40

def profit (x : ℝ) : ℝ :=
  (initial_price - cost_price - x) * (initial_quantity + quantity_increase_rate * x)

theorem profit_and_max_profit :
  (∃ x : ℝ, profit x = 1920 ∧ x = 2) ∧
  (∃ x : ℝ, ∀ y : ℝ, profit y ≤ profit x ∧ x = 4) ∧
  (∃ x : ℝ, profit x = 2560 ∧ x = 4) := by sorry

end profit_and_max_profit_l1775_177589


namespace max_value_constraint_l1775_177580

theorem max_value_constraint (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) 
  (h1 : x + y - 3 ≤ 0) (h2 : 2 * x + y - 4 ≥ 0) : 
  2 * x + 3 * y ≤ 8 :=
by sorry

end max_value_constraint_l1775_177580


namespace bikes_added_per_week_l1775_177598

/-- 
Proves that the number of bikes added per week is 3, given the initial stock,
bikes sold in a month, stock after one month, and the number of weeks in a month.
-/
theorem bikes_added_per_week 
  (initial_stock : ℕ) 
  (bikes_sold : ℕ) 
  (final_stock : ℕ) 
  (weeks_in_month : ℕ) 
  (h1 : initial_stock = 51)
  (h2 : bikes_sold = 18)
  (h3 : final_stock = 45)
  (h4 : weeks_in_month = 4)
  : (final_stock - (initial_stock - bikes_sold)) / weeks_in_month = 3 := by
  sorry

end bikes_added_per_week_l1775_177598


namespace initial_average_height_l1775_177575

/-- The initially calculated average height of students in a class with measurement error -/
theorem initial_average_height (n : ℕ) (incorrect_height actual_height : ℝ) (actual_average : ℝ) 
  (hn : n = 20)
  (h_incorrect : incorrect_height = 151)
  (h_actual : actual_height = 136)
  (h_average : actual_average = 174.25) :
  ∃ (initial_average : ℝ), 
    initial_average * n = actual_average * n - (incorrect_height - actual_height) ∧ 
    initial_average = 173.5 := by
  sorry


end initial_average_height_l1775_177575


namespace store_price_reduction_l1775_177551

theorem store_price_reduction (original_price : ℝ) (first_reduction : ℝ) :
  first_reduction > 0 →
  first_reduction < 100 →
  let second_reduction := 10
  let final_price_percentage := 82.8
  (original_price * (1 - first_reduction / 100) * (1 - second_reduction / 100)) / original_price * 100 = final_price_percentage →
  first_reduction = 8 := by
sorry

end store_price_reduction_l1775_177551


namespace quadratic_equal_roots_l1775_177529

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + x + m = 0 ∧ (∀ y : ℝ, y^2 + y + m = 0 → y = x)) → m = 1/4 :=
by sorry

end quadratic_equal_roots_l1775_177529


namespace unique_coin_combination_l1775_177574

/-- Represents the number of coins of each denomination -/
structure CoinCombination where
  bronze : Nat
  silver : Nat
  gold : Nat

/-- Calculates the total value of a coin combination -/
def totalValue (c : CoinCombination) : Nat :=
  c.bronze + 9 * c.silver + 81 * c.gold

/-- Calculates the total number of coins in a combination -/
def totalCoins (c : CoinCombination) : Nat :=
  c.bronze + c.silver + c.gold

/-- Checks if a coin combination is valid for the problem -/
def isValidCombination (c : CoinCombination) : Prop :=
  totalCoins c = 23 ∧ totalValue c < 700

/-- Checks if a coin combination has the minimum number of coins for its value -/
def isMinimalCombination (c : CoinCombination) : Prop :=
  ∀ c', isValidCombination c' → totalValue c' = totalValue c → totalCoins c' ≥ totalCoins c

/-- The main theorem to prove -/
theorem unique_coin_combination : 
  ∃! c : CoinCombination, isValidCombination c ∧ isMinimalCombination c ∧ totalValue c = 647 :=
sorry

end unique_coin_combination_l1775_177574


namespace move_point_right_l1775_177559

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point horizontally by a given distance -/
def moveHorizontally (p : Point) (distance : ℝ) : Point :=
  { x := p.x + distance, y := p.y }

theorem move_point_right : 
  let A : Point := { x := -2, y := 3 }
  let movedA : Point := moveHorizontally A 2
  movedA = { x := 0, y := 3 } := by
  sorry


end move_point_right_l1775_177559


namespace purple_chip_value_l1775_177552

def blue_value : ℕ := 1
def green_value : ℕ := 5
def red_value : ℕ := 11

def is_valid_product (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), blue_value^a * green_value^b * x^c * red_value^d = 28160

theorem purple_chip_value :
  ∀ x : ℕ,
  green_value < x →
  x < red_value →
  is_valid_product x →
  x = 7 :=
by sorry

end purple_chip_value_l1775_177552


namespace sara_paycheck_l1775_177526

/-- Sara's paycheck calculation --/
theorem sara_paycheck (weeks : ℕ) (hours_per_week : ℕ) (hourly_rate : ℚ) (tire_cost : ℚ) :
  weeks = 2 →
  hours_per_week = 40 →
  hourly_rate = 11.5 →
  tire_cost = 410 →
  (weeks * hours_per_week : ℚ) * hourly_rate - tire_cost = 510 :=
by sorry

end sara_paycheck_l1775_177526


namespace B_subset_A_l1775_177564

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x^2)}

-- Define set B
def B : Set ℝ := {x | ∃ m ∈ A, x = m^2}

-- Theorem statement
theorem B_subset_A : B ⊆ A := by
  sorry

end B_subset_A_l1775_177564


namespace smallest_absolute_value_at_0_l1775_177593

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- The property that a polynomial P satisfies P(-10) = 145 and P(9) = 164 -/
def SatisfiesConditions (P : IntPolynomial) : Prop :=
  P (-10) = 145 ∧ P 9 = 164

/-- The smallest possible absolute value of P(0) for polynomials satisfying the conditions -/
def SmallestAbsoluteValueAt0 : ℕ := 25

theorem smallest_absolute_value_at_0 :
  ∀ P : IntPolynomial,
  SatisfiesConditions P →
  ∀ n : ℕ,
  n < SmallestAbsoluteValueAt0 →
  ¬(|P 0| = n) :=
by sorry

end smallest_absolute_value_at_0_l1775_177593


namespace equation_solution_l1775_177540

theorem equation_solution :
  ∃ x : ℚ, x + 5/6 = 7/18 - 2/9 ∧ x = -2/3 := by
  sorry

end equation_solution_l1775_177540


namespace special_polyhedron_interior_segments_l1775_177596

/-- A convex polyhedron with specific face composition -/
structure SpecialPolyhedron where
  /-- The polyhedron is convex -/
  is_convex : Bool
  /-- Number of square faces -/
  num_square_faces : Nat
  /-- Number of regular hexagonal faces -/
  num_hexagonal_faces : Nat
  /-- Number of regular octagonal faces -/
  num_octagonal_faces : Nat
  /-- Property that exactly one square, one hexagon, and one octagon meet at each vertex -/
  vertex_property : Bool

/-- Calculate the number of interior segments in the special polyhedron -/
def interior_segments (p : SpecialPolyhedron) : Nat :=
  sorry

/-- Theorem stating the number of interior segments in the special polyhedron -/
theorem special_polyhedron_interior_segments 
  (p : SpecialPolyhedron) 
  (h1 : p.is_convex = true)
  (h2 : p.num_square_faces = 12)
  (h3 : p.num_hexagonal_faces = 8)
  (h4 : p.num_octagonal_faces = 6)
  (h5 : p.vertex_property = true) :
  interior_segments p = 840 := by
  sorry

end special_polyhedron_interior_segments_l1775_177596


namespace no_solution_implies_non_positive_product_l1775_177520

theorem no_solution_implies_non_positive_product (a b : ℝ) : 
  (∀ x : ℝ, (3*a + 8*b)*x + 7 ≠ 0) → a*b ≤ 0 := by
  sorry

end no_solution_implies_non_positive_product_l1775_177520


namespace pencils_per_row_l1775_177543

/-- Given 6 pencils placed equally into 2 rows, prove that there are 3 pencils in each row. -/
theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) : 
  total_pencils = 6 → num_rows = 2 → total_pencils = num_rows * pencils_per_row → pencils_per_row = 3 := by
  sorry

end pencils_per_row_l1775_177543


namespace complement_of_union_l1775_177590

def U : Set Nat := {1, 2, 3, 4, 5}
def P : Set Nat := {1, 2, 3}
def Q : Set Nat := {2, 3, 4}

theorem complement_of_union :
  (U \ (P ∪ Q)) = {5} := by sorry

end complement_of_union_l1775_177590


namespace lauren_revenue_l1775_177568

def commercial_revenue (per_commercial : ℚ) (num_commercials : ℕ) : ℚ :=
  per_commercial * num_commercials

def subscription_revenue (per_subscription : ℚ) (num_subscriptions : ℕ) : ℚ :=
  per_subscription * num_subscriptions

theorem lauren_revenue 
  (per_commercial : ℚ) 
  (per_subscription : ℚ) 
  (num_commercials : ℕ) 
  (num_subscriptions : ℕ) 
  (total_revenue : ℚ) :
  per_subscription = 1 →
  num_commercials = 100 →
  num_subscriptions = 27 →
  total_revenue = 77 →
  commercial_revenue per_commercial num_commercials + 
    subscription_revenue per_subscription num_subscriptions = total_revenue →
  per_commercial = 1/2 := by
sorry

end lauren_revenue_l1775_177568


namespace same_solution_d_value_l1775_177544

theorem same_solution_d_value (x : ℝ) (d : ℝ) : 
  (3 * x + 8 = 4) ∧ (d * x - 15 = -5) → d = -7.5 := by
  sorry

end same_solution_d_value_l1775_177544


namespace simplify_expression_l1775_177561

theorem simplify_expression (a : ℝ) : (3 * a)^2 * a^5 = 9 * a^7 := by
  sorry

end simplify_expression_l1775_177561


namespace maximize_product_l1775_177594

theorem maximize_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 50) :
  x^6 * y^3 ≤ (100/3)^6 * (50/3)^3 ∧
  x^6 * y^3 = (100/3)^6 * (50/3)^3 ↔ x = 100/3 ∧ y = 50/3 :=
by sorry

end maximize_product_l1775_177594


namespace vectors_form_basis_l1775_177569

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, 7)

theorem vectors_form_basis : 
  LinearIndependent ℝ (fun i => if i = 0 then e₁ else e₂) ∧ 
  Submodule.span ℝ {e₁, e₂} = ⊤ :=
sorry

end vectors_form_basis_l1775_177569


namespace homework_problem_l1775_177588

/-- Given a homework assignment with a total number of problems, 
    finished problems, and remaining pages, calculate the number of 
    problems per page assuming each page has the same number of problems. -/
def problems_per_page (total : ℕ) (finished : ℕ) (pages : ℕ) : ℕ :=
  (total - finished) / pages

/-- Theorem stating that for the given homework scenario, 
    there are 7 problems per page. -/
theorem homework_problem : 
  problems_per_page 40 26 2 = 7 := by
  sorry

end homework_problem_l1775_177588


namespace basketball_players_count_l1775_177549

def students_jumping_rope : ℕ := 6

def students_playing_basketball : ℕ := 4 * students_jumping_rope

theorem basketball_players_count : students_playing_basketball = 24 := by
  sorry

end basketball_players_count_l1775_177549


namespace age_puzzle_l1775_177510

theorem age_puzzle (A : ℕ) (x : ℕ) (h1 : A = 50) (h2 : 5 * (A + x) - 5 * (A - 5) = A) : x = 5 := by
  sorry

end age_puzzle_l1775_177510


namespace equation_solution_exists_l1775_177542

theorem equation_solution_exists : ∃ x : ℝ, 85 * x^2 + ((20 - 7) * 4)^3 / 2 - 15 * 7 = 75000 := by
  sorry

end equation_solution_exists_l1775_177542


namespace linear_system_solution_l1775_177538

/-- The system of linear equations ax + by = 10 -/
def linear_system (a b : ℝ) (x y : ℝ) : Prop := a * x + b * y = 10

theorem linear_system_solution :
  ∃ (a b : ℝ),
    (linear_system a b 2 4 ∧ linear_system a b 3 1) ∧
    (a = 3 ∧ b = 1) ∧
    (∀ x : ℝ, x > 10 / 3 → linear_system a b x 0 → linear_system a b x y → y < 0) :=
  sorry

end linear_system_solution_l1775_177538


namespace machine_selling_price_l1775_177522

/-- Calculates the selling price of a machine given its costs and profit percentage -/
def selling_price (purchase_price repair_cost transportation_charges profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transportation_charges
  let profit := (total_cost * profit_percentage) / 100
  total_cost + profit

/-- Theorem stating that the selling price of the machine is 22500 Rs -/
theorem machine_selling_price :
  selling_price 9000 5000 1000 50 = 22500 := by
  sorry

end machine_selling_price_l1775_177522


namespace kylie_coins_l1775_177560

theorem kylie_coins (initial_coins : ℕ) (received_coins1 : ℕ) (received_coins2 : ℕ) (given_away : ℕ) :
  initial_coins = 15 →
  received_coins1 = 13 →
  received_coins2 = 8 →
  given_away = 21 →
  initial_coins + received_coins1 + received_coins2 - given_away = 15 :=
by
  sorry

end kylie_coins_l1775_177560


namespace hua_luogeng_uses_golden_ratio_l1775_177521

-- Define the possible methods for optimal selection
inductive OptimalSelectionMethod
  | GoldenRatio
  | Mean
  | Mode
  | Median

-- Define Hua Luogeng's optimal selection method
def huaLuogengMethod : OptimalSelectionMethod := OptimalSelectionMethod.GoldenRatio

-- Theorem stating that Hua Luogeng's method uses the golden ratio
theorem hua_luogeng_uses_golden_ratio :
  huaLuogengMethod = OptimalSelectionMethod.GoldenRatio := by sorry

end hua_luogeng_uses_golden_ratio_l1775_177521


namespace two_sector_area_l1775_177577

/-- The area of a figure formed by two sectors of a circle -/
theorem two_sector_area (r : ℝ) (θ : ℝ) : 
  r = 15 → θ = 90 → 2 * (θ / 360) * π * r^2 = 112.5 * π := by sorry

end two_sector_area_l1775_177577


namespace part_one_part_two_l1775_177533

noncomputable section

-- Define the function f
def f (a k : ℝ) (x : ℝ) : ℝ := a^x - (k + 1) * a^(-x)

-- Define the function g
def g (a m : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2 * m * f a 0 x

-- Theorem for part (1)
theorem part_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ k : ℝ, ∀ x : ℝ, f a k x = -f a k (-x)) → k = 0 :=
sorry

-- Theorem for part (2)
theorem part_two (a m : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : f a 0 1 = 3/2) 
  (h4 : ∀ x : ℝ, x ≥ 0 → g a m x ≥ -6) 
  (h5 : ∃ x : ℝ, x ≥ 0 ∧ g a m x = -6) :
  m = 2 * Real.sqrt 2 ∨ m = -2 * Real.sqrt 2 :=
sorry

end

end part_one_part_two_l1775_177533


namespace larger_number_given_hcf_lcm_ratio_l1775_177581

theorem larger_number_given_hcf_lcm_ratio (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 84)
  (lcm_eq : Nat.lcm a b = 21)
  (ratio : a * 4 = b) :
  max a b = 84 := by
sorry

end larger_number_given_hcf_lcm_ratio_l1775_177581


namespace min_value_reciprocal_sum_l1775_177553

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 3*y = 1) :
  (1/x + 1/(3*y)) ≥ 4 := by
sorry

end min_value_reciprocal_sum_l1775_177553


namespace equation_solution_l1775_177525

theorem equation_solution (a : ℝ) : 
  (2*a + 4*(-1) = (-1) + 5*a) → 
  (a = -1) ∧ 
  (∀ y : ℝ, (-1)*y + 6 = 6*(-1) + 2*y → y = 4) := by
sorry

end equation_solution_l1775_177525


namespace fraction_of_fraction_of_fraction_fraction_multiplication_main_theorem_l1775_177545

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d :=
by sorry

theorem fraction_multiplication (a b c : ℚ) (n : ℕ) :
  (a * b * c : ℚ) * n = (n : ℚ) / ((1 / a) * (1 / b) * (1 / c)) :=
by sorry

theorem main_theorem : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 7 : ℚ) * 126 = 3 :=
by sorry

end fraction_of_fraction_of_fraction_fraction_multiplication_main_theorem_l1775_177545


namespace garden_dimensions_l1775_177528

/-- Represents a rectangular garden with walkways -/
structure Garden where
  L : ℝ  -- Length of the garden
  W : ℝ  -- Width of the garden
  w : ℝ  -- Width of the walkways
  h_L_gt_W : L > W  -- Length is greater than width

/-- The theorem representing the garden problem -/
theorem garden_dimensions (g : Garden) 
  (h1 : g.w * g.L = 228)  -- First walkway area
  (h2 : g.w * g.W = 117)  -- Second walkway area
  (h3 : g.w * g.L - g.w^2 = 219)  -- Third walkway area
  (h4 : g.w * g.L - (g.w * g.L - g.w^2) = g.w^2)  -- Difference between first and third walkway areas
  : g.L = 76 ∧ g.W = 42 ∧ g.w = 3 := by
  sorry

end garden_dimensions_l1775_177528


namespace solution_set_of_f_gt_zero_l1775_177515

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_monotone_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

-- Theorem statement
theorem solution_set_of_f_gt_zero
  (h_even : is_even f)
  (h_monotone : is_monotone_increasing_on_nonneg f)
  (h_f_one : f 1 = 0) :
  {x : ℝ | f x > 0} = {x : ℝ | x < -1 ∨ x > 1} := by
  sorry

end solution_set_of_f_gt_zero_l1775_177515


namespace cos_5pi_4_plus_x_l1775_177592

theorem cos_5pi_4_plus_x (x : ℝ) (h : Real.sin (π/4 - x) = -1/5) : 
  Real.cos (5*π/4 + x) = 1/5 := by
  sorry

end cos_5pi_4_plus_x_l1775_177592


namespace birthday_product_difference_l1775_177572

theorem birthday_product_difference (n : ℕ) (h : n = 7) : (n + 1)^2 - n^2 = 15 := by
  sorry

end birthday_product_difference_l1775_177572


namespace intersection_characterization_l1775_177501

-- Define the sets M and N
def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Define the intersection of N and the complement of M
def intersection : Set ℝ := N ∩ (Set.univ \ M)

-- State the theorem
theorem intersection_characterization : intersection = {x | 1 < x ∧ x ≤ 2} := by sorry

end intersection_characterization_l1775_177501


namespace unique_number_with_divisor_properties_l1775_177586

theorem unique_number_with_divisor_properties :
  ∀ (N p q r : ℕ) (α β γ : ℕ),
    (∃ (h_prime_p : Nat.Prime p) (h_prime_q : Nat.Prime q) (h_prime_r : Nat.Prime r),
      N = p^α * q^β * r^γ ∧
      p * q - r = 3 ∧
      p * r - q = 9 ∧
      (Nat.divisors (N / p)).card = (Nat.divisors N).card - 20 ∧
      (Nat.divisors (N / q)).card = (Nat.divisors N).card - 12 ∧
      (Nat.divisors (N / r)).card = (Nat.divisors N).card - 15) →
    N = 857500 := by
  sorry

end unique_number_with_divisor_properties_l1775_177586


namespace algebraic_identities_l1775_177541

theorem algebraic_identities :
  (∀ (a : ℝ), a ≠ 0 → 2 * a^5 + a^7 / a^2 = 3 * a^5) ∧
  (∀ (x y : ℝ), (x + y) * (x - y) + x * (2 * y - x) = 2 * x * y - y^2) :=
by sorry

end algebraic_identities_l1775_177541


namespace cube_sum_difference_l1775_177587

/-- Represents a face of a cube --/
inductive Face
| One
| Two
| Three
| Four
| Five
| Six

/-- A single small cube with numbered faces --/
structure SmallCube where
  faces : List Face
  face_count : faces.length = 6
  opposite_faces : 
    (Face.One ∈ faces ↔ Face.Two ∈ faces) ∧
    (Face.Three ∈ faces ↔ Face.Five ∈ faces) ∧
    (Face.Four ∈ faces ↔ Face.Six ∈ faces)

/-- The large 2×2×2 cube composed of small cubes --/
structure LargeCube where
  small_cubes : List SmallCube
  cube_count : small_cubes.length = 8

/-- The sum of numbers on the outer surface of the large cube --/
def outer_surface_sum (lc : LargeCube) : ℕ := sorry

/-- The maximum possible sum of numbers on the outer surface --/
def max_sum (lc : LargeCube) : ℕ := sorry

/-- The minimum possible sum of numbers on the outer surface --/
def min_sum (lc : LargeCube) : ℕ := sorry

/-- The main theorem to prove --/
theorem cube_sum_difference (lc : LargeCube) : 
  max_sum lc - min_sum lc = 24 := by sorry

end cube_sum_difference_l1775_177587


namespace gcd_of_repeated_numbers_l1775_177582

def repeated_number (n : ℕ) : ℕ := 1001001001 * n

theorem gcd_of_repeated_numbers :
  ∃ (m : ℕ), m > 0 ∧ m < 1000 ∧
  (∀ (n : ℕ), n > 0 ∧ n < 1000 → Nat.gcd (repeated_number m) (repeated_number n) = 1001001001) :=
sorry

end gcd_of_repeated_numbers_l1775_177582


namespace bakery_theft_l1775_177536

/-- The number of breads remaining after a thief takes their share -/
def breads_after_thief (initial : ℕ) : ℕ → ℕ
  | 0 => initial
  | n + 1 => (breads_after_thief initial n - 1) / 2

/-- The proposition that given 5 thieves and 3 breads remaining at the end, 
    the initial number of breads was 127 -/
theorem bakery_theft (initial : ℕ) :
  breads_after_thief initial 5 = 3 → initial = 127 := by
  sorry

#check bakery_theft

end bakery_theft_l1775_177536


namespace greatest_integer_quadratic_inequality_l1775_177508

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 13*n + 40 ≤ 0 ∧
  ∀ (m : ℤ), m^2 - 13*m + 40 ≤ 0 → m ≤ n :=
by sorry

end greatest_integer_quadratic_inequality_l1775_177508


namespace circle_center_l1775_177585

/-- The center of a circle with equation x^2 - 8x + y^2 - 4y = 16 is (4, 2) -/
theorem circle_center (x y : ℝ) : 
  (x^2 - 8*x + y^2 - 4*y = 16) → 
  (∃ r : ℝ, (x - 4)^2 + (y - 2)^2 = r^2) := by
sorry

end circle_center_l1775_177585


namespace total_chocolates_l1775_177550

theorem total_chocolates (bags : ℕ) (chocolates_per_bag : ℕ) 
  (h1 : bags = 20) (h2 : chocolates_per_bag = 156) :
  bags * chocolates_per_bag = 3120 :=
by sorry

end total_chocolates_l1775_177550


namespace factorization_sum_l1775_177518

theorem factorization_sum (d e f : ℤ) :
  (∀ x : ℝ, x^2 + 20*x + 75 = (x + d)*(x + e)) →
  (∀ x : ℝ, x^2 - 22*x + 120 = (x - e)*(x - f)) →
  d + e + f = 37 := by
  sorry

end factorization_sum_l1775_177518


namespace garden_length_l1775_177537

/-- Proves that a rectangular garden with length twice its width and perimeter 900 yards has a length of 300 yards -/
theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width →  -- The length is twice the width
  2 * length + 2 * width = 900 →  -- The perimeter is 900 yards
  length = 300 := by
sorry

end garden_length_l1775_177537


namespace janet_farmland_acreage_l1775_177517

/-- Represents Janet's farm and fertilizer production system -/
structure FarmSystem where
  horses : ℕ
  fertilizer_per_horse : ℕ
  fertilizer_per_acre : ℕ
  acres_spread_per_day : ℕ
  days_to_fertilize : ℕ

/-- Calculates the total acreage of Janet's farmland -/
def total_acreage (farm : FarmSystem) : ℕ :=
  farm.acres_spread_per_day * farm.days_to_fertilize

/-- Theorem: Janet's farmland is 100 acres given the specified conditions -/
theorem janet_farmland_acreage :
  let farm := FarmSystem.mk 80 5 400 4 25
  total_acreage farm = 100 := by
  sorry


end janet_farmland_acreage_l1775_177517


namespace polynomial_coefficient_identity_l1775_177523

theorem polynomial_coefficient_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
sorry

end polynomial_coefficient_identity_l1775_177523


namespace polar_to_rectangular_equivalence_l1775_177516

/-- Given a curve in polar coordinates ρ = 4sin θ, prove its equivalence to the rectangular form x² + y² - 4y = 0 -/
theorem polar_to_rectangular_equivalence :
  ∀ (x y ρ θ : ℝ), 
  (ρ = 4 * Real.sin θ) ∧ 
  (x = ρ * Real.cos θ) ∧ 
  (y = ρ * Real.sin θ) →
  (x^2 + y^2 - 4*y = 0) :=
by sorry

end polar_to_rectangular_equivalence_l1775_177516


namespace complex_modulus_problem_l1775_177509

theorem complex_modulus_problem (z : ℂ) : z = (1 + Complex.I) / (1 - Complex.I) + 2 * Complex.I → Complex.abs z = 3 := by
  sorry

end complex_modulus_problem_l1775_177509


namespace inverse_proportion_points_l1775_177539

def inverse_proportion (x y : ℝ) : Prop := y = -4 / x

theorem inverse_proportion_points :
  inverse_proportion (-2) 2 ∧
  ¬ inverse_proportion 1 4 ∧
  ¬ inverse_proportion (-2) (-2) ∧
  ¬ inverse_proportion (-4) (-1) := by
  sorry

end inverse_proportion_points_l1775_177539


namespace jimmy_water_consumption_l1775_177531

/-- Represents the amount of water Jimmy drinks each time in ounces -/
def water_per_time (times_per_day : ℕ) (days : ℕ) (total_gallons : ℚ) (ounce_to_gallon : ℚ) : ℚ :=
  (total_gallons / ounce_to_gallon) / (times_per_day * days)

/-- Theorem stating that Jimmy drinks 8 ounces of water each time -/
theorem jimmy_water_consumption :
  water_per_time 8 5 (5/2) (1/128) = 8 := by
sorry

end jimmy_water_consumption_l1775_177531


namespace pizza_slices_remaining_l1775_177584

theorem pizza_slices_remaining (total_slices : ℕ) (given_to_first_group : ℕ) (given_to_second_group : ℕ) :
  total_slices = 8 →
  given_to_first_group = 3 →
  given_to_second_group = 4 →
  total_slices - (given_to_first_group + given_to_second_group) = 1 := by
  sorry

end pizza_slices_remaining_l1775_177584


namespace line_points_equation_l1775_177571

/-- Given a line and two points on it, prove an equation relating to the x-coordinate of the first point -/
theorem line_points_equation (m n : ℝ) : 
  (∀ x y, x - 5/2 * y + 1 = 0 → 
    ((x = m ∧ y = n) ∨ (x = m + 1/2 ∧ y = n + 1)) → 
      m + 1 = m - 3) := by
  sorry

end line_points_equation_l1775_177571


namespace log_sum_equals_two_l1775_177532

theorem log_sum_equals_two : Real.log 4 + 2 * Real.log 5 = 2 * Real.log 10 := by
  sorry

end log_sum_equals_two_l1775_177532


namespace square_difference_l1775_177530

theorem square_difference (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) : a^2 - b^2 = -8 := by
  sorry

end square_difference_l1775_177530


namespace shortest_altitude_right_triangle_l1775_177547

/-- The shortest altitude of a right triangle with legs 9 and 12 is 7.2 -/
theorem shortest_altitude_right_triangle :
  let a : ℝ := 9
  let b : ℝ := 12
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let area : ℝ := (1/2) * a * b
  let h : ℝ := (2 * area) / c
  h = 7.2 := by sorry

end shortest_altitude_right_triangle_l1775_177547


namespace max_planes_15_points_l1775_177514

/-- The maximum number of planes determined by 15 points in space, where no four points are coplanar -/
def max_planes (n : ℕ) : ℕ :=
  Nat.choose n 3

/-- Theorem stating that the maximum number of planes determined by 15 points in space, 
    where no four points are coplanar, is equal to 455 -/
theorem max_planes_15_points : max_planes 15 = 455 := by
  sorry

end max_planes_15_points_l1775_177514


namespace derivative_y_l1775_177503

noncomputable def y (x : ℝ) : ℝ := (Real.cos (2 * x)) ^ ((Real.log (Real.cos (2 * x))) / 4)

theorem derivative_y (x : ℝ) (h : Real.cos (2 * x) ≠ 0) :
  deriv y x = -(Real.cos (2 * x)) ^ ((Real.log (Real.cos (2 * x))) / 4) * 
               Real.tan (2 * x) * 
               Real.log (Real.cos (2 * x)) :=
by sorry

end derivative_y_l1775_177503


namespace largest_integer_below_root_l1775_177591

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 6

theorem largest_integer_below_root :
  ∃ (x₀ : ℝ), f x₀ = 0 ∧
  (∀ x > 0, x < x₀ → f x < 0) ∧
  (∀ x > x₀, f x > 0) ∧
  (∀ n : ℤ, (n : ℝ) ≤ x₀ → n ≤ 4) ∧
  ((4 : ℝ) ≤ x₀) :=
sorry

end largest_integer_below_root_l1775_177591


namespace circle_equation_l1775_177505

theorem circle_equation (x y : ℝ) : 
  (∃ c : ℝ, x^2 + (y - c)^2 = 1 ∧ 1^2 + (2 - c)^2 = 1) → 
  x^2 + (y - 2)^2 = 1 := by
sorry

end circle_equation_l1775_177505


namespace sine_sum_identity_l1775_177579

theorem sine_sum_identity (α β γ : ℝ) (h : α + β + γ = 0) :
  Real.sin α + Real.sin β + Real.sin γ = -4 * Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2) := by
  sorry

end sine_sum_identity_l1775_177579


namespace gcd_set_divisors_l1775_177556

theorem gcd_set_divisors (a b c d : ℕ+) (h1 : a * d ≠ b * c) (h2 : Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd a d = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd b d = 1 ∧ Nat.gcd c d = 1) :
  ∃ k : ℕ, {x : ℕ | ∃ n : ℕ+, x = Nat.gcd (a * n + b) (c * n + d)} = {x : ℕ | x ∣ k} := by
  sorry


end gcd_set_divisors_l1775_177556


namespace constant_value_l1775_177504

theorem constant_value (t : ℝ) (C : ℝ) : 
  let x := 1 - 4 * t
  let y := 2 * t + C
  (x = y → t = 0.5) → C = -2 := by
  sorry

end constant_value_l1775_177504


namespace prop_a_false_prop_b_true_prop_c_true_prop_d_true_propositions_bcd_true_a_false_l1775_177513

-- Proposition A (false)
theorem prop_a_false : ¬ (∀ a b : ℝ, a > b → 1 / b > 1 / a) := by sorry

-- Proposition B (true)
theorem prop_b_true : ∀ a b : ℝ, a > b ∧ b > 0 → a^2 > a*b ∧ a*b > b^2 := by sorry

-- Proposition C (true)
theorem prop_c_true : ∀ a b c : ℝ, c ≠ 0 → (a*c^2 > b*c^2 → a > b) := by sorry

-- Proposition D (true)
theorem prop_d_true : ∀ a b c d : ℝ, a > b ∧ b > 0 ∧ c < d ∧ d < 0 → 1 / (a - c) < 1 / (b - d) := by sorry

-- Combined theorem
theorem propositions_bcd_true_a_false : 
  (¬ (∀ a b : ℝ, a > b → 1 / b > 1 / a)) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > a*b ∧ a*b > b^2) ∧
  (∀ a b c : ℝ, c ≠ 0 → (a*c^2 > b*c^2 → a > b)) ∧
  (∀ a b c d : ℝ, a > b ∧ b > 0 ∧ c < d ∧ d < 0 → 1 / (a - c) < 1 / (b - d)) := by
  exact ⟨prop_a_false, prop_b_true, prop_c_true, prop_d_true⟩

end prop_a_false_prop_b_true_prop_c_true_prop_d_true_propositions_bcd_true_a_false_l1775_177513


namespace peters_vacation_savings_l1775_177565

/-- Peter's vacation savings problem -/
theorem peters_vacation_savings 
  (current_savings : ℕ) 
  (monthly_savings : ℕ) 
  (months_to_wait : ℕ) 
  (h1 : current_savings = 2900)
  (h2 : monthly_savings = 700)
  (h3 : months_to_wait = 3) :
  current_savings + monthly_savings * months_to_wait = 5000 :=
by sorry

end peters_vacation_savings_l1775_177565


namespace max_peak_consumption_l1775_177548

theorem max_peak_consumption (original_price peak_price off_peak_price total_consumption : ℝ)
  (h1 : original_price = 0.52)
  (h2 : peak_price = 0.55)
  (h3 : off_peak_price = 0.35)
  (h4 : total_consumption = 200)
  (h5 : ∀ x : ℝ, x ≥ 0 ∧ x ≤ total_consumption →
    (x * peak_price + (total_consumption - x) * off_peak_price) ≤ 0.9 * (total_consumption * original_price)) :
  ∃ max_peak : ℝ, max_peak = 118 ∧
    ∀ y : ℝ, y > max_peak →
      (y * peak_price + (total_consumption - y) * off_peak_price) > 0.9 * (total_consumption * original_price) :=
by sorry

end max_peak_consumption_l1775_177548


namespace sqrt_equation_solution_l1775_177527

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 + Real.sqrt x) = 4 → x = 121 := by
  sorry

end sqrt_equation_solution_l1775_177527


namespace total_clothing_pieces_l1775_177558

theorem total_clothing_pieces (shirts trousers : ℕ) 
  (h1 : shirts = 589) 
  (h2 : trousers = 345) : 
  shirts + trousers = 934 := by
  sorry

end total_clothing_pieces_l1775_177558


namespace root_properties_l1775_177599

theorem root_properties : 
  (∃ x : ℝ, x^3 = -9 ∧ x = -3) ∧ 
  (∀ y : ℝ, y^2 = 9 ↔ y = 3 ∨ y = -3) := by
  sorry

end root_properties_l1775_177599


namespace seller_loss_is_30_l1775_177507

/-- Represents the transaction between a seller and a buyer -/
structure Transaction where
  goods_value : ℕ
  payment : ℕ
  counterfeit : Bool

/-- Calculates the seller's loss given a transaction -/
def seller_loss (t : Transaction) : ℕ :=
  if t.counterfeit then
    t.payment + (t.payment - t.goods_value)
  else
    0

/-- Theorem stating that the seller's loss is 30 rubles given the specific transaction -/
theorem seller_loss_is_30 (t : Transaction) 
  (h1 : t.goods_value = 10)
  (h2 : t.payment = 25)
  (h3 : t.counterfeit = true) : 
  seller_loss t = 30 := by
  sorry

#eval seller_loss { goods_value := 10, payment := 25, counterfeit := true }

end seller_loss_is_30_l1775_177507


namespace mika_stickers_total_l1775_177563

/-- The total number of stickers Mika has -/
def total_stickers (initial bought birthday sister mother : ℝ) : ℝ :=
  initial + bought + birthday + sister + mother

/-- Theorem stating that Mika has 130.0 stickers in total -/
theorem mika_stickers_total :
  total_stickers 20.0 26.0 20.0 6.0 58.0 = 130.0 := by
  sorry

end mika_stickers_total_l1775_177563


namespace unique_cube_root_l1775_177511

theorem unique_cube_root (M : ℕ+) : 18^3 * 50^3 = 30^3 * M^3 ↔ M = 30 := by
  sorry

end unique_cube_root_l1775_177511


namespace range_of_a_l1775_177524

-- Define the propositions p and q
def p (x : ℝ) : Prop := |4 - x| ≤ 6
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≥ 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, ¬(p x) → q x a) →
  (∃ x : ℝ, p x ∧ q x a) →
  (0 < a ∧ a ≤ 3) :=
sorry

end range_of_a_l1775_177524


namespace topsoil_cost_l1775_177555

/-- The cost of topsoil in euros per cubic meter -/
def cost_per_cubic_meter : ℝ := 12

/-- The volume of topsoil to be purchased in cubic meters -/
def volume : ℝ := 3

/-- The total cost of purchasing the topsoil -/
def total_cost : ℝ := cost_per_cubic_meter * volume

/-- Theorem stating that the total cost of purchasing 3 cubic meters of topsoil is 36 euros -/
theorem topsoil_cost : total_cost = 36 := by
  sorry

end topsoil_cost_l1775_177555


namespace infinite_series_sum_l1775_177512

theorem infinite_series_sum : 
  (∑' n : ℕ, (n^2 + 3*n + 2) / (n * (n + 1) * (n + 3))) = 1 := by
  sorry

end infinite_series_sum_l1775_177512


namespace square_difference_equality_l1775_177566

theorem square_difference_equality : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end square_difference_equality_l1775_177566


namespace arithmetic_sequence_sum_l1775_177502

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 + a 10 = 16 → a 4 + a 6 + a 8 = 24 := by
  sorry

end arithmetic_sequence_sum_l1775_177502


namespace power_inequality_l1775_177595

theorem power_inequality : 0.2^0.3 < 0.3^0.3 ∧ 0.3^0.3 < 0.3^0.2 := by
  sorry

end power_inequality_l1775_177595


namespace farthest_point_l1775_177597

def points : List (ℝ × ℝ) := [(0, 7), (2, 3), (-4, 1), (5, -5), (7, 0)]

def distance_squared (p : ℝ × ℝ) : ℝ :=
  p.1 ^ 2 + p.2 ^ 2

theorem farthest_point :
  ∀ p ∈ points, distance_squared (5, -5) ≥ distance_squared p :=
by sorry

end farthest_point_l1775_177597


namespace positive_integer_divisible_by_15_with_sqrt_between_33_and_33_5_l1775_177570

theorem positive_integer_divisible_by_15_with_sqrt_between_33_and_33_5 :
  ∃ n : ℕ+, 
    (∃ k : ℕ, n = 15 * k) ∧ 
    (33 * 33 : ℝ) ≤ (n : ℝ) ∧ (n : ℝ) < (33.5 * 33.5) ∧
    (n = 1095 ∨ n = 1110) :=
by sorry

end positive_integer_divisible_by_15_with_sqrt_between_33_and_33_5_l1775_177570
