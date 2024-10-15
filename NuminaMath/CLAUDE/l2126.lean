import Mathlib

namespace NUMINAMATH_CALUDE_queenie_earnings_l2126_212660

/-- Calculates the total earnings for a part-time clerk with overtime -/
def total_earnings (daily_rate : ℕ) (overtime_rate : ℕ) (days_worked : ℕ) (overtime_hours : ℕ) : ℕ :=
  daily_rate * days_worked + overtime_rate * overtime_hours

/-- Proves that Queenie's total earnings are $770 -/
theorem queenie_earnings : total_earnings 150 5 5 4 = 770 := by
  sorry

end NUMINAMATH_CALUDE_queenie_earnings_l2126_212660


namespace NUMINAMATH_CALUDE_savings_calculation_l2126_212628

theorem savings_calculation (income expenditure savings : ℕ) : 
  (income : ℚ) / expenditure = 8 / 7 →
  income = 40000 →
  savings = income - expenditure →
  savings = 5000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l2126_212628


namespace NUMINAMATH_CALUDE_initial_retail_price_l2126_212649

/-- Calculates the initial retail price of a machine given wholesale price, shipping, tax, discount, and profit margin. -/
theorem initial_retail_price
  (wholesale_with_shipping : ℝ)
  (shipping : ℝ)
  (tax_rate : ℝ)
  (discount_rate : ℝ)
  (profit_rate : ℝ)
  (h1 : wholesale_with_shipping = 90)
  (h2 : shipping = 10)
  (h3 : tax_rate = 0.05)
  (h4 : discount_rate = 0.10)
  (h5 : profit_rate = 0.20) :
  let wholesale := (wholesale_with_shipping - shipping) / (1 + tax_rate)
  let cost := wholesale_with_shipping
  let initial_price := cost / (1 - profit_rate - discount_rate + discount_rate * profit_rate)
  initial_price = 125 := by sorry

end NUMINAMATH_CALUDE_initial_retail_price_l2126_212649


namespace NUMINAMATH_CALUDE_platform_length_l2126_212609

/-- The length of a platform given a goods train's speed, length, and time to cross the platform. -/
theorem platform_length (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 → 
  train_length = 280.0416 → 
  crossing_time = 26 → 
  (train_speed * 1000 / 3600 * crossing_time) - train_length = 239.9584 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l2126_212609


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2126_212641

theorem quadratic_root_range (k : ℝ) (α β : ℝ) : 
  (∃ x, 7 * x^2 - (k + 13) * x + k^2 - k - 2 = 0) →
  (∀ x, 7 * x^2 - (k + 13) * x + k^2 - k - 2 = 0 → x = α ∨ x = β) →
  0 < α → α < 1 → 1 < β → β < 2 →
  (3 < k ∧ k < 4) ∨ (-2 < k ∧ k < -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2126_212641


namespace NUMINAMATH_CALUDE_f_magnitude_relationship_l2126_212699

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_decreasing : ∀ x₁ x₂, x₁ ∈ Set.Ici (0 : ℝ) → x₂ ∈ Set.Ici (0 : ℝ) → x₁ ≠ x₂ → 
  (x₁ - x₂) * (f x₁ - f x₂) < 0

-- State the theorem to be proved
theorem f_magnitude_relationship : f 0 > f (-2) ∧ f (-2) > f 3 :=
sorry

end NUMINAMATH_CALUDE_f_magnitude_relationship_l2126_212699


namespace NUMINAMATH_CALUDE_interval_condition_l2126_212627

theorem interval_condition (x : ℝ) : 
  (2 < 3*x ∧ 3*x < 3 ∧ 2 < 4*x ∧ 4*x < 3) ↔ (2/3 < x ∧ x < 3/4) := by
  sorry

end NUMINAMATH_CALUDE_interval_condition_l2126_212627


namespace NUMINAMATH_CALUDE_correct_oranges_to_put_back_l2126_212618

/-- Represents the fruit selection problem -/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  desired_avg_price : ℚ

/-- Calculates the number of oranges to put back -/
def oranges_to_put_back (fs : FruitSelection) : ℕ :=
  sorry

/-- Theorem stating the correct number of oranges to put back -/
theorem correct_oranges_to_put_back (fs : FruitSelection) 
  (h1 : fs.apple_price = 40/100)
  (h2 : fs.orange_price = 60/100)
  (h3 : fs.total_fruits = 10)
  (h4 : fs.initial_avg_price = 56/100)
  (h5 : fs.desired_avg_price = 50/100) :
  oranges_to_put_back fs = 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_oranges_to_put_back_l2126_212618


namespace NUMINAMATH_CALUDE_max_sum_of_squares_max_sum_of_squares_achievable_l2126_212666

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 20 →
  a * b + c + d = 90 →
  a * d + b * c = 210 →
  c * d = 125 →
  a^2 + b^2 + c^2 + d^2 ≤ 1450 := by
  sorry

theorem max_sum_of_squares_achievable : 
  ∃ (a b c d : ℝ),
    a + b = 20 ∧
    a * b + c + d = 90 ∧
    a * d + b * c = 210 ∧
    c * d = 125 ∧
    a^2 + b^2 + c^2 + d^2 = 1450 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_max_sum_of_squares_achievable_l2126_212666


namespace NUMINAMATH_CALUDE_inequality_implications_l2126_212606

theorem inequality_implications (a b : ℝ) :
  (b > 0 ∧ 0 > a → 1/a < 1/b) ∧
  (0 > a ∧ a > b → 1/a < 1/b) ∧
  (a > b ∧ b > 0 → 1/a < 1/b) ∧
  ¬(∀ a b : ℝ, a > 0 ∧ 0 > b → 1/a < 1/b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_implications_l2126_212606


namespace NUMINAMATH_CALUDE_card_position_retained_l2126_212634

theorem card_position_retained (n : ℕ) : 
  (∃ (total_cards : ℕ), 
    total_cards = 2 * n ∧ 
    201 ≤ n ∧ 
    (∀ (card : ℕ), card ≤ total_cards → 
      (card ≤ n → (card + n).mod 2 = 1) ∧ 
      (n < card → card.mod 2 = 0))) →
  201 = n :=
by sorry

end NUMINAMATH_CALUDE_card_position_retained_l2126_212634


namespace NUMINAMATH_CALUDE_renovation_constraint_l2126_212675

/-- Represents the constraint condition for hiring workers in a renovation project. -/
theorem renovation_constraint (x y : ℕ) : 
  (50 : ℝ) * x + (40 : ℝ) * y ≤ 2000 ↔ (5 : ℝ) * x + (4 : ℝ) * y ≤ 200 :=
by sorry

#check renovation_constraint

end NUMINAMATH_CALUDE_renovation_constraint_l2126_212675


namespace NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus_1_l2126_212671

theorem quadratic_root_sqrt5_minus_1 :
  ∃ (a b c : ℚ), (a ≠ 0) ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x^2 + 2*x - 6 = 0) ∧
  (Real.sqrt 5 - 1)^2 + 2*(Real.sqrt 5 - 1) - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus_1_l2126_212671


namespace NUMINAMATH_CALUDE_billy_sleep_theorem_l2126_212676

def night1_sleep : ℕ := 6

def night2_sleep : ℕ := night1_sleep + 2

def night3_sleep : ℕ := night2_sleep / 2

def night4_sleep : ℕ := night3_sleep * 3

def total_sleep : ℕ := night1_sleep + night2_sleep + night3_sleep + night4_sleep

theorem billy_sleep_theorem : total_sleep = 30 := by
  sorry

end NUMINAMATH_CALUDE_billy_sleep_theorem_l2126_212676


namespace NUMINAMATH_CALUDE_smallest_digit_correction_l2126_212687

def original_sum : ℕ := 356 + 781 + 492
def incorrect_sum : ℕ := 1529
def corrected_number : ℕ := 256

theorem smallest_digit_correction :
  (original_sum = incorrect_sum + 100) ∧
  (corrected_number + 781 + 492 = incorrect_sum) ∧
  (∀ n : ℕ, n < 356 → n > corrected_number → n + 781 + 492 ≠ incorrect_sum) := by
  sorry

end NUMINAMATH_CALUDE_smallest_digit_correction_l2126_212687


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2126_212647

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (11^3 + 3^3 : ℝ)^(1/3) → 
  ∃ (n : ℤ), n = 11 ∧ ∀ (m : ℤ), |x - n| ≤ |x - m| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2126_212647


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l2126_212616

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set M
def M : Set Nat := {1, 2, 3}

-- Define set N
def N : Set Nat := {2, 3, 4}

-- Theorem statement
theorem complement_intersection_equals_set :
  (M ∩ N)ᶜ = {1, 4} :=
by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l2126_212616


namespace NUMINAMATH_CALUDE_jenna_stamps_problem_l2126_212604

theorem jenna_stamps_problem :
  Nat.gcd 1260 1470 = 210 := by
  sorry

end NUMINAMATH_CALUDE_jenna_stamps_problem_l2126_212604


namespace NUMINAMATH_CALUDE_drunk_driving_wait_time_l2126_212679

theorem drunk_driving_wait_time (p₀ r : ℝ) (h1 : p₀ = 89) (h2 : 61 = 89 * Real.exp (2 * r)) : 
  ⌈Real.log (20 / 89) / r⌉ = 8 := by
  sorry

end NUMINAMATH_CALUDE_drunk_driving_wait_time_l2126_212679


namespace NUMINAMATH_CALUDE_largest_prime_divisor_test_l2126_212623

theorem largest_prime_divisor_test (n : ℕ) (h1 : 1100 ≤ n) (h2 : n ≤ 1150) :
  Prime n → ∀ p, Prime p → p > 31 → ¬(p ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_test_l2126_212623


namespace NUMINAMATH_CALUDE_sides_formula_l2126_212653

/-- The number of sides in the nth figure of a sequence starting with a hexagon,
    where each subsequent figure has 5 more sides than the previous one. -/
def sides (n : ℕ) : ℕ := 5 * n + 1

/-- Theorem stating that the number of sides in the nth figure is 5n + 1 -/
theorem sides_formula (n : ℕ) : sides n = 5 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_sides_formula_l2126_212653


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l2126_212691

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = a * b - 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = x * y - 1 → a + 2 * b ≤ x + 2 * y ∧
  ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = a₀ * b₀ - 1 ∧ a₀ + 2 * b₀ = 5 + 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l2126_212691


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2126_212626

theorem sufficient_but_not_necessary : 
  (∀ x₁ x₂ : ℝ, x₁ > 3 ∧ x₂ > 3 → x₁ * x₂ > 9 ∧ x₁ + x₂ > 6) ∧
  (∃ x₁ x₂ : ℝ, x₁ * x₂ > 9 ∧ x₁ + x₂ > 6 ∧ ¬(x₁ > 3 ∧ x₂ > 3)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2126_212626


namespace NUMINAMATH_CALUDE_no_root_intersection_l2126_212622

theorem no_root_intersection : ∀ x : ℝ,
  (∃ y : ℝ, y = Real.sqrt x ∧ y = Real.sqrt (x - 6) + 1) →
  x^2 - 5*x + 6 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_root_intersection_l2126_212622


namespace NUMINAMATH_CALUDE_mateo_absent_days_l2126_212658

/-- Calculates the number of days not worked given weekly salary, work days per week, and deducted salary -/
def daysNotWorked (weeklySalary workDaysPerWeek deductedSalary : ℚ) : ℕ :=
  let dailySalary := weeklySalary / workDaysPerWeek
  let exactDaysNotWorked := deductedSalary / dailySalary
  (exactDaysNotWorked + 1/2).floor.toNat

/-- Proves that given the specific conditions, the number of days not worked is 2 -/
theorem mateo_absent_days :
  daysNotWorked 791 5 339 = 2 := by
  sorry

#eval daysNotWorked 791 5 339

end NUMINAMATH_CALUDE_mateo_absent_days_l2126_212658


namespace NUMINAMATH_CALUDE_fraction_is_positive_integer_l2126_212633

theorem fraction_is_positive_integer (p : ℕ+) :
  (↑p : ℚ) = 3 ↔ (∃ (k : ℕ+), ((4 * p + 35) : ℚ) / ((3 * p - 8) : ℚ) = ↑k) := by
  sorry

end NUMINAMATH_CALUDE_fraction_is_positive_integer_l2126_212633


namespace NUMINAMATH_CALUDE_sin_cos_sixty_degrees_l2126_212693

theorem sin_cos_sixty_degrees :
  Real.sin (π / 3) = Real.sqrt 3 / 2 ∧ Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixty_degrees_l2126_212693


namespace NUMINAMATH_CALUDE_expression_evaluation_l2126_212696

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  2 * (x^2 + 2*x*y) - 2*x^2 - x*y = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2126_212696


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l2126_212615

-- Define the custom operation
def custom_op (x y : ℚ) : ℚ := (x * y / 3) - 2 * y

-- Theorem statement
theorem smallest_integer_solution :
  ∀ a : ℤ, (custom_op 2 (↑a) ≤ 2) → (a ≥ -1) ∧ 
  ∀ b : ℤ, (b < -1) → (custom_op 2 (↑b) > 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l2126_212615


namespace NUMINAMATH_CALUDE_calculate_expression_l2126_212612

theorem calculate_expression : 
  3 / Real.sqrt 3 - (Real.pi + Real.sqrt 3) ^ 0 - Real.sqrt 27 + |Real.sqrt 3 - 2| = -3 * Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2126_212612


namespace NUMINAMATH_CALUDE_four_digit_to_two_digit_ratio_l2126_212688

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≥ 1 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a TwoDigitNumber to its numerical value -/
def TwoDigitNumber.toNat (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

/-- Converts a TwoDigitNumber to a four-digit number by repeating it -/
def TwoDigitNumber.toFourDigitNumber (n : TwoDigitNumber) : Nat :=
  1000 * n.tens + 100 * n.ones + 10 * n.tens + n.ones

/-- Theorem stating the ratio of the four-digit number to the original two-digit number is 101 -/
theorem four_digit_to_two_digit_ratio (n : TwoDigitNumber) :
    (n.toFourDigitNumber : ℚ) / (n.toNat : ℚ) = 101 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_to_two_digit_ratio_l2126_212688


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2126_212689

theorem inequality_system_solution (m n : ℝ) : 
  (∀ x, -1 < x ∧ x < 3 ↔ (x - 3*m < 0 ∧ n - 2*x < 0)) →
  (m + n)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2126_212689


namespace NUMINAMATH_CALUDE_trash_can_prices_min_A_type_cans_l2126_212656

-- Define variables for trash can prices
variable (price_A : ℝ) (price_B : ℝ)

-- Define the cost equations
def cost_equation_1 (price_A price_B : ℝ) : Prop :=
  3 * price_A + 4 * price_B = 580

def cost_equation_2 (price_A price_B : ℝ) : Prop :=
  6 * price_A + 5 * price_B = 860

-- Define the total number of trash cans
def total_cans : ℕ := 200

-- Define the budget constraint
def budget : ℝ := 15000

-- Theorem for part 1
theorem trash_can_prices :
  cost_equation_1 price_A price_B ∧ cost_equation_2 price_A price_B →
  price_A = 60 ∧ price_B = 100 := by sorry

-- Theorem for part 2
theorem min_A_type_cans (num_A : ℕ) :
  num_A * price_A + (total_cans - num_A) * price_B ≤ budget →
  num_A ≥ 125 := by sorry

end NUMINAMATH_CALUDE_trash_can_prices_min_A_type_cans_l2126_212656


namespace NUMINAMATH_CALUDE_not_always_achievable_all_plus_l2126_212631

/-- Represents a sign in a cell of the grid -/
inductive Sign
| Plus
| Minus

/-- Represents the grid -/
def Grid := Fin 8 → Fin 8 → Sign

/-- Represents a square subgrid -/
structure Square where
  size : Nat
  row : Fin 8
  col : Fin 8

/-- Checks if a square is valid (3x3 or 4x4) -/
def Square.isValid (s : Square) : Prop :=
  (s.size = 3 ∨ s.size = 4) ∧
  s.row + s.size ≤ 8 ∧
  s.col + s.size ≤ 8

/-- Applies an operation to the grid -/
def applyOperation (g : Grid) (s : Square) : Grid :=
  sorry

/-- Checks if a grid is filled with only Plus signs -/
def isAllPlus (g : Grid) : Prop :=
  ∀ i j, g i j = Sign.Plus

/-- Main theorem: It's not always possible to achieve all Plus signs -/
theorem not_always_achievable_all_plus :
  ∃ (initial : Grid), ¬∃ (operations : List Square),
    (∀ s ∈ operations, s.isValid) →
    isAllPlus (operations.foldl applyOperation initial) :=
  sorry

end NUMINAMATH_CALUDE_not_always_achievable_all_plus_l2126_212631


namespace NUMINAMATH_CALUDE_tetrahedron_symmetry_l2126_212695

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- The center of mass of a tetrahedron -/
def centerOfMass (t : Tetrahedron) : Point3D := sorry

/-- The center of the circumscribed sphere of a tetrahedron -/
def circumCenter (t : Tetrahedron) : Point3D := sorry

/-- Check if a line intersects an edge of a tetrahedron -/
def intersectsEdge (l : Line3D) (p1 p2 : Point3D) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Theorem statement -/
theorem tetrahedron_symmetry (t : Tetrahedron) 
  (l : Line3D) 
  (h1 : l.point = centerOfMass t) 
  (h2 : l.point = circumCenter t) 
  (h3 : intersectsEdge l t.A t.B) 
  (h4 : intersectsEdge l t.C t.D) : 
  distance t.A t.C = distance t.B t.D ∧ 
  distance t.A t.D = distance t.B t.C := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_symmetry_l2126_212695


namespace NUMINAMATH_CALUDE_min_effort_for_mop_l2126_212697

/-- Represents the effort and points for each exam --/
structure ExamEffort :=
  (effort : ℕ)
  (points : ℕ)

/-- Defines the problem of Alex making MOP --/
def MakeMOP (amc : ExamEffort) (aime : ExamEffort) (usamo : ExamEffort) : Prop :=
  let total_points := amc.points + aime.points
  let total_effort := amc.effort + aime.effort + usamo.effort
  total_points ≥ 200 ∧ usamo.points ≥ 21 ∧ total_effort = 320

/-- Theorem stating the minimum effort required for Alex to make MOP --/
theorem min_effort_for_mop :
  ∃ (amc aime usamo : ExamEffort),
    amc.effort = 3 * (amc.points / 6) ∧
    aime.effort = 7 * (aime.points / 10) ∧
    usamo.effort = 10 * usamo.points ∧
    MakeMOP amc aime usamo ∧
    ∀ (amc' aime' usamo' : ExamEffort),
      amc'.effort = 3 * (amc'.points / 6) →
      aime'.effort = 7 * (aime'.points / 10) →
      usamo'.effort = 10 * usamo'.points →
      MakeMOP amc' aime' usamo' →
      amc'.effort + aime'.effort + usamo'.effort ≥ 320 :=
by
  sorry

end NUMINAMATH_CALUDE_min_effort_for_mop_l2126_212697


namespace NUMINAMATH_CALUDE_existence_of_n_l2126_212638

theorem existence_of_n (p : ℕ) (a k : ℕ+) (h_prime : Nat.Prime p) 
  (h_bound : p ^ a.val < k.val ∧ k.val < 2 * p ^ a.val) :
  ∃ n : ℕ, n < p ^ (2 * a.val) ∧ 
    (Nat.choose n k.val) % (p ^ a.val) = n % (p ^ a.val) ∧ 
    n % (p ^ a.val) = k.val % (p ^ a.val) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_l2126_212638


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l2126_212684

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 2*y + 6 = 0

/-- Point P -/
def P : ℝ × ℝ := (1, -2)

/-- First tangent line equation -/
def tangent1 (x y : ℝ) : Prop :=
  5*x - 12*y - 29 = 0

/-- Second tangent line equation -/
def tangent2 (x : ℝ) : Prop :=
  x = 1

/-- Theorem stating that the tangent lines from P to the circle have the given equations -/
theorem tangent_lines_to_circle :
  ∃ (x y : ℝ), circle_equation x y ∧
  ((tangent1 x y ∧ (x, y) ≠ P) ∨ (tangent2 x ∧ y ≠ -2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l2126_212684


namespace NUMINAMATH_CALUDE_line_translation_l2126_212657

/-- A line in the 2D plane represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Vertical translation of a line. -/
def verticalTranslate (l : Line) (d : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - d }

theorem line_translation (x : ℝ) :
  let original := Line.mk 2 0
  let transformed := Line.mk 2 (-3)
  transformed = verticalTranslate original 3 := by sorry

end NUMINAMATH_CALUDE_line_translation_l2126_212657


namespace NUMINAMATH_CALUDE_courier_travel_times_l2126_212640

/-- Represents a courier with their travel times -/
structure Courier where
  meetingTime : ℝ
  remainingTime : ℝ

/-- Proves that given the conditions, the couriers' total travel times are 28 and 21 hours -/
theorem courier_travel_times (c1 c2 : Courier) 
  (h1 : c1.remainingTime = 16)
  (h2 : c2.remainingTime = 9)
  (h3 : c1.meetingTime = c2.meetingTime)
  (h4 : c1.meetingTime * (1 / c1.meetingTime + 1 / c2.meetingTime) = 1) :
  (c1.meetingTime + c1.remainingTime = 28) ∧ 
  (c2.meetingTime + c2.remainingTime = 21) := by
  sorry

#check courier_travel_times

end NUMINAMATH_CALUDE_courier_travel_times_l2126_212640


namespace NUMINAMATH_CALUDE_bruce_fruit_purchase_total_l2126_212682

/-- Calculates the discounted price for a fruit purchase -/
def discountedPrice (quantity : ℕ) (pricePerKg : ℚ) (discountPercentage : ℚ) : ℚ :=
  let originalPrice := quantity * pricePerKg
  originalPrice - (originalPrice * discountPercentage / 100)

/-- Represents Bruce's fruit purchases -/
structure FruitPurchase where
  grapes : ℕ × ℚ × ℚ
  mangoes : ℕ × ℚ × ℚ
  oranges : ℕ × ℚ × ℚ
  apples : ℕ × ℚ × ℚ

/-- Calculates the total amount paid for all fruit purchases -/
def totalAmountPaid (purchase : FruitPurchase) : ℚ :=
  discountedPrice purchase.grapes.1 purchase.grapes.2.1 purchase.grapes.2.2 +
  discountedPrice purchase.mangoes.1 purchase.mangoes.2.1 purchase.mangoes.2.2 +
  discountedPrice purchase.oranges.1 purchase.oranges.2.1 purchase.oranges.2.2 +
  discountedPrice purchase.apples.1 purchase.apples.2.1 purchase.apples.2.2

theorem bruce_fruit_purchase_total :
  let purchase : FruitPurchase := {
    grapes := (9, 70, 10),
    mangoes := (7, 55, 5),
    oranges := (5, 45, 15),
    apples := (3, 80, 20)
  }
  totalAmountPaid purchase = 1316.25 := by
  sorry

end NUMINAMATH_CALUDE_bruce_fruit_purchase_total_l2126_212682


namespace NUMINAMATH_CALUDE_simon_red_stamps_count_l2126_212642

/-- The number of red stamps Simon has -/
def simon_red_stamps : ℕ := 34

/-- The number of white stamps Peter has -/
def peter_white_stamps : ℕ := 80

/-- The selling price of a red stamp in dollars -/
def red_stamp_price : ℚ := 1/2

/-- The selling price of a white stamp in dollars -/
def white_stamp_price : ℚ := 1/5

/-- The difference in the amount of money they make in dollars -/
def money_difference : ℚ := 1

theorem simon_red_stamps_count : 
  (simon_red_stamps : ℚ) * red_stamp_price - (peter_white_stamps : ℚ) * white_stamp_price = money_difference :=
by sorry

end NUMINAMATH_CALUDE_simon_red_stamps_count_l2126_212642


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l2126_212643

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 4 < 3 * ((6 * n + 15) / 6)) → n ≥ -1 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l2126_212643


namespace NUMINAMATH_CALUDE_ellipse_equation_l2126_212611

/-- Given two ellipses C1 and C2, where C1 is defined by x²/4 + y² = 1,
    C2 has the same eccentricity as C1, and the minor axis of C2 is
    the same as the major axis of C1, prove that the equation of C2 is
    y²/16 + x²/4 = 1 -/
theorem ellipse_equation (x y : ℝ) :
  let C1 := {(x, y) | x^2/4 + y^2 = 1}
  let e1 := Real.sqrt (1 - (2^2)/(4^2))  -- eccentricity of C1
  let C2 := {(x, y) | ∃ (a : ℝ), a > 2 ∧ y^2/a^2 + x^2/4 = 1 ∧ Real.sqrt (1 - (2^2)/(a^2)) = e1}
  ∀ (x y : ℝ), (x, y) ∈ C2 ↔ y^2/16 + x^2/4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2126_212611


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_l2126_212614

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Predicate for a parabola having a vertical axis of symmetry -/
def has_vertical_axis_of_symmetry (p : Parabola) : Prop :=
  ∃ h : ℝ, ∀ x y : ℝ, p.y_coord (h + x) = p.y_coord (h - x)

theorem parabola_coefficient_sum (p : Parabola) :
  p.y_coord (-3) = 4 →  -- vertex condition
  has_vertical_axis_of_symmetry p →  -- vertical axis of symmetry
  p.y_coord (-1) = 16 →  -- point condition
  p.a + p.b + p.c = 52 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_sum_l2126_212614


namespace NUMINAMATH_CALUDE_fraction_of_8000_l2126_212663

theorem fraction_of_8000 (x : ℝ) : x = 0.1 →
  x * 8000 - (1 / 20) * (1 / 100) * 8000 = 796 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_8000_l2126_212663


namespace NUMINAMATH_CALUDE_chocolate_milk_students_l2126_212685

theorem chocolate_milk_students (strawberry_milk : ℕ) (regular_milk : ℕ) (total_milk : ℕ) :
  strawberry_milk = 15 →
  regular_milk = 3 →
  total_milk = 20 →
  total_milk - (strawberry_milk + regular_milk) = 2 := by
sorry

end NUMINAMATH_CALUDE_chocolate_milk_students_l2126_212685


namespace NUMINAMATH_CALUDE_min_coeff_x2_and_coeff_x7_l2126_212617

def f (m n : ℕ) (x : ℝ) : ℝ := (1 + x)^m + (1 + x)^n

theorem min_coeff_x2_and_coeff_x7 (m n : ℕ) :
  (∃ k : ℕ, k = m + n ∧ k = 19) →
  (∃ min_coeff_x2 : ℕ, 
    min_coeff_x2 = Nat.min (m * (m - 1) / 2 + n * (n - 1) / 2) 
                           ((m + 1) * m / 2 + (n - 1) * (n - 2) / 2) ∧
    min_coeff_x2 = 81) ∧
  (∃ coeff_x7 : ℕ, 
    (m = 10 ∧ n = 9 ∨ m = 9 ∧ n = 10) →
    coeff_x7 = Nat.choose 10 7 + Nat.choose 9 7 ∧
    coeff_x7 = 156) :=
by sorry

end NUMINAMATH_CALUDE_min_coeff_x2_and_coeff_x7_l2126_212617


namespace NUMINAMATH_CALUDE_motorcyclist_travel_l2126_212678

theorem motorcyclist_travel (total_distance : ℕ) (first_two_days : ℕ) (second_day_extra : ℕ)
  (h1 : total_distance = 980)
  (h2 : first_two_days = 725)
  (h3 : second_day_extra = 123) :
  ∃ (day1 day2 day3 : ℕ),
    day1 + day2 + day3 = total_distance ∧
    day1 + day2 = first_two_days ∧
    day2 = day3 + second_day_extra ∧
    day1 = 347 ∧
    day2 = 378 ∧
    day3 = 255 := by
  sorry

end NUMINAMATH_CALUDE_motorcyclist_travel_l2126_212678


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2126_212625

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x + 3 < 3*x - 4 → x ≥ 4 ∧ 4 + 3 < 3*4 - 4 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2126_212625


namespace NUMINAMATH_CALUDE_value_of_a_l2126_212620

theorem value_of_a (a c : ℝ) (h1 : c / a = 4) (h2 : a + c = 30) : a = 6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2126_212620


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l2126_212644

-- Define the quadratic function
def f (a c : ℝ) (x : ℝ) := a * x^2 + 2 * x + c

-- Define the solution set
def solution_set (a c : ℝ) := {x : ℝ | x < -1 ∨ x > 2}

-- State the theorem
theorem quadratic_inequality_properties
  (a c : ℝ)
  (h : ∀ x, f a c x < 0 ↔ x ∈ solution_set a c) :
  (a + c = 2) ∧
  (c^(1/a) = 1/2) ∧
  (∃! y, y ∈ {x : ℝ | x^2 - 2*a*x + c = 0}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l2126_212644


namespace NUMINAMATH_CALUDE_mistaken_division_correct_multiplication_l2126_212652

theorem mistaken_division_correct_multiplication : 
  ∀ n : ℕ, 
  (n / 96 = 5) → 
  (n % 96 = 17) → 
  (n * 69 = 34293) := by
sorry

end NUMINAMATH_CALUDE_mistaken_division_correct_multiplication_l2126_212652


namespace NUMINAMATH_CALUDE_binary_decimal_octal_conversion_l2126_212674

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_octal (n : Nat) : List Nat :=
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_decimal_octal_conversion :
  binary_to_decimal binary_101101 = 45 ∧
  decimal_to_octal 45 = [5, 5] := by
  sorry

end NUMINAMATH_CALUDE_binary_decimal_octal_conversion_l2126_212674


namespace NUMINAMATH_CALUDE_overall_average_score_l2126_212651

theorem overall_average_score 
  (morning_avg : ℝ) 
  (evening_avg : ℝ) 
  (student_ratio : ℚ) 
  (h_morning_avg : morning_avg = 82) 
  (h_evening_avg : evening_avg = 75) 
  (h_student_ratio : student_ratio = 5 / 3) :
  let m := (student_ratio * evening_students : ℝ)
  let e := evening_students
  let total_students := m + e
  let total_score := morning_avg * m + evening_avg * e
  total_score / total_students = 79.375 :=
by
  sorry

#check overall_average_score

end NUMINAMATH_CALUDE_overall_average_score_l2126_212651


namespace NUMINAMATH_CALUDE_rowing_speed_problem_l2126_212662

/-- Represents the rowing speed problem -/
theorem rowing_speed_problem (v c : ℝ) : 
  c = 1.4 → 
  (v + c) = 2 * (v - c) → 
  v = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_rowing_speed_problem_l2126_212662


namespace NUMINAMATH_CALUDE_sarahs_waist_cm_l2126_212600

-- Define the conversion factor from inches to centimeters
def inches_to_cm : ℝ := 2.54

-- Define Sarah's waist size in inches
def sarahs_waist_inches : ℝ := 27

-- Theorem to prove Sarah's waist size in centimeters
theorem sarahs_waist_cm : 
  ∃ (waist_cm : ℝ), abs (waist_cm - (sarahs_waist_inches * inches_to_cm)) < 0.05 :=
by
  sorry

end NUMINAMATH_CALUDE_sarahs_waist_cm_l2126_212600


namespace NUMINAMATH_CALUDE_direct_proportion_m_value_l2126_212677

/-- A function f: ℝ → ℝ is a direct proportion if there exists a constant k such that f(x) = k * x for all x ∈ ℝ -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The given function -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ -7 * x + 2 + m

theorem direct_proportion_m_value :
  (∃ m : ℝ, is_direct_proportion (f m)) → (∃ m : ℝ, m = -2 ∧ is_direct_proportion (f m)) :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_m_value_l2126_212677


namespace NUMINAMATH_CALUDE_circle_equation_l2126_212692

/-- The ellipse with equation x²/16 + y²/4 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 16) + (p.2^2 / 4) = 1}

/-- The vertices of the ellipse -/
def EllipseVertices : Set (ℝ × ℝ) :=
  {p | p ∈ Ellipse ∧ (p.1 = 0 ∨ p.2 = 0)}

/-- The circle C passing through (6,0) and the vertices of the ellipse -/
def CircleC : Set (ℝ × ℝ) :=
  {p | ∃ (c : ℝ), (c, 0) ∈ Ellipse ∧ 
    ((p.1 - c)^2 + p.2^2 = (6 - c)^2) ∧
    (∀ v ∈ EllipseVertices, (p.1 - c)^2 + p.2^2 = (v.1 - c)^2 + v.2^2)}

theorem circle_equation : 
  CircleC = {p | (p.1 - 8/3)^2 + p.2^2 = 100/9} := by
  sorry


end NUMINAMATH_CALUDE_circle_equation_l2126_212692


namespace NUMINAMATH_CALUDE_katie_juice_problem_l2126_212650

theorem katie_juice_problem (initial_juice : ℚ) (given_away : ℚ) (remaining : ℚ) :
  initial_juice = 5 →
  given_away = 18 / 7 →
  remaining = initial_juice - given_away →
  remaining = 17 / 7 := by
sorry

end NUMINAMATH_CALUDE_katie_juice_problem_l2126_212650


namespace NUMINAMATH_CALUDE_cone_base_circumference_l2126_212637

/-- The circumference of the base of a right circular cone formed from a sector of a circle --/
theorem cone_base_circumference (r : ℝ) (θ : ℝ) (h_r : r = 6) (h_θ : θ = 240) :
  let original_circumference := 2 * π * r
  let sector_proportion := θ / 360
  let base_circumference := sector_proportion * original_circumference
  base_circumference = 8 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l2126_212637


namespace NUMINAMATH_CALUDE_car_speed_problem_l2126_212636

-- Define the parameters of the problem
def initial_distance : ℝ := 10
def final_distance : ℝ := 8
def time : ℝ := 2.25
def speed_A : ℝ := 58

-- Define the speed of Car B as a variable
def speed_B : ℝ := 50

-- Theorem statement
theorem car_speed_problem :
  initial_distance + 
  speed_A * time = 
  speed_B * time + 
  initial_distance + 
  final_distance := by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2126_212636


namespace NUMINAMATH_CALUDE_hearty_blue_packages_l2126_212630

/-- The number of packages of red beads -/
def red_packages : ℕ := 5

/-- The number of beads in each package -/
def beads_per_package : ℕ := 40

/-- The total number of beads Hearty has -/
def total_beads : ℕ := 320

/-- The number of packages of blue beads Hearty bought -/
def blue_packages : ℕ := (total_beads - red_packages * beads_per_package) / beads_per_package

theorem hearty_blue_packages : blue_packages = 3 := by
  sorry

end NUMINAMATH_CALUDE_hearty_blue_packages_l2126_212630


namespace NUMINAMATH_CALUDE_no_positive_integers_satisfy_divisibility_l2126_212646

theorem no_positive_integers_satisfy_divisibility : ¬ ∃ (a b c : ℕ+), (3 * (a * b + b * c + c * a)) ∣ (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integers_satisfy_divisibility_l2126_212646


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2126_212698

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  1 / (a + 1) + 1 / (b + 3) ≥ 28 / 49 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2126_212698


namespace NUMINAMATH_CALUDE_jacks_lifetime_l2126_212670

theorem jacks_lifetime (L : ℝ) : 
  L > 0 → 
  (1/6 : ℝ) * L + (1/12 : ℝ) * L + (1/7 : ℝ) * L + 5 + (1/2 : ℝ) * L + 4 = L → 
  L = 84 := by
sorry

end NUMINAMATH_CALUDE_jacks_lifetime_l2126_212670


namespace NUMINAMATH_CALUDE_relationship_exists_l2126_212665

/-- Represents the contingency table --/
structure ContingencyTable where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  n : ℕ

/-- Calculates K^2 value --/
def calculate_k_squared (ct : ContingencyTable) : ℚ :=
  (ct.n * (ct.a * ct.d - ct.b * ct.c)^2 : ℚ) / 
  ((ct.a + ct.b) * (ct.c + ct.d) * (ct.a + ct.c) * (ct.b + ct.d) : ℚ)

/-- Theorem stating the conditions and the result --/
theorem relationship_exists (a : ℕ) : 
  5 < a ∧ 
  a < 10 ∧
  let ct := ContingencyTable.mk a (15 - a) (20 - a) (30 + a) 65
  calculate_k_squared ct ≥ (6635 : ℚ) / 1000 →
  a = 9 := by
  sorry

#check relationship_exists

end NUMINAMATH_CALUDE_relationship_exists_l2126_212665


namespace NUMINAMATH_CALUDE_theta_range_l2126_212668

theorem theta_range (θ : Real) : 
  (∀ x ∈ Set.Icc 0 1, x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) →
  π / 12 < θ ∧ θ < 5 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_theta_range_l2126_212668


namespace NUMINAMATH_CALUDE_decimal_equals_scientific_l2126_212632

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_abs_coeff : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The given number in decimal form -/
def decimal_number : ℝ := -0.0000406

/-- The scientific notation representation of the number -/
def scientific_form : ScientificNotation := {
  coefficient := -4.06,
  exponent := -5,
  one_le_abs_coeff := by sorry
}

/-- Theorem stating that the decimal number is equal to its scientific notation representation -/
theorem decimal_equals_scientific : 
  decimal_number = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by sorry

end NUMINAMATH_CALUDE_decimal_equals_scientific_l2126_212632


namespace NUMINAMATH_CALUDE_equality_check_l2126_212672

theorem equality_check : 
  (3^2 ≠ 2^3) ∧ 
  ((-2)^3 = -2^3) ∧ 
  (-3^2 ≠ (-3)^2) ∧ 
  (-(-2) ≠ -|-2|) :=
by sorry

end NUMINAMATH_CALUDE_equality_check_l2126_212672


namespace NUMINAMATH_CALUDE_scenario_one_count_scenario_two_count_l2126_212683

/-- Represents the number of products --/
def total_products : ℕ := 10

/-- Represents the number of defective products --/
def defective_products : ℕ := 4

/-- Calculates the number of testing methods for scenario 1 --/
def scenario_one_methods : ℕ := sorry

/-- Calculates the number of testing methods for scenario 2 --/
def scenario_two_methods : ℕ := sorry

/-- Theorem for scenario 1 --/
theorem scenario_one_count :
  scenario_one_methods = 103680 :=
sorry

/-- Theorem for scenario 2 --/
theorem scenario_two_count :
  scenario_two_methods = 576 :=
sorry

end NUMINAMATH_CALUDE_scenario_one_count_scenario_two_count_l2126_212683


namespace NUMINAMATH_CALUDE_infinite_solutions_exist_l2126_212602

theorem infinite_solutions_exist :
  ∃ m : ℕ+, ∀ n : ℕ, ∃ a b c : ℕ+,
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / (a * b * c) = m / (a + b + c) :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_exist_l2126_212602


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2126_212655

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₃ + a₈ = 6,
    prove that 3a₂ + a₁₆ = 12 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 3 + a 8 = 6) : 
  3 * a 2 + a 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2126_212655


namespace NUMINAMATH_CALUDE_remaining_travel_distance_l2126_212601

theorem remaining_travel_distance 
  (total_distance : ℕ)
  (amoli_speed : ℕ)
  (amoli_time : ℕ)
  (anayet_speed : ℕ)
  (anayet_time : ℕ)
  (h1 : total_distance = 369)
  (h2 : amoli_speed = 42)
  (h3 : amoli_time = 3)
  (h4 : anayet_speed = 61)
  (h5 : anayet_time = 2) :
  total_distance - (amoli_speed * amoli_time + anayet_speed * anayet_time) = 121 :=
by sorry

end NUMINAMATH_CALUDE_remaining_travel_distance_l2126_212601


namespace NUMINAMATH_CALUDE_derivative_f_l2126_212664

noncomputable def f (x : ℝ) : ℝ := (1/4) * Real.log ((x-1)/(x+1)) - (1/2) * Real.arctan x

theorem derivative_f (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  deriv f x = 1 / (x^4 - 1) := by sorry

end NUMINAMATH_CALUDE_derivative_f_l2126_212664


namespace NUMINAMATH_CALUDE_percentage_boys_playing_soccer_l2126_212605

theorem percentage_boys_playing_soccer 
  (total_students : ℕ) 
  (num_boys : ℕ) 
  (num_playing_soccer : ℕ) 
  (num_girls_not_playing : ℕ) 
  (h1 : total_students = 500) 
  (h2 : num_boys = 350) 
  (h3 : num_playing_soccer = 250) 
  (h4 : num_girls_not_playing = 115) :
  (num_boys - (total_students - num_boys - num_girls_not_playing)) / num_playing_soccer * 100 = 86 := by
sorry

end NUMINAMATH_CALUDE_percentage_boys_playing_soccer_l2126_212605


namespace NUMINAMATH_CALUDE_susie_investment_l2126_212619

/-- Proves that Susie's investment at Safe Savings Bank is 0 --/
theorem susie_investment (total_investment : ℝ) (safe_rate : ℝ) (risky_rate : ℝ) (total_after_year : ℝ) 
  (h1 : total_investment = 2000)
  (h2 : safe_rate = 0.04)
  (h3 : risky_rate = 0.06)
  (h4 : total_after_year = 2120)
  (h5 : ∀ x : ℝ, x * (1 + safe_rate) + (total_investment - x) * (1 + risky_rate) = total_after_year) :
  ∃ x : ℝ, x = 0 ∧ x * (1 + safe_rate) + (total_investment - x) * (1 + risky_rate) = total_after_year :=
sorry

end NUMINAMATH_CALUDE_susie_investment_l2126_212619


namespace NUMINAMATH_CALUDE_circular_garden_area_l2126_212694

theorem circular_garden_area (AB DC : ℝ) (h1 : AB = 20) (h2 : DC = 12) : 
  let AD := AB / 2
  let R := (AD ^ 2 + DC ^ 2).sqrt
  π * R ^ 2 = 244 * π := by sorry

end NUMINAMATH_CALUDE_circular_garden_area_l2126_212694


namespace NUMINAMATH_CALUDE_gcd_182_98_l2126_212635

theorem gcd_182_98 : Nat.gcd 182 98 = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcd_182_98_l2126_212635


namespace NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l2126_212673

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of an ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ := sorry

/-- Theorem: The distance between the foci of the specific ellipse is 6√3 -/
theorem specific_ellipse_foci_distance :
  let e : ParallelAxisEllipse := ⟨(6, 0), (0, 3)⟩
  foci_distance e = 6 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l2126_212673


namespace NUMINAMATH_CALUDE_a_perpendicular_to_a_minus_b_l2126_212648

def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (-1, 3)

theorem a_perpendicular_to_a_minus_b : a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_perpendicular_to_a_minus_b_l2126_212648


namespace NUMINAMATH_CALUDE_tan_135_deg_l2126_212629

/-- Tangent of 135 degrees is -1 -/
theorem tan_135_deg : Real.tan (135 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_135_deg_l2126_212629


namespace NUMINAMATH_CALUDE_isosceles_triangle_with_50_degree_angle_l2126_212608

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define two angles, as the third can be derived
  angle1 : ℝ
  angle2 : ℝ
  -- Condition: The triangle is isosceles (two angles are equal)
  isIsosceles : angle1 = angle2 ∨ angle1 = 180 - angle1 - angle2 ∨ angle2 = 180 - angle1 - angle2
  -- Condition: The sum of angles in a triangle is 180°
  sumIs180 : angle1 + angle2 + (180 - angle1 - angle2) = 180

-- Define our theorem
theorem isosceles_triangle_with_50_degree_angle 
  (triangle : IsoscelesTriangle) 
  (has50DegreeAngle : triangle.angle1 = 50 ∨ triangle.angle2 = 50 ∨ (180 - triangle.angle1 - triangle.angle2) = 50) :
  triangle.angle1 = 50 ∨ triangle.angle1 = 65 ∨ triangle.angle2 = 50 ∨ triangle.angle2 = 65 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_with_50_degree_angle_l2126_212608


namespace NUMINAMATH_CALUDE_linear_function_increasing_l2126_212645

/-- A linear function f(x) = mx + b where m > 0 is increasing -/
theorem linear_function_increasing (m b : ℝ) (h : m > 0) :
  Monotone (fun x => m * x + b) := by sorry

end NUMINAMATH_CALUDE_linear_function_increasing_l2126_212645


namespace NUMINAMATH_CALUDE_expression_value_l2126_212639

theorem expression_value (x y z : ℝ) (hx : x = 1 + Real.sqrt 2) (hy : y = x + 1) (hz : z = x - 1) :
  y^2 * z^4 - 4 * y^3 * z^3 + 6 * y^2 * z^2 + 4 * y = -120 - 92 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2126_212639


namespace NUMINAMATH_CALUDE_cube_preserves_order_l2126_212610

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l2126_212610


namespace NUMINAMATH_CALUDE_largest_prime_to_check_primality_l2126_212624

theorem largest_prime_to_check_primality (n : ℕ) : 
  1000 ≤ n → n ≤ 1100 → 
  (∀ p : ℕ, p.Prime → p ≤ 31 → n % p ≠ 0) → 
  (∀ p : ℕ, p.Prime → p < n → n % p ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_largest_prime_to_check_primality_l2126_212624


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l2126_212680

/-- The function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem extreme_value_implies_a_equals_5 (a : ℝ) :
  (f' a (-3) = 0) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l2126_212680


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2126_212661

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2126_212661


namespace NUMINAMATH_CALUDE_derivative_f_at_negative_one_l2126_212681

def f (x : ℝ) : ℝ := x^6

theorem derivative_f_at_negative_one :
  deriv f (-1) = -6 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_negative_one_l2126_212681


namespace NUMINAMATH_CALUDE_max_stores_visited_l2126_212667

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ) 
  (two_store_visitors : ℕ) (h1 : total_stores = 8) (h2 : total_visits = 23) 
  (h3 : total_shoppers = 12) (h4 : two_store_visitors = 8) 
  (h5 : two_store_visitors ≤ total_shoppers) 
  (h6 : 2 * two_store_visitors ≤ total_visits) :
  ∃ (max_visits : ℕ), max_visits ≤ 4 ∧ 
  ∀ (individual_visits : ℕ), individual_visits ≤ max_visits ∧
  (total_visits = 2 * two_store_visitors + 
    (total_shoppers - two_store_visitors) + 
    (individual_visits - 1)) :=
by sorry

end NUMINAMATH_CALUDE_max_stores_visited_l2126_212667


namespace NUMINAMATH_CALUDE_bankers_discount_l2126_212607

/-- Banker's discount calculation -/
theorem bankers_discount (bankers_gain : ℝ) (rate : ℝ) (time : ℝ) : 
  bankers_gain = 270 → rate = 12 → time = 3 → 
  ∃ (bankers_discount : ℝ), abs (bankers_discount - 421.88) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_bankers_discount_l2126_212607


namespace NUMINAMATH_CALUDE_probability_for_given_box_l2126_212654

/-- Represents the contents of the box -/
structure Box where
  blue : Nat
  red : Nat
  green : Nat

/-- The probability of drawing all blue chips before both green chips -/
def probability_all_blue_before_both_green (box : Box) : Rat :=
  17/36

/-- Theorem stating the probability for the given box configuration -/
theorem probability_for_given_box :
  let box : Box := { blue := 4, red := 3, green := 2 }
  probability_all_blue_before_both_green box = 17/36 := by
  sorry

end NUMINAMATH_CALUDE_probability_for_given_box_l2126_212654


namespace NUMINAMATH_CALUDE_correct_marble_distribution_l2126_212603

/-- Represents the distribution of marbles among three boys -/
structure MarbleDistribution where
  x : ℕ
  first_boy : ℕ := 5 * x + 2
  second_boy : ℕ := 2 * x - 1
  third_boy : ℕ := x + 3

/-- The theorem stating the correct distribution of marbles -/
theorem correct_marble_distribution :
  ∃ (d : MarbleDistribution),
    d.first_boy + d.second_boy + d.third_boy = 60 ∧
    d.first_boy = 37 ∧
    d.second_boy = 13 ∧
    d.third_boy = 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_marble_distribution_l2126_212603


namespace NUMINAMATH_CALUDE_probability_all_colors_l2126_212669

/-- The probability of selecting 4 balls from a bag containing 2 red balls, 3 white balls, and 4 yellow balls, such that the selection includes balls of all three colors, is equal to 4/7. -/
theorem probability_all_colors (red : ℕ) (white : ℕ) (yellow : ℕ) (total_select : ℕ) :
  red = 2 →
  white = 3 →
  yellow = 4 →
  total_select = 4 →
  (Nat.choose (red + white + yellow) total_select : ℚ) ≠ 0 →
  (↑(Nat.choose red 2 * Nat.choose white 1 * Nat.choose yellow 1 +
     Nat.choose red 1 * Nat.choose white 2 * Nat.choose yellow 1 +
     Nat.choose red 1 * Nat.choose white 1 * Nat.choose yellow 2) /
   Nat.choose (red + white + yellow) total_select : ℚ) = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_probability_all_colors_l2126_212669


namespace NUMINAMATH_CALUDE_fred_tim_marbles_comparison_l2126_212613

theorem fred_tim_marbles_comparison :
  let fred_marbles : ℕ := 110
  let tim_marbles : ℕ := 5
  (fred_marbles / tim_marbles : ℚ) = 22 :=
by sorry

end NUMINAMATH_CALUDE_fred_tim_marbles_comparison_l2126_212613


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l2126_212659

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricProgression (a : ℝ) (r : ℝ) := fun (n : ℕ) => a * r ^ (n - 1)

theorem geometric_progression_first_term
  (a : ℝ) (r : ℝ) (h1 : r ≠ 0)
  (h2 : GeometricProgression a r 2 = 5)
  (h3 : GeometricProgression a r 3 = 1) :
  a = 25 := by
  sorry

#check geometric_progression_first_term

end NUMINAMATH_CALUDE_geometric_progression_first_term_l2126_212659


namespace NUMINAMATH_CALUDE_ellipse_intersection_property_l2126_212686

/-- An ellipse with semi-major axis 2 and semi-minor axis √3 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + (p.2^2 / 3) = 1}

/-- Check if two points are symmetric about the x-axis -/
def symmetric_about_x (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

/-- The intersection point of two lines -/
def intersection (p₁ p₂ q₁ q₂ : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: For the given ellipse, if points satisfy the specified conditions,
    then the x-coordinates of A and B multiply to give 4 -/
theorem ellipse_intersection_property
  (d e : ℝ × ℝ)
  (h_d : d ∈ Ellipse)
  (h_e : e ∈ Ellipse)
  (h_sym : symmetric_about_x d e)
  (x₁ x₂ : ℝ)
  (h_not_tangent : ∀ y, (x₁, y) ≠ d)
  (c : ℝ × ℝ)
  (h_c_intersection : c = intersection d (x₁, 0) e (x₂, 0))
  (h_c_on_ellipse : c ∈ Ellipse) :
  x₁ * x₂ = 4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_property_l2126_212686


namespace NUMINAMATH_CALUDE_solution_set_f_gt_2_min_value_f_l2126_212690

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_gt_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -7 ∨ x > 5/3} := by sorry

-- Theorem for the minimum value of f(x)
theorem min_value_f :
  ∃ (x : ℝ), f x = -9/2 ∧ ∀ (y : ℝ), f y ≥ -9/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_2_min_value_f_l2126_212690


namespace NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l2126_212621

theorem tagged_fish_in_second_catch 
  (initial_tagged : ℕ) 
  (second_catch : ℕ) 
  (total_fish : ℕ) 
  (h1 : initial_tagged = 80) 
  (h2 : second_catch = 80) 
  (h3 : total_fish = 3200) :
  ∃ (tagged_in_second : ℕ), 
    tagged_in_second = 2 ∧ 
    (tagged_in_second : ℚ) / second_catch = initial_tagged / total_fish :=
by
  sorry

end NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l2126_212621
