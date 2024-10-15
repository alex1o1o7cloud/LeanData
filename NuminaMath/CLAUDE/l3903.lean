import Mathlib

namespace NUMINAMATH_CALUDE_power_function_continuous_l3903_390311

theorem power_function_continuous (n : ℕ+) :
  Continuous (fun x : ℝ => x ^ (n : ℕ)) :=
sorry

end NUMINAMATH_CALUDE_power_function_continuous_l3903_390311


namespace NUMINAMATH_CALUDE_unique_triple_prime_l3903_390302

theorem unique_triple_prime (p : ℕ) : 
  (p > 0 ∧ Nat.Prime p ∧ Nat.Prime (p + 4) ∧ Nat.Prime (p + 8)) ↔ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_prime_l3903_390302


namespace NUMINAMATH_CALUDE_sum_2x_2y_l3903_390360

theorem sum_2x_2y (x y : ℝ) (h1 : x^2 - y^2 = 8) (h2 : x - y = 6) : 2*x + 2*y = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_2x_2y_l3903_390360


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_24_l3903_390320

theorem largest_four_digit_divisible_by_24 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 24 ∣ n → n ≤ 9984 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_24_l3903_390320


namespace NUMINAMATH_CALUDE_virginia_eggs_problem_l3903_390317

theorem virginia_eggs_problem (initial_eggs : ℕ) (amy_takes : ℕ) (john_takes : ℕ) (laura_takes : ℕ) 
  (h1 : initial_eggs = 372)
  (h2 : amy_takes = 15)
  (h3 : john_takes = 27)
  (h4 : laura_takes = 63) :
  initial_eggs - amy_takes - john_takes - laura_takes = 267 := by
  sorry

end NUMINAMATH_CALUDE_virginia_eggs_problem_l3903_390317


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l3903_390346

/-- Given a polynomial f(x) = ax^7 - bx^3 + cx - 5 where f(2) = 3, prove that f(-2) = -13 -/
theorem polynomial_symmetry (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^7 - b * x^3 + c * x - 5
  (f 2 = 3) → (f (-2) = -13) := by
sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l3903_390346


namespace NUMINAMATH_CALUDE_tom_car_lease_annual_cost_l3903_390372

/-- Calculates the annual cost of Tom's car lease -/
theorem tom_car_lease_annual_cost :
  let miles_mon_wed_fri : ℕ := 50
  let miles_other_days : ℕ := 100
  let days_mon_wed_fri : ℕ := 3
  let days_other : ℕ := 4
  let cost_per_mile : ℚ := 1 / 10
  let weekly_fee : ℕ := 100
  let weeks_per_year : ℕ := 52

  let weekly_miles : ℕ := miles_mon_wed_fri * days_mon_wed_fri + miles_other_days * days_other
  let weekly_mileage_cost : ℚ := (weekly_miles : ℚ) * cost_per_mile
  let total_weekly_cost : ℚ := weekly_mileage_cost + weekly_fee
  let annual_cost : ℚ := total_weekly_cost * weeks_per_year

  annual_cost = 8060 := by
sorry


end NUMINAMATH_CALUDE_tom_car_lease_annual_cost_l3903_390372


namespace NUMINAMATH_CALUDE_sum_of_coordinates_equals_eight_l3903_390328

def point_C : ℝ × ℝ := (3, 4)

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def point_D : ℝ × ℝ := reflect_over_y_axis point_C

theorem sum_of_coordinates_equals_eight :
  point_C.1 + point_C.2 + point_D.1 + point_D.2 = 8 := by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_equals_eight_l3903_390328


namespace NUMINAMATH_CALUDE_ellipse_a_range_l3903_390376

theorem ellipse_a_range (a b : ℝ) (e : ℝ) :
  a > b ∧ b > 0 ∧
  e ∈ Set.Icc (1 / Real.sqrt 3) (1 / Real.sqrt 2) ∧
  (∃ (M N : ℝ × ℝ),
    (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧
    (N.1^2 / a^2 + N.2^2 / b^2 = 1) ∧
    (M.2 = -M.1 + 1) ∧
    (N.2 = -N.1 + 1) ∧
    (M.1 * N.1 + M.2 * N.2 = 0)) →
  Real.sqrt 5 / 2 ≤ a ∧ a ≤ Real.sqrt 6 / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_a_range_l3903_390376


namespace NUMINAMATH_CALUDE_sin_shift_l3903_390331

theorem sin_shift (x : ℝ) : Real.sin (x + π/3) = Real.sin (x + π/3) := by sorry

end NUMINAMATH_CALUDE_sin_shift_l3903_390331


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_problem_solution_l3903_390313

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) (n : ℕ) :
  a * (b * (c * n)) = (a * b * c) * n :=
by sorry

theorem problem_solution : (2 / 5 : ℚ) * ((3 / 4 : ℚ) * ((1 / 6 : ℚ) * 120)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_problem_solution_l3903_390313


namespace NUMINAMATH_CALUDE_cosine_product_square_root_l3903_390303

theorem cosine_product_square_root : 
  Real.sqrt ((2 - Real.cos (π / 9) ^ 2) * (2 - Real.cos (2 * π / 9) ^ 2) * (2 - Real.cos (3 * π / 9) ^ 2)) = Real.sqrt 377 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_square_root_l3903_390303


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3903_390323

theorem modulus_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (3 + i) / (i^2)
  Complex.abs z = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3903_390323


namespace NUMINAMATH_CALUDE_relay_race_arrangements_l3903_390325

/-- The number of students to choose from -/
def total_students : ℕ := 10

/-- The number of legs in the relay race -/
def race_legs : ℕ := 4

/-- Function to calculate permutations -/
def permutations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / Nat.factorial (n - k)

/-- The main theorem -/
theorem relay_race_arrangements :
  permutations total_students race_legs
  - permutations (total_students - 1) (race_legs - 1)  -- A not in first leg
  - permutations (total_students - 1) (race_legs - 1)  -- B not in last leg
  + permutations (total_students - 2) (race_legs - 2)  -- Neither A in first nor B in last
  = 4008 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l3903_390325


namespace NUMINAMATH_CALUDE_orange_preference_percentage_l3903_390399

def survey_results : List (String × Nat) :=
  [("Red", 70), ("Orange", 50), ("Green", 60), ("Yellow", 80), ("Blue", 40), ("Purple", 50)]

def total_responses : Nat :=
  (survey_results.map (λ (_, count) => count)).sum

def orange_preference : Nat :=
  match survey_results.find? (λ (color, _) => color = "Orange") with
  | some (_, count) => count
  | none => 0

theorem orange_preference_percentage :
  (orange_preference : ℚ) / (total_responses : ℚ) * 100 = 14 := by sorry

end NUMINAMATH_CALUDE_orange_preference_percentage_l3903_390399


namespace NUMINAMATH_CALUDE_meatballs_cost_is_five_l3903_390355

/-- A dinner consisting of pasta, sauce, and meatballs -/
structure Dinner where
  total_cost : ℝ
  pasta_cost : ℝ
  sauce_cost : ℝ
  meatballs_cost : ℝ

/-- The cost of the dinner components add up to the total cost -/
def cost_sum (d : Dinner) : Prop :=
  d.total_cost = d.pasta_cost + d.sauce_cost + d.meatballs_cost

/-- Theorem: Given the total cost, pasta cost, and sauce cost, 
    prove that the meatballs cost $5 -/
theorem meatballs_cost_is_five (d : Dinner) 
  (h1 : d.total_cost = 8)
  (h2 : d.pasta_cost = 1)
  (h3 : d.sauce_cost = 2)
  (h4 : cost_sum d) : 
  d.meatballs_cost = 5 := by
  sorry


end NUMINAMATH_CALUDE_meatballs_cost_is_five_l3903_390355


namespace NUMINAMATH_CALUDE_product_digit_sum_l3903_390390

def repeat_digits (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^(3*n) - 1) / 999

def number1 : ℕ := repeat_digits 400 333
def number2 : ℕ := repeat_digits 606 333

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem product_digit_sum :
  tens_digit (number1 * number2) + units_digit (number1 * number2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l3903_390390


namespace NUMINAMATH_CALUDE_number_categorization_l3903_390304

def given_numbers : List ℚ := [8, -1, -2/5, 3/5, 0, 1/3, -10/7, 5, -20/7]

def is_positive (x : ℚ) : Prop := x > 0
def is_negative (x : ℚ) : Prop := x < 0
def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n
def is_fraction (x : ℚ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b
def is_non_negative_rational (x : ℚ) : Prop := x ≥ 0

def positive_set : Set ℚ := {x | is_positive x}
def negative_set : Set ℚ := {x | is_negative x}
def integer_set : Set ℚ := {x | is_integer x}
def fraction_set : Set ℚ := {x | is_fraction x}
def non_negative_rational_set : Set ℚ := {x | is_non_negative_rational x}

theorem number_categorization :
  positive_set = {8, 3/5, 1/3, 5} ∧
  negative_set = {-1, -2/5, -10/7, -20/7} ∧
  integer_set = {8, -1, 0, 5} ∧
  fraction_set = {-2/5, 3/5, 1/3, -10/7, -20/7} ∧
  non_negative_rational_set = {8, 3/5, 0, 1/3, 5} := by
  sorry

end NUMINAMATH_CALUDE_number_categorization_l3903_390304


namespace NUMINAMATH_CALUDE_work_completion_time_l3903_390352

/-- 
Given a group of ladies that can complete a piece of work in 12 days,
prove that a group with twice as many ladies will complete half of the work in 3 days.
-/
theorem work_completion_time (num_ladies : ℕ) (total_work : ℝ) : 
  (num_ladies * 12 : ℝ) * total_work = 12 * total_work →
  ((2 * num_ladies : ℝ) * 3) * (total_work / 2) = 12 * (total_work / 2) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3903_390352


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l3903_390364

theorem quadratic_roots_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (hroots : a^2 - c * a^2 + c = 0 ∧ b^2 - c * b^2 + c = 0) :
  (a * Real.sqrt (1 - 1 / b^2) + b * Real.sqrt (1 - 1 / a^2) = 2) ∨
  (a * Real.sqrt (1 - 1 / b^2) + b * Real.sqrt (1 - 1 / a^2) = -2) ∨
  (a * Real.sqrt (1 - 1 / b^2) + b * Real.sqrt (1 - 1 / a^2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l3903_390364


namespace NUMINAMATH_CALUDE_product_property_l3903_390336

theorem product_property : ∃ (n : ℕ), 10 ≤ n ∧ n ≤ 99 ∧ (∃ (k : ℤ), 4.02 * (n : ℝ) = (k : ℝ)) ∧ 10 * (4.02 * (n : ℝ)) = 2010 := by
  sorry

end NUMINAMATH_CALUDE_product_property_l3903_390336


namespace NUMINAMATH_CALUDE_parts_cost_is_800_l3903_390340

/-- Represents the business model of John's computer assembly and sales --/
structure ComputerBusiness where
  partsCost : ℝ  -- Cost of parts for each computer
  sellMultiplier : ℝ  -- Multiplier for selling price
  monthlyProduction : ℕ  -- Number of computers produced per month
  monthlyRent : ℝ  -- Monthly rent cost
  monthlyExtraExpenses : ℝ  -- Monthly non-rent extra expenses
  monthlyProfit : ℝ  -- Monthly profit

/-- Calculates the monthly revenue --/
def monthlyRevenue (b : ComputerBusiness) : ℝ :=
  b.monthlyProduction * (b.sellMultiplier * b.partsCost)

/-- Calculates the monthly expenses --/
def monthlyExpenses (b : ComputerBusiness) : ℝ :=
  b.monthlyProduction * b.partsCost + b.monthlyRent + b.monthlyExtraExpenses

/-- Theorem stating that the cost of parts for each computer is $800 --/
theorem parts_cost_is_800 (b : ComputerBusiness)
    (h1 : b.sellMultiplier = 1.4)
    (h2 : b.monthlyProduction = 60)
    (h3 : b.monthlyRent = 5000)
    (h4 : b.monthlyExtraExpenses = 3000)
    (h5 : b.monthlyProfit = 11200)
    (h6 : monthlyRevenue b - monthlyExpenses b = b.monthlyProfit) :
    b.partsCost = 800 := by
  sorry

end NUMINAMATH_CALUDE_parts_cost_is_800_l3903_390340


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l3903_390305

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_equation_solution :
  ∀ N : ℕ, N > 0 → (factorial 5 * factorial 9 = 12 * factorial N) → N = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l3903_390305


namespace NUMINAMATH_CALUDE_birthday_candles_sharing_l3903_390333

theorem birthday_candles_sharing (ambika_candles : ℕ) (aniyah_multiplier : ℕ) : 
  ambika_candles = 4 →
  aniyah_multiplier = 6 →
  ((ambika_candles + aniyah_multiplier * ambika_candles) / 2 : ℕ) = 14 :=
by sorry

end NUMINAMATH_CALUDE_birthday_candles_sharing_l3903_390333


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3903_390321

def M : Set ℝ := {x : ℝ | x^2 - 3*x = 0}
def N : Set ℝ := {x : ℝ | x^2 - 5*x + 6 = 0}

theorem union_of_M_and_N : M ∪ N = {0, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3903_390321


namespace NUMINAMATH_CALUDE_expression_unbounded_l3903_390315

theorem expression_unbounded (M : ℝ) (hM : M > 0) :
  ∃ x y z : ℝ, -1 < x ∧ x < 1 ∧ -1 < y ∧ y < 1 ∧ -1 < z ∧ z < 1 ∧
    (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)) +
     1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))) > M :=
by sorry

end NUMINAMATH_CALUDE_expression_unbounded_l3903_390315


namespace NUMINAMATH_CALUDE_hexagon_side_length_l3903_390309

/-- A regular hexagon with perimeter 60 inches has sides of length 10 inches. -/
theorem hexagon_side_length : ∀ (side_length : ℝ), 
  side_length > 0 →
  6 * side_length = 60 →
  side_length = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l3903_390309


namespace NUMINAMATH_CALUDE_semicircle_segment_sum_l3903_390316

-- Define the semicircle and its properties
structure Semicircle where
  r : ℝ
  a : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  h_diameter : dist A B = 2 * r
  h_AT : a > 0 ∧ 2 * a < r / 2
  h_M_on_semicircle : dist M A * dist M B = r ^ 2
  h_N_on_semicircle : dist N A * dist N B = r ^ 2
  h_M_condition : dist M (0, -2 * a) / dist M A = 1
  h_N_condition : dist N (0, -2 * a) / dist N A = 1
  h_M_N_distinct : M ≠ N

-- State the theorem
theorem semicircle_segment_sum (s : Semicircle) : dist s.A s.M + dist s.A s.N = dist s.A s.B := by
  sorry

end NUMINAMATH_CALUDE_semicircle_segment_sum_l3903_390316


namespace NUMINAMATH_CALUDE_number_puzzle_l3903_390334

theorem number_puzzle (x : ℤ) (h : x - 46 = 15) : x - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3903_390334


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3903_390326

theorem quadratic_root_value (m : ℝ) : 
  (1 : ℝ)^2 + (1 : ℝ) - m = 0 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3903_390326


namespace NUMINAMATH_CALUDE_fourth_person_height_l3903_390375

/-- Represents a person with height, weight, and age -/
structure Person where
  height : ℝ
  weight : ℝ
  age : ℕ

/-- Given conditions for the problem -/
def fourPeople (p1 p2 p3 p4 : Person) : Prop :=
  p1.height < p2.height ∧ p2.height < p3.height ∧ p3.height < p4.height ∧
  p2.height - p1.height = 2 ∧
  p3.height - p2.height = 3 ∧
  p4.height - p3.height = 6 ∧
  p1.weight + p2.weight + p3.weight + p4.weight = 600 ∧
  p1.age = 25 ∧ p2.age = 32 ∧ p3.age = 37 ∧ p4.age = 46 ∧
  (p1.height + p2.height + p3.height + p4.height) / 4 = 72 ∧
  ∀ (i j : Fin 4), (i.val < j.val) → 
    (p1.height * p1.age = p2.height * p2.age) ∧
    (p1.height * p2.weight = p2.height * p1.weight)

/-- Theorem: The fourth person's height is 78.5 inches -/
theorem fourth_person_height (p1 p2 p3 p4 : Person) 
  (h : fourPeople p1 p2 p3 p4) : p4.height = 78.5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l3903_390375


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l3903_390393

theorem triangle_cosine_theorem (A B C : ℝ) (h1 : A + C = 2 * B) 
  (h2 : 1 / Real.cos A + 1 / Real.cos C = -Real.sqrt 2 / Real.cos B) :
  Real.cos ((A - C) / 2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l3903_390393


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l3903_390322

theorem sin_cos_pi_12 : 2 * Real.sin (π / 12) * Real.cos (π / 12) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l3903_390322


namespace NUMINAMATH_CALUDE_number_ratio_problem_l3903_390353

/-- Given three numbers satisfying specific conditions, prove their ratios -/
theorem number_ratio_problem (a b c : ℚ) : 
  a + b + c = 98 → 
  b = 30 → 
  c = (8/5) * b → 
  a / b = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_problem_l3903_390353


namespace NUMINAMATH_CALUDE_number_is_composite_l3903_390386

theorem number_is_composite : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 10^1962 + 1 = a * b := by
  sorry

end NUMINAMATH_CALUDE_number_is_composite_l3903_390386


namespace NUMINAMATH_CALUDE_exists_tangent_circle_l3903_390312

-- Define the basic geometric objects
structure Point :=
  (x y : ℝ)

structure Line :=
  (a b c : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define the given objects
variable (M : Point)
variable (l : Line)
variable (S : Circle)

-- Define the tangency and passing through relations
def isTangentToLine (c : Circle) (l : Line) : Prop := sorry
def isTangentToCircle (c1 c2 : Circle) : Prop := sorry
def passesThrough (c : Circle) (p : Point) : Prop := sorry

-- Theorem statement
theorem exists_tangent_circle :
  ∃ (Ω : Circle),
    passesThrough Ω M ∧
    isTangentToLine Ω l ∧
    isTangentToCircle Ω S :=
sorry

end NUMINAMATH_CALUDE_exists_tangent_circle_l3903_390312


namespace NUMINAMATH_CALUDE_parallel_transitivity_l3903_390369

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define parallelism between two lines
def parallel (l1 l2 : Line) : Prop :=
  ∀ (x y : ℝ), l1 x y ↔ l2 x y

-- State the theorem
theorem parallel_transitivity (l1 l2 l3 : Line) :
  parallel l1 l3 → parallel l2 l3 → parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l3903_390369


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3903_390318

theorem geometric_sequence_problem (b : ℝ) (h1 : b > 0) :
  (∃ r : ℝ, 30 * r = b ∧ b * r = 3/8) → b = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3903_390318


namespace NUMINAMATH_CALUDE_factorization_sum_l3903_390356

theorem factorization_sum (a b : ℤ) : 
  (∀ x, 25 * x^2 - 160 * x - 144 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = -68 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l3903_390356


namespace NUMINAMATH_CALUDE_smallest_N_is_110_l3903_390398

/-- Represents a point in the rectangular array -/
structure Point where
  row : Fin 6
  col : ℕ

/-- The x-coordinate of a point after initial numbering -/
def x (p : Point) (N : ℕ) : ℕ := p.row.val * N + p.col

/-- The y-coordinate of a point after renumbering -/
def y (p : Point) : ℕ := (p.col - 1) * 6 + p.row.val + 1

/-- Predicate that checks if the given conditions are satisfied -/
def satisfiesConditions (N : ℕ) (p₁ p₂ p₃ p₄ p₅ p₆ : Point) : Prop :=
  x p₁ N = y p₂ ∧
  x p₂ N = y p₁ ∧
  x p₃ N = y p₄ ∧
  x p₄ N = y p₅ ∧
  x p₅ N = y p₆ ∧
  x p₆ N = y p₃

/-- The main theorem stating that 110 is the smallest N satisfying the conditions -/
theorem smallest_N_is_110 :
  ∃ (p₁ p₂ p₃ p₄ p₅ p₆ : Point),
    satisfiesConditions 110 p₁ p₂ p₃ p₄ p₅ p₆ ∧
    ∀ (N : ℕ), N < 110 → ¬∃ (q₁ q₂ q₃ q₄ q₅ q₆ : Point),
      satisfiesConditions N q₁ q₂ q₃ q₄ q₅ q₆ :=
by sorry

end NUMINAMATH_CALUDE_smallest_N_is_110_l3903_390398


namespace NUMINAMATH_CALUDE_digit_sum_property_l3903_390392

/-- A function that returns true if a number has no zero digits -/
def has_no_zero_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ≠ 0

/-- A function that returns all digit permutations of a number -/
def digit_permutations (n : ℕ) : Finset ℕ :=
  sorry

/-- A function that checks if a number is composed entirely of ones -/
def all_ones (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1

/-- A function that checks if a number has a digit 5 or greater -/
def has_digit_ge_5 (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d ≥ 5

theorem digit_sum_property (n : ℕ) (h1 : has_no_zero_digits n) 
  (h2 : all_ones (n + (Finset.sum (digit_permutations n) id))) :
  has_digit_ge_5 n :=
sorry

end NUMINAMATH_CALUDE_digit_sum_property_l3903_390392


namespace NUMINAMATH_CALUDE_units_digit_of_factorial_sum_plus_three_l3903_390396

-- Define a function to calculate factorial
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the sum of factorials from 1 to 10
def factorialSum : ℕ := 
  List.sum (List.map factorial (List.range 10))

-- Theorem to prove
theorem units_digit_of_factorial_sum_plus_three : 
  unitsDigit (factorialSum + 3) = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_factorial_sum_plus_three_l3903_390396


namespace NUMINAMATH_CALUDE_ivan_tsarevich_revival_l3903_390379

/-- Represents the scenario of Wolf, Ivan Tsarevich, and the Raven --/
structure RevivalScenario where
  initialDistance : ℝ
  wolfSpeed : ℝ
  waterNeeded : ℝ
  springFlowRate : ℝ
  ravenSpeed : ℝ
  ravenWaterLossRate : ℝ

/-- Determines if Ivan Tsarevich can be revived after the given time --/
def canRevive (scenario : RevivalScenario) (time : ℝ) : Prop :=
  let waterCollectionTime := scenario.waterNeeded / scenario.springFlowRate
  let wolfDistance := scenario.wolfSpeed * waterCollectionTime
  let remainingDistance := scenario.initialDistance - wolfDistance
  let meetingTime := remainingDistance / (scenario.ravenSpeed + scenario.wolfSpeed)
  let totalTime := waterCollectionTime + meetingTime
  let waterLost := scenario.ravenWaterLossRate * meetingTime
  totalTime ≤ time ∧ scenario.waterNeeded - waterLost > 0

/-- The main theorem stating that Ivan Tsarevich can be revived after 4 hours --/
theorem ivan_tsarevich_revival (scenario : RevivalScenario)
  (h1 : scenario.initialDistance = 20)
  (h2 : scenario.wolfSpeed = 3)
  (h3 : scenario.waterNeeded = 1)
  (h4 : scenario.springFlowRate = 0.5)
  (h5 : scenario.ravenSpeed = 6)
  (h6 : scenario.ravenWaterLossRate = 0.25) :
  canRevive scenario 4 := by
  sorry

end NUMINAMATH_CALUDE_ivan_tsarevich_revival_l3903_390379


namespace NUMINAMATH_CALUDE_bob_investment_l3903_390344

theorem bob_investment (interest_rate_1 interest_rate_2 total_interest investment_1 : ℝ)
  (h1 : interest_rate_1 = 0.18)
  (h2 : interest_rate_2 = 0.14)
  (h3 : total_interest = 3360)
  (h4 : investment_1 = 7000)
  (h5 : investment_1 * interest_rate_1 + (total_investment - investment_1) * interest_rate_2 = total_interest) :
  ∃ (total_investment : ℝ), total_investment = 22000 := by
sorry

end NUMINAMATH_CALUDE_bob_investment_l3903_390344


namespace NUMINAMATH_CALUDE_tiffany_found_two_bags_l3903_390335

/-- The number of bags Tiffany found on the next day -/
def bags_found_next_day (bags_monday : ℕ) (total_bags : ℕ) : ℕ :=
  total_bags - bags_monday

/-- Theorem: Tiffany found 2 bags on the next day -/
theorem tiffany_found_two_bags :
  let bags_monday := 4
  let total_bags := 6
  bags_found_next_day bags_monday total_bags = 2 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_found_two_bags_l3903_390335


namespace NUMINAMATH_CALUDE_square_perimeter_greater_than_circle_circumference_l3903_390373

theorem square_perimeter_greater_than_circle_circumference :
  ∀ (a r : ℝ), a > 0 → r > 0 →
  a^2 = π * r^2 →
  4 * a > 2 * π * r :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_greater_than_circle_circumference_l3903_390373


namespace NUMINAMATH_CALUDE_triangle_base_length_l3903_390300

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 3 → height = 3 → area = (base * height) / 2 → base = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_base_length_l3903_390300


namespace NUMINAMATH_CALUDE_total_routes_to_school_l3903_390384

theorem total_routes_to_school (bus_routes subway_routes : ℕ) 
  (h1 : bus_routes = 3) 
  (h2 : subway_routes = 2) : 
  bus_routes + subway_routes = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_routes_to_school_l3903_390384


namespace NUMINAMATH_CALUDE_problem_solution_l3903_390394

theorem problem_solution (a b : ℝ) (h1 : a + b = 4) (h2 : a * b = 1) : 
  (a - b)^2 = 12 ∧ a^5*b - 2*a^4*b^4 + a*b^5 = 192 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3903_390394


namespace NUMINAMATH_CALUDE_smallest_three_digit_number_l3903_390330

def digits : Finset Nat := {3, 0, 2, 5, 7}

def isValidNumber (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ 
  (Finset.card (Finset.filter (λ d => d ∈ digits) (Finset.image (λ i => (n / (10^i)) % 10) {0, 1, 2})) = 3)

def smallestValidNumber : Nat := 203

theorem smallest_three_digit_number :
  (isValidNumber smallestValidNumber) ∧
  (∀ n : Nat, isValidNumber n → n ≥ smallestValidNumber) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_number_l3903_390330


namespace NUMINAMATH_CALUDE_hill_depth_ratio_l3903_390383

/-- Given a hill with its base 300m above the seabed and a total height of 900m,
    prove that the ratio of the depth from the base to the seabed
    to the total height of the hill is 1/3. -/
theorem hill_depth_ratio (base_height : ℝ) (total_height : ℝ) :
  base_height = 300 →
  total_height = 900 →
  base_height / total_height = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hill_depth_ratio_l3903_390383


namespace NUMINAMATH_CALUDE_sand_cone_weight_l3903_390339

/-- The weight of a sand cone given its dimensions and sand density -/
theorem sand_cone_weight (diameter : ℝ) (height_ratio : ℝ) (sand_density : ℝ) :
  diameter = 12 →
  height_ratio = 0.8 →
  sand_density = 100 →
  let radius := diameter / 2
  let height := height_ratio * diameter
  let volume := (1/3) * π * radius^2 * height
  volume * sand_density = 11520 * π :=
by sorry

end NUMINAMATH_CALUDE_sand_cone_weight_l3903_390339


namespace NUMINAMATH_CALUDE_triangle_ratio_l3903_390307

/-- Triangle PQR with angle bisector PS intersecting MN at X -/
structure Triangle (P Q R S M N X : ℝ × ℝ) : Prop where
  m_on_pq : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • P + t • Q
  n_on_pr : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ N = (1 - t) • P + t • R
  ps_bisector : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (1 - t) • P + t • ((2/3) • Q + (1/3) • R)
  x_on_mn : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (1 - t) • M + t • N
  x_on_ps : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (1 - t) • P + t • S

/-- Given lengths in the triangle -/
structure TriangleLengths (P Q R S M N X : ℝ × ℝ) : Prop where
  pm_eq : ‖M - P‖ = 2
  mq_eq : ‖Q - M‖ = 6
  pn_eq : ‖N - P‖ = 3
  nr_eq : ‖R - N‖ = 9

/-- The main theorem -/
theorem triangle_ratio 
  (P Q R S M N X : ℝ × ℝ) 
  (h1 : Triangle P Q R S M N X) 
  (h2 : TriangleLengths P Q R S M N X) : 
  ‖X - P‖ / ‖S - P‖ = 1/4 :=
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3903_390307


namespace NUMINAMATH_CALUDE_first_number_is_45_l3903_390348

/-- Given two positive integers with a ratio of 3:4 and LCM 180, prove the first number is 45 -/
theorem first_number_is_45 (a b : ℕ+) (h1 : a.val * 4 = b.val * 3) (h2 : Nat.lcm a.val b.val = 180) : a = 45 := by
  sorry

end NUMINAMATH_CALUDE_first_number_is_45_l3903_390348


namespace NUMINAMATH_CALUDE_merchant_loss_l3903_390327

/-- The total loss incurred by a merchant on a counterfeit transaction -/
def total_loss (purchase_cost : ℕ) (additional_price : ℕ) : ℕ :=
  purchase_cost + additional_price

/-- Theorem stating that under the given conditions, the total loss is 92 yuan -/
theorem merchant_loss :
  let purchase_cost : ℕ := 80
  let additional_price : ℕ := 12
  total_loss purchase_cost additional_price = 92 := by
  sorry

#check merchant_loss

end NUMINAMATH_CALUDE_merchant_loss_l3903_390327


namespace NUMINAMATH_CALUDE_smallest_y_value_l3903_390324

theorem smallest_y_value (x y z : ℝ) : 
  (4 < x ∧ x < z ∧ z < y ∧ y < 10) →
  (∀ a b : ℝ, (4 < a ∧ a < z ∧ z < b ∧ b < 10) → (⌊b⌋ - ⌊a⌋ : ℤ) ≤ 5) →
  (∃ a b : ℝ, (4 < a ∧ a < z ∧ z < b ∧ b < 10) ∧ (⌊b⌋ - ⌊a⌋ : ℤ) = 5) →
  9 ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_value_l3903_390324


namespace NUMINAMATH_CALUDE_unique_arrangement_l3903_390362

-- Define the shapes and colors
inductive Shape : Type
| Triangle : Shape
| Circle : Shape
| Rectangle : Shape
| Rhombus : Shape

inductive Color : Type
| Red : Color
| Blue : Color
| Yellow : Color
| Green : Color

-- Define the position type
inductive Position : Type
| First : Position
| Second : Position
| Third : Position
| Fourth : Position

-- Define the figure type
structure Figure :=
(shape : Shape)
(color : Color)
(position : Position)

def Arrangement := List Figure

-- Define the conditions
def redBetweenBlueAndGreen (arr : Arrangement) : Prop := sorry
def rhombusRightOfYellow (arr : Arrangement) : Prop := sorry
def circleRightOfTriangleAndRhombus (arr : Arrangement) : Prop := sorry
def triangleNotAtEdge (arr : Arrangement) : Prop := sorry
def blueAndYellowNotAdjacent (arr : Arrangement) : Prop := sorry

-- Define the correct arrangement
def correctArrangement : Arrangement := [
  ⟨Shape.Rectangle, Color.Yellow, Position.First⟩,
  ⟨Shape.Rhombus, Color.Green, Position.Second⟩,
  ⟨Shape.Triangle, Color.Red, Position.Third⟩,
  ⟨Shape.Circle, Color.Blue, Position.Fourth⟩
]

-- Theorem statement
theorem unique_arrangement :
  ∀ (arr : Arrangement),
    (redBetweenBlueAndGreen arr) →
    (rhombusRightOfYellow arr) →
    (circleRightOfTriangleAndRhombus arr) →
    (triangleNotAtEdge arr) →
    (blueAndYellowNotAdjacent arr) →
    (arr = correctArrangement) :=
by sorry

end NUMINAMATH_CALUDE_unique_arrangement_l3903_390362


namespace NUMINAMATH_CALUDE_function_property_l3903_390389

/-- Given a function f: ℕ → ℕ satisfying the property that
    for all positive integers a, b, n such that a + b = 3^n,
    f(a) + f(b) = 2n^2, prove that f(3003) = 44 -/
theorem function_property (f : ℕ → ℕ) 
  (h : ∀ (a b n : ℕ), 0 < a → 0 < b → 0 < n → a + b = 3^n → f a + f b = 2*n^2) :
  f 3003 = 44 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3903_390389


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_min_value_achievable_l3903_390359

theorem min_value_trigonometric_expression (α β : ℝ) :
  (3 * Real.cos α + 6 * Real.sin β - 10)^2 + 
  (3 * Real.sin α + 6 * Real.cos β + 4 * Real.cos (α + β) - 20)^2 ≥ 500 :=
by sorry

theorem min_value_achievable :
  ∃ α β : ℝ, (3 * Real.cos α + 6 * Real.sin β - 10)^2 + 
             (3 * Real.sin α + 6 * Real.cos β + 4 * Real.cos (α + β) - 20)^2 = 500 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_min_value_achievable_l3903_390359


namespace NUMINAMATH_CALUDE_fraction_division_simplification_l3903_390343

theorem fraction_division_simplification : (3 / 4) / (5 / 6) = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_simplification_l3903_390343


namespace NUMINAMATH_CALUDE_vertical_line_angle_is_90_degrees_l3903_390391

/-- The angle of inclination of a vertical line -/
def angle_of_vertical_line : ℝ := 90

/-- A vertical line is defined by the equation x = 0 -/
def is_vertical_line (f : ℝ → ℝ) : Prop := ∀ y, f y = 0

theorem vertical_line_angle_is_90_degrees (f : ℝ → ℝ) (h : is_vertical_line f) :
  angle_of_vertical_line = 90 := by
  sorry

end NUMINAMATH_CALUDE_vertical_line_angle_is_90_degrees_l3903_390391


namespace NUMINAMATH_CALUDE_function_linearity_l3903_390350

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (h_continuous : Continuous f)
variable (h_additive : ∀ x y : ℝ, f (x + y) = f x + f y)

-- State the theorem
theorem function_linearity :
  ∃ C : ℝ, (∀ x : ℝ, f x = C * x) ∧ C = f 1 :=
sorry

end NUMINAMATH_CALUDE_function_linearity_l3903_390350


namespace NUMINAMATH_CALUDE_speed_above_limit_l3903_390366

/-- Proves that given a travel distance of 150 miles, a travel time of 2 hours,
    and a speed limit of 60 mph, the difference between the average speed
    and the speed limit is 15 mph. -/
theorem speed_above_limit (distance : ℝ) (time : ℝ) (speed_limit : ℝ) :
  distance = 150 ∧ time = 2 ∧ speed_limit = 60 →
  distance / time - speed_limit = 15 := by
  sorry

end NUMINAMATH_CALUDE_speed_above_limit_l3903_390366


namespace NUMINAMATH_CALUDE_base_8_6_equality_l3903_390338

/-- Checks if a number is a valid digit in a given base -/
def isValidDigit (digit : ℕ) (base : ℕ) : Prop :=
  digit < base

/-- Converts a two-digit number from a given base to base 10 -/
def toBase10 (c d : ℕ) (base : ℕ) : ℕ :=
  base * c + d

/-- The main theorem stating that 0 is the only number satisfying the conditions -/
theorem base_8_6_equality (n : ℕ) : n > 0 → 
  (∃ (c d : ℕ), isValidDigit c 8 ∧ isValidDigit d 8 ∧ 
   isValidDigit c 6 ∧ isValidDigit d 6 ∧
   n = toBase10 c d 8 ∧ n = toBase10 d c 6) → n = 0 :=
by sorry

end NUMINAMATH_CALUDE_base_8_6_equality_l3903_390338


namespace NUMINAMATH_CALUDE_basketball_non_gymnastics_percentage_l3903_390377

theorem basketball_non_gymnastics_percentage 
  (total : ℝ)
  (h_total_pos : total > 0)
  (h_basketball : total * (50 / 100) = total * 0.5)
  (h_gymnastics : total * (40 / 100) = total * 0.4)
  (h_both : (total * 0.5) * (30 / 100) = total * 0.15) :
  let non_gymnastics := total * 0.6
  let basketball_non_gymnastics := total * 0.35
  (basketball_non_gymnastics / non_gymnastics) * 100 = 58 := by
sorry

end NUMINAMATH_CALUDE_basketball_non_gymnastics_percentage_l3903_390377


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l3903_390381

/-- Given that M(3,7) is the midpoint of CD and C(5,3) is one endpoint, 
    the product of the coordinates of point D is 11. -/
theorem midpoint_coordinate_product : 
  ∀ (D : ℝ × ℝ),
  (3, 7) = ((5 + D.1) / 2, (3 + D.2) / 2) →
  D.1 * D.2 = 11 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l3903_390381


namespace NUMINAMATH_CALUDE_simplify_fraction_l3903_390380

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3903_390380


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3903_390367

theorem perfect_square_condition (n : ℕ) : 
  (∃ (k : ℕ), 2^(n+1) * n = k^2) ↔ 
  (∃ (m : ℕ), n = 2 * m^2) ∨ 
  (∃ (k : ℕ), n = k^2 ∧ k % 2 = 1) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3903_390367


namespace NUMINAMATH_CALUDE_largest_two_digit_multiple_of_seven_l3903_390387

def digits : Set Nat := {3, 5, 6, 7}

def is_two_digit (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100

def formed_from_digits (n : Nat) : Prop :=
  ∃ (d1 d2 : Nat), d1 ∈ digits ∧ d2 ∈ digits ∧ d1 ≠ d2 ∧ n = 10 * d1 + d2

theorem largest_two_digit_multiple_of_seven :
  ∀ n : Nat, is_two_digit n → formed_from_digits n → n % 7 = 0 →
  n ≤ 63 :=
sorry

end NUMINAMATH_CALUDE_largest_two_digit_multiple_of_seven_l3903_390387


namespace NUMINAMATH_CALUDE_smallest_start_number_for_2520_divisibility_l3903_390351

theorem smallest_start_number_for_2520_divisibility : 
  ∃ (n : ℕ), n > 0 ∧ n ≤ 10 ∧ 
  (∀ (k : ℕ), n ≤ k ∧ k ≤ 10 → 2520 % k = 0) ∧
  (∀ (m : ℕ), m < n → ∃ (j : ℕ), m < j ∧ j ≤ 10 ∧ 2520 % j ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_start_number_for_2520_divisibility_l3903_390351


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l3903_390397

theorem two_digit_number_puzzle : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n / 10 + n % 10 = 13) ∧
  (10 * (n % 10) + (n / 10) = n - 27) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l3903_390397


namespace NUMINAMATH_CALUDE_simplify_expression_l3903_390306

theorem simplify_expression (x y : ℝ) (h : y = Real.sqrt (x - 2) + Real.sqrt (2 - x) + 2) :
  |y - Real.sqrt 3| - (x - 2 + Real.sqrt 2)^2 = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3903_390306


namespace NUMINAMATH_CALUDE_chosen_number_proof_l3903_390388

theorem chosen_number_proof :
  ∃! (x : ℝ), x > 0 ∧ (Real.sqrt (x^2) / 6) - 189 = 3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l3903_390388


namespace NUMINAMATH_CALUDE_tennis_ball_order_l3903_390349

theorem tennis_ball_order (white yellow : ℕ) (h1 : white = yellow)
  (h2 : (white : ℚ) / ((yellow : ℚ) + 20) = 8 / 13) :
  white + yellow = 64 := by
  sorry

end NUMINAMATH_CALUDE_tennis_ball_order_l3903_390349


namespace NUMINAMATH_CALUDE_alice_bob_number_game_l3903_390363

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem alice_bob_number_game (alice_num bob_num : ℕ) : 
  (1 ≤ alice_num ∧ alice_num ≤ 50) →
  (1 ≤ bob_num ∧ bob_num ≤ 50) →
  (alice_num ≠ 1) →
  (is_prime bob_num) →
  (∃ m : ℕ, 100 * bob_num + alice_num = m * m) →
  (alice_num = 24 ∨ alice_num = 61) :=
by sorry

end NUMINAMATH_CALUDE_alice_bob_number_game_l3903_390363


namespace NUMINAMATH_CALUDE_modified_tic_tac_toe_tie_probability_l3903_390370

theorem modified_tic_tac_toe_tie_probability 
  (amy_win_prob : ℚ) 
  (lily_win_prob : ℚ) 
  (h1 : amy_win_prob = 2/5) 
  (h2 : lily_win_prob = 1/4) 
  (h3 : amy_win_prob ≥ 2 * lily_win_prob ∨ lily_win_prob ≥ 2 * amy_win_prob) : 
  1 - (amy_win_prob + lily_win_prob) = 7/20 :=
by sorry

end NUMINAMATH_CALUDE_modified_tic_tac_toe_tie_probability_l3903_390370


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3903_390354

/-- The atomic weight of Hydrogen in atomic mass units (amu) -/
def atomic_weight_H : ℝ := 1.008

/-- The atomic weight of Chromium in atomic mass units (amu) -/
def atomic_weight_Cr : ℝ := 51.996

/-- The atomic weight of Oxygen in atomic mass units (amu) -/
def atomic_weight_O : ℝ := 15.999

/-- The number of Hydrogen atoms in the compound -/
def num_H : ℕ := 2

/-- The number of Chromium atoms in the compound -/
def num_Cr : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 4

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight : ℝ := num_H * atomic_weight_H + num_Cr * atomic_weight_Cr + num_O * atomic_weight_O

theorem compound_molecular_weight : molecular_weight = 118.008 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3903_390354


namespace NUMINAMATH_CALUDE_apples_left_l3903_390314

/-- The number of apples Mike picked -/
def mike_apples : ℝ := 7.0

/-- The number of apples Nancy picked -/
def nancy_apples : ℝ := 3.0

/-- The number of apples Keith ate -/
def keith_apples : ℝ := 6.0

/-- Theorem: The number of apples left after Mike and Nancy picked apples and Keith ate some -/
theorem apples_left : mike_apples + nancy_apples - keith_apples = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_l3903_390314


namespace NUMINAMATH_CALUDE_equation_equivalence_l3903_390395

theorem equation_equivalence (x y : ℝ) :
  (3 * x^2 + 9 * x + 7 * y + 2 = 0) ∧ (3 * x + 2 * y + 5 = 0) →
  4 * y^2 + 23 * y - 14 = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3903_390395


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3903_390308

def M : Set ℝ := {x | x < (1 : ℝ) / 2}
def N : Set ℝ := {x | x ≥ -4}

theorem intersection_of_M_and_N : M ∩ N = {x | -4 ≤ x ∧ x < (1 : ℝ) / 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3903_390308


namespace NUMINAMATH_CALUDE_smallest_base_for_200_proof_l3903_390378

/-- The smallest base in which 200 (base 10) has exactly 6 digits -/
def smallest_base_for_200 : ℕ := 2

theorem smallest_base_for_200_proof :
  smallest_base_for_200 = 2 ∧
  2^7 ≤ 200 ∧
  200 < 2^8 ∧
  ∀ b : ℕ, 1 < b → b < 2 →
    (b^5 > 200 ∨ b^6 ≤ 200) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_200_proof_l3903_390378


namespace NUMINAMATH_CALUDE_min_beta_delta_sum_l3903_390358

theorem min_beta_delta_sum (g : ℂ → ℂ) (β δ : ℂ) :
  (∀ z, g z = (3 + 2 * Complex.I) * z^2 + β * z + δ) →
  (g 1).im = 0 →
  (g (-Complex.I)).im = 0 →
  ∃ (β₀ δ₀ : ℂ), Complex.abs β₀ + Complex.abs δ₀ = 2 ∧
    ∀ β' δ', (∀ z, g z = (3 + 2 * Complex.I) * z^2 + β' * z + δ') →
              (g 1).im = 0 →
              (g (-Complex.I)).im = 0 →
              Complex.abs β' + Complex.abs δ' ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_beta_delta_sum_l3903_390358


namespace NUMINAMATH_CALUDE_quadratic_to_linear_equations_l3903_390310

theorem quadratic_to_linear_equations :
  ∀ x y : ℝ, x^2 - 4*x*y + 4*y^2 = 4 ↔ (x - 2*y + 2 = 0 ∨ x - 2*y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_to_linear_equations_l3903_390310


namespace NUMINAMATH_CALUDE_min_difference_for_equal_f_values_l3903_390371

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then Real.log x else (1/2) * x + (1/2)

theorem min_difference_for_equal_f_values :
  ∃ (min_diff : ℝ),
    min_diff = 3 - 2 * Real.log 2 ∧
    ∀ (m n : ℝ), m < n → f m = f n → n - m ≥ min_diff :=
by sorry

end NUMINAMATH_CALUDE_min_difference_for_equal_f_values_l3903_390371


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l3903_390341

/-- Represents a number with n repetitions of a digit in base 10 -/
def repeatedDigit (digit : Nat) (n : Nat) : Nat :=
  digit * ((10^n - 1) / 9)

/-- Calculates the sum of digits of a number in base 10 -/
def sumOfDigits (n : Nat) : Nat :=
  sorry

theorem sum_of_digits_9ab :
  let a := repeatedDigit 9 1977
  let b := repeatedDigit 6 1977
  sumOfDigits (9 * a * b) = 25694 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l3903_390341


namespace NUMINAMATH_CALUDE_sock_selection_l3903_390385

theorem sock_selection (n : ℕ) : 
  (Nat.choose 10 n = 90) → n = 2 := by
sorry

end NUMINAMATH_CALUDE_sock_selection_l3903_390385


namespace NUMINAMATH_CALUDE_linear_function_characterization_l3903_390382

/-- A linear function f satisfying f(f(x)) = 16x - 15 -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  (∃ a b : ℝ, ∀ x, f x = a * x + b) ∧
  (∀ x, f (f x) = 16 * x - 15)

/-- The theorem stating that a linear function satisfying f(f(x)) = 16x - 15 
    must be either 4x - 3 or -4x + 5 -/
theorem linear_function_characterization (f : ℝ → ℝ) :
  LinearFunction f → 
  ((∀ x, f x = 4 * x - 3) ∨ (∀ x, f x = -4 * x + 5)) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_characterization_l3903_390382


namespace NUMINAMATH_CALUDE_percentage_problem_l3903_390345

theorem percentage_problem (P : ℝ) (x : ℝ) (h1 : x = 412.5) 
  (h2 : (P / 100) * x = (1 / 3) * x + 110) : P = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3903_390345


namespace NUMINAMATH_CALUDE_smallest_m_for_nth_root_in_T_l3903_390342

def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

theorem smallest_m_for_nth_root_in_T : 
  (∀ n : ℕ, n ≥ 12 → ∃ z ∈ T, z^n = 1) ∧ 
  (∀ m : ℕ, m < 12 → ∃ n : ℕ, n ≥ m ∧ ∀ z ∈ T, z^n ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_nth_root_in_T_l3903_390342


namespace NUMINAMATH_CALUDE_sum_of_first_100_inverse_terms_l3903_390319

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => sequence_a n + n + 1

def sequence_inverse_a (n : ℕ) : ℚ := 1 / sequence_a n

theorem sum_of_first_100_inverse_terms :
  (Finset.range 100).sum sequence_inverse_a = 200 / 101 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_100_inverse_terms_l3903_390319


namespace NUMINAMATH_CALUDE_million_is_ten_to_six_roundness_of_million_l3903_390301

/-- Roundness of a positive integer is the sum of the exponents in its prime factorization. -/
def roundness (n : ℕ) : ℕ := sorry

/-- 1,000,000 can be expressed as 10^6 -/
theorem million_is_ten_to_six : 1000000 = 10^6 := by sorry

/-- The roundness of 1,000,000 is 12 -/
theorem roundness_of_million : roundness 1000000 = 12 := by sorry

end NUMINAMATH_CALUDE_million_is_ten_to_six_roundness_of_million_l3903_390301


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3903_390357

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 7) :
  let perimeter := a + 2 * b
  perimeter = 17 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3903_390357


namespace NUMINAMATH_CALUDE_chicken_bucket_capacity_l3903_390329

/-- Represents the cost of a chicken bucket with sides in dollars -/
def bucket_cost : ℚ := 12

/-- Represents the total amount Monty spent in dollars -/
def total_spent : ℚ := 72

/-- Represents the number of family members Monty fed -/
def family_members : ℕ := 36

/-- Represents the number of people one chicken bucket with sides can feed -/
def people_per_bucket : ℕ := 6

/-- Proves that one chicken bucket with sides can feed 6 people -/
theorem chicken_bucket_capacity :
  (total_spent / bucket_cost) * people_per_bucket = family_members :=
by sorry

end NUMINAMATH_CALUDE_chicken_bucket_capacity_l3903_390329


namespace NUMINAMATH_CALUDE_no_smallest_rational_l3903_390332

theorem no_smallest_rational : ¬ ∃ q : ℚ, ∀ r : ℚ, q ≤ r := by
  sorry

end NUMINAMATH_CALUDE_no_smallest_rational_l3903_390332


namespace NUMINAMATH_CALUDE_tobias_driveways_shoveled_l3903_390361

/-- Calculates the number of driveways shoveled by Tobias given his income and expenses. -/
theorem tobias_driveways_shoveled 
  (original_price : ℚ)
  (discount_rate : ℚ)
  (tax_rate : ℚ)
  (monthly_allowance : ℚ)
  (lawn_fee : ℚ)
  (driveway_fee : ℚ)
  (hourly_wage : ℚ)
  (remaining_money : ℚ)
  (months_saved : ℕ)
  (hours_worked : ℕ)
  (lawns_mowed : ℕ)
  (h1 : original_price = 95)
  (h2 : discount_rate = 1/10)
  (h3 : tax_rate = 1/20)
  (h4 : monthly_allowance = 5)
  (h5 : lawn_fee = 15)
  (h6 : driveway_fee = 7)
  (h7 : hourly_wage = 8)
  (h8 : remaining_money = 15)
  (h9 : months_saved = 3)
  (h10 : hours_worked = 10)
  (h11 : lawns_mowed = 4) :
  ∃ (driveways_shoveled : ℕ), driveways_shoveled = 7 :=
by sorry


end NUMINAMATH_CALUDE_tobias_driveways_shoveled_l3903_390361


namespace NUMINAMATH_CALUDE_tetrahedron_octahedron_volume_ratio_l3903_390374

/-- The volume of a regular tetrahedron -/
def tetrahedronVolume (edgeLength : ℝ) : ℝ := sorry

/-- The volume of a regular octahedron -/
def octahedronVolume (edgeLength : ℝ) : ℝ := sorry

/-- Theorem: The ratio of the volume of a regular tetrahedron to the volume of a regular octahedron 
    with the same edge length is 1/2 -/
theorem tetrahedron_octahedron_volume_ratio (edgeLength : ℝ) (h : edgeLength > 0) : 
  tetrahedronVolume edgeLength / octahedronVolume edgeLength = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_octahedron_volume_ratio_l3903_390374


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3903_390347

theorem point_in_fourth_quadrant (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ x y : ℝ, x ≠ y ∧ a * x^2 - x - 1/4 = 0 ∧ a * y^2 - y - 1/4 = 0) :
  (a + 1 > 0) ∧ (-3 - a < 0) := by
  sorry


end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3903_390347


namespace NUMINAMATH_CALUDE_problem_solution_l3903_390368

theorem problem_solution : (2210 - 2137)^2 + (2137 - 2028)^2 = 64 * 268.90625 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3903_390368


namespace NUMINAMATH_CALUDE_pet_store_house_cats_l3903_390337

theorem pet_store_house_cats 
  (initial_siamese : ℕ)
  (sold : ℕ)
  (remaining : ℕ)
  (h1 : initial_siamese = 19)
  (h2 : sold = 56)
  (h3 : remaining = 8) :
  ∃ initial_house : ℕ, 
    initial_house = 45 ∧ 
    initial_siamese + initial_house = sold + remaining :=
by sorry

end NUMINAMATH_CALUDE_pet_store_house_cats_l3903_390337


namespace NUMINAMATH_CALUDE_afternoon_letters_indeterminate_l3903_390365

/-- Represents the number of items Jack received at different times of the day -/
structure JacksItems where
  morning_emails : ℕ
  morning_letters : ℕ
  afternoon_emails : ℕ
  afternoon_letters : ℕ

/-- The given conditions about Jack's received items -/
def jack_conditions (items : JacksItems) : Prop :=
  items.morning_emails = 10 ∧
  items.morning_letters = 12 ∧
  items.afternoon_emails = 3 ∧
  items.morning_emails = items.afternoon_emails + 7

/-- Theorem stating that the number of afternoon letters cannot be determined -/
theorem afternoon_letters_indeterminate (items : JacksItems) 
  (h : jack_conditions items) : 
  ¬∃ (n : ℕ), ∀ (items' : JacksItems), 
    jack_conditions items' → items'.afternoon_letters = n :=
sorry

end NUMINAMATH_CALUDE_afternoon_letters_indeterminate_l3903_390365
