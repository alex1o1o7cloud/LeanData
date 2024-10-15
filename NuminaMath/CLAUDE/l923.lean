import Mathlib

namespace NUMINAMATH_CALUDE_no_divisible_lilac_flowers_l923_92396

theorem no_divisible_lilac_flowers : ¬∃ (q c : ℕ), 
  (∃ (p₁ p₂ : ℕ), q + c = p₂^2 ∧ 4*q + 5*c = p₁^2) ∧ 
  (∃ (x : ℕ), q = c * x) := by
sorry

end NUMINAMATH_CALUDE_no_divisible_lilac_flowers_l923_92396


namespace NUMINAMATH_CALUDE_substitution_remainder_l923_92311

/-- Represents the number of available players -/
def total_players : ℕ := 15

/-- Represents the number of starting players -/
def starting_players : ℕ := 5

/-- Represents the maximum number of substitutions allowed -/
def max_substitutions : ℕ := 4

/-- Calculates the number of ways to make substitutions for a given number of substitutions -/
def substitution_ways (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then starting_players * (total_players - starting_players)
  else starting_players * (total_players - starting_players - n + 2) * substitution_ways (n - 1)

/-- Calculates the total number of ways to make substitutions -/
def total_substitution_ways : ℕ :=
  (List.range (max_substitutions + 1)).map substitution_ways |>.sum

/-- The main theorem stating that the remainder of total substitution ways divided by 1000 is 301 -/
theorem substitution_remainder :
  total_substitution_ways % 1000 = 301 := by sorry

end NUMINAMATH_CALUDE_substitution_remainder_l923_92311


namespace NUMINAMATH_CALUDE_smallest_positive_a_for_parabola_l923_92399

theorem smallest_positive_a_for_parabola :
  ∀ (a b c : ℚ),
  (∀ x y : ℚ, y = a * x^2 + b * x + c ↔ y + 5/4 = a * (x - 1/2)^2) →
  a > 0 →
  ∃ n : ℤ, a + b + c = n →
  (∀ a' : ℚ, a' > 0 → (∃ b' c' : ℚ, (∀ x y : ℚ, y = a' * x^2 + b' * x + c' ↔ y + 5/4 = a' * (x - 1/2)^2) ∧ 
                      (∃ n' : ℤ, a' + b' + c' = n')) → a' ≥ a) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_a_for_parabola_l923_92399


namespace NUMINAMATH_CALUDE_range_of_a_l923_92387

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 + 1 ≥ a

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 1 = 0

-- Define the main theorem
theorem range_of_a (a : ℝ) : 
  (¬(¬(p a) ∨ ¬(q a))) → (a ≤ -1 ∨ (1 ≤ a ∧ a ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l923_92387


namespace NUMINAMATH_CALUDE_sum_of_50th_terms_l923_92373

theorem sum_of_50th_terms (a₁ a₅₀ : ℝ) (d : ℝ) (g₁ g₅₀ : ℝ) (r : ℝ) : 
  a₁ = 3 → d = 6 → g₁ = 2 → r = 3 →
  a₅₀ = a₁ + 49 * d →
  g₅₀ = g₁ * r^49 →
  a₅₀ + g₅₀ = 297 + 2 * 3^49 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_50th_terms_l923_92373


namespace NUMINAMATH_CALUDE_ratio_sum_squares_to_sum_l923_92397

theorem ratio_sum_squares_to_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : b = 2 * a) (h5 : c = 4 * a) (h6 : a^2 + b^2 + c^2 = 1701) : 
  a + b + c = 63 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_squares_to_sum_l923_92397


namespace NUMINAMATH_CALUDE_tank_capacity_l923_92339

/-- Represents a cylindrical water tank -/
structure WaterTank where
  capacity : ℝ
  initialWater : ℝ

/-- Proves that the tank's capacity is 30 liters given the conditions -/
theorem tank_capacity (tank : WaterTank)
  (h1 : tank.initialWater / tank.capacity = 1 / 6)
  (h2 : (tank.initialWater + 5) / tank.capacity = 1 / 3) :
  tank.capacity = 30 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l923_92339


namespace NUMINAMATH_CALUDE_necessary_is_necessary_necessary_not_sufficient_l923_92369

-- Define the proposition p
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 4*x + 2*m ≥ 0

-- Define the necessary condition
def necessary_condition (m : ℝ) : Prop := m ≥ 1

-- Theorem: The necessary condition is indeed necessary
theorem necessary_is_necessary : 
  ∀ m : ℝ, p m → necessary_condition m := by sorry

-- Theorem: The necessary condition is not sufficient
theorem necessary_not_sufficient :
  ∃ m : ℝ, necessary_condition m ∧ ¬(p m) := by sorry

end NUMINAMATH_CALUDE_necessary_is_necessary_necessary_not_sufficient_l923_92369


namespace NUMINAMATH_CALUDE_smallest_special_number_is_correct_l923_92368

/-- The smallest positive integer that is not prime, not a square, and has no prime factor less than 100 -/
def smallest_special_number : ℕ := 10403

/-- A number is special if it is not prime, not a square, and has no prime factor less than 100 -/
def is_special (n : ℕ) : Prop :=
  ¬ Nat.Prime n ∧ ¬ ∃ m : ℕ, n = m * m ∧ ∀ p : ℕ, Nat.Prime p → p < 100 → ¬ p ∣ n

theorem smallest_special_number_is_correct :
  is_special smallest_special_number ∧
  ∀ n : ℕ, 0 < n → n < smallest_special_number → ¬ is_special n :=
sorry

end NUMINAMATH_CALUDE_smallest_special_number_is_correct_l923_92368


namespace NUMINAMATH_CALUDE_greatest_number_with_odd_factors_l923_92317

theorem greatest_number_with_odd_factors : ∃ n : ℕ, 
  n < 200 ∧ 
  (∃ k : ℕ, n = k^2) ∧
  (∀ m : ℕ, m < 200 → (∃ j : ℕ, m = j^2) → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_odd_factors_l923_92317


namespace NUMINAMATH_CALUDE_internal_diagonal_intersects_576_cubes_l923_92390

def rectangular_solid_dimensions : ℕ × ℕ × ℕ := (120, 210, 336)

-- Function to calculate the number of cubes intersected by the diagonal
def intersected_cubes (dims : ℕ × ℕ × ℕ) : ℕ :=
  let (x, y, z) := dims
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

theorem internal_diagonal_intersects_576_cubes :
  intersected_cubes rectangular_solid_dimensions = 576 := by
  sorry

end NUMINAMATH_CALUDE_internal_diagonal_intersects_576_cubes_l923_92390


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_U_l923_92308

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {3, 4, 5}

theorem complement_of_A_wrt_U : (U \ A) = {1, 2, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_U_l923_92308


namespace NUMINAMATH_CALUDE_expansion_terms_count_l923_92300

theorem expansion_terms_count (N : ℕ+) : 
  (Nat.choose N 5 = 2002) ↔ (N = 16) := by sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l923_92300


namespace NUMINAMATH_CALUDE_rectangle_area_l923_92324

/-- Given a rectangle made from a wire of length 28 cm with a width of 6 cm, prove that its area is 48 cm². -/
theorem rectangle_area (wire_length : ℝ) (width : ℝ) (area : ℝ) :
  wire_length = 28 →
  width = 6 →
  area = (wire_length / 2 - width) * width →
  area = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l923_92324


namespace NUMINAMATH_CALUDE_ellipse_foci_l923_92326

/-- The equation of an ellipse in the form (x²/a² + y²/b² = 1) -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- The coordinates of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given ellipse equation (x²/2 + y² = 1), prove that its foci are at (±1, 0) -/
theorem ellipse_foci (e : Ellipse) (h : e.a^2 = 2 ∧ e.b^2 = 1) :
  ∃ (p₁ p₂ : Point), p₁.x = 1 ∧ p₁.y = 0 ∧ p₂.x = -1 ∧ p₂.y = 0 ∧
  (∀ (p : Point), (p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1) →
    (p = p₁ ∨ p = p₂ → 
      (p.x - 0)^2 + (p.y - 0)^2 = (e.a^2 - e.b^2))) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l923_92326


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l923_92381

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ (l1.a ≠ 0 ∨ l1.b ≠ 0) ∧ (l2.a ≠ 0 ∨ l2.b ≠ 0)

theorem parallel_lines_m_value :
  let l1 : Line := { a := 3, b := 4, c := -3 }
  let l2 : Line := { a := 6, b := m, c := 14 }
  parallel l1 l2 → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l923_92381


namespace NUMINAMATH_CALUDE_range_of_positive_integers_in_list_l923_92380

def consecutive_integers (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => start + i)

def positive_integers (l : List ℤ) : List ℤ :=
  l.filter (λ x => x > 0)

def range_of_list (l : List ℤ) : ℤ :=
  l.maximum.getD 0 - l.minimum.getD 0

theorem range_of_positive_integers_in_list :
  let d := consecutive_integers (-4) 12
  let positives := positive_integers d
  range_of_list positives = 6 := by
sorry

end NUMINAMATH_CALUDE_range_of_positive_integers_in_list_l923_92380


namespace NUMINAMATH_CALUDE_quadratic_polynomial_proof_l923_92363

theorem quadratic_polynomial_proof : ∃ (q : ℝ → ℝ),
  (∀ x, q x = (19 * x^2 - 2 * x + 13) / 15) ∧
  q (-2) = 9 ∧
  q 1 = 2 ∧
  q 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_proof_l923_92363


namespace NUMINAMATH_CALUDE_rectangle_area_18_l923_92386

def rectangle_pairs : Set (ℕ × ℕ) :=
  {p | p.1 * p.2 = 18 ∧ p.1 < p.2 ∧ p.1 > 0 ∧ p.2 > 0}

theorem rectangle_area_18 :
  rectangle_pairs = {(1, 18), (2, 9), (3, 6)} := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_18_l923_92386


namespace NUMINAMATH_CALUDE_different_color_probability_l923_92305

theorem different_color_probability (total : ℕ) (white : ℕ) (black : ℕ) (drawn : ℕ) :
  total = white + black →
  total = 5 →
  white = 3 →
  black = 2 →
  drawn = 2 →
  (Nat.choose white 1 * Nat.choose black 1 : ℚ) / Nat.choose total drawn = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l923_92305


namespace NUMINAMATH_CALUDE_cube_immersion_theorem_l923_92375

/-- The edge length of a cube that, when immersed in a rectangular vessel,
    causes a specific rise in water level. -/
def cube_edge_length (vessel_length vessel_width water_rise : ℝ) : ℝ :=
  (vessel_length * vessel_width * water_rise) ^ (1/3)

/-- Theorem stating that a cube with edge length 16 cm, when immersed in a
    rectangular vessel with base 20 cm × 15 cm, causes a water level rise
    of 13.653333333333334 cm. -/
theorem cube_immersion_theorem :
  cube_edge_length 20 15 13.653333333333334 = 16 := by
  sorry

end NUMINAMATH_CALUDE_cube_immersion_theorem_l923_92375


namespace NUMINAMATH_CALUDE_rahul_ppf_savings_l923_92325

/-- Represents Rahul's savings in rupees -/
structure RahulSavings where
  nsc : ℕ  -- National Savings Certificate
  ppf : ℕ  -- Public Provident Fund

/-- The conditions of Rahul's savings -/
def savingsConditions (s : RahulSavings) : Prop :=
  s.nsc + s.ppf = 180000 ∧ s.nsc / 3 = s.ppf / 2

/-- Theorem stating Rahul's Public Provident Fund savings -/
theorem rahul_ppf_savings (s : RahulSavings) (h : savingsConditions s) : s.ppf = 72000 := by
  sorry

#check rahul_ppf_savings

end NUMINAMATH_CALUDE_rahul_ppf_savings_l923_92325


namespace NUMINAMATH_CALUDE_converse_not_always_true_l923_92321

theorem converse_not_always_true : ∃ (a b : ℝ), a < b ∧ ¬(∀ (m : ℝ), a * m^2 < b * m^2) :=
sorry

end NUMINAMATH_CALUDE_converse_not_always_true_l923_92321


namespace NUMINAMATH_CALUDE_internet_cost_decrease_l923_92370

theorem internet_cost_decrease (initial_cost final_cost : ℝ) 
  (h1 : initial_cost = 120)
  (h2 : final_cost = 45) : 
  (initial_cost - final_cost) / initial_cost * 100 = 62.5 := by
sorry

end NUMINAMATH_CALUDE_internet_cost_decrease_l923_92370


namespace NUMINAMATH_CALUDE_difference_of_squares_l923_92304

def digits : List Nat := [9, 8, 7, 6, 4, 2, 1, 5]

def largest_number : Nat := 98765421

def smallest_number : Nat := 12456789

theorem difference_of_squares (d : List Nat) (largest smallest : Nat) :
  d = digits →
  largest = largest_number →
  smallest = smallest_number →
  largest * largest - smallest * smallest = 9599477756293120 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l923_92304


namespace NUMINAMATH_CALUDE_camera_price_difference_l923_92314

/-- The list price of Camera Y in dollars -/
def list_price : ℝ := 59.99

/-- The discount percentage at Budget Buys -/
def budget_buys_discount : ℝ := 0.15

/-- The discount amount at Frugal Finds in dollars -/
def frugal_finds_discount : ℝ := 20

/-- The sale price at Budget Buys in dollars -/
def budget_buys_price : ℝ := list_price * (1 - budget_buys_discount)

/-- The sale price at Frugal Finds in dollars -/
def frugal_finds_price : ℝ := list_price - frugal_finds_discount

/-- The price difference in cents -/
def price_difference_cents : ℝ := (budget_buys_price - frugal_finds_price) * 100

theorem camera_price_difference :
  price_difference_cents = 1099.15 := by
  sorry

end NUMINAMATH_CALUDE_camera_price_difference_l923_92314


namespace NUMINAMATH_CALUDE_square_ratio_side_length_l923_92357

theorem square_ratio_side_length (area_ratio : ℚ) :
  area_ratio = 75 / 128 →
  ∃ (a b c : ℕ), 
    (a = 5 ∧ b = 6 ∧ c = 16) ∧
    (Real.sqrt area_ratio = a * Real.sqrt b / c) :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_l923_92357


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_700_l923_92395

theorem greatest_multiple_of_5_and_6_less_than_700 : 
  ∃ n : ℕ, n = 690 ∧ 
  (∀ m : ℕ, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 700 → m ≤ n) ∧
  n % 5 = 0 ∧ n % 6 = 0 ∧ n < 700 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_700_l923_92395


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l923_92303

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s * s = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l923_92303


namespace NUMINAMATH_CALUDE_expected_weight_of_disks_l923_92313

/-- The expected weight of 100 disks with manufacturing errors -/
theorem expected_weight_of_disks (nominal_diameter : Real) (perfect_weight : Real) 
  (radius_std_dev : Real) (h1 : nominal_diameter = 1) (h2 : perfect_weight = 100) 
  (h3 : radius_std_dev = 0.01) : 
  ∃ (expected_weight : Real), 
    expected_weight = 10004 ∧ 
    expected_weight = 100 * perfect_weight * (1 + (radius_std_dev / (nominal_diameter / 2))^2) :=
by sorry

end NUMINAMATH_CALUDE_expected_weight_of_disks_l923_92313


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l923_92348

theorem simplify_complex_fraction (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -3) :
  (x + 2 - 5 / (x - 2)) / ((x + 3) / (x - 2)) = x - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l923_92348


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_equals_zero_l923_92337

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- Definition of line l₁ -/
def line_l₁ (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => a * x + 2 * y - 1 = 0

/-- Definition of line l₂ -/
def line_l₂ (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => 3 * x - a * y + 1 = 0

/-- The main theorem -/
theorem perpendicular_lines_a_equals_zero (a : ℝ) :
  perpendicular (a / 2) (-3 / a) → a = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_equals_zero_l923_92337


namespace NUMINAMATH_CALUDE_apps_files_difference_l923_92398

/-- Given Dave's initial and final numbers of apps and files on his phone, prove that he has 7 more apps than files left. -/
theorem apps_files_difference (initial_apps initial_files final_apps final_files : ℕ) :
  initial_apps = 24 →
  initial_files = 9 →
  final_apps = 12 →
  final_files = 5 →
  final_apps - final_files = 7 := by
  sorry

end NUMINAMATH_CALUDE_apps_files_difference_l923_92398


namespace NUMINAMATH_CALUDE_quadratic_equation_satisfaction_l923_92344

theorem quadratic_equation_satisfaction (p q : ℝ) : 
  p^2 + 9*q^2 + 3*p - p*q = 30 ∧ p - 5*q - 8 = 0 → p^2 - p - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_satisfaction_l923_92344


namespace NUMINAMATH_CALUDE_sufficient_condition_ranges_not_sufficient_condition_ranges_l923_92382

/-- Condition p: (x+1)(2-x) ≥ 0 -/
def p (x : ℝ) : Prop := (x + 1) * (2 - x) ≥ 0

/-- Condition q: x^2+mx-2m^2-3m-1 < 0, where m > -2/3 -/
def q (x m : ℝ) : Prop := x^2 + m*x - 2*m^2 - 3*m - 1 < 0 ∧ m > -2/3

theorem sufficient_condition_ranges (m : ℝ) :
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) → m > 1 :=
sorry

theorem not_sufficient_condition_ranges (m : ℝ) :
  (∀ x, ¬p x → ¬q x m) ∧ (∃ x, ¬q x m ∧ p x) → -2/3 < m ∧ m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_ranges_not_sufficient_condition_ranges_l923_92382


namespace NUMINAMATH_CALUDE_complex_sum_power_l923_92393

theorem complex_sum_power (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^100 + z^101 + z^102 + z^103 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_sum_power_l923_92393


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l923_92342

theorem min_value_of_exponential_sum (a b : ℝ) (h : 2 * a + 3 * b = 4) :
  ∃ (m : ℝ), m = 8 ∧ ∀ x y, 2 * x + 3 * y = 4 → 4^x + 8^y ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l923_92342


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l923_92301

theorem sum_of_special_numbers : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  a % 10^8 = 0 ∧ 
  b % 10^8 = 0 ∧ 
  (Nat.divisors a).card = 90 ∧ 
  (Nat.divisors b).card = 90 ∧ 
  a + b = 700000000 := by
sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l923_92301


namespace NUMINAMATH_CALUDE_angle_measure_problem_l923_92302

theorem angle_measure_problem (x : ℝ) : 
  x = 2 * (180 - x) + 30 → x = 130 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_problem_l923_92302


namespace NUMINAMATH_CALUDE_periodic_function_value_l923_92346

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β) + 4,
    where a, b, α, β are non-zero real numbers, and f(2007) = 5,
    prove that f(2008) = 3 -/
theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  f 2007 = 5 → f 2008 = 3 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l923_92346


namespace NUMINAMATH_CALUDE_max_value_z_l923_92361

theorem max_value_z (x y : ℝ) (h1 : 6 ≤ x + y) (h2 : x + y ≤ 8) (h3 : -2 ≤ x - y) (h4 : x - y ≤ 0) :
  ∃ (z : ℝ), z = 2 * x + 5 * y ∧ z ≤ 8 ∧ ∀ (w : ℝ), w = 2 * x + 5 * y → w ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_value_z_l923_92361


namespace NUMINAMATH_CALUDE_older_sibling_age_l923_92391

def mother_charge : ℚ := 495 / 100
def child_charge_per_year : ℚ := 35 / 100
def total_bill : ℚ := 985 / 100

def is_valid_age_combination (twin_age older_age : ℕ) : Prop :=
  twin_age ≤ older_age ∧
  mother_charge + child_charge_per_year * (2 * twin_age + older_age) = total_bill

theorem older_sibling_age :
  ∃ (twin_age older_age : ℕ), is_valid_age_combination twin_age older_age ∧
  (older_age = 4 ∨ older_age = 6) :=
by sorry

end NUMINAMATH_CALUDE_older_sibling_age_l923_92391


namespace NUMINAMATH_CALUDE_complex_number_properties_l923_92345

theorem complex_number_properties (z : ℂ) (h : z * (2 + Complex.I) = Complex.I ^ 10) :
  (Complex.abs z = Real.sqrt 5 / 5) ∧
  (Complex.re z < 0 ∧ Complex.im z > 0) :=
sorry

end NUMINAMATH_CALUDE_complex_number_properties_l923_92345


namespace NUMINAMATH_CALUDE_divisibility_of_n_squared_n_squared_minus_one_l923_92352

theorem divisibility_of_n_squared_n_squared_minus_one (n : ℤ) : 
  12 ∣ n^2 * (n^2 - 1) := by sorry

end NUMINAMATH_CALUDE_divisibility_of_n_squared_n_squared_minus_one_l923_92352


namespace NUMINAMATH_CALUDE_rental_cost_equality_l923_92338

/-- Represents the rental cost scenario for two computers -/
structure RentalCost where
  B : ℝ  -- Hourly rate for computer B
  T : ℝ  -- Time taken by computer A to complete the job

/-- The total cost is the same for both computers and equals 70 times the hourly rate of computer B -/
theorem rental_cost_equality (rc : RentalCost) : 
  1.4 * rc.B * rc.T = rc.B * (rc.T + 20) ∧ 
  1.4 * rc.B * rc.T = 70 * rc.B :=
by sorry

end NUMINAMATH_CALUDE_rental_cost_equality_l923_92338


namespace NUMINAMATH_CALUDE_cycle_selling_price_l923_92360

/-- Calculates the selling price of a cycle given its cost price and gain percent -/
def calculate_selling_price (cost_price : ℚ) (gain_percent : ℚ) : ℚ :=
  cost_price * (1 + gain_percent / 100)

/-- Theorem: The selling price of a cycle bought for 840 with 45.23809523809524% gain is 1220 -/
theorem cycle_selling_price : 
  calculate_selling_price 840 45.23809523809524 = 1220 := by
  sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l923_92360


namespace NUMINAMATH_CALUDE_turtleneck_discount_l923_92385

theorem turtleneck_discount (C : ℝ) (C_pos : C > 0) : 
  let initial_price := 1.2 * C
  let marked_up_price := 1.25 * initial_price
  let final_price := (1 - 0.08) * marked_up_price
  final_price = 1.38 * C := by sorry

end NUMINAMATH_CALUDE_turtleneck_discount_l923_92385


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l923_92388

/-- Given a triangle ABC with points D and E as described, prove that the intersection P of BE and AD
    has the vector representation P = (8/14)A + (1/14)B + (4/14)C -/
theorem intersection_point_coordinates (A B C D E P : ℝ × ℝ) : 
  (∃ (k : ℝ), D = k • C + (1 - k) • B ∧ k = 5/4) →  -- BD:DC = 4:1
  (∃ (m : ℝ), E = m • A + (1 - m) • C ∧ m = 2/3) →  -- AE:EC = 2:1
  (∃ (t : ℝ), P = t • A + (1 - t) • D) →            -- P is on AD
  (∃ (s : ℝ), P = s • B + (1 - s) • E) →            -- P is on BE
  (∃ (x y z : ℝ), P = x • A + y • B + z • C ∧ x + y + z = 1) →
  P = (8/14) • A + (1/14) • B + (4/14) • C :=
by sorry


end NUMINAMATH_CALUDE_intersection_point_coordinates_l923_92388


namespace NUMINAMATH_CALUDE_system_solution_l923_92349

theorem system_solution :
  ∃ (x y : ℝ), 
    y * (x + y)^2 = 9 ∧
    y * (x^3 - y^3) = 7 ∧
    x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l923_92349


namespace NUMINAMATH_CALUDE_max_value_abc_l923_92323

theorem max_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b * c * (a + b + c + a * b)) / ((a + b)^3 * (b + c)^3) ≤ 1/16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_abc_l923_92323


namespace NUMINAMATH_CALUDE_min_distinct_values_l923_92320

/-- Represents a list of integers with a unique mode -/
structure IntegerList where
  elements : List Nat
  mode_count : Nat
  distinct_count : Nat
  mode_is_unique : Bool

/-- Properties of the integer list -/
def valid_integer_list (L : IntegerList) : Prop :=
  L.elements.length = 2018 ∧
  L.mode_count = 10 ∧
  L.mode_is_unique = true

/-- Theorem stating the minimum number of distinct values -/
theorem min_distinct_values (L : IntegerList) :
  valid_integer_list L → L.distinct_count ≥ 225 := by
  sorry

#check min_distinct_values

end NUMINAMATH_CALUDE_min_distinct_values_l923_92320


namespace NUMINAMATH_CALUDE_seven_pow_2015_ends_with_43_l923_92333

/-- The last two digits of a natural number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- 7^2015 ends with 43 -/
theorem seven_pow_2015_ends_with_43 : lastTwoDigits (7^2015) = 43 := by
  sorry

#check seven_pow_2015_ends_with_43

end NUMINAMATH_CALUDE_seven_pow_2015_ends_with_43_l923_92333


namespace NUMINAMATH_CALUDE_incorrect_statement_l923_92351

theorem incorrect_statement (p q : Prop) 
  (hp : p ↔ (2 + 2 = 5)) 
  (hq : q ↔ (3 > 2)) : 
  ¬((¬p ∧ ¬q) ∧ ¬p) := by
sorry

end NUMINAMATH_CALUDE_incorrect_statement_l923_92351


namespace NUMINAMATH_CALUDE_gcd_m_n_l923_92327

def m : ℕ := 333333333
def n : ℕ := 9999999999

theorem gcd_m_n : Nat.gcd m n = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_m_n_l923_92327


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l923_92383

/-- Given a line segment with one endpoint (-2, 5) and midpoint (1, 0),
    the sum of the coordinates of the other endpoint is -1. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  ((-2 + x) / 2 = 1 ∧ (5 + y) / 2 = 0) → 
  x + y = -1 :=
by sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l923_92383


namespace NUMINAMATH_CALUDE_professor_seating_count_l923_92309

/-- The number of chairs in a row --/
def num_chairs : ℕ := 9

/-- The number of professors --/
def num_professors : ℕ := 3

/-- The number of students --/
def num_students : ℕ := 6

/-- Represents the possible seating arrangements for professors --/
def professor_seating_arrangements : ℕ := sorry

/-- Theorem stating the number of ways professors can choose their chairs --/
theorem professor_seating_count :
  professor_seating_arrangements = 238 :=
sorry

end NUMINAMATH_CALUDE_professor_seating_count_l923_92309


namespace NUMINAMATH_CALUDE_dimas_age_l923_92362

theorem dimas_age (dima_age brother_age sister_age : ℕ) : 
  dima_age = 2 * brother_age →
  dima_age = 3 * sister_age →
  (dima_age + brother_age + sister_age) / 3 = 11 →
  dima_age = 18 := by
sorry

end NUMINAMATH_CALUDE_dimas_age_l923_92362


namespace NUMINAMATH_CALUDE_hockey_league_games_l923_92367

theorem hockey_league_games (n : ℕ) (total_games : ℕ) (h1 : n = 15) (h2 : total_games = 1050) :
  ∃ (games_per_pair : ℕ), 
    games_per_pair * (n * (n - 1) / 2) = total_games ∧ 
    games_per_pair = 10 := by
sorry

end NUMINAMATH_CALUDE_hockey_league_games_l923_92367


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l923_92358

theorem quadratic_equation_solution (k : ℝ) : 
  (8 * ((-15 - Real.sqrt 145) / 8)^2 + 15 * ((-15 - Real.sqrt 145) / 8) + k = 0) → 
  (k = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l923_92358


namespace NUMINAMATH_CALUDE_peanut_plantation_revenue_l923_92335

-- Define the plantation and region sizes
def plantation_size : ℕ × ℕ := (500, 500)
def region_a_size : ℕ × ℕ := (200, 300)
def region_b_size : ℕ × ℕ := (200, 200)
def region_c_size : ℕ × ℕ := (100, 500)

-- Define production rates (grams per square foot)
def region_a_rate : ℕ := 60
def region_b_rate : ℕ := 45
def region_c_rate : ℕ := 30

-- Define peanut butter production rate
def peanut_to_butter_ratio : ℚ := 5 / 20

-- Define monthly selling prices (dollars per kg)
def monthly_prices : List ℚ := [12, 10, 14, 8, 11]

-- Function to calculate area
def area (size : ℕ × ℕ) : ℕ := size.1 * size.2

-- Function to calculate peanut production for a region
def region_production (size : ℕ × ℕ) (rate : ℕ) : ℕ := area size * rate

-- Calculate total peanut production
def total_peanut_production : ℕ :=
  region_production region_a_size region_a_rate +
  region_production region_b_size region_b_rate +
  region_production region_c_size region_c_rate

-- Calculate peanut butter production in kg
def peanut_butter_production : ℚ :=
  (total_peanut_production : ℚ) * peanut_to_butter_ratio / 1000

-- Calculate total revenue
def total_revenue : ℚ :=
  monthly_prices.foldl (fun acc price => acc + price * peanut_butter_production) 0

-- Theorem statement
theorem peanut_plantation_revenue :
  total_revenue = 94875 := by sorry

end NUMINAMATH_CALUDE_peanut_plantation_revenue_l923_92335


namespace NUMINAMATH_CALUDE_root_equation_c_value_l923_92366

theorem root_equation_c_value :
  ∀ (c d e : ℚ),
  (∃ (x : ℝ), x = -2 + 3 * Real.sqrt 5 ∧ x^4 + c*x^3 + d*x^2 + e*x - 48 = 0) →
  c = 5 := by
sorry

end NUMINAMATH_CALUDE_root_equation_c_value_l923_92366


namespace NUMINAMATH_CALUDE_green_marbles_count_l923_92371

/-- The number of marbles in a jar with blue, red, yellow, and green marbles -/
def total_marbles : ℕ := 164

/-- The number of yellow marbles in the jar -/
def yellow_marbles : ℕ := 14

/-- The number of blue marbles in the jar -/
def blue_marbles : ℕ := total_marbles / 2

/-- The number of red marbles in the jar -/
def red_marbles : ℕ := total_marbles / 4

/-- The number of green marbles in the jar -/
def green_marbles : ℕ := total_marbles - (blue_marbles + red_marbles + yellow_marbles)

/-- Theorem stating that the number of green marbles is 27 -/
theorem green_marbles_count : green_marbles = 27 := by
  sorry

end NUMINAMATH_CALUDE_green_marbles_count_l923_92371


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l923_92347

theorem min_value_quadratic_form (x y z : ℝ) (h : 3 * x + 2 * y + z = 1) :
  ∃ (m : ℝ), m = 3 / 34 ∧ x^2 + 2 * y^2 + 3 * z^2 ≥ m ∧
  ∃ (x₀ y₀ z₀ : ℝ), 3 * x₀ + 2 * y₀ + z₀ = 1 ∧ x₀^2 + 2 * y₀^2 + 3 * z₀^2 = m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l923_92347


namespace NUMINAMATH_CALUDE_unique_balance_point_iff_m_eq_two_or_neg_one_or_one_l923_92374

/-- A function f : ℝ → ℝ has a balance point at t if f(t) = t -/
def HasBalancePoint (f : ℝ → ℝ) (t : ℝ) : Prop :=
  f t = t

/-- A function f : ℝ → ℝ has a unique balance point if there exists exactly one t such that f(t) = t -/
def HasUniqueBalancePoint (f : ℝ → ℝ) : Prop :=
  ∃! t, HasBalancePoint f t

/-- The function we're considering -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (m - 1) * x^2 - 3 * x + 2 * m

theorem unique_balance_point_iff_m_eq_two_or_neg_one_or_one :
  ∀ m : ℝ, HasUniqueBalancePoint (f m) ↔ m = 2 ∨ m = -1 ∨ m = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_balance_point_iff_m_eq_two_or_neg_one_or_one_l923_92374


namespace NUMINAMATH_CALUDE_range_of_a_given_p_necessary_not_sufficient_for_q_l923_92329

theorem range_of_a_given_p_necessary_not_sufficient_for_q :
  ∀ a : ℝ,
  (∀ x : ℝ, x^2 ≤ 5*x - 4 → x^2 - (a+2)*x + 2*a ≤ 0) ∧
  (∃ x : ℝ, x^2 - (a+2)*x + 2*a ≤ 0 ∧ x^2 > 5*x - 4) →
  1 ≤ a ∧ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_given_p_necessary_not_sufficient_for_q_l923_92329


namespace NUMINAMATH_CALUDE_unique_prime_in_form_l923_92319

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def form_number (B A : ℕ) : ℕ := 210000 + B * 100 + A

theorem unique_prime_in_form :
  ∃! B : ℕ, B < 10 ∧ ∃ A : ℕ, A < 10 ∧ is_prime (form_number B A) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_in_form_l923_92319


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l923_92336

/-- A linear function y = ax + b where y increases as x increases and ab < 0 -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  increasing : a > 0
  product_negative : a * b < 0

/-- The point P(a,b) -/
def point (f : LinearFunction) : ℝ × ℝ := (f.a, f.b)

/-- A point (x,y) lies in the fourth quadrant if x > 0 and y < 0 -/
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant (f : LinearFunction) :
  in_fourth_quadrant (point f) := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l923_92336


namespace NUMINAMATH_CALUDE_kindergarten_tissues_l923_92354

/-- The number of tissues in each mini tissue box -/
def tissues_per_box : ℕ := 40

/-- The number of students in each kindergartner group -/
def group_sizes : List ℕ := [9, 10, 11]

/-- The total number of tissues brought by all kindergartner groups -/
def total_tissues : ℕ := (group_sizes.sum) * tissues_per_box

theorem kindergarten_tissues :
  total_tissues = 1200 :=
by sorry

end NUMINAMATH_CALUDE_kindergarten_tissues_l923_92354


namespace NUMINAMATH_CALUDE_julians_boy_friends_percentage_l923_92389

theorem julians_boy_friends_percentage 
  (julian_total_friends : ℕ)
  (julian_girls_percentage : ℚ)
  (boyd_total_friends : ℕ)
  (boyd_boys_percentage : ℚ)
  (h1 : julian_total_friends = 80)
  (h2 : julian_girls_percentage = 40/100)
  (h3 : boyd_total_friends = 100)
  (h4 : boyd_boys_percentage = 36/100)
  (h5 : (boyd_total_friends : ℚ) * (1 - boyd_boys_percentage) = 2 * (julian_total_friends : ℚ) * julian_girls_percentage) :
  (julian_total_friends : ℚ) * (1 - julian_girls_percentage) / julian_total_friends = 60/100 := by
  sorry

end NUMINAMATH_CALUDE_julians_boy_friends_percentage_l923_92389


namespace NUMINAMATH_CALUDE_inequality_theorem_l923_92384

theorem inequality_theorem (a b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ 0) :
  (1 + a)^2 / (1 + b) ≤ 1 + a^2 / b ↔
    ((a ≠ b ∧ (b < -1 ∨ b > 0)) ∨ (a = b ∧ a ≠ -1 ∧ a ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l923_92384


namespace NUMINAMATH_CALUDE_age_difference_proof_l923_92376

theorem age_difference_proof :
  ∀ (a b : ℕ),
    a + b = 2 →
    (10 * a + b) + (10 * b + a) = 22 →
    (10 * a + b + 7) = 3 * (10 * b + a + 7) →
    (10 * a + b) - (10 * b + a) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l923_92376


namespace NUMINAMATH_CALUDE_kiddie_scoop_cost_l923_92340

/-- The cost of ice cream scoops for the Martin family --/
def ice_cream_cost (kiddie_scoop : ℕ) : Prop :=
  let regular_scoop : ℕ := 4
  let double_scoop : ℕ := 6
  let total_cost : ℕ := 32
  let num_regular : ℕ := 2  -- Mr. and Mrs. Martin
  let num_kiddie : ℕ := 2   -- Their two children
  let num_double : ℕ := 3   -- Their three teenage children
  
  total_cost = num_regular * regular_scoop + num_kiddie * kiddie_scoop + num_double * double_scoop

theorem kiddie_scoop_cost : ice_cream_cost 3 := by
  sorry

end NUMINAMATH_CALUDE_kiddie_scoop_cost_l923_92340


namespace NUMINAMATH_CALUDE_conference_handshakes_l923_92365

/-- Calculates the maximum number of handshakes in a conference with given constraints -/
def max_handshakes (total : ℕ) (committee : ℕ) (red_badges : ℕ) : ℕ :=
  let participants := total - committee - red_badges
  participants * (participants - 1) / 2

/-- Theorem stating the maximum number of handshakes for the given conference -/
theorem conference_handshakes :
  max_handshakes 50 10 5 = 595 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l923_92365


namespace NUMINAMATH_CALUDE_total_cloud_count_l923_92306

def cloud_count (carson_count : ℕ) (brother_multiplier : ℕ) (sister_divisor : ℕ) : ℕ :=
  carson_count + (carson_count * brother_multiplier) + (carson_count / sister_divisor)

theorem total_cloud_count :
  cloud_count 12 5 2 = 78 :=
by sorry

end NUMINAMATH_CALUDE_total_cloud_count_l923_92306


namespace NUMINAMATH_CALUDE_inequality_solution_set_l923_92343

/-- Given an inequality tx^2 - 6x + t^2 < 0 with solution set (-∞,a)∪(1,+∞), prove that a = -3 -/
theorem inequality_solution_set (t : ℝ) (a : ℝ) :
  (∀ x : ℝ, (t * x^2 - 6 * x + t^2 < 0) ↔ (x < a ∨ x > 1)) →
  a = -3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l923_92343


namespace NUMINAMATH_CALUDE_follower_point_coords_follower_on_axis_follower_distance_l923_92330

-- Define a-level follower point
def a_level_follower (a : ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (x + a * y, a * x + y)

-- Statement 1
theorem follower_point_coords : a_level_follower 3 (-3, 5) = (12, -4) := by sorry

-- Statement 2
theorem follower_on_axis (c : ℝ) : 
  (∃ x y, a_level_follower (-3) (c, 2*c + 2) = (x, y) ∧ (x = 0 ∨ y = 0)) →
  a_level_follower (-3) (c, 2*c + 2) = (-16, 0) ∨ 
  a_level_follower (-3) (c, 2*c + 2) = (0, 16/5) := by sorry

-- Statement 3
theorem follower_distance (x : ℝ) (a : ℝ) :
  x > 0 →
  let P : ℝ × ℝ := (x, 0)
  let P3 := a_level_follower a P
  let PP3_length := Real.sqrt ((P3.1 - P.1)^2 + (P3.2 - P.2)^2)
  let OP_length := Real.sqrt (P.1^2 + P.2^2)
  PP3_length = 2 * OP_length →
  a = 2 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_follower_point_coords_follower_on_axis_follower_distance_l923_92330


namespace NUMINAMATH_CALUDE_max_three_digit_with_remainders_l923_92332

theorem max_three_digit_with_remainders :
  ∀ N : ℕ,
  (100 ≤ N ∧ N ≤ 999) →
  (N % 3 = 1) →
  (N % 7 = 3) →
  (N % 11 = 8) →
  (∀ M : ℕ, (100 ≤ M ∧ M ≤ 999) → (M % 3 = 1) → (M % 7 = 3) → (M % 11 = 8) → M ≤ N) →
  N = 976 := by
sorry

end NUMINAMATH_CALUDE_max_three_digit_with_remainders_l923_92332


namespace NUMINAMATH_CALUDE_domain_of_f_l923_92359

def f (x : ℝ) : ℝ := (x + 1) ^ 0

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -1} :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l923_92359


namespace NUMINAMATH_CALUDE_remainder_91_power_91_mod_100_l923_92315

/-- The remainder when 91^91 is divided by 100 is 91. -/
theorem remainder_91_power_91_mod_100 : 91^91 % 100 = 91 := by
  sorry

end NUMINAMATH_CALUDE_remainder_91_power_91_mod_100_l923_92315


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l923_92331

theorem gcd_lcm_product (a b : Nat) (h1 : a = 180) (h2 : b = 250) :
  (Nat.gcd a b) * (Nat.lcm a b) = 45000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l923_92331


namespace NUMINAMATH_CALUDE_george_oranges_l923_92322

theorem george_oranges (george_oranges : ℕ) (george_apples : ℕ) (amelia_oranges : ℕ) (amelia_apples : ℕ) : 
  george_apples = amelia_apples + 5 →
  amelia_oranges = george_oranges - 18 →
  amelia_apples = 15 →
  george_oranges + george_apples + amelia_oranges + amelia_apples = 107 →
  george_oranges = 45 := by
sorry

end NUMINAMATH_CALUDE_george_oranges_l923_92322


namespace NUMINAMATH_CALUDE_right_triangle_base_length_l923_92310

theorem right_triangle_base_length 
  (height : ℝ) 
  (area : ℝ) 
  (hypotenuse : ℝ) 
  (h1 : height = 8) 
  (h2 : area = 24) 
  (h3 : hypotenuse = 10) : 
  ∃ (base : ℝ), base = 6 ∧ area = (1/2) * base * height ∧ hypotenuse^2 = height^2 + base^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_base_length_l923_92310


namespace NUMINAMATH_CALUDE_f_simplification_f_value_in_second_quadrant_l923_92334

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 5 * Real.pi / 2) * Real.cos (3 * Real.pi / 2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (Real.pi - α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by sorry

theorem f_value_in_second_quadrant (α : Real) 
  (h1 : Real.cos (α + 3 * Real.pi / 2) = 1/5) 
  (h2 : 0 < α ∧ α < Real.pi) : 
  f α = 2 * Real.sqrt 6 / 5 := by sorry

end NUMINAMATH_CALUDE_f_simplification_f_value_in_second_quadrant_l923_92334


namespace NUMINAMATH_CALUDE_arrangement_counts_l923_92328

/-- Represents the number of people in the row -/
def n : ℕ := 5

/-- Calculates the factorial of a natural number -/
def factorial (k : ℕ) : ℕ :=
  match k with
  | 0 => 1
  | k + 1 => (k + 1) * factorial k

/-- The number of arrangements with Person A at the head -/
def arrangements_A_at_head : ℕ := factorial (n - 1)

/-- The number of arrangements with Person A and Person B adjacent -/
def arrangements_A_B_adjacent : ℕ := factorial (n - 1) * 2

/-- The number of arrangements with Person A not at the head and Person B not at the end -/
def arrangements_A_not_head_B_not_end : ℕ := (n - 1) * (n - 2) * factorial (n - 2)

/-- The number of arrangements with Person A to the left of and taller than Person B, and not adjacent -/
def arrangements_A_left_taller_not_adjacent : ℕ := 3 * factorial (n - 2)

theorem arrangement_counts :
  arrangements_A_at_head = 24 ∧
  arrangements_A_B_adjacent = 48 ∧
  arrangements_A_not_head_B_not_end = 72 ∧
  arrangements_A_left_taller_not_adjacent = 18 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_counts_l923_92328


namespace NUMINAMATH_CALUDE_simplify_expression_l923_92377

theorem simplify_expression (x y : ℝ) : x^5 * x^3 * y^2 * y^4 = x^8 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l923_92377


namespace NUMINAMATH_CALUDE_gold_cube_profit_calculation_l923_92312

/-- Calculates the profit from selling a gold cube -/
def goldCubeProfit (side : ℝ) (density : ℝ) (purchasePrice : ℝ) (markupFactor : ℝ) : ℝ :=
  let volume := side^3
  let mass := volume * density
  let cost := mass * purchasePrice
  let sellingPrice := cost * markupFactor
  sellingPrice - cost

/-- Theorem stating the profit from selling a specific gold cube -/
theorem gold_cube_profit_calculation :
  goldCubeProfit 6 19 60 1.5 = 123120 := by sorry

end NUMINAMATH_CALUDE_gold_cube_profit_calculation_l923_92312


namespace NUMINAMATH_CALUDE_zachary_crunch_pushup_difference_l923_92318

/-- Given information about Zachary's and David's exercises, prove that Zachary did 12 more crunches than push-ups. -/
theorem zachary_crunch_pushup_difference :
  ∀ (zachary_pushups zachary_crunches david_pushups david_crunches : ℕ),
    zachary_pushups = 46 →
    zachary_crunches = 58 →
    david_pushups = zachary_pushups + 38 →
    david_crunches = zachary_crunches - 62 →
    zachary_crunches - zachary_pushups = 12 :=
by sorry

end NUMINAMATH_CALUDE_zachary_crunch_pushup_difference_l923_92318


namespace NUMINAMATH_CALUDE_min_production_cost_l923_92356

/-- Raw material requirements for products A and B --/
structure RawMaterial where
  a : ℕ  -- kg of material A required
  b : ℕ  -- kg of material B required

/-- Available raw materials and production constraints --/
structure ProductionConstraints where
  total_units : ℕ        -- Total units to be produced
  available_a : ℕ        -- Available kg of material A
  available_b : ℕ        -- Available kg of material B
  product_a : RawMaterial  -- Raw material requirements for product A
  product_b : RawMaterial  -- Raw material requirements for product B

/-- Cost information for products --/
structure CostInfo where
  cost_a : ℕ  -- Cost per unit of product A
  cost_b : ℕ  -- Cost per unit of product B

/-- Main theorem stating the minimum production cost --/
theorem min_production_cost 
  (constraints : ProductionConstraints)
  (costs : CostInfo)
  (h_constraints : constraints.total_units = 50 ∧ 
                   constraints.available_a = 360 ∧ 
                   constraints.available_b = 290 ∧
                   constraints.product_a = ⟨9, 4⟩ ∧
                   constraints.product_b = ⟨3, 10⟩)
  (h_costs : costs.cost_a = 70 ∧ costs.cost_b = 90) :
  ∃ (x : ℕ), x = 32 ∧ 
    (constraints.total_units - x) = 18 ∧
    costs.cost_a * x + costs.cost_b * (constraints.total_units - x) = 3860 :=
sorry

end NUMINAMATH_CALUDE_min_production_cost_l923_92356


namespace NUMINAMATH_CALUDE_max_value_fraction_l923_92350

theorem max_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + y * z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 2 / 2 ∧
  ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    (x' * y' + y' * z') / (x'^2 + y'^2 + z'^2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_l923_92350


namespace NUMINAMATH_CALUDE_candy_bowl_problem_l923_92307

theorem candy_bowl_problem (talitha_pieces solomon_pieces remaining_pieces : ℕ) 
  (h1 : talitha_pieces = 108)
  (h2 : solomon_pieces = 153)
  (h3 : remaining_pieces = 88) :
  talitha_pieces + solomon_pieces + remaining_pieces = 349 := by
  sorry

end NUMINAMATH_CALUDE_candy_bowl_problem_l923_92307


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_l923_92341

theorem least_positive_integer_multiple (x : ℕ) : x = 42 ↔ 
  (x > 0 ∧ ∀ y : ℕ, y > 0 → y < x → ¬(∃ k : ℤ, (2 * y + 45)^2 = 43 * k)) ∧
  (∃ k : ℤ, (2 * x + 45)^2 = 43 * k) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_l923_92341


namespace NUMINAMATH_CALUDE_optimal_landing_point_l923_92392

/-- The optimal landing point for a messenger traveling from a boat to a camp on shore -/
theorem optimal_landing_point (boat_distance : ℝ) (camp_distance : ℝ) 
  (row_speed : ℝ) (walk_speed : ℝ) : ℝ :=
let landing_point := 12
let total_time (x : ℝ) := 
  (Real.sqrt (boat_distance^2 + x^2)) / row_speed + (camp_distance - x) / walk_speed
have h1 : boat_distance = 9 := by sorry
have h2 : camp_distance = 15 := by sorry
have h3 : row_speed = 4 := by sorry
have h4 : walk_speed = 5 := by sorry
have h5 : ∀ x, total_time landing_point ≤ total_time x := by sorry
landing_point

#check optimal_landing_point

end NUMINAMATH_CALUDE_optimal_landing_point_l923_92392


namespace NUMINAMATH_CALUDE_squared_plus_greater_than_self_l923_92355

-- Define a monotonically increasing function on R
def monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem squared_plus_greater_than_self
  (f : ℝ → ℝ) (h_monotone : monotone_increasing f) (t : ℝ) (h_t : t ≠ 0) :
  f (t^2 + t) > f t :=
sorry

end NUMINAMATH_CALUDE_squared_plus_greater_than_self_l923_92355


namespace NUMINAMATH_CALUDE_total_students_count_l923_92378

/-- The number of students wishing to go on a scavenger hunting trip -/
def scavenger_hunting : ℕ := 4000

/-- The number of students wishing to go on a skiing trip -/
def skiing : ℕ := 2 * scavenger_hunting

/-- The number of students wishing to go on a camping trip -/
def camping : ℕ := skiing + (skiing * 15 / 100)

/-- The total number of students wishing to go on any trip -/
def total_students : ℕ := scavenger_hunting + skiing + camping

theorem total_students_count : total_students = 21200 := by
  sorry

end NUMINAMATH_CALUDE_total_students_count_l923_92378


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_4_l923_92316

theorem greatest_integer_with_gcf_4 : ∃ n : ℕ, 
  n < 200 ∧ 
  Nat.gcd n 24 = 4 ∧ 
  ∀ m : ℕ, m < 200 → Nat.gcd m 24 = 4 → m ≤ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_4_l923_92316


namespace NUMINAMATH_CALUDE_polar_to_cartesian_intersecting_lines_l923_92394

/-- The polar coordinate equation ρ(cos²θ - sin²θ) = 0 represents two intersecting lines -/
theorem polar_to_cartesian_intersecting_lines :
  ∃ (x y : ℝ → ℝ), 
    (∀ θ : ℝ, x θ^2 = y θ^2) ∧ 
    (∀ θ : ℝ, x θ = y θ ∨ x θ = -y θ) ∧
    (∀ ρ θ : ℝ, ρ * (Real.cos θ^2 - Real.sin θ^2) = 0 → 
      x θ = ρ * Real.cos θ ∧ y θ = ρ * Real.sin θ) :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_intersecting_lines_l923_92394


namespace NUMINAMATH_CALUDE_tangent_line_equation_l923_92379

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Define the slope of the line parallel to 3x + y = 0
def k : ℝ := -3

-- Define the point of tangency
def x₀ : ℝ := 1
def y₀ : ℝ := f x₀

-- State the theorem
theorem tangent_line_equation :
  ∃ (x y : ℝ), 3*x + y - 1 = 0 ∧
  y - y₀ = k * (x - x₀) ∧
  f' x₀ = k ∧
  y₀ = f x₀ := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l923_92379


namespace NUMINAMATH_CALUDE_cirrus_count_l923_92364

/-- The number of cumulonimbus clouds -/
def cumulonimbus : ℕ := 3

/-- The number of cumulus clouds -/
def cumulus : ℕ := 12 * cumulonimbus

/-- The number of cirrus clouds -/
def cirrus : ℕ := 4 * cumulus

/-- The number of altostratus clouds -/
def altostratus : ℕ := 6 * (cirrus + cumulus)

/-- Theorem stating that the number of cirrus clouds is 144 -/
theorem cirrus_count : cirrus = 144 := by sorry

end NUMINAMATH_CALUDE_cirrus_count_l923_92364


namespace NUMINAMATH_CALUDE_eighth_number_in_set_l923_92372

theorem eighth_number_in_set (known_numbers : List ℕ) (average : ℚ) : 
  known_numbers = [1, 2, 4, 5, 6, 9, 9, 12] ∧ 
  average = 7 ∧
  (List.sum known_numbers + 12) / 9 = average →
  ∃ x : ℕ, x = 3 ∧ x ∈ (known_numbers ++ [12]) :=
by sorry

end NUMINAMATH_CALUDE_eighth_number_in_set_l923_92372


namespace NUMINAMATH_CALUDE_maze_side_length_l923_92353

/-- Represents a maze on a square grid -/
structure Maze where
  sideLength : ℕ
  wallLength : ℕ

/-- Checks if the maze satisfies the unique path property -/
def hasUniquePaths (m : Maze) : Prop :=
  m.sideLength ^ 2 = 2 * m.sideLength * (m.sideLength - 1) - m.wallLength + 1

theorem maze_side_length (m : Maze) :
  m.wallLength = 400 → hasUniquePaths m → m.sideLength = 21 := by
  sorry

end NUMINAMATH_CALUDE_maze_side_length_l923_92353
