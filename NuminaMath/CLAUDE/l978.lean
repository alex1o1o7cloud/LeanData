import Mathlib

namespace NUMINAMATH_CALUDE_module_stock_worth_l978_97881

/-- Calculates the total worth of a stock of modules -/
theorem module_stock_worth (total_modules : ℕ) (cheap_modules : ℕ) (expensive_cost : ℚ) (cheap_cost : ℚ) 
  (h1 : total_modules = 22)
  (h2 : cheap_modules = 21)
  (h3 : expensive_cost = 10)
  (h4 : cheap_cost = 5/2)
  : (total_modules - cheap_modules) * expensive_cost + cheap_modules * cheap_cost = 125/2 := by
  sorry

#eval (22 - 21) * 10 + 21 * (5/2)  -- This should output 62.5

end NUMINAMATH_CALUDE_module_stock_worth_l978_97881


namespace NUMINAMATH_CALUDE_polygon_sides_with_45_degree_exterior_angles_l978_97889

theorem polygon_sides_with_45_degree_exterior_angles :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    exterior_angle = 45 →
    (n : ℝ) * exterior_angle = 360 →
    n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_with_45_degree_exterior_angles_l978_97889


namespace NUMINAMATH_CALUDE_distance_sum_theorem_l978_97817

theorem distance_sum_theorem (x z w : ℝ) 
  (hx : x = -1)
  (hz : z = 3.7)
  (hw : w = 9.3) :
  |z - x| + |w - x| = 15 := by
sorry

end NUMINAMATH_CALUDE_distance_sum_theorem_l978_97817


namespace NUMINAMATH_CALUDE_evaluate_expression_l978_97861

theorem evaluate_expression : -(18 / 3 * 7^2 - 80 + 4 * 7) = -242 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l978_97861


namespace NUMINAMATH_CALUDE_abs_neg_three_squared_plus_four_l978_97840

theorem abs_neg_three_squared_plus_four : |-3^2 + 4| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_squared_plus_four_l978_97840


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l978_97871

theorem arithmetic_calculation : 1435 + 180 / 60 * 3 - 435 = 1009 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l978_97871


namespace NUMINAMATH_CALUDE_hayes_laundry_loads_l978_97863

/-- The number of detergent pods in a pack -/
def pods_per_pack : ℕ := 39

/-- The number of packs Hayes needs for a full year -/
def packs_per_year : ℕ := 4

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The number of loads of laundry Hayes does in a week -/
def loads_per_week : ℕ := (pods_per_pack * packs_per_year) / weeks_per_year

theorem hayes_laundry_loads : loads_per_week = 3 := by sorry

end NUMINAMATH_CALUDE_hayes_laundry_loads_l978_97863


namespace NUMINAMATH_CALUDE_triangle_perimeter_lower_bound_l978_97869

theorem triangle_perimeter_lower_bound 
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (h : ℝ) -- Height on side BC
  (ha : a = 1) -- Side a equals 1
  (hh : h = Real.tan A) -- Height equals tan A
  (hA : 0 < A ∧ A < Real.pi / 2) -- A is in the range (0, π/2)
  (hS : (1/2) * a * h = (1/2) * b * c * Real.sin A) -- Area formula
  (hC : c^2 = a^2 + b^2 - 2*a*b*Real.cos C) -- Cosine rule
  : a + b + c > Real.sqrt 5 + 1 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_lower_bound_l978_97869


namespace NUMINAMATH_CALUDE_cloth_sold_calculation_l978_97895

/-- The number of meters of cloth sold by a trader -/
def meters_of_cloth : ℕ := 40

/-- The profit per meter of cloth in rupees -/
def profit_per_meter : ℕ := 30

/-- The total profit earned by the trader in rupees -/
def total_profit : ℕ := 1200

/-- Theorem stating that the number of meters of cloth sold is 40 -/
theorem cloth_sold_calculation :
  meters_of_cloth * profit_per_meter = total_profit :=
by sorry

end NUMINAMATH_CALUDE_cloth_sold_calculation_l978_97895


namespace NUMINAMATH_CALUDE_sum_three_numbers_l978_97873

theorem sum_three_numbers (a b c N : ℝ) : 
  a + b + c = 60 ∧ 
  a - 7 = N ∧ 
  b + 7 = N ∧ 
  7 * c = N → 
  N = 28 := by
sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l978_97873


namespace NUMINAMATH_CALUDE_prime_sum_2019_power_l978_97833

theorem prime_sum_2019_power (p q : ℕ) : 
  Prime p → Prime q → p + q = 2019 → (p - 1)^(q - 1) = 1 ∨ (p - 1)^(q - 1) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_2019_power_l978_97833


namespace NUMINAMATH_CALUDE_david_started_with_at_least_six_iphones_l978_97853

/-- Represents the number of cell phones in various categories -/
structure CellPhoneInventory where
  samsung_end : ℕ
  iphone_end : ℕ
  samsung_damaged : ℕ
  iphone_defective : ℕ
  total_sold : ℕ

/-- Given the end-of-day inventory and sales data, proves that David started with at least 6 iPhones -/
theorem david_started_with_at_least_six_iphones 
  (inventory : CellPhoneInventory)
  (h1 : inventory.samsung_end = 10)
  (h2 : inventory.iphone_end = 5)
  (h3 : inventory.samsung_damaged = 2)
  (h4 : inventory.iphone_defective = 1)
  (h5 : inventory.total_sold = 4) :
  ∃ (initial_iphones : ℕ), initial_iphones ≥ 6 ∧ 
    initial_iphones ≥ inventory.iphone_end + inventory.iphone_defective :=
by sorry

end NUMINAMATH_CALUDE_david_started_with_at_least_six_iphones_l978_97853


namespace NUMINAMATH_CALUDE_kateDisprovesPeter_l978_97838

/-- Represents a card with a character on one side and a natural number on the other -/
structure Card where
  letter : Char
  number : Nat

/-- Checks if a given character is a vowel -/
def isVowel (c : Char) : Bool :=
  c ∈ ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']

/-- Checks if a given natural number is even -/
def isEven (n : Nat) : Bool :=
  n % 2 = 0

/-- Represents Peter's statement about vowels and even numbers -/
def petersStatement (c : Card) : Bool :=
  isVowel c.letter → isEven c.number

/-- The set of cards on the table -/
def cardsOnTable : List Card := [
  ⟨'A', 0⟩,  -- Placeholder number
  ⟨'B', 0⟩,  -- Placeholder number
  ⟨'C', 1⟩,  -- Assuming 'C' for the third card
  ⟨'D', 7⟩,  -- The fourth card we know about
  ⟨'U', 0⟩   -- Placeholder number
]

theorem kateDisprovesPeter :
  ∃ (c : Card), c ∈ cardsOnTable ∧ ¬(petersStatement c) ∧ c.number = 7 := by
  sorry

#check kateDisprovesPeter

end NUMINAMATH_CALUDE_kateDisprovesPeter_l978_97838


namespace NUMINAMATH_CALUDE_stone_splitting_game_winner_l978_97846

/-- The stone-splitting game -/
def StoneSplittingGame (n : ℕ) : Prop :=
  ∃ (winner : Bool),
    (winner = true → n.Prime ∨ ∃ k, n = 2^k) ∧
    (winner = false → ¬(n.Prime ∨ ∃ k, n = 2^k))

/-- Theorem: Characterization of winning conditions in the stone-splitting game -/
theorem stone_splitting_game_winner (n : ℕ) :
  StoneSplittingGame n ↔ (n.Prime ∨ ∃ k, n = 2^k) := by sorry

end NUMINAMATH_CALUDE_stone_splitting_game_winner_l978_97846


namespace NUMINAMATH_CALUDE_paving_stone_width_l978_97886

/-- The width of a paving stone given the dimensions of a rectangular courtyard and the number of stones required to pave it. -/
theorem paving_stone_width 
  (courtyard_length : ℝ) 
  (courtyard_width : ℝ) 
  (num_stones : ℕ) 
  (stone_length : ℝ) 
  (h1 : courtyard_length = 50) 
  (h2 : courtyard_width = 16.5) 
  (h3 : num_stones = 165) 
  (h4 : stone_length = 2.5) : 
  ∃ (stone_width : ℝ), stone_width = 2 ∧ 
    courtyard_length * courtyard_width = ↑num_stones * stone_length * stone_width := by
  sorry

end NUMINAMATH_CALUDE_paving_stone_width_l978_97886


namespace NUMINAMATH_CALUDE_first_reduction_percentage_l978_97862

theorem first_reduction_percentage (x : ℝ) :
  (1 - x / 100) * (1 - 50 / 100) = 1 - 62.5 / 100 →
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_first_reduction_percentage_l978_97862


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_two_ninety_eight_satisfies_conditions_ninety_eight_is_greatest_l978_97876

theorem greatest_integer_with_gcf_two (n : ℕ) : n < 100 → Nat.gcd n 12 = 2 → n ≤ 98 := by
  sorry

theorem ninety_eight_satisfies_conditions : 
  98 < 100 ∧ Nat.gcd 98 12 = 2 := by
  sorry

theorem ninety_eight_is_greatest : 
  ∀ (m : ℕ), m < 100 → Nat.gcd m 12 = 2 → m ≤ 98 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_two_ninety_eight_satisfies_conditions_ninety_eight_is_greatest_l978_97876


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l978_97842

/-- Proves that a rectangular plot with length to breadth ratio 7:5 and perimeter 288 meters has an area of 5040 square meters. -/
theorem rectangular_plot_area (length width : ℝ) : 
  (length / width = 7 / 5) →  -- ratio condition
  (2 * (length + width) = 288) →  -- perimeter condition
  (length * width = 5040) :=  -- area to prove
by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l978_97842


namespace NUMINAMATH_CALUDE_second_expression_proof_l978_97872

theorem second_expression_proof (a x : ℝ) (h1 : ((2 * a + 16) + x) / 2 = 74) (h2 : a = 28) : x = 76 := by
  sorry

end NUMINAMATH_CALUDE_second_expression_proof_l978_97872


namespace NUMINAMATH_CALUDE_intersection_implies_a_equals_three_l978_97844

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {3, 4, 2*a - 4}
def B (a : ℝ) : Set ℝ := {a}

-- State the theorem
theorem intersection_implies_a_equals_three (a : ℝ) :
  (A a ∩ B a).Nonempty → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_equals_three_l978_97844


namespace NUMINAMATH_CALUDE_expression_simplification_l978_97851

theorem expression_simplification (y : ℝ) : 
  3*y - 7*y^2 + 15 - (6 - 5*y + 7*y^2) = -14*y^2 + 8*y + 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l978_97851


namespace NUMINAMATH_CALUDE_arithmetic_progression_zero_term_l978_97837

/-- An arithmetic progression with a_{2n} / a_{2m} = -1 has a zero term at position n+m -/
theorem arithmetic_progression_zero_term 
  (a : ℕ → ℝ) 
  (h_arith : ∃ (d : ℝ), ∀ (k : ℕ), a (k + 1) = a k + d) 
  (n m : ℕ) 
  (h_ratio : a (2 * n) / a (2 * m) = -1) :
  ∃ (k : ℕ), k = n + m ∧ a k = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_zero_term_l978_97837


namespace NUMINAMATH_CALUDE_solution_values_solution_set_when_a_negative_l978_97892

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - a

-- Define the solution set condition
def hasSolutionSet (a b : ℝ) : Prop :=
  ∀ x, f a x < b ↔ x < -1 ∨ x > 3

-- Theorem statement
theorem solution_values (a b : ℝ) (h : hasSolutionSet a b) : a = -1/2 ∧ b = -1 := by
  sorry

-- Additional theorem for part 2
theorem solution_set_when_a_negative (a : ℝ) (h : a < 0) :
  (∀ x, f a x > 1 ↔ 
    (a < -1/2 ∧ -((a+1)/a) < x ∧ x < 1) ∨
    (a = -1/2 ∧ False) ∨
    (-1/2 < a ∧ a < 0 ∧ 1 < x ∧ x < -((a+1)/a))) := by
  sorry

end NUMINAMATH_CALUDE_solution_values_solution_set_when_a_negative_l978_97892


namespace NUMINAMATH_CALUDE_circle_radius_proof_l978_97864

theorem circle_radius_proof (r p q : ℕ) (m n : ℕ+) :
  -- r is an odd integer
  Odd r →
  -- p and q are prime numbers
  Nat.Prime p →
  Nat.Prime q →
  -- (p^m, q^n) is on the circle with radius r
  p^(m:ℕ) * p^(m:ℕ) + q^(n:ℕ) * q^(n:ℕ) = r * r →
  -- The radius r is equal to 5
  r = 5 := by sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l978_97864


namespace NUMINAMATH_CALUDE_difference_zero_iff_k_ge_five_l978_97818

/-- Definition of the sequence u_n -/
def u (n : ℕ) : ℕ := n^4 + 2*n^2

/-- Definition of the first difference operator -/
def Δ₁ (f : ℕ → ℕ) (n : ℕ) : ℕ := f (n + 1) - f n

/-- Definition of the k-th difference operator -/
def Δ (k : ℕ) (f : ℕ → ℕ) : ℕ → ℕ :=
  match k with
  | 0 => f
  | k + 1 => Δ₁ (Δ k f)

/-- Theorem: The k-th difference of u_n is zero for all n if and only if k ≥ 5 -/
theorem difference_zero_iff_k_ge_five :
  ∀ k : ℕ, (∀ n : ℕ, Δ k u n = 0) ↔ k ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_difference_zero_iff_k_ge_five_l978_97818


namespace NUMINAMATH_CALUDE_least_common_addition_primes_l978_97882

theorem least_common_addition_primes (x y : ℕ) : 
  Nat.Prime x → Nat.Prime y → x < y → x + y = 36 → 4 * x + y = 51 := by
  sorry

end NUMINAMATH_CALUDE_least_common_addition_primes_l978_97882


namespace NUMINAMATH_CALUDE_drill_bits_total_cost_l978_97888

/-- Calculates the total amount paid for drill bits including tax -/
theorem drill_bits_total_cost (num_sets : ℕ) (cost_per_set : ℚ) (tax_rate : ℚ) : 
  num_sets = 5 → cost_per_set = 6 → tax_rate = (1/10) → 
  num_sets * cost_per_set * (1 + tax_rate) = 33 := by
  sorry

end NUMINAMATH_CALUDE_drill_bits_total_cost_l978_97888


namespace NUMINAMATH_CALUDE_sum_equals_200_l978_97855

theorem sum_equals_200 : 148 + 32 + 18 + 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_200_l978_97855


namespace NUMINAMATH_CALUDE_min_sum_of_primes_l978_97832

theorem min_sum_of_primes (p q r s : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
  (30 ∣ p * q - r * s) →
  54 ≤ p + q + r + s :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_primes_l978_97832


namespace NUMINAMATH_CALUDE_complex_equation_solution_l978_97803

theorem complex_equation_solution (z : ℂ) : (2 * I) / z = 1 - I → z = -1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l978_97803


namespace NUMINAMATH_CALUDE_digit_append_theorem_l978_97899

theorem digit_append_theorem (n : ℕ) : 
  (∃ d : ℕ, d ≤ 9 ∧ 10 * n + d = 13 * n) ↔ (n = 1 ∨ n = 2 ∨ n = 3) :=
sorry

end NUMINAMATH_CALUDE_digit_append_theorem_l978_97899


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_to_fifth_l978_97820

theorem imaginary_part_of_one_plus_i_to_fifth (i : ℂ) : i * i = -1 → Complex.im ((1 + i) ^ 5) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_to_fifth_l978_97820


namespace NUMINAMATH_CALUDE_distance_to_point_l978_97854

theorem distance_to_point : Real.sqrt (8^2 + (-15)^2) = 17 := by sorry

end NUMINAMATH_CALUDE_distance_to_point_l978_97854


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l978_97858

theorem min_sum_of_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : x₁ + 3 * x₂ + 5 * x₃ = 100) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 2000 / 7 ∧ 
  ∃ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ 
    y₁ + 3 * y₂ + 5 * y₃ = 100 ∧ 
    y₁^2 + y₂^2 + y₃^2 = 2000 / 7 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l978_97858


namespace NUMINAMATH_CALUDE_fifth_group_size_l978_97893

/-- Represents a choir split into groups -/
structure Choir :=
  (total_members : ℕ)
  (group1 : ℕ)
  (group2 : ℕ)
  (group3 : ℕ)
  (group4 : ℕ)
  (group5 : ℕ)

/-- The choir satisfies the given conditions -/
def choir_conditions (c : Choir) : Prop :=
  c.total_members = 150 ∧
  c.group1 = 18 ∧
  c.group2 = 29 ∧
  c.group3 = 34 ∧
  c.group4 = 23 ∧
  c.total_members = c.group1 + c.group2 + c.group3 + c.group4 + c.group5

/-- Theorem: The fifth group has 46 members -/
theorem fifth_group_size (c : Choir) (h : choir_conditions c) : c.group5 = 46 := by
  sorry

end NUMINAMATH_CALUDE_fifth_group_size_l978_97893


namespace NUMINAMATH_CALUDE_power_fraction_equality_l978_97860

theorem power_fraction_equality : (2^2015 + 2^2011) / (2^2015 - 2^2011) = 17/15 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l978_97860


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l978_97848

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0, 
    and the length of the conjugate axis is twice that of the transverse axis (b = 2a),
    prove that its eccentricity is √5. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_axis : b = 2 * a) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l978_97848


namespace NUMINAMATH_CALUDE_new_average_after_adding_l978_97834

theorem new_average_after_adding (n : ℕ) (original_avg : ℝ) (added_value : ℝ) :
  n > 0 →
  n = 15 →
  original_avg = 40 →
  added_value = 14 →
  (n * original_avg + n * added_value) / n = 54 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_adding_l978_97834


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l978_97843

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n ∧ a n > 0

/-- Three terms form an arithmetic sequence -/
def ArithmeticSequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticSequence (3 * a 1) ((1 / 2) * a 3) (2 * a 2) →
  (a 11 + a 13) / (a 8 + a 10) = 27 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l978_97843


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l978_97812

/-- A quadratic function satisfying certain properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (∃ x₀, ∀ x, f x ≥ f x₀) ∧
  (∃ x₀, f x₀ = -1) ∧
  (f 0 = 0)

theorem quadratic_function_properties (f : ℝ → ℝ) (m : ℝ) 
  (hf : QuadraticFunction f) 
  (h_above : ∀ x ∈ Set.Icc 0 1, f x > 2 * x + 1 + m) :
  (∀ x, f x = x^2 - 2*x) ∧ m < -4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l978_97812


namespace NUMINAMATH_CALUDE_sin_graph_shift_l978_97875

/-- Theorem: Shifting the graph of y = 3sin(2x) to the right by π/16 units 
    results in the graph of y = 3sin(2x - π/8) -/
theorem sin_graph_shift (x : ℝ) : 
  3 * Real.sin (2 * (x - π/16)) = 3 * Real.sin (2 * x - π/8) := by
  sorry

end NUMINAMATH_CALUDE_sin_graph_shift_l978_97875


namespace NUMINAMATH_CALUDE_rectangular_prism_problem_l978_97891

/-- The number of valid triples (a, b, c) for the rectangular prism problem -/
def valid_triples : Nat :=
  (Finset.filter (fun a => a < 1995 ∧ (1995 * 1995) % a = 0)
    (Finset.range 1995)).card

/-- The theorem stating that there are exactly 40 valid triples -/
theorem rectangular_prism_problem :
  valid_triples = 40 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_problem_l978_97891


namespace NUMINAMATH_CALUDE_unique_divisible_by_45_l978_97809

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def five_digit_number (x : ℕ) : ℕ := x * 10000 + 2 * 1000 + 7 * 100 + x * 10 + 5

theorem unique_divisible_by_45 : 
  ∃! x : ℕ, digit x ∧ is_divisible_by (five_digit_number x) 45 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_45_l978_97809


namespace NUMINAMATH_CALUDE_non_intersecting_chords_eq_catalan_number_l978_97865

/-- The number of ways to draw n non-intersecting chords joining 2n points on a circle's circumference -/
def numberOfNonIntersectingChords (n : ℕ) : ℕ :=
  Nat.choose (2 * n) n / (n + 1)

/-- The nth Catalan number -/
def catalanNumber (n : ℕ) : ℕ :=
  Nat.choose (2 * n) n / (n + 1)

theorem non_intersecting_chords_eq_catalan_number :
  numberOfNonIntersectingChords 6 = catalanNumber 6 := by
  sorry

end NUMINAMATH_CALUDE_non_intersecting_chords_eq_catalan_number_l978_97865


namespace NUMINAMATH_CALUDE_calcium_phosphate_yield_l978_97841

/-- Represents the coefficients of a chemical reaction --/
structure ReactionCoefficients where
  fe2o3 : ℚ
  caco3 : ℚ
  ca3po42 : ℚ

/-- Represents the available moles of reactants --/
structure AvailableMoles where
  fe2o3 : ℚ
  caco3 : ℚ

/-- Calculates the theoretical yield of Ca3(PO4)2 based on the balanced reaction and available moles --/
def theoreticalYield (coeff : ReactionCoefficients) (available : AvailableMoles) : ℚ :=
  min 
    (available.fe2o3 * coeff.ca3po42 / coeff.fe2o3)
    (available.caco3 * coeff.ca3po42 / coeff.caco3)

/-- Theorem stating the theoretical yield of Ca3(PO4)2 for the given reaction and available moles --/
theorem calcium_phosphate_yield : 
  let coeff : ReactionCoefficients := ⟨2, 6, 3⟩
  let available : AvailableMoles := ⟨4, 10⟩
  theoreticalYield coeff available = 5 := by
  sorry

end NUMINAMATH_CALUDE_calcium_phosphate_yield_l978_97841


namespace NUMINAMATH_CALUDE_train_crossing_problem_l978_97810

/-- Calculates the length of a train given the length and speed of another train, 
    their crossing time, and the speed of the train we're calculating. -/
def train_length (other_length : ℝ) (other_speed : ℝ) (this_speed : ℝ) (cross_time : ℝ) : ℝ :=
  ((other_speed + this_speed) * cross_time - other_length)

theorem train_crossing_problem : 
  let first_train_length : ℝ := 290
  let first_train_speed : ℝ := 120 * 1000 / 3600  -- Convert km/h to m/s
  let second_train_speed : ℝ := 80 * 1000 / 3600  -- Convert km/h to m/s
  let crossing_time : ℝ := 9
  abs (train_length first_train_length first_train_speed second_train_speed crossing_time - 209.95) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_problem_l978_97810


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l978_97829

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l978_97829


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l978_97868

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- State the theorem
theorem min_value_and_inequality :
  (∃ (a : ℝ), ∀ (x : ℝ), f x ≥ a ∧ ∃ (x₀ : ℝ), f x₀ = a) ∧
  (∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → p + q + r = 3 → p^2 + q^2 + r^2 ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l978_97868


namespace NUMINAMATH_CALUDE_S_is_three_rays_l978_97835

/-- The set S of points (x,y) in the coordinate plane satisfying the given conditions -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (4 = x + 1 ∧ y - 3 ≤ 4) ∨
               (4 = y - 3 ∧ x + 1 ≤ 4) ∨
               (x + 1 = y - 3 ∧ 4 ≤ x + 1)}

/-- A ray starting from a point in a given direction -/
def Ray (start : ℝ × ℝ) (direction : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = (start.1 + t * direction.1, start.2 + t * direction.2)}

/-- The theorem stating that S consists of three rays with a common point -/
theorem S_is_three_rays :
  ∃ (r₁ r₂ r₃ : Set (ℝ × ℝ)) (common_point : ℝ × ℝ),
    S = r₁ ∪ r₂ ∪ r₃ ∧
    (∃ d₁ d₂ d₃ : ℝ × ℝ, r₁ = Ray common_point d₁ ∧
                         r₂ = Ray common_point d₂ ∧
                         r₃ = Ray common_point d₃) ∧
    common_point = (3, 7) := by
  sorry

end NUMINAMATH_CALUDE_S_is_three_rays_l978_97835


namespace NUMINAMATH_CALUDE_asymptote_angle_is_90_degrees_l978_97890

/-- A hyperbola with equation x^2 - y^2/b^2 = 1 (b > 0) and eccentricity √2 -/
structure Hyperbola where
  b : ℝ
  h_b_pos : b > 0
  h_ecc : Real.sqrt 2 = Real.sqrt (1 + 1 / b^2)

/-- The angle between the asymptotes of the hyperbola -/
def asymptote_angle (h : Hyperbola) : ℝ := sorry

/-- Theorem stating that the angle between the asymptotes is 90° -/
theorem asymptote_angle_is_90_degrees (h : Hyperbola) :
  asymptote_angle h = 90 * π / 180 := by sorry

end NUMINAMATH_CALUDE_asymptote_angle_is_90_degrees_l978_97890


namespace NUMINAMATH_CALUDE_parade_probability_l978_97897

/-- The number of possible permutations of n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The probability of an event occurring, given the number of favorable outcomes and total outcomes -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := 
  if total = 0 then 0 else (favorable : ℚ) / (total : ℚ)

/-- The number of formations in the parade -/
def num_formations : ℕ := 3

/-- The number of favorable outcomes (B passes before both A and C) -/
def favorable_outcomes : ℕ := 2

theorem parade_probability :
  probability favorable_outcomes (factorial num_formations) = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_parade_probability_l978_97897


namespace NUMINAMATH_CALUDE_area_enclosed_by_line_and_curve_l978_97878

def dataSet : List ℝ := [1, 2, 0, 0, 8, 7, 6, 5]

def median (l : List ℝ) : ℝ := sorry

def areaEnclosed (a : ℝ) : ℝ := sorry

theorem area_enclosed_by_line_and_curve (a : ℝ) :
  a ∈ dataSet →
  median dataSet = 4 →
  areaEnclosed a = 9/2 := by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_line_and_curve_l978_97878


namespace NUMINAMATH_CALUDE_log_product_l978_97824

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_product (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  lg (m * n) = lg m + lg n :=
sorry

end NUMINAMATH_CALUDE_log_product_l978_97824


namespace NUMINAMATH_CALUDE_worker_selection_theorem_l978_97800

/-- The number of workers who can only work as pliers workers -/
def pliers_only : ℕ := 5

/-- The number of workers who can only work as car workers -/
def car_only : ℕ := 4

/-- The number of workers who can work both as pliers and car workers -/
def both : ℕ := 2

/-- The total number of workers -/
def total_workers : ℕ := pliers_only + car_only + both

/-- The number of workers to be selected for pliers -/
def pliers_needed : ℕ := 4

/-- The number of workers to be selected for cars -/
def cars_needed : ℕ := 4

/-- The function to calculate the number of ways to select workers -/
def select_workers : ℕ := sorry

theorem worker_selection_theorem : select_workers = 185 := by sorry

end NUMINAMATH_CALUDE_worker_selection_theorem_l978_97800


namespace NUMINAMATH_CALUDE_tulip_fraction_l978_97847

/-- Represents the composition of a bouquet of flowers -/
structure Bouquet where
  pink_lilies : ℝ
  red_lilies : ℝ
  pink_tulips : ℝ
  red_tulips : ℝ

/-- The fraction of tulips in a bouquet satisfying given conditions -/
theorem tulip_fraction (b : Bouquet) 
  (half_pink_lilies : b.pink_lilies = b.pink_tulips)
  (third_red_tulips : b.red_tulips = (1/3) * (b.red_lilies + b.red_tulips))
  (three_fifths_pink : b.pink_lilies + b.pink_tulips = (3/5) * (b.pink_lilies + b.red_lilies + b.pink_tulips + b.red_tulips)) :
  (b.pink_tulips + b.red_tulips) / (b.pink_lilies + b.red_lilies + b.pink_tulips + b.red_tulips) = 13/30 := by
  sorry

#check tulip_fraction

end NUMINAMATH_CALUDE_tulip_fraction_l978_97847


namespace NUMINAMATH_CALUDE_sets_properties_l978_97850

def A : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) = 0}

theorem sets_properties :
  (A = {-1, 3}) ∧
  (∀ a : ℝ, {-1, 1, 3} ⊆ A ∪ B a) ∧
  (∀ a : ℝ, a ≠ -1 ∧ a ≠ 1 ∧ a ≠ 3 → A ∪ B a = {-1, 1, 3, a}) ∧
  (A ∩ B 1 = ∅) ∧
  (∀ a : ℝ, a ≠ -1 ∧ a ≠ 1 ∧ a ≠ 3 → A ∩ B a = ∅) ∧
  (A ∩ B (-1) = {-1}) ∧
  (A ∩ B 3 = {3}) := by
  sorry

end NUMINAMATH_CALUDE_sets_properties_l978_97850


namespace NUMINAMATH_CALUDE_chess_club_mixed_groups_l978_97819

/-- Represents the chess club and its game statistics -/
structure ChessClub where
  total_children : ℕ
  total_groups : ℕ
  children_per_group : ℕ
  boy_vs_boy_games : ℕ
  girl_vs_girl_games : ℕ

/-- Calculates the number of mixed groups in the chess club -/
def mixed_groups (club : ChessClub) : ℕ := 
  let total_games := club.total_groups * (club.children_per_group.choose 2)
  let mixed_games := total_games - club.boy_vs_boy_games - club.girl_vs_girl_games
  mixed_games / 2

/-- The main theorem stating the number of mixed groups -/
theorem chess_club_mixed_groups :
  let club : ChessClub := {
    total_children := 90,
    total_groups := 30,
    children_per_group := 3,
    boy_vs_boy_games := 30,
    girl_vs_girl_games := 14
  }
  mixed_groups club = 23 := by sorry

end NUMINAMATH_CALUDE_chess_club_mixed_groups_l978_97819


namespace NUMINAMATH_CALUDE_eric_pencil_boxes_l978_97883

theorem eric_pencil_boxes (pencils_per_box : ℕ) (total_pencils : ℕ) (h1 : pencils_per_box = 9) (h2 : total_pencils = 27) :
  total_pencils / pencils_per_box = 3 :=
by sorry

end NUMINAMATH_CALUDE_eric_pencil_boxes_l978_97883


namespace NUMINAMATH_CALUDE_fraction_equality_l978_97877

theorem fraction_equality (x y : ℝ) (h : y / x = 3 / 7) : (x - y) / x = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l978_97877


namespace NUMINAMATH_CALUDE_road_travel_cost_l978_97879

/-- Calculate the cost of traveling two perpendicular roads on a rectangular lawn -/
theorem road_travel_cost (lawn_length lawn_width road_width cost_per_sqm : ℝ) :
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_width = 15 ∧ 
  cost_per_sqm = 3 →
  (road_width * lawn_width + road_width * lawn_length - road_width * road_width) * cost_per_sqm = 5625 := by
  sorry


end NUMINAMATH_CALUDE_road_travel_cost_l978_97879


namespace NUMINAMATH_CALUDE_product_of_roots_plus_one_l978_97826

theorem product_of_roots_plus_one (a b c : ℝ) : 
  (x^3 - 15*x^2 + 25*x - 12 = 0 → x = a ∨ x = b ∨ x = c) →
  (1 + a) * (1 + b) * (1 + c) = 53 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_one_l978_97826


namespace NUMINAMATH_CALUDE_teacher_pen_cost_l978_97831

/-- The total cost of pens purchased by a teacher -/
theorem teacher_pen_cost : 
  let black_pens : ℕ := 7
  let blue_pens : ℕ := 9
  let red_pens : ℕ := 5
  let black_pen_cost : ℚ := 125/100
  let blue_pen_cost : ℚ := 150/100
  let red_pen_cost : ℚ := 175/100
  (black_pens : ℚ) * black_pen_cost + 
  (blue_pens : ℚ) * blue_pen_cost + 
  (red_pens : ℚ) * red_pen_cost = 31 :=
by sorry

end NUMINAMATH_CALUDE_teacher_pen_cost_l978_97831


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l978_97807

theorem ice_cream_sundaes (n : ℕ) (h : n = 8) : 
  n + n.choose 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l978_97807


namespace NUMINAMATH_CALUDE_antonov_candy_count_l978_97896

/-- The number of candies in a pack -/
def candies_per_pack : ℕ := 20

/-- The number of packs Antonov has left -/
def packs_left : ℕ := 2

/-- The number of packs Antonov gave away -/
def packs_given : ℕ := 1

/-- The total number of candies Antonov bought initially -/
def total_candies : ℕ := (packs_left + packs_given) * candies_per_pack

theorem antonov_candy_count : total_candies = 60 := by sorry

end NUMINAMATH_CALUDE_antonov_candy_count_l978_97896


namespace NUMINAMATH_CALUDE_smaller_two_digit_factor_of_4680_l978_97825

theorem smaller_two_digit_factor_of_4680 (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4680 → min a b = 52 := by
  sorry

end NUMINAMATH_CALUDE_smaller_two_digit_factor_of_4680_l978_97825


namespace NUMINAMATH_CALUDE_square_and_cube_sum_l978_97805

theorem square_and_cube_sum (p q : ℝ) 
  (h1 : p * q = 15) 
  (h2 : p + q = 8) : 
  p^2 + q^2 = 34 ∧ p^3 + q^3 = 152 := by
  sorry

end NUMINAMATH_CALUDE_square_and_cube_sum_l978_97805


namespace NUMINAMATH_CALUDE_b_n_formula_l978_97811

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem b_n_formula (a b : ℕ → ℚ) :
  arithmetic_sequence a →
  a 3 = 2 →
  a 8 = 12 →
  b 1 = 4 →
  (∀ n : ℕ, n > 1 → a n + b n = b (n - 1)) →
  ∀ n : ℕ, b n = -n^2 + 3*n + 2 := by
sorry

end NUMINAMATH_CALUDE_b_n_formula_l978_97811


namespace NUMINAMATH_CALUDE_smallest_natural_number_divisibility_l978_97874

theorem smallest_natural_number_divisibility : ∃! n : ℕ,
  (∀ m : ℕ, m < n → ¬((m + 2018) % 2020 = 0 ∧ (m + 2020) % 2018 = 0)) ∧
  (n + 2018) % 2020 = 0 ∧
  (n + 2020) % 2018 = 0 ∧
  n = 2030102 := by
  sorry

end NUMINAMATH_CALUDE_smallest_natural_number_divisibility_l978_97874


namespace NUMINAMATH_CALUDE_correct_proposition_l978_97894

-- Define proposition p
def p : Prop := ∀ a b c : ℝ, a > b → a * c^2 > b * c^2

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 - Real.log x₀ = 0

-- Theorem to prove
theorem correct_proposition : ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_correct_proposition_l978_97894


namespace NUMINAMATH_CALUDE_rectangle_division_l978_97857

theorem rectangle_division (original_width original_height : ℕ) 
  (piece1_width piece1_height : ℕ) (piece2_width piece2_height : ℕ) 
  (piece3_width piece3_height : ℕ) (piece4_width piece4_height : ℕ) :
  original_width = 15 ∧ original_height = 7 ∧
  piece1_width = 7 ∧ piece1_height = 7 ∧
  piece2_width = 8 ∧ piece2_height = 3 ∧
  piece3_width = 7 ∧ piece3_height = 4 ∧
  piece4_width = 8 ∧ piece4_height = 4 →
  original_width * original_height = 
    piece1_width * piece1_height + 
    piece2_width * piece2_height + 
    piece3_width * piece3_height + 
    piece4_width * piece4_height :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_l978_97857


namespace NUMINAMATH_CALUDE_power_of_power_of_three_l978_97815

theorem power_of_power_of_three :
  (3^3)^(3^3) = 27^27 := by sorry

end NUMINAMATH_CALUDE_power_of_power_of_three_l978_97815


namespace NUMINAMATH_CALUDE_sum_of_sixth_powers_l978_97898

theorem sum_of_sixth_powers (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : a^3 + b^3 + c^3 = 8)
  (h3 : a^5 + b^5 + c^5 = 32) :
  a^6 + b^6 + c^6 = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_sixth_powers_l978_97898


namespace NUMINAMATH_CALUDE_inequality_proof_l978_97849

theorem inequality_proof (a b x : ℝ) (h : 0 ≤ x ∧ x < Real.pi / 2) :
  a^2 * Real.tan x * (Real.cos x)^(1/3) + b^2 * Real.sin x ≥ 2 * x * a * b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l978_97849


namespace NUMINAMATH_CALUDE_quadratic_max_value_l978_97828

/-- The quadratic function f(x) = ax² + 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

/-- The maximum value of f(x) on the interval [-3, 2] -/
def max_value : ℝ := 9

/-- The lower bound of the interval -/
def lower_bound : ℝ := -3

/-- The upper bound of the interval -/
def upper_bound : ℝ := 2

theorem quadratic_max_value (a : ℝ) :
  (∀ x, lower_bound ≤ x ∧ x ≤ upper_bound → f a x ≤ max_value) ∧
  (∃ x, lower_bound ≤ x ∧ x ≤ upper_bound ∧ f a x = max_value) →
  a = 1 ∨ a = -8 :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l978_97828


namespace NUMINAMATH_CALUDE_equation_holds_for_all_x_l978_97836

theorem equation_holds_for_all_x : ∃ (a b c : ℝ), ∀ (x : ℝ), 
  (x + a)^2 + (2*x + b)^2 + (2*x + c)^2 = (3*x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_for_all_x_l978_97836


namespace NUMINAMATH_CALUDE_total_interest_calculation_total_interest_is_1530_l978_97814

/-- Calculates the total interest earned on two certificates of deposit --/
theorem total_interest_calculation (total_investment : ℝ) (rate1 rate2 : ℝ) 
  (fraction_higher_rate : ℝ) : ℝ :=
  let amount_higher_rate := total_investment * fraction_higher_rate
  let amount_lower_rate := total_investment - amount_higher_rate
  let interest_higher_rate := amount_higher_rate * rate2
  let interest_lower_rate := amount_lower_rate * rate1
  interest_higher_rate + interest_lower_rate

/-- Proves that the total interest earned is $1,530 given the problem conditions --/
theorem total_interest_is_1530 : 
  total_interest_calculation 20000 0.06 0.09 0.55 = 1530 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_calculation_total_interest_is_1530_l978_97814


namespace NUMINAMATH_CALUDE_cube_edge_length_from_circumscribed_sphere_volume_l978_97821

theorem cube_edge_length_from_circumscribed_sphere_volume 
  (V : ℝ) (a : ℝ) (h : V = (32 / 3) * Real.pi) :
  (V = (4 / 3) * Real.pi * (a * Real.sqrt 3 / 2)^3) → a = (4 * Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_from_circumscribed_sphere_volume_l978_97821


namespace NUMINAMATH_CALUDE_parabola_midpoint_distance_l978_97852

/-- Given a parabola y² = 4x and a line passing through its focus,
    intersecting the parabola at points A and B with |AB| = 7,
    the distance from the midpoint M of AB to the directrix is 7/2. -/
theorem parabola_midpoint_distance (x₁ y₁ x₂ y₂ : ℝ) :
  y₁^2 = 4*x₁ →
  y₂^2 = 4*x₂ →
  (x₁ - 1)^2 + y₁^2 = (x₂ - 1)^2 + y₂^2 →  -- line passes through focus (1, 0)
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = 49 →         -- |AB| = 7
  (((x₁ + x₂)/2 + 1) : ℝ) = 7/2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_midpoint_distance_l978_97852


namespace NUMINAMATH_CALUDE_discounted_price_is_nine_l978_97839

/-- The final price after applying a discount --/
def final_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

/-- Theorem: The final price of a $10 item after a 10% discount is $9 --/
theorem discounted_price_is_nine :
  final_price 10 0.1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_is_nine_l978_97839


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l978_97884

theorem inverse_variation_problem (k : ℝ) (x y : ℝ → ℝ) (h1 : ∀ t, 5 * y t = k / (x t)^2)
  (h2 : y 1 = 16) (h3 : x 1 = 1) : y 8 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l978_97884


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l978_97808

def vector_a : Fin 2 → ℝ := ![1, -2]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![x, 4]

theorem parallel_vectors_magnitude (x : ℝ) :
  (∃ k : ℝ, vector_a = k • vector_b x) →
  Real.sqrt ((vector_a 0 - vector_b x 0)^2 + (vector_a 1 - vector_b x 1)^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l978_97808


namespace NUMINAMATH_CALUDE_tagged_ratio_is_two_fiftieths_l978_97806

/-- Represents the fish population and tagging experiment in a pond -/
structure FishExperiment where
  initial_tagged : ℕ
  second_catch : ℕ
  tagged_in_second : ℕ
  total_fish : ℕ

/-- The ratio of tagged fish to total fish in the second catch -/
def tagged_ratio (e : FishExperiment) : ℚ :=
  e.tagged_in_second / e.second_catch

/-- The given experiment data -/
def pond_experiment : FishExperiment :=
  { initial_tagged := 70
  , second_catch := 50
  , tagged_in_second := 2
  , total_fish := 1750 }

/-- Theorem stating that the ratio of tagged fish in the second catch is 2/50 -/
theorem tagged_ratio_is_two_fiftieths :
  tagged_ratio pond_experiment = 2 / 50 := by
  sorry

end NUMINAMATH_CALUDE_tagged_ratio_is_two_fiftieths_l978_97806


namespace NUMINAMATH_CALUDE_mistaken_calculation_correction_l978_97870

theorem mistaken_calculation_correction (x : ℝ) : 
  x * 4 = 166.08 → x / 4 = 10.38 := by
sorry

end NUMINAMATH_CALUDE_mistaken_calculation_correction_l978_97870


namespace NUMINAMATH_CALUDE_sphere_in_truncated_cone_l978_97816

/-- 
Given a sphere perfectly fitted inside a truncated right circular cone, 
if the volume of the truncated cone is three times that of the sphere, 
then the ratio of the radius of the larger base to the radius of the smaller base 
of the truncated cone is (5 + √21) / 2.
-/
theorem sphere_in_truncated_cone (R r s : ℝ) 
  (h_fit : s^2 = R * r)  -- sphere fits perfectly inside the truncated cone
  (h_volume : (π / 3) * (R^2 + R*r + r^2) * (2*s + (2*s*r)/(R-r)) - 
              (π / 3) * r^2 * ((2*s*r)/(R-r)) = 
              4 * π * s^3) :  -- volume relation
  R / r = (5 + Real.sqrt 21) / 2 := by
sorry

end NUMINAMATH_CALUDE_sphere_in_truncated_cone_l978_97816


namespace NUMINAMATH_CALUDE_age_problem_l978_97885

theorem age_problem (P R J M : ℕ) : 
  P = R / 2 →
  R + 12 = J + 12 + 7 →
  J + 12 = 3 * P →
  M + 8 = J + 8 + 9 →
  M + 4 = 2 * (R + 4) →
  P = 5 ∧ R = 10 ∧ J = 3 ∧ M = 24 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l978_97885


namespace NUMINAMATH_CALUDE_unique_prime_with_square_free_remainders_l978_97867

theorem unique_prime_with_square_free_remainders : ∃! p : ℕ, 
  Nat.Prime p ∧ 
  (∀ q : ℕ, Nat.Prime q → q < p → 
    ∀ k r : ℕ, p = k * q + r → 0 ≤ r → r < q → 
      ∀ a : ℕ, a > 1 → ¬(a * a ∣ r)) ∧
  p = 13 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_with_square_free_remainders_l978_97867


namespace NUMINAMATH_CALUDE_smallest_special_number_l978_97887

def unit_digit (n : ℕ) : ℕ := n % 10

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem smallest_special_number : 
  ∀ n : ℕ, 
    (unit_digit n = 5 ∧ 
     is_perfect_square n ∧ 
     (∃ k : ℕ, k * k = n ∧ digit_sum k = 9)) → 
    n ≥ 2025 :=
sorry

end NUMINAMATH_CALUDE_smallest_special_number_l978_97887


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l978_97845

theorem smallest_number_with_remainders : ∃! x : ℕ, 
  x > 0 ∧
  x % 45 = 4 ∧
  x % 454 = 45 ∧
  x % 4545 = 454 ∧
  x % 45454 = 4545 ∧
  ∀ y : ℕ, y > 0 ∧ y % 45 = 4 ∧ y % 454 = 45 ∧ y % 4545 = 454 ∧ y % 45454 = 4545 → x ≤ y :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l978_97845


namespace NUMINAMATH_CALUDE_symmetric_complex_product_l978_97801

theorem symmetric_complex_product (z₁ z₂ : ℂ) :
  (z₁.re = z₂.im ∧ z₁.im = z₂.re) →  -- symmetry about y=x
  z₁ * z₂ = Complex.I * 9 →          -- product condition
  Complex.abs z₁ = 3 := by            
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_product_l978_97801


namespace NUMINAMATH_CALUDE_pizzeria_sales_l978_97804

theorem pizzeria_sales (small_price large_price total_revenue small_count : ℕ) 
  (h1 : small_price = 2)
  (h2 : large_price = 8)
  (h3 : total_revenue = 40)
  (h4 : small_count = 8) :
  ∃ (large_count : ℕ), 
    small_price * small_count + large_price * large_count = total_revenue ∧ 
    large_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_pizzeria_sales_l978_97804


namespace NUMINAMATH_CALUDE_new_person_weight_l978_97823

theorem new_person_weight (n : ℕ) (initial_weight replaced_weight avg_increase : ℝ) :
  n = 8 ∧ 
  replaced_weight = 65 ∧
  avg_increase = 4 →
  n * avg_increase + replaced_weight = 97 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l978_97823


namespace NUMINAMATH_CALUDE_small_boxes_count_l978_97880

theorem small_boxes_count (chocolate_per_small_box : ℕ) (total_chocolate : ℕ) : 
  chocolate_per_small_box = 25 → total_chocolate = 475 → 
  total_chocolate / chocolate_per_small_box = 19 := by
  sorry

end NUMINAMATH_CALUDE_small_boxes_count_l978_97880


namespace NUMINAMATH_CALUDE_largest_non_prime_consecutive_l978_97822

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_integers (start : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (λ i => start + i)

theorem largest_non_prime_consecutive :
  ∃ (start : ℕ),
    start + 5 = 35 ∧
    start < 50 ∧
    (∀ n ∈ consecutive_integers start 6, n < 50 ∧ ¬(is_prime n)) ∧
    (∀ m : ℕ, m > start + 5 →
      ¬(∃ s : ℕ, s + 5 = m ∧
        s < 50 ∧
        (∀ n ∈ consecutive_integers s 6, n < 50 ∧ ¬(is_prime n)))) :=
sorry

end NUMINAMATH_CALUDE_largest_non_prime_consecutive_l978_97822


namespace NUMINAMATH_CALUDE_percentage_increase_l978_97827

theorem percentage_increase (initial_earnings new_earnings : ℝ) (h1 : initial_earnings = 60) (h2 : new_earnings = 72) :
  (new_earnings - initial_earnings) / initial_earnings * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l978_97827


namespace NUMINAMATH_CALUDE_difference_of_squares_l978_97866

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l978_97866


namespace NUMINAMATH_CALUDE_max_min_on_interval_l978_97813

/-- A function satisfying the given properties -/
def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x > 0 → f x < 0) ∧
  (f 2 = -1)

/-- Theorem stating the existence and values of maximum and minimum on [-6,6] -/
theorem max_min_on_interval (f : ℝ → ℝ) (h : f_properties f) :
  (∃ max_val : ℝ, IsGreatest {y | ∃ x ∈ Set.Icc (-6) 6, f x = y} max_val ∧ max_val = 3) ∧
  (∃ min_val : ℝ, IsLeast {y | ∃ x ∈ Set.Icc (-6) 6, f x = y} min_val ∧ min_val = -3) :=
sorry

end NUMINAMATH_CALUDE_max_min_on_interval_l978_97813


namespace NUMINAMATH_CALUDE_group_frequency_l978_97802

theorem group_frequency (sample_capacity : ℕ) (group_frequency : ℚ) : 
  sample_capacity = 1000 → 
  group_frequency = 6/10 → 
  (↑sample_capacity * group_frequency : ℚ) = 600 := by
  sorry

end NUMINAMATH_CALUDE_group_frequency_l978_97802


namespace NUMINAMATH_CALUDE_protons_equal_atomic_number_oxygen16_protons_l978_97859

/-- Represents an atom with mass number and atomic number -/
structure Atom where
  mass_number : ℕ
  atomic_number : ℕ

/-- The oxygen-16 atom -/
def oxygen16 : Atom := { mass_number := 16, atomic_number := 8 }

/-- The number of protons in an atom is equal to its atomic number -/
theorem protons_equal_atomic_number (a : Atom) : a.atomic_number = a.atomic_number := by sorry

theorem oxygen16_protons : oxygen16.atomic_number = 8 := by sorry

end NUMINAMATH_CALUDE_protons_equal_atomic_number_oxygen16_protons_l978_97859


namespace NUMINAMATH_CALUDE_min_points_on_circle_l978_97830

theorem min_points_on_circle (n : ℕ) (h : n ≥ 3) :
  let N := if (2*n - 1) % 3 = 0 then n else n - 1
  ∀ (S : Finset (Fin (2*n - 1))),
    S.card ≥ N →
    ∃ (i j : Fin (2*n - 1)), i ∈ S ∧ j ∈ S ∧
      (((j - i : ℤ) + (2*n - 1)) % (2*n - 1) = n ∨
       ((i - j : ℤ) + (2*n - 1)) % (2*n - 1) = n) :=
by sorry

end NUMINAMATH_CALUDE_min_points_on_circle_l978_97830


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_C_R_B_l978_97856

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 ≥ 0}

-- Define the complement of B in ℝ
def C_R_B : Set ℝ := {x | ¬ (x ∈ B)}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

-- Theorem for the union of A and complement of B
theorem union_A_C_R_B : A ∪ C_R_B = {x : ℝ | -4 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_C_R_B_l978_97856
