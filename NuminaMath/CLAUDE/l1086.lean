import Mathlib

namespace NUMINAMATH_CALUDE_expression_equality_l1086_108654

theorem expression_equality : (-1)^2023 - Real.sqrt 9 + |1 - Real.sqrt 2| - ((-8) ^ (1/3 : ℝ)) = Real.sqrt 2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1086_108654


namespace NUMINAMATH_CALUDE_hex_to_binary_bits_l1086_108691

/-- The hexadecimal number A3F52 -/
def hex_number : ℕ := 0xA3F52

/-- The number of bits in the binary representation of a natural number -/
def num_bits (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log2 n + 1

theorem hex_to_binary_bits :
  num_bits hex_number = 20 := by sorry

end NUMINAMATH_CALUDE_hex_to_binary_bits_l1086_108691


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_17_l1086_108624

theorem consecutive_integers_sqrt_17 (a b : ℕ) :
  a > 0 ∧ b > 0 ∧ b = a + 1 ∧ (a : ℝ) < Real.sqrt 17 ∧ Real.sqrt 17 < (b : ℝ) → a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_17_l1086_108624


namespace NUMINAMATH_CALUDE_profit_ratio_theorem_l1086_108602

/-- Given two partners p and q with investment ratio 7:5, where p invests for 10 months
    and q invests for 20 months, prove that the ratio of their profits is 7:10 -/
theorem profit_ratio_theorem (p q : ℕ) (investment_p investment_q : ℝ) 
  (time_p time_q : ℕ) (profit_p profit_q : ℝ) :
  investment_p / investment_q = 7 / 5 →
  time_p = 10 →
  time_q = 20 →
  profit_p = investment_p * time_p →
  profit_q = investment_q * time_q →
  profit_p / profit_q = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_profit_ratio_theorem_l1086_108602


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l1086_108638

/-- Given a 40-liter solution of alcohol and water, prove that the initial percentage
    of alcohol is 5% if adding 4.5 liters of alcohol and 5.5 liters of water
    results in a 50-liter solution that is 13% alcohol. -/
theorem initial_alcohol_percentage
  (initial_volume : ℝ)
  (added_alcohol : ℝ)
  (added_water : ℝ)
  (final_volume : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 40)
  (h2 : added_alcohol = 4.5)
  (h3 : added_water = 5.5)
  (h4 : final_volume = initial_volume + added_alcohol + added_water)
  (h5 : final_percentage = 13)
  (h6 : final_percentage / 100 * final_volume = 
        initial_volume * (initial_percentage / 100) + added_alcohol) :
  initial_percentage = 5 :=
by sorry

end NUMINAMATH_CALUDE_initial_alcohol_percentage_l1086_108638


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1086_108600

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) : 
  Complex.im ((1 - 2*i) / (2 + i)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1086_108600


namespace NUMINAMATH_CALUDE_solution_when_a_eq_one_two_solutions_range_max_value_F_l1086_108621

-- Define the functions
def f (a x : ℝ) := |x - a|
def g (a x : ℝ) := a * x
def F (a x : ℝ) := g a x * f a x

-- Theorem 1
theorem solution_when_a_eq_one :
  ∃ x : ℝ, f 1 x = g 1 x ∧ x = 1/2 := by sorry

-- Theorem 2
theorem two_solutions_range :
  ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a x = g a x ∧ f a y = g a y) ↔ 
  (a > -1 ∧ a < 0) ∨ (a > 0 ∧ a < 1) := by sorry

-- Theorem 3
theorem max_value_F :
  ∀ a : ℝ, a > 0 → 
  (∃ max : ℝ, ∀ x : ℝ, x ∈ Set.Icc 1 2 → F a x ≤ max) ∧
  (let max := if a < 5/3 then 4*a - 2*a^2
              else if a ≤ 2 then a^2 - a
              else if a < 4 then a^3/4
              else 2*a^2 - 4*a;
   ∀ x : ℝ, x ∈ Set.Icc 1 2 → F a x ≤ max) := by sorry

end NUMINAMATH_CALUDE_solution_when_a_eq_one_two_solutions_range_max_value_F_l1086_108621


namespace NUMINAMATH_CALUDE_composition_equation_solution_l1086_108608

/-- Given functions f and g, and a condition on their composition, prove the value of a. -/
theorem composition_equation_solution (a : ℝ) : 
  (let f (x : ℝ) := (x + 4) / 7 + 2
   let g (x : ℝ) := 5 - 2 * x
   f (g a) = 8) → 
  a = -33/2 := by
sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l1086_108608


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1086_108685

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |1/2 * x| = 2 ↔ x = 4 ∨ x = -4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1086_108685


namespace NUMINAMATH_CALUDE_white_rabbit_distance_proof_l1086_108613

/-- The hopping distance of the white rabbit per minute -/
def white_rabbit_distance : ℝ := 15

/-- The hopping distance of the brown rabbit per minute -/
def brown_rabbit_distance : ℝ := 12

/-- The total time the rabbits hop -/
def total_time : ℝ := 5

/-- The total distance both rabbits hop in the given time -/
def total_distance : ℝ := 135

theorem white_rabbit_distance_proof :
  white_rabbit_distance * total_time + brown_rabbit_distance * total_time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_white_rabbit_distance_proof_l1086_108613


namespace NUMINAMATH_CALUDE_max_adjusted_employees_range_of_a_l1086_108651

/- Define the total number of employees -/
def total_employees : ℕ := 1000

/- Define the original average profit per employee (in yuan) -/
def original_profit : ℕ := 100000

/- Define the function for adjusted employees' average profit -/
def adjusted_profit (a x : ℝ) : ℝ := 10000 * (a - 0.008 * x)

/- Define the function for remaining employees' average profit -/
def remaining_profit (x : ℝ) : ℝ := original_profit * (1 + 0.004 * x)

/- Theorem for part I -/
theorem max_adjusted_employees :
  ∃ (max_x : ℕ), max_x = 750 ∧
  ∀ (x : ℕ), x > 0 → x ≤ max_x →
  (total_employees - x : ℝ) * remaining_profit x ≥ total_employees * original_profit ∧
  ¬∃ (y : ℕ), y > max_x ∧ y > 0 ∧
  (total_employees - y : ℝ) * remaining_profit y ≥ total_employees * original_profit :=
sorry

/- Theorem for part II -/
theorem range_of_a (x : ℝ) (hx : 0 < x ∧ x ≤ 750) :
  ∃ (lower upper : ℝ), lower = 0 ∧ upper = 7 ∧
  ∀ (a : ℝ), a > lower ∧ a ≤ upper →
  x * adjusted_profit a x ≤ (total_employees - x) * remaining_profit x ∧
  ¬∃ (b : ℝ), b > upper ∧
  x * adjusted_profit b x ≤ (total_employees - x) * remaining_profit x :=
sorry

end NUMINAMATH_CALUDE_max_adjusted_employees_range_of_a_l1086_108651


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1086_108642

theorem gcd_of_three_numbers : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1086_108642


namespace NUMINAMATH_CALUDE_equation_solution_l1086_108671

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    x₁ > 0 ∧ x₂ > 0 ∧
    (∀ (x : ℝ), x > 0 → 
      ((1/3) * (4*x^2 - 3) = (x^2 - 75*x - 15) * (x^2 + 40*x + 8)) ↔ 
      (x = x₁ ∨ x = x₂)) ∧
    x₁ = (75 + Real.sqrt 5677) / 2 ∧
    x₂ = (-40 + Real.sqrt 1572) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1086_108671


namespace NUMINAMATH_CALUDE_equation_holds_iff_conditions_l1086_108644

theorem equation_holds_iff_conditions (a b c : ℤ) :
  a * (a - b) + b * (b - c) + c * (c - a) = 2 ↔ 
  ((a = b ∧ b = c + 1) ∨ (a = c ∧ b - 1 = c)) := by
sorry

end NUMINAMATH_CALUDE_equation_holds_iff_conditions_l1086_108644


namespace NUMINAMATH_CALUDE_inequality_proof_l1086_108649

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1086_108649


namespace NUMINAMATH_CALUDE_card_game_combinations_l1086_108660

theorem card_game_combinations : Nat.choose 52 13 = 635013587600 := by
  sorry

end NUMINAMATH_CALUDE_card_game_combinations_l1086_108660


namespace NUMINAMATH_CALUDE_liquid_film_radius_l1086_108686

theorem liquid_film_radius (volume : ℝ) (thickness : ℝ) (radius : ℝ) : 
  volume = 320 →
  thickness = 0.05 →
  volume = π * radius^2 * thickness →
  radius = Real.sqrt (6400 / π) := by
sorry

end NUMINAMATH_CALUDE_liquid_film_radius_l1086_108686


namespace NUMINAMATH_CALUDE_triangle_area_l1086_108604

theorem triangle_area (a c : ℝ) (B : ℝ) (h1 : a = 3 * Real.sqrt 3) (h2 : c = 2) (h3 : B = π / 3) :
  (1 / 2) * a * c * Real.sin B = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1086_108604


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1086_108695

theorem min_value_of_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 2/n = 1) :
  m + n ≥ (Real.sqrt 2 + 1)^2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1086_108695


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1086_108625

theorem triangle_angle_measure (a b c A B C : Real) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  -- Given conditions
  (a = Real.sqrt 2) →
  (b = 2) →
  (B = π / 4) →
  -- Conclusion
  (A = π / 6) := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1086_108625


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_l1086_108619

theorem sum_of_even_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5) →
  a₀ + a₂ + a₄ = -16 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_l1086_108619


namespace NUMINAMATH_CALUDE_ideal_function_fixed_point_l1086_108639

/-- An ideal function is a function f: [0,1] → ℝ satisfying:
    1) ∀ x ∈ [0,1], f(x) ≥ 0
    2) f(1) = 1
    3) ∀ x₁ x₂ ≥ 0 with x₁ + x₂ ≤ 1, f(x₁ + x₂) ≥ f(x₁) + f(x₂) -/
def IdealFunction (f : Real → Real) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧ 
  (f 1 = 1) ∧
  (∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

theorem ideal_function_fixed_point 
  (f : Real → Real) (h : IdealFunction f) 
  (x₀ : Real) (hx₀ : x₀ ∈ Set.Icc 0 1) 
  (hfx₀ : f x₀ ∈ Set.Icc 0 1) (hffx₀ : f (f x₀) = x₀) : 
  f x₀ = x₀ := by
  sorry

end NUMINAMATH_CALUDE_ideal_function_fixed_point_l1086_108639


namespace NUMINAMATH_CALUDE_complex_product_equality_l1086_108676

theorem complex_product_equality (x : ℂ) (h : x = Complex.exp (2 * Real.pi * Complex.I / 9)) : 
  (3 * x + x^3) * (3 * x^3 + x^9) * (3 * x^6 + x^18) = 
  22 - 9 * x^5 - 9 * x^2 + 3 * x^6 + 4 * x^3 + 3 * x :=
by sorry

end NUMINAMATH_CALUDE_complex_product_equality_l1086_108676


namespace NUMINAMATH_CALUDE_equal_salary_at_5000_sales_l1086_108681

/-- Represents the monthly salary options for Juliet --/
structure SalaryOptions where
  flat_salary : ℝ
  base_salary : ℝ
  commission_rate : ℝ

/-- Calculates the total salary for the commission-based option --/
def commission_salary (options : SalaryOptions) (sales : ℝ) : ℝ :=
  options.base_salary + options.commission_rate * sales

/-- The specific salary options given in the problem --/
def juliet_options : SalaryOptions :=
  { flat_salary := 1800
    base_salary := 1600
    commission_rate := 0.04 }

/-- Theorem stating that the sales amount for equal salaries is $5000 --/
theorem equal_salary_at_5000_sales (options : SalaryOptions := juliet_options) :
  ∃ (sales : ℝ), sales = 5000 ∧ options.flat_salary = commission_salary options sales :=
by
  sorry

end NUMINAMATH_CALUDE_equal_salary_at_5000_sales_l1086_108681


namespace NUMINAMATH_CALUDE_leftover_grass_seed_coverage_l1086_108623

/-- Proves the leftover grass seed coverage for Drew's lawn -/
theorem leftover_grass_seed_coverage 
  (lawn_length : ℕ) 
  (lawn_width : ℕ) 
  (seed_bags : ℕ) 
  (coverage_per_bag : ℕ) :
  lawn_length = 22 →
  lawn_width = 36 →
  seed_bags = 4 →
  coverage_per_bag = 250 →
  seed_bags * coverage_per_bag - lawn_length * lawn_width = 208 :=
by sorry

end NUMINAMATH_CALUDE_leftover_grass_seed_coverage_l1086_108623


namespace NUMINAMATH_CALUDE_sum_of_seventh_row_l1086_108605

-- Define the sum function for the triangular array
def f : ℕ → ℕ
  | 0 => 0  -- Base case: f(0) = 0 (not used in the problem, but needed for recursion)
  | 1 => 2  -- Base case: f(1) = 2
  | n + 1 => 2 * f n + 4  -- Recursive case: f(n+1) = 2f(n) + 4

-- Theorem statement
theorem sum_of_seventh_row : f 7 = 284 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seventh_row_l1086_108605


namespace NUMINAMATH_CALUDE_rectangle_area_l1086_108682

theorem rectangle_area (length width : ℚ) (h1 : length = 1/3) (h2 : width = 1/5) :
  length * width = 1/15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1086_108682


namespace NUMINAMATH_CALUDE_final_pen_count_l1086_108648

theorem final_pen_count (x : ℝ) (x_pos : x > 0) : 
  let after_mike := x + 0.5 * x
  let after_cindy := 2 * after_mike
  let given_to_sharon := 0.25 * after_cindy
  after_cindy - given_to_sharon = 2.25 * x :=
by sorry

end NUMINAMATH_CALUDE_final_pen_count_l1086_108648


namespace NUMINAMATH_CALUDE_tan_beta_value_l1086_108614

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l1086_108614


namespace NUMINAMATH_CALUDE_move_right_four_units_l1086_108622

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Move a point horizontally in a Cartesian coordinate system -/
def moveHorizontal (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

/-- The theorem stating that moving (-2, 3) 4 units right results in (2, 3) -/
theorem move_right_four_units :
  let initial : Point := { x := -2, y := 3 }
  let final : Point := moveHorizontal initial 4
  final.x = 2 ∧ final.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_move_right_four_units_l1086_108622


namespace NUMINAMATH_CALUDE_cary_savings_l1086_108615

/-- Proves that the amount Cary has already saved is equal to the cost of the shoes
    minus the amount he will earn in the next 6 weekends. -/
theorem cary_savings (shoe_cost : ℕ) (lawn_pay : ℕ) (lawns_per_weekend : ℕ) (weekends_left : ℕ)
    (h1 : shoe_cost = 120)
    (h2 : lawn_pay = 5)
    (h3 : lawns_per_weekend = 3)
    (h4 : weekends_left = 6) :
  shoe_cost - (lawn_pay * lawns_per_weekend * weekends_left) = 30 :=
by sorry

end NUMINAMATH_CALUDE_cary_savings_l1086_108615


namespace NUMINAMATH_CALUDE_modular_power_congruence_l1086_108629

theorem modular_power_congruence (p : ℕ) (n : ℕ) (a b : ℤ) 
  (h_prime : Nat.Prime p) (h_cong : a ≡ b [ZMOD p^n]) :
  a^p ≡ b^p [ZMOD p^(n+1)] := by sorry

end NUMINAMATH_CALUDE_modular_power_congruence_l1086_108629


namespace NUMINAMATH_CALUDE_locus_of_centers_l1086_108617

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C₂ with equation (x - 3)² + y² = 9 -/
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

/-- A circle is externally tangent to C₁ if the distance between their centers
    equals the sum of their radii -/
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C₂ if the distance between their centers
    equals the difference of their radii -/
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 3)^2 + b^2 = (3 - r)^2

/-- The locus of centers (a,b) of circles externally tangent to C₁ and internally tangent to C₂
    satisfies the equation 28a² + 64b² - 84a - 49 = 0 -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) → 
  28 * a^2 + 64 * b^2 - 84 * a - 49 = 0 := by
  sorry

end NUMINAMATH_CALUDE_locus_of_centers_l1086_108617


namespace NUMINAMATH_CALUDE_garrett_cat_count_l1086_108674

/-- The number of cats Mrs. Sheridan has -/
def sheridan_cats : ℕ := 11

/-- The difference between Mrs. Garrett's and Mrs. Sheridan's cats -/
def cat_difference : ℕ := 13

/-- Mrs. Garrett's cats -/
def garrett_cats : ℕ := sheridan_cats + cat_difference

theorem garrett_cat_count : garrett_cats = 24 := by
  sorry

end NUMINAMATH_CALUDE_garrett_cat_count_l1086_108674


namespace NUMINAMATH_CALUDE_business_profit_calculation_l1086_108666

/-- Represents the total profit of a business partnership --/
def total_profit (a_investment b_investment : ℕ) (a_management_fee : ℚ) (a_total_received : ℕ) : ℚ :=
  let total_investment := a_investment + b_investment
  let remaining_profit_share := 1 - a_management_fee
  let a_profit_share := (a_investment : ℚ) / (total_investment : ℚ) * remaining_profit_share
  (a_total_received : ℚ) / (a_management_fee + a_profit_share)

/-- Theorem stating the total profit of the business partnership --/
theorem business_profit_calculation :
  total_profit 3500 2500 (1/10) 6000 = 9600 := by
  sorry

#eval total_profit 3500 2500 (1/10) 6000

end NUMINAMATH_CALUDE_business_profit_calculation_l1086_108666


namespace NUMINAMATH_CALUDE_sum_abc_equals_42_l1086_108673

theorem sum_abc_equals_42 
  (a b c : ℕ+) 
  (h1 : a * b + c = 41)
  (h2 : b * c + a = 41)
  (h3 : a * c + b = 41) : 
  a + b + c = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_abc_equals_42_l1086_108673


namespace NUMINAMATH_CALUDE_jacob_rental_cost_l1086_108618

/-- Calculates the total cost of renting a car given the daily rate, per-mile rate, number of days, and miles driven. -/
def total_rental_cost (daily_rate : ℚ) (mile_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mile_rate * miles

/-- Proves that Jacob's total car rental cost is $237.5 given the specified conditions. -/
theorem jacob_rental_cost :
  let daily_rate : ℚ := 30
  let mile_rate : ℚ := 1/4
  let rental_days : ℕ := 5
  let miles_driven : ℕ := 350
  total_rental_cost daily_rate mile_rate rental_days miles_driven = 237.5 := by
sorry

end NUMINAMATH_CALUDE_jacob_rental_cost_l1086_108618


namespace NUMINAMATH_CALUDE_largest_number_l1086_108633

-- Define the function to convert a number from any base to decimal (base 10)
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the numbers in their respective bases
def binary : List Nat := [1, 1, 1, 1, 1]
def ternary : List Nat := [1, 2, 2, 1]
def quaternary : List Nat := [2, 1, 3]
def octal : List Nat := [6, 5]

-- State the theorem
theorem largest_number :
  to_decimal quaternary 4 = 54 ∧
  to_decimal quaternary 4 > to_decimal binary 2 ∧
  to_decimal quaternary 4 > to_decimal ternary 3 ∧
  to_decimal quaternary 4 > to_decimal octal 8 :=
sorry

end NUMINAMATH_CALUDE_largest_number_l1086_108633


namespace NUMINAMATH_CALUDE_students_checked_out_early_l1086_108656

theorem students_checked_out_early (total_students : Nat) (students_left : Nat) : 
  total_students = 16 → students_left = 9 → total_students - students_left = 7 := by
  sorry

end NUMINAMATH_CALUDE_students_checked_out_early_l1086_108656


namespace NUMINAMATH_CALUDE_financial_equation_solution_l1086_108637

theorem financial_equation_solution (g t p : ℂ) : 
  3 * g * p - t = 9000 ∧ g = 3 ∧ t = 3 + 75 * Complex.I → 
  p = 1000 + 1/3 + 8 * Complex.I + 1/3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_financial_equation_solution_l1086_108637


namespace NUMINAMATH_CALUDE_equal_distribution_iff_even_total_l1086_108680

/-- Two piles of nuts with different numbers of nuts -/
structure NutPiles :=
  (pile1 : ℕ)
  (pile2 : ℕ)
  (different : pile1 ≠ pile2)

/-- The total number of nuts in both piles -/
def total_nuts (piles : NutPiles) : ℕ := piles.pile1 + piles.pile2

/-- A predicate indicating whether equal distribution is possible -/
def equal_distribution_possible (piles : NutPiles) : Prop :=
  ∃ (k : ℕ), piles.pile1 - k = piles.pile2 + k

/-- Theorem stating that equal distribution is possible if and only if the total number of nuts is even -/
theorem equal_distribution_iff_even_total (piles : NutPiles) :
  equal_distribution_possible piles ↔ Even (total_nuts piles) :=
sorry

end NUMINAMATH_CALUDE_equal_distribution_iff_even_total_l1086_108680


namespace NUMINAMATH_CALUDE_square_sum_equals_eight_l1086_108657

theorem square_sum_equals_eight (a b : ℝ) 
  (h1 : (a + b)^2 = 11) 
  (h2 : (a - b)^2 = 5) : 
  a^2 + b^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_eight_l1086_108657


namespace NUMINAMATH_CALUDE_extreme_value_a_1_monotonicity_a_leq_neg_1_monotonicity_a_gt_neg_1_l1086_108677

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 1 + a) / x - a * Real.log x

-- Theorem for the extreme value when a = 1
theorem extreme_value_a_1 :
  ∃ (x_min : ℝ), x_min > 0 ∧ 
  (∀ x > 0, f 1 x_min ≤ f 1 x) ∧
  f 1 x_min = Real.sqrt 2 + 3/2 - (1/2) * Real.log 2 :=
sorry

-- Theorem for monotonicity when a ≤ -1
theorem monotonicity_a_leq_neg_1 (a : ℝ) (h : a ≤ -1) :
  ∀ x y, 0 < x → 0 < y → x < y → f a x < f a y :=
sorry

-- Theorem for monotonicity when a > -1
theorem monotonicity_a_gt_neg_1 (a : ℝ) (h : a > -1) :
  (∀ x y, 0 < x → x < y → y < 1 + a → f a x > f a y) ∧
  (∀ x y, 1 + a < x → x < y → f a x < f a y) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_a_1_monotonicity_a_leq_neg_1_monotonicity_a_gt_neg_1_l1086_108677


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l1086_108658

theorem arithmetic_evaluation : (7 - 6 * (-5)) - 4 * (-3) / (-2) = 31 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l1086_108658


namespace NUMINAMATH_CALUDE_ones_digit_of_8_to_47_l1086_108606

theorem ones_digit_of_8_to_47 : 8^47 % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_8_to_47_l1086_108606


namespace NUMINAMATH_CALUDE_bus_passengers_l1086_108620

theorem bus_passengers (initial_passengers : ℕ) : 
  initial_passengers + 16 - 22 + 5 = 49 → initial_passengers = 50 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l1086_108620


namespace NUMINAMATH_CALUDE_sum_of_consecutive_odd_primes_has_three_factors_l1086_108628

/-- Two natural numbers are consecutive primes if they are both prime and there is no prime between them. -/
def ConsecutivePrimes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ ∀ k, p < k → k < q → ¬Nat.Prime k

/-- A natural number is the product of at least three factors greater than 1 if it can be written as the product of three or more natural numbers, each greater than 1. -/
def ProductOfAtLeastThreeFactors (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ n = a * b * c

/-- For any two consecutive odd prime numbers, their sum is a product of at least three positive integers greater than 1. -/
theorem sum_of_consecutive_odd_primes_has_three_factors (p q : ℕ) :
  ConsecutivePrimes p q → Odd p → Odd q → ProductOfAtLeastThreeFactors (p + q) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_odd_primes_has_three_factors_l1086_108628


namespace NUMINAMATH_CALUDE_smallest_k_inequality_l1086_108659

theorem smallest_k_inequality (x y z : ℝ) :
  ∃ (k : ℝ), k = 3 ∧ (x^2 + y^2 + z^2)^2 ≤ k * (x^4 + y^4 + z^4) ∧
  ∀ (k' : ℝ), (∀ (a b c : ℝ), (a^2 + b^2 + c^2)^2 ≤ k' * (a^4 + b^4 + c^4)) → k' ≥ k :=
sorry

end NUMINAMATH_CALUDE_smallest_k_inequality_l1086_108659


namespace NUMINAMATH_CALUDE_pens_left_in_jar_l1086_108635

/-- The number of pens left in a jar after removing some pens -/
def pens_left (initial_blue initial_black initial_red removed_blue removed_black : ℕ) : ℕ :=
  (initial_blue - removed_blue) + (initial_black - removed_black) + initial_red

/-- Theorem stating the number of pens left in the jar -/
theorem pens_left_in_jar : pens_left 9 21 6 4 7 = 25 := by
  sorry

end NUMINAMATH_CALUDE_pens_left_in_jar_l1086_108635


namespace NUMINAMATH_CALUDE_plane_perpendicular_parallel_implies_perpendicular_l1086_108689

-- Define the plane type
structure Plane where
  -- Add necessary fields or leave it abstract

-- Define the perpendicular and parallel relations
def perpendicular (p q : Plane) : Prop := sorry

def parallel (p q : Plane) : Prop := sorry

-- State the theorem
theorem plane_perpendicular_parallel_implies_perpendicular 
  (α β γ : Plane) 
  (h1 : α ≠ β) (h2 : β ≠ γ) (h3 : α ≠ γ)
  (h4 : perpendicular α β) 
  (h5 : parallel β γ) : 
  perpendicular α γ := by sorry

end NUMINAMATH_CALUDE_plane_perpendicular_parallel_implies_perpendicular_l1086_108689


namespace NUMINAMATH_CALUDE_alternating_dodecagon_area_l1086_108653

/-- An equilateral 12-gon with alternating interior angles -/
structure AlternatingDodecagon where
  side_length : ℝ
  interior_angles : Fin 12 → ℝ
  is_equilateral : ∀ i : Fin 12, side_length > 0
  angle_pattern : ∀ i : Fin 12, interior_angles i = 
    if i % 3 = 0 ∨ i % 3 = 1 then 90 else 270

/-- The area of the alternating dodecagon -/
noncomputable def area (d : AlternatingDodecagon) : ℝ := sorry

/-- Theorem stating that the area of the specific alternating dodecagon is 500 -/
theorem alternating_dodecagon_area :
  ∀ d : AlternatingDodecagon, d.side_length = 10 → area d = 500 := by sorry

end NUMINAMATH_CALUDE_alternating_dodecagon_area_l1086_108653


namespace NUMINAMATH_CALUDE_choose_four_from_nine_l1086_108669

theorem choose_four_from_nine (n : ℕ) (k : ℕ) : n = 9 → k = 4 → Nat.choose n k = 126 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_nine_l1086_108669


namespace NUMINAMATH_CALUDE_company_females_count_l1086_108675

theorem company_females_count (total_employees : ℕ) 
  (advanced_degrees : ℕ) (males_college_only : ℕ) (females_advanced : ℕ) 
  (h1 : total_employees = 148)
  (h2 : advanced_degrees = 78)
  (h3 : males_college_only = 31)
  (h4 : females_advanced = 53) :
  total_employees - advanced_degrees - males_college_only + females_advanced = 92 :=
by sorry

end NUMINAMATH_CALUDE_company_females_count_l1086_108675


namespace NUMINAMATH_CALUDE_concert_drive_distance_l1086_108650

/-- Calculates the remaining distance to drive given the total distance and the distance already driven. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Theorem stating that for a total distance of 78 miles and a driven distance of 32 miles, 
    the remaining distance is 46 miles. -/
theorem concert_drive_distance : remaining_distance 78 32 = 46 := by
  sorry

end NUMINAMATH_CALUDE_concert_drive_distance_l1086_108650


namespace NUMINAMATH_CALUDE_multiple_of_six_last_digit_l1086_108632

theorem multiple_of_six_last_digit (n : ℕ) : 
  n ≥ 85670 ∧ n < 85680 ∧ n % 6 = 0 → n = 85676 := by sorry

end NUMINAMATH_CALUDE_multiple_of_six_last_digit_l1086_108632


namespace NUMINAMATH_CALUDE_third_shift_members_l1086_108611

theorem third_shift_members (first_shift : ℕ) (second_shift : ℕ) (first_participation : ℚ)
  (second_participation : ℚ) (third_participation : ℚ) (total_participation : ℚ)
  (h1 : first_shift = 60)
  (h2 : second_shift = 50)
  (h3 : first_participation = 20 / 100)
  (h4 : second_participation = 40 / 100)
  (h5 : third_participation = 10 / 100)
  (h6 : total_participation = 24 / 100) :
  ∃ (third_shift : ℕ),
    (first_shift * first_participation + second_shift * second_participation + third_shift * third_participation) /
    (first_shift + second_shift + third_shift) = total_participation ∧
    third_shift = 40 := by
  sorry

end NUMINAMATH_CALUDE_third_shift_members_l1086_108611


namespace NUMINAMATH_CALUDE_burger_problem_l1086_108636

theorem burger_problem (total_burgers : ℕ) (total_cost : ℚ) (single_cost : ℚ) (double_cost : ℚ) 
  (h1 : total_burgers = 50)
  (h2 : total_cost = 64.5)
  (h3 : single_cost = 1)
  (h4 : double_cost = 1.5) :
  ∃ (single_count double_count : ℕ),
    single_count + double_count = total_burgers ∧
    single_cost * single_count + double_cost * double_count = total_cost ∧
    double_count = 29 := by
  sorry

end NUMINAMATH_CALUDE_burger_problem_l1086_108636


namespace NUMINAMATH_CALUDE_triangle_inequality_from_squared_sum_l1086_108690

theorem triangle_inequality_from_squared_sum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ineq : (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4)) :
  a + b > c ∧ b + c > a ∧ c + a > b := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_from_squared_sum_l1086_108690


namespace NUMINAMATH_CALUDE_range_of_fraction_l1086_108646

theorem range_of_fraction (x y : ℝ) (h : x^2 + y^2 + 2*x = 0) :
  ∃ (t : ℝ), y / (x - 1) = t ∧ -Real.sqrt 3 / 3 ≤ t ∧ t ≤ Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l1086_108646


namespace NUMINAMATH_CALUDE_age_difference_l1086_108601

/-- Given Billy's current age and the ratio of my age to Billy's, 
    prove the difference between our ages. -/
theorem age_difference (billy_age : ℕ) (age_ratio : ℕ) : 
  billy_age = 4 → age_ratio = 4 → age_ratio * billy_age - billy_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1086_108601


namespace NUMINAMATH_CALUDE_ravi_jump_multiple_l1086_108694

def jump_heights : List ℝ := [23, 27, 28]
def ravi_jump : ℝ := 39

theorem ravi_jump_multiple :
  ravi_jump / (jump_heights.sum / jump_heights.length) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ravi_jump_multiple_l1086_108694


namespace NUMINAMATH_CALUDE_sandy_paint_area_l1086_108668

/-- The area Sandy needs to paint on a wall with a decorative region -/
theorem sandy_paint_area (wall_height wall_length decor_height decor_length : ℝ)
  (h_wall_height : wall_height = 10)
  (h_wall_length : wall_length = 15)
  (h_decor_height : decor_height = 3)
  (h_decor_length : decor_length = 5) :
  wall_height * wall_length - decor_height * decor_length = 135 := by
  sorry

end NUMINAMATH_CALUDE_sandy_paint_area_l1086_108668


namespace NUMINAMATH_CALUDE_sequence_properties_l1086_108664

def S (n : ℕ) (a : ℕ → ℝ) : ℝ := a n + n^2 - 1

def b_relation (n : ℕ) (a b : ℕ → ℝ) : Prop :=
  3^n * b (n+1) = (n+1) * a (n+1) - n * a n

theorem sequence_properties (a b : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n, S n a = a n + n^2 - 1) →
  (∀ n, b_relation n a b) →
  b 1 = 3 →
  (∀ n, a n = 2*n + 1) ∧
  (∀ n, b n = (4*n - 1) / 3^(n-1)) ∧
  (∀ n, T n = 15/2 - (4*n + 5) / (2 * 3^(n-1))) ∧
  (∀ n > 3, T n ≥ 7) ∧
  (T 3 < 7) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1086_108664


namespace NUMINAMATH_CALUDE_correct_oranges_count_l1086_108631

/-- Calculates the number of oranges needed to reach a desired total fruit count -/
def oranges_needed (total_desired : ℕ) (apples : ℕ) (bananas : ℕ) : ℕ :=
  total_desired - (apples + bananas)

theorem correct_oranges_count : oranges_needed 12 3 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_oranges_count_l1086_108631


namespace NUMINAMATH_CALUDE_marathon_distance_yards_l1086_108678

/-- Represents the distance of a marathon in miles and yards -/
structure MarathonDistance :=
  (miles : ℕ)
  (yards : ℕ)

/-- Represents a total distance in miles and yards -/
structure TotalDistance :=
  (miles : ℕ)
  (yards : ℕ)

/-- The number of yards in a mile -/
def yardsPerMile : ℕ := 1760

/-- The distance of a single marathon -/
def marathonDistance : MarathonDistance :=
  { miles := 26, yards := 395 }

/-- The number of marathons Leila has run -/
def marathonCount : ℕ := 8

/-- Calculates the total distance run in multiple marathons -/
def totalMarathonDistance (marathonDist : MarathonDistance) (count : ℕ) : TotalDistance :=
  { miles := marathonDist.miles * count,
    yards := marathonDist.yards * count }

/-- Converts a TotalDistance to a normalized form where yards < yardsPerMile -/
def normalizeDistance (dist : TotalDistance) : TotalDistance :=
  { miles := dist.miles + dist.yards / yardsPerMile,
    yards := dist.yards % yardsPerMile }

theorem marathon_distance_yards :
  (normalizeDistance (totalMarathonDistance marathonDistance marathonCount)).yards = 1400 := by
  sorry

end NUMINAMATH_CALUDE_marathon_distance_yards_l1086_108678


namespace NUMINAMATH_CALUDE_vikkis_take_home_pay_is_correct_l1086_108665

/-- Calculates Vikki's take-home pay after all deductions --/
def vikkis_take_home_pay (hours_worked : ℕ) (hourly_rate : ℚ) 
  (federal_tax_rate_low : ℚ) (federal_tax_rate_high : ℚ) (federal_tax_threshold : ℚ)
  (state_tax_rate : ℚ) (retirement_rate : ℚ) (insurance_rate : ℚ) (union_dues : ℚ) : ℚ :=
  let gross_earnings := hours_worked * hourly_rate
  let federal_tax_low := min federal_tax_threshold gross_earnings * federal_tax_rate_low
  let federal_tax_high := max 0 (gross_earnings - federal_tax_threshold) * federal_tax_rate_high
  let state_tax := gross_earnings * state_tax_rate
  let retirement := gross_earnings * retirement_rate
  let insurance := gross_earnings * insurance_rate
  let total_deductions := federal_tax_low + federal_tax_high + state_tax + retirement + insurance + union_dues
  gross_earnings - total_deductions

/-- Theorem stating that Vikki's take-home pay is $328.48 --/
theorem vikkis_take_home_pay_is_correct : 
  vikkis_take_home_pay 42 12 (15/100) (22/100) 300 (7/100) (6/100) (3/100) 5 = 328.48 := by
  sorry

end NUMINAMATH_CALUDE_vikkis_take_home_pay_is_correct_l1086_108665


namespace NUMINAMATH_CALUDE_max_quartets_correct_max_quartets_5x5_l1086_108663

def max_quartets (m n : ℕ) : ℕ :=
  if m % 2 = 0 ∧ n % 2 = 0 then
    m * n / 4
  else if (m % 2 = 0 ∧ n % 2 = 1) ∨ (m % 2 = 1 ∧ n % 2 = 0) then
    m * (n - 1) / 4
  else
    (m * (n - 1) - 2) / 4

theorem max_quartets_correct (m n : ℕ) :
  max_quartets m n = 
    if m % 2 = 0 ∧ n % 2 = 0 then
      m * n / 4
    else if (m % 2 = 0 ∧ n % 2 = 1) ∨ (m % 2 = 1 ∧ n % 2 = 0) then
      m * (n - 1) / 4
    else
      (m * (n - 1) - 2) / 4 :=
by sorry

theorem max_quartets_5x5 : max_quartets 5 5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_quartets_correct_max_quartets_5x5_l1086_108663


namespace NUMINAMATH_CALUDE_least_n_for_adjacent_probability_l1086_108693

/-- The probability of two randomly selected unit squares being adjacent in an n × n grid. -/
def adjacent_probability (n : ℕ) : ℚ :=
  (4 * n^2 - 4 * n + 8) / (n^2 * (n^2 - 1))

/-- Theorem stating that 90 is the least positive integer n such that the probability
    of two randomly selected unit squares being adjacent in an n × n grid is less than 1/2015. -/
theorem least_n_for_adjacent_probability : ∀ k : ℕ, k > 0 →
  (∀ n : ℕ, n ≥ k → adjacent_probability n < 1 / 2015) →
  k ≥ 90 :=
sorry

end NUMINAMATH_CALUDE_least_n_for_adjacent_probability_l1086_108693


namespace NUMINAMATH_CALUDE_wallet_theorem_l1086_108647

def wallet_problem (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ) (twenty_dollar_bills : ℕ) : Prop :=
  let total_amount : ℕ := 150
  let ten_dollar_amount : ℕ := 50
  let twenty_dollar_count : ℕ := 4
  (5 * five_dollar_bills + 10 * ten_dollar_bills + 20 * twenty_dollar_bills = total_amount) ∧
  (10 * ten_dollar_bills = ten_dollar_amount) ∧
  (twenty_dollar_bills = twenty_dollar_count) ∧
  (five_dollar_bills + ten_dollar_bills + twenty_dollar_bills = 13)

theorem wallet_theorem :
  ∃ (five_dollar_bills ten_dollar_bills twenty_dollar_bills : ℕ),
    wallet_problem five_dollar_bills ten_dollar_bills twenty_dollar_bills :=
by
  sorry

end NUMINAMATH_CALUDE_wallet_theorem_l1086_108647


namespace NUMINAMATH_CALUDE_numera_transaction_l1086_108692

/-- Represents a number in base s --/
def BaseS (digits : List Nat) (s : Nat) : Nat :=
  digits.foldr (fun d acc => d + s * acc) 0

/-- The transaction in the galaxy of Numera --/
theorem numera_transaction (s : Nat) : 
  s > 1 →  -- s must be greater than 1 to be a valid base
  BaseS [6, 3, 0] s + BaseS [2, 5, 0] s = BaseS [8, 8, 0] s →  -- cost of gadgets
  BaseS [4, 7, 0] s = BaseS [1, 0, 0, 0] s * 2 - BaseS [8, 8, 0] s →  -- change received
  s = 5 := by
  sorry

end NUMINAMATH_CALUDE_numera_transaction_l1086_108692


namespace NUMINAMATH_CALUDE_function_decomposition_l1086_108670

theorem function_decomposition (f : ℝ → ℝ) :
  ∃ (g h : ℝ → ℝ),
    (∀ x, f x = g x + h x) ∧
    (∀ x, g x = g (-x)) ∧
    (∀ x, h (1 + x) = h (1 - x)) := by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_l1086_108670


namespace NUMINAMATH_CALUDE_trains_meeting_time_l1086_108684

/-- Two trains meeting problem -/
theorem trains_meeting_time (distance : ℝ) (express_speed : ℝ) (speed_difference : ℝ) : 
  distance = 390 →
  express_speed = 80 →
  speed_difference = 30 →
  (distance / (express_speed + (express_speed - speed_difference))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trains_meeting_time_l1086_108684


namespace NUMINAMATH_CALUDE_triangle_side_length_l1086_108688

/-- Given a triangle ABC with circumradius R, prove that if cos B and cos A are known,
    then the length of side c can be determined. -/
theorem triangle_side_length (A B C : Real) (R : Real) (h1 : R = 5/6)
  (h2 : Real.cos B = 3/5) (h3 : Real.cos A = 12/13) :
  2 * R * Real.sin (A + B) = 21/13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1086_108688


namespace NUMINAMATH_CALUDE_total_pet_owners_l1086_108697

/-- The number of people who own only dogs -/
def only_dogs : ℕ := 15

/-- The number of people who own only cats -/
def only_cats : ℕ := 10

/-- The number of people who own only cats and dogs -/
def cats_and_dogs : ℕ := 5

/-- The number of people who own cats, dogs, and snakes -/
def cats_dogs_snakes : ℕ := 3

/-- The total number of snakes -/
def total_snakes : ℕ := 29

/-- Theorem stating the total number of pet owners -/
theorem total_pet_owners : 
  only_dogs + only_cats + cats_and_dogs + cats_dogs_snakes = 33 := by
  sorry


end NUMINAMATH_CALUDE_total_pet_owners_l1086_108697


namespace NUMINAMATH_CALUDE_exists_k_configuration_l1086_108640

/-- A configuration of black cells on an infinite white checker plane. -/
structure BlackCellConfiguration where
  cells : Set (ℤ × ℤ)
  finite : Set.Finite cells

/-- A line on the infinite plane (vertical, horizontal, or diagonal). -/
inductive Line
  | Vertical (x : ℤ) : Line
  | Horizontal (y : ℤ) : Line
  | Diagonal (a b c : ℤ) : Line

/-- The number of black cells on a given line for a given configuration. -/
def blackCellsOnLine (config : BlackCellConfiguration) (line : Line) : ℕ :=
  sorry

/-- A configuration satisfies the k-condition if every line contains
    either k black cells or no black cells. -/
def satisfiesKCondition (config : BlackCellConfiguration) (k : ℕ+) : Prop :=
  ∀ line, blackCellsOnLine config line = k ∨ blackCellsOnLine config line = 0

/-- For any positive integer k, there exists a configuration of black cells
    satisfying the k-condition. -/
theorem exists_k_configuration (k : ℕ+) :
  ∃ (config : BlackCellConfiguration), satisfiesKCondition config k :=
sorry

end NUMINAMATH_CALUDE_exists_k_configuration_l1086_108640


namespace NUMINAMATH_CALUDE_basketball_players_l1086_108634

theorem basketball_players (total : ℕ) (hockey : ℕ) (neither : ℕ) (both : ℕ) 
  (h_total : total = 25)
  (h_hockey : hockey = 15)
  (h_neither : neither = 4)
  (h_both : both = 10) :
  ∃ basketball : ℕ, basketball = 16 :=
by sorry

end NUMINAMATH_CALUDE_basketball_players_l1086_108634


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1086_108627

-- Define the quadratic polynomial
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality (a b c : ℝ) 
  (h : ∀ x, quadratic a b c x < 0) :
  b / a < c / a + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1086_108627


namespace NUMINAMATH_CALUDE_speeding_ticket_problem_l1086_108603

theorem speeding_ticket_problem (total_motorists : ℕ) 
  (h1 : total_motorists > 0) 
  (exceed_limit : ℕ) 
  (receive_tickets : ℕ) 
  (h2 : exceed_limit = total_motorists * 25 / 100) 
  (h3 : receive_tickets = total_motorists * 20 / 100) :
  (exceed_limit - receive_tickets) * 100 / exceed_limit = 20 := by
sorry

end NUMINAMATH_CALUDE_speeding_ticket_problem_l1086_108603


namespace NUMINAMATH_CALUDE_ferry_tourists_sum_l1086_108630

/-- The number of trips the ferry makes in a day -/
def num_trips : ℕ := 9

/-- The initial number of tourists on the first trip -/
def initial_tourists : ℕ := 120

/-- The decrease in number of tourists for each subsequent trip -/
def tourist_decrease : ℤ := 2

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℕ) * d)

theorem ferry_tourists_sum :
  arithmetic_sum initial_tourists (-tourist_decrease) num_trips = 1008 := by
  sorry

end NUMINAMATH_CALUDE_ferry_tourists_sum_l1086_108630


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l1086_108662

theorem simplify_nested_roots : 
  (65536 : ℝ) = 2^16 →
  (((1 / 65536)^(1/2))^(1/3))^(1/4) = 1 / (4^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l1086_108662


namespace NUMINAMATH_CALUDE_square_of_negative_sum_l1086_108607

theorem square_of_negative_sum (x y : ℝ) : (-x - y)^2 = x^2 + 2*x*y + y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_sum_l1086_108607


namespace NUMINAMATH_CALUDE_black_and_white_films_count_l1086_108672

theorem black_and_white_films_count 
  (B : ℕ) -- number of black-and-white films
  (x y : ℚ) -- parameters for selection percentage and color films
  (h1 : y / x > 0) -- ensure y/x is positive
  (h2 : y > 0) -- ensure y is positive
  (h3 : (4 * y) / ((y / x * B / 100) + 4 * y) = 10 / 11) -- fraction of selected color films
  : B = 40 * x := by
sorry

end NUMINAMATH_CALUDE_black_and_white_films_count_l1086_108672


namespace NUMINAMATH_CALUDE_max_value_fraction_l1086_108645

theorem max_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -3 ∧ 1 ≤ y' ∧ y' ≤ 3 → (x' + y') / x' ≤ (x + y) / x) →
  (x + y) / x = 0.4 := by
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1086_108645


namespace NUMINAMATH_CALUDE_largest_k_for_inequality_l1086_108699

theorem largest_k_for_inequality (a b c : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c) 
  (h3 : a * b + b * c + c * a = 0) 
  (h4 : a * b * c = 1) :
  (∀ k : ℝ, (∀ a b c : ℝ, a ≤ b → b ≤ c → a * b + b * c + c * a = 0 → a * b * c = 1 → 
    |a + b| ≥ k * |c|) → k ≤ 4) ∧
  (∀ a b c : ℝ, a ≤ b → b ≤ c → a * b + b * c + c * a = 0 → a * b * c = 1 → 
    |a + b| ≥ 4 * |c|) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_inequality_l1086_108699


namespace NUMINAMATH_CALUDE_problem_1_l1086_108641

theorem problem_1 (m : ℝ) (h : -m^2 = m) : m^2 + m + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1086_108641


namespace NUMINAMATH_CALUDE_secretary_typing_arrangements_l1086_108652

def remaining_letters : Finset Nat := {1, 2, 3, 4, 6, 7, 8, 10}

def possible_arrangements (s : Finset Nat) : Nat :=
  Finset.card s + 2

theorem secretary_typing_arrangements :
  (Finset.powerset remaining_letters).sum (fun s => Nat.choose 8 (Finset.card s) * possible_arrangements s) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_secretary_typing_arrangements_l1086_108652


namespace NUMINAMATH_CALUDE_positive_X_value_l1086_108609

-- Define the # relation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- Theorem statement
theorem positive_X_value (X : ℝ) :
  (hash X 7 = 170) → (X = 11 ∨ X = -11) :=
by sorry

end NUMINAMATH_CALUDE_positive_X_value_l1086_108609


namespace NUMINAMATH_CALUDE_m_equals_two_sufficient_not_necessary_l1086_108610

def A (m : ℝ) : Set ℝ := {1, m^2}
def B : Set ℝ := {2, 4}

theorem m_equals_two_sufficient_not_necessary :
  (∀ m : ℝ, m = 2 → A m ∩ B = {4}) ∧
  (∃ m : ℝ, m ≠ 2 ∧ A m ∩ B = {4}) :=
sorry

end NUMINAMATH_CALUDE_m_equals_two_sufficient_not_necessary_l1086_108610


namespace NUMINAMATH_CALUDE_water_speed_calculation_l1086_108698

theorem water_speed_calculation (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  still_water_speed = 12 →
  distance = 8 →
  time = 4 →
  ∃ water_speed : ℝ, water_speed = still_water_speed - (distance / time) :=
by
  sorry

end NUMINAMATH_CALUDE_water_speed_calculation_l1086_108698


namespace NUMINAMATH_CALUDE_geometric_sequence_identity_l1086_108696

/-- 
Given a geometric sequence and three of its terms L, M, N at positions l, m, n respectively,
prove that L^(m-n) * M^(n-l) * N^(l-m) = 1.
-/
theorem geometric_sequence_identity 
  {α : Type*} [Field α] 
  (a : ℕ → α) 
  (q : α) 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (l m n : ℕ) :
  (a l) ^ (m - n) * (a m) ^ (n - l) * (a n) ^ (l - m) = 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_identity_l1086_108696


namespace NUMINAMATH_CALUDE_sum_of_digits_8_to_1002_l1086_108643

theorem sum_of_digits_8_to_1002 :
  let n := 8^1002
  let tens_digit := (n / 10) % 10
  let units_digit := n % 10
  tens_digit + units_digit = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_8_to_1002_l1086_108643


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l1086_108679

theorem unique_four_digit_number : ∃! n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧
  (n / 10) % 10 = n % 10 + 2 ∧
  (n / 1000) = (n / 100) % 10 + 2 ∧
  n = 9742 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l1086_108679


namespace NUMINAMATH_CALUDE_soccer_league_games_l1086_108687

/-- Proves that in a soccer league with 11 teams and 55 total games, each team plays others 2 times -/
theorem soccer_league_games (num_teams : ℕ) (total_games : ℕ) (games_per_pair : ℕ) : 
  num_teams = 11 → 
  total_games = 55 → 
  total_games = (num_teams * (num_teams - 1) * games_per_pair) / 2 → 
  games_per_pair = 2 := by
sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1086_108687


namespace NUMINAMATH_CALUDE_distance_from_origin_and_point_specific_distances_l1086_108667

theorem distance_from_origin_and_point (d : ℝ) (p : ℝ) :
  -- A point at distance d from the origin represents either d or -d
  (∃ x : ℝ, x = d ∨ x = -d ∧ |x| = d) ∧
  -- A point at distance d from p represents either p + d or p - d
  (∃ y : ℝ, y = p + d ∨ y = p - d ∧ |y - p| = d) :=
by
  sorry

-- Specific instances for the given problem
theorem specific_distances :
  -- A point at distance √5 from the origin represents either √5 or -√5
  (∃ x : ℝ, x = Real.sqrt 5 ∨ x = -Real.sqrt 5 ∧ |x| = Real.sqrt 5) ∧
  -- A point at distance 2√5 from √5 represents either 3√5 or -√5
  (∃ y : ℝ, y = 3 * Real.sqrt 5 ∨ y = -Real.sqrt 5 ∧ |y - Real.sqrt 5| = 2 * Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_and_point_specific_distances_l1086_108667


namespace NUMINAMATH_CALUDE_fraction_of_married_women_l1086_108683

/-- Given a company with employees, prove that 3/4 of women are married under specific conditions -/
theorem fraction_of_married_women (total : ℕ) (h_total_pos : total > 0) : 
  let women := (64 : ℚ) / 100 * total
  let married := (60 : ℚ) / 100 * total
  let men := total - women
  let single_men := (2 : ℚ) / 3 * men
  let married_men := men - single_men
  let married_women := married - married_men
  married_women / women = (3 : ℚ) / 4 := by
  sorry


end NUMINAMATH_CALUDE_fraction_of_married_women_l1086_108683


namespace NUMINAMATH_CALUDE_gcd_count_for_product_600_l1086_108626

theorem gcd_count_for_product_600 : 
  ∃ (S : Finset Nat), 
    (∀ d ∈ S, ∃ a b : Nat, 
      gcd a b = d ∧ Nat.lcm a b * d = 600) ∧
    (∀ d : Nat, (∃ a b : Nat, 
      gcd a b = d ∧ Nat.lcm a b * d = 600) → d ∈ S) ∧
    S.card = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_600_l1086_108626


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1086_108616

theorem quadratic_roots_relation (m p q : ℝ) (hm : m ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0) :
  (∃ s₁ s₂ : ℝ, (s₁ + s₂ = -q ∧ s₁ * s₂ = m) ∧
               (3 * s₁ + 3 * s₂ = -m ∧ 9 * s₁ * s₂ = p)) →
  p / q = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1086_108616


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l1086_108661

theorem sum_of_squares_problem (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 52)
  (h_sum_products : x*y + y*z + z*x = 24) :
  x + y + z = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l1086_108661


namespace NUMINAMATH_CALUDE_coefficient_x3_is_80_l1086_108655

/-- The coefficient of x^3 in the expansion of (1+2x)^5 -/
def coefficient_x3 : ℕ :=
  Nat.choose 5 3 * 2^3

/-- Theorem stating that the coefficient of x^3 in (1+2x)^5 is 80 -/
theorem coefficient_x3_is_80 : coefficient_x3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3_is_80_l1086_108655


namespace NUMINAMATH_CALUDE_largest_n_value_l1086_108612

def base_8_to_10 (a b c : ℕ) : ℕ := 64 * a + 8 * b + c

def base_9_to_10 (c b a : ℕ) : ℕ := 81 * c + 9 * b + a

theorem largest_n_value (n : ℕ) (a b c : ℕ) :
  (n > 0) →
  (a < 8 ∧ b < 8 ∧ c < 8) →
  (a ≤ 8 ∧ b ≤ 8 ∧ c ≤ 8) →
  (n = base_8_to_10 a b c) →
  (n = base_9_to_10 c b a) →
  (∀ m, m > 0 ∧ 
    (∃ x y z, x < 8 ∧ y < 8 ∧ z < 8 ∧ m = base_8_to_10 x y z) ∧
    (∃ x y z, x ≤ 8 ∧ y ≤ 8 ∧ z ≤ 8 ∧ m = base_9_to_10 z y x) →
    m ≤ n) →
  n = 511 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_value_l1086_108612
