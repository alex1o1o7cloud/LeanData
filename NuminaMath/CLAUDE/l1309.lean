import Mathlib

namespace NUMINAMATH_CALUDE_total_candies_l1309_130958

theorem total_candies (linda_candies chloe_candies : ℕ) 
  (h1 : linda_candies = 34) 
  (h2 : chloe_candies = 28) : 
  linda_candies + chloe_candies = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l1309_130958


namespace NUMINAMATH_CALUDE_kiana_and_twins_ages_l1309_130979

theorem kiana_and_twins_ages (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 162 → a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_kiana_and_twins_ages_l1309_130979


namespace NUMINAMATH_CALUDE_max_q_minus_r_l1309_130930

theorem max_q_minus_r (q r : ℕ+) (h : 961 = 23 * q + r) : q - r ≤ 23 := by
  sorry

end NUMINAMATH_CALUDE_max_q_minus_r_l1309_130930


namespace NUMINAMATH_CALUDE_annual_pension_formula_l1309_130922

/-- Represents an employee's pension calculation -/
structure PensionCalculation where
  x : ℝ  -- Years of service
  c : ℝ  -- Additional years scenario 1
  d : ℝ  -- Additional years scenario 2
  r : ℝ  -- Pension increase for scenario 1
  s : ℝ  -- Pension increase for scenario 2
  h1 : c ≠ d  -- Assumption that c and d are different

/-- The pension is proportional to years of service squared -/
def pension_proportional (p : PensionCalculation) (k : ℝ) : Prop :=
  ∃ (base_pension : ℝ), base_pension = k * p.x^2

/-- The pension increase after c more years of service -/
def pension_increase_c (p : PensionCalculation) (k : ℝ) : Prop :=
  k * (p.x + p.c)^2 - k * p.x^2 = p.r

/-- The pension increase after d more years of service -/
def pension_increase_d (p : PensionCalculation) (k : ℝ) : Prop :=
  k * (p.x + p.d)^2 - k * p.x^2 = p.s

/-- The theorem stating the formula for the annual pension -/
theorem annual_pension_formula (p : PensionCalculation) :
  ∃ (k : ℝ), 
    pension_proportional p k ∧ 
    pension_increase_c p k ∧ 
    pension_increase_d p k → 
    k = (p.s - p.r) / (2 * p.x * (p.d - p.c) + p.d^2 - p.c^2) :=
by sorry

end NUMINAMATH_CALUDE_annual_pension_formula_l1309_130922


namespace NUMINAMATH_CALUDE_prob_divisible_by_11_is_correct_l1309_130928

/-- The probability of reaching a number divisible by 11 in the described process -/
def prob_divisible_by_11 : ℚ := 11 / 20

/-- The process of building an integer as described in the problem -/
def build_integer (start : ℕ) (stop_condition : ℕ → Bool) : ℕ → ℚ := sorry

/-- The main theorem stating that the probability of reaching a number divisible by 11 is 11/20 -/
theorem prob_divisible_by_11_is_correct :
  build_integer 9 (λ n => n % 11 = 0 ∨ n % 11 = 1) 0 = prob_divisible_by_11 := by sorry

end NUMINAMATH_CALUDE_prob_divisible_by_11_is_correct_l1309_130928


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_count_l1309_130944

theorem diophantine_equation_solutions_count : 
  ∃ (S : Finset ℤ), 
    (∀ p ∈ S, 1 ≤ p ∧ p ≤ 15) ∧ 
    (∀ p ∈ S, ∃ q : ℤ, p * q - 8 * p - 3 * q = 15) ∧
    S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_count_l1309_130944


namespace NUMINAMATH_CALUDE_box_dimensions_l1309_130994

theorem box_dimensions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < b) (hbc : b < c)
  (sum_ac : a + c = 17)
  (sum_ab : a + b = 13)
  (perimeter_bc : 2 * (b + c) = 40) :
  a = 5 ∧ b = 8 ∧ c = 12 := by
sorry

end NUMINAMATH_CALUDE_box_dimensions_l1309_130994


namespace NUMINAMATH_CALUDE_log_sum_inequality_l1309_130931

theorem log_sum_inequality (a b : ℝ) (h1 : 2^a = Real.pi) (h2 : 5^b = Real.pi) :
  1/a + 1/b > 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_inequality_l1309_130931


namespace NUMINAMATH_CALUDE_sphere_volume_equal_surface_area_l1309_130974

theorem sphere_volume_equal_surface_area (r : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_equal_surface_area_l1309_130974


namespace NUMINAMATH_CALUDE_a_8_value_l1309_130991

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem a_8_value (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 6 + a 10 = -6) →
  (a 6 * a 10 = 2) →
  (a 6 < 0) →
  (a 10 < 0) →
  a 8 = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_a_8_value_l1309_130991


namespace NUMINAMATH_CALUDE_tysons_swimming_problem_l1309_130915

/-- Tyson's swimming problem -/
theorem tysons_swimming_problem 
  (lake_speed : ℝ) 
  (ocean_speed : ℝ) 
  (total_races : ℕ) 
  (total_time : ℝ) 
  (h1 : lake_speed = 3)
  (h2 : ocean_speed = 2.5)
  (h3 : total_races = 10)
  (h4 : total_time = 11)
  (h5 : total_races % 2 = 0) -- Ensures even number of races for equal distribution
  : ∃ (race_distance : ℝ), 
    race_distance = 3 ∧ 
    (total_races / 2 : ℝ) * (race_distance / lake_speed + race_distance / ocean_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_tysons_swimming_problem_l1309_130915


namespace NUMINAMATH_CALUDE_max_square_plots_l1309_130975

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  length : ℕ
  width : ℕ

/-- Represents the available fence length for internal fencing -/
def available_fence : ℕ := 2200

/-- Calculates the number of square plots given the number of plots in a column -/
def num_plots (n : ℕ) : ℕ := n * (11 * n / 6)

/-- Calculates the required fence length for a given number of plots in a column -/
def required_fence (n : ℕ) : ℕ := 187 * n - 132

/-- The maximum number of square plots that can partition the field -/
def max_plots : ℕ := 264

/-- Theorem stating the maximum number of square plots -/
theorem max_square_plots (field : FieldDimensions) 
  (h1 : field.length = 36) 
  (h2 : field.width = 66) : 
  (∀ n : ℕ, num_plots n ≤ max_plots ∧ required_fence n ≤ available_fence) ∧ 
  (∃ n : ℕ, num_plots n = max_plots ∧ required_fence n ≤ available_fence) :=
sorry

end NUMINAMATH_CALUDE_max_square_plots_l1309_130975


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_is_3_l1309_130904

def chocolate_bar_cost (total_cost : ℕ) (num_chocolate_bars : ℕ) (num_gummy_packs : ℕ) (num_chip_bags : ℕ) (gummy_cost : ℕ) (chip_cost : ℕ) : ℕ :=
  (total_cost - (num_gummy_packs * gummy_cost + num_chip_bags * chip_cost)) / num_chocolate_bars

theorem chocolate_bar_cost_is_3 :
  chocolate_bar_cost 150 10 10 20 2 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_is_3_l1309_130904


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l1309_130969

/-- Calculates the total cost of purchasing a puppy and related items. -/
def total_cost (puppy_cost : ℚ) (food_consumption_per_day : ℚ) (food_duration_weeks : ℕ) 
  (food_cost_per_bag : ℚ) (food_amount_per_bag : ℚ) (leash_cost : ℚ) (collar_cost : ℚ) 
  (dog_bed_cost : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let food_total_consumption := food_consumption_per_day * (food_duration_weeks * 7)
  let food_bags_needed := (food_total_consumption / food_amount_per_bag).ceil
  let food_cost := food_bags_needed * food_cost_per_bag
  let collar_discounted := collar_cost * (1 - 0.1)
  let taxable_items_cost := leash_cost + collar_discounted + dog_bed_cost
  let tax_amount := taxable_items_cost * sales_tax_rate
  puppy_cost + food_cost + taxable_items_cost + tax_amount

/-- Theorem stating that the total cost is $211.85 given the specified conditions. -/
theorem total_cost_is_correct : 
  total_cost 150 (1/3) 6 2 (7/2) 15 12 25 (6/100) = 21185/100 := by
  sorry


end NUMINAMATH_CALUDE_total_cost_is_correct_l1309_130969


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1309_130962

theorem trigonometric_identities (α : Real) (h : Real.tan (α / 2) = 3) :
  (Real.tan (α + Real.pi / 3) = (48 - 4 * Real.sqrt 3) / 11) ∧
  ((Real.sin α + 2 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = -5 / 17) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1309_130962


namespace NUMINAMATH_CALUDE_original_room_length_l1309_130978

theorem original_room_length :
  ∀ (x : ℝ),
  (4 * ((x + 2) * 20) + 2 * ((x + 2) * 20) = 1800) →
  x = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_original_room_length_l1309_130978


namespace NUMINAMATH_CALUDE_three_solutions_l1309_130961

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The number of positive integers n satisfying n + S(n) + S(S(n)) = 2500 -/
def count_solutions : ℕ := sorry

/-- Theorem stating that there are exactly 3 solutions -/
theorem three_solutions : count_solutions = 3 := by sorry

end NUMINAMATH_CALUDE_three_solutions_l1309_130961


namespace NUMINAMATH_CALUDE_b_over_c_value_l1309_130965

theorem b_over_c_value (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : a * b * c / (d * e * f) = 0.1875)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) :
  b / c = 3 := by
sorry

end NUMINAMATH_CALUDE_b_over_c_value_l1309_130965


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1309_130946

-- Define the quadratic function
def f (p : ℝ) (x : ℝ) : ℝ := x^2 + p*x - 6

-- Define the solution set
def solution_set (p : ℝ) : Set ℝ := {x : ℝ | f p x < 0}

-- Theorem statement
theorem quadratic_inequality_solution (p : ℝ) :
  solution_set p = {x : ℝ | -3 < x ∧ x < 2} → p = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1309_130946


namespace NUMINAMATH_CALUDE_sphere_triangle_distance_l1309_130948

/-- The distance from the center of a sphere to the plane of a tangent triangle -/
theorem sphere_triangle_distance (r : ℝ) (a b c : ℝ) (h_sphere : r = 10) 
  (h_triangle : a = 18 ∧ b = 18 ∧ c = 30) (h_tangent : True) : 
  ∃ d : ℝ, d = (10 * Real.sqrt 37) / 33 ∧ 
  d^2 + ((a + b + c) / 2 * (2 * a * b) / (a + b + c))^2 = r^2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_triangle_distance_l1309_130948


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_l1309_130997

theorem least_subtrahend_for_divisibility (n : ℕ) (a b c : ℕ) (h_n : n = 157632) (h_a : a = 12) (h_b : b = 18) (h_c : c = 24) :
  ∃ (k : ℕ), k = 24 ∧
  (∀ (m : ℕ), m < k → ¬(∃ (q : ℕ), n - m = q * a ∧ n - m = q * b ∧ n - m = q * c)) ∧
  (∃ (q : ℕ), n - k = q * a ∧ n - k = q * b ∧ n - k = q * c) :=
by sorry

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_l1309_130997


namespace NUMINAMATH_CALUDE_integer_sum_problem_l1309_130913

theorem integer_sum_problem (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l1309_130913


namespace NUMINAMATH_CALUDE_line_perpendicular_implies_planes_perpendicular_l1309_130953

-- Define the structure for a plane
structure Plane :=
  (points : Set Point)

-- Define the structure for a line
structure Line :=
  (points : Set Point)

-- Define the perpendicular relation between a line and a plane
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- Define the contained relation between a line and a plane
def contained (l : Line) (p : Plane) : Prop := sorry

-- Define the perpendicular relation between two planes
def perpendicularPlanes (p1 p2 : Plane) : Prop := sorry

-- Theorem statement
theorem line_perpendicular_implies_planes_perpendicular 
  (α β : Plane) (m : Line) 
  (h_distinct : α ≠ β) 
  (h_perp : perpendicular m β) 
  (h_contained : contained m α) : 
  perpendicularPlanes α β := by sorry

end NUMINAMATH_CALUDE_line_perpendicular_implies_planes_perpendicular_l1309_130953


namespace NUMINAMATH_CALUDE_range_of_a_l1309_130992

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a * x + 1 else Real.log x

-- Define symmetry about the origin
def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∃ x > 0, f (-x) = -f x

-- Theorem statement
theorem range_of_a (a : ℝ) :
  symmetric_about_origin (f a) → a ∈ Set.Iic 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1309_130992


namespace NUMINAMATH_CALUDE_rollo_guinea_pigs_l1309_130955

/-- The amount of food eaten by the first guinea pig -/
def first_guinea_pig_food : ℕ := 2

/-- The amount of food eaten by the second guinea pig -/
def second_guinea_pig_food : ℕ := 2 * first_guinea_pig_food

/-- The amount of food eaten by the third guinea pig -/
def third_guinea_pig_food : ℕ := second_guinea_pig_food + 3

/-- The total amount of food needed to feed all guinea pigs -/
def total_food_needed : ℕ := 13

/-- The number of guinea pigs Rollo has -/
def number_of_guinea_pigs : ℕ := 3

theorem rollo_guinea_pigs :
  first_guinea_pig_food + second_guinea_pig_food + third_guinea_pig_food = total_food_needed ∧
  number_of_guinea_pigs = 3 := by
  sorry

end NUMINAMATH_CALUDE_rollo_guinea_pigs_l1309_130955


namespace NUMINAMATH_CALUDE_base7_3652_equals_base10_1360_l1309_130981

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

/-- The theorem stating that 3652 in base 7 is equal to 1360 in base 10 --/
theorem base7_3652_equals_base10_1360 : base7ToBase10 3 6 5 2 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_base7_3652_equals_base10_1360_l1309_130981


namespace NUMINAMATH_CALUDE_square_eq_nine_solutions_l1309_130959

theorem square_eq_nine_solutions (x : ℝ) : x^2 = 9 ↔ x = 3 ∨ x = -3 := by sorry

end NUMINAMATH_CALUDE_square_eq_nine_solutions_l1309_130959


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l1309_130936

theorem least_addition_for_divisibility (n m : ℕ) : 
  ∃ x : ℕ, x ≤ m - 1 ∧ (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0 :=
by sorry

theorem problem_solution : 
  ∃ x : ℕ, x ≤ 22 ∧ (1054 + x) % 23 = 0 ∧ ∀ y : ℕ, y < x → (1054 + y) % 23 ≠ 0 ∧ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l1309_130936


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_one_l1309_130993

theorem negation_of_universal_positive_square_plus_one :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_one_l1309_130993


namespace NUMINAMATH_CALUDE_hoseok_number_l1309_130903

theorem hoseok_number : ∃ n : ℤ, n / 6 = 11 ∧ n = 66 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_number_l1309_130903


namespace NUMINAMATH_CALUDE_total_sheets_is_400_l1309_130957

/-- Calculates the total number of sheets of paper used for all students --/
def total_sheets (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) : ℕ :=
  num_classes * students_per_class * sheets_per_student

/-- Proves that the total number of sheets used is 400 --/
theorem total_sheets_is_400 :
  total_sheets 4 20 5 = 400 := by
  sorry

end NUMINAMATH_CALUDE_total_sheets_is_400_l1309_130957


namespace NUMINAMATH_CALUDE_fifth_sixth_sum_l1309_130977

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  sum_1_2 : a 1 + a 2 = 20
  sum_3_4 : a 3 + a 4 = 40

/-- The theorem stating that a₅ + a₆ = 80 for the given geometric sequence -/
theorem fifth_sixth_sum (seq : GeometricSequence) : seq.a 5 + seq.a 6 = 80 := by
  sorry

end NUMINAMATH_CALUDE_fifth_sixth_sum_l1309_130977


namespace NUMINAMATH_CALUDE_division_by_fraction_problem_solution_l1309_130912

theorem division_by_fraction (a b c : ℚ) (hb : b ≠ 0) (hc : c ≠ 0) :
  a / (b / c) = (a * c) / b := by
  sorry

theorem problem_solution :
  (5 : ℚ) / ((8 : ℚ) / 15) = 75 / 8 := by
  sorry

end NUMINAMATH_CALUDE_division_by_fraction_problem_solution_l1309_130912


namespace NUMINAMATH_CALUDE_max_value_on_interval_l1309_130956

-- Define the function
def f (x : ℝ) : ℝ := x^4 - 2*x^2 + 5

-- State the theorem
theorem max_value_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) (2 : ℝ) ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) (2 : ℝ) → f x ≤ f c ∧
  f c = 13 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_interval_l1309_130956


namespace NUMINAMATH_CALUDE_square_is_quadratic_l1309_130924

/-- A quadratic function is of the form y = ax² + bx + c, where a, b, and c are constants, and a ≠ 0 -/
def IsQuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x² is a quadratic function -/
theorem square_is_quadratic : IsQuadraticFunction (λ x => x^2) := by
  sorry

end NUMINAMATH_CALUDE_square_is_quadratic_l1309_130924


namespace NUMINAMATH_CALUDE_solution_implication_l1309_130960

theorem solution_implication (m n : ℝ) : 
  (2 * m + n = 8 ∧ 2 * n - m = 1) → 
  Real.sqrt (2 * m - n) = 2 := by
sorry

end NUMINAMATH_CALUDE_solution_implication_l1309_130960


namespace NUMINAMATH_CALUDE_c_range_l1309_130927

def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem c_range (a b c : ℝ) :
  (0 < f a b c (-1) ∧ f a b c (-1) = f a b c (-2) ∧ f a b c (-2) = f a b c (-3) ∧ f a b c (-3) ≤ 3) →
  (6 < c ∧ c ≤ 9) := by
  sorry

end NUMINAMATH_CALUDE_c_range_l1309_130927


namespace NUMINAMATH_CALUDE_cos_sin_expression_in_terms_of_p_q_l1309_130976

open Real

theorem cos_sin_expression_in_terms_of_p_q (x : ℝ) 
  (p : ℝ) (hp : p = (1 - cos x) * (1 + sin x))
  (q : ℝ) (hq : q = (1 + cos x) * (1 - sin x)) :
  cos x ^ 2 - cos x ^ 4 - sin (2 * x) + 2 = p * q - (p + q) := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_expression_in_terms_of_p_q_l1309_130976


namespace NUMINAMATH_CALUDE_complex_square_euler_l1309_130986

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Euler's formula -/
axiom euler_formula (x : ℝ) : Complex.exp (i * x) = Complex.cos x + i * Complex.sin x

/-- The statement to prove -/
theorem complex_square_euler (x : ℝ) :
  (Complex.cos x + i * Complex.sin x)^2 = Complex.cos (2*x) + i * Complex.sin (2*x) := by
  sorry

end NUMINAMATH_CALUDE_complex_square_euler_l1309_130986


namespace NUMINAMATH_CALUDE_fencing_cost_l1309_130920

/-- Given a rectangular field with sides in ratio 3:4 and area 10092 sq. m,
    prove that the cost of fencing at 25 paise per metre is 101.5 rupees. -/
theorem fencing_cost (length width : ℝ) (h1 : length / width = 3 / 4)
  (h2 : length * width = 10092) : 
  (2 * (length + width) * 25 / 100) = 101.5 :=
by sorry

end NUMINAMATH_CALUDE_fencing_cost_l1309_130920


namespace NUMINAMATH_CALUDE_calculation_proof_l1309_130938

theorem calculation_proof : (Real.sqrt 5 - 1)^0 + 3⁻¹ - |-(1/3)| = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1309_130938


namespace NUMINAMATH_CALUDE_racecourse_length_l1309_130911

/-- Racecourse problem -/
theorem racecourse_length
  (speed_a speed_b : ℝ)
  (head_start : ℝ)
  (h1 : speed_a = 2 * speed_b)
  (h2 : head_start = 64)
  (h3 : speed_a > 0)
  (h4 : speed_b > 0) :
  ∃ (length : ℝ), 
    length > 0 ∧
    length / speed_a = (length - head_start) / speed_b ∧
    length = 128 := by
  sorry

end NUMINAMATH_CALUDE_racecourse_length_l1309_130911


namespace NUMINAMATH_CALUDE_students_per_bus_l1309_130917

theorem students_per_bus 
  (total_students : ℕ) 
  (num_buses : ℕ) 
  (students_in_cars : ℕ) 
  (h1 : total_students = 396) 
  (h2 : num_buses = 7) 
  (h3 : students_in_cars = 4) 
  (h4 : num_buses > 0) : 
  (total_students - students_in_cars) / num_buses = 56 := by
sorry

end NUMINAMATH_CALUDE_students_per_bus_l1309_130917


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1309_130951

theorem quadratic_inequality_condition (x : ℝ) :
  (x > 2 → x^2 + 5*x - 6 > 0) ∧ 
  ¬(x^2 + 5*x - 6 > 0 → x > 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1309_130951


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1309_130990

theorem pure_imaginary_condition (m : ℝ) : 
  (Complex.I * (m^2 - 1) = m^2 + m - 2 + Complex.I * (m^2 - 1)) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1309_130990


namespace NUMINAMATH_CALUDE_prob_shortest_diagonal_nonagon_l1309_130945

/-- A regular polygon with n sides. -/
structure RegularPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The number of diagonals in a polygon with n sides. -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in a regular polygon with n sides. -/
def num_shortest_diagonals (n : ℕ) : ℕ := n

/-- The probability of selecting a shortest diagonal from all diagonals in a regular polygon. -/
def prob_shortest_diagonal (n : ℕ) : ℚ :=
  (num_shortest_diagonals n : ℚ) / (num_diagonals n : ℚ)

theorem prob_shortest_diagonal_nonagon :
  prob_shortest_diagonal 9 = 1/3 := by sorry

end NUMINAMATH_CALUDE_prob_shortest_diagonal_nonagon_l1309_130945


namespace NUMINAMATH_CALUDE_slower_train_speed_l1309_130905

/-- Proves that the speed of the slower train is 36 km/hr given the specified conditions -/
theorem slower_train_speed 
  (train_length : ℝ) 
  (faster_train_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 25) 
  (h2 : faster_train_speed = 46) 
  (h3 : passing_time = 18) : 
  ∃ (slower_train_speed : ℝ), 
    slower_train_speed = 36 ∧ 
    (faster_train_speed - slower_train_speed) * (5 / 18) * passing_time = 2 * train_length :=
by sorry

end NUMINAMATH_CALUDE_slower_train_speed_l1309_130905


namespace NUMINAMATH_CALUDE_adams_earnings_l1309_130901

/-- Adam's lawn mowing earnings problem -/
theorem adams_earnings (earnings_per_lawn : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ) :
  earnings_per_lawn = 9 →
  total_lawns = 12 →
  forgotten_lawns = 8 →
  (total_lawns - forgotten_lawns) * earnings_per_lawn = 36 :=
by sorry

end NUMINAMATH_CALUDE_adams_earnings_l1309_130901


namespace NUMINAMATH_CALUDE_lattice_points_sum_l1309_130971

/-- Number of lattice points in a plane region -/
noncomputable def N (D : Set (ℝ × ℝ)) : ℕ := sorry

/-- Region A -/
def A : Set (ℝ × ℝ) := {(x, y) | y = x^2 ∧ x ≤ 0 ∧ x ≥ -10 ∧ y ≤ 1}

/-- Region B -/
def B : Set (ℝ × ℝ) := {(x, y) | y = x^2 ∧ x ≥ 0 ∧ x ≤ 1 ∧ y ≤ 100}

/-- Theorem: The sum of lattice points in the union and intersection of A and B is 1010 -/
theorem lattice_points_sum : N (A ∪ B) + N (A ∩ B) = 1010 := by sorry

end NUMINAMATH_CALUDE_lattice_points_sum_l1309_130971


namespace NUMINAMATH_CALUDE_percent_lost_is_twenty_l1309_130943

/-- Represents the number of games in each category -/
structure GameStats where
  won : ℕ
  lost : ℕ
  tied : ℕ

/-- Calculates the percentage of games lost -/
def percentLost (stats : GameStats) : ℚ :=
  stats.lost / (stats.won + stats.lost + stats.tied) * 100

/-- Theorem stating that for a team with a 7:3 win-to-loss ratio and 5 tied games,
    the percentage of games lost is 20% -/
theorem percent_lost_is_twenty {x : ℕ} (stats : GameStats)
    (h1 : stats.won = 7 * x)
    (h2 : stats.lost = 3 * x)
    (h3 : stats.tied = 5) :
  percentLost stats = 20 := by
  sorry

#eval percentLost ⟨7, 3, 5⟩

end NUMINAMATH_CALUDE_percent_lost_is_twenty_l1309_130943


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1309_130985

/-- Given a train and a bridge with specific lengths and time to pass, 
    calculate the speed of the train in km/hour -/
theorem train_speed_calculation 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (time_to_pass : ℝ) 
  (h1 : train_length = 560)
  (h2 : bridge_length = 140)
  (h3 : time_to_pass = 56) :
  (train_length + bridge_length) / time_to_pass * 3.6 = 45 :=
by sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l1309_130985


namespace NUMINAMATH_CALUDE_max_sections_five_lines_l1309_130970

/-- The number of sections created by n line segments in a rectangle,
    where each new line intersects all previous lines -/
def num_sections (n : ℕ) : ℕ :=
  1 + (List.range n).sum

/-- The property that each new line intersects all previous lines -/
def intersects_all_previous (n : ℕ) : Prop :=
  ∀ k, k < n → num_sections k < num_sections (k + 1)

theorem max_sections_five_lines :
  intersects_all_previous 5 →
  num_sections 5 = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_sections_five_lines_l1309_130970


namespace NUMINAMATH_CALUDE_y_min_at_a_or_b_l1309_130902

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := (x - a)^3 + (x - b)^3

/-- Theorem stating that the minimum of y occurs at either a or b -/
theorem y_min_at_a_or_b (a b : ℝ) :
  ∃ (x : ℝ), (∀ (z : ℝ), y z a b ≥ y x a b) ∧ (x = a ∨ x = b) := by
  sorry

end NUMINAMATH_CALUDE_y_min_at_a_or_b_l1309_130902


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l1309_130929

-- Define the functions
def p (x : ℝ) : ℝ := x^2 - 4*x + 3
def q (x : ℝ) : ℝ := -p x + 2
def r (x : ℝ) : ℝ := p (-x)

-- Define the number of intersection points
def c : ℕ := 2  -- Number of intersections between p and q
def d : ℕ := 1  -- Number of intersections between p and r

-- Theorem statement
theorem intersection_points_theorem :
  (∀ x : ℝ, p x = q x → x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) ∧
  (∀ x : ℝ, p x = r x → x = 0) ∧
  (10 * c + d = 21) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l1309_130929


namespace NUMINAMATH_CALUDE_mr_resty_total_units_l1309_130988

/-- Represents the number of apartment units on each floor of a building -/
def BuildingUnits := List Nat

/-- Building A's unit distribution -/
def building_a : BuildingUnits := [2, 4, 6, 8, 10, 12]

/-- Building B's unit distribution (identical to A) -/
def building_b : BuildingUnits := building_a

/-- Building C's unit distribution -/
def building_c : BuildingUnits := [3, 5, 7, 9]

/-- Calculate the total number of units in a building -/
def total_units (building : BuildingUnits) : Nat :=
  building.sum

/-- The main theorem stating the total number of apartment units Mr. Resty has -/
theorem mr_resty_total_units : 
  total_units building_a + total_units building_b + total_units building_c = 108 := by
  sorry

end NUMINAMATH_CALUDE_mr_resty_total_units_l1309_130988


namespace NUMINAMATH_CALUDE_marble_difference_l1309_130935

theorem marble_difference (pink orange purple : ℕ) : 
  pink = 13 →
  orange < pink →
  purple = 4 * orange →
  pink + orange + purple = 33 →
  pink - orange = 9 := by
sorry

end NUMINAMATH_CALUDE_marble_difference_l1309_130935


namespace NUMINAMATH_CALUDE_subtract_fractions_l1309_130995

theorem subtract_fractions : (5 : ℚ) / 9 - (1 : ℚ) / 6 = (7 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fractions_l1309_130995


namespace NUMINAMATH_CALUDE_initial_capacity_proof_l1309_130933

/-- The daily processing capacity of each machine before modernization. -/
def initial_capacity : ℕ := 1215

/-- The number of machines before modernization. -/
def initial_machines : ℕ := 32

/-- The daily processing capacity of each machine after modernization. -/
def new_capacity : ℕ := 1280

/-- The number of machines after modernization. -/
def new_machines : ℕ := initial_machines + 3

/-- The total daily processing before modernization. -/
def total_before : ℕ := 38880

/-- The total daily processing after modernization. -/
def total_after : ℕ := 44800

theorem initial_capacity_proof :
  initial_capacity * initial_machines = total_before ∧
  new_capacity * new_machines = total_after ∧
  initial_capacity < new_capacity :=
by sorry

end NUMINAMATH_CALUDE_initial_capacity_proof_l1309_130933


namespace NUMINAMATH_CALUDE_find_q_l1309_130954

theorem find_q (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 2/b)^2 - p*(a + 2/b) + q = 0) → 
  ((b + 2/a)^2 - p*(b + 2/a) + q = 0) → 
  q = 25/3 :=
by sorry

end NUMINAMATH_CALUDE_find_q_l1309_130954


namespace NUMINAMATH_CALUDE_extreme_point_implies_a_zero_l1309_130980

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2 - (a + 2) * x + 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * x - (a + 2)

theorem extreme_point_implies_a_zero :
  ∀ a : ℝ, (f_derivative a 1 = 0) → a = 0 :=
by sorry

#check extreme_point_implies_a_zero

end NUMINAMATH_CALUDE_extreme_point_implies_a_zero_l1309_130980


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_prism_l1309_130987

theorem sphere_volume_circumscribing_rectangular_prism :
  let edge1 : ℝ := 1
  let edge2 : ℝ := Real.sqrt 10
  let edge3 : ℝ := 5
  let space_diagonal : ℝ := Real.sqrt (edge1^2 + edge2^2 + edge3^2)
  let sphere_radius : ℝ := space_diagonal / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius^3
  sphere_volume = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_prism_l1309_130987


namespace NUMINAMATH_CALUDE_unique_valid_number_l1309_130910

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (∀ i : ℕ, i < 3 → (n / 10^i % 10 + n / 10^(i+1) % 10) ≤ 2) ∧
  (∀ i : ℕ, i < 2 → (n / 10^i % 10 + n / 10^(i+1) % 10 + n / 10^(i+2) % 10) ≥ 3)

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l1309_130910


namespace NUMINAMATH_CALUDE_max_squares_visited_l1309_130950

/-- Represents a square on the board --/
structure Square where
  x : Nat
  y : Nat

/-- Represents a move of the limp rook --/
inductive Move
  | Up
  | Down
  | Left
  | Right

/-- Defines the board size --/
def boardSize : Nat := 999

/-- Checks if a move is valid (adjacent and perpendicular to the previous move) --/
def isValidMove (prev : Move) (curr : Move) : Bool :=
  match prev, curr with
  | Move.Up, Move.Left    => true
  | Move.Up, Move.Right   => true
  | Move.Down, Move.Left  => true
  | Move.Down, Move.Right => true
  | Move.Left, Move.Up    => true
  | Move.Left, Move.Down  => true
  | Move.Right, Move.Up   => true
  | Move.Right, Move.Down => true
  | _, _                  => false

/-- Checks if a square is within the board --/
def isOnBoard (s : Square) : Bool :=
  s.x ≥ 0 && s.x < boardSize && s.y ≥ 0 && s.y < boardSize

/-- Represents a route of the limp rook --/
def Route := List Square

/-- Checks if a route is non-intersecting --/
def isNonIntersecting (r : Route) : Bool :=
  sorry

/-- Checks if a route is cyclic --/
def isCyclic (r : Route) : Bool :=
  sorry

/-- The main theorem --/
theorem max_squares_visited :
  ∃ (r : Route), isNonIntersecting r ∧ isCyclic r ∧ r.length = 996000 ∧
  (∀ (r' : Route), isNonIntersecting r' → isCyclic r' → r'.length ≤ 996000) :=
sorry

end NUMINAMATH_CALUDE_max_squares_visited_l1309_130950


namespace NUMINAMATH_CALUDE_max_rectangle_area_l1309_130934

/-- Represents the length of a wire segment between two marks -/
def segment_length : ℕ := 3

/-- Represents the total length of the wire -/
def wire_length : ℕ := 78

/-- Represents the total number of segments in the wire -/
def total_segments : ℕ := wire_length / segment_length

/-- Represents the perimeter of the rectangle in terms of segments -/
def perimeter_segments : ℕ := total_segments / 2

/-- Calculates the area of a rectangle given its length and width in segments -/
def rectangle_area (length width : ℕ) : ℕ :=
  (length * segment_length) * (width * segment_length)

/-- Theorem stating that the maximum area of the rectangle is 378 square centimeters -/
theorem max_rectangle_area :
  (∃ length width : ℕ,
    length + width = perimeter_segments ∧
    rectangle_area length width = 378 ∧
    ∀ l w : ℕ, l + w = perimeter_segments → rectangle_area l w ≤ 378) :=
by sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l1309_130934


namespace NUMINAMATH_CALUDE_f_max_min_range_l1309_130967

/-- A cubic function with parameter a -/
def f (a x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f_derivative (a x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

/-- Theorem stating the range of a for which f has both maximum and minimum values -/
theorem f_max_min_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
    (∀ z : ℝ, f a z ≤ f a x) ∧ 
    (∀ z : ℝ, f a z ≥ f a y)) ↔ 
  a < -3 ∨ a > 6 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_range_l1309_130967


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1309_130947

theorem arithmetic_equality : 5 * 7 + 6 * 12 + 7 * 4 + 2 * 9 = 153 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1309_130947


namespace NUMINAMATH_CALUDE_power_tower_mod_2000_l1309_130921

theorem power_tower_mod_2000 : 
  (5 : ℕ) ^ (5 ^ (5 ^ 5)) ≡ 625 [MOD 2000] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_2000_l1309_130921


namespace NUMINAMATH_CALUDE_value_in_numerator_l1309_130940

theorem value_in_numerator (N V : ℤ) : 
  N = 1280 → (N + 720) / 125 = V / 462 → V = 7392 := by sorry

end NUMINAMATH_CALUDE_value_in_numerator_l1309_130940


namespace NUMINAMATH_CALUDE_intersection_slope_l1309_130998

/-- Given two lines that intersect at a point, find the slope of one line -/
theorem intersection_slope (m : ℝ) : 
  (∃ (x y : ℝ), y = 2*x + 3 ∧ y = m*x + 1 ∧ x = 1 ∧ y = 5) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_slope_l1309_130998


namespace NUMINAMATH_CALUDE_min_attempts_to_guarantee_two_charged_l1309_130909

/-- Represents a set of batteries -/
def Battery := Fin 8

/-- Represents a pair of batteries -/
def BatteryPair := (Battery × Battery)

/-- The set of all possible battery pairs -/
def allPairs : Finset BatteryPair := sorry

/-- The set of charged batteries -/
def chargedBatteries : Finset Battery := sorry

/-- A function that determines if a set of battery pairs guarantees finding two charged batteries -/
def guaranteesTwoCharged (pairs : Finset BatteryPair) : Prop := sorry

/-- The minimum number of attempts required -/
def minAttempts : ℕ := sorry

theorem min_attempts_to_guarantee_two_charged :
  (minAttempts = 12) ∧
  (∃ (pairs : Finset BatteryPair), pairs.card = minAttempts ∧ guaranteesTwoCharged pairs) ∧
  (∀ (pairs : Finset BatteryPair), pairs.card < minAttempts → ¬guaranteesTwoCharged pairs) := by
  sorry

end NUMINAMATH_CALUDE_min_attempts_to_guarantee_two_charged_l1309_130909


namespace NUMINAMATH_CALUDE_binomial_expansions_l1309_130964

theorem binomial_expansions (a b : ℝ) : 
  ((a + b)^3 = a^3 + 3*a^2*b + 3*a*b^2 + b^3) ∧ 
  ((a + b)^4 = a^4 + 4*a^3*b + 6*a^2*b^2 + 4*a*b^3 + b^4) ∧
  ((a + b)^5 = a^5 + 5*a^4*b + 10*a^3*b^2 + 10*a^2*b^3 + 5*a*b^4 + b^5) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansions_l1309_130964


namespace NUMINAMATH_CALUDE_rook_placement_impossibility_l1309_130937

theorem rook_placement_impossibility :
  ∀ (r b g : ℕ),
  r + b + g = 50 →
  2 * r ≤ b →
  2 * b ≤ g →
  2 * g ≤ r →
  False :=
by sorry

end NUMINAMATH_CALUDE_rook_placement_impossibility_l1309_130937


namespace NUMINAMATH_CALUDE_N_subset_M_l1309_130900

-- Define the sets M and N
def M : Set ℝ := Set.univ
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -x^2}

-- State the theorem
theorem N_subset_M : N ⊆ M := by sorry

end NUMINAMATH_CALUDE_N_subset_M_l1309_130900


namespace NUMINAMATH_CALUDE_rectangular_box_height_l1309_130906

/-- Proves that the height of a rectangular box is 2 cm, given its volume, length, and width. -/
theorem rectangular_box_height (volume : ℝ) (length width : ℝ) (h1 : volume = 144) (h2 : length = 12) (h3 : width = 6) :
  volume = length * width * 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_height_l1309_130906


namespace NUMINAMATH_CALUDE_family_eating_habits_l1309_130963

theorem family_eating_habits (only_veg only_nonveg total_veg : ℕ) 
  (h1 : only_veg = 13)
  (h2 : only_nonveg = 8)
  (h3 : total_veg = 19) :
  total_veg - only_veg = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_family_eating_habits_l1309_130963


namespace NUMINAMATH_CALUDE_parabola_tangent_min_area_l1309_130926

/-- The parabola equation -/
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

/-- The point M -/
def M (y₀ : ℝ) : ℝ × ℝ := (-1, y₀)

/-- The area of triangle MAB -/
noncomputable def triangleArea (p : ℝ) (y₀ : ℝ) : ℝ :=
  2 * Real.sqrt (y₀^2 + 2*p)

/-- The main theorem -/
theorem parabola_tangent_min_area (p : ℝ) :
  p > 0 →
  (∀ y₀ : ℝ, triangleArea p y₀ ≥ 4) →
  (∃ y₀ : ℝ, triangleArea p y₀ = 4) →
  p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_tangent_min_area_l1309_130926


namespace NUMINAMATH_CALUDE_apple_street_highest_intersection_l1309_130907

/-- Calculates the highest-numbered intersection on a street -/
def highest_numbered_intersection (street_length : ℕ) (intersection_interval : ℕ) : ℕ :=
  (street_length / intersection_interval) - 2

/-- The theorem states that for a street of 3200 meters with intersections every 200 meters,
    the highest-numbered intersection is the 14th. -/
theorem apple_street_highest_intersection :
  highest_numbered_intersection 3200 200 = 14 := by
  sorry

end NUMINAMATH_CALUDE_apple_street_highest_intersection_l1309_130907


namespace NUMINAMATH_CALUDE_scientific_notation_502000_l1309_130968

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_502000 :
  toScientificNotation 502000 = ScientificNotation.mk 5.02 5 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_502000_l1309_130968


namespace NUMINAMATH_CALUDE_toms_age_ratio_l1309_130989

theorem toms_age_ratio (T N : ℝ) : 
  (T > 0) →  -- Tom's age is positive
  (N > 0) →  -- N is positive (years in the past)
  (T - N > 0) →  -- Tom's age N years ago was positive
  (T - 4*N > 0) →  -- The sum of children's ages N years ago was positive
  (T - N = 3 * (T - 4*N)) →  -- Condition about ages N years ago
  T / N = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l1309_130989


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1309_130918

-- Define the universal set U
def U : Set ℕ := {1, 3, 5, 7, 9}

-- Define set A
def A : Set ℕ := {1, 5, 9}

-- Define set B
def B : Set ℕ := {3, 7, 9}

-- Theorem statement
theorem complement_A_intersect_B : (Aᶜ ∩ B) = {3, 7} := by
  sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1309_130918


namespace NUMINAMATH_CALUDE_equation_holds_for_all_y_l1309_130923

theorem equation_holds_for_all_y (x : ℝ) : 
  (∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7 = 0) ↔ x = 3/2 := by
sorry

end NUMINAMATH_CALUDE_equation_holds_for_all_y_l1309_130923


namespace NUMINAMATH_CALUDE_final_mixture_concentration_l1309_130996

/-- Represents a vessel with a given capacity and alcohol concentration -/
structure Vessel where
  capacity : ℝ
  alcoholConcentration : ℝ

/-- Calculates the amount of alcohol in a vessel -/
def alcoholAmount (v : Vessel) : ℝ := v.capacity * v.alcoholConcentration

/-- Theorem: The concentration of the final mixture is (5.75 / 18) * 100% -/
theorem final_mixture_concentration 
  (vessel1 : Vessel)
  (vessel2 : Vessel)
  (vessel3 : Vessel)
  (vessel4 : Vessel)
  (finalVessel : ℝ) :
  vessel1.capacity = 2 →
  vessel1.alcoholConcentration = 0.25 →
  vessel2.capacity = 6 →
  vessel2.alcoholConcentration = 0.40 →
  vessel3.capacity = 3 →
  vessel3.alcoholConcentration = 0.55 →
  vessel4.capacity = 4 →
  vessel4.alcoholConcentration = 0.30 →
  finalVessel = 18 →
  (alcoholAmount vessel1 + alcoholAmount vessel2 + alcoholAmount vessel3 + alcoholAmount vessel4) / finalVessel = 5.75 / 18 := by
  sorry

#eval (5.75 / 18) * 100 -- Approximately 31.94%

end NUMINAMATH_CALUDE_final_mixture_concentration_l1309_130996


namespace NUMINAMATH_CALUDE_intersection_implies_p_value_l1309_130925

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (p : ℝ) (t : ℝ) : ℝ × ℝ := (2 * p * t, 2 * p * Real.sqrt t)
def C₂ (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Define the distance between two points
def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- State the theorem
theorem intersection_implies_p_value (p : ℝ) :
  p > 0 →
  ∃ (A B : ℝ × ℝ) (t₁ t₂ θ₁ θ₂ : ℝ),
    C₁ p t₁ = A ∧
    C₁ p t₂ = B ∧
    C₂ θ₁ = Real.sqrt (A.1^2 + A.2^2) ∧
    C₂ θ₂ = Real.sqrt (B.1^2 + B.2^2) ∧
    distance A B = 2 * Real.sqrt 3 →
    p = 3 * Real.sqrt 3 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_intersection_implies_p_value_l1309_130925


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_four_l1309_130942

/-- Given vectors a and b in ℝ², prove that if a + 3b is parallel to a - b, then the x-coordinate of b is 4. -/
theorem parallel_vectors_imply_x_equals_four (a b : ℝ × ℝ) 
  (ha : a = (2, 1)) 
  (hb : b.2 = 2) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a + 3 • b = k • (a - b)) : 
  b.1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_four_l1309_130942


namespace NUMINAMATH_CALUDE_flower_pots_total_cost_l1309_130983

/-- The number of flower pots -/
def num_pots : ℕ := 6

/-- The price difference between consecutive pots -/
def price_diff : ℚ := 25 / 100

/-- The price of the largest pot -/
def largest_pot_price : ℚ := 1925 / 1000

/-- Calculate the total cost of all flower pots -/
def total_cost : ℚ :=
  let smallest_pot_price := largest_pot_price - (num_pots - 1 : ℕ) * price_diff
  (num_pots : ℚ) * smallest_pot_price + (num_pots - 1 : ℕ) * (num_pots : ℚ) * price_diff / 2

theorem flower_pots_total_cost :
  total_cost = 780 / 100 := by sorry

end NUMINAMATH_CALUDE_flower_pots_total_cost_l1309_130983


namespace NUMINAMATH_CALUDE_wall_ratio_l1309_130949

/-- Proves that for a rectangular wall with given dimensions, the ratio of length to height is 7:1 -/
theorem wall_ratio (w h l : ℝ) (volume : ℝ) : 
  h = 6 * w →
  volume = l * w * h →
  w = 4 →
  volume = 16128 →
  l / h = 7 := by
  sorry

end NUMINAMATH_CALUDE_wall_ratio_l1309_130949


namespace NUMINAMATH_CALUDE_inequality_proof_l1309_130916

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1309_130916


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1309_130999

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1309_130999


namespace NUMINAMATH_CALUDE_not_iff_right_angle_and_equation_l1309_130914

/-- Definition of a triangle with sides a, b, c and altitude m from vertex C -/
structure Triangle :=
  (a b c m : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (positive_altitude : 0 < m)

/-- The equation in question -/
def satisfies_equation (t : Triangle) : Prop :=
  1 / t.m^2 = 1 / t.a^2 + 1 / t.b^2

/-- Theorem stating that the original statement is not true in general -/
theorem not_iff_right_angle_and_equation :
  ∃ (t : Triangle), satisfies_equation t ∧ ¬(t.a^2 + t.b^2 = t.c^2) :=
sorry

end NUMINAMATH_CALUDE_not_iff_right_angle_and_equation_l1309_130914


namespace NUMINAMATH_CALUDE_second_cat_weight_l1309_130952

theorem second_cat_weight (total_weight first_weight third_weight : ℕ) 
  (h1 : total_weight = 13)
  (h2 : first_weight = 2)
  (h3 : third_weight = 4) :
  total_weight - first_weight - third_weight = 7 := by
  sorry

end NUMINAMATH_CALUDE_second_cat_weight_l1309_130952


namespace NUMINAMATH_CALUDE_expression_value_l1309_130919

theorem expression_value : 3 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2400 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1309_130919


namespace NUMINAMATH_CALUDE_divisors_of_8n_cubed_l1309_130973

def is_product_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n = p * q

def count_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisors_of_8n_cubed (n : ℕ) 
  (h1 : is_product_of_two_primes n)
  (h2 : count_divisors n = 22)
  (h3 : Odd n) :
  count_divisors (8 * n^3) = 496 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_8n_cubed_l1309_130973


namespace NUMINAMATH_CALUDE_oxford_high_school_population_is_349_l1309_130941

/-- The number of people in Oxford High School -/
def oxford_high_school_population : ℕ :=
  let teachers : ℕ := 48
  let principal : ℕ := 1
  let classes : ℕ := 15
  let students_per_class : ℕ := 20
  let total_students : ℕ := classes * students_per_class
  teachers + principal + total_students

/-- Theorem stating the total number of people in Oxford High School -/
theorem oxford_high_school_population_is_349 :
  oxford_high_school_population = 349 := by
  sorry

end NUMINAMATH_CALUDE_oxford_high_school_population_is_349_l1309_130941


namespace NUMINAMATH_CALUDE_vector_sum_equality_l1309_130908

theorem vector_sum_equality (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (hx : x₁ + x₂ + x₃ = 0)
  (hy : y₁ + y₂ + y₃ = 0)
  (hxy : x₁*y₁ + x₂*y₂ + x₃*y₃ = 0)
  (hnz : (x₁^2 + x₂^2 + x₃^2) * (y₁^2 + y₂^2 + y₃^2) > 0) :
  x₁^2 / (x₁^2 + x₂^2 + x₃^2) + y₁^2 / (y₁^2 + y₂^2 + y₃^2) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_equality_l1309_130908


namespace NUMINAMATH_CALUDE_matthew_lollipops_l1309_130972

theorem matthew_lollipops (total_lollipops : ℕ) (friends : ℕ) (h1 : total_lollipops = 500) (h2 : friends = 15) :
  total_lollipops % friends = 5 := by
  sorry

end NUMINAMATH_CALUDE_matthew_lollipops_l1309_130972


namespace NUMINAMATH_CALUDE_wendys_score_l1309_130966

/-- The score for each treasure found in the game. -/
def points_per_treasure : ℕ := 5

/-- The number of treasures Wendy found on the first level. -/
def treasures_level1 : ℕ := 4

/-- The number of treasures Wendy found on the second level. -/
def treasures_level2 : ℕ := 3

/-- Wendy's total score in the game. -/
def total_score : ℕ := points_per_treasure * (treasures_level1 + treasures_level2)

/-- Theorem stating that Wendy's total score is 35 points. -/
theorem wendys_score : total_score = 35 := by
  sorry

end NUMINAMATH_CALUDE_wendys_score_l1309_130966


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1309_130982

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1309_130982


namespace NUMINAMATH_CALUDE_mike_has_one_unbroken_seashell_l1309_130932

/-- Represents the number of unbroken seashells Mike has left after his beach trip and giving away one shell. -/
def unbroken_seashells_left : ℕ :=
  let total_seashells := 6
  let cone_shells := 3
  let conch_shells := 3
  let broken_cone_shells := 2
  let broken_conch_shells := 2
  let unbroken_cone_shells := cone_shells - broken_cone_shells
  let unbroken_conch_shells := conch_shells - broken_conch_shells
  let given_away_shells := 1
  unbroken_cone_shells + (unbroken_conch_shells - given_away_shells)

/-- Theorem stating that Mike has 1 unbroken seashell left. -/
theorem mike_has_one_unbroken_seashell : unbroken_seashells_left = 1 := by
  sorry

end NUMINAMATH_CALUDE_mike_has_one_unbroken_seashell_l1309_130932


namespace NUMINAMATH_CALUDE_diagonal_cut_result_l1309_130939

/-- Represents a scarf with areas of different colors -/
structure Scarf where
  white : ℚ
  gray : ℚ
  black : ℚ

/-- The original square scarf -/
def original_scarf : Scarf where
  white := 1/2
  gray := 1/3
  black := 1/6

/-- The first triangular scarf after cutting -/
def first_triangular_scarf : Scarf where
  white := 3/4
  gray := 2/9
  black := 1/36

/-- The second triangular scarf after cutting -/
def second_triangular_scarf : Scarf where
  white := 1/4
  gray := 4/9
  black := 11/36

/-- Theorem stating that cutting the original square scarf diagonally 
    results in the two specified triangular scarves -/
theorem diagonal_cut_result : 
  (original_scarf.white + original_scarf.gray + original_scarf.black = 1) →
  (first_triangular_scarf.white + first_triangular_scarf.gray + first_triangular_scarf.black = 1) ∧
  (second_triangular_scarf.white + second_triangular_scarf.gray + second_triangular_scarf.black = 1) ∧
  (first_triangular_scarf.white = 3/4) ∧
  (first_triangular_scarf.gray = 2/9) ∧
  (first_triangular_scarf.black = 1/36) ∧
  (second_triangular_scarf.white = 1/4) ∧
  (second_triangular_scarf.gray = 4/9) ∧
  (second_triangular_scarf.black = 11/36) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_cut_result_l1309_130939


namespace NUMINAMATH_CALUDE_smallest_c_value_l1309_130984

theorem smallest_c_value : ∃ c : ℚ, (∀ x : ℚ, (3 * x + 4) * (x - 2) = 9 * x → c ≤ x) ∧ (3 * c + 4) * (c - 2) = 9 * c ∧ c = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l1309_130984
