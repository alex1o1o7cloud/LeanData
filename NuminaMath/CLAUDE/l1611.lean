import Mathlib

namespace NUMINAMATH_CALUDE_mult_inverse_mod_million_mult_inverse_specific_l1611_161117

/-- The multiplicative inverse of (A * B) modulo 1,000,000 is 466390 -/
theorem mult_inverse_mod_million : Int → Int → Prop :=
  fun A B => (A * B * 466390) % 1000000 = 1

/-- The theorem holds for A = 123456 and B = 162037 -/
theorem mult_inverse_specific : mult_inverse_mod_million 123456 162037 := by
  sorry

end NUMINAMATH_CALUDE_mult_inverse_mod_million_mult_inverse_specific_l1611_161117


namespace NUMINAMATH_CALUDE_tan_equality_periodic_l1611_161161

theorem tan_equality_periodic (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (243 * π / 180) → n = 63 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_periodic_l1611_161161


namespace NUMINAMATH_CALUDE_vanaspati_percentage_in_original_mixture_l1611_161119

/-- Represents the composition of a ghee mixture -/
structure GheeMixture where
  total : ℝ
  pure_percentage : ℝ

/-- Calculates the percentage of vanaspati in a ghee mixture -/
def vanaspati_percentage (mixture : GheeMixture) : ℝ :=
  100 - mixture.pure_percentage

theorem vanaspati_percentage_in_original_mixture 
  (original : GheeMixture)
  (h_original_total : original.total = 10)
  (h_original_pure : original.pure_percentage = 60)
  (h_after_addition : 
    let new_total := original.total + 10
    let new_pure := original.total * (original.pure_percentage / 100) + 10
    (100 - (new_pure / new_total * 100)) = 20) :
  vanaspati_percentage original = 40 := by
  sorry

#eval vanaspati_percentage { total := 10, pure_percentage := 60 }

end NUMINAMATH_CALUDE_vanaspati_percentage_in_original_mixture_l1611_161119


namespace NUMINAMATH_CALUDE_monthly_salary_calculation_l1611_161193

/-- Represents the monthly salary in Rupees -/
def monthly_salary : ℝ := sorry

/-- Represents the savings rate as a decimal -/
def savings_rate : ℝ := 0.2

/-- Represents the expense increase rate as a decimal -/
def expense_increase_rate : ℝ := 0.2

/-- Represents the new monthly savings amount in Rupees -/
def new_savings : ℝ := 240

theorem monthly_salary_calculation :
  monthly_salary * (1 - (1 + expense_increase_rate) * (1 - savings_rate)) = new_savings ∧
  monthly_salary = 6000 := by sorry

end NUMINAMATH_CALUDE_monthly_salary_calculation_l1611_161193


namespace NUMINAMATH_CALUDE_simplest_square_root_l1611_161146

/-- Given real numbers a and b, with a ≠ 0, prove that √(a^2 + b^2) is the simplest form among:
    √(16a), √(a^2 + b^2), √(b/a), and √45 -/
theorem simplest_square_root (a b : ℝ) (ha : a ≠ 0) :
  ∃ (f : ℝ → ℝ), f (Real.sqrt (a^2 + b^2)) = Real.sqrt (a^2 + b^2) ∧
    (∀ g : ℝ → ℝ, g (Real.sqrt (16*a)) ≠ Real.sqrt (16*a) ∨
                   g (Real.sqrt (b/a)) ≠ Real.sqrt (b/a) ∨
                   g (Real.sqrt 45) ≠ Real.sqrt 45 ∨
                   g = f) :=
by sorry

end NUMINAMATH_CALUDE_simplest_square_root_l1611_161146


namespace NUMINAMATH_CALUDE_expression_simplification_l1611_161132

theorem expression_simplification
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hbc : b - 2 / c ≠ 0) :
  (a - 2 / b) / (b - 2 / c) = c / b :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1611_161132


namespace NUMINAMATH_CALUDE_product_relation_l1611_161125

theorem product_relation (x y z : ℝ) (h : x^2 + y^2 = x*y*(z + 1/z)) :
  x = y*z ∨ y = x*z :=
by sorry

end NUMINAMATH_CALUDE_product_relation_l1611_161125


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1611_161166

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 65 / 3) + (5 / 3))

theorem sum_of_coefficients (a b c : ℕ+) : 
  y^120 = 3*y^117 + 17*y^114 + 13*y^112 - y^60 + (a:ℝ)*y^55 + (b:ℝ)*y^53 + (c:ℝ)*y^50 →
  a + b + c = 131 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1611_161166


namespace NUMINAMATH_CALUDE_polygon_sides_count_l1611_161155

theorem polygon_sides_count (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 2 * 360 ↔ n = 6 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l1611_161155


namespace NUMINAMATH_CALUDE_largest_angle_in_789_ratio_triangle_l1611_161103

/-- Given a triangle with interior angles in a 7:8:9 ratio, 
    the largest interior angle measures 67.5 degrees. -/
theorem largest_angle_in_789_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    a + b + c = 180 →
    b = (8/7) * a →
    c = (9/7) * a →
    max a (max b c) = 67.5 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_789_ratio_triangle_l1611_161103


namespace NUMINAMATH_CALUDE_min_value_theorem_l1611_161167

theorem min_value_theorem (α₁ α₂ : ℝ) 
  (h : (1 / (2 + Real.sin α₁)) + (1 / (2 + Real.sin (2 * α₂))) = 2) :
  ∃ (k₁ k₂ : ℤ), ∀ (m₁ m₂ : ℤ), 
    |10 * Real.pi - α₁ - α₂| ≥ |10 * Real.pi - ((-π/2 + 2*↑k₁*π) + (-π/4 + ↑k₂*π))| ∧
    |10 * Real.pi - ((-π/2 + 2*↑k₁*π) + (-π/4 + ↑k₂*π))| = π/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1611_161167


namespace NUMINAMATH_CALUDE_estimate_fish_population_l1611_161153

/-- Estimate the total number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population (marked_fish : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) :
  marked_fish = 200 →
  second_catch = 100 →
  marked_in_second = 10 →
  (marked_fish * second_catch) / marked_in_second = 2000 :=
by
  sorry

#check estimate_fish_population

end NUMINAMATH_CALUDE_estimate_fish_population_l1611_161153


namespace NUMINAMATH_CALUDE_binomial_12_choose_6_l1611_161158

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_6_l1611_161158


namespace NUMINAMATH_CALUDE_find_divisor_l1611_161144

theorem find_divisor (N D : ℕ) (h1 : N = D * 8) (h2 : N % 5 = 4) : D = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1611_161144


namespace NUMINAMATH_CALUDE_polynomial_factor_l1611_161159

-- Define the polynomial
def f (b : ℝ) (x : ℝ) : ℝ := 3 * x^3 + b * x + 12

-- Define the quadratic factor
def g (p : ℝ) (x : ℝ) : ℝ := x^2 + p * x + 2

-- Theorem statement
theorem polynomial_factor (b : ℝ) :
  (∃ p : ℝ, ∀ x : ℝ, ∃ k : ℝ, f b x = g p x * (3 * x + 6)) →
  b = -6 := by sorry

end NUMINAMATH_CALUDE_polynomial_factor_l1611_161159


namespace NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l1611_161172

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex polygon with 30 sides has 405 diagonals -/
theorem thirty_sided_polygon_diagonals : num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l1611_161172


namespace NUMINAMATH_CALUDE_expression_values_l1611_161147

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  expr = 5 ∨ expr = 1 ∨ expr = -3 := by
sorry

end NUMINAMATH_CALUDE_expression_values_l1611_161147


namespace NUMINAMATH_CALUDE_subtraction_and_decimal_conversion_l1611_161142

theorem subtraction_and_decimal_conversion : 3/4 - 1/16 = 0.6875 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_and_decimal_conversion_l1611_161142


namespace NUMINAMATH_CALUDE_angle_value_l1611_161137

theorem angle_value (θ : Real) (h : θ > 0 ∧ θ < 90) : 
  (∃ (x y : Real), x = Real.sin (10 * π / 180) ∧ 
                   y = 1 + Real.sin (80 * π / 180) ∧ 
                   x = Real.sin θ ∧ 
                   y = Real.cos θ) → 
  θ = 85 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_value_l1611_161137


namespace NUMINAMATH_CALUDE_reciprocal_equation_solution_l1611_161196

theorem reciprocal_equation_solution (x : ℝ) : 
  2 - (1 / (2 - x)^3) = 1 / (2 - x)^3 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_solution_l1611_161196


namespace NUMINAMATH_CALUDE_connie_additional_money_l1611_161152

def additional_money_needed (savings : ℚ) (watch_cost : ℚ) (strap_cost : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_cost_before_tax := watch_cost + strap_cost
  let tax_amount := total_cost_before_tax * tax_rate
  let total_cost_with_tax := total_cost_before_tax + tax_amount
  total_cost_with_tax - savings

theorem connie_additional_money :
  additional_money_needed 39 55 15 (8/100) = 366/10 := by
  sorry

end NUMINAMATH_CALUDE_connie_additional_money_l1611_161152


namespace NUMINAMATH_CALUDE_m_range_theorem_l1611_161101

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 1| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem m_range_theorem (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(q x m) → ¬(p x)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  (0 < m ∧ m ≤ 2) :=
by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_m_range_theorem_l1611_161101


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1611_161175

def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -2)

theorem parallel_vectors_x_value :
  ∀ x : ℝ, (∃ k : ℝ, k ≠ 0 ∧ a + b x = k • (2 • a - b x)) → x = -4 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1611_161175


namespace NUMINAMATH_CALUDE_average_marks_proof_l1611_161191

theorem average_marks_proof (total_subjects : Nat) 
                             (avg_five_subjects : ℝ) 
                             (sixth_subject_marks : ℝ) : 
  total_subjects = 6 →
  avg_five_subjects = 74 →
  sixth_subject_marks = 50 →
  ((avg_five_subjects * 5 + sixth_subject_marks) / total_subjects : ℝ) = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_proof_l1611_161191


namespace NUMINAMATH_CALUDE_max_video_game_hours_l1611_161149

/-- Proves that given the conditions of Max's video game playing schedule,
    he must have played 2 hours on Wednesday. -/
theorem max_video_game_hours :
  ∀ x : ℝ,
  (x + x + (x + 3)) / 3 = 3 →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_max_video_game_hours_l1611_161149


namespace NUMINAMATH_CALUDE_unique_four_digit_reverse_l1611_161182

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem unique_four_digit_reverse : ∃! n : ℕ, is_four_digit n ∧ n + 8802 = reverse_digits n :=
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_reverse_l1611_161182


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1611_161145

/-- A line passing through point (2,-1) and perpendicular to x+y-3=0 has the equation x-y-3=0 -/
theorem perpendicular_line_equation :
  let point : ℝ × ℝ := (2, -1)
  let perpendicular_to : ℝ → ℝ → ℝ := fun x y => x + y - 3
  let line_equation : ℝ → ℝ → ℝ := fun x y => x - y - 3
  (∀ x y, perpendicular_to x y = 0 → (line_equation x y = 0 ↔ 
    (x - point.1) * 1 = (y - point.2) * 1 ∧
    1 * (-1) = -1)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1611_161145


namespace NUMINAMATH_CALUDE_db_length_l1611_161160

/-- Triangle ABC with altitudes and median -/
structure TriangleABC where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  M : ℝ × ℝ
  -- CD is altitude to AB
  cd_altitude : (C.1 - D.1) * (B.1 - A.1) + (C.2 - D.2) * (B.2 - A.2) = 0
  -- AE is altitude to BC
  ae_altitude : (A.1 - E.1) * (C.1 - B.1) + (A.2 - E.2) * (C.2 - B.2) = 0
  -- AM is median to BC
  am_median : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  -- Given lengths
  ab_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 12
  cd_length : Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 5
  ae_length : Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 4
  am_length : Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) = 6

/-- The length of DB in the given triangle is 15 -/
theorem db_length (t : TriangleABC) : 
  Real.sqrt ((t.D.1 - t.B.1)^2 + (t.D.2 - t.B.2)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_db_length_l1611_161160


namespace NUMINAMATH_CALUDE_coordinates_uniquely_determine_position_l1611_161136

-- Define a structure for geographical coordinates
structure GeoCoord where
  longitude : Real
  latitude : Real

-- Define a type for position descriptors
inductive PositionDescriptor
  | Distance (d : Real) (reference : String)
  | RoadName (name : String)
  | Coordinates (coord : GeoCoord)
  | Direction (angle : Real) (reference : String)

-- Function to check if a descriptor uniquely determines a position
def uniquelyDeterminesPosition (descriptor : PositionDescriptor) : Prop :=
  match descriptor with
  | PositionDescriptor.Coordinates _ => True
  | _ => False

-- Theorem stating that only coordinates uniquely determine a position
theorem coordinates_uniquely_determine_position
  (descriptor : PositionDescriptor) :
  uniquelyDeterminesPosition descriptor ↔
  ∃ (coord : GeoCoord), descriptor = PositionDescriptor.Coordinates coord :=
sorry

#check coordinates_uniquely_determine_position

end NUMINAMATH_CALUDE_coordinates_uniquely_determine_position_l1611_161136


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1611_161135

theorem fraction_sum_equality : (3 + 6 + 9) / (2 + 5 + 8) + (2 + 5 + 8) / (3 + 6 + 9) = 61 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1611_161135


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1611_161121

theorem complex_fraction_evaluation (u v : ℂ) (hu : u ≠ 0) (hv : v ≠ 0) 
  (h : u^2 + u*v + v^2 = 0) : 
  (u^7 + v^7) / (u + v)^7 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1611_161121


namespace NUMINAMATH_CALUDE_roots_magnitude_l1611_161187

theorem roots_magnitude (p : ℝ) (r₁ r₂ : ℝ) : 
  (r₁ ≠ r₂) →  -- r₁ and r₂ are distinct
  (r₁^2 + p*r₁ + 12 = 0) →  -- r₁ is a root of the equation
  (r₂^2 + p*r₂ + 12 = 0) →  -- r₂ is a root of the equation
  (abs r₁ > 3 ∨ abs r₂ > 3) := by
sorry

end NUMINAMATH_CALUDE_roots_magnitude_l1611_161187


namespace NUMINAMATH_CALUDE_positive_numbers_l1611_161154

theorem positive_numbers (a b c : ℝ) 
  (sum_positive : a + b + c > 0)
  (sum_products_positive : a * b + b * c + c * a > 0)
  (product_positive : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_l1611_161154


namespace NUMINAMATH_CALUDE_perpendicular_point_sets_l1611_161133

-- Define the concept of a "perpendicular point set"
def isPerpendicular (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (x₁ y₁ : ℝ), (x₁, y₁) ∈ M → 
    ∃ (x₂ y₂ : ℝ), (x₂, y₂) ∈ M ∧ x₁ * x₂ + y₁ * y₂ = 0

-- Define the sets
def M₁ : Set (ℝ × ℝ) := {(x, y) | y = 1 / x^2 ∧ x ≠ 0}
def M₂ : Set (ℝ × ℝ) := {(x, y) | y = Real.log x / Real.log 2 ∧ x > 0}
def M₃ : Set (ℝ × ℝ) := {(x, y) | y = 2^x - 2}
def M₄ : Set (ℝ × ℝ) := {(x, y) | y = Real.sin x + 1}

-- State the theorem
theorem perpendicular_point_sets :
  isPerpendicular M₁ ∧ 
  ¬(isPerpendicular M₂) ∧ 
  isPerpendicular M₃ ∧ 
  isPerpendicular M₄ := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_point_sets_l1611_161133


namespace NUMINAMATH_CALUDE_percentage_of_x_l1611_161180

theorem percentage_of_x (x y : ℝ) (h1 : x / y = 4) (h2 : y ≠ 0) :
  (2 * x - y) / x * 100 = 175 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_x_l1611_161180


namespace NUMINAMATH_CALUDE_woody_writing_time_l1611_161139

/-- Proves that Woody spent 1.5 years writing his book given the conditions -/
theorem woody_writing_time :
  ∀ (woody_months ivanka_months : ℕ),
  ivanka_months = woody_months + 3 →
  woody_months + ivanka_months = 39 →
  (woody_months : ℚ) / 12 = 3/2 := by
sorry

end NUMINAMATH_CALUDE_woody_writing_time_l1611_161139


namespace NUMINAMATH_CALUDE_land_conversion_rates_l1611_161116

/-- Represents the daily conversion rates and conditions for land conversion --/
structure LandConversion where
  total_area : ℝ
  rate_ratio : ℝ
  time_difference : ℝ
  team_b_rate : ℝ

/-- Theorem stating the correct daily conversion rates given the conditions --/
theorem land_conversion_rates (lc : LandConversion)
  (h1 : lc.total_area = 1500)
  (h2 : lc.rate_ratio = 1.2)
  (h3 : lc.time_difference = 5)
  (h4 : lc.total_area / lc.team_b_rate - lc.time_difference = lc.total_area / (lc.rate_ratio * lc.team_b_rate)) :
  lc.team_b_rate = 50 ∧ lc.rate_ratio * lc.team_b_rate = 60 := by
  sorry

end NUMINAMATH_CALUDE_land_conversion_rates_l1611_161116


namespace NUMINAMATH_CALUDE_equation_positive_root_l1611_161197

/-- Given an equation (x / (x - 5) = 3 - a / (x - 5)) with a positive root, prove that a = -5 --/
theorem equation_positive_root (x a : ℝ) (h : x > 0) 
  (eq : x / (x - 5) = 3 - a / (x - 5)) : a = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_root_l1611_161197


namespace NUMINAMATH_CALUDE_wall_width_proof_l1611_161105

theorem wall_width_proof (width height length : ℝ) 
  (height_def : height = 6 * width)
  (length_def : length = 7 * height)
  (volume_def : width * height * length = 86436) :
  width = 7 :=
by sorry

end NUMINAMATH_CALUDE_wall_width_proof_l1611_161105


namespace NUMINAMATH_CALUDE_smallest_sum_with_same_probability_l1611_161192

/-- Represents a symmetrical die with 6 faces --/
structure SymmetricalDie :=
  (faces : Fin 6)

/-- Represents a set of symmetrical dice --/
def DiceSet := List SymmetricalDie

/-- The probability of getting a specific sum --/
def probability (dice : DiceSet) (sum : Nat) : ℝ := sorry

theorem smallest_sum_with_same_probability 
  (dice : DiceSet) 
  (p : ℝ) 
  (h1 : p > 0) 
  (h2 : probability dice 2022 = p) : 
  ∃ (smallest_sum : Nat), 
    smallest_sum = 337 ∧ 
    probability dice smallest_sum = p ∧ 
    ∀ (other_sum : Nat), 
      other_sum < smallest_sum → probability dice other_sum ≠ p :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_with_same_probability_l1611_161192


namespace NUMINAMATH_CALUDE_average_gas_mileage_calculation_l1611_161108

theorem average_gas_mileage_calculation (distance_to_university : ℝ) (sedan_efficiency : ℝ)
  (weekend_trip_distance : ℝ) (truck_efficiency : ℝ)
  (h1 : distance_to_university = 150)
  (h2 : sedan_efficiency = 25)
  (h3 : weekend_trip_distance = 200)
  (h4 : truck_efficiency = 15) :
  let total_distance := distance_to_university + weekend_trip_distance
  let sedan_gas_used := distance_to_university / sedan_efficiency
  let truck_gas_used := weekend_trip_distance / truck_efficiency
  let total_gas_used := sedan_gas_used + truck_gas_used
  total_distance / total_gas_used = 1050 / 58 := by sorry

end NUMINAMATH_CALUDE_average_gas_mileage_calculation_l1611_161108


namespace NUMINAMATH_CALUDE_parabola_intersection_l1611_161195

theorem parabola_intersection :
  let f (x : ℝ) := 4 * x^2 + 5 * x - 6
  let g (x : ℝ) := x^2 + 14
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧
    f x₁ = g x₁ ∧ f x₂ = g x₂ ∧
    x₁ = -4 ∧ x₂ = 5/3 ∧
    f x₁ = 38 ∧ f x₂ = 121/9 ∧
    ∀ (x : ℝ), f x = g x → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1611_161195


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l1611_161157

/-- The quadratic function f(x) = (x-3)^2 + 1 -/
def f (x : ℝ) : ℝ := (x - 3)^2 + 1

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (3, 1)

/-- Theorem: The vertex of the quadratic function f(x) = (x-3)^2 + 1 is at the point (3,1) -/
theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ f (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l1611_161157


namespace NUMINAMATH_CALUDE_circles_intersect_l1611_161120

theorem circles_intersect : ∃ (x y : ℝ), 
  (x^2 + y^2 + 6*x - 7 = 0) ∧ (x^2 + y^2 + 6*y - 27 = 0) := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l1611_161120


namespace NUMINAMATH_CALUDE_intersection_distance_l1611_161169

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -4*y

-- Define the line
def line (x y : ℝ) : Prop := y = x - 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem intersection_distance : 
  ∃ (M N : ℝ × ℝ),
    (parabola M.1 M.2) ∧ 
    (parabola N.1 N.2) ∧
    (line M.1 M.2) ∧ 
    (line N.1 N.2) ∧
    (line focus.1 focus.2) ∧
    (M ≠ N) ∧
    (Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 8) :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l1611_161169


namespace NUMINAMATH_CALUDE_ryan_weekly_commute_l1611_161148

/-- Represents the different routes Ryan can take --/
inductive Route
| A
| B

/-- Represents the days of the week --/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

/-- Represents the different transportation methods --/
inductive TransportMethod
| Bike
| Bus
| FriendRide
| Walk

/-- Function to calculate biking time based on the route --/
def bikingTime (route : Route) : ℕ :=
  match route with
  | Route.A => 30
  | Route.B => 40

/-- Function to calculate bus time based on the day --/
def busTime (day : Day) : ℕ :=
  match day with
  | Day.Tuesday => 50
  | _ => 40

/-- Function to calculate friend's ride time based on the day --/
def friendRideTime (day : Day) : ℕ :=
  match day with
  | Day.Wednesday => 25
  | _ => 10

/-- Function to calculate walking time --/
def walkingTime : ℕ := 90

/-- Function to calculate total weekly commuting time --/
def totalWeeklyCommutingTime : ℕ :=
  (bikingTime Route.A + bikingTime Route.B) +
  (busTime Day.Monday + busTime Day.Tuesday + busTime Day.Wednesday) +
  friendRideTime Day.Wednesday +
  walkingTime

/-- Theorem stating that Ryan's total weekly commuting time is 315 minutes --/
theorem ryan_weekly_commute : totalWeeklyCommutingTime = 315 := by
  sorry

end NUMINAMATH_CALUDE_ryan_weekly_commute_l1611_161148


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1611_161184

theorem quadratic_solution_sum (c d : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 2 * x + 8 = 0 ↔ x = c + d * I ∨ x = c - d * I) → 
  c + d^2 = 44/25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1611_161184


namespace NUMINAMATH_CALUDE_compute_expression_l1611_161189

theorem compute_expression : 3 * 3^4 - 9^60 / 9^57 = -486 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1611_161189


namespace NUMINAMATH_CALUDE_surface_area_theorem_l1611_161140

/-- Represents a right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side_length : ℝ

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the solid RUVWX -/
structure SlicedSolid where
  prism : RightPrism
  V : Point3D
  W : Point3D
  X : Point3D

/-- Calculate the surface area of the sliced solid RUVWX -/
def surface_area_RUVWX (solid : SlicedSolid) : ℝ :=
  sorry

/-- The main theorem stating the surface area of RUVWX -/
theorem surface_area_theorem (solid : SlicedSolid) 
  (h1 : solid.prism.height = 20)
  (h2 : solid.prism.base_side_length = 10)
  (h3 : solid.V = Point3D.mk 5 0 10)
  (h4 : solid.W = Point3D.mk 5 (5 * Real.sqrt 3) 10)
  (h5 : solid.X = Point3D.mk 0 0 10) :
  surface_area_RUVWX solid = 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2 :=
sorry

end NUMINAMATH_CALUDE_surface_area_theorem_l1611_161140


namespace NUMINAMATH_CALUDE_total_peppers_weight_l1611_161171

/-- The weight of green peppers bought by Hannah's Vegetarian Restaurant -/
def green_peppers : ℚ := 0.3333333333333333

/-- The weight of red peppers bought by Hannah's Vegetarian Restaurant -/
def red_peppers : ℚ := 0.3333333333333333

/-- The total weight of peppers bought by Hannah's Vegetarian Restaurant -/
def total_peppers : ℚ := green_peppers + red_peppers

/-- Theorem stating that the total weight of peppers is 0.6666666666666666 pounds -/
theorem total_peppers_weight : total_peppers = 0.6666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_total_peppers_weight_l1611_161171


namespace NUMINAMATH_CALUDE_fifteenth_prime_l1611_161124

def is_prime (n : ℕ) : Prop := sorry

def nth_prime (n : ℕ) : ℕ := sorry

theorem fifteenth_prime :
  (nth_prime 5 = 11) → (nth_prime 15 = 47) :=
sorry

end NUMINAMATH_CALUDE_fifteenth_prime_l1611_161124


namespace NUMINAMATH_CALUDE_unique_functional_equation_solution_l1611_161100

theorem unique_functional_equation_solution :
  ∃! f : ℕ → ℕ, ∀ m n : ℕ, f (m + f n) = f m + f n + f (n + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_functional_equation_solution_l1611_161100


namespace NUMINAMATH_CALUDE_part_probabilities_l1611_161151

/-- Given two machines producing parts with known quantities of standard parts,
    this theorem proves the probabilities of selecting a standard part overall
    and conditionally based on which machine produced it. -/
theorem part_probabilities
  (total_parts_1 : ℕ) (standard_parts_1 : ℕ)
  (total_parts_2 : ℕ) (standard_parts_2 : ℕ)
  (h1 : total_parts_1 = 200)
  (h2 : standard_parts_1 = 190)
  (h3 : total_parts_2 = 300)
  (h4 : standard_parts_2 = 280) :
  let total_parts := total_parts_1 + total_parts_2
  let total_standard := standard_parts_1 + standard_parts_2
  let p_A := total_standard / total_parts
  let p_A_given_B := standard_parts_1 / total_parts_1
  let p_A_given_not_B := standard_parts_2 / total_parts_2
  p_A = 47/50 ∧ p_A_given_B = 19/20 ∧ p_A_given_not_B = 14/15 :=
by sorry

end NUMINAMATH_CALUDE_part_probabilities_l1611_161151


namespace NUMINAMATH_CALUDE_golden_ratio_trigonometric_identity_l1611_161141

theorem golden_ratio_trigonometric_identity :
  let m := 2 * Real.sin (18 * π / 180)
  (Real.sin (42 * π / 180) + m) / Real.cos (42 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_trigonometric_identity_l1611_161141


namespace NUMINAMATH_CALUDE_min_value_expression_l1611_161173

theorem min_value_expression (a b c : ℝ) (h1 : c > a) (h2 : a > b) (h3 : c ≠ 0) :
  ((a - c)^2 + (c - b)^2 + (b - a)^2) / c^2 ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1611_161173


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l1611_161163

theorem simplify_sqrt_sum : 
  Real.sqrt (8 + 6 * Real.sqrt 3) + Real.sqrt (8 - 6 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l1611_161163


namespace NUMINAMATH_CALUDE_triangular_array_sum_l1611_161162

/-- Represents the sum of numbers in the nth row of the triangular array. -/
def f (n : ℕ) : ℕ :=
  4 * 2^(n-1) - 2*n

/-- The triangular array starts with 1 on top and increases by 1 for each subsequent outer number.
    Interior numbers are obtained by adding two adjacent numbers from the previous row. -/
theorem triangular_array_sum (n : ℕ) (h : n > 0) :
  f n = 4 * 2^(n-1) - 2*n :=
sorry

end NUMINAMATH_CALUDE_triangular_array_sum_l1611_161162


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l1611_161110

theorem sum_of_x_solutions_is_zero (x y : ℝ) :
  y = 6 →
  x^2 + y^2 = 169 →
  ∃ x₁ x₂ : ℝ, x₁ + x₂ = 0 ∧ 
    ((x = x₁ ∨ x = x₂) ↔ (y = 6 ∧ x^2 + y^2 = 169)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l1611_161110


namespace NUMINAMATH_CALUDE_problem_odometer_distance_l1611_161164

/-- Represents an odometer that skips certain digits -/
structure SkippingOdometer :=
  (skipped_digits : List Nat)
  (displayed_value : Nat)

/-- Calculates the actual distance for a skipping odometer -/
def actualDistance (o : SkippingOdometer) : Nat :=
  sorry

/-- The specific odometer from the problem -/
def problemOdometer : SkippingOdometer :=
  { skipped_digits := [4, 7],
    displayed_value := 3008 }

theorem problem_odometer_distance :
  actualDistance problemOdometer = 1542 :=
sorry

end NUMINAMATH_CALUDE_problem_odometer_distance_l1611_161164


namespace NUMINAMATH_CALUDE_train_speed_problem_l1611_161115

/-- Given a train journey with the following properties:
  * Total distance is 3x km
  * First part of the journey covers x km at speed V kmph
  * Second part of the journey covers 2x km at 20 kmph
  * Average speed for the entire journey is 27 kmph
  Then, the speed V of the first part of the journey is 90 kmph. -/
theorem train_speed_problem (x : ℝ) (V : ℝ) (h_x_pos : x > 0) (h_V_pos : V > 0) :
  (x / V + 2 * x / 20) = 3 * x / 27 → V = 90 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1611_161115


namespace NUMINAMATH_CALUDE_music_class_students_l1611_161179

theorem music_class_students :
  ∃! n : ℕ, n < 50 ∧ n % 8 = 5 ∧ n % 6 = 3 ∧ n = 45 := by
sorry

end NUMINAMATH_CALUDE_music_class_students_l1611_161179


namespace NUMINAMATH_CALUDE_license_plate_count_l1611_161165

/-- The number of letters in the alphabet. -/
def alphabet_size : ℕ := 26

/-- The number of possible odd digits. -/
def odd_digits : ℕ := 5

/-- The number of possible even digits. -/
def even_digits : ℕ := 5

/-- The number of possible digits that are multiples of 3. -/
def multiples_of_three : ℕ := 4

/-- The total number of license plates with the given constraints. -/
def total_license_plates : ℕ := alphabet_size ^ 3 * odd_digits * even_digits * multiples_of_three

theorem license_plate_count :
  total_license_plates = 17576000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1611_161165


namespace NUMINAMATH_CALUDE_root_in_interval_l1611_161109

-- Define the function f(x) = x^3 + x - 1
def f (x : ℝ) : ℝ := x^3 + x - 1

-- State the theorem
theorem root_in_interval :
  f 0.6 < 0 → f 0.7 > 0 → ∃ x ∈ Set.Ioo 0.6 0.7, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l1611_161109


namespace NUMINAMATH_CALUDE_sugar_ratio_l1611_161118

theorem sugar_ratio (a₁ a₂ a₃ a₄ : ℝ) (h₁ : a₁ = 24) (h₂ : a₄ = 3)
  (h_geom : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r) :
  a₂ / a₁ = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sugar_ratio_l1611_161118


namespace NUMINAMATH_CALUDE_price_reduction_proof_l1611_161198

/-- The original selling price -/
def original_price : ℝ := 40

/-- The cost price -/
def cost_price : ℝ := 30

/-- The initial daily sales volume -/
def initial_sales : ℝ := 48

/-- The price after two reductions -/
def reduced_price : ℝ := 32.4

/-- The additional sales per yuan of price reduction -/
def sales_increase_rate : ℝ := 8

/-- The target daily profit -/
def target_profit : ℝ := 504

/-- The percentage reduction in price -/
def reduction_percentage : ℝ := 0.1

/-- The price reduction amount -/
def price_reduction : ℝ := 3

theorem price_reduction_proof :
  (∃ x : ℝ, (1 - x)^2 * original_price = reduced_price ∧ x = reduction_percentage) ∧
  (∃ m : ℝ, (original_price - m - cost_price) * (initial_sales + sales_increase_rate * m) = target_profit ∧ m = price_reduction) := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_proof_l1611_161198


namespace NUMINAMATH_CALUDE_gcd_1426_1643_l1611_161107

theorem gcd_1426_1643 : Nat.gcd 1426 1643 = 31 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1426_1643_l1611_161107


namespace NUMINAMATH_CALUDE_angle_tangent_product_l1611_161156

theorem angle_tangent_product (A C : ℝ) (h : 5 * (Real.cos A + Real.cos C) + 4 * (Real.cos A * Real.cos C + 1) = 0) :
  Real.tan (A / 2) * Real.tan (C / 2) = 3 := by
sorry

end NUMINAMATH_CALUDE_angle_tangent_product_l1611_161156


namespace NUMINAMATH_CALUDE_unit_vector_parallel_to_a_l1611_161194

def vector_a : ℝ × ℝ := (12, 5)

theorem unit_vector_parallel_to_a :
  ∃ (u : ℝ × ℝ), (u.1 * u.1 + u.2 * u.2 = 1) ∧
  (∃ (k : ℝ), vector_a = (k * u.1, k * u.2)) ∧
  (u = (12/13, 5/13) ∨ u = (-12/13, -5/13)) :=
sorry

end NUMINAMATH_CALUDE_unit_vector_parallel_to_a_l1611_161194


namespace NUMINAMATH_CALUDE_f_seven_half_value_l1611_161104

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_seven_half_value 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_unit : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x) :
  f 7.5 = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_f_seven_half_value_l1611_161104


namespace NUMINAMATH_CALUDE_range_of_m_l1611_161199

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

-- Define the set of m values that satisfy the conditions
def S : Set ℝ := {m | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

-- State the theorem
theorem range_of_m : S = Set.Ioo 1 2 ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1611_161199


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1611_161111

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A a ∩ B a = {-3} → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1611_161111


namespace NUMINAMATH_CALUDE_simplify_expression_l1611_161178

theorem simplify_expression : (81 / 16) ^ (3 / 4) - (-1) ^ 0 = 19 / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1611_161178


namespace NUMINAMATH_CALUDE_ellipse_foci_ratio_l1611_161143

/-- The equation of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

/-- A point on the ellipse -/
def P : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Right-angled triangle formed by P, F₁, and F₂ -/
def is_right_triangle (P F₁ F₂ : ℝ × ℝ) : Prop := sorry

theorem ellipse_foci_ratio :
  is_on_ellipse P.1 P.2 →
  is_right_triangle P F₁ F₂ →
  distance P F₁ > distance P F₂ →
  (distance P F₁ / distance P F₂ = 7/2) ∨ (distance P F₁ / distance P F₂ = 2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_ratio_l1611_161143


namespace NUMINAMATH_CALUDE_geometric_sum_specific_l1611_161170

/-- Sum of a finite geometric series -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem: The sum of the first 6 terms of the geometric sequence with
    first term 1/5 and common ratio 1/5 is equal to 1953/7812 -/
theorem geometric_sum_specific : geometric_sum (1/5) (1/5) 6 = 1953/7812 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_specific_l1611_161170


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1611_161122

/-- Given a quadratic function f(x) = ax² + bx + c, if f(2) - f(-2) = 4, then b = 1. -/
theorem quadratic_coefficient (a b c : ℝ) (y₁ y₂ : ℝ) : 
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = 4 →
  b = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1611_161122


namespace NUMINAMATH_CALUDE_factorization_of_3m_squared_minus_12_l1611_161177

theorem factorization_of_3m_squared_minus_12 (m : ℝ) : 3 * m^2 - 12 = 3 * (m - 2) * (m + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_3m_squared_minus_12_l1611_161177


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l1611_161130

def pizza_sharing (eva gwen noah mia : ℚ) : Prop :=
  eva = 1/4 ∧ gwen = 1/6 ∧ noah = 1/5 ∧ mia = 1 - (eva + gwen + noah)

theorem pizza_consumption_order (eva gwen noah mia : ℚ) 
  (h : pizza_sharing eva gwen noah mia) : 
  eva > mia ∧ mia > noah ∧ noah > gwen :=
by
  sorry

#check pizza_consumption_order

end NUMINAMATH_CALUDE_pizza_consumption_order_l1611_161130


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1611_161131

theorem arithmetic_mean_of_fractions : 
  let a : ℚ := 3/4
  let b : ℚ := 5/8
  (a + b) / 2 = 11/16 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1611_161131


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1611_161128

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = 2) (h2 : b = -1) :
  (2 * a^2 - a * b - b^2) - 2 * (a^2 - 2 * a * b + b^2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1611_161128


namespace NUMINAMATH_CALUDE_inequality_proof_l1611_161126

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x, f (-x) = f x
axiom f_increasing : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 5 → f x < f y

-- State the theorem
theorem inequality_proof : f 4 > f (-Real.pi) ∧ f (-Real.pi) > f 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1611_161126


namespace NUMINAMATH_CALUDE_minimum_living_allowance_growth_l1611_161181

/-- The average annual growth rate of the minimum living allowance -/
def average_growth_rate : ℝ := 0.3

/-- The initial minimum living allowance in yuan -/
def initial_allowance : ℝ := 200

/-- The final minimum living allowance in yuan -/
def final_allowance : ℝ := 338

/-- The number of years -/
def years : ℕ := 2

theorem minimum_living_allowance_growth :
  initial_allowance * (1 + average_growth_rate) ^ years = final_allowance := by
  sorry

end NUMINAMATH_CALUDE_minimum_living_allowance_growth_l1611_161181


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l1611_161168

theorem quadratic_roots_difference (a b c : ℝ) (r₁ r₂ : ℝ) : 
  a * r₁^2 + b * r₁ + c = 0 →
  a * r₂^2 + b * r₂ + c = 0 →
  a = 1 →
  b = -8 →
  c = 15 →
  r₁ + r₂ = 8 →
  ∃ n : ℤ, (r₁ + r₂ : ℝ) = n^2 →
  r₁ - r₂ = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l1611_161168


namespace NUMINAMATH_CALUDE_max_common_segment_for_coprime_l1611_161186

/-- The maximum length of the common initial segment of two sequences with coprime periods -/
def max_common_segment (m n : ℕ) : ℕ :=
  m + n - 2

/-- Theorem: For coprime positive integers m and n, the maximum length of the common
    initial segment of two sequences with periods m and n respectively is m + n - 2 -/
theorem max_common_segment_for_coprime (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) 
    (h_coprime : Nat.Coprime m n) : 
  max_common_segment m n = m + n - 2 := by
  sorry

#check max_common_segment_for_coprime

end NUMINAMATH_CALUDE_max_common_segment_for_coprime_l1611_161186


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_and_parabola_l1611_161190

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the equation y^4 - 9x^6 = 3y^2 - 1 -/
def equation (p : Point) : Prop :=
  p.y^4 - 9*p.x^6 = 3*p.y^2 - 1

/-- Represents a hyperbola -/
def is_hyperbola (S : Set Point) : Prop :=
  ∃ a b c d e f : ℝ, ∀ p ∈ S, a*p.x^2 + b*p.y^2 + c*p.x*p.y + d*p.x + e*p.y + f = 0 ∧ a*b < 0

/-- Represents a parabola -/
def is_parabola (S : Set Point) : Prop :=
  ∃ a b c d : ℝ, a ≠ 0 ∧ ∀ p ∈ S, p.y = a*p.x^2 + b*p.x + c ∨ p.x = a*p.y^2 + b*p.y + d

/-- The theorem to be proved -/
theorem equation_represents_hyperbola_and_parabola :
  ∃ S₁ S₂ : Set Point,
    (∀ p, p ∈ S₁ ∪ S₂ ↔ equation p) ∧
    is_hyperbola S₁ ∧
    is_parabola S₂ :=
sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_and_parabola_l1611_161190


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l1611_161188

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem f_derivative_at_2 (a b : ℝ) :
  f a b 1 = -2 → (deriv (f a b)) 1 = 0 → (deriv (f a b)) 2 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l1611_161188


namespace NUMINAMATH_CALUDE_A_equals_B_l1611_161134

/-- Number of partitions of n where even parts are distinct -/
def A (n : ℕ) : ℕ := sorry

/-- Number of partitions of n where each part appears at most 3 times -/
def B (n : ℕ) : ℕ := sorry

/-- Theorem stating that A_n equals B_n for all natural numbers n -/
theorem A_equals_B : ∀ n : ℕ, A n = B n := by sorry

end NUMINAMATH_CALUDE_A_equals_B_l1611_161134


namespace NUMINAMATH_CALUDE_five_g_base_stations_scientific_notation_l1611_161150

theorem five_g_base_stations_scientific_notation :
  (819000 : ℝ) = 8.19 * (10 ^ 5) := by sorry

end NUMINAMATH_CALUDE_five_g_base_stations_scientific_notation_l1611_161150


namespace NUMINAMATH_CALUDE_problems_per_page_is_three_l1611_161174

/-- The number of problems on each page of homework -/
def problems_per_page : ℕ := sorry

/-- The number of pages of math homework -/
def math_pages : ℕ := 6

/-- The number of pages of reading homework -/
def reading_pages : ℕ := 4

/-- The total number of problems -/
def total_problems : ℕ := 30

/-- Theorem stating that the number of problems per page is 3 -/
theorem problems_per_page_is_three :
  problems_per_page = 3 ∧
  (math_pages + reading_pages) * problems_per_page = total_problems :=
sorry

end NUMINAMATH_CALUDE_problems_per_page_is_three_l1611_161174


namespace NUMINAMATH_CALUDE_print_shop_charges_l1611_161176

/-- 
Given:
- Print shop X charges $1.25 per color copy
- Print shop Y charges $60 more than print shop X for 40 color copies

Prove that print shop Y charges $2.75 per color copy
-/
theorem print_shop_charges (x y : ℝ) : 
  x = 1.25 → 
  40 * y = 40 * x + 60 → 
  y = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_charges_l1611_161176


namespace NUMINAMATH_CALUDE_geometric_mean_of_4_and_16_l1611_161127

theorem geometric_mean_of_4_and_16 (x : ℝ) :
  x ^ 2 = 4 * 16 → x = 8 ∨ x = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_4_and_16_l1611_161127


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1611_161183

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation 
  (train_length : Real) 
  (bridge_length : Real) 
  (crossing_time : Real) 
  (h1 : train_length = 100) 
  (h2 : bridge_length = 275) 
  (h3 : crossing_time = 30) : 
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1611_161183


namespace NUMINAMATH_CALUDE_sum_x_and_5_nonpositive_l1611_161112

theorem sum_x_and_5_nonpositive (x : ℝ) : (x + 5 ≤ 0) ↔ (∀ y : ℝ, y ≤ 0 → x + 5 ≤ y) := by
  sorry

end NUMINAMATH_CALUDE_sum_x_and_5_nonpositive_l1611_161112


namespace NUMINAMATH_CALUDE_vertex_of_our_parabola_l1611_161185

/-- A parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- Our specific parabola -/
def our_parabola : Parabola :=
  ⟨λ x => 2 * (x - 1)^2 + 5⟩

theorem vertex_of_our_parabola :
  vertex our_parabola = (1, 5) := by sorry

end NUMINAMATH_CALUDE_vertex_of_our_parabola_l1611_161185


namespace NUMINAMATH_CALUDE_puzzle_unique_solution_l1611_161113

/-- Represents a mapping from letters to digits -/
def LetterMapping := Char → Fin 10

/-- Checks if a mapping is valid (different letters map to different digits) -/
def is_valid_mapping (m : LetterMapping) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → m c1 ≠ m c2

/-- Converts a word to a number using the given mapping -/
def word_to_number (word : List Char) (m : LetterMapping) : ℕ :=
  word.foldl (fun acc d => 10 * acc + (m d).val) 0

/-- The cryptarithmetic puzzle equation -/
def puzzle_equation (m : LetterMapping) : Prop :=
  let dodge := word_to_number ['D', 'O', 'D', 'G', 'E'] m
  let strike := word_to_number ['S', 'T', 'R', 'I', 'K', 'E'] m
  let fighting := word_to_number ['F', 'I', 'G', 'H', 'T', 'I', 'N', 'G'] m
  dodge + strike = fighting

/-- The main theorem stating that the puzzle has a unique solution -/
theorem puzzle_unique_solution :
  ∃! m : LetterMapping, is_valid_mapping m ∧ puzzle_equation m :=
sorry

end NUMINAMATH_CALUDE_puzzle_unique_solution_l1611_161113


namespace NUMINAMATH_CALUDE_min_value_sin_2x_minus_pi_4_l1611_161102

theorem min_value_sin_2x_minus_pi_4 :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), Real.sin (2 * x - Real.pi / 4) ≥ -Real.sqrt 2 / 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), Real.sin (2 * x - Real.pi / 4) = -Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_sin_2x_minus_pi_4_l1611_161102


namespace NUMINAMATH_CALUDE_car_speed_problem_l1611_161123

theorem car_speed_problem (x : ℝ) :
  x > 0 →
  (x + 60) / 2 = 75 →
  x = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1611_161123


namespace NUMINAMATH_CALUDE_probability_three_girls_l1611_161114

/-- The probability of choosing 3 girls from a club with 15 members (8 girls and 7 boys) is 8/65 -/
theorem probability_three_girls (total : ℕ) (girls : ℕ) (boys : ℕ) (h1 : total = 15) (h2 : girls = 8) (h3 : boys = 7) (h4 : total = girls + boys) :
  (Nat.choose girls 3 : ℚ) / (Nat.choose total 3) = 8 / 65 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_girls_l1611_161114


namespace NUMINAMATH_CALUDE_boisjoli_farm_egg_boxes_l1611_161138

/-- Calculates the number of egg boxes filled per week given the number of hens,
    eggs per hen per day, eggs per box, and days per week. -/
def boxes_per_week (hens : ℕ) (eggs_per_hen_per_day : ℕ) (eggs_per_box : ℕ) (days_per_week : ℕ) : ℕ :=
  (hens * eggs_per_hen_per_day * days_per_week) / eggs_per_box

/-- Proves that given 270 hens, each laying one egg per day, packed in boxes of 6,
    collected 7 days a week, the total number of boxes filled per week is 315. -/
theorem boisjoli_farm_egg_boxes :
  boxes_per_week 270 1 6 7 = 315 := by
  sorry

end NUMINAMATH_CALUDE_boisjoli_farm_egg_boxes_l1611_161138


namespace NUMINAMATH_CALUDE_lindas_calculation_l1611_161129

theorem lindas_calculation (x y z : ℝ) 
  (h1 : x - (y + z) = 5) 
  (h2 : x - y + z = -1) : 
  x - y = 2 := by
  sorry

end NUMINAMATH_CALUDE_lindas_calculation_l1611_161129


namespace NUMINAMATH_CALUDE_parabolas_intersection_l1611_161106

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1
def parabola2 (x : ℝ) : ℝ := x^2 - 4 * x + 4

-- Define a function to check if a point is on both parabolas
def is_intersection (x y : ℝ) : Prop :=
  parabola1 x = y ∧ parabola2 x = y

-- Theorem statement
theorem parabolas_intersection :
  (is_intersection (-3) 25) ∧ 
  (is_intersection 1 1) ∧
  (∀ x y : ℝ, is_intersection x y → (x = -3 ∧ y = 25) ∨ (x = 1 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l1611_161106
