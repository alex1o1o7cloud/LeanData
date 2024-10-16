import Mathlib

namespace NUMINAMATH_CALUDE_wall_area_2_by_4_l929_92977

/-- The area of a rectangular wall -/
def wall_area (width height : ℝ) : ℝ := width * height

/-- Theorem: The area of a wall that is 2 feet wide and 4 feet tall is 8 square feet -/
theorem wall_area_2_by_4 : wall_area 2 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_wall_area_2_by_4_l929_92977


namespace NUMINAMATH_CALUDE_tony_age_l929_92918

/-- Given that Tony and Belinda have a combined age of 56, and Belinda is 40 years old,
    prove that Tony is 16 years old. -/
theorem tony_age (total_age : ℕ) (belinda_age : ℕ) (h1 : total_age = 56) (h2 : belinda_age = 40) :
  total_age - belinda_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_tony_age_l929_92918


namespace NUMINAMATH_CALUDE_mikes_muffins_l929_92965

/-- The number of muffins in a dozen -/
def dozen : ℕ := 12

/-- The number of boxes needed to pack all muffins -/
def boxes : ℕ := 8

/-- The total number of muffins Mike has -/
def total_muffins : ℕ := boxes * dozen

theorem mikes_muffins : total_muffins = 96 := by
  sorry

end NUMINAMATH_CALUDE_mikes_muffins_l929_92965


namespace NUMINAMATH_CALUDE_taxi_overtakes_bus_l929_92938

/-- 
Given a taxi and a bus with the following conditions:
- The taxi travels at 45 mph
- The bus travels 30 mph slower than the taxi
- The taxi starts 4 hours after the bus
This theorem proves that the taxi will overtake the bus in 2 hours.
-/
theorem taxi_overtakes_bus (taxi_speed : ℝ) (bus_speed : ℝ) (head_start : ℝ) 
  (overtake_time : ℝ) :
  taxi_speed = 45 →
  bus_speed = taxi_speed - 30 →
  head_start = 4 →
  overtake_time = 2 →
  taxi_speed * overtake_time = bus_speed * (overtake_time + head_start) :=
by
  sorry

#check taxi_overtakes_bus

end NUMINAMATH_CALUDE_taxi_overtakes_bus_l929_92938


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_2_l929_92972

/-- Hyperbola eccentricity theorem -/
theorem hyperbola_eccentricity_sqrt_2 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (asymptote_eq : ∀ x, b / a * x = x)
  (F2 : ℝ × ℝ)
  (hF2 : F2 = (c, 0))
  (M : ℝ × ℝ)
  (hM : M.1 = 0)
  (N : ℝ × ℝ)
  (perpendicular : (M.2 - N.2) * (b / a) = -(M.1 - N.1))
  (midpoint : N = ((F2.1 + M.1) / 2, (F2.2 + M.2) / 2))
  : c / a = Real.sqrt 2 := by
  sorry

#check hyperbola_eccentricity_sqrt_2

end NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_2_l929_92972


namespace NUMINAMATH_CALUDE_percentage_equation_solution_l929_92995

theorem percentage_equation_solution :
  ∃ x : ℝ, (12.4 * 350) + (9.9 * 275) = (8.6 * x) + (5.3 * (2250 - x)) := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_solution_l929_92995


namespace NUMINAMATH_CALUDE_candies_per_house_l929_92978

/-- Proves that the number of candies received from each house is 7,
    given that there are 5 houses in a block and 35 candies are received from each block. -/
theorem candies_per_house
  (houses_per_block : ℕ)
  (candies_per_block : ℕ)
  (h1 : houses_per_block = 5)
  (h2 : candies_per_block = 35) :
  candies_per_block / houses_per_block = 7 :=
by sorry

end NUMINAMATH_CALUDE_candies_per_house_l929_92978


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l929_92993

theorem deal_or_no_deal_probability (total_boxes : ℕ) (high_value_boxes : ℕ) (eliminated_boxes : ℕ) :
  total_boxes = 30 →
  high_value_boxes = 10 →
  eliminated_boxes = 20 →
  (total_boxes - eliminated_boxes : ℚ) / 2 ≤ high_value_boxes :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l929_92993


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_geq_2_l929_92994

-- Define the function f(x) = |x-a|
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a)

-- State the theorem
theorem monotone_decreasing_implies_a_geq_2 (a : ℝ) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 2 → f a x ≥ f a y) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_geq_2_l929_92994


namespace NUMINAMATH_CALUDE_cherry_pie_count_l929_92928

theorem cherry_pie_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) : 
  total_pies = 36 →
  apple_ratio = 2 →
  blueberry_ratio = 5 →
  cherry_ratio = 4 →
  (cherry_ratio : ℚ) * total_pies / (apple_ratio + blueberry_ratio + cherry_ratio) = 13 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pie_count_l929_92928


namespace NUMINAMATH_CALUDE_interval_intersection_l929_92945

theorem interval_intersection (x : ℝ) : 
  (2 < 3*x ∧ 3*x < 3) ∧ (2 < 4*x ∧ 4*x < 3) ↔ (2/3 < x ∧ x < 3/4) :=
by sorry

end NUMINAMATH_CALUDE_interval_intersection_l929_92945


namespace NUMINAMATH_CALUDE_teacher_student_probability_teacher_student_probability_correct_l929_92917

/-- The probability that neither teacher stands at either end when 2 teachers and 2 students
    stand in a row for a group photo. -/
theorem teacher_student_probability : ℚ :=
  let num_teachers : ℕ := 2
  let num_students : ℕ := 2
  let total_arrangements : ℕ := Nat.factorial 4
  let favorable_arrangements : ℕ := Nat.factorial 2 * Nat.factorial 2
  1 / 6

/-- Proof that the probability is correct. -/
theorem teacher_student_probability_correct : teacher_student_probability = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_teacher_student_probability_teacher_student_probability_correct_l929_92917


namespace NUMINAMATH_CALUDE_total_payment_calculation_l929_92923

def worker_count : ℝ := 2.5
def hourly_rate : ℝ := 15
def daily_hours : List ℝ := [12, 10, 8, 6, 14]

theorem total_payment_calculation :
  worker_count * hourly_rate * (daily_hours.sum) = 1875 := by sorry

end NUMINAMATH_CALUDE_total_payment_calculation_l929_92923


namespace NUMINAMATH_CALUDE_series_divergence_l929_92986

theorem series_divergence (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, a n ≤ a (2 * n) + a (2 * n + 1)) :
  ¬ (Summable a) :=
sorry

end NUMINAMATH_CALUDE_series_divergence_l929_92986


namespace NUMINAMATH_CALUDE_min_beta_delta_sum_min_beta_delta_value_l929_92953

open Complex

/-- The function g(z) as defined in the problem -/
def g (β δ : ℂ) (z : ℂ) : ℂ := (3 + 2*I)*z^2 + β*z + δ

/-- The theorem stating the minimum value of |β| + |δ| -/
theorem min_beta_delta_sum :
  ∃ (β δ : ℂ), 
    (g β δ 1).im = 0 ∧ 
    (g β δ (-I)).im = 0 ∧
    ∀ (β' δ' : ℂ), (g β' δ' 1).im = 0 → (g β' δ' (-I)).im = 0 → 
      abs β + abs δ ≤ abs β' + abs δ' :=
by
  sorry

/-- The actual minimum value of |β| + |δ| -/
theorem min_beta_delta_value :
  ∃ (β δ : ℂ), 
    (g β δ 1).im = 0 ∧ 
    (g β δ (-I)).im = 0 ∧
    abs β + abs δ = Real.sqrt 40 :=
by
  sorry

end NUMINAMATH_CALUDE_min_beta_delta_sum_min_beta_delta_value_l929_92953


namespace NUMINAMATH_CALUDE_ashoks_marks_l929_92941

theorem ashoks_marks (total_subjects : ℕ) (average_6_subjects : ℝ) (marks_6th_subject : ℝ) :
  total_subjects = 6 →
  average_6_subjects = 80 →
  marks_6th_subject = 110 →
  let total_marks := average_6_subjects * total_subjects
  let marks_5_subjects := total_marks - marks_6th_subject
  let average_5_subjects := marks_5_subjects / 5
  average_5_subjects = 74 := by
sorry

end NUMINAMATH_CALUDE_ashoks_marks_l929_92941


namespace NUMINAMATH_CALUDE_quadratic_equation_property_l929_92911

/-- A quadratic equation with two equal real roots -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  condition : a - b + c = 0
  equal_roots : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ (∀ y : ℝ, a * y^2 + b * y + c = 0 → y = x)

/-- Theorem stating that for a quadratic equation with two equal real roots and a - b + c = 0, we have 2a - b = 0 -/
theorem quadratic_equation_property (eq : QuadraticEquation) : 2 * eq.a - eq.b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_property_l929_92911


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l929_92982

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, 1 < x ∧ x < Real.pi / 2 → (x - 1) * Real.tan x > 0) ∧
  (∃ x : ℝ, (x - 1) * Real.tan x > 0 ∧ ¬(1 < x ∧ x < Real.pi / 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l929_92982


namespace NUMINAMATH_CALUDE_expression_simplification_l929_92933

theorem expression_simplification (x y : ℚ) 
  (hx : x = -1/2) (hy : y = 2) : 
  6 * (x^2 - (1/3) * x * y) - 3 * (x^2 - x * y) - 2 * x^2 = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l929_92933


namespace NUMINAMATH_CALUDE_roberto_outfits_l929_92999

/-- Represents the number of different outfits Roberto can create --/
def number_of_outfits (trousers shirts jackets constrained_trousers constrained_jackets : ℕ) : ℕ :=
  ((trousers - constrained_trousers) * jackets + constrained_trousers * constrained_jackets) * shirts

/-- Theorem stating the number of outfits Roberto can create given his wardrobe constraints --/
theorem roberto_outfits :
  let trousers : ℕ := 5
  let shirts : ℕ := 7
  let jackets : ℕ := 3
  let constrained_trousers : ℕ := 2
  let constrained_jackets : ℕ := 2
  number_of_outfits trousers shirts jackets constrained_trousers constrained_jackets = 91 :=
by
  sorry


end NUMINAMATH_CALUDE_roberto_outfits_l929_92999


namespace NUMINAMATH_CALUDE_future_age_difference_l929_92955

/-- Proves that the number of years in the future when the father's age will be 20 years more than twice the son's age is 4, given the conditions stated in the problem. -/
theorem future_age_difference (father_age son_age x : ℕ) : 
  father_age = 44 →
  father_age = 4 * son_age + 4 →
  father_age + x = 2 * (son_age + x) + 20 →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_future_age_difference_l929_92955


namespace NUMINAMATH_CALUDE_expression_equality_l929_92956

theorem expression_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x = y * z) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) * (x + y + y * z)⁻¹ * ((x * y)⁻¹ + (y * z)⁻¹ + (x * z)⁻¹) = 
  1 / (y^3 * z^3 * (y + 1)^2) :=
sorry

end NUMINAMATH_CALUDE_expression_equality_l929_92956


namespace NUMINAMATH_CALUDE_pencils_multiple_of_fifty_l929_92916

/-- Given a number of students, pens, and pencils, we define a valid distribution --/
def ValidDistribution (S P : ℕ) : Prop :=
  S > 0 ∧ S ≤ 50 ∧ 100 % S = 0 ∧ P % S = 0

/-- Theorem stating that the number of pencils must be a multiple of 50 --/
theorem pencils_multiple_of_fifty (P : ℕ) :
  (∃ S : ℕ, ValidDistribution S P) → P % 50 = 0 := by
  sorry

end NUMINAMATH_CALUDE_pencils_multiple_of_fifty_l929_92916


namespace NUMINAMATH_CALUDE_function_is_constant_l929_92942

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def is_continuous (f : ℝ → ℝ) : Prop := Continuous f

def satisfies_inequality (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (3 * x - 2) ≤ f x ∧ f x ≤ f (2 * x - 1)

-- State the theorem
theorem function_is_constant
  (h_continuous : is_continuous f)
  (h_inequality : satisfies_inequality f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end NUMINAMATH_CALUDE_function_is_constant_l929_92942


namespace NUMINAMATH_CALUDE_remainder_of_Q_mod_1000_l929_92907

theorem remainder_of_Q_mod_1000 :
  (202^1 + 20^21 + 2^21) % 1000 = 354 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_Q_mod_1000_l929_92907


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l929_92975

theorem quadratic_root_ratio (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l929_92975


namespace NUMINAMATH_CALUDE_berries_to_buy_l929_92980

def total_needed : ℕ := 36
def strawberries : ℕ := 4
def blueberries : ℕ := 8
def raspberries : ℕ := 3
def blackberries : ℕ := 5

theorem berries_to_buy (total_needed strawberries blueberries raspberries blackberries : ℕ) :
  total_needed - (strawberries + blueberries + raspberries + blackberries) = 16 := by
  sorry

end NUMINAMATH_CALUDE_berries_to_buy_l929_92980


namespace NUMINAMATH_CALUDE_power_set_intersection_nonempty_l929_92904

theorem power_set_intersection_nonempty :
  ∃ (A B : Set α), (A ∩ B).Nonempty ∧ (𝒫 A ∩ 𝒫 B).Nonempty :=
sorry

end NUMINAMATH_CALUDE_power_set_intersection_nonempty_l929_92904


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l929_92963

open Real

theorem trigonometric_equation_solution (x : ℝ) :
  (2 * cos (π + x) - 5 * cos ((3/2) * π - x)) / (cos ((3/2) * π + x) - cos (π - x)) = 3/2 ↔
  ∃ k : ℤ, x = (π/4) * (4 * k + 1) := by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l929_92963


namespace NUMINAMATH_CALUDE_inequality_holds_iff_equal_l929_92912

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The inequality holds for all real α and β iff m = n -/
theorem inequality_holds_iff_equal (m n : ℕ+) : 
  (∀ α β : ℝ, floor ((m + n : ℝ) * α) + floor ((m + n : ℝ) * β) ≥ 
    floor (m * α) + floor (n * β) + floor (n * (α + β))) ↔ m = n := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_equal_l929_92912


namespace NUMINAMATH_CALUDE_unique_solution_cube_root_equation_l929_92924

def f (x : ℝ) := (20 * x + (20 * x + 13) ^ (1/3)) ^ (1/3)

theorem unique_solution_cube_root_equation :
  ∃! x : ℝ, f x = 13 ∧ x = 546/5 := by sorry

end NUMINAMATH_CALUDE_unique_solution_cube_root_equation_l929_92924


namespace NUMINAMATH_CALUDE_magical_stack_131_l929_92949

/-- Definition of a magical stack -/
def is_magical_stack (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≤ n ∧ b > n ∧ b ≤ 2*n ∧
  (a = 2*a - 1 ∨ b = 2*(b - n))

/-- Theorem: A stack with 392 cards where card 131 retains its position is magical -/
theorem magical_stack_131 :
  ∃ (n : ℕ), 2*n = 392 ∧ is_magical_stack n ∧ 131 ≤ n ∧ 131 = 2*131 - 1 := by
  sorry

end NUMINAMATH_CALUDE_magical_stack_131_l929_92949


namespace NUMINAMATH_CALUDE_sally_bought_48_eggs_l929_92940

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Sally bought -/
def dozens_bought : ℕ := 4

/-- Theorem: Sally bought 48 eggs -/
theorem sally_bought_48_eggs : dozens_bought * eggs_per_dozen = 48 := by
  sorry

end NUMINAMATH_CALUDE_sally_bought_48_eggs_l929_92940


namespace NUMINAMATH_CALUDE_new_species_growth_pattern_l929_92968

/-- Represents the shape of population growth --/
inductive GrowthShape
  | J
  | S

/-- Represents a species in a new area --/
structure Species where
  isNew : Bool
  populationSize : ℕ → ℕ  -- population size as a function of time
  growthPattern : List GrowthShape
  kValue : ℕ

/-- The maximum population allowed by environmental conditions --/
def environmentalCapacity (s : Species) : ℕ := s.kValue

theorem new_species_growth_pattern (s : Species) 
  (h1 : s.isNew = true) 
  (h2 : ∀ t, s.populationSize (t + 1) ≠ s.populationSize t) 
  (h3 : s.growthPattern.length ≥ 2) 
  (h4 : ∃ t, ∀ t' ≥ t, s.populationSize t' = s.kValue) 
  (h5 : s.kValue = environmentalCapacity s) :
  s.growthPattern = [GrowthShape.J, GrowthShape.S] := by
  sorry

end NUMINAMATH_CALUDE_new_species_growth_pattern_l929_92968


namespace NUMINAMATH_CALUDE_largest_increase_1998_l929_92979

def sales : ℕ → ℕ
| 0 => 3000    -- 1994
| 1 => 4500    -- 1995
| 2 => 6000    -- 1996
| 3 => 6750    -- 1997
| 4 => 8400    -- 1998
| 5 => 9000    -- 1999
| 6 => 9600    -- 2000
| 7 => 10400   -- 2001
| 8 => 9500    -- 2002
| 9 => 6500    -- 2003
| _ => 0       -- undefined for other years

def salesIncrease (year : ℕ) : ℤ :=
  (sales (year + 1) : ℤ) - (sales year : ℤ)

theorem largest_increase_1998 :
  ∀ y : ℕ, y ≥ 0 ∧ y < 9 → salesIncrease 4 ≥ salesIncrease y :=
by sorry

end NUMINAMATH_CALUDE_largest_increase_1998_l929_92979


namespace NUMINAMATH_CALUDE_digit_product_le_unique_solution_l929_92929

-- Define p(n) as the product of digits of n
def digit_product (n : ℕ) : ℕ := sorry

-- Theorem 1: For any natural number n, p(n) ≤ n
theorem digit_product_le (n : ℕ) : digit_product n ≤ n := by sorry

-- Theorem 2: 45 is the only natural number satisfying 10p(n) = n^2 + 4n - 2005
theorem unique_solution :
  ∀ n : ℕ, 10 * (digit_product n) = n^2 + 4*n - 2005 ↔ n = 45 := by sorry

end NUMINAMATH_CALUDE_digit_product_le_unique_solution_l929_92929


namespace NUMINAMATH_CALUDE_fraction_equality_l929_92910

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) :
  1 / 7 + (2 * q - p) / (2 * q + p) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l929_92910


namespace NUMINAMATH_CALUDE_midpoint_theorem_l929_92935

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

-- Define the potential midpoints
def midpoint1 : ℝ × ℝ := (1, 1)
def midpoint2 : ℝ × ℝ := (-1, 2)
def midpoint3 : ℝ × ℝ := (1, 3)
def midpoint4 : ℝ × ℝ := (-1, -4)

-- Define a function to check if a point is a valid midpoint
def is_valid_midpoint (m : ℝ × ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    hyperbola x1 y1 ∧ hyperbola x2 y2 ∧
    m.1 = (x1 + x2) / 2 ∧ m.2 = (y1 + y2) / 2

-- Theorem statement
theorem midpoint_theorem :
  ¬(is_valid_midpoint midpoint1) ∧
  ¬(is_valid_midpoint midpoint2) ∧
  ¬(is_valid_midpoint midpoint3) ∧
  is_valid_midpoint midpoint4 := by sorry

end NUMINAMATH_CALUDE_midpoint_theorem_l929_92935


namespace NUMINAMATH_CALUDE_cos_A_eq_one_l929_92926

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_A_eq_C (q : Quadrilateral) : Prop :=
  sorry -- ∠A = ∠C

def side_AB_eq_240 (q : Quadrilateral) : ℝ :=
  sorry -- Distance between A and B

def side_CD_eq_240 (q : Quadrilateral) : ℝ :=
  sorry -- Distance between C and D

def side_AD_ne_BC (q : Quadrilateral) : Prop :=
  sorry -- AD ≠ BC

def perimeter_eq_960 (q : Quadrilateral) : ℝ :=
  sorry -- Perimeter of ABCD

-- Theorem statement
theorem cos_A_eq_one (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_angle : angle_A_eq_C q)
  (h_AB : side_AB_eq_240 q = 240)
  (h_CD : side_CD_eq_240 q = 240)
  (h_AD_ne_BC : side_AD_ne_BC q)
  (h_perimeter : perimeter_eq_960 q = 960) :
  let cos_A := sorry -- Definition of cos A for the quadrilateral
  cos_A = 1 := by sorry

end NUMINAMATH_CALUDE_cos_A_eq_one_l929_92926


namespace NUMINAMATH_CALUDE_negative_cube_squared_l929_92946

theorem negative_cube_squared (x : ℝ) : (-x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l929_92946


namespace NUMINAMATH_CALUDE_team_selection_count_l929_92936

def people : Finset Char := {'a', 'b', 'c', 'd', 'e'}

theorem team_selection_count :
  let all_selections := (people.powerset.filter (fun s => s.card = 2)).card
  let invalid_selections := (people.erase 'a').card
  all_selections - invalid_selections = 16 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l929_92936


namespace NUMINAMATH_CALUDE_crease_length_l929_92990

/-- A right triangle with sides 5, 12, and 13 inches -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  a_eq_5 : a = 5
  b_eq_12 : b = 12
  c_eq_13 : c = 13

/-- The length of the perpendicular bisector from the right angle to the hypotenuse -/
def perp_bisector_length (t : RightTriangle) : ℝ := t.b

theorem crease_length (t : RightTriangle) :
  perp_bisector_length t = 12 := by sorry

end NUMINAMATH_CALUDE_crease_length_l929_92990


namespace NUMINAMATH_CALUDE_edwards_spending_l929_92913

theorem edwards_spending (initial_amount : ℚ) : 
  initial_amount - 130 - (0.25 * (initial_amount - 130)) = 270 → 
  initial_amount = 490 := by
sorry

end NUMINAMATH_CALUDE_edwards_spending_l929_92913


namespace NUMINAMATH_CALUDE_positive_poly_nonneg_ratio_l929_92984

/-- A polynomial with real coefficients -/
def RealPolynomial := Polynomial ℝ

/-- A polynomial with nonnegative real coefficients -/
def NonnegPolynomial := {p : RealPolynomial // ∀ i, 0 ≤ p.coeff i}

/-- The theorem statement -/
theorem positive_poly_nonneg_ratio
  (P : RealPolynomial)
  (h : ∀ x : ℝ, 0 < x → 0 < P.eval x) :
  ∃ (Q R : NonnegPolynomial), ∀ x : ℝ, 0 < x →
    P.eval x = (Q.val.eval x) / (R.val.eval x) :=
sorry

end NUMINAMATH_CALUDE_positive_poly_nonneg_ratio_l929_92984


namespace NUMINAMATH_CALUDE_mary_nickels_theorem_l929_92959

/-- The number of nickels Mary's dad gave her -/
def nickels_from_dad (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

theorem mary_nickels_theorem (initial : ℕ) (final : ℕ) 
  (h1 : initial = 7) 
  (h2 : final = 12) : 
  nickels_from_dad initial final = 5 := by
  sorry

end NUMINAMATH_CALUDE_mary_nickels_theorem_l929_92959


namespace NUMINAMATH_CALUDE_repeating_block_length_seven_thirteenths_l929_92944

/-- The length of the smallest repeating block in the decimal expansion of 7/13 is 6. -/
theorem repeating_block_length_seven_thirteenths : 
  ∃ (d : ℕ) (n : ℕ), d = 6 ∧ 7 * (10^d - 1) = 13 * n :=
by sorry

end NUMINAMATH_CALUDE_repeating_block_length_seven_thirteenths_l929_92944


namespace NUMINAMATH_CALUDE_min_value_sum_of_inverses_l929_92973

theorem min_value_sum_of_inverses (x y z p q r : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (pos_p : p > 0) (pos_q : q > 0) (pos_r : r > 0)
  (sum_eq_10 : x + y + z + p + q + r = 10) :
  1/x + 9/y + 4/z + 25/p + 16/q + 36/r ≥ 441/10 ∧ 
  ∃ (x' y' z' p' q' r' : ℝ), 
    x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ p' > 0 ∧ q' > 0 ∧ r' > 0 ∧
    x' + y' + z' + p' + q' + r' = 10 ∧
    1/x' + 9/y' + 4/z' + 25/p' + 16/q' + 36/r' = 441/10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_inverses_l929_92973


namespace NUMINAMATH_CALUDE_tangent_condition_l929_92900

theorem tangent_condition (a b : ℝ) : 
  (∃ (x y : ℝ), x + y = 0 ∧ (x - a)^2 + (y - b)^2 = 2) → 
  (a + b = 2 → ∃ (x y : ℝ), x + y = 0 ∧ (x - a)^2 + (y - b)^2 = 2) ∧
  (∃ (a b : ℝ), (∃ (x y : ℝ), x + y = 0 ∧ (x - a)^2 + (y - b)^2 = 2) ∧ a + b ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_condition_l929_92900


namespace NUMINAMATH_CALUDE_scientific_notation_450_million_l929_92998

theorem scientific_notation_450_million :
  (450000000 : ℝ) = 4.5 * (10 : ℝ)^8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_450_million_l929_92998


namespace NUMINAMATH_CALUDE_max_value_implies_ratio_l929_92943

/-- Given a function f(x) = x^3 + ax^2 + bx - a^2 - 7a that reaches its maximum value of 10 at x = 1,
    prove that a/b = -2/3 -/
theorem max_value_implies_ratio (a b : ℝ) :
  let f := fun (x : ℝ) => x^3 + a*x^2 + b*x - a^2 - 7*a
  (∀ x, f x ≤ f 1) ∧ (f 1 = 10) → a/b = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_ratio_l929_92943


namespace NUMINAMATH_CALUDE_range_of_a_l929_92947

theorem range_of_a (a : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2*y + 2*z) →
  (a ≤ -2 ∨ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l929_92947


namespace NUMINAMATH_CALUDE_william_claire_game_bounds_l929_92967

/-- A move in William's strategy -/
inductive Move
| reciprocal : Move  -- Replace y with 1/y
| increment  : Move  -- Replace y with y+1

/-- William's strategy for rearranging the numbers -/
def Strategy := List Move

/-- The result of applying a strategy to a sequence of numbers -/
def applyStrategy (s : Strategy) (xs : List ℝ) : List ℝ := sorry

/-- Predicate to check if a list is strictly increasing -/
def isStrictlyIncreasing (xs : List ℝ) : Prop := sorry

/-- The theorem to be proved -/
theorem william_claire_game_bounds :
  ∃ (A B : ℝ) (hA : A > 0) (hB : B > 0),
    ∀ (n : ℕ) (hn : n > 1),
      -- Part (a): William can always succeed in at most An log n moves
      (∀ (xs : List ℝ) (hxs : xs.length = n) (hdistinct : xs.Nodup),
        ∃ (s : Strategy),
          isStrictlyIncreasing (applyStrategy s xs) ∧
          s.length ≤ A * n * Real.log n) ∧
      -- Part (b): Claire can force William to use at least Bn log n moves
      (∃ (xs : List ℝ) (hxs : xs.length = n) (hdistinct : xs.Nodup),
        ∀ (s : Strategy),
          isStrictlyIncreasing (applyStrategy s xs) →
          s.length ≥ B * n * Real.log n) :=
sorry

end NUMINAMATH_CALUDE_william_claire_game_bounds_l929_92967


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l929_92974

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose 18 (3*n + 6) = Nat.choose 18 (4*n - 2)) ↔ n = 2 :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l929_92974


namespace NUMINAMATH_CALUDE_election_vote_count_l929_92927

/-- Represents an election with two candidates -/
structure TwoCandidateElection where
  totalVotes : ℕ
  loserPercentage : ℚ
  voteDifference : ℕ

/-- 
Theorem: In a two-candidate election where the losing candidate received 40% of the votes
and lost by 5000 votes, the total number of votes cast was 25000.
-/
theorem election_vote_count (e : TwoCandidateElection) 
  (h1 : e.loserPercentage = 40 / 100)
  (h2 : e.voteDifference = 5000) : 
  e.totalVotes = 25000 := by
  sorry

#eval (40 : ℚ) / 100  -- To verify the rational number representation

end NUMINAMATH_CALUDE_election_vote_count_l929_92927


namespace NUMINAMATH_CALUDE_max_rabbit_population_l929_92969

/-- Represents the properties of a rabbit population --/
structure RabbitPopulation where
  total : ℕ
  longEars : ℕ
  jumpFar : ℕ
  bothTraits : ℕ

/-- Checks if a rabbit population satisfies the given conditions --/
def isValidPopulation (pop : RabbitPopulation) : Prop :=
  pop.longEars = 13 ∧
  pop.jumpFar = 17 ∧
  pop.bothTraits ≥ 3 ∧
  pop.longEars + pop.jumpFar - pop.bothTraits ≤ pop.total

/-- Theorem stating that 27 is the maximum number of rabbits satisfying the conditions --/
theorem max_rabbit_population :
  ∀ (pop : RabbitPopulation), isValidPopulation pop → pop.total ≤ 27 :=
sorry

end NUMINAMATH_CALUDE_max_rabbit_population_l929_92969


namespace NUMINAMATH_CALUDE_parallel_lines_m_l929_92919

/-- Two lines are parallel if their slopes are equal -/
def parallel (a b c d e f : ℝ) : Prop :=
  a * e = b * d ∧ a * f ≠ b * c

/-- The problem statement -/
theorem parallel_lines_m (m : ℝ) :
  parallel m 1 (-1) 9 m (-(2 * m + 3)) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_l929_92919


namespace NUMINAMATH_CALUDE_polynomial_factorization_l929_92970

theorem polynomial_factorization :
  (∀ x : ℝ, x^2 + 14*x + 49 = (x + 7)^2) ∧
  (∀ m n : ℝ, (m - 1) + n^2*(1 - m) = (m - 1)*(1 - n)*(1 + n)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l929_92970


namespace NUMINAMATH_CALUDE_replaced_person_weight_l929_92992

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (new_person_weight : ℝ) (average_increase : ℝ) : ℝ :=
  new_person_weight - (initial_count * average_increase)

/-- Theorem stating the weight of the replaced person under given conditions -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 97 4 = 65 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l929_92992


namespace NUMINAMATH_CALUDE_windows_preference_count_l929_92908

theorem windows_preference_count (total : ℕ) (mac_pref : ℕ) (no_pref : ℕ) : 
  total = 210 → 
  mac_pref = 60 → 
  no_pref = 90 → 
  ∃ (windows_pref : ℕ), 
    windows_pref = total - (mac_pref + (mac_pref / 3) + no_pref) ∧ 
    windows_pref = 40 := by
  sorry

end NUMINAMATH_CALUDE_windows_preference_count_l929_92908


namespace NUMINAMATH_CALUDE_talitha_took_108_pieces_l929_92976

/-- Given an initial candy count, the number of pieces Solomon took, and the final candy count,
    calculate the number of pieces Talitha took. -/
def talitha_candy_count (initial : ℕ) (solomon_took : ℕ) (final : ℕ) : ℕ :=
  initial - solomon_took - final

/-- Theorem stating that Talitha took 108 pieces of candy. -/
theorem talitha_took_108_pieces :
  talitha_candy_count 349 153 88 = 108 := by
  sorry

end NUMINAMATH_CALUDE_talitha_took_108_pieces_l929_92976


namespace NUMINAMATH_CALUDE_problem_proof_l929_92925

theorem problem_proof : (1 / (2 - Real.sqrt 3)) - 1 - 2 * (Real.sqrt 3 / 2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_problem_proof_l929_92925


namespace NUMINAMATH_CALUDE_equation_is_linear_l929_92954

/-- A linear equation with two variables is of the form ax + by = c, where a, b, and c are constants, and a and b are not both zero. -/
def IsLinearEquationWithTwoVariables (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0), ∀ x y, f x y ↔ a * x + b * y = c

/-- The equation 2x = 3y + 1 -/
def Equation (x y : ℝ) : Prop := 2 * x = 3 * y + 1

theorem equation_is_linear : IsLinearEquationWithTwoVariables Equation := by
  sorry

end NUMINAMATH_CALUDE_equation_is_linear_l929_92954


namespace NUMINAMATH_CALUDE_computer_operations_per_hour_l929_92988

theorem computer_operations_per_hour :
  let additions_per_second : ℕ := 12000
  let multiplications_per_second : ℕ := 8000
  let seconds_per_hour : ℕ := 3600
  let total_operations_per_second : ℕ := additions_per_second + multiplications_per_second
  let operations_per_hour : ℕ := total_operations_per_second * seconds_per_hour
  operations_per_hour = 72000000 := by
sorry

end NUMINAMATH_CALUDE_computer_operations_per_hour_l929_92988


namespace NUMINAMATH_CALUDE_combinatorial_equality_l929_92906

theorem combinatorial_equality (n : ℕ) : 
  (n.choose 2) * 2 = 42 → n.choose 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_equality_l929_92906


namespace NUMINAMATH_CALUDE_matrix_sum_proof_l929_92905

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 2, 5]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 8; -3, -10]

theorem matrix_sum_proof :
  A + B = !![(-2 : ℤ), 5; -1, -5] := by sorry

end NUMINAMATH_CALUDE_matrix_sum_proof_l929_92905


namespace NUMINAMATH_CALUDE_line_m_equation_l929_92903

-- Define the xy-plane
def xy_plane : Set (ℝ × ℝ) := Set.univ

-- Define lines ℓ and m
def line_ℓ : Set (ℝ × ℝ) := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 = 0}
def line_m : Set (ℝ × ℝ) := {p : ℝ × ℝ | 7 * p.1 - p.2 = 0}

-- Define points
def Q : ℝ × ℝ := (-3, 2)
def Q'' : ℝ × ℝ := (-4, -3)

-- Define the reflection operation (as a placeholder, actual implementation not provided)
def reflect (point : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

theorem line_m_equation :
  line_ℓ ⊆ xy_plane ∧
  line_m ⊆ xy_plane ∧
  line_ℓ ≠ line_m ∧
  (0, 0) ∈ line_ℓ ∩ line_m ∧
  Q ∈ xy_plane ∧
  Q'' ∈ xy_plane ∧
  reflect (reflect Q line_ℓ) line_m = Q'' →
  line_m = {p : ℝ × ℝ | 7 * p.1 - p.2 = 0} :=
by sorry

end NUMINAMATH_CALUDE_line_m_equation_l929_92903


namespace NUMINAMATH_CALUDE_outfit_combinations_l929_92991

theorem outfit_combinations (shirts : ℕ) (ties : ℕ) : shirts = 7 → ties = 6 → shirts * (ties + 1) = 49 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l929_92991


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l929_92948

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = Complex.I) : 
  z.im = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l929_92948


namespace NUMINAMATH_CALUDE_smallest_greater_discount_l929_92915

def discount_sequence_1 (x : ℝ) : ℝ := (1 - 0.2) * (1 - 0.1) * x
def discount_sequence_2 (x : ℝ) : ℝ := (1 - 0.08)^3 * x
def discount_sequence_3 (x : ℝ) : ℝ := (1 - 0.15) * (1 - 0.12) * x

def effective_discount_1 (x : ℝ) : ℝ := x - discount_sequence_1 x
def effective_discount_2 (x : ℝ) : ℝ := x - discount_sequence_2 x
def effective_discount_3 (x : ℝ) : ℝ := x - discount_sequence_3 x

theorem smallest_greater_discount : 
  ∀ x > 0, 
    effective_discount_1 x / x < 0.29 ∧ 
    effective_discount_2 x / x < 0.29 ∧ 
    effective_discount_3 x / x < 0.29 ∧
    ∀ n : ℕ, n < 29 → 
      (effective_discount_1 x / x > n / 100 ∨ 
       effective_discount_2 x / x > n / 100 ∨ 
       effective_discount_3 x / x > n / 100) :=
by sorry

end NUMINAMATH_CALUDE_smallest_greater_discount_l929_92915


namespace NUMINAMATH_CALUDE_correct_participants_cars_needed_rental_plans_and_min_cost_l929_92939

/-- Represents the number of teachers -/
def teachers : ℕ := 6

/-- Represents the number of students -/
def students : ℕ := 234

/-- Represents the total number of participants -/
def total_participants : ℕ := teachers + students

/-- Represents the capacity of bus A -/
def bus_A_capacity : ℕ := 45

/-- Represents the capacity of bus B -/
def bus_B_capacity : ℕ := 30

/-- Represents the rental cost of bus A -/
def bus_A_cost : ℕ := 400

/-- Represents the rental cost of bus B -/
def bus_B_cost : ℕ := 280

/-- Represents the total rental cost limit -/
def total_cost_limit : ℕ := 2300

/-- Theorem stating the correctness of the number of teachers and students -/
theorem correct_participants : teachers = 6 ∧ students = 234 ∧
  38 * teachers + 6 = students ∧ 40 * teachers - 6 = students := by sorry

/-- Theorem stating the number of cars needed -/
theorem cars_needed : ∃ (n : ℕ), n = 6 ∧ 
  n * bus_A_capacity ≥ total_participants ∧
  n ≥ teachers := by sorry

/-- Theorem stating the number of rental car plans and minimum cost -/
theorem rental_plans_and_min_cost : 
  ∃ (plans : ℕ) (min_cost : ℕ), plans = 2 ∧ min_cost = 2160 ∧
  ∀ (x : ℕ), 4 ≤ x ∧ x ≤ 5 →
    x * bus_A_capacity + (6 - x) * bus_B_capacity ≥ total_participants ∧
    x * bus_A_cost + (6 - x) * bus_B_cost ≤ total_cost_limit ∧
    (x = 4 → x * bus_A_cost + (6 - x) * bus_B_cost = min_cost) := by sorry

end NUMINAMATH_CALUDE_correct_participants_cars_needed_rental_plans_and_min_cost_l929_92939


namespace NUMINAMATH_CALUDE_hall_breadth_l929_92961

/-- The breadth of a hall given its length, number of stones, and stone dimensions. -/
theorem hall_breadth (hall_length : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ) : 
  hall_length = 36 →
  num_stones = 3600 →
  stone_length = 0.3 →
  stone_width = 0.5 →
  hall_length * (num_stones * stone_length * stone_width / hall_length) = 15 := by
  sorry

end NUMINAMATH_CALUDE_hall_breadth_l929_92961


namespace NUMINAMATH_CALUDE_intersection_A_B_l929_92951

def A : Set ℤ := {1, 2, 3}

def B : Set ℤ := {x | (x + 1) * (x - 2) < 0}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l929_92951


namespace NUMINAMATH_CALUDE_prove_c_value_l929_92957

theorem prove_c_value (c : ℕ) : 
  (5 ^ 5) * (9 ^ 3) = c * (15 ^ 5) → c = 3 → c = 3 := by
  sorry

end NUMINAMATH_CALUDE_prove_c_value_l929_92957


namespace NUMINAMATH_CALUDE_shoe_ratio_problem_l929_92952

/-- Proof of the shoe ratio problem -/
theorem shoe_ratio_problem (brian_shoes edward_shoes jacob_shoes : ℕ) : 
  (edward_shoes = 3 * brian_shoes) →
  (brian_shoes = 22) →
  (jacob_shoes + edward_shoes + brian_shoes = 121) →
  (jacob_shoes : ℚ) / edward_shoes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_shoe_ratio_problem_l929_92952


namespace NUMINAMATH_CALUDE_books_loaned_out_special_collection_loaned_books_l929_92922

/-- Proves that the number of books loaned out during the month is 20 --/
theorem books_loaned_out 
  (initial_books : ℕ) 
  (final_books : ℕ) 
  (return_rate : ℚ) : ℕ :=
  let loaned_books := (initial_books - final_books) / (1 - return_rate)
  20

/-- Given conditions --/
def initial_books : ℕ := 75
def final_books : ℕ := 68
def return_rate : ℚ := 65 / 100

/-- Main theorem --/
theorem special_collection_loaned_books : 
  books_loaned_out initial_books final_books return_rate = 20 := by
  sorry

end NUMINAMATH_CALUDE_books_loaned_out_special_collection_loaned_books_l929_92922


namespace NUMINAMATH_CALUDE_circle_tangency_radius_l929_92966

theorem circle_tangency_radius (r_P r_Q r_R : ℝ) : 
  r_P = 4 ∧ 
  r_Q = 4 * r_R ∧ 
  r_P > r_Q ∧ 
  r_P > r_R ∧
  r_Q > r_R ∧
  r_P = r_Q + r_R →
  r_Q = 16 ∧ 
  r_Q = Real.sqrt 256 - 0 := by
sorry

end NUMINAMATH_CALUDE_circle_tangency_radius_l929_92966


namespace NUMINAMATH_CALUDE_wage_percentage_proof_l929_92962

def company_finances (revenue : ℝ) (num_employees : ℕ) (tax_rate : ℝ) 
  (marketing_rate : ℝ) (operational_rate : ℝ) (employee_wage : ℝ) : Prop :=
  let after_tax := revenue * (1 - tax_rate)
  let after_marketing := after_tax * (1 - marketing_rate)
  let after_operational := after_marketing * (1 - operational_rate)
  let total_wages := num_employees * employee_wage
  total_wages / after_operational = 0.15

theorem wage_percentage_proof :
  company_finances 400000 10 0.10 0.05 0.20 4104 := by
  sorry

end NUMINAMATH_CALUDE_wage_percentage_proof_l929_92962


namespace NUMINAMATH_CALUDE_pictures_in_first_album_l929_92909

def total_pictures : ℕ := 25
def num_other_albums : ℕ := 5
def pics_per_other_album : ℕ := 3

theorem pictures_in_first_album :
  total_pictures - (num_other_albums * pics_per_other_album) = 10 := by
  sorry

end NUMINAMATH_CALUDE_pictures_in_first_album_l929_92909


namespace NUMINAMATH_CALUDE_fran_speed_l929_92985

/-- Given Joann's bike ride parameters and Fran's time, calculate Fran's required speed --/
theorem fran_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5) :
  (joann_speed * joann_time) / fran_time = 60 / 3.5 := by
  sorry

end NUMINAMATH_CALUDE_fran_speed_l929_92985


namespace NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_leq_neg_one_l929_92930

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a - 1)*x + 2

-- State the theorem
theorem monotonic_decreasing_implies_a_leq_neg_one :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → y ≤ 2 → f a x > f a y) → a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_leq_neg_one_l929_92930


namespace NUMINAMATH_CALUDE_parker_dumbbell_weight_l929_92931

/-- Given an initial setup of dumbbells and additional dumbbells added, 
    calculate the total weight Parker is using for his exercises. -/
theorem parker_dumbbell_weight 
  (initial_count : ℕ) 
  (additional_count : ℕ) 
  (weight_per_dumbbell : ℕ) : 
  initial_count = 4 → 
  additional_count = 2 → 
  weight_per_dumbbell = 20 → 
  (initial_count + additional_count) * weight_per_dumbbell = 120 := by
  sorry

end NUMINAMATH_CALUDE_parker_dumbbell_weight_l929_92931


namespace NUMINAMATH_CALUDE_parallel_vectors_dot_product_l929_92901

/-- Given two vectors a and b in ℝ², if a is parallel to b and a = (1,3) and b = (-3,x), 
    then their dot product is -30 -/
theorem parallel_vectors_dot_product (x : ℝ) : 
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (-3, x)
  (∃ (k : ℝ), b = k • a) → a.1 * b.1 + a.2 * b.2 = -30 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_dot_product_l929_92901


namespace NUMINAMATH_CALUDE_unique_point_on_curve_l929_92934

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def second_quadrant (x y : ℤ) : Prop := x < 0 ∧ y > 0

/-- A point (x, y) is on the curve if x^2 + 5x + 1 = 3y -/
def on_curve (x y : ℤ) : Prop := x^2 + 5*x + 1 = 3*y

theorem unique_point_on_curve : 
  ∀ x y : ℤ, second_quadrant x y → on_curve x y → (x = -7 ∧ y = 5) :=
by sorry

end NUMINAMATH_CALUDE_unique_point_on_curve_l929_92934


namespace NUMINAMATH_CALUDE_dance_event_relationship_l929_92971

/-- Represents a dance event with boys and girls. -/
structure DanceEvent where
  boys : ℕ
  girls : ℕ
  first_boy_dances : ℕ
  increment : ℕ

/-- The relationship between boys and girls in a specific dance event. -/
def dance_relationship (event : DanceEvent) : Prop :=
  event.boys = (event.girls - 4) / 2

/-- Theorem stating the relationship between boys and girls in the dance event. -/
theorem dance_event_relationship :
  ∀ (event : DanceEvent),
  event.first_boy_dances = 6 →
  event.increment = 2 →
  (∀ n : ℕ, n < event.boys → event.first_boy_dances + n * event.increment ≤ event.girls) →
  event.first_boy_dances + (event.boys - 1) * event.increment = event.girls →
  dance_relationship event :=
sorry

end NUMINAMATH_CALUDE_dance_event_relationship_l929_92971


namespace NUMINAMATH_CALUDE_A_finishes_in_two_days_l929_92958

/-- The number of days A needs to finish the remaining work after B worked for 10 days -/
def days_for_A_to_finish (days_A days_B : ℕ) (days_B_worked : ℕ) : ℚ :=
  (1 - (days_B_worked : ℚ) / days_B) / (1 / days_A)

/-- Theorem stating that A needs 2 days to finish the remaining work -/
theorem A_finishes_in_two_days :
  days_for_A_to_finish 6 15 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_A_finishes_in_two_days_l929_92958


namespace NUMINAMATH_CALUDE_parabola_properties_l929_92950

-- Define the parabola function
def f (x : ℝ) : ℝ := (x - 1)^2 - 2

-- Theorem stating the properties of the parabola
theorem parabola_properties :
  (∀ x y : ℝ, f x = y → ∃ a : ℝ, a > 0 ∧ y = a * (x - 1)^2 - 2) ∧ 
  (∀ x y : ℝ, f x = y → f (2 - x) = y) ∧
  (f 1 = -2 ∧ ∀ x : ℝ, f x ≥ -2) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ > 1 ∧ x₁ > x₂ → f x₁ > f x₂) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l929_92950


namespace NUMINAMATH_CALUDE_range_of_f_l929_92987

/-- The function f defined on real numbers. -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- The range of f is [1, +∞) -/
theorem range_of_f : Set.range f = {y : ℝ | y ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l929_92987


namespace NUMINAMATH_CALUDE_max_trees_on_road_l929_92989

theorem max_trees_on_road (road_length : ℕ) (interval : ℕ) (h1 : road_length = 28) (h2 : interval = 4) :
  (road_length / interval) + 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_trees_on_road_l929_92989


namespace NUMINAMATH_CALUDE_rogers_nickels_l929_92902

theorem rogers_nickels :
  ∀ (N : ℕ),
  (42 + N + 15 : ℕ) - 66 = 27 →
  N = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_rogers_nickels_l929_92902


namespace NUMINAMATH_CALUDE_total_gold_value_l929_92983

/-- Calculates the total value of gold for Legacy, Aleena, and Briana -/
theorem total_gold_value (legacy_bars : ℕ) (aleena_bars_diff : ℕ) (briana_bars : ℕ)
  (legacy_aleena_value : ℕ) (briana_value : ℕ) :
  legacy_bars = 12 →
  aleena_bars_diff = 4 →
  briana_bars = 8 →
  legacy_aleena_value = 3500 →
  briana_value = 4000 →
  (legacy_bars * legacy_aleena_value) +
  ((legacy_bars - aleena_bars_diff) * legacy_aleena_value) +
  (briana_bars * briana_value) = 102000 :=
by sorry

end NUMINAMATH_CALUDE_total_gold_value_l929_92983


namespace NUMINAMATH_CALUDE_exists_acute_triangle_l929_92996

/-- Given five positive real numbers that can form triangles in any combination of three,
    there exists at least one acute-angled triangle among them. -/
theorem exists_acute_triangle
  (a b c d e : ℝ)
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) (pos_e : e > 0)
  (triangle_abc : a + b > c ∧ b + c > a ∧ c + a > b)
  (triangle_abd : a + b > d ∧ b + d > a ∧ d + a > b)
  (triangle_abe : a + b > e ∧ b + e > a ∧ e + a > b)
  (triangle_acd : a + c > d ∧ c + d > a ∧ d + a > c)
  (triangle_ace : a + c > e ∧ c + e > a ∧ e + a > c)
  (triangle_ade : a + d > e ∧ d + e > a ∧ e + a > d)
  (triangle_bcd : b + c > d ∧ c + d > b ∧ d + b > c)
  (triangle_bce : b + c > e ∧ c + e > b ∧ e + b > c)
  (triangle_bde : b + d > e ∧ d + e > b ∧ e + b > d)
  (triangle_cde : c + d > e ∧ d + e > c ∧ e + c > d) :
  ∃ (x y z : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
                 (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
                 (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
                 x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
                 x^2 + y^2 > z^2 ∧ y^2 + z^2 > x^2 ∧ z^2 + x^2 > y^2 :=
sorry

end NUMINAMATH_CALUDE_exists_acute_triangle_l929_92996


namespace NUMINAMATH_CALUDE_election_winner_percentage_l929_92960

theorem election_winner_percentage (winner_votes loser_votes : ℕ) : 
  winner_votes = 750 →
  winner_votes - loser_votes = 500 →
  (winner_votes : ℚ) / (winner_votes + loser_votes : ℚ) * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l929_92960


namespace NUMINAMATH_CALUDE_additional_tank_capacity_l929_92920

theorem additional_tank_capacity
  (existing_tanks : ℕ)
  (fish_per_existing_tank : ℕ)
  (additional_tanks : ℕ)
  (total_fish : ℕ)
  (h1 : existing_tanks = 3)
  (h2 : fish_per_existing_tank = 15)
  (h3 : additional_tanks = 3)
  (h4 : total_fish = 75) :
  (total_fish - existing_tanks * fish_per_existing_tank) / additional_tanks = 10 :=
by sorry

end NUMINAMATH_CALUDE_additional_tank_capacity_l929_92920


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l929_92932

theorem sin_2alpha_value (α : Real) 
  (h1 : α > 0 ∧ α < Real.pi) 
  (h2 : 3 * Real.cos (2 * α) - 4 * Real.cos α + 1 = 0) : 
  Real.sin (2 * α) = -4 * Real.sqrt 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l929_92932


namespace NUMINAMATH_CALUDE_pages_copied_example_l929_92981

/-- Given a cost per page in cents, a flat service charge in cents, and a total budget in cents,
    calculate the maximum number of pages that can be copied. -/
def max_pages_copied (cost_per_page : ℕ) (service_charge : ℕ) (total_budget : ℕ) : ℕ :=
  (total_budget - service_charge) / cost_per_page

/-- Prove that with a cost of 3 cents per page, a flat service charge of 500 cents,
    and a total budget of 5000 cents, the maximum number of pages that can be copied is 1500. -/
theorem pages_copied_example : max_pages_copied 3 500 5000 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_example_l929_92981


namespace NUMINAMATH_CALUDE_periodic_function_decomposition_l929_92937

-- Define the type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define the property of being 2π-periodic
def isPeriodic2Pi (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (x + 2 * Real.pi) = f x

-- Define the property of being an even function
def isEven (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define the property of being π-periodic
def isPeriodicPi (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (x + Real.pi) = f x

theorem periodic_function_decomposition (f : RealFunction) (h : isPeriodic2Pi f) :
  ∃ (f₁ f₂ f₃ f₄ : RealFunction),
    (∀ i ∈ [f₁, f₂, f₃, f₄], isEven i ∧ isPeriodicPi i) ∧
    (∀ x : ℝ, f x = f₁ x + f₂ x * Real.cos x + f₃ x * Real.sin x + f₄ x * Real.sin (2 * x)) :=
sorry

end NUMINAMATH_CALUDE_periodic_function_decomposition_l929_92937


namespace NUMINAMATH_CALUDE_boys_count_l929_92997

/-- Represents the number of boys on the chess team -/
def boys : ℕ := sorry

/-- Represents the number of girls on the chess team -/
def girls : ℕ := sorry

/-- The total number of team members is 30 -/
axiom total_members : boys + girls = 30

/-- 18 members attended the last meeting -/
axiom attendees : (2 * girls / 3 : ℚ) + boys = 18

/-- Proves that the number of boys on the chess team is 6 -/
theorem boys_count : boys = 6 := by sorry

end NUMINAMATH_CALUDE_boys_count_l929_92997


namespace NUMINAMATH_CALUDE_train_length_l929_92914

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 58 → time_s = 9 → 
  ∃ length_m : ℝ, abs (length_m - 144.99) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l929_92914


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l929_92921

/-- The function f(x) = x^3 - 2x^2 + 5x - 1 -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5*x - 1

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x + 5

theorem f_derivative_at_one : f' 1 = 4 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l929_92921


namespace NUMINAMATH_CALUDE_circles_tangent_internally_l929_92964

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y + 1 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 17 = 0

-- Define the center and radius of C1
def center_C1 : ℝ × ℝ := (-1, -2)
def radius_C1 : ℝ := 2

-- Define the center and radius of C2
def center_C2 : ℝ × ℝ := (2, -2)
def radius_C2 : ℝ := 5

-- Define the distance between centers
def distance_between_centers : ℝ := 3

-- Theorem stating that the circles are tangent internally
theorem circles_tangent_internally :
  distance_between_centers = abs (radius_C2 - radius_C1) ∧
  distance_between_centers < radius_C1 + radius_C2 :=
sorry

end NUMINAMATH_CALUDE_circles_tangent_internally_l929_92964
