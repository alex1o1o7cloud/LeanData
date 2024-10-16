import Mathlib

namespace NUMINAMATH_CALUDE_expression_value_l2806_280699

theorem expression_value (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  (x - 3)^2 + (x + 4)*(x - 4) = -9 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2806_280699


namespace NUMINAMATH_CALUDE_prob_log_is_integer_l2806_280669

/-- A four-digit positive integer -/
def FourDigitInt := {n : ℕ // 1000 ≤ n ∧ n ≤ 9999}

/-- The count of four-digit positive integers -/
def countFourDigitInts : ℕ := 9000

/-- Predicate for N being a power of 10 -/
def isPowerOfTen (N : FourDigitInt) : Prop :=
  ∃ k : ℕ, N.val = 10^k

/-- The count of four-digit numbers that are powers of 10 -/
def countPowersOfTen : ℕ := 1

/-- The probability of a randomly chosen four-digit number being a power of 10 -/
def probPowerOfTen : ℚ :=
  countPowersOfTen / countFourDigitInts

theorem prob_log_is_integer :
  probPowerOfTen = 1 / 9000 := by sorry

end NUMINAMATH_CALUDE_prob_log_is_integer_l2806_280669


namespace NUMINAMATH_CALUDE_intersection_points_l2806_280675

theorem intersection_points (x : ℝ) : 
  (∃ y : ℝ, y = 10 / (x^2 + 1) ∧ x^2 + y = 3) ↔ 
  (x = Real.sqrt (1 + 2 * Real.sqrt 2) ∨ x = -Real.sqrt (1 + 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_l2806_280675


namespace NUMINAMATH_CALUDE_triangle_circles_area_sum_l2806_280639

theorem triangle_circles_area_sum (r s t : ℝ) : 
  r > 0 ∧ s > 0 ∧ t > 0 →
  r + s = 5 →
  r + t = 12 →
  s + t = 13 →
  π * (r^2 + s^2 + t^2) = 113 * π :=
by sorry

end NUMINAMATH_CALUDE_triangle_circles_area_sum_l2806_280639


namespace NUMINAMATH_CALUDE_subtract_decimals_l2806_280612

theorem subtract_decimals : 34.25 - 0.45 = 33.8 := by
  sorry

end NUMINAMATH_CALUDE_subtract_decimals_l2806_280612


namespace NUMINAMATH_CALUDE_statements_evaluation_l2806_280655

theorem statements_evaluation :
  (∃ a b : ℝ, a > b ∧ ¬(a^2 > b^2)) ∧
  (∀ a b : ℝ, |a| > |b| → a^2 > b^2) ∧
  (∃ a b c : ℝ, a > b ∧ ¬(a*c^2 > b*c^2)) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → 1/a < 1/b) := by
  sorry


end NUMINAMATH_CALUDE_statements_evaluation_l2806_280655


namespace NUMINAMATH_CALUDE_aira_rubber_bands_l2806_280600

theorem aira_rubber_bands (samantha aira joe : ℕ) : 
  samantha = aira + 5 →
  joe = aira + 1 →
  samantha + aira + joe = 18 →
  aira = 4 := by
sorry

end NUMINAMATH_CALUDE_aira_rubber_bands_l2806_280600


namespace NUMINAMATH_CALUDE_bingley_has_four_bracelets_l2806_280605

/-- The number of bracelets Bingley has remaining after the exchanges -/
def bingley_remaining_bracelets : ℕ :=
  let bingley_initial : ℕ := 5
  let kelly_initial : ℕ := 16
  let kelly_gives : ℕ := kelly_initial / 4 / 3
  let bingley_after_receiving : ℕ := bingley_initial + kelly_gives
  let bingley_gives : ℕ := bingley_after_receiving / 3
  bingley_after_receiving - bingley_gives

/-- Theorem stating that Bingley has 4 bracelets remaining -/
theorem bingley_has_four_bracelets : bingley_remaining_bracelets = 4 := by
  sorry

end NUMINAMATH_CALUDE_bingley_has_four_bracelets_l2806_280605


namespace NUMINAMATH_CALUDE_correlation_identification_l2806_280646

-- Define the relationships
inductive Relationship
| TeacherStudent
| SphereVolume
| AppleProduction
| CrowsCawing
| TreeDimensions
| StudentID

-- Define a property for correlation
def has_correlation : Relationship → Prop
| Relationship.TeacherStudent => true
| Relationship.SphereVolume => false
| Relationship.AppleProduction => true
| Relationship.CrowsCawing => false
| Relationship.TreeDimensions => true
| Relationship.StudentID => false

-- Theorem statement
theorem correlation_identification :
  (∀ r : Relationship, has_correlation r ↔ 
    (r = Relationship.TeacherStudent ∨ 
     r = Relationship.AppleProduction ∨ 
     r = Relationship.TreeDimensions)) := by
  sorry

end NUMINAMATH_CALUDE_correlation_identification_l2806_280646


namespace NUMINAMATH_CALUDE_problem_proof_l2806_280632

theorem problem_proof : ((12^12 / 12^11)^2 * 4^2) / 2^4 = 144 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l2806_280632


namespace NUMINAMATH_CALUDE_count_special_numbers_l2806_280629

/-- Represents a relation where two integers have the same digits (possibly rearranged) -/
def digit_rearrangement (a b : ℕ) : Prop := sorry

/-- Counts the number of digits in a natural number -/
def digit_count (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is an 11-digit positive integer -/
def is_11_digit_positive (n : ℕ) : Prop :=
  digit_count n = 11 ∧ n > 0

/-- The main theorem stating the count of numbers satisfying the given conditions -/
theorem count_special_numbers :
  ∃ (S : Finset ℕ),
    (∀ K ∈ S, is_11_digit_positive K ∧ 2 ∣ K ∧ 3 ∣ K ∧ 5 ∣ K ∧
      ∃ K', digit_rearrangement K K' ∧
        7 ∣ K' ∧ 11 ∣ K' ∧ 13 ∣ K' ∧ 17 ∣ K' ∧ 101 ∣ K' ∧ 9901 ∣ K') ∧
    S.card = 3628800 :=
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_l2806_280629


namespace NUMINAMATH_CALUDE_train_crossing_time_l2806_280689

/-- Time taken for a faster train to cross a slower train moving in the same direction -/
theorem train_crossing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 300) 
  (h2 : length2 = 500)
  (h3 : speed1 = 72)
  (h4 : speed2 = 36)
  (h5 : speed1 > speed2) : 
  (length1 + length2) / ((speed1 - speed2) * (1000 / 3600)) = 80 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2806_280689


namespace NUMINAMATH_CALUDE_janes_shopping_theorem_l2806_280625

theorem janes_shopping_theorem :
  ∀ (s f : ℕ),
  s + f = 7 →
  (90 * s + 60 * f) % 100 = 0 →
  s = 4 :=
by sorry

end NUMINAMATH_CALUDE_janes_shopping_theorem_l2806_280625


namespace NUMINAMATH_CALUDE_parabola_b_value_l2806_280610

/-- Given a parabola y = ax^2 + bx + c with vertex (p, -p) and passing through (0, p),
    where p ≠ 0, the value of b is -4/p. -/
theorem parabola_b_value (a b c p : ℝ) (h_p : p ≠ 0) :
  (∀ x, a * x^2 + b * x + c = a * (x - p)^2 - p) →
  (a * 0^2 + b * 0 + c = p) →
  b = -4 / p := by sorry

end NUMINAMATH_CALUDE_parabola_b_value_l2806_280610


namespace NUMINAMATH_CALUDE_apple_pyramid_theorem_l2806_280698

/-- Calculates the number of apples in a layer of the pyramid -/
def apples_in_layer (base_width : ℕ) (base_length : ℕ) (layer : ℕ) : ℕ :=
  (base_width - layer + 1) * (base_length - layer + 1)

/-- Calculates the total number of apples in the pyramid stack -/
def total_apples (base_width : ℕ) (base_length : ℕ) : ℕ :=
  let num_layers := min base_width base_length
  (List.range num_layers).foldl (fun acc i => acc + apples_in_layer base_width base_length i) 0 + 1

theorem apple_pyramid_theorem :
  total_apples 6 9 = 155 := by
  sorry

#eval total_apples 6 9

end NUMINAMATH_CALUDE_apple_pyramid_theorem_l2806_280698


namespace NUMINAMATH_CALUDE_project_hours_difference_l2806_280683

/-- Given a project where three people charged time, prove that one person charged 100 more hours than another. -/
theorem project_hours_difference (total_hours kate_hours pat_hours mark_hours : ℕ) : 
  total_hours = 180 ∧ 
  pat_hours = 2 * kate_hours ∧ 
  pat_hours * 3 = mark_hours ∧
  total_hours = kate_hours + pat_hours + mark_hours →
  mark_hours = kate_hours + 100 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_difference_l2806_280683


namespace NUMINAMATH_CALUDE_factor_and_divisor_properties_l2806_280636

theorem factor_and_divisor_properties :
  (∃ n : ℕ, 25 = 5 * n) ∧
  (∃ m : ℕ, 171 = 9 * m) ∧
  ¬(209 % 19 = 0 ∧ 57 % 19 ≠ 0) ∧
  (90 % 30 = 0 ∨ 75 % 30 = 0) ∧
  ¬(51 % 17 = 0 ∧ 68 % 17 ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_factor_and_divisor_properties_l2806_280636


namespace NUMINAMATH_CALUDE_problem_solution_l2806_280623

def f (x a : ℝ) : ℝ := |2*x - a| + |2*x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x (-1) ≤ 2 ↔ -1/2 ≤ x ∧ x ≤ 1/2) ∧
  ((∀ x : ℝ, 1/2 ≤ x ∧ x ≤ 1 → f x a ≤ |2*x + 1|) → 0 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2806_280623


namespace NUMINAMATH_CALUDE_balance_spheres_l2806_280649

/-- Represents the density of a material -/
structure Density where
  value : ℝ
  positive : value > 0

/-- Represents the volume of a sphere -/
structure Volume where
  value : ℝ
  positive : value > 0

/-- Represents the mass of a sphere -/
structure Mass where
  value : ℝ
  positive : value > 0

/-- Represents a sphere with its properties -/
structure Sphere where
  density : Density
  volume : Volume
  mass : Mass

/-- Theorem: Balance of two spheres in air -/
theorem balance_spheres (cast_iron wood : Sphere) (air_density : Density) : 
  cast_iron.density.value > wood.density.value →
  cast_iron.volume.value < wood.volume.value →
  cast_iron.mass.value < wood.mass.value →
  (cast_iron.density.value - air_density.value) * cast_iron.volume.value = 
  (wood.density.value - air_density.value) * wood.volume.value →
  ∃ (fulcrum_position : ℝ), 
    fulcrum_position > 0 ∧ 
    fulcrum_position < 1 ∧ 
    fulcrum_position * cast_iron.mass.value = (1 - fulcrum_position) * wood.mass.value :=
by
  sorry

end NUMINAMATH_CALUDE_balance_spheres_l2806_280649


namespace NUMINAMATH_CALUDE_james_training_hours_l2806_280606

/-- James' Olympic training schedule and yearly hours --/
theorem james_training_hours :
  (sessions_per_day : ℕ) →
  (hours_per_session : ℕ) →
  (training_days_per_week : ℕ) →
  (weeks_per_year : ℕ) →
  sessions_per_day = 2 →
  hours_per_session = 4 →
  training_days_per_week = 5 →
  weeks_per_year = 52 →
  sessions_per_day * hours_per_session * training_days_per_week * weeks_per_year = 2080 :=
by sorry

end NUMINAMATH_CALUDE_james_training_hours_l2806_280606


namespace NUMINAMATH_CALUDE_simplify_fraction_l2806_280647

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2806_280647


namespace NUMINAMATH_CALUDE_smallest_number_range_l2806_280666

theorem smallest_number_range 
  (a b c d e : ℝ) 
  (h_distinct : a < b ∧ b < c ∧ c < d ∧ d < e) 
  (h_sum1 : a + b = 20) 
  (h_sum2 : a + c = 200) 
  (h_sum3 : d + e = 2014) 
  (h_sum4 : c + e = 2000) : 
  -793 < a ∧ a < 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_range_l2806_280666


namespace NUMINAMATH_CALUDE_electricity_billing_theorem_l2806_280622

/-- Represents a three-tariff meter reading --/
structure MeterReading where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  h_ordered : a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f

/-- Represents tariff prices --/
structure TariffPrices where
  t₁ : ℝ
  t₂ : ℝ
  t₃ : ℝ

/-- Calculates the maximum additional payment --/
def maxAdditionalPayment (reading : MeterReading) (prices : TariffPrices) (actualPayment : ℝ) : ℝ :=
  sorry

/-- Calculates the expected value of the difference --/
def expectedDifference (reading : MeterReading) (prices : TariffPrices) (actualPayment : ℝ) : ℝ :=
  sorry

/-- Main theorem --/
theorem electricity_billing_theorem (reading : MeterReading) (prices : TariffPrices) :
  let actualPayment := 660.72
  prices.t₁ = 4.03 ∧ prices.t₂ = 1.01 ∧ prices.t₃ = 3.39 →
  reading.a = 1214 ∧ reading.b = 1270 ∧ reading.c = 1298 ∧
  reading.d = 1337 ∧ reading.e = 1347 ∧ reading.f = 1402 →
  maxAdditionalPayment reading prices actualPayment = 397.34 ∧
  expectedDifference reading prices actualPayment = 19.30 :=
sorry

end NUMINAMATH_CALUDE_electricity_billing_theorem_l2806_280622


namespace NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_l2806_280692

theorem a_gt_one_sufficient_not_necessary (a : ℝ) (h : a ≠ 0) :
  (∀ a, a > 1 → a > 1/a) ∧ (∃ a, a > 1/a ∧ a ≤ 1) := by sorry

end NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_l2806_280692


namespace NUMINAMATH_CALUDE_smallest_K_for_inequality_l2806_280630

theorem smallest_K_for_inequality : 
  ∃ (K : ℝ), K = Real.sqrt 6 / 3 ∧ 
  (∀ (a b c : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 → 
    K + (a + b + c) / 3 ≥ (K + 1) * Real.sqrt ((a^2 + b^2 + c^2) / 3)) ∧
  (∀ (K' : ℝ), K' > 0 ∧ K' < K → 
    ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧
      K' + (a + b + c) / 3 < (K' + 1) * Real.sqrt ((a^2 + b^2 + c^2) / 3)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_K_for_inequality_l2806_280630


namespace NUMINAMATH_CALUDE_zoo_lion_cubs_l2806_280690

theorem zoo_lion_cubs (initial_count final_count : ℕ) 
  (gorillas_sent : ℕ) (hippo_adopted : ℕ) (rhinos_taken : ℕ) : 
  initial_count = 68 →
  gorillas_sent = 6 →
  hippo_adopted = 1 →
  rhinos_taken = 3 →
  final_count = 90 →
  ∃ (lion_cubs : ℕ), 
    final_count = initial_count - gorillas_sent + hippo_adopted + rhinos_taken + lion_cubs + 2 * lion_cubs ∧
    lion_cubs = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_lion_cubs_l2806_280690


namespace NUMINAMATH_CALUDE_johns_tax_rate_johns_tax_rate_approx_30_percent_l2806_280659

/-- Calculates John's tax rate given the incomes and tax rates of John and Ingrid --/
theorem johns_tax_rate (john_income ingrid_income : ℝ) 
                       (ingrid_tax_rate combined_tax_rate : ℝ) : ℝ :=
  let total_income := john_income + ingrid_income
  let total_tax := combined_tax_rate * total_income
  let ingrid_tax := ingrid_tax_rate * ingrid_income
  let john_tax := total_tax - ingrid_tax
  john_tax / john_income

/-- John's tax rate is approximately 30.00% --/
theorem johns_tax_rate_approx_30_percent : 
  ∃ ε > 0, 
    |johns_tax_rate 58000 72000 0.40 0.3554 - 0.30| < ε ∧ 
    ε < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_johns_tax_rate_johns_tax_rate_approx_30_percent_l2806_280659


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_gt_one_l2806_280626

theorem solution_set_reciprocal_gt_one (x : ℝ) : 1 / x > 1 ↔ 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_gt_one_l2806_280626


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l2806_280644

/-- Calculates the maximum number of tiles that can fit on a rectangular floor --/
def max_tiles (floor_length floor_width tile_length tile_width : ℕ) : ℕ :=
  let orientation1 := (floor_length / tile_length) * (floor_width / tile_width)
  let orientation2 := (floor_length / tile_width) * (floor_width / tile_length)
  max orientation1 orientation2

/-- Theorem stating the maximum number of tiles that can be accommodated on the given floor --/
theorem max_tiles_on_floor :
  max_tiles 180 120 25 16 = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l2806_280644


namespace NUMINAMATH_CALUDE_cross_sectional_area_of_cone_l2806_280685

-- Define the cone
structure Cone :=
  (baseRadius : ℝ)
  (height : ℝ)

-- Define the cutting plane
structure CuttingPlane :=
  (distanceFromBase : ℝ)
  (isParallelToBase : Bool)

-- Theorem statement
theorem cross_sectional_area_of_cone (c : Cone) (p : CuttingPlane) :
  c.baseRadius = 2 →
  p.distanceFromBase = c.height / 2 →
  p.isParallelToBase = true →
  (π : ℝ) = π := by sorry

end NUMINAMATH_CALUDE_cross_sectional_area_of_cone_l2806_280685


namespace NUMINAMATH_CALUDE_smallest_prime_not_three_l2806_280664

theorem smallest_prime_not_three : ¬(∀ p : ℕ, Prime p → p ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_not_three_l2806_280664


namespace NUMINAMATH_CALUDE_vector_ratio_theorem_l2806_280640

theorem vector_ratio_theorem (a b : ℝ × ℝ) :
  let angle := Real.pi / 3
  let magnitude (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)
  let sum := (a.1 + b.1, a.2 + b.2)
  (∃ (d : ℝ), magnitude b = magnitude a + d ∧ magnitude sum = magnitude a + 2*d) →
  (a.1 * b.1 + a.2 * b.2 = magnitude a * magnitude b * Real.cos angle) →
  ∃ (k : ℝ), k > 0 ∧ magnitude a = 3*k ∧ magnitude b = 5*k ∧ magnitude sum = 7*k := by
sorry

end NUMINAMATH_CALUDE_vector_ratio_theorem_l2806_280640


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2806_280656

theorem ratio_x_to_y (x y : ℚ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) :
  x / y = 23 / 24 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2806_280656


namespace NUMINAMATH_CALUDE_absent_children_l2806_280682

/-- Proves that the number of absent children is 70 given the conditions of the problem -/
theorem absent_children (total_children : ℕ) (sweets_per_child : ℕ) (extra_sweets : ℕ) : 
  total_children = 190 →
  sweets_per_child = 38 →
  extra_sweets = 14 →
  (total_children - (total_children - sweets_per_child * total_children / (sweets_per_child - extra_sweets))) = 70 := by
  sorry

end NUMINAMATH_CALUDE_absent_children_l2806_280682


namespace NUMINAMATH_CALUDE_probability_of_defective_product_l2806_280643

/-- Given a product with three grades (first-grade, second-grade, and defective),
    prove that the probability of selecting a defective product is 0.05,
    given the probabilities of selecting first-grade and second-grade products. -/
theorem probability_of_defective_product
  (p_first : ℝ)
  (p_second : ℝ)
  (h_first : p_first = 0.65)
  (h_second : p_second = 0.3)
  (h_nonneg_first : 0 ≤ p_first)
  (h_nonneg_second : 0 ≤ p_second)
  (h_sum_le_one : p_first + p_second ≤ 1) :
  1 - (p_first + p_second) = 0.05 := by
sorry

end NUMINAMATH_CALUDE_probability_of_defective_product_l2806_280643


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2806_280662

theorem sqrt_fraction_simplification : 
  Real.sqrt (16 / 25 + 9 / 4) = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2806_280662


namespace NUMINAMATH_CALUDE_circle_radius_determines_c_l2806_280611

/-- The equation of a circle with center (h, k) and radius r can be written as
    (x - h)^2 + (y - k)^2 = r^2 -/
def CircleEquation (h k r c : ℝ) : Prop :=
  ∀ x y, x^2 + 6*x + y^2 - 4*y + c = 0 ↔ (x + 3)^2 + (y - 2)^2 = r^2

theorem circle_radius_determines_c : 
  ∀ c : ℝ, (CircleEquation (-3) 2 4 c) → c = -3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_determines_c_l2806_280611


namespace NUMINAMATH_CALUDE_parallelogram_probability_l2806_280651

-- Define the vertices of the parallelogram
def P : ℝ × ℝ := (4, 4)
def Q : ℝ × ℝ := (-2, -2)
def R : ℝ × ℝ := (-8, -2)
def S : ℝ × ℝ := (-2, 4)

-- Define the line y = -1
def line (x : ℝ) : ℝ := -1

-- Define the area of a parallelogram given base and height
def parallelogram_area (base height : ℝ) : ℝ := base * height

-- Theorem statement
theorem parallelogram_probability : 
  let total_area := parallelogram_area (P.1 - S.1) (P.2 - Q.2)
  let below_line_area := parallelogram_area (P.1 - S.1) 1
  below_line_area / total_area = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_parallelogram_probability_l2806_280651


namespace NUMINAMATH_CALUDE_x_intercept_implies_m_slope_implies_m_l2806_280601

/-- The equation of line l -/
def line_equation (m x y : ℝ) : Prop :=
  (m^2 - 2*m - 3)*x + (2*m^2 + m - 1)*y = 2*m - 6

/-- The x-intercept of line l is -3 -/
def x_intercept (m : ℝ) : Prop :=
  line_equation m (-3) 0

/-- The slope of line l is -1 -/
def slope_negative_one (m : ℝ) : Prop :=
  m^2 - 2*m - 3 = -(2*m^2 + m - 1) ∧ m^2 - 2*m - 3 ≠ 0

theorem x_intercept_implies_m (m : ℝ) :
  x_intercept m → m = -5/3 :=
sorry

theorem slope_implies_m (m : ℝ) :
  slope_negative_one m → m = 4/3 :=
sorry

end NUMINAMATH_CALUDE_x_intercept_implies_m_slope_implies_m_l2806_280601


namespace NUMINAMATH_CALUDE_power_of_two_equality_l2806_280617

theorem power_of_two_equality (x : ℕ) : (1 / 8 : ℝ) * 2^33 = 2^x → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l2806_280617


namespace NUMINAMATH_CALUDE_student_mistake_fraction_l2806_280678

theorem student_mistake_fraction (x y : ℚ) : 
  (x / y) * 576 = 480 → x / y = 5 / 6 :=
by sorry

end NUMINAMATH_CALUDE_student_mistake_fraction_l2806_280678


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_k_range_l2806_280677

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k + 3}

-- State the theorem
theorem intersection_nonempty_implies_k_range (k : ℝ) :
  (M ∩ N k).Nonempty → k ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_k_range_l2806_280677


namespace NUMINAMATH_CALUDE_journey_length_l2806_280658

theorem journey_length : 
  ∀ (L : ℝ) (T : ℝ),
  L = 60 * T →
  L = 50 * (T + 3/4) →
  L = 225 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_length_l2806_280658


namespace NUMINAMATH_CALUDE_log_cutting_problem_l2806_280667

/-- Represents the number of cuts needed to split a log into 1-meter pieces -/
def cuts_needed (length : ℕ) : ℕ := length - 1

/-- Represents the total number of logs -/
def total_logs : ℕ := 30

/-- Represents the total length of all logs in meters -/
def total_length : ℕ := 100

/-- Represents the possible lengths of logs in meters -/
inductive LogLength
| short : LogLength  -- 3 meters
| long : LogLength   -- 4 meters

/-- Calculates the minimum number of cuts needed for the given log configuration -/
def min_cuts (x y : ℕ) : Prop :=
  x + y = total_logs ∧
  3 * x + 4 * y = total_length ∧
  x * cuts_needed 3 + y * cuts_needed 4 = 70

theorem log_cutting_problem :
  ∃ x y : ℕ, min_cuts x y :=
sorry

end NUMINAMATH_CALUDE_log_cutting_problem_l2806_280667


namespace NUMINAMATH_CALUDE_hyperbola_center_l2806_280670

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 432 * y - 783 = 0

/-- The center of a hyperbola -/
def is_center (c : ℝ × ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    eq x y ↔ (x - c.1)^2 / a^2 - (y - c.2)^2 / b^2 = 1

/-- Theorem: The center of the given hyperbola is (3, 6) -/
theorem hyperbola_center : is_center (3, 6) hyperbola_eq :=
sorry

end NUMINAMATH_CALUDE_hyperbola_center_l2806_280670


namespace NUMINAMATH_CALUDE_largest_number_l2806_280635

theorem largest_number : 
  let numbers : List ℝ := [0.9791, 0.97019, 0.97909, 0.971, 0.97109]
  ∀ x ∈ numbers, x ≤ 0.9791 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l2806_280635


namespace NUMINAMATH_CALUDE_square_eq_nine_solutions_l2806_280616

theorem square_eq_nine_solutions (x : ℝ) : (x + 1)^2 = 9 ↔ x = 2 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_eq_nine_solutions_l2806_280616


namespace NUMINAMATH_CALUDE_greatest_prime_factor_f_24_l2806_280676

def f (m : ℕ) : ℕ := Finset.prod (Finset.range (m/2)) (fun i => 2 * (i + 1))

theorem greatest_prime_factor_f_24 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ f 24 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ f 24 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_f_24_l2806_280676


namespace NUMINAMATH_CALUDE_x_cubed_coefficient_equation_l2806_280686

theorem x_cubed_coefficient_equation (a : ℝ) : 
  (∃ k : ℝ, k = 56 ∧ k = 6 * a^2 - 15 * a + 20) ↔ (a = 6 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_x_cubed_coefficient_equation_l2806_280686


namespace NUMINAMATH_CALUDE_power_calculation_l2806_280638

theorem power_calculation : (8^5 / 8^3) * 4^6 = 262144 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2806_280638


namespace NUMINAMATH_CALUDE_zero_exponent_equals_one_l2806_280621

theorem zero_exponent_equals_one (r : ℚ) (h : r ≠ 0) : r ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_equals_one_l2806_280621


namespace NUMINAMATH_CALUDE_g_four_equals_thirteen_l2806_280615

-- Define the function g
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^6 + b * x^4 + c * x^2 + 7

-- State the theorem
theorem g_four_equals_thirteen 
  (a b c : ℝ) 
  (h : g a b c (-4) = 13) : 
  g a b c 4 = 13 := by
sorry

end NUMINAMATH_CALUDE_g_four_equals_thirteen_l2806_280615


namespace NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l2806_280652

/-- Represents the ratio of ingredients in a recipe -/
structure Ratio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original recipe ratio -/
def original_ratio : Ratio := ⟨7, 2, 1⟩

/-- The new recipe ratio -/
def new_ratio : Ratio :=
  let flour_water_doubled := original_ratio.flour / original_ratio.water * 2
  let flour_sugar_halved := original_ratio.flour / original_ratio.sugar / 2
  ⟨flour_water_doubled * original_ratio.water, original_ratio.water, flour_sugar_halved⟩

/-- The amount of water in the new recipe (in cups) -/
def new_water_amount : ℚ := 2

theorem sugar_amount_in_new_recipe :
  (new_water_amount * new_ratio.sugar / new_ratio.water) = 1 :=
sorry

end NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l2806_280652


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l2806_280628

theorem simplify_complex_fraction :
  1 / ((1 / (Real.sqrt 2 + 2)) + (3 / (2 * Real.sqrt 3 - 1))) =
  1 / (25 - 11 * Real.sqrt 2 + 6 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l2806_280628


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2806_280680

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if there exists a point C(0, √(2b)) such that the perpendicular bisector of AC
    (where A is the left vertex) passes through B (the right vertex),
    then the eccentricity of the hyperbola is √10/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ x^2 / a^2 - y^2 / b^2
  let A : ℝ × ℝ := (-a, 0)
  let B : ℝ × ℝ := (a, 0)
  let C : ℝ × ℝ := (0, Real.sqrt (2 * b^2))
  f B = 1 ∧ f A = 1 ∧ f C = -1 ∧
  (∃ M : ℝ × ℝ, M.1 = (A.1 + C.1) / 2 ∧ M.2 = (A.2 + C.2) / 2 ∧
    (B.2 - M.2) * (C.1 - A.1) = (B.1 - M.1) * (C.2 - A.2)) →
  Real.sqrt (a^2 + b^2) / a = Real.sqrt 10 / 2 := by
sorry


end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2806_280680


namespace NUMINAMATH_CALUDE_earliest_retirement_is_2009_l2806_280618

/-- Rule of 70 provision: An employee can retire when age + years of employment ≥ 70 -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year the employee was hired -/
def hire_year : ℕ := 1990

/-- The employee's age when hired -/
def hire_age : ℕ := 32

/-- The earliest retirement year satisfies the rule of 70 -/
def earliest_retirement_year (year : ℕ) : Prop :=
  rule_of_70 (hire_age + (year - hire_year)) (year - hire_year) ∧
  ∀ y < year, ¬rule_of_70 (hire_age + (y - hire_year)) (y - hire_year)

/-- Theorem: The earliest retirement year for the employee is 2009 -/
theorem earliest_retirement_is_2009 : earliest_retirement_year 2009 := by
  sorry

end NUMINAMATH_CALUDE_earliest_retirement_is_2009_l2806_280618


namespace NUMINAMATH_CALUDE_negative_ten_meters_westward_l2806_280674

-- Define the direction as an enumeration
inductive Direction
  | East
  | West

-- Define a function to convert a signed distance to a direction and magnitude
def interpretDistance (d : ℤ) : Direction × ℕ :=
  if d ≥ 0 then (Direction.East, d.natAbs) else (Direction.West, d.natAbs)

-- State the theorem
theorem negative_ten_meters_westward :
  interpretDistance (-10) = (Direction.West, 10) := by
  sorry

end NUMINAMATH_CALUDE_negative_ten_meters_westward_l2806_280674


namespace NUMINAMATH_CALUDE_path_order_paths_through_A_paths_through_B_total_paths_correct_l2806_280642

-- Define the grid points
inductive GridPoint
| X | Y | A | B | C | D | E | F | G

-- Define a function to count paths through a point
def pathsThrough (p : GridPoint) : ℕ := sorry

-- Total number of paths from X to Y
def totalPaths : ℕ := 924

-- Theorem stating the order of points based on number of paths
theorem path_order :
  pathsThrough GridPoint.A > pathsThrough GridPoint.F ∧
  pathsThrough GridPoint.F > pathsThrough GridPoint.C ∧
  pathsThrough GridPoint.C > pathsThrough GridPoint.G ∧
  pathsThrough GridPoint.G > pathsThrough GridPoint.E ∧
  pathsThrough GridPoint.E > pathsThrough GridPoint.D ∧
  pathsThrough GridPoint.D > pathsThrough GridPoint.B :=
by sorry

-- Theorem stating that the sum of paths through A and the point below X equals totalPaths
theorem paths_through_A :
  pathsThrough GridPoint.A = totalPaths / 2 :=
by sorry

-- Theorem stating that there's only one path through B
theorem paths_through_B :
  pathsThrough GridPoint.B = 1 :=
by sorry

-- Theorem stating that the total number of paths is correct
theorem total_paths_correct :
  (pathsThrough GridPoint.A) * 2 = totalPaths :=
by sorry

end NUMINAMATH_CALUDE_path_order_paths_through_A_paths_through_B_total_paths_correct_l2806_280642


namespace NUMINAMATH_CALUDE_victory_circle_count_l2806_280688

/-- Represents the different types of medals -/
inductive Medal
  | Gold
  | Silver
  | Bronze
  | Titanium
  | Copper

/-- Represents a runner in the race -/
structure Runner :=
  (position : Nat)
  (medal : Option Medal)

/-- Represents a victory circle configuration -/
def VictoryCircle := List Runner

/-- The number of runners in the race -/
def num_runners : Nat := 8

/-- The maximum number of medals that can be awarded -/
def max_medals : Nat := 5

/-- The minimum number of medals that can be awarded -/
def min_medals : Nat := 3

/-- Generates all possible victory circles for the given scenarios -/
def generate_victory_circles : List VictoryCircle := sorry

/-- Counts the number of unique victory circles -/
def count_victory_circles (circles : List VictoryCircle) : Nat := sorry

/-- Main theorem: The number of different victory circles is 28 -/
theorem victory_circle_count :
  count_victory_circles generate_victory_circles = 28 := by sorry

end NUMINAMATH_CALUDE_victory_circle_count_l2806_280688


namespace NUMINAMATH_CALUDE_irrational_between_neg_three_and_neg_two_l2806_280671

theorem irrational_between_neg_three_and_neg_two :
  ∃ x : ℝ, Irrational x ∧ -3 < x ∧ x < -2 := by sorry

end NUMINAMATH_CALUDE_irrational_between_neg_three_and_neg_two_l2806_280671


namespace NUMINAMATH_CALUDE_river_joe_pricing_l2806_280627

/-- River Joe's Seafood Diner pricing problem -/
theorem river_joe_pricing
  (total_orders : ℕ)
  (total_revenue : ℚ)
  (catfish_price : ℚ)
  (popcorn_shrimp_orders : ℕ)
  (h1 : total_orders = 26)
  (h2 : total_revenue = 133.5)
  (h3 : catfish_price = 6)
  (h4 : popcorn_shrimp_orders = 9) :
  ∃ (popcorn_shrimp_price : ℚ),
    popcorn_shrimp_price = 3.5 ∧
    total_revenue = (total_orders - popcorn_shrimp_orders) * catfish_price +
                    popcorn_shrimp_orders * popcorn_shrimp_price :=
by sorry

end NUMINAMATH_CALUDE_river_joe_pricing_l2806_280627


namespace NUMINAMATH_CALUDE_fraction_value_l2806_280608

theorem fraction_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : y = x / (x + 1)) :
  (x - y + 4 * x * y) / (x * y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2806_280608


namespace NUMINAMATH_CALUDE_pencil_packing_problem_l2806_280645

theorem pencil_packing_problem :
  ∃ a : ℕ, 200 ≤ a ∧ a ≤ 300 ∧ 
    a % 10 = 7 ∧ 
    a % 12 = 9 ∧
    (a = 237 ∨ a = 297) := by
  sorry

end NUMINAMATH_CALUDE_pencil_packing_problem_l2806_280645


namespace NUMINAMATH_CALUDE_parabola_vertex_l2806_280697

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 - 4*y + 3*x + 7 = 0

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the vertex of the parabola is (-1, 2) -/
theorem parabola_vertex :
  ∀ (x y : ℝ), parabola_equation x y → 
  ∃! (vx vy : ℝ), vx = vertex.1 ∧ vy = vertex.2 ∧
  (∀ (x' y' : ℝ), parabola_equation x' y' → (x' - vx)^2 + (y' - vy)^2 ≤ (x - vx)^2 + (y - vy)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2806_280697


namespace NUMINAMATH_CALUDE_base_five_product_l2806_280693

/-- Converts a base 5 number to decimal --/
def baseToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base 5 --/
def decimalToBase (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

theorem base_five_product :
  let a := [2, 3, 1] -- represents 132₅ in reverse order
  let b := [2, 1]    -- represents 12₅ in reverse order
  let product := [4, 3, 1, 2] -- represents 2134₅ in reverse order
  (baseToDecimal a) * (baseToDecimal b) = baseToDecimal product ∧
  decimalToBase ((baseToDecimal a) * (baseToDecimal b)) = product.reverse :=
sorry

end NUMINAMATH_CALUDE_base_five_product_l2806_280693


namespace NUMINAMATH_CALUDE_frank_peanuts_theorem_l2806_280602

def frank_peanuts (one_dollar_bills five_dollar_bills ten_dollar_bills twenty_dollar_bills : ℕ)
  (peanut_cost_per_pound : ℚ) (change : ℚ) (days_in_week : ℕ) : Prop :=
  let initial_money : ℚ := one_dollar_bills + 5 * five_dollar_bills + 10 * ten_dollar_bills + 20 * twenty_dollar_bills
  let spent_money : ℚ := initial_money - change
  let pounds_bought : ℚ := spent_money / peanut_cost_per_pound
  pounds_bought / days_in_week = 3

theorem frank_peanuts_theorem :
  frank_peanuts 7 4 2 1 3 4 7 :=
by
  sorry

end NUMINAMATH_CALUDE_frank_peanuts_theorem_l2806_280602


namespace NUMINAMATH_CALUDE_not_right_triangle_only_234_l2806_280668

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem not_right_triangle_only_234 :
  (is_right_triangle 1 1 (Real.sqrt 2)) ∧
  ¬(is_right_triangle 2 3 4) ∧
  (is_right_triangle 1 (Real.sqrt 3) 2) ∧
  (is_right_triangle 3 4 (Real.sqrt 7)) := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_only_234_l2806_280668


namespace NUMINAMATH_CALUDE_masks_donated_to_museum_l2806_280633

/-- Given that Alicia initially had 90 sets of masks and was left with 39 sets after donating to a museum,
    prove that she gave 51 sets to the museum. -/
theorem masks_donated_to_museum (initial_sets : ℕ) (remaining_sets : ℕ) 
    (h1 : initial_sets = 90) 
    (h2 : remaining_sets = 39) : 
  initial_sets - remaining_sets = 51 := by
  sorry

end NUMINAMATH_CALUDE_masks_donated_to_museum_l2806_280633


namespace NUMINAMATH_CALUDE_hotel_bubble_bath_amount_l2806_280660

/-- Calculates the total amount of bubble bath needed for a hotel --/
def total_bubble_bath_needed (luxury_suites rooms_for_couples single_rooms family_rooms : ℕ)
  (luxury_capacity couple_capacity single_capacity family_capacity : ℕ)
  (adult_bath_ml child_bath_ml : ℕ) : ℕ :=
  let total_guests := 
    luxury_suites * luxury_capacity + 
    rooms_for_couples * couple_capacity + 
    single_rooms * single_capacity + 
    family_rooms * family_capacity
  let adults := (2 * total_guests) / 3
  let children := total_guests - adults
  adults * adult_bath_ml + children * child_bath_ml

/-- The amount of bubble bath needed for the given hotel configuration --/
theorem hotel_bubble_bath_amount : 
  total_bubble_bath_needed 6 12 15 4 5 2 1 7 20 15 = 1760 := by
  sorry

end NUMINAMATH_CALUDE_hotel_bubble_bath_amount_l2806_280660


namespace NUMINAMATH_CALUDE_stone_slab_floor_area_l2806_280654

/-- Calculates the total floor area covered by square stone slabs -/
theorem stone_slab_floor_area 
  (num_slabs : ℕ) 
  (slab_length : ℝ) 
  (h_num_slabs : num_slabs = 30)
  (h_slab_length : slab_length = 140) : 
  (num_slabs * slab_length^2) / 10000 = 58.8 := by
  sorry

end NUMINAMATH_CALUDE_stone_slab_floor_area_l2806_280654


namespace NUMINAMATH_CALUDE_money_ratio_proof_l2806_280691

theorem money_ratio_proof (natasha_money carla_money cosima_money : ℚ) :
  natasha_money = 3 * carla_money →
  carla_money = cosima_money →
  natasha_money = 60 →
  (7 / 5) * (natasha_money + carla_money + cosima_money) - (natasha_money + carla_money + cosima_money) = 36 →
  carla_money / cosima_money = 1 := by
  sorry

end NUMINAMATH_CALUDE_money_ratio_proof_l2806_280691


namespace NUMINAMATH_CALUDE_simplify_expression_l2806_280609

theorem simplify_expression (p : ℝ) : 
  ((7 * p - 3) - 3 * p * 2) * 2 + (5 - 2 / 2) * (8 * p - 12) = 34 * p - 54 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2806_280609


namespace NUMINAMATH_CALUDE_inequality_proof_l2806_280684

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) : a^3 * b^2 < a^2 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2806_280684


namespace NUMINAMATH_CALUDE_relationship_x_y_l2806_280673

theorem relationship_x_y (x y : ℝ) 
  (h1 : 3 * x - 2 * y > 4 * x + 1) 
  (h2 : 2 * x + 3 * y < 5 * y - 2) : 
  x < 1 - y := by
sorry

end NUMINAMATH_CALUDE_relationship_x_y_l2806_280673


namespace NUMINAMATH_CALUDE_son_age_proof_l2806_280603

theorem son_age_proof (son_age father_age : ℕ) : 
  father_age = son_age + 22 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 20 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l2806_280603


namespace NUMINAMATH_CALUDE_shortest_distance_l2806_280614

theorem shortest_distance (a b : ℝ) (ha : a = 8) (hb : b = 6) :
  Real.sqrt (a ^ 2 + b ^ 2) = 10 := by
sorry

end NUMINAMATH_CALUDE_shortest_distance_l2806_280614


namespace NUMINAMATH_CALUDE_point_comparison_l2806_280604

/-- Given points A(-3,m) and B(2,n) lie on the line y = -2x + 1, prove that m > n -/
theorem point_comparison (m n : ℝ) : 
  ((-3 : ℝ), m) ∈ {p : ℝ × ℝ | p.2 = -2 * p.1 + 1} → 
  ((2 : ℝ), n) ∈ {p : ℝ × ℝ | p.2 = -2 * p.1 + 1} → 
  m > n := by
  sorry

end NUMINAMATH_CALUDE_point_comparison_l2806_280604


namespace NUMINAMATH_CALUDE_composition_of_linear_functions_l2806_280661

theorem composition_of_linear_functions 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h_f : ∀ x, f x = a * x + b) 
  (h_g : ∀ x, g x = 3 * x - 7) 
  (h_comp : ∀ x, g (f x) = 4 * x + 5) : 
  a + b = 16/3 := by
sorry

end NUMINAMATH_CALUDE_composition_of_linear_functions_l2806_280661


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l2806_280634

/-- Given a square with side length 4 cm, and an infinite series of squares where each subsequent
    square is formed by joining the midpoints of the sides of the previous square,
    the sum of the areas of all squares is 32 cm². -/
theorem sum_of_square_areas (first_square_side : ℝ) (h : first_square_side = 4) :
  let area_sequence : ℕ → ℝ := λ n => first_square_side^2 / 2^n
  ∑' n, area_sequence n = 32 :=
sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l2806_280634


namespace NUMINAMATH_CALUDE_no_maximum_value_l2806_280641

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def symmetric_about_point (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (2*a - x) = 2*b - f x

theorem no_maximum_value (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_sym : symmetric_about_point f 1 1) : 
  ¬ ∃ M, ∀ x, f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_no_maximum_value_l2806_280641


namespace NUMINAMATH_CALUDE_opposite_of_three_l2806_280653

theorem opposite_of_three : -(3 : ℝ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l2806_280653


namespace NUMINAMATH_CALUDE_network_engineers_from_university_a_l2806_280607

theorem network_engineers_from_university_a 
  (total_original : ℕ) 
  (new_hires : ℕ) 
  (fraction_from_a : ℚ) :
  total_original = 20 →
  new_hires = 8 →
  fraction_from_a = 3/4 →
  (fraction_from_a * (total_original + new_hires : ℚ) - new_hires) / total_original = 13/20 :=
by sorry

end NUMINAMATH_CALUDE_network_engineers_from_university_a_l2806_280607


namespace NUMINAMATH_CALUDE_nancy_boots_count_l2806_280624

theorem nancy_boots_count :
  ∀ (B : ℕ),
  (∃ (S H : ℕ),
    S = B + 9 ∧
    H = 3 * (S + B) ∧
    2 * B + 2 * S + 2 * H = 168) →
  B = 6 := by
sorry

end NUMINAMATH_CALUDE_nancy_boots_count_l2806_280624


namespace NUMINAMATH_CALUDE_negation_of_at_most_one_obtuse_l2806_280681

/-- Represents a triangle -/
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  all_positive : ∀ i, angles i > 0

/-- An angle is obtuse if it's greater than 90 degrees -/
def is_obtuse (angle : ℝ) : Prop := angle > 90

/-- At most one interior angle is obtuse -/
def at_most_one_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) → ¬is_obtuse (t.angles 1) ∧ ¬is_obtuse (t.angles 2)) ∧
  (is_obtuse (t.angles 1) → ¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 2)) ∧
  (is_obtuse (t.angles 2) → ¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 1))

/-- At least two interior angles are obtuse -/
def at_least_two_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 1)) ∨
  (is_obtuse (t.angles 1) ∧ is_obtuse (t.angles 2)) ∨
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 2))

/-- The main theorem: the negation of "at most one obtuse" is "at least two obtuse" -/
theorem negation_of_at_most_one_obtuse (t : Triangle) :
  ¬(at_most_one_obtuse t) ↔ at_least_two_obtuse t :=
sorry

end NUMINAMATH_CALUDE_negation_of_at_most_one_obtuse_l2806_280681


namespace NUMINAMATH_CALUDE_bus_trip_speed_l2806_280613

theorem bus_trip_speed (distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) 
  (h1 : distance = 280)
  (h2 : speed_increase = 5)
  (h3 : time_decrease = 1)
  (h4 : distance / speed - time_decrease = distance / (speed + speed_increase)) :
  speed = 35 := by
  sorry

end NUMINAMATH_CALUDE_bus_trip_speed_l2806_280613


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l2806_280619

theorem quadratic_equation_root (b : ℝ) :
  (∃ x : ℝ, 2 * x^2 + b * x - 119 = 0) ∧ (2 * 7^2 + b * 7 - 119 = 0) →
  b = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l2806_280619


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l2806_280620

def calculate_money_left (initial_amount : ℝ) (candy_bars : ℕ) (chips : ℕ) (soft_drinks : ℕ)
  (candy_bar_price : ℝ) (chips_price : ℝ) (soft_drink_price : ℝ)
  (candy_discount : ℝ) (chips_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  let candy_cost := candy_bars * candy_bar_price
  let chips_cost := chips * chips_price
  let soft_drinks_cost := soft_drinks * soft_drink_price
  let total_before_discounts := candy_cost + chips_cost + soft_drinks_cost
  let candy_discount_amount := candy_cost * candy_discount
  let chips_discount_amount := chips_cost * chips_discount
  let total_after_discounts := total_before_discounts - candy_discount_amount - chips_discount_amount
  let tax_amount := total_after_discounts * sales_tax
  let final_cost := total_after_discounts + tax_amount
  initial_amount - final_cost

theorem money_left_after_purchase :
  calculate_money_left 200 25 10 15 3 2.5 1.75 0.1 0.05 0.06 = 75.45 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l2806_280620


namespace NUMINAMATH_CALUDE_popped_kernel_probability_l2806_280631

theorem popped_kernel_probability (p_white p_yellow p_red : ℝ)
  (pop_white pop_yellow pop_red : ℝ) :
  p_white = 1/2 →
  p_yellow = 1/3 →
  p_red = 1/6 →
  pop_white = 1/2 →
  pop_yellow = 2/3 →
  pop_red = 1/3 →
  (p_white * pop_white) / (p_white * pop_white + p_yellow * pop_yellow + p_red * pop_red) = 9/19 := by
sorry

end NUMINAMATH_CALUDE_popped_kernel_probability_l2806_280631


namespace NUMINAMATH_CALUDE_impossibleArrangement_l2806_280672

/-- Represents a person at the table -/
structure Person :=
  (index : Fin 40)

/-- Represents the circular table with 40 people -/
def Table := Fin 40 → Person

/-- Calculates the number of people between two given people -/
def distance (table : Table) (p1 p2 : Person) : Nat :=
  sorry

/-- Determines if two people have a mutual acquaintance -/
def hasCommonAcquaintance (table : Table) (p1 p2 : Person) : Prop :=
  sorry

/-- The main theorem stating the impossibility of the arrangement -/
theorem impossibleArrangement :
  ¬ ∃ (table : Table),
    (∀ (p1 p2 : Person),
      hasCommonAcquaintance table p1 p2 ↔ Even (distance table p1 p2)) :=
  sorry

end NUMINAMATH_CALUDE_impossibleArrangement_l2806_280672


namespace NUMINAMATH_CALUDE_y_equal_at_one_y_diff_five_at_two_l2806_280679

-- Define the functions y₁ and y₂
def y₁ (x : ℝ) : ℝ := -2 * x + 3
def y₂ (x : ℝ) : ℝ := 3 * x - 2

-- Theorem 1: y₁ = y₂ when x = 1
theorem y_equal_at_one : y₁ 1 = y₂ 1 := by sorry

-- Theorem 2: y₁ + 5 = y₂ when x = 2
theorem y_diff_five_at_two : y₁ 2 + 5 = y₂ 2 := by sorry

end NUMINAMATH_CALUDE_y_equal_at_one_y_diff_five_at_two_l2806_280679


namespace NUMINAMATH_CALUDE_pokemon_cards_distribution_l2806_280694

theorem pokemon_cards_distribution (total_cards : ℕ) (num_friends : ℕ) 
  (h1 : total_cards = 432)
  (h2 : num_friends = 4)
  (h3 : total_cards % num_friends = 0)
  (h4 : (total_cards / num_friends) % 12 = 0) :
  (total_cards / num_friends) / 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_distribution_l2806_280694


namespace NUMINAMATH_CALUDE_basketball_league_games_l2806_280648

/-- The number of games played in a basketball league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem: In a league with 10 teams, where each team plays 4 games with each other team, 
    the total number of games is 180 -/
theorem basketball_league_games : total_games 10 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_basketball_league_games_l2806_280648


namespace NUMINAMATH_CALUDE_mother_triple_age_l2806_280665

/-- Represents the age difference between Serena and her mother -/
def age_difference : ℕ := 30

/-- Represents Serena's current age -/
def serena_age : ℕ := 9

/-- Represents the number of years until Serena's mother is three times as old as Serena -/
def years_until_triple : ℕ := 6

/-- Theorem stating that after 'years_until_triple' years, Serena's mother will be three times as old as Serena -/
theorem mother_triple_age :
  serena_age + years_until_triple = (serena_age + age_difference + years_until_triple) / 3 :=
sorry

end NUMINAMATH_CALUDE_mother_triple_age_l2806_280665


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2806_280687

theorem sum_with_radical_conjugate :
  let x : ℝ := 5 - Real.sqrt 500
  let y : ℝ := 5 + Real.sqrt 500
  x + y = 10 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2806_280687


namespace NUMINAMATH_CALUDE_strawberry_distribution_l2806_280696

theorem strawberry_distribution (initial : ℕ) (additional : ℕ) (boxes : ℕ) 
  (h1 : initial = 42)
  (h2 : additional = 78)
  (h3 : boxes = 6)
  (h4 : boxes ≠ 0) :
  (initial + additional) / boxes = 20 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_distribution_l2806_280696


namespace NUMINAMATH_CALUDE_not_nth_power_of_sum_of_powers_l2806_280650

theorem not_nth_power_of_sum_of_powers (p n : ℕ) (hp : Nat.Prime p) (hn : n > 1) :
  ¬ ∃ m : ℕ, (2^p : ℕ) + (3^p : ℕ) = m^n :=
sorry

end NUMINAMATH_CALUDE_not_nth_power_of_sum_of_powers_l2806_280650


namespace NUMINAMATH_CALUDE_even_and_divisible_by_six_l2806_280695

theorem even_and_divisible_by_six (n : ℕ) : 
  (2 ∣ n * (n + 1)) ∧ (6 ∣ n * (n + 1) * (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_even_and_divisible_by_six_l2806_280695


namespace NUMINAMATH_CALUDE_complex_number_equality_l2806_280663

theorem complex_number_equality (b : ℝ) : 
  let z := (3 - b * Complex.I) / (2 + Complex.I)
  z.re = z.im → b = -9 := by sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2806_280663


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2806_280657

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (3*x - 1)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 4^7 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2806_280657


namespace NUMINAMATH_CALUDE_wall_clock_interval_l2806_280637

/-- Represents a wall clock that rings at regular intervals -/
structure WallClock where
  rings_per_day : ℕ
  first_ring : ℕ
  hours_in_day : ℕ

/-- Calculates the interval between rings for a given wall clock -/
def ring_interval (clock : WallClock) : ℚ :=
  clock.hours_in_day / clock.rings_per_day

/-- Theorem: If a clock rings 8 times in a 24-hour day, starting at 1 A.M., 
    then the interval between each ring is 3 hours -/
theorem wall_clock_interval (clock : WallClock) 
    (h1 : clock.rings_per_day = 8) 
    (h2 : clock.first_ring = 1) 
    (h3 : clock.hours_in_day = 24) : 
    ring_interval clock = 3 := by
  sorry

end NUMINAMATH_CALUDE_wall_clock_interval_l2806_280637
