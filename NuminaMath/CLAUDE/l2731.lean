import Mathlib

namespace NUMINAMATH_CALUDE_lines_skew_iff_b_neq_l2731_273157

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Definition of skew lines -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∀ t u : ℝ, l1.point + t • l1.direction ≠ l2.point + u • l2.direction

/-- The problem statement -/
theorem lines_skew_iff_b_neq (b : ℝ) :
  let l1 : Line3D := ⟨(1, 2, b), (2, 3, 4)⟩
  let l2 : Line3D := ⟨(3, 0, -1), (5, 3, 1)⟩
  are_skew l1 l2 ↔ b ≠ 11/3 := by
  sorry

end NUMINAMATH_CALUDE_lines_skew_iff_b_neq_l2731_273157


namespace NUMINAMATH_CALUDE_cyclist_heartbeats_l2731_273149

/-- Calculates the total number of heartbeats during a cycling race -/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proves that the total number of heartbeats is 24000 for the given conditions -/
theorem cyclist_heartbeats :
  let heart_rate : ℕ := 120  -- beats per minute
  let race_distance : ℕ := 50  -- miles
  let pace : ℕ := 4  -- minutes per mile
  total_heartbeats heart_rate race_distance pace = 24000 := by
sorry


end NUMINAMATH_CALUDE_cyclist_heartbeats_l2731_273149


namespace NUMINAMATH_CALUDE_line_intercept_sum_l2731_273166

theorem line_intercept_sum (d : ℝ) : 
  (∃ x y : ℝ, 3 * x + 5 * y + d = 0 ∧ x + y = 16) → d = -30 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l2731_273166


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2731_273145

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    (4 * π * r^2 = 256 * π) →
    ((4 / 3) * π * r^3 = (2048 / 3) * π) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2731_273145


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l2731_273144

/-- Calculates the perimeter of a T-shaped figure formed by two rectangles -/
def t_perimeter (rect1_width rect1_height rect2_width rect2_height overlap : ℕ) : ℕ :=
  2 * (rect1_width + rect1_height) + 2 * (rect2_width + rect2_height) - 2 * overlap

/-- The perimeter of the T-shaped figure is 26 inches -/
theorem t_shape_perimeter : t_perimeter 3 5 2 5 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l2731_273144


namespace NUMINAMATH_CALUDE_positive_correlation_implies_positive_slope_l2731_273114

/-- Represents a simple linear regression model --/
structure LinearRegression where
  b : ℝ  -- slope
  a : ℝ  -- y-intercept
  r : ℝ  -- correlation coefficient

/-- Theorem stating that a positive correlation coefficient implies a positive slope --/
theorem positive_correlation_implies_positive_slope (model : LinearRegression) :
  model.r > 0 → model.b > 0 := by
  sorry


end NUMINAMATH_CALUDE_positive_correlation_implies_positive_slope_l2731_273114


namespace NUMINAMATH_CALUDE_exactly_two_rotational_homotheties_l2731_273118

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A rotational homothety with 90° rotation --/
structure RotationalHomothety where
  center : ℝ × ℝ
  scale : ℝ

/-- The number of rotational homotheties with 90° rotation that map one circle to another --/
def num_rotational_homotheties (S₁ S₂ : Circle) : ℕ :=
  sorry

/-- Two circles are non-concentric if their centers are different --/
def non_concentric (S₁ S₂ : Circle) : Prop :=
  S₁.center ≠ S₂.center

theorem exactly_two_rotational_homotheties (S₁ S₂ : Circle) 
  (h : non_concentric S₁ S₂) : num_rotational_homotheties S₁ S₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_rotational_homotheties_l2731_273118


namespace NUMINAMATH_CALUDE_relay_race_assignments_l2731_273163

theorem relay_race_assignments (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) :
  (n.factorial / (n - k).factorial : ℕ) = 32760 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_assignments_l2731_273163


namespace NUMINAMATH_CALUDE_parallel_lines_coincident_lines_perpendicular_lines_l2731_273168

-- Define the lines l₁ and l₂
def l₁ (m : ℚ) : ℚ → ℚ → Prop := λ x y => (m + 3) * x + 4 * y = 5 - 3 * m
def l₂ (m : ℚ) : ℚ → ℚ → Prop := λ x y => 2 * x + (m + 5) * y = 8

-- Define parallel lines
def parallel (l₁ l₂ : ℚ → ℚ → Prop) : Prop :=
  ∃ k : ℚ, ∀ x y, l₁ x y ↔ l₂ (k * x) (k * y)

-- Define coincident lines
def coincident (l₁ l₂ : ℚ → ℚ → Prop) : Prop :=
  ∀ x y, l₁ x y ↔ l₂ x y

-- Define perpendicular lines
def perpendicular (l₁ l₂ : ℚ → ℚ → Prop) : Prop :=
  ∃ k : ℚ, ∀ x y, l₁ x y → l₂ y (-x)

-- Theorem statements
theorem parallel_lines : parallel (l₁ (-7)) (l₂ (-7)) := sorry

theorem coincident_lines : coincident (l₁ (-1)) (l₂ (-1)) := sorry

theorem perpendicular_lines : perpendicular (l₁ (-13/3)) (l₂ (-13/3)) := sorry

end NUMINAMATH_CALUDE_parallel_lines_coincident_lines_perpendicular_lines_l2731_273168


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2731_273138

theorem cubic_equation_solution (x : ℝ) (hx : x ≠ 0) :
  1 - 6 / x + 9 / x^2 - 2 / x^3 = 0 →
  (3 / x = 3 / 2) ∨ (3 / x = 3 / (2 + Real.sqrt 3)) ∨ (3 / x = 3 / (2 - Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2731_273138


namespace NUMINAMATH_CALUDE_base_5_to_octal_conversion_l2731_273183

def base_5_to_decimal (n : ℕ) : ℕ := n

def decimal_to_octal (n : ℕ) : ℕ := n

theorem base_5_to_octal_conversion :
  decimal_to_octal (base_5_to_decimal 1234) = 302 := by
  sorry

end NUMINAMATH_CALUDE_base_5_to_octal_conversion_l2731_273183


namespace NUMINAMATH_CALUDE_range_of_f_l2731_273148

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then |x| - 1 else Real.sin x ^ 2

theorem range_of_f :
  Set.range f = Set.Ioi (-1) := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2731_273148


namespace NUMINAMATH_CALUDE_umbrella_arrangement_count_l2731_273155

/-- The number of ways to arrange 7 people in an umbrella shape -/
def umbrella_arrangements : ℕ := sorry

/-- The binomial coefficient (n choose k) -/
def choose (n k : ℕ) : ℕ := sorry

theorem umbrella_arrangement_count : umbrella_arrangements = 20 := by
  sorry

end NUMINAMATH_CALUDE_umbrella_arrangement_count_l2731_273155


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2731_273185

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 - x| < 1} = Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2731_273185


namespace NUMINAMATH_CALUDE_gem_selection_count_is_22_l2731_273180

/-- The number of ways to choose gems under given constraints -/
def gem_selection_count : ℕ :=
  let red_gems : ℕ := 9
  let blue_gems : ℕ := 5
  let green_gems : ℕ := 6
  let total_to_choose : ℕ := 10
  let min_red : ℕ := 2
  let min_blue : ℕ := 2
  let max_green : ℕ := 3
  
  (Finset.range (max_green + 1)).sum (λ g =>
    (Finset.range (red_gems + 1)).sum (λ r =>
      if r ≥ min_red ∧ 
         total_to_choose - r - g ≥ min_blue ∧ 
         total_to_choose - r - g ≤ blue_gems
      then 1
      else 0
    )
  )

/-- Theorem stating that the number of ways to choose gems is 22 -/
theorem gem_selection_count_is_22 : gem_selection_count = 22 := by
  sorry

end NUMINAMATH_CALUDE_gem_selection_count_is_22_l2731_273180


namespace NUMINAMATH_CALUDE_identity_not_T_function_exponential_T_function_cosine_T_function_iff_l2731_273129

-- Definition of a "T function"
def is_T_function (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = T * f x

-- Statement 1
theorem identity_not_T_function :
  ¬ is_T_function (λ x : ℝ => x) := by sorry

-- Statement 2
theorem exponential_T_function (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  (∃ x : ℝ, a^x = x) → is_T_function (λ x : ℝ => a^x) := by sorry

-- Statement 3
theorem cosine_T_function_iff (m : ℝ) :
  is_T_function (λ x : ℝ => Real.cos (m * x)) ↔ ∃ k : ℤ, m = k * Real.pi := by sorry

end NUMINAMATH_CALUDE_identity_not_T_function_exponential_T_function_cosine_T_function_iff_l2731_273129


namespace NUMINAMATH_CALUDE_nested_expression_value_l2731_273101

/-- The nested expression that needs to be evaluated -/
def nestedExpression : ℕ := 2*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4)))))))))

/-- Theorem stating that the nested expression equals 699050 -/
theorem nested_expression_value : nestedExpression = 699050 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l2731_273101


namespace NUMINAMATH_CALUDE_complex_imaginary_part_l2731_273184

theorem complex_imaginary_part (z : ℂ) (h : (3 + 4*I)*z = 5) : 
  z.im = -4/5 := by sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_l2731_273184


namespace NUMINAMATH_CALUDE_compound_interest_problem_l2731_273143

/-- Given a principal amount where the simple interest for 3 years at 10% per annum is 900,
    prove that the compound interest for the same period and rate is 993. -/
theorem compound_interest_problem (P : ℝ) : 
  P * 0.10 * 3 = 900 → 
  P * (1 + 0.10)^3 - P = 993 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l2731_273143


namespace NUMINAMATH_CALUDE_complementary_implies_mutually_exclusive_exists_mutually_exclusive_not_complementary_l2731_273158

variable {Ω : Type} [MeasurableSpace Ω]
variable (A₁ A₂ : Set Ω)

-- Define mutually exclusive events
def mutually_exclusive (A₁ A₂ : Set Ω) : Prop := A₁ ∩ A₂ = ∅

-- Define complementary events
def complementary (A₁ A₂ : Set Ω) : Prop := A₁ ∪ A₂ = Ω ∧ A₁ ∩ A₂ = ∅

-- Theorem 1: Complementary events are mutually exclusive
theorem complementary_implies_mutually_exclusive :
  complementary A₁ A₂ → mutually_exclusive A₁ A₂ := by sorry

-- Theorem 2: Existence of mutually exclusive events that are not complementary
theorem exists_mutually_exclusive_not_complementary :
  ∃ A₁ A₂ : Set Ω, mutually_exclusive A₁ A₂ ∧ ¬complementary A₁ A₂ := by sorry

end NUMINAMATH_CALUDE_complementary_implies_mutually_exclusive_exists_mutually_exclusive_not_complementary_l2731_273158


namespace NUMINAMATH_CALUDE_sqrt_square_eq_x_l2731_273176

theorem sqrt_square_eq_x (x : ℝ) (h : x ≥ 0) : (Real.sqrt x)^2 = x := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_x_l2731_273176


namespace NUMINAMATH_CALUDE_copper_ion_beakers_l2731_273189

theorem copper_ion_beakers (total_beakers : ℕ) (drops_per_test : ℕ) (total_drops_used : ℕ) (negative_beakers : ℕ) : 
  total_beakers = 22 → 
  drops_per_test = 3 → 
  total_drops_used = 45 → 
  negative_beakers = 7 → 
  total_beakers - negative_beakers = 15 := by
sorry

end NUMINAMATH_CALUDE_copper_ion_beakers_l2731_273189


namespace NUMINAMATH_CALUDE_rectangle_to_square_l2731_273188

theorem rectangle_to_square (k : ℕ) (n : ℕ) : 
  k > 7 →
  k * (k - 7) = n^2 →
  n < k →
  n = 24 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l2731_273188


namespace NUMINAMATH_CALUDE_polynomial_minimum_value_l2731_273100

theorem polynomial_minimum_value : 
  (∀ a b : ℝ, a^2 + 2*b^2 + 2*a + 4*b + 2008 ≥ 2005) ∧ 
  (∃ a b : ℝ, a^2 + 2*b^2 + 2*a + 4*b + 2008 = 2005) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_minimum_value_l2731_273100


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l2731_273160

theorem sarahs_bowling_score (g s : ℕ) : 
  s = g + 50 → (s + g) / 2 = 95 → s = 120 := by
sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l2731_273160


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2731_273146

theorem solution_set_quadratic_inequality (a : ℝ) :
  {x : ℝ | x^2 - 2*a + a^2 - 1 < 0} = {x : ℝ | a - 1 < x ∧ x < a + 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2731_273146


namespace NUMINAMATH_CALUDE_exactly_one_double_digit_sum_two_l2731_273133

/-- Sum of digits function -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Predicate for two-digit numbers -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The main theorem -/
theorem exactly_one_double_digit_sum_two :
  ∃! x : ℕ, is_two_digit x ∧ digit_sum (digit_sum x) = 2 := by sorry

end NUMINAMATH_CALUDE_exactly_one_double_digit_sum_two_l2731_273133


namespace NUMINAMATH_CALUDE_count_four_digit_divisible_by_13_l2731_273108

theorem count_four_digit_divisible_by_13 : 
  (Finset.filter (fun n => n % 13 = 0) (Finset.range 9000)).card = 693 := by sorry

end NUMINAMATH_CALUDE_count_four_digit_divisible_by_13_l2731_273108


namespace NUMINAMATH_CALUDE_tommy_bike_ride_l2731_273199

theorem tommy_bike_ride (E : ℕ) : 
  (4 * (E + 2) : ℕ) * 4 = 80 → E = 3 := by sorry

end NUMINAMATH_CALUDE_tommy_bike_ride_l2731_273199


namespace NUMINAMATH_CALUDE_samuel_bought_two_dozen_l2731_273198

/-- The number of dozens of doughnuts Samuel bought -/
def samuel_dozens : ℕ := sorry

/-- The number of dozens of doughnuts Cathy bought -/
def cathy_dozens : ℕ := 3

/-- The total number of people sharing the doughnuts -/
def total_people : ℕ := 10

/-- The number of doughnuts each person received -/
def doughnuts_per_person : ℕ := 6

/-- Theorem stating that Samuel bought 2 dozen doughnuts -/
theorem samuel_bought_two_dozen : samuel_dozens = 2 := by
  sorry

end NUMINAMATH_CALUDE_samuel_bought_two_dozen_l2731_273198


namespace NUMINAMATH_CALUDE_trig_function_problem_l2731_273135

theorem trig_function_problem (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (2 * x)) :
  f (Real.sin (π / 12)) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_function_problem_l2731_273135


namespace NUMINAMATH_CALUDE_faulty_odometer_conversion_l2731_273106

/-- Represents an odometer that skips certain digits -/
structure FaultyOdometer where
  reading : Nat
  skipped_digits : List Nat

/-- Converts a faulty odometer reading to actual miles traveled -/
def actual_miles (odo : FaultyOdometer) : Nat :=
  sorry

/-- The theorem stating that a faulty odometer reading of 003006 
    (skipping 3 and 4) represents 1030 actual miles -/
theorem faulty_odometer_conversion :
  let odo : FaultyOdometer := { reading := 3006, skipped_digits := [3, 4] }
  actual_miles odo = 1030 := by
  sorry

end NUMINAMATH_CALUDE_faulty_odometer_conversion_l2731_273106


namespace NUMINAMATH_CALUDE_min_a6_geometric_sequence_l2731_273154

theorem min_a6_geometric_sequence (a : ℕ → ℕ) (q : ℚ) :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 6 → a n > 0) →
  (∀ n : ℕ, 1 ≤ n ∧ n < 6 → a (n + 1) = (a n : ℚ) * q) →
  1 < q ∧ q < 2 →
  243 ≤ a 6 :=
by sorry

end NUMINAMATH_CALUDE_min_a6_geometric_sequence_l2731_273154


namespace NUMINAMATH_CALUDE_mother_daughter_age_difference_l2731_273175

theorem mother_daughter_age_difference :
  ∀ (mother_age daughter_age : ℕ),
    mother_age = 55 →
    mother_age - 1 = 2 * (daughter_age - 1) →
    mother_age - daughter_age = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_mother_daughter_age_difference_l2731_273175


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2731_273161

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number we want to express in scientific notation -/
def original_number : ℕ := 12060000

/-- The scientific notation representation of the original number -/
def scientific_representation : ScientificNotation := {
  coefficient := 1.206
  exponent := 7
  is_valid := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : 
  (original_number : ℝ) = scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2731_273161


namespace NUMINAMATH_CALUDE_largest_radius_is_61_l2731_273126

/-- A circle containing specific points and the unit circle -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  contains_points : center.1^2 + 11^2 = radius^2
  contains_unit_circle : ∀ (x y : ℝ), x^2 + y^2 < 1 → 
    (x - center.1)^2 + (y - center.2)^2 < radius^2

/-- The largest possible radius of a SpecialCircle is 61 -/
theorem largest_radius_is_61 : 
  (∃ (c : SpecialCircle), true) → 
  (∀ (c : SpecialCircle), c.radius ≤ 61) ∧ 
  (∃ (c : SpecialCircle), c.radius = 61) :=
sorry

end NUMINAMATH_CALUDE_largest_radius_is_61_l2731_273126


namespace NUMINAMATH_CALUDE_nut_mixture_price_l2731_273112

/-- Calculates the selling price per pound of a nut mixture --/
theorem nut_mixture_price
  (cashew_price : ℝ)
  (brazil_price : ℝ)
  (total_weight : ℝ)
  (cashew_weight : ℝ)
  (h1 : cashew_price = 6.75)
  (h2 : brazil_price = 5.00)
  (h3 : total_weight = 50)
  (h4 : cashew_weight = 20)
  : (cashew_weight * cashew_price + (total_weight - cashew_weight) * brazil_price) / total_weight = 5.70 := by
  sorry

end NUMINAMATH_CALUDE_nut_mixture_price_l2731_273112


namespace NUMINAMATH_CALUDE_smallest_m_with_integer_price_l2731_273178

theorem smallest_m_with_integer_price : ∃ m : ℕ+, 
  (∀ k < m, ¬∃ x : ℕ, (107 : ℚ) * x = 100 * k) ∧
  (∃ x : ℕ, (107 : ℚ) * x = 100 * m) ∧
  m = 107 := by
sorry

end NUMINAMATH_CALUDE_smallest_m_with_integer_price_l2731_273178


namespace NUMINAMATH_CALUDE_temp_difference_changtai_beijing_l2731_273194

/-- The difference in temperature between two locations -/
def temperature_difference (temp1 : Int) (temp2 : Int) : Int :=
  temp1 - temp2

/-- The lowest temperature in Beijing (in °C) -/
def beijing_temp : Int := -6

/-- The lowest temperature in Changtai County (in °C) -/
def changtai_temp : Int := 15

/-- Theorem: The temperature difference between Changtai County and Beijing is 21°C -/
theorem temp_difference_changtai_beijing :
  temperature_difference changtai_temp beijing_temp = 21 := by
  sorry

end NUMINAMATH_CALUDE_temp_difference_changtai_beijing_l2731_273194


namespace NUMINAMATH_CALUDE_region_area_correct_l2731_273182

/-- Given a circle with radius 36, two chords of length 66 intersecting at a point 12 units from
    the center at a 45° angle, this function calculates the area of one region formed by the
    intersection of the chords. -/
def calculate_region_area (radius : ℝ) (chord_length : ℝ) (intersection_distance : ℝ) 
    (intersection_angle : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the calculated area is correct for the given conditions. -/
theorem region_area_correct (radius : ℝ) (chord_length : ℝ) (intersection_distance : ℝ) 
    (intersection_angle : ℝ) :
  radius = 36 ∧ 
  chord_length = 66 ∧ 
  intersection_distance = 12 ∧ 
  intersection_angle = 45 * π / 180 →
  calculate_region_area radius chord_length intersection_distance intersection_angle > 0 :=
by sorry

end NUMINAMATH_CALUDE_region_area_correct_l2731_273182


namespace NUMINAMATH_CALUDE_triangle_angle_B_l2731_273107

theorem triangle_angle_B (A B C : ℝ) (a b c : ℝ) : 
  A = π/4 → a = 6 → b = 3 * Real.sqrt 2 → 
  0 < A ∧ A < π → 0 < B ∧ B < π → 0 < C ∧ C < π →
  a * Real.sin B = b * Real.sin A → 
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  B = π/6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l2731_273107


namespace NUMINAMATH_CALUDE_divisibility_and_infinite_pairs_l2731_273186

theorem divisibility_and_infinite_pairs (c d : ℤ) :
  (∃ f : ℕ → ℤ × ℤ, Function.Injective f ∧
    ∀ n, (f n).1 ∣ (c * (f n).2 + d) ∧ (f n).2 ∣ (c * (f n).1 + d)) ↔
  c ∣ d :=
by sorry

end NUMINAMATH_CALUDE_divisibility_and_infinite_pairs_l2731_273186


namespace NUMINAMATH_CALUDE_power_difference_value_l2731_273187

theorem power_difference_value (a x y : ℝ) (ha : a > 0) (hx : a^x = 2) (hy : a^y = 3) :
  a^(x - y) = 2/3 := by sorry

end NUMINAMATH_CALUDE_power_difference_value_l2731_273187


namespace NUMINAMATH_CALUDE_another_max_occurrence_sequence_l2731_273110

/-- Represents a circular strip of zeros and ones -/
def CircularStrip := List Bool

/-- Counts the number of occurrences of a sequence in a circular strip -/
def count_occurrences (strip : CircularStrip) (seq : List Bool) : Nat :=
  sorry

/-- The sequence with the maximum number of occurrences -/
def max_seq (n : Nat) : List Bool :=
  [true, true] ++ List.replicate (n - 2) false

/-- The sequence with the minimum number of occurrences -/
def min_seq (n : Nat) : List Bool :=
  List.replicate (n - 2) false ++ [true, true]

theorem another_max_occurrence_sequence 
  (n : Nat) 
  (h_n : n > 5) 
  (strip : CircularStrip) 
  (h_strip : strip.length > 0) 
  (h_max : ∀ seq : List Bool, seq.length = n → 
    count_occurrences strip seq ≤ count_occurrences strip (max_seq n)) 
  (h_min : count_occurrences strip (min_seq n) < count_occurrences strip (max_seq n)) :
  ∃ seq : List Bool, 
    seq.length = n ∧ 
    seq ≠ max_seq n ∧ 
    count_occurrences strip seq = count_occurrences strip (max_seq n) :=
  sorry

end NUMINAMATH_CALUDE_another_max_occurrence_sequence_l2731_273110


namespace NUMINAMATH_CALUDE_max_students_for_given_supplies_l2731_273147

/-- The maximum number of students among whom pens and pencils can be distributed equally -/
def max_students (pens : ℕ) (pencils : ℕ) : ℕ :=
  Nat.gcd pens pencils

/-- Theorem stating that the GCD of 1048 and 828 is the maximum number of students -/
theorem max_students_for_given_supplies : 
  max_students 1048 828 = 4 := by sorry

end NUMINAMATH_CALUDE_max_students_for_given_supplies_l2731_273147


namespace NUMINAMATH_CALUDE_exists_divisibility_property_l2731_273191

theorem exists_divisibility_property (n : ℕ+) : ∃ (a b : ℕ+), n ∣ (4 * a^2 + 9 * b^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_divisibility_property_l2731_273191


namespace NUMINAMATH_CALUDE_unique_integer_solution_fourth_power_equation_l2731_273136

theorem unique_integer_solution_fourth_power_equation :
  ∀ x y : ℤ, x^4 + y^4 = 3*x^3*y → x = 0 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_fourth_power_equation_l2731_273136


namespace NUMINAMATH_CALUDE_product_zero_l2731_273169

theorem product_zero (r : ℂ) (h1 : r^4 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l2731_273169


namespace NUMINAMATH_CALUDE_set_operations_l2731_273131

-- Define the sets
def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def C_U_N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define N (this is what we need to prove)
def N : Set ℝ := {x | (-3 ≤ x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x ≤ 3)}

-- State the theorem
theorem set_operations :
  (N = {x : ℝ | (-3 ≤ x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x ≤ 3)}) ∧
  (M ∩ C_U_N = {x : ℝ | 0 < x ∧ x < 1}) ∧
  (M ∪ N = {x : ℝ | (-3 ≤ x ∧ x < 1) ∨ (2 ≤ x ∧ x ≤ 3)}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l2731_273131


namespace NUMINAMATH_CALUDE_jamie_balls_l2731_273172

theorem jamie_balls (R : ℕ) : 
  (R - 6) + 2 * R + 32 = 74 → R = 16 := by
sorry

end NUMINAMATH_CALUDE_jamie_balls_l2731_273172


namespace NUMINAMATH_CALUDE_pencil_eraser_notebook_cost_l2731_273125

theorem pencil_eraser_notebook_cost 
  (h1 : 20 * x + 3 * y + 2 * z = 32) 
  (h2 : 39 * x + 5 * y + 3 * z = 58) : 
  5 * x + 5 * y + 5 * z = 30 := by
  sorry

end NUMINAMATH_CALUDE_pencil_eraser_notebook_cost_l2731_273125


namespace NUMINAMATH_CALUDE_group_formation_count_l2731_273170

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of officers --/
def totalOfficers : ℕ := 10

/-- The total number of jawans --/
def totalJawans : ℕ := 15

/-- The number of officers in each group --/
def officersPerGroup : ℕ := 3

/-- The number of jawans in each group --/
def jawansPerGroup : ℕ := 5

/-- The number of ways to form groups --/
def numberOfGroups : ℕ := 
  totalOfficers * 
  (choose (totalOfficers - 1) (officersPerGroup - 1)) * 
  (choose totalJawans jawansPerGroup)

theorem group_formation_count : numberOfGroups = 1081080 := by
  sorry

end NUMINAMATH_CALUDE_group_formation_count_l2731_273170


namespace NUMINAMATH_CALUDE_problem_solution_l2731_273134

theorem problem_solution (x y : ℤ) : 
  y = 3 * x^2 ∧ 
  (2 * x : ℚ) / 5 = 1 / (1 - 2 / (3 + 1 / (4 - 5 / (6 - x)))) → 
  y = 147 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2731_273134


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l2731_273179

theorem quadratic_root_problem (m : ℝ) :
  (∃ x : ℝ, x^2 + m*x + 6 = 0 ∧ x = -2) →
  (∃ x : ℝ, x^2 + m*x + 6 = 0 ∧ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l2731_273179


namespace NUMINAMATH_CALUDE_emily_big_garden_seeds_l2731_273165

/-- The number of seeds Emily started with -/
def total_seeds : ℕ := 41

/-- The number of small gardens Emily has -/
def num_small_gardens : ℕ := 3

/-- The number of seeds Emily planted in each small garden -/
def seeds_per_small_garden : ℕ := 4

/-- The number of seeds Emily planted in the big garden -/
def seeds_in_big_garden : ℕ := total_seeds - (num_small_gardens * seeds_per_small_garden)

theorem emily_big_garden_seeds : seeds_in_big_garden = 29 := by
  sorry

end NUMINAMATH_CALUDE_emily_big_garden_seeds_l2731_273165


namespace NUMINAMATH_CALUDE_smallest_sum_M_N_l2731_273121

/-- Alice's transformation function -/
def aliceTransform (x : ℕ) : ℕ := 3 * x + 2

/-- Bob's transformation function -/
def bobTransform (x : ℕ) : ℕ := 2 * x + 27

/-- Alice's board after 4 moves -/
def aliceFourMoves (M : ℕ) : ℕ := aliceTransform (aliceTransform (aliceTransform (aliceTransform M)))

/-- Bob's board after 4 moves -/
def bobFourMoves (N : ℕ) : ℕ := bobTransform (bobTransform (bobTransform (bobTransform N)))

/-- The theorem stating the smallest sum of M and N -/
theorem smallest_sum_M_N : 
  ∃ (M N : ℕ), 
    M > 0 ∧ N > 0 ∧
    aliceFourMoves M = bobFourMoves N ∧
    (∀ (M' N' : ℕ), M' > 0 → N' > 0 → aliceFourMoves M' = bobFourMoves N' → M + N ≤ M' + N') ∧
    M + N = 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_M_N_l2731_273121


namespace NUMINAMATH_CALUDE_high_school_total_students_l2731_273173

/-- Represents a high school with three grades in its senior section. -/
structure HighSchool where
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ

/-- Represents a stratified sample from the high school. -/
structure Sample where
  total : ℕ
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ

/-- The total number of students in the high school section. -/
def total_students (hs : HighSchool) : ℕ :=
  hs.first_grade + hs.second_grade + hs.third_grade

/-- The theorem stating the total number of students in the high school section. -/
theorem high_school_total_students 
  (hs : HighSchool)
  (sample : Sample)
  (h1 : hs.first_grade = 400)
  (h2 : sample.total = 45)
  (h3 : sample.second_grade = 15)
  (h4 : sample.third_grade = 10)
  (h5 : sample.first_grade = sample.total - sample.second_grade - sample.third_grade)
  (h6 : sample.first_grade * hs.first_grade = sample.total * 20) :
  total_students hs = 900 := by
  sorry


end NUMINAMATH_CALUDE_high_school_total_students_l2731_273173


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l2731_273142

theorem complex_fraction_calculation : 
  let initial := 104 + 2 / 5
  let step1 := (initial / (3 / 8))
  let step2 := step1 / 2
  let step3 := step2 + (14 + 1 / 2)
  let step4 := step3 * (4 / 7)
  let final := step4 - (2 + 3 / 28)
  final = 86 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l2731_273142


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l2731_273177

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 4*x + 6

-- Define the closed interval [1, 5]
def I : Set ℝ := Set.Icc 1 5

-- Theorem statement
theorem max_min_values_of_f :
  ∃ (a b : ℝ), a ∈ I ∧ b ∈ I ∧
  (∀ x ∈ I, f x ≤ f a) ∧
  (∀ x ∈ I, f x ≥ f b) ∧
  f a = 11 ∧ f b = 2 :=
sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l2731_273177


namespace NUMINAMATH_CALUDE_divisibility_by_30_l2731_273140

theorem divisibility_by_30 (p : ℕ) (h1 : p.Prime) (h2 : p ≥ 7) : 30 ∣ p^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_30_l2731_273140


namespace NUMINAMATH_CALUDE_parallelogram_sum_l2731_273139

/-- A parallelogram with sides of lengths 5, 11, 3y+2, and 4x-1 -/
structure Parallelogram (x y : ℝ) :=
  (side1 : ℝ := 5)
  (side2 : ℝ := 11)
  (side3 : ℝ := 3 * y + 2)
  (side4 : ℝ := 4 * x - 1)

/-- Theorem: In a parallelogram with sides of lengths 5, 11, 3y+2, and 4x-1, x + y = 4 -/
theorem parallelogram_sum (x y : ℝ) (p : Parallelogram x y) : x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_sum_l2731_273139


namespace NUMINAMATH_CALUDE_base_number_inequality_l2731_273151

theorem base_number_inequality (x : ℝ) : 64^8 > x^22 ↔ x = 2^(24/11) := by
  sorry

end NUMINAMATH_CALUDE_base_number_inequality_l2731_273151


namespace NUMINAMATH_CALUDE_inequality_solution_l2731_273137

theorem inequality_solution (x : ℝ) : (x - 2) / (x + 5) ≥ 0 ↔ x < -5 ∨ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2731_273137


namespace NUMINAMATH_CALUDE_reciprocal_difference_decreases_l2731_273128

theorem reciprocal_difference_decreases (n : ℕ) : 
  (1 : ℚ) / n - (1 : ℚ) / (n + 1) = 1 / (n * (n + 1)) ∧
  ∀ m : ℕ, m > n → (1 : ℚ) / m - (1 : ℚ) / (m + 1) < (1 : ℚ) / n - (1 : ℚ) / (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_difference_decreases_l2731_273128


namespace NUMINAMATH_CALUDE_triangle_equation_l2731_273195

/-- A non-isosceles triangle with side lengths a, b, c opposite to angles A, B, C respectively,
    where A, B, C form an arithmetic sequence. -/
structure NonIsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  nonIsosceles : a ≠ b ∧ b ≠ c ∧ a ≠ c
  oppositeAngles : angleA.cos = (b^2 + c^2 - a^2) / (2*b*c) ∧
                   angleB.cos = (a^2 + c^2 - b^2) / (2*a*c) ∧
                   angleC.cos = (a^2 + b^2 - c^2) / (2*a*b)
  arithmeticSequence : ∃ (d : ℝ), angleB = angleA + d ∧ angleC = angleB + d

/-- The main theorem stating the equation holds for non-isosceles triangles with angles
    in arithmetic sequence. -/
theorem triangle_equation (t : NonIsoscelesTriangle) :
  1 / (t.a - t.b) + 1 / (t.c - t.b) = 3 / (t.a - t.b + t.c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_equation_l2731_273195


namespace NUMINAMATH_CALUDE_aunt_gemma_dog_food_l2731_273115

/-- The number of sacks of dog food Aunt Gemma bought -/
def num_sacks : ℕ := 2

/-- The number of dogs Aunt Gemma has -/
def num_dogs : ℕ := 4

/-- The number of times Aunt Gemma feeds her dogs per day -/
def feeds_per_day : ℕ := 2

/-- The amount of food each dog consumes per meal in grams -/
def food_per_meal : ℕ := 250

/-- The weight of each sack of dog food in kilograms -/
def sack_weight : ℕ := 50

/-- The number of days the dog food will last -/
def days_lasting : ℕ := 50

theorem aunt_gemma_dog_food :
  num_sacks = (num_dogs * feeds_per_day * food_per_meal * days_lasting) / (sack_weight * 1000) := by
  sorry

end NUMINAMATH_CALUDE_aunt_gemma_dog_food_l2731_273115


namespace NUMINAMATH_CALUDE_total_books_proof_l2731_273117

/-- The number of books in the 'crazy silly school' series -/
def total_books : ℕ := 19

/-- The number of books already read -/
def books_read : ℕ := 4

/-- The number of books yet to be read -/
def books_to_read : ℕ := 15

/-- Theorem stating that the total number of books is the sum of books read and books to be read -/
theorem total_books_proof : total_books = books_read + books_to_read := by
  sorry

end NUMINAMATH_CALUDE_total_books_proof_l2731_273117


namespace NUMINAMATH_CALUDE_rose_count_l2731_273102

theorem rose_count (total : ℕ) : 
  (300 ≤ total ∧ total ≤ 400) →
  (∃ x : ℕ, total = 21 * x + 13) →
  (∃ y : ℕ, total = 15 * y - 8) →
  total = 307 := by
sorry

end NUMINAMATH_CALUDE_rose_count_l2731_273102


namespace NUMINAMATH_CALUDE_first_native_is_liar_and_path_is_incorrect_l2731_273164

-- Define the types of natives
inductive NativeType
| Truthful
| Liar

-- Define a native
structure Native where
  type : NativeType

-- Define the claim about being a Liar
def claimToBeLiar (n : Native) : Prop :=
  match n.type with
  | NativeType.Truthful => False
  | NativeType.Liar => False

-- Define the first native's report about the second native's claim
def firstNativeReport (first : Native) (second : Native) : Prop :=
  claimToBeLiar second

-- Define the correctness of the path indication
def correctPathIndication (n : Native) : Prop :=
  match n.type with
  | NativeType.Truthful => True
  | NativeType.Liar => False

-- Theorem statement
theorem first_native_is_liar_and_path_is_incorrect 
  (first : Native) (second : Native) :
  firstNativeReport first second →
  first.type = NativeType.Liar ∧ ¬(correctPathIndication first) := by
  sorry

end NUMINAMATH_CALUDE_first_native_is_liar_and_path_is_incorrect_l2731_273164


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_equals_one_l2731_273159

-- Define the equation
def equation (a x : ℝ) : Prop :=
  3^(x^2 - 2*a*x + a^2) = a*x^2 - 2*a^2*x + a^3 + a^2 - 4*a + 4

-- Define the property of having exactly one solution
def has_exactly_one_solution (a : ℝ) : Prop :=
  ∃! x, equation a x

-- Theorem statement
theorem unique_solution_iff_a_equals_one :
  ∀ a : ℝ, has_exactly_one_solution a ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_equals_one_l2731_273159


namespace NUMINAMATH_CALUDE_barney_weight_difference_l2731_273130

/-- The weight difference between Barney and five regular dinosaurs -/
def weight_difference : ℕ → ℕ → ℕ → ℕ
  | regular_weight, total_weight, num_regular =>
    total_weight - num_regular * regular_weight

theorem barney_weight_difference :
  weight_difference 800 9500 5 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_barney_weight_difference_l2731_273130


namespace NUMINAMATH_CALUDE_middle_school_run_time_average_l2731_273120

/-- Represents the average number of minutes run per day by students in a specific grade -/
structure GradeRunTime where
  grade : Nat
  average_minutes : ℝ

/-- Represents the ratio of students between two grades -/
structure GradeRatio where
  higher_grade : Nat
  lower_grade : Nat
  ratio : Nat

/-- Calculates the average run time for all students given the run times for each grade and the ratios between grades -/
def calculate_average_run_time (run_times : List GradeRunTime) (ratios : List GradeRatio) : ℝ :=
  sorry

theorem middle_school_run_time_average :
  let sixth_grade := GradeRunTime.mk 6 20
  let seventh_grade := GradeRunTime.mk 7 18
  let eighth_grade := GradeRunTime.mk 8 16
  let ratio_sixth_seventh := GradeRatio.mk 6 7 3
  let ratio_seventh_eighth := GradeRatio.mk 7 8 3
  let run_times := [sixth_grade, seventh_grade, eighth_grade]
  let ratios := [ratio_sixth_seventh, ratio_seventh_eighth]
  calculate_average_run_time run_times ratios = 250 / 13 := by
    sorry

end NUMINAMATH_CALUDE_middle_school_run_time_average_l2731_273120


namespace NUMINAMATH_CALUDE_facebook_employee_bonus_l2731_273167

/-- Represents the Facebook employee bonus problem -/
theorem facebook_employee_bonus (
  total_employees : ℕ
  ) (annual_earnings : ℕ) (bonus_percentage : ℚ) (bonus_per_mother : ℕ) :
  total_employees = 3300 →
  annual_earnings = 5000000 →
  bonus_percentage = 1/4 →
  bonus_per_mother = 1250 →
  ∃ (non_mother_employees : ℕ),
    non_mother_employees = 1200 ∧
    non_mother_employees = 
      (2/3 : ℚ) * total_employees - 
      (bonus_percentage * annual_earnings) / bonus_per_mother :=
by sorry


end NUMINAMATH_CALUDE_facebook_employee_bonus_l2731_273167


namespace NUMINAMATH_CALUDE_project_duration_proof_l2731_273141

/-- The original duration of the project in months -/
def original_duration : ℝ := 30

/-- The reduction in project duration when efficiency is increased -/
def duration_reduction : ℝ := 6

/-- The factor by which efficiency is increased -/
def efficiency_increase : ℝ := 1.25

theorem project_duration_proof :
  (original_duration - duration_reduction) / original_duration = 1 / efficiency_increase :=
by sorry

#check project_duration_proof

end NUMINAMATH_CALUDE_project_duration_proof_l2731_273141


namespace NUMINAMATH_CALUDE_two_and_half_dozens_eq_30_l2731_273197

/-- The number of items in a dozen -/
def dozen : ℕ := 12

/-- The number of pens in two and one-half dozens -/
def two_and_half_dozens : ℕ := 2 * dozen + dozen / 2

/-- Theorem stating that two and one-half dozens of pens is equal to 30 pens -/
theorem two_and_half_dozens_eq_30 : two_and_half_dozens = 30 := by
  sorry

end NUMINAMATH_CALUDE_two_and_half_dozens_eq_30_l2731_273197


namespace NUMINAMATH_CALUDE_average_height_theorem_l2731_273119

def tree_heights (h₁ h₂ h₃ h₄ h₅ : ℝ) : Prop :=
  h₂ = 15 ∧
  (h₂ = h₁ + 5 ∨ h₂ = h₁ - 3) ∧
  (h₃ = h₂ + 5 ∨ h₃ = h₂ - 3) ∧
  (h₄ = h₃ + 5 ∨ h₄ = h₃ - 3) ∧
  (h₅ = h₄ + 5 ∨ h₅ = h₄ - 3)

theorem average_height_theorem (h₁ h₂ h₃ h₄ h₅ : ℝ) :
  tree_heights h₁ h₂ h₃ h₄ h₅ →
  ∃ (k : ℤ), (h₁ + h₂ + h₃ + h₄ + h₅) / 5 = k + 0.4 →
  (h₁ + h₂ + h₃ + h₄ + h₅) / 5 = 20.4 :=
by sorry

end NUMINAMATH_CALUDE_average_height_theorem_l2731_273119


namespace NUMINAMATH_CALUDE_sum_cos_fractions_24_pi_zero_l2731_273123

def simplest_proper_fractions_24 : List ℚ := [
  1/24, 5/24, 7/24, 11/24, 13/24, 17/24, 19/24, 23/24
]

theorem sum_cos_fractions_24_pi_zero : 
  (simplest_proper_fractions_24.map (fun x => Real.cos (x * Real.pi))).sum = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_cos_fractions_24_pi_zero_l2731_273123


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l2731_273162

/-- The axis of symmetry of a parabola y = (x - h)^2 + k is the line x = h -/
theorem parabola_axis_of_symmetry (h k : ℝ) :
  let f : ℝ → ℝ := λ x => (x - h)^2 + k
  ∀ x, f (h + x) = f (h - x) :=
by sorry

/-- The axis of symmetry of the parabola y = (x - 1)^2 + 3 is the line x = 1 -/
theorem specific_parabola_axis_of_symmetry :
  let f : ℝ → ℝ := λ x => (x - 1)^2 + 3
  ∀ x, f (1 + x) = f (1 - x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l2731_273162


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2731_273152

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 36*x^2 + 215*x - 470

-- State the theorem
theorem root_sum_reciprocal (a b c : ℝ) (D E F : ℝ) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  p a = 0 ∧ p b = 0 ∧ p c = 0 →
  (∀ t : ℝ, t ≠ a ∧ t ≠ b ∧ t ≠ c →
    1 / (t^3 - 36*t^2 + 215*t - 470) = D / (t - a) + E / (t - b) + F / (t - c)) →
  1 / D + 1 / E + 1 / F = 105 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2731_273152


namespace NUMINAMATH_CALUDE_smallest_value_complex_sum_l2731_273127

theorem smallest_value_complex_sum (a b c : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_omega_power : ω^4 = 1)
  (h_omega_neq_one : ω ≠ 1) :
  ∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (∀ (p q r : ℤ), p ≠ q ∧ q ≠ r ∧ p ≠ r → 
      Complex.abs (x + y*ω + z*ω^3) ≤ Complex.abs (p + q*ω + r*ω^3)) ∧
    Complex.abs (x + y*ω + z*ω^3) = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_complex_sum_l2731_273127


namespace NUMINAMATH_CALUDE_solve_inequality_system_simplify_expression_l2731_273116

-- Part 1: System of inequalities
theorem solve_inequality_system (x : ℝ) :
  (x + 2) / 5 < 1 ∧ 3 * x - 1 ≥ 2 * x ↔ 1 ≤ x ∧ x < 3 := by sorry

-- Part 2: Algebraic expression
theorem simplify_expression (m : ℝ) (hm : m ≠ 0) :
  (m - 1 / m) * (m^2 - m) / (m^2 - 2*m + 1) = m + 1 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_system_simplify_expression_l2731_273116


namespace NUMINAMATH_CALUDE_eight_last_to_appear_l2731_273113

def tribonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 4
  | n + 3 => tribonacci n + tribonacci (n + 1) + tribonacci (n + 2)

def lastDigit (n : ℕ) : ℕ := n % 10

def digitAppears (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ lastDigit (tribonacci k) = d

theorem eight_last_to_appear :
  ∃ N, ∀ n, n ≥ N → 
    (∀ d, d ≠ 8 → digitAppears d n) ∧
    ¬(digitAppears 8 n) ∧
    digitAppears 8 (n + 1) := by sorry

end NUMINAMATH_CALUDE_eight_last_to_appear_l2731_273113


namespace NUMINAMATH_CALUDE_smallest_a_l2731_273103

-- Define the polynomial
def P (a b x : ℤ) : ℤ := x^3 - a*x^2 + b*x - 1806

-- Define the property of having three positive integer roots
def has_three_positive_integer_roots (a b : ℤ) : Prop :=
  ∃ (x y z : ℤ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  P a b x = 0 ∧ P a b y = 0 ∧ P a b z = 0

-- State the theorem
theorem smallest_a :
  ∃ (a : ℤ), has_three_positive_integer_roots a (a*56 - 1806) ∧
  (∀ (a' : ℤ), has_three_positive_integer_roots a' (a'*56 - 1806) → a ≤ a') :=
sorry

end NUMINAMATH_CALUDE_smallest_a_l2731_273103


namespace NUMINAMATH_CALUDE_parabola_through_points_intersects_interval_l2731_273181

/-- Represents a parabola of the form y = ax² + c -/
structure Parabola where
  a : ℝ
  c : ℝ
  h_a_neg : a < 0

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ := p.a * x^2 + p.c

theorem parabola_through_points_intersects_interval
  (p : Parabola)
  (h_through_A : p.y_at 0 = 9)
  (h_through_P : p.y_at 2 = 8.1) :
  -9/49 < p.a ∧ p.a < -1/4 ∧
  ∃ x, 6 < x ∧ x < 7 ∧ p.y_at x = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_through_points_intersects_interval_l2731_273181


namespace NUMINAMATH_CALUDE_money_distribution_l2731_273105

theorem money_distribution (A B C : ℕ) : 
  A + B + C = 500 → A + C = 200 → B + C = 330 → C = 30 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l2731_273105


namespace NUMINAMATH_CALUDE_car_comparison_l2731_273192

-- Define the speeds and times for both cars
def speed_M : ℝ := 1  -- Arbitrary unit speed for Car M
def speed_N : ℝ := 3 * speed_M
def start_time_M : ℝ := 0
def start_time_N : ℝ := 2
def total_time : ℝ := 3  -- From the solution, but not directly given in the problem

-- Define the distance traveled by each car
def distance_M (t : ℝ) : ℝ := speed_M * t
def distance_N (t : ℝ) : ℝ := speed_N * (t - start_time_N)

-- Theorem statement
theorem car_comparison :
  ∃ (t : ℝ), t > start_time_N ∧
  distance_M t = distance_N t ∧
  speed_N = 3 * speed_M ∧
  start_time_N - start_time_M = 2 := by
  sorry

end NUMINAMATH_CALUDE_car_comparison_l2731_273192


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2731_273196

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 16) = 5 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2731_273196


namespace NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l2731_273156

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeroes in 500! is 124 -/
theorem trailing_zeroes_500_factorial :
  trailingZeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l2731_273156


namespace NUMINAMATH_CALUDE_tan_sum_simplification_l2731_273124

theorem tan_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / 
  Real.cos (40 * π / 180) = (Real.sqrt 3 + 1) / (Real.sqrt 3 * Real.cos (40 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_simplification_l2731_273124


namespace NUMINAMATH_CALUDE_largest_x_absolute_value_equation_l2731_273111

theorem largest_x_absolute_value_equation :
  ∃ (x : ℝ), x = 17 ∧ |2*x - 4| = 30 ∧ ∀ (y : ℝ), |2*y - 4| = 30 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_x_absolute_value_equation_l2731_273111


namespace NUMINAMATH_CALUDE_grid_coverage_possible_specific_case_101x101_l2731_273171

/-- Represents a square stamp with black cells -/
structure Stamp :=
  (size : ℕ)
  (black_cells : ℕ)

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Predicate to check if a grid can be covered by a stamp, leaving one corner uncovered -/
def can_cover (s : Stamp) (g : Grid) (num_stamps : ℕ) : Prop :=
  ∃ (N : ℕ), 
    g.size = 2*N + 1 ∧ 
    s.size = 2*N ∧ 
    s.black_cells = 4*N + 2 ∧ 
    num_stamps = 4*N

/-- Theorem stating that it's possible to cover a (2N+1) x (2N+1) grid with a 2N x 2N stamp -/
theorem grid_coverage_possible :
  ∀ (N : ℕ), N > 0 → 
    let s : Stamp := ⟨2*N, 4*N + 2⟩
    let g : Grid := ⟨2*N + 1⟩
    can_cover s g (4*N) :=
by
  sorry

/-- The specific case for the 101 x 101 grid with 102 black cells on the stamp -/
theorem specific_case_101x101 :
  let s : Stamp := ⟨100, 102⟩
  let g : Grid := ⟨101⟩
  can_cover s g 100 :=
by
  sorry

end NUMINAMATH_CALUDE_grid_coverage_possible_specific_case_101x101_l2731_273171


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2731_273174

theorem opposite_of_2023 : 
  ∃ x : ℤ, (2023 + x = 0) ∧ (x = -2023) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2731_273174


namespace NUMINAMATH_CALUDE_ratio_of_repeating_decimals_l2731_273153

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (d : RepeatingDecimal) : ℚ :=
  sorry

/-- The repeating decimal 0.75̅ -/
def zeroPointSevenFive : RepeatingDecimal :=
  { integerPart := 0, repeatingPart := 75 }

/-- The repeating decimal 2.25̅ -/
def twoPointTwoFive : RepeatingDecimal :=
  { integerPart := 2, repeatingPart := 25 }

/-- Theorem stating that the ratio of 0.75̅ to 2.25̅ is equal to 2475/7329 -/
theorem ratio_of_repeating_decimals :
  (toRational zeroPointSevenFive) / (toRational twoPointTwoFive) = 2475 / 7329 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_repeating_decimals_l2731_273153


namespace NUMINAMATH_CALUDE_inequalities_solution_l2731_273150

theorem inequalities_solution (x : ℝ) : 
  (((2*x - 4)*(x - 5) < 0) ↔ (x > 2 ∧ x < 5)) ∧
  ((3*x^2 + 5*x + 1 > 0) ↔ (x < (-5 - Real.sqrt 13) / 6 ∨ x > (-5 + Real.sqrt 13) / 6)) ∧
  (∀ x, -x^2 + x < 2) ∧
  (¬∃ x, 7*x^2 + 5*x + 1 ≤ 0) ∧
  ((4*x ≥ 4*x^2 + 1) ↔ (x = 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_solution_l2731_273150


namespace NUMINAMATH_CALUDE_class_size_l2731_273193

theorem class_size (chinese : ℕ) (math : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : chinese = 15)
  (h2 : math = 18)
  (h3 : both = 8)
  (h4 : neither = 20) :
  chinese + math - both + neither = 45 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l2731_273193


namespace NUMINAMATH_CALUDE_no_valid_two_digit_number_l2731_273132

theorem no_valid_two_digit_number : ¬ ∃ (N : ℕ), 
  (10 ≤ N ∧ N < 100) ∧ 
  (∃ (x : ℕ), 
    x > 3 ∧ 
    N - (10 * (N % 10) + N / 10) = x^3) :=
sorry

end NUMINAMATH_CALUDE_no_valid_two_digit_number_l2731_273132


namespace NUMINAMATH_CALUDE_always_integer_l2731_273190

theorem always_integer (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : (k + 2) ∣ n) :
  ∃ m : ℤ, (n - 3 * k - 2 : ℤ) / (k + 2 : ℤ) * (n.choose k) = m :=
sorry

end NUMINAMATH_CALUDE_always_integer_l2731_273190


namespace NUMINAMATH_CALUDE_max_min_values_on_interval_l2731_273109

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def additive (f : ℝ → ℝ) : Prop := ∀ x y, f (x + y) = f x + f y

theorem max_min_values_on_interval
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_additive : additive f)
  (h_neg : ∀ x > 0, f x < 0)
  (h_f1 : f 1 = -2) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ 6) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ -6) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = 6) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = -6) :=
sorry

end NUMINAMATH_CALUDE_max_min_values_on_interval_l2731_273109


namespace NUMINAMATH_CALUDE_min_triangle_area_l2731_273104

/-- Given a line that passes through (1,2) and intersects positive semi-axes, 
    prove that the minimum area of the triangle formed is 4 -/
theorem min_triangle_area (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_line : 1/m + 2/n = 1) : 
  ∃ (A B : ℝ × ℝ), 
    A.1 > 0 ∧ A.2 = 0 ∧ 
    B.1 = 0 ∧ B.2 > 0 ∧
    (∀ (x y : ℝ), x/m + y/n = 1 → (x = A.1 ∧ y = 0) ∨ (x = 0 ∧ y = B.2)) ∧
    (∀ (C : ℝ × ℝ), C.1 > 0 ∧ C.2 > 0 ∧ C.1/m + C.2/n = 1 → 
      1/2 * A.1 * B.2 ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_l2731_273104


namespace NUMINAMATH_CALUDE_cube_surface_area_l2731_273122

theorem cube_surface_area (d : ℝ) (h : d = 8 * Real.sqrt 3) :
  6 * (d / Real.sqrt 3)^2 = 384 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2731_273122
