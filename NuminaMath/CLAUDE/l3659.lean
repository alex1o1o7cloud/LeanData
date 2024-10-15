import Mathlib

namespace NUMINAMATH_CALUDE_flying_blade_diameter_l3659_365969

theorem flying_blade_diameter (d : ℝ) (n : ℤ) : 
  d = 0.0000009 → d = 9 * 10^n → n = -7 := by sorry

end NUMINAMATH_CALUDE_flying_blade_diameter_l3659_365969


namespace NUMINAMATH_CALUDE_cube_painting_l3659_365989

theorem cube_painting (m : ℚ) : 
  m > 0 → 
  let n : ℚ := 12 / m
  6 * (n - 2)^2 = 12 * (n - 2) → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_cube_painting_l3659_365989


namespace NUMINAMATH_CALUDE_axis_of_symmetry_cosine_l3659_365999

theorem axis_of_symmetry_cosine (x : ℝ) : 
  ∀ k : ℤ, (2 * x + π / 3 = k * π) ↔ x = -π / 6 := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_cosine_l3659_365999


namespace NUMINAMATH_CALUDE_rectangle_max_volume_l3659_365936

def bar_length : ℝ := 18

theorem rectangle_max_volume (length width height : ℝ) :
  length > 0 ∧ width > 0 ∧ height > 0 →
  length = 2 * width →
  2 * (length + width) = bar_length →
  length = 2 ∧ width = 1 ∧ height = 1.5 →
  ∀ (l w h : ℝ), l > 0 ∧ w > 0 ∧ h > 0 →
    l = 2 * w →
    2 * (l + w) = bar_length →
    l * w * h ≤ length * width * height :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_volume_l3659_365936


namespace NUMINAMATH_CALUDE_annual_production_exceeds_plan_l3659_365964

/-- Represents the annual car production plan and actual quarterly production --/
structure CarProduction where
  annual_plan : ℝ
  first_quarter : ℝ
  second_quarter : ℝ
  third_quarter : ℝ
  fourth_quarter : ℝ

/-- Conditions for car production --/
def production_conditions (p : CarProduction) : Prop :=
  p.first_quarter = 0.25 * p.annual_plan ∧
  p.second_quarter = 1.08 * p.first_quarter ∧
  ∃ (k : ℝ), p.second_quarter = 11.25 * k ∧
              p.third_quarter = 12 * k ∧
              p.fourth_quarter = 13.5 * k

/-- Theorem stating that the annual production exceeds the plan by 13.2% --/
theorem annual_production_exceeds_plan (p : CarProduction) 
  (h : production_conditions p) : 
  (p.first_quarter + p.second_quarter + p.third_quarter + p.fourth_quarter) / p.annual_plan = 1.132 :=
sorry

end NUMINAMATH_CALUDE_annual_production_exceeds_plan_l3659_365964


namespace NUMINAMATH_CALUDE_third_vertex_coordinates_l3659_365950

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem third_vertex_coordinates (y : ℝ) :
  y < 0 →
  triangleArea ⟨8, 6⟩ ⟨0, 0⟩ ⟨0, y⟩ = 48 →
  y = -12 := by
  sorry

end NUMINAMATH_CALUDE_third_vertex_coordinates_l3659_365950


namespace NUMINAMATH_CALUDE_expression_value_l3659_365947

theorem expression_value (a b : ℝ) (h : a + b = 1) : a^2 - b^2 + 2*b + 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3659_365947


namespace NUMINAMATH_CALUDE_discount_percentage_l3659_365970

theorem discount_percentage (original_price : ℝ) (discount_rate : ℝ) 
  (h1 : discount_rate = 0.8) : 
  original_price * (1 - discount_rate) = original_price * 0.8 := by
  sorry

#check discount_percentage

end NUMINAMATH_CALUDE_discount_percentage_l3659_365970


namespace NUMINAMATH_CALUDE_age_problem_l3659_365961

theorem age_problem (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 72 → a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l3659_365961


namespace NUMINAMATH_CALUDE_quarters_put_aside_l3659_365926

theorem quarters_put_aside (original_quarters : ℕ) (remaining_quarters : ℕ) : 
  (5 * original_quarters = 350) →
  (remaining_quarters + 350 = 392) →
  (original_quarters - remaining_quarters : ℚ) / original_quarters = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_quarters_put_aside_l3659_365926


namespace NUMINAMATH_CALUDE_prepend_append_divisible_by_72_l3659_365979

theorem prepend_append_divisible_by_72 : ∃ (a b : Nat), 
  a < 10 ∧ b < 10 ∧ 
  (1000 * a + 100 + b = 4104) ∧ 
  (4104 % 72 = 0) := by
  sorry

end NUMINAMATH_CALUDE_prepend_append_divisible_by_72_l3659_365979


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3659_365973

/-- The sum of specific fractions is equal to -2/15 -/
theorem sum_of_fractions :
  (1 : ℚ) / 3 + 1 / 2 + (-5) / 6 + 1 / 5 + 1 / 4 + (-9) / 20 + (-2) / 15 = -2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3659_365973


namespace NUMINAMATH_CALUDE_first_hour_rate_l3659_365975

def shift_duration : ℕ := 4 -- hours
def masks_per_shift : ℕ := 45
def later_rate : ℕ := 6 -- minutes per mask after the first hour

-- x is the time (in minutes) to make one mask in the first hour
theorem first_hour_rate (x : ℕ) : x = 4 ↔ 
  (60 / x : ℚ) + (shift_duration - 1) * (60 / later_rate : ℚ) = masks_per_shift :=
by sorry

end NUMINAMATH_CALUDE_first_hour_rate_l3659_365975


namespace NUMINAMATH_CALUDE_lollipops_kept_by_winnie_l3659_365997

def cherry_lollipops : ℕ := 53
def wintergreen_lollipops : ℕ := 130
def grape_lollipops : ℕ := 12
def shrimp_cocktail_lollipops : ℕ := 240
def number_of_friends : ℕ := 13

def total_lollipops : ℕ := cherry_lollipops + wintergreen_lollipops + grape_lollipops + shrimp_cocktail_lollipops

theorem lollipops_kept_by_winnie :
  total_lollipops % number_of_friends = 6 :=
by sorry

end NUMINAMATH_CALUDE_lollipops_kept_by_winnie_l3659_365997


namespace NUMINAMATH_CALUDE_hayden_ironing_time_l3659_365949

/-- The total ironing time over 4 weeks given daily ironing times and weekly frequency -/
def total_ironing_time (shirt_time pants_time days_per_week num_weeks : ℕ) : ℕ :=
  (shirt_time + pants_time) * days_per_week * num_weeks

/-- Theorem stating that Hayden's total ironing time over 4 weeks is 160 minutes -/
theorem hayden_ironing_time :
  total_ironing_time 5 3 5 4 = 160 := by
  sorry

#eval total_ironing_time 5 3 5 4

end NUMINAMATH_CALUDE_hayden_ironing_time_l3659_365949


namespace NUMINAMATH_CALUDE_simplify_and_add_square_roots_l3659_365948

theorem simplify_and_add_square_roots :
  let x := Real.sqrt 726 / Real.sqrt 484
  let y := Real.sqrt 245 / Real.sqrt 147
  let z := Real.sqrt 1089 / Real.sqrt 441
  x + y + z = (87 + 14 * Real.sqrt 15) / 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_add_square_roots_l3659_365948


namespace NUMINAMATH_CALUDE_wool_price_calculation_l3659_365932

theorem wool_price_calculation (num_sheep : ℕ) (shearing_cost : ℕ) (wool_per_sheep : ℕ) (profit : ℕ) : 
  num_sheep = 200 → 
  shearing_cost = 2000 → 
  wool_per_sheep = 10 → 
  profit = 38000 → 
  (profit + shearing_cost) / (num_sheep * wool_per_sheep) = 20 := by
sorry

end NUMINAMATH_CALUDE_wool_price_calculation_l3659_365932


namespace NUMINAMATH_CALUDE_exponential_decreasing_zero_two_l3659_365901

theorem exponential_decreasing_zero_two (m n : ℝ) : m > n → (0.2 : ℝ) ^ m < (0.2 : ℝ) ^ n := by
  sorry

end NUMINAMATH_CALUDE_exponential_decreasing_zero_two_l3659_365901


namespace NUMINAMATH_CALUDE_inequality_abc_l3659_365902

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b / (a^5 + b^5 + a * b)) + (b * c / (b^5 + c^5 + b * c)) + (c * a / (c^5 + a^5 + c * a)) ≤ 1 ∧
  ((a * b / (a^5 + b^5 + a * b)) + (b * c / (b^5 + c^5 + b * c)) + (c * a / (c^5 + a^5 + c * a)) = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_abc_l3659_365902


namespace NUMINAMATH_CALUDE_b_approximation_l3659_365910

/-- Given that a = 2.68 * 0.74, prove that b = a^2 + cos(a) is approximately 2.96535 -/
theorem b_approximation (a : ℝ) (h : a = 2.68 * 0.74) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ |a^2 + Real.cos a - 2.96535| < ε :=
sorry

end NUMINAMATH_CALUDE_b_approximation_l3659_365910


namespace NUMINAMATH_CALUDE_absolute_difference_of_powers_greater_than_half_l3659_365928

theorem absolute_difference_of_powers_greater_than_half :
  |2^3000 - 3^2006| > (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_absolute_difference_of_powers_greater_than_half_l3659_365928


namespace NUMINAMATH_CALUDE_anns_sledding_speed_l3659_365956

/-- Given the conditions of Mary and Ann's sledding trip, prove Ann's speed -/
theorem anns_sledding_speed 
  (mary_hill_length : ℝ) 
  (mary_speed : ℝ) 
  (ann_hill_length : ℝ) 
  (time_difference : ℝ)
  (h1 : mary_hill_length = 630)
  (h2 : mary_speed = 90)
  (h3 : ann_hill_length = 800)
  (h4 : time_difference = 13)
  : ∃ (ann_speed : ℝ), ann_speed = 40 ∧ 
    ann_hill_length / ann_speed = mary_hill_length / mary_speed + time_difference :=
by sorry

end NUMINAMATH_CALUDE_anns_sledding_speed_l3659_365956


namespace NUMINAMATH_CALUDE_riding_to_total_ratio_l3659_365963

/-- Given a group of horses and men with specific conditions, 
    prove the ratio of riding owners to total owners --/
theorem riding_to_total_ratio 
  (total_horses : ℕ) 
  (total_men : ℕ) 
  (legs_on_ground : ℕ) 
  (h1 : total_horses = 16)
  (h2 : total_men = total_horses)
  (h3 : legs_on_ground = 80) : 
  (total_horses - (legs_on_ground - 4 * total_horses) / 2) / total_horses = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_riding_to_total_ratio_l3659_365963


namespace NUMINAMATH_CALUDE_division_problem_l3659_365918

theorem division_problem (A : ℕ) : A = 8 ↔ 41 = 5 * A + 1 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3659_365918


namespace NUMINAMATH_CALUDE_sum_256_130_in_base6_l3659_365962

-- Define a function to convert a number from base 10 to base 6
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

-- State the theorem
theorem sum_256_130_in_base6 :
  toBase6 (256 + 130) = [1, 0, 4, 2] :=
sorry

end NUMINAMATH_CALUDE_sum_256_130_in_base6_l3659_365962


namespace NUMINAMATH_CALUDE_sum_of_integers_l3659_365935

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val ^ 2 + y.val ^ 2 = 130)
  (h2 : x.val * y.val = 36)
  (h3 : x.val - y.val = 4) :
  x.val + y.val = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3659_365935


namespace NUMINAMATH_CALUDE_inequality_solutions_l3659_365941

theorem inequality_solutions :
  (∀ x : ℝ, |x - 6| ≤ 2 ↔ 4 ≤ x ∧ x ≤ 8) ∧
  (∀ x : ℝ, (x + 3)^2 < 1 ↔ -4 < x ∧ x < -2) ∧
  (∀ x : ℝ, |x| > x ↔ x < 0) ∧
  (∀ x : ℝ, |x^2 - 4*x - 5| > x^2 - 4*x - 5 ↔ -1 < x ∧ x < 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l3659_365941


namespace NUMINAMATH_CALUDE_shortened_tripod_height_l3659_365983

/-- Represents a tripod with potentially unequal legs -/
structure Tripod where
  leg1 : ℝ
  leg2 : ℝ
  leg3 : ℝ

/-- Calculates the height of a tripod given its leg lengths -/
def tripodHeight (t : Tripod) : ℝ := sorry

/-- The original tripod with equal legs -/
def originalTripod : Tripod := ⟨5, 5, 5⟩

/-- The height of the original tripod -/
def originalHeight : ℝ := 4

/-- The tripod with one shortened leg -/
def shortenedTripod : Tripod := ⟨4, 5, 5⟩

/-- The theorem to be proved -/
theorem shortened_tripod_height :
  tripodHeight shortenedTripod = 144 / Real.sqrt (5 * 317) := by sorry

end NUMINAMATH_CALUDE_shortened_tripod_height_l3659_365983


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l3659_365921

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (r s : ℝ), 16 * x^2 + 32 * x - 2048 = 0 ↔ (x + r)^2 = s ∧ s = 129 :=
by sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l3659_365921


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l3659_365958

theorem sum_of_three_consecutive_integers (a : ℤ) (h : a = 29) :
  a + (a + 1) + (a + 2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l3659_365958


namespace NUMINAMATH_CALUDE_harmonic_mean_of_square_sides_l3659_365984

theorem harmonic_mean_of_square_sides (a b c : ℝ) (ha : a = 25) (hb : b = 64) (hc : c = 144) :
  3 / (1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c) = 360 / 49 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_of_square_sides_l3659_365984


namespace NUMINAMATH_CALUDE_mary_flour_needed_l3659_365924

/-- The number of cups of flour required by the recipe -/
def recipe_flour : ℕ := 9

/-- The number of cups of flour Mary has already added -/
def added_flour : ℕ := 2

/-- The number of cups of flour Mary needs to add -/
def flour_needed : ℕ := recipe_flour - added_flour

theorem mary_flour_needed : flour_needed = 7 := by sorry

end NUMINAMATH_CALUDE_mary_flour_needed_l3659_365924


namespace NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l3659_365980

theorem function_inequality_implies_upper_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 1, ∃ x₂ ∈ Set.Icc 2 3, 
    x₁ + 4/x₁ ≥ 2^x₂ + a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l3659_365980


namespace NUMINAMATH_CALUDE_ships_within_visibility_range_l3659_365929

/-- Two ships traveling on perpendicular courses -/
structure Ship where
  x : ℝ
  y : ℝ
  v : ℝ

/-- The problem setup -/
def ship_problem (v : ℝ) : Prop :=
  let ship1 : Ship := ⟨20, 0, v⟩
  let ship2 : Ship := ⟨0, 15, v⟩
  ∃ t : ℝ, t ≥ 0 ∧ 
    ((20 - v * t)^2 + (15 - v * t)^2) ≤ 4^2

/-- The main theorem -/
theorem ships_within_visibility_range (v : ℝ) (h : v > 0) : 
  ship_problem v :=
sorry

end NUMINAMATH_CALUDE_ships_within_visibility_range_l3659_365929


namespace NUMINAMATH_CALUDE_water_sales_profit_profit_for_240_barrels_barrels_for_760_profit_l3659_365923

/-- Represents the daily sales and profit of a water sales department -/
structure WaterSales where
  fixed_costs : ℕ := 200
  cost_price : ℕ := 5
  selling_price : ℕ := 8

/-- Calculates the daily profit based on the number of barrels sold -/
def daily_profit (ws : WaterSales) (x : ℕ) : ℤ :=
  (ws.selling_price * x : ℤ) - (ws.cost_price * x : ℤ) - ws.fixed_costs

theorem water_sales_profit (ws : WaterSales) :
  ∀ x : ℕ, daily_profit ws x = 3 * x - 200 := by sorry

theorem profit_for_240_barrels (ws : WaterSales) :
  daily_profit ws 240 = 520 := by sorry

theorem barrels_for_760_profit (ws : WaterSales) :
  ∃ x : ℕ, daily_profit ws x = 760 ∧ x = 320 := by sorry

end NUMINAMATH_CALUDE_water_sales_profit_profit_for_240_barrels_barrels_for_760_profit_l3659_365923


namespace NUMINAMATH_CALUDE_max_min_on_interval_l3659_365990

def f (x : ℝ) := x^3 - 3*x^2 + 5

theorem max_min_on_interval :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc 1 3, f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc 1 3, f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc 1 3, f x₂ = max) ∧
    min = 1 ∧ max = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_min_on_interval_l3659_365990


namespace NUMINAMATH_CALUDE_all_positive_integers_expressible_l3659_365981

theorem all_positive_integers_expressible (n : ℕ+) :
  ∃ (a b c : ℤ), (n : ℤ) = a^2 + b^2 + c^2 + c := by sorry

end NUMINAMATH_CALUDE_all_positive_integers_expressible_l3659_365981


namespace NUMINAMATH_CALUDE_area_is_seven_and_half_l3659_365904

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (hf_continuous : Continuous f)
variable (hf_monotone : Monotone f)
variable (hf_0 : f 0 = 0)
variable (hf_1 : f 1 = 1)

-- Define the area calculation function
def area_bounded (f : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem area_is_seven_and_half :
  area_bounded f = 7.5 :=
sorry

end NUMINAMATH_CALUDE_area_is_seven_and_half_l3659_365904


namespace NUMINAMATH_CALUDE_complement_of_29_45_l3659_365951

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- The complement of an angle is the angle that when added to the original angle results in 90° -/
def complement (a : Angle) : Angle :=
  sorry

theorem complement_of_29_45 :
  complement ⟨29, 45⟩ = ⟨60, 15⟩ :=
sorry

end NUMINAMATH_CALUDE_complement_of_29_45_l3659_365951


namespace NUMINAMATH_CALUDE_vector_addition_l3659_365982

theorem vector_addition (a b : ℝ × ℝ) :
  a = (2, 1) → b = (-3, 4) → a + b = (-1, 5) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l3659_365982


namespace NUMINAMATH_CALUDE_cosine_equality_l3659_365945

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (865 * π / 180) → n = 35 ∨ n = 145 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l3659_365945


namespace NUMINAMATH_CALUDE_sine_cosine_increasing_interval_l3659_365995

theorem sine_cosine_increasing_interval :
  ∀ (a b : ℝ), (a = -π / 2 ∧ b = 0) ↔ 
    (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → 
      (Real.sin x < Real.sin y ∧ Real.cos x < Real.cos y)) ∧
    (¬(∀ x y, -π ≤ x ∧ x < y ∧ y ≤ -π / 2 → 
      (Real.sin x < Real.sin y ∧ Real.cos x < Real.cos y))) ∧
    (¬(∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 2 → 
      (Real.sin x < Real.sin y ∧ Real.cos x < Real.cos y))) ∧
    (¬(∀ x y, π / 2 ≤ x ∧ x < y ∧ y ≤ π → 
      (Real.sin x < Real.sin y ∧ Real.cos x < Real.cos y))) :=
by sorry

end NUMINAMATH_CALUDE_sine_cosine_increasing_interval_l3659_365995


namespace NUMINAMATH_CALUDE_common_chord_length_is_10_l3659_365959

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 2*y - 40 = 0

/-- The length of the common chord of two intersecting circles -/
def common_chord_length : ℝ := 10

/-- Theorem: The length of the common chord of the given intersecting circles is 10 -/
theorem common_chord_length_is_10 :
  ∃ (A B : ℝ × ℝ),
    circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
    circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = common_chord_length :=
by sorry

end NUMINAMATH_CALUDE_common_chord_length_is_10_l3659_365959


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_no_solution_for_2009_l3659_365992

theorem cubic_equation_solutions (n : ℕ+) :
  (∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = n) →
  (∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℤ),
    (x₁^3 - 3*x₁*y₁^2 + y₁^3 = n) ∧
    (x₂^3 - 3*x₂*y₂^2 + y₂^3 = n) ∧
    (x₃^3 - 3*x₃*y₃^2 + y₃^3 = n) ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₃, y₃)) :=
by sorry

theorem no_solution_for_2009 :
  ¬ ∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = 2009 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_no_solution_for_2009_l3659_365992


namespace NUMINAMATH_CALUDE_power_relation_l3659_365927

theorem power_relation (a m n : ℝ) (h1 : a^m = 6) (h2 : a^(m-n) = 2) : a^n = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l3659_365927


namespace NUMINAMATH_CALUDE_meal_combinations_l3659_365934

theorem meal_combinations (entrees drinks desserts : ℕ) 
  (h_entrees : entrees = 4)
  (h_drinks : drinks = 4)
  (h_desserts : desserts = 2) : 
  entrees * drinks * desserts = 32 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_l3659_365934


namespace NUMINAMATH_CALUDE_equation_solution_l3659_365925

/-- Given the equation and values for a, b, c, and d, prove that x equals 26544.74 -/
theorem equation_solution :
  let a : ℝ := 3
  let b : ℝ := 5
  let c : ℝ := 2
  let d : ℝ := 4
  let x : ℝ := ((a^2 * b * (47 / 100 * 1442)) - (c * d * (36 / 100 * 1412))) + 63
  x = 26544.74 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3659_365925


namespace NUMINAMATH_CALUDE_sport_formulation_comparison_l3659_365939

/-- Represents the ratio of flavoring to corn syrup to water in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation of the drink -/
def standard : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport : DrinkRatio :=
  { flavoring := 1, corn_syrup := 4, water := 60 }

theorem sport_formulation_comparison : 
  (sport.flavoring / sport.water) / (standard.flavoring / standard.water) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sport_formulation_comparison_l3659_365939


namespace NUMINAMATH_CALUDE_prob_more_than_4_draws_is_31_35_l3659_365909

-- Define the number of new and old coins
def new_coins : ℕ := 3
def old_coins : ℕ := 4
def total_coins : ℕ := new_coins + old_coins

-- Define the probability function
noncomputable def prob_more_than_4_draws : ℚ :=
  1 - (
    -- Probability of drawing all new coins in first 3 draws
    (new_coins / total_coins) * ((new_coins - 1) / (total_coins - 1)) * ((new_coins - 2) / (total_coins - 2)) +
    -- Probability of drawing all new coins in first 4 draws (but not in first 3)
    3 * ((old_coins / total_coins) * (new_coins / (total_coins - 1)) * ((new_coins - 1) / (total_coins - 2)) * ((new_coins - 2) / (total_coins - 3)))
  )

-- Theorem statement
theorem prob_more_than_4_draws_is_31_35 : prob_more_than_4_draws = 31 / 35 :=
  sorry

end NUMINAMATH_CALUDE_prob_more_than_4_draws_is_31_35_l3659_365909


namespace NUMINAMATH_CALUDE_geometric_difference_sequence_properties_l3659_365943

/-- A geometric difference sequence -/
def GeometricDifferenceSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, (a (n + 2) / a (n + 1)) - (a (n + 1) / a n) = d

theorem geometric_difference_sequence_properties
  (a : ℕ → ℚ)
  (h_gds : GeometricDifferenceSequence a)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 1)
  (h_a3 : a 3 = 3) :
  a 5 = 105 ∧ a 31 / a 29 = 3363 := by
  sorry

end NUMINAMATH_CALUDE_geometric_difference_sequence_properties_l3659_365943


namespace NUMINAMATH_CALUDE_solve_potatoes_problem_l3659_365905

def potatoes_problem (initial : ℕ) (gina : ℕ) : Prop :=
  let tom := 2 * gina
  let anne := tom / 3
  let remaining := initial - (gina + tom + anne)
  remaining = 47

theorem solve_potatoes_problem :
  potatoes_problem 300 69 := by
  sorry

end NUMINAMATH_CALUDE_solve_potatoes_problem_l3659_365905


namespace NUMINAMATH_CALUDE_inequality_proof_l3659_365954

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3659_365954


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3659_365996

theorem algebraic_expression_value (a b : ℝ) 
  (h1 : a + b = 3) 
  (h2 : a * b = 2) : 
  a^2 * b + 2 * a^2 * b^2 + a * b^3 = 18 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3659_365996


namespace NUMINAMATH_CALUDE_gorilla_to_cat_dog_ratio_l3659_365957

/-- Represents the lengths of animal videos and their ratio -/
structure AnimalVideos where
  cat_length : ℕ
  dog_length : ℕ
  total_time : ℕ
  gorilla_length : ℕ
  ratio : Rat

/-- Theorem stating the ratio of gorilla video length to combined cat and dog video length -/
theorem gorilla_to_cat_dog_ratio (v : AnimalVideos) 
  (h1 : v.cat_length = 4)
  (h2 : v.dog_length = 2 * v.cat_length)
  (h3 : v.total_time = 36)
  (h4 : v.gorilla_length = v.total_time - (v.cat_length + v.dog_length))
  (h5 : v.ratio = v.gorilla_length / (v.cat_length + v.dog_length)) :
  v.ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_gorilla_to_cat_dog_ratio_l3659_365957


namespace NUMINAMATH_CALUDE_fraction_simplification_l3659_365967

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 48 + 3 * Real.sqrt 75 + 5 * Real.sqrt 27) = (5 * Real.sqrt 3) / 102 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3659_365967


namespace NUMINAMATH_CALUDE_range_of_a_l3659_365917

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x * Real.exp x else a * x^2 - 2 * x

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, y ≥ -(1 / Real.exp 1) → ∃ x : ℝ, f a x = y) →
  (∀ x : ℝ, f a x ≥ -(1 / Real.exp 1)) →
  a ≥ Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3659_365917


namespace NUMINAMATH_CALUDE_remaining_work_completion_time_l3659_365933

theorem remaining_work_completion_time 
  (a_completion_time b_completion_time b_work_days : ℝ) 
  (ha : a_completion_time = 6)
  (hb : b_completion_time = 15)
  (hbw : b_work_days = 10) : 
  (1 - b_work_days / b_completion_time) / (1 / a_completion_time) = 2 := by
  sorry

end NUMINAMATH_CALUDE_remaining_work_completion_time_l3659_365933


namespace NUMINAMATH_CALUDE_bridge_length_l3659_365987

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 140 ∧ train_speed_kmh = 45 ∧ crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 235 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_l3659_365987


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3659_365953

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    (4 * π * r^2 : ℝ) = 400 * π → 
    (4 / 3 : ℝ) * π * r^3 = (4000 / 3 : ℝ) * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3659_365953


namespace NUMINAMATH_CALUDE_system_solution_sum_of_squares_l3659_365930

theorem system_solution_sum_of_squares (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + 2*x + 2*y = 120) :
  x^2 + y^2 = 80 := by
sorry

end NUMINAMATH_CALUDE_system_solution_sum_of_squares_l3659_365930


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3659_365977

/-- Given a line segment from (4, 3) to (x, 9) with length 15 and x < 0, prove x = 4 - √189 -/
theorem line_segment_endpoint (x : ℝ) : 
  x < 0 → 
  ((x - 4)^2 + (9 - 3)^2 : ℝ) = 15^2 → 
  x = 4 - Real.sqrt 189 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3659_365977


namespace NUMINAMATH_CALUDE_five_by_five_decomposition_l3659_365944

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square grid -/
structure Grid where
  size : ℕ

/-- Checks if a list of rectangles can fit in a grid -/
def canFitInGrid (grid : Grid) (rectangles : List Rectangle) : Prop :=
  (rectangles.map (λ r => r.width * r.height)).sum = grid.size * grid.size

/-- Theorem: A 5x5 grid can be decomposed into 1x3 and 1x4 rectangles -/
theorem five_by_five_decomposition :
  ∃ (rectangles : List Rectangle),
    (∀ r ∈ rectangles, r.width = 1 ∧ (r.height = 3 ∨ r.height = 4)) ∧
    canFitInGrid { size := 5 } rectangles :=
  sorry

end NUMINAMATH_CALUDE_five_by_five_decomposition_l3659_365944


namespace NUMINAMATH_CALUDE_min_max_area_14_sided_lattice_polygon_l3659_365972

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A lattice polygon is a polygon with all vertices at lattice points -/
structure LatticePolygon where
  vertices : List LatticePoint
  is_convex : Bool
  is_closed : Bool

/-- A lattice parallelogram is a parallelogram with all vertices at lattice points -/
structure LatticeParallelogram where
  vertices : List LatticePoint
  is_parallelogram : Bool

def area (p : LatticeParallelogram) : ℚ :=
  sorry

def can_be_divided_into_parallelograms (poly : LatticePolygon) (parallelograms : List LatticeParallelogram) : Prop :=
  sorry

theorem min_max_area_14_sided_lattice_polygon :
  ∀ (poly : LatticePolygon) (parallelograms : List LatticeParallelogram),
    poly.vertices.length = 14 →
    poly.is_convex →
    can_be_divided_into_parallelograms poly parallelograms →
    (∃ (C : ℚ), ∀ (p : LatticeParallelogram), p ∈ parallelograms → area p ≤ C) →
    (∀ (C : ℚ), (∀ (p : LatticeParallelogram), p ∈ parallelograms → area p ≤ C) → C ≥ 5) :=
  sorry

end NUMINAMATH_CALUDE_min_max_area_14_sided_lattice_polygon_l3659_365972


namespace NUMINAMATH_CALUDE_point_opposite_sides_line_value_range_l3659_365974

/-- Given that the points (3,1) and (-4,6) lie on opposite sides of the line 3x - 2y + a = 0,
    prove that the value range of a is -7 < a < 24. -/
theorem point_opposite_sides_line_value_range (a : ℝ) : 
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 → -7 < a ∧ a < 24 := by
  sorry

end NUMINAMATH_CALUDE_point_opposite_sides_line_value_range_l3659_365974


namespace NUMINAMATH_CALUDE_josh_age_at_marriage_l3659_365965

/-- Proves that Josh's age at marriage was 22 given the conditions of the problem -/
theorem josh_age_at_marriage :
  ∀ (josh_age_at_marriage : ℕ),
    (josh_age_at_marriage + 30 + (28 + 30) = 5 * josh_age_at_marriage) →
    josh_age_at_marriage = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_josh_age_at_marriage_l3659_365965


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_negative_one_l3659_365960

theorem intersection_nonempty_implies_a_greater_than_negative_one 
  (A : Set ℝ) (B : Set ℝ) (a : ℝ) :
  A = {x : ℝ | -1 ≤ x ∧ x < 2} →
  B = {x : ℝ | x < a} →
  (A ∩ B).Nonempty →
  a > -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_negative_one_l3659_365960


namespace NUMINAMATH_CALUDE_proposition_and_variants_are_false_l3659_365911

theorem proposition_and_variants_are_false :
  (¬ ∀ (a b : ℝ), ab ≤ 0 → a ≤ 0 ∨ b ≤ 0) ∧
  (¬ ∀ (a b : ℝ), (a ≤ 0 ∨ b ≤ 0) → ab ≤ 0) ∧
  (¬ ∀ (a b : ℝ), ab > 0 → a > 0 ∧ b > 0) ∧
  (¬ ∀ (a b : ℝ), (a > 0 ∧ b > 0) → ab > 0) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_variants_are_false_l3659_365911


namespace NUMINAMATH_CALUDE_ace_probabilities_l3659_365940

/-- The number of cards in a standard deck without jokers -/
def deck_size : ℕ := 52

/-- The number of Aces in a standard deck -/
def num_aces : ℕ := 4

/-- The probability of drawing an Ace twice without replacement from a standard deck -/
def prob_two_aces : ℚ := 1 / 221

/-- The conditional probability of drawing an Ace on the second draw, given that the first card drawn is an Ace -/
def prob_second_ace_given_first : ℚ := 1 / 17

/-- Theorem stating the probabilities for drawing Aces from a standard deck -/
theorem ace_probabilities :
  (prob_two_aces = (num_aces : ℚ) / deck_size * (num_aces - 1) / (deck_size - 1)) ∧
  (prob_second_ace_given_first = (num_aces - 1 : ℚ) / (deck_size - 1)) :=
sorry

end NUMINAMATH_CALUDE_ace_probabilities_l3659_365940


namespace NUMINAMATH_CALUDE_relative_complement_M_N_l3659_365922

def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {1, 2}

theorem relative_complement_M_N : (M \ N) = {-1, 0, 3} := by
  sorry

end NUMINAMATH_CALUDE_relative_complement_M_N_l3659_365922


namespace NUMINAMATH_CALUDE_sams_candy_count_l3659_365955

/-- Represents the candy count for each friend -/
structure CandyCounts where
  bob : ℕ
  mary : ℕ
  john : ℕ
  sue : ℕ
  sam : ℕ

/-- The total candy count for all friends -/
def totalCandy : ℕ := 50

/-- The given candy counts for Bob, Mary, John, and Sue -/
def givenCounts : CandyCounts where
  bob := 10
  mary := 5
  john := 5
  sue := 20
  sam := 0  -- We don't know Sam's count yet, so we initialize it to 0

/-- Theorem stating that Sam's candy count is equal to the total minus the sum of others -/
theorem sams_candy_count (c : CandyCounts) (h : c = givenCounts) :
  c.sam = totalCandy - (c.bob + c.mary + c.john + c.sue) :=
by
  sorry

#check sams_candy_count

end NUMINAMATH_CALUDE_sams_candy_count_l3659_365955


namespace NUMINAMATH_CALUDE_third_derivative_y_l3659_365913

noncomputable def y (x : ℝ) : ℝ := (Real.log (3 + x)) / (3 + x)

theorem third_derivative_y (x : ℝ) (h : x ≠ -3) : 
  (deriv^[3] y) x = (11 - 6 * Real.log (3 + x)) / (3 + x)^4 :=
by sorry

end NUMINAMATH_CALUDE_third_derivative_y_l3659_365913


namespace NUMINAMATH_CALUDE_counterexample_exists_l3659_365966

theorem counterexample_exists : ∃ n : ℕ, 
  n > 1 ∧ 
  ¬(Nat.Prime n) ∧ 
  ¬(Nat.Prime (n + 2)) ∧ 
  n = 14 :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3659_365966


namespace NUMINAMATH_CALUDE_face_mask_profit_l3659_365907

/-- Calculates the profit from selling face masks given the following conditions:
  * 12 boxes of face masks were bought at $9 per box
  * Each box contains 50 masks
  * 6 boxes were repacked and sold at $5 per 25 pieces
  * The remaining 300 masks were sold in baggies at 10 pieces for $3
-/
def calculate_profit : ℤ :=
  let boxes_bought := 12
  let cost_per_box := 9
  let masks_per_box := 50
  let repacked_boxes := 6
  let price_per_25_pieces := 5
  let remaining_masks := 300
  let price_per_10_pieces := 3

  let total_cost := boxes_bought * cost_per_box
  let revenue_repacked := repacked_boxes * (masks_per_box / 25) * price_per_25_pieces
  let revenue_baggies := (remaining_masks / 10) * price_per_10_pieces
  let total_revenue := revenue_repacked + revenue_baggies
  
  total_revenue - total_cost

/-- Theorem stating that the profit from selling face masks under the given conditions is $42 -/
theorem face_mask_profit : calculate_profit = 42 := by
  sorry

end NUMINAMATH_CALUDE_face_mask_profit_l3659_365907


namespace NUMINAMATH_CALUDE_min_value_3x_plus_9y_l3659_365914

theorem min_value_3x_plus_9y (x y : ℝ) (h : x + 2 * y = 2) :
  3 * x + 9 * y ≥ 6 ∧ ∃ x₀ y₀ : ℝ, x₀ + 2 * y₀ = 2 ∧ 3 * x₀ + 9 * y₀ = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_3x_plus_9y_l3659_365914


namespace NUMINAMATH_CALUDE_circle_equation_l3659_365993

/-- The standard equation of a circle with given center and radius -/
theorem circle_equation (x y : ℝ) : 
  (∃ (C : ℝ × ℝ) (r : ℝ), C = (1, -2) ∧ r = 3) →
  ((x - 1)^2 + (y + 2)^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3659_365993


namespace NUMINAMATH_CALUDE_total_students_is_thirteen_l3659_365998

/-- The total number of students in a presentation lineup, given Eunjung's position and the number of students after her. -/
def total_students (eunjung_position : ℕ) (students_after : ℕ) : ℕ :=
  eunjung_position + students_after

/-- Theorem stating that the total number of students is 13, given the conditions from the problem. -/
theorem total_students_is_thirteen :
  total_students 6 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_thirteen_l3659_365998


namespace NUMINAMATH_CALUDE_intersection_properties_l3659_365920

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (2, -1)

-- Define the intersection points M and N (existence assumed)
axiom exists_intersection_points : ∃ (M N : ℝ × ℝ), 
  curve_C M.1 M.2 ∧ line_l M.1 M.2 ∧
  curve_C N.1 N.2 ∧ line_l N.1 N.2 ∧
  M ≠ N

-- State the theorem
theorem intersection_properties :
  ∃ (M N : ℝ × ℝ), 
    curve_C M.1 M.2 ∧ line_l M.1 M.2 ∧
    curve_C N.1 N.2 ∧ line_l N.1 N.2 ∧
    M ≠ N ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 8 ∧
    Real.sqrt ((M.1 - point_P.1)^2 + (M.2 - point_P.2)^2) *
    Real.sqrt ((N.1 - point_P.1)^2 + (N.2 - point_P.2)^2) = 14 := by
  sorry


end NUMINAMATH_CALUDE_intersection_properties_l3659_365920


namespace NUMINAMATH_CALUDE_solve_equation_l3659_365986

theorem solve_equation (x : ℝ) : (2 * x + 7) / 7 = 13 → x = 42 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3659_365986


namespace NUMINAMATH_CALUDE_find_divisor_l3659_365915

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 140)
  (h2 : quotient = 9)
  (h3 : remainder = 5)
  : ∃ divisor : ℕ, 
    dividend = divisor * quotient + remainder ∧ 
    divisor = 15 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3659_365915


namespace NUMINAMATH_CALUDE_P_greater_than_Q_l3659_365942

theorem P_greater_than_Q (a : ℝ) : (a^2 + 2*a) > (3*a - 1) := by
  sorry

end NUMINAMATH_CALUDE_P_greater_than_Q_l3659_365942


namespace NUMINAMATH_CALUDE_max_subset_with_distinct_sums_l3659_365919

def S (A : Finset ℕ) : Finset ℕ :=
  Finset.powerset A \ {∅} |>.image (λ B => B.sum id)

theorem max_subset_with_distinct_sums :
  (∃ (A : Finset ℕ), A ⊆ Finset.range 16 ∧ A.card = 5 ∧ S A.toSet.toFinset = S A) ∧
  ¬(∃ (A : Finset ℕ), A ⊆ Finset.range 16 ∧ A.card = 6 ∧ S A.toSet.toFinset = S A) :=
sorry

end NUMINAMATH_CALUDE_max_subset_with_distinct_sums_l3659_365919


namespace NUMINAMATH_CALUDE_island_perimeter_l3659_365937

/-- The perimeter of an island consisting of an equilateral triangle and two half circles -/
theorem island_perimeter (base : ℝ) (h : base = 4) : 
  let triangle_perimeter := 3 * base
  let half_circles_perimeter := 2 * π * base
  triangle_perimeter + half_circles_perimeter = 12 + 4 * π := by
  sorry

end NUMINAMATH_CALUDE_island_perimeter_l3659_365937


namespace NUMINAMATH_CALUDE_farmer_seeds_l3659_365991

theorem farmer_seeds (final_buckets sowed_buckets : ℝ) 
  (h1 : final_buckets = 6)
  (h2 : sowed_buckets = 2.75) : 
  final_buckets + sowed_buckets = 8.75 := by
  sorry

end NUMINAMATH_CALUDE_farmer_seeds_l3659_365991


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3659_365968

/-- A function f satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The coefficients of f in its quadratic form -/
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

/-- The main theorem stating that a + b + c = 50 -/
theorem sum_of_coefficients :
  (∀ x, f (x + 5) = 5 * x^2 + 9 * x + 6) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 50 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3659_365968


namespace NUMINAMATH_CALUDE_square_coins_problem_l3659_365938

theorem square_coins_problem (perimeter_coins : ℕ) (h : perimeter_coins = 240) :
  let side_length := (perimeter_coins + 4) / 4
  side_length * side_length = 3721 := by
  sorry

end NUMINAMATH_CALUDE_square_coins_problem_l3659_365938


namespace NUMINAMATH_CALUDE_g_three_sixteenths_l3659_365916

-- Define the properties of g
def g_properties (g : ℝ → ℝ) : Prop :=
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ g x ∧ g x ≤ 1) ∧
  (g 0 = 0) ∧
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = g x / 2)

-- Theorem statement
theorem g_three_sixteenths (g : ℝ → ℝ) (h : g_properties g) : g (3/16) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_g_three_sixteenths_l3659_365916


namespace NUMINAMATH_CALUDE_ellipse_and_point_theorem_l3659_365988

-- Define the ellipse M
structure Ellipse :=
  (a b : ℝ)
  (center_x center_y : ℝ)
  (eccentricity : ℝ)

-- Define the line l
structure Line :=
  (m : ℝ)
  (b : ℝ)

-- Define a point
structure Point :=
  (x y : ℝ)

-- Define the problem
theorem ellipse_and_point_theorem 
  (M : Ellipse)
  (l : Line)
  (N : ℝ → Point) :
  M.center_x = 0 ∧ 
  M.center_y = 0 ∧ 
  M.a = 2 ∧ 
  M.eccentricity = 1/2 ∧
  l.m ≠ 0 ∧
  (∀ t, N t = Point.mk t 0) →
  (M.a^2 * M.b^2 = 12 ∧ 
   (∀ t, (0 < t ∧ t < 1/4) ↔ 
     ∃ A B : Point, 
       A.x^2 / 4 + A.y^2 / 3 = 1 ∧
       B.x^2 / 4 + B.y^2 / 3 = 1 ∧
       A.x = l.m * A.y + l.b ∧
       B.x = l.m * B.y + l.b ∧
       ((A.x - t)^2 + A.y^2 = (B.x - t)^2 + B.y^2) ∧
       ((A.x - (N t).x) * (B.y - A.y) = (A.y - (N t).y) * (B.x - A.x)))) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_point_theorem_l3659_365988


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3659_365976

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 3 / 4 → 
  Nat.gcd a b = 8 → 
  Nat.lcm a b = 96 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3659_365976


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l3659_365994

/-- Represents the mall's sales and profit model -/
structure MallSales where
  initial_sales : ℕ  -- Initial daily sales
  initial_profit : ℕ  -- Initial profit per item in yuan
  sales_increase_rate : ℕ  -- Additional items sold per yuan of price reduction
  price_reduction : ℕ  -- Price reduction in yuan

/-- Calculates the daily profit given a MallSales structure -/
def daily_profit (m : MallSales) : ℕ :=
  let new_sales := m.initial_sales + m.sales_increase_rate * m.price_reduction
  let new_profit_per_item := m.initial_profit - m.price_reduction
  new_sales * new_profit_per_item

/-- Theorem stating that a price reduction of 20 yuan results in a daily profit of 2100 yuan -/
theorem price_reduction_theorem (m : MallSales) 
  (h1 : m.initial_sales = 30)
  (h2 : m.initial_profit = 50)
  (h3 : m.sales_increase_rate = 2)
  (h4 : m.price_reduction = 20) :
  daily_profit m = 2100 := by
  sorry

#eval daily_profit { initial_sales := 30, initial_profit := 50, sales_increase_rate := 2, price_reduction := 20 }

end NUMINAMATH_CALUDE_price_reduction_theorem_l3659_365994


namespace NUMINAMATH_CALUDE_quadratic_solution_l3659_365908

theorem quadratic_solution (c : ℝ) : 
  ((-9 : ℝ)^2 + c * (-9 : ℝ) + 45 = 0) → c = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3659_365908


namespace NUMINAMATH_CALUDE_greatest_fleet_number_l3659_365978

/-- A ship is a set of connected unit squares on a grid. -/
def Ship := Set (Nat × Nat)

/-- A fleet is a set of vertex-disjoint ships. -/
def Fleet := Set Ship

/-- The grid size. -/
def gridSize : Nat := 10

/-- Checks if two ships are vertex-disjoint. -/
def vertexDisjoint (s1 s2 : Ship) : Prop := sorry

/-- Checks if a fleet is valid (all ships are vertex-disjoint). -/
def validFleet (f : Fleet) : Prop := sorry

/-- Checks if a ship is within the grid bounds. -/
def inGridBounds (s : Ship) : Prop := sorry

/-- Checks if a fleet configuration is valid for a given partition. -/
def validFleetForPartition (n : Nat) (partition : List Nat) (f : Fleet) : Prop := sorry

/-- The main theorem stating that 25 is the greatest number satisfying the fleet condition. -/
theorem greatest_fleet_number : 
  (∀ (partition : List Nat), partition.sum = 25 → 
    ∃ (f : Fleet), validFleet f ∧ validFleetForPartition 25 partition f) ∧
  (∀ (n : Nat), n > 25 → 
    ∃ (partition : List Nat), partition.sum = n ∧
      ¬∃ (f : Fleet), validFleet f ∧ validFleetForPartition n partition f) :=
sorry

end NUMINAMATH_CALUDE_greatest_fleet_number_l3659_365978


namespace NUMINAMATH_CALUDE_milk_fraction_after_pouring_l3659_365903

/-- Represents a cup containing a mixture of tea and milk -/
structure Cup where
  tea : ℚ
  milk : ℚ

/-- The pouring process described in the problem -/
def pour_process (initial_tea_cup : Cup) (initial_milk_cup : Cup) : Cup :=
  let first_pour := Cup.mk (initial_tea_cup.tea - 2) initial_milk_cup.milk
  let second_cup_total := initial_milk_cup.tea + initial_milk_cup.milk + 2
  let milk_ratio := initial_milk_cup.milk / second_cup_total
  let tea_ratio := (initial_milk_cup.tea + 2) / second_cup_total
  Cup.mk (first_pour.tea + 2 * tea_ratio) (first_pour.milk + 2 * milk_ratio)

theorem milk_fraction_after_pouring 
  (initial_tea_cup : Cup) 
  (initial_milk_cup : Cup) 
  (h1 : initial_tea_cup.tea = 6) 
  (h2 : initial_tea_cup.milk = 0) 
  (h3 : initial_milk_cup.tea = 0) 
  (h4 : initial_milk_cup.milk = 6) :
  let final_cup := pour_process initial_tea_cup initial_milk_cup
  (final_cup.milk / (final_cup.tea + final_cup.milk)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_milk_fraction_after_pouring_l3659_365903


namespace NUMINAMATH_CALUDE_calculation_proof_l3659_365906

theorem calculation_proof : ((-1/3)⁻¹ : ℝ) - (Real.sqrt 3 - 2)^0 + 4 * Real.cos (π/4) = -4 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3659_365906


namespace NUMINAMATH_CALUDE_mathematics_puzzle_solution_l3659_365900

/-- Represents a mapping from characters to either digits or arithmetic operations -/
def LetterMapping := Char → Option (Nat ⊕ Bool)

/-- The word to be mapped -/
def word : List Char := ['M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S']

/-- Evaluates an expression given a mapping -/
def evalExpression (mapping : LetterMapping) (expr : List Char) : Option Int := sorry

/-- Checks if a mapping is valid according to the problem constraints -/
def isValidMapping (mapping : LetterMapping) : Prop := sorry

theorem mathematics_puzzle_solution :
  ∃ (mapping : LetterMapping),
    isValidMapping mapping ∧
    evalExpression mapping word = some 2014 := by sorry

end NUMINAMATH_CALUDE_mathematics_puzzle_solution_l3659_365900


namespace NUMINAMATH_CALUDE_tetrahedron_coloring_tetrahedron_coloring_converse_l3659_365952

/-- The number of distinct colorings of a regular tetrahedron -/
def distinct_colorings (n : ℕ) : ℚ := (n^4 + 11*n^2) / 12

/-- The theorem stating the possible values of n -/
theorem tetrahedron_coloring (n : ℕ) (hn : n > 0) :
  distinct_colorings n = n^3 → n = 1 ∨ n = 11 := by
sorry

/-- The converse of the theorem -/
theorem tetrahedron_coloring_converse (n : ℕ) (hn : n > 0) :
  (n = 1 ∨ n = 11) → distinct_colorings n = n^3 := by
sorry

end NUMINAMATH_CALUDE_tetrahedron_coloring_tetrahedron_coloring_converse_l3659_365952


namespace NUMINAMATH_CALUDE_horner_v3_equals_neg_57_l3659_365971

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 12 + 35x - 8x² + 79x³ + 6x⁴ + 5x⁵ + 3x⁶ -/
def f : List ℝ := [12, 35, -8, 79, 6, 5, 3]

/-- Theorem: V₃ in Horner's method for f(x) at x = -4 is -57 -/
theorem horner_v3_equals_neg_57 :
  (horner (f.reverse.take 4) (-4) : ℝ) = -57 := by
  sorry

end NUMINAMATH_CALUDE_horner_v3_equals_neg_57_l3659_365971


namespace NUMINAMATH_CALUDE_characterize_satisfying_functions_l3659_365931

/-- A function satisfying the given inequality for all real numbers x < y < z -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, x < y → y < z →
    f y - ((z - y) / (z - x) * f x + (y - x) / (z - x) * f z) ≤ f ((x + z) / 2) - (f x + f z) / 2

/-- The characterization of functions satisfying the inequality -/
theorem characterize_satisfying_functions :
  ∀ f : ℝ → ℝ, SatisfiesInequality f ↔
    ∃ a b c : ℝ, a ≤ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c :=
by sorry

end NUMINAMATH_CALUDE_characterize_satisfying_functions_l3659_365931


namespace NUMINAMATH_CALUDE_linear_function_properties_l3659_365946

-- Define the linear function
def f (k b x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem linear_function_properties
  (k b : ℝ)
  (h1 : f k b 1 = 0)
  (h2 : f k b 0 = 2)
  (m : ℝ)
  (h3 : -2 < m)
  (h4 : m ≤ 3) :
  k = -2 ∧ b = 2 ∧ -4 ≤ f k b m ∧ f k b m < 6 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l3659_365946


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l3659_365912

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x^2 - 4 * x

theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (y = f x) →
    (y = m * (x - 1) + f 1) →
    (m * x - y - b = 0) →
    (m = Real.exp 1) ∧
    (b = 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l3659_365912


namespace NUMINAMATH_CALUDE_arrangement_count_l3659_365985

def committee_size : ℕ := 12
def num_men : ℕ := 3
def num_women : ℕ := 9

theorem arrangement_count :
  (committee_size.choose num_men) = 220 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l3659_365985
