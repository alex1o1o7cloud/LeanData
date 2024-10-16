import Mathlib

namespace NUMINAMATH_CALUDE_jeff_saturday_laps_l2013_201312

theorem jeff_saturday_laps (total_laps : ℕ) (sunday_morning_laps : ℕ) (remaining_laps : ℕ) 
  (h1 : total_laps = 98)
  (h2 : sunday_morning_laps = 15)
  (h3 : remaining_laps = 56) :
  total_laps - (sunday_morning_laps + remaining_laps) = 27 := by
  sorry

end NUMINAMATH_CALUDE_jeff_saturday_laps_l2013_201312


namespace NUMINAMATH_CALUDE_min_draws_for_eighteen_l2013_201329

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  purple : Nat

/-- The minimum number of balls needed to guarantee at least n of a single color -/
def minDraws (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The specific ball counts in our problem -/
def problemCounts : BallCounts :=
  { red := 34, green := 25, yellow := 18, blue := 21, purple := 13 }

/-- The theorem stating the minimum number of draws needed -/
theorem min_draws_for_eighteen (counts : BallCounts) :
  counts = problemCounts → minDraws counts 18 = 82 :=
  sorry

end NUMINAMATH_CALUDE_min_draws_for_eighteen_l2013_201329


namespace NUMINAMATH_CALUDE_james_profit_l2013_201304

/-- Calculates the profit from selling toys --/
def calculate_profit (initial_quantity : ℕ) (buy_price sell_price : ℚ) (sell_percentage : ℚ) : ℚ :=
  let total_cost := initial_quantity * buy_price
  let sold_quantity := (initial_quantity : ℚ) * sell_percentage
  let total_revenue := sold_quantity * sell_price
  total_revenue - total_cost

/-- Proves that James' profit is $800 --/
theorem james_profit :
  calculate_profit 200 20 30 (4/5) = 800 := by
  sorry

end NUMINAMATH_CALUDE_james_profit_l2013_201304


namespace NUMINAMATH_CALUDE_complete_square_sum_l2013_201365

theorem complete_square_sum (x : ℝ) : ∃ (a b c : ℤ), 
  (25 * x^2 + 30 * x - 75 = 0 ↔ (a * x + b)^2 = c) ∧ 
  a > 0 ∧ 
  a + b + c = -58 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l2013_201365


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l2013_201361

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.a + t.b - t.c) = 3 * t.a * t.b

def condition2 (t : Triangle) : Prop :=
  2 * Real.cos t.A * Real.sin t.B = Real.sin t.C

-- Theorem statement
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) : 
  t.a = t.b ∧ t.b = t.c ∧ t.A = t.B ∧ t.B = t.C ∧ t.C = Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l2013_201361


namespace NUMINAMATH_CALUDE_square_area_l2013_201377

/-- The parabola function -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 4

/-- The line function -/
def g (x : ℝ) : ℝ := 3

/-- The theorem stating the area of the square -/
theorem square_area : 
  ∃ (x₁ x₂ : ℝ), 
    f x₁ = g x₁ ∧ 
    f x₂ = g x₂ ∧ 
    x₁ ≠ x₂ ∧
    (x₂ - x₁)^2 = 8 :=
sorry

end NUMINAMATH_CALUDE_square_area_l2013_201377


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l2013_201376

/-- The perimeter of a semicircle with radius 10 is approximately 51.4 -/
theorem semicircle_perimeter_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |((10 : ℝ) * Real.pi + 20) - 51.4| < ε := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l2013_201376


namespace NUMINAMATH_CALUDE_maria_number_transformation_l2013_201340

theorem maria_number_transformation (x : ℚ) : 
  (2 * (x + 3) - 2) / 3 = 8 → x = 10 := by sorry

end NUMINAMATH_CALUDE_maria_number_transformation_l2013_201340


namespace NUMINAMATH_CALUDE_proportion_ones_is_42_233_l2013_201319

/-- The number of three-digit integers -/
def num_three_digit_ints : ℕ := 999 - 100 + 1

/-- The total number of digits in all three-digit integers -/
def total_digits : ℕ := num_three_digit_ints * 3

/-- The number of times each digit (1-9) appears in the three-digit integers -/
def digit_occurrences : ℕ := 100 + 90 + 90

/-- The number of times zero appears in the three-digit integers -/
def zero_occurrences : ℕ := 90 + 90

/-- The total number of digits after squaring -/
def total_squared_digits : ℕ := 
  (4 * digit_occurrences + zero_occurrences) + (6 * digit_occurrences * 2)

/-- The number of ones after squaring -/
def num_ones : ℕ := 3 * digit_occurrences

/-- The proportion of ones in the squared digits -/
def proportion_ones : ℚ := num_ones / total_squared_digits

theorem proportion_ones_is_42_233 : proportion_ones = 42 / 233 := by sorry

end NUMINAMATH_CALUDE_proportion_ones_is_42_233_l2013_201319


namespace NUMINAMATH_CALUDE_simple_random_sampling_is_most_appropriate_l2013_201357

/-- Represents a box containing units of a product -/
structure Box where
  name : String
  units : ℕ

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Cluster

/-- Determines if a sampling method is appropriate for the given boxes and sample size -/
def is_appropriate_sampling_method (boxes : List Box) (sample_size : ℕ) (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.SimpleRandom => true
  | _ => false

theorem simple_random_sampling_is_most_appropriate :
  let boxes : List Box := [
    { name := "large", units := 120 },
    { name := "medium", units := 60 },
    { name := "small", units := 20 }
  ]
  let sample_size : ℕ := 25
  ∀ method : SamplingMethod,
    is_appropriate_sampling_method boxes sample_size method →
    method = SamplingMethod.SimpleRandom :=
by
  sorry


end NUMINAMATH_CALUDE_simple_random_sampling_is_most_appropriate_l2013_201357


namespace NUMINAMATH_CALUDE_parallelogram_base_l2013_201379

/-- The base of a parallelogram with area 240 square cm and height 10 cm is 24 cm. -/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 240 ∧ height = 10 ∧ area = base * height → base = 24 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l2013_201379


namespace NUMINAMATH_CALUDE_quadratic_form_coefficients_l2013_201356

theorem quadratic_form_coefficients :
  let f : ℝ → ℝ := λ x => 2 * x * (x - 1) - 3 * x
  ∃ a b c : ℝ, (∀ x, f x = a * x^2 + b * x + c) ∧ a = 2 ∧ b = -5 ∧ c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_coefficients_l2013_201356


namespace NUMINAMATH_CALUDE_unique_solution_g100_l2013_201392

-- Define g₀(x)
def g₀ (x : ℝ) : ℝ := 2 * x + |x - 50| - |x + 50|

-- Define gₙ(x) recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

-- Theorem statement
theorem unique_solution_g100 :
  ∃! x, g 100 x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_g100_l2013_201392


namespace NUMINAMATH_CALUDE_calculation_proof_l2013_201339

theorem calculation_proof : 
  let sin_30 : ℝ := 1/2
  let sqrt_2_gt_1 : 1 < Real.sqrt 2 := by sorry
  let power_zero (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry
  2 * sin_30 - |1 - Real.sqrt 2| + (π - 2022)^0 = 3 - Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_calculation_proof_l2013_201339


namespace NUMINAMATH_CALUDE_lucas_raspberry_candies_l2013_201351

-- Define the variables
def original_raspberry : ℕ := sorry
def original_lemon : ℕ := sorry

-- Define the conditions
axiom initial_ratio : original_raspberry = 3 * original_lemon
axiom after_giving_away : original_raspberry - 5 = 4 * (original_lemon - 5)

-- Theorem to prove
theorem lucas_raspberry_candies : original_raspberry = 45 := by
  sorry

end NUMINAMATH_CALUDE_lucas_raspberry_candies_l2013_201351


namespace NUMINAMATH_CALUDE_eight_people_arrangements_l2013_201354

/-- The number of ways to arrange n distinct objects in a line -/
def linearArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- There are 8! ways to arrange 8 people in a line -/
theorem eight_people_arrangements : linearArrangements 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_eight_people_arrangements_l2013_201354


namespace NUMINAMATH_CALUDE_total_days_1999_to_2005_l2013_201387

def is_leap_year (year : ℕ) : Bool :=
  year = 2000 || year = 2004

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def total_days (start_year end_year : ℕ) : ℕ :=
  (List.range (end_year - start_year + 1)).map (fun i => days_in_year (start_year + i))
    |>.sum

theorem total_days_1999_to_2005 :
  total_days 1999 2005 = 2557 := by
  sorry

end NUMINAMATH_CALUDE_total_days_1999_to_2005_l2013_201387


namespace NUMINAMATH_CALUDE_product_divisible_by_12_l2013_201308

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The probability that a single roll is not divisible by 3 -/
def prob_not_div_3 : ℚ := 5 / 8

/-- The probability that a single roll is divisible by 4 -/
def prob_div_4 : ℚ := 1 / 4

/-- The probability that the product of the rolls is divisible by 12 -/
def prob_div_12 : ℚ := 149 / 256

theorem product_divisible_by_12 :
  (1 - prob_not_div_3 ^ num_dice) *
  (1 - (1 - prob_div_4) ^ num_dice - num_dice * prob_div_4 * (1 - prob_div_4) ^ (num_dice - 1)) =
  prob_div_12 := by sorry

end NUMINAMATH_CALUDE_product_divisible_by_12_l2013_201308


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_real_iff_x_geq_2_l2013_201367

theorem sqrt_x_minus_2_real_iff_x_geq_2 (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_real_iff_x_geq_2_l2013_201367


namespace NUMINAMATH_CALUDE_apartment_office_sale_net_effect_l2013_201334

theorem apartment_office_sale_net_effect :
  ∀ (apartment_cost office_cost : ℝ),
  apartment_cost * (1 - 0.25) = 15000 →
  office_cost * (1 + 0.25) = 15000 →
  apartment_cost + office_cost - 2 * 15000 = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_apartment_office_sale_net_effect_l2013_201334


namespace NUMINAMATH_CALUDE_park_visitors_difference_l2013_201345

theorem park_visitors_difference (saturday_visitors : ℕ) (total_visitors : ℕ) 
    (h1 : saturday_visitors = 200)
    (h2 : total_visitors = 440) : 
  total_visitors - 2 * saturday_visitors = 40 := by
  sorry

end NUMINAMATH_CALUDE_park_visitors_difference_l2013_201345


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2013_201394

/-- The hyperbola and parabola share a common focus -/
structure SharedFocus (a b : ℝ) :=
  (a_pos : a > 0)
  (b_pos : b > 0)
  (hyperbola : ℝ → ℝ → Prop)
  (parabola : ℝ → ℝ → Prop)
  (focus : ℝ × ℝ)
  (hyperbola_eq : ∀ x y, hyperbola x y ↔ x^2 / a^2 - y^2 / b^2 = 1)
  (parabola_eq : ∀ x y, parabola x y ↔ y^2 = 8*x)
  (shared_focus : ∃ (x y : ℝ), hyperbola x y ∧ parabola x y)

/-- The intersection point P and its distance from the focus -/
structure IntersectionPoint (a b : ℝ) extends SharedFocus a b :=
  (P : ℝ × ℝ)
  (on_hyperbola : hyperbola P.1 P.2)
  (on_parabola : parabola P.1 P.2)
  (distance_PF : Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) = 5)

/-- The theorem statement -/
theorem hyperbola_asymptote 
  {a b : ℝ} (h : IntersectionPoint a b) :
  ∃ (k : ℝ), k = Real.sqrt 3 ∧ 
  (∀ x y, h.hyperbola x y → (x = k*y ∨ x = -k*y)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2013_201394


namespace NUMINAMATH_CALUDE_symmetrical_circle_l2013_201343

/-- Given a circle with equation x² + y² + 2x = 0, 
    its symmetrical circle with respect to the y-axis 
    has the equation x² + y² - 2x = 0 -/
theorem symmetrical_circle (x y : ℝ) : 
  (x^2 + y^2 + 2*x = 0) → 
  ∃ (x' y' : ℝ), (x'^2 + y'^2 - 2*x' = 0 ∧ 
                  x' = -x ∧ 
                  y' = y) :=
by sorry

end NUMINAMATH_CALUDE_symmetrical_circle_l2013_201343


namespace NUMINAMATH_CALUDE_trinomial_square_l2013_201388

theorem trinomial_square (a : ℚ) : 
  (∃ b : ℚ, ∀ x : ℚ, 9*x^2 + 21*x + a = (3*x + b)^2) → a = 49/4 := by
  sorry

end NUMINAMATH_CALUDE_trinomial_square_l2013_201388


namespace NUMINAMATH_CALUDE_triple_hash_70_approx_8_l2013_201374

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.4 * N + 2

-- State the theorem
theorem triple_hash_70_approx_8 : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |hash (hash (hash 70)) - 8| < ε :=
sorry

end NUMINAMATH_CALUDE_triple_hash_70_approx_8_l2013_201374


namespace NUMINAMATH_CALUDE_percentage_seven_plus_years_l2013_201333

/-- Represents the number of employees in each employment duration range --/
structure EmployeeDistribution :=
  (less_than_1_year : ℕ)
  (one_to_two_years : ℕ)
  (two_to_three_years : ℕ)
  (three_to_four_years : ℕ)
  (four_to_five_years : ℕ)
  (five_to_six_years : ℕ)
  (six_to_seven_years : ℕ)
  (seven_to_eight_years : ℕ)
  (eight_to_nine_years : ℕ)
  (nine_to_ten_years : ℕ)
  (ten_plus_years : ℕ)

/-- Calculates the total number of employees --/
def total_employees (d : EmployeeDistribution) : ℕ :=
  d.less_than_1_year + d.one_to_two_years + d.two_to_three_years + d.three_to_four_years +
  d.four_to_five_years + d.five_to_six_years + d.six_to_seven_years + d.seven_to_eight_years +
  d.eight_to_nine_years + d.nine_to_ten_years + d.ten_plus_years

/-- Calculates the number of employees employed for 7 years or more --/
def employees_seven_plus_years (d : EmployeeDistribution) : ℕ :=
  d.seven_to_eight_years + d.eight_to_nine_years + d.nine_to_ten_years + d.ten_plus_years

/-- Theorem stating that the percentage of employees employed for 7 years or more is 21.43% --/
theorem percentage_seven_plus_years (d : EmployeeDistribution) 
  (h : d = {
    less_than_1_year := 4,
    one_to_two_years := 6,
    two_to_three_years := 5,
    three_to_four_years := 2,
    four_to_five_years := 3,
    five_to_six_years := 1,
    six_to_seven_years := 1,
    seven_to_eight_years := 2,
    eight_to_nine_years := 2,
    nine_to_ten_years := 1,
    ten_plus_years := 1
  }) :
  (employees_seven_plus_years d : ℚ) / (total_employees d : ℚ) * 100 = 21.43 := by
  sorry

end NUMINAMATH_CALUDE_percentage_seven_plus_years_l2013_201333


namespace NUMINAMATH_CALUDE_polynomial_equality_l2013_201315

theorem polynomial_equality : 105^5 - 5 * 105^4 + 10 * 105^3 - 10 * 105^2 + 5 * 105 - 1 = 11714628224 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2013_201315


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l2013_201335

/-- The coordinates of a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Given a point P, returns its symmetric point with respect to the origin -/
def symmetricPoint (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

/-- Theorem: The symmetric point of P(1, 3, -5) with respect to the origin is (-1, -3, 5) -/
theorem symmetric_point_theorem :
  let P : Point3D := { x := 1, y := 3, z := -5 }
  symmetricPoint P = { x := -1, y := -3, z := 5 } := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_theorem_l2013_201335


namespace NUMINAMATH_CALUDE_hoseok_persimmons_l2013_201353

theorem hoseok_persimmons (jungkook_persimmons hoseok_persimmons : ℕ) : 
  jungkook_persimmons = 25 → 
  3 * hoseok_persimmons = jungkook_persimmons - 4 →
  hoseok_persimmons = 7 := by
sorry

end NUMINAMATH_CALUDE_hoseok_persimmons_l2013_201353


namespace NUMINAMATH_CALUDE_diamond_calculation_l2013_201347

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- State the theorem
theorem diamond_calculation :
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29/132 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l2013_201347


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2013_201398

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  second_quadrant (-2 : ℝ) (3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2013_201398


namespace NUMINAMATH_CALUDE_complement_union_problem_l2013_201338

def U : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {1, 3, 5}
def B : Finset Nat := {2, 3}

theorem complement_union_problem : 
  (U \ A) ∪ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_problem_l2013_201338


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2013_201311

/-- Proves that the speed of a boat in still water is 16 km/hr given specific downstream conditions. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_time = 6)
  (h3 : downstream_distance = 126) :
  downstream_distance = (boat_speed + stream_speed) * downstream_time → 
  boat_speed = 16 :=
by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2013_201311


namespace NUMINAMATH_CALUDE_gcd_6273_14593_l2013_201301

theorem gcd_6273_14593 : Nat.gcd 6273 14593 = 3 := by sorry

end NUMINAMATH_CALUDE_gcd_6273_14593_l2013_201301


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l2013_201360

theorem unique_solution_floor_equation :
  ∃! x : ℝ, x + ⌊x⌋ = 20.2 ∧ x = 10.2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l2013_201360


namespace NUMINAMATH_CALUDE_profit_increase_l2013_201378

theorem profit_increase (m : ℝ) : 
  (m + 8) / 0.92 = m + 10 → m = 15 := by
  sorry

end NUMINAMATH_CALUDE_profit_increase_l2013_201378


namespace NUMINAMATH_CALUDE_community_A_sample_l2013_201372

/-- Represents the number of low-income households in a community -/
structure Community where
  households : ℕ

/-- Represents the total number of affordable housing units -/
def housing_units : ℕ := 90

/-- Calculates the number of households to be sampled from a community using stratified sampling -/
def stratified_sample (community : Community) (total_households : ℕ) : ℕ :=
  (community.households * housing_units) / total_households

/-- Theorem: The number of low-income households to be sampled from community A is 40 -/
theorem community_A_sample :
  let community_A : Community := ⟨360⟩
  let community_B : Community := ⟨270⟩
  let community_C : Community := ⟨180⟩
  let total_households := community_A.households + community_B.households + community_C.households
  stratified_sample community_A total_households = 40 := by
  sorry

end NUMINAMATH_CALUDE_community_A_sample_l2013_201372


namespace NUMINAMATH_CALUDE_swimming_time_against_current_l2013_201337

theorem swimming_time_against_current 
  (swimming_speed : ℝ) 
  (water_speed : ℝ) 
  (time_with_current : ℝ) 
  (h1 : swimming_speed = 4) 
  (h2 : water_speed = 2) 
  (h3 : time_with_current = 4) : 
  (swimming_speed + water_speed) * time_with_current / (swimming_speed - water_speed) = 12 := by
  sorry

end NUMINAMATH_CALUDE_swimming_time_against_current_l2013_201337


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2013_201366

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 + m - 2021 = 0) → (n^2 + n - 2021 = 0) → (m^2 + 2*m + n = 2020) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2013_201366


namespace NUMINAMATH_CALUDE_set_intersection_equality_l2013_201346

theorem set_intersection_equality (S T : Set ℝ) : 
  S = {y | ∃ x, y = (3 : ℝ) ^ x} →
  T = {y | ∃ x, y = x^2 + 1} →
  S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l2013_201346


namespace NUMINAMATH_CALUDE_select_two_from_four_l2013_201397

theorem select_two_from_four : Nat.choose 4 2 = 6 := by sorry

end NUMINAMATH_CALUDE_select_two_from_four_l2013_201397


namespace NUMINAMATH_CALUDE_bet_winnings_ratio_l2013_201324

def initial_amount : ℕ := 400
def final_amount : ℕ := 1200

def amount_won : ℕ := final_amount - initial_amount

theorem bet_winnings_ratio :
  (amount_won : ℚ) / initial_amount = 2 := by sorry

end NUMINAMATH_CALUDE_bet_winnings_ratio_l2013_201324


namespace NUMINAMATH_CALUDE_larger_number_proof_l2013_201380

theorem larger_number_proof (A B : ℕ+) : 
  (Nat.gcd A B = 20) → 
  (∃ (x : ℕ+), Nat.lcm A B = 20 * 11 * 15 * x) → 
  (A ≤ B) →
  B = 300 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2013_201380


namespace NUMINAMATH_CALUDE_linear_equation_property_l2013_201328

theorem linear_equation_property (x y : ℝ) (h : x + 6 * y = 17) :
  7 * x + 42 * y = 119 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_property_l2013_201328


namespace NUMINAMATH_CALUDE_length_MN_l2013_201399

-- Define the points on the line
variable (A B C D M N : ℝ)

-- Define the conditions
axiom order : A < B ∧ B < C ∧ C < D
axiom midpoint_M : M = (A + C) / 2
axiom midpoint_N : N = (B + D) / 2
axiom length_AD : D - A = 68
axiom length_BC : C - B = 20

-- Theorem to prove
theorem length_MN : N - M = 24 := by sorry

end NUMINAMATH_CALUDE_length_MN_l2013_201399


namespace NUMINAMATH_CALUDE_nellie_gift_wrap_sales_l2013_201369

/-- Given that Nellie needs to sell 45 rolls of gift wrap in total and has already sold some, 
    prove that she needs to sell 28 more rolls. -/
theorem nellie_gift_wrap_sales (total_needed : ℕ) (sold_to_grandmother : ℕ) (sold_to_uncle : ℕ) (sold_to_neighbor : ℕ) 
    (h1 : total_needed = 45)
    (h2 : sold_to_grandmother = 1)
    (h3 : sold_to_uncle = 10)
    (h4 : sold_to_neighbor = 6) :
    total_needed - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 28 := by
  sorry

end NUMINAMATH_CALUDE_nellie_gift_wrap_sales_l2013_201369


namespace NUMINAMATH_CALUDE_ac_over_bd_equals_15_l2013_201358

theorem ac_over_bd_equals_15 
  (a b c d : ℝ) 
  (h1 : a = 3 * b) 
  (h2 : b = 2 * c) 
  (h3 : c = 5 * d) 
  (h4 : d ≠ 0) : 
  (a * c) / (b * d) = 15 := by
sorry

end NUMINAMATH_CALUDE_ac_over_bd_equals_15_l2013_201358


namespace NUMINAMATH_CALUDE_max_root_sum_l2013_201327

theorem max_root_sum (a b c : ℝ) : 
  (a^3 - 4 * Real.sqrt 3 * a^2 + 13 * a - 2 * Real.sqrt 3 = 0) →
  (b^3 - 4 * Real.sqrt 3 * b^2 + 13 * b - 2 * Real.sqrt 3 = 0) →
  (c^3 - 4 * Real.sqrt 3 * c^2 + 13 * c - 2 * Real.sqrt 3 = 0) →
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  max (a + b - c) (max (a - b + c) (-a + b + c)) = 2 * Real.sqrt 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_root_sum_l2013_201327


namespace NUMINAMATH_CALUDE_baseball_league_games_l2013_201370

theorem baseball_league_games (P Q : ℕ) : 
  P > 2 * Q →
  Q > 6 →
  4 * P + 5 * Q = 82 →
  4 * P = 52 := by
sorry

end NUMINAMATH_CALUDE_baseball_league_games_l2013_201370


namespace NUMINAMATH_CALUDE_teaching_arrangements_count_l2013_201384

def number_of_teachers : ℕ := 3
def number_of_classes : ℕ := 6
def classes_per_teacher : ℕ := 2

theorem teaching_arrangements_count :
  (Nat.choose number_of_classes classes_per_teacher) *
  (Nat.choose (number_of_classes - classes_per_teacher) classes_per_teacher) *
  (Nat.choose (number_of_classes - 2 * classes_per_teacher) classes_per_teacher) = 90 := by
  sorry

end NUMINAMATH_CALUDE_teaching_arrangements_count_l2013_201384


namespace NUMINAMATH_CALUDE_preston_received_correct_amount_l2013_201362

/-- Calculates the total amount Preston received from Abra Company's order --/
def prestonReceived (
  sandwichPrice : ℚ)
  (sideDishPrice : ℚ)
  (drinkPrice : ℚ)
  (deliveryFee : ℚ)
  (sandwichCount : ℕ)
  (sideDishCount : ℕ)
  (drinkCount : ℕ)
  (tipPercentage : ℚ)
  (discountPercentage : ℚ) : ℚ :=
  let foodCost := sandwichPrice * sandwichCount + sideDishPrice * sideDishCount
  let drinkCost := drinkPrice * drinkCount
  let discountAmount := discountPercentage * foodCost
  let subtotal := foodCost + drinkCost - discountAmount + deliveryFee
  let tipAmount := tipPercentage * subtotal
  subtotal + tipAmount

/-- Theorem stating that Preston received $158.95 from Abra Company's order --/
theorem preston_received_correct_amount :
  prestonReceived 5 3 (3/2) 20 18 10 15 (1/10) (15/100) = 15895/100 := by
  sorry

end NUMINAMATH_CALUDE_preston_received_correct_amount_l2013_201362


namespace NUMINAMATH_CALUDE_team_a_win_probability_l2013_201310

/-- The probability of Team A winning a single game -/
def p_win : ℚ := 3/5

/-- The probability of Team A losing a single game -/
def p_lose : ℚ := 2/5

/-- The number of ways to choose 2 wins out of 3 games -/
def combinations : ℕ := 3

theorem team_a_win_probability :
  combinations * p_win^3 * p_lose = 162/625 := by
  sorry

end NUMINAMATH_CALUDE_team_a_win_probability_l2013_201310


namespace NUMINAMATH_CALUDE_staplers_left_after_stapling_l2013_201313

/-- The number of staplers left after stapling some reports -/
def staplers_left (initial_staplers : ℕ) (dozen_reports_stapled : ℕ) : ℕ :=
  initial_staplers - dozen_reports_stapled * 12

/-- Theorem: Given 50 initial staplers and 3 dozen reports stapled, 14 staplers are left -/
theorem staplers_left_after_stapling :
  staplers_left 50 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_staplers_left_after_stapling_l2013_201313


namespace NUMINAMATH_CALUDE_min_value_ab_l2013_201373

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + 4 * b + 5) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = x + 4 * y + 5 → a * b ≤ x * y ∧ a * b = 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_l2013_201373


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2013_201331

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0) (h2 : arithmetic_sequence a d)
  (h3 : a 2021 = a 20 + a 21) :
  a 1 / d = 1981 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2013_201331


namespace NUMINAMATH_CALUDE_maria_towels_problem_l2013_201348

theorem maria_towels_problem (green_towels white_towels given_to_mother : ℝ) 
  (h1 : green_towels = 124.5)
  (h2 : white_towels = 67.7)
  (h3 : given_to_mother = 85.35) :
  green_towels + white_towels - given_to_mother = 106.85 := by
sorry

end NUMINAMATH_CALUDE_maria_towels_problem_l2013_201348


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l2013_201321

theorem quadratic_form_ratio (x : ℝ) :
  let f := x^2 + 2600*x + 2600
  ∃ d e : ℝ, (∀ x, f = (x + d)^2 + e) ∧ e / d = -1298 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l2013_201321


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2013_201314

theorem sqrt_fraction_simplification : 
  (Real.sqrt 3) / ((Real.sqrt 3) + (Real.sqrt 12)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2013_201314


namespace NUMINAMATH_CALUDE_sum_x_y_equals_three_l2013_201371

/-- Given a system of linear equations, prove that x + y = 3 -/
theorem sum_x_y_equals_three (x y : ℝ) 
  (eq1 : 2 * x + y = 5) 
  (eq2 : x + 2 * y = 4) : 
  x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_three_l2013_201371


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2013_201320

theorem sphere_volume_equals_surface_area (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3 = 4 * Real.pi * r^2) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2013_201320


namespace NUMINAMATH_CALUDE_smallest_norwegian_l2013_201305

def is_norwegian (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ a + b + c = 2022

theorem smallest_norwegian : ∀ n : ℕ, is_norwegian n → n ≥ 1344 :=
sorry

end NUMINAMATH_CALUDE_smallest_norwegian_l2013_201305


namespace NUMINAMATH_CALUDE_orange_count_indeterminate_l2013_201352

/-- Represents Philip's fruit collection -/
structure FruitCollection where
  banana_count : ℕ
  banana_groups : ℕ
  bananas_per_group : ℕ
  orange_groups : ℕ

/-- Predicate to check if the banana count is consistent with the groups and bananas per group -/
def banana_count_consistent (collection : FruitCollection) : Prop :=
  collection.banana_count = collection.banana_groups * collection.bananas_per_group

/-- Theorem stating that the number of oranges cannot be determined -/
theorem orange_count_indeterminate (collection : FruitCollection)
  (h1 : collection.banana_count = 290)
  (h2 : collection.banana_groups = 2)
  (h3 : collection.bananas_per_group = 145)
  (h4 : collection.orange_groups = 93)
  (h5 : banana_count_consistent collection) :
  ¬∃ (orange_count : ℕ), ∀ (other_collection : FruitCollection),
    collection.banana_count = other_collection.banana_count ∧
    collection.banana_groups = other_collection.banana_groups ∧
    collection.bananas_per_group = other_collection.bananas_per_group ∧
    collection.orange_groups = other_collection.orange_groups →
    orange_count = (other_collection.orange_groups : ℕ) * (orange_count / other_collection.orange_groups) :=
sorry

end NUMINAMATH_CALUDE_orange_count_indeterminate_l2013_201352


namespace NUMINAMATH_CALUDE_tshirt_packages_l2013_201350

theorem tshirt_packages (total_tshirts : ℕ) (tshirts_per_package : ℕ) 
  (h1 : total_tshirts = 56)
  (h2 : tshirts_per_package = 2) :
  total_tshirts / tshirts_per_package = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_tshirt_packages_l2013_201350


namespace NUMINAMATH_CALUDE_modulus_of_z_l2013_201300

theorem modulus_of_z (z : ℂ) (h : z * (4 - 3*I) = 1) : Complex.abs z = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2013_201300


namespace NUMINAMATH_CALUDE_factoring_expression_l2013_201326

theorem factoring_expression (y : ℝ) : 5 * y * (y + 2) + 9 * (y + 2) = (5 * y + 9) * (y + 2) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l2013_201326


namespace NUMINAMATH_CALUDE_greatest_possible_median_l2013_201341

theorem greatest_possible_median (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 18 →
  k < m → m < r → r < s → s < t →
  t = 40 →
  r ≤ 23 ∧ ∃ (k' m' r' s' : ℕ), 
    k' > 0 ∧ m' > 0 ∧ r' > 0 ∧ s' > 0 ∧
    (k' + m' + r' + s' + 40) / 5 = 18 ∧
    k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < 40 ∧
    r' = 23 := by
  sorry

end NUMINAMATH_CALUDE_greatest_possible_median_l2013_201341


namespace NUMINAMATH_CALUDE_expression_evaluation_l2013_201363

theorem expression_evaluation : (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2013_201363


namespace NUMINAMATH_CALUDE_bees_flew_in_l2013_201323

/-- Given an initial number of bees in a hive and a total number of bees after more flew in,
    this theorem proves that the number of bees that flew in is equal to the difference
    between the total and initial number of bees. -/
theorem bees_flew_in (initial_bees total_bees : ℕ) 
    (h1 : initial_bees = 16) 
    (h2 : total_bees = 26) : 
  total_bees - initial_bees = 10 := by
  sorry

#check bees_flew_in

end NUMINAMATH_CALUDE_bees_flew_in_l2013_201323


namespace NUMINAMATH_CALUDE_dividend_calculation_l2013_201316

theorem dividend_calculation (quotient divisor remainder : ℚ) :
  quotient = 9.3 →
  divisor = 4/3 →
  remainder = 5 →
  (divisor * quotient) + remainder = 17.4 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2013_201316


namespace NUMINAMATH_CALUDE_square_sum_product_l2013_201393

theorem square_sum_product (x y : ℝ) 
  (h1 : (x + y)^2 = 49) 
  (h2 : x * y = 8) : 
  x^2 + y^2 + 3 * x * y = 57 := by
sorry

end NUMINAMATH_CALUDE_square_sum_product_l2013_201393


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_quadratic_roots_l2013_201344

theorem perpendicular_lines_from_quadratic_roots : 
  ∀ (k₁ k₂ : ℝ), 
    k₁^2 - 3*k₁ - 1 = 0 → 
    k₂^2 - 3*k₂ - 1 = 0 → 
    k₁ ≠ k₂ →
    k₁ * k₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_quadratic_roots_l2013_201344


namespace NUMINAMATH_CALUDE_henry_twice_jills_age_l2013_201322

theorem henry_twice_jills_age (henry_age jill_age : ℕ) 
  (henry_age_val : henry_age = 25)
  (jill_age_val : jill_age = 16)
  (sum_ages : henry_age + jill_age = 41) : 
  ∃ (years_ago : ℕ), 
    henry_age - years_ago = 2 * (jill_age - years_ago) ∧ 
    years_ago = 7 := by
  sorry

end NUMINAMATH_CALUDE_henry_twice_jills_age_l2013_201322


namespace NUMINAMATH_CALUDE_symmetric_points_sum_sum_of_symmetric_point_l2013_201325

/-- Given two points A and B in a 2D plane, where A is symmetric to B with respect to the origin,
    prove that the sum of B's coordinates equals the negative sum of A's coordinates. -/
theorem symmetric_points_sum (A B : ℝ × ℝ) (hSymmetric : B = (-A.1, -A.2)) :
  B.1 + B.2 = -(A.1 + A.2) := by
  sorry

/-- Prove that if point A(-2022, -1) is symmetric with respect to the origin O to point B(a, b),
    then a + b = 2023. -/
theorem sum_of_symmetric_point :
  ∃ (a b : ℝ), (a, b) = (-(-2022), -(-1)) → a + b = 2023 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_sum_of_symmetric_point_l2013_201325


namespace NUMINAMATH_CALUDE_absent_workers_l2013_201349

theorem absent_workers (total_workers : ℕ) (original_days : ℕ) (actual_days : ℕ) 
  (h1 : total_workers = 42)
  (h2 : original_days = 12)
  (h3 : actual_days = 14) :
  ∃ (absent : ℕ), 
    absent = 6 ∧ 
    (total_workers * original_days = (total_workers - absent) * actual_days) :=
by sorry

end NUMINAMATH_CALUDE_absent_workers_l2013_201349


namespace NUMINAMATH_CALUDE_watch_correction_theorem_l2013_201332

/-- The number of days between noon on March 1st and 6 P.M. on March 10th -/
def days_elapsed : ℝ := 9.25

/-- The rate at which the watch loses time, in minutes per day -/
def loss_rate : ℝ := 3

/-- The positive correction needed for the watch, in minutes -/
def correction (d : ℝ) (r : ℝ) : ℝ := d * r

theorem watch_correction_theorem :
  correction days_elapsed loss_rate = 27.75 := by
  sorry

end NUMINAMATH_CALUDE_watch_correction_theorem_l2013_201332


namespace NUMINAMATH_CALUDE_point_on_unit_circle_l2013_201302

/-- A point on the unit circle reached by moving counterclockwise from (1,0) along an arc length of 2π/3 has coordinates (-1/2, √3/2). -/
theorem point_on_unit_circle (Q : ℝ × ℝ) : 
  (Q.1^2 + Q.2^2 = 1) →  -- Q is on the unit circle
  (Real.cos (2 * Real.pi / 3) = Q.1 ∧ Real.sin (2 * Real.pi / 3) = Q.2) →  -- Q is reached by moving 2π/3 radians counterclockwise from (1,0)
  (Q.1 = -1/2 ∧ Q.2 = Real.sqrt 3 / 2) :=  -- Q has coordinates (-1/2, √3/2)
by sorry

end NUMINAMATH_CALUDE_point_on_unit_circle_l2013_201302


namespace NUMINAMATH_CALUDE_thirteen_divides_six_digit_reverse_perm_l2013_201359

/-- A 6-digit positive integer whose first three digits are a permutation of its last three digits taken in reverse order. -/
def SixDigitReversePerm : Type :=
  {n : ℕ // 100000 ≤ n ∧ n < 1000000 ∧ ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ a ≠ 0 ∧
    (n = 100000*a + 10000*b + 1000*c + 100*c + 10*b + a ∨
     n = 100000*a + 10000*c + 1000*b + 100*b + 10*c + a ∨
     n = 100000*b + 10000*a + 1000*c + 100*c + 10*a + b ∨
     n = 100000*b + 10000*c + 1000*a + 100*a + 10*c + b ∨
     n = 100000*c + 10000*a + 1000*b + 100*b + 10*a + c ∨
     n = 100000*c + 10000*b + 1000*a + 100*a + 10*b + c)}

theorem thirteen_divides_six_digit_reverse_perm (x : SixDigitReversePerm) :
  13 ∣ x.val :=
sorry

end NUMINAMATH_CALUDE_thirteen_divides_six_digit_reverse_perm_l2013_201359


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2013_201395

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) : ℂ).re = 0 ∧ (a ^ 2 - 3 * a + 2 : ℂ).re = 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2013_201395


namespace NUMINAMATH_CALUDE_cube_sum_eq_product_squares_l2013_201368

theorem cube_sum_eq_product_squares (x y z n : ℕ+) :
  x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 ↔ n = 1 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_eq_product_squares_l2013_201368


namespace NUMINAMATH_CALUDE_aliyah_vivienne_phone_difference_l2013_201396

theorem aliyah_vivienne_phone_difference :
  ∀ (aliyah_phones : ℕ) (vivienne_phones : ℕ),
    vivienne_phones = 40 →
    (aliyah_phones + vivienne_phones) * 400 = 36000 →
    aliyah_phones - vivienne_phones = 10 := by
  sorry

end NUMINAMATH_CALUDE_aliyah_vivienne_phone_difference_l2013_201396


namespace NUMINAMATH_CALUDE_jakes_weight_l2013_201317

theorem jakes_weight (jake_weight sister_weight : ℝ) 
  (h1 : jake_weight - 8 = 2 * sister_weight)
  (h2 : jake_weight + sister_weight = 278) : 
  jake_weight = 188 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l2013_201317


namespace NUMINAMATH_CALUDE_twenty_four_is_forty_eight_percent_of_fifty_l2013_201336

theorem twenty_four_is_forty_eight_percent_of_fifty :
  ∃ x : ℝ, (24 : ℝ) / x = 48 / 100 ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_twenty_four_is_forty_eight_percent_of_fifty_l2013_201336


namespace NUMINAMATH_CALUDE_negation_divisible_by_five_l2013_201318

theorem negation_divisible_by_five (n : ℕ) : 
  ¬(∀ n : ℕ, n % 5 = 0 → n % 10 = 0) ↔ 
  ∃ n : ℕ, n % 5 = 0 ∧ n % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_divisible_by_five_l2013_201318


namespace NUMINAMATH_CALUDE_smallest_number_proof_l2013_201391

/-- A function that checks if a natural number contains all digits from 0 to 9 exactly once -/
def has_all_digits_once (n : ℕ) : Prop := sorry

/-- A function that returns the smallest natural number that is a multiple of 36 and contains all digits from 0 to 9 exactly once -/
def smallest_number_with_all_digits_divisible_by_36 : ℕ := sorry

theorem smallest_number_proof :
  smallest_number_with_all_digits_divisible_by_36 = 1023457896 ∧
  has_all_digits_once smallest_number_with_all_digits_divisible_by_36 ∧
  smallest_number_with_all_digits_divisible_by_36 % 36 = 0 ∧
  ∀ m : ℕ, m < smallest_number_with_all_digits_divisible_by_36 →
    ¬(has_all_digits_once m ∧ m % 36 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l2013_201391


namespace NUMINAMATH_CALUDE_sotka_not_divisible_by_nine_l2013_201385

/-- Represents a digit in the range 0 to 9 -/
def Digit := Fin 10

/-- Represents the mapping of letters to digits -/
def LetterToDigit := Char → Digit

/-- Checks if all characters in a string are mapped to unique digits -/
def allUnique (s : String) (m : LetterToDigit) : Prop :=
  ∀ c₁ c₂, c₁ ∈ s.data → c₂ ∈ s.data → c₁ ≠ c₂ → m c₁ ≠ m c₂

/-- Converts a string to a number using the given mapping -/
def toNumber (s : String) (m : LetterToDigit) : ℕ :=
  s.data.foldr (λ c acc => acc * 10 + (m c).val) 0

/-- The main theorem -/
theorem sotka_not_divisible_by_nine (m : LetterToDigit) : 
  allUnique "ДЕВЯНОСТО" m →
  allUnique "ДЕВЯТКА" m →
  allUnique "СОТКА" m →
  90 ∣ toNumber "ДЕВЯНОСТО" m →
  9 ∣ toNumber "ДЕВЯТКА" m →
  ¬(9 ∣ toNumber "СОТКА" m) := by
  sorry


end NUMINAMATH_CALUDE_sotka_not_divisible_by_nine_l2013_201385


namespace NUMINAMATH_CALUDE_sams_initial_dimes_l2013_201389

/-- The problem of determining Sam's initial number of dimes -/
theorem sams_initial_dimes :
  ∀ (initial_dimes current_dimes : ℕ),
    initial_dimes - 4 = current_dimes →
    current_dimes = 4 →
    initial_dimes = 8 := by
  sorry

end NUMINAMATH_CALUDE_sams_initial_dimes_l2013_201389


namespace NUMINAMATH_CALUDE_function_symmetric_about_two_lines_is_periodic_l2013_201330

/-- Given a function f: ℝ → ℝ that is symmetric about x = a and x = b (where a ≠ b),
    prove that f is periodic with period 2b - 2a. -/
theorem function_symmetric_about_two_lines_is_periodic
  (f : ℝ → ℝ) (a b : ℝ) (h_neq : a ≠ b)
  (h_sym_a : ∀ x, f (a - x) = f (a + x))
  (h_sym_b : ∀ x, f (b - x) = f (b + x)) :
  ∀ x, f x = f (x + 2*b - 2*a) :=
sorry

end NUMINAMATH_CALUDE_function_symmetric_about_two_lines_is_periodic_l2013_201330


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l2013_201309

/-- Given that the terminal side of angle α intersects the unit circle 
    at point P(-4/5, 3/5), prove that cos 2α = 7/25 -/
theorem cos_double_angle_special_case (α : Real) 
  (h : ∃ (P : Real × Real), P.1 = -4/5 ∧ P.2 = 3/5 ∧ P.1^2 + P.2^2 = 1 ∧ 
       P.1 = Real.cos α ∧ P.2 = Real.sin α) : 
  Real.cos (2 * α) = 7/25 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l2013_201309


namespace NUMINAMATH_CALUDE_milk_replacement_problem_l2013_201307

/-- Given a container initially full of milk, prove that if x liters are drawn out and 
    replaced with water twice, resulting in a milk to water ratio of 9:16 in a 
    total mixture of 15 liters, then x must equal 12 liters. -/
theorem milk_replacement_problem (x : ℝ) : 
  x > 0 →
  (15 - x) - x * ((15 - x) / 15) = (9 / 25) * 15 →
  x = 12 := by
  sorry

end NUMINAMATH_CALUDE_milk_replacement_problem_l2013_201307


namespace NUMINAMATH_CALUDE_flowers_in_vase_l2013_201306

/-- Given that Lara bought 52 stems of flowers, gave 15 to her mom, and gave 6 more to her grandma
    than to her mom, prove that she put 16 stems in the vase. -/
theorem flowers_in_vase (total : ℕ) (to_mom : ℕ) (extra_to_grandma : ℕ)
    (h1 : total = 52)
    (h2 : to_mom = 15)
    (h3 : extra_to_grandma = 6)
    : total - (to_mom + (to_mom + extra_to_grandma)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_flowers_in_vase_l2013_201306


namespace NUMINAMATH_CALUDE_minas_age_l2013_201383

/-- Given the ages of Minho, Suhong, and Mina, prove that Mina is 10 years old -/
theorem minas_age (suhong minho mina : ℕ) : 
  minho = 3 * suhong →  -- Minho's age is three times Suhong's age
  mina = 2 * suhong - 2 →  -- Mina's age is two years younger than twice Suhong's age
  suhong + minho + mina = 34 →  -- The sum of the ages of the three is 34
  mina = 10 := by
sorry


end NUMINAMATH_CALUDE_minas_age_l2013_201383


namespace NUMINAMATH_CALUDE_compound_interest_problem_l2013_201390

/-- Given a principal amount P, where the simple interest on P for 2 years at 10% per annum is $660,
    prove that the compound interest on P for 2 years at the same rate is $693. -/
theorem compound_interest_problem (P : ℝ) : 
  P * 0.1 * 2 = 660 → P * (1 + 0.1)^2 - P = 693 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l2013_201390


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2013_201381

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 5}

theorem complement_of_A_in_U :
  (U \ A) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2013_201381


namespace NUMINAMATH_CALUDE_point_on_line_k_l2013_201355

/-- A line passing through the origin with slope 1/5 -/
def line_k (x y : ℝ) : Prop := y = (1/5) * x

theorem point_on_line_k (x y : ℝ) :
  line_k x 1 →
  line_k 5 y →
  y = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_k_l2013_201355


namespace NUMINAMATH_CALUDE_no_common_points_l2013_201342

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem stating that there are no common points
theorem no_common_points : ¬ ∃ (x y : ℝ), circle1 x y ∧ circle2 x y :=
sorry

end NUMINAMATH_CALUDE_no_common_points_l2013_201342


namespace NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l2013_201382

theorem and_sufficient_not_necessary_for_or :
  (∃ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l2013_201382


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l2013_201386

theorem quadratic_equation_m_value : ∃! m : ℤ, |m| = 2 ∧ m + 2 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l2013_201386


namespace NUMINAMATH_CALUDE_path_count_equals_binomial_coefficient_l2013_201364

/-- The number of paths composed of n rises and n descents of the same amplitude -/
def pathCount (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- Theorem: The number of paths composed of n rises and n descents of the same amplitude
    is equal to the binomial coefficient (2n choose n) -/
theorem path_count_equals_binomial_coefficient (n : ℕ) :
  pathCount n = Nat.choose (2 * n) n := by sorry

end NUMINAMATH_CALUDE_path_count_equals_binomial_coefficient_l2013_201364


namespace NUMINAMATH_CALUDE_min_value_problem_l2013_201375

theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  (1 / a + 1 / b + 1 / c ≥ 9) ∧ (1 / (3 * a + 2) + 1 / (3 * b + 2) + 1 / (3 * c + 2) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2013_201375


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_f_geq_three_halves_l2013_201303

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |a*x + 1| + |x - a|
def g (x : ℝ) : ℝ := x^2 + x

-- State the theorems
theorem solution_set_when_a_is_one :
  ∀ x : ℝ, g x ≥ f 1 x ↔ x ≤ -3 ∨ x ≥ 1 := by sorry

theorem range_of_a_for_f_geq_three_halves :
  (∀ x : ℝ, f a x ≥ 3/2) → a ≥ Real.sqrt 2 / 2 := by sorry

-- Note: We assume 'a' is positive as given in the original problem
variable (a : ℝ) (ha : a > 0)

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_f_geq_three_halves_l2013_201303
