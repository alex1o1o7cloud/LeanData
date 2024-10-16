import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_for_non_negative_f_l569_56904

/-- The range of a for which f(x) = x³ - x² - 2a has a non-negative value in (-∞, a] -/
theorem range_of_a_for_non_negative_f (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ≤ a ∧ x₀^3 - x₀^2 - 2*a ≥ 0) ↔ a ∈ Set.Icc (-1) 0 ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_non_negative_f_l569_56904


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l569_56960

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - 2*x - 3 < 0) → (-2 < x ∧ x < 3) ∧
  ∃ y : ℝ, -2 < y ∧ y < 3 ∧ ¬(y^2 - 2*y - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l569_56960


namespace NUMINAMATH_CALUDE_factorization_proof_l569_56901

theorem factorization_proof (a x y : ℝ) : 2*a*(x-y) - (x-y) = (x-y)*(2*a-1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l569_56901


namespace NUMINAMATH_CALUDE_closer_to_cottage_l569_56985

theorem closer_to_cottage (c m p : ℝ) 
  (hc : c > 0)
  (hm : m + 3/2 * (1/2 * m) = c)
  (hp : 2*p + 1/3 * (2*p) = c) : 
  m/c > p/c := by
sorry

end NUMINAMATH_CALUDE_closer_to_cottage_l569_56985


namespace NUMINAMATH_CALUDE_max_integer_solution_inequality_system_l569_56959

theorem max_integer_solution_inequality_system :
  ∀ x : ℤ, (3 * x - 1 < x + 1 ∧ 2 * (2 * x - 1) ≤ 5 * x + 1) →
  x ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_solution_inequality_system_l569_56959


namespace NUMINAMATH_CALUDE_money_redistribution_theorem_l569_56993

theorem money_redistribution_theorem (a b j t : ℝ) : 
  t = 48 ∧ 
  a ≠ b ∧ a ≠ j ∧ a ≠ t ∧ b ≠ j ∧ b ≠ t ∧ j ≠ t ∧
  384 - (8 * (a - (b + j + t)) + 8 * b + 8 * j) = 48 →
  a + b + j + t = 720 := by
  sorry

end NUMINAMATH_CALUDE_money_redistribution_theorem_l569_56993


namespace NUMINAMATH_CALUDE_company_earnings_difference_l569_56978

/-- Represents a company selling bottled milk -/
structure Company where
  price : ℝ  -- Price of a big bottle
  sold : ℕ   -- Number of big bottles sold

/-- Calculates the earnings of a company -/
def earnings (c : Company) : ℝ := c.price * c.sold

/-- The problem statement -/
theorem company_earnings_difference 
  (company_a company_b : Company)
  (ha : company_a.price = 4)
  (hb : company_b.price = 3.5)
  (sa : company_a.sold = 300)
  (sb : company_b.sold = 350) :
  earnings company_b - earnings company_a = 25 := by
  sorry

end NUMINAMATH_CALUDE_company_earnings_difference_l569_56978


namespace NUMINAMATH_CALUDE_multiply_difference_of_cubes_l569_56902

theorem multiply_difference_of_cubes (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_difference_of_cubes_l569_56902


namespace NUMINAMATH_CALUDE_no_natural_solution_l569_56912

theorem no_natural_solution :
  ¬ ∃ (x y : ℕ), x^4 - y^4 = x^3 + y^3 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l569_56912


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l569_56917

theorem complex_fraction_equality : Complex.I ^ 2 + Complex.I ^ 3 + Complex.I ^ 4 = (1 / 2 - Complex.I / 2) * (1 - Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l569_56917


namespace NUMINAMATH_CALUDE_circle_area_decrease_l569_56947

theorem circle_area_decrease (r : ℝ) (hr : r > 0) :
  let original_area := π * r^2
  let new_radius := r / 2
  let new_area := π * new_radius^2
  (original_area - new_area) / original_area = 3/4 := by
sorry

end NUMINAMATH_CALUDE_circle_area_decrease_l569_56947


namespace NUMINAMATH_CALUDE_min_value_expression_l569_56982

open Real

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min : ℝ), min = Real.sqrt 39 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 →
    (|6*x - 4*y| + |3*(x + y*Real.sqrt 3) + 2*(x*Real.sqrt 3 - y)|) / Real.sqrt (x^2 + y^2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l569_56982


namespace NUMINAMATH_CALUDE_f_11_equals_149_l569_56990

def f (n : ℕ) : ℕ := n^2 + n + 17

theorem f_11_equals_149 : f 11 = 149 := by sorry

end NUMINAMATH_CALUDE_f_11_equals_149_l569_56990


namespace NUMINAMATH_CALUDE_linear_function_increasing_l569_56979

/-- Given a linear function f(x) = 2x - 1, prove that for any two points
    (x₁, y₁) and (x₂, y₂) on its graph, if x₁ > x₂, then y₁ > y₂ -/
theorem linear_function_increasing (x₁ x₂ y₁ y₂ : ℝ) 
    (h1 : y₁ = 2 * x₁ - 1)
    (h2 : y₂ = 2 * x₂ - 1)
    (h3 : x₁ > x₂) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_linear_function_increasing_l569_56979


namespace NUMINAMATH_CALUDE_jan_roses_cost_l569_56966

theorem jan_roses_cost : 
  let dozen : ℕ := 12
  let roses_bought : ℕ := 5 * dozen
  let cost_per_rose : ℕ := 6
  let discount_rate : ℚ := 4/5
  (roses_bought * cost_per_rose : ℚ) * discount_rate = 288 := by
  sorry

end NUMINAMATH_CALUDE_jan_roses_cost_l569_56966


namespace NUMINAMATH_CALUDE_shirt_price_reduction_l569_56958

theorem shirt_price_reduction (original_price : ℝ) (first_reduction_percent : ℝ) (second_reduction_percent : ℝ) : 
  original_price = 20 →
  first_reduction_percent = 20 →
  second_reduction_percent = 40 →
  (1 - second_reduction_percent / 100) * ((1 - first_reduction_percent / 100) * original_price) = 9.60 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_reduction_l569_56958


namespace NUMINAMATH_CALUDE_map_scale_conversion_l569_56944

/-- Given a map scale where 15 cm represents 90 km, prove that 25 cm represents 150 km -/
theorem map_scale_conversion (scale_cm : ℝ) (scale_km : ℝ) (distance_cm : ℝ) :
  scale_cm = 15 ∧ scale_km = 90 ∧ distance_cm = 25 →
  (distance_cm / scale_cm) * scale_km = 150 := by
sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l569_56944


namespace NUMINAMATH_CALUDE_arithmetic_combination_equals_24_l569_56991

theorem arithmetic_combination_equals_24 : ∃ (expr : ℝ → ℝ → ℝ → ℝ → ℝ), 
  (expr 5 7 8 8 = 24) ∧ 
  (∀ a b c d, expr a b c d = ((b + c) / a) * d ∨ expr a b c d = ((b - a) * c) + d) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_combination_equals_24_l569_56991


namespace NUMINAMATH_CALUDE_odd_product_probability_l569_56927

theorem odd_product_probability (n : ℕ) (h : n = 2020) :
  let total := n
  let odds := n / 2
  let p := (odds / total) * ((odds - 1) / (total - 1)) * ((odds - 2) / (total - 2))
  p < 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_odd_product_probability_l569_56927


namespace NUMINAMATH_CALUDE_total_time_knife_and_vegetables_l569_56984

/-- Proves that the total time spent on knife sharpening and vegetable peeling is 40 minutes -/
theorem total_time_knife_and_vegetables (knife_time vegetable_time total_time : ℕ) : 
  knife_time = 10 →
  vegetable_time = 3 * knife_time →
  total_time = knife_time + vegetable_time →
  total_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_time_knife_and_vegetables_l569_56984


namespace NUMINAMATH_CALUDE_parabola_symmetry_condition_l569_56948

theorem parabola_symmetry_condition (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    y₁ = a * x₁^2 - 1 ∧ 
    y₂ = a * x₂^2 - 1 ∧ 
    x₁ + y₁ = -(x₂ + y₂) ∧ 
    x₁ ≠ x₂) → 
  a > 3/4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_condition_l569_56948


namespace NUMINAMATH_CALUDE_minimum_coins_l569_56908

def nickel : ℚ := 5 / 100
def dime : ℚ := 10 / 100
def quarter : ℚ := 25 / 100
def half_dollar : ℚ := 50 / 100

def total_amount : ℚ := 3

theorem minimum_coins (n d q h : ℕ) : 
  n ≥ 1 → d ≥ 1 → q ≥ 1 → h ≥ 1 →
  n * nickel + d * dime + q * quarter + h * half_dollar = total_amount →
  n + d + q + h ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_minimum_coins_l569_56908


namespace NUMINAMATH_CALUDE_concept_laws_theorem_l569_56916

/-- The probability that exactly M laws are included in the Concept -/
def prob_M_laws_included (K N M : ℕ) (p : ℝ) : ℝ :=
  Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)

/-- The expected number of laws included in the Concept -/
def expected_laws_included (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

theorem concept_laws_theorem (K N M : ℕ) (p : ℝ) 
  (hK : K > 0) (hN : N > 0) (hM : M ≤ K) (hp : 0 ≤ p ∧ p ≤ 1) :
  (prob_M_laws_included K N M p = Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)) ∧
  (expected_laws_included K N p = K * (1 - (1 - p)^N)) := by
  sorry

end NUMINAMATH_CALUDE_concept_laws_theorem_l569_56916


namespace NUMINAMATH_CALUDE_volume_of_R_revolution_l569_56989

-- Define the region R
def R := {(x, y) : ℝ × ℝ | |8 - x| + y ≤ 10 ∧ 3 * y - x ≥ 15}

-- Define the axis of revolution
def axis := {(x, y) : ℝ × ℝ | 3 * y - x = 15}

-- Define the volume of the solid of revolution
def volume_of_revolution (region : Set (ℝ × ℝ)) (axis : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem volume_of_R_revolution :
  volume_of_revolution R axis = (343 * Real.pi) / (12 * Real.sqrt 10) := by sorry

end NUMINAMATH_CALUDE_volume_of_R_revolution_l569_56989


namespace NUMINAMATH_CALUDE_sqrt_49284_squared_times_3_l569_56924

theorem sqrt_49284_squared_times_3 : (Real.sqrt 49284)^2 * 3 = 147852 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49284_squared_times_3_l569_56924


namespace NUMINAMATH_CALUDE_propositions_proof_l569_56945

theorem propositions_proof :
  (∀ (a b c : ℝ), c ≠ 0 → a * c^2 > b * c^2 → a > b) ∧
  (∀ (a b c d : ℝ), a > b → c > d → a + c > b + d) ∧
  (∃ (a b c d : ℝ), a > b ∧ c > d ∧ a * c ≤ b * d) ∧
  (∃ (a b c : ℝ), b > a ∧ a > 0 ∧ c > 0 ∧ a / b ≤ (a + c) / (b + c)) :=
by sorry

end NUMINAMATH_CALUDE_propositions_proof_l569_56945


namespace NUMINAMATH_CALUDE_zhuhai_visitors_scientific_notation_l569_56963

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem zhuhai_visitors_scientific_notation :
  toScientificNotation 3001000 = ScientificNotation.mk 3.001 6 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_zhuhai_visitors_scientific_notation_l569_56963


namespace NUMINAMATH_CALUDE_power_difference_quotient_l569_56956

theorem power_difference_quotient : 
  (2^12)^2 - (2^10)^2 = 4 * ((2^11)^2 - (2^9)^2) := by
  sorry

end NUMINAMATH_CALUDE_power_difference_quotient_l569_56956


namespace NUMINAMATH_CALUDE_intersection_when_m_neg_one_intersection_empty_iff_m_nonnegative_l569_56931

def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

theorem intersection_when_m_neg_one :
  B (-1) ∩ A = {x | 1 < x ∧ x < 2} := by sorry

theorem intersection_empty_iff_m_nonnegative (m : ℝ) :
  A ∩ B m = ∅ ↔ m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_neg_one_intersection_empty_iff_m_nonnegative_l569_56931


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_l569_56910

/-- Represents the minimum bailing rate problem --/
theorem minimum_bailing_rate 
  (distance_to_shore : ℝ) 
  (leaking_rate : ℝ) 
  (max_water_tolerance : ℝ) 
  (boat_speed : ℝ) 
  (h1 : distance_to_shore = 2) 
  (h2 : leaking_rate = 8) 
  (h3 : max_water_tolerance = 50) 
  (h4 : boat_speed = 3) :
  ∃ (bailing_rate : ℝ), 
    bailing_rate ≥ 7 ∧ 
    bailing_rate < 8 ∧
    (leaking_rate - bailing_rate) * (distance_to_shore / boat_speed * 60) ≤ max_water_tolerance :=
by sorry

end NUMINAMATH_CALUDE_minimum_bailing_rate_l569_56910


namespace NUMINAMATH_CALUDE_unique_pairs_count_l569_56909

/-- Represents the colors of marbles Tom has --/
inductive MarbleColor
  | Red
  | Green
  | Blue
  | Yellow
  | Orange

/-- Represents Tom's collection of marbles --/
def toms_marbles : List MarbleColor :=
  [MarbleColor.Red, MarbleColor.Green, MarbleColor.Blue,
   MarbleColor.Yellow, MarbleColor.Yellow,
   MarbleColor.Orange, MarbleColor.Orange]

/-- Counts the number of unique pairs of marbles --/
def count_unique_pairs (marbles : List MarbleColor) : Nat :=
  sorry

/-- Theorem stating that the number of unique pairs Tom can choose is 12 --/
theorem unique_pairs_count :
  count_unique_pairs toms_marbles = 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_pairs_count_l569_56909


namespace NUMINAMATH_CALUDE_smallest_integer_y_minus_three_is_smallest_l569_56941

theorem smallest_integer_y (y : ℤ) : 3 - 5 * y < 23 ↔ y ≥ -3 :=
  sorry

theorem minus_three_is_smallest : ∃ (y : ℤ), 3 - 5 * y < 23 ∧ ∀ (z : ℤ), 3 - 5 * z < 23 → z ≥ y :=
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_minus_three_is_smallest_l569_56941


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l569_56964

theorem sum_of_two_numbers (x : ℤ) : 
  x + 35 = 62 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l569_56964


namespace NUMINAMATH_CALUDE_no_three_common_tangents_l569_56981

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the number of common tangents between two circles -/
def commonTangents (c1 c2 : Circle) : ℕ :=
  sorry

/-- Theorem: It's impossible for two circles in the same plane to have exactly 3 common tangents -/
theorem no_three_common_tangents (c1 c2 : Circle) : 
  commonTangents c1 c2 ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_no_three_common_tangents_l569_56981


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l569_56996

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age : ℝ) 
  (group1_size group2_size group3_size : Nat) 
  (group1_avg group2_avg group3_avg : ℝ) :
  total_students = 15 →
  avg_age = 15 →
  group1_size = 5 →
  group2_size = 6 →
  group3_size = 3 →
  group1_avg = 13 →
  group2_avg = 15 →
  group3_avg = 17 →
  ∃ (fifteenth_student_age : ℝ),
    fifteenth_student_age = 19 ∧
    (group1_size * group1_avg + group2_size * group2_avg + group3_size * group3_avg + fifteenth_student_age) / total_students = avg_age :=
by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l569_56996


namespace NUMINAMATH_CALUDE_girls_from_pine_l569_56939

theorem girls_from_pine (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (maple_students : ℕ) (pine_students : ℕ) (maple_boys : ℕ)
  (h1 : total_students = 120)
  (h2 : total_boys = 70)
  (h3 : total_girls = 50)
  (h4 : maple_students = 50)
  (h5 : pine_students = 70)
  (h6 : maple_boys = 25)
  (h7 : total_students = total_boys + total_girls)
  (h8 : total_students = maple_students + pine_students)
  (h9 : maple_students = maple_boys + (total_girls - (pine_students - (total_boys - maple_boys)))) :
  pine_students - (total_boys - maple_boys) = 25 := by
  sorry

end NUMINAMATH_CALUDE_girls_from_pine_l569_56939


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l569_56950

theorem line_passes_through_fixed_point (m : ℝ) :
  (m + 1) * 1 + (2 * m - 1) * (-1) + m - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l569_56950


namespace NUMINAMATH_CALUDE_tangent_slope_xe_pow_x_l569_56915

open Real

theorem tangent_slope_xe_pow_x (e : ℝ) (h : e = exp 1) :
  let f : ℝ → ℝ := λ x ↦ x * exp x
  let df : ℝ → ℝ := λ x ↦ (1 + x) * exp x
  df 1 = 2 * e :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_xe_pow_x_l569_56915


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l569_56980

/-- An arithmetic sequence with first term -1 and common difference 2 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  2 * n - 17

/-- The sum of the first n terms of the arithmetic sequence -/
def sequence_sum (n : ℕ) : ℤ :=
  n^2 - 6*n

theorem arithmetic_sequence_properties :
  ∀ n : ℕ,
  (n > 0) →
  (arithmetic_sequence n = 2 * n - 17) ∧
  (sequence_sum n = n^2 - 6*n) ∧
  (∀ t : ℝ, (∀ k : ℕ, k > 0 → sequence_sum k > t) ↔ t < -6) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l569_56980


namespace NUMINAMATH_CALUDE_yellow_apples_probability_l569_56976

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The probability of an event -/
def probability (favorable_outcomes total_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

theorem yellow_apples_probability :
  let total_apples : ℕ := 10
  let yellow_apples : ℕ := 5
  let selected_apples : ℕ := 3
  probability (choose yellow_apples selected_apples) (choose total_apples selected_apples) = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_apples_probability_l569_56976


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_given_line_l569_56925

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def areParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem to prove -/
theorem line_through_point_parallel_to_given_line :
  let A : Point2D := ⟨2, 3⟩
  let givenLine : Line2D := ⟨2, 4, -3⟩
  let resultLine : Line2D := ⟨1, 2, -8⟩
  pointOnLine A resultLine ∧ areParallel resultLine givenLine := by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_given_line_l569_56925


namespace NUMINAMATH_CALUDE_hike_taxi_count_hike_taxi_count_is_six_l569_56913

/-- Calculates the number of taxis required for a hike --/
theorem hike_taxi_count (total_people : ℕ) (car_count : ℕ) (van_count : ℕ) 
  (people_per_car : ℕ) (people_per_van : ℕ) (people_per_taxi : ℕ) : ℕ :=
  let people_in_cars := car_count * people_per_car
  let people_in_vans := van_count * people_per_van
  let people_in_taxis := total_people - (people_in_cars + people_in_vans)
  people_in_taxis / people_per_taxi

/-- Proves that 6 taxis were required for the hike --/
theorem hike_taxi_count_is_six : 
  hike_taxi_count 58 3 2 4 5 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hike_taxi_count_hike_taxi_count_is_six_l569_56913


namespace NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_sum_l569_56954

/-- The sum of arithmetic sequence with 5 terms, starting from x and with common difference 3 -/
def sequence_sum (x : ℕ) : ℕ := x + (x + 3) + (x + 6) + (x + 9) + (x + 12)

/-- A natural number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem smallest_x_for_perfect_cube_sum : 
  (∀ x : ℕ, x > 0 ∧ x < 19 → ¬(is_perfect_cube (sequence_sum x))) ∧ 
  (is_perfect_cube (sequence_sum 19)) := by
sorry

end NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_sum_l569_56954


namespace NUMINAMATH_CALUDE_leader_assistant_combinations_l569_56997

/-- The number of ways to choose a team leader and an assistant of the same gender -/
def choose_leader_and_assistant (total : ℕ) (boys : ℕ) (girls : ℕ) : ℕ :=
  boys * (boys - 1) + girls * (girls - 1)

/-- Theorem: There are 98 ways to choose a team leader and an assistant of the same gender
    from a class of 15 students, consisting of 8 boys and 7 girls -/
theorem leader_assistant_combinations :
  choose_leader_and_assistant 15 8 7 = 98 := by
  sorry

end NUMINAMATH_CALUDE_leader_assistant_combinations_l569_56997


namespace NUMINAMATH_CALUDE_thief_speed_l569_56937

/-- Proves that given the initial conditions, the speed of the thief is 8 km/hr -/
theorem thief_speed (initial_distance : ℝ) (policeman_speed : ℝ) (thief_distance : ℝ)
  (h1 : initial_distance = 175 / 1000) -- Convert 175 meters to kilometers
  (h2 : policeman_speed = 10)
  (h3 : thief_distance = 700 / 1000) -- Convert 700 meters to kilometers
  : ∃ (thief_speed : ℝ), thief_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_thief_speed_l569_56937


namespace NUMINAMATH_CALUDE_circle_center_point_is_center_l569_56906

/-- The center of a circle given by the equation x^2 - 6x + y^2 + 8y - 16 = 0 is (3, -4) -/
theorem circle_center (x y : ℝ) : 
  (x^2 - 6*x + y^2 + 8*y - 16 = 0) ↔ ((x - 3)^2 + (y + 4)^2 = 9) :=
by sorry

/-- The point (3, -4) is the center of the circle -/
theorem point_is_center : 
  ∃ (r : ℝ), ∀ (x y : ℝ), x^2 - 6*x + y^2 + 8*y - 16 = 0 ↔ (x - 3)^2 + (y + 4)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_point_is_center_l569_56906


namespace NUMINAMATH_CALUDE_rational_absolute_value_and_negative_numbers_l569_56957

theorem rational_absolute_value_and_negative_numbers :
  (∀ x : ℚ, |x| ≥ 0 ∧ (|x| = 0 ↔ x = 0)) ∧
  (∀ x : ℝ, -x > x → x < 0) := by
  sorry

end NUMINAMATH_CALUDE_rational_absolute_value_and_negative_numbers_l569_56957


namespace NUMINAMATH_CALUDE_original_painting_height_l569_56952

/-- Proves that given a painting with width 15 inches and a print of the painting with width 37.5 inches and height 25 inches, the height of the original painting is 10 inches. -/
theorem original_painting_height
  (original_width : ℝ)
  (print_width : ℝ)
  (print_height : ℝ)
  (h_original_width : original_width = 15)
  (h_print_width : print_width = 37.5)
  (h_print_height : print_height = 25) :
  print_height / (print_width / original_width) = 10 := by
  sorry


end NUMINAMATH_CALUDE_original_painting_height_l569_56952


namespace NUMINAMATH_CALUDE_rationalize_denominator_l569_56919

theorem rationalize_denominator : 
  ∃ (A B C D E F : ℚ), 
    F > 0 ∧ 
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) = 
    (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    A = -1 ∧ B = -3 ∧ C = 1 ∧ D = 2/3 ∧ E = 165 ∧ F = 17 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l569_56919


namespace NUMINAMATH_CALUDE_andy_initial_candies_l569_56968

/-- The number of candies each person has initially and after distribution --/
structure CandyDistribution where
  billy_initial : ℕ
  caleb_initial : ℕ
  andy_initial : ℕ
  father_bought : ℕ
  billy_received : ℕ
  caleb_received : ℕ
  andy_final_diff : ℕ

/-- Theorem stating that Andy initially took 9 candies --/
theorem andy_initial_candies (d : CandyDistribution) 
  (h1 : d.billy_initial = 6)
  (h2 : d.caleb_initial = 11)
  (h3 : d.father_bought = 36)
  (h4 : d.billy_received = 8)
  (h5 : d.caleb_received = 11)
  (h6 : d.andy_final_diff = 4)
  : d.andy_initial = 9 := by
  sorry


end NUMINAMATH_CALUDE_andy_initial_candies_l569_56968


namespace NUMINAMATH_CALUDE_area_BXC_specific_trapezoid_l569_56934

/-- Represents a trapezoid ABCD with intersection point X of diagonals -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  area : ℝ

/-- Calculates the area of triangle BXC in the trapezoid -/
def area_BXC (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of triangle BXC in the specific trapezoid -/
theorem area_BXC_specific_trapezoid :
  let t : Trapezoid := { AB := 20, CD := 30, area := 300 }
  area_BXC t = 72 := by sorry

end NUMINAMATH_CALUDE_area_BXC_specific_trapezoid_l569_56934


namespace NUMINAMATH_CALUDE_students_enjoying_both_sports_l569_56953

theorem students_enjoying_both_sports 
  (total : ℕ) 
  (running : ℕ) 
  (basketball : ℕ) 
  (neither : ℕ) 
  (h1 : total = 38) 
  (h2 : running = 21) 
  (h3 : basketball = 15) 
  (h4 : neither = 10) :
  running + basketball - (total - neither) = 8 :=
by sorry

end NUMINAMATH_CALUDE_students_enjoying_both_sports_l569_56953


namespace NUMINAMATH_CALUDE_professor_seating_theorem_l569_56911

/-- The number of chairs in a row -/
def num_chairs : ℕ := 10

/-- The number of professors -/
def num_professors : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 7

/-- The minimum number of students required between each professor -/
def min_students_between : ℕ := 2

/-- A function that calculates the number of ways professors can choose their chairs -/
def professor_seating_arrangements (n_chairs : ℕ) (n_profs : ℕ) (n_students : ℕ) (min_between : ℕ) : ℕ :=
  sorry -- The actual implementation is not provided here

/-- Theorem stating that the number of seating arrangements for professors is 6 -/
theorem professor_seating_theorem :
  professor_seating_arrangements num_chairs num_professors num_students min_students_between = 6 :=
by sorry

end NUMINAMATH_CALUDE_professor_seating_theorem_l569_56911


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l569_56977

def U : Set ℤ := {-1, 0, 1, 2}

def A : Set ℤ := {x ∈ U | x^2 < 1}

theorem complement_of_A_in_U : Set.compl A = {-1, 1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l569_56977


namespace NUMINAMATH_CALUDE_no_solution_in_interval_l569_56962

theorem no_solution_in_interval : 
  ¬ ∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ (3 * x - 2) ≥ 3 * (12 - 3 * x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_in_interval_l569_56962


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l569_56998

theorem gcf_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l569_56998


namespace NUMINAMATH_CALUDE_compare_fractions_l569_56920

theorem compare_fractions : -2/7 > -3/10 := by sorry

end NUMINAMATH_CALUDE_compare_fractions_l569_56920


namespace NUMINAMATH_CALUDE_exponent_calculation_l569_56986

theorem exponent_calculation (a : ℝ) : (-a)^10 / (-a)^3 = -a^7 := by sorry

end NUMINAMATH_CALUDE_exponent_calculation_l569_56986


namespace NUMINAMATH_CALUDE_factorization_quadratic_l569_56995

theorem factorization_quadratic (x : ℝ) : x^2 + 2*x = x*(x+2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_quadratic_l569_56995


namespace NUMINAMATH_CALUDE_train_length_l569_56987

/-- The length of a train given its speed, platform length, and crossing time -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 96 * (5 / 18) →
  platform_length = 480 →
  crossing_time = 36 →
  ∃ (train_length : ℝ), abs (train_length - 480.12) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l569_56987


namespace NUMINAMATH_CALUDE_birth_year_problem_l569_56973

theorem birth_year_problem (x : ℕ) : 
  (1850 ≤ x^2 - 2*x + 1) ∧ (x^2 - 2*x + 1 < 1900) →
  (x^2 - x + 1 - (x^2 - 2*x + 1) = x) →
  x^2 - 2*x + 1 = 1849 := by
sorry

end NUMINAMATH_CALUDE_birth_year_problem_l569_56973


namespace NUMINAMATH_CALUDE_problem_statement_l569_56935

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -7)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 8) :
  b / (a + b) + c / (b + c) + a / (c + a) = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l569_56935


namespace NUMINAMATH_CALUDE_single_intersection_l569_56974

/-- The parabola function -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 2 * y + 4

/-- The line function -/
def line (k : ℝ) : ℝ := k

/-- Theorem stating the condition for single intersection -/
theorem single_intersection (k : ℝ) : 
  (∃! y, parabola y = line k) ↔ k = 13/3 := by sorry

end NUMINAMATH_CALUDE_single_intersection_l569_56974


namespace NUMINAMATH_CALUDE_m_range_l569_56983

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, |x - 1| > m - 1
def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(5 - 2*m))^x > (-(5 - 2*m))^y

-- Define the theorem
theorem m_range (m : ℝ) : 
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (1 ≤ m ∧ m < 2) :=
by sorry

end NUMINAMATH_CALUDE_m_range_l569_56983


namespace NUMINAMATH_CALUDE_constant_term_value_l569_56969

theorem constant_term_value (y : ℝ) (d : ℝ) :
  y = 2 → (5 * y^2 - 8 * y + 55 = d ↔ d = 59) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_value_l569_56969


namespace NUMINAMATH_CALUDE_effective_treatment_combination_l569_56955

structure Treatment where
  name : String
  relieves : List String
  causes : List String

def aspirin : Treatment :=
  { name := "Aspirin"
  , relieves := ["headache", "rheumatic knee pain"]
  , causes := ["heart pain", "stomach pain"] }

def antibiotics : Treatment :=
  { name := "Antibiotics"
  , relieves := ["migraine", "heart pain"]
  , causes := ["stomach pain", "knee pain", "itching"] }

def warmCompress : Treatment :=
  { name := "Warm compress"
  , relieves := ["itching", "stomach pain"]
  , causes := [] }

def initialSymptom : String := "headache"

def isEffectiveCombination (treatments : List Treatment) : Prop :=
  (initialSymptom ∈ (treatments.bind (λ t => t.relieves))) ∧
  (∀ s, s ∈ (treatments.bind (λ t => t.causes)) →
    ∃ t ∈ treatments, s ∈ t.relieves)

theorem effective_treatment_combination :
  isEffectiveCombination [aspirin, antibiotics, warmCompress] :=
sorry

end NUMINAMATH_CALUDE_effective_treatment_combination_l569_56955


namespace NUMINAMATH_CALUDE_girls_equal_barefoot_children_l569_56975

theorem girls_equal_barefoot_children (B_b G_b G_s : ℕ) :
  B_b = G_s →
  B_b + G_b = G_b + G_s :=
by sorry

end NUMINAMATH_CALUDE_girls_equal_barefoot_children_l569_56975


namespace NUMINAMATH_CALUDE_power_of_product_l569_56936

theorem power_of_product (x y : ℝ) : (x^2 * y)^3 = x^6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l569_56936


namespace NUMINAMATH_CALUDE_profit_maximization_and_cost_l569_56943

/-- Represents the relationship between selling price and daily sales volume -/
def sales_volume (x : ℝ) : ℝ := -30 * x + 1500

/-- Calculates the daily sales profit -/
def sales_profit (x : ℝ) : ℝ := sales_volume x * (x - 30)

/-- Calculates the daily profit including additional cost a -/
def total_profit (x a : ℝ) : ℝ := sales_volume x * (x - 30 - a)

theorem profit_maximization_and_cost (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 10) :
  (∀ x, sales_profit x ≤ sales_profit 40) ∧
  (∃ x, 40 ≤ x ∧ x ≤ 45 ∧ total_profit x a = 2430) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_profit_maximization_and_cost_l569_56943


namespace NUMINAMATH_CALUDE_rectangle_max_area_l569_56994

/-- A rectangle with whole number dimensions and perimeter 40 has a maximum area of 100 -/
theorem rectangle_max_area :
  ∀ l w : ℕ,
  l > 0 → w > 0 →
  2 * l + 2 * w = 40 →
  l * w ≤ 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l569_56994


namespace NUMINAMATH_CALUDE_spherical_coordinate_transformation_l569_56972

/-- Given a point in rectangular coordinates (3, -8, 6) with corresponding
    spherical coordinates (ρ, θ, φ), this theorem proves the rectangular
    coordinates of the point with spherical coordinates (ρ, θ + π/4, -φ). -/
theorem spherical_coordinate_transformation (ρ θ φ : ℝ) :
  3 = ρ * Real.sin φ * Real.cos θ →
  -8 = ρ * Real.sin φ * Real.sin θ →
  6 = ρ * Real.cos φ →
  ∃ (x y : ℝ),
    x = -ρ * Real.sin φ * (Real.sqrt 2 / 2 * Real.cos θ - Real.sqrt 2 / 2 * Real.sin θ) ∧
    y = -ρ * Real.sin φ * (Real.sqrt 2 / 2 * Real.sin θ + Real.sqrt 2 / 2 * Real.cos θ) ∧
    6 = ρ * Real.cos φ :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_transformation_l569_56972


namespace NUMINAMATH_CALUDE_perspective_difference_l569_56918

def num_students : ℕ := 250
def num_teachers : ℕ := 6
def class_sizes : List ℕ := [100, 50, 50, 25, 15, 10]

def teacher_perspective (sizes : List ℕ) : ℚ :=
  (sizes.sum : ℚ) / num_teachers

def student_perspective (sizes : List ℕ) : ℚ :=
  (sizes.map (λ x => x * x)).sum / num_students

theorem perspective_difference :
  teacher_perspective class_sizes - student_perspective class_sizes = -22.13 := by
  sorry

end NUMINAMATH_CALUDE_perspective_difference_l569_56918


namespace NUMINAMATH_CALUDE_line_increase_percentage_l569_56921

theorem line_increase_percentage (initial_lines : ℕ) 
  (h1 : initial_lines + 80 = 240) : 
  (80 : ℝ) / initial_lines * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_line_increase_percentage_l569_56921


namespace NUMINAMATH_CALUDE_policeman_catch_condition_l569_56992

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the square arena -/
structure Square where
  sideLength : ℝ
  center : Point

/-- Represents the gangster -/
structure Gangster where
  position : Point
  speed : ℝ

/-- Represents the policeman -/
structure Policeman where
  position : Point
  speed : ℝ

/-- Determines if the policeman can catch the gangster -/
def canCatchGangster (s : Square) (g : Gangster) (p : Policeman) : Prop :=
  ∃ t : ℝ, t ≥ 0 ∧ 
    (∃ catchPoint : Point, 
      catchPoint.x ≥ s.center.x - s.sideLength / 2 ∧
      catchPoint.x ≤ s.center.x + s.sideLength / 2 ∧
      catchPoint.y ≥ s.center.y - s.sideLength / 2 ∧
      catchPoint.y ≤ s.center.y + s.sideLength / 2 ∧
      (catchPoint.x - p.position.x)^2 + (catchPoint.y - p.position.y)^2 ≤ (p.speed * t)^2 ∧
      (abs (catchPoint.x - g.position.x) + abs (catchPoint.y - g.position.y)) ≤ g.speed * t)

theorem policeman_catch_condition (s : Square) (g : Gangster) (p : Policeman) :
  s.sideLength = 6 ∧ g.speed = 3 ∧ p.position = s.center →
  canCatchGangster s g p ↔ p.speed > 1 :=
sorry

end NUMINAMATH_CALUDE_policeman_catch_condition_l569_56992


namespace NUMINAMATH_CALUDE_smallest_number_l569_56900

theorem smallest_number (a b c d : ℤ) (ha : a = -2) (hb : b = -1) (hc : c = 1) (hd : d = 0) :
  a ≤ b ∧ a ≤ c ∧ a ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l569_56900


namespace NUMINAMATH_CALUDE_deepak_age_l569_56967

theorem deepak_age (arun_age deepak_age : ℕ) : 
  arun_age / deepak_age = 2 / 3 →
  arun_age + 5 = 25 →
  deepak_age = 30 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l569_56967


namespace NUMINAMATH_CALUDE_rock_paper_scissors_games_l569_56933

/-- The number of students in the group -/
def num_students : ℕ := 9

/-- The number of neighbors each student doesn't play with -/
def neighbors : ℕ := 2

/-- The number of games each student plays -/
def games_per_student : ℕ := num_students - 1 - neighbors

/-- The total number of games played, counting each game twice -/
def total_games : ℕ := num_students * games_per_student

/-- The number of unique games played -/
def unique_games : ℕ := total_games / 2

theorem rock_paper_scissors_games :
  unique_games = 27 :=
sorry

end NUMINAMATH_CALUDE_rock_paper_scissors_games_l569_56933


namespace NUMINAMATH_CALUDE_gambler_win_rate_is_40_percent_l569_56932

/-- Represents the gambler's statistics -/
structure GamblerStats where
  games_played : ℕ
  future_games : ℕ
  future_win_rate : ℚ
  target_win_rate : ℚ

/-- Calculates the current win rate of the gambler -/
def current_win_rate (stats : GamblerStats) : ℚ :=
  let total_games := stats.games_played + stats.future_games
  let future_wins := stats.future_win_rate * stats.future_games
  let total_wins := stats.target_win_rate * total_games
  (total_wins - future_wins) / stats.games_played

/-- Theorem stating the gambler's current win rate is 40% under given conditions -/
theorem gambler_win_rate_is_40_percent (stats : GamblerStats) 
  (h1 : stats.games_played = 40)
  (h2 : stats.future_games = 80)
  (h3 : stats.future_win_rate = 7/10)
  (h4 : stats.target_win_rate = 6/10) :
  current_win_rate stats = 4/10 := by
  sorry

#eval current_win_rate { games_played := 40, future_games := 80, future_win_rate := 7/10, target_win_rate := 6/10 }

end NUMINAMATH_CALUDE_gambler_win_rate_is_40_percent_l569_56932


namespace NUMINAMATH_CALUDE_system_solvability_l569_56946

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  x = 6 / a - |y - a| ∧ x^2 + y^2 + b^2 + 63 = 2 * (b * y - 8 * x)

-- Define the set of valid 'a' values
def valid_a_set : Set ℝ := {a | a ≤ -2/3 ∨ a > 0}

-- Theorem statement
theorem system_solvability (a : ℝ) :
  (∃ b x y, system a b x y) ↔ a ∈ valid_a_set :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l569_56946


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l569_56949

theorem arithmetic_calculations :
  (4.6 - (1.75 + 2.08) = 0.77) ∧
  (9.5 + 4.85 - 6.36 = 7.99) ∧
  (5.6 + 2.7 + 4.4 = 12.7) ∧
  (13 - 4.85 - 3.15 = 5) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l569_56949


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l569_56965

/-- Calculates the maximum number of complete books that can be read given the reading speed, book length, and available time. -/
def max_complete_books_read (reading_speed : ℕ) (book_length : ℕ) (available_time : ℕ) : ℕ :=
  (available_time * reading_speed) / book_length

/-- Proves that Robert can read at most 2 complete 360-page books in 8 hours at a speed of 120 pages per hour. -/
theorem robert_reading_capacity :
  max_complete_books_read 120 360 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l569_56965


namespace NUMINAMATH_CALUDE_expression_equality_l569_56999

theorem expression_equality : Real.sqrt 8 ^ (1/3) - |2 - Real.sqrt 3| + (1/2)^0 - Real.sqrt 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l569_56999


namespace NUMINAMATH_CALUDE_teddy_bear_shelves_l569_56938

theorem teddy_bear_shelves (total_bears : ℕ) (shelf_capacity : ℕ) (filled_shelves : ℕ) : 
  total_bears = 98 → 
  shelf_capacity = 7 → 
  filled_shelves = total_bears / shelf_capacity →
  filled_shelves = 14 := by
sorry

end NUMINAMATH_CALUDE_teddy_bear_shelves_l569_56938


namespace NUMINAMATH_CALUDE_muffin_mix_buyers_l569_56926

/-- Given a set of buyers with specific purchasing patterns for cake and muffin mixes,
    prove that the number of buyers who purchase muffin mix is 40. -/
theorem muffin_mix_buyers (total : ℕ) (cake : ℕ) (both : ℕ) (neither_prob : ℚ) :
  total = 100 →
  cake = 50 →
  both = 16 →
  neither_prob = 26 / 100 →
  ∃ (muffin : ℕ),
    muffin = 40 ∧
    (cake + muffin - both : ℚ) = total - (neither_prob * total) :=
by sorry

end NUMINAMATH_CALUDE_muffin_mix_buyers_l569_56926


namespace NUMINAMATH_CALUDE_ratio_problem_l569_56930

theorem ratio_problem (first_term : ℝ) (ratio_percent : ℝ) (second_term : ℝ) :
  first_term = 15 ∧ 
  ratio_percent = 60 ∧ 
  (first_term / second_term) * 100 = ratio_percent →
  second_term = 25 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l569_56930


namespace NUMINAMATH_CALUDE_arithmetic_mean_scaling_l569_56942

theorem arithmetic_mean_scaling (b₁ b₂ b₃ b₄ b₅ : ℝ) :
  let original_set := [b₁, b₂, b₃, b₃, b₅]
  let scaled_set := original_set.map (· * 3)
  let original_mean := (b₁ + b₂ + b₃ + b₄ + b₅) / 5
  let scaled_mean := (scaled_set.sum) / 5
  scaled_mean = 3 * original_mean := by
sorry


end NUMINAMATH_CALUDE_arithmetic_mean_scaling_l569_56942


namespace NUMINAMATH_CALUDE_identifiable_bulbs_for_two_trips_three_states_max_identifiable_bulbs_is_power_l569_56905

/-- The maximum number of bulbs and switches that can be identified -/
def max_identifiable_bulbs (n : ℕ) (m : ℕ) : ℕ := m^n

/-- Theorem: With 2 trips and 3 states, 9 bulbs and switches can be identified -/
theorem identifiable_bulbs_for_two_trips_three_states :
  max_identifiable_bulbs 2 3 = 9 := by
  sorry

/-- Theorem: The maximum number of identifiable bulbs is always a power of the number of states -/
theorem max_identifiable_bulbs_is_power (n m : ℕ) :
  ∃ k, max_identifiable_bulbs n m = m^k := by
  sorry

end NUMINAMATH_CALUDE_identifiable_bulbs_for_two_trips_three_states_max_identifiable_bulbs_is_power_l569_56905


namespace NUMINAMATH_CALUDE_last_number_in_first_set_l569_56929

theorem last_number_in_first_set (x : ℝ) (last_number : ℝ) : 
  (28 + x + 42 + 78 + last_number) / 5 = 90 ∧ 
  (128 + 255 + 511 + 1023 + x) / 5 = 423 → 
  last_number = 104 := by
sorry

end NUMINAMATH_CALUDE_last_number_in_first_set_l569_56929


namespace NUMINAMATH_CALUDE_range_of_a_l569_56970

-- Define the function f(x) = (a^2 - 1)x^2 - (a-1)x - 1
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 1) * x^2 - (a - 1) * x - 1

-- Define the property that f(x) < 0 for all real x
def always_negative (a : ℝ) : Prop := ∀ x : ℝ, f a x < 0

-- Theorem statement
theorem range_of_a : 
  {a : ℝ | always_negative a} = Set.Ioc (- 3/5) 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l569_56970


namespace NUMINAMATH_CALUDE_lucas_chocolate_theorem_l569_56940

/-- Represents the number of pieces of chocolate candy Lucas makes for each student. -/
def pieces_per_student : ℕ := 4

/-- Represents the total number of pieces of chocolate candy Lucas made last Monday. -/
def total_pieces_last_monday : ℕ := 40

/-- Represents the number of students who will not be coming to class this upcoming Monday. -/
def absent_students : ℕ := 3

/-- Calculates the number of pieces of chocolate candy Lucas will make for his class on the upcoming Monday. -/
def pieces_for_upcoming_monday : ℕ :=
  ((total_pieces_last_monday / pieces_per_student) - absent_students) * pieces_per_student

/-- Theorem stating that Lucas will make 28 pieces of chocolate candy for his class on the upcoming Monday. -/
theorem lucas_chocolate_theorem :
  pieces_for_upcoming_monday = 28 := by sorry

end NUMINAMATH_CALUDE_lucas_chocolate_theorem_l569_56940


namespace NUMINAMATH_CALUDE_sum_of_numbers_l569_56923

theorem sum_of_numbers (a b : ℕ+) 
  (hcf : Nat.gcd a b = 5)
  (lcm : Nat.lcm a b = 120)
  (sum_reciprocals : (1 : ℚ) / a + (1 : ℚ) / b = 11 / 120) :
  a + b = 55 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l569_56923


namespace NUMINAMATH_CALUDE_g_of_two_eq_zero_l569_56922

/-- Given a function g(x) = x^2 - 4 for all real x, prove that g(2) = 0 -/
theorem g_of_two_eq_zero (g : ℝ → ℝ) (h : ∀ x, g x = x^2 - 4) : g 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_of_two_eq_zero_l569_56922


namespace NUMINAMATH_CALUDE_doritos_distribution_l569_56971

theorem doritos_distribution (total_bags : ℕ) (doritos_fraction : ℚ) (num_piles : ℕ) : 
  total_bags = 80 →
  doritos_fraction = 1/4 →
  num_piles = 4 →
  (total_bags : ℚ) * doritos_fraction / num_piles = 5 := by
  sorry

end NUMINAMATH_CALUDE_doritos_distribution_l569_56971


namespace NUMINAMATH_CALUDE_number_equation_proof_l569_56914

theorem number_equation_proof (x : ℝ) (N : ℝ) : 
  x = 32 → 
  N - (23 - (15 - x)) = 12 * 2 / (1 / 2) → 
  N = 88 := by
sorry

end NUMINAMATH_CALUDE_number_equation_proof_l569_56914


namespace NUMINAMATH_CALUDE_remainder_of_polynomial_l569_56928

theorem remainder_of_polynomial (n : ℤ) (k : ℤ) (h : ∃ a : ℤ, n = 75 * a - 2) :
  (n^2 + 2*n + 3) % 75 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_polynomial_l569_56928


namespace NUMINAMATH_CALUDE_rational_number_equation_l569_56951

theorem rational_number_equation (A B : ℝ) (x : ℚ) :
  x = (1 / 2) * x + (1 / 5) * ((3 / 4) * (A + B) - (2 / 3) * (A + B)) →
  x = (1 / 30) * (A + B) := by
  sorry

end NUMINAMATH_CALUDE_rational_number_equation_l569_56951


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l569_56907

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l569_56907


namespace NUMINAMATH_CALUDE_triangle_side_length_l569_56903

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  b = 7 →
  c = 3 →
  Real.cos (B - C) = 17/18 →
  a = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l569_56903


namespace NUMINAMATH_CALUDE_cauliflower_increase_40401_l569_56988

/-- Represents the increase in cauliflower production from one year to the next,
    given a square garden where each cauliflower takes 1 square foot. -/
def cauliflower_increase (this_year_production : ℕ) : ℕ :=
  this_year_production - (Nat.sqrt this_year_production - 1)^2

/-- Theorem stating that for a square garden with 40401 cauliflowers this year,
    the increase in production from last year is 401 cauliflowers. -/
theorem cauliflower_increase_40401 :
  cauliflower_increase 40401 = 401 := by
  sorry

#eval cauliflower_increase 40401

end NUMINAMATH_CALUDE_cauliflower_increase_40401_l569_56988


namespace NUMINAMATH_CALUDE_hit_first_third_fifth_probability_hit_exactly_three_probability_l569_56961

-- Define the probability of hitting the target
def hit_probability : ℚ := 3/5

-- Define the number of shots
def num_shots : ℕ := 5

-- Theorem for the first part of the problem
theorem hit_first_third_fifth_probability :
  (hit_probability * (1 - hit_probability) * hit_probability * (1 - hit_probability) * hit_probability : ℚ) = 108/3125 :=
sorry

-- Theorem for the second part of the problem
theorem hit_exactly_three_probability :
  (Nat.choose num_shots 3 : ℚ) * hit_probability^3 * (1 - hit_probability)^2 = 216/625 :=
sorry

end NUMINAMATH_CALUDE_hit_first_third_fifth_probability_hit_exactly_three_probability_l569_56961
