import Mathlib

namespace sum_of_digits_is_nine_l3605_360585

/-- The sum of the tens digit and the units digit in the decimal representation of 7^1974 -/
def sum_of_digits : ℕ :=
  let n := 7^1974
  let tens_digit := (n / 10) % 10
  let units_digit := n % 10
  tens_digit + units_digit

/-- The sum of the tens digit and the units digit in the decimal representation of 7^1974 is 9 -/
theorem sum_of_digits_is_nine : sum_of_digits = 9 := by
  sorry

end sum_of_digits_is_nine_l3605_360585


namespace longest_side_of_triangle_l3605_360503

/-- Given a triangle with sides in ratio 1/2 : 1/3 : 1/4 and perimeter 104 cm, 
    the longest side is 48 cm. -/
theorem longest_side_of_triangle (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → -- sides are positive
  a / b = 3 / 2 ∧ b / c = 4 / 3 → -- ratio condition
  a + b + c = 104 → -- perimeter condition
  a = 48 := by sorry

end longest_side_of_triangle_l3605_360503


namespace cubic_not_always_square_l3605_360534

theorem cubic_not_always_square (a b c : ℤ) : ∃ n : ℕ+, ¬∃ m : ℤ, (n : ℤ)^3 + a*(n : ℤ)^2 + b*(n : ℤ) + c = m^2 := by
  sorry

end cubic_not_always_square_l3605_360534


namespace neighbor_field_size_l3605_360547

/-- Represents the yield of a cornfield over a period of time -/
structure CornYield where
  amount : ℕ
  months : ℕ

/-- Represents a cornfield -/
structure Cornfield where
  hectares : ℕ
  yield_per_period : CornYield

def total_yield (field : Cornfield) (months : ℕ) : ℕ :=
  field.hectares * field.yield_per_period.amount * (months / field.yield_per_period.months)

def johnson_field : Cornfield :=
  { hectares := 1
  , yield_per_period := { amount := 80, months := 2 }
  }

def neighbor_field (hectares : ℕ) : Cornfield :=
  { hectares := hectares
  , yield_per_period := { amount := 160, months := 2 }
  }

theorem neighbor_field_size :
  ∃ (x : ℕ), total_yield johnson_field 6 + total_yield (neighbor_field x) 6 = 1200 ∧ x = 2 := by
  sorry

end neighbor_field_size_l3605_360547


namespace smallest_mustang_is_12_inches_l3605_360533

/-- The length of the smallest Mustang model given the full-size and scaling factors -/
def smallest_mustang_length (full_size : ℝ) (mid_size_factor : ℝ) (smallest_factor : ℝ) : ℝ :=
  full_size * mid_size_factor * smallest_factor

/-- Theorem stating that the smallest Mustang model is 12 inches long -/
theorem smallest_mustang_is_12_inches :
  smallest_mustang_length 240 (1/10) (1/2) = 12 := by
  sorry

end smallest_mustang_is_12_inches_l3605_360533


namespace floor_sqrt_80_l3605_360556

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end floor_sqrt_80_l3605_360556


namespace range_of_power_function_l3605_360543

theorem range_of_power_function (k : ℝ) (h : k > 0) :
  Set.range (fun x : ℝ => x ^ k) ∩ Set.Ici 1 = Set.Ici 1 := by sorry

end range_of_power_function_l3605_360543


namespace smoking_lung_disease_relation_l3605_360510

/-- Represents the Chi-square statistic -/
def K_squared : ℝ := 5.231

/-- The probability that K² is greater than or equal to 3.841 -/
def p_value_95 : ℝ := 0.05

/-- The probability that K² is greater than or equal to 6.635 -/
def p_value_99 : ℝ := 0.01

/-- Confidence level for the relationship between smoking and lung disease -/
def confidence_level : ℝ := 1 - p_value_95

theorem smoking_lung_disease_relation :
  K_squared ≥ 3.841 ∧ K_squared < 6.635 →
  confidence_level > 0.95 :=
sorry

end smoking_lung_disease_relation_l3605_360510


namespace equation_solution_l3605_360535

theorem equation_solution (a b c : ℤ) : 
  (∀ x : ℤ, (x - a) * (x - 12) + 1 = (x + b) * (x + c)) ↔ 
  ((a = 10 ∧ b = -11 ∧ c = -11) ∨ (a = 14 ∧ b = -13 ∧ c = -13)) := by
sorry

end equation_solution_l3605_360535


namespace lisa_savings_analysis_l3605_360589

def lisa_savings : Fin 6 → ℝ
  | 0 => 100  -- January
  | 1 => 300  -- February
  | 2 => 200  -- March
  | 3 => 200  -- April
  | 4 => 100  -- May
  | 5 => 100  -- June

theorem lisa_savings_analysis :
  let total_average := (lisa_savings 0 + lisa_savings 1 + lisa_savings 2 + 
                        lisa_savings 3 + lisa_savings 4 + lisa_savings 5) / 6
  let first_trimester_average := (lisa_savings 0 + lisa_savings 1 + lisa_savings 2) / 3
  let second_trimester_average := (lisa_savings 3 + lisa_savings 4 + lisa_savings 5) / 3
  (total_average = 1000 / 6) ∧
  (first_trimester_average = 200) ∧
  (second_trimester_average = 400 / 3) ∧
  (first_trimester_average - second_trimester_average = 200 / 3) :=
by sorry

end lisa_savings_analysis_l3605_360589


namespace floors_per_house_l3605_360598

/-- The number of floors in each house given the building conditions -/
theorem floors_per_house 
  (builders_per_floor : ℕ)
  (days_per_floor : ℕ)
  (daily_wage : ℕ)
  (total_builders : ℕ)
  (total_houses : ℕ)
  (total_cost : ℕ)
  (h1 : builders_per_floor = 3)
  (h2 : days_per_floor = 30)
  (h3 : daily_wage = 100)
  (h4 : total_builders = 6)
  (h5 : total_houses = 5)
  (h6 : total_cost = 270000) :
  total_cost / (total_builders * daily_wage * days_per_floor * total_houses) = 3 :=
sorry

end floors_per_house_l3605_360598


namespace sqrt_x_plus_reciprocal_l3605_360508

theorem sqrt_x_plus_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_x_plus_reciprocal_l3605_360508


namespace carrot_sticks_leftover_l3605_360513

theorem carrot_sticks_leftover (total_carrots : ℕ) (num_people : ℕ) (h1 : total_carrots = 74) (h2 : num_people = 12) :
  total_carrots % num_people = 2 := by
  sorry

end carrot_sticks_leftover_l3605_360513


namespace geometric_sequence_common_ratio_l3605_360596

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_diff_1 : a 2 - a 1 = 1)
  (h_diff_2 : a 5 - a 4 = 8) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 :=
sorry

end geometric_sequence_common_ratio_l3605_360596


namespace geometric_sum_example_l3605_360550

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Proof that the sum of the first 8 terms of the geometric sequence
    with first term 1/3 and common ratio 1/3 is 9840/19683 -/
theorem geometric_sum_example :
  geometric_sum (1/3) (1/3) 8 = 9840/19683 := by
  sorry

end geometric_sum_example_l3605_360550


namespace integers_starting_with_6_divisible_by_25_no_integers_divisible_by_35_without_first_digit_l3605_360526

def starts_with_6 (x : ℕ) : Prop :=
  ∃ n : ℕ, x = 6 * 10^n + (x % 10^n)

def divisible_by_25_without_first_digit (x : ℕ) : Prop :=
  ∃ n : ℕ, (x % 10^n) % 25 = 0

def divisible_by_35_without_first_digit (x : ℕ) : Prop :=
  ∃ n : ℕ, (x % 10^n) % 35 = 0

theorem integers_starting_with_6_divisible_by_25 :
  ∀ x : ℕ, starts_with_6 x ∧ divisible_by_25_without_first_digit x →
    ∃ k : ℕ, x = 625 * 10^k :=
sorry

theorem no_integers_divisible_by_35_without_first_digit :
  ¬ ∃ x : ℕ, divisible_by_35_without_first_digit x :=
sorry

end integers_starting_with_6_divisible_by_25_no_integers_divisible_by_35_without_first_digit_l3605_360526


namespace square_fraction_count_l3605_360593

theorem square_fraction_count : 
  ∃! n : ℤ, (∃ k : ℤ, 30 - 2*n ≠ 0 ∧ n/(30 - 2*n) = k^2 ∧ n/(30 - 2*n) ≥ 0) :=
by sorry

end square_fraction_count_l3605_360593


namespace circles_intersect_and_common_chord_l3605_360577

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 6 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 6 = 0

-- Define the intersection of the circles
def intersect : Prop := ∃ x y : ℝ, circle1 x y ∧ circle2 x y

-- Define the common chord equation
def commonChord (x y : ℝ) : Prop := 3*x - 2*y = 0

theorem circles_intersect_and_common_chord :
  intersect ∧ (∀ x y : ℝ, circle1 x y ∧ circle2 x y → commonChord x y) :=
sorry

end circles_intersect_and_common_chord_l3605_360577


namespace quadratic_inequality_range_l3605_360579

/-- Given a quadratic function f(x) = 2x^2 + bx + c with solution set (0, 2) for f(x) < 0,
    if f(x) + t ≥ 2 holds for all real x, then t ≥ 4 -/
theorem quadratic_inequality_range (b c t : ℝ) : 
  (∀ x, x ∈ Set.Ioo 0 2 ↔ 2*x^2 + b*x + c < 0) →
  (∀ x, 2*x^2 + b*x + c + t ≥ 2) →
  t ≥ 4 := by
  sorry


end quadratic_inequality_range_l3605_360579


namespace sum_of_sequence_l3605_360505

def arithmetic_sequence : List ℕ := [81, 83, 85, 87, 89, 91, 93, 95, 97, 99]

theorem sum_of_sequence : 
  2 * (arithmetic_sequence.sum) = 1800 := by
  sorry

end sum_of_sequence_l3605_360505


namespace purchase_price_calculation_l3605_360545

theorem purchase_price_calculation (markup : ℝ) (overhead_percentage : ℝ) (net_profit : ℝ) :
  markup = 45 ∧ 
  overhead_percentage = 0.20 ∧ 
  net_profit = 12 →
  ∃ purchase_price : ℝ, 
    markup = overhead_percentage * purchase_price + net_profit ∧
    purchase_price = 165 := by
  sorry

end purchase_price_calculation_l3605_360545


namespace alla_boris_meeting_l3605_360558

/-- The number of streetlights -/
def total_streetlights : ℕ := 400

/-- Alla's position when the snapshot is taken -/
def alla_snapshot : ℕ := 55

/-- Boris's position when the snapshot is taken -/
def boris_snapshot : ℕ := 321

/-- The meeting point of Alla and Boris -/
def meeting_point : ℕ := 163

theorem alla_boris_meeting :
  ∀ (v_a v_b : ℝ), v_a > 0 → v_b > 0 →
  (alla_snapshot - 1 : ℝ) / v_a = (total_streetlights - boris_snapshot : ℝ) / v_b →
  (meeting_point - 1 : ℝ) / v_a = (total_streetlights - meeting_point : ℝ) / v_b :=
by sorry

end alla_boris_meeting_l3605_360558


namespace equality_of_abc_l3605_360504

theorem equality_of_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a^2 * (b + c - a) = b^2 * (c + a - b) ∧ b^2 * (c + a - b) = c^2 * (a + b - c)) :
  a = b ∧ b = c := by
  sorry

end equality_of_abc_l3605_360504


namespace symmetric_complex_numbers_l3605_360581

/-- Two complex numbers are symmetric with respect to the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal. -/
def symmetric_wrt_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

/-- If z₁ and z₂ are complex numbers symmetric with respect to the imaginary axis,
    and z₁ = 2 + i, then z₂ = -2 + i. -/
theorem symmetric_complex_numbers (z₁ z₂ : ℂ) 
    (h_sym : symmetric_wrt_imaginary_axis z₁ z₂) 
    (h_z₁ : z₁ = 2 + I) : 
    z₂ = -2 + I := by
  sorry

end symmetric_complex_numbers_l3605_360581


namespace find_p_l3605_360557

theorem find_p (a b p : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : a^2 - 5*p*a + 2*p^3 = 0)
  (h2 : b^2 - 5*p*b + 2*p^3 = 0)
  (h3 : ∃! x, x^2 - a*x + b = 0) :
  p = 3 := by sorry

end find_p_l3605_360557


namespace distance_to_center_is_five_l3605_360586

/-- A square with side length 10 and a circle passing through two opposite vertices
    and tangent to one side -/
structure SquareWithCircle where
  /-- The side length of the square -/
  sideLength : ℝ
  /-- The circle passes through two opposite vertices -/
  circlePassesThroughOppositeVertices : Bool
  /-- The circle is tangent to one side -/
  circleTangentToSide : Bool

/-- The distance from the center of the circle to a vertex of the square -/
def distanceToCenterFromVertex (s : SquareWithCircle) : ℝ := sorry

/-- Theorem stating that the distance from the center of the circle to a vertex is 5 -/
theorem distance_to_center_is_five (s : SquareWithCircle) 
  (h1 : s.sideLength = 10)
  (h2 : s.circlePassesThroughOppositeVertices = true)
  (h3 : s.circleTangentToSide = true) : 
  distanceToCenterFromVertex s = 5 := by sorry

end distance_to_center_is_five_l3605_360586


namespace product_ends_with_three_zeros_l3605_360509

theorem product_ends_with_three_zeros :
  ∃ n : ℕ, 350 * 60 = n * 1000 ∧ n % 10 ≠ 0 :=
by sorry

end product_ends_with_three_zeros_l3605_360509


namespace circle_passes_through_fixed_point_l3605_360518

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the line x = -1
def line (x : ℝ) : Prop := x = -1

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define when a circle is tangent to a line
def is_tangent_to_line (c : Circle) (l : ℝ → Prop) : Prop :=
  abs (c.center.1 - (-1)) = c.radius

-- Define when a point is on a circle
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- The main theorem
theorem circle_passes_through_fixed_point :
  ∀ c : Circle,
    parabola c.center →
    is_tangent_to_line c line →
    point_on_circle (1, 0) c :=
sorry

end circle_passes_through_fixed_point_l3605_360518


namespace car_dealership_silver_percentage_l3605_360520

theorem car_dealership_silver_percentage
  (initial_cars : ℕ)
  (initial_silver_percentage : ℚ)
  (new_shipment : ℕ)
  (new_non_silver_percentage : ℚ)
  (h1 : initial_cars = 40)
  (h2 : initial_silver_percentage = 1/5)
  (h3 : new_shipment = 80)
  (h4 : new_non_silver_percentage = 7/20)
  : (initial_silver_percentage * initial_cars + (1 - new_non_silver_percentage) * new_shipment) / (initial_cars + new_shipment) = 1/2 := by
  sorry

end car_dealership_silver_percentage_l3605_360520


namespace simplest_fraction_sum_l3605_360511

theorem simplest_fraction_sum (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 0.375 ∧ 
  ∀ (c d : ℕ+), (c : ℚ) / (d : ℚ) = 0.375 → a ≤ c ∧ b ≤ d → 
  a + b = 11 := by
sorry

end simplest_fraction_sum_l3605_360511


namespace four_books_equals_one_kg_l3605_360572

/-- Proves that 4 books weighing 250 grams each is equal to 1 kilogram -/
theorem four_books_equals_one_kg (book_weight : ℕ) (kg_in_grams : ℕ) : 
  book_weight = 250 → kg_in_grams = 1000 → 4 * book_weight = kg_in_grams := by
  sorry

end four_books_equals_one_kg_l3605_360572


namespace angle_is_2pi_3_l3605_360575

open Real

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_is_2pi_3 (a b : ℝ × ℝ) :
  b.1 * (a.1 + b.1) + b.2 * (a.2 + b.2) = 3 →
  a.1^2 + a.2^2 = 1 →
  b.1^2 + b.2^2 = 4 →
  angle_between_vectors a b = 2 * π / 3 := by sorry

end angle_is_2pi_3_l3605_360575


namespace equilateral_triangle_area_perimeter_ratio_l3605_360597

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 12 is √3 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 12
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 := by sorry

end equilateral_triangle_area_perimeter_ratio_l3605_360597


namespace base_three_sum_l3605_360553

/-- Represents a number in base 3 --/
def BaseThree : Type := List Nat

/-- Converts a base 3 number to its decimal representation --/
def toDecimal (n : BaseThree) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The theorem to prove --/
theorem base_three_sum :
  let a : BaseThree := [2]
  let b : BaseThree := [0, 2, 1]
  let c : BaseThree := [1, 2, 0, 2]
  let d : BaseThree := [2, 0, 1, 1]
  let result : BaseThree := [2, 2, 2, 2]
  toDecimal a + toDecimal b + toDecimal c + toDecimal d = toDecimal result :=
by sorry

end base_three_sum_l3605_360553


namespace geometric_sequence_common_ratio_l3605_360537

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a1a3 : a 1 * a 3 = 36)
  (h_a4 : a 4 = 54) :
  ∃ q : ℝ, q = 3 ∧ ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end geometric_sequence_common_ratio_l3605_360537


namespace f_properties_l3605_360542

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x + a else -x - 2 * a

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (f (-3) 10 = -4 ∧ f (-3) (f (-3) 10) = -11) ∧
  (∀ b : ℝ, b ≠ 0 → (f b (1 - b) = f b (1 + b) ↔ b = -3/4)) := by
  sorry

end f_properties_l3605_360542


namespace batsman_average_increase_l3605_360517

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  inningsPlayed : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the increase in average after a new innings -/
def averageIncrease (prevStats : BatsmanStats) (newInningRuns : ℕ) : ℚ :=
  let newAverage := (prevStats.totalRuns + newInningRuns : ℚ) / (prevStats.inningsPlayed + 1 : ℚ)
  newAverage - prevStats.average

/-- Theorem: The increase in the batsman's average is 2 runs per inning -/
theorem batsman_average_increase :
  ∀ (prevStats : BatsmanStats),
    prevStats.inningsPlayed = 16 →
    averageIncrease prevStats 50 = 18 - prevStats.average →
    averageIncrease prevStats 50 = 2 := by
  sorry

end batsman_average_increase_l3605_360517


namespace line_slope_intercept_product_l3605_360529

/-- Given a line passing through points (-2, 3) and (3, -2), 
    the product of the square of its slope and its y-intercept equals 1 -/
theorem line_slope_intercept_product : 
  ∀ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b) →  -- Line equation
    (3 = m * (-2) + b) →          -- Point (-2, 3) satisfies the equation
    (-2 = m * 3 + b) →            -- Point (3, -2) satisfies the equation
    m^2 * b = 1 := by
  sorry

end line_slope_intercept_product_l3605_360529


namespace no_sequence_satisfying_inequality_l3605_360559

theorem no_sequence_satisfying_inequality :
  ¬ ∃ (α : ℝ) (a : ℕ → ℝ), 
    (0 < α ∧ α < 1) ∧
    (∀ n, 0 < a n) ∧
    (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) :=
by sorry

end no_sequence_satisfying_inequality_l3605_360559


namespace unique_prime_pair_l3605_360532

theorem unique_prime_pair : ∃! (p q : ℕ), 
  Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 3 ∧ q ≠ 3 ∧
  (∀ α : ℤ, (α ^ (3 * p * q) - α) % (3 * p * q) = 0) ∧
  p = 11 ∧ q = 17 := by
sorry

end unique_prime_pair_l3605_360532


namespace friends_earrings_count_l3605_360525

def total_earrings (bella_earrings monica_earrings rachel_earrings olivia_earrings : ℕ) : ℕ :=
  bella_earrings + monica_earrings + rachel_earrings + olivia_earrings

theorem friends_earrings_count :
  ∀ (bella_earrings monica_earrings rachel_earrings olivia_earrings : ℕ),
    bella_earrings = 10 →
    bella_earrings = monica_earrings / 4 →
    monica_earrings = 2 * rachel_earrings →
    olivia_earrings = bella_earrings + monica_earrings + rachel_earrings + 5 →
    total_earrings bella_earrings monica_earrings rachel_earrings olivia_earrings = 145 :=
by
  sorry

#check friends_earrings_count

end friends_earrings_count_l3605_360525


namespace k_value_l3605_360524

def length (k : ℕ) : ℕ :=
  (Nat.factors k).length

theorem k_value (k : ℕ) (h1 : k > 1) (h2 : length k = 4) (h3 : k = 2 * 2 * 2 * 3) :
  k = 24 := by
  sorry

end k_value_l3605_360524


namespace student_age_proof_l3605_360549

theorem student_age_proof (n : ℕ) (initial_avg : ℚ) (new_avg : ℚ) (teacher_age : ℕ) 
  (h1 : n = 30)
  (h2 : initial_avg = 10)
  (h3 : new_avg = 11)
  (h4 : teacher_age = 41) :
  ∃ (student_age : ℕ), 
    (n : ℚ) * initial_avg - student_age + teacher_age = (n : ℚ) * new_avg ∧ 
    student_age = 11 := by
  sorry

end student_age_proof_l3605_360549


namespace decreasing_at_half_implies_a_le_two_l3605_360500

/-- A quadratic function f(x) = -2x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := -2 * x^2 + a * x + 1

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := -4 * x + a

theorem decreasing_at_half_implies_a_le_two (a : ℝ) :
  (f_deriv a (1/2) ≤ 0) → a ≤ 2 := by
  sorry

#check decreasing_at_half_implies_a_le_two

end decreasing_at_half_implies_a_le_two_l3605_360500


namespace composite_function_evaluation_l3605_360516

def f (x : ℝ) : ℝ := 2 * x + 4

def g (x : ℝ) : ℝ := 5 * x + 2

theorem composite_function_evaluation : f (g (f 3)) = 108 := by
  sorry

end composite_function_evaluation_l3605_360516


namespace possible_sum_less_than_100_l3605_360538

/-- Represents a team in the tournament -/
structure Team :=
  (id : Nat)
  (score : Nat)

/-- Represents the tournament -/
structure Tournament :=
  (teams : List Team)
  (num_teams : Nat)
  (num_games : Nat)

/-- The scoring system for the tournament -/
def scoring_system (winner_rank : Nat) (loser_rank : Nat) : Nat :=
  if winner_rank ≤ 5 then 3 else 2

/-- The theorem stating that it's possible for the sum of scores to be less than 100 -/
theorem possible_sum_less_than_100 (t : Tournament) :
  t.num_teams = 10 →
  t.num_games = (t.num_teams * (t.num_teams - 1)) / 2 →
  ∃ (scores : List Nat), 
    scores.length = t.num_teams ∧ 
    scores.sum < 100 ∧
    (∀ (i j : Nat), i < j → j < t.num_teams → 
      ∃ (points : Nat), points ≤ (scoring_system (i + 1) (j + 1)) ∧
        (scores.get! i + scores.get! j = points)) :=
sorry

end possible_sum_less_than_100_l3605_360538


namespace parabola_normal_intersection_l3605_360539

/-- Given a parabola y = x^2, for any point (x₀, y₀) on the parabola,
    if the normal line at this point intersects the y-axis at (0, y₁),
    then y₁ - y₀ = 1/2 -/
theorem parabola_normal_intersection (x₀ y₀ y₁ : ℝ) : 
  y₀ = x₀^2 →  -- point (x₀, y₀) is on the parabola
  (∃ k : ℝ, k * (x - x₀) = y - y₀ ∧  -- equation of the normal line
            k = -(2 * x₀)⁻¹ ∧        -- slope of the normal line
            y₁ = k * (-x₀) + y₀) →   -- y₁ is the y-intercept of the normal line
  y₁ - y₀ = 1/2 := by
sorry

end parabola_normal_intersection_l3605_360539


namespace patients_arrangement_exists_l3605_360560

theorem patients_arrangement_exists :
  ∃ (cow she_wolf beetle worm : ℝ),
    0 ≤ cow ∧ cow < she_wolf ∧ she_wolf < beetle ∧ beetle < worm ∧ worm = 6 ∧
    she_wolf - cow = 1 ∧
    beetle - cow = 2 ∧
    (she_wolf - cow) + (beetle - she_wolf) + (worm - beetle) = 7 ∧
    (beetle - cow) + (worm - beetle) = 8 :=
by
  sorry

end patients_arrangement_exists_l3605_360560


namespace family_park_cost_l3605_360528

/-- Calculates the total cost for a family to visit a park and one attraction -/
def total_cost (park_fee : ℕ) (child_attraction_fee : ℕ) (adult_attraction_fee : ℕ) 
                (num_children : ℕ) (num_parents : ℕ) (num_grandparents : ℕ) : ℕ :=
  let total_family_members := num_children + num_parents + num_grandparents
  let total_adults := num_parents + num_grandparents
  let park_cost := total_family_members * park_fee
  let children_attraction_cost := num_children * child_attraction_fee
  let adult_attraction_cost := total_adults * adult_attraction_fee
  park_cost + children_attraction_cost + adult_attraction_cost

/-- Theorem: The total cost for the specified family composition is $55 -/
theorem family_park_cost : 
  total_cost 5 2 4 4 2 1 = 55 := by
  sorry

end family_park_cost_l3605_360528


namespace interest_difference_approximation_l3605_360544

/-- Calculates the balance after compound interest --/
def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- Calculates the balance difference between two accounts --/
def balance_difference (
  principal : ℝ)
  (rate1 : ℝ) (periods1 : ℕ)
  (rate2 : ℝ) (periods2 : ℕ) : ℝ :=
  compound_interest principal rate1 periods1 - compound_interest principal rate2 periods2

theorem interest_difference_approximation :
  let principal := 10000
  let rate_alice := 0.03  -- 6% / 2 for semiannual compounding
  let periods_alice := 20  -- 2 * 10 years
  let rate_bob := 0.04
  let periods_bob := 10
  abs (balance_difference principal rate_alice periods_alice rate_bob periods_bob - 3259) < 1 := by
  sorry

#eval balance_difference 10000 0.03 20 0.04 10

end interest_difference_approximation_l3605_360544


namespace lcm_of_165_and_396_l3605_360555

theorem lcm_of_165_and_396 : Nat.lcm 165 396 = 1980 := by
  sorry

end lcm_of_165_and_396_l3605_360555


namespace todds_initial_money_l3605_360507

/-- Represents the problem of calculating Todd's initial amount of money -/
theorem todds_initial_money (num_candies : ℕ) (candy_cost : ℕ) (money_left : ℕ) : 
  num_candies = 4 → candy_cost = 2 → money_left = 12 → 
  num_candies * candy_cost + money_left = 20 := by
  sorry

end todds_initial_money_l3605_360507


namespace quiz_competition_order_l3605_360578

theorem quiz_competition_order (A B C : ℕ) 
  (h1 : A + B = 2 * C)
  (h2 : 3 * A > 3 * B + 3 * C + 10)
  (h3 : 3 * B = 3 * C + 5)
  (h4 : A > 0 ∧ B > 0 ∧ C > 0) :
  C > A ∧ A > B := by
  sorry

end quiz_competition_order_l3605_360578


namespace nine_qualified_products_possible_l3605_360536

/-- The probability of success (pass rate) -/
def p : ℝ := 0.9

/-- The number of trials (products inspected) -/
def n : ℕ := 10

/-- The number of successes (qualified products) we're interested in -/
def k : ℕ := 9

/-- The binomial probability of k successes in n trials with probability p -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

theorem nine_qualified_products_possible : binomialProbability n k p > 0 := by
  sorry

end nine_qualified_products_possible_l3605_360536


namespace factorization_proof_l3605_360565

theorem factorization_proof (a : ℝ) : 2*a - 2*a^3 = 2*a*(1+a)*(1-a) := by
  sorry

end factorization_proof_l3605_360565


namespace arithmetic_sequence_sum_l3605_360506

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 4 + a 5 + a 6 = 27) →
  (a 1 + a 9 = 18) := by
sorry

end arithmetic_sequence_sum_l3605_360506


namespace perpendicular_lines_l3605_360592

theorem perpendicular_lines (a : ℝ) : 
  (∃ (x y : ℝ), y = a * x - 2) ∧ 
  (∃ (x y : ℝ), y = (a + 2) * x + 1) ∧ 
  (a * (a + 2) = -1) → 
  a = -1 := by
sorry

end perpendicular_lines_l3605_360592


namespace _l3605_360584

def main_theorem (f : ℝ → ℝ) (h1 : ∀ p q, f (p + q) = f p * f q) (h2 : f 1 = 3) : 
  (f 1 ^ 2 + f 2) / f 1 + (f 2 ^ 2 + f 4) / f 3 + (f 3 ^ 2 + f 6) / f 5 + 
  (f 4 ^ 2 + f 8) / f 7 + (f 5 ^ 2 + f 10) / f 9 = 30 := by
  sorry

end _l3605_360584


namespace arithmetic_progression_with_squares_is_integer_l3605_360583

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_progression (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A sequence contains the squares of its first three terms. -/
def contains_first_three_squares (a : ℕ → ℚ) : Prop :=
  ∃ k₁ k₂ k₃ : ℕ, a k₁ = (a 1)^2 ∧ a k₂ = (a 2)^2 ∧ a k₃ = (a 3)^2

/-- If an arithmetic progression contains the squares of its first three terms,
    then all terms in the progression are integers. -/
theorem arithmetic_progression_with_squares_is_integer
  (a : ℕ → ℚ)
  (h₁ : is_arithmetic_progression a)
  (h₂ : contains_first_three_squares a) :
  ∀ n : ℕ, ∃ k : ℤ, a n = k :=
sorry

end arithmetic_progression_with_squares_is_integer_l3605_360583


namespace expression_equality_l3605_360570

theorem expression_equality : (481 * 7 + 426 * 5)^3 - 4 * (481 * 7) * (426 * 5) = 166021128033 := by
  sorry

end expression_equality_l3605_360570


namespace problem_solution_l3605_360514

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- Define the theorem
theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  -- Part 1: Solution set of f(x) + f(2x+1) ≥ 6
  {x : ℝ | f x + f (2*x + 1) ≥ 6} = Set.Iic (-1) ∪ Set.Ici 3 ∧
  -- Part 2: Range of m given the condition
  ∀ m : ℝ, (∀ x : ℝ, f (x - m) - f (-x) ≤ 4/a + 1/b) → -13 ≤ m ∧ m ≤ 5 := by
  sorry

end problem_solution_l3605_360514


namespace speed_equivalence_l3605_360541

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in meters per second -/
def speed_mps : ℝ := 115.00919999999999

/-- The calculated speed in kilometers per hour -/
def speed_kmph : ℝ := 414.03312

/-- Theorem stating that the given speed in m/s is equivalent to the calculated speed in km/h -/
theorem speed_equivalence : speed_mps * mps_to_kmph = speed_kmph := by
  sorry

end speed_equivalence_l3605_360541


namespace classroom_desks_proof_l3605_360599

/-- The number of rows in the classroom -/
def num_rows : ℕ := 8

/-- The number of desks in the first row -/
def first_row_desks : ℕ := 10

/-- The maximum number of students that can be seated -/
def max_students : ℕ := 136

/-- The number of additional desks in each subsequent row -/
def additional_desks : ℕ := 2

/-- Calculates the total number of desks in the classroom -/
def total_desks (n : ℕ) : ℕ :=
  num_rows * first_row_desks + (num_rows - 1) * num_rows * n / 2

theorem classroom_desks_proof :
  total_desks additional_desks = max_students :=
sorry

end classroom_desks_proof_l3605_360599


namespace min_value_trig_expression_l3605_360574

theorem min_value_trig_expression (θ : Real) (h : θ ∈ Set.Ioo 0 (π / 2)) :
  1 / (Real.sin θ)^2 + 4 / (Real.cos θ)^2 ≥ 9 := by
  sorry

end min_value_trig_expression_l3605_360574


namespace intersection_line_property_l3605_360569

-- Define the circles
def circle_ω₁ : Set (ℝ × ℝ) := {p | (p.1^2 + (p.2 - 6)^2) = 900}
def circle_ω₂ : Set (ℝ × ℝ) := {p | ((p.1 - 20)^2 + p.2^2) = 900}

-- Define the intersection points
def X : ℝ × ℝ := sorry
def Y : ℝ × ℝ := sorry

-- Define the line passing through X and Y
def m : ℝ := sorry
def b : ℝ := sorry

-- Theorem statement
theorem intersection_line_property :
  X ∈ circle_ω₁ ∧ X ∈ circle_ω₂ ∧
  Y ∈ circle_ω₁ ∧ Y ∈ circle_ω₂ ∧
  (∀ p : ℝ × ℝ, p.2 = m * p.1 + b ↔ (p = X ∨ p = Y)) →
  100 * m + b = 303 := by sorry

end intersection_line_property_l3605_360569


namespace tea_trader_profit_percentage_tea_trader_profit_is_35_percent_l3605_360566

/-- Calculates the profit percentage for a tea trader --/
theorem tea_trader_profit_percentage 
  (tea1_weight : ℝ) (tea1_cost : ℝ) 
  (tea2_weight : ℝ) (tea2_cost : ℝ) 
  (sale_price : ℝ) : ℝ :=
  let total_weight := tea1_weight + tea2_weight
  let total_cost := tea1_weight * tea1_cost + tea2_weight * tea2_cost
  let total_sale := total_weight * sale_price
  let profit := total_sale - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage

/-- Proves that the profit percentage is 35% for the given scenario --/
theorem tea_trader_profit_is_35_percent : 
  tea_trader_profit_percentage 80 15 20 20 21.6 = 35 := by
  sorry

end tea_trader_profit_percentage_tea_trader_profit_is_35_percent_l3605_360566


namespace beach_trip_ratio_l3605_360571

theorem beach_trip_ratio (total : ℕ) (remaining : ℕ) (beach : ℕ) :
  total = 1000 →
  remaining = (total - beach) / 2 →
  remaining = 250 →
  beach / total = 1 / 2 := by
sorry

end beach_trip_ratio_l3605_360571


namespace focal_length_determination_l3605_360540

/-- Represents a converging lens with a right isosceles triangle -/
structure LensSystem where
  focalLength : ℝ
  triangleArea : ℝ
  imageArea : ℝ

/-- The conditions of the lens system -/
def validLensSystem (s : LensSystem) : Prop :=
  s.triangleArea = 8 ∧ s.imageArea = s.triangleArea / 2

/-- The theorem statement -/
theorem focal_length_determination (s : LensSystem) 
  (h : validLensSystem s) : s.focalLength = 2 := by
  sorry

end focal_length_determination_l3605_360540


namespace tangent_line_m_squared_l3605_360576

/-- A line that intersects an ellipse and a circle exactly once -/
structure TangentLine where
  m : ℝ
  -- Line equation: y = mx + 2
  line : ℝ → ℝ := fun x => m * x + 2
  -- Ellipse equation: x^2 + 9y^2 = 9
  ellipse : ℝ × ℝ → Prop := fun (x, y) => x^2 + 9 * y^2 = 9
  -- Circle equation: x^2 + y^2 = 4
  circle : ℝ × ℝ → Prop := fun (x, y) => x^2 + y^2 = 4
  -- The line intersects both the ellipse and the circle exactly once
  h_tangent_ellipse : ∃! x, ellipse (x, line x)
  h_tangent_circle : ∃! x, circle (x, line x)

/-- The theorem stating that m^2 = 1/3 for a line tangent to both the ellipse and circle -/
theorem tangent_line_m_squared (l : TangentLine) : l.m^2 = 1/3 := by
  sorry


end tangent_line_m_squared_l3605_360576


namespace tan_product_pi_ninths_l3605_360595

theorem tan_product_pi_ninths : Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end tan_product_pi_ninths_l3605_360595


namespace sum_of_exponents_15_factorial_l3605_360521

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largestPerfectSquareDivisor (n : ℕ) : ℕ :=
  sorry

def sumOfExponents (n : ℕ) : ℕ :=
  sorry

theorem sum_of_exponents_15_factorial :
  sumOfExponents (largestPerfectSquareDivisor (factorial 15).sqrt) = 10 :=
sorry

end sum_of_exponents_15_factorial_l3605_360521


namespace sqrt_sum_equals_2sqrt14_l3605_360591

theorem sqrt_sum_equals_2sqrt14 : 
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 14 := by
  sorry

end sqrt_sum_equals_2sqrt14_l3605_360591


namespace approx_value_of_625_power_l3605_360552

theorem approx_value_of_625_power (ε : Real) (hε : ε > 0) :
  ∃ (x : Real), abs (x - ((625 : Real)^(0.2 : Real) * (625 : Real)^(0.12 : Real))) < ε ∧
                 abs (x - 17.15) < ε :=
sorry

end approx_value_of_625_power_l3605_360552


namespace probability_at_least_one_boy_one_girl_l3605_360522

theorem probability_at_least_one_boy_one_girl :
  let p_boy : ℝ := 1 / 2
  let p_girl : ℝ := 1 - p_boy
  let num_children : ℕ := 4
  let p_all_boys : ℝ := p_boy ^ num_children
  let p_all_girls : ℝ := p_girl ^ num_children
  p_all_boys + p_all_girls = 1 / 8 →
  1 - (p_all_boys + p_all_girls) = 7 / 8 :=
by sorry

end probability_at_least_one_boy_one_girl_l3605_360522


namespace fraction_equality_implies_c_geq_one_l3605_360590

theorem fraction_equality_implies_c_geq_one 
  (a b : ℕ+) 
  (c : ℝ) 
  (h_c_pos : c > 0) 
  (h_eq : (a + 1 : ℝ) / (b + c) = (b : ℝ) / a) : 
  c ≥ 1 := by
sorry

end fraction_equality_implies_c_geq_one_l3605_360590


namespace arithmetic_sequence_sum_l3605_360515

/-- Given an arithmetic sequence {a_n} where a_4 = 4, prove that S_7 = 28 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = n / 2 * (a 1 + a n)) →  -- Definition of S_n
  (∀ k m, a (k + m) - a k = m * (a 2 - a 1)) →  -- Definition of arithmetic sequence
  a 4 = 4 →  -- Given condition
  S 7 = 28 := by sorry

end arithmetic_sequence_sum_l3605_360515


namespace highest_winner_number_l3605_360548

/-- Represents a single-elimination tournament with wrestlers having qualification numbers. -/
structure WrestlingTournament where
  num_wrestlers : ℕ
  can_win : ℕ → ℕ → Prop

/-- The conditions of our specific tournament. -/
def our_tournament : WrestlingTournament where
  num_wrestlers := 512
  can_win := fun a b => b ≤ a + 2

/-- The number of rounds in a single-elimination tournament. -/
def num_rounds (t : WrestlingTournament) : ℕ :=
  Nat.log 2 t.num_wrestlers

/-- The highest possible qualification number for the winner. -/
def max_winner_number (t : WrestlingTournament) : ℕ :=
  1 + 2 * num_rounds t

theorem highest_winner_number (t : WrestlingTournament) :
  t = our_tournament →
  max_winner_number t = 18 :=
by sorry

end highest_winner_number_l3605_360548


namespace triangle_area_l3605_360587

/-- The area of a triangle with vertices at (2, -3), (8, 1), and (2, 3) is 18 square units. -/
theorem triangle_area : Real := by
  -- Define the vertices of the triangle
  let A : (ℝ × ℝ) := (2, -3)
  let B : (ℝ × ℝ) := (8, 1)
  let C : (ℝ × ℝ) := (2, 3)

  -- Calculate the area of the triangle
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

  -- Prove that the area is equal to 18
  sorry

end triangle_area_l3605_360587


namespace area_of_APEG_l3605_360502

/-- Two squares with side lengths 8 and 6 placed side by side -/
structure TwoSquares where
  squareABCD : Set (ℝ × ℝ)
  squareBEFG : Set (ℝ × ℝ)
  sideAB : ℝ
  sideBE : ℝ
  B : ℝ × ℝ
  common_point : B ∈ squareABCD ∩ squareBEFG
  sideAB_length : sideAB = 8
  sideBE_length : sideBE = 6

/-- The quadrilateral APEG formed by the intersection of DE and BG -/
def quadrilateralAPEG (ts : TwoSquares) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The main theorem: The area of quadrilateral APEG is 18 -/
theorem area_of_APEG (ts : TwoSquares) : area (quadrilateralAPEG ts) = 18 :=
  sorry

end area_of_APEG_l3605_360502


namespace pauls_tips_amount_l3605_360554

def pauls_tips : ℕ := 14
def vinnies_earnings : ℕ := 30

theorem pauls_tips_amount :
  (∃ (p : ℕ), p = pauls_tips ∧ vinnies_earnings = p + 16) →
  pauls_tips = 14 := by
  sorry

end pauls_tips_amount_l3605_360554


namespace loss_recording_l3605_360580

/-- Records a financial transaction as a number, where profits are positive and losses are negative. -/
def recordTransaction (amount : ℤ) : ℤ := amount

/-- Given that a profit of 100 yuan is recorded as +100, prove that a loss of 50 yuan is recorded as -50. -/
theorem loss_recording (h : recordTransaction 100 = 100) : recordTransaction (-50) = -50 := by
  sorry

end loss_recording_l3605_360580


namespace output_for_15_l3605_360527

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 40 then
    step1 + 10
  else
    step1 - 7

theorem output_for_15 : function_machine 15 = 38 := by
  sorry

end output_for_15_l3605_360527


namespace minimum_bookmarks_l3605_360594

def is_divisible_by (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

theorem minimum_bookmarks : 
  ∀ (n : ℕ), n > 0 → 
  (is_divisible_by n 3 ∧ 
   is_divisible_by n 4 ∧ 
   is_divisible_by n 5 ∧ 
   is_divisible_by n 7 ∧ 
   is_divisible_by n 8) → 
  n ≥ 840 :=
by
  sorry

end minimum_bookmarks_l3605_360594


namespace sum_of_x_and_y_is_two_l3605_360530

theorem sum_of_x_and_y_is_two (x y : ℝ) (h : x^2 + y^2 = 10*x - 6*y - 34) : x + y = 2 := by
  sorry

end sum_of_x_and_y_is_two_l3605_360530


namespace union_equality_implies_a_equals_three_l3605_360567

-- Define the sets A and B
def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}

-- State the theorem
theorem union_equality_implies_a_equals_three (a : ℝ) : A ∪ B a = A → a = 3 := by
  sorry

end union_equality_implies_a_equals_three_l3605_360567


namespace probability_of_rolling_seven_l3605_360551

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := sides * sides

/-- The number of ways to roll a sum of 7 with two dice -/
def waysToRollSeven : ℕ := 6

/-- The probability of rolling a sum of 7 with two fair 6-sided dice -/
theorem probability_of_rolling_seven :
  (waysToRollSeven : ℚ) / totalOutcomes = 1 / 6 := by sorry

end probability_of_rolling_seven_l3605_360551


namespace imaginary_sum_equals_negative_one_l3605_360512

theorem imaginary_sum_equals_negative_one (i : ℂ) (hi : i^2 = -1) : 
  i^10 + i^20 + i^34 = -1 := by
  sorry

end imaginary_sum_equals_negative_one_l3605_360512


namespace subway_length_l3605_360519

/-- The length of a subway given its speed, time to cross a bridge, and bridge length. -/
theorem subway_length
  (speed : ℝ)  -- Speed of the subway in km/min
  (time : ℝ)   -- Time to cross the bridge in minutes
  (bridge_length : ℝ)  -- Length of the bridge in km
  (h1 : speed = 1.6)  -- The subway speed is 1.6 km/min
  (h2 : time = 3.25)  -- The time to cross the bridge is 3 min and 15 sec (3.25 min)
  (h3 : bridge_length = 4.85)  -- The bridge length is 4.85 km
  : (speed * time - bridge_length) * 1000 = 350 :=
by sorry

end subway_length_l3605_360519


namespace truth_telling_probability_l3605_360523

/-- The probability of two independent events occurring simultaneously -/
def simultaneous_probability (p_a p_b : ℝ) : ℝ := p_a * p_b

/-- Proof that given A speaks the truth 55% of the times and B speaks the truth 60% of the times, 
    the probability that they both tell the truth simultaneously is 0.33 -/
theorem truth_telling_probability : 
  let p_a : ℝ := 0.55
  let p_b : ℝ := 0.60
  simultaneous_probability p_a p_b = 0.33 := by
sorry

end truth_telling_probability_l3605_360523


namespace circle_and_max_distance_l3605_360564

-- Define the circle C
def Circle (a b r : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + (y - b)^2 = r^2}

-- Define the conditions for the circle
def CircleConditions (a b r : ℝ) : Prop :=
  3 * a - b = 0 ∧ 
  a ≥ 0 ∧ 
  |a - 4| = r ∧ 
  ((3 * a + 4 * b + 10)^2 / 25 + 12 = r^2)

-- Define points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-2, 0)

-- Define the distance squared function
def DistanceSquared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Theorem statement
theorem circle_and_max_distance :
  ∃ a b r : ℝ, 
    CircleConditions a b r → 
    (Circle a b r = Circle 0 0 4) ∧
    (∀ p ∈ Circle 0 0 4, DistanceSquared p A + DistanceSquared p B ≤ 38 + 8 * Real.sqrt 2) ∧
    (∃ p ∈ Circle 0 0 4, DistanceSquared p A + DistanceSquared p B = 38 + 8 * Real.sqrt 2) :=
  sorry

end circle_and_max_distance_l3605_360564


namespace sequence_problem_l3605_360573

def sequence_rule (x y z : ℕ) : Prop := z = 2 * (x + y)

theorem sequence_problem : 
  ∀ (a b c : ℕ), 
  sequence_rule 10 a 30 → 
  sequence_rule a 30 b → 
  sequence_rule 30 b c → 
  c = 200 := by
sorry

end sequence_problem_l3605_360573


namespace f_properties_l3605_360531

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (-2^x + a) / (2^x + 1)

theorem f_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 1 ∧
   (∀ x y, x < y → f a y < f a x) ∧
   (∀ k, (∀ t, 1 ≤ t ∧ t ≤ 3 → f a (t^2 - 2*t) + f a (2*t^2 - k) < 0) → k < -1/3)) :=
by sorry

end f_properties_l3605_360531


namespace arithmetic_sequence_sum_l3605_360501

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a, if a₁ + a₉ + a₂ + a₈ = 20, then a₃ + a₇ = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) 
    (sum_condition : a 1 + a 9 + a 2 + a 8 = 20) : 
  a 3 + a 7 = 10 := by
  sorry

end arithmetic_sequence_sum_l3605_360501


namespace amcb_paths_count_l3605_360582

/-- Represents the number of paths from one letter to the next -/
structure PathCount where
  a_to_m : Nat
  m_to_c : Nat
  c_to_b : Nat

/-- The configuration of the letter arrangement -/
structure LetterArrangement where
  central_a : Nat
  m_adjacent_to_a : Nat
  c_adjacent_to_m : Nat
  b_adjacent_to_c : Nat

/-- Calculates the total number of paths spelling "AMCB" -/
def total_paths (arrangement : LetterArrangement) : Nat :=
  arrangement.central_a * arrangement.m_adjacent_to_a * arrangement.c_adjacent_to_m * arrangement.b_adjacent_to_c

/-- The specific arrangement for this problem -/
def amcb_arrangement : LetterArrangement :=
  { central_a := 1
  , m_adjacent_to_a := 4
  , c_adjacent_to_m := 2
  , b_adjacent_to_c := 3 }

theorem amcb_paths_count :
  total_paths amcb_arrangement = 24 :=
sorry

end amcb_paths_count_l3605_360582


namespace root_product_l3605_360561

theorem root_product (x₁ x₂ : ℝ) (h₁ : x₁ * Real.log x₁ = 2006) (h₂ : x₂ * Real.exp x₂ = 2006) : 
  x₁ * x₂ = 2006 := by
sorry

end root_product_l3605_360561


namespace root_equation_value_l3605_360546

theorem root_equation_value (a b m : ℝ) : 
  a * m^2 + b * m + 5 = 0 → a * m^2 + b * m - 7 = -12 := by
  sorry

end root_equation_value_l3605_360546


namespace mendel_pea_experiment_l3605_360563

/-- Represents the genotype of a pea plant -/
inductive Genotype
| DD
| Dd
| dd

/-- Represents a generation of pea plants -/
structure Generation where
  DD_ratio : ℚ
  Dd_ratio : ℚ
  dd_ratio : ℚ
  sum_to_one : DD_ratio + Dd_ratio + dd_ratio = 1

/-- First generation with all Dd genotype -/
def first_gen : Generation where
  DD_ratio := 0
  Dd_ratio := 1
  dd_ratio := 0
  sum_to_one := by norm_num

/-- Function to calculate the next generation's ratios -/
def next_gen (g : Generation) : Generation where
  DD_ratio := g.DD_ratio^2 + g.DD_ratio * g.Dd_ratio + (g.Dd_ratio^2) / 4
  Dd_ratio := g.DD_ratio * g.Dd_ratio + g.Dd_ratio * g.dd_ratio + (g.Dd_ratio^2) / 2
  dd_ratio := g.dd_ratio^2 + g.dd_ratio * g.Dd_ratio + (g.Dd_ratio^2) / 4
  sum_to_one := by sorry

/-- Second generation -/
def second_gen : Generation := next_gen first_gen

/-- Third generation -/
def third_gen : Generation := next_gen second_gen

/-- Probability of dominant trait in a generation -/
def prob_dominant (g : Generation) : ℚ := g.DD_ratio + g.Dd_ratio

theorem mendel_pea_experiment :
  (third_gen.dd_ratio = 1/4) ∧
  (3 * (prob_dominant third_gen)^2 * (1 - prob_dominant third_gen) = 27/64) := by sorry

end mendel_pea_experiment_l3605_360563


namespace abcd_sum_l3605_360568

theorem abcd_sum (a b c d : ℝ) 
  (eq1 : a + b + c = 3)
  (eq2 : a + b + d = -2)
  (eq3 : a + c + d = 8)
  (eq4 : b + c + d = -1) :
  a * b + c * d = -190 / 9 := by
  sorry

end abcd_sum_l3605_360568


namespace multiplication_of_monomials_l3605_360562

theorem multiplication_of_monomials (a : ℝ) : 3 * a * (4 * a^2) = 12 * a^3 := by
  sorry

end multiplication_of_monomials_l3605_360562


namespace remaining_pokemon_cards_l3605_360588

/-- Theorem: Calculating remaining Pokemon cards after a sale --/
theorem remaining_pokemon_cards 
  (initial_cards : ℕ) 
  (sold_cards : ℕ) 
  (h1 : initial_cards = 676)
  (h2 : sold_cards = 224) :
  initial_cards - sold_cards = 452 :=
by
  sorry

end remaining_pokemon_cards_l3605_360588
