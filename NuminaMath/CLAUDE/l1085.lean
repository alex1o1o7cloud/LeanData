import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_l1085_108541

theorem system_solution :
  let eq1 (x y z : ℝ) := x^3 + y^3 + z^3 = 8
  let eq2 (x y z : ℝ) := x^2 + y^2 + z^2 = 22
  let eq3 (x y z : ℝ) := 1/x + 1/y + 1/z = -z/(x*y)
  ∀ (x y z : ℝ),
    ((x = 3 ∧ y = 2 ∧ z = -3) ∨
     (x = -3 ∧ y = 2 ∧ z = 3) ∨
     (x = 2 ∧ y = 3 ∧ z = -3) ∨
     (x = 2 ∧ y = -3 ∧ z = 3)) →
    (eq1 x y z ∧ eq2 x y z ∧ eq3 x y z) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1085_108541


namespace NUMINAMATH_CALUDE_solutions_for_14_solutions_for_0_1_solutions_for_neg_0_0544_l1085_108526

-- Define the equation
def f (x : ℝ) := (x - 1) * (2 * x - 3) * (3 * x - 4) * (6 * x - 5)

-- Theorem for a = 14
theorem solutions_for_14 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 14 ∧ f x₂ = 14 ∧
  ∀ x : ℝ, f x = 14 → x = x₁ ∨ x = x₂ :=
sorry

-- Theorem for a = 0.1
theorem solutions_for_0_1 :
  ∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
  f x₁ = 0.1 ∧ f x₂ = 0.1 ∧ f x₃ = 0.1 ∧ f x₄ = 0.1 ∧
  ∀ x : ℝ, f x = 0.1 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄ :=
sorry

-- Theorem for a = -0.0544
theorem solutions_for_neg_0_0544 :
  ∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
  f x₁ = -0.0544 ∧ f x₂ = -0.0544 ∧ f x₃ = -0.0544 ∧ f x₄ = -0.0544 ∧
  ∀ x : ℝ, f x = -0.0544 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄ :=
sorry

end NUMINAMATH_CALUDE_solutions_for_14_solutions_for_0_1_solutions_for_neg_0_0544_l1085_108526


namespace NUMINAMATH_CALUDE_opposite_of_negative_sqrt_two_l1085_108557

theorem opposite_of_negative_sqrt_two : -(-(Real.sqrt 2)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_sqrt_two_l1085_108557


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l1085_108510

theorem solve_quadratic_equation (x : ℝ) :
  2 * (x - 1)^2 = 8 → x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l1085_108510


namespace NUMINAMATH_CALUDE_sum_of_segments_l1085_108517

/-- Given a number line with points P at 3 and V at 33, and the line between them
    divided into six equal parts, the sum of the lengths of PS and TV is 25. -/
theorem sum_of_segments (P V Q R S T U : ℝ) : 
  P = 3 → V = 33 → 
  Q - P = R - Q → R - Q = S - R → S - R = T - S → T - S = U - T → U - T = V - U →
  (S - P) + (V - T) = 25 := by sorry

end NUMINAMATH_CALUDE_sum_of_segments_l1085_108517


namespace NUMINAMATH_CALUDE_initial_money_calculation_l1085_108570

/-- Calculates the initial amount of money given the spending pattern and final amount --/
theorem initial_money_calculation (final_amount : ℚ) : 
  final_amount = 500 →
  ∃ initial_amount : ℚ,
    initial_amount * (1 - 1/3) * (1 - 1/5) * (1 - 1/4) = final_amount ∧
    initial_amount = 1250 :=
by sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l1085_108570


namespace NUMINAMATH_CALUDE_triangle_inradius_l1085_108502

/-- Given a triangle with perimeter 60 cm and area 75 cm², its inradius is 2.5 cm. -/
theorem triangle_inradius (perimeter : ℝ) (area : ℝ) (inradius : ℝ) :
  perimeter = 60 ∧ area = 75 → inradius = 2.5 := by
  sorry

#check triangle_inradius

end NUMINAMATH_CALUDE_triangle_inradius_l1085_108502


namespace NUMINAMATH_CALUDE_average_cost_per_meter_l1085_108530

def silk_length : Real := 9.25
def silk_cost : Real := 416.25
def cotton_length : Real := 7.5
def cotton_cost : Real := 337.50
def wool_length : Real := 6
def wool_cost : Real := 378

def total_length : Real := silk_length + cotton_length + wool_length
def total_cost : Real := silk_cost + cotton_cost + wool_cost

theorem average_cost_per_meter : total_cost / total_length = 49.75 := by sorry

end NUMINAMATH_CALUDE_average_cost_per_meter_l1085_108530


namespace NUMINAMATH_CALUDE_combined_distance_is_1890_l1085_108591

/-- The combined swimming distance for Jamir, Sarah, and Julien for a week -/
def combined_swimming_distance (julien_distance : ℕ) : ℕ :=
  let sarah_distance := 2 * julien_distance
  let jamir_distance := sarah_distance + 20
  let days_in_week := 7
  (julien_distance + sarah_distance + jamir_distance) * days_in_week

/-- Theorem stating that the combined swimming distance for a week is 1890 meters -/
theorem combined_distance_is_1890 :
  combined_swimming_distance 50 = 1890 := by
  sorry

end NUMINAMATH_CALUDE_combined_distance_is_1890_l1085_108591


namespace NUMINAMATH_CALUDE_ball_distribution_problem_l1085_108564

def total_arrangements : ℕ := 90
def arrangements_with_1_and_2_together : ℕ := 18

theorem ball_distribution_problem :
  let n_balls : ℕ := 6
  let n_boxes : ℕ := 3
  let balls_per_box : ℕ := 2
  total_arrangements - arrangements_with_1_and_2_together = 72 :=
by sorry

end NUMINAMATH_CALUDE_ball_distribution_problem_l1085_108564


namespace NUMINAMATH_CALUDE_min_buses_for_field_trip_l1085_108586

/-- The minimum number of buses needed to transport students for a field trip. -/
def min_buses (total_students : ℕ) (bus_capacity : ℕ) (min_buses : ℕ) : ℕ :=
  max (min_buses) (((total_students + bus_capacity - 1) / bus_capacity) : ℕ)

/-- Theorem stating the minimum number of buses needed for the given conditions. -/
theorem min_buses_for_field_trip :
  min_buses 500 45 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_buses_for_field_trip_l1085_108586


namespace NUMINAMATH_CALUDE_complex_number_real_condition_l1085_108581

theorem complex_number_real_condition (a : ℝ) : 
  (2 * Complex.I - a / (1 - Complex.I)).im = 0 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_real_condition_l1085_108581


namespace NUMINAMATH_CALUDE_bridget_apples_l1085_108578

theorem bridget_apples (x : ℕ) : 
  (x / 2 - (x / 2) / 3 = 5) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_l1085_108578


namespace NUMINAMATH_CALUDE_triangle_center_distance_l1085_108507

/-- Given a triangle with circumradius R, inradius r, and distance d between
    the circumcenter and incenter, prove that d^2 = R^2 - 2Rr. -/
theorem triangle_center_distance (R r d : ℝ) (hR : R > 0) (hr : r > 0) (hd : d > 0) :
  d^2 = R^2 - 2*R*r := by
  sorry

end NUMINAMATH_CALUDE_triangle_center_distance_l1085_108507


namespace NUMINAMATH_CALUDE_aspirations_necessary_for_reaching_l1085_108589

-- Define the universe of discourse
variable (Person : Type)

-- Define predicates
variable (has_aspirations : Person → Prop)
variable (can_reach_extraordinary : Person → Prop)
variable (is_remote_dangerous : Person → Prop)
variable (few_venture : Person → Prop)

-- State the theorem
theorem aspirations_necessary_for_reaching :
  (∀ p : Person, is_remote_dangerous p → few_venture p) →
  (∀ p : Person, can_reach_extraordinary p → has_aspirations p) →
  (∀ p : Person, can_reach_extraordinary p → has_aspirations p) :=
by
  sorry


end NUMINAMATH_CALUDE_aspirations_necessary_for_reaching_l1085_108589


namespace NUMINAMATH_CALUDE_interest_period_calculation_l1085_108519

theorem interest_period_calculation 
  (initial_amount : ℝ) 
  (rate_A : ℝ) 
  (rate_B : ℝ) 
  (gain_B : ℝ) 
  (h1 : initial_amount = 2800)
  (h2 : rate_A = 0.15)
  (h3 : rate_B = 0.185)
  (h4 : gain_B = 294) :
  ∃ t : ℝ, t = 3 ∧ initial_amount * (rate_B - rate_A) * t = gain_B :=
sorry

end NUMINAMATH_CALUDE_interest_period_calculation_l1085_108519


namespace NUMINAMATH_CALUDE_linear_function_condition_passes_through_origin_l1085_108518

/-- A linear function of x with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := (2*m + 1)*x + m - 3

theorem linear_function_condition (m : ℝ) :
  (∀ x, ∃ y, f m x = y) ↔ m ≠ -1/2 :=
sorry

theorem passes_through_origin (m : ℝ) :
  f m 0 = 0 ↔ m = 3 :=
sorry

end NUMINAMATH_CALUDE_linear_function_condition_passes_through_origin_l1085_108518


namespace NUMINAMATH_CALUDE_abc_inequality_l1085_108542

/-- Given a = √2, b = √7 - √3, and c = √6 - √2, prove that a > c > b -/
theorem abc_inequality :
  let a : ℝ := Real.sqrt 2
  let b : ℝ := Real.sqrt 7 - Real.sqrt 3
  let c : ℝ := Real.sqrt 6 - Real.sqrt 2
  a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l1085_108542


namespace NUMINAMATH_CALUDE_order_combinations_l1085_108503

theorem order_combinations (num_drinks num_salads num_pizzas : ℕ) 
  (h1 : num_drinks = 3)
  (h2 : num_salads = 2)
  (h3 : num_pizzas = 5) :
  num_drinks * num_salads * num_pizzas = 30 := by
  sorry

end NUMINAMATH_CALUDE_order_combinations_l1085_108503


namespace NUMINAMATH_CALUDE_max_xy_value_l1085_108588

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 200) : x * y ≤ 348 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l1085_108588


namespace NUMINAMATH_CALUDE_octagon_side_length_l1085_108597

/-- The side length of a regular octagon with an area equal to the sum of the areas of three regular octagons with side lengths 3, 4, and 12 units is 13 units. -/
theorem octagon_side_length (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a = 3) → (b = 4) → (c = 12) →
  (d^2 = a^2 + b^2 + c^2) →
  d = 13 := by sorry

end NUMINAMATH_CALUDE_octagon_side_length_l1085_108597


namespace NUMINAMATH_CALUDE_range_of_m_l1085_108583

def is_hyperbola (m : ℝ) : Prop := (m + 2) * (m - 3) > 0

def no_positive_roots (m : ℝ) : Prop :=
  m = 0 ∨ (m ≠ 0 ∧ (∀ x : ℝ, x > 0 → m * x^2 + (m + 3) * x + 4 ≠ 0))

theorem range_of_m (m : ℝ) :
  (is_hyperbola m ∨ no_positive_roots m) ∧
  ¬(is_hyperbola m ∧ no_positive_roots m) →
  m < -2 ∨ (0 ≤ m ∧ m ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1085_108583


namespace NUMINAMATH_CALUDE_fraction_zero_solution_l1085_108505

theorem fraction_zero_solution (x : ℝ) : 
  (x^2 - 4) / (x + 2) = 0 ∧ x + 2 ≠ 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_solution_l1085_108505


namespace NUMINAMATH_CALUDE_rational_function_value_l1085_108548

-- Define the property of the function f
def satisfies_equation (f : ℚ → ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 0 → 4 * f (1 / x) + 3 * f x / x = x^3

-- State the theorem
theorem rational_function_value (f : ℚ → ℚ) (h : satisfies_equation f) : 
  f (-3) = -6565 / 189 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l1085_108548


namespace NUMINAMATH_CALUDE_brenda_banana_pudding_trays_l1085_108587

/-- Proof that Brenda can make 3 trays of banana pudding given the conditions --/
theorem brenda_banana_pudding_trays :
  ∀ (cookies_per_tray : ℕ) 
    (cookies_per_box : ℕ) 
    (cost_per_box : ℚ) 
    (total_spent : ℚ),
  cookies_per_tray = 80 →
  cookies_per_box = 60 →
  cost_per_box = 7/2 →
  total_spent = 14 →
  (total_spent / cost_per_box * cookies_per_box) / cookies_per_tray = 3 :=
by
  sorry

#check brenda_banana_pudding_trays

end NUMINAMATH_CALUDE_brenda_banana_pudding_trays_l1085_108587


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1085_108520

theorem polynomial_divisibility : ∀ (x : ℂ),
  (x^5 + x^4 + x^3 + x^2 + x + 1 = 0) →
  (x^55 + x^44 + x^33 + x^22 + x^11 + 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1085_108520


namespace NUMINAMATH_CALUDE_garden_perimeter_l1085_108515

/-- A rectangular garden with a diagonal of 20 meters and an area of 96 square meters has a perimeter of 49 meters. -/
theorem garden_perimeter : ∀ a b : ℝ,
  a > 0 → b > 0 →
  a^2 + b^2 = 20^2 →
  a * b = 96 →
  2 * (a + b) = 49 := by
sorry

end NUMINAMATH_CALUDE_garden_perimeter_l1085_108515


namespace NUMINAMATH_CALUDE_dusting_team_combinations_l1085_108563

theorem dusting_team_combinations (n : ℕ) (k : ℕ) : n = 5 → k = 3 → Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_dusting_team_combinations_l1085_108563


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1085_108501

theorem quadratic_inequality (x : ℝ) : -8 * x^2 + 6 * x - 1 < 0 ↔ 0.25 < x ∧ x < 0.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1085_108501


namespace NUMINAMATH_CALUDE_mean_median_difference_l1085_108555

/-- Represents the score distribution on a math test -/
structure ScoreDistribution where
  score65 : ℝ
  score75 : ℝ
  score85 : ℝ
  score92 : ℝ
  score98 : ℝ
  sum_to_one : score65 + score75 + score85 + score92 + score98 = 1

/-- Calculates the mean score given a score distribution -/
def mean_score (sd : ScoreDistribution) : ℝ :=
  65 * sd.score65 + 75 * sd.score75 + 85 * sd.score85 + 92 * sd.score92 + 98 * sd.score98

/-- Determines the median score given a score distribution -/
noncomputable def median_score (sd : ScoreDistribution) : ℝ :=
  if sd.score65 + sd.score75 > 0.5 then 75
  else if sd.score65 + sd.score75 + sd.score85 > 0.5 then 85
  else if sd.score65 + sd.score75 + sd.score85 + sd.score92 > 0.5 then 92
  else 98

/-- Theorem stating that the absolute difference between mean and median is 1.05 -/
theorem mean_median_difference (sd : ScoreDistribution) 
  (h1 : sd.score65 = 0.15)
  (h2 : sd.score75 = 0.20)
  (h3 : sd.score85 = 0.30)
  (h4 : sd.score92 = 0.10) :
  |mean_score sd - median_score sd| = 1.05 := by
  sorry


end NUMINAMATH_CALUDE_mean_median_difference_l1085_108555


namespace NUMINAMATH_CALUDE_paco_initial_salty_cookies_l1085_108504

/-- Represents the number of cookies Paco has --/
structure CookieCount where
  salty : ℕ
  sweet : ℕ

/-- The problem of determining Paco's initial salty cookie count --/
theorem paco_initial_salty_cookies 
  (initial : CookieCount) 
  (eaten : CookieCount) 
  (final : CookieCount) : 
  (initial.sweet = 17) →
  (eaten.sweet = 14) →
  (eaten.salty = 9) →
  (final.salty = 17) →
  (initial.salty = final.salty + eaten.salty) →
  (initial.salty = 26) := by
sorry


end NUMINAMATH_CALUDE_paco_initial_salty_cookies_l1085_108504


namespace NUMINAMATH_CALUDE_min_weighings_for_extremes_l1085_108590

/-- Represents a coin with a weight -/
structure Coin where
  weight : ℕ

/-- Represents a weighing operation that compares two coins -/
def weighing (a b : Coin) : Bool :=
  a.weight > b.weight

theorem min_weighings_for_extremes (coins : List Coin) : 
  coins.length = 68 → (∃ n : ℕ, n = 100 ∧ 
    (∀ m : ℕ, m < n → ¬(∃ heaviest lightest : Coin, 
      heaviest ∈ coins ∧ lightest ∈ coins ∧
      (∀ c : Coin, c ∈ coins → c.weight ≤ heaviest.weight) ∧
      (∀ c : Coin, c ∈ coins → c.weight ≥ lightest.weight) ∧
      (heaviest ≠ lightest)))) :=
by
  sorry

end NUMINAMATH_CALUDE_min_weighings_for_extremes_l1085_108590


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1085_108599

/-- Given a triangle with angles in the ratio 3:4:9 and an external angle equal to the smallest 
    internal angle attached at the largest angle, prove that the largest internal angle is 101.25°. -/
theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 →  -- Angles are positive
    b = (4/3) * a ∧ c = 3 * a →  -- Ratio of angles is 3:4:9
    a + b + c = 180 →  -- Sum of internal angles is 180°
    c + a = 12 * a →  -- External angle equals smallest internal angle
    c = 101.25 := by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1085_108599


namespace NUMINAMATH_CALUDE_twin_primes_difference_divisible_by_twelve_l1085_108528

/-- Twin primes are prime numbers that differ by 2 -/
def IsTwinPrime (p q : ℕ) : Prop :=
  Prime p ∧ Prime q ∧ (q = p + 2 ∨ p = q + 2)

/-- The main theorem statement -/
theorem twin_primes_difference_divisible_by_twelve 
  (p q r s : ℕ) 
  (hp : p > 3) 
  (hq : q > 3) 
  (hr : r > 3) 
  (hs : s > 3) 
  (hpq : IsTwinPrime p q) 
  (hrs : IsTwinPrime r s) : 
  12 ∣ (p * r - q * s) := by
  sorry

end NUMINAMATH_CALUDE_twin_primes_difference_divisible_by_twelve_l1085_108528


namespace NUMINAMATH_CALUDE_cookfire_logs_after_three_hours_l1085_108538

/-- Calculates the number of logs left in a cookfire after a given number of hours. -/
def logs_left (initial_logs : ℕ) (burn_rate : ℕ) (add_rate : ℕ) (hours : ℕ) : ℕ :=
  initial_logs + add_rate * hours - burn_rate * hours

/-- Theorem: After 3 hours, a cookfire that starts with 6 logs, burns 3 logs per hour, 
    and receives 2 logs at the end of each hour will have 3 logs left. -/
theorem cookfire_logs_after_three_hours :
  logs_left 6 3 2 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookfire_logs_after_three_hours_l1085_108538


namespace NUMINAMATH_CALUDE_red_ball_probability_l1085_108500

/-- Represents a container with red and green balls -/
structure Container where
  red : Nat
  green : Nat

/-- The probability of selecting a red ball from the given containers -/
def redBallProbability (x y z : Container) : Rat :=
  (1 / 3 : Rat) * (x.red / (x.red + x.green : Rat)) +
  (1 / 3 : Rat) * (y.red / (y.red + y.green : Rat)) +
  (1 / 3 : Rat) * (z.red / (z.red + z.green : Rat))

/-- Theorem stating the probability of selecting a red ball -/
theorem red_ball_probability :
  let x : Container := { red := 3, green := 7 }
  let y : Container := { red := 7, green := 3 }
  let z : Container := { red := 7, green := 3 }
  redBallProbability x y z = 17 / 30 := by
  sorry


end NUMINAMATH_CALUDE_red_ball_probability_l1085_108500


namespace NUMINAMATH_CALUDE_margin_formula_l1085_108562

theorem margin_formula (n : ℝ) (C S M : ℝ) 
  (h1 : n > 0) 
  (h2 : M = (2/n) * C) 
  (h3 : S - M = C) : 
  M = (2/(n+2)) * S := 
by sorry

end NUMINAMATH_CALUDE_margin_formula_l1085_108562


namespace NUMINAMATH_CALUDE_similar_triangles_AB_length_l1085_108521

/-- Two similar triangles with given side lengths and angles -/
structure SimilarTriangles where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  DE : ℝ
  EF : ℝ
  DF : ℝ
  angleBAC : ℝ
  angleEDF : ℝ

/-- Theorem stating that for the given similar triangles, AB = 75/17 -/
theorem similar_triangles_AB_length (t : SimilarTriangles)
  (h1 : t.AB = 5)
  (h2 : t.BC = 17)
  (h3 : t.AC = 12)
  (h4 : t.DE = 9)
  (h5 : t.EF = 15)
  (h6 : t.DF = 12)
  (h7 : t.angleBAC = 120)
  (h8 : t.angleEDF = 120) :
  t.AB = 75 / 17 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_AB_length_l1085_108521


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1085_108577

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 4*x^4 + 4*x^2 - 1 = (x - 1)^3 * (x + 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1085_108577


namespace NUMINAMATH_CALUDE_broken_bulbs_in_foyer_l1085_108580

/-- The number of light bulbs in the kitchen -/
def kitchen_bulbs : ℕ := 35

/-- The fraction of broken light bulbs in the kitchen -/
def kitchen_broken_fraction : ℚ := 3 / 5

/-- The fraction of broken light bulbs in the foyer -/
def foyer_broken_fraction : ℚ := 1 / 3

/-- The number of light bulbs not broken in both the foyer and kitchen -/
def total_not_broken : ℕ := 34

/-- The number of broken light bulbs in the foyer -/
def foyer_broken : ℕ := 10

theorem broken_bulbs_in_foyer :
  foyer_broken = 10 := by sorry

end NUMINAMATH_CALUDE_broken_bulbs_in_foyer_l1085_108580


namespace NUMINAMATH_CALUDE_new_person_weight_l1085_108544

def initial_group_size : ℕ := 4
def average_weight_increase : ℝ := 3
def replaced_person_weight : ℝ := 70

theorem new_person_weight :
  let total_weight_increase : ℝ := initial_group_size * average_weight_increase
  let new_person_weight : ℝ := replaced_person_weight + total_weight_increase
  new_person_weight = 82 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1085_108544


namespace NUMINAMATH_CALUDE_parallel_lines_theorem_l1085_108559

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  (l1.a * l2.b = l1.b * l2.a) ∧ (l1.a ≠ 0 ∨ l1.b ≠ 0) ∧ (l2.a ≠ 0 ∨ l2.b ≠ 0)

/-- The main theorem -/
theorem parallel_lines_theorem (k : ℝ) :
  let l1 : Line := { a := k - 2, b := 4 - k, c := 1 }
  let l2 : Line := { a := 2 * (k - 2), b := -2, c := 3 }
  are_parallel l1 l2 ↔ k = 2 ∨ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_theorem_l1085_108559


namespace NUMINAMATH_CALUDE_machine_work_time_l1085_108575

/-- The number of shirts made today -/
def shirts_today : ℕ := 8

/-- The number of shirts that can be made per minute -/
def shirts_per_minute : ℕ := 2

/-- The number of minutes the machine worked today -/
def minutes_worked : ℚ := shirts_today / shirts_per_minute

theorem machine_work_time : minutes_worked = 4 := by
  sorry

end NUMINAMATH_CALUDE_machine_work_time_l1085_108575


namespace NUMINAMATH_CALUDE_donation_scientific_correct_l1085_108529

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The donation amount in yuan -/
def donation_amount : ℝ := 2175000000

/-- The scientific notation of the donation amount -/
def donation_scientific : ScientificNotation := {
  coefficient := 2.175,
  exponent := 9,
  valid := by sorry
}

/-- Theorem stating that the donation amount is correctly represented in scientific notation -/
theorem donation_scientific_correct : 
  donation_amount = donation_scientific.coefficient * (10 : ℝ) ^ donation_scientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_donation_scientific_correct_l1085_108529


namespace NUMINAMATH_CALUDE_initial_workers_correct_l1085_108535

/-- The number of initial workers required to complete a job -/
def initialWorkers : ℕ := 6

/-- The total amount of work for the job -/
def totalWork : ℕ := initialWorkers * 8

/-- Proves that the initial number of workers is correct given the problem conditions -/
theorem initial_workers_correct :
  totalWork = initialWorkers * 3 + (initialWorkers + 4) * 3 :=
by sorry

end NUMINAMATH_CALUDE_initial_workers_correct_l1085_108535


namespace NUMINAMATH_CALUDE_petya_vasya_meeting_l1085_108560

/-- The number of lampposts along the alley -/
def num_lampposts : ℕ := 100

/-- The lamppost where Petya is observed -/
def petya_observed : ℕ := 22

/-- The lamppost where Vasya is observed -/
def vasya_observed : ℕ := 88

/-- The meeting point of Petya and Vasya -/
def meeting_point : ℕ := 64

theorem petya_vasya_meeting :
  ∀ (petya_speed vasya_speed : ℚ),
    petya_speed > 0 →
    vasya_speed > 0 →
    (petya_observed - 1 : ℚ) / petya_speed = (num_lampposts - vasya_observed : ℚ) / vasya_speed →
    (meeting_point - 1 : ℚ) / petya_speed = (num_lampposts - meeting_point : ℚ) / vasya_speed :=
by sorry

end NUMINAMATH_CALUDE_petya_vasya_meeting_l1085_108560


namespace NUMINAMATH_CALUDE_jo_kate_difference_l1085_108565

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_ten (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

def kate_sum (n : ℕ) : ℕ :=
  (List.range n).map round_to_nearest_ten |>.sum

theorem jo_kate_difference :
  kate_sum 100 - sum_of_first_n 100 = 500 := by
  sorry

end NUMINAMATH_CALUDE_jo_kate_difference_l1085_108565


namespace NUMINAMATH_CALUDE_farm_tree_count_l1085_108592

/-- Represents the number of trees of each type that fell during the typhoon -/
structure FallenTrees where
  narra : ℕ
  mahogany : ℕ
  total : ℕ
  one_more_mahogany : mahogany = narra + 1
  sum_equals_total : narra + mahogany = total

/-- Calculates the final number of trees on the farm -/
def final_tree_count (initial_mahogany initial_narra total_fallen : ℕ) (fallen : FallenTrees) : ℕ :=
  let remaining := initial_mahogany + initial_narra - total_fallen
  let new_narra := 2 * fallen.narra
  let new_mahogany := 3 * fallen.mahogany
  remaining + new_narra + new_mahogany

/-- The theorem to be proved -/
theorem farm_tree_count :
  ∃ (fallen : FallenTrees),
    fallen.total = 5 ∧
    final_tree_count 50 30 5 fallen = 88 := by
  sorry

end NUMINAMATH_CALUDE_farm_tree_count_l1085_108592


namespace NUMINAMATH_CALUDE_modulus_of_z_l1085_108534

theorem modulus_of_z (z : ℂ) (h : z^2 = 16 - 30*I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1085_108534


namespace NUMINAMATH_CALUDE_second_number_possibilities_l1085_108594

def is_valid_pair (x y : ℤ) : Prop :=
  (x = 14 ∨ y = 14) ∧ 2*x + 3*y = 94

theorem second_number_possibilities :
  ∃ (a b : ℤ), a ≠ b ∧ 
  (∀ x y, is_valid_pair x y → (x = 14 ∧ y = a) ∨ (y = 14 ∧ x = b)) :=
sorry

end NUMINAMATH_CALUDE_second_number_possibilities_l1085_108594


namespace NUMINAMATH_CALUDE_probability_of_three_in_18_23_l1085_108595

/-- The decimal representation of a rational number -/
def decimalRepresentation (n d : ℕ) : List ℕ :=
  sorry

/-- Count the occurrences of a digit in a list of digits -/
def countOccurrences (digit : ℕ) (digits : List ℕ) : ℕ :=
  sorry

/-- The probability of selecting a specific digit from a decimal representation -/
def probabilityOfDigit (n d digit : ℕ) : ℚ :=
  let digits := decimalRepresentation n d
  (countOccurrences digit digits : ℚ) / (digits.length : ℚ)

theorem probability_of_three_in_18_23 :
  probabilityOfDigit 18 23 3 = 3 / 26 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_three_in_18_23_l1085_108595


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1085_108550

theorem vector_sum_magnitude (x : ℝ) : 
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (x + 2, -2)
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1085_108550


namespace NUMINAMATH_CALUDE_lawn_width_proof_l1085_108553

/-- Proves that the width of a rectangular lawn is 60 meters given specific conditions --/
theorem lawn_width_proof (W : ℝ) : 
  W > 0 →  -- Width is positive
  (10 * W + 10 * 70 - 10 * 10) * 3 = 3600 →  -- Cost equation
  W = 60 := by
  sorry

end NUMINAMATH_CALUDE_lawn_width_proof_l1085_108553


namespace NUMINAMATH_CALUDE_ellie_bike_oil_needed_l1085_108584

/-- The amount of oil needed to fix a bicycle --/
def oil_needed (oil_per_wheel : ℕ) (oil_for_rest : ℕ) (num_wheels : ℕ) : ℕ :=
  oil_per_wheel * num_wheels + oil_for_rest

/-- Theorem: The total amount of oil needed to fix Ellie's bike is 25ml --/
theorem ellie_bike_oil_needed :
  oil_needed 10 5 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ellie_bike_oil_needed_l1085_108584


namespace NUMINAMATH_CALUDE_octahedron_non_blue_probability_l1085_108579

structure Octahedron :=
  (total_faces : ℕ)
  (blue_faces : ℕ)
  (red_faces : ℕ)
  (green_faces : ℕ)

def non_blue_probability (o : Octahedron) : ℚ :=
  (o.red_faces + o.green_faces : ℚ) / o.total_faces

theorem octahedron_non_blue_probability :
  ∀ o : Octahedron,
  o.total_faces = 8 →
  o.blue_faces = 3 →
  o.red_faces = 3 →
  o.green_faces = 2 →
  non_blue_probability o = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_non_blue_probability_l1085_108579


namespace NUMINAMATH_CALUDE_average_of_numbers_l1085_108566

def numbers : List ℕ := [12, 13, 14, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers : (numbers.sum : ℚ) / numbers.length = 125781 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1085_108566


namespace NUMINAMATH_CALUDE_ab_neq_zero_sufficient_not_necessary_for_a_neq_zero_l1085_108543

theorem ab_neq_zero_sufficient_not_necessary_for_a_neq_zero :
  (∀ a b : ℝ, ab ≠ 0 → a ≠ 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ ab = 0) :=
by sorry

end NUMINAMATH_CALUDE_ab_neq_zero_sufficient_not_necessary_for_a_neq_zero_l1085_108543


namespace NUMINAMATH_CALUDE_no_integer_solution_for_1980_l1085_108569

theorem no_integer_solution_for_1980 : ∀ m n : ℤ, m^2 + n^2 ≠ 1980 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_1980_l1085_108569


namespace NUMINAMATH_CALUDE_expected_value_is_negative_one_fifth_l1085_108508

/-- A die with two faces: Star and Moon -/
inductive DieFace
| star
| moon

/-- The probability of getting a Star face -/
def probStar : ℚ := 2/5

/-- The probability of getting a Moon face -/
def probMoon : ℚ := 3/5

/-- The winnings for Star face -/
def winStar : ℚ := 4

/-- The losses for Moon face -/
def lossMoon : ℚ := -3

/-- The expected value of one roll of the die -/
def expectedValue : ℚ := probStar * winStar + probMoon * lossMoon

theorem expected_value_is_negative_one_fifth :
  expectedValue = -1/5 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_negative_one_fifth_l1085_108508


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1085_108549

theorem quadratic_inequality_solution (a b c : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → 0 ≤ a * x^2 + b * x + c ∧ a * x^2 + b * x + c ≤ 1) →
  a > 0 →
  b = -a →
  c = -2 * a + 1 →
  0 < a ∧ a ≤ 4/9 →
  3 * a + 2 * b + c ≠ 1/3 ∧ 3 * a + 2 * b + c ≠ 5/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1085_108549


namespace NUMINAMATH_CALUDE_percentage_of_women_parents_l1085_108540

theorem percentage_of_women_parents (P : ℝ) (M : ℝ) (F : ℝ) : 
  P > 0 →
  M + F = P →
  (1 / 8) * M + (1 / 4) * F = (17.5 / 100) * P →
  M / P = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_women_parents_l1085_108540


namespace NUMINAMATH_CALUDE_base9_perfect_square_last_digit_l1085_108551

/-- Represents a number in base 9 of the form ab4d -/
structure Base9Number where
  a : Nat
  b : Nat
  d : Nat
  a_nonzero : a ≠ 0

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : Nat :=
  729 * n.a + 81 * n.b + 36 + n.d

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, n = m * m

theorem base9_perfect_square_last_digit 
  (n : Base9Number) 
  (h : isPerfectSquare (toDecimal n)) : 
  n.d = 0 ∨ n.d = 1 ∨ n.d = 4 ∨ n.d = 7 := by
  sorry

end NUMINAMATH_CALUDE_base9_perfect_square_last_digit_l1085_108551


namespace NUMINAMATH_CALUDE_quadratic_roots_identity_l1085_108546

theorem quadratic_roots_identity (p q r s k : ℝ) (α β : ℝ) : 
  (α^2 + p*α + q = 0) →
  (β^2 + r*β + s = 0) →
  (α / β = k) →
  (q - k^2 * s)^2 + k * (p - k * r) * (k * p * s - q * r) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_identity_l1085_108546


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_7_neg7_4_l1085_108516

/-- Converts rectangular coordinates to cylindrical coordinates -/
def rect_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ := sorry

theorem rect_to_cylindrical_7_neg7_4 :
  let (r, θ, z) := rect_to_cylindrical 7 (-7) 4
  r = 7 * Real.sqrt 2 ∧
  θ = 7 * Real.pi / 4 ∧
  z = 4 ∧
  r > 0 ∧
  0 ≤ θ ∧ θ < 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_7_neg7_4_l1085_108516


namespace NUMINAMATH_CALUDE_original_proposition_is_true_negation_is_false_l1085_108525

theorem original_proposition_is_true : ∀ (a b : ℝ), a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1) := by
  sorry

theorem negation_is_false : ¬(∀ (a b : ℝ), a + b ≥ 2 → (a < 1 ∧ b < 1)) := by
  sorry

end NUMINAMATH_CALUDE_original_proposition_is_true_negation_is_false_l1085_108525


namespace NUMINAMATH_CALUDE_negation_square_nonnegative_l1085_108585

theorem negation_square_nonnegative (x : ℝ) : 
  ¬(x ≥ 0 → x^2 > 0) ↔ (x < 0 → x^2 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_square_nonnegative_l1085_108585


namespace NUMINAMATH_CALUDE_volume_of_circumscribed_polyhedron_l1085_108512

-- Define a polyhedron circumscribed around a sphere
structure CircumscribedPolyhedron where
  -- R is the radius of the inscribed sphere
  R : ℝ
  -- S is the surface area of the polyhedron
  S : ℝ
  -- V is the volume of the polyhedron
  V : ℝ
  -- Ensure R and S are positive
  R_pos : 0 < R
  S_pos : 0 < S

-- Theorem statement
theorem volume_of_circumscribed_polyhedron (p : CircumscribedPolyhedron) :
  p.V = (p.S * p.R) / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_circumscribed_polyhedron_l1085_108512


namespace NUMINAMATH_CALUDE_catherine_pencil_distribution_l1085_108524

theorem catherine_pencil_distribution (initial_pens : ℕ) (initial_pencils : ℕ) 
  (friends : ℕ) (pens_per_friend : ℕ) (total_left : ℕ) :
  initial_pens = 60 →
  initial_pencils = initial_pens →
  friends = 7 →
  pens_per_friend = 8 →
  total_left = 22 →
  ∃ (pencils_per_friend : ℕ),
    pencils_per_friend * friends = initial_pencils - (total_left - (initial_pens - pens_per_friend * friends)) ∧
    pencils_per_friend = 6 :=
by sorry

end NUMINAMATH_CALUDE_catherine_pencil_distribution_l1085_108524


namespace NUMINAMATH_CALUDE_work_completion_time_l1085_108533

theorem work_completion_time (A : ℝ) (h1 : A > 0) : 
  (∃ B : ℝ, B = A / 2 ∧ 1 / A + 1 / B = 1 / 6) → A = 18 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1085_108533


namespace NUMINAMATH_CALUDE_blue_marble_percent_is_35_l1085_108573

/-- Represents the composition of items in an urn -/
structure UrnComposition where
  button_percent : ℝ
  red_marble_percent : ℝ
  blue_marble_percent : ℝ

/-- The percentage of blue marbles in the urn -/
def blue_marble_percentage (urn : UrnComposition) : ℝ :=
  urn.blue_marble_percent

/-- Theorem stating the percentage of blue marbles in the urn -/
theorem blue_marble_percent_is_35 (urn : UrnComposition) 
  (h1 : urn.button_percent = 0.3)
  (h2 : urn.red_marble_percent = 0.5 * (1 - urn.button_percent)) :
  blue_marble_percentage urn = 0.35 := by
  sorry

#check blue_marble_percent_is_35

end NUMINAMATH_CALUDE_blue_marble_percent_is_35_l1085_108573


namespace NUMINAMATH_CALUDE_distance_between_points_l1085_108558

-- Define the complex numbers
def z_J : ℂ := 3 + 4 * Complex.I
def z_G : ℂ := 2 - 3 * Complex.I

-- Define the scaled version of Gracie's point
def scaled_z_G : ℂ := 2 * z_G

-- Theorem statement
theorem distance_between_points : Complex.abs (z_J - scaled_z_G) = Real.sqrt 101 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1085_108558


namespace NUMINAMATH_CALUDE_regular_hexagon_interior_angle_l1085_108532

/-- The measure of an interior angle of a regular hexagon -/
def interior_angle_regular_hexagon : ℝ := 120

/-- Theorem: The measure of each interior angle of a regular hexagon is 120 degrees -/
theorem regular_hexagon_interior_angle :
  interior_angle_regular_hexagon = 120 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_interior_angle_l1085_108532


namespace NUMINAMATH_CALUDE_lesser_fraction_l1085_108552

theorem lesser_fraction (x y : ℚ) (sum_eq : x + y = 17/24) (prod_eq : x * y = 1/8) :
  min x y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_lesser_fraction_l1085_108552


namespace NUMINAMATH_CALUDE_solve_equation_l1085_108509

theorem solve_equation : ∃ x : ℚ, 5 * (x - 9) = 6 * (3 - 3 * x) + 6 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1085_108509


namespace NUMINAMATH_CALUDE_norris_october_savings_l1085_108571

/-- Proves that Norris saved $25 in October given the savings and spending information. -/
theorem norris_october_savings :
  ∀ (september_savings october_savings november_savings spent remaining : ℕ),
  september_savings = 29 →
  november_savings = 31 →
  spent = 75 →
  remaining = 10 →
  september_savings + october_savings + november_savings = spent + remaining →
  october_savings = 25 := by
sorry

end NUMINAMATH_CALUDE_norris_october_savings_l1085_108571


namespace NUMINAMATH_CALUDE_task_probability_l1085_108531

theorem task_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 5/8) 
  (h2 : p2 = 3/5) 
  (h3 : p3 = 7/10) 
  (h4 : p4 = 9/12) : 
  p1 * (1 - p2) * (1 - p3) * p4 = 9/160 := by
  sorry

end NUMINAMATH_CALUDE_task_probability_l1085_108531


namespace NUMINAMATH_CALUDE_sequence_formula_l1085_108514

theorem sequence_formula (a : ℕ → ℚ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → 3 * a (n + 1) + 2 * a (n + 1) * a n - a n = 0) →
  ∀ n : ℕ, n ≥ 1 → a n = 1 / (2 * 3^(n - 1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formula_l1085_108514


namespace NUMINAMATH_CALUDE_max_value_expression_l1085_108522

theorem max_value_expression (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0) :
  (k * x + y)^2 / (x^2 + y^2) ≤ k^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l1085_108522


namespace NUMINAMATH_CALUDE_total_trees_planted_l1085_108572

theorem total_trees_planted (total_gardeners : ℕ) (street_a_gardeners : ℕ) (street_b_gardeners : ℕ) 
  (h1 : total_gardeners = street_a_gardeners + street_b_gardeners)
  (h2 : total_gardeners = 19)
  (h3 : street_a_gardeners = 4)
  (h4 : street_b_gardeners = 15)
  (h5 : ∃ x : ℕ, street_b_gardeners * x - 1 = 4 * (street_a_gardeners * x - 1)) :
  ∃ trees_per_gardener : ℕ, total_gardeners * trees_per_gardener = 57 :=
by sorry

end NUMINAMATH_CALUDE_total_trees_planted_l1085_108572


namespace NUMINAMATH_CALUDE_min_hours_to_drive_l1085_108556

/-- The legal blood alcohol content (BAC) limit for driving -/
def legal_bac_limit : ℝ := 0.2

/-- The initial BAC after drinking -/
def initial_bac : ℝ := 0.8

/-- The rate at which BAC decreases per hour -/
def bac_decrease_rate : ℝ := 0.5

/-- The minimum number of hours to wait before driving -/
def min_wait_hours : ℕ := 2

/-- Theorem stating the minimum number of hours to wait before driving -/
theorem min_hours_to_drive :
  (initial_bac * (1 - bac_decrease_rate) ^ min_wait_hours ≤ legal_bac_limit) ∧
  (∀ h : ℕ, h < min_wait_hours → initial_bac * (1 - bac_decrease_rate) ^ h > legal_bac_limit) :=
sorry

end NUMINAMATH_CALUDE_min_hours_to_drive_l1085_108556


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1085_108537

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio : 
  let side_length : ℝ := 6
  let area : ℝ := side_length^2 * Real.sqrt 3 / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1085_108537


namespace NUMINAMATH_CALUDE_derivative_of_f_l1085_108574

/-- The function f(x) = 3x^2 -/
def f (x : ℝ) : ℝ := 3 * x^2

/-- The derivative of f(x) = 3x^2 is 6x -/
theorem derivative_of_f :
  deriv f = fun x ↦ 6 * x := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l1085_108574


namespace NUMINAMATH_CALUDE_nested_radical_value_l1085_108536

/-- Given a continuous nested radical X = √(x√(y√(z√(x√(y√(z...)))))), 
    prove that X = ∛(x^4 * y^2 * z) -/
theorem nested_radical_value (x y z : ℝ) (X : ℝ) 
  (h : X = Real.sqrt (x * Real.sqrt (y * Real.sqrt (z * X)))) :
  X = (x^4 * y^2 * z)^(1/7) := by
sorry

end NUMINAMATH_CALUDE_nested_radical_value_l1085_108536


namespace NUMINAMATH_CALUDE_min_value_theorem_l1085_108545

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 12) :
  2 * x + 3 * y + 6 * z ≥ 18 * Real.rpow 2 (1/3) ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 12 ∧
  2 * x₀ + 3 * y₀ + 6 * z₀ = 18 * Real.rpow 2 (1/3) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1085_108545


namespace NUMINAMATH_CALUDE_P_bounds_l1085_108598

/-- Represents the minimum number of transformations needed to convert
    any triangulation of a convex n-gon to any other triangulation. -/
def P (n : ℕ) : ℕ := sorry

/-- The main theorem about the bounds of P(n) -/
theorem P_bounds (n : ℕ) : 
  (n ≥ 3 → P n ≥ n - 3) ∧ 
  (n ≥ 3 → P n ≤ 2*n - 7) ∧ 
  (n ≥ 13 → P n ≤ 2*n - 10) := by
  sorry

end NUMINAMATH_CALUDE_P_bounds_l1085_108598


namespace NUMINAMATH_CALUDE_simplify_expression_l1085_108539

theorem simplify_expression (y : ℝ) : (3*y)^3 - (2*y)*(y^2) + y^4 = 25*y^3 + y^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1085_108539


namespace NUMINAMATH_CALUDE_principal_calculation_l1085_108593

/-- Proves that given the specified conditions, the principal is 6200 --/
theorem principal_calculation (rate : ℚ) (time : ℕ) (interest_difference : ℚ) :
  rate = 5 / 100 →
  time = 10 →
  interest_difference = 3100 →
  ∃ (principal : ℚ), principal * rate * time = principal - interest_difference ∧ principal = 6200 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l1085_108593


namespace NUMINAMATH_CALUDE_total_sodas_sold_l1085_108506

theorem total_sodas_sold (morning_sales afternoon_sales : ℕ) 
  (h1 : morning_sales = 77)
  (h2 : afternoon_sales = 19) :
  morning_sales + afternoon_sales = 96 := by
sorry

end NUMINAMATH_CALUDE_total_sodas_sold_l1085_108506


namespace NUMINAMATH_CALUDE_mortgage_duration_l1085_108513

theorem mortgage_duration (house_price deposit monthly_payment : ℕ) :
  house_price = 280000 →
  deposit = 40000 →
  monthly_payment = 2000 →
  (house_price - deposit) / monthly_payment / 12 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mortgage_duration_l1085_108513


namespace NUMINAMATH_CALUDE_omega_sum_l1085_108568

theorem omega_sum (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 + ω^32 + ω^34 + ω^36 + ω^38 + ω^40 + ω^42 + 
  ω^44 + ω^46 + ω^48 + ω^50 + ω^52 + ω^54 + ω^56 + ω^58 + ω^60 + ω^62 + ω^64 + ω^66 + ω^68 + ω^70 + ω^72 = -ω^7 :=
by sorry

end NUMINAMATH_CALUDE_omega_sum_l1085_108568


namespace NUMINAMATH_CALUDE_show_length_ratio_l1085_108527

theorem show_length_ratio (first_show_length second_show_length total_time : ℕ) 
  (h1 : first_show_length = 30)
  (h2 : total_time = 150)
  (h3 : second_show_length = total_time - first_show_length) :
  second_show_length / first_show_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_show_length_ratio_l1085_108527


namespace NUMINAMATH_CALUDE_congruence_solution_l1085_108596

theorem congruence_solution (n : ℤ) : 
  -20 ≤ n ∧ n ≤ 20 ∧ n ≡ -127 [ZMOD 7] → n = -13 ∨ n = 1 ∨ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1085_108596


namespace NUMINAMATH_CALUDE_custom_op_example_l1085_108576

/-- Custom binary operation @ defined as a @ b = 5a - 2b -/
def custom_op (a b : ℝ) : ℝ := 5 * a - 2 * b

/-- Theorem stating that 4 @ 7 = 6 under the custom operation -/
theorem custom_op_example : custom_op 4 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l1085_108576


namespace NUMINAMATH_CALUDE_symmetry_composition_l1085_108511

/-- Given three lines l₁, l₂, and l₃ in a plane, where l₃ is the reflection of l₂ across l₁,
    this theorem states that the symmetry transformation with respect to l₃
    is equivalent to the composition of symmetry transformations:
    first with respect to l₁, then l₂, and finally l₁ again. -/
theorem symmetry_composition (l₁ l₂ l₃ : Line) (h : l₃ = S_l₁ l₂) :
  S_l₃ = S_l₁ ∘ S_l₂ ∘ S_l₁ :=
sorry

/-- Symmetry transformation with respect to a line -/
def S_l (l : Line) : Point → Point :=
sorry

/-- Reflection of a line across another line -/
def S_l₁ (l : Line) : Line :=
sorry

end NUMINAMATH_CALUDE_symmetry_composition_l1085_108511


namespace NUMINAMATH_CALUDE_chord_length_l1085_108523

theorem chord_length (R : ℝ) (AB AC : ℝ) (h1 : R = 8) (h2 : AB = 10) 
  (h3 : AC = (2 * Real.pi * R) / 3) : 
  (AC : ℝ) = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l1085_108523


namespace NUMINAMATH_CALUDE_other_number_l1085_108582

theorem other_number (x : ℝ) : 
  0.5 > x ∧ 0.5 - x = 0.16666666666666669 → x = 0.3333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_other_number_l1085_108582


namespace NUMINAMATH_CALUDE_garden_fence_posts_l1085_108547

/-- Calculates the number of fence posts required for a rectangular garden -/
def fencePostsRequired (length width postSpacing : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let postsOnPerimeter := perimeter / postSpacing
  postsOnPerimeter + 1

/-- Theorem stating the number of fence posts required for the specific garden -/
theorem garden_fence_posts :
  fencePostsRequired 72 32 8 = 26 := by
  sorry

end NUMINAMATH_CALUDE_garden_fence_posts_l1085_108547


namespace NUMINAMATH_CALUDE_bee_travel_distance_l1085_108554

theorem bee_travel_distance (initial_distance : ℝ) (speed_A : ℝ) (speed_B : ℝ) (speed_bee : ℝ)
  (h1 : initial_distance = 120)
  (h2 : speed_A = 30)
  (h3 : speed_B = 10)
  (h4 : speed_bee = 60) :
  let relative_speed := speed_A + speed_B
  let meeting_time := initial_distance / relative_speed
  speed_bee * meeting_time = 180 := by
sorry

end NUMINAMATH_CALUDE_bee_travel_distance_l1085_108554


namespace NUMINAMATH_CALUDE_workbook_arrangement_count_l1085_108567

/-- The number of ways to arrange 2 Korean and 2 English workbooks in a row with English workbooks side by side -/
def arrange_workbooks : ℕ :=
  let korean_books := 2
  let english_books := 2
  let total_units := korean_books + 1  -- English books count as one unit
  let unit_arrangements := Nat.factorial total_units
  let english_arrangements := Nat.factorial english_books
  unit_arrangements * english_arrangements

/-- Theorem stating that the number of arrangements is 12 -/
theorem workbook_arrangement_count : arrange_workbooks = 12 := by
  sorry

end NUMINAMATH_CALUDE_workbook_arrangement_count_l1085_108567


namespace NUMINAMATH_CALUDE_geometric_sequence_b6_l1085_108561

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

/-- The theorem statement -/
theorem geometric_sequence_b6 (b : ℕ → ℝ) :
  geometric_sequence b → b 3 * b 9 = 9 → b 6 = 3 ∨ b 6 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_b6_l1085_108561
