import Mathlib

namespace NUMINAMATH_CALUDE_product_of_solutions_abs_y_eq_3_abs_y_minus_2_l3962_396299

theorem product_of_solutions_abs_y_eq_3_abs_y_minus_2 :
  ∃ (y₁ y₂ : ℝ), (|y₁| = 3 * (|y₁| - 2)) ∧ (|y₂| = 3 * (|y₂| - 2)) ∧ y₁ ≠ y₂ ∧ y₁ * y₂ = -9 :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_abs_y_eq_3_abs_y_minus_2_l3962_396299


namespace NUMINAMATH_CALUDE_f_composition_fixed_points_l3962_396248

def f (x : ℝ) := x^3 - 3*x^2

theorem f_composition_fixed_points :
  ∃ (x : ℝ), f (f x) = f x ∧ (x = 0 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_f_composition_fixed_points_l3962_396248


namespace NUMINAMATH_CALUDE_not_in_range_iff_a_in_interval_l3962_396220

/-- The function g(x) = x^2 + ax + 3 -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

/-- Theorem: -3 is not in the range of g(x) if and only if a is in the open interval (-√24, √24) -/
theorem not_in_range_iff_a_in_interval (a : ℝ) :
  (∀ x : ℝ, g a x ≠ -3) ↔ a > -Real.sqrt 24 ∧ a < Real.sqrt 24 := by
  sorry

end NUMINAMATH_CALUDE_not_in_range_iff_a_in_interval_l3962_396220


namespace NUMINAMATH_CALUDE_xy_range_l3962_396251

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 3*y + 2/x + 4/y = 10) : 
  1 ≤ x*y ∧ x*y ≤ 8/3 := by
sorry

end NUMINAMATH_CALUDE_xy_range_l3962_396251


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_is_eight_l3962_396231

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (12 - x^4) + x^2) / x^3 + 4

def X : Set ℝ := {x | x ∈ Set.Icc (-1) 0 ∪ Set.Ioc 0 1}

theorem sum_of_max_and_min_is_eight :
  ∃ (A B : ℝ), (∀ x ∈ X, f x ≤ A) ∧ 
               (∃ x ∈ X, f x = A) ∧ 
               (∀ x ∈ X, B ≤ f x) ∧ 
               (∃ x ∈ X, f x = B) ∧ 
               A + B = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_is_eight_l3962_396231


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l3962_396261

theorem quadratic_two_roots (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x - b^2 / (4 * a)
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l3962_396261


namespace NUMINAMATH_CALUDE_max_tickets_for_hockey_l3962_396283

def max_tickets (ticket_price : ℕ) (budget : ℕ) : ℕ :=
  (budget / ticket_price : ℕ)

theorem max_tickets_for_hockey (ticket_price : ℕ) (budget : ℕ) 
  (h1 : ticket_price = 20) (h2 : budget = 150) : 
  max_tickets ticket_price budget = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_tickets_for_hockey_l3962_396283


namespace NUMINAMATH_CALUDE_find_N_l3962_396264

theorem find_N : ∃ N : ℝ, (0.2 * N = 0.3 * 2500) ∧ (N = 3750) := by
  sorry

end NUMINAMATH_CALUDE_find_N_l3962_396264


namespace NUMINAMATH_CALUDE_xy_equation_implications_l3962_396221

theorem xy_equation_implications (x y : ℝ) (h : x^2 + y^2 - x*y = 1) :
  (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_xy_equation_implications_l3962_396221


namespace NUMINAMATH_CALUDE_player2_is_best_l3962_396246

structure Player where
  id : Nat
  average_time : ℝ
  variance : ℝ

def players : List Player := [
  { id := 1, average_time := 51, variance := 3.5 },
  { id := 2, average_time := 50, variance := 3.5 },
  { id := 3, average_time := 51, variance := 14.5 },
  { id := 4, average_time := 50, variance := 14.4 }
]

def is_better_performer (p1 p2 : Player) : Prop :=
  p1.average_time < p2.average_time ∨ 
  (p1.average_time = p2.average_time ∧ p1.variance < p2.variance)

theorem player2_is_best : 
  ∀ p ∈ players, p.id ≠ 2 → is_better_performer (players[1]) p :=
sorry

end NUMINAMATH_CALUDE_player2_is_best_l3962_396246


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l3962_396269

theorem lcm_gcf_problem (n m : ℕ+) : 
  Nat.lcm n m = 54 → Nat.gcd n m = 8 → n = 36 → m = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l3962_396269


namespace NUMINAMATH_CALUDE_max_ratio_in_triangle_l3962_396244

/-- Given a triangle OAB where O is the origin, A is the point (4,3), and B is the point (x,0) with x > 0,
    this theorem states that the maximum value of the ratio x/l(x) is 5/3,
    where l(x) is the length of line segment AB. -/
theorem max_ratio_in_triangle (x : ℝ) (hx : x > 0) : 
  let A : ℝ × ℝ := (4, 3)
  let B : ℝ × ℝ := (x, 0)
  let l : ℝ → ℝ := fun x => Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  (∀ y > 0, y / l y ≤ x / l x) → x / l x = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_max_ratio_in_triangle_l3962_396244


namespace NUMINAMATH_CALUDE_discount_store_purchase_l3962_396272

/-- Represents the number of items of each type bought by one person -/
structure ItemCounts where
  typeA : ℕ
  typeB : ℕ

/-- Represents the prices of items -/
structure Prices where
  typeA : ℕ
  typeB : ℕ

def total_spent (counts : ItemCounts) (prices : Prices) : ℕ :=
  counts.typeA * prices.typeA + counts.typeB * prices.typeB

theorem discount_store_purchase : ∃ (counts : ItemCounts),
  let prices : Prices := ⟨8, 9⟩
  total_spent counts prices = 172 ∧
  counts.typeA + counts.typeB = counts.typeA + counts.typeB ∧
  counts.typeA = 4 ∧
  counts.typeB = 6 := by
  sorry

end NUMINAMATH_CALUDE_discount_store_purchase_l3962_396272


namespace NUMINAMATH_CALUDE_range_of_m_l3962_396232

theorem range_of_m (m : ℝ) : 
  ¬((m + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m*x + 1 > 0)) → 
  m ≤ -2 ∨ m > -1 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3962_396232


namespace NUMINAMATH_CALUDE_interest_rate_difference_l3962_396297

/-- Given a principal amount, time period, and difference in interest earned,
    calculate the difference between two interest rates. -/
theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (interest_diff : ℝ)
  (h1 : principal = 200)
  (h2 : time = 10)
  (h3 : interest_diff = 100) :
  (interest_diff / (principal * time)) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l3962_396297


namespace NUMINAMATH_CALUDE_fifth_term_value_l3962_396260

/-- A geometric sequence with the given property -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

/-- The property of the sequence given in the problem -/
def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n, a n + a (n + 1) = 3 * (1/2)^n

theorem fifth_term_value (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : sequence_property a) : 
  a 5 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l3962_396260


namespace NUMINAMATH_CALUDE_square_ends_with_self_l3962_396265

theorem square_ends_with_self (A : ℕ) : 
  (100 ≤ A ∧ A ≤ 999) → (A^2 ≡ A [ZMOD 1000]) ↔ (A = 376 ∨ A = 625) := by
  sorry

end NUMINAMATH_CALUDE_square_ends_with_self_l3962_396265


namespace NUMINAMATH_CALUDE_three_cubic_yards_to_cubic_feet_l3962_396271

-- Define the conversion factor
def yard_to_foot : ℝ := 3

-- Define the volume in cubic yards
def cubic_yards : ℝ := 3

-- Theorem to prove
theorem three_cubic_yards_to_cubic_feet :
  cubic_yards * (yard_to_foot ^ 3) = 81 := by
  sorry

end NUMINAMATH_CALUDE_three_cubic_yards_to_cubic_feet_l3962_396271


namespace NUMINAMATH_CALUDE_estimated_percentage_is_5_7_l3962_396240

/-- Represents the data from the household survey -/
structure SurveyData where
  total_households : ℕ
  ordinary_families : ℕ
  high_income_families : ℕ
  ordinary_sample_size : ℕ
  high_income_sample_size : ℕ
  ordinary_with_3plus_houses : ℕ
  high_income_with_3plus_houses : ℕ

/-- Calculates the estimated percentage of families with 3 or more houses -/
def estimatePercentage (data : SurveyData) : ℚ :=
  let ordinary_estimate := (data.ordinary_families : ℚ) * (data.ordinary_with_3plus_houses : ℚ) / (data.ordinary_sample_size : ℚ)
  let high_income_estimate := (data.high_income_families : ℚ) * (data.high_income_with_3plus_houses : ℚ) / (data.high_income_sample_size : ℚ)
  let total_estimate := ordinary_estimate + high_income_estimate
  (total_estimate / (data.total_households : ℚ)) * 100

/-- The survey data for the household study -/
def surveyData : SurveyData := {
  total_households := 100000,
  ordinary_families := 99000,
  high_income_families := 1000,
  ordinary_sample_size := 990,
  high_income_sample_size := 100,
  ordinary_with_3plus_houses := 50,
  high_income_with_3plus_houses := 70
}

/-- Theorem stating that the estimated percentage of families with 3 or more houses is 5.7% -/
theorem estimated_percentage_is_5_7 :
  estimatePercentage surveyData = 57/10 := by
  sorry


end NUMINAMATH_CALUDE_estimated_percentage_is_5_7_l3962_396240


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3962_396215

theorem arithmetic_mean_of_fractions : 
  (5 : ℚ) / 6 = ((9 : ℚ) / 12 + (11 : ℚ) / 12) / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3962_396215


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l3962_396277

theorem cube_root_unity_sum (ω : ℂ) : 
  ω^3 = 1 → ω ≠ 1 → (2 - ω + ω^2)^4 + (2 + ω - ω^2)^4 = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l3962_396277


namespace NUMINAMATH_CALUDE_base_9_to_base_10_conversion_l3962_396236

-- Define the base-9 number
def base_9_number : ℕ := 5126

-- Define the conversion function from base 9 to base 10
def base_9_to_base_10 (n : ℕ) : ℕ :=
  (n % 10) +
  ((n / 10) % 10) * 9 +
  ((n / 100) % 10) * 9^2 +
  ((n / 1000) % 10) * 9^3

-- Theorem statement
theorem base_9_to_base_10_conversion :
  base_9_to_base_10 base_9_number = 3750 := by
  sorry

end NUMINAMATH_CALUDE_base_9_to_base_10_conversion_l3962_396236


namespace NUMINAMATH_CALUDE_flagpole_break_height_l3962_396287

/-- Proves that a 6-meter flagpole breaking and touching the ground 2 meters away
    from its base breaks at a height of 3 meters. -/
theorem flagpole_break_height :
  ∀ (h x : ℝ),
  h = 6 →                            -- Total height of flagpole
  x > 0 →                            -- Breaking point is above ground
  x < h →                            -- Breaking point is below the top
  x^2 + 2^2 = (h - x)^2 →            -- Pythagorean theorem
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_flagpole_break_height_l3962_396287


namespace NUMINAMATH_CALUDE_distance_on_line_l3962_396273

/-- The distance between two points (p, q) and (r, s) on the line y = 2x + 3, where s = 2r + 6 -/
theorem distance_on_line (p r : ℝ) : 
  let q := 2 * p + 3
  let s := 2 * r + 6
  Real.sqrt ((r - p)^2 + (s - q)^2) = Real.sqrt (5 * (r - p)^2 + 12 * (r - p) + 9) := by
  sorry

end NUMINAMATH_CALUDE_distance_on_line_l3962_396273


namespace NUMINAMATH_CALUDE_evelyn_lost_bottle_caps_l3962_396288

/-- The number of bottle caps Evelyn lost -/
def bottle_caps_lost (initial : ℝ) (final : ℝ) : ℝ :=
  initial - final

/-- Proof that Evelyn lost 18.0 bottle caps -/
theorem evelyn_lost_bottle_caps :
  bottle_caps_lost 63.0 45 = 18.0 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_lost_bottle_caps_l3962_396288


namespace NUMINAMATH_CALUDE_q_polynomial_l3962_396263

theorem q_polynomial (x : ℝ) (q : ℝ → ℝ) 
  (h : ∀ x, q x + (2*x^6 + 4*x^4 - 5*x^3 + 2*x) = (3*x^4 + x^3 - 11*x^2 + 6*x + 3)) :
  q x = -2*x^6 - x^4 + 6*x^3 - 11*x^2 + 4*x + 3 := by
sorry

end NUMINAMATH_CALUDE_q_polynomial_l3962_396263


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3962_396282

theorem complex_equation_solution (z : ℂ) (h : z * (1 + Complex.I) = 2 - 4 * Complex.I) :
  z = -1 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3962_396282


namespace NUMINAMATH_CALUDE_water_displaced_squared_5ft_l3962_396242

/-- The volume of water displaced by a fully submerged cube -/
def water_displaced (cube_side : ℝ) : ℝ := cube_side ^ 3

/-- The square of the volume of water displaced by a fully submerged cube -/
def water_displaced_squared (cube_side : ℝ) : ℝ := (water_displaced cube_side) ^ 2

/-- Theorem: The square of the volume of water displaced by a fully submerged cube
    with side length 5 feet is equal to 15625 (cubic feet)^2 -/
theorem water_displaced_squared_5ft :
  water_displaced_squared 5 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_water_displaced_squared_5ft_l3962_396242


namespace NUMINAMATH_CALUDE_technician_round_trip_completion_l3962_396270

theorem technician_round_trip_completion (distance : ℝ) (h : distance > 0) :
  let one_way := distance
  let round_trip := 2 * distance
  let completed := distance + 0.2 * distance
  (completed / round_trip) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_technician_round_trip_completion_l3962_396270


namespace NUMINAMATH_CALUDE_parabola_sum_l3962_396278

/-- A parabola with equation y = ax^2 + bx + c, vertex (-3, 4), vertical axis of symmetry, 
    and passing through (4, -2) has a + b + c = 100/49 -/
theorem parabola_sum (a b c : ℚ) : 
  (∀ x y : ℚ, y = a * x^2 + b * x + c ↔ 
    (x = -3 ∧ y = 4) ∨ 
    (x = 4 ∧ y = -2) ∨ 
    (∃ k : ℚ, y - 4 = k * (x + 3)^2)) →
  a + b + c = 100/49 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l3962_396278


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3962_396223

/-- An isosceles triangle with given perimeter and leg length -/
structure IsoscelesTriangle where
  perimeter : ℝ
  leg_length : ℝ

/-- The base length of an isosceles triangle -/
def base_length (t : IsoscelesTriangle) : ℝ :=
  t.perimeter - 2 * t.leg_length

/-- Theorem: The base length of an isosceles triangle with perimeter 62 and leg length 25 is 12 -/
theorem isosceles_triangle_base_length :
  let t : IsoscelesTriangle := { perimeter := 62, leg_length := 25 }
  base_length t = 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3962_396223


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_ratio_l3962_396257

/-- An isosceles trapezoid with a point inside dividing it into four triangles -/
structure IsoscelesTrapezoidWithPoint where
  -- The lengths of the parallel bases
  AB : ℝ
  CD : ℝ
  -- Areas of the four triangles formed by the point
  area_PAB : ℝ
  area_PBC : ℝ
  area_PCD : ℝ
  area_PDA : ℝ
  -- Conditions
  AB_gt_CD : AB > CD
  areas_clockwise : area_PAB = 9 ∧ area_PBC = 7 ∧ area_PCD = 3 ∧ area_PDA = 5

/-- The ratio of the parallel bases in the isosceles trapezoid is 3 -/
theorem isosceles_trapezoid_ratio 
  (T : IsoscelesTrapezoidWithPoint) : T.AB / T.CD = 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_ratio_l3962_396257


namespace NUMINAMATH_CALUDE_equation_solution_existence_l3962_396279

theorem equation_solution_existence (a : ℝ) :
  (∃ x : ℝ, 3 * 4^(x - 2) + 27 = a + a * 4^(x - 2)) ↔ (3 < a ∧ a < 27) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_existence_l3962_396279


namespace NUMINAMATH_CALUDE_turtle_problem_l3962_396218

theorem turtle_problem (initial_turtles : ℕ) (h1 : initial_turtles = 25) :
  let additional_turtles := 5 * initial_turtles - 4
  let total_turtles := initial_turtles + additional_turtles
  let remaining_turtles := total_turtles - (total_turtles / 3)
  remaining_turtles = 98 := by
sorry

end NUMINAMATH_CALUDE_turtle_problem_l3962_396218


namespace NUMINAMATH_CALUDE_binomial_inequality_l3962_396254

theorem binomial_inequality (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m < n) :
  (n^n : ℚ) / (m^m * (n-m)^(n-m)) > (n.factorial : ℚ) / (m.factorial * (n-m).factorial) ∧
  (n.factorial : ℚ) / (m.factorial * (n-m).factorial) > (n^n : ℚ) / (m^m * (n+1) * (n-m)^(n-m)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_inequality_l3962_396254


namespace NUMINAMATH_CALUDE_triangle_side_length_l3962_396237

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if the area is √3, angle B is 60°, and a² + c² = 3ac, then b = 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : Real) : 
  (1/2) * a * c * Real.sin B = Real.sqrt 3 →
  B = π/3 →
  a^2 + c^2 = 3*a*c →
  b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3962_396237


namespace NUMINAMATH_CALUDE_a_integer_not_multiple_of_five_l3962_396226

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 6*x + 1 = 0

-- Define the sequence aₙ
def a (n : ℕ) (x₁ x₂ : ℝ) : ℝ := x₁^n + x₂^n

-- State the theorem
theorem a_integer_not_multiple_of_five 
  (x₁ x₂ : ℝ) 
  (h₁ : quadratic_equation x₁) 
  (h₂ : quadratic_equation x₂) :
  ∀ n : ℕ, ∃ k : ℤ, (a n x₁ x₂ = k) ∧ ¬(∃ m : ℤ, k = 5 * m) :=
by sorry

end NUMINAMATH_CALUDE_a_integer_not_multiple_of_five_l3962_396226


namespace NUMINAMATH_CALUDE_sweep_probability_l3962_396217

/-- Represents a clock with four equally spaced points -/
structure Clock :=
  (points : Fin 4 → ℕ)
  (equally_spaced : ∀ i : Fin 4, points i = i.val * 3)

/-- Represents a 20-minute period on the clock -/
def Period : ℕ := 20

/-- Calculates the number of favorable intervals in a 60-minute period -/
def favorable_intervals (c : Clock) (p : ℕ) : ℕ :=
  4 * 5  -- 4 intervals of 5 minutes each

/-- The probability of sweeping exactly two points in the given period -/
def probability (c : Clock) (p : ℕ) : ℚ :=
  (favorable_intervals c p : ℚ) / 60

/-- The main theorem stating the probability is 1/3 -/
theorem sweep_probability (c : Clock) :
  probability c Period = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sweep_probability_l3962_396217


namespace NUMINAMATH_CALUDE_farmer_apples_l3962_396206

/-- The number of apples the farmer has after giving some away -/
def remaining_apples (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: The farmer has 4337 apples after giving away 3588 from his initial 7925 apples -/
theorem farmer_apples : remaining_apples 7925 3588 = 4337 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l3962_396206


namespace NUMINAMATH_CALUDE_parabola_translation_l3962_396258

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation :
  let original := Parabola.mk (-2) 0 0
  let translated := translate original 1 (-3)
  translated = Parabola.mk (-2) 4 (-3) := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3962_396258


namespace NUMINAMATH_CALUDE_volleyball_lineup_count_l3962_396286

/-- Represents the number of ways to choose a starting lineup for a volleyball team -/
def volleyballLineupCount (totalPlayers : ℕ) (versatilePlayers : ℕ) (specializedPlayers : ℕ) : ℕ :=
  totalPlayers * (totalPlayers - 1) * versatilePlayers * (versatilePlayers - 1) * (versatilePlayers - 2)

/-- Theorem stating the number of ways to choose a starting lineup for a volleyball team with given conditions -/
theorem volleyball_lineup_count :
  volleyballLineupCount 10 8 2 = 30240 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_lineup_count_l3962_396286


namespace NUMINAMATH_CALUDE_manager_percentage_problem_l3962_396266

theorem manager_percentage_problem (total_employees : ℕ) 
  (managers_left : ℕ) (final_percentage : ℚ) :
  total_employees = 500 →
  managers_left = 250 →
  final_percentage = 98/100 →
  (total_employees - managers_left) * final_percentage = 
    total_employees - managers_left - 
    ((100 - 99)/100 * total_employees) →
  99/100 * total_employees = total_employees - 
    ((100 - 99)/100 * total_employees) :=
by sorry

end NUMINAMATH_CALUDE_manager_percentage_problem_l3962_396266


namespace NUMINAMATH_CALUDE_f_greater_than_one_exists_max_a_l3962_396235

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - (1/2) * a * x^2

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- Statement 1
theorem f_greater_than_one (x : ℝ) (h : x > 0) : f 2 x > 1 := by
  sorry

-- Statement 2
theorem exists_max_a :
  ∃ (a : ℕ), (∀ (x : ℝ), x > 0 → f_deriv a x ≥ x^2 * Real.log x) ∧
  (∀ (b : ℕ), (∀ (x : ℝ), x > 0 → f_deriv b x ≥ x^2 * Real.log x) → b ≤ a) := by
  sorry

end

end NUMINAMATH_CALUDE_f_greater_than_one_exists_max_a_l3962_396235


namespace NUMINAMATH_CALUDE_mark_can_bench_press_55_pounds_l3962_396295

/-- The weight that Mark can bench press -/
def marks_bench_press (daves_weight : ℝ) : ℝ :=
  let daves_bench_press := 3 * daves_weight
  let craigs_bench_press := 0.2 * daves_bench_press
  craigs_bench_press - 50

/-- Proof that Mark can bench press 55 pounds -/
theorem mark_can_bench_press_55_pounds :
  marks_bench_press 175 = 55 := by
  sorry

end NUMINAMATH_CALUDE_mark_can_bench_press_55_pounds_l3962_396295


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l3962_396209

theorem perfect_square_divisibility (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) :
  ∃ k : ℕ, a = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l3962_396209


namespace NUMINAMATH_CALUDE_trisection_intersection_l3962_396281

/-- Given two points on the natural logarithm curve, prove that the x-coordinate of the 
    intersection point between a horizontal line through the first trisection point and 
    the curve is 2^(7/3). -/
theorem trisection_intersection (A B C : ℝ × ℝ) : 
  A.1 = 2 → 
  A.2 = Real.log 2 →
  B.1 = 32 → 
  B.2 = Real.log 32 →
  C.2 = (2 / 3) * A.2 + (1 / 3) * B.2 →
  ∃ (x : ℝ), x > 0 ∧ Real.log x = C.2 →
  x = 2^(7/3) := by
sorry

end NUMINAMATH_CALUDE_trisection_intersection_l3962_396281


namespace NUMINAMATH_CALUDE_expression_value_l3962_396225

theorem expression_value (x y : ℝ) (h : x - 2 * y^2 = 1) :
  -2 * x + 4 * y^2 + 1 = -1 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3962_396225


namespace NUMINAMATH_CALUDE_bob_has_22_pennies_l3962_396213

/-- The number of pennies Alex currently has -/
def alex_pennies : ℕ := sorry

/-- The number of pennies Bob currently has -/
def bob_pennies : ℕ := sorry

/-- Condition 1: If Alex gives Bob two pennies, Bob will have four times as many pennies as Alex has left -/
axiom condition1 : bob_pennies + 2 = 4 * (alex_pennies - 2)

/-- Condition 2: If Bob gives Alex two pennies, Bob will have twice as many pennies as Alex has -/
axiom condition2 : bob_pennies - 2 = 2 * (alex_pennies + 2)

/-- Theorem: Bob currently has 22 pennies -/
theorem bob_has_22_pennies : bob_pennies = 22 := by sorry

end NUMINAMATH_CALUDE_bob_has_22_pennies_l3962_396213


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3962_396274

-- Define the quadratic function
def f (t x : ℝ) : ℝ := x^2 - 2*t*x + 3

-- State the theorem
theorem quadratic_function_properties (t : ℝ) (h_t : t > 0) :
  -- Part 1
  (f t 2 = 1 → t = 3/2) ∧
  -- Part 2
  (∃ (x_min : ℝ), 0 ≤ x_min ∧ x_min ≤ 3 ∧
    (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f t x ≥ f t x_min) ∧
    f t x_min = -2 → t = Real.sqrt 5) ∧
  -- Part 3
  (∀ (m a b : ℝ),
    f t (m - 2) = a ∧ f t 4 = b ∧ f t m = a ∧ a < b ∧ b < 3 →
    (3 < m ∧ m < 4) ∨ m > 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3962_396274


namespace NUMINAMATH_CALUDE_volleyball_team_starters_l3962_396239

-- Define the total number of players
def total_players : ℕ := 16

-- Define the number of quadruplets
def quadruplets : ℕ := 4

-- Define the number of starters to choose
def starters : ℕ := 6

-- Define the maximum number of quadruplets allowed in the starting lineup
def max_quadruplets_in_lineup : ℕ := 1

-- Theorem statement
theorem volleyball_team_starters :
  (Nat.choose (total_players - quadruplets) starters) +
  (quadruplets * Nat.choose (total_players - quadruplets) (starters - 1)) = 4092 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_starters_l3962_396239


namespace NUMINAMATH_CALUDE_truncated_cone_inscribed_sphere_l3962_396245

/-- Given a truncated cone with an inscribed sphere, this theorem relates the ratio of their volumes
    to the angle between the generatrix and the base of the cone, and specifies the allowable values for the ratio. -/
theorem truncated_cone_inscribed_sphere (k : ℝ) (α : ℝ) :
  k > (3/2) →
  (∃ (V_cone V_sphere : ℝ), V_cone > 0 ∧ V_sphere > 0 ∧ V_cone / V_sphere = k) →
  α = Real.arctan (2 / Real.sqrt (2 * k - 3)) ∧
  α = angle_between_generatrix_and_base :=
by sorry

/-- Defines the angle between the generatrix and the base of the truncated cone. -/
def angle_between_generatrix_and_base : ℝ :=
sorry

end NUMINAMATH_CALUDE_truncated_cone_inscribed_sphere_l3962_396245


namespace NUMINAMATH_CALUDE_trig_expression_equals_negative_four_l3962_396233

theorem trig_expression_equals_negative_four :
  (Real.sqrt 3 * Real.sin (10 * π / 180) - Real.cos (10 * π / 180)) /
  (Real.cos (10 * π / 180) * Real.sin (10 * π / 180)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_negative_four_l3962_396233


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3962_396243

/-- A geometric sequence with first term a₁ and common ratio q. -/
def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ :=
  λ n => a₁ * q^(n - 1)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (h_geom : ∃ (a₁ q : ℝ), ∀ n, a n = geometric_sequence a₁ q n)
  (h_a₁ : a 1 = 2) (h_a₄ : a 4 = 16) :
  ∃ q, ∀ n, a n = geometric_sequence 2 q n ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3962_396243


namespace NUMINAMATH_CALUDE_part_one_part_two_l3962_396262

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_arithmetic_sequence (B A C : ℝ) : Prop :=
  ∃ d : ℝ, A - B = C - A ∧ A - B = d

-- Theorem 1
theorem part_one (t : Triangle) (m : ℝ) 
  (h1 : is_arithmetic_sequence t.B t.A t.C)
  (h2 : t.a^2 - t.c^2 = t.b^2 - m*t.b*t.c) : 
  m = 1 := by sorry

-- Theorem 2
theorem part_two (t : Triangle)
  (h1 : is_arithmetic_sequence t.B t.A t.C)
  (h2 : t.a = Real.sqrt 3)
  (h3 : t.b + t.c = 3) :
  (1/2 : ℝ) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3962_396262


namespace NUMINAMATH_CALUDE_tom_run_distance_l3962_396250

theorem tom_run_distance (total_distance : ℝ) (walk_speed : ℝ) (run_speed : ℝ) 
  (friend_time : ℝ) (max_total_time : ℝ) :
  total_distance = 2800 →
  walk_speed = 75 →
  run_speed = 225 →
  friend_time = 5 →
  max_total_time = 30 →
  ∃ (x : ℝ), x ≥ 0 ∧ x ≤ total_distance ∧
    (x / walk_speed + (total_distance - x) / run_speed + friend_time ≤ max_total_time) ∧
    (total_distance - x ≤ 1387.5) :=
by sorry

end NUMINAMATH_CALUDE_tom_run_distance_l3962_396250


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l3962_396256

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 210) (h2 : b = 4620) :
  (Nat.gcd a b) * (3 * Nat.lcm a b) = 2910600 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l3962_396256


namespace NUMINAMATH_CALUDE_fourth_number_ninth_row_l3962_396241

/-- Represents the lattice structure with the given pattern -/
def lattice_sequence (row : ℕ) (position : ℕ) : ℕ :=
  8 * (row - 1) + position

/-- The problem statement -/
theorem fourth_number_ninth_row :
  lattice_sequence 9 4 = 68 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_ninth_row_l3962_396241


namespace NUMINAMATH_CALUDE_alternating_color_probability_alternating_color_probability_value_l3962_396284

/-- The probability of drawing 10 balls with alternating colors (starting and ending with the same color) from a box containing 5 white and 5 black balls. -/
theorem alternating_color_probability : ℚ :=
  let total_balls : ℕ := 10
  let white_balls : ℕ := 5
  let black_balls : ℕ := 5
  let successful_sequences : ℕ := 2
  let total_arrangements : ℕ := Nat.choose total_balls white_balls
  successful_sequences / total_arrangements

/-- The probability of drawing 10 balls with alternating colors (starting and ending with the same color) from a box containing 5 white and 5 black balls is 1/126. -/
theorem alternating_color_probability_value : alternating_color_probability = 1 / 126 := by
  sorry

end NUMINAMATH_CALUDE_alternating_color_probability_alternating_color_probability_value_l3962_396284


namespace NUMINAMATH_CALUDE_perimeter_3x3_grid_l3962_396298

/-- The perimeter of a square grid of unit squares -/
def grid_perimeter (rows columns : ℕ) : ℕ :=
  2 * (rows + columns)

/-- Theorem: The perimeter of a 3x3 grid of unit squares is 18 -/
theorem perimeter_3x3_grid : grid_perimeter 3 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_3x3_grid_l3962_396298


namespace NUMINAMATH_CALUDE_fraction_sum_difference_l3962_396234

theorem fraction_sum_difference : (3 / 50 + 2 / 25 - 5 / 1000 : ℚ) = 0.135 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_l3962_396234


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3962_396252

theorem imaginary_part_of_z : Complex.im ((1 + Complex.I)^2 + Complex.I^2011) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3962_396252


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l3962_396230

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≤ a (n + 1)

theorem geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) :
  (geometric_sequence a q) →
  (¬(((a 1 * q > 0) → increasing_sequence a) ∧
     (increasing_sequence a → (a 1 * q > 0)))) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l3962_396230


namespace NUMINAMATH_CALUDE_integer_set_property_l3962_396296

theorem integer_set_property (n : ℕ+) :
  ∃ (S : Finset ℤ), Finset.card S = n ∧
    ∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b →
      ∃ (k : ℤ), a * b = k * (a - b)^2 :=
sorry

end NUMINAMATH_CALUDE_integer_set_property_l3962_396296


namespace NUMINAMATH_CALUDE_intersection_distance_l3962_396292

/-- The distance between the intersection points of y = -2 and y = 3x^2 + 2x - 5 -/
theorem intersection_distance : 
  let f (x : ℝ) := 3 * x^2 + 2 * x - 5
  let y := -2
  let roots := {x : ℝ | f x = y}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ roots ∧ x₂ ∈ roots ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = 2 * Real.sqrt 10 / 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l3962_396292


namespace NUMINAMATH_CALUDE_vasya_drove_two_fifths_l3962_396214

/-- Represents the fraction of the total distance driven by each person -/
structure DistanceFractions where
  anton : ℝ
  vasya : ℝ
  sasha : ℝ
  dima : ℝ

/-- Conditions of the driving problem -/
def driving_conditions (d : DistanceFractions) : Prop :=
  d.anton + d.vasya + d.sasha + d.dima = 1 ∧  -- Total distance is 1
  d.anton = d.vasya / 2 ∧                     -- Anton drove half of Vasya's distance
  d.sasha = d.anton + d.dima ∧                -- Sasha drove as long as Anton and Dima combined
  d.dima = 1 / 10                             -- Dima drove one-tenth of the distance

/-- Theorem: Under the given conditions, Vasya drove 2/5 of the total distance -/
theorem vasya_drove_two_fifths (d : DistanceFractions) 
  (h : driving_conditions d) : d.vasya = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vasya_drove_two_fifths_l3962_396214


namespace NUMINAMATH_CALUDE_sum_r_s_equals_48_l3962_396253

/-- Parabola P with equation y = x^2 + 4x + 4 -/
def P : ℝ → ℝ := λ x => x^2 + 4*x + 4

/-- Point Q (10, 24) -/
def Q : ℝ × ℝ := (10, 24)

/-- Line through Q with slope m -/
def line_through_Q (m : ℝ) : ℝ → ℝ := λ x => m * (x - Q.1) + Q.2

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line_through_Q m x

/-- Theorem: Sum of r and s equals 48 -/
theorem sum_r_s_equals_48 (r s : ℝ) 
  (h : ∀ m, no_intersection m ↔ r < m ∧ m < s) : 
  r + s = 48 := by sorry

end NUMINAMATH_CALUDE_sum_r_s_equals_48_l3962_396253


namespace NUMINAMATH_CALUDE_town_population_problem_l3962_396205

theorem town_population_problem :
  ∃ n : ℕ, 
    (∃ a b : ℕ, 
      n * (n + 1) / 2 + 121 = a^2 ∧
      n * (n + 1) / 2 + 121 + 144 = b^2) ∧
    n * (n + 1) / 2 = 2280 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l3962_396205


namespace NUMINAMATH_CALUDE_dot_product_PA_PB_is_negative_one_l3962_396294

/-- The dot product of vectors PA and PB is always -1, where P is any point on the curve y = x + 2/x (x > 0),
    A is the foot of the perpendicular from P to y = x, and B is the foot of the perpendicular from P to the y-axis. -/
theorem dot_product_PA_PB_is_negative_one :
  ∀ (x₀ : ℝ), x₀ > 0 →
  let P : ℝ × ℝ := (x₀, x₀ + 2 / x₀)
  let A : ℝ × ℝ := (x₀ + 1 / x₀, x₀ + 1 / x₀)
  let B : ℝ × ℝ := (0, x₀ + 2 / x₀)
  let PA : ℝ × ℝ := (A.1 - P.1, A.2 - P.2)
  let PB : ℝ × ℝ := (B.1 - P.1, B.2 - P.2)
  (PA.1 * PB.1 + PA.2 * PB.2 : ℝ) = -1 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_PA_PB_is_negative_one_l3962_396294


namespace NUMINAMATH_CALUDE_increase_mode_effect_l3962_396259

def shoe_sizes : List ℕ := [35, 36, 37, 38, 39]
def sales_quantities : List ℕ := [2, 8, 10, 6, 2]

def mode (l : List ℕ) : ℕ := sorry

def mean (l : List ℕ) : ℚ := sorry

def median (l : List ℕ) : ℚ := sorry

def variance (l : List ℕ) : ℚ := sorry

theorem increase_mode_effect 
  (most_common : ℕ) 
  (h1 : most_common ∈ shoe_sizes) 
  (h2 : ∀ x ∈ shoe_sizes, (sales_quantities.count most_common) ≥ (sales_quantities.count x)) :
  ∃ n : ℕ, 
    (mode (sales_quantities.map (λ x => if x = most_common then x + n else x)) = mode sales_quantities) ∧
    (mean (sales_quantities.map (λ x => if x = most_common then x + n else x)) ≠ mean sales_quantities ∨
     median (sales_quantities.map (λ x => if x = most_common then x + n else x)) = median sales_quantities ∨
     variance (sales_quantities.map (λ x => if x = most_common then x + n else x)) ≠ variance sales_quantities) :=
by sorry

end NUMINAMATH_CALUDE_increase_mode_effect_l3962_396259


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l3962_396267

theorem imaginary_part_of_complex_division : 
  let z : ℂ := 1 / (2 + Complex.I)
  Complex.im z = -1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l3962_396267


namespace NUMINAMATH_CALUDE_expression_evaluation_l3962_396285

theorem expression_evaluation :
  let f (x : ℚ) := ((x + 1) / (x - 1) - 1) * ((x + 1) / (x - 1) + 1)
  f (-1/2) = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3962_396285


namespace NUMINAMATH_CALUDE_sqrt_three_plus_two_range_l3962_396210

theorem sqrt_three_plus_two_range :
  ∃ (x : ℝ), x = Real.sqrt 3 ∧ Irrational x ∧ 1 < x ∧ x < 2 → 3.5 < x + 2 ∧ x + 2 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_plus_two_range_l3962_396210


namespace NUMINAMATH_CALUDE_phone_watch_sales_l3962_396268

/-- Represents the total sales amount for two months of phone watch sales -/
def total_sales (x : ℕ) : ℝ := 600 * 60 + 500 * (x - 60)

/-- States that the total sales amount is no less than $86000 -/
def sales_condition (x : ℕ) : Prop := total_sales x ≥ 86000

theorem phone_watch_sales (x : ℕ) : 
  sales_condition x ↔ 600 * 60 + 500 * (x - 60) ≥ 86000 := by sorry

end NUMINAMATH_CALUDE_phone_watch_sales_l3962_396268


namespace NUMINAMATH_CALUDE_probability_of_two_red_balls_l3962_396293

-- Define the number of balls of each color
def red_balls : ℕ := 5
def blue_balls : ℕ := 4
def green_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := red_balls + blue_balls + green_balls

-- Define the number of balls to be picked
def balls_picked : ℕ := 2

-- Theorem statement
theorem probability_of_two_red_balls :
  (Nat.choose red_balls balls_picked : ℚ) / (Nat.choose total_balls balls_picked) = 5 / 33 :=
sorry

end NUMINAMATH_CALUDE_probability_of_two_red_balls_l3962_396293


namespace NUMINAMATH_CALUDE_remainder_problem_l3962_396289

theorem remainder_problem (x : ℤ) : x % 95 = 31 → x % 19 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3962_396289


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l3962_396207

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem solution_part1 (a : ℝ) (h : a ≤ 2) :
  {x : ℝ | f a x ≥ 2} = {x : ℝ | x ≤ 1/2 ∨ x ≥ 5/2} :=
sorry

-- Part 2
theorem solution_part2 (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, f a x + |x - 1| ≥ 1) → a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l3962_396207


namespace NUMINAMATH_CALUDE_triangle_inradius_l3962_396249

/-- Given a triangle with perimeter 32 cm and area 56 cm², its inradius is 3.5 cm. -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
    (h1 : p = 32) 
    (h2 : A = 56) 
    (h3 : A = r * p / 2) : r = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l3962_396249


namespace NUMINAMATH_CALUDE_PB_equation_l3962_396204

-- Define the points A, B, and P
variable (A B P : ℝ × ℝ)

-- Define the conditions
axiom A_on_x_axis : A.2 = 0
axiom B_on_x_axis : B.2 = 0
axiom P_x_coord : P.1 = 2
axiom PA_PB_equal : (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2
axiom PA_equation : ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.1 - p.2 + 1 = 0} ↔ (x - P.1) * (A.2 - P.2) = (y - P.2) * (A.1 - P.1)

-- State the theorem
theorem PB_equation :
  ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.1 + p.2 - 5 = 0} ↔ (x - P.1) * (B.2 - P.2) = (y - P.2) * (B.1 - P.1) :=
sorry

end NUMINAMATH_CALUDE_PB_equation_l3962_396204


namespace NUMINAMATH_CALUDE_tan_equality_l3962_396203

theorem tan_equality : 
  3.439 * Real.tan (110 * π / 180) + Real.tan (50 * π / 180) + Real.tan (20 * π / 180) = 
  Real.tan (110 * π / 180) * Real.tan (50 * π / 180) * Real.tan (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_l3962_396203


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l3962_396224

theorem complex_magnitude_theorem (s : ℝ) (w : ℂ) 
  (h1 : |s| < 3) 
  (h2 : w + 2 / w = s) : 
  Complex.abs w = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l3962_396224


namespace NUMINAMATH_CALUDE_extreme_value_condition_l3962_396208

theorem extreme_value_condition (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ (a * x^2 - 1) * Real.exp x
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l3962_396208


namespace NUMINAMATH_CALUDE_range_of_positive_integers_in_list_K_l3962_396202

def consecutive_integers (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => start + i)

def positive_integers (l : List ℤ) : List ℤ :=
  l.filter (λ x => x > 0)

def range (l : List ℤ) : ℤ :=
  l.maximum.getD 0 - l.minimum.getD 0

theorem range_of_positive_integers_in_list_K :
  let K := consecutive_integers (-5) 12
  range (positive_integers K) = 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_positive_integers_in_list_K_l3962_396202


namespace NUMINAMATH_CALUDE_direction_vector_coefficient_l3962_396276

/-- Given a line passing through points (-2, 5) and (1, 0), prove that its direction vector of the form (a, -1) has a = 3/5 -/
theorem direction_vector_coefficient (p1 p2 : ℝ × ℝ) (a : ℝ) : 
  p1 = (-2, 5) → p2 = (1, 0) → 
  (p2.1 - p1.1, p2.2 - p1.2) = (3 * a, -3 * a) → 
  a = 3/5 := by sorry

end NUMINAMATH_CALUDE_direction_vector_coefficient_l3962_396276


namespace NUMINAMATH_CALUDE_theta_range_l3962_396291

theorem theta_range (θ : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, x^2 * Real.cos θ - x * (1 - x) + (1 - x^2) * Real.sin θ > 0) →
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 6 < θ ∧ θ < 2 * k * Real.pi + Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_theta_range_l3962_396291


namespace NUMINAMATH_CALUDE_defective_bulb_probability_l3962_396201

/-- The probability of selecting at least one defective bulb when choosing two bulbs at random from a box containing 24 bulbs, of which 4 are defective, is 43/138. -/
theorem defective_bulb_probability (total_bulbs : ℕ) (defective_bulbs : ℕ) 
  (h1 : total_bulbs = 24) (h2 : defective_bulbs = 4) :
  let non_defective : ℕ := total_bulbs - defective_bulbs
  let prob_both_non_defective : ℚ := (non_defective / total_bulbs) * ((non_defective - 1) / (total_bulbs - 1))
  1 - prob_both_non_defective = 43 / 138 := by
  sorry

end NUMINAMATH_CALUDE_defective_bulb_probability_l3962_396201


namespace NUMINAMATH_CALUDE_committee_selection_ways_l3962_396275

-- Define the total number of team owners
def total_owners : ℕ := 30

-- Define the number of owners who don't want to serve
def ineligible_owners : ℕ := 3

-- Define the size of the committee
def committee_size : ℕ := 5

-- Define the number of eligible owners
def eligible_owners : ℕ := total_owners - ineligible_owners

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem committee_selection_ways : 
  combination eligible_owners committee_size = 65780 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l3962_396275


namespace NUMINAMATH_CALUDE_jimin_seokjin_money_sum_l3962_396228

/-- Calculates the total amount of money for a person given their coin distribution --/
def calculate_total (coins_100 : Nat) (coins_50 : Nat) (coins_10 : Nat) : Nat :=
  100 * coins_100 + 50 * coins_50 + 10 * coins_10

/-- Represents the coin distribution and total money for Jimin and Seokjin --/
theorem jimin_seokjin_money_sum :
  let jimin_total := calculate_total 5 1 0
  let seokjin_total := calculate_total 2 0 7
  jimin_total + seokjin_total = 820 := by
  sorry

#check jimin_seokjin_money_sum

end NUMINAMATH_CALUDE_jimin_seokjin_money_sum_l3962_396228


namespace NUMINAMATH_CALUDE_book_ratio_problem_l3962_396222

theorem book_ratio_problem (lit sci : ℕ) (h : lit * 5 = sci * 8) : 
  (lit - sci : ℚ) / sci = 3 / 5 ∧ (lit - sci : ℚ) / lit = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_book_ratio_problem_l3962_396222


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_difference_l3962_396216

/-- The sum of 0.666... (repeating) and 0.222... (repeating) minus 0.444... (repeating) equals 4/9 -/
theorem repeating_decimal_sum_difference (x y z : ℚ) :
  x = 2/3 ∧ y = 2/9 ∧ z = 4/9 →
  x + y - z = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_difference_l3962_396216


namespace NUMINAMATH_CALUDE_complement_of_intersection_l3962_396200

def U : Set Nat := {1,2,3,4,5}
def A : Set Nat := {1,2,3}
def B : Set Nat := {2,3,4}

theorem complement_of_intersection (U A B : Set Nat) 
  (hU : U = {1,2,3,4,5}) 
  (hA : A = {1,2,3}) 
  (hB : B = {2,3,4}) : 
  (A ∩ B)ᶜ = {1,4,5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l3962_396200


namespace NUMINAMATH_CALUDE_braden_final_amount_l3962_396290

/-- Calculates the final amount in Braden's money box after winning a bet -/
def final_amount (initial_amount : ℕ) (bet_multiplier : ℕ) : ℕ :=
  initial_amount + bet_multiplier * initial_amount

/-- Theorem stating that given the initial conditions, Braden's final amount is $1200 -/
theorem braden_final_amount :
  let initial_amount : ℕ := 400
  let bet_multiplier : ℕ := 2
  final_amount initial_amount bet_multiplier = 1200 := by sorry

end NUMINAMATH_CALUDE_braden_final_amount_l3962_396290


namespace NUMINAMATH_CALUDE_boys_in_school_after_increase_l3962_396238

/-- The number of boys in a school after an increase -/
def boys_after_increase (initial_boys : ℕ) (additional_boys : ℕ) : ℕ :=
  initial_boys + additional_boys

theorem boys_in_school_after_increase :
  boys_after_increase 214 910 = 1124 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_school_after_increase_l3962_396238


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l3962_396212

theorem simplify_and_ratio (m : ℚ) : 
  let expr := (6 * m + 18) / 6
  let simplified := m + 3
  expr = simplified ∧ 
  (∃ (c d : ℤ), simplified = c * m + d ∧ c / d = 1 / 3) := by
sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l3962_396212


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_1_minimum_value_range_l3962_396255

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := |3 * x - 1| + a * x + 3

-- Part 1: Prove the solution set for f(x) ≤ 5 when a = 1
theorem solution_set_for_a_eq_1 :
  {x : ℝ | f 1 x ≤ 5} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/4} := by sorry

-- Part 2: Prove the range of a for which f(x) has a minimum value
theorem minimum_value_range :
  {a : ℝ | ∃ (x : ℝ), ∀ (y : ℝ), f a x ≤ f a y} = {a : ℝ | -3 ≤ a ∧ a ≤ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_1_minimum_value_range_l3962_396255


namespace NUMINAMATH_CALUDE_least_four_digit_solution_l3962_396227

theorem least_four_digit_solution (x : ℕ) : x = 1002 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧
  (∀ y : ℕ, y ≥ 1000 ∧ y < 10000 →
    (5 * y ≡ 10 [ZMOD 10] ∧
     3 * y + 20 ≡ 29 [ZMOD 12] ∧
     -3 * y + 2 ≡ 2 * y [ZMOD 30]) →
    x ≤ y) ∧
  (5 * x ≡ 10 [ZMOD 10]) ∧
  (3 * x + 20 ≡ 29 [ZMOD 12]) ∧
  (-3 * x + 2 ≡ 2 * x [ZMOD 30]) := by
sorry

end NUMINAMATH_CALUDE_least_four_digit_solution_l3962_396227


namespace NUMINAMATH_CALUDE_sin_240_l3962_396219

-- Define the cofunction identity
axiom cofunction_identity (α : Real) : Real.sin (180 + α) = -Real.sin α

-- Define the special angle value
axiom sin_60 : Real.sin 60 = Real.sqrt 3 / 2

-- State the theorem to be proved
theorem sin_240 : Real.sin 240 = -(Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_240_l3962_396219


namespace NUMINAMATH_CALUDE_g_zero_at_three_l3962_396211

-- Define the polynomial g(x)
def g (x s : ℝ) : ℝ := 3 * x^5 + 2 * x^4 - x^3 + 2 * x^2 - 5 * x + s

-- Theorem statement
theorem g_zero_at_three (s : ℝ) : g 3 s = 0 ↔ s = -867 := by sorry

end NUMINAMATH_CALUDE_g_zero_at_three_l3962_396211


namespace NUMINAMATH_CALUDE_max_mineral_value_l3962_396280

/-- Represents a type of mineral with its weight and value --/
structure Mineral where
  weight : ℕ
  value : ℕ

/-- The problem setup --/
def mineral_problem : Prop :=
  ∃ (j k l : Mineral) (x y z : ℕ),
    j.weight = 6 ∧ j.value = 17 ∧
    k.weight = 3 ∧ k.value = 9 ∧
    l.weight = 2 ∧ l.value = 5 ∧
    x * j.weight + y * k.weight + z * l.weight ≤ 20 ∧
    ∀ (a b c : ℕ),
      a * j.weight + b * k.weight + c * l.weight ≤ 20 →
      a * j.value + b * k.value + c * l.value ≤ x * j.value + y * k.value + z * l.value ∧
      x * j.value + y * k.value + z * l.value = 60

theorem max_mineral_value : mineral_problem := by sorry

end NUMINAMATH_CALUDE_max_mineral_value_l3962_396280


namespace NUMINAMATH_CALUDE_cos_150_degrees_l3962_396229

theorem cos_150_degrees :
  Real.cos (150 * π / 180) = -(1 / 2) := by sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l3962_396229


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3962_396247

theorem geometric_sequence_minimum_value (a b c : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- a, b, c form a geometric sequence
  (∀ x : ℝ, (x - 2) * Real.exp x ≥ b) →  -- b is the minimum value of (x-2)e^x
  a * c = Real.exp 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3962_396247
