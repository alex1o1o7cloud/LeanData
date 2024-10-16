import Mathlib

namespace NUMINAMATH_CALUDE_green_ball_probability_l1999_199963

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a container -/
def containerProb : ℚ := 1 / 3

/-- Calculate the probability of drawing a green ball from a container -/
def greenProb (c : Container) : ℚ := c.green / (c.red + c.green)

/-- The three containers A, B, and C -/
def containerA : Container := ⟨5, 5⟩
def containerB : Container := ⟨8, 2⟩
def containerC : Container := ⟨3, 7⟩

/-- The probability of selecting a green ball -/
def probGreenBall : ℚ :=
  containerProb * greenProb containerA +
  containerProb * greenProb containerB +
  containerProb * greenProb containerC

theorem green_ball_probability :
  probGreenBall = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l1999_199963


namespace NUMINAMATH_CALUDE_area_under_curve_l1999_199944

-- Define the curve
def curve (x : ℝ) : ℝ := 3 * x^2

-- Define the bounds of the region
def lower_bound : ℝ := 0
def upper_bound : ℝ := 1

-- Theorem statement
theorem area_under_curve :
  ∫ x in lower_bound..upper_bound, curve x = 1 := by sorry

end NUMINAMATH_CALUDE_area_under_curve_l1999_199944


namespace NUMINAMATH_CALUDE_trailing_zeros_count_l1999_199947

def N : ℕ := 10^2018 + 1

theorem trailing_zeros_count (n : ℕ) : 
  ∃ k : ℕ, (N^2017 - 1) % 10^2018 = 0 ∧ (N^2017 - 1) % 10^2019 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_count_l1999_199947


namespace NUMINAMATH_CALUDE_number_sequence_count_l1999_199939

theorem number_sequence_count : ∀ (N : ℕ) (S : ℝ),
  S / N = 44 →
  (11 * 48 + 11 * 41 - 55) / N = 44 →
  N = 21 := by
sorry

end NUMINAMATH_CALUDE_number_sequence_count_l1999_199939


namespace NUMINAMATH_CALUDE_real_number_classification_l1999_199949

theorem real_number_classification :
  Set.univ = {x : ℝ | x > 0} ∪ {x : ℝ | x < 0} ∪ {(0 : ℝ)} := by sorry

end NUMINAMATH_CALUDE_real_number_classification_l1999_199949


namespace NUMINAMATH_CALUDE_complex_number_range_l1999_199916

theorem complex_number_range (a : ℝ) (z : ℂ) : 
  z = 2 + (a + 1) * I → Complex.abs z < 2 * Real.sqrt 2 → -3 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_range_l1999_199916


namespace NUMINAMATH_CALUDE_ceiling_floor_calculation_l1999_199989

theorem ceiling_floor_calculation : 
  ⌈(15 : ℚ) / 8 * (-34 : ℚ) / 4⌉ - ⌊(15 : ℚ) / 8 * ⌊(-34 : ℚ) / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_calculation_l1999_199989


namespace NUMINAMATH_CALUDE_factorial_difference_not_seven_l1999_199966

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem factorial_difference_not_seven (a b : ℕ) (h : b > a) :
  ∃ k : ℕ, (factorial b - factorial a) % 10 ≠ 7 :=
sorry

end NUMINAMATH_CALUDE_factorial_difference_not_seven_l1999_199966


namespace NUMINAMATH_CALUDE_average_difference_l1999_199968

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 115) 
  (h2 : (b + c) / 2 = 160) : 
  a - c = -90 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l1999_199968


namespace NUMINAMATH_CALUDE_greeting_cards_exchange_l1999_199955

theorem greeting_cards_exchange (x : ℕ) : x > 0 → x * (x - 1) = 1980 → ∀ (i j : ℕ), i < x ∧ j < x ∧ i ≠ j → ∃ (total : ℕ), total = 1980 ∧ total = x * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_greeting_cards_exchange_l1999_199955


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l1999_199965

theorem inscribed_cube_volume (large_cube_edge : ℝ) (sphere_diameter : ℝ) (small_cube_edge : ℝ) :
  large_cube_edge = 12 →
  sphere_diameter = large_cube_edge →
  small_cube_edge * Real.sqrt 3 = sphere_diameter →
  small_cube_edge ^ 3 = 192 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l1999_199965


namespace NUMINAMATH_CALUDE_complex_number_equivalence_l1999_199946

theorem complex_number_equivalence : 
  let z : ℂ := (1 - I) / (2 + I)
  z = 1/5 - 3/5*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_equivalence_l1999_199946


namespace NUMINAMATH_CALUDE_regular_18gon_relation_l1999_199976

/-- A regular 18-gon inscribed in a circle -/
structure Regular18Gon where
  /-- The radius of the circumscribed circle -/
  r : ℝ
  /-- The side length of the 18-gon -/
  a : ℝ
  /-- The radius is positive -/
  r_pos : 0 < r

/-- Theorem: For a regular 18-gon inscribed in a circle, a^3 + r^3 = 3ar^2 -/
theorem regular_18gon_relation (polygon : Regular18Gon) : 
  polygon.a^3 + polygon.r^3 = 3 * polygon.a * polygon.r^2 := by
  sorry

end NUMINAMATH_CALUDE_regular_18gon_relation_l1999_199976


namespace NUMINAMATH_CALUDE_range_of_m_l1999_199940

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the property of having two roots in [0, 2]
def has_two_roots_in_interval (m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 0 ≤ x₁ ∧ x₁ ≤ 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ 2 ∧ 
  f x₁ + m = 0 ∧ f x₂ + m = 0

-- Theorem statement
theorem range_of_m (m : ℝ) :
  has_two_roots_in_interval m → 0 ≤ m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1999_199940


namespace NUMINAMATH_CALUDE_remaining_space_for_regular_toenails_l1999_199994

/-- Represents the capacity of the jar in terms of regular toenails -/
def jarCapacity : ℕ := 100

/-- Represents the space occupied by a big toenail in terms of regular toenails -/
def bigToenailSpace : ℕ := 2

/-- Represents the number of big toenails already in the jar -/
def bigToenailsInJar : ℕ := 20

/-- Represents the number of regular toenails already in the jar -/
def regularToenailsInJar : ℕ := 40

/-- Theorem stating that the remaining space in the jar can fit exactly 20 regular toenails -/
theorem remaining_space_for_regular_toenails : 
  jarCapacity - (bigToenailsInJar * bigToenailSpace + regularToenailsInJar) = 20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_space_for_regular_toenails_l1999_199994


namespace NUMINAMATH_CALUDE_opposite_seats_imply_38_seats_l1999_199923

/-- Represents a round table with equally spaced seats -/
structure RoundTable where
  total_seats : ℕ
  seats_numbered_clockwise : Bool

/-- Defines two people sitting opposite each other on a round table -/
structure OppositeSeats (table : RoundTable) where
  seat1 : ℕ
  seat2 : ℕ
  are_opposite : seat2 - seat1 = table.total_seats / 2

/-- Theorem stating that if two people sit in seats 10 and 29 opposite each other,
    then the total number of seats is 38 -/
theorem opposite_seats_imply_38_seats (table : RoundTable)
  (opposite_pair : OppositeSeats table)
  (h1 : opposite_pair.seat1 = 10)
  (h2 : opposite_pair.seat2 = 29)
  (h3 : table.seats_numbered_clockwise = true) :
  table.total_seats = 38 := by
  sorry

end NUMINAMATH_CALUDE_opposite_seats_imply_38_seats_l1999_199923


namespace NUMINAMATH_CALUDE_parabola_line_slope_l1999_199948

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through a point with a given slope
def line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define a point on the latus rectum
def on_latus_rectum (x y : ℝ) : Prop := x = -1

-- Define a point in the first quadrant
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the midpoint condition
def is_midpoint (x1 y1 x2 y2 x3 y3 : ℝ) : Prop := 
  x2 = (x1 + x3) / 2 ∧ y2 = (y1 + y3) / 2

theorem parabola_line_slope (k : ℝ) (x1 y1 x2 y2 x3 y3 : ℝ) : 
  parabola x1 y1 →
  parabola x2 y2 →
  line k x1 y1 →
  line k x2 y2 →
  line k x3 y3 →
  on_latus_rectum x3 y3 →
  in_first_quadrant x1 y1 →
  is_midpoint x1 y1 x2 y2 x3 y3 →
  k = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_line_slope_l1999_199948


namespace NUMINAMATH_CALUDE_log_eight_three_equals_512_l1999_199912

theorem log_eight_three_equals_512 (y : ℝ) :
  Real.log y / Real.log 8 = 3 → y = 512 := by
  sorry

end NUMINAMATH_CALUDE_log_eight_three_equals_512_l1999_199912


namespace NUMINAMATH_CALUDE_meaningful_fraction_l1999_199973

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l1999_199973


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1999_199926

def p (x : ℝ) : Prop := x = 1

def q (x : ℝ) : Prop := x^3 - 2*x + 1 = 0

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1999_199926


namespace NUMINAMATH_CALUDE_toms_spending_ratio_l1999_199962

def monthly_allowance : ℚ := 12
def first_week_spending_ratio : ℚ := 1/3
def remaining_money : ℚ := 6

theorem toms_spending_ratio :
  let first_week_spending := monthly_allowance * first_week_spending_ratio
  let money_after_first_week := monthly_allowance - first_week_spending
  let second_week_spending := money_after_first_week - remaining_money
  second_week_spending / money_after_first_week = 1/4 := by
sorry

end NUMINAMATH_CALUDE_toms_spending_ratio_l1999_199962


namespace NUMINAMATH_CALUDE_symmetrical_triangles_are_congruent_l1999_199931

/-- Two triangles are symmetrical about a line if each point of one triangle has a corresponding point in the other triangle that is equidistant from the line of symmetry. -/
def symmetrical_triangles (t1 t2 : Set Point) (l : Line) : Prop := sorry

/-- Two triangles are congruent if they have the same shape and size. -/
def congruent_triangles (t1 t2 : Set Point) : Prop := sorry

/-- If two triangles are symmetrical about a line, then they are congruent. -/
theorem symmetrical_triangles_are_congruent (t1 t2 : Set Point) (l : Line) :
  symmetrical_triangles t1 t2 l → congruent_triangles t1 t2 := by sorry

end NUMINAMATH_CALUDE_symmetrical_triangles_are_congruent_l1999_199931


namespace NUMINAMATH_CALUDE_function_minimum_implies_inequality_l1999_199905

/-- Given a function f(x) = ax^2 + bx - ln(x) where a > 0 and b ∈ ℝ,
    if f(x) ≥ f(1) for all x > 0, then ln(a) < -2b -/
theorem function_minimum_implies_inequality 
  (a b : ℝ) 
  (ha : a > 0)
  (hf : ∀ x > 0, a * x^2 + b * x - Real.log x ≥ a + b) :
  Real.log a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_implies_inequality_l1999_199905


namespace NUMINAMATH_CALUDE_range_of_x₀_l1999_199920

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point M
def point_M (x₀ : ℝ) : ℝ × ℝ := (x₀, 2 - x₀)

-- Define the angle OMN
def angle_OMN (O M N : ℝ × ℝ) : ℝ := sorry

-- Define the existence of point N on circle O
def exists_N (x₀ : ℝ) : Prop :=
  ∃ N : ℝ × ℝ, circle_O N.1 N.2 ∧ angle_OMN (0, 0) (point_M x₀) N = 30

-- Theorem statement
theorem range_of_x₀ (x₀ : ℝ) :
  exists_N x₀ → 0 ≤ x₀ ∧ x₀ ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x₀_l1999_199920


namespace NUMINAMATH_CALUDE_problem_solution_l1999_199935

-- Define proposition p
def p : Prop := ∀ x : ℝ, 2^x > x^2

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ - 2 > 0

-- Theorem to prove
theorem problem_solution : ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1999_199935


namespace NUMINAMATH_CALUDE_worker_completion_times_l1999_199980

/-- 
Given two positive real numbers p and q, where p < q, and three workers with the following properties:
1. The first worker takes p more days than the second worker to complete a job.
2. The first worker takes q more days than the third worker to complete the job.
3. The first two workers together can complete the job in the same amount of time as the third worker alone.

This theorem proves the time needed for each worker to complete the job individually.
-/
theorem worker_completion_times (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (hpq : p < q) :
  let x := q + Real.sqrt (q * (q - p))
  let y := q - p + Real.sqrt (q * (q - p))
  let z := Real.sqrt (q * (q - p))
  (1 / x + 1 / (x - p) = 1 / (x - q)) ∧
  (x > 0) ∧ (x - p > 0) ∧ (x - q > 0) ∧
  (x = q + Real.sqrt (q * (q - p))) ∧
  (y = q - p + Real.sqrt (q * (q - p))) ∧
  (z = Real.sqrt (q * (q - p))) := by
  sorry

#check worker_completion_times

end NUMINAMATH_CALUDE_worker_completion_times_l1999_199980


namespace NUMINAMATH_CALUDE_population_after_20_years_l1999_199984

/-- The population growth over time with a constant growth rate -/
def population_growth (initial_population : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_population * (1 + growth_rate) ^ years

/-- The theorem stating the population after 20 years with 1% growth rate -/
theorem population_after_20_years :
  population_growth 13 0.01 20 = 13 * (1 + 0.01)^20 := by
  sorry

#eval population_growth 13 0.01 20

end NUMINAMATH_CALUDE_population_after_20_years_l1999_199984


namespace NUMINAMATH_CALUDE_cars_to_trucks_ratio_l1999_199921

theorem cars_to_trucks_ratio (total_vehicles : ℕ) (trucks : ℕ) 
  (h1 : total_vehicles = 60) (h2 : trucks = 20) : 
  (total_vehicles - trucks) / trucks = 2 := by
  sorry

end NUMINAMATH_CALUDE_cars_to_trucks_ratio_l1999_199921


namespace NUMINAMATH_CALUDE_ten_cubes_shaded_l1999_199945

/-- Represents a 4x4x4 cube with a specific shading pattern -/
structure ShadedCube where
  /-- Total number of smaller cubes -/
  total_cubes : Nat
  /-- Number of cubes per edge -/
  edge_length : Nat
  /-- Number of shaded cubes per face -/
  shaded_per_face : Nat
  /-- Condition: total cubes is 64 -/
  total_is_64 : total_cubes = 64
  /-- Condition: edge length is 4 -/
  edge_is_4 : edge_length = 4
  /-- Condition: 5 cubes are shaded per face -/
  five_shaded : shaded_per_face = 5

/-- The number of uniquely shaded cubes in the ShadedCube -/
def uniquely_shaded_cubes (c : ShadedCube) : Nat :=
  8 + 2  -- 8 corner cubes + 2 center cubes on opposite faces

/-- Theorem stating that exactly 10 cubes are uniquely shaded -/
theorem ten_cubes_shaded (c : ShadedCube) :
  uniquely_shaded_cubes c = 10 := by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ten_cubes_shaded_l1999_199945


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l1999_199959

theorem factorization_cubic_minus_linear (a x : ℝ) : 
  a * x^3 - 16 * a * x = a * x * (x + 4) * (x - 4) := by
sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l1999_199959


namespace NUMINAMATH_CALUDE_shared_root_quadratic_equation_l1999_199933

theorem shared_root_quadratic_equation (a b p q : ℝ) (h : a ≠ p ∧ b ≠ q) :
  ∃ (α β γ : ℝ),
    (α^2 + a*α + b = 0 ∧ α^2 + p*α + q = 0) →
    (β^2 + a*β + b = 0 ∧ β ≠ α) →
    (γ^2 + p*γ + q = 0 ∧ γ ≠ α) →
    (x^2 - (-p - (b - q)/(p - a))*x + (b*q*(p - a)^2)/(b - q)^2 = (x - β)*(x - γ)) := by
  sorry

end NUMINAMATH_CALUDE_shared_root_quadratic_equation_l1999_199933


namespace NUMINAMATH_CALUDE_inequality_proof_l1999_199914

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b - c)^2 * (b + c) / a + (c - a)^2 * (c + a) / b + (a - b)^2 * (a + b) / c ≥ 
  2 * (a^2 + b^2 + c^2 - a*b - b*c - c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1999_199914


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptotes_l1999_199967

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the asymptotes of the hyperbola
def asymptote_pos (x y : ℝ) : Prop := y = Real.sqrt 2 * x
def asymptote_neg (x y : ℝ) : Prop := y = -Real.sqrt 2 * x

-- State the theorem
theorem distance_focus_to_asymptotes :
  ∃ (d : ℝ), d = Real.sqrt 6 / 3 ∧
  (∀ (x y : ℝ), asymptote_pos x y →
    d = abs (Real.sqrt 2 * focus.1 - focus.2) / Real.sqrt (1 + 2)) ∧
  (∀ (x y : ℝ), asymptote_neg x y →
    d = abs (-Real.sqrt 2 * focus.1 - focus.2) / Real.sqrt (1 + 2)) :=
sorry

end NUMINAMATH_CALUDE_distance_focus_to_asymptotes_l1999_199967


namespace NUMINAMATH_CALUDE_solution_equals_one_l1999_199956

theorem solution_equals_one (x y : ℝ) 
  (eq1 : 2 * x + y = 4) 
  (eq2 : x + 2 * y = 5) : 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_equals_one_l1999_199956


namespace NUMINAMATH_CALUDE_toris_growth_l1999_199958

theorem toris_growth (original_height current_height : Real) 
  (h1 : original_height = 4.4)
  (h2 : current_height = 7.26) :
  current_height - original_height = 2.86 := by
  sorry

end NUMINAMATH_CALUDE_toris_growth_l1999_199958


namespace NUMINAMATH_CALUDE_ball_count_l1999_199927

theorem ball_count (red green blue total : ℕ) 
  (ratio : red = 15 ∧ green = 13 ∧ blue = 17)
  (red_count : red = 907) :
  total = 2721 :=
by sorry

end NUMINAMATH_CALUDE_ball_count_l1999_199927


namespace NUMINAMATH_CALUDE_log_base_10_of_7_l1999_199952

theorem log_base_10_of_7 (p q : ℝ) 
  (hp : Real.log 5 / Real.log 4 = p) 
  (hq : Real.log 7 / Real.log 5 = q) : 
  Real.log 7 / Real.log 10 = 2 * p * q / (2 * p + 1) := by
  sorry

end NUMINAMATH_CALUDE_log_base_10_of_7_l1999_199952


namespace NUMINAMATH_CALUDE_company_kw_price_l1999_199961

theorem company_kw_price (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (1.2 * a = 0.75 * (a + b)) → (1.2 * a = 2 * b) := by
  sorry

end NUMINAMATH_CALUDE_company_kw_price_l1999_199961


namespace NUMINAMATH_CALUDE_money_split_ratio_l1999_199950

/-- Given two people splitting money in a ratio of 2:3, where the smaller share is $50,
    prove that the total amount shared is $125. -/
theorem money_split_ratio (parker_share richie_share total : ℕ) : 
  parker_share = 50 →
  parker_share + richie_share = total →
  2 * richie_share = 3 * parker_share →
  total = 125 := by
sorry

end NUMINAMATH_CALUDE_money_split_ratio_l1999_199950


namespace NUMINAMATH_CALUDE_function_inequality_l1999_199986

/-- Given functions f and g, prove that if f(x) ≥ g(x) - exp(x) for all x ≥ 1, then a ≥ 1/(2*exp(1)) -/
theorem function_inequality (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → a * x - Real.exp x ≥ Real.log x / x - Real.exp x) →
  a ≥ 1 / (2 * Real.exp 1) := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l1999_199986


namespace NUMINAMATH_CALUDE_randy_initial_money_l1999_199957

theorem randy_initial_money :
  ∀ (initial_money : ℝ),
  let lunch_cost : ℝ := 10
  let remaining_after_lunch : ℝ := initial_money - lunch_cost
  let ice_cream_cost : ℝ := 5
  let ice_cream_fraction : ℝ := 1/4
  ice_cream_cost = ice_cream_fraction * remaining_after_lunch →
  initial_money = 30 :=
λ initial_money =>
  let lunch_cost : ℝ := 10
  let remaining_after_lunch : ℝ := initial_money - lunch_cost
  let ice_cream_cost : ℝ := 5
  let ice_cream_fraction : ℝ := 1/4
  λ h : ice_cream_cost = ice_cream_fraction * remaining_after_lunch =>
  sorry

#check randy_initial_money

end NUMINAMATH_CALUDE_randy_initial_money_l1999_199957


namespace NUMINAMATH_CALUDE_journey_time_ratio_l1999_199951

theorem journey_time_ratio (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) 
  (h1 : distance = 180)
  (h2 : original_time = 6)
  (h3 : new_speed = 20)
  : (distance / new_speed) / original_time = 3 / 2 := by
  sorry

#check journey_time_ratio

end NUMINAMATH_CALUDE_journey_time_ratio_l1999_199951


namespace NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l1999_199924

/-- The coefficient of x^2 in the expansion of (3x^2 + 5x + 2)(4x^2 + 2x + 1) -/
def coefficient_x_squared : ℤ := 21

/-- The first polynomial in the product -/
def p (x : ℚ) : ℚ := 3 * x^2 + 5 * x + 2

/-- The second polynomial in the product -/
def q (x : ℚ) : ℚ := 4 * x^2 + 2 * x + 1

/-- The theorem stating that the coefficient of x^2 in the expansion of (3x^2 + 5x + 2)(4x^2 + 2x + 1) is 21 -/
theorem coefficient_x_squared_expansion :
  ∃ (a b c d e : ℚ), (p * q) = (λ x => a * x^4 + b * x^3 + coefficient_x_squared * x^2 + d * x + e) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l1999_199924


namespace NUMINAMATH_CALUDE_trig_identity_l1999_199997

theorem trig_identity (θ : Real) (h : Real.tan (θ + π/4) = 2) :
  Real.sin θ^2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ^2 = -7/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1999_199997


namespace NUMINAMATH_CALUDE_triangle_reciprocal_sum_l1999_199908

/-- Given a triangle with sides a, b, c, semiperimeter p, inradius r, and circumradius R,
    prove that 1/ab + 1/bc + 1/ac = 1/(2rR) -/
theorem triangle_reciprocal_sum (a b c p r R : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ p > 0 ∧ r > 0 ∧ R > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_inradius : r = (a * b * c) / (4 * p))
  (h_circumradius : R = (a * b * c) / (4 * (p - a) * (p - b) * (p - c))) :
  1 / (a * b) + 1 / (b * c) + 1 / (a * c) = 1 / (2 * r * R) := by
  sorry

end NUMINAMATH_CALUDE_triangle_reciprocal_sum_l1999_199908


namespace NUMINAMATH_CALUDE_second_agency_per_mile_charge_l1999_199970

theorem second_agency_per_mile_charge : 
  let first_agency_daily_charge : ℝ := 20.25
  let first_agency_per_mile_charge : ℝ := 0.14
  let second_agency_daily_charge : ℝ := 18.25
  let miles_at_equal_cost : ℝ := 25
  let second_agency_per_mile_charge : ℝ := 
    (first_agency_daily_charge + first_agency_per_mile_charge * miles_at_equal_cost - second_agency_daily_charge) / miles_at_equal_cost
  second_agency_per_mile_charge = 0.22 := by
sorry

end NUMINAMATH_CALUDE_second_agency_per_mile_charge_l1999_199970


namespace NUMINAMATH_CALUDE_arithmetic_operations_l1999_199910

theorem arithmetic_operations :
  ((-16) + (-29) = -45) ∧
  ((-10) - 7 = -17) ∧
  (5 * (-2) = -10) ∧
  ((-16) / (-2) = 8) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l1999_199910


namespace NUMINAMATH_CALUDE_existence_of_square_between_l1999_199990

theorem existence_of_square_between (a b c d : ℕ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : a * d = b * c) : 
  ∃ m : ℤ, (↑a : ℝ) < m^2 ∧ (m^2 : ℝ) < ↑d :=
by sorry

end NUMINAMATH_CALUDE_existence_of_square_between_l1999_199990


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l1999_199928

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem ellipse_and_line_intersection
  (C : Ellipse)
  (h_point : C.a^2 * (6 / C.a^2 + 1 / C.b^2) = C.a^2) -- Point (√6, 1) lies on the ellipse
  (h_focus : C.a^2 - C.b^2 = 4) -- Left focus is at (-2, 0)
  (m : ℝ)
  (h_distinct : ∃ (A B : Point), A ≠ B ∧
    C.a^2 * ((A.x^2 / C.a^2) + (A.y^2 / C.b^2)) = C.a^2 ∧
    C.a^2 * ((B.x^2 / C.a^2) + (B.y^2 / C.b^2)) = C.a^2 ∧
    A.y = A.x + m ∧ B.y = B.x + m)
  (h_midpoint : ∃ (M : Point), M.x^2 + M.y^2 = 1 ∧
    ∃ (A B : Point), A ≠ B ∧
      C.a^2 * ((A.x^2 / C.a^2) + (A.y^2 / C.b^2)) = C.a^2 ∧
      C.a^2 * ((B.x^2 / C.a^2) + (B.y^2 / C.b^2)) = C.a^2 ∧
      A.y = A.x + m ∧ B.y = B.x + m ∧
      M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2) :
  C.a = 2 * Real.sqrt 2 ∧ C.b = 2 ∧ m = 3 * Real.sqrt 5 / 5 ∨ m = -3 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l1999_199928


namespace NUMINAMATH_CALUDE_orange_juice_fraction_l1999_199996

theorem orange_juice_fraction : 
  let pitcher1_capacity : ℚ := 500
  let pitcher2_capacity : ℚ := 600
  let pitcher1_juice_ratio : ℚ := 1/4
  let pitcher2_juice_ratio : ℚ := 1/3
  let total_juice := pitcher1_capacity * pitcher1_juice_ratio + pitcher2_capacity * pitcher2_juice_ratio
  let total_volume := pitcher1_capacity + pitcher2_capacity
  total_juice / total_volume = 13/44 := by sorry

end NUMINAMATH_CALUDE_orange_juice_fraction_l1999_199996


namespace NUMINAMATH_CALUDE_min_cut_edges_hexagonal_prism_l1999_199922

/-- Represents a hexagonal prism -/
structure HexagonalPrism :=
  (total_edges : ℕ)
  (uncut_edges : ℕ)
  (h_total : total_edges = 18)
  (h_uncut : uncut_edges ≤ total_edges)

/-- The minimum number of edges that need to be cut to unfold a hexagonal prism -/
def min_cut_edges (prism : HexagonalPrism) : ℕ :=
  prism.total_edges - prism.uncut_edges

theorem min_cut_edges_hexagonal_prism (prism : HexagonalPrism) 
  (h_uncut : prism.uncut_edges = 7) : 
  min_cut_edges prism = 11 := by
  sorry

end NUMINAMATH_CALUDE_min_cut_edges_hexagonal_prism_l1999_199922


namespace NUMINAMATH_CALUDE_largest_s_value_l1999_199937

theorem largest_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3) : 
  (59 * (s - 2) * r = 58 * s * (r - 2)) → s ≤ 117 ∧ ∃ r', r' ≥ s ∧ 59 * (117 - 2) * r' = 58 * 117 * (r' - 2) := by
  sorry

#check largest_s_value

end NUMINAMATH_CALUDE_largest_s_value_l1999_199937


namespace NUMINAMATH_CALUDE_smallest_other_integer_l1999_199991

theorem smallest_other_integer (x : ℕ) (m n : ℕ+) :
  m = 36 →
  Nat.gcd m n = x + 5 →
  Nat.lcm m n = x * (x + 5) →
  ∃ n_min : ℕ+, n_min ≤ n ∧ n_min = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l1999_199991


namespace NUMINAMATH_CALUDE_simplify_fraction_l1999_199911

theorem simplify_fraction (a : ℝ) (ha : a ≠ 0) (ha2 : a ≠ 2) :
  (a^2 - 6*a + 9) / (a^2 - 2*a) / (1 - 1/(a - 2)) = (a - 3) / a :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1999_199911


namespace NUMINAMATH_CALUDE_sqrt_expression_value_l1999_199972

theorem sqrt_expression_value : 
  (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt 0.81) / (Real.sqrt 0.49) = 158 / 63 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_value_l1999_199972


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1999_199983

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given non-zero vectors a and b such that ‖a + 3b‖ = ‖a - 3b‖, 
    the angle between them is 90 degrees. -/
theorem angle_between_vectors (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) 
    (h : ‖a + 3 • b‖ = ‖a - 3 • b‖) : 
    Real.arccos (inner a b / (‖a‖ * ‖b‖)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1999_199983


namespace NUMINAMATH_CALUDE_complex_power_sum_l1999_199975

theorem complex_power_sum (z : ℂ) (h : z + (1 / z) = 2 * Real.cos (5 * π / 180)) :
  z^2010 + (1 / z^2010) = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1999_199975


namespace NUMINAMATH_CALUDE_fraction_difference_l1999_199934

theorem fraction_difference (p q : ℝ) (hp : 3 ≤ p ∧ p ≤ 10) (hq : 12 ≤ q ∧ q ≤ 21) :
  (10 / 12 : ℝ) - (3 / 21 : ℝ) = 29 / 42 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l1999_199934


namespace NUMINAMATH_CALUDE_man_swimming_speed_l1999_199988

/-- The speed of a man in still water given his downstream and upstream swimming times and distances -/
theorem man_swimming_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (time : ℝ) 
  (h_downstream : downstream_distance = 36) 
  (h_upstream : upstream_distance = 48) 
  (h_time : time = 6) : 
  ∃ (v_man : ℝ) (v_stream : ℝ), 
    v_man + v_stream = downstream_distance / time ∧ 
    v_man - v_stream = upstream_distance / time ∧ 
    v_man = 7 := by
  sorry

#check man_swimming_speed

end NUMINAMATH_CALUDE_man_swimming_speed_l1999_199988


namespace NUMINAMATH_CALUDE_call_center_theorem_l1999_199901

/-- Represents the ratio of team A's size to team B's size -/
def team_size_ratio : ℚ := 5/8

/-- Represents the fraction of total calls processed by team B -/
def team_b_call_fraction : ℚ := 4/5

/-- Represents the ratio of calls processed by each member of team A to each member of team B -/
def member_call_ratio : ℚ := 2/5

theorem call_center_theorem :
  let total_calls : ℚ := 1
  let team_a_call_fraction : ℚ := total_calls - team_b_call_fraction
  team_size_ratio * (team_a_call_fraction / team_b_call_fraction) = member_call_ratio := by
  sorry

end NUMINAMATH_CALUDE_call_center_theorem_l1999_199901


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l1999_199985

theorem inequality_and_equality_conditions (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : a + b + c = a * b + b * c + c * a) 
  (h5 : a + b + c > 0) : 
  (Real.sqrt (b * c) * (a + 1) ≥ 2) ∧ 
  (Real.sqrt (b * c) * (a + 1) = 2 ↔ 
    (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 0 ∧ b = 2 ∧ c = 2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l1999_199985


namespace NUMINAMATH_CALUDE_target_same_type_as_reference_l1999_199906

/-- Represents a monomial term with variables x and y -/
structure Monomial :=
  (x_exp : ℕ)
  (y_exp : ℕ)

/-- Determines if two monomials are of the same type -/
def same_type (m1 m2 : Monomial) : Prop :=
  m1.x_exp = m2.x_exp ∧ m1.y_exp = m2.y_exp

/-- The reference monomial 3x²y -/
def reference : Monomial :=
  ⟨2, 1⟩

/-- The monomial -yx² -/
def target : Monomial :=
  ⟨2, 1⟩

theorem target_same_type_as_reference : same_type target reference :=
  sorry

end NUMINAMATH_CALUDE_target_same_type_as_reference_l1999_199906


namespace NUMINAMATH_CALUDE_yoki_cans_collected_l1999_199903

/-- Given the conditions of the can collection problem, prove that Yoki picked up 9 cans. -/
theorem yoki_cans_collected (total_cans ladonna_cans prikya_cans avi_cans yoki_cans : ℕ) : 
  total_cans = 85 →
  ladonna_cans = 25 →
  prikya_cans = 2 * ladonna_cans - 3 →
  avi_cans = 8 / 2 →
  yoki_cans = total_cans - (ladonna_cans + prikya_cans + avi_cans) →
  yoki_cans = 9 := by
sorry

end NUMINAMATH_CALUDE_yoki_cans_collected_l1999_199903


namespace NUMINAMATH_CALUDE_common_element_exists_l1999_199998

-- Define a type for the index of sets (1 to 2011)
def SetIndex := Fin 2011

-- Define the property of being a set of consecutive integers
def IsConsecutiveSet (S : Set ℤ) : Prop :=
  ∃ a b : ℤ, a ≤ b ∧ S = Finset.Ico a (b + 1)

-- Define the main theorem
theorem common_element_exists
  (S : SetIndex → Set ℤ)
  (h_nonempty : ∀ i, (S i).Nonempty)
  (h_consecutive : ∀ i, IsConsecutiveSet (S i))
  (h_common : ∀ i j, i ≠ j → (S i ∩ S j).Nonempty) :
  ∃ n : ℤ, n > 0 ∧ ∀ i, n ∈ S i :=
sorry

end NUMINAMATH_CALUDE_common_element_exists_l1999_199998


namespace NUMINAMATH_CALUDE_recycling_problem_l1999_199932

/-- Recycling problem -/
theorem recycling_problem (pounds_per_point : ℕ) (gwen_pounds : ℕ) (total_points : ℕ) 
  (h1 : pounds_per_point = 3)
  (h2 : gwen_pounds = 5)
  (h3 : total_points = 6) :
  gwen_pounds / pounds_per_point + (total_points - gwen_pounds / pounds_per_point) * pounds_per_point = 15 :=
by sorry

end NUMINAMATH_CALUDE_recycling_problem_l1999_199932


namespace NUMINAMATH_CALUDE_expected_mass_with_error_l1999_199941

/-- The expected mass of 100 metal disks with manufacturing errors -/
theorem expected_mass_with_error (
  nominal_diameter : ℝ)
  (perfect_disk_mass : ℝ)
  (radius_std_dev : ℝ)
  (disk_count : ℕ)
  (h1 : nominal_diameter = 1)
  (h2 : perfect_disk_mass = 100)
  (h3 : radius_std_dev = 0.01)
  (h4 : disk_count = 100) :
  ∃ (expected_mass : ℝ), 
    expected_mass = disk_count * perfect_disk_mass * (1 + 4 * (radius_std_dev / nominal_diameter)^2) ∧
    expected_mass = 10004 :=
by sorry

end NUMINAMATH_CALUDE_expected_mass_with_error_l1999_199941


namespace NUMINAMATH_CALUDE_triangle_side_lengths_and_circumradius_l1999_199917

/-- Given a triangle ABC with side lengths a, b, and c satisfying the equation,
    prove that the side lengths are 3, 4, 5 and the circumradius is 2.5 -/
theorem triangle_side_lengths_and_circumradius 
  (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 - 6*a - 8*b - 10*c + 50 = 0) : 
  a = 3 ∧ b = 4 ∧ c = 5 ∧ (2.5 : ℝ) = (1/2 : ℝ) * c := by
  sorry

#check triangle_side_lengths_and_circumradius

end NUMINAMATH_CALUDE_triangle_side_lengths_and_circumradius_l1999_199917


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1999_199979

theorem right_rectangular_prism_volume
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 18)
  (h_front : front_area = 12)
  (h_bottom : bottom_area = 8) :
  ∃ (a b c : ℝ),
    a * b = side_area ∧
    b * c = front_area ∧
    a * c = bottom_area ∧
    a * b * c = 24 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1999_199979


namespace NUMINAMATH_CALUDE_insect_eggs_l1999_199987

def base_6_to_10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6 + c

theorem insect_eggs : base_6_to_10 2 5 3 = 105 := by sorry

end NUMINAMATH_CALUDE_insect_eggs_l1999_199987


namespace NUMINAMATH_CALUDE_inequality_proof_l1999_199969

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : d - a < c - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1999_199969


namespace NUMINAMATH_CALUDE_expression_evaluation_l1999_199992

theorem expression_evaluation (a b : ℚ) (ha : a = 3/4) (hb : b = 4/3) :
  let expr := ((a/b + b/a + 2) * ((a+b)/(2*a) - b/(a+b))) / 
               ((a + 2*b + b^2/a) * (a/(a+b) + b/(a-b)))
  expr = -7/24 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1999_199992


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l1999_199943

theorem max_sum_given_constraints (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) :
  x + y ≤ 6 * Real.sqrt 5 ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 100 ∧ x₀ * y₀ = 40 ∧ x₀ + y₀ = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l1999_199943


namespace NUMINAMATH_CALUDE_all_options_incorrect_l1999_199913

-- Define the types for functions
def Function := ℝ → ℝ

-- Define properties of functions
def Periodic (f : Function) : Prop := 
  ∃ T > 0, ∀ x, f (x + T) = f x

def Monotonic (f : Function) : Prop := 
  ∀ x y, x < y → f x < f y

-- Original proposition
def OriginalProposition : Prop :=
  ∀ f : Function, Periodic f → ¬(Monotonic f)

-- Theorem to prove
theorem all_options_incorrect (original : OriginalProposition) : 
  (¬(∀ f : Function, Monotonic f → ¬(Periodic f))) ∧ 
  (¬(∀ f : Function, Periodic f → Monotonic f)) ∧ 
  (¬(∀ f : Function, Monotonic f → Periodic f)) :=
sorry

end NUMINAMATH_CALUDE_all_options_incorrect_l1999_199913


namespace NUMINAMATH_CALUDE_help_sign_white_area_l1999_199982

theorem help_sign_white_area :
  let sign_width : ℕ := 18
  let sign_height : ℕ := 7
  let h_area : ℕ := 13
  let e_area : ℕ := 11
  let l_area : ℕ := 8
  let p_area : ℕ := 11
  let total_black_area : ℕ := h_area + e_area + l_area + p_area
  let total_sign_area : ℕ := sign_width * sign_height
  total_sign_area - total_black_area = 83 := by
  sorry

end NUMINAMATH_CALUDE_help_sign_white_area_l1999_199982


namespace NUMINAMATH_CALUDE_flatrate_calculation_l1999_199902

/-- Represents the tutoring session details and pricing -/
structure TutoringSession where
  flatRate : ℕ
  perMinuteRate : ℕ
  durationMinutes : ℕ
  totalAmount : ℕ

/-- Theorem stating the flat rate for the given tutoring session -/
theorem flatrate_calculation (session : TutoringSession)
  (h1 : session.perMinuteRate = 7)
  (h2 : session.durationMinutes = 18)
  (h3 : session.totalAmount = 146)
  (h4 : session.totalAmount = session.flatRate + session.perMinuteRate * session.durationMinutes) :
  session.flatRate = 20 := by
  sorry

#check flatrate_calculation

end NUMINAMATH_CALUDE_flatrate_calculation_l1999_199902


namespace NUMINAMATH_CALUDE_group_distribution_theorem_l1999_199942

def number_of_ways (n_men n_women : ℕ) (group_sizes : List ℕ) : ℕ :=
  sorry

theorem group_distribution_theorem :
  let n_men := 4
  let n_women := 5
  let group_sizes := [3, 3, 3]
  number_of_ways n_men n_women group_sizes = 1440 :=
by sorry

end NUMINAMATH_CALUDE_group_distribution_theorem_l1999_199942


namespace NUMINAMATH_CALUDE_c_highest_prob_exactly_two_passing_l1999_199918

-- Define the probabilities of passing each exam for A, B, and C
def probATheory : ℚ := 4/5
def probAPractical : ℚ := 1/2
def probBTheory : ℚ := 3/4
def probBPractical : ℚ := 2/3
def probCTheory : ℚ := 2/3
def probCPractical : ℚ := 5/6

-- Define the probabilities of obtaining the "certificate of passing" for A, B, and C
def probAPassing : ℚ := probATheory * probAPractical
def probBPassing : ℚ := probBTheory * probBPractical
def probCPassing : ℚ := probCTheory * probCPractical

-- Theorem 1: C has the highest probability of obtaining the "certificate of passing"
theorem c_highest_prob : 
  probCPassing > probAPassing ∧ probCPassing > probBPassing :=
sorry

-- Theorem 2: The probability that exactly two out of A, B, and C obtain the "certificate of passing" is 11/30
theorem exactly_two_passing :
  probAPassing * probBPassing * (1 - probCPassing) +
  probAPassing * (1 - probBPassing) * probCPassing +
  (1 - probAPassing) * probBPassing * probCPassing = 11/30 :=
sorry

end NUMINAMATH_CALUDE_c_highest_prob_exactly_two_passing_l1999_199918


namespace NUMINAMATH_CALUDE_johns_phone_bill_l1999_199954

/-- Calculates the total phone bill given the monthly fee, per-minute rate, and minutes used. -/
def total_bill (monthly_fee : ℝ) (per_minute_rate : ℝ) (minutes_used : ℝ) : ℝ :=
  monthly_fee + per_minute_rate * minutes_used

/-- Theorem stating that John's phone bill is $12.02 given the specified conditions. -/
theorem johns_phone_bill :
  let monthly_fee : ℝ := 5
  let per_minute_rate : ℝ := 0.25
  let minutes_used : ℝ := 28.08
  total_bill monthly_fee per_minute_rate minutes_used = 12.02 := by
sorry


end NUMINAMATH_CALUDE_johns_phone_bill_l1999_199954


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l1999_199978

open Matrix

theorem matrix_equation_proof :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  M^3 - 3 • M^2 + 4 • M = !![6, 12; 3, 6] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l1999_199978


namespace NUMINAMATH_CALUDE_range_of_a_l1999_199936

theorem range_of_a (p q : ℝ → Prop) (a : ℝ) : 
  (∀ x, p x ↔ 2*x^2 - 3*x + 1 ≤ 0) →
  (∀ x, q x ↔ (x - a)*(x - a - 1) ≤ 0) →
  (∀ x, p x → (1/2 : ℝ) ≤ x ∧ x ≤ 1) →
  (∀ x, q x → a ≤ x ∧ x ≤ a + 1) →
  (∀ x, ¬(p x) → ¬(q x)) →
  (∃ x, ¬(p x) ∧ q x) →
  0 ≤ a ∧ a ≤ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1999_199936


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1999_199974

/-- Definition of a quadratic equation in x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² - 3x -/
def f (x : ℝ) : ℝ := x^2 - 3*x

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f :=
sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1999_199974


namespace NUMINAMATH_CALUDE_power_of_negative_product_l1999_199919

theorem power_of_negative_product (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_product_l1999_199919


namespace NUMINAMATH_CALUDE_complement_of_intersection_l1999_199999

open Set

def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

theorem complement_of_intersection (U A B : Finset ℕ) 
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 2, 3})
  (hB : B = {2, 3, 4}) :
  (U \ (A ∩ B)) = {1, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l1999_199999


namespace NUMINAMATH_CALUDE_thief_speed_calculation_chase_problem_l1999_199925

/-- Represents the chase scenario between a policeman and a thief -/
structure ChaseScenario where
  initial_distance : ℝ  -- in meters
  policeman_speed : ℝ   -- in km/hr
  thief_distance : ℝ    -- in meters
  thief_speed : ℝ       -- in km/hr

/-- Theorem stating the relationship between the given parameters and the thief's speed -/
theorem thief_speed_calculation (scenario : ChaseScenario) 
  (h1 : scenario.initial_distance = 160)
  (h2 : scenario.policeman_speed = 10)
  (h3 : scenario.thief_distance = 640) :
  scenario.thief_speed = 8 := by
  sorry

/-- Main theorem proving the specific case -/
theorem chase_problem : 
  ∃ (scenario : ChaseScenario), 
    scenario.initial_distance = 160 ∧ 
    scenario.policeman_speed = 10 ∧ 
    scenario.thief_distance = 640 ∧ 
    scenario.thief_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_thief_speed_calculation_chase_problem_l1999_199925


namespace NUMINAMATH_CALUDE_max_prism_plane_intersections_l1999_199909

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  base : Set (ℝ × ℝ)  -- Represents the base of the prism
  height : ℝ           -- Represents the height of the prism

/-- A plane in three-dimensional space. -/
structure Plane where
  normal : ℝ × ℝ × ℝ  -- Normal vector of the plane
  d : ℝ                -- Distance from the origin

/-- Represents the number of edges a plane intersects with a prism. -/
def intersectionCount (prism : Prism) (plane : Plane) : ℕ :=
  sorry  -- Implementation details omitted

/-- Theorem: The maximum number of edges a plane can intersect in a prism is 8. -/
theorem max_prism_plane_intersections (prism : Prism) :
  ∀ plane : Plane, intersectionCount prism plane ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_prism_plane_intersections_l1999_199909


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1999_199900

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the constant term
def constantTerm (a b : ℕ) : ℕ :=
  binomial 8 3 * (5 ^ 5) * (2 ^ 3)

-- Theorem statement
theorem constant_term_expansion :
  constantTerm 5 2 = 1400000 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1999_199900


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1999_199930

theorem right_triangle_third_side (x y z : ℝ) : 
  (x > 0 ∧ y > 0 ∧ z > 0) →  -- positive sides
  (x^2 + y^2 = z^2 ∨ x^2 + z^2 = y^2 ∨ y^2 + z^2 = x^2) →  -- right triangle condition
  (|x - 4| + Real.sqrt (y - 3) = 0) →  -- given equation
  (z = 5 ∨ z = Real.sqrt 7) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1999_199930


namespace NUMINAMATH_CALUDE_union_when_t_is_two_B_subset_A_iff_l1999_199981

-- Define sets A and B
def A (t : ℝ) : Set ℝ := {x | x^2 + (1-t)*x - t ≤ 0}
def B : Set ℝ := {x | |x-2| < 1}

-- Statement 1
theorem union_when_t_is_two :
  A 2 ∪ B = {x | -1 ≤ x ∧ x < 3} := by sorry

-- Statement 2
theorem B_subset_A_iff (t : ℝ) :
  B ⊆ A t ↔ t ≥ 3 := by sorry

end NUMINAMATH_CALUDE_union_when_t_is_two_B_subset_A_iff_l1999_199981


namespace NUMINAMATH_CALUDE_right_triangle_sine_l1999_199960

theorem right_triangle_sine (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a = 9) (h3 : c = 15) :
  a / c = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sine_l1999_199960


namespace NUMINAMATH_CALUDE_classroom_students_count_l1999_199929

theorem classroom_students_count :
  ∃! n : ℕ, n < 60 ∧ n % 8 = 5 ∧ n % 6 = 2 ∧ n = 53 :=
by sorry

end NUMINAMATH_CALUDE_classroom_students_count_l1999_199929


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1999_199904

theorem arithmetic_expression_equality : 5 * 7 + 10 * 4 - 36 / 3 + 6 * 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1999_199904


namespace NUMINAMATH_CALUDE_min_sum_of_product_36_l1999_199964

theorem min_sum_of_product_36 (a b : ℤ) (h : a * b = 36) : 
  ∀ (x y : ℤ), x * y = 36 → a + b ≤ x + y ∧ ∃ (a₀ b₀ : ℤ), a₀ * b₀ = 36 ∧ a₀ + b₀ = -37 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_36_l1999_199964


namespace NUMINAMATH_CALUDE_translate_upward_5_units_l1999_199977

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Translates a linear function vertically by a given amount -/
def translateVertically (f : LinearFunction) (δ : ℝ) : LinearFunction :=
  { m := f.m, b := f.b + δ }

/-- The theorem to prove -/
theorem translate_upward_5_units :
  let f : LinearFunction := { m := 2, b := -3 }
  let g : LinearFunction := translateVertically f 5
  g = { m := 2, b := 2 } := by sorry

end NUMINAMATH_CALUDE_translate_upward_5_units_l1999_199977


namespace NUMINAMATH_CALUDE_simplify_expression_l1999_199995

theorem simplify_expression (x : ℝ) : (2*x)^5 + (3*x)*(x^4) + 2*x^3 = 35*x^5 + 2*x^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1999_199995


namespace NUMINAMATH_CALUDE_hotel_assignment_theorem_l1999_199953

/-- The number of ways to assign 6 friends to 6 rooms with given constraints -/
def assignmentWays : ℕ := sorry

/-- The total number of rooms available -/
def totalRooms : ℕ := 6

/-- The number of friends to be assigned -/
def totalFriends : ℕ := 6

/-- The maximum number of friends allowed per room -/
def maxFriendsPerRoom : ℕ := 2

/-- The maximum number of rooms that can be used -/
def maxRoomsUsed : ℕ := 5

theorem hotel_assignment_theorem :
  assignmentWays = 10440 ∧
  totalRooms = 6 ∧
  totalFriends = 6 ∧
  maxFriendsPerRoom = 2 ∧
  maxRoomsUsed = 5 := by sorry

end NUMINAMATH_CALUDE_hotel_assignment_theorem_l1999_199953


namespace NUMINAMATH_CALUDE_geometric_subsequence_exists_l1999_199938

/-- An arithmetic progression with first term 1 -/
def ArithmeticProgression (d : ℕ) : ℕ → ℕ :=
  fun n => 1 + (n - 1) * d

/-- A geometric progression -/
def GeometricProgression (a : ℕ) : ℕ → ℕ :=
  fun k => a^k

theorem geometric_subsequence_exists :
  ∃ (d a : ℕ), ∃ (start : ℕ),
    (∀ k, k ∈ Finset.range 2015 →
      ArithmeticProgression d (start + k) = GeometricProgression a (k + 1)) :=
sorry

end NUMINAMATH_CALUDE_geometric_subsequence_exists_l1999_199938


namespace NUMINAMATH_CALUDE_class_average_problem_l1999_199971

theorem class_average_problem (x : ℝ) :
  (0.45 * x + 0.50 * 78 + 0.05 * 60 = 84.75) →
  x = 95 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l1999_199971


namespace NUMINAMATH_CALUDE_fraction_equals_49_l1999_199993

theorem fraction_equals_49 : (3100 - 3037)^2 / 81 = 49 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_49_l1999_199993


namespace NUMINAMATH_CALUDE_tangent_line_at_one_condition_holds_iff_l1999_199915

-- Define the function f(x) = 2x³ - 3ax²
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * a * x^2

-- Define the derivative of f
def f_prime (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * a * x

theorem tangent_line_at_one (a : ℝ) (h : a = 2) :
  ∃ m b : ℝ, m = -6 ∧ b = 2 ∧
  ∀ x : ℝ, f a x + (f_prime a 1) * (x - 1) = m * x + b :=
sorry

theorem condition_holds_iff (a : ℝ) :
  (∀ x₁ : ℝ, x₁ ∈ Set.Icc 0 2 →
    ∃ x₂ : ℝ, x₂ ∈ Set.Icc 0 1 ∧ f a x₁ ≥ f_prime a x₂) ↔
  a ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_condition_holds_iff_l1999_199915


namespace NUMINAMATH_CALUDE_set_inclusion_equivalence_l1999_199907

theorem set_inclusion_equivalence (a : ℤ) : 
  let A := {x : ℝ | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}
  let B := {x : ℝ | 3 ≤ x ∧ x ≤ 32}
  (A ⊆ A ∩ B ∧ A.Nonempty) ↔ (6 ≤ a ∧ a ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_set_inclusion_equivalence_l1999_199907
