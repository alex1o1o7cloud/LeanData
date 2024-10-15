import Mathlib

namespace NUMINAMATH_CALUDE_inequality_reversal_l1929_192999

theorem inequality_reversal (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_reversal_l1929_192999


namespace NUMINAMATH_CALUDE_product_percentage_of_x_l1929_192929

theorem product_percentage_of_x (w x y z : ℝ) 
  (h1 : 0.45 * z = 1.2 * y)
  (h2 : y = 0.75 * x)
  (h3 : z = 0.8 * w) :
  w * y = 1.875 * x := by
sorry

end NUMINAMATH_CALUDE_product_percentage_of_x_l1929_192929


namespace NUMINAMATH_CALUDE_water_bottle_cost_l1929_192914

def initial_amount : ℕ := 50
def final_amount : ℕ := 44
def num_baguettes : ℕ := 2
def cost_per_baguette : ℕ := 2
def num_water_bottles : ℕ := 2

theorem water_bottle_cost :
  (initial_amount - final_amount - num_baguettes * cost_per_baguette) / num_water_bottles = 1 :=
by sorry

end NUMINAMATH_CALUDE_water_bottle_cost_l1929_192914


namespace NUMINAMATH_CALUDE_five_digit_four_digit_division_l1929_192931

theorem five_digit_four_digit_division (a b : ℕ) : 
  (a * 11111 = 16 * (b * 1111) + (a * 1111 - 16 * (b * 111) + 2000)) →
  (a ≤ 9) →
  (b ≤ 9) →
  (a * 11111 ≥ b * 1111) →
  (a * 1111 ≥ b * 111) →
  (a = 5 ∧ b = 3) := by
sorry

end NUMINAMATH_CALUDE_five_digit_four_digit_division_l1929_192931


namespace NUMINAMATH_CALUDE_triangle_inequality_l1929_192955

theorem triangle_inequality (x y z : ℝ) (A B C : ℝ) 
  (h_triangle : A + B + C = Real.pi) :
  (x + y + z)^2 ≥ 4 * (y * z * Real.sin A^2 + z * x * Real.sin B^2 + x * y * Real.sin C^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1929_192955


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1929_192962

theorem fraction_sum_equality (n : ℕ) (hn : n > 1) :
  ∃ (i j : ℕ), (1 : ℚ) / n = (1 : ℚ) / i - (1 : ℚ) / (j + 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1929_192962


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_quadratics_l1929_192938

theorem unique_prime_with_prime_quadratics :
  ∃! p : ℕ, Prime p ∧ Prime (4 * p^2 + 1) ∧ Prime (6 * p^2 + 1) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_quadratics_l1929_192938


namespace NUMINAMATH_CALUDE_volunteer_transfer_l1929_192912

theorem volunteer_transfer (initial_group1 initial_group2 : ℕ) 
  (h1 : initial_group1 = 20)
  (h2 : initial_group2 = 26) :
  ∃ x : ℚ, x = 32 / 3 ∧ 
    initial_group1 + x = 2 * (initial_group2 - x) := by
  sorry

end NUMINAMATH_CALUDE_volunteer_transfer_l1929_192912


namespace NUMINAMATH_CALUDE_science_olympiad_participation_l1929_192932

theorem science_olympiad_participation 
  (j s : ℕ) -- j: number of juniors, s: number of seniors
  (h1 : (3 : ℚ) / 7 * j = (2 : ℚ) / 7 * s) -- equal number of participants
  : s = 3 * j := by
sorry

end NUMINAMATH_CALUDE_science_olympiad_participation_l1929_192932


namespace NUMINAMATH_CALUDE_four_numbers_solution_l1929_192951

/-- A sequence of four real numbers satisfying the given conditions -/
structure FourNumbers where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  arithmetic_seq : b - a = c - b
  geometric_seq : c * c = b * d
  sum_first_last : a + d = 16
  sum_middle : b + c = 12

/-- The theorem stating that there are only two possible sets of four numbers satisfying the conditions -/
theorem four_numbers_solution (x : FourNumbers) :
  (x.a = 0 ∧ x.b = 4 ∧ x.c = 8 ∧ x.d = 16) ∨
  (x.a = 15 ∧ x.b = 9 ∧ x.c = 3 ∧ x.d = 1) :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_solution_l1929_192951


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l1929_192983

/-- Given a cube with a sphere inscribed within it, and another cube inscribed within the sphere, 
    this theorem relates the surface areas of the outer and inner cubes. -/
theorem inscribed_cube_surface_area 
  (outer_cube_surface_area : ℝ) 
  (inner_cube_surface_area : ℝ) 
  (h_outer : outer_cube_surface_area = 54) :
  inner_cube_surface_area = 18 :=
sorry

#check inscribed_cube_surface_area

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l1929_192983


namespace NUMINAMATH_CALUDE_cereal_boxes_purchased_l1929_192926

/-- Given the initial price, price reduction, and total payment for monster boxes of cereal,
    prove that the number of boxes purchased is 20. -/
theorem cereal_boxes_purchased
  (initial_price : ℕ)
  (price_reduction : ℕ)
  (total_payment : ℕ)
  (h1 : initial_price = 104)
  (h2 : price_reduction = 24)
  (h3 : total_payment = 1600) :
  total_payment / (initial_price - price_reduction) = 20 :=
by sorry

end NUMINAMATH_CALUDE_cereal_boxes_purchased_l1929_192926


namespace NUMINAMATH_CALUDE_banana_group_size_l1929_192993

theorem banana_group_size (total_bananas : ℕ) (num_groups : ℕ) (h1 : total_bananas = 180) (h2 : num_groups = 10) :
  total_bananas / num_groups = 18 := by
  sorry

end NUMINAMATH_CALUDE_banana_group_size_l1929_192993


namespace NUMINAMATH_CALUDE_base_eight_sum_l1929_192987

theorem base_eight_sum (A B C : ℕ) : 
  A ≠ 0 → B ≠ 0 → C ≠ 0 →
  A ≠ B → B ≠ C → A ≠ C →
  A < 8 → B < 8 → C < 8 →
  (8^2 * A + 8 * B + C) + (8^2 * B + 8 * C + A) + (8^2 * C + 8 * A + B) = 8^3 * A + 8^2 * A + 8 * A →
  B + C = 7 := by
sorry

end NUMINAMATH_CALUDE_base_eight_sum_l1929_192987


namespace NUMINAMATH_CALUDE_f_960_minus_f_640_l1929_192961

/-- Sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Function f(n) defined as sigma(n) / n -/
def f (n : ℕ+) : ℚ := (sigma n : ℚ) / n

/-- Theorem stating that f(960) - f(640) = 5/8 -/
theorem f_960_minus_f_640 : f 960 - f 640 = 5/8 := by sorry

end NUMINAMATH_CALUDE_f_960_minus_f_640_l1929_192961


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l1929_192941

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    a + b + c + d = 18 ∧
    b + c = 7 ∧
    a - d = 3 ∧
    n % 9 = 0

theorem unique_four_digit_number :
  ∃! (n : ℕ), is_valid_number n ∧ n = 6453 := by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l1929_192941


namespace NUMINAMATH_CALUDE_chosen_number_proof_l1929_192978

theorem chosen_number_proof (x : ℝ) : (x / 12) - 240 = 8 ↔ x = 2976 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l1929_192978


namespace NUMINAMATH_CALUDE_hearty_red_packages_l1929_192976

/-- The number of beads in each package -/
def beads_per_package : ℕ := 40

/-- The number of packages of blue beads Hearty bought -/
def blue_packages : ℕ := 3

/-- The total number of beads Hearty has -/
def total_beads : ℕ := 320

/-- The number of packages of red beads Hearty bought -/
def red_packages : ℕ := (total_beads - blue_packages * beads_per_package) / beads_per_package

theorem hearty_red_packages : red_packages = 5 := by
  sorry

end NUMINAMATH_CALUDE_hearty_red_packages_l1929_192976


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1929_192907

/-- Prove that the given equation represents a circle with center (2, -1) passing through (-1, 3) -/
theorem circle_equation_proof (x y : ℝ) : 
  let center : ℝ × ℝ := (2, -1)
  let point : ℝ × ℝ := (-1, 3)
  ((x - center.1)^2 + (y - center.2)^2 = 
   (point.1 - center.1)^2 + (point.2 - center.2)^2) ↔
  ((x - 2)^2 + (y + 1)^2 = 25) :=
by sorry


end NUMINAMATH_CALUDE_circle_equation_proof_l1929_192907


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_1987_l1929_192901

theorem tens_digit_of_13_pow_1987 : ∃ n : ℕ, 13^1987 ≡ 10 * n + 7 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_13_pow_1987_l1929_192901


namespace NUMINAMATH_CALUDE_triangle_proof_l1929_192937

/-- Triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

theorem triangle_proof (t : Triangle) 
  (h1 : 2 * t.c - t.b = 2 * t.a * Real.cos t.B)
  (h2 : 1/2 * t.b * t.c * Real.sin t.A = 3/2 * Real.sqrt 3)
  (h3 : t.c = Real.sqrt 3) :
  t.A = Real.pi / 3 ∧ t.B = Real.pi / 2 := by
  sorry

#check triangle_proof

end NUMINAMATH_CALUDE_triangle_proof_l1929_192937


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1929_192957

theorem sum_of_solutions_quadratic (z : ℂ) : 
  (z^2 = 16*z - 10) → (∃ (z1 z2 : ℂ), z1^2 = 16*z1 - 10 ∧ z2^2 = 16*z2 - 10 ∧ z1 + z2 = 16) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1929_192957


namespace NUMINAMATH_CALUDE_time_period_is_three_years_l1929_192995

/-- Calculates the time period for which a sum is due given the banker's gain, banker's discount, and interest rate. -/
def calculate_time_period (bankers_gain : ℚ) (bankers_discount : ℚ) (interest_rate : ℚ) : ℚ :=
  let true_discount := bankers_discount - bankers_gain
  let ratio := bankers_discount / true_discount
  (ratio - 1) / (interest_rate / 100)

/-- Theorem stating that given the specific values in the problem, the time period is 3 years. -/
theorem time_period_is_three_years :
  let bankers_gain : ℚ := 90
  let bankers_discount : ℚ := 340
  let interest_rate : ℚ := 12
  calculate_time_period bankers_gain bankers_discount interest_rate = 3 := by
  sorry

#eval calculate_time_period 90 340 12

end NUMINAMATH_CALUDE_time_period_is_three_years_l1929_192995


namespace NUMINAMATH_CALUDE_water_in_tank_after_rain_l1929_192965

/-- Calculates the final amount of water in a tank after rainfall, considering inflow, leakage, and evaporation. -/
def final_water_amount (initial_water : ℝ) (inflow_rate : ℝ) (leakage_rate : ℝ) (evaporation_rate : ℝ) (duration : ℝ) : ℝ :=
  initial_water + (inflow_rate - leakage_rate - evaporation_rate) * duration

/-- Theorem stating that the final amount of water in the tank is 226 L -/
theorem water_in_tank_after_rain (initial_water : ℝ) (inflow_rate : ℝ) (leakage_rate : ℝ) (evaporation_rate : ℝ) (duration : ℝ) :
  initial_water = 100 ∧
  inflow_rate = 2 ∧
  leakage_rate = 0.5 ∧
  evaporation_rate = 0.1 ∧
  duration = 90 →
  final_water_amount initial_water inflow_rate leakage_rate evaporation_rate duration = 226 :=
by sorry

end NUMINAMATH_CALUDE_water_in_tank_after_rain_l1929_192965


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1929_192975

theorem perfect_square_condition (X M : ℕ) : 
  (1000 < X ∧ X < 8000) → 
  (M > 1) → 
  (X = M * M^2) → 
  (∃ k : ℕ, X = k^2) → 
  M = 16 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1929_192975


namespace NUMINAMATH_CALUDE_log_equation_implies_relationship_l1929_192974

theorem log_equation_implies_relationship (c d y : ℝ) (hc : c > 0) (hd : d > 0) (hy : y > 0) (hy1 : y ≠ 1) :
  9 * (Real.log y / Real.log c)^2 + 5 * (Real.log y / Real.log d)^2 = 18 * (Real.log y)^2 / (Real.log c * Real.log d) →
  d = c^(1/Real.sqrt 3) ∨ d = c^(Real.sqrt 3) ∨ d = c^(1/Real.sqrt 0.6) ∨ d = c^(Real.sqrt 0.6) := by
sorry

end NUMINAMATH_CALUDE_log_equation_implies_relationship_l1929_192974


namespace NUMINAMATH_CALUDE_high_school_ten_games_l1929_192956

def league_size : ℕ := 10
def non_league_games_per_team : ℕ := 6

def intra_league_games (n : ℕ) : ℕ :=
  n * (n - 1)

def total_games (n : ℕ) (m : ℕ) : ℕ :=
  (intra_league_games n) + (n * m)

theorem high_school_ten_games :
  total_games league_size non_league_games_per_team = 150 := by
  sorry

end NUMINAMATH_CALUDE_high_school_ten_games_l1929_192956


namespace NUMINAMATH_CALUDE_max_distance_between_circle_centers_l1929_192902

theorem max_distance_between_circle_centers 
  (rectangle_width : ℝ) 
  (rectangle_height : ℝ) 
  (circle_diameter : ℝ) 
  (h1 : rectangle_width = 20)
  (h2 : rectangle_height = 15)
  (h3 : circle_diameter = 8)
  (h4 : circle_diameter ≤ rectangle_width ∧ circle_diameter ≤ rectangle_height) :
  let max_distance := Real.sqrt ((rectangle_width - circle_diameter)^2 + (rectangle_height - circle_diameter)^2)
  max_distance = Real.sqrt 193 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_between_circle_centers_l1929_192902


namespace NUMINAMATH_CALUDE_tire_circumference_l1929_192945

/-- Given a tire rotating at 400 revolutions per minute on a car traveling at 120 km/h,
    the circumference of the tire is 5 meters. -/
theorem tire_circumference (rpm : ℝ) (speed : ℝ) (circ : ℝ) : 
  rpm = 400 → speed = 120 → circ * rpm = speed * 1000 / 60 → circ = 5 := by
  sorry

#check tire_circumference

end NUMINAMATH_CALUDE_tire_circumference_l1929_192945


namespace NUMINAMATH_CALUDE_vertical_shift_proof_l1929_192970

/-- Represents a line in slope-intercept form -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- Shifts a line vertically by a given amount -/
def vertical_shift (l : Line) (shift : ℚ) : Line :=
  { slope := l.slope, intercept := l.intercept + shift }

theorem vertical_shift_proof (x : ℚ) :
  let l1 : Line := { slope := -3/4, intercept := 0 }
  let l2 : Line := { slope := -3/4, intercept := -4 }
  vertical_shift l1 (-4) = l2 := by
  sorry

end NUMINAMATH_CALUDE_vertical_shift_proof_l1929_192970


namespace NUMINAMATH_CALUDE_bobby_blocks_l1929_192998

theorem bobby_blocks (initial_blocks final_blocks given_blocks : ℕ) 
  (h1 : final_blocks = initial_blocks + given_blocks)
  (h2 : final_blocks = 8)
  (h3 : given_blocks = 6) : 
  initial_blocks = 2 := by sorry

end NUMINAMATH_CALUDE_bobby_blocks_l1929_192998


namespace NUMINAMATH_CALUDE_meatballs_stolen_l1929_192986

theorem meatballs_stolen (original_total original_beef original_chicken original_pork remaining_beef remaining_chicken remaining_pork : ℕ) :
  original_total = 30 →
  original_beef = 15 →
  original_chicken = 10 →
  original_pork = 5 →
  remaining_beef = 10 →
  remaining_chicken = 10 →
  remaining_pork = 5 →
  original_beef - remaining_beef = 5 :=
by sorry

end NUMINAMATH_CALUDE_meatballs_stolen_l1929_192986


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1929_192920

-- Define a geometric sequence with positive common ratio
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a)
  (h_cond : a 4 * a 8 = 2 * (a 5)^2)
  (h_a2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1929_192920


namespace NUMINAMATH_CALUDE_sqrt_p_div_sqrt_q_l1929_192934

theorem sqrt_p_div_sqrt_q (p q : ℝ) (h : (1/3)^2 + (1/4)^2 = ((25*p)/(61*q)) * ((1/5)^2 + (1/6)^2)) :
  Real.sqrt p / Real.sqrt q = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_p_div_sqrt_q_l1929_192934


namespace NUMINAMATH_CALUDE_vector_magnitude_l1929_192927

def a : ℝ × ℝ := (1, 2)
def b : ℝ → ℝ × ℝ := λ t ↦ (2, t)

theorem vector_magnitude (t : ℝ) (h : a.1 * (b t).1 + a.2 * (b t).2 = 0) :
  Real.sqrt ((b t).1^2 + (b t).2^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1929_192927


namespace NUMINAMATH_CALUDE_triangle_area_range_l1929_192915

/-- Given an obtuse-angled triangle ABC with side c = 2 and angle B = π/3,
    the area S of the triangle satisfies: S ∈ (0, √3/2) ∪ (2√3, +∞) -/
theorem triangle_area_range (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  c = 2 ∧  -- Given condition
  B = π / 3 ∧  -- Given condition
  (A > π / 2 ∨ B > π / 2 ∨ C > π / 2) ∧  -- Obtuse-angled triangle condition
  S = (1 / 2) * a * c * Real.sin B →  -- Area formula
  S ∈ Set.Ioo 0 (Real.sqrt 3 / 2) ∪ Set.Ioi (2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_range_l1929_192915


namespace NUMINAMATH_CALUDE_fraction_inequality_l1929_192972

theorem fraction_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hca : c > a) (hab : a > b) : 
  a / (c - a) > b / (c - b) := by
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1929_192972


namespace NUMINAMATH_CALUDE_min_sum_squares_l1929_192936

theorem min_sum_squares (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (m : ℝ), m = 1/3 ∧ x^2 + y^2 + z^2 ≥ m ∧ ∃ (a b c : ℝ), a + b + c = 1 ∧ a^2 + b^2 + c^2 = m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1929_192936


namespace NUMINAMATH_CALUDE_pumpkin_multiple_l1929_192971

theorem pumpkin_multiple (moonglow sunshine : ℕ) (h1 : moonglow = 14) (h2 : sunshine = 54) :
  ∃ x : ℕ, x * moonglow + 12 = sunshine ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_multiple_l1929_192971


namespace NUMINAMATH_CALUDE_probability_one_from_a_is_11_21_l1929_192960

/-- Represents the number of factories in each area -/
structure FactoryCounts where
  areaA : Nat
  areaB : Nat
  areaC : Nat

/-- Represents the number of factories selected from each area -/
structure SelectedCounts where
  areaA : Nat
  areaB : Nat
  areaC : Nat

/-- Calculates the probability of selecting at least one factory from area A
    when choosing 2 out of 7 stratified sampled factories -/
def probabilityAtLeastOneFromA (counts : FactoryCounts) (selected : SelectedCounts) : Rat :=
  sorry

/-- The main theorem stating the probability is 11/21 given the specific conditions -/
theorem probability_one_from_a_is_11_21 :
  let counts : FactoryCounts := ⟨18, 27, 18⟩
  let selected : SelectedCounts := ⟨2, 3, 2⟩
  probabilityAtLeastOneFromA counts selected = 11 / 21 := by sorry

end NUMINAMATH_CALUDE_probability_one_from_a_is_11_21_l1929_192960


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1929_192904

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence where a_4 = 4, a_3 * a_5 = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) (h_a4 : a 4 = 4) : a 3 * a 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1929_192904


namespace NUMINAMATH_CALUDE_rachel_milk_consumption_l1929_192906

/-- The amount of milk Rachel drinks given the initial amount and fractions poured and drunk -/
theorem rachel_milk_consumption (initial_milk : ℚ) 
  (h1 : initial_milk = 3 / 7)
  (poured_fraction : ℚ) 
  (h2 : poured_fraction = 1 / 2)
  (drunk_fraction : ℚ)
  (h3 : drunk_fraction = 3 / 4) : 
  drunk_fraction * (poured_fraction * initial_milk) = 9 / 56 := by
  sorry

#check rachel_milk_consumption

end NUMINAMATH_CALUDE_rachel_milk_consumption_l1929_192906


namespace NUMINAMATH_CALUDE_max_cars_with_ac_no_stripes_l1929_192946

theorem max_cars_with_ac_no_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (cars_with_stripes : ℕ) 
  (h1 : total_cars = 100)
  (h2 : cars_without_ac = 49)
  (h3 : cars_with_stripes ≥ 51) :
  ∃ (max_cars : ℕ), 
    max_cars ≤ total_cars - cars_without_ac ∧
    max_cars ≤ total_cars - cars_with_stripes ∧
    ∀ (n : ℕ), n ≤ total_cars - cars_without_ac ∧ 
               n ≤ total_cars - cars_with_stripes → 
               n ≤ max_cars ∧
    max_cars = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_cars_with_ac_no_stripes_l1929_192946


namespace NUMINAMATH_CALUDE_vasya_multiplication_error_l1929_192933

-- Define a structure for a two-digit number
structure TwoDigitNumber where
  tens : Fin 10
  ones : Fin 10
  different : tens ≠ ones

-- Define a structure for the result DDEE
structure ResultDDEE where
  d : Fin 10
  e : Fin 10
  different : d ≠ e

-- Define the main theorem
theorem vasya_multiplication_error 
  (ab vg : TwoDigitNumber) 
  (result : ResultDDEE) 
  (h1 : ab.tens ≠ vg.tens)
  (h2 : ab.tens ≠ vg.ones)
  (h3 : ab.ones ≠ vg.tens)
  (h4 : ab.ones ≠ vg.ones)
  (h5 : (ab.tens * 10 + ab.ones) * (vg.tens * 10 + vg.ones) = result.d * 1000 + result.d * 100 + result.e * 10 + result.e) :
  False :=
sorry

end NUMINAMATH_CALUDE_vasya_multiplication_error_l1929_192933


namespace NUMINAMATH_CALUDE_optimal_garden_length_l1929_192973

/-- Represents the length of the side perpendicular to the greenhouse -/
def x : ℝ := sorry

/-- The total amount of fencing available -/
def total_fence : ℝ := 280

/-- The maximum allowed length of the side parallel to the greenhouse -/
def max_parallel_length : ℝ := 300

/-- The length of the side parallel to the greenhouse -/
def parallel_length (x : ℝ) : ℝ := total_fence - 2 * x

/-- The area of the garden as a function of x -/
def garden_area (x : ℝ) : ℝ := x * (parallel_length x)

/-- Theorem stating that the optimal length of the side parallel to the greenhouse is 140 feet -/
theorem optimal_garden_length :
  ∃ (x : ℝ), 
    x > 0 ∧ 
    parallel_length x ≤ max_parallel_length ∧ 
    parallel_length x = 140 ∧ 
    ∀ (y : ℝ), y > 0 ∧ parallel_length y ≤ max_parallel_length → 
      garden_area x ≥ garden_area y :=
by sorry

end NUMINAMATH_CALUDE_optimal_garden_length_l1929_192973


namespace NUMINAMATH_CALUDE_min_value_of_f_l1929_192940

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Theorem statement
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = -3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1929_192940


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1929_192997

theorem sum_of_numbers : ∀ (a b : ℤ), 
  a = 9 → 
  b = -a + 2 → 
  a + b = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1929_192997


namespace NUMINAMATH_CALUDE_x_equals_four_l1929_192989

theorem x_equals_four : ∃! x : ℤ, 2^4 + x = 3^3 - 7 :=
by
  sorry

end NUMINAMATH_CALUDE_x_equals_four_l1929_192989


namespace NUMINAMATH_CALUDE_number_of_cans_l1929_192928

/-- Proves the number of cans given space requirements before and after compaction --/
theorem number_of_cans 
  (space_before : ℝ) 
  (compaction_ratio : ℝ) 
  (total_space_after : ℝ) 
  (h1 : space_before = 30) 
  (h2 : compaction_ratio = 0.2) 
  (h3 : total_space_after = 360) : 
  ℕ :=
by
  sorry

#check number_of_cans

end NUMINAMATH_CALUDE_number_of_cans_l1929_192928


namespace NUMINAMATH_CALUDE_team_a_more_uniform_l1929_192992

/-- Represents a dance team -/
structure DanceTeam where
  name : String
  mean_height : ℝ
  height_variance : ℝ

/-- Define the concept of height uniformity -/
def more_uniform_heights (team1 team2 : DanceTeam) : Prop :=
  team1.height_variance < team2.height_variance

theorem team_a_more_uniform : 
  ∀ (team_a team_b : DanceTeam),
    team_a.name = "A" →
    team_b.name = "B" →
    team_a.mean_height = 1.65 →
    team_b.mean_height = 1.65 →
    team_a.height_variance = 1.5 →
    team_b.height_variance = 2.4 →
    more_uniform_heights team_a team_b :=
by sorry

end NUMINAMATH_CALUDE_team_a_more_uniform_l1929_192992


namespace NUMINAMATH_CALUDE_triangle_area_l1929_192981

theorem triangle_area (base height : ℝ) (h1 : base = 4) (h2 : height = 6) :
  (base * height) / 2 = 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1929_192981


namespace NUMINAMATH_CALUDE_complement_of_A_l1929_192991

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | |x - 1| > 1}

-- State the theorem
theorem complement_of_A : 
  Set.compl A = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1929_192991


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l1929_192954

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + |2*x - a|

theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x < 2} = {x : ℝ | 1/4 < x ∧ x < 5/4} := by sorry

theorem range_of_a :
  (∀ x : ℝ, f a x ≥ 3*a + 2) ↔ -3/2 ≤ a ∧ a ≤ -1/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l1929_192954


namespace NUMINAMATH_CALUDE_second_player_wins_l1929_192967

/-- Represents a point on an infinite grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents the game state --/
structure GameState where
  marked_points : List GridPoint
  current_player : Bool  -- true for first player, false for second player

/-- Checks if a set of points forms a convex polygon --/
def is_convex (points : List GridPoint) : Prop :=
  sorry  -- Implementation details omitted

/-- Checks if a move is valid given the current game state --/
def is_valid_move (state : GameState) (new_point : GridPoint) : Prop :=
  is_convex (new_point :: state.marked_points)

/-- Represents a game strategy --/
def Strategy := GameState → Option GridPoint

/-- Checks if a strategy is winning for the current player --/
def is_winning_strategy (strategy : Strategy) : Prop :=
  sorry  -- Implementation details omitted

/-- The main theorem stating that the second player has a winning strategy --/
theorem second_player_wins :
  ∃ (strategy : Strategy), is_winning_strategy strategy ∧
    ∀ (initial_state : GameState),
      initial_state.current_player = false →
      is_winning_strategy (λ state => strategy state) :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l1929_192967


namespace NUMINAMATH_CALUDE_regular_hexagon_perimeter_l1929_192994

/-- The perimeter of a regular hexagon with side length 5 meters is 30 meters. -/
theorem regular_hexagon_perimeter : 
  ∀ (side_length : ℝ), 
  side_length = 5 → 
  (6 : ℝ) * side_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_perimeter_l1929_192994


namespace NUMINAMATH_CALUDE_find_unknown_number_l1929_192952

theorem find_unknown_number (known_numbers : List ℕ) (average : ℕ) : 
  known_numbers = [55, 48, 507, 2, 42] → 
  average = 223 → 
  ∃ x : ℕ, (List.sum known_numbers + x) / 6 = average ∧ x = 684 :=
by sorry

end NUMINAMATH_CALUDE_find_unknown_number_l1929_192952


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1929_192916

theorem quadratic_function_properties (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^2 + b * x + c
  (f (-2) = 0) → (f 3 = 0) → (f (-b / (2 * a)) > 0) →
  (a < 0) ∧ 
  ({x : ℝ | a * x + c > 0} = {x : ℝ | x > 6}) ∧
  (a + b + c > 0) ∧
  ({x : ℝ | c * x^2 - b * x + a < 0} = {x : ℝ | -1/3 < x ∧ x < 1/2}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1929_192916


namespace NUMINAMATH_CALUDE_ap_contains_sixth_power_l1929_192988

/-- An arithmetic progression containing squares and cubes contains a sixth power -/
theorem ap_contains_sixth_power (a h : ℕ) (p q : ℕ) : 
  0 < a → 0 < h → p ≠ q → p > 0 → q > 0 →
  (∃ k : ℕ, a + k * h = p^2) → 
  (∃ m : ℕ, a + m * h = q^3) → 
  (∃ n x : ℕ, a + n * h = x^6) :=
sorry

end NUMINAMATH_CALUDE_ap_contains_sixth_power_l1929_192988


namespace NUMINAMATH_CALUDE_negation_of_positive_square_l1929_192913

theorem negation_of_positive_square (a : ℝ) :
  ¬(a > 0 → a^2 > 0) ↔ (a ≤ 0 → a^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_positive_square_l1929_192913


namespace NUMINAMATH_CALUDE_exterior_angle_regular_nonagon_exterior_angle_regular_nonagon_proof_l1929_192918

/-- The measure of an exterior angle in a regular nonagon is 40 degrees. -/
theorem exterior_angle_regular_nonagon : ℝ :=
  40

/-- A regular nonagon has 9 sides. -/
def regular_nonagon_sides : ℕ := 9

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

/-- An exterior angle and its corresponding interior angle sum to 180 degrees. -/
axiom exterior_interior_sum : ℝ → ℝ → Prop

/-- The measure of an exterior angle in a regular nonagon is 40 degrees. -/
theorem exterior_angle_regular_nonagon_proof :
  exterior_angle_regular_nonagon =
    180 - (sum_interior_angles regular_nonagon_sides / regular_nonagon_sides) :=
by
  sorry

#check exterior_angle_regular_nonagon_proof

end NUMINAMATH_CALUDE_exterior_angle_regular_nonagon_exterior_angle_regular_nonagon_proof_l1929_192918


namespace NUMINAMATH_CALUDE_donut_sharing_l1929_192953

def total_donuts (delta_donuts : ℕ) (gamma_donuts : ℕ) (beta_multiplier : ℕ) : ℕ :=
  delta_donuts + gamma_donuts + (beta_multiplier * gamma_donuts)

theorem donut_sharing :
  let delta_donuts : ℕ := 8
  let gamma_donuts : ℕ := 8
  let beta_multiplier : ℕ := 3
  total_donuts delta_donuts gamma_donuts beta_multiplier = 40 := by
  sorry

end NUMINAMATH_CALUDE_donut_sharing_l1929_192953


namespace NUMINAMATH_CALUDE_angle_b_in_special_triangle_l1929_192950

/-- In a triangle ABC, if angle A is 80° and angle B equals angle C, then angle B is 50°. -/
theorem angle_b_in_special_triangle (A B C : Real) (h1 : A = 80)
  (h2 : B = C) (h3 : A + B + C = 180) : B = 50 := by
  sorry

end NUMINAMATH_CALUDE_angle_b_in_special_triangle_l1929_192950


namespace NUMINAMATH_CALUDE_basketball_score_proof_l1929_192909

theorem basketball_score_proof (joe tim ken : ℕ) 
  (h1 : tim = joe + 20)
  (h2 : tim * 2 = ken)
  (h3 : joe + tim + ken = 100) :
  tim = 30 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l1929_192909


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_l1929_192996

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The GDP value in yuan -/
def gdp : ℝ := 338.8e9

theorem gdp_scientific_notation :
  toScientificNotation gdp = ScientificNotation.mk 3.388 10 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_l1929_192996


namespace NUMINAMATH_CALUDE_range_of_a_l1929_192984

open Real

theorem range_of_a (a : ℝ) (h_a : a > 0) : 
  (∀ x₁ : ℝ, x₁ > 0 → ∀ x₂ : ℝ, 1 ≤ x₂ ∧ x₂ ≤ Real.exp 1 → 
    x₁ + a^2 / x₁ ≥ x₂ - Real.log x₂) → 
  a ≥ Real.sqrt (Real.exp 1 - 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1929_192984


namespace NUMINAMATH_CALUDE_units_digit_of_p_l1929_192959

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Predicate for an integer having a positive units digit -/
def hasPositiveUnitsDigit (n : ℤ) : Prop := unitsDigit n.natAbs ≠ 0

theorem units_digit_of_p (p : ℤ) 
  (hp_pos : hasPositiveUnitsDigit p)
  (hp_cube_square : unitsDigit (p^3).natAbs = unitsDigit (p^2).natAbs)
  (hp_plus_two : unitsDigit ((p + 2).natAbs) = 8) :
  unitsDigit p.natAbs = 6 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_p_l1929_192959


namespace NUMINAMATH_CALUDE_triangle_area_is_nine_l1929_192919

-- Define the slopes and intersection point
def slope1 : ℚ := 1/3
def slope2 : ℚ := 3
def intersection : ℚ × ℚ := (1, 1)

-- Define the lines
def line1 (x : ℚ) : ℚ := slope1 * (x - intersection.1) + intersection.2
def line2 (x : ℚ) : ℚ := slope2 * (x - intersection.1) + intersection.2
def line3 (x y : ℚ) : Prop := x + y = 8

-- Define the triangle area function
def triangle_area (A B C : ℚ × ℚ) : ℚ :=
  (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Theorem statement
theorem triangle_area_is_nine :
  ∃ A B C : ℚ × ℚ,
    A = intersection ∧
    line3 B.1 B.2 ∧
    line3 C.1 C.2 ∧
    B.2 = line1 B.1 ∧
    C.2 = line2 C.1 ∧
    triangle_area A B C = 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_nine_l1929_192919


namespace NUMINAMATH_CALUDE_point_translation_l1929_192968

/-- Given a point B with coordinates (5, -1) that is translated upwards by 2 units
    to obtain point A with coordinates (a+b, a-b), prove that a = 3 and b = 2. -/
theorem point_translation (a b : ℝ) : 
  (5 : ℝ) = a + b ∧ (1 : ℝ) = a - b → a = 3 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_translation_l1929_192968


namespace NUMINAMATH_CALUDE_cube_side_length_is_one_l1929_192980

/-- The side length of a cube -/
def m : ℕ := sorry

/-- The number of blue faces on the unit cubes -/
def blue_faces : ℕ := 2 * m^2

/-- The total number of faces on all unit cubes -/
def total_faces : ℕ := 6 * m^3

/-- The theorem stating that if one-third of the total faces are blue, then m = 1 -/
theorem cube_side_length_is_one : 
  (blue_faces : ℚ) / total_faces = 1 / 3 → m = 1 := by sorry

end NUMINAMATH_CALUDE_cube_side_length_is_one_l1929_192980


namespace NUMINAMATH_CALUDE_min_value_expression_l1929_192930

theorem min_value_expression (x y : ℝ) : (x*y - 2)^2 + (x^2 + y^2)^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1929_192930


namespace NUMINAMATH_CALUDE_initial_student_count_l1929_192935

/-- Given the initial average weight, new average weight after admitting a new student,
    and the weight of the new student, prove that the initial number of students is 19. -/
theorem initial_student_count
  (initial_avg : ℝ)
  (new_avg : ℝ)
  (new_student_weight : ℝ)
  (h1 : initial_avg = 15)
  (h2 : new_avg = 14.8)
  (h3 : new_student_weight = 11) :
  ∃ n : ℕ, n * initial_avg + new_student_weight = (n + 1) * new_avg ∧ n = 19 := by
  sorry

#check initial_student_count

end NUMINAMATH_CALUDE_initial_student_count_l1929_192935


namespace NUMINAMATH_CALUDE_alice_has_ball_after_two_turns_l1929_192977

/-- Probability of Alice tossing the ball to Bob -/
def alice_toss_prob : ℚ := 1 / 3

/-- Probability of Alice keeping the ball -/
def alice_keep_prob : ℚ := 2 / 3

/-- Probability of Bob tossing the ball to Alice -/
def bob_toss_prob : ℚ := 1 / 4

/-- Probability of Bob keeping the ball -/
def bob_keep_prob : ℚ := 3 / 4

/-- Alice starts with the ball -/
def alice_starts : Prop := True

theorem alice_has_ball_after_two_turns :
  alice_starts →
  (alice_toss_prob * bob_toss_prob + alice_keep_prob * alice_keep_prob : ℚ) = 37 / 108 :=
by sorry

end NUMINAMATH_CALUDE_alice_has_ball_after_two_turns_l1929_192977


namespace NUMINAMATH_CALUDE_distance_between_points_l1929_192948

theorem distance_between_points : 
  let pointA : ℝ × ℝ := (1, -3)
  let pointB : ℝ × ℝ := (4, 6)
  Real.sqrt ((pointB.1 - pointA.1)^2 + (pointB.2 - pointA.2)^2) = Real.sqrt 90 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1929_192948


namespace NUMINAMATH_CALUDE_chicken_nuggets_distribution_l1929_192963

theorem chicken_nuggets_distribution (total : ℕ) (alyssa : ℕ) : 
  total = 100 → alyssa + 2 * alyssa + 2 * alyssa = total → alyssa = 20 := by
  sorry

end NUMINAMATH_CALUDE_chicken_nuggets_distribution_l1929_192963


namespace NUMINAMATH_CALUDE_triangle_area_after_10_seconds_l1929_192917

/-- Represents the position of a runner at time t -/
def RunnerPosition (t : ℝ) := ℝ × ℝ

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Represents the position of a runner over time -/
structure Runner where
  initialPos : ℝ × ℝ
  velocity : ℝ

/-- Calculates the position of a runner at time t -/
def runnerPosition (r : Runner) (t : ℝ) : RunnerPosition t := sorry

theorem triangle_area_after_10_seconds
  (a b c : Runner)
  (h1 : triangleArea (runnerPosition a 0) (runnerPosition b 0) (runnerPosition c 0) = 2)
  (h2 : triangleArea (runnerPosition a 5) (runnerPosition b 5) (runnerPosition c 5) = 3) :
  (triangleArea (runnerPosition a 10) (runnerPosition b 10) (runnerPosition c 10) = 4) ∨
  (triangleArea (runnerPosition a 10) (runnerPosition b 10) (runnerPosition c 10) = 8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_after_10_seconds_l1929_192917


namespace NUMINAMATH_CALUDE_rationalize_denominator_cube_root_l1929_192979

theorem rationalize_denominator_cube_root (x : ℝ) (h : x > 0) :
  x / (x^(1/3)) = x^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_cube_root_l1929_192979


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1929_192990

theorem absolute_value_inequality (y : ℝ) : 
  (2 ≤ |y - 5| ∧ |y - 5| ≤ 8) ↔ ((-3 ≤ y ∧ y ≤ 3) ∨ (7 ≤ y ∧ y ≤ 13)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1929_192990


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l1929_192949

/-- The area of a square with perimeter 48 cm is 144 cm² -/
theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 48 → area = (perimeter / 4) ^ 2 → area = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l1929_192949


namespace NUMINAMATH_CALUDE_mark_car_repair_cost_l1929_192900

/-- Calculates the total cost of car repair for Mark -/
theorem mark_car_repair_cost :
  let labor_hours : ℝ := 2
  let labor_rate : ℝ := 75
  let part_cost : ℝ := 150
  let cleaning_hours : ℝ := 1
  let cleaning_rate : ℝ := 60
  let labor_discount : ℝ := 0.1
  let tax_rate : ℝ := 0.08

  let labor_cost := labor_hours * labor_rate
  let discounted_labor := labor_cost * (1 - labor_discount)
  let cleaning_cost := cleaning_hours * cleaning_rate
  let subtotal := discounted_labor + part_cost + cleaning_cost
  let total_cost := subtotal * (1 + tax_rate)

  total_cost = 372.60 := by sorry

end NUMINAMATH_CALUDE_mark_car_repair_cost_l1929_192900


namespace NUMINAMATH_CALUDE_max_g_given_max_f_l1929_192905

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def g (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

theorem max_g_given_max_f (a b c : ℝ) :
  (∀ x ∈ Set.Icc 0 1, |f a b c x| ≤ 1) →
  (∃ a' b' c', ∀ x ∈ Set.Icc 0 1, |g a' b' c' x| ≤ 8 ∧ 
    ∃ x' ∈ Set.Icc 0 1, |g a' b' c' x'| = 8) :=
sorry

end NUMINAMATH_CALUDE_max_g_given_max_f_l1929_192905


namespace NUMINAMATH_CALUDE_volume_ratio_equals_edge_product_ratio_l1929_192964

/-- Represent a tetrahedron with vertex O and edges OA, OB, OC -/
structure Tetrahedron where
  a : ℝ  -- length of OA
  b : ℝ  -- length of OB
  c : ℝ  -- length of OC
  volume : ℝ  -- volume of the tetrahedron

/-- Two tetrahedrons with congruent trihedral angles at O and O' -/
def CongruentTrihedralTetrahedrons (t1 t2 : Tetrahedron) : Prop :=
  -- We don't explicitly define the congruence, as it's given in the problem statement
  True

theorem volume_ratio_equals_edge_product_ratio
  (t1 t2 : Tetrahedron)
  (h : CongruentTrihedralTetrahedrons t1 t2) :
  t2.volume / t1.volume = (t2.a * t2.b * t2.c) / (t1.a * t1.b * t1.c) := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_equals_edge_product_ratio_l1929_192964


namespace NUMINAMATH_CALUDE_diane_needs_38_cents_l1929_192985

/-- The cost of the cookies in cents -/
def cookie_cost : ℕ := 65

/-- The amount Diane has in cents -/
def diane_has : ℕ := 27

/-- The additional amount Diane needs in cents -/
def additional_amount : ℕ := cookie_cost - diane_has

theorem diane_needs_38_cents : additional_amount = 38 := by
  sorry

end NUMINAMATH_CALUDE_diane_needs_38_cents_l1929_192985


namespace NUMINAMATH_CALUDE_absolute_value_of_h_l1929_192922

theorem absolute_value_of_h (h : ℝ) : 
  (∃ x y : ℝ, x^2 - 4*h*x = 8 ∧ y^2 - 4*h*y = 8 ∧ x^2 + y^2 = 80) → 
  |h| = 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_of_h_l1929_192922


namespace NUMINAMATH_CALUDE_pear_peach_weight_equivalence_l1929_192908

/-- If 9 pears weigh the same as 6 peaches, then 36 pears weigh the same as 24 peaches. -/
theorem pear_peach_weight_equivalence :
  ∀ (pear_weight peach_weight : ℝ),
  9 * pear_weight = 6 * peach_weight →
  36 * pear_weight = 24 * peach_weight :=
by
  sorry


end NUMINAMATH_CALUDE_pear_peach_weight_equivalence_l1929_192908


namespace NUMINAMATH_CALUDE_non_intersecting_path_count_l1929_192966

/-- A path on a grid from (0,0) to (n,n) that can only move top or right -/
def GridPath (n : ℕ) := List (Bool)

/-- Two paths are non-intersecting if they don't share any point except (0,0) and (n,n) -/
def NonIntersecting (n : ℕ) (p1 p2 : GridPath n) : Prop := sorry

/-- The number of non-intersecting pairs of paths from (0,0) to (n,n) -/
def NonIntersectingPathCount (n : ℕ) : ℕ := sorry

theorem non_intersecting_path_count (n : ℕ) : 
  NonIntersectingPathCount n = (Nat.choose (2*n-2) (n-1))^2 - (Nat.choose (2*n-2) (n-2))^2 := by sorry

end NUMINAMATH_CALUDE_non_intersecting_path_count_l1929_192966


namespace NUMINAMATH_CALUDE_sequence_properties_l1929_192939

/-- Given a sequence {aₙ} with sum Sₙ satisfying Sₙ = t(Sₙ - aₙ + 1) where t ≠ 0 and t ≠ 1,
    and a sequence {bₙ} defined as bₙ = aₙ² + Sₙ · aₙ which is geometric,
    prove that {aₙ} is geometric and find the general term of {bₙ}. -/
theorem sequence_properties (t : ℝ) (a b : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : t ≠ 0) (h2 : t ≠ 1)
  (h3 : ∀ n, S n = t * (S n - a n + 1))
  (h4 : ∀ n, b n = a n ^ 2 + S n * a n)
  (h5 : ∃ q, ∀ n, b (n + 1) = q * b n) :
  (∀ n, a (n + 1) = t * a n) ∧
  (∀ n, b n = t^(n + 1) * (2 * t + 1)^(n - 1) / 2^(n - 2)) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1929_192939


namespace NUMINAMATH_CALUDE_computer_factory_month_days_l1929_192982

/-- Proves the number of days in a month given computer production rates --/
theorem computer_factory_month_days
  (monthly_production : ℕ)
  (half_hour_production : ℚ)
  (h1 : monthly_production = 3024)
  (h2 : half_hour_production = 225 / 100) :
  (monthly_production : ℚ) / ((half_hour_production * 2 * 24) : ℚ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_computer_factory_month_days_l1929_192982


namespace NUMINAMATH_CALUDE_terms_before_five_l1929_192903

/-- Given an arithmetic sequence starting with 75 and having a common difference of -5,
    this theorem proves that the number of terms that appear before 5 is 14. -/
theorem terms_before_five (a : ℕ → ℤ) :
  a 0 = 75 ∧ 
  (∀ n : ℕ, a (n + 1) - a n = -5) →
  (∃ k : ℕ, a k = 5 ∧ k = 15) ∧ 
  (∀ m : ℕ, m < 15 → a m > 5) :=
by sorry

end NUMINAMATH_CALUDE_terms_before_five_l1929_192903


namespace NUMINAMATH_CALUDE_equation_roots_l1929_192942

theorem equation_roots : 
  ∃ (x₁ x₂ x₃ x₄ : ℂ), 
    (x₁ = -1/12 ∧ x₂ = 1/2 ∧ x₃ = (5 + Complex.I * Real.sqrt 39) / 24 ∧ x₄ = (5 - Complex.I * Real.sqrt 39) / 24) ∧
    (∀ x : ℂ, (12*x - 1)*(6*x - 1)*(4*x - 1)*(3*x - 1) = 5 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l1929_192942


namespace NUMINAMATH_CALUDE_gcd_654321_543210_l1929_192944

theorem gcd_654321_543210 : Nat.gcd 654321 543210 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_654321_543210_l1929_192944


namespace NUMINAMATH_CALUDE_remaining_macaroons_weight_l1929_192943

theorem remaining_macaroons_weight
  (coconut_count : ℕ) (coconut_weight : ℕ) (almond_count : ℕ) (almond_weight : ℕ)
  (coconut_bags : ℕ) (almond_bags : ℕ) :
  coconut_count = 12 →
  coconut_weight = 5 →
  almond_count = 8 →
  almond_weight = 8 →
  coconut_bags = 4 →
  almond_bags = 2 →
  (coconut_count * coconut_weight - (coconut_count / coconut_bags) * coconut_weight) +
  (almond_count * almond_weight - (almond_count / almond_bags) * almond_weight / 2) = 93 :=
by sorry

end NUMINAMATH_CALUDE_remaining_macaroons_weight_l1929_192943


namespace NUMINAMATH_CALUDE_complex_equation_square_sum_l1929_192911

theorem complex_equation_square_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (a - i) * i = b - i → a^2 + b^2 = 2 := by sorry

end NUMINAMATH_CALUDE_complex_equation_square_sum_l1929_192911


namespace NUMINAMATH_CALUDE_modular_congruence_problem_l1929_192910

theorem modular_congruence_problem : ∃ m : ℕ, 
  (215 * 953 + 100) % 50 = m ∧ 0 ≤ m ∧ m < 50 :=
by
  use 45
  sorry

end NUMINAMATH_CALUDE_modular_congruence_problem_l1929_192910


namespace NUMINAMATH_CALUDE_largest_n_divisibility_equality_l1929_192925

/-- Count of integers less than or equal to n divisible by d -/
def count_divisible (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

/-- Count of integers less than or equal to n divisible by either a or b -/
def count_divisible_either (n : ℕ) (a b : ℕ) : ℕ :=
  count_divisible n a + count_divisible n b - count_divisible n (a * b)

theorem largest_n_divisibility_equality : ∀ m : ℕ, m > 65 →
  (count_divisible m 3 ≠ count_divisible_either m 5 7) ∧
  (count_divisible 65 3 = count_divisible_either 65 5 7) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_equality_l1929_192925


namespace NUMINAMATH_CALUDE_lemon_juice_per_lemon_l1929_192921

/-- The amount of lemon juice needed for one dozen cupcakes, in tablespoons -/
def juice_per_dozen : ℚ := 12

/-- The number of dozens of cupcakes to be made -/
def dozens_to_make : ℚ := 3

/-- The number of lemons needed for the total amount of cupcakes -/
def lemons_needed : ℚ := 9

/-- Proves that each lemon provides 4 tablespoons of juice -/
theorem lemon_juice_per_lemon : 
  (juice_per_dozen * dozens_to_make) / lemons_needed = 4 := by
  sorry

end NUMINAMATH_CALUDE_lemon_juice_per_lemon_l1929_192921


namespace NUMINAMATH_CALUDE_concentric_circles_radius_l1929_192924

/-- Given a configuration of two concentric circles and four identical circles
    tangent to each other and the concentric circles, if the radius of the smaller
    concentric circle is 1, then the radius of the larger concentric circle is 3 + 2√2. -/
theorem concentric_circles_radius (r : ℝ) : 
  r > 0 ∧ 
  r^2 - 2*r - 1 = 0 → 
  1 + 2*r = 3 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_radius_l1929_192924


namespace NUMINAMATH_CALUDE_fraction_equality_l1929_192958

theorem fraction_equality (a b c d e f : ℚ) 
  (h1 : a / b = 1 / 3) 
  (h2 : c / d = 1 / 3) 
  (h3 : e / f = 1 / 3) : 
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1929_192958


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_six_l1929_192947

theorem largest_four_digit_divisible_by_six :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 6 = 0 → n ≤ 9960 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_six_l1929_192947


namespace NUMINAMATH_CALUDE_rhombus_area_l1929_192923

/-- The area of a rhombus with side length 13 cm and one diagonal 24 cm is 120 cm² -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ) : 
  side = 13 → diagonal1 = 24 → side ^ 2 = (diagonal1 / 2) ^ 2 + (diagonal2 / 2) ^ 2 → 
  (diagonal1 * diagonal2) / 2 = 120 := by
  sorry

#check rhombus_area

end NUMINAMATH_CALUDE_rhombus_area_l1929_192923


namespace NUMINAMATH_CALUDE_vertex_on_x_axis_l1929_192969

/-- The parabola equation -/
def parabola (x d : ℝ) : ℝ := x^2 - 6*x + d

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 3

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y (d : ℝ) : ℝ := parabola vertex_x d

theorem vertex_on_x_axis (d : ℝ) : vertex_y d = 0 ↔ d = 9 := by sorry

end NUMINAMATH_CALUDE_vertex_on_x_axis_l1929_192969
