import Mathlib

namespace NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l3755_375566

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 := by
  sorry

end NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l3755_375566


namespace NUMINAMATH_CALUDE_function_g_property_l3755_375505

theorem function_g_property (g : ℝ → ℝ) 
  (h1 : ∀ (b c : ℝ), c^2 * g b = b^2 * g c) 
  (h2 : g 3 ≠ 0) : 
  (g 6 - g 4) / g 3 = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_function_g_property_l3755_375505


namespace NUMINAMATH_CALUDE_trivia_team_distribution_l3755_375510

theorem trivia_team_distribution (total : ℕ) (not_picked : ℕ) (groups : ℕ) 
  (h1 : total = 58) 
  (h2 : not_picked = 10) 
  (h3 : groups = 8) :
  (total - not_picked) / groups = 6 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_distribution_l3755_375510


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3755_375587

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, x^2 - 4*x + (k - 1) = 0) ↔ k ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3755_375587


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3755_375517

/-- Given a hyperbola C₁ and a parabola C₂ in the Cartesian coordinate system (xOy):
    C₁: x²/a² - y²/b² = 1 (a > 0, b > 0)
    C₂: x² = 2py (p > 0)
    
    The asymptotes of C₁ intersect with C₂ at points O, A, B.
    The orthocenter of triangle OAB is the focus of C₂.

    This theorem states that the eccentricity of C₁ is 3/2. -/
theorem hyperbola_eccentricity (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) : 
  let C₁ := fun (x y : ℝ) => x^2/a^2 - y^2/b^2 = 1
  let C₂ := fun (x y : ℝ) => x^2 = 2*p*y
  let asymptotes := fun (x y : ℝ) => y = (b/a)*x ∨ y = -(b/a)*x
  let O := (0, 0)
  let A := (2*p*b/a, 2*p*b^2/a^2)
  let B := (-2*p*b/a, 2*p*b^2/a^2)
  let focus := (0, p/2)
  let orthocenter := focus
  let eccentricity := Real.sqrt (1 + b^2/a^2)
  (∀ x y, asymptotes x y → C₂ x y → (x = 0 ∨ x = 2*p*b/a ∨ x = -2*p*b/a)) →
  (orthocenter = focus) →
  eccentricity = 3/2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3755_375517


namespace NUMINAMATH_CALUDE_function_satisfying_cross_ratio_is_linear_l3755_375582

/-- A function satisfying the given cross-ratio condition is linear -/
theorem function_satisfying_cross_ratio_is_linear (f : ℝ → ℝ) :
  (∀ (a b c d : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    (a - b) / (b - c) + (a - d) / (d - c) = 0 →
    f a ≠ f b ∧ f b ≠ f c ∧ f c ≠ f d ∧ f a ≠ f c ∧ f a ≠ f d ∧ f b ≠ f d →
    (f a - f b) / (f b - f c) + (f a - f d) / (f d - f c) = 0) →
  ∃ (k m : ℝ), ∀ x, f x = k * x + m :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_cross_ratio_is_linear_l3755_375582


namespace NUMINAMATH_CALUDE_angle_B_measure_l3755_375568

-- Define the triangles and angles
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define congruence between triangles
def congruent (t1 t2 : Triangle) : Prop :=
  t1.A = t2.A ∧ t1.B = t2.B ∧ t1.C = t2.C

-- Theorem statement
theorem angle_B_measure (ABC DEF : Triangle) :
  congruent ABC DEF →
  ABC.A = 30 →
  DEF.C = 85 →
  ABC.B = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_l3755_375568


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3755_375529

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3755_375529


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l3755_375534

theorem solve_cubic_equation :
  ∃ y : ℝ, (y - 3)^3 = (1/27)⁻¹ ∧ y = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l3755_375534


namespace NUMINAMATH_CALUDE_expression_value_l3755_375591

theorem expression_value : ∀ a b : ℝ, 
  (a - 2)^2 + |b + 3| = 0 → 
  3*a^2*b - (2*a*b^2 - 2*(a*b - 3/2*a^2*b) + a*b) + 3*a*b^2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3755_375591


namespace NUMINAMATH_CALUDE_sin_300_cos_0_l3755_375516

theorem sin_300_cos_0 : Real.sin (300 * π / 180) * Real.cos 0 = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_cos_0_l3755_375516


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l3755_375576

theorem base_10_to_base_7 : 
  ∃ (a b c d : ℕ), 
    875 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
    a = 2 ∧ b = 3 ∧ c = 6 ∧ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l3755_375576


namespace NUMINAMATH_CALUDE_find_taco_order_l3755_375544

/-- Represents the number of tacos and enchiladas in an order -/
structure Order where
  tacos : ℕ
  enchiladas : ℕ

/-- Represents the cost of an order in dollars -/
def cost (order : Order) (taco_price enchilada_price : ℝ) : ℝ :=
  taco_price * order.tacos + enchilada_price * order.enchiladas

theorem find_taco_order : ∃ (my_order : Order) (enchilada_price : ℝ),
  my_order.enchiladas = 3 ∧
  cost my_order 0.9 enchilada_price = 7.8 ∧
  cost (Order.mk 3 5) 0.9 enchilada_price = 12.7 ∧
  my_order.tacos = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_taco_order_l3755_375544


namespace NUMINAMATH_CALUDE_expression_evaluation_l3755_375554

theorem expression_evaluation :
  let a : ℤ := -2
  3 * a * (2 * a^2 - 4 * a + 3) - 2 * a^2 * (3 * a + 4) = -98 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3755_375554


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l3755_375579

theorem cab_driver_average_income (incomes : List ℝ) 
  (h1 : incomes = [300, 150, 750, 200, 600]) : 
  (incomes.sum / incomes.length) = 400 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_average_income_l3755_375579


namespace NUMINAMATH_CALUDE_john_playing_time_l3755_375508

theorem john_playing_time (beats_per_minute : ℕ) (total_days : ℕ) (total_beats : ℕ) :
  beats_per_minute = 200 →
  total_days = 3 →
  total_beats = 72000 →
  (total_beats / beats_per_minute / 60) / total_days = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_john_playing_time_l3755_375508


namespace NUMINAMATH_CALUDE_dimes_in_jar_l3755_375588

/-- The number of dimes in a jar with equal numbers of dimes, quarters, and half-dollars totaling $20.40 -/
def num_dimes : ℕ := 24

/-- The total value of coins in cents -/
def total_value : ℕ := 2040

theorem dimes_in_jar : 
  10 * num_dimes + 25 * num_dimes + 50 * num_dimes = total_value := by
  sorry

end NUMINAMATH_CALUDE_dimes_in_jar_l3755_375588


namespace NUMINAMATH_CALUDE_zero_points_sum_gt_one_l3755_375504

theorem zero_points_sum_gt_one (x₁ x₂ m : ℝ) 
  (h₁ : x₁ < x₂) 
  (h₂ : Real.log x₁ + 1 / (2 * x₁) = m) 
  (h₃ : Real.log x₂ + 1 / (2 * x₂) = m) : 
  x₁ + x₂ > 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_points_sum_gt_one_l3755_375504


namespace NUMINAMATH_CALUDE_minimum_value_implies_b_equals_one_l3755_375514

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + x + b

-- State the theorem
theorem minimum_value_implies_b_equals_one (a : ℝ) :
  (∃ b : ℝ, (f a b 1 = 1) ∧ 
    (∀ x : ℝ, f a b x ≥ 1) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → f a b x < f a b 1 + ε)) →
  (∃ b : ℝ, b = 1 ∧ (f a b 1 = 1) ∧ 
    (∀ x : ℝ, f a b x ≥ 1) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → f a b x < f a b 1 + ε)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_implies_b_equals_one_l3755_375514


namespace NUMINAMATH_CALUDE_rectangle_diagonal_parts_l3755_375537

theorem rectangle_diagonal_parts (m n : ℕ) (hm : m = 1000) (hn : n = 1979) :
  m + n - Nat.gcd m n = 2978 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_parts_l3755_375537


namespace NUMINAMATH_CALUDE_jerrys_age_l3755_375561

theorem jerrys_age (mickey_age jerry_age : ℝ) : 
  mickey_age = 2.5 * jerry_age - 5 →
  mickey_age = 20 →
  jerry_age = 10 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l3755_375561


namespace NUMINAMATH_CALUDE_winter_solstice_shadow_length_l3755_375526

/-- Given an arithmetic sequence of 12 terms, if the sum of the 1st, 4th, and 7th terms is 37.5
    and the 12th term is 4.5, then the 1st term is 15.5. -/
theorem winter_solstice_shadow_length 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 0 + a 3 + a 6 = 37.5) 
  (h_last : a 11 = 4.5) : 
  a 0 = 15.5 := by
sorry

end NUMINAMATH_CALUDE_winter_solstice_shadow_length_l3755_375526


namespace NUMINAMATH_CALUDE_ellipse_dist_to_directrix_l3755_375550

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < b ∧ b < a

/-- A point on an ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The distance from a point to the left focus of an ellipse -/
def distToLeftFocus (E : Ellipse) (P : PointOnEllipse E) : ℝ := sorry

/-- The distance from a point to the right directrix of an ellipse -/
def distToRightDirectrix (E : Ellipse) (P : PointOnEllipse E) : ℝ := sorry

/-- Theorem: For the given ellipse, if a point on the ellipse is at distance 8 from the left focus,
    then its distance to the right directrix is 5/2 -/
theorem ellipse_dist_to_directrix (E : Ellipse) (P : PointOnEllipse E) :
  E.a = 5 ∧ E.b = 3 ∧ distToLeftFocus E P = 8 → distToRightDirectrix E P = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_dist_to_directrix_l3755_375550


namespace NUMINAMATH_CALUDE_group_size_calculation_l3755_375581

theorem group_size_calculation (average_increase : ℝ) (original_weight : ℝ) (new_weight : ℝ) :
  average_increase = 3.5 →
  original_weight = 75 →
  new_weight = 99.5 →
  (new_weight - original_weight) / average_increase = 7 := by
sorry

end NUMINAMATH_CALUDE_group_size_calculation_l3755_375581


namespace NUMINAMATH_CALUDE_jackpot_probability_correct_l3755_375549

/-- The total number of numbers in the lottery -/
def total_numbers : ℕ := 45

/-- The number of numbers to be chosen in each ticket -/
def numbers_per_ticket : ℕ := 6

/-- The number of tickets bought by the player -/
def tickets_bought : ℕ := 100

/-- The probability of hitting the jackpot with the given number of tickets -/
def jackpot_probability : ℚ :=
  tickets_bought / Nat.choose total_numbers numbers_per_ticket

theorem jackpot_probability_correct :
  jackpot_probability = tickets_bought / Nat.choose total_numbers numbers_per_ticket :=
by sorry

end NUMINAMATH_CALUDE_jackpot_probability_correct_l3755_375549


namespace NUMINAMATH_CALUDE_equation_solution_l3755_375522

theorem equation_solution : 
  Real.sqrt (1 + Real.sqrt (2 + Real.sqrt 49)) = (1 + Real.sqrt 49) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3755_375522


namespace NUMINAMATH_CALUDE_units_digit_37_power_37_l3755_375571

theorem units_digit_37_power_37 : 37^37 % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_37_power_37_l3755_375571


namespace NUMINAMATH_CALUDE_elliptical_cone_theorem_l3755_375572

/-- Given a cone with a 30° aperture and an elliptical base, 
    prove that the square of the minor axis of the ellipse 
    is equal to the product of the shortest and longest slant heights of the cone. -/
theorem elliptical_cone_theorem (b : ℝ) (AC BC : ℝ) : 
  b > 0 → AC > 0 → BC > 0 → (2 * b)^2 = AC * BC := by
  sorry

end NUMINAMATH_CALUDE_elliptical_cone_theorem_l3755_375572


namespace NUMINAMATH_CALUDE_distance_after_one_hour_l3755_375533

/-- The distance between two people moving in opposite directions for 1 hour -/
def distance_between (speed1 speed2 : ℝ) : ℝ :=
  speed1 + speed2

theorem distance_after_one_hour :
  let riya_speed : ℝ := 21
  let priya_speed : ℝ := 22
  distance_between riya_speed priya_speed = 43 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_one_hour_l3755_375533


namespace NUMINAMATH_CALUDE_alcohol_concentration_in_mixture_l3755_375551

structure Vessel where
  capacity : ℝ
  alcoholConcentration : ℝ

def totalAlcohol (vessels : List Vessel) : ℝ :=
  vessels.foldl (fun acc v => acc + v.capacity * v.alcoholConcentration) 0

def largeContainerCapacity : ℝ := 25

theorem alcohol_concentration_in_mixture 
  (vessels : List Vessel)
  (h1 : vessels = [
    ⟨2, 0.3⟩, 
    ⟨6, 0.4⟩, 
    ⟨4, 0.25⟩, 
    ⟨3, 0.35⟩, 
    ⟨5, 0.2⟩
  ]) :
  (totalAlcohol vessels) / largeContainerCapacity = 0.242 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_concentration_in_mixture_l3755_375551


namespace NUMINAMATH_CALUDE_simplify_expression_l3755_375565

theorem simplify_expression (a : ℝ) (h : a = Real.sqrt 3 + 1/2) :
  (a - Real.sqrt 3) * (a + Real.sqrt 3) - a * (a - 6) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3755_375565


namespace NUMINAMATH_CALUDE_smallest_divisor_of_repeated_three_digit_number_l3755_375585

theorem smallest_divisor_of_repeated_three_digit_number : ∀ a b c : ℕ,
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
  let abc := 100 * a + 10 * b + c
  let abcabcabc := 1000000 * abc + 1000 * abc + abc
  (101 ∣ abcabcabc) ∧ ∀ d : ℕ, 1 < d → d < 101 → ¬(d ∣ abcabcabc) :=
by sorry

#check smallest_divisor_of_repeated_three_digit_number

end NUMINAMATH_CALUDE_smallest_divisor_of_repeated_three_digit_number_l3755_375585


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l3755_375548

theorem smallest_number_with_remainder_two : ∃ n : ℕ, 
  (n > 1) ∧ 
  (n % 3 = 2) ∧ 
  (n % 4 = 2) ∧ 
  (n % 5 = 2) ∧ 
  (∀ m : ℕ, m > 1 → (m % 3 = 2) → (m % 4 = 2) → (m % 5 = 2) → m ≥ n) ∧
  (n = 62) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l3755_375548


namespace NUMINAMATH_CALUDE_fresh_grape_weight_l3755_375527

/-- Theorem: Weight of fresh grapes given dried grape weight and water content -/
theorem fresh_grape_weight
  (fresh_water_content : ℝ)
  (dried_water_content : ℝ)
  (dried_grape_weight : ℝ)
  (h1 : fresh_water_content = 0.7)
  (h2 : dried_water_content = 0.1)
  (h3 : dried_grape_weight = 33.33333333333333)
  : ∃ (fresh_grape_weight : ℝ),
    fresh_grape_weight * (1 - fresh_water_content) =
    dried_grape_weight * (1 - dried_water_content) ∧
    fresh_grape_weight = 100 := by
  sorry

end NUMINAMATH_CALUDE_fresh_grape_weight_l3755_375527


namespace NUMINAMATH_CALUDE_sweet_cookies_eaten_indeterminate_l3755_375507

def initial_salty_cookies : ℕ := 26
def initial_sweet_cookies : ℕ := 17
def salty_cookies_eaten : ℕ := 9
def salty_cookies_left : ℕ := 17

theorem sweet_cookies_eaten_indeterminate :
  ∀ (sweet_cookies_eaten : ℕ),
    sweet_cookies_eaten ≤ initial_sweet_cookies →
    salty_cookies_left = initial_salty_cookies - salty_cookies_eaten →
    ∃ (sweet_cookies_eaten' : ℕ),
      sweet_cookies_eaten' ≠ sweet_cookies_eaten ∧
      sweet_cookies_eaten' ≤ initial_sweet_cookies :=
by sorry

end NUMINAMATH_CALUDE_sweet_cookies_eaten_indeterminate_l3755_375507


namespace NUMINAMATH_CALUDE_sum_of_coefficients_for_specific_polynomial_l3755_375584

/-- A polynomial with real coefficients -/
def RealPolynomial (p q r s : ℝ) : ℂ → ℂ :=
  fun x => x^4 + p*x^3 + q*x^2 + r*x + s

/-- Theorem: Sum of coefficients for a specific polynomial -/
theorem sum_of_coefficients_for_specific_polynomial
  (p q r s : ℝ) :
  (RealPolynomial p q r s (3*I) = 0) →
  (RealPolynomial p q r s (3+I) = 0) →
  p + q + r + s = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_for_specific_polynomial_l3755_375584


namespace NUMINAMATH_CALUDE_jims_remaining_distance_l3755_375501

theorem jims_remaining_distance 
  (total_distance : ℕ) 
  (driven_distance : ℕ) 
  (b_to_c : ℕ) 
  (c_to_d : ℕ) 
  (d_to_e : ℕ) 
  (h1 : total_distance = 2500) 
  (h2 : driven_distance = 642) 
  (h3 : b_to_c = 400) 
  (h4 : c_to_d = 550) 
  (h5 : d_to_e = 200) : 
  total_distance - driven_distance = b_to_c + c_to_d + d_to_e :=
by sorry

end NUMINAMATH_CALUDE_jims_remaining_distance_l3755_375501


namespace NUMINAMATH_CALUDE_min_value_expression_l3755_375546

theorem min_value_expression (a b c d e f : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
  (h_upper_bound : a ≤ 3 ∧ b ≤ 3 ∧ c ≤ 3 ∧ d ≤ 3 ∧ e ≤ 3 ∧ f ≤ 3)
  (h_sum1 : a + b + c + d = 6)
  (h_sum2 : e + f = 2) :
  (Real.sqrt (a^2 + 4) + Real.sqrt (b^2 + e^2) + Real.sqrt (c^2 + f^2) + Real.sqrt (d^2 + 4))^2 ≥ 72 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3755_375546


namespace NUMINAMATH_CALUDE_train_speed_l3755_375525

-- Define the train length in meters
def train_length : ℝ := 180

-- Define the time to cross in seconds
def crossing_time : ℝ := 12

-- Define the conversion factor from m/s to km/h
def ms_to_kmh : ℝ := 3.6

-- Theorem to prove
theorem train_speed :
  (train_length / crossing_time) * ms_to_kmh = 54 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l3755_375525


namespace NUMINAMATH_CALUDE_bricks_decrease_by_one_l3755_375567

/-- Represents a brick wall with a given number of rows, total bricks, and bricks in the bottom row. -/
structure BrickWall where
  rows : ℕ
  totalBricks : ℕ
  bottomRowBricks : ℕ

/-- Calculates the number of bricks in a given row of the wall. -/
def bricksInRow (wall : BrickWall) (row : ℕ) : ℕ :=
  wall.bottomRowBricks - (row - 1)

/-- Theorem stating that for a specific brick wall, the number of bricks decreases by 1 in each row going up. -/
theorem bricks_decrease_by_one (wall : BrickWall)
    (h1 : wall.rows = 5)
    (h2 : wall.totalBricks = 200)
    (h3 : wall.bottomRowBricks = 38) :
    ∀ row : ℕ, row > 1 → row ≤ wall.rows →
      bricksInRow wall row = bricksInRow wall (row - 1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_bricks_decrease_by_one_l3755_375567


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_equal_diagonal_regular_polygon_l3755_375521

/-- A regular polygon with all diagonals equal -/
structure EqualDiagonalRegularPolygon where
  /-- The number of sides of the polygon -/
  sides : ℕ
  /-- The polygon is regular -/
  regular : True
  /-- All diagonals of the polygon are equal -/
  equal_diagonals : True
  /-- The polygon has at least 3 sides -/
  sides_ge_three : sides ≥ 3

/-- The sum of interior angles of a polygon -/
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: The sum of interior angles of a regular polygon with all diagonals equal
    is either 360° or 540° -/
theorem sum_of_interior_angles_equal_diagonal_regular_polygon
  (p : EqualDiagonalRegularPolygon) :
  sum_of_interior_angles p.sides = 360 ∨ sum_of_interior_angles p.sides = 540 :=
sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_equal_diagonal_regular_polygon_l3755_375521


namespace NUMINAMATH_CALUDE_equation_transformation_l3755_375545

theorem equation_transformation (m : ℝ) : 2 * m - 1 = 3 → 2 * m = 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l3755_375545


namespace NUMINAMATH_CALUDE_petrol_price_equation_l3755_375502

/-- The original price of petrol satisfies the equation relating to a 15% price reduction and additional 7 gallons for $300 -/
theorem petrol_price_equation (P : ℝ) : P > 0 → 300 / (0.85 * P) = 300 / P + 7 := by
  sorry

end NUMINAMATH_CALUDE_petrol_price_equation_l3755_375502


namespace NUMINAMATH_CALUDE_georges_earnings_l3755_375562

-- Define the daily wages and hours worked
def monday_wage : ℝ := 5
def monday_hours : ℝ := 7
def tuesday_wage : ℝ := 6
def tuesday_hours : ℝ := 2
def wednesday_wage : ℝ := 4
def wednesday_hours : ℝ := 5
def saturday_wage : ℝ := 7
def saturday_hours : ℝ := 3

-- Define the tax rate and uniform fee
def tax_rate : ℝ := 0.1
def uniform_fee : ℝ := 15

-- Calculate total earnings before deductions
def total_earnings : ℝ := 
  monday_wage * monday_hours + 
  tuesday_wage * tuesday_hours + 
  wednesday_wage * wednesday_hours + 
  saturday_wage * saturday_hours

-- Calculate earnings after tax deduction
def earnings_after_tax : ℝ := total_earnings * (1 - tax_rate)

-- Calculate final earnings after uniform fee deduction
def final_earnings : ℝ := earnings_after_tax - uniform_fee

-- Theorem statement
theorem georges_earnings : final_earnings = 64.2 := by
  sorry

end NUMINAMATH_CALUDE_georges_earnings_l3755_375562


namespace NUMINAMATH_CALUDE_fathers_age_l3755_375594

theorem fathers_age (man_age father_age : ℝ) : 
  man_age = (2 / 5) * father_age ∧ 
  man_age + 5 = (1 / 2) * (father_age + 5) → 
  father_age = 25 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l3755_375594


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l3755_375513

theorem factor_difference_of_squares (y : ℝ) : 100 - 25 * y^2 = 25 * (2 - y) * (2 + y) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l3755_375513


namespace NUMINAMATH_CALUDE_truck_speed_on_dirt_l3755_375531

/-- Represents the speed of a truck on different road types -/
structure TruckSpeed where
  dirt : ℝ
  paved : ℝ

/-- Represents the travel time and distance for a truck journey -/
structure TruckJourney where
  speed : TruckSpeed
  time_dirt : ℝ
  time_paved : ℝ
  total_distance : ℝ

/-- Calculates the total distance traveled given a TruckJourney -/
def total_distance (j : TruckJourney) : ℝ :=
  j.speed.dirt * j.time_dirt + j.speed.paved * j.time_paved

/-- Theorem stating the speed of the truck on the dirt road -/
theorem truck_speed_on_dirt (j : TruckJourney) 
  (h1 : j.total_distance = 200)
  (h2 : j.time_paved = 2)
  (h3 : j.time_dirt = 3)
  (h4 : j.speed.paved = j.speed.dirt + 20) :
  j.speed.dirt = 32 := by
  sorry

#check truck_speed_on_dirt

end NUMINAMATH_CALUDE_truck_speed_on_dirt_l3755_375531


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_and_even_l3755_375535

def f (x : ℝ) : ℝ := -2 * x^2

theorem f_monotone_decreasing_and_even :
  (∀ x y, x > 0 → y > 0 → x < y → f x > f y) ∧
  (∀ x, x > 0 → f x = f (-x)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_and_even_l3755_375535


namespace NUMINAMATH_CALUDE_peanut_difference_l3755_375577

theorem peanut_difference (jose_peanuts kenya_peanuts : ℕ) 
  (h1 : jose_peanuts = 85)
  (h2 : kenya_peanuts = 133)
  (h3 : kenya_peanuts > jose_peanuts) :
  kenya_peanuts - jose_peanuts = 48 := by
  sorry

end NUMINAMATH_CALUDE_peanut_difference_l3755_375577


namespace NUMINAMATH_CALUDE_root_in_interval_l3755_375555

theorem root_in_interval (a : ℤ) : 
  (∃ x : ℝ, x > a ∧ x < a + 1 ∧ Real.log x + x - 5 = 0) → a = 3 := by
sorry

end NUMINAMATH_CALUDE_root_in_interval_l3755_375555


namespace NUMINAMATH_CALUDE_mathematics_encoding_l3755_375542

def encode (c : Char) : ℕ :=
  match c with
  | 'M' => 22
  | 'A' => 32
  | 'T' => 33
  | 'E' => 11
  | 'I' => 23
  | 'K' => 13
  | _   => 0

def encodeWord (s : String) : List ℕ :=
  s.toList.map encode

theorem mathematics_encoding :
  encodeWord "MATHEMATICS" = [22, 32, 33, 11, 22, 32, 33, 23, 13, 32] :=
by sorry

end NUMINAMATH_CALUDE_mathematics_encoding_l3755_375542


namespace NUMINAMATH_CALUDE_g_4_equals_10_l3755_375518

/-- A function g satisfying xg(y) = yg(x) for all real x and y, and g(12) = 30 -/
def g : ℝ → ℝ :=
  sorry

/-- The property that xg(y) = yg(x) for all real x and y -/
axiom g_property : ∀ x y : ℝ, x * g y = y * g x

/-- The given condition that g(12) = 30 -/
axiom g_12 : g 12 = 30

/-- Theorem stating that g(4) = 10 -/
theorem g_4_equals_10 : g 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_g_4_equals_10_l3755_375518


namespace NUMINAMATH_CALUDE_prime_power_divides_l3755_375590

theorem prime_power_divides (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → p ∣ a^n → p^n ∣ a^n := by sorry

end NUMINAMATH_CALUDE_prime_power_divides_l3755_375590


namespace NUMINAMATH_CALUDE_pipe_filling_time_l3755_375519

/-- Given two pipes A and B that can fill a tank, this theorem proves the time
    it takes for pipe B to fill the tank alone, given the filling times for
    pipe A alone and both pipes together. -/
theorem pipe_filling_time (fill_time_A fill_time_both : ℝ) 
  (h1 : fill_time_A = 30) 
  (h2 : fill_time_both = 18) : 
  (1 / fill_time_A + 1 / (1 / (1 / fill_time_both - 1 / fill_time_A)))⁻¹ = 45 := by
  sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l3755_375519


namespace NUMINAMATH_CALUDE_expression_simplification_l3755_375589

theorem expression_simplification (x : ℝ) (h : x = (1/2)⁻¹) :
  (x^2 - 2*x + 1) / (x^2 - 1) * (1 + 1/x) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3755_375589


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3755_375524

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {2, 3, 4}

-- Define set B
def B : Finset Nat := {4, 5}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3755_375524


namespace NUMINAMATH_CALUDE_geometric_series_equality_l3755_375559

/-- Defines the sum of the first n terms of the geometric series A_n -/
def A (n : ℕ) : ℚ := 704 * (1 - (1/2)^n) / (1 - 1/2)

/-- Defines the sum of the first n terms of the geometric series B_n -/
def B (n : ℕ) : ℚ := 1984 * (1 - (1/(-2))^n) / (1 + 1/2)

/-- Proves that the smallest positive integer n for which A_n = B_n is 5 -/
theorem geometric_series_equality :
  ∀ n : ℕ, n ≥ 1 → (A n = B n ↔ n = 5) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l3755_375559


namespace NUMINAMATH_CALUDE_simplify_expression_l3755_375503

theorem simplify_expression (m : ℝ) (h1 : m ≠ 1) (h2 : m ≠ -2) :
  (m^2 - 4*m + 4) / (m - 1) / ((3 / (m - 1)) - m - 1) = (2 - m) / (2 + m) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3755_375503


namespace NUMINAMATH_CALUDE_min_value_re_z4_over_re_z4_l3755_375500

theorem min_value_re_z4_over_re_z4 (z : ℂ) (h : (z.re : ℝ) ≠ 0) :
  (z^4).re / (z.re^4 : ℝ) ≥ -8 := by sorry

end NUMINAMATH_CALUDE_min_value_re_z4_over_re_z4_l3755_375500


namespace NUMINAMATH_CALUDE_function_zeros_and_monotonicity_l3755_375532

theorem function_zeros_and_monotonicity (a : ℝ) : 
  a ≠ 0 →
  (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2 * a * x^2 - x - 1 = 0) →
  ¬(∀ x y : ℝ, x > 0 → y > 0 → x < y → x^(2-a) > y^(2-a)) →
  1 < a ∧ a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_function_zeros_and_monotonicity_l3755_375532


namespace NUMINAMATH_CALUDE_clown_balloons_l3755_375541

/-- The number of balloons the clown blew up initially -/
def initial_balloons : ℕ := sorry

/-- The number of additional balloons the clown blew up -/
def additional_balloons : ℕ := 13

/-- The total number of balloons the clown has now -/
def total_balloons : ℕ := 60

theorem clown_balloons : initial_balloons = 47 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l3755_375541


namespace NUMINAMATH_CALUDE_triangle_equation_no_real_roots_l3755_375596

theorem triangle_equation_no_real_roots 
  (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  ∀ x : ℝ, a^2 * x^2 - (c^2 - a^2 - b^2) * x + b^2 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_equation_no_real_roots_l3755_375596


namespace NUMINAMATH_CALUDE_spaceship_journey_theorem_l3755_375569

/-- A spaceship's journey to another planet -/
def spaceship_journey (total_journey_time : ℕ) (initial_travel_time : ℕ) (first_break : ℕ) (second_travel_time : ℕ) (second_break : ℕ) (travel_segment : ℕ) (break_duration : ℕ) : Prop :=
  let total_hours : ℕ := total_journey_time * 24
  let initial_breaks : ℕ := first_break + second_break
  let initial_total_time : ℕ := initial_travel_time + second_travel_time + initial_breaks
  let remaining_time : ℕ := total_hours - initial_total_time
  let full_segments : ℕ := remaining_time / (travel_segment + break_duration)
  let total_breaks : ℕ := initial_breaks + full_segments * break_duration
  total_breaks = 8

theorem spaceship_journey_theorem :
  spaceship_journey 3 10 3 10 1 11 1 := by sorry

end NUMINAMATH_CALUDE_spaceship_journey_theorem_l3755_375569


namespace NUMINAMATH_CALUDE_rachel_brownies_l3755_375506

/-- Rachel's brownie problem -/
theorem rachel_brownies (total : ℕ) (brought_to_school : ℕ) (left_at_home : ℕ) : 
  total = 40 → brought_to_school = 16 → left_at_home = total - brought_to_school →
  left_at_home = 24 := by
  sorry

end NUMINAMATH_CALUDE_rachel_brownies_l3755_375506


namespace NUMINAMATH_CALUDE_soda_pizza_ratio_is_one_to_two_l3755_375563

/-- Represents the cost of items and the number of people -/
structure PurchaseInfo where
  num_people : ℕ
  pizza_cost : ℚ
  total_spent : ℚ

/-- Calculates the ratio of soda cost to pizza cost -/
def soda_to_pizza_ratio (info : PurchaseInfo) : ℚ × ℚ :=
  let pizza_total := info.pizza_cost * info.num_people
  let soda_total := info.total_spent - pizza_total
  let soda_cost := soda_total / info.num_people
  (soda_cost, info.pizza_cost)

/-- Theorem stating the ratio of soda cost to pizza cost is 1:2 -/
theorem soda_pizza_ratio_is_one_to_two (info : PurchaseInfo) 
  (h1 : info.num_people = 6)
  (h2 : info.pizza_cost = 1)
  (h3 : info.total_spent = 9) :
  soda_to_pizza_ratio info = (1/2, 1) := by
  sorry

end NUMINAMATH_CALUDE_soda_pizza_ratio_is_one_to_two_l3755_375563


namespace NUMINAMATH_CALUDE_inequality_reversal_l3755_375595

theorem inequality_reversal (a b : ℝ) (h : a > b) : ¬(a / (-2) > b / (-2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_reversal_l3755_375595


namespace NUMINAMATH_CALUDE_other_juice_cost_is_five_l3755_375580

/-- Represents the cost and quantity information for a juice bar order --/
structure JuiceOrder where
  totalSpent : ℕ
  pineappleCost : ℕ
  pineappleSpent : ℕ
  totalPeople : ℕ

/-- Calculates the cost per glass of the other type of juice --/
def otherJuiceCost (order : JuiceOrder) : ℕ :=
  let pineappleGlasses := order.pineappleSpent / order.pineappleCost
  let otherGlasses := order.totalPeople - pineappleGlasses
  let otherSpent := order.totalSpent - order.pineappleSpent
  otherSpent / otherGlasses

/-- Theorem stating that the cost of the other type of juice is $5 per glass --/
theorem other_juice_cost_is_five (order : JuiceOrder) 
  (h1 : order.totalSpent = 94)
  (h2 : order.pineappleCost = 6)
  (h3 : order.pineappleSpent = 54)
  (h4 : order.totalPeople = 17) :
  otherJuiceCost order = 5 := by
  sorry

end NUMINAMATH_CALUDE_other_juice_cost_is_five_l3755_375580


namespace NUMINAMATH_CALUDE_valid_sequences_count_l3755_375512

/-- The number of colors available at each station -/
def num_colors : ℕ := 4

/-- The number of stations (including start and end) -/
def num_stations : ℕ := 4

/-- A function that calculates the number of valid color sequences -/
def count_valid_sequences : ℕ :=
  num_colors * (num_colors - 1)^(num_stations - 1)

/-- Theorem stating that the number of valid color sequences is 108 -/
theorem valid_sequences_count :
  count_valid_sequences = 108 := by sorry

end NUMINAMATH_CALUDE_valid_sequences_count_l3755_375512


namespace NUMINAMATH_CALUDE_games_sale_value_l3755_375538

def initial_cost : ℝ := 200
def value_multiplier : ℝ := 3
def sold_percentage : ℝ := 0.4

theorem games_sale_value :
  let new_value := initial_cost * value_multiplier
  let sold_value := new_value * sold_percentage
  sold_value = 240 := by
  sorry

end NUMINAMATH_CALUDE_games_sale_value_l3755_375538


namespace NUMINAMATH_CALUDE_peter_hunts_triple_mark_l3755_375560

/-- The number of animals hunted by each person in a day --/
structure HuntingData where
  sam : ℕ
  rob : ℕ
  mark : ℕ
  peter : ℕ

/-- The conditions of the hunting problem --/
def huntingProblem (h : HuntingData) : Prop :=
  h.sam = 6 ∧
  h.rob = h.sam / 2 ∧
  h.mark = (h.sam + h.rob) / 3 ∧
  h.sam + h.rob + h.mark + h.peter = 21

/-- The theorem stating that Peter hunts 3 times more animals than Mark --/
theorem peter_hunts_triple_mark (h : HuntingData) 
  (hcond : huntingProblem h) : h.peter = 3 * h.mark := by
  sorry

end NUMINAMATH_CALUDE_peter_hunts_triple_mark_l3755_375560


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3755_375564

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  a.val * 13 = b.val * 7 →
  Nat.gcd a.val b.val = 23 →
  Nat.lcm a.val b.val = 2093 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3755_375564


namespace NUMINAMATH_CALUDE_max_books_borrowed_l3755_375575

theorem max_books_borrowed (total_students : Nat) (no_books : Nat) (one_book : Nat) (two_books : Nat) 
  (avg_books : Nat) (h1 : total_students = 32) (h2 : no_books = 2) (h3 : one_book = 12) (h4 : two_books = 10) 
  (h5 : avg_books = 2) : 
  ∃ (max_books : Nat), max_books = 11 ∧ 
  (∀ (student_books : Nat), student_books ≤ max_books) ∧
  (∃ (rest_books : Nat), 
    rest_books * (total_students - no_books - one_book - two_books) + 
    no_books * 0 + one_book * 1 + two_books * 2 + max_books = 
    total_students * avg_books) :=
by sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l3755_375575


namespace NUMINAMATH_CALUDE_cube_side_ratio_l3755_375547

theorem cube_side_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (6 * a^2) / (6 * b^2) = 4 → a / b = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l3755_375547


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3755_375515

/-- The volume of a cube with surface area 24 cm² is 8 cm³. -/
theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 24 → s^3 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3755_375515


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3755_375536

theorem geometric_sequence_product (a : ℕ → ℝ) (h : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) :
  a 3 * a 7 = 6 → a 2 * a 4 * a 6 * a 8 = 36 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3755_375536


namespace NUMINAMATH_CALUDE_triangle_base_length_l3755_375598

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- Perimeter of the triangle -/
  perimeter : ℝ
  /-- Length of the segment of the tangent to the inscribed circle, drawn parallel to the base and contained between the sides of the triangle -/
  tangent_segment : ℝ

/-- Theorem stating that for a triangle with perimeter 20 cm and an inscribed circle, 
    if the segment of the tangent to the circle drawn parallel to the base and 
    contained between the sides of the triangle is 2.4 cm, 
    then the base of the triangle is either 4 cm or 6 cm -/
theorem triangle_base_length (t : TriangleWithInscribedCircle) 
  (h_perimeter : t.perimeter = 20)
  (h_tangent : t.tangent_segment = 2.4) :
  ∃ (base : ℝ), (base = 4 ∨ base = 6) ∧ 
  (∃ (side1 side2 : ℝ), side1 + side2 + base = t.perimeter) :=
sorry

end NUMINAMATH_CALUDE_triangle_base_length_l3755_375598


namespace NUMINAMATH_CALUDE_water_fraction_after_replacements_l3755_375530

-- Define the radiator capacity
def radiator_capacity : ℚ := 20

-- Define the volume replaced each time
def replacement_volume : ℚ := 5

-- Define the number of replacements
def num_replacements : ℕ := 5

-- Define the fraction of liquid remaining after each replacement
def remaining_fraction : ℚ := (radiator_capacity - replacement_volume) / radiator_capacity

-- Statement of the problem
theorem water_fraction_after_replacements :
  (remaining_fraction ^ num_replacements : ℚ) = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_water_fraction_after_replacements_l3755_375530


namespace NUMINAMATH_CALUDE_greatest_value_of_a_l3755_375520

noncomputable def f (a : ℝ) : ℝ :=
  (5 * Real.sqrt ((2 * a) ^ 2 + 1) - 4 * a ^ 2 - 2 * a) / (Real.sqrt (1 + 4 * a ^ 2) + 5)

theorem greatest_value_of_a :
  ∃ (a_max : ℝ), a_max = Real.sqrt 6 ∧
  f a_max = 1 ∧
  ∀ (a : ℝ), f a = 1 → a ≤ a_max :=
sorry

end NUMINAMATH_CALUDE_greatest_value_of_a_l3755_375520


namespace NUMINAMATH_CALUDE_worker_b_completion_time_worker_b_time_is_9_l3755_375597

/-- Given workers a, b, and c who can complete a task together or individually,
    this theorem proves the time taken by worker b to complete the task alone. -/
theorem worker_b_completion_time
  (total_rate : ℝ)
  (rate_a : ℝ)
  (rate_b : ℝ)
  (rate_c : ℝ)
  (h1 : total_rate = rate_a + rate_b + rate_c)
  (h2 : total_rate = 1 / 4)
  (h3 : rate_a = 1 / 12)
  (h4 : rate_c = 1 / 18) :
  rate_b = 1 / 9 := by
  sorry

/-- The time taken by worker b to complete the task alone -/
def time_b : ℝ := 9

/-- Proves that the time taken by worker b is indeed 9 days -/
theorem worker_b_time_is_9
  (total_rate : ℝ)
  (rate_a : ℝ)
  (rate_b : ℝ)
  (rate_c : ℝ)
  (h1 : total_rate = rate_a + rate_b + rate_c)
  (h2 : total_rate = 1 / 4)
  (h3 : rate_a = 1 / 12)
  (h4 : rate_c = 1 / 18) :
  time_b = 1 / rate_b := by
  sorry

end NUMINAMATH_CALUDE_worker_b_completion_time_worker_b_time_is_9_l3755_375597


namespace NUMINAMATH_CALUDE_new_scheme_fixed_salary_is_1000_l3755_375557

/-- Represents the salesman's compensation scheme -/
structure CompensationScheme where
  fixedSalary : ℕ
  commissionRate : ℚ
  commissionThreshold : ℕ

/-- Calculates the total compensation for a given sales amount and compensation scheme -/
def calculateCompensation (sales : ℕ) (scheme : CompensationScheme) : ℚ :=
  scheme.fixedSalary + scheme.commissionRate * max (sales - scheme.commissionThreshold) 0

/-- Theorem stating that the fixed salary in the new scheme is 1000 -/
theorem new_scheme_fixed_salary_is_1000 (totalSales : ℕ) (oldScheme newScheme : CompensationScheme) :
  totalSales = 12000 →
  oldScheme.fixedSalary = 0 →
  oldScheme.commissionRate = 1/20 →
  oldScheme.commissionThreshold = 0 →
  newScheme.commissionRate = 1/40 →
  newScheme.commissionThreshold = 4000 →
  calculateCompensation totalSales newScheme = calculateCompensation totalSales oldScheme + 600 →
  newScheme.fixedSalary = 1000 := by
  sorry

#eval calculateCompensation 12000 { fixedSalary := 1000, commissionRate := 1/40, commissionThreshold := 4000 }
#eval calculateCompensation 12000 { fixedSalary := 0, commissionRate := 1/20, commissionThreshold := 0 }

end NUMINAMATH_CALUDE_new_scheme_fixed_salary_is_1000_l3755_375557


namespace NUMINAMATH_CALUDE_triangle_DEF_angle_F_l3755_375523

theorem triangle_DEF_angle_F (D E F : Real) : 
  0 < D ∧ 0 < E ∧ 0 < F ∧ D + E + F = Real.pi →
  2 * Real.sin D + 3 * Real.cos E = 3 →
  3 * Real.sin E + 5 * Real.cos D = 4 →
  Real.sin F = 1/4 := by
sorry

end NUMINAMATH_CALUDE_triangle_DEF_angle_F_l3755_375523


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3755_375578

theorem inscribed_circle_radius (s : ℝ) (r : ℝ) (h : s > 0) :
  3 * s = π * r^2 ∧ r = (s * Real.sqrt 3) / 6 →
  r = 6 * Real.sqrt 3 / π :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3755_375578


namespace NUMINAMATH_CALUDE_fishing_ratio_proof_l3755_375592

/-- The ratio of Brian's fishing trips to Chris's fishing trips -/
def fishing_ratio : ℚ := 26/15

theorem fishing_ratio_proof (brian_catch : ℕ) (total_catch : ℕ) (chris_trips : ℕ) :
  brian_catch = 400 →
  total_catch = 13600 →
  chris_trips = 10 →
  (∃ (brian_trips : ℚ),
    brian_trips * brian_catch + chris_trips * (brian_catch * 5/3) = total_catch ∧
    brian_trips / chris_trips = fishing_ratio) :=
by sorry

end NUMINAMATH_CALUDE_fishing_ratio_proof_l3755_375592


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3755_375573

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 0}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3755_375573


namespace NUMINAMATH_CALUDE_distinct_z_values_exist_l3755_375593

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  1000 * d4 + 100 * d3 + 10 * d2 + d1

def z (x : ℕ) : ℕ := Int.natAbs (x - reverse_digits x)

theorem distinct_z_values_exist :
  ∃ (x y : ℕ), is_four_digit x ∧ is_four_digit y ∧ 
  y = reverse_digits x ∧ z x ≠ z y :=
sorry

end NUMINAMATH_CALUDE_distinct_z_values_exist_l3755_375593


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l3755_375540

theorem indefinite_integral_proof (x : ℝ) :
  deriv (fun x => (2 - 3*x) * Real.exp (2*x)) = fun x => (1 - 6*x) * Real.exp (2*x) := by
  sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l3755_375540


namespace NUMINAMATH_CALUDE_two_x_minus_one_gt_zero_is_linear_inequality_l3755_375599

/-- Definition of a linear inequality in one variable -/
def is_linear_inequality_one_var (f : ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x ↔ a * x + b > 0 ∨ a * x + b < 0 ∨ a * x + b = 0

/-- The inequality 2x - 1 > 0 is a linear inequality in one variable -/
theorem two_x_minus_one_gt_zero_is_linear_inequality :
  is_linear_inequality_one_var (λ x : ℝ => 2 * x - 1 > 0) :=
sorry

end NUMINAMATH_CALUDE_two_x_minus_one_gt_zero_is_linear_inequality_l3755_375599


namespace NUMINAMATH_CALUDE_johann_delivery_correct_l3755_375558

/-- The number of pieces Johann needs to deliver -/
def johann_delivery (total friend1 friend2 friend3 friend4 : ℕ) : ℕ :=
  total - (friend1 + friend2 + friend3 + friend4)

/-- Theorem stating that Johann's delivery is correct -/
theorem johann_delivery_correct (total friend1 friend2 friend3 friend4 : ℕ) 
  (h_total : total = 250)
  (h_friend1 : friend1 = 35)
  (h_friend2 : friend2 = 42)
  (h_friend3 : friend3 = 38)
  (h_friend4 : friend4 = 45) :
  johann_delivery total friend1 friend2 friend3 friend4 = 90 := by
  sorry

#eval johann_delivery 250 35 42 38 45

end NUMINAMATH_CALUDE_johann_delivery_correct_l3755_375558


namespace NUMINAMATH_CALUDE_formula_satisfies_table_l3755_375509

def table : List (ℕ × ℕ) := [(1, 1), (2, 8), (3, 27), (4, 64), (5, 125)]

theorem formula_satisfies_table : ∀ (pair : ℕ × ℕ), pair ∈ table → (pair.2 : ℚ) = (pair.1 : ℚ) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_formula_satisfies_table_l3755_375509


namespace NUMINAMATH_CALUDE_max_value_2sin_l3755_375528

theorem max_value_2sin (f : ℝ → ℝ) (h : f = λ x => 2 * Real.sin x) :
  ∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, f x ≤ M := by
sorry

end NUMINAMATH_CALUDE_max_value_2sin_l3755_375528


namespace NUMINAMATH_CALUDE_product_equals_eight_l3755_375511

theorem product_equals_eight :
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_eight_l3755_375511


namespace NUMINAMATH_CALUDE_haley_albums_l3755_375574

theorem haley_albums (total_pics : ℕ) (first_album_pics : ℕ) (pics_per_album : ℕ) 
  (h1 : total_pics = 65)
  (h2 : first_album_pics = 17)
  (h3 : pics_per_album = 8) :
  (total_pics - first_album_pics) / pics_per_album = 6 := by
  sorry

end NUMINAMATH_CALUDE_haley_albums_l3755_375574


namespace NUMINAMATH_CALUDE_circles_tangent_m_values_l3755_375543

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def C₂ (x y m : ℝ) : Prop := x^2 + y^2 - 8*x + 8*y + m = 0

-- Define tangency condition
def are_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y m ∧
  (∀ (x' y' : ℝ), C₁ x' y' ∧ C₂ x' y' m → (x' = x ∧ y' = y))

-- Theorem statement
theorem circles_tangent_m_values :
  ∀ m : ℝ, are_tangent m ↔ (m = -4 ∨ m = 16) :=
sorry

end NUMINAMATH_CALUDE_circles_tangent_m_values_l3755_375543


namespace NUMINAMATH_CALUDE_dryer_price_difference_dryer_costs_less_l3755_375539

/-- Given a washing machine price of $100 and a dryer with an unknown price,
    if there's a 10% discount on the total and the final price is $153,
    then the dryer costs $30 less than the washing machine. -/
theorem dryer_price_difference (dryer_price : ℝ) : 
  (100 + dryer_price) * 0.9 = 153 → dryer_price = 70 :=
by
  sorry

/-- The difference in price between the washing machine and the dryer -/
def price_difference : ℝ := 100 - 70

theorem dryer_costs_less : price_difference = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_dryer_price_difference_dryer_costs_less_l3755_375539


namespace NUMINAMATH_CALUDE_no_valid_operation_l3755_375570

def equation (op : ℝ → ℝ → ℝ) : Prop :=
  op 8 2 * 3 + 7 - (5 - 3) = 16

theorem no_valid_operation : 
  ¬ (equation (·/·) ∨ equation (·*·) ∨ equation (·+·) ∨ equation (·-·)) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_operation_l3755_375570


namespace NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l3755_375586

theorem max_value_of_expression (x : ℝ) :
  (x^4) / (x^8 + 4*x^6 + x^4 + 4*x^2 + 16) ≤ 1/17 :=
by sorry

theorem max_value_achievable :
  ∃ x : ℝ, (x^4) / (x^8 + 4*x^6 + x^4 + 4*x^2 + 16) = 1/17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l3755_375586


namespace NUMINAMATH_CALUDE_airplane_seats_theorem_l3755_375552

/-- Represents the total number of seats on an airplane -/
def total_seats : ℕ := 360

/-- Represents the number of First Class seats -/
def first_class_seats : ℕ := 36

/-- Represents the fraction of total seats in Business Class -/
def business_class_fraction : ℚ := 3/10

/-- Represents the fraction of total seats in Economy -/
def economy_fraction : ℚ := 6/10

theorem airplane_seats_theorem :
  (first_class_seats : ℚ) + 
  (business_class_fraction * total_seats) + 
  (economy_fraction * total_seats) = total_seats := by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_theorem_l3755_375552


namespace NUMINAMATH_CALUDE_gcd_98_75_l3755_375583

theorem gcd_98_75 : Nat.gcd 98 75 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_75_l3755_375583


namespace NUMINAMATH_CALUDE_equation_solution_l3755_375553

theorem equation_solution : 
  ∃ x : ℝ, (2 / x = 3 / (x + 1)) ∧ (x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3755_375553


namespace NUMINAMATH_CALUDE_tenth_replacement_in_january_l3755_375556

/-- Represents months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Calculates the month after a given number of months have passed -/
def monthAfter (start : Month) (months : ℕ) : Month := sorry

/-- The number of months between battery replacements -/
def replacementInterval : ℕ := 4

/-- The ordinal number of the replacement we're interested in -/
def targetReplacement : ℕ := 10

/-- Theorem stating that the 10th replacement will occur in January -/
theorem tenth_replacement_in_january :
  monthAfter Month.January ((targetReplacement - 1) * replacementInterval) = Month.January := by
  sorry

end NUMINAMATH_CALUDE_tenth_replacement_in_january_l3755_375556
