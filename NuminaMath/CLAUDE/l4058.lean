import Mathlib

namespace NUMINAMATH_CALUDE_existence_of_good_subset_l4058_405852

def M : Finset ℕ := Finset.range 20

def is_valid_function (f : Finset ℕ → ℕ) : Prop :=
  ∀ S : Finset ℕ, S ⊆ M → S.card = 9 → f S ∈ M

theorem existence_of_good_subset (f : Finset ℕ → ℕ) (h : is_valid_function f) :
  ∃ T : Finset ℕ, T ⊆ M ∧ T.card = 10 ∧ ∀ k ∈ T, f (T \ {k}) ≠ k := by sorry

end NUMINAMATH_CALUDE_existence_of_good_subset_l4058_405852


namespace NUMINAMATH_CALUDE_houses_with_garage_count_l4058_405886

/-- Represents the number of houses with various features in a development --/
structure Development where
  total : ℕ
  withPool : ℕ
  withBoth : ℕ
  withNeither : ℕ

/-- Calculates the number of houses with a two-car garage --/
def housesWithGarage (d : Development) : ℕ :=
  d.total + d.withBoth - d.withPool - d.withNeither

/-- Theorem stating that in the given development, 75 houses have a two-car garage --/
theorem houses_with_garage_count (d : Development) 
  (h1 : d.total = 85)
  (h2 : d.withPool = 40)
  (h3 : d.withBoth = 35)
  (h4 : d.withNeither = 30) :
  housesWithGarage d = 75 := by
  sorry

#eval housesWithGarage ⟨85, 40, 35, 30⟩

end NUMINAMATH_CALUDE_houses_with_garage_count_l4058_405886


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_achieved_l4058_405806

theorem max_value_inequality (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + 2*c^2 = 1) : 
  a*b*Real.sqrt 3 + 3*b*c ≤ Real.sqrt 7 :=
sorry

theorem max_value_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ a b c : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 
  a^2 + b^2 + 2*c^2 = 1 ∧ 
  Real.sqrt 7 - ε < a*b*Real.sqrt 3 + 3*b*c :=
sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_achieved_l4058_405806


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l4058_405898

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 10) : x^2 + 1/x^2 = 98 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l4058_405898


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l4058_405801

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3| - 5
def g (x : ℝ) : ℝ := |x + 2| - 2

-- Theorem for the solution set of f(x) ≤ 2
theorem solution_set_f (x : ℝ) : f x ≤ 2 ↔ -4 ≤ x ∧ x ≤ 10 := by sorry

-- Theorem for the range of m
theorem range_of_m : 
  ∀ m : ℝ, (∃ x : ℝ, f x - g x ≥ m - 3) ↔ m ≤ 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l4058_405801


namespace NUMINAMATH_CALUDE_equation_solutions_l4058_405881

theorem equation_solutions : 
  {x : ℝ | 3 * x + 6 = |(-10 + 5 * x)|} = {8, (1/2 : ℝ)} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4058_405881


namespace NUMINAMATH_CALUDE_function_inequality_l4058_405802

/-- Given functions f and g, prove that g(x) > f(x) + kx - 1 for all x > 0 and a ∈ (0, e^2/2] -/
theorem function_inequality (k : ℝ) :
  ∀ (x a : ℝ), x > 0 → 0 < a → a ≤ Real.exp 2 / 2 →
  (Real.exp x) / (a * x) > Real.log x - k * x + 1 + k * x - 1 := by
  sorry


end NUMINAMATH_CALUDE_function_inequality_l4058_405802


namespace NUMINAMATH_CALUDE_fifteen_times_number_equals_three_hundred_l4058_405811

theorem fifteen_times_number_equals_three_hundred :
  ∃ x : ℝ, 15 * x = 300 ∧ x = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_fifteen_times_number_equals_three_hundred_l4058_405811


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_45_l4058_405894

theorem four_digit_divisible_by_45 : ∃ (a b : ℕ), 
  a < 10 ∧ b < 10 ∧ 
  (1000 * a + 520 + b) % 45 = 0 ∧
  (∃ (c d : ℕ), c < 10 ∧ d < 10 ∧ c ≠ a ∧ d ≠ b ∧ (1000 * c + 520 + d) % 45 = 0) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_45_l4058_405894


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_l4058_405842

theorem pure_imaginary_complex (a : ℝ) : 
  (a - (10 : ℂ) / (3 - I)).im ≠ 0 ∧ (a - (10 : ℂ) / (3 - I)).re = 0 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_l4058_405842


namespace NUMINAMATH_CALUDE_min_value_in_geometric_sequence_l4058_405838

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r > 0, ∀ k, a (k + 1) = r * a k

-- Define the theorem
theorem min_value_in_geometric_sequence (a : ℕ → ℝ) 
  (h1 : is_positive_geometric_sequence a) 
  (h2 : a 4 * a 14 = 8) : 
  (∀ x y, x > 0 ∧ y > 0 ∧ x * y = 8 → 2*x + y ≥ 8) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x * y = 8 ∧ 2*x + y = 8) :=
sorry

end NUMINAMATH_CALUDE_min_value_in_geometric_sequence_l4058_405838


namespace NUMINAMATH_CALUDE_sector_area_rate_of_change_l4058_405851

/-- The rate of change of a circular sector's area --/
theorem sector_area_rate_of_change
  (r : ℝ)
  (θ : ℝ → ℝ)
  (h_r : r = 12)
  (h_θ : ∀ t, θ t = 38 + 5 * t) :
  ∀ t, (deriv (λ t => (1/2) * r^2 * (θ t * π / 180))) t = 2 * π :=
sorry

end NUMINAMATH_CALUDE_sector_area_rate_of_change_l4058_405851


namespace NUMINAMATH_CALUDE_simplify_product_l4058_405816

theorem simplify_product : 8 * (15 / 4) * (-24 / 25) = -144 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_l4058_405816


namespace NUMINAMATH_CALUDE_power_relation_l4058_405884

theorem power_relation (a m n : ℝ) (h1 : a^m = 6) (h2 : a^(m-n) = 2) : a^n = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l4058_405884


namespace NUMINAMATH_CALUDE_quadrilateral_relation_l4058_405878

/-- Given four points A, B, C, D in a plane satisfying certain conditions,
    prove that CD = 12 / AB -/
theorem quadrilateral_relation (A B C D : ℝ × ℝ) :
  (∀ (t : ℝ), ‖A - D‖ = 2 ∧ ‖B - C‖ = 2) →
  (∀ (t : ℝ), ‖A - C‖ = 4 ∧ ‖B - D‖ = 4) →
  (∃ (P : ℝ × ℝ), ∃ (s t : ℝ), 0 ≤ s ∧ s ≤ 1 ∧ 0 ≤ t ∧ t ≤ 1 ∧
    P = (1 - s) • A + s • C ∧ P = (1 - t) • B + t • D) →
  ‖C - D‖ = 12 / ‖A - B‖ :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_relation_l4058_405878


namespace NUMINAMATH_CALUDE_diana_hourly_wage_l4058_405808

/-- Diana's work schedule and earnings --/
structure DianaWork where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculate Diana's hourly wage --/
def hourly_wage (d : DianaWork) : ℚ :=
  d.weekly_earnings / (d.monday_hours + d.tuesday_hours + d.wednesday_hours + d.thursday_hours + d.friday_hours)

/-- Theorem: Diana's hourly wage is $30 --/
theorem diana_hourly_wage :
  let d : DianaWork := {
    monday_hours := 10,
    tuesday_hours := 15,
    wednesday_hours := 10,
    thursday_hours := 15,
    friday_hours := 10,
    weekly_earnings := 1800
  }
  hourly_wage d = 30 := by sorry

end NUMINAMATH_CALUDE_diana_hourly_wage_l4058_405808


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l4058_405839

theorem geometric_arithmetic_sequence_problem (a b c : ℝ) : 
  a + b + c = 114 →
  b / a = c / b →
  b / a ≠ 1 →
  b - a = c - b →
  c - a = 24 * (b - a) →
  a = 2 ∧ b = 14 ∧ c = 98 := by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l4058_405839


namespace NUMINAMATH_CALUDE_function_inequality_l4058_405889

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_inequality (h : ∀ x, (x - 1) * (deriv^[2] f x) > 0) :
  f 0 + f 2 > 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l4058_405889


namespace NUMINAMATH_CALUDE_row_sum_is_odd_square_l4058_405876

/-- The sum of an arithmetic progression -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The statement to be proved -/
theorem row_sum_is_odd_square (n : ℕ) (h : n > 0) :
  arithmetic_sum n 1 (2 * n - 1) = (2 * n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_row_sum_is_odd_square_l4058_405876


namespace NUMINAMATH_CALUDE_quadratic_points_relationship_l4058_405807

theorem quadratic_points_relationship (m : ℝ) (y₁ y₂ y₃ : ℝ) : 
  (2^2 - 4*2 - m = y₁) →
  (3^2 - 4*3 - m = y₂) →
  ((-1)^2 - 4*(-1) - m = y₃) →
  (y₃ > y₂ ∧ y₂ > y₁) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_points_relationship_l4058_405807


namespace NUMINAMATH_CALUDE_johns_umbrella_cost_l4058_405862

/-- The total cost of John's umbrellas -/
def total_cost (house_umbrellas car_umbrellas cost_per_umbrella : ℕ) : ℕ :=
  (house_umbrellas + car_umbrellas) * cost_per_umbrella

/-- Proof that John's total cost for umbrellas is $24 -/
theorem johns_umbrella_cost :
  total_cost 2 1 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_johns_umbrella_cost_l4058_405862


namespace NUMINAMATH_CALUDE_luncheon_cost_theorem_l4058_405810

/-- Cost of a luncheon item -/
structure LuncheonItem where
  sandwich : ℚ
  coffee : ℚ
  pie : ℚ

/-- Calculate the total cost of a luncheon -/
def luncheonCost (item : LuncheonItem) (s c p : ℕ) : ℚ :=
  s * item.sandwich + c * item.coffee + p * item.pie

theorem luncheon_cost_theorem (item : LuncheonItem) : 
  luncheonCost item 2 5 1 = 3 ∧
  luncheonCost item 5 8 1 = 27/5 ∧
  luncheonCost item 3 4 1 = 18/5 →
  luncheonCost item 2 2 1 = 13/5 := by
sorry

#eval (13 : ℚ) / 5  -- Expected output: 2.6

end NUMINAMATH_CALUDE_luncheon_cost_theorem_l4058_405810


namespace NUMINAMATH_CALUDE_weight_sum_l4058_405854

/-- Given the weights of four people (a, b, c, d) in pairs,
    prove that the sum of the weights of the first and last person is 310 pounds. -/
theorem weight_sum (a b c d : ℝ) 
  (h1 : a + b = 280) 
  (h2 : b + c = 230) 
  (h3 : c + d = 260) : 
  a + d = 310 := by
  sorry

end NUMINAMATH_CALUDE_weight_sum_l4058_405854


namespace NUMINAMATH_CALUDE_dolphin_count_theorem_l4058_405897

/-- Given an initial number of dolphins in the ocean and a factor for additional dolphins joining,
    calculate the total number of dolphins after joining. -/
def total_dolphins (initial : ℕ) (joining_factor : ℕ) : ℕ :=
  initial + joining_factor * initial

/-- Theorem stating that with 65 initial dolphins and 3 times that number joining,
    the total number of dolphins is 260. -/
theorem dolphin_count_theorem :
  total_dolphins 65 3 = 260 := by
  sorry

end NUMINAMATH_CALUDE_dolphin_count_theorem_l4058_405897


namespace NUMINAMATH_CALUDE_ellipse_properties_l4058_405803

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = -8 * x

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  y = x - 2

theorem ellipse_properties (a b c : ℝ) :
  a > b ∧ b > 0 ∧
  ellipse a b (-Real.sqrt 3) 1 ∧
  c = 2 ∧
  b^2 = a^2 - 4 ∧
  (∃ x y, ellipse a b x y ∧ parabola x y) →
  (a^2 = 6 ∧
   (∀ x y, ellipse a b x y ↔ x^2 / 6 + y^2 / 2 = 1) ∧
   (∃ x₁ y₁ x₂ y₂, 
      ellipse a b x₁ y₁ ∧ 
      ellipse a b x₂ y₂ ∧
      line_l x₁ y₁ ∧
      line_l x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = 6)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l4058_405803


namespace NUMINAMATH_CALUDE_platform_length_l4058_405872

/-- Given a train that passes a pole and a platform, calculate the platform length -/
theorem platform_length (train_length : ℝ) (pole_time : ℝ) (platform_time : ℝ) :
  train_length = 100 →
  pole_time = 15 →
  platform_time = 40 →
  ∃ (platform_length : ℝ),
    platform_length = 500 / 3 ∧
    train_length / pole_time = (train_length + platform_length) / platform_time :=
by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l4058_405872


namespace NUMINAMATH_CALUDE_infinitely_many_primes_mod_3_eq_2_l4058_405875

theorem infinitely_many_primes_mod_3_eq_2 : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 3 = 2} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_mod_3_eq_2_l4058_405875


namespace NUMINAMATH_CALUDE_apples_left_l4058_405813

theorem apples_left (initial : ℝ) (gift : ℝ) (pie_needed : ℝ) 
  (h1 : initial = 10.0)
  (h2 : gift = 5.5)
  (h3 : pie_needed = 4.25) :
  initial + gift - pie_needed = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_l4058_405813


namespace NUMINAMATH_CALUDE_cubic_and_quadratic_equations_l4058_405835

theorem cubic_and_quadratic_equations :
  (∃ x : ℝ, 8 * x^3 = 27 ∧ x = 3/2) ∧
  (∃ x y : ℝ, (x - 2)^2 = 3 ∧ (y - 2)^2 = 3 ∧ 
   x = Real.sqrt 3 + 2 ∧ y = -Real.sqrt 3 + 2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_and_quadratic_equations_l4058_405835


namespace NUMINAMATH_CALUDE_min_value_xyz_l4058_405843

/-- Given positive real numbers x, y, and z satisfying 1/x + 1/y + 1/z = 9,
    the minimum value of x^2 * y^3 * z is 729/6912 -/
theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) :
  ∃ (m : ℝ), m = 729/6912 ∧ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    1/a + 1/b + 1/c = 9 → a^2 * b^3 * c ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_xyz_l4058_405843


namespace NUMINAMATH_CALUDE_jesse_gift_amount_l4058_405893

/-- Prove that Jesse received $50 as a gift -/
theorem jesse_gift_amount (novel_cost lunch_cost remaining_amount : ℕ) : 
  novel_cost = 7 →
  lunch_cost = 2 * novel_cost →
  remaining_amount = 29 →
  novel_cost + lunch_cost + remaining_amount = 50 := by
  sorry

end NUMINAMATH_CALUDE_jesse_gift_amount_l4058_405893


namespace NUMINAMATH_CALUDE_passengers_between_fourth_and_fifth_stops_l4058_405814

/-- Calculates the number of passengers on a bus after a given number of stops -/
def passengers_after_stops (initial_passengers : ℕ) (stops : ℕ) : ℕ :=
  initial_passengers + 2 * stops

/-- The number of passengers on the bus between the fourth and fifth stops -/
theorem passengers_between_fourth_and_fifth_stops :
  passengers_after_stops 18 3 = 24 := by
  sorry

#eval passengers_after_stops 18 3

end NUMINAMATH_CALUDE_passengers_between_fourth_and_fifth_stops_l4058_405814


namespace NUMINAMATH_CALUDE_lauryn_earnings_l4058_405890

theorem lauryn_earnings (x : ℝ) : 
  x + 0.7 * x = 3400 → x = 2000 := by
  sorry

end NUMINAMATH_CALUDE_lauryn_earnings_l4058_405890


namespace NUMINAMATH_CALUDE_lineup_combinations_l4058_405859

def total_members : ℕ := 12
def offensive_linemen : ℕ := 5
def positions_to_fill : ℕ := 5

def choose_lineup : ℕ := offensive_linemen * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

theorem lineup_combinations :
  choose_lineup = 39600 := by
  sorry

end NUMINAMATH_CALUDE_lineup_combinations_l4058_405859


namespace NUMINAMATH_CALUDE_area_triangle_QCA_l4058_405865

/-- The area of triangle QCA given the coordinates of points Q, A, and C -/
theorem area_triangle_QCA (p : ℝ) : 
  let Q : ℝ × ℝ := (0, 15)
  let A : ℝ × ℝ := (3, 15)
  let C : ℝ × ℝ := (0, p)
  let area := (1/2) * (A.1 - Q.1) * (Q.2 - C.2)
  area = 45/2 - 3*p/2 := by
sorry

end NUMINAMATH_CALUDE_area_triangle_QCA_l4058_405865


namespace NUMINAMATH_CALUDE_two_digit_primes_with_units_digit_9_l4058_405891

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def has_units_digit_9 (n : ℕ) : Prop := n % 10 = 9

theorem two_digit_primes_with_units_digit_9 :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, is_two_digit n ∧ has_units_digit_9 n ∧ Nat.Prime n) ∧ 
    (∀ n, is_two_digit n → has_units_digit_9 n → Nat.Prime n → n ∈ s) ∧
    s.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_primes_with_units_digit_9_l4058_405891


namespace NUMINAMATH_CALUDE_quiz_average_after_drop_l4058_405822

theorem quiz_average_after_drop (n : ℕ) (initial_avg : ℚ) (dropped_score : ℕ) :
  n = 16 →
  initial_avg = 60.5 →
  dropped_score = 8 →
  let total_score := n * initial_avg
  let remaining_score := total_score - dropped_score
  let new_avg := remaining_score / (n - 1)
  new_avg = 64 := by sorry

end NUMINAMATH_CALUDE_quiz_average_after_drop_l4058_405822


namespace NUMINAMATH_CALUDE_decimal_point_error_l4058_405860

theorem decimal_point_error (actual_amount : ℚ) : 
  (actual_amount * 10 - actual_amount = 153) → actual_amount = 17 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_error_l4058_405860


namespace NUMINAMATH_CALUDE_triangle_area_l4058_405825

theorem triangle_area (base height : ℝ) (h1 : base = 8) (h2 : height = 4) :
  (base * height) / 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l4058_405825


namespace NUMINAMATH_CALUDE_emily_final_score_l4058_405892

def emily_game (round1 round2 round3 round4 round5 round6_initial : Int) : Int :=
  let round6 := round6_initial - (2 * round5) / 3
  round1 + round2 + round3 + round4 + round5 + round6

theorem emily_final_score :
  emily_game 16 33 (-25) 46 12 30 = 104 := by
  sorry

end NUMINAMATH_CALUDE_emily_final_score_l4058_405892


namespace NUMINAMATH_CALUDE_joe_lift_weight_l4058_405841

theorem joe_lift_weight (first_lift second_lift : ℕ) 
  (total_weight : first_lift + second_lift = 900)
  (lift_relation : 2 * first_lift = second_lift + 300) :
  first_lift = 400 := by
  sorry

end NUMINAMATH_CALUDE_joe_lift_weight_l4058_405841


namespace NUMINAMATH_CALUDE_janet_lives_gained_l4058_405866

theorem janet_lives_gained (initial_lives : ℕ) (lives_lost : ℕ) (final_lives : ℕ) 
  (h1 : initial_lives = 47)
  (h2 : lives_lost = 23)
  (h3 : final_lives = 70) :
  final_lives - (initial_lives - lives_lost) = 46 := by
  sorry

end NUMINAMATH_CALUDE_janet_lives_gained_l4058_405866


namespace NUMINAMATH_CALUDE_roberts_balls_l4058_405833

theorem roberts_balls (robert_initial : ℕ) (tim_initial : ℕ) : 
  robert_initial = 25 → 
  tim_initial = 40 → 
  robert_initial + tim_initial / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_roberts_balls_l4058_405833


namespace NUMINAMATH_CALUDE_sum_of_integers_l4058_405827

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val ^ 2 + y.val ^ 2 = 130)
  (h2 : x.val * y.val = 36)
  (h3 : x.val - y.val = 4) :
  x.val + y.val = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l4058_405827


namespace NUMINAMATH_CALUDE_old_barbell_cost_l4058_405847

theorem old_barbell_cost (new_barbell_cost : ℝ) (percentage_increase : ℝ) : 
  new_barbell_cost = 325 →
  percentage_increase = 0.30 →
  new_barbell_cost = (1 + percentage_increase) * (new_barbell_cost / (1 + percentage_increase)) →
  new_barbell_cost / (1 + percentage_increase) = 250 := by
sorry

end NUMINAMATH_CALUDE_old_barbell_cost_l4058_405847


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l4058_405821

theorem necessary_not_sufficient_condition (a : ℝ) : 
  (∀ a, (1 / a > 1 → a < 1)) ∧ 
  (∃ a, a < 1 ∧ ¬(1 / a > 1)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l4058_405821


namespace NUMINAMATH_CALUDE_total_students_is_thirteen_l4058_405857

/-- The total number of students in a presentation lineup, given Eunjung's position and the number of students after her. -/
def total_students (eunjung_position : ℕ) (students_after : ℕ) : ℕ :=
  eunjung_position + students_after

/-- Theorem stating that the total number of students is 13, given the conditions from the problem. -/
theorem total_students_is_thirteen :
  total_students 6 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_thirteen_l4058_405857


namespace NUMINAMATH_CALUDE_lollipops_kept_by_winnie_l4058_405856

def cherry_lollipops : ℕ := 53
def wintergreen_lollipops : ℕ := 130
def grape_lollipops : ℕ := 12
def shrimp_cocktail_lollipops : ℕ := 240
def number_of_friends : ℕ := 13

def total_lollipops : ℕ := cherry_lollipops + wintergreen_lollipops + grape_lollipops + shrimp_cocktail_lollipops

theorem lollipops_kept_by_winnie :
  total_lollipops % number_of_friends = 6 :=
by sorry

end NUMINAMATH_CALUDE_lollipops_kept_by_winnie_l4058_405856


namespace NUMINAMATH_CALUDE_no_solution_arccos_arcsin_l4058_405817

theorem no_solution_arccos_arcsin : ¬∃ x : ℝ, Real.arccos (4/5) - Real.arccos (-4/5) = Real.arcsin x := by
  sorry

end NUMINAMATH_CALUDE_no_solution_arccos_arcsin_l4058_405817


namespace NUMINAMATH_CALUDE_min_xy_min_x_plus_y_min_values_exact_min_values_l4058_405885

-- Define the variables and conditions
variables (x y : ℝ) (hx : x > 0) (hy : y > 0)

-- Define the given equation
def equation_holds := Real.log x + Real.log y = Real.log (x + y + 3)

-- Theorem for the minimum value of xy
theorem min_xy (h : equation_holds x y) : 
  ∀ a b : ℝ, a > 0 → b > 0 → equation_holds a b → x * y ≤ a * b :=
sorry

-- Theorem for the minimum value of x + y
theorem min_x_plus_y (h : equation_holds x y) : 
  ∀ a b : ℝ, a > 0 → b > 0 → equation_holds a b → x + y ≤ a + b :=
sorry

-- Theorem stating the minimum values
theorem min_values (h : equation_holds x y) : x * y ≥ 9 ∧ x + y ≥ 6 :=
sorry

-- Theorem for the exact minimum values
theorem exact_min_values : ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ equation_holds x y ∧ x * y = 9 ∧ x + y = 6 :=
sorry

end NUMINAMATH_CALUDE_min_xy_min_x_plus_y_min_values_exact_min_values_l4058_405885


namespace NUMINAMATH_CALUDE_sum_of_intersection_points_l4058_405888

/-- A type representing a line in a plane -/
structure Line :=
  (id : ℕ)

/-- A type representing an intersection point of two lines -/
structure IntersectionPoint :=
  (line1 : Line)
  (line2 : Line)

/-- A configuration of lines in a plane -/
structure LineConfiguration :=
  (lines : Finset Line)
  (intersections : Finset IntersectionPoint)
  (distinct_lines : lines.card = 5)
  (no_triple_intersections : ∀ p q r : Line, p ∈ lines → q ∈ lines → r ∈ lines → 
    p ≠ q → q ≠ r → p ≠ r → 
    ¬∃ i : IntersectionPoint, i ∈ intersections ∧ 
      (i.line1 = p ∧ i.line2 = q) ∧
      (i.line1 = q ∧ i.line2 = r) ∧
      (i.line1 = p ∧ i.line2 = r))

/-- The theorem to be proved -/
theorem sum_of_intersection_points (config : LineConfiguration) :
  (Finset.range 11).sum (λ n => n * (Finset.filter (λ c : LineConfiguration => c.intersections.card = n) {config}).card) = 54 :=
sorry

end NUMINAMATH_CALUDE_sum_of_intersection_points_l4058_405888


namespace NUMINAMATH_CALUDE_work_completed_in_three_days_l4058_405858

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 14
def work_rate_C : ℚ := 1 / 7

-- Define the total work to be done
def total_work : ℚ := 1

-- Define the work done in the first two days by A and B
def work_done_first_two_days : ℚ := 2 * (work_rate_A + work_rate_B)

-- Define the work done on the third day by A, B, and C
def work_done_third_day : ℚ := work_rate_A + work_rate_B + work_rate_C

-- Theorem to prove
theorem work_completed_in_three_days :
  work_done_first_two_days + work_done_third_day ≥ total_work :=
by sorry

end NUMINAMATH_CALUDE_work_completed_in_three_days_l4058_405858


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l4058_405873

theorem quadratic_equivalence (c : ℝ) : 
  ({a : ℝ | ∀ x : ℝ, x^2 + a*x + a/4 + 1/2 > 0} = {x : ℝ | x^2 - x + c < 0}) → 
  c = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l4058_405873


namespace NUMINAMATH_CALUDE_inequality_solutions_l4058_405879

theorem inequality_solutions :
  (∀ x : ℝ, |x - 6| ≤ 2 ↔ 4 ≤ x ∧ x ≤ 8) ∧
  (∀ x : ℝ, (x + 3)^2 < 1 ↔ -4 < x ∧ x < -2) ∧
  (∀ x : ℝ, |x| > x ↔ x < 0) ∧
  (∀ x : ℝ, |x^2 - 4*x - 5| > x^2 - 4*x - 5 ↔ -1 < x ∧ x < 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l4058_405879


namespace NUMINAMATH_CALUDE_smallest_n_for_314_fraction_l4058_405887

def is_relatively_prime (m n : ℕ) : Prop := Nat.gcd m n = 1

def contains_314 (q : ℚ) : Prop :=
  ∃ k : ℕ, (10^k * q - (10^k * q).floor) * 1000 ≥ 314 ∧
            (10^k * q - (10^k * q).floor) * 1000 < 315

theorem smallest_n_for_314_fraction :
  ∃ (m n : ℕ), 
    n = 159 ∧
    m < n ∧
    is_relatively_prime m n ∧
    contains_314 (m / n) ∧
    (∀ (m' n' : ℕ), n' < 159 → m' < n' → is_relatively_prime m' n' → ¬contains_314 (m' / n')) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_314_fraction_l4058_405887


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l4058_405869

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a3 : a 3 = -4) 
  (h_a7 : a 7 = -16) : 
  a 5 = -8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l4058_405869


namespace NUMINAMATH_CALUDE_cookies_packages_bought_l4058_405870

def num_children : ℕ := 5
def cookies_per_package : ℕ := 25
def cookies_per_child : ℕ := 15

theorem cookies_packages_bought : 
  (num_children * cookies_per_child) / cookies_per_package = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookies_packages_bought_l4058_405870


namespace NUMINAMATH_CALUDE_no_real_roots_l4058_405819

theorem no_real_roots : ¬ ∃ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 6) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l4058_405819


namespace NUMINAMATH_CALUDE_factorization_problems_l4058_405830

variable (a b : ℝ)

theorem factorization_problems :
  (-25 + a^4 = (a^2 + 5) * (a + 5) * (a - 5)) ∧
  (a^3 * b - 10 * a^2 * b + 25 * a * b = a * b * (a - 5)^2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problems_l4058_405830


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_2102012_base7_l4058_405845

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- Gets the largest prime divisor of a number -/
def largestPrimeDivisor (n : ℕ) : ℕ := sorry

theorem largest_prime_divisor_of_2102012_base7 :
  largestPrimeDivisor (base7ToBase10 2102012) = 79 := by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_2102012_base7_l4058_405845


namespace NUMINAMATH_CALUDE_pineapple_cost_theorem_l4058_405804

/-- The cost of each pineapple before shipping -/
def pineapple_cost_before_shipping (n : ℕ) (shipping_cost total_cost_per_pineapple : ℚ) : ℚ :=
  total_cost_per_pineapple - (shipping_cost / n)

/-- Theorem: The cost of each pineapple before shipping is $1.25 -/
theorem pineapple_cost_theorem (n : ℕ) (shipping_cost total_cost_per_pineapple : ℚ) 
  (h1 : n = 12)
  (h2 : shipping_cost = 21)
  (h3 : total_cost_per_pineapple = 3) :
  pineapple_cost_before_shipping n shipping_cost total_cost_per_pineapple = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_pineapple_cost_theorem_l4058_405804


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l4058_405840

/-- A quadratic function that takes values 6, 5, and 5 for three consecutive natural values. -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = 6 ∧ f (n + 1) = 5 ∧ f (n + 2) = 5

/-- The theorem stating that the minimum value of the quadratic function is 5. -/
theorem quadratic_function_minimum (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l4058_405840


namespace NUMINAMATH_CALUDE_inequality_of_means_l4058_405877

theorem inequality_of_means (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a + a^2) / 2 > a^(3/2) ∧ a^(3/2) > 2 * a^2 / (1 + a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_means_l4058_405877


namespace NUMINAMATH_CALUDE_P_greater_than_Q_l4058_405880

theorem P_greater_than_Q (a : ℝ) : (a^2 + 2*a) > (3*a - 1) := by
  sorry

end NUMINAMATH_CALUDE_P_greater_than_Q_l4058_405880


namespace NUMINAMATH_CALUDE_rachel_reading_homework_l4058_405867

/-- The number of pages of math homework Rachel had to complete -/
def math_homework_pages : ℕ := 8

/-- The additional pages of reading homework compared to math homework -/
def additional_reading_pages : ℕ := 6

/-- The total number of pages of reading homework Rachel had to complete -/
def reading_homework_pages : ℕ := math_homework_pages + additional_reading_pages

theorem rachel_reading_homework : reading_homework_pages = 14 := by
  sorry

end NUMINAMATH_CALUDE_rachel_reading_homework_l4058_405867


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l4058_405863

theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n, ∃ r, a (n + 1) = r * a n) →
  (a 1)^2 - 10*(a 1) + 16 = 0 →
  (a 19)^2 - 10*(a 19) + 16 = 0 →
  a 8 * a 10 * a 12 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l4058_405863


namespace NUMINAMATH_CALUDE_diana_paint_remaining_l4058_405849

/-- The amount of paint required for each statue in gallons -/
def paint_per_statue : ℚ := 1 / 16

/-- The number of statues Diana can paint -/
def number_of_statues : ℕ := 14

/-- The total amount of paint Diana has remaining in gallons -/
def total_paint : ℚ := paint_per_statue * number_of_statues

/-- Theorem stating that the total paint Diana has remaining is 7/8 gallon -/
theorem diana_paint_remaining : total_paint = 7 / 8 := by sorry

end NUMINAMATH_CALUDE_diana_paint_remaining_l4058_405849


namespace NUMINAMATH_CALUDE_max_value_of_inequality_l4058_405826

theorem max_value_of_inequality (x : ℝ) : 
  (∀ y : ℝ, (6 + 5*y + y^2) * Real.sqrt (2*y^2 - y^3 - y) ≤ 0 → y ≤ x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_inequality_l4058_405826


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_l4058_405855

theorem sqrt_50_between_consecutive_integers : ∃ (n : ℕ), n > 0 ∧ n^2 < 50 ∧ (n+1)^2 > 50 ∧ n * (n+1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_l4058_405855


namespace NUMINAMATH_CALUDE_range_of_m_l4058_405850

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h_eq : 2/x + 3/y = 1) 
  (h_ineq : ∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 3/y = 1 → 3*x + 2*y > m^2 + 2*m) : 
  -6 < m ∧ m < 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l4058_405850


namespace NUMINAMATH_CALUDE_no_solution_implies_m_greater_2023_l4058_405848

theorem no_solution_implies_m_greater_2023 (m : ℝ) :
  (∀ x : ℝ, ¬(x ≥ m ∧ x ≤ 2023)) → m > 2023 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_greater_2023_l4058_405848


namespace NUMINAMATH_CALUDE_books_combination_l4058_405823

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem books_combination : choose 15 3 = 455 := by sorry

end NUMINAMATH_CALUDE_books_combination_l4058_405823


namespace NUMINAMATH_CALUDE_probability_theorem_l4058_405837

def total_shoes : ℕ := 28
def black_pairs : ℕ := 7
def brown_pairs : ℕ := 4
def gray_pairs : ℕ := 2
def white_pairs : ℕ := 1

def probability_same_color_left_right : ℚ :=
  (black_pairs * 2 / total_shoes) * (black_pairs / (total_shoes - 1)) +
  (brown_pairs * 2 / total_shoes) * (brown_pairs / (total_shoes - 1)) +
  (gray_pairs * 2 / total_shoes) * (gray_pairs / (total_shoes - 1)) +
  (white_pairs * 2 / total_shoes) * (white_pairs / (total_shoes - 1))

theorem probability_theorem :
  probability_same_color_left_right = 35 / 189 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l4058_405837


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l4058_405820

/-- The number of games played in a chess tournament where each participant
    plays exactly one game with each other participant. -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that a chess tournament with 105 games has 15 participants. -/
theorem chess_tournament_participants :
  ∃ n : ℕ, n > 0 ∧ num_games n = 105 ∧ n = 15 := by
  sorry

#check chess_tournament_participants

end NUMINAMATH_CALUDE_chess_tournament_participants_l4058_405820


namespace NUMINAMATH_CALUDE_island_perimeter_l4058_405829

/-- The perimeter of an island consisting of an equilateral triangle and two half circles -/
theorem island_perimeter (base : ℝ) (h : base = 4) : 
  let triangle_perimeter := 3 * base
  let half_circles_perimeter := 2 * π * base
  triangle_perimeter + half_circles_perimeter = 12 + 4 * π := by
  sorry

end NUMINAMATH_CALUDE_island_perimeter_l4058_405829


namespace NUMINAMATH_CALUDE_log_xyz_t_equals_three_l4058_405809

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_xyz_t_equals_three 
  (t x y z : ℝ) 
  (h1 : log x t = 6)
  (h2 : log y t = 10)
  (h3 : log z t = 15) :
  log (x * y * z) t = 3 :=
by sorry

end NUMINAMATH_CALUDE_log_xyz_t_equals_three_l4058_405809


namespace NUMINAMATH_CALUDE_theorem_1_theorem_2_theorem_3_l4058_405805

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - 1

-- Theorem 1
theorem theorem_1 (a : ℝ) :
  f a 1 = 2 → a = 1 ∧ ∀ x, f 1 x ≥ -2 :=
sorry

-- Theorem 2
theorem theorem_2 (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 0 :=
sorry

-- Theorem 3
theorem theorem_3 (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ x, f a x ≤ f a y) → a ≤ -4 :=
sorry

end NUMINAMATH_CALUDE_theorem_1_theorem_2_theorem_3_l4058_405805


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l4058_405846

/-- The number of people at the table -/
def total_people : ℕ := 8

/-- The number of people on each side of Cara -/
def people_per_side : ℕ := 3

/-- The number of potential neighbors for Cara -/
def potential_neighbors : ℕ := 2 * people_per_side - 2

/-- The number of people in each pair next to Cara -/
def pair_size : ℕ := 2

theorem cara_seating_arrangements :
  Nat.choose potential_neighbors pair_size = 6 :=
sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l4058_405846


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_equation_proof_l4058_405832

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point2D.liesOn (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line2D.isParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (given_point : Point2D) 
  (given_line : Line2D) 
  (result_line : Line2D) : Prop :=
  (given_point.liesOn result_line) ∧ 
  (result_line.isParallel given_line) →
  (result_line.a = 2 ∧ result_line.b = -1 ∧ result_line.c = 4)

#check line_through_point_parallel_to_line 
  (Point2D.mk 0 4) 
  (Line2D.mk 2 (-1) (-3)) 
  (Line2D.mk 2 (-1) 4)

theorem line_equation_proof : 
  line_through_point_parallel_to_line 
    (Point2D.mk 0 4) 
    (Line2D.mk 2 (-1) (-3)) 
    (Line2D.mk 2 (-1) 4) := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_equation_proof_l4058_405832


namespace NUMINAMATH_CALUDE_number_of_boys_l4058_405815

theorem number_of_boys (initial_avg : ℝ) (incorrect_height : ℝ) (correct_height : ℝ) (actual_avg : ℝ) :
  initial_avg = 184 →
  incorrect_height = 166 →
  correct_height = 106 →
  actual_avg = 182 →
  ∃ n : ℕ, n * initial_avg - (incorrect_height - correct_height) = n * actual_avg ∧ n = 30 :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_l4058_405815


namespace NUMINAMATH_CALUDE_rectangle_max_volume_l4058_405828

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

end NUMINAMATH_CALUDE_rectangle_max_volume_l4058_405828


namespace NUMINAMATH_CALUDE_price_increase_quantity_decrease_l4058_405831

theorem price_increase_quantity_decrease (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let original_cost := P * Q
  let new_price := P * 1.15
  let new_quantity := Q * 0.6
  let new_cost := new_price * new_quantity
  new_cost = original_cost * 0.69 :=
by sorry

end NUMINAMATH_CALUDE_price_increase_quantity_decrease_l4058_405831


namespace NUMINAMATH_CALUDE_quadratic_abs_equivalence_l4058_405824

theorem quadratic_abs_equivalence (a : ℝ) : a^2 + 4*a - 5 > 0 ↔ |a + 2| > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_abs_equivalence_l4058_405824


namespace NUMINAMATH_CALUDE_circle_quadratic_intersection_l4058_405836

/-- Given a circle and a quadratic equation, prove the center coordinates and condition --/
theorem circle_quadratic_intersection (p q b c : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*p*x - 2*q*y + 2*q - 1 = 0 ↔ 
   (y = 0 → x^2 + b*x + c = 0)) →
  (p = -b/2 ∧ q = (1+c)/2 ∧ b^2 - 4*c ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_circle_quadratic_intersection_l4058_405836


namespace NUMINAMATH_CALUDE_max_min_on_interval_l4058_405800

def f (x : ℝ) := x^3 - 3*x^2 + 5

theorem max_min_on_interval :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc 1 3, f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc 1 3, f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc 1 3, f x₂ = max) ∧
    min = 1 ∧ max = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_min_on_interval_l4058_405800


namespace NUMINAMATH_CALUDE_vector_operations_l4058_405853

def a : Fin 2 → ℝ := ![3, 4]
def b : Fin 2 → ℝ := ![2, -6]
def c : Fin 2 → ℝ := ![4, 1]

theorem vector_operations :
  (a • (a + b) = 7) ∧
  (c = a + (1/2 : ℝ) • b) := by sorry

end NUMINAMATH_CALUDE_vector_operations_l4058_405853


namespace NUMINAMATH_CALUDE_apple_fractions_l4058_405818

/-- Given that Simone ate 1/2 of an apple each day for 16 days,
    Lauri ate x fraction of an apple each day for 15 days,
    and the total number of apples eaten by both girls is 13,
    prove that x = 1/3 -/
theorem apple_fractions (x : ℚ) : 
  (16 * (1/2 : ℚ)) + (15 * x) = 13 → x = 1/3 := by sorry

end NUMINAMATH_CALUDE_apple_fractions_l4058_405818


namespace NUMINAMATH_CALUDE_smaller_to_larger_volume_ratio_l4058_405899

/-- Represents a regular octahedron -/
structure RegularOctahedron where
  -- Add necessary fields if needed

/-- Represents the smaller octahedron formed by face centers -/
def smaller_octahedron (o : RegularOctahedron) : RegularOctahedron :=
  sorry

/-- Calculates the volume of an octahedron -/
def volume (o : RegularOctahedron) : ℝ :=
  sorry

/-- Theorem stating the volume ratio of smaller to larger octahedron -/
theorem smaller_to_larger_volume_ratio (o : RegularOctahedron) :
  volume (smaller_octahedron o) / volume o = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_smaller_to_larger_volume_ratio_l4058_405899


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l4058_405874

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    (4 * π * r^2 : ℝ) = 400 * π → 
    (4 / 3 : ℝ) * π * r^3 = (4000 / 3 : ℝ) * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l4058_405874


namespace NUMINAMATH_CALUDE_equal_projections_imply_relation_l4058_405864

-- Define the points A, B, and C
def A (a : ℝ) : ℝ × ℝ := (a, 1)
def B (b : ℝ) : ℝ × ℝ := (2, b)
def C : ℝ × ℝ := (3, 4)

-- Define vectors OA, OB, and OC
def OA (a : ℝ) : ℝ × ℝ := A a
def OB (b : ℝ) : ℝ × ℝ := B b
def OC : ℝ × ℝ := C

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem equal_projections_imply_relation (a b : ℝ) :
  dot_product (OA a) OC = dot_product (OB b) OC →
  3 * a - 4 * b = 2 := by
  sorry


end NUMINAMATH_CALUDE_equal_projections_imply_relation_l4058_405864


namespace NUMINAMATH_CALUDE_unique_solution_sin_system_l4058_405861

theorem unique_solution_sin_system (a b c d : Real) 
  (h_sum : a + b + c + d = Real.pi) :
  ∃! (x y z w : Real),
    x = Real.sin (a + b) ∧
    y = Real.sin (b + c) ∧
    z = Real.sin (c + d) ∧
    w = Real.sin (d + a) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_sin_system_l4058_405861


namespace NUMINAMATH_CALUDE_lemon_pie_degrees_l4058_405882

/-- The number of degrees in a circle -/
def circle_degrees : ℕ := 360

/-- The total number of students -/
def total_students : ℕ := 45

/-- The number of students preferring chocolate pie -/
def chocolate_preference : ℕ := 15

/-- The number of students preferring apple pie -/
def apple_preference : ℕ := 10

/-- The number of students preferring blueberry pie -/
def blueberry_preference : ℕ := 7

/-- The number of students preferring lemon pie -/
def lemon_preference : ℕ := (total_students - (chocolate_preference + apple_preference + blueberry_preference)) / 2

theorem lemon_pie_degrees :
  (lemon_preference : ℚ) / total_students * circle_degrees = 56 := by
  sorry

end NUMINAMATH_CALUDE_lemon_pie_degrees_l4058_405882


namespace NUMINAMATH_CALUDE_g_over_log16_2_eq_4n_l4058_405834

/-- Sum of squares of elements in nth row of Pascal's triangle -/
def pascal_row_sum_squares (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- Base-16 logarithm function -/
noncomputable def log16 (x : ℝ) : ℝ := Real.log x / Real.log 16

/-- Function g(n) as defined in the problem -/
noncomputable def g (n : ℕ) : ℝ := log16 (pascal_row_sum_squares n)

/-- Theorem stating the relationship between g(n) and n -/
theorem g_over_log16_2_eq_4n (n : ℕ) : g n / log16 2 = 4 * n := by sorry

end NUMINAMATH_CALUDE_g_over_log16_2_eq_4n_l4058_405834


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_four_zeros_l4058_405896

theorem smallest_multiplier_for_four_zeros (n : ℕ) : 
  (∀ m : ℕ, m > 0 → m < n → ¬(10000 ∣ (975 * 935 * 972 * m))) →
  (10000 ∣ (975 * 935 * 972 * n)) →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_four_zeros_l4058_405896


namespace NUMINAMATH_CALUDE_independence_test_distribution_X_expected_value_Y_variance_Y_l4058_405883

-- Define the contingency table
def male_noodles : ℕ := 30
def male_rice : ℕ := 25
def female_noodles : ℕ := 20
def female_rice : ℕ := 25
def total_students : ℕ := 100

-- Define the chi-square formula
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value at α = 0.05
def critical_value : ℚ := 3841 / 1000

-- Theorem for independence test
theorem independence_test :
  chi_square male_noodles male_rice female_noodles female_rice < critical_value :=
sorry

-- Define the distribution of X
def prob_X (x : ℕ) : ℚ :=
  match x with
  | 0 => 3 / 10
  | 1 => 3 / 5
  | 2 => 1 / 10
  | _ => 0

-- Theorem for the distribution of X
theorem distribution_X :
  (prob_X 0 + prob_X 1 + prob_X 2 = 1) ∧
  (∀ x, x > 2 → prob_X x = 0) :=
sorry

-- Define Y as a binomial distribution
def p_Y : ℚ := 3 / 5
def n_Y : ℕ := 3

-- Theorems for expected value and variance of Y
theorem expected_value_Y :
  (n_Y : ℚ) * p_Y = 9 / 5 :=
sorry

theorem variance_Y :
  (n_Y : ℚ) * p_Y * (1 - p_Y) = 18 / 25 :=
sorry

end NUMINAMATH_CALUDE_independence_test_distribution_X_expected_value_Y_variance_Y_l4058_405883


namespace NUMINAMATH_CALUDE_population_decrease_proof_l4058_405871

/-- The annual rate of population decrease -/
def annual_decrease_rate : ℝ := 0.1

/-- The population after 2 years -/
def population_after_2_years : ℕ := 6480

/-- The initial population of the town -/
def initial_population : ℕ := 8000

theorem population_decrease_proof :
  (1 - annual_decrease_rate)^2 * initial_population = population_after_2_years :=
by sorry

end NUMINAMATH_CALUDE_population_decrease_proof_l4058_405871


namespace NUMINAMATH_CALUDE_davids_windows_l4058_405844

/-- Represents the time taken to wash windows -/
def wash_time : ℕ := 160

/-- Represents the number of windows washed in a single set -/
def windows_per_set : ℕ := 4

/-- Represents the time taken to wash one set of windows -/
def time_per_set : ℕ := 10

/-- Theorem stating the number of windows in David's house -/
theorem davids_windows : 
  (wash_time / time_per_set) * windows_per_set = 64 := by
  sorry

end NUMINAMATH_CALUDE_davids_windows_l4058_405844


namespace NUMINAMATH_CALUDE_point_on_line_with_distance_l4058_405868

theorem point_on_line_with_distance (x₀ y₀ : ℝ) :
  (3 * x₀ + y₀ - 5 = 0) →
  (|x₀ - y₀ - 1| / Real.sqrt 2 = Real.sqrt 2) →
  ((x₀ = 1 ∧ y₀ = 2) ∨ (x₀ = 2 ∧ y₀ = -1)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_with_distance_l4058_405868


namespace NUMINAMATH_CALUDE_parabola_circle_theorem_l4058_405895

/-- Given a parabola y = ax^2 + bx + c (a ≠ 0) intersecting the x-axis at points A and B,
    the equation of the circle with AB as diameter is ax^2 + bx + c + ay^2 = 0. -/
theorem parabola_circle_theorem (a b c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ →
  ∀ x y : ℝ, a * x^2 + b * x + c + a * y^2 = 0 ↔ 
    ∃ t : ℝ, x = (1 - t) * x₁ + t * x₂ ∧ 
             y^2 = t * (1 - t) * (x₂ - x₁)^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_circle_theorem_l4058_405895


namespace NUMINAMATH_CALUDE_shortest_side_length_l4058_405812

/-- Triangle with specific properties -/
structure SpecialTriangle where
  -- Base of the triangle
  base : ℝ
  -- One base angle in radians
  baseAngle : ℝ
  -- Sum of the other two sides
  sumOtherSides : ℝ
  -- Conditions
  base_positive : base > 0
  baseAngle_in_range : 0 < baseAngle ∧ baseAngle < π
  sumOtherSides_positive : sumOtherSides > 0

/-- The length of the shortest side in the special triangle -/
def shortestSide (t : SpecialTriangle) : ℝ := sorry

/-- Theorem stating the length of the shortest side in the specific triangle -/
theorem shortest_side_length (t : SpecialTriangle) 
  (h1 : t.base = 80)
  (h2 : t.baseAngle = π / 3)  -- 60° in radians
  (h3 : t.sumOtherSides = 90) :
  shortestSide t = 40 := by sorry

end NUMINAMATH_CALUDE_shortest_side_length_l4058_405812
