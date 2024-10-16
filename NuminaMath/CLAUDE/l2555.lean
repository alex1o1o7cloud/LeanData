import Mathlib

namespace NUMINAMATH_CALUDE_cup_purchase_properties_prize_purchase_properties_l2555_255586

/-- Represents the cost and quantity of insulated cups --/
structure CupPurchase where
  cost_a : ℕ  -- Cost of A type cup
  cost_b : ℕ  -- Cost of B type cup
  quantity_a : ℕ  -- Quantity of A type cups
  quantity_b : ℕ  -- Quantity of B type cups

/-- Theorem stating the properties of the cup purchase --/
theorem cup_purchase_properties :
  ∃ (purchase : CupPurchase),
    -- B type cup costs 10 yuan more than A type cup
    purchase.cost_b = purchase.cost_a + 10 ∧
    -- 1200 yuan buys 1.5 times as many A cups as 1000 yuan buys B cups
    1200 / purchase.cost_a = (3/2) * (1000 / purchase.cost_b) ∧
    -- Company buys 9 fewer B cups than A cups
    purchase.quantity_b = purchase.quantity_a - 9 ∧
    -- Number of A cups is not less than 38
    purchase.quantity_a ≥ 38 ∧
    -- Total cost does not exceed 3150 yuan
    purchase.cost_a * purchase.quantity_a + purchase.cost_b * purchase.quantity_b ≤ 3150 ∧
    -- Cost of A type cup is 40 yuan
    purchase.cost_a = 40 ∧
    -- Cost of B type cup is 50 yuan
    purchase.cost_b = 50 ∧
    -- There are exactly three valid purchasing schemes
    (∃ (scheme1 scheme2 scheme3 : CupPurchase),
      scheme1.quantity_a = 38 ∧ scheme1.quantity_b = 29 ∧
      scheme2.quantity_a = 39 ∧ scheme2.quantity_b = 30 ∧
      scheme3.quantity_a = 40 ∧ scheme3.quantity_b = 31 ∧
      ∀ (other : CupPurchase),
        (other.quantity_a ≥ 38 ∧
         other.quantity_b = other.quantity_a - 9 ∧
         other.cost_a * other.quantity_a + other.cost_b * other.quantity_b ≤ 3150) →
        (other = scheme1 ∨ other = scheme2 ∨ other = scheme3)) :=
by
  sorry

/-- Represents the quantity of prizes --/
structure PrizePurchase where
  quantity_a : ℕ  -- Quantity of A type prizes
  quantity_b : ℕ  -- Quantity of B type prizes

/-- Theorem stating the properties of the prize purchase --/
theorem prize_purchase_properties :
  ∃ (prize : PrizePurchase),
    -- A type prize costs 270 yuan
    -- B type prize costs 240 yuan
    -- Total cost of prizes equals minimum cost from part 2 (2970 yuan)
    270 * prize.quantity_a + 240 * prize.quantity_b = 2970 ∧
    -- There are 3 A type prizes and 9 B type prizes
    prize.quantity_a = 3 ∧
    prize.quantity_b = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_cup_purchase_properties_prize_purchase_properties_l2555_255586


namespace NUMINAMATH_CALUDE_circle_equation_proof_l2555_255506

theorem circle_equation_proof (x y : ℝ) : 
  let equation := x^2 + y^2 - 10*y
  let center := (0, 5)
  let radius := 5
  -- The circle's equation
  (equation = 0) →
  -- Center is on the y-axis
  (center.1 = 0) ∧
  -- Circle is tangent to x-axis (distance from center to x-axis equals radius)
  (center.2 = radius) ∧
  -- Circle passes through (3, 1)
  ((3 - center.1)^2 + (1 - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l2555_255506


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l2555_255527

/-- Given plane vectors a and b, if ka + b is perpendicular to a, then k = -1/5 -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h3 : (k • a.1 + b.1) * a.1 + (k • a.2 + b.2) * a.2 = 0) :
  k = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l2555_255527


namespace NUMINAMATH_CALUDE_triangle_properties_l2555_255578

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  c = Real.sqrt 2 →
  Real.cos A = -(Real.sqrt 2) / 4 →
  a = b →
  b = c →
  c = a →
  Real.sin C = Real.sqrt 7 / 4 ∧
  b = 1 ∧
  Real.cos (2 * A + π / 3) = (-3 + Real.sqrt 21) / 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2555_255578


namespace NUMINAMATH_CALUDE_lcm_of_12_and_18_l2555_255564

theorem lcm_of_12_and_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_and_18_l2555_255564


namespace NUMINAMATH_CALUDE_secret_spread_day_l2555_255533

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day when 3280 people know the secret -/
theorem secret_spread_day :
  ∃ n : ℕ, secret_spread n = 3280 ∧ n = 7 :=
sorry

end NUMINAMATH_CALUDE_secret_spread_day_l2555_255533


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2555_255587

/-- The polar equation ρ = 4cosθ is equivalent to the Cartesian equation (x-2)^2 + y^2 = 4 -/
theorem polar_to_cartesian_circle :
  ∀ (x y ρ θ : ℝ), 
  (ρ = 4 * Real.cos θ) ∧ 
  (x = ρ * Real.cos θ) ∧ 
  (y = ρ * Real.sin θ) → 
  ((x - 2)^2 + y^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2555_255587


namespace NUMINAMATH_CALUDE_square_roots_problem_l2555_255519

theorem square_roots_problem (a m : ℝ) (ha : 0 < a) 
  (h1 : (2 * m - 1)^2 = a) (h2 : (m + 4)^2 = a) : a = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2555_255519


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l2555_255589

theorem solve_quadratic_equation : 
  ∃ x : ℚ, (10 - 2*x)^2 = 4*x^2 + 20*x ∧ x = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l2555_255589


namespace NUMINAMATH_CALUDE_basketball_games_count_l2555_255547

/-- Proves that a basketball team played 94 games in a season given specific conditions -/
theorem basketball_games_count :
  ∀ (total_games : ℕ) 
    (first_40_wins : ℕ) 
    (remaining_wins : ℕ),
  first_40_wins = 14 →  -- 35% of 40 games
  remaining_wins ≥ (0.7 : ℝ) * (total_games - 40) →  -- At least 70% of remaining games
  first_40_wins + remaining_wins = (0.55 : ℝ) * total_games →  -- 55% total win rate
  total_games = 94 := by
sorry

end NUMINAMATH_CALUDE_basketball_games_count_l2555_255547


namespace NUMINAMATH_CALUDE_product_minus_constant_l2555_255558

theorem product_minus_constant (P Q R S : ℕ+) : 
  (P + Q + R + S : ℝ) = 104 →
  (P : ℝ) + 5 = (Q : ℝ) - 5 →
  (P : ℝ) + 5 = (R : ℝ) * 2 →
  (P : ℝ) + 5 = (S : ℝ) / 2 →
  (P : ℝ) * (Q : ℝ) * (R : ℝ) * (S : ℝ) - 200 = 267442.5 := by
sorry

end NUMINAMATH_CALUDE_product_minus_constant_l2555_255558


namespace NUMINAMATH_CALUDE_max_stamps_purchasable_l2555_255530

def stamp_price : ℕ := 35
def discount_threshold : ℕ := 100
def discount_rate : ℚ := 5 / 100
def budget : ℕ := 3200

theorem max_stamps_purchasable :
  let max_stamps := (budget / stamp_price : ℕ)
  let discounted_price := stamp_price * (1 - discount_rate)
  let max_stamps_with_discount := (budget / discounted_price).floor
  (max_stamps_with_discount ≤ discount_threshold) ∧
  (max_stamps = 91) := by
sorry

end NUMINAMATH_CALUDE_max_stamps_purchasable_l2555_255530


namespace NUMINAMATH_CALUDE_toby_speed_proof_l2555_255509

/-- Toby's speed when pulling an unloaded sled -/
def unloaded_speed : ℝ := 20

/-- Distance of the first loaded part of the journey -/
def loaded_distance1 : ℝ := 180

/-- Distance of the first unloaded part of the journey -/
def unloaded_distance1 : ℝ := 120

/-- Distance of the second loaded part of the journey -/
def loaded_distance2 : ℝ := 80

/-- Distance of the second unloaded part of the journey -/
def unloaded_distance2 : ℝ := 140

/-- Total time of the journey -/
def total_time : ℝ := 39

/-- Toby's speed when pulling a loaded sled -/
def loaded_speed : ℝ := 10

theorem toby_speed_proof :
  (loaded_distance1 / loaded_speed + unloaded_distance1 / unloaded_speed +
   loaded_distance2 / loaded_speed + unloaded_distance2 / unloaded_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_toby_speed_proof_l2555_255509


namespace NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l2555_255591

-- Problem 1
theorem calculation_proof :
  Real.sqrt 4 - 2 * Real.sin (45 * π / 180) + (1/3)⁻¹ + |-(Real.sqrt 2)| = 5 := by sorry

-- Problem 2
theorem inequality_system_solution (x : ℝ) :
  (3*x + 1 < 2*x + 3 ∧ 2*x > (3*x - 1)/2) ↔ (-1 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l2555_255591


namespace NUMINAMATH_CALUDE_combined_age_proof_l2555_255508

def jeremy_age : ℕ := 66

theorem combined_age_proof (amy_age chris_age : ℕ) 
  (h1 : amy_age = jeremy_age / 3)
  (h2 : chris_age = 2 * amy_age) :
  amy_age + jeremy_age + chris_age = 132 := by
  sorry

end NUMINAMATH_CALUDE_combined_age_proof_l2555_255508


namespace NUMINAMATH_CALUDE_chicken_cost_per_person_l2555_255534

/-- Given the cost of groceries and the agreement to split the cost of chicken,
    this theorem proves the amount each person should pay for the chicken. -/
theorem chicken_cost_per_person
  (beef_price : ℝ)
  (beef_weight : ℝ)
  (oil_price : ℝ)
  (total_cost : ℝ)
  (num_people : ℕ)
  (h1 : beef_price = 4)
  (h2 : beef_weight = 3)
  (h3 : oil_price = 1)
  (h4 : total_cost = 16)
  (h5 : num_people = 3)
  : (total_cost - (beef_price * beef_weight + oil_price)) / num_people = 1 := by
  sorry

#check chicken_cost_per_person

end NUMINAMATH_CALUDE_chicken_cost_per_person_l2555_255534


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l2555_255557

/-- The displacement function of an object with respect to time -/
def displacement (t : ℝ) : ℝ := 4 - 2*t + t^2

/-- The velocity function of an object with respect to time -/
def velocity (t : ℝ) : ℝ := 2*t - 2

theorem instantaneous_velocity_at_4_seconds :
  velocity 4 = 6 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l2555_255557


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l2555_255561

/-- The quadratic inequality kx^2 - 2x + 6k < 0 -/
def quadratic_inequality (k : ℝ) (x : ℝ) : Prop := k * x^2 - 2 * x + 6 * k < 0

/-- The solution set for case 1: x < -3 or x > -2 -/
def solution_set_1 (x : ℝ) : Prop := x < -3 ∨ x > -2

/-- The solution set for case 2: all real numbers -/
def solution_set_2 (x : ℝ) : Prop := True

/-- The solution set for case 3: empty set -/
def solution_set_3 (x : ℝ) : Prop := False

theorem quadratic_inequality_solutions (k : ℝ) (h : k ≠ 0) :
  (∀ x, quadratic_inequality k x ↔ solution_set_1 x) → k = -2/5 ∧
  (∀ x, quadratic_inequality k x ↔ solution_set_2 x) → k < -Real.sqrt 6 / 6 ∧
  (∀ x, quadratic_inequality k x ↔ solution_set_3 x) → k ≥ Real.sqrt 6 / 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l2555_255561


namespace NUMINAMATH_CALUDE_math_only_students_l2555_255516

theorem math_only_students (total : ℕ) (math : ℕ) (foreign : ℕ) 
  (h1 : total = 93) 
  (h2 : math = 70) 
  (h3 : foreign = 54) : 
  math - (math + foreign - total) = 39 := by
  sorry

end NUMINAMATH_CALUDE_math_only_students_l2555_255516


namespace NUMINAMATH_CALUDE_ellipse_properties_l2555_255566

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
def short_axis_length (b : ℝ) : Prop :=
  2 * b = 2 * Real.sqrt 3

def slope_product (a : ℝ) (x y : ℝ) : Prop :=
  y^2 / (x^2 - a^2) = 3 / 4

-- Define the theorem
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : short_axis_length b) (h4 : ∀ x y, ellipse a b x y → slope_product a x y) :
  (∀ x y, ellipse a b x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ m : ℝ, m ≠ 0 → 
    ∃ Q : ℝ × ℝ, 
      (∃ A B : ℝ × ℝ, 
        ellipse a b A.1 A.2 ∧ 
        ellipse a b B.1 B.2 ∧
        A.1 = m * A.2 + 1 ∧
        B.1 = m * B.2 + 1 ∧
        Q.1 = A.1 + (Q.2 - A.2) * (A.1 + a) / A.2 ∧
        Q.1 = B.1 + (Q.2 - B.2) * (B.1 - a) / B.2) →
      Q.1 = 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2555_255566


namespace NUMINAMATH_CALUDE_transformer_current_load_transformer_current_load_is_700A_l2555_255500

theorem transformer_current_load : ℕ → Prop :=
  fun total_load =>
    let units_40A := 3
    let units_60A := 2
    let units_25A := 1
    let running_current_40A := 40
    let running_current_60A := 60
    let running_current_25A := 25
    let starting_multiplier_40A := 2
    let starting_multiplier_60A := 3
    let starting_multiplier_25A := 4
    let total_start_current_40A := units_40A * running_current_40A * starting_multiplier_40A
    let total_start_current_60A := units_60A * running_current_60A * starting_multiplier_60A
    let total_start_current_25A := units_25A * running_current_25A * starting_multiplier_25A
    total_load = total_start_current_40A + total_start_current_60A + total_start_current_25A

theorem transformer_current_load_is_700A : transformer_current_load 700 := by
  sorry

end NUMINAMATH_CALUDE_transformer_current_load_transformer_current_load_is_700A_l2555_255500


namespace NUMINAMATH_CALUDE_inequality_solution_set_inequality_with_conditions_l2555_255583

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for the solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | f x + f (x + 4) ≥ 8} = {x : ℝ | x ≤ -5 ∨ x ≥ 3} := by sorry

-- Theorem for the inequality with conditions
theorem inequality_with_conditions (a b : ℝ) 
  (ha : |a| < 1) (hb : |b| < 1) (ha_neq_zero : a ≠ 0) :
  f (a * b) > |a| * f (b / a) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_inequality_with_conditions_l2555_255583


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2555_255546

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = 4*x
def line (x y : ℝ) : Prop := y = 2*x - 4

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the length of chord AB
def chordLength : ℝ := sorry

-- Define point P on the parabola
def P : ℝ × ℝ := sorry

-- Define the area of triangle ABP
def triangleArea : ℝ := sorry

theorem parabola_line_intersection :
  (parabola A.1 A.2 ∧ line A.1 A.2) ∧
  (parabola B.1 B.2 ∧ line B.1 B.2) ∧
  chordLength = 3 * Real.sqrt 5 ∧
  parabola P.1 P.2 ∧
  triangleArea = 12 →
  (P = (9, 6) ∨ P = (4, -4)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2555_255546


namespace NUMINAMATH_CALUDE_first_year_after_2000_with_digit_sum_15_l2555_255510

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2000 ∧ sum_of_digits year = 15

theorem first_year_after_2000_with_digit_sum_15 :
  ∀ year : ℕ, is_valid_year year → year ≥ 2049 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2000_with_digit_sum_15_l2555_255510


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l2555_255503

/-- Calculates the total wet surface area of a rectangular cistern -/
def totalWetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem: The total wet surface area of a cistern with given dimensions -/
theorem cistern_wet_surface_area :
  let length : ℝ := 9
  let width : ℝ := 4
  let depth : ℝ := 1.25
  totalWetSurfaceArea length width depth = 68.5 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l2555_255503


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2555_255568

theorem sphere_volume_equals_surface_area (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3) = (4 * Real.pi * r^2) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2555_255568


namespace NUMINAMATH_CALUDE_triangle_inequality_l2555_255563

/-- Given a triangle with sides a, b, and c, and s = (a+b+c)/2, 
    if s^2 = 2ab, then s < 2a -/
theorem triangle_inequality (a b c s : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_s_def : s = (a + b + c) / 2)
  (h_s_sq : s^2 = 2*a*b) : 
  s < 2*a := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2555_255563


namespace NUMINAMATH_CALUDE_polynomial_division_existence_l2555_255552

theorem polynomial_division_existence :
  ∃ (Q R : Polynomial ℚ),
    4 * X^5 - 7 * X^4 + 3 * X^3 + 9 * X^2 - 23 * X + 8 = (5 * X^2 + 2 * X - 1) * Q + R ∧
    R.degree < (5 * X^2 + 2 * X - 1).degree := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_existence_l2555_255552


namespace NUMINAMATH_CALUDE_min_distinct_values_l2555_255550

/-- A list of positive integers -/
def IntegerList := List ℕ+

/-- The number of occurrences of the most frequent element in a list -/
def modeCount (l : IntegerList) : ℕ := sorry

/-- The number of distinct elements in a list -/
def distinctCount (l : IntegerList) : ℕ := sorry

/-- Theorem: Minimum number of distinct values in a list of 4000 positive integers
    with a unique mode occurring exactly 20 times is 211 -/
theorem min_distinct_values (l : IntegerList) 
  (h1 : l.length = 4000)
  (h2 : ∃! x, modeCount l = x)
  (h3 : modeCount l = 20) :
  distinctCount l ≥ 211 := by sorry

end NUMINAMATH_CALUDE_min_distinct_values_l2555_255550


namespace NUMINAMATH_CALUDE_percentage_boys_playing_soccer_is_86_percent_l2555_255551

/-- Calculates the percentage of boys among students playing soccer -/
def percentage_boys_playing_soccer (total_students : ℕ) (num_boys : ℕ) (students_playing_soccer : ℕ) (girls_not_playing_soccer : ℕ) : ℚ :=
  let total_girls : ℕ := total_students - num_boys
  let girls_playing_soccer : ℕ := total_girls - girls_not_playing_soccer
  let boys_playing_soccer : ℕ := students_playing_soccer - girls_playing_soccer
  (boys_playing_soccer : ℚ) / (students_playing_soccer : ℚ) * 100

/-- Theorem stating that the percentage of boys playing soccer is 86% -/
theorem percentage_boys_playing_soccer_is_86_percent :
  percentage_boys_playing_soccer 420 312 250 73 = 86 := by
  sorry

end NUMINAMATH_CALUDE_percentage_boys_playing_soccer_is_86_percent_l2555_255551


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2555_255548

theorem right_triangle_side_length 
  (area : ℝ) 
  (side1 : ℝ) 
  (is_right_triangle : Bool) 
  (h1 : area = 8) 
  (h2 : side1 = Real.sqrt 10) 
  (h3 : is_right_triangle = true) : 
  ∃ side2 : ℝ, side2 = 1.6 * Real.sqrt 10 ∧ (1/2) * side1 * side2 = area :=
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2555_255548


namespace NUMINAMATH_CALUDE_angle_bisector_length_l2555_255514

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let pr := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  pq = 8 ∧ qr = 15 ∧ pr = 17

-- Define the angle bisector QS
def AngleBisector (P Q R S : ℝ × ℝ) : Prop :=
  let ps := Real.sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2)
  let rs := Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2)
  ps / rs = 8 / 15

-- Theorem statement
theorem angle_bisector_length (P Q R S : ℝ × ℝ) :
  Triangle P Q R → AngleBisector P Q R S →
  Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) = 4 * Real.sqrt 3272 / 23 :=
by sorry


end NUMINAMATH_CALUDE_angle_bisector_length_l2555_255514


namespace NUMINAMATH_CALUDE_average_age_of_boys_l2555_255502

def boys_ages (x : ℝ) : Fin 3 → ℝ
| 0 => 3 * x
| 1 => 5 * x
| 2 => 7 * x

theorem average_age_of_boys (x : ℝ) (h1 : boys_ages x 2 = 21) :
  (boys_ages x 0 + boys_ages x 1 + boys_ages x 2) / 3 = 15 := by
  sorry

#check average_age_of_boys

end NUMINAMATH_CALUDE_average_age_of_boys_l2555_255502


namespace NUMINAMATH_CALUDE_max_value_expression_l2555_255549

theorem max_value_expression (x y : ℝ) : 
  (2 * x + Real.sqrt 2 * y) / (2 * x^4 + 4 * y^4 + 9) ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2555_255549


namespace NUMINAMATH_CALUDE_monotonicity_condition_l2555_255543

/-- Represents a voting system with n voters and m candidates. -/
structure VotingSystem where
  n : ℕ  -- number of voters
  m : ℕ  -- number of candidates
  k : ℕ  -- number of top choices each voter selects

/-- Represents a poll profile (arrangement of candidate rankings by voters). -/
def PollProfile (vs : VotingSystem) := Fin vs.n → (Fin vs.m → Fin vs.m)

/-- Determines if a candidate is a winner in a given poll profile. -/
def isWinner (vs : VotingSystem) (profile : PollProfile vs) (candidate : Fin vs.m) : Prop :=
  sorry

/-- Determines if one profile is a-good compared to another. -/
def isAGood (vs : VotingSystem) (a : Fin vs.m) (R R' : PollProfile vs) : Prop :=
  ∀ (voter : Fin vs.n) (candidate : Fin vs.m),
    (R voter candidate > R voter a) → (R' voter candidate > R' voter a)

/-- Defines the monotonicity property for a voting system. -/
def isMonotone (vs : VotingSystem) : Prop :=
  ∀ (R R' : PollProfile vs) (a : Fin vs.m),
    isWinner vs R a → isAGood vs a R R' → isWinner vs R' a

/-- The main theorem stating the condition for monotonicity. -/
theorem monotonicity_condition (vs : VotingSystem) :
  isMonotone vs ↔ vs.k > (vs.m * (vs.n - 1)) / vs.n :=
  sorry

end NUMINAMATH_CALUDE_monotonicity_condition_l2555_255543


namespace NUMINAMATH_CALUDE_workshop_attendees_count_l2555_255592

/-- Calculates the total number of people at a workshop given the number of novelists and the ratio of novelists to poets -/
def total_workshop_attendees (num_novelists : ℕ) (novelist_ratio : ℕ) (poet_ratio : ℕ) : ℕ :=
  num_novelists + (num_novelists * poet_ratio) / novelist_ratio

/-- Theorem stating that for a workshop with 15 novelists and a 5:3 ratio of novelists to poets, there are 24 people in total -/
theorem workshop_attendees_count :
  total_workshop_attendees 15 5 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_workshop_attendees_count_l2555_255592


namespace NUMINAMATH_CALUDE_sebastian_ticket_cost_l2555_255545

/-- The total cost of tickets for Sebastian and his parents -/
def total_cost (num_people : ℕ) (ticket_price : ℕ) (service_fee : ℕ) : ℕ :=
  num_people * ticket_price + service_fee

/-- Theorem stating that the total cost for Sebastian's tickets is $150 -/
theorem sebastian_ticket_cost :
  total_cost 3 44 18 = 150 := by
  sorry

end NUMINAMATH_CALUDE_sebastian_ticket_cost_l2555_255545


namespace NUMINAMATH_CALUDE_same_color_probability_l2555_255594

/-- The number of red plates -/
def red_plates : ℕ := 6

/-- The number of blue plates -/
def blue_plates : ℕ := 5

/-- The number of green plates -/
def green_plates : ℕ := 3

/-- The total number of plates -/
def total_plates : ℕ := red_plates + blue_plates + green_plates

/-- The number of ways to choose 3 plates from the total number of plates -/
def total_ways : ℕ := Nat.choose total_plates 3

/-- The number of ways to choose 3 red plates -/
def red_ways : ℕ := Nat.choose red_plates 3

/-- The number of ways to choose 3 blue plates -/
def blue_ways : ℕ := Nat.choose blue_plates 3

/-- The number of ways to choose 3 green plates -/
def green_ways : ℕ := Nat.choose green_plates 3

/-- The total number of favorable outcomes (all same color) -/
def favorable_outcomes : ℕ := red_ways + blue_ways + green_ways

/-- The probability of selecting three plates of the same color -/
theorem same_color_probability : 
  (favorable_outcomes : ℚ) / total_ways = 31 / 364 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2555_255594


namespace NUMINAMATH_CALUDE_line_shift_l2555_255515

/-- The vertical shift of a line -/
def vertical_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := fun x ↦ f x + shift

/-- The original line equation -/
def original_line : ℝ → ℝ := fun x ↦ 3 * x - 2

/-- Theorem: Moving the line y = 3x - 2 up by 6 units results in y = 3x + 4 -/
theorem line_shift :
  vertical_shift original_line 6 = fun x ↦ 3 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_line_shift_l2555_255515


namespace NUMINAMATH_CALUDE_min_value_inequality_l2555_255596

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (1 / x + 4 / y) ≥ 9/4 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2555_255596


namespace NUMINAMATH_CALUDE_mean_calculation_l2555_255562

theorem mean_calculation (x : ℝ) : 
  (28 + x + 42 + 78 + 104) / 5 = 62 → 
  (48 + 62 + 98 + 124 + x) / 5 = 78 := by
sorry

end NUMINAMATH_CALUDE_mean_calculation_l2555_255562


namespace NUMINAMATH_CALUDE_inequality_property_l2555_255581

theorem inequality_property (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l2555_255581


namespace NUMINAMATH_CALUDE_unique_triple_solution_l2555_255539

theorem unique_triple_solution :
  ∃! (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y = 3 ∧ x * y - z^2 = 4 ∧ x = 1 ∧ y = 2 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l2555_255539


namespace NUMINAMATH_CALUDE_speaking_orders_eq_552_l2555_255570

/-- The number of students in the class -/
def total_students : ℕ := 7

/-- The number of students to be selected for speaking -/
def speakers : ℕ := 4

/-- Function to calculate the number of different speaking orders -/
def speaking_orders : ℕ :=
  let only_one_ab := 2 * (total_students - 2).choose (speakers - 1) * (speakers).factorial
  let both_ab := (total_students - 3).choose (speakers - 2) * 2 * 6
  only_one_ab + both_ab

/-- Theorem stating that the number of different speaking orders is 552 -/
theorem speaking_orders_eq_552 : speaking_orders = 552 := by
  sorry

end NUMINAMATH_CALUDE_speaking_orders_eq_552_l2555_255570


namespace NUMINAMATH_CALUDE_equation_solution_l2555_255525

theorem equation_solution (x : ℝ) : 
  x ≠ 3 ∧ x ≠ -3 → (4 / (x^2 - 9) - x / (3 - x) = 1 ↔ x = -13/3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2555_255525


namespace NUMINAMATH_CALUDE_apple_distribution_l2555_255593

theorem apple_distribution (x y : ℕ) : 
  (y - 1 = x + 1) →
  (y + 1 = 3 * (x - 1)) →
  (x = 3 ∧ y = 5) :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l2555_255593


namespace NUMINAMATH_CALUDE_max_stamps_purchased_l2555_255573

/-- Given a stamp price of 45 cents and $50 to spend, 
    the maximum number of stamps that can be purchased is 111. -/
theorem max_stamps_purchased (stamp_price : ℕ) (budget : ℕ) : 
  stamp_price = 45 → budget = 5000 → 
  (∀ n : ℕ, n * stamp_price ≤ budget → n ≤ 111) ∧ 
  111 * stamp_price ≤ budget :=
by sorry

end NUMINAMATH_CALUDE_max_stamps_purchased_l2555_255573


namespace NUMINAMATH_CALUDE_cricket_players_count_l2555_255501

theorem cricket_players_count (total_players hockey_players football_players softball_players : ℕ) 
  (h1 : total_players = 77)
  (h2 : hockey_players = 15)
  (h3 : football_players = 21)
  (h4 : softball_players = 19) :
  total_players - (hockey_players + football_players + softball_players) = 22 := by
sorry

end NUMINAMATH_CALUDE_cricket_players_count_l2555_255501


namespace NUMINAMATH_CALUDE_count_solutions_quadratic_congruence_l2555_255522

theorem count_solutions_quadratic_congruence (p : Nat) (a : Int) 
  (h_p : p.Prime ∧ p > 2) :
  let S := {(x, y) : Fin p × Fin p | (x.val^2 + y.val^2) % p = a % p}
  Fintype.card S = p + 1 := by
sorry

end NUMINAMATH_CALUDE_count_solutions_quadratic_congruence_l2555_255522


namespace NUMINAMATH_CALUDE_zoo_trip_short_amount_l2555_255560

/-- Represents the zoo trip expenses and budget for two people -/
structure ZooTrip where
  total_budget : ℕ
  zoo_entry_cost : ℕ
  aquarium_entry_cost : ℕ
  animal_show_cost : ℕ
  bus_fare : ℕ
  num_transfers : ℕ
  souvenir_budget : ℕ
  noah_lunch_cost : ℕ
  ava_lunch_cost : ℕ
  beverage_cost : ℕ
  num_people : ℕ

/-- Calculates the amount short for lunch and snacks -/
def amount_short (trip : ZooTrip) : ℕ :=
  let total_entry_cost := (trip.zoo_entry_cost + trip.aquarium_entry_cost + trip.animal_show_cost) * trip.num_people
  let total_bus_fare := trip.bus_fare * trip.num_transfers * trip.num_people
  let total_lunch_cost := trip.noah_lunch_cost + trip.ava_lunch_cost
  let total_beverage_cost := trip.beverage_cost * trip.num_people
  let total_expenses := total_entry_cost + total_bus_fare + trip.souvenir_budget + total_lunch_cost + total_beverage_cost
  total_expenses - trip.total_budget

/-- Theorem stating that the amount short for lunch and snacks is $12 -/
theorem zoo_trip_short_amount (trip : ZooTrip) 
  (h1 : trip.total_budget = 100)
  (h2 : trip.zoo_entry_cost = 5)
  (h3 : trip.aquarium_entry_cost = 7)
  (h4 : trip.animal_show_cost = 4)
  (h5 : trip.bus_fare = 150) -- Using cents for precise integer arithmetic
  (h6 : trip.num_transfers = 4)
  (h7 : trip.souvenir_budget = 20)
  (h8 : trip.noah_lunch_cost = 10)
  (h9 : trip.ava_lunch_cost = 8)
  (h10 : trip.beverage_cost = 3)
  (h11 : trip.num_people = 2) :
  amount_short trip = 12 := by
  sorry


end NUMINAMATH_CALUDE_zoo_trip_short_amount_l2555_255560


namespace NUMINAMATH_CALUDE_geometric_sum_eight_terms_l2555_255582

theorem geometric_sum_eight_terms : 
  let a : ℕ := 2
  let r : ℕ := 2
  let n : ℕ := 8
  a * (r^n - 1) / (r - 1) = 510 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_eight_terms_l2555_255582


namespace NUMINAMATH_CALUDE_log_equation_solution_l2555_255575

-- Define the logarithm function for base 2
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem log_equation_solution :
  ∃ x : ℝ, log2 (x + 3) + 2 * log2 5 = 4 ∧ x = -59 / 25 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2555_255575


namespace NUMINAMATH_CALUDE_rational_abs_four_and_self_reciprocal_l2555_255571

theorem rational_abs_four_and_self_reciprocal :
  (∀ x : ℚ, |x| = 4 ↔ x = -4 ∨ x = 4) ∧
  (∀ x : ℝ, x⁻¹ = x ↔ x = -1 ∨ x = 1) := by sorry

end NUMINAMATH_CALUDE_rational_abs_four_and_self_reciprocal_l2555_255571


namespace NUMINAMATH_CALUDE_exponent_multiplication_and_zero_power_l2555_255523

theorem exponent_multiplication_and_zero_power :
  (∀ x : ℝ, x^2 * x^4 = x^6) ∧ ((-5^2)^0 = 1) := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_and_zero_power_l2555_255523


namespace NUMINAMATH_CALUDE_quadratic_roots_l2555_255567

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = 1 ∧ 
  (∀ x : ℝ, x^2 = x ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2555_255567


namespace NUMINAMATH_CALUDE_prime_pairs_divisibility_l2555_255585

theorem prime_pairs_divisibility (p q : ℕ) : 
  Prime p ∧ Prime q ∧ 
  (p^2 ∣ q^3 + 1) ∧ 
  (q^2 ∣ p^6 - 1) ↔ 
  ((p = 3 ∧ q = 2) ∨ (p = 2 ∧ q = 3)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_divisibility_l2555_255585


namespace NUMINAMATH_CALUDE_garage_cars_count_l2555_255537

/-- The number of cars in Connor's garage -/
def num_cars : ℕ := 10

/-- The number of bicycles in the garage -/
def num_bicycles : ℕ := 20

/-- The number of motorcycles in the garage -/
def num_motorcycles : ℕ := 5

/-- The total number of wheels in the garage -/
def total_wheels : ℕ := 90

/-- The number of wheels on a bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a car -/
def wheels_per_car : ℕ := 4

/-- The number of wheels on a motorcycle -/
def wheels_per_motorcycle : ℕ := 2

theorem garage_cars_count :
  num_bicycles * wheels_per_bicycle +
  num_cars * wheels_per_car +
  num_motorcycles * wheels_per_motorcycle = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_garage_cars_count_l2555_255537


namespace NUMINAMATH_CALUDE_triangle_properties_l2555_255569

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Law of Sines -/
axiom law_of_sines (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- Law of Cosines -/
axiom law_of_cosines (t : Triangle) : t.b ^ 2 = t.a ^ 2 + t.c ^ 2 - 2 * t.a * t.c * Real.cos t.B

theorem triangle_properties (t : Triangle) (h1 : t.a = 2) (h2 : Real.cos t.B = 3/5) :
  (t.b = 4 → Real.sin t.A = 2/5) ∧
  (area t = 4 → t.b = Real.sqrt 17 ∧ t.c = 5) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2555_255569


namespace NUMINAMATH_CALUDE_sqrt_eighteen_div_sqrt_two_eq_three_l2555_255579

theorem sqrt_eighteen_div_sqrt_two_eq_three : 
  Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eighteen_div_sqrt_two_eq_three_l2555_255579


namespace NUMINAMATH_CALUDE_solution_exists_l2555_255590

theorem solution_exists : ∃ (v : ℝ), 4 * v^2 = 144 ∧ v = 6 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l2555_255590


namespace NUMINAMATH_CALUDE_probability_through_C_and_D_l2555_255584

/-- Represents the number of eastward and southward moves between two intersections -/
structure Moves where
  east : Nat
  south : Nat

/-- Calculates the number of possible paths given a number of eastward and southward moves -/
def pathCount (m : Moves) : Nat :=
  Nat.choose (m.east + m.south) m.east

/-- The moves from A to C -/
def movesAC : Moves := ⟨3, 2⟩

/-- The moves from C to D -/
def movesCD : Moves := ⟨2, 1⟩

/-- The moves from D to B -/
def movesDB : Moves := ⟨1, 2⟩

/-- The total moves from A to B -/
def movesAB : Moves := ⟨movesAC.east + movesCD.east + movesDB.east, movesAC.south + movesCD.south + movesDB.south⟩

/-- The probability of choosing a specific path at each intersection -/
def pathProbability (m : Moves) : Rat :=
  1 / (2 ^ (m.east + m.south))

theorem probability_through_C_and_D :
  (pathCount movesAC * pathCount movesCD * pathCount movesDB : Rat) /
  (pathCount movesAB : Rat) = 15 / 77 := by sorry

end NUMINAMATH_CALUDE_probability_through_C_and_D_l2555_255584


namespace NUMINAMATH_CALUDE_unique_factorial_sum_l2555_255518

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem unique_factorial_sum (x a b c d : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < x)
  (h6 : a ≤ b) (h7 : b ≤ c) (h8 : c ≤ d) (h9 : d < x)
  (h10 : factorial x = factorial a + factorial b + factorial c + factorial d) :
  x = 4 ∧ a = 3 ∧ b = 3 ∧ c = 3 ∧ d = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_factorial_sum_l2555_255518


namespace NUMINAMATH_CALUDE_max_people_served_l2555_255521

theorem max_people_served (total_budget : ℚ) (min_food_spend : ℚ) (cheapest_food_cost : ℚ) (cheapest_drink_cost : ℚ) 
  (h1 : total_budget = 12.5)
  (h2 : min_food_spend = 10)
  (h3 : cheapest_food_cost = 0.6)
  (h4 : cheapest_drink_cost = 0.5) :
  ∃ (n : ℕ), n = 10 ∧ 
    n * (cheapest_food_cost + cheapest_drink_cost) ≤ total_budget ∧
    n * cheapest_food_cost ≥ min_food_spend ∧
    ∀ (m : ℕ), m > n → 
      m * (cheapest_food_cost + cheapest_drink_cost) > total_budget ∨
      m * cheapest_food_cost < min_food_spend :=
by
  sorry

#check max_people_served

end NUMINAMATH_CALUDE_max_people_served_l2555_255521


namespace NUMINAMATH_CALUDE_min_green_fraction_of_4x4x4_cube_l2555_255542

/-- Represents a cube with colored unit cubes -/
structure ColoredCube where
  edge_length : ℕ
  total_cubes : ℕ
  blue_cubes : ℕ
  green_cubes : ℕ

/-- Calculates the surface area of a cube -/
def surface_area (c : ColoredCube) : ℕ := 6 * c.edge_length^2

/-- Calculates the minimum visible green surface area -/
def min_green_surface_area (c : ColoredCube) : ℕ := c.green_cubes - 4

theorem min_green_fraction_of_4x4x4_cube (c : ColoredCube) 
  (h1 : c.edge_length = 4)
  (h2 : c.total_cubes = 64)
  (h3 : c.blue_cubes = 56)
  (h4 : c.green_cubes = 8) :
  (min_green_surface_area c : ℚ) / (surface_area c : ℚ) = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_min_green_fraction_of_4x4x4_cube_l2555_255542


namespace NUMINAMATH_CALUDE_max_m_value_l2555_255541

def M (m : ℝ) : Set (ℝ × ℝ) := {p | p.1 ≤ -1 ∧ p.2 ≤ m}

theorem max_m_value :
  ∃ m : ℝ, m = 1 ∧
  (∀ m' : ℝ, (∀ p ∈ M m', p.1 * 2^p.2 - p.2 - 3*p.1 ≥ 0) →
  m' ≤ m) ∧
  (∀ p ∈ M m, p.1 * 2^p.2 - p.2 - 3*p.1 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l2555_255541


namespace NUMINAMATH_CALUDE_peggy_final_doll_count_l2555_255528

/-- Calculates the final number of dolls Peggy has -/
def peggy_dolls (initial : ℕ) (grandmother_gift : ℕ) : ℕ :=
  initial + grandmother_gift + (grandmother_gift / 2)

/-- Theorem stating that Peggy's final doll count is 51 -/
theorem peggy_final_doll_count :
  peggy_dolls 6 30 = 51 := by
  sorry

end NUMINAMATH_CALUDE_peggy_final_doll_count_l2555_255528


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l2555_255512

theorem arithmetic_to_geometric_sequence (a₁ a₂ a₃ a₄ d : ℝ) : 
  d ≠ 0 ∧ 
  a₂ = a₁ + d ∧ 
  a₃ = a₁ + 2*d ∧ 
  a₄ = a₁ + 3*d ∧
  ((a₁*a₃ = a₂^2) ∨ (a₁*a₄ = a₂^2) ∨ (a₁*a₄ = a₃^2) ∨ (a₂*a₄ = a₃^2)) →
  a₁/d = 1 ∨ a₁/d = -4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l2555_255512


namespace NUMINAMATH_CALUDE_total_blue_balloons_l2555_255588

theorem total_blue_balloons (joan sally jessica : ℕ) 
  (h1 : joan = 9) 
  (h2 : sally = 5) 
  (h3 : jessica = 2) : 
  joan + sally + jessica = 16 := by
sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l2555_255588


namespace NUMINAMATH_CALUDE_wall_width_is_eight_l2555_255524

/-- Proves that the width of a wall with given proportions and volume is 8 meters -/
theorem wall_width_is_eight (w h l : ℝ) (h_height : h = 6 * w) (h_length : l = 7 * h) (h_volume : w * h * l = 129024) :
  w = 8 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_is_eight_l2555_255524


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2555_255554

/-- A quadratic equation in terms of x is a polynomial equation of degree 2 in x. -/
def IsQuadraticInX (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing the equation x(x-1) = 0 -/
def f (x : ℝ) : ℝ := x * (x - 1)

/-- Theorem stating that f is a quadratic equation in terms of x -/
theorem f_is_quadratic : IsQuadraticInX f := by sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2555_255554


namespace NUMINAMATH_CALUDE_abs_frac_gt_three_iff_x_in_intervals_l2555_255599

theorem abs_frac_gt_three_iff_x_in_intervals (x : ℝ) :
  x ≠ 2 →
  (|(3 * x - 2) / (x - 2)| > 3) ↔ (x > 4/3 ∧ x < 2) ∨ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_abs_frac_gt_three_iff_x_in_intervals_l2555_255599


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2555_255559

variable {n : Type*} [DecidableEq n] [Fintype n]

theorem matrix_equation_solution 
  (B : Matrix n n ℝ) 
  (h_inv : Invertible B) 
  (h_eq : (B - 3 • (1 : Matrix n n ℝ)) * (B - 5 • (1 : Matrix n n ℝ)) = 0) :
  B + 9 • B⁻¹ = 8 • (1 : Matrix n n ℝ) := by
sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2555_255559


namespace NUMINAMATH_CALUDE_weight_problem_l2555_255540

/-- Given the average weights of three people and two pairs, prove the weight of one person. -/
theorem weight_problem (A B C : ℝ) : 
  (A + B + C) / 3 = 45 ∧ 
  (A + B) / 2 = 40 ∧ 
  (B + C) / 2 = 43 → 
  B = 31 := by
  sorry

end NUMINAMATH_CALUDE_weight_problem_l2555_255540


namespace NUMINAMATH_CALUDE_waiter_earnings_l2555_255580

def lunch_shift (total_customers : ℕ) (tipping_customers : ℕ) 
  (tip_8 : ℕ) (tip_10 : ℕ) (tip_12 : ℕ) (meal_cost : ℕ) : ℕ :=
  let total_tips := 8 * tip_8 + 10 * tip_10 + 12 * tip_12
  total_tips - meal_cost

theorem waiter_earnings : 
  lunch_shift 12 6 3 2 1 5 = 51 := by sorry

end NUMINAMATH_CALUDE_waiter_earnings_l2555_255580


namespace NUMINAMATH_CALUDE_line_is_intersection_l2555_255556

/-- The line of intersection of two planes -/
def line_of_intersection (p₁ p₂ : ℝ → ℝ → ℝ → Prop) : ℝ → ℝ → ℝ → Prop :=
  λ x y z => (x + 3) / (-3) = y / (-4) ∧ y / (-4) = z / (-9)

/-- First plane equation -/
def plane1 : ℝ → ℝ → ℝ → Prop :=
  λ x y z => 2*x + 3*y - 2*z + 6 = 0

/-- Second plane equation -/
def plane2 : ℝ → ℝ → ℝ → Prop :=
  λ x y z => x - 3*y + z + 3 = 0

/-- Theorem stating that the line is the intersection of the two planes -/
theorem line_is_intersection :
  ∀ x y z, line_of_intersection plane1 plane2 x y z ↔ (plane1 x y z ∧ plane2 x y z) :=
sorry

end NUMINAMATH_CALUDE_line_is_intersection_l2555_255556


namespace NUMINAMATH_CALUDE_inequality_proof_l2555_255511

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / a) + (4 / b) ≥ 9 / (a + b) := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2555_255511


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2555_255576

theorem inverse_proportion_problem (x y : ℝ) (C : ℝ) :
  (x * y = C) →  -- x and y are inversely proportional
  (x + y = 32) →
  (x - y = 8) →
  (x = 4) →
  y = 60 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2555_255576


namespace NUMINAMATH_CALUDE_average_of_multiples_of_seven_l2555_255505

theorem average_of_multiples_of_seven (n : ℕ) : 
  (n / 2 : ℚ) * (7 + 7 * n) / n = 77 → n = 21 := by
  sorry

end NUMINAMATH_CALUDE_average_of_multiples_of_seven_l2555_255505


namespace NUMINAMATH_CALUDE_N2O_molecular_weight_l2555_255520

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of N2O in g/mol -/
def molecular_weight_N2O : ℝ := 2 * atomic_weight_N + atomic_weight_O

/-- Theorem stating that the molecular weight of N2O is 44.02 g/mol -/
theorem N2O_molecular_weight : molecular_weight_N2O = 44.02 := by
  sorry

end NUMINAMATH_CALUDE_N2O_molecular_weight_l2555_255520


namespace NUMINAMATH_CALUDE_grid_cutting_ways_l2555_255538

-- Define the shape of the grid
def GridShape : Type := Unit  -- Placeholder for the specific grid shape

-- Define the property of being cuttable into 1×2 rectangles
def IsCuttableInto1x2Rectangles (g : GridShape) : Prop := sorry

-- Define the function that counts the number of ways to cut the grid
def NumberOfWaysToCut (g : GridShape) : ℕ := sorry

-- The main theorem
theorem grid_cutting_ways (g : GridShape) : 
  IsCuttableInto1x2Rectangles g → NumberOfWaysToCut g = 27 := by sorry

end NUMINAMATH_CALUDE_grid_cutting_ways_l2555_255538


namespace NUMINAMATH_CALUDE_arianna_work_hours_l2555_255507

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Represents the number of hours Arianna spends on chores -/
def hours_on_chores : ℕ := 5

/-- Represents the number of hours Arianna spends sleeping -/
def hours_sleeping : ℕ := 13

/-- Theorem stating that Arianna spends 6 hours at work -/
theorem arianna_work_hours :
  hours_in_day - (hours_on_chores + hours_sleeping) = 6 := by
  sorry

end NUMINAMATH_CALUDE_arianna_work_hours_l2555_255507


namespace NUMINAMATH_CALUDE_minimum_parents_needed_tour_parents_theorem_l2555_255574

theorem minimum_parents_needed (num_children : ℕ) (car_capacity : ℕ) : ℕ :=
  let total_people := num_children
  let cars_needed := (total_people + car_capacity - 1) / car_capacity
  cars_needed

theorem tour_parents_theorem :
  minimum_parents_needed 50 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_minimum_parents_needed_tour_parents_theorem_l2555_255574


namespace NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_l2555_255526

-- Equation 1: x^2 - 6x + 1 = 0
theorem solve_equation_1 : 
  ∃ x₁ x₂ : ℝ, x₁ = 3 + 2 * Real.sqrt 2 ∧ 
             x₂ = 3 - 2 * Real.sqrt 2 ∧ 
             x₁^2 - 6*x₁ + 1 = 0 ∧ 
             x₂^2 - 6*x₂ + 1 = 0 := by
  sorry

-- Equation 2: 2x^2 + 3x - 5 = 0
theorem solve_equation_2 : 
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ 
             x₂ = -5/2 ∧ 
             2*x₁^2 + 3*x₁ - 5 = 0 ∧ 
             2*x₂^2 + 3*x₂ - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_l2555_255526


namespace NUMINAMATH_CALUDE_work_completion_rate_l2555_255513

theorem work_completion_rate (a_days : ℕ) (b_days : ℕ) : 
  a_days = 8 → b_days = a_days / 2 → (1 : ℚ) / a_days + (1 : ℚ) / b_days = 3 / 8 := by
  sorry

#check work_completion_rate

end NUMINAMATH_CALUDE_work_completion_rate_l2555_255513


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2555_255577

theorem greatest_divisor_with_remainders (n : ℕ) : 
  (1657 % n = 10 ∧ 2037 % n = 7) → n = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2555_255577


namespace NUMINAMATH_CALUDE_ellipse_focus_coincides_with_center_l2555_255504

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- Returns the focus with larger x-coordinate for an ellipse -/
def focus_with_larger_x (e : Ellipse) : Point :=
  e.center

theorem ellipse_focus_coincides_with_center (e : Ellipse) 
    (h1 : e.center = ⟨3, -2⟩)
    (h2 : e.semi_major_axis = 3)
    (h3 : e.semi_minor_axis = 3) :
  focus_with_larger_x e = ⟨3, -2⟩ := by
  sorry

#check ellipse_focus_coincides_with_center

end NUMINAMATH_CALUDE_ellipse_focus_coincides_with_center_l2555_255504


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2555_255532

theorem complex_modulus_problem (z : ℂ) : (1 - Complex.I) * z = 2 * Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2555_255532


namespace NUMINAMATH_CALUDE_power_of_25_equals_power_of_5_l2555_255553

theorem power_of_25_equals_power_of_5 : (25 : ℕ) ^ 5 = 5 ^ 10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_25_equals_power_of_5_l2555_255553


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2555_255544

def consecutive_integers (n : ℕ) (start : ℤ) : List ℤ :=
  List.range n |>.map (λ i => start + i)

theorem largest_divisor_five_consecutive_integers :
  ∀ start : ℤ, 
  ∃ m : ℕ, m = 240 ∧ 
  (m : ℤ) ∣ (List.prod (consecutive_integers 5 start)) ∧
  ∀ k : ℕ, k > m → ¬((k : ℤ) ∣ (List.prod (consecutive_integers 5 start))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2555_255544


namespace NUMINAMATH_CALUDE_smallest_quotient_l2555_255555

/-- Represents a three-digit number with different non-zero digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_nonzero : hundreds ≠ 0
  t_nonzero : tens ≠ 0
  o_nonzero : ones ≠ 0
  h_lt_ten : hundreds < 10
  t_lt_ten : tens < 10
  o_lt_ten : ones < 10
  all_different : hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

/-- The value of a ThreeDigitNumber -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The sum of digits of a ThreeDigitNumber -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

/-- The quotient of a ThreeDigitNumber divided by its digit sum -/
def quotient (n : ThreeDigitNumber) : Rat :=
  (value n : Rat) / (digitSum n : Rat)

theorem smallest_quotient :
  ∃ (n : ThreeDigitNumber), ∀ (m : ThreeDigitNumber), quotient n ≤ quotient m ∧ quotient n = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_quotient_l2555_255555


namespace NUMINAMATH_CALUDE_tim_cell_phone_cost_l2555_255517

/-- Calculates the total cost of a cell phone plan -/
def calculate_total_cost (base_cost : ℝ) (text_cost : ℝ) (extra_minute_cost : ℝ) 
                         (free_hours : ℝ) (texts_sent : ℝ) (hours_talked : ℝ) : ℝ :=
  let text_total := text_cost * texts_sent
  let extra_minutes := (hours_talked - free_hours) * 60
  let extra_minute_total := extra_minute_cost * extra_minutes
  base_cost + text_total + extra_minute_total

theorem tim_cell_phone_cost :
  let base_cost : ℝ := 30
  let text_cost : ℝ := 0.04
  let extra_minute_cost : ℝ := 0.15
  let free_hours : ℝ := 40
  let texts_sent : ℝ := 200
  let hours_talked : ℝ := 42
  calculate_total_cost base_cost text_cost extra_minute_cost free_hours texts_sent hours_talked = 56 := by
  sorry


end NUMINAMATH_CALUDE_tim_cell_phone_cost_l2555_255517


namespace NUMINAMATH_CALUDE_only_valid_N_l2555_255572

theorem only_valid_N : 
  {N : ℕ+ | (∃ a b : ℕ, N = 2^a * 5^b) ∧ 
            (∃ k : ℕ, N + 25 = k^2)} = 
  {200, 2000} := by sorry

end NUMINAMATH_CALUDE_only_valid_N_l2555_255572


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2555_255597

theorem complex_equation_solution (a : ℝ) : (a + Complex.I) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2555_255597


namespace NUMINAMATH_CALUDE_triangle_sin_C_l2555_255598

theorem triangle_sin_C (a c : ℝ) (A : ℝ) :
  a = 7 →
  c = 3 →
  A = π / 3 →
  Real.sin (Real.arcsin ((c * Real.sin A) / a)) = 3 * Real.sqrt 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sin_C_l2555_255598


namespace NUMINAMATH_CALUDE_cow_count_is_18_l2555_255536

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs for a given AnimalCount -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads for a given AnimalCount -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem stating that the number of cows is 18 given the problem conditions -/
theorem cow_count_is_18 (count : AnimalCount) :
  totalLegs count = 2 * totalHeads count + 36 →
  count.cows = 18 := by
  sorry

#check cow_count_is_18

end NUMINAMATH_CALUDE_cow_count_is_18_l2555_255536


namespace NUMINAMATH_CALUDE_pen_and_pencil_cost_l2555_255529

theorem pen_and_pencil_cost (pencil_cost : ℝ) (pen_cost : ℝ) : 
  pencil_cost = 8 → pen_cost = pencil_cost / 2 → pencil_cost + pen_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_pen_and_pencil_cost_l2555_255529


namespace NUMINAMATH_CALUDE_largest_negative_congruent_to_one_mod_seventeen_l2555_255531

theorem largest_negative_congruent_to_one_mod_seventeen :
  ∃ (n : ℤ), 
    n = -1002 ∧ 
    n ≡ 1 [ZMOD 17] ∧ 
    n < 0 ∧ 
    -9999 ≤ n ∧
    ∀ (m : ℤ), m ≡ 1 [ZMOD 17] ∧ m < 0 ∧ -9999 ≤ m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_congruent_to_one_mod_seventeen_l2555_255531


namespace NUMINAMATH_CALUDE_min_value_sum_cubes_l2555_255565

/-- Given positive real numbers x and y satisfying x³ + y³ + 3xy = 1,
    the expression (x + 1/x)³ + (y + 1/y)³ has a minimum value of 125/4. -/
theorem min_value_sum_cubes (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : x^3 + y^3 + 3*x*y = 1) : 
    ∃ m : ℝ, m = 125/4 ∧ ∀ a b : ℝ, a > 0 → b > 0 → a^3 + b^3 + 3*a*b = 1 → 
    (a + 1/a)^3 + (b + 1/b)^3 ≥ m :=
  sorry

end NUMINAMATH_CALUDE_min_value_sum_cubes_l2555_255565


namespace NUMINAMATH_CALUDE_function_identity_l2555_255595

theorem function_identity (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f (x - 1/x) = x^2 + 1/x^2) →
  (∀ x : ℝ, f x = x^2 + 2) :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l2555_255595


namespace NUMINAMATH_CALUDE_parabola_transformation_l2555_255535

/-- A parabola is defined by its coefficient and horizontal shift -/
structure Parabola where
  a : ℝ
  h : ℝ

/-- The equation of a parabola y = a(x-h)^2 -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2

/-- The transformation that shifts a parabola horizontally -/
def horizontal_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, h := p.h + shift }

theorem parabola_transformation (p1 p2 : Parabola) :
  p1.a = 2 ∧ p1.h = 0 ∧ p2.a = 2 ∧ p2.h = 3 →
  ∃ (shift : ℝ), shift = 3 ∧ horizontal_shift p1 shift = p2 :=
sorry

end NUMINAMATH_CALUDE_parabola_transformation_l2555_255535
