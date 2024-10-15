import Mathlib

namespace NUMINAMATH_CALUDE_vending_machine_probability_l3072_307278

-- Define the number of toys
def num_toys : ℕ := 10

-- Define the cost range of toys
def min_cost : ℚ := 1/2
def max_cost : ℚ := 5

-- Define the cost increment
def cost_increment : ℚ := 1/2

-- Define Sam's initial quarters
def initial_quarters : ℕ := 10

-- Define the cost of Sam's favorite toy
def favorite_toy_cost : ℚ := 3

-- Define the function to calculate toy prices
def toy_price (n : ℕ) : ℚ := min_cost + (n - 1) * cost_increment

-- Define the probability of needing to break the twenty-dollar bill
def prob_break_bill : ℚ := 14/15

-- Theorem statement
theorem vending_machine_probability :
  (∀ n ∈ Finset.range num_toys, toy_price n ≤ max_cost) →
  (∀ n ∈ Finset.range num_toys, toy_price n ≥ min_cost) →
  (∀ n ∈ Finset.range (num_toys - 1), toy_price (n + 1) = toy_price n + cost_increment) →
  (favorite_toy_cost ∈ Finset.image toy_price (Finset.range num_toys)) →
  (initial_quarters * (1/4 : ℚ) < favorite_toy_cost) →
  (prob_break_bill = 14/15) :=
sorry

end NUMINAMATH_CALUDE_vending_machine_probability_l3072_307278


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3072_307267

theorem cube_volume_from_surface_area :
  ∀ s : ℝ,
  s > 0 →
  6 * s^2 = 864 →
  s^3 = 1728 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3072_307267


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3072_307225

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ≥ 5) →
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) ∧
  ¬(∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3072_307225


namespace NUMINAMATH_CALUDE_marks_age_multiple_l3072_307289

theorem marks_age_multiple (mark_current_age : ℕ) (aaron_current_age : ℕ) : 
  mark_current_age = 28 →
  mark_current_age - 3 = 3 * (aaron_current_age - 3) + 1 →
  ∃ x : ℕ, mark_current_age + 4 = x * (aaron_current_age + 4) + 2 →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_marks_age_multiple_l3072_307289


namespace NUMINAMATH_CALUDE_digital_earth_properties_digital_earth_properties_complete_l3072_307259

/-- Represents the concept of the digital Earth -/
structure DigitalEarth where
  /-- Geographic information technology is the foundation -/
  geoInfoTechFoundation : Prop
  /-- Ability to simulate reality -/
  simulatesReality : Prop
  /-- Manages Earth's information digitally through computer networks -/
  managesInfoDigitally : Prop
  /-- Method of information storage (centralized or not) -/
  centralizedStorage : Bool

/-- Theorem stating the correct properties of the digital Earth -/
theorem digital_earth_properties :
  ∀ (de : DigitalEarth),
    de.geoInfoTechFoundation ∧
    de.simulatesReality ∧
    de.managesInfoDigitally ∧
    ¬de.centralizedStorage :=
by
  sorry

/-- Theorem stating that these are the only correct properties -/
theorem digital_earth_properties_complete (de : DigitalEarth) :
  (de.geoInfoTechFoundation ∧ de.simulatesReality ∧ de.managesInfoDigitally ∧ ¬de.centralizedStorage) ↔
  (de.geoInfoTechFoundation ∧ de.simulatesReality ∧ de.managesInfoDigitally) :=
by
  sorry

end NUMINAMATH_CALUDE_digital_earth_properties_digital_earth_properties_complete_l3072_307259


namespace NUMINAMATH_CALUDE_child_ticket_cost_child_ticket_cost_is_6_l3072_307212

/-- Proves that the cost of a child's ticket is 6 dollars given the specified conditions -/
theorem child_ticket_cost (adult_ticket_cost : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (children_attending : ℕ) : ℕ :=
  let child_ticket_cost := (total_revenue - adult_ticket_cost * (total_tickets - children_attending)) / children_attending
  have h1 : adult_ticket_cost = 9 := by sorry
  have h2 : total_tickets = 225 := by sorry
  have h3 : total_revenue = 1875 := by sorry
  have h4 : children_attending = 50 := by sorry
  have h5 : child_ticket_cost * children_attending + adult_ticket_cost * (total_tickets - children_attending) = total_revenue := by sorry
  6

theorem child_ticket_cost_is_6 : child_ticket_cost 9 225 1875 50 = 6 := by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_child_ticket_cost_is_6_l3072_307212


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l3072_307213

theorem min_value_of_function (x : ℝ) (h : x > 0) : 4 * x + 9 / x ≥ 12 :=
sorry

theorem min_value_achieved : ∃ x : ℝ, x > 0 ∧ 4 * x + 9 / x = 12 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l3072_307213


namespace NUMINAMATH_CALUDE_inequality_not_holding_l3072_307202

theorem inequality_not_holding (x y : ℝ) (h : x > y) : ¬(-3*x > -3*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_holding_l3072_307202


namespace NUMINAMATH_CALUDE_binary_11010_equals_octal_32_l3072_307234

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_11010_equals_octal_32 :
  decimal_to_octal (binary_to_decimal [false, true, false, true, true]) = [3, 2] := by
  sorry

end NUMINAMATH_CALUDE_binary_11010_equals_octal_32_l3072_307234


namespace NUMINAMATH_CALUDE_complex_square_root_l3072_307288

theorem complex_square_root (p q : ℕ+) (h : (p + q * Complex.I) ^ 2 = 7 + 24 * Complex.I) :
  p + q * Complex.I = 4 + 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_root_l3072_307288


namespace NUMINAMATH_CALUDE_rectangular_distance_problem_l3072_307281

-- Define the rectangular distance function
def rectangular_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define points A, O, and B
def A : ℝ × ℝ := (-1, 3)
def O : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the line equation
def on_line (x y : ℝ) : Prop :=
  x - y + 2 = 0

theorem rectangular_distance_problem :
  (rectangular_distance A.1 A.2 O.1 O.2 = 4) ∧
  (∃ min_dist : ℝ, min_dist = 3 ∧
    ∀ x y : ℝ, on_line x y →
      rectangular_distance B.1 B.2 x y ≥ min_dist) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_distance_problem_l3072_307281


namespace NUMINAMATH_CALUDE_system_solution_l3072_307293

theorem system_solution (a b : ℝ) :
  ∃ (x y : ℝ), 
    (x + y = a ∧ Real.tan x * Real.tan y = b) ∧
    ((b ≠ 1 → 
      ∃ (k : ℤ), 
        x = (a + 2 * k * Real.pi + Real.arccos ((1 + b) / (1 - b) * Real.cos a)) / 2 ∧
        y = (a - 2 * k * Real.pi - Real.arccos ((1 + b) / (1 - b) * Real.cos a)) / 2) ∨
     (b ≠ 1 → 
      ∃ (k : ℤ), 
        x = (a + 2 * k * Real.pi - Real.arccos ((1 + b) / (1 - b) * Real.cos a)) / 2 ∧
        y = (a - 2 * k * Real.pi + Real.arccos ((1 + b) / (1 - b) * Real.cos a)) / 2)) ∧
    (b = 1 ∧ ∃ (k : ℤ), a = Real.pi / 2 + k * Real.pi → y = Real.pi / 2 + k * Real.pi - x) :=
by
  sorry


end NUMINAMATH_CALUDE_system_solution_l3072_307293


namespace NUMINAMATH_CALUDE_function_analysis_l3072_307207

def f (x : ℝ) := x^3 - 3*x^2 - 9*x + 11

theorem function_analysis :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f x ≤ f (-1)) ∧
  f (-1) = 16 ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (3 - ε) (3 + ε), f x ≥ f 3) ∧
  f 3 = -16 ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) 1, f x < f 1) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo 1 (1 + ε), f x > f 1) ∧
  f 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_function_analysis_l3072_307207


namespace NUMINAMATH_CALUDE_textbook_order_solution_l3072_307274

/-- Represents the textbook order problem -/
structure TextbookOrder where
  red_cost : ℝ
  trad_cost : ℝ
  red_price_ratio : ℝ
  quantity_diff : ℕ
  total_quantity : ℕ
  max_trad_quantity : ℕ
  max_total_cost : ℝ

/-- Theorem stating the solution to the textbook order problem -/
theorem textbook_order_solution (order : TextbookOrder)
  (h1 : order.red_cost = 14000)
  (h2 : order.trad_cost = 7000)
  (h3 : order.red_price_ratio = 1.4)
  (h4 : order.quantity_diff = 300)
  (h5 : order.total_quantity = 1000)
  (h6 : order.max_trad_quantity = 400)
  (h7 : order.max_total_cost = 12880) :
  ∃ (red_price trad_price min_cost : ℝ),
    red_price = 14 ∧
    trad_price = 10 ∧
    min_cost = 12400 ∧
    red_price = order.red_price_ratio * trad_price ∧
    order.red_cost / red_price - order.trad_cost / trad_price = order.quantity_diff ∧
    min_cost ≤ order.max_total_cost ∧
    (∀ (trad_quantity : ℕ),
      trad_quantity ≤ order.max_trad_quantity →
      trad_quantity * trad_price + (order.total_quantity - trad_quantity) * red_price ≥ min_cost) :=
by sorry

end NUMINAMATH_CALUDE_textbook_order_solution_l3072_307274


namespace NUMINAMATH_CALUDE_total_buyers_is_140_l3072_307227

/-- The number of buyers in a grocery store over three consecutive days -/
structure BuyerCount where
  day_before_yesterday : ℕ
  yesterday : ℕ
  today : ℕ

/-- Conditions for the buyer count problem -/
def buyer_count_conditions (b : BuyerCount) : Prop :=
  b.today = b.yesterday + 40 ∧
  b.yesterday = b.day_before_yesterday / 2 ∧
  b.day_before_yesterday = 50

/-- The total number of buyers over three days -/
def total_buyers (b : BuyerCount) : ℕ :=
  b.day_before_yesterday + b.yesterday + b.today

/-- Theorem stating that given the conditions, the total number of buyers is 140 -/
theorem total_buyers_is_140 (b : BuyerCount) (h : buyer_count_conditions b) :
  total_buyers b = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_buyers_is_140_l3072_307227


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3072_307286

theorem quadratic_coefficient (b : ℝ) (n : ℝ) : 
  b < 0 ∧ 
  (∀ x, x^2 + b*x + 1/5 = (x+n)^2 + 1/20) →
  b = -Real.sqrt (3/5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3072_307286


namespace NUMINAMATH_CALUDE_butterfat_mixture_l3072_307270

theorem butterfat_mixture (initial_volume : ℝ) (initial_butterfat : ℝ) 
  (added_volume : ℝ) (added_butterfat : ℝ) (target_butterfat : ℝ) :
  initial_volume = 8 →
  initial_butterfat = 0.3 →
  added_butterfat = 0.1 →
  target_butterfat = 0.2 →
  added_volume = 8 →
  (initial_volume * initial_butterfat + added_volume * added_butterfat) / 
  (initial_volume + added_volume) = target_butterfat :=
by
  sorry

#check butterfat_mixture

end NUMINAMATH_CALUDE_butterfat_mixture_l3072_307270


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_400_l3072_307219

theorem largest_multiple_of_15_under_400 : ∃ n : ℕ, n * 15 = 390 ∧ 
  390 < 400 ∧ 
  (∀ m : ℕ, m * 15 < 400 → m * 15 ≤ 390) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_400_l3072_307219


namespace NUMINAMATH_CALUDE_raccoon_stall_time_l3072_307240

/-- The time (in minutes) the first lock stalls the raccoons -/
def T1 : ℕ := 5

/-- The time (in minutes) the second lock stalls the raccoons -/
def T2 : ℕ := 3 * T1 - 3

/-- The time (in minutes) both locks together stall the raccoons -/
def both_locks : ℕ := 5 * T2

theorem raccoon_stall_time : both_locks = 60 := by
  sorry

end NUMINAMATH_CALUDE_raccoon_stall_time_l3072_307240


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l3072_307230

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (D : ℝ) :
  A = 400 * Real.pi →
  A = Real.pi * r^2 →
  D = 2 * r →
  D = 40 := by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l3072_307230


namespace NUMINAMATH_CALUDE_equation_solution_l3072_307245

theorem equation_solution : ∃! x : ℝ, (4 : ℝ) ^ (x + 1) = (64 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3072_307245


namespace NUMINAMATH_CALUDE_meal_cost_theorem_l3072_307229

-- Define variables for item costs
variable (s c p k : ℝ)

-- Define the equations from the given meals
def meal1_equation : Prop := 2 * s + 5 * c + 2 * p + 3 * k = 6.30
def meal2_equation : Prop := 3 * s + 8 * c + 2 * p + 4 * k = 8.40

-- Theorem to prove
theorem meal_cost_theorem 
  (h1 : meal1_equation s c p k)
  (h2 : meal2_equation s c p k) :
  s + c + p + k = 3.15 := by
  sorry


end NUMINAMATH_CALUDE_meal_cost_theorem_l3072_307229


namespace NUMINAMATH_CALUDE_given_equation_is_quadratic_l3072_307252

/-- An equation is quadratic in one variable if it can be expressed in the form ax² + bx + c = 0, where a ≠ 0 and x is the variable. --/
def is_quadratic_one_var (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation (x - 1)(x + 2) = 1 --/
def given_equation (x : ℝ) : ℝ :=
  (x - 1) * (x + 2) - 1

theorem given_equation_is_quadratic :
  is_quadratic_one_var given_equation :=
sorry

end NUMINAMATH_CALUDE_given_equation_is_quadratic_l3072_307252


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3072_307254

-- Define the propositions p and q
def p (a : ℝ) : Prop := 1/a > 1/4

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 + a*x + 1 > 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary :
  (∃ a : ℝ, p a ∧ q a) ∧ (∃ a : ℝ, ¬p a ∧ q a) ∧ (∀ a : ℝ, p a → q a) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3072_307254


namespace NUMINAMATH_CALUDE_unique_solution_mn_l3072_307208

theorem unique_solution_mn : ∃! (m n : ℕ+), 10 * m * n = 45 - 5 * m - 3 * n := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_mn_l3072_307208


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3072_307228

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 9*x + 14 = 0 → ∃ s₁ s₂ : ℝ, s₁ + s₂ = 9 ∧ s₁ * s₂ = 14 ∧ s₁^2 + s₂^2 = 53 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3072_307228


namespace NUMINAMATH_CALUDE_prime_divisors_50_factorial_l3072_307273

/-- The number of prime divisors of 50! -/
def num_prime_divisors_50_factorial : ℕ := sorry

/-- Theorem stating that the number of prime divisors of 50! is 15 -/
theorem prime_divisors_50_factorial :
  num_prime_divisors_50_factorial = 15 := by sorry

end NUMINAMATH_CALUDE_prime_divisors_50_factorial_l3072_307273


namespace NUMINAMATH_CALUDE_power_comparison_l3072_307249

theorem power_comparison : 2^1997 > 5^850 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l3072_307249


namespace NUMINAMATH_CALUDE_negation_of_existence_cubic_inequality_negation_l3072_307277

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x₀ : ℝ, f x₀ ≤ 0) ↔ (∀ x : ℝ, f x > 0) :=
by sorry

theorem cubic_inequality_negation :
  (¬ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_cubic_inequality_negation_l3072_307277


namespace NUMINAMATH_CALUDE_angle_problem_l3072_307256

theorem angle_problem (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 + angle2 = 180)
  (h2 : angle2 + angle3 + angle4 = 180)
  (h3 : angle1 = 70)
  (h4 : angle3 = 40) : 
  angle4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_problem_l3072_307256


namespace NUMINAMATH_CALUDE_sum_of_cubes_plus_one_divisible_by_5_l3072_307243

def sum_of_cubes_plus_one (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i => (i + 1)^3 + 1)

theorem sum_of_cubes_plus_one_divisible_by_5 :
  5 ∣ sum_of_cubes_plus_one 50 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_plus_one_divisible_by_5_l3072_307243


namespace NUMINAMATH_CALUDE_triangle_case1_triangle_case2_l3072_307283

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangleConditions (t : Triangle) : Prop :=
  t.c = 2 * Real.sqrt 3 ∧ Real.sin t.B = 2 * Real.sin t.A

-- Theorem 1: When C = π/3
theorem triangle_case1 (t : Triangle) (h : triangleConditions t) (hC : t.C = π / 3) :
  t.a = 2 ∧ t.b = 4 := by sorry

-- Theorem 2: When cos C = 1/4
theorem triangle_case2 (t : Triangle) (h : triangleConditions t) (hC : Real.cos t.C = 1 / 4) :
  (1 / 2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 15) / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_case1_triangle_case2_l3072_307283


namespace NUMINAMATH_CALUDE_min_value_of_x_l3072_307222

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ Real.log 2 + (1/2) * Real.log x) : x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_x_l3072_307222


namespace NUMINAMATH_CALUDE_min_turn_angles_sum_l3072_307201

/-- Represents a broken line path in a circular arena -/
structure BrokenLinePath where
  /-- Radius of the circular arena in meters -/
  arena_radius : ℝ
  /-- Total length of the path in meters -/
  total_length : ℝ
  /-- List of angles between consecutive segments in radians -/
  turn_angles : List ℝ

/-- Theorem: The sum of turn angles in a broken line path is at least 2998 radians
    given the specified arena radius and path length -/
theorem min_turn_angles_sum (path : BrokenLinePath)
    (h_radius : path.arena_radius = 10)
    (h_length : path.total_length = 30000) :
    (path.turn_angles.sum ≥ 2998) := by
  sorry


end NUMINAMATH_CALUDE_min_turn_angles_sum_l3072_307201


namespace NUMINAMATH_CALUDE_unique_cube_prime_l3072_307239

theorem unique_cube_prime (p : ℕ) : Prime p → (∃ n : ℕ, 2 * p + 1 = n ^ 3) ↔ p = 13 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_prime_l3072_307239


namespace NUMINAMATH_CALUDE_tournament_games_l3072_307253

theorem tournament_games (total_teams : Nat) (preliminary_teams : Nat) (preliminary_matches : Nat) :
  total_teams = 24 →
  preliminary_teams = 16 →
  preliminary_matches = 8 →
  preliminary_teams = 2 * preliminary_matches →
  (total_games : Nat) = preliminary_matches + (total_teams - preliminary_matches) - 1 →
  total_games = 23 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_l3072_307253


namespace NUMINAMATH_CALUDE_zero_geometric_mean_with_one_l3072_307298

def geometric_mean (list : List ℝ) : ℝ := (list.prod) ^ (1 / list.length)

theorem zero_geometric_mean_with_one {n : ℕ} (h : n > 1) :
  let list : List ℝ := 1 :: List.replicate (n - 1) 0
  geometric_mean list = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_geometric_mean_with_one_l3072_307298


namespace NUMINAMATH_CALUDE_pencil_packing_problem_l3072_307257

theorem pencil_packing_problem :
  ∃ (a k m : ℤ),
    200 ≤ a ∧ a ≤ 300 ∧
    a % 10 = 7 ∧
    a % 12 = 9 ∧
    a = 60 * m + 57 ∧
    (m = 3 ∨ m = 4) :=
by sorry

end NUMINAMATH_CALUDE_pencil_packing_problem_l3072_307257


namespace NUMINAMATH_CALUDE_expression_evaluation_l3072_307237

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 3 - 1
  (2 / (x + 1) + 1 / (x - 2)) / ((x - 1) / (x - 2)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3072_307237


namespace NUMINAMATH_CALUDE_circle_equation_with_PQ_diameter_l3072_307250

/-- Given circle equation -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + x - 6*y + 3 = 0

/-- Given line equation -/
def given_line (x y : ℝ) : Prop :=
  x + 2*y - 3 = 0

/-- Intersection points P and Q -/
def intersection_points (P Q : ℝ × ℝ) : Prop :=
  given_circle P.1 P.2 ∧ given_line P.1 P.2 ∧
  given_circle Q.1 Q.2 ∧ given_line Q.1 Q.2 ∧
  P ≠ Q

/-- Circle equation with PQ as diameter -/
def circle_PQ_diameter (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- Theorem statement -/
theorem circle_equation_with_PQ_diameter
  (P Q : ℝ × ℝ) (h : intersection_points P Q) :
  ∀ x y : ℝ, circle_PQ_diameter x y ↔
    (x - P.1)^2 + (y - P.2)^2 = (Q.1 - P.1)^2 + (Q.2 - P.2)^2 / 4 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_with_PQ_diameter_l3072_307250


namespace NUMINAMATH_CALUDE_correct_young_sample_size_l3072_307220

/-- Represents the stratified sampling problem for a company's employees. -/
structure CompanySampling where
  total_employees : ℕ
  young_employees : ℕ
  sample_size : ℕ
  young_in_sample : ℕ

/-- Theorem stating the correct number of young employees in the sample. -/
theorem correct_young_sample_size (c : CompanySampling) 
    (h1 : c.total_employees = 200)
    (h2 : c.young_employees = 120)
    (h3 : c.sample_size = 25)
    (h4 : c.young_in_sample = c.young_employees * c.sample_size / c.total_employees) :
  c.young_in_sample = 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_young_sample_size_l3072_307220


namespace NUMINAMATH_CALUDE_beryl_radishes_l3072_307204

def radishes_problem (first_basket : ℕ) (difference : ℕ) : Prop :=
  let second_basket := first_basket + difference
  let total := first_basket + second_basket
  total = 88

theorem beryl_radishes : radishes_problem 37 14 := by
  sorry

end NUMINAMATH_CALUDE_beryl_radishes_l3072_307204


namespace NUMINAMATH_CALUDE_sum_reciprocals_inequality_l3072_307248

theorem sum_reciprocals_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a^2 + 1 / b^2 ≥ 8) ∧ (1 / a + 1 / b + 1 / (a * b) ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_inequality_l3072_307248


namespace NUMINAMATH_CALUDE_number_divided_by_expression_equals_one_l3072_307262

theorem number_divided_by_expression_equals_one :
  ∃ x : ℝ, x / (5 + 3 / 0.75) = 1 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_expression_equals_one_l3072_307262


namespace NUMINAMATH_CALUDE_one_solution_condition_l3072_307255

theorem one_solution_condition (a : ℝ) :
  (∃! x : ℝ, x ≠ -4 ∧ x ≠ 1 ∧ |x + 1| = |x - 4| + a) ↔ a ∈ Set.Ioo (-5 : ℝ) (-1) ∪ Set.Ioo (-1 : ℝ) 5 :=
by sorry

end NUMINAMATH_CALUDE_one_solution_condition_l3072_307255


namespace NUMINAMATH_CALUDE_min_socks_for_pair_is_four_l3072_307211

/-- Represents a drawer of socks with three colors -/
structure SockDrawer :=
  (white : Nat)
  (green : Nat)
  (red : Nat)

/-- Ensures that there is at least one sock of each color -/
def hasAllColors (drawer : SockDrawer) : Prop :=
  drawer.white > 0 ∧ drawer.green > 0 ∧ drawer.red > 0

/-- The minimum number of socks needed to ensure at least two of the same color -/
def minSocksForPair (drawer : SockDrawer) : Nat :=
  4

theorem min_socks_for_pair_is_four (drawer : SockDrawer) 
  (h : hasAllColors drawer) : 
  minSocksForPair drawer = 4 := by
  sorry

#check min_socks_for_pair_is_four

end NUMINAMATH_CALUDE_min_socks_for_pair_is_four_l3072_307211


namespace NUMINAMATH_CALUDE_sector_area_l3072_307295

-- Define the parameters
def arc_length : ℝ := 1
def radius : ℝ := 4

-- Define the theorem
theorem sector_area : 
  let θ := arc_length / radius
  (1/2) * radius^2 * θ = 2 := by sorry

end NUMINAMATH_CALUDE_sector_area_l3072_307295


namespace NUMINAMATH_CALUDE_unique_solution_xy_l3072_307232

theorem unique_solution_xy (x y : ℕ) :
  x * (x + 1) = 4 * y * (y + 1) → (x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_xy_l3072_307232


namespace NUMINAMATH_CALUDE_symmetric_complex_number_l3072_307205

/-- Given that z is symmetric to 2/(1-i) with respect to the imaginary axis, prove that z = -1 + i -/
theorem symmetric_complex_number (z : ℂ) : 
  (z.re = -(2 / (1 - I)).re ∧ z.im = (2 / (1 - I)).im) → z = -1 + I :=
by sorry

end NUMINAMATH_CALUDE_symmetric_complex_number_l3072_307205


namespace NUMINAMATH_CALUDE_tangent_line_at_point_l3072_307223

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

theorem tangent_line_at_point (x₀ y₀ : ℝ) (h : y₀ = f x₀) :
  let m := (3*x₀^2 - 4*x₀ - 4)  -- Derivative of f at x₀
  (5 : ℝ) * x + y - 2 = 0 ↔ y - y₀ = m * (x - x₀) ∧ x₀ = 1 ∧ y₀ = -3 :=
by sorry

#check tangent_line_at_point

end NUMINAMATH_CALUDE_tangent_line_at_point_l3072_307223


namespace NUMINAMATH_CALUDE_additional_discount_percentage_l3072_307271

/-- Represents the price of duty shoes in cents -/
def full_price : ℕ := 8500

/-- Represents the first discount percentage for officers who served at least a year -/
def first_discount_percent : ℕ := 20

/-- Represents the price paid by officers who served at least three years in cents -/
def price_three_years : ℕ := 5100

/-- Calculates the price after the first discount -/
def price_after_first_discount : ℕ := full_price - (full_price * first_discount_percent / 100)

/-- Represents the additional discount percentage for officers who served at least three years -/
def additional_discount_percent : ℕ := 25

theorem additional_discount_percentage :
  (price_after_first_discount - price_three_years) * 100 / price_after_first_discount = additional_discount_percent := by
  sorry

end NUMINAMATH_CALUDE_additional_discount_percentage_l3072_307271


namespace NUMINAMATH_CALUDE_walking_rate_ratio_l3072_307233

theorem walking_rate_ratio 
  (D : ℝ) -- Distance to school
  (R : ℝ) -- Usual walking rate
  (R' : ℝ) -- New walking rate
  (h1 : D = R * 21) -- Usual time equation
  (h2 : D = R' * 18) -- New time equation
  : R' / R = 7 / 6 := by sorry

end NUMINAMATH_CALUDE_walking_rate_ratio_l3072_307233


namespace NUMINAMATH_CALUDE_divisor_count_equals_equation_solutions_l3072_307242

/-- The prime factorization of 2310 -/
def prime_factors : List Nat := [2, 3, 5, 7, 11]

/-- The exponent of 2310 in the number we're considering -/
def exponent : Nat := 2310

/-- A function that counts the number of positive integer divisors of n^exponent 
    that are divisible by exactly 48 positive integers, 
    where n is the product of the prime factors -/
def count_specific_divisors (prime_factors : List Nat) (exponent : Nat) : Nat :=
  sorry

/-- A function that counts the number of solutions (a,b,c,d,e) to the equation 
    (a+1)(b+1)(c+1)(d+1)(e+1) = 48, where a,b,c,d,e are non-negative integers -/
def count_equation_solutions : Nat :=
  sorry

/-- The main theorem stating the equality of the two counting functions -/
theorem divisor_count_equals_equation_solutions : 
  count_specific_divisors prime_factors exponent = count_equation_solutions :=
  sorry

end NUMINAMATH_CALUDE_divisor_count_equals_equation_solutions_l3072_307242


namespace NUMINAMATH_CALUDE_crayons_left_l3072_307238

theorem crayons_left (total : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) : 
  total = 120 →
  kiley_fraction = 3/8 →
  joe_fraction = 5/9 →
  (total - kiley_fraction * total) - joe_fraction * (total - kiley_fraction * total) = 33 := by
  sorry

end NUMINAMATH_CALUDE_crayons_left_l3072_307238


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3072_307265

theorem polynomial_simplification (x : ℝ) :
  (3 * x^6 + 2 * x^5 + x^4 + x - 5) - (x^6 + 3 * x^5 + 2 * x^3 + 6) =
  2 * x^6 - x^5 + x^4 - 2 * x^3 + x + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3072_307265


namespace NUMINAMATH_CALUDE_weighted_sum_square_inequality_l3072_307299

theorem weighted_sum_square_inequality (x y a b : ℝ) 
  (h1 : a + b = 1) (h2 : a ≥ 0) (h3 : b ≥ 0) : 
  (a * x + b * y)^2 ≤ a * x^2 + b * y^2 := by
  sorry

end NUMINAMATH_CALUDE_weighted_sum_square_inequality_l3072_307299


namespace NUMINAMATH_CALUDE_regular_octagon_area_l3072_307203

/-- Regular octagon with area A, longest diagonal d_max, and shortest diagonal d_min -/
structure RegularOctagon where
  A : ℝ
  d_max : ℝ
  d_min : ℝ

/-- The area of a regular octagon is equal to the product of its longest and shortest diagonals -/
theorem regular_octagon_area (o : RegularOctagon) : o.A = o.d_max * o.d_min := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_area_l3072_307203


namespace NUMINAMATH_CALUDE_intersection_count_l3072_307297

/-- Two circles in a plane -/
structure TwoCircles where
  /-- Center of the first circle -/
  center1 : ℝ × ℝ
  /-- Radius of the first circle -/
  radius1 : ℝ
  /-- Center of the second circle -/
  center2 : ℝ × ℝ
  /-- Radius of the second circle -/
  radius2 : ℝ

/-- The number of intersection points between two circles -/
def intersectionPoints (circles : TwoCircles) : ℕ :=
  sorry

/-- Theorem: The number of intersection points between the given circles is 4 -/
theorem intersection_count : 
  let circles : TwoCircles := {
    center1 := (0, 3),
    radius1 := 3,
    center2 := (3/2, 0),
    radius2 := 3/2
  }
  intersectionPoints circles = 4 := by sorry

end NUMINAMATH_CALUDE_intersection_count_l3072_307297


namespace NUMINAMATH_CALUDE_factory_door_production_l3072_307296

/-- Calculates the number of doors produced by a car factory given various production changes -/
theorem factory_door_production
  (doors_per_car : ℕ)
  (initial_plan : ℕ)
  (shortage_decrease : ℕ)
  (pandemic_cut : Rat)
  (h1 : doors_per_car = 5)
  (h2 : initial_plan = 200)
  (h3 : shortage_decrease = 50)
  (h4 : pandemic_cut = 1/2) :
  (initial_plan - shortage_decrease) * pandemic_cut * doors_per_car = 375 := by
  sorry

end NUMINAMATH_CALUDE_factory_door_production_l3072_307296


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3072_307224

/-- Given that x and y are inversely proportional, prove that when x = -12, y = -39.0625 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 50 ∧ x₀ = 3 * y₀ ∧ x₀ * y₀ = k) :
  x = -12 → y = -39.0625 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3072_307224


namespace NUMINAMATH_CALUDE_count_with_zero_1000_l3072_307209

def count_with_zero (n : ℕ) : ℕ :=
  (n + 1) - (9 * 9 * 10)

theorem count_with_zero_1000 : count_with_zero 1000 = 181 := by
  sorry

end NUMINAMATH_CALUDE_count_with_zero_1000_l3072_307209


namespace NUMINAMATH_CALUDE_nigels_money_ratio_l3072_307217

/-- Represents Nigel's money transactions and proves the final ratio --/
theorem nigels_money_ratio :
  ∀ (original : ℝ) (given_away : ℝ),
  original > 0 →
  given_away > 0 →
  original + 45 - given_away + 80 - 25 = 2 * original + 25 →
  (original + 45 + 80 - 25) / original = 3 :=
by sorry

end NUMINAMATH_CALUDE_nigels_money_ratio_l3072_307217


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l3072_307216

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Definition of circle M -/
def circle_M : Circle :=
  { center := (-1, 0), radius := 1 }

/-- Definition of circle N -/
def circle_N : Circle :=
  { center := (1, 0), radius := 5 }

/-- Definition of external tangency -/
def is_externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

/-- Definition of internal tangency -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius - c2.radius)^2

/-- Theorem: The trajectory of the center of circle P is an ellipse -/
theorem trajectory_is_ellipse (P : ℝ × ℝ) :
  is_externally_tangent { center := P, radius := 0 } circle_M →
  is_internally_tangent { center := P, radius := 0 } circle_N →
  P.1^2 / 9 + P.2^2 / 8 = 1 := by sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l3072_307216


namespace NUMINAMATH_CALUDE_function_composition_l3072_307260

theorem function_composition (f : ℝ → ℝ) (x : ℝ) : 
  (∀ y, f y = y^2 + 2*y - 1) → f (x - 1) = x^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l3072_307260


namespace NUMINAMATH_CALUDE_intersection_A_B_l3072_307287

def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {-2, 0, 2}

theorem intersection_A_B : A ∩ B = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3072_307287


namespace NUMINAMATH_CALUDE_unique_mythical_with_most_divisors_l3072_307263

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def is_mythical (n : ℕ) : Prop :=
  n > 0 ∧ ∀ d : ℕ, d ∣ n → ∃ p : ℕ, is_prime p ∧ d = p - 2

def number_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem unique_mythical_with_most_divisors :
  is_mythical 135 ∧
  ∀ n : ℕ, is_mythical n → number_of_divisors n ≤ number_of_divisors 135 ∧
  (number_of_divisors n = number_of_divisors 135 → n = 135) :=
sorry

end NUMINAMATH_CALUDE_unique_mythical_with_most_divisors_l3072_307263


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l3072_307268

theorem smallest_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 3457) % 15 = 1537 % 15 ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 3457) % 15 = 1537 % 15 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l3072_307268


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l3072_307272

theorem greatest_integer_radius (r : ℕ) : (∀ n : ℕ, n > r → (n : ℝ)^2 * Real.pi ≥ 75 * Real.pi) ∧ r^2 * Real.pi < 75 * Real.pi → r = 8 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l3072_307272


namespace NUMINAMATH_CALUDE_absolute_difference_always_less_than_one_l3072_307210

theorem absolute_difference_always_less_than_one :
  ∀ (m : ℝ), ∀ (x : ℝ), |x - m| < 1 :=
by sorry

end NUMINAMATH_CALUDE_absolute_difference_always_less_than_one_l3072_307210


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_existence_l3072_307258

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ c > 0, p c) ↔ (∀ c > 0, ¬p c) :=
by sorry

def has_solution (c : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - x + c = 0

theorem negation_of_quadratic_existence :
  (¬∃ c > 0, has_solution c) ↔ (∀ c > 0, ¬has_solution c) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_existence_l3072_307258


namespace NUMINAMATH_CALUDE_square_of_sum_l3072_307291

theorem square_of_sum (a b : ℝ) : (a + b)^2 = a^2 + 2*a*b + b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l3072_307291


namespace NUMINAMATH_CALUDE_watch_loss_percentage_loss_percentage_is_ten_percent_l3072_307235

/-- Proves that the loss percentage is 10% for a watch sale scenario --/
theorem watch_loss_percentage : ℝ → Prop :=
  λ L : ℝ =>
    let cost_price : ℝ := 2000
    let selling_price : ℝ := cost_price - (L / 100 * cost_price)
    let new_selling_price : ℝ := cost_price + (4 / 100 * cost_price)
    new_selling_price = selling_price + 280 →
    L = 10

/-- The loss percentage is indeed 10% --/
theorem loss_percentage_is_ten_percent : watch_loss_percentage 10 := by
  sorry

end NUMINAMATH_CALUDE_watch_loss_percentage_loss_percentage_is_ten_percent_l3072_307235


namespace NUMINAMATH_CALUDE_eight_b_plus_one_composite_l3072_307206

theorem eight_b_plus_one_composite (a b : ℕ) (h1 : a > b) (h2 : a - b = 5 * b^2 - 4 * a^2) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ 8 * b + 1 = x * y :=
sorry

end NUMINAMATH_CALUDE_eight_b_plus_one_composite_l3072_307206


namespace NUMINAMATH_CALUDE_polynomial_equality_l3072_307266

theorem polynomial_equality (x : ℝ) : 
  let k : ℝ := -9
  let a : ℝ := 15
  let b : ℝ := 72
  (3 * x^2 - 4 * x + 5) * (5 * x^2 + k * x + 8) = 
    15 * x^4 - 47 * x^3 + a * x^2 - b * x + 40 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3072_307266


namespace NUMINAMATH_CALUDE_solution_set_x_squared_geq_four_l3072_307269

theorem solution_set_x_squared_geq_four :
  {x : ℝ | x^2 ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_geq_four_l3072_307269


namespace NUMINAMATH_CALUDE_max_revenue_theorem_l3072_307279

/-- Represents the advertising allocation problem for a company --/
structure AdvertisingProblem where
  totalTime : ℝ
  totalBudget : ℝ
  rateA : ℝ
  rateB : ℝ
  revenueA : ℝ
  revenueB : ℝ

/-- Represents a solution to the advertising allocation problem --/
structure AdvertisingSolution where
  timeA : ℝ
  timeB : ℝ
  revenue : ℝ

/-- Checks if a solution is valid for a given problem --/
def isValidSolution (p : AdvertisingProblem) (s : AdvertisingSolution) : Prop :=
  s.timeA ≥ 0 ∧ s.timeB ≥ 0 ∧
  s.timeA + s.timeB ≤ p.totalTime ∧
  s.timeA * p.rateA + s.timeB * p.rateB ≤ p.totalBudget ∧
  s.revenue = s.timeA * p.revenueA + s.timeB * p.revenueB

/-- Theorem stating that the given solution maximizes revenue --/
theorem max_revenue_theorem (p : AdvertisingProblem)
  (h1 : p.totalTime = 300)
  (h2 : p.totalBudget = 90000)
  (h3 : p.rateA = 500)
  (h4 : p.rateB = 200)
  (h5 : p.revenueA = 0.3)
  (h6 : p.revenueB = 0.2) :
  ∃ (s : AdvertisingSolution),
    isValidSolution p s ∧
    s.timeA = 100 ∧
    s.timeB = 200 ∧
    s.revenue = 70 ∧
    ∀ (s' : AdvertisingSolution), isValidSolution p s' → s'.revenue ≤ s.revenue :=
sorry

end NUMINAMATH_CALUDE_max_revenue_theorem_l3072_307279


namespace NUMINAMATH_CALUDE_valid_sequences_count_l3072_307236

def is_valid_sequence (s : List ℕ) : Prop :=
  s.length = 8 ∧
  s.toFinset.card = 8 ∧
  ∀ x ∈ s, 1 ≤ x ∧ x ≤ 11 ∧
  ∀ n, 1 ≤ n ∧ n ≤ 8 → (s.take n).sum % n = 0

theorem valid_sequences_count :
  ∃! (sequences : List (List ℕ)),
    sequences.length = 8 ∧
    ∀ s ∈ sequences, is_valid_sequence s :=
by sorry

end NUMINAMATH_CALUDE_valid_sequences_count_l3072_307236


namespace NUMINAMATH_CALUDE_chloe_win_prob_is_25_91_l3072_307292

/-- Represents the probability of rolling a specific number on a six-sided die -/
def roll_probability : ℚ := 1 / 6

/-- Represents the probability of not rolling a '6' on a six-sided die -/
def not_six_probability : ℚ := 5 / 6

/-- Calculates the probability of Chloe winning on her nth turn -/
def chloe_win_nth_turn (n : ℕ) : ℚ :=
  (not_six_probability ^ (3 * n - 1)) * roll_probability

/-- Calculates the sum of the geometric series representing Chloe's win probability -/
def chloe_win_probability : ℚ :=
  (chloe_win_nth_turn 1) / (1 - (not_six_probability ^ 3))

/-- Theorem stating that the probability of Chloe winning is 25/91 -/
theorem chloe_win_prob_is_25_91 : chloe_win_probability = 25 / 91 := by
  sorry

end NUMINAMATH_CALUDE_chloe_win_prob_is_25_91_l3072_307292


namespace NUMINAMATH_CALUDE_total_is_41X_l3072_307282

/-- Represents the number of people in different categories of a community -/
structure Community where
  children : ℕ
  teenagers : ℕ
  women : ℕ
  men : ℕ

/-- Defines a community with the given relationships between categories -/
def specialCommunity (X : ℕ) : Community where
  children := X
  teenagers := 4 * X
  women := 3 * (4 * X)
  men := 2 * (3 * (4 * X))

/-- Calculates the total number of people in a community -/
def totalPeople (c : Community) : ℕ :=
  c.children + c.teenagers + c.women + c.men

/-- Theorem stating that the total number of people in the special community is 41X -/
theorem total_is_41X (X : ℕ) :
  totalPeople (specialCommunity X) = 41 * X := by
  sorry

end NUMINAMATH_CALUDE_total_is_41X_l3072_307282


namespace NUMINAMATH_CALUDE_jamie_remaining_capacity_l3072_307241

/-- Jamie's bathroom limit in ounces -/
def bathroom_limit : ℕ := 32

/-- Amount of milk Jamie consumed in ounces -/
def milk_consumed : ℕ := 8

/-- Amount of grape juice Jamie consumed in ounces -/
def grape_juice_consumed : ℕ := 16

/-- Total amount of liquid Jamie consumed before the test -/
def total_consumed : ℕ := milk_consumed + grape_juice_consumed

/-- Theorem: Jamie can drink 8 ounces during the test before needing the bathroom -/
theorem jamie_remaining_capacity : bathroom_limit - total_consumed = 8 := by
  sorry

end NUMINAMATH_CALUDE_jamie_remaining_capacity_l3072_307241


namespace NUMINAMATH_CALUDE_mikaela_savings_l3072_307294

-- Define the hourly rate
def hourly_rate : ℕ := 10

-- Define the hours worked in the first month
def first_month_hours : ℕ := 35

-- Define the additional hours worked in the second month
def additional_hours : ℕ := 5

-- Define the fraction of earnings spent on personal needs
def spent_fraction : ℚ := 4/5

-- Function to calculate total earnings
def total_earnings (rate : ℕ) (hours1 : ℕ) (hours2 : ℕ) : ℕ :=
  rate * (hours1 + hours2)

-- Function to calculate savings
def savings (total : ℕ) (spent_frac : ℚ) : ℚ :=
  (1 - spent_frac) * total

-- Theorem statement
theorem mikaela_savings :
  savings (total_earnings hourly_rate first_month_hours (first_month_hours + additional_hours)) spent_fraction = 150 := by
  sorry


end NUMINAMATH_CALUDE_mikaela_savings_l3072_307294


namespace NUMINAMATH_CALUDE_equal_first_two_numbers_l3072_307246

theorem equal_first_two_numbers (a : Fin 17 → ℕ) 
  (h : ∀ i : Fin 17, (a i) ^ (a (i + 1)) = (a ((i + 1) % 17)) ^ (a ((i + 2) % 17))) : 
  a 0 = a 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_first_two_numbers_l3072_307246


namespace NUMINAMATH_CALUDE_remainder_3_pow_23_mod_11_l3072_307221

theorem remainder_3_pow_23_mod_11 : 3^23 % 11 = 5 := by sorry

end NUMINAMATH_CALUDE_remainder_3_pow_23_mod_11_l3072_307221


namespace NUMINAMATH_CALUDE_max_intersections_circle_quadrilateral_l3072_307261

/-- A circle in a 2D plane -/
structure Circle where
  -- We don't need to define the specifics of a circle for this problem

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral where
  -- We don't need to define the specifics of a quadrilateral for this problem

/-- The number of sides in a quadrilateral -/
def quadrilateral_sides : ℕ := 4

/-- The maximum number of intersections between a line segment and a circle -/
def max_intersections_line_circle : ℕ := 2

/-- Theorem: The maximum number of intersection points between a circle and a quadrilateral is 8 -/
theorem max_intersections_circle_quadrilateral (c : Circle) (q : Quadrilateral) :
  (quadrilateral_sides * max_intersections_line_circle) = 8 := by
  sorry

#check max_intersections_circle_quadrilateral

end NUMINAMATH_CALUDE_max_intersections_circle_quadrilateral_l3072_307261


namespace NUMINAMATH_CALUDE_power_gt_one_iff_product_gt_zero_l3072_307284

theorem power_gt_one_iff_product_gt_zero {a b : ℝ} (ha : a > 0) (ha' : a ≠ 1) :
  a^b > 1 ↔ (a - 1) * b > 0 := by
  sorry

end NUMINAMATH_CALUDE_power_gt_one_iff_product_gt_zero_l3072_307284


namespace NUMINAMATH_CALUDE_max_d_minus_r_value_l3072_307247

theorem max_d_minus_r_value : ∃ (d r : ℕ), 
  (2017 % d = r) ∧ (1029 % d = r) ∧ (725 % d = r) ∧
  (∀ (d' r' : ℕ), (2017 % d' = r') ∧ (1029 % d' = r') ∧ (725 % d' = r') → d' - r' ≤ d - r) ∧
  (d - r = 35) := by
  sorry

end NUMINAMATH_CALUDE_max_d_minus_r_value_l3072_307247


namespace NUMINAMATH_CALUDE_initial_paint_amount_l3072_307215

/-- The amount of paint Jimin used for his house -/
def paint_for_house : ℝ := 4.3

/-- The amount of paint Jimin used for his friend's house -/
def paint_for_friend : ℝ := 4.3

/-- The amount of paint remaining after painting both houses -/
def paint_remaining : ℝ := 8.8

/-- The initial amount of paint Jimin had -/
def initial_paint : ℝ := paint_for_house + paint_for_friend + paint_remaining

theorem initial_paint_amount : initial_paint = 17.4 := by sorry

end NUMINAMATH_CALUDE_initial_paint_amount_l3072_307215


namespace NUMINAMATH_CALUDE_skirt_ratio_is_two_thirds_l3072_307218

-- Define the number of skirts in each valley
def purple_skirts : ℕ := 10
def azure_skirts : ℕ := 60

-- Define the relationship between Purple and Seafoam Valley skirts
def seafoam_skirts : ℕ := 4 * purple_skirts

-- Define the ratio of Seafoam to Azure Valley skirts
def skirt_ratio : Rat := seafoam_skirts / azure_skirts

-- Theorem to prove
theorem skirt_ratio_is_two_thirds : skirt_ratio = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_skirt_ratio_is_two_thirds_l3072_307218


namespace NUMINAMATH_CALUDE_factor_quadratic_l3072_307285

theorem factor_quadratic (x : ℝ) : 2 * x^2 - 12 * x + 18 = 2 * (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factor_quadratic_l3072_307285


namespace NUMINAMATH_CALUDE_regular_decagon_diagonal_side_difference_l3072_307231

/-- In a regular decagon inscribed in a circle, the difference between the length of the diagonal 
    connecting vertices 3 apart and the side length is equal to the radius of the circumcircle. -/
theorem regular_decagon_diagonal_side_difference (R : ℝ) : 
  let side_length := 2 * R * Real.sin (π / 10)
  let diagonal_length := 2 * R * Real.sin (3 * π / 10)
  diagonal_length - side_length = R := by sorry

end NUMINAMATH_CALUDE_regular_decagon_diagonal_side_difference_l3072_307231


namespace NUMINAMATH_CALUDE_students_called_back_l3072_307200

theorem students_called_back (girls : ℕ) (boys : ℕ) (didnt_make_cut : ℕ) 
  (h1 : girls = 39)
  (h2 : boys = 4)
  (h3 : didnt_make_cut = 17) :
  girls + boys - didnt_make_cut = 26 := by
  sorry

end NUMINAMATH_CALUDE_students_called_back_l3072_307200


namespace NUMINAMATH_CALUDE_prime_power_divisibility_l3072_307276

theorem prime_power_divisibility (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → p ∣ a^n → p^n ∣ a^n := by
  sorry

end NUMINAMATH_CALUDE_prime_power_divisibility_l3072_307276


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3072_307244

def A : Set ℕ := {x | x - 4 < 0}
def B : Set ℕ := {0, 1, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3072_307244


namespace NUMINAMATH_CALUDE_problem_statement_l3072_307251

theorem problem_statement (a b c : ℝ) 
  (eq_condition : a - 2*b + c = 0) 
  (ineq_condition : a + 2*b + c < 0) : 
  b < 0 ∧ b^2 - a*c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3072_307251


namespace NUMINAMATH_CALUDE_buckingham_palace_visitor_difference_l3072_307290

/-- The number of visitors to Buckingham Palace on the current day -/
def current_day_visitors : ℕ := 317

/-- The number of visitors to Buckingham Palace on the previous day -/
def previous_day_visitors : ℕ := 295

/-- The difference in visitors between the current day and the previous day -/
def visitor_difference : ℕ := current_day_visitors - previous_day_visitors

theorem buckingham_palace_visitor_difference :
  visitor_difference = 22 :=
by sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitor_difference_l3072_307290


namespace NUMINAMATH_CALUDE_no_solution_exists_l3072_307226

theorem no_solution_exists : ¬∃ x : ℝ, 2 * ((x - 3) / 2 + 3) = x + 6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3072_307226


namespace NUMINAMATH_CALUDE_count_squares_below_line_l3072_307214

/-- The number of 1x1 squares in the first quadrant lying entirely below the line 6x + 216y = 1296 -/
def squaresBelowLine : ℕ :=
  -- Definition goes here
  sorry

/-- The equation of the line -/
def lineEquation (x y : ℝ) : Prop :=
  6 * x + 216 * y = 1296

theorem count_squares_below_line :
  squaresBelowLine = 537 := by
  sorry

end NUMINAMATH_CALUDE_count_squares_below_line_l3072_307214


namespace NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l3072_307264

/-- Systematic sampling theorem for a specific case -/
theorem systematic_sampling_smallest_number
  (total_items : ℕ)
  (sample_size : ℕ)
  (highest_drawn : ℕ)
  (h1 : total_items = 32)
  (h2 : sample_size = 8)
  (h3 : highest_drawn = 31)
  (h4 : highest_drawn ≤ total_items)
  : ∃ (smallest_drawn : ℕ),
    smallest_drawn = 3 ∧
    smallest_drawn > 0 ∧
    smallest_drawn ≤ highest_drawn ∧
    (highest_drawn - smallest_drawn) % (total_items / sample_size) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l3072_307264


namespace NUMINAMATH_CALUDE_lunch_pizzas_calculation_l3072_307275

def total_pizzas : ℕ := 15
def dinner_pizzas : ℕ := 6

theorem lunch_pizzas_calculation :
  total_pizzas - dinner_pizzas = 9 := by
  sorry

end NUMINAMATH_CALUDE_lunch_pizzas_calculation_l3072_307275


namespace NUMINAMATH_CALUDE_tournament_has_25_players_l3072_307280

/-- Represents a tournament with the given conditions -/
structure Tournament where
  n : ℕ  -- number of players not in the lowest 5
  total_players : ℕ := n + 5
  total_games : ℕ := (total_players * (total_players - 1)) / 2
  points_top_n : ℕ := (n * (n - 1)) / 2
  points_bottom_5 : ℕ := 10

/-- The theorem stating that a tournament satisfying the given conditions must have 25 players -/
theorem tournament_has_25_players (t : Tournament) : t.total_players = 25 := by
  sorry

#check tournament_has_25_players

end NUMINAMATH_CALUDE_tournament_has_25_players_l3072_307280
