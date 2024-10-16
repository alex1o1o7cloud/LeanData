import Mathlib

namespace NUMINAMATH_CALUDE_percentage_sum_equality_l3014_301468

theorem percentage_sum_equality : 
  (25 / 100 * 2018) + (2018 / 100 * 25) = 1009 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_equality_l3014_301468


namespace NUMINAMATH_CALUDE_same_acquaintance_count_l3014_301414

theorem same_acquaintance_count (n : ℕ) (h : n > 0) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧
  (∃ (f : Fin n → ℕ), (∀ x, f x < n) ∧ f i = k ∧ f j = k) :=
by sorry

end NUMINAMATH_CALUDE_same_acquaintance_count_l3014_301414


namespace NUMINAMATH_CALUDE_car_speed_l3014_301497

/-- Given a car that travels 390 miles in 6 hours, prove its speed is 65 miles per hour -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 390 ∧ time = 6 ∧ speed = distance / time → speed = 65 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_l3014_301497


namespace NUMINAMATH_CALUDE_triangle_sum_equals_22_l3014_301443

/-- The triangle operation defined as 2a - b + c -/
def triangle_op (a b c : ℤ) : ℤ := 2*a - b + c

/-- The vertices of the first triangle -/
def triangle1 : List ℤ := [3, 7, 5]

/-- The vertices of the second triangle -/
def triangle2 : List ℤ := [6, 2, 8]

theorem triangle_sum_equals_22 : 
  triangle_op triangle1[0] triangle1[1] triangle1[2] + 
  triangle_op triangle2[0] triangle2[1] triangle2[2] = 22 := by
sorry

end NUMINAMATH_CALUDE_triangle_sum_equals_22_l3014_301443


namespace NUMINAMATH_CALUDE_average_income_q_and_r_l3014_301456

/-- Given the average monthly incomes of P and Q, P and R, and P's income,
    prove that the average monthly income of Q and R is 6250. -/
theorem average_income_q_and_r (p q r : ℕ) : 
  (p + q) / 2 = 5050 →
  (p + r) / 2 = 5200 →
  p = 4000 →
  (q + r) / 2 = 6250 := by
sorry

end NUMINAMATH_CALUDE_average_income_q_and_r_l3014_301456


namespace NUMINAMATH_CALUDE_power_three_mod_seven_l3014_301493

theorem power_three_mod_seven : 3^123 % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_power_three_mod_seven_l3014_301493


namespace NUMINAMATH_CALUDE_caesars_rental_cost_is_800_l3014_301492

/-- Caesar's room rental cost -/
def caesars_rental_cost : ℝ := sorry

/-- Caesar's per-person meal cost -/
def caesars_meal_cost : ℝ := 30

/-- Venus Hall's room rental cost -/
def venus_rental_cost : ℝ := 500

/-- Venus Hall's per-person meal cost -/
def venus_meal_cost : ℝ := 35

/-- Number of guests at which the costs are equal -/
def equal_cost_guests : ℕ := 60

theorem caesars_rental_cost_is_800 :
  caesars_rental_cost = 800 :=
by
  have h : caesars_rental_cost + caesars_meal_cost * equal_cost_guests =
           venus_rental_cost + venus_meal_cost * equal_cost_guests :=
    sorry
  sorry

end NUMINAMATH_CALUDE_caesars_rental_cost_is_800_l3014_301492


namespace NUMINAMATH_CALUDE_parabola_circle_equation_l3014_301489

/-- The equation of a circle with center at the focus of a parabola and diameter
    equal to the line segment formed by the intersection of the parabola with a
    line perpendicular to the x-axis passing through the focus. -/
theorem parabola_circle_equation (x y : ℝ) : 
  let parabola := {(x, y) | y^2 = 4*x}
  let focus := (1, 0)
  let perpendicular_line := {(x, y) | x = 1}
  let intersection := parabola ∩ perpendicular_line
  true → (x - 1)^2 + y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_circle_equation_l3014_301489


namespace NUMINAMATH_CALUDE_circle_tangency_l3014_301403

/-- Two circles are internally tangent if the distance between their centers
    is equal to the difference of their radii. -/
def internally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r2 - r1)^2

theorem circle_tangency (n : ℝ) : 
  internally_tangent (0, 0) (3, 4) 1 (Real.sqrt (25 - n)) → n = -11 := by
  sorry

#check circle_tangency

end NUMINAMATH_CALUDE_circle_tangency_l3014_301403


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3014_301438

theorem quadratic_roots_relation (m n p q : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0) :
  (∃ s₁ s₂ : ℝ, s₁ + s₂ = -p ∧ s₁ * s₂ = q ∧
               3 * s₁ + 3 * s₂ = -m ∧ 9 * s₁ * s₂ = n) →
  n / q = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3014_301438


namespace NUMINAMATH_CALUDE_canteen_leak_rate_l3014_301432

/-- Proves that the canteen leak rate is 1 cup per hour given the hiking conditions -/
theorem canteen_leak_rate
  (total_distance : ℝ)
  (initial_water : ℝ)
  (hike_duration : ℝ)
  (remaining_water : ℝ)
  (last_mile_consumption : ℝ)
  (first_three_miles_rate : ℝ)
  (h1 : total_distance = 4)
  (h2 : initial_water = 6)
  (h3 : hike_duration = 2)
  (h4 : remaining_water = 1)
  (h5 : last_mile_consumption = 1)
  (h6 : first_three_miles_rate = 0.6666666666666666)
  : (initial_water - remaining_water - (first_three_miles_rate * 3 + last_mile_consumption)) / hike_duration = 1 := by
  sorry

#check canteen_leak_rate

end NUMINAMATH_CALUDE_canteen_leak_rate_l3014_301432


namespace NUMINAMATH_CALUDE_min_m_plus_n_for_1978_power_divisibility_l3014_301437

theorem min_m_plus_n_for_1978_power_divisibility (m n : ℕ) : 
  m > n → n ≥ 1 → (1000 ∣ 1978^m - 1978^n) → m + n ≥ 106 ∧ ∃ (m₀ n₀ : ℕ), m₀ > n₀ ∧ n₀ ≥ 1 ∧ (1000 ∣ 1978^m₀ - 1978^n₀) ∧ m₀ + n₀ = 106 :=
by sorry

end NUMINAMATH_CALUDE_min_m_plus_n_for_1978_power_divisibility_l3014_301437


namespace NUMINAMATH_CALUDE_perpendicular_lines_coefficient_l3014_301499

theorem perpendicular_lines_coefficient (a : ℝ) : 
  (∀ x y : ℝ, ax + y - 1 = 0 ↔ x + ay - 1 = 0) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_coefficient_l3014_301499


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_gcf_of_145_and_30_less_than_150_is_greatest_main_result_l3014_301479

theorem greatest_integer_with_gcf_five (n : ℕ) : n < 150 ∧ Nat.gcd n 30 = 5 → n ≤ 145 :=
by sorry

theorem gcf_of_145_and_30 : Nat.gcd 145 30 = 5 :=
by sorry

theorem less_than_150 : 145 < 150 :=
by sorry

theorem is_greatest : ∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ 145 :=
by sorry

theorem main_result : (∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ 
  (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n)) ∧ 
  (∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ 
  (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) ∧ n = 145) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_gcf_of_145_and_30_less_than_150_is_greatest_main_result_l3014_301479


namespace NUMINAMATH_CALUDE_bowling_record_proof_l3014_301466

/-- The old record average score per player per round in a bowling league -/
def old_record : ℝ := 287

/-- Number of players in a team -/
def players_per_team : ℕ := 4

/-- Number of rounds in a season -/
def rounds_per_season : ℕ := 10

/-- Total score of the team after 9 rounds -/
def score_after_nine_rounds : ℕ := 10440

/-- Difference between old record and minimum average needed in final round -/
def score_difference : ℕ := 27

theorem bowling_record_proof :
  old_record = 
    (score_after_nine_rounds + players_per_team * (old_record - score_difference)) / 
    (players_per_team * rounds_per_season) := by
  sorry

end NUMINAMATH_CALUDE_bowling_record_proof_l3014_301466


namespace NUMINAMATH_CALUDE_composition_of_functions_l3014_301439

theorem composition_of_functions (f g : ℝ → ℝ) :
  (∀ x, f x = 5 - 2 * x) →
  (∀ x, g x = x^2 + x + 1) →
  f (g (Real.sqrt 3)) = -3 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_composition_of_functions_l3014_301439


namespace NUMINAMATH_CALUDE_total_students_is_150_l3014_301462

/-- Proves that the total number of students is 150 given the conditions -/
theorem total_students_is_150 
  (total : ℕ) 
  (boys : ℕ) 
  (girls : ℕ) 
  (h1 : total = boys + girls) 
  (h2 : boys = 60 → girls = (60 * total) / 100) : 
  total = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_150_l3014_301462


namespace NUMINAMATH_CALUDE_john_shoe_purchase_cost_l3014_301450

/-- Calculate the total cost including tax for two items -/
def total_cost (price1 : ℝ) (price2 : ℝ) (tax_rate : ℝ) : ℝ :=
  let total_before_tax := price1 + price2
  let tax_amount := total_before_tax * tax_rate
  total_before_tax + tax_amount

/-- Theorem stating the total cost for the given problem -/
theorem john_shoe_purchase_cost :
  total_cost 150 120 0.1 = 297 := by
  sorry

end NUMINAMATH_CALUDE_john_shoe_purchase_cost_l3014_301450


namespace NUMINAMATH_CALUDE_complex_number_conditions_complex_number_on_line_l3014_301441

def z (a : ℝ) : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2)

theorem complex_number_conditions (a : ℝ) :
  (z a).re < 0 ∧ (z a).im > 0 ↔ -2 < a ∧ a < 1 :=
sorry

theorem complex_number_on_line (a : ℝ) :
  (z a).re = (z a).im ↔ a = 1 :=
sorry

end NUMINAMATH_CALUDE_complex_number_conditions_complex_number_on_line_l3014_301441


namespace NUMINAMATH_CALUDE_triangle_circle_inequality_l3014_301408

/-- Given a triangle ABC with semiperimeter p and incircle radius r, and a circle Γ
    with radius t tangent to the semicircles constructed on the sides of the triangle,
    prove that p/2 < t ≤ p/2 + (1 - √3/2) * r. -/
theorem triangle_circle_inequality (p r t : ℝ) (hp : p > 0) (hr : r > 0) (ht : t > 0) :
  p / 2 < t ∧ t ≤ p / 2 + (1 - Real.sqrt 3 / 2) * r := by
  sorry

end NUMINAMATH_CALUDE_triangle_circle_inequality_l3014_301408


namespace NUMINAMATH_CALUDE_meaningful_expression_l3014_301474

theorem meaningful_expression (a : ℝ) : 
  (∃ x : ℝ, x = (a + 3) / Real.sqrt (a - 3)) ↔ a > 3 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3014_301474


namespace NUMINAMATH_CALUDE_mike_earnings_l3014_301495

def mower_blade_cost : ℕ := 10
def game_cost : ℕ := 8
def number_of_games : ℕ := 4

def total_money_earned : ℕ :=
  mower_blade_cost + number_of_games * game_cost

theorem mike_earnings : total_money_earned = 42 := by
  sorry

end NUMINAMATH_CALUDE_mike_earnings_l3014_301495


namespace NUMINAMATH_CALUDE_unique_prime_in_set_l3014_301471

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def six_digit_number (B : ℕ) : ℕ := 303200 + B

theorem unique_prime_in_set :
  ∃! B : ℕ, B ≤ 9 ∧ is_prime (six_digit_number B) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_in_set_l3014_301471


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3014_301409

theorem polynomial_simplification (x : ℝ) : 
  (3*x^3 - x^2 + 2*x + 9)*(2*x + 1) - (2*x + 1)*(x^3 - 3*x^2 + 16) + (4*x - 15)*(2*x + 1)*(x - 3) = 
  2*x^4 + 6*x^3 - 21*x^2 - 12*x + 76 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3014_301409


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3014_301405

theorem intersection_of_sets : ∀ (A B : Set ℕ),
  A = {1, 2, 3, 4, 5} →
  B = {3, 5} →
  A ∩ B = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3014_301405


namespace NUMINAMATH_CALUDE_S_has_maximum_l3014_301458

def S (n : ℕ+) : ℤ := -2 * n.val ^ 3 + 21 * n.val ^ 2 + 23 * n.val

theorem S_has_maximum : ∃ (m : ℕ+), ∀ (n : ℕ+), S n ≤ S m ∧ S m = 504 := by
  sorry

end NUMINAMATH_CALUDE_S_has_maximum_l3014_301458


namespace NUMINAMATH_CALUDE_bella_friends_count_l3014_301427

/-- The number of beads needed per bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of beads Bella currently has -/
def current_beads : ℕ := 36

/-- The number of additional beads Bella needs -/
def additional_beads : ℕ := 12

/-- The number of friends Bella is making bracelets for -/
def num_friends : ℕ := (current_beads + additional_beads) / beads_per_bracelet

theorem bella_friends_count : num_friends = 6 := by
  sorry

end NUMINAMATH_CALUDE_bella_friends_count_l3014_301427


namespace NUMINAMATH_CALUDE_total_buttons_is_1600_l3014_301496

/-- The number of 3-button shirts ordered -/
def shirts_3_button : ℕ := 200

/-- The number of 5-button shirts ordered -/
def shirts_5_button : ℕ := 200

/-- The number of buttons on a 3-button shirt -/
def buttons_per_3_button_shirt : ℕ := 3

/-- The number of buttons on a 5-button shirt -/
def buttons_per_5_button_shirt : ℕ := 5

/-- The total number of buttons used for all shirts -/
def total_buttons : ℕ := shirts_3_button * buttons_per_3_button_shirt + 
                          shirts_5_button * buttons_per_5_button_shirt

theorem total_buttons_is_1600 : total_buttons = 1600 := by
  sorry

end NUMINAMATH_CALUDE_total_buttons_is_1600_l3014_301496


namespace NUMINAMATH_CALUDE_abs_x_minus_one_leq_one_iff_x_leq_two_l3014_301415

theorem abs_x_minus_one_leq_one_iff_x_leq_two :
  ∀ x : ℝ, |x - 1| ≤ 1 ↔ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_leq_one_iff_x_leq_two_l3014_301415


namespace NUMINAMATH_CALUDE_total_earnings_theorem_l3014_301482

/-- Represents the investment and return ratios for three investors -/
structure InvestmentData where
  investment_ratio : Fin 3 → ℕ
  return_ratio : Fin 3 → ℕ

/-- Calculates the total earnings given investment data and the earnings difference between two investors -/
def calculate_total_earnings (data : InvestmentData) (earnings_diff : ℕ) : ℕ := sorry

/-- The main theorem stating the total earnings for the given scenario -/
theorem total_earnings_theorem (data : InvestmentData) (earnings_diff : ℕ) : 
  data.investment_ratio 0 = 3 ∧ 
  data.investment_ratio 1 = 4 ∧ 
  data.investment_ratio 2 = 5 ∧
  data.return_ratio 0 = 6 ∧ 
  data.return_ratio 1 = 5 ∧ 
  data.return_ratio 2 = 4 ∧
  earnings_diff = 120 →
  calculate_total_earnings data earnings_diff = 3480 := by sorry

end NUMINAMATH_CALUDE_total_earnings_theorem_l3014_301482


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l3014_301470

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis (p : Point) :
  p.x = -3 ∧ p.y = 2 → (reflect_x p).x = -3 ∧ (reflect_x p).y = -2 := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l3014_301470


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3014_301411

/-- A quadratic function with vertex form (x + h)^2 passing through a specific point -/
def QuadraticFunction (a : ℝ) (h : ℝ) (x₀ : ℝ) (y₀ : ℝ) : Prop :=
  y₀ = a * (x₀ + h)^2

theorem quadratic_coefficient (a : ℝ) :
  QuadraticFunction a 3 2 (-50) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3014_301411


namespace NUMINAMATH_CALUDE_one_fifth_greater_than_decimal_l3014_301424

theorem one_fifth_greater_than_decimal : 1/5 = 0.20000001 + 1/(5*10^8) := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_greater_than_decimal_l3014_301424


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3014_301454

def U : Set ℝ := Set.univ

def A : Set ℝ := {-3, -1, 0, 1, 3}

def B : Set ℝ := {x | |x - 1| > 1}

theorem intersection_A_complement_B : A ∩ (U \ B) = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3014_301454


namespace NUMINAMATH_CALUDE_possible_values_of_a_minus_b_l3014_301435

theorem possible_values_of_a_minus_b (a b : ℝ) (ha : |a| = 7) (hb : |b| = 5) :
  {x | ∃ (a' b' : ℝ), |a'| = 7 ∧ |b'| = 5 ∧ x = a' - b'} = {2, 12, -12, -2} := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_minus_b_l3014_301435


namespace NUMINAMATH_CALUDE_negative_three_squared_l3014_301490

theorem negative_three_squared : (-3 : ℤ)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_squared_l3014_301490


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3014_301498

theorem least_subtraction_for_divisibility :
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), y < x → ¬(12 ∣ (427398 - y))) ∧
  (12 ∣ (427398 - x)) := by
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3014_301498


namespace NUMINAMATH_CALUDE_problem_solution_l3014_301484

def A (a : ℝ) : Set ℝ := {x | a - 2 < x ∧ x < a + 2}

def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 2) * x + 2 * a = 0}

theorem problem_solution :
  (∀ x, x ∈ (A 0 ∪ B 0) ↔ -2 < x ∧ x ≤ 2) ∧
  (∀ a, (Aᶜ a ∩ B a).Nonempty ↔ a ≤ 0 ∨ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3014_301484


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3014_301488

theorem vector_difference_magnitude (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 1) 
  (h3 : ‖a + b‖ = ‖a - b‖) : 
  ‖a - b‖ = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3014_301488


namespace NUMINAMATH_CALUDE_volume_of_sphere_wedge_l3014_301429

/-- The volume of a wedge from a sphere cut into six congruent parts, given the sphere's circumference --/
theorem volume_of_sphere_wedge (circumference : ℝ) :
  circumference = 18 * Real.pi →
  (1 / 6) * (4 / 3) * Real.pi * (circumference / (2 * Real.pi))^3 = 162 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_volume_of_sphere_wedge_l3014_301429


namespace NUMINAMATH_CALUDE_cindy_marbles_problem_l3014_301431

theorem cindy_marbles_problem (initial_marbles : ℕ) (num_friends : ℕ) (marbles_per_friend : ℕ) :
  initial_marbles = 1000 →
  num_friends = 6 →
  marbles_per_friend = 120 →
  7 * (initial_marbles - num_friends * marbles_per_friend) = 1960 :=
by
  sorry

end NUMINAMATH_CALUDE_cindy_marbles_problem_l3014_301431


namespace NUMINAMATH_CALUDE_system_solution_l3014_301421

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℚ),
    (x₁^2 - 9*y₁^2 = 36 ∧ 3*x₁ + y₁ = 6) ∧
    (x₂^2 - 9*y₂^2 = 36 ∧ 3*x₂ + y₂ = 6) ∧
    x₁ = 12/5 ∧ y₁ = -6/5 ∧ x₂ = 3 ∧ y₂ = -3 ∧
    ∀ (x y : ℚ), (x^2 - 9*y^2 = 36 ∧ 3*x + y = 6) → ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l3014_301421


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3014_301494

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 - 8 * x + 3
  ∃ x₁ x₂ : ℝ, x₁ = 2 + (Real.sqrt 10) / 2 ∧
              x₂ = 2 - (Real.sqrt 10) / 2 ∧
              f x₁ = 0 ∧ f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_equation_solution_l3014_301494


namespace NUMINAMATH_CALUDE_johnson_family_seating_l3014_301417

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def total_seatings (n : ℕ) : ℕ := factorial n

def boys_not_adjacent (boys girls : ℕ) : ℕ := 
  2 * factorial boys * factorial girls

theorem johnson_family_seating (boys girls : ℕ) : 
  boys = 5 → girls = 4 → 
  total_seatings (boys + girls) - boys_not_adjacent boys girls = 357120 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l3014_301417


namespace NUMINAMATH_CALUDE_concert_attendance_difference_l3014_301446

theorem concert_attendance_difference (first_concert : Nat) (second_concert : Nat)
  (h1 : first_concert = 65899)
  (h2 : second_concert = 66018) :
  second_concert - first_concert = 119 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_difference_l3014_301446


namespace NUMINAMATH_CALUDE_kids_played_correct_l3014_301407

/-- The number of kids Julia played with on each day --/
structure KidsPlayed where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- The conditions of the problem --/
def satisfiesConditions (k : KidsPlayed) : Prop :=
  k.tuesday = 14 ∧
  k.wednesday = k.tuesday + (k.tuesday / 4 + 1) ∧
  k.thursday = 2 * k.wednesday - 4 ∧
  k.monday = k.tuesday + 8

/-- The theorem to prove --/
theorem kids_played_correct : 
  ∃ (k : KidsPlayed), satisfiesConditions k ∧ 
    k.monday = 22 ∧ k.tuesday = 14 ∧ k.wednesday = 18 ∧ k.thursday = 32 := by
  sorry

end NUMINAMATH_CALUDE_kids_played_correct_l3014_301407


namespace NUMINAMATH_CALUDE_count_divisible_by_3_5_7_60_l3014_301442

def count_divisible (n : ℕ) (d : ℕ) : ℕ := n / d

def count_divisible_by_3_5_7 (upper_bound : ℕ) : ℕ :=
  let div3 := count_divisible upper_bound 3
  let div5 := count_divisible upper_bound 5
  let div7 := count_divisible upper_bound 7
  let div3_5 := count_divisible upper_bound 15
  let div3_7 := count_divisible upper_bound 21
  let div5_7 := count_divisible upper_bound 35
  div3 + div5 + div7 - (div3_5 + div3_7 + div5_7)

theorem count_divisible_by_3_5_7_60 : count_divisible_by_3_5_7 60 = 33 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_by_3_5_7_60_l3014_301442


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l3014_301451

theorem cosine_sine_identity (α : Real) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6)^2 = -(Real.sqrt 3 + 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l3014_301451


namespace NUMINAMATH_CALUDE_five_regular_polyhedra_l3014_301455

/-- A convex regular polyhedron with n edges meeting at each vertex and k sides on each face. -/
structure ConvexRegularPolyhedron where
  n : ℕ
  k : ℕ
  n_ge_three : n ≥ 3
  k_ge_three : k ≥ 3

/-- The inequality that must be satisfied by a convex regular polyhedron. -/
def satisfies_inequality (p : ConvexRegularPolyhedron) : Prop :=
  (1 : ℚ) / p.n + (1 : ℚ) / p.k > (1 : ℚ) / 2

/-- The theorem stating that there are only five types of convex regular polyhedra. -/
theorem five_regular_polyhedra :
  ∀ p : ConvexRegularPolyhedron, satisfies_inequality p ↔
    (p.n = 3 ∧ p.k = 3) ∨
    (p.n = 3 ∧ p.k = 4) ∨
    (p.n = 3 ∧ p.k = 5) ∨
    (p.n = 4 ∧ p.k = 3) ∨
    (p.n = 5 ∧ p.k = 3) :=
by sorry

end NUMINAMATH_CALUDE_five_regular_polyhedra_l3014_301455


namespace NUMINAMATH_CALUDE_book_cost_l3014_301430

theorem book_cost (book_cost bookmark_cost : ℚ) 
  (total_cost : book_cost + bookmark_cost = (7/2 : ℚ))
  (price_difference : book_cost = bookmark_cost + 3) : 
  book_cost = (13/4 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_book_cost_l3014_301430


namespace NUMINAMATH_CALUDE_job_completion_time_l3014_301472

/-- Represents the time (in hours) it takes for a single machine to complete the job -/
def single_machine_time : ℝ := 216

/-- Represents the number of machines of each type used -/
def machines_per_type : ℕ := 9

/-- Represents the time (in hours) it takes for all machines working together to complete the job -/
def total_job_time : ℝ := 12

theorem job_completion_time :
  (((1 / single_machine_time) * machines_per_type + 
    (1 / single_machine_time) * machines_per_type) * total_job_time = 1) →
  single_machine_time = 216 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l3014_301472


namespace NUMINAMATH_CALUDE_crystal_meal_options_l3014_301480

/-- The number of distinct possible meals given the number of options for each category --/
def distinct_meals (entrees drinks desserts : ℕ) : ℕ :=
  entrees * drinks * desserts

/-- Theorem: Given 4 entree options, 4 drink options, and 2 dessert options,
    the number of distinct possible meals is 32 --/
theorem crystal_meal_options :
  distinct_meals 4 4 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_crystal_meal_options_l3014_301480


namespace NUMINAMATH_CALUDE_sequence_properties_l3014_301449

def is_root (a : ℝ) : Prop := a^2 - 3*a - 5 = 0

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m

theorem sequence_properties (a : ℕ → ℝ) :
  (is_root (a 3) ∧ is_root (a 10) ∧ arithmetic_sequence a → a 5 + a 8 = 3) ∧
  (is_root (a 3) ∧ is_root (a 10) ∧ geometric_sequence a → a 6 * a 7 = -5) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l3014_301449


namespace NUMINAMATH_CALUDE_sqrt_two_four_three_two_five_two_l3014_301436

theorem sqrt_two_four_three_two_five_two : Real.sqrt (2^4 * 3^2 * 5^2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_four_three_two_five_two_l3014_301436


namespace NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l3014_301445

/-- The width of a rectangle whose area is equal to the area of a triangle with base 16 and height equal to the rectangle's length -/
theorem rectangle_triangle_equal_area (x : ℝ) (y : ℝ) 
  (h : x * y = (1/2) * 16 * x) : y = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l3014_301445


namespace NUMINAMATH_CALUDE_three_digit_subtraction_l3014_301401

/-- Given a three-digit number of the form 4c1 and another of the form 3d5,
    prove that if 786 - 4c1 = 3d5 and 3d5 is divisible by 7, then c + d = 8 -/
theorem three_digit_subtraction (c d : ℕ) : 
  (786 - (400 + c * 10 + 1) = 300 + d * 10 + 5) →
  (300 + d * 10 + 5) % 7 = 0 →
  c + d = 8 := by
sorry

end NUMINAMATH_CALUDE_three_digit_subtraction_l3014_301401


namespace NUMINAMATH_CALUDE_land_area_decreases_l3014_301428

theorem land_area_decreases (a : ℝ) (h : a > 4) : a^2 > (a+4)*(a-4) := by
  sorry

end NUMINAMATH_CALUDE_land_area_decreases_l3014_301428


namespace NUMINAMATH_CALUDE_equal_area_split_line_slope_l3014_301434

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  passesThrough : ℝ × ℝ

/-- Checks if a line splits the area of circles equally -/
def splitAreaEqually (line : Line) (circles : List Circle) : Prop :=
  sorry

/-- The main theorem -/
theorem equal_area_split_line_slope :
  let circles : List Circle := [
    { center := (10, 100), radius := 4 },
    { center := (13, 82),  radius := 4 },
    { center := (15, 90),  radius := 4 }
  ]
  let line : Line := { slope := 0.5, passesThrough := (13, 82) }
  splitAreaEqually line circles ∧ 
  ∀ (m : ℝ), splitAreaEqually { slope := m, passesThrough := (13, 82) } circles → 
    |m| = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_split_line_slope_l3014_301434


namespace NUMINAMATH_CALUDE_intersection_sum_l3014_301412

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 5*x < 0}
def N (p : ℝ) : Set ℝ := {x | p < x ∧ x < 6}

-- Define the theorem
theorem intersection_sum (p q : ℝ) : 
  M ∩ N p = {x | 2 < x ∧ x < q} → p + q = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l3014_301412


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l3014_301422

/-- The speed of the stream in mph -/
def stream_speed : ℝ := 3.5

/-- The speed of the boat in still water in mph -/
def boat_speed : ℝ := 15

/-- The distance traveled in miles -/
def distance : ℝ := 60

/-- The time difference between upstream and downstream trips in hours -/
def time_difference : ℝ := 2

theorem stream_speed_calculation :
  (distance / (boat_speed - stream_speed)) - (distance / (boat_speed + stream_speed)) = time_difference :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_calculation_l3014_301422


namespace NUMINAMATH_CALUDE_acrobats_count_l3014_301423

/-- The number of acrobats in a parade group -/
def num_acrobats : ℕ := 10

/-- The number of elephants in a parade group -/
def num_elephants : ℕ := 20 - num_acrobats

/-- The total number of legs in the parade group -/
def total_legs : ℕ := 60

/-- The total number of heads in the parade group -/
def total_heads : ℕ := 20

/-- Theorem stating that the number of acrobats is 10 given the conditions -/
theorem acrobats_count :
  (2 * num_acrobats + 4 * num_elephants = total_legs) ∧
  (num_acrobats + num_elephants = total_heads) ∧
  (num_acrobats = 10) := by
  sorry

end NUMINAMATH_CALUDE_acrobats_count_l3014_301423


namespace NUMINAMATH_CALUDE_industrial_lubricants_percentage_l3014_301460

theorem industrial_lubricants_percentage 
  (microphotonics : ℝ) 
  (home_electronics : ℝ) 
  (food_additives : ℝ) 
  (genetically_modified_microorganisms : ℝ) 
  (basic_astrophysics_degrees : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 20 →
  genetically_modified_microorganisms = 29 →
  basic_astrophysics_degrees = 18 →
  let basic_astrophysics := (basic_astrophysics_degrees / 360) * 100
  let total_known := microphotonics + home_electronics + food_additives + 
                     genetically_modified_microorganisms + basic_astrophysics
  let industrial_lubricants := 100 - total_known
  industrial_lubricants = 8 :=
by sorry

end NUMINAMATH_CALUDE_industrial_lubricants_percentage_l3014_301460


namespace NUMINAMATH_CALUDE_rectangle_perimeter_area_inequality_l3014_301475

theorem rectangle_perimeter_area_inequality (l S : ℝ) (hl : l > 0) (hS : S > 0) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ l = 2 * (a + b) ∧ S = a * b) → l^2 ≥ 16 * S :=
by sorry

#check rectangle_perimeter_area_inequality

end NUMINAMATH_CALUDE_rectangle_perimeter_area_inequality_l3014_301475


namespace NUMINAMATH_CALUDE_inequality_range_l3014_301487

theorem inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1) 2, x^2 - a*x - 3 < 0) → 
  a ∈ Set.Ioo (1/2) 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3014_301487


namespace NUMINAMATH_CALUDE_complex_number_equality_l3014_301483

theorem complex_number_equality (z : ℂ) (h : z * (1 - Complex.I) = 2) : z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3014_301483


namespace NUMINAMATH_CALUDE_non_trivial_solution_exists_l3014_301457

theorem non_trivial_solution_exists (a b c : ℤ) (p : ℕ) (hp : Nat.Prime p) :
  ∃ x y z : ℤ, (x, y, z) ≠ (0, 0, 0) ∧ (a * x^2 + b * y^2 + c * z^2) % p = 0 := by
  sorry

end NUMINAMATH_CALUDE_non_trivial_solution_exists_l3014_301457


namespace NUMINAMATH_CALUDE_tan_cos_tan_equality_l3014_301459

theorem tan_cos_tan_equality : Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_cos_tan_equality_l3014_301459


namespace NUMINAMATH_CALUDE_carlos_won_one_game_l3014_301416

/-- Represents a chess player in the tournament -/
structure Player where
  wins : ℕ
  losses : ℕ

/-- Represents the chess tournament -/
structure Tournament where
  laura : Player
  mike : Player
  carlos : Player
  total_games : ℕ

/-- The number of games Carlos won in the tournament -/
def carlos_wins (t : Tournament) : ℕ :=
  t.total_games - (t.laura.wins + t.laura.losses + t.mike.wins + t.mike.losses + t.carlos.losses)

theorem carlos_won_one_game (t : Tournament) 
  (h1 : t.laura.wins = 5)
  (h2 : t.laura.losses = 4)
  (h3 : t.mike.wins = 7)
  (h4 : t.mike.losses = 2)
  (h5 : t.carlos.losses = 5)
  (h6 : t.total_games = (t.laura.wins + t.laura.losses + t.mike.wins + t.mike.losses + t.carlos.losses + carlos_wins t) / 2) :
  carlos_wins t = 1 := by
  sorry

end NUMINAMATH_CALUDE_carlos_won_one_game_l3014_301416


namespace NUMINAMATH_CALUDE_parabola_intersection_range_l3014_301400

/-- A parabola defined by y = ax^2 -/
def Parabola (a : ℝ) : ℝ → ℝ := λ x ↦ a * x^2

/-- A line passing through (1, -2) with slope k -/
def Line (k : ℝ) : ℝ → ℝ := λ x ↦ k * (x - 1) - 2

/-- Checks if a parabola intersects a line -/
def intersects (p : ℝ → ℝ) (l : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, p x = l x

theorem parabola_intersection_range (a : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ (intersects (Parabola a) (Line k) ∨ intersects (Parabola a) (Line (-1/k)))) →
  a < 0 ∨ (0 < a ∧ a ≤ 1/8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_range_l3014_301400


namespace NUMINAMATH_CALUDE_f_properties_l3014_301425

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x^2 + 1) + x - 1) / (Real.sqrt (x^2 + 1) + x + 1)

theorem f_properties :
  (∀ x : ℝ, f (-x) ≠ -f x) ∧
  (∀ x : ℝ, ∃ y : ℝ, f x = y) ∧
  (∀ y : ℝ, -1 < y ∧ y < 1 ↔ ∃ x : ℝ, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3014_301425


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l3014_301481

theorem binomial_expansion_sum (n : ℕ) : 
  (∀ k : ℕ, k ≠ 2 → Nat.choose n 2 > Nat.choose n k) → 
  (1 - 2)^n = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l3014_301481


namespace NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l3014_301426

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / x + a) * Real.log (1 + x)

theorem tangent_line_and_monotonicity (a : ℝ) :
  (∀ x > 0, ∀ y > 0, x ≠ y → (f (-1) x - f (-1) y) / (x - y) = Real.log 2 * x + f (-1) x - Real.log 2) ∧
  (∀ x > 0, Monotone (f a) ↔ a ≥ (1 : ℝ) / 2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l3014_301426


namespace NUMINAMATH_CALUDE_f_at_neg_point_two_eq_approx_l3014_301406

/-- Horner's algorithm for polynomial evaluation -/
def horner (coeffs : List Float) (x : Float) : Float :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial function f(x) -/
def f (x : Float) : Float :=
  horner [1, 1, 0.5, 0.16667, 0.04167, 0.00833] x

/-- Theorem stating that f(-0.2) equals 0.81873 (approximately) -/
theorem f_at_neg_point_two_eq_approx :
  (f (-0.2) - 0.81873).abs < 1e-5 := by
  sorry

#eval f (-0.2)

end NUMINAMATH_CALUDE_f_at_neg_point_two_eq_approx_l3014_301406


namespace NUMINAMATH_CALUDE_inverse_proportion_l3014_301418

/-- Given that x is inversely proportional to y, if x = 4 when y = -2, 
    then x = 4/5 when y = -10 -/
theorem inverse_proportion (x y : ℝ) (h : ∃ k : ℝ, ∀ x y, x * y = k) 
  (h1 : 4 * (-2) = x * y) : 
  x * (-10) = 4/5 * (-10) := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_l3014_301418


namespace NUMINAMATH_CALUDE_determinant_of_special_matrix_l3014_301447

theorem determinant_of_special_matrix (y : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := 
    ![![2*y + 1, 2*y, 2*y],
      ![2*y, 2*y + 1, 2*y],
      ![2*y, 2*y, 2*y + 1]]
  Matrix.det A = 6*y + 1 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_special_matrix_l3014_301447


namespace NUMINAMATH_CALUDE_plane_equation_proof_l3014_301413

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by its direction ratios -/
structure Line3D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A plane in 3D space defined by its equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

/-- Check if a point lies on a plane -/
def point_on_plane (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- Check if a line is contained in a plane -/
def line_in_plane (l : Line3D) (plane : Plane) : Prop :=
  plane.A * l.a + plane.B * l.b + plane.C * l.c = 0

/-- The main theorem -/
theorem plane_equation_proof (p : Point3D) (l : Line3D) :
  p = Point3D.mk 0 7 (-7) →
  l = Line3D.mk (-3) 2 1 →
  let plane := Plane.mk 1 1 1 0
  point_on_plane p plane ∧ line_in_plane l plane := by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l3014_301413


namespace NUMINAMATH_CALUDE_subtraction_decimal_result_l3014_301477

theorem subtraction_decimal_result : 5.3567 - 2.1456 - 1.0211 = 2.1900 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_decimal_result_l3014_301477


namespace NUMINAMATH_CALUDE_season_games_calculation_l3014_301463

/-- Represents the number of games in a basketball season -/
def SeasonGames : ℕ := 20

/-- Represents Donovan Mitchell's current average points per game -/
def CurrentAverage : ℚ := 26

/-- Represents the number of games played so far -/
def GamesPlayed : ℕ := 15

/-- Represents Donovan Mitchell's goal average for the entire season -/
def GoalAverage : ℚ := 30

/-- Represents the required average for the remaining games -/
def RequiredAverage : ℚ := 42

theorem season_games_calculation :
  CurrentAverage * GamesPlayed + RequiredAverage * (SeasonGames - GamesPlayed) =
  GoalAverage * SeasonGames := by
  sorry

#check season_games_calculation

end NUMINAMATH_CALUDE_season_games_calculation_l3014_301463


namespace NUMINAMATH_CALUDE_error_percentage_calculation_l3014_301420

theorem error_percentage_calculation (x : ℝ) (h : x > 0) :
  let correct_result := x + 5
  let erroneous_result := x - 5
  let error := abs (correct_result - erroneous_result)
  let error_percentage := (error / correct_result) * 100
  error_percentage = (10 / (x + 5)) * 100 := by sorry

end NUMINAMATH_CALUDE_error_percentage_calculation_l3014_301420


namespace NUMINAMATH_CALUDE_factorization_expression1_l3014_301473

theorem factorization_expression1 (x y : ℝ) : 2 * x^2 * y - 4 * x * y + 2 * y = 2 * y * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_expression1_l3014_301473


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_rectangle_area_is_588_l3014_301467

/-- The area of a rectangle with an inscribed circle of radius 7 and length-to-width ratio of 3:1 -/
theorem rectangle_area_with_inscribed_circle : ℝ :=
  let radius : ℝ := 7
  let length_width_ratio : ℝ := 3
  let diameter : ℝ := 2 * radius
  let width : ℝ := diameter
  let length : ℝ := length_width_ratio * width
  let area : ℝ := length * width
  area

/-- Proof that the area of the rectangle is 588 -/
theorem rectangle_area_is_588 : rectangle_area_with_inscribed_circle = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_rectangle_area_is_588_l3014_301467


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_1728_l3014_301404

theorem modular_inverse_13_mod_1728 : ∃ x : ℕ, x < 1728 ∧ (13 * x) % 1728 = 1 :=
by
  use 133
  sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_1728_l3014_301404


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_U_l3014_301465

def U : Set Nat := {1, 2, 3}
def A : Set Nat := {1, 2}

theorem complement_of_A_wrt_U : (U \ A) = {3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_U_l3014_301465


namespace NUMINAMATH_CALUDE_mateo_net_salary_l3014_301440

def weekly_salary : ℝ := 791

def absence_deduction (day : ℕ) : ℝ :=
  match day with
  | 1 => 0.01 * weekly_salary
  | 2 => 0.02 * weekly_salary
  | 3 => 0.03 * weekly_salary
  | 4 => 0.04 * weekly_salary
  | _ => 0

def total_absence_deduction : ℝ :=
  (absence_deduction 1) + (absence_deduction 2) + (absence_deduction 3) + (absence_deduction 4)

def salary_after_deductions : ℝ :=
  weekly_salary - total_absence_deduction

def income_tax_rate : ℝ := 0.07

def income_tax : ℝ :=
  income_tax_rate * salary_after_deductions

def net_salary : ℝ :=
  salary_after_deductions - income_tax

theorem mateo_net_salary : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |net_salary - 662.07| < ε :=
sorry

end NUMINAMATH_CALUDE_mateo_net_salary_l3014_301440


namespace NUMINAMATH_CALUDE_number_of_browsers_l3014_301464

/-- Given that each browser has 3 windows, each window has 10 tabs,
    and the total number of tabs in all browsers is 60,
    prove that the number of browsers is 2. -/
theorem number_of_browsers :
  ∀ (num_browsers : ℕ),
    (∀ (browser : ℕ), browser ≤ num_browsers → 
      ∃ (windows : ℕ), windows = 3 ∧
        ∃ (tabs_per_window : ℕ), tabs_per_window = 10) →
    (num_browsers * 3 * 10 = 60) →
    num_browsers = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_browsers_l3014_301464


namespace NUMINAMATH_CALUDE_eccentricity_of_ellipse_through_roots_l3014_301478

-- Define the complex equation
def complex_equation (z : ℂ) : Prop :=
  (z - 2) * (z^2 + 3*z + 5) * (z^2 + 5*z + 8) = 0

-- Define the set of roots
def roots : Set ℂ :=
  {z : ℂ | complex_equation z}

-- Define the ellipse passing through the roots
def ellipse_through_roots (E : Set (ℝ × ℝ)) : Prop :=
  ∀ z ∈ roots, (z.re, z.im) ∈ E

-- Define the eccentricity of an ellipse
def eccentricity (E : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem eccentricity_of_ellipse_through_roots :
  ∃ E : Set (ℝ × ℝ), ellipse_through_roots E ∧ eccentricity E = Real.sqrt (1/7) :=
sorry

end NUMINAMATH_CALUDE_eccentricity_of_ellipse_through_roots_l3014_301478


namespace NUMINAMATH_CALUDE_johns_pre_raise_earnings_l3014_301444

/-- The amount John makes per week after the raise, in dollars. -/
def post_raise_earnings : ℝ := 60

/-- The percentage increase of John's earnings. -/
def percentage_increase : ℝ := 50

/-- John's weekly earnings before the raise, in dollars. -/
def pre_raise_earnings : ℝ := 40

/-- Theorem stating that John's pre-raise earnings were $40, given the conditions. -/
theorem johns_pre_raise_earnings : 
  pre_raise_earnings * (1 + percentage_increase / 100) = post_raise_earnings := by
  sorry

end NUMINAMATH_CALUDE_johns_pre_raise_earnings_l3014_301444


namespace NUMINAMATH_CALUDE_only_math_is_75_l3014_301419

/-- Represents the number of students in different subject combinations -/
structure StudentCounts where
  total : ℕ
  math : ℕ
  foreignLanguage : ℕ
  science : ℕ
  allThree : ℕ

/-- The actual student counts from the problem -/
def actualCounts : StudentCounts :=
  { total := 120
  , math := 85
  , foreignLanguage := 65
  , science := 75
  , allThree := 20 }

/-- Calculate the number of students taking only math -/
def onlyMathCount (counts : StudentCounts) : ℕ :=
  counts.math - (counts.total - (counts.math + counts.foreignLanguage + counts.science - counts.allThree))

/-- Theorem stating that the number of students taking only math is 75 -/
theorem only_math_is_75 : onlyMathCount actualCounts = 75 := by
  sorry

end NUMINAMATH_CALUDE_only_math_is_75_l3014_301419


namespace NUMINAMATH_CALUDE_february_first_day_l3014_301476

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

-- Define a function to advance a given number of days
def advanceDays (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | n + 1 => nextDay (advanceDays start n)

-- Theorem statement
theorem february_first_day (january_first : DayOfWeek) 
  (h : january_first = DayOfWeek.Monday) : 
  advanceDays january_first 31 = DayOfWeek.Thursday := by
  sorry


end NUMINAMATH_CALUDE_february_first_day_l3014_301476


namespace NUMINAMATH_CALUDE_trigonometric_equalities_l3014_301452

theorem trigonometric_equalities :
  (6 * (Real.tan (30 * π / 180))^2 - Real.sqrt 3 * Real.sin (60 * π / 180) - 2 * Real.sin (45 * π / 180) = 1/2 - Real.sqrt 2) ∧
  (Real.sqrt 2 / 2 * Real.cos (45 * π / 180) - (Real.tan (40 * π / 180) + 1)^0 + Real.sqrt (1/4) + Real.sin (30 * π / 180) = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equalities_l3014_301452


namespace NUMINAMATH_CALUDE_train_speed_problem_l3014_301433

/-- The speed of train B given the conditions of the problem -/
theorem train_speed_problem (length_A length_B : ℝ) (speed_A : ℝ) (crossing_time : ℝ) :
  length_A = 150 ∧ 
  length_B = 150 ∧ 
  speed_A = 54 ∧ 
  crossing_time = 12 →
  (length_A + length_B) / crossing_time * 3.6 - speed_A = 36 := by
  sorry

#check train_speed_problem

end NUMINAMATH_CALUDE_train_speed_problem_l3014_301433


namespace NUMINAMATH_CALUDE_laura_change_l3014_301485

/-- The change Laura received after purchasing pants and shirts -/
theorem laura_change (pants_cost shirt_cost : ℕ) (pants_quantity shirt_quantity : ℕ) (amount_given : ℕ) : 
  pants_cost = 54 → 
  pants_quantity = 2 → 
  shirt_cost = 33 → 
  shirt_quantity = 4 → 
  amount_given = 250 → 
  amount_given - (pants_cost * pants_quantity + shirt_cost * shirt_quantity) = 10 := by
  sorry

end NUMINAMATH_CALUDE_laura_change_l3014_301485


namespace NUMINAMATH_CALUDE_pythagorean_triple_with_8_and_17_l3014_301448

/-- A Pythagorean triple is a set of three positive integers (a, b, c) such that a² + b² = c² -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- Given a set of Pythagorean triples containing 8 and 17, the third number is 15 -/
theorem pythagorean_triple_with_8_and_17 :
  ∃ (x : ℕ), (is_pythagorean_triple 8 15 17 ∨ is_pythagorean_triple 8 17 15 ∨ is_pythagorean_triple 15 8 17) ∧
  ¬∃ (y : ℕ), y ≠ 15 ∧ (is_pythagorean_triple 8 y 17 ∨ is_pythagorean_triple 8 17 y ∨ is_pythagorean_triple y 8 17) :=
sorry

end NUMINAMATH_CALUDE_pythagorean_triple_with_8_and_17_l3014_301448


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l3014_301402

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base k to base 10 -/
def baseKToBase10 (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The theorem stating that the base k in which 167₈ is written as 315ₖ is equal to 6 -/
theorem base_conversion_theorem : 
  ∃ k : ℕ, k > 1 ∧ base8ToBase10 167 = baseKToBase10 315 k ∧ k = 6 := by sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l3014_301402


namespace NUMINAMATH_CALUDE_root_in_interval_l3014_301469

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x - 5

-- State the theorem
theorem root_in_interval :
  Continuous f →
  f 2 < 0 →
  f 2.5 > 0 →
  ∃ x ∈ Set.Ioo 2 2.5, f x = 0 := by sorry

end NUMINAMATH_CALUDE_root_in_interval_l3014_301469


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_product_l3014_301491

theorem ellipse_hyperbola_product (a b : ℝ) : 
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → (x = 8 ∧ y = 0) ∨ (x = -8 ∧ y = 0)) →
  |a * b| = Real.sqrt 867.75 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_product_l3014_301491


namespace NUMINAMATH_CALUDE_triangle_problem_l3014_301461

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a + b = 5, c = √7, and 4sin²((A + B)/2) - cos(2C) = 7/2,
    then the measure of angle C is π/3 and the area of triangle ABC is 3√3/2 -/
theorem triangle_problem (a b c A B C : Real) : 
  a + b = 5 →
  c = Real.sqrt 7 →
  4 * Real.sin (A + B) ^ 2 / 4 - Real.cos (2 * C) = 7 / 2 →
  C = π / 3 ∧ 
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l3014_301461


namespace NUMINAMATH_CALUDE_max_m_for_factorizable_quadratic_l3014_301453

/-- 
Given a quadratic expression 5x^2 + mx + 45 that can be factored as the product 
of two linear factors with integer coefficients, the maximum possible value of m is 226.
-/
theorem max_m_for_factorizable_quadratic : 
  ∀ m : ℤ, 
  (∃ A B : ℤ, 5*x^2 + m*x + 45 = (5*x + A)*(x + B)) → 
  m ≤ 226 :=
by sorry

end NUMINAMATH_CALUDE_max_m_for_factorizable_quadratic_l3014_301453


namespace NUMINAMATH_CALUDE_fixed_charge_is_28_l3014_301486

-- Define the variables
def fixed_charge : ℝ := sorry
def january_call_charge : ℝ := sorry
def february_call_charge : ℝ := sorry

-- Define the conditions
axiom january_bill : fixed_charge + january_call_charge = 52
axiom february_bill : fixed_charge + february_call_charge = 76
axiom february_double_january : february_call_charge = 2 * january_call_charge

-- Theorem to prove
theorem fixed_charge_is_28 : fixed_charge = 28 := by sorry

end NUMINAMATH_CALUDE_fixed_charge_is_28_l3014_301486


namespace NUMINAMATH_CALUDE_banana_bread_theorem_l3014_301410

def bananas_per_loaf : ℕ := 4
def monday_loaves : ℕ := 3
def tuesday_loaves : ℕ := 2 * monday_loaves

def total_loaves : ℕ := monday_loaves + tuesday_loaves
def total_bananas : ℕ := total_loaves * bananas_per_loaf

theorem banana_bread_theorem : total_bananas = 36 := by
  sorry

end NUMINAMATH_CALUDE_banana_bread_theorem_l3014_301410
