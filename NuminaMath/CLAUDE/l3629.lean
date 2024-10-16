import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3629_362958

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^2 - 20 * X + 62 = (X - 6) * q + 50 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3629_362958


namespace NUMINAMATH_CALUDE_sum_divides_8n_count_l3629_362911

theorem sum_divides_8n_count : 
  (∃ (S : Finset ℕ), S.card = 4 ∧ 
    (∀ n : ℕ, n > 0 → (n ∈ S ↔ (8 * n) % ((n * (n + 1)) / 2) = 0))) := by
  sorry

end NUMINAMATH_CALUDE_sum_divides_8n_count_l3629_362911


namespace NUMINAMATH_CALUDE_alice_acorn_price_l3629_362946

/-- The price Alice paid for each acorn -/
def alice_price_per_acorn (alice_acorns : ℕ) (alice_bob_price_ratio : ℕ) (bob_total_price : ℕ) : ℚ :=
  (alice_bob_price_ratio * bob_total_price : ℚ) / alice_acorns

/-- Proof that Alice paid $15 for each acorn -/
theorem alice_acorn_price :
  alice_price_per_acorn 3600 9 6000 = 15 := by
  sorry

#eval alice_price_per_acorn 3600 9 6000

end NUMINAMATH_CALUDE_alice_acorn_price_l3629_362946


namespace NUMINAMATH_CALUDE_diamond_ratio_sixteen_two_over_two_sixteen_l3629_362970

-- Define the diamond operation
def diamond (n m : ℕ) : ℕ := n^4 * m^2

-- State the theorem
theorem diamond_ratio_sixteen_two_over_two_sixteen : 
  (diamond 16 2) / (diamond 2 16) = 64 := by sorry

end NUMINAMATH_CALUDE_diamond_ratio_sixteen_two_over_two_sixteen_l3629_362970


namespace NUMINAMATH_CALUDE_equation_solution_l3629_362930

theorem equation_solution (x : ℝ) (h : x ≥ -1) :
  Real.sqrt (x + 1) - 1 = x / (Real.sqrt (x + 1) + 1) := by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3629_362930


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l3629_362902

theorem complex_sum_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1) 
  (h2 : Complex.abs z₂ = 1) 
  (h3 : Complex.abs (z₁ - z₂) = Real.sqrt 3) : 
  Complex.abs (z₁ + z₂) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l3629_362902


namespace NUMINAMATH_CALUDE_intersection_implies_range_l3629_362977

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

theorem intersection_implies_range (a : ℝ) : A ∩ B a = A → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_range_l3629_362977


namespace NUMINAMATH_CALUDE_late_time_calculation_l3629_362910

/-- Calculates the total late time for five students given the lateness of one student and the additional lateness of the other four. -/
def totalLateTime (firstStudentLateness : ℕ) (additionalLateness : ℕ) : ℕ :=
  firstStudentLateness + 4 * (firstStudentLateness + additionalLateness)

/-- Theorem stating that for the given scenario, the total late time is 140 minutes. -/
theorem late_time_calculation :
  totalLateTime 20 10 = 140 := by
  sorry

end NUMINAMATH_CALUDE_late_time_calculation_l3629_362910


namespace NUMINAMATH_CALUDE_no_positive_sequence_with_sum_property_l3629_362918

open Real
open Set
open Nat

theorem no_positive_sequence_with_sum_property :
  ¬ (∃ b : ℕ → ℝ, 
    (∀ i : ℕ, i > 0 → b i > 0) ∧ 
    (∀ m : ℕ, m > 0 → (∑' k : ℕ, b (m * k)) = 1 / m)) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_sequence_with_sum_property_l3629_362918


namespace NUMINAMATH_CALUDE_work_completion_theorem_l3629_362973

theorem work_completion_theorem (initial_men : ℕ) (initial_days : ℕ) (new_days : ℕ) (new_men : ℕ) :
  initial_men = 36 →
  initial_days = 18 →
  new_days = 8 →
  initial_men * initial_days = new_men * new_days →
  new_men = 81 := by
sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l3629_362973


namespace NUMINAMATH_CALUDE_odd_function_range_l3629_362937

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def range (f : ℝ → ℝ) : Set ℝ := {y | ∃ x, f x = y}

theorem odd_function_range (f : ℝ → ℝ) (h_odd : is_odd f) (h_pos : ∀ x > 0, f x = 2) :
  range f = {-2, 0, 2} := by
  sorry

end NUMINAMATH_CALUDE_odd_function_range_l3629_362937


namespace NUMINAMATH_CALUDE_jeans_pricing_l3629_362922

theorem jeans_pricing (manufacturing_cost : ℝ) (manufacturing_cost_pos : manufacturing_cost > 0) :
  let retail_price := manufacturing_cost * (1 + 0.4)
  let customer_price := retail_price * (1 + 0.1)
  (customer_price - manufacturing_cost) / manufacturing_cost = 0.54 := by
sorry

end NUMINAMATH_CALUDE_jeans_pricing_l3629_362922


namespace NUMINAMATH_CALUDE_average_squares_first_11_even_numbers_l3629_362947

theorem average_squares_first_11_even_numbers :
  let first_11_even : List Nat := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
  let squares := first_11_even.map (λ x => x * x)
  let sum_squares := squares.sum
  let average := sum_squares / first_11_even.length
  average = 184 := by
sorry

end NUMINAMATH_CALUDE_average_squares_first_11_even_numbers_l3629_362947


namespace NUMINAMATH_CALUDE_complex_number_properties_l3629_362912

def complex_number (m : ℝ) : ℂ := (m^2 - 5*m + 6 : ℝ) + (m^2 - 3*m : ℝ) * Complex.I

theorem complex_number_properties :
  (∀ m : ℝ, (complex_number m).im = 0 ↔ m = 0 ∨ m = 3) ∧
  (∀ m : ℝ, (complex_number m).re = 0 ↔ m = 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3629_362912


namespace NUMINAMATH_CALUDE_initial_candies_count_l3629_362975

def candies_remaining (initial : ℕ) (day : ℕ) : ℤ :=
  match day with
  | 0 => initial
  | n + 1 => (candies_remaining initial n / 2 : ℤ) - 1

theorem initial_candies_count :
  ∃ initial : ℕ, 
    candies_remaining initial 3 = 0 ∧ 
    ∀ d : ℕ, d < 3 → candies_remaining initial d > 0 ∧ 
    initial = 14 :=
by sorry

end NUMINAMATH_CALUDE_initial_candies_count_l3629_362975


namespace NUMINAMATH_CALUDE_train_speed_proof_l3629_362945

/-- The average speed of a train with stoppages, in km/h -/
def speed_with_stoppages : ℝ := 60

/-- The duration of stoppages per hour, in minutes -/
def stoppage_duration : ℝ := 15

/-- The average speed of a train without stoppages, in km/h -/
def speed_without_stoppages : ℝ := 80

theorem train_speed_proof :
  speed_without_stoppages * ((60 - stoppage_duration) / 60) = speed_with_stoppages :=
by sorry

end NUMINAMATH_CALUDE_train_speed_proof_l3629_362945


namespace NUMINAMATH_CALUDE_beatrice_gilbert_ratio_l3629_362921

/-- The number of crayons in each person's box -/
structure CrayonBoxes where
  karen : ℕ
  beatrice : ℕ
  gilbert : ℕ
  judah : ℕ

/-- The conditions of the problem -/
def problem_conditions (boxes : CrayonBoxes) : Prop :=
  boxes.karen = 2 * boxes.beatrice ∧
  boxes.beatrice = boxes.gilbert ∧
  boxes.gilbert = 4 * boxes.judah ∧
  boxes.karen = 128 ∧
  boxes.judah = 8

/-- The theorem stating that Beatrice and Gilbert have the same number of crayons -/
theorem beatrice_gilbert_ratio (boxes : CrayonBoxes) 
  (h : problem_conditions boxes) : boxes.beatrice = boxes.gilbert := by
  sorry

#check beatrice_gilbert_ratio

end NUMINAMATH_CALUDE_beatrice_gilbert_ratio_l3629_362921


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l3629_362985

theorem opposite_of_negative_2023 : -((-2023) : ℤ) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l3629_362985


namespace NUMINAMATH_CALUDE_optimal_washing_effect_l3629_362963

/-- Represents the laundry scenario with given parameters -/
structure LaundryScenario where
  tub_capacity : ℝ
  clothes_weight : ℝ
  initial_detergent_scoops : ℕ
  scoop_weight : ℝ
  optimal_ratio : ℝ

/-- Calculates the optimal amount of detergent and water to add -/
def optimal_addition (scenario : LaundryScenario) : ℝ × ℝ :=
  sorry

/-- Theorem stating that the calculated optimal addition achieves the desired washing effect -/
theorem optimal_washing_effect (scenario : LaundryScenario) 
  (h1 : scenario.tub_capacity = 20)
  (h2 : scenario.clothes_weight = 5)
  (h3 : scenario.initial_detergent_scoops = 2)
  (h4 : scenario.scoop_weight = 0.02)
  (h5 : scenario.optimal_ratio = 0.004) :
  let (added_detergent, added_water) := optimal_addition scenario
  (added_detergent = 0.02) ∧ 
  (added_water = 14.94) ∧
  (scenario.tub_capacity = scenario.clothes_weight + added_water + added_detergent + scenario.initial_detergent_scoops * scenario.scoop_weight) ∧
  (added_detergent + scenario.initial_detergent_scoops * scenario.scoop_weight = scenario.optimal_ratio * (added_water + scenario.initial_detergent_scoops * scenario.scoop_weight)) :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_washing_effect_l3629_362963


namespace NUMINAMATH_CALUDE_intersection_equality_l3629_362934

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 ≥ p.1^2}
def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - a)^2 ≤ 1}

-- State the theorem
theorem intersection_equality (a : ℝ) : M ∩ N a = N a ↔ a ≥ 5/4 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_l3629_362934


namespace NUMINAMATH_CALUDE_no_solution_condition_l3629_362974

theorem no_solution_condition (a : ℝ) :
  (∀ x : ℝ, 9 * |x - 4*a| + |x - a^2| + 8*x - 2*a ≠ 0) ↔ (a < -26 ∨ a > 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l3629_362974


namespace NUMINAMATH_CALUDE_max_value_of_function_l3629_362900

theorem max_value_of_function (x : ℝ) (h : x > 0) : 2 - x - 4 / x ≤ -2 ∧ (2 - x - 4 / x = -2 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3629_362900


namespace NUMINAMATH_CALUDE_scrabble_game_result_l3629_362941

/-- The Scrabble game between Brenda and David -/
def ScrabbleGame (brenda_turn1 brenda_turn2 brenda_turn3 david_turn1 david_turn2 david_turn3 brenda_lead_before_turn3 : ℕ) : Prop :=
  let brenda_total := brenda_turn1 + brenda_turn2 + brenda_turn3
  let david_total := david_turn1 + david_turn2 + david_turn3
  brenda_total + 19 = david_total

theorem scrabble_game_result :
  ScrabbleGame 18 25 15 10 35 32 22 := by
  sorry

end NUMINAMATH_CALUDE_scrabble_game_result_l3629_362941


namespace NUMINAMATH_CALUDE_cubic_third_root_l3629_362925

theorem cubic_third_root (a b : ℚ) :
  (∀ x : ℚ, a * x^3 + (a + 4*b) * x^2 + (b - 5*a) * x + (10 - a) = 0 ↔ x = -1 ∨ x = 4 ∨ x = 8/3) →
  (a * (-1)^3 + (a + 4*b) * (-1)^2 + (b - 5*a) * (-1) + (10 - a) = 0) →
  (a * 4^3 + (a + 4*b) * 4^2 + (b - 5*a) * 4 + (10 - a) = 0) →
  ∃ x : ℚ, x = 8/3 ∧ a * x^3 + (a + 4*b) * x^2 + (b - 5*a) * x + (10 - a) = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_third_root_l3629_362925


namespace NUMINAMATH_CALUDE_traditionalist_fraction_l3629_362952

theorem traditionalist_fraction (num_provinces : ℕ) (num_traditionalists_per_province : ℚ) 
  (total_progressives : ℚ) :
  num_provinces = 4 →
  num_traditionalists_per_province = total_progressives / 12 →
  (num_provinces : ℚ) * num_traditionalists_per_province / 
    (total_progressives + (num_provinces : ℚ) * num_traditionalists_per_province) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_traditionalist_fraction_l3629_362952


namespace NUMINAMATH_CALUDE_inequality_proof_l3629_362994

theorem inequality_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  3 * ((x^2 / y^2) + (y^2 / x^2)) - 8 * ((x / y) + (y / x)) + 10 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3629_362994


namespace NUMINAMATH_CALUDE_sin_alpha_cos_beta_value_l3629_362987

theorem sin_alpha_cos_beta_value (α β : Real) 
  (h1 : Real.sin (α + β) = 1/2) 
  (h2 : Real.sin (α - β) = 1/4) : 
  Real.sin α * Real.cos β = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_cos_beta_value_l3629_362987


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3629_362997

theorem arithmetic_calculation : 15 * 35 - 15 * 5 + 25 * 15 = 825 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3629_362997


namespace NUMINAMATH_CALUDE_girls_from_maple_grove_l3629_362971

-- Define the total number of students
def total_students : ℕ := 150

-- Define the number of boys
def num_boys : ℕ := 82

-- Define the number of girls
def num_girls : ℕ := 68

-- Define the number of students from Pine Ridge School
def pine_ridge_students : ℕ := 70

-- Define the number of students from Maple Grove School
def maple_grove_students : ℕ := 80

-- Define the number of boys from Pine Ridge School
def pine_ridge_boys : ℕ := 36

-- Theorem to prove
theorem girls_from_maple_grove :
  total_students = num_boys + num_girls ∧
  total_students = pine_ridge_students + maple_grove_students ∧
  num_boys = pine_ridge_boys + (num_boys - pine_ridge_boys) →
  maple_grove_students - (num_boys - pine_ridge_boys) = 34 :=
by sorry

end NUMINAMATH_CALUDE_girls_from_maple_grove_l3629_362971


namespace NUMINAMATH_CALUDE_triangle_sum_vertices_sides_l3629_362976

/-- Definition of a triangle -/
structure Triangle where
  vertices : ℕ
  sides : ℕ

/-- The sum of vertices and sides of a triangle is 6 -/
theorem triangle_sum_vertices_sides : ∀ t : Triangle, t.vertices + t.sides = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_vertices_sides_l3629_362976


namespace NUMINAMATH_CALUDE_students_per_minibus_l3629_362956

theorem students_per_minibus (total_vehicles : Nat) (num_vans : Nat) (num_minibusses : Nat)
  (students_per_van : Nat) (total_students : Nat) :
  total_vehicles = num_vans + num_minibusses →
  num_vans = 6 →
  num_minibusses = 4 →
  students_per_van = 10 →
  total_students = 156 →
  (total_students - num_vans * students_per_van) / num_minibusses = 24 := by
  sorry

#check students_per_minibus

end NUMINAMATH_CALUDE_students_per_minibus_l3629_362956


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3629_362914

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (m n : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : c > 0) (h4 : m * n = 2 / 9) :
  let f (x y : ℝ) := x^2 / a^2 - y^2 / b^2
  let asymptote (x : ℝ) := b / a * x
  let A : ℝ × ℝ := (c, asymptote c)
  let B : ℝ × ℝ := (c, -asymptote c)
  let P : ℝ × ℝ := ((m + n) * c, (m - n) * asymptote c)
  (f (P.1) (P.2) = 1) →
  (c / a = 3 * Real.sqrt 2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3629_362914


namespace NUMINAMATH_CALUDE_sequence_periodicity_l3629_362972

/-- A cubic polynomial with rational coefficients -/
def CubicPolynomial (α : ℚ → ℚ) : Prop :=
  ∃ a b c d : ℚ, ∀ x, α x = a * x^3 + b * x^2 + c * x + d

/-- A sequence of rational numbers satisfying the given condition -/
def SequenceSatisfyingCondition (p : ℚ → ℚ) (q : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, q n = p (q (n + 1))

theorem sequence_periodicity
  (p : ℚ → ℚ) (q : ℕ → ℚ)
  (h_cubic : CubicPolynomial p)
  (h_seq : SequenceSatisfyingCondition p q) :
  ∃ k : ℕ, k ≥ 1 ∧ ∀ n : ℕ, n ≥ 1 → q (n + k) = q n :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l3629_362972


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l3629_362938

/-- Given a point (r, θ) in polar coordinates and a line θ = α, 
    the symmetric point with respect to this line has coordinates (r, 2α - θ) -/
def symmetric_point (r : ℝ) (θ : ℝ) (α : ℝ) : ℝ × ℝ := (r, 2*α - θ)

/-- The point symmetric to (3, π/2) with respect to the line θ = π/6 
    has polar coordinates (3, -π/6) -/
theorem symmetric_point_theorem : 
  symmetric_point 3 (π/2) (π/6) = (3, -π/6) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l3629_362938


namespace NUMINAMATH_CALUDE_no_solution_implies_a_equals_one_l3629_362995

/-- If the system of equations ax + y = 1 and x + y = 2 has no solution, then a = 1 -/
theorem no_solution_implies_a_equals_one (a : ℝ) : 
  (∀ x y : ℝ, ¬(ax + y = 1 ∧ x + y = 2)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_equals_one_l3629_362995


namespace NUMINAMATH_CALUDE_geometry_theorem_l3629_362961

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem statement
theorem geometry_theorem 
  (a b : Line) (α β : Plane) 
  (h : perpendicular b α) :
  (parallel_line_plane a α → perpendicular_lines a b) ∧
  (perpendicular b β → parallel_planes α β) := by
  sorry

end NUMINAMATH_CALUDE_geometry_theorem_l3629_362961


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3629_362901

theorem complex_equation_solution (z : ℂ) : 
  Complex.abs z + z = 2 + 4 * Complex.I → z = -3 + 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3629_362901


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3629_362931

theorem purely_imaginary_complex_number (a : ℝ) :
  let z : ℂ := a^2 - 4 + (a - 2) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → a = -2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3629_362931


namespace NUMINAMATH_CALUDE_romanov_family_savings_l3629_362992

/-- Represents the electricity cost calculation problem for the Romanov family -/
def electricity_cost_problem (multi_tariff_meter_cost : ℝ) (installation_cost : ℝ)
  (monthly_consumption : ℝ) (night_consumption : ℝ) (day_rate : ℝ) (night_rate : ℝ)
  (standard_rate : ℝ) : Prop :=
  let day_consumption := monthly_consumption - night_consumption
  let multi_tariff_yearly_cost := (night_consumption * night_rate + day_consumption * day_rate) * 12
  let standard_yearly_cost := monthly_consumption * standard_rate * 12
  let multi_tariff_total_cost := multi_tariff_meter_cost + installation_cost + multi_tariff_yearly_cost * 3
  let standard_total_cost := standard_yearly_cost * 3
  standard_total_cost - multi_tariff_total_cost = 3824

/-- The theorem stating the savings of the Romanov family -/
theorem romanov_family_savings :
  electricity_cost_problem 3500 1100 300 230 5.2 3.4 4.6 :=
by
  sorry

end NUMINAMATH_CALUDE_romanov_family_savings_l3629_362992


namespace NUMINAMATH_CALUDE_prime_factor_difference_l3629_362908

theorem prime_factor_difference (n : Nat) (h : n = 173459) :
  ∃ (p₁ p₂ p₃ p₄ : Nat),
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧
    n = p₁ * p₂ * p₃ * p₄ ∧
    p₁ ≤ p₂ ∧ p₂ ≤ p₃ ∧ p₃ ≤ p₄ ∧
    p₄ - p₂ = 144 :=
by sorry

end NUMINAMATH_CALUDE_prime_factor_difference_l3629_362908


namespace NUMINAMATH_CALUDE_sock_drawing_probability_l3629_362988

def total_socks : ℕ := 12
def socks_per_color_a : ℕ := 3
def colors_with_three_socks : ℕ := 3
def colors_with_one_sock : ℕ := 2
def socks_drawn : ℕ := 5

def favorable_outcomes : ℕ := 
  (colors_with_three_socks.choose 2) * 
  (colors_with_one_sock.choose 1) * 
  (socks_per_color_a.choose 2) * 
  (socks_per_color_a.choose 2) * 
  1

def total_outcomes : ℕ := total_socks.choose socks_drawn

theorem sock_drawing_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 44 := by sorry

end NUMINAMATH_CALUDE_sock_drawing_probability_l3629_362988


namespace NUMINAMATH_CALUDE_circle_condition_l3629_362929

theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - x + y + m = 0 ∧ 
   ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - x + y + m = 0) 
  → m < (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_circle_condition_l3629_362929


namespace NUMINAMATH_CALUDE_equation_solution_l3629_362924

theorem equation_solution (x : ℝ) : (x - 3)^2 = x^2 - 9 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3629_362924


namespace NUMINAMATH_CALUDE_propositions_correctness_l3629_362928

theorem propositions_correctness : 
  (∃ x : ℝ, x^2 ≥ x) ∧ 
  (4 ≥ 3) ∧ 
  ¬(∀ x : ℝ, x^2 ≥ x) ∧
  ¬(∀ x : ℝ, x^2 ≠ 1 ↔ (x ≠ 1 ∨ x ≠ -1)) := by
  sorry

end NUMINAMATH_CALUDE_propositions_correctness_l3629_362928


namespace NUMINAMATH_CALUDE_fence_length_proof_l3629_362984

theorem fence_length_proof (darren_length : ℝ) (doug_length : ℝ) : 
  darren_length = 1.2 * doug_length →
  darren_length = 360 →
  darren_length + doug_length = 660 :=
by
  sorry

end NUMINAMATH_CALUDE_fence_length_proof_l3629_362984


namespace NUMINAMATH_CALUDE_fourth_number_nth_row_l3629_362951

/-- The kth number in the triangular array -/
def triangular_array (k : ℕ) : ℕ := 2^(k - 1)

/-- The position of the 4th number from left to right in the nth row -/
def fourth_number_position (n : ℕ) : ℕ := n * (n - 1) / 2 + 4

theorem fourth_number_nth_row (n : ℕ) (h : n ≥ 4) :
  triangular_array (fourth_number_position n) = 2^((n^2 - n + 6) / 2) :=
sorry

end NUMINAMATH_CALUDE_fourth_number_nth_row_l3629_362951


namespace NUMINAMATH_CALUDE_solve_system_l3629_362916

theorem solve_system (x y : ℚ) : 
  (12 * x + 198 = 12 * y + 176) → 
  (x + y = 29) → 
  (x = 163 / 12 ∧ y = 185 / 12) := by
sorry

end NUMINAMATH_CALUDE_solve_system_l3629_362916


namespace NUMINAMATH_CALUDE_two_distinct_roots_l3629_362955

/-- The equation has two distinct roots if and only if a is in the specified ranges -/
theorem two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 + x * |x| = 2 * (3 + a * x - 2 * a) ∧ 
    y^2 + y * |y| = 2 * (3 + a * y - 2 * a)) ↔ 
  ((3/4 ≤ a ∧ a < 1) ∨ a > 3) ∨ (0 < a ∧ a < 3/4) :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l3629_362955


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l3629_362915

theorem max_value_on_ellipse (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 13 ∧ x + y ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l3629_362915


namespace NUMINAMATH_CALUDE_workshop_handshakes_l3629_362913

/-- Represents the workshop scenario -/
structure Workshop where
  total_people : Nat
  trainers : Nat
  participants : Nat
  knowledgeable_participants : Nat
  trainers_known_by_knowledgeable : Nat

/-- Calculate the number of handshakes in the workshop -/
def count_handshakes (w : Workshop) : Nat :=
  let unknown_participants := w.participants - w.knowledgeable_participants
  let handshakes_unknown := unknown_participants * (w.total_people - 1)
  let handshakes_knowledgeable := w.knowledgeable_participants * (w.total_people - w.trainers_known_by_knowledgeable - 1)
  handshakes_unknown + handshakes_knowledgeable

/-- The theorem to be proved -/
theorem workshop_handshakes :
  let w : Workshop := {
    total_people := 40,
    trainers := 25,
    participants := 15,
    knowledgeable_participants := 5,
    trainers_known_by_knowledgeable := 10
  }
  count_handshakes w = 540 := by sorry

end NUMINAMATH_CALUDE_workshop_handshakes_l3629_362913


namespace NUMINAMATH_CALUDE_pin_pierces_all_sheets_l3629_362906

/-- Represents a rectangular sheet of paper -/
structure Sheet :=
  (width : ℝ)
  (height : ℝ)
  (center : ℝ × ℝ)

/-- Represents a collection of sheets on a table -/
structure TableSetup :=
  (sheets : List Sheet)
  (top_sheet : Sheet)

/-- Predicate to check if a point is on a sheet -/
def point_on_sheet (p : ℝ × ℝ) (s : Sheet) : Prop :=
  let (x, y) := p
  let (cx, cy) := s.center
  |x - cx| ≤ s.width / 2 ∧ |y - cy| ≤ s.height / 2

/-- The main theorem -/
theorem pin_pierces_all_sheets (setup : TableSetup) 
  (h_identical : ∀ s ∈ setup.sheets, s = setup.top_sheet)
  (h_cover : ∀ s ∈ setup.sheets, s ≠ setup.top_sheet → 
    (Set.inter (Set.range (point_on_sheet · setup.top_sheet)) 
               (Set.range (point_on_sheet · s))).ncard > 
    (Set.range (point_on_sheet · s)).ncard / 2) :
  ∃ p : ℝ × ℝ, ∀ s ∈ setup.sheets, point_on_sheet p s :=
sorry

end NUMINAMATH_CALUDE_pin_pierces_all_sheets_l3629_362906


namespace NUMINAMATH_CALUDE_max_k_value_l3629_362943

theorem max_k_value (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : k * a * b * c / (a + b + c) ≤ (a + b)^2 + (a + b + 4*c)^2) : 
  k ≤ 100 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l3629_362943


namespace NUMINAMATH_CALUDE_function_value_at_three_l3629_362960

/-- Given a function f(x) = ax^4 + b cos(x) - x where f(-3) = 7, prove that f(3) = 1 -/
theorem function_value_at_three (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^4 + b * Real.cos x - x)
  (h2 : f (-3) = 7) : 
  f 3 = 1 := by sorry

end NUMINAMATH_CALUDE_function_value_at_three_l3629_362960


namespace NUMINAMATH_CALUDE_arrow_sequence_for_multiples_of_four_l3629_362926

def arrow_direction (n : ℕ) : Bool × Bool :=
  if n % 4 = 0 then (false, true) else (true, false)

theorem arrow_sequence_for_multiples_of_four (n : ℕ) (h : n % 4 = 0) :
  arrow_direction n = (false, true) := by sorry

end NUMINAMATH_CALUDE_arrow_sequence_for_multiples_of_four_l3629_362926


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3629_362935

/-- Calculates the molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (fe_count o_count ca_count c_count : ℕ) 
                    (fe_weight o_weight ca_weight c_weight : ℝ) : ℝ :=
  fe_count * fe_weight + o_count * o_weight + ca_count * ca_weight + c_count * c_weight

/-- Theorem stating that the molecular weight of the given compound is 223.787 amu -/
theorem compound_molecular_weight :
  let fe_count : ℕ := 2
  let o_count : ℕ := 3
  let ca_count : ℕ := 1
  let c_count : ℕ := 2
  let fe_weight : ℝ := 55.845
  let o_weight : ℝ := 15.999
  let ca_weight : ℝ := 40.078
  let c_weight : ℝ := 12.011
  molecularWeight fe_count o_count ca_count c_count fe_weight o_weight ca_weight c_weight = 223.787 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3629_362935


namespace NUMINAMATH_CALUDE_geometric_sum_n_eq_1_l3629_362979

theorem geometric_sum_n_eq_1 (a : ℝ) (h : a ≠ 1) :
  1 + a = (1 - a^3) / (1 - a) := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_n_eq_1_l3629_362979


namespace NUMINAMATH_CALUDE_down_payment_calculation_l3629_362923

theorem down_payment_calculation (purchase_price : ℝ) (monthly_payment : ℝ) 
  (num_payments : ℕ) (interest_rate : ℝ) (down_payment : ℝ) : 
  purchase_price = 130 ∧ 
  monthly_payment = 10 ∧ 
  num_payments = 12 ∧ 
  interest_rate = 0.23076923076923077 →
  down_payment = purchase_price + interest_rate * purchase_price - num_payments * monthly_payment :=
by
  sorry

end NUMINAMATH_CALUDE_down_payment_calculation_l3629_362923


namespace NUMINAMATH_CALUDE_complex_magnitude_l3629_362903

theorem complex_magnitude (z : ℂ) : z = 1 + 2*I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3629_362903


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3629_362991

theorem complex_sum_theorem (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : 1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / ω^2) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l3629_362991


namespace NUMINAMATH_CALUDE_girls_in_class_l3629_362983

/-- The number of boys in the class -/
def num_boys : ℕ := 13

/-- The number of ways to select 1 girl and 2 boys -/
def num_selections : ℕ := 780

/-- The number of girls in the class -/
def num_girls : ℕ := 10

theorem girls_in_class : 
  num_girls * (num_boys.choose 2) = num_selections :=
sorry

end NUMINAMATH_CALUDE_girls_in_class_l3629_362983


namespace NUMINAMATH_CALUDE_max_rectangle_area_l3629_362990

/-- Given a string of length 32 cm, the maximum area of a rectangle that can be formed is 64 cm². -/
theorem max_rectangle_area (string_length : ℝ) (h : string_length = 32) : 
  (∀ w h : ℝ, w > 0 → h > 0 → 2*w + 2*h ≤ string_length → w * h ≤ 64) ∧ 
  (∃ w h : ℝ, w > 0 ∧ h > 0 ∧ 2*w + 2*h = string_length ∧ w * h = 64) :=
by sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l3629_362990


namespace NUMINAMATH_CALUDE_bicycle_price_increase_l3629_362982

theorem bicycle_price_increase (original_price new_price : ℝ) 
  (h1 : original_price = 220)
  (h2 : new_price = 253) :
  (new_price - original_price) / original_price * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_increase_l3629_362982


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l3629_362942

/-- Given an annual interest rate and time period, calculates the compound interest
    if the simple interest is known. -/
theorem compound_interest_calculation
  (P : ℝ) -- Principal amount
  (R : ℝ) -- Annual interest rate (as a percentage)
  (T : ℝ) -- Time period in years
  (h1 : R = 20)
  (h2 : T = 2)
  (h3 : P * R * T / 100 = 400) -- Simple interest formula
  : P * (1 + R/100)^T - P = 440 := by
  sorry

#check compound_interest_calculation

end NUMINAMATH_CALUDE_compound_interest_calculation_l3629_362942


namespace NUMINAMATH_CALUDE_sum_is_square_l3629_362981

theorem sum_is_square (x y z : ℕ+) 
  (h1 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / z)
  (h2 : Nat.gcd (Nat.gcd x.val y.val) z.val = 1) :
  ∃ n : ℕ, x.val + y.val = n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_square_l3629_362981


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_set_l3629_362962

theorem quadratic_inequality_empty_set (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x + 1 ≥ 0) ↔ 0 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_set_l3629_362962


namespace NUMINAMATH_CALUDE_point_config_theorem_l3629_362999

/-- Given three points A, B, C on a straight line in the Cartesian coordinate system -/
structure PointConfig where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  m : ℝ
  n : ℝ
  on_line : A.1 < B.1 ∧ B.1 < C.1 ∨ C.1 < B.1 ∧ B.1 < A.1

/-- The given conditions -/
def satisfies_conditions (config : PointConfig) : Prop :=
  config.A = (-3, config.m + 1) ∧
  config.B = (config.n, 3) ∧
  config.C = (7, 4) ∧
  config.A.1 * config.B.1 + config.A.2 * config.B.2 = 0 ∧  -- OA ⟂ OB
  ∃ (G : ℝ × ℝ), (G.1 = 2/3 * config.B.1 ∧ G.2 = 2/3 * config.B.2)  -- OG = (2/3) * OB

/-- The theorem to prove -/
theorem point_config_theorem (config : PointConfig) 
  (h : satisfies_conditions config) :
  (config.m = 1 ∧ config.n = 2) ∨ (config.m = 8 ∧ config.n = 9) ∧
  (config.A.1 * config.C.1 + config.A.2 * config.C.2) / 
  (Real.sqrt (config.A.1^2 + config.A.2^2) * Real.sqrt (config.C.1^2 + config.C.2^2)) = -Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_point_config_theorem_l3629_362999


namespace NUMINAMATH_CALUDE_partnership_capital_fraction_l3629_362980

theorem partnership_capital_fraction :
  ∀ (T : ℚ) (x : ℚ),
    x > 0 →
    T > 0 →
    x * T + (1/4) * T + (1/5) * T + ((11/20 - x) * T) = T →
    805 / 2415 = x →
    x = 161 / 483 := by
  sorry

end NUMINAMATH_CALUDE_partnership_capital_fraction_l3629_362980


namespace NUMINAMATH_CALUDE_project_completion_time_l3629_362904

theorem project_completion_time 
  (initial_people : ℕ) 
  (initial_days : ℕ) 
  (additional_people : ℕ) 
  (h1 : initial_people = 12)
  (h2 : initial_days = 15)
  (h3 : additional_people = 8) : 
  (initial_days + (initial_people * initial_days * 2) / (initial_people + additional_people)) = 33 :=
by sorry

end NUMINAMATH_CALUDE_project_completion_time_l3629_362904


namespace NUMINAMATH_CALUDE_square_side_lengths_average_l3629_362949

theorem square_side_lengths_average (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 16) (h₂ : a₂ = 49) (h₃ : a₃ = 169) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_lengths_average_l3629_362949


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l3629_362964

theorem number_exceeding_fraction : 
  ∀ x : ℚ, x = (3/8) * x + 20 → x = 32 := by
sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l3629_362964


namespace NUMINAMATH_CALUDE_second_number_value_l3629_362953

theorem second_number_value (a b c : ℝ) : 
  a + b + c = 110 ∧ 
  a = 2 * b ∧ 
  c = (1/3) * a 
  → b = 30 := by sorry

end NUMINAMATH_CALUDE_second_number_value_l3629_362953


namespace NUMINAMATH_CALUDE_square_of_binomial_l3629_362957

theorem square_of_binomial (a : ℚ) : 
  (∃ r s : ℚ, ∀ x : ℚ, a * x^2 + 18 * x + 16 = (r * x + s)^2) → 
  a = 81 / 16 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_l3629_362957


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3629_362948

theorem solution_set_inequality (a b : ℝ) (h : |a - b| > 2) :
  ∀ x : ℝ, |x - a| + |x - b| > 2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3629_362948


namespace NUMINAMATH_CALUDE_symmetric_point_of_A_l3629_362936

/-- A point in 3D Cartesian space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin in 3D Cartesian space -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- Point A with coordinates (1, 1, 2) -/
def pointA : Point3D := ⟨1, 1, 2⟩

/-- A point is symmetric to another point with respect to the origin if the origin is the midpoint of the line segment connecting the two points -/
def isSymmetricWrtOrigin (p q : Point3D) : Prop :=
  origin.x = (p.x + q.x) / 2 ∧
  origin.y = (p.y + q.y) / 2 ∧
  origin.z = (p.z + q.z) / 2

/-- The theorem stating that the point symmetric to A(1, 1, 2) with respect to the origin has coordinates (-1, -1, -2) -/
theorem symmetric_point_of_A :
  ∃ (B : Point3D), isSymmetricWrtOrigin pointA B ∧ B = ⟨-1, -1, -2⟩ :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_of_A_l3629_362936


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3629_362939

/-- Given an arithmetic sequence {a_n} where a_3 and a_15 are the roots of x^2 - 6x + 8 = 0,
    the sum a_7 + a_8 + a_9 + a_10 + a_11 is equal to 15. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (a 3)^2 - 6 * (a 3) + 8 = 0 →  -- a_3 is a root
  (a 15)^2 - 6 * (a 15) + 8 = 0 →  -- a_15 is a root
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3629_362939


namespace NUMINAMATH_CALUDE_area_eq_twice_radius_squared_l3629_362954

/-- A rectangle with a circle tangent to three sides and passing through the diagonal midpoint -/
structure TangentCircleRectangle where
  -- The length of the rectangle
  length : ℝ
  -- The width of the rectangle
  width : ℝ
  -- The radius of the tangent circle
  radius : ℝ
  -- The circle is tangent to three sides
  tangent_to_sides : True
  -- The circle passes through the midpoint of the diagonal
  passes_through_midpoint : True
  -- The width is equal to the radius (from the problem configuration)
  width_eq_radius : width = radius
  -- The length is twice the radius (derived from the midpoint condition)
  length_eq_double_radius : length = 2 * radius

/-- The area of a rectangle with a tangent circle -/
def area (rect : TangentCircleRectangle) : ℝ :=
  rect.length * rect.width

/-- Theorem: The area of the rectangle is twice the square of the circle's radius -/
theorem area_eq_twice_radius_squared (rect : TangentCircleRectangle) :
  area rect = 2 * rect.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_area_eq_twice_radius_squared_l3629_362954


namespace NUMINAMATH_CALUDE_water_bottles_needed_l3629_362933

/-- Calculates the total number of water bottles needed for a family road trip -/
theorem water_bottles_needed
  (family_size : ℕ)
  (travel_time : ℕ)
  (water_consumption : ℚ)
  (h1 : family_size = 4)
  (h2 : travel_time = 16)
  (h3 : water_consumption = 1 / 2) :
  ↑family_size * ↑travel_time * water_consumption = 32 := by
  sorry

#check water_bottles_needed

end NUMINAMATH_CALUDE_water_bottles_needed_l3629_362933


namespace NUMINAMATH_CALUDE_no_division_into_non_convex_quadrilaterals_l3629_362959

/-- A polygon is a set of points in the plane -/
def Polygon : Type := Set (ℝ × ℝ)

/-- A convex polygon is a polygon where any line segment between two points in the polygon lies entirely within the polygon -/
def ConvexPolygon (P : Polygon) : Prop := sorry

/-- A quadrilateral is a polygon with exactly four vertices -/
def Quadrilateral (Q : Polygon) : Prop := sorry

/-- A non-convex quadrilateral is a quadrilateral that is not convex -/
def NonConvexQuadrilateral (Q : Polygon) : Prop := Quadrilateral Q ∧ ¬ConvexPolygon Q

/-- A division of a polygon into quadrilaterals is a finite set of quadrilaterals that cover the polygon without overlap -/
def DivisionIntoQuadrilaterals (P : Polygon) (Qs : Finset Polygon) : Prop := sorry

/-- Theorem: It's impossible to divide a convex polygon into a finite number of non-convex quadrilaterals -/
theorem no_division_into_non_convex_quadrilaterals (P : Polygon) (Qs : Finset Polygon) :
  ConvexPolygon P → DivisionIntoQuadrilaterals P Qs → ¬(∀ Q ∈ Qs, NonConvexQuadrilateral Q) := by
  sorry

end NUMINAMATH_CALUDE_no_division_into_non_convex_quadrilaterals_l3629_362959


namespace NUMINAMATH_CALUDE_coin_combination_difference_l3629_362967

def coin_values : List ℕ := [10, 20, 50]
def target_amount : ℕ := 45

def valid_combination (coins : List ℕ) : Prop :=
  coins.all (λ c => c ∈ coin_values) ∧ coins.sum = target_amount

def num_coins (coins : List ℕ) : ℕ := coins.length

theorem coin_combination_difference :
  ∃ (min_coins max_coins : List ℕ),
    valid_combination min_coins ∧
    valid_combination max_coins ∧
    (∀ coins, valid_combination coins → num_coins min_coins ≤ num_coins coins) ∧
    (∀ coins, valid_combination coins → num_coins coins ≤ num_coins max_coins) ∧
    num_coins max_coins - num_coins min_coins = 0 :=
sorry

end NUMINAMATH_CALUDE_coin_combination_difference_l3629_362967


namespace NUMINAMATH_CALUDE_lg_expression_equals_zero_l3629_362907

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_expression_equals_zero :
  lg 5 * lg 2 + lg (2^2) - lg 2 = 0 :=
by
  -- Properties of logarithms
  have h1 : ∀ m n : ℝ, lg (m^n) = n * lg m := sorry
  have h2 : ∀ a b : ℝ, lg (a * b) = lg a + lg b := sorry
  have h3 : lg 1 = 0 := sorry
  have h4 : lg 2 > 0 := sorry
  have h5 : lg 5 > 0 := sorry
  
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_lg_expression_equals_zero_l3629_362907


namespace NUMINAMATH_CALUDE_basketball_game_second_half_score_l3629_362932

/-- Represents the points scored by a team in each quarter -/
structure QuarterlyPoints where
  q1 : ℝ
  q2 : ℝ
  q3 : ℝ
  q4 : ℝ

/-- The game between Raiders and Wildcats -/
structure BasketballGame where
  raiders : QuarterlyPoints
  wildcats : QuarterlyPoints

def BasketballGame.total_score (game : BasketballGame) : ℝ :=
  game.raiders.q1 + game.raiders.q2 + game.raiders.q3 + game.raiders.q4 +
  game.wildcats.q1 + game.wildcats.q2 + game.wildcats.q3 + game.wildcats.q4

def BasketballGame.second_half_score (game : BasketballGame) : ℝ :=
  game.raiders.q3 + game.raiders.q4 + game.wildcats.q3 + game.wildcats.q4

theorem basketball_game_second_half_score
  (a b d r : ℝ)
  (hr : r ≥ 1)
  (game : BasketballGame)
  (h1 : game.raiders = ⟨a, a*r, a*r^2, a*r^3⟩)
  (h2 : game.wildcats = ⟨b, b+d, b+2*d, b+3*d⟩)
  (h3 : game.raiders.q1 = game.wildcats.q1)
  (h4 : game.total_score = 2 * game.raiders.q1 + game.raiders.q2 + game.raiders.q3 + game.raiders.q4 +
                           game.wildcats.q2 + game.wildcats.q3 + game.wildcats.q4)
  (h5 : game.total_score = 2 * (4*b + 6*d + 3))
  (h6 : ∀ q, q ∈ [game.raiders.q1, game.raiders.q2, game.raiders.q3, game.raiders.q4,
                  game.wildcats.q1, game.wildcats.q2, game.wildcats.q3, game.wildcats.q4] → q ≤ 100) :
  game.second_half_score = 112 :=
sorry

end NUMINAMATH_CALUDE_basketball_game_second_half_score_l3629_362932


namespace NUMINAMATH_CALUDE_triple_integral_equality_l3629_362996

open MeasureTheory Interval Set

theorem triple_integral_equality {f : ℝ → ℝ} (hf : ContinuousOn f (Ioo 0 1)) :
  ∫ x in (Icc 0 1), ∫ y in (Icc x 1), ∫ z in (Icc x y), f x * f y * f z = 
  (1 / 6) * (∫ x in (Icc 0 1), f x) ^ 3 := by sorry

end NUMINAMATH_CALUDE_triple_integral_equality_l3629_362996


namespace NUMINAMATH_CALUDE_cubic_equation_result_l3629_362993

theorem cubic_equation_result (x : ℝ) (h : x^2 + 3*x - 1 = 0) : 
  x^3 + 5*x^2 + 5*x + 18 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_result_l3629_362993


namespace NUMINAMATH_CALUDE_combined_train_length_l3629_362927

/-- Calculates the combined length of two trains given their speeds and passing times. -/
theorem combined_train_length
  (speed_A speed_B speed_bike : ℝ)
  (time_A time_B : ℝ)
  (h1 : speed_A = 120)
  (h2 : speed_B = 100)
  (h3 : speed_bike = 64)
  (h4 : time_A = 75)
  (h5 : time_B = 90)
  (h6 : speed_A > speed_bike)
  (h7 : speed_B > speed_bike)
  (h8 : (speed_A - speed_bike) * time_A / 3600 + (speed_B - speed_bike) * time_B / 3600 = 2.067) :
  (speed_A - speed_bike) * time_A * 1000 / 3600 + (speed_B - speed_bike) * time_B * 1000 / 3600 = 2067 := by
  sorry

#check combined_train_length

end NUMINAMATH_CALUDE_combined_train_length_l3629_362927


namespace NUMINAMATH_CALUDE_A_intersection_Z_l3629_362966

def A : Set ℝ := {x : ℝ | |x - 1| < 2}

theorem A_intersection_Z : A ∩ (Set.range (Int.cast : ℤ → ℝ)) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_A_intersection_Z_l3629_362966


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l3629_362969

theorem ice_cream_flavors (total_flavors : ℕ) (tried_two_years_ago : ℕ) (tried_last_year : ℕ) : 
  total_flavors = 100 →
  tried_two_years_ago = total_flavors / 4 →
  tried_last_year = 2 * tried_two_years_ago →
  total_flavors - (tried_two_years_ago + tried_last_year) = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l3629_362969


namespace NUMINAMATH_CALUDE_car_speed_problem_l3629_362986

/-- Proves that car R's speed is 75 mph given the problem conditions -/
theorem car_speed_problem (distance : ℝ) (time_difference : ℝ) (speed_difference : ℝ) :
  distance = 1200 →
  time_difference = 4 →
  speed_difference = 20 →
  ∃ (speed_R : ℝ),
    distance / speed_R - time_difference = distance / (speed_R + speed_difference) ∧
    speed_R = 75 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_problem_l3629_362986


namespace NUMINAMATH_CALUDE_school_supplies_ratio_l3629_362944

/-- Proves the ratio of school supplies spending to remaining money after textbooks is 1:4 --/
theorem school_supplies_ratio (total : ℕ) (remaining : ℕ) : 
  total = 960 →
  remaining = 360 →
  (total - total / 2 - remaining) / (total / 2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_school_supplies_ratio_l3629_362944


namespace NUMINAMATH_CALUDE_unique_steakmaker_pair_l3629_362909

/-- A pair of positive integers (m,n) is 'steakmaker' if 1 + 2^m = n^2 -/
def is_steakmaker (m n : ℕ+) : Prop := 1 + 2^(m.val) = n.val^2

theorem unique_steakmaker_pair :
  ∃! (m n : ℕ+), is_steakmaker m n ∧ m.val * n.val = 9 :=
sorry

#check unique_steakmaker_pair

end NUMINAMATH_CALUDE_unique_steakmaker_pair_l3629_362909


namespace NUMINAMATH_CALUDE_circle_O_diameter_l3629_362998

/-- The circle O with equation x^2 + y^2 - 2x + my - 4 = 0 -/
def circle_O (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + m*y - 4 = 0

/-- The line with equation 2x + y = 0 -/
def symmetry_line (x y : ℝ) : Prop :=
  2*x + y = 0

/-- Two points are symmetric about a line if the line is the perpendicular bisector of the segment connecting the points -/
def symmetric_points (M N : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (midpoint : ℝ × ℝ), 
    line midpoint.1 midpoint.2 ∧ 
    (midpoint.1 = (M.1 + N.1) / 2) ∧ 
    (midpoint.2 = (M.2 + N.2) / 2) ∧
    ((N.1 - M.1) * 2 + (N.2 - M.2) = 0)

theorem circle_O_diameter : 
  ∃ (m : ℝ) (M N : ℝ × ℝ),
    circle_O m M.1 M.2 ∧ 
    circle_O m N.1 N.2 ∧ 
    symmetric_points M N symmetry_line →
    ∃ (center : ℝ × ℝ) (radius : ℝ),
      center = (1, -m/2) ∧ 
      radius = 3 ∧ 
      2 * radius = 6 :=
sorry

end NUMINAMATH_CALUDE_circle_O_diameter_l3629_362998


namespace NUMINAMATH_CALUDE_tea_cost_price_l3629_362905

/-- The cost price per kg of the 80 kg of tea -/
def x : ℝ := 15

/-- The theorem stating that the cost price per kg of the 80 kg of tea is 15 -/
theorem tea_cost_price : 
  ∀ (quantity_1 quantity_2 cost_2 profit sale_price : ℝ),
  quantity_1 = 80 →
  quantity_2 = 20 →
  cost_2 = 20 →
  profit = 0.2 →
  sale_price = 19.2 →
  x = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_tea_cost_price_l3629_362905


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3629_362917

theorem quadratic_inequality_condition (a b c : ℝ) :
  (a > 0 ∧ b^2 - 4*a*c < 0) → (∀ x, a*x^2 + b*x + c > 0) ∧
  ¬(∀ x, a*x^2 + b*x + c > 0 → (a > 0 ∧ b^2 - 4*a*c < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3629_362917


namespace NUMINAMATH_CALUDE_probability_of_losing_l3629_362940

theorem probability_of_losing (p_win p_draw : ℚ) (h1 : p_win = 1/3) (h2 : p_draw = 1/2) 
  (h3 : p_win + p_draw + p_lose = 1) : p_lose = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_losing_l3629_362940


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3629_362920

/-- A geometric sequence with a_1 = 1 and a_3 = 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 3 = 2 ∧ ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : geometric_sequence a) :
  (a 5 + a 10) / (a 1 + a 6) = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3629_362920


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3629_362989

theorem quadratic_rewrite (a b c : ℤ) :
  (∀ x : ℚ, 16 * x^2 + 40 * x + 18 = (a * x + b)^2 + c) →
  a * b = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3629_362989


namespace NUMINAMATH_CALUDE_parallelogram_cut_slope_sum_l3629_362968

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Defines the specific parallelogram from the problem -/
def problemParallelogram : Parallelogram :=
  { v1 := { x := 15, y := 70 }
  , v2 := { x := 15, y := 210 }
  , v3 := { x := 45, y := 280 }
  , v4 := { x := 45, y := 140 }
  }

/-- A line through the origin with slope m/n -/
structure Line where
  m : ℕ
  n : ℕ
  coprime : Nat.Coprime m n

/-- Predicate to check if a line cuts the parallelogram into two congruent polygons -/
def cutsIntoCongruentPolygons (l : Line) (p : Parallelogram) : Prop :=
  sorry -- Definition omitted for brevity

theorem parallelogram_cut_slope_sum :
  ∃ (l : Line), cutsIntoCongruentPolygons l problemParallelogram ∧ l.m + l.n = 41 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_cut_slope_sum_l3629_362968


namespace NUMINAMATH_CALUDE_sum_of_integers_l3629_362919

theorem sum_of_integers (a b : ℕ+) (h1 : a - b = 14) (h2 : a * b = 120) : a + b = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3629_362919


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l3629_362965

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw_x : w / x = 4 / 3)
  (hy_z : y / z = 5 / 3)
  (hz_x : z / x = 1 / 6) :
  w / y = 24 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l3629_362965


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_l3629_362978

-- Define the function
def f (x : ℝ) : ℝ := x^2 + x - 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 2*x + 1

-- Theorem statement
theorem tangent_line_at_point_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let k : ℝ := f' x₀
  ∀ x y : ℝ, (k * (x - x₀) = y - y₀) ↔ (3*x - y - 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_l3629_362978


namespace NUMINAMATH_CALUDE_power_sum_equality_l3629_362950

theorem power_sum_equality : (-2)^1999 + (-2)^2000 = 2^1999 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3629_362950
