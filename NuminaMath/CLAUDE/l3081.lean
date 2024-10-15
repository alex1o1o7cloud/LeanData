import Mathlib

namespace NUMINAMATH_CALUDE_certain_number_problem_l3081_308176

theorem certain_number_problem (x : ℝ) : 
  (0.55 * x = (4 / 5 : ℝ) * 25 + 2) → x = 40 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3081_308176


namespace NUMINAMATH_CALUDE_train_crossing_time_l3081_308180

/-- Proves that a train with given length and speed takes a specific time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 900 →
  train_speed_kmh = 180 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3081_308180


namespace NUMINAMATH_CALUDE_B_power_103_l3081_308185

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, 1, 0;
     0, 0, 1;
     1, 0, 0]

theorem B_power_103 : B^103 = B := by sorry

end NUMINAMATH_CALUDE_B_power_103_l3081_308185


namespace NUMINAMATH_CALUDE_candy_bar_cost_l3081_308132

/-- The cost of a candy bar given initial amount and change --/
theorem candy_bar_cost (initial_amount change : ℕ) (h1 : initial_amount = 50) (h2 : change = 5) :
  initial_amount - change = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l3081_308132


namespace NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l3081_308158

/-- The volume of a cylinder formed by rotating a square about its horizontal line of symmetry -/
theorem volume_cylinder_from_square_rotation (side_length : ℝ) (volume : ℝ) :
  side_length = 16 →
  volume = π * side_length^3 / 4 →
  volume = 1024 * π :=
by sorry

end NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l3081_308158


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3081_308162

theorem algebraic_expression_equality (a b : ℝ) (h : 5 * a + 3 * b = -4) :
  2 * (a + b) + 4 * (2 * a + b) - 10 = -18 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3081_308162


namespace NUMINAMATH_CALUDE_power_of_product_l3081_308195

theorem power_of_product (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3081_308195


namespace NUMINAMATH_CALUDE_carla_book_count_l3081_308141

theorem carla_book_count (ceiling_tiles : ℕ) (tuesday_count : ℕ) : 
  ceiling_tiles = 38 → 
  tuesday_count = 301 → 
  ∃ (books : ℕ), 2 * ceiling_tiles + 3 * books = tuesday_count ∧ books = 75 :=
by sorry

end NUMINAMATH_CALUDE_carla_book_count_l3081_308141


namespace NUMINAMATH_CALUDE_complex_reciprocal_sum_l3081_308126

theorem complex_reciprocal_sum (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_complex_reciprocal_sum_l3081_308126


namespace NUMINAMATH_CALUDE_triangle_properties_l3081_308156

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def condition (t : Triangle) : Prop :=
  (Real.cos t.A - 2 * Real.cos t.C) / Real.cos t.B = (2 * t.c - t.a) / t.b

theorem triangle_properties (t : Triangle) (h : condition t) :
  Real.sin t.C / Real.sin t.A = 2 ∧
  (Real.cos t.B = 1/4 ∧ t.b = 2 → 
    1/2 * t.a * t.c * Real.sin t.B = Real.sqrt 15 / 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3081_308156


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3081_308115

/-- The perimeter of a rhombus inscribed in a rectangle --/
theorem rhombus_perimeter (w l : ℝ) (hw : w = 20) (hl : l = 25) :
  let s := Real.sqrt (w^2 / 4 + l^2 / 4)
  let perimeter := 4 * s
  ∃ ε > 0, abs (perimeter - 64.04) < ε := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3081_308115


namespace NUMINAMATH_CALUDE_power_division_nineteen_l3081_308197

theorem power_division_nineteen : 19^11 / 19^8 = 6859 := by sorry

end NUMINAMATH_CALUDE_power_division_nineteen_l3081_308197


namespace NUMINAMATH_CALUDE_lunas_budget_l3081_308170

/-- Luna's monthly budget problem -/
theorem lunas_budget (H F : ℝ) : 
  H + F = 240 →  -- Total budget for house rental and food
  H + F + 0.1 * F = 249 →  -- Total budget including phone bill
  F / H = 0.6  -- Food budget is 60% of house rental budget
:= by sorry

end NUMINAMATH_CALUDE_lunas_budget_l3081_308170


namespace NUMINAMATH_CALUDE_organizing_teams_count_l3081_308142

theorem organizing_teams_count (total_members senior_members team_size : ℕ) 
  (h1 : total_members = 12)
  (h2 : senior_members = 5)
  (h3 : team_size = 5) :
  (Nat.choose total_members team_size) - 
  ((Nat.choose (total_members - senior_members) team_size) + 
   (Nat.choose senior_members 1 * Nat.choose (total_members - senior_members) (team_size - 1))) = 596 := by
sorry

end NUMINAMATH_CALUDE_organizing_teams_count_l3081_308142


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l3081_308100

theorem framed_painting_ratio : 
  ∀ (y : ℝ),
  y > 0 →
  (20 + 2*y) * (30 + 6*y) = 2 * 20 * 30 →
  (min (20 + 2*y) (30 + 6*y)) / (max (20 + 2*y) (30 + 6*y)) = 4/7 :=
by sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l3081_308100


namespace NUMINAMATH_CALUDE_fraction_of_25_l3081_308129

theorem fraction_of_25 : 
  ∃ (x : ℚ), x * 25 + 8 = 70 * 40 / 100 ∧ x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_25_l3081_308129


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l3081_308199

theorem smallest_integer_satisfying_conditions :
  ∃ x : ℤ, (3 * |x| + 4 < 25) ∧ (x + 3 > 0) ∧
  (∀ y : ℤ, (3 * |y| + 4 < 25) ∧ (y + 3 > 0) → x ≤ y) ∧
  x = -3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l3081_308199


namespace NUMINAMATH_CALUDE_smallest_three_digit_prime_with_composite_reverse_l3081_308139

/-- A function that reverses the digits of a three-digit number -/
def reverseDigits (n : Nat) : Nat :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

/-- A predicate that checks if a number is prime -/
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- A predicate that checks if a number is composite -/
def isComposite (n : Nat) : Prop :=
  n > 1 ∧ ∃ d : Nat, d > 1 ∧ d < n ∧ n % d = 0

theorem smallest_three_digit_prime_with_composite_reverse :
  ∃ (p : Nat),
    p = 103 ∧
    isPrime p ∧
    100 ≤ p ∧ p < 1000 ∧
    isComposite (reverseDigits p) ∧
    ∀ (q : Nat),
      isPrime q ∧
      100 ≤ q ∧ q < p →
      ¬(isComposite (reverseDigits q)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_prime_with_composite_reverse_l3081_308139


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l3081_308169

theorem binomial_expansion_example : 
  8^4 + 4*(8^3)*2 + 6*(8^2)*(2^2) + 4*8*(2^3) + 2^4 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l3081_308169


namespace NUMINAMATH_CALUDE_xy_squared_sum_l3081_308163

theorem xy_squared_sum (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 3) :
  x^2 * y + x * y^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_sum_l3081_308163


namespace NUMINAMATH_CALUDE_tangent_line_is_correct_l3081_308175

/-- The equation of a parabola -/
def parabola (x : ℝ) : ℝ := 4 * x^2

/-- The slope of the tangent line at a given x-coordinate -/
def tangent_slope (x : ℝ) : ℝ := 8 * x

/-- The point of tangency -/
def point_of_tangency : ℝ × ℝ := (1, 4)

/-- The proposed equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 8 * x - y - 4 = 0

theorem tangent_line_is_correct :
  let (x₀, y₀) := point_of_tangency
  tangent_line x₀ y₀ ∧
  y₀ = parabola x₀ ∧
  (∀ x y, tangent_line x y ↔ y - y₀ = tangent_slope x₀ * (x - x₀)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_is_correct_l3081_308175


namespace NUMINAMATH_CALUDE_smarties_leftover_l3081_308121

theorem smarties_leftover (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_smarties_leftover_l3081_308121


namespace NUMINAMATH_CALUDE_finite_solutions_factorial_square_sum_l3081_308187

theorem finite_solutions_factorial_square_sum (a : ℕ) :
  ∃ (n : ℕ), ∀ (x y : ℕ), x! = y^2 + a^2 → x ≤ n :=
sorry

end NUMINAMATH_CALUDE_finite_solutions_factorial_square_sum_l3081_308187


namespace NUMINAMATH_CALUDE_limes_given_to_sara_l3081_308131

/-- Given that Dan initially picked some limes and gave some to Sara, 
    prove that the number of limes Dan gave to Sara is equal to 
    the difference between his initial and final number of limes. -/
theorem limes_given_to_sara 
  (initial_limes : ℕ) 
  (final_limes : ℕ) 
  (h1 : initial_limes = 9)
  (h2 : final_limes = 5) :
  initial_limes - final_limes = 4 := by
  sorry

end NUMINAMATH_CALUDE_limes_given_to_sara_l3081_308131


namespace NUMINAMATH_CALUDE_handshake_count_l3081_308153

/-- Represents a basketball game setup with two teams and referees -/
structure BasketballGame where
  team_size : Nat
  coach_per_team : Nat
  referee_count : Nat

/-- Calculates the total number of handshakes in a basketball game -/
def total_handshakes (game : BasketballGame) : Nat :=
  let inter_team_handshakes := game.team_size * game.team_size
  let total_team_members := game.team_size + game.coach_per_team
  let intra_team_handshakes := 2 * (total_team_members.choose 2)
  let team_referee_handshakes := 2 * total_team_members * game.referee_count
  let referee_handshakes := game.referee_count.choose 2
  inter_team_handshakes + intra_team_handshakes + team_referee_handshakes + referee_handshakes

/-- The main theorem stating the total number of handshakes in the given game setup -/
theorem handshake_count :
  let game : BasketballGame := {
    team_size := 6
    coach_per_team := 1
    referee_count := 2
  }
  total_handshakes game = 107 := by
  sorry


end NUMINAMATH_CALUDE_handshake_count_l3081_308153


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3081_308113

theorem sqrt_equation_solution (x : ℝ) :
  (3 * x - 2 > 0) →
  (Real.sqrt (3 * x - 2) + 9 / Real.sqrt (3 * x - 2) = 6) ↔
  (x = 11 / 3) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3081_308113


namespace NUMINAMATH_CALUDE_license_plate_count_is_9360_l3081_308116

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possibilities for the second character (letter or digit) -/
def num_second_char : ℕ := num_letters + num_digits

/-- The number of ways to design a 4-character license plate with the given conditions -/
def license_plate_count : ℕ := num_letters * num_second_char * 1 * num_digits

theorem license_plate_count_is_9360 : license_plate_count = 9360 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_is_9360_l3081_308116


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3081_308147

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im (i^2 * (1 + i)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3081_308147


namespace NUMINAMATH_CALUDE_calculation_one_l3081_308136

theorem calculation_one : (-3/8) + (-5/8) * (-6) = 27/8 := by sorry

end NUMINAMATH_CALUDE_calculation_one_l3081_308136


namespace NUMINAMATH_CALUDE_can_add_flights_to_5000_l3081_308149

/-- A graph representing cities and flights --/
structure CityGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  edge_symmetric : ∀ {a b}, (a, b) ∈ edges → (b, a) ∈ edges
  no_self_loops : ∀ {a}, (a, a) ∉ edges

/-- The number of cities --/
def num_cities : Nat := 998

/-- Check if the graph satisfies the flight laws --/
def satisfies_laws (g : CityGraph) : Prop :=
  (g.vertices.card = num_cities) ∧
  (∀ k : Finset Nat, k ⊆ g.vertices →
    (g.edges.filter (fun e => e.1 ∈ k ∧ e.2 ∈ k)).card ≤ 5 * k.card + 10)

/-- The theorem to be proved --/
theorem can_add_flights_to_5000 (g : CityGraph) (h : satisfies_laws g) :
  ∃ g' : CityGraph, satisfies_laws g' ∧
    g.edges ⊆ g'.edges ∧
    g'.edges.card = 5000 := by
  sorry

end NUMINAMATH_CALUDE_can_add_flights_to_5000_l3081_308149


namespace NUMINAMATH_CALUDE_raja_medicine_percentage_l3081_308105

/-- Raja's monthly expenses and savings --/
def monthly_expenses (income medicine_percentage : ℝ) : Prop :=
  let household_percentage : ℝ := 0.35
  let clothes_percentage : ℝ := 0.20
  let savings : ℝ := 15000
  household_percentage * income + 
  clothes_percentage * income + 
  medicine_percentage * income + 
  savings = income

theorem raja_medicine_percentage : 
  ∃ (medicine_percentage : ℝ), 
    monthly_expenses 37500 medicine_percentage ∧ 
    medicine_percentage = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_raja_medicine_percentage_l3081_308105


namespace NUMINAMATH_CALUDE_real_part_of_complex_number_l3081_308109

theorem real_part_of_complex_number (i : ℂ) (h : i^2 = -1) : 
  Complex.re ((-1 + 2*i)*i) = -2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_number_l3081_308109


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3081_308168

/-- Given a hyperbola with the equation (x²/a² - y²/b² = 1) where a > 0 and b > 0,
    and a line y = 2x - 4 that passes through the right focus F and intersects
    the hyperbola at only one point, prove that the equation of the hyperbola
    is (5x²/4 - 5y²/16 = 1). -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ F : ℝ × ℝ,
    (F.1 > 0) ∧  -- F is the right focus
    (F.2 = 0) ∧  -- F is on the x-axis
    (∀ x y : ℝ, y = 2*x - 4 → (x - F.1)^2 + y^2 = (a^2 + b^2)) ∧  -- line passes through F
    (∃! P : ℝ × ℝ, P.2 = 2*P.1 - 4 ∧ P.1^2/a^2 - P.2^2/b^2 = 1))  -- line intersects hyperbola at one point
  →
  a^2 = 4/5 ∧ b^2 = 16/5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3081_308168


namespace NUMINAMATH_CALUDE_f_properties_l3081_308112

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x - a) - Real.log (x + a)

theorem f_properties :
  (∀ x > -1/2, f (1/2) x < f (1/2) (1/2)) ∧ 
  (∀ x > 1/2, f (1/2) x > f (1/2) (1/2)) ∧
  (f (1/2) (1/2) = 1) ∧
  (∀ a ≤ 1, ∀ x > -a, f a x > 0) := by sorry

end NUMINAMATH_CALUDE_f_properties_l3081_308112


namespace NUMINAMATH_CALUDE_daily_wage_of_c_l3081_308151

theorem daily_wage_of_c (a b c : ℕ) (total_earning : ℚ) : 
  a = 6 ∧ b = 9 ∧ c = 4 → 
  ∃ (x : ℚ), 
    (3 * x * a + 4 * x * b + 5 * x * c = total_earning) ∧
    (total_earning = 1480) →
    5 * x = 100 := by
  sorry

end NUMINAMATH_CALUDE_daily_wage_of_c_l3081_308151


namespace NUMINAMATH_CALUDE_complex_number_representation_l3081_308114

theorem complex_number_representation : ∃ (z : ℂ), z = 1 + 2*I := by sorry

end NUMINAMATH_CALUDE_complex_number_representation_l3081_308114


namespace NUMINAMATH_CALUDE_donut_circumference_ratio_l3081_308165

/-- The ratio of the outer circumference to the inner circumference of a donut-shaped object
    is equal to the ratio of their respective radii. -/
theorem donut_circumference_ratio (inner_radius outer_radius : ℝ)
  (h1 : inner_radius = 2)
  (h2 : outer_radius = 6) :
  (2 * Real.pi * outer_radius) / (2 * Real.pi * inner_radius) = outer_radius / inner_radius := by
  sorry

end NUMINAMATH_CALUDE_donut_circumference_ratio_l3081_308165


namespace NUMINAMATH_CALUDE_reflected_line_x_intercept_l3081_308173

/-- The x-intercept of a line reflected in the y-axis -/
theorem reflected_line_x_intercept :
  let original_line : ℝ → ℝ := λ x => 2 * x - 6
  let reflected_line : ℝ → ℝ := λ x => -2 * x - 6
  let x_intercept : ℝ := -3
  (reflected_line x_intercept = 0) ∧ 
  (∀ y : ℝ, reflected_line y = 0 → y = x_intercept) :=
by sorry

end NUMINAMATH_CALUDE_reflected_line_x_intercept_l3081_308173


namespace NUMINAMATH_CALUDE_smallest_bob_number_l3081_308130

def alice_number : ℕ := 36

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ m → p ∣ n)

def is_multiple_of_five (n : ℕ) : Prop :=
  5 ∣ n

theorem smallest_bob_number :
  ∃ (bob_number : ℕ),
    has_all_prime_factors bob_number alice_number ∧
    is_multiple_of_five bob_number ∧
    (∀ k : ℕ, k < bob_number →
      ¬(has_all_prime_factors k alice_number ∧ is_multiple_of_five k)) ∧
    bob_number = 30 :=
by sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l3081_308130


namespace NUMINAMATH_CALUDE_expression_factorization_l3081_308144

theorem expression_factorization (y : ℝ) : 
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 - 9) = 6 * y^4 * (2 * y^2 + 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3081_308144


namespace NUMINAMATH_CALUDE_negation_of_existence_original_proposition_negation_l3081_308106

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x > 1, p x) ↔ ∀ x > 1, ¬ p x := by sorry

theorem original_proposition_negation :
  (¬ ∃ x > 1, 3*x + 1 > 5) ↔ (∀ x > 1, 3*x + 1 ≤ 5) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_original_proposition_negation_l3081_308106


namespace NUMINAMATH_CALUDE_alternative_rate_calculation_l3081_308198

/-- Calculates simple interest -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time

theorem alternative_rate_calculation (principal : ℚ) (time : ℚ) (actual_rate : ℚ) 
  (interest_difference : ℚ) (alternative_rate : ℚ) : 
  principal = 2500 →
  time = 2 →
  actual_rate = 18 / 100 →
  interest_difference = 300 →
  simple_interest principal actual_rate time - simple_interest principal alternative_rate time = interest_difference →
  alternative_rate = 12 / 100 := by
sorry

end NUMINAMATH_CALUDE_alternative_rate_calculation_l3081_308198


namespace NUMINAMATH_CALUDE_non_black_cows_l3081_308119

theorem non_black_cows (total : ℕ) (black : ℕ) (h1 : total = 18) (h2 : black = total / 2 + 5) :
  total - black = 4 := by
sorry

end NUMINAMATH_CALUDE_non_black_cows_l3081_308119


namespace NUMINAMATH_CALUDE_distance_can_be_four_l3081_308182

/-- A circle with radius 3 -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (h_radius : radius = 3)

/-- A point outside the circle -/
structure OutsidePoint (c : Circle) :=
  (point : ℝ × ℝ)
  (h_outside : dist point c.center > c.radius)

/-- The theorem stating that the distance between the center and the outside point can be 4 -/
theorem distance_can_be_four (c : Circle) (p : OutsidePoint c) : 
  ∃ (q : OutsidePoint c), dist q.point c.center = 4 :=
sorry

end NUMINAMATH_CALUDE_distance_can_be_four_l3081_308182


namespace NUMINAMATH_CALUDE_train_sequence_count_l3081_308161

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- The total number of departure sequences for 6 trains under given conditions -/
def train_sequences : ℕ :=
  let total_trains : ℕ := 6
  let trains_per_group : ℕ := 3
  let remaining_trains : ℕ := total_trains - 2  -- excluding A and B
  let ways_to_group : ℕ := choose remaining_trains (trains_per_group - 1)
  let ways_to_arrange_group : ℕ := factorial trains_per_group
  ways_to_group * ways_to_arrange_group * ways_to_arrange_group

theorem train_sequence_count : train_sequences = 216 := by sorry

end NUMINAMATH_CALUDE_train_sequence_count_l3081_308161


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3081_308120

/-- A quadratic equation in terms of x is of the form ax^2 + bx + c = 0, where a ≠ 0, b, and c are real numbers. -/
def IsQuadraticEquation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing 2x^2 - 1 = 0 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : IsQuadraticEquation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3081_308120


namespace NUMINAMATH_CALUDE_hip_size_conversion_l3081_308127

/-- Converts inches to millimeters given the conversion factors -/
def inches_to_mm (inches_per_foot : ℚ) (mm_per_foot : ℚ) (inches : ℚ) : ℚ :=
  inches * (mm_per_foot / inches_per_foot)

/-- Proves that 42 inches is equivalent to 1067.5 millimeters -/
theorem hip_size_conversion (inches_per_foot mm_per_foot : ℚ) 
  (h1 : inches_per_foot = 12)
  (h2 : mm_per_foot = 305) : 
  inches_to_mm inches_per_foot mm_per_foot 42 = 1067.5 := by
  sorry

#eval inches_to_mm 12 305 42

end NUMINAMATH_CALUDE_hip_size_conversion_l3081_308127


namespace NUMINAMATH_CALUDE_new_person_weight_l3081_308122

/-- The weight of the new person given the conditions of the problem -/
theorem new_person_weight (n : ℕ) (initial_weight : ℝ) (weight_increase : ℝ) : 
  n = 9 → 
  initial_weight = 86 → 
  weight_increase = 5.5 → 
  (n : ℝ) * weight_increase + initial_weight = 135.5 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l3081_308122


namespace NUMINAMATH_CALUDE_cos_420_plus_sin_330_equals_zero_l3081_308192

theorem cos_420_plus_sin_330_equals_zero :
  Real.cos (420 * π / 180) + Real.sin (330 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_420_plus_sin_330_equals_zero_l3081_308192


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l3081_308134

-- Define a line type
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

-- Define a line passing through a point
def passes_through (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

-- Define the given line
def given_line : Line := { slope := 2, y_intercept := 4 }

-- Define the point that line b passes through
def given_point : Point := { x := 3, y := 7 }

-- Theorem statement
theorem y_intercept_of_parallel_line :
  ∃ (b : Line),
    parallel b given_line ∧
    passes_through b given_point ∧
    b.y_intercept = 1 := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l3081_308134


namespace NUMINAMATH_CALUDE_shoe_size_mode_and_median_l3081_308183

def shoe_sizes : List ℝ := [25, 25, 25.5, 25.5, 25.5, 25.5, 26, 26, 26.5, 27]

def mode (list : List ℝ) : ℝ := sorry

def median (list : List ℝ) : ℝ := sorry

theorem shoe_size_mode_and_median :
  mode shoe_sizes = 25.5 ∧ median shoe_sizes = 25.5 := by sorry

end NUMINAMATH_CALUDE_shoe_size_mode_and_median_l3081_308183


namespace NUMINAMATH_CALUDE_average_equals_black_dots_l3081_308184

/-- Represents the types of butterflies -/
inductive ButterflyType
  | A
  | B
  | C

/-- Returns the number of black dots for a given butterfly type -/
def blackDots (t : ButterflyType) : ℕ :=
  match t with
  | .A => 545
  | .B => 780
  | .C => 1135

/-- Returns the number of butterflies for a given type -/
def butterflyCount (t : ButterflyType) : ℕ :=
  match t with
  | .A => 15
  | .B => 25
  | .C => 35

/-- Calculates the average number of black dots per butterfly for a given type -/
def averageBlackDots (t : ButterflyType) : ℚ :=
  (blackDots t : ℚ) * (butterflyCount t : ℚ) / (butterflyCount t : ℚ)

theorem average_equals_black_dots (t : ButterflyType) :
  averageBlackDots t = blackDots t := by
  sorry

#eval averageBlackDots ButterflyType.A
#eval averageBlackDots ButterflyType.B
#eval averageBlackDots ButterflyType.C

end NUMINAMATH_CALUDE_average_equals_black_dots_l3081_308184


namespace NUMINAMATH_CALUDE_cost_of_flour_l3081_308143

/-- Given the total cost of flour and cake stand, and the cost of the cake stand,
    prove that the cost of flour is $5. -/
theorem cost_of_flour (total_cost cake_stand_cost : ℕ)
  (h1 : total_cost = 33)
  (h2 : cake_stand_cost = 28) :
  total_cost - cake_stand_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_flour_l3081_308143


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l3081_308160

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (keeper_age_diff : ℕ) (team_avg_age : ℚ) :
  team_size = 11 →
  captain_age = 26 →
  keeper_age_diff = 3 →
  team_avg_age = 23 →
  let keeper_age := captain_age + keeper_age_diff
  let total_team_age := team_avg_age * team_size
  let remaining_players := team_size - 2
  let remaining_age := total_team_age - (captain_age + keeper_age)
  let remaining_avg_age := remaining_age / remaining_players
  (team_avg_age - remaining_avg_age) = 1 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l3081_308160


namespace NUMINAMATH_CALUDE_xy_sum_difference_l3081_308164

theorem xy_sum_difference (x y : ℝ) 
  (h1 : x + Real.sqrt (x * y) + y = 9) 
  (h2 : x^2 + x*y + y^2 = 27) : 
  x - Real.sqrt (x * y) + y = 3 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_difference_l3081_308164


namespace NUMINAMATH_CALUDE_project_hours_difference_l3081_308124

/-- 
Given a project where:
- The total hours charged is 180
- Pat charged twice as much time as Kate
- Pat charged 1/3 as much time as Mark

Prove that Mark charged 100 more hours than Kate.
-/
theorem project_hours_difference (kate : ℝ) (pat : ℝ) (mark : ℝ) 
  (h1 : kate + pat + mark = 180)
  (h2 : pat = 2 * kate)
  (h3 : pat = (1/3) * mark) : 
  mark - kate = 100 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_difference_l3081_308124


namespace NUMINAMATH_CALUDE_sin_cos_45_sum_l3081_308128

theorem sin_cos_45_sum : Real.sin (π / 4) + Real.cos (π / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_45_sum_l3081_308128


namespace NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l3081_308172

theorem x_range_for_quadratic_inequality :
  ∀ x : ℝ, (∀ a : ℝ, a ∈ Set.Icc (-3) 3 → x^2 - a*x + 1 ≥ 1) →
  x ≥ (3 + Real.sqrt 5) / 2 ∨ x ≤ (-3 - Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l3081_308172


namespace NUMINAMATH_CALUDE_technician_salary_l3081_308108

/-- Proves that the average salary of technicians is 12000 given the workshop conditions --/
theorem technician_salary (total_workers : ℕ) (technicians : ℕ) (avg_salary : ℕ) (non_tech_salary : ℕ) :
  total_workers = 21 →
  technicians = 7 →
  avg_salary = 8000 →
  non_tech_salary = 6000 →
  (avg_salary * total_workers = 12000 * technicians + non_tech_salary * (total_workers - technicians)) :=
by
  sorry

#check technician_salary

end NUMINAMATH_CALUDE_technician_salary_l3081_308108


namespace NUMINAMATH_CALUDE_skyline_hospital_quadruplets_l3081_308135

theorem skyline_hospital_quadruplets :
  ∀ (twins triplets quads : ℕ),
    triplets = 5 * quads →
    twins = 3 * triplets →
    2 * twins + 3 * triplets + 4 * quads = 1200 →
    4 * quads = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_skyline_hospital_quadruplets_l3081_308135


namespace NUMINAMATH_CALUDE_middle_card_is_five_l3081_308103

/-- Represents a triple of distinct positive integers in ascending order --/
structure CardTriple where
  left : Nat
  middle : Nat
  right : Nat
  distinct : left < middle ∧ middle < right
  sum_20 : left + middle + right = 20

/-- Predicate for Bella's statement --/
def bella_cant_determine (t : CardTriple) : Prop :=
  ∃ t' : CardTriple, t'.left = t.left ∧ t' ≠ t

/-- Predicate for Della's statement --/
def della_cant_determine (t : CardTriple) : Prop :=
  ∃ t' : CardTriple, t'.middle = t.middle ∧ t' ≠ t

/-- Predicate for Nella's statement --/
def nella_cant_determine (t : CardTriple) : Prop :=
  ∃ t' : CardTriple, t'.right = t.right ∧ t' ≠ t

/-- The main theorem --/
theorem middle_card_is_five :
  ∀ t : CardTriple,
    bella_cant_determine t →
    della_cant_determine t →
    nella_cant_determine t →
    t.middle = 5 := by
  sorry

end NUMINAMATH_CALUDE_middle_card_is_five_l3081_308103


namespace NUMINAMATH_CALUDE_only_vegetarian_count_l3081_308186

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  total_veg : ℕ
  only_nonveg : ℕ
  both : ℕ

/-- Given the specified family diet, prove that the number of people who eat only vegetarian is 13 -/
theorem only_vegetarian_count (f : FamilyDiet) 
  (h1 : f.total_veg = 21)
  (h2 : f.only_nonveg = 7)
  (h3 : f.both = 8) :
  f.total_veg - f.both = 13 := by
  sorry

#check only_vegetarian_count

end NUMINAMATH_CALUDE_only_vegetarian_count_l3081_308186


namespace NUMINAMATH_CALUDE_min_cubes_for_specific_box_l3081_308110

/-- The minimum number of cubes required to build a box -/
def min_cubes (length width height cube_size : ℕ) : ℕ :=
  (length * width * height) / (cube_size ^ 3)

/-- Theorem: The minimum number of 3 cm³ cubes required to build a 9 cm × 12 cm × 3 cm box is 108 -/
theorem min_cubes_for_specific_box :
  min_cubes 9 12 3 3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_specific_box_l3081_308110


namespace NUMINAMATH_CALUDE_min_attacking_pairs_8x8_16rooks_l3081_308140

/-- Represents a chessboard configuration with rooks -/
structure ChessboardWithRooks where
  size : Nat
  num_rooks : Nat
  rook_positions : List (Nat × Nat)
  different_squares : rook_positions.length = num_rooks ∧ 
                      rook_positions.Nodup

/-- Counts the number of pairs of rooks that can attack each other -/
def count_attacking_pairs (board : ChessboardWithRooks) : Nat :=
  sorry

/-- Theorem stating the minimum number of attacking pairs for a specific configuration -/
theorem min_attacking_pairs_8x8_16rooks :
  ∀ (board : ChessboardWithRooks),
    board.size = 8 ∧ 
    board.num_rooks = 16 →
    count_attacking_pairs board ≥ 16 :=
  sorry

end NUMINAMATH_CALUDE_min_attacking_pairs_8x8_16rooks_l3081_308140


namespace NUMINAMATH_CALUDE_fraction_equality_l3081_308167

theorem fraction_equality (a b : ℝ) (h : a / b = 3 / 4) : (b - a) / b = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3081_308167


namespace NUMINAMATH_CALUDE_meal_cost_calculation_l3081_308137

theorem meal_cost_calculation (initial_friends : ℕ) (additional_friends : ℕ) 
  (cost_decrease : ℚ) (total_cost : ℚ) : 
  initial_friends = 4 →
  additional_friends = 5 →
  cost_decrease = 6 →
  (total_cost / initial_friends.cast) - (total_cost / (initial_friends + additional_friends).cast) = cost_decrease →
  total_cost = 216/5 := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_calculation_l3081_308137


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3081_308190

theorem quadratic_function_range (a : ℝ) : 
  (∃ y₁ y₂ y₃ y₄ : ℝ, 
    (y₁ = a * (-4)^2 + 4 * a * (-4) - 6) ∧
    (y₂ = a * (-3)^2 + 4 * a * (-3) - 6) ∧
    (y₃ = a * 0^2 + 4 * a * 0 - 6) ∧
    (y₄ = a * 2^2 + 4 * a * 2 - 6) ∧
    ((y₁ > 0 ∧ y₂ ≤ 0 ∧ y₃ ≤ 0 ∧ y₄ ≤ 0) ∨
     (y₁ ≤ 0 ∧ y₂ > 0 ∧ y₃ ≤ 0 ∧ y₄ ≤ 0) ∨
     (y₁ ≤ 0 ∧ y₂ ≤ 0 ∧ y₃ > 0 ∧ y₄ ≤ 0) ∨
     (y₁ ≤ 0 ∧ y₂ ≤ 0 ∧ y₃ ≤ 0 ∧ y₄ > 0))) →
  (a < -2 ∨ a > 1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3081_308190


namespace NUMINAMATH_CALUDE_polynomial_parity_l3081_308157

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial := List Int

/-- Multiplies two polynomials -/
def polyMul (p q : IntPolynomial) : IntPolynomial := sorry

/-- Checks if all coefficients of a polynomial are even -/
def allCoeffsEven (p : IntPolynomial) : Prop := sorry

/-- Checks if all coefficients of a polynomial are divisible by 4 -/
def allCoeffsDivBy4 (p : IntPolynomial) : Prop := sorry

/-- Checks if a polynomial has at least one odd coefficient -/
def hasOddCoeff (p : IntPolynomial) : Prop := sorry

theorem polynomial_parity (p q : IntPolynomial) :
  (allCoeffsEven (polyMul p q)) ∧ ¬(allCoeffsDivBy4 (polyMul p q)) →
  (allCoeffsEven p ∧ hasOddCoeff q) ∨ (allCoeffsEven q ∧ hasOddCoeff p) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_parity_l3081_308157


namespace NUMINAMATH_CALUDE_angies_taxes_paid_l3081_308189

/-- Represents the weekly expenses for necessities, taxes, and utilities -/
structure WeeklyExpenses where
  necessities : ℕ
  taxes : ℕ
  utilities : ℕ

/-- Represents Angie's monthly finances -/
structure MonthlyFinances where
  salary : ℕ
  week1 : WeeklyExpenses
  week2 : WeeklyExpenses
  week3 : WeeklyExpenses
  week4 : WeeklyExpenses
  leftover : ℕ

/-- Calculates the total taxes paid in a month -/
def totalTaxesPaid (finances : MonthlyFinances) : ℕ :=
  finances.week1.taxes + finances.week2.taxes + finances.week3.taxes + finances.week4.taxes

/-- Theorem stating that Angie's total taxes paid for the month is $30 -/
theorem angies_taxes_paid (finances : MonthlyFinances) 
    (h1 : finances.salary = 80)
    (h2 : finances.week1 = ⟨12, 8, 5⟩)
    (h3 : finances.week2 = ⟨15, 6, 7⟩)
    (h4 : finances.week3 = ⟨10, 9, 6⟩)
    (h5 : finances.week4 = ⟨14, 7, 4⟩)
    (h6 : finances.leftover = 18) :
    totalTaxesPaid finances = 30 := by
  sorry

#eval totalTaxesPaid ⟨80, ⟨12, 8, 5⟩, ⟨15, 6, 7⟩, ⟨10, 9, 6⟩, ⟨14, 7, 4⟩, 18⟩

end NUMINAMATH_CALUDE_angies_taxes_paid_l3081_308189


namespace NUMINAMATH_CALUDE_problem_solution_l3081_308118

def three_digit_number (x y z : ℕ) : ℕ := 100 * x + 10 * y + z

theorem problem_solution (a b : ℕ) : 
  (three_digit_number 5 b 9) - (three_digit_number 2 a 3) = 326 →
  (three_digit_number 5 6 9) % 9 = 0 →
  a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3081_308118


namespace NUMINAMATH_CALUDE_system_solution_l3081_308181

theorem system_solution (a : ℕ+) 
  (h_system : ∃ (x y : ℝ), a * x + y = -4 ∧ 2 * x + y = -2 ∧ x < 0 ∧ y > 0) :
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3081_308181


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_main_theorem_l3081_308155

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem main_theorem : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 72 = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_main_theorem_l3081_308155


namespace NUMINAMATH_CALUDE_range_of_m_l3081_308166

-- Define p and q as functions of x and m
def p (x : ℝ) : Prop := |x - 3| ≤ 2

def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (p q : Prop) : Prop :=
  (q → p) ∧ ¬(p → q)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, necessary_not_sufficient (¬(p x)) (¬(q x m))) →
  (m ≥ 2 ∧ m ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3081_308166


namespace NUMINAMATH_CALUDE_parking_lot_buses_l3081_308148

/-- The total number of buses in a parking lot after more buses arrive -/
def total_buses (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Given 7 initial buses and 6 additional buses, the total is 13 -/
theorem parking_lot_buses : total_buses 7 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_buses_l3081_308148


namespace NUMINAMATH_CALUDE_stadium_entry_exit_options_l3081_308188

theorem stadium_entry_exit_options (south_gates north_gates : ℕ) 
  (h1 : south_gates = 4) 
  (h2 : north_gates = 3) : 
  (south_gates + north_gates) * (south_gates + north_gates) = 49 := by
  sorry

end NUMINAMATH_CALUDE_stadium_entry_exit_options_l3081_308188


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3081_308101

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3081_308101


namespace NUMINAMATH_CALUDE_meaningful_range_l3081_308107

def is_meaningful (x : ℝ) : Prop :=
  x + 1 ≥ 0 ∧ x ≠ 0

theorem meaningful_range :
  ∀ x : ℝ, is_meaningful x ↔ x ≥ -1 ∧ x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_range_l3081_308107


namespace NUMINAMATH_CALUDE_bug_probability_after_10_moves_l3081_308138

/-- Probability of the bug being at the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n+1 => (1/3) * (1 - Q n)

/-- The probability of the bug returning to its starting vertex on a square after 10 moves is 3431/19683 -/
theorem bug_probability_after_10_moves :
  Q 10 = 3431 / 19683 := by
  sorry

end NUMINAMATH_CALUDE_bug_probability_after_10_moves_l3081_308138


namespace NUMINAMATH_CALUDE_people_in_line_l3081_308125

theorem people_in_line (people_between : ℕ) (h : people_between = 5) : 
  people_between + 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_people_in_line_l3081_308125


namespace NUMINAMATH_CALUDE_hair_cut_total_l3081_308102

theorem hair_cut_total : 
  let monday : ℚ := 38 / 100
  let tuesday : ℚ := 1 / 2
  let wednesday : ℚ := 1 / 4
  let thursday : ℚ := 87 / 100
  monday + tuesday + wednesday + thursday = 2 := by sorry

end NUMINAMATH_CALUDE_hair_cut_total_l3081_308102


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_absolute_value_l3081_308150

-- Define the function
def f (x : ℝ) : ℝ := |x - 4| + 1

-- State the theorem
theorem minimum_point_of_translated_absolute_value :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x₀ ≤ f x) ∧ (x₀ = 4 ∧ f x₀ = 1) :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_absolute_value_l3081_308150


namespace NUMINAMATH_CALUDE_perfect_square_concatenation_l3081_308145

theorem perfect_square_concatenation (b m : ℕ) (h_b_odd : Odd b) :
  let A : ℕ := (5^b + 1) / 2
  let B : ℕ := 2^b * A * 100^m
  let AB : ℕ := 10^(Nat.digits 10 B).length * A + B
  ∃ (n : ℕ), AB = n^2 ∧ AB = 2 * A * B := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_concatenation_l3081_308145


namespace NUMINAMATH_CALUDE_pastries_made_pastries_made_correct_l3081_308123

/-- Given information about Baker's cakes and pastries -/
structure BakerInfo where
  cakes_made : ℕ
  cakes_sold : ℕ
  pastries_sold : ℕ
  cakes_pastries_diff : ℕ
  h1 : cakes_made = 157
  h2 : cakes_sold = 158
  h3 : pastries_sold = 147
  h4 : cakes_sold - pastries_sold = cakes_pastries_diff
  h5 : cakes_pastries_diff = 11

/-- Theorem stating the number of pastries Baker made -/
theorem pastries_made (info : BakerInfo) : ℕ := by
  sorry

#check @pastries_made

/-- The actual number of pastries Baker made -/
def actual_pastries_made : ℕ := 146

/-- Theorem proving that the calculated number of pastries matches the actual number -/
theorem pastries_made_correct (info : BakerInfo) : pastries_made info = actual_pastries_made := by
  sorry

end NUMINAMATH_CALUDE_pastries_made_pastries_made_correct_l3081_308123


namespace NUMINAMATH_CALUDE_simplify_expression_l3081_308159

theorem simplify_expression (x : ℝ) : 120*x - 32*x + 15 - 15 = 88*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3081_308159


namespace NUMINAMATH_CALUDE_product_of_symmetrical_complex_l3081_308146

/-- Two complex numbers are symmetrical about y = x if their real and imaginary parts are swapped -/
def symmetrical_about_y_eq_x (z₁ z₂ : ℂ) : Prop :=
  z₁.re = z₂.im ∧ z₁.im = z₂.re

theorem product_of_symmetrical_complex : ∀ z₁ z₂ : ℂ,
  symmetrical_about_y_eq_x z₁ z₂ →
  z₁ = 3 + 2*I →
  z₁ * z₂ = 13*I :=
sorry

end NUMINAMATH_CALUDE_product_of_symmetrical_complex_l3081_308146


namespace NUMINAMATH_CALUDE_differential_system_properties_l3081_308152

-- Define the system of differential equations
def system_ode (u : ℝ → ℝ) (x y : ℝ → ℝ) : Prop :=
  ∀ t, deriv x t = -2 * y t + u t ∧ deriv y t = -2 * x t + u t

-- Define the theorem
theorem differential_system_properties
  (u : ℝ → ℝ) (x y : ℝ → ℝ) (x₀ y₀ : ℝ)
  (h_cont : Continuous u)
  (h_system : system_ode u x y)
  (h_init : x 0 = x₀ ∧ y 0 = y₀) :
  (x₀ ≠ y₀ → ∀ t, x t - y t ≠ 0) ∧
  (x₀ = y₀ → ∀ T > 0, ∃ u : ℝ → ℝ, Continuous u ∧ x T = 0 ∧ y T = 0) :=
sorry

end NUMINAMATH_CALUDE_differential_system_properties_l3081_308152


namespace NUMINAMATH_CALUDE_solve_for_q_l3081_308174

theorem solve_for_q (k l q : ℚ) 
  (eq1 : 3/4 = k/48)
  (eq2 : 3/4 = (k + l)/56)
  (eq3 : 3/4 = (q - l)/160) : 
  q = 126 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l3081_308174


namespace NUMINAMATH_CALUDE_manoj_transaction_gain_l3081_308191

/-- Calculate simple interest -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (principal * rate * time) / 100

/-- The amount Manoj borrowed from Anwar -/
def borrowed_amount : ℚ := 3900

/-- The interest rate Manoj pays to Anwar -/
def borrowing_rate : ℚ := 6

/-- The amount Manoj lent to Ramu -/
def lent_amount : ℚ := 5655

/-- The interest rate Manoj charges Ramu -/
def lending_rate : ℚ := 9

/-- The time period for both transactions in years -/
def time_period : ℚ := 3

/-- Manoj's gain from the transaction -/
def manoj_gain : ℚ :=
  simple_interest lent_amount lending_rate time_period -
  simple_interest borrowed_amount borrowing_rate time_period

theorem manoj_transaction_gain :
  manoj_gain = 824.85 := by sorry

end NUMINAMATH_CALUDE_manoj_transaction_gain_l3081_308191


namespace NUMINAMATH_CALUDE_student_distribution_l3081_308133

/-- The number of ways to distribute n students between two cities --/
def distribute (n : ℕ) (min1 min2 : ℕ) : ℕ :=
  (Finset.range (n - min1 - min2 + 1)).sum (λ k => Nat.choose n (min1 + k))

/-- The theorem stating the number of arrangements for 6 students --/
theorem student_distribution : distribute 6 2 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_student_distribution_l3081_308133


namespace NUMINAMATH_CALUDE_eva_total_score_2019_l3081_308178

/-- Represents Eva's scores in a semester -/
structure SemesterScores where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Calculates the total score for a semester -/
def totalScore (scores : SemesterScores) : ℕ :=
  scores.maths + scores.arts + scores.science

/-- Represents Eva's scores for the year 2019 -/
structure YearScores where
  firstSemester : SemesterScores
  secondSemester : SemesterScores

/-- Theorem stating Eva's total score for 2019 -/
theorem eva_total_score_2019 (scores : YearScores) : 
  totalScore scores.firstSemester + totalScore scores.secondSemester = 485 :=
  by
  have h1 : scores.firstSemester.maths = scores.secondSemester.maths + 10 := by sorry
  have h2 : scores.firstSemester.arts = scores.secondSemester.arts - 15 := by sorry
  have h3 : scores.firstSemester.science = scores.secondSemester.science - scores.secondSemester.science / 3 := by sorry
  have h4 : scores.secondSemester.maths = 80 := by sorry
  have h5 : scores.secondSemester.arts = 90 := by sorry
  have h6 : scores.secondSemester.science = 90 := by sorry
  sorry

end NUMINAMATH_CALUDE_eva_total_score_2019_l3081_308178


namespace NUMINAMATH_CALUDE_remaining_area_of_semicircle_l3081_308194

theorem remaining_area_of_semicircle (d : ℝ) (h : d > 0) :
  let r := d / 2
  let chord_length := 2 * Real.sqrt 7
  chord_length ^ 2 + r ^ 2 = d ^ 2 →
  (π * r ^ 2 / 2) - 2 * (π * (r / 2) ^ 2 / 2) = 7 * π :=
by sorry

end NUMINAMATH_CALUDE_remaining_area_of_semicircle_l3081_308194


namespace NUMINAMATH_CALUDE_smallest_positive_m_squared_l3081_308111

/-- Definition of circle w₁ -/
def w₁ (x y : ℝ) : Prop := x^2 + y^2 + 10*x - 24*y - 87 = 0

/-- Definition of circle w₂ -/
def w₂ (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 24*y + 153 = 0

/-- Definition of a line y = ax -/
def line (a x y : ℝ) : Prop := y = a * x

/-- Definition of external tangency to w₂ -/
def externally_tangent_w₂ (x y r : ℝ) : Prop :=
  (x - 5)^2 + (y - 12)^2 = (r + 4)^2

/-- Definition of internal tangency to w₁ -/
def internally_tangent_w₁ (x y r : ℝ) : Prop :=
  (x + 5)^2 + (y - 12)^2 = (16 - r)^2

/-- The main theorem -/
theorem smallest_positive_m_squared (m : ℝ) : 
  (∀ a : ℝ, a > 0 → (∃ x y r : ℝ, line a x y ∧ externally_tangent_w₂ x y r ∧ internally_tangent_w₁ x y r) → m ≤ a) ∧
  (∃ x y r : ℝ, line m x y ∧ externally_tangent_w₂ x y r ∧ internally_tangent_w₁ x y r) →
  m^2 = 69/100 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_m_squared_l3081_308111


namespace NUMINAMATH_CALUDE_algebraic_expression_evaluation_l3081_308177

theorem algebraic_expression_evaluation :
  let x : ℝ := 2 - Real.sqrt 3
  (7 + 4 * Real.sqrt 3) * x^2 - (2 + Real.sqrt 3) * x + Real.sqrt 3 = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_evaluation_l3081_308177


namespace NUMINAMATH_CALUDE_quadratic_equation_constant_term_l3081_308154

theorem quadratic_equation_constant_term (m : ℝ) : 
  (∀ x, (m - 2) * x^2 + 3 * x + m^2 - 4 = 0) → 
  m^2 - 4 = 0 → 
  m - 2 ≠ 0 → 
  m = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_constant_term_l3081_308154


namespace NUMINAMATH_CALUDE_least_repeating_digits_seven_thirteenths_l3081_308196

/-- The least number of digits in a repeating block of 7/13 -/
def leastRepeatingDigits : ℕ := 6

/-- 7/13 is a repeating decimal -/
axiom seven_thirteenths_repeating : ∃ (n : ℕ) (k : ℕ+), (7 : ℚ) / 13 = ↑n / (10 ^ k.val - 1)

theorem least_repeating_digits_seven_thirteenths :
  leastRepeatingDigits = 6 ∧
  ∀ m : ℕ, m < leastRepeatingDigits → ¬∃ (n : ℕ) (k : ℕ+), (7 : ℚ) / 13 = ↑n / (10 ^ m - 1) :=
sorry

end NUMINAMATH_CALUDE_least_repeating_digits_seven_thirteenths_l3081_308196


namespace NUMINAMATH_CALUDE_l_shape_area_is_58_l3081_308179

/-- The area of an "L" shaped figure formed by removing a smaller rectangle from a larger rectangle -/
def l_shape_area (large_width large_height small_width small_height : ℕ) : ℕ :=
  large_width * large_height - small_width * small_height

/-- Theorem: The area of the "L" shaped figure is 58 square units -/
theorem l_shape_area_is_58 :
  l_shape_area 10 7 4 3 = 58 := by
  sorry

#eval l_shape_area 10 7 4 3

end NUMINAMATH_CALUDE_l_shape_area_is_58_l3081_308179


namespace NUMINAMATH_CALUDE_james_vote_percentage_l3081_308193

theorem james_vote_percentage (total_votes : ℕ) (john_votes : ℕ) (third_candidate_extra : ℕ) :
  total_votes = 1150 →
  john_votes = 150 →
  third_candidate_extra = 150 →
  let third_candidate_votes := john_votes + third_candidate_extra
  let remaining_votes := total_votes - john_votes
  let james_votes := total_votes - (john_votes + third_candidate_votes)
  james_votes / remaining_votes = 7 / 10 := by
  sorry

#check james_vote_percentage

end NUMINAMATH_CALUDE_james_vote_percentage_l3081_308193


namespace NUMINAMATH_CALUDE_oranges_left_l3081_308171

/-- The number of oranges originally in the basket -/
def original_oranges : ℕ := 8

/-- The number of oranges taken from the basket -/
def oranges_taken : ℕ := 5

/-- Theorem: The number of oranges left in the basket is 3 -/
theorem oranges_left : original_oranges - oranges_taken = 3 := by
  sorry

end NUMINAMATH_CALUDE_oranges_left_l3081_308171


namespace NUMINAMATH_CALUDE_family_ages_theorem_l3081_308117

/-- Represents the ages and birth times of a father and his two children -/
structure FamilyAges where
  fatherCurrentAge : ℝ
  sonAgeFiveYearsAgo : ℝ
  daughterAgeFiveYearsAgo : ℝ
  sonCurrentAge : ℝ
  daughterCurrentAge : ℝ
  fatherAgeAtSonBirth : ℝ
  fatherAgeAtDaughterBirth : ℝ

/-- Theorem about the ages in a family based on given conditions -/
theorem family_ages_theorem (f : FamilyAges)
    (h1 : f.fatherCurrentAge = 38)
    (h2 : f.sonAgeFiveYearsAgo = 7)
    (h3 : f.daughterAgeFiveYearsAgo = f.sonAgeFiveYearsAgo / 2)
    (h4 : f.sonCurrentAge = f.sonAgeFiveYearsAgo + 5)
    (h5 : f.daughterCurrentAge = f.daughterAgeFiveYearsAgo + 5)
    (h6 : f.fatherAgeAtSonBirth = f.fatherCurrentAge - f.sonCurrentAge)
    (h7 : f.fatherAgeAtDaughterBirth = f.fatherCurrentAge - f.daughterCurrentAge) :
    f.sonCurrentAge = 12 ∧
    f.daughterCurrentAge = 8.5 ∧
    f.fatherAgeAtSonBirth = 26 ∧
    f.fatherAgeAtDaughterBirth = 29.5 := by
  sorry


end NUMINAMATH_CALUDE_family_ages_theorem_l3081_308117


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3081_308104

theorem necessary_but_not_sufficient_condition (x : ℝ) : 
  (∀ y : ℝ, 2*y > 2 → y > -1) ∧ 
  (∃ z : ℝ, z > -1 ∧ ¬(2*z > 2)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3081_308104
