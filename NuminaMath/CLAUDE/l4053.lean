import Mathlib

namespace NUMINAMATH_CALUDE_four_term_expression_l4053_405370

theorem four_term_expression (x : ℝ) : 
  ∃ (a b c d : ℝ), (x^3 - 2)^2 + (x^2 + 2*x)^2 = a*x^6 + b*x^4 + c*x^2 + d ∧ 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_four_term_expression_l4053_405370


namespace NUMINAMATH_CALUDE_girls_in_class_l4053_405383

theorem girls_in_class (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 35 →
  boys + girls = total →
  girls = (2 / 5 : ℚ) * boys →
  girls = 10 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l4053_405383


namespace NUMINAMATH_CALUDE_circle_center_sum_l4053_405360

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x - 14*y + 55 = 0

/-- The center of the circle -/
def center (h k : ℝ) : Prop :=
  ∀ x y, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 55)

theorem circle_center_sum : ∃ h k, center h k ∧ h + k = 11 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l4053_405360


namespace NUMINAMATH_CALUDE_E_equals_F_l4053_405353

def E : Set ℝ := {x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3)}

def F : Set ℝ := {x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6)}

theorem E_equals_F : E = F := by sorry

end NUMINAMATH_CALUDE_E_equals_F_l4053_405353


namespace NUMINAMATH_CALUDE_max_value_of_product_sum_l4053_405321

theorem max_value_of_product_sum (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a + b + c + d = 200 → 
  a * b + b * c + c * d ≤ 2500 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_product_sum_l4053_405321


namespace NUMINAMATH_CALUDE_divisor_count_power_of_two_l4053_405333

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ+) : ℕ := sorry

/-- Number of divisors function -/
def num_of_divisors (n : ℕ+) : ℕ := sorry

/-- A natural number is a power of two -/
def is_power_of_two (n : ℕ) : Prop := sorry

theorem divisor_count_power_of_two (n : ℕ+) :
  is_power_of_two (sum_of_divisors n) → is_power_of_two (num_of_divisors n) := by
  sorry

end NUMINAMATH_CALUDE_divisor_count_power_of_two_l4053_405333


namespace NUMINAMATH_CALUDE_sum_of_large_prime_factors_2310_l4053_405388

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem sum_of_large_prime_factors_2310 : 
  ∃ (factors : List ℕ), 
    (∀ f ∈ factors, is_prime f ∧ f > 5) ∧ 
    (factors.prod = 2310 / (2 * 3 * 5)) ∧
    (factors.sum = 18) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_large_prime_factors_2310_l4053_405388


namespace NUMINAMATH_CALUDE_sequence_sum_correct_l4053_405377

/-- Given a sequence where:
    - The first term is 2
    - The sum of the first two terms is 7
    - The sum of the first three terms is 18
    This function represents the sum of the first n terms of the sequence. -/
def sequenceSum (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1)) / 4 - (3 * n * (n + 1)) / 4 + 2 * n

/-- Theorem stating that the sequenceSum function correctly represents
    the sum of the first n terms of the sequence with the given properties. -/
theorem sequence_sum_correct (n : ℕ) :
  sequenceSum n = (n * (n + 1) * (2 * n + 1)) / 4 - (3 * n * (n + 1)) / 4 + 2 * n ∧
  sequenceSum 1 = 2 ∧
  sequenceSum 2 = 7 ∧
  sequenceSum 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_correct_l4053_405377


namespace NUMINAMATH_CALUDE_stations_visited_l4053_405375

theorem stations_visited (total_nails : ℕ) (nails_per_station : ℕ) (h1 : total_nails = 140) (h2 : nails_per_station = 7) :
  total_nails / nails_per_station = 20 := by
  sorry

end NUMINAMATH_CALUDE_stations_visited_l4053_405375


namespace NUMINAMATH_CALUDE_triangle_side_length_l4053_405369

theorem triangle_side_length (a b c : ℝ) : 
  a = 1 → b = 3 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  c ∈ ({3, 4, 5, 6} : Set ℝ) →
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4053_405369


namespace NUMINAMATH_CALUDE_rahul_savings_l4053_405303

/-- Rahul's savings problem -/
theorem rahul_savings (nsc ppf : ℕ) : 
  (1/3 : ℚ) * nsc = (1/2 : ℚ) * ppf →
  nsc + ppf = 180000 →
  ppf = 72000 := by
sorry

end NUMINAMATH_CALUDE_rahul_savings_l4053_405303


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l4053_405392

/-- A color type representing red, white, and blue -/
inductive Color
  | Red
  | White
  | Blue

/-- A point in the grid -/
structure Point where
  x : Nat
  y : Nat

/-- A coloring function that assigns a color to each point in the grid -/
def Coloring := Point → Color

/-- A rectangle in the grid -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- Theorem stating the existence of a monochromatic rectangle in a 12x12 grid -/
theorem monochromatic_rectangle_exists (coloring : Coloring) :
  ∃ (rect : Rectangle) (c : Color),
    rect.topLeft.x ≤ 12 ∧ rect.topLeft.y ≤ 12 ∧
    rect.bottomRight.x ≤ 12 ∧ rect.bottomRight.y ≤ 12 ∧
    coloring rect.topLeft = c ∧
    coloring { x := rect.topLeft.x, y := rect.bottomRight.y } = c ∧
    coloring { x := rect.bottomRight.x, y := rect.topLeft.y } = c ∧
    coloring rect.bottomRight = c := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l4053_405392


namespace NUMINAMATH_CALUDE_lieutenant_age_l4053_405374

theorem lieutenant_age : ∃ (n : ℕ), ∃ (x : ℕ), 
  -- Initial arrangement: n rows with n+5 soldiers each
  -- New arrangement: x rows (lieutenant's age) with n+9 soldiers each
  -- Total number of soldiers remains the same
  n * (n + 5) = x * (n + 9) ∧
  -- x represents a reasonable age for a lieutenant
  x > 18 ∧ x < 65 ∧
  -- The solution
  x = 24 := by
sorry

end NUMINAMATH_CALUDE_lieutenant_age_l4053_405374


namespace NUMINAMATH_CALUDE_condition_for_cubic_equation_l4053_405338

theorem condition_for_cubic_equation (a b : ℝ) (h : a * b ≠ 0) :
  (a - b = 1) ↔ (a^3 - b^3 - a*b - a^2 - b^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_for_cubic_equation_l4053_405338


namespace NUMINAMATH_CALUDE_makeup_exam_average_score_l4053_405356

/-- Represents the average score of students who took the exam on the make-up date -/
def makeup_avg : ℝ := 90

theorem makeup_exam_average_score 
  (total_students : ℕ) 
  (assigned_day_percent : ℝ) 
  (assigned_day_avg : ℝ) 
  (total_avg : ℝ) 
  (h1 : total_students = 100)
  (h2 : assigned_day_percent = 70)
  (h3 : assigned_day_avg = 60)
  (h4 : total_avg = 69) :
  makeup_avg = 90 := by
  sorry

#check makeup_exam_average_score

end NUMINAMATH_CALUDE_makeup_exam_average_score_l4053_405356


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4053_405311

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 < 0}
def B : Set ℝ := {x | x < 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4053_405311


namespace NUMINAMATH_CALUDE_athletes_game_count_l4053_405355

theorem athletes_game_count (malik_yards josiah_yards darnell_yards total_yards : ℕ) 
  (h1 : malik_yards = 18)
  (h2 : josiah_yards = 22)
  (h3 : darnell_yards = 11)
  (h4 : total_yards = 204) :
  ∃ n : ℕ, n * (malik_yards + josiah_yards + darnell_yards) = total_yards ∧ n = 4 := by
sorry

end NUMINAMATH_CALUDE_athletes_game_count_l4053_405355


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l4053_405304

/-- The coefficient of x^3 in the expansion of (1+2x)^6 is 160 -/
theorem coefficient_x_cubed_in_binomial_expansion : 
  (Finset.range 7).sum (fun k => (Nat.choose 6 k) * 2^k * if k = 3 then 1 else 0) = 160 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l4053_405304


namespace NUMINAMATH_CALUDE_line_y_axis_intersection_l4053_405395

def line (x y : ℝ) : Prop := y = 2 * x + 1

def y_axis (x : ℝ) : Prop := x = 0

def intersection_point : Set (ℝ × ℝ) := {(0, 1)}

theorem line_y_axis_intersection :
  {p : ℝ × ℝ | line p.1 p.2 ∧ y_axis p.1} = intersection_point := by
sorry

end NUMINAMATH_CALUDE_line_y_axis_intersection_l4053_405395


namespace NUMINAMATH_CALUDE_abc_inequality_l4053_405354

theorem abc_inequality (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 ∧
  ((a - b) * (b - c) * (a - c) = 2 ↔ 
    ((a = 2 ∧ b = 1 ∧ c = 0) ∨ (a = 1 ∧ b = 0 ∧ c = 2) ∨ (a = 0 ∧ b = 2 ∧ c = 1))) :=
by sorry


end NUMINAMATH_CALUDE_abc_inequality_l4053_405354


namespace NUMINAMATH_CALUDE_choir_members_count_l4053_405339

theorem choir_members_count :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, 100 ≤ n ∧ n ≤ 200) ∧
    (∀ n ∈ s, (n + 3) % 7 = 0 ∧ (n + 5) % 8 = 0) ∧
    s.card = 2 ∧
    123 ∈ s ∧ 179 ∈ s :=
by sorry

end NUMINAMATH_CALUDE_choir_members_count_l4053_405339


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l4053_405306

theorem trigonometric_equation_solution (x : ℝ) : 
  2 * (Real.sin x ^ 6 + Real.cos x ^ 6) - 3 * (Real.sin x ^ 4 + Real.cos x ^ 4) = Real.cos (2 * x) →
  ∃ k : ℤ, x = π / 2 * (2 * ↑k + 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l4053_405306


namespace NUMINAMATH_CALUDE_minimal_vertices_2007_gon_l4053_405320

/-- Given a regular polygon with n sides, returns the minimal number k such that
    among every k vertices of the polygon, there always exists 4 vertices forming
    a convex quadrilateral with 3 sides being sides of the polygon. -/
def minimalVerticesForQuadrilateral (n : ℕ) : ℕ :=
  ⌈(3 * n : ℚ) / 4⌉₊

theorem minimal_vertices_2007_gon :
  minimalVerticesForQuadrilateral 2007 = 1506 := by
  sorry

#eval minimalVerticesForQuadrilateral 2007

end NUMINAMATH_CALUDE_minimal_vertices_2007_gon_l4053_405320


namespace NUMINAMATH_CALUDE_log_sum_equality_l4053_405331

theorem log_sum_equality : Real.log 4 / Real.log 10 + 2 * (Real.log 5 / Real.log 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l4053_405331


namespace NUMINAMATH_CALUDE_sum_a_n_1_to_1499_l4053_405326

def a_n (n : ℕ) : ℕ :=
  if n < 1500 then
    if n % 10 = 0 ∧ n % 15 = 0 then 15
    else if n % 15 = 0 ∧ n % 12 = 0 then 10
    else if n % 12 = 0 ∧ n % 10 = 0 then 12
    else 0
  else 0

theorem sum_a_n_1_to_1499 :
  (Finset.range 1499).sum a_n = 1263 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_n_1_to_1499_l4053_405326


namespace NUMINAMATH_CALUDE_inequality_proof_l4053_405393

theorem inequality_proof (x y z t : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : t ≥ 0)
  (h5 : x * y * z = 2) (h6 : y + z + t = 2 * Real.sqrt 2) :
  2 * x^2 + y^2 + z^2 + t^2 ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4053_405393


namespace NUMINAMATH_CALUDE_closed_set_A_l4053_405361

def f (x : ℚ) : ℚ := (1 + x) / (1 - x)

def A : Set ℚ := {2, -3, -1/2, 1/3}

theorem closed_set_A :
  (2 ∈ A) ∧
  (∀ x ∈ A, f x ∈ A) ∧
  (∀ S : Set ℚ, 2 ∈ S → (∀ x ∈ S, f x ∈ S) → A ⊆ S) :=
sorry

end NUMINAMATH_CALUDE_closed_set_A_l4053_405361


namespace NUMINAMATH_CALUDE_min_r_for_perfect_square_l4053_405381

theorem min_r_for_perfect_square : 
  ∃ (r : ℕ), r > 0 ∧ 
  (∃ (n : ℕ), 4^3 + 4^r + 4^4 = n^2) ∧
  (∀ (s : ℕ), s > 0 ∧ s < r → ¬∃ (m : ℕ), 4^3 + 4^s + 4^4 = m^2) ∧
  r = 4 := by
sorry

end NUMINAMATH_CALUDE_min_r_for_perfect_square_l4053_405381


namespace NUMINAMATH_CALUDE_profit_maximized_at_150_l4053_405366

/-- The profit function for a company -/
def profit_function (a : ℝ) (x : ℝ) : ℝ := -a * x^2 + 7500 * x

/-- The derivative of the profit function -/
def profit_derivative (a : ℝ) (x : ℝ) : ℝ := -2 * a * x + 7500

theorem profit_maximized_at_150 (a : ℝ) :
  (profit_derivative a 150 = 0) → (a = 25) :=
by sorry

#check profit_maximized_at_150

end NUMINAMATH_CALUDE_profit_maximized_at_150_l4053_405366


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_6241_l4053_405371

theorem largest_prime_factor_of_6241 : ∃ (p : ℕ), p.Prime ∧ p ∣ 6241 ∧ ∀ (q : ℕ), q.Prime → q ∣ 6241 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_6241_l4053_405371


namespace NUMINAMATH_CALUDE_tank_capacity_is_90_l4053_405349

/-- Represents a gasoline tank with a certain capacity -/
structure GasolineTank where
  capacity : ℚ
  initialFraction : ℚ
  finalFraction : ℚ
  usedAmount : ℚ

/-- Theorem stating that the tank capacity is 90 gallons given the conditions -/
theorem tank_capacity_is_90 (tank : GasolineTank)
  (h1 : tank.initialFraction = 5/6)
  (h2 : tank.finalFraction = 2/3)
  (h3 : tank.usedAmount = 15)
  (h4 : tank.initialFraction * tank.capacity - tank.finalFraction * tank.capacity = tank.usedAmount) :
  tank.capacity = 90 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_is_90_l4053_405349


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l4053_405324

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 14) (h2 : x * y = 45) :
  1 / x + 1 / y = 14 / 45 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l4053_405324


namespace NUMINAMATH_CALUDE_ball_max_height_l4053_405394

/-- The height function of a ball traveling along a parabolic path -/
def h (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 20

/-- The maximum height reached by the ball -/
def max_height : ℝ := 145

theorem ball_max_height :
  ∃ t : ℝ, h t = max_height ∧ ∀ s : ℝ, h s ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l4053_405394


namespace NUMINAMATH_CALUDE_maximize_negative_products_l4053_405336

theorem maximize_negative_products (n : ℕ) (h : n > 0) :
  let f : ℕ → ℕ := λ k => k * (n - k)
  let max_k : ℕ := if n % 2 = 0 then n / 2 else (n - 1) / 2
  ∀ k, k ≤ n → f k ≤ f max_k ∧
    (n % 2 ≠ 0 → f k ≤ f ((n + 1) / 2)) :=
by sorry


end NUMINAMATH_CALUDE_maximize_negative_products_l4053_405336


namespace NUMINAMATH_CALUDE_diagonal_not_parallel_to_sides_l4053_405348

theorem diagonal_not_parallel_to_sides (n : ℕ) (h : n > 0) :
  n * (2 * n - 3) > 2 * n * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_not_parallel_to_sides_l4053_405348


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_special_case_l4053_405341

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a circle with radius r -/
structure Circle where
  r : ℝ
  h_positive : 0 < r

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Eccentricity of ellipse under specific conditions -/
theorem ellipse_eccentricity_special_case 
  (E : Ellipse) 
  (O : Circle)
  (B : Point)
  (A : Point)
  (h_circle : O.r = E.a)
  (h_B_on_y : B.x = 0 ∧ B.y = E.a)
  (h_B_on_ellipse : B.x^2 / E.a^2 + B.y^2 / E.b^2 = 1)
  (h_B_on_circle : B.x^2 + B.y^2 = O.r^2)
  (h_A_on_circle : A.x^2 + A.y^2 = O.r^2)
  (h_tangent : ∃ (m : ℝ), (A.y - B.y) = m * (A.x - B.x) ∧ 
               ∀ (x y : ℝ), y = m * (x - B.x) + B.y → x^2 / E.a^2 + y^2 / E.b^2 ≥ 1)
  (h_angle : Real.cos (60 * π / 180) = (A.x * B.x + A.y * B.y) / (O.r^2)) :
  let e := Real.sqrt (E.a^2 - E.b^2) / E.a
  e = Real.sqrt 3 / 3 := by
    sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_special_case_l4053_405341


namespace NUMINAMATH_CALUDE_magnitude_of_vector_sum_l4053_405301

def a : Fin 2 → ℝ := ![1, 0]
def b : Fin 2 → ℝ := ![2, 1]

theorem magnitude_of_vector_sum :
  ‖a + 3 • b‖ = Real.sqrt 58 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_sum_l4053_405301


namespace NUMINAMATH_CALUDE_complexity_theorem_l4053_405384

/-- Complexity of an integer is the number of prime factors in its prime decomposition -/
def complexity (n : ℕ) : ℕ := sorry

/-- n is a power of two -/
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem complexity_theorem (n : ℕ) (h : n > 1) :
  (∀ m : ℕ, n < m ∧ m ≤ 2*n → complexity m ≤ complexity n) ↔ is_power_of_two n ∧
  ¬∃ n : ℕ, ∀ m : ℕ, n < m ∧ m ≤ 2*n → complexity m < complexity n :=
sorry

end NUMINAMATH_CALUDE_complexity_theorem_l4053_405384


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l4053_405386

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (remaining_age_diff : ℝ),
    team_size = 15 →
    captain_age = 32 →
    wicket_keeper_age_diff = 5 →
    remaining_age_diff = 2 →
    ∃ (team_avg_age : ℝ),
      team_avg_age * team_size =
        captain_age + (captain_age + wicket_keeper_age_diff) +
        (team_size - 2) * (team_avg_age - remaining_age_diff) ∧
      team_avg_age = 21.5 := by
sorry


end NUMINAMATH_CALUDE_cricket_team_average_age_l4053_405386


namespace NUMINAMATH_CALUDE_base7_to_base10_54231_l4053_405398

/-- Converts a base 7 number to base 10 -/
def base7_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- The base 7 representation of the number -/
def base7_num : List Nat := [1, 3, 2, 4, 5]

theorem base7_to_base10_54231 :
  base7_to_base10 base7_num = 13497 := by sorry

end NUMINAMATH_CALUDE_base7_to_base10_54231_l4053_405398


namespace NUMINAMATH_CALUDE_divisibility_condition_pairs_l4053_405327

theorem divisibility_condition_pairs :
  ∀ m n : ℕ+,
  (∃ k : ℤ, (m : ℤ) + (n : ℤ)^2 = k * ((m : ℤ)^2 - (n : ℤ))) →
  (∃ l : ℤ, (n : ℤ) + (m : ℤ)^2 = l * ((n : ℤ)^2 - (m : ℤ))) →
  ((m = 2 ∧ n = 2) ∨ (m = 3 ∧ n = 3) ∨ (m = 1 ∧ n = 2) ∨
   (m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 3)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_pairs_l4053_405327


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l4053_405389

/-- The number of matches in a round-robin tournament -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A 10-player round-robin tournament has 45 matches -/
theorem ten_player_tournament_matches :
  roundRobinMatches 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_tournament_matches_l4053_405389


namespace NUMINAMATH_CALUDE_building_height_is_270_l4053_405328

/-- Calculates the height of a building with specified story heights -/
def buildingHeight (totalStories : ℕ) (firstHalfHeight : ℕ) (heightIncrease : ℕ) : ℕ :=
  let firstHalfStories := totalStories / 2
  let secondHalfStories := totalStories - firstHalfStories
  let firstHalfTotalHeight := firstHalfStories * firstHalfHeight
  let secondHalfHeight := firstHalfHeight + heightIncrease
  let secondHalfTotalHeight := secondHalfStories * secondHalfHeight
  firstHalfTotalHeight + secondHalfTotalHeight

/-- Theorem: The height of a 20-story building with specified story heights is 270 feet -/
theorem building_height_is_270 :
  buildingHeight 20 12 3 = 270 :=
by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_building_height_is_270_l4053_405328


namespace NUMINAMATH_CALUDE_fraction_zero_implies_a_equals_two_l4053_405357

theorem fraction_zero_implies_a_equals_two (a : ℝ) 
  (h1 : (a^2 - 4) / (a + 2) = 0) 
  (h2 : a + 2 ≠ 0) : 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_a_equals_two_l4053_405357


namespace NUMINAMATH_CALUDE_highway_mileage_l4053_405330

/-- Proves that the highway mileage is 37 mpg given the problem conditions -/
theorem highway_mileage (city_mpg : ℝ) (total_miles : ℝ) (total_gallons : ℝ) (highway_city_diff : ℝ) :
  city_mpg = 30 →
  total_miles = 365 →
  total_gallons = 11 →
  highway_city_diff = 5 →
  ∃ (city_miles highway_miles : ℝ),
    city_miles + highway_miles = total_miles ∧
    highway_miles = city_miles + highway_city_diff ∧
    city_miles / city_mpg + highway_miles / 37 = total_gallons :=
by sorry

end NUMINAMATH_CALUDE_highway_mileage_l4053_405330


namespace NUMINAMATH_CALUDE_k_value_l4053_405300

theorem k_value (a b c k : ℝ) 
  (h1 : 2 * a / (b + c) = k) 
  (h2 : 2 * b / (a + c) = k) 
  (h3 : 2 * c / (a + b) = k) : 
  k = 1 ∨ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l4053_405300


namespace NUMINAMATH_CALUDE_sum_cube_value_l4053_405307

theorem sum_cube_value (x y : ℝ) (h1 : x * (x + y) = 49) (h2 : y * (x + y) = 63) :
  (x + y)^3 = 448 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_sum_cube_value_l4053_405307


namespace NUMINAMATH_CALUDE_boxes_per_case_l4053_405372

theorem boxes_per_case (total_boxes : ℕ) (total_cases : ℕ) (boxes_per_case : ℕ) : 
  total_boxes = 20 → total_cases = 5 → total_boxes = total_cases * boxes_per_case → boxes_per_case = 4 := by
  sorry

end NUMINAMATH_CALUDE_boxes_per_case_l4053_405372


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l4053_405387

theorem smallest_n_congruence : 
  ∃! (n : ℕ), n > 0 ∧ (3 * n) % 24 = 1410 % 24 ∧ ∀ m : ℕ, m > 0 → (3 * m) % 24 = 1410 % 24 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l4053_405387


namespace NUMINAMATH_CALUDE_total_onion_weight_is_10_9_l4053_405399

/-- The total weight of onions grown by Sara, Sally, Fred, and Jack -/
def total_onion_weight : ℝ :=
  let sara_onions := 4
  let sara_weight := 0.5
  let sally_onions := 5
  let sally_weight := 0.4
  let fred_onions := 9
  let fred_weight := 0.3
  let jack_onions := 7
  let jack_weight := 0.6
  sara_onions * sara_weight +
  sally_onions * sally_weight +
  fred_onions * fred_weight +
  jack_onions * jack_weight

/-- Proof that the total weight of onions is 10.9 pounds -/
theorem total_onion_weight_is_10_9 :
  total_onion_weight = 10.9 := by sorry

end NUMINAMATH_CALUDE_total_onion_weight_is_10_9_l4053_405399


namespace NUMINAMATH_CALUDE_integer_sum_problem_l4053_405305

theorem integer_sum_problem (x y : ℤ) : 
  x > 0 → y > 0 → x - y = 8 → x * y = 120 → x + y = 4 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l4053_405305


namespace NUMINAMATH_CALUDE_number_multiplied_by_48_l4053_405373

theorem number_multiplied_by_48 : ∃ x : ℤ, x * 48 = 173 * 240 ∧ x = 865 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_48_l4053_405373


namespace NUMINAMATH_CALUDE_sequence_general_term_l4053_405337

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ+) : ℚ := 3 * n.val ^ 2 + 8 * n.val

/-- The general term of the sequence -/
def a (n : ℕ+) : ℚ := 6 * n.val + 5

/-- Theorem stating that the given general term formula is correct for the sequence -/
theorem sequence_general_term (n : ℕ+) : a n = S n - S (n - 1) := by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l4053_405337


namespace NUMINAMATH_CALUDE_minimum_room_size_for_table_l4053_405352

theorem minimum_room_size_for_table (table_length : ℝ) (table_width : ℝ) 
  (h1 : table_length = 12) (h2 : table_width = 9) : 
  ∃ (S : ℕ), S = 15 ∧ 
  (∀ (room_size : ℕ), (Real.sqrt (table_length^2 + table_width^2) ≤ room_size) ↔ (S ≤ room_size)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_room_size_for_table_l4053_405352


namespace NUMINAMATH_CALUDE_inequality_proof_l4053_405342

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a - b)^2 / (2 * (a + b)) ≤ Real.sqrt ((a^2 + b^2) / 2) - Real.sqrt (a * b) ∧
  Real.sqrt ((a^2 + b^2) / 2) - Real.sqrt (a * b) ≤ (a - b)^2 / (Real.sqrt 2 * (a + b)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4053_405342


namespace NUMINAMATH_CALUDE_krishan_money_l4053_405397

/-- Represents the money ratios and changes for Ram, Gopal, Shyam, and Krishan --/
structure MoneyProblem where
  ram_initial : ℚ
  gopal_initial : ℚ
  shyam_initial : ℚ
  krishan_ratio : ℚ
  ram_increase_percent : ℚ
  shyam_decrease_percent : ℚ
  ram_final : ℚ
  shyam_final : ℚ

/-- Theorem stating that given the conditions, Krishan's money is 3400 --/
theorem krishan_money (p : MoneyProblem)
  (h1 : p.ram_initial = 7)
  (h2 : p.gopal_initial = 17)
  (h3 : p.shyam_initial = 10)
  (h4 : p.krishan_ratio = 16)
  (h5 : p.ram_increase_percent = 18.5)
  (h6 : p.shyam_decrease_percent = 20)
  (h7 : p.ram_final = 699.8)
  (h8 : p.shyam_final = 800)
  (h9 : p.gopal_initial / p.ram_initial = 8 / p.krishan_ratio)
  (h10 : p.gopal_initial / p.shyam_initial = 8 / 9) :
  ∃ (x : ℚ), x * p.krishan_ratio = 3400 := by
  sorry


end NUMINAMATH_CALUDE_krishan_money_l4053_405397


namespace NUMINAMATH_CALUDE_syllogism_validity_l4053_405358

theorem syllogism_validity (a b c : Prop) : 
  ((b → c) ∧ (a → b)) → (a → c) := by sorry

end NUMINAMATH_CALUDE_syllogism_validity_l4053_405358


namespace NUMINAMATH_CALUDE_triangle_side_length_l4053_405365

/-- An equilateral triangle divided into three congruent trapezoids -/
structure TriangleDivision where
  /-- The side length of the equilateral triangle -/
  triangle_side : ℝ
  /-- The length of the shorter base of each trapezoid -/
  trapezoid_short_base : ℝ
  /-- The length of the longer base of each trapezoid -/
  trapezoid_long_base : ℝ
  /-- The length of the legs of each trapezoid -/
  trapezoid_leg : ℝ
  /-- The trapezoids are congruent -/
  congruent_trapezoids : trapezoid_long_base = 2 * trapezoid_short_base
  /-- The triangle is divided into three trapezoids -/
  triangle_composition : triangle_side = trapezoid_short_base + 2 * trapezoid_leg
  /-- The perimeter of each trapezoid is 10 + 5√3 -/
  trapezoid_perimeter : trapezoid_short_base + trapezoid_long_base + 2 * trapezoid_leg = 10 + 5 * Real.sqrt 3

/-- Theorem: The side length of the equilateral triangle is 6 + 3√3 -/
theorem triangle_side_length (td : TriangleDivision) : td.triangle_side = 6 + 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4053_405365


namespace NUMINAMATH_CALUDE_polygon_sides_l4053_405323

theorem polygon_sides (n : ℕ) (sum_interior_angles : ℝ) : sum_interior_angles = 1080 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l4053_405323


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l4053_405312

/-- Given a right triangle with legs a and b, and hypotenuse c, where a:b = 2:5,
    if a perpendicular from the right angle to the hypotenuse divides it into
    segments r (adjacent to a) and s (adjacent to b), then r/s = 4/25. -/
theorem right_triangle_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a ^ 2 + b ^ 2 = c ^ 2 →  -- Pythagorean theorem
  a / b = 2 / 5 →  -- given ratio of legs
  r * s = a * b →  -- geometric mean theorem
  r + s = c →  -- sum of segments equals hypotenuse
  r / s = 4 / 25 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l4053_405312


namespace NUMINAMATH_CALUDE_hyperbola_a_plus_h_l4053_405346

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- First asymptote equation: y = m₁x + c₁ -/
  m₁ : ℝ
  c₁ : ℝ
  /-- Second asymptote equation: y = m₂x + c₂ -/
  m₂ : ℝ
  c₂ : ℝ
  /-- Point that the hyperbola passes through -/
  p : ℝ × ℝ

/-- The standard form of a hyperbola: (y-k)^2/a^2 - (x-h)^2/b^2 = 1 -/
structure StandardForm where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  a_pos : a > 0
  b_pos : b > 0

/-- Theorem stating the value of a + h for the given hyperbola -/
theorem hyperbola_a_plus_h (hyp : Hyperbola) 
    (h : hyp.m₁ = 3 ∧ hyp.c₁ = 6 ∧ hyp.m₂ = -3 ∧ hyp.c₂ = 2 ∧ hyp.p = (1, 8)) :
    ∃ (sf : StandardForm), sf.a + sf.h = (Real.sqrt 119 - 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_a_plus_h_l4053_405346


namespace NUMINAMATH_CALUDE_abs_value_sum_l4053_405329

theorem abs_value_sum (a b c : ℚ) : 
  (abs a = 2) → 
  (abs b = 2) → 
  (abs c = 3) → 
  (b < 0) → 
  (0 < a) → 
  ((a + b + c = 3) ∨ (a + b + c = -3)) := by
sorry

end NUMINAMATH_CALUDE_abs_value_sum_l4053_405329


namespace NUMINAMATH_CALUDE_sum_of_composite_function_l4053_405309

def p (x : ℝ) : ℝ := |x| - 3

def q (x : ℝ) : ℝ := -x^2

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_of_composite_function :
  (x_values.map (fun x => q (p x))).sum = -29 := by sorry

end NUMINAMATH_CALUDE_sum_of_composite_function_l4053_405309


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l4053_405318

theorem quadratic_equation_two_distinct_roots (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 3*k*x₁ - 2 = 0 ∧ x₂^2 - 3*k*x₂ - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l4053_405318


namespace NUMINAMATH_CALUDE_set_difference_equals_open_interval_l4053_405344

/-- The set A of real numbers x such that |4x - 1| > 9 -/
def A : Set ℝ := {x | |4*x - 1| > 9}

/-- The set B of non-negative real numbers -/
def B : Set ℝ := {x | x ≥ 0}

/-- The open interval (5/2, +∞) -/
def openInterval : Set ℝ := {x | x > 5/2}

/-- Theorem stating that the set difference A - B is equal to the open interval (5/2, +∞) -/
theorem set_difference_equals_open_interval : A \ B = openInterval := by
  sorry

end NUMINAMATH_CALUDE_set_difference_equals_open_interval_l4053_405344


namespace NUMINAMATH_CALUDE_mina_pi_digits_l4053_405367

/-- The number of digits of pi memorized by each person -/
structure PiDigits where
  sam : ℕ
  carlos : ℕ
  mina : ℕ

/-- The conditions of the problem -/
def problem_conditions (d : PiDigits) : Prop :=
  d.sam = d.carlos + 6 ∧
  d.mina = 6 * d.carlos ∧
  d.sam = 10

/-- The theorem to prove -/
theorem mina_pi_digits (d : PiDigits) : 
  problem_conditions d → d.mina = 24 := by
  sorry

end NUMINAMATH_CALUDE_mina_pi_digits_l4053_405367


namespace NUMINAMATH_CALUDE_total_lobster_amount_l4053_405322

/-- The amount of lobster in pounds for each harbor -/
structure HarborLobster where
  hooperBay : ℝ
  harborA : ℝ
  harborB : ℝ
  harborC : ℝ
  harborD : ℝ

/-- The conditions for the lobster distribution problem -/
def LobsterDistribution (h : HarborLobster) : Prop :=
  h.harborA = 50 ∧
  h.harborB = 70.5 ∧
  h.harborC = (2/3) * h.harborB ∧
  h.harborD = h.harborA - 0.15 * h.harborA ∧
  h.hooperBay = 3 * (h.harborA + h.harborB + h.harborC + h.harborD)

/-- The theorem stating that the total amount of lobster is 840 pounds -/
theorem total_lobster_amount (h : HarborLobster) 
  (hDist : LobsterDistribution h) : 
  h.hooperBay + h.harborA + h.harborB + h.harborC + h.harborD = 840 := by
  sorry

end NUMINAMATH_CALUDE_total_lobster_amount_l4053_405322


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l4053_405316

-- Define the universal set U
def U : Finset Int := {-3, -2, -1, 0, 1, 2, 3}

-- Define set A
def A : Finset Int := {-1, 0, 1, 2}

-- Define set B
def B : Finset Int := {-3, 0, 2, 3}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l4053_405316


namespace NUMINAMATH_CALUDE_rd_funding_exceeds_2_million_l4053_405302

/-- R&D funding function -/
def rd_funding (x : ℕ) : ℝ := 1.3 * (1 + 0.12)^x

/-- Year when funding exceeds 2 million -/
def exceed_year : ℕ := 4

theorem rd_funding_exceeds_2_million : 
  rd_funding exceed_year > 2 ∧ 
  ∀ y : ℕ, y < exceed_year → rd_funding y ≤ 2 := by
  sorry

#eval exceed_year + 2015

end NUMINAMATH_CALUDE_rd_funding_exceeds_2_million_l4053_405302


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4053_405368

theorem inverse_variation_problem (z x : ℝ) (h : ∃ k : ℝ, ∀ x z, z * x^2 = k) :
  (2 * 3^2 = z * 3^2) → (8 * x^2 = z * 3^2) → x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4053_405368


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l4053_405364

/-- Given a > 0 and a ≠ 1, prove that the function f(x) = a^(x+2) - 2 
    always passes through the point (-2, -1) regardless of the value of a -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 2) - 2
  f (-2) = -1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l4053_405364


namespace NUMINAMATH_CALUDE_unique_sum_value_l4053_405315

theorem unique_sum_value (n m : ℤ) 
  (h1 : 3 * n - m ≤ 4)
  (h2 : n + m ≥ 27)
  (h3 : 3 * m - 2 * n ≤ 45) :
  2 * n + m = 36 := by
  sorry

end NUMINAMATH_CALUDE_unique_sum_value_l4053_405315


namespace NUMINAMATH_CALUDE_advertising_department_size_l4053_405319

theorem advertising_department_size 
  (total_employees : ℕ) 
  (sample_size : ℕ) 
  (ad_dept_sample : ℕ) 
  (h1 : total_employees = 1000)
  (h2 : sample_size = 80)
  (h3 : ad_dept_sample = 4) :
  (ad_dept_sample : ℚ) / sample_size * total_employees = 50 := by
  sorry

end NUMINAMATH_CALUDE_advertising_department_size_l4053_405319


namespace NUMINAMATH_CALUDE_square_and_arithmetic_computation_l4053_405351

theorem square_and_arithmetic_computation : 7^2 - (4 * 6) / 2 + 6^2 = 73 := by
  sorry

end NUMINAMATH_CALUDE_square_and_arithmetic_computation_l4053_405351


namespace NUMINAMATH_CALUDE_floor_equality_condition_l4053_405391

theorem floor_equality_condition (a b : ℝ) :
  (∀ n : ℕ+, a * ⌊b * n⌋ = b * ⌊a * n⌋) ↔ (a = b ∨ a = 0 ∨ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_condition_l4053_405391


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l4053_405325

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a 1 * q^n

theorem geometric_sequence_solution (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 * a 2 * a 3 = 27 →
  a 2 + a 4 = 30 →
  ((a 1 = 1 ∧ ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a 1 * q^n) ∧ q = 3) ∨
   (a 1 = -1 ∧ ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a 1 * q^n) ∧ q = -3)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l4053_405325


namespace NUMINAMATH_CALUDE_arithmetic_mean_4_16_l4053_405335

theorem arithmetic_mean_4_16 (x : ℝ) : x = (4 + 16) / 2 → x = 10 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_4_16_l4053_405335


namespace NUMINAMATH_CALUDE_dimes_count_l4053_405396

/-- Represents the number of coins of each type --/
structure CoinCounts where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Calculates the total value in cents for a given set of coin counts --/
def totalValue (coins : CoinCounts) : Nat :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- Theorem: Given the total amount and the number of other coins, the number of dimes is 3 --/
theorem dimes_count (coins : CoinCounts) :
  coins.quarters = 10 ∧ coins.nickels = 3 ∧ coins.pennies = 5 ∧ totalValue coins = 300 →
  coins.dimes = 3 := by
  sorry


end NUMINAMATH_CALUDE_dimes_count_l4053_405396


namespace NUMINAMATH_CALUDE_max_necklaces_proof_l4053_405308

def necklace_green_beads : ℕ := 9
def necklace_white_beads : ℕ := 6
def necklace_orange_beads : ℕ := 3
def available_beads : ℕ := 45

def max_necklaces : ℕ := 5

theorem max_necklaces_proof :
  min (available_beads / necklace_green_beads)
      (min (available_beads / necklace_white_beads)
           (available_beads / necklace_orange_beads)) = max_necklaces := by
  sorry

end NUMINAMATH_CALUDE_max_necklaces_proof_l4053_405308


namespace NUMINAMATH_CALUDE_largest_interesting_is_correct_l4053_405380

/-- An interesting number is a natural number where all digits, except for the first and last,
    are less than the arithmetic mean of their two neighboring digits. -/
def is_interesting (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i, 1 < i ∧ i < digits.length - 1 →
    digits[i]! < (digits[i-1]! + digits[i+1]!) / 2

/-- The largest interesting number -/
def largest_interesting : ℕ := 96433469

theorem largest_interesting_is_correct :
  is_interesting largest_interesting ∧
  ∀ m : ℕ, is_interesting m → m ≤ largest_interesting :=
sorry

end NUMINAMATH_CALUDE_largest_interesting_is_correct_l4053_405380


namespace NUMINAMATH_CALUDE_cube_root_always_real_l4053_405313

theorem cube_root_always_real : 
  ∀ x : ℝ, ∃ y : ℝ, y^3 = -(x + 3)^3 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_always_real_l4053_405313


namespace NUMINAMATH_CALUDE_distance_to_school_l4053_405359

/-- The distance to school given travel conditions -/
theorem distance_to_school (total_time : ℝ) (speed_to_school : ℝ) (speed_from_school : ℝ)
  (h1 : total_time = 1)
  (h2 : speed_to_school = 5)
  (h3 : speed_from_school = 30) :
  ∃ d : ℝ, d = 30 / 7 ∧ total_time = d / speed_to_school + d / speed_from_school :=
by sorry

end NUMINAMATH_CALUDE_distance_to_school_l4053_405359


namespace NUMINAMATH_CALUDE_equal_probabilities_l4053_405362

/-- Represents a box containing balls of different colors -/
structure Box where
  red : ℕ
  green : ℕ

/-- Represents the state of both boxes -/
structure BoxState where
  red_box : Box
  green_box : Box

/-- Initial state of the boxes -/
def initial_state : BoxState :=
  { red_box := { red := 100, green := 0 },
    green_box := { red := 0, green := 100 } }

/-- State after transferring 8 red balls from red box to green box -/
def after_first_transfer (state : BoxState) : BoxState :=
  { red_box := { red := state.red_box.red - 8, green := state.red_box.green },
    green_box := { red := state.green_box.red + 8, green := state.green_box.green } }

/-- State after transferring 8 balls from green box to red box -/
def after_second_transfer (state : BoxState) : BoxState :=
  { red_box := { red := state.red_box.red + 8, green := state.red_box.green },
    green_box := { red := state.green_box.red - 8, green := state.green_box.green } }

/-- Final state after all transfers -/
def final_state : BoxState :=
  after_second_transfer (after_first_transfer initial_state)

/-- Probability of drawing a specific color ball from a box -/
def prob_draw (box : Box) (color : String) : ℚ :=
  match color with
  | "red" => box.red / (box.red + box.green)
  | "green" => box.green / (box.red + box.green)
  | _ => 0

theorem equal_probabilities :
  prob_draw final_state.red_box "green" = prob_draw final_state.green_box "red" := by
  sorry

end NUMINAMATH_CALUDE_equal_probabilities_l4053_405362


namespace NUMINAMATH_CALUDE_milk_left_l4053_405382

theorem milk_left (initial_milk : ℚ) (given_milk : ℚ) (remaining_milk : ℚ) : 
  initial_milk = 5 → given_milk = 18 / 7 → remaining_milk = initial_milk - given_milk → 
  remaining_milk = 17 / 7 := by
  sorry

end NUMINAMATH_CALUDE_milk_left_l4053_405382


namespace NUMINAMATH_CALUDE_heating_pad_cost_per_use_l4053_405340

/-- Calculates the cost per use of a heating pad. -/
def cost_per_use (total_cost : ℚ) (uses_per_week : ℕ) (num_weeks : ℕ) : ℚ :=
  total_cost / (uses_per_week * num_weeks)

/-- Theorem stating that a $30 heating pad used 3 times a week for 2 weeks costs $5 per use. -/
theorem heating_pad_cost_per_use :
  cost_per_use 30 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_heating_pad_cost_per_use_l4053_405340


namespace NUMINAMATH_CALUDE_karcsi_travels_further_l4053_405332

def karcsi_speed : ℝ := 6
def joska_speed : ℝ := 4
def bus_speed : ℝ := 60

theorem karcsi_travels_further (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h : x / karcsi_speed + (x + y) / bus_speed = y / joska_speed) : x > y := by
  sorry

end NUMINAMATH_CALUDE_karcsi_travels_further_l4053_405332


namespace NUMINAMATH_CALUDE_production_equation_correct_l4053_405378

/-- Represents the production rates and total production of a master and apprentice -/
structure ProductionData where
  total_rate : ℝ  -- Combined production rate per hour
  master_total : ℝ  -- Total parts made by master
  apprentice_total : ℝ  -- Total parts made by apprentice

/-- Checks if the given equation correctly represents the production relationship -/
def is_correct_equation (data : ProductionData) (master_rate : ℝ) : Prop :=
  (data.master_total / master_rate = data.apprentice_total / (data.total_rate - master_rate))

/-- The main theorem stating that the given equation is correct for the provided data -/
theorem production_equation_correct (data : ProductionData) 
  (h1 : data.total_rate = 40)
  (h2 : data.master_total = 300)
  (h3 : data.apprentice_total = 100) :
  ∃ (x : ℝ), is_correct_equation data x :=
sorry

end NUMINAMATH_CALUDE_production_equation_correct_l4053_405378


namespace NUMINAMATH_CALUDE_range_of_K_l4053_405376

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 2| - 1
def g (x : ℝ) : ℝ := |3 - x| + 2

-- Define the theorem
theorem range_of_K (K : ℝ) : 
  (∀ x : ℝ, f x - g x ≤ K) → K ∈ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_K_l4053_405376


namespace NUMINAMATH_CALUDE_possible_values_of_a_l4053_405317

def A (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}
def B : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (A a ∪ B = B) ↔ (a = -1/2 ∨ a = 0 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l4053_405317


namespace NUMINAMATH_CALUDE_exists_same_color_transformation_l4053_405310

/-- Represents the color of a square on the chessboard -/
inductive Color
| Black
| White

/-- Represents a 16x16 chessboard -/
def Chessboard := Fin 16 → Fin 16 → Color

/-- Initial chessboard with alternating colors -/
def initialChessboard : Chessboard :=
  fun i j => if (i.val + j.val) % 2 = 0 then Color.Black else Color.White

/-- Apply operation A to the chessboard at position (i, j) -/
def applyOperationA (board : Chessboard) (i j : Fin 16) : Chessboard :=
  fun x y =>
    if x = i || y = j then
      match board x y with
      | Color.Black => Color.White
      | Color.White => Color.Black
    else
      board x y

/-- Check if all squares on the chessboard have the same color -/
def allSameColor (board : Chessboard) : Prop :=
  ∀ i j : Fin 16, board i j = board 0 0

/-- Theorem: There exists a sequence of operations A that transforms all squares to the same color -/
theorem exists_same_color_transformation :
  ∃ (operations : List (Fin 16 × Fin 16)),
    allSameColor (operations.foldl (fun b (i, j) => applyOperationA b i j) initialChessboard) :=
  sorry

end NUMINAMATH_CALUDE_exists_same_color_transformation_l4053_405310


namespace NUMINAMATH_CALUDE_point_symmetry_l4053_405350

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

theorem point_symmetry (a b : ℝ) :
  symmetric_wrt_origin (3, a - 2) (b, a) → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_l4053_405350


namespace NUMINAMATH_CALUDE_same_terminal_side_l4053_405343

theorem same_terminal_side (θ : ℝ) : 
  ∃ k : ℤ, θ + 360 * k = 330 → θ = -30 := by sorry

end NUMINAMATH_CALUDE_same_terminal_side_l4053_405343


namespace NUMINAMATH_CALUDE_expression_evaluation_l4053_405345

theorem expression_evaluation (x y z : ℤ) (hx : x = 25) (hy : y = 33) (hz : z = 7) :
  (x - (y - z)) - ((x - y) - z) = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4053_405345


namespace NUMINAMATH_CALUDE_maximize_product_l4053_405379

theorem maximize_product (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 28) :
  x^5 * y^3 ≤ 17.5^5 * 10.5^3 ∧
  (x^5 * y^3 = 17.5^5 * 10.5^3 ↔ x = 17.5 ∧ y = 10.5) :=
by sorry

end NUMINAMATH_CALUDE_maximize_product_l4053_405379


namespace NUMINAMATH_CALUDE_min_composite_with_small_factors_l4053_405385

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def has_prime_factorization (n : ℕ) (max_factor : ℕ) : Prop :=
  ∃ (factors : List ℕ), 
    factors.length ≥ 2 ∧
    (∀ p ∈ factors, is_prime p ∧ p ≤ max_factor) ∧
    factors.prod = n

theorem min_composite_with_small_factors :
  ∀ n : ℕ, 
    ¬ is_prime n →
    has_prime_factorization n 10 →
    n ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_composite_with_small_factors_l4053_405385


namespace NUMINAMATH_CALUDE_two_tangent_lines_l4053_405334

-- Define the function f(x) = x³ - x²
def f (x : ℝ) : ℝ := x^3 - x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x

-- Define a point of tangency
structure TangentPoint where
  x : ℝ
  y : ℝ
  slope : ℝ

-- Define a tangent line that passes through (1,0)
def isTangentLineThroughPoint (tp : TangentPoint) : Prop :=
  tp.y = f tp.x ∧ 
  tp.slope = f' tp.x ∧
  0 = tp.y + tp.slope * (1 - tp.x)

-- Theorem: There are exactly 2 tangent lines to f(x) that pass through (1,0)
theorem two_tangent_lines : 
  ∃! (s : Finset TangentPoint), 
    (∀ tp ∈ s, isTangentLineThroughPoint tp) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_tangent_lines_l4053_405334


namespace NUMINAMATH_CALUDE_fourth_tea_price_theorem_l4053_405347

/-- Calculates the price of the fourth tea variety given the prices of three varieties,
    their mixing ratios, and the final mixture price. -/
def fourth_tea_price (p1 p2 p3 mix_price : ℚ) : ℚ :=
  let r1 : ℚ := 2
  let r2 : ℚ := 3
  let r3 : ℚ := 4
  let r4 : ℚ := 5
  let total_ratio : ℚ := r1 + r2 + r3 + r4
  (mix_price * total_ratio - (p1 * r1 + p2 * r2 + p3 * r3)) / r4

/-- Theorem stating that given the prices of three tea varieties, their mixing ratios,
    and the final mixture price, the price of the fourth variety is 205.8. -/
theorem fourth_tea_price_theorem (p1 p2 p3 mix_price : ℚ) 
  (h1 : p1 = 126) (h2 : p2 = 135) (h3 : p3 = 156) (h4 : mix_price = 165) :
  fourth_tea_price p1 p2 p3 mix_price = 205.8 := by
  sorry

#eval fourth_tea_price 126 135 156 165

end NUMINAMATH_CALUDE_fourth_tea_price_theorem_l4053_405347


namespace NUMINAMATH_CALUDE_perimeter_difference_l4053_405314

/-- Calculates the perimeter of a rectangle --/
def rectanglePerimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Calculates the perimeter of a modified rectangle with a vertical shift --/
def modifiedRectanglePerimeter (length width shift : ℕ) : ℕ :=
  2 * length + 2 * width + 2 * shift

/-- The positive difference between the perimeter of a 6x1 rectangle with a vertical shift
    and the perimeter of a 4x1 rectangle is 6 units --/
theorem perimeter_difference : 
  modifiedRectanglePerimeter 6 1 1 - rectanglePerimeter 4 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_l4053_405314


namespace NUMINAMATH_CALUDE_corporation_total_employees_l4053_405390

/-- The total number of employees in a corporation -/
def total_employees (part_time full_time contractors interns consultants : ℕ) : ℕ :=
  part_time + full_time + contractors + interns + consultants

/-- Theorem: The corporation employs 66907 workers in total -/
theorem corporation_total_employees :
  total_employees 2047 63109 1500 333 918 = 66907 := by
  sorry

end NUMINAMATH_CALUDE_corporation_total_employees_l4053_405390


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_1_condition_l4053_405363

-- Define the condition for the square root to be meaningful
def is_meaningful (x : ℝ) : Prop := 2 * x - 1 ≥ 0

-- State the theorem
theorem sqrt_2x_minus_1_condition (x : ℝ) :
  is_meaningful x ↔ x ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_1_condition_l4053_405363
