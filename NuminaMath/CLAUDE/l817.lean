import Mathlib

namespace NUMINAMATH_CALUDE_temperature_difference_l817_81749

/-- The temperature difference between the highest and lowest temperatures is 12°C, 
    given that the highest temperature is 11°C and the lowest temperature is -1°C. -/
theorem temperature_difference (highest lowest : ℝ) 
  (h_highest : highest = 11) 
  (h_lowest : lowest = -1) : 
  highest - lowest = 12 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l817_81749


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_can_be_prime_l817_81713

theorem sum_of_four_consecutive_integers_can_be_prime : 
  ∃ n : ℤ, Prime (n + (n + 1) + (n + 2) + (n + 3)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_can_be_prime_l817_81713


namespace NUMINAMATH_CALUDE_ladder_rope_difference_l817_81775

/-- Proves that the ladder is 10 feet longer than the rope given the climbing scenario -/
theorem ladder_rope_difference (
  num_flights : ℕ) 
  (flight_height : ℝ) 
  (total_height : ℝ) 
  (h1 : num_flights = 3)
  (h2 : flight_height = 10)
  (h3 : total_height = 70) : 
  let stairs_height := num_flights * flight_height
  let rope_height := stairs_height / 2
  let ladder_height := total_height - (stairs_height + rope_height)
  ladder_height - rope_height = 10 := by
sorry

end NUMINAMATH_CALUDE_ladder_rope_difference_l817_81775


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l817_81766

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 2) :
  (1/a + 2/b) ≥ 9/2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 2 ∧ 1/a₀ + 2/b₀ = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l817_81766


namespace NUMINAMATH_CALUDE_no_three_naturals_with_prime_sums_l817_81712

theorem no_three_naturals_with_prime_sums :
  ¬ ∃ (a b c : ℕ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.Prime (a + b) ∧ 
    Nat.Prime (a + c) ∧ 
    Nat.Prime (b + c) :=
sorry

end NUMINAMATH_CALUDE_no_three_naturals_with_prime_sums_l817_81712


namespace NUMINAMATH_CALUDE_line_equation_of_parabola_points_l817_81781

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 3*y

-- Define the quadratic equation
def quadratic_equation (x p q : ℝ) : Prop := x^2 + p*x + q = 0

theorem line_equation_of_parabola_points (p q : ℝ) (h : p^2 - 4*q > 0) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    quadratic_equation x₁ p q ∧ quadratic_equation x₂ p q ∧
    x₁ ≠ x₂ ∧
    ∀ (x y : ℝ), (p*x + 3*y + q = 0) ↔ (y - y₁) = ((y₂ - y₁) / (x₂ - x₁)) * (x - x₁) :=
sorry

end NUMINAMATH_CALUDE_line_equation_of_parabola_points_l817_81781


namespace NUMINAMATH_CALUDE_river_speed_l817_81793

theorem river_speed (still_water_speed : ℝ) (total_time : ℝ) (total_distance : ℝ) : 
  still_water_speed = 8 →
  total_time = 1 →
  total_distance = 7.5 →
  ∃ (river_speed : ℝ),
    river_speed = 2 ∧
    (total_distance / 2) / (still_water_speed - river_speed) + 
    (total_distance / 2) / (still_water_speed + river_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_river_speed_l817_81793


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l817_81752

/-- Converts a binary number (represented as a list of bits) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits (true for 1, false for 0) -/
def binary_10110 : List Bool := [true, false, true, true, false]
def binary_1101 : List Bool := [true, true, false, true]
def binary_110 : List Bool := [true, true, false]
def binary_101 : List Bool := [true, false, true]
def binary_1010 : List Bool := [true, false, true, false]

/-- The main theorem to prove -/
theorem binary_arithmetic_equality :
  binary_to_decimal binary_10110 - binary_to_decimal binary_1101 +
  binary_to_decimal binary_110 - binary_to_decimal binary_101 =
  binary_to_decimal binary_1010 := by
  sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l817_81752


namespace NUMINAMATH_CALUDE_ball_count_theorem_l817_81731

theorem ball_count_theorem (B W : ℕ) (h1 : W = 3 * B) 
  (h2 : 5 * B + W = 2 * (B + W)) : 
  B + 5 * W = 4 * (B + W) := by
sorry

end NUMINAMATH_CALUDE_ball_count_theorem_l817_81731


namespace NUMINAMATH_CALUDE_millie_bracelets_l817_81744

/-- The number of bracelets Millie had initially -/
def initial_bracelets : ℕ := 9

/-- The number of bracelets Millie lost -/
def lost_bracelets : ℕ := 2

/-- The number of bracelets Millie has left -/
def remaining_bracelets : ℕ := initial_bracelets - lost_bracelets

theorem millie_bracelets : remaining_bracelets = 7 := by
  sorry

end NUMINAMATH_CALUDE_millie_bracelets_l817_81744


namespace NUMINAMATH_CALUDE_books_second_shop_correct_l817_81719

/-- The number of books bought from the second shop -/
def books_second_shop : ℕ := 20

/-- The number of books bought from the first shop -/
def books_first_shop : ℕ := 27

/-- The cost of books from the first shop in rupees -/
def cost_first_shop : ℕ := 581

/-- The cost of books from the second shop in rupees -/
def cost_second_shop : ℕ := 594

/-- The average price per book in rupees -/
def average_price : ℕ := 25

theorem books_second_shop_correct : 
  books_second_shop = 20 ∧
  books_first_shop = 27 ∧
  cost_first_shop = 581 ∧
  cost_second_shop = 594 ∧
  average_price = 25 →
  (cost_first_shop + cost_second_shop : ℚ) / (books_first_shop + books_second_shop) = average_price := by
  sorry

end NUMINAMATH_CALUDE_books_second_shop_correct_l817_81719


namespace NUMINAMATH_CALUDE_students_per_table_l817_81733

theorem students_per_table (total_tables : ℕ) (total_students : ℕ) 
  (h1 : total_tables = 34) (h2 : total_students = 204) : 
  total_students / total_tables = 6 := by
  sorry

end NUMINAMATH_CALUDE_students_per_table_l817_81733


namespace NUMINAMATH_CALUDE_parallelogram_count_parallelogram_count_proof_l817_81761

/-- Given a triangle ABC with each side divided into n equal parts and parallel lines drawn through
    the division points, the number of parallelograms formed is 3 * (n choose 2). -/
theorem parallelogram_count (n : ℕ) : ℕ :=
  3 * (n.choose 2)

#check parallelogram_count

/-- Proof of the parallelogram count theorem -/
theorem parallelogram_count_proof (n : ℕ) :
  parallelogram_count n = 3 * (n.choose 2) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_count_parallelogram_count_proof_l817_81761


namespace NUMINAMATH_CALUDE_average_of_w_and_x_l817_81770

theorem average_of_w_and_x (w x y : ℝ) 
  (h1 : 2 / w + 2 / x = 2 / y) 
  (h2 : w * x = y) : 
  (w + x) / 2 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_average_of_w_and_x_l817_81770


namespace NUMINAMATH_CALUDE_church_capacity_l817_81735

/-- Calculates the total number of people that can sit in a church when it's full -/
theorem church_capacity (rows : ℕ) (chairs_per_row : ℕ) (people_per_chair : ℕ) : 
  rows = 20 → chairs_per_row = 6 → people_per_chair = 5 → 
  rows * chairs_per_row * people_per_chair = 600 := by
  sorry

#check church_capacity

end NUMINAMATH_CALUDE_church_capacity_l817_81735


namespace NUMINAMATH_CALUDE_initial_production_rate_l817_81732

/-- Proves that the initial production rate is 15 cogs per hour given the problem conditions --/
theorem initial_production_rate : 
  ∀ (initial_rate : ℝ),
  (∃ (initial_time : ℝ),
    initial_rate * initial_time = 60 ∧  -- Initial order production
    initial_time + 1 = 120 / 24 ∧       -- Total time equation
    (60 + 60) / (initial_time + 1) = 24 -- Average output equation
  ) → initial_rate = 15 := by
  sorry


end NUMINAMATH_CALUDE_initial_production_rate_l817_81732


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l817_81710

/-- The cost of apples in dollars per 3 pounds -/
def apple_cost_per_3_pounds : ℚ := 3

/-- The weight of apples in pounds that we want to calculate the cost for -/
def apple_weight : ℚ := 18

/-- Theorem stating that the cost of 18 pounds of apples is 18 dollars -/
theorem apple_cost_calculation : 
  (apple_weight / 3) * apple_cost_per_3_pounds = 18 := by
  sorry


end NUMINAMATH_CALUDE_apple_cost_calculation_l817_81710


namespace NUMINAMATH_CALUDE_four_term_expression_l817_81787

theorem four_term_expression (x : ℝ) : 
  ∃ (a b c d : ℝ), (x^4 - 3)^2 + (x^3 + 3*x)^2 = a*x^8 + b*x^6 + c*x^2 + d ∧ 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_four_term_expression_l817_81787


namespace NUMINAMATH_CALUDE_f_properties_l817_81789

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

theorem f_properties :
  let T := Real.pi
  let φ := 7 * Real.pi / 12
  (∀ x, f (x + T) = f x) ∧
  (∀ y, T ≤ y → (∀ x, f (x + y) = f x) → y = T) ∧
  (Real.pi / 2 < φ ∧ φ < Real.pi) ∧
  (∀ x, f (x + φ) = f (-x + φ)) ∧
  (∀ ψ, Real.pi / 2 < ψ ∧ ψ < Real.pi → (∀ x, f (x + ψ) = f (-x + ψ)) → ψ = φ) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l817_81789


namespace NUMINAMATH_CALUDE_grandma_molly_statues_l817_81780

/-- The number of statues Grandma Molly created in the first year -/
def initial_statues : ℕ := sorry

/-- The total number of statues after four years -/
def total_statues : ℕ := 31

/-- The number of statues broken in the third year -/
def broken_statues : ℕ := 3

theorem grandma_molly_statues :
  initial_statues = 4 ∧
  (4 * initial_statues + 12 - broken_statues + 2 * broken_statues = total_statues) :=
sorry

end NUMINAMATH_CALUDE_grandma_molly_statues_l817_81780


namespace NUMINAMATH_CALUDE_chess_tournament_games_l817_81738

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 30 players, where each player plays every other player exactly once,
    the total number of games played is 435. --/
theorem chess_tournament_games :
  num_games 30 = 435 := by
  sorry

#eval num_games 30  -- This will evaluate to 435

end NUMINAMATH_CALUDE_chess_tournament_games_l817_81738


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_2083_l817_81768

theorem units_digit_of_7_to_2083 : 7^2083 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_2083_l817_81768


namespace NUMINAMATH_CALUDE_triangle_area_example_l817_81763

/-- The area of a triangle given its vertices -/
def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  let v := (a.1 - c.1, a.2 - c.2)
  let w := (b.1 - c.1, b.2 - c.2)
  0.5 * abs (v.1 * w.2 - v.2 * w.1)

/-- Theorem: The area of the triangle with vertices (3, -5), (-2, 0), and (5, -8) is 2.5 -/
theorem triangle_area_example : triangleArea (3, -5) (-2, 0) (5, -8) = 2.5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_example_l817_81763


namespace NUMINAMATH_CALUDE_average_salary_of_all_employees_l817_81743

/-- Calculates the average salary of all employees in an office -/
theorem average_salary_of_all_employees 
  (officer_salary : ℝ) 
  (non_officer_salary : ℝ) 
  (num_officers : ℕ) 
  (num_non_officers : ℕ) 
  (h1 : officer_salary = 420)
  (h2 : non_officer_salary = 110)
  (h3 : num_officers = 15)
  (h4 : num_non_officers = 450) :
  (officer_salary * num_officers + non_officer_salary * num_non_officers) / (num_officers + num_non_officers) = 120 :=
by
  sorry

#check average_salary_of_all_employees

end NUMINAMATH_CALUDE_average_salary_of_all_employees_l817_81743


namespace NUMINAMATH_CALUDE_ratio_of_segments_l817_81788

/-- Given points A, B, C, D, and E on a line in that order, prove the ratio of AC to BD -/
theorem ratio_of_segments (A B C D E : ℝ) : 
  A < B → B < C → C < D → D < E →  -- Points lie on a line in order
  B - A = 3 →                      -- AB = 3
  C - B = 7 →                      -- BC = 7
  E - D = 4 →                      -- DE = 4
  D - A = 17 →                     -- AD = 17
  (C - A) / (D - B) = 5 / 7 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l817_81788


namespace NUMINAMATH_CALUDE_value_of_x_l817_81790

theorem value_of_x : ∀ w y z x : ℤ,
  w = 90 →
  z = w + 15 →
  y = z - 3 →
  x = y + 7 →
  x = 109 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l817_81790


namespace NUMINAMATH_CALUDE_angle_cosine_equivalence_l817_81720

theorem angle_cosine_equivalence (A B : Real) (hA : 0 < A ∧ A < Real.pi) (hB : 0 < B ∧ B < Real.pi) :
  A > B ↔ Real.cos A < Real.cos B := by
  sorry

end NUMINAMATH_CALUDE_angle_cosine_equivalence_l817_81720


namespace NUMINAMATH_CALUDE_m_minus_n_equals_three_l817_81782

-- Define the sets M and N
def M (m : ℕ) : Set ℕ := {1, 2, 3, m}
def N (n : ℕ) : Set ℕ := {4, 7, n^4, n^2 + 3*n}

-- Define the function f
def f (x : ℕ) : ℕ := 3*x + 1

-- State the theorem
theorem m_minus_n_equals_three (m n : ℕ) : 
  (∃ y ∈ M m, ∃ z ∈ N n, f y = z) → m - n = 3 := by
  sorry

end NUMINAMATH_CALUDE_m_minus_n_equals_three_l817_81782


namespace NUMINAMATH_CALUDE_art_piece_value_increase_l817_81777

def original_price : ℝ := 4000
def future_price : ℝ := 3 * original_price

theorem art_piece_value_increase : future_price - original_price = 8000 := by
  sorry

end NUMINAMATH_CALUDE_art_piece_value_increase_l817_81777


namespace NUMINAMATH_CALUDE_isosceles_triangle_ef_length_l817_81716

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  /-- The length of two equal sides -/
  side_length : ℝ
  /-- The ratio of EG to GF -/
  eg_gf_ratio : ℝ
  /-- The side length is positive -/
  side_length_pos : 0 < side_length
  /-- The ratio of EG to GF is 2 -/
  eg_gf_ratio_is_two : eg_gf_ratio = 2

/-- The theorem stating the length of EF in the isosceles triangle -/
theorem isosceles_triangle_ef_length (t : IsoscelesTriangle) (h : t.side_length = 10) :
  ∃ (ef : ℝ), ef = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_ef_length_l817_81716


namespace NUMINAMATH_CALUDE_opposite_of_six_l817_81756

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- Theorem stating that the opposite of 6 is -6
theorem opposite_of_six : opposite 6 = -6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_six_l817_81756


namespace NUMINAMATH_CALUDE_quadratic_condition_l817_81701

/-- The equation (m+1)x^2 - mx + 1 = 0 is quadratic if and only if m ≠ -1 -/
theorem quadratic_condition (m : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, (m + 1) * x^2 - m * x + 1 = a * x^2 + b * x + c) ↔ m ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_condition_l817_81701


namespace NUMINAMATH_CALUDE_sum_first_eight_primes_mod_tenth_prime_l817_81725

def first_eight_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]
def tenth_prime : Nat := 29

theorem sum_first_eight_primes_mod_tenth_prime :
  (first_eight_primes.sum) % tenth_prime = 19 := by sorry

end NUMINAMATH_CALUDE_sum_first_eight_primes_mod_tenth_prime_l817_81725


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l817_81778

def total_players : ℕ := 15
def lineup_size : ℕ := 5
def preselected_players : ℕ := 3

theorem starting_lineup_combinations : 
  Nat.choose (total_players - preselected_players) (lineup_size - preselected_players) = 66 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l817_81778


namespace NUMINAMATH_CALUDE_cosine_equation_solutions_l817_81708

theorem cosine_equation_solutions (x : Real) :
  (∃ (s : Finset Real), s.card = 14 ∧ 
    (∀ y ∈ s, -π ≤ y ∧ y ≤ π ∧ 
      Real.cos (6 * y) + (Real.cos (3 * y))^4 + (Real.sin (2 * y))^2 + (Real.cos y)^2 = 0) ∧
    (∀ z, -π ≤ z ∧ z ≤ π ∧ 
      Real.cos (6 * z) + (Real.cos (3 * z))^4 + (Real.sin (2 * z))^2 + (Real.cos z)^2 = 0 → 
      z ∈ s)) :=
by sorry

end NUMINAMATH_CALUDE_cosine_equation_solutions_l817_81708


namespace NUMINAMATH_CALUDE_cubic_expansion_sum_l817_81704

theorem cubic_expansion_sum (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (2 * x + 1)^3 = a₀ * x^3 + a₁ * x^2 + a₂ * x + a₃) →
  a₁ + a₃ = 13 := by
sorry

end NUMINAMATH_CALUDE_cubic_expansion_sum_l817_81704


namespace NUMINAMATH_CALUDE_triangle_perimeter_l817_81797

theorem triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 4 ∧ c^2 - 10*c + 16 = 0 ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l817_81797


namespace NUMINAMATH_CALUDE_angle_C_is_pi_third_max_area_when_c_is_2root3_l817_81742

noncomputable section

open Real

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = π)
  (h3 : a > 0 ∧ b > 0 ∧ c > 0)
  (h4 : a * sin B = b * sin A)
  (h5 : b * sin C = c * sin B)
  (h6 : c * sin A = a * sin C)

variable (t : Triangle)

-- Given condition
axiom given_condition : t.a * cos t.B + t.b * cos t.A = 2 * t.c * cos t.C

-- Theorem 1: Prove that C = π/3
theorem angle_C_is_pi_third : t.C = π/3 :=
sorry

-- Theorem 2: Prove that if c = 2√3, the maximum area of triangle ABC is 3√3
theorem max_area_when_c_is_2root3 (h : t.c = 2 * Real.sqrt 3) :
  (∀ s : Triangle, s.c = t.c → t.a * t.b * sin t.C / 2 ≥ s.a * s.b * sin s.C / 2) ∧
  t.a * t.b * sin t.C / 2 = 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_angle_C_is_pi_third_max_area_when_c_is_2root3_l817_81742


namespace NUMINAMATH_CALUDE_floor_equation_solution_l817_81715

-- Define the floor function
def floor (x : ℚ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_equation_solution :
  ∀ x : ℚ, floor (5 * x - 2) = 3 * x.num + x.den → x = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l817_81715


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l817_81711

def daily_incomes : List ℝ := [45, 50, 60, 65, 70]
def num_days : ℕ := 5

theorem cab_driver_average_income : 
  (daily_incomes.sum / num_days : ℝ) = 58 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_average_income_l817_81711


namespace NUMINAMATH_CALUDE_milkshake_production_theorem_l817_81758

/-- Represents the milkshake production scenario -/
structure MilkshakeProduction where
  augustus_rate : ℕ
  luna_rate : ℕ
  neptune_rate : ℕ
  total_hours : ℕ
  neptune_start : ℕ
  break_interval : ℕ
  extra_break : ℕ
  break_consumption : ℕ

/-- Calculates the total number of milkshakes produced -/
def total_milkshakes (prod : MilkshakeProduction) : ℕ :=
  sorry

/-- The main theorem stating that given the conditions, 93 milkshakes are produced -/
theorem milkshake_production_theorem (prod : MilkshakeProduction)
  (h1 : prod.augustus_rate = 3)
  (h2 : prod.luna_rate = 7)
  (h3 : prod.neptune_rate = 5)
  (h4 : prod.total_hours = 12)
  (h5 : prod.neptune_start = 3)
  (h6 : prod.break_interval = 3)
  (h7 : prod.extra_break = 7)
  (h8 : prod.break_consumption = 18) :
  total_milkshakes prod = 93 :=
sorry

end NUMINAMATH_CALUDE_milkshake_production_theorem_l817_81758


namespace NUMINAMATH_CALUDE_tangent_length_is_6_l817_81754

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the line passing through the center
def line_equation (x y : ℝ) (a : ℝ) : Prop :=
  x + a*y - 1 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 1)

-- Define point A
def point_A (a : ℝ) : ℝ × ℝ := (-4, a)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem statement
theorem tangent_length_is_6 :
  ∃ (a : ℝ),
    line_equation (circle_center.1) (circle_center.2) a ∧
    (∃ (x y : ℝ), circle_equation x y ∧
      ∃ (B : ℝ × ℝ), B.1 = x ∧ B.2 = y ∧
        (point_A a).1 - B.1 = a * (B.2 - (point_A a).2) ∧
        Real.sqrt (((point_A a).1 - B.1)^2 + ((point_A a).2 - B.2)^2) = 6) :=
by sorry

end NUMINAMATH_CALUDE_tangent_length_is_6_l817_81754


namespace NUMINAMATH_CALUDE_seven_is_target_digit_l817_81779

/-- The numeral we're examining -/
def numeral : ℕ := 657903

/-- The difference between local value and face value -/
def difference : ℕ := 6993

/-- Function to get the local value of a digit in a specific place -/
def localValue (digit : ℕ) (place : ℕ) : ℕ := digit * (10 ^ place)

/-- Function to get the face value of a digit -/
def faceValue (digit : ℕ) : ℕ := digit

/-- Theorem stating that 7 is the only digit in the numeral with the given difference -/
theorem seven_is_target_digit :
  ∃! d : ℕ, d < 10 ∧ 
    (∃ p : ℕ, p < 6 ∧ 
      (numeral / (10 ^ p)) % 10 = d ∧
      localValue d p - faceValue d = difference) :=
sorry

end NUMINAMATH_CALUDE_seven_is_target_digit_l817_81779


namespace NUMINAMATH_CALUDE_sum_f_positive_l817_81785

def f (x : ℝ) : ℝ := x^3 + x

theorem sum_f_positive (a b c : ℝ) (hab : a + b > 0) (hbc : b + c > 0) (hca : c + a > 0) :
  f a + f b + f c > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l817_81785


namespace NUMINAMATH_CALUDE_count_proposition_permutations_l817_81776

/-- The number of distinct permutations of letters in "PROPOSITION" -/
def proposition_permutations : ℕ :=
  Nat.factorial 10 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating the number of distinct permutations of "PROPOSITION" -/
theorem count_proposition_permutations :
  proposition_permutations = 453600 := by
  sorry

end NUMINAMATH_CALUDE_count_proposition_permutations_l817_81776


namespace NUMINAMATH_CALUDE_polynomial_sum_simplification_l817_81769

-- Define the polynomials
def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

-- State the theorem
theorem polynomial_sum_simplification :
  ∀ x : ℝ, p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_sum_simplification_l817_81769


namespace NUMINAMATH_CALUDE_circles_intersection_product_of_coordinates_l817_81728

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 10*y + 25 = 0
def circle2 (x y : ℝ) : Prop := x^2 - 8*x + y^2 - 10*y + 37 = 0

-- Theorem stating that (2, 5) is the intersection point of the two circles
theorem circles_intersection :
  ∃! (x y : ℝ), circle1 x y ∧ circle2 x y ∧ x = 2 ∧ y = 5 := by
  sorry

-- Theorem stating that the product of the coordinates of the intersection point is 10
theorem product_of_coordinates :
  ∀ (x y : ℝ), circle1 x y ∧ circle2 x y → x * y = 10 := by
  sorry

end NUMINAMATH_CALUDE_circles_intersection_product_of_coordinates_l817_81728


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l817_81760

theorem imaginary_part_of_reciprocal (z : ℂ) : z = 1 - 3*I → (1/z).im = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l817_81760


namespace NUMINAMATH_CALUDE_jerrys_age_l817_81757

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 14 → 
  mickey_age = 3 * jerry_age - 4 → 
  jerry_age = 6 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l817_81757


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l817_81736

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n ≥ 2, a n - a (n - 1) = 2) ∧ (a 1 = 1)

theorem tenth_term_of_sequence (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 10 = 19 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l817_81736


namespace NUMINAMATH_CALUDE_quadratic_linear_system_solution_l817_81727

theorem quadratic_linear_system_solution : 
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
    (x₁^2 - 6*x₁ + 8 = 0) ∧
    (x₂^2 - 6*x₂ + 8 = 0) ∧
    (2*x₁ - y₁ = 6) ∧
    (2*x₂ - y₂ = 6) ∧
    (y₁ = 2) ∧
    (y₂ = -2) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_linear_system_solution_l817_81727


namespace NUMINAMATH_CALUDE_lily_pad_half_coverage_l817_81799

/-- Represents the number of days it takes for lily pads to cover the entire lake -/
def full_coverage_days : ℕ := 39

/-- Represents the growth factor of lily pads per day -/
def daily_growth_factor : ℕ := 2

/-- Calculates the number of days required to cover half the lake -/
def half_coverage_days : ℕ := full_coverage_days - 1

theorem lily_pad_half_coverage :
  half_coverage_days = 38 :=
sorry

end NUMINAMATH_CALUDE_lily_pad_half_coverage_l817_81799


namespace NUMINAMATH_CALUDE_investment_interest_calculation_l817_81751

/-- Proves that an investment of $31,200 with a simple annual interest rate of 9% yields a monthly interest payment of $234 -/
theorem investment_interest_calculation (principal : ℝ) (annual_rate : ℝ) (monthly_interest : ℝ) : 
  principal = 31200 ∧ annual_rate = 0.09 → monthly_interest = 234 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_interest_calculation_l817_81751


namespace NUMINAMATH_CALUDE_common_tangent_sum_l817_81796

/-- Given two curves f and g with a common tangent at their intersection point (0, m),
    prove that the sum of their coefficients a and b is 1. -/
theorem common_tangent_sum (a b m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * Real.cos x
  let g : ℝ → ℝ := λ x ↦ x^2 + b*x + 1
  let f' : ℝ → ℝ := λ x ↦ -a * Real.sin x
  let g' : ℝ → ℝ := λ x ↦ 2*x + b
  (f 0 = g 0) ∧ (f' 0 = g' 0) → a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l817_81796


namespace NUMINAMATH_CALUDE_f_negative_one_value_l817_81703

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_negative_one_value :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is an odd function
  (∀ x : ℝ, x > 0 → f x = 2*x - 1) →  -- Definition of f for positive x
  f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_one_value_l817_81703


namespace NUMINAMATH_CALUDE_train_crossing_time_l817_81734

-- Define the given parameters
def train_speed_kmph : ℝ := 72
def train_speed_ms : ℝ := 20
def platform_length : ℝ := 260
def time_cross_platform : ℝ := 31

-- Define the theorem
theorem train_crossing_time (train_length : ℝ) 
  (h1 : train_length + platform_length = train_speed_ms * time_cross_platform)
  (h2 : train_speed_kmph * (1000 / 3600) = train_speed_ms) :
  train_length / train_speed_ms = 18 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_train_crossing_time_l817_81734


namespace NUMINAMATH_CALUDE_borrowed_sum_l817_81747

/-- Given a principal P borrowed at 5% simple interest per annum,
    if after 5 years the interest is Rs. 750 less than P,
    then P must be Rs. 1000. -/
theorem borrowed_sum (P : ℝ) : 
  (P * 0.05 * 5 = P - 750) → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_sum_l817_81747


namespace NUMINAMATH_CALUDE_vincent_book_expenditure_l817_81706

def animal_books : ℕ := 10
def space_books : ℕ := 1
def train_books : ℕ := 3
def book_cost : ℕ := 16

theorem vincent_book_expenditure :
  (animal_books + space_books + train_books) * book_cost = 224 := by
  sorry

end NUMINAMATH_CALUDE_vincent_book_expenditure_l817_81706


namespace NUMINAMATH_CALUDE_company_fund_problem_l817_81767

/-- Proves that the initial amount in the company fund was $950 given the problem conditions --/
theorem company_fund_problem (n : ℕ) : 
  (60 * n - 10 = 50 * n + 150) → 
  (60 * n - 10 = 950) :=
by
  sorry

end NUMINAMATH_CALUDE_company_fund_problem_l817_81767


namespace NUMINAMATH_CALUDE_prob_select_copresidents_from_random_club_l817_81726

/-- Represents a math club with a given number of students and two co-presidents -/
structure MathClub where
  students : Nat
  has_two_copresidents : Bool

/-- Calculates the probability of selecting two co-presidents when choosing three members from a club -/
def prob_select_copresidents (club : MathClub) : Rat :=
  if club.has_two_copresidents then
    (Nat.choose (club.students - 2) 1 : Rat) / (Nat.choose club.students 3 : Rat)
  else
    0

/-- The list of math clubs in the school district -/
def math_clubs : List MathClub := [
  { students := 5, has_two_copresidents := true },
  { students := 7, has_two_copresidents := true },
  { students := 8, has_two_copresidents := true }
]

/-- Theorem stating the probability of selecting two co-presidents when randomly choosing
    three members from a randomly selected club among the given math clubs -/
theorem prob_select_copresidents_from_random_club : 
  (1 / (math_clubs.length : Rat)) * (math_clubs.map prob_select_copresidents).sum = 11 / 60 := by
  sorry

end NUMINAMATH_CALUDE_prob_select_copresidents_from_random_club_l817_81726


namespace NUMINAMATH_CALUDE_solve_for_q_l817_81740

theorem solve_for_q (n d p q : ℝ) (h1 : d ≠ 0) (h2 : p ≠ 0) (h3 : q ≠ 0) 
  (h4 : n = (2 * d * p * q) / (p - q)) : 
  q = (n * p) / (2 * d * p + n) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_q_l817_81740


namespace NUMINAMATH_CALUDE_unique_lcm_gcd_relation_l817_81755

theorem unique_lcm_gcd_relation : 
  ∃! (n : ℕ), n > 0 ∧ Nat.lcm n 100 = Nat.gcd n 100 + 450 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_lcm_gcd_relation_l817_81755


namespace NUMINAMATH_CALUDE_divided_stick_properties_l817_81705

/-- Represents a stick divided into segments by different colored lines -/
structure DividedStick where
  length : ℝ
  red_segments : ℕ
  blue_segments : ℕ
  black_segments : ℕ

/-- Calculates the total number of segments after cutting -/
def total_segments (stick : DividedStick) : ℕ := sorry

/-- Calculates the length of the shortest segment -/
def shortest_segment (stick : DividedStick) : ℝ := sorry

/-- Theorem stating the properties of a stick divided into 8, 12, and 18 segments -/
theorem divided_stick_properties (L : ℝ) (h : L > 0) :
  let stick := DividedStick.mk L 8 12 18
  total_segments stick = 28 ∧ shortest_segment stick = L / 72 := by sorry

end NUMINAMATH_CALUDE_divided_stick_properties_l817_81705


namespace NUMINAMATH_CALUDE_stratified_sampling_survey_l817_81798

theorem stratified_sampling_survey (total_households : ℕ) 
                                   (middle_income : ℕ) 
                                   (low_income : ℕ) 
                                   (high_income_selected : ℕ) : 
  total_households = 480 →
  middle_income = 200 →
  low_income = 160 →
  high_income_selected = 6 →
  ∃ (total_selected : ℕ), 
    total_selected * (total_households - middle_income - low_income) = 
    high_income_selected * total_households ∧
    total_selected = 24 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_survey_l817_81798


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_k_bound_l817_81771

def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 + 2 * k * x - 8

theorem monotone_decreasing_implies_k_bound :
  (∀ x₁ x₂, -5 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ -1 → f k x₁ > f k x₂) →
  k ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_k_bound_l817_81771


namespace NUMINAMATH_CALUDE_divisibility_condition_exists_divisibility_for_all_implies_equality_l817_81765

-- Part (a)
theorem divisibility_condition_exists (n : ℕ+) :
  ∃ (x y : ℕ+), x ≠ y ∧ ∀ j ∈ Finset.range n, (x + j) ∣ (y + j) := by sorry

-- Part (b)
theorem divisibility_for_all_implies_equality (x y : ℕ+) :
  (∀ j : ℕ+, (x + j) ∣ (y + j)) → x = y := by sorry

end NUMINAMATH_CALUDE_divisibility_condition_exists_divisibility_for_all_implies_equality_l817_81765


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l817_81786

theorem power_mod_thirteen : 7^137 % 13 = 11 := by sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l817_81786


namespace NUMINAMATH_CALUDE_pizza_area_increase_l817_81723

theorem pizza_area_increase (r : ℝ) (hr : r > 0) :
  let medium_area := π * r^2
  let large_radius := 1.1 * r
  let large_area := π * large_radius^2
  (large_area - medium_area) / medium_area = 0.21 := by sorry

end NUMINAMATH_CALUDE_pizza_area_increase_l817_81723


namespace NUMINAMATH_CALUDE_games_per_box_l817_81714

theorem games_per_box (initial_games : ℕ) (sold_games : ℕ) (num_boxes : ℕ) 
  (h1 : initial_games = 35)
  (h2 : sold_games = 19)
  (h3 : num_boxes = 2)
  (h4 : initial_games > sold_games) :
  (initial_games - sold_games) / num_boxes = 8 := by
sorry

end NUMINAMATH_CALUDE_games_per_box_l817_81714


namespace NUMINAMATH_CALUDE_quartic_sum_at_3_and_neg_3_l817_81762

def quartic_polynomial (d a b c m : ℝ) (x : ℝ) : ℝ :=
  d * x^4 + a * x^3 + b * x^2 + c * x + m

theorem quartic_sum_at_3_and_neg_3 
  (d a b c m : ℝ) 
  (h1 : quartic_polynomial d a b c m 0 = m)
  (h2 : quartic_polynomial d a b c m 1 = 3 * m)
  (h3 : quartic_polynomial d a b c m (-1) = 4 * m) :
  quartic_polynomial d a b c m 3 + quartic_polynomial d a b c m (-3) = 144 * d + 47 * m := by
  sorry

end NUMINAMATH_CALUDE_quartic_sum_at_3_and_neg_3_l817_81762


namespace NUMINAMATH_CALUDE_min_value_and_existence_l817_81794

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + t*x + 1) / Real.log 2

theorem min_value_and_existence (t : ℝ) (h : t > -2) :
  (∀ x ∈ Set.Icc 0 2, f t x ≥ (if -2 < t ∧ t < 0 then Real.log (1 - t^2/4) / Real.log 2 else 0)) ∧
  (∃ a b : ℝ, a ≠ b ∧ a ∈ Set.Ioo 0 2 ∧ b ∈ Set.Ioo 0 2 ∧ 
   f t a = Real.log a / Real.log 2 ∧ f t b = Real.log b / Real.log 2 ↔ 
   t > -3/2 ∧ t < -1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_existence_l817_81794


namespace NUMINAMATH_CALUDE_red_candies_count_l817_81795

theorem red_candies_count (total : ℕ) (blue : ℕ) (h1 : total = 3409) (h2 : blue = 3264) :
  total - blue = 145 := by
  sorry

end NUMINAMATH_CALUDE_red_candies_count_l817_81795


namespace NUMINAMATH_CALUDE_largest_prime_common_factor_l817_81791

def is_largest_prime_common_factor (n : ℕ) : Prop :=
  n.Prime ∧
  n ∣ 462 ∧
  n ∣ 385 ∧
  ∀ m : ℕ, m.Prime → m ∣ 462 → m ∣ 385 → m ≤ n

theorem largest_prime_common_factor :
  is_largest_prime_common_factor 7 := by sorry

end NUMINAMATH_CALUDE_largest_prime_common_factor_l817_81791


namespace NUMINAMATH_CALUDE_parallelogram_area_14_24_l817_81707

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 14 cm and height 24 cm is 336 cm² -/
theorem parallelogram_area_14_24 : parallelogramArea 14 24 = 336 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_14_24_l817_81707


namespace NUMINAMATH_CALUDE_pet_shelter_adoption_time_l817_81774

/-- Given a pet shelter scenario, calculate the number of days needed to adopt all puppies -/
theorem pet_shelter_adoption_time (initial_puppies : ℕ) (additional_puppies : ℕ) (adoption_rate : ℕ) : 
  initial_puppies = 9 → additional_puppies = 12 → adoption_rate = 3 →
  (initial_puppies + additional_puppies) / adoption_rate = 7 := by
sorry

end NUMINAMATH_CALUDE_pet_shelter_adoption_time_l817_81774


namespace NUMINAMATH_CALUDE_second_car_speed_l817_81729

/-- Given two cars traveling in opposite directions for 2.5 hours,
    with one car traveling at 60 mph and the total distance between them
    being 310 miles after 2.5 hours, prove that the speed of the second car is 64 mph. -/
theorem second_car_speed (car1_speed : ℝ) (car2_speed : ℝ) (time : ℝ) (total_distance : ℝ) :
  car1_speed = 60 →
  time = 2.5 →
  total_distance = 310 →
  car1_speed * time + car2_speed * time = total_distance →
  car2_speed = 64 := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_l817_81729


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l817_81772

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l817_81772


namespace NUMINAMATH_CALUDE_door_unlock_problem_l817_81724

-- Define the number of buttons and the number of buttons to press
def total_buttons : ℕ := 10
def buttons_to_press : ℕ := 3

-- Define the time for each attempt
def time_per_attempt : ℕ := 2

-- Calculate the total number of combinations
def total_combinations : ℕ := Nat.choose total_buttons buttons_to_press

-- Define the maximum time needed (in seconds)
def max_time : ℕ := total_combinations * time_per_attempt

-- Define the average time needed (in seconds)
def avg_time : ℚ := (1 + total_combinations : ℚ) / 2 * time_per_attempt

-- Define the maximum number of attempts in 60 seconds
def max_attempts_in_minute : ℕ := 60 / time_per_attempt

theorem door_unlock_problem :
  (max_time = 240) ∧
  (avg_time = 121) ∧
  (max_attempts_in_minute = 30) ∧
  ((max_attempts_in_minute - 1 : ℚ) / total_combinations = 29 / 120) := by
  sorry

end NUMINAMATH_CALUDE_door_unlock_problem_l817_81724


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l817_81792

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := k * x^2 + (2*k - 1) * x + k

-- Define the condition for two real roots
def has_two_real_roots (k : ℝ) : Prop :=
  k ≠ 0 ∧ (2*k - 1)^2 - 4*k*k ≥ 0

-- Theorem statement
theorem quadratic_roots_condition (k : ℝ) :
  has_two_real_roots k ↔ k ≤ 1/4 ∧ k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l817_81792


namespace NUMINAMATH_CALUDE_three_four_five_pythagorean_triple_l817_81745

/-- Definition of a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

/-- Theorem: (3,4,5) is a Pythagorean triple -/
theorem three_four_five_pythagorean_triple :
  isPythagoreanTriple 3 4 5 := by
  sorry

end NUMINAMATH_CALUDE_three_four_five_pythagorean_triple_l817_81745


namespace NUMINAMATH_CALUDE_prop_C_and_D_l817_81764

theorem prop_C_and_D : 
  (∀ a b : ℝ, a > b → a^3 > b^3) ∧ 
  (∀ a b c d : ℝ, (a > b ∧ c > d) → a - d > b - c) := by
  sorry

end NUMINAMATH_CALUDE_prop_C_and_D_l817_81764


namespace NUMINAMATH_CALUDE_trip_duration_l817_81753

/-- Represents the duration of a car trip with varying speeds -/
def car_trip (initial_hours : ℝ) (initial_speed : ℝ) (additional_speed : ℝ) (average_speed : ℝ) : Prop :=
  ∃ (additional_hours : ℝ),
    let total_hours := initial_hours + additional_hours
    let total_distance := initial_hours * initial_speed + additional_hours * additional_speed
    (total_distance / total_hours = average_speed) ∧
    (total_hours = 15)

/-- The main theorem stating that under given conditions, the trip duration is 15 hours -/
theorem trip_duration :
  car_trip 5 30 42 38 := by
  sorry

end NUMINAMATH_CALUDE_trip_duration_l817_81753


namespace NUMINAMATH_CALUDE_opposite_points_theorem_l817_81702

def number_line_points (a b : ℝ) : Prop :=
  a < b ∧ a = -b ∧ b - a = 6.4

theorem opposite_points_theorem (a b : ℝ) :
  number_line_points a b → a = -3.2 ∧ b = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_points_theorem_l817_81702


namespace NUMINAMATH_CALUDE_max_x_minus_y_value_l817_81737

theorem max_x_minus_y_value (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) :
  ∃ (max : ℝ), max = 2 / Real.sqrt 3 ∧ ∀ (a b : ℝ), 3 * (a^2 + b^2) = a + b → a - b ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_value_l817_81737


namespace NUMINAMATH_CALUDE_emily_necklaces_l817_81748

def necklace_problem (total_beads : ℕ) (num_necklaces : ℕ) (beads_per_necklace : ℕ) : Prop :=
  total_beads = num_necklaces * beads_per_necklace

theorem emily_necklaces : 
  ∃ (beads_per_necklace : ℕ), necklace_problem 308 11 beads_per_necklace ∧ beads_per_necklace = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_emily_necklaces_l817_81748


namespace NUMINAMATH_CALUDE_increasing_positive_function_inequality_l817_81784

theorem increasing_positive_function_inequality 
  (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y)
  (h_positive : ∀ x, f x > 0)
  (h_differentiable : Differentiable ℝ f) :
  3 * f (-2) > 2 * f (-3) := by
  sorry

end NUMINAMATH_CALUDE_increasing_positive_function_inequality_l817_81784


namespace NUMINAMATH_CALUDE_pizza_topping_cost_l817_81746

/-- Represents the cost of a pizza with toppings -/
def pizza_cost (base_cost : ℚ) (first_topping_cost : ℚ) (next_two_toppings_cost : ℚ) 
  (num_slices : ℕ) (cost_per_slice : ℚ) (num_toppings : ℕ) : Prop :=
  let total_cost := cost_per_slice * num_slices
  let known_cost := base_cost + first_topping_cost + 2 * next_two_toppings_cost
  let remaining_toppings_cost := total_cost - known_cost
  let num_remaining_toppings := num_toppings - 3
  remaining_toppings_cost / num_remaining_toppings = 0.5

theorem pizza_topping_cost : 
  pizza_cost 10 2 1 8 2 7 :=
by sorry

end NUMINAMATH_CALUDE_pizza_topping_cost_l817_81746


namespace NUMINAMATH_CALUDE_fixed_costs_correct_l817_81759

/-- Represents the fixed monthly costs for producing electronic components -/
def fixed_monthly_costs : ℝ := 16500

/-- Represents the production cost per component -/
def production_cost_per_unit : ℝ := 80

/-- Represents the shipping cost per component -/
def shipping_cost_per_unit : ℝ := 5

/-- Represents the number of components produced and sold monthly -/
def monthly_units : ℕ := 150

/-- Represents the lowest selling price per component -/
def lowest_selling_price : ℝ := 195

/-- Theorem stating that the fixed monthly costs are correct given the conditions -/
theorem fixed_costs_correct :
  fixed_monthly_costs =
    monthly_units * lowest_selling_price -
    monthly_units * (production_cost_per_unit + shipping_cost_per_unit) := by
  sorry

#check fixed_costs_correct

end NUMINAMATH_CALUDE_fixed_costs_correct_l817_81759


namespace NUMINAMATH_CALUDE_find_x_l817_81700

theorem find_x : ∃ X : ℝ, (X + 20 / 90) * 90 = 9020 ∧ X = 9000 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l817_81700


namespace NUMINAMATH_CALUDE_inequality_subtraction_l817_81741

theorem inequality_subtraction (a b c d : ℝ) (h1 : a > b) (h2 : c < d) : a - c > b - d := by
  sorry

end NUMINAMATH_CALUDE_inequality_subtraction_l817_81741


namespace NUMINAMATH_CALUDE_correct_statements_l817_81750

theorem correct_statements (a b c d : ℝ) :
  (ab > 0 ∧ bc - ad > 0 → c / a - d / b > 0) ∧
  (a > b ∧ c > d → a - d > b - c) :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_l817_81750


namespace NUMINAMATH_CALUDE_square_difference_40_39_l817_81717

theorem square_difference_40_39 : (40 : ℕ)^2 - (39 : ℕ)^2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_40_39_l817_81717


namespace NUMINAMATH_CALUDE_car_rate_problem_l817_81709

/-- Given two cars starting at the same time and point, with one car traveling at 60 mph,
    if after 3 hours the distance between them is 30 miles,
    then the rate of the other car is 50 mph. -/
theorem car_rate_problem (rate1 : ℝ) : 
  (60 * 3 = rate1 * 3 + 30) → rate1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_rate_problem_l817_81709


namespace NUMINAMATH_CALUDE_negation_of_proposition_l817_81773

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 0 → x^2 + x + 1 > 0)) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + x₀ + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l817_81773


namespace NUMINAMATH_CALUDE_quadrilateral_offset_l817_81739

theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) (area : ℝ) :
  diagonal = 10 →
  offset1 = 3 →
  area = 50 →
  area = (1 / 2) * diagonal * offset1 + (1 / 2) * diagonal * offset2 →
  offset2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_offset_l817_81739


namespace NUMINAMATH_CALUDE_parabola_through_point_2_4_l817_81730

/-- A parabola passing through the point (2, 4) can be represented by either y² = 8x or x² = y -/
theorem parabola_through_point_2_4 :
  ∃ (f : ℝ → ℝ), (f 2 = 4 ∧ (∀ x y : ℝ, y = f x ↔ (y^2 = 8*x ∨ x^2 = y))) :=
by sorry

end NUMINAMATH_CALUDE_parabola_through_point_2_4_l817_81730


namespace NUMINAMATH_CALUDE_smallest_value_l817_81718

theorem smallest_value (a b : ℝ) (h : b < 0) : (a + b < a) ∧ (a + b < a - b) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_l817_81718


namespace NUMINAMATH_CALUDE_initial_temperature_l817_81722

theorem initial_temperature (T : ℝ) : (2 * T - 30) * 0.70 + 24 = 59 ↔ T = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_temperature_l817_81722


namespace NUMINAMATH_CALUDE_sameTerminalSideAs315_eq_l817_81783

/-- The set of angles with the same terminal side as 315° -/
def sameTerminalSideAs315 : Set ℝ :=
  {α | ∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 4}

/-- Theorem stating that the set of angles with the same terminal side as 315° 
    is equal to {α | α = 2kπ - π/4, k ∈ ℤ} -/
theorem sameTerminalSideAs315_eq : 
  sameTerminalSideAs315 = {α | ∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 4} := by
  sorry


end NUMINAMATH_CALUDE_sameTerminalSideAs315_eq_l817_81783


namespace NUMINAMATH_CALUDE_ab_power_2023_l817_81721

theorem ab_power_2023 (a b : ℝ) (h : |a - 2| + (b + 1/2)^2 = 0) : (a * b)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ab_power_2023_l817_81721
