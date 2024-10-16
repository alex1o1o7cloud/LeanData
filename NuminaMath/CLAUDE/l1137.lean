import Mathlib

namespace NUMINAMATH_CALUDE_sprite_volume_l1137_113718

def maazaVolume : ℕ := 80
def pepsiVolume : ℕ := 144
def totalCans : ℕ := 37

def canVolume : ℕ := Nat.gcd maazaVolume pepsiVolume

theorem sprite_volume :
  ∃ (spriteVolume : ℕ),
    spriteVolume = canVolume * (totalCans - (maazaVolume / canVolume + pepsiVolume / canVolume)) ∧
    spriteVolume = 368 := by
  sorry

end NUMINAMATH_CALUDE_sprite_volume_l1137_113718


namespace NUMINAMATH_CALUDE_nonzero_real_equation_solution_l1137_113742

theorem nonzero_real_equation_solution (x : ℝ) (h : x ≠ 0) :
  (5 * x)^10 = (10 * x)^5 ↔ x = 2/5 := by sorry

end NUMINAMATH_CALUDE_nonzero_real_equation_solution_l1137_113742


namespace NUMINAMATH_CALUDE_complex_arithmetic_result_l1137_113708

theorem complex_arithmetic_result :
  let A : ℂ := 5 - 2*I
  let M : ℂ := -3 + 3*I
  let S : ℂ := 2*I
  let P : ℝ := (1/2 : ℝ)
  A - M + S - (P : ℂ) = 7.5 - 3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_result_l1137_113708


namespace NUMINAMATH_CALUDE_correct_first_grade_sample_size_l1137_113706

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  first_grade_students : ℕ
  sample_size : ℕ

/-- Calculates the number of first-grade students to be selected in a stratified sample -/
def stratified_sample_size (school : School) : ℕ :=
  (school.first_grade_students * school.sample_size) / school.total_students

/-- Theorem stating the correct number of first-grade students to be selected -/
theorem correct_first_grade_sample_size (school : School) 
  (h1 : school.total_students = 2000)
  (h2 : school.first_grade_students = 400)
  (h3 : school.sample_size = 200) :
  stratified_sample_size school = 40 := by
  sorry

#eval stratified_sample_size { total_students := 2000, first_grade_students := 400, sample_size := 200 }

end NUMINAMATH_CALUDE_correct_first_grade_sample_size_l1137_113706


namespace NUMINAMATH_CALUDE_polynomial_sum_l1137_113762

-- Define the polynomial g(x)
def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem polynomial_sum (a b c d : ℝ) :
  g a b c d (1 + I) = 0 → g a b c d (3*I) = 0 → a + b + c + d = 27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l1137_113762


namespace NUMINAMATH_CALUDE_parallel_planes_from_common_perpendicular_l1137_113759

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_from_common_perpendicular 
  (m : Line) (α β : Plane) (h_diff : α ≠ β) :
  perpendicular m α → perpendicular m β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_common_perpendicular_l1137_113759


namespace NUMINAMATH_CALUDE_team_formation_count_l1137_113736

/-- The number of ways to form a team of 4 students from 6 university students -/
def team_formation_ways : ℕ := 180

/-- The number of university students -/
def total_students : ℕ := 6

/-- The number of students in the team -/
def team_size : ℕ := 4

/-- The number of team leaders -/
def num_leaders : ℕ := 1

/-- The number of deputy team leaders -/
def num_deputies : ℕ := 1

/-- The number of ordinary members -/
def num_ordinary : ℕ := 2

theorem team_formation_count :
  team_formation_ways = 
    (total_students.choose num_leaders) * 
    ((total_students - num_leaders).choose num_deputies) * 
    ((total_students - num_leaders - num_deputies).choose num_ordinary) :=
sorry

end NUMINAMATH_CALUDE_team_formation_count_l1137_113736


namespace NUMINAMATH_CALUDE_average_height_of_four_l1137_113750

/-- Given the heights of four people with specific relationships, prove their average height --/
theorem average_height_of_four (zara_height brixton_height zora_height itzayana_height : ℕ) : 
  zara_height = 64 →
  brixton_height = zara_height →
  zora_height = brixton_height - 8 →
  itzayana_height = zora_height + 4 →
  (zara_height + brixton_height + zora_height + itzayana_height) / 4 = 61 := by
  sorry

end NUMINAMATH_CALUDE_average_height_of_four_l1137_113750


namespace NUMINAMATH_CALUDE_max_teams_double_round_robin_l1137_113784

/-- A schedule for a double round robin tournament. -/
def Schedule (n : ℕ) := Fin n → Fin 4 → List (Fin n)

/-- Predicate to check if a schedule is valid according to the tournament rules. -/
def is_valid_schedule (n : ℕ) (s : Schedule n) : Prop :=
  -- Each team plays with every other team twice
  (∀ i j : Fin n, i ≠ j → (∃ w : Fin 4, i ∈ s j w) ∧ (∃ w : Fin 4, j ∈ s i w)) ∧
  -- If a team has a home game in a week, it cannot have any away games that week
  (∀ i : Fin n, ∀ w : Fin 4, (s i w).length > 0 → ∀ j : Fin n, i ∉ s j w)

/-- The maximum number of teams that can complete the tournament in 4 weeks is 6. -/
theorem max_teams_double_round_robin : 
  (∃ s : Schedule 6, is_valid_schedule 6 s) ∧ 
  (∀ s : Schedule 7, ¬ is_valid_schedule 7 s) :=
sorry

end NUMINAMATH_CALUDE_max_teams_double_round_robin_l1137_113784


namespace NUMINAMATH_CALUDE_black_ball_probability_l1137_113773

theorem black_ball_probability (total : ℕ) (white yellow black : ℕ) :
  total = white + yellow + black →
  white = 10 →
  yellow = 5 →
  black = 10 →
  (black : ℚ) / (yellow + black) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_black_ball_probability_l1137_113773


namespace NUMINAMATH_CALUDE_integer_solution_for_inequalities_l1137_113791

theorem integer_solution_for_inequalities : 
  ∃! (n : ℤ), n + 15 > 16 ∧ -3 * n^2 > -27 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_integer_solution_for_inequalities_l1137_113791


namespace NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l1137_113779

theorem unique_solution_for_prime_equation :
  ∀ a b : ℕ,
  Prime a →
  b > 0 →
  9 * (2 * a + b)^2 = 509 * (4 * a + 511 * b) →
  a = 251 ∧ b = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l1137_113779


namespace NUMINAMATH_CALUDE_unit_circle_arc_angle_l1137_113705

/-- The central angle (in radians) corresponding to an arc of length 1 in a unit circle is 1. -/
theorem unit_circle_arc_angle (θ : ℝ) : θ = 1 := by
  sorry

end NUMINAMATH_CALUDE_unit_circle_arc_angle_l1137_113705


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l1137_113741

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l1137_113741


namespace NUMINAMATH_CALUDE_min_box_value_l1137_113798

theorem min_box_value (a b Box : ℤ) : 
  a ≠ b ∧ a ≠ Box ∧ b ≠ Box →
  (∀ x, (a*x + b)*(b*x + a) = 30*x^2 + Box*x + 30) →
  a * b = 30 →
  Box = a^2 + b^2 →
  (∀ a' b' Box' : ℤ, 
    a' ≠ b' ∧ a' ≠ Box' ∧ b' ≠ Box' →
    (∀ x, (a'*x + b')*(b'*x + a') = 30*x^2 + Box'*x + 30) →
    a' * b' = 30 →
    Box' = a'^2 + b'^2 →
    Box ≤ Box') →
  Box = 61 := by
sorry

end NUMINAMATH_CALUDE_min_box_value_l1137_113798


namespace NUMINAMATH_CALUDE_max_parts_formula_initial_values_correct_l1137_113761

/-- The maximum number of parts that n ellipses can divide a plane into -/
def max_parts (n : ℕ+) : ℕ :=
  2 * n.val * n.val - 2 * n.val + 2

/-- Theorem stating the formula for the maximum number of parts -/
theorem max_parts_formula (n : ℕ+) : max_parts n = 2 * n.val * n.val - 2 * n.val + 2 := by
  sorry

/-- The first few values of the sequence are correct -/
theorem initial_values_correct :
  max_parts 1 = 2 ∧ max_parts 2 = 6 ∧ max_parts 3 = 14 ∧ max_parts 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_max_parts_formula_initial_values_correct_l1137_113761


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1137_113760

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 9*x + 20 < 1 ↔ (9 - Real.sqrt 5) / 2 < x ∧ x < (9 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1137_113760


namespace NUMINAMATH_CALUDE_arithmetic_sequence_logarithm_l1137_113788

theorem arithmetic_sequence_logarithm (a b : ℝ) (m : ℝ) :
  a > 0 ∧ b > 0 ∧
  (2 : ℝ) ^ a = m ∧
  (3 : ℝ) ^ b = m ∧
  2 * a * b = a + b →
  m = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_logarithm_l1137_113788


namespace NUMINAMATH_CALUDE_line_hyperbola_tangency_l1137_113794

/-- The line y = k(x - √2) and the hyperbola x^2 - y^2 = 1 have only one point in common if and only if k = 1 or k = -1 -/
theorem line_hyperbola_tangency (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * (p.1 - Real.sqrt 2) ∧ p.1^2 - p.2^2 = 1) ↔ (k = 1 ∨ k = -1) := by
  sorry


end NUMINAMATH_CALUDE_line_hyperbola_tangency_l1137_113794


namespace NUMINAMATH_CALUDE_tangent_circles_max_product_l1137_113795

/-- Two externally tangent circles with given equations have a maximum product of their x-offsets --/
theorem tangent_circles_max_product (a b : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y + 2)^2 = 4) →
  (∃ x y : ℝ, (x + b)^2 + (y + 2)^2 = 1) →
  (∀ x y : ℝ, (x - a)^2 + (y + 2)^2 = 4 → (x + b)^2 + (y + 2)^2 = 1 → 
    ∃ t : ℝ, (x - a)^2 + (y + 2)^2 = (x + b)^2 + (y + 2)^2 + ((2 - 1) * t)^2) →
  (∀ c : ℝ, a * b ≤ c → c ≤ 9/4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_max_product_l1137_113795


namespace NUMINAMATH_CALUDE_distance_to_origin_l1137_113723

theorem distance_to_origin : Real.sqrt (3^2 + (-2)^2) = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_distance_to_origin_l1137_113723


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1137_113715

/-- The equation of the line passing through the center of the circle x^2 + 2x + y^2 = 0
    and perpendicular to the line x + y = 0 is x - y + 1 = 0. -/
theorem perpendicular_line_equation : ∃ (a b c : ℝ),
  (∀ x y : ℝ, x^2 + 2*x + y^2 = 0 → (x + 1)^2 + y^2 = 1) ∧ 
  (a*1 + b*1 = 0) ∧
  (a*x + b*y + c = 0 ↔ x - y + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1137_113715


namespace NUMINAMATH_CALUDE_cubic_factorization_l1137_113703

theorem cubic_factorization (x : ℝ) : x^3 - 9*x = x*(x+3)*(x-3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1137_113703


namespace NUMINAMATH_CALUDE_solution_of_equation_l1137_113756

theorem solution_of_equation (x : ℝ) : 2 * x - 4 * x = 0 ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l1137_113756


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1137_113770

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 15) :
  x^3 + y^3 = 152 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1137_113770


namespace NUMINAMATH_CALUDE_worker_r_earnings_l1137_113726

/-- Given the daily earnings of three workers p, q, and r, prove that r earns 50 per day. -/
theorem worker_r_earnings
  (p q r : ℚ)  -- Daily earnings of workers p, q, and r
  (h1 : 9 * (p + q + r) = 1800)  -- p, q, and r together earn 1800 in 9 days
  (h2 : 5 * (p + r) = 600)  -- p and r can earn 600 in 5 days
  (h3 : 7 * (q + r) = 910)  -- q and r can earn 910 in 7 days
  : r = 50 := by
  sorry


end NUMINAMATH_CALUDE_worker_r_earnings_l1137_113726


namespace NUMINAMATH_CALUDE_average_weight_increase_l1137_113730

theorem average_weight_increase (initial_weight : ℝ) : 
  let initial_average := (initial_weight + 65) / 2
  let new_average := (initial_weight + 74) / 2
  new_average - initial_average = 4.5 := by
sorry


end NUMINAMATH_CALUDE_average_weight_increase_l1137_113730


namespace NUMINAMATH_CALUDE_triangle_area_l1137_113751

/-- The area of a triangle with vertices (0,4,13), (-2,3,9), and (-5,6,9) is (3√30)/4 -/
theorem triangle_area : 
  let A : ℝ × ℝ × ℝ := (0, 4, 13)
  let B : ℝ × ℝ × ℝ := (-2, 3, 9)
  let C : ℝ × ℝ × ℝ := (-5, 6, 9)
  let area := Real.sqrt (
    let s := (Real.sqrt 21 + 3 * Real.sqrt 2 + 3 * Real.sqrt 5) / 2
    s * (s - Real.sqrt 21) * (s - 3 * Real.sqrt 2) * (s - 3 * Real.sqrt 5)
  )
  area = 3 * Real.sqrt 30 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1137_113751


namespace NUMINAMATH_CALUDE_total_gold_stars_l1137_113737

def monday_stars : ℕ := 4
def tuesday_stars : ℕ := 7
def wednesday_stars : ℕ := 3
def thursday_stars : ℕ := 8
def friday_stars : ℕ := 2

theorem total_gold_stars : 
  monday_stars + tuesday_stars + wednesday_stars + thursday_stars + friday_stars = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_gold_stars_l1137_113737


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1137_113740

theorem right_rectangular_prism_volume 
  (a b c : ℝ) 
  (h1 : a * b = 36) 
  (h2 : a * c = 48) 
  (h3 : b * c = 72) : 
  a * b * c = 168 := by
sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1137_113740


namespace NUMINAMATH_CALUDE_count_quadratic_integer_solutions_l1137_113763

theorem count_quadratic_integer_solutions :
  ∃ (S : Finset ℕ), 
    (∀ a ∈ S, a > 0 ∧ a ≤ 40) ∧
    (∀ a ∈ S, ∃ x y : ℤ, x ≠ y ∧ x^2 + (3*a + 2)*x + a^2 = 0 ∧ y^2 + (3*a + 2)*y + a^2 = 0) ∧
    (∀ a : ℕ, a > 0 → a ≤ 40 →
      (∃ x y : ℤ, x ≠ y ∧ x^2 + (3*a + 2)*x + a^2 = 0 ∧ y^2 + (3*a + 2)*y + a^2 = 0) →
      a ∈ S) ∧
    Finset.card S = 5 :=
sorry

end NUMINAMATH_CALUDE_count_quadratic_integer_solutions_l1137_113763


namespace NUMINAMATH_CALUDE_weekly_reading_time_l1137_113789

def daily_meditation_time : ℝ := 1
def daily_reading_time : ℝ := 2 * daily_meditation_time
def days_in_week : ℕ := 7

theorem weekly_reading_time :
  daily_reading_time * (days_in_week : ℝ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_weekly_reading_time_l1137_113789


namespace NUMINAMATH_CALUDE_smallest_clock_equivalent_square_l1137_113733

theorem smallest_clock_equivalent_square : ∃ (n : ℕ), 
  n > 4 ∧ 
  24 ∣ (n^2 - n) ∧ 
  ∀ (m : ℕ), m > 4 ∧ m < n → ¬(24 ∣ (m^2 - m)) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_clock_equivalent_square_l1137_113733


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1137_113721

theorem arithmetic_sequence_length : 
  ∀ (a d : ℤ) (n : ℕ), 
    a - d * (n - 1) = 39 → 
    a = 147 → 
    d = 3 → 
    n = 37 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1137_113721


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1137_113734

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (5 + n) = 8 → n = 59 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1137_113734


namespace NUMINAMATH_CALUDE_matrix_max_min_element_l1137_113745

theorem matrix_max_min_element
  (m n : ℕ)
  (a : Fin m → ℝ)
  (b : Fin n → ℝ)
  (p : Fin m → ℝ)
  (q : Fin n → ℝ)
  (hp : ∀ i, p i > 0)
  (hq : ∀ j, q j > 0) :
  ∃ (k : Fin m) (l : Fin n),
    (∀ j, (a k + b l) / (p k + q l) ≥ (a k + b j) / (p k + q j)) ∧
    (∀ i, (a k + b l) / (p k + q l) ≤ (a i + b l) / (p i + q l)) :=
by sorry

end NUMINAMATH_CALUDE_matrix_max_min_element_l1137_113745


namespace NUMINAMATH_CALUDE_no_eight_roots_for_composite_quadratics_l1137_113707

/-- A quadratic trinomial is a polynomial of degree 2 -/
def QuadraticTrinomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem no_eight_roots_for_composite_quadratics :
  ¬ ∃ (f g h : ℝ → ℝ),
    QuadraticTrinomial f ∧ QuadraticTrinomial g ∧ QuadraticTrinomial h ∧
    (∀ x, f (g (h x)) = 0 ↔ x ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Set ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_no_eight_roots_for_composite_quadratics_l1137_113707


namespace NUMINAMATH_CALUDE_sqrt_8_minus_abs_neg_2_plus_reciprocal_1_3_minus_2_cos_45_l1137_113793

theorem sqrt_8_minus_abs_neg_2_plus_reciprocal_1_3_minus_2_cos_45 :
  Real.sqrt 8 - abs (-2) + (1/3)⁻¹ - 2 * Real.cos (45 * π / 180) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_8_minus_abs_neg_2_plus_reciprocal_1_3_minus_2_cos_45_l1137_113793


namespace NUMINAMATH_CALUDE_sin_sum_identity_l1137_113767

theorem sin_sum_identity (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin (5*π/6 - x) + (Real.sin (π/3 - x))^2 = 19/16 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_identity_l1137_113767


namespace NUMINAMATH_CALUDE_profit_percentage_l1137_113714

theorem profit_percentage (cost selling : ℝ) (h : cost > 0) :
  60 * cost = 40 * selling →
  (selling - cost) / cost * 100 = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l1137_113714


namespace NUMINAMATH_CALUDE_solve_for_t_l1137_113744

theorem solve_for_t (s t : ℤ) (eq1 : 9 * s + 5 * t = 108) (eq2 : s = t - 2) : t = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l1137_113744


namespace NUMINAMATH_CALUDE_trig_problem_l1137_113722

theorem trig_problem (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (π - α) + Real.cos (2 * π + α) = Real.sqrt 2 / 3) : 
  (Real.sin α - Real.cos α = 4 / 3) ∧ 
  (Real.tan α = -(9 + 4 * Real.sqrt 2) / 7) := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l1137_113722


namespace NUMINAMATH_CALUDE_variations_difference_l1137_113786

theorem variations_difference (n : ℕ) : n ^ 3 = n * (n - 1) * (n - 2) + 225 ↔ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_variations_difference_l1137_113786


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1137_113749

theorem trigonometric_identities :
  (2 * Real.sin (30 * π / 180) - Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1 / 2) ∧
  ((-1)^2023 + 2 * Real.sin (45 * π / 180) - Real.cos (30 * π / 180) + 
   Real.sin (60 * π / 180) + Real.tan (60 * π / 180)^2 = 2 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1137_113749


namespace NUMINAMATH_CALUDE_system_solution_existence_l1137_113792

theorem system_solution_existence (m : ℝ) : 
  (∃ x y : ℝ, y = m * x + 5 ∧ y = (3 * m - 2) * x + 6) ↔ m ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_existence_l1137_113792


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1137_113771

/-- Proves that the complex number z = (-8 - 7i)(-3i) is located in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant : 
  let z : ℂ := (-8 - 7*I) * (-3*I)
  (z.re < 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1137_113771


namespace NUMINAMATH_CALUDE_probability_three_kings_or_ace_value_l1137_113748

/-- Represents a standard deck of cards --/
structure Deck :=
  (total_cards : ℕ)
  (queens : ℕ)
  (kings : ℕ)
  (aces : ℕ)

/-- The probability of drawing either three Kings or at least one Ace --/
def probability_three_kings_or_ace (d : Deck) : ℚ :=
  let p_three_kings := (d.kings : ℚ) / d.total_cards * (d.kings - 1) / (d.total_cards - 1) * (d.kings - 2) / (d.total_cards - 2)
  let p_no_aces := (d.total_cards - d.aces : ℚ) / d.total_cards * (d.total_cards - d.aces - 1) / (d.total_cards - 1) * (d.total_cards - d.aces - 2) / (d.total_cards - 2)
  p_three_kings + (1 - p_no_aces)

/-- The theorem to be proved --/
theorem probability_three_kings_or_ace_value :
  let d : Deck := ⟨52, 4, 4, 4⟩
  probability_three_kings_or_ace d = 961 / 4420 := by
  sorry


end NUMINAMATH_CALUDE_probability_three_kings_or_ace_value_l1137_113748


namespace NUMINAMATH_CALUDE_right_triangle_sides_l1137_113765

theorem right_triangle_sides : ∃! (a b c : ℝ), 
  ((a = 1 ∧ b = 2 ∧ c = 2) ∨
   (a = 1 ∧ b = 1 ∧ c = Real.sqrt 3) ∨
   (a = 3 ∧ b = 4 ∧ c = 5) ∨
   (a = 4 ∧ b = 5 ∧ c = 6)) ∧
  (a^2 + b^2 = c^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l1137_113765


namespace NUMINAMATH_CALUDE_geometric_sequence_product_property_l1137_113747

/-- A sequence is geometric if there exists a non-zero common ratio between consecutive terms. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- The property that a_m * a_n = a_p * a_q for specific m, n, p, q. -/
def HasProductProperty (a : ℕ → ℝ) (m n p q : ℕ) : Prop :=
  a m * a n = a p * a q

theorem geometric_sequence_product_property 
  (a : ℕ → ℝ) (m n p q : ℕ) 
  (hm : m > 0) (hn : n > 0) (hp : p > 0) (hq : q > 0)
  (h_sum : m + n = p + q) :
  IsGeometricSequence a → HasProductProperty a m n p q :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_property_l1137_113747


namespace NUMINAMATH_CALUDE_rachel_apple_trees_l1137_113701

/-- The number of apple trees Rachel has -/
def num_trees : ℕ := 3

/-- The number of apples picked from each tree -/
def apples_per_tree : ℕ := 8

/-- The total number of apples remaining after picking -/
def apples_remaining : ℕ := 9

/-- The initial total number of apples on all trees -/
def initial_apples : ℕ := 33

theorem rachel_apple_trees :
  num_trees * apples_per_tree + apples_remaining = initial_apples :=
sorry

end NUMINAMATH_CALUDE_rachel_apple_trees_l1137_113701


namespace NUMINAMATH_CALUDE_function_properties_l1137_113752

/-- Given a function f(x) = x - a*exp(x) + b, where a > 0 and b is real,
    this theorem states two properties:
    1. The maximum value of f(x) is ln(1/a) - 1 + b
    2. If f has two distinct zeros x₁ and x₂, then x₁ + x₂ < -2*ln(a) -/
theorem function_properties (a b : ℝ) (ha : a > 0) :
  let f := fun x => x - a * Real.exp x + b
  (∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = Real.log (1 / a) - 1 + b) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ = 0 → f x₂ = 0 → x₁ + x₂ < -2 * Real.log a) := by
  sorry


end NUMINAMATH_CALUDE_function_properties_l1137_113752


namespace NUMINAMATH_CALUDE_vacant_seats_l1137_113746

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (h1 : total_seats = 600) (h2 : filled_percentage = 62/100) : 
  (total_seats : ℚ) * (1 - filled_percentage) = 228 := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l1137_113746


namespace NUMINAMATH_CALUDE_sugar_amount_is_correct_l1137_113732

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- Calculates the amount of sugar needed in the new recipe -/
def sugar_needed (original : RecipeRatio) (water_new : ℚ) : ℚ :=
  let new_flour_water_ratio := (original.flour / original.water) * 3
  let new_flour_sugar_ratio := (original.flour / original.sugar) / 3
  let flour_new := (new_flour_water_ratio * water_new)
  flour_new / new_flour_sugar_ratio

/-- Theorem: Given the conditions, the amount of sugar needed is 0.75 cups -/
theorem sugar_amount_is_correct (original : RecipeRatio) 
  (h1 : original.flour = 11)
  (h2 : original.water = 8)
  (h3 : original.sugar = 1)
  (h4 : sugar_needed original 6 = 3/4) : 
  sugar_needed original 6 = 0.75 := by
  sorry

#eval sugar_needed ⟨11, 8, 1⟩ 6

end NUMINAMATH_CALUDE_sugar_amount_is_correct_l1137_113732


namespace NUMINAMATH_CALUDE_sufficient_conditions_for_x_squared_less_than_one_l1137_113720

theorem sufficient_conditions_for_x_squared_less_than_one :
  (∀ x : ℝ, (0 < x ∧ x < 1) → x^2 < 1) ∧
  (∀ x : ℝ, (-1 < x ∧ x < 0) → x^2 < 1) ∧
  (∀ x : ℝ, (-1 < x ∧ x < 1) → x^2 < 1) ∧
  (∃ x : ℝ, x < 1 ∧ ¬(x^2 < 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_conditions_for_x_squared_less_than_one_l1137_113720


namespace NUMINAMATH_CALUDE_count_numbers_with_6_or_7_correct_l1137_113774

/-- The number of integers from 1 to 729 (inclusive) in base 9 that contain at least one digit 6 or 7 -/
def count_numbers_with_6_or_7 : ℕ := 386

/-- The total number of integers we're considering -/
def total_numbers : ℕ := 729

/-- The base of the number system we're using -/
def base : ℕ := 9

/-- The number of digits available that are neither 6 nor 7 -/
def digits_without_6_or_7 : ℕ := 7

theorem count_numbers_with_6_or_7_correct :
  count_numbers_with_6_or_7 = total_numbers - digits_without_6_or_7^3 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_with_6_or_7_correct_l1137_113774


namespace NUMINAMATH_CALUDE_expression_simplification_l1137_113700

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (x - 1) / (x^2 - 2*x + 1) / ((x + 1) / (x - 1) + 1) = (Real.sqrt 2 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1137_113700


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l1137_113731

theorem max_sum_of_factors (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 1995 →
  ∀ w x y z : ℕ+, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    w * x * y * z = 1995 →
    a + b + c + d ≥ w + x + y + z :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l1137_113731


namespace NUMINAMATH_CALUDE_decimal_25_equals_base5_100_l1137_113743

/-- Converts a natural number to its base 5 representation --/
def toBaseFive (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBaseFive (n / 5)

/-- Theorem: The decimal number 25 is equivalent to 100₅ in base 5 --/
theorem decimal_25_equals_base5_100 : toBaseFive 25 = [0, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_decimal_25_equals_base5_100_l1137_113743


namespace NUMINAMATH_CALUDE_four_balls_two_boxes_l1137_113766

/-- The number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := 
  (k ^ n) / (Nat.factorial k)

/-- Theorem: There are 8 ways to put 4 distinguishable balls into 2 indistinguishable boxes -/
theorem four_balls_two_boxes : ways_to_put_balls_in_boxes 4 2 = 8 := by
  sorry

#eval ways_to_put_balls_in_boxes 4 2

end NUMINAMATH_CALUDE_four_balls_two_boxes_l1137_113766


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l1137_113735

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - x^2 + x - 2

-- Define the roots
variable (p q r : ℝ)

-- State the theorem
theorem sum_of_cubes_of_roots :
  (f p = 0) → (f q = 0) → (f r = 0) → 
  p ≠ q → q ≠ r → r ≠ p →
  p^3 + q^3 + r^3 = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l1137_113735


namespace NUMINAMATH_CALUDE_roots_greater_than_three_l1137_113799

/-- For a quadratic equation x^2 - 6ax + (2 - 2a + 9a^2) = 0, both roots are greater than 3 
    if and only if a > 11/9 -/
theorem roots_greater_than_three (a : ℝ) : 
  (∀ x : ℝ, x^2 - 6*a*x + (2 - 2*a + 9*a^2) = 0 → x > 3) ↔ a > 11/9 := by
  sorry

end NUMINAMATH_CALUDE_roots_greater_than_three_l1137_113799


namespace NUMINAMATH_CALUDE_x_power_twenty_is_negative_one_l1137_113758

theorem x_power_twenty_is_negative_one (x : ℂ) (h : x + 1/x = Real.sqrt 2) : x^20 = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_power_twenty_is_negative_one_l1137_113758


namespace NUMINAMATH_CALUDE_specific_dumbbell_system_weight_l1137_113780

/-- The total weight of a dumbbell system with three pairs of dumbbells -/
def dumbbell_system_weight (weight1 weight2 weight3 : ℕ) : ℕ :=
  2 * weight1 + 2 * weight2 + 2 * weight3

/-- Theorem: The weight of the specific dumbbell system is 32 lb -/
theorem specific_dumbbell_system_weight :
  dumbbell_system_weight 3 5 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_specific_dumbbell_system_weight_l1137_113780


namespace NUMINAMATH_CALUDE_vegetable_cost_l1137_113754

theorem vegetable_cost (beef_weight : ℝ) (vegetable_weight : ℝ) (total_cost : ℝ) :
  beef_weight = 4 →
  vegetable_weight = 6 →
  total_cost = 36 →
  ∃ (v : ℝ), v * vegetable_weight + 3 * v * beef_weight = total_cost ∧ v = 2 :=
by sorry

end NUMINAMATH_CALUDE_vegetable_cost_l1137_113754


namespace NUMINAMATH_CALUDE_optimal_pole_is_twelve_l1137_113772

/-- Represents the number of intervals in the path -/
def intervals : ℕ := 28

/-- Represents Dodson's walking time for one interval (in minutes) -/
def dodson_walk_time : ℕ := 9

/-- Represents Williams' walking time for one interval (in minutes) -/
def williams_walk_time : ℕ := 11

/-- Represents the riding time on Bolivar for one interval (in minutes) -/
def bolivar_ride_time : ℕ := 3

/-- Calculates Dodson's total travel time given the pole number -/
def dodson_total_time (pole : ℕ) : ℚ :=
  (pole * bolivar_ride_time + (intervals - pole) * dodson_walk_time) / intervals

/-- Calculates Williams' total travel time given the pole number -/
def williams_total_time (pole : ℕ) : ℚ :=
  (pole * williams_walk_time + (intervals - pole) * bolivar_ride_time) / intervals

/-- Theorem stating that the 12th pole is the optimal point to tie Bolivar -/
theorem optimal_pole_is_twelve :
  ∃ (pole : ℕ), pole = 12 ∧
  ∀ (k : ℕ), 1 ≤ k ∧ k ≤ intervals →
    max (dodson_total_time pole) (williams_total_time pole) ≤
    max (dodson_total_time k) (williams_total_time k) :=
by sorry

end NUMINAMATH_CALUDE_optimal_pole_is_twelve_l1137_113772


namespace NUMINAMATH_CALUDE_unique_a_value_l1137_113717

def A (a : ℝ) : Set ℝ := {0, 2, a^2}
def B (a : ℝ) : Set ℝ := {1, a}

theorem unique_a_value : ∃! a : ℝ, A a ∪ B a = {0, 1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1137_113717


namespace NUMINAMATH_CALUDE_eiffel_tower_lower_than_burj_khalifa_l1137_113776

/-- The height of the Eiffel Tower in meters -/
def eiffel_tower_height : ℝ := 324

/-- The height of the Burj Khalifa in meters -/
def burj_khalifa_height : ℝ := 830

/-- The difference in height between the Burj Khalifa and the Eiffel Tower -/
def height_difference : ℝ := burj_khalifa_height - eiffel_tower_height

/-- Theorem stating that the Eiffel Tower is 506 meters lower than the Burj Khalifa -/
theorem eiffel_tower_lower_than_burj_khalifa : 
  height_difference = 506 := by sorry

end NUMINAMATH_CALUDE_eiffel_tower_lower_than_burj_khalifa_l1137_113776


namespace NUMINAMATH_CALUDE_division_subtraction_problem_l1137_113704

theorem division_subtraction_problem : (12 / (2/3)) - 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_division_subtraction_problem_l1137_113704


namespace NUMINAMATH_CALUDE_factorization_equality_l1137_113777

theorem factorization_equality (x : ℝ) : 
  2*x*(x-3) + 3*(x-3) + 5*x^2*(x-3) = (x-3)*(5*x^2 + 2*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1137_113777


namespace NUMINAMATH_CALUDE_malfunctioning_clock_correct_time_l1137_113711

/-- Represents a 12-hour digital clock with a malfunction where '2' is displayed as '5' -/
structure MalfunctioningClock where
  /-- The number of hours in the clock (12) -/
  total_hours : ℕ
  /-- The number of minutes per hour (60) -/
  minutes_per_hour : ℕ
  /-- The number of hours affected by the malfunction -/
  incorrect_hours : ℕ
  /-- The number of minutes per hour affected by the malfunction -/
  incorrect_minutes : ℕ

/-- The fraction of the day a malfunctioning clock shows the correct time -/
def correct_time_fraction (clock : MalfunctioningClock) : ℚ :=
  ((clock.total_hours - clock.incorrect_hours : ℚ) / clock.total_hours) *
  ((clock.minutes_per_hour - clock.incorrect_minutes : ℚ) / clock.minutes_per_hour)

theorem malfunctioning_clock_correct_time :
  ∃ (clock : MalfunctioningClock),
    clock.total_hours = 12 ∧
    clock.minutes_per_hour = 60 ∧
    clock.incorrect_hours = 2 ∧
    clock.incorrect_minutes = 15 ∧
    correct_time_fraction clock = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_malfunctioning_clock_correct_time_l1137_113711


namespace NUMINAMATH_CALUDE_product_first_three_terms_l1137_113738

def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem product_first_three_terms 
  (a : ℕ → ℕ) 
  (h1 : arithmetic_sequence a 2)
  (h2 : a 7 = 20) : 
  a 1 * a 2 * a 3 = 960 := by
sorry

end NUMINAMATH_CALUDE_product_first_three_terms_l1137_113738


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l1137_113710

def line1 (x : ℝ) : ℝ := -x
def line2 (x : ℝ) : ℝ := 2*x - 1

def intersection_point : ℝ × ℝ :=
  let x := 1
  let y := -1
  (x, y)

theorem intersection_in_fourth_quadrant :
  let (x, y) := intersection_point
  x > 0 ∧ y < 0 ∧ line1 x = y ∧ line2 x = y :=
sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l1137_113710


namespace NUMINAMATH_CALUDE_arrangement_count_l1137_113782

/-- The number of ways to arrange 3 male and 3 female students in a row with exactly two female students adjacent -/
def num_arrangements : ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := 6

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 3

theorem arrangement_count :
  (total_students = num_male + num_female) →
  (num_arrangements = 432) := by sorry

end NUMINAMATH_CALUDE_arrangement_count_l1137_113782


namespace NUMINAMATH_CALUDE_total_cost_of_pens_l1137_113769

/-- The cost of a single pen in dollars -/
def cost_per_pen : ℚ := 2

/-- The number of pens -/
def number_of_pens : ℕ := 10

/-- The total cost of pens -/
def total_cost : ℚ := cost_per_pen * number_of_pens

/-- Theorem stating that the total cost of 10 pens is $20 -/
theorem total_cost_of_pens : total_cost = 20 := by sorry

end NUMINAMATH_CALUDE_total_cost_of_pens_l1137_113769


namespace NUMINAMATH_CALUDE_circle_radius_is_4_l1137_113797

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 10*x + y^2 + 4*y + 13 = 0

-- Define the radius of the circle
def circle_radius : ℝ := 4

-- Theorem statement
theorem circle_radius_is_4 :
  ∀ x y : ℝ, circle_equation x y → 
  ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_4_l1137_113797


namespace NUMINAMATH_CALUDE_equation_has_solution_l1137_113729

theorem equation_has_solution (a b c : ℝ) : ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_solution_l1137_113729


namespace NUMINAMATH_CALUDE_six_eight_ten_pythagorean_triple_l1137_113724

/-- A Pythagorean triple is a set of three positive integers (a, b, c) that satisfies a² + b² = c² --/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- The set (6, 8, 10) is a Pythagorean triple --/
theorem six_eight_ten_pythagorean_triple : is_pythagorean_triple 6 8 10 := by
  sorry

end NUMINAMATH_CALUDE_six_eight_ten_pythagorean_triple_l1137_113724


namespace NUMINAMATH_CALUDE_y_share_per_x_rupee_l1137_113713

/-- Given a sum divided among x, y, and z, prove that y gets 9/20 rupees for each rupee x gets. -/
theorem y_share_per_x_rupee (x y z : ℝ) (total : ℝ) (y_share : ℝ) (y_per_x : ℝ) : 
  total = 234 →
  y_share = 54 →
  x + y + z = total →
  y = y_per_x * x →
  z = 0.5 * x →
  y_per_x = 9/20 := by
  sorry

end NUMINAMATH_CALUDE_y_share_per_x_rupee_l1137_113713


namespace NUMINAMATH_CALUDE_dinner_cost_bret_dinner_cost_l1137_113739

theorem dinner_cost (people : ℕ) (main_meal_cost appetizer_cost : ℚ) 
  (appetizers : ℕ) (tip_percentage : ℚ) (rush_fee : ℚ) : ℚ :=
  let main_meals_total := people * main_meal_cost
  let appetizers_total := appetizers * appetizer_cost
  let subtotal := main_meals_total + appetizers_total
  let tip := tip_percentage * subtotal
  let total := subtotal + tip + rush_fee
  total

theorem bret_dinner_cost : 
  dinner_cost 4 12 6 2 (20/100) 5 = 77 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_bret_dinner_cost_l1137_113739


namespace NUMINAMATH_CALUDE_circular_dome_larger_interior_angle_l1137_113753

/-- A circular dome structure constructed from congruent isosceles trapezoids. -/
structure CircularDome where
  /-- The number of trapezoids in the dome -/
  num_trapezoids : ℕ
  /-- The measure of the larger interior angle of each trapezoid in degrees -/
  larger_interior_angle : ℝ

/-- Theorem: In a circular dome constructed from 10 congruent isosceles trapezoids,
    where the non-parallel sides of the trapezoids extend to meet at the center of
    the circle formed by the base of the dome, the measure of the larger interior
    angle of each trapezoid is 81°. -/
theorem circular_dome_larger_interior_angle
  (dome : CircularDome)
  (h₁ : dome.num_trapezoids = 10)
  : dome.larger_interior_angle = 81 := by
  sorry

end NUMINAMATH_CALUDE_circular_dome_larger_interior_angle_l1137_113753


namespace NUMINAMATH_CALUDE_middle_brother_height_l1137_113775

theorem middle_brother_height (h₁ h₂ h₃ : ℝ) :
  h₁ ≤ h₂ ∧ h₂ ≤ h₃ →
  (h₁ + h₂ + h₃) / 3 = 1.74 →
  (h₁ + h₃) / 2 = 1.75 →
  h₂ = 1.72 := by
sorry

end NUMINAMATH_CALUDE_middle_brother_height_l1137_113775


namespace NUMINAMATH_CALUDE_profit_scenario_theorem_l1137_113785

/-- Represents the profit scenarios for Bill's product sales -/
structure ProfitScenarios where
  original_purchase_price : ℝ
  original_profit_rate : ℝ
  second_purchase_discount : ℝ
  second_profit_rate : ℝ
  second_additional_profit : ℝ
  third_purchase_discount : ℝ
  third_profit_rate : ℝ
  third_additional_profit : ℝ

/-- Calculates the selling prices for each scenario given the profit conditions -/
def calculate_selling_prices (s : ProfitScenarios) : ℝ × ℝ × ℝ :=
  let original_selling_price := s.original_purchase_price * (1 + s.original_profit_rate)
  let second_selling_price := original_selling_price + s.second_additional_profit
  let third_selling_price := original_selling_price + s.third_additional_profit
  (original_selling_price, second_selling_price, third_selling_price)

/-- Theorem stating that given the profit conditions, the selling prices are as calculated -/
theorem profit_scenario_theorem (s : ProfitScenarios) 
  (h1 : s.original_profit_rate = 0.1)
  (h2 : s.second_purchase_discount = 0.1)
  (h3 : s.second_profit_rate = 0.3)
  (h4 : s.second_additional_profit = 35)
  (h5 : s.third_purchase_discount = 0.15)
  (h6 : s.third_profit_rate = 0.5)
  (h7 : s.third_additional_profit = 70) :
  calculate_selling_prices s = (550, 585, 620) := by
  sorry

end NUMINAMATH_CALUDE_profit_scenario_theorem_l1137_113785


namespace NUMINAMATH_CALUDE_mixed_doubles_probability_l1137_113727

/-- The number of athletes -/
def total_athletes : ℕ := 6

/-- The number of male athletes -/
def male_athletes : ℕ := 3

/-- The number of female athletes -/
def female_athletes : ℕ := 3

/-- The number of coaches -/
def coaches : ℕ := 3

/-- The number of players each coach selects -/
def players_per_coach : ℕ := 2

/-- The probability of all coaches forming mixed doubles teams -/
def probability_mixed_doubles : ℚ := 2/5

theorem mixed_doubles_probability :
  let total_outcomes := (total_athletes.choose players_per_coach * 
                         (total_athletes - players_per_coach).choose players_per_coach * 
                         (total_athletes - 2*players_per_coach).choose players_per_coach) / coaches.factorial
  let favorable_outcomes := male_athletes.choose 1 * female_athletes.choose 1 * 
                            (male_athletes - 1).choose 1 * (female_athletes - 1).choose 1 * 
                            (male_athletes - 2).choose 1 * (female_athletes - 2).choose 1 * 
                            coaches.factorial
  (favorable_outcomes : ℚ) / total_outcomes = probability_mixed_doubles :=
sorry

end NUMINAMATH_CALUDE_mixed_doubles_probability_l1137_113727


namespace NUMINAMATH_CALUDE_inequality_condition_l1137_113790

theorem inequality_condition (A B C : ℝ) : 
  (∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) ↔ 
  (A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ A^2 + B^2 + C^2 ≤ 2 * (A * B + A * C + B * C)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l1137_113790


namespace NUMINAMATH_CALUDE_toothpaste_duration_l1137_113787

/-- Represents the amount of toothpaste in grams --/
def toothpasteAmount : ℝ := 105

/-- Represents the amount of toothpaste used by Anne's dad per brushing --/
def dadUsage : ℝ := 3

/-- Represents the amount of toothpaste used by Anne's mom per brushing --/
def momUsage : ℝ := 2

/-- Represents the amount of toothpaste used by Anne per brushing --/
def anneUsage : ℝ := 1

/-- Represents the amount of toothpaste used by Anne's brother per brushing --/
def brotherUsage : ℝ := 1

/-- Represents the number of times each family member brushes their teeth per day --/
def brushingsPerDay : ℕ := 3

/-- Theorem stating that the toothpaste will last for 5 days --/
theorem toothpaste_duration : 
  ∃ (days : ℝ), days = 5 ∧ 
  days * (dadUsage + momUsage + anneUsage + brotherUsage) * brushingsPerDay = toothpasteAmount :=
by sorry

end NUMINAMATH_CALUDE_toothpaste_duration_l1137_113787


namespace NUMINAMATH_CALUDE_lanas_roses_l1137_113796

theorem lanas_roses (tulips : ℕ) (used_flowers : ℕ) (extra_flowers : ℕ) :
  tulips = 36 → used_flowers = 70 → extra_flowers = 3 →
  used_flowers + extra_flowers - tulips = 37 := by
  sorry

end NUMINAMATH_CALUDE_lanas_roses_l1137_113796


namespace NUMINAMATH_CALUDE_function_value_2012_l1137_113728

theorem function_value_2012 (m n α₁ α₂ : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hα₁ : α₁ ≠ 0) (hα₂ : α₂ ≠ 0) :
  let f : ℝ → ℝ := λ x => m * Real.sin (π * x + α₁) + n * Real.cos (π * x + α₂)
  f 2011 = 1 → f 2012 = -1 := by
sorry

end NUMINAMATH_CALUDE_function_value_2012_l1137_113728


namespace NUMINAMATH_CALUDE_triangle_construction_theorem_l1137_113755

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Midpoint (A B M : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def OnLine (P : Point) (L : Line) : Prop :=
  L.a * P.x + L.b * P.y + L.c = 0

def AngleBisector (A B C : Point) (L : Line) : Prop :=
  -- This is a simplified definition and may need to be expanded
  OnLine A L ∧ OnLine B L

-- The main theorem
theorem triangle_construction_theorem :
  ∀ (N M : Point) (l : Line),
  ∃ (A B C : Point),
    Midpoint A C N ∧
    Midpoint B C M ∧
    AngleBisector A B C l :=
sorry

end NUMINAMATH_CALUDE_triangle_construction_theorem_l1137_113755


namespace NUMINAMATH_CALUDE_second_number_value_l1137_113764

theorem second_number_value (x y z : ℝ) 
  (sum_eq : x + y + z = 120) 
  (ratio_xy : x / y = 3 / 4) 
  (ratio_yz : y / z = 4 / 7) : 
  y = 34 := by sorry

end NUMINAMATH_CALUDE_second_number_value_l1137_113764


namespace NUMINAMATH_CALUDE_missing_number_is_four_l1137_113702

/-- The structure of the problem -/
structure BoxStructure where
  top_left : ℕ
  top_right : ℕ
  middle_left : ℕ
  middle_right : ℕ
  bottom : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (b : BoxStructure) : Prop :=
  b.middle_left = b.top_left * b.top_right ∧
  b.bottom = b.middle_left * b.middle_right ∧
  b.middle_left = 30 ∧
  b.top_left = 6 ∧
  b.top_right = 5 ∧
  b.bottom = 600

/-- The theorem to prove -/
theorem missing_number_is_four :
  ∀ b : BoxStructure, satisfies_conditions b → b.middle_right = 4 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_is_four_l1137_113702


namespace NUMINAMATH_CALUDE_max_value_is_nine_l1137_113757

def max_value (a b c : ℕ) : ℕ := c * b^a

theorem max_value_is_nine :
  ∃ (a b c : ℕ), a ∈ ({1, 2, 3} : Set ℕ) ∧ b ∈ ({1, 2, 3} : Set ℕ) ∧ c ∈ ({1, 2, 3} : Set ℕ) ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  max_value a b c = 9 ∧
  ∀ (x y z : ℕ), x ∈ ({1, 2, 3} : Set ℕ) → y ∈ ({1, 2, 3} : Set ℕ) → z ∈ ({1, 2, 3} : Set ℕ) →
  x ≠ y → y ≠ z → x ≠ z →
  max_value x y z ≤ 9 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_is_nine_l1137_113757


namespace NUMINAMATH_CALUDE_train_length_l1137_113768

/-- Given a train traveling at 90 kmph and crossing a pole in 4 seconds, its length is 100 meters. -/
theorem train_length (speed_kmph : ℝ) (crossing_time : ℝ) (train_length : ℝ) : 
  speed_kmph = 90 → 
  crossing_time = 4 → 
  train_length = speed_kmph * (1000 / 3600) * crossing_time →
  train_length = 100 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1137_113768


namespace NUMINAMATH_CALUDE_juan_saw_three_bicycles_l1137_113725

/-- The number of bicycles Juan saw -/
def num_bicycles : ℕ := 3

/-- The number of cars Juan saw -/
def num_cars : ℕ := 15

/-- The number of pickup trucks Juan saw -/
def num_pickup_trucks : ℕ := 8

/-- The number of tricycles Juan saw -/
def num_tricycles : ℕ := 1

/-- The total number of tires on all vehicles Juan saw -/
def total_tires : ℕ := 101

/-- The number of tires on a car -/
def tires_per_car : ℕ := 4

/-- The number of tires on a pickup truck -/
def tires_per_pickup : ℕ := 4

/-- The number of tires on a tricycle -/
def tires_per_tricycle : ℕ := 3

/-- The number of tires on a bicycle -/
def tires_per_bicycle : ℕ := 2

theorem juan_saw_three_bicycles :
  num_bicycles * tires_per_bicycle + 
  num_cars * tires_per_car + 
  num_pickup_trucks * tires_per_pickup + 
  num_tricycles * tires_per_tricycle = total_tires :=
by sorry

end NUMINAMATH_CALUDE_juan_saw_three_bicycles_l1137_113725


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l1137_113716

theorem smallest_integer_with_given_remainders : ∃ x : ℕ, 
  (x > 0) ∧ 
  (x % 3 = 2) ∧ 
  (x % 4 = 3) ∧ 
  (x % 5 = 4) ∧ 
  (∀ y : ℕ, y > 0 → y % 3 = 2 → y % 4 = 3 → y % 5 = 4 → x ≤ y) ∧
  (x = 59) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l1137_113716


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_div_2_l1137_113778

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.sin x

theorem derivative_f_at_pi_div_2 :
  deriv f (π / 2) = π := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_div_2_l1137_113778


namespace NUMINAMATH_CALUDE_set_union_problem_l1137_113719

theorem set_union_problem (a b : ℕ) :
  let M : Set ℕ := {3, 2^a}
  let N : Set ℕ := {a, b}
  M ∩ N = {2} → M ∪ N = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l1137_113719


namespace NUMINAMATH_CALUDE_tan_seven_pi_fourth_l1137_113783

theorem tan_seven_pi_fourth : Real.tan (7 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_fourth_l1137_113783


namespace NUMINAMATH_CALUDE_determinant_equality_l1137_113709

theorem determinant_equality (a b c d : ℝ) :
  Matrix.det !![a, b; c, d] = -7 →
  Matrix.det !![2*a + c, b - 2*d; c, 2*d] = -28 + 3*b*c + 2*c*d + 2*d*c := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l1137_113709


namespace NUMINAMATH_CALUDE_johns_expense_ratio_l1137_113781

/-- Proves that the ratio of John's book expenses to money given to Kaylee is 1:1 --/
theorem johns_expense_ratio :
  -- Define the given conditions
  let total_days : ℕ := 30
  let sundays : ℕ := 4
  let daily_earnings : ℕ := 10
  let book_expense : ℕ := 50
  let money_left : ℕ := 160
  
  -- Calculate working days
  let working_days : ℕ := total_days - sundays
  
  -- Calculate total earnings
  let total_earnings : ℕ := working_days * daily_earnings
  
  -- Calculate money given to Kaylee
  let money_to_kaylee : ℕ := total_earnings - book_expense - money_left
  
  -- State the theorem
  book_expense = money_to_kaylee := by
    sorry

end NUMINAMATH_CALUDE_johns_expense_ratio_l1137_113781


namespace NUMINAMATH_CALUDE_clothing_popularity_l1137_113712

/-- Represents the sales of clothing on a given day in July -/
def sales (n : ℕ) : ℕ :=
  if n ≤ 13 then 3 * n else 65 - 2 * n

/-- Represents the cumulative sales up to a given day in July -/
def cumulative_sales (n : ℕ) : ℕ :=
  if n ≤ 13 then (3 + 3 * n) * n / 2 else 273 + (51 - n) * (n - 13)

/-- The day when the clothing becomes popular -/
def popular_start : ℕ := 12

/-- The day when the clothing is no longer popular -/
def popular_end : ℕ := 22

theorem clothing_popularity :
  (∀ n : ℕ, n ≥ popular_start → n ≤ popular_end → cumulative_sales n ≥ 200) ∧
  (∀ n : ℕ, n > popular_end → sales n < 20) ∧
  popular_end - popular_start + 1 = 11 := by sorry

end NUMINAMATH_CALUDE_clothing_popularity_l1137_113712
