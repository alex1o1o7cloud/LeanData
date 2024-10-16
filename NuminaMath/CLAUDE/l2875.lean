import Mathlib

namespace NUMINAMATH_CALUDE_five_lines_sixteen_sections_l2875_287573

/-- The number of sections created by drawing n line segments through a rectangle,
    assuming each new line intersects all previous lines. -/
def max_sections (n : ℕ) : ℕ :=
  if n = 0 then 1 else max_sections (n - 1) + n

/-- The theorem stating that 5 line segments create 16 sections in a rectangle. -/
theorem five_lines_sixteen_sections :
  max_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_five_lines_sixteen_sections_l2875_287573


namespace NUMINAMATH_CALUDE_unique_solution_fraction_equation_l2875_287570

theorem unique_solution_fraction_equation :
  ∃! x : ℝ, (x ≠ 3 ∧ x ≠ 4) ∧ (3 / (x - 3) = 4 / (x - 4)) ∧ x = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_fraction_equation_l2875_287570


namespace NUMINAMATH_CALUDE_john_mary_probability_l2875_287523

-- Define the set of people
inductive Person : Type
| John : Person
| Mary : Person
| Alice : Person
| Bob : Person
| Clara : Person

-- Define the seating arrangement
structure Seating :=
(long_side1 : Person × Person)
(long_side2 : Person × Person)
(short_side1 : Person)
(short_side2 : Person)

-- Define a function to check if John and Mary are seated together on a longer side
def john_and_mary_together (s : Seating) : Prop :=
  (s.long_side1 = (Person.John, Person.Mary) ∨ s.long_side1 = (Person.Mary, Person.John)) ∨
  (s.long_side2 = (Person.John, Person.Mary) ∨ s.long_side2 = (Person.Mary, Person.John))

-- Define the set of all possible seating arrangements
def all_seatings : Set Seating := sorry

-- Define the probability measure on the set of all seating arrangements
def prob : Set Seating → ℝ := sorry

-- The main theorem
theorem john_mary_probability :
  prob {s ∈ all_seatings | john_and_mary_together s} = 1/4 := by sorry

end NUMINAMATH_CALUDE_john_mary_probability_l2875_287523


namespace NUMINAMATH_CALUDE_total_goals_scored_l2875_287558

def soccer_match (team_a_first_half : ℕ) (team_b_second_half : ℕ) : Prop :=
  let team_b_first_half := team_a_first_half / 2
  let team_a_second_half := team_b_second_half - 2
  let team_a_total := team_a_first_half + team_a_second_half
  let team_b_total := team_b_first_half + team_b_second_half
  team_a_total + team_b_total = 26

theorem total_goals_scored :
  soccer_match 8 8 := by sorry

end NUMINAMATH_CALUDE_total_goals_scored_l2875_287558


namespace NUMINAMATH_CALUDE_projection_matrix_values_l2875_287513

def is_projection_matrix (Q : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  Q ^ 2 = Q

theorem projection_matrix_values :
  ∀ (x y : ℝ),
  let Q : Matrix (Fin 2) (Fin 2) ℝ := !![x, 1/5; y, 4/5]
  is_projection_matrix Q ↔ x = 1 ∧ y = 0 := by
sorry


end NUMINAMATH_CALUDE_projection_matrix_values_l2875_287513


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2875_287541

theorem opposite_of_negative_2023 : -(-(2023 : ℤ)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2875_287541


namespace NUMINAMATH_CALUDE_product_of_exponents_l2875_287543

theorem product_of_exponents (p r s : ℕ) : 
  3^p + 3^5 = 252 → 
  2^r + 58 = 122 → 
  5^3 * 6^s = 117000 → 
  p * r * s = 36 := by
  sorry

end NUMINAMATH_CALUDE_product_of_exponents_l2875_287543


namespace NUMINAMATH_CALUDE_rational_square_property_l2875_287578

theorem rational_square_property (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ (z : ℚ), 1 - x*y = z^2 := by
  sorry

end NUMINAMATH_CALUDE_rational_square_property_l2875_287578


namespace NUMINAMATH_CALUDE_modular_multiplication_l2875_287547

theorem modular_multiplication (m : ℕ) : 
  0 ≤ m ∧ m < 25 ∧ m ≡ (66 * 77 * 88) [ZMOD 25] → m = 16 := by
  sorry

end NUMINAMATH_CALUDE_modular_multiplication_l2875_287547


namespace NUMINAMATH_CALUDE_expected_value_is_1866_l2875_287501

/-- Represents the available keys on the calculator -/
inductive Key
| One
| Two
| Three
| Plus
| Minus

/-- A sequence of 5 keystrokes -/
def Sequence := Vector Key 5

/-- Evaluates a sequence of keystrokes according to the problem rules -/
def evaluate : Sequence → ℤ := sorry

/-- The probability of pressing any specific key -/
def keyProbability : ℚ := 1 / 5

/-- The expected value of the result after evaluating a random sequence -/
def expectedValue : ℚ := sorry

/-- Theorem stating that the expected value is 1866 -/
theorem expected_value_is_1866 : expectedValue = 1866 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_1866_l2875_287501


namespace NUMINAMATH_CALUDE_gum_pieces_per_package_l2875_287594

/-- The number of packages of gum Robin has -/
def num_packages : ℕ := 5

/-- The number of extra pieces of gum Robin has -/
def extra_pieces : ℕ := 6

/-- The total number of pieces of gum Robin has -/
def total_pieces : ℕ := 41

/-- The number of pieces of gum in each package -/
def pieces_per_package : ℕ := (total_pieces - extra_pieces) / num_packages

theorem gum_pieces_per_package : pieces_per_package = 7 := by
  sorry

end NUMINAMATH_CALUDE_gum_pieces_per_package_l2875_287594


namespace NUMINAMATH_CALUDE_domain_f_minus_one_l2875_287568

-- Define a real-valued function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_plus_one : Set ℝ := Set.Icc (-2) 3

-- Theorem statement
theorem domain_f_minus_one (h : ∀ x, f (x + 1) ∈ domain_f_plus_one ↔ x ∈ domain_f_plus_one) :
  ∀ x, f (x - 1) ∈ Set.Icc 0 5 ↔ x ∈ Set.Icc 0 5 :=
sorry

end NUMINAMATH_CALUDE_domain_f_minus_one_l2875_287568


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2875_287521

theorem parallel_vectors_magnitude (k : ℝ) : 
  let a : Fin 2 → ℝ := ![(-1), 2]
  let b : Fin 2 → ℝ := ![2, k]
  (∃ (c : ℝ), a = c • b) →
  ‖2 • a - b‖ = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2875_287521


namespace NUMINAMATH_CALUDE_hollow_cube_side_length_l2875_287503

/-- The number of cubes required to construct a hollow cube -/
def hollow_cube_cubes (n : ℕ) : ℕ := n^3 - (n-2)^3

/-- Theorem: A hollow cube made of 98 unit cubes has a side length of 5 -/
theorem hollow_cube_side_length :
  ∃ (n : ℕ), n > 0 ∧ hollow_cube_cubes n = 98 → n = 5 := by
sorry

end NUMINAMATH_CALUDE_hollow_cube_side_length_l2875_287503


namespace NUMINAMATH_CALUDE_shaded_square_fraction_l2875_287535

theorem shaded_square_fraction :
  let large_square_side : ℝ := 6
  let small_square_side : ℝ := Real.sqrt 2
  let large_square_area : ℝ := large_square_side ^ 2
  let small_square_area : ℝ := small_square_side ^ 2
  (small_square_area / large_square_area) = (1 : ℝ) / 18 := by sorry

end NUMINAMATH_CALUDE_shaded_square_fraction_l2875_287535


namespace NUMINAMATH_CALUDE_solution_range_l2875_287512

theorem solution_range (a : ℝ) : 
  (∀ x : ℝ, x < 3 → (a - 1) * x < a + 3) ↔ (1 ≤ a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l2875_287512


namespace NUMINAMATH_CALUDE_homework_time_calculation_l2875_287552

/-- The time Max spent on biology homework in minutes -/
def biology_time : ℝ := 24

/-- The time Max spent on history homework in minutes -/
def history_time : ℝ := 1.5 * biology_time

/-- The time Max spent on chemistry homework in minutes -/
def chemistry_time : ℝ := biology_time * 0.7

/-- The time Max spent on English homework in minutes -/
def english_time : ℝ := 2 * (history_time + chemistry_time)

/-- The time Max spent on geography homework in minutes -/
def geography_time : ℝ := 3 * history_time + 0.75 * english_time

/-- The total time Max spent on homework in minutes -/
def total_homework_time : ℝ := biology_time + history_time + chemistry_time + english_time + geography_time

theorem homework_time_calculation :
  total_homework_time = 369.6 := by sorry

end NUMINAMATH_CALUDE_homework_time_calculation_l2875_287552


namespace NUMINAMATH_CALUDE_sequence_convergence_and_general_term_l2875_287574

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The sequence a_n -/
noncomputable def a (x y : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => y
  | n + 2 => (a x y (n + 1) * a x y n + 1) / (a x y (n + 1) + a x y n)

/-- The general term formula for a_n when n ≥ 2 -/
noncomputable def a_general_term (x y : ℝ) (n : ℕ) : ℝ :=
  let num := 2 * ((y - 1) / (y + 1)) ^ (fib (n - 1)) * ((x - 1) / (x + 1)) ^ (fib (n - 2))
  let den := 1 - ((y - 1) / (y + 1)) ^ (fib (n - 1)) * ((x - 1) / (x + 1)) ^ (fib (n - 2))
  num / den - 1

theorem sequence_convergence_and_general_term (x y : ℝ) :
  (∃ n₀ : ℕ+, ∀ n ≥ n₀, a x y n = 1 ∨ a x y n = -1) ↔
    ((x = 1 ∧ y ≠ -1) ∨ (x = -1 ∧ y ≠ 1) ∨ (y = 1 ∧ x ≠ -1) ∨ (y = -1 ∧ x ≠ 1)) ∧
  ∀ n ≥ 2, a x y n = a_general_term x y n :=
by sorry

end NUMINAMATH_CALUDE_sequence_convergence_and_general_term_l2875_287574


namespace NUMINAMATH_CALUDE_carpet_cost_l2875_287504

/-- Calculates the total cost of carpeting a rectangular floor with square carpet tiles -/
theorem carpet_cost (floor_length floor_width carpet_side_length carpet_cost : ℝ) :
  floor_length = 6 ∧ 
  floor_width = 10 ∧ 
  carpet_side_length = 2 ∧ 
  carpet_cost = 15 →
  (floor_length * floor_width) / (carpet_side_length * carpet_side_length) * carpet_cost = 225 := by
  sorry

#check carpet_cost

end NUMINAMATH_CALUDE_carpet_cost_l2875_287504


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l2875_287557

theorem smallest_number_satisfying_conditions : ∃ n : ℕ, n > 0 ∧ 
  n % 6 = 2 ∧ n % 8 = 3 ∧ n % 9 = 4 ∧ 
  ∀ m : ℕ, m > 0 → m % 6 = 2 → m % 8 = 3 → m % 9 = 4 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l2875_287557


namespace NUMINAMATH_CALUDE_max_abs_f_value_l2875_287576

-- Define the band region type
def band_region (k l : ℝ) (y : ℝ) : Prop := k ≤ y ∧ y ≤ l

-- Define the quadratic function
variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem max_abs_f_value :
  ∀ a b c : ℝ,
  (band_region 0 4 (f a b c (-2) + 2)) ∧
  (band_region 0 4 (f a b c 0 + 2)) ∧
  (band_region 0 4 (f a b c 2 + 2)) →
  (∀ t : ℝ, band_region (-1) 3 (t + 1) → |f a b c t| ≤ 5/2) ∧
  (∃ t : ℝ, band_region (-1) 3 (t + 1) ∧ |f a b c t| = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_max_abs_f_value_l2875_287576


namespace NUMINAMATH_CALUDE_average_increase_calculation_l2875_287545

theorem average_increase_calculation (current_matches : ℕ) (current_average : ℚ) (next_match_score : ℕ) : 
  current_matches = 10 →
  current_average = 34 →
  next_match_score = 78 →
  (current_matches + 1) * (current_average + (next_match_score - current_matches * current_average) / (current_matches + 1)) = 
  current_matches * current_average + next_match_score →
  (next_match_score - current_matches * current_average) / (current_matches + 1) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_average_increase_calculation_l2875_287545


namespace NUMINAMATH_CALUDE_gcd_8008_11011_l2875_287598

theorem gcd_8008_11011 : Nat.gcd 8008 11011 = 1001 := by sorry

end NUMINAMATH_CALUDE_gcd_8008_11011_l2875_287598


namespace NUMINAMATH_CALUDE_bicycle_time_calculation_l2875_287553

def total_distance : ℝ := 20
def bicycle_speed : ℝ := 30
def running_speed : ℝ := 8
def total_time : ℝ := 117

theorem bicycle_time_calculation (t : ℝ) 
  (h1 : t ≥ 0) 
  (h2 : t ≤ total_time) 
  (h3 : (t / 60) * bicycle_speed + ((total_time - t) / 60) * running_speed = total_distance) : 
  t = 12 := by sorry

end NUMINAMATH_CALUDE_bicycle_time_calculation_l2875_287553


namespace NUMINAMATH_CALUDE_jill_net_salary_l2875_287584

/-- Represents Jill's financial situation --/
structure JillFinances where
  net_salary : ℝ
  discretionary_income : ℝ
  vacation_fund_percent : ℝ
  savings_percent : ℝ
  socializing_percent : ℝ
  remaining_amount : ℝ

/-- Theorem stating Jill's net monthly salary given her financial conditions --/
theorem jill_net_salary (j : JillFinances) 
  (h1 : j.discretionary_income = j.net_salary / 5)
  (h2 : j.vacation_fund_percent = 0.3)
  (h3 : j.savings_percent = 0.2)
  (h4 : j.socializing_percent = 0.35)
  (h5 : j.remaining_amount = 108)
  (h6 : (1 - (j.vacation_fund_percent + j.savings_percent + j.socializing_percent)) * j.discretionary_income = j.remaining_amount) :
  j.net_salary = 3600 := by
  sorry

#check jill_net_salary

end NUMINAMATH_CALUDE_jill_net_salary_l2875_287584


namespace NUMINAMATH_CALUDE_intersection_vector_sum_l2875_287538

noncomputable def f (x : ℝ) : ℝ := (2 * Real.cos x ^ 2 + 1) / Real.log ((2 + x) / (2 - x))

theorem intersection_vector_sum (a : ℝ) (h_a : a ≠ 0) :
  ∃ (A B : ℝ × ℝ), 
    (∀ x : ℝ, a * x - (f x) = 0 → x = A.1 ∨ x = B.1) →
    (A ≠ B) →
    (∀ m n : ℝ, 
      (A.1 - m, A.2 - n) + (B.1 - m, B.2 - n) = (m - 6, n) →
      m + n = 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_vector_sum_l2875_287538


namespace NUMINAMATH_CALUDE_ava_apple_trees_l2875_287559

theorem ava_apple_trees (lily_trees : ℕ) : 
  (lily_trees + 3) + lily_trees = 15 → (lily_trees + 3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ava_apple_trees_l2875_287559


namespace NUMINAMATH_CALUDE_reggie_has_70_marbles_l2875_287596

/-- Calculates the number of marbles Reggie has after playing a series of games -/
def reggies_marbles (total_games : ℕ) (lost_games : ℕ) (marbles_per_game : ℕ) : ℕ :=
  (total_games - lost_games) * marbles_per_game - lost_games * marbles_per_game

/-- Proves that Reggie has 70 marbles after playing 9 games, losing 1, with 10 marbles bet per game -/
theorem reggie_has_70_marbles :
  reggies_marbles 9 1 10 = 70 := by
  sorry

#eval reggies_marbles 9 1 10

end NUMINAMATH_CALUDE_reggie_has_70_marbles_l2875_287596


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2875_287587

/-- Given a rectangular frame made with 240 cm of wire, where the ratio of
    length:width:height is 3:2:1, prove that the dimensions are 30 cm, 20 cm,
    and 10 cm respectively. -/
theorem rectangle_dimensions (total_wire : ℝ) (length width height : ℝ)
    (h1 : total_wire = 240)
    (h2 : length + width + height = total_wire / 4)
    (h3 : length = 3 * height)
    (h4 : width = 2 * height) :
    length = 30 ∧ width = 20 ∧ height = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2875_287587


namespace NUMINAMATH_CALUDE_pardee_road_length_is_12000_l2875_287591

/-- The length of Pardee Road in meters, given the conditions of the problem -/
def pardee_road_length : ℕ :=
  let telegraph_road_km : ℕ := 162
  let difference_km : ℕ := 150
  let meters_per_km : ℕ := 1000
  (telegraph_road_km - difference_km) * meters_per_km

theorem pardee_road_length_is_12000 : pardee_road_length = 12000 := by
  sorry

end NUMINAMATH_CALUDE_pardee_road_length_is_12000_l2875_287591


namespace NUMINAMATH_CALUDE_diophantine_equation_prime_divisor_l2875_287506

theorem diophantine_equation_prime_divisor (b : ℕ+) (h : Nat.gcd b.val 6 = 1) :
  (∃ (x y : ℕ+), (1 : ℚ) / x.val + (1 : ℚ) / y.val = (3 : ℚ) / b.val) ↔
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ b.val ∧ ∃ (k : ℕ), p = 6 * k - 1 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_prime_divisor_l2875_287506


namespace NUMINAMATH_CALUDE_sum_of_squares_implies_sum_l2875_287537

theorem sum_of_squares_implies_sum : ∀ (a b c : ℝ), 
  (2*a - 6)^2 + (3*b - 9)^2 + (4*c - 12)^2 = 0 → a + 2*b + 3*c = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_implies_sum_l2875_287537


namespace NUMINAMATH_CALUDE_special_triangle_bc_length_l2875_287579

/-- A triangle with special properties -/
structure SpecialTriangle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Length of side AB is 1
  ab_length : dist A B = 1
  -- Length of side AC is 2
  ac_length : dist A C = 2
  -- Median from A to BC has same length as BC
  median_eq_bc : dist A ((B + C) / 2) = dist B C

/-- The length of BC in a SpecialTriangle is √2 -/
theorem special_triangle_bc_length (t : SpecialTriangle) : dist t.B t.C = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_bc_length_l2875_287579


namespace NUMINAMATH_CALUDE_cone_height_l2875_287569

theorem cone_height (r : Real) (h : Real) :
  (3 : Real) * (2 * Real.pi / 3) = 2 * Real.pi * r →
  h ^ 2 + r ^ 2 = 3 ^ 2 →
  h = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l2875_287569


namespace NUMINAMATH_CALUDE_tan_675_degrees_l2875_287528

theorem tan_675_degrees : Real.tan (675 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_675_degrees_l2875_287528


namespace NUMINAMATH_CALUDE_parabola_vertex_l2875_287597

/-- Define a parabola with equation y = (x+2)^2 + 3 -/
def parabola (x : ℝ) : ℝ := (x + 2)^2 + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-2, 3)

/-- Theorem: The vertex of the parabola y = (x+2)^2 + 3 is (-2, 3) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2875_287597


namespace NUMINAMATH_CALUDE_complex_absolute_value_l2875_287575

theorem complex_absolute_value (z : ℂ) : z = 2 / (1 - Complex.I * Real.sqrt 3) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l2875_287575


namespace NUMINAMATH_CALUDE_customers_before_rush_count_l2875_287567

/-- The number of customers who didn't leave a tip -/
def no_tip : ℕ := 49

/-- The number of customers who left a tip -/
def left_tip : ℕ := 2

/-- The number of additional customers during lunch rush -/
def additional_customers : ℕ := 12

/-- The total number of customers after the lunch rush -/
def total_after_rush : ℕ := no_tip + left_tip

/-- The number of customers before the lunch rush -/
def customers_before_rush : ℕ := total_after_rush - additional_customers

theorem customers_before_rush_count : customers_before_rush = 39 := by
  sorry

end NUMINAMATH_CALUDE_customers_before_rush_count_l2875_287567


namespace NUMINAMATH_CALUDE_intersection_one_element_l2875_287500

theorem intersection_one_element (a : ℝ) : 
  let A : Set ℝ := {1, a, 5}
  let B : Set ℝ := {2, a^2 + 1}
  (∃! x, x ∈ A ∩ B) → (a = 0 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_one_element_l2875_287500


namespace NUMINAMATH_CALUDE_exists_b_for_even_f_l2875_287561

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = 2x^2 - bx where b is a real number -/
def f (b : ℝ) : ℝ → ℝ := λ x ↦ 2 * x^2 - b * x

/-- There exists a real number b such that f(x) = 2x^2 - bx is an even function -/
theorem exists_b_for_even_f : ∃ b : ℝ, IsEven (f b) := by
  sorry

end NUMINAMATH_CALUDE_exists_b_for_even_f_l2875_287561


namespace NUMINAMATH_CALUDE_midpoint_movement_l2875_287542

/-- Given two points A and B with midpoint M, prove the new midpoint M' and distance between M and M' after moving A and B -/
theorem midpoint_movement (a b c d m n : ℝ) :
  m = (a + c) / 2 →
  n = (b + d) / 2 →
  let m' := (a + 4 + c - 15) / 2
  let n' := (b + 12 + d - 5) / 2
  (m' = m - 11 / 2 ∧ n' = n + 7 / 2) ∧
  Real.sqrt ((m' - m) ^ 2 + (n' - n) ^ 2) = Real.sqrt 42.5 := by
  sorry


end NUMINAMATH_CALUDE_midpoint_movement_l2875_287542


namespace NUMINAMATH_CALUDE_base_b_is_seven_l2875_287515

/-- Given that in base b, the square of 22_b is 514_b, prove that b = 7 -/
theorem base_b_is_seven (b : ℕ) (h : b > 1) : 
  (2 * b + 2)^2 = 5 * b^2 + b + 4 → b = 7 := by
  sorry

end NUMINAMATH_CALUDE_base_b_is_seven_l2875_287515


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l2875_287532

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l2875_287532


namespace NUMINAMATH_CALUDE_distance_after_three_minutes_l2875_287556

/-- The distance between two vehicles after a given time -/
def distance_between_vehicles (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

theorem distance_after_three_minutes :
  let truck_speed : ℝ := 65
  let car_speed : ℝ := 85
  let time_in_hours : ℝ := 3 / 60
  distance_between_vehicles truck_speed car_speed time_in_hours = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_three_minutes_l2875_287556


namespace NUMINAMATH_CALUDE_min_value_constraint_l2875_287531

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + a^2 - 2 * a + 2

-- Define the theorem
theorem min_value_constraint (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, f a x ≥ 3) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 3) ↔ 
  a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_constraint_l2875_287531


namespace NUMINAMATH_CALUDE_max_product_sum_300_l2875_287529

theorem max_product_sum_300 : 
  (∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500) ∧ 
  (∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l2875_287529


namespace NUMINAMATH_CALUDE_inverse_of_A_l2875_287588

def A : Matrix (Fin 3) (Fin 3) ℚ := !![1, 2, 3; 0, -1, 2; 3, 0, 7]

def A_inverse : Matrix (Fin 3) (Fin 3) ℚ := !![-1/2, -1, 1/2; 3/7, -1/7, -1/7; 3/14, 3/7, -1/14]

theorem inverse_of_A : A⁻¹ = A_inverse := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l2875_287588


namespace NUMINAMATH_CALUDE_greatest_n_for_perfect_square_T_l2875_287549

def g (x : ℕ) : ℕ := 
  if x % 2 = 1 then 1 else 0

def T (n : ℕ) : ℕ := 2^n

def is_perfect_square (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y = x

theorem greatest_n_for_perfect_square_T : 
  (∃ n : ℕ, n < 500 ∧ is_perfect_square (T n) ∧ 
    ∀ m : ℕ, m < 500 → is_perfect_square (T m) → m ≤ n) →
  (∃ n : ℕ, n = 498 ∧ n < 500 ∧ is_perfect_square (T n) ∧ 
    ∀ m : ℕ, m < 500 → is_perfect_square (T m) → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_n_for_perfect_square_T_l2875_287549


namespace NUMINAMATH_CALUDE_hall_breadth_proof_l2875_287533

/-- Given a rectangular hall and stones with specified dimensions, 
    prove that the breadth of the hall is 15 meters. -/
theorem hall_breadth_proof (hall_length : ℝ) (stone_length : ℝ) (stone_width : ℝ) 
                            (num_stones : ℕ) :
  hall_length = 36 →
  stone_length = 0.4 →
  stone_width = 0.5 →
  num_stones = 2700 →
  hall_length * (num_stones * stone_length * stone_width / hall_length) = 15 := by
  sorry

#check hall_breadth_proof

end NUMINAMATH_CALUDE_hall_breadth_proof_l2875_287533


namespace NUMINAMATH_CALUDE_line_intersection_theorem_l2875_287527

/-- A line y = mx + b intersecting a circle and a hyperbola -/
structure LineIntersection where
  m : ℝ
  b : ℝ
  h_m : |m| < 1
  h_b : |b| < 1

/-- Points of intersection -/
structure IntersectionPoints (l : LineIntersection) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  h_circle_P : P.1^2 + P.2^2 = 1 ∧ P.2 = l.m * P.1 + l.b
  h_circle_Q : Q.1^2 + Q.2^2 = 1 ∧ Q.2 = l.m * Q.1 + l.b
  h_hyperbola_R : R.1^2 - R.2^2 = 1 ∧ R.2 = l.m * R.1 + l.b
  h_hyperbola_S : S.1^2 - S.2^2 = 1 ∧ S.2 = l.m * S.1 + l.b
  h_trisect : dist P R = dist P Q ∧ dist Q S = dist P Q

/-- The main theorem -/
theorem line_intersection_theorem (l : LineIntersection) (p : IntersectionPoints l) :
  (l.m = 0 ∧ l.b^2 = 4/5) ∨ (l.b = 0 ∧ l.m^2 = 4/5) := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_theorem_l2875_287527


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2875_287544

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 8) * (Real.sqrt 6 / Real.sqrt 9) = Real.sqrt 35 / 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2875_287544


namespace NUMINAMATH_CALUDE_max_volume_at_one_sixth_l2875_287502

/-- The volume of an open-topped box made from a square sheet of cardboard --/
def boxVolume (a x : ℝ) : ℝ := (a - 2*x)^2 * x

/-- The theorem stating that the volume is maximized when the cutout side length is a/6 --/
theorem max_volume_at_one_sixth (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), x > 0 ∧ x < a/2 ∧
  ∀ (y : ℝ), y > 0 → y < a/2 → boxVolume a x ≥ boxVolume a y ∧
  x = a/6 :=
sorry

end NUMINAMATH_CALUDE_max_volume_at_one_sixth_l2875_287502


namespace NUMINAMATH_CALUDE_cole_trip_time_l2875_287522

/-- Proves that given a round trip where the outbound journey is at 75 km/h,
    the return journey is at 105 km/h, and the total trip time is 4 hours,
    the time taken for the outbound journey is 140 minutes. -/
theorem cole_trip_time (distance : ℝ) :
  distance / 75 + distance / 105 = 4 →
  distance / 75 * 60 = 140 := by
sorry

end NUMINAMATH_CALUDE_cole_trip_time_l2875_287522


namespace NUMINAMATH_CALUDE_lydias_current_age_l2875_287572

/-- Represents the time it takes for an apple tree to bear fruit -/
def apple_tree_fruit_time : ℕ := 7

/-- Represents Lydia's age when she planted the tree -/
def planting_age : ℕ := 4

/-- Represents Lydia's age when she can eat an apple from her tree for the first time -/
def first_apple_age : ℕ := 11

/-- Represents Lydia's current age -/
def current_age : ℕ := 11

theorem lydias_current_age :
  current_age = first_apple_age ∧
  current_age = planting_age + apple_tree_fruit_time :=
by sorry

end NUMINAMATH_CALUDE_lydias_current_age_l2875_287572


namespace NUMINAMATH_CALUDE_parameterization_valid_iff_l2875_287551

/-- A parameterization of a line is represented by an initial point and a direction vector -/
structure Parameterization where
  x₀ : ℝ
  y₀ : ℝ
  dx : ℝ
  dy : ℝ

/-- The line y = 2x - 4 -/
def line (x : ℝ) : ℝ := 2 * x - 4

/-- A parameterization is valid for the line y = 2x - 4 -/
def is_valid_parameterization (p : Parameterization) : Prop :=
  line p.x₀ = p.y₀ ∧ ∃ (t : ℝ), p.dx = t * 1 ∧ p.dy = t * 2

/-- Theorem: A parameterization is valid if and only if it satisfies the conditions -/
theorem parameterization_valid_iff (p : Parameterization) :
  is_valid_parameterization p ↔ 
  (line p.x₀ = p.y₀ ∧ ∃ (t : ℝ), p.dx = t * 1 ∧ p.dy = t * 2) :=
by sorry

end NUMINAMATH_CALUDE_parameterization_valid_iff_l2875_287551


namespace NUMINAMATH_CALUDE_power_of_three_mod_five_l2875_287530

theorem power_of_three_mod_five : 3^304 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_five_l2875_287530


namespace NUMINAMATH_CALUDE_prob_blue_or_green_with_replacement_l2875_287525

def total_balls : ℕ := 15
def blue_balls : ℕ := 8
def green_balls : ℕ := 2

def prob_blue : ℚ := blue_balls / total_balls
def prob_green : ℚ := green_balls / total_balls

def prob_two_blue : ℚ := prob_blue * prob_blue
def prob_two_green : ℚ := prob_green * prob_green

theorem prob_blue_or_green_with_replacement :
  prob_two_blue + prob_two_green = 68 / 225 := by
  sorry

end NUMINAMATH_CALUDE_prob_blue_or_green_with_replacement_l2875_287525


namespace NUMINAMATH_CALUDE_x_bounds_and_sqrt2_inequality_l2875_287514

theorem x_bounds_and_sqrt2_inequality :
  ∃ x : ℝ,
    (x = (x^2 + 1) / 198) ∧
    (1/198 < x) ∧
    (x < 197.99494949) ∧
    (Real.sqrt 2 < 1.41421356) := by
  sorry

end NUMINAMATH_CALUDE_x_bounds_and_sqrt2_inequality_l2875_287514


namespace NUMINAMATH_CALUDE_power_three_2023_mod_10_l2875_287595

theorem power_three_2023_mod_10 : 3^2023 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_three_2023_mod_10_l2875_287595


namespace NUMINAMATH_CALUDE_solve_for_a_l2875_287539

theorem solve_for_a : ∃ (a : ℝ), 
  let A : Set ℝ := {2, 3, a^2 + 2*a - 3}
  let B : Set ℝ := {|a + 3|, 2}
  5 ∈ A ∧ 5 ∉ B ∧ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2875_287539


namespace NUMINAMATH_CALUDE_no_inscribed_sphere_after_truncation_l2875_287536

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  vertices : Set (ℝ × ℝ × ℝ)
  faces : Set (Set (ℝ × ℝ × ℝ))
  is_convex : Bool
  vertex_count : ℕ
  face_count : ℕ
  vertex_ge_face : vertex_count ≥ face_count

/-- Truncation operation on a convex polyhedron -/
def truncate (P : ConvexPolyhedron) : ConvexPolyhedron :=
  sorry

/-- A sphere -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a sphere is inscribed in a polyhedron -/
def is_inscribed (S : Sphere) (P : ConvexPolyhedron) : Prop :=
  sorry

/-- Theorem stating that a truncated convex polyhedron cannot have an inscribed sphere -/
theorem no_inscribed_sphere_after_truncation (P : ConvexPolyhedron) :
  ¬ ∃ (S : Sphere), is_inscribed S (truncate P) :=
sorry

end NUMINAMATH_CALUDE_no_inscribed_sphere_after_truncation_l2875_287536


namespace NUMINAMATH_CALUDE_instagram_followers_after_year_l2875_287517

/-- Calculates the final number of followers for an Instagram influencer after a year --/
theorem instagram_followers_after_year 
  (initial_followers : ℕ) 
  (new_followers_per_day : ℕ) 
  (days_in_year : ℕ) 
  (unfollowers : ℕ) 
  (h1 : initial_followers = 100000)
  (h2 : new_followers_per_day = 1000)
  (h3 : days_in_year = 365)
  (h4 : unfollowers = 20000) :
  initial_followers + (new_followers_per_day * days_in_year) - unfollowers = 445000 :=
by sorry

end NUMINAMATH_CALUDE_instagram_followers_after_year_l2875_287517


namespace NUMINAMATH_CALUDE_line_inclination_is_30_degrees_l2875_287505

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 2 = 0

-- Define the angle of inclination
def angle_of_inclination (α : ℝ) : Prop := 
  ∃ (k : ℝ), k = Real.sqrt 3 / 3 ∧ k = Real.tan α

-- Theorem statement
theorem line_inclination_is_30_degrees : 
  ∃ (α : ℝ), angle_of_inclination α ∧ α = π / 6 :=
sorry

end NUMINAMATH_CALUDE_line_inclination_is_30_degrees_l2875_287505


namespace NUMINAMATH_CALUDE_abs_minus_self_nonnegative_l2875_287509

theorem abs_minus_self_nonnegative (x : ℝ) : |x| - x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_minus_self_nonnegative_l2875_287509


namespace NUMINAMATH_CALUDE_final_peanut_count_l2875_287563

def peanut_problem (initial_peanuts : ℕ) (mary_adds : ℕ) (john_takes : ℕ) (friends : ℕ) : ℕ :=
  initial_peanuts + mary_adds - john_takes

theorem final_peanut_count :
  peanut_problem 4 4 2 2 = 6 := by sorry

end NUMINAMATH_CALUDE_final_peanut_count_l2875_287563


namespace NUMINAMATH_CALUDE_product_equality_implies_sum_l2875_287546

theorem product_equality_implies_sum (g h : ℚ) : 
  (∀ d : ℚ, (8*d^2 - 5*d + g) * (2*d^2 + h*d - 9) = 16*d^4 + 21*d^3 - 73*d^2 - 41*d + 45) →
  g + h = -82/25 := by
sorry

end NUMINAMATH_CALUDE_product_equality_implies_sum_l2875_287546


namespace NUMINAMATH_CALUDE_total_distance_calculation_l2875_287520

/-- Represents the problem of calculating the total distance traveled by a person
    given specific conditions. -/
theorem total_distance_calculation (d : ℝ) : 
  (d / 6 + d / 12 + d / 18 + d / 24 + d / 30 = 17 / 60) → 
  (5 * d = 425 / 114) := by
  sorry

#check total_distance_calculation

end NUMINAMATH_CALUDE_total_distance_calculation_l2875_287520


namespace NUMINAMATH_CALUDE_three_solutions_when_a_is_9_l2875_287508

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - a * x^2 + 6

-- Theorem statement
theorem three_solutions_when_a_is_9 :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f 9 x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_three_solutions_when_a_is_9_l2875_287508


namespace NUMINAMATH_CALUDE_angle_c_not_five_sixths_pi_l2875_287554

theorem angle_c_not_five_sixths_pi (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_eq1 : 3 * Real.sin A + 4 * Real.cos B = 6) 
  (h_eq2 : 3 * Real.cos A + 4 * Real.sin B = 1) : 
  C ≠ 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_not_five_sixths_pi_l2875_287554


namespace NUMINAMATH_CALUDE_least_n_for_fraction_inequality_l2875_287599

theorem least_n_for_fraction_inequality : 
  ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, k > 0 → k < n → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ 1 / 15) ∧
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < 1 / 15) ∧ n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_least_n_for_fraction_inequality_l2875_287599


namespace NUMINAMATH_CALUDE_coffee_machine_price_l2875_287581

/-- The original price of a coffee machine given certain conditions -/
theorem coffee_machine_price (discount : ℕ) (payback_days : ℕ) (old_daily_cost new_daily_cost : ℕ) : 
  discount = 20 →
  payback_days = 36 →
  old_daily_cost = 8 →
  new_daily_cost = 3 →
  (payback_days * (old_daily_cost - new_daily_cost)) + discount = 200 :=
by sorry

end NUMINAMATH_CALUDE_coffee_machine_price_l2875_287581


namespace NUMINAMATH_CALUDE_ball_ratio_l2875_287583

theorem ball_ratio (R B x : ℕ) : 
  R > 0 → B > 0 → x > 0 →
  R = (R + B + x) / 4 →
  R + x = (B + x) / 2 →
  R / B = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ball_ratio_l2875_287583


namespace NUMINAMATH_CALUDE_power_equation_solution_l2875_287518

theorem power_equation_solution (n : ℕ) : (3^n)^2 = 3^16 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2875_287518


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l2875_287526

/-- Given a circle with area 25π m², prove its diameter is 10 m. -/
theorem circle_diameter_from_area :
  ∀ (r : ℝ), 
    r > 0 →
    π * r^2 = 25 * π →
    2 * r = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l2875_287526


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2875_287548

/-- Given two isosceles right triangles with leg lengths 1, let x be the side length of a square
    inscribed in the first triangle with one vertex at the right angle, and y be the side length
    of a square inscribed in the second triangle with one side on the hypotenuse. -/
theorem inscribed_squares_ratio (x y : ℝ) 
  (hx : x = (1 : ℝ) / 2)  -- x is the side length of the square in the first triangle
  (hy : y = Real.sqrt 2 / 2) -- y is the side length of the square in the second triangle
  : x / y = Real.sqrt 2 := by
  sorry

#check inscribed_squares_ratio

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2875_287548


namespace NUMINAMATH_CALUDE_difference_of_products_equals_one_l2875_287555

theorem difference_of_products_equals_one : (1011 : ℕ) * 1011 - 1010 * 1012 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_products_equals_one_l2875_287555


namespace NUMINAMATH_CALUDE_intersection_point_l2875_287516

/-- The line defined by the equation y = -7x + 9 -/
def line (x : ℝ) : ℝ := -7 * x + 9

/-- The y-axis is defined as the set of points with x-coordinate equal to 0 -/
def y_axis (p : ℝ × ℝ) : Prop := p.1 = 0

/-- A point is on the line if its y-coordinate equals the line function at its x-coordinate -/
def on_line (p : ℝ × ℝ) : Prop := p.2 = line p.1

theorem intersection_point :
  ∃! p : ℝ × ℝ, y_axis p ∧ on_line p ∧ p = (0, 9) := by sorry

end NUMINAMATH_CALUDE_intersection_point_l2875_287516


namespace NUMINAMATH_CALUDE_darnell_call_minutes_l2875_287586

/-- Represents the monthly phone usage and plans for Darnell -/
structure PhoneUsage where
  unlimited_plan_cost : ℝ
  alt_plan_text_cost : ℝ
  alt_plan_text_limit : ℝ
  alt_plan_call_cost : ℝ
  alt_plan_call_limit : ℝ
  texts_sent : ℝ
  alt_plan_savings : ℝ

/-- Calculates the number of minutes Darnell spends on the phone each month -/
def calculate_call_minutes (usage : PhoneUsage) : ℝ :=
  sorry

/-- Theorem stating that given the conditions, Darnell spends 60 minutes on the phone each month -/
theorem darnell_call_minutes (usage : PhoneUsage) 
  (h1 : usage.unlimited_plan_cost = 12)
  (h2 : usage.alt_plan_text_cost = 1)
  (h3 : usage.alt_plan_text_limit = 30)
  (h4 : usage.alt_plan_call_cost = 3)
  (h5 : usage.alt_plan_call_limit = 20)
  (h6 : usage.texts_sent = 60)
  (h7 : usage.alt_plan_savings = 1) :
  calculate_call_minutes usage = 60 :=
sorry

end NUMINAMATH_CALUDE_darnell_call_minutes_l2875_287586


namespace NUMINAMATH_CALUDE_max_ballpoint_pens_l2875_287534

/-- Represents the number of pens of each type -/
structure PenCounts where
  ballpoint : ℕ
  gel : ℕ
  fountain : ℕ

/-- The cost of each type of pen in rubles -/
def penCosts : PenCounts := { ballpoint := 10, gel := 30, fountain := 60 }

/-- The total cost of a given combination of pens -/
def totalCost (counts : PenCounts) : ℕ :=
  counts.ballpoint * penCosts.ballpoint +
  counts.gel * penCosts.gel +
  counts.fountain * penCosts.fountain

/-- The total number of pens -/
def totalPens (counts : PenCounts) : ℕ :=
  counts.ballpoint + counts.gel + counts.fountain

/-- Predicate for a valid pen combination -/
def isValidCombination (counts : PenCounts) : Prop :=
  totalPens counts = 20 ∧
  totalCost counts = 500 ∧
  counts.ballpoint > 0 ∧
  counts.gel > 0 ∧
  counts.fountain > 0

/-- Theorem: The maximum number of ballpoint pens is 11 -/
theorem max_ballpoint_pens :
  ∃ (counts : PenCounts), isValidCombination counts ∧
    counts.ballpoint = 11 ∧
    ∀ (other : PenCounts), isValidCombination other →
      other.ballpoint ≤ counts.ballpoint :=
by sorry

end NUMINAMATH_CALUDE_max_ballpoint_pens_l2875_287534


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2875_287592

-- Problem 1
theorem problem_1 : (3 + Real.sqrt 5) * (Real.sqrt 5 - 2) = Real.sqrt 5 - 1 := by
  sorry

-- Problem 2
theorem problem_2 : (Real.sqrt 12 + Real.sqrt 27) / Real.sqrt 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2875_287592


namespace NUMINAMATH_CALUDE_integer_divisible_by_15_with_sqrt_between_30_and_30_5_l2875_287590

theorem integer_divisible_by_15_with_sqrt_between_30_and_30_5 :
  ∃ (n : ℕ), n > 0 ∧ 15 ∣ n ∧ 30 < Real.sqrt n ∧ Real.sqrt n < 30.5 ∧
  (n = 900 ∨ n = 915 ∨ n = 930) := by
sorry

end NUMINAMATH_CALUDE_integer_divisible_by_15_with_sqrt_between_30_and_30_5_l2875_287590


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2875_287519

theorem rationalize_denominator :
  ∃ (A B : ℚ) (C : ℕ),
    (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C ∧
    A = 11 / 4 ∧
    B = 5 / 4 ∧
    C = 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2875_287519


namespace NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l2875_287560

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem: The y-intercept of the common external tangent with positive slope for two given circles --/
theorem common_external_tangent_y_intercept 
  (c1 : Circle) 
  (c2 : Circle) 
  (h1 : c1.center = (5, -2)) 
  (h2 : c1.radius = 5)
  (h3 : c2.center = (20, 6))
  (h4 : c2.radius = 12) :
  ∃ (m b : ℝ), m > 0 ∧ b = -2100/161 ∧ 
  (∀ (x y : ℝ), y = m * x + b ↔ 
    (y - c1.center.2)^2 + (x - c1.center.1)^2 = (c1.radius + c2.radius)^2 ∧
    (y - c2.center.2)^2 + (x - c2.center.1)^2 = (c1.radius + c2.radius)^2) :=
sorry

end NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l2875_287560


namespace NUMINAMATH_CALUDE_prob_six_consecutive_heads_l2875_287577

/-- A fair coin is flipped 10 times. -/
def coin_flips : ℕ := 10

/-- The probability of getting heads on a single flip of a fair coin. -/
def prob_heads : ℚ := 1/2

/-- The set of all possible outcomes when flipping a coin 10 times. -/
def all_outcomes : Finset (Fin coin_flips → Bool) := sorry

/-- The set of outcomes with at least 6 consecutive heads. -/
def outcomes_with_six_consecutive_heads : Finset (Fin coin_flips → Bool) := sorry

/-- The probability of getting at least 6 consecutive heads in 10 flips of a fair coin. -/
theorem prob_six_consecutive_heads :
  (Finset.card outcomes_with_six_consecutive_heads : ℚ) / (Finset.card all_outcomes : ℚ) = 129/1024 :=
sorry

end NUMINAMATH_CALUDE_prob_six_consecutive_heads_l2875_287577


namespace NUMINAMATH_CALUDE_election_votes_theorem_l2875_287593

theorem election_votes_theorem (total_votes : ℕ) : 
  (∃ (candidate_votes rival_votes : ℕ),
    candidate_votes = total_votes / 10 ∧
    rival_votes = candidate_votes + 16000 ∧
    candidate_votes + rival_votes = total_votes) →
  total_votes = 20000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l2875_287593


namespace NUMINAMATH_CALUDE_johnny_closed_days_l2875_287580

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Represents the number of crab dishes Johnny makes per day -/
def dishes_per_day : ℕ := 40

/-- Represents the amount of crab meat used per dish in pounds -/
def crab_per_dish : ℚ := 3/2

/-- Represents the cost of crab meat per pound in dollars -/
def crab_cost_per_pound : ℕ := 8

/-- Represents Johnny's weekly expenditure on crab meat in dollars -/
def weekly_expenditure : ℕ := 1920

/-- Theorem stating that Johnny is closed 3 days a week -/
theorem johnny_closed_days : 
  days_in_week - (weekly_expenditure / (dishes_per_day * crab_per_dish * crab_cost_per_pound)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_johnny_closed_days_l2875_287580


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2875_287566

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_first_term : a 1 = 2) 
  (h_sum_condition : a 2 + a 4 = a 6) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) - a n = d) ∧ d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2875_287566


namespace NUMINAMATH_CALUDE_binomial_20_17_l2875_287524

theorem binomial_20_17 : Nat.choose 20 17 = 1140 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_17_l2875_287524


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_sides_l2875_287507

-- Define the quadrilateral
structure InscribedQuadrilateral where
  radius : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  perpendicular : Bool

-- Define the theorem
theorem inscribed_quadrilateral_sides
  (q : InscribedQuadrilateral)
  (h1 : q.radius = 10)
  (h2 : q.diagonal1 = 12)
  (h3 : q.diagonal2 = 10 * Real.sqrt 3)
  (h4 : q.perpendicular = true) :
  ∃ (s1 s2 s3 s4 : ℝ),
    (s1 = 4 * Real.sqrt 15 + 2 * Real.sqrt 5 ∧
     s2 = 4 * Real.sqrt 15 - 2 * Real.sqrt 5 ∧
     s3 = 4 * Real.sqrt 5 + 2 * Real.sqrt 15 ∧
     s4 = 4 * Real.sqrt 5 - 2 * Real.sqrt 15) ∨
    (s1 = 4 * Real.sqrt 15 + 2 * Real.sqrt 5 ∧
     s2 = 4 * Real.sqrt 15 - 2 * Real.sqrt 5 ∧
     s3 = 4 * Real.sqrt 5 - 2 * Real.sqrt 15 ∧
     s4 = 4 * Real.sqrt 5 + 2 * Real.sqrt 15) :=
by sorry


end NUMINAMATH_CALUDE_inscribed_quadrilateral_sides_l2875_287507


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l2875_287562

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem: The tangent line to y = x^3 - 3x at (0, 0) is y = -3x
theorem tangent_line_at_origin : 
  ∀ x : ℝ, (f' 0) * x = -3 * x := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l2875_287562


namespace NUMINAMATH_CALUDE_complement_of_union_l2875_287540

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {3, 4}

theorem complement_of_union :
  (U \ (A ∪ B)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2875_287540


namespace NUMINAMATH_CALUDE_triangle_square_perimeter_l2875_287582

theorem triangle_square_perimeter (a b c : ℝ) (s : ℝ) : 
  a = 5 → b = 12 → c = 13 → 
  (1/2) * a * b = s^2 → 
  4 * s = 4 * Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_triangle_square_perimeter_l2875_287582


namespace NUMINAMATH_CALUDE_spaatz_frankie_relation_binkie_frankie_relation_binkie_has_24_gemstones_l2875_287571

/-- The number of gemstones on Spaatz's collar -/
def spaatz_gemstones : ℕ := 1

/-- The number of gemstones on Frankie's collar -/
def frankie_gemstones : ℕ := 6

/-- The relationship between Spaatz's and Frankie's gemstones -/
theorem spaatz_frankie_relation : spaatz_gemstones = frankie_gemstones / 2 - 2 := by sorry

/-- The relationship between Binkie's and Frankie's gemstones -/
theorem binkie_frankie_relation : ∃ (binkie_gemstones : ℕ), binkie_gemstones = 4 * frankie_gemstones := by sorry

/-- The main theorem: Binkie has 24 gemstones -/
theorem binkie_has_24_gemstones : ∃ (binkie_gemstones : ℕ), binkie_gemstones = 24 := by sorry

end NUMINAMATH_CALUDE_spaatz_frankie_relation_binkie_frankie_relation_binkie_has_24_gemstones_l2875_287571


namespace NUMINAMATH_CALUDE_largest_divisor_l2875_287564

def product (n : ℕ) : ℕ := (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)*(n+15)

theorem largest_divisor (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (m : ℕ), m = 14175 ∧ 
  (∀ k : ℕ, k ∣ product n → k ≤ m) ∧
  (m ∣ product n) := by sorry

end NUMINAMATH_CALUDE_largest_divisor_l2875_287564


namespace NUMINAMATH_CALUDE_diameter_of_circle_with_radius_seven_l2875_287585

/-- The diameter of a circle is twice its radius -/
def diameter (radius : ℝ) : ℝ := 2 * radius

/-- For a circle with radius 7, the diameter is 14 -/
theorem diameter_of_circle_with_radius_seven :
  diameter 7 = 14 := by sorry

end NUMINAMATH_CALUDE_diameter_of_circle_with_radius_seven_l2875_287585


namespace NUMINAMATH_CALUDE_art_students_count_l2875_287565

theorem art_students_count (total : ℕ) (music : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : music = 40)
  (h3 : both = 10)
  (h4 : neither = 450) :
  ∃ art : ℕ, art = 20 ∧ total = (music - both) + (art - both) + both + neither :=
by sorry

end NUMINAMATH_CALUDE_art_students_count_l2875_287565


namespace NUMINAMATH_CALUDE_cos_equality_solution_l2875_287511

theorem cos_equality_solution (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (1230 * π / 180) → n = 150 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_solution_l2875_287511


namespace NUMINAMATH_CALUDE_brother_got_two_l2875_287550

-- Define the type for grades
inductive Grade : Type
  | one : Grade
  | two : Grade
  | three : Grade
  | four : Grade
  | five : Grade

-- Define the sneezing function
def grandmother_sneezes (statement : Prop) : Prop := sorry

-- Define the brother's grade
def brothers_grade : Grade := sorry

-- Theorem statement
theorem brother_got_two :
  -- Condition 1: When the brother tells the truth, the grandmother sneezes
  (∀ (statement : Prop), statement → grandmother_sneezes statement) →
  -- Condition 2: The brother said he got a "5", but the grandmother didn't sneeze
  (¬ grandmother_sneezes (brothers_grade = Grade.five)) →
  -- Condition 3: The brother said he got a "4", and the grandmother sneezed
  (grandmother_sneezes (brothers_grade = Grade.four)) →
  -- Condition 4: The brother said he got at least a "3", but the grandmother didn't sneeze
  (¬ grandmother_sneezes (brothers_grade = Grade.three ∨ brothers_grade = Grade.four ∨ brothers_grade = Grade.five)) →
  -- Conclusion: The brother's grade is 2
  brothers_grade = Grade.two :=
by
  sorry

end NUMINAMATH_CALUDE_brother_got_two_l2875_287550


namespace NUMINAMATH_CALUDE_cosine_sum_equality_l2875_287510

theorem cosine_sum_equality : 
  Real.cos (43 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (43 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_equality_l2875_287510


namespace NUMINAMATH_CALUDE_driveways_shoveled_l2875_287589

-- Define the prices and quantities
def candy_bar_price : ℚ := 3/4
def candy_bar_quantity : ℕ := 2
def lollipop_price : ℚ := 1/4
def lollipop_quantity : ℕ := 4
def driveway_price : ℚ := 3/2

-- Define the total spent at the candy store
def total_spent : ℚ := candy_bar_price * candy_bar_quantity + lollipop_price * lollipop_quantity

-- Define the fraction of earnings spent
def fraction_spent : ℚ := 1/6

-- Theorem to prove
theorem driveways_shoveled :
  (total_spent / fraction_spent) / driveway_price = 10 := by
  sorry


end NUMINAMATH_CALUDE_driveways_shoveled_l2875_287589
