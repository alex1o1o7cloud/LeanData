import Mathlib

namespace NUMINAMATH_CALUDE_students_playing_both_sports_l2408_240858

/-- Given a school with students playing football and cricket, calculate the number of students playing both sports. -/
theorem students_playing_both_sports 
  (total : ℕ) 
  (football : ℕ) 
  (cricket : ℕ) 
  (neither : ℕ) 
  (h1 : total = 470) 
  (h2 : football = 325) 
  (h3 : cricket = 175) 
  (h4 : neither = 50) : 
  football + cricket - (total - neither) = 80 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l2408_240858


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_8820_l2408_240894

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem count_perfect_square_factors_8820 :
  let factorization := prime_factorization 8820
  factorization = [(2, 2), (3, 2), (5, 1), (7, 2)] →
  count_perfect_square_factors 8820 = 8 := by sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_8820_l2408_240894


namespace NUMINAMATH_CALUDE_graph_decomposition_l2408_240806

/-- The graph of the equation (x^2 - 1)(x+y) = y^2(x+y) -/
def GraphEquation (x y : ℝ) : Prop :=
  (x^2 - 1) * (x + y) = y^2 * (x + y)

/-- The line y = -x -/
def Line (x y : ℝ) : Prop :=
  y = -x

/-- The hyperbola (x+y)(x-y) = 1 -/
def Hyperbola (x y : ℝ) : Prop :=
  (x + y) * (x - y) = 1

theorem graph_decomposition :
  ∀ x y : ℝ, GraphEquation x y ↔ (Line x y ∨ Hyperbola x y) :=
sorry

end NUMINAMATH_CALUDE_graph_decomposition_l2408_240806


namespace NUMINAMATH_CALUDE_triangle_shape_l2408_240850

theorem triangle_shape (a b : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → 
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.cos A = b * Real.cos B →
  (A = B ∨ A + B = π / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l2408_240850


namespace NUMINAMATH_CALUDE_solution_set_f_positive_l2408_240877

/-- A function f is even if f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem solution_set_f_positive
    (f : ℝ → ℝ)
    (h_even : EvenFunction f)
    (h_nonneg : ∀ x ≥ 0, f x = 2^x - 4) :
    {x : ℝ | f x > 0} = Set.Ioi 2 ∪ Set.Iio (-2) :=
  sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_l2408_240877


namespace NUMINAMATH_CALUDE_set_union_problem_l2408_240898

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem set_union_problem (x : ℝ) :
  (A x ∩ B x = {9}) → (A x ∪ B x = {-8, -4, 4, -7, 9}) :=
by sorry

end NUMINAMATH_CALUDE_set_union_problem_l2408_240898


namespace NUMINAMATH_CALUDE_isabel_piggy_bank_l2408_240830

theorem isabel_piggy_bank (X : ℝ) : 
  (X > 0) → 
  ((1 - 0.25) * (1 / 2) * (2 / 3) * X = 60) → 
  (X = 720) := by
sorry

end NUMINAMATH_CALUDE_isabel_piggy_bank_l2408_240830


namespace NUMINAMATH_CALUDE_cone_height_from_circular_sector_l2408_240841

theorem cone_height_from_circular_sector (r : ℝ) (h : r = 8) :
  let sector_arc_length := 2 * π * r / 4
  let cone_base_radius := sector_arc_length / (2 * π)
  let cone_slant_height := r
  cone_slant_height ^ 2 - cone_base_radius ^ 2 = (2 * Real.sqrt 15) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_from_circular_sector_l2408_240841


namespace NUMINAMATH_CALUDE_expression_evaluation_l2408_240854

theorem expression_evaluation : 2 + 3 * 4 - 5 + 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2408_240854


namespace NUMINAMATH_CALUDE_square_perimeters_sum_l2408_240849

theorem square_perimeters_sum (x y : ℝ) (h1 : x^2 + y^2 = 125) (h2 : x^2 - y^2 = 65) :
  4*x + 4*y = 60 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeters_sum_l2408_240849


namespace NUMINAMATH_CALUDE_parabola_line_intersection_length_no_isosceles_right_triangle_on_parabola_l2408_240835

/-- Given a parabola y^2 = 2px where p > 0, and a line y = k(x - p/2) intersecting 
    the parabola at points A and B, the length of AB is (2p(k^2 + 1)) / k^2 -/
theorem parabola_line_intersection_length (p k : ℝ) (hp : p > 0) :
  let f : ℝ → ℝ := λ k => (2 * p * (k^2 + 1)) / k^2
  let parabola : ℝ × ℝ → Prop := λ (x, y) => y^2 = 2 * p * x
  let line : ℝ → ℝ := λ x => k * (x - p / 2)
  let A := (x₁, line x₁)
  let B := (x₂, line x₂)
  parabola A ∧ parabola B → abs (x₂ - x₁) = f k :=
by sorry

/-- There does not exist a point C on the parabola y^2 = 2px such that 
    triangle ABC is an isosceles right triangle with C as the vertex of the right angle -/
theorem no_isosceles_right_triangle_on_parabola (p : ℝ) (hp : p > 0) :
  let parabola : ℝ × ℝ → Prop := λ (x, y) => y^2 = 2 * p * x
  ¬ ∃ (A B C : ℝ × ℝ), parabola A ∧ parabola B ∧ parabola C ∧
    (C.1 < (A.1 + B.1) / 2) ∧
    (abs (A.1 - C.1) = abs (B.1 - C.1)) ∧
    ((A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_length_no_isosceles_right_triangle_on_parabola_l2408_240835


namespace NUMINAMATH_CALUDE_cube_root_problem_l2408_240891

theorem cube_root_problem : (0.07 : ℝ)^3 = 0.000343 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l2408_240891


namespace NUMINAMATH_CALUDE_focus_of_given_parabola_l2408_240851

/-- A parabola is defined by the equation y = ax^2 + bx + c where a ≠ 0 -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- The given parabola y = 4x^2 + 8x - 5 -/
def given_parabola : Parabola :=
  { a := 4
    b := 8
    c := -5
    a_nonzero := by norm_num }

theorem focus_of_given_parabola :
  focus given_parabola = (-1, -143/16) := by sorry

end NUMINAMATH_CALUDE_focus_of_given_parabola_l2408_240851


namespace NUMINAMATH_CALUDE_mono_increasing_sufficient_not_necessary_l2408_240836

open Set
open Function

-- Define a monotonically increasing function
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Statement B
def StatementB (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ < x₂ ∧ f x₁ < f x₂

-- Theorem to prove
theorem mono_increasing_sufficient_not_necessary :
  (∀ f : ℝ → ℝ, MonoIncreasing f → StatementB f) ∧
  (∃ g : ℝ → ℝ, ¬MonoIncreasing g ∧ StatementB g) :=
by sorry

end NUMINAMATH_CALUDE_mono_increasing_sufficient_not_necessary_l2408_240836


namespace NUMINAMATH_CALUDE_distance_between_parallel_lines_l2408_240828

/-- The distance between two parallel lines -/
theorem distance_between_parallel_lines 
  (a : ℝ) -- Coefficient of x in the second line
  (h_parallel : a = 6) -- Condition for parallelism
  : (|(-24) - 11|) / Real.sqrt (3^2 + 4^2) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_parallel_lines_l2408_240828


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_101011101_base_7_l2408_240862

def base_seven_to_decimal (n : ℕ) : ℕ := 
  7^8 + 7^6 + 7^4 + 7^3 + 7^2 + 1

theorem largest_prime_divisor_of_101011101_base_7 :
  ∃ (p : ℕ), Prime p ∧ p ∣ base_seven_to_decimal 101011101 ∧
  ∀ (q : ℕ), Prime q → q ∣ base_seven_to_decimal 101011101 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_101011101_base_7_l2408_240862


namespace NUMINAMATH_CALUDE_apollonius_circle_locus_l2408_240831

/-- Given two points A and B in a 2D plane, and a positive real number n,
    the Apollonius circle is the locus of points P such that PA = n * PB -/
theorem apollonius_circle_locus 
  (A B : EuclideanSpace ℝ (Fin 2))  -- Two given points in 2D space
  (n : ℝ) 
  (hn : n > 0) :  -- n is positive
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ),
    ∀ P : EuclideanSpace ℝ (Fin 2), 
      dist P A = n * dist P B ↔ 
      dist P center = radius :=
sorry

end NUMINAMATH_CALUDE_apollonius_circle_locus_l2408_240831


namespace NUMINAMATH_CALUDE_remaining_amount_proof_l2408_240866

-- Define the deposit percentage
def deposit_percentage : ℚ := 10 / 100

-- Define the deposit amount
def deposit_amount : ℚ := 55

-- Define the total cost
def total_cost : ℚ := deposit_amount / deposit_percentage

-- Define the remaining amount to be paid
def remaining_amount : ℚ := total_cost - deposit_amount

-- Theorem to prove
theorem remaining_amount_proof : remaining_amount = 495 := by
  sorry

end NUMINAMATH_CALUDE_remaining_amount_proof_l2408_240866


namespace NUMINAMATH_CALUDE_number_of_bags_l2408_240840

/-- 
Given:
- Each bag contains 7 cookies
- Each box contains 12 cookies
- 8 boxes contain 33 more cookies than the number of bags we're looking for

Prove that the number of bags is 9.
-/
theorem number_of_bags : ∃ (B : ℕ), 
  (8 * 12 = 7 * B + 33) ∧ 
  B = 9 := by
sorry

end NUMINAMATH_CALUDE_number_of_bags_l2408_240840


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l2408_240876

theorem sum_of_three_consecutive_cubes_divisible_by_nine (n : ℕ) :
  ∃ k : ℕ, n^3 + (n+1)^3 + (n+2)^3 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l2408_240876


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_equality_l2408_240855

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Theorem statement
theorem arithmetic_sequence_sum_equality 
  (a : ℕ → ℝ) (h : is_arithmetic_sequence a) : 
  a 1 + a 8 = a 4 + a 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_equality_l2408_240855


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l2408_240890

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit on a floor -/
def maxTiles (floor : Dimensions) (tile : Dimensions) : ℕ :=
  let horizontal := (floor.length / tile.length) * (floor.width / tile.width)
  let vertical := (floor.length / tile.width) * (floor.width / tile.length)
  max horizontal vertical

/-- Theorem stating the maximum number of tiles on the given floor -/
theorem max_tiles_on_floor :
  let floor : Dimensions := ⟨100, 150⟩
  let tile : Dimensions := ⟨20, 30⟩
  maxTiles floor tile = 25 := by
  sorry

#eval maxTiles ⟨100, 150⟩ ⟨20, 30⟩

end NUMINAMATH_CALUDE_max_tiles_on_floor_l2408_240890


namespace NUMINAMATH_CALUDE_land_price_per_acre_l2408_240823

theorem land_price_per_acre (total_acres : ℕ) (num_lots : ℕ) (price_per_lot : ℕ) : 
  total_acres = 4 →
  num_lots = 9 →
  price_per_lot = 828 →
  (num_lots * price_per_lot) / total_acres = 1863 := by
sorry

end NUMINAMATH_CALUDE_land_price_per_acre_l2408_240823


namespace NUMINAMATH_CALUDE_squirrel_solution_l2408_240873

/-- The number of walnuts the girl squirrel ate -/
def squirrel_problem (initial : ℕ) (boy_gathered : ℕ) (boy_dropped : ℕ) (girl_brought : ℕ) (final : ℕ) : ℕ :=
  initial + boy_gathered - boy_dropped + girl_brought - final

/-- Theorem stating the solution to the squirrel problem -/
theorem squirrel_solution : squirrel_problem 12 6 1 5 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_solution_l2408_240873


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l2408_240867

/-- Given a train of length 1200 m that crosses a tree in 120 seconds,
    prove that the time required for the train to pass a platform of length 400 m is 160 seconds. -/
theorem train_platform_crossing_time :
  let train_length : ℝ := 1200
  let tree_crossing_time : ℝ := 120
  let platform_length : ℝ := 400
  let train_speed : ℝ := train_length / tree_crossing_time
  let total_distance : ℝ := train_length + platform_length
  total_distance / train_speed = 160 := by sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l2408_240867


namespace NUMINAMATH_CALUDE_count_divisible_by_11_equals_v_l2408_240815

/-- Concatenates the squares of integers from 1 to n -/
def b (n : ℕ) : ℕ := sorry

/-- Counts how many numbers b_k are divisible by 11 for 1 ≤ k ≤ 50 -/
def count_divisible_by_11 : ℕ := sorry

/-- The correct count of numbers b_k divisible by 11 for 1 ≤ k ≤ 50 -/
def v : ℕ := sorry

theorem count_divisible_by_11_equals_v : count_divisible_by_11 = v := by sorry

end NUMINAMATH_CALUDE_count_divisible_by_11_equals_v_l2408_240815


namespace NUMINAMATH_CALUDE_gnomes_and_ponies_count_l2408_240896

/-- Represents the number of gnomes -/
def num_gnomes : ℕ := 12

/-- Represents the number of ponies -/
def num_ponies : ℕ := 3

/-- The total number of heads in the caravan -/
def total_heads : ℕ := 15

/-- The total number of legs in the caravan -/
def total_legs : ℕ := 36

/-- Each gnome has this many legs -/
def gnome_legs : ℕ := 2

/-- Each pony has this many legs -/
def pony_legs : ℕ := 4

theorem gnomes_and_ponies_count :
  (num_gnomes + num_ponies = total_heads) ∧
  (num_gnomes * gnome_legs + num_ponies * pony_legs = total_legs) :=
by sorry

end NUMINAMATH_CALUDE_gnomes_and_ponies_count_l2408_240896


namespace NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l2408_240889

theorem largest_n_for_sin_cos_inequality : 
  (∀ n : ℕ, n > 8 → ∃ x : ℝ, (Real.sin x)^n + (Real.cos x)^n < 1 / (2 * n)) ∧ 
  (∀ x : ℝ, (Real.sin x)^8 + (Real.cos x)^8 ≥ 1 / 16) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l2408_240889


namespace NUMINAMATH_CALUDE_complex_problem_l2408_240820

theorem complex_problem (a : ℝ) (z₁ : ℂ) (h₁ : a < 0) (h₂ : z₁ = 1 + a * Complex.I) 
  (h₃ : Complex.re (z₁^2) = 0) : 
  a = -1 ∧ Complex.abs ((z₁ / (1 + Complex.I)) + 2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_problem_l2408_240820


namespace NUMINAMATH_CALUDE_constant_function_integral_equals_one_l2408_240800

theorem constant_function_integral_equals_one : 
  ∫ x in (0 : ℝ)..1, (1 : ℝ) = 1 := by sorry

end NUMINAMATH_CALUDE_constant_function_integral_equals_one_l2408_240800


namespace NUMINAMATH_CALUDE_number_manipulation_l2408_240878

theorem number_manipulation (x : ℚ) (h : (7/8) * x = 28) : (x + 16) * (5/16) = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_manipulation_l2408_240878


namespace NUMINAMATH_CALUDE_more_pockets_than_dollars_per_wallet_l2408_240816

/-- Represents the distribution of dollars, wallets, and pockets -/
structure Distribution where
  total_dollars : ℕ
  num_wallets : ℕ
  num_pockets : ℕ
  dollars_per_pocket : ℕ → ℕ
  dollars_per_wallet : ℕ → ℕ

/-- The conditions of the problem -/
def problem_conditions (d : Distribution) : Prop :=
  d.total_dollars = 2003 ∧
  d.num_wallets > 0 ∧
  d.num_pockets > 0 ∧
  (∀ p, p < d.num_pockets → d.dollars_per_pocket p < d.num_wallets) ∧
  (∀ w, w < d.num_wallets → d.dollars_per_wallet w ≤ d.total_dollars / d.num_wallets)

/-- The theorem to be proved -/
theorem more_pockets_than_dollars_per_wallet (d : Distribution) 
  (h : problem_conditions d) : 
  ∀ w, w < d.num_wallets → d.num_pockets > d.dollars_per_wallet w :=
sorry

end NUMINAMATH_CALUDE_more_pockets_than_dollars_per_wallet_l2408_240816


namespace NUMINAMATH_CALUDE_election_result_count_l2408_240856

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of positions to be filled -/
def num_positions : ℕ := 2

/-- The number of ways to select students for positions with at least one female student -/
def ways_with_female : ℕ := total_students.choose num_positions * num_positions.factorial - num_male.choose num_positions * num_positions.factorial

theorem election_result_count : ways_with_female = 14 := by
  sorry

end NUMINAMATH_CALUDE_election_result_count_l2408_240856


namespace NUMINAMATH_CALUDE_colten_chicken_count_l2408_240868

/-- Represents the number of chickens each person has. -/
structure ChickenCount where
  colten : ℕ
  skylar : ℕ
  quentin : ℕ

/-- The conditions of the chicken problem. -/
def ChickenProblem (c : ChickenCount) : Prop :=
  c.colten + c.skylar + c.quentin = 383 ∧
  c.quentin = 25 + 2 * c.skylar ∧
  c.skylar = 3 * c.colten - 4

theorem colten_chicken_count :
  ∀ c : ChickenCount, ChickenProblem c → c.colten = 37 := by
  sorry

end NUMINAMATH_CALUDE_colten_chicken_count_l2408_240868


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2408_240839

/-- The area of a quadrilateral with vertices at (0, 0), (0, 2), (3, 2), and (5, 5) is 5.5 square units. -/
theorem quadrilateral_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 2)
  let C : ℝ × ℝ := (3, 2)
  let D : ℝ × ℝ := (5, 5)
  let area_triangle (p q r : ℝ × ℝ) : ℝ := 
    (1/2) * abs ((p.1 * (q.2 - r.2) + q.1 * (r.2 - p.2) + r.1 * (p.2 - q.2)) : ℝ)
  (area_triangle A B C) + (area_triangle A C D) = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l2408_240839


namespace NUMINAMATH_CALUDE_probability_two_black_marbles_l2408_240892

/-- The probability of drawing two black marbles without replacement from a jar -/
theorem probability_two_black_marbles (blue yellow black : ℕ) 
  (h_blue : blue = 4)
  (h_yellow : yellow = 5)
  (h_black : black = 12) : 
  (black / (blue + yellow + black)) * ((black - 1) / (blue + yellow + black - 1)) = 11 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_black_marbles_l2408_240892


namespace NUMINAMATH_CALUDE_deepak_age_l2408_240865

/-- Proves that Deepak's current age is 42 years given the specified conditions --/
theorem deepak_age (arun deepak kamal : ℕ) : 
  arun * 7 = deepak * 5 →
  kamal * 5 = deepak * 9 →
  arun + 6 = 36 →
  kamal + 6 = 2 * (deepak + 6) →
  deepak = 42 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l2408_240865


namespace NUMINAMATH_CALUDE_remainder_seven_divisors_of_sixtyone_l2408_240818

theorem remainder_seven_divisors_of_sixtyone : 
  (Finset.filter (fun n : ℕ => n > 7 ∧ 61 % n = 7) (Finset.range 62)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_divisors_of_sixtyone_l2408_240818


namespace NUMINAMATH_CALUDE_power_negative_two_m_squared_cubed_l2408_240887

theorem power_negative_two_m_squared_cubed (m : ℝ) : (-2 * m^2)^3 = -8 * m^6 := by
  sorry

end NUMINAMATH_CALUDE_power_negative_two_m_squared_cubed_l2408_240887


namespace NUMINAMATH_CALUDE_union_A_B_when_a_is_one_range_of_a_for_necessary_not_sufficient_l2408_240804

-- Define set A
def A (a : ℝ) : Set ℝ := {x | (x - a) / (x - 3*a) < 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 5*x + 6 < 0}

-- Theorem for part (1)
theorem union_A_B_when_a_is_one : 
  A 1 ∪ B = {x | 1 < x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem range_of_a_for_necessary_not_sufficient : 
  {a : ℝ | ∀ x, x ∈ B → x ∈ A a ∧ ∃ y, y ∈ A a ∧ y ∉ B} = {a | 1 ≤ a ∧ a ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_is_one_range_of_a_for_necessary_not_sufficient_l2408_240804


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2408_240824

theorem fraction_to_decimal : (58 : ℚ) / 160 = 0.3625 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2408_240824


namespace NUMINAMATH_CALUDE_trig_product_identities_l2408_240825

open Real

theorem trig_product_identities (α : ℝ) :
  (1 + sin α = 2 * sin ((π/2 + α)/2) * cos ((π/2 - α)/2)) ∧
  (1 - sin α = 2 * cos ((π/2 + α)/2) * sin ((π/2 - α)/2)) ∧
  (1 + 2 * sin α = 4 * sin ((π/6 + α)/2) * cos ((π/6 - α)/2)) ∧
  (1 - 2 * sin α = 4 * cos ((π/6 + α)/2) * sin ((π/6 - α)/2)) ∧
  (1 + 2 * cos α = 4 * cos ((π/3 + α)/2) * cos ((π/3 - α)/2)) ∧
  (1 - 2 * cos α = -4 * sin ((π/3 + α)/2) * sin ((π/3 - α)/2)) :=
by sorry


end NUMINAMATH_CALUDE_trig_product_identities_l2408_240825


namespace NUMINAMATH_CALUDE_ball_probabilities_l2408_240870

-- Define the number of balls in each can
def can_A : Fin 3 → ℕ
| 0 => 5  -- red balls
| 1 => 2  -- white balls
| 2 => 3  -- black balls

def can_B : Fin 3 → ℕ
| 0 => 4  -- red balls
| 1 => 3  -- white balls
| 2 => 3  -- black balls

-- Define the probability of drawing a ball of each color from can A
def prob_A (i : Fin 3) : ℚ :=
  (can_A i : ℚ) / (can_A 0 + can_A 1 + can_A 2 : ℚ)

-- Define the probability of drawing a red ball from can B after moving a ball from A
def prob_B_red (i : Fin 3) : ℚ :=
  (can_B 0 + (if i = 0 then 1 else 0) : ℚ) / 
  ((can_B 0 + can_B 1 + can_B 2 + 1) : ℚ)

theorem ball_probabilities :
  (prob_B_red 0 = 5/11) ∧ 
  (prob_A 2 * prob_B_red 2 = 6/55) ∧
  (prob_A 0 * prob_B_red 0 / (prob_A 0 * prob_B_red 0 + prob_A 1 * prob_B_red 1 + prob_A 2 * prob_B_red 2) = 5/9) := by
  sorry


end NUMINAMATH_CALUDE_ball_probabilities_l2408_240870


namespace NUMINAMATH_CALUDE_probability_above_parabola_l2408_240884

/-- A single-digit positive integer -/
def SingleDigit := { n : ℕ | 1 ≤ n ∧ n ≤ 9 }

/-- The total number of possible (a, b) combinations -/
def TotalCombinations : ℕ := 81

/-- The number of valid (a, b) combinations where (a, b) lies above y = ax^2 + bx -/
def ValidCombinations : ℕ := 72

/-- The probability that a randomly chosen point (a, b) lies above y = ax^2 + bx -/
def ProbabilityAboveParabola : ℚ := ValidCombinations / TotalCombinations

theorem probability_above_parabola :
  ProbabilityAboveParabola = 8 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_above_parabola_l2408_240884


namespace NUMINAMATH_CALUDE_water_depth_in_cistern_l2408_240883

theorem water_depth_in_cistern (length width total_wet_area : ℝ) 
  (h_length : length = 7)
  (h_width : width = 4)
  (h_total_wet_area : total_wet_area = 55.5)
  : ∃ depth : ℝ, 
    depth = 1.25 ∧ 
    total_wet_area = length * width + 2 * length * depth + 2 * width * depth :=
by sorry

end NUMINAMATH_CALUDE_water_depth_in_cistern_l2408_240883


namespace NUMINAMATH_CALUDE_orangeade_pricing_l2408_240826

/-- Orangeade pricing problem -/
theorem orangeade_pricing
  (orange_juice : ℝ) -- Amount of orange juice (same for both days)
  (water_day1 : ℝ) -- Amount of water on day 1
  (price_day1 : ℝ) -- Price per glass on day 1
  (h1 : water_day1 = orange_juice) -- Equal amounts of orange juice and water on day 1
  (h2 : price_day1 = 0.60) -- Price per glass on day 1 is $0.60
  : -- Price per glass on day 2
    (price_day1 * (orange_juice + water_day1)) / (orange_juice + 2 * water_day1) = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_orangeade_pricing_l2408_240826


namespace NUMINAMATH_CALUDE_box_C_in_A_l2408_240859

/-- The number of Box B that can fill one Box A -/
def box_B_in_A : ℕ := 4

/-- The number of Box C that can fill one Box B -/
def box_C_in_B : ℕ := 6

/-- The theorem stating that 24 Box C are needed to fill Box A -/
theorem box_C_in_A : box_B_in_A * box_C_in_B = 24 := by
  sorry

end NUMINAMATH_CALUDE_box_C_in_A_l2408_240859


namespace NUMINAMATH_CALUDE_simplify_expression_l2408_240879

theorem simplify_expression : 
  Real.sqrt 6 * (Real.sqrt 2 + Real.sqrt 3) - 3 * Real.sqrt (1/3) = Real.sqrt 3 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2408_240879


namespace NUMINAMATH_CALUDE_tetromino_properties_l2408_240802

/-- A tetromino is a shape made up of 4 squares. -/
structure Tetromino where
  squares : Finset (ℤ × ℤ)
  card_eq_four : squares.card = 4

/-- Two tetrominos are considered identical if they can be superimposed by rotating but not by flipping. -/
def are_identical (t1 t2 : Tetromino) : Prop := sorry

/-- The set of all distinct tetrominos. -/
def distinct_tetrominos : Finset Tetromino := sorry

/-- A 4 × 7 rectangle. -/
def rectangle : Finset (ℤ × ℤ) := sorry

/-- Tiling a rectangle with tetrominos. -/
def tiling (r : Finset (ℤ × ℤ)) (ts : Finset Tetromino) : Prop := sorry

theorem tetromino_properties :
  (distinct_tetrominos.card = 7) ∧
  ¬ (tiling rectangle distinct_tetrominos) := by sorry

end NUMINAMATH_CALUDE_tetromino_properties_l2408_240802


namespace NUMINAMATH_CALUDE_base6_addition_l2408_240843

-- Define a function to convert from base 6 to base 10
def base6ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 6
def base10ToBase6 (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem base6_addition : 
  base10ToBase6 (base6ToBase10 25 + base6ToBase10 35) = 104 := by sorry

end NUMINAMATH_CALUDE_base6_addition_l2408_240843


namespace NUMINAMATH_CALUDE_original_price_from_decreased_price_l2408_240848

/-- 
If an article's price after a 50% decrease is 620 (in some currency unit),
then its original price was 1240 (in the same currency unit).
-/
theorem original_price_from_decreased_price (decreased_price : ℝ) 
  (h : decreased_price = 620) : 
  ∃ (original_price : ℝ), 
    original_price * 0.5 = decreased_price ∧ 
    original_price = 1240 :=
by sorry

end NUMINAMATH_CALUDE_original_price_from_decreased_price_l2408_240848


namespace NUMINAMATH_CALUDE_M_equals_N_l2408_240819

def M : Set ℝ := {x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6)}

def N : Set ℝ := {y | ∃ n : ℤ, y = Real.cos (n * Real.pi / 3)}

theorem M_equals_N : M = N := by sorry

end NUMINAMATH_CALUDE_M_equals_N_l2408_240819


namespace NUMINAMATH_CALUDE_xyz_value_l2408_240807

theorem xyz_value (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 280 / 3 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l2408_240807


namespace NUMINAMATH_CALUDE_bakery_earnings_l2408_240885

/-- Represents the daily production and prices of baked goods in a bakery --/
structure BakeryData where
  cupcake_price : ℝ
  cookie_price : ℝ
  biscuit_price : ℝ
  daily_cupcakes : ℕ
  daily_cookies : ℕ
  daily_biscuits : ℕ

/-- Calculates the total earnings for a given number of days --/
def total_earnings (data : BakeryData) (days : ℕ) : ℝ :=
  (data.cupcake_price * data.daily_cupcakes +
   data.cookie_price * data.daily_cookies +
   data.biscuit_price * data.daily_biscuits) * days

/-- Theorem stating that the total earnings for 5 days is $350 --/
theorem bakery_earnings (data : BakeryData) 
  (h1 : data.cupcake_price = 1.5)
  (h2 : data.cookie_price = 2)
  (h3 : data.biscuit_price = 1)
  (h4 : data.daily_cupcakes = 20)
  (h5 : data.daily_cookies = 10)
  (h6 : data.daily_biscuits = 20) :
  total_earnings data 5 = 350 := by
  sorry

end NUMINAMATH_CALUDE_bakery_earnings_l2408_240885


namespace NUMINAMATH_CALUDE_birthday_cookies_l2408_240875

/-- The number of pans of cookies -/
def num_pans : ℕ := 5

/-- The number of cookies per pan -/
def cookies_per_pan : ℕ := 8

/-- The total number of cookies baked -/
def total_cookies : ℕ := num_pans * cookies_per_pan

theorem birthday_cookies : total_cookies = 40 := by
  sorry

end NUMINAMATH_CALUDE_birthday_cookies_l2408_240875


namespace NUMINAMATH_CALUDE_jovana_shells_added_l2408_240814

/-- The amount of shells added to a bucket -/
def shells_added (initial_amount final_amount : ℕ) : ℕ :=
  final_amount - initial_amount

/-- Theorem: The amount of shells Jovana added is 23 pounds -/
theorem jovana_shells_added :
  let initial_amount : ℕ := 5
  let final_amount : ℕ := 28
  shells_added initial_amount final_amount = 23 := by
  sorry

end NUMINAMATH_CALUDE_jovana_shells_added_l2408_240814


namespace NUMINAMATH_CALUDE_return_probability_after_2012_moves_chessboard_return_probability_l2408_240844

/-- Represents the size of the chessboard -/
def boardSize : ℕ := 8

/-- Represents the total number of moves -/
def totalMoves : ℕ := 2012

/-- Represents the probability of returning to the original position after a given number of moves -/
noncomputable def returnProbability (n : ℕ) : ℚ :=
  ((1 + 2^(n / 2 - 1)) / 2^(n / 2 + 1))^2

/-- Theorem stating the probability of returning to the original position after 2012 moves -/
theorem return_probability_after_2012_moves :
  returnProbability totalMoves = ((1 + 2^1005) / 2^1007)^2 := by
  sorry

/-- Theorem stating that the calculated probability is correct for the given chessboard and moves -/
theorem chessboard_return_probability :
  boardSize = 8 →
  totalMoves = 2012 →
  returnProbability totalMoves = ((1 + 2^1005) / 2^1007)^2 := by
  sorry

end NUMINAMATH_CALUDE_return_probability_after_2012_moves_chessboard_return_probability_l2408_240844


namespace NUMINAMATH_CALUDE_max_regular_hours_is_40_l2408_240805

/-- Calculates the maximum number of regular hours worked given total pay, overtime hours, and pay rates. -/
def max_regular_hours (total_pay : ℚ) (overtime_hours : ℚ) (regular_rate : ℚ) : ℚ :=
  let overtime_rate := 2 * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  let regular_pay := total_pay - overtime_pay
  regular_pay / regular_rate

/-- Proves that given the specified conditions, the maximum number of regular hours is 40. -/
theorem max_regular_hours_is_40 :
  max_regular_hours 168 8 3 = 40 := by
  sorry

#eval max_regular_hours 168 8 3

end NUMINAMATH_CALUDE_max_regular_hours_is_40_l2408_240805


namespace NUMINAMATH_CALUDE_fifteen_choose_three_l2408_240861

theorem fifteen_choose_three : 
  Nat.choose 15 3 = 455 := by sorry

end NUMINAMATH_CALUDE_fifteen_choose_three_l2408_240861


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l2408_240832

-- Define the original number
def original_number : ℝ := 850000

-- Define the scientific notation components
def coefficient : ℝ := 8.5
def exponent : ℤ := 5

-- Theorem statement
theorem scientific_notation_equality :
  original_number = coefficient * (10 : ℝ) ^ exponent := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l2408_240832


namespace NUMINAMATH_CALUDE_union_and_intersection_range_of_m_l2408_240810

-- Define the sets A, B, and C
def A : Set ℝ := {x | -4 < x ∧ x < 2}
def B : Set ℝ := {x | x < -5 ∨ x > 1}
def C (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m + 1}

-- Theorem for part (1)
theorem union_and_intersection :
  (A ∪ B = {x | x < -5 ∨ x > -4}) ∧
  (A ∩ (Set.univ \ B) = {x | -4 < x ∧ x ≤ 1}) := by sorry

-- Theorem for part (2)
theorem range_of_m (m : ℝ) :
  (B ∩ C m = ∅) → (m ∈ Set.Icc (-4) 0) := by sorry

end NUMINAMATH_CALUDE_union_and_intersection_range_of_m_l2408_240810


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2408_240852

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 8 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2408_240852


namespace NUMINAMATH_CALUDE_tetrahedron_surface_area_bound_l2408_240880

/-- Tetrahedron with given edge lengths and surface area -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  S : ℝ
  edge_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f
  surface_positive : 0 < S

/-- The surface area of a tetrahedron is bounded by a function of its edge lengths -/
theorem tetrahedron_surface_area_bound (t : Tetrahedron) :
    t.S ≤ (Real.sqrt 3 / 6) * (t.a^2 + t.b^2 + t.c^2 + t.d^2 + t.e^2 + t.f^2) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_surface_area_bound_l2408_240880


namespace NUMINAMATH_CALUDE_somu_father_age_ratio_l2408_240811

/-- Represents the ratio of two integers as a pair of integers -/
def Ratio := ℤ × ℤ

/-- Somu's present age in years -/
def somu_age : ℕ := 10

/-- Calculates the father's age given Somu's age -/
def father_age (s : ℕ) : ℕ :=
  5 * (s - 5) + 5

/-- Simplifies a ratio by dividing both numbers by their greatest common divisor -/
def simplify_ratio (r : Ratio) : Ratio :=
  let gcd := r.1.gcd r.2
  (r.1 / gcd, r.2 / gcd)

theorem somu_father_age_ratio :
  simplify_ratio (somu_age, father_age somu_age) = (1, 3) := by
  sorry

end NUMINAMATH_CALUDE_somu_father_age_ratio_l2408_240811


namespace NUMINAMATH_CALUDE_interest_difference_theorem_l2408_240857

/-- Calculates the difference between the principal and the simple interest --/
def interestDifference (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal - (principal * rate * time)

/-- Theorem stating that the difference between the principal and the simple interest
    is 340 for the given conditions --/
theorem interest_difference_theorem :
  interestDifference 500 0.04 8 = 340 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_theorem_l2408_240857


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2408_240893

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : geometric_sequence a q)
  (h_cond : 2 * a 5 = a 3 - a 4)
  (h_exist : ∃ n m : ℕ, a 1 = 4 * Real.sqrt (a n * a m)) :
  ∃ n m : ℕ, a 1 = 4 * Real.sqrt (a n * a m) ∧ n + m = 6 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2408_240893


namespace NUMINAMATH_CALUDE_g_properties_and_range_l2408_240869

def f (x : ℝ) : ℝ := x^2 - 3*x + 2

def g (x : ℝ) : ℝ := |x|^2 - 3*|x| + 2

theorem g_properties_and_range :
  (∀ x : ℝ, g (-x) = g x) ∧
  (∀ x : ℝ, x ≥ 0 → g x = f x) ∧
  ({m : ℝ | g m > 2} = {m : ℝ | m < -3 ∨ m > 3}) := by
  sorry

end NUMINAMATH_CALUDE_g_properties_and_range_l2408_240869


namespace NUMINAMATH_CALUDE_complex_inequality_l2408_240847

theorem complex_inequality (a : ℝ) : 
  (1 - Complex.I) + (1 + Complex.I) * a ≠ 0 → a ≠ -1 ∧ a ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l2408_240847


namespace NUMINAMATH_CALUDE_quadratic_range_l2408_240827

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 1)^2 + 1

-- State the theorem
theorem quadratic_range :
  ∀ x : ℝ, (2 ≤ f x ∧ f x < 5) ↔ (-1 < x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_l2408_240827


namespace NUMINAMATH_CALUDE_solution_and_inequality_l2408_240899

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - 4| - t

-- State the theorem
theorem solution_and_inequality (t : ℝ) 
  (h1 : Set.Icc (-1 : ℝ) 5 = {x | f t (x + 2) ≤ 2}) 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h2 : a + b + c = t) : 
  t = 1 ∧ a^2 / b + b^2 / c + c^2 / a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_and_inequality_l2408_240899


namespace NUMINAMATH_CALUDE_pencils_remainder_l2408_240817

theorem pencils_remainder : Nat.mod 13254839 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_remainder_l2408_240817


namespace NUMINAMATH_CALUDE_bus_stop_time_l2408_240895

/-- Given a bus with speeds excluding and including stoppages, calculate the stop time per hour -/
theorem bus_stop_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 50)
  (h2 : speed_with_stops = 43) :
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 8.4 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_time_l2408_240895


namespace NUMINAMATH_CALUDE_abs_sum_simplification_l2408_240863

theorem abs_sum_simplification (m x : ℝ) (h1 : 0 < m) (h2 : m < 10) (h3 : m ≤ x) (h4 : x ≤ 10) :
  |x - m| + |x - 10| + |x - m - 10| = 20 - x := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_simplification_l2408_240863


namespace NUMINAMATH_CALUDE_heidi_to_danielle_ratio_l2408_240845

/-- The number of rooms in Danielle's apartment -/
def danielles_rooms : ℕ := 6

/-- The number of rooms in Grant's apartment -/
def grants_rooms : ℕ := 2

/-- The ratio of Grant's rooms to Heidi's rooms -/
def grant_to_heidi_ratio : ℚ := 1 / 9

/-- The number of rooms in Heidi's apartment -/
def heidis_rooms : ℕ := grants_rooms * 9

theorem heidi_to_danielle_ratio : 
  (heidis_rooms : ℚ) / danielles_rooms = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_heidi_to_danielle_ratio_l2408_240845


namespace NUMINAMATH_CALUDE_arc_length_for_45_degree_angle_l2408_240886

/-- Given a circle with circumference 90 meters and a central angle of 45°,
    the length of the corresponding arc is 11.25 meters. -/
theorem arc_length_for_45_degree_angle (D : Real) (E F : Real) : 
  D = 90 →  -- circumference of circle D is 90 meters
  (E - F) = 45 * π / 180 →  -- central angle ∠EDF is 45° (converted to radians)
  D * (E - F) / (2 * π) = 11.25 :=  -- length of arc EF
by sorry

end NUMINAMATH_CALUDE_arc_length_for_45_degree_angle_l2408_240886


namespace NUMINAMATH_CALUDE_boat_upstream_speed_l2408_240842

/-- Calculates the upstream speed of a boat given its still water speed and downstream speed -/
def upstream_speed (still_water_speed downstream_speed : ℝ) : ℝ :=
  2 * still_water_speed - downstream_speed

/-- Theorem: Given a boat with a speed of 8.5 km/hr in still water and a downstream speed of 13 km/hr, its upstream speed is 4 km/hr -/
theorem boat_upstream_speed :
  upstream_speed 8.5 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_boat_upstream_speed_l2408_240842


namespace NUMINAMATH_CALUDE_characterize_M_and_m_l2408_240813

-- Define the set S
def S : Set ℝ := {1, 2, 3, 6}

-- Define the set M
def M (m : ℝ) : Set ℝ := {x | x^2 - m*x + 6 = 0}

-- State the theorem
theorem characterize_M_and_m :
  ∀ m : ℝ, (M m ∩ S = M m) →
  ((M m = {2, 3} ∧ m = 7) ∨
   (M m = {1, 6} ∧ m = 5) ∨
   (M m = ∅ ∧ m > -2*Real.sqrt 6 ∧ m < 2*Real.sqrt 6)) :=
by sorry

end NUMINAMATH_CALUDE_characterize_M_and_m_l2408_240813


namespace NUMINAMATH_CALUDE_dot_product_implies_t_l2408_240801

/-- Given vectors a and b in R^2, if their dot product is -2, then the second component of b is -4 -/
theorem dot_product_implies_t (a b : Fin 2 → ℝ) (h : a 0 = 5 ∧ a 1 = -7 ∧ b 0 = -6) :
  (a 0 * b 0 + a 1 * b 1 = -2) → b 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_implies_t_l2408_240801


namespace NUMINAMATH_CALUDE_average_of_9_15_N_l2408_240829

theorem average_of_9_15_N (N : ℝ) (h1 : 12 < N) (h2 : N < 25) :
  let avg := (9 + 15 + N) / 3
  avg = 15 ∨ avg = 17 :=
by sorry

end NUMINAMATH_CALUDE_average_of_9_15_N_l2408_240829


namespace NUMINAMATH_CALUDE_negation_divisible_odd_tan_equality_condition_necessary_not_sufficient_l2408_240812

-- Define divisibility
def divides (a b : ℤ) := ∃ k, b = a * k

-- Define evenness
def even (n : ℤ) := ∃ k, n = 2 * k

-- Statement 1
theorem negation_divisible_odd :
  (∀ x : ℤ, divides 3 x → ¬(even x)) ↔ 
  ¬(∃ x : ℤ, divides 3 x ∧ even x) :=
sorry

-- Statement 2
theorem tan_equality_condition (α β : ℝ) :
  (∃ k : ℤ, α = k * Real.pi + β) ↔ Real.tan α = Real.tan β :=
sorry

-- Statement 3
theorem necessary_not_sufficient (a b : ℝ) :
  (a ≠ 0 → a * b ≠ 0) ∧ ¬(a ≠ 0 ↔ a * b ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_divisible_odd_tan_equality_condition_necessary_not_sufficient_l2408_240812


namespace NUMINAMATH_CALUDE_seven_eighths_of_48_l2408_240821

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_48_l2408_240821


namespace NUMINAMATH_CALUDE_distance_between_foci_l2408_240803

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y + 3)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 24

-- Define the foci
def focus1 : ℝ × ℝ := (2, -3)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem stating the distance between foci
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 4 * Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_distance_between_foci_l2408_240803


namespace NUMINAMATH_CALUDE_imo_42_inequality_l2408_240809

theorem imo_42_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*a*c) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_imo_42_inequality_l2408_240809


namespace NUMINAMATH_CALUDE_second_quadrant_necessary_not_sufficient_for_obtuse_l2408_240882

/-- An angle is in the second quadrant if it's between 90° and 180° or between -270° and -180° --/
def is_second_quadrant_angle (α : Real) : Prop :=
  (90 < α ∧ α ≤ 180) ∨ (-270 < α ∧ α ≤ -180)

/-- An angle is obtuse if it's between 90° and 180° --/
def is_obtuse_angle (α : Real) : Prop :=
  90 < α ∧ α < 180

/-- Theorem stating that "α is a second quadrant angle" is a necessary but not sufficient condition for "α is an obtuse angle" --/
theorem second_quadrant_necessary_not_sufficient_for_obtuse :
  (∀ α : Real, is_obtuse_angle α → is_second_quadrant_angle α) ∧
  (∃ α : Real, is_second_quadrant_angle α ∧ ¬is_obtuse_angle α) :=
by sorry

end NUMINAMATH_CALUDE_second_quadrant_necessary_not_sufficient_for_obtuse_l2408_240882


namespace NUMINAMATH_CALUDE_robot_rascals_shipment_l2408_240897

theorem robot_rascals_shipment (total : ℝ) : 
  (0.7 * total = 168) → total = 240 := by
  sorry

end NUMINAMATH_CALUDE_robot_rascals_shipment_l2408_240897


namespace NUMINAMATH_CALUDE_valid_numbers_l2408_240871

def isValidNumber (n : ℕ) : Prop :=
  n ≥ 500 ∧ n < 1000 ∧
  (n / 100 % 2 = 1) ∧
  ((n / 10) % 10 % 2 = 0) ∧
  (n % 10 % 2 = 0) ∧
  (n / 100 % 3 = 0) ∧
  ((n / 10) % 10 % 3 = 0) ∧
  (n % 10 % 3 ≠ 0)

theorem valid_numbers :
  {n : ℕ | isValidNumber n} = {902, 904, 908, 962, 964, 968} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l2408_240871


namespace NUMINAMATH_CALUDE_sixth_term_value_l2408_240872

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem sixth_term_value (seq : ArithmeticSequence) 
    (first_term : seq.a 1 = 1)
    (sum_5 : seq.S 5 = 15) :
  seq.a 6 = 6 := by
  sorry


end NUMINAMATH_CALUDE_sixth_term_value_l2408_240872


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2408_240808

theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, (a * (1 + Complex.I)) * x^2 + (1 + a^2 * Complex.I) * x + (a^2 + Complex.I) = 0) ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2408_240808


namespace NUMINAMATH_CALUDE_ascending_order_proof_l2408_240853

theorem ascending_order_proof : 222^2 < 22^22 ∧ 22^22 < 2^222 := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_proof_l2408_240853


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2408_240838

theorem quadratic_equation_properties (m : ℝ) :
  m < 4 →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4*x₁ + m = 0 ∧ x₂^2 - 4*x₂ + m = 0) ∧
  ((-1)^2 - 4*(-1) + m = 0 → m = -5 ∧ 5^2 - 4*5 + m = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2408_240838


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2408_240837

/-- The repeating decimal 0.567567567... expressed as a rational number -/
def repeating_decimal : ℚ := 567 / 999

theorem repeating_decimal_equals_fraction : repeating_decimal = 21 / 37 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2408_240837


namespace NUMINAMATH_CALUDE_calculation_proof_l2408_240888

theorem calculation_proof :
  (5 / (-5/3) * (-2) = 6) ∧
  (-(1^2) + 3 * (-2)^2 + (-9) / (-1/3)^2 = -70) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2408_240888


namespace NUMINAMATH_CALUDE_rectangle_area_l2408_240833

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2

/-- Represents a rectangle defined by four vertices -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Theorem: Area of a specific rectangle -/
theorem rectangle_area (rect : Rectangle)
  (h1 : rect.P = ⟨1, 1⟩)
  (h2 : rect.Q = ⟨-3, 2⟩)
  (h3 : rect.R = ⟨-1, 6⟩)
  (h4 : rect.S = ⟨3, 5⟩)
  (h5 : squaredDistance rect.P rect.Q = squaredDistance rect.R rect.S) -- PQ is one side
  (h6 : squaredDistance rect.P rect.R = squaredDistance rect.Q rect.S) -- PR is a diagonal
  : (squaredDistance rect.P rect.Q * squaredDistance rect.P rect.R : ℝ) = 4 * 51 :=
sorry


end NUMINAMATH_CALUDE_rectangle_area_l2408_240833


namespace NUMINAMATH_CALUDE_find_x_in_ratio_l2408_240834

/-- Given t = 5, prove that the positive integer x satisfying 2 : m : t = m : 32 : x is 20 -/
theorem find_x_in_ratio (t : ℕ) (h_t : t = 5) :
  ∃ (m : ℤ) (x : ℕ), 2 * 32 * t = m * m * x ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_find_x_in_ratio_l2408_240834


namespace NUMINAMATH_CALUDE_kwik_e_tax_center_problem_l2408_240864

/-- The Kwik-e-Tax Center problem -/
theorem kwik_e_tax_center_problem 
  (federal_charge : ℕ) 
  (state_charge : ℕ) 
  (quarterly_charge : ℕ)
  (federal_sold : ℕ) 
  (state_sold : ℕ) 
  (total_revenue : ℕ)
  (h1 : federal_charge = 50)
  (h2 : state_charge = 30)
  (h3 : quarterly_charge = 80)
  (h4 : federal_sold = 60)
  (h5 : state_sold = 20)
  (h6 : total_revenue = 4400) :
  ∃ (quarterly_sold : ℕ), 
    federal_charge * federal_sold + 
    state_charge * state_sold + 
    quarterly_charge * quarterly_sold = total_revenue ∧ 
    quarterly_sold = 10 := by
  sorry

end NUMINAMATH_CALUDE_kwik_e_tax_center_problem_l2408_240864


namespace NUMINAMATH_CALUDE_smallest_base_for_mnmn_cube_mnmn_cube_in_base_seven_smallest_base_is_seven_l2408_240874

def is_mnmn (n : ℕ) (b : ℕ) : Prop :=
  ∃ m n : ℕ, m < b ∧ n < b ∧ n = m * (b^3 + b) + n * (b^2 + 1)

def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^3

theorem smallest_base_for_mnmn_cube :
  ∀ b : ℕ, b > 1 →
    (∃ n : ℕ, is_mnmn n b ∧ is_cube n) →
    b ≥ 7 :=
by sorry

theorem mnmn_cube_in_base_seven :
  ∃ n : ℕ, is_mnmn n 7 ∧ is_cube n :=
by sorry

theorem smallest_base_is_seven :
  (∀ b : ℕ, b > 1 → b < 7 → ¬∃ n : ℕ, is_mnmn n b ∧ is_cube n) ∧
  (∃ n : ℕ, is_mnmn n 7 ∧ is_cube n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_mnmn_cube_mnmn_cube_in_base_seven_smallest_base_is_seven_l2408_240874


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l2408_240822

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^13 + i^18 + i^23 + i^28 + i^33 = i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l2408_240822


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2408_240846

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2408_240846


namespace NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_433_over_18_l2408_240860

theorem sqrt_fraction_sum_equals_sqrt_433_over_18 :
  Real.sqrt (25 / 36 + 16 / 81 + 4 / 9) = Real.sqrt 433 / 18 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_433_over_18_l2408_240860


namespace NUMINAMATH_CALUDE_quadratic_zero_discriminant_geometric_progression_l2408_240881

/-- 
Given a quadratic equation ax^2 + bx + c = 0 with zero discriminant,
prove that a, b, and c form a geometric progression.
-/
theorem quadratic_zero_discriminant_geometric_progression 
  (a b c : ℝ) (h : b^2 - 4*a*c = 0) : 
  ∃ (r : ℝ), b = a*r ∧ c = b*r :=
sorry

end NUMINAMATH_CALUDE_quadratic_zero_discriminant_geometric_progression_l2408_240881
