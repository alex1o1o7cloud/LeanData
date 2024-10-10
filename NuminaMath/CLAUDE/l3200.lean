import Mathlib

namespace volleyball_team_selection_l3200_320031

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

theorem volleyball_team_selection (total_players starters : ℕ) (twins : ℕ) : 
  total_players = 16 → 
  starters = 6 → 
  twins = 2 →
  (choose (total_players - twins) (starters - twins) + 
   choose (total_players - twins) starters) = 4004 := by
  sorry

end volleyball_team_selection_l3200_320031


namespace base_5_representation_of_425_l3200_320056

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

theorem base_5_representation_of_425 :
  toBase5 425 = [3, 2, 0, 0] :=
sorry

end base_5_representation_of_425_l3200_320056


namespace rearranged_prism_surface_area_l3200_320051

structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

def cut_heights : List ℝ := [0.6, 0.3, 0.05, 0.05]

def surface_area (prism : RectangularPrism) (cuts : List ℝ) : ℝ :=
  2 * (prism.length * prism.width + prism.length * prism.height + prism.width * prism.height)

theorem rearranged_prism_surface_area :
  let original_prism : RectangularPrism := { length := 2, width := 2, height := 1 }
  surface_area original_prism cut_heights = 20 := by
  sorry

end rearranged_prism_surface_area_l3200_320051


namespace root_exists_in_interval_l3200_320093

theorem root_exists_in_interval :
  ∃ x : ℝ, 3/2 < x ∧ x < 2 ∧ 2^x = x^2 + 1/2 := by
  sorry

end root_exists_in_interval_l3200_320093


namespace tg_plus_ctg_l3200_320068

theorem tg_plus_ctg (x : ℝ) (h : (1 / Real.cos x) - (1 / Real.sin x) = Real.sqrt 35) :
  (Real.tan x + (1 / Real.tan x) = 7) ∨ (Real.tan x + (1 / Real.tan x) = -5) := by
sorry

end tg_plus_ctg_l3200_320068


namespace sum_of_leading_digits_is_seven_l3200_320066

/-- N is a 200-digit number where each digit is 8 -/
def N : ℕ := 8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888

/-- f(r) is the leading digit of the r-th root of N -/
def f (r : ℕ) : ℕ := sorry

/-- The sum of f(r) for r from 3 to 7 is 7 -/
theorem sum_of_leading_digits_is_seven :
  f 3 + f 4 + f 5 + f 6 + f 7 = 7 := by sorry

end sum_of_leading_digits_is_seven_l3200_320066


namespace ratio_problem_l3200_320018

theorem ratio_problem (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 1 / 4) :
  e / f = 9 / 4 := by
sorry

end ratio_problem_l3200_320018


namespace rational_opposite_and_number_line_order_l3200_320054

-- Define the concept of opposite for rational numbers
def opposite (a : ℚ) : ℚ := -a

-- Define a property for the order of numbers on a number line
def left_of (x y : ℝ) : Prop := x < y

theorem rational_opposite_and_number_line_order :
  (∀ a : ℚ, opposite a = -a) ∧
  (∀ x y : ℝ, x ≠ y → (left_of x y ↔ x < y)) :=
sorry

end rational_opposite_and_number_line_order_l3200_320054


namespace probability_theorem_l3200_320052

-- Define the number of doctors and cities
def num_doctors : ℕ := 5
def num_cities : ℕ := 3

-- Define a function to calculate the probability
def probability_one_doctor_one_city (n_doctors : ℕ) (n_cities : ℕ) : ℚ :=
  7/75

-- State the theorem
theorem probability_theorem :
  probability_one_doctor_one_city num_doctors num_cities = 7/75 :=
by sorry

end probability_theorem_l3200_320052


namespace phone_number_remainder_l3200_320070

theorem phone_number_remainder :
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧
  (312837 % n = 96) ∧ (310650 % n = 96) := by
  sorry

end phone_number_remainder_l3200_320070


namespace deer_distribution_l3200_320033

theorem deer_distribution (a : ℚ) (d : ℚ) : 
  (5 * a + 10 * d = 5) →  -- Sum of 5 terms equals 5
  (a + 3 * d = 2/3) →     -- Fourth term is 2/3
  a = 1/3 :=              -- First term (Gong Shi's share) is 1/3
by
  sorry

end deer_distribution_l3200_320033


namespace quadratic_roots_l3200_320022

-- Define the quadratic expression
def quadratic (c : ℝ) (x : ℝ) : ℝ := -x^2 + c*x + 8

-- State the theorem
theorem quadratic_roots (c : ℝ) : 
  (∀ x, quadratic c x > 0 ↔ x < -2 ∨ x > 4) → c = 2 := by
  sorry

end quadratic_roots_l3200_320022


namespace smallest_four_digit_divisible_by_35_l3200_320005

theorem smallest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 → n ≥ 1015 :=
by sorry

end smallest_four_digit_divisible_by_35_l3200_320005


namespace usd_share_change_l3200_320042

/-- The change in the share of the US dollar in the NWF from 01.02.2021 to 01.04.2021 -/
theorem usd_share_change (total_nwf : ℝ) (other_currencies : ℝ) (initial_usd_share : ℝ) :
  total_nwf = 794.26 →
  other_currencies = 34.72 + 8.55 + 600.3 + 110.54 + 0.31 →
  initial_usd_share = 49.17 →
  ∃ (ε : ℝ), abs ε < 0.5 ∧
    (((total_nwf - other_currencies) / total_nwf * 100 - initial_usd_share) + ε = -44) := by
  sorry

end usd_share_change_l3200_320042


namespace daniel_purchase_cost_l3200_320098

/-- The total amount spent on a magazine and pencil after applying a coupon discount -/
def total_spent (magazine_cost pencil_cost coupon_discount : ℚ) : ℚ :=
  magazine_cost + pencil_cost - coupon_discount

/-- Theorem stating that given specific costs and discount, the total spent is $1.00 -/
theorem daniel_purchase_cost :
  total_spent 0.85 0.50 0.35 = 1.00 := by
  sorry

end daniel_purchase_cost_l3200_320098


namespace complement_intersection_equality_l3200_320008

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {2, 4}
def N : Set Nat := {3, 5}

theorem complement_intersection_equality :
  (U \ M) ∩ N = {3, 5} := by sorry

end complement_intersection_equality_l3200_320008


namespace number_transformation_l3200_320036

theorem number_transformation (x : ℕ) : x = 5 → 3 * (2 * x + 9) = 57 := by
  sorry

end number_transformation_l3200_320036


namespace joe_school_travel_time_l3200_320073

-- Define the variables
def walking_speed : ℝ := 1 -- Arbitrary unit speed
def running_speed : ℝ := 3 * walking_speed
def walking_time : ℝ := 9 -- minutes
def break_time : ℝ := 1 -- minute

-- Define the theorem
theorem joe_school_travel_time :
  let running_time := walking_time / 3
  walking_time + break_time + running_time = 13 := by
  sorry

end joe_school_travel_time_l3200_320073


namespace problem_2013_l3200_320028

theorem problem_2013 : 
  (2013^3 - 2 * 2013^2 * 2014 + 3 * 2013 * 2014^2 - 2014^3 + 1) / (2013 * 2014) = 2013 := by
  sorry

end problem_2013_l3200_320028


namespace breakfast_calories_l3200_320099

def total_calories : ℕ := 2200
def lunch_calories : ℕ := 885
def snack_calories : ℕ := 130
def dinner_calories : ℕ := 832

theorem breakfast_calories : 
  total_calories - lunch_calories - snack_calories - dinner_calories = 353 := by
sorry

end breakfast_calories_l3200_320099


namespace infinitely_many_commuting_functions_l3200_320013

/-- A bijective function from ℝ to ℝ -/
def BijectiveFunc := {f : ℝ → ℝ // Function.Bijective f}

/-- The set of functions g that satisfy f(g(x)) = g(f(x)) for all x -/
def CommutingFunctions (f : BijectiveFunc) :=
  {g : ℝ → ℝ | ∀ x, f.val (g x) = g (f.val x)}

/-- The theorem stating that there are infinitely many commuting functions -/
theorem infinitely_many_commuting_functions (f : BijectiveFunc) :
  Set.Infinite (CommutingFunctions f) := by
  sorry

end infinitely_many_commuting_functions_l3200_320013


namespace midpoint_trajectory_of_moving_chord_l3200_320071

/-- Given a circle and a moving chord, prove the equation of the midpoint's trajectory -/
theorem midpoint_trajectory_of_moving_chord 
  (x y : ℝ) (M : ℝ × ℝ) : 
  (∀ (C D : ℝ × ℝ), 
    (C.1^2 + C.2^2 = 25) ∧ 
    (D.1^2 + D.2^2 = 25) ∧ 
    ((C.1 - D.1)^2 + (C.2 - D.2)^2 = 64) ∧ 
    (M = ((C.1 + D.1)/2, (C.2 + D.2)/2))) →
  (M.1^2 + M.2^2 = 9) := by
sorry

end midpoint_trajectory_of_moving_chord_l3200_320071


namespace function_extrema_implies_a_range_l3200_320034

-- Define the function f(x) = x^2 - 2x + 3
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the theorem
theorem function_extrema_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 a, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 2) ↔
  a ∈ Set.Icc 1 2 :=
sorry

end function_extrema_implies_a_range_l3200_320034


namespace minimizes_y_l3200_320069

/-- The function y in terms of x, a, b, and c -/
def y (x a b c : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + (x - c)^2

/-- The theorem stating that (a + b + c) / 3 minimizes y -/
theorem minimizes_y (a b c : ℝ) :
  let x_min : ℝ := (a + b + c) / 3
  ∀ x : ℝ, y x_min a b c ≤ y x a b c :=
sorry

end minimizes_y_l3200_320069


namespace moores_law_2010_l3200_320014

/-- Moore's law doubling period in months -/
def doubling_period : ℕ := 18

/-- Initial number of transistors in 1995 -/
def initial_transistors : ℕ := 2500000

/-- Number of months between 1995 and 2010 -/
def months_elapsed : ℕ := (2010 - 1995) * 12

/-- Number of doublings that occurred between 1995 and 2010 -/
def num_doublings : ℕ := months_elapsed / doubling_period

/-- Calculates the number of transistors after a given number of doublings -/
def transistors_after_doublings (initial : ℕ) (doublings : ℕ) : ℕ :=
  initial * (2^doublings)

theorem moores_law_2010 :
  transistors_after_doublings initial_transistors num_doublings = 2560000000 := by
  sorry

end moores_law_2010_l3200_320014


namespace root_exists_in_interval_l3200_320075

-- Define the function f(x) = 4x³ - 5x + 6
def f (x : ℝ) : ℝ := 4 * x^3 - 5 * x + 6

-- State the theorem
theorem root_exists_in_interval :
  ∃ x ∈ Set.Ioo (-2 : ℝ) (-1 : ℝ), f x = 0 := by
  sorry

end root_exists_in_interval_l3200_320075


namespace correct_statements_l3200_320072

-- Define the statements
def statement1 : Prop := False
def statement2 : Prop := False
def statement3 : Prop := True
def statement4 : Prop := True

-- Define the regression line
def regression_line (x : ℝ) : ℝ := 0.1 * x + 10

-- Theorem to prove
theorem correct_statements :
  (statement3 ∧ statement4) ∧ 
  (¬statement1 ∧ ¬statement2) ∧
  (∀ x y : ℝ, y = regression_line x → regression_line (x + 1) = y + 0.1) :=
sorry

end correct_statements_l3200_320072


namespace negation_equivalence_l3200_320047

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) := by
  sorry

end negation_equivalence_l3200_320047


namespace larger_integer_value_l3200_320004

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 294) :
  (a : ℝ) = 7 * Real.sqrt 14 := by
  sorry

end larger_integer_value_l3200_320004


namespace magazines_sold_l3200_320084

theorem magazines_sold (total : ℝ) (newspapers : ℝ) (h1 : total = 425.0) (h2 : newspapers = 275.0) :
  total - newspapers = 150.0 := by
  sorry

end magazines_sold_l3200_320084


namespace prob_different_ranks_l3200_320059

def total_cards : ℕ := 5
def ace_count : ℕ := 1
def king_count : ℕ := 2
def queen_count : ℕ := 2

def different_ranks_probability : ℚ := 4/5

theorem prob_different_ranks :
  let total_combinations := total_cards.choose 2
  let same_rank_combinations := king_count.choose 2 + queen_count.choose 2
  (total_combinations - same_rank_combinations : ℚ) / total_combinations = different_ranks_probability :=
sorry

end prob_different_ranks_l3200_320059


namespace geometric_sequence_sum_l3200_320085

/-- Given a geometric sequence where S_n = 48 and S_2n = 60, prove that S_3n = 63 -/
theorem geometric_sequence_sum (S : ℕ → ℝ) (n : ℕ) 
  (h1 : S n = 48) 
  (h2 : S (2 * n) = 60) 
  (h_geometric : ∀ k : ℕ, S (k + 1) / S k = S (k + 2) / S (k + 1)) :
  S (3 * n) = 63 := by
  sorry

end geometric_sequence_sum_l3200_320085


namespace not_same_size_and_precision_l3200_320030

/-- Represents the precision of a decimal number -/
inductive Precision
| Tenths
| Hundredths

/-- Represents a decimal number with its value and precision -/
structure DecimalNumber where
  value : ℚ
  precision : Precision

/-- Check if two DecimalNumbers have the same size and precision -/
def sameSizeAndPrecision (a b : DecimalNumber) : Prop :=
  a.value = b.value ∧ a.precision = b.precision

theorem not_same_size_and_precision :
  ¬(sameSizeAndPrecision
    { value := 1.2, precision := Precision.Hundredths }
    { value := 1.2, precision := Precision.Tenths }) := by
  sorry

end not_same_size_and_precision_l3200_320030


namespace work_completion_time_relation_l3200_320082

/-- Given a constant amount of work, if 100 workers complete it in 5 days,
    then 40 workers will complete it in 12.5 days. -/
theorem work_completion_time_relation :
  ∀ (total_work : ℝ),
    total_work > 0 →
    ∃ (worker_rate : ℝ),
      worker_rate > 0 ∧
      total_work = 100 * worker_rate * 5 →
      total_work = 40 * worker_rate * 12.5 :=
by sorry

end work_completion_time_relation_l3200_320082


namespace symmetry_shift_l3200_320061

/-- Given a function f(x) = √3 cos x - sin x, this theorem states that
    the smallest positive value of θ such that the graph of f(x-θ) is
    symmetrical about the line x = π/6 is π/3. -/
theorem symmetry_shift (f : ℝ → ℝ) (h : ∀ x, f x = Real.sqrt 3 * Real.cos x - Real.sin x) :
  ∃ θ : ℝ, θ > 0 ∧
    (∀ θ' > 0, (∀ x, f (x - θ') = f (π / 3 - (x - π / 6))) → θ ≤ θ') ∧
    (∀ x, f (x - θ) = f (π / 3 - (x - π / 6))) ∧
    θ = π / 3 := by
  sorry

end symmetry_shift_l3200_320061


namespace cuboid_faces_at_vertex_l3200_320019

/-- A cuboid is a three-dimensional shape with six rectangular faces. -/
structure Cuboid where
  -- We don't need to define the specific properties of a cuboid for this problem

/-- The number of faces meeting at one vertex of a cuboid -/
def faces_at_vertex (c : Cuboid) : ℕ := 3

/-- Theorem: The number of faces meeting at one vertex of a cuboid is 3 -/
theorem cuboid_faces_at_vertex (c : Cuboid) : faces_at_vertex c = 3 := by
  sorry

end cuboid_faces_at_vertex_l3200_320019


namespace flagpole_height_l3200_320021

/-- Given a flagpole and a building under similar shadow-casting conditions,
    calculate the height of the flagpole. -/
theorem flagpole_height
  (flagpole_shadow : ℝ)
  (building_height : ℝ)
  (building_shadow : ℝ)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_height : building_height = 22)
  (h_building_shadow : building_shadow = 55)
  : ∃ (flagpole_height : ℝ),
    flagpole_height / flagpole_shadow = building_height / building_shadow ∧
    flagpole_height = 18 := by
  sorry

end flagpole_height_l3200_320021


namespace arithmetic_sequence_fifth_term_l3200_320027

/-- Given an arithmetic sequence with first term 3 and second term 7,
    prove that the 5th term is 19. -/
theorem arithmetic_sequence_fifth_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 3 →                            -- first term is 3
    a 1 = 7 →                            -- second term is 7
    a 4 = 19 :=                          -- 5th term (index 4) is 19
by
  sorry

end arithmetic_sequence_fifth_term_l3200_320027


namespace triangle_division_possibility_l3200_320015

/-- Given a triangle ABC with sides a, b, c (where c > b > a), it is possible to construct 
    a line that divides the triangle into a quadrilateral with 2/3 of the triangle's area 
    and a smaller triangle with 1/3 of the area, if and only if c ≤ 3a. -/
theorem triangle_division_possibility (a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) :
  (∃ (x y : ℝ), 0 < x ∧ x < c ∧ 0 < y ∧ y < b ∧ 
    (x * y) / 2 = (2/3) * (a * b) / 2) ↔ c ≤ 3 * a :=
by sorry

end triangle_division_possibility_l3200_320015


namespace cover_ways_2x13_l3200_320046

/-- The number of ways to cover a 2 × n rectangular board with 1 × 2 tiles -/
def cover_ways : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| (n + 3) => cover_ways (n + 2) + cover_ways (n + 1)

/-- Tiles of size 1 × 2 -/
structure Tile :=
  (width : ℕ := 1)
  (height : ℕ := 2)

/-- A 2 × 13 rectangular board -/
structure Board :=
  (width : ℕ := 2)
  (height : ℕ := 13)

/-- Theorem: The number of ways to cover a 2 × 13 board with 1 × 2 tiles is 377 -/
theorem cover_ways_2x13 : cover_ways 13 = 377 := by
  sorry

end cover_ways_2x13_l3200_320046


namespace equiangular_parallelogram_iff_rectangle_l3200_320009

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
structure Parallelogram :=
  (is_parallel : Bool)

/-- An equiangular parallelogram is a parallelogram with all angles equal. -/
structure EquiangularParallelogram extends Parallelogram :=
  (all_angles_equal : Bool)

/-- A rectangle is a parallelogram with all right angles. -/
structure Rectangle extends Parallelogram :=
  (all_angles_right : Bool)

/-- Theorem: A parallelogram is equiangular if and only if it is a rectangle. -/
theorem equiangular_parallelogram_iff_rectangle :
  ∀ p : Parallelogram, (∃ ep : EquiangularParallelogram, ep.toParallelogram = p) ↔ (∃ r : Rectangle, r.toParallelogram = p) :=
sorry

end equiangular_parallelogram_iff_rectangle_l3200_320009


namespace task_force_count_l3200_320041

theorem task_force_count (total : ℕ) (executives : ℕ) (task_force_size : ℕ) (min_executives : ℕ) :
  total = 12 →
  executives = 5 →
  task_force_size = 5 →
  min_executives = 2 →
  (Nat.choose total task_force_size -
   (Nat.choose (total - executives) task_force_size +
    Nat.choose executives 1 * Nat.choose (total - executives) (task_force_size - 1))) = 596 := by
  sorry

end task_force_count_l3200_320041


namespace no_two_digit_factors_of_1976_l3200_320017

theorem no_two_digit_factors_of_1976 : 
  ¬∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ a * b = 1976 := by
sorry

end no_two_digit_factors_of_1976_l3200_320017


namespace polynomial_remainder_l3200_320026

/-- The remainder when x^4 - 8x^3 + 5x^2 + 22x - 7 is divided by x-4 is -95 -/
theorem polynomial_remainder : 
  (fun x : ℝ => x^4 - 8*x^3 + 5*x^2 + 22*x - 7) 4 = -95 := by
sorry

end polynomial_remainder_l3200_320026


namespace expression_evaluation_l3200_320010

theorem expression_evaluation :
  let x : ℚ := 1/4
  let y : ℚ := 1/2
  let z : ℚ := 3
  4 * (x^3 * y^2 * z^2) = 9/64 := by
sorry

end expression_evaluation_l3200_320010


namespace hyperbola_eccentricity_l3200_320011

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) : 
  a > 0 → b > 0 → 
  a^2 + b^2 = c^2 → 
  b^2 * c^2 = a^2 * (b^2 + c^2) → 
  c / a = (1 + Real.sqrt 5) / 2 := by
  sorry

end hyperbola_eccentricity_l3200_320011


namespace geometric_sequence_a7_l3200_320045

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 2 * a 4 * a 5 = a 3 * a 6 →
  a 9 * a 10 = -8 →
  a 7 = -2 := by
sorry

end geometric_sequence_a7_l3200_320045


namespace vector_magnitude_proof_l3200_320063

/-- Given two vectors a and b in ℝ², prove that under certain conditions, 
    the magnitude of 2a + 3b equals √91. -/
theorem vector_magnitude_proof (a b : ℝ × ℝ) : 
  a = (4, -3) → 
  ‖b‖ = 3 → 
  (a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖) = -1/2 → 
  ‖2 • a + 3 • b‖ = Real.sqrt 91 := by
  sorry

end vector_magnitude_proof_l3200_320063


namespace lowest_hundred_year_flood_level_l3200_320090

/-- Represents the frequency distribution of water levels -/
structure WaterLevelDistribution where
  -- Add necessary fields to represent the distribution
  -- This is a simplified representation
  lowest_hundred_year_flood : ℝ

/-- The hydrological observation point data -/
def observation_point : WaterLevelDistribution :=
  { lowest_hundred_year_flood := 50 }

/-- Theorem stating the lowest water level of the hundred-year flood -/
theorem lowest_hundred_year_flood_level :
  observation_point.lowest_hundred_year_flood = 50 := by
  sorry

#check lowest_hundred_year_flood_level

end lowest_hundred_year_flood_level_l3200_320090


namespace sqrt_two_times_sqrt_half_equals_one_l3200_320079

theorem sqrt_two_times_sqrt_half_equals_one :
  Real.sqrt 2 * Real.sqrt (1/2) = 1 := by
  sorry

end sqrt_two_times_sqrt_half_equals_one_l3200_320079


namespace root_sum_theorem_l3200_320074

theorem root_sum_theorem (a b c d r : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (r - a) * (r - b) * (r - c) * (r - d) = 4 →
  4 * r = a + b + c + d := by
sorry

end root_sum_theorem_l3200_320074


namespace triangle_obtuse_from_trig_inequality_l3200_320038

/-- Given a triangle ABC, if sin A * sin B < cos A * cos B, then ABC is an obtuse triangle -/
theorem triangle_obtuse_from_trig_inequality (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_inequality : Real.sin A * Real.sin B < Real.cos A * Real.cos B) : 
  π / 2 < C ∧ C < π :=
sorry

end triangle_obtuse_from_trig_inequality_l3200_320038


namespace three_lines_intersection_l3200_320077

/-- Three lines intersecting at a point -/
structure ThreeLines where
  a : ℝ
  l₁ : ℝ → ℝ → Prop
  l₂ : ℝ → ℝ → Prop
  l₃ : ℝ → ℝ → Prop
  h₁ : l₁ x y ↔ a * x + 2 * y + 6 = 0
  h₂ : l₂ x y ↔ x + y - 4 = 0
  h₃ : l₃ x y ↔ 2 * x - y + 1 = 0
  intersection : ∃ (x y : ℝ), l₁ x y ∧ l₂ x y ∧ l₃ x y

/-- The value of a when three lines intersect at a point -/
theorem three_lines_intersection (t : ThreeLines) : t.a = -12 := by
  sorry

end three_lines_intersection_l3200_320077


namespace factorization_equality_l3200_320040

theorem factorization_equality (x : ℝ) : 3 * x^2 * (x - 5) + 5 * (x - 5) = (3 * x^2 + 5) * (x - 5) := by
  sorry

end factorization_equality_l3200_320040


namespace four_digit_number_puzzle_l3200_320095

theorem four_digit_number_puzzle :
  ∀ (A B x y : ℕ),
    1000 ≤ A ∧ A < 10000 →
    0 ≤ x ∧ x < 10 →
    0 ≤ y ∧ y < 10 →
    B = 100000 * x + 10 * A + y →
    B = 21 * A →
    A = 9091 ∧ B = 190911 := by
  sorry

end four_digit_number_puzzle_l3200_320095


namespace second_number_value_l3200_320039

theorem second_number_value (x y z : ℝ) (h1 : x + y + z = 120) (h2 : x / y = 3 / 4) (h3 : y / z = 7 / 9) :
  ∃ (n : ℕ), n = 40 ∧ abs (y - n) ≤ 1/2 := by
  sorry

end second_number_value_l3200_320039


namespace count_differing_blocks_l3200_320078

/-- Represents the properties of a block -/
structure BlockProperties where
  material : Fin 3
  size : Fin 3
  color : Fin 4
  shape : Fin 5

/-- The reference block: metal medium blue hexagon -/
def referenceBlock : BlockProperties := {
  material := 2, -- Assuming 2 represents metal
  size := 1,     -- Assuming 1 represents medium
  color := 0,    -- Assuming 0 represents blue
  shape := 1     -- Assuming 1 represents hexagon
}

/-- Count of blocks differing in exactly two properties -/
def countDifferingBlocks : Nat :=
  let materialOptions := 2  -- Excluding the reference material
  let sizeOptions := 2      -- Excluding the reference size
  let colorOptions := 3     -- Excluding the reference color
  let shapeOptions := 4     -- Excluding the reference shape
  
  -- Sum of all combinations of choosing 2 properties to vary
  materialOptions * sizeOptions +
  materialOptions * colorOptions +
  materialOptions * shapeOptions +
  sizeOptions * colorOptions +
  sizeOptions * shapeOptions +
  colorOptions * shapeOptions

/-- Theorem stating the count of blocks differing in exactly two properties -/
theorem count_differing_blocks :
  countDifferingBlocks = 44 := by
  sorry


end count_differing_blocks_l3200_320078


namespace nth_terms_equal_condition_l3200_320023

/-- 
Given two arithmetic progressions with n terms, where the sum of the first progression 
is n^2 + pn and the sum of the second progression is 3n^2 - 2n, this theorem states the 
condition for their n-th terms to be equal.
-/
theorem nth_terms_equal_condition (n : ℕ) (p : ℝ) 
  (sum1 : ℝ → ℝ → ℝ) (sum2 : ℝ → ℝ) 
  (h1 : sum1 n p = n^2 + p*n) 
  (h2 : sum2 n = 3*n^2 - 2*n) : 
  (∃ (a1 b1 d e : ℝ), 
    (∀ k : ℕ, k > 0 → k ≤ n → 
      sum1 n p = (n : ℝ)/2 * (a1 + (a1 + (n - 1)*d))) ∧
    (∀ k : ℕ, k > 0 → k ≤ n → 
      sum2 n = (n : ℝ)/2 * (b1 + (b1 + (n - 1)*e))) ∧
    a1 + (n - 1)*d = b1 + (n - 1)*e) ↔ 
  p = 4*(n - 1) := by
sorry

end nth_terms_equal_condition_l3200_320023


namespace unique_four_digit_reverse_9multiple_l3200_320087

/-- Reverses a four-digit number -/
def reverse (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := n / 1000
  d0 * 1000 + d1 * 100 + d2 * 10 + d3

/-- A four-digit number is a natural number between 1000 and 9999 -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem unique_four_digit_reverse_9multiple :
  ∃! n : ℕ, is_four_digit n ∧ 9 * n = reverse n :=
by
  -- The proof goes here
  sorry

#eval reverse 1089  -- Expected output: 9801
#eval 9 * 1089      -- Expected output: 9801

end unique_four_digit_reverse_9multiple_l3200_320087


namespace solution_set_inequality_l3200_320060

theorem solution_set_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  {x : ℝ | -b < 1/x ∧ 1/x < a} = {x : ℝ | x < -1/b ∨ x > 1/a} := by sorry

end solution_set_inequality_l3200_320060


namespace contribution_ratio_l3200_320053

-- Define the contributions and profit
def robi_contribution : ℝ := 4000
def rudy_contribution : ℝ → ℝ := λ x => robi_contribution + x
def total_contribution : ℝ → ℝ := λ x => robi_contribution + rudy_contribution x
def profit_rate : ℝ := 0.20
def individual_profit : ℝ := 900

-- Define the theorem
theorem contribution_ratio :
  ∃ x : ℝ,
    x > 0 ∧
    profit_rate * total_contribution x = 2 * individual_profit ∧
    rudy_contribution x / robi_contribution = 5 / 4 := by
  sorry


end contribution_ratio_l3200_320053


namespace no_real_solution_for_inequality_l3200_320055

theorem no_real_solution_for_inequality :
  ¬∃ (x : ℝ), 3 * x^2 + 9 * x ≤ -12 := by sorry

end no_real_solution_for_inequality_l3200_320055


namespace probability_of_exactly_three_primes_out_of_five_dice_l3200_320096

def is_prime (n : ℕ) : Prop := sorry

def number_of_primes_up_to_20 : ℕ := 8

def probability_of_prime_on_20_sided_die : ℚ := 
  number_of_primes_up_to_20 / 20

def number_of_dice : ℕ := 5

def number_of_dice_showing_prime : ℕ := 3

theorem probability_of_exactly_three_primes_out_of_five_dice : 
  (Nat.choose number_of_dice number_of_dice_showing_prime : ℚ) * 
  (probability_of_prime_on_20_sided_die ^ number_of_dice_showing_prime) *
  ((1 - probability_of_prime_on_20_sided_die) ^ (number_of_dice - number_of_dice_showing_prime)) = 
  5 / 16 :=
sorry

end probability_of_exactly_three_primes_out_of_five_dice_l3200_320096


namespace largest_three_digit_square_base_9_l3200_320020

/-- The largest integer whose square has exactly 3 digits when written in base 9 -/
def N : ℕ := 26

/-- Condition for a number to have exactly 3 digits in base 9 -/
def has_three_digits_base_9 (n : ℕ) : Prop :=
  9^2 ≤ n^2 ∧ n^2 < 9^3

/-- Convert a natural number to its base 9 representation -/
def to_base_9 (n : ℕ) : ℕ :=
  (n / 9) * 10 + (n % 9)

theorem largest_three_digit_square_base_9 :
  (N = 26) ∧
  (has_three_digits_base_9 N) ∧
  (∀ m : ℕ, m > N → ¬(has_three_digits_base_9 m)) ∧
  (to_base_9 N = 28) :=
sorry

end largest_three_digit_square_base_9_l3200_320020


namespace mono_increasing_implies_g_neg_one_lt_g_one_l3200_320062

-- Define a monotonically increasing function
def MonotonicallyIncreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → g x < g y

-- Theorem statement
theorem mono_increasing_implies_g_neg_one_lt_g_one
    (g : ℝ → ℝ) (h : MonotonicallyIncreasing g) :
    g (-1) < g 1 := by
  sorry

end mono_increasing_implies_g_neg_one_lt_g_one_l3200_320062


namespace am_gm_inequality_two_terms_l3200_320024

theorem am_gm_inequality_two_terms (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end am_gm_inequality_two_terms_l3200_320024


namespace liquid_mixture_problem_l3200_320097

/-- Proves that the initial amount of liquid A is 21 litres given the conditions of the problem -/
theorem liquid_mixture_problem (initial_ratio_A : ℚ) (initial_ratio_B : ℚ) 
  (drawn_off : ℚ) (new_ratio_A : ℚ) (new_ratio_B : ℚ) :
  initial_ratio_A = 7 ∧ 
  initial_ratio_B = 5 ∧ 
  drawn_off = 9 ∧ 
  new_ratio_A = 7 ∧ 
  new_ratio_B = 9 → 
  ∃ (x : ℚ), 
    7 * x = 21 ∧ 
    (7 * x - (7 / 12) * drawn_off) / (5 * x - (5 / 12) * drawn_off + drawn_off) = new_ratio_A / new_ratio_B :=
by sorry

end liquid_mixture_problem_l3200_320097


namespace goods_train_speed_l3200_320083

theorem goods_train_speed 
  (speed_A : ℝ) 
  (length_B : ℝ) 
  (passing_time : ℝ) 
  (h1 : speed_A = 70) 
  (h2 : length_B = 0.45) 
  (h3 : passing_time = 15 / 3600) : 
  ∃ (speed_B : ℝ), speed_B = 38 := by
sorry

end goods_train_speed_l3200_320083


namespace max_successful_teams_l3200_320032

/-- Represents a football tournament --/
structure Tournament :=
  (num_teams : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Calculate the maximum possible points for a single team in the tournament --/
def max_points (t : Tournament) : ℕ :=
  (t.num_teams - 1) * t.points_for_win

/-- Calculate the minimum points required for a team to be considered successful --/
def min_success_points (t : Tournament) : ℕ :=
  (max_points t + 1) / 2

/-- The main theorem stating the maximum number of successful teams --/
theorem max_successful_teams (t : Tournament) 
  (h1 : t.num_teams = 16)
  (h2 : t.points_for_win = 3)
  (h3 : t.points_for_draw = 1)
  (h4 : t.points_for_loss = 0) :
  ∃ (n : ℕ), n = 15 ∧ 
  (∀ (m : ℕ), m > n → 
    ¬ ∃ (points : List ℕ), 
      points.length = t.num_teams ∧
      points.sum = (t.num_teams * (t.num_teams - 1) / 2) * t.points_for_win ∧
      (points.filter (λ x => x ≥ min_success_points t)).length = m) :=
sorry

end max_successful_teams_l3200_320032


namespace parabola_through_AC_not_ABC_l3200_320016

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a parabola of the form y = ax^2 + bx + 1 -/
def Parabola (a b : ℝ) (p : Point) : Prop :=
  p.y = a * p.x^2 + b * p.x + 1

/-- The given points A, B, and C -/
def A : Point := ⟨1, 2⟩
def B : Point := ⟨2, 3⟩
def C : Point := ⟨2, 1⟩

theorem parabola_through_AC_not_ABC :
  (∃ a b : ℝ, Parabola a b A ∧ Parabola a b C) ∧
  (¬ ∃ a b : ℝ, Parabola a b A ∧ Parabola a b B ∧ Parabola a b C) := by
  sorry

end parabola_through_AC_not_ABC_l3200_320016


namespace sum_of_digits_of_power_l3200_320025

theorem sum_of_digits_of_power : ∃ (tens ones : ℕ),
  (tens * 10 + ones = (3 + 4)^15 % 100) ∧ 
  (tens + ones = 7) := by sorry

end sum_of_digits_of_power_l3200_320025


namespace max_y_coordinate_l3200_320043

theorem max_y_coordinate (x y : ℝ) : 
  x^2 / 49 + (y - 3)^2 / 25 = 0 → y ≤ 3 :=
by sorry

end max_y_coordinate_l3200_320043


namespace pig_to_cow_ratio_l3200_320092

/-- Represents the farm scenario with cows and pigs -/
structure Farm where
  numCows : ℕ
  revenueCow : ℕ
  revenuePig : ℕ
  totalRevenue : ℕ

/-- Calculates the number of pigs based on the farm data -/
def calculatePigs (farm : Farm) : ℕ :=
  (farm.totalRevenue - farm.numCows * farm.revenueCow) / farm.revenuePig

/-- Theorem stating that the ratio of pigs to cows is 4:1 -/
theorem pig_to_cow_ratio (farm : Farm)
  (h1 : farm.numCows = 20)
  (h2 : farm.revenueCow = 800)
  (h3 : farm.revenuePig = 400)
  (h4 : farm.totalRevenue = 48000) :
  (calculatePigs farm) / farm.numCows = 4 := by
  sorry

#check pig_to_cow_ratio

end pig_to_cow_ratio_l3200_320092


namespace dividend_calculation_l3200_320058

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 14)
  (h2 : quotient = 12)
  (h3 : remainder = 8) :
  divisor * quotient + remainder = 176 := by
  sorry

end dividend_calculation_l3200_320058


namespace certain_number_value_l3200_320001

theorem certain_number_value (n : ℝ) : 22 + Real.sqrt (-4 + 6 * 4 * n) = 24 → n = 1 / 3 := by
  sorry

end certain_number_value_l3200_320001


namespace largest_divisor_of_expression_l3200_320094

theorem largest_divisor_of_expression (p q : ℤ) 
  (hp : Odd p) (hq : Odd q) (hlt : q < p) : 
  ∃ k : ℤ, p^2 - q^2 + 2*p - 2*q = 8 * k := by
  sorry

end largest_divisor_of_expression_l3200_320094


namespace policeman_speed_l3200_320044

/-- Calculates the speed of a policeman chasing a criminal given the initial conditions --/
theorem policeman_speed
  (initial_distance : ℝ)
  (criminal_speed : ℝ)
  (time_elapsed : ℝ)
  (final_distance : ℝ)
  (h1 : initial_distance = 180)
  (h2 : criminal_speed = 8)
  (h3 : time_elapsed = 5 / 60)
  (h4 : final_distance = 96.66666666666667)
  : ∃ (policeman_speed : ℝ), policeman_speed = 1000 := by
  sorry

#check policeman_speed

end policeman_speed_l3200_320044


namespace not_always_possible_to_empty_bags_l3200_320081

/-- Represents the state of the two bags --/
structure BagState where
  m : ℕ
  n : ℕ

/-- Allowed operations on the bags --/
inductive Operation
  | remove : ℕ → Operation
  | tripleFirst : Operation
  | tripleSecond : Operation

/-- Applies an operation to a bag state --/
def applyOperation (state : BagState) (op : Operation) : BagState :=
  match op with
  | Operation.remove k => ⟨state.m - k, state.n - k⟩
  | Operation.tripleFirst => ⟨3 * state.m, state.n⟩
  | Operation.tripleSecond => ⟨state.m, 3 * state.n⟩

/-- A sequence of operations --/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a bag state --/
def applySequence (state : BagState) (seq : OperationSequence) : BagState :=
  seq.foldl applyOperation state

/-- Theorem: There exist initial values of m and n for which it's impossible to empty both bags --/
theorem not_always_possible_to_empty_bags : 
  ∃ (m n : ℕ), m ≥ 1 ∧ n ≥ 1 ∧ 
  ∀ (seq : OperationSequence), 
  let final_state := applySequence ⟨m, n⟩ seq
  (final_state.m ≠ 0 ∨ final_state.n ≠ 0) :=
by sorry

end not_always_possible_to_empty_bags_l3200_320081


namespace divisibility_by_16_l3200_320035

theorem divisibility_by_16 (x : ℤ) : 
  16 ∣ (9*x^2 + 29*x + 62) ↔ ∃ t : ℤ, (x = 16*t + 6 ∨ x = 16*t + 5) :=
sorry

end divisibility_by_16_l3200_320035


namespace gcd_problem_l3200_320091

/-- The operation * represents the greatest common divisor -/
def gcd_op (a b : ℕ) : ℕ := Nat.gcd a b

/-- Theorem stating that ((16 * 20) * (18 * 24)) = 2 using the gcd operation -/
theorem gcd_problem : gcd_op (gcd_op 16 20) (gcd_op 18 24) = 2 := by
  sorry

end gcd_problem_l3200_320091


namespace fractional_equation_range_l3200_320050

theorem fractional_equation_range (a x : ℝ) : 
  ((a + 2) / (x + 1) = 1 ∧ x ≤ 0 ∧ x + 1 ≠ 0) → (a ≤ -1 ∧ a ≠ -2) :=
by sorry

end fractional_equation_range_l3200_320050


namespace smallest_c_value_l3200_320067

theorem smallest_c_value (c : ℝ) (h1 : c > 0) : 
  (∀ x : ℝ, 3 * Real.cos (6 * x + c) ≤ 3 * Real.cos (6 * (-π/6) + c)) → 
  c = π :=
sorry

end smallest_c_value_l3200_320067


namespace five_classrooms_formed_l3200_320003

/-- Represents the problem of forming classrooms with equal numbers of boys and girls -/
def ClassroomFormation (total_boys total_girls students_per_class : ℕ) : Prop :=
  ∃ (num_classrooms : ℕ),
    -- Each classroom has an equal number of boys and girls
    ∃ (boys_per_class : ℕ),
      2 * boys_per_class = students_per_class ∧
      -- The total number of students in all classrooms doesn't exceed the available students
      num_classrooms * students_per_class ≤ total_boys + total_girls ∧
      -- All boys and girls are assigned to classrooms
      num_classrooms * boys_per_class ≤ total_boys ∧
      num_classrooms * boys_per_class ≤ total_girls ∧
      -- This is the maximum number of classrooms possible
      ∀ (larger_num_classrooms : ℕ), larger_num_classrooms > num_classrooms →
        larger_num_classrooms * boys_per_class > total_boys ∨
        larger_num_classrooms * boys_per_class > total_girls

/-- The main theorem stating that 5 classrooms can be formed under the given conditions -/
theorem five_classrooms_formed :
  ClassroomFormation 56 44 25 ∧
  ∀ (n : ℕ), ClassroomFormation 56 44 25 → n ≤ 5 :=
by sorry

end five_classrooms_formed_l3200_320003


namespace count_three_digit_even_numbers_l3200_320012

/-- The set of available digits -/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A function that checks if a number is even -/
def isEven (n : Nat) : Bool := n % 2 = 0

/-- A function that checks if a number has three distinct digits -/
def hasThreeDistinctDigits (n : Nat) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

/-- The main theorem -/
theorem count_three_digit_even_numbers : 
  (Finset.filter (fun n => n ≥ 100 ∧ n < 1000 ∧ isEven n ∧ hasThreeDistinctDigits n ∧ 
    (∀ d, d ∈ digits → (n / 100 = d ∨ (n / 10) % 10 = d ∨ n % 10 = d)))
    (Finset.range 1000)).card = 52 := by
  sorry

end count_three_digit_even_numbers_l3200_320012


namespace arithmetic_sequence_properties_l3200_320029

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- General term of the sequence
  S : ℕ → ℝ  -- Sum of the first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a4_eq_1 : a 4 = 1
  S15_eq_75 : S 15 = 75

/-- The theorem statement -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = n - 3) ∧
  (∃ c : ℝ, c ≠ 0 ∧
    (∀ n m, (seq.S (n + 1) / ((n + 1) + c) - seq.S n / (n + c)) =
            (seq.S (m + 1) / ((m + 1) + c) - seq.S m / (m + c))) →
    c = -5) :=
  sorry

end arithmetic_sequence_properties_l3200_320029


namespace map_distance_calculation_l3200_320065

/-- Represents the scale of a map in feet per inch -/
def map_scale : ℝ := 500

/-- Represents the length of a line segment on the map in inches -/
def map_length : ℝ := 7.2

/-- Calculates the actual distance represented by a map length -/
def actual_distance (scale : ℝ) (map_length : ℝ) : ℝ := scale * map_length

/-- Theorem: The actual distance represented by a 7.2-inch line segment on a map with a scale of 1 inch = 500 feet is 3600 feet -/
theorem map_distance_calculation :
  actual_distance map_scale map_length = 3600 := by
  sorry

end map_distance_calculation_l3200_320065


namespace system_of_equations_solution_l3200_320048

theorem system_of_equations_solution (a b x y : ℝ) : 
  (2 * a - 3 * b = 13 ∧ 3 * a + 5 * b = 30.9) →
  (a = 8.3 ∧ b = 1.2) →
  (2 * (x + 2) - 3 * (y - 1) = 13 ∧ 3 * (x + 2) + 5 * (y - 1) = 30.9) →
  (x = 6.3 ∧ y = 2.2) :=
by sorry

end system_of_equations_solution_l3200_320048


namespace magician_earned_four_dollars_l3200_320086

/-- The amount earned by a magician selling magic card decks -/
def magician_earnings (initial_decks : ℕ) (remaining_decks : ℕ) (price_per_deck : ℕ) : ℕ :=
  (initial_decks - remaining_decks) * price_per_deck

/-- Theorem: The magician earned 4 dollars -/
theorem magician_earned_four_dollars :
  magician_earnings 5 3 2 = 4 := by
  sorry

end magician_earned_four_dollars_l3200_320086


namespace unique_solution_l3200_320088

def system_equations (n : ℕ) (x : ℕ → ℝ) : Prop :=
  n ≥ 2 ∧
  (∀ i : ℕ, i ∈ Finset.range n → 
    max (i + 1 : ℝ) (x i) = if i + 1 = n then n * x 0 else x (i + 1))

theorem unique_solution (n : ℕ) (x : ℕ → ℝ) :
  system_equations n x → (∀ i : ℕ, i ∈ Finset.range n → x i = 1) :=
sorry

end unique_solution_l3200_320088


namespace min_balls_drawn_l3200_320000

/-- Given a bag with 10 white balls, 5 black balls, and 4 blue balls,
    the minimum number of balls to be drawn to ensure at least 2 balls of each color is 17. -/
theorem min_balls_drawn (white : Nat) (black : Nat) (blue : Nat)
    (h_white : white = 10)
    (h_black : black = 5)
    (h_blue : blue = 4) :
    (∃ n : Nat, n = 17 ∧
      ∀ m : Nat, m < n →
        ¬(∃ w b l : Nat, w ≥ 2 ∧ b ≥ 2 ∧ l ≥ 2 ∧
          w + b + l = m ∧ w ≤ white ∧ b ≤ black ∧ l ≤ blue)) :=
by sorry

end min_balls_drawn_l3200_320000


namespace div_sqrt_three_equals_sqrt_three_l3200_320064

theorem div_sqrt_three_equals_sqrt_three : 3 / Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end div_sqrt_three_equals_sqrt_three_l3200_320064


namespace line_arrangement_with_restriction_l3200_320057

def number_of_students : ℕ := 5

def total_permutations (n : ℕ) : ℕ := Nat.factorial n

def restricted_permutations (n : ℕ) : ℕ := 
  Nat.factorial (n - 1) * 2

theorem line_arrangement_with_restriction : 
  total_permutations number_of_students - restricted_permutations number_of_students = 72 := by
  sorry

end line_arrangement_with_restriction_l3200_320057


namespace unique_solution_when_a_is_seven_l3200_320037

/-- The equation has exactly one solution when a = 7 -/
theorem unique_solution_when_a_is_seven (x : ℝ) (a : ℝ) : 
  (a ≠ 1 ∧ x ≠ -3) →
  (∃! x, ((|((a*x^2 - a*x - 12*a + x^2 + x + 12) / (a*x + 3*a - x - 3))| - a) * |4*a - 3*x - 19| = 0)) ↔
  a = 7 :=
sorry

end unique_solution_when_a_is_seven_l3200_320037


namespace only_sqrt_6_is_quadratic_radical_l3200_320076

-- Define what it means for an expression to be a quadratic radical
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ y : ℝ, y ≥ 0 ∧ x = Real.sqrt y

-- Theorem statement
theorem only_sqrt_6_is_quadratic_radical :
  is_quadratic_radical (Real.sqrt 6) ∧
  ¬is_quadratic_radical (Real.sqrt (-5)) ∧
  ¬is_quadratic_radical (8 ^ (1/3 : ℝ)) ∧
  ¬∀ a : ℝ, is_quadratic_radical (Real.sqrt a) :=
by sorry

end only_sqrt_6_is_quadratic_radical_l3200_320076


namespace tangent_curve_b_value_l3200_320006

/-- The curve equation -/
def curve (x a b : ℝ) : ℝ := x^3 + a*x + b

/-- The tangent line equation -/
def tangent_line (x : ℝ) : ℝ := 2*x + 1

/-- The derivative of the curve -/
def curve_derivative (x a : ℝ) : ℝ := 3*x^2 + a

theorem tangent_curve_b_value (a b : ℝ) : 
  (curve 1 a b = tangent_line 1) ∧ 
  (curve_derivative 1 a = 2) → 
  b = 3 := by
  sorry

end tangent_curve_b_value_l3200_320006


namespace house_transaction_result_l3200_320049

def house_transaction (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : ℝ :=
  let first_sale := initial_value * (1 - loss_percent)
  let second_sale := first_sale * (1 + gain_percent)
  second_sale - initial_value

theorem house_transaction_result :
  house_transaction 12000 0.15 0.20 = -240 := by
  sorry

end house_transaction_result_l3200_320049


namespace arithmetic_sequence_sum_l3200_320007

/-- An arithmetic sequence with a_4 = 5 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d ∧ a 4 = 5

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  2 * (a 1) - (a 5) + (a 11) = 10 := by
  sorry

end arithmetic_sequence_sum_l3200_320007


namespace current_speed_calculation_l3200_320080

/-- The speed of the current in a river -/
def current_speed : ℝ := 3

/-- The speed at which a man can row in still water (in km/hr) -/
def still_water_speed : ℝ := 15

/-- The time taken to cover 100 meters downstream (in seconds) -/
def downstream_time : ℝ := 20

/-- The distance covered downstream (in meters) -/
def downstream_distance : ℝ := 100

/-- Conversion factor from m/s to km/hr -/
def ms_to_kmhr : ℝ := 3.6

theorem current_speed_calculation :
  current_speed = 
    (downstream_distance / downstream_time * ms_to_kmhr) - still_water_speed :=
by sorry

end current_speed_calculation_l3200_320080


namespace point_in_second_quadrant_implies_m_greater_than_three_l3200_320089

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate. -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The point P with coordinates (3-m, m-1) -/
def P (m : ℝ) : ℝ × ℝ := (3 - m, m - 1)

/-- If point P(3-m, m-1) is in the second quadrant, then m > 3 -/
theorem point_in_second_quadrant_implies_m_greater_than_three (m : ℝ) :
  second_quadrant (P m).1 (P m).2 → m > 3 := by
  sorry

end point_in_second_quadrant_implies_m_greater_than_three_l3200_320089


namespace product_of_polynomials_l3200_320002

theorem product_of_polynomials (p q : ℝ) : 
  (∀ m : ℝ, (9 * m^2 - 2 * m + p) * (4 * m^2 + q * m - 5) = 
             36 * m^4 - 23 * m^3 - 31 * m^2 + 6 * m - 10) →
  p + q = 0 := by
sorry

end product_of_polynomials_l3200_320002
