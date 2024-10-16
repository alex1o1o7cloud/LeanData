import Mathlib

namespace NUMINAMATH_CALUDE_equal_positive_reals_from_inequalities_l26_2646

theorem equal_positive_reals_from_inequalities 
  (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (pos₁ : x₁ > 0) (pos₂ : x₂ > 0) (pos₃ : x₃ > 0) (pos₄ : x₄ > 0) (pos₅ : x₅ > 0)
  (ineq₁ : (x₁^2 - x₃*x₃)*(x₂^2 - x₃*x₃) ≤ 0)
  (ineq₂ : (x₃^2 - x₁*x₁)*(x₃^2 - x₁*x₁) ≤ 0)
  (ineq₃ : (x₃^2 - x₃*x₂)*(x₁^2 - x₃*x₂) ≤ 0)
  (ineq₄ : (x₁^2 - x₁*x₃)*(x₃^2 - x₁*x₃) ≤ 0)
  (ineq₅ : (x₃^2 - x₂*x₁)*(x₁^2 - x₂*x₁) ≤ 0) :
  x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅ := by
  sorry


end NUMINAMATH_CALUDE_equal_positive_reals_from_inequalities_l26_2646


namespace NUMINAMATH_CALUDE_oranges_thrown_away_l26_2658

theorem oranges_thrown_away (initial_oranges : ℕ) (new_oranges : ℕ) (final_oranges : ℕ)
  (h1 : initial_oranges = 50)
  (h2 : new_oranges = 24)
  (h3 : final_oranges = 34)
  : initial_oranges - (initial_oranges - new_oranges + final_oranges) = 40 := by
  sorry

end NUMINAMATH_CALUDE_oranges_thrown_away_l26_2658


namespace NUMINAMATH_CALUDE_set_equivalence_l26_2670

theorem set_equivalence : 
  {x : ℕ+ | x - 3 < 2} = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_set_equivalence_l26_2670


namespace NUMINAMATH_CALUDE_group_size_l26_2682

theorem group_size (N : ℝ) 
  (h1 : N / 5 = N * (1 / 5))  -- 1/5 of the group plays at least one instrument
  (h2 : N * (1 / 5) - 128 = N * 0.04)  -- Probability of playing exactly one instrument is 0.04
  : N = 800 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l26_2682


namespace NUMINAMATH_CALUDE_minimum_at_two_l26_2631

/-- The function f(x) with parameter t -/
def f (t : ℝ) (x : ℝ) : ℝ := x^3 - 2*t*x^2 + t^2*x

/-- The derivative of f(x) with respect to x -/
def f' (t : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*t*x + t^2

theorem minimum_at_two (t : ℝ) : 
  (∀ x : ℝ, f t x ≥ f t 2) ↔ t = 2 := by sorry

end NUMINAMATH_CALUDE_minimum_at_two_l26_2631


namespace NUMINAMATH_CALUDE_hexagon_circle_visibility_l26_2618

theorem hexagon_circle_visibility (s : ℝ) (r : ℝ) (h1 : s = 3) (h2 : r > 0) : 
  let a := s * Real.sqrt 3 / 2
  (2 * Real.pi * r / 3) / (2 * Real.pi * r) = 1 / 3 → r = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_circle_visibility_l26_2618


namespace NUMINAMATH_CALUDE_algebraic_simplification_l26_2655

theorem algebraic_simplification (a : ℝ) : (-2*a)^3 * a^3 + (-3*a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l26_2655


namespace NUMINAMATH_CALUDE_max_true_statements_l26_2625

theorem max_true_statements (y : ℝ) : 
  let statements := [
    (1 < y^2 ∧ y^2 < 4),
    (y^2 > 4),
    (-2 < y ∧ y < 0),
    (0 < y ∧ y < 2),
    (0 < y^3 - y^2 ∧ y^3 - y^2 < 4)
  ]
  ∃ (true_statements : List Bool), 
    (∀ i, true_statements.get! i = true → statements.get! i) ∧
    true_statements.count true ≤ 3 ∧
    ∀ (other_true_statements : List Bool),
      (∀ i, other_true_statements.get! i = true → statements.get! i) →
      other_true_statements.count true ≤ true_statements.count true :=
by sorry


end NUMINAMATH_CALUDE_max_true_statements_l26_2625


namespace NUMINAMATH_CALUDE_regular_pentagon_perimeter_l26_2615

/-- The sum of sides of a regular pentagon with side length 15 cm is 75 cm. -/
theorem regular_pentagon_perimeter (side_length : ℝ) (n_sides : ℕ) : 
  side_length = 15 → n_sides = 5 → side_length * n_sides = 75 := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_perimeter_l26_2615


namespace NUMINAMATH_CALUDE_runner_stops_in_third_quarter_l26_2664

theorem runner_stops_in_third_quarter 
  (track_circumference : ℝ) 
  (total_distance : ℝ) 
  (quarter_length : ℝ) :
  track_circumference = 50 →
  total_distance = 5280 →
  quarter_length = track_circumference / 4 →
  ∃ (n : ℕ) (remaining_distance : ℝ),
    total_distance = n * track_circumference + remaining_distance ∧
    remaining_distance > 2 * quarter_length ∧
    remaining_distance ≤ 3 * quarter_length :=
by sorry

end NUMINAMATH_CALUDE_runner_stops_in_third_quarter_l26_2664


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l26_2626

/-- The lateral surface area of a cone with base radius 3 cm and lateral surface forming a semicircle when unfolded -/
theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), 
    r = 3 → -- base radius is 3 cm
    l = 6 → -- slant height is 6 cm (derived from the semicircle condition)
    (1/2 : ℝ) * Real.pi * l^2 = 18 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l26_2626


namespace NUMINAMATH_CALUDE_lychee_theorem_l26_2688

def lychee_yield (n : ℕ) : ℕ → ℕ
  | 0 => 1
  | i + 1 =>
    if i < 9 then 2 * lychee_yield n i + 1
    else if i < 15 then lychee_yield n 9
    else (lychee_yield n i) / 2

def total_yield (n : ℕ) : ℕ :=
  (List.range n).map (lychee_yield n) |>.sum

theorem lychee_theorem : total_yield 25 = 8173 := by
  sorry

end NUMINAMATH_CALUDE_lychee_theorem_l26_2688


namespace NUMINAMATH_CALUDE_other_communities_count_l26_2683

/-- Given a school with 300 boys, calculate the number of boys belonging to other communities -/
theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total = 300 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  ↑total * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 54 :=
by sorry

end NUMINAMATH_CALUDE_other_communities_count_l26_2683


namespace NUMINAMATH_CALUDE_flu_transmission_rate_l26_2699

theorem flu_transmission_rate : ∃ x : ℝ, 
  (x > 0) ∧ ((1 + x)^2 = 100) ∧ (x = 9) := by
  sorry

end NUMINAMATH_CALUDE_flu_transmission_rate_l26_2699


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l26_2656

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l26_2656


namespace NUMINAMATH_CALUDE_max_zero_point_quadratic_l26_2643

theorem max_zero_point_quadratic (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  let f := fun x : ℝ => a * x^2 + (3 + 1/b) * x - a
  let zero_points := {x : ℝ | f x = 0}
  ∃ (x : ℝ), x ∈ zero_points ∧ ∀ (y : ℝ), y ∈ zero_points → y ≤ x ∧ x = (-9 + Real.sqrt 85) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_zero_point_quadratic_l26_2643


namespace NUMINAMATH_CALUDE_cost_calculation_theorem_l26_2604

/-- Represents the cost calculation for purchasing table tennis equipment --/
def cost_calculation (x : ℕ) : Prop :=
  let racket_price : ℕ := 80
  let ball_price : ℕ := 20
  let racket_quantity : ℕ := 20
  let option1_cost : ℕ := racket_price * racket_quantity
  let option2_cost : ℕ := (racket_price * racket_quantity + ball_price * x) * 9 / 10
  x > 20 → option1_cost = 1600 ∧ option2_cost = 1440 + 18 * x

/-- Theorem stating the cost calculation for purchasing table tennis equipment --/
theorem cost_calculation_theorem (x : ℕ) : cost_calculation x := by
  sorry

#check cost_calculation_theorem

end NUMINAMATH_CALUDE_cost_calculation_theorem_l26_2604


namespace NUMINAMATH_CALUDE_initial_books_in_bin_l26_2630

theorem initial_books_in_bin (initial_books sold_books added_books final_books : ℕ) :
  sold_books = 3 →
  added_books = 10 →
  final_books = 11 →
  initial_books - sold_books + added_books = final_books →
  initial_books = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_in_bin_l26_2630


namespace NUMINAMATH_CALUDE_positive_A_value_l26_2671

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value :
  ∃ A : ℝ, A > 0 ∧ hash A 3 = 145 ∧ A = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l26_2671


namespace NUMINAMATH_CALUDE_yeast_growth_20_minutes_l26_2657

/-- Represents the population growth of yeast cells over time -/
def yeast_population (initial_population : ℕ) (growth_factor : ℕ) (intervals : ℕ) : ℕ :=
  initial_population * growth_factor ^ intervals

theorem yeast_growth_20_minutes :
  let initial_population := 30
  let growth_factor := 3
  let intervals := 5
  yeast_population initial_population growth_factor intervals = 7290 := by sorry

end NUMINAMATH_CALUDE_yeast_growth_20_minutes_l26_2657


namespace NUMINAMATH_CALUDE_window_area_calculation_l26_2617

/-- Calculates the area of a window given its length in meters and width in feet -/
def windowArea (lengthMeters : ℝ) (widthFeet : ℝ) : ℝ :=
  let meterToFeet : ℝ := 3.28084
  let lengthFeet : ℝ := lengthMeters * meterToFeet
  lengthFeet * widthFeet

theorem window_area_calculation :
  windowArea 2 15 = 98.4252 := by
  sorry

end NUMINAMATH_CALUDE_window_area_calculation_l26_2617


namespace NUMINAMATH_CALUDE_area_of_intersection_l26_2654

-- Define the square ABCD
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_unit : A = (0, 1) ∧ B = (1, 1) ∧ C = (1, 0) ∧ D = (0, 0))

-- Define the rotation
def rotate (p : ℝ × ℝ) (center : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ := sorry

-- Define the rotated square A'B'C'D'
def rotated_square (S : Square) (angle : ℝ) : Square := sorry

-- Define the intersection quadrilateral DALC'
structure Quadrilateral :=
  (D A L C' : ℝ × ℝ)

-- Define the area function for a quadrilateral
def area (Q : Quadrilateral) : ℝ := sorry

-- Main theorem
theorem area_of_intersection (S : Square) (α : ℝ) :
  let S' := rotated_square S α
  let Q := Quadrilateral.mk S.D S.A (Real.cos α, 1) (Real.cos α, Real.sin α)
  area Q = 1/2 * (1 - Real.sin α * Real.cos α) :=
sorry

end NUMINAMATH_CALUDE_area_of_intersection_l26_2654


namespace NUMINAMATH_CALUDE_parallelogram_smaller_angle_measure_l26_2603

/-- 
Given a parallelogram where one angle exceeds the other by 50 degrees,
prove that the smaller angle measures 65 degrees.
-/
theorem parallelogram_smaller_angle_measure : 
  ∀ (small_angle large_angle : ℝ),
  small_angle > 0 →
  large_angle > 0 →
  large_angle = small_angle + 50 →
  small_angle + large_angle = 180 →
  small_angle = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_smaller_angle_measure_l26_2603


namespace NUMINAMATH_CALUDE_student_calculation_l26_2627

theorem student_calculation (x : ℕ) (h : x = 120) : 2 * x - 138 = 102 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_l26_2627


namespace NUMINAMATH_CALUDE_binary_101101_to_octal_55_l26_2622

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldr (λ x acc => 2 * acc + if x then 1 else 0) 0

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_101101_to_octal_55 :
  decimal_to_octal (binary_to_decimal binary_101101) = [5, 5] := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_to_octal_55_l26_2622


namespace NUMINAMATH_CALUDE_gcd_of_mersenne_numbers_l26_2696

theorem gcd_of_mersenne_numbers : Nat.gcd (2^2048 - 1) (2^2035 - 1) = 2^13 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_mersenne_numbers_l26_2696


namespace NUMINAMATH_CALUDE_min_b_minus_a_l26_2689

noncomputable def f (a b x : ℝ) : ℝ := (2 * x^2 + x) * Real.log x - (2 * a + 1) * x^2 - (a + 1) * x + b

theorem min_b_minus_a (a b : ℝ) :
  (∀ x > 0, f a b x ≥ 0) → 
  ∃ m, m = 3/4 + Real.log 2 ∧ b - a ≥ m ∧ ∀ ε > 0, ∃ a' b', b' - a' < m + ε :=
sorry

end NUMINAMATH_CALUDE_min_b_minus_a_l26_2689


namespace NUMINAMATH_CALUDE_path_count_l26_2624

/-- A simple directed graph with vertices A, B, C, and D -/
structure Graph :=
  (paths_AB : ℕ)
  (paths_BC : ℕ)
  (paths_CD : ℕ)
  (direct_AC : ℕ)

/-- The total number of paths from A to D in the graph -/
def total_paths (g : Graph) : ℕ :=
  g.paths_AB * g.paths_BC * g.paths_CD + g.direct_AC * g.paths_CD

/-- Theorem stating that the total number of paths from A to D is 15 -/
theorem path_count (g : Graph) 
  (h1 : g.paths_AB = 2)
  (h2 : g.paths_BC = 2)
  (h3 : g.paths_CD = 3)
  (h4 : g.direct_AC = 1) : 
  total_paths g = 15 := by
  sorry

end NUMINAMATH_CALUDE_path_count_l26_2624


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_with_ratio_l26_2633

theorem smallest_angle_in_triangle_with_ratio (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- angles are positive
  a + b + c = 180 →  -- sum of angles is 180°
  b = 2 * a →  -- ratio condition
  c = 3 * a →  -- ratio condition
  a = 30 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_with_ratio_l26_2633


namespace NUMINAMATH_CALUDE_factorial_inequality_l26_2609

theorem factorial_inequality (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : n ≤ m) :
  (2^n : ℝ) * (n.factorial : ℝ) ≤ ((m+n).factorial : ℝ) / ((m-n).factorial : ℝ) ∧
  ((m+n).factorial : ℝ) / ((m-n).factorial : ℝ) ≤ ((m^2 + m : ℝ)^n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_factorial_inequality_l26_2609


namespace NUMINAMATH_CALUDE_cannot_return_to_start_l26_2691

-- Define the type for points on the plane
def Point := ℝ × ℝ

-- Define the allowed moves
def move_up (p : Point) : Point := (p.1, p.2 + 2*p.1)
def move_down (p : Point) : Point := (p.1, p.2 - 2*p.1)
def move_right (p : Point) : Point := (p.1 + 2*p.2, p.2)
def move_left (p : Point) : Point := (p.1 - 2*p.2, p.2)

-- Define a sequence of moves
inductive Move
| up : Move
| down : Move
| right : Move
| left : Move

def apply_move (p : Point) (m : Move) : Point :=
  match m with
  | Move.up => move_up p
  | Move.down => move_down p
  | Move.right => move_right p
  | Move.left => move_left p

def apply_moves (p : Point) (ms : List Move) : Point :=
  ms.foldl apply_move p

-- The main theorem
theorem cannot_return_to_start : 
  ∀ (ms : List Move), apply_moves (1, Real.sqrt 2) ms ≠ (1, Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_cannot_return_to_start_l26_2691


namespace NUMINAMATH_CALUDE_engineer_number_theorem_l26_2676

def proper_divisors (n : ℕ) : Set ℕ :=
  {d | d ∣ n ∧ d ≠ 1 ∧ d ≠ n}

def increased_divisors (n : ℕ) : Set ℕ :=
  {d + 1 | d ∈ proper_divisors n}

theorem engineer_number_theorem :
  {n : ℕ | ∃ m : ℕ, increased_divisors n = proper_divisors m} = {4, 8} := by
sorry

end NUMINAMATH_CALUDE_engineer_number_theorem_l26_2676


namespace NUMINAMATH_CALUDE_perfect_square_increased_by_prime_l26_2687

theorem perfect_square_increased_by_prime (n : ℕ) : ∃ n : ℕ, 
  (∃ a : ℕ, n^2 = a^2) ∧ 
  (∃ b : ℕ, n^2 + 461 = b^2) ∧ 
  (∃ c : ℕ, n^2 = 5 * c) ∧ 
  (∃ d : ℕ, n^2 + 461 = 5 * d) ∧ 
  n^2 = 52900 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_increased_by_prime_l26_2687


namespace NUMINAMATH_CALUDE_sequence_sum_l26_2690

/-- Given a sequence {a_n} with sum of first n terms S_n, prove S_n = -3^(n-1) -/
theorem sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (a 1 = -1) → 
  (∀ n : ℕ, a (n + 1) = 2 * S n) → 
  (∀ n : ℕ, S n = -(3^(n - 1))) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l26_2690


namespace NUMINAMATH_CALUDE_village_population_is_72_l26_2637

/-- The number of people a vampire drains per week -/
def vampire_drain_rate : ℕ := 3

/-- The number of people a werewolf eats per week -/
def werewolf_eat_rate : ℕ := 5

/-- The number of weeks the village lasts -/
def weeks_lasted : ℕ := 9

/-- The total number of people in the village -/
def village_population : ℕ := vampire_drain_rate * weeks_lasted + werewolf_eat_rate * weeks_lasted

theorem village_population_is_72 : village_population = 72 := by
  sorry

end NUMINAMATH_CALUDE_village_population_is_72_l26_2637


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l26_2667

theorem geometric_arithmetic_progression_sum (b q : ℝ) (h1 : b > 0) (h2 : q > 0) :
  let a := b
  let d := (b * q^3 - b) / 3
  (∃ (n : ℕ), q^n = 2) →
  (3 * a + 10 * d = 148 / 9) →
  (b * (q^4 - 1) / (q - 1) = 700 / 27) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l26_2667


namespace NUMINAMATH_CALUDE_shopping_money_calculation_l26_2645

theorem shopping_money_calculation (initial_amount : ℝ) : 
  (0.7 * initial_amount = 350) → initial_amount = 500 := by
  sorry

end NUMINAMATH_CALUDE_shopping_money_calculation_l26_2645


namespace NUMINAMATH_CALUDE_division_equality_not_always_true_l26_2619

theorem division_equality_not_always_true (x y m : ℝ) :
  ¬(∀ x y m : ℝ, x = y → x / m = y / m) :=
sorry

end NUMINAMATH_CALUDE_division_equality_not_always_true_l26_2619


namespace NUMINAMATH_CALUDE_june_score_l26_2653

theorem june_score (april_may_avg : ℝ) (april_may_june_avg : ℝ) (june_score : ℝ) :
  april_may_avg = 89 →
  april_may_june_avg = 88 →
  june_score = 3 * april_may_june_avg - 2 * april_may_avg →
  june_score = 86 := by
sorry

end NUMINAMATH_CALUDE_june_score_l26_2653


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l26_2694

theorem modulus_of_complex_number :
  let z : ℂ := (1 + 2*Complex.I) / Complex.I^2
  Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l26_2694


namespace NUMINAMATH_CALUDE_subset_of_A_l26_2648

def A : Set ℝ := {x | x > -1}

theorem subset_of_A : {0} ⊆ A := by sorry

end NUMINAMATH_CALUDE_subset_of_A_l26_2648


namespace NUMINAMATH_CALUDE_millionaire_allocation_l26_2602

/-- The number of ways to allocate millionaires to hotel rooms -/
def allocate_millionaires (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 36 ways to allocate 13 millionaires to 3 types of rooms -/
theorem millionaire_allocation :
  allocate_millionaires 10 3 = 36 :=
sorry

#eval allocate_millionaires 10 3

end NUMINAMATH_CALUDE_millionaire_allocation_l26_2602


namespace NUMINAMATH_CALUDE_line_equation_l26_2695

-- Define the points A and B
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (1, 4)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the property of having equal intercepts on both axes
def equal_intercepts (a b c : ℝ) : Prop := ∃ k : ℝ, a * k = c ∧ b * k = c ∧ k ≠ 0

-- Define the main theorem
theorem line_equation :
  ∃ (a b c : ℝ),
    -- The line passes through point A
    (a * A.1 + b * A.2 + c = 0) ∧
    -- The line is perpendicular to 2x + y - 5 = 0
    (a * 2 + b * 1 = 0) ∧
    -- The line passes through point B
    (a * B.1 + b * B.2 + c = 0) ∧
    -- The line has equal intercepts on both axes
    (equal_intercepts a b c) ∧
    -- The equation of the line is either x + y - 5 = 0 or 4x - y = 0
    ((a = 1 ∧ b = 1 ∧ c = -5) ∨ (a = 4 ∧ b = -1 ∧ c = 0)) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l26_2695


namespace NUMINAMATH_CALUDE_part1_part2_l26_2639

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x + 3| - |2*x - a|

-- Part 1
theorem part1 (a : ℝ) : 
  (∃ x, f a x ≤ -5) → (a ≤ -8 ∨ a ≥ 2) := by sorry

-- Part 2
theorem part2 (a : ℝ) : 
  (∀ x, f a (x - 1/2) + f a (-x - 1/2) = 0) → a = 1 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l26_2639


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l26_2612

/-- The trajectory of the midpoint of a line segment connecting a point on a parabola and a fixed point -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ (P : ℝ × ℝ), 
    P.2 = 2 * P.1^2 + 1 ∧ 
    P.1 = 2 * x ∧ 
    P.2 = 2 * y + 1) →
  y = 4 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l26_2612


namespace NUMINAMATH_CALUDE_expression_equality_l26_2629

theorem expression_equality : 
  Real.sqrt 12 + 2 * Real.tan (π / 4) - Real.sin (π / 3) - (1 / 2)⁻¹ = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l26_2629


namespace NUMINAMATH_CALUDE_trigonometric_fraction_equals_one_l26_2621

theorem trigonometric_fraction_equals_one : 
  (Real.tan (30 * π / 180))^2 - (Real.sin (30 * π / 180))^2 = 
  (Real.tan (30 * π / 180))^2 * (Real.sin (30 * π / 180))^2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_fraction_equals_one_l26_2621


namespace NUMINAMATH_CALUDE_parallelogram_area_l26_2662

theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 18) (h2 : b = 10) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin (π - θ) = 90 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l26_2662


namespace NUMINAMATH_CALUDE_unique_positive_solution_l26_2669

theorem unique_positive_solution : 
  ∃! (x : ℝ), x > 0 ∧ x^101 + 100^99 = x^99 + 100^101 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l26_2669


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l26_2650

def f (x a : ℝ) := |2*x - 1| + |x - a|

theorem solution_part1 : 
  {x : ℝ | f x 3 ≤ 4} = Set.Icc 0 2 := by sorry

theorem solution_part2 (a : ℝ) :
  (∀ x, f x a = |x - 1 + a|) → 
  (a < 1/2 → {x : ℝ | f x a = |x - 1 + a|} = Set.Icc a (1/2)) ∧
  (a = 1/2 → {x : ℝ | f x a = |x - 1 + a|} = {1/2}) ∧
  (a > 1/2 → {x : ℝ | f x a = |x - 1 + a|} = Set.Icc (1/2) a) := by sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l26_2650


namespace NUMINAMATH_CALUDE_perpendicular_parallel_properties_l26_2638

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)

-- Given condition
variable (l : Line) (α : Plane)
variable (h : perp_plane l α)

-- Theorem to prove
theorem perpendicular_parallel_properties :
  (∀ m : Line, perp_plane m α → parallel m l) ∧
  (∀ m : Line, parallel_plane m α → perp m l) ∧
  (∀ m : Line, parallel m l → perp_plane m α) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_properties_l26_2638


namespace NUMINAMATH_CALUDE_cot_thirty_degrees_l26_2661

theorem cot_thirty_degrees : Real.cos (π / 6) / Real.sin (π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_thirty_degrees_l26_2661


namespace NUMINAMATH_CALUDE_committee_formation_count_l26_2673

def total_students : ℕ := 8
def committee_size : ℕ := 4
def required_students : ℕ := 2
def remaining_students : ℕ := total_students - required_students

theorem committee_formation_count : 
  Nat.choose remaining_students (committee_size - required_students) = 15 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l26_2673


namespace NUMINAMATH_CALUDE_family_ages_l26_2680

/-- Family ages problem -/
theorem family_ages :
  ∀ (son_age man_age daughter_age wife_age : ℝ),
  (man_age = son_age + 29) →
  (man_age + 2 = 2 * (son_age + 2)) →
  (daughter_age = son_age - 3.5) →
  (wife_age = 1.5 * daughter_age) →
  (son_age = 27 ∧ man_age = 56 ∧ daughter_age = 23.5 ∧ wife_age = 35.25) :=
by
  sorry

#check family_ages

end NUMINAMATH_CALUDE_family_ages_l26_2680


namespace NUMINAMATH_CALUDE_problem_solution_l26_2636

theorem problem_solution (x y : ℝ) 
  (h1 : |x| + x + y = 8) 
  (h2 : x + |y| - y = 10) : 
  x + y = 14/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l26_2636


namespace NUMINAMATH_CALUDE_exact_two_females_one_male_probability_l26_2605

def total_contestants : ℕ := 8
def female_contestants : ℕ := 5
def male_contestants : ℕ := 3
def selected_contestants : ℕ := 3

theorem exact_two_females_one_male_probability :
  (Nat.choose female_contestants 2 * Nat.choose male_contestants 1) / 
  Nat.choose total_contestants selected_contestants = 15 / 28 := by
  sorry

end NUMINAMATH_CALUDE_exact_two_females_one_male_probability_l26_2605


namespace NUMINAMATH_CALUDE_unique_perfect_square_p_l26_2675

/-- The polynomial p(x) = x^4 + 6x^3 + 11x^2 + 3x + 31 -/
def p (x : ℤ) : ℤ := x^4 + 6*x^3 + 11*x^2 + 3*x + 31

/-- A function that checks if a given integer is a perfect square -/
def is_perfect_square (n : ℤ) : Prop :=
  ∃ m : ℤ, n = m^2

/-- Theorem stating that there exists exactly one integer x for which p(x) is a perfect square -/
theorem unique_perfect_square_p :
  ∃! x : ℤ, is_perfect_square (p x) :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_p_l26_2675


namespace NUMINAMATH_CALUDE_proposition_implication_l26_2684

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 5) : 
  ¬ P 4 := by
  sorry

end NUMINAMATH_CALUDE_proposition_implication_l26_2684


namespace NUMINAMATH_CALUDE_min_box_height_is_19_l26_2685

/-- Represents the specifications for packaging a fine arts collection --/
structure PackagingSpecs where
  totalVolume : ℝ  -- Total volume needed in cubic inches
  boxCost : ℝ      -- Cost per box in dollars
  totalCost : ℝ    -- Total cost spent on boxes in dollars

/-- Calculates the minimum height of cubic boxes needed to package a collection --/
def minBoxHeight (specs : PackagingSpecs) : ℕ :=
  sorry

/-- Theorem stating that the minimum box height for the given specifications is 19 inches --/
theorem min_box_height_is_19 :
  let specs : PackagingSpecs := {
    totalVolume := 3060000,  -- 3.06 million cubic inches
    boxCost := 0.5,          -- $0.50 per box
    totalCost := 255         -- $255 total cost
  }
  minBoxHeight specs = 19 := by sorry

end NUMINAMATH_CALUDE_min_box_height_is_19_l26_2685


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l26_2660

theorem min_value_quadratic_form :
  ∀ x y : ℝ, x^2 - x*y + y^2 ≥ 0 ∧ (x^2 - x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l26_2660


namespace NUMINAMATH_CALUDE_sum_m_n_equals_19_l26_2693

theorem sum_m_n_equals_19 (m n : ℕ+) 
  (h1 : (m.val.choose n.val) * 2 = 272)
  (h2 : (m.val.factorial / (m.val - n.val).factorial) = 272) :
  m + n = 19 := by sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_19_l26_2693


namespace NUMINAMATH_CALUDE_kylie_apple_picking_l26_2620

/-- Kylie's apple picking problem -/
theorem kylie_apple_picking : ∀ (first_hour : ℕ),
  first_hour = 66 →
  (first_hour + 2 * first_hour + first_hour / 3) = 220 := by
  sorry

end NUMINAMATH_CALUDE_kylie_apple_picking_l26_2620


namespace NUMINAMATH_CALUDE_work_completion_time_l26_2623

theorem work_completion_time (a_time b_time : ℝ) (work_left : ℝ) (days_worked : ℝ) : 
  a_time = 15 →
  b_time = 20 →
  work_left = 0.5333333333333333 →
  (1 / a_time + 1 / b_time) * days_worked = 1 - work_left →
  days_worked = 4 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l26_2623


namespace NUMINAMATH_CALUDE_maxwell_age_proof_l26_2651

/-- Maxwell's current age --/
def maxwell_age : ℕ := 6

/-- Maxwell's sister's current age --/
def sister_age : ℕ := 2

/-- Years into the future when the age relationship holds --/
def years_future : ℕ := 2

theorem maxwell_age_proof :
  maxwell_age = 6 ∧
  sister_age = 2 ∧
  maxwell_age + years_future = 2 * (sister_age + years_future) :=
by sorry

end NUMINAMATH_CALUDE_maxwell_age_proof_l26_2651


namespace NUMINAMATH_CALUDE_polynomial_factorization_l26_2672

theorem polynomial_factorization :
  ∀ x : ℝ, (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 7) = 
           (x^2 + 7*x + 2) * (x^2 + 5*x + 19) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l26_2672


namespace NUMINAMATH_CALUDE_complement_of_union_is_multiples_of_three_l26_2697

-- Define the set of integers
variable (U : Set Int)

-- Define sets A and B
def A : Set Int := {x | ∃ k : Int, x = 3 * k + 1}
def B : Set Int := {x | ∃ k : Int, x = 3 * k + 2}

-- State the theorem
theorem complement_of_union_is_multiples_of_three (hU : U = Set.univ) :
  (U \ (A ∪ B)) = {x : Int | ∃ k : Int, x = 3 * k} :=
sorry

end NUMINAMATH_CALUDE_complement_of_union_is_multiples_of_three_l26_2697


namespace NUMINAMATH_CALUDE_max_x_value_l26_2613

theorem max_x_value (x : ℝ) : 
  ((5 * x - 20) / (4 * x - 5))^2 + (5 * x - 20) / (4 * x - 5) = 18 → x ≤ 50 / 29 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l26_2613


namespace NUMINAMATH_CALUDE_dividend_calculation_l26_2608

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 19)
  (h2 : quotient = 9)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 176 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l26_2608


namespace NUMINAMATH_CALUDE_gcd_117_182_f_neg_one_l26_2601

-- Define the polynomial f(x)
def f (x : ℤ) : ℤ := 1 - 9*x + 8*x^2 - 4*x^4 + 5*x^5 + 3*x^6

-- Theorem for the GCD of 117 and 182
theorem gcd_117_182 : Nat.gcd 117 182 = 13 := by sorry

-- Theorem for the value of f(-1)
theorem f_neg_one : f (-1) = 12 := by sorry

end NUMINAMATH_CALUDE_gcd_117_182_f_neg_one_l26_2601


namespace NUMINAMATH_CALUDE_rosie_pie_making_l26_2678

/-- Represents the number of pies that can be made from a given number of apples -/
def pies_from_apples (apples : ℚ) : ℚ :=
  (2 / 9) * apples

/-- Represents the number of apples left after making pies -/
def apples_left (total_apples : ℚ) (pies_made : ℚ) : ℚ :=
  total_apples - (pies_made * (9 / 2))

theorem rosie_pie_making (total_apples : ℚ) 
  (h1 : total_apples = 36) : 
  pies_from_apples total_apples = 8 ∧ 
  apples_left total_apples (pies_from_apples total_apples) = 0 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pie_making_l26_2678


namespace NUMINAMATH_CALUDE_range_of_y_over_x_l26_2663

theorem range_of_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 3) :
  ∃ (k : ℝ), y / x = k ∧ -Real.sqrt 3 ≤ k ∧ k ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_y_over_x_l26_2663


namespace NUMINAMATH_CALUDE_largest_prime_factor_is_29_l26_2652

def numbers : List Nat := [145, 187, 221, 299, 169]

/-- Returns the largest prime factor of a natural number -/
def largestPrimeFactor (n : Nat) : Nat :=
  sorry

theorem largest_prime_factor_is_29 : 
  ∀ n ∈ numbers, largestPrimeFactor n ≤ 29 ∧ 
  ∃ m ∈ numbers, largestPrimeFactor m = 29 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_is_29_l26_2652


namespace NUMINAMATH_CALUDE_cost_effectiveness_l26_2634

/-- Represents the cost-effective choice between two malls --/
inductive Choice
  | MallA
  | MallB
  | Either

/-- Calculates the price per unit for Mall A based on the number of items --/
def mall_a_price (n : ℕ) : ℚ :=
  if n * 4 ≤ 40 then 80 - n * 4
  else 40

/-- Calculates the price per unit for Mall B --/
def mall_b_price : ℚ := 80 * (1 - 0.3)

/-- Determines the cost-effective choice based on the number of employees --/
def cost_effective_choice (num_employees : ℕ) : Choice :=
  if num_employees < 6 then Choice.MallB
  else if num_employees = 6 then Choice.Either
  else Choice.MallA

theorem cost_effectiveness 
  (num_employees : ℕ) : 
  (cost_effective_choice num_employees = Choice.MallB ↔ num_employees < 6) ∧
  (cost_effective_choice num_employees = Choice.Either ↔ num_employees = 6) ∧
  (cost_effective_choice num_employees = Choice.MallA ↔ num_employees > 6) :=
sorry

end NUMINAMATH_CALUDE_cost_effectiveness_l26_2634


namespace NUMINAMATH_CALUDE_unique_solution_condition_l26_2616

theorem unique_solution_condition (a b : ℝ) :
  (∃! x : ℝ, 3 * x - 5 + a = b * x + 1) ↔ b ≠ 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l26_2616


namespace NUMINAMATH_CALUDE_union_complement_equal_l26_2640

def U : Finset Nat := {0,1,2,4,6,8}
def M : Finset Nat := {0,4,6}
def N : Finset Nat := {0,1,6}

theorem union_complement_equal : M ∪ (U \ N) = {0,2,4,6,8} := by sorry

end NUMINAMATH_CALUDE_union_complement_equal_l26_2640


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l26_2607

theorem smaller_solution_quadratic_equation :
  ∃ x : ℝ, x^2 + 10*x - 40 = 0 ∧ 
  (∀ y : ℝ, y^2 + 10*y - 40 = 0 → x ≤ y) ∧
  x = -8 := by
sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l26_2607


namespace NUMINAMATH_CALUDE_log_simplification_l26_2668

theorem log_simplification (u v w t : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) (ht : t > 0) :
  Real.log (u / v) + Real.log (v / (2 * w)) + Real.log (w / (4 * t)) - Real.log (u / t) = Real.log (1 / 8) := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l26_2668


namespace NUMINAMATH_CALUDE_equation_solution_l26_2674

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  -2 * x^2 = (4 * x + 2) / (x - 2) ↔ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l26_2674


namespace NUMINAMATH_CALUDE_meeting_point_distance_l26_2600

/-- Proves that the distance between Jack and Jill's meeting point and the hilltop is 35/27 km -/
theorem meeting_point_distance (total_distance : ℝ) (uphill_distance : ℝ)
  (jack_start_earlier : ℝ) (jack_uphill_speed : ℝ) (jack_downhill_speed : ℝ)
  (jill_uphill_speed : ℝ) :
  total_distance = 10 →
  uphill_distance = 5 →
  jack_start_earlier = 1/6 →
  jack_uphill_speed = 15 →
  jack_downhill_speed = 20 →
  jill_uphill_speed = 16 →
  ∃ (meeting_point_distance : ℝ), meeting_point_distance = 35/27 := by
  sorry

end NUMINAMATH_CALUDE_meeting_point_distance_l26_2600


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l26_2632

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The sum of specific terms in the sequence equals 200 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 + a 7 + a 9 + a 11 = 200

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumCondition a) : 
  4 * a 5 - 2 * a 3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l26_2632


namespace NUMINAMATH_CALUDE_prob_five_odd_in_six_rolls_l26_2642

def fair_six_sided_die : Fin 6 → ℚ
  | _ => 1 / 6

def is_odd (n : Fin 6) : Bool :=
  n.val % 2 = 1

def prob_exactly_k_odd (k : Nat) (n : Nat) : ℚ :=
  (Nat.choose n k) * (1/2)^k * (1/2)^(n-k)

theorem prob_five_odd_in_six_rolls :
  prob_exactly_k_odd 5 6 = 3/32 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_odd_in_six_rolls_l26_2642


namespace NUMINAMATH_CALUDE_x_squared_mod_25_l26_2606

theorem x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 2 * x ≡ 22 [ZMOD 25]) :
  x^2 ≡ 9 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_25_l26_2606


namespace NUMINAMATH_CALUDE_emergency_kit_problem_l26_2611

/-- Given the conditions of Veronica's emergency-preparedness kits problem, 
    prove that the number of food cans must be a multiple of 4 and at least 4. -/
theorem emergency_kit_problem (num_water_bottles : Nat) (num_food_cans : Nat) :
  num_water_bottles = 20 →
  num_water_bottles % 4 = 0 →
  num_food_cans % 4 = 0 →
  (num_water_bottles + num_food_cans) % 4 = 0 →
  num_food_cans ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_emergency_kit_problem_l26_2611


namespace NUMINAMATH_CALUDE_emmalyn_fence_count_l26_2679

/-- The number of fences Emmalyn painted -/
def number_of_fences : ℕ := 50

/-- The price per meter in dollars -/
def price_per_meter : ℚ := 0.20

/-- The length of each fence in meters -/
def fence_length : ℕ := 500

/-- The total earnings in dollars -/
def total_earnings : ℕ := 5000

/-- Theorem stating that the number of fences Emmalyn painted is correct -/
theorem emmalyn_fence_count :
  number_of_fences = total_earnings / (price_per_meter * fence_length) := by
  sorry

end NUMINAMATH_CALUDE_emmalyn_fence_count_l26_2679


namespace NUMINAMATH_CALUDE_inner_probability_is_16_25_l26_2614

/-- The size of one side of the square checkerboard -/
def boardSize : ℕ := 10

/-- The total number of squares on the checkerboard -/
def totalSquares : ℕ := boardSize * boardSize

/-- The number of squares on the perimeter of the checkerboard -/
def perimeterSquares : ℕ := 4 * boardSize - 4

/-- The number of squares not on the perimeter of the checkerboard -/
def innerSquares : ℕ := totalSquares - perimeterSquares

/-- The probability of choosing a square not on the perimeter -/
def innerProbability : ℚ := innerSquares / totalSquares

theorem inner_probability_is_16_25 : innerProbability = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_inner_probability_is_16_25_l26_2614


namespace NUMINAMATH_CALUDE_k_range_l26_2659

-- Define the function f(x)
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- Define the interval (5, 20)
def interval : Set ℝ := Set.Ioo 5 20

-- Define the property of having no maximum or minimum in the interval
def no_extremum (g : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x ∈ S, ∃ y ∈ S, g y > g x ∧ ∃ z ∈ S, g z < g x

-- State the theorem
theorem k_range (k : ℝ) :
  no_extremum (f k) interval → k ∈ Set.Iic 40 ∪ Set.Ici 160 := by sorry

end NUMINAMATH_CALUDE_k_range_l26_2659


namespace NUMINAMATH_CALUDE_popped_kernel_probability_l26_2666

theorem popped_kernel_probability (total : ℝ) (white : ℝ) (yellow : ℝ) 
  (white_pop_rate : ℝ) (yellow_pop_rate : ℝ) :
  white / total = 3 / 4 →
  yellow / total = 1 / 4 →
  white_pop_rate = 2 / 5 →
  yellow_pop_rate = 3 / 4 →
  (white * white_pop_rate) / ((white * white_pop_rate) + (yellow * yellow_pop_rate)) = 24 / 39 := by
  sorry

end NUMINAMATH_CALUDE_popped_kernel_probability_l26_2666


namespace NUMINAMATH_CALUDE_blueberry_muffin_probability_l26_2635

theorem blueberry_muffin_probability :
  let n : ℕ := 7
  let k : ℕ := 5
  let p : ℚ := 3/4
  let q : ℚ := 1 - p
  Nat.choose n k * p^k * q^(n-k) = 5103/16384 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_muffin_probability_l26_2635


namespace NUMINAMATH_CALUDE_f_of_three_eq_seventeen_l26_2686

/-- Given a function f(x) = ax + bx + c where c is a constant,
    if f(1) = 7 and f(2) = 12, then f(3) = 17 -/
theorem f_of_three_eq_seventeen
  (f : ℝ → ℝ)
  (a b c : ℝ)
  (h1 : ∀ x, f x = a * x + b * x + c)
  (h2 : f 1 = 7)
  (h3 : f 2 = 12) :
  f 3 = 17 := by
sorry

end NUMINAMATH_CALUDE_f_of_three_eq_seventeen_l26_2686


namespace NUMINAMATH_CALUDE_optimal_strategy_l26_2628

/-- Represents the expected score when answering question A first -/
def E_xi (P1 P2 a b : ℝ) : ℝ := a * P1 * (1 - P2) + (a + b) * P1 * P2

/-- Represents the expected score when answering question B first -/
def E_epsilon (P1 P2 a b : ℝ) : ℝ := b * P2 * (1 - P1) + (a + b) * P1 * P2

/-- The theorem states that given P1 = 2/5, a = 10, b = 20, 
    choosing to answer question A first is optimal when 0 ≤ P2 ≤ 1/4 -/
theorem optimal_strategy (P2 : ℝ) :
  0 ≤ P2 ∧ P2 ≤ 1/4 ↔ E_xi (2/5) P2 10 20 ≥ E_epsilon (2/5) P2 10 20 := by
  sorry

end NUMINAMATH_CALUDE_optimal_strategy_l26_2628


namespace NUMINAMATH_CALUDE_equation_solution_l26_2698

theorem equation_solution :
  ∃ x : ℝ, x ≠ 4 ∧ (x - 3) / (4 - x) - 1 = 1 / (x - 4) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l26_2698


namespace NUMINAMATH_CALUDE_marble_arrangement_remainder_l26_2692

/-- Represents the number of green marbles --/
def green_marbles : ℕ := 7

/-- Represents the minimum number of red marbles required --/
def min_red_marbles : ℕ := green_marbles + 1

/-- Represents the maximum number of additional red marbles that can be added --/
def max_additional_reds : ℕ := min_red_marbles

/-- Represents the total number of spaces where additional red marbles can be placed --/
def total_spaces : ℕ := green_marbles + 1

/-- Represents the number of ways to arrange the marbles --/
def arrangement_count : ℕ := Nat.choose (max_additional_reds + total_spaces - 1) (total_spaces - 1)

theorem marble_arrangement_remainder :
  arrangement_count % 1000 = 435 := by sorry

end NUMINAMATH_CALUDE_marble_arrangement_remainder_l26_2692


namespace NUMINAMATH_CALUDE_force_in_elevator_l26_2610

/-- The force exerted by a person on the floor of a decelerating elevator -/
def force_on_elevator_floor (mass : ℝ) (gravity_accel : ℝ) (elevator_decel : ℝ) : ℝ :=
  mass * (gravity_accel - elevator_decel)

/-- Theorem: The force exerted by a 70 kg person on the floor of an elevator
    decelerating at 5 m/s² with gravitational acceleration of 10 m/s² is 350 N -/
theorem force_in_elevator :
  let mass := 70
  let gravity_accel := 10
  let elevator_decel := 5
  force_on_elevator_floor mass gravity_accel elevator_decel = 350 := by
  sorry

end NUMINAMATH_CALUDE_force_in_elevator_l26_2610


namespace NUMINAMATH_CALUDE_triangle_existence_theorem_l26_2649

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_x_values : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

theorem triangle_existence_theorem :
  ∀ x : ℕ, x > 0 → (triangle_exists 8 11 (x + 3) ↔ x ∈ valid_x_values) :=
by sorry

end NUMINAMATH_CALUDE_triangle_existence_theorem_l26_2649


namespace NUMINAMATH_CALUDE_triangle_area_in_nested_rectangles_l26_2641

/-- Given a rectangle with dimensions a × b and a smaller rectangle inside with dimensions u × v,
    where the sides are parallel, the area of one of the four congruent right triangles formed by
    connecting the vertices of the smaller rectangle to the midpoints of the sides of the larger
    rectangle is (a-u)(b-v)/8. -/
theorem triangle_area_in_nested_rectangles (a b u v : ℝ) (ha : 0 < a) (hb : 0 < b)
    (hu : 0 < u) (hv : 0 < v) (hu_lt_a : u < a) (hv_lt_b : v < b) :
  (a - u) * (b - v) / 8 = (a - u) * (b - v) / 8 := by sorry

end NUMINAMATH_CALUDE_triangle_area_in_nested_rectangles_l26_2641


namespace NUMINAMATH_CALUDE_negation_equivalence_l26_2677

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l26_2677


namespace NUMINAMATH_CALUDE_heart_then_club_probability_l26_2665

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (hearts : ℕ)
  (clubs : ℕ)

/-- Calculates the probability of drawing a heart first and a club second from a standard deck -/
def probability_heart_then_club (d : Deck) : ℚ :=
  (d.hearts : ℚ) / d.total_cards * d.clubs / (d.total_cards - 1)

/-- Theorem stating the probability of drawing a heart first and a club second from a standard 52-card deck -/
theorem heart_then_club_probability :
  let standard_deck : Deck := ⟨52, 13, 13⟩
  probability_heart_then_club standard_deck = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_heart_then_club_probability_l26_2665


namespace NUMINAMATH_CALUDE_two_times_binomial_twelve_choose_three_l26_2644

theorem two_times_binomial_twelve_choose_three : 2 * (Nat.choose 12 3) = 440 := by
  sorry

end NUMINAMATH_CALUDE_two_times_binomial_twelve_choose_three_l26_2644


namespace NUMINAMATH_CALUDE_specific_factory_production_l26_2681

/-- A factory that produces toys -/
structure ToyFactory where
  workingDaysPerWeek : ℕ
  dailyProduction : ℕ
  constantProduction : Prop

/-- Calculate the weekly production of a toy factory -/
def weeklyProduction (factory : ToyFactory) : ℕ :=
  factory.workingDaysPerWeek * factory.dailyProduction

/-- Theorem stating the weekly production of a specific factory -/
theorem specific_factory_production :
  ∀ (factory : ToyFactory),
    factory.workingDaysPerWeek = 4 →
    factory.dailyProduction = 1375 →
    factory.constantProduction →
    weeklyProduction factory = 5500 := by
  sorry

end NUMINAMATH_CALUDE_specific_factory_production_l26_2681


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l26_2647

-- Define the position function
def s (t : ℝ) : ℝ := 5 * t^2

-- Define the velocity function as the derivative of the position function
def v (t : ℝ) : ℝ := 10 * t

-- Theorem stating that the instantaneous velocity at t=3 is 30
theorem instantaneous_velocity_at_3 : v 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l26_2647
