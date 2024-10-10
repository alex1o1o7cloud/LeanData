import Mathlib

namespace abs_neg_sqrt_six_l2485_248529

theorem abs_neg_sqrt_six : |(-Real.sqrt 6)| = Real.sqrt 6 := by
  sorry

end abs_neg_sqrt_six_l2485_248529


namespace ratio_x_to_y_l2485_248503

theorem ratio_x_to_y (x y : ℝ) (h : y = 0.25 * x) : x / y = 4 := by
  sorry

end ratio_x_to_y_l2485_248503


namespace symmetric_quadratic_property_symmetric_quadratic_comparison_l2485_248536

/-- A quadratic function with a positive leading coefficient and symmetric about x = 2 -/
def symmetric_quadratic (a b c : ℝ) (h : a > 0) : ℝ → ℝ :=
  fun x ↦ a * x^2 + b * x + c

theorem symmetric_quadratic_property {a b c : ℝ} (h : a > 0) :
  ∀ x, symmetric_quadratic a b c h (2 + x) = symmetric_quadratic a b c h (2 - x) :=
by sorry

theorem symmetric_quadratic_comparison {a b c : ℝ} (h : a > 0) :
  symmetric_quadratic a b c h 0.5 > symmetric_quadratic a b c h π :=
by sorry

end symmetric_quadratic_property_symmetric_quadratic_comparison_l2485_248536


namespace fraction_sum_division_simplification_l2485_248552

theorem fraction_sum_division_simplification :
  (3 : ℚ) / 7 + 5 / 8 + 1 / 3 / ((5 : ℚ) / 12 + 2 / 9) = 2097 / 966 := by
  sorry

end fraction_sum_division_simplification_l2485_248552


namespace set_problem_l2485_248546

def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, 3*a}

theorem set_problem (a : ℝ) :
  (A ⊆ B a → 4/3 ≤ a ∧ a ≤ 2) ∧
  (A ∩ B a ≠ ∅ → 2/3 < a ∧ a < 4) := by
  sorry

end set_problem_l2485_248546


namespace parcel_weight_sum_l2485_248508

theorem parcel_weight_sum (x y z : ℝ) 
  (h1 : x + y = 168) 
  (h2 : y + z = 174) 
  (h3 : x + z = 180) : 
  x + y + z = 261 := by
  sorry

end parcel_weight_sum_l2485_248508


namespace golden_ratio_exponential_monotonicity_l2485_248545

theorem golden_ratio_exponential_monotonicity 
  (a : ℝ) 
  (f : ℝ → ℝ) 
  (m n : ℝ) 
  (h1 : a = (Real.sqrt 5 - 1) / 2) 
  (h2 : ∀ x, f x = a ^ x) 
  (h3 : f m > f n) : 
  m < n := by
sorry

end golden_ratio_exponential_monotonicity_l2485_248545


namespace f_strictly_increasing_l2485_248570

def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

theorem f_strictly_increasing :
  (∀ x y, x < y ∧ y < -1 → f x < f y) ∧
  (∀ x y, 11 < x ∧ x < y → f x < f y) :=
sorry

end f_strictly_increasing_l2485_248570


namespace rectangle_uncovered_area_l2485_248539

/-- The area of the portion of a rectangle not covered by four circles --/
theorem rectangle_uncovered_area (rectangle_length : ℝ) (rectangle_width : ℝ) (circle_radius : ℝ) :
  rectangle_length = 4 →
  rectangle_width = 8 →
  circle_radius = 1 →
  (rectangle_length * rectangle_width) - (4 * Real.pi * circle_radius ^ 2) = 32 - 4 * Real.pi := by
  sorry

#check rectangle_uncovered_area

end rectangle_uncovered_area_l2485_248539


namespace tea_party_waiting_time_l2485_248593

/-- Mad Hatter's clock speed relative to real time -/
def mad_hatter_clock_speed : ℚ := 5/4

/-- March Hare's clock speed relative to real time -/
def march_hare_clock_speed : ℚ := 5/6

/-- Time shown on both clocks when they meet (in hours after noon) -/
def meeting_time : ℚ := 5

theorem tea_party_waiting_time :
  let mad_hatter_arrival_time := meeting_time / mad_hatter_clock_speed
  let march_hare_arrival_time := meeting_time / march_hare_clock_speed
  march_hare_arrival_time - mad_hatter_arrival_time = 2 := by sorry

end tea_party_waiting_time_l2485_248593


namespace candy_box_price_increase_candy_box_price_after_increase_l2485_248534

theorem candy_box_price_increase (initial_soda_price : ℝ) 
  (candy_increase_rate : ℝ) (soda_increase_rate : ℝ) 
  (initial_total_price : ℝ) : ℝ :=
  let initial_candy_price := initial_total_price - initial_soda_price
  let final_candy_price := initial_candy_price * (1 + candy_increase_rate)
  final_candy_price

theorem candy_box_price_after_increase :
  candy_box_price_increase 12 0.25 0.5 16 = 5 := by
  sorry

end candy_box_price_increase_candy_box_price_after_increase_l2485_248534


namespace two_planes_parallel_to_same_line_are_parallel_two_planes_parallel_to_same_line_not_always_parallel_l2485_248542

-- Define the concept of a plane in 3D space
variable (Plane : Type)

-- Define the concept of a line in 3D space
variable (Line : Type)

-- Define the parallel relation between planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the parallel relation between a plane and a line
variable (parallel_plane_line : Plane → Line → Prop)

-- Theorem to be proven false
theorem two_planes_parallel_to_same_line_are_parallel 
  (p1 p2 : Plane) (l : Line) : 
  parallel_plane_line p1 l → parallel_plane_line p2 l → parallel_planes p1 p2 := by
  sorry

-- The actual theorem should be that the above statement is false
theorem two_planes_parallel_to_same_line_not_always_parallel : 
  ¬∀ (p1 p2 : Plane) (l : Line), 
    parallel_plane_line p1 l → parallel_plane_line p2 l → parallel_planes p1 p2 := by
  sorry

end two_planes_parallel_to_same_line_are_parallel_two_planes_parallel_to_same_line_not_always_parallel_l2485_248542


namespace intersection_A_notB_when_a_is_neg_two_union_A_B_equals_B_implies_a_range_l2485_248531

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Define the complement of B
def notB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Theorem for part I
theorem intersection_A_notB_when_a_is_neg_two : 
  A (-2) ∩ notB = {x | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part II
theorem union_A_B_equals_B_implies_a_range (a : ℝ) : 
  A a ∪ B = B → a < -4 ∨ a > 5 := by sorry

end intersection_A_notB_when_a_is_neg_two_union_A_B_equals_B_implies_a_range_l2485_248531


namespace julies_savings_l2485_248590

theorem julies_savings (monthly_salary : ℝ) (savings_fraction : ℝ) : 
  monthly_salary > 0 →
  savings_fraction > 0 →
  savings_fraction < 1 →
  12 * monthly_salary * savings_fraction = 4 * monthly_salary * (1 - savings_fraction) →
  1 - savings_fraction = 3/4 := by
sorry

end julies_savings_l2485_248590


namespace triangulation_count_l2485_248566

/-- A triangulation of a square with marked interior points. -/
structure SquareTriangulation where
  /-- The number of marked points inside the square. -/
  num_points : ℕ
  /-- The number of triangles in the triangulation. -/
  num_triangles : ℕ

/-- Theorem stating the number of triangles in a specific triangulation. -/
theorem triangulation_count (t : SquareTriangulation) 
  (h_points : t.num_points = 100) : 
  t.num_triangles = 202 := by sorry

end triangulation_count_l2485_248566


namespace total_accidents_l2485_248512

/-- Represents the number of vehicles involved in accidents per 100 million vehicles -/
def A (k : ℝ) (x : ℝ) : ℝ := 96 + k * x

/-- The constant k for morning hours -/
def k_morning : ℝ := 1

/-- The constant k for evening hours -/
def k_evening : ℝ := 3

/-- The number of vehicles (in billions) during morning hours -/
def x_morning : ℝ := 2

/-- The number of vehicles (in billions) during evening hours -/
def x_evening : ℝ := 1

/-- Theorem stating the total number of vehicles involved in accidents -/
theorem total_accidents : 
  A k_morning (100 * x_morning) + A k_evening (100 * x_evening) = 5192 := by
  sorry

end total_accidents_l2485_248512


namespace cats_in_center_l2485_248588

/-- The number of cats that can jump -/
def jump : ℕ := 60

/-- The number of cats that can fetch -/
def fetch : ℕ := 35

/-- The number of cats that can spin -/
def spin : ℕ := 40

/-- The number of cats that can jump and fetch -/
def jump_fetch : ℕ := 20

/-- The number of cats that can fetch and spin -/
def fetch_spin : ℕ := 15

/-- The number of cats that can jump and spin -/
def jump_spin : ℕ := 25

/-- The number of cats that can do all three tricks -/
def all_three : ℕ := 10

/-- The number of cats that can do no tricks -/
def no_tricks : ℕ := 8

/-- The total number of cats in the center -/
def total_cats : ℕ := 93

theorem cats_in_center : 
  jump + fetch + spin - jump_fetch - fetch_spin - jump_spin + all_three + no_tricks = total_cats :=
by sorry

end cats_in_center_l2485_248588


namespace total_components_is_900_l2485_248577

/-- Represents the total number of components --/
def total_components : ℕ := 900

/-- Represents the number of type B components --/
def type_b_components : ℕ := 300

/-- Represents the number of type C components --/
def type_c_components : ℕ := 200

/-- Represents the sample size --/
def sample_size : ℕ := 45

/-- Represents the number of type A components in the sample --/
def sample_type_a : ℕ := 20

/-- Represents the number of type C components in the sample --/
def sample_type_c : ℕ := 10

/-- Theorem stating that the total number of components is 900 --/
theorem total_components_is_900 :
  total_components = 900 ∧
  type_b_components = 300 ∧
  type_c_components = 200 ∧
  sample_size = 45 ∧
  sample_type_a = 20 ∧
  sample_type_c = 10 ∧
  (sample_type_c : ℚ) / (sample_size : ℚ) = (type_c_components : ℚ) / (total_components : ℚ) :=
by sorry

#check total_components_is_900

end total_components_is_900_l2485_248577


namespace largest_number_on_board_l2485_248518

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

def satisfies_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ n % 6 = 0 ∧ ends_in_4 n

theorem largest_number_on_board :
  ∃ (m : ℕ), satisfies_conditions m ∧
  ∀ (n : ℕ), satisfies_conditions n → n ≤ m :=
by
  sorry

end largest_number_on_board_l2485_248518


namespace quadratic_equations_solutions_l2485_248558

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 ∧ x₁^2 - 4*x₁ - 1 = 0 ∧ x₂^2 - 4*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -1/3 ∧ x₂ = 2 ∧ x₁*(3*x₁ + 1) = 2*(3*x₁ + 1) ∧ x₂*(3*x₂ + 1) = 2*(3*x₂ + 1)) ∧
  (∃ x₁ x₂ : ℝ, x₁ = (-1 + Real.sqrt 33) / 4 ∧ x₂ = (-1 - Real.sqrt 33) / 4 ∧ 2*x₁^2 + x₁ - 4 = 0 ∧ 2*x₂^2 + x₂ - 4 = 0) ∧
  (∀ x : ℝ, 4*x^2 - 3*x + 1 ≠ 0) :=
by sorry

end quadratic_equations_solutions_l2485_248558


namespace triangle_perimeter_triangle_perimeter_holds_l2485_248560

/-- Given a triangle with two sides of lengths 2 and 5, and the third side being an odd number,
    the perimeter of the triangle is 12. -/
theorem triangle_perimeter : ℕ → Prop :=
  fun third_side =>
    (third_side > 0) →  -- Ensure positive length
    (third_side % 2 = 1) →  -- Odd number condition
    (2 < third_side) →  -- Lower bound from triangle inequality
    (third_side < 7) →  -- Upper bound from triangle inequality
    (2 + 5 + third_side = 12)

/-- The theorem holds. -/
theorem triangle_perimeter_holds : ∃ n, triangle_perimeter n :=
sorry

end triangle_perimeter_triangle_perimeter_holds_l2485_248560


namespace smallest_integer_with_remainders_l2485_248533

theorem smallest_integer_with_remainders (k : ℕ) : k = 61 ↔ 
  (k > 1) ∧ 
  (∀ m : ℕ, m < k → 
    (m % 12 ≠ 1 ∨ m % 5 ≠ 1 ∨ m % 3 ≠ 1)) ∧
  (k % 12 = 1 ∧ k % 5 = 1 ∧ k % 3 = 1) :=
by sorry

end smallest_integer_with_remainders_l2485_248533


namespace cake_frosting_time_difference_l2485_248598

/-- The time difference in frosting cakes with normal and sprained conditions -/
theorem cake_frosting_time_difference 
  (normal_time : ℕ) -- Time to frost one cake under normal conditions
  (sprained_time : ℕ) -- Time to frost one cake with sprained wrist
  (num_cakes : ℕ) -- Number of cakes to frost
  (h1 : normal_time = 5) -- Normal frosting time is 5 minutes
  (h2 : sprained_time = 8) -- Sprained wrist frosting time is 8 minutes
  (h3 : num_cakes = 10) -- Number of cakes to frost is 10
  : sprained_time * num_cakes - normal_time * num_cakes = 30 := by
  sorry

end cake_frosting_time_difference_l2485_248598


namespace perfect_square_polynomial_l2485_248500

theorem perfect_square_polynomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 8*x + k = (x + a)^2) → k = 16 := by
  sorry

end perfect_square_polynomial_l2485_248500


namespace point_alignment_implies_m_value_l2485_248582

/-- Three points lie on the same straight line if and only if 
    the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) / (x₂ - x₁) = (y₃ - y₁) / (x₃ - x₁)

theorem point_alignment_implies_m_value :
  ∀ m : ℝ, collinear 1 (-2) 3 4 6 (m/3) → m = 39 := by
  sorry


end point_alignment_implies_m_value_l2485_248582


namespace root_product_cubic_polynomial_l2485_248521

theorem root_product_cubic_polynomial :
  let p := fun (x : ℝ) => 3 * x^3 - 4 * x^2 + x - 10
  ∃ a b c : ℝ, p a = 0 ∧ p b = 0 ∧ p c = 0 ∧ a * b * c = 10/3 :=
by sorry

end root_product_cubic_polynomial_l2485_248521


namespace minimal_moves_l2485_248537

/-- Represents a permutation of 2n numbers -/
def Permutation (n : ℕ) := Fin (2 * n) → Fin (2 * n)

/-- Represents a move that can be applied to a permutation -/
inductive Move (n : ℕ)
  | swap : Fin (2 * n) → Fin (2 * n) → Move n
  | cyclic : Fin (2 * n) → Fin (2 * n) → Fin (2 * n) → Move n

/-- Applies a move to a permutation -/
def applyMove (n : ℕ) (p : Permutation n) (m : Move n) : Permutation n :=
  sorry

/-- Checks if a permutation is in increasing order -/
def isIncreasing (n : ℕ) (p : Permutation n) : Prop :=
  sorry

/-- The main theorem: n moves are necessary and sufficient -/
theorem minimal_moves (n : ℕ) :
  (∃ (moves : List (Move n)), moves.length = n ∧
    ∀ (p : Permutation n), ∃ (appliedMoves : List (Move n)),
      appliedMoves.length ≤ n ∧
      isIncreasing n (appliedMoves.foldl (applyMove n) p)) ∧
  (∀ (k : ℕ), k < n →
    ∃ (p : Permutation n), ∀ (moves : List (Move n)),
      moves.length ≤ k → ¬isIncreasing n (moves.foldl (applyMove n) p)) :=
  sorry

end minimal_moves_l2485_248537


namespace library_book_return_days_l2485_248561

theorem library_book_return_days 
  (daily_charge : ℚ)
  (total_books : ℕ)
  (days_for_one_book : ℕ)
  (total_cost : ℚ)
  (h1 : daily_charge = 1/2)
  (h2 : total_books = 3)
  (h3 : days_for_one_book = 20)
  (h4 : total_cost = 41) :
  (total_cost - daily_charge * days_for_one_book) / (daily_charge * 2) = 31 := by
sorry

end library_book_return_days_l2485_248561


namespace min_value_expression_l2485_248564

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 2018) + (y + 1/x) * (y + 1/x - 2018) ≥ -2036162 := by
  sorry

end min_value_expression_l2485_248564


namespace tailoring_cost_james_suits_tailoring_cost_l2485_248544

theorem tailoring_cost (cost_first_suit : ℕ) (total_cost : ℕ) : ℕ :=
  let cost_second_suit := 3 * cost_first_suit
  let tailoring_cost := total_cost - cost_first_suit - cost_second_suit
  tailoring_cost

theorem james_suits_tailoring_cost : tailoring_cost 300 1400 = 200 := by
  sorry

end tailoring_cost_james_suits_tailoring_cost_l2485_248544


namespace first_year_after_2010_with_sum_of_digits_10_l2485_248505

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def isFirstYearAfter2010WithSumOfDigits10 (year : ℕ) : Prop :=
  year > 2010 ∧ 
  sumOfDigits year = 10 ∧
  ∀ y, 2010 < y ∧ y < year → sumOfDigits y ≠ 10

theorem first_year_after_2010_with_sum_of_digits_10 :
  isFirstYearAfter2010WithSumOfDigits10 2017 := by
  sorry

end first_year_after_2010_with_sum_of_digits_10_l2485_248505


namespace concave_arithmetic_sequence_condition_l2485_248509

/-- An arithmetic sequence with first term a and common difference d -/
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

/-- A sequence is concave if a_{n-1} + a_{n+1} ≥ 2a_n for n ≥ 2 -/
def is_concave (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a (n - 1) + a (n + 1) ≥ 2 * a n

theorem concave_arithmetic_sequence_condition (d : ℝ) :
  let b := arithmetic_sequence 4 d
  is_concave (λ n => b n / n) → d ≤ 4 := by
  sorry

end concave_arithmetic_sequence_condition_l2485_248509


namespace side_e_length_l2485_248572

-- Define the triangle DEF
structure Triangle where
  D : Real
  E : Real
  F : Real
  d : Real
  e : Real
  f : Real

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.E = 4 * t.D ∧ t.d = 18 ∧ t.f = 27

-- State the theorem
theorem side_e_length (t : Triangle) 
  (h : triangle_conditions t) : t.e = 27 := by
  sorry

end side_e_length_l2485_248572


namespace proportion_solution_l2485_248527

theorem proportion_solution (y : ℝ) : y / 1.35 = 5 / 9 → y = 0.75 := by
  sorry

end proportion_solution_l2485_248527


namespace find_m_value_l2485_248526

theorem find_m_value (a : ℝ) (m : ℝ) : 
  (∀ x, 2*x^2 - 3*x + a < 0 ↔ m < x ∧ x < 1) →
  (2*m^2 - 3*m + a = 0 ∧ 2*1^2 - 3*1 + a = 0) →
  m = 1/2 := by sorry

end find_m_value_l2485_248526


namespace unique_four_letter_product_l2485_248551

def letter_value (c : Char) : ℕ :=
  c.toNat - 'A'.toNat + 1

def four_letter_product (s : String) : ℕ :=
  if s.length = 4 then
    s.foldl (fun acc c => acc * letter_value c) 1
  else
    0

theorem unique_four_letter_product : ∀ s : String,
  s.length = 4 ∧ s ≠ "MNOQ" ∧ four_letter_product s = four_letter_product "MNOQ" →
  s = "NOQZ" :=
sorry

end unique_four_letter_product_l2485_248551


namespace sum_of_digits_l2485_248581

theorem sum_of_digits (a b c d : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  100 * a + 10 * b + c + 100 * d + 10 * c + b = 1100 →
  a + b + c + d = 20 := by
sorry

end sum_of_digits_l2485_248581


namespace perpendicular_lines_l2485_248519

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, 2*x + y + 2 = 0 ∧ a*x + 4*y - 2 = 0 → 
    ((-1/2) * (-a/4) = -1)) → 
  a = -8 :=
by sorry

end perpendicular_lines_l2485_248519


namespace wine_cost_proof_l2485_248587

/-- The current cost of a bottle of wine -/
def current_cost : ℝ := sorry

/-- The future cost of a bottle of wine after the price increase -/
def future_cost : ℝ := 1.25 * current_cost

/-- The increase in cost for five bottles -/
def total_increase : ℝ := 25

/-- The number of bottles -/
def num_bottles : ℕ := 5

theorem wine_cost_proof :
  (future_cost - current_cost) * num_bottles = total_increase ∧ current_cost = 20 := by sorry

end wine_cost_proof_l2485_248587


namespace euler_identity_complex_power_cexp_sum_bound_cexp_diff_not_always_bounded_l2485_248530

-- Define the complex exponential function
noncomputable def cexp (x : ℝ) : ℂ := Complex.exp (x * Complex.I)

-- Euler's formula
axiom euler_formula (x : ℝ) : cexp x = Complex.cos x + Complex.I * Complex.sin x

-- Theorem 1
theorem euler_identity : cexp π + 1 = 0 := by sorry

-- Theorem 2
theorem complex_power : (Complex.ofReal (1/2) + Complex.I * Complex.ofReal (Real.sqrt 3 / 2)) ^ 2022 = 1 := by sorry

-- Theorem 3
theorem cexp_sum_bound (x : ℝ) : Complex.abs (cexp x + cexp (-x)) ≤ 2 := by sorry

-- Theorem 4
theorem cexp_diff_not_always_bounded :
  ¬ (∀ x : ℝ, -2 ≤ (cexp x - cexp (-x)).re ∧ (cexp x - cexp (-x)).re ≤ 2 ∧
               -2 ≤ (cexp x - cexp (-x)).im ∧ (cexp x - cexp (-x)).im ≤ 2) := by sorry

end euler_identity_complex_power_cexp_sum_bound_cexp_diff_not_always_bounded_l2485_248530


namespace no_club_member_is_fraternity_member_l2485_248514

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (FraternityMember : U → Prop)
variable (ClubMember : U → Prop)
variable (Honest : U → Prop)

-- Define the given conditions
axiom some_students_not_honest : ∃ x, Student x ∧ ¬Honest x
axiom all_fraternity_members_honest : ∀ x, FraternityMember x → Honest x
axiom no_club_members_honest : ∀ x, ClubMember x → ¬Honest x

-- Theorem to prove
theorem no_club_member_is_fraternity_member :
  ∀ x, ClubMember x → ¬FraternityMember x :=
sorry

end no_club_member_is_fraternity_member_l2485_248514


namespace total_letters_in_names_l2485_248540

/-- Represents the number of letters in a person's name -/
structure NameLength where
  firstName : Nat
  surname : Nat

/-- Calculates the total number of letters in a person's full name -/
def totalLetters (name : NameLength) : Nat :=
  name.firstName + name.surname

/-- Theorem: The total number of letters in Jonathan's and his sister's names is 33 -/
theorem total_letters_in_names : 
  let jonathan : NameLength := { firstName := 8, surname := 10 }
  let sister : NameLength := { firstName := 5, surname := 10 }
  totalLetters jonathan + totalLetters sister = 33 := by
  sorry

end total_letters_in_names_l2485_248540


namespace decimal_to_fraction_l2485_248584

theorem decimal_to_fraction : 
  ∃ (n d : ℕ), d ≠ 0 ∧ 3.36 = (n : ℚ) / (d : ℚ) ∧ (n.gcd d = 1) ∧ n = 84 ∧ d = 25 := by
  sorry

end decimal_to_fraction_l2485_248584


namespace line_tangent_to_circle_l2485_248563

/-- A line with equation x = my + 2 is tangent to the circle x^2 + 2x + y^2 + 2y = 0 
    if and only if m = 1 or m = -7 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y : ℝ, x = m * y + 2 → x^2 + 2*x + y^2 + 2*y ≠ 0) ∧ 
  (∃ x y : ℝ, x = m * y + 2 ∧ x^2 + 2*x + y^2 + 2*y = 0) ↔ 
  m = 1 ∨ m = -7 :=
sorry

end line_tangent_to_circle_l2485_248563


namespace garage_sale_pricing_l2485_248528

theorem garage_sale_pricing (prices : Finset ℕ) (radio_price : ℕ) (n : ℕ) :
  prices.card = 36 →
  prices.toList.Nodup →
  radio_price ∈ prices →
  (prices.filter (λ x => x > radio_price)).card = n - 1 →
  (prices.filter (λ x => x < radio_price)).card = 21 →
  n = 16 := by
  sorry

end garage_sale_pricing_l2485_248528


namespace regular_quadrilateral_pyramid_angle_l2485_248554

/-- A regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  /-- The angle between a slant edge and the base plane -/
  slant_base_angle : ℝ
  /-- The angle between a slant edge and the plane of the lateral face that does not contain this edge -/
  slant_lateral_angle : ℝ
  /-- The angles are equal -/
  angle_equality : slant_base_angle = slant_lateral_angle

/-- The theorem stating the angle in a regular quadrilateral pyramid -/
theorem regular_quadrilateral_pyramid_angle (pyramid : RegularQuadrilateralPyramid) :
  pyramid.slant_base_angle = Real.arctan (Real.sqrt (3 / 2)) := by
  sorry

end regular_quadrilateral_pyramid_angle_l2485_248554


namespace intersection_point_d_l2485_248555

/-- The function g(x) = 2x + c -/
def g (c : ℤ) : ℝ → ℝ := λ x ↦ 2 * x + c

theorem intersection_point_d (c d : ℤ) :
  g c (-4) = d ∧ g c d = -4 → d = -4 := by
  sorry

end intersection_point_d_l2485_248555


namespace nested_series_sum_l2485_248592

def nested_series : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * (1 + nested_series n)

theorem nested_series_sum : nested_series 5 = 126 := by
  sorry

end nested_series_sum_l2485_248592


namespace abs_min_value_min_value_at_two_unique_min_value_l2485_248583

theorem abs_min_value (x : ℝ) : |x - 2| + 3 ≥ 3 := by sorry

theorem min_value_at_two : ∃ (x : ℝ), |x - 2| + 3 = 3 := by sorry

theorem unique_min_value (x : ℝ) : |x - 2| + 3 = 3 ↔ x = 2 := by sorry

end abs_min_value_min_value_at_two_unique_min_value_l2485_248583


namespace one_integer_solution_implies_a_range_l2485_248510

theorem one_integer_solution_implies_a_range (a : ℝ) :
  (∃! x : ℤ, (x : ℝ) - a ≥ 0 ∧ 2 * (x : ℝ) - 10 < 0) →
  3 < a ∧ a ≤ 4 := by
  sorry

end one_integer_solution_implies_a_range_l2485_248510


namespace andrew_total_payment_l2485_248517

def grapes_quantity : ℕ := 15
def grapes_rate : ℕ := 98
def mangoes_quantity : ℕ := 8
def mangoes_rate : ℕ := 120
def pineapples_quantity : ℕ := 5
def pineapples_rate : ℕ := 75
def oranges_quantity : ℕ := 10
def oranges_rate : ℕ := 60

def total_cost : ℕ := 
  grapes_quantity * grapes_rate + 
  mangoes_quantity * mangoes_rate + 
  pineapples_quantity * pineapples_rate + 
  oranges_quantity * oranges_rate

theorem andrew_total_payment : total_cost = 3405 := by
  sorry

end andrew_total_payment_l2485_248517


namespace slope_angle_l2485_248532

def line_equation (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 2 = 0

theorem slope_angle (x y : ℝ) (h : line_equation x y) : 
  ∃ (θ : ℝ), θ = 120 * Real.pi / 180 ∧ Real.tan θ = -Real.sqrt 3 :=
sorry

end slope_angle_l2485_248532


namespace bomb_defusal_probability_l2485_248548

theorem bomb_defusal_probability :
  let n : ℕ := 4  -- Total number of wires
  let k : ℕ := 2  -- Number of wires that need to be cut
  let total_combinations : ℕ := n.choose k  -- Total number of possible combinations
  let successful_combinations : ℕ := 1  -- Number of successful combinations
  (successful_combinations : ℚ) / total_combinations = 1 / 6 :=
by
  sorry

end bomb_defusal_probability_l2485_248548


namespace dance_partners_l2485_248513

theorem dance_partners (total_participants : ℕ) (n : ℕ) : 
  total_participants = 42 →
  (∀ k : ℕ, k ≥ 1 ∧ k ≤ n → k + 6 ≤ total_participants - n) →
  n + 6 = total_participants - n →
  n = 18 ∧ total_participants - n = 24 :=
by sorry

end dance_partners_l2485_248513


namespace lowest_possible_price_l2485_248522

def manufacturer_price : ℝ := 45.00
def max_regular_discount : ℝ := 0.30
def additional_sale_discount : ℝ := 0.20

theorem lowest_possible_price :
  let regular_discounted_price := manufacturer_price * (1 - max_regular_discount)
  let final_price := regular_discounted_price * (1 - additional_sale_discount)
  final_price = 25.20 := by
sorry

end lowest_possible_price_l2485_248522


namespace hotel_has_21_rooms_l2485_248597

/-- Represents the inventory and room requirements for a hotel. -/
structure HotelInventory where
  total_lamps : ℕ
  total_chairs : ℕ
  total_bed_sheets : ℕ
  lamps_per_room : ℕ
  chairs_per_room : ℕ
  bed_sheets_per_room : ℕ

/-- Calculates the number of rooms in a hotel based on its inventory and room requirements. -/
def calculateRooms (inventory : HotelInventory) : ℕ :=
  min (inventory.total_lamps / inventory.lamps_per_room)
    (min (inventory.total_chairs / inventory.chairs_per_room)
      (inventory.total_bed_sheets / inventory.bed_sheets_per_room))

/-- Theorem stating that the hotel has 21 rooms based on the given inventory. -/
theorem hotel_has_21_rooms (inventory : HotelInventory)
    (h1 : inventory.total_lamps = 147)
    (h2 : inventory.total_chairs = 84)
    (h3 : inventory.total_bed_sheets = 210)
    (h4 : inventory.lamps_per_room = 7)
    (h5 : inventory.chairs_per_room = 4)
    (h6 : inventory.bed_sheets_per_room = 10) :
    calculateRooms inventory = 21 := by
  sorry

end hotel_has_21_rooms_l2485_248597


namespace currency_notes_theorem_l2485_248547

theorem currency_notes_theorem (x y z : ℕ) : 
  x + y + z = 130 →
  95 * x + 45 * y + 20 * z = 7000 →
  75 * x + 25 * y = 4400 := by
sorry

end currency_notes_theorem_l2485_248547


namespace course_passing_logic_l2485_248524

variable (Student : Type)
variable (answered_correctly : Student → Prop)
variable (passed_course : Student → Prop)

theorem course_passing_logic :
  (∀ s : Student, answered_correctly s → passed_course s) →
  (∀ s : Student, ¬passed_course s → ¬answered_correctly s) :=
by sorry

end course_passing_logic_l2485_248524


namespace mod_equivalence_problem_l2485_248565

theorem mod_equivalence_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 21 ∧ -200 ≡ n [ZMOD 21] ∧ n = 10 := by
  sorry

end mod_equivalence_problem_l2485_248565


namespace homothety_circle_transformation_l2485_248501

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℚ

/-- Applies a homothety transformation to a point -/
def homothety (center : Point) (scale : ℚ) (p : Point) : Point :=
  { x := center.x + scale * (p.x - center.x)
  , y := center.y + scale * (p.y - center.y) }

theorem homothety_circle_transformation :
  let O : Point := { x := 3, y := 4 }
  let originalCircle : Circle := { center := O, radius := 8 }
  let P : Point := { x := 11, y := 12 }
  let scale : ℚ := 2/3
  let newCenter : Point := homothety P scale O
  let newRadius : ℚ := scale * originalCircle.radius
  newCenter.x = 17/3 ∧ newCenter.y = 20/3 ∧ newRadius = 16/3 :=
by sorry

end homothety_circle_transformation_l2485_248501


namespace polygon_symmetry_l2485_248573

-- Define a convex polygon
def ConvexPolygon : Type := sorry

-- Define a point inside a polygon
def PointInside (P : ConvexPolygon) : Type := sorry

-- Define a line passing through a point
def LineThroughPoint (P : ConvexPolygon) (O : PointInside P) : Type := sorry

-- Define the property of a line dividing the polygon area in half
def DividesAreaInHalf (P : ConvexPolygon) (O : PointInside P) (l : LineThroughPoint P O) : Prop := sorry

-- Define central symmetry of a polygon
def CentrallySymmetric (P : ConvexPolygon) : Prop := sorry

-- Define center of symmetry
def CenterOfSymmetry (P : ConvexPolygon) (O : PointInside P) : Prop := sorry

-- The main theorem
theorem polygon_symmetry (P : ConvexPolygon) (O : PointInside P) 
  (h : ∀ (l : LineThroughPoint P O), DividesAreaInHalf P O l) : 
  CentrallySymmetric P ∧ CenterOfSymmetry P O := by
  sorry

end polygon_symmetry_l2485_248573


namespace cubic_equation_solution_l2485_248576

theorem cubic_equation_solution (x : ℚ) : (5*x - 2)^3 + 125 = 0 ↔ x = -3/5 := by
  sorry

end cubic_equation_solution_l2485_248576


namespace executive_committee_formation_l2485_248515

/-- Represents the number of members in each department -/
def membersPerDepartment : ℕ := 10

/-- Represents the total number of departments -/
def totalDepartments : ℕ := 3

/-- Represents the size of the executive committee -/
def committeeSize : ℕ := 5

/-- Represents the total number of club members -/
def totalMembers : ℕ := membersPerDepartment * totalDepartments

/-- Calculates the number of ways to choose the executive committee -/
def waysToChooseCommittee : ℕ := 
  membersPerDepartment ^ totalDepartments * (Nat.choose (totalMembers - totalDepartments) (committeeSize - totalDepartments))

theorem executive_committee_formation :
  waysToChooseCommittee = 351000 := by sorry

end executive_committee_formation_l2485_248515


namespace exponent_addition_l2485_248599

theorem exponent_addition (a : ℝ) (m n : ℕ) : a ^ m * a ^ n = a ^ (m + n) := by
  sorry

end exponent_addition_l2485_248599


namespace infinite_solutions_cube_fifth_square_l2485_248516

theorem infinite_solutions_cube_fifth_square (x y z : ℕ+) (k : ℕ+) 
  (h : x^3 + y^5 = z^2) :
  (k^10 * x)^3 + (k^6 * y)^5 = (k^15 * z)^2 := by
  sorry

#check infinite_solutions_cube_fifth_square

end infinite_solutions_cube_fifth_square_l2485_248516


namespace champion_wins_39_l2485_248594

/-- Represents a basketball championship. -/
structure BasketballChampionship where
  n : ℕ                -- Number of teams
  totalPoints : ℕ      -- Total points of non-champion teams
  champPoints : ℕ      -- Points of the champion

/-- The number of matches won by the champion. -/
def championWins (championship : BasketballChampionship) : ℕ :=
  championship.champPoints - (championship.n - 1) * 2

/-- Theorem stating the number of matches won by the champion. -/
theorem champion_wins_39 (championship : BasketballChampionship) :
  championship.n = 27 ∧
  championship.totalPoints = 2015 ∧
  championship.champPoints = 3 * championship.n^2 - 3 * championship.n - championship.totalPoints →
  championWins championship = 39 := by
  sorry

#eval championWins { n := 27, totalPoints := 2015, champPoints := 91 }

end champion_wins_39_l2485_248594


namespace opposite_of_neg_two_l2485_248586

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- Theorem: The opposite of -2 is 2 -/
theorem opposite_of_neg_two : opposite (-2) = 2 := by
  sorry

end opposite_of_neg_two_l2485_248586


namespace class_size_l2485_248562

/-- The number of students in a class with given language course enrollments -/
theorem class_size (french : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : french = 41)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 6) :
  french + german - both + neither = 60 := by
  sorry

end class_size_l2485_248562


namespace divisible_by_seven_l2485_248523

theorem divisible_by_seven (n : ℕ) : 7 ∣ (6^(2*n+1) + 1) := by
  sorry

end divisible_by_seven_l2485_248523


namespace vacation_cost_share_l2485_248567

/-- Calculates each person's share of vacation costs -/
theorem vacation_cost_share
  (num_people : ℕ)
  (airbnb_cost : ℕ)
  (car_cost : ℕ)
  (h1 : num_people = 8)
  (h2 : airbnb_cost = 3200)
  (h3 : car_cost = 800) :
  (airbnb_cost + car_cost) / num_people = 500 := by
  sorry

#check vacation_cost_share

end vacation_cost_share_l2485_248567


namespace elimination_failure_l2485_248520

theorem elimination_failure (x y : ℝ) : 
  (2 * x - 3 * y = 5) → 
  (3 * x - 2 * y = 7) → 
  (2 * (2 * x - 3 * y) - (-3) * (3 * x - 2 * y) ≠ 0) := by
sorry

end elimination_failure_l2485_248520


namespace abcd_inequality_l2485_248506

theorem abcd_inequality (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : a^2/(1+a^2) + b^2/(1+b^2) + c^2/(1+c^2) + d^2/(1+d^2) = 1) : 
  a * b * c * d ≤ 1/9 := by
sorry

end abcd_inequality_l2485_248506


namespace gcd_102_238_l2485_248568

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l2485_248568


namespace box_volume_calculation_l2485_248525

/-- The conversion factor from feet to meters -/
def feet_to_meters : ℝ := 0.3048

/-- The edge length of each box in feet -/
def edge_length_feet : ℝ := 5

/-- The number of boxes -/
def num_boxes : ℕ := 4

/-- The total volume of the boxes in cubic meters -/
def total_volume : ℝ := 14.144

theorem box_volume_calculation :
  (num_boxes : ℝ) * (edge_length_feet * feet_to_meters)^3 = total_volume := by
  sorry

end box_volume_calculation_l2485_248525


namespace solution_implies_k_value_l2485_248535

theorem solution_implies_k_value (k : ℝ) : 
  (∃ x : ℝ, 2 * x + k = 3) → 
  (2 * 1 + k = 3) →
  k = 1 := by
sorry

end solution_implies_k_value_l2485_248535


namespace path_count_is_210_l2485_248571

/-- Number of paths on a grid from C to D -/
def num_paths (total_steps : ℕ) (right_steps : ℕ) (up_steps : ℕ) : ℕ :=
  Nat.choose total_steps up_steps

/-- Theorem: The number of different paths from C to D is 210 -/
theorem path_count_is_210 :
  num_paths 10 6 4 = 210 := by
  sorry

end path_count_is_210_l2485_248571


namespace remaining_bottles_calculation_l2485_248578

theorem remaining_bottles_calculation (small_initial big_initial medium_initial : ℕ)
  (small_sold_percent small_damaged_percent : ℚ)
  (big_sold_percent big_damaged_percent : ℚ)
  (medium_sold_percent medium_damaged_percent : ℚ)
  (h_small_initial : small_initial = 6000)
  (h_big_initial : big_initial = 15000)
  (h_medium_initial : medium_initial = 5000)
  (h_small_sold : small_sold_percent = 11/100)
  (h_small_damaged : small_damaged_percent = 3/100)
  (h_big_sold : big_sold_percent = 12/100)
  (h_big_damaged : big_damaged_percent = 2/100)
  (h_medium_sold : medium_sold_percent = 8/100)
  (h_medium_damaged : medium_damaged_percent = 4/100) :
  (small_initial - (small_initial * small_sold_percent).floor - (small_initial * small_damaged_percent).floor) +
  (big_initial - (big_initial * big_sold_percent).floor - (big_initial * big_damaged_percent).floor) +
  (medium_initial - (medium_initial * medium_sold_percent).floor - (medium_initial * medium_damaged_percent).floor) = 22560 := by
sorry

end remaining_bottles_calculation_l2485_248578


namespace magnitude_of_vector_combination_l2485_248557

/-- Given two vectors a and b in R^2, prove that the magnitude of 2a - b is √17 -/
theorem magnitude_of_vector_combination (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (3, -2) → ‖2 • a - b‖ = Real.sqrt 17 := by
  sorry

end magnitude_of_vector_combination_l2485_248557


namespace jackson_and_williams_money_l2485_248595

theorem jackson_and_williams_money (jackson_money : ℝ) (williams_money : ℝ) :
  jackson_money = 125 →
  jackson_money = 5 * williams_money →
  jackson_money + williams_money = 145.83 :=
by sorry

end jackson_and_williams_money_l2485_248595


namespace number_of_boats_l2485_248549

/-- Given a lake with boats, where each boat has 3 people and there are 15 people on boats,
    prove that the number of boats is 5. -/
theorem number_of_boats (people_per_boat : ℕ) (total_people : ℕ) (num_boats : ℕ) :
  people_per_boat = 3 →
  total_people = 15 →
  num_boats * people_per_boat = total_people →
  num_boats = 5 := by
sorry

end number_of_boats_l2485_248549


namespace pollen_mass_scientific_notation_l2485_248580

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem pollen_mass_scientific_notation :
  let mass : ℝ := 0.000037
  let scientific := toScientificNotation mass
  scientific.coefficient = 3.7 ∧ scientific.exponent = -5 :=
sorry

end pollen_mass_scientific_notation_l2485_248580


namespace secant_length_l2485_248504

noncomputable section

def Circle (O : ℝ × ℝ) (R : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = R^2}

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def isTangent (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) := 
  ∃! P, P ∈ l ∩ c

def isSecant (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) :=
  ∃ P Q, P ≠ Q ∧ P ∈ l ∩ c ∧ Q ∈ l ∩ c

def isEquidistant (P : ℝ × ℝ) (Q : ℝ × ℝ) (l : Set (ℝ × ℝ)) :=
  ∀ X ∈ l, distance P X = distance Q X

theorem secant_length (O : ℝ × ℝ) (R : ℝ) (A : ℝ × ℝ) 
  (h1 : distance O A = 2 * R)
  (c : Set (ℝ × ℝ)) (h2 : c = Circle O R)
  (t : Set (ℝ × ℝ)) (h3 : isTangent t c)
  (s : Set (ℝ × ℝ)) (h4 : isSecant s c)
  (B : ℝ × ℝ) (h5 : B ∈ t ∩ c)
  (h6 : isEquidistant O B s) :
  ∃ C G : ℝ × ℝ, C ∈ s ∩ c ∧ G ∈ s ∩ c ∧ distance C G = 2 * R * Real.sqrt (10/13) :=
sorry

end secant_length_l2485_248504


namespace triangle_properties_l2485_248511

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle -/
def satisfies_condition (t : Triangle) : Prop :=
  2 * Real.sqrt 2 * (Real.sin t.A ^ 2 - Real.sin t.C ^ 2) = (t.a - t.b) * Real.sin t.B

/-- The circumradius of the triangle is √2 -/
def has_circumradius_sqrt2 (t : Triangle) : Prop :=
  ∃ (R : ℝ), R = Real.sqrt 2

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : has_circumradius_sqrt2 t) : 
  t.C = Real.pi / 3 ∧ 
  ∃ (S : ℝ), S ≤ 3 * Real.sqrt 3 / 2 ∧ 
  (∀ (S' : ℝ), S' = 1/2 * t.a * t.b * Real.sin t.C → S' ≤ S) :=
sorry

end triangle_properties_l2485_248511


namespace maintenance_check_time_l2485_248575

/-- 
Proves that if an additive doubles the time between maintenance checks 
and the new time is 60 days, then the original time was 30 days.
-/
theorem maintenance_check_time (original_time : ℕ) : 
  (2 * original_time = 60) → original_time = 30 := by
  sorry

end maintenance_check_time_l2485_248575


namespace max_value_of_function_l2485_248559

theorem max_value_of_function (x : ℝ) (h : -1 < x ∧ x < 1) : 
  (∀ y : ℝ, -1 < y ∧ y < 1 → x / (x - 1) + x ≥ y / (y - 1) + y) → 
  x / (x - 1) + x = 0 :=
by sorry

end max_value_of_function_l2485_248559


namespace cafe_menu_problem_l2485_248543

theorem cafe_menu_problem (total_dishes : ℕ) 
  (vegan_ratio : ℚ) (gluten_ratio : ℚ) (nut_ratio : ℚ) :
  total_dishes = 30 →
  vegan_ratio = 1 / 3 →
  gluten_ratio = 2 / 5 →
  nut_ratio = 1 / 4 →
  (total_dishes : ℚ) * vegan_ratio * (1 - gluten_ratio - nut_ratio) / total_dishes = 1 / 10 := by
  sorry

end cafe_menu_problem_l2485_248543


namespace line_through_point_with_given_slope_l2485_248569

/-- Given a line L1: 2x + y - 10 = 0 and a point P(1, 0),
    prove that the line L2 passing through P with the same slope as L1
    has the equation 2x + y - 2 = 0 -/
theorem line_through_point_with_given_slope (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 2 * x + y - 10 = 0
  let P : ℝ × ℝ := (1, 0)
  let L2 : ℝ → ℝ → Prop := λ x y => 2 * x + y - 2 = 0
  (∀ x y, L1 x y ↔ 2 * x + y = 10) →
  (L2 (P.1) (P.2)) →
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → (y₂ - y₁) / (x₂ - x₁) = (y - P.2) / (x - P.1)) →
  ∀ x y, L2 x y ↔ 2 * x + y = 2 :=
by sorry

end line_through_point_with_given_slope_l2485_248569


namespace projection_bound_implies_coverage_l2485_248538

/-- A figure in a metric space -/
class Figure (α : Type*) [MetricSpace α]

/-- The projection of a figure onto a line -/
def projection (α : Type*) [MetricSpace α] (Φ : Figure α) (l : Set α) : ℝ := sorry

/-- A figure Φ is covered by a circle of diameter d -/
def covered_by_circle (α : Type*) [MetricSpace α] (Φ : Figure α) (d : ℝ) : Prop := sorry

theorem projection_bound_implies_coverage 
  (α : Type*) [MetricSpace α] (Φ : Figure α) :
  (∀ l : Set α, projection α Φ l ≤ 1) →
  (¬ covered_by_circle α Φ 1) ∧ (covered_by_circle α Φ 1.5) := by sorry

end projection_bound_implies_coverage_l2485_248538


namespace problem_solution_l2485_248550

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem problem_solution :
  (∀ m : ℝ, m = 4 → A ∪ B m = {x | -2 ≤ x ∧ x ≤ 7}) ∧
  (∀ m : ℝ, (B m ∩ A = B m) ↔ m ≤ 3) :=
sorry

end problem_solution_l2485_248550


namespace restricted_arrangements_l2485_248553

/-- The number of ways to arrange n people in a row. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row with one person fixed at the left end. -/
def permutations_with_left_fixed (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to arrange n people in a row with one person fixed at the right end. -/
def permutations_with_right_fixed (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to arrange n people in a row with two people fixed at both ends. -/
def permutations_with_both_ends_fixed (n : ℕ) : ℕ := Nat.factorial (n - 2)

theorem restricted_arrangements (n : ℕ) (h : n = 4) : 
  permutations n - permutations_with_left_fixed n - permutations_with_right_fixed n + permutations_with_both_ends_fixed n = 2 :=
sorry

end restricted_arrangements_l2485_248553


namespace police_chase_distance_l2485_248589

/-- Calculates the distance between a police station and a thief's starting location
    given their speeds and chase duration. -/
def police_station_distance (thief_speed : ℝ) (police_speed : ℝ) 
                             (head_start : ℝ) (chase_duration : ℝ) : ℝ :=
  police_speed * chase_duration - 
  (thief_speed * head_start + thief_speed * chase_duration)

/-- Theorem stating that given specific chase parameters, 
    the police station is 60 km away from the thief's starting point. -/
theorem police_chase_distance : 
  police_station_distance 20 40 1 4 = 60 := by sorry

end police_chase_distance_l2485_248589


namespace add_negative_two_l2485_248507

theorem add_negative_two : 1 + (-2) = -1 := by sorry

end add_negative_two_l2485_248507


namespace age_ratio_problem_l2485_248579

theorem age_ratio_problem (albert mary betty : ℕ) 
  (h1 : ∃ k : ℕ, albert = k * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 22)
  (h4 : betty = 11) :
  albert / mary = 2 := by
  sorry

end age_ratio_problem_l2485_248579


namespace center_distance_of_isosceles_triangle_l2485_248585

/-- An isosceles triangle with two sides of length 6 and one side of length 10 -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  is_isosceles : side1 = side2
  side_lengths : side1 = 6 ∧ base = 10

/-- The distance between the centers of the circumscribed and inscribed circles of the triangle -/
def center_distance (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem stating the distance between the centers of the circumscribed and inscribed circles -/
theorem center_distance_of_isosceles_triangle (t : IsoscelesTriangle) :
  center_distance t = (5 * Real.sqrt 110) / 11 := by sorry

end center_distance_of_isosceles_triangle_l2485_248585


namespace total_wage_proof_l2485_248541

/-- The weekly payment for employee B -/
def wage_B : ℝ := 249.99999999999997

/-- The weekly payment for employee A -/
def wage_A : ℝ := 1.2 * wage_B

/-- The total weekly payment for both employees -/
def total_wage : ℝ := wage_A + wage_B

theorem total_wage_proof : total_wage = 549.9999999999999 := by sorry

end total_wage_proof_l2485_248541


namespace customers_served_today_l2485_248596

theorem customers_served_today (x : ℕ) 
  (h1 : (65 : ℝ) = (65 * x) / x) 
  (h2 : (90 : ℝ) = (65 * x + C) / (x + 1)) 
  (h3 : x = 1) : C = 115 := by
  sorry

end customers_served_today_l2485_248596


namespace cube_decomposition_largest_number_l2485_248574

theorem cube_decomposition_largest_number :
  let n : ℕ := 10
  let sum_of_terms : ℕ → ℕ := λ k => k * (k + 1) / 2
  let total_terms : ℕ := sum_of_terms n - sum_of_terms 1
  2 * total_terms + 1 = 109 :=
by sorry

end cube_decomposition_largest_number_l2485_248574


namespace distance_A_to_C_l2485_248556

/-- Proves that the distance between city A and C is 300 km given the provided conditions -/
theorem distance_A_to_C (
  eddy_time : ℝ)
  (freddy_time : ℝ)
  (distance_A_to_B : ℝ)
  (speed_ratio : ℝ)
  (h1 : eddy_time = 3)
  (h2 : freddy_time = 4)
  (h3 : distance_A_to_B = 510)
  (h4 : speed_ratio = 2.2666666666666666)
  : ℝ := by
  sorry

#check distance_A_to_C

end distance_A_to_C_l2485_248556


namespace always_quadratic_l2485_248502

/-- A quadratic equation is of the form ax^2 + bx + c = 0 where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation (a^2 + 1)x^2 + bx + c = 0 is always a quadratic equation -/
theorem always_quadratic (a b c : ℝ) :
  is_quadratic_equation (fun x => (a^2 + 1) * x^2 + b * x + c) :=
sorry

end always_quadratic_l2485_248502


namespace EL_length_l2485_248591

-- Define the rectangle
def rectangle_EFGH : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define points E and K
def E : ℝ × ℝ := (0, 1)
def K : ℝ × ℝ := (1, 0)

-- Define the inscribed circle ω
def ω : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + (p.2 - 0.5)^2 = 0.25}

-- Define the line EK
def line_EK (x : ℝ) : ℝ := -x + 1

-- Define point L as the intersection of EK and ω (different from K)
def L : ℝ × ℝ :=
  let x := 0.5
  (x, line_EK x)

-- Theorem statement
theorem EL_length :
  let el_length := Real.sqrt ((L.1 - E.1)^2 + (L.2 - E.2)^2)
  el_length = Real.sqrt 2 / 2 :=
sorry

end EL_length_l2485_248591
