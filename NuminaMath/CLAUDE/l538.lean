import Mathlib

namespace NUMINAMATH_CALUDE_odd_positive_poly_one_real_zero_l538_53811

/-- A polynomial with positive real coefficients of odd degree -/
structure OddPositivePoly where
  degree : Nat
  coeffs : Fin degree → ℝ
  odd_degree : Odd degree
  positive_coeffs : ∀ i, coeffs i > 0

/-- A permutation of the coefficients of a polynomial -/
def PermutedPoly (p : OddPositivePoly) :=
  { σ : Equiv (Fin p.degree) (Fin p.degree) // True }

/-- The number of real zeros of a polynomial -/
noncomputable def num_real_zeros (p : OddPositivePoly) (perm : PermutedPoly p) : ℕ :=
  sorry

/-- Theorem: For any odd degree polynomial with positive coefficients,
    there exists a permutation of its coefficients such that
    the resulting polynomial has exactly one real zero -/
theorem odd_positive_poly_one_real_zero (p : OddPositivePoly) :
  ∃ perm : PermutedPoly p, num_real_zeros p perm = 1 :=
sorry

end NUMINAMATH_CALUDE_odd_positive_poly_one_real_zero_l538_53811


namespace NUMINAMATH_CALUDE_odd_function_theorem_l538_53826

/-- A function f: ℝ → ℝ is odd if f(x) = -f(-x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

/-- The main theorem: if f is odd and satisfies the given functional equation,
    then f is the zero function -/
theorem odd_function_theorem (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_eq : ∀ x y, f (x + y) * f (x - y) = f x ^ 2 * f y ^ 2) : 
    ∀ x, f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_odd_function_theorem_l538_53826


namespace NUMINAMATH_CALUDE_range_of_5m_minus_n_l538_53864

-- Define a decreasing and odd function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_decreasing : ∀ x y, x < y → f x > f y)
variable (h_odd : ∀ x, f (-x) = -f x)

-- Define the conditions on m and n
variable (m n : ℝ)
variable (h_cond1 : f m + f (n - 2) ≤ 0)
variable (h_cond2 : f (m - n - 1) ≤ 0)

-- Theorem statement
theorem range_of_5m_minus_n : 5 * m - n ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_range_of_5m_minus_n_l538_53864


namespace NUMINAMATH_CALUDE_baseball_cards_total_l538_53833

def total_baseball_cards (carlos_cards matias_cards jorge_cards : ℕ) : ℕ :=
  carlos_cards + matias_cards + jorge_cards

theorem baseball_cards_total (carlos_cards matias_cards jorge_cards : ℕ) 
  (h1 : carlos_cards = 20)
  (h2 : matias_cards = carlos_cards - 6)
  (h3 : jorge_cards = matias_cards) :
  total_baseball_cards carlos_cards matias_cards jorge_cards = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_total_l538_53833


namespace NUMINAMATH_CALUDE_initial_gum_count_l538_53877

/-- The number of gum pieces Adrianna had initially -/
def initial_gum : ℕ := sorry

/-- The number of gum pieces Adrianna bought -/
def bought_gum : ℕ := 3

/-- The number of friends Adrianna gave gum to -/
def friends : ℕ := 11

/-- The number of gum pieces Adrianna has left -/
def remaining_gum : ℕ := 2

/-- Theorem stating that the initial number of gum pieces was 10 -/
theorem initial_gum_count : initial_gum = 10 := by sorry

end NUMINAMATH_CALUDE_initial_gum_count_l538_53877


namespace NUMINAMATH_CALUDE_train_length_calculation_l538_53894

/-- The length of a train that crosses an electric pole in a given time at a given speed. -/
def trainLength (crossingTime : ℝ) (speed : ℝ) : ℝ :=
  crossingTime * speed

/-- Theorem stating that a train crossing an electric pole in 10 seconds at 108 m/s has a length of 1080 meters. -/
theorem train_length_calculation :
  trainLength 10 108 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l538_53894


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l538_53878

-- Define the sets A and B
def A (p : ℝ) : Set ℝ := {x | x^2 + p*x + 12 = 0}
def B (q : ℝ) : Set ℝ := {x | x^2 - 5*x + q = 0}

-- State the theorem
theorem union_of_A_and_B (p q : ℝ) :
  (Set.compl (A p) ∩ B q = {2}) →
  (A p ∩ Set.compl (B q) = {4}) →
  (A p ∪ B q = {2, 3, 6}) := by
  sorry


end NUMINAMATH_CALUDE_union_of_A_and_B_l538_53878


namespace NUMINAMATH_CALUDE_quadratic_sum_of_squares_l538_53863

/-- A quadratic function f(x) = x^2 + ax + b satisfying certain conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  root_condition : ∃ (p q : ℤ), p * q = 2 ∧ p + q = -a
  functional_equation : ∀ (x : ℝ), x ≠ 0 → 
    (x + 1/x)^2 + a * (x + 1/x) + b = (x^2 + a*x + b) + ((1/x)^2 + a*(1/x) + b)

/-- The main theorem stating that a^2 + b^2 = 13 for the given quadratic function -/
theorem quadratic_sum_of_squares (f : QuadraticFunction) : f.a^2 + f.b^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_squares_l538_53863


namespace NUMINAMATH_CALUDE_other_number_proof_l538_53838

theorem other_number_proof (A B : ℕ+) (hcf lcm : ℕ+) : 
  hcf = 12 →
  lcm = 396 →
  A = 48 →
  Nat.gcd A.val B.val = hcf.val →
  Nat.lcm A.val B.val = lcm.val →
  B = 99 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l538_53838


namespace NUMINAMATH_CALUDE_semicircle_area_ratio_l538_53822

/-- Proves that for a rectangle with sides 8 meters and 12 meters, with semicircles
    drawn on each side (diameters coinciding with the sides), the ratio of the area
    of the large semicircles to the area of the small semicircles is 2.25. -/
theorem semicircle_area_ratio (π : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ)
  (h1 : rectangle_width = 8)
  (h2 : rectangle_length = 12)
  (h3 : π > 0) :
  (2 * π * (rectangle_length / 2)^2 / 2) / (2 * π * (rectangle_width / 2)^2 / 2) = 2.25 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_area_ratio_l538_53822


namespace NUMINAMATH_CALUDE_other_diagonal_length_l538_53835

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ  -- Length of the first diagonal
  d2 : ℝ  -- Length of the second diagonal
  area : ℝ -- Area of the rhombus

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.d1 * r.d2) / 2

/-- Given a rhombus with area 330 and one diagonal 30, the other diagonal is 22 -/
theorem other_diagonal_length :
  ∀ (r : Rhombus), r.area = 330 ∧ r.d1 = 30 → r.d2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l538_53835


namespace NUMINAMATH_CALUDE_insulation_minimum_cost_l538_53889

/-- Represents the total cost function over 20 years for insulation thickness x (in cm) -/
def f (x : ℝ) : ℝ := 800 - 74 * x

/-- The domain of the function f -/
def domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 10

theorem insulation_minimum_cost :
  ∃ (x : ℝ), domain x ∧ f x = 700 ∧ ∀ (y : ℝ), domain y → f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_insulation_minimum_cost_l538_53889


namespace NUMINAMATH_CALUDE_workers_wage_problem_l538_53842

/-- Worker's wage problem -/
theorem workers_wage_problem (total_days : ℕ) (overall_avg : ℝ) 
  (first_5_avg : ℝ) (second_5_avg : ℝ) (third_5_increase : ℝ) (last_5_decrease : ℝ) :
  total_days = 20 →
  overall_avg = 100 →
  first_5_avg = 90 →
  second_5_avg = 110 →
  third_5_increase = 0.05 →
  last_5_decrease = 0.10 →
  ∃ (eleventh_day_wage : ℝ),
    eleventh_day_wage = second_5_avg * (1 + third_5_increase) ∧
    eleventh_day_wage = 115.50 :=
by sorry

end NUMINAMATH_CALUDE_workers_wage_problem_l538_53842


namespace NUMINAMATH_CALUDE_monotonically_decreasing_interval_of_f_shifted_l538_53820

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the derivative of f
axiom f_derivative (x : ℝ) : deriv f x = 2 * x - 4

-- Theorem statement
theorem monotonically_decreasing_interval_of_f_shifted :
  ∀ x : ℝ, (∀ y : ℝ, y < 3 → deriv (fun z ↦ f (z - 1)) y < 0) ∧
           (∀ y : ℝ, y ≥ 3 → deriv (fun z ↦ f (z - 1)) y ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_interval_of_f_shifted_l538_53820


namespace NUMINAMATH_CALUDE_trajectory_equation_l538_53888

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- The point P -/
def P : ℝ × ℝ := (2, 2)

/-- A point is on the trajectory if it's the midpoint of a line segment AB,
    where A and B are intersection points of a line through P and the ellipse -/
def on_trajectory (M : ℝ × ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    (∃ (t : ℝ), A = P + t • (B - P)) ∧
    M = (A + B) / 2

/-- The theorem stating the equation of the trajectory -/
theorem trajectory_equation (x y : ℝ) :
  on_trajectory (x, y) → (x - 1)^2 + 2*(y - 1)^2 = 3 :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l538_53888


namespace NUMINAMATH_CALUDE_election_winner_percentage_l538_53830

theorem election_winner_percentage (total_votes : ℕ) (majority : ℕ) : 
  total_votes = 460 → majority = 184 → 
  (70 : ℚ) = (100 * (total_votes / 2 + majority) : ℚ) / total_votes := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l538_53830


namespace NUMINAMATH_CALUDE_collinear_vectors_l538_53828

/-- Given vectors a and b in ℝ², if ma + nb is collinear with a - 2b, then m/n = -1/2 -/
theorem collinear_vectors (a b : ℝ × ℝ) (m n : ℝ) 
  (h1 : a = (2, 3))
  (h2 : b = (-1, 2))
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ (m • a + n • b) = k • (a - 2 • b)) :
  m / n = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l538_53828


namespace NUMINAMATH_CALUDE_heart_then_ten_probability_l538_53897

/-- The number of cards in a double standard deck -/
def deck_size : ℕ := 104

/-- The number of hearts in a double standard deck -/
def num_hearts : ℕ := 26

/-- The number of 10s in a double standard deck -/
def num_tens : ℕ := 8

/-- The number of 10 of hearts in a double standard deck -/
def num_ten_hearts : ℕ := 2

/-- The probability of drawing a heart as the first card and a 10 as the second card -/
def prob_heart_then_ten : ℚ := 47 / 2678

theorem heart_then_ten_probability :
  prob_heart_then_ten = 
    (num_hearts - num_ten_hearts) / deck_size * num_tens / (deck_size - 1) +
    num_ten_hearts / deck_size * (num_tens - num_ten_hearts) / (deck_size - 1) :=
by sorry

end NUMINAMATH_CALUDE_heart_then_ten_probability_l538_53897


namespace NUMINAMATH_CALUDE_top_square_after_folds_l538_53896

/-- Represents a position on the 5x5 grid -/
structure Position :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents the state of the grid after folding -/
structure FoldedGrid :=
  (top_square : ℕ)
  (visible_squares : List ℕ)

/-- Initial numbering of the grid in row-major order -/
def initial_grid : Position → ℕ
  | ⟨r, c⟩ => r.val * 5 + c.val + 1

/-- Fold along the diagonal from bottom left to top right -/
def fold_diagonal (grid : Position → ℕ) : FoldedGrid :=
  sorry

/-- Fold the left half over the right half -/
def fold_left_to_right (fg : FoldedGrid) : FoldedGrid :=
  sorry

/-- Fold the top half over the bottom half -/
def fold_top_to_bottom (fg : FoldedGrid) : FoldedGrid :=
  sorry

/-- Fold the bottom half over the top half -/
def fold_bottom_to_top (fg : FoldedGrid) : FoldedGrid :=
  sorry

/-- Apply all folding steps -/
def apply_all_folds (grid : Position → ℕ) : FoldedGrid :=
  fold_bottom_to_top (fold_top_to_bottom (fold_left_to_right (fold_diagonal grid)))

theorem top_square_after_folds :
  (apply_all_folds initial_grid).top_square = 13 := by
  sorry

end NUMINAMATH_CALUDE_top_square_after_folds_l538_53896


namespace NUMINAMATH_CALUDE_routes_count_l538_53819

/-- The number of routes from A to B with 6 horizontal and 6 vertical moves -/
def num_routes : ℕ := 924

/-- The total number of moves -/
def total_moves : ℕ := 12

/-- The number of horizontal moves -/
def horizontal_moves : ℕ := 6

/-- The number of vertical moves -/
def vertical_moves : ℕ := 6

theorem routes_count : 
  num_routes = Nat.choose total_moves horizontal_moves :=
by sorry

end NUMINAMATH_CALUDE_routes_count_l538_53819


namespace NUMINAMATH_CALUDE_functional_equation_solution_l538_53808

-- Define the function type
def FunctionType (k : ℝ) := {f : ℝ → ℝ // ∀ x, x ∈ Set.Icc (-k) k → f x ∈ Set.Icc 0 k}

-- State the theorem
theorem functional_equation_solution (k : ℝ) (h_k : k > 0) :
  ∀ f : FunctionType k,
    (∀ x y, x ∈ Set.Icc (-k) k → y ∈ Set.Icc (-k) k → x + y ∈ Set.Icc (-k) k →
      (f.val x)^2 + (f.val y)^2 - 2*x*y = k^2 + (f.val (x + y))^2) →
    ∃ a c : ℝ, ∀ x ∈ Set.Icc (-k) k,
      f.val x = Real.sqrt (a * x + c - x^2) ∧
      0 ≤ a * x + c - x^2 ∧
      a * x + c - x^2 ≤ k^2 :=
by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l538_53808


namespace NUMINAMATH_CALUDE_gcd_problem_l538_53824

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 570 * k) :
  Int.gcd (5 * b^3 + 2 * b^2 + 6 * b + 95) b = 95 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l538_53824


namespace NUMINAMATH_CALUDE_power_function_not_through_origin_l538_53870

theorem power_function_not_through_origin (m : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (m^2 - 3*m + 3) * x^(m^2 - m - 2) ≠ 0) →
  (m = 1 ∨ m = 2) := by
  sorry

end NUMINAMATH_CALUDE_power_function_not_through_origin_l538_53870


namespace NUMINAMATH_CALUDE_factorization_equality_l538_53881

theorem factorization_equality (m n : ℝ) : m^2 * n - n = n * (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l538_53881


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l538_53834

theorem smallest_prime_divisor_of_sum (p : Nat) :
  Prime p ∧ p ∣ (3^15 + 11^9) ∧ ∀ q < p, Prime q → ¬(q ∣ (3^15 + 11^9)) →
  p = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l538_53834


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l538_53857

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 12) :
  (1 / x + 1 / y) ≥ 1 / 3 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 12 ∧ 1 / x + 1 / y = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l538_53857


namespace NUMINAMATH_CALUDE_max_xy_min_reciprocal_sum_min_squared_sum_max_sqrt_sum_l538_53843

variable (x y : ℝ)

-- Define the condition
def condition (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ 2 * x + y = 1

-- Theorems to prove
theorem max_xy (h : condition x y) : 
  ∃ (m : ℝ), m = 1/8 ∧ ∀ (a b : ℝ), condition a b → a * b ≤ m :=
sorry

theorem min_reciprocal_sum (h : condition x y) :
  ∃ (m : ℝ), m = 9 ∧ ∀ (a b : ℝ), condition a b → m ≤ 2/a + 1/b :=
sorry

theorem min_squared_sum (h : condition x y) :
  ∃ (m : ℝ), m = 1/2 ∧ ∀ (a b : ℝ), condition a b → m ≤ 4*a^2 + b^2 :=
sorry

theorem max_sqrt_sum (h : condition x y) :
  ∃ (m : ℝ), m < 2 ∧ ∀ (a b : ℝ), condition a b → Real.sqrt (2*a) + Real.sqrt b ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_xy_min_reciprocal_sum_min_squared_sum_max_sqrt_sum_l538_53843


namespace NUMINAMATH_CALUDE_right_triangle_incircle_area_ratio_l538_53882

theorem right_triangle_incircle_area_ratio 
  (h r : ℝ) 
  (h_pos : h > 0) 
  (r_pos : r > 0) : 
  ∃ (x y : ℝ), 
    x > 0 ∧ y > 0 ∧ 
    x^2 + y^2 = h^2 ∧ 
    (π * r^2) / ((1/2) * x * y) = π * r / (h + r) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_incircle_area_ratio_l538_53882


namespace NUMINAMATH_CALUDE_emma_reaches_jack_emma_reaches_jack_proof_l538_53840

/-- The time it takes for Emma to reach Jack given their initial conditions -/
theorem emma_reaches_jack : ℝ :=
  let initial_distance : ℝ := 30
  let combined_speed : ℝ := 2
  let jack_emma_speed_ratio : ℝ := 2
  let jack_stop_time : ℝ := 6
  
  33

theorem emma_reaches_jack_proof (initial_distance : ℝ) (combined_speed : ℝ) 
  (jack_emma_speed_ratio : ℝ) (jack_stop_time : ℝ) 
  (h1 : initial_distance = 30)
  (h2 : combined_speed = 2)
  (h3 : jack_emma_speed_ratio = 2)
  (h4 : jack_stop_time = 6) :
  emma_reaches_jack = 33 := by
  sorry

#check emma_reaches_jack_proof

end NUMINAMATH_CALUDE_emma_reaches_jack_emma_reaches_jack_proof_l538_53840


namespace NUMINAMATH_CALUDE_condo_units_per_floor_l538_53867

/-- The number of units on each regular floor in a condo development -/
def units_per_regular_floor (total_floors : ℕ) (penthouse_floors : ℕ) (units_per_penthouse : ℕ) (total_units : ℕ) : ℕ :=
  (total_units - penthouse_floors * units_per_penthouse) / (total_floors - penthouse_floors)

/-- Theorem stating that the number of units on each regular floor is 12 -/
theorem condo_units_per_floor :
  units_per_regular_floor 23 2 2 256 = 12 := by
  sorry

end NUMINAMATH_CALUDE_condo_units_per_floor_l538_53867


namespace NUMINAMATH_CALUDE_initial_red_marbles_l538_53829

theorem initial_red_marbles (initial_blue : ℕ) (removed_red : ℕ) (total_remaining : ℕ) : 
  initial_blue = 30 →
  removed_red = 3 →
  total_remaining = 35 →
  ∃ initial_red : ℕ, 
    initial_red = 20 ∧ 
    total_remaining = (initial_red - removed_red) + (initial_blue - 4 * removed_red) :=
by sorry

end NUMINAMATH_CALUDE_initial_red_marbles_l538_53829


namespace NUMINAMATH_CALUDE_third_plane_passenger_count_l538_53846

/-- The number of passengers on the third plane -/
def third_plane_passengers : ℕ := 40

/-- The speed of an empty plane in MPH -/
def empty_plane_speed : ℕ := 600

/-- The speed reduction per passenger in MPH -/
def speed_reduction_per_passenger : ℕ := 2

/-- The number of passengers on the first plane -/
def first_plane_passengers : ℕ := 50

/-- The number of passengers on the second plane -/
def second_plane_passengers : ℕ := 60

/-- The average speed of the three planes in MPH -/
def average_speed : ℕ := 500

theorem third_plane_passenger_count :
  (empty_plane_speed - speed_reduction_per_passenger * first_plane_passengers +
   empty_plane_speed - speed_reduction_per_passenger * second_plane_passengers +
   empty_plane_speed - speed_reduction_per_passenger * third_plane_passengers) / 3 = average_speed :=
by sorry

end NUMINAMATH_CALUDE_third_plane_passenger_count_l538_53846


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l538_53844

theorem absolute_value_inequality (x : ℝ) (h : x ≠ 1) :
  |((2 * x - 1) / (x - 1))| > 3 ↔ (4/5 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l538_53844


namespace NUMINAMATH_CALUDE_yahs_to_bahs_conversion_l538_53852

/-- Given the conversion rates between bahs, rahs, and yahs, 
    prove that 1500 yahs are equivalent to 500 bahs. -/
theorem yahs_to_bahs_conversion 
  (bah_to_rah : (20 : ℚ) / 36 = 1 / (36 / 20)) 
  (rah_to_yah : (12 : ℚ) / 20 = 1 / (20 / 12)) : 
  (1500 : ℚ) * (12 / 20) * (20 / 36) = 500 :=
sorry

end NUMINAMATH_CALUDE_yahs_to_bahs_conversion_l538_53852


namespace NUMINAMATH_CALUDE_parabola_properties_l538_53855

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the parabola D
def parabolaD (x y : ℝ) : Prop := y^2 = 4*x

-- Define point P
def P : ℝ × ℝ := (4, 0)

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 4)

-- Define the midpoint of PQ
def midpoint_PQ (Q : ℝ × ℝ) : Prop := (0, 0) = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the angle equality
def angle_equality (A B Q : ℝ × ℝ) : Prop :=
  (A.2 - Q.2) / (A.1 - Q.1) = -(B.2 - Q.2) / (B.1 - Q.1)

-- Define the line m
def line_m (x : ℝ) : Prop := x = 3

-- Theorem statement
theorem parabola_properties :
  ∀ (A B Q : ℝ × ℝ) (k : ℝ),
  parabolaD A.1 A.2 ∧ parabolaD B.1 B.2 ∧
  line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧
  midpoint_PQ Q →
  (∀ (x y : ℝ), parabolaD x y ↔ y^2 = 4*x) ∧
  angle_equality A B Q ∧
  (∃ (x : ℝ), line_m x ∧
    ∀ (A : ℝ × ℝ), parabolaD A.1 A.2 →
    ∃ (c : ℝ), ∀ (y : ℝ), 
      (x - (A.1 + 4) / 2)^2 + (y - A.2 / 2)^2 = ((A.1 - 4)^2 + A.2^2) / 4 →
      (x - 3)^2 + y^2 = c) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l538_53855


namespace NUMINAMATH_CALUDE_vector_linear_combination_l538_53879

/-- Given two vectors in ℝ², prove that their linear combination results in the expected vector. -/
theorem vector_linear_combination (a b : ℝ × ℝ) (h1 : a = (-1, 0)) (h2 : b = (0, 2)) :
  (2 : ℝ) • a - (3 : ℝ) • b = (-2, -6) := by sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l538_53879


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l538_53862

theorem interest_rate_calculation (P R : ℝ) 
  (h1 : P * (1 + 4 * R / 100) = 400) 
  (h2 : P * (1 + 6 * R / 100) = 500) : 
  R = 25 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l538_53862


namespace NUMINAMATH_CALUDE_handshakes_in_exhibition_l538_53849

/-- Represents a mixed-doubles tennis exhibition -/
structure MixedDoublesExhibition where
  num_teams : Nat
  players_per_team : Nat

/-- Calculates the total number of handshakes in a mixed-doubles tennis exhibition -/
def total_handshakes (exhibition : MixedDoublesExhibition) : Nat :=
  let total_players := exhibition.num_teams * exhibition.players_per_team
  let handshakes_per_player := total_players - 2  -- Exclude self and partner
  (total_players * handshakes_per_player) / 2

/-- Theorem stating that the total number of handshakes in the given exhibition is 24 -/
theorem handshakes_in_exhibition :
  ∃ (exhibition : MixedDoublesExhibition),
    exhibition.num_teams = 4 ∧
    exhibition.players_per_team = 2 ∧
    total_handshakes exhibition = 24 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_in_exhibition_l538_53849


namespace NUMINAMATH_CALUDE_ellipse_foci_l538_53841

/-- An ellipse defined by parametric equations -/
structure ParametricEllipse where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The foci of an ellipse -/
structure EllipseFoci where
  x : ℝ
  y : ℝ

/-- Theorem: The foci of the ellipse defined by x = 3cos(θ) and y = 5sin(θ) are (0, ±4) -/
theorem ellipse_foci (e : ParametricEllipse) 
    (hx : e.x = fun θ => 3 * Real.cos θ)
    (hy : e.y = fun θ => 5 * Real.sin θ) :
  ∃ (f₁ f₂ : EllipseFoci), f₁.x = 0 ∧ f₁.y = 4 ∧ f₂.x = 0 ∧ f₂.y = -4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l538_53841


namespace NUMINAMATH_CALUDE_appropriate_methods_l538_53861

/-- Represents different income levels of families -/
inductive IncomeLevel
  | High
  | Medium
  | Low

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Structure representing a survey -/
structure Survey where
  population : Nat
  sampleSize : Nat
  categories : Option (List (IncomeLevel × Nat))

/-- Function to determine the most appropriate sampling method for a given survey -/
def mostAppropriateMethod (s : Survey) : SamplingMethod :=
  if s.categories.isSome && s.population > 100 && s.sampleSize > 50
  then SamplingMethod.Stratified
  else SamplingMethod.SimpleRandom

/-- The two surveys from the problem -/
def survey1 : Survey :=
  { population := 420,
    sampleSize := 100,
    categories := some [(IncomeLevel.High, 125), (IncomeLevel.Medium, 200), (IncomeLevel.Low, 95)] }

def survey2 : Survey :=
  { population := 5,
    sampleSize := 3,
    categories := none }

/-- Theorem stating that the most appropriate methods for the given surveys are as expected -/
theorem appropriate_methods :
  (mostAppropriateMethod survey1 = SamplingMethod.Stratified) ∧
  (mostAppropriateMethod survey2 = SamplingMethod.SimpleRandom) := by
  sorry

end NUMINAMATH_CALUDE_appropriate_methods_l538_53861


namespace NUMINAMATH_CALUDE_kohens_apples_l538_53818

/-- Kohen's Apple Business Theorem -/
theorem kohens_apples (boxes : ℕ) (apples_per_box : ℕ) (sold_fraction : ℚ) 
  (h1 : boxes = 10)
  (h2 : apples_per_box = 300)
  (h3 : sold_fraction = 3/4) : 
  boxes * apples_per_box - (sold_fraction * (boxes * apples_per_box)).num = 750 := by
  sorry

end NUMINAMATH_CALUDE_kohens_apples_l538_53818


namespace NUMINAMATH_CALUDE_stock_price_change_l538_53815

def total_stocks : ℕ := 8000

theorem stock_price_change (higher lower : ℕ) 
  (h1 : higher + lower = total_stocks)
  (h2 : higher = lower + lower / 2) :
  higher = 4800 := by
sorry

end NUMINAMATH_CALUDE_stock_price_change_l538_53815


namespace NUMINAMATH_CALUDE_coefficient_d_nonzero_l538_53823

-- Define the polynomial Q(x)
def Q (a b c d e f : ℝ) (x : ℝ) : ℝ :=
  x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f

-- Define the property of having six distinct x-intercepts
def has_six_distinct_intercepts (a b c d e f : ℝ) : Prop :=
  ∃ p q r s t : ℝ, 
    (p ≠ q) ∧ (p ≠ r) ∧ (p ≠ s) ∧ (p ≠ t) ∧ (p ≠ 0) ∧
    (q ≠ r) ∧ (q ≠ s) ∧ (q ≠ t) ∧ (q ≠ 0) ∧
    (r ≠ s) ∧ (r ≠ t) ∧ (r ≠ 0) ∧
    (s ≠ t) ∧ (s ≠ 0) ∧
    (t ≠ 0) ∧
    (Q a b c d e f p = 0) ∧ (Q a b c d e f q = 0) ∧ 
    (Q a b c d e f r = 0) ∧ (Q a b c d e f s = 0) ∧ 
    (Q a b c d e f t = 0) ∧ (Q a b c d e f 0 = 0)

-- Theorem statement
theorem coefficient_d_nonzero 
  (a b c d e f : ℝ) 
  (h : has_six_distinct_intercepts a b c d e f) : 
  d ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_d_nonzero_l538_53823


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l538_53850

theorem factorial_fraction_equality : (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 48 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l538_53850


namespace NUMINAMATH_CALUDE_smallest_reciprocal_l538_53825

theorem smallest_reciprocal (a b c d e : ℚ) : 
  a = 1/4 → b = 3/7 → c = -2 → d = 10 → e = 2023 →
  (1/c < 1/a ∧ 1/c < 1/b ∧ 1/c < 1/d ∧ 1/c < 1/e) := by
  sorry

end NUMINAMATH_CALUDE_smallest_reciprocal_l538_53825


namespace NUMINAMATH_CALUDE_max_grandchildren_problem_l538_53868

/-- The number of children Max has -/
def max_children : ℕ := 8

/-- The number of Max's children who have the same number of children as Max -/
def children_with_same : ℕ := 6

/-- The total number of Max's grandchildren -/
def total_grandchildren : ℕ := 58

/-- The number of children each exception has -/
def exception_children : ℕ := 5

theorem max_grandchildren_problem :
  (children_with_same * max_children) + 
  (2 * exception_children) = total_grandchildren :=
by sorry

end NUMINAMATH_CALUDE_max_grandchildren_problem_l538_53868


namespace NUMINAMATH_CALUDE_sports_event_distribution_l538_53875

/-- Represents the number of medals remaining after k days -/
def remaining_medals (k : ℕ) (m : ℕ) : ℚ :=
  if k = 0 then m
  else (6/7) * remaining_medals (k-1) m - (6/7) * k

/-- The sports event distribution problem -/
theorem sports_event_distribution (n m : ℕ) : 
  (∀ k, 1 ≤ k ∧ k < n → 
    remaining_medals k m = remaining_medals (k-1) m - (k + (1/7) * (remaining_medals (k-1) m - k))) ∧
  remaining_medals (n-1) m = n ∧
  remaining_medals n m = 0 →
  n = 6 ∧ m = 36 := by
sorry

end NUMINAMATH_CALUDE_sports_event_distribution_l538_53875


namespace NUMINAMATH_CALUDE_school_event_handshakes_l538_53872

/-- Represents the number of handshakes in a group of children -/
def handshakes (n : ℕ) : ℕ := 
  (n * (n - 1)) / 2

/-- The problem statement -/
theorem school_event_handshakes : 
  handshakes 8 = 36 := by sorry

end NUMINAMATH_CALUDE_school_event_handshakes_l538_53872


namespace NUMINAMATH_CALUDE_justice_palms_l538_53895

/-- The number of palms Justice has -/
def num_palms : ℕ := sorry

/-- The total number of plants Justice wants -/
def total_plants : ℕ := 24

/-- The number of ferns Justice has -/
def num_ferns : ℕ := 3

/-- The number of succulent plants Justice has -/
def num_succulents : ℕ := 7

/-- The number of additional plants Justice needs -/
def additional_plants : ℕ := 9

theorem justice_palms : num_palms = 5 := by sorry

end NUMINAMATH_CALUDE_justice_palms_l538_53895


namespace NUMINAMATH_CALUDE_angela_problems_count_l538_53883

def total_problems : ℕ := 20
def martha_problems : ℕ := 2
def jenna_problems : ℕ := 4 * martha_problems - 2
def mark_problems : ℕ := jenna_problems / 2

theorem angela_problems_count : 
  total_problems - (martha_problems + jenna_problems + mark_problems) = 9 := by
  sorry

end NUMINAMATH_CALUDE_angela_problems_count_l538_53883


namespace NUMINAMATH_CALUDE_water_bucket_problem_l538_53884

theorem water_bucket_problem (a b : ℝ) : 
  (a - 6 = (1/3) * (b + 6)) →
  (b - 6 = (1/2) * (a + 6)) →
  a = 13.2 := by
  sorry

end NUMINAMATH_CALUDE_water_bucket_problem_l538_53884


namespace NUMINAMATH_CALUDE_infinitely_many_double_numbers_plus_one_square_not_power_of_ten_l538_53899

theorem infinitely_many_double_numbers_plus_one_square_not_power_of_ten :
  ∀ m : ℕ, ∃ k > m, ∃ N : ℕ,
    Odd k ∧
    ∃ t : ℕ, N * (10^k + 1) + 1 = t^2 ∧
    ¬∃ n : ℕ, N * (10^k + 1) + 1 = 10^n := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_double_numbers_plus_one_square_not_power_of_ten_l538_53899


namespace NUMINAMATH_CALUDE_minimum_weights_l538_53890

def is_valid_weight_set (weights : List ℕ) : Prop :=
  ∀ w : ℕ, 1 ≤ w ∧ w ≤ 20 →
    ∃ (a b : ℕ), a ∈ weights ∧ b ∈ weights ∧ (w = a ∨ w = a + b)

theorem minimum_weights :
  ∃ (weights : List ℕ),
    weights.length = 6 ∧
    is_valid_weight_set weights ∧
    ∀ (other_weights : List ℕ),
      is_valid_weight_set other_weights →
      other_weights.length ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_minimum_weights_l538_53890


namespace NUMINAMATH_CALUDE_opposite_of_23_l538_53837

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_23 : opposite 23 = -23 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_23_l538_53837


namespace NUMINAMATH_CALUDE_trapezoid_area_l538_53885

/-- Represents a rectangle PQRS with a trapezoid TURS inside it -/
structure RectangleWithTrapezoid where
  /-- Length of the rectangle PQRS -/
  length : ℝ
  /-- Width of the rectangle PQRS -/
  width : ℝ
  /-- Distance from P to T (same as distance from Q to U) -/
  side_length : ℝ
  /-- Area of rectangle PQRS is 24 -/
  area_eq : length * width = 24
  /-- T and U are on the top side of PQRS -/
  side_constraint : side_length < length

/-- The area of trapezoid TURS is 16 square units -/
theorem trapezoid_area (rect : RectangleWithTrapezoid) : 
  rect.width * (rect.length - 2 * rect.side_length) + 2 * (rect.side_length * rect.width / 2) = 16 := by
  sorry

#check trapezoid_area

end NUMINAMATH_CALUDE_trapezoid_area_l538_53885


namespace NUMINAMATH_CALUDE_seedlings_per_packet_l538_53809

theorem seedlings_per_packet (total_seedlings : ℕ) (num_packets : ℕ) 
  (h1 : total_seedlings = 420) (h2 : num_packets = 60) :
  total_seedlings / num_packets = 7 := by
  sorry

end NUMINAMATH_CALUDE_seedlings_per_packet_l538_53809


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_x_axis_l538_53848

/-- Given a line with equation 3x - 4y + 5 = 0, its symmetric line with respect to the x-axis has the equation 3x + 4y + 5 = 0 -/
theorem symmetric_line_wrt_x_axis :
  ∀ (x y : ℝ), 3 * x - 4 * y + 5 = 0 →
  ∃ (x' y' : ℝ), x' = x ∧ y' = -y ∧ 3 * x' + 4 * y' + 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_x_axis_l538_53848


namespace NUMINAMATH_CALUDE_fourth_minus_third_tiles_l538_53821

/-- The side length of the n-th square in the sequence -/
def side_length (n : ℕ) : ℕ := n^2

/-- The number of tiles in the n-th square -/
def tiles (n : ℕ) : ℕ := (side_length n)^2

theorem fourth_minus_third_tiles : tiles 4 - tiles 3 = 175 := by
  sorry

end NUMINAMATH_CALUDE_fourth_minus_third_tiles_l538_53821


namespace NUMINAMATH_CALUDE_stone_breadth_proof_l538_53856

/-- Given a hall and stones with specified dimensions, prove the breadth of each stone -/
theorem stone_breadth_proof (hall_length hall_width : ℝ) (stone_length : ℝ) (num_stones : ℕ) 
  (h1 : hall_length = 36)
  (h2 : hall_width = 15)
  (h3 : stone_length = 0.3)
  (h4 : num_stones = 3600) :
  ∃ (stone_breadth : ℝ), 
    stone_breadth = 0.5 ∧ 
    (hall_length * hall_width * 100) = (stone_length * stone_breadth * num_stones) :=
by sorry

end NUMINAMATH_CALUDE_stone_breadth_proof_l538_53856


namespace NUMINAMATH_CALUDE_custom_op_theorem_l538_53812

-- Define the custom operation x
def customOp (M N : Set ℕ) : Set ℕ := {x | x ∈ M ∨ x ∈ N ∧ x ∉ M ∩ N}

-- Define sets M and N
def M : Set ℕ := {0, 2, 4, 6, 8, 10}
def N : Set ℕ := {0, 3, 6, 9, 12, 15}

-- Theorem statement
theorem custom_op_theorem : customOp (customOp M N) M = N := by
  sorry

end NUMINAMATH_CALUDE_custom_op_theorem_l538_53812


namespace NUMINAMATH_CALUDE_cage_cost_l538_53805

/-- The cost of the cage given the payment and change -/
theorem cage_cost (bill : ℝ) (change : ℝ) (h1 : bill = 20) (h2 : change = 0.26) :
  bill - change = 19.74 := by
  sorry

end NUMINAMATH_CALUDE_cage_cost_l538_53805


namespace NUMINAMATH_CALUDE_problem_solution_l538_53831

theorem problem_solution (x y z a b : ℝ) 
  (h1 : (x + y) / 2 = (z + x) / 3)
  (h2 : (x + y) / 2 = (y + z) / 4)
  (h3 : x + y + z = 36 * a)
  (h4 : b = x + y) :
  b = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l538_53831


namespace NUMINAMATH_CALUDE_derivative_of_f_l538_53802

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 / (x + 3)

theorem derivative_of_f (x : ℝ) (h : x ≠ -3) :
  deriv f x = (x^2 + 6*x) / (x + 3)^2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l538_53802


namespace NUMINAMATH_CALUDE_remainder_of_binary_div_four_l538_53813

def binary_number : ℕ := 110110111101

theorem remainder_of_binary_div_four :
  binary_number % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_binary_div_four_l538_53813


namespace NUMINAMATH_CALUDE_root_equation_value_l538_53866

theorem root_equation_value (m : ℝ) : 
  m^2 + m - 1 = 0 → 3*m^2 + 3*m + 2006 = 2009 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l538_53866


namespace NUMINAMATH_CALUDE_georginas_parrot_learning_rate_l538_53880

/-- The number of phrases Georgina's parrot knows now -/
def current_phrases : ℕ := 17

/-- The number of phrases the parrot knew when Georgina bought it -/
def initial_phrases : ℕ := 3

/-- The number of days Georgina has had the parrot -/
def days_owned : ℕ := 49

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of phrases Georgina teaches her parrot per week -/
def phrases_per_week : ℚ :=
  (current_phrases - initial_phrases) / (days_owned / days_per_week)

theorem georginas_parrot_learning_rate :
  phrases_per_week = 2 := by sorry

end NUMINAMATH_CALUDE_georginas_parrot_learning_rate_l538_53880


namespace NUMINAMATH_CALUDE_abc_inequality_l538_53858

theorem abc_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h_sum : a^2 + b^2 + c^2 + a*b*c = 4) : 
  0 ≤ a*b + b*c + c*a - a*b*c ∧ a*b + b*c + c*a - a*b*c ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l538_53858


namespace NUMINAMATH_CALUDE_sin_translation_left_l538_53869

/-- Translating the graph of y = sin(2x) to the left by π/3 units results in y = sin(2x + 2π/3) -/
theorem sin_translation_left (x : ℝ) : 
  let f (t : ℝ) := Real.sin (2 * t)
  let g (t : ℝ) := f (t + π/3)
  g x = Real.sin (2 * x + 2 * π/3) := by
  sorry

end NUMINAMATH_CALUDE_sin_translation_left_l538_53869


namespace NUMINAMATH_CALUDE_gcd_2146_1813_l538_53801

theorem gcd_2146_1813 : Nat.gcd 2146 1813 = 37 := by sorry

end NUMINAMATH_CALUDE_gcd_2146_1813_l538_53801


namespace NUMINAMATH_CALUDE_special_function_uniqueness_l538_53860

/-- A function satisfying the given properties -/
def special_function (g : ℝ → ℝ) : Prop :=
  g 2 = 2 ∧ ∀ x y : ℝ, g (x * y + g x) = x * g y + g x

/-- The main theorem stating that any function satisfying the special properties
    is equivalent to the function f(x) = 2x -/
theorem special_function_uniqueness (g : ℝ → ℝ) (hg : special_function g) :
  ∀ x : ℝ, g x = 2 * x :=
sorry

end NUMINAMATH_CALUDE_special_function_uniqueness_l538_53860


namespace NUMINAMATH_CALUDE_max_value_theorem_l538_53892

theorem max_value_theorem (a b : ℝ) 
  (h1 : a + b - 2 ≥ 0) 
  (h2 : b - a - 1 ≤ 0) 
  (h3 : a ≤ 1) : 
  (∀ x y : ℝ, x + y - 2 ≥ 0 → y - x - 1 ≤ 0 → x ≤ 1 → 
    (x + 2*y) / (2*x + y) ≤ (a + 2*b) / (2*a + b)) ∧ 
  (a + 2*b) / (2*a + b) = 7/5 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l538_53892


namespace NUMINAMATH_CALUDE_abc_sign_sum_l538_53814

theorem abc_sign_sum (a b c : ℚ) (h : |a*b*c| / (a*b*c) = 1) :
  |a| / a + |b| / b + |c| / c = -1 ∨ |a| / a + |b| / b + |c| / c = 3 := by
  sorry

end NUMINAMATH_CALUDE_abc_sign_sum_l538_53814


namespace NUMINAMATH_CALUDE_sum_of_integers_l538_53873

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x^2 + y^2 = 90) 
  (h2 : x * y = 27) : 
  x + y = 12 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l538_53873


namespace NUMINAMATH_CALUDE_kit_savings_percentage_l538_53854

/-- The price of the camera lens filter kit -/
def kit_price : ℚ := 75.50

/-- The number of filters in the kit -/
def num_filters : ℕ := 5

/-- The price of the first type of filter -/
def filter_price1 : ℚ := 7.35

/-- The number of filters of the first type -/
def num_filters1 : ℕ := 3

/-- The price of the second type of filter -/
def filter_price2 : ℚ := 12.05

/-- The number of filters of the second type (only 2 are used in the kit) -/
def num_filters2 : ℕ := 2

/-- The price of the third type of filter -/
def filter_price3 : ℚ := 12.50

/-- The number of filters of the third type -/
def num_filters3 : ℕ := 1

/-- The total price of filters if purchased individually -/
def total_individual_price : ℚ :=
  filter_price1 * num_filters1 + filter_price2 * num_filters2 + filter_price3 * num_filters3

/-- The amount saved by purchasing the kit -/
def amount_saved : ℚ := total_individual_price - kit_price

/-- The percentage saved by purchasing the kit -/
def percentage_saved : ℚ := (amount_saved / total_individual_price) * 100

theorem kit_savings_percentage :
  percentage_saved = 28.72 := by sorry

end NUMINAMATH_CALUDE_kit_savings_percentage_l538_53854


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l538_53851

def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

theorem min_value_and_inequality :
  (∃ (M : ℝ), (∀ (m : ℝ), (∃ (x₀ : ℝ), f x₀ ≤ m) → M ≤ m) ∧ (∃ (x₀ : ℝ), f x₀ ≤ M) ∧ M = 4) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → 3*a + b = 4 → 3/b + 1/a ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l538_53851


namespace NUMINAMATH_CALUDE_locus_of_circle_center_l538_53806

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the condition for a circle to be tangent to both given circles
def is_tangent_to_both (cx cy r : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    circle1 x1 y1 ∧ circle2 x2 y2 ∧
    (cx - x1)^2 + (cy - y1)^2 = (r + 1)^2 ∧
    (cx - x2)^2 + (cy - y2)^2 = (r + 3)^2

-- State the theorem
theorem locus_of_circle_center :
  ∀ (x y : ℝ), x < 0 →
    (∃ (r : ℝ), is_tangent_to_both x y r) ↔ x^2 - y^2/8 = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_of_circle_center_l538_53806


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_union_l538_53839

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | 4 - x^2 ≤ 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by sorry

-- Theorem for (¬_U A) ∪ (¬_U B)
theorem complement_union :
  (Set.univ \ A) ∪ (Set.univ \ B) = {x : ℝ | x < -2 ∨ x > -1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_union_l538_53839


namespace NUMINAMATH_CALUDE_equivalent_representations_l538_53810

theorem equivalent_representations : 
  ∀ (a b c d e f : ℚ),
  (a = 9/18) → 
  (b = 1/2) → 
  (c = 27/54) → 
  (d = 1/2) → 
  (e = 1/2) → 
  (f = 1/2) → 
  (a = b ∧ b = c ∧ c = d ∧ d = e ∧ e = f) := by
  sorry

#check equivalent_representations

end NUMINAMATH_CALUDE_equivalent_representations_l538_53810


namespace NUMINAMATH_CALUDE_equation_solution_l538_53827

theorem equation_solution (x : ℝ) :
  (1 : ℝ) = 1 / (4 * x^2 + 2 * x + 1) →
  x = 0 ∨ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l538_53827


namespace NUMINAMATH_CALUDE_sqrt_difference_squared_l538_53832

theorem sqrt_difference_squared : (Real.sqrt 169 - Real.sqrt 25)^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_squared_l538_53832


namespace NUMINAMATH_CALUDE_largest_divisible_n_l538_53876

theorem largest_divisible_n : ∃ (n : ℕ), n = 15544 ∧ 
  (∀ m : ℕ, m > n → ¬(n + 26 ∣ n^3 + 2006)) ∧
  (n + 26 ∣ n^3 + 2006) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l538_53876


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l538_53893

theorem square_sum_given_product_and_sum (r s : ℝ) 
  (h1 : r * s = 16) 
  (h2 : r + s = 8) : 
  r^2 + s^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l538_53893


namespace NUMINAMATH_CALUDE_arithmetic_operations_l538_53804

theorem arithmetic_operations (a b : ℝ) : 
  (a ≠ 0 → a / a = 1) ∧ 
  (b ≠ 0 → a / b = a * (1 / b)) ∧ 
  (a * 1 = a) ∧ 
  (0 / b = 0) :=
sorry

#check arithmetic_operations

end NUMINAMATH_CALUDE_arithmetic_operations_l538_53804


namespace NUMINAMATH_CALUDE_riverside_total_multiple_of_five_l538_53816

/-- Represents the population of animals and people in Riverside --/
structure Riverside where
  people : ℕ
  horses : ℕ
  sheep : ℕ
  cows : ℕ
  ducks : ℕ

/-- The conditions given in the problem --/
def valid_riverside (r : Riverside) : Prop :=
  r.people = 5 * r.horses ∧
  r.sheep = 6 * r.cows ∧
  r.ducks = 4 * r.people ∧
  r.sheep * 2 = r.ducks

/-- The theorem states that the total population in a valid Riverside setup is always a multiple of 5 --/
theorem riverside_total_multiple_of_five (r : Riverside) (h : valid_riverside r) :
  ∃ k : ℕ, r.people + r.horses + r.sheep + r.cows + r.ducks = 5 * k :=
sorry

end NUMINAMATH_CALUDE_riverside_total_multiple_of_five_l538_53816


namespace NUMINAMATH_CALUDE_trigonometric_identity_l538_53836

theorem trigonometric_identity (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.sin y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l538_53836


namespace NUMINAMATH_CALUDE_gcd_840_1764_gcd_98_63_l538_53847

-- Part 1: GCD of 840 and 1764
theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by sorry

-- Part 2: GCD of 98 and 63
theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by sorry

end NUMINAMATH_CALUDE_gcd_840_1764_gcd_98_63_l538_53847


namespace NUMINAMATH_CALUDE_sock_combinations_l538_53817

theorem sock_combinations (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 2) :
  Nat.choose n k = 36 := by
  sorry

end NUMINAMATH_CALUDE_sock_combinations_l538_53817


namespace NUMINAMATH_CALUDE_village_population_l538_53859

theorem village_population (initial_population : ℕ) 
  (h1 : initial_population = 4599) :
  let died := (initial_population : ℚ) * (1/10)
  let remained_after_death := initial_population - ⌊died⌋
  let left := (remained_after_death : ℚ) * (1/5)
  initial_population - ⌊died⌋ - ⌊left⌋ = 3312 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l538_53859


namespace NUMINAMATH_CALUDE_line_through_points_equation_l538_53803

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  slope : ℝ
  yIntercept : ℝ

-- Define the two given points
def pointA : Point2D := { x := 3, y := 0 }
def pointB : Point2D := { x := -3, y := 0 }

-- Theorem: The line passing through pointA and pointB has the equation y = 0
theorem line_through_points_equation :
  ∃ (l : Line2D), l.slope = 0 ∧ l.yIntercept = 0 ∧
  (l.slope * pointA.x + l.yIntercept = pointA.y) ∧
  (l.slope * pointB.x + l.yIntercept = pointB.y) :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_equation_l538_53803


namespace NUMINAMATH_CALUDE_at_operation_difference_l538_53865

def at_operation (x y : ℤ) : ℤ := x^2 * y - 3 * x

theorem at_operation_difference : at_operation 5 3 - at_operation 3 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_at_operation_difference_l538_53865


namespace NUMINAMATH_CALUDE_basketball_card_price_basketball_card_price_proof_l538_53891

/-- The price of a basketball card pack given the following conditions:
  * Olivia bought 2 packs of basketball cards
  * She bought 5 decks of baseball cards at $4 each
  * She had one $50 bill and received $24 in change
-/
theorem basketball_card_price : ℝ :=
  let baseball_card_price : ℝ := 4
  let baseball_card_count : ℕ := 5
  let basketball_card_count : ℕ := 2
  let total_money : ℝ := 50
  let change : ℝ := 24
  let spent_money : ℝ := total_money - change
  let baseball_total : ℝ := baseball_card_price * baseball_card_count
  3

theorem basketball_card_price_proof :
  let baseball_card_price : ℝ := 4
  let baseball_card_count : ℕ := 5
  let basketball_card_count : ℕ := 2
  let total_money : ℝ := 50
  let change : ℝ := 24
  let spent_money : ℝ := total_money - change
  let baseball_total : ℝ := baseball_card_price * baseball_card_count
  basketball_card_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_basketball_card_price_basketball_card_price_proof_l538_53891


namespace NUMINAMATH_CALUDE_mary_gave_three_green_crayons_l538_53874

/-- The number of green crayons Mary gave to Becky -/
def green_crayons_given : ℕ := 3

/-- The initial number of green crayons Mary had -/
def initial_green : ℕ := 5

/-- The initial number of blue crayons Mary had -/
def initial_blue : ℕ := 8

/-- The number of blue crayons Mary gave away -/
def blue_given : ℕ := 1

/-- The number of crayons Mary has left -/
def crayons_left : ℕ := 9

theorem mary_gave_three_green_crayons :
  green_crayons_given = initial_green - (initial_green + initial_blue - blue_given - crayons_left) :=
by sorry

end NUMINAMATH_CALUDE_mary_gave_three_green_crayons_l538_53874


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l538_53871

/-- 
Given a parabola defined by the equation x = a * y^2 where a ≠ 0,
prove that the coordinates of its focus are (1/(4*a), 0).
-/
theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  let parabola := {p : ℝ × ℝ | p.1 = a * p.2^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (1 / (4 * a), 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l538_53871


namespace NUMINAMATH_CALUDE_ellipse_equation_l538_53800

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  eccentricity : ℝ
  perimeter_triangle : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : eccentricity = Real.sqrt 3 / 3
  h4 : perimeter_triangle = 4 * Real.sqrt 3

/-- The theorem stating that an ellipse with the given properties has the equation x²/3 + y²/2 = 1 -/
theorem ellipse_equation (C : Ellipse) : 
  C.a = Real.sqrt 3 ∧ C.b = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l538_53800


namespace NUMINAMATH_CALUDE_recipe_total_cups_l538_53807

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a recipe ratio and the amount of flour used -/
def totalCups (ratio : RecipeRatio) (flourUsed : ℕ) : ℕ :=
  let partSize := flourUsed / ratio.flour
  partSize * (ratio.butter + ratio.flour + ratio.sugar)

/-- Theorem stating that for the given recipe ratio and flour amount, the total cups is 20 -/
theorem recipe_total_cups :
  let ratio : RecipeRatio := ⟨2, 3, 5⟩
  let flourUsed : ℕ := 6
  totalCups ratio flourUsed = 20 := by
  sorry


end NUMINAMATH_CALUDE_recipe_total_cups_l538_53807


namespace NUMINAMATH_CALUDE_equation_solution_l538_53853

theorem equation_solution : ∃ m : ℚ, (24 / (3 / 2) = (24 / 3) / m) ∧ m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l538_53853


namespace NUMINAMATH_CALUDE_wall_length_approximation_l538_53898

/-- Given a square mirror and a rectangular wall, if the mirror's area is exactly half the wall's area,
    prove that the length of the wall is approximately 43 inches. -/
theorem wall_length_approximation (mirror_side : ℝ) (wall_width : ℝ) (wall_length : ℝ) : 
  mirror_side = 34 →
  wall_width = 54 →
  mirror_side ^ 2 = (wall_width * wall_length) / 2 →
  ∃ ε > 0, |wall_length - 43| < ε :=
by sorry

end NUMINAMATH_CALUDE_wall_length_approximation_l538_53898


namespace NUMINAMATH_CALUDE_p_at_5_l538_53887

/-- A monic quartic polynomial with specific values at x = 1, 2, 3, and 4 -/
def p : ℝ → ℝ :=
  fun x => x^4 + a*x^3 + b*x^2 + c*x + d
  where
    a : ℝ := sorry
    b : ℝ := sorry
    c : ℝ := sorry
    d : ℝ := sorry

/-- The polynomial p satisfies the given conditions -/
axiom p_cond1 : p 1 = 2
axiom p_cond2 : p 2 = 3
axiom p_cond3 : p 3 = 6
axiom p_cond4 : p 4 = 11

/-- The theorem to be proved -/
theorem p_at_5 : p 5 = 48 := by
  sorry

end NUMINAMATH_CALUDE_p_at_5_l538_53887


namespace NUMINAMATH_CALUDE_batsman_average_after_15th_inning_l538_53886

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : ℕ
  totalScore : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (score : ℕ) : ℚ :=
  (stats.totalScore + score : ℚ) / (stats.innings + 1)

/-- Theorem: A batsman's new average after the 15th inning is 33 -/
theorem batsman_average_after_15th_inning
  (stats : BatsmanStats)
  (h1 : stats.innings = 14)
  (h2 : newAverage stats 75 = stats.average + 3)
  : newAverage stats 75 = 33 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_after_15th_inning_l538_53886


namespace NUMINAMATH_CALUDE_water_bottle_theorem_l538_53845

def water_bottle_problem (water_A : ℝ) (extra_B : ℝ) (extra_C : ℝ) : Prop :=
  let water_B : ℝ := water_A + extra_B
  let water_C_ml : ℝ := (water_B / 10) * 1000 + extra_C
  let water_C_L : ℝ := water_C_ml / 1000
  water_C_L = 4.94

theorem water_bottle_theorem :
  water_bottle_problem 3.8 8.4 3720 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_theorem_l538_53845
