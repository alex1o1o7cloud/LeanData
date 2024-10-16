import Mathlib

namespace NUMINAMATH_CALUDE_money_distribution_l4184_418486

/-- Proves that B and C together have Rs. 450 given the conditions of the problem -/
theorem money_distribution (total : ℕ) (ac_sum : ℕ) (c_amount : ℕ) 
  (h1 : total = 600)
  (h2 : ac_sum = 250)
  (h3 : c_amount = 100) : 
  total - (ac_sum - c_amount) + c_amount = 450 := by
  sorry

#check money_distribution

end NUMINAMATH_CALUDE_money_distribution_l4184_418486


namespace NUMINAMATH_CALUDE_adams_clothing_ratio_l4184_418471

theorem adams_clothing_ratio :
  let initial_clothes : ℕ := 36
  let friend_count : ℕ := 3
  let total_donated : ℕ := 126
  let friends_donation := friend_count * initial_clothes
  let adams_kept := initial_clothes - (friends_donation + initial_clothes - total_donated)
  adams_kept = 0 ∧ initial_clothes ≠ 0 →
  (adams_kept : ℚ) / initial_clothes = 0 := by
sorry

end NUMINAMATH_CALUDE_adams_clothing_ratio_l4184_418471


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l4184_418408

theorem solution_set_of_inequality (a : ℝ) (h : a > 1) :
  {x : ℝ | |x| + a > 1} = Set.univ :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l4184_418408


namespace NUMINAMATH_CALUDE_rectangle_breadth_l4184_418485

/-- A rectangle with length three times its breadth and area 675 square meters has a breadth of 15 meters. -/
theorem rectangle_breadth (b : ℝ) (h1 : b > 0) : 
  (3 * b) * b = 675 → b = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_breadth_l4184_418485


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l4184_418445

theorem alcohol_mixture_percentage (x : ℝ) : 
  (8 * 0.25 + 2 * (x / 100)) / (8 + 2) = 0.224 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l4184_418445


namespace NUMINAMATH_CALUDE_value_of_c_l4184_418418

theorem value_of_c : ∃ c : ℝ, 
  (∀ x : ℝ, x * (4 * x + 2) < c ↔ -5/2 < x ∧ x < 3) ∧ c = 45 := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l4184_418418


namespace NUMINAMATH_CALUDE_arithmetic_equation_l4184_418435

theorem arithmetic_equation : 80 + 5 * 12 / (180 / 3) = 81 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l4184_418435


namespace NUMINAMATH_CALUDE_thursday_withdrawal_l4184_418451

/-- Calculates the number of books withdrawn on Thursday given the initial number of books,
    the number of books taken out on Tuesday, the number of books returned on Wednesday,
    and the final number of books in the library. -/
def books_withdrawn_thursday (initial : ℕ) (taken_tuesday : ℕ) (returned_wednesday : ℕ) (final : ℕ) : ℕ :=
  initial - taken_tuesday + returned_wednesday - final

/-- Proves that the number of books withdrawn on Thursday is 15, given the specific values
    from the problem. -/
theorem thursday_withdrawal : books_withdrawn_thursday 250 120 35 150 = 15 := by
  sorry

end NUMINAMATH_CALUDE_thursday_withdrawal_l4184_418451


namespace NUMINAMATH_CALUDE_sin_cos_sum_equality_l4184_418473

theorem sin_cos_sum_equality : Real.sin (315 * π / 180) - Real.cos (135 * π / 180) + 2 * Real.sin (570 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equality_l4184_418473


namespace NUMINAMATH_CALUDE_johns_initial_marbles_l4184_418450

/-- Given that:
    - Ben had 18 marbles initially
    - John had an unknown number of marbles initially
    - Ben gave half of his marbles to John
    - After the transfer, John had 17 more marbles than Ben
    Prove that John had 17 marbles initially -/
theorem johns_initial_marbles :
  ∀ (john_initial : ℕ),
  let ben_initial : ℕ := 18
  let ben_gave : ℕ := ben_initial / 2
  let ben_final : ℕ := ben_initial - ben_gave
  let john_final : ℕ := john_initial + ben_gave
  john_final = ben_final + 17 →
  john_initial = 17 := by
sorry

end NUMINAMATH_CALUDE_johns_initial_marbles_l4184_418450


namespace NUMINAMATH_CALUDE_multiplication_formula_98_102_l4184_418453

theorem multiplication_formula_98_102 : 98 * 102 = 9996 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_formula_98_102_l4184_418453


namespace NUMINAMATH_CALUDE_motorcycle_meeting_distance_l4184_418498

/-- The distance traveled by a constant speed motorcyclist when meeting an accelerating motorcyclist on a circular track -/
theorem motorcycle_meeting_distance (v : ℝ) (a : ℝ) : 
  v > 0 → a > 0 →
  v * (1 / v) = 1 →
  (1/2) * a * (1 / v)^2 = 1 →
  ∃ (T : ℝ), T > 0 ∧ v * T + (1/2) * a * T^2 = 1 →
  v * T = (-1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_motorcycle_meeting_distance_l4184_418498


namespace NUMINAMATH_CALUDE_initial_trees_count_l4184_418409

/-- The number of dogwood trees initially in the park -/
def initial_trees : ℕ := sorry

/-- The number of trees planted today -/
def trees_planted_today : ℕ := 5

/-- The number of trees planted tomorrow -/
def trees_planted_tomorrow : ℕ := 4

/-- The total number of trees after planting -/
def final_trees : ℕ := 16

/-- The number of workers who finished the work -/
def num_workers : ℕ := 8

theorem initial_trees_count : 
  initial_trees = final_trees - (trees_planted_today + trees_planted_tomorrow) :=
by sorry

end NUMINAMATH_CALUDE_initial_trees_count_l4184_418409


namespace NUMINAMATH_CALUDE_a_less_than_one_l4184_418427

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State the conditions
axiom deriv_f : ∀ x, HasDerivAt f (f' x) x
axiom symm_cond : ∀ x, f x + f (-x) = x^2
axiom deriv_gt : ∀ x ≥ 0, f' x > x
axiom ineq_cond : ∀ a, f (2 - a) + 2*a > f a + 2

-- State the theorem
theorem a_less_than_one (a : ℝ) : a < 1 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_one_l4184_418427


namespace NUMINAMATH_CALUDE_square_perimeter_when_area_equals_side_l4184_418425

theorem square_perimeter_when_area_equals_side : ∀ s : ℝ,
  s > 0 → s^2 = s → 4 * s = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_when_area_equals_side_l4184_418425


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l4184_418489

theorem quadratic_integer_roots (n : ℕ+) :
  (∃ x : ℤ, x^2 - 4*x + n = 0) ↔ (n = 3 ∨ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l4184_418489


namespace NUMINAMATH_CALUDE_vectors_parallel_iff_l4184_418496

def a (m : ℝ) : Fin 2 → ℝ := ![1, m + 1]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 2]

theorem vectors_parallel_iff (m : ℝ) :
  (∃ (k : ℝ), a m = k • b m) ↔ m = -2 ∨ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_vectors_parallel_iff_l4184_418496


namespace NUMINAMATH_CALUDE_pizza_sharing_l4184_418469

theorem pizza_sharing (total_slices : ℕ) (buzz_ratio waiter_ratio : ℕ) 
  (h1 : total_slices = 78)
  (h2 : buzz_ratio = 5)
  (h3 : waiter_ratio = 8) :
  waiter_ratio * (total_slices / (buzz_ratio + waiter_ratio)) - 20 = 28 :=
by sorry

end NUMINAMATH_CALUDE_pizza_sharing_l4184_418469


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l4184_418463

/-- The length of the major axis of an ellipse with given foci and tangent to x-axis -/
theorem ellipse_major_axis_length : 
  ∀ (E : Set (ℝ × ℝ)) (F₁ F₂ : ℝ × ℝ),
  F₁ = (2, 10) →
  F₂ = (26, 35) →
  (∃ (X : ℝ), (X, 0) ∈ E) →
  (∀ (P : ℝ × ℝ), P ∈ E ↔ 
    ∃ (k : ℝ), dist P F₁ + dist P F₂ = k ∧ 
    ∀ (Q : ℝ × ℝ), Q ∈ E → dist Q F₁ + dist Q F₂ = k) →
  ∃ (A B : ℝ × ℝ), A ∈ E ∧ B ∈ E ∧ dist A B = 102 ∧
    ∀ (P Q : ℝ × ℝ), P ∈ E → Q ∈ E → dist P Q ≤ 102 :=
by sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l4184_418463


namespace NUMINAMATH_CALUDE_todds_initial_gum_pieces_todds_initial_gum_pieces_proof_l4184_418446

theorem todds_initial_gum_pieces : ℝ → Prop :=
  fun x =>
    let additional_pieces : ℝ := 150
    let percentage_increase : ℝ := 0.25
    let final_total : ℝ := 890
    (x + additional_pieces = final_total) ∧
    (additional_pieces = percentage_increase * x) →
    x = 712

-- The proof is omitted
theorem todds_initial_gum_pieces_proof : todds_initial_gum_pieces 712 := by
  sorry

end NUMINAMATH_CALUDE_todds_initial_gum_pieces_todds_initial_gum_pieces_proof_l4184_418446


namespace NUMINAMATH_CALUDE_task_assignment_count_l4184_418478

/-- The number of ways to assign tasks to volunteers -/
def task_assignments (num_volunteers : ℕ) (num_tasks : ℕ) : ℕ :=
  -- Number of ways to divide tasks into groups
  (num_tasks.choose (num_tasks - num_volunteers)) *
  -- Number of ways to permute volunteers
  (num_volunteers.factorial)

/-- Theorem: There are 36 ways to assign 4 tasks to 3 volunteers -/
theorem task_assignment_count :
  task_assignments 3 4 = 36 := by
sorry

end NUMINAMATH_CALUDE_task_assignment_count_l4184_418478


namespace NUMINAMATH_CALUDE_angle_x_measure_l4184_418421

-- Define the triangle ABD
structure Triangle :=
  (A B D : Point)

-- Define the angles in the triangle
def angle_ABC (t : Triangle) : ℝ := 108
def angle_ABD (t : Triangle) : ℝ := 180 - angle_ABC t
def angle_BAD (t : Triangle) : ℝ := 26

-- Theorem statement
theorem angle_x_measure (t : Triangle) :
  180 - angle_ABD t - angle_BAD t = 82 :=
sorry

end NUMINAMATH_CALUDE_angle_x_measure_l4184_418421


namespace NUMINAMATH_CALUDE_triangle_sum_in_closed_shape_l4184_418430

theorem triangle_sum_in_closed_shape (n : ℕ) (C : ℝ) : 
  n > 0 → C = 3 * 360 - 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_in_closed_shape_l4184_418430


namespace NUMINAMATH_CALUDE_levels_for_110_blocks_l4184_418422

/-- The number of blocks in the nth level of the pattern -/
def blocks_in_level (n : ℕ) : ℕ := 2 + 2 * (n - 1)

/-- The total number of blocks used up to the nth level -/
def total_blocks (n : ℕ) : ℕ := n * (n + 1)

/-- The theorem stating that 10 levels are needed to use exactly 110 blocks -/
theorem levels_for_110_blocks :
  ∃ (n : ℕ), n > 0 ∧ total_blocks n = 110 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m ≠ n → total_blocks m ≠ 110) :=
sorry

end NUMINAMATH_CALUDE_levels_for_110_blocks_l4184_418422


namespace NUMINAMATH_CALUDE_residue_calculation_l4184_418412

theorem residue_calculation : (204 * 15 - 16 * 8 + 5) % 17 = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_calculation_l4184_418412


namespace NUMINAMATH_CALUDE_not_pythagorean_triple_l4184_418441

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem not_pythagorean_triple : 
  (is_pythagorean_triple 3 4 5) ∧ 
  (is_pythagorean_triple 5 12 13) ∧ 
  (is_pythagorean_triple 6 8 10) ∧ 
  ¬(is_pythagorean_triple 7 25 26) := by
  sorry

end NUMINAMATH_CALUDE_not_pythagorean_triple_l4184_418441


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4184_418424

def M : Set ℝ := {x : ℝ | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 > 0}

theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | (-4 ≤ x ∧ x ≤ -2) ∨ (3 < x ∧ x ≤ 7)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4184_418424


namespace NUMINAMATH_CALUDE_concert_tickets_cost_l4184_418420

def total_cost (adult_tickets child_tickets adult_price child_price adult_discount child_discount total_discount : ℚ) : ℚ :=
  let adult_cost := adult_tickets * adult_price * (1 - adult_discount)
  let child_cost := child_tickets * child_price * (1 - child_discount)
  let subtotal := adult_cost + child_cost
  subtotal * (1 - total_discount)

theorem concert_tickets_cost :
  total_cost 12 12 10 5 0.4 0.3 0.1 = 102.6 := by
  sorry

end NUMINAMATH_CALUDE_concert_tickets_cost_l4184_418420


namespace NUMINAMATH_CALUDE_cubic_roots_properties_l4184_418470

theorem cubic_roots_properties (x₁ x₂ x₃ : ℝ) :
  (x₁^3 - 17*x₁ - 18 = 0) →
  (x₂^3 - 17*x₂ - 18 = 0) →
  (x₃^3 - 17*x₃ - 18 = 0) →
  (-4 < x₁) → (x₁ < -3) →
  (4 < x₃) → (x₃ < 5) →
  (⌊x₂⌋ = -2) ∧ (Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = -π/4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_properties_l4184_418470


namespace NUMINAMATH_CALUDE_unique_prime_with_six_divisors_l4184_418456

/-- A function that counts the number of distinct divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def is_prime (p : ℕ) : Prop := sorry

theorem unique_prime_with_six_divisors :
  ∀ p : ℕ, is_prime p → (count_divisors (p^2 + 11) = 6) → p = 3 := by sorry

end NUMINAMATH_CALUDE_unique_prime_with_six_divisors_l4184_418456


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l4184_418432

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-3, 0]
def interval : Set ℝ := {x | -3 ≤ x ∧ x ≤ 0}

-- Theorem stating the maximum and minimum values of f(x) on the given interval
theorem f_max_min_on_interval :
  ∃ (max min : ℝ), 
    (∀ x ∈ interval, f x ≤ max) ∧ 
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧ 
    (∃ x ∈ interval, f x = min) ∧
    max = 2 ∧ min = -18 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l4184_418432


namespace NUMINAMATH_CALUDE_local_minimum_implies_m_eq_one_l4184_418439

/-- The function f(x) = x(x-m)^2 has a local minimum at x = 1 -/
def has_local_minimum_at_one (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - 1| < δ → f x ≥ f 1

/-- The main theorem: if f(x) = x(x-m)^2 has a local minimum at x = 1, then m = 1 -/
theorem local_minimum_implies_m_eq_one (m : ℝ) :
  has_local_minimum_at_one (fun x => x * (x - m)^2) m → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_local_minimum_implies_m_eq_one_l4184_418439


namespace NUMINAMATH_CALUDE_unique_solution_x_squared_minus_two_factorial_y_l4184_418406

theorem unique_solution_x_squared_minus_two_factorial_y : 
  ∃! (x y : ℕ+), x^2 - 2 * Nat.factorial y.val = 2021 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_x_squared_minus_two_factorial_y_l4184_418406


namespace NUMINAMATH_CALUDE_quadratic_inequality_l4184_418458

theorem quadratic_inequality (x : ℝ) : -3*x^2 + 6*x + 9 > 0 ↔ -1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l4184_418458


namespace NUMINAMATH_CALUDE_ball_difference_l4184_418455

theorem ball_difference (blue red : ℕ) 
  (h1 : red - 152 = (blue + 152) + 346) : 
  red - blue = 650 := by
sorry

end NUMINAMATH_CALUDE_ball_difference_l4184_418455


namespace NUMINAMATH_CALUDE_mother_three_times_daughter_age_l4184_418448

/-- Proves that in 9 years, a mother who is currently 42 years old will be three times as old as her daughter who is currently 8 years old. -/
theorem mother_three_times_daughter_age (mother_age : ℕ) (daughter_age : ℕ) (years : ℕ) : 
  mother_age = 42 → daughter_age = 8 → years = 9 → 
  mother_age + years = 3 * (daughter_age + years) :=
by sorry

end NUMINAMATH_CALUDE_mother_three_times_daughter_age_l4184_418448


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l4184_418480

theorem trigonometric_equation_solution :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 110 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l4184_418480


namespace NUMINAMATH_CALUDE_amoeba_count_after_week_l4184_418497

/-- The number of amoebas after n days, given an initial population of 1 and a tripling rate each day -/
def amoeba_count (n : ℕ) : ℕ := 3^n

/-- Theorem: After 7 days, the number of amoebas is 2187 -/
theorem amoeba_count_after_week : amoeba_count 7 = 2187 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_after_week_l4184_418497


namespace NUMINAMATH_CALUDE_sum_and_double_l4184_418434

theorem sum_and_double : (153 + 39 + 27 + 21) * 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_double_l4184_418434


namespace NUMINAMATH_CALUDE_range_of_a_l4184_418490

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 < 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≤ 0

-- State the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x, ¬(p x) → ¬(q x a)) →
  (∃ x, ¬(p x) ∧ (q x a)) →
  a ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4184_418490


namespace NUMINAMATH_CALUDE_barry_sotter_magic_l4184_418464

theorem barry_sotter_magic (n : ℕ) : (n + 3 : ℚ) / 3 = 50 ↔ n = 147 := by sorry

end NUMINAMATH_CALUDE_barry_sotter_magic_l4184_418464


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l4184_418426

theorem geometric_sequence_second_term
  (a : ℕ+) -- first term
  (r : ℕ+) -- common ratio
  (h1 : a = 6)
  (h2 : a * r^3 = 768) :
  a * r = 24 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l4184_418426


namespace NUMINAMATH_CALUDE_no_solution_when_m_negative_four_point_five_l4184_418407

/-- The vector equation has no solutions when m = -4.5 -/
theorem no_solution_when_m_negative_four_point_five :
  let m : ℝ := -4.5
  let v1 : ℝ × ℝ := (1, 3)
  let v2 : ℝ × ℝ := (2, -3)
  let v3 : ℝ × ℝ := (-1, 4)
  let v4 : ℝ × ℝ := (3, m)
  ¬∃ (t s : ℝ), v1 + t • v2 = v3 + s • v4 :=
by sorry


end NUMINAMATH_CALUDE_no_solution_when_m_negative_four_point_five_l4184_418407


namespace NUMINAMATH_CALUDE_number_line_inequalities_l4184_418415

theorem number_line_inequalities (a b c d : ℝ) 
  (ha_neg : a < 0) (hb_neg : b < 0) (hc_pos : c > 0) (hd_pos : d > 0)
  (hc_bounds : 0 < |c| ∧ |c| < 1)
  (hb_bounds : 1 < |b| ∧ |b| < 2)
  (ha_bounds : 2 < |a| ∧ |a| < 4)
  (hd_bounds : 1 < |d| ∧ |d| < 2) : 
  (|a| < 4) ∧ 
  (|b| < 2) ∧ 
  (|c| < 2) ∧ 
  (|a| > |b|) ∧ 
  (|c| < |d|) ∧ 
  (|a - b| < 4) ∧ 
  (|b - c| < 2) ∧ 
  (|c - a| > 1) := by
sorry

end NUMINAMATH_CALUDE_number_line_inequalities_l4184_418415


namespace NUMINAMATH_CALUDE_circle_configuration_theorem_l4184_418452

/-- A configuration of circles as described in the problem -/
structure CircleConfiguration where
  R : ℝ  -- Radius of the semicircle
  r : ℝ  -- Radius of circle O
  r₁ : ℝ  -- Radius of circle O₁
  r₂ : ℝ  -- Radius of circle O₂
  h_positive_R : 0 < R
  h_positive_r : 0 < r
  h_positive_r₁ : 0 < r₁
  h_positive_r₂ : 0 < r₂
  h_tangent_O : r < R  -- O is tangent to the semicircle and its diameter
  h_tangent_O₁ : r₁ < R  -- O₁ is tangent to the semicircle and its diameter
  h_tangent_O₂ : r₂ < R  -- O₂ is tangent to the semicircle and its diameter
  h_tangent_O₁_O : r + r₁ < R  -- O₁ is tangent to O
  h_tangent_O₂_O : r + r₂ < R  -- O₂ is tangent to O

/-- The main theorem to be proved -/
theorem circle_configuration_theorem (c : CircleConfiguration) :
  1 / Real.sqrt c.r₁ + 1 / Real.sqrt c.r₂ = 2 * Real.sqrt 2 / Real.sqrt c.r :=
sorry

end NUMINAMATH_CALUDE_circle_configuration_theorem_l4184_418452


namespace NUMINAMATH_CALUDE_distance_bound_l4184_418410

/-- Given two points A and B, and their distances to a third point (school),
    prove that the distance between A and B is bounded. -/
theorem distance_bound (dist_A_school dist_B_school d : ℝ) : 
  dist_A_school = 5 →
  dist_B_school = 2 →
  3 ≤ d ∧ d ≤ 7 :=
by
  sorry

#check distance_bound

end NUMINAMATH_CALUDE_distance_bound_l4184_418410


namespace NUMINAMATH_CALUDE_max_socks_is_eighteen_l4184_418488

/-- Represents the amount of yarn needed for different items -/
structure YarnAmount where
  sock : ℕ
  hat : ℕ
  sweater : ℕ

/-- Represents the two balls of yarn -/
structure YarnBalls where
  large : YarnAmount
  small : YarnAmount

/-- The given conditions for the yarn balls -/
def yarn_conditions : YarnBalls where
  large := { sock := 3, hat := 5, sweater := 1 }
  small := { sock := 0, hat := 2, sweater := 1/2 }

/-- The maximum number of socks that can be knitted -/
def max_socks : ℕ := 18

/-- Theorem stating that the maximum number of socks that can be knitted is 18 -/
theorem max_socks_is_eighteen (y : YarnBalls) (h : y = yarn_conditions) : 
  (∃ (n : ℕ), n ≤ max_socks ∧ 
    n * y.large.sock ≤ y.large.hat * y.large.sock + y.small.hat * y.large.sock ∧
    ∀ (m : ℕ), m * y.large.sock ≤ y.large.hat * y.large.sock + y.small.hat * y.large.sock → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_max_socks_is_eighteen_l4184_418488


namespace NUMINAMATH_CALUDE_triangle_side_length_l4184_418474

/-- Given a triangle ABC with area √3, angle B = 60°, and a² + c² = 3ac, prove that the length of side b is 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2) * a * c * Real.sin B = Real.sqrt 3 →  -- Area condition
  B = π/3 →                                   -- Angle B = 60°
  a^2 + c^2 = 3*a*c →                         -- Given equation
  b = 2 * Real.sqrt 2 := by                   -- Conclusion
sorry


end NUMINAMATH_CALUDE_triangle_side_length_l4184_418474


namespace NUMINAMATH_CALUDE_one_true_statement_proves_normal_one_false_statement_proves_normal_one_statement_proves_normal_l4184_418499

-- Define the types of people on the island
inductive PersonType
  | Knight
  | Liar
  | Normal

-- Define a statement as either true or false
inductive Statement
  | True
  | False

-- Define a function to determine if a person can make a given statement
def canMakeStatement (person : PersonType) (statement : Statement) : Prop :=
  match person, statement with
  | PersonType.Knight, Statement.True => True
  | PersonType.Knight, Statement.False => False
  | PersonType.Liar, Statement.True => False
  | PersonType.Liar, Statement.False => True
  | PersonType.Normal, _ => True

-- Theorem: One true statement is sufficient to prove one is a normal person
theorem one_true_statement_proves_normal (person : PersonType) :
  canMakeStatement person Statement.True → person = PersonType.Normal :=
sorry

-- Theorem: One false statement is sufficient to prove one is a normal person
theorem one_false_statement_proves_normal (person : PersonType) :
  canMakeStatement person Statement.False → person = PersonType.Normal :=
sorry

-- Main theorem: Either one true or one false statement is sufficient to prove one is a normal person
theorem one_statement_proves_normal (person : PersonType) :
  (canMakeStatement person Statement.True ∨ canMakeStatement person Statement.False) →
  person = PersonType.Normal :=
sorry

end NUMINAMATH_CALUDE_one_true_statement_proves_normal_one_false_statement_proves_normal_one_statement_proves_normal_l4184_418499


namespace NUMINAMATH_CALUDE_min_max_problem_l4184_418437

theorem min_max_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (xy = 15 → x + y ≥ 2 * Real.sqrt 15 ∧ (x + y = 2 * Real.sqrt 15 ↔ x = Real.sqrt 15 ∧ y = Real.sqrt 15)) ∧
  (x + y = 15 → x * y ≤ 225 / 4 ∧ (x * y = 225 / 4 ↔ x = 15 / 2 ∧ y = 15 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_min_max_problem_l4184_418437


namespace NUMINAMATH_CALUDE_large_cube_volume_l4184_418413

theorem large_cube_volume (small_cube_surface_area : ℝ) (num_small_cubes : ℕ) :
  small_cube_surface_area = 96 →
  num_small_cubes = 8 →
  let small_cube_side := Real.sqrt (small_cube_surface_area / 6)
  let large_cube_side := small_cube_side * 2
  large_cube_side ^ 3 = 512 :=
by
  sorry

end NUMINAMATH_CALUDE_large_cube_volume_l4184_418413


namespace NUMINAMATH_CALUDE_det_of_specific_matrix_l4184_418461

theorem det_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![8, 4; -2, 3]
  Matrix.det A = 32 := by
sorry

end NUMINAMATH_CALUDE_det_of_specific_matrix_l4184_418461


namespace NUMINAMATH_CALUDE_principal_is_12000_l4184_418460

/-- Calculates the principal amount given the interest rate, time, and total interest. -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℚ) : ℚ :=
  interest / (rate * time.cast / 100)

/-- Theorem stating that given the specified conditions, the principal amount is $12000. -/
theorem principal_is_12000 (rate : ℚ) (time : ℕ) (interest : ℚ) 
  (h_rate : rate = 12)
  (h_time : time = 3)
  (h_interest : interest = 4320) :
  calculate_principal rate time interest = 12000 := by
  sorry

#eval calculate_principal 12 3 4320

end NUMINAMATH_CALUDE_principal_is_12000_l4184_418460


namespace NUMINAMATH_CALUDE_andrews_friends_pizza_l4184_418443

theorem andrews_friends_pizza (total_slices : ℕ) (slices_per_friend : ℕ) (num_friends : ℕ) :
  total_slices = 16 →
  slices_per_friend = 4 →
  total_slices = num_friends * slices_per_friend →
  num_friends = 4 := by
sorry

end NUMINAMATH_CALUDE_andrews_friends_pizza_l4184_418443


namespace NUMINAMATH_CALUDE_online_employees_probability_l4184_418495

/-- Probability of exactly k successes in n independent trials with probability p each -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ := sorry

/-- Probability of at least k successes in n independent trials with probability p each -/
def at_least_probability (n k : ℕ) (p : ℝ) : ℝ := sorry

theorem online_employees_probability (n : ℕ) (p : ℝ) 
  (h_n : n = 6) (h_p : p = 0.5) : 
  at_least_probability n 3 p = 21/32 ∧ 
  (∀ k : ℕ, at_least_probability n k p < 0.3 ↔ k ≥ 4) := by sorry

end NUMINAMATH_CALUDE_online_employees_probability_l4184_418495


namespace NUMINAMATH_CALUDE_aaron_reading_challenge_l4184_418433

theorem aaron_reading_challenge (average_pages : ℕ) (total_days : ℕ) (day1 day2 day3 day4 day5 : ℕ) :
  average_pages = 15 →
  total_days = 6 →
  day1 = 18 →
  day2 = 12 →
  day3 = 23 →
  day4 = 10 →
  day5 = 17 →
  ∃ (day6 : ℕ), (day1 + day2 + day3 + day4 + day5 + day6) / total_days = average_pages ∧ day6 = 10 :=
by sorry

end NUMINAMATH_CALUDE_aaron_reading_challenge_l4184_418433


namespace NUMINAMATH_CALUDE_units_digit_of_2009_pow_2008_plus_2013_l4184_418476

theorem units_digit_of_2009_pow_2008_plus_2013 :
  (2009^2008 + 2013) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2009_pow_2008_plus_2013_l4184_418476


namespace NUMINAMATH_CALUDE_trig_identity_l4184_418404

-- Statement of the trigonometric identity
theorem trig_identity (α β γ : ℝ) :
  Real.sin α + Real.sin β + Real.sin γ = 4 * Real.cos (α/2) * Real.cos (β/2) * Real.cos (γ/2) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l4184_418404


namespace NUMINAMATH_CALUDE_rogers_coin_donation_l4184_418429

theorem rogers_coin_donation (pennies nickels dimes coins_left : ℕ) :
  pennies = 42 →
  nickels = 36 →
  dimes = 15 →
  coins_left = 27 →
  pennies + nickels + dimes - coins_left = 66 := by
  sorry

end NUMINAMATH_CALUDE_rogers_coin_donation_l4184_418429


namespace NUMINAMATH_CALUDE_officers_on_duty_l4184_418491

theorem officers_on_duty (total_female : ℕ) (on_duty : ℕ) 
  (h1 : total_female = 1000)
  (h2 : on_duty / 2 = total_female / 4) : 
  on_duty = 500 := by
  sorry

end NUMINAMATH_CALUDE_officers_on_duty_l4184_418491


namespace NUMINAMATH_CALUDE_inscribed_sphere_in_cone_l4184_418459

/-- Given a right cone with base radius 15 cm and height 30 cm, 
    and an inscribed sphere with radius r = b√d - b cm, 
    prove that b + d = 12.5 -/
theorem inscribed_sphere_in_cone (b d : ℝ) : 
  let cone_base_radius : ℝ := 15
  let cone_height : ℝ := 30
  let sphere_radius : ℝ := b * (d.sqrt - 1)
  sphere_radius = (cone_height * cone_base_radius) / (cone_base_radius + (cone_base_radius^2 + cone_height^2).sqrt) →
  b + d = 12.5 := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_in_cone_l4184_418459


namespace NUMINAMATH_CALUDE_wrong_height_calculation_l4184_418431

/-- Proves that the wrongly written height of a boy is 176 cm given the following conditions:
  * There are 35 boys in a class
  * The initially calculated average height was 182 cm
  * One boy's height was recorded incorrectly
  * The boy's actual height is 106 cm
  * The correct average height is 180 cm
-/
theorem wrong_height_calculation (n : ℕ) (initial_avg correct_avg actual_height : ℝ) :
  n = 35 →
  initial_avg = 182 →
  correct_avg = 180 →
  actual_height = 106 →
  (n : ℝ) * initial_avg - (n : ℝ) * correct_avg + actual_height = 176 := by
  sorry

end NUMINAMATH_CALUDE_wrong_height_calculation_l4184_418431


namespace NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l4184_418423

/-- The orthocenter of a triangle ABC in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates is (5/2, 3, 7/2) -/
theorem orthocenter_of_specific_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (6, 4, 2)
  let C : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter A B C = (5/2, 3, 7/2) :=
sorry

end NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l4184_418423


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l4184_418466

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 1 ≥ -3 ∧ -2 * (x + 3) > 0}
  S = {x : ℝ | -4 ≤ x ∧ x < -3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l4184_418466


namespace NUMINAMATH_CALUDE_marble_problem_l4184_418438

theorem marble_problem (initial_marbles : ℕ) : 
  (initial_marbles * 40 / 100 / 2 = 20) → initial_marbles = 100 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l4184_418438


namespace NUMINAMATH_CALUDE_system_solution_l4184_418401

theorem system_solution (x y z : ℝ) : 
  (x * y * z) / (x + y) = 6/5 ∧ 
  (x * y * z) / (y + z) = 2 ∧ 
  (x * y * z) / (z + x) = 3/2 →
  ((x = 3 ∧ y = 2 ∧ z = 1) ∨ (x = -3 ∧ y = -2 ∧ z = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4184_418401


namespace NUMINAMATH_CALUDE_speed_conversion_l4184_418402

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- Given speed in meters per second -/
def speed_mps : ℝ := 20.0016

/-- Speed in kilometers per hour to be proven -/
def speed_kmph : ℝ := 72.00576

/-- Theorem stating that the given speed in km/h is equivalent to the speed in m/s -/
theorem speed_conversion : speed_kmph = speed_mps * mps_to_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l4184_418402


namespace NUMINAMATH_CALUDE_new_students_weight_l4184_418481

theorem new_students_weight (initial_count : ℕ) (replaced_weight1 replaced_weight2 avg_decrease : ℝ) :
  initial_count = 8 →
  replaced_weight1 = 85 →
  replaced_weight2 = 96 →
  avg_decrease = 7.5 →
  (initial_count : ℝ) * avg_decrease = (replaced_weight1 + replaced_weight2) - (new_student_weight1 + new_student_weight2) →
  new_student_weight1 + new_student_weight2 = 121 :=
by
  sorry

#check new_students_weight

end NUMINAMATH_CALUDE_new_students_weight_l4184_418481


namespace NUMINAMATH_CALUDE_centroid_equal_areas_l4184_418462

/-- The centroid of a triangle divides it into three equal-area triangles -/
theorem centroid_equal_areas (P Q R S : ℝ × ℝ) : 
  P = (-1, 3) → Q = (2, 7) → R = (4, 0) → 
  S.1 = (P.1 + Q.1 + R.1) / 3 → 
  S.2 = (P.2 + Q.2 + R.2) / 3 → 
  8 * S.1 + 3 * S.2 = 70 / 3 := by
  sorry

#check centroid_equal_areas

end NUMINAMATH_CALUDE_centroid_equal_areas_l4184_418462


namespace NUMINAMATH_CALUDE_ceiling_sqrt_twelve_count_l4184_418414

theorem ceiling_sqrt_twelve_count : 
  (Finset.filter (fun x : ℕ => ⌈Real.sqrt x⌉ = 12) (Finset.range 1000)).card = 25 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_twelve_count_l4184_418414


namespace NUMINAMATH_CALUDE_perpendicular_lines_sum_l4184_418479

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 4 * y - 2 = 0
def l₂ (b : ℝ) (x y : ℝ) : Prop := 2 * x - 5 * y + b = 0

-- Define perpendicularity of two lines
def perpendicular (a b : ℝ) : Prop := a * 2 + 4 * (-5) = 0

-- Define the foot of the perpendicular
def foot_of_perpendicular (a b c : ℝ) : Prop := l₁ a 1 c ∧ l₂ b 1 c

-- Theorem statement
theorem perpendicular_lines_sum (a b c : ℝ) :
  perpendicular a b →
  foot_of_perpendicular a b c →
  a + b + c = -4 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sum_l4184_418479


namespace NUMINAMATH_CALUDE_prob_three_consecutive_in_ten_l4184_418493

/-- The number of ways to arrange n items -/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange n items with a block of k consecutive items -/
def arrangements_with_block (n k : ℕ) : ℕ := (n - k + 1) * k.factorial * (n - k).factorial

/-- The probability of k specific items being consecutive in a random arrangement of n items -/
def prob_consecutive (n k : ℕ) : ℚ :=
  (arrangements_with_block n k : ℚ) / (arrangements n : ℚ)

theorem prob_three_consecutive_in_ten :
  prob_consecutive 10 3 = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_prob_three_consecutive_in_ten_l4184_418493


namespace NUMINAMATH_CALUDE_x_squared_y_squared_range_l4184_418400

theorem x_squared_y_squared_range (x y : ℝ) (h : x^2 + y^2 = 2*x) :
  0 ≤ x^2 * y^2 ∧ x^2 * y^2 ≤ 27/16 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_y_squared_range_l4184_418400


namespace NUMINAMATH_CALUDE_weight_density_half_fluid_density_l4184_418492

/-- A spring-mass system submerged in a fluid -/
structure SpringMassSystem where
  /-- Spring constant -/
  k : ℝ
  /-- Mass of the weight -/
  m : ℝ
  /-- Acceleration due to gravity -/
  g : ℝ
  /-- Density of the fluid (kerosene) -/
  ρ_fluid : ℝ
  /-- Density of the weight material -/
  ρ_material : ℝ
  /-- Extension of the spring -/
  x : ℝ

/-- The theorem stating that the density of the weight material is half the density of the fluid -/
theorem weight_density_half_fluid_density (system : SpringMassSystem) 
  (h1 : system.k * system.x = system.m * system.g)  -- Force balance in air
  (h2 : system.m * system.g + system.k * system.x = system.ρ_fluid * system.g * (system.m / system.ρ_material))  -- Force balance in fluid
  (h3 : system.ρ_fluid > 0)  -- Fluid density is positive
  (h4 : system.m > 0)  -- Mass is positive
  (h5 : system.g > 0)  -- Gravity is positive
  : system.ρ_material = system.ρ_fluid / 2 := by
  sorry

#eval 800 / 2  -- Should output 400

end NUMINAMATH_CALUDE_weight_density_half_fluid_density_l4184_418492


namespace NUMINAMATH_CALUDE_range_of_a_range_of_x_l4184_418447

-- Define the quadratic function
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Part 1
theorem range_of_a (a b : ℝ) :
  (f a b 1 = 1) →
  (∀ x ∈ Set.Ioo 2 5, f a b x > 0) →
  a ∈ Set.Ioi (3 - 2 * Real.sqrt 2) :=
sorry

-- Part 2
theorem range_of_x (a b x : ℝ) :
  (f a b 1 = 1) →
  (∀ a ∈ Set.Icc (-2) (-1), f a b x > 0) →
  x ∈ Set.Ioo ((1 - Real.sqrt 17) / 4) ((1 + Real.sqrt 17) / 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_x_l4184_418447


namespace NUMINAMATH_CALUDE_triangle_properties_l4184_418465

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  -- Area condition
  3 * Real.sin t.A = (1/2) * t.b * t.c * Real.sin t.A ∧
  -- Perimeter condition
  t.a + t.b + t.c = 4 * (Real.sqrt 2 + 1) ∧
  -- Sine condition
  Real.sin t.B + Real.sin t.C = Real.sqrt 2 * Real.sin t.A

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  t.a = 4 ∧ 
  Real.cos t.A = 1/3 ∧ 
  Real.cos (2 * t.A - π/3) = (4 * Real.sqrt 6 - 7) / 18 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l4184_418465


namespace NUMINAMATH_CALUDE_lizzys_money_l4184_418416

theorem lizzys_money (mother_gave : ℕ) (spent_on_candy : ℕ) (uncle_gave : ℕ) (final_amount : ℕ) :
  mother_gave = 80 →
  spent_on_candy = 50 →
  uncle_gave = 70 →
  final_amount = 140 →
  ∃ (father_gave : ℕ), father_gave = 40 ∧ mother_gave + father_gave - spent_on_candy + uncle_gave = final_amount :=
by sorry

end NUMINAMATH_CALUDE_lizzys_money_l4184_418416


namespace NUMINAMATH_CALUDE_no_self_power_divisibility_l4184_418405

theorem no_self_power_divisibility (n : ℕ) : n > 1 → ¬(n ∣ 2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_no_self_power_divisibility_l4184_418405


namespace NUMINAMATH_CALUDE_negative_three_inequality_l4184_418444

theorem negative_three_inequality (a b : ℝ) : a < b → -3 * a > -3 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_three_inequality_l4184_418444


namespace NUMINAMATH_CALUDE_inequality_proof_l4184_418475

theorem inequality_proof (x : ℝ) : 
  Real.sqrt (3 * x^2 + 2 * x + 1) + Real.sqrt (3 * x^2 - 4 * x + 2) ≥ Real.sqrt 51 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4184_418475


namespace NUMINAMATH_CALUDE_simplify_expression_l4184_418457

theorem simplify_expression (a : ℝ) (h1 : a ≠ -3) (h2 : a ≠ 1) :
  (1 - 4 / (a + 3)) / ((a^2 - 2*a + 1) / (2*a + 6)) = 2 / (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4184_418457


namespace NUMINAMATH_CALUDE_marble_difference_l4184_418403

theorem marble_difference (total_yellow : ℕ) (jar1_red_ratio jar1_yellow_ratio jar2_red_ratio jar2_yellow_ratio : ℕ) :
  total_yellow = 140 →
  jar1_red_ratio = 7 →
  jar1_yellow_ratio = 3 →
  jar2_red_ratio = 3 →
  jar2_yellow_ratio = 2 →
  ∃ (jar1_total jar2_total : ℕ),
    jar1_total = jar2_total ∧
    jar1_total * jar1_yellow_ratio / (jar1_red_ratio + jar1_yellow_ratio) +
    jar2_total * jar2_yellow_ratio / (jar2_red_ratio + jar2_yellow_ratio) = total_yellow ∧
    jar1_total * jar1_red_ratio / (jar1_red_ratio + jar1_yellow_ratio) -
    jar2_total * jar2_red_ratio / (jar2_red_ratio + jar2_yellow_ratio) = 20 :=
by sorry

end NUMINAMATH_CALUDE_marble_difference_l4184_418403


namespace NUMINAMATH_CALUDE_a_10_equals_21_l4184_418417

def S (n : ℕ+) : ℕ := n^2 + 2*n

def a (n : ℕ+) : ℕ := S n - S (n-1)

theorem a_10_equals_21 : a 10 = 21 := by sorry

end NUMINAMATH_CALUDE_a_10_equals_21_l4184_418417


namespace NUMINAMATH_CALUDE_H_function_iff_non_decreasing_l4184_418467

/-- A function f: ℝ → ℝ is an H function if for any x₁ ≠ x₂, x₁f(x₁) + x₂f(x₂) ≥ x₁f(x₂) + x₂f(x₁) -/
def is_H_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ ≥ x₁ * f x₂ + x₂ * f x₁

/-- A function f: ℝ → ℝ is non-decreasing if for any x₁ < x₂, f(x₁) ≤ f(x₂) -/
def is_non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂

theorem H_function_iff_non_decreasing (f : ℝ → ℝ) :
  is_H_function f ↔ is_non_decreasing f := by
  sorry

end NUMINAMATH_CALUDE_H_function_iff_non_decreasing_l4184_418467


namespace NUMINAMATH_CALUDE_parallelogram_dimensions_l4184_418468

/-- Proves the side lengths and perimeter of a parallelogram given its area, side ratio, and one angle -/
theorem parallelogram_dimensions (area : ℝ) (angle : ℝ) (h_area : area = 972) (h_angle : angle = 45 * π / 180) :
  ∃ (side1 side2 perimeter : ℝ),
    side1 / side2 = 4 / 3 ∧
    area = side1 * side2 * Real.sin angle ∧
    side1 = 36 * 2^(3/4) ∧
    side2 = 27 * 2^(3/4) ∧
    perimeter = 126 * 2^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_dimensions_l4184_418468


namespace NUMINAMATH_CALUDE_digit_sum_problem_l4184_418454

theorem digit_sum_problem (x y z u : ℕ) : 
  x < 10 → y < 10 → z < 10 → u < 10 →
  x ≠ y → x ≠ z → x ≠ u → y ≠ z → y ≠ u → z ≠ u →
  10 * x + y + 10 * z + x = 10 * u + x - (10 * z + x) →
  x + y + z + u = 18 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l4184_418454


namespace NUMINAMATH_CALUDE_rex_to_total_ratio_l4184_418487

/-- Represents the number of Pokemon cards collected by each person -/
structure PokemonCards where
  nicole : ℕ
  cindy : ℕ
  rex : ℕ

/-- Represents the problem statement and conditions -/
def pokemon_card_problem (cards : PokemonCards) : Prop :=
  cards.nicole = 400 ∧
  cards.cindy = 2 * cards.nicole ∧
  cards.rex = 150 * 4 ∧
  cards.rex < cards.nicole + cards.cindy

/-- Theorem stating the ratio of Rex's cards to Nicole and Cindy's combined total -/
theorem rex_to_total_ratio (cards : PokemonCards) 
  (h : pokemon_card_problem cards) : 
  (cards.rex : ℚ) / (cards.nicole + cards.cindy : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rex_to_total_ratio_l4184_418487


namespace NUMINAMATH_CALUDE_student_number_problem_l4184_418419

theorem student_number_problem (x : ℝ) : 2 * x - 152 = 102 → x = 127 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l4184_418419


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l4184_418484

/-- The distance between the vertices of the hyperbola x^2/64 - y^2/49 = 1 is 16 -/
theorem hyperbola_vertex_distance : 
  let h : ℝ → ℝ → Prop := λ x y => x^2/64 - y^2/49 = 1
  ∃ x₁ x₂ : ℝ, h x₁ 0 ∧ h x₂ 0 ∧ |x₁ - x₂| = 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l4184_418484


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l4184_418440

theorem quadratic_solution_property (k : ℚ) : 
  (∃ a b : ℚ, 
    (5 * a^2 + 7 * a + k = 0) ∧ 
    (5 * b^2 + 7 * b + k = 0) ∧ 
    (abs (a - b) = a^2 + b^2)) ↔ 
  (k = 21/25 ∨ k = -21/25) := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l4184_418440


namespace NUMINAMATH_CALUDE_fraction_addition_l4184_418472

theorem fraction_addition (d : ℝ) : (6 + 4 * d) / 9 + 3 / 2 = (39 + 8 * d) / 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l4184_418472


namespace NUMINAMATH_CALUDE_triangle_area_rational_l4184_418428

-- Define a point with integer coordinates
structure IntPoint where
  x : Int
  y : Int

-- Define a triangle with three IntPoints
structure IntTriangle where
  p1 : IntPoint
  p2 : IntPoint
  p3 : IntPoint

-- Function to calculate the area of a triangle given its vertices
def triangleArea (t : IntTriangle) : ℚ :=
  let x1 := t.p1.x
  let y1 := t.p1.y
  let x2 := t.p2.x
  let y2 := t.p2.y
  let x3 := t.p3.x
  let y3 := t.p3.y
  (1 / 2 : ℚ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

-- Theorem stating that the area of the triangle is rational
theorem triangle_area_rational (t : IntTriangle) 
  (h1 : t.p1.x - t.p1.y = 1)
  (h2 : t.p2.x - t.p2.y = 1)
  (h3 : t.p3.x - t.p3.y = 1) :
  ∃ (q : ℚ), triangleArea t = q :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_area_rational_l4184_418428


namespace NUMINAMATH_CALUDE_solution_set_equality_l4184_418449

def equation (x : ℝ) : Prop :=
  (1 / (x^2 + 12*x - 9)) + (1 / (x^2 + 3*x - 9)) + (1 / (x^2 - 12*x - 9)) = 0

theorem solution_set_equality :
  {x : ℝ | equation x} = {1, -9, 3, -3} := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l4184_418449


namespace NUMINAMATH_CALUDE_inequality_solution_l4184_418442

theorem inequality_solution (x : ℝ) : 
  (x^2 / (x - 2) ≥ 3 / (x + 2) + 7 / 5) ↔ (x > -2 ∧ x ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l4184_418442


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l4184_418482

theorem painted_cube_theorem (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 ↔ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l4184_418482


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4184_418483

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2) 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4184_418483


namespace NUMINAMATH_CALUDE_fraction_simplification_l4184_418494

theorem fraction_simplification (x y : ℚ) 
  (hx : x = 2/7) 
  (hy : y = 8/11) : 
  (7*x + 11*y) / (77*x*y) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4184_418494


namespace NUMINAMATH_CALUDE_part1_part2_part3_l4184_418436

/-- Definition of X(n) function -/
def is_X_n_function (f : ℝ → ℝ) : Prop :=
  ∃ (n : ℝ), ∀ (x : ℝ), f (2 * n - x) = f x

/-- Part 1: Prove that |x| and x^2 - x are X(n) functions -/
theorem part1 :
  (is_X_n_function (fun x => |x|)) ∧
  (is_X_n_function (fun x => x^2 - x)) :=
sorry

/-- Part 2: Prove k = -1 for the given parabola conditions -/
theorem part2 (k : ℝ) :
  (∀ x, (x^2 + k - 4) * (x^2 + k - 4) ≤ 0 → 
   ((0 - x)^2 + (k - 4))^2 = 3 * (x^2 + k - 4)^2) →
  k = -1 :=
sorry

/-- Part 3: Prove t = -2 or t = 0 for the given quadratic function conditions -/
theorem part3 (a b t : ℝ) :
  (∀ x, (a*x^2 + b*x - 4) = (a*(2-x)^2 + b*(2-x) - 4)) →
  (a*(-1)^2 + b*(-1) - 4 = 2) →
  (∀ x ∈ Set.Icc t (t+4), a*x^2 + b*x - 4 ≥ -6) →
  (∃ x ∈ Set.Icc t (t+4), a*x^2 + b*x - 4 = 12) →
  (t = -2 ∨ t = 0) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l4184_418436


namespace NUMINAMATH_CALUDE_alice_numbers_l4184_418477

theorem alice_numbers : ∃ x y : ℝ, x * y = 12 ∧ x + y = 7 ∧ ({x, y} : Set ℝ) = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_alice_numbers_l4184_418477


namespace NUMINAMATH_CALUDE_hazel_drank_one_cup_l4184_418411

def lemonade_problem (total_cups : ℕ) (sold_to_kids : ℕ) : Prop :=
  let sold_to_crew : ℕ := total_cups / 2
  let given_to_friends : ℕ := sold_to_kids / 2
  let remaining_cups : ℕ := total_cups - (sold_to_crew + sold_to_kids + given_to_friends)
  remaining_cups = 1

theorem hazel_drank_one_cup : lemonade_problem 56 18 := by
  sorry

end NUMINAMATH_CALUDE_hazel_drank_one_cup_l4184_418411
