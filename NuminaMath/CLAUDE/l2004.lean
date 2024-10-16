import Mathlib

namespace NUMINAMATH_CALUDE_sams_new_nickels_l2004_200431

/-- The number of nickels Sam's dad gave him -/
def nickels_from_dad (initial_nickels final_nickels : ℕ) : ℕ :=
  final_nickels - initial_nickels

/-- Proof that Sam's dad gave him 39 nickels -/
theorem sams_new_nickels :
  let initial_nickels : ℕ := 24
  let final_nickels : ℕ := 63
  nickels_from_dad initial_nickels final_nickels = 39 := by
sorry

end NUMINAMATH_CALUDE_sams_new_nickels_l2004_200431


namespace NUMINAMATH_CALUDE_two_tangent_lines_l2004_200419

/-- The cubic function f(x) = -x³ + 6x² - 9x + 8 -/
def f (x : ℝ) : ℝ := -x^3 + 6*x^2 - 9*x + 8

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := -3*x^2 + 12*x - 9

/-- Theorem: There are exactly two tangent lines from (0, 0) to the graph of f(x) -/
theorem two_tangent_lines :
  ∃! (s : Finset ℝ), s.card = 2 ∧
    ∀ x₀ ∈ s, f x₀ + f' x₀ * (-x₀) = 0 ∧
    ∀ x ∉ s, f x + f' x * (-x) ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_two_tangent_lines_l2004_200419


namespace NUMINAMATH_CALUDE_job_completion_time_l2004_200485

theorem job_completion_time (x : ℝ) : 
  x > 0 → 
  4 * (1/x + 1/30) = 0.4 → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l2004_200485


namespace NUMINAMATH_CALUDE_exp_equals_derivative_l2004_200409

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem exp_equals_derivative :
  ∀ x : ℝ, f x = deriv f x :=
by sorry

end NUMINAMATH_CALUDE_exp_equals_derivative_l2004_200409


namespace NUMINAMATH_CALUDE_min_sum_with_constraint_l2004_200448

theorem min_sum_with_constraint (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x^2 + y^2 + z^2 + x + 2*y + 3*z = 13/4) :
  x + y + z ≥ (-3 + Real.sqrt 22) / 2 ∧
  ∃ (x' y' z' : ℝ), 0 ≤ x' ∧ 0 ≤ y' ∧ 0 ≤ z' ∧
    x'^2 + y'^2 + z'^2 + x' + 2*y' + 3*z' = 13/4 ∧
    x' + y' + z' = (-3 + Real.sqrt 22) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_constraint_l2004_200448


namespace NUMINAMATH_CALUDE_banana_arrangements_l2004_200490

/-- The number of unique arrangements of letters in a word -/
def uniqueArrangements (totalLetters : Nat) (repetitions : List Nat) : Nat :=
  Nat.factorial totalLetters / (repetitions.map Nat.factorial).prod

/-- The word "BANANA" has 6 letters -/
def totalLetters : Nat := 6

/-- The repetitions of letters in "BANANA": 3 A's, 2 N's, and 1 B (which we don't need to include) -/
def letterRepetitions : List Nat := [3, 2]

/-- Theorem: The number of unique arrangements of letters in "BANANA" is 60 -/
theorem banana_arrangements :
  uniqueArrangements totalLetters letterRepetitions = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l2004_200490


namespace NUMINAMATH_CALUDE_line_perpendicular_implies_plane_perpendicular_l2004_200426

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (in_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_implies_plane_perpendicular
  (α β : Plane) (m : Line)
  (distinct : α ≠ β)
  (m_in_α : in_plane m α)
  (m_perp_β : perpendicular_line_plane m β) :
  perpendicular_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_implies_plane_perpendicular_l2004_200426


namespace NUMINAMATH_CALUDE_fish_weight_l2004_200404

theorem fish_weight : 
  ∀ w : ℝ, w = 2 + w / 3 → w = 3 := by sorry

end NUMINAMATH_CALUDE_fish_weight_l2004_200404


namespace NUMINAMATH_CALUDE_victor_trips_l2004_200445

/-- Calculate the number of trips needed to carry a given number of trays -/
def tripsNeeded (trays : ℕ) (capacity : ℕ) : ℕ :=
  (trays + capacity - 1) / capacity

/-- The problem setup -/
def victorProblem : Prop :=
  let capacity := 6
  let table1 := 23
  let table2 := 5
  let table3 := 12
  let table4 := 18
  let table5 := 27
  let totalTrips := tripsNeeded table1 capacity + tripsNeeded table2 capacity +
                    tripsNeeded table3 capacity + tripsNeeded table4 capacity +
                    tripsNeeded table5 capacity
  totalTrips = 15

theorem victor_trips : victorProblem := by
  sorry

end NUMINAMATH_CALUDE_victor_trips_l2004_200445


namespace NUMINAMATH_CALUDE_cuboid_height_l2004_200497

/-- Proves that a rectangular parallelepiped with given dimensions has a specific height -/
theorem cuboid_height (width length sum_of_edges : ℝ) (h : ℝ) : 
  width = 30 →
  length = 22 →
  sum_of_edges = 224 →
  4 * length + 4 * width + 4 * h = sum_of_edges →
  h = 4 := by
sorry

end NUMINAMATH_CALUDE_cuboid_height_l2004_200497


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l2004_200482

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_equal_volume (l w h : ℝ) (cube_edge : ℝ) :
  l = 10 ∧ w = 5 ∧ h = 20 →
  cube_edge^3 = l * w * h →
  6 * cube_edge^2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l2004_200482


namespace NUMINAMATH_CALUDE_garment_fraction_l2004_200477

theorem garment_fraction (bikini_fraction trunks_fraction : ℝ) 
  (h1 : bikini_fraction = 0.38) 
  (h2 : trunks_fraction = 0.25) : 
  bikini_fraction + trunks_fraction = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_garment_fraction_l2004_200477


namespace NUMINAMATH_CALUDE_brendas_age_l2004_200446

theorem brendas_age (addison brenda janet : ℚ) 
  (h1 : addison = 4 * brenda)
  (h2 : janet = brenda + 8)
  (h3 : addison = janet + 2) :
  brenda = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_brendas_age_l2004_200446


namespace NUMINAMATH_CALUDE_jimmy_speed_l2004_200430

theorem jimmy_speed (mary_speed : ℝ) (total_distance : ℝ) (time : ℝ) (jimmy_speed : ℝ) : 
  mary_speed = 5 →
  total_distance = 9 →
  time = 1 →
  jimmy_speed = total_distance - mary_speed * time →
  jimmy_speed = 4 :=
by sorry

end NUMINAMATH_CALUDE_jimmy_speed_l2004_200430


namespace NUMINAMATH_CALUDE_equation_solution_l2004_200479

theorem equation_solution : ∃ x : ℚ, (3/4 : ℚ) + 1/x = 7/8 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2004_200479


namespace NUMINAMATH_CALUDE_salary_change_percentage_salary_loss_percentage_l2004_200413

theorem salary_change_percentage (original : ℝ) (original_positive : 0 < original) :
  let decreased := original * (1 - 0.6)
  let increased := decreased * (1 + 0.6)
  increased = original * 0.64 :=
by
  sorry

theorem salary_loss_percentage (original : ℝ) (original_positive : 0 < original) :
  let decreased := original * (1 - 0.6)
  let increased := decreased * (1 + 0.6)
  (original - increased) / original = 0.36 :=
by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_salary_loss_percentage_l2004_200413


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l2004_200452

/-- The x-intercept of the line 3x + 5y = 20 is (20/3, 0) -/
theorem x_intercept_of_line (x y : ℚ) : 
  3 * x + 5 * y = 20 → y = 0 → x = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l2004_200452


namespace NUMINAMATH_CALUDE_prop_A_sufficient_not_necessary_for_prop_B_l2004_200463

theorem prop_A_sufficient_not_necessary_for_prop_B :
  (∀ a b : ℝ, a < b ∧ b < 0 → a * b > b^2) ∧
  (∃ a b : ℝ, a * b > b^2 ∧ ¬(a < b ∧ b < 0)) :=
by sorry

end NUMINAMATH_CALUDE_prop_A_sufficient_not_necessary_for_prop_B_l2004_200463


namespace NUMINAMATH_CALUDE_peters_age_one_third_of_jacobs_l2004_200478

/-- Proves the number of years ago when Peter's age was one-third of Jacob's age -/
theorem peters_age_one_third_of_jacobs (peter_current_age jacob_current_age years_ago : ℕ) :
  peter_current_age = 16 →
  jacob_current_age = peter_current_age + 12 →
  peter_current_age - years_ago = (jacob_current_age - years_ago) / 3 →
  years_ago = 10 := by sorry

end NUMINAMATH_CALUDE_peters_age_one_third_of_jacobs_l2004_200478


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l2004_200457

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 4) (h2 : l^2 + w^2 = d^2) :
  l * w = (20 / 41) * d^2 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l2004_200457


namespace NUMINAMATH_CALUDE_tetrahedral_pile_remaining_marbles_l2004_200427

/-- The number of marbles in a tetrahedral pile of height k -/
def tetrahedralPile (k : ℕ) : ℕ := k * (k + 1) * (k + 2) / 6

/-- The total number of marbles -/
def totalMarbles : ℕ := 60

/-- The height of the largest possible tetrahedral pile -/
def maxHeight : ℕ := 6

/-- The number of remaining marbles -/
def remainingMarbles : ℕ := totalMarbles - tetrahedralPile maxHeight

theorem tetrahedral_pile_remaining_marbles :
  remainingMarbles = 4 := by sorry

end NUMINAMATH_CALUDE_tetrahedral_pile_remaining_marbles_l2004_200427


namespace NUMINAMATH_CALUDE_fraction_simplification_l2004_200466

theorem fraction_simplification (d : ℝ) : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2004_200466


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2004_200459

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_sum : a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 16) :
  a 2 + a 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2004_200459


namespace NUMINAMATH_CALUDE_fibonacci_rectangle_division_l2004_200439

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- A rectangle that can be divided into n squares -/
structure DivisibleRectangle (n : ℕ) :=
  (width : ℕ)
  (height : ℕ)
  (divides_into_squares : ∃ (squares : Finset (ℕ × ℕ)), 
    squares.card = n ∧ 
    (∀ (s : ℕ × ℕ), s ∈ squares → s.1 * s.2 ≤ width * height) ∧
    (∀ (s1 s2 s3 : ℕ × ℕ), s1 ∈ squares → s2 ∈ squares → s3 ∈ squares → 
      s1 = s2 ∧ s2 = s3 → s1 = s2))

/-- Theorem: For each natural number n, there exists a rectangle that can be 
    divided into n squares with no more than two squares being the same size -/
theorem fibonacci_rectangle_division (n : ℕ) : 
  ∃ (rect : DivisibleRectangle n), rect.width = fib n ∧ rect.height = fib (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_rectangle_division_l2004_200439


namespace NUMINAMATH_CALUDE_no_solutions_exist_l2004_200433

theorem no_solutions_exist : ¬ ∃ (x y z : ℝ), (x + y = 3) ∧ (x * y - z^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_exist_l2004_200433


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l2004_200471

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (2^(Real.tan x) - 2^(Real.sin x)) / x^2 else 0

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = Real.log (Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l2004_200471


namespace NUMINAMATH_CALUDE_triangle_tangent_ratio_l2004_200493

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if tan A * tan B = 4(tan A + tan B) * tan C, then (a^2 + b^2) / c^2 = 9 -/
theorem triangle_tangent_ratio (a b c A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) → (0 < B) → (B < π) → (0 < C) → (C < π) →
  (A + B + C = π) →
  (Real.tan A * Real.tan B = 4 * (Real.tan A + Real.tan B) * Real.tan C) →
  ((a^2 + b^2) / c^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_ratio_l2004_200493


namespace NUMINAMATH_CALUDE_tractor_oil_theorem_l2004_200444

/-- Represents the remaining oil in liters after t hours of work -/
def remaining_oil (initial_oil : ℝ) (consumption_rate : ℝ) (t : ℝ) : ℝ :=
  initial_oil - consumption_rate * t

theorem tractor_oil_theorem (initial_oil : ℝ) (consumption_rate : ℝ) (t : ℝ) :
  initial_oil = 50 → consumption_rate = 8 →
  (∀ t, remaining_oil initial_oil consumption_rate t = 50 - 8 * t) ∧
  (remaining_oil initial_oil consumption_rate 4 = 18) := by
  sorry


end NUMINAMATH_CALUDE_tractor_oil_theorem_l2004_200444


namespace NUMINAMATH_CALUDE_lunch_ratio_is_one_half_l2004_200406

/-- The number of school days in the academic year -/
def total_school_days : ℕ := 180

/-- The number of days Becky packs her lunch -/
def becky_lunch_days : ℕ := 45

/-- The number of days Aliyah packs her lunch -/
def aliyah_lunch_days : ℕ := 2 * becky_lunch_days

/-- The ratio of Aliyah's lunch-packing days to total school days -/
def lunch_ratio : ℚ := aliyah_lunch_days / total_school_days

theorem lunch_ratio_is_one_half : lunch_ratio = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_lunch_ratio_is_one_half_l2004_200406


namespace NUMINAMATH_CALUDE_angle_C_is_30_l2004_200455

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the property that the sum of angles in a triangle is 180°
axiom triangle_angle_sum (t : Triangle) : t.A + t.B + t.C = 180

-- Theorem: If the sum of angles A and B in triangle ABC is 150°, then angle C is 30°
theorem angle_C_is_30 (t : Triangle) (h : t.A + t.B = 150) : t.C = 30 := by
  sorry


end NUMINAMATH_CALUDE_angle_C_is_30_l2004_200455


namespace NUMINAMATH_CALUDE_pass_percentage_l2004_200411

theorem pass_percentage 
  (passed_english : Real) 
  (passed_math : Real) 
  (failed_both : Real) 
  (h1 : passed_english = 63) 
  (h2 : passed_math = 65) 
  (h3 : failed_both = 27) : 
  100 - failed_both = 73 := by
  sorry

end NUMINAMATH_CALUDE_pass_percentage_l2004_200411


namespace NUMINAMATH_CALUDE_max_tan_A_in_triangle_l2004_200487

open Real

theorem max_tan_A_in_triangle (a b c A B C : ℝ) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  -- Given conditions
  a = 2 →
  b * cos C - c * cos B = 4 →
  π/4 ≤ C ∧ C ≤ π/3 →
  -- Conclusion
  (∃ (max_tan_A : ℝ), max_tan_A = 1/2 ∧ ∀ (tan_A : ℝ), tan_A = tan A → tan_A ≤ max_tan_A) :=
by sorry

end NUMINAMATH_CALUDE_max_tan_A_in_triangle_l2004_200487


namespace NUMINAMATH_CALUDE_cookie_bags_theorem_l2004_200480

/-- Given a total number of cookies and cookies per bag, calculate the number of bags. -/
def number_of_bags (total_cookies : ℕ) (cookies_per_bag : ℕ) : ℕ :=
  total_cookies / cookies_per_bag

/-- Theorem: Given 33 cookies in total and 11 cookies per bag, the number of bags is 3. -/
theorem cookie_bags_theorem :
  number_of_bags 33 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookie_bags_theorem_l2004_200480


namespace NUMINAMATH_CALUDE_halloween_candy_distribution_l2004_200418

theorem halloween_candy_distribution (initial_candy : ℕ) (eaten_candy : ℕ) (num_piles : ℕ) 
  (h1 : initial_candy = 78)
  (h2 : eaten_candy = 30)
  (h3 : num_piles = 6)
  : (initial_candy - eaten_candy) / num_piles = 8 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_distribution_l2004_200418


namespace NUMINAMATH_CALUDE_triangle_problem_l2004_200468

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_problem (abc : Triangle) (h1 : abc.a * Real.tan abc.B = 2 * abc.b * Real.sin abc.A)
  (h2 : abc.b = Real.sqrt 3) (h3 : abc.A = 5 * Real.pi / 12) :
  abc.B = Real.pi / 3 ∧ 
  (1 / 2 * abc.b * abc.c * Real.sin abc.A) = (3 + Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2004_200468


namespace NUMINAMATH_CALUDE_det_A_eq_121_l2004_200494

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, -2; 4, 5, -3; 6, 2, 7]

theorem det_A_eq_121 : A.det = 121 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_121_l2004_200494


namespace NUMINAMATH_CALUDE_unique_w_value_l2004_200410

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ := sorry

def consecutive_digit_sums_prime (n : ℕ) : Prop := sorry

theorem unique_w_value (w : ℕ) :
  w > 0 →
  digit_sum (10^w - 74) = 440 →
  consecutive_digit_sums_prime (10^w - 74) →
  w = 50 := by sorry

end NUMINAMATH_CALUDE_unique_w_value_l2004_200410


namespace NUMINAMATH_CALUDE_gerald_donated_quarter_l2004_200442

/-- The fraction of toy cars Gerald donated -/
def gerald_donation (initial_cars : ℕ) (remaining_cars : ℕ) : ℚ :=
  (initial_cars - remaining_cars : ℚ) / initial_cars

/-- Theorem stating that Gerald donated 1/4 of his toy cars -/
theorem gerald_donated_quarter :
  gerald_donation 20 15 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_gerald_donated_quarter_l2004_200442


namespace NUMINAMATH_CALUDE_exponential_function_property_l2004_200499

theorem exponential_function_property (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  ∀ (x₁ x₂ : ℝ), (fun x ↦ a^x) (x₁ + x₂) = (fun x ↦ a^x) x₁ * (fun x ↦ a^x) x₂ := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_property_l2004_200499


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2004_200473

theorem linear_equation_solution (m : ℤ) :
  (∃ x : ℚ, x^(|m|) - m*x + 1 = 0 ∧ ∃ a b : ℚ, a ≠ 0 ∧ a*x + b = 0) →
  (∃ x : ℚ, x^(|m|) - m*x + 1 = 0 ∧ x = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l2004_200473


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l2004_200495

theorem quadratic_roots_sum_squares (h : ℝ) : 
  (∃ x y : ℝ, x^2 + 4*h*x = 8 ∧ y^2 + 4*h*y = 8 ∧ x^2 + y^2 = 20) →
  |h| = 1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l2004_200495


namespace NUMINAMATH_CALUDE_a1_iff_a2017_positive_l2004_200465

/-- An arithmetic-geometric sequence -/
structure ArithmeticGeometricSequence where
  a : ℕ → ℝ
  q : ℝ

/-- The theorem stating that for an arithmetic-geometric sequence with q = 0,
    a₁ > 0 if and only if a₂₀₁₇ > 0 -/
theorem a1_iff_a2017_positive (seq : ArithmeticGeometricSequence) 
    (h_q : seq.q = 0) :
    seq.a 1 > 0 ↔ seq.a 2017 > 0 := by
  sorry

end NUMINAMATH_CALUDE_a1_iff_a2017_positive_l2004_200465


namespace NUMINAMATH_CALUDE_fraction_ordering_l2004_200408

theorem fraction_ordering : 8 / 31 < 11 / 33 ∧ 11 / 33 < 12 / 29 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l2004_200408


namespace NUMINAMATH_CALUDE_initial_sets_count_l2004_200436

/-- The number of letters available (A through J) -/
def num_letters : ℕ := 10

/-- The number of letters in each set of initials -/
def set_size : ℕ := 3

/-- The number of arrangements for three letters where two are identical -/
def repeated_letter_arrangements : ℕ := 3

/-- The number of different three-letter sets of initials possible using letters A through J, 
    where one letter can appear twice and the third must be different -/
theorem initial_sets_count : 
  num_letters * (num_letters - 1) * repeated_letter_arrangements = 270 := by
  sorry

end NUMINAMATH_CALUDE_initial_sets_count_l2004_200436


namespace NUMINAMATH_CALUDE_two_tails_one_head_prob_l2004_200456

/-- Represents a biased coin with probabilities for heads and tails -/
structure BiasedCoin where
  probHeads : ℝ
  probTails : ℝ
  prob_sum : probHeads + probTails = 1

/-- Calculates the probability of getting exactly two tails followed by one head within 5 flips -/
def prob_two_tails_one_head (c : BiasedCoin) : ℝ :=
  3 * (c.probTails * c.probTails * c.probTails * c.probHeads)

/-- The main theorem to be proved -/
theorem two_tails_one_head_prob :
  let c : BiasedCoin := ⟨0.3, 0.7, by norm_num⟩
  prob_two_tails_one_head c = 0.3087 := by
  sorry


end NUMINAMATH_CALUDE_two_tails_one_head_prob_l2004_200456


namespace NUMINAMATH_CALUDE_bd_squared_equals_nine_l2004_200407

theorem bd_squared_equals_nine 
  (h1 : a - b - c + d = 12) 
  (h2 : a + b - c - d = 6) : 
  (b - d)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_bd_squared_equals_nine_l2004_200407


namespace NUMINAMATH_CALUDE_same_sign_range_l2004_200454

theorem same_sign_range (m : ℝ) : (2 - m) * (|m| - 3) > 0 ↔ m ∈ Set.Ioo 2 3 ∪ Set.Iio (-3) := by
  sorry

end NUMINAMATH_CALUDE_same_sign_range_l2004_200454


namespace NUMINAMATH_CALUDE_family_reunion_ratio_l2004_200441

theorem family_reunion_ratio (male_adults female_adults children total_adults total_people : ℕ) : 
  female_adults = male_adults + 50 →
  male_adults = 100 →
  total_adults = male_adults + female_adults →
  total_people = 750 →
  total_people = total_adults + children →
  (children : ℚ) / total_adults = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_family_reunion_ratio_l2004_200441


namespace NUMINAMATH_CALUDE_atomic_numbers_descending_l2004_200402

/-- Atomic number of Chlorine -/
def atomic_number_Cl : ℕ := 17

/-- Atomic number of Oxygen -/
def atomic_number_O : ℕ := 8

/-- Atomic number of Lithium -/
def atomic_number_Li : ℕ := 3

/-- Theorem stating that the atomic numbers of Cl, O, and Li are in descending order -/
theorem atomic_numbers_descending :
  atomic_number_Cl > atomic_number_O ∧ atomic_number_O > atomic_number_Li :=
sorry

end NUMINAMATH_CALUDE_atomic_numbers_descending_l2004_200402


namespace NUMINAMATH_CALUDE_pentomino_tiling_l2004_200423

-- Define the pentomino types
inductive Pentomino
| UShaped
| CrossShaped

-- Define a function to check if a rectangle can be tiled
def canTile (width height : ℕ) : Prop :=
  ∃ (arrangement : ℕ → ℕ → Pentomino), 
    (∀ x y, x < width ∧ y < height → ∃ (px py : ℕ) (p : Pentomino), 
      arrangement px py = p ∧ 
      (px ≤ x ∧ x < px + 5) ∧ 
      (py ≤ y ∧ y < py + 5))

-- State the theorem
theorem pentomino_tiling (n : ℕ) :
  n > 1 ∧ canTile 15 n ↔ n ≠ 2 ∧ n ≠ 4 ∧ n ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_pentomino_tiling_l2004_200423


namespace NUMINAMATH_CALUDE_inequality_of_positive_reals_l2004_200449

theorem inequality_of_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  b * c / a + a * c / b + a * b / c ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_reals_l2004_200449


namespace NUMINAMATH_CALUDE_bug_crawl_distance_l2004_200434

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ

/-- Calculates the shortest distance between two points on the surface of a cone -/
noncomputable def shortestDistance (c : Cone) (p1 p2 : ConePoint) : ℝ :=
  sorry

theorem bug_crawl_distance (c : Cone) (p1 p2 : ConePoint) :
  c.baseRadius = 500 →
  c.height = 250 * Real.sqrt 3 →
  p1.distanceFromVertex = 100 →
  p2.distanceFromVertex = 300 * Real.sqrt 3 →
  shortestDistance c p1 p2 = 100 * Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_bug_crawl_distance_l2004_200434


namespace NUMINAMATH_CALUDE_hotel_expenditure_l2004_200461

theorem hotel_expenditure (n : ℕ) (m : ℕ) (individual_cost : ℕ) (extra_cost : ℕ) 
  (h1 : n = 9)
  (h2 : m = 8)
  (h3 : individual_cost = 12)
  (h4 : extra_cost = 8) :
  m * individual_cost + (individual_cost + (m * individual_cost + individual_cost + extra_cost) / n) = 117 := by
  sorry

end NUMINAMATH_CALUDE_hotel_expenditure_l2004_200461


namespace NUMINAMATH_CALUDE_coefficient_sum_equals_15625_l2004_200492

theorem coefficient_sum_equals_15625 (b₆ b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (2*x + 3)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 15625 := by
sorry

end NUMINAMATH_CALUDE_coefficient_sum_equals_15625_l2004_200492


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l2004_200484

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := π / 2
  let φ : ℝ := π / 3
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (0, 2 * Real.sqrt 3, 2) := by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l2004_200484


namespace NUMINAMATH_CALUDE_stock_percentage_example_l2004_200496

/-- The percentage of stock that yields a given income from a given investment --/
def stock_percentage (income : ℚ) (investment : ℚ) : ℚ :=
  (income * 100) / investment

/-- Theorem: The stock percentage for an income of 15000 and investment of 37500 is 40% --/
theorem stock_percentage_example : stock_percentage 15000 37500 = 40 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_example_l2004_200496


namespace NUMINAMATH_CALUDE_age_ratio_proof_l2004_200450

/-- Given three people a, b, and c, with the following conditions:
  1. a is two years older than b
  2. The total of the ages of a, b, and c is 72
  3. b is 28 years old
Prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 72 →
  b = 28 →
  b / c = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l2004_200450


namespace NUMINAMATH_CALUDE_snack_sales_averages_l2004_200414

/-- Represents the snack sales data for a special weekend event -/
structure EventData where
  tickets : ℕ
  crackers : ℕ
  crackerPrice : ℚ
  beverages : ℕ
  beveragePrice : ℚ
  chocolates : ℕ
  chocolatePrice : ℚ

/-- Calculates the total snack sales for an event -/
def totalSales (e : EventData) : ℚ :=
  e.crackers * e.crackerPrice + e.beverages * e.beveragePrice + e.chocolates * e.chocolatePrice

/-- Calculates the average snack sales per ticket for an event -/
def averageSales (e : EventData) : ℚ :=
  totalSales e / e.tickets

/-- Theorem stating the average snack sales for each event and the combined average -/
theorem snack_sales_averages 
  (valentines : EventData)
  (stPatricks : EventData)
  (christmas : EventData)
  (h1 : valentines = ⟨10, 4, 11/5, 6, 3/2, 7, 6/5⟩)
  (h2 : stPatricks = ⟨8, 3, 2, 5, 25/20, 8, 1⟩)
  (h3 : christmas = ⟨9, 6, 43/20, 4, 17/12, 9, 11/10⟩) :
  averageSales valentines = 131/50 ∧
  averageSales stPatricks = 253/100 ∧
  averageSales christmas = 79/25 ∧
  (totalSales valentines + totalSales stPatricks + totalSales christmas) / 
  (valentines.tickets + stPatricks.tickets + christmas.tickets) = 139/50 := by
  sorry

end NUMINAMATH_CALUDE_snack_sales_averages_l2004_200414


namespace NUMINAMATH_CALUDE_vet_count_l2004_200453

theorem vet_count (total : ℕ) 
  (puppy_kibble : ℕ → ℕ) (yummy_kibble : ℕ → ℕ)
  (h1 : puppy_kibble total = (20 * total) / 100)
  (h2 : yummy_kibble total = (30 * total) / 100)
  (h3 : yummy_kibble total - puppy_kibble total = 100) :
  total = 1000 := by
sorry

end NUMINAMATH_CALUDE_vet_count_l2004_200453


namespace NUMINAMATH_CALUDE_product_equals_zero_l2004_200451

theorem product_equals_zero (n : ℤ) (h : n = 3) :
  (n - 3) * (n - 2) * (n - 1) * n * (n + 1) * (n + 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l2004_200451


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l2004_200437

theorem existence_of_special_integers : ∃ (a b c : ℤ), 
  (a > 2011) ∧ (b > 2011) ∧ (c > 2011) ∧
  ∃ (n : ℕ), (((a + Real.sqrt b)^c : ℝ) / 10000 - n : ℝ) = 0.20102011 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l2004_200437


namespace NUMINAMATH_CALUDE_part_one_part_two_l2004_200498

-- Part 1
theorem part_one (f h : ℝ → ℝ) (m : ℝ) :
  (∀ x > 1, f x = x^2 - m * Real.log x) →
  (∀ x > 1, h x = x^2 - x) →
  (∀ x > 1, f x ≥ h x) →
  m ≤ Real.exp 1 :=
sorry

-- Part 2
theorem part_two (f h k : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x^2 - 2 * Real.log x) →
  (∀ x, h x = x^2 - x + a) →
  (∀ x, k x = f x - h x) →
  (∃ x y, x ∈ Set.Icc 1 3 ∧ y ∈ Set.Icc 1 3 ∧ x < y ∧ k x = 0 ∧ k y = 0 ∧ 
    ∀ z ∈ Set.Icc 1 3, k z = 0 → (z = x ∨ z = y)) →
  2 - 2 * Real.log 2 < a ∧ a ≤ 3 - 2 * Real.log 3 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2004_200498


namespace NUMINAMATH_CALUDE_parallelogram_missing_vertex_l2004_200464

/-- A parallelogram in a 2D coordinate system -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- The area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Theorem: Given a parallelogram with three known vertices and a known area,
    prove that the fourth vertex has specific coordinates -/
theorem parallelogram_missing_vertex 
  (p : Parallelogram)
  (h1 : p.v1 = (4, 4))
  (h2 : p.v3 = (5, 9))
  (h3 : p.v4 = (8, 9))
  (h4 : area p = 5) :
  p.v2 = (3, 4) := by sorry

end NUMINAMATH_CALUDE_parallelogram_missing_vertex_l2004_200464


namespace NUMINAMATH_CALUDE_base10_to_base7_5423_l2004_200440

/-- Converts a base 10 number to base 7 --/
def toBase7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: toBase7 (n / 7)

/-- Converts a list of digits in base 7 to a natural number --/
def fromBase7 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 7 * acc) 0

theorem base10_to_base7_5423 :
  toBase7 5423 = [5, 4, 5, 1, 2] ∧ fromBase7 [5, 4, 5, 1, 2] = 5423 := by sorry

end NUMINAMATH_CALUDE_base10_to_base7_5423_l2004_200440


namespace NUMINAMATH_CALUDE_base_conversion_problem_l2004_200428

theorem base_conversion_problem :
  ∀ (a b : ℕ),
  (a < 10 ∧ b < 10) →
  (5 * 7^2 + 2 * 7 + 5 = 3 * 10 * a + b) →
  (a * b) / 15 = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l2004_200428


namespace NUMINAMATH_CALUDE_gingerbread_theorem_l2004_200488

def gingerbread_problem (red_hats blue_boots both : ℕ) : Prop :=
  let total := red_hats + blue_boots - both
  (red_hats : ℚ) / total * 100 = 50

theorem gingerbread_theorem :
  gingerbread_problem 6 9 3 := by
  sorry

end NUMINAMATH_CALUDE_gingerbread_theorem_l2004_200488


namespace NUMINAMATH_CALUDE_park_trees_theorem_l2004_200422

/-- The number of dogwood trees remaining in the park after a day's work -/
def remaining_trees (first_part : ℝ) (second_part : ℝ) (third_part : ℝ) 
  (trees_cut : ℝ) (trees_planted : ℝ) : ℝ :=
  first_part + second_part + third_part - trees_cut + trees_planted

/-- Theorem stating the number of remaining trees after the day's work -/
theorem park_trees_theorem (first_part : ℝ) (second_part : ℝ) (third_part : ℝ) 
  (trees_cut : ℝ) (trees_planted : ℝ) :
  first_part = 5.0 →
  second_part = 4.0 →
  third_part = 6.0 →
  trees_cut = 7.0 →
  trees_planted = 3.0 →
  remaining_trees first_part second_part third_part trees_cut trees_planted = 11.0 :=
by
  sorry

#eval remaining_trees 5.0 4.0 6.0 7.0 3.0

end NUMINAMATH_CALUDE_park_trees_theorem_l2004_200422


namespace NUMINAMATH_CALUDE_bicycling_problem_l2004_200435

/-- The number of days after which the condition is satisfied -/
def days : ℕ := 12

/-- The total distance between points A and B in kilometers -/
def total_distance : ℕ := 600

/-- The distance person A travels per day in kilometers -/
def person_a_speed : ℕ := 40

/-- The effective daily distance person B travels in kilometers -/
def person_b_speed : ℕ := 30

/-- The remaining distance for person A after the given number of days -/
def remaining_distance_a : ℕ := total_distance - person_a_speed * days

/-- The remaining distance for person B after the given number of days -/
def remaining_distance_b : ℕ := total_distance - person_b_speed * days

theorem bicycling_problem :
  remaining_distance_b = 2 * remaining_distance_a :=
sorry

end NUMINAMATH_CALUDE_bicycling_problem_l2004_200435


namespace NUMINAMATH_CALUDE_probability_no_brown_is_51_310_l2004_200460

def total_balls : ℕ := 32
def brown_balls : ℕ := 14
def non_brown_balls : ℕ := total_balls - brown_balls

def probability_no_brown : ℚ := (Nat.choose non_brown_balls 3 : ℚ) / (Nat.choose total_balls 3 : ℚ)

theorem probability_no_brown_is_51_310 : probability_no_brown = 51 / 310 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_brown_is_51_310_l2004_200460


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2004_200425

theorem inequality_system_solution (m : ℝ) : 
  (∃ x : ℤ, x > 2*m ∧ x ≥ m - 3 ∧ (∀ y : ℤ, y > 2*m ∧ y ≥ m - 3 → y ≥ x) ∧ x = 1) 
  ↔ 0 ≤ m ∧ m < 1/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2004_200425


namespace NUMINAMATH_CALUDE_next_price_reduction_l2004_200400

def price_sequence (n : ℕ) : ℚ :=
  (1024 : ℚ) * (5/8 : ℚ)^n

theorem next_price_reduction : price_sequence 4 = 156.25 := by
  sorry

end NUMINAMATH_CALUDE_next_price_reduction_l2004_200400


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2004_200403

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (9 * z) / (3 * x + 2 * y) + (9 * x) / (2 * y + 3 * z) + (4 * y) / (2 * x + z) ≥ 9 / 2 :=
by sorry

theorem min_value_achievable :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    (9 * z) / (3 * x + 2 * y) + (9 * x) / (2 * y + 3 * z) + (4 * y) / (2 * x + z) = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2004_200403


namespace NUMINAMATH_CALUDE_smallest_value_complex_sum_l2004_200421

theorem smallest_value_complex_sum (a b c : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_omega_cube : ω^3 = 1)
  (h_omega_neq_one : ω ≠ 1) :
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ 
  (∀ (x y z : ℤ) (h_xyz_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z),
    Complex.abs (↑x + ↑y * ω + ↑z * ω^2) ≥ m) ∧
  (∃ (p q r : ℤ) (h_pqr_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r),
    Complex.abs (↑p + ↑q * ω + ↑r * ω^2) = m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_complex_sum_l2004_200421


namespace NUMINAMATH_CALUDE_locus_of_point_P_l2004_200474

-- Define the 2D plane
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable [Fact (finrank ℝ V = 2)]

-- Define points A, B, and P
variable (A B P : V)

-- Define the distance function
def dist (x y : V) : ℝ := ‖x - y‖

-- Theorem statement
theorem locus_of_point_P (h1 : dist A B = 3) (h2 : dist A P + dist B P = 3) :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B :=
sorry

end NUMINAMATH_CALUDE_locus_of_point_P_l2004_200474


namespace NUMINAMATH_CALUDE_quadratic_radical_condition_l2004_200467

theorem quadratic_radical_condition (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 3) ↔ x ≥ -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_radical_condition_l2004_200467


namespace NUMINAMATH_CALUDE_expression_evaluation_l2004_200432

theorem expression_evaluation (a b : ℤ) (ha : a = 4) (hb : b = -2) :
  -a - b^4 + a*b = -28 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2004_200432


namespace NUMINAMATH_CALUDE_beef_weight_before_processing_l2004_200486

theorem beef_weight_before_processing (weight_after : ℝ) (percent_lost : ℝ) : 
  weight_after = 240 ∧ percent_lost = 40 → 
  weight_after / (1 - percent_lost / 100) = 400 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_before_processing_l2004_200486


namespace NUMINAMATH_CALUDE_false_proposition_l2004_200491

-- Define proposition p
def p : Prop := ∃ x : ℝ, (Real.cos x)^2 - (Real.sin x)^2 = 7

-- Define proposition q
def q : Prop := ∀ x : ℝ, Real.exp x > 0

-- Theorem statement
theorem false_proposition : ¬(¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_false_proposition_l2004_200491


namespace NUMINAMATH_CALUDE_square_to_rectangle_area_increase_l2004_200424

theorem square_to_rectangle_area_increase (s : ℝ) (h : s > 0) :
  let original_area := s^2
  let new_length := 1.3 * s
  let new_width := 1.2 * s
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.56 := by
sorry

end NUMINAMATH_CALUDE_square_to_rectangle_area_increase_l2004_200424


namespace NUMINAMATH_CALUDE_min_a_value_l2004_200481

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x - 1 - a * Real.log x
def g (x : ℝ) : ℝ := x / Real.exp (x - 1)

-- State the theorem
theorem min_a_value (a : ℝ) :
  (a < 0) →
  (∀ x₁ x₂ : ℝ, 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 4 →
    (f a x₁ - f a x₂) / (g x₁ - g x₂) > -1 / (g x₁ * g x₂)) →
  a ≥ 3 - 2 / 3 * Real.exp 2 :=
sorry

end

end NUMINAMATH_CALUDE_min_a_value_l2004_200481


namespace NUMINAMATH_CALUDE_equation_transformation_l2004_200405

theorem equation_transformation (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : 3 * a = 4 * b) :
  a / 4 = b / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l2004_200405


namespace NUMINAMATH_CALUDE_investment_rate_calculation_l2004_200476

theorem investment_rate_calculation 
  (total_investment : ℝ) 
  (first_rate : ℝ) 
  (first_amount : ℝ) 
  (total_interest : ℝ) :
  total_investment = 10000 →
  first_rate = 0.06 →
  first_amount = 7200 →
  total_interest = 684 →
  let second_amount := total_investment - first_amount
  let first_interest := first_amount * first_rate
  let second_interest := total_interest - first_interest
  let second_rate := second_interest / second_amount
  second_rate = 0.09 := by sorry

end NUMINAMATH_CALUDE_investment_rate_calculation_l2004_200476


namespace NUMINAMATH_CALUDE_mary_earnings_l2004_200412

/-- Mary's earnings problem -/
theorem mary_earnings (earnings_per_home : ℕ) (homes_cleaned : ℕ) : 
  earnings_per_home = 46 → homes_cleaned = 6 → earnings_per_home * homes_cleaned = 276 := by
  sorry

end NUMINAMATH_CALUDE_mary_earnings_l2004_200412


namespace NUMINAMATH_CALUDE_greatest_value_l2004_200415

theorem greatest_value (a b : ℝ) (ha : a = 2) (hb : b = 5) :
  let expr1 := a / b
  let expr2 := b / a
  let expr3 := a - b
  let expr4 := b - a
  let expr5 := (1 / 2) * a
  (expr4 ≥ expr1) ∧ (expr4 ≥ expr2) ∧ (expr4 ≥ expr3) ∧ (expr4 ≥ expr5) :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_l2004_200415


namespace NUMINAMATH_CALUDE_alpha_value_l2004_200462

theorem alpha_value (α : Real) 
  (h1 : (1 - 4 * Real.sin α) / Real.tan α = Real.sqrt 3)
  (h2 : α ∈ Set.Ioo 0 (Real.pi / 2)) :
  α = Real.pi / 18 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l2004_200462


namespace NUMINAMATH_CALUDE_min_value_problem_l2004_200417

theorem min_value_problem (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) (hmn : m + n = 1) :
  m^2 / (m + 2) + n^2 / (n + 1) ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2004_200417


namespace NUMINAMATH_CALUDE_first_player_min_score_l2004_200401

/-- Represents a game state with remaining numbers -/
def GameState := List Nat

/-- Removes a list of numbers from the game state -/
def removeNumbers (state : GameState) (toRemove : List Nat) : GameState :=
  state.filter (λ n => n ∉ toRemove)

/-- Calculates the score based on the two remaining numbers -/
def calculateScore (state : GameState) : Nat :=
  if state.length = 2 then
    state.maximum.getD 0 - state.minimum.getD 0
  else
    0

/-- Represents a player's strategy -/
def Strategy := GameState → List Nat

/-- Simulates a game given two strategies -/
def playGame (player1 : Strategy) (player2 : Strategy) : Nat :=
  let initialState : GameState := List.range 101
  let finalState := (List.range 11).foldl
    (λ state round =>
      let state' := removeNumbers state (player1 state)
      removeNumbers state' (player2 state'))
    initialState
  calculateScore finalState

/-- Theorem: The first player can always ensure a score of at least 52 -/
theorem first_player_min_score :
  ∃ (player1 : Strategy), ∀ (player2 : Strategy), playGame player1 player2 ≥ 52 := by
  sorry


end NUMINAMATH_CALUDE_first_player_min_score_l2004_200401


namespace NUMINAMATH_CALUDE_pet_store_puppies_l2004_200475

def initial_puppies (sold : ℕ) (puppies_per_cage : ℕ) (cages_used : ℕ) : ℕ :=
  sold + (puppies_per_cage * cages_used)

theorem pet_store_puppies :
  initial_puppies 7 2 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l2004_200475


namespace NUMINAMATH_CALUDE_division_cannot_be_operation_l2004_200472

def P : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

theorem division_cannot_be_operation :
  ¬(∀ a b : ℤ, a ∈ P → b ∈ P → (a / b) ∈ P) :=
by
  sorry

end NUMINAMATH_CALUDE_division_cannot_be_operation_l2004_200472


namespace NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l2004_200443

theorem least_positive_integer_with_given_remainders : ∃ b : ℕ, 
  b > 0 ∧
  b % 6 = 5 ∧
  b % 7 = 6 ∧
  b % 8 = 7 ∧
  b % 9 = 8 ∧
  (∀ x : ℕ, x > 0 ∧ x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7 ∧ x % 9 = 8 → x ≥ b) ∧
  b = 503 :=
sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l2004_200443


namespace NUMINAMATH_CALUDE_fraction_problem_l2004_200416

theorem fraction_problem (x : ℚ) : 
  x < 0.4 ∧ x * 180 = 48 → x = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2004_200416


namespace NUMINAMATH_CALUDE_product_of_square_roots_l2004_200420

theorem product_of_square_roots (x : ℝ) (h : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (8 * x) = 60 * x * Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l2004_200420


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l2004_200483

/-- Two congruent squares with side length 20 overlap to form a 20 by 40 rectangle.
    The shaded area is the overlap of the two squares. -/
theorem shaded_area_percentage (square_side : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) :
  square_side = 20 →
  rectangle_width = 20 →
  rectangle_length = 40 →
  (square_side * square_side) / (rectangle_width * rectangle_length) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l2004_200483


namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l2004_200458

theorem multiplication_table_odd_fraction :
  let n : ℕ := 15
  let total_products : ℕ := (n + 1) * (n + 1)
  let odd_numbers : ℕ := (n + 1) / 2
  let odd_products : ℕ := odd_numbers * odd_numbers
  odd_products / total_products = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l2004_200458


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2004_200469

theorem division_remainder_problem (L S R : ℕ) : 
  L - S = 1375 →
  L = 1632 →
  L = 6 * S + R →
  R < S →
  R = 90 :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2004_200469


namespace NUMINAMATH_CALUDE_polyhedron_volume_l2004_200470

/-- The volume of a polyhedron formed by a cube and a tetrahedron -/
theorem polyhedron_volume (cube_side : ℝ) (tetra_base_area : ℝ) (tetra_height : ℝ) :
  cube_side = 2 →
  tetra_base_area = 2 →
  tetra_height = 2 →
  cube_side ^ 3 + (1/3) * tetra_base_area * tetra_height = 28/3 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_l2004_200470


namespace NUMINAMATH_CALUDE_ellipse_theorem_l2004_200429

/-- Ellipse with focus at (-√3, 0) and point (1, y) on it --/
structure Ellipse where
  a : ℝ
  b : ℝ
  y : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_y_pos : y > 0
  h_eq : 1 / a^2 + y^2 / b^2 = 1
  h_focus : -Real.sqrt 3 = -Real.sqrt (a^2 - b^2)
  h_area : 1/2 * Real.sqrt 3 * y = 3/4

/-- The main theorem --/
theorem ellipse_theorem (e : Ellipse) :
  (∃ (k : ℝ), (∀ (x y : ℝ),
    (x^2 / 4 + y^2 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
    (y = k * (x - 2) →
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        x₁^2 / 4 + y₁^2 = 1 ∧
        x₂^2 / 4 + y₂^2 = 1 ∧
        y₁ = k * (x₁ - 2) ∧
        y₂ = k * (x₂ - 2) ∧
        ∃ (t₁ t₂ : ℝ),
          t₁^2 + t₂^2 = 1 ∧
          t₁ = Real.sqrt 5 / 5 * (2 + x₂) ∧
          t₂ = Real.sqrt 5 / 5 * (y₂)))) ∧
  (k = 1/2 ∨ k = -1/2) := by sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l2004_200429


namespace NUMINAMATH_CALUDE_alex_class_size_l2004_200447

/-- Represents a student's ranking in a class -/
structure StudentRanking where
  best : Nat
  worst : Nat

/-- Calculates the total number of students in a class given a student's ranking -/
def totalStudents (ranking : StudentRanking) : Nat :=
  ranking.best + ranking.worst - 1

/-- Theorem: If a student is ranked 20th best and 20th worst, there are 39 students in the class -/
theorem alex_class_size (ranking : StudentRanking) 
  (h1 : ranking.best = 20) 
  (h2 : ranking.worst = 20) : 
  totalStudents ranking = 39 := by
  sorry

end NUMINAMATH_CALUDE_alex_class_size_l2004_200447


namespace NUMINAMATH_CALUDE_triangle_inequality_l2004_200489

/-- 
Given a triangle ABC with circumradius R = 1 and area S = 1/4, 
prove that sqrt(a) + sqrt(b) + sqrt(c) < 1/a + 1/b + 1/c, 
where a, b, and c are the side lengths of the triangle.
-/
theorem triangle_inequality (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_circumradius : (a * b * c) / (4 * (1/4)) = 1) 
  (h_area : (1/4) > 0) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c < 1/a + 1/b + 1/c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2004_200489


namespace NUMINAMATH_CALUDE_minimum_h_22_l2004_200438

def IsTenuous (h : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, h x + h y > (y : ℤ)^2

def SumUpTo30 (h : ℕ+ → ℤ) : ℤ :=
  (Finset.range 30).sum (fun i => h ⟨i + 1, Nat.succ_pos i⟩)

theorem minimum_h_22 (h : ℕ+ → ℤ) (h_tenuous : IsTenuous h) 
    (h_min : ∀ g : ℕ+ → ℤ, IsTenuous g → SumUpTo30 h ≤ SumUpTo30 g) :
    h ⟨22, by norm_num⟩ ≥ 357 := by
  sorry

end NUMINAMATH_CALUDE_minimum_h_22_l2004_200438
