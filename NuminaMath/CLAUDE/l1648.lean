import Mathlib

namespace NUMINAMATH_CALUDE_flag_design_count_l1648_164820

/-- The number of school colors -/
def num_colors : ℕ := 3

/-- The number of horizontal stripes on the flag -/
def num_horizontal_stripes : ℕ := 3

/-- The number of options for the vertical stripe (3 colors + no stripe) -/
def vertical_stripe_options : ℕ := num_colors + 1

/-- The total number of possible flag designs -/
def total_flag_designs : ℕ := num_colors ^ num_horizontal_stripes * vertical_stripe_options

theorem flag_design_count :
  total_flag_designs = 108 :=
sorry

end NUMINAMATH_CALUDE_flag_design_count_l1648_164820


namespace NUMINAMATH_CALUDE_cube_face_planes_divide_space_l1648_164862

-- Define a cube in 3D space
def Cube := Set (ℝ × ℝ × ℝ)

-- Define the planes that each face of the cube lies on
def FacePlanes (c : Cube) := Set (Set (ℝ × ℝ × ℝ))

-- Define a function that counts the number of regions created by the face planes
def countRegions (c : Cube) : ℕ := sorry

-- Theorem stating that the face planes of a cube divide space into 27 regions
theorem cube_face_planes_divide_space (c : Cube) : 
  countRegions c = 27 := by sorry

end NUMINAMATH_CALUDE_cube_face_planes_divide_space_l1648_164862


namespace NUMINAMATH_CALUDE_abs_difference_range_l1648_164855

theorem abs_difference_range (t : ℝ) : let f := λ x : ℝ => Real.sin x + Real.cos x
                                        let g := λ x : ℝ => 2 * Real.cos x
                                        0 ≤ |f t - g t| ∧ |f t - g t| ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_difference_range_l1648_164855


namespace NUMINAMATH_CALUDE_carpet_coverage_percentage_l1648_164804

theorem carpet_coverage_percentage (carpet_length : ℝ) (carpet_width : ℝ) (room_area : ℝ) :
  carpet_length = 4 →
  carpet_width = 9 →
  room_area = 60 →
  (carpet_length * carpet_width) / room_area * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_carpet_coverage_percentage_l1648_164804


namespace NUMINAMATH_CALUDE_modular_arithmetic_problems_l1648_164802

theorem modular_arithmetic_problems :
  (∃ k : ℕ, 19^10 = 6 * k + 1) ∧
  (∃ m : ℕ, 19^14 = 70 * m + 11) ∧
  (∃ n : ℕ, 17^9 = 48 * n + 17) ∧
  (∃ p : ℕ, 14^(14^14) = 100 * p + 36) := by
sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problems_l1648_164802


namespace NUMINAMATH_CALUDE_sin_sum_equals_half_l1648_164863

theorem sin_sum_equals_half : 
  Real.sin (163 * π / 180) * Real.sin (223 * π / 180) + 
  Real.sin (253 * π / 180) * Real.sin (313 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_equals_half_l1648_164863


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l1648_164837

/-- Theorem: For a point P with coordinates (x, -6), if the distance from the x-axis to P
    is half the distance from the y-axis to P, then the distance from the y-axis to P is 12 units. -/
theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -6)
  let distance_to_x_axis := |P.2|
  let distance_to_y_axis := |P.1|
  distance_to_x_axis = (1/2 : ℝ) * distance_to_y_axis →
  distance_to_y_axis = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l1648_164837


namespace NUMINAMATH_CALUDE_A_union_B_eq_A_l1648_164890

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | (x + 1) * (x - 4) < 0}
def B : Set ℝ := {x : ℝ | Real.log x < 1}

-- State the theorem
theorem A_union_B_eq_A : A ∪ B = A := by sorry

end NUMINAMATH_CALUDE_A_union_B_eq_A_l1648_164890


namespace NUMINAMATH_CALUDE_fixed_cost_calculation_publishing_company_fixed_cost_l1648_164872

theorem fixed_cost_calculation (marketing_cost : ℕ) (selling_price : ℕ) (break_even_quantity : ℕ) : ℕ :=
  let net_revenue_per_book := selling_price - marketing_cost
  let fixed_cost := net_revenue_per_book * break_even_quantity
  fixed_cost

theorem publishing_company_fixed_cost :
  fixed_cost_calculation 4 9 10000 = 50000 := by
  sorry

end NUMINAMATH_CALUDE_fixed_cost_calculation_publishing_company_fixed_cost_l1648_164872


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1648_164875

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3/2 < 0) ↔ k ∈ Set.Ioc (-12) 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1648_164875


namespace NUMINAMATH_CALUDE_lcm_of_3_4_6_15_l1648_164884

def numbers : List ℕ := [3, 4, 6, 15]

theorem lcm_of_3_4_6_15 : Nat.lcm (Nat.lcm (Nat.lcm 3 4) 6) 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_3_4_6_15_l1648_164884


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1648_164843

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : a > 0) (k : b > 0) 
  (eq : (Real.sin θ ^ 6 / a ^ 2) + (Real.cos θ ^ 6 / b ^ 2) = 1 / (a + b)) :
  (Real.sin θ ^ 12 / a ^ 5) + (Real.cos θ ^ 12 / b ^ 5) = 1 / (a + b) ^ 5 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1648_164843


namespace NUMINAMATH_CALUDE_function_equivalence_l1648_164857

theorem function_equivalence (x : ℝ) (h : x ≠ 0) :
  (2 * x + 3) / x = 2 + 3 / x := by sorry

end NUMINAMATH_CALUDE_function_equivalence_l1648_164857


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l1648_164831

theorem min_value_theorem (a : ℝ) (h : a > 2) : a + 4 / (a - 2) ≥ 6 :=
sorry

theorem equality_condition (a : ℝ) (h : a > 2) : 
  ∃ a₀ > 2, a₀ + 4 / (a₀ - 2) = 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l1648_164831


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l1648_164825

theorem root_difference_implies_k_value (k : ℝ) : 
  (∀ r s : ℝ, r^2 + k*r + 12 = 0 ∧ s^2 + k*s + 12 = 0 → 
    (r+3)^2 - k*(r+3) + 12 = 0 ∧ (s+3)^2 - k*(s+3) + 12 = 0) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l1648_164825


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l1648_164833

theorem smallest_number_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 7 = 2 ∧
  n % 11 = 2 ∧
  n % 13 = 2 ∧
  n % 17 = 3 ∧
  n % 23 = 0 ∧
  n % 5 = 0 ∧
  (∀ m : ℕ, m > 0 ∧
    m % 7 = 2 ∧
    m % 11 = 2 ∧
    m % 13 = 2 ∧
    m % 17 = 3 ∧
    m % 23 = 0 ∧
    m % 5 = 0 → m ≥ n) ∧
  n = 391410 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l1648_164833


namespace NUMINAMATH_CALUDE_bouquet_lilies_percentage_l1648_164839

theorem bouquet_lilies_percentage (F : ℚ) (F_pos : F > 0) : 
  let purple_flowers := (7 / 10) * F
  let purple_tulips := (1 / 2) * purple_flowers
  let yellow_flowers := F - purple_flowers
  let yellow_lilies := (2 / 3) * yellow_flowers
  let total_lilies := (purple_flowers - purple_tulips) + yellow_lilies
  (total_lilies / F) * 100 = 55 := by sorry

end NUMINAMATH_CALUDE_bouquet_lilies_percentage_l1648_164839


namespace NUMINAMATH_CALUDE_omega_value_l1648_164850

/-- Given a function f(x) = 3sin(ωx) - √3cos(ωx) where ω > 0 and x ∈ ℝ,
    if f(x) is monotonically increasing in (-ω, 2ω) and
    symmetric about x = -ω, then ω = √(3π)/3 -/
theorem omega_value (ω : ℝ) (h_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)
  (∀ x ∈ Set.Ioo (-ω) (2*ω), StrictMonoOn f (Set.Ioo (-ω) (2*ω))) →
  (∀ x, f (x - ω) = -f (-x - ω)) →
  ω = Real.sqrt (3 * Real.pi) / 3 := by
sorry

end NUMINAMATH_CALUDE_omega_value_l1648_164850


namespace NUMINAMATH_CALUDE_company_employee_increase_l1648_164838

theorem company_employee_increase (jan_employees dec_employees : ℝ) 
  (h_jan : jan_employees = 426.09)
  (h_dec : dec_employees = 490) :
  let increase := dec_employees - jan_employees
  let percentage_increase := (increase / jan_employees) * 100
  ∃ ε > 0, |percentage_increase - 15| < ε :=
by sorry

end NUMINAMATH_CALUDE_company_employee_increase_l1648_164838


namespace NUMINAMATH_CALUDE_election_winner_votes_l1648_164895

theorem election_winner_votes 
  (total_votes : ℕ)
  (winner_percentage : ℚ)
  (vote_difference : ℕ)
  (h1 : winner_percentage = 62 / 100)
  (h2 : vote_difference = 312)
  (h3 : ↑total_votes * winner_percentage - ↑total_votes * (1 - winner_percentage) = vote_difference) :
  ↑total_votes * winner_percentage = 806 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_votes_l1648_164895


namespace NUMINAMATH_CALUDE_divisible_by_24_l1648_164814

theorem divisible_by_24 (n : ℤ) : ∃ k : ℤ, n^4 + 6*n^3 + 11*n^2 + 6*n = 24*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l1648_164814


namespace NUMINAMATH_CALUDE_binary_quadratic_equation_value_l1648_164842

/-- Represents a binary quadratic equation in x and y with a constant m -/
def binary_quadratic_equation (x y m : ℝ) : Prop :=
  x^2 + 2*x*y + 8*y^2 + 14*y + m = 0

/-- Represents that an equation is equivalent to two lines -/
def represents_two_lines (f : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ), ∀ x y m,
    f x y m ↔ (a₁*x + b₁*y + c₁ = 0 ∧ a₂*x + b₂*y + c₂ = 0)

theorem binary_quadratic_equation_value :
  represents_two_lines binary_quadratic_equation → ∃ m, ∀ x y, binary_quadratic_equation x y m :=
by
  sorry

end NUMINAMATH_CALUDE_binary_quadratic_equation_value_l1648_164842


namespace NUMINAMATH_CALUDE_negation_of_exists_equals_sin_l1648_164882

theorem negation_of_exists_equals_sin (x : ℝ) : 
  (¬ ∃ x : ℝ, x = Real.sin x) ↔ (∀ x : ℝ, x ≠ Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_equals_sin_l1648_164882


namespace NUMINAMATH_CALUDE_peach_problem_l1648_164810

theorem peach_problem (steven jake jill hanna lucy : ℕ) : 
  steven = 19 →
  jake = steven - 12 →
  jake = 3 * jill →
  hanna = jake + 3 →
  lucy = hanna + 5 →
  lucy + jill = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_peach_problem_l1648_164810


namespace NUMINAMATH_CALUDE_spending_fraction_is_three_fourths_l1648_164892

/-- Represents a person's monthly savings and spending habits -/
structure SavingsHabit where
  monthly_salary : ℝ
  savings_fraction : ℝ
  spending_fraction : ℝ
  savings_fraction_nonneg : 0 ≤ savings_fraction
  spending_fraction_nonneg : 0 ≤ spending_fraction
  fractions_sum_to_one : savings_fraction + spending_fraction = 1

/-- The theorem stating that if yearly savings are 4 times monthly spending, 
    then the spending fraction is 3/4 -/
theorem spending_fraction_is_three_fourths 
  (habit : SavingsHabit) 
  (yearly_savings_eq_four_times_monthly_spending : 
    12 * habit.savings_fraction * habit.monthly_salary = 
    4 * habit.spending_fraction * habit.monthly_salary) :
  habit.spending_fraction = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_spending_fraction_is_three_fourths_l1648_164892


namespace NUMINAMATH_CALUDE_rectangle_two_axes_l1648_164894

-- Define the types of shapes
inductive Shape
  | EquilateralTriangle
  | Parallelogram
  | Rectangle
  | Square

-- Define a function to count axes of symmetry
def axesOfSymmetry (s : Shape) : ℕ :=
  match s with
  | Shape.EquilateralTriangle => 3
  | Shape.Parallelogram => 0
  | Shape.Rectangle => 2
  | Shape.Square => 4

-- Theorem statement
theorem rectangle_two_axes :
  ∀ s : Shape, axesOfSymmetry s = 2 ↔ s = Shape.Rectangle :=
by sorry

end NUMINAMATH_CALUDE_rectangle_two_axes_l1648_164894


namespace NUMINAMATH_CALUDE_solve_equation_l1648_164861

theorem solve_equation (y : ℚ) (h : (1 : ℚ) / 3 + 1 / y = 7 / 12) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1648_164861


namespace NUMINAMATH_CALUDE_video_likes_dislikes_ratio_l1648_164849

theorem video_likes_dislikes_ratio :
  ∀ (initial_dislikes : ℕ),
    (initial_dislikes + 1000 = 2600) →
    (initial_dislikes : ℚ) / 3000 = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_video_likes_dislikes_ratio_l1648_164849


namespace NUMINAMATH_CALUDE_modular_inverse_32_mod_33_l1648_164844

theorem modular_inverse_32_mod_33 : ∃ x : ℕ, x ≤ 32 ∧ (32 * x) % 33 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_32_mod_33_l1648_164844


namespace NUMINAMATH_CALUDE_chessboard_star_property_l1648_164896

/-- Represents a chessboard with stars -/
structure Chessboard (n : ℕ) where
  has_star : Fin n → Fin n → Prop

/-- Represents a set of rows or columns -/
def Subset (n : ℕ) := Fin n → Prop

/-- Checks if a subset is not the entire set -/
def is_proper_subset {n : ℕ} (s : Subset n) : Prop :=
  ∃ i, ¬s i

/-- Checks if a column has exactly one uncrossed star after crossing out rows -/
def column_has_one_star {n : ℕ} (b : Chessboard n) (crossed_rows : Subset n) (j : Fin n) : Prop :=
  ∃! i, ¬crossed_rows i ∧ b.has_star i j

/-- Checks if a row has exactly one uncrossed star after crossing out columns -/
def row_has_one_star {n : ℕ} (b : Chessboard n) (crossed_cols : Subset n) (i : Fin n) : Prop :=
  ∃! j, ¬crossed_cols j ∧ b.has_star i j

/-- The main theorem -/
theorem chessboard_star_property {n : ℕ} (b : Chessboard n) :
  (∀ crossed_rows : Subset n, is_proper_subset crossed_rows →
    ∃ j, column_has_one_star b crossed_rows j) →
  (∀ crossed_cols : Subset n, is_proper_subset crossed_cols →
    ∃ i, row_has_one_star b crossed_cols i) :=
by sorry

end NUMINAMATH_CALUDE_chessboard_star_property_l1648_164896


namespace NUMINAMATH_CALUDE_rectangle_area_l1648_164826

theorem rectangle_area (x : ℝ) (h1 : (x + 5) * (2 * (x + 10)) = 3 * x * (x + 10)) (h2 : x > 0) :
  x * (x + 10) = 200 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1648_164826


namespace NUMINAMATH_CALUDE_equation_two_roots_l1648_164832

-- Define the equation
def equation (x k : ℂ) : Prop :=
  x / (x + 3) + x / (x + 4) = k * x

-- Define the set of valid k values
def valid_k_values : Set ℂ :=
  {0, 7/12, Complex.I, -Complex.I}

-- Theorem statement
theorem equation_two_roots (k : ℂ) :
  (∃! (r₁ r₂ : ℂ), r₁ ≠ r₂ ∧ equation r₁ k ∧ equation r₂ k) ↔ k ∈ valid_k_values :=
sorry

end NUMINAMATH_CALUDE_equation_two_roots_l1648_164832


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1648_164824

/-- Calculates the speed of a train passing a bridge -/
theorem train_speed_calculation (bridge_length : ℝ) (train_length : ℝ) (time : ℝ) : 
  bridge_length = 650 →
  train_length = 200 →
  time = 17 →
  (bridge_length + train_length) / time = 50 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l1648_164824


namespace NUMINAMATH_CALUDE_second_student_wrong_answers_second_student_wrong_answers_value_l1648_164823

theorem second_student_wrong_answers 
  (total_questions : Nat) 
  (hannah_correct : Nat) 
  (hannah_highest_score : Bool) : Nat :=
  let second_student_correct := hannah_correct - 1
  let second_student_wrong := total_questions - second_student_correct
  second_student_wrong

#check second_student_wrong_answers

theorem second_student_wrong_answers_value :
  second_student_wrong_answers 40 39 true = 2 := by sorry

end NUMINAMATH_CALUDE_second_student_wrong_answers_second_student_wrong_answers_value_l1648_164823


namespace NUMINAMATH_CALUDE_equation_roots_and_ellipse_condition_l1648_164836

theorem equation_roots_and_ellipse_condition (m n : ℝ) : 
  ¬(((m^2 - 4*n ≥ 0 ∧ m > 0 ∧ n > 0) → (m > 0 ∧ n > 0 ∧ m ≠ n)) ∧ 
    ((m > 0 ∧ n > 0 ∧ m ≠ n) → (m^2 - 4*n ≥ 0 ∧ m > 0 ∧ n > 0))) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_and_ellipse_condition_l1648_164836


namespace NUMINAMATH_CALUDE_triangle_tangent_segment_length_l1648_164822

/-- Represents a triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point on a line segment -/
structure PointOnSegment where
  segment : ℝ
  position : ℝ

/-- Checks if a point is on the incircle of a triangle -/
def isOnIncircle (t : Triangle) (p : PointOnSegment) : Prop := sorry

/-- Checks if a line segment is tangent to the incircle of a triangle -/
def isTangentToIncircle (t : Triangle) (p1 p2 : PointOnSegment) : Prop := sorry

/-- Main theorem -/
theorem triangle_tangent_segment_length 
  (t : Triangle) 
  (x y : PointOnSegment) :
  t.a = 19 ∧ t.b = 20 ∧ t.c = 21 →
  x.segment = t.a ∧ y.segment = t.c →
  x.position + y.position = t.a →
  isTangentToIncircle t x y →
  x.position = 67 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_segment_length_l1648_164822


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1648_164835

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1648_164835


namespace NUMINAMATH_CALUDE_linear_function_composition_l1648_164876

theorem linear_function_composition (a b : ℝ) :
  (∀ x : ℝ, (3 * ((a * x + b) : ℝ) - 4 : ℝ) = 4 * x + 5) →
  a + b = 13/3 := by
sorry

end NUMINAMATH_CALUDE_linear_function_composition_l1648_164876


namespace NUMINAMATH_CALUDE_phantom_needs_more_money_l1648_164808

/-- The amount of additional money Phantom needs to buy printer inks -/
def additional_money_needed (initial_money : ℚ) 
  (black_price red_price yellow_price blue_price magenta_price cyan_price : ℚ)
  (black_count red_count yellow_count blue_count magenta_count cyan_count : ℕ)
  (tax_rate : ℚ) : ℚ :=
  let subtotal := black_price * black_count + red_price * red_count + 
                  yellow_price * yellow_count + blue_price * blue_count + 
                  magenta_price * magenta_count + cyan_price * cyan_count
  let total_cost := subtotal + subtotal * tax_rate
  total_cost - initial_money

theorem phantom_needs_more_money :
  additional_money_needed 50 12 16 14 17 15 18 3 4 3 2 2 1 (5/100) = 185.20 := by
  sorry

end NUMINAMATH_CALUDE_phantom_needs_more_money_l1648_164808


namespace NUMINAMATH_CALUDE_odd_implies_symmetric_abs_not_vice_versa_l1648_164809

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The graph of |f(x)| is symmetric about the y-axis if |f(-x)| = |f(x)| for all x ∈ ℝ -/
def IsSymmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x, |f (-x)| = |f x|

/-- If f is odd, then |f(x)| is symmetric about the y-axis, but not vice versa -/
theorem odd_implies_symmetric_abs_not_vice_versa :
  (∃ f : ℝ → ℝ, IsOdd f → IsSymmetricAboutYAxis f) ∧
  (∃ g : ℝ → ℝ, IsSymmetricAboutYAxis g ∧ ¬IsOdd g) := by
  sorry

end NUMINAMATH_CALUDE_odd_implies_symmetric_abs_not_vice_versa_l1648_164809


namespace NUMINAMATH_CALUDE_arm_wrestling_tournament_l1648_164819

/-- The number of participants with k points after m rounds in a tournament with 2^n participants -/
def f (n m k : ℕ) : ℕ := 2^(n - m) * Nat.choose m k

theorem arm_wrestling_tournament (n : ℕ) (h1 : n > 7) (h2 : f n 7 5 = 42) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_arm_wrestling_tournament_l1648_164819


namespace NUMINAMATH_CALUDE_difference_of_odd_squares_divisible_by_eight_l1648_164840

theorem difference_of_odd_squares_divisible_by_eight (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 2 * k + 1) 
  (hb : ∃ m : ℤ, b = 2 * m + 1) : 
  ∃ n : ℤ, a ^ 2 - b ^ 2 = 8 * n := by
  sorry

end NUMINAMATH_CALUDE_difference_of_odd_squares_divisible_by_eight_l1648_164840


namespace NUMINAMATH_CALUDE_pool_water_removal_l1648_164851

/-- Calculates the number of gallons of water removed from a rectangular pool when lowering the water level. -/
def gallonsRemoved (length width depth : ℝ) (conversionFactor : ℝ) : ℝ :=
  length * width * depth * conversionFactor

/-- Proves that lowering the water level in a 60 ft by 10 ft rectangular pool by 6 inches removes 2250 gallons of water. -/
theorem pool_water_removal :
  let length : ℝ := 60
  let width : ℝ := 10
  let depth : ℝ := 0.5  -- 6 inches in feet
  let conversionFactor : ℝ := 7.5  -- 1 cubic foot = 7.5 gallons
  gallonsRemoved length width depth conversionFactor = 2250 := by
  sorry

#eval gallonsRemoved 60 10 0.5 7.5

end NUMINAMATH_CALUDE_pool_water_removal_l1648_164851


namespace NUMINAMATH_CALUDE_cube_difference_divisibility_l1648_164829

theorem cube_difference_divisibility (a b : ℤ) :
  ∃ k : ℤ, (2*a + 1)^3 - (2*b + 1)^3 + 8 = 16 * k := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_divisibility_l1648_164829


namespace NUMINAMATH_CALUDE_binomial_10_5_l1648_164864

theorem binomial_10_5 : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_5_l1648_164864


namespace NUMINAMATH_CALUDE_max_value_theorem_l1648_164877

theorem max_value_theorem (a c : ℝ) (ha : 0 < a) (hc : 0 < c) :
  (∀ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + c^2)) ≤ a^2 + c^2) ∧
  (∃ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + c^2)) = a^2 + c^2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1648_164877


namespace NUMINAMATH_CALUDE_circle_properties_l1648_164846

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y + 3)^2 = 4

-- Define a point being inside a circle
def is_inside_circle (x y : ℝ) : Prop := x^2 + (y + 3)^2 < 4

-- Define the line y = x
def line_y_eq_x (x y : ℝ) : Prop := y = x

-- Theorem statement
theorem circle_properties :
  (is_inside_circle 1 (-2)) ∧
  (∀ x y : ℝ, line_y_eq_x x y → ¬ circle_C x y) := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l1648_164846


namespace NUMINAMATH_CALUDE_simplify_radical_product_l1648_164858

theorem simplify_radical_product (x : ℝ) (hx : x > 0) :
  Real.sqrt (45 * x) * Real.sqrt (32 * x) * Real.sqrt (18 * x) * (27 * x) ^ (1/3) = 72 * x ^ (1/3) * Real.sqrt (5 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l1648_164858


namespace NUMINAMATH_CALUDE_max_value_is_320_l1648_164888

def operation := ℝ → ℝ → ℝ

def add : operation := λ x y => x + y
def sub : operation := λ x y => x - y
def mul : operation := λ x y => x * y

def evaluate (op1 op2 op3 op4 : operation) : ℝ :=
  op4 (op3 (op2 (op1 25 1.2) 15) 18.8) 2.3

def is_valid_operation (op : operation) : Prop :=
  op = add ∨ op = sub ∨ op = mul

theorem max_value_is_320 :
  ∀ op1 op2 op3 op4 : operation,
    is_valid_operation op1 →
    is_valid_operation op2 →
    is_valid_operation op3 →
    is_valid_operation op4 →
    evaluate op1 op2 op3 op4 ≤ 320 :=
sorry

end NUMINAMATH_CALUDE_max_value_is_320_l1648_164888


namespace NUMINAMATH_CALUDE_product_of_tans_equals_two_l1648_164848

theorem product_of_tans_equals_two : (1 + Real.tan (1 * π / 180)) * (1 + Real.tan (44 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_tans_equals_two_l1648_164848


namespace NUMINAMATH_CALUDE_fraction_equality_l1648_164866

theorem fraction_equality (P Q : ℤ) (x : ℝ) 
  (h : x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 5) :
  (P / (x + 3 : ℝ)) + (Q / ((x^2 : ℝ) - 5*x)) = 
    ((x^2 : ℝ) - 3*x + 12) / (x^3 + x^2 - 15*x) →
  (Q : ℚ) / P = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1648_164866


namespace NUMINAMATH_CALUDE_fence_poles_needed_l1648_164871

theorem fence_poles_needed (length width pole_distance : ℕ) : 
  length = 90 → width = 40 → pole_distance = 5 →
  (2 * (length + width)) / pole_distance = 52 := by
sorry

end NUMINAMATH_CALUDE_fence_poles_needed_l1648_164871


namespace NUMINAMATH_CALUDE_athlete_arrangements_correct_l1648_164897

/-- The number of ways to arrange 5 athletes on 5 tracks with exactly two matches -/
def athlete_arrangements : ℕ := 20

/-- Proof that the number of arrangements is correct -/
theorem athlete_arrangements_correct : athlete_arrangements = 20 := by
  sorry

end NUMINAMATH_CALUDE_athlete_arrangements_correct_l1648_164897


namespace NUMINAMATH_CALUDE_student_subtraction_problem_l1648_164834

theorem student_subtraction_problem (x : ℝ) (h : x = 155) :
  ∃! y : ℝ, 2 * x - y = 110 ∧ y = 200 := by
sorry

end NUMINAMATH_CALUDE_student_subtraction_problem_l1648_164834


namespace NUMINAMATH_CALUDE_bike_sharing_growth_model_l1648_164847

/-- Represents the bike-sharing company's growth model -/
theorem bike_sharing_growth_model (x : ℝ) :
  let initial_bikes : ℕ := 1000
  let additional_bikes : ℕ := 440
  let growth_factor : ℝ := (1 + x)
  let months : ℕ := 2
  (initial_bikes : ℝ) * growth_factor ^ months = (initial_bikes : ℝ) + additional_bikes :=
by
  sorry

end NUMINAMATH_CALUDE_bike_sharing_growth_model_l1648_164847


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_l1648_164854

theorem lcm_gcf_ratio (a b : ℕ) (ha : a = 210) (hb : b = 462) : 
  Nat.lcm a b / Nat.gcd a b = 55 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_l1648_164854


namespace NUMINAMATH_CALUDE_gcd_1458_1479_l1648_164889

theorem gcd_1458_1479 : Nat.gcd 1458 1479 = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1458_1479_l1648_164889


namespace NUMINAMATH_CALUDE_both_chromatids_contain_N15_l1648_164860

/-- Represents a chromatid -/
structure Chromatid where
  hasN15 : Bool

/-- Represents a chromosome with two chromatids -/
structure Chromosome where
  chromatid1 : Chromatid
  chromatid2 : Chromatid

/-- Represents a cell at the tetraploid stage -/
structure TetraploidCell where
  chromosomes : List Chromosome

/-- Represents the initial condition of progenitor cells -/
def initialProgenitorCell : Bool := true

/-- Represents the culture medium containing N -/
def cultureMediumWithN : Bool := true

/-- Theorem stating that both chromatids contain N15 at the tetraploid stage -/
theorem both_chromatids_contain_N15 (cell : TetraploidCell) 
  (h1 : initialProgenitorCell = true) 
  (h2 : cultureMediumWithN = true) : 
  ∀ c ∈ cell.chromosomes, c.chromatid1.hasN15 ∧ c.chromatid2.hasN15 := by
  sorry


end NUMINAMATH_CALUDE_both_chromatids_contain_N15_l1648_164860


namespace NUMINAMATH_CALUDE_import_tax_percentage_l1648_164881

/-- The import tax percentage calculation problem -/
theorem import_tax_percentage 
  (total_value : ℝ)
  (tax_threshold : ℝ)
  (tax_paid : ℝ)
  (h1 : total_value = 2570)
  (h2 : tax_threshold = 1000)
  (h3 : tax_paid = 109.90) :
  (tax_paid / (total_value - tax_threshold)) * 100 = 7 := by
sorry

end NUMINAMATH_CALUDE_import_tax_percentage_l1648_164881


namespace NUMINAMATH_CALUDE_product_increase_fifteen_times_l1648_164865

theorem product_increase_fifteen_times :
  ∃ (a₁ a₂ a₃ a₄ a₅ : ℕ),
    ((a₁ - 3) * (a₂ - 3) * (a₃ - 3) * (a₄ - 3) * (a₅ - 3) : ℤ) = 
    15 * (a₁ * a₂ * a₃ * a₄ * a₅) := by
  sorry

end NUMINAMATH_CALUDE_product_increase_fifteen_times_l1648_164865


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_seven_mod_nineteen_l1648_164830

theorem largest_four_digit_congruent_to_seven_mod_nineteen :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 19 = 7 → n ≤ 9982 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_seven_mod_nineteen_l1648_164830


namespace NUMINAMATH_CALUDE_unique_solution_l1648_164845

theorem unique_solution : ∃! x : ℚ, x * 8 / 3 - (2 + 3) * 2 = 6 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1648_164845


namespace NUMINAMATH_CALUDE_exists_special_sequence_l1648_164813

/-- An integer sequence satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℤ) (m : ℕ) : Prop :=
  (a 0 = 1) ∧ 
  (a 1 = 337) ∧ 
  (∀ n : ℕ, n ≥ 1 → (a (n+1) * a (n-1) - a n ^ 2) + 3 * (a (n+1) + a (n-1) - 2 * a n) / 4 = m) ∧
  (∀ n : ℕ, ∃ k : ℤ, (a n + 1) * (2 * a n + 1) / 6 = k ^ 2)

/-- Theorem stating the existence of a natural number m and a sequence satisfying the conditions -/
theorem exists_special_sequence : ∃ (m : ℕ) (a : ℕ → ℤ), SpecialSequence a m := by
  sorry

end NUMINAMATH_CALUDE_exists_special_sequence_l1648_164813


namespace NUMINAMATH_CALUDE_jacob_younger_than_michael_l1648_164811

/-- Represents the age difference between Michael and Jacob -/
def age_difference (jacob_age michael_age : ℕ) : ℕ := michael_age - jacob_age

/-- Proves that Jacob is 14 years younger than Michael given the problem conditions -/
theorem jacob_younger_than_michael :
  ∀ (jacob_age michael_age : ℕ),
    (jacob_age < michael_age) →                        -- Jacob is younger than Michael
    (michael_age + 9 = 2 * (jacob_age + 9)) →          -- 9 years from now, Michael will be twice as old as Jacob
    (jacob_age + 4 = 9) →                              -- Jacob will be 9 years old in 4 years
    age_difference jacob_age michael_age = 14 :=        -- The age difference is 14 years
by
  sorry  -- Proof omitted


end NUMINAMATH_CALUDE_jacob_younger_than_michael_l1648_164811


namespace NUMINAMATH_CALUDE_pq_ratio_implies_pg_ps_ratio_l1648_164827

/-- Triangle PQR with angle bisector PS intersecting MN at G -/
structure Triangle (P Q R S M N G : ℝ × ℝ) :=
  (M_on_PQ : ∃ t : ℝ, M = (1 - t) • P + t • Q ∧ 0 ≤ t ∧ t ≤ 1)
  (N_on_PR : ∃ t : ℝ, N = (1 - t) • P + t • R ∧ 0 ≤ t ∧ t ≤ 1)
  (S_angle_bisector : ∃ t : ℝ, S = (1 - t) • P + t • ((Q + R) / 2) ∧ 0 < t)
  (G_on_MN : ∃ t : ℝ, G = (1 - t) • M + t • N ∧ 0 ≤ t ∧ t ≤ 1)
  (G_on_PS : ∃ t : ℝ, G = (1 - t) • P + t • S ∧ 0 ≤ t ∧ t ≤ 1)

/-- The main theorem -/
theorem pq_ratio_implies_pg_ps_ratio 
  (P Q R S M N G : ℝ × ℝ) 
  (h : Triangle P Q R S M N G) 
  (hPM_MQ : ∃ (t : ℝ), M = (1 - t) • P + t • Q ∧ t = 1/4) 
  (hPN_NR : ∃ (t : ℝ), N = (1 - t) • P + t • R ∧ t = 1/4) :
  ∃ (t : ℝ), G = (1 - t) • P + t • S ∧ t = 5/18 :=
sorry

end NUMINAMATH_CALUDE_pq_ratio_implies_pg_ps_ratio_l1648_164827


namespace NUMINAMATH_CALUDE_expression_evaluation_l1648_164815

theorem expression_evaluation (a : ℤ) (h : a = -1) : 
  (2*a + 1) * (2*a - 1) - 4*a*(a - 1) = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1648_164815


namespace NUMINAMATH_CALUDE_fixed_point_of_parabola_l1648_164828

theorem fixed_point_of_parabola (t : ℝ) : 
  5 * (3 : ℝ)^2 + t * 3 - 3 * t = 45 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_parabola_l1648_164828


namespace NUMINAMATH_CALUDE_basketball_shooting_frequency_l1648_164817

/-- Given a basketball player who made 90 total shots with 63 successful shots,
    prove that the shooting frequency is equal to 0.7. -/
theorem basketball_shooting_frequency :
  let total_shots : ℕ := 90
  let successful_shots : ℕ := 63
  let shooting_frequency := (successful_shots : ℚ) / total_shots
  shooting_frequency = 0.7 := by sorry

end NUMINAMATH_CALUDE_basketball_shooting_frequency_l1648_164817


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1648_164821

theorem largest_digit_divisible_by_six : 
  ∀ N : ℕ, N ≤ 9 → (57890 + N).mod 6 = 0 → N ≤ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1648_164821


namespace NUMINAMATH_CALUDE_quadratic_roots_l1648_164806

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = -1 ∧ 
  (∀ x : ℝ, x^2 + x = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1648_164806


namespace NUMINAMATH_CALUDE_soup_bins_calculation_l1648_164805

def total_bins : ℚ := 75/100
def vegetable_bins : ℚ := 12/100
def pasta_bins : ℚ := 1/2

theorem soup_bins_calculation : 
  total_bins - (vegetable_bins + pasta_bins) = 13/100 := by
  sorry

end NUMINAMATH_CALUDE_soup_bins_calculation_l1648_164805


namespace NUMINAMATH_CALUDE_count_multiples_theorem_l1648_164873

/-- The count of positive integers not exceeding 500 that are multiples of 2 or 5 but not 6 -/
def count_multiples : ℕ := sorry

/-- The upper bound of the range -/
def upper_bound : ℕ := 500

/-- Predicate for a number being a multiple of 2 or 5 but not 6 -/
def is_valid_multiple (n : ℕ) : Prop :=
  n ≤ upper_bound ∧ (n % 2 = 0 ∨ n % 5 = 0) ∧ n % 6 ≠ 0

theorem count_multiples_theorem : count_multiples = 217 := by sorry

end NUMINAMATH_CALUDE_count_multiples_theorem_l1648_164873


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1648_164880

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 200 →
  b = 20 →
  a^2 + b^2 = c^2 →
  a + b + c = 40 + 20 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1648_164880


namespace NUMINAMATH_CALUDE_pyramid_z_value_l1648_164856

/-- Represents a three-level pyramid structure -/
structure Pyramid where
  z : ℕ
  x : ℕ
  y : ℕ
  bottom_left : ℕ
  bottom_middle : ℕ
  bottom_right : ℕ

/-- Checks if the pyramid satisfies the given conditions -/
def is_valid_pyramid (p : Pyramid) : Prop :=
  p.bottom_left = p.z * p.x ∧
  p.bottom_middle = p.x * p.y ∧
  p.bottom_right = p.y * p.z

theorem pyramid_z_value :
  ∀ p : Pyramid,
    is_valid_pyramid p →
    p.bottom_left = 8 →
    p.bottom_middle = 40 →
    p.bottom_right = 10 →
    p.z = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_pyramid_z_value_l1648_164856


namespace NUMINAMATH_CALUDE_min_value_f_max_value_g_l1648_164869

-- Define the functions
def f (m : ℝ) : ℝ := m^2 + 2*m + 3
def g (m : ℝ) : ℝ := -m^2 + 2*m + 3

-- Theorem for the minimum value of f
theorem min_value_f : ∀ m : ℝ, f m ≥ 2 ∧ ∃ m₀ : ℝ, f m₀ = 2 :=
sorry

-- Theorem for the maximum value of g
theorem max_value_g : ∀ m : ℝ, g m ≤ 4 ∧ ∃ m₀ : ℝ, g m₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_max_value_g_l1648_164869


namespace NUMINAMATH_CALUDE_zoo_animals_count_l1648_164868

/-- The number of tiger enclosures in the zoo -/
def tiger_enclosures : ℕ := 4

/-- The number of zebra enclosures behind each tiger enclosure -/
def zebra_enclosures_per_tiger : ℕ := 2

/-- The ratio of giraffe enclosures to zebra enclosures -/
def giraffe_to_zebra_ratio : ℕ := 3

/-- The number of tigers in each tiger enclosure -/
def tigers_per_enclosure : ℕ := 4

/-- The number of zebras in each zebra enclosure -/
def zebras_per_enclosure : ℕ := 10

/-- The number of giraffes in each giraffe enclosure -/
def giraffes_per_enclosure : ℕ := 2

/-- The total number of animals in the zoo -/
def total_animals : ℕ := 144

theorem zoo_animals_count :
  tiger_enclosures * tigers_per_enclosure +
  (tiger_enclosures * zebra_enclosures_per_tiger) * zebras_per_enclosure +
  (tiger_enclosures * zebra_enclosures_per_tiger * giraffe_to_zebra_ratio) * giraffes_per_enclosure =
  total_animals := by sorry

end NUMINAMATH_CALUDE_zoo_animals_count_l1648_164868


namespace NUMINAMATH_CALUDE_some_number_value_l1648_164893

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * n * 49) : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1648_164893


namespace NUMINAMATH_CALUDE_soda_cans_for_euros_l1648_164807

/-- The number of cans of soda that can be purchased for E euros, given that S cans can be purchased for Q quarters and 1 euro is worth 5 quarters. -/
theorem soda_cans_for_euros (S Q E : ℚ) (h1 : S > 0) (h2 : Q > 0) (h3 : E > 0) :
  (S / Q) * (5 * E) = (5 * S * E) / Q := by
  sorry

#check soda_cans_for_euros

end NUMINAMATH_CALUDE_soda_cans_for_euros_l1648_164807


namespace NUMINAMATH_CALUDE_milk_fraction_in_cup1_l1648_164874

theorem milk_fraction_in_cup1 (initial_tea : ℝ) (initial_milk : ℝ) (cup_size : ℝ) : 
  initial_tea = 6 →
  initial_milk = 8 →
  cup_size = 12 →
  let tea_transferred_to_cup2 := initial_tea / 3
  let tea_in_cup1_after_first_transfer := initial_tea - tea_transferred_to_cup2
  let total_in_cup2_after_first_transfer := initial_milk + tea_transferred_to_cup2
  let amount_transferred_back := total_in_cup2_after_first_transfer / 4
  let milk_ratio_in_cup2 := initial_milk / total_in_cup2_after_first_transfer
  let milk_transferred_back := amount_transferred_back * milk_ratio_in_cup2
  let final_tea_in_cup1 := tea_in_cup1_after_first_transfer + (amount_transferred_back - milk_transferred_back)
  let final_milk_in_cup1 := milk_transferred_back
  let total_liquid_in_cup1 := final_tea_in_cup1 + final_milk_in_cup1
  final_milk_in_cup1 / total_liquid_in_cup1 = 2 / 6.5 :=
by sorry

end NUMINAMATH_CALUDE_milk_fraction_in_cup1_l1648_164874


namespace NUMINAMATH_CALUDE_inverse_proportion_graph_l1648_164801

/-- Given that point A(2,4) lies on the graph of y = k/x, prove that (4,2) also lies on the graph
    while (-2,4), (2,-4), and (-4,2) do not. -/
theorem inverse_proportion_graph (k : ℝ) (h : k ≠ 0) : 
  (4 : ℝ) = k / 2 →  -- Point A(2,4) lies on the graph
  (2 : ℝ) = k / 4 ∧  -- Point (4,2) lies on the graph
  (4 : ℝ) ≠ k / (-2) ∧  -- Point (-2,4) does not lie on the graph
  (-4 : ℝ) ≠ k / 2 ∧  -- Point (2,-4) does not lie on the graph
  (2 : ℝ) ≠ k / (-4) :=  -- Point (-4,2) does not lie on the graph
by
  sorry


end NUMINAMATH_CALUDE_inverse_proportion_graph_l1648_164801


namespace NUMINAMATH_CALUDE_shed_blocks_count_l1648_164803

/-- Represents the dimensions of a rectangular structure -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular structure -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- Represents the specifications of the shed -/
structure ShedSpecs where
  outer : Dimensions
  wallThickness : ℝ

/-- Calculates the inner dimensions of the shed -/
def innerDimensions (s : ShedSpecs) : Dimensions :=
  { length := s.outer.length - 2 * s.wallThickness,
    width := s.outer.width - 2 * s.wallThickness,
    height := s.outer.height - 2 * s.wallThickness }

/-- Calculates the number of blocks used in the shed construction -/
def blocksUsed (s : ShedSpecs) : ℝ :=
  volume s.outer - volume (innerDimensions s)

/-- The main theorem stating the number of blocks used in the shed construction -/
theorem shed_blocks_count :
  let shedSpecs : ShedSpecs := {
    outer := { length := 15, width := 12, height := 7 },
    wallThickness := 1.5
  }
  blocksUsed shedSpecs = 828 := by sorry

end NUMINAMATH_CALUDE_shed_blocks_count_l1648_164803


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1648_164885

theorem trigonometric_identity (α : ℝ) :
  3.404 * (8 * Real.cos α ^ 4 - 4 * Real.cos α ^ 3 - 8 * Real.cos α ^ 2 + 3 * Real.cos α + 1) /
  (8 * Real.cos α ^ 4 + 4 * Real.cos α ^ 3 - 8 * Real.cos α ^ 2 - 3 * Real.cos α + 1) =
  -Real.tan (7 * α / 2) * Real.tan (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1648_164885


namespace NUMINAMATH_CALUDE_multiply_negative_two_l1648_164816

theorem multiply_negative_two : 3 * (-2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_multiply_negative_two_l1648_164816


namespace NUMINAMATH_CALUDE_waiter_customers_l1648_164870

theorem waiter_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) 
  (h1 : num_tables = 8)
  (h2 : women_per_table = 7)
  (h3 : men_per_table = 4) :
  num_tables * (women_per_table + men_per_table) = 88 := by
sorry

end NUMINAMATH_CALUDE_waiter_customers_l1648_164870


namespace NUMINAMATH_CALUDE_min_sin_cos_expression_l1648_164852

theorem min_sin_cos_expression (A : Real) : 
  let f := λ x : Real => Real.sin (x / 2) - Real.sqrt 3 * Real.cos (x / 2)
  ∃ m : Real, (∀ x, f x ≥ m) ∧ f (-π/3) = m :=
sorry

end NUMINAMATH_CALUDE_min_sin_cos_expression_l1648_164852


namespace NUMINAMATH_CALUDE_expression_value_l1648_164800

theorem expression_value : 3 * (24 + 7)^2 - (24^2 + 7^2) = 2258 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1648_164800


namespace NUMINAMATH_CALUDE_max_value_of_function_l1648_164899

theorem max_value_of_function (x : ℝ) (h : 0 < x ∧ x < 3/2) : 
  x * (3 - 2*x) ≤ 9/8 ∧ ∃ x₀, 0 < x₀ ∧ x₀ < 3/2 ∧ x₀ * (3 - 2*x₀) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1648_164899


namespace NUMINAMATH_CALUDE_roots_of_quadratic_sum_of_fourth_powers_l1648_164818

theorem roots_of_quadratic_sum_of_fourth_powers (α β : ℝ) : 
  α^2 - 2*α - 8 = 0 → β^2 - 2*β - 8 = 0 → 3*α^4 + 4*β^4 = 1232 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_sum_of_fourth_powers_l1648_164818


namespace NUMINAMATH_CALUDE_essay_completion_time_l1648_164812

-- Define the essay parameters
def essay_length : ℕ := 1200
def initial_speed : ℕ := 400
def initial_duration : ℕ := 2
def subsequent_speed : ℕ := 200

-- Theorem statement
theorem essay_completion_time :
  let initial_words := initial_speed * initial_duration
  let remaining_words := essay_length - initial_words
  let subsequent_duration := remaining_words / subsequent_speed
  initial_duration + subsequent_duration = 4 := by
  sorry

end NUMINAMATH_CALUDE_essay_completion_time_l1648_164812


namespace NUMINAMATH_CALUDE_sine_absolute_value_integral_l1648_164878

theorem sine_absolute_value_integral : ∫ x in (0)..(2 * Real.pi), |Real.sin x| = 4 := by
  sorry

end NUMINAMATH_CALUDE_sine_absolute_value_integral_l1648_164878


namespace NUMINAMATH_CALUDE_trampoline_jumps_l1648_164841

theorem trampoline_jumps (ronald_jumps : ℕ) (rupert_extra_jumps : ℕ) : 
  ronald_jumps = 157 → rupert_extra_jumps = 86 → 
  ronald_jumps + (ronald_jumps + rupert_extra_jumps) = 400 := by
sorry

end NUMINAMATH_CALUDE_trampoline_jumps_l1648_164841


namespace NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_17_l1648_164853

theorem three_digit_numbers_divisible_by_17 : 
  (Finset.filter (fun k => 100 ≤ 17 * k ∧ 17 * k ≤ 999) (Finset.range 1000)).card = 53 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_17_l1648_164853


namespace NUMINAMATH_CALUDE_geometric_progression_values_l1648_164867

theorem geometric_progression_values (p : ℝ) : 
  (4*p + 5 ≠ 0 ∧ 2*p ≠ 0 ∧ |p - 3| ≠ 0) ∧
  (2*p)^2 = (4*p + 5) * |p - 3| ↔ 
  p = -1 ∨ p = 15/8 := by sorry

end NUMINAMATH_CALUDE_geometric_progression_values_l1648_164867


namespace NUMINAMATH_CALUDE_triangle_cos_inequality_l1648_164859

/-- For any real numbers A, B, C that are angles of a triangle, 
    the inequality 8 cos A · cos B · cos C ≤ 1 holds. -/
theorem triangle_cos_inequality (A B C : Real) (h : A + B + C = π) :
  8 * Real.cos A * Real.cos B * Real.cos C ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cos_inequality_l1648_164859


namespace NUMINAMATH_CALUDE_odd_function_half_value_l1648_164898

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x

theorem odd_function_half_value (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) → f a (1/2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_half_value_l1648_164898


namespace NUMINAMATH_CALUDE_sum_of_digits_equals_16_l1648_164887

/-- The sum of the digits of (10^38) - 85 when written as a base 10 integer -/
def sumOfDigits : ℕ :=
  -- Define the sum of digits here
  sorry

/-- Theorem stating that the sum of the digits of (10^38) - 85 is 16 -/
theorem sum_of_digits_equals_16 : sumOfDigits = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_equals_16_l1648_164887


namespace NUMINAMATH_CALUDE_range_of_m_l1648_164886

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |x - m| ≤ 2) → 
  -1 ≤ m ∧ m ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1648_164886


namespace NUMINAMATH_CALUDE_circle_ratio_after_diameter_increase_l1648_164879

/-- Theorem: For any circle with an initial diameter of 2r units, 
if the diameter is increased by 4 units, 
the ratio of the new circumference to the new diameter is equal to π. -/
theorem circle_ratio_after_diameter_increase (r : ℝ) (r_pos : r > 0) : 
  let initial_diameter : ℝ := 2 * r
  let new_diameter : ℝ := initial_diameter + 4
  let new_circumference : ℝ := 2 * π * (r + 2)
  new_circumference / new_diameter = π :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_diameter_increase_l1648_164879


namespace NUMINAMATH_CALUDE_gasoline_cost_calculation_l1648_164883

theorem gasoline_cost_calculation
  (cost_per_litre : ℝ)
  (distance_per_litre : ℝ)
  (distance_to_travel : ℝ)
  (cost_per_litre_positive : 0 < cost_per_litre)
  (distance_per_litre_positive : 0 < distance_per_litre) :
  cost_per_litre * distance_to_travel / distance_per_litre =
  cost_per_litre * (distance_to_travel / distance_per_litre) :=
by sorry

#check gasoline_cost_calculation

end NUMINAMATH_CALUDE_gasoline_cost_calculation_l1648_164883


namespace NUMINAMATH_CALUDE_fraction_addition_l1648_164891

theorem fraction_addition : (1 : ℚ) / 210 + 17 / 35 = 103 / 210 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1648_164891
