import Mathlib

namespace three_by_three_grid_paths_l3632_363261

/-- The number of paths from (0,0) to (n,m) on a grid, moving only right or down -/
def grid_paths (n m : ℕ) : ℕ := Nat.choose (n + m) n

/-- Theorem: There are 20 distinct paths from the top-left to the bottom-right corner of a 3x3 grid -/
theorem three_by_three_grid_paths : grid_paths 3 3 = 20 := by sorry

end three_by_three_grid_paths_l3632_363261


namespace quadratic_always_positive_l3632_363234

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → x^2 - 2*k*x + 2*k - 1 > 0) ↔ k ≥ 1 := by
  sorry

end quadratic_always_positive_l3632_363234


namespace consecutive_odd_integers_sum_l3632_363219

theorem consecutive_odd_integers_sum (x y : ℤ) : 
  x = 63 → 
  y = x + 2 → 
  Odd x → 
  Odd y → 
  x + y = 128 := by
sorry

end consecutive_odd_integers_sum_l3632_363219


namespace option_A_not_sufficient_l3632_363207

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Two lines are perpendicular -/
def perp_lines (l1 l2 : Line3D) : Prop :=
  sorry

theorem option_A_not_sufficient
  (a b : Line3D)
  (α β : Plane3D)
  (h1 : a ≠ b)
  (h2 : α ≠ β)
  (h3 : line_parallel_plane a α)
  (h4 : line_parallel_plane b β)
  (h5 : line_perp_plane a β) :
  ¬ (perp_lines a b) :=
sorry

end option_A_not_sufficient_l3632_363207


namespace room_height_from_curtain_l3632_363289

/-- The height of a room from the curtain rod to the floor, given curtain length and pooling material. -/
theorem room_height_from_curtain (curtain_length : ℕ) (pooling_material : ℕ) : 
  curtain_length = 101 ∧ pooling_material = 5 → curtain_length - pooling_material = 96 := by
  sorry

#check room_height_from_curtain

end room_height_from_curtain_l3632_363289


namespace investment_pays_off_after_9_months_l3632_363211

/-- Cumulative net income function for the first 5 months after improvement -/
def g (n : ℕ) : ℚ :=
  if n ≤ 5 then n^2 + 100*n else 109*n - 20

/-- Monthly income without improvement (in 10,000 yuan) -/
def monthly_income : ℚ := 70

/-- Fine function without improvement (in 10,000 yuan) -/
def fine (n : ℕ) : ℚ := n^2 + 2*n

/-- Initial investment (in 10,000 yuan) -/
def investment : ℚ := 500

/-- One-time reward after improvement (in 10,000 yuan) -/
def reward : ℚ := 100

/-- Cumulative net income with improvement (in 10,000 yuan) -/
def income_with_improvement (n : ℕ) : ℚ :=
  g n - investment + reward

/-- Cumulative net income without improvement (in 10,000 yuan) -/
def income_without_improvement (n : ℕ) : ℚ :=
  n * monthly_income - fine n

theorem investment_pays_off_after_9_months :
  ∀ n : ℕ, n ≥ 9 → income_with_improvement n > income_without_improvement n :=
sorry

end investment_pays_off_after_9_months_l3632_363211


namespace xyz_sum_l3632_363294

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : z * x + y = 47) :
  x + y + z = 48 := by
  sorry

end xyz_sum_l3632_363294


namespace expression_evaluation_l3632_363279

theorem expression_evaluation :
  let x : ℚ := -2
  (x - 2)^2 + (2 + x)*(x - 2) - 2*x*(2*x - 1) = -4 := by sorry

end expression_evaluation_l3632_363279


namespace binomial_expansion_102_l3632_363240

theorem binomial_expansion_102 : 
  102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 104060401 := by
  sorry

end binomial_expansion_102_l3632_363240


namespace perfect_square_trinomial_condition_l3632_363217

/-- A trinomial ax^2 + bx + c is a perfect square if there exist p and q such that
    ax^2 + bx + c = (px + q)^2 for all x. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

/-- If 4x^2 + mx + 25 is a perfect square trinomial, then m = 20. -/
theorem perfect_square_trinomial_condition (m : ℝ) :
  is_perfect_square_trinomial 4 m 25 → m = 20 :=
by sorry

end perfect_square_trinomial_condition_l3632_363217


namespace sqrt_difference_equals_three_sqrt_three_l3632_363248

theorem sqrt_difference_equals_three_sqrt_three : 
  Real.sqrt 75 - Real.sqrt 12 = 3 * Real.sqrt 3 := by
  sorry

end sqrt_difference_equals_three_sqrt_three_l3632_363248


namespace median_and_mode_of_scores_l3632_363232

/-- Represents the score distribution of students in the competition -/
def score_distribution : List (Nat × Nat) :=
  [(85, 1), (88, 7), (90, 11), (93, 10), (94, 13), (97, 7), (99, 1)]

/-- The total number of students -/
def total_students : Nat := 50

/-- Calculates the median of the given score distribution -/
def median (dist : List (Nat × Nat)) (total : Nat) : Nat :=
  sorry

/-- Calculates the mode of the given score distribution -/
def mode (dist : List (Nat × Nat)) : Nat :=
  sorry

/-- Theorem stating that the median is 93 and the mode is 94 for the given distribution -/
theorem median_and_mode_of_scores :
  median score_distribution total_students = 93 ∧
  mode score_distribution = 94 :=
sorry

end median_and_mode_of_scores_l3632_363232


namespace crackers_per_friend_l3632_363208

theorem crackers_per_friend (initial_crackers : ℕ) (friends : ℕ) (remaining_crackers : ℕ) 
  (h1 : initial_crackers = 15)
  (h2 : friends = 5)
  (h3 : remaining_crackers = 10) :
  (initial_crackers - remaining_crackers) / friends = 1 := by
  sorry

end crackers_per_friend_l3632_363208


namespace inequality_system_solution_set_l3632_363238

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 1 ≥ 0 ∧ (x - 1) / 2 < 1}
  S = {x : ℝ | -1 ≤ x ∧ x < 3} := by
sorry

end inequality_system_solution_set_l3632_363238


namespace wax_calculation_l3632_363251

/-- Given the total required wax and additional wax needed, calculates the amount of wax already possessed. -/
def wax_already_possessed (total_required : ℕ) (additional_needed : ℕ) : ℕ :=
  total_required - additional_needed

/-- Proves that given the specific values in the problem, the wax already possessed is 331 g. -/
theorem wax_calculation :
  let total_required : ℕ := 353
  let additional_needed : ℕ := 22
  wax_already_possessed total_required additional_needed = 331 := by
  sorry

end wax_calculation_l3632_363251


namespace base7_to_base10_23456_l3632_363227

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- The given number in base 7 --/
def base7Number : List Nat := [6, 5, 4, 3, 2]

/-- Theorem stating that the base 10 equivalent of 23456 in base 7 is 6068 --/
theorem base7_to_base10_23456 :
  base7ToBase10 base7Number = 6068 := by
  sorry

end base7_to_base10_23456_l3632_363227


namespace circle_equation_through_ABC_circle_equation_center_y_2_l3632_363271

-- Define the circle P
def CircleP : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (c : ℝ × ℝ) (r : ℝ), (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}

-- Define the points A, B, and C
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (6, -2)

-- Theorem 1
theorem circle_equation_through_ABC :
  A ∈ CircleP ∧ B ∈ CircleP ∧ C ∈ CircleP →
  ∃ (D E F : ℝ), ∀ (x y : ℝ), (x, y) ∈ CircleP ↔ x^2 + y^2 + D*x + E*y + F = 0 :=
sorry

-- Theorem 2
theorem circle_equation_center_y_2 :
  A ∈ CircleP ∧ B ∈ CircleP ∧ (∃ (c : ℝ × ℝ), c ∈ CircleP ∧ c.2 = 2) →
  ∃ (c : ℝ × ℝ) (r : ℝ), c = (5/2, 2) ∧ r = 5/2 ∧
    ∀ (x y : ℝ), (x, y) ∈ CircleP ↔ (x - c.1)^2 + (y - c.2)^2 = r^2 :=
sorry

end circle_equation_through_ABC_circle_equation_center_y_2_l3632_363271


namespace symmetric_points_sum_l3632_363210

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- Given that point A(m, 1) is symmetric to point B(2, n) with respect to the x-axis,
    prove that m + n = 1 -/
theorem symmetric_points_sum (m n : ℝ) :
  symmetric_x_axis (m, 1) (2, n) → m + n = 1 := by
  sorry

end symmetric_points_sum_l3632_363210


namespace sum_of_two_numbers_l3632_363230

theorem sum_of_two_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 23 := by
  sorry

end sum_of_two_numbers_l3632_363230


namespace positive_A_value_l3632_363275

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value :
  ∃ A : ℝ, A > 0 ∧ hash A 3 = 145 ∧ A = 2 * Real.sqrt 34 := by
  sorry

end positive_A_value_l3632_363275


namespace greatest_integer_third_side_l3632_363265

theorem greatest_integer_third_side (a b : ℝ) (ha : a = 7) (hb : b = 11) :
  ∃ (c : ℕ), c = 17 ∧ 
  (∀ (x : ℕ), x > c → ¬(a + b > x ∧ a + x > b ∧ b + x > a)) :=
by sorry

end greatest_integer_third_side_l3632_363265


namespace dictation_mistakes_l3632_363293

theorem dictation_mistakes (n : ℕ) (max_mistakes : ℕ) 
  (h1 : n = 30) 
  (h2 : max_mistakes = 12) : 
  ∃ k : ℕ, ∃ (s : Finset (Fin n)), s.card ≥ 3 ∧ 
  ∀ i ∈ s, ∃ f : Fin n → ℕ, f i = k ∧ f i ≤ max_mistakes :=
by sorry

end dictation_mistakes_l3632_363293


namespace no_distinct_complex_numbers_satisfying_equations_l3632_363269

theorem no_distinct_complex_numbers_satisfying_equations :
  ∀ (a b c d : ℂ),
  (a^3 - b*c*d = b^3 - c*d*a) ∧
  (b^3 - c*d*a = c^3 - d*a*b) ∧
  (c^3 - d*a*b = d^3 - a*b*c) →
  ¬(a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :=
by sorry

end no_distinct_complex_numbers_satisfying_equations_l3632_363269


namespace max_min_difference_2a_minus_b_l3632_363295

theorem max_min_difference_2a_minus_b : 
  ∃ (max min : ℝ), 
    (∀ a b : ℝ, a^2 + b^2 - 2*a - 4 = 0 → 2*a - b ≤ max) ∧
    (∀ a b : ℝ, a^2 + b^2 - 2*a - 4 = 0 → 2*a - b ≥ min) ∧
    (∃ a1 b1 a2 b2 : ℝ, 
      a1^2 + b1^2 - 2*a1 - 4 = 0 ∧
      a2^2 + b2^2 - 2*a2 - 4 = 0 ∧
      2*a1 - b1 = max ∧
      2*a2 - b2 = min) ∧
    max - min = 10 :=
by sorry

end max_min_difference_2a_minus_b_l3632_363295


namespace safari_park_acrobats_l3632_363252

theorem safari_park_acrobats :
  ∀ (acrobats giraffes : ℕ),
    2 * acrobats + 4 * giraffes = 32 →
    acrobats + giraffes = 10 →
    acrobats = 4 :=
by
  sorry

end safari_park_acrobats_l3632_363252


namespace principal_calculation_l3632_363260

/-- Given a principal P and an interest rate R (as a percentage),
    if the amount after 2 years is 780 and after 7 years is 1020,
    then the principal P is 684. -/
theorem principal_calculation (P R : ℚ) 
  (h1 : P + (P * R * 2) / 100 = 780)
  (h2 : P + (P * R * 7) / 100 = 1020) : 
  P = 684 := by
sorry

end principal_calculation_l3632_363260


namespace polygon_triangulation_l3632_363296

/-- Theorem: For any polygon with n sides divided into k triangles, k ≥ n - 2 -/
theorem polygon_triangulation (n k : ℕ) (h_n : n ≥ 3) (h_k : k > 0) : k ≥ n - 2 := by
  sorry


end polygon_triangulation_l3632_363296


namespace simplify_absolute_value_expression_l3632_363281

noncomputable def f (x : ℝ) : ℝ := |2*x + 1| - |x - 3| + |x - 6|

noncomputable def g (x : ℝ) : ℝ :=
  if x < -1/2 then -2*x + 2
  else if x < 3 then 2*x + 4
  else if x < 6 then 10
  else 2*x - 2

theorem simplify_absolute_value_expression :
  ∀ x : ℝ, f x = g x := by sorry

end simplify_absolute_value_expression_l3632_363281


namespace wrench_force_calculation_l3632_363212

/-- Given two wrenches with different handle lengths, calculate the force required for the second wrench -/
theorem wrench_force_calculation (l₁ l₂ f₁ : ℝ) (h₁ : l₁ > 0) (h₂ : l₂ > 0) (h₃ : f₁ > 0) :
  let f₂ := (l₁ * f₁) / l₂
  l₁ = 12 ∧ f₁ = 450 ∧ l₂ = 18 → f₂ = 300 := by
  sorry

#check wrench_force_calculation

end wrench_force_calculation_l3632_363212


namespace parabola_line_intersection_l3632_363273

/-- Parabola represented by the equation y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Line represented by the equation y = kx - 1 -/
def line (k x y : ℝ) : Prop := y = k*x - 1

/-- Focus of the parabola y² = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- The line passes through the focus of the parabola -/
def line_passes_through_focus (k : ℝ) : Prop :=
  line k (focus.1) (focus.2)

/-- The line intersects the parabola at two points -/
def line_intersects_parabola (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂

theorem parabola_line_intersection 
  (h1 : line_passes_through_focus k)
  (h2 : line_intersects_parabola k) :
  k = 1 ∧ ∃ (x₁ x₂ : ℝ), x₁ + x₂ + 2 = 8 :=
sorry

end parabola_line_intersection_l3632_363273


namespace purely_imaginary_complex_number_l3632_363222

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) = a^2 - 1 + Complex.I * (a - 1)) → a = -1 := by
  sorry

end purely_imaginary_complex_number_l3632_363222


namespace right_triangle_third_side_l3632_363274

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  ((a = Real.sqrt 2 ∧ b = Real.sqrt 3) ∨ (a = Real.sqrt 3 ∧ b = Real.sqrt 2)) →
  c = Real.sqrt 5 ∨ c = 1 := by
sorry

end right_triangle_third_side_l3632_363274


namespace tangent_slope_implies_a_f_upper_bound_implies_a_range_l3632_363202

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + Real.log x

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2 * a * x + 1 / x

theorem tangent_slope_implies_a (a : ℝ) :
  f_deriv a 1 = -1 → a = -1 := by sorry

theorem f_upper_bound_implies_a_range (a : ℝ) :
  a < 0 →
  (∀ x > 0, f a x ≤ -1/2) →
  a ≤ -1/2 := by sorry

end

end tangent_slope_implies_a_f_upper_bound_implies_a_range_l3632_363202


namespace complex_sum_equals_one_l3632_363291

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_sum_equals_one :
  (i + i^3)^100 + (i + i^2 + i^3 + i^4 + i^5)^120 = 1 :=
by
  sorry

end complex_sum_equals_one_l3632_363291


namespace square_sum_equals_twice_square_l3632_363298

theorem square_sum_equals_twice_square (a : ℝ) : a^2 + a^2 = 2 * a^2 := by
  sorry

end square_sum_equals_twice_square_l3632_363298


namespace functional_equation_implies_g_50_eq_0_l3632_363255

/-- A function satisfying the given functional equation for all positive real numbers -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → x * g y - y * g x = g (x / y) + g (x + y)

/-- The main theorem stating that any function satisfying the functional equation must have g(50) = 0 -/
theorem functional_equation_implies_g_50_eq_0 (g : ℝ → ℝ) (h : FunctionalEquation g) : g 50 = 0 := by
  sorry

#check functional_equation_implies_g_50_eq_0

end functional_equation_implies_g_50_eq_0_l3632_363255


namespace quadratic_equation_condition_l3632_363264

theorem quadratic_equation_condition (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ (m - 2) * x^2 + (2*m + 1) * x - m = a * x^2 + b * x + c) →
  m = -2 := by
sorry

end quadratic_equation_condition_l3632_363264


namespace polynomial_factorization_l3632_363276

theorem polynomial_factorization :
  ∀ x : ℝ, (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 7) = 
           (x^2 + 7*x + 2) * (x^2 + 5*x + 19) := by
  sorry

end polynomial_factorization_l3632_363276


namespace student_number_problem_l3632_363206

theorem student_number_problem (x : ℝ) : (3/2 : ℝ) * x + 53.4 = -78.9 → x = -88.2 := by
  sorry

end student_number_problem_l3632_363206


namespace guitar_ratio_proof_l3632_363201

/-- Proves that the ratio of Barbeck's guitars to Steve's guitars is 2:1 given the problem conditions -/
theorem guitar_ratio_proof (total_guitars : ℕ) (davey_guitars : ℕ) (barbeck_guitars : ℕ) (steve_guitars : ℕ) : 
  total_guitars = 27 →
  davey_guitars = 18 →
  barbeck_guitars = steve_guitars →
  davey_guitars = 3 * barbeck_guitars →
  total_guitars = davey_guitars + barbeck_guitars + steve_guitars →
  (barbeck_guitars : ℚ) / steve_guitars = 2 / 1 :=
by sorry


end guitar_ratio_proof_l3632_363201


namespace xy_value_l3632_363268

theorem xy_value (x y : ℝ) 
  (h1 : (8:ℝ)^x / 2^(x+y) = 16)
  (h2 : (16:ℝ)^(x+y) / 4^(5*y) = 1024) : 
  x * y = 7/8 := by
sorry

end xy_value_l3632_363268


namespace wire_cut_ratio_l3632_363288

/-- Given a wire cut into two pieces of lengths a and b, where a forms a square
    and b forms a circle, and the perimeter of the square equals the circumference
    of the circle, prove that a/b = 1. -/
theorem wire_cut_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (4 * (a / 4) = 2 * Real.pi * (b / (2 * Real.pi))) → a / b = 1 := by
  sorry

end wire_cut_ratio_l3632_363288


namespace zhou_yu_age_theorem_l3632_363249

/-- Represents the equation for Zhou Yu's age at death -/
def zhou_yu_age_equation (x : ℕ) : Prop :=
  x^2 = 10 * (x - 3) + x

/-- Theorem stating the conditions and the equation for Zhou Yu's age at death -/
theorem zhou_yu_age_theorem (x : ℕ) :
  (x ≥ 10 ∧ x < 100) →  -- Two-digit number
  (x / 10 = x % 10 - 3) →  -- Tens digit is 3 less than units digit
  (x^2 = 10 * (x - 3) + x) →  -- Square of units digit equals the age
  zhou_yu_age_equation x :=
by
  sorry

#check zhou_yu_age_theorem

end zhou_yu_age_theorem_l3632_363249


namespace area_bounded_by_cos_sin_squared_l3632_363209

theorem area_bounded_by_cos_sin_squared (f : ℝ → ℝ) (h : ∀ x, f x = Real.cos x * Real.sin x ^ 2) :
  ∫ x in (0)..(Real.pi / 2), f x = 1 / 3 := by
  sorry

end area_bounded_by_cos_sin_squared_l3632_363209


namespace inequality_holds_iff_l3632_363218

theorem inequality_holds_iff (n : ℕ) :
  (∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 →
    (1 / a^n + 1 / b^n ≥ a^m + b^m)) ↔ (m = n ∨ m = n + 1) :=
by sorry

end inequality_holds_iff_l3632_363218


namespace solve_for_p_l3632_363239

theorem solve_for_p (n m p : ℚ) 
  (h1 : (3 : ℚ) / 4 = n / 48)
  (h2 : (3 : ℚ) / 4 = (m + n) / 96)
  (h3 : (3 : ℚ) / 4 = (p - m) / 160) : 
  p = 156 := by sorry

end solve_for_p_l3632_363239


namespace ellipse_intersection_fixed_point_l3632_363205

/-- Ellipse C with equation x²/4 + y²/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Line l with equation y = kx + m -/
def line_l (k m x y : ℝ) : Prop := y = k*x + m

/-- Point A is the right vertex of the ellipse -/
def point_A : ℝ × ℝ := (2, 0)

/-- Circle with diameter MN passes through point A -/
def circle_passes_through_A (M N : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  (x₁ - 2) * (x₂ - 2) + y₁ * y₂ = 0

theorem ellipse_intersection_fixed_point (k m : ℝ) :
  ∃ (M N : ℝ × ℝ),
    ellipse_C M.1 M.2 ∧
    ellipse_C N.1 N.2 ∧
    line_l k m M.1 M.2 ∧
    line_l k m N.1 N.2 ∧
    circle_passes_through_A M N →
    line_l k m (2/7) 0 :=
  sorry

end ellipse_intersection_fixed_point_l3632_363205


namespace total_distance_to_fountain_l3632_363229

/-- The distance from Mrs. Hilt's desk to the water fountain in feet -/
def distance_to_fountain : ℕ := 30

/-- The number of trips Mrs. Hilt makes to the water fountain -/
def number_of_trips : ℕ := 4

/-- Theorem: The total distance Mrs. Hilt walks to the water fountain is 120 feet -/
theorem total_distance_to_fountain :
  distance_to_fountain * number_of_trips = 120 := by
  sorry

end total_distance_to_fountain_l3632_363229


namespace max_factors_is_231_l3632_363266

/-- The number of positive factors of b^n, where b and n are positive integers -/
def num_factors (b n : ℕ+) : ℕ := sorry

/-- The maximum number of positive factors for b^n given constraints -/
def max_num_factors : ℕ := sorry

theorem max_factors_is_231 :
  ∀ b n : ℕ+, b ≤ 20 → n ≤ 10 → num_factors b n ≤ max_num_factors ∧ max_num_factors = 231 := by sorry

end max_factors_is_231_l3632_363266


namespace birthday_pigeonhole_l3632_363215

theorem birthday_pigeonhole (n : ℕ) (h : n = 50) :
  ∃ (m : ℕ) (S : Finset (Fin n)), S.card ≥ 5 ∧ (∀ i ∈ S, (i : ℕ) % 12 + 1 = m) :=
sorry

end birthday_pigeonhole_l3632_363215


namespace smaller_partner_profit_theorem_l3632_363250

/-- Represents a partnership between two individuals -/
structure Partnership where
  investment_ratio : ℚ  -- Ratio of investments (larger / smaller)
  time_ratio : ℚ        -- Ratio of investment periods (longer / shorter)
  total_profit : ℕ      -- Total profit in rupees

/-- Calculates the profit of the partner with the smaller investment -/
def smaller_partner_profit (p : Partnership) : ℚ :=
  p.total_profit * (1 / (1 + p.investment_ratio * p.time_ratio))

/-- Theorem stating the profit of the partner with smaller investment -/
theorem smaller_partner_profit_theorem (p : Partnership) 
  (h1 : p.investment_ratio = 3)
  (h2 : p.time_ratio = 2)
  (h3 : p.total_profit = 35000) :
  ⌊smaller_partner_profit p⌋ = 5000 := by
  sorry

#eval ⌊smaller_partner_profit ⟨3, 2, 35000⟩⌋

end smaller_partner_profit_theorem_l3632_363250


namespace intersection_of_A_and_B_l3632_363225

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ 2*x + 1 ∧ 2*x + 1 ≤ 3}
def B : Set ℝ := {x : ℝ | x ≠ 0 ∧ (x - 3) / (2*x) ≤ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l3632_363225


namespace quadratic_equations_solutions_l3632_363267

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -9 ∧ x₂ = 1 ∧ x₁^2 + 8*x₁ - 9 = 0 ∧ x₂^2 + 8*x₂ - 9 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -3 ∧ y₂ = 1 ∧ y₁*(y₁-1) + 3*(y₁-1) = 0 ∧ y₂*(y₂-1) + 3*(y₂-1) = 0) :=
by sorry

end quadratic_equations_solutions_l3632_363267


namespace students_walking_home_l3632_363246

theorem students_walking_home (bus automobile skateboard bicycle : ℚ)
  (h_bus : bus = 1 / 3)
  (h_auto : automobile = 1 / 5)
  (h_skate : skateboard = 1 / 8)
  (h_bike : bicycle = 1 / 10)
  (h_total : bus + automobile + skateboard + bicycle < 1) :
  1 - (bus + automobile + skateboard + bicycle) = 29 / 120 := by
  sorry

end students_walking_home_l3632_363246


namespace problem_solution_l3632_363253

theorem problem_solution (a b c d : ℕ+) 
  (h1 : a^6 = b^5) 
  (h2 : c^4 = d^3) 
  (h3 : c - a = 31) : 
  c - b = 7 := by
  sorry

end problem_solution_l3632_363253


namespace inscribed_cube_volume_l3632_363247

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let inner_cube_edge := sphere_diameter / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end inscribed_cube_volume_l3632_363247


namespace wire_around_square_field_l3632_363221

theorem wire_around_square_field (area : ℝ) (wire_length : ℝ) (times_around : ℕ) : 
  area = 69696 →
  wire_length = 15840 →
  times_around = 15 →
  wire_length = times_around * (4 * Real.sqrt area) :=
by
  sorry

end wire_around_square_field_l3632_363221


namespace major_axis_length_l3632_363231

/-- Rectangle PQRS with ellipse passing through P and R, foci at Q and S -/
structure EllipseInRectangle where
  /-- Area of the rectangle PQRS -/
  rect_area : ℝ
  /-- Area of the ellipse -/
  ellipse_area : ℝ
  /-- The ellipse passes through P and R, and has foci at Q and S -/
  ellipse_through_PR_foci_QS : Bool

/-- Given the specific rectangle and ellipse, prove the length of the major axis -/
theorem major_axis_length (e : EllipseInRectangle) 
  (h1 : e.rect_area = 4050)
  (h2 : e.ellipse_area = 3240 * Real.pi)
  (h3 : e.ellipse_through_PR_foci_QS = true) : 
  ∃ (major_axis : ℝ), major_axis = 144 := by
  sorry

end major_axis_length_l3632_363231


namespace count_even_greater_than_20000_position_of_35214_count_divisible_by_6_l3632_363284

/-- The set of available digits --/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A five-digit number formed from the available digits --/
structure FiveDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  d5 : Nat
  h1 : d1 ∈ digits
  h2 : d2 ∈ digits
  h3 : d3 ∈ digits
  h4 : d4 ∈ digits
  h5 : d5 ∈ digits
  h6 : d1 ≠ 0  -- Ensures it's a five-digit number

/-- The value of a FiveDigitNumber --/
def FiveDigitNumber.value (n : FiveDigitNumber) : Nat :=
  10000 * n.d1 + 1000 * n.d2 + 100 * n.d3 + 10 * n.d4 + n.d5

/-- The set of all valid FiveDigitNumbers --/
def allFiveDigitNumbers : Finset FiveDigitNumber := sorry

theorem count_even_greater_than_20000 :
  (allFiveDigitNumbers.filter (λ n => n.value % 2 = 0 ∧ n.value > 20000)).card = 240 := by sorry

theorem position_of_35214 :
  (allFiveDigitNumbers.filter (λ n => n.value < 35214)).card + 1 = 351 := by sorry

theorem count_divisible_by_6 :
  (allFiveDigitNumbers.filter (λ n => n.value % 6 = 0)).card = 108 := by sorry

end count_even_greater_than_20000_position_of_35214_count_divisible_by_6_l3632_363284


namespace complement_of_union_l3632_363233

def U : Set ℕ := {x | x > 0 ∧ x < 9}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_of_union : 
  (U \ (M ∪ N)) = {2, 4, 8} := by sorry

end complement_of_union_l3632_363233


namespace intersection_of_A_and_B_l3632_363200

def set_A : Set ℝ := {x | |x - 1| < 3}
def set_B : Set ℝ := {x | (x - 1) / (x - 5) < 0}

theorem intersection_of_A_and_B : 
  set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 4} := by sorry

end intersection_of_A_and_B_l3632_363200


namespace rohan_monthly_salary_l3632_363292

/-- Rohan's monthly expenses and savings --/
structure RohanFinances where
  food_percent : ℝ
  rent_percent : ℝ
  entertainment_percent : ℝ
  conveyance_percent : ℝ
  taxes_percent : ℝ
  miscellaneous_percent : ℝ
  savings : ℝ

/-- Theorem: Rohan's monthly salary calculation --/
theorem rohan_monthly_salary (r : RohanFinances) 
  (h1 : r.food_percent = 0.40)
  (h2 : r.rent_percent = 0.20)
  (h3 : r.entertainment_percent = 0.10)
  (h4 : r.conveyance_percent = 0.10)
  (h5 : r.taxes_percent = 0.05)
  (h6 : r.miscellaneous_percent = 0.07)
  (h7 : r.savings = 1000) :
  ∃ (salary : ℝ), salary = 12500 ∧ 
    (1 - (r.food_percent + r.rent_percent + r.entertainment_percent + 
          r.conveyance_percent + r.taxes_percent + r.miscellaneous_percent)) * salary = r.savings :=
by sorry


end rohan_monthly_salary_l3632_363292


namespace smallest_a_value_l3632_363297

/-- The smallest possible value of a given the conditions -/
theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b)
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x)) :
  a ≥ 2 * Real.pi - 17 ∧ ∃ (a₀ : ℝ), a₀ = 2 * Real.pi - 17 ∧ 
  (∃ b₀ : ℝ, 0 ≤ b₀ ∧ ∀ x : ℤ, Real.sin (a₀ * ↑x + b₀) = Real.sin (17 * ↑x)) :=
by sorry

end smallest_a_value_l3632_363297


namespace notebook_savings_correct_l3632_363254

def notebook_savings (quantity : ℕ) (original_price : ℝ) (individual_discount_rate : ℝ) (bulk_discount_rate : ℝ) (bulk_discount_threshold : ℕ) : ℝ :=
  let discounted_price := original_price * (1 - individual_discount_rate)
  let total_without_discount := quantity * original_price
  let total_with_individual_discount := quantity * discounted_price
  let final_total := if quantity > bulk_discount_threshold
                     then total_with_individual_discount * (1 - bulk_discount_rate)
                     else total_with_individual_discount
  total_without_discount - final_total

theorem notebook_savings_correct :
  notebook_savings 8 3 0.1 0.05 6 = 3.48 :=
sorry

end notebook_savings_correct_l3632_363254


namespace quadratic_inequality_solution_l3632_363245

theorem quadratic_inequality_solution (x : ℝ) : 
  -3 * x^2 + 5 * x + 4 < 0 ↔ -4/3 < x ∧ x < 1 := by
  sorry

end quadratic_inequality_solution_l3632_363245


namespace tangent_slope_at_point_l3632_363287

/-- The slope of the tangent line to y = x^3 - 4x at (1, -1) is -1 -/
theorem tangent_slope_at_point : 
  let f (x : ℝ) := x^3 - 4*x
  let x₀ : ℝ := 1
  let y₀ : ℝ := -1
  (deriv f) x₀ = -1 := by sorry

end tangent_slope_at_point_l3632_363287


namespace max_value_problem_l3632_363224

theorem max_value_problem (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  ∀ x, x = (a * b * c * d) ^ (1/4) + ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1/2) → x ≤ 1 :=
by sorry

end max_value_problem_l3632_363224


namespace triangle_angle_equality_l3632_363258

theorem triangle_angle_equality (A B C : ℝ) (a b c : ℝ) :
  0 < B ∧ B < π →
  0 < a ∧ 0 < b ∧ 0 < c →
  2 * b * Real.cos B = a * Real.cos C + c * Real.cos A →
  B = π / 3 := by
  sorry

end triangle_angle_equality_l3632_363258


namespace triangle_area_proof_l3632_363290

-- Define the slopes of the two lines
def slope1 : ℝ := 3
def slope2 : ℝ := -1

-- Define the intersection point
def intersection_point : ℝ × ℝ := (5, 3)

-- Define the equation of the third line
def third_line (x y : ℝ) : Prop := x + y = 4

-- Define the area of the triangle
def triangle_area : ℝ := 4

-- Theorem statement
theorem triangle_area_proof :
  ∃ (A B C : ℝ × ℝ),
    -- A is on the line with slope1 and passes through intersection_point
    (A.2 - intersection_point.2 = slope1 * (A.1 - intersection_point.1)) ∧
    -- B is on the line with slope2 and passes through intersection_point
    (B.2 - intersection_point.2 = slope2 * (B.1 - intersection_point.1)) ∧
    -- C is on the third line
    third_line C.1 C.2 ∧
    -- The area of the triangle formed by A, B, and C is equal to triangle_area
    abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2) = triangle_area :=
sorry

end triangle_area_proof_l3632_363290


namespace unique_solution_for_sum_and_product_l3632_363259

theorem unique_solution_for_sum_and_product (x y z : ℝ) :
  x + y + z = 38 →
  x * y * z = 2002 →
  0 < x →
  x ≤ 11 →
  z ≥ 14 →
  x = 11 ∧ y = 13 ∧ z = 14 :=
by sorry

end unique_solution_for_sum_and_product_l3632_363259


namespace b_47_mod_49_l3632_363277

/-- Definition of the sequence b_n -/
def b (n : ℕ) : ℕ := 7^n + 9^n

/-- The remainder of b_47 when divided by 49 is 14 -/
theorem b_47_mod_49 : b 47 % 49 = 14 := by
  sorry

end b_47_mod_49_l3632_363277


namespace range_of_a_l3632_363228

-- Define the propositions
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def Q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(5-2*a))^x > (-(5-2*a))^y

-- State the theorem
theorem range_of_a :
  (∃ a : ℝ, (¬(P a) ∧ Q a) ∨ (P a ∧ ¬(Q a))) →
  (∃ a : ℝ, a ≤ -2 ∧ ∀ b : ℝ, b ≤ -2 → (¬(P b) ∧ Q b) ∨ (P b ∧ ¬(Q b))) :=
by sorry

end range_of_a_l3632_363228


namespace balcony_orchestra_difference_l3632_363226

/-- Represents the number of tickets sold for a theater performance -/
structure TheaterTickets where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total number of tickets sold -/
def totalTickets (t : TheaterTickets) : ℕ := t.orchestra + t.balcony

/-- Calculates the total revenue from ticket sales -/
def totalRevenue (t : TheaterTickets) : ℕ := 12 * t.orchestra + 8 * t.balcony

theorem balcony_orchestra_difference (t : TheaterTickets) :
  totalTickets t = 355 → totalRevenue t = 3320 → t.balcony - t.orchestra = 115 := by
  sorry

#check balcony_orchestra_difference

end balcony_orchestra_difference_l3632_363226


namespace function_always_positive_l3632_363243

/-- A function satisfying the given differential inequality is always positive -/
theorem function_always_positive
  (f : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hf' : Differentiable ℝ (deriv f))
  (h : ∀ x, x * (deriv^[2] f x) + 2 * f x > x^2) :
  ∀ x, f x > 0 := by sorry

end function_always_positive_l3632_363243


namespace turtleneck_discount_theorem_l3632_363282

theorem turtleneck_discount_theorem (C : ℝ) (D : ℝ) : 
  C > 0 →  -- Cost is positive
  (1.50 * C) * (1 - D / 100) = 1.125 * C → -- Equation from profit condition
  D = 25 := by
sorry

end turtleneck_discount_theorem_l3632_363282


namespace limit_function_equals_one_half_l3632_363237

/-- The limit of ((1+8x)/(2+11x))^(1/(x^2+1)) as x approaches 0 is 1/2 -/
theorem limit_function_equals_one_half :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ →
    |(((1 + 8*x) / (2 + 11*x)) ^ (1 / (x^2 + 1))) - (1/2)| < ε := by
  sorry

end limit_function_equals_one_half_l3632_363237


namespace festival_allowance_rate_l3632_363216

/-- The daily rate for a festival allowance given the number of staff members,
    number of days, and total amount. -/
def daily_rate (staff_members : ℕ) (days : ℕ) (total_amount : ℕ) : ℚ :=
  total_amount / (staff_members * days)

/-- Theorem stating that the daily rate for the festival allowance is 110
    given the problem conditions. -/
theorem festival_allowance_rate : 
  daily_rate 20 30 66000 = 110 := by sorry

end festival_allowance_rate_l3632_363216


namespace base7_digit_sum_l3632_363236

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a base-7 number --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem base7_digit_sum :
  let a := toBase10 45
  let b := toBase10 16
  let c := toBase10 12
  let result := toBase7 ((a * b) + c)
  sumOfDigitsBase7 result = 17 := by sorry

end base7_digit_sum_l3632_363236


namespace min_sum_with_reciprocal_constraint_l3632_363263

theorem min_sum_with_reciprocal_constraint (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) : 
  ∀ z w : ℝ, z > 0 → w > 0 → 1/z + 9/w = 1 → x + y ≤ z + w ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 9/b = 1 ∧ a + b = 16 :=
sorry

end min_sum_with_reciprocal_constraint_l3632_363263


namespace positive_sixth_root_of_64_l3632_363203

theorem positive_sixth_root_of_64 (y : ℝ) (h1 : y > 0) (h2 : y^6 = 64) : y = 2 := by
  sorry

end positive_sixth_root_of_64_l3632_363203


namespace max_value_quadratic_function_l3632_363286

/-- Given a > 1, the maximum value of f(x) = -x^2 - 2ax + 1 on the interval [-1,1] is 2a -/
theorem max_value_quadratic_function (a : ℝ) (h : a > 1) :
  ∃ (max : ℝ), max = 2 * a ∧ ∀ x ∈ Set.Icc (-1) 1, -x^2 - 2*a*x + 1 ≤ max :=
sorry

end max_value_quadratic_function_l3632_363286


namespace unique_intersection_point_l3632_363213

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 5*x^2 + 10*x + 20

-- State the theorem
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (-4, -4) :=
sorry

end unique_intersection_point_l3632_363213


namespace factor_expression_l3632_363272

theorem factor_expression (x : ℝ) : 72 * x^5 - 162 * x^9 = -18 * x^5 * (9 * x^4 - 4) := by
  sorry

end factor_expression_l3632_363272


namespace value_of_b_l3632_363242

theorem value_of_b (a b : ℝ) (eq1 : 3 * a + 1 = 1) (eq2 : b - a = 2) : b = 2 := by
  sorry

end value_of_b_l3632_363242


namespace right_triangle_m_values_l3632_363280

/-- A right triangle in a 2D Cartesian coordinate system -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∨
             (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 ∨
             (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

/-- The theorem to be proved -/
theorem right_triangle_m_values (t : RightTriangle) 
    (h1 : t.B.1 - t.A.1 = 1 ∧ t.B.2 - t.A.2 = 1)
    (h2 : t.C.1 - t.A.1 = 2 ∧ ∃ m : ℝ, t.C.2 - t.A.2 = m) :
  ∃ m : ℝ, (t.C.2 - t.A.2 = m ∧ (m = -2 ∨ m = 0)) := by
  sorry


end right_triangle_m_values_l3632_363280


namespace y_intercept_of_specific_line_l3632_363214

/-- A line is defined by its slope and a point it passes through -/
structure Line where
  slope : ℚ
  point : ℚ × ℚ

/-- The y-intercept of a line is the y-coordinate where the line crosses the y-axis -/
def y_intercept (l : Line) : ℚ := 
  l.point.2 - l.slope * l.point.1

theorem y_intercept_of_specific_line : 
  let l : Line := { slope := -3/2, point := (4, 0) }
  y_intercept l = 6 := by
  sorry

#check y_intercept_of_specific_line

end y_intercept_of_specific_line_l3632_363214


namespace english_alphabet_is_set_l3632_363204

-- Define the type for English alphabet letters
inductive EnglishLetter
| A | B | C | D | E | F | G | H | I | J | K | L | M
| N | O | P | Q | R | S | T | U | V | W | X | Y | Z

-- Define the properties of set elements
def isDefinite (x : Type) : Prop := sorry
def isDistinct (x : Type) : Prop := sorry
def isUnordered (x : Type) : Prop := sorry

-- Define what it means to be a valid set
def isValidSet (x : Type) : Prop :=
  isDefinite x ∧ isDistinct x ∧ isUnordered x

-- Theorem stating that the English alphabet forms a set
theorem english_alphabet_is_set :
  isValidSet EnglishLetter :=
sorry

end english_alphabet_is_set_l3632_363204


namespace remainder_theorem_l3632_363256

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 40 * k - 1) :
  (n^2 - 3*n + 5) % 40 = 9 := by sorry

end remainder_theorem_l3632_363256


namespace male_average_tickets_l3632_363270

/-- Proves that the average number of tickets sold by male members is 58,
    given the overall average, female average, and male-to-female ratio. -/
theorem male_average_tickets (total_members : ℕ) (male_members : ℕ) (female_members : ℕ) :
  male_members > 0 →
  female_members = 2 * male_members →
  (male_members * q + female_members * 70) / total_members = 66 →
  total_members = male_members + female_members →
  q = 58 :=
by sorry

end male_average_tickets_l3632_363270


namespace inequality_proof_l3632_363262

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) : 
  (a^2 + b^2 + c^2 + d^2 ≥ 1/4) ∧ 
  (a^2/b + b^2/c + c^2/d + d^2/a ≥ 1) := by
  sorry

end inequality_proof_l3632_363262


namespace constant_term_of_liams_polynomial_l3632_363299

/-- Represents a polynomial with degree 5 -/
structure Poly5 where
  coeffs : Fin 6 → ℝ
  monic : coeffs 5 = 1

/-- The product of two polynomials -/
def poly_product (p q : Poly5) : Fin 11 → ℝ := sorry

theorem constant_term_of_liams_polynomial 
  (serena_poly liam_poly : Poly5)
  (same_constant : serena_poly.coeffs 0 = liam_poly.coeffs 0)
  (positive_constant : serena_poly.coeffs 0 > 0)
  (same_z2_coeff : serena_poly.coeffs 2 = liam_poly.coeffs 2)
  (product : poly_product serena_poly liam_poly = 
    fun i => match i with
    | 0 => 9  | 1 => 5  | 2 => 10 | 3 => 4  | 4 => 9
    | 5 => 6  | 6 => 5  | 7 => 4  | 8 => 3  | 9 => 2
    | 10 => 1
  ) :
  liam_poly.coeffs 0 = 3 := by sorry

end constant_term_of_liams_polynomial_l3632_363299


namespace intersection_k_value_l3632_363283

/-- Given two lines that intersect at x = -15, prove the value of k -/
theorem intersection_k_value (k : ℝ) : 
  (∀ x y : ℝ, -3 * x + y = k ∧ 0.3 * x + y = 10) →
  (∃ y : ℝ, -3 * (-15) + y = k ∧ 0.3 * (-15) + y = 10) →
  k = 59.5 := by
sorry

end intersection_k_value_l3632_363283


namespace legacy_gain_satisfies_conditions_l3632_363285

/-- The legacy gain received by Ms. Emily Smith -/
def legacy_gain : ℝ := 46345

/-- The federal tax rate as a decimal -/
def federal_tax_rate : ℝ := 0.25

/-- The regional tax rate as a decimal -/
def regional_tax_rate : ℝ := 0.15

/-- The total amount of taxes paid -/
def total_taxes_paid : ℝ := 16800

/-- Theorem stating that the legacy gain satisfies the given conditions -/
theorem legacy_gain_satisfies_conditions :
  federal_tax_rate * legacy_gain + 
  regional_tax_rate * (legacy_gain - federal_tax_rate * legacy_gain) = 
  total_taxes_paid := by sorry

end legacy_gain_satisfies_conditions_l3632_363285


namespace quadratic_factorization_l3632_363257

theorem quadratic_factorization (m : ℝ) : 2 * m^2 - 12 * m + 18 = 2 * (m - 3)^2 := by
  sorry

end quadratic_factorization_l3632_363257


namespace line_slope_intercept_sum_l3632_363220

/-- Given a line passing through two points, prove that the sum of its slope and y-intercept is -5/2 --/
theorem line_slope_intercept_sum (m b : ℚ) : 
  ((-1 : ℚ) = m * (1/2) + b) → 
  (2 = m * (-1/2) + b) → 
  m + b = -5/2 := by sorry

end line_slope_intercept_sum_l3632_363220


namespace decimal_to_binary_21_l3632_363278

theorem decimal_to_binary_21 : 
  (21 : ℕ) = (1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) :=
by sorry

end decimal_to_binary_21_l3632_363278


namespace quadratic_roots_sum_l3632_363244

theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, a * x^2 + b * x - 2 = 0 ↔ x = -2 ∨ x = -1/4) → 
  a + b = -13 := by
sorry

end quadratic_roots_sum_l3632_363244


namespace equation_solution_exists_l3632_363235

theorem equation_solution_exists : ∃ (MA TE TI KA : ℕ),
  MA < 10 ∧ TE < 10 ∧ TI < 10 ∧ KA < 10 ∧
  MA ≠ TE ∧ MA ≠ TI ∧ MA ≠ KA ∧ TE ≠ TI ∧ TE ≠ KA ∧ TI ≠ KA ∧
  MA * TE * MA * TI * KA = 2016000 := by
  sorry

end equation_solution_exists_l3632_363235


namespace parallel_vectors_m_value_l3632_363223

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (-1, m)
  are_parallel a b → m = 2 := by
  sorry

end parallel_vectors_m_value_l3632_363223


namespace absolute_difference_of_roots_l3632_363241

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 7*x + 12 = 0

-- Define the roots of the equation
noncomputable def r₁ : ℝ := sorry
noncomputable def r₂ : ℝ := sorry

-- State the theorem
theorem absolute_difference_of_roots : 
  quadratic_equation r₁ ∧ quadratic_equation r₂ → |r₁ - r₂| = 1 := by sorry

end absolute_difference_of_roots_l3632_363241
