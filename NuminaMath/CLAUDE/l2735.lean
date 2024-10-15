import Mathlib

namespace NUMINAMATH_CALUDE_num_possible_strings_l2735_273575

/-- Represents the allowed moves in the string transformation game -/
inductive Move
| HM_to_MH
| MT_to_TM
| TH_to_HT

/-- The initial string in the game -/
def initial_string : String := "HHMMMMTT"

/-- The number of H's in the initial string -/
def num_H : Nat := 2

/-- The number of M's in the initial string -/
def num_M : Nat := 4

/-- The number of T's in the initial string -/
def num_T : Nat := 2

/-- The total length of the string -/
def total_length : Nat := num_H + num_M + num_T

/-- Theorem stating that the number of possible strings after zero or more moves
    is equal to the number of ways to choose num_M positions out of total_length positions -/
theorem num_possible_strings :
  (Nat.choose total_length num_M) = 70 := by sorry

end NUMINAMATH_CALUDE_num_possible_strings_l2735_273575


namespace NUMINAMATH_CALUDE_perpendicular_vector_implies_y_equals_five_l2735_273540

/-- Given points A and B, and vector a, proves that if AB is perpendicular to a, then y = 5 -/
theorem perpendicular_vector_implies_y_equals_five (A B : ℝ × ℝ) (a : ℝ × ℝ) :
  A = (10, 1) →
  B.1 = 2 →
  a = (1, 2) →
  (B.1 - A.1) * a.1 + (B.2 - A.2) * a.2 = 0 →
  B.2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_implies_y_equals_five_l2735_273540


namespace NUMINAMATH_CALUDE_division_problem_l2735_273514

theorem division_problem (L S Q : ℕ) : 
  L - S = 1515 →
  L = 1600 →
  L = Q * S + 15 →
  Q = 18 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2735_273514


namespace NUMINAMATH_CALUDE_tangent_parallel_point_l2735_273529

theorem tangent_parallel_point (x y : ℝ) : 
  y = Real.exp x → -- Point A (x, y) is on the curve y = e^x
  (Real.exp x) = 1 → -- Tangent at A is parallel to x - y + 3 = 0 (slope = 1)
  x = 0 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_parallel_point_l2735_273529


namespace NUMINAMATH_CALUDE_triangle_theorem_l2735_273551

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition from the problem -/
def satisfies_condition (t : Triangle) : Prop :=
  t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C - t.b - t.c = 0

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) 
  (h : satisfies_condition t) : 
  t.A = π / 3 ∧ 
  (t.a = Real.sqrt 3 → 
    ∀ (area : ℝ), area ≤ 3 * Real.sqrt 3 / 4 → 
      ∃ (t' : Triangle), satisfies_condition t' ∧ t'.a = Real.sqrt 3 ∧ 
        area = 1 / 2 * t'.b * t'.c * Real.sin t'.A) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2735_273551


namespace NUMINAMATH_CALUDE_vertical_line_slope_undefined_l2735_273594

/-- The slope of a line passing through two distinct points with the same x-coordinate does not exist -/
theorem vertical_line_slope_undefined (y : ℝ) (h : y ≠ -3) :
  ¬∃ m : ℝ, ∀ x, x = 5 → (y - (-3)) = m * (x - 5) :=
sorry

end NUMINAMATH_CALUDE_vertical_line_slope_undefined_l2735_273594


namespace NUMINAMATH_CALUDE_seashells_per_day_l2735_273532

/-- 
Given a 5-day beach trip where 35 seashells were found in total, 
and assuming an equal number of seashells were found each day, 
prove that the number of seashells found per day is 7.
-/
theorem seashells_per_day 
  (days : ℕ) 
  (total_seashells : ℕ) 
  (seashells_per_day : ℕ) 
  (h1 : days = 5) 
  (h2 : total_seashells = 35) 
  (h3 : seashells_per_day * days = total_seashells) : 
  seashells_per_day = 7 := by
sorry

end NUMINAMATH_CALUDE_seashells_per_day_l2735_273532


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2735_273556

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = q * a n ∧ a n > 0

theorem geometric_sequence_formula 
  (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_first : a 1 = 1) 
  (h_second : ∃ x : ℝ, a 2 = x + 1) 
  (h_third : ∃ x : ℝ, a 3 = 2 * x + 5) : 
  ∀ n : ℕ, a n = 3^(n - 1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2735_273556


namespace NUMINAMATH_CALUDE_final_amount_calculation_l2735_273549

/-- Calculates the final amount paid after applying a discount based on complete hundreds spent. -/
theorem final_amount_calculation (purchase_amount : ℕ) (discount_per_hundred : ℕ) : 
  purchase_amount = 250 ∧ discount_per_hundred = 10 →
  purchase_amount - (purchase_amount / 100) * discount_per_hundred = 230 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_calculation_l2735_273549


namespace NUMINAMATH_CALUDE_crunch_difference_l2735_273535

/-- Given that Zachary did 17 crunches and David did 4 crunches,
    prove that David did 13 less crunches than Zachary. -/
theorem crunch_difference (zachary_crunches : ℕ) (david_crunches : ℕ)
  (h1 : zachary_crunches = 17)
  (h2 : david_crunches = 4) :
  zachary_crunches - david_crunches = 13 := by
  sorry

end NUMINAMATH_CALUDE_crunch_difference_l2735_273535


namespace NUMINAMATH_CALUDE_linear_equation_implies_k_equals_one_l2735_273563

/-- A function that represents the linearity condition of an equation -/
def is_linear_equation (k : ℝ) : Prop :=
  (k + 1 ≠ 0) ∧ (|k| = 1)

/-- Theorem stating that if (k+1)x + 8y^|k| + 3 = 0 is a linear equation in x and y, then k = 1 -/
theorem linear_equation_implies_k_equals_one :
  is_linear_equation k → k = 1 := by sorry

end NUMINAMATH_CALUDE_linear_equation_implies_k_equals_one_l2735_273563


namespace NUMINAMATH_CALUDE_election_combinations_theorem_l2735_273598

/-- Represents a club with members of different genders and ages -/
structure Club where
  total_members : Nat
  girls : Nat
  boys : Nat
  girls_age_order : Fin girls → Nat
  boys_age_order : Fin boys → Nat

/-- Represents the election rules for the club -/
structure ElectionRules where
  president_must_be_girl : Bool
  vp_must_be_boy : Bool
  vp_younger_than_president : Bool

/-- Calculates the number of ways to elect a president and vice-president -/
def election_combinations (club : Club) (rules : ElectionRules) : Nat :=
  sorry

/-- Theorem stating the number of election combinations for the given club and rules -/
theorem election_combinations_theorem (club : Club) (rules : ElectionRules) :
  club.total_members = 25 ∧
  club.girls = 13 ∧
  club.boys = 12 ∧
  rules.president_must_be_girl = true ∧
  rules.vp_must_be_boy = true ∧
  rules.vp_younger_than_president = true →
  election_combinations club rules = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_election_combinations_theorem_l2735_273598


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2735_273530

theorem fraction_evaluation (a b : ℚ) (ha : a = 5) (hb : b = -2) : 
  5 / (a + b) = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2735_273530


namespace NUMINAMATH_CALUDE_min_value_expression_l2735_273572

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (min : ℝ), min = Real.sqrt 6 ∧
  ∀ (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0),
    x^2 + 2*y^2 + 1/x^2 + 2*y/x ≥ min ∧
    ∃ (a₀ b₀ : ℝ) (ha₀ : a₀ ≠ 0) (hb₀ : b₀ ≠ 0),
      a₀^2 + 2*b₀^2 + 1/a₀^2 + 2*b₀/a₀ = min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2735_273572


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2735_273561

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 1 + a 9 = 10)
  (h_second : a 2 = -1) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2735_273561


namespace NUMINAMATH_CALUDE_find_a_and_b_l2735_273512

/-- Set A defined by the equation ax - y² + b = 0 -/
def A (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 - p.2^2 + b = 0}

/-- Set B defined by the equation x² - ay - b = 0 -/
def B (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - a * p.2 - b = 0}

/-- Theorem stating that a = -3 and b = 7 given the conditions -/
theorem find_a_and_b :
  ∃ (a b : ℝ), (1, 2) ∈ A a b ∩ B a b ∧ a = -3 ∧ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_a_and_b_l2735_273512


namespace NUMINAMATH_CALUDE_show_episodes_count_l2735_273522

/-- The number of episodes watched on Mondays each week -/
def monday_episodes : ℕ := 1

/-- The number of episodes watched on Wednesdays each week -/
def wednesday_episodes : ℕ := 2

/-- The number of weeks it takes to watch the whole series -/
def total_weeks : ℕ := 67

/-- The total number of episodes in the show -/
def total_episodes : ℕ := 201

theorem show_episodes_count : 
  monday_episodes + wednesday_episodes * total_weeks = total_episodes := by
  sorry

end NUMINAMATH_CALUDE_show_episodes_count_l2735_273522


namespace NUMINAMATH_CALUDE_range_of_a_l2735_273543

-- Define the feasible region
def feasible_region (x y : ℝ) : Prop :=
  2 * x + y ≥ 4 ∧ x - y ≥ 1 ∧ x - 2 * y ≤ 2

-- Define the function z
def z (a x y : ℝ) : ℝ := a * x + y

-- Define the minimum point
def min_point : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ,
  (∀ x y : ℝ, feasible_region x y → z a x y ≥ z a (min_point.1) (min_point.2)) →
  (∃ x y : ℝ, feasible_region x y ∧ z a x y = z a (min_point.1) (min_point.2) → (x, y) = min_point) →
  -1/2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2735_273543


namespace NUMINAMATH_CALUDE_inverse_theorem_not_exists_l2735_273578

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- side lengths
  (α β γ : ℝ)  -- angles

-- Define congruence for triangles
def isCongruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Define equality of corresponding angles
def hasEqualAngles (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- Theorem statement
theorem inverse_theorem_not_exists :
  ¬(∀ t1 t2 : Triangle, hasEqualAngles t1 t2 → isCongruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_inverse_theorem_not_exists_l2735_273578


namespace NUMINAMATH_CALUDE_number_of_divisors_3003_l2735_273517

theorem number_of_divisors_3003 : Finset.card (Nat.divisors 3003) = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_3003_l2735_273517


namespace NUMINAMATH_CALUDE_money_distribution_l2735_273593

theorem money_distribution (total money_ac money_bc : ℕ) 
  (h1 : total = 600)
  (h2 : money_ac = 250)
  (h3 : money_bc = 450) :
  ∃ (a b c : ℕ), a + b + c = total ∧ a + c = money_ac ∧ b + c = money_bc ∧ c = 100 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l2735_273593


namespace NUMINAMATH_CALUDE_x_equals_y_l2735_273506

theorem x_equals_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) : x = y :=
by sorry

end NUMINAMATH_CALUDE_x_equals_y_l2735_273506


namespace NUMINAMATH_CALUDE_range_of_a_l2735_273508

-- Define the sets A and B
def A : Set ℝ := {x | x < 3}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∪ B a = Set.univ → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2735_273508


namespace NUMINAMATH_CALUDE_intersection_complement_when_m_2_union_equals_B_iff_l2735_273560

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 6}

theorem intersection_complement_when_m_2 :
  A ∩ (Bᶜ 2) = {x | -1 ≤ x ∧ x < 2} := by sorry

theorem union_equals_B_iff (m : ℝ) :
  A ∪ B m = B m ↔ -3 ≤ m ∧ m ≤ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_m_2_union_equals_B_iff_l2735_273560


namespace NUMINAMATH_CALUDE_fraction_equality_l2735_273504

theorem fraction_equality (x : ℝ) (h : x ≠ 1) : -2 / (2 * x - 2) = 1 / (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2735_273504


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2735_273531

theorem necessary_but_not_sufficient :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ∧
  (∃ x y : ℝ, x + y > 2 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2735_273531


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2735_273581

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2735_273581


namespace NUMINAMATH_CALUDE_unique_sequence_exists_l2735_273526

def sequence_condition (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ ∀ n : ℕ, n ≥ 1 → (a (n + 1))^3 = a n * a (n + 2) - 1

theorem unique_sequence_exists : ∃! a : ℕ → ℕ, sequence_condition a := by
  sorry

end NUMINAMATH_CALUDE_unique_sequence_exists_l2735_273526


namespace NUMINAMATH_CALUDE_phone_number_probability_l2735_273590

/-- Represents the possible prefixes for the phone number -/
def prefixes : Finset String := {"296", "299", "298"}

/-- Represents the digits for the remaining part of the phone number -/
def remainingDigits : Finset Char := {'0', '1', '6', '7', '9'}

/-- The total number of digits in the phone number -/
def totalDigits : Nat := 8

theorem phone_number_probability :
  (Finset.card prefixes * (Finset.card remainingDigits).factorial : ℚ)⁻¹ = 1 / 360 := by
  sorry

end NUMINAMATH_CALUDE_phone_number_probability_l2735_273590


namespace NUMINAMATH_CALUDE_parabola_unique_intersection_l2735_273592

/-- A parabola defined by y = x^2 - 6x + m -/
def parabola (x m : ℝ) : ℝ := x^2 - 6*x + m

/-- Condition for the parabola to intersect the x-axis -/
def intersects_x_axis (m : ℝ) : Prop :=
  ∃ x, parabola x m = 0

/-- Condition for the parabola to have exactly one intersection with the x-axis -/
def unique_intersection (m : ℝ) : Prop :=
  ∃! x, parabola x m = 0

theorem parabola_unique_intersection :
  ∃! m, unique_intersection m ∧ m = 9 :=
sorry

end NUMINAMATH_CALUDE_parabola_unique_intersection_l2735_273592


namespace NUMINAMATH_CALUDE_even_function_shift_l2735_273591

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the property of being an even function
def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- Theorem statement
theorem even_function_shift (a : ℝ) :
  is_even (fun x ↦ f (x + a)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_shift_l2735_273591


namespace NUMINAMATH_CALUDE_largest_two_three_digit_multiples_sum_l2735_273500

theorem largest_two_three_digit_multiples_sum : ∃ (a b : ℕ), 
  (a > 0 ∧ a < 100 ∧ a % 5 = 0 ∧ ∀ x : ℕ, x > 0 ∧ x < 100 ∧ x % 5 = 0 → x ≤ a) ∧
  (b > 0 ∧ b < 1000 ∧ b % 7 = 0 ∧ ∀ y : ℕ, y > 0 ∧ y < 1000 ∧ y % 7 = 0 → y ≤ b) ∧
  a + b = 1089 := by
sorry

end NUMINAMATH_CALUDE_largest_two_three_digit_multiples_sum_l2735_273500


namespace NUMINAMATH_CALUDE_polygon_sides_l2735_273537

theorem polygon_sides (interior_angle : ℝ) (h : interior_angle = 140) :
  (360 : ℝ) / (180 - interior_angle) = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2735_273537


namespace NUMINAMATH_CALUDE_ellipse_condition_l2735_273533

/-- The equation represents an ellipse with foci on the x-axis -/
def is_ellipse_on_x_axis (n : ℝ) : Prop :=
  2 - n > 0 ∧ n + 1 > 0 ∧ 2 - n > n + 1

/-- The condition -1 < n < 2 is sufficient but not necessary for the equation to represent an ellipse with foci on the x-axis -/
theorem ellipse_condition (n : ℝ) :
  ((-1 < n ∧ n < 2) → is_ellipse_on_x_axis n) ∧
  ¬(is_ellipse_on_x_axis n → (-1 < n ∧ n < 2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2735_273533


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2735_273554

/-- The hyperbola C: x² - y² = a² intersects with the directrix of the parabola y² = 16x 
    at two points with distance 4√3 between them. 
    This theorem states that the length of the real axis of hyperbola C is 4. -/
theorem hyperbola_real_axis_length (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.1 = -4 ∧ A.1^2 - A.2^2 = a^2) ∧ 
    (B.1 = -4 ∧ B.1^2 - B.2^2 = a^2) ∧ 
    (A.2 - B.2)^2 = 48) →
  2 * a = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2735_273554


namespace NUMINAMATH_CALUDE_complex_number_equality_l2735_273545

theorem complex_number_equality (a b : ℝ) (i : ℂ) : 
  i * i = -1 →
  (a - 2 * i) * i = b - i →
  (a + b * i : ℂ) = -1 + 2 * i := by
sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2735_273545


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2735_273536

theorem triangle_perimeter (a : ℕ) (h1 : 2 < a) (h2 : a < 8) (h3 : Even a) :
  2 + 6 + a = 14 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2735_273536


namespace NUMINAMATH_CALUDE_prove_initial_stock_l2735_273562

-- Define the total number of books sold
def books_sold : ℕ := 272

-- Define the percentage of books sold as a rational number
def percentage_sold : ℚ := 19.42857142857143 / 100

-- Define the initial stock of books
def initial_stock : ℕ := 1400

-- Theorem statement
theorem prove_initial_stock : 
  (books_sold : ℚ) / initial_stock = percentage_sold :=
by sorry

end NUMINAMATH_CALUDE_prove_initial_stock_l2735_273562


namespace NUMINAMATH_CALUDE_at_least_two_same_connections_l2735_273525

-- Define the type for interns
def Intern : Type := ℕ

-- Define the knowing relation
def knows : Intern → Intern → Prop := sorry

-- The number of interns
def num_interns : ℕ := 80

-- The knowing relation is symmetric
axiom knows_symmetric : ∀ (a b : Intern), knows a b ↔ knows b a

-- Function to count how many interns a given intern knows
def num_known (i : Intern) : ℕ := sorry

-- Theorem statement
theorem at_least_two_same_connections : 
  ∃ (i j : Intern), i ≠ j ∧ num_known i = num_known j :=
sorry

end NUMINAMATH_CALUDE_at_least_two_same_connections_l2735_273525


namespace NUMINAMATH_CALUDE_triangle_inequality_l2735_273552

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (a + b + c)^2 < 4 * (a * b + b * c + c * a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2735_273552


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2735_273516

-- Define the quadratic function f(x)
def f (x : ℝ) : ℝ := -x^2 + 2*x + 15

-- Define g(x) in terms of f(x) and a
def g (a x : ℝ) : ℝ := (2 - 2*a)*x - f x

-- Theorem statement
theorem quadratic_function_properties :
  -- f(x) has vertex (1, 16)
  (f 1 = 16 ∧ ∀ x, f x ≤ f 1) ∧
  -- The roots of f(x) are 8 units apart
  (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 8) →
  -- 1. f(x) = -x^2 + 2x + 15
  (∀ x, f x = -x^2 + 2*x + 15) ∧
  -- 2. g(x) is monotonically increasing on [0, 2] iff a ≤ 0
  (∀ a, (∀ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 → g a x₁ < g a x₂) ↔ a ≤ 0) ∧
  -- 3. Minimum value of g(x) on [0, 2]
  (∀ a, (∃ m, ∀ x, 0 ≤ x ∧ x ≤ 2 → m ≤ g a x ∧
    ((a > 2 → m = -4*a - 11) ∧
     (a < 0 → m = -15) ∧
     (0 ≤ a ∧ a ≤ 2 → m = -a^2 - 15)))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2735_273516


namespace NUMINAMATH_CALUDE_solve_system_l2735_273570

theorem solve_system (s t : ℚ) 
  (eq1 : 7 * s + 8 * t = 150)
  (eq2 : s = 2 * t + 3) : 
  s = 162 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2735_273570


namespace NUMINAMATH_CALUDE_set_inclusion_l2735_273542

-- Define the sets M, N, and P
def M : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 < 1}

def N : Set (ℝ × ℝ) := {p | Real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + 
                             Real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * Real.sqrt 2}

def P : Set (ℝ × ℝ) := {p | abs (p.1 + p.2) < 1 ∧ abs p.1 < 1 ∧ abs p.2 < 1}

-- State the theorem
theorem set_inclusion : M ⊆ P ∧ P ⊆ N := by sorry

end NUMINAMATH_CALUDE_set_inclusion_l2735_273542


namespace NUMINAMATH_CALUDE_cubic_factorization_l2735_273538

theorem cubic_factorization (x : ℝ) : 4 * x^3 - 4 * x^2 + x = x * (2 * x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2735_273538


namespace NUMINAMATH_CALUDE_plot_area_calculation_l2735_273513

/-- Represents the area of a rectangular plot of land in acres, given its dimensions in miles. -/
def plot_area (length width : ℝ) : ℝ :=
  length * width * 640

/-- Theorem stating that a rectangular plot of land with dimensions 20 miles by 30 miles has an area of 384000 acres. -/
theorem plot_area_calculation :
  plot_area 30 20 = 384000 := by
  sorry


end NUMINAMATH_CALUDE_plot_area_calculation_l2735_273513


namespace NUMINAMATH_CALUDE_three_card_selection_l2735_273555

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (face_cards_per_suit : Nat)

/-- Calculates the number of ways to choose 3 cards from a deck
    such that all three cards are of different suits and one is a face card -/
def choose_three_cards (d : Deck) : Nat :=
  d.suits * d.face_cards_per_suit * (d.suits - 1).choose 2 * (d.cards_per_suit ^ 2)

/-- Theorem stating the number of ways to choose 3 cards from a standard deck
    with the given conditions -/
theorem three_card_selection (d : Deck) 
  (h1 : d.cards = 52)
  (h2 : d.suits = 4)
  (h3 : d.cards_per_suit = 13)
  (h4 : d.face_cards_per_suit = 3) :
  choose_three_cards d = 6084 := by
  sorry

#eval choose_three_cards { cards := 52, suits := 4, cards_per_suit := 13, face_cards_per_suit := 3 }

end NUMINAMATH_CALUDE_three_card_selection_l2735_273555


namespace NUMINAMATH_CALUDE_stratified_sampling_third_year_l2735_273520

theorem stratified_sampling_third_year 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (first_year : ℕ) 
  (second_year : ℕ) 
  (h1 : total_students = 2400) 
  (h2 : sample_size = 120) 
  (h3 : first_year = 760) 
  (h4 : second_year = 840) : 
  (total_students - first_year - second_year) * sample_size / total_students = 40 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_year_l2735_273520


namespace NUMINAMATH_CALUDE_calculate_expression_l2735_273521

theorem calculate_expression : 15 * 30 + 45 * 15 = 1125 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2735_273521


namespace NUMINAMATH_CALUDE_justin_flower_gathering_time_l2735_273580

/-- Proves that Justin has been gathering flowers for 1 hour given the problem conditions -/
theorem justin_flower_gathering_time :
  let classmates : ℕ := 30
  let time_per_flower : ℕ := 10  -- minutes
  let lost_flowers : ℕ := 3
  let remaining_time : ℕ := 210  -- minutes
  let total_flowers_needed : ℕ := classmates
  let remaining_flowers : ℕ := remaining_time / time_per_flower + lost_flowers
  let gathered_flowers : ℕ := total_flowers_needed - remaining_flowers
  let gathering_time : ℕ := gathered_flowers * time_per_flower
  gathering_time / 60 = 1  -- hours
  := by sorry

end NUMINAMATH_CALUDE_justin_flower_gathering_time_l2735_273580


namespace NUMINAMATH_CALUDE_sum_x_y_z_l2735_273528

def x : ℕ := (List.range 11).map (· + 30) |>.sum

def y : ℕ := (List.range 11).map (· + 30) |>.filter (· % 2 = 0) |>.length

def z : ℕ := (List.range 11).map (· + 30) |>.filter (· % 2 ≠ 0) |>.prod

theorem sum_x_y_z : x + y + z = 51768016 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_z_l2735_273528


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l2735_273518

theorem profit_percentage_calculation (cost_price selling_price : ℝ) :
  cost_price = 500 ∧ selling_price = 800 →
  (selling_price - cost_price) / cost_price * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l2735_273518


namespace NUMINAMATH_CALUDE_total_pencils_donna_marcia_l2735_273539

/-- The number of pencils Cindi bought -/
def cindi_pencils : ℕ := 60

/-- The number of pencils Marcia bought -/
def marcia_pencils : ℕ := 2 * cindi_pencils

/-- The number of pencils Donna bought -/
def donna_pencils : ℕ := 3 * marcia_pencils

/-- Theorem: The total number of pencils bought by Donna and Marcia is 480 -/
theorem total_pencils_donna_marcia : donna_pencils + marcia_pencils = 480 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_donna_marcia_l2735_273539


namespace NUMINAMATH_CALUDE_exponent_equality_l2735_273524

theorem exponent_equality : 8^5 * 3^5 * 8^3 * 3^7 = 8^8 * 3^12 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l2735_273524


namespace NUMINAMATH_CALUDE_floor_length_proof_l2735_273574

/-- Proves that the length of a rectangular floor is 24 meters given specific conditions -/
theorem floor_length_proof (width : ℝ) (square_size : ℝ) (total_cost : ℝ) (square_cost : ℝ) :
  width = 64 →
  square_size = 8 →
  total_cost = 576 →
  square_cost = 24 →
  (total_cost / square_cost) * square_size * square_size / width = 24 := by
sorry

end NUMINAMATH_CALUDE_floor_length_proof_l2735_273574


namespace NUMINAMATH_CALUDE_equation_solution_l2735_273577

theorem equation_solution (a : ℝ) :
  (a ≠ 0 → ∃! x : ℝ, x ≠ 0 ∧ x ≠ a ∧ 3 * x^2 + 2 * a * x - a^2 = Real.log ((x - a) / (2 * x))) ∧
  (a = 0 → ¬∃ x : ℝ, x ≠ 0 ∧ 3 * x^2 = Real.log (1 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2735_273577


namespace NUMINAMATH_CALUDE_w_change_factor_l2735_273595

theorem w_change_factor (w w' m z : ℝ) (h_pos_m : m > 0) (h_pos_z : z > 0) :
  let q := 5 * w / (4 * m * z^2)
  let q' := 5 * w' / (4 * (2 * m) * (3 * z)^2)
  q' = 0.2222222222222222 * q → w' = 4 * w := by
  sorry

end NUMINAMATH_CALUDE_w_change_factor_l2735_273595


namespace NUMINAMATH_CALUDE_four_students_three_lectures_l2735_273571

/-- The number of ways students can choose lectures -/
def lecture_choices (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: 4 students choosing from 3 lectures results in 81 different selections -/
theorem four_students_three_lectures : 
  lecture_choices 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_four_students_three_lectures_l2735_273571


namespace NUMINAMATH_CALUDE_find_divisor_l2735_273502

theorem find_divisor (dividend quotient : ℕ) (h1 : dividend = 62976) (h2 : quotient = 123) :
  dividend / quotient = 512 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2735_273502


namespace NUMINAMATH_CALUDE_function_properties_l2735_273519

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.cos x * (2 * Real.sqrt 3 * Real.sin x - Real.cos x) + a * (Real.sin x)^2

theorem function_properties (a : ℝ) :
  f a (Real.pi / 12) = 0 →
  (∃ T > 0, ∀ x, f a (x + T) = f a x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f a (y + S) ≠ f a y) ∧
  T = Real.pi ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f a x ≤ Real.sqrt 3) ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f a x ≥ -2) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f a x = Real.sqrt 3) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f a x = -2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2735_273519


namespace NUMINAMATH_CALUDE_bob_age_proof_l2735_273596

/-- Bob's age in years -/
def bob_age : ℝ := 51.25

/-- Jim's age in years -/
def jim_age : ℝ := 75 - bob_age

/-- Theorem stating Bob's age given the conditions -/
theorem bob_age_proof :
  (bob_age = 3 * jim_age - 20) ∧
  (bob_age + jim_age = 75) →
  bob_age = 51.25 := by
sorry

end NUMINAMATH_CALUDE_bob_age_proof_l2735_273596


namespace NUMINAMATH_CALUDE_pokemon_card_difference_l2735_273515

theorem pokemon_card_difference : ∀ (orlando_cards : ℕ),
  orlando_cards > 6 →
  6 + orlando_cards + 3 * orlando_cards = 38 →
  orlando_cards - 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_pokemon_card_difference_l2735_273515


namespace NUMINAMATH_CALUDE_milk_cartons_accepted_l2735_273507

theorem milk_cartons_accepted (total_cartons : ℕ) (num_customers : ℕ) (damaged_per_customer : ℕ) 
  (h1 : total_cartons = 400)
  (h2 : num_customers = 4)
  (h3 : damaged_per_customer = 60) :
  (total_cartons / num_customers - damaged_per_customer) * num_customers = 160 :=
by sorry

end NUMINAMATH_CALUDE_milk_cartons_accepted_l2735_273507


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2735_273527

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_problem (x y : ℝ → ℝ) 
  (h_prop : InverselyProportional x y) 
  (h_init : x 8 = 40 ∧ y 8 = 8) :
  x 10 = 32 ∧ y 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2735_273527


namespace NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l2735_273503

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a^2 - |b|

-- Theorem to prove
theorem otimes_neg_two_neg_one : otimes (-2) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l2735_273503


namespace NUMINAMATH_CALUDE_integral_f_equals_344_over_15_l2735_273566

-- Define the function to be integrated
def f (x : ℝ) : ℝ := (x^2 + 2*x - 3) * (4*x^2 - x + 1)

-- State the theorem
theorem integral_f_equals_344_over_15 : 
  ∫ x in (0)..(2), f x = 344 / 15 := by sorry

end NUMINAMATH_CALUDE_integral_f_equals_344_over_15_l2735_273566


namespace NUMINAMATH_CALUDE_sum_30_45_base3_l2735_273547

/-- Converts a natural number to its base 3 representation as a list of digits -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec go (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc
    else go (m / 3) ((m % 3) :: acc)
  go n []

/-- Theorem: The sum of 30 and 45 in base 10 is equal to 22010 in base 3 -/
theorem sum_30_45_base3 : toBase3 (30 + 45) = [2, 2, 0, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_sum_30_45_base3_l2735_273547


namespace NUMINAMATH_CALUDE_sum_of_remaining_digits_l2735_273501

theorem sum_of_remaining_digits 
  (total_count : Nat) 
  (known_count : Nat) 
  (total_average : ℚ) 
  (known_average : ℚ) 
  (h1 : total_count = 20) 
  (h2 : known_count = 14) 
  (h3 : total_average = 500) 
  (h4 : known_average = 390) :
  (total_count : ℚ) * total_average - (known_count : ℚ) * known_average = 4540 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_remaining_digits_l2735_273501


namespace NUMINAMATH_CALUDE_ramon_twice_loui_age_l2735_273588

/-- The age of Loui today -/
def loui_age : ℕ := 23

/-- The age of Ramon today -/
def ramon_age : ℕ := 26

/-- The number of years until Ramon is twice as old as Loui is today -/
def years_until_double : ℕ := 20

/-- Theorem stating that in 'years_until_double' years, Ramon will be twice as old as Loui is today -/
theorem ramon_twice_loui_age : 
  ramon_age + years_until_double = 2 * loui_age := by
  sorry

end NUMINAMATH_CALUDE_ramon_twice_loui_age_l2735_273588


namespace NUMINAMATH_CALUDE_binomial_1409_1_l2735_273511

theorem binomial_1409_1 : (1409 : ℕ).choose 1 = 1409 := by sorry

end NUMINAMATH_CALUDE_binomial_1409_1_l2735_273511


namespace NUMINAMATH_CALUDE_det_trig_matrix_zero_l2735_273541

theorem det_trig_matrix_zero (a c : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![1, Real.sin (a + c), Real.sin a; 
                                        Real.sin (a + c), 1, Real.sin c; 
                                        Real.sin a, Real.sin c, 1]
  Matrix.det M = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_trig_matrix_zero_l2735_273541


namespace NUMINAMATH_CALUDE_cubic_polynomial_interpolation_l2735_273544

-- Define the set of cubic polynomials over ℝ
def CubicPolynomial : Type := ℝ → ℝ

-- Define the property of being a cubic polynomial
def IsCubicPolynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x : ℝ, p x = a * x^3 + b * x^2 + c * x + d

-- Theorem statement
theorem cubic_polynomial_interpolation
  (P Q R : CubicPolynomial)
  (hP : IsCubicPolynomial P)
  (hQ : IsCubicPolynomial Q)
  (hR : IsCubicPolynomial R)
  (h_order : ∀ x : ℝ, P x ≤ Q x ∧ Q x ≤ R x)
  (h_equal : ∃ x₀ : ℝ, P x₀ = R x₀) :
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ ∀ x : ℝ, Q x = k * P x + (1 - k) * R x :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_interpolation_l2735_273544


namespace NUMINAMATH_CALUDE_jennifer_remaining_money_l2735_273584

def total_money : ℚ := 180

def sandwich_fraction : ℚ := 1/5
def museum_fraction : ℚ := 1/6
def book_fraction : ℚ := 1/2

def remaining_money : ℚ := total_money - (sandwich_fraction * total_money + museum_fraction * total_money + book_fraction * total_money)

theorem jennifer_remaining_money :
  remaining_money = 24 := by sorry

end NUMINAMATH_CALUDE_jennifer_remaining_money_l2735_273584


namespace NUMINAMATH_CALUDE_deck_cost_l2735_273585

/-- The cost of the deck of playing cards given the allowances and sticker purchases -/
theorem deck_cost (lola_allowance dora_allowance : ℕ)
                  (sticker_boxes : ℕ)
                  (dora_sticker_packs : ℕ)
                  (h1 : lola_allowance = 9)
                  (h2 : dora_allowance = 9)
                  (h3 : sticker_boxes = 2)
                  (h4 : dora_sticker_packs = 2) :
  let total_allowance := lola_allowance + dora_allowance
  let total_sticker_packs := 2 * dora_sticker_packs
  let sticker_cost := sticker_boxes * 2
  total_allowance - sticker_cost = 10 := by
sorry

end NUMINAMATH_CALUDE_deck_cost_l2735_273585


namespace NUMINAMATH_CALUDE_village_population_percentage_l2735_273565

theorem village_population_percentage : 
  let part : ℕ := 23040
  let total : ℕ := 38400
  let percentage : ℚ := (part : ℚ) / (total : ℚ) * 100
  percentage = 60 := by sorry

end NUMINAMATH_CALUDE_village_population_percentage_l2735_273565


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2735_273546

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where:
    - The distance from the focus to the asymptote is 2√3
    - The minimum distance from a point on the right branch to the right focus is 2
    Then the eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity (a b c : ℝ) : 
  (∀ x y, x^2/a^2 - y^2/b^2 = 1 → 
    b * c / Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 3 ∧ 
    c - a = 2) → 
  c / a = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2735_273546


namespace NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l2735_273569

theorem sin_negative_thirty_degrees :
  let θ : Real := 30 * Real.pi / 180
  (∀ x, Real.sin (-x) = -Real.sin x) →  -- sine is an odd function
  Real.sin θ = 1/2 →                    -- sin 30° = 1/2
  Real.sin (-θ) = -1/2 := by
    sorry

end NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l2735_273569


namespace NUMINAMATH_CALUDE_square_rectangle_perimeter_sum_l2735_273550

theorem square_rectangle_perimeter_sum :
  ∀ (s l w : ℝ),
  s > 0 ∧ l > 0 ∧ w > 0 →
  s^2 + l * w = 130 →
  s^2 - l * w = 50 →
  l = 2 * w →
  4 * s + 2 * (l + w) = 12 * Real.sqrt 10 + 12 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_square_rectangle_perimeter_sum_l2735_273550


namespace NUMINAMATH_CALUDE_percentage_problem_l2735_273583

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.15 * 1600 - 15 → x = 900 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2735_273583


namespace NUMINAMATH_CALUDE_no_two_right_angles_l2735_273567

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180

-- Define a right angle
def is_right_angle (angle : ℝ) : Prop := angle = 90

-- Theorem statement
theorem no_two_right_angles (t : Triangle) : 
  ¬(is_right_angle t.A ∧ is_right_angle t.B) ∧ 
  ¬(is_right_angle t.B ∧ is_right_angle t.C) ∧ 
  ¬(is_right_angle t.A ∧ is_right_angle t.C) :=
sorry

end NUMINAMATH_CALUDE_no_two_right_angles_l2735_273567


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2735_273576

theorem pure_imaginary_complex_number (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 + m - 2) (m^2 + 4*m - 5)
  (z.re = 0 ∧ z.im ≠ 0) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2735_273576


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2735_273573

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℝ), ∀ x : ℝ,
    (x^3 - 2*x^2 + x - 1) / (x^3 + 2*x^2 + x + 1) = 
    P / (x + 1) + (Q*x + R) / (x^2 + 1) ∧
    P = -2 ∧ Q = 0 ∧ R = 1 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2735_273573


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2735_273557

-- Problem 1
theorem problem_1 : (Real.sqrt 50 * Real.sqrt 32) / Real.sqrt 8 - 4 * Real.sqrt 2 = 6 * Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 : Real.sqrt 12 - 6 * Real.sqrt (1/3) + Real.sqrt 48 = 4 * Real.sqrt 3 := by sorry

-- Problem 3
theorem problem_3 : (Real.sqrt 5 + 3) * (3 - Real.sqrt 5) - (Real.sqrt 3 - 1)^2 = 2 * Real.sqrt 3 := by sorry

-- Problem 4
theorem problem_4 : (Real.sqrt 24 + Real.sqrt 50) / Real.sqrt 2 - 6 * Real.sqrt (1/3) = 5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2735_273557


namespace NUMINAMATH_CALUDE_parabola_directrix_l2735_273553

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop :=
  y = (x^2 - 8*x + 12) / 16

/-- The directrix equation -/
def directrix_eq (y : ℝ) : Prop :=
  y = -5/4

/-- Theorem: The directrix of the given parabola is y = -5/4 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_eq x y → ∃ d : ℝ, directrix_eq d ∧ 
  (∀ p q : ℝ, parabola_eq p q → (p - x)^2 + (q - y)^2 = (q - d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2735_273553


namespace NUMINAMATH_CALUDE_smallest_x_for_equation_l2735_273597

theorem smallest_x_for_equation : 
  ∀ x : ℝ, x > 0 → (⌊x^2⌋ : ℤ) - x * (⌊x⌋ : ℤ) = 10 → x ≥ 131/11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_for_equation_l2735_273597


namespace NUMINAMATH_CALUDE_initial_speed_is_five_l2735_273589

/-- Proves that the initial speed is 5 km/hr given the conditions of the journey --/
theorem initial_speed_is_five (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) 
  (h1 : total_distance = 26.67)
  (h2 : total_time = 6)
  (h3 : second_half_speed = 4)
  (h4 : (total_distance / 2) / v + (total_distance / 2) / second_half_speed = total_time)
  : v = 5 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_is_five_l2735_273589


namespace NUMINAMATH_CALUDE_expression_evaluation_l2735_273568

/-- Proves that the given expression evaluates to 11 when x = -2 and y = -1 -/
theorem expression_evaluation :
  let x : ℝ := -2
  let y : ℝ := -1
  3 * (2 * x^2 + x*y + 1/3) - (3 * x^2 + 4*x*y - y^2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2735_273568


namespace NUMINAMATH_CALUDE_maria_carrots_l2735_273510

def total_carrots (initial : ℕ) (thrown_out : ℕ) (additional : ℕ) : ℕ :=
  initial - thrown_out + additional

theorem maria_carrots : total_carrots 48 11 15 = 52 := by
  sorry

end NUMINAMATH_CALUDE_maria_carrots_l2735_273510


namespace NUMINAMATH_CALUDE_sum_of_number_and_its_square_l2735_273559

theorem sum_of_number_and_its_square (x : ℝ) : x = 4 → x + x^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_its_square_l2735_273559


namespace NUMINAMATH_CALUDE_max_largest_integer_l2735_273586

theorem max_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℚ) / 5 = 70 →
  e - a = 10 →
  a < b ∧ b < c ∧ c < d ∧ d < e →
  e ≤ 340 :=
sorry

end NUMINAMATH_CALUDE_max_largest_integer_l2735_273586


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2735_273558

/-- Given a boat traveling downstream with a current of 3 km/hr,
    prove that its speed in still water is 15 km/hr if it travels 3.6 km in 12 minutes. -/
theorem boat_speed_in_still_water : ∀ (b : ℝ),
  (b + 3) * (1 / 5) = 3.6 →
  b = 15 := by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2735_273558


namespace NUMINAMATH_CALUDE_wen_family_theater_cost_l2735_273548

/-- Represents the cost of tickets for a family theater outing -/
def theater_cost (regular_price : ℚ) : ℚ :=
  let senior_price := regular_price * (1 - 0.2)
  let child_price := regular_price * (1 - 0.4)
  let total_before_discount := 2 * senior_price + 2 * regular_price + 2 * child_price
  total_before_discount * (1 - 0.1)

/-- Theorem stating the total cost for the Wen family's theater tickets -/
theorem wen_family_theater_cost :
  ∃ (regular_price : ℚ),
    (regular_price * (1 - 0.2) = 7.5) ∧
    (theater_cost regular_price = 40.5) := by
  sorry


end NUMINAMATH_CALUDE_wen_family_theater_cost_l2735_273548


namespace NUMINAMATH_CALUDE_clothing_sale_price_l2735_273534

theorem clothing_sale_price (a : ℝ) : 
  (∃ x y : ℝ, 
    x * 1.25 = a ∧ 
    y * 0.75 = a ∧ 
    x + y - 2*a = -8) → 
  a = 60 := by
sorry

end NUMINAMATH_CALUDE_clothing_sale_price_l2735_273534


namespace NUMINAMATH_CALUDE_min_sum_of_squares_min_sum_of_squares_value_l2735_273505

theorem min_sum_of_squares (a b c d : ℤ) : 
  a^2 ≠ b^2 → a^2 ≠ c^2 → a^2 ≠ d^2 → b^2 ≠ c^2 → b^2 ≠ d^2 → c^2 ≠ d^2 →
  (a*b + c*d)^2 + (a*d - b*c)^2 = 2004 →
  ∀ (w x y z : ℤ), w^2 ≠ x^2 → w^2 ≠ y^2 → w^2 ≠ z^2 → x^2 ≠ y^2 → x^2 ≠ z^2 → y^2 ≠ z^2 →
  (w*x + y*z)^2 + (w*z - x*y)^2 = 2004 →
  a^2 + b^2 + c^2 + d^2 ≤ w^2 + x^2 + y^2 + z^2 :=
by sorry

theorem min_sum_of_squares_value (a b c d : ℤ) : 
  a^2 ≠ b^2 → a^2 ≠ c^2 → a^2 ≠ d^2 → b^2 ≠ c^2 → b^2 ≠ d^2 → c^2 ≠ d^2 →
  (a*b + c*d)^2 + (a*d - b*c)^2 = 2004 →
  ∃ (x y : ℤ), x^2 + y^2 = 2004 ∧ a^2 + b^2 + c^2 + d^2 = 2 * (x + y) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_min_sum_of_squares_value_l2735_273505


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_l2735_273582

/-- The equation of the curve in polar form -/
def polar_equation (r θ : ℝ) : Prop :=
  r = 1 / (1 - Real.cos θ - Real.sin θ)

/-- The equation of the curve in Cartesian form -/
def cartesian_equation (x y : ℝ) : Prop :=
  x * y + x + y + (1/2) = 0

/-- Theorem stating that the curve is a hyperbola -/
theorem curve_is_hyperbola :
  ∃ (x y : ℝ), cartesian_equation x y ∧
  ∃ (r θ : ℝ), polar_equation r θ ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ :=
sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_l2735_273582


namespace NUMINAMATH_CALUDE_yangzhou_construction_area_scientific_notation_l2735_273564

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem yangzhou_construction_area_scientific_notation :
  toScientificNotation 330100000 = ScientificNotation.mk 3.301 8 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_yangzhou_construction_area_scientific_notation_l2735_273564


namespace NUMINAMATH_CALUDE_brian_read_chapters_l2735_273523

/-- The number of chapters Brian read -/
def total_chapters (book1 book2 book3 book4 : ℕ) : ℕ :=
  book1 + book2 + book3 + book4

/-- The theorem stating the total number of chapters Brian read -/
theorem brian_read_chapters : ∃ (book4 : ℕ),
  let book1 := 20
  let book2 := 15
  let book3 := 15
  book4 = (book1 + book2 + book3) / 2 ∧
  total_chapters book1 book2 book3 book4 = 75 := by
  sorry

end NUMINAMATH_CALUDE_brian_read_chapters_l2735_273523


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2735_273509

theorem fraction_subtraction : 
  (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2735_273509


namespace NUMINAMATH_CALUDE_string_displacement_impossible_l2735_273579

/-- A rectangular parallelepiped box with strings. -/
structure StringBox where
  a : ℝ
  b : ℝ
  c : ℝ
  N : ℝ × ℝ × ℝ
  P : ℝ × ℝ × ℝ

/-- Strings cross at right angles at N and P. -/
def strings_cross_at_right_angles (box : StringBox) : Prop :=
  sorry

/-- Strings are strongly glued at N and P. -/
def strings_strongly_glued (box : StringBox) : Prop :=
  sorry

/-- Any displacement of the strings is impossible. -/
def no_displacement_possible (box : StringBox) : Prop :=
  sorry

/-- Theorem: If strings cross at right angles and are strongly glued at N and P,
    then any displacement of the strings is impossible. -/
theorem string_displacement_impossible (box : StringBox) :
  strings_cross_at_right_angles box →
  strings_strongly_glued box →
  no_displacement_possible box :=
by
  sorry

end NUMINAMATH_CALUDE_string_displacement_impossible_l2735_273579


namespace NUMINAMATH_CALUDE_equality_condition_l2735_273587

theorem equality_condition (a b c k : ℝ) : 
  a + b + c = 1 → (k * (a + b * c) = (a + b) * (a + c) ↔ k = 1) := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l2735_273587


namespace NUMINAMATH_CALUDE_apple_to_cucumber_ratio_l2735_273599

/-- Given the cost ratios of fruits, calculate the equivalent number of cucumbers for 20 apples -/
theorem apple_to_cucumber_ratio 
  (apple_banana_ratio : ℚ) 
  (banana_cucumber_ratio : ℚ) 
  (h1 : apple_banana_ratio = 10 / 5)  -- 10 apples = 5 bananas
  (h2 : banana_cucumber_ratio = 3 / 4)  -- 3 bananas = 4 cucumbers
  : (20 : ℚ) / apple_banana_ratio * banana_cucumber_ratio⁻¹ = 40 / 3 :=
by sorry

end NUMINAMATH_CALUDE_apple_to_cucumber_ratio_l2735_273599
