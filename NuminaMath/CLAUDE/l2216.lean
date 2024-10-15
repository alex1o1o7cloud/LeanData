import Mathlib

namespace NUMINAMATH_CALUDE_product_11_4_sum_144_l2216_221662

theorem product_11_4_sum_144 (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 11^4 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) + (d : ℕ) = 144 :=
by sorry

end NUMINAMATH_CALUDE_product_11_4_sum_144_l2216_221662


namespace NUMINAMATH_CALUDE_markers_problem_l2216_221606

theorem markers_problem (initial_markers : ℕ) (markers_per_box : ℕ) (total_markers : ℕ) :
  initial_markers = 32 →
  markers_per_box = 9 →
  total_markers = 86 →
  (total_markers - initial_markers) / markers_per_box = 6 :=
by sorry

end NUMINAMATH_CALUDE_markers_problem_l2216_221606


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2216_221623

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Theorem statement
theorem arithmetic_sequence_common_difference 
  (a₁ : ℝ) 
  (d : ℝ) 
  (h_d_nonzero : d ≠ 0) 
  (h_sum : arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 5 = 13)
  (h_geometric : ∃ r : ℝ, r ≠ 0 ∧ 
    arithmetic_sequence a₁ d 2 = arithmetic_sequence a₁ d 1 * r ∧ 
    arithmetic_sequence a₁ d 5 = arithmetic_sequence a₁ d 2 * r) :
  d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2216_221623


namespace NUMINAMATH_CALUDE_arrangements_ends_correct_arrangements_together_correct_arrangements_not_ends_correct_l2216_221639

/-- The number of people standing in a row -/
def n : ℕ := 7

/-- The number of arrangements with A and B at the ends -/
def arrangements_ends : ℕ := 240

/-- The number of arrangements with A, B, and C together -/
def arrangements_together : ℕ := 720

/-- The number of arrangements with A not at beginning and B not at end -/
def arrangements_not_ends : ℕ := 3720

/-- Theorem for the number of arrangements with A and B at the ends -/
theorem arrangements_ends_correct : 
  arrangements_ends = 2 * Nat.factorial (n - 2) := by sorry

/-- Theorem for the number of arrangements with A, B, and C together -/
theorem arrangements_together_correct : 
  arrangements_together = 6 * Nat.factorial (n - 3) := by sorry

/-- Theorem for the number of arrangements with A not at beginning and B not at end -/
theorem arrangements_not_ends_correct : 
  arrangements_not_ends = Nat.factorial n - 2 * Nat.factorial (n - 1) + Nat.factorial (n - 2) := by sorry

end NUMINAMATH_CALUDE_arrangements_ends_correct_arrangements_together_correct_arrangements_not_ends_correct_l2216_221639


namespace NUMINAMATH_CALUDE_inequality_theorem_l2216_221602

theorem inequality_theorem (a b : ℝ) (h1 : a^3 > b^3) (h2 : a * b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2216_221602


namespace NUMINAMATH_CALUDE_current_speed_l2216_221617

/-- Given a man's speed with and against a current, calculate the speed of the current. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 15)
  (h2 : speed_against_current = 10) :
  ∃ (current_speed : ℝ), current_speed = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l2216_221617


namespace NUMINAMATH_CALUDE_fruits_left_l2216_221657

-- Define the initial quantities of fruits
def initial_bananas : ℕ := 12
def initial_apples : ℕ := 7
def initial_grapes : ℕ := 19

-- Define the quantities of fruits eaten
def eaten_bananas : ℕ := 4
def eaten_apples : ℕ := 2
def eaten_grapes : ℕ := 10

-- Define the function to calculate remaining fruits
def remaining_fruits : ℕ := 
  (initial_bananas - eaten_bananas) + 
  (initial_apples - eaten_apples) + 
  (initial_grapes - eaten_grapes)

-- Theorem statement
theorem fruits_left : remaining_fruits = 22 := by
  sorry

end NUMINAMATH_CALUDE_fruits_left_l2216_221657


namespace NUMINAMATH_CALUDE_exists_square_composition_function_l2216_221603

theorem exists_square_composition_function : ∃ F : ℕ → ℕ, ∀ n : ℕ, (F ∘ F) n = n^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_square_composition_function_l2216_221603


namespace NUMINAMATH_CALUDE_cards_distribution_l2216_221649

/-- Given 60 cards dealt to 9 people as evenly as possible, 
    the number of people with fewer than 7 cards is 3. -/
theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) 
    (h1 : total_cards = 60) (h2 : num_people = 9) :
  let cards_per_person := total_cards / num_people
  let remainder := total_cards % num_people
  let people_with_extra := remainder
  let people_with_fewer := num_people - people_with_extra
  people_with_fewer = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l2216_221649


namespace NUMINAMATH_CALUDE_max_d_value_l2216_221668

def is_valid_number (d e : ℕ) : Prop :=
  d < 10 ∧ e < 10 ∧ (552200 + d * 100 + e * 11) % 22 = 0

theorem max_d_value :
  (∃ d e, is_valid_number d e) →
  (∀ d e, is_valid_number d e → d ≤ 6) ∧
  (∃ e, is_valid_number 6 e) :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_l2216_221668


namespace NUMINAMATH_CALUDE_complex_to_exponential_form_l2216_221620

theorem complex_to_exponential_form (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 →
  ∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧ θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_to_exponential_form_l2216_221620


namespace NUMINAMATH_CALUDE_circle_outside_square_area_l2216_221677

/-- The area inside a circle with radius 1/2 but outside a square with side length 1, 
    when both shapes share the same center, is equal to π/4 - 1. -/
theorem circle_outside_square_area :
  let square_side : ℝ := 1
  let circle_radius : ℝ := 1/2
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  circle_area - square_area = π/4 - 1 := by
sorry

end NUMINAMATH_CALUDE_circle_outside_square_area_l2216_221677


namespace NUMINAMATH_CALUDE_tan_two_theta_minus_pi_over_six_l2216_221681

theorem tan_two_theta_minus_pi_over_six (θ : Real) 
  (h : 4 * Real.cos (θ + π/3) * Real.cos (θ - π/6) = Real.sin (2*θ)) : 
  Real.tan (2*θ - π/6) = Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_theta_minus_pi_over_six_l2216_221681


namespace NUMINAMATH_CALUDE_negation_of_conditional_l2216_221626

theorem negation_of_conditional (x : ℝ) :
  (¬(x = 3 → x^2 - 2*x - 3 = 0)) ↔ (x ≠ 3 → x^2 - 2*x - 3 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_conditional_l2216_221626


namespace NUMINAMATH_CALUDE_shorts_cost_l2216_221643

def total_spent : ℝ := 33.56
def shirt_cost : ℝ := 12.14
def jacket_cost : ℝ := 7.43

theorem shorts_cost (shorts_cost : ℝ) : 
  shorts_cost = total_spent - shirt_cost - jacket_cost → shorts_cost = 13.99 := by
  sorry

end NUMINAMATH_CALUDE_shorts_cost_l2216_221643


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2216_221675

/-- Given sets A and B with the specified elements, if their intersection is {-3},
    then a = -1. -/
theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, 2*a-1, a^2+1}
  (A ∩ B : Set ℝ) = {-3} → a = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2216_221675


namespace NUMINAMATH_CALUDE_rose_bush_count_l2216_221688

theorem rose_bush_count (initial_bushes planted_bushes : ℕ) :
  initial_bushes = 2 → planted_bushes = 4 →
  initial_bushes + planted_bushes = 6 := by
  sorry

end NUMINAMATH_CALUDE_rose_bush_count_l2216_221688


namespace NUMINAMATH_CALUDE_rectangle_width_l2216_221682

theorem rectangle_width (width : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * width → 
  area = length * width → 
  area = 48 → 
  width = 4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l2216_221682


namespace NUMINAMATH_CALUDE_jose_age_l2216_221693

theorem jose_age (maria_age jose_age : ℕ) : 
  jose_age = maria_age + 12 →
  maria_age + jose_age = 40 →
  jose_age = 26 := by
sorry

end NUMINAMATH_CALUDE_jose_age_l2216_221693


namespace NUMINAMATH_CALUDE_veronica_initial_marbles_l2216_221676

/-- Represents the number of marbles each person has -/
structure Marbles where
  dilan : ℕ
  martha : ℕ
  phillip : ℕ
  veronica : ℕ

/-- The initial distribution of marbles -/
def initial_marbles : Marbles where
  dilan := 14
  martha := 20
  phillip := 19
  veronica := 7  -- We'll prove this is correct

/-- The number of people -/
def num_people : ℕ := 4

/-- The number of marbles each person has after redistribution -/
def marbles_after_redistribution : ℕ := 15

theorem veronica_initial_marbles :
  (initial_marbles.dilan +
   initial_marbles.martha +
   initial_marbles.phillip +
   initial_marbles.veronica) =
  (num_people * marbles_after_redistribution) :=
by sorry

end NUMINAMATH_CALUDE_veronica_initial_marbles_l2216_221676


namespace NUMINAMATH_CALUDE_smallest_positive_period_of_f_l2216_221655

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem smallest_positive_period_of_f : 
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_period_of_f_l2216_221655


namespace NUMINAMATH_CALUDE_equivalent_systems_intersection_l2216_221616

-- Define the type for a linear equation
def LinearEquation := ℝ → ℝ → ℝ

-- Define a system of two linear equations
structure LinearSystem :=
  (eq1 eq2 : LinearEquation)

-- Define the solution set of a linear system
def SolutionSet (sys : LinearSystem) := {p : ℝ × ℝ | sys.eq1 p.1 p.2 = 0 ∧ sys.eq2 p.1 p.2 = 0}

-- Define equivalence of two linear systems
def EquivalentSystems (sys1 sys2 : LinearSystem) :=
  SolutionSet sys1 = SolutionSet sys2

-- Define the intersection points of two lines
def IntersectionPoints (eq1 eq2 : LinearEquation) :=
  {p : ℝ × ℝ | eq1 p.1 p.2 = 0 ∧ eq2 p.1 p.2 = 0}

-- Theorem statement
theorem equivalent_systems_intersection
  (sys1 sys2 : LinearSystem)
  (h : EquivalentSystems sys1 sys2) :
  IntersectionPoints sys1.eq1 sys1.eq2 = IntersectionPoints sys2.eq1 sys2.eq2 := by
  sorry


end NUMINAMATH_CALUDE_equivalent_systems_intersection_l2216_221616


namespace NUMINAMATH_CALUDE_parallel_vectors_l2216_221628

/-- Given vectors a and b in ℝ², prove that k = -1/3 makes k*a + b parallel to a - 3*b -/
theorem parallel_vectors (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (-3, 2)) :
  let k : ℝ := -1/3
  let v1 : ℝ × ℝ := (k * a.1 + b.1, k * a.2 + b.2)
  let v2 : ℝ × ℝ := (a.1 - 3 * b.1, a.2 - 3 * b.2)
  ∃ (c : ℝ), v1 = (c * v2.1, c * v2.2) := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l2216_221628


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2216_221650

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| < 2}
def B : Set ℝ := {x : ℝ | x^2 - 3*x < 0}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2216_221650


namespace NUMINAMATH_CALUDE_point_outside_circle_l2216_221684

theorem point_outside_circle (m : ℝ) : 
  let P : ℝ × ℝ := (m^2, 5)
  let circle_equation (x y : ℝ) := x^2 + y^2 = 24
  ∀ x y, circle_equation x y → (P.1 - x)^2 + (P.2 - y)^2 > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l2216_221684


namespace NUMINAMATH_CALUDE_notebook_distribution_l2216_221629

theorem notebook_distribution (class_a class_b notebooks_a notebooks_b : ℕ) 
  (h1 : notebooks_a = class_a / 8)
  (h2 : notebooks_b = 2 * class_a)
  (h3 : 16 = (class_a / 2) / 8)
  (h4 : class_a + class_b = (120 * class_a) / 100) :
  class_a * notebooks_a + class_b * notebooks_b = 2176 := by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l2216_221629


namespace NUMINAMATH_CALUDE_missing_sale_is_correct_l2216_221671

/-- Calculates the missing sale amount given sales for 5 out of 6 months and the average sale -/
def calculate_missing_sale (sale1 sale2 sale3 sale5 sale6 average_sale : ℕ) : ℕ :=
  6 * average_sale - (sale1 + sale2 + sale3 + sale5 + sale6)

/-- Theorem: The calculated missing sale is correct given the conditions -/
theorem missing_sale_is_correct (sale1 sale2 sale3 sale5 sale6 average_sale : ℕ) :
  let sale4 := calculate_missing_sale sale1 sale2 sale3 sale5 sale6 average_sale
  (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average_sale := by
  sorry

#eval calculate_missing_sale 7435 7927 7855 7562 5991 7500

end NUMINAMATH_CALUDE_missing_sale_is_correct_l2216_221671


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2216_221641

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the property of being a pure imaginary number
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem complex_magnitude_problem (z : ℂ) (a : ℝ) 
  (h1 : is_pure_imaginary z) 
  (h2 : (2 + i) * z = 1 + a * i^3) : 
  Complex.abs (a + z) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2216_221641


namespace NUMINAMATH_CALUDE_smallest_square_l2216_221614

theorem smallest_square (a b : ℕ+) 
  (h1 : ∃ r : ℕ, (15 : ℤ) * a + (16 : ℤ) * b = r^2)
  (h2 : ∃ s : ℕ, (16 : ℤ) * a - (15 : ℤ) * b = s^2) :
  (481 : ℕ)^2 ≤ min ((15 : ℤ) * a + (16 : ℤ) * b) ((16 : ℤ) * a - (15 : ℤ) * b) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_l2216_221614


namespace NUMINAMATH_CALUDE_min_value_T_l2216_221637

/-- Given a quadratic inequality that holds for all real x, prove the minimum value of T -/
theorem min_value_T (a b c : ℝ) (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) (h2 : a < b) :
  (∀ a' b' c' : ℝ, (∀ x : ℝ, a' * x^2 + b' * x + c' ≥ 0) → a' < b' → 
    (a' + b' + c') / (b' - a') ≥ (a + b + c) / (b - a)) → 
  (a + b + c) / (b - a) = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_T_l2216_221637


namespace NUMINAMATH_CALUDE_tetrahedron_circumscribed_sphere_area_l2216_221690

/-- Given a tetrahedron with three mutually perpendicular lateral edges of lengths 1, √2, and √3,
    the surface area of its circumscribed sphere is 6π. -/
theorem tetrahedron_circumscribed_sphere_area (a b c : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 2) (h3 : c = Real.sqrt 3) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_circumscribed_sphere_area_l2216_221690


namespace NUMINAMATH_CALUDE_angle_not_sharing_terminal_side_l2216_221648

/-- Two angles share the same terminal side if their difference is a multiple of 360° -/
def ShareTerminalSide (a b : ℝ) : Prop :=
  ∃ k : ℤ, a - b = 360 * k

/-- The main theorem -/
theorem angle_not_sharing_terminal_side :
  let angles : List ℝ := [330, -30, 680, -1110]
  ∀ a ∈ angles, a ≠ 680 → ShareTerminalSide a (-750) ∧
  ¬ ShareTerminalSide 680 (-750) := by
  sorry


end NUMINAMATH_CALUDE_angle_not_sharing_terminal_side_l2216_221648


namespace NUMINAMATH_CALUDE_min_value_x_plus_inverse_equality_condition_l2216_221698

theorem min_value_x_plus_inverse (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 :=
by
  sorry

theorem equality_condition (x : ℝ) (hx : x > 0) : x + 1/x = 2 ↔ x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_inverse_equality_condition_l2216_221698


namespace NUMINAMATH_CALUDE_polygon_sides_count_l2216_221694

theorem polygon_sides_count (n : ℕ) : n > 2 → (n - 2) * 180 = 3 * 360 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l2216_221694


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_neg_one_l2216_221604

theorem sqrt_meaningful_iff_geq_neg_one (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 1) ↔ x ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_neg_one_l2216_221604


namespace NUMINAMATH_CALUDE_cecilia_always_wins_l2216_221659

theorem cecilia_always_wins (a : ℕ+) : ∃ b : ℕ+, 
  (Nat.gcd a b = 1) ∧ 
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ 
    (p * q * r ∣ a^3 + b^3)) := by
  sorry

end NUMINAMATH_CALUDE_cecilia_always_wins_l2216_221659


namespace NUMINAMATH_CALUDE_more_selected_in_B_l2216_221656

def total_candidates : ℕ := 8000
def selection_rate_A : ℚ := 6 / 100
def selection_rate_B : ℚ := 7 / 100

theorem more_selected_in_B : 
  ⌊(selection_rate_B * total_candidates : ℚ)⌋ - ⌊(selection_rate_A * total_candidates : ℚ)⌋ = 80 := by
  sorry

end NUMINAMATH_CALUDE_more_selected_in_B_l2216_221656


namespace NUMINAMATH_CALUDE_periodic_sum_implies_periodic_components_l2216_221627

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem periodic_sum_implies_periodic_components
  (f g h : ℝ → ℝ) (T : ℝ)
  (h₁ : is_periodic (λ x => f x + g x) T)
  (h₂ : is_periodic (λ x => f x + h x) T)
  (h₃ : is_periodic (λ x => g x + h x) T) :
  is_periodic f T ∧ is_periodic g T ∧ is_periodic h T :=
sorry

end NUMINAMATH_CALUDE_periodic_sum_implies_periodic_components_l2216_221627


namespace NUMINAMATH_CALUDE_geometric_sequence_tan_l2216_221689

/-- Given a geometric sequence {a_n} satisfying certain conditions, 
    prove that tan((a_4 * a_6 / 3) * π) = -√3 -/
theorem geometric_sequence_tan (a : ℕ → ℝ) : 
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- {a_n} is a geometric sequence
  a 2 * a 3 * a 4 = -a 7^2 →                 -- a_2 * a_3 * a_4 = -a_7^2
  a 2 * a 3 * a 4 = -64 →                    -- a_2 * a_3 * a_4 = -64
  Real.tan ((a 4 * a 6 / 3) * Real.pi) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_tan_l2216_221689


namespace NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l2216_221605

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b + 2 * b * c) / (a^2 + b^2 + c^2) ≤ Real.sqrt 5 / 2 :=
by sorry

theorem max_value_achievable (ε : ℝ) (hε : ε > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a * b + 2 * b * c) / (a^2 + b^2 + c^2) > Real.sqrt 5 / 2 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l2216_221605


namespace NUMINAMATH_CALUDE_problem_solution_l2216_221673

theorem problem_solution : 
  ∃ x : ℝ, ((15 - 2 + 4) / 2) * x = 77 ∧ x = 77 / 8.5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2216_221673


namespace NUMINAMATH_CALUDE_rope_cutting_l2216_221640

/-- Given two ropes of lengths 18 and 24 meters, this theorem proves that
    the maximum length of equal segments that can be cut from both ropes
    without remainder is 6 meters, and the total number of such segments is 7. -/
theorem rope_cutting (rope1 : ℕ) (rope2 : ℕ) 
  (h1 : rope1 = 18) (h2 : rope2 = 24) : 
  ∃ (segment_length : ℕ) (total_segments : ℕ),
    segment_length = 6 ∧ 
    total_segments = 7 ∧
    rope1 % segment_length = 0 ∧
    rope2 % segment_length = 0 ∧
    rope1 / segment_length + rope2 / segment_length = total_segments ∧
    ∀ (l : ℕ), l > segment_length → (rope1 % l ≠ 0 ∨ rope2 % l ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_rope_cutting_l2216_221640


namespace NUMINAMATH_CALUDE_least_possible_c_l2216_221642

theorem least_possible_c (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 20 →
  a ≤ b ∧ b ≤ c →
  b - a ≥ 2 ∧ c - b ≥ 2 →
  a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0 →
  b = a + 13 →
  c ≥ 33 ∧ ∀ (c' : ℕ), c' ≥ 33 → c' % 3 = 0 → c' - b ≥ 2 → c ≤ c' :=
by sorry

end NUMINAMATH_CALUDE_least_possible_c_l2216_221642


namespace NUMINAMATH_CALUDE_base_conversion_1729_l2216_221600

theorem base_conversion_1729 :
  2 * (5 ^ 4) + 3 * (5 ^ 3) + 4 * (5 ^ 2) + 0 * (5 ^ 1) + 4 * (5 ^ 0) = 1729 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1729_l2216_221600


namespace NUMINAMATH_CALUDE_calculator_problem_l2216_221696

/-- Represents the possible operations on the calculator --/
inductive Operation
| addOne
| addThree
| double

/-- Applies a single operation to a number --/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.addOne => n + 1
  | Operation.addThree => n + 3
  | Operation.double => n * 2

/-- Applies a sequence of operations to a number --/
def applySequence (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation start

/-- Checks if a sequence of operations transforms start into target --/
def isValidSequence (start target : ℕ) (ops : List Operation) : Prop :=
  applySequence start ops = target

/-- The main theorem to be proved --/
theorem calculator_problem :
  ∃ (ops : List Operation),
    ops.length = 10 ∧
    isValidSequence 1 410 ops ∧
    ∀ (shorter_ops : List Operation),
      shorter_ops.length < 10 →
      ¬ isValidSequence 1 410 shorter_ops :=
sorry

end NUMINAMATH_CALUDE_calculator_problem_l2216_221696


namespace NUMINAMATH_CALUDE_total_fat_ingested_l2216_221674

def fat_content (fish : String) : ℝ :=
  match fish with
  | "herring" => 40
  | "eel" => 20
  | "pike" => 30
  | "salmon" => 35
  | "halibut" => 50
  | _ => 0

def cooking_loss_rate : ℝ := 0.1
def indigestible_rate : ℝ := 0.08

def digestible_fat (fish : String) : ℝ :=
  let initial_fat := fat_content fish
  let after_cooking := initial_fat * (1 - cooking_loss_rate)
  after_cooking * (1 - indigestible_rate)

def fish_counts : List (String × ℕ) := [
  ("herring", 40),
  ("eel", 30),
  ("pike", 25),
  ("salmon", 20),
  ("halibut", 15)
]

theorem total_fat_ingested :
  (fish_counts.map (λ (fish, count) => (digestible_fat fish) * count)).sum = 3643.2 := by
  sorry

end NUMINAMATH_CALUDE_total_fat_ingested_l2216_221674


namespace NUMINAMATH_CALUDE_probability_more_than_seven_is_five_twelfths_l2216_221621

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when throwing two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of favorable outcomes (totals greater than 7) -/
def favorableOutcomes : ℕ := 15

/-- The probability of getting a total more than 7 when throwing a pair of dice -/
def probabilityMoreThanSeven : ℚ := favorableOutcomes / totalOutcomes

theorem probability_more_than_seven_is_five_twelfths :
  probabilityMoreThanSeven = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_more_than_seven_is_five_twelfths_l2216_221621


namespace NUMINAMATH_CALUDE_factorization_equality_l2216_221697

theorem factorization_equality (a x y : ℝ) : a * x^2 + 2 * a * x * y + a * y^2 = a * (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2216_221697


namespace NUMINAMATH_CALUDE_paper_fold_crease_length_l2216_221646

theorem paper_fold_crease_length :
  ∀ (width : ℝ) (angle : ℝ),
  width = 8 →
  angle = π / 4 →
  ∃ (crease_length : ℝ),
  crease_length = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_paper_fold_crease_length_l2216_221646


namespace NUMINAMATH_CALUDE_fifth_root_fraction_l2216_221653

theorem fifth_root_fraction : 
  (9 / 16.2) ^ (1/5 : ℝ) = (5/9 : ℝ) ^ (1/5 : ℝ) := by sorry

end NUMINAMATH_CALUDE_fifth_root_fraction_l2216_221653


namespace NUMINAMATH_CALUDE_total_blankets_collected_l2216_221608

/-- Represents the blanket collection problem over three days --/
def blanket_collection (original_members : ℕ) (new_members : ℕ) 
  (blankets_per_original : ℕ) (blankets_per_new : ℕ) 
  (school_blankets : ℕ) (online_blankets : ℕ) : ℕ :=
  let day1 := original_members * blankets_per_original
  let day2_team := original_members * blankets_per_original + new_members * blankets_per_new
  let day2 := day2_team + 3 * day1
  let day3 := school_blankets + online_blankets
  day1 + day2 + day3

/-- The main theorem stating the total number of blankets collected --/
theorem total_blankets_collected : 
  blanket_collection 15 5 2 4 22 30 = 222 := by
  sorry

end NUMINAMATH_CALUDE_total_blankets_collected_l2216_221608


namespace NUMINAMATH_CALUDE_wall_width_l2216_221652

theorem wall_width (w h l : ℝ) (volume : ℝ) : 
  h = 4 * w →
  l = 3 * h →
  volume = w * h * l →
  volume = 10368 →
  w = 6 := by
sorry

end NUMINAMATH_CALUDE_wall_width_l2216_221652


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l2216_221663

theorem inequality_not_always_true 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (hc : c ≠ 0) : 
  ∃ c, ¬(a * c > b * c) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l2216_221663


namespace NUMINAMATH_CALUDE_total_cost_theorem_l2216_221638

/-- The total cost of buying thermometers and masks -/
def total_cost (a b : ℝ) : ℝ := 3 * a + b

/-- Theorem: The total cost of buying 3 thermometers at 'a' yuan each
    and 'b' masks at 1 yuan each is equal to (3a + b) yuan -/
theorem total_cost_theorem (a b : ℝ) :
  total_cost a b = 3 * a + b := by sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l2216_221638


namespace NUMINAMATH_CALUDE_distance_between_parallel_lines_l2216_221601

/-- The distance between two parallel lines in R² --/
theorem distance_between_parallel_lines :
  let line1 : ℝ → ℝ × ℝ := λ t ↦ (4 + 2*t, -1 - 6*t)
  let line2 : ℝ → ℝ × ℝ := λ s ↦ (3 + 2*s, -2 - 6*s)
  let v : ℝ × ℝ := (3 - 4, -2 - (-1))
  let d : ℝ × ℝ := (2, -6)
  let distance := ‖v - (((v.1 * d.1 + v.2 * d.2) / (d.1^2 + d.2^2)) • d)‖
  distance = 2 * Real.sqrt 10 / 5 := by
sorry


end NUMINAMATH_CALUDE_distance_between_parallel_lines_l2216_221601


namespace NUMINAMATH_CALUDE_todds_profit_l2216_221636

/-- Calculates Todd's remaining money after his snow cone business venture -/
def todds_remaining_money (borrowed : ℕ) (repay : ℕ) (ingredients_cost : ℕ) 
  (num_sold : ℕ) (price_per_cone : ℚ) : ℚ :=
  let total_sales := num_sold * price_per_cone
  let remaining := total_sales - repay
  remaining

/-- Proves that Todd's remaining money is $40 after his snow cone business venture -/
theorem todds_profit : 
  todds_remaining_money 100 110 75 200 (75/100) = 40 := by
  sorry

end NUMINAMATH_CALUDE_todds_profit_l2216_221636


namespace NUMINAMATH_CALUDE_speedboat_speed_l2216_221667

/-- Proves that the speed of a speedboat crossing a lake is 30 miles per hour,
    given specific conditions about the lake width, sailboat speed, and wait time. -/
theorem speedboat_speed
  (lake_width : ℝ)
  (sailboat_speed : ℝ)
  (wait_time : ℝ)
  (h_lake : lake_width = 60)
  (h_sail : sailboat_speed = 12)
  (h_wait : wait_time = 3)
  : (lake_width / (lake_width / sailboat_speed - wait_time)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_speedboat_speed_l2216_221667


namespace NUMINAMATH_CALUDE_cube_power_eq_l2216_221634

theorem cube_power_eq : (3^3 * 6^3)^2 = 34062224 := by
  sorry

end NUMINAMATH_CALUDE_cube_power_eq_l2216_221634


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2216_221691

/-- Given a hyperbola with equation 9y² - 25x² = 169, 
    its asymptotes are given by the equation y = ± (5/3)x -/
theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), 9 * y^2 - 25 * x^2 = 169 →
  ∃ (k : ℝ), k = 5/3 ∧ (y = k * x ∨ y = -k * x) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2216_221691


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l2216_221664

theorem complex_equation_solutions :
  let f : ℂ → ℂ := λ z => (z^4 - 1) / (z^3 - 3*z + 2)
  ∃! (s : Finset ℂ), s.card = 3 ∧ ∀ z ∈ s, f z = 0 ∧ ∀ w ∉ s, f w ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l2216_221664


namespace NUMINAMATH_CALUDE_tangent_slope_at_origin_l2216_221660

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_slope_at_origin :
  (deriv f) 0 = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_origin_l2216_221660


namespace NUMINAMATH_CALUDE_point_C_complex_number_l2216_221672

/-- Given points A, B, and C in the complex plane, prove that C corresponds to 4-2i -/
theorem point_C_complex_number (A B C : ℂ) : 
  A = 2 + I →
  B - A = 1 + 2*I →
  C - B = 3 - I →
  C = 4 - 2*I := by sorry

end NUMINAMATH_CALUDE_point_C_complex_number_l2216_221672


namespace NUMINAMATH_CALUDE_valid_pairs_l2216_221687

def is_valid_pair (n p : ℕ) : Prop :=
  n > 1 ∧ Nat.Prime p ∧ ((p - 1)^n + 1) % n^(p - 1) = 0

theorem valid_pairs :
  ∀ n p : ℕ, is_valid_pair n p ↔ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l2216_221687


namespace NUMINAMATH_CALUDE_two_machines_half_hour_copies_l2216_221685

/-- Represents a copy machine with a constant copying rate. -/
structure CopyMachine where
  copies_per_minute : ℕ

/-- Calculates the total number of copies made by two machines in a given time. -/
def total_copies (machine1 machine2 : CopyMachine) (minutes : ℕ) : ℕ :=
  (machine1.copies_per_minute + machine2.copies_per_minute) * minutes

/-- Theorem stating that two specific copy machines working together for 30 minutes will produce 2850 copies. -/
theorem two_machines_half_hour_copies :
  let machine1 : CopyMachine := ⟨40⟩
  let machine2 : CopyMachine := ⟨55⟩
  total_copies machine1 machine2 30 = 2850 := by
  sorry


end NUMINAMATH_CALUDE_two_machines_half_hour_copies_l2216_221685


namespace NUMINAMATH_CALUDE_gumdrop_purchase_l2216_221692

theorem gumdrop_purchase (total_cents : ℕ) (cost_per_gumdrop : ℕ) (max_gumdrops : ℕ) : 
  total_cents = 224 → cost_per_gumdrop = 8 → max_gumdrops = total_cents / cost_per_gumdrop → max_gumdrops = 28 := by
  sorry

end NUMINAMATH_CALUDE_gumdrop_purchase_l2216_221692


namespace NUMINAMATH_CALUDE_joan_seashells_l2216_221635

/-- The number of seashells Joan has after a series of events -/
def final_seashells (initial : ℕ) (given : ℕ) (found : ℕ) (traded : ℕ) (received : ℕ) (lost : ℕ) : ℕ :=
  initial - given + found - traded + received - lost

/-- Theorem stating that Joan ends up with 51 seashells -/
theorem joan_seashells :
  final_seashells 79 63 45 20 15 5 = 51 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l2216_221635


namespace NUMINAMATH_CALUDE_two_lines_two_intersections_l2216_221654

/-- The number of intersection points for n lines on a plane -/
def intersection_points (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: If n lines on a plane intersect at exactly 2 points, then n = 2 -/
theorem two_lines_two_intersections (n : ℕ) (h : intersection_points n = 2) : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_lines_two_intersections_l2216_221654


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l2216_221680

theorem normal_distribution_std_dev (μ : ℝ) (x : ℝ) (σ : ℝ) 
  (h1 : μ = 14.5)
  (h2 : x = 11.1)
  (h3 : x = μ - 2 * σ) :
  σ = 1.7 := by
sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l2216_221680


namespace NUMINAMATH_CALUDE_convex_polyhedron_same_edge_count_l2216_221645

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : ℕ
  max_edges : ℕ
  faces_ge_max_edges : faces ≥ max_edges
  min_edges_per_face : max_edges ≥ 3

/-- Theorem: A convex polyhedron always has two faces with the same number of edges -/
theorem convex_polyhedron_same_edge_count (P : ConvexPolyhedron) :
  ∃ (e : ℕ) (f₁ f₂ : ℕ), f₁ ≠ f₂ ∧ f₁ ≤ P.faces ∧ f₂ ≤ P.faces ∧
  (∃ (edges_of_face : ℕ → ℕ), 
    (∀ f, f ≤ P.faces → 3 ≤ edges_of_face f ∧ edges_of_face f ≤ P.max_edges) ∧
    edges_of_face f₁ = e ∧ edges_of_face f₂ = e) :=
sorry

end NUMINAMATH_CALUDE_convex_polyhedron_same_edge_count_l2216_221645


namespace NUMINAMATH_CALUDE_factorial_ratio_l2216_221678

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l2216_221678


namespace NUMINAMATH_CALUDE_polynomial_equality_sum_l2216_221651

theorem polynomial_equality_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = 
    a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4 + 
    a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + 
    a₉*(x+2)^9 + a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_sum_l2216_221651


namespace NUMINAMATH_CALUDE_two_circles_congruent_l2216_221669

-- Define the square
def Square := {s : ℝ // s > 0}

-- Define a circle with center and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define the configuration of three circles in a square
structure ThreeCirclesInSquare where
  square : Square
  circle1 : Circle
  circle2 : Circle
  circle3 : Circle
  
  -- Each circle touches two sides of the square
  touches_sides1 : 
    (circle1.center.1 = circle1.radius ∨ circle1.center.1 = square.val - circle1.radius) ∧
    (circle1.center.2 = circle1.radius ∨ circle1.center.2 = square.val - circle1.radius)
  touches_sides2 : 
    (circle2.center.1 = circle2.radius ∨ circle2.center.1 = square.val - circle2.radius) ∧
    (circle2.center.2 = circle2.radius ∨ circle2.center.2 = square.val - circle2.radius)
  touches_sides3 : 
    (circle3.center.1 = circle3.radius ∨ circle3.center.1 = square.val - circle3.radius) ∧
    (circle3.center.2 = circle3.radius ∨ circle3.center.2 = square.val - circle3.radius)

  -- Circles are externally tangent to each other
  externally_tangent12 : (circle1.center.1 - circle2.center.1)^2 + (circle1.center.2 - circle2.center.2)^2 = (circle1.radius + circle2.radius)^2
  externally_tangent13 : (circle1.center.1 - circle3.center.1)^2 + (circle1.center.2 - circle3.center.2)^2 = (circle1.radius + circle3.radius)^2
  externally_tangent23 : (circle2.center.1 - circle3.center.1)^2 + (circle2.center.2 - circle3.center.2)^2 = (circle2.radius + circle3.radius)^2

-- Theorem statement
theorem two_circles_congruent (config : ThreeCirclesInSquare) :
  config.circle1.radius = config.circle2.radius ∨ 
  config.circle1.radius = config.circle3.radius ∨ 
  config.circle2.radius = config.circle3.radius :=
sorry

end NUMINAMATH_CALUDE_two_circles_congruent_l2216_221669


namespace NUMINAMATH_CALUDE_find_constant_b_l2216_221695

theorem find_constant_b (b d c : ℚ) : 
  (∀ x : ℚ, (7 * x^2 - 5 * x + 11/4) * (d * x^2 + b * x + c) = 
    21 * x^4 - 26 * x^3 + 34 * x^2 - (55/4) * x + 33/4) → 
  b = -11/7 := by
sorry

end NUMINAMATH_CALUDE_find_constant_b_l2216_221695


namespace NUMINAMATH_CALUDE_sin_870_degrees_l2216_221609

theorem sin_870_degrees : Real.sin (870 * π / 180) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_870_degrees_l2216_221609


namespace NUMINAMATH_CALUDE_perpendicular_circle_exists_l2216_221611

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define perpendicularity between circles
def isPerpendicular (c1 c2 : Circle) : Prop := sorry

-- Define a point passing through a circle
def passesThroughPoint (c : Circle) (p : ℝ × ℝ) : Prop := sorry

theorem perpendicular_circle_exists (A : ℝ × ℝ) (S1 S2 : Circle) :
  ∃! C : Circle, passesThroughPoint C A ∧ isPerpendicular C S1 ∧ isPerpendicular C S2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_circle_exists_l2216_221611


namespace NUMINAMATH_CALUDE_initial_cookies_count_l2216_221631

/-- The number of cookies initially in the package -/
def initial_cookies : ℕ := sorry

/-- The number of cookies left after eating some -/
def cookies_left : ℕ := 9

/-- The number of cookies eaten -/
def cookies_eaten : ℕ := 9

/-- Theorem stating that the initial number of cookies is 18 -/
theorem initial_cookies_count : initial_cookies = cookies_left + cookies_eaten := by sorry

end NUMINAMATH_CALUDE_initial_cookies_count_l2216_221631


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l2216_221632

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 1230000

/-- The scientific notation representation of the number -/
def scientific_representation : ScientificNotation :=
  { coefficient := 1.23
    exponent := 6
    valid := by sorry }

/-- Theorem stating that the scientific notation representation is correct -/
theorem scientific_notation_correct :
  (scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent) = number := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l2216_221632


namespace NUMINAMATH_CALUDE_equi_partite_implies_a_equals_two_l2216_221619

/-- A complex number is equi-partite if its real and imaginary parts are equal -/
def is_equi_partite (z : ℂ) : Prop := z.re = z.im

/-- The complex number z in terms of a -/
def z (a : ℝ) : ℂ := (1 + a * Complex.I) - Complex.I

/-- Theorem: If z(a) is an equi-partite complex number, then a = 2 -/
theorem equi_partite_implies_a_equals_two (a : ℝ) :
  is_equi_partite (z a) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_equi_partite_implies_a_equals_two_l2216_221619


namespace NUMINAMATH_CALUDE_expression_value_l2216_221610

theorem expression_value : 
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  x^2 + y^2 - z^2 + 2*x*y + x*y*z = -7 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2216_221610


namespace NUMINAMATH_CALUDE_picture_placement_l2216_221683

theorem picture_placement (wall_width : ℝ) (picture_width : ℝ) (space_between : ℝ)
  (h1 : wall_width = 25)
  (h2 : picture_width = 2)
  (h3 : space_between = 1)
  (h4 : 2 * picture_width + space_between < wall_width) :
  let distance := (wall_width - (2 * picture_width + space_between)) / 2
  distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_picture_placement_l2216_221683


namespace NUMINAMATH_CALUDE_fraction_product_squares_l2216_221661

theorem fraction_product_squares : 
  (4/5)^2 * (3/7)^2 * (2/3)^2 = 64/1225 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_squares_l2216_221661


namespace NUMINAMATH_CALUDE_parentheses_make_equations_true_l2216_221607

theorem parentheses_make_equations_true : 
  (5 * (4 + 3) = 35) ∧ (32 / (9 - 5) = 8) := by
  sorry

end NUMINAMATH_CALUDE_parentheses_make_equations_true_l2216_221607


namespace NUMINAMATH_CALUDE_decimal_addition_l2216_221613

theorem decimal_addition : 5.763 + 2.489 = 8.152 := by sorry

end NUMINAMATH_CALUDE_decimal_addition_l2216_221613


namespace NUMINAMATH_CALUDE_test_questions_l2216_221679

theorem test_questions (total_points : ℕ) (four_point_questions : ℕ) 
  (h1 : total_points = 100)
  (h2 : four_point_questions = 10) : 
  ∃ (two_point_questions : ℕ),
    two_point_questions * 2 + four_point_questions * 4 = total_points ∧
    two_point_questions + four_point_questions = 40 := by
  sorry

end NUMINAMATH_CALUDE_test_questions_l2216_221679


namespace NUMINAMATH_CALUDE_martha_juice_bottles_l2216_221615

theorem martha_juice_bottles (initial_bottles pantry_bottles fridge_bottles consumed_bottles final_bottles : ℕ) 
  (h1 : initial_bottles = pantry_bottles + fridge_bottles)
  (h2 : pantry_bottles = 4)
  (h3 : fridge_bottles = 4)
  (h4 : consumed_bottles = 3)
  (h5 : final_bottles = 10) : 
  final_bottles - (initial_bottles - consumed_bottles) = 5 := by
  sorry

end NUMINAMATH_CALUDE_martha_juice_bottles_l2216_221615


namespace NUMINAMATH_CALUDE_count_385_consecutive_sums_l2216_221633

/-- Represents a sequence of consecutive positive integers -/
structure ConsecutiveSequence where
  start : ℕ
  length : ℕ
  length_ge_two : length ≥ 2

/-- The sum of a consecutive sequence -/
def sum_consecutive_sequence (seq : ConsecutiveSequence) : ℕ :=
  seq.length * (2 * seq.start + seq.length - 1) / 2

/-- Predicate for a valid sequence summing to 385 -/
def is_valid_sequence (seq : ConsecutiveSequence) : Prop :=
  sum_consecutive_sequence seq = 385

/-- The main theorem statement -/
theorem count_385_consecutive_sums :
  (∃ (seqs : Finset ConsecutiveSequence), 
    (∀ seq ∈ seqs, is_valid_sequence seq) ∧ 
    (∀ seq, is_valid_sequence seq → seq ∈ seqs) ∧
    seqs.card = 9) := by
  sorry

end NUMINAMATH_CALUDE_count_385_consecutive_sums_l2216_221633


namespace NUMINAMATH_CALUDE_median_of_special_sequence_l2216_221624

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem median_of_special_sequence : 
  let N : ℕ := sequence_sum 200
  let median_index : ℕ := N / 2
  let cumulative_count (n : ℕ) := sequence_sum n
  ∃ (n : ℕ), 
    cumulative_count n ≥ median_index ∧ 
    cumulative_count (n - 1) < median_index ∧
    n = 141 :=
by sorry

end NUMINAMATH_CALUDE_median_of_special_sequence_l2216_221624


namespace NUMINAMATH_CALUDE_no_solutions_in_interval_l2216_221625

theorem no_solutions_in_interval (x : ℝ) :
  x ∈ Set.Ioo 0 (π / 6) →
  3 * Real.tan (2 * x) - 4 * Real.tan (3 * x) ≠ Real.tan (3 * x) ^ 2 * Real.tan (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_in_interval_l2216_221625


namespace NUMINAMATH_CALUDE_max_value_of_g_l2216_221622

/-- Given a function f(x) = a*cos(x) + b where a and b are constants,
    if the maximum value of f(x) is 1 and the minimum value of f(x) is -7,
    then the maximum value of g(x) = a*cos(x) + b*sin(x) is 5. -/
theorem max_value_of_g (a b : ℝ) :
  (∃ x : ℝ, a * Real.cos x + b = 1) →
  (∃ x : ℝ, a * Real.cos x + b = -7) →
  (∃ x : ℝ, a * Real.cos x + b * Real.sin x = 5) ∧
  (∀ x : ℝ, a * Real.cos x + b * Real.sin x ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_g_l2216_221622


namespace NUMINAMATH_CALUDE_f_sqrt5_minus1_eq_neg_half_l2216_221618

def is_monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≤ f y

theorem f_sqrt5_minus1_eq_neg_half
  (f : ℝ → ℝ)
  (h1 : is_monotone_increasing f)
  (h2 : ∀ x > 0, f x * f (f x + 1 / x) = 1) :
  f (Real.sqrt 5 - 1) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_sqrt5_minus1_eq_neg_half_l2216_221618


namespace NUMINAMATH_CALUDE_problem_2015_l2216_221658

theorem problem_2015 : (2015^2 + 2015 - 1) / 2015 = 2016 - 1/2015 := by
  sorry

end NUMINAMATH_CALUDE_problem_2015_l2216_221658


namespace NUMINAMATH_CALUDE_solve_for_x_l2216_221670

theorem solve_for_x (x y : ℝ) (h1 : x + 3 * y = 10) (h2 : y = 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2216_221670


namespace NUMINAMATH_CALUDE_find_divisor_l2216_221666

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = 217 →
  quotient = 54 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 4 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2216_221666


namespace NUMINAMATH_CALUDE_arithmetic_sum_11_l2216_221612

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem: The sum of the first 11 terms of the arithmetic sequence
    with a₁ = -11 and d = 2 is equal to -11 -/
theorem arithmetic_sum_11 :
  arithmetic_sum (-11) 2 11 = -11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_11_l2216_221612


namespace NUMINAMATH_CALUDE_distance_between_stations_distance_is_65km_l2216_221699

/-- The distance between two stations given train travel information -/
theorem distance_between_stations : ℝ :=
let train_p_speed : ℝ := 20
let train_q_speed : ℝ := 25
let train_p_time : ℝ := 2
let train_q_time : ℝ := 1
let distance_p : ℝ := train_p_speed * train_p_time
let distance_q : ℝ := train_q_speed * train_q_time
distance_p + distance_q

/-- Proof that the distance between the stations is 65 km -/
theorem distance_is_65km : distance_between_stations = 65 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_stations_distance_is_65km_l2216_221699


namespace NUMINAMATH_CALUDE_quadratic_circle_intersection_l2216_221644

/-- Given a quadratic polynomial ax^2 + bx + c where a ≠ 0, if a circle passes through
    its three intersection points with the coordinate axes and intersects the y-axis
    at a fourth point with ordinate y₀, then y₀ = 1/a -/
theorem quadratic_circle_intersection 
  (a b c : ℝ) (h : a ≠ 0) : 
  ∃ (x₁ x₂ : ℝ), 
    (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
    (∃ y₀ : ℝ, y₀ * c = x₁ * x₂) →
    (∀ y₀ : ℝ, y₀ * c = x₁ * x₂ → y₀ = 1 / a) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_circle_intersection_l2216_221644


namespace NUMINAMATH_CALUDE_power_division_l2216_221630

theorem power_division (x : ℝ) : x^8 / x^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l2216_221630


namespace NUMINAMATH_CALUDE_remainder_of_55_power_55_plus_55_mod_56_l2216_221665

theorem remainder_of_55_power_55_plus_55_mod_56 :
  (55^55 + 55) % 56 = 54 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_55_power_55_plus_55_mod_56_l2216_221665


namespace NUMINAMATH_CALUDE_power_of_power_l2216_221647

theorem power_of_power (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2216_221647


namespace NUMINAMATH_CALUDE_pen_price_calculation_l2216_221686

theorem pen_price_calculation (total_cost : ℝ) (num_pens : ℕ) (num_pencils : ℕ) (pencil_price : ℝ) :
  total_cost = 690 →
  num_pens = 30 →
  num_pencils = 75 →
  pencil_price = 2 →
  (total_cost - num_pencils * pencil_price) / num_pens = 18 :=
by sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l2216_221686
