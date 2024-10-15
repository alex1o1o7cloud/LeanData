import Mathlib

namespace NUMINAMATH_CALUDE_side_is_one_third_perimeter_l4040_404021

-- Define a triangle with an inscribed circle
structure TriangleWithInscribedCircle where
  -- We don't need to explicitly define the triangle or circle, 
  -- just the properties we're interested in
  side_length : ℝ
  perimeter : ℝ
  midpoint : ℝ × ℝ
  altitude_foot : ℝ × ℝ
  tangency_point : ℝ × ℝ

-- Define the symmetry condition
def is_symmetrical (t : TriangleWithInscribedCircle) : Prop :=
  let midpoint_distance := (t.midpoint.1 - t.tangency_point.1)^2 + (t.midpoint.2 - t.tangency_point.2)^2
  let altitude_foot_distance := (t.altitude_foot.1 - t.tangency_point.1)^2 + (t.altitude_foot.2 - t.tangency_point.2)^2
  midpoint_distance = altitude_foot_distance

-- State the theorem
theorem side_is_one_third_perimeter (t : TriangleWithInscribedCircle) 
  (h : is_symmetrical t) : t.side_length = t.perimeter / 3 := by
  sorry

end NUMINAMATH_CALUDE_side_is_one_third_perimeter_l4040_404021


namespace NUMINAMATH_CALUDE_complex_number_with_conditions_l4040_404029

theorem complex_number_with_conditions (z : ℂ) :
  (((1 : ℂ) + 2 * Complex.I) * z).im = 0 →
  Complex.abs z = Real.sqrt 5 →
  z = 1 - 2 * Complex.I ∨ z = -1 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_with_conditions_l4040_404029


namespace NUMINAMATH_CALUDE_probability_reach_target_l4040_404003

-- Define the step type
inductive Step
| Left
| Right
| Up
| Down

-- Define the position type
structure Position :=
  (x : Int) (y : Int)

-- Define the function to take a step
def takeStep (pos : Position) (step : Step) : Position :=
  match step with
  | Step.Left  => ⟨pos.x - 1, pos.y⟩
  | Step.Right => ⟨pos.x + 1, pos.y⟩
  | Step.Up    => ⟨pos.x, pos.y + 1⟩
  | Step.Down  => ⟨pos.x, pos.y - 1⟩

-- Define the probability of a single step
def stepProbability : ℚ := 1/4

-- Define the function to check if a position is (3,1)
def isTarget (pos : Position) : Bool :=
  pos.x = 3 ∧ pos.y = 1

-- Define the theorem
theorem probability_reach_target :
  ∃ (paths : Finset (List Step)),
    (∀ path ∈ paths, path.length ≤ 8) ∧
    (∀ path ∈ paths, isTarget (path.foldl takeStep ⟨0, 0⟩)) ∧
    (paths.card : ℚ) * stepProbability ^ 8 = 7/128 :=
sorry

end NUMINAMATH_CALUDE_probability_reach_target_l4040_404003


namespace NUMINAMATH_CALUDE_c_range_l4040_404073

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def solution_set_is_real (f : ℝ → ℝ) : Prop :=
  ∀ x, f x > 1

theorem c_range (c : ℝ) (hc : c > 0) :
  let p := is_increasing (λ x => Real.log ((1 - c) * x - 1) / Real.log 10)
  let q := solution_set_is_real (λ x => x + |x - 2 * c|)
  (p ∨ q) ∧ ¬(p ∧ q) →
  c ∈ Set.Ioo 0 (1/2) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_c_range_l4040_404073


namespace NUMINAMATH_CALUDE_exactly_one_black_and_two_red_mutually_exclusive_but_not_complementary_l4040_404097

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Black

/-- Represents the outcome of drawing two balls -/
structure TwoDrawOutcome :=
  (first second : BallColor)

/-- The sample space of all possible outcomes when drawing two balls -/
def sampleSpace : Finset TwoDrawOutcome := sorry

/-- The event of drawing exactly one black ball -/
def exactlyOneBlack (outcome : TwoDrawOutcome) : Prop := sorry

/-- The event of drawing exactly two red balls -/
def exactlyTwoRed (outcome : TwoDrawOutcome) : Prop := sorry

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (event1 event2 : TwoDrawOutcome → Prop) : Prop := sorry

/-- Two events are complementary if their union is the entire sample space -/
def complementary (event1 event2 : TwoDrawOutcome → Prop) : Prop := sorry

theorem exactly_one_black_and_two_red_mutually_exclusive_but_not_complementary :
  mutuallyExclusive exactlyOneBlack exactlyTwoRed ∧
  ¬complementary exactlyOneBlack exactlyTwoRed := by sorry

end NUMINAMATH_CALUDE_exactly_one_black_and_two_red_mutually_exclusive_but_not_complementary_l4040_404097


namespace NUMINAMATH_CALUDE_vector_parallel_problem_l4040_404018

theorem vector_parallel_problem (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, m]
  let b : Fin 2 → ℝ := ![-1, 1]
  let c : Fin 2 → ℝ := ![3, 0]
  (∃ (k : ℝ), a = k • (b + c)) → m = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_problem_l4040_404018


namespace NUMINAMATH_CALUDE_pie_chart_probability_l4040_404007

theorem pie_chart_probability (W X Y Z : ℝ) : 
  W = 1/4 → X = 1/3 → Z = 1/6 → W + X + Y + Z = 1 → Y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_pie_chart_probability_l4040_404007


namespace NUMINAMATH_CALUDE_arithmetic_computation_l4040_404001

theorem arithmetic_computation : 5 * 7 + 6 * 12 + 10 * 4 + 7 * 6 + 30 / 5 = 195 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l4040_404001


namespace NUMINAMATH_CALUDE_inequality_proof_l4040_404055

theorem inequality_proof (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (hsum : x + y + z = 1) : 
  (2 * (x^2 + y^2 + z^2) + 9*x*y*z ≥ 1) ∧ 
  (x*y + y*z + z*x - 3*x*y*z ≤ 1/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4040_404055


namespace NUMINAMATH_CALUDE_expression_equals_3840_factorial_l4040_404027

/-- Custom factorial definition for positive p and b -/
def custom_factorial (p b : ℕ) : ℕ :=
  sorry

/-- The result of the expression 120₁₀!/20₃! + (10₂!)! -/
def expression_result : ℕ :=
  sorry

/-- Theorem stating that the expression equals (3840)! -/
theorem expression_equals_3840_factorial :
  expression_result = Nat.factorial 3840 :=
  sorry

end NUMINAMATH_CALUDE_expression_equals_3840_factorial_l4040_404027


namespace NUMINAMATH_CALUDE_camphor_ball_shrinkage_l4040_404031

/-- The time it takes for a camphor ball to shrink to a specific volume -/
theorem camphor_ball_shrinkage (a k : ℝ) (h1 : a > 0) (h2 : k > 0) : 
  let V : ℝ → ℝ := λ t => a * Real.exp (-k * t)
  (V 50 = 4/9 * a) → (V 75 = 8/27 * a) := by
  sorry

end NUMINAMATH_CALUDE_camphor_ball_shrinkage_l4040_404031


namespace NUMINAMATH_CALUDE_chess_tournament_games_l4040_404030

/-- The number of games in a chess tournament where each player plays twice with every other player -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 7 players, where each player plays twice with every other player, the total number of games played is 84 -/
theorem chess_tournament_games :
  tournament_games 7 * 2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l4040_404030


namespace NUMINAMATH_CALUDE_square_of_difference_with_sqrt_l4040_404017

theorem square_of_difference_with_sqrt (x : ℝ) : 
  (7 - Real.sqrt (x^2 - 49*x + 169))^2 = x^2 - 49*x + 218 - 14*Real.sqrt (x^2 - 49*x + 169) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_with_sqrt_l4040_404017


namespace NUMINAMATH_CALUDE_football_tournament_max_points_l4040_404010

theorem football_tournament_max_points (num_teams : ℕ) (points_win : ℕ) (points_draw : ℕ) (points_loss : ℕ) :
  num_teams = 15 →
  points_win = 3 →
  points_draw = 1 →
  points_loss = 0 →
  ∃ (N : ℕ), N = 34 ∧ 
    (∀ (M : ℕ), (∃ (teams : Finset (Fin num_teams)), teams.card ≥ 6 ∧ 
      (∀ t ∈ teams, ∃ (score : ℕ), score ≥ M)) → M ≤ N) :=
by sorry

end NUMINAMATH_CALUDE_football_tournament_max_points_l4040_404010


namespace NUMINAMATH_CALUDE_intersection_sum_l4040_404099

theorem intersection_sum (a b : ℚ) : 
  (∀ x y : ℚ, x = (1/3) * y + a ↔ y = (1/3) * x + b) →
  (3 : ℚ) = (1/3) * 1 + a →
  (1 : ℚ) = (1/3) * 3 + b →
  a + b = 8/3 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l4040_404099


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l4040_404040

theorem triangle_determinant_zero (A B C : Real) 
  (h : A + B + C = π) : -- Condition that A, B, C are angles of a triangle
  let M : Matrix (Fin 3) (Fin 3) Real := ![
    ![Real.cos A ^ 2, Real.tan A, 1],
    ![Real.cos B ^ 2, Real.tan B, 1],
    ![Real.cos C ^ 2, Real.tan C, 1]
  ]
  Matrix.det M = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l4040_404040


namespace NUMINAMATH_CALUDE_age_difference_proof_l4040_404077

theorem age_difference_proof (p m n : ℕ) 
  (h1 : 5 * p = 3 * m)  -- p:m = 3:5
  (h2 : 5 * m = 3 * n)  -- m:n = 3:5
  (h3 : p + m + n = 245) : 
  n - p = 80 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l4040_404077


namespace NUMINAMATH_CALUDE_expression_inequality_l4040_404067

theorem expression_inequality : 
  let x : ℚ := 3 + 1/10 + 4/100
  let y : ℚ := 3 + 5/110
  x ≠ y :=
by
  sorry

end NUMINAMATH_CALUDE_expression_inequality_l4040_404067


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4040_404080

theorem complex_equation_solution (z : ℂ) (h : z * (1 - Complex.I) = 1 + Complex.I) : z = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4040_404080


namespace NUMINAMATH_CALUDE_polynomial_expansion_property_l4040_404058

theorem polynomial_expansion_property (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 + x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₂ - a₁ + a₄ - a₃ = -15 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_property_l4040_404058


namespace NUMINAMATH_CALUDE_expression_value_l4040_404043

theorem expression_value (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = 0) 
  (h2 : a * b * c < 0) : 
  a / |a| + b / |b| + c / |c| = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4040_404043


namespace NUMINAMATH_CALUDE_work_increase_percentage_l4040_404096

theorem work_increase_percentage (p : ℕ) (W : ℝ) (h : p > 0) :
  let original_work_per_person := W / p
  let remaining_persons := (2 : ℝ) / 3 * p
  let new_work_per_person := W / remaining_persons
  (new_work_per_person - original_work_per_person) / original_work_per_person * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_work_increase_percentage_l4040_404096


namespace NUMINAMATH_CALUDE_trigonometric_calculations_l4040_404024

theorem trigonometric_calculations :
  (2 * Real.sin (60 * π / 180) - 3 * Real.tan (45 * π / 180) + Real.sqrt 9 = Real.sqrt 3) ∧
  (Real.cos (30 * π / 180) / (1 + Real.sin (30 * π / 180)) + Real.tan (60 * π / 180) = 4 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_calculations_l4040_404024


namespace NUMINAMATH_CALUDE_count_pears_l4040_404089

/-- Given a box of fruits with apples and pears, prove the number of pears. -/
theorem count_pears (total_fruits : ℕ) (apples : ℕ) (pears : ℕ) : 
  total_fruits = 51 → apples = 12 → total_fruits = pears + apples → pears = 39 := by
  sorry

end NUMINAMATH_CALUDE_count_pears_l4040_404089


namespace NUMINAMATH_CALUDE_duck_cow_problem_l4040_404068

theorem duck_cow_problem (D C : ℕ) : 
  2 * D + 4 * C = 2 * (D + C) + 24 → C = 12 := by
  sorry

end NUMINAMATH_CALUDE_duck_cow_problem_l4040_404068


namespace NUMINAMATH_CALUDE_sum_of_f_values_l4040_404019

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem sum_of_f_values : 
  Real.sqrt 3 * (f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + 
                 f 1 + f 2 + f 3 + f 4 + f 5 + f 6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_values_l4040_404019


namespace NUMINAMATH_CALUDE_f_minus_two_range_l4040_404081

def f (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem f_minus_two_range (a b : ℝ) :
  (1 ≤ f a b (-1) ∧ f a b (-1) ≤ 2) →
  (2 ≤ f a b 1 ∧ f a b 1 ≤ 4) →
  ∃ (y : ℝ), y = f a b (-2) ∧ 3 ≤ y ∧ y ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_f_minus_two_range_l4040_404081


namespace NUMINAMATH_CALUDE_stock_percent_change_l4040_404072

theorem stock_percent_change (initial_value : ℝ) : 
  let day1_value := initial_value * (1 - 0.1)
  let day2_value := day1_value * (1 + 0.2)
  (day2_value - initial_value) / initial_value = 0.08 := by
sorry

end NUMINAMATH_CALUDE_stock_percent_change_l4040_404072


namespace NUMINAMATH_CALUDE_area_relation_l4040_404012

-- Define the triangles
structure Triangle :=
  (O A B : ℝ × ℝ)

-- Define properties of isosceles right triangles
def IsIsoscelesRight (t : Triangle) : Prop :=
  let (xO, yO) := t.O
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  (xA - xO)^2 + (yA - yO)^2 = (xB - xO)^2 + (yB - yO)^2 ∧
  (xB - xA)^2 + (yB - yA)^2 = 2 * ((xA - xO)^2 + (yA - yO)^2)

-- Define the area of a triangle
def Area (t : Triangle) : ℝ :=
  let (xO, yO) := t.O
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  0.5 * abs ((xA - xO) * (yB - yO) - (xB - xO) * (yA - yO))

-- Theorem statement
theorem area_relation (OAB OBC OCD : Triangle) :
  IsIsoscelesRight OAB ∧ IsIsoscelesRight OBC ∧ IsIsoscelesRight OCD →
  Area OCD = 12 →
  Area OAB = 3 :=
by sorry

end NUMINAMATH_CALUDE_area_relation_l4040_404012


namespace NUMINAMATH_CALUDE_equal_sets_implies_sum_l4040_404098

-- Define sets A and B
def A (a b : ℝ) : Set ℝ := {a, b/a, 1}
def B (a b : ℝ) : Set ℝ := {a^2, a+b, 0}

-- Theorem statement
theorem equal_sets_implies_sum (a b : ℝ) (h : A a b = B a b) :
  a^2013 + b^2014 = -1 :=
sorry

end NUMINAMATH_CALUDE_equal_sets_implies_sum_l4040_404098


namespace NUMINAMATH_CALUDE_P_in_fourth_quadrant_iff_m_gt_two_l4040_404092

/-- A point P in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point P with coordinates (m, 2-m) -/
def P (m : ℝ) : Point :=
  ⟨m, 2 - m⟩

/-- Theorem stating that for P(m, 2-m) to be in the fourth quadrant, m > 2 -/
theorem P_in_fourth_quadrant_iff_m_gt_two (m : ℝ) :
  in_fourth_quadrant (P m) ↔ m > 2 := by
  sorry


end NUMINAMATH_CALUDE_P_in_fourth_quadrant_iff_m_gt_two_l4040_404092


namespace NUMINAMATH_CALUDE_equation_value_l4040_404083

theorem equation_value (a b c : ℝ) 
  (eq1 : 3 * a - 2 * b - 2 * c = 30)
  (eq2 : a + b + c = 10) :
  Real.sqrt (3 * a) - Real.sqrt (2 * b + 2 * c) = Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_l4040_404083


namespace NUMINAMATH_CALUDE_initial_wage_solution_l4040_404037

def initial_wage_problem (x : ℝ) : Prop :=
  let after_raise := x * 1.20
  let after_cut := after_raise * 0.75
  after_cut = 9

theorem initial_wage_solution :
  ∃ x : ℝ, initial_wage_problem x ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_wage_solution_l4040_404037


namespace NUMINAMATH_CALUDE_palmer_photos_l4040_404047

def total_photos (initial : ℕ) (first_week : ℕ) (third_fourth_week : ℕ) : ℕ :=
  initial + first_week + 2 * first_week + third_fourth_week

theorem palmer_photos : total_photos 100 50 80 = 330 := by
  sorry

end NUMINAMATH_CALUDE_palmer_photos_l4040_404047


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l4040_404085

/-- Given an arithmetic sequence where the sum of the third and fifth terms is 10,
    prove that the fourth term is 5. -/
theorem arithmetic_sequence_fourth_term (b y : ℝ) 
  (h : b + (b + 2*y) = 10) : b + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l4040_404085


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l4040_404034

/-- An isosceles triangle with base 8 and side difference 2 has sides of length 10 or 6 -/
theorem isosceles_triangle_side_length (AC BC : ℝ) : 
  BC = 8 → 
  |AC - BC| = 2 → 
  (AC = 10 ∨ AC = 6) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l4040_404034


namespace NUMINAMATH_CALUDE_store_refusal_illegal_l4040_404006

/-- Represents a banknote --/
structure Banknote where
  issued_by_bank_of_russia : Bool
  has_tears : Bool

/-- Represents the store's action --/
inductive StoreAction
  | Accept
  | Refuse

/-- Determines if a banknote is legal tender --/
def is_legal_tender (note : Banknote) : Prop :=
  note.issued_by_bank_of_russia ∧ (note.has_tears ∨ ¬note.has_tears)

/-- Determines if the store's action is legal --/
def is_legal_action (note : Banknote) (action : StoreAction) : Prop :=
  is_legal_tender note → action = StoreAction.Accept

/-- The main theorem --/
theorem store_refusal_illegal 
  (lydia_note : Banknote)
  (h1 : lydia_note.has_tears)
  (h2 : lydia_note.issued_by_bank_of_russia)
  (h3 : ∀ (note : Banknote), note.has_tears → is_legal_tender note)
  (store_action : StoreAction)
  (h4 : store_action = StoreAction.Refuse) :
  ¬(is_legal_action lydia_note store_action) :=
by sorry

end NUMINAMATH_CALUDE_store_refusal_illegal_l4040_404006


namespace NUMINAMATH_CALUDE_sheepdog_roundup_percentage_l4040_404016

theorem sheepdog_roundup_percentage 
  (total_sheep : ℕ) 
  (wandered_off : ℕ) 
  (in_pen : ℕ) 
  (h1 : wandered_off = total_sheep / 10)
  (h2 : wandered_off = 9)
  (h3 : in_pen = 81)
  (h4 : total_sheep = in_pen + wandered_off) :
  (in_pen : ℚ) / total_sheep * 100 = 90 := by
sorry

end NUMINAMATH_CALUDE_sheepdog_roundup_percentage_l4040_404016


namespace NUMINAMATH_CALUDE_largest_positive_root_bound_l4040_404088

theorem largest_positive_root_bound (b₂ b₁ b₀ : ℝ) 
  (h₂ : |b₂| ≤ 3) (h₁ : |b₁| ≤ 5) (h₀ : |b₀| ≤ 3) :
  ∃ s : ℝ, s > 4 ∧ s < 5 ∧
  (∀ x : ℝ, x > 0 → x^3 + b₂*x^2 + b₁*x + b₀ = 0 → x ≤ s) ∧
  (∃ b₂' b₁' b₀' : ℝ, |b₂'| ≤ 3 ∧ |b₁'| ≤ 5 ∧ |b₀'| ≤ 3 ∧
    s^3 + b₂'*s^2 + b₁'*s + b₀' = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_positive_root_bound_l4040_404088


namespace NUMINAMATH_CALUDE_no_integer_roots_l4040_404009

theorem no_integer_roots (a b : ℤ) : ¬ ∃ x : ℤ, x^2 + 3*a*x + 3*(2 - b^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l4040_404009


namespace NUMINAMATH_CALUDE_factor_expression_l4040_404075

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l4040_404075


namespace NUMINAMATH_CALUDE_recurrence_sequence_has_composite_l4040_404062

/-- A sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (a (n + 1) = 2 * a n + 1 ∨ a (n + 1) = 2 * a n - 1)

/-- A number is composite if it's greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

/-- The main theorem stating that any sequence satisfying the recurrence relation contains a composite number -/
theorem recurrence_sequence_has_composite
  (a : ℕ → ℕ)
  (h_seq : RecurrenceSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_nonconstant : ∃ m n, m ≠ n ∧ a m ≠ a n) :
  ∃ k, IsComposite (a k) :=
sorry

end NUMINAMATH_CALUDE_recurrence_sequence_has_composite_l4040_404062


namespace NUMINAMATH_CALUDE_multiply_63_57_l4040_404093

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end NUMINAMATH_CALUDE_multiply_63_57_l4040_404093


namespace NUMINAMATH_CALUDE_unique_row_contains_101_l4040_404071

/-- The number of rows in Pascal's Triangle that contain the number 101 -/
def rows_containing_101 : ℕ := 1

/-- 101 is a prime number -/
axiom prime_101 : Nat.Prime 101

/-- A number appears in Pascal's Triangle if it's a binomial coefficient -/
def appears_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (row k : ℕ), Nat.choose row k = n

theorem unique_row_contains_101 :
  (∃! row : ℕ, appears_in_pascals_triangle 101 ∧ row > 0) ∧
  rows_containing_101 = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_row_contains_101_l4040_404071


namespace NUMINAMATH_CALUDE_list_property_l4040_404084

theorem list_property (list : List ℝ) (n : ℝ) : 
  list.Nodup →
  n ∈ list →
  n = 4 * ((list.sum - n) / (list.length - 1)) →
  n = (1 / 6) * list.sum →
  list.length = 21 := by
  sorry

end NUMINAMATH_CALUDE_list_property_l4040_404084


namespace NUMINAMATH_CALUDE_matching_probability_five_pairs_l4040_404052

/-- A box containing shoes -/
structure ShoeBox where
  pairs : ℕ
  total : ℕ
  total_eq : total = 2 * pairs

/-- The probability of selecting a matching pair of shoes -/
def matchingProbability (box : ShoeBox) : ℚ :=
  box.pairs / (box.total * (box.total - 1) / 2)

/-- Theorem: The probability of selecting a matching pair from a box with 5 pairs is 1/9 -/
theorem matching_probability_five_pairs :
  let box : ShoeBox := ⟨5, 10, rfl⟩
  matchingProbability box = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_matching_probability_five_pairs_l4040_404052


namespace NUMINAMATH_CALUDE_total_bike_rides_l4040_404035

theorem total_bike_rides (billy_rides : ℕ) (john_rides : ℕ) (mother_rides : ℕ) : 
  billy_rides = 17 →
  john_rides = 2 * billy_rides →
  mother_rides = john_rides + 10 →
  billy_rides + john_rides + mother_rides = 95 := by
sorry

end NUMINAMATH_CALUDE_total_bike_rides_l4040_404035


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_numbers_l4040_404059

/-- Given 5 consecutive even numbers whose sum is 240, prove that the smallest of these numbers is 44 -/
theorem smallest_of_five_consecutive_even_numbers (n : ℕ) : 
  (∃ (a b c d : ℕ), 
    a = n + 2 ∧ 
    b = n + 4 ∧ 
    c = n + 6 ∧ 
    d = n + 8 ∧ 
    n + a + b + c + d = 240 ∧ 
    Even n ∧ Even a ∧ Even b ∧ Even c ∧ Even d) → 
  n = 44 :=
by sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_numbers_l4040_404059


namespace NUMINAMATH_CALUDE_not_a_implies_condition_l4040_404076

/-- Represents a student in the course -/
structure Student :=
  (name : String)

/-- Represents the exam result for a student -/
structure ExamResult :=
  (student : Student)
  (allMultipleChoiceCorrect : Bool)
  (essayScore : ℝ)
  (receivedA : Bool)

/-- The professor's grading policy -/
axiom grading_policy : 
  ∀ (result : ExamResult), 
    result.allMultipleChoiceCorrect ∧ result.essayScore ≥ 80 → result.receivedA

/-- The theorem to be proved -/
theorem not_a_implies_condition (result : ExamResult) : 
  ¬result.receivedA → ¬result.allMultipleChoiceCorrect ∨ result.essayScore < 80 :=
sorry

end NUMINAMATH_CALUDE_not_a_implies_condition_l4040_404076


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4040_404046

theorem sufficient_not_necessary_condition :
  ∃ (S : Set ℝ), 
    (∀ x ∈ S, x^2 - 4*x < 0) ∧ 
    (S ⊂ {x : ℝ | 0 < x ∧ x < 4}) ∧
    S = {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4040_404046


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_a_value_for_minimum_four_l4040_404095

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + |a*x - 5|

theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x ≥ 9} = {x : ℝ | x ≤ -1 ∨ x > 5} := by sorry

theorem a_value_for_minimum_four :
  ∀ a : ℝ, 0 < a → a < 5 → (∃ m : ℝ, ∀ x : ℝ, f a x ≥ m ∧ (∃ y : ℝ, f a y = m) ∧ m = 4) → a = 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_a_value_for_minimum_four_l4040_404095


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l4040_404032

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

/-- Theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions : 
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 8 := by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l4040_404032


namespace NUMINAMATH_CALUDE_troy_vegetable_purchase_l4040_404002

/-- The number of pounds of vegetables Troy buys -/
def vegetable_pounds : ℝ := 6

/-- The number of pounds of beef Troy buys -/
def beef_pounds : ℝ := 4

/-- The cost of vegetables per pound in dollars -/
def vegetable_cost_per_pound : ℝ := 2

/-- The total cost of Troy's purchase in dollars -/
def total_cost : ℝ := 36

/-- Theorem stating that the number of pounds of vegetables Troy buys is 6 -/
theorem troy_vegetable_purchase :
  vegetable_pounds = 6 ∧
  beef_pounds = 4 ∧
  vegetable_cost_per_pound = 2 ∧
  total_cost = 36 ∧
  (3 * vegetable_cost_per_pound * beef_pounds + vegetable_cost_per_pound * vegetable_pounds = total_cost) :=
by sorry

end NUMINAMATH_CALUDE_troy_vegetable_purchase_l4040_404002


namespace NUMINAMATH_CALUDE_circle_center_transformation_l4040_404057

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 - d)

theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (-2, 6)
  let reflected := reflect_y initial_center
  let final_position := translate_down reflected 8
  final_position = (2, -2) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l4040_404057


namespace NUMINAMATH_CALUDE_alvin_friend_gave_wood_l4040_404061

/-- The number of pieces of wood Alvin needs in total -/
def total_needed : ℕ := 376

/-- The number of pieces of wood Alvin's brother gave him -/
def brother_gave : ℕ := 136

/-- The number of pieces of wood Alvin still needs to gather -/
def still_needed : ℕ := 117

/-- The number of pieces of wood Alvin's friend gave him -/
def friend_gave : ℕ := total_needed - brother_gave - still_needed

theorem alvin_friend_gave_wood : friend_gave = 123 := by
  sorry

end NUMINAMATH_CALUDE_alvin_friend_gave_wood_l4040_404061


namespace NUMINAMATH_CALUDE_cross_section_area_theorem_l4040_404074

/-- Regular hexagonal pyramid -/
structure HexagonalPyramid where
  base_side : ℝ
  height : ℝ

/-- Cutting plane for the pyramid -/
structure CuttingPlane where
  distance_from_apex : ℝ

/-- The area of the cross-section of a regular hexagonal pyramid -/
noncomputable def cross_section_area (p : HexagonalPyramid) (c : CuttingPlane) : ℝ :=
  sorry

/-- Theorem stating the area of the cross-section for the given conditions -/
theorem cross_section_area_theorem (p : HexagonalPyramid) (c : CuttingPlane) :
  p.base_side = 2 →
  c.distance_from_apex = 1 →
  cross_section_area p c = 34 * Real.sqrt 3 / 35 :=
sorry

end NUMINAMATH_CALUDE_cross_section_area_theorem_l4040_404074


namespace NUMINAMATH_CALUDE_complex_modulus_theorem_l4040_404025

theorem complex_modulus_theorem (ω : ℂ) (h : ω = 8 + I) : 
  Complex.abs (ω^2 - 4*ω + 13) = 4 * Real.sqrt 130 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_theorem_l4040_404025


namespace NUMINAMATH_CALUDE_factorial_simplification_l4040_404013

theorem factorial_simplification : (15 : ℕ).factorial / ((12 : ℕ).factorial + 3 * (11 : ℕ).factorial) = 4680 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l4040_404013


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4040_404036

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (1 / a + 2 / b) ≥ 3 / 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4040_404036


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l4040_404051

/-- The speed of cyclist C in miles per hour -/
def speed_C : ℝ := 10

/-- The speed of cyclist D in miles per hour -/
def speed_D : ℝ := speed_C + 5

/-- The distance to the town in miles -/
def distance_to_town : ℝ := 90

/-- The distance from the meeting point to the town in miles -/
def distance_meeting_to_town : ℝ := 18

theorem cyclist_speed_problem :
  /- Given conditions -/
  (speed_D = speed_C + 5) →
  (distance_to_town = 90) →
  (distance_meeting_to_town = 18) →
  /- The time taken by C to reach the meeting point equals
     the time taken by D to reach the town and return to the meeting point -/
  ((distance_to_town - distance_meeting_to_town) / speed_C =
   (distance_to_town + distance_meeting_to_town) / speed_D) →
  /- Conclusion: The speed of cyclist C is 10 mph -/
  speed_C = 10 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l4040_404051


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l4040_404090

-- Define the equation of motion
def s (t : ℝ) : ℝ := -t + t^2

-- Define the velocity function
def v (t : ℝ) : ℝ := (-1) + 2*t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 5 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l4040_404090


namespace NUMINAMATH_CALUDE_composite_function_solution_l4040_404028

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := x^2 + 5
def g (x : ℝ) : ℝ := x^2 - 3
def h (x : ℝ) : ℝ := 2*x + 1

-- State the theorem
theorem composite_function_solution (a : ℝ) (ha : a > 0) 
  (h_eq : f (g (h a)) = 17) : 
  a = (-1 + Real.sqrt (3 + 2 * Real.sqrt 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_solution_l4040_404028


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_four_is_smallest_k_l4040_404079

/-- The quadratic equation 3x(kx-5)-x^2+7=0 has no real roots when k ≥ 4 -/
theorem smallest_k_no_real_roots : 
  ∀ k : ℤ, (∀ x : ℝ, 3 * x * (k * x - 5) - x^2 + 7 ≠ 0) ↔ k ≥ 4 :=
by sorry

/-- 4 is the smallest integer k for which 3x(kx-5)-x^2+7=0 has no real roots -/
theorem four_is_smallest_k : 
  ∀ k : ℤ, k < 4 → ∃ x : ℝ, 3 * x * (k * x - 5) - x^2 + 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_four_is_smallest_k_l4040_404079


namespace NUMINAMATH_CALUDE_complement_of_M_l4040_404011

def M : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 5}

theorem complement_of_M :
  (Set.univ : Set ℝ) \ M = {x : ℝ | x < -3 ∨ x ≥ 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l4040_404011


namespace NUMINAMATH_CALUDE_smallest_b_value_l4040_404064

def is_factor (m n : ℕ) : Prop := n % m = 0

theorem smallest_b_value (a b : ℕ) : 
  a = 363 → 
  is_factor 112 (a * 43 * 62 * b) → 
  is_factor 33 (a * 43 * 62 * b) → 
  b ≥ 56 ∧ is_factor 112 (a * 43 * 62 * 56) ∧ is_factor 33 (a * 43 * 62 * 56) :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l4040_404064


namespace NUMINAMATH_CALUDE_set_equivalence_l4040_404014

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem set_equivalence : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equivalence_l4040_404014


namespace NUMINAMATH_CALUDE_even_increasing_nonpositive_property_l4040_404041

-- Define an even function that is increasing on (-∞, 0]
def is_even_and_increasing_nonpositive (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y)

-- State the theorem
theorem even_increasing_nonpositive_property 
  (f : ℝ → ℝ) (h : is_even_and_increasing_nonpositive f) :
  ∀ a : ℝ, f (a^2) > f (a^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_even_increasing_nonpositive_property_l4040_404041


namespace NUMINAMATH_CALUDE_smallest_fraction_l4040_404022

theorem smallest_fraction (x : ℝ) (h : x > 2022) :
  min (x / 2022) (min (2022 / (x - 1)) (min ((x + 1) / 2022) (min (2022 / x) (2022 / (x + 1))))) = 2022 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_l4040_404022


namespace NUMINAMATH_CALUDE_eugenes_living_room_length_l4040_404048

/-- Represents the properties of a rectangular room --/
structure RectangularRoom where
  width : ℝ
  area : ℝ
  length : ℝ

/-- Theorem stating the length of Eugene's living room --/
theorem eugenes_living_room_length (room : RectangularRoom)
  (h1 : room.width = 14)
  (h2 : room.area = 215.6)
  (h3 : room.area = room.length * room.width) :
  room.length = 15.4 := by
  sorry

end NUMINAMATH_CALUDE_eugenes_living_room_length_l4040_404048


namespace NUMINAMATH_CALUDE_shirts_per_pants_l4040_404066

/-- 
Given:
- Mr. Jones has 40 pants.
- The total number of pieces of clothes he owns is 280.
- Mr. Jones has a certain number of shirts for every pair of pants.

Prove that Mr. Jones has 6 shirts for every pair of pants.
-/
theorem shirts_per_pants (num_pants : ℕ) (total_clothes : ℕ) (shirts_per_pants : ℕ) : 
  num_pants = 40 → total_clothes = 280 → shirts_per_pants * num_pants + num_pants = total_clothes → 
  shirts_per_pants = 6 := by
  sorry

end NUMINAMATH_CALUDE_shirts_per_pants_l4040_404066


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l4040_404063

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 → -- angles are supplementary
  a / b = 5 / 3 → -- ratio of angles is 5:3
  b = 67.5 -- smaller angle is 67.5°
  := by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l4040_404063


namespace NUMINAMATH_CALUDE_handshakes_theorem_l4040_404078

/-- Calculate the number of handshakes in a single meeting -/
def handshakes_in_meeting (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculate the total number of handshakes in two meetings -/
def total_handshakes (first_meeting_attendees second_meeting_attendees overlap : ℕ) : ℕ :=
  handshakes_in_meeting first_meeting_attendees +
  handshakes_in_meeting second_meeting_attendees -
  handshakes_in_meeting overlap

/-- Prove that the total number of handshakes in the two meetings is 41 -/
theorem handshakes_theorem :
  let first_meeting_attendees : ℕ := 7
  let second_meeting_attendees : ℕ := 7
  let overlap : ℕ := 2
  total_handshakes first_meeting_attendees second_meeting_attendees overlap = 41 := by
  sorry

#eval total_handshakes 7 7 2

end NUMINAMATH_CALUDE_handshakes_theorem_l4040_404078


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4040_404033

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 0}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4040_404033


namespace NUMINAMATH_CALUDE_difference_of_odd_squares_divisible_by_eight_l4040_404054

theorem difference_of_odd_squares_divisible_by_eight (a b : Int) 
  (ha : a % 2 = 1) (hb : b % 2 = 1) : 
  ∃ k : Int, a^2 - b^2 = 8 * k := by
sorry

end NUMINAMATH_CALUDE_difference_of_odd_squares_divisible_by_eight_l4040_404054


namespace NUMINAMATH_CALUDE_no_solution_condition_l4040_404053

theorem no_solution_condition (r : ℝ) :
  (∀ x y : ℝ, x^2 = y^2 ∧ (x - r)^2 + y^2 = 1 → False) ↔ r < -Real.sqrt 2 ∨ r > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l4040_404053


namespace NUMINAMATH_CALUDE_triangle_properties_l4040_404020

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- angles
  (a b c : Real)  -- sides opposite to A, B, C respectively

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.b * Real.sin abc.A = Real.sqrt 3 * abc.a * Real.cos abc.B)
  (h2 : abc.b = 3)
  (h3 : Real.sin abc.C = 2 * Real.sin abc.A) :
  (abc.B = π/3) ∧ (abc.a = Real.sqrt 3) ∧ (abc.c = 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l4040_404020


namespace NUMINAMATH_CALUDE_inequality_not_always_preserved_l4040_404094

theorem inequality_not_always_preserved (a b : ℝ) (h : a < b) :
  ∃ m : ℝ, m^2 * a ≤ m^2 * b :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_preserved_l4040_404094


namespace NUMINAMATH_CALUDE_nina_widget_problem_l4040_404045

theorem nina_widget_problem (x : ℝ) 
  (h1 : 15 * x = 25 * (x - 5)) : 
  15 * x = 187.50 := by
  sorry

end NUMINAMATH_CALUDE_nina_widget_problem_l4040_404045


namespace NUMINAMATH_CALUDE_dog_collar_nylon_l4040_404050

/-- The number of inches of nylon needed for one cat collar -/
def cat_collar_nylon : ℕ := 10

/-- The total number of inches of nylon needed for 9 dog collars and 3 cat collars -/
def total_nylon : ℕ := 192

/-- The number of dog collars made -/
def num_dog_collars : ℕ := 9

/-- The number of cat collars made -/
def num_cat_collars : ℕ := 3

/-- Theorem stating that 18 inches of nylon are needed for one dog collar -/
theorem dog_collar_nylon : 
  ∃ (x : ℕ), x * num_dog_collars + cat_collar_nylon * num_cat_collars = total_nylon ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_dog_collar_nylon_l4040_404050


namespace NUMINAMATH_CALUDE_molecular_weight_proof_l4040_404044

/-- Given a compound where 3 moles have a molecular weight of 222,
    prove that the molecular weight of 1 mole is 74 g/mol. -/
theorem molecular_weight_proof (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 222)
  (h2 : num_moles = 3) :
  total_weight / num_moles = 74 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_proof_l4040_404044


namespace NUMINAMATH_CALUDE_right_triangle_roots_l4040_404000

theorem right_triangle_roots (p : ℝ) : 
  (∃ a b c : ℝ, 
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (a^3 - 2*p*(p+1)*a^2 + (p^4 + 4*p^3 - 1)*a - 3*p^3 = 0) ∧
    (b^3 - 2*p*(p+1)*b^2 + (p^4 + 4*p^3 - 1)*b - 3*p^3 = 0) ∧
    (c^3 - 2*p*(p+1)*c^2 + (p^4 + 4*p^3 - 1)*c - 3*p^3 = 0) ∧
    (a^2 + b^2 = c^2)) ↔ 
  p = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_roots_l4040_404000


namespace NUMINAMATH_CALUDE_solve_linear_system_l4040_404049

theorem solve_linear_system (a b : ℤ) 
  (eq1 : 2009 * a + 2013 * b = 2021)
  (eq2 : 2011 * a + 2015 * b = 2023) :
  a - b = -5 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_system_l4040_404049


namespace NUMINAMATH_CALUDE_jill_study_time_l4040_404039

/-- Calculates the total minutes studied over 3 days given a specific study pattern -/
def totalMinutesStudied (day1Hours : ℕ) : ℕ :=
  let day2Hours := 2 * day1Hours
  let day3Hours := day2Hours - 1
  (day1Hours + day2Hours + day3Hours) * 60

/-- Proves that given Jill's study pattern, she studies for 540 minutes over 3 days -/
theorem jill_study_time : totalMinutesStudied 2 = 540 := by
  sorry

end NUMINAMATH_CALUDE_jill_study_time_l4040_404039


namespace NUMINAMATH_CALUDE_binomial_unique_parameters_l4040_404056

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  hp : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: For a binomial random variable X with E(X) = 1.6 and D(X) = 1.28, n = 8 and p = 0.2 -/
theorem binomial_unique_parameters :
  ∀ X : BinomialRV,
  expectation X = 1.6 →
  variance X = 1.28 →
  X.n = 8 ∧ X.p = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_unique_parameters_l4040_404056


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l4040_404087

theorem polynomial_remainder_theorem (x : ℝ) : 
  (8 * x^3 - 20 * x^2 + 28 * x - 30) % (4 * x - 8) = 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l4040_404087


namespace NUMINAMATH_CALUDE_probability_two_hearts_one_spade_l4040_404070

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of hearts in a standard deck -/
def numberOfHearts : ℕ := 13

/-- The number of spades in a standard deck -/
def numberOfSpades : ℕ := 13

/-- The probability of drawing two hearts followed by a spade from a standard 52-card deck -/
theorem probability_two_hearts_one_spade :
  (numberOfHearts * (numberOfHearts - 1) * numberOfSpades : ℚ) / 
  (standardDeckSize * (standardDeckSize - 1) * (standardDeckSize - 2)) = 78 / 5115 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_hearts_one_spade_l4040_404070


namespace NUMINAMATH_CALUDE_right_triangle_solution_l4040_404004

theorem right_triangle_solution (A B C : Real) (a b c : ℝ) :
  -- Given conditions
  (A + B + C = π) →  -- Sum of angles in a triangle
  (C = π / 2) →      -- Right angle at C
  (a = Real.sqrt 5) →
  (b = Real.sqrt 15) →
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem
  (Real.tan A = a / b) →  -- Definition of tangent
  -- Conclusions
  (c = 2 * Real.sqrt 5) ∧
  (A = π / 6) ∧  -- 30 degrees in radians
  (B = π / 3) :=  -- 60 degrees in radians
by sorry

end NUMINAMATH_CALUDE_right_triangle_solution_l4040_404004


namespace NUMINAMATH_CALUDE_rachels_brownies_l4040_404023

/-- Rachel's brownie problem -/
theorem rachels_brownies (total : ℕ) (left_at_home : ℕ) (brought_to_school : ℕ) : 
  total = 40 → left_at_home = 24 → brought_to_school = total - left_at_home → brought_to_school = 16 := by
  sorry

#check rachels_brownies

end NUMINAMATH_CALUDE_rachels_brownies_l4040_404023


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_ratio_l4040_404060

/-- The ratio of the volume of a sphere inscribed in a right circular cylinder
    to the volume of the cylinder. -/
theorem sphere_cylinder_volume_ratio :
  ∀ (r : ℝ), r > 0 →
  (4 / 3 * π * r^3) / (π * r^2 * (2 * r)) = 2 * Real.sqrt 3 * π / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_ratio_l4040_404060


namespace NUMINAMATH_CALUDE_only_point_distance_no_conditional_l4040_404038

-- Define the four types of mathematical problems
inductive MathProblem
  | QuadraticEquation
  | LineCircleRelationship
  | StudentRanking
  | PointDistance

-- Define a function that determines if a problem requires conditional statements
def requiresConditionalStatements (problem : MathProblem) : Prop :=
  match problem with
  | MathProblem.QuadraticEquation => true
  | MathProblem.LineCircleRelationship => true
  | MathProblem.StudentRanking => true
  | MathProblem.PointDistance => false

-- Theorem stating that only PointDistance does not require conditional statements
theorem only_point_distance_no_conditional :
  ∀ (problem : MathProblem),
    ¬(requiresConditionalStatements problem) ↔ problem = MathProblem.PointDistance := by
  sorry

#check only_point_distance_no_conditional

end NUMINAMATH_CALUDE_only_point_distance_no_conditional_l4040_404038


namespace NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l4040_404065

-- Define the quadratic function
def f (x : ℝ) := -4 * x^2 + 4 * x + 7

-- State the theorem
theorem quadratic_function_satisfies_conditions :
  (f 2 = -1) ∧ 
  (f (-1) = -1) ∧ 
  (∀ x : ℝ, f x ≤ 8) ∧
  (∃ x : ℝ, f x = 8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l4040_404065


namespace NUMINAMATH_CALUDE_shoe_selection_problem_l4040_404015

theorem shoe_selection_problem (n : ℕ) (h : n = 10) : 
  (n.choose 1) * ((n - 1).choose 2) * (2^2) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_shoe_selection_problem_l4040_404015


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4040_404008

/-- A sequence where each term is twice the previous term -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h1 : GeometricSequence a) 
  (h2 : a 1 + a 4 = 2) : 
  a 5 + a 8 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4040_404008


namespace NUMINAMATH_CALUDE_quadrilateral_ABCD_is_parallelogram_l4040_404026

-- Define the vertices of the quadrilateral
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, -1)
def C : ℝ × ℝ := (4, 2)
def D : ℝ × ℝ := (2, 3)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def DC : ℝ × ℝ := (C.1 - D.1, C.2 - D.2)

-- Define a function to check if two vectors are equal
def vectors_equal (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 = v2.1 ∧ v1.2 = v2.2

-- Define what it means for a quadrilateral to be a parallelogram
def is_parallelogram (a b c d : ℝ × ℝ) : Prop :=
  vectors_equal (b.1 - a.1, b.2 - a.2) (c.1 - d.1, c.2 - d.2) ∧
  vectors_equal (c.1 - b.1, c.2 - b.2) (d.1 - a.1, d.2 - a.2)

-- Theorem statement
theorem quadrilateral_ABCD_is_parallelogram :
  is_parallelogram A B C D :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_ABCD_is_parallelogram_l4040_404026


namespace NUMINAMATH_CALUDE_class_size_l4040_404091

theorem class_size (n : ℕ) (h1 : n > 0) :
  (∃ (x : ℕ), x > 0 ∧ x = 6 + 7 - 1) →
  3 * n = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_l4040_404091


namespace NUMINAMATH_CALUDE_consecutive_integer_averages_l4040_404069

theorem consecutive_integer_averages (a : ℤ) (c : ℚ) : 
  (a > 0) →
  (c = (7 * a + 21) / 7) →
  ((7 * c + 21) / 7 : ℚ) = a + 6 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integer_averages_l4040_404069


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l4040_404042

theorem trigonometric_inequality (φ : Real) (h : 0 < φ ∧ φ < Real.pi / 2) :
  Real.sin (Real.cos φ) < Real.cos φ ∧ Real.cos φ < Real.cos (Real.sin φ) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l4040_404042


namespace NUMINAMATH_CALUDE_apple_pear_equivalence_l4040_404086

theorem apple_pear_equivalence : 
  ∀ (apple_value pear_value : ℚ),
  (3/4 * 16 : ℚ) * apple_value = 10 * pear_value →
  (2/5 * 20 : ℚ) * apple_value = (20/3 : ℚ) * pear_value := by
sorry

end NUMINAMATH_CALUDE_apple_pear_equivalence_l4040_404086


namespace NUMINAMATH_CALUDE_opposite_of_negative_seven_thirds_l4040_404082

theorem opposite_of_negative_seven_thirds :
  ∃ y : ℚ, -7/3 + y = 0 ∧ y = 7/3 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_negative_seven_thirds_l4040_404082


namespace NUMINAMATH_CALUDE_ring_toss_earnings_l4040_404005

theorem ring_toss_earnings (daily_earnings : ℕ) (days : ℕ) (total_earnings : ℕ) : 
  daily_earnings = 144 → days = 22 → total_earnings = daily_earnings * days → total_earnings = 3168 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_earnings_l4040_404005
