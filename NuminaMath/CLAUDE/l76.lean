import Mathlib

namespace NUMINAMATH_CALUDE_ribbon_ratio_l76_7622

theorem ribbon_ratio : 
  ∀ (original reduced : ℕ), 
  original = 55 → reduced = 35 → 
  (original : ℚ) / (reduced : ℚ) = 11 / 7 := by
sorry

end NUMINAMATH_CALUDE_ribbon_ratio_l76_7622


namespace NUMINAMATH_CALUDE_angles_on_squared_paper_l76_7625

/-- Three angles marked on squared paper sum to 90 degrees -/
theorem angles_on_squared_paper (α β γ : ℝ) : α + β + γ = 90 := by
  sorry

end NUMINAMATH_CALUDE_angles_on_squared_paper_l76_7625


namespace NUMINAMATH_CALUDE_select_questions_theorem_l76_7636

/-- The number of ways to select 3 questions from a set of questions with the given conditions -/
def select_questions (multiple_choice : ℕ) (fill_in_blank : ℕ) (open_ended : ℕ) : ℕ :=
  let total_questions := multiple_choice + fill_in_blank + open_ended
  let one_each := Nat.choose multiple_choice 1 * Nat.choose fill_in_blank 1 * Nat.choose open_ended 1
  let two_multiple_one_open := Nat.choose multiple_choice 2 * Nat.choose open_ended 1
  let one_multiple_two_open := Nat.choose multiple_choice 1 * Nat.choose open_ended 2
  one_each + two_multiple_one_open + one_multiple_two_open

theorem select_questions_theorem :
  select_questions 12 4 6 = 864 := by
  sorry

end NUMINAMATH_CALUDE_select_questions_theorem_l76_7636


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l76_7611

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) : (x - 3*y) - (y - 2*x) = 3*x - 4*y := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) : 5*a*b^2 - 3*(2*a^2*b - 2*(a^2*b - 2*a*b^2)) = -7*a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l76_7611


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l76_7634

def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

theorem complement_intersection_equals_set : 
  (A ∪ B)ᶜ ∩ (A ∩ B)ᶜ = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l76_7634


namespace NUMINAMATH_CALUDE_alternating_sequence_sum_l76_7667

def arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def alternating_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  arithmetic_sequence_sum a₁ (2 * d) ((n + 1) / 2) -
  arithmetic_sequence_sum (a₁ + d) (2 * d) (n / 2)

theorem alternating_sequence_sum :
  alternating_sum 2 3 19 = 29 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sequence_sum_l76_7667


namespace NUMINAMATH_CALUDE_litter_size_l76_7675

/-- Represents the number of puppies in the litter -/
def puppies : ℕ := sorry

/-- The profit John makes from selling the puppies -/
def profit : ℕ := 1500

/-- The amount John pays to the stud owner -/
def stud_fee : ℕ := 300

/-- The price for which John sells each puppy -/
def price_per_puppy : ℕ := 600

theorem litter_size : 
  puppies = 8 ∧ 
  (puppies / 2 - 1) * price_per_puppy - stud_fee = profit :=
sorry

end NUMINAMATH_CALUDE_litter_size_l76_7675


namespace NUMINAMATH_CALUDE_passes_through_neg1_0_two_a_plus_c_positive_roots_between_neg3_and_1_l76_7646

/-- A parabola defined by y = ax^2 - 2ax + c, where a and c are constants, a ≠ 0, c > 0,
    and the parabola passes through the point (3,0) -/
structure Parabola where
  a : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  c_positive : c > 0
  passes_through_3_0 : a * 3^2 - 2 * a * 3 + c = 0

/-- The parabola passes through the point (-1,0) -/
theorem passes_through_neg1_0 (p : Parabola) : p.a * (-1)^2 - 2 * p.a * (-1) + p.c = 0 := by sorry

/-- 2a + c > 0 -/
theorem two_a_plus_c_positive (p : Parabola) : 2 * p.a + p.c > 0 := by sorry

/-- If m and n (m < n) are the two roots of ax^2 + 2ax + c = p, where p > 0,
    then -3 < m < n < 1 -/
theorem roots_between_neg3_and_1 (p : Parabola) (m n : ℝ) (p_pos : ℝ) 
  (h_roots : m < n ∧ p.a * m^2 + 2 * p.a * m + p.c = p_pos ∧ p.a * n^2 + 2 * p.a * n + p.c = p_pos)
  (h_p_pos : p_pos > 0) : -3 < m ∧ m < n ∧ n < 1 := by sorry

end NUMINAMATH_CALUDE_passes_through_neg1_0_two_a_plus_c_positive_roots_between_neg3_and_1_l76_7646


namespace NUMINAMATH_CALUDE_remainder_seven_n_l76_7618

theorem remainder_seven_n (n : ℤ) (h : n % 5 = 3) : (7 * n) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_n_l76_7618


namespace NUMINAMATH_CALUDE_work_completion_time_l76_7658

theorem work_completion_time 
  (a_time b_time c_time : ℝ) 
  (ha : a_time = 7) 
  (hb : b_time = 14) 
  (hc : c_time = 28) : 
  1 / (1 / a_time + 1 / b_time + 1 / c_time) = 4 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l76_7658


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l76_7691

theorem right_triangle_acute_angles (α β : ℝ) : 
  α + β = 90 →  -- sum of acute angles in a right triangle is 90°
  α = 54 →      -- one acute angle is 54°
  β = 36 :=     -- the other acute angle is 36°
by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l76_7691


namespace NUMINAMATH_CALUDE_yardwork_earnings_contribution_l76_7616

def earnings : List ℕ := [18, 22, 30, 35, 45]
def max_contribution : ℕ := 40
def num_friends : ℕ := 5

theorem yardwork_earnings_contribution :
  let total := (earnings.sum - 45 + max_contribution)
  let equal_share := total / num_friends
  35 - equal_share = 6 := by sorry

end NUMINAMATH_CALUDE_yardwork_earnings_contribution_l76_7616


namespace NUMINAMATH_CALUDE_p_less_q_less_r_l76_7664

-- Define the logarithm function (base 2)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem p_less_q_less_r (a b : ℝ) 
  (h1 : a > b) (h2 : b > 1) 
  (P : ℝ) (hP : P = lg a * lg b)
  (Q : ℝ) (hQ : Q = lg a + lg b)
  (R : ℝ) (hR : R = lg (a * b)) :
  P < Q ∧ Q < R := by
  sorry

end NUMINAMATH_CALUDE_p_less_q_less_r_l76_7664


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l76_7673

theorem inequality_system_solutions : 
  {x : ℤ | x ≥ 0 ∧ 5*x - 1 < 3*(x + 1) ∧ (1 - x) / 3 ≤ 1} = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l76_7673


namespace NUMINAMATH_CALUDE_parallelogram_area_l76_7697

/-- The area of a parallelogram with sides a and b and angle γ between them is ab sin γ -/
theorem parallelogram_area (a b γ : ℝ) (ha : a > 0) (hb : b > 0) (hγ : 0 < γ ∧ γ < π) :
  ∃ S : ℝ, S = a * b * Real.sin γ ∧ S > 0 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l76_7697


namespace NUMINAMATH_CALUDE_only_zhong_symmetric_l76_7662

-- Define a type for Chinese characters
inductive ChineseCharacter : Type
  | ai   : ChineseCharacter  -- 爱
  | wo   : ChineseCharacter  -- 我
  | zhong : ChineseCharacter  -- 中
  | hua  : ChineseCharacter  -- 华

-- Define a property for vertical symmetry
def hasVerticalSymmetry (c : ChineseCharacter) : Prop :=
  match c with
  | ChineseCharacter.zhong => True
  | _ => False

-- Theorem stating that only 中 (zhong) has vertical symmetry
theorem only_zhong_symmetric :
  ∀ (c : ChineseCharacter),
    hasVerticalSymmetry c ↔ c = ChineseCharacter.zhong :=
by
  sorry


end NUMINAMATH_CALUDE_only_zhong_symmetric_l76_7662


namespace NUMINAMATH_CALUDE_max_volume_regular_pyramid_l76_7654

/-- 
For a regular n-sided pyramid with surface area S, 
prove that the maximum volume V is given by the formula:
V = (√2 / 12) * (S^(3/2)) / √(n * tan(π/n))
-/
theorem max_volume_regular_pyramid (n : ℕ) (S : ℝ) (h₁ : n ≥ 3) (h₂ : S > 0) :
  ∃ V : ℝ, V = (Real.sqrt 2 / 12) * S^(3/2) / Real.sqrt (n * Real.tan (π / n)) ∧
    ∀ V' : ℝ, (∃ (Q h : ℝ), V' = (1/3) * Q * h ∧ 
      S = Q + n * Q / (2 * Real.cos (π / n))) → V' ≤ V := by
  sorry


end NUMINAMATH_CALUDE_max_volume_regular_pyramid_l76_7654


namespace NUMINAMATH_CALUDE_jerry_added_figures_l76_7641

/-- Represents the shelf of action figures -/
structure ActionFigureShelf :=
  (initial_count : Nat)
  (final_count : Nat)
  (removed_count : Nat)
  (is_arithmetic_sequence : Bool)
  (first_last_preserved : Bool)
  (common_difference_preserved : Bool)

/-- Calculates the number of action figures added to the shelf -/
def added_figures (shelf : ActionFigureShelf) : Nat :=
  shelf.final_count + shelf.removed_count - shelf.initial_count

/-- Theorem stating the number of added action figures -/
theorem jerry_added_figures (shelf : ActionFigureShelf) 
  (h1 : shelf.initial_count = 7)
  (h2 : shelf.final_count = 8)
  (h3 : shelf.removed_count = 10)
  (h4 : shelf.is_arithmetic_sequence = true)
  (h5 : shelf.first_last_preserved = true)
  (h6 : shelf.common_difference_preserved = true) :
  added_figures shelf = 18 := by
  sorry

#check jerry_added_figures

end NUMINAMATH_CALUDE_jerry_added_figures_l76_7641


namespace NUMINAMATH_CALUDE_ellipse_fixed_points_l76_7685

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  center : Point

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

/-- Calculates the dot product of two vectors -/
def dotProduct (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Theorem about the existence of fixed points Q on the ellipse -/
theorem ellipse_fixed_points (e : Ellipse) (f a : Point) :
  e.a = 2 ∧ e.b = Real.sqrt 3 ∧ e.center = Point.mk 0 0 ∧
  f = Point.mk 1 0 ∧ a = Point.mk (-2) 0 →
  ∃ q1 q2 : Point,
    q1 = Point.mk 1 0 ∧ q2 = Point.mk 7 0 ∧
    ∀ b c m n : Point,
      isOnEllipse e b ∧ isOnEllipse e c ∧
      b ≠ c ∧
      (∃ t : ℝ, b.x = t * b.y + 1 ∧ c.x = t * c.y + 1) ∧
      m.x = 4 ∧ n.x = 4 ∧
      (m.y - a.y) / (m.x - a.x) = (b.y - a.y) / (b.x - a.x) ∧
      (n.y - a.y) / (n.x - a.x) = (c.y - a.y) / (c.x - a.x) →
      dotProduct (Point.mk (q1.x - m.x) (q1.y - m.y)) (Point.mk (q1.x - n.x) (q1.y - n.y)) = 0 ∧
      dotProduct (Point.mk (q2.x - m.x) (q2.y - m.y)) (Point.mk (q2.x - n.x) (q2.y - n.y)) = 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_fixed_points_l76_7685


namespace NUMINAMATH_CALUDE_students_interested_in_both_l76_7601

theorem students_interested_in_both (total : ℕ) (sports : ℕ) (entertainment : ℕ) (neither : ℕ) :
  total = 1400 →
  sports = 1250 →
  entertainment = 952 →
  neither = 60 →
  ∃ x : ℕ, x = 862 ∧
    total = neither + x + (sports - x) + (entertainment - x) :=
by sorry

end NUMINAMATH_CALUDE_students_interested_in_both_l76_7601


namespace NUMINAMATH_CALUDE_diamond_two_three_l76_7699

def diamond (a b : ℝ) : ℝ := a * b^2 - b + 1

theorem diamond_two_three : diamond 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_three_l76_7699


namespace NUMINAMATH_CALUDE_unique_four_digit_reverse_l76_7676

/-- Reverses the digits of a four-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  d * 1000 + c * 100 + b * 10 + a

/-- Checks if a number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_reverse : ∃! n : ℕ, is_four_digit n ∧ 4 * n = reverse_digits n :=
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_reverse_l76_7676


namespace NUMINAMATH_CALUDE_largest_integer_in_special_set_l76_7660

theorem largest_integer_in_special_set (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d →                   -- Four different integers
  (a + b + c + d) / 4 = 70 →                -- Average is 70
  a ≥ 13 →                                  -- Smallest integer is at least 13
  d ≤ 238 :=                                -- Largest integer is at most 238
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_special_set_l76_7660


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l76_7688

theorem polynomial_product_sum (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) : 
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃)) →
  b₁*c₁ + b₂*c₂ + b₃*c₃ = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l76_7688


namespace NUMINAMATH_CALUDE_inverse_of_exponential_function_l76_7661

noncomputable def f (x : ℝ) : ℝ := 3^x

theorem inverse_of_exponential_function (x : ℝ) (h : x > 0) : 
  f⁻¹ x = Real.log x / Real.log 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_of_exponential_function_l76_7661


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l76_7613

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

theorem reflection_across_y_axis :
  let P : Point := { x := 4, y := -1 }
  reflectAcrossYAxis P = { x := -4, y := -1 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l76_7613


namespace NUMINAMATH_CALUDE_remainder_problem_l76_7680

theorem remainder_problem (n : ℤ) (h : 2 * n % 15 = 2) : n % 30 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l76_7680


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l76_7628

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) → a ∈ Set.Icc (-8) 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l76_7628


namespace NUMINAMATH_CALUDE_current_rate_calculation_l76_7620

/-- Given a boat with speed in still water and its downstream travel details, 
    calculate the rate of the current. -/
theorem current_rate_calculation (boat_speed : ℝ) (distance : ℝ) (time : ℝ) :
  boat_speed = 15 →
  distance = 3.6 →
  time = 1/5 →
  ∃ (current_rate : ℝ), current_rate = 3 ∧ 
    distance = (boat_speed + current_rate) * time :=
by sorry

end NUMINAMATH_CALUDE_current_rate_calculation_l76_7620


namespace NUMINAMATH_CALUDE_truncation_result_l76_7668

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  edges : ℕ
  convex : Bool

/-- Represents a truncated convex polyhedron -/
structure TruncatedPolyhedron where
  original : ConvexPolyhedron
  vertices : ℕ
  edges : ℕ
  truncated : Bool

/-- Function that performs truncation on a convex polyhedron -/
def truncate (p : ConvexPolyhedron) : TruncatedPolyhedron :=
  { original := p
  , vertices := 2 * p.edges
  , edges := 3 * p.edges
  , truncated := true }

/-- Theorem stating the result of truncating a specific convex polyhedron -/
theorem truncation_result :
  ∀ (p : ConvexPolyhedron),
  p.edges = 100 →
  p.convex = true →
  let tp := truncate p
  tp.vertices = 200 ∧ tp.edges = 300 := by
  sorry

end NUMINAMATH_CALUDE_truncation_result_l76_7668


namespace NUMINAMATH_CALUDE_finite_moves_l76_7642

/-- Represents the position of a number after m minutes -/
def position (initial_pos : ℕ) (m : ℕ) : ℕ :=
  if m ∣ initial_pos then initial_pos + m - 1 else initial_pos - 1

/-- Represents whether a number at initial_pos has moved after m minutes -/
def has_moved (initial_pos : ℕ) (m : ℕ) : Prop :=
  position initial_pos m ≠ initial_pos

/-- The main theorem stating that each natural number moves only finitely many times -/
theorem finite_moves (n : ℕ) : ∃ (M : ℕ), ∀ (m : ℕ), m ≥ M → ¬(has_moved n m) := by
  sorry


end NUMINAMATH_CALUDE_finite_moves_l76_7642


namespace NUMINAMATH_CALUDE_alyssa_soccer_spending_l76_7696

/-- Calculates the total amount Alyssa spends on soccer games over three years -/
def total_soccer_spending (
  year1_games : ℕ)
  (year2_in_person : ℕ)
  (year2_missed : ℕ)
  (year2_online : ℕ)
  (year2_streaming_cost : ℕ)
  (year3_in_person : ℕ)
  (year3_online : ℕ)
  (year3_friends_games : ℕ)
  (year3_streaming_cost : ℕ)
  (ticket_price : ℕ) : ℕ :=
  let year1_cost := year1_games * ticket_price
  let year2_cost := year2_in_person * ticket_price + year2_streaming_cost
  let year3_cost := year3_in_person * ticket_price + year3_streaming_cost
  let friends_payback := year3_friends_games * 2 * ticket_price
  year1_cost + year2_cost + year3_cost - friends_payback

/-- Theorem stating that Alyssa's total spending on soccer games over three years is $850 -/
theorem alyssa_soccer_spending :
  total_soccer_spending 13 11 12 8 120 15 10 5 150 20 = 850 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_soccer_spending_l76_7696


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_system_l76_7602

theorem smallest_solution_congruence_system :
  ∃ (x : ℕ), x > 0 ∧ 
    (6 * x) % 31 = 17 % 31 ∧
    x % 7 = 3 % 7 ∧
    (∀ (y : ℕ), y > 0 ∧ (6 * y) % 31 = 17 % 31 ∧ y % 7 = 3 % 7 → x ≤ y) ∧
    x = 24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_system_l76_7602


namespace NUMINAMATH_CALUDE_frog_hop_probability_l76_7635

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Represents the possible moves -/
inductive Move
  | Up
  | Down
  | Left
  | Right
  | Stay

/-- Defines whether a position is on the edge of the grid -/
def isEdge (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Applies a move to a position, wrapping around the grid -/
def applyMove (p : Position) (m : Move) : Position :=
  match m with
  | Move.Up    => ⟨p.x, (p.y + 1) % 4⟩
  | Move.Down  => ⟨p.x, (p.y - 1 + 4) % 4⟩
  | Move.Left  => ⟨(p.x - 1 + 4) % 4, p.y⟩
  | Move.Right => ⟨(p.x + 1) % 4, p.y⟩
  | Move.Stay  => p

/-- Calculates the probability of reaching an edge after exactly n hops -/
def probReachEdge (start : Position) (n : Nat) : ℚ :=
  sorry  -- Actual implementation would go here

/-- The main theorem to prove -/
theorem frog_hop_probability :
  probReachEdge ⟨1, 1⟩ 5 = 605 / 625 := by
  sorry

end NUMINAMATH_CALUDE_frog_hop_probability_l76_7635


namespace NUMINAMATH_CALUDE_equation_linear_implies_a_equals_one_l76_7643

theorem equation_linear_implies_a_equals_one (a : ℝ) :
  (∀ x, (a^2 - 1) * x^2 - a*x - x + 2 = 0 → ∃ m b, (a^2 - 1) * x^2 - a*x - x + 2 = m*x + b) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_linear_implies_a_equals_one_l76_7643


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l76_7689

theorem inequality_not_always_true (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ¬ (∀ a b c, c < b ∧ b < a ∧ a * c < 0 → c * b < a * b) := by
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l76_7689


namespace NUMINAMATH_CALUDE_complex_equality_sum_l76_7619

theorem complex_equality_sum (a b : ℝ) : 
  (a + b * Complex.I : ℂ) = Complex.I ^ 2 → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_sum_l76_7619


namespace NUMINAMATH_CALUDE_volume_maximized_at_one_meter_l76_7659

/-- Represents the dimensions of a rectangular box --/
structure BoxDimensions where
  x : Real  -- Length of the shorter side of the base
  h : Real  -- Height of the box

/-- Calculates the volume of the box given its dimensions --/
def boxVolume (d : BoxDimensions) : Real :=
  2 * d.x^2 * d.h

/-- Calculates the total wire length used for the box frame --/
def wireLengthUsed (d : BoxDimensions) : Real :=
  12 * d.x + 4 * d.h

/-- Theorem stating that the volume is maximized when the shorter side is 1m --/
theorem volume_maximized_at_one_meter :
  ∃ (d : BoxDimensions),
    wireLengthUsed d = 18 ∧
    (∀ (d' : BoxDimensions), wireLengthUsed d' = 18 → boxVolume d' ≤ boxVolume d) ∧
    d.x = 1 :=
  sorry

end NUMINAMATH_CALUDE_volume_maximized_at_one_meter_l76_7659


namespace NUMINAMATH_CALUDE_rolling_circle_traces_hypotrochoid_l76_7639

/-- A circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point on a 2D plane -/
def Point := ℝ × ℝ

/-- Represents a hypotrochoid curve -/
def Hypotrochoid := Point → ℝ → Point

theorem rolling_circle_traces_hypotrochoid 
  (large_circle : Circle)
  (small_circle : Circle)
  (h1 : large_circle.radius = 2 * small_circle.radius)
  (h2 : small_circle.radius > 0)
  (point : Point) 
  (h3 : ∃ (θ : ℝ), point = 
    (small_circle.center.1 + small_circle.radius * Real.cos θ, 
     small_circle.center.2 + small_circle.radius * Real.sin θ))
  : ∃ (curve : Hypotrochoid), 
    ∀ (t : ℝ), curve point t = 
      ((large_circle.radius - small_circle.radius) * Real.cos t + small_circle.radius * Real.cos ((large_circle.radius / small_circle.radius - 1) * t),
       (large_circle.radius - small_circle.radius) * Real.sin t - small_circle.radius * Real.sin ((large_circle.radius / small_circle.radius - 1) * t)) :=
by sorry

end NUMINAMATH_CALUDE_rolling_circle_traces_hypotrochoid_l76_7639


namespace NUMINAMATH_CALUDE_rational_function_value_l76_7612

-- Define the polynomials p and q
def p (k m : ℝ) (x : ℝ) : ℝ := k * x + m
def q (x : ℝ) : ℝ := (x + 4) * (x - 1)

-- State the theorem
theorem rational_function_value (k m : ℝ) :
  (p k m 0) / (q 0) = 0 →
  (p k m 2) / (q 2) = -1 →
  (p k m (-1)) / (q (-1)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l76_7612


namespace NUMINAMATH_CALUDE_peters_class_size_l76_7629

/-- Represents the number of students with a specific number of hands -/
structure HandDistribution :=
  (hands : ℕ)
  (students : ℕ)

/-- Represents the class data -/
structure ClassData :=
  (total_hands : ℕ)
  (distribution : List HandDistribution)
  (unspecified_students : ℕ)

/-- Calculates the total number of students in Peter's class -/
def total_students (data : ClassData) : ℕ :=
  (data.distribution.map (λ d => d.students)).sum + data.unspecified_students + 1

/-- Theorem stating that the total number of students in Peter's class is 17 -/
theorem peters_class_size (data : ClassData) 
  (h1 : data.total_hands = 20)
  (h2 : data.distribution = [
    ⟨2, 7⟩, 
    ⟨1, 3⟩, 
    ⟨3, 1⟩, 
    ⟨0, 2⟩
  ])
  (h3 : data.unspecified_students = 3) :
  total_students data = 17 := by
  sorry

end NUMINAMATH_CALUDE_peters_class_size_l76_7629


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l76_7694

theorem smallest_number_satisfying_conditions : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 6 = 2 ∧ 
  n % 7 = 3 ∧ 
  n % 8 = 4 ∧ 
  ∀ m : ℕ, m > 0 → m % 6 = 2 → m % 7 = 3 → m % 8 = 4 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l76_7694


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l76_7614

/-- The minimum number of additional coins needed for unique distribution -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (friends * (friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum additional coins needed for Alex's distribution -/
theorem alex_coin_distribution (friends : ℕ) (initial_coins : ℕ) 
  (h1 : friends = 15) (h2 : initial_coins = 94) :
  min_additional_coins friends initial_coins = 26 := by
  sorry

#eval min_additional_coins 15 94

end NUMINAMATH_CALUDE_alex_coin_distribution_l76_7614


namespace NUMINAMATH_CALUDE_sin_90_degrees_l76_7600

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l76_7600


namespace NUMINAMATH_CALUDE_angles_on_x_axis_characterization_l76_7657

/-- The set of angles with terminal sides on the x-axis -/
def AnglesOnXAxis : Set ℝ := {α | ∃ k : ℤ, α = k * Real.pi}

/-- Theorem: The set of angles with terminal sides on the x-axis is equal to {α | α = kπ, k ∈ ℤ} -/
theorem angles_on_x_axis_characterization :
  AnglesOnXAxis = {α : ℝ | ∃ k : ℤ, α = k * Real.pi} := by
  sorry

end NUMINAMATH_CALUDE_angles_on_x_axis_characterization_l76_7657


namespace NUMINAMATH_CALUDE_modular_inverse_11_mod_101_l76_7655

theorem modular_inverse_11_mod_101 :
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 100 ∧ (11 * x) % 101 = 1 :=
by
  use 46
  sorry

end NUMINAMATH_CALUDE_modular_inverse_11_mod_101_l76_7655


namespace NUMINAMATH_CALUDE_f_2014_equals_2_l76_7670

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_2014_equals_2
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x * f (x + 2) = 1)
  (h2 : f 1 = 3)
  (h3 : f 2 = 2) :
  f 2014 = 2 :=
sorry

end NUMINAMATH_CALUDE_f_2014_equals_2_l76_7670


namespace NUMINAMATH_CALUDE_complement_union_theorem_l76_7679

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_theorem : (U \ A) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l76_7679


namespace NUMINAMATH_CALUDE_unique_positive_solution_l76_7695

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (2/3) * x = (144/216) * (1/x) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l76_7695


namespace NUMINAMATH_CALUDE_factor_quadratic_l76_7608

theorem factor_quadratic (t : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, -6 * x^2 + 17 * x + 7 = k * (x - t)) ↔ 
  (t = (17 + Real.sqrt 457) / 12 ∨ t = (17 - Real.sqrt 457) / 12) :=
by sorry

end NUMINAMATH_CALUDE_factor_quadratic_l76_7608


namespace NUMINAMATH_CALUDE_cakes_ratio_l76_7621

/-- Carter's usual weekly baking schedule -/
def usual_cheesecakes : ℕ := 6
def usual_muffins : ℕ := 5
def usual_red_velvet : ℕ := 8

/-- Total number of cakes Carter usually bakes in a week -/
def usual_total : ℕ := usual_cheesecakes + usual_muffins + usual_red_velvet

/-- Additional cakes baked this week -/
def additional_cakes : ℕ := 38

/-- Theorem stating the ratio of cakes baked this week to usual weeks -/
theorem cakes_ratio :
  ∃ (x : ℕ), x * usual_total = usual_total + additional_cakes ∧
  (x * usual_total : ℚ) / usual_total = 3 := by
  sorry

end NUMINAMATH_CALUDE_cakes_ratio_l76_7621


namespace NUMINAMATH_CALUDE_mary_uber_time_l76_7650

/-- Mary's business trip timeline --/
def business_trip_timeline (t : ℕ) : Prop :=
  let uber_to_house := t
  let uber_to_airport := 5 * t
  let check_bag := 15
  let security := 3 * 15
  let wait_boarding := 20
  let wait_takeoff := 2 * 20
  uber_to_house + uber_to_airport + check_bag + security + wait_boarding + wait_takeoff = 180

theorem mary_uber_time : ∃ t : ℕ, business_trip_timeline t ∧ t = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_uber_time_l76_7650


namespace NUMINAMATH_CALUDE_binomial_coefficient_1500_2_l76_7630

theorem binomial_coefficient_1500_2 : Nat.choose 1500 2 = 1124250 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1500_2_l76_7630


namespace NUMINAMATH_CALUDE_expectation_of_function_l76_7632

-- Define the random variable ξ
variable (ξ : ℝ → ℝ)

-- Define the expectation operator
noncomputable def E (X : ℝ → ℝ) : ℝ := sorry

-- Define the variance operator
noncomputable def D (X : ℝ → ℝ) : ℝ := E (fun x => (X x - E X)^2)

theorem expectation_of_function (ξ : ℝ → ℝ) 
  (h1 : E ξ = -1) 
  (h2 : D ξ = 3) : 
  E (fun x => 3 * ((ξ x)^2 - 2)) = 6 := 
sorry

end NUMINAMATH_CALUDE_expectation_of_function_l76_7632


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l76_7672

/-- A rectangular parallelepiped divided into eight parts -/
structure Parallelepiped where
  volume_A : ℝ
  volume_C : ℝ
  volume_B_prime : ℝ
  volume_C_prime : ℝ

/-- The theorem stating that the total volume of the parallelepiped is 790 -/
theorem parallelepiped_volume 
  (p : Parallelepiped) 
  (h1 : p.volume_A = 40)
  (h2 : p.volume_C = 300)
  (h3 : p.volume_B_prime = 360)
  (h4 : p.volume_C_prime = 90) :
  p.volume_A + p.volume_C + p.volume_B_prime + p.volume_C_prime = 790 :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l76_7672


namespace NUMINAMATH_CALUDE_unique_c_value_l76_7649

-- Define the function f(x) = x⋅(2x+1)
def f (x : ℝ) : ℝ := x * (2 * x + 1)

-- Define the open interval (-2, 3/2)
def interval : Set ℝ := {x | -2 < x ∧ x < 3/2}

-- State the theorem
theorem unique_c_value : ∃! c : ℝ, ∀ x : ℝ, x ∈ interval ↔ f x < c :=
  sorry

end NUMINAMATH_CALUDE_unique_c_value_l76_7649


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l76_7605

def h (t : ℝ) : ℝ := -4.9 * t^2 + 10 * t

theorem instantaneous_velocity_at_one :
  (deriv h) 1 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l76_7605


namespace NUMINAMATH_CALUDE_plot_length_l76_7638

/-- Proves that the length of a rectangular plot is 55 meters given the specified conditions -/
theorem plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) : 
  length = breadth + 10 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  total_cost = perimeter * cost_per_meter →
  length = 55 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_l76_7638


namespace NUMINAMATH_CALUDE_second_smallest_hotdog_pack_l76_7698

def is_valid_hotdog_pack (n : ℕ) : Prop :=
  ∃ b : ℕ, 12 * n - 10 * b = 6 ∧ n % 5 = 3

theorem second_smallest_hotdog_pack :
  ∃ n : ℕ, is_valid_hotdog_pack n ∧
  (∀ m : ℕ, m < n → ¬is_valid_hotdog_pack m ∨ 
   (∃ k : ℕ, k < m ∧ is_valid_hotdog_pack k)) ∧
  n = 8 :=
sorry

end NUMINAMATH_CALUDE_second_smallest_hotdog_pack_l76_7698


namespace NUMINAMATH_CALUDE_average_and_difference_l76_7648

theorem average_and_difference (x : ℝ) : 
  (35 + x) / 2 = 45 → |x - 35| = 20 := by
sorry

end NUMINAMATH_CALUDE_average_and_difference_l76_7648


namespace NUMINAMATH_CALUDE_three_false_propositions_l76_7678

theorem three_false_propositions :
  (¬ ∀ a b : ℝ, (1 / a < 1 / b) → (a > b)) ∧
  (¬ ∀ a b c : ℝ, (a > b ∧ b > c) → (a * |c| > b * |c|)) ∧
  (¬ ∃ x₀ : ℝ, ∀ x : ℝ, x + 1/x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_three_false_propositions_l76_7678


namespace NUMINAMATH_CALUDE_f_max_at_neg_four_l76_7617

-- Define the function
def f (x : ℝ) : ℝ := -x^2 - 8*x + 16

-- State the theorem
theorem f_max_at_neg_four :
  ∀ x : ℝ, f x ≤ f (-4) :=
by sorry

end NUMINAMATH_CALUDE_f_max_at_neg_four_l76_7617


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l76_7653

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a < b ∧ b < 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l76_7653


namespace NUMINAMATH_CALUDE_sin_cube_identity_l76_7633

theorem sin_cube_identity (θ : ℝ) : 
  Real.sin θ ^ 3 = (-1/4 : ℝ) * Real.sin (3 * θ) + (3/4 : ℝ) * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sin_cube_identity_l76_7633


namespace NUMINAMATH_CALUDE_unique_n_satisfying_equation_l76_7687

theorem unique_n_satisfying_equation : ∃! (n : ℕ), 
  n + Int.floor (Real.sqrt n) + Int.floor (Real.sqrt (Real.sqrt n)) = 2017 :=
by sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_equation_l76_7687


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l76_7652

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 8*x + c

-- State the theorem
theorem quadratic_inequality_solution (c : ℝ) :
  (c > 0) → (∃ x, f c x < 0) ↔ (c > 0 ∧ c < 16) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l76_7652


namespace NUMINAMATH_CALUDE_max_triangle_area_l76_7665

/-- Line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 6 = 0

/-- Circle C in the xy-plane -/
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

/-- Point on circle C -/
def point_on_C (P : ℝ × ℝ) : Prop := circle_C P.1 P.2

/-- Intersection points of line l and circle C -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

/-- Area of triangle PAB -/
noncomputable def triangle_area (P A B : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Maximum area of triangle PAB -/
theorem max_triangle_area :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  ∃ max_area : ℝ, max_area = (27 * Real.sqrt 3) / 4 ∧
  ∀ P : ℝ × ℝ, point_on_C P → triangle_area P A B ≤ max_area :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l76_7665


namespace NUMINAMATH_CALUDE_expected_value_is_point_seven_l76_7686

/-- A two-point distribution random variable -/
structure TwoPointDistribution where
  p : ℝ  -- Probability of X=1
  h1 : 0 ≤ p ∧ p ≤ 1  -- Probability is between 0 and 1
  h2 : p - (1 - p) = 0.4  -- Given condition

/-- Expected value of a two-point distribution -/
def expectedValue (X : TwoPointDistribution) : ℝ := X.p

theorem expected_value_is_point_seven (X : TwoPointDistribution) :
  expectedValue X = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_point_seven_l76_7686


namespace NUMINAMATH_CALUDE_polygon_division_theorem_l76_7637

/-- A polygon is a closed planar figure with straight sides -/
structure Polygon where
  sides : ℕ
  is_closed : Bool
  is_planar : Bool

/-- Represents a division of a polygon into shapes -/
structure PolygonDivision (P : Polygon) (n : ℕ) (shape : Type) where
  num_divisions : ℕ
  is_valid : Bool

/-- Given a polygon that can be divided into 100 rectangles but not 99,
    it cannot be divided into 100 triangles -/
theorem polygon_division_theorem (P : Polygon) 
  (h1 : ∃ (d : PolygonDivision P 100 Rectangle), d.is_valid)
  (h2 : ¬ ∃ (d : PolygonDivision P 99 Rectangle), d.is_valid) :
  ¬ ∃ (d : PolygonDivision P 100 Triangle), d.is_valid :=
by sorry

end NUMINAMATH_CALUDE_polygon_division_theorem_l76_7637


namespace NUMINAMATH_CALUDE_power_multiplication_l76_7624

theorem power_multiplication (m : ℝ) : m^2 * m^3 = m^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l76_7624


namespace NUMINAMATH_CALUDE_monitor_pixel_count_l76_7647

/-- Calculate the total number of pixels on a monitor given its dimensions and pixel density. -/
theorem monitor_pixel_count (width : ℕ) (height : ℕ) (pixel_density : ℕ) : 
  width = 32 → height = 18 → pixel_density = 150 → 
  width * height * pixel_density * pixel_density = 12960000 := by
  sorry

end NUMINAMATH_CALUDE_monitor_pixel_count_l76_7647


namespace NUMINAMATH_CALUDE_divisible_count_equality_l76_7693

theorem divisible_count_equality (n : Nat) : n = 56000 →
  (Finset.filter (fun x => x % 7 = 0 ∧ x % 8 ≠ 0) (Finset.range (n + 1))).card =
  (Finset.filter (fun x => x % 8 = 0) (Finset.range (n + 1))).card := by
  sorry

end NUMINAMATH_CALUDE_divisible_count_equality_l76_7693


namespace NUMINAMATH_CALUDE_common_elements_count_l76_7671

def S := Finset.range 2005
def T := Finset.range 2005

def multiples_of_4 (n : ℕ) : ℕ := (n + 1) * 4
def multiples_of_6 (n : ℕ) : ℕ := (n + 1) * 6

def S_set := S.image multiples_of_4
def T_set := T.image multiples_of_6

theorem common_elements_count : (S_set ∩ T_set).card = 668 := by
  sorry

end NUMINAMATH_CALUDE_common_elements_count_l76_7671


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l76_7651

def i : ℂ := Complex.I

theorem complex_number_in_first_quadrant :
  let z : ℂ := i * (1 - i) * i
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l76_7651


namespace NUMINAMATH_CALUDE_math_city_intersections_l76_7681

/-- Represents a city with a given number of streets -/
structure City where
  num_streets : ℕ
  no_parallel_streets : Bool
  no_three_streets_intersect : Bool

/-- Calculates the maximum number of intersections in a city -/
def max_intersections (c : City) : ℕ :=
  if c.num_streets ≤ 1 then 0
  else (c.num_streets - 1) * (c.num_streets - 2) / 2

/-- Theorem: A city with 12 streets, no parallel streets, and no three streets intersecting at a point has 66 intersections -/
theorem math_city_intersections :
  ∀ (c : City), c.num_streets = 12 → c.no_parallel_streets = true → c.no_three_streets_intersect = true →
  max_intersections c = 66 :=
by sorry

end NUMINAMATH_CALUDE_math_city_intersections_l76_7681


namespace NUMINAMATH_CALUDE_quadratic_point_relation_l76_7690

/-- The quadratic function f(x) = x^2 - 4x - 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x - 3

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (-2, f (-2))

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (1, f 1)

/-- Point C on the graph of f -/
def C : ℝ × ℝ := (4, f 4)

theorem quadratic_point_relation :
  A.2 > C.2 ∧ C.2 > B.2 := by sorry

end NUMINAMATH_CALUDE_quadratic_point_relation_l76_7690


namespace NUMINAMATH_CALUDE_base7_subtraction_l76_7683

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a decimal number to base 7 --/
def decimalToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem base7_subtraction :
  let a := [4, 3, 2, 1]  -- 1234 in base 7
  let b := [2, 5, 6]     -- 652 in base 7
  let result := [2, 5, 2] -- 252 in base 7
  decimalToBase7 (base7ToDecimal a - base7ToDecimal b) = result := by
  sorry

end NUMINAMATH_CALUDE_base7_subtraction_l76_7683


namespace NUMINAMATH_CALUDE_integers_abs_lt_3_l76_7609

theorem integers_abs_lt_3 : 
  {n : ℤ | |n| < 3} = {-2, -1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_integers_abs_lt_3_l76_7609


namespace NUMINAMATH_CALUDE_ten_people_round_table_l76_7669

-- Define the number of people
def n : ℕ := 10

-- Define the function to calculate the number of distinct arrangements
def distinct_circular_arrangements (m : ℕ) : ℕ := Nat.factorial (m - 1)

-- Theorem statement
theorem ten_people_round_table : 
  distinct_circular_arrangements n = Nat.factorial 9 :=
sorry

end NUMINAMATH_CALUDE_ten_people_round_table_l76_7669


namespace NUMINAMATH_CALUDE_toms_restaurant_bill_l76_7615

/-- The total bill for a group at Tom's Restaurant -/
def total_bill (adults children meal_cost : ℕ) : ℕ :=
  (adults + children) * meal_cost

/-- Theorem: The bill for 2 adults and 5 children with $8 meals is $56 -/
theorem toms_restaurant_bill : total_bill 2 5 8 = 56 := by
  sorry

end NUMINAMATH_CALUDE_toms_restaurant_bill_l76_7615


namespace NUMINAMATH_CALUDE_claire_cakes_l76_7603

/-- The number of cakes Claire can make -/
def num_cakes (packages_per_cake : ℕ) (price_per_package : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent / price_per_package) / packages_per_cake

theorem claire_cakes : num_cakes 2 3 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_claire_cakes_l76_7603


namespace NUMINAMATH_CALUDE_periodic_function_theorem_l76_7604

/-- A function f is periodic with period b if f(x + b) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f (x + b) = f x

/-- The functional equation property for f -/
def HasFunctionalEquation (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (x + a) = 1/2 + Real.sqrt (f x - (f x)^2)

theorem periodic_function_theorem (f : ℝ → ℝ) (a : ℝ) (ha : a > 0) 
    (h : HasFunctionalEquation f a) : 
    IsPeriodic f (2 * a) := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_theorem_l76_7604


namespace NUMINAMATH_CALUDE_expression_equality_l76_7606

theorem expression_equality (y a : ℝ) (h1 : y > 0) 
  (h2 : (a * y) / 20 + (3 * y) / 10 = 0.5 * y) : a = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l76_7606


namespace NUMINAMATH_CALUDE_g_max_value_l76_7674

/-- The function g(x) = 4x - x^4 -/
def g (x : ℝ) : ℝ := 4 * x - x^4

/-- The maximum value of g(x) on the interval [0, 2] is 3 -/
theorem g_max_value : ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧ g c = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_max_value_l76_7674


namespace NUMINAMATH_CALUDE_mother_younger_by_two_l76_7645

/-- A family consisting of a father, mother, brother, sister, and Kaydence. -/
structure Family where
  total_age : ℕ
  father_age : ℕ
  brother_age : ℕ
  sister_age : ℕ
  kaydence_age : ℕ

/-- The age difference between the father and mother in the family. -/
def age_difference (f : Family) : ℕ :=
  f.father_age - (f.total_age - (f.father_age + f.brother_age + f.sister_age + f.kaydence_age))

/-- Theorem stating the age difference between the father and mother is 2 years. -/
theorem mother_younger_by_two (f : Family) 
    (h1 : f.total_age = 200)
    (h2 : f.father_age = 60)
    (h3 : f.brother_age = f.father_age / 2)
    (h4 : f.sister_age = 40)
    (h5 : f.kaydence_age = 12) :
    age_difference f = 2 := by
  sorry

end NUMINAMATH_CALUDE_mother_younger_by_two_l76_7645


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l76_7626

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (hS : S = 8) 
  (hsum : sum_first_two = 5) : 
  ∃ a : ℝ, (a = 8 * (1 - Real.sqrt (3/8)) ∨ a = 8 * (1 + Real.sqrt (3/8))) ∧ 
    (∃ r : ℝ, S = a / (1 - r) ∧ sum_first_two = a + a * r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l76_7626


namespace NUMINAMATH_CALUDE_vector_operation_result_l76_7666

def v1 : Fin 3 → ℝ := ![-3, 2, -1]
def v2 : Fin 3 → ℝ := ![1, 10, -2]
def scalar : ℝ := 2

theorem vector_operation_result :
  scalar • v1 + v2 = ![(-5 : ℝ), 14, -4] := by sorry

end NUMINAMATH_CALUDE_vector_operation_result_l76_7666


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l76_7627

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + b = 2) :
  ∃ m : ℝ, m = 6 ∧ ∀ x y : ℝ, x + y = 2 → 3^x + 3^y ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l76_7627


namespace NUMINAMATH_CALUDE_total_watermelon_seeds_l76_7644

theorem total_watermelon_seeds (bom gwi yeon : ℕ) : 
  bom = 300 →
  gwi = bom + 40 →
  yeon = 3 * gwi →
  bom + gwi + yeon = 1660 := by
sorry

end NUMINAMATH_CALUDE_total_watermelon_seeds_l76_7644


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l76_7682

theorem least_n_satisfying_inequality : 
  ∃ n : ℕ+, (∀ k : ℕ+, k < n → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ (1 : ℚ) / 15) ∧
             ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧
             n = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l76_7682


namespace NUMINAMATH_CALUDE_rectangle_perimeter_width_ratio_l76_7692

theorem rectangle_perimeter_width_ratio 
  (area : ℝ) (length : ℝ) (width : ℝ) (perimeter : ℝ) :
  area = 150 →
  length = 15 →
  area = length * width →
  perimeter = 2 * (length + width) →
  perimeter / width = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_width_ratio_l76_7692


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l76_7640

theorem quadratic_root_problem (m n k : ℝ) : 
  (m^2 + 2*m + k = 0) → 
  (n^2 + 2*n + k = 0) → 
  (1/m + 1/n = 6) → 
  (k = -1/3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l76_7640


namespace NUMINAMATH_CALUDE_initial_capacity_proof_l76_7656

/-- The initial capacity of a barrel in liters -/
def initial_capacity : ℝ := 220

/-- The percentage of contents remaining after the leak -/
def remaining_percentage : ℝ := 0.9

/-- The amount of liquid remaining in the barrel after the leak, in liters -/
def remaining_liquid : ℝ := 198

/-- Theorem stating that the initial capacity is correct given the conditions -/
theorem initial_capacity_proof : 
  initial_capacity * remaining_percentage = remaining_liquid :=
by sorry

end NUMINAMATH_CALUDE_initial_capacity_proof_l76_7656


namespace NUMINAMATH_CALUDE_randy_wipes_days_l76_7610

/-- Calculates the number of days Randy can use wipes given the number of packs and wipes per pack -/
def days_of_wipes (walks_per_day : ℕ) (paws : ℕ) (packs : ℕ) (wipes_per_pack : ℕ) : ℕ :=
  let wipes_per_day := walks_per_day * paws
  let total_wipes := packs * wipes_per_pack
  total_wipes / wipes_per_day

/-- Theorem stating that Randy needs wipes for 90 days -/
theorem randy_wipes_days :
  days_of_wipes 2 4 6 120 = 90 := by
  sorry

end NUMINAMATH_CALUDE_randy_wipes_days_l76_7610


namespace NUMINAMATH_CALUDE_candy_remaining_l76_7631

theorem candy_remaining (initial : ℝ) (talitha_took : ℝ) (solomon_took : ℝ) (maya_took : ℝ)
  (h1 : initial = 1012.5)
  (h2 : talitha_took = 283.7)
  (h3 : solomon_took = 398.2)
  (h4 : maya_took = 197.6) :
  initial - (talitha_took + solomon_took + maya_took) = 133 := by
  sorry

end NUMINAMATH_CALUDE_candy_remaining_l76_7631


namespace NUMINAMATH_CALUDE_running_track_dimensions_l76_7684

/-- Given two concentric circles forming a running track, prove the width and radii -/
theorem running_track_dimensions (r₁ r₂ : ℝ) : 
  (2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi) →  -- Difference in circumferences
  (2 * Real.pi * r₂ = 40 * Real.pi) →                     -- Circumference of inner circle
  (r₁ - r₂ = 10) ∧                                        -- Width of the track
  (r₂ = 20) ∧                                             -- Radius of inner circle
  (r₁ = 30) :=                                            -- Radius of outer circle
by
  sorry

end NUMINAMATH_CALUDE_running_track_dimensions_l76_7684


namespace NUMINAMATH_CALUDE_one_pattern_cannot_fold_to_pyramid_l76_7607

/-- Represents a pattern of identical squares -/
structure Pattern :=
  (squares : ℕ)
  (foldable : Bool)

/-- Represents a pyramid with a square base -/
structure Pyramid :=
  (base : ℕ)
  (sides : ℕ)

/-- Function to check if a pattern can be folded into a pyramid -/
def can_fold_to_pyramid (p : Pattern) (pyr : Pyramid) : Prop :=
  p.squares = pyr.base + pyr.sides ∧ p.foldable

/-- Theorem stating that exactly one pattern cannot be folded into a pyramid -/
theorem one_pattern_cannot_fold_to_pyramid 
  (A B C D : Pattern) 
  (pyr : Pyramid) 
  (h_pyr : pyr.base = 1 ∧ pyr.sides = 4) 
  (h_ABC : can_fold_to_pyramid A pyr ∧ can_fold_to_pyramid B pyr ∧ can_fold_to_pyramid C pyr) 
  (h_D : ¬can_fold_to_pyramid D pyr) : 
  ∃! p : Pattern, ¬can_fold_to_pyramid p pyr :=
sorry

end NUMINAMATH_CALUDE_one_pattern_cannot_fold_to_pyramid_l76_7607


namespace NUMINAMATH_CALUDE_square_area_9cm_l76_7623

/-- The area of a square with side length 9 cm is 81 cm² -/
theorem square_area_9cm (square : Real → Real) (h : ∀ x, square x = x * x) :
  square 9 = 81 :=
by sorry

end NUMINAMATH_CALUDE_square_area_9cm_l76_7623


namespace NUMINAMATH_CALUDE_equation_solution_l76_7677

theorem equation_solution : 
  ∃ x : ℚ, (3 + 2*x) / (1 + 2*x) - (5 + 2*x) / (7 + 2*x) = 1 - (4*x^2 - 2) / (7 + 16*x + 4*x^2) ∧ x = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l76_7677


namespace NUMINAMATH_CALUDE_inequality_solution_set_l76_7663

-- Define the inequality
def inequality (x : ℝ) : Prop := x^2 < -2*x + 15

-- Define the solution set
def solution_set : Set ℝ := {x | -5 < x ∧ x < 3}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l76_7663
