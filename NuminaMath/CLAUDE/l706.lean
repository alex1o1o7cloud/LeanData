import Mathlib

namespace NUMINAMATH_CALUDE_tetrahedron_face_areas_sum_squares_tetrahedron_face_areas_volume_inequality_l706_70686

-- Define the tetrahedron structure
structure Tetrahedron where
  V : ℝ  -- Volume
  S_A : ℝ  -- Face area opposite to vertex A
  S_B : ℝ  -- Face area opposite to vertex B
  S_C : ℝ  -- Face area opposite to vertex C
  S_D : ℝ  -- Face area opposite to vertex D
  a : ℝ   -- Length of edge BC
  a' : ℝ  -- Length of edge DA
  b : ℝ   -- Length of edge CA
  b' : ℝ  -- Length of edge DB
  c : ℝ   -- Length of edge AB
  c' : ℝ  -- Length of edge DC
  α : ℝ   -- Angle between opposite edges BC and DA
  β : ℝ   -- Angle between opposite edges CA and DB
  γ : ℝ   -- Angle between opposite edges AB and DC

-- Theorem statements
theorem tetrahedron_face_areas_sum_squares (t : Tetrahedron) :
  t.S_A^2 + t.S_B^2 + t.S_C^2 + t.S_D^2 = 
    1/4 * ((t.a * t.a' * Real.sin t.α)^2 + 
           (t.b * t.b' * Real.sin t.β)^2 + 
           (t.c * t.c' * Real.sin t.γ)^2) :=
  sorry

theorem tetrahedron_face_areas_volume_inequality (t : Tetrahedron) :
  t.S_A^2 + t.S_B^2 + t.S_C^2 + t.S_D^2 ≥ 9 * (3 * t.V^4)^(1/3) :=
  sorry

end NUMINAMATH_CALUDE_tetrahedron_face_areas_sum_squares_tetrahedron_face_areas_volume_inequality_l706_70686


namespace NUMINAMATH_CALUDE_AMC9_paths_l706_70683

-- Define the grid structure
structure Grid :=
  (has_A : Bool)
  (has_M_left : Bool)
  (has_M_right : Bool)
  (C_count_left : Nat)
  (C_count_right : Nat)
  (nine_count_per_C : Nat)

-- Define the path counting function
def count_paths (g : Grid) : Nat :=
  let left_paths := if g.has_M_left then g.C_count_left * g.nine_count_per_C else 0
  let right_paths := if g.has_M_right then g.C_count_right * g.nine_count_per_C else 0
  left_paths + right_paths

-- Theorem statement
theorem AMC9_paths (g : Grid) 
  (h1 : g.has_A)
  (h2 : g.has_M_left)
  (h3 : g.has_M_right)
  (h4 : g.C_count_left = 4)
  (h5 : g.C_count_right = 2)
  (h6 : g.nine_count_per_C = 2) :
  count_paths g = 24 := by
  sorry


end NUMINAMATH_CALUDE_AMC9_paths_l706_70683


namespace NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_l706_70674

theorem tan_alpha_two_implies_fraction (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_l706_70674


namespace NUMINAMATH_CALUDE_max_k_for_quadratic_root_difference_l706_70685

theorem max_k_for_quadratic_root_difference (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   x^2 + k*x + 10 = 0 ∧ 
   y^2 + k*y + 10 = 0 ∧ 
   |x - y| = Real.sqrt 81) →
  k ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_k_for_quadratic_root_difference_l706_70685


namespace NUMINAMATH_CALUDE_profit_maximizing_price_l706_70635

/-- Represents the profit maximization problem for a product -/
structure ProfitProblem where
  initialPurchasePrice : ℝ
  initialSellingPrice : ℝ
  initialQuantity : ℝ
  priceElasticity : ℝ

/-- Calculates the profit for a given price increase -/
def profit (problem : ProfitProblem) (priceIncrease : ℝ) : ℝ :=
  let newPrice := problem.initialSellingPrice + priceIncrease
  let newQuantity := problem.initialQuantity - problem.priceElasticity * priceIncrease
  (newPrice - problem.initialPurchasePrice) * newQuantity

/-- Theorem stating that the profit-maximizing price is 95 yuan -/
theorem profit_maximizing_price (problem : ProfitProblem) 
  (h1 : problem.initialPurchasePrice = 80)
  (h2 : problem.initialSellingPrice = 90)
  (h3 : problem.initialQuantity = 400)
  (h4 : problem.priceElasticity = 20) :
  ∃ (maxProfit : ℝ), ∀ (price : ℝ), 
    profit problem (price - problem.initialSellingPrice) ≤ maxProfit ∧
    profit problem (95 - problem.initialSellingPrice) = maxProfit :=
sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_l706_70635


namespace NUMINAMATH_CALUDE_triangle_properties_l706_70639

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin t.A * Real.sin t.B + (Real.sin t.C)^2 = (Real.sin t.A)^2 + (Real.sin t.B)^2)
  (h2 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0)
  (h3 : t.A + t.B + t.C = π)
  (h4 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0)
  (h5 : t.a / Real.sin t.A = t.b / Real.sin t.B)
  (h6 : t.b / Real.sin t.B = t.c / Real.sin t.C) :
  -- Part 1: A, C, B form an arithmetic sequence
  ∃ d : Real, t.B = t.C + d ∧ t.C = t.A + d ∧
  -- Part 2: If c = 2, the maximum area is √3
  (t.c = 2 → ∀ (s : Real), s = 1/2 * t.a * t.b * Real.sin t.C → s ≤ Real.sqrt 3) :=
sorry


end NUMINAMATH_CALUDE_triangle_properties_l706_70639


namespace NUMINAMATH_CALUDE_area_generally_irrational_l706_70615

/-- A regular polygon with n sides and rational side length -/
structure RegularPolygon where
  n : ℕ
  sideLength : ℚ

/-- The area of a regular polygon -/
noncomputable def area (p : RegularPolygon) : ℝ :=
  (p.n : ℝ) * p.sideLength^2 / (4 * Real.tan (Real.pi / p.n))

/-- Theorem: The area of a regular polygon with rational side length is generally irrational -/
theorem area_generally_irrational (p : RegularPolygon) : 
  ∃ (n : ℕ) (s : ℚ), Irrational (area { n := n, sideLength := s }) :=
sorry

end NUMINAMATH_CALUDE_area_generally_irrational_l706_70615


namespace NUMINAMATH_CALUDE_least_number_with_remainder_4_l706_70645

def is_valid_divisor (n : ℕ) : Prop := n > 0 ∧ 252 % n = 0

theorem least_number_with_remainder_4 : 
  (∀ x : ℕ, is_valid_divisor x → 256 % x = 4) ∧ 
  (∀ n : ℕ, n < 256 → ∃ y : ℕ, is_valid_divisor y ∧ n % y ≠ 4) :=
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_4_l706_70645


namespace NUMINAMATH_CALUDE_christina_total_driving_time_l706_70660

-- Define the total journey distance
def total_distance : ℝ := 210

-- Define the speed limits for each segment
def speed_limit_1 : ℝ := 30
def speed_limit_2 : ℝ := 40
def speed_limit_3 : ℝ := 50
def speed_limit_4 : ℝ := 60

-- Define the distances covered in the second and third segments
def distance_2 : ℝ := 120
def distance_3 : ℝ := 50

-- Define the time spent in the second and third segments
def time_2 : ℝ := 3
def time_3 : ℝ := 1

-- Define Christina's driving time function
def christina_driving_time : ℝ := by sorry

-- Theorem statement
theorem christina_total_driving_time :
  christina_driving_time = 100 / 60 := by sorry

end NUMINAMATH_CALUDE_christina_total_driving_time_l706_70660


namespace NUMINAMATH_CALUDE_francine_work_schedule_l706_70698

/-- The number of days Francine does not go to work every week -/
def days_not_working : ℕ :=
  7 - (2240 / (4 * 140))

theorem francine_work_schedule :
  days_not_working = 3 :=
sorry

end NUMINAMATH_CALUDE_francine_work_schedule_l706_70698


namespace NUMINAMATH_CALUDE_exactly_five_ladybugs_l706_70671

/-- Represents a ladybug with a specific number of spots -/
inductive Ladybug
  | sixSpots
  | fourSpots

/-- Represents a statement made by a ladybug -/
inductive Statement
  | allSame
  | totalThirty
  | totalTwentySix

/-- The meadow containing ladybugs -/
structure Meadow where
  ladybugs : List Ladybug

/-- Evaluates whether a statement is true for a given meadow -/
def isStatementTrue (m : Meadow) (s : Statement) : Bool :=
  match s with
  | Statement.allSame => sorry
  | Statement.totalThirty => sorry
  | Statement.totalTwentySix => sorry

/-- Counts the number of true statements in a list of statements for a given meadow -/
def countTrueStatements (m : Meadow) (statements : List Statement) : Nat :=
  statements.filter (isStatementTrue m) |>.length

/-- Theorem stating that there are exactly 5 ladybugs in the meadow -/
theorem exactly_five_ladybugs :
  ∃ (m : Meadow),
    m.ladybugs.length = 5 ∧
    (∀ l : Ladybug, l ∈ m.ladybugs → (l = Ladybug.sixSpots ∨ l = Ladybug.fourSpots)) ∧
    countTrueStatements m [Statement.allSame, Statement.totalThirty, Statement.totalTwentySix] = 1 :=
  sorry

end NUMINAMATH_CALUDE_exactly_five_ladybugs_l706_70671


namespace NUMINAMATH_CALUDE_solve_lemonade_problem_l706_70669

def lemonade_problem (price_per_cup : ℝ) (cups_sold : ℕ) (cost_lemons : ℝ) (cost_sugar : ℝ) (total_profit : ℝ) : Prop :=
  let total_revenue := price_per_cup * (cups_sold : ℝ)
  let known_expenses := cost_lemons + cost_sugar
  let cost_cups := total_revenue - known_expenses - total_profit
  cost_cups = 3

theorem solve_lemonade_problem :
  lemonade_problem 4 21 10 5 66 := by
  sorry

end NUMINAMATH_CALUDE_solve_lemonade_problem_l706_70669


namespace NUMINAMATH_CALUDE_seojun_pizza_problem_l706_70693

/-- Seojun's pizza problem -/
theorem seojun_pizza_problem (initial_pizza : ℚ) : 
  initial_pizza - 7/3 = 3/2 →
  initial_pizza + 7/3 = 37/6 := by
  sorry

end NUMINAMATH_CALUDE_seojun_pizza_problem_l706_70693


namespace NUMINAMATH_CALUDE_bobik_distance_l706_70681

/-- The problem of Seryozha, Valera, and Bobik's movement --/
theorem bobik_distance (distance : ℝ) (speed_seryozha speed_valera speed_bobik : ℝ) :
  distance = 21 →
  speed_seryozha = 4 →
  speed_valera = 3 →
  speed_bobik = 11 →
  speed_bobik * (distance / (speed_seryozha + speed_valera)) = 33 :=
by sorry

end NUMINAMATH_CALUDE_bobik_distance_l706_70681


namespace NUMINAMATH_CALUDE_circle_area_circumference_ratio_l706_70654

theorem circle_area_circumference_ratio (r₁ r₂ : ℝ) (h : π * r₁^2 / (π * r₂^2) = 49 / 64) :
  (2 * π * r₁) / (2 * π * r₂) = 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_circle_area_circumference_ratio_l706_70654


namespace NUMINAMATH_CALUDE_factorization_1_l706_70642

theorem factorization_1 (m n : ℝ) :
  3 * m^2 * n - 12 * m * n + 12 * n = 3 * n * (m - 2)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_1_l706_70642


namespace NUMINAMATH_CALUDE_mario_expected_doors_l706_70667

/-- The expected number of doors Mario will pass before reaching Bowser's level -/
def expected_doors (d r : ℕ) : ℚ :=
  (d * (d^r - 1)) / (d - 1)

/-- Theorem stating the expected number of doors Mario will pass -/
theorem mario_expected_doors (d r : ℕ) (hd : d > 1) (hr : r > 0) :
  let E := expected_doors d r
  ∀ k : ℕ, k ≤ r → 
    (∃ Ek : ℚ, Ek = E ∧ 
      Ek = 1 + (d - 1) / d * E + 1 / d * expected_doors d (r - k)) :=
by sorry

end NUMINAMATH_CALUDE_mario_expected_doors_l706_70667


namespace NUMINAMATH_CALUDE_incorrect_expression_l706_70647

theorem incorrect_expression (x y : ℚ) (h : x / y = 5 / 6) :
  ¬ ((x - y) / y = -1 / 6) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l706_70647


namespace NUMINAMATH_CALUDE_apple_sale_total_l706_70695

theorem apple_sale_total (red_apples : ℕ) (ratio_red : ℕ) (ratio_green : ℕ) : 
  red_apples = 32 → 
  ratio_red = 8 → 
  ratio_green = 3 → 
  red_apples + (red_apples * ratio_green / ratio_red) = 44 := by
sorry

end NUMINAMATH_CALUDE_apple_sale_total_l706_70695


namespace NUMINAMATH_CALUDE_intersection_nonempty_intersection_equals_A_l706_70666

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x : ℝ | x < -1 ∨ x > 5}

-- Theorem 1
theorem intersection_nonempty (a : ℝ) : 
  (A a ∩ B).Nonempty ↔ a < -1 ∨ a > 2 := by sorry

-- Theorem 2
theorem intersection_equals_A (a : ℝ) : 
  A a ∩ B = A a ↔ a < -4 ∨ a > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_intersection_equals_A_l706_70666


namespace NUMINAMATH_CALUDE_apple_purchase_change_l706_70617

/-- Calculates the change to be returned after a purchase. -/
theorem apple_purchase_change (apple_weight : ℝ) (price_per_kg : ℝ) (money_given : ℝ) :
  apple_weight = 6 →
  price_per_kg = 2.2 →
  money_given = 50 →
  money_given - apple_weight * price_per_kg = 36.8 := by
  sorry


end NUMINAMATH_CALUDE_apple_purchase_change_l706_70617


namespace NUMINAMATH_CALUDE_evaluate_expression_l706_70602

theorem evaluate_expression : -25 - 5 * (8 / 4) = -35 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l706_70602


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l706_70629

theorem sqrt_sum_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l706_70629


namespace NUMINAMATH_CALUDE_cone_base_circumference_l706_70684

/-- The circumference of the base of a right circular cone with volume 18π cubic centimeters and height 6 cm is 6π cm. -/
theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = 18 * Real.pi ∧ h = 6 ∧ V = (1/3) * Real.pi * r^2 * h →
  2 * Real.pi * r = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l706_70684


namespace NUMINAMATH_CALUDE_tree_planting_event_l706_70676

theorem tree_planting_event (boys girls : ℕ) : 
  girls - boys = 400 →
  boys = 600 →
  girls > boys →
  (60 : ℚ) / 100 * (boys + girls) = 960 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_event_l706_70676


namespace NUMINAMATH_CALUDE_rectangle_perimeter_from_squares_l706_70656

theorem rectangle_perimeter_from_squares (side_length : ℝ) : 
  side_length = 3 → 
  ∃ (perimeter₁ perimeter₂ : ℝ), 
    (perimeter₁ = 24 ∧ perimeter₂ = 30) ∧ 
    (∀ (p : ℝ), p ≠ perimeter₁ ∧ p ≠ perimeter₂ → 
      ¬∃ (length width : ℝ), 
        (length * width = 4 * side_length^2) ∧ 
        (2 * (length + width) = p)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_from_squares_l706_70656


namespace NUMINAMATH_CALUDE_max_correct_answers_l706_70677

/-- Represents the scoring system and results of a math contest. -/
structure MathContest where
  total_questions : ℕ
  correct_points : ℤ
  blank_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Theorem stating the maximum number of correct answers for the given contest conditions. -/
theorem max_correct_answers (contest : MathContest)
  (h1 : contest.total_questions = 60)
  (h2 : contest.correct_points = 5)
  (h3 : contest.blank_points = 0)
  (h4 : contest.incorrect_points = -2)
  (h5 : contest.total_score = 139) :
  ∃ (max_correct : ℕ), max_correct = 37 ∧
  ∀ (correct : ℕ), correct ≤ contest.total_questions →
    (∃ (blank incorrect : ℕ),
      correct + blank + incorrect = contest.total_questions ∧
      contest.correct_points * correct + contest.blank_points * blank + contest.incorrect_points * incorrect = contest.total_score) →
    correct ≤ max_correct :=
sorry

end NUMINAMATH_CALUDE_max_correct_answers_l706_70677


namespace NUMINAMATH_CALUDE_tangent_points_constant_sum_l706_70636

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by x^2 = 4y -/
def Parabola := {p : Point | p.x^2 = 4 * p.y}

/-- Checks if a line through two points is tangent to the parabola -/
def isTangent (p1 p2 : Point) : Prop :=
  p2 ∈ Parabola ∧ (∃ k : ℝ, p1.y - p2.y = k * (p1.x - p2.x) ∧ k = p2.x / 2)

theorem tangent_points_constant_sum (a : ℝ) :
  ∀ A B : Point,
  isTangent (Point.mk a (-2)) A ∧
  isTangent (Point.mk a (-2)) B ∧
  A ≠ B →
  A.x * B.x + A.y * B.y = -4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_points_constant_sum_l706_70636


namespace NUMINAMATH_CALUDE_particular_number_problem_l706_70692

theorem particular_number_problem (x : ℝ) :
  4 * (x - 220) = 320 → (5 * x) / 3 = 500 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_problem_l706_70692


namespace NUMINAMATH_CALUDE_digit_sum_theorem_l706_70616

/-- The digit sum of a natural number in base 4038 -/
def digitSum4038 (n : ℕ) : ℕ :=
  sorry

/-- A sequence of distinct positive integers -/
def IsValidSequence (s : ℕ → ℕ) : Prop :=
  (∀ i j, i ≠ j → s i ≠ s j) ∧ (∀ n, s n > 0)

/-- The property that infinitely many terms in the sequence have digit sums not divisible by 2019 -/
def InfinitelyManyNotDivisible (s : ℕ → ℕ) : Prop :=
  ∀ N, ∃ n > N, ¬ 2019 ∣ digitSum4038 (s n)

/-- The main theorem -/
theorem digit_sum_theorem (a : ℝ) :
  (a ≥ 1) →
  (∀ s : ℕ → ℕ, IsValidSequence s → (∀ n, (s n : ℝ) ≤ a * n) → InfinitelyManyNotDivisible s) ↔
  (1 ≤ a ∧ a < 2019) :=
sorry

end NUMINAMATH_CALUDE_digit_sum_theorem_l706_70616


namespace NUMINAMATH_CALUDE_find_number_l706_70689

theorem find_number : ∃ N : ℚ, (5/6 * N) - (5/16 * N) = 250 ∧ N = 480 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l706_70689


namespace NUMINAMATH_CALUDE_remainder_452867_div_9_l706_70607

theorem remainder_452867_div_9 : 452867 % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_452867_div_9_l706_70607


namespace NUMINAMATH_CALUDE_average_and_product_problem_l706_70651

theorem average_and_product_problem (x y : ℝ) : 
  (10 + 25 + x + y) / 4 = 20 →
  x * y = 156 →
  ((x = 12 ∧ y = 33) ∨ (x = 33 ∧ y = 12)) :=
by sorry

end NUMINAMATH_CALUDE_average_and_product_problem_l706_70651


namespace NUMINAMATH_CALUDE_garbage_classification_test_l706_70662

theorem garbage_classification_test (p_idea : ℝ) (p_no_idea : ℝ) (p_B : ℝ) :
  p_idea = 2/3 →
  p_no_idea = 1/4 →
  p_B = 0.6 →
  let E_A := (3/4 * p_idea + 1/4 * p_no_idea) * 2
  let E_B := p_B * 2
  E_B > E_A :=
by sorry

end NUMINAMATH_CALUDE_garbage_classification_test_l706_70662


namespace NUMINAMATH_CALUDE_triangle_inequality_l706_70614

-- Define a structure for a triangle
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z
  triangle_ineq : x + y > z ∧ y + z > x ∧ z + x > y

-- State the theorem
theorem triangle_inequality (t : Triangle) :
  |t.x^2 * (t.y - t.z) + t.y^2 * (t.z - t.x) + t.z^2 * (t.x - t.y)| < t.x * t.y * t.z :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l706_70614


namespace NUMINAMATH_CALUDE_vanya_cookies_l706_70650

theorem vanya_cookies (total : ℚ) (vanya_before : ℚ) (shared : ℚ) :
  total > 0 ∧ vanya_before ≥ 0 ∧ shared ≥ 0 ∧
  total = vanya_before + shared ∧
  vanya_before + shared / 2 = 5 * (shared / 2) →
  vanya_before / total = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_vanya_cookies_l706_70650


namespace NUMINAMATH_CALUDE_scissors_count_l706_70638

/-- The total number of scissors after adding more to an initial amount -/
def total_scissors (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: Given 54 initial scissors and 22 added scissors, the total is 76 -/
theorem scissors_count : total_scissors 54 22 = 76 := by
  sorry

end NUMINAMATH_CALUDE_scissors_count_l706_70638


namespace NUMINAMATH_CALUDE_expected_value_of_x_l706_70628

/-- Represents the contingency table data -/
structure ContingencyTable where
  boys_a : ℕ
  boys_b : ℕ
  girls_a : ℕ
  girls_b : ℕ

/-- Represents the distribution of X -/
structure Distribution where
  p0 : ℚ
  p1 : ℚ
  p2 : ℚ
  p3 : ℚ

/-- Main theorem statement -/
theorem expected_value_of_x (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ) 
  (table : ContingencyTable) (dist : Distribution) : 
  total_students = 450 →
  total_boys = 250 →
  total_girls = 200 →
  table.boys_a + table.boys_b = total_boys →
  table.girls_a + table.girls_b = total_girls →
  table.boys_b = 150 →
  table.girls_a = 50 →
  dist.p0 = 1/6 →
  dist.p1 = 1/2 →
  dist.p2 = 3/10 →
  dist.p3 = 1/30 →
  0 * dist.p0 + 1 * dist.p1 + 2 * dist.p2 + 3 * dist.p3 = 6/5 := by
  sorry


end NUMINAMATH_CALUDE_expected_value_of_x_l706_70628


namespace NUMINAMATH_CALUDE_function_value_at_two_l706_70690

/-- Given a function f: ℝ → ℝ satisfying f(x) + 2f(1/x) = 3x for all x ∈ ℝ, prove that f(2) = -3/2 -/
theorem function_value_at_two (f : ℝ → ℝ) (h : ∀ x : ℝ, f x + 2 * f (1/x) = 3 * x) : f 2 = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l706_70690


namespace NUMINAMATH_CALUDE_jared_tom_age_ratio_l706_70622

theorem jared_tom_age_ratio : 
  ∀ (tom_future_age jared_current_age : ℕ),
    tom_future_age = 30 →
    jared_current_age = 48 →
    ∃ (jared_past_age tom_past_age : ℕ),
      jared_past_age = jared_current_age - 2 ∧
      tom_past_age = tom_future_age - 7 ∧
      jared_past_age = 2 * tom_past_age :=
by sorry

end NUMINAMATH_CALUDE_jared_tom_age_ratio_l706_70622


namespace NUMINAMATH_CALUDE_alex_born_in_1989_l706_70697

/-- The year when the first Math Kangaroo test was held -/
def first_math_kangaroo_year : ℕ := 1991

/-- The number of the Math Kangaroo test Alex participated in -/
def alex_participation_number : ℕ := 9

/-- Alex's age when he participated in the Math Kangaroo test -/
def alex_age_at_participation : ℕ := 10

/-- Calculate the year of Alex's birth -/
def alex_birth_year : ℕ := first_math_kangaroo_year + alex_participation_number - 1 - alex_age_at_participation

theorem alex_born_in_1989 : alex_birth_year = 1989 := by
  sorry

end NUMINAMATH_CALUDE_alex_born_in_1989_l706_70697


namespace NUMINAMATH_CALUDE_baseball_game_earnings_l706_70675

theorem baseball_game_earnings (total : ℝ) (difference : ℝ) (wednesday : ℝ) (sunday : ℝ)
  (h1 : total = 4994.50)
  (h2 : difference = 1330.50)
  (h3 : wednesday + sunday = total)
  (h4 : wednesday = sunday - difference) :
  wednesday = 1832 := by
sorry

end NUMINAMATH_CALUDE_baseball_game_earnings_l706_70675


namespace NUMINAMATH_CALUDE_missing_number_proof_l706_70653

theorem missing_number_proof (x : ℝ) (y : ℝ) : 
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (128 + 255 + y + 1023 + x) / 5 = 398.2 →
  y = 511 := by
sorry

end NUMINAMATH_CALUDE_missing_number_proof_l706_70653


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l706_70694

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ), 
    (∀ x : ℚ, x ≠ 7 ∧ x ≠ -2 → 
      (5 * x - 3) / (x^2 - 5*x - 14) = C / (x - 7) + D / (x + 2)) ∧
    C = 32/9 ∧ D = 13/9 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l706_70694


namespace NUMINAMATH_CALUDE_field_resizing_problem_l706_70643

theorem field_resizing_problem : ∃ m : ℝ, 
  m > 0 ∧ (3 * m + 14) * (m + 1) = 240 ∧ abs (m - 6.3) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_field_resizing_problem_l706_70643


namespace NUMINAMATH_CALUDE_number_operation_result_l706_70673

theorem number_operation_result (x : ℝ) : x + 7 = 27 → ((x / 5) + 5) * 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_result_l706_70673


namespace NUMINAMATH_CALUDE_number_plus_eight_equals_500_l706_70687

theorem number_plus_eight_equals_500 (x : ℤ) : x + 8 = 500 → x = 492 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_eight_equals_500_l706_70687


namespace NUMINAMATH_CALUDE_curve_parameter_value_l706_70646

/-- Given a curve C with parametric equations x = 1 + 3t and y = at² + 2,
    where t is the parameter and a is a real number,
    prove that if the point (4,3) lies on C, then a = 1. -/
theorem curve_parameter_value (a : ℝ) :
  (∃ t : ℝ, 1 + 3 * t = 4 ∧ a * t^2 + 2 = 3) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_curve_parameter_value_l706_70646


namespace NUMINAMATH_CALUDE_translation_result_l706_70608

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally and vertically -/
def translate (p : Point2D) (dx dy : ℝ) : Point2D :=
  { x := p.x + dx, y := p.y + dy }

theorem translation_result :
  let p := Point2D.mk (-3) 2
  let p_translated := translate (translate p 2 0) 0 (-4)
  p_translated = Point2D.mk (-1) (-2) := by
  sorry


end NUMINAMATH_CALUDE_translation_result_l706_70608


namespace NUMINAMATH_CALUDE_difference_of_squares_701_697_l706_70624

theorem difference_of_squares_701_697 : 701^2 - 697^2 = 5592 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_701_697_l706_70624


namespace NUMINAMATH_CALUDE_martha_problems_l706_70655

theorem martha_problems (total : ℕ) (angela_unique : ℕ) : total = 20 → angela_unique = 9 → ∃ martha : ℕ,
  martha + (4 * martha - 2) + ((4 * martha - 2) / 2) + angela_unique = total ∧ martha = 2 := by
  sorry

end NUMINAMATH_CALUDE_martha_problems_l706_70655


namespace NUMINAMATH_CALUDE_sarah_marriage_prediction_l706_70657

/-- Predicts the marriage age based on a person's name length and current age -/
def predict_marriage_age (name_length : ℕ) (current_age : ℕ) : ℕ :=
  name_length + 2 * current_age

/-- Theorem stating that for Sarah, the predicted marriage age is 23 -/
theorem sarah_marriage_prediction :
  let sarah_name_length : ℕ := 5
  let sarah_current_age : ℕ := 9
  predict_marriage_age sarah_name_length sarah_current_age = 23 := by
  sorry

end NUMINAMATH_CALUDE_sarah_marriage_prediction_l706_70657


namespace NUMINAMATH_CALUDE_binary_11011_equals_27_l706_70652

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 11011₂ -/
def binary_11011 : List Bool := [true, true, false, true, true]

theorem binary_11011_equals_27 :
  binary_to_decimal binary_11011 = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_11011_equals_27_l706_70652


namespace NUMINAMATH_CALUDE_fourth_month_sales_l706_70688

def sales_1 : ℕ := 2500
def sales_2 : ℕ := 6500
def sales_3 : ℕ := 9855
def sales_5 : ℕ := 7000
def sales_6 : ℕ := 11915
def average_sale : ℕ := 7500
def num_months : ℕ := 6

theorem fourth_month_sales (sales_4 : ℕ) : 
  (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sale → 
  sales_4 = 14230 := by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sales_l706_70688


namespace NUMINAMATH_CALUDE_marble_probability_l706_70668

theorem marble_probability (total : ℕ) (p_white p_green p_yellow p_orange : ℚ) :
  total = 500 →
  p_white = 1/4 →
  p_green = 1/5 →
  p_yellow = 1/6 →
  p_orange = 1/10 →
  let p_red_blue := 1 - (p_white + p_green + p_yellow + p_orange)
  p_red_blue = 71/250 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l706_70668


namespace NUMINAMATH_CALUDE_tim_works_six_days_l706_70621

/-- Represents Tim's work schedule and earnings --/
structure TimsWork where
  tasks_per_day : ℕ
  pay_per_task : ℚ
  weekly_earnings : ℚ

/-- Calculates the number of days Tim works per week --/
def days_worked (w : TimsWork) : ℚ :=
  w.weekly_earnings / (w.tasks_per_day * w.pay_per_task)

/-- Theorem stating that Tim works 6 days a week --/
theorem tim_works_six_days (w : TimsWork) 
  (h1 : w.tasks_per_day = 100)
  (h2 : w.pay_per_task = 6/5) -- $1.2 represented as a fraction
  (h3 : w.weekly_earnings = 720) :
  days_worked w = 6 := by
  sorry

#eval days_worked { tasks_per_day := 100, pay_per_task := 6/5, weekly_earnings := 720 }

end NUMINAMATH_CALUDE_tim_works_six_days_l706_70621


namespace NUMINAMATH_CALUDE_john_half_decks_l706_70601

/-- The number of cards in a full deck -/
def full_deck : ℕ := 52

/-- The number of full decks John has -/
def num_full_decks : ℕ := 3

/-- The number of cards John threw away -/
def discarded_cards : ℕ := 34

/-- The number of cards John has after discarding -/
def remaining_cards : ℕ := 200

/-- Calculates the number of half-full decks John found -/
def num_half_decks : ℕ :=
  (remaining_cards + discarded_cards - num_full_decks * full_deck) / (full_deck / 2)

theorem john_half_decks :
  num_half_decks = 3 := by sorry

end NUMINAMATH_CALUDE_john_half_decks_l706_70601


namespace NUMINAMATH_CALUDE_ammonium_nitrate_reaction_l706_70680

-- Define the chemical species as types
structure ChemicalSpecies where
  name : String
  formula : String

-- Define the reaction
def reaction : List (ℕ × ChemicalSpecies) → List (ℕ × ChemicalSpecies) → Prop :=
  sorry

-- Define the chemical species involved
def nh4no3 : ChemicalSpecies := ⟨"Ammonium nitrate", "NH4NO3"⟩
def naoh : ChemicalSpecies := ⟨"Sodium hydroxide", "NaOH"⟩
def nano3 : ChemicalSpecies := ⟨"Sodium nitrate", "NaNO3"⟩
def nh3 : ChemicalSpecies := ⟨"Ammonia", "NH3"⟩
def h2o : ChemicalSpecies := ⟨"Water", "H2O"⟩

-- State the theorem
theorem ammonium_nitrate_reaction 
  (balanced_equation : reaction [(1, nh4no3), (1, naoh)] [(1, nano3), (1, nh3), (1, h2o)])
  (naoh_reacted : ℕ) (nano3_formed : ℕ) (nh3_formed : ℕ)
  (h1 : naoh_reacted = 3)
  (h2 : nano3_formed = 3)
  (h3 : nh3_formed = 3) :
  ∃ (nh4no3_required : ℕ) (h2o_formed : ℕ),
    nh4no3_required = 3 ∧ h2o_formed = 3 :=
  sorry

end NUMINAMATH_CALUDE_ammonium_nitrate_reaction_l706_70680


namespace NUMINAMATH_CALUDE_tan_negative_405_degrees_l706_70632

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_405_degrees_l706_70632


namespace NUMINAMATH_CALUDE_tan_60_minus_sin_60_l706_70609

theorem tan_60_minus_sin_60 : Real.tan (π / 3) - Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_60_minus_sin_60_l706_70609


namespace NUMINAMATH_CALUDE_total_money_found_l706_70691

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10
def nickel_value : ℚ := 0.05
def penny_value : ℚ := 0.01

def num_quarters : ℕ := 10
def num_dimes : ℕ := 3
def num_nickels : ℕ := 3
def num_pennies : ℕ := 5

theorem total_money_found :
  (num_quarters : ℚ) * quarter_value +
  (num_dimes : ℚ) * dime_value +
  (num_nickels : ℚ) * nickel_value +
  (num_pennies : ℚ) * penny_value = 3 := by sorry

end NUMINAMATH_CALUDE_total_money_found_l706_70691


namespace NUMINAMATH_CALUDE_expression_simplification_l706_70630

theorem expression_simplification (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  ((x + y) * (x - y) - (x - y)^2 + 2 * y * (x - y)) / (4 * y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l706_70630


namespace NUMINAMATH_CALUDE_total_crayons_l706_70672

theorem total_crayons (orange_boxes : Nat) (orange_per_box : Nat)
                      (blue_boxes : Nat) (blue_per_box : Nat)
                      (red_boxes : Nat) (red_per_box : Nat) :
  orange_boxes = 6 →
  orange_per_box = 8 →
  blue_boxes = 7 →
  blue_per_box = 5 →
  red_boxes = 1 →
  red_per_box = 11 →
  orange_boxes * orange_per_box + blue_boxes * blue_per_box + red_boxes * red_per_box = 94 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l706_70672


namespace NUMINAMATH_CALUDE_simplify_expression_l706_70625

theorem simplify_expression (x y : ℚ) (hx : x = 3) (hy : y = 2) :
  12 * x^3 * y^4 / (9 * x^2 * y^3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l706_70625


namespace NUMINAMATH_CALUDE_travel_time_calculation_l706_70682

/-- Given a constant rate of travel where 1 mile takes 4 minutes,
    prove that the time required to travel 5 miles is 20 minutes. -/
theorem travel_time_calculation (rate : ℝ) (distance : ℝ) :
  rate = 1 / 4 → distance = 5 → rate * distance = 20 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l706_70682


namespace NUMINAMATH_CALUDE_system_solution_l706_70604

theorem system_solution :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (2*x - Real.sqrt (x*y) - 4*Real.sqrt (x/y) + 2 = 0 ∧
   2*x^2 + x^2*y^4 = 18*y^2) →
  ((x = 2 ∧ y = 2) ∨
   (x = (Real.sqrt (Real.sqrt 286))/4 ∧ y = Real.sqrt (Real.sqrt 286))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l706_70604


namespace NUMINAMATH_CALUDE_expression_equality_l706_70661

theorem expression_equality : 
  |1 - Real.sqrt 2| - 2 * Real.cos (45 * π / 180) + (1 / 2)⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l706_70661


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l706_70618

theorem diophantine_equation_solutions :
  ∀ x y : ℕ, 7^x - 3 * 2^y = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l706_70618


namespace NUMINAMATH_CALUDE_x_plus_y_equals_32_l706_70696

theorem x_plus_y_equals_32 (x y : ℝ) 
  (h1 : (4 : ℝ)^x = 16^(y+1)) 
  (h2 : (27 : ℝ)^y = 9^(x-6)) : 
  x + y = 32 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_32_l706_70696


namespace NUMINAMATH_CALUDE_even_function_property_l706_70605

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def MonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

theorem even_function_property (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc (-6) 6, HasDerivAt f (f x) x) →
  IsEven f →
  MonoDecreasing f (-6) 0 →
  f 4 - f 1 > 0 :=
sorry

end NUMINAMATH_CALUDE_even_function_property_l706_70605


namespace NUMINAMATH_CALUDE_candy_distribution_l706_70631

theorem candy_distribution (total : ℕ) (portions : ℕ) (increment : ℕ) (smallest : ℕ) : 
  total = 40 →
  portions = 4 →
  increment = 2 →
  (smallest + (smallest + increment) + (smallest + 2 * increment) + (smallest + 3 * increment) = total) →
  smallest = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l706_70631


namespace NUMINAMATH_CALUDE_boat_speed_difference_l706_70626

/-- Proves that the difference between boat speed and current speed in a channel is 1 km/h -/
theorem boat_speed_difference (V : ℝ) : ∃ (U : ℝ),
  (1 / (U - V) - 1 / (U + V) + 1 / (2 * V + 1) = 1) ∧ (U - V = 1) :=
by
  sorry

#check boat_speed_difference

end NUMINAMATH_CALUDE_boat_speed_difference_l706_70626


namespace NUMINAMATH_CALUDE_parallel_lines_min_value_l706_70619

theorem parallel_lines_min_value (m n : ℕ+) : 
  (∀ x y : ℝ, x + (n.val - 1) * y - 2 = 0 ↔ m.val * x + y + 3 = 0) →
  (∀ k : ℕ+, 2 * m.val + n.val ≤ k.val → k.val = 11) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_min_value_l706_70619


namespace NUMINAMATH_CALUDE_inequality_theorem_l706_70623

theorem inequality_theorem (n : ℕ) (hn : n > 0) :
  (∀ x : ℝ, x > 0 → x + (n^n : ℝ) / x^n ≥ n + 1) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → x + a / x^n ≥ n + 1) → a = n^n) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l706_70623


namespace NUMINAMATH_CALUDE_min_sequence_length_l706_70633

def S : Finset Nat := {1, 2, 3, 4}

def isValidPermutation (perm : List Nat) : Prop :=
  perm.length = 4 ∧ perm.toFinset = S ∧ perm.getLast? ≠ some 1

def containsAllValidPermutations (seq : List Nat) : Prop :=
  ∀ perm : List Nat, isValidPermutation perm →
    ∃ i₁ i₂ i₃ i₄ : Nat,
      i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧
      i₄ ≤ seq.length ∧
      seq.get? i₁ = some (perm.get! 0) ∧
      seq.get? i₂ = some (perm.get! 1) ∧
      seq.get? i₃ = some (perm.get! 2) ∧
      seq.get? i₄ = some (perm.get! 3)

theorem min_sequence_length :
  ∃ seq : List Nat, seq.length = 11 ∧ containsAllValidPermutations seq ∧
  ∀ seq' : List Nat, seq'.length < 11 → ¬containsAllValidPermutations seq' :=
sorry

end NUMINAMATH_CALUDE_min_sequence_length_l706_70633


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l706_70658

/-- The distance between the vertices of the hyperbola x^2/48 - y^2/16 = 1 is 8√3 -/
theorem hyperbola_vertices_distance :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ x^2 / 48 - y^2 / 16
  ∃ (a b : ℝ), a ≠ b ∧ f (a, 0) = 1 ∧ f (b, 0) = 1 ∧ |a - b| = 8 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l706_70658


namespace NUMINAMATH_CALUDE_total_straws_for_mats_l706_70648

/-- The number of mats to be made -/
def num_mats : ℕ := 10

/-- The number of red straws required for one mat -/
def red_straws_per_mat : ℕ := 20

/-- The number of orange straws required for one mat -/
def orange_straws_per_mat : ℕ := 30

/-- The number of green straws required for one mat -/
def green_straws_per_mat : ℕ := orange_straws_per_mat / 2

/-- The total number of straws required for one mat -/
def straws_per_mat : ℕ := red_straws_per_mat + orange_straws_per_mat + green_straws_per_mat

/-- Theorem stating the total number of straws needed for 10 mats -/
theorem total_straws_for_mats : num_mats * straws_per_mat = 650 := by
  sorry

end NUMINAMATH_CALUDE_total_straws_for_mats_l706_70648


namespace NUMINAMATH_CALUDE_ceiling_lights_difference_l706_70663

theorem ceiling_lights_difference (medium large small : ℕ) : 
  medium = 12 →
  large = 2 * medium →
  small + 2 * medium + 3 * large = 118 →
  small - medium = 10 := by
sorry

end NUMINAMATH_CALUDE_ceiling_lights_difference_l706_70663


namespace NUMINAMATH_CALUDE_candidate_votes_l706_70678

theorem candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_percent : ℚ) :
  total_votes = 560000 →
  invalid_percent = 15 / 100 →
  candidate_percent = 75 / 100 →
  ↑⌊(1 - invalid_percent) * candidate_percent * total_votes⌋ = 357000 := by
  sorry

end NUMINAMATH_CALUDE_candidate_votes_l706_70678


namespace NUMINAMATH_CALUDE_no_natural_solutions_l706_70606

theorem no_natural_solutions :
  (∀ x y z : ℕ, x^2 + y^2 + z^2 ≠ 2*x*y*z) ∧
  (∀ x y z u : ℕ, x^2 + y^2 + z^2 + u^2 ≠ 2*x*y*z*u) := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l706_70606


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l706_70644

theorem circle_radius_from_area :
  ∀ (r : ℝ), r > 0 → π * r^2 = 64 * π → r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l706_70644


namespace NUMINAMATH_CALUDE_solution_set_inequality_l706_70610

/-- The solution set of the inequality x(9-x) > 0 is the open interval (0,9) -/
theorem solution_set_inequality (x : ℝ) : x * (9 - x) > 0 ↔ x ∈ Set.Ioo 0 9 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l706_70610


namespace NUMINAMATH_CALUDE_inequality_proof_l706_70620

theorem inequality_proof (a b c : ℝ) (h1 : b > a) (h2 : a > 0) :
  a^2 < b^2 ∧ a*b < b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l706_70620


namespace NUMINAMATH_CALUDE_stratified_sample_for_model_a_l706_70603

/-- Calculates the number of items to be selected in a stratified sample -/
def stratified_sample_size (model_volume : ℕ) (total_volume : ℕ) (total_sample : ℕ) : ℕ :=
  (model_volume * total_sample) / total_volume

theorem stratified_sample_for_model_a 
  (volume_a volume_b volume_c total_sample : ℕ) 
  (h_positive : volume_a > 0 ∧ volume_b > 0 ∧ volume_c > 0 ∧ total_sample > 0) :
  stratified_sample_size volume_a (volume_a + volume_b + volume_c) total_sample = 
    (volume_a * total_sample) / (volume_a + volume_b + volume_c) :=
by
  sorry

#eval stratified_sample_size 1200 9200 46

end NUMINAMATH_CALUDE_stratified_sample_for_model_a_l706_70603


namespace NUMINAMATH_CALUDE_bottles_taken_home_l706_70664

def bottles_brought : ℕ := 50
def bottles_drunk : ℕ := 38

theorem bottles_taken_home : 
  bottles_brought - bottles_drunk = 12 := by sorry

end NUMINAMATH_CALUDE_bottles_taken_home_l706_70664


namespace NUMINAMATH_CALUDE_fraction_of_books_sold_l706_70600

theorem fraction_of_books_sold (total_revenue : ℕ) (remaining_books : ℕ) (price_per_book : ℕ) :
  total_revenue = 288 →
  remaining_books = 36 →
  price_per_book = 4 →
  (total_revenue / price_per_book : ℚ) / ((total_revenue / price_per_book) + remaining_books) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_books_sold_l706_70600


namespace NUMINAMATH_CALUDE_coffee_doughnut_problem_l706_70613

theorem coffee_doughnut_problem :
  ∀ (c d : ℕ),
    c + d = 7 →
    (90 * c + 60 * d) % 100 = 0 →
    c = 6 := by
  sorry

end NUMINAMATH_CALUDE_coffee_doughnut_problem_l706_70613


namespace NUMINAMATH_CALUDE_smallest_e_value_l706_70679

theorem smallest_e_value (a b c d e : ℤ) : 
  (∃ (x : ℝ), a * x^4 + b * x^3 + c * x^2 + d * x + e = 0) →
  (a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0) →
  (a * 4^4 + b * 4^3 + c * 4^2 + d * 4 + e = 0) →
  (a * 8^4 + b * 8^3 + c * 8^2 + d * 8 + e = 0) →
  (a * (-1/4)^4 + b * (-1/4)^3 + c * (-1/4)^2 + d * (-1/4) + e = 0) →
  e > 0 →
  e ≥ 96 := by
sorry

end NUMINAMATH_CALUDE_smallest_e_value_l706_70679


namespace NUMINAMATH_CALUDE_tetrahedron_surface_area_l706_70611

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- Represents the projection of a tetrahedron onto a plane -/
structure TetrahedronProjection where
  area : ℝ
  has_60_degree_angle : Bool

/-- 
Given a regular tetrahedron and its projection onto a plane parallel to the line segment 
connecting the midpoints of two opposite edges, prove that the surface area of the tetrahedron 
is 2x^2 √2/3, where x is the edge length of the tetrahedron.
-/
theorem tetrahedron_surface_area 
  (t : RegularTetrahedron) 
  (p : TetrahedronProjection) 
  (h : p.has_60_degree_angle = true) : 
  ℝ :=
by
  sorry

#check tetrahedron_surface_area

end NUMINAMATH_CALUDE_tetrahedron_surface_area_l706_70611


namespace NUMINAMATH_CALUDE_total_grain_calculation_l706_70649

/-- The amount of grain in kilograms transported from the first warehouse. -/
def transported : ℕ := 2500

/-- The amount of grain in kilograms in the second warehouse. -/
def second_warehouse : ℕ := 50200

/-- The total amount of grain in kilograms in both warehouses. -/
def total_grain : ℕ := second_warehouse + (second_warehouse + transported)

theorem total_grain_calculation :
  total_grain = 102900 :=
by sorry

end NUMINAMATH_CALUDE_total_grain_calculation_l706_70649


namespace NUMINAMATH_CALUDE_leo_statement_true_only_on_tuesday_l706_70659

-- Define the days of the week
inductive Day : Type
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

-- Define Leo's lying pattern
def lies_on_day (d : Day) : Prop :=
  match d with
  | Day.monday => True
  | Day.tuesday => True
  | Day.wednesday => True
  | _ => False

-- Define the 'yesterday' and 'tomorrow' functions
def yesterday (d : Day) : Day :=
  match d with
  | Day.monday => Day.sunday
  | Day.tuesday => Day.monday
  | Day.wednesday => Day.tuesday
  | Day.thursday => Day.wednesday
  | Day.friday => Day.thursday
  | Day.saturday => Day.friday
  | Day.sunday => Day.saturday

def tomorrow (d : Day) : Day :=
  match d with
  | Day.monday => Day.tuesday
  | Day.tuesday => Day.wednesday
  | Day.wednesday => Day.thursday
  | Day.thursday => Day.friday
  | Day.friday => Day.saturday
  | Day.saturday => Day.sunday
  | Day.sunday => Day.monday

-- Define Leo's statement
def leo_statement (d : Day) : Prop :=
  lies_on_day (yesterday d) ∧ lies_on_day (tomorrow d)

-- Theorem: Leo's statement is true only on Tuesday
theorem leo_statement_true_only_on_tuesday :
  ∀ (d : Day), leo_statement d ↔ d = Day.tuesday :=
by sorry

end NUMINAMATH_CALUDE_leo_statement_true_only_on_tuesday_l706_70659


namespace NUMINAMATH_CALUDE_sum_of_divisors_of_29_l706_70612

theorem sum_of_divisors_of_29 (h : Nat.Prime 29) : 
  (Finset.filter (· ∣ 29) (Finset.range 30)).sum id = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_of_29_l706_70612


namespace NUMINAMATH_CALUDE_greatest_value_b_l706_70641

theorem greatest_value_b (b : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + 3 < -x + 6 → x ≤ b) ↔ b = (3 + Real.sqrt 21) / 2 :=
sorry

end NUMINAMATH_CALUDE_greatest_value_b_l706_70641


namespace NUMINAMATH_CALUDE_product_sum_theorem_l706_70699

theorem product_sum_theorem (P Q : ℕ) : 
  P < 10 → Q < 10 → 39 * P * 10 + 39 * P * 3 + Q * 300 + Q * 3 = 32951 → P + Q = 15 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l706_70699


namespace NUMINAMATH_CALUDE_product_of_equal_sums_l706_70665

theorem product_of_equal_sums (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_equal_sums_l706_70665


namespace NUMINAMATH_CALUDE_replacement_stove_cost_l706_70634

/-- The cost of a replacement stove and wall repair, given specific conditions. -/
theorem replacement_stove_cost (stove_cost wall_cost : ℚ) : 
  wall_cost = (1 : ℚ) / 6 * stove_cost →
  stove_cost + wall_cost = 1400 →
  stove_cost = 1200 := by
sorry

end NUMINAMATH_CALUDE_replacement_stove_cost_l706_70634


namespace NUMINAMATH_CALUDE_equation_solution_l706_70637

theorem equation_solution :
  let f (x : ℝ) := (7 * x + 3) / (3 * x^2 + 7 * x - 6)
  let g (x : ℝ) := (3 * x) / (3 * x - 2)
  let sol₁ := (-1 + Real.sqrt 10) / 3
  let sol₂ := (-1 - Real.sqrt 10) / 3
  ∀ x : ℝ, x ≠ 2/3 →
    (f x = g x ↔ x = sol₁ ∨ x = sol₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l706_70637


namespace NUMINAMATH_CALUDE_triples_satisfying_equation_l706_70627

theorem triples_satisfying_equation : 
  ∀ (a b p : ℕ), 
    0 < a ∧ 0 < b ∧ 0 < p ∧ 
    Nat.Prime p ∧
    a^p - b^p = 2013 →
    ((a = 337 ∧ b = 334 ∧ p = 2) ∨ 
     (a = 97 ∧ b = 86 ∧ p = 2) ∨ 
     (a = 47 ∧ b = 14 ∧ p = 2)) :=
by sorry

end NUMINAMATH_CALUDE_triples_satisfying_equation_l706_70627


namespace NUMINAMATH_CALUDE_hotel_room_charge_comparison_l706_70670

theorem hotel_room_charge_comparison (P R G : ℝ) 
  (h1 : P = R - 0.2 * R) 
  (h2 : P = G - 0.1 * G) : 
  R = G * (1 + 0.125) := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_charge_comparison_l706_70670


namespace NUMINAMATH_CALUDE_similar_triangles_perimeter_l706_70640

/-- Given two similar right triangles, where the smaller triangle has legs of 5 and 12,
    and the larger triangle has a hypotenuse of 39, prove that the perimeter of the larger triangle is 90. -/
theorem similar_triangles_perimeter (small_leg1 small_leg2 large_hypotenuse : ℝ)
    (h1 : small_leg1 = 5)
    (h2 : small_leg2 = 12)
    (h3 : large_hypotenuse = 39)
    (h4 : small_leg1^2 + small_leg2^2 = (small_leg1^2 + small_leg2^2).sqrt^2) -- Pythagorean theorem for smaller triangle
    (h5 : ∃ k : ℝ, k * (small_leg1^2 + small_leg2^2).sqrt = large_hypotenuse) -- Similarity condition
    : ∃ large_leg1 large_leg2 : ℝ,
      large_leg1^2 + large_leg2^2 = large_hypotenuse^2 ∧ -- Pythagorean theorem for larger triangle
      large_leg1 + large_leg2 + large_hypotenuse = 90 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_perimeter_l706_70640
