import Mathlib

namespace NUMINAMATH_CALUDE_sum_b_formula_l2941_294147

def a (n : ℕ) : ℕ := 2 * n + 1

def b (n : ℕ) : ℚ :=
  (Finset.sum (Finset.range n) (fun i => a (i + 1))) / n

def sum_b (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (fun i => b (i + 1))

theorem sum_b_formula (n : ℕ) : sum_b n = (n * (n + 5) : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_b_formula_l2941_294147


namespace NUMINAMATH_CALUDE_min_value_theorem_l2941_294159

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- The first circle equation -/
def circle1 (x y : ℝ) : Prop := (x+2)^2 + y^2 = 4

/-- The second circle equation -/
def circle2 (x y : ℝ) : Prop := (x-2)^2 + y^2 = 1

/-- The expression |PM|^2 - |PN|^2 -/
def expr (x : ℝ) : ℝ := 8*x - 3

/-- The theorem stating the minimum value of |PM|^2 - |PN|^2 -/
theorem min_value_theorem (x y : ℝ) (h1 : hyperbola x y) (h2 : x ≥ 1) :
  ∃ (m : ℝ), m = 5 ∧ ∀ (x' y' : ℝ), hyperbola x' y' → x' ≥ 1 → expr x' ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2941_294159


namespace NUMINAMATH_CALUDE_lassie_bones_problem_l2941_294172

/-- The number of bones Lassie started with before Saturday -/
def initial_bones : ℕ := 50

/-- The number of bones Lassie has after eating on Saturday -/
def bones_after_saturday : ℕ := initial_bones / 2

/-- The number of bones Lassie receives on Sunday -/
def bones_received_sunday : ℕ := 10

/-- The total number of bones Lassie has after Sunday -/
def total_bones_after_sunday : ℕ := 35

theorem lassie_bones_problem :
  bones_after_saturday + bones_received_sunday = total_bones_after_sunday :=
by sorry

end NUMINAMATH_CALUDE_lassie_bones_problem_l2941_294172


namespace NUMINAMATH_CALUDE_test_total_points_l2941_294121

theorem test_total_points : 
  ∀ (total_problems : ℕ) 
    (three_point_problems : ℕ) 
    (four_point_problems : ℕ),
  total_problems = 30 →
  four_point_problems = 10 →
  three_point_problems + four_point_problems = total_problems →
  3 * three_point_problems + 4 * four_point_problems = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_test_total_points_l2941_294121


namespace NUMINAMATH_CALUDE_balloon_difference_l2941_294146

theorem balloon_difference (your_balloons friend_balloons : ℕ) 
  (h1 : your_balloons = 7) 
  (h2 : friend_balloons = 5) : 
  your_balloons - friend_balloons = 2 := by
sorry

end NUMINAMATH_CALUDE_balloon_difference_l2941_294146


namespace NUMINAMATH_CALUDE_banana_split_difference_l2941_294176

/-- The number of ice cream scoops in Oli's banana split -/
def oli_scoops : ℕ := 4

/-- The number of ice cream scoops in Victoria's banana split -/
def victoria_scoops : ℕ := 2 * oli_scoops + oli_scoops

/-- The number of ice cream scoops in Brian's banana split -/
def brian_scoops : ℕ := oli_scoops + 3

/-- The total difference in scoops of ice cream between Oli's, Victoria's, and Brian's banana splits -/
def total_difference : ℕ := 
  (victoria_scoops - oli_scoops) + (brian_scoops - oli_scoops) + (victoria_scoops - brian_scoops)

theorem banana_split_difference : total_difference = 16 := by
  sorry

end NUMINAMATH_CALUDE_banana_split_difference_l2941_294176


namespace NUMINAMATH_CALUDE_three_tails_one_head_probability_l2941_294167

/-- The probability of getting exactly three tails and one head when tossing four fair coins -/
theorem three_tails_one_head_probability : ℝ := by
  -- Define the probability of getting heads or tails on a single coin toss
  let single_coin_prob : ℝ := 1 / 2

  -- Define the number of coins
  let num_coins : ℕ := 4

  -- Define the number of ways to get 3 tails and 1 head
  let num_favorable_outcomes : ℕ := 4

  -- Define the total number of possible outcomes
  let total_outcomes : ℕ := 2^num_coins

  -- The probability we want to prove
  let target_prob : ℝ := 1 / 4

  sorry

end NUMINAMATH_CALUDE_three_tails_one_head_probability_l2941_294167


namespace NUMINAMATH_CALUDE_probability_distance_sqrt2_over_2_l2941_294168

/-- A point on a unit square, either a vertex or the center -/
inductive SquarePoint
  | vertex : Fin 4 → SquarePoint
  | center : SquarePoint

/-- The distance between two points on a unit square -/
def distance (p q : SquarePoint) : ℝ :=
  sorry

/-- The set of all possible pairs of points -/
def allPairs : Finset (SquarePoint × SquarePoint) :=
  sorry

/-- The set of pairs of points with distance √2/2 -/
def pairsWithDistance : Finset (SquarePoint × SquarePoint) :=
  sorry

theorem probability_distance_sqrt2_over_2 :
  (Finset.card pairsWithDistance : ℚ) / (Finset.card allPairs : ℚ) = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_distance_sqrt2_over_2_l2941_294168


namespace NUMINAMATH_CALUDE_max_y_value_l2941_294112

theorem max_y_value (x y : ℝ) (h : x^2 + y^2 = 18*x + 40*y) : 
  ∃ (max_y : ℝ), max_y = 20 + Real.sqrt 481 ∧ ∀ (y' : ℝ), ∃ (x' : ℝ), x'^2 + y'^2 = 18*x' + 40*y' → y' ≤ max_y := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l2941_294112


namespace NUMINAMATH_CALUDE_complement_of_union_eq_nonpositive_l2941_294109

-- Define the sets U, P, and Q
def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x * (x - 2) < 0}

-- State the theorem
theorem complement_of_union_eq_nonpositive :
  (U \ (P ∪ Q)) = {x : ℝ | x ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_eq_nonpositive_l2941_294109


namespace NUMINAMATH_CALUDE_system_of_inequalities_l2941_294180

theorem system_of_inequalities (x : ℝ) : 2*x + 1 > x ∧ x < -3*x + 8 → -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l2941_294180


namespace NUMINAMATH_CALUDE_adam_initial_amount_l2941_294177

/-- The cost of the airplane in dollars -/
def airplane_cost : ℚ := 4.28

/-- The change Adam receives after buying the airplane in dollars -/
def change_received : ℚ := 0.72

/-- Adam's initial amount of money in dollars -/
def initial_amount : ℚ := airplane_cost + change_received

theorem adam_initial_amount :
  initial_amount = 5 :=
by sorry

end NUMINAMATH_CALUDE_adam_initial_amount_l2941_294177


namespace NUMINAMATH_CALUDE_ball_arrangement_count_l2941_294103

/-- The number of ways to arrange 8 balls in a row, with 5 red balls (3 of which must be consecutive) and 3 white balls. -/
def ball_arrangements : ℕ := 30

/-- The total number of balls -/
def total_balls : ℕ := 8

/-- The number of red balls -/
def red_balls : ℕ := 5

/-- The number of white balls -/
def white_balls : ℕ := 3

/-- The number of consecutive red balls -/
def consecutive_red_balls : ℕ := 3

theorem ball_arrangement_count : 
  ball_arrangements = (Nat.choose (total_balls - consecutive_red_balls + 1) white_balls) * 
                      (Nat.choose (total_balls - white_balls - consecutive_red_balls + 1) 1) / 
                      (Nat.factorial (red_balls - consecutive_red_balls)) :=
sorry

end NUMINAMATH_CALUDE_ball_arrangement_count_l2941_294103


namespace NUMINAMATH_CALUDE_slant_height_angle_is_30_degrees_l2941_294151

/-- Regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  /-- Side length of the base square -/
  base_side : ℝ
  /-- Angle between lateral face and base plane -/
  lateral_angle : ℝ

/-- Angle between slant height and adjacent face -/
def slant_height_angle (p : RegularQuadPyramid) : ℝ :=
  sorry

theorem slant_height_angle_is_30_degrees (p : RegularQuadPyramid) 
  (h : p.lateral_angle = Real.pi / 4) : 
  slant_height_angle p = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_slant_height_angle_is_30_degrees_l2941_294151


namespace NUMINAMATH_CALUDE_total_late_time_l2941_294143

def charlize_late : ℕ := 20

def ana_late : ℕ := charlize_late + charlize_late / 4
def ben_late : ℕ := charlize_late * 3 / 4
def clara_late : ℕ := charlize_late * 2
def daniel_late : ℕ := 30 * 4 / 5

def ana_missed : ℕ := 5
def ben_missed : ℕ := 2
def clara_missed : ℕ := 15
def daniel_missed : ℕ := 10

theorem total_late_time :
  charlize_late +
  (ana_late + ana_missed) +
  (ben_late + ben_missed) +
  (clara_late + clara_missed) +
  (daniel_late + daniel_missed) = 156 := by
  sorry

end NUMINAMATH_CALUDE_total_late_time_l2941_294143


namespace NUMINAMATH_CALUDE_theo_donut_holes_l2941_294110

/-- Represents a worker coating donut holes -/
structure Worker where
  name : String
  radius : ℕ

/-- Calculates the surface area of a spherical donut hole -/
def surfaceArea (r : ℕ) : ℕ := 4 * r * r

/-- Calculates the number of donut holes coated by a worker when all workers finish simultaneously -/
def donutHolesCoated (workers : List Worker) (w : Worker) : ℕ :=
  let surfaces := workers.map (λ worker => surfaceArea worker.radius)
  let lcm := surfaces.foldl Nat.lcm 1
  lcm / (surfaceArea w.radius)

/-- The main theorem stating the number of donut holes Theo will coat -/
theorem theo_donut_holes (workers : List Worker) :
  workers = [
    ⟨"Niraek", 5⟩,
    ⟨"Theo", 7⟩,
    ⟨"Akshaj", 9⟩,
    ⟨"Mira", 11⟩
  ] →
  donutHolesCoated workers (Worker.mk "Theo" 7) = 1036830 := by
  sorry

end NUMINAMATH_CALUDE_theo_donut_holes_l2941_294110


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2941_294128

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def Q : Set ℝ := {y | ∃ x : ℝ, y = x + 1}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {y | y ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2941_294128


namespace NUMINAMATH_CALUDE_square_of_105_l2941_294126

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_square_of_105_l2941_294126


namespace NUMINAMATH_CALUDE_power_of_product_l2941_294152

theorem power_of_product (a : ℝ) : (2 * a)^3 = 8 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2941_294152


namespace NUMINAMATH_CALUDE_smallest_solution_of_quadratic_l2941_294192

theorem smallest_solution_of_quadratic : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 10*x - 24
  ∃ (x : ℝ), f x = 0 ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ x = -12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_quadratic_l2941_294192


namespace NUMINAMATH_CALUDE_oil_price_reduction_l2941_294102

theorem oil_price_reduction (original_price : ℝ) : 
  (original_price > 0) →
  (1100 = (1100 / original_price) * original_price) →
  (1100 = ((1100 / original_price) + 5) * (0.75 * original_price)) →
  (0.75 * original_price = 55) := by
sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l2941_294102


namespace NUMINAMATH_CALUDE_f_composition_at_one_over_e_l2941_294119

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x else Real.log x

-- State the theorem
theorem f_composition_at_one_over_e :
  f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_at_one_over_e_l2941_294119


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2941_294188

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a = 4 →
  c = 2 * Real.sqrt 2 →
  Real.cos A = -(Real.sqrt 2) / 4 →
  b = 2 ∧
  Real.sin C = (Real.sqrt 7) / 4 ∧
  Real.cos (2 * A + π / 6) = (Real.sqrt 7 - 3 * Real.sqrt 3) / 8 :=
by sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l2941_294188


namespace NUMINAMATH_CALUDE_x_cubed_coefficient_is_one_l2941_294162

-- Define the polynomial expression
def poly (x : ℝ) : ℝ := 2 * (x^3 - 2*x^2) + 3 * (x^2 - x^3 + x^4) - (5*x^4 - 2*x^3)

-- Theorem stating that the coefficient of x^3 in the expanded form of poly is 1
theorem x_cubed_coefficient_is_one :
  ∃ a b c d, ∀ x, poly x = a*x^4 + b*x^3 + c*x^2 + d*x ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_x_cubed_coefficient_is_one_l2941_294162


namespace NUMINAMATH_CALUDE_problem_statement_l2941_294155

theorem problem_statement (x y : ℝ) (h : -x + 2*y = 5) : 
  5*(x - 2*y)^2 - 3*(x - 2*y) - 60 = 80 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2941_294155


namespace NUMINAMATH_CALUDE_courtyard_paving_l2941_294149

/-- Calculate the number of bricks required to pave a courtyard -/
theorem courtyard_paving (courtyard_length courtyard_width brick_length brick_width : ℝ) 
  (h1 : courtyard_length = 35)
  (h2 : courtyard_width = 24)
  (h3 : brick_length = 0.15)
  (h4 : brick_width = 0.08) :
  (courtyard_length * courtyard_width) / (brick_length * brick_width) = 70000 := by
  sorry

#eval (35 * 24) / (0.15 * 0.08)

end NUMINAMATH_CALUDE_courtyard_paving_l2941_294149


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2941_294144

def polynomial (x : ℤ) : ℤ := x^3 - 4*x^2 - 11*x + 24

def is_root (x : ℤ) : Prop := polynomial x = 0

theorem integer_roots_of_polynomial :
  {x : ℤ | is_root x} = {-1, 3, 4} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2941_294144


namespace NUMINAMATH_CALUDE_group_size_proof_l2941_294199

theorem group_size_proof (n : ℕ) (k : ℕ) : 
  k = 7 → 
  (n : ℚ) - k ≠ 0 → 
  ((n - k) / n - k / n : ℚ) = 0.30000000000000004 → 
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_group_size_proof_l2941_294199


namespace NUMINAMATH_CALUDE_right_triangle_area_l2941_294174

/-- The area of a right triangle with legs 20 and 21 is 210 -/
theorem right_triangle_area : 
  let a : ℝ := 20
  let b : ℝ := 21
  let area : ℝ := (1/2) * a * b
  area = 210 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2941_294174


namespace NUMINAMATH_CALUDE_sets_with_property_P_l2941_294190

-- Define property P
def property_P (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ) (k : ℝ), (x, y) ∈ M → 0 < k → k < 1 → (k * x, k * y) ∈ M

-- Define the four sets
def set1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 ≥ p.2}
def set2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1^2 + p.2^2 < 1}
def set3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * p.1 + 2 * p.2 = 0}
def set4 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^3 + p.2^3 - p.1^2 * p.2 = 0}

-- Theorem stating which sets possess property P
theorem sets_with_property_P :
  property_P set2 ∧ property_P set4 ∧ ¬property_P set1 ∧ ¬property_P set3 := by
  sorry

end NUMINAMATH_CALUDE_sets_with_property_P_l2941_294190


namespace NUMINAMATH_CALUDE_tangent_circle_position_l2941_294186

/-- Represents a trapezoid EFGH with a circle tangent to two sides --/
structure TrapezoidWithTangentCircle where
  -- Lengths of the trapezoid sides
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  -- Q is the center of the circle on EF
  EQ : ℝ
  -- Assumption that EF is parallel to GH is implicit in the structure

/-- The main theorem about the tangent circle in a specific trapezoid --/
theorem tangent_circle_position 
  (t : TrapezoidWithTangentCircle)
  (h1 : t.EF = 86)
  (h2 : t.FG = 60)
  (h3 : t.GH = 26)
  (h4 : t.HE = 80)
  (h5 : t.EQ > 0)
  (h6 : t.EQ < t.EF) :
  t.EQ = 160 / 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_position_l2941_294186


namespace NUMINAMATH_CALUDE_ratio_equals_seven_l2941_294113

theorem ratio_equals_seven (x y z : ℝ) 
  (eq1 : 3 * x - 4 * y - 2 * z = 0)
  (eq2 : 2 * x + 6 * y - 21 * z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 + 4*x*y) / (y^2 + z^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equals_seven_l2941_294113


namespace NUMINAMATH_CALUDE_sqrt_product_equation_l2941_294133

theorem sqrt_product_equation (x : ℝ) (hx : x > 0) :
  Real.sqrt (16 * x) * Real.sqrt (25 * x) * Real.sqrt (5 * x) * Real.sqrt (20 * x) = 40 →
  x = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equation_l2941_294133


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l2941_294182

theorem simplify_complex_fraction (a b : ℝ) : 
  ((a - b)^2 + a*b) / ((a + b)^2 - a*b) / 
  ((a^5 + b^5 + a^2*b^3 + a^3*b^2) / 
   ((a^3 + b^3 + a^2*b + a*b^2) * (a^3 - b^3))) = a - b :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l2941_294182


namespace NUMINAMATH_CALUDE_interval_for_720_recordings_l2941_294125

/-- Calculates the time interval between recordings given the number of recordings in an hour -/
def timeInterval (recordings : ℕ) : ℚ :=
  3600 / recordings

/-- Theorem stating that 720 recordings in an hour results in a 5-second interval -/
theorem interval_for_720_recordings :
  timeInterval 720 = 5 := by
  sorry

end NUMINAMATH_CALUDE_interval_for_720_recordings_l2941_294125


namespace NUMINAMATH_CALUDE_albert_betty_age_ratio_l2941_294140

/-- Given the ages of Albert, Mary, and Betty, prove that the ratio of Albert's age to Betty's age is 4:1 -/
theorem albert_betty_age_ratio :
  ∀ (albert mary betty : ℕ),
  albert = 2 * mary →
  mary = albert - 8 →
  betty = 4 →
  (albert : ℚ) / betty = 4 / 1 := by
sorry

end NUMINAMATH_CALUDE_albert_betty_age_ratio_l2941_294140


namespace NUMINAMATH_CALUDE_overlap_difference_l2941_294139

/-- Represents the student population --/
def StudentPopulation : Set ℕ := {n : ℕ | 1000 ≤ n ∧ n ≤ 1200}

/-- Represents the number of students studying German --/
def GermanStudents (n : ℕ) : Set ℕ := {g : ℕ | (70 * n + 99) / 100 ≤ g ∧ g ≤ (75 * n) / 100}

/-- Represents the number of students studying Russian --/
def RussianStudents (n : ℕ) : Set ℕ := {r : ℕ | (35 * n + 99) / 100 ≤ r ∧ r ≤ (45 * n) / 100}

/-- The minimum number of students studying both languages --/
def m (n : ℕ) (g : ℕ) (r : ℕ) : ℕ := g + r - n

/-- The maximum number of students studying both languages --/
def M (n : ℕ) (g : ℕ) (r : ℕ) : ℕ := min g r

/-- Main theorem --/
theorem overlap_difference (n : StudentPopulation) 
  (g : GermanStudents n) (r : RussianStudents n) : 
  ∃ (m_val : ℕ) (M_val : ℕ), 
    m_val = m n g r ∧ 
    M_val = M n g r ∧ 
    M_val - m_val = 190 := by
  sorry

end NUMINAMATH_CALUDE_overlap_difference_l2941_294139


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l2941_294191

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle ABCD -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ := sorry

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := sorry

/-- Finds the point E on CD that is one-fourth the way from C to D -/
def findPointE (C D : Point) : Point := sorry

/-- Finds the intersection point F of BE and AC -/
def findIntersectionF (A B C E : Point) : Point := sorry

theorem rectangle_area_theorem (ABCD : Rectangle) :
  let E := findPointE ABCD.C ABCD.D
  let F := findIntersectionF ABCD.A ABCD.B ABCD.C E
  quadrilateralArea ABCD.A F E ABCD.D = 36 →
  rectangleArea ABCD = 144 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l2941_294191


namespace NUMINAMATH_CALUDE_cupboard_cost_price_l2941_294131

/-- The cost price of a cupboard satisfying certain conditions -/
def cost_price : ℝ := 5625

/-- The selling price of the cupboard -/
def selling_price : ℝ := 0.84 * cost_price

/-- The increased selling price that would result in a profit -/
def increased_selling_price : ℝ := 1.16 * cost_price

/-- Theorem stating that the cost price satisfies the given conditions -/
theorem cupboard_cost_price : 
  selling_price = 0.84 * cost_price ∧ 
  increased_selling_price = selling_price + 1800 :=
by sorry

end NUMINAMATH_CALUDE_cupboard_cost_price_l2941_294131


namespace NUMINAMATH_CALUDE_sum_of_squares_l2941_294178

theorem sum_of_squares (x y z : ℝ) :
  (x + y + z) / 3 = 10 →
  (x * y * z) ^ (1/3 : ℝ) = 7 →
  3 / (1/x + 1/y + 1/z) = 4 →
  x^2 + y^2 + z^2 = 385.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2941_294178


namespace NUMINAMATH_CALUDE_joan_pencils_l2941_294160

theorem joan_pencils (initial_pencils final_pencils : ℕ) 
  (h1 : initial_pencils = 33)
  (h2 : final_pencils = 60) :
  final_pencils - initial_pencils = 27 := by
  sorry

end NUMINAMATH_CALUDE_joan_pencils_l2941_294160


namespace NUMINAMATH_CALUDE_pentagon_perimeter_l2941_294100

/-- Pentagon FGHIJ with specified properties --/
structure Pentagon where
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  I : ℝ × ℝ
  J : ℝ × ℝ

/-- The perimeter of a pentagon --/
def perimeter (p : Pentagon) : ℝ :=
  dist p.F p.G + dist p.G p.H + dist p.H p.I + dist p.I p.J + dist p.J p.F

/-- The specific pentagon described in the problem --/
def specificPentagon : Pentagon where
  F := (0, 0)
  G := (0, -2)
  H := (2, -2)
  I := (3, -1)
  J := (1.5, 1.5)

theorem pentagon_perimeter :
  perimeter specificPentagon = 4 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_l2941_294100


namespace NUMINAMATH_CALUDE_largest_integer_x_l2941_294171

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def fraction (x : ℤ) : ℚ := (x^2 + 3*x + 8) / (x - 2)

theorem largest_integer_x : 
  (∀ x : ℤ, x > 1 → ¬ is_integer (fraction x)) ∧ 
  is_integer (fraction 1) :=
sorry

end NUMINAMATH_CALUDE_largest_integer_x_l2941_294171


namespace NUMINAMATH_CALUDE_chromatic_number_bound_l2941_294185

/-- A graph G is represented by its vertex set and edge set. -/
structure Graph (V : Type) where
  edges : Set (V × V)

/-- The chromatic number of a graph. -/
def chromaticNumber {V : Type} (G : Graph V) : ℕ :=
  sorry

/-- The number of edges in a graph. -/
def numEdges {V : Type} (G : Graph V) : ℕ :=
  sorry

/-- Theorem: The chromatic number of a graph is bounded by a function of its edge count. -/
theorem chromatic_number_bound {V : Type} (G : Graph V) :
  (chromaticNumber G : ℝ) ≤ 1/2 + Real.sqrt (2 * (numEdges G : ℝ) + 1/4) :=
sorry

end NUMINAMATH_CALUDE_chromatic_number_bound_l2941_294185


namespace NUMINAMATH_CALUDE_total_flour_in_bowl_l2941_294136

-- Define the initial amount of flour in the bowl
def initial_flour : ℚ := 2 + 3/4

-- Define the amount of flour added
def added_flour : ℚ := 45/100

-- Theorem to prove
theorem total_flour_in_bowl :
  initial_flour + added_flour = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_total_flour_in_bowl_l2941_294136


namespace NUMINAMATH_CALUDE_inequality_preservation_l2941_294117

theorem inequality_preservation (m n : ℝ) (h : m > n) : m - 6 > n - 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l2941_294117


namespace NUMINAMATH_CALUDE_ellipse_m_value_l2941_294130

/-- Given an ellipse with equation x²/10 + y²/m = 1, foci on the y-axis, and major axis 8, prove that m = 16 -/
theorem ellipse_m_value (x y m : ℝ) : 
  (∀ x y, x^2 / 10 + y^2 / m = 1) →  -- Ellipse equation
  (∃ c, c > 0 ∧ ∀ x, x^2 / 10 + (y + c)^2 / m = 1 ∧ x^2 / 10 + (y - c)^2 / m = 1) →  -- Foci on y-axis
  (∃ y, y^2 / m = 1 ∧ y = 4) →  -- Major axis is 8 (semi-major axis is 4)
  m = 16 := by
sorry


end NUMINAMATH_CALUDE_ellipse_m_value_l2941_294130


namespace NUMINAMATH_CALUDE_age_difference_l2941_294157

theorem age_difference :
  ∀ (a b : ℕ),
  (0 < a ∧ a < 10) →
  (0 < b ∧ b < 10) →
  (10 * a + b + 5 = 2 * (10 * b + a + 5)) →
  (10 * a + b) - (10 * b + a) = 18 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2941_294157


namespace NUMINAMATH_CALUDE_N_is_composite_l2941_294127

theorem N_is_composite : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (2011 * 2012 * 2013 * 2014 + 1 = a * b) := by
  sorry

end NUMINAMATH_CALUDE_N_is_composite_l2941_294127


namespace NUMINAMATH_CALUDE_fraction_simplification_l2941_294122

theorem fraction_simplification :
  (18 : ℚ) / 22 * 52 / 24 * 33 / 39 * 22 / 52 = 33 / 52 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2941_294122


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l2941_294107

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The main theorem stating that if the given points are collinear, then k = 24. -/
theorem collinear_points_k_value (k : ℝ) :
  collinear 1 (-2) 3 2 6 (k/3) → k = 24 := by
  sorry

#check collinear_points_k_value

end NUMINAMATH_CALUDE_collinear_points_k_value_l2941_294107


namespace NUMINAMATH_CALUDE_books_per_bookshelf_l2941_294116

/-- Given that Bryan has 34 books distributed equally in 2 bookshelves,
    prove that there are 17 books in each bookshelf. -/
theorem books_per_bookshelf :
  ∀ (total_books : ℕ) (num_bookshelves : ℕ) (books_per_shelf : ℕ),
    total_books = 34 →
    num_bookshelves = 2 →
    total_books = num_bookshelves * books_per_shelf →
    books_per_shelf = 17 := by
  sorry

end NUMINAMATH_CALUDE_books_per_bookshelf_l2941_294116


namespace NUMINAMATH_CALUDE_private_schools_in_B_l2941_294101

/-- Represents the three types of schools -/
inductive SchoolType
  | Public
  | Parochial
  | PrivateIndependent

/-- Represents the three districts -/
inductive District
  | A
  | B
  | C

/-- The total number of high schools -/
def total_schools : ℕ := 50

/-- The number of public schools -/
def public_schools : ℕ := 25

/-- The number of parochial schools -/
def parochial_schools : ℕ := 16

/-- The number of private independent schools -/
def private_schools : ℕ := 9

/-- The number of schools in District A -/
def schools_in_A : ℕ := 18

/-- The number of schools in District B -/
def schools_in_B : ℕ := 17

/-- Function to calculate the number of schools in District C -/
def schools_in_C : ℕ := total_schools - schools_in_A - schools_in_B

/-- Function to calculate the number of each type of school in District C -/
def schools_per_type_in_C : ℕ := schools_in_C / 3

theorem private_schools_in_B : 
  private_schools - schools_per_type_in_C = 4 := by sorry

end NUMINAMATH_CALUDE_private_schools_in_B_l2941_294101


namespace NUMINAMATH_CALUDE_dilution_proof_l2941_294184

def initial_volume : ℝ := 12
def initial_concentration : ℝ := 0.60
def final_concentration : ℝ := 0.40
def water_added : ℝ := 6

theorem dilution_proof :
  let initial_alcohol := initial_volume * initial_concentration
  let final_volume := initial_volume + water_added
  initial_alcohol / final_volume = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_dilution_proof_l2941_294184


namespace NUMINAMATH_CALUDE_line_direction_vector_l2941_294187

-- Define the two points on the line
def point1 : ℝ × ℝ := (-3, 0)
def point2 : ℝ × ℝ := (0, 3)

-- Define the direction vector
def direction_vector : ℝ × ℝ := (3, 3)

-- Theorem statement
theorem line_direction_vector :
  (point2.1 - point1.1, point2.2 - point1.2) = direction_vector :=
sorry

end NUMINAMATH_CALUDE_line_direction_vector_l2941_294187


namespace NUMINAMATH_CALUDE_bankers_discount_calculation_l2941_294166

/-- Banker's discount calculation -/
theorem bankers_discount_calculation (face_value : ℝ) (interest_rate : ℝ) (true_discount : ℝ)
  (h1 : face_value = 74500)
  (h2 : interest_rate = 0.15)
  (h3 : true_discount = 11175) :
  face_value * interest_rate = true_discount :=
by sorry

end NUMINAMATH_CALUDE_bankers_discount_calculation_l2941_294166


namespace NUMINAMATH_CALUDE_odd_divisors_implies_perfect_square_l2941_294153

theorem odd_divisors_implies_perfect_square (n : ℕ) :
  (∃ d : ℕ, Odd (Nat.divisors n).card) → ∃ k : ℕ, n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_odd_divisors_implies_perfect_square_l2941_294153


namespace NUMINAMATH_CALUDE_triangle_inequality_with_interior_point_l2941_294134

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ := sorry

-- Define a point inside the triangle
def insidePoint (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_inequality_with_interior_point (t : Triangle) :
  let P := perimeter t
  let O := insidePoint t
  P / 2 < distance O t.A + distance O t.B + distance O t.C ∧
  distance O t.A + distance O t.B + distance O t.C < P :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_interior_point_l2941_294134


namespace NUMINAMATH_CALUDE_no_rational_solution_to_5x2_plus_3y2_eq_1_l2941_294135

theorem no_rational_solution_to_5x2_plus_3y2_eq_1 :
  ¬ ∃ (x y : ℚ), 5 * x^2 + 3 * y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_no_rational_solution_to_5x2_plus_3y2_eq_1_l2941_294135


namespace NUMINAMATH_CALUDE_unique_prime_pair_divisibility_l2941_294150

theorem unique_prime_pair_divisibility : 
  ∃! (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    (3 * p^(q-1) + 1) ∣ (11^p + 17^p) ∧
    p = 3 ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_divisibility_l2941_294150


namespace NUMINAMATH_CALUDE_hyperbola_parabola_relation_l2941_294169

theorem hyperbola_parabola_relation (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) :
  (∃ (c : ℝ), c = 2 * a ∧ c^2 = a^2 + b^2) →
  (2 = (p / 2 / b) / Real.sqrt ((1 / a^2) + (1 / b^2))) →
  p = 8 := by sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_relation_l2941_294169


namespace NUMINAMATH_CALUDE_angle_measure_from_point_l2941_294137

/-- If a point P(sin 40°, 1 + cos 40°) is on the terminal side of an acute angle α, then α = 70°. -/
theorem angle_measure_from_point (α : Real) : 
  α > 0 ∧ α < 90 ∧ 
  ∃ (P : ℝ × ℝ), P.1 = Real.sin (40 * π / 180) ∧ P.2 = 1 + Real.cos (40 * π / 180) ∧
  P.2 / P.1 = Real.tan α → 
  α = 70 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_from_point_l2941_294137


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2941_294194

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and an asymptote 2x - √3y = 0,
    prove that its eccentricity is √21/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : ∀ x y : ℝ, 2 * x - Real.sqrt 3 * y = 0 → 
    (x^2 / a^2 - y^2 / b^2 = 1 ↔ x = 0 ∧ y = 0)) : 
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 21 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2941_294194


namespace NUMINAMATH_CALUDE_mod_equivalence_l2941_294124

theorem mod_equivalence (m : ℕ) : 
  198 * 864 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 22 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_l2941_294124


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l2941_294156

-- Define the line equation
def line (x y : ℝ) : Prop := 4 * x + 7 * y + 49 = 0

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define what it means for a line to be tangent to a parabola
def is_tangent (l : ℝ → ℝ → Prop) (p : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₀ y₀ : ℝ), l x₀ y₀ ∧ p x₀ y₀ ∧
    ∀ (x y : ℝ), l x y ∧ p x y → (x, y) = (x₀, y₀)

-- Theorem statement
theorem line_tangent_to_parabola :
  is_tangent line parabola :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l2941_294156


namespace NUMINAMATH_CALUDE_at_least_ten_mutual_reports_l2941_294179

-- Define the type for spies
def Spy : Type := ℕ

-- Define the total number of spies
def total_spies : ℕ := 20

-- Define the number of colleagues each spy reports on
def reports_per_spy : ℕ := 10

-- Define the reporting relation
def reports_on (s₁ s₂ : Spy) : Prop := sorry

-- State the theorem
theorem at_least_ten_mutual_reports :
  ∃ (mutual_reports : Finset (Spy × Spy)),
    (∀ (pair : Spy × Spy), pair ∈ mutual_reports →
      reports_on pair.1 pair.2 ∧ reports_on pair.2 pair.1) ∧
    mutual_reports.card ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_at_least_ten_mutual_reports_l2941_294179


namespace NUMINAMATH_CALUDE_a_equals_two_l2941_294170

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 4

-- State the theorem
theorem a_equals_two (a : ℝ) : f a a = f a 1 + 2 * (1 - 1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_two_l2941_294170


namespace NUMINAMATH_CALUDE_f_not_mapping_l2941_294193

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

-- Define the function f
def f (x : ℝ) : ℝ := x

-- Theorem stating that f is not a mapping from A to B
theorem f_not_mapping : ¬(∀ x ∈ A, f x ∈ B) :=
sorry

end NUMINAMATH_CALUDE_f_not_mapping_l2941_294193


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l2941_294142

theorem y_in_terms_of_x (x y : ℝ) (h : x - 2 = y + 3*x) : y = -2*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l2941_294142


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l2941_294158

theorem cos_double_angle_special_case (α : Real) 
  (h : Real.sin (π / 2 - α) = 1 / 4) : 
  Real.cos (2 * α) = - 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l2941_294158


namespace NUMINAMATH_CALUDE_choir_members_count_l2941_294145

theorem choir_members_count : ∃! n : ℕ, 
  150 < n ∧ n < 250 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 4 ∧ 
  n % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_choir_members_count_l2941_294145


namespace NUMINAMATH_CALUDE_det_transformation_l2941_294196

/-- Given a 2x2 matrix with determinant 7, prove that a specific transformation of this matrix also has determinant 7. -/
theorem det_transformation (p q r s : ℝ) : 
  Matrix.det !![p, q; r, s] = 7 → 
  Matrix.det !![p + 2*r, q + 2*s; r, s] = 7 := by
  sorry

end NUMINAMATH_CALUDE_det_transformation_l2941_294196


namespace NUMINAMATH_CALUDE_smallest_positive_linear_combination_l2941_294163

theorem smallest_positive_linear_combination : 
  (∃ (k : ℕ+), k = Nat.gcd 3003 60606 ∧ 
   (∀ (x : ℕ+), (∃ (m n : ℤ), x.val = 3003 * m + 60606 * n) → k ≤ x) ∧
   (∃ (m n : ℤ), k.val = 3003 * m + 60606 * n)) ∧
  Nat.gcd 3003 60606 = 273 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_linear_combination_l2941_294163


namespace NUMINAMATH_CALUDE_orange_ribbons_l2941_294198

theorem orange_ribbons (total : ℕ) (yellow purple orange silver : ℕ) : 
  yellow + purple + orange + silver = total →
  4 * yellow = total →
  3 * purple = total →
  6 * orange = total →
  silver = 40 →
  orange = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_ribbons_l2941_294198


namespace NUMINAMATH_CALUDE_f_has_real_roots_a_range_l2941_294154

-- Define the quadratic function f
def f (a x : ℝ) : ℝ := x^2 + (2*a - 1)*x + 1 - 2*a

-- Theorem 1: For all a ∈ ℝ, f(x) = 1 has real roots
theorem f_has_real_roots (a : ℝ) : ∃ x : ℝ, f a x = 1 := by sorry

-- Theorem 2: If f has zero points in (-1,0) and (0,1/2), then 1/2 < a < 3/4
theorem a_range (a : ℝ) (h1 : f a (-1) > 0) (h2 : f a 0 < 0) (h3 : f a (1/2) > 0) :
  1/2 < a ∧ a < 3/4 := by sorry

end NUMINAMATH_CALUDE_f_has_real_roots_a_range_l2941_294154


namespace NUMINAMATH_CALUDE_division_of_fractions_calculate_fraction_division_l2941_294118

theorem division_of_fractions (a b c : ℚ) (hb : b ≠ 0) :
  a / (c / b) = (a * b) / c :=
by sorry

theorem calculate_fraction_division :
  (4 : ℚ) / (5 / 7) = 28 / 5 :=
by sorry

end NUMINAMATH_CALUDE_division_of_fractions_calculate_fraction_division_l2941_294118


namespace NUMINAMATH_CALUDE_newer_train_theorem_l2941_294108

/-- Calculates the distance traveled by a newer train given the distance of an older train and the percentage increase in distance. -/
def newer_train_distance (old_distance : ℝ) (percent_increase : ℝ) : ℝ :=
  old_distance * (1 + percent_increase)

/-- Theorem stating that a newer train traveling 30% farther than an older train that goes 180 miles will travel 234 miles. -/
theorem newer_train_theorem :
  newer_train_distance 180 0.3 = 234 := by
  sorry

#eval newer_train_distance 180 0.3

end NUMINAMATH_CALUDE_newer_train_theorem_l2941_294108


namespace NUMINAMATH_CALUDE_sum_x_z_equals_4036_l2941_294183

theorem sum_x_z_equals_4036 (x y z : ℝ) 
  (eq1 : x + y + z = 0)
  (eq2 : 2016 * x + 2017 * y + 2018 * z = 0)
  (eq3 : 2016^2 * x + 2017^2 * y + 2018^2 * z = 2018) :
  x + z = 4036 := by sorry

end NUMINAMATH_CALUDE_sum_x_z_equals_4036_l2941_294183


namespace NUMINAMATH_CALUDE_tree_planting_change_l2941_294195

/-- Represents the road with tree planting configuration -/
structure RoadConfig where
  length : ℕ
  initial_spacing : ℕ
  new_spacing : ℕ

/-- Calculates the number of trees for a given spacing -/
def trees_count (config : RoadConfig) (spacing : ℕ) : ℕ :=
  config.length / spacing + 1

/-- Calculates the change in number of holes -/
def hole_change (config : RoadConfig) : ℤ :=
  (trees_count config config.new_spacing : ℤ) - (trees_count config config.initial_spacing : ℤ)

theorem tree_planting_change (config : RoadConfig) 
  (h_length : config.length = 240)
  (h_initial : config.initial_spacing = 8)
  (h_new : config.new_spacing = 6) :
  hole_change config = 10 ∧ max (-(hole_change config)) 0 = 0 :=
sorry

end NUMINAMATH_CALUDE_tree_planting_change_l2941_294195


namespace NUMINAMATH_CALUDE_area_code_count_l2941_294148

/-- The number of uppercase letters available -/
def uppercaseLetters : Nat := 26

/-- The number of lowercase letters and digits available for the second character -/
def secondCharOptions : Nat := 36

/-- The number of special characters available -/
def specialChars : Nat := 10

/-- The number of digits available -/
def digits : Nat := 10

/-- The total number of unique area codes that can be created -/
def totalAreaCodes : Nat := 
  (uppercaseLetters * secondCharOptions) + 
  (uppercaseLetters * secondCharOptions * specialChars) + 
  (uppercaseLetters * secondCharOptions * specialChars * digits)

theorem area_code_count : totalAreaCodes = 103896 := by
  sorry

end NUMINAMATH_CALUDE_area_code_count_l2941_294148


namespace NUMINAMATH_CALUDE_parallel_lines_symmetry_intersecting_lines_symmetry_l2941_294123

-- Define a type for lines in a plane
structure Line2D where
  slope : ℝ
  intercept : ℝ

-- Define a type for points in a plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line2D) : Prop :=
  l1.slope = l2.slope

-- Define a function to check if two lines intersect
def intersect (l1 l2 : Line2D) : Prop :=
  ¬(are_parallel l1 l2)

-- Define a type for axis of symmetry
structure AxisOfSymmetry where
  line : Line2D

-- Define a type for center of symmetry
structure CenterOfSymmetry where
  point : Point2D

-- Theorem for parallel lines
theorem parallel_lines_symmetry (l1 l2 : Line2D) (h : are_parallel l1 l2) :
  ∃ (axis : AxisOfSymmetry), (∀ (center : CenterOfSymmetry), True) :=
sorry

-- Theorem for intersecting lines
theorem intersecting_lines_symmetry (l1 l2 : Line2D) (h : intersect l1 l2) :
  ∃ (axis1 axis2 : AxisOfSymmetry) (center : CenterOfSymmetry),
    axis1.line.slope * axis2.line.slope = -1 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_symmetry_intersecting_lines_symmetry_l2941_294123


namespace NUMINAMATH_CALUDE_probability_yellow_ball_l2941_294111

theorem probability_yellow_ball (total_balls yellow_balls : ℕ) 
  (h1 : total_balls = 8)
  (h2 : yellow_balls = 5) :
  (yellow_balls : ℚ) / total_balls = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_ball_l2941_294111


namespace NUMINAMATH_CALUDE_complement_of_union_is_two_l2941_294106

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {1, 4}

-- Define set B
def B : Set ℕ := {3, 4}

-- Theorem statement
theorem complement_of_union_is_two :
  (U \ (A ∪ B)) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_two_l2941_294106


namespace NUMINAMATH_CALUDE_circle_equation_with_diameter_mn_l2941_294138

/-- Given points M (1, -1) and N (-1, 1), prove that the equation of the circle with diameter MN is x² + y² = 2 -/
theorem circle_equation_with_diameter_mn (x y : ℝ) : 
  let m : ℝ × ℝ := (1, -1)
  let n : ℝ × ℝ := (-1, 1)
  let center : ℝ × ℝ := ((m.1 + n.1) / 2, (m.2 + n.2) / 2)
  let radius : ℝ := Real.sqrt ((m.1 - n.1)^2 + (m.2 - n.2)^2) / 2
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ x^2 + y^2 = 2 :=
by sorry


end NUMINAMATH_CALUDE_circle_equation_with_diameter_mn_l2941_294138


namespace NUMINAMATH_CALUDE_milk_set_cost_l2941_294165

/-- The cost of a set of 2 packs of 500 mL milk -/
def set_cost : ℝ := 2.50

/-- The cost of an individual pack of 500 mL milk -/
def individual_cost : ℝ := 1.30

/-- The total savings when buying ten sets of 2 packs -/
def total_savings : ℝ := 1

theorem milk_set_cost :
  set_cost = 2 * individual_cost - total_savings / 10 :=
by sorry

end NUMINAMATH_CALUDE_milk_set_cost_l2941_294165


namespace NUMINAMATH_CALUDE_unique_number_pair_l2941_294132

theorem unique_number_pair : ∃! (a b : ℕ), 
  a + b = 2015 ∧ 
  ∃ (c : ℕ), c ≤ 9 ∧ a = 10 * b + c ∧
  a = 1832 ∧ b = 183 := by
sorry

end NUMINAMATH_CALUDE_unique_number_pair_l2941_294132


namespace NUMINAMATH_CALUDE_unique_prime_p_l2941_294161

theorem unique_prime_p : ∃! p : ℕ, Prime p ∧ Prime (5 * p + 1) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_p_l2941_294161


namespace NUMINAMATH_CALUDE_quadratic_inequality_necessary_condition_l2941_294120

theorem quadratic_inequality_necessary_condition (m : ℝ) :
  (∀ x : ℝ, x^2 - x + m > 0) → m > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_necessary_condition_l2941_294120


namespace NUMINAMATH_CALUDE_expression_evaluation_l2941_294164

theorem expression_evaluation (x y : ℚ) (hx : x = 1/3) (hy : y = -2) :
  (x * (x + y) - (x - y)^2) / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2941_294164


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l2941_294173

theorem complex_purely_imaginary (a : ℝ) : 
  (Complex.I * (2 * a + 1) : ℂ) = (2 + Complex.I) * (1 + a * Complex.I) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l2941_294173


namespace NUMINAMATH_CALUDE_shoe_refund_percentage_l2941_294129

/-- Given Will's shopping scenario, prove the percentage of shoe price refunded --/
theorem shoe_refund_percentage 
  (initial_amount : ℝ) 
  (sweater_cost : ℝ) 
  (tshirt_cost : ℝ) 
  (shoe_cost : ℝ) 
  (final_amount : ℝ) 
  (h1 : initial_amount = 74) 
  (h2 : sweater_cost = 9) 
  (h3 : tshirt_cost = 11) 
  (h4 : shoe_cost = 30) 
  (h5 : final_amount = 51) : 
  (final_amount - (initial_amount - (sweater_cost + tshirt_cost + shoe_cost))) / shoe_cost * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_shoe_refund_percentage_l2941_294129


namespace NUMINAMATH_CALUDE_positive_number_equality_l2941_294141

theorem positive_number_equality (x : ℝ) (h1 : x > 0) :
  (2 / 3) * x = (25 / 216) * (1 / x) → x = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_positive_number_equality_l2941_294141


namespace NUMINAMATH_CALUDE_solution_of_fraction_equation_l2941_294175

theorem solution_of_fraction_equation :
  ∃ x : ℝ, (3 - x) / (4 + 2*x) = 0 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_of_fraction_equation_l2941_294175


namespace NUMINAMATH_CALUDE_george_room_painting_choices_l2941_294189

theorem george_room_painting_choices :
  (Nat.choose 10 3) * 5 = 600 := by sorry

end NUMINAMATH_CALUDE_george_room_painting_choices_l2941_294189


namespace NUMINAMATH_CALUDE_max_area_is_one_l2941_294197

/-- A right triangle with legs 3 and 4, and hypotenuse 5 -/
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  is_right_triangle : leg1^2 + leg2^2 = hypotenuse^2
  leg1_is_3 : leg1 = 3
  leg2_is_4 : leg2 = 4
  hypotenuse_is_5 : hypotenuse = 5

/-- A rectangle inscribed in the right triangle with one side along the hypotenuse -/
structure InscribedRectangle (t : RightTriangle) where
  base : ℝ  -- Length of the rectangle's side along the hypotenuse
  height : ℝ -- Height of the rectangle
  is_inscribed : height ≤ t.leg2 * (1 - base / t.hypotenuse)
  on_hypotenuse : base ≤ t.hypotenuse

/-- The area of an inscribed rectangle -/
def area (t : RightTriangle) (r : InscribedRectangle t) : ℝ :=
  r.base * r.height

/-- The maximum area of an inscribed rectangle is 1 -/
theorem max_area_is_one (t : RightTriangle) : 
  ∃ (r : InscribedRectangle t), ∀ (r' : InscribedRectangle t), area t r ≥ area t r' ∧ area t r = 1 :=
sorry

end NUMINAMATH_CALUDE_max_area_is_one_l2941_294197


namespace NUMINAMATH_CALUDE_arbor_day_tree_planting_l2941_294181

theorem arbor_day_tree_planting 
  (original_trees_per_row : ℕ) 
  (original_rows : ℕ) 
  (new_rows : ℕ) 
  (h1 : original_trees_per_row = 20) 
  (h2 : original_rows = 18) 
  (h3 : new_rows = 10) : 
  (original_trees_per_row * original_rows) / new_rows = 36 := by
sorry

end NUMINAMATH_CALUDE_arbor_day_tree_planting_l2941_294181


namespace NUMINAMATH_CALUDE_cans_collected_l2941_294115

/-- The total number of cans collected by six people -/
def total_cans (solomon juwan levi gaby michelle sarah : ℕ) : ℕ :=
  solomon + juwan + levi + gaby + michelle + sarah

/-- Theorem stating the total number of cans collected by six people -/
theorem cans_collected :
  ∀ (solomon juwan levi gaby michelle sarah : ℕ),
    solomon = 66 →
    solomon = 3 * juwan →
    levi = juwan / 2 →
    gaby = (5 * solomon) / 2 →
    michelle = gaby / 3 →
    sarah = gaby - levi - 6 →
    total_cans solomon juwan levi gaby michelle sarah = 467 := by
  sorry

end NUMINAMATH_CALUDE_cans_collected_l2941_294115


namespace NUMINAMATH_CALUDE_condition_implies_increasing_l2941_294104

def IsIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem condition_implies_increasing (a : ℕ → ℝ) 
    (h : ∀ n : ℕ, a (n + 1) > |a n|) : 
  IsIncreasing a := by
  sorry

end NUMINAMATH_CALUDE_condition_implies_increasing_l2941_294104


namespace NUMINAMATH_CALUDE_sandwich_cost_l2941_294114

/-- The cost of Anna's sandwich given her breakfast and lunch expenses -/
theorem sandwich_cost (bagel_cost orange_juice_cost milk_cost lunch_difference : ℝ) : 
  bagel_cost = 0.95 →
  orange_juice_cost = 0.85 →
  milk_cost = 1.15 →
  lunch_difference = 4 →
  ∃ sandwich_cost : ℝ, 
    sandwich_cost + milk_cost = (bagel_cost + orange_juice_cost) + lunch_difference ∧
    sandwich_cost = 4.65 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_l2941_294114


namespace NUMINAMATH_CALUDE_original_fraction_l2941_294105

theorem original_fraction (x y : ℚ) : 
  (1.2 * x) / (0.9 * y) = 20 / 21 → x / y = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_l2941_294105
