import Mathlib

namespace connor_hourly_wage_l4064_406430

def sarah_daily_wage : ℝ := 288
def sarah_hours_worked : ℝ := 8
def sarah_connor_wage_ratio : ℝ := 6

theorem connor_hourly_wage :
  let sarah_hourly_wage := sarah_daily_wage / sarah_hours_worked
  sarah_hourly_wage / sarah_connor_wage_ratio = 6 := by
  sorry

end connor_hourly_wage_l4064_406430


namespace square_minus_four_l4064_406453

theorem square_minus_four (y : ℤ) (h : y^2 = 2209) : (y + 2) * (y - 2) = 2205 := by
  sorry

end square_minus_four_l4064_406453


namespace no_integer_solutions_for_fourth_power_equation_l4064_406476

theorem no_integer_solutions_for_fourth_power_equation :
  ¬ ∃ (a b c : ℤ), a^4 + b^4 = c^4 + 3 := by
  sorry

end no_integer_solutions_for_fourth_power_equation_l4064_406476


namespace product_congruence_l4064_406417

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def product_of_list (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

theorem product_congruence :
  let seq := arithmetic_sequence 3 5 21
  (product_of_list seq) % 6 = 3 := by
  sorry

end product_congruence_l4064_406417


namespace arithmetic_sequence_sum_l4064_406456

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 4 + a 5 = 12) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 :=
sorry

end arithmetic_sequence_sum_l4064_406456


namespace vectors_not_collinear_l4064_406437

def a : Fin 3 → ℝ := ![3, 5, 4]
def b : Fin 3 → ℝ := ![5, 9, 7]
def c₁ : Fin 3 → ℝ := fun i => -2 * a i + b i
def c₂ : Fin 3 → ℝ := fun i => 3 * a i - 2 * b i

theorem vectors_not_collinear : ¬ ∃ (k : ℝ), c₁ = fun i => k * c₂ i := by
  sorry

end vectors_not_collinear_l4064_406437


namespace sine_of_angle_between_vectors_l4064_406434

/-- Given vectors a and b with an angle θ between them, 
    if a = (2, 1) and 3b + a = (5, 4), then sin θ = √10/10 -/
theorem sine_of_angle_between_vectors (a b : ℝ × ℝ) (θ : ℝ) :
  a = (2, 1) →
  3 • b + a = (5, 4) →
  Real.sin θ = (Real.sqrt 10) / 10 := by
  sorry

end sine_of_angle_between_vectors_l4064_406434


namespace number_divisibility_l4064_406489

theorem number_divisibility (A B C D : ℤ) :
  let N := 1000*D + 100*C + 10*B + A
  (∃ k : ℤ, A + 2*B = 4*k → ∃ m : ℤ, N = 4*m) ∧
  (∃ k : ℤ, A + 2*B + 4*C = 8*k → ∃ m : ℤ, N = 8*m) ∧
  (∃ k : ℤ, A + 2*B + 4*C + 8*D = 16*k ∧ ∃ j : ℤ, B = 2*j → ∃ m : ℤ, N = 16*m) :=
by sorry

end number_divisibility_l4064_406489


namespace problem_solution_l4064_406433

theorem problem_solution : (2021^2 - 2021) / 2021 = 2020 := by
  sorry

end problem_solution_l4064_406433


namespace mice_without_coins_l4064_406498

theorem mice_without_coins (total_mice : ℕ) (total_coins : ℕ) 
  (h1 : total_mice = 40)
  (h2 : total_coins = 40)
  (h3 : ∃ (y z : ℕ), 
    2 * 2 + 7 * y + 4 * z = total_coins ∧
    2 + y + z + (total_mice - (2 + y + z)) = total_mice) :
  total_mice - (2 + y + z) = 32 :=
by sorry

end mice_without_coins_l4064_406498


namespace triangle_inequality_implies_equilateral_l4064_406467

/-- A triangle with sides a, b, c, area S, and centroid distances x, y, z from the vertices. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  x : ℝ
  y : ℝ
  z : ℝ

/-- The theorem stating that if a triangle satisfies the given inequality, it is equilateral. -/
theorem triangle_inequality_implies_equilateral (t : Triangle) :
  (t.x + t.y + t.z)^2 ≤ (t.a^2 + t.b^2 + t.c^2)/2 + 2*t.S*Real.sqrt 3 →
  t.a = t.b ∧ t.b = t.c :=
by sorry

end triangle_inequality_implies_equilateral_l4064_406467


namespace pipe_filling_time_l4064_406452

theorem pipe_filling_time (t_b t_ab : ℝ) (h_b : t_b = 20) (h_ab : t_ab = 20/3) :
  let t_a := (t_b * t_ab) / (t_b - t_ab)
  t_a = 10 := by sorry

end pipe_filling_time_l4064_406452


namespace solution_set_when_a_is_one_range_of_a_for_real_solutions_l4064_406416

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a^2| + |x + 2*a - 5|

-- Theorem for part (1)
theorem solution_set_when_a_is_one :
  {x : ℝ | |x + 1| + |x - 3| < 5} = Set.Ioo (-3/2) (7/2) := by sorry

-- Theorem for part (2)
theorem range_of_a_for_real_solutions :
  {a : ℝ | ∃ x, f x a < 5} = Set.Ioo 0 2 := by sorry

end solution_set_when_a_is_one_range_of_a_for_real_solutions_l4064_406416


namespace stone_counting_135_l4064_406413

/-- Represents the stone counting pattern described in the problem -/
def stoneCounting (n : ℕ) : ℕ := 
  let cycle := n % 24
  if cycle ≤ 12 
  then (cycle + 1) / 2 
  else (25 - cycle) / 2

/-- The problem statement -/
theorem stone_counting_135 : stoneCounting 135 = 3 := by
  sorry

end stone_counting_135_l4064_406413


namespace least_triangle_area_l4064_406447

/-- The solutions of the equation (z+4)^10 = 32 form a convex regular decagon in the complex plane. -/
def is_solution (z : ℂ) : Prop := (z + 4) ^ 10 = 32

/-- The set of all solutions forms a convex regular decagon. -/
def solution_set : Set ℂ := {z | is_solution z}

/-- A point is a vertex of the decagon if it's a solution. -/
def is_vertex (z : ℂ) : Prop := z ∈ solution_set

/-- The area of a triangle formed by three vertices of the decagon. -/
def triangle_area (v1 v2 v3 : ℂ) : ℝ :=
  sorry -- Definition of the area calculation

/-- The theorem stating the least possible area of a triangle formed by three vertices. -/
theorem least_triangle_area :
  ∃ (v1 v2 v3 : ℂ), is_vertex v1 ∧ is_vertex v2 ∧ is_vertex v3 ∧
    (∀ (w1 w2 w3 : ℂ), is_vertex w1 → is_vertex w2 → is_vertex w3 →
      triangle_area v1 v2 v3 ≤ triangle_area w1 w2 w3) ∧
    triangle_area v1 v2 v3 = (2^(2/5) * (Real.sqrt 5 - 1)) / 8 :=
sorry

end least_triangle_area_l4064_406447


namespace ages_four_years_ago_l4064_406485

/-- Represents the ages of four people: Amar, Akbar, Anthony, and Alex -/
structure Ages :=
  (amar : ℕ)
  (akbar : ℕ)
  (anthony : ℕ)
  (alex : ℕ)

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.amar + ages.akbar + ages.anthony + ages.alex = 88 ∧
  (ages.amar - 4) + (ages.akbar - 4) + (ages.anthony - 4) = 66 ∧
  ages.amar = 2 * ages.alex ∧
  ages.akbar = ages.amar - 3

/-- The theorem to be proved -/
theorem ages_four_years_ago (ages : Ages) 
  (h : satisfies_conditions ages) : 
  (ages.amar - 4) + (ages.akbar - 4) + (ages.anthony - 4) + (ages.alex - 4) = 72 := by
  sorry

end ages_four_years_ago_l4064_406485


namespace roses_in_vase_l4064_406455

/-- The number of roses in a vase after adding new roses -/
def total_roses (initial_roses new_roses : ℕ) : ℕ :=
  initial_roses + new_roses

/-- Theorem: There are 23 roses in the vase after Jessica adds her newly cut roses -/
theorem roses_in_vase : total_roses 7 16 = 23 := by
  sorry

end roses_in_vase_l4064_406455


namespace maria_change_l4064_406421

/-- The change Maria receives when buying apples -/
theorem maria_change (num_apples : ℕ) (price_per_apple : ℚ) (paid_amount : ℚ) : 
  num_apples = 5 → 
  price_per_apple = 3/4 → 
  paid_amount = 10 → 
  paid_amount - (num_apples : ℚ) * price_per_apple = 25/4 := by
  sorry

#check maria_change

end maria_change_l4064_406421


namespace triangle_solution_l4064_406464

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line in the 2D plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem setup
def triangle_problem (t : Triangle) (median_CM : Line) (altitude_BH : Line) : Prop :=
  t.A = (5, 1) ∧
  median_CM = ⟨2, -1, -5⟩ ∧
  altitude_BH = ⟨1, -2, -5⟩

-- Theorem statement
theorem triangle_solution (t : Triangle) (median_CM : Line) (altitude_BH : Line) 
  (h : triangle_problem t median_CM altitude_BH) :
  (∃ (line_AC : Line), line_AC = ⟨2, 1, -11⟩) ∧ 
  t.B = (-1, -3) := by
  sorry

end triangle_solution_l4064_406464


namespace divisibility_by_17_l4064_406486

theorem divisibility_by_17 (x y : ℤ) : 
  (∃ k : ℤ, 2*x + 3*y = 17*k) → (∃ m : ℤ, 9*x + 5*y = 17*m) := by
  sorry

end divisibility_by_17_l4064_406486


namespace joshuas_share_l4064_406458

theorem joshuas_share (total : ℕ) (joshua_share : ℕ) (justin_share : ℕ) : 
  total = 40 → 
  joshua_share = 3 * justin_share → 
  total = joshua_share + justin_share → 
  joshua_share = 30 := by
sorry

end joshuas_share_l4064_406458


namespace hammond_statues_weight_l4064_406465

/-- The weight of Hammond's marble statues problem -/
theorem hammond_statues_weight (original_weight : ℕ) (first_statue : ℕ) (third_statue : ℕ) (fourth_statue : ℕ) (discarded : ℕ) :
  original_weight = 80 ∧ 
  first_statue = 10 ∧ 
  third_statue = 15 ∧ 
  fourth_statue = 15 ∧ 
  discarded = 22 →
  ∃ (second_statue : ℕ), 
    second_statue = 18 ∧ 
    original_weight = first_statue + second_statue + third_statue + fourth_statue + discarded :=
by sorry

end hammond_statues_weight_l4064_406465


namespace exists_asymmetric_but_rotational_invariant_figure_l4064_406445

/-- A convex figure in a 2D plane. -/
structure ConvexFigure where
  -- We don't need to fully define the structure, just declare it exists
  dummy : Unit

/-- Represents a rotation in 2D space. -/
structure Rotation where
  angle : ℝ

/-- Checks if a figure has an axis of symmetry. -/
def hasAxisOfSymmetry (figure : ConvexFigure) : Prop :=
  sorry

/-- Applies a rotation to a figure. -/
def applyRotation (figure : ConvexFigure) (rotation : Rotation) : ConvexFigure :=
  sorry

/-- Checks if a figure is invariant under a given rotation. -/
def isInvariantUnderRotation (figure : ConvexFigure) (rotation : Rotation) : Prop :=
  applyRotation figure rotation = figure

/-- The main theorem: There exists a convex figure with no axis of symmetry
    but invariant under 120° rotation. -/
theorem exists_asymmetric_but_rotational_invariant_figure :
  ∃ (figure : ConvexFigure),
    ¬(hasAxisOfSymmetry figure) ∧
    isInvariantUnderRotation figure ⟨2 * Real.pi / 3⟩ := by
  sorry

end exists_asymmetric_but_rotational_invariant_figure_l4064_406445


namespace hyperbola_equation_l4064_406477

/-- Given a hyperbola with asymptote equations y = ± (1/3)x and one focus at (√10, 0),
    its standard equation is x²/9 - y² = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ (k : ℝ), k > 0 ∧ y = k * x / 3 ∨ y = -k * x / 3) →  -- asymptote equations
  (∃ (c : ℝ), c^2 = 10 ∧ (c, 0) ∈ {p : ℝ × ℝ | p.1^2 / 9 - p.2^2 = 1}) →  -- focus condition
  (x^2 / 9 - y^2 = 1) :=  -- standard equation
by sorry

end hyperbola_equation_l4064_406477


namespace subset_condition_disjoint_condition_l4064_406407

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < 2*a + 1}

-- Theorem for A ⊆ B
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ 1/2 ≤ a ∧ a ≤ 1 := by sorry

-- Theorem for A ∩ B = ∅
theorem disjoint_condition (a : ℝ) : A ∩ B a = ∅ ↔ a ≥ 3/2 ∨ a ≤ 0 := by sorry

end subset_condition_disjoint_condition_l4064_406407


namespace min_value_implies_a_solution_set_inequality_l4064_406436

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4|

-- Theorem for part 1
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f (2*x + a) + f (2*x - a) ≥ 4) ∧
  (∃ x, f (2*x + a) + f (2*x - a) = 4) →
  a = 2 ∨ a = -2 :=
sorry

-- Theorem for part 2
theorem solution_set_inequality :
  {x : ℝ | f x > 1 - (1/2)*x} = {x : ℝ | x > -2 ∨ x < -10} :=
sorry

end min_value_implies_a_solution_set_inequality_l4064_406436


namespace jake_sister_weight_ratio_l4064_406451

/-- Represents the weight ratio problem of Jake and his sister -/
theorem jake_sister_weight_ratio :
  let jake_present_weight : ℕ := 108
  let total_weight : ℕ := 156
  let weight_loss : ℕ := 12
  let jake_new_weight : ℕ := jake_present_weight - weight_loss
  let sister_weight : ℕ := total_weight - jake_new_weight
  (jake_new_weight : ℚ) / sister_weight = 8 / 5 :=
by sorry

end jake_sister_weight_ratio_l4064_406451


namespace complete_square_sum_l4064_406409

theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → 
  b + c = 5 := by
sorry

end complete_square_sum_l4064_406409


namespace m_range_theorem_l4064_406400

/-- Proposition p: There exists x ∈ ℝ, such that 2x² + (m-1)x + 1/2 ≤ 0 -/
def p (m : ℝ) : Prop := ∃ x : ℝ, 2 * x^2 + (m - 1) * x + 1/2 ≤ 0

/-- Proposition q: The curve C₁: x²/m² + y²/(2m+8) = 1 represents an ellipse with foci on the x-axis -/
def q (m : ℝ) : Prop := m ≠ 0 ∧ 2 * m + 8 > 0 ∧ m^2 > 2 * m + 8

/-- The range of m satisfying the given conditions -/
def m_range (m : ℝ) : Prop :=
  (3 ≤ m ∧ m ≤ 4) ∨ (-2 ≤ m ∧ m ≤ -1) ∨ m ≤ -4

theorem m_range_theorem (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m := by
  sorry

end m_range_theorem_l4064_406400


namespace ellipse_equation_l4064_406460

/-- An ellipse with center at origin, eccentricity √3/2, and one focus coinciding with
    the focus of the parabola x² = -4√3y has the equation x² + y²/4 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let e : ℝ := Real.sqrt 3 / 2
  let c : ℝ := Real.sqrt 3  -- Distance from center to focus
  let a : ℝ := c / e        -- Semi-major axis
  let b : ℝ := Real.sqrt (a^2 - c^2)  -- Semi-minor axis
  (e = Real.sqrt 3 / 2) → 
  (c = Real.sqrt 3) →      -- Focus coincides with parabola focus
  (x^2 + y^2 / 4 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1) :=
by sorry

end ellipse_equation_l4064_406460


namespace f_increasing_sufficient_not_necessary_l4064_406469

/-- The function f(x) defined as |x-a| + |x| --/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x|

/-- f is increasing on [0, +∞) --/
def is_increasing_on_nonneg (a : ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f a x ≤ f a y

theorem f_increasing_sufficient_not_necessary :
  (∀ a : ℝ, a < 0 → is_increasing_on_nonneg a) ∧
  (∃ a : ℝ, a ≥ 0 ∧ is_increasing_on_nonneg a) :=
sorry

end f_increasing_sufficient_not_necessary_l4064_406469


namespace exam_score_problem_l4064_406454

theorem exam_score_problem (correct_score : ℕ) (wrong_score : ℕ) (total_score : ℕ) (correct_answers : ℕ) :
  correct_score = 4 →
  wrong_score = 1 →
  total_score = 160 →
  correct_answers = 44 →
  ∃ (total_questions : ℕ),
    total_questions = correct_answers + (total_score - correct_score * correct_answers) / wrong_score ∧
    total_questions = 60 :=
by sorry

end exam_score_problem_l4064_406454


namespace geometric_sequence_fourth_term_l4064_406459

theorem geometric_sequence_fourth_term :
  ∀ (a : ℕ → ℝ),
    (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
    a 1 = 6 →                            -- First term
    a 5 = 1458 →                         -- Last term
    a 4 = 162 :=                         -- Fourth term to prove
by
  sorry

end geometric_sequence_fourth_term_l4064_406459


namespace total_football_games_l4064_406412

/-- The total number of football games in a year, given the number of games
    Keith attended and missed. -/
theorem total_football_games (attended : ℕ) (missed : ℕ) 
  (h1 : attended = 4) (h2 : missed = 4) : 
  attended + missed = 8 := by
  sorry

end total_football_games_l4064_406412


namespace min_value_of_function_l4064_406410

theorem min_value_of_function (x : ℝ) (h : x > 5/4) : 
  ∀ y : ℝ, y = 4*x + 1/(4*x - 5) → y ≥ 7 := by sorry

end min_value_of_function_l4064_406410


namespace hyperbola_ellipse_shared_foci_l4064_406432

theorem hyperbola_ellipse_shared_foci (m : ℝ) : 
  (∃ (c : ℝ), c > 0 ∧ 
    c^2 = 12 - 4 ∧ 
    c^2 = m + 1) → 
  m = 7 :=
by sorry

end hyperbola_ellipse_shared_foci_l4064_406432


namespace solution_form_l4064_406492

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop :=
  x / y + y / z + z / x = x / z + z / y + y / x

def equation2 (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 = x*y + y*z + z*x + 4

-- Theorem statement
theorem solution_form (x y z : ℝ) :
  equation1 x y z ∧ equation2 x y z →
  (∃ t : ℝ, (x = t ∧ y = t - 2 ∧ z = t - 2) ∨ (x = t ∧ y = t + 2 ∧ z = t + 2)) :=
by sorry

end solution_form_l4064_406492


namespace johann_oranges_l4064_406468

theorem johann_oranges (initial : ℕ) (eaten : ℕ) (returned : ℕ) (final : ℕ) : 
  initial = 60 →
  returned = 5 →
  final = 30 →
  (initial - eaten) / 2 + returned = final →
  eaten = 10 := by
sorry

end johann_oranges_l4064_406468


namespace acute_angle_range_l4064_406415

/-- The range of values for the acute angle of a line with slope k = 2m / (m^2 + 1) -/
theorem acute_angle_range (m : ℝ) (h1 : m ≥ 0) (h2 : m^2 + 1 ≥ 2*m) :
  let k := 2*m / (m^2 + 1)
  let θ := Real.arctan k
  0 ≤ θ ∧ θ ≤ π/4 :=
sorry

end acute_angle_range_l4064_406415


namespace exists_point_on_line_with_sum_of_distances_l4064_406487

-- Define the line l
variable (l : Line)

-- Define points A and B
variable (A B : Point)

-- Define the given segment length
variable (a : ℝ)

-- Define the property that A and B are on the same side of l
def sameSideOfLine (A B : Point) (l : Line) : Prop := sorry

-- Define the distance between two points
def distance (P Q : Point) : ℝ := sorry

-- Define what it means for a point to be on a line
def onLine (P : Point) (l : Line) : Prop := sorry

-- State the theorem
theorem exists_point_on_line_with_sum_of_distances
  (h_same_side : sameSideOfLine A B l) :
  ∃ M : Point, onLine M l ∧ distance M A + distance M B = a := sorry

end exists_point_on_line_with_sum_of_distances_l4064_406487


namespace regular_polygon_sides_l4064_406472

theorem regular_polygon_sides (n : ℕ) : 
  n > 2 → 
  (180 * (n - 2) : ℝ) / n = 160 → 
  n = 18 := by sorry

end regular_polygon_sides_l4064_406472


namespace height_relation_l4064_406497

/-- Two right circular cylinders with equal volume and related radii -/
structure TwoCylinders where
  r1 : ℝ  -- radius of the first cylinder
  h1 : ℝ  -- height of the first cylinder
  r2 : ℝ  -- radius of the second cylinder
  h2 : ℝ  -- height of the second cylinder
  r1_pos : 0 < r1  -- r1 is positive
  h1_pos : 0 < h1  -- h1 is positive
  r2_pos : 0 < r2  -- r2 is positive
  h2_pos : 0 < h2  -- h2 is positive
  volume_eq : r1^2 * h1 = r2^2 * h2  -- volumes are equal
  radius_relation : r2 = 1.2 * r1  -- second radius is 20% more than the first

/-- The height of the first cylinder is 44% more than the height of the second cylinder -/
theorem height_relation (c : TwoCylinders) : c.h1 = 1.44 * c.h2 := by
  sorry

end height_relation_l4064_406497


namespace distribute_5_2_l4064_406457

/-- The number of ways to distribute n indistinguishable objects into k indistinguishable containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 3 ways to distribute 5 indistinguishable balls into 2 indistinguishable boxes -/
theorem distribute_5_2 : distribute 5 2 = 3 := by sorry

end distribute_5_2_l4064_406457


namespace road_construction_progress_l4064_406419

theorem road_construction_progress (total_length : ℚ) 
  (h1 : total_length = 1/2) 
  (day1_progress : ℚ) (h2 : day1_progress = 1/10)
  (day2_progress : ℚ) (h3 : day2_progress = 1/5) :
  1 - day1_progress - day2_progress = 7/10 := by
sorry

end road_construction_progress_l4064_406419


namespace parking_lot_cars_l4064_406482

/-- Given a parking lot with large and small cars, prove the number of each type. -/
theorem parking_lot_cars (total_vehicles : ℕ) (total_wheels : ℕ) 
  (large_car_wheels : ℕ) (small_car_wheels : ℕ) 
  (h_total_vehicles : total_vehicles = 6)
  (h_total_wheels : total_wheels = 32)
  (h_large_car_wheels : large_car_wheels = 6)
  (h_small_car_wheels : small_car_wheels = 4) :
  ∃ (large_cars small_cars : ℕ),
    large_cars + small_cars = total_vehicles ∧
    large_cars * large_car_wheels + small_cars * small_car_wheels = total_wheels ∧
    large_cars = 4 ∧
    small_cars = 2 := by
  sorry

end parking_lot_cars_l4064_406482


namespace nancy_marks_l4064_406463

theorem nancy_marks (history : ℕ) (home_economics : ℕ) (physical_education : ℕ) (art : ℕ) (average : ℕ) 
  (h1 : history = 75)
  (h2 : home_economics = 52)
  (h3 : physical_education = 68)
  (h4 : art = 89)
  (h5 : average = 70) :
  ∃ (american_literature : ℕ), 
    (history + home_economics + physical_education + art + american_literature) / 5 = average ∧ 
    american_literature = 66 := by
  sorry

end nancy_marks_l4064_406463


namespace car_sale_profit_l4064_406488

theorem car_sale_profit (P : ℝ) (h : P > 0) :
  let buying_price := 0.95 * P
  let selling_price := 1.52 * P
  (selling_price - buying_price) / buying_price * 100 = 60 := by
  sorry

end car_sale_profit_l4064_406488


namespace sugar_cube_theorem_l4064_406444

/-- Represents a box of sugar cubes -/
structure SugarBox where
  height : Nat
  width : Nat
  depth : Nat

/-- Calculates the number of remaining cubes in a sugar box after eating layers -/
def remaining_cubes (box : SugarBox) : Set Nat :=
  if box.width * box.depth = 77 ∧ box.height * box.depth = 55 then
    if box.depth = 1 then {0}
    else if box.depth = 11 then {300}
    else ∅
  else ∅

/-- Theorem stating that the number of remaining cubes is either 300 or 0 -/
theorem sugar_cube_theorem (box : SugarBox) :
  remaining_cubes box ⊆ {0, 300} :=
by sorry

end sugar_cube_theorem_l4064_406444


namespace complex_modulus_problem_l4064_406481

theorem complex_modulus_problem (x y : ℝ) (z : ℂ) :
  z = x + y * Complex.I →
  x / (1 - Complex.I) = 1 + y * Complex.I →
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l4064_406481


namespace power_division_rule_l4064_406427

theorem power_division_rule (a : ℝ) : a^7 / a^3 = a^4 := by
  sorry

end power_division_rule_l4064_406427


namespace shortest_altitude_right_triangle_l4064_406440

theorem shortest_altitude_right_triangle :
  ∀ (a b c h : ℝ),
  a = 9 ∧ b = 12 ∧ c = 15 →
  a^2 + b^2 = c^2 →
  h = (2 * (a * b) / 2) / c →
  h ≤ a ∧ h ≤ b →
  h = 7.2 :=
by sorry

end shortest_altitude_right_triangle_l4064_406440


namespace least_integer_absolute_value_l4064_406418

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, |3*y - 5| ≤ 22 → y ≥ x) ↔ x = -5 := by
sorry

end least_integer_absolute_value_l4064_406418


namespace number_calculation_l4064_406491

theorem number_calculation (x : ℝ) : (0.5 * x - 10 = 25) → x = 70 := by
  sorry

end number_calculation_l4064_406491


namespace integer_pair_divisibility_l4064_406493

theorem integer_pair_divisibility (m n : ℕ+) : 
  (∃ k : ℤ, (m : ℤ) + (n : ℤ)^2 = k * ((m : ℤ)^2 - (n : ℤ))) ∧ 
  (∃ l : ℤ, (n : ℤ) + (m : ℤ)^2 = l * ((n : ℤ)^2 - (m : ℤ))) ↔ 
  ((m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2)) := by
sorry

end integer_pair_divisibility_l4064_406493


namespace find_number_l4064_406461

theorem find_number : ∃ x : ℝ, 4.75 + 0.432 + x = 5.485 ∧ x = 0.303 := by
  sorry

end find_number_l4064_406461


namespace car_profit_percentage_l4064_406462

theorem car_profit_percentage (P : ℝ) (h : P > 0) : 
  let buying_price := P * (1 - 0.2)
  let selling_price := buying_price * (1 + 0.45)
  let profit := selling_price - P
  profit / P * 100 = 16 := by
sorry

end car_profit_percentage_l4064_406462


namespace difference_d_minus_b_l4064_406466

theorem difference_d_minus_b (a b c d : ℕ+) 
  (h1 : a^5 = b^4) 
  (h2 : c^3 = d^2) 
  (h3 : c - a = 19) : 
  d - b = 757 := by
  sorry

end difference_d_minus_b_l4064_406466


namespace unique_modular_congruence_l4064_406420

theorem unique_modular_congruence :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -825 [ZMOD 8] := by
  sorry

end unique_modular_congruence_l4064_406420


namespace symmetric_points_y_coordinate_l4064_406424

/-- Given two points P and Q in a 2D Cartesian coordinate system that are symmetric about the origin,
    prove that the y-coordinate of Q is -3. -/
theorem symmetric_points_y_coordinate
  (P Q : ℝ × ℝ)  -- P and Q are points in 2D real space
  (h_P : P = (-3, 5))  -- Coordinates of P
  (h_Q : Q.1 = 3 ∧ Q.2 = m - 2)  -- x-coordinate of Q is 3, y-coordinate is m-2
  (h_sym : P.1 = -Q.1 ∧ P.2 = -Q.2)  -- P and Q are symmetric about the origin
  : m = -3 := by
  sorry

end symmetric_points_y_coordinate_l4064_406424


namespace inscribed_sphere_radius_l4064_406435

/-- A regular tetrahedron with unit edge length -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_eq_one : edge_length = 1

/-- A sphere touching three faces of a regular tetrahedron and three sides of its fourth face -/
structure InscribedSphere (t : RegularTetrahedron) where
  radius : ℝ
  touches_three_faces : True  -- Placeholder for the condition
  touches_three_sides_of_fourth_face : True  -- Placeholder for the condition

/-- The radius of the inscribed sphere is √6/8 -/
theorem inscribed_sphere_radius (t : RegularTetrahedron) (s : InscribedSphere t) :
  s.radius = Real.sqrt 6 / 8 := by
  sorry

end inscribed_sphere_radius_l4064_406435


namespace moon_arrangements_l4064_406484

def word : String := "MOON"

theorem moon_arrangements :
  (List.permutations (word.toList)).length = 12 :=
by
  sorry

end moon_arrangements_l4064_406484


namespace exists_non_prime_product_l4064_406480

/-- The k-th prime number -/
def nthPrime (k : ℕ) : ℕ := sorry

/-- The product of the first n prime numbers plus 1 -/
def primeProduct (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc * nthPrime (i + 1)) 1 + 1

/-- Theorem stating that there exists a number n such that primeProduct n is not prime -/
theorem exists_non_prime_product : ∃ n : ℕ, ¬ Nat.Prime (primeProduct n) := by sorry

end exists_non_prime_product_l4064_406480


namespace james_living_room_set_price_l4064_406428

/-- The final price James paid for the living room set after discount -/
theorem james_living_room_set_price (coach : ℝ) (sectional : ℝ) (other : ℝ) 
  (h1 : coach = 2500)
  (h2 : sectional = 3500)
  (h3 : other = 2000)
  (discount_rate : ℝ) 
  (h4 : discount_rate = 0.1) : 
  (coach + sectional + other) * (1 - discount_rate) = 7200 := by
  sorry

end james_living_room_set_price_l4064_406428


namespace quadratic_equation_root_and_sum_l4064_406470

theorem quadratic_equation_root_and_sum : 
  ∃ (a b c : ℚ), 
    (a = 1 ∧ b = 6 ∧ c = -4) ∧ 
    (a * (Real.sqrt 5 - 3)^2 + b * (Real.sqrt 5 - 3) + c = 0) ∧
    (∀ x y : ℝ, x^2 + 6*x - 4 = 0 ∧ y^2 + 6*y - 4 = 0 → x + y = -6) :=
by sorry

end quadratic_equation_root_and_sum_l4064_406470


namespace part1_part2_l4064_406403

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 6

-- Part 1
theorem part1 : ∀ x : ℝ, f 5 x < 0 ↔ -3 < x ∧ x < -2 := by sorry

-- Part 2
theorem part2 : ∀ a : ℝ, (∀ x : ℝ, f a x > 0) ↔ -2 * Real.sqrt 6 < a ∧ a < 2 * Real.sqrt 6 := by sorry

end part1_part2_l4064_406403


namespace inequality_proof_l4064_406450

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + a^2) ≥ Real.sqrt 2 * (a + b + c + d) := by
  sorry

end inequality_proof_l4064_406450


namespace solution_set_when_a_is_3_a_value_when_p_intersect_q_equals_q_l4064_406478

-- Define the solution sets P and Q
def P (a : ℝ) : Set ℝ := {x : ℝ | (x - a) / (x + 1) < 0}
def Q : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

-- Theorem 1: When a = 3, P = {x | -1 < x < 3}
theorem solution_set_when_a_is_3 : 
  P 3 = {x : ℝ | -1 < x ∧ x < 3} := by sorry

-- Theorem 2: When P ∩ Q = Q, a = 2
theorem a_value_when_p_intersect_q_equals_q : 
  (∃ a : ℝ, a > 0 ∧ P a ∩ Q = Q) → (∃ a : ℝ, a = 2 ∧ P a ∩ Q = Q) := by sorry

end solution_set_when_a_is_3_a_value_when_p_intersect_q_equals_q_l4064_406478


namespace first_replaced_man_age_l4064_406408

/-- The age of the first replaced man in a group scenario --/
def age_of_first_replaced_man (initial_count : ℕ) (age_increase : ℕ) (second_replaced_age : ℕ) (new_men_average_age : ℕ) : ℕ :=
  initial_count * age_increase + new_men_average_age * 2 - second_replaced_age - initial_count * age_increase

/-- Theorem stating the age of the first replaced man is 21 --/
theorem first_replaced_man_age :
  age_of_first_replaced_man 15 2 23 37 = 21 := by
  sorry

#eval age_of_first_replaced_man 15 2 23 37

end first_replaced_man_age_l4064_406408


namespace bicycle_cost_price_l4064_406426

theorem bicycle_cost_price 
  (profit_A_to_B : ℝ) 
  (profit_B_to_C : ℝ) 
  (price_C : ℝ) 
  (h1 : profit_A_to_B = 0.20)
  (h2 : profit_B_to_C = 0.25)
  (h3 : price_C = 225) :
  ∃ (cost_price_A : ℝ), 
    cost_price_A * (1 + profit_A_to_B) * (1 + profit_B_to_C) = price_C ∧ 
    cost_price_A = 150 := by
  sorry

end bicycle_cost_price_l4064_406426


namespace tan_sum_identity_l4064_406401

theorem tan_sum_identity : 
  (1 + Real.tan (23 * π / 180)) * (1 + Real.tan (22 * π / 180)) = 
  2 + Real.tan (23 * π / 180) * Real.tan (22 * π / 180) := by
  sorry

end tan_sum_identity_l4064_406401


namespace book_stack_thickness_l4064_406441

/-- Calculates the thickness of a stack of books in inches -/
def stack_thickness (num_books : ℕ) (pages_per_book : ℕ) (pages_per_inch : ℕ) : ℚ :=
  (num_books * pages_per_book : ℚ) / pages_per_inch

/-- Proves that the thickness of a stack of 6 books, each with 160 pages,
    is 12 inches when 80 pages make one inch of thickness -/
theorem book_stack_thickness :
  stack_thickness 6 160 80 = 12 := by
  sorry

end book_stack_thickness_l4064_406441


namespace bd_always_greater_than_10_l4064_406439

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right angle at C
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0 ∧
  -- Angle at B is 45°
  (C.1 - B.1) * (A.1 - B.1) + (C.2 - B.2) * (A.2 - B.2) = 
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) / Real.sqrt 2 ∧
  -- Length of AB is 20
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 400

-- Define a point P inside the triangle
def InsideTriangle (P A B C : ℝ × ℝ) : Prop :=
  ∃ (α β γ : ℝ), α ≥ 0 ∧ β ≥ 0 ∧ γ ≥ 0 ∧ α + β + γ = 1 ∧
  P = (α * A.1 + β * B.1 + γ * C.1, α * A.2 + β * B.2 + γ * C.2)

-- Define point D as the intersection of BP and AC
def IntersectionPoint (D B P A C : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), D = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2)) ∧
              ∃ (s : ℝ), D = (B.1 + s * (P.1 - B.1), B.2 + s * (P.2 - B.2))

-- Theorem statement
theorem bd_always_greater_than_10 
  (A B C P D: ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : InsideTriangle P A B C) 
  (h3 : IntersectionPoint D B P A C) : 
  (D.1 - B.1)^2 + (D.2 - B.2)^2 > 100 := by
  sorry

end bd_always_greater_than_10_l4064_406439


namespace ellipse_parabola_properties_l4064_406446

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a parabola with focal length c -/
structure Parabola where
  c : ℝ
  h_pos : 0 < c

/-- The configuration of the ellipse and parabola as described in the problem -/
structure Configuration where
  C₁ : Ellipse
  C₂ : Parabola
  h_focus : C₂.c = C₁.a - C₁.b  -- Right focus of C₁ coincides with focus of C₂
  h_center : C₁.a = 2 * C₂.c    -- Center of C₁ coincides with vertex of C₂
  h_chord_ratio : 3 * C₂.c = 2 * C₁.b^2 / C₁.a  -- |CD| = 4/3 |AB|
  h_vertex_sum : C₁.a + C₂.c = 6  -- Sum of distances from vertices to directrix is 12

/-- The main theorem stating the properties to be proved -/
theorem ellipse_parabola_properties (config : Configuration) :
  config.C₁.a = 4 ∧ 
  config.C₁.b^2 = 12 ∧ 
  config.C₂.c = 2 ∧
  (config.C₁.a - config.C₁.b) / config.C₁.a = 1/2 := by
  sorry

end ellipse_parabola_properties_l4064_406446


namespace max_value_of_function_l4064_406448

theorem max_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1/2) : 
  ∃ (max_y : ℝ), max_y = 1/8 ∧ ∀ y, y = x * (1 - 2*x) → y ≤ max_y :=
sorry

end max_value_of_function_l4064_406448


namespace annies_class_size_l4064_406405

theorem annies_class_size :
  ∀ (total_spent : ℚ) (candy_cost : ℚ) (candies_per_classmate : ℕ) (candies_left : ℕ),
    total_spent = 8 →
    candy_cost = 1/10 →
    candies_per_classmate = 2 →
    candies_left = 12 →
    (total_spent / candy_cost - candies_left) / candies_per_classmate = 34 := by
  sorry

end annies_class_size_l4064_406405


namespace simplify_fraction_l4064_406496

theorem simplify_fraction : (90 : ℚ) / 126 = 5 / 7 := by
  sorry

end simplify_fraction_l4064_406496


namespace counterexample_prime_plus_two_l4064_406442

theorem counterexample_prime_plus_two :
  ∃ n : ℕ, Nat.Prime n ∧ ¬(Nat.Prime (n + 2)) :=
sorry

end counterexample_prime_plus_two_l4064_406442


namespace parabola_vertex_coefficients_l4064_406490

/-- Prove that for a parabola y = ax² + bx with vertex at (3,3), the values of a and b are a = -1/3 and b = 2. -/
theorem parabola_vertex_coefficients (a b : ℝ) : 
  (∀ x, 3 = a * x^2 + b * x ↔ x = 3) ∧ (3 = a * 3^2 + b * 3) → 
  a = -1/3 ∧ b = 2 := by
sorry

end parabola_vertex_coefficients_l4064_406490


namespace motorcycle_trip_distance_l4064_406483

/-- Given a motorcycle trip from A to B to C with the following conditions:
  - The average speed for the entire trip is 25 miles per hour
  - The time from A to B is 3 times the time from B to C
  - The distance from B to C is half the distance from A to B
Prove that the distance from A to B is 100/3 miles -/
theorem motorcycle_trip_distance (average_speed : ℝ) (time_ratio : ℝ) (distance_ratio : ℝ) :
  average_speed = 25 →
  time_ratio = 3 →
  distance_ratio = 1/2 →
  ∃ (distance_AB : ℝ), distance_AB = 100/3 :=
sorry

end motorcycle_trip_distance_l4064_406483


namespace task_completion_time_l4064_406423

/-- The time taken to complete a task when two people work together, with one person stopping early. -/
def completionTime (john_rate : ℚ) (jane_rate : ℚ) (early_stop : ℕ) : ℚ :=
  let combined_rate := john_rate + jane_rate
  let x := (1 - john_rate * early_stop) / combined_rate
  x + early_stop

theorem task_completion_time :
  let john_rate : ℚ := 1 / 20
  let jane_rate : ℚ := 1 / 12
  let early_stop : ℕ := 4
  completionTime john_rate jane_rate early_stop = 10 := by
  sorry

#eval completionTime (1 / 20 : ℚ) (1 / 12 : ℚ) 4

end task_completion_time_l4064_406423


namespace candy_bar_sales_l4064_406406

theorem candy_bar_sales (seth_sales : ℕ) (other_sales : ℕ) 
  (h1 : seth_sales = 3 * other_sales + 6) 
  (h2 : seth_sales = 78) : 
  other_sales = 24 := by
sorry

end candy_bar_sales_l4064_406406


namespace union_of_sets_l4064_406495

theorem union_of_sets : 
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {2, 3, 4}
  M ∪ N = {1, 2, 3, 4} := by
  sorry

end union_of_sets_l4064_406495


namespace catch_up_distance_l4064_406425

/-- Proves that B catches up with A 200 km from the start given the specified conditions -/
theorem catch_up_distance (a_speed b_speed : ℝ) (time_diff : ℝ) (catch_up_dist : ℝ) : 
  a_speed = 10 →
  b_speed = 20 →
  time_diff = 10 →
  catch_up_dist = 200 →
  catch_up_dist = b_speed * (time_diff + catch_up_dist / b_speed) :=
by sorry

end catch_up_distance_l4064_406425


namespace polynomial_at_most_one_zero_l4064_406429

theorem polynomial_at_most_one_zero (n : ℤ) :
  ∃! (r : ℝ), r^4 - 1994*r^3 + (1993 + n : ℝ)*r^2 - 11*r + (n : ℝ) = 0 :=
sorry

end polynomial_at_most_one_zero_l4064_406429


namespace rotation_result_l4064_406422

-- Define the shapes
inductive Shape
| Triangle
| SmallCircle
| Pentagon

-- Define the positions
inductive Position
| Top
| LowerLeft
| LowerRight

-- Define the configuration as a function from Shape to Position
def Configuration := Shape → Position

-- Define the initial configuration
def initial_config : Configuration
| Shape.Triangle => Position.Top
| Shape.SmallCircle => Position.LowerLeft
| Shape.Pentagon => Position.LowerRight

-- Define the rotation function
def rotate_150_clockwise (config : Configuration) : Configuration :=
  fun shape =>
    match config shape with
    | Position.Top => Position.LowerRight
    | Position.LowerLeft => Position.Top
    | Position.LowerRight => Position.LowerLeft

-- Theorem statement
theorem rotation_result :
  let final_config := rotate_150_clockwise initial_config
  final_config Shape.Triangle = Position.LowerRight ∧
  final_config Shape.SmallCircle = Position.Top ∧
  final_config Shape.Pentagon = Position.LowerLeft :=
sorry

end rotation_result_l4064_406422


namespace student_tickets_sold_l4064_406473

theorem student_tickets_sold (adult_price student_price : ℚ)
  (total_tickets : ℕ) (total_amount : ℚ)
  (h1 : adult_price = 4)
  (h2 : student_price = 5/2)
  (h3 : total_tickets = 59)
  (h4 : total_amount = 445/2) :
  ∃ (student_tickets : ℕ),
    student_tickets = 9 ∧
    student_tickets ≤ total_tickets ∧
    ∃ (adult_tickets : ℕ),
      adult_tickets + student_tickets = total_tickets ∧
      adult_price * adult_tickets + student_price * student_tickets = total_amount :=
by sorry

end student_tickets_sold_l4064_406473


namespace subway_length_l4064_406431

/-- Calculates the length of a subway given its speed, distance between stations, and time to pass a station. -/
theorem subway_length
  (speed : ℝ)                  -- Speed of the subway in km/min
  (station_distance : ℝ)       -- Distance between stations in km
  (passing_time : ℝ)           -- Time to pass the station in minutes
  (h1 : speed = 1.6)           -- Given speed
  (h2 : station_distance = 4.85) -- Given distance between stations
  (h3 : passing_time = 3.25)   -- Given time to pass the station
  : (speed * passing_time - station_distance) * 1000 = 350 :=
by sorry

end subway_length_l4064_406431


namespace max_area_difference_line_l4064_406449

/-- The line that maximizes the area difference when passing through P(1,1) and dividing the circle (x-2)^2+y^2 ≤ 4 -/
theorem max_area_difference_line (x y : ℝ) : 
  (∀ (a b : ℝ), (a - 2)^2 + b^2 ≤ 4 → 
    (x - y = 0 → 
      ∀ (m n : ℝ), (m + n - 2 = 0 ∨ y - 1 = 0 ∨ m + 3*n - 4 = 0) → 
        (abs ((a - 2)^2 + b^2 - ((a - x)^2 + (b - y)^2)) ≤ 
         abs ((a - 2)^2 + b^2 - ((a - m)^2 + (b - n)^2))))) :=
by sorry

end max_area_difference_line_l4064_406449


namespace non_negative_product_l4064_406402

theorem non_negative_product (a b c d e f g h : ℝ) :
  (max (ac + bd) (max (ae + bf) (max (ag + bh) (max (ce + df) (max (cg + dh) (eg + fh)))))) ≥ 0 := by
  sorry

end non_negative_product_l4064_406402


namespace log_xyz_value_l4064_406474

-- Define the variables
variable (x y z : ℝ)
variable (log : ℝ → ℝ)

-- State the theorem
theorem log_xyz_value (h1 : log (x * y^3 * z) = 2) (h2 : log (x^2 * y * z^2) = 3) :
  log (x * y * z) = 8/5 := by
  sorry

end log_xyz_value_l4064_406474


namespace dog_weight_gain_exists_l4064_406414

/-- Represents a dog with age and weight -/
structure Dog where
  age : ℕ
  weight : ℝ

/-- Represents the annual weight gain of a dog -/
def annualGain (d : Dog) (gain : ℝ) : Prop :=
  ∃ (initialWeight : ℝ), initialWeight + gain * (d.age - 1) = d.weight

/-- Theorem stating that for any dog, there exists some annual weight gain -/
theorem dog_weight_gain_exists (d : Dog) : ∃ (gain : ℝ), annualGain d gain :=
sorry

end dog_weight_gain_exists_l4064_406414


namespace least_sum_of_exponents_l4064_406479

theorem least_sum_of_exponents (h : ℕ+) (a b c d e : ℕ) 
  (h_div_225 : 225 ∣ h) (h_div_216 : 216 ∣ h) (h_div_847 : 847 ∣ h)
  (h_factorization : h = 2^a * 3^b * 5^c * 7^d * 11^e) :
  ∃ (a' b' c' d' e' : ℕ), 
    h = 2^a' * 3^b' * 5^c' * 7^d' * 11^e' ∧
    a' + b' + c' + d' + e' ≤ a + b + c + d + e ∧
    a' + b' + c' + d' + e' = 10 :=
sorry

end least_sum_of_exponents_l4064_406479


namespace softball_team_ratio_l4064_406443

theorem softball_team_ratio : 
  ∀ (men women : ℕ), 
  women = men + 6 →
  men + women = 24 →
  (men : ℚ) / women = 3 / 5 := by
sorry

end softball_team_ratio_l4064_406443


namespace second_group_frequency_l4064_406475

theorem second_group_frequency (total : ℕ) (group1 group2 group3 group4 group5 : ℕ) 
  (h1 : total = 50)
  (h2 : group1 = 2)
  (h3 : group3 = 8)
  (h4 : group4 = 10)
  (h5 : group5 = 20)
  (h6 : total = group1 + group2 + group3 + group4 + group5) :
  group2 = 10 := by
  sorry

end second_group_frequency_l4064_406475


namespace sqrt_sum_equals_seventeen_sixths_l4064_406494

theorem sqrt_sum_equals_seventeen_sixths : 
  Real.sqrt (9 / 4) + Real.sqrt (16 / 9) = 17 / 6 := by
  sorry

end sqrt_sum_equals_seventeen_sixths_l4064_406494


namespace infinitely_many_triplets_sum_of_squares_l4064_406411

theorem infinitely_many_triplets_sum_of_squares :
  ∃ f : ℕ → ℤ, ∀ k : ℕ,
    (∃ a b : ℤ, f k = a^2 + b^2) ∧
    (∃ c d : ℤ, f k + 1 = c^2 + d^2) ∧
    (∃ e g : ℤ, f k + 2 = e^2 + g^2) :=
by sorry

end infinitely_many_triplets_sum_of_squares_l4064_406411


namespace problem_sequence_sum_largest_fib_is_196418_l4064_406438

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The sequence in the problem -/
def problem_sequence : List ℤ :=
  [2, -3, -5, 8, 13, -21, -34, 55, 89, -144, -233, 377, 46368, -75025, -121393, 196418]

/-- The sum of the problem sequence -/
def sequence_sum : ℤ := problem_sequence.sum

/-- Theorem stating that the sum of the problem sequence equals 196418 -/
theorem problem_sequence_sum : sequence_sum = 196418 := by
  sorry

/-- The largest Fibonacci number in the sequence -/
def largest_fib : ℕ := 196418

/-- Theorem stating that the largest Fibonacci number in the sequence is 196418 -/
theorem largest_fib_is_196418 : fib 27 = largest_fib := by
  sorry

end problem_sequence_sum_largest_fib_is_196418_l4064_406438


namespace divide_books_into_portions_l4064_406471

theorem divide_books_into_portions (n : ℕ) (k : ℕ) : n = 6 → k = 3 → 
  (Nat.choose n 2 * Nat.choose (n - 2) 2) / Nat.factorial k = 15 := by
  sorry

end divide_books_into_portions_l4064_406471


namespace sum_of_first_20_a_l4064_406404

def odd_number (n : ℕ) : ℕ := 2 * n - 1

def a (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ k => odd_number (n * (n - 1) + 1 + k))

theorem sum_of_first_20_a : Finset.sum (Finset.range 20) (λ i => a (i + 1)) = 44100 := by
  sorry

end sum_of_first_20_a_l4064_406404


namespace max_value_abc_l4064_406499

theorem max_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  a^4 * b^2 * c ≤ 1024/117649 := by sorry

end max_value_abc_l4064_406499
