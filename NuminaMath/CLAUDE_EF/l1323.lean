import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mustache_area_l1323_132339

open Real Set Interval

-- Define the lower and upper bounds of the mustache
noncomputable def lower_bound (x : ℝ) : ℝ := 4 + 4 * cos (π * x / 24)
noncomputable def upper_bound (x : ℝ) : ℝ := 6 + 6 * cos (π * x / 24)

-- Define the interval
def interval : Set ℝ := Icc (-24) 24

-- Theorem statement
theorem mustache_area : 
  ∫ x in interval, (upper_bound x - lower_bound x) = 96 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mustache_area_l1323_132339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_star_operation_l1323_132397

/-- The star operation for real numbers -/
noncomputable def star (x y : ℝ) : ℝ := (x + y) / (x - y)

/-- Theorem stating the result of the nested star operation -/
theorem nested_star_operation :
  star (star (-2) (1/2)) (-3/4) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_star_operation_l1323_132397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l1323_132380

noncomputable section

open Real

theorem angle_values (θ : ℝ) (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x = -sqrt 3 ∧ y = m) →
  sin θ = (sqrt 2 / 4) * m →
  ((cos θ = -1 ∧ tan θ = 0) ∨
   (cos θ = -sqrt 6 / 4 ∧ tan θ = -sqrt 15 / 3) ∨
   (cos θ = -sqrt 6 / 4 ∧ tan θ = sqrt 15 / 3)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l1323_132380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1323_132392

noncomputable section

open Real

variable (f : ℝ → ℝ)

axiom f_domain : ∀ x : ℝ, x > 0 → f x ≠ 0
axiom f_pos : ∀ x : ℝ, x > 1 → f x > 0
axiom f_add : ∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_third : f (1/3) = -1

theorem f_properties :
  (∀ x : ℝ, x > 0 → f (1/x) = -f x) ∧
  (StrictMono f) ∧
  ({x : ℝ | x > 0 ∧ f x - f (1/(x-2)) ≥ 2} = {x : ℝ | x ≥ 1 + sqrt 10}) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1323_132392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_eq_four_ninths_l1323_132352

/-- The sum of the infinite series ∑(k/(4^k)) for k from 1 to infinity -/
noncomputable def infiniteSeries : ℝ := ∑' k, k / (4 : ℝ) ^ k

/-- Theorem stating that the infinite series sums to 4/9 -/
theorem infiniteSeries_eq_four_ninths : infiniteSeries = 4 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_eq_four_ninths_l1323_132352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tulip_count_l1323_132362

theorem tulip_count (total_flowers : ℕ) (rose_ratio : ℚ) (daisy_count : ℕ) : 
  total_flowers = 102 →
  rose_ratio = 5 / 6 →
  daisy_count = 6 →
  ∃ tulip_count : ℕ, 
    tulip_count = 16 ∧ 
    tulip_count + daisy_count + (rose_ratio * (total_flowers - daisy_count : ℚ)).floor = total_flowers :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tulip_count_l1323_132362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_leg_l1323_132366

/-- Represents a 45-45-90 triangle -/
structure RightIsoscelesTriangle where
  leg : ℝ
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = leg * Real.sqrt 2

/-- A series of four 45-45-90 triangles where the hypotenuse of each smaller triangle
    is the leg of the adjacent larger triangle -/
def TriangleSeries : Fin 4 → RightIsoscelesTriangle :=
  sorry -- Placeholder definition

/-- The hypotenuse of the largest triangle is 16 cm -/
axiom largest_triangle_hypotenuse : (TriangleSeries 0).hypotenuse = 16

/-- Each smaller triangle's hypotenuse is the leg of the adjacent larger triangle -/
axiom adjacent_triangles (i : Fin 3) : 
  (TriangleSeries i.succ).hypotenuse = (TriangleSeries i).leg

/-- The leg of the smallest triangle is 4 cm -/
theorem smallest_triangle_leg : (TriangleSeries 3).leg = 4 := by
  sorry

#check smallest_triangle_leg

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_leg_l1323_132366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1323_132321

-- Define the functions h and j
def h (x : ℝ) : ℝ := x^2 + 9
def j (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem solution_exists (b : ℝ) (hb : b > 0) (heq : h (j b) = 15) : 
  b = Real.sqrt (Real.sqrt 6 - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1323_132321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_abs_d_l1323_132326

/-- Given a polynomial equation a(3+i)^4 + b(3+i)^3 + c(3+i)^2 + d(3+i) + a = 0,
    where a, b, c, d are integers with gcd 1, prove that |d| = 66 -/
theorem polynomial_root_abs_d (a b c d : ℤ) : 
  (a * (Complex.I + 3)^4 + b * (Complex.I + 3)^3 + c * (Complex.I + 3)^2 + d * (Complex.I + 3) + a = 0) →
  (Int.gcd a b = 1 ∧ Int.gcd b c = 1 ∧ Int.gcd c d = 1) →
  Int.natAbs d = 66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_abs_d_l1323_132326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1323_132314

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    one focus at (c, 0), and one endpoint of the imaginary axis at (0, b),
    if the line connecting these points is perpendicular to the asymptote y = (b/a)x,
    then the eccentricity of the hyperbola is (1 + √5) / 2. -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (c^2 = a^2 + b^2) →  -- Relation between c, a, and b for a hyperbola
  ((b / c) * (b / a) = 1) →  -- Perpendicularity condition
  c / a = (1 + Real.sqrt 5) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1323_132314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_factorial_base_18_zeros_l1323_132338

/-- The number of zeros at the end of n! when expressed in base b -/
def trailing_zeros (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- 15 factorial -/
def factorial_15 : ℕ := Nat.factorial 15

/-- Base 18 -/
def base_18 : ℕ := 18

theorem fifteen_factorial_base_18_zeros :
  trailing_zeros factorial_15 base_18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_factorial_base_18_zeros_l1323_132338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1323_132390

-- Define the hyperbola
noncomputable def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 81 = 1

-- Define the asymptote slope
noncomputable def asymptote_slope : ℝ := 3/4

-- Define the vertex x-coordinate
noncomputable def vertex_x : ℝ := 12

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y : ℝ, hyperbola x y → 
    (y = asymptote_slope * x ∨ y = -asymptote_slope * x)) ∧
  (∃ y : ℝ, hyperbola vertex_x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1323_132390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_C_largest_area_l1323_132364

-- Define the basic shapes
noncomputable def triangle_area : ℝ := 1/2 * 1 * 1
noncomputable def rectangle_area : ℝ := 2 * 1

-- Define the areas of each figure
noncomputable def figure_A_area : ℝ := 4 * triangle_area + rectangle_area
noncomputable def figure_B_area : ℝ := 2 * triangle_area + 2 * rectangle_area
noncomputable def figure_C_area : ℝ := 3 * rectangle_area
noncomputable def figure_D_area : ℝ := 5 * triangle_area

-- Theorem statement
theorem figure_C_largest_area :
  figure_C_area > figure_A_area ∧
  figure_C_area > figure_B_area ∧
  figure_C_area > figure_D_area := by
  -- Expand definitions
  unfold figure_C_area figure_A_area figure_B_area figure_D_area
  unfold triangle_area rectangle_area
  -- Simplify expressions
  simp
  -- Split into three conjuncts
  apply And.intro
  · -- Prove figure_C_area > figure_A_area
    norm_num
  · apply And.intro
    · -- Prove figure_C_area > figure_B_area
      norm_num
    · -- Prove figure_C_area > figure_D_area
      norm_num

#check figure_C_largest_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_C_largest_area_l1323_132364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l1323_132342

/-- Given square and rectangular paper pieces, prove the perimeter of a quadrilateral formed from them -/
theorem quadrilateral_perimeter (a b : ℝ) : 
  (2 * a^2 + 7 * b^2 + 3 * a * b = (a + 3*b) * (2*a + b)) → 
  (2 * ((a + 3*b) + (2*a + b)) = 6*a + 8*b) :=
by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l1323_132342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_relationship_l1323_132341

theorem number_relationship : 
  let a : ℝ := (0.31 : ℝ)^2
  let b : ℝ := Real.log 0.31 / Real.log 2
  let c : ℝ := (2 : ℝ)^(0.31 : ℝ)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_relationship_l1323_132341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_circle_l1323_132310

/-- The curve defined by the polar equation r = 3 sin θ cos θ -/
noncomputable def polar_curve (θ : ℝ) : ℝ × ℝ :=
  let r := 3 * Real.sin θ * Real.cos θ
  (r * Real.cos θ, r * Real.sin θ)

/-- The Cartesian equation of the curve -/
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 + (y - 3/2)^2 = 9/4

theorem polar_curve_is_circle :
  ∀ (x y : ℝ), (∃ θ, polar_curve θ = (x, y)) ↔ cartesian_equation x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_circle_l1323_132310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_from_rectangle_l1323_132384

theorem square_perimeter_from_rectangle (x y : ℝ) (h : x - y = 5) :
  4 * Real.sqrt (5 * x * y) = 4 * Real.sqrt (5 * x * y) := by
  -- Definitions
  let rectangle_area := x * y
  let square_area := 5 * rectangle_area
  let square_side := Real.sqrt square_area
  
  -- Proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_from_rectangle_l1323_132384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1323_132357

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - Real.sqrt (7 - Real.sqrt (x^2)))

theorem domain_of_f :
  ∀ x : ℝ, f x ∈ Set.Icc 0 (Real.sqrt 4) ↔ x ∈ Set.Icc (-7 : ℝ) 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1323_132357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l1323_132358

theorem trig_problem (α β : Real) 
  (h1 : Real.cos α = 1/7)
  (h2 : Real.cos (α - β) = 13/14)
  (h3 : 0 < β) (h4 : β < α) (h5 : α < π/2) :
  Real.tan (2*α) = -8*Real.sqrt 3/47 ∧ Real.cos β = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l1323_132358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_covered_by_five_strips_l1323_132379

/-- Represents a rectangular strip -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a single strip -/
def stripArea (s : Strip) : ℝ := s.length * s.width

/-- Represents the configuration of overlapping strips -/
structure StripConfiguration where
  strips : List Strip
  numIntersections : ℕ
  overlapAreaPerIntersection : ℝ

/-- Calculates the total area covered by the strips -/
def totalAreaCovered (config : StripConfiguration) : ℝ :=
  let totalStripArea := config.strips.map stripArea |>.sum
  let totalOverlapArea := (config.numIntersections : ℝ) * config.overlapAreaPerIntersection
  totalStripArea - totalOverlapArea

/-- The main theorem statement -/
theorem area_covered_by_five_strips :
  let strip := Strip.mk 12 2
  let config := StripConfiguration.mk (List.replicate 5 strip) 10 4
  totalAreaCovered config = 80 := by
  sorry

#eval totalAreaCovered (StripConfiguration.mk (List.replicate 5 (Strip.mk 12 2)) 10 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_covered_by_five_strips_l1323_132379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_S_union_T_equals_interval_l1323_132329

-- Define the sets S and T
def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

-- State the theorem
theorem complement_S_union_T_equals_interval :
  (Set.compl S) ∪ T = Set.Iic (1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_S_union_T_equals_interval_l1323_132329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_product_max_value_l1323_132300

theorem trigonometric_product_max_value :
  ∀ x y z : ℝ,
  (Real.sin (3 * x) + Real.sin (2 * y) + Real.sin z) * (Real.cos (3 * x) + Real.cos (2 * y) + Real.cos z) ≤ 9/2 ∧
  ∃ x y z : ℝ, (Real.sin (3 * x) + Real.sin (2 * y) + Real.sin z) * (Real.cos (3 * x) + Real.cos (2 * y) + Real.cos z) = 9/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_product_max_value_l1323_132300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_target_state_l1323_132350

/-- Represents the money state of the three players -/
structure GameState where
  john : ℕ
  emma : ℕ
  lucas : ℕ

/-- The initial state of the game -/
def initial_state : GameState :=
  ⟨3, 2, 1⟩

/-- Represents a single round of the game -/
def game_round (state : GameState) : GameState :=
  sorry

/-- Probability of transitioning from one state to another in a single round -/
def transition_probability (from_state to_state : GameState) : ℚ :=
  sorry

/-- The target state we want to reach -/
def target_state : GameState :=
  ⟨2, 2, 2⟩

/-- The theorem to be proved -/
theorem probability_of_target_state :
  ∃ (n : ℕ), n > 1000 →
    transition_probability (game_round^[n] initial_state) target_state = 1/27 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_target_state_l1323_132350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_six_fifths_l1323_132309

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic polynomial at a given point -/
noncomputable def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The sum of the roots of a quadratic polynomial -/
noncomputable def QuadraticPolynomial.sumOfRoots (p : QuadraticPolynomial) : ℝ :=
  -p.b / p.a

/-- The main theorem -/
theorem sum_of_roots_is_six_fifths (p : QuadraticPolynomial) 
  (h : ∀ x : ℝ, p.eval (2*x^5 + 3*x) ≥ p.eval (3*x^4 + 2*x^2 + 1)) : 
  p.sumOfRoots = 6/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_six_fifths_l1323_132309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_four_digit_number_l1323_132363

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧ 
  ∃ (a b c d : ℕ), n = 1000 * a + 100 * b + 10 * c + d ∧
  ({a, b, c, d} : Finset ℕ) = {1, 5, 9, 4}

theorem largest_four_digit_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 9541 :=
by
  intro n h
  sorry

#check largest_four_digit_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_four_digit_number_l1323_132363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l1323_132368

/-- An arithmetic sequence starting with 1 -/
def arithmetic_seq (d : ℤ) : ℕ → ℤ
  | 0 => 1
  | n + 1 => arithmetic_seq d n + d

/-- A geometric sequence starting with 1 -/
def geometric_seq (r : ℤ) : ℕ → ℤ
  | 0 => 1
  | n + 1 => geometric_seq r n * r

/-- The sum of corresponding terms in arithmetic and geometric sequences -/
def c_seq (d r : ℤ) (n : ℕ) : ℤ := arithmetic_seq d n + geometric_seq r n

theorem c_k_value (d r : ℤ) : 
  (∃ k : ℕ, k > 2 ∧ 
    c_seq d r (k - 1) = 50 ∧ 
    c_seq d r (k + 1) = 1500 ∧ 
    d < 0 ∧ r > 1) → 
  ∃ k : ℕ, c_seq d r k = 2406 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l1323_132368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_in_second_quadrant_l1323_132367

theorem cosine_of_angle_in_second_quadrant (α : ℝ) :
  (π / 2 < α ∧ α < π) →  -- α is in the second quadrant
  Real.sin α = 5 / 13 →
  Real.cos α = -12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_in_second_quadrant_l1323_132367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_b_equals_two_l1323_132336

theorem sum_a_b_equals_two (a b : ℝ) 
  (hM : Set.Subset {b/a, 1} (Set.univ : Set ℝ))
  (hN : Set.Subset {a, 0} (Set.univ : Set ℝ))
  (hf : ∀ x ∈ ({b/a, 1} : Set ℝ), 2*x ∈ ({a, 0} : Set ℝ)) : 
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_b_equals_two_l1323_132336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_scores_b_l1323_132374

/-- Represents the score of a participant in the math test -/
def Score := Nat

/-- Represents the number of questions in the test -/
def NumQuestions : Nat := 10

/-- Represents the points awarded for a correct answer -/
def PointsPerCorrectAnswer : Nat := 3

/-- Represents the number of questions where A and B have different answers -/
def DifferentAnswers : Nat := 2

/-- Represents the sum of A and B's scores -/
def TotalScore : Nat := 54

/-- The set of possible scores for participant B -/
def PossibleScoresB : Set Nat := {24, 27, 30}

/-- Theorem stating that given the conditions of the problem, 
    the set of possible scores for B is {24, 27, 30} -/
theorem possible_scores_b : 
  ∀ (score_a score_b : Nat),
    score_a + score_b = TotalScore →
    score_a ≤ NumQuestions * PointsPerCorrectAnswer →
    score_b ≤ NumQuestions * PointsPerCorrectAnswer →
    (∃ (correct_a correct_b : Nat),
      correct_a + correct_b = 2 * NumQuestions - DifferentAnswers ∧
      score_a = correct_a * PointsPerCorrectAnswer ∧
      score_b = correct_b * PointsPerCorrectAnswer) →
    score_b ∈ PossibleScoresB := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_scores_b_l1323_132374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1323_132312

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^k * Real.sin x

-- State the theorem
theorem range_of_f (k : ℝ) (hk : k > 0) :
  Set.range (fun x => f k x) ∩ Set.Icc 0 (2 * Real.pi) =
  Set.Icc (-(3 * Real.pi / 2)^k) ((Real.pi / 2)^k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1323_132312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_speed_to_arrive_on_time_l1323_132348

/-- Represents the travel scenario for Mr. Earl E. Bird -/
structure TravelScenario where
  distance : ℝ  -- Distance to work
  ideal_time : ℝ  -- Ideal travel time

/-- Calculates the arrival time difference given a speed -/
noncomputable def arrival_time_diff (scenario : TravelScenario) (speed : ℝ) : ℝ :=
  scenario.distance / speed - scenario.ideal_time

/-- Theorem stating the correct speed to arrive on time -/
theorem correct_speed_to_arrive_on_time (scenario : TravelScenario) 
  (h1 : arrival_time_diff scenario 50 = 1/12)  -- 5 minutes late at 50 mph
  (h2 : arrival_time_diff scenario 70 = -1/12) -- 5 minutes early at 70 mph
  : ∃ (speed : ℝ), 57.5 < speed ∧ speed < 58.5 ∧ arrival_time_diff scenario speed = 0 := by
  sorry

#check correct_speed_to_arrive_on_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_speed_to_arrive_on_time_l1323_132348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_A_optimal_price_l1323_132340

/-- Represents the selling price of toy A in yuan -/
def selling_price : ℝ → ℝ := λ x => x

/-- Represents the number of units sold per day as a function of the selling price -/
def units_sold (x : ℝ) : ℝ := 1800 - 20 * x

/-- Represents the profit per unit as a function of the selling price -/
def profit_per_unit (x : ℝ) : ℝ := x - 60

/-- Represents the total profit per day as a function of the selling price -/
def total_profit (x : ℝ) : ℝ := (profit_per_unit x) * (units_sold x)

/-- The purchase price of toy A in yuan -/
def purchase_price : ℝ := 60

/-- The maximum allowed profit margin -/
def max_profit_margin : ℝ := 0.4

theorem toy_A_optimal_price :
  ∃ (x : ℝ),
    x > purchase_price ∧
    x ≤ purchase_price * (1 + max_profit_margin) ∧
    total_profit x = 2500 ∧
    x = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_A_optimal_price_l1323_132340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_199_l1323_132386

/-- An arithmetic sequence with given first three terms -/
def arithmeticSequence (a₁ a₂ a₃ : ℝ) : ℕ → ℝ
  | 0 => a₁  -- Adding case for 0
  | 1 => a₁
  | 2 => a₂
  | 3 => a₃
  | n + 4 => arithmeticSequence a₁ a₂ a₃ (n + 3) + (a₂ - a₁)

/-- The 15th term of the arithmetic sequence with first three terms 3, 17, and 31 is 199 -/
theorem fifteenth_term_is_199 :
  arithmeticSequence 3 17 31 15 = 199 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_199_l1323_132386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_vertical_asymptote_l1323_132383

noncomputable section

-- Define the rational function g(x)
def g (k : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + k) / (x^2 + 2*x - 8)

-- Define what it means for a function to have a vertical asymptote at a point
def has_vertical_asymptote (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - a| ∧ |x - a| < δ → |f x| > 1/ε)

-- Define the property of having exactly one vertical asymptote
def has_exactly_one_vertical_asymptote (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, has_vertical_asymptote f a ∧
    ∀ b : ℝ, has_vertical_asymptote f b → b = a

-- State the theorem
theorem g_one_vertical_asymptote (k : ℝ) :
  has_exactly_one_vertical_asymptote (g k) ↔ k = 0 ∨ k = -24 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_vertical_asymptote_l1323_132383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tulip_fraction_l1323_132319

-- Define the total number of flowers
variable (total : ℕ)

-- Define the number of yellow flowers
def yellow (total : ℕ) : ℕ := (4 * total) / 5

-- Define the number of blue flowers
def blue (total : ℕ) : ℕ := total - yellow total

-- Define the number of yellow tulips
def yellow_tulips (total : ℕ) : ℕ := yellow total / 2

-- Define the number of blue tulips
def blue_tulips (total : ℕ) : ℕ := (2 * blue total) / 3

-- Theorem to prove
theorem tulip_fraction (total : ℕ) (h : total > 0) :
  (yellow_tulips total + blue_tulips total : ℚ) / total = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tulip_fraction_l1323_132319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_other_vertices_y_sum_l1323_132345

/-- Predicate to check if a set of points forms a rectangle -/
def IsRectangle (s : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if two points are opposite vertices of a rectangle -/
def AreOppositeVertices (s : Set (ℝ × ℝ)) (a b : ℝ × ℝ) : Prop :=
  sorry

/-- Given a rectangle with two opposite vertices at (2, 10) and (-6, -6),
    the sum of the y-coordinates of the other two vertices is 4. -/
theorem rectangle_other_vertices_y_sum :
  ∀ (rect : Set (ℝ × ℝ)) (a b : ℝ × ℝ),
  IsRectangle rect →
  a ∈ rect →
  b ∈ rect →
  AreOppositeVertices rect a b →
  a = (2, 10) →
  b = (-6, -6) →
  ∃ (c d : ℝ × ℝ), c ∈ rect ∧ d ∈ rect ∧ c.2 + d.2 = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_other_vertices_y_sum_l1323_132345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_point_of_f_l1323_132306

noncomputable def f (z : ℂ) : ℂ := ((1 - Complex.I) * z + (4 - 6 * Complex.I)) / 2

theorem rotation_point_of_f :
  let c : ℂ := 5 - 5 * Complex.I
  f c = c := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the complex arithmetic
  simp [Complex.I, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im]
  -- The proof is complete, but we'll use sorry to skip the details
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_point_of_f_l1323_132306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_implies_line_equation_line_equation_implies_chord_length_l1323_132303

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/2 + y^2/4 = 1

-- Define a line
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def is_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Theorem 1
theorem midpoint_implies_line_equation (A B M : Point) :
  is_on_ellipse A.x A.y →
  is_on_ellipse B.x B.y →
  M.x = (A.x + B.x) / 2 →
  M.y = (A.y + B.y) / 2 →
  M.x = 1 →
  M.y = 1 →
  ∃ l : Line, l.slope = -2 ∧ l.y_intercept = 3 ∧ 
    is_on_line A l ∧ is_on_line B l :=
by sorry

-- Theorem 2
theorem line_equation_implies_chord_length (A B : Point) (l : Line) :
  is_on_ellipse A.x A.y →
  is_on_ellipse B.x B.y →
  is_on_line A l →
  is_on_line B l →
  l.slope = 1 →
  l.y_intercept = 2 →
  distance A B = 4 * Real.sqrt 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_implies_line_equation_line_equation_implies_chord_length_l1323_132303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1323_132304

noncomputable def f (x : ℝ) := Real.sin (2*x + Real.pi/3) + Real.cos (2*x + Real.pi/6) + 2 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∃ x₀ ∈ Set.Icc 0 (Real.pi/2), ∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≤ f x₀) ∧
  (∃ x₁ ∈ Set.Icc 0 (Real.pi/2), ∀ x ∈ Set.Icc 0 (Real.pi/2), f x₁ ≤ f x) ∧
  (∃ x₀ ∈ Set.Icc 0 (Real.pi/2), f x₀ = 2) ∧
  (∃ x₁ ∈ Set.Icc 0 (Real.pi/2), f x₁ = -Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1323_132304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_l1323_132343

/-- Represents a sequence of digits --/
def Digits := List Nat

/-- The operation of selecting two adjacent digits, decrementing each by 1, and swapping their positions --/
def transform (d : Digits) : Digits → Prop :=
  sorry

/-- The relation that one digit sequence can be obtained from another through multiple applications of the transform operation --/
def reachable (start finish : Digits) : Prop :=
  sorry

/-- The initial number --/
def initial : Digits := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- The smallest possible number --/
def smallest : Digits := [1, 0, 1, 0, 1, 0, 1, 0, 1]

/-- Define a custom less than or equal to relation for Digits --/
def le_digits (d1 d2 : Digits) : Prop :=
  d1.length = d2.length ∧ (d1.zip d2).all (fun (x, y) => x ≤ y)

/-- Theorem stating that the smallest number is reachable from the initial number and that no smaller number is reachable --/
theorem smallest_number : 
  reachable initial smallest ∧ 
  ∀ (d : Digits), reachable initial d → le_digits smallest d :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_l1323_132343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_2n_gon_projection_l1323_132322

/-- A regular polygon with 2n sides -/
structure Regular2nGon (n : ℕ) where
  vertices : Fin (2*n) → ℝ × ℝ
  is_regular : Sorry

/-- A polyhedron in 3D space -/
structure Polyhedron where
  vertices : List (ℝ × ℝ × ℝ)
  faces : List (List ℕ)

/-- A projection from 3D to 2D -/
def project (p : ℝ × ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2.1)

/-- The theorem to be proved -/
theorem regular_2n_gon_projection (n : ℕ) :
  ∃ (poly : Polyhedron),
    (List.length poly.faces ≤ n + 2) ∧
    (∃ (g : Regular2nGon n),
      ∀ v, v ∈ (List.ofFn g.vertices) →
        ∃ p ∈ poly.vertices, project p = v) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_2n_gon_projection_l1323_132322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_coordinate_l1323_132315

noncomputable def r (θ : ℝ) : ℝ := Real.cos (2 * θ)

noncomputable def x (θ : ℝ) : ℝ := r θ * Real.cos θ

theorem max_x_coordinate :
  ∃ (max_x : ℝ), (∀ θ, x θ ≤ max_x) ∧ (max_x = 4 * Real.sqrt 6 / 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_coordinate_l1323_132315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l1323_132394

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

/-- The foci of the hyperbola -/
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

/-- Distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The main theorem -/
theorem hyperbola_triangle_area 
  (P : ℝ × ℝ)
  (h_on_hyperbola : hyperbola P.1 P.2)
  (h_distance_ratio : 3 * distance P F₁ = 4 * distance P F₂) :
  1/2 * distance P F₁ * distance P F₂ * Real.sqrt (1 - ((distance P F₁)^2 + (distance P F₂)^2 - (distance F₁ F₂)^2)^2 / (4 * (distance P F₁)^2 * (distance P F₂)^2)) = 3 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l1323_132394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_exponential_has_inverse_l1323_132308

-- Define the four functions
noncomputable def f1 (x : ℝ) : ℝ := x^2
noncomputable def f2 (x : ℝ) : ℝ := Real.sin x
noncomputable def f3 (x : ℝ) : ℝ := (2 : ℝ)^x
def f4 (_ : ℝ) : ℝ := 1

-- Define a property for a function to have an inverse within its domain
def has_inverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Theorem statement
theorem only_exponential_has_inverse :
  ¬(has_inverse f1) ∧
  ¬(has_inverse f2) ∧
  (has_inverse f3) ∧
  ¬(has_inverse f4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_exponential_has_inverse_l1323_132308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_5_l1323_132356

noncomputable def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 3)

theorem g_of_5 : g 5 = 13 / 2 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the numerator and denominator
  simp [mul_add, add_div]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_5_l1323_132356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prisoner_strategy_correct_l1323_132365

/-- Represents the state of the lamp in the room -/
inductive LampState
  | On
  | Off

/-- Represents a prisoner's action in the room -/
inductive PrisonerAction
  | TurnOn
  | TurnOff
  | NoAction

/-- Represents the strategy for the prisoners -/
structure PrisonerStrategy where
  isCounter : Bool
  hasVisited : Bool
  hasTurnedOn : Bool

/-- Helper function to count visits based on the strategy -/
def countVisits 
  (n : Nat)
  (prisoners : Fin n → PrisonerStrategy) 
  (visit_sequence : Nat → Fin n) 
  (t : Nat) : Nat :=
sorry

/-- The main theorem stating that the strategy guarantees correct counting -/
theorem prisoner_strategy_correct 
  (n : Nat) 
  (h_n : n = 100) 
  (prisoners : Fin n → PrisonerStrategy) 
  (h_one_counter : ∃! i, (prisoners i).isCounter) 
  (visit_sequence : Nat → Fin n) 
  (h_all_visit : ∀ i, ∃ k, visit_sequence k = i) 
  (h_infinite_visits : ∀ i k, ∃ l > k, visit_sequence l = i) :
  ∃ t, 
    (∀ i, (prisoners i).hasVisited) ∧ 
    (∃ i, (prisoners i).isCounter ∧ 
      (prisoners i).hasVisited ∧ 
      countVisits n prisoners visit_sequence t = n - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prisoner_strategy_correct_l1323_132365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_element_exists_l1323_132301

/-- A collection of sets satisfying the given conditions -/
structure SetCollection where
  sets : Finset (Finset ℕ)
  card_sets : sets.card = 1978
  card_each : ∀ s, s ∈ sets → s.card = 40
  intersection_size : ∀ s t, s ∈ sets → t ∈ sets → s ≠ t → (s ∩ t).card = 1

/-- The theorem statement -/
theorem common_element_exists (c : SetCollection) :
  ∃ x, ∀ s, s ∈ c.sets → x ∈ s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_element_exists_l1323_132301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_divisors_bound_l1323_132378

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := (Finset.filter (· ∣ n.val) (Finset.range n.val)).card

/-- Theorem: The number of divisors of n does not exceed 2√n -/
theorem num_divisors_bound (n : ℕ+) : num_divisors n ≤ 2 * Real.sqrt (n.val : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_divisors_bound_l1323_132378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l1323_132330

/-- Line l in the cartesian plane -/
def line_l (k : ℝ) (x y : ℝ) : Prop := x - k * y + 1 = 0

/-- Circle C in the cartesian plane -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Point on the cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector in the cartesian plane -/
structure Vec where
  x : ℝ
  y : ℝ

/-- Vector addition -/
def vector_add (v1 v2 : Vec) : Vec :=
  ⟨v1.x + v2.x, v1.y + v2.y⟩

/-- Convert a point to a vector from the origin -/
def point_to_vector (p : Point) : Vec :=
  ⟨p.x, p.y⟩

/-- The main theorem -/
theorem intersection_property (k : ℝ) (A B M : Point) :
  (∃ (x y : ℝ), line_l k x y ∧ circle_C x y) →  -- Line l intersects Circle C
  (vector_add (point_to_vector A) (point_to_vector B) = point_to_vector M) →  -- OM = OA + OB
  circle_C M.x M.y →  -- Point M lies on circle C
  k = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l1323_132330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_greater_than_ten_thousand_l1323_132375

theorem abs_diff_greater_than_ten_thousand
  (a b : ℤ)
  (h_distinct : a ≠ b)
  (h_a_large : a > 10^6)
  (h_b_large : b > 10^6)
  (h_divisible : (a + b)^3 % (a * b) = 0) :
  |a - b| > 10^4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_greater_than_ten_thousand_l1323_132375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_f_l1323_132381

-- Define the original function
def f (x : ℝ) : ℝ := 3 + 2*x - x^2

-- Define the two potential inverse functions
noncomputable def g₁ (x : ℝ) : ℝ := -1 + Real.sqrt (4 - x)
noncomputable def g₂ (x : ℝ) : ℝ := -1 - Real.sqrt (4 - x)

-- State the theorem
theorem inverse_of_f :
  (∀ x, f (g₁ x) = x ∧ g₁ (f x) = x) ∧
  (∀ x, f (g₂ x) = x ∧ g₂ (f x) = x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_f_l1323_132381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upper_bound_values_l1323_132324

/-- The golden ratio, approximately 0.618 --/
noncomputable def φ : ℝ := (Real.sqrt 5 - 1) / 2

/-- A good point using the 0.618 method --/
def goodPoint : ℝ := 2.382

/-- The lower bound of the optimization range --/
def lowerBound : ℝ := 2

/-- Checks if a value is a valid upper bound for the optimization range --/
def isValidUpperBound (b : ℝ) : Prop :=
  (lowerBound + (b - lowerBound) * φ = goodPoint) ∨
  (b - (b - lowerBound) * φ = goodPoint)

/-- Theorem stating the possible values for the upper bound --/
theorem upper_bound_values :
  ∀ b : ℝ, isValidUpperBound b → (b = 2.618 ∨ b = 3) :=
by sorry

#check upper_bound_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upper_bound_values_l1323_132324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positional_relationship_determined_by_distance_l1323_132361

/-- Line l in parametric form -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 * t, 1 + t)

/-- Curve C in polar form -/
noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.cos (θ + Real.pi / 4)

/-- Distance from a point to a line defined by two points -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (l1 l2 : ℝ × ℝ) : ℝ := 
  sorry

/-- Center of the curve C -/
def center_C : ℝ × ℝ := (0, 0)

/-- Radius of the curve C -/
noncomputable def radius_C : ℝ := 2 * Real.sqrt 2

/-- Two points on line l -/
noncomputable def point_l1 : ℝ × ℝ := line_l 0
noncomputable def point_l2 : ℝ × ℝ := line_l 1

/-- Theorem stating that the positional relationship can be determined by comparing distances -/
theorem positional_relationship_determined_by_distance :
  ∃ (d : ℝ), distance_point_to_line center_C point_l1 point_l2 = d ∧
  (d < radius_C ∨ d = radius_C ∨ d > radius_C) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_positional_relationship_determined_by_distance_l1323_132361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_iff_first_or_second_quadrant_l1323_132351

def is_meaningful (θ : Real) : Prop :=
  0 < Real.sin θ ∧ Real.sin θ ≠ 1

def in_first_or_second_quadrant (θ : Real) : Prop :=
  0 < θ ∧ θ < Real.pi

theorem meaningful_iff_first_or_second_quadrant (θ : Real) :
  is_meaningful θ ↔ in_first_or_second_quadrant θ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_iff_first_or_second_quadrant_l1323_132351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_present_ages_l1323_132349

-- Define the present ages of p and q
def P : ℕ := sorry
def Q : ℕ := sorry

-- Define the conditions
axiom age_relation_12_years_ago : P - 12 = (Q - 12) / 2
axiom present_age_ratio : P * 4 = Q * 3

-- Theorem to prove
theorem total_present_ages : P + Q = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_present_ages_l1323_132349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1323_132313

-- Define the triangle ABC and point P
variable (A B C P : ℝ × ℝ)

-- Define the conditions
variable (ABC_is_scalene_right_triangle : Prop)
variable (P_on_AC : Prop)
variable (angle_ABP_30_degrees : Prop)
def AP_length : ℝ := 1
def CP_length : ℝ := 3

-- Define the area function
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem area_of_triangle_ABC :
  ABC_is_scalene_right_triangle →
  P_on_AC →
  angle_ABP_30_degrees →
  AP_length = 1 →
  CP_length = 3 →
  area_triangle A B C = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1323_132313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_inverse_and_eigenvalues_l1323_132316

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 2, 1]
noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![1/3, 0; -2/3, 1]

theorem matrix_inverse_and_eigenvalues :
  (A * A_inv = 1) ∧
  (∃ (v : Matrix (Fin 2) (Fin 1) ℝ), v ≠ 0 ∧ A * v = v) ∧
  (∃ (v : Matrix (Fin 2) (Fin 1) ℝ), v ≠ 0 ∧ A * v = 3 • v) := by
  sorry

#check matrix_inverse_and_eigenvalues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_inverse_and_eigenvalues_l1323_132316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_24_equals_3_7_l1323_132373

def my_sequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 6/7
  | n+1 =>
    let a := my_sequence n
    if 0 ≤ a ∧ a < 1/2 then 2*a
    else if 1/2 ≤ a ∧ a < 1 then 2*a - 1
    else 0  -- This case should never occur, but Lean requires it for completeness

theorem sequence_24_equals_3_7 : my_sequence 23 = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_24_equals_3_7_l1323_132373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_total_distance_l1323_132325

/-- Represents the total distance of a triathlon -/
def triathlon_distance (d : ℝ) : Prop := True

/-- The cycling distance as a fraction of the total distance -/
noncomputable def cycling_fraction : ℝ := 3/4

/-- The running distance as a fraction of the total distance -/
noncomputable def running_fraction : ℝ := 1/5

/-- The swimming distance in kilometers -/
def swimming_distance : ℝ := 2

theorem triathlon_total_distance :
  ∀ d : ℝ, triathlon_distance d →
  cycling_fraction * d + running_fraction * d + swimming_distance = d →
  d = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_total_distance_l1323_132325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_cos_probability_is_7_15_l1323_132369

/-- An arithmetic sequence with special properties -/
structure SpecialArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  sum_4_is_pi : (a 1) + (a 2) + (a 3) + (a 4) = Real.pi
  a4_is_2a2 : a 4 = 2 * (a 2)
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- The probability that cos(a_n) < 0 for n = 1 to 30 -/
noncomputable def negative_cos_probability (seq : SpecialArithmeticSequence) : ℝ :=
  (Finset.filter (fun n => Real.cos (seq.a n) < 0) (Finset.range 30)).card / 30

/-- The main theorem -/
theorem negative_cos_probability_is_7_15 (seq : SpecialArithmeticSequence) :
  negative_cos_probability seq = 7/15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_cos_probability_is_7_15_l1323_132369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_values_l1323_132355

-- Define the slopes of the lines
noncomputable def slope_l1 (k : ℝ) : ℝ := (3 - k) / (5 - k)
noncomputable def slope_l2 (k : ℝ) : ℝ := k - 3

-- Define the perpendicularity condition
def perpendicular (k : ℝ) : Prop := slope_l1 k * slope_l2 k = -1

-- State the theorem
theorem perpendicular_lines_k_values :
  ∀ k : ℝ, perpendicular k → k = 1 ∨ k = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_values_l1323_132355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1323_132354

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the distance from intersection points to asymptotes
noncomputable def distance_to_asymptotes (a : ℝ) : ℝ := Real.sqrt 2 * a

-- Define the equation of asymptotes
def asymptote_equation (x y : ℝ) : Prop :=
  (Real.sqrt 2 * x + y = 0) ∨ (Real.sqrt 2 * x - y = 0)

-- Theorem statement
theorem hyperbola_asymptotes 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : ∀ (x y : ℝ), hyperbola a b x y → 
    ∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y ∧ 
    ∃ (q : ℝ × ℝ), asymptote_equation q.1 q.2 ∧ 
    dist p q = distance_to_asymptotes a) :
  ∀ (x y : ℝ), asymptote_equation x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1323_132354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_distances_l1323_132302

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (2 + 3 * Real.cos α, 3 * Real.sin α)

-- Define the line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop := 2 * ρ * Real.cos θ - ρ * Real.sin θ - 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (0, -1)

-- Define the intersection points A and B (existence assumed)
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ

-- Assume A and B are on both the curve and the line
axiom A_on_curve : ∃ α : ℝ, curve_C α = A
axiom B_on_curve : ∃ α : ℝ, curve_C α = B
axiom A_on_line : ∃ ρ θ : ℝ, line_l ρ θ ∧ (ρ * Real.cos θ, ρ * Real.sin θ) = A
axiom B_on_line : ∃ ρ θ : ℝ, line_l ρ θ ∧ (ρ * Real.cos θ, ρ * Real.sin θ) = B

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem sum_of_reciprocal_distances :
  1 / distance point_P A + 1 / distance point_P B = 3 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_distances_l1323_132302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaohua_walking_side_l1323_132359

-- Define cardinal directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define the relative position of two locations
def RelativePosition (a b : Direction) : Prop :=
  match a, b with
  | Direction.South, Direction.North => True
  | _, _ => False

-- Define the right side of a direction
def RightSide (dir : Direction) : Direction :=
  match dir with
  | Direction.North => Direction.East
  | Direction.East => Direction.South
  | Direction.South => Direction.West
  | Direction.West => Direction.North

-- Define the problem statement
theorem xiaohua_walking_side 
  (school_to_home : RelativePosition Direction.South Direction.North)
  (walk_on_right_side : ∀ (dir : Direction), RightSide dir = RightSide dir) :
  RightSide Direction.North = Direction.East := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaohua_walking_side_l1323_132359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_twenty_percent_l1323_132320

/-- Represents the compound interest calculation -/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

/-- Theorem stating that the given conditions result in a 20% annual interest rate -/
theorem interest_rate_is_twenty_percent (P A n t : ℝ)
  (h1 : P = 140)
  (h2 : A = 169.40)
  (h3 : n = 2)
  (h4 : t = 1)
  : ∃ r : ℝ, compound_interest P r n t = A ∧ r = 0.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_twenty_percent_l1323_132320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l1323_132396

-- Define the function f
noncomputable def f (A w φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (w * x + φ)

-- State the theorem
theorem function_analysis 
  (A w φ : ℝ) 
  (h_A : A > 0) 
  (h_w : w > 0) 
  (h_φ : 0 < φ ∧ φ < Real.pi / 2) 
  (h_period : ∀ x, f A w φ (x + Real.pi) = f A w φ x) 
  (h_lowest : f A w φ (2 * Real.pi / 3) = -2) :
  (∀ x, f A w φ x = 2 * Real.sin (2 * x + Real.pi / 6)) ∧ 
  (∀ x, x ≥ 0 ∧ x ≤ Real.pi / 2 → f A w φ x ≥ -1 ∧ f A w φ x ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l1323_132396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_negative_two_l1323_132389

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (4 * x + 8)

theorem vertical_asymptote_at_negative_two :
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), 0 < δ → δ < ε →
    (|f (- 2 + δ)| > (1 / δ) ∨ |f (- 2 - δ)| > (1 / δ)) := by
  sorry

#check vertical_asymptote_at_negative_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_negative_two_l1323_132389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_class_size_l1323_132399

/-- Represents a class of students -/
structure ClassOfStudents where
  students : Finset Nat
  friendship : Nat → Nat → Prop

/-- Properties of the class -/
def ValidClass (c : ClassOfStudents) : Prop :=
  ∀ s ∈ c.students,
    (∃ friends, friends ⊆ c.students ∧ friends.card = 5 ∧ ∀ f ∈ friends, c.friendship s f) ∨
    (∃ friends, friends ⊆ c.students ∧ friends.card = 6 ∧ ∀ f ∈ friends, c.friendship s f)

/-- Mutual friendship -/
def MutualFriendship (c : ClassOfStudents) : Prop :=
  ∀ s1 s2, s1 ∈ c.students → s2 ∈ c.students → c.friendship s1 s2 ↔ c.friendship s2 s1

/-- Different number of friends for any two friends -/
def DifferentFriendCount (c : ClassOfStudents) : Prop :=
  ∀ s1 s2, s1 ∈ c.students → s2 ∈ c.students → c.friendship s1 s2 →
    (∃ f1 f2, f1 ⊆ c.students ∧ f2 ⊆ c.students ∧
      (∀ f ∈ f1, c.friendship s1 f) ∧ (∀ f ∈ f2, c.friendship s2 f) ∧
      f1.card ≠ f2.card)

/-- The main theorem -/
theorem min_class_size :
  ∀ c : ClassOfStudents, ValidClass c → MutualFriendship c → DifferentFriendCount c →
    c.students.card > 0 → c.students.card ≥ 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_class_size_l1323_132399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_share_l1323_132335

/-- Represents the investment and profit sharing scenario of three partners -/
structure Partnership where
  initial_investment : ℝ
  annual_gain : ℝ

/-- Calculates a partner's share based on their investment amount and duration -/
noncomputable def investment_share (p : Partnership) (amount : ℝ) (duration : ℝ) : ℝ :=
  (amount * duration) / (p.initial_investment * 12 + 2 * p.initial_investment * 6 + 3 * p.initial_investment * 4)

/-- Theorem stating that partner A's share of the annual gain is one-third of the total -/
theorem partner_a_share (p : Partnership) (h : p.annual_gain = 24000) :
  investment_share p p.initial_investment 12 * p.annual_gain = 8000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_share_l1323_132335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hari_contribution_is_9000_l1323_132360

/-- Represents the business partnership between Praveen and Hari -/
structure Partnership where
  praveen_initial : ℕ  -- Praveen's initial investment
  hari_join_month : ℕ  -- Month when Hari joins
  total_months : ℕ     -- Total duration of business in months
  profit_ratio : Rat   -- Ratio of profit division (Praveen : Hari)
  hari_contribution : ℕ  -- Hari's contribution to be determined

/-- The specific partnership described in the problem -/
def problem_partnership : Partnership := {
  praveen_initial := 3500,
  hari_join_month := 5,
  total_months := 12,
  profit_ratio := 2 / 3,
  hari_contribution := 9000  -- To be proved
}

/-- Theorem stating that Hari's contribution in the given partnership is 9000 -/
theorem hari_contribution_is_9000 (p : Partnership) 
  (h1 : p = problem_partnership) : 
  p.hari_contribution = 9000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hari_contribution_is_9000_l1323_132360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_l1323_132388

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_side_calculation (abc : Triangle) 
  (h1 : abc.a * Real.sin abc.A + abc.b * Real.sin abc.B - abc.c * Real.sin abc.C = 
        (6 * Real.sqrt 7 / 7) * abc.a * Real.sin abc.B * Real.sin abc.C)
  (h2 : abc.a = 3)
  (h3 : abc.b = 2) : 
  abc.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_l1323_132388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_of_triangle_max_area_achieved_l1323_132323

/-- The ellipse C₁ with equation x²/3 + y² = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- The point P through which all lines pass -/
def P : ℝ × ℝ := (0, 2)

/-- A line passing through point P -/
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

/-- The area of triangle OAB given the slope of the line -/
noncomputable def triangle_area (k : ℝ) : ℝ := 6 * Real.sqrt (k^2 - 1) / (3 * k^2 - 1)

/-- The maximum area of triangle OAB -/
noncomputable def max_triangle_area : ℝ := Real.sqrt 3 / 2

theorem max_area_of_triangle :
  ∀ k : ℝ, k^2 > 1 → triangle_area k ≤ max_triangle_area :=
by sorry

theorem max_area_achieved :
  ∃ k : ℝ, k^2 > 1 ∧ triangle_area k = max_triangle_area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_of_triangle_max_area_achieved_l1323_132323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_equals_four_l1323_132387

/-- A function f with domain and range [1, b] where b > 1 -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (1/2) * (x - 1)^2 + a

/-- The theorem stating that a + b = 4 for the given conditions -/
theorem a_plus_b_equals_four (a b : ℝ) :
  b > 1 ∧
  (∀ x, x ∈ Set.Icc 1 b ↔ f a b x ∈ Set.Icc 1 b) →
  a + b = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_equals_four_l1323_132387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_price_l1323_132327

/-- The price in kopecks for 9 books -/
def price_9_books : ℕ := 1134

/-- The total number of books purchased -/
def total_books : ℕ := 17

/-- Theorem stating the conditions and the conclusion about the total price -/
theorem library_book_price : 
  (1130 < price_9_books ∧ price_9_books < 1140) ∧
  (price_9_books % 9 = 0) →
  price_9_books * total_books / 9 = 2142 := by
  intro h
  have h1 : 1130 < price_9_books ∧ price_9_books < 1140 := h.left
  have h2 : price_9_books % 9 = 0 := h.right
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_price_l1323_132327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_subset_sum_not_divisible_l1323_132395

theorem largest_subset_sum_not_divisible : ∃ (S : Finset ℕ), 
  (∀ x, x ∈ S → x ∈ Finset.range 1964) ∧ 
  (∀ x y, x ∈ S → y ∈ S → x ≠ y → (x + y) % 26 ≠ 0) ∧
  S.card = 76 ∧
  (∀ T : Finset ℕ, (∀ x, x ∈ T → x ∈ Finset.range 1964) → 
    (∀ x y, x ∈ T → y ∈ T → x ≠ y → (x + y) % 26 ≠ 0) → T.card ≤ 76) := by
  sorry

#check largest_subset_sum_not_divisible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_subset_sum_not_divisible_l1323_132395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1323_132332

/-- The line passing through (2, 1) and satisfying the given conditions -/
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 3}

/-- Point P -/
def P : ℝ × ℝ := (2, 1)

/-- Point A, where the line intersects the x-axis -/
def A : ℝ × ℝ := (3, 0)

/-- Point B, where the line intersects the y-axis -/
def B : ℝ × ℝ := (0, 3)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem line_equation_proof :
  (P ∈ line_l) ∧
  (A ∈ line_l) ∧
  (B ∈ line_l) ∧
  (A.1 > 0) ∧
  (B.2 > 0) ∧
  (distance P A * distance P B = 4) ∧
  (∀ p : ℝ × ℝ, p ∈ line_l ↔ p.1 + p.2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1323_132332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1323_132331

/-- A function f(x) with specific properties -/
noncomputable def f (A ω φ B : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ) + B

/-- Theorem stating the properties of the function and its roots -/
theorem function_properties 
  (A ω φ B : ℝ)
  (h_A : A > 0)
  (h_ω : ω > 0)
  (h_φ : |φ| < Real.pi / 2)
  (h_max : ∀ x, f A ω φ B x ≤ 2 * Real.sqrt 2)
  (h_min : ∀ x, f A ω φ B x ≥ -Real.sqrt 2)
  (h_period : ∀ x, f A ω φ B (x + Real.pi / ω) = f A ω φ B x)
  (h_point : f A ω φ B 0 = -Real.sqrt 2 / 4)
  (a : ℝ)
  (h_roots : ∃ α β, α ∈ Set.Icc 0 (7 * Real.pi / 12) ∧ 
                    β ∈ Set.Icc 0 (7 * Real.pi / 12) ∧ 
                    f A ω φ B α = a ∧ 
                    f A ω φ B β = a ∧ 
                    α ≠ β) :
  (∀ x, f A ω φ B x = 3 * Real.sqrt 2 / 2 * Real.sin (2 * x - Real.pi / 6) + Real.sqrt 2 / 2) ∧
  (∃ α β, f A ω φ B α = a ∧ f A ω φ B β = a ∧ α + β = 2 * Real.pi / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1323_132331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_installment_value_l1323_132370

/-- Calculates the value of each installment for a TV set purchase with given conditions -/
noncomputable def calculate_installment_value (tv_price : ℝ) (num_installments : ℕ) 
  (annual_interest_rate : ℝ) (last_installment : ℝ) : ℝ :=
  let monthly_interest_rate := annual_interest_rate / 12
  let interest_sum := (num_installments - 1) * num_installments / 2
  let x := (tv_price - last_installment) / 
    ((num_installments - 1) - monthly_interest_rate * interest_sum)
  x

/-- Theorem stating the approximate value of each installment for the given TV purchase scenario -/
theorem tv_installment_value :
  let tv_price : ℝ := 10000
  let num_installments : ℕ := 20
  let annual_interest_rate : ℝ := 0.06
  let last_installment : ℝ := 9000
  let installment_value := calculate_installment_value tv_price num_installments annual_interest_rate last_installment
  ∃ ε > 0, |installment_value - 55.40| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_installment_value_l1323_132370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l1323_132372

-- Define the power function as noncomputable
noncomputable def power_function (α : ℝ) : ℝ → ℝ := λ x => x ^ α

-- State the theorem
theorem power_function_theorem :
  ∀ α m : ℝ,
  (power_function α 2 = Real.sqrt 2) →
  (power_function α m = 4) →
  m = 16 := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l1323_132372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proofs_l1323_132377

theorem calculation_proofs :
  (Real.sqrt 9 + (-2020 : ℝ)^(0 : ℝ) - (1/4 : ℝ)^(-1 : ℝ) = 0) ∧
  (∀ (a b : ℝ), (2*a - b)^2 - (a + b)*(b - a) = 5*a^2 - 4*a*b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proofs_l1323_132377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_hyperbola_l1323_132333

open Real

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  b : ℝ
  eq : (x : ℝ) → (y : ℝ) → Prop

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (A B C : Point) : ℝ :=
  abs ((B.x - A.x) * C.y) / 2

/-- Theorem: Maximum area of triangle ABC for the given hyperbola -/
theorem max_triangle_area_hyperbola (h : Hyperbola) (A B C : Point) :
  (0 < h.b) → (h.b < 2) →
  h.eq A.x A.y →
  h.eq B.x B.y →
  A.y = 0 →
  B.y = 0 →
  C = ⟨0, h.b⟩ →
  (∀ b : ℝ, 0 < b → b < 2 → triangleArea A B C ≤ 2) ∧
  (∃ b : ℝ, 0 < b ∧ b < 2 ∧ triangleArea A B C = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_hyperbola_l1323_132333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_l1323_132371

-- Define the new operation ⊕
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  (oplus 1 x) * x

-- Theorem statement
theorem range_of_f_on_interval :
  (∀ y ∈ Set.Icc 0 8, ∃ x ∈ Set.Icc 0 2, f x = y) ∧
  (∀ x ∈ Set.Icc 0 2, f x ∈ Set.Icc 0 8) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_l1323_132371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_and_area_l1323_132337

/-- Triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Perimeter of a triangle -/
noncomputable def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Semi-perimeter of a triangle -/
noncomputable def semiperimeter (t : Triangle) : ℝ := perimeter t / 2

/-- Area of a triangle using Heron's formula -/
noncomputable def area (t : Triangle) : ℝ :=
  let s := semiperimeter t
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- Our specific triangle -/
def our_triangle : Triangle := { a := 10, b := 15, c := 18 }

theorem triangle_perimeter_and_area :
  perimeter our_triangle = 43 ∧ 
  74.45 < area our_triangle ∧ area our_triangle < 74.47 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_and_area_l1323_132337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_table_height_for_given_triangle_l1323_132391

/-- Triangle DEF with side lengths a, b, c -/
structure Triangle (a b c : ℝ) where
  side_a : a > 0
  side_b : b > 0
  side_c : c > 0

/-- Calculate the semi-perimeter of a triangle -/
noncomputable def semiPerimeter (t : Triangle a b c) : ℝ := (a + b + c) / 2

/-- Calculate the area of a triangle using Heron's formula -/
noncomputable def area (t : Triangle a b c) : ℝ :=
  let s := semiPerimeter t
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Calculate the altitude of a triangle given its area and base -/
noncomputable def altitude (A : ℝ) (base : ℝ) : ℝ := 2 * A / base

/-- The maximum height of the table -/
noncomputable def maxTableHeight (t : Triangle a b c) : ℝ :=
  let A := area t
  let h_a := altitude A a
  let h_b := altitude A b
  (h_a * h_b) / (h_a + h_b)

/-- The main theorem -/
theorem max_table_height_for_given_triangle :
  ∀ (t : Triangle 24 28 32),
  maxTableHeight t = 340 * Real.sqrt 35 / 39 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_table_height_for_given_triangle_l1323_132391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1323_132328

theorem calculation_proof :
  ((π - 3)^0 + Real.sqrt ((-4)^2) - (-1)^2023 = 6) ∧
  ((Real.sqrt 24 - Real.sqrt 6) / Real.sqrt 3 - (Real.sqrt 3 - Real.sqrt 2) * (Real.sqrt 3 + Real.sqrt 2) = Real.sqrt 2 - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1323_132328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1323_132317

open Real

/-- The number of unique intersection points with positive x-coordinates where at least two of the given logarithmic functions meet -/
def num_intersections : ℕ := 3

/-- The four logarithmic functions -/
noncomputable def f₁ (x : ℝ) : ℝ := log x / log 4
noncomputable def f₂ (x : ℝ) : ℝ := 1 / (log x / log 4)
noncomputable def f₃ (x : ℝ) : ℝ := -(log x / log 4)
noncomputable def f₄ (x : ℝ) : ℝ := -1 / (log x / log 4)

/-- A point is an intersection if it satisfies at least two of the functions -/
def is_intersection (x y : ℝ) : Prop :=
  (x > 0) ∧ (∃ i j, i ≠ j ∧ i ∈ ({1, 2, 3, 4} : Set ℕ) ∧ j ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    (if i = 1 then f₁ x = y else if i = 2 then f₂ x = y else if i = 3 then f₃ x = y else f₄ x = y) ∧
    (if j = 1 then f₁ x = y else if j = 2 then f₂ x = y else if j = 3 then f₃ x = y else f₄ x = y))

theorem intersection_points_count :
  ∃! (points : Set (ℝ × ℝ)), points.ncard = num_intersections ∧
    ∀ p ∈ points, is_intersection p.1 p.2 ∧
    ∀ x y, is_intersection x y → (x, y) ∈ points :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1323_132317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tail_to_body_ratio_l1323_132318

/-- Represents the measurements of a dog -/
structure DogMeasurements where
  overall_length : ℚ
  tail_length : ℚ
  head_ratio : ℚ

/-- Calculates the body length of the dog -/
def body_length (d : DogMeasurements) : ℚ :=
  (d.overall_length - d.tail_length) / (1 + d.head_ratio)

/-- Theorem stating the ratio of tail length to body length -/
theorem tail_to_body_ratio (d : DogMeasurements) 
  (h1 : d.overall_length = 30)
  (h2 : d.tail_length = 9)
  (h3 : d.head_ratio = 1/6) :
  d.tail_length / body_length d = 1/2 := by
  sorry

#check tail_to_body_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tail_to_body_ratio_l1323_132318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_intersection_l1323_132382

noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x + 8) / (x^2 - 6*x + 9)

theorem asymptotes_intersection :
  let vertical_asymptote := (3 : ℝ)
  let horizontal_asymptote := (1 : ℝ)
  (vertical_asymptote, horizontal_asymptote) = (3, 1) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - vertical_asymptote| < δ → |f x| > 1/ε) ∧
  (∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - horizontal_asymptote| < ε) :=
by sorry

#check asymptotes_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_intersection_l1323_132382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_six_count_l1323_132307

theorem remainder_six_count : 
  ∃! n : ℕ, n = (Finset.filter (fun N => N > 0 ∧ 111 % N = 6) (Finset.range 112)).card ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_six_count_l1323_132307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_5_or_7_not_9_up_to_200_l1323_132398

theorem multiples_5_or_7_not_9_up_to_200 : 
  Finset.card (Finset.filter (λ n => (5 ∣ n ∨ 7 ∣ n) ∧ ¬(9 ∣ n)) (Finset.range 201)) = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_5_or_7_not_9_up_to_200_l1323_132398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_eq_110_l1323_132305

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : ∀ n, a (n + 1) = a n + d
  h2 : d ≠ 0
  h3 : a 1 = 20
  h4 : (a 5) ^ 2 = (a 2) * (a 7)

/-- The sum of the first n terms of an arithmetic sequence -/
def sum (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

/-- The main theorem: S₁₀ = 110 for the given arithmetic sequence -/
theorem sum_10_eq_110 (seq : ArithmeticSequence) : sum seq 10 = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_eq_110_l1323_132305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_inequality_l1323_132344

theorem cosine_product_inequality (A B C : Real) (h_triangle : A + B + C = Real.pi) :
  Real.cos A * Real.cos B * Real.cos C ≤ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_inequality_l1323_132344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l1323_132347

-- Define the function f(x) = (x - 4)e^x
noncomputable def f (x : ℝ) : ℝ := (x - 4) * Real.exp x

-- State the theorem
theorem f_monotone_decreasing :
  StrictMonoOn f (Set.Iio 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l1323_132347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_correct_l1323_132311

noncomputable def sandwich_cost : ℝ := 4
noncomputable def soda_cost : ℝ := 3
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 5
noncomputable def discount_threshold : ℝ := 30
noncomputable def discount_rate : ℝ := 0.1

noncomputable def total_cost : ℝ :=
  let subtotal := sandwich_cost * num_sandwiches + soda_cost * num_sodas
  if subtotal > discount_threshold
  then subtotal * (1 - discount_rate)
  else subtotal

theorem total_cost_is_correct : total_cost = 38.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_correct_l1323_132311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1323_132334

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2-x) + 1

-- Define the theorem
theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) (hmn : m * n > 0) :
  (∃ (x y : ℝ), f a x = y ∧ m * x + n * y = 1) →
  (1 / m + 1 / n ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), m₀ * n₀ > 0 ∧ 1 / m₀ + 1 / n₀ = 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1323_132334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l1323_132385

noncomputable def f (x m : ℝ) : ℝ :=
  if x > 1 then Real.log x
  else 2 * x + m^3

theorem find_m : ∃ m : ℝ, f (f (Real.exp 1) m) m = 10 ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l1323_132385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cardinality_valid_subset_l1323_132346

/-- A subset of integers from 1 to 150 where no member is 4 times another --/
def ValidSubset (S : Finset ℤ) : Prop :=
  ∀ x ∈ S, x ≥ 1 ∧ x ≤ 150 ∧ ∀ y ∈ S, y ≠ 4 * x

/-- The maximum cardinality of a valid subset is 140 --/
theorem max_cardinality_valid_subset :
  ∃ S : Finset ℤ, ValidSubset S ∧ S.card = 140 ∧
  ∀ T : Finset ℤ, ValidSubset T → T.card ≤ 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cardinality_valid_subset_l1323_132346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_eccentricity_l1323_132353

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A parabola with focus at the origin -/
structure Parabola where
  p : ℝ
  h_pos : 0 < p

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The theorem stating the eccentricity of the ellipse under given conditions -/
theorem ellipse_parabola_intersection_eccentricity
  (e : Ellipse) (c : Parabola)
  (h1 : c.p = e.a * eccentricity e) -- Parabola's focus is at ellipse's center
  (h2 : c.p = 2 * e.b) -- Parabola passes through ellipse's foci
  (h3 : ∃ (x y : ℝ), x^2 / e.a^2 + y^2 / e.b^2 = 1 ∧ y = x^2 / (2 * c.p)) -- Three intersection points
  : eccentricity e = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_eccentricity_l1323_132353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_coordinates_l1323_132376

/-- A parallelogram with opposite vertices (2,-3) and (12,9) -/
structure Parallelogram where
  v1 : ℝ × ℝ := (2, -3)
  v2 : ℝ × ℝ := (12, 9)

/-- The point where the diagonals of the parallelogram intersect -/
noncomputable def diagonalIntersection (p : Parallelogram) : ℝ × ℝ :=
  ((p.v1.1 + p.v2.1) / 2, (p.v1.2 + p.v2.2) / 2)

theorem diagonal_intersection_coordinates (p : Parallelogram) :
  diagonalIntersection p = (7, 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_coordinates_l1323_132376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_difference_equals_seven_factorial_l1323_132393

theorem factorial_difference_equals_seven_factorial :
  Nat.factorial 8 - 6 * Nat.factorial 7 - Nat.factorial 7 = Nat.factorial 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_difference_equals_seven_factorial_l1323_132393
