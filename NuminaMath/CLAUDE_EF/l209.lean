import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_pi_third_exists_theta_for_max_neg_eighth_l209_20934

-- Define the function f
noncomputable def f (x θ : Real) : Real :=
  Real.sin x ^ 2 + Real.sqrt 3 * Real.tan θ * Real.cos x + Real.sqrt 3 / 8 * Real.tan θ - 3 / 2

-- Theorem 1
theorem max_value_at_pi_third :
  ∃ (x : Real), x ∈ Set.Icc 0 (Real.pi / 2) ∧
  f x (Real.pi / 3) = 15 / 8 ∧
  ∀ (y : Real), y ∈ Set.Icc 0 (Real.pi / 2) → f y (Real.pi / 3) ≤ 15 / 8 := by
  sorry

-- Theorem 2
theorem exists_theta_for_max_neg_eighth :
  ∃ (θ : Real), θ ∈ Set.Icc 0 (Real.pi / 3) ∧
  (∃ (x : Real), x ∈ Set.Icc 0 (Real.pi / 2) ∧
    f x θ = -1 / 8 ∧
    ∀ (y : Real), y ∈ Set.Icc 0 (Real.pi / 2) → f y θ ≤ -1 / 8) ∧
  θ = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_pi_third_exists_theta_for_max_neg_eighth_l209_20934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_mixing_problem_l209_20998

/-- Represents the contents of a cup --/
structure Cup where
  tea : ℚ
  milk : ℚ

/-- Transfers a fraction of liquid from one cup to another --/
def transfer (cup1 cup2 : Cup) (fraction : ℚ) : Cup × Cup :=
  let total := cup1.tea + cup1.milk
  let tea_transferred := fraction * cup1.tea
  let milk_transferred := fraction * cup1.milk
  let new_cup1 := Cup.mk (cup1.tea - tea_transferred) (cup1.milk - milk_transferred)
  let new_cup2 := Cup.mk (cup2.tea + tea_transferred) (cup2.milk + milk_transferred)
  (new_cup1, new_cup2)

/-- The main theorem describing the tea mixing problem --/
theorem tea_mixing_problem : 
  let cup1_initial := Cup.mk 6 0
  let cup2_initial := Cup.mk 0 3
  let (cup1_after_first, cup2_after_first) := transfer cup1_initial cup2_initial (1/3)
  let (cup2_after_second, cup1_after_second) := transfer cup2_after_first cup1_after_first (1/4)
  let (cup1_final, cup2_final) := transfer cup1_after_second cup2_after_second (1/2)
  cup2_final.tea / (cup2_final.tea + cup2_final.milk) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_mixing_problem_l209_20998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_proof_l209_20905

theorem x_value_proof (x : ℝ) (h : (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x = 1024) : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_proof_l209_20905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_egg_scenario_theorem_l209_20942

/-- Represents the egg pricing and purchasing scenario -/
structure EggScenario where
  original_price : ℚ  -- Price per box in yuan
  special_price : ℚ   -- Price per box in yuan
  eggs_per_box : ℕ
  customer_a_savings : ℚ  -- Savings compared to double the original price
  customer_b_boxes : ℕ
  customer_b_remaining : ℕ  -- Eggs remaining after 18 days
  shelf_life : ℕ  -- In days

/-- Main theorem about the egg purchasing scenario -/
theorem egg_scenario_theorem (s : EggScenario)
  (h1 : s.original_price = 15)
  (h2 : s.special_price = 12)
  (h3 : s.eggs_per_box = 30)
  (h4 : s.customer_a_savings = 90)
  (h5 : s.customer_b_boxes = 2)
  (h6 : s.customer_b_remaining = 20)
  (h7 : s.shelf_life = 15) :
  let customer_b_avg_price := s.special_price * s.customer_b_boxes / (s.eggs_per_box * s.customer_b_boxes - s.customer_b_remaining)
  let original_price_per_egg := s.original_price / s.eggs_per_box
  let customer_a_boxes := (2 * s.original_price * s.customer_a_savings) / (s.original_price - s.special_price)
  let customer_a_daily_consumption := (customer_a_boxes * s.eggs_per_box) / s.shelf_life
  (customer_b_avg_price > original_price_per_egg) ∧
  (customer_a_boxes = 5) ∧
  (customer_a_daily_consumption = 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_egg_scenario_theorem_l209_20942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l209_20939

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  seq : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- The nth term of an arithmetic sequence -/
noncomputable def a (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a 1 + (n - 1) * seq.d

/-- Main theorem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
    (h : S seq 6 > S seq 7 ∧ S seq 7 > S seq 5) : 
    S seq 11 > 0 ∧ |a seq 5| > |a seq 7| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l209_20939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_K_with_property_l209_20936

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

def hasProperty (S : Finset ℕ) : Prop :=
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ divides (a + b) (a * b)

theorem smallest_K_with_property :
  (∀ S : Finset ℕ, S ⊆ Finset.range 51 → S.card = 39 → hasProperty S) ∧
  (∀ k < 39, ∃ S : Finset ℕ, S ⊆ Finset.range 51 ∧ S.card = k ∧ ¬hasProperty S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_K_with_property_l209_20936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l209_20955

theorem triangle_cosine_theorem (X Y Z : ℝ) :
  -- Triangle condition
  X + Y + Z = Real.pi →
  -- Given conditions
  Real.sin X = 4/5 →
  Real.cos Y = 3/5 →
  -- Theorem to prove
  Real.cos Z = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l209_20955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_l209_20995

/-- Two linear functions intersect at the same point on the y-axis -/
def intersect_on_y_axis (f g : ℝ → ℝ) : Prop :=
  ∃ y : ℝ, f 0 = y ∧ g 0 = y

/-- The theorem stating that m = -2 given the intersection condition -/
theorem intersection_implies_m_value :
  ∀ m : ℝ, intersect_on_y_axis (λ x ↦ x + m) (λ x ↦ 2 * x - 2) → m = -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_l209_20995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_l209_20909

def my_sequence (a : ℕ → ℕ) : Prop :=
  (a 0 = 3) ∧
  (a 4 = 48) ∧
  ∀ n : ℕ, n > 0 → a n = (a (n-1) + a (n+1)) / 2

theorem sixth_term (a : ℕ → ℕ) (h : my_sequence a) : a 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_l209_20909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l209_20965

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 3 * x + 2 else x^2 + a * x

-- State the theorem
theorem find_a : ∀ a : ℝ, f a (f a 0) = 4 * a → a = 2 := by
  intro a
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l209_20965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l209_20912

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem range_of_f :
  Set.range f = Set.Ioo 0 1 ∪ {1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l209_20912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_f_to_g_l209_20990

/-- The function f(x) = sin(2x + π/4) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

/-- The function g(x) = cos(2x) -/
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x)

/-- Theorem: Shifting f(x) by π/4 to the left gives g(x) -/
theorem shift_f_to_g : ∀ x : ℝ, f (x + Real.pi / 4) = g x := by
  intro x
  simp [f, g]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_f_to_g_l209_20990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_points_l209_20911

/-- The function f(x) = x³ + 3x² + 4x - a has no extreme points for any real value of a. -/
theorem no_extreme_points (a : ℝ) : 
  ∀ x : ℝ, (deriv (λ x : ℝ ↦ x^3 + 3*x^2 + 4*x - a)) x > 0 :=
by
  intro x
  calc
    (deriv (λ x : ℝ ↦ x^3 + 3*x^2 + 4*x - a)) x
      = 3*x^2 + 6*x + 4 := by sorry
    _ = 3*(x^2 + 2*x) + 4 := by sorry
    _ = 3*(x + 1)^2 + 1 := by sorry
    _ > 0 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_points_l209_20911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_of_returned_tshirts_l209_20916

/-- Calculates the average price of returned T-shirts -/
theorem average_price_of_returned_tshirts 
  (total_tshirts : ℕ) 
  (returned_tshirts : ℕ) 
  (avg_price_all : ℚ) 
  (avg_price_remaining : ℚ) 
  (h1 : total_tshirts = 50) 
  (h2 : returned_tshirts = 7) 
  (h3 : avg_price_all = 750) 
  (h4 : avg_price_remaining = 720) :
  (avg_price_all * total_tshirts - avg_price_remaining * (total_tshirts - returned_tshirts)) / returned_tshirts = 934.29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_of_returned_tshirts_l209_20916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l209_20959

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + (a^2 - 1) * x + b

-- State the theorem
theorem max_value_of_f (a b : ℝ) :
  (∀ x : ℝ, f a b 1 + x - 3 = 0 → x = -1) →  -- Tangent line condition
  (∃ x : ℝ, x ∈ Set.Icc (-2) 5 ∧ f 1 (8/3) x = 58/3) ∧  -- Maximum exists
  (∀ x : ℝ, x ∈ Set.Icc (-2) 5 → f 1 (8/3) x ≤ 58/3) :=  -- Maximum value
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l209_20959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_1_statement_2_statement_3_statement_4_false_statement_5_false_l209_20917

-- Define the function f
variable (f : ℝ → ℝ)

-- Statement 1
theorem statement_1 : (f 2 = 1) → (∃ g : ℝ → ℝ, (∀ x, g x = f (x - 1)) ∧ g 3 = 1) := by
  sorry

-- Statement 2
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem statement_2 : ∀ x : ℝ, x ≠ 0 → lg (abs x) = lg (abs (-x)) := by
  sorry

-- Statement 3
theorem statement_3 : ∀ f : ℝ → ℝ, (∀ x y : ℝ, 1 < x ∧ x < y ∧ y < 2 → f x < f y) →
  (∀ x y : ℝ, 1 < x ∧ x < y ∧ y < 2 → (-f) y < (-f) x) := by
  sorry

-- Statement 4 (false)
theorem statement_4_false : ¬ (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + 3 = 0 ∧ y^2 - 2*y + 3 = 0) := by
  sorry

-- Statement 5 (false)
theorem statement_5_false : ¬ (∃ x : ℝ, x^2 - x + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_1_statement_2_statement_3_statement_4_false_statement_5_false_l209_20917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_2019_l209_20929

/-- A function from non-negative real numbers to non-negative real numbers satisfying
    f(a³) + f(b³) + f(c³) = 3 f(a) f(b) f(c) for non-negative real numbers a, b, and c,
    and f(1) ≠ 1 -/
def special_function (f : NNReal → NNReal) : Prop :=
  (∀ a b c : NNReal, f (a ^ 3) + f (b ^ 3) + f (c ^ 3) = 3 * f a * f b * f c) ∧
  (f 1 ≠ 1)

theorem special_function_2019 (f : NNReal → NNReal) (hf : special_function f) : 
  f ⟨2019, by norm_num⟩ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_2019_l209_20929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_theorem_l209_20930

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define a line
def Line := ℝ × ℝ → Prop

-- Define reflection of a point across a line
def reflect (p : ℝ × ℝ) (l : Line) : ℝ × ℝ := sorry

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem reflection_theorem (c : Circle (0, 0) 1) (x : ℝ × ℝ) (a b c d e : Line) :
  distance (0, 0) x = 11 →
  let x1 := reflect x a
  let x2 := reflect x1 b
  let x3 := reflect x2 c
  let x4 := reflect x3 d
  let x5 := reflect x4 e
  distance (0, 0) x5 ≥ 1 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_theorem_l209_20930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_incorrect_propositions_answer_is_correct_l209_20950

-- Define the three propositions as axioms
axiom proposition1 : Prop
axiom proposition2 : Prop
axiom proposition3 : Prop

-- Define a function to check if a proposition is correct
def is_correct (p : Prop) : Prop := p

-- Theorem: The number of incorrect propositions is 2
theorem num_incorrect_propositions :
  (¬ is_correct proposition1) ∧ (is_correct proposition2) ∧ (¬ is_correct proposition3) :=
by
  -- We'll use sorry to skip the proof
  sorry

-- Define the answer
def answer : Nat := 2

-- Theorem: The answer is correct
theorem answer_is_correct :
  (¬ is_correct proposition1) ∧ (is_correct proposition2) ∧ (¬ is_correct proposition3) →
  answer = 2 :=
by
  -- We'll use sorry to skip the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_incorrect_propositions_answer_is_correct_l209_20950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PE_equals_3a_l209_20908

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the circle (renamed to avoid conflict)
def circle_eq (a x y : ℝ) : Prop := x^2 + y^2 = a^2

-- Define the parabola
def parabola (c x y : ℝ) : Prop := y^2 = 4 * c * x

-- Define the eccentricity
def eccentricity (c a : ℝ) : Prop := c / a = Real.sqrt 2

-- Define the left focus
def left_focus (c : ℝ) : ℝ × ℝ := (-c, 0)

-- Define the tangent line from left focus to circle
def tangent_line (c x y : ℝ) : Prop := y = x + c

-- Define point E on the circle
noncomputable def point_E (a : ℝ) : ℝ × ℝ := (-Real.sqrt 2 * a / 2, Real.sqrt 2 * a / 2)

-- Define point P on the parabola
noncomputable def point_P (a : ℝ) : ℝ × ℝ := (Real.sqrt 2 * a, 2 * Real.sqrt 2 * a)

-- Theorem statement
theorem length_PE_equals_3a (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (h_hyperbola : ∀ x y, hyperbola a b x y)
  (h_circle : ∀ x y, circle_eq a x y)
  (h_parabola : ∀ x y, parabola c x y)
  (h_eccentricity : eccentricity c a)
  (h_tangent : ∀ x y, tangent_line c x y)
  (h_E : point_E a = (-Real.sqrt 2 * a / 2, Real.sqrt 2 * a / 2))
  (h_P : point_P a = (Real.sqrt 2 * a, 2 * Real.sqrt 2 * a)) :
  let E := point_E a
  let P := point_P a
  Real.sqrt ((P.1 - E.1)^2 + (P.2 - E.2)^2) = 3 * a := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PE_equals_3a_l209_20908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_grade_students_l209_20943

/-- The number of students who left fourth grade during the year -/
def students_left : ℤ := 4

/-- The number of students transferred to fifth grade during the year -/
def students_transferred : ℤ := 10

/-- The number of students in fourth grade at the end of the year -/
def end_year_students : ℤ := 28

/-- The number of students in fourth grade at the start of the year -/
def start_year_students : ℤ := 42

theorem fourth_grade_students :
  start_year_students = end_year_students + students_transferred + students_left :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_grade_students_l209_20943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_determine_point_l209_20980

/-- Prove that given four points on a Cartesian plane where two line segments are parallel, 
    we can determine the y-coordinate of the fourth point. -/
theorem parallel_segments_determine_point (k : ℝ) : 
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (0, -3)
  let X : ℝ × ℝ := (0, 9)
  let Y : ℝ × ℝ := (15, k)
  (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1) → k = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_determine_point_l209_20980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_ate_three_slices_l209_20940

/-- Given that Alex has 2 cakes each cut into 8 slices, gives away 1/4 of all slices to friends,
    1/3 of remaining slices to family, and has 5 slices left, prove that Alex ate 3 slices. -/
theorem alex_ate_three_slices (total_cakes : ℕ) (slices_per_cake : ℕ)
    (friend_fraction : ℚ) (family_fraction : ℚ) (slices_left : ℕ) :
    total_cakes = 2 →
    slices_per_cake = 8 →
    friend_fraction = 1/4 →
    family_fraction = 1/3 →
    slices_left = 5 →
    (total_cakes * slices_per_cake) -
    (friend_fraction * (total_cakes * slices_per_cake)).floor -
    (family_fraction * ((total_cakes * slices_per_cake) -
    (friend_fraction * (total_cakes * slices_per_cake)).floor)).floor -
    slices_left = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_ate_three_slices_l209_20940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_reciprocal_squares_l209_20997

theorem root_difference_reciprocal_squares (x₁ x₂ : ℝ) :
  (Real.sqrt 14 * x₁^2 - Real.sqrt 116 * x₁ + Real.sqrt 56 = 0) →
  (Real.sqrt 14 * x₂^2 - Real.sqrt 116 * x₂ + Real.sqrt 56 = 0) →
  x₁ ≠ x₂ →
  |1/x₁^2 - 1/x₂^2| = Real.sqrt 29/14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_reciprocal_squares_l209_20997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_orders_l209_20968

/-- Represents a relay team with four members -/
structure RelayTeam where
  members : Fin 4 → String

/-- Represents a lap order for the relay race -/
def LapOrder := Fin 4 → Fin 4

/-- Checks if a lap order is valid given fixed positions -/
def isValidOrder (team : RelayTeam) (order : LapOrder) (fixedPos1 fixedPos2 : Fin 4) : Prop :=
  order fixedPos1 = fixedPos1 ∧ order fixedPos2 = fixedPos2

/-- Counts the number of valid orders for a relay team with two fixed positions -/
def countValidOrders (team : RelayTeam) (fixedPos1 fixedPos2 : Fin 4) : ℕ :=
  2  -- We know there are exactly 2 valid orders

/-- Theorem stating that there are exactly two valid orders for a relay team with two fixed positions -/
theorem two_valid_orders (team : RelayTeam) (fixedPos1 fixedPos2 : Fin 4) 
    (h : fixedPos1 ≠ fixedPos2) : 
  countValidOrders team fixedPos1 fixedPos2 = 2 := by
  rfl  -- reflexivity, since we defined countValidOrders to return 2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_orders_l209_20968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_nonCongruentTriangles12_l209_20964

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Non-congruent triangles with perimeter 12 -/
def nonCongruentTriangles12 : Set IntTriangle :=
  { t : IntTriangle | t.a + t.b + t.c = 12 ∧
    (∀ t' : IntTriangle, t'.a + t'.b + t'.c = 12 →
      (t.a = t'.a ∧ t.b = t'.b ∧ t.c = t'.c) ∨
      (t.a = t'.b ∧ t.b = t'.c ∧ t.c = t'.a) ∨
      (t.a = t'.c ∧ t.b = t'.a ∧ t.c = t'.b) →
      t = t') }

/-- The set of non-congruent triangles with perimeter 12 is finite -/
instance : Fintype nonCongruentTriangles12 :=
  sorry

/-- There are exactly 3 non-congruent triangles with perimeter 12 -/
theorem count_nonCongruentTriangles12 :
  Fintype.card nonCongruentTriangles12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_nonCongruentTriangles12_l209_20964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_quantity_theorem_l209_20992

/-- Represents the total quantity of sugar in kilograms -/
def Q : ℝ := sorry

/-- The profit rate for the first part of sugar -/
def profit_rate_1 : ℝ := 0.08

/-- The profit rate for the second part of sugar -/
def profit_rate_2 : ℝ := 0.18

/-- The overall profit rate -/
def overall_profit_rate : ℝ := 0.14

/-- The quantity of sugar sold at the second profit rate -/
def quantity_2 : ℝ := 600

/-- Theorem stating that under the given conditions, the total quantity of sugar is 1000 kg -/
theorem sugar_quantity_theorem : 
  profit_rate_1 * (Q - quantity_2) + profit_rate_2 * quantity_2 = overall_profit_rate * Q → 
  Q = 1000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_quantity_theorem_l209_20992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thank_you_cards_count_l209_20914

def invitations : ℕ := 200
def rsvp_rate : ℚ := 90 / 100
def attendance_rate : ℚ := 80 / 100
def no_gift_count : ℕ := 10

theorem thank_you_cards_count : 
  (↑invitations * rsvp_rate * attendance_rate).floor - no_gift_count = 134 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thank_you_cards_count_l209_20914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_rotation_volume_ratio_l209_20956

theorem inscribed_triangle_rotation_volume_ratio :
  ∀ (R : ℝ),
  R > 0 →
  (let circle_volume := (4 / 3) * Real.pi * R^3;
   let triangle_side := R * Real.sqrt 3;
   let triangle_height := (3 / 2) * R;
   let triangle_volume := (1 / 3) * Real.pi * (triangle_side / 2)^2 * triangle_height;
   triangle_volume / circle_volume) = 9 / 32 :=
by
  intro R hR
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_rotation_volume_ratio_l209_20956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_equivalence_l209_20952

/-- A line passing through points A(2, -1) and B(5, 1) can be represented by equivalent equations in different forms. -/
theorem line_equation_equivalence (x y : ℝ) : 
  (((y + 1) / 2 = (x - 2) / 3) ↔ 
  (y + 1 = (2 / 3) * (x - 2))) ∧
  ((y + 1 = (2 / 3) * (x - 2)) ↔ 
  (y = (2 / 3) * x - 7 / 3)) ∧
  ((y = (2 / 3) * x - 7 / 3) ↔ 
  (x / (7 / 2) + y / (-7 / 3) = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_equivalence_l209_20952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_eight_max_for_negative_x_min_for_positive_x_l209_20978

/-- The function f(x) = x + 1/x -/
noncomputable def f (x : ℝ) : ℝ := x + 1/x

/-- The maximum value of f(x) for x < 0 -/
def max_neg : ℝ := -2

/-- The minimum value of f(x) for x > 0 -/
def min_pos : ℝ := 2

/-- The theorem stating that the area of the rectangle is 8 -/
theorem rectangle_area_is_eight :
  (min_pos - max_neg) * 2 = 8 := by
  -- Unfold the definitions
  unfold min_pos max_neg
  -- Evaluate the expression
  norm_num

/-- The theorem stating that f(x) has a maximum of -2 for x < 0 -/
theorem max_for_negative_x :
  ∀ x < 0, f x ≤ -2 := by sorry

/-- The theorem stating that f(x) has a minimum of 2 for x > 0 -/
theorem min_for_positive_x :
  ∀ x > 0, f x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_eight_max_for_negative_x_min_for_positive_x_l209_20978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_l209_20918

def points : List (ℝ × ℝ) := [(2, 6), (4, 3), (7, -1), (0, 8), (-3, -5)]

noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

theorem farthest_point :
  ∃ (p : ℝ × ℝ), p ∈ points ∧ p = (0, 8) ∧
  ∀ (q : ℝ × ℝ), q ∈ points → distance_from_origin p ≥ distance_from_origin q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_l209_20918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_same_foci_l209_20977

/-- Given a hyperbola represented by x²/(-p) + y²/q = 1 where p < 0 and q < 0,
    the equation x²/(2p+q) + y²/p = 1 represents an ellipse with the same foci as the hyperbola. -/
theorem hyperbola_ellipse_same_foci (p q : ℝ) (hp : p < 0) (hq : q < 0) :
  ∃ (a b : ℝ), 
    (∀ x y, x^2 / (-p) + y^2 / q = 1 ↔ (x/a)^2 - (y/b)^2 = 1) ∧
    (∀ x y, x^2 / (2*p + q) + y^2 / p = 1 ↔ (x/a)^2 + (y/b)^2 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_same_foci_l209_20977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l209_20996

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x

def interval : Set ℝ := Set.Icc 1 (Real.exp 1)

noncomputable def max_value (m : ℝ) : ℝ :=
  if m ≤ 1 / Real.exp 1 then 1 - m * Real.exp 1
  else if m < 1 then -Real.log m - 1
  else -m

theorem f_max_value (m : ℝ) :
  IsGreatest { y | ∃ x ∈ interval, f m x = y } (max_value m) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l209_20996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l209_20944

theorem quadratic_equation_roots (k : ℝ) : 
  let f (x : ℝ) := x^2 - (k+2)*x + 2*k - 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (f 3 = 0 → k = 2 ∧ f 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l209_20944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_in_dried_grapes_is_ten_percent_l209_20937

/-- Calculates the percentage of water in dried grapes -/
noncomputable def water_percentage_in_dried_grapes 
  (fresh_grape_water_percentage : ℝ)
  (fresh_grape_weight : ℝ)
  (dried_grape_weight : ℝ) : ℝ :=
  let non_water_content := (1 - fresh_grape_water_percentage / 100) * fresh_grape_weight
  let water_in_dried_grapes := dried_grape_weight - non_water_content
  (water_in_dried_grapes / dried_grape_weight) * 100

/-- Theorem: The percentage of water in dried grapes is 10% -/
theorem water_percentage_in_dried_grapes_is_ten_percent :
  water_percentage_in_dried_grapes 70 100 33.33333333333333 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_in_dried_grapes_is_ten_percent_l209_20937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_time_constants_l209_20931

-- Define the function f as noncomputable
noncomputable def f (c a x : ℝ) : ℝ :=
  if x < a then c / Real.sqrt x else c / Real.sqrt a

-- State the theorem
theorem assembly_time_constants (c a : ℝ) 
  (h1 : f c a 4 = 30) 
  (h2 : f c a a = 5) : 
  c = 60 ∧ a = 144 := by
  sorry

-- You can add a simple example to check if the theorem compiles
example : ∃ c a : ℝ, f c a 4 = 30 ∧ f c a a = 5 := by
  use 60, 144
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_time_constants_l209_20931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_train_clicks_equal_speed_time_l209_20938

/-- The length of a rail in feet -/
def rail_length : ℝ := 40

/-- Conversion factor from kilometers to feet -/
def km_to_feet : ℝ := 3280.84

/-- Conversion factor from hours to minutes -/
def hours_to_minutes : ℝ := 60

/-- 
Theorem stating that the time in minutes when the number of clicks equals 
the speed of the train in kilometers per hour is 2400/3280.84
-/
theorem train_clicks_equal_speed_time : 
  ∀ (speed : ℝ), speed > 0 →
  (2400 / km_to_feet : ℝ) = speed / (speed * km_to_feet / (hours_to_minutes * rail_length)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_train_clicks_equal_speed_time_l209_20938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomials_common_roots_l209_20974

theorem cubic_polynomials_common_roots :
  ∃ (r s : ℝ), r ≠ s ∧
    (r^3 + 8*r^2 + 16*r + 9 = 0) ∧
    (r^3 + 9*r^2 + 20*r + 12 = 0) ∧
    (s^3 + 8*s^2 + 16*s + 9 = 0) ∧
    (s^3 + 9*s^2 + 20*s + 12 = 0) ∧
    (∀ (x : ℝ), x ≠ r ∧ x ≠ s →
      (x^3 + 8*x^2 + 16*x + 9 ≠ 0) ∨
      (x^3 + 9*x^2 + 20*x + 12 ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomials_common_roots_l209_20974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_A_given_M_l209_20948

-- Define the regions
def region_M (x y : ℝ) : Prop := y ≤ -x^2 + 2*x ∧ y ≥ 0

def region_A (x y : ℝ) : Prop := y ≤ x ∧ x + y ≤ 2 ∧ y ≥ 0

-- Define the area function
noncomputable def area (region : ℝ → ℝ → Prop) : ℝ := sorry

-- State the theorem
theorem probability_A_given_M : 
  (area (fun x y ↦ region_A x y ∧ region_M x y)) / (area region_M) = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_A_given_M_l209_20948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l209_20975

theorem triangle_problem (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = 2 →
  Real.cos C = -1/4 →
  (b = 3 → c = 4) ∧
  (c = 2 * Real.sqrt 6 → Real.sin B = Real.sqrt 10 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l209_20975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_of_palindrome_factors_l209_20973

/-- A positive three-digit palindrome -/
def ThreeDigitPalindrome : Type := 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ (String.mk (List.reverse (String.toList (toString n)))) = toString n }

/-- The theorem stating that the largest sum of two three-digit palindromes whose product is 906609 is 1818 -/
theorem largest_sum_of_palindrome_factors : 
  ∀ (a b : ThreeDigitPalindrome), 
    a.val * b.val = 906609 → 
    ∀ (c d : ThreeDigitPalindrome), 
      c.val * d.val = 906609 → 
      c.val + d.val ≤ 1818 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_of_palindrome_factors_l209_20973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_preserves_inequality_l209_20972

theorem odd_power_preserves_inequality (n : ℕ) :
  (∀ a b : ℝ, a > b → a^n > b^n) ↔ Odd n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_preserves_inequality_l209_20972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_assignments_l209_20953

-- Define the types for teachers, cities, and subjects
inductive Teacher : Type
| Zhang
| Li
| Wang

inductive City : Type
| Beijing
| Shanghai
| Shenzhen

inductive Subject : Type
| Math
| Chinese
| English

-- Define the functions for assigning cities and subjects to teachers
variable (city : Teacher → City)
variable (subject : Teacher → Subject)

-- State the theorem
theorem teacher_assignments :
  (city Teacher.Zhang ≠ City.Beijing) →
  (city Teacher.Li ≠ City.Shanghai) →
  (∀ t : Teacher, city t = City.Beijing → subject t ≠ Subject.English) →
  (∀ t : Teacher, city t = City.Shanghai → subject t = Subject.Math) →
  (subject Teacher.Li ≠ Subject.Chinese) →
  (city Teacher.Zhang = City.Shanghai ∧ subject Teacher.Zhang = Subject.Math) ∧
  (city Teacher.Wang = City.Beijing ∧ subject Teacher.Wang = Subject.Chinese) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_assignments_l209_20953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boiling_point_fahrenheit_l209_20926

-- Define the conversion function from Celsius to Fahrenheit
noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ := (9/5) * c + 32

-- State the theorem
theorem water_boiling_point_fahrenheit :
  let water_boiling_celsius : ℝ := 100
  let ice_melting_fahrenheit : ℝ := 32
  let ice_melting_celsius : ℝ := 0
  let known_celsius : ℝ := 40
  let known_fahrenheit : ℝ := 104
  celsius_to_fahrenheit known_celsius = known_fahrenheit →
  celsius_to_fahrenheit water_boiling_celsius = 212 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boiling_point_fahrenheit_l209_20926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l209_20985

theorem polynomial_identity (P : Polynomial ℝ) 
  (h1 : P.eval 0 = 0) 
  (h2 : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) : 
  P = Polynomial.X := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l209_20985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_at_negative_three_l209_20913

-- Define a power function
noncomputable def power_function (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

-- Theorem statement
theorem power_function_value_at_negative_three 
  (f : ℝ → ℝ) (a : ℝ) 
  (h1 : f = power_function a) 
  (h2 : f 2 = 8) : 
  f (-3) = -27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_at_negative_three_l209_20913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zeros_imply_a_range_l209_20951

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (a + 1/2) * x^2 + (a^2 + a) * x - (1/2) * a^2 + 1/2

theorem function_zeros_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) →
  a > -7/2 ∧ a < -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zeros_imply_a_range_l209_20951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_BEIH_l209_20994

noncomputable section

-- Define the square ABCD
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (4, 0)
def D : ℝ × ℝ := (4, 4)

-- Define point E as the midpoint of AB
def E : ℝ × ℝ := (0, 2)

-- Define point F
def F : ℝ × ℝ := (3, 0)

-- Define the lines AF and DE
noncomputable def line_AF (x : ℝ) : ℝ := -4/3 * x + 4
noncomputable def line_DE (x : ℝ) : ℝ := 1/2 * x + 2

-- Define the intersection point I
def I : ℝ × ℝ := (6/7, 20/7)

-- Define the diagonal line BD
noncomputable def line_BD (x : ℝ) : ℝ := x

-- Define the intersection point H
def H : ℝ × ℝ := (3/7, 3/7)

-- State the theorem
theorem area_of_BEIH : 
  let points := [B, E, I, H]
  let area := (1/2) * abs (
    (points[0].1 * points[1].2 + points[1].1 * points[2].2 + points[2].1 * points[3].2 + points[3].1 * points[0].2) -
    (points[0].2 * points[1].1 + points[1].2 * points[2].1 + points[2].2 * points[3].1 + points[3].2 * points[0].1)
  )
  area = 3/49 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_BEIH_l209_20994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alexs_math_problem_l209_20921

theorem alexs_math_problem (y : ℤ) (h : (y - 8) / 5 = 79) : 
  (y - 5 : ℚ) / 8 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alexs_math_problem_l209_20921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_term_condition_case1_result_case2_result_l209_20946

/-- Represents an arithmetic sequence --/
structure ArithmeticSequence where
  e : ℝ  -- first term
  u : ℝ  -- last term
  n : ℕ  -- number of terms
  h : n > 1

/-- Checks if a given number k can be a term in the arithmetic sequence --/
def isTermInSequence (seq : ArithmeticSequence) (k : ℝ) : Prop :=
  ∃ (p q : ℕ), q ∣ (seq.n - 1) ∧ (k - seq.e) / (seq.u - seq.e) = p / q

theorem arithmetic_sequence_term_condition (seq : ArithmeticSequence) (k : ℝ) :
  (isTermInSequence seq k ↔ 
    ∃ (m : ℕ), 1 < m ∧ m < seq.n ∧ k = seq.e + (m - 1) * ((seq.u - seq.e) / (seq.n - 1))) :=
  sorry

/-- Specific cases --/
def case1 : ArithmeticSequence := ⟨1, 1000, 100, by norm_num⟩

noncomputable def case2 : ArithmeticSequence := 
  ⟨81 * Real.sqrt 2 - 64 * Real.sqrt 3, 54 * Real.sqrt 2 - 28 * Real.sqrt 3, 100, by norm_num⟩

theorem case1_result : ¬ isTermInSequence case1 343 := sorry

theorem case2_result : isTermInSequence case2 (69 * Real.sqrt 2 - 48 * Real.sqrt 3) := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_term_condition_case1_result_case2_result_l209_20946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neither_is_correct_problem_answer_is_correct_l209_20947

/-- Represents the possible answers to the question --/
inductive Answer
  | None
  | Neither
  | Both
  | Each

/-- Represents a family with two parents --/
structure Family :=
  (parent1_speaks_english : Bool)
  (parent2_speaks_english : Bool)

/-- Determines the correct answer based on the family's language situation --/
def determine_answer (f : Family) : Answer :=
  match f.parent1_speaks_english, f.parent2_speaks_english with
  | false, false => Answer.Neither
  | _, _ => Answer.Both  -- This is just a placeholder for other cases

/-- Theorem stating that if neither parent speaks English, the answer is "Neither" --/
theorem neither_is_correct (f : Family) :
  f.parent1_speaks_english = false ∧ f.parent2_speaks_english = false →
  determine_answer f = Answer.Neither := by
  intro h
  simp [determine_answer]
  rw [h.1, h.2]

/-- The specific family situation in the problem --/
def problem_family : Family :=
  { parent1_speaks_english := false
  , parent2_speaks_english := false }

/-- The correct answer for the problem --/
def correct_answer : Answer := Answer.Neither

/-- Theorem stating that the answer for the problem family is correct --/
theorem problem_answer_is_correct :
  determine_answer problem_family = correct_answer := by
  simp [determine_answer, problem_family, correct_answer]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_neither_is_correct_problem_answer_is_correct_l209_20947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_companion_point_on_unit_circle_companion_point_symmetry_l209_20935

noncomputable def companion_point (x y : ℝ) : ℝ × ℝ :=
  if x = 0 ∧ y = 0 then (0, 0)
  else (y / (x^2 + y^2), -x / (x^2 + y^2))

theorem companion_point_on_unit_circle (x y : ℝ) :
  x^2 + y^2 = 1 → (let (x', y') := companion_point x y; x'^2 + y'^2 = 1) := by
  sorry

theorem companion_point_symmetry (x y : ℝ) :
  let (x1, y1) := companion_point x y
  let (x2, y2) := companion_point x (-y)
  x1 = -x2 ∧ y1 = y2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_companion_point_on_unit_circle_companion_point_symmetry_l209_20935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_values_l209_20970

/-- Two lines are parallel if and only if their slopes are equal or both undefined -/
def parallel_lines (m1 m2 : ℝ → Prop) : Prop :=
  (∀ a, m1 a ↔ m2 a) ∨ (∀ a, ¬m1 a ∧ ¬m2 a)

/-- The slope of the first line (x + a²y + 6 = 0) -/
def slope1 (a : ℝ) : ℝ → Prop :=
  λ m ↦ m = -1/a^2 ∧ a ≠ 0

/-- The slope of the second line ((a-2)x + 3ay + 2a = 0) -/
def slope2 (a : ℝ) : ℝ → Prop :=
  λ m ↦ m = -(a-2)/(3*a) ∧ a ≠ 0

/-- The main theorem: if the two lines are parallel, then a = 0 or a = -1 -/
theorem parallel_lines_imply_a_values :
  ∀ a : ℝ, parallel_lines (slope1 a) (slope2 a) → a = 0 ∨ a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_values_l209_20970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_with_inequality_l209_20976

/-- The distance from a point (x, y) to the line ax + by + c = 0 -/
noncomputable def distancePointToLine (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

theorem point_on_line_with_inequality (m : ℝ) :
  distancePointToLine m 3 4 (-3) 1 = 4 ∧ 2 * m + 3 < 3 → m = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_with_inequality_l209_20976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_touchdowns_per_game_l209_20904

/-- Represents the number of touchdowns James scores per game -/
def touchdowns_per_game : ℕ → ℕ := sorry

/-- The number of games in the season -/
def games_in_season : ℕ := 15

/-- The number of points per touchdown -/
def points_per_touchdown : ℕ := 6

/-- The number of 2-point conversions James scores in the season -/
def two_point_conversions : ℕ := 6

/-- The old record for points in a season -/
def old_record : ℕ := 300

/-- The number of points James scored above the old record -/
def points_above_record : ℕ := 72

theorem james_touchdowns_per_game :
  ∃ (t : ℕ), 
    touchdowns_per_game t = 4 ∧
    (touchdowns_per_game t * points_per_touchdown * games_in_season) + (two_point_conversions * 2) = old_record + points_above_record :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_touchdowns_per_game_l209_20904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l209_20963

theorem problem_solution :
  ((-1 : ℝ)^0 + (27 : ℝ)^(1/3) + Real.sqrt 4 + abs (Real.sqrt 3 - 2) = 8 - Real.sqrt 3) ∧
  (∃ x : ℝ, x / (x + 2) + 1 / (x - 2) = 1 ∧ x = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l209_20963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_t_values_l209_20993

/-- The equation of the circle parametrized by m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*m*y + m^2 - 2*m - 2 = 0

/-- The circle C with center at (1, 1) -/
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}

/-- The line that intersects circle C -/
def intersecting_line (t : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1 + p.2 + t = 0}

/-- Theorem stating the possible values of t -/
theorem intersection_t_values :
  ∀ (t : ℝ) (A B : ℝ × ℝ),
    A ∈ circle_C → B ∈ circle_C →
    A ∈ intersecting_line t → B ∈ intersecting_line t →
    (A.1 - 1) * (B.1 - 1) + (A.2 - 1) * (B.2 - 1) = 0 →
    t = -3 ∨ t = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_t_values_l209_20993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l209_20927

/-- The circle with equation x^2 + y^2 = 4 -/
def myCircle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- The point P(√3, 1) -/
noncomputable def point_P : ℝ × ℝ := (Real.sqrt 3, 1)

/-- The line with equation √3x + y - 4 = 0 -/
noncomputable def tangent_line : Set (ℝ × ℝ) := {p | Real.sqrt 3 * p.1 + p.2 - 4 = 0}

/-- Theorem: The line √3x + y - 4 = 0 passes through point P(√3, 1) and is tangent to the circle x^2 + y^2 = 4 -/
theorem tangent_line_proof : 
  point_P ∈ tangent_line ∧ 
  point_P ∈ myCircle ∧ 
  ∀ p ∈ myCircle, p ∈ tangent_line → p = point_P :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l209_20927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_growth_rate_l209_20932

/-- A tree's growth over time -/
structure TreeGrowth where
  initial_height : ℝ
  initial_age : ℝ
  final_height : ℝ
  final_age : ℝ

/-- Calculate the growth rate of a tree -/
noncomputable def growth_rate (t : TreeGrowth) : ℝ :=
  (t.final_height - t.initial_height) / (t.final_age - t.initial_age)

theorem tree_growth_rate (t : TreeGrowth) 
  (h1 : t.initial_height = 5)
  (h2 : t.initial_age = 1)
  (h3 : t.final_height = 23)
  (h4 : t.final_age = 7) :
  growth_rate t = 3 := by
  sorry

#eval "Tree growth rate theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_growth_rate_l209_20932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_geometric_mean_l209_20962

/-- An arithmetic sequence with first term 1 and common difference d -/
def arithmetic_sequence (d : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => arithmetic_sequence d n + d

theorem arithmetic_sequence_geometric_mean (d : ℝ) (h : d ≠ 0) :
  (arithmetic_sequence d 2) ^ 2 = (arithmetic_sequence d 1) * (arithmetic_sequence d 4) →
  d = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_geometric_mean_l209_20962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l209_20991

/-- Line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

/-- Curve C in the xy-plane -/
def curve_C (x y : ℝ) : Prop := x^2 + y^2/3 = 1

/-- Distance between a point (x, y) and line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x - y + 3| / Real.sqrt 2

/-- Theorem stating the range of distances between curve C and line l -/
theorem distance_range : 
  ∃ (d_min d_max : ℝ), d_min = Real.sqrt 2 / 2 ∧ d_max = 5 * Real.sqrt 2 / 2 ∧
  ∀ (x y : ℝ), curve_C x y → 
    d_min ≤ distance_to_line x y ∧ distance_to_line x y ≤ d_max :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l209_20991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_greater_equal_threshold_l209_20901

noncomputable def numbers : List ℝ := [1.4, 9/10, 1.2, 0.5, 13/10]

def threshold : ℝ := 1.1

theorem smallest_greater_equal_threshold :
  1.2 = (numbers.filter (λ x => x ≥ threshold)).minimum?.get! := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_greater_equal_threshold_l209_20901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l209_20954

/-- The function m(x) defined in the problem -/
noncomputable def m (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- The main theorem -/
theorem problem_solution (a b c d : ℝ) 
  (h1 : ¬(c = 0 ∧ d = 0))
  (h2 : ∃ x : ℝ, x = m a b c d (m a b c d x) ∧ x ≠ m a b c d x) :
  a + d = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l209_20954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pdf_Y_is_correct_pdf_Y_integrates_to_one_l209_20949

/-- The probability density function of Y = cos X, where X is uniformly distributed in (0, 2π) -/
noncomputable def pdf_Y (y : ℝ) : ℝ :=
  if -1 < y ∧ y < 1 then 1 / (Real.pi * Real.sqrt (1 - y^2)) else 0

/-- X is a random variable uniformly distributed in (0, 2π) -/
def X : Type := ℝ

/-- Y is defined as cos X -/
noncomputable def Y : X → ℝ := fun x => Real.cos x

/-- Theorem stating that pdf_Y is the correct probability density function for Y -/
theorem pdf_Y_is_correct : 
  ∀ (y : ℝ), pdf_Y y = if -1 < y ∧ y < 1 then 1 / (Real.pi * Real.sqrt (1 - y^2)) else 0 :=
by
  sorry

/-- The integral of pdf_Y over its domain equals 1 -/
theorem pdf_Y_integrates_to_one : 
  ∫ y in (-1)..(1), pdf_Y y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pdf_Y_is_correct_pdf_Y_integrates_to_one_l209_20949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l209_20923

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x ^ 3 - Real.sin x + 2 * (Real.sin (x / 2) - Real.cos (x / 2)) ^ 2

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l209_20923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_fourth_l209_20989

/-- Given an angle θ with vertex at the origin and initial side along the positive x-axis,
    if the terminal side passes through the point (1,2), then tan(θ + π/4) = -3. -/
theorem tan_theta_plus_pi_fourth (θ : Real) :
  (∃ (x y : Real), x = 1 ∧ y = 2 ∧ y / x = Real.tan θ) →
  Real.tan (θ + π/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_fourth_l209_20989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_repeated_digits_approx_l209_20920

-- Define the total number of five-digit numbers where the first digit is not zero
def total_numbers : ℕ := 90000

-- Define the number of five-digit numbers with all unique digits
def unique_digit_numbers : ℕ := 9 * 9 * 8 * 7 * 6

-- Define the number of five-digit numbers with at least one repeated digit
def repeated_digit_numbers : ℕ := total_numbers - unique_digit_numbers

-- Define the percentage of numbers with at least one repeated digit
noncomputable def percentage_repeated : ℝ := (repeated_digit_numbers : ℝ) / (total_numbers : ℝ) * 100

-- Theorem statement
theorem percentage_of_repeated_digits_approx :
  abs (percentage_repeated - 69.8) < 0.1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_repeated_digits_approx_l209_20920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_property_l209_20958

-- Define the line l
noncomputable def line_l (x : ℝ) : ℝ := Real.sqrt 3 * (x - 2)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define point M
def point_M : ℝ × ℝ := (2, 0)

-- Define the intersection points A and B
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ

-- A and B are on line l
axiom A_on_l : (A.2 : ℝ) = line_l A.1
axiom B_on_l : (B.2 : ℝ) = line_l B.1

-- A and B are on curve C
axiom A_on_C : curve_C A.1 A.2
axiom B_on_C : curve_C B.1 B.2

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem to prove
theorem intersection_point_property :
  |1 / distance point_M A - 1 / distance point_M B| = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_property_l209_20958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_plus_cos_over_pi_interval_l209_20988

open Real MeasureTheory Interval

theorem integral_sin_plus_cos_over_pi_interval :
  ∫ x in (-π/2)..(π/2), sin x + cos x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_plus_cos_over_pi_interval_l209_20988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_zero_l209_20960

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) + x^2

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_zero_l209_20960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_zero_sufficient_not_necessary_l209_20957

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = x^2 + a|x| + b -/
def f (a b : ℝ) : ℝ → ℝ := λ x ↦ x^2 + a * abs x + b

/-- "a=0" is a sufficient but not necessary condition for "f is even" -/
theorem a_zero_sufficient_not_necessary (a b : ℝ) :
  (a = 0 → IsEven (f a b)) ∧ ¬(IsEven (f a b) → a = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_zero_sufficient_not_necessary_l209_20957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_term_is_50_l209_20966

def sequenceCount : ℕ → ℕ
  | 0 => 2
  | n + 1 => 
    if n < 9 then sequenceCount n + 2
    else sequenceCount n + 3

theorem thirtieth_term_is_50 : sequenceCount 29 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_term_is_50_l209_20966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_german_team_goals_l209_20986

def journalist1 (x : ℕ) : Prop := x > 10 ∧ x < 17
def journalist2 (x : ℕ) : Prop := x > 11 ∧ x < 18
def journalist3 (x : ℕ) : Prop := x % 2 = 1

def exactly_two_correct (x : ℕ) : Prop :=
  (journalist1 x ∧ journalist2 x ∧ ¬journalist3 x) ∨
  (journalist1 x ∧ ¬journalist2 x ∧ journalist3 x) ∨
  (¬journalist1 x ∧ journalist2 x ∧ journalist3 x)

theorem german_team_goals :
  ∀ x : ℕ, exactly_two_correct x ↔ x ∈ ({11, 12, 14, 16, 17} : Set ℕ) :=
by
  intro x
  apply Iff.intro
  · intro h
    -- Proof of forward direction
    sorry
  · intro h
    -- Proof of backward direction
    sorry

#check german_team_goals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_german_team_goals_l209_20986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_relation_in_triangle_l209_20982

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if cos C = 2/3 and a = 3b, then cos A = -√6/6 -/
theorem cosine_relation_in_triangle (a b c : ℝ) (A B C : Real) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  A > 0 ∧ B > 0 ∧ C > 0 →  -- Positive angles
  A + B + C = Real.pi →  -- Angle sum in a triangle
  a * Real.sin B = b * Real.sin A →  -- Law of sines
  b * Real.sin C = c * Real.sin B →  -- Law of sines
  c * Real.sin A = a * Real.sin C →  -- Law of sines
  Real.cos C = 2/3 →
  a = 3*b →
  Real.cos A = -Real.sqrt 6 / 6 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_relation_in_triangle_l209_20982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_cos_symmetry_l209_20961

open Real

-- Define the original cosine function
noncomputable def original_cos (x : ℝ) : ℝ := cos x

-- Define the transformed cosine function
noncomputable def transformed_cos (x : ℝ) : ℝ := cos ((x + π) / 2)

-- Theorem stating that x = -π is an axis of symmetry for the transformed function
theorem transformed_cos_symmetry :
  ∀ (x : ℝ), transformed_cos ((-π) + x) = transformed_cos ((-π) - x) := by
  sorry

-- Assumption that x = kπ are axes of symmetry for the original cosine function
axiom original_cos_symmetry (k : ℤ) :
  ∀ (x : ℝ), original_cos (k * π + x) = original_cos (k * π - x)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_cos_symmetry_l209_20961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequality_l209_20971

/-- Given a triangle ABC with side lengths a, b, c, and area Δ, 
    and positive real numbers l, m, n, prove the inequality. -/
theorem triangle_area_inequality 
  (a b c : ℝ) 
  (l m n : ℝ) 
  (Δ : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_pos_l : 0 < l) 
  (h_pos_m : 0 < m) 
  (h_pos_n : 0 < n) 
  (h_triangle : Δ = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  Δ ≤ (a * b * c)^(2/3) * (l * m + m * n + n * l) / (4 * Real.sqrt 3 * (l * m * n)^(2/3)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequality_l209_20971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strict_decreasing_interval_l209_20907

noncomputable section

-- Define the function
def f (x : ℝ) : ℝ := 3 * Real.cos (Real.pi / 4 + x)

-- Define the strict decreasing interval
def is_strict_decreasing_interval (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

-- Theorem statement
theorem f_strict_decreasing_interval (k : ℤ) :
  is_strict_decreasing_interval (2 * ↑k * Real.pi - Real.pi / 4) (2 * ↑k * Real.pi + 3 * Real.pi / 4) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strict_decreasing_interval_l209_20907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_l209_20922

def f (x : ℝ) := (x + 1) * (x - 1) + (x - 1) * (x - 2) + (x - 2) * (x + 1)

theorem f_zeros :
  ∃ (z₁ z₂ : ℝ), z₁ ∈ Set.Ioo (-1) 1 ∧ z₂ ∈ Set.Ioo 1 2 ∧
  f z₁ = 0 ∧ f z₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = z₁ ∨ x = z₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_l209_20922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equations_solutions_l209_20915

theorem quadratic_equations_solutions :
  (∃ x : ℝ, 2 * x^2 - 2 * Real.sqrt 2 * x + 1 = 0) ∧
  (∃ x : ℝ, x * (2 * x - 5) = 4 * x - 10) :=
by
  -- First equation: 2x^2 - 2√2x + 1 = 0
  have h1 : ∃ x : ℝ, 2 * x^2 - 2 * Real.sqrt 2 * x + 1 = 0 := by
    use Real.sqrt 2 / 2
    sorry

  -- Second equation: x(2x - 5) = 4x - 10
  have h2 : ∃ x : ℝ, x * (2 * x - 5) = 4 * x - 10 := by
    use 5/2
    sorry

  exact ⟨h1, h2⟩

#check quadratic_equations_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equations_solutions_l209_20915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_between_one_and_two_l209_20983

theorem mn_between_one_and_two 
  (M N : ℝ) 
  (h1 : Real.log N = 2 * Real.log M) 
  (h2 : M ≠ N) 
  (h3 : M * N > 0) 
  (h4 : M ≠ 1) 
  (h5 : N ≠ 1) : 
  1 < M * N ∧ M * N < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_between_one_and_two_l209_20983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_approximation_l209_20928

-- Define the constants from the problem
def a : ℝ := 38472.56
def b : ℝ := 28384.29
def c : ℝ := 2765
def d : ℝ := 5238

-- Define the equation
def equation (x : ℝ) : Prop :=
  5 * x = Real.sqrt (a + b - (7/11) * (c + d))

-- State the theorem
theorem solution_approximation :
  ∃ x : ℝ, equation x ∧ |x - 49.7044| < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_approximation_l209_20928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_hand_distance_45_minutes_l209_20902

/-- The distance traveled by the tip of a clock's second hand -/
noncomputable def second_hand_distance (length : ℝ) (minutes : ℕ) : ℝ :=
  2 * Real.pi * length * (minutes : ℝ)

theorem second_hand_distance_45_minutes :
  second_hand_distance 9 45 = 810 * Real.pi :=
by
  -- Unfold the definition of second_hand_distance
  unfold second_hand_distance
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_hand_distance_45_minutes_l209_20902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_chord_length_l209_20925

-- Define the line
def line (x y b : ℝ) : Prop := y = x + b

-- Define the circle
def circle' (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y + 1 = 0

theorem tangent_and_chord_length :
  ∃ (b : ℝ),
    (∀ x y : ℝ, line x y b ∧ circle' x y → b = 2 + Real.sqrt 2 ∨ b = 2 - Real.sqrt 2) ∧
    (∀ x₁ y₁ x₂ y₂ : ℝ, 
      line x₁ y₁ 1 ∧ line x₂ y₂ 1 ∧ circle' x₁ y₁ ∧ circle' x₂ y₂ → 
      Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_chord_length_l209_20925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l209_20987

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin x)^2 * Real.log (Real.sin x) / Real.log 2 + 2 * (Real.cos x)^2 * Real.log (Real.cos x) / Real.log 2

def is_in_domain (x : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < x ∧ x < Real.pi / 2 + 2 * k * Real.pi

theorem f_properties :
  (∀ t : ℝ, is_in_domain (Real.pi/4 + t) → is_in_domain (Real.pi/4 - t) → 
    f (Real.pi/4 + t) = f (Real.pi/4 - t)) ∧
  (∃ m : ℝ, (∀ x : ℝ, is_in_domain x → f x ≥ m) ∧ (∃ x : ℝ, is_in_domain x ∧ f x = m) ∧ m = -1) ∧
  (∀ k : ℤ, ∀ x y : ℝ, 2 * k * Real.pi < x ∧ x < y ∧ y < Real.pi/4 + 2 * k * Real.pi → 
    f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l209_20987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l209_20924

/-- Quadratic function passing through (0, 1) with a unique zero at -1 -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- F(x) = f(x) - kx -/
def F (k : ℝ) (x : ℝ) : ℝ := f x - k*x

/-- Minimum value of F(x) for x ∈ [-2, 2] -/
noncomputable def g (k : ℝ) : ℝ :=
  if k ≤ -2 then k + 3
  else if k ≤ 6 then -(k^2 - 4*k) / 4
  else 9 - 2*k

theorem quadratic_function_properties :
  (∀ x, f x = x^2 + 2*x + 1) ∧
  (∀ k, ∀ x ∈ Set.Icc (-2) 2, F k x ≥ g k) ∧
  (∀ k, ∃ x ∈ Set.Icc (-2) 2, F k x = g k) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l209_20924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_radius_l209_20999

/-- A line with equation √3x - 2y = 0 -/
def line_eq (x y : ℝ) : Prop := Real.sqrt 3 * x - 2 * y = 0

/-- A circle with center (4, 0) and radius r -/
def circle_eq (x y r : ℝ) : Prop := (x - 4)^2 + y^2 = r^2

/-- The distance from a point (a, b) to the line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (a b A B C : ℝ) : ℝ :=
  abs (A * a + B * b + C) / Real.sqrt (A^2 + B^2)

/-- The theorem stating that if the line √3x - 2y = 0 is tangent to the circle (x-4)² + y² = r² (r > 0),
    then r = (4√21)/7 -/
theorem tangent_line_circle_radius : 
  ∀ r : ℝ, r > 0 → 
  (∃ x y : ℝ, line_eq x y ∧ circle_eq x y r) →
  (∀ x y : ℝ, line_eq x y → circle_eq x y r → 
    distance_point_to_line 4 0 (Real.sqrt 3) (-2) 0 = r) →
  r = 4 * Real.sqrt 21 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_radius_l209_20999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l209_20910

/-- Given vectors a, b, c, and p satisfying the condition
    ||p - b|| = 3 ||p - a||, prove that p is equidistant from
    (9/2)a + (1/4)b and c. -/
theorem equidistant_point (a b c p : EuclideanSpace ℝ (Fin 3)) 
  (h : ‖p - b‖ = 3 * ‖p - a‖) :
  ‖p - ((9/2 : ℝ) • a + (1/4 : ℝ) • b)‖ = ‖p - c‖ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l209_20910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_circle_center_l209_20933

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := 3*x + 2*y + 1 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2*x - 3*y + 3 = 0

-- Theorem statement
theorem perpendicular_line_through_circle_center :
  ∃ (x₀ y₀ : ℝ), 
    (∀ x y, circle_C x y ↔ (x - x₀)^2 + (y - y₀)^2 = 4) ∧
    perp_line x₀ y₀ ∧
    (∀ x y, line_l x y → (y - y₀ = -(3/2) * (x - x₀) ↔ perp_line x y)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_circle_center_l209_20933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l209_20900

/-- The function f(x) = sin(πx/2 + π/3) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 2 * x + Real.pi / 3)

/-- Theorem stating that the minimum distance between x₁ and x₂ is 2 -/
theorem min_distance_theorem :
  (∀ x : ℝ, ∃ x₁ x₂ : ℝ, f x₁ ≤ f x ∧ f x ≤ f x₂) →
  (∃ d : ℝ, d = 2 ∧ ∀ x₁ x₂ : ℝ, f x₁ ≤ f x₂ → d ≤ |x₁ - x₂|) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l209_20900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_complete_expression_l209_20906

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 - 2*x - 1
  else if x < 0 then -x^2 - 2*x + 1
  else 0

-- State the theorem
theorem odd_function_complete_expression :
  (∀ x : ℝ, f (-x) = -f x) ∧  -- f is odd
  (∀ x : ℝ, x > 0 → f x = x^2 - 2*x - 1) →  -- given condition for x > 0
  ∀ x : ℝ, f x = 
    if x > 0 then x^2 - 2*x - 1
    else if x < 0 then -x^2 - 2*x + 1
    else 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_complete_expression_l209_20906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_maxima_l209_20941

/-- Ellipse properties and related maxima -/
theorem ellipse_properties_and_maxima 
  (a b : ℝ) 
  (h_positive : a > b ∧ b > 0) 
  (h_eccentricity : a * (Real.sqrt 2 / 2) = Real.sqrt (a^2 - b^2))
  (h_distance : Real.sqrt (b^2 + (a^2 - b^2)) = Real.sqrt 2) :
  ∃ (max_distance max_area : ℝ),
    (∀ x y : ℝ, x^2 / 2 + y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
    (max_distance = 4 * Real.sqrt 3 / 3) ∧
    (∀ m : ℝ, ∀ A B : ℝ × ℝ, 
      (A.1^2 / 2 + A.2^2 = 1) ∧ (B.1^2 / 2 + B.2^2 = 1) ∧ 
      A.2 = A.1 + m ∧ B.2 = B.1 + m →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ max_distance) ∧
    (max_area = Real.sqrt 2 / 2) ∧
    (∀ m : ℝ, ∀ A B : ℝ × ℝ,
      (A.1^2 / 2 + A.2^2 = 1) ∧ (B.1^2 / 2 + B.2^2 = 1) ∧
      A.2 = A.1 + m ∧ B.2 = B.1 + m →
      abs ((A.1 * B.2 - A.2 * B.1) / 2) ≤ max_area) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_maxima_l209_20941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_at_two_points_l209_20984

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the distance between two points in a 2D plane -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Determines the number of intersection points between two circles -/
noncomputable def intersectionPoints (c1 c2 : Circle) : ℕ :=
  let d := distance c1.center c2.center
  if d > c1.radius + c2.radius then 0
  else if d < abs (c1.radius - c2.radius) then 0
  else if d = c1.radius + c2.radius then 1
  else if d = abs (c1.radius - c2.radius) then 1
  else if c1.center = c2.center ∧ c1.radius = c2.radius then 3
  else 2

theorem circles_intersect_at_two_points :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (8, 0)
  let C_A : Circle := ⟨A, 3⟩
  let C_B : Circle := ⟨B, 6⟩
  intersectionPoints C_A C_B = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_at_two_points_l209_20984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l209_20981

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 3]

theorem det_B_squared_minus_3B : Matrix.det (B^2 - 3 • B) = -8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l209_20981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_range_l209_20945

/-- The theorem states that given a line y = √3x - m intersecting a circle x² + y² = 9
    at two distinct points M and N, where |MN| ≥ √6 |OM + ON| (O being the origin),
    the range of m is -6√7/7 ≤ m ≤ 6√7/7 -/
theorem line_circle_intersection_range (m : ℝ) : 
  (∃ (M N : ℝ × ℝ), 
    M.1^2 + M.2^2 = 9 ∧ 
    N.1^2 + N.2^2 = 9 ∧ 
    M.2 = Real.sqrt 3 * M.1 - m ∧ 
    N.2 = Real.sqrt 3 * N.1 - m ∧ 
    M ≠ N ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 6 * ((M.1 + N.1)^2 + (M.2 + N.2)^2) / 4) →
  -6 * Real.sqrt 7 / 7 ≤ m ∧ m ≤ 6 * Real.sqrt 7 / 7 := by
  sorry

#check line_circle_intersection_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_range_l209_20945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rectangle_area_l209_20919

/-- The area of the smallest rectangle that can contain a circle with radius r and a square inscribed within this circle. -/
noncomputable def area_of_smallest_rectangle_containing_circle_and_inscribed_square (r : ℝ) : ℝ :=
  (2 * r) * (2 * r)

theorem smallest_rectangle_area (r : ℝ) (h : r = 5) : ∃ A : ℝ,
  A = (2 * r) * (2 * r) ∧
  A = area_of_smallest_rectangle_containing_circle_and_inscribed_square r :=
by
  let A := (2 * r) * (2 * r)
  use A
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rectangle_area_l209_20919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l209_20979

theorem line_tangent_to_circle :
  ∀ θ : ℝ, 
  (∀ x y : ℝ, x * Real.cos θ + y * Real.sin θ = 1 → x^2 + y^2 = 1) ∧
  (∃ x y : ℝ, x * Real.cos θ + y * Real.sin θ = 1 ∧ x^2 + y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l209_20979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_cosine_sum_l209_20967

theorem min_value_of_cosine_sum (x y z : Real) (hx : x ∈ Set.Icc 0 Real.pi) (hy : y ∈ Set.Icc 0 Real.pi) (hz : z ∈ Set.Icc 0 Real.pi) :
  ∃ (m : Real), m = -1 ∧ ∀ x' y' z', x' ∈ Set.Icc 0 Real.pi → y' ∈ Set.Icc 0 Real.pi → z' ∈ Set.Icc 0 Real.pi →
    Real.cos (x' - y') + Real.cos (y' - z') + Real.cos (z' - x') ≥ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_cosine_sum_l209_20967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_every_large_integer_in_set_l209_20969

/-- A set with the given closure property -/
def ClosedSet (S : Set ℕ) : Prop :=
  ∀ x y z, x ∈ S → y ∈ S → z ∈ S → (x + y + z) ∈ S

theorem every_large_integer_in_set
  (a b : ℕ)
  (ha : a > 0)
  (hb : b > 0)
  (hab_coprime : Nat.Coprime a b)
  (hab_parity : a % 2 ≠ b % 2)
  (S : Set ℕ)
  (ha_in_S : a ∈ S)
  (hb_in_S : b ∈ S)
  (hS_closed : ClosedSet S) :
  ∀ n : ℕ, n > 2 * a * b → n ∈ S :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_every_large_integer_in_set_l209_20969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_21_l209_20903

-- Define the sequence
def a : ℕ → ℕ
  | 0 => 1  -- Add this case for 0
  | 1 => 1
  | (n + 2) => a (n + 1) + 2 * (n + 1)

-- State the theorem
theorem a_5_equals_21 : a 5 = 21 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_21_l209_20903
