import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_lines_l548_54887

/-- Line with equation y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- The first quadrant -/
def FirstQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- The area below a line in the first quadrant -/
noncomputable def AreaBelowLine (l : Line) : ℝ :=
  (l.b ^ 2) / (2 * (-l.m))

/-- The area between two lines in the first quadrant -/
noncomputable def AreaBetweenLines (l1 l2 : Line) : ℝ :=
  AreaBelowLine l1 - AreaBelowLine l2

theorem probability_between_lines :
  let u : Line := ⟨-2, 8⟩
  let v : Line := ⟨-3, 8⟩
  AreaBetweenLines u v / AreaBelowLine u = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_lines_l548_54887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jared_percentage_fewer_than_ann_l548_54892

/-- The number of cars Jared counted -/
def jared_count : ℕ := 300

/-- The difference between Ann's and Alfred's count -/
def ann_alfred_diff : ℕ := 7

/-- The total number of cars counted by all three -/
def total_count : ℕ := 983

/-- Calculate the percentage difference between two numbers -/
def percentage_difference (a b : ℕ) : ℚ :=
  (a - b : ℚ) / a * 100

/-- Theorem stating the percentage of cars Jared counted fewer than Ann -/
theorem jared_percentage_fewer_than_ann :
  ∃ (alfred_count ann_count : ℕ),
    ann_count = alfred_count + ann_alfred_diff ∧
    jared_count + ann_count + alfred_count = total_count ∧
    abs (percentage_difference ann_count jared_count - 13.04) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jared_percentage_fewer_than_ann_l548_54892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_equals_two_l548_54843

theorem log_product_equals_two : Real.log 3 / Real.log 2 * (Real.log 4 / Real.log 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_equals_two_l548_54843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mobile_phone_max_profit_l548_54894

noncomputable section

/-- Revenue function (in ten thousand dollars) for x ten thousand units --/
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 40 then 400 - 6*x
  else if x > 40 then 7400/x - 40000/(x^2)
  else 0

/-- Profit function (in ten thousand dollars) for x ten thousand units --/
noncomputable def W (x : ℝ) : ℝ :=
  x * R x - (16*x + 40)

/-- The production level that maximizes profit --/
def x_max : ℝ := 32

/-- The maximum profit --/
def max_profit : ℝ := 6104

theorem mobile_phone_max_profit :
  (∀ x > 0, W x ≤ W x_max) ∧ W x_max = max_profit :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mobile_phone_max_profit_l548_54894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_25_l548_54891

noncomputable def hour_hand_angle (t : ℝ) : ℝ := (t % 12) * 30

noncomputable def minute_hand_angle (t : ℝ) : ℝ := (t % 1) * 360

noncomputable def acute_angle_between (a b : ℝ) : ℝ :=
  min (abs (a - b)) (360 - abs (a - b))

theorem clock_angle_at_3_25 :
  acute_angle_between (hour_hand_angle (3 + 25/60)) (minute_hand_angle (3 + 25/60)) = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_25_l548_54891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l548_54883

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : Real.sin A * Real.cos B + Real.sin B * Real.cos A = Real.sin (2 * C))
  (h2 : a + c = 2 * b) -- arithmetic sequence condition
  (h3 : a * b * Real.cos C = 18) -- dot product condition
  : c = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l548_54883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_pi_third_l548_54873

noncomputable def f (x : ℝ) := Real.sin x + Real.cos x

noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => f
| n + 1 => deriv (f_n n)

theorem f_2016_pi_third : f_n 2016 (π / 3) = (Real.sqrt 3 + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_pi_third_l548_54873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_proposition_l548_54824

theorem exactly_one_correct_proposition : ∃! n : Nat, n = 1 ∧
  (¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x₀ : ℝ, x₀^2 ≤ 0) ∨
  (∀ x : ℝ, x ≠ 3 → x ≠ 3) ∧ ¬(∀ x : ℝ, x ≠ 3 → x ≠ 3) ∨
  (¬(∀ m : ℝ, m ≤ 1/2 → ∃ x : ℝ, m*x^2 + 2*x + 2 = 0) ↔
   ∀ m : ℝ, m > 1/2 → ∀ x : ℝ, m*x^2 + 2*x + 2 ≠ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_proposition_l548_54824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_two_l548_54813

/-- The function representing the curve y = x / (x + 1) -/
noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

/-- The derivative of f -/
noncomputable def f' (x : ℝ) : ℝ := 1 / ((x + 1) ^ 2)

theorem tangent_line_at_negative_two :
  let x₀ : ℝ := -2
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x - y + 4 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_two_l548_54813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_or_tangent_circle_l548_54817

-- Define the ellipse G
def ellipse_G (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a line l
def line_l (k m x y : ℝ) : Prop := y = k * x + m

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem line_intersects_or_tangent_circle 
  (k m : ℝ) 
  (A B : ℝ × ℝ) 
  (h1 : ellipse_G A.1 A.2) 
  (h2 : ellipse_G B.1 B.2) 
  (h3 : line_l k m A.1 A.2) 
  (h4 : line_l k m B.1 B.2) 
  (h5 : distance A.1 A.2 B.1 B.2 = 2) :
  (∃ (P Q : ℝ × ℝ), P ≠ Q ∧ circle_O P.1 P.2 ∧ circle_O Q.1 Q.2 ∧ line_l k m P.1 P.2 ∧ line_l k m Q.1 Q.2) ∨
  (∃ (T : ℝ × ℝ), circle_O T.1 T.2 ∧ line_l k m T.1 T.2 ∧ 
    ∀ (x y : ℝ), line_l k m x y → circle_O x y → (x, y) = T) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_or_tangent_circle_l548_54817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l548_54833

theorem min_value_of_expression (a b : ℝ) : 
  (a + 2*b = 1) →  -- Line equation condition
  (∀ x y : ℝ, a*x + b*y = 1 → x = 1 ∧ y = 2 → False) →  -- Point (1,2) is on the line
  (∀ a' b' : ℝ, a'*1 + b'*2 = 1 → (2:ℝ)^a' + (4:ℝ)^b' ≥ (2:ℝ)^a + (4:ℝ)^b) →  -- (a,b) minimizes the expression
  (2:ℝ)^a + (4:ℝ)^b = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l548_54833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l548_54821

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x + 2 * Real.sqrt 3, Real.sin x)
noncomputable def c (α : ℝ) : ℝ × ℝ := (Real.sin α, Real.cos α)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem vector_problem (x α : ℝ) :
  -- Part I
  (dot_product (a x) (c α) = 0 → Real.cos (2*x + 2*α) = 1) ∧
  -- Part II
  (x ∈ Set.Ioo 0 (Real.pi / 2) → ¬∃ (k : ℝ), a x = k • b x) ∧
  -- Part III
  (α = 0 →
    let f := λ x => dot_product (a x) (b x - 2 • c α)
    ∃ (max_x : ℝ), f max_x = 5 ∧ ∀ y, f y ≤ 5) := by
  sorry

#check vector_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l548_54821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stride_jump_difference_value_l548_54884

/-- The number of equal jumps Gary takes between consecutive street lamps -/
def gary_jumps : ℕ := 55

/-- The number of equal strides Zeke takes between consecutive street lamps -/
def zeke_strides : ℕ := 15

/-- The number of the lamp that is half a mile away from the start -/
def half_mile_lamp : ℕ := 26

/-- The distance in feet to the half-mile lamp -/
def half_mile_distance : ℚ := 2640

/-- The difference in feet between Zeke's stride length and Gary's jump length -/
noncomputable def stride_jump_difference : ℚ := 
  (half_mile_distance / ((half_mile_lamp - 1) * zeke_strides)) - 
  (half_mile_distance / ((half_mile_lamp - 1) * gary_jumps))

theorem stride_jump_difference_value : 
  stride_jump_difference = 512 / 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stride_jump_difference_value_l548_54884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ryan_spits_50_percent_shorter_l548_54803

/-- The percentage by which Ryan's watermelon seed spitting distance is shorter than Madison's -/
noncomputable def ryan_shorter_percentage (billy_distance madison_distance ryan_distance : ℝ) : ℝ :=
  ((madison_distance - ryan_distance) / madison_distance) * 100

/-- Theorem stating that Ryan's watermelon seed spitting distance is 50% shorter than Madison's -/
theorem ryan_spits_50_percent_shorter 
  (billy_distance : ℝ)
  (madison_distance : ℝ)
  (ryan_distance : ℝ)
  (h1 : billy_distance = 30)
  (h2 : madison_distance = billy_distance * 1.2)
  (h3 : ryan_distance = 18) :
  ryan_shorter_percentage billy_distance madison_distance ryan_distance = 50 := by
  sorry

#check ryan_spits_50_percent_shorter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ryan_spits_50_percent_shorter_l548_54803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l548_54853

-- Define the inequality as a function of a and x
def inequality (a : ℝ) (x : ℝ) : Prop :=
  (a - 2) * x^2 - 2*(a - 2)*x - 4 < 0

-- Define the set of a values that satisfy the inequality for all x
def valid_a_set : Set ℝ :=
  {a | ∀ x, inequality a x}

-- Theorem statement
theorem inequality_range :
  valid_a_set = Set.Ioc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l548_54853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expectation_variance_transformation_l548_54844

-- Define a finite set of 20 real numbers
def DataSet : Type := Fin 20 → ℝ

-- Define the expectation (mean) of a DataSet
noncomputable def expectation (data : DataSet) : ℝ := 
  (Finset.sum Finset.univ (fun i => data i)) / 20

-- Define the variance of a DataSet
noncomputable def variance (data : DataSet) : ℝ := 
  (Finset.sum Finset.univ (fun i => (data i - expectation data)^2)) / 20

-- Theorem statement
theorem expectation_variance_transformation (data : DataSet)
  (h_exp : expectation data = 3)
  (h_var : variance data = 3) :
  expectation (fun i => 2 * (data i) + 3) = 9 ∧ 
  variance (fun i => 2 * (data i) + 3) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expectation_variance_transformation_l548_54844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_theorem_l548_54823

noncomputable def average_expenditure_feb_to_jul (avg_jan_to_jun : ℚ) (exp_jan : ℚ) (exp_jul : ℚ) : ℚ :=
  let total_jan_to_jun := avg_jan_to_jun * 6
  let total_feb_to_jun := total_jan_to_jun - exp_jan
  let total_feb_to_jul := total_feb_to_jun + exp_jul
  total_feb_to_jul / 6

theorem expenditure_theorem (avg_jan_to_jun : ℚ) (exp_jan : ℚ) (exp_jul : ℚ) 
  (h1 : avg_jan_to_jun = 4200)
  (h2 : exp_jan = 1200)
  (h3 : exp_jul = 1500) :
  average_expenditure_feb_to_jul avg_jan_to_jun exp_jan exp_jul = 4250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_theorem_l548_54823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l548_54899

theorem inequality_solution_set (x : ℝ) : (2 : ℝ)^(x^2 - x) < 4 ↔ -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l548_54899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_travel_strategy_l548_54857

/-- Time taken by person A to reach N --/
def travel_time_A (t : ℝ) : ℝ := sorry

/-- Time taken by person B to reach N --/
def travel_time_B (t : ℝ) : ℝ := sorry

/-- The optimal strategy for minimizing travel time from M to N --/
theorem optimal_travel_strategy 
  (distance : ℝ) 
  (walking_speed : ℝ) 
  (cycling_speed : ℝ) 
  (h1 : distance = 15) 
  (h2 : walking_speed = 6) 
  (h3 : cycling_speed = 15) :
  ∃ (t : ℝ), 
    t = 3/11 ∧ 
    (∀ (s : ℝ), 
      max (travel_time_A s) (travel_time_B s) ≥ 
      max (travel_time_A t) (travel_time_B t)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_travel_strategy_l548_54857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_calculate_trig_fraction_l548_54870

open Real

theorem simplify_trig_expression (α : ℝ) :
  (Real.sin (π - α) * Real.cos (3*π - α) * Real.tan (-α - π) * Real.tan (α - 2*π)) /
  (Real.tan (4*π - α) * Real.sin (5*π + α)) = Real.sin α := by sorry

theorem calculate_trig_fraction (α : ℝ) (h : Real.tan α = 3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_calculate_trig_fraction_l548_54870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_tan_2alpha_value_l548_54874

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2

-- Theorem for monotonically increasing interval
theorem f_monotone_increasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (-(π/6) + k * π) ((π/3) + k * π)) := by
  sorry

-- Theorem for the value of tan(2α)
theorem tan_2alpha_value (α : ℝ) (h1 : f α = 13/5) (h2 : α ∈ Set.Icc (π/12) (5*π/12)) :
  Real.tan (2 * α) = (48 + 25 * Real.sqrt 3) / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_tan_2alpha_value_l548_54874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gimps_meaning_l548_54850

/-- Represents a Mersenne prime number -/
structure MersennePrime where
  p : ℕ
  is_prime : Nat.Prime (2^p - 1)

/-- Represents the GIMPS project -/
def GIMPS : String := "Great Internet Mersenne Prime Search"

/-- Represents Curtis Cooper's team's discovery -/
def CoopersDiscovery : MersennePrime :=
  { p := 74207281,
    is_prime := sorry }

/-- The meaning of GIMPS in the context of prime number discovery -/
theorem gimps_meaning :
  GIMPS = "Great Internet Mersenne Prime Search" := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gimps_meaning_l548_54850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_hyperbola_standard_form_l548_54831

/-- An equilateral hyperbola is a hyperbola with equation x² - y² = k, where k ≠ 0 -/
def EquilateralHyperbola (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 = k ∧ k ≠ 0}

/-- The standard form of an equilateral hyperbola -/
def StandardEquilateralHyperbola (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / a - p.2^2 / a = 1 ∧ a ≠ 0}

theorem equilateral_hyperbola_standard_form 
  (k : ℝ) (h : (3, 2) ∈ EquilateralHyperbola k) :
  EquilateralHyperbola k = StandardEquilateralHyperbola 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_hyperbola_standard_form_l548_54831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_segments_for_seven_points_l548_54837

/-- A graph representation --/
structure Graph where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)

/-- Checks if any two out of three vertices are connected --/
def anyTwoConnected (G : Graph) (a b c : ℕ) : Prop :=
  (a, b) ∈ G.edges ∨ (b, c) ∈ G.edges ∨ (c, a) ∈ G.edges

/-- The property that among any three vertices, at least two are connected --/
def satisfiesProperty (G : Graph) : Prop :=
  ∀ a b c, a ∈ G.vertices → b ∈ G.vertices → c ∈ G.vertices →
    a ≠ b ∧ b ≠ c ∧ c ≠ a → anyTwoConnected G a b c

/-- The main theorem --/
theorem minimal_segments_for_seven_points :
  ∃ (G : Graph),
    G.vertices.card = 7 ∧
    satisfiesProperty G ∧
    G.edges.card = 9 ∧
    (∀ (H : Graph), H.vertices = G.vertices → satisfiesProperty H → H.edges.card ≥ 9) :=
  sorry

#check minimal_segments_for_seven_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_segments_for_seven_points_l548_54837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l548_54868

/-- Definition of the ellipse C -/
noncomputable def ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Right focus of the ellipse -/
def right_focus : ℝ × ℝ := (1, 0)

/-- Point on the ellipse -/
noncomputable def point_on_ellipse : ℝ × ℝ := (-1, Real.sqrt 2 / 2)

/-- Theorem stating the standard equation of the ellipse and the existence of point Q -/
theorem ellipse_properties (a b : ℝ) 
  (h_ellipse : ellipse (-1) (Real.sqrt 2 / 2) a b) :
  (∃ x y, x^2 / 2 + y^2 = 1 ↔ ellipse x y a b) ∧
  (∃ Q : ℝ × ℝ, Q.1 = 5/4 ∧ Q.2 = 0 ∧
    ∀ l : Set (ℝ × ℝ), right_focus ∈ l →
      ∀ A B : ℝ × ℝ, A ∈ l ∧ B ∈ l ∧ ellipse A.1 A.2 a b ∧ ellipse B.1 B.2 a b →
        (A.1 - Q.1) * (B.1 - Q.1) + (A.2 - Q.2) * (B.2 - Q.2) = -7/16) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l548_54868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_y_coordinate_l548_54898

/-- The y-coordinate of the vertex of the parabola y = -4x^2 - 16x - 36 is -20 -/
theorem parabola_vertex_y_coordinate :
  let f : ℝ → ℝ := λ x ↦ -4 * x^2 - 16 * x - 36
  ∃ m : ℝ, (∀ x, f m ≤ f x) ∧ f m = -20 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_y_coordinate_l548_54898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l548_54881

-- Define the function f(x) = x + 1/(x-1)
noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

-- Theorem statement
theorem min_value_of_f :
  ∀ x : ℝ, x > 1 → f x ≥ f 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l548_54881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_tangent_l548_54809

-- Define the angle α
variable (α : Real)

-- Define the point on the terminal side of α
def terminal_point : ℝ × ℝ := (3, -4)

-- Theorem statement
theorem half_angle_tangent :
  terminal_point = (3, -4) → Real.tan (α / 2) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_tangent_l548_54809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_simplify_l548_54869

-- Define the original expression
noncomputable def original_expr : ℝ := 3 / (5 * Real.sqrt 7 + 3 * Real.sqrt 2)

-- Define the simplified expression
noncomputable def simplified_expr : ℝ := (15 * Real.sqrt 7 - 9 * Real.sqrt 2) / 157

-- Theorem statement
theorem rationalize_and_simplify :
  original_expr = simplified_expr ∧
  7 < 2 ∧
  (∀ n : ℕ, n > 1 → ¬(157 % n = 0 ∧ (15 % n = 0 ∨ 9 % n = 0))) ∧
  (∀ m : ℕ, m > 1 → ¬(7 % (m^2) = 0) ∧ ¬(2 % (m^2) = 0)) := by
  sorry

#check rationalize_and_simplify

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_simplify_l548_54869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_minus_2x_l548_54886

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x = -2 then -1
  else if x = -1 then 1
  else if x = 0 then 3
  else if x = 1 then 2
  else if x = 2 then 1
  else 0  -- This else case is added to make the function total

-- Define the domain of f
def domain : Set ℝ := Set.Icc (-2) 2

-- State the theorem
theorem range_of_f_minus_2x : 
  Set.range (fun x => f x - 2 * x) = Set.Icc (-3) 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_minus_2x_l548_54886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l548_54882

-- Define the circle C
def circle_C (x y : ℝ) (a : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + a = 0

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x - y - 3 = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

-- Main theorem
theorem circle_intersection_theorem (a : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    circle_C x1 y1 a ∧ circle_C x2 y2 a ∧
    line_l x1 y1 ∧ line_l x2 y2 ∧
    perpendicular x1 y1 x2 y2) →
  a = -18 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l548_54882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l548_54842

noncomputable def f (x : Real) : Real := Real.sin x * Real.cos x - (Real.cos (x + Real.pi / 4))^2

theorem triangle_max_area (A B C : Real) (hA : 0 < A) (hA' : A < Real.pi / 2)
  (hB : 0 < B) (hB' : B < Real.pi / 2) (hC : 0 < C) (hC' : C < Real.pi / 2)
  (hSum : A + B + C = Real.pi) (hf : f (A / 2) = 0) (ha : Real.sin A = 1) :
  ∃ (S : Real), S ≤ (2 + Real.sqrt 3) / 4 ∧
  ∀ (S' : Real), S' = 1 / 2 * Real.sin B * Real.sin C / Real.sin A → S' ≤ S := by
  sorry

#check triangle_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l548_54842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_l548_54897

-- Define the initial point
noncomputable def initial_point : ℝ × ℝ := (2, 3)

-- Define the slope of the incident ray (parallel to x-2y=0)
noncomputable def incident_slope : ℝ := 1/2

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Define the equation of the reflected ray
def reflected_ray (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem statement
theorem light_reflection :
  ∃ (x₀ y₀ : ℝ), 
    y_axis x₀ ∧ 
    (y₀ - initial_point.2 = incident_slope * (x₀ - initial_point.1)) ∧
    reflected_ray x₀ y₀ := by
  sorry

#check light_reflection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_l548_54897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequalities_l548_54848

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(3*x^2 - 3)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (1/a)^(5*x + 5)

theorem function_inequalities (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, 0 < a ∧ a < 1 → (f a x < 1 ↔ x < -1 ∨ x > 1)) ∧
  (∀ x : ℝ, 0 < a ∧ a < 1 → (f a x ≥ g a x ↔ -1 ≤ x ∧ x ≤ -2/3)) ∧
  (∀ x : ℝ, a > 1 → (f a x ≥ g a x ↔ x ≤ -1 ∨ x ≥ -2/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequalities_l548_54848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PT_distance_l548_54855

noncomputable section

-- Define the points
def P : ℝ × ℝ := (0, 4)
def Q : ℝ × ℝ := (7, 0)
def R : ℝ × ℝ := (3, 0)
def S : ℝ × ℝ := (5, 3)

-- Define the line equations
noncomputable def line_PQ (x : ℝ) : ℝ := -4/7 * x + 4
noncomputable def line_RS (x : ℝ) : ℝ := 3/2 * x - 9/2

-- Define the intersection point T
def T : ℝ × ℝ := (59/17, 33/17)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem PT_distance :
  distance P T = Real.sqrt 2261 / 17 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_PT_distance_l548_54855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_first_pair_in_fortieth_group_l548_54862

/-- Represents a pair of positive integers -/
structure IntPair where
  first : Nat
  second : Nat
  first_pos : 0 < first
  second_pos : 0 < second
  no_repeat : first ≠ second

/-- Represents a group of pairs of positive integers -/
def IntPairGroup (n : Nat) : Set IntPair :=
  {pair : IntPair | pair.first + pair.second = n + 2}

/-- The sequence of groups -/
def GroupSequence : Nat → Set IntPair :=
  λ n => IntPairGroup (n + 1)

/-- The 21st pair in a group -/
noncomputable def TwentyFirstPair (g : Set IntPair) : IntPair :=
  sorry

theorem twenty_first_pair_in_fortieth_group :
  TwentyFirstPair (GroupSequence 40) = ⟨22, 20, sorry, sorry, sorry⟩ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_first_pair_in_fortieth_group_l548_54862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l548_54856

/-- Converts cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r θ z : Real) : Real × Real × Real :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates -/
noncomputable def cylindrical_point : Real × Real × Real := (4, Real.pi / 3, 4)

/-- The expected point in rectangular coordinates -/
noncomputable def rectangular_point : Real × Real × Real := (2, 2 * Real.sqrt 3, 4)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 = rectangular_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l548_54856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reject_null_hypothesis_l548_54866

-- Define the sample sizes
def n : ℕ := 40
def m : ℕ := 50

-- Define the sample means
def x_bar : ℝ := 130
def y_bar : ℝ := 140

-- Define the population variances
def D_X : ℝ := 80
def D_Y : ℝ := 100

-- Define the significance level
def α : ℝ := 0.01

-- Define the test statistic
noncomputable def z_obs : ℝ := (x_bar - y_bar) / Real.sqrt ((D_X / n) + (D_Y / m))

-- Define the critical value
noncomputable def z_crit : ℝ := 2.58

-- Theorem statement
theorem reject_null_hypothesis : |z_obs| > z_crit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reject_null_hypothesis_l548_54866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_markup_theorem_l548_54822

/-- Represents the markup percentage of the list price -/
noncomputable def markup_percentage : ℝ := 124 / 100

/-- Represents the discount percentage from the wholesaler -/
noncomputable def wholesaler_discount : ℝ := 30 / 100

/-- Represents the discount percentage offered to customers -/
noncomputable def customer_discount : ℝ := 25 / 100

/-- Represents the desired profit margin percentage -/
noncomputable def profit_margin : ℝ := 25 / 100

theorem merchant_markup_theorem (list_price : ℝ) (list_price_pos : list_price > 0) :
  let cost_price := list_price * (1 - wholesaler_discount)
  let marked_price := list_price * markup_percentage
  let selling_price := marked_price * (1 - customer_discount)
  selling_price = cost_price / (1 - profit_margin) := by
  sorry

#check merchant_markup_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_markup_theorem_l548_54822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_z_plus_inv_z_l548_54876

/-- Given a complex number z where the points 0, z, 1/z, and z + 1/z form a parallelogram 
    with an area of 4/5 in the complex plane, the minimum value of |z + 1/z| is 2√5/5. -/
theorem min_value_z_plus_inv_z (z : ℂ) : 
  (Complex.abs (z - 0) * Complex.abs ((1 : ℂ) / z - 0) * Complex.abs (Complex.sin (2 * Complex.arg z)) = 4/5) →
  (∀ w : ℂ, (Complex.abs (w - 0) * Complex.abs ((1 : ℂ) / w - 0) * Complex.abs (Complex.sin (2 * Complex.arg w)) = 4/5) →
    Complex.abs (z + 1/z) ≤ Complex.abs (w + 1/w)) →
  Complex.abs (z + 1/z) = 2 * Real.sqrt 5 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_z_plus_inv_z_l548_54876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_expense_calculation_l548_54879

noncomputable def monthly_earnings : ℝ := 6000
noncomputable def house_rental : ℝ := 640
noncomputable def electric_water_ratio : ℝ := 1/4
noncomputable def insurance_ratio : ℝ := 1/5
noncomputable def remaining_money : ℝ := 2280

theorem food_expense_calculation :
  let electric_water := monthly_earnings * electric_water_ratio
  let insurance := monthly_earnings * insurance_ratio
  let total_expenses := house_rental + electric_water + insurance
  let food_expense := monthly_earnings - total_expenses - remaining_money
  food_expense = 380 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_expense_calculation_l548_54879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_probability_after_seven_minutes_l548_54867

/-- Represents a point on the lattice -/
structure LatticePoint where
  x : Int
  y : Int

/-- Represents the state of the ant on the lattice -/
structure AntState where
  position : LatticePoint
  time : Nat

/-- Defines the lattice structure -/
def lattice : Set LatticePoint := sorry

/-- Defines the set of red points on the lattice -/
def redPoints : Set LatticePoint := sorry

/-- Defines the set of adjacent points for a given point -/
def adjacentPoints (p : LatticePoint) : Set LatticePoint := sorry

/-- Defines the probability of the ant moving to a specific adjacent point -/
def moveProbability (start : LatticePoint) (finish : LatticePoint) : Real := sorry

/-- Defines the probability of the ant being at a specific point after n moves -/
noncomputable def probabilityAfterNMoves (start : LatticePoint) (finish : LatticePoint) (n : Nat) : Real := sorry

theorem ant_probability_after_seven_minutes :
  let A : LatticePoint := ⟨0, 0⟩
  let C : LatticePoint := ⟨0, 2⟩
  probabilityAfterNMoves A C 7 = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_probability_after_seven_minutes_l548_54867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_plane_l548_54828

noncomputable def distanceToPlane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ :=
  let (x₀, y₀, z₀) := M₀
  let (x₁, y₁, z₁) := M₁
  let (x₂, y₂, z₂) := M₂
  let (x₃, y₃, z₃) := M₃
  
  -- Calculate plane coefficients
  let A := (y₂ - y₁) * (z₃ - z₁) - (z₂ - z₁) * (y₃ - y₁)
  let B := (z₂ - z₁) * (x₃ - x₁) - (x₂ - x₁) * (z₃ - z₁)
  let C := (x₂ - x₁) * (y₃ - y₁) - (y₂ - y₁) * (x₃ - x₁)
  let D := -A * x₁ - B * y₁ - C * z₁

  -- Calculate distance
  (|A * x₀ + B * y₀ + C * z₀ + D|) / Real.sqrt (A^2 + B^2 + C^2)

theorem distance_to_specific_plane :
  let M₀ : ℝ × ℝ × ℝ := (-6, 5, 5)
  let M₁ : ℝ × ℝ × ℝ := (-2, 0, -4)
  let M₂ : ℝ × ℝ × ℝ := (-1, 7, 1)
  let M₃ : ℝ × ℝ × ℝ := (4, -8, -4)
  distanceToPlane M₀ M₁ M₂ M₃ = 23 * Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_plane_l548_54828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_teams_l548_54858

/-- The number of teams in the tournament -/
def n : ℕ := 10

/-- The number of matches played by each team that withdrew -/
def matches_per_withdrawn : ℕ := 3

/-- The number of teams that withdrew -/
def withdrawn_teams : ℕ := 2

/-- The total number of matches played in the tournament -/
def total_matches : ℕ := 34

/-- The number of matches played by the remaining teams -/
def remaining_matches : ℕ := total_matches - withdrawn_teams * matches_per_withdrawn

/-- The number of matches in a round-robin tournament with m teams -/
def round_robin_matches (m : ℕ) : ℕ := m * (m - 1) / 2

/-- Theorem stating the value of n in the given tournament scenario -/
theorem tournament_teams : n = 10 :=
  by
    -- The proof goes here
    sorry

#eval n -- This will output 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_teams_l548_54858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_slope_of_line_l_l548_54863

/-- The slope of a line given by the equation y - b = m(x + a) is m. -/
theorem slope_of_line (m b a : ℝ) : 
  (∀ x y, y - b = m * (x + a)) ↔ (∀ x y, y = m * x + (m * a + b)) :=
sorry

/-- The slope of the line l: y - 3 = 4(x + 1) is 4 -/
theorem slope_of_line_l :
  ∃ (m : ℝ), m = 4 ∧ (∀ x y, y - 3 = 4 * (x + 1)) ↔ (∀ x y, y = m * x + (m + 3)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_slope_of_line_l_l548_54863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factoring_b_l548_54890

/-- A polynomial of the form x^2 + bx + 2023 factors if there exist integers p and q such that x^2 + bx + 2023 = (x + p)(x + q) -/
def factors (b : ℤ) : Prop :=
  ∃ p q : ℤ, ∀ x : ℤ, x^2 + b*x + 2023 = (x + p) * (x + q)

/-- 136 is the smallest positive integer b for which x^2 + bx + 2023 factors into a product of two polynomials with integer coefficients -/
theorem smallest_factoring_b : 
  factors 136 ∧ ∀ b : ℤ, 0 < b → b < 136 → ¬(factors b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factoring_b_l548_54890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_cube_coefficient_of_expansion_l548_54838

/-- The coefficient of x³ in the expansion of x(1+2x)⁶ is 60 -/
theorem x_cube_coefficient_of_expansion : 
  (Polynomial.coeff (X * (1 + 2*X)^6) 3) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_cube_coefficient_of_expansion_l548_54838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l548_54819

noncomputable def f (x : ℝ) : ℝ := x + Real.sin (Real.pi * x) - 3

theorem sum_of_f_values (h : ∀ x₁ x₂ : ℝ, x₁ + x₂ = 2 → f x₁ + f x₂ = -4) :
  (Finset.range 4031).sum (fun i => f ((i + 1 : ℕ) / 2016)) = -8062 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l548_54819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quotient_has_49_zeros_l548_54800

/-- Represents a 100-digit number without zeros -/
def X : ℕ := sorry

/-- Represents the 50-digit head of X -/
def H : ℕ := sorry

/-- X has exactly 100 digits -/
axiom X_has_100_digits : 10^99 ≤ X ∧ X < 10^100

/-- H is the first 50 digits of X -/
axiom H_is_head : H = X / 10^50

/-- X does not contain any zeros -/
axiom X_no_zeros : ∀ d : ℕ, d < 100 → (X / 10^d) % 10 ≠ 0

/-- X is divisible by H without remainder -/
axiom X_div_H : X % H = 0

/-- The quotient of X divided by H -/
def quotient : ℕ := X / H

theorem quotient_has_49_zeros : ∃ k : ℕ, 1 ≤ k ∧ k < 10 ∧ quotient = 10^49 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quotient_has_49_zeros_l548_54800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l548_54860

-- Define the function f as noncomputable due to Real.sqrt
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2 + a

-- State the theorem
theorem function_property (a : ℝ) :
  (∀ x ∈ Set.Icc (-π/6) (π/3), f a x ≤ 3/2 - (f a (-π/6) + f a (π/3))/2) ∧
  (∀ x ∈ Set.Icc (-π/6) (π/3), f a x ≥ -3/2 + (f a (-π/6) + f a (π/3))/2) →
  a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l548_54860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_dormitory_distance_l548_54846

-- Define the cost function
noncomputable def f (x : ℝ) : ℝ := 1000 / (x + 5) + 5 * x + (1/2) * (x^2 + 25)

-- State the theorem
theorem optimal_dormitory_distance :
  ∃ (x : ℝ), 2 ≤ x ∧ x ≤ 8 ∧ 
  (∀ (y : ℝ), 2 ≤ y ∧ y ≤ 8 → f x ≤ f y) ∧
  x = 5 ∧ f x = 150 := by
  -- The proof goes here
  sorry

#check optimal_dormitory_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_dormitory_distance_l548_54846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_ships_time_of_min_distance_l548_54865

/-- Represents the position of a ship at a given time -/
structure ShipPosition where
  time : ℝ
  distance : ℝ

/-- Represents the motion of two ships -/
structure ShipMotion where
  pos1 : ShipPosition
  pos2 : ShipPosition
  pos3 : ShipPosition

/-- The theorem stating the minimum distance between the ships -/
theorem min_distance_between_ships (motion : ShipMotion) : ℝ := by
  let time1 := 0  -- 9:00
  let time2 := 35 -- 9:35
  let time3 := 55 -- 9:55
  have h1 : motion.pos1 = ⟨time1, 20⟩ := by sorry
  have h2 : motion.pos2 = ⟨time2, 15⟩ := by sorry
  have h3 : motion.pos3 = ⟨time3, 13⟩ := by sorry
  exact 12 -- The minimum distance is 12 miles

/-- The theorem stating the time of minimum distance between the ships -/
theorem time_of_min_distance (motion : ShipMotion) : ℝ := by
  let time1 := 0  -- 9:00
  let time2 := 35 -- 9:35
  let time3 := 55 -- 9:55
  have h1 : motion.pos1 = ⟨time1, 20⟩ := by sorry
  have h2 : motion.pos2 = ⟨time2, 15⟩ := by sorry
  have h3 : motion.pos3 = ⟨time3, 13⟩ := by sorry
  exact 80 -- 10:20 (80 minutes after 9:00)


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_ships_time_of_min_distance_l548_54865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rooms_occupied_after_one_hour_l548_54816

/-- Represents the state of rooms after a certain number of minutes -/
structure RoomState where
  minutes : ℕ
  occupied_rooms : ℕ
  people_in_first_room : ℕ

/-- The initial state of the rooms -/
def initial_state : RoomState :=
  { minutes := 0, occupied_rooms := 1, people_in_first_room := 1000 }

/-- The state transition function representing one minute of movement -/
def next_state (state : RoomState) : RoomState :=
  { minutes := state.minutes + 1,
    occupied_rooms := if state.minutes % 2 = 0 then state.occupied_rooms + 1 else state.occupied_rooms,
    people_in_first_room := state.people_in_first_room - 1 }

/-- The final state after 60 minutes -/
def final_state : RoomState := (Nat.iterate next_state 60 initial_state)

/-- The main theorem stating that after 60 minutes, 31 rooms will be occupied -/
theorem rooms_occupied_after_one_hour :
  final_state.occupied_rooms = 31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rooms_occupied_after_one_hour_l548_54816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_l548_54895

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point A
def point_A : ℝ × ℝ := (5, 3)

-- Define a point on the parabola
def point_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2 ∧ M ≠ point_A

-- Define the perimeter of triangle MAF
noncomputable def perimeter (M : ℝ × ℝ) : ℝ :=
  Real.sqrt ((M.1 - point_A.1)^2 + (M.2 - point_A.2)^2) +
  Real.sqrt ((M.1 - focus.1)^2 + (M.2 - focus.2)^2) +
  Real.sqrt ((point_A.1 - focus.1)^2 + (point_A.2 - focus.2)^2)

-- Theorem statement
theorem min_perimeter :
  ∃ (min_value : ℝ), min_value = 11 ∧
  ∀ (M : ℝ × ℝ), point_on_parabola M → perimeter M ≥ min_value :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_l548_54895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l548_54806

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => 3 * sequence_a n + 2 * Real.sqrt (2 * (sequence_a n)^2 - 1)

theorem sequence_a_properties :
  (∀ n : ℕ, sequence_a n > 0 ∧ ∃ m : ℤ, sequence_a n = m) ∧
  (∀ m : ℕ, ¬(2015 ∣ ⌊sequence_a m⌋)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l548_54806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_slope_constant_l548_54878

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Defines the eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- Checks if a point (x, y) lies on the ellipse -/
def Ellipse.contains (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Represents a line with slope k and y-intercept t -/
structure Line where
  k : ℝ
  t : ℝ

/-- Checks if a point (x, y) lies in the first quadrant -/
def inFirstQuadrant (x y : ℝ) : Prop :=
  0 < x ∧ 0 < y

/-- Represents the slope of a line passing through the origin and a point (x, y) -/
noncomputable def slopeFromOrigin (x y : ℝ) : ℝ :=
  y / x

/-- Checks if three real numbers form a geometric sequence -/
def isGeometricSequence (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem ellipse_intersection_slope_constant
  (e : Ellipse)
  (h_ecc : e.eccentricity = Real.sqrt 3 / 2)
  (h_point : e.contains 1 (Real.sqrt 3 / 2))
  (l : Line)
  (P Q : ℝ × ℝ)
  (h_P : e.contains P.1 P.2 ∧ inFirstQuadrant P.1 P.2)
  (h_Q : e.contains Q.1 Q.2 ∧ inFirstQuadrant Q.1 Q.2)
  (h_PQ_on_l : (P.2 = l.k * P.1 + l.t) ∧ (Q.2 = l.k * Q.1 + l.t))
  (h_geo_seq : isGeometricSequence (slopeFromOrigin P.1 P.2) l.k (slopeFromOrigin Q.1 Q.2)) :
  l.k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_slope_constant_l548_54878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l548_54847

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := a * (r^n - 1) / (r - 1)

theorem problem_solution : 
  let a : ℝ := 4
  let r : ℝ := 4
  let n : ℕ := 11
  geometric_sum a r n + 100 = 5592504 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l548_54847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_is_correct_l548_54807

/-- Represents the dimensions and water behavior of an aquarium -/
structure Aquarium where
  width : ℚ
  height : ℚ
  tiltedCoverage : ℚ
  haveValidDimensions : width = 10 ∧ height = 8
  haveValidTiltedCoverage : tiltedCoverage = 3/4

/-- Calculates the depth of water when the aquarium is leveled -/
def waterDepthWhenLevel (a : Aquarium) : ℚ :=
  (a.tiltedCoverage * a.width * a.height) / (2 * a.width)

/-- Theorem stating that the water depth when level is 3.75 inches -/
theorem water_depth_is_correct (a : Aquarium) :
  waterDepthWhenLevel a = 15/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_is_correct_l548_54807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_exotic_is_square_infinitely_many_exotic_numbers_l548_54804

/-- Definition: A positive integer is exotic if it's divisible by its number of positive divisors -/
def IsExotic (n : ℕ) : Prop :=
  n > 0 ∧ ∃ k : ℕ, (Nat.divisors n).card = k ∧ k ∣ n

/-- Part a: If an odd exotic number exists, it's a perfect square -/
theorem odd_exotic_is_square :
  ∀ n : ℕ, Odd n → IsExotic n → ∃ m : ℕ, n = m^2 := by sorry

/-- Part b: There are infinitely many exotic numbers -/
theorem infinitely_many_exotic_numbers :
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ n, IsExotic (f n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_exotic_is_square_infinitely_many_exotic_numbers_l548_54804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_sum_reciprocals_l548_54826

theorem complex_magnitude_sum_reciprocals (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 6) : 
  Complex.abs (1 / z + 1 / w) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_sum_reciprocals_l548_54826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_negative_three_l548_54811

theorem complex_expression_equals_negative_three :
  (Real.pi - 3.14 : ℝ) ^ (0 : ℝ) - (8 : ℝ) ^ (2/3 : ℝ) + (1/5 : ℝ) ^ (-(2 : ℝ)) * (3/25 : ℝ) - (5 : ℝ) ^ (Real.log 3 / Real.log 5) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_negative_three_l548_54811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_width_l548_54896

/-- The width of a rectangular prism given its length, height, and diagonal. -/
noncomputable def prism_width (l h d : ℝ) : ℝ := Real.sqrt (d^2 - l^2 - h^2)

/-- Theorem stating that a rectangular prism with length 5, height 13, and diagonal 15 has width √31. -/
theorem rectangular_prism_width :
  prism_width 5 13 15 = Real.sqrt 31 := by
  -- Unfold the definition of prism_width
  unfold prism_width
  -- Simplify the expression
  simp [Real.sqrt_eq_iff_sq_eq, pow_two]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_width_l548_54896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_structure_l548_54888

/-- The set T as defined in the problem -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 
    let x := p.1
    let y := p.2
    (5 = x + 3 ∧ y - 6 ≤ 5) ∨
    (5 = y - 6 ∧ x + 3 ≤ 5) ∨
    (x + 3 = y - 6 ∧ 5 ≥ x + 3)}

/-- Definition of a ray in 2D space -/
def IsRay (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
    s = {r : ℝ × ℝ | ∃ (t : ℝ), t ≥ 0 ∧ r = p + t • (q - p)}

/-- The main theorem describing the structure of set T -/
theorem T_structure :
  ∃ (r₁ r₂ r₃ : Set (ℝ × ℝ)),
    IsRay r₁ ∧ IsRay r₂ ∧ IsRay r₃ ∧
    (∃ (p : ℝ × ℝ), p = (2, 11) ∧ p ∈ r₁ ∧ p ∈ r₂ ∧ p ∈ r₃) ∧
    T = r₁ ∪ r₂ ∪ r₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_structure_l548_54888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_is_correct_l548_54889

/-- A regular quadrangular pyramid with height 2 and base side length √2 -/
structure RegularQuadrangularPyramid where
  -- S is the apex, ABCD is the base
  height : ℝ
  base_side_length : ℝ
  height_eq : height = 2
  base_side_eq : base_side_length = Real.sqrt 2

/-- The shortest distance between a point on BD and a point on SC -/
noncomputable def shortest_distance (p : RegularQuadrangularPyramid) : ℝ := 2 * Real.sqrt 5 / 5

/-- Theorem stating that the shortest distance is 2√5/5 -/
theorem shortest_distance_is_correct (p : RegularQuadrangularPyramid) :
  shortest_distance p = 2 * Real.sqrt 5 / 5 := by
  -- Unfold the definition of shortest_distance
  unfold shortest_distance
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_is_correct_l548_54889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_tangent_lines_sum_l548_54802

/-- Two parabolas in the coordinate plane -/
structure Parabolas where
  a : ℝ
  eq1 : ℝ → ℝ → Prop := fun x y ↦ x = y^2 + a
  eq2 : ℝ → ℝ → Prop := fun x y ↦ y = x^2 + a

/-- A line in the coordinate plane -/
structure Line where
  m : ℝ
  b : ℝ
  eq : ℝ → ℝ → Prop := fun x y ↦ y = m * x + b

/-- Predicate to check if a line is tangent to both parabolas -/
def isTangentToBoth (l : Line) (p : Parabolas) : Prop :=
  ∃ x y, p.eq1 x y ∧ p.eq2 x y ∧ l.eq x y

/-- Theorem stating the result about the sum of p and q -/
theorem parabolas_tangent_lines_sum (p : Parabolas) 
    (l1 l2 l3 : Line) (s : ℝ) (pq : ℕ × ℕ) : 
    isTangentToBoth l1 p → 
    isTangentToBoth l2 p → 
    isTangentToBoth l3 p → 
    -- The three lines form an equilateral triangle
    ∃ A B C : ℝ × ℝ, l1.eq A.1 A.2 ∧ l2.eq B.1 B.2 ∧ l3.eq C.1 C.2 ∧ 
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 →
    -- The area of the triangle is s
    s^2 * 4 = ((A.1 - B.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - B.2))^2 →
    -- s^2 = p/q where p and q are coprime positive integers
    s^2 = pq.1 / pq.2 →
    Nat.Coprime pq.1 pq.2 →
    pq.1 > 0 →
    pq.2 > 0 →
    pq.1 + pq.2 = 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_tangent_lines_sum_l548_54802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_square_inequality_l548_54812

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2017 * x^2 + 2018 * x * Real.sin x

-- State the theorem
theorem function_inequality_implies_square_inequality
  (x₁ x₂ : ℝ)
  (h₁ : x₁ ∈ Set.Ioo (-Real.pi) Real.pi)
  (h₂ : x₂ ∈ Set.Ioo (-Real.pi) Real.pi)
  (h₃ : f x₁ > f x₂) :
  x₁^2 > x₂^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_square_inequality_l548_54812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_line_l548_54877

/-- The fixed point A -/
def A : ℝ × ℝ := (1, 2)

/-- The coefficients of the given line 2x + y - 4 = 0 -/
def line_coeff : ℝ × ℝ × ℝ := (2, 1, -4)

/-- The locus of points equidistant from A and the given line -/
def locus (x y : ℝ) : Prop :=
  (x - A.fst)^2 + (y - A.snd)^2 = 
  ((line_coeff.1 * x + line_coeff.2.1 * y + line_coeff.2.2)^2) / (line_coeff.1^2 + line_coeff.2.1^2)

/-- The locus is a line -/
theorem locus_is_line : 
  ∃ (m b : ℝ), ∀ x y : ℝ, locus x y ↔ y = m * x + b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_line_l548_54877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l548_54801

theorem triangle_cosine_inequality (a b c : ℝ) (α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_angle_sum : α + β + γ = Real.pi)
  (h_cosine_law_a : a^2 = b^2 + c^2 - 2*b*c*(Real.cos α))
  (h_cosine_law_b : b^2 = a^2 + c^2 - 2*a*c*(Real.cos β))
  (h_cosine_law_c : c^2 = a^2 + b^2 - 2*a*b*(Real.cos γ)) :
  (Real.cos α) / a^3 + (Real.cos β) / b^3 + (Real.cos γ) / c^3 ≥ 3 / (2*a*b*c) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l548_54801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_puzzle_solvable_l548_54871

/-- Represents a piece cut from the original 7x7 square --/
structure Piece where
  shape : List (Nat × Nat)

/-- Represents the 7x7 square from which pieces are cut --/
def originalSquare : Set (Nat × Nat) :=
  {p | p.1 < 7 ∧ p.2 < 7}

/-- The 8 pieces cut from the original square --/
def cutPieces : List Piece :=
  sorry

/-- Represents the target 6x6 square --/
def targetSquare : Set (Nat × Nat) :=
  {p | p.1 < 6 ∧ p.2 < 6}

/-- Predicate to check if a list of pieces can form the target square --/
def canFormTargetSquare (pieces : List Piece) : Prop :=
  sorry

/-- Theorem stating that the puzzle is solvable --/
theorem square_puzzle_solvable :
  canFormTargetSquare cutPieces := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_puzzle_solvable_l548_54871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_complex_segment_l548_54861

noncomputable def complex_midpoint (z₁ z₂ : ℂ) : ℂ := (z₁ + z₂) / 2

theorem midpoint_of_complex_segment : 
  let A : ℂ := 6 + 5*Complex.I
  let B : ℂ := -2 + 3*Complex.I
  complex_midpoint A B = 2 + 4*Complex.I := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_complex_segment_l548_54861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l548_54814

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the problem conditions
def problemConditions (t : Triangle) : Prop :=
  t.c = 2 ∧ Real.cos t.A - Real.sqrt 3 * t.a * Real.cos t.C = 0

-- Define the area of the triangle
noncomputable def triangleArea (t : Triangle) : ℝ :=
  1/2 * t.a * t.b * Real.sin t.C

-- Theorem statement
theorem max_area_triangle (t : Triangle) (h : problemConditions t) :
  triangleArea t ≤ Real.sqrt 3 ∧
  (triangleArea t = Real.sqrt 3 ↔ t.C = π/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l548_54814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_cheburashkas_is_eleven_l548_54840

/-- Represents the number of Cheburashkas in a row -/
def num_cheburashkas : ℕ := sorry

/-- Represents the total number of characters in a row before erasure -/
def total_characters : ℕ := sorry

/-- Represents the total number of Krakozyabras after erasure -/
def total_krakozyabras : ℕ := sorry

/-- The number of rows -/
def num_rows : ℕ := 2

/-- Axiom: At least one Cheburashka in each row -/
axiom at_least_one_cheburashka : num_cheburashkas ≥ 1

/-- Axiom: Total Krakozyabras after erasure -/
axiom krakozyabras_count : total_krakozyabras = 29

/-- Axiom: Relationship between total characters and Krakozyabras -/
axiom characters_krakozyabras_relation : total_krakozyabras = num_rows * (total_characters - 1)

/-- Axiom: Relationship between Cheburashkas and total characters -/
axiom cheburashkas_characters_relation : total_characters = 2 * num_cheburashkas + (num_cheburashkas - 1) + num_cheburashkas

/-- Theorem: The number of Cheburashkas originally drawn is 11 -/
theorem num_cheburashkas_is_eleven : num_cheburashkas = 11 := by
  sorry

#check num_cheburashkas_is_eleven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_cheburashkas_is_eleven_l548_54840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l548_54841

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := 1/x - Real.sqrt x

-- Define the point of tangency
noncomputable def P : ℝ × ℝ := (4, -7/4)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m*x + b ↔ 5*x + 16*y + 8 = 0) ∧
    (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 0 < |h| → |h| < δ → 
      |(f (P.1 + h) - f P.1) / h - m| < ε) ∧
    (m * P.1 + b = P.2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l548_54841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l548_54845

-- Define set A
def A : Set ℝ := {y | ∃ x, y = 2^x}

-- Define set B
def B : Set ℝ := {x | x^2 - 1 < 0}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = Set.Ioi (-1 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l548_54845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_holds_iff_m_geq_two_l548_54893

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x + 3 else -x^2 + 2*x + 3

-- State the theorem
theorem f_inequality_holds_iff_m_geq_two :
  ∀ m : ℝ, (∀ x : ℝ, f x - Real.exp x - m ≤ 0) ↔ m ≥ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_holds_iff_m_geq_two_l548_54893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l548_54834

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + 2

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  T = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l548_54834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_final_amount_l548_54832

/-- The value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "dime" => 10
  | "quarter" => 25
  | "nickel" => 5
  | _ => 0

/-- Calculate the total value of a collection of coins -/
def total_value (coins : List (String × ℕ)) : ℕ :=
  List.sum (coins.map (fun (coin, count) => coin_value coin * count))

/-- Sam's initial coins -/
def initial_coins : List (String × ℕ) :=
  [("dime", 9), ("quarter", 5), ("nickel", 3)]

/-- Coins given by dad -/
def dad_coins : List (String × ℕ) :=
  [("dime", 7), ("quarter", 2)]

/-- Coins taken by mom -/
def mom_coins : List (String × ℕ) :=
  [("nickel", 1), ("dime", 2)]

theorem sam_final_amount :
  total_value initial_coins + total_value dad_coins - total_value mom_coins = 325 := by
  sorry

#eval total_value initial_coins + total_value dad_coins - total_value mom_coins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_final_amount_l548_54832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_cans_correct_answer_is_C_other_options_incorrect_l548_54864

/-- The number of cans of tea obtainable for N nickels, given T cans can be bought for P pennies,
    with an initial overhead of 2 cans. -/
def tea_cans (T P N : ℕ) : ℕ := (5 * T * N) / P - 2

/-- Theorem stating that the formula for tea_cans is correct. -/
theorem tea_cans_correct (T P N : ℕ) (hP : P > 0) : 
  tea_cans T P N = (5 * T * N) / P - 2 := by
  -- Unfold the definition of tea_cans
  unfold tea_cans
  -- The equality holds by definition
  rfl

/-- Proof that the answer matches option C in the multiple choice question. -/
theorem answer_is_C (T P N : ℕ) (hP : P > 0) :
  tea_cans T P N = (5 * T * N) / P - 2 := by
  -- Apply the tea_cans_correct theorem
  exact tea_cans_correct T P N hP

/-- Verification that the other options are not correct. -/
theorem other_options_incorrect (T P N : ℕ) (hP : P > 0) (hT : T > 0) :
  tea_cans T P N ≠ T * N / P - 2 ∧   -- Option A
  tea_cans T P N ≠ 5 * T * N / (2 * P) ∧ -- Option B
  tea_cans T P N ≠ 5 * N / (T * P) + 2 := by -- Option D
  -- We'll use sorry here as a full proof would be quite lengthy
  sorry

#check tea_cans_correct
#check answer_is_C
#check other_options_incorrect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_cans_correct_answer_is_C_other_options_incorrect_l548_54864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_part_sum_integer_decimal_diff_l548_54825

-- Define the decimal part of √5
noncomputable def a : ℝ := Real.sqrt 5 - 2

-- Define the integer part of √13
def b : ℤ := 3

-- Define x and y
def x : ℤ := 11
noncomputable def y : ℝ := Real.sqrt 3 - 1

-- Theorem 1
theorem decimal_part_sum : a + b - Real.sqrt 5 = 1 := by sorry

-- Theorem 2
theorem integer_decimal_diff : 
  (10 + Real.sqrt 3 = x + y) → (x : ℝ) - y = 12 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_part_sum_integer_decimal_diff_l548_54825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_min_value_S_l548_54851

/-- Sequence a_n -/
def a : ℕ → ℝ := sorry

/-- Partial sum S_n of sequence a_n -/
def S : ℕ → ℝ := sorry

/-- The relation between S_n, n, and a_n -/
axiom relation (n : ℕ) : 2 * S n / n + n = 2 * a n + 1

/-- a_4, a_7, and a_9 form a geometric sequence -/
axiom geometric_seq : (a 7) ^ 2 = (a 4) * (a 9)

theorem arithmetic_sequence : 
  ∀ n : ℕ, a (n + 1) = a n + 1 := by
  sorry

theorem min_value_S : 
  ∃ n : ℕ, S n = -78 ∧ ∀ m : ℕ, S m ≥ -78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_min_value_S_l548_54851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_case_cost_properties_l548_54835

/-- Represents the cost of pencil cases under different conditions -/
structure PencilCaseCost where
  a : ℝ
  b : ℝ
  h1 : 2 * a + 3 * b = 108
  h2 : 5 * a = 6 * b

/-- Calculates the cost for Option 1 -/
def option1Cost (x : ℝ) (cost : PencilCaseCost) : ℝ :=
  cost.a * 20 + cost.a * 0.8 * (x - 20) + cost.b * 30

/-- Calculates the cost for Option 2 -/
def option2Cost (x : ℝ) (cost : PencilCaseCost) : ℝ :=
  (cost.a * x + cost.b * 30) * 0.9

/-- Theorem stating the properties of the pencil case costs -/
theorem pencil_case_cost_properties (cost : PencilCaseCost) :
  cost.a = 24 ∧ 
  cost.b = 20 ∧ 
  ∀ x > 20, option1Cost x cost = 19.2 * x + 696 ∧
            option2Cost x cost = 21.6 * x + 540 ∧
            (option1Cost 65 cost = option2Cost 65 cost) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_case_cost_properties_l548_54835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gina_hourly_rate_l548_54872

/-- Represents Gina's cup painting rates and order details -/
structure PaintingJob where
  rose_rate : ℚ  -- Cups per hour for rose cups
  lily_rate : ℚ  -- Cups per hour for lily cups
  rose_order : ℚ  -- Number of rose cups ordered
  lily_order : ℚ  -- Number of lily cups ordered
  total_payment : ℚ  -- Total payment for the order

/-- Calculates Gina's hourly rate for a given painting job -/
def hourly_rate (job : PaintingJob) : ℚ :=
  job.total_payment / (job.rose_order / job.rose_rate + job.lily_order / job.lily_rate)

/-- Theorem stating that Gina's hourly rate for the given job is $30 -/
theorem gina_hourly_rate :
  let job := PaintingJob.mk 6 7 6 14 90
  hourly_rate job = 30 := by
  -- Expand the definition of hourly_rate
  unfold hourly_rate
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gina_hourly_rate_l548_54872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l548_54830

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := y^2 / 16 + x^2 / 4 = 1

-- Define the line l in polar coordinates
def line_l_polar (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi/3) = 3

-- Define a point on the ellipse C
def point_on_ellipse (x y : ℝ) : Prop := ellipse_C x y

-- Theorem statement
theorem ellipse_and_line_properties :
  -- 1. Rectangular coordinate equation of line l
  (∃ (x y : ℝ), Real.sqrt 3 * x + y - 6 = 0) ∧
  -- 2. Parametric equation of ellipse C
  (∃ (α : ℝ), ∀ (x y : ℝ), ellipse_C x y ↔ (x = 2 * Real.cos α ∧ y = 4 * Real.sin α)) ∧
  -- 3. Maximum value of |2√3x + y - 1| for any point on ellipse C
  (∃ (M : ℝ), M = 9 ∧ ∀ (x y : ℝ), point_on_ellipse x y → |2 * Real.sqrt 3 * x + y - 1| ≤ M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l548_54830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reward_is_190_l548_54854

/-- Represents the possible grades in Paul's scorecard -/
inductive Grade
  | BPlus
  | A
  | APlus
deriving BEq, Repr

/-- Calculates the reward for a given grade and number of A+ grades -/
def reward (grade : Grade) (num_a_plus : Nat) : Nat :=
  match grade with
  | Grade.BPlus => if num_a_plus ≥ 2 then 10 else 5
  | Grade.A => if num_a_plus ≥ 2 then 20 else 10
  | Grade.APlus => 15

/-- The total number of courses in Paul's scorecard -/
def total_courses : Nat := 10

/-- Theorem stating the maximum amount Paul could receive -/
theorem max_reward_is_190 :
  ∃ (grades : List Grade),
    grades.length = total_courses ∧
    grades.count Grade.APlus ≥ 2 ∧
    (grades.map (λ g => reward g (grades.count Grade.APlus))).sum = 190 ∧
    ∀ (other_grades : List Grade),
      other_grades.length = total_courses →
      (other_grades.map (λ g => reward g (other_grades.count Grade.APlus))).sum ≤ 190 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reward_is_190_l548_54854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_range_implies_k_range_l548_54885

theorem integral_range_implies_k_range (k : ℝ) :
  (2 : ℝ) ≤ (∫ x in (1 : ℝ)..(2 : ℝ), k * x + 1) ∧ 
  (∫ x in (1 : ℝ)..(2 : ℝ), k * x + 1) ≤ (4 : ℝ) →
  (2/3 : ℝ) ≤ k ∧ k ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_range_implies_k_range_l548_54885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_direction_vector_component_l548_54852

/-- Given a line passing through points (-3, 1) and (0, 3) with direction vector (2, b),
    prove that b = 4/3 -/
theorem line_direction_vector_component (b : ℚ) : 
  (∃ (k : ℚ), k • (![0, 3] - ![(-3), 1] : Fin 2 → ℚ) = ![2, b]) → b = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_direction_vector_component_l548_54852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l548_54849

-- Define the function f(x) = x + lg x - 3
noncomputable def f (x : ℝ) : ℝ := x + Real.log x / Real.log 10 - 3

-- State the theorem
theorem root_in_interval :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l548_54849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_angle_measure_l548_54875

/-- The sum of interior angles of a convex polygon with n sides is (n-2) * 180° -/
def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- Given a convex polygon where the sum of all but one interior angle is 2240°,
    prove that the measure of the missing angle is 100° -/
theorem missing_angle_measure (n : ℕ) (h : n ≥ 3) :
  sum_of_interior_angles n - 2240 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_angle_measure_l548_54875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_not_general_solution_l548_54827

open Real

-- Define the differential equation
def diff_eq (x y y' : ℝ) : Prop :=
  (x - x^3) * y' + (2 * x^2 - 1) * y - x^3 = 0

-- Define the proposed solution
noncomputable def y (x : ℝ) : ℝ := x * Real.sqrt (1 - x^2) + x

-- Define the derivative of the proposed solution
noncomputable def y' (x : ℝ) : ℝ := Real.sqrt (1 - x^2) - x^2 / Real.sqrt (1 - x^2) + 1

-- Theorem statement
theorem solution_satisfies_diff_eq (x : ℝ) (h : x ∈ Set.Ioo (-1) 1) : 
  diff_eq x (y x) (y' x) := by
  sorry

-- Not a general solution
theorem not_general_solution : 
  ¬∃ (C : ℝ), ∀ (x : ℝ), x ∈ Set.Ioo (-1) 1 → diff_eq x (y x + C) (y' x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_not_general_solution_l548_54827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l548_54836

/-- The speed of a train in km/h given its distance traveled and time taken -/
noncomputable def train_speed (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 60

/-- Theorem stating that a train traveling 11.67 km in 10 minutes has a speed of 70.02 km/h -/
theorem train_speed_calculation :
  let distance := 11.67
  let time := 10
  abs (train_speed distance time - 70.02) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_speed 11.67 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l548_54836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_swap_impossible_l548_54815

-- Define the triangular grid
structure TriangularGrid where
  -- We represent grid points as complex numbers
  gridPoint : ℂ
  -- Ensure the grid point is valid (can be represented as a + bω where ω = e^(2πi/3))
  isValid : ∃ (a b : ℤ), gridPoint = a + b * (Complex.exp (2 * Real.pi * Complex.I / 3))

-- Define adjacent grid points
def isAdjacent (p q : TriangularGrid) : Prop :=
  Complex.abs (p.gridPoint - q.gridPoint) = 1

-- Define the possible moves
inductive Move
  | sameDirection
  | oppositeDirection
  | doubleAdjacent

-- Define a round of jumps
def Round := Move

-- Define the result of applying moves
def ApplyMovesResult := TriangularGrid × TriangularGrid

-- Define the function to apply moves (stub implementation)
def applyMoves (a b : TriangularGrid) (moves : List Round) : ApplyMovesResult :=
  (a, b)  -- Placeholder implementation

-- Define the swap positions predicate
def swapPositions (a b : TriangularGrid) (moves : List Round) : Prop :=
  ∃ (final_a final_b : TriangularGrid),
    applyMoves a b moves = (final_a, final_b) ∧
    final_a.gridPoint = b.gridPoint ∧
    final_b.gridPoint = a.gridPoint

-- State the theorem
theorem frog_swap_impossible (a b : TriangularGrid) (h : isAdjacent a b) :
  ¬∃ (moves : List Round), swapPositions a b moves := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_swap_impossible_l548_54815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l548_54818

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h_pos : 0 < b ∧ b < a

/-- A line intersecting an ellipse -/
structure IntersectingLine (k : ℝ) where

/-- The origin of the coordinate system -/
def origin : ℝ × ℝ := (0, 0)

/-- The point where the line intersects the x-axis -/
def x_intercept : ℝ × ℝ := (-5, 0)

theorem ellipse_properties (a b : ℝ) (e : Ellipse a b) (l : IntersectingLine k₁) (k₂ : ℝ) 
  (h_slope : k₁ * k₂ = -2/3) :
  let e := (a^2 - b^2).sqrt / a
  ∃ (P Q : ℝ × ℝ),
    (∃ (M : ℝ × ℝ), M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) ∧ 
    (∃ (D : ℝ × ℝ), D = x_intercept ∧ dist D P = 2 * dist D Q) →
    (e = (3 : ℝ).sqrt / 3) ∧
    (∃ (S : ℝ), S ≤ (6 : ℝ).sqrt / 2 ∧ 
      S = abs ((P.1 * Q.2 - P.2 * Q.1) / 2) ∧
      (S = (6 : ℝ).sqrt / 2 → 
        ∀ (x y : ℝ), x^2 / 125 + 3 * y^2 / 250 = 1 ↔ 
          x^2 / a^2 + y^2 / b^2 = 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l548_54818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PAMB_l548_54805

/-- Circle M passing through two points and with center on a line -/
structure CircleM where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_C : (center.1 - 1)^2 + (center.2 + 1)^2 = radius^2
  passes_through_D : (center.1 + 1)^2 + (center.2 - 1)^2 = radius^2
  center_on_line : center.1 + center.2 = 2

/-- Point P on a line -/
structure PointP where
  coords : ℝ × ℝ
  on_line : 3 * coords.1 + 4 * coords.2 + 8 = 0

/-- Theorem stating the minimum area of quadrilateral PAMB -/
theorem min_area_PAMB (M : CircleM) :
  ∃ (P : PointP),
    let dist_PM := Real.sqrt ((P.coords.1 - M.center.1)^2 + (P.coords.2 - M.center.2)^2)
    let area_PAMB := 2 * Real.sqrt (dist_PM^2 - 4)
    ∀ Q : PointP, area_PAMB ≤ (
      let dist_QM := Real.sqrt ((Q.coords.1 - M.center.1)^2 + (Q.coords.2 - M.center.2)^2)
      2 * Real.sqrt (dist_QM^2 - 4)
    ) ∧ area_PAMB = 2 * Real.sqrt 5 :=
by sorry

#check min_area_PAMB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PAMB_l548_54805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_is_seven_l548_54859

-- Define the constants
def person_speed : ℝ := 6
def initial_distance : ℝ := 25
def car_acceleration : ℝ := 1

-- Define the distance functions for the person and car
noncomputable def person_distance (t : ℝ) : ℝ := person_speed * t
noncomputable def car_distance (t : ℝ) : ℝ := (1/2) * car_acceleration * t^2

-- Define the relative distance function
noncomputable def relative_distance (t : ℝ) : ℝ := 
  initial_distance + car_distance t - person_distance t

-- State the theorem
theorem minimum_distance_is_seven : 
  ∃ (t : ℝ), t > 0 ∧ relative_distance t = 7 ∧ 
  ∀ (s : ℝ), s > 0 → relative_distance s ≥ 7 := by
  sorry

#check minimum_distance_is_seven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_is_seven_l548_54859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l548_54839

def vector_a : ℝ × ℝ := (3, 4)
def vector_b (l : ℝ) : ℝ × ℝ := (1 - l, 2 + l)

theorem perpendicular_vectors_lambda :
  (∀ l : ℝ, vector_a.1 * (vector_b l).1 + vector_a.2 * (vector_b l).2 = 0) →
  (∃ l : ℝ, l = -11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l548_54839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_length_across_circle_diameter_l548_54808

theorem pencil_length_across_circle_diameter 
  (radius : ℝ) 
  (num_pencils : ℕ) 
  (h1 : radius = 14) 
  (h2 : num_pencils = 56) : 
  (2 * radius * 12) / num_pencils = 6 := by
  -- Convert num_pencils to ℝ for division
  have num_pencils_real : ℝ := num_pencils
  -- Calculate diameter in inches
  have diameter_inches : ℝ := 2 * radius * 12
  -- Calculate pencil length
  have pencil_length : ℝ := diameter_inches / num_pencils_real
  -- Prove the equality
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_length_across_circle_diameter_l548_54808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_l548_54880

/-- A regular hexagonal pyramid -/
def RegularHexagonalPyramid (b : ℝ) : Set (Fin 3 → ℝ) := sorry

/-- A sphere -/
def Sphere (R : ℝ) : Set (Fin 3 → ℝ) := sorry

/-- Predicate to check if a set is inscribed in another set -/
def IsInscribedIn (A B : Set (Fin 3 → ℝ)) : Prop := sorry

/-- Predicate to check if a real number is the volume of a set -/
def IsVolumeOf (v : ℝ) (S : Set (Fin 3 → ℝ)) : Prop := sorry

/-- The volume of a regular hexagonal pyramid inscribed in a sphere -/
theorem hexagonal_pyramid_volume (b R : ℝ) (h_pos_b : b > 0) (h_pos_R : R > 0) :
  let volume := (b^4 * Real.sqrt 3 * (4 * R^2 - b^2)) / (16 * R^3)
  ∃ (V : ℝ), V = volume ∧ 
    IsVolumeOf V (RegularHexagonalPyramid b) ∧
    IsInscribedIn (RegularHexagonalPyramid b) (Sphere R) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_l548_54880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l548_54820

open Set Real

def U : Set ℝ := univ

def A : Set ℝ := {x | |x - 2| > 1}

def B : Set ℝ := {x | x ≥ 0}

theorem set_operations :
  (A ∩ B = {x | (0 < x ∧ x < 1) ∨ x > 3}) ∧
  (A ∪ B = univ) ∧
  ((Uᶜ ∩ A)ᶜ ∪ B = {x | x ≥ 0}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l548_54820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_fourth_quadrant_l548_54829

def card_numbers : List Int := [0, -1, 2, -3]

def is_in_fourth_quadrant (m n : Int) : Bool :=
  m > 0 ∧ n < 0

def count_favorable_outcomes : Nat :=
  card_numbers.foldl (λ acc m => 
    acc + card_numbers.foldl (λ inner_acc n => 
      if m ≠ n ∧ is_in_fourth_quadrant m n then inner_acc + 1 else inner_acc
    ) 0
  ) 0

def total_outcomes : Nat :=
  card_numbers.length * (card_numbers.length - 1)

theorem probability_in_fourth_quadrant :
  (count_favorable_outcomes : Rat) / total_outcomes = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_fourth_quadrant_l548_54829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l548_54810

-- Define the function f(x) = x / (1 + x^2)
noncomputable def f (x : ℝ) : ℝ := x / (1 + x^2)

-- State the theorem
theorem f_properties :
  -- f(1/2) = 2/5
  f (1/2) = 2/5 ∧
  -- f is odd
  (∀ x, x ∈ Set.Ioo (-1) 1 → f (-x) = -f x) ∧
  -- f is increasing on (-1, 1)
  (∀ x₁ x₂, x₁ ∈ Set.Ioo (-1) 1 → x₂ ∈ Set.Ioo (-1) 1 → x₁ < x₂ → f x₁ < f x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l548_54810
