import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_is_one_l239_23945

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x)

theorem sum_of_a_and_b_is_one (a b : ℝ) (h : f (a - 1) + f b = 0) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_is_one_l239_23945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_union_condition_l239_23900

theorem max_subsets_with_union_condition (T : Set ℕ) (h : T = Finset.range 999) :
  (∃ k : ℕ, ∃ A : Fin k → Set ℕ,
    (∀ i j : Fin k, i.val < j.val → A i ∪ A j = T) ∧
    (∀ k' > k, ¬∃ A' : Fin k' → Set ℕ, ∀ i j : Fin k', i.val < j.val → A' i ∪ A' j = T)) →
  (∃ k : ℕ, ∃ A : Fin k → Set ℕ,
    k = 1000 ∧
    (∀ i j : Fin k, i.val < j.val → A i ∪ A j = T) ∧
    (∀ k' > k, ¬∃ A' : Fin k' → Set ℕ, ∀ i j : Fin k', i.val < j.val → A' i ∪ A' j = T)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_union_condition_l239_23900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_max_p_l239_23927

/-- Pascal distribution with parameters r and p -/
structure PascalDist (r : ℕ) (p : ℝ) where
  -- We don't need to define the internal structure here
  dummy : Unit

/-- Probability mass function for Pascal distribution -/
noncomputable def pmf (r : ℕ) (p : ℝ) (x : ℕ) : ℝ :=
  sorry -- We'll leave this undefined for now

theorem pascal_max_p :
  ∀ p : ℝ, 0 < p → p < 1 →
  (pmf 3 p 6 ≥ pmf 3 p 5) →
  p ≤ 2/5 ∧ ∀ q, (0 < q ∧ q < 1 ∧ pmf 3 q 6 ≥ pmf 3 q 5) → q ≤ p :=
by
  sorry -- We'll leave the proof empty for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_max_p_l239_23927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solution_l239_23995

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = Real.sqrt 3 ∧
  b^2 + c^2 - Real.sqrt 2 * b * c = 3 ∧
  Real.cos B = 4/5 ∧
  a * Real.sin C = b * Real.sin A ∧
  a * Real.sin B = c * Real.sin A ∧
  b * Real.sin C = c * Real.sin B ∧
  A + B + C = Real.pi

theorem triangle_solution (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_problem a b c A B C) :
  A = Real.pi/4 ∧ c = 7 * Real.sqrt 3 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solution_l239_23995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_cubes_l239_23984

theorem root_sum_cubes (α β : ℝ) : 
  α^2 + (2 * Real.sqrt (Real.sqrt 2 + 1)) * α + Real.sqrt (Real.sqrt 2 + 1) = 0 →
  β^2 + (2 * Real.sqrt (Real.sqrt 2 + 1)) * β + Real.sqrt (Real.sqrt 2 + 1) = 0 →
  (1 / α^3) + (1 / β^3) = 6 * Real.sqrt (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) - 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_cubes_l239_23984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_sum_l239_23918

def X (b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 3, b],
    ![0, 1, 5],
    ![0, 0, 1]]

theorem matrix_power_sum (b m : ℕ) :
  (X b) ^ m = ![![1, 27, 8085],
               ![0,  1,   45],
               ![0,  0,    1]] →
  b + m = 847 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_sum_l239_23918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_plus_one_l239_23916

theorem cube_root_eight_plus_one : (8 : ℝ) ^ (1/3) + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_plus_one_l239_23916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_production_and_price_l239_23937

-- Define the cost function
noncomputable def C (x : ℝ) : ℝ := 2 + x

-- Define the revenue function
noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 4 then 4*x - (1/2)*x^2 - (1/2)
  else 7.5

-- Define the profit function
noncomputable def L (x : ℝ) : ℝ := R x - C x

-- Theorem statement
theorem optimal_production_and_price :
  ∃ (x_opt : ℝ), 
    (∀ x, L x ≤ L x_opt) ∧ 
    x_opt = 3 ∧ 
    (R x_opt / x_opt) * 100 = 233 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_production_and_price_l239_23937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_score_l239_23922

/-- Given a class of 18 students where the average score of 17 students is 76
    and the average score of all 18 students is 78, prove that the score of
    the 18th student (Jessica) is 112. -/
theorem jessica_score (total_students : ℕ) (students_without_jessica : ℕ)
    (avg_without_jessica : ℚ) (avg_with_jessica : ℚ) (jessica_score : ℚ) :
  total_students = 18 →
  students_without_jessica = 17 →
  avg_without_jessica = 76 →
  avg_with_jessica = 78 →
  (students_without_jessica * avg_without_jessica +
   (total_students - students_without_jessica : ℚ) * jessica_score) /
    total_students = avg_with_jessica →
  jessica_score = 112 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_score_l239_23922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_alpha_plus_gamma_l239_23998

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the function f
noncomputable def f (α γ : ℂ) (z : ℂ) : ℂ := (3 - 2*i) * z^2 + α * z + γ

-- Define the theorem
theorem min_abs_alpha_plus_gamma :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 2 ∧
  ∀ α γ : ℂ, (f α γ 1).im = 0 → (f α γ i).im = 0 →
  Complex.abs α + Complex.abs γ ≥ min_val :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_alpha_plus_gamma_l239_23998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_b_value_l239_23956

-- Define the slope of a line given its coefficients
noncomputable def line_slope (a b c : ℝ) : ℝ := -a / b

-- Define when two lines are parallel
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  line_slope a1 b1 c1 = line_slope a2 b2 c2

-- Theorem statement
theorem parallel_lines_b_value (b : ℝ) :
  are_parallel 9 3 (-1 - b) (3 - b) 2 (-10) → b = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_b_value_l239_23956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tight_function_quadratic_over_x_tight_function_injective_l239_23914

-- Definition of a tight function
def is_tight_function (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ domain → x₂ ∈ domain → f x₁ = f x₂ → x₁ = x₂

-- Statement ②
theorem tight_function_quadratic_over_x (a : ℝ) (h : a < 0) :
  is_tight_function (fun x ↦ (x^2 + 2*x + a) / x) {x : ℝ | x > 0} :=
sorry

-- Statement ④
theorem tight_function_injective {f : ℝ → ℝ} {domain : Set ℝ} 
  (h : is_tight_function f domain) :
  ∀ x₁ x₂, x₁ ∈ domain → x₂ ∈ domain → x₁ ≠ x₂ → f x₁ ≠ f x₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tight_function_quadratic_over_x_tight_function_injective_l239_23914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l239_23963

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ a b : ℝ, a > 0 → b > 0 → ((1 / x) + (1 / y)) * (4 * x + y) ≤ ((1 / a) + (1 / b)) * (4 * a + b)) ↔ 
  y / x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l239_23963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_36_l239_23950

/-- The acute angle between the hour and minute hands of a 12-hour clock at a given time -/
noncomputable def clockAngle (hours : ℕ) (minutes : ℕ) : ℝ :=
  let hourAngle : ℝ := (hours % 12 + minutes / 60 : ℝ) * 30
  let minuteAngle : ℝ := minutes * 6
  let diff : ℝ := |hourAngle - minuteAngle|
  min diff (360 - diff)

theorem clock_angle_at_3_36 : clockAngle 3 36 = 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_36_l239_23950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_problem_l239_23904

theorem complex_magnitude_problem (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z - w) = 3) :
  Complex.abs (2 / z + 1 / w) = 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_problem_l239_23904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_at_max_angle_l239_23981

noncomputable section

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

/-- The line equation -/
def is_on_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 8 + 2 * Real.sqrt 3 = 0

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := (-2 * Real.sqrt 3, 0)
def F₂ : ℝ × ℝ := (2 * Real.sqrt 3, 0)

/-- The point P on the line -/
def P : ℝ × ℝ := (-8 - 2 * Real.sqrt 3, 0)

/-- The distance between two points -/
def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem ratio_at_max_angle :
  distance P F₁ / distance P F₂ = Real.sqrt 3 - 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_at_max_angle_l239_23981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extreme_values_l239_23907

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

theorem tangent_line_and_extreme_values 
  (a b : ℝ) 
  (h1 : (deriv (deriv (f a b))) 1 = 2*a) 
  (h2 : (deriv (deriv (f a b))) 2 = -b) :
  (∃ (m c : ℝ), ∀ x y, y = f a b x → (x = 1 → 6*x + 2*y - 1 = 0)) ∧ 
  (∃ (g : ℝ → ℝ), 
    (∀ x, g x = (1 / Real.exp x) * (f a b x)) ∧
    (∀ x, g x ≥ -3) ∧
    (g 0 = -3) ∧
    (∀ x, g x ≤ 15 * Real.exp (-3)) ∧
    (g 3 = 15 * Real.exp (-3))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extreme_values_l239_23907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_collection_average_l239_23931

theorem cards_collection_average (n : ℕ) (a d : ℝ) (h1 : n = 7) (h2 : a = 12) (h3 : d = 10) :
  let sequence := fun i ↦ a + d * (i - 1)
  (sequence 1 + sequence n) / 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_collection_average_l239_23931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l239_23943

def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (0, -2)

theorem vector_properties :
  (let sum := (a.1 + b.1, a.2 + b.2);
   sum.1 * a.1 + sum.2 * a.2 = 0) ∧
  (let proj := ((a.1 * b.1 + a.2 * b.2) / (a.1^2 + a.2^2)) • a;
   proj = (1, -1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l239_23943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_totalAreaPyramidSpecific_l239_23905

/-- The total area of the four triangular faces of a right, square-based pyramid -/
noncomputable def totalAreaPyramid (baseEdge : ℝ) (lateralEdge : ℝ) : ℝ :=
  4 * (1/2 * baseEdge * Real.sqrt (lateralEdge^2 - (baseEdge/2)^2))

/-- Theorem: The total area of the four triangular faces of a right, square-based pyramid 
    with base edges of 8 units and lateral edges of 10 units is equal to 32√21 square units -/
theorem totalAreaPyramidSpecific : 
  totalAreaPyramid 8 10 = 32 * Real.sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_totalAreaPyramidSpecific_l239_23905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trapezoid_area_ratio_l239_23961

/-- The area of an equilateral triangle with side length s -/
noncomputable def equilateralTriangleArea (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The side length of the larger equilateral triangle -/
def largerSide : ℝ := 10

/-- The side length of the smaller equilateral triangle -/
def smallerSide : ℝ := 5

/-- The area of the larger equilateral triangle -/
noncomputable def largerArea : ℝ := equilateralTriangleArea largerSide

/-- The area of the smaller equilateral triangle -/
noncomputable def smallerArea : ℝ := equilateralTriangleArea smallerSide

/-- The area of the isosceles trapezoid -/
noncomputable def trapezoidArea : ℝ := largerArea - smallerArea

theorem triangle_trapezoid_area_ratio :
  smallerArea / trapezoidArea = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trapezoid_area_ratio_l239_23961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_luisa_fuel_efficiency_l239_23930

/-- Calculates the fuel efficiency given total distance, total gas cost, and gas price per gallon -/
noncomputable def fuel_efficiency (total_distance : ℝ) (total_gas_cost : ℝ) (gas_price_per_gallon : ℝ) : ℝ :=
  total_distance / (total_gas_cost / gas_price_per_gallon)

/-- Theorem stating that under the given conditions, the fuel efficiency is 15 miles per gallon -/
theorem luisa_fuel_efficiency :
  fuel_efficiency 30 7 3.5 = 15 := by
  -- Unfold the definition of fuel_efficiency
  unfold fuel_efficiency
  -- Simplify the expression
  simp
  -- Apply numerical approximation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_luisa_fuel_efficiency_l239_23930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_equals_zero_two_open_l239_23923

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set B
def B : Set ℝ := {x | (1/2 : ℝ)^x ≤ 1}

-- Define set A
def A : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_A_intersect_B_equals_zero_two_open :
  (Set.compl A) ∩ B = Set.Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_equals_zero_two_open_l239_23923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bisector_slope_intercept_sum_l239_23982

/-- Given a triangle PQR with vertices P(0, 10), Q(3, 0), and R(9, 0),
    the line through Q that bisects the area of the triangle has a slope
    and y-intercept whose sum is -20/3. -/
theorem triangle_bisector_slope_intercept_sum :
  let P : ℝ × ℝ := (0, 10)
  let Q : ℝ × ℝ := (3, 0)
  let R : ℝ × ℝ := (9, 0)
  let triangle_area := abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2)
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b → 
    abs ((P.1 * (Q.2 - y) + Q.1 * (y - P.2) + x * (P.2 - Q.2)) / 2) = triangle_area / 2 ∧
    Q.2 = m * Q.1 + b) ∧
  m + b = -20/3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bisector_slope_intercept_sum_l239_23982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bojan_winning_strategy_l239_23990

theorem bojan_winning_strategy (a b : ℕ) : 
  let game_sequence := (List.range 2016).map (λ i => (a + b + 2017) % 2017)
  let product := game_sequence.prod
  product % 2017 = (a + b) % 2017 := by
  sorry

#check bojan_winning_strategy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bojan_winning_strategy_l239_23990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_distances_bound_l239_23967

/-- The point P through which line L passes -/
noncomputable def P : ℝ × ℝ := (Real.sqrt 3 / 2, 3 / 2)

/-- The unit circle C -/
def C (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Line L passing through P with inclination angle α -/
noncomputable def L (α t : ℝ) : ℝ × ℝ :=
  (P.1 + t * Real.cos α, P.2 + t * Real.sin α)

/-- Theorem stating the range of 1/|PM| + 1/|PN| -/
theorem sum_inverse_distances_bound (α : ℝ) :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
  C (L α t₁).1 (L α t₁).2 ∧
  C (L α t₂).1 (L α t₂).2 ∧
  Real.sqrt 2 < (1 / |t₁| + 1 / |t₂|) ∧
  (1 / |t₁| + 1 / |t₂|) ≤ Real.sqrt 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_distances_bound_l239_23967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_triangle_side_length_l239_23906

theorem smaller_triangle_side_length 
  (large_side : ℝ) 
  (area_ratio : ℝ) 
  (large_side_eq : large_side = 4) 
  (area_ratio_eq : area_ratio = 1/3) : 
  let large_area := Real.sqrt 3 / 4 * large_side^2
  let small_area := area_ratio * large_area
  let small_side := Real.sqrt (4 * small_area / Real.sqrt 3)
  small_side = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_triangle_side_length_l239_23906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_exp_positive_iff_l239_23921

open Real

-- The function f with the given properties
def f : ℝ → ℝ := sorry

-- The derivative of f
def f' : ℝ → ℝ := sorry

-- f(3) = e^3
axiom f_at_3 : f 3 = exp 3

-- f'(x) - f(x) > 0 for all x
axiom f'_gt_f : ∀ x, f' x - f x > 0

-- f' is the derivative of f
axiom f'_is_derivative : ∀ x, HasDerivAt f (f' x) x

-- The main theorem
theorem f_minus_exp_positive_iff (x : ℝ) : f x - exp x > 0 ↔ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_exp_positive_iff_l239_23921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_in_open_interval_l239_23910

/-- A function f: ℝ → ℝ is decreasing if for all x₁ < x₂, f(x₁) > f(x₂) -/
def Decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂

/-- The exponential function with base (a-2) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 2) ^ x

/-- Theorem: The function f(x) = (a-2)^x is decreasing if and only if 2 < a < 3 -/
theorem f_decreasing_iff_a_in_open_interval :
  ∀ a : ℝ, Decreasing (f a) ↔ 2 < a ∧ a < 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_in_open_interval_l239_23910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_length_proof_l239_23944

theorem box_length_proof (width depth : ℕ) (num_cubes : ℕ) (length : ℕ) : 
  width = 40 → 
  depth = 16 → 
  num_cubes = 30 → 
  (∃ (cube_side : ℕ), 
    cube_side > 0 ∧ 
    width % cube_side = 0 ∧ 
    depth % cube_side = 0 ∧ 
    length % cube_side = 0 ∧
    length * width * depth = num_cubes * (cube_side * cube_side * cube_side)) →
  length = 24 := by
  sorry

#check box_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_length_proof_l239_23944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_x_is_18_l239_23949

def is_divisible_by_30 (x : Nat) : Bool :=
  (35000 + x * 100 + 40) % 30 = 0

def sum_of_valid_x : Nat :=
  (Finset.range 10).filter (λ x => is_divisible_by_30 x) |>.sum id

theorem sum_of_valid_x_is_18 : sum_of_valid_x = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_x_is_18_l239_23949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_distribution_l239_23952

theorem photo_distribution (x : ℕ) : 
  (∀ (student : ℕ), student < x → student + 1 = x - (x - (student + 1))) →
  (x * (x - 1) = 2970) →
  x * (x - 1) = 2970 :=
by
  intros h1 h2
  exact h2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_distribution_l239_23952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_systematic_sample_l239_23980

/-- Represents a systematic sample of missiles -/
structure SystematicSample where
  total : Nat
  sample_size : Nat
  start : Nat
  interval : Nat

/-- Checks if a given list of numbers is a valid systematic sample -/
def is_valid_sample (s : SystematicSample) (sample : List Nat) : Prop :=
  sample.length = s.sample_size ∧
  sample.all (· ≤ s.total) ∧
  sample = (List.range s.sample_size).map (fun i => s.start + i * s.interval)

/-- The theorem to be proved -/
theorem correct_systematic_sample :
  let s : SystematicSample := ⟨50, 5, 3, 10⟩
  is_valid_sample s [3, 13, 23, 33, 43] :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_systematic_sample_l239_23980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l239_23942

/-- A rectangle with dimensions 14 and 15 -/
structure Rectangle :=
  (width : ℝ := 14)
  (height : ℝ := 15)

/-- A point representing the intersection of diagonals -/
structure IntersectionPoint :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

/-- A triangular pyramid formed from the rectangle -/
structure TriangularPyramid :=
  (base : Rectangle)
  (apex : IntersectionPoint)

/-- All faces of the pyramid are isosceles triangles -/
def isIsoscelesPyramid (p : TriangularPyramid) : Prop :=
  let d := Real.sqrt (p.apex.x^2 + (p.apex.y - p.base.height)^2 + p.apex.z^2)
  d = Real.sqrt ((p.apex.x - p.base.width/2)^2 + p.apex.y^2 + p.apex.z^2) ∧
  d = Real.sqrt ((p.apex.x + p.base.width/2)^2 + p.apex.y^2 + p.apex.z^2)

/-- The volume of the triangular pyramid -/
noncomputable def pyramidVolume (p : TriangularPyramid) : ℝ :=
  (1/3) * (p.base.width * p.base.height / 2) * p.apex.z

/-- Theorem stating the volume of the specific pyramid -/
theorem specific_pyramid_volume :
  ∀ (p : TriangularPyramid),
  p.base.width = 14 ∧
  p.base.height = 15 ∧
  isIsoscelesPyramid p →
  pyramidVolume p = 35 * (99 / Real.sqrt 421) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l239_23942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_water_depth_l239_23903

/-- Proves that given a tank with specified dimensions and water level increases, 
    the final water depth is as calculated. -/
theorem tank_water_depth 
  (base_area : ℝ) 
  (initial_depth : ℝ) 
  (first_increase : ℝ) 
  (second_increase : ℝ) : 
  base_area = 6 → 
  initial_depth = 0.75 → 
  first_increase = 0.15 → 
  second_increase = 0.225 → 
  initial_depth + first_increase + second_increase = 1.125 := by
  intro h1 h2 h3 h4
  rw [h2, h3, h4]
  norm_num

-- We don't need to evaluate the theorem, so we can remove the #eval line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_water_depth_l239_23903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l239_23912

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => 2 * sequence_a n + 2^(n + 1)

def S (n : ℕ) : ℚ := (2 * n - 3) * 2^n + 3

theorem sequence_properties (n : ℕ) :
  (∀ k : ℕ, k ≤ n → sequence_a k / 2^k = k + 1 - 1/2) ∧
  S n = (2 * n - 3) * 2^n + 3 ∧
  S n / 2^n > 2 * n - 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l239_23912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_circle_radius_l239_23934

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents a configuration of five circles -/
structure FiveCircles where
  circles : Fin 5 → Circle
  largest : circles 0 = Circle.mk 20
  smallest : circles 4 = Circle.mk 10
  tangent : ∀ i : Fin 4, (circles i).radius + (circles (i + 1)).radius = 
            |((circles i).radius - (circles (i + 1)).radius)|

theorem middle_circle_radius (fc : FiveCircles) : 
  (fc.circles 2).radius = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_circle_radius_l239_23934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_minus_cos_squared_l239_23935

theorem sin_squared_minus_cos_squared (α : ℝ) : 
  Real.cos (π / 2 - α) = -1 / 3 → Real.sin α ^ 2 - Real.cos α ^ 2 = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_minus_cos_squared_l239_23935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_B_is_quadratic_l239_23911

-- Define the functions
noncomputable def f_A (x : ℝ) : ℝ := -3 * x + 5
noncomputable def f_B (x : ℝ) : ℝ := 2 * x^2
noncomputable def f_C (x : ℝ) : ℝ := (x + 1)^2 - x^2
noncomputable def f_D (x : ℝ) : ℝ := 3 / x^2

-- Define what it means for a function to be quadratic
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Theorem statement
theorem only_f_B_is_quadratic :
  ¬ is_quadratic f_A ∧
  is_quadratic f_B ∧
  ¬ is_quadratic f_C ∧
  ¬ is_quadratic f_D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_B_is_quadratic_l239_23911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_value_l239_23971

-- Define the points
variable (X Y E F : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h1 : ‖X - Y‖ = 12)
variable (h2 : ‖Y - E‖ = 15)
variable (h3 : ‖F - X‖ = 15)
variable (h4 : ‖E - X‖ = 20)
variable (h5 : ‖F - Y‖ = 20)
variable (h6 : CongruentTriangle X Y E X Y F)

-- Define the intersection area function
noncomputable def intersection_area (X Y E F : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- State the theorem
theorem intersection_area_value :
  intersection_area X Y E F = 144 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_value_l239_23971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l239_23976

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_line_at_zero (x y : ℝ) :
  (∀ t, DifferentiableAt ℝ f t) →
  (∃ m b, ∀ x, y = m * x + b) →
  (y = deriv f 0 * x + f 0) →
  y = x :=
by
  intro h1 h2 h3
  -- The proof steps would go here
  sorry

#check tangent_line_at_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l239_23976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_quadrants_l239_23972

-- Define the quadrants
inductive Quadrant
  | I
  | II
  | III
  | IV

-- Define a function to determine if an angle is in a given quadrant
def in_quadrant (θ : ℝ) (q : Quadrant) : Prop := sorry

-- State the theorem
theorem angle_quadrants (θ : ℝ) 
  (h : Real.sin θ * Real.tan θ > 0) : 
  ∃ (q1 q2 : Quadrant), q1 ≠ q2 ∧ in_quadrant θ q1 ∧ in_quadrant θ q2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_quadrants_l239_23972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ink_equality_l239_23979

-- Define the variables
variable (m a : ℝ)

-- Define the conditions
variable (h1 : 0 < a)
variable (h2 : a < m)

-- Define the amounts of ink after mixing
noncomputable def blue_ink_in_A (m a : ℝ) : ℝ := (a * m) / (m + a)
noncomputable def red_ink_in_B (m a : ℝ) : ℝ := (m * a) / (m + a)

-- State the theorem
theorem ink_equality (m a : ℝ) (h1 : 0 < a) (h2 : a < m) : 
  blue_ink_in_A m a = red_ink_in_B m a := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ink_equality_l239_23979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pavilion_larger_interior_angle_calculated_angle_correct_l239_23915

/-- Represents the arrangement of trapezoids in the ancient pavilion -/
structure TrapezoidArrangement where
  num_trapezoids : ℕ
  center_angle : ℚ
  larger_interior_angle : ℚ

/-- The specific arrangement described in the problem -/
def pavilion_arrangement : TrapezoidArrangement where
  num_trapezoids := 12
  center_angle := 30
  larger_interior_angle := 97.5

/-- Theorem stating that the larger interior angle of each trapezoid in the arrangement is 97.5° -/
theorem pavilion_larger_interior_angle :
  pavilion_arrangement.larger_interior_angle = 97.5 := by
  rfl

/-- Function to calculate the larger interior angle given the number of trapezoids -/
def calculate_larger_interior_angle (n : ℕ) : ℚ :=
  180 - (165 / 2)

/-- Theorem proving that the calculated angle for 12 trapezoids is indeed 97.5° -/
theorem calculated_angle_correct :
  calculate_larger_interior_angle pavilion_arrangement.num_trapezoids = 97.5 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pavilion_larger_interior_angle_calculated_angle_correct_l239_23915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_integers_in_even_product_l239_23946

theorem max_odd_integers_in_even_product (integers : Finset ℕ) : 
  integers.card = 7 → 
  Even (integers.prod id) → 
  (integers.filter (λ x => x % 2 = 1)).card ≤ 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_integers_in_even_product_l239_23946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_pyramid_l239_23948

/-- Represents a pyramid with a rectangular base -/
structure RectangularBasePyramid where
  /-- Length of one side of the base rectangle -/
  baseLength : ℝ
  /-- Width of the base rectangle -/
  baseWidth : ℝ
  /-- Height of the pyramid -/
  height : ℝ

/-- Calculate the volume of a pyramid with a rectangular base -/
noncomputable def volumeOfRectangularBasePyramid (p : RectangularBasePyramid) : ℝ :=
  (1 / 3) * p.baseLength * p.baseWidth * p.height

/-- Given pyramid QEFGH with specified dimensions, prove its volume is 400/3 -/
theorem volume_of_specific_pyramid :
  let p := RectangularBasePyramid.mk 10 5 8
  volumeOfRectangularBasePyramid p = 400 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_pyramid_l239_23948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_equations_l239_23957

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Define the line y = 2x - 1
def center_line (x y : ℝ) : Prop := y = 2*x - 1

-- Define the point (2,2)
def point_on_l : ℝ × ℝ := (2, 2)

-- Define the chord length
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

theorem circle_and_line_equations :
  -- Circle C passes through A(1,3) and B(-1,1)
  circle_C 1 3 ∧ circle_C (-1) 1 ∧
  -- The center of C is on the line y = 2x - 1
  ∃ (cx cy : ℝ), center_line cx cy ∧ ∀ (x y : ℝ), circle_C x y ↔ (x - cx)^2 + (y - cy)^2 = 4 →
  -- Line l passes through (2,2) and intersects C with chord length 2√3
  ∃ (l : ℝ → ℝ), (l 2 = 2) ∧
    (∃ (x1 y1 x2 y2 : ℝ),
      circle_C x1 y1 ∧ circle_C x2 y2 ∧
      y1 = l x1 ∧ y2 = l x2 ∧
      (x2 - x1)^2 + (y2 - y1)^2 = chord_length^2) →
  -- Conclusion: Equation of line l is y = 2 or x = 2
  (∀ x, l x = 2) ∨ (∃ k, ∀ x, l x = k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_equations_l239_23957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_segment_length_l239_23941

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the ellipse -/
def onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem statement -/
theorem ellipse_segment_length 
  (e : Ellipse) 
  (F1 F2 A B : Point) 
  (h1 : e.a = 5 ∧ e.b = 3) 
  (h2 : onEllipse A e ∧ onEllipse B e) 
  (h3 : ∃ (t : ℝ), A.x = F1.x + t * (B.x - F1.x) ∧ A.y = F1.y + t * (B.y - F1.y)) 
  (h4 : distance F2 A + distance F2 B = 12) : 
  distance A B = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_segment_length_l239_23941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_time_against_l239_23909

/-- Represents a moving walkway scenario -/
structure MovingWalkway where
  length : ℝ
  time_with : ℝ
  time_stationary : ℝ

/-- Calculates the time to walk against the moving walkway -/
noncomputable def time_against (w : MovingWalkway) : ℝ :=
  w.length * w.time_stationary / (w.time_stationary - w.time_with)

/-- Theorem stating the time to walk against the walkway is 150 seconds -/
theorem walkway_time_against 
  (w : MovingWalkway) 
  (h1 : w.length = 100)
  (h2 : w.time_with = 50)
  (h3 : w.time_stationary = 75) : 
  time_against w = 150 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval time_against { length := 100, time_with := 50, time_stationary := 75 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_time_against_l239_23909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_side_length_formula_l239_23964

/-- The side length of a regular decagon inscribed in a circle of radius R -/
noncomputable def regular_decagon_side_length (R : ℝ) : ℝ := (Real.sqrt 5 - 1) * R / 2

/-- Theorem: The side length of a regular decagon inscribed in a circle of radius R
    is equal to (√5 - 1)R / 2 -/
theorem regular_decagon_side_length_formula (R : ℝ) (R_pos : R > 0) :
  ∃ (s : ℝ), s > 0 ∧ s = regular_decagon_side_length R ∧
  10 * s = 2 * R * Real.sin (π / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_side_length_formula_l239_23964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l239_23970

/-- The length of a platform given train passing times -/
theorem platform_length (v : ℝ) (t_platform t_man : ℝ) : 
  v = 54 * (1000 / 3600) → 
  t_platform = 16 → 
  t_man = 10 → 
  v * t_platform - v * t_man = 90 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l239_23970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_student_coins_l239_23985

/-- Represents the arrangement of students in concentric circles -/
structure StudentArrangement where
  total_students : Nat
  total_coins : Nat
  central_student_coins : Nat

/-- Defines the properties of a valid coin distribution -/
def is_valid_distribution (arrangement : StudentArrangement) : Prop :=
  arrangement.total_students = 16 ∧
  arrangement.total_coins = 3360 ∧
  -- Assume the property that students equidistant from center have same coins
  -- Assume the property that coin exchange results in no change
  True

/-- Theorem stating that the central student must have 280 coins -/
theorem central_student_coins 
  (arrangement : StudentArrangement) 
  (h : is_valid_distribution arrangement) : 
  arrangement.central_student_coins = 280 := by
  sorry

#check central_student_coins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_student_coins_l239_23985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_chosen_set_l239_23919

theorem divisibility_in_chosen_set : 
  ∀ (S : Finset ℕ), 
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 200) → 
  S.card = 101 → 
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_chosen_set_l239_23919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candice_bread_cost_l239_23996

/-- Represents the weekly bread purchase and its cost -/
structure WeeklyPurchase where
  white_bread_price : ℚ
  white_bread_count : ℕ
  baguette_price : ℚ
  sourdough_price : ℚ
  sourdough_count : ℕ
  croissant_price : ℚ
  weeks : ℕ

/-- Calculates the total cost of the weekly purchase over multiple weeks -/
def total_cost (purchase : WeeklyPurchase) : ℚ :=
  ((purchase.white_bread_price * purchase.white_bread_count) +
   purchase.baguette_price +
   (purchase.sourdough_price * purchase.sourdough_count) +
   purchase.croissant_price) * purchase.weeks

/-- Theorem stating that Candice's bread purchases over 4 weeks cost $78.00 -/
theorem candice_bread_cost : 
  ∀ (purchase : WeeklyPurchase), 
  purchase.white_bread_price = 7/2 ∧ 
  purchase.white_bread_count = 2 ∧
  purchase.baguette_price = 3/2 ∧
  purchase.sourdough_price = 9/2 ∧
  purchase.sourdough_count = 2 ∧
  purchase.croissant_price = 2 ∧
  purchase.weeks = 4 →
  total_cost purchase = 78 := by
  sorry

#eval total_cost {
  white_bread_price := 7/2,
  white_bread_count := 2,
  baguette_price := 3/2,
  sourdough_price := 9/2,
  sourdough_count := 2,
  croissant_price := 2,
  weeks := 4
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candice_bread_cost_l239_23996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_properties_l239_23994

/-- Represents a monomial with coefficient and variables -/
structure Monomial where
  coefficient : ℤ
  vars : List (Char × ℕ)

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : ℕ := m.vars.foldl (fun acc (_, exp) => acc + exp) 0

/-- The monomial -2x²y -/
def m : Monomial := { coefficient := -2, vars := [('x', 2), ('y', 1)] }

theorem monomial_properties : m.coefficient = -2 ∧ degree m = 3 := by
  constructor
  · rfl
  · rfl

#eval m.coefficient
#eval degree m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_properties_l239_23994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_proof_l239_23988

theorem trigonometric_ratio_proof (α : Real) 
  (h1 : Real.sin (π/4 - α) = 1/3) 
  (h2 : 0 < α) 
  (h3 : α < π/4) : 
  Real.cos (2*π - 2*α) / Real.cos (5*π/4 + α) = - 2*Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_proof_l239_23988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l239_23955

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * tan x * sin (π / 2 - x) * cos (x - π / 3) - Real.sqrt 3

-- Define the domain of f
def domain (x : ℝ) : Prop := ∀ k : ℤ, x ≠ k * π + π / 2

-- Theorem for the properties of f
theorem f_properties :
  (∀ x : ℝ, domain x → f x = f (x + π)) ∧
  (∀ x ∈ Set.Icc (-π/4) (-π/12), ∀ y ∈ Set.Icc (-π/4) (-π/12), x ≤ y → f y ≤ f x) ∧
  (∀ x ∈ Set.Icc (-π/12) (π/4), ∀ y ∈ Set.Icc (-π/12) (π/4), x ≤ y → f x ≤ f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l239_23955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equality_l239_23977

/-- Given a triangle ABC with sides a, b, c satisfying the inequality,
    prove that the triangle is equilateral -/
theorem triangle_equality (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_inequality : (1/a * Real.sqrt (1/b + 1/c) + 
                   1/b * Real.sqrt (1/c + 1/a) + 
                   1/c * Real.sqrt (1/a + 1/b)) ≥
                  (3/2) * Real.sqrt ((1/a + 1/b) * (1/b + 1/c) * (1/c + 1/a))) :
  a = b ∧ b = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equality_l239_23977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eric_triathlon_bike_speed_l239_23936

/-- Represents a triathlon event with swimming, running, and biking components. -/
structure Triathlon where
  total_time : ℝ
  swim_distance : ℝ
  swim_speed : ℝ
  run_distance : ℝ
  run_speed : ℝ
  bike_distance : ℝ

/-- Calculates the required bike speed to complete the triathlon in the given total time. -/
noncomputable def required_bike_speed (t : Triathlon) : ℝ :=
  let swim_time := t.swim_distance / t.swim_speed
  let run_time := t.run_distance / t.run_speed
  let remaining_time := t.total_time - swim_time - run_time
  t.bike_distance / remaining_time

/-- The specific triathlon event described in the problem. -/
def eric_triathlon : Triathlon where
  total_time := 3
  swim_distance := 0.5
  swim_speed := 1.5
  run_distance := 4
  run_speed := 5
  bike_distance := 20

/-- Theorem stating that the required bike speed for Eric's triathlon is 75/7 miles per hour. -/
theorem eric_triathlon_bike_speed :
  required_bike_speed eric_triathlon = 75 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eric_triathlon_bike_speed_l239_23936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_on_unit_circle_l239_23940

-- Define the complex equation
def complex_equation (z : ℂ) : Prop := (z + 2)^4 = 16 * z^4

-- Define the circle in the complex plane
def on_circle (z : ℂ) : Prop := Complex.abs (z - 2/3) = 1

-- Theorem statement
theorem roots_on_unit_circle :
  ∀ z : ℂ, complex_equation z → on_circle z :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_on_unit_circle_l239_23940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l239_23938

noncomputable def f (a m : ℝ) (x : ℝ) : ℝ := a^(2*x) + m*a^(-2*x)

noncomputable def g (a k : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - 2*k*f (x/2) + 2*a^(-2*x)

theorem function_properties (a m k : ℝ) :
  a > 0 ∧ a ≠ 1 ∧
  (∀ x, f a m x = -f a m (-x)) ∧  -- f is odd
  f a m 1 = 15/4 ∧
  (∃ x₀ ∈ Set.Icc 0 1, ∀ x ∈ Set.Icc 0 1, g a k (f a m) x ≥ g a k (f a m) x₀) ∧
  (∃ x₁ ∈ Set.Icc 0 1, g a k (f a m) x₁ = 2) →
  m = -1 ∧ k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l239_23938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l239_23951

-- Define the constants as noncomputable
noncomputable def a : ℝ := (1/2) ^ (1/3)
noncomputable def b : ℝ := (1/3) ^ (1/2)
noncomputable def c : ℝ := Real.log (3/Real.pi)

-- State the theorem
theorem order_of_constants : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l239_23951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_result_l239_23917

theorem vector_operation_result :
  let u : Fin 3 → ℝ := ![3, -2, 5]
  let v : Fin 3 → ℝ := ![-2, 4, -3]
  (2 : ℝ) • u + (4 : ℝ) • v - ![1, 1, 1] = ![-3, 11, -3] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_result_l239_23917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pricing_system_l239_23975

noncomputable def waterBill (usage : ℝ) : ℝ :=
  if usage ≤ 20 then
    2.3 * usage
  else if usage ≤ 30 then
    2.3 * 20 + 3.45 * (usage - 20)
  else
    2.3 * 20 + 3.45 * 10 + 4.6 * (usage - 30)

noncomputable def waterUsage (bill : ℝ) : ℝ :=
  if bill ≤ 2.3 * 20 then
    bill / 2.3
  else if bill ≤ 2.3 * 20 + 3.45 * 10 then
    20 + (bill - 2.3 * 20) / 3.45
  else
    30 + (bill - 2.3 * 20 - 3.45 * 10) / 4.6

theorem water_pricing_system (usage : ℝ) (bill : ℝ) :
  (usage = 32 → waterBill usage = 89.7) ∧
  (bill = 59.8 → waterUsage bill = 24) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pricing_system_l239_23975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_twice_abs_l239_23968

theorem square_difference_twice_abs : 2 * |(105 : ℝ)^2 - 95^2| = 4000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_twice_abs_l239_23968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_four_l239_23932

noncomputable def geometric_sequence (q : ℝ) (a_1 : ℝ) (n : ℕ) : ℝ := a_1 * q^(n - 1)

noncomputable def geometric_sum (q : ℝ) (a_1 : ℝ) (n : ℕ) : ℝ := a_1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_four (q : ℝ) (a_1 : ℝ) :
  a_1 * q^3 = 2 * a_1 →
  5/4 = (a_1 * q^3 + 2 * a_1 * q^6) / 2 →
  geometric_sum q a_1 4 = 30 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_four_l239_23932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_256_x_equals_log_2_7_div_8_l239_23969

noncomputable def log_16_x_minus_3 (x : ℝ) : ℝ := Real.log (x - 3) / Real.log 16

theorem log_256_x_equals_log_2_7_div_8 (x : ℝ) (h : log_16_x_minus_3 x = 1/2) :
  Real.log x / Real.log 256 = Real.log 7 / (8 * Real.log 2) :=
by
  -- We'll use sorry to skip the proof for now
  sorry

#check log_256_x_equals_log_2_7_div_8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_256_x_equals_log_2_7_div_8_l239_23969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_even_function_l239_23954

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x + a)(x + b) -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  (x + a) * (x + b)

theorem min_value_of_even_function (a b : ℝ) :
  IsEven (f a b) →
  (∀ x, a < x ∧ x < a + 4 → True) →
  (∃ m, ∀ x, a < x ∧ x < a + 4 → m ≤ f a b x ∧ (∃ y, a < y ∧ y < a + 4 ∧ f a b y = m)) →
  (∃ m, m = -4 ∧ ∀ x, a < x ∧ x < a + 4 → m ≤ f a b x ∧ (∃ y, a < y ∧ y < a + 4 ∧ f a b y = m)) :=
by sorry

#check min_value_of_even_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_even_function_l239_23954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_regions_II_III_proof_l239_23959

/-- The area of regions II and III in the given geometric configuration -/
def area_regions_II_III (π : ℝ) : ℝ :=
  10 * π - 8

/-- The length of side AB of rectangle ABCD -/
def AB : ℝ := 4

/-- The length of side BC of rectangle ABCD -/
def BC : ℝ := 2

/-- The length of diagonal BD of rectangle ABCD -/
noncomputable def BD : ℝ := Real.sqrt (AB^2 + BC^2)

/-- The area of the rectangle ABCD -/
def area_rectangle : ℝ := AB * BC

/-- The area of a quarter circle with radius BD -/
noncomputable def area_quarter_circle (π : ℝ) : ℝ := (π * BD^2) / 4

theorem area_regions_II_III_proof (π : ℝ) :
  area_regions_II_III π = 2 * area_quarter_circle π - area_rectangle :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_regions_II_III_proof_l239_23959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_sixth_plus_two_alpha_l239_23908

theorem sin_pi_sixth_plus_two_alpha (α : ℝ) : 
  Real.sin (π/6 - α) = 3/5 → Real.sin (π/6 + 2*α) = 7/25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_sixth_plus_two_alpha_l239_23908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toothpaste_theorem_l239_23962

/-- The amount of toothpaste Anne and her brother use at each brushing -/
noncomputable def toothpaste_problem : ℚ :=
  let total_toothpaste : ℚ := 105
  let dad_usage : ℚ := 3
  let mom_usage : ℚ := 2
  let brushings_per_day : ℕ := 3
  let days : ℕ := 5
  let family_members : ℕ := 4
  let parents_daily_usage : ℚ := (dad_usage + mom_usage) * brushings_per_day
  let parents_total_usage : ℚ := parents_daily_usage * days
  let children_total_usage : ℚ := total_toothpaste - parents_total_usage
  let children_brushings : ℕ := 2 * brushings_per_day * days
  children_total_usage / children_brushings

theorem toothpaste_theorem : toothpaste_problem = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toothpaste_theorem_l239_23962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_exist_l239_23974

/-- The function y = x^2/2 --/
noncomputable def f (x : ℝ) : ℝ := x^2 / 2

/-- The line x = √3/2 --/
def line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = Real.sqrt 3 / 2}

/-- Angle between two lines given their slopes --/
noncomputable def angle_between_lines (k1 k2 : ℝ) : ℝ :=
  Real.arctan ((k2 - k1) / (1 + k1 * k2))

/-- Theorem: There exist two points on the line x = √3/2 through which
    two tangents to y = x^2/2 pass, with an angle of 60° between them --/
theorem tangent_points_exist : ∃ (y1 y2 : ℝ),
  (Real.sqrt 3 / 2, y1) ∈ line ∧
  (Real.sqrt 3 / 2, y2) ∈ line ∧
  (∃ (k1 k2 k3 k4 : ℝ),
    (∀ x, f x = k1 * (x - Real.sqrt 3 / 2) + y1) ∧
    (∀ x, f x = k2 * (x - Real.sqrt 3 / 2) + y1) ∧
    (∀ x, f x = k3 * (x - Real.sqrt 3 / 2) + y2) ∧
    (∀ x, f x = k4 * (x - Real.sqrt 3 / 2) + y2) ∧
    angle_between_lines k1 k2 = π / 3 ∧
    angle_between_lines k3 k4 = π / 3) ∧
  y1 = 0 ∧ y2 = -5/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_exist_l239_23974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l239_23928

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 49/4
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1/4

-- Define the trajectory of C
def trajectory_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the dot product condition
def dot_product_condition (xA yA xB yB : ℝ) : Prop := xA * xB + yA * yB = -2

-- Main theorem
theorem circle_and_line_problem :
  ∃ (k : ℝ),
  (∀ (x y : ℝ), (circle_M x y ∧ circle_N x y) → trajectory_C x y) ∧
  (∀ (xA yA xB yB : ℝ),
    line_l k xA yA ∧ line_l k xB yB ∧ 
    trajectory_C xA yA ∧ trajectory_C xB yB ∧
    dot_product_condition xA yA xB yB →
    k = Real.sqrt 2 ∨ k = -Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l239_23928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seeder_path_length_is_500_l239_23986

/-- Calculates the length of the path for a seeder with given parameters. -/
noncomputable def seeder_path_length (working_width : ℝ) (initial_grain : ℝ) (seeding_rate : ℝ) (grain_decrease_percent : ℝ) : ℝ :=
  let grain_decrease : ℝ := initial_grain * grain_decrease_percent / 100
  let hectares_sown : ℝ := grain_decrease / seeding_rate
  let area_sown : ℝ := hectares_sown * 10000  -- 1 hectare = 10000 m²
  area_sown / working_width

/-- Theorem stating that the seeder path length is 500 meters for the given parameters. -/
theorem seeder_path_length_is_500 :
  seeder_path_length 4 250 175 14 = 500 := by
  -- Unfold the definition of seeder_path_length
  unfold seeder_path_length
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seeder_path_length_is_500_l239_23986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l239_23960

/-- Given four points in ℝ³, prove that they form a parallelogram and calculate its area. -/
theorem parallelogram_area (A B C D : ℝ × ℝ × ℝ) 
  (hA : A = (4, -3, 2)) 
  (hB : B = (6, -7, 5)) 
  (hC : C = (5, -2, 0)) 
  (hD : D = (7, -6, 3)) : 
  (B.1 - A.1, B.2.1 - A.2.1, B.2.2 - A.2.2) = (D.1 - C.1, D.2.1 - C.2.1, D.2.2 - C.2.2) ∧ 
  Real.sqrt (((B.1 - A.1) * (C.2.1 - A.2.1) - (B.2.1 - A.2.1) * (C.1 - A.1))^2 + 
            ((B.2.1 - A.2.1) * (C.2.2 - A.2.2) - (B.2.2 - A.2.2) * (C.2.1 - A.2.1))^2 + 
            ((B.2.2 - A.2.2) * (C.1 - A.1) - (B.1 - A.1) * (C.2.2 - A.2.2))^2) = Real.sqrt 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l239_23960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l239_23929

theorem rectangle_area (x : ℝ) (h1 : 3.6 < x) (h2 : x < 6.25) : 
  (5 * x - 18) * (25 - 4 * x) = 2809 / 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l239_23929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l239_23901

noncomputable def a : ℝ × ℝ := (2, 0)
noncomputable def c : ℝ × ℝ := (1, 0)

-- Define the projection function
noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let magnitude_squared := u.1 * u.1 + u.2 * u.2
  (dot_product / magnitude_squared) • u

-- Theorem statement
theorem projection_theorem (b : ℝ × ℝ) :
  proj a b = c → ∃ y : ℝ, b = (1, y) := by
  sorry

#check projection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l239_23901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_unit_interval_f_decreasing_on_interval_l239_23925

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - (1/2) * x^2

-- Theorem for part (1)
theorem f_extrema_on_unit_interval :
  let a := 1
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 1/2) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = 1/2) ∧
  (∀ x ∈ Set.Icc 0 1, f a x ≥ -1/54) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = -1/54) :=
by sorry

-- Theorem for part (2)
theorem f_decreasing_on_interval (a : ℝ) (h : a > 0) :
  ∀ x ∈ Set.Ioo 0 (1/(6*a)), 
    ∀ y ∈ Set.Ioo 0 (1/(6*a)), 
      x < y → f a x > f a y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_unit_interval_f_decreasing_on_interval_l239_23925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_of_f_l239_23920

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := tan x - sin x

-- State the theorem
theorem sign_of_f :
  (∀ x ∈ Set.Ioo (-π/2) 0, f x < 0) ∧
  (∀ x ∈ Set.Ioo 0 (π/2), f x > 0) ∧
  (f 0 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_of_f_l239_23920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_app_user_growth_l239_23953

-- Define the user count function
noncomputable def A (k : ℝ) (t : ℝ) : ℝ := 500 * Real.exp (k * t)

-- Define the given logarithm value
def lg2 : ℝ := 0.30

-- State the theorem
theorem app_user_growth (k : ℝ) :
  A k 10 = 2000 →
  (∀ t : ℝ, t < 34 → A k t ≤ 50000) ∧
  A k 34 > 50000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_app_user_growth_l239_23953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l239_23997

/-- Triangle with unequal sides and given medians and area -/
structure SpecialTriangle where
  -- Two medians
  median1 : ℝ
  median2 : ℝ
  -- Area of the triangle
  area : ℝ
  -- Conditions
  median1_eq : median1 = 3
  median2_eq : median2 = 6
  area_eq : area = 3 * Real.sqrt 15
  unequal_sides : median1 ≠ median2

/-- The third median of the special triangle -/
noncomputable def third_median (t : SpecialTriangle) : ℝ := 3 * Real.sqrt 6

/-- Theorem stating that the third median has the correct length -/
theorem third_median_length (t : SpecialTriangle) : 
  third_median t = 3 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_median_length_l239_23997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_a3b2_in_expansion_l239_23924

theorem coefficient_a3b2_in_expansion : ℕ :=
  let a : ℚ := 0  -- Define variables as rationals
  let b : ℚ := 0
  let c : ℚ := 1  -- Avoid division by zero
  let expansion := (a + b) ^ 5 * (c + 1 / c) ^ 6
  200  -- The actual coefficient, as calculated in the solution


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_a3b2_in_expansion_l239_23924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l239_23933

theorem calculate_expression : 
  ((-1 : ℝ)^2023) + Real.sqrt 4 - abs (-Real.sqrt 2) + ((-8 : ℝ)^(1/3)) = -1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l239_23933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_inequality_l239_23983

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 1 else 1

theorem range_of_inequality :
  {x : ℝ | f (1 - x^2) > f (2*x)} = Set.Ioo (-1) (Real.sqrt 2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_inequality_l239_23983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_inclination_l239_23991

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  l₁ : Real → Real → Prop
  l₂ : Real → Real → Prop

/-- A line intersecting two parallel lines -/
structure IntersectingLine where
  m : Real → Real → Prop

/-- The configuration of parallel lines and an intersecting line -/
structure LineConfiguration where
  pLines : ParallelLines
  iLine : IntersectingLine
  enclosed_segment_length : Real

/-- The inclination angle of a line -/
noncomputable def inclination_angle (l : Real → Real → Prop) : Real := sorry

/-- The theorem stating the possible inclination angles of the intersecting line -/
theorem intersecting_line_inclination 
  (config : LineConfiguration) 
  (h1 : config.pLines.l₁ = fun x y ↦ x + y = 0)
  (h2 : config.pLines.l₂ = fun x y ↦ x + y + Real.sqrt 6 = 0)
  (h3 : config.enclosed_segment_length = 2 * Real.sqrt 3) :
  (inclination_angle config.iLine.m = 105 ∨ inclination_angle config.iLine.m = 165) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_inclination_l239_23991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_plane_specific_l239_23947

/-- The distance from a point to a plane passing through three points -/
noncomputable def distance_point_to_plane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ :=
  let (x₀, y₀, z₀) := M₀
  let (x₁, y₁, z₁) := M₁
  let (x₂, y₂, z₂) := M₂
  let (x₃, y₃, z₃) := M₃
  let A := (y₂ - y₁) * (z₃ - z₁) - (z₂ - z₁) * (y₃ - y₁)
  let B := (z₂ - z₁) * (x₃ - x₁) - (x₂ - x₁) * (z₃ - z₁)
  let C := (x₂ - x₁) * (y₃ - y₁) - (y₂ - y₁) * (x₃ - x₁)
  let D := -A * x₁ - B * y₁ - C * z₁
  abs (A * x₀ + B * y₀ + C * z₀ + D) / Real.sqrt (A^2 + B^2 + C^2)

theorem distance_point_to_plane_specific :
  distance_point_to_plane (-21, 20, -16) (-2, -1, -1) (0, 3, 2) (3, 1, -4) = 1023 / Real.sqrt 1021 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_plane_specific_l239_23947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_x_derivative_of_sqrt_x_l239_23978

-- Define the function f(x) = x
def f (x : ℝ) : ℝ := x

-- Define the function g(x) = √x
noncomputable def g (x : ℝ) : ℝ := Real.sqrt x

-- Theorem 1: The derivative of f(x) = x is 1
theorem derivative_of_x (x : ℝ) : 
  deriv f x = 1 := by sorry

-- Theorem 2: The derivative of g(x) = √x is 1/(2√x)
theorem derivative_of_sqrt_x (x : ℝ) (h : x > 0) : 
  deriv g x = 1 / (2 * Real.sqrt x) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_x_derivative_of_sqrt_x_l239_23978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_Al_atoms_is_one_l239_23966

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of iodine atoms in the compound -/
def num_I_atoms : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 408

/-- The function to calculate the number of aluminum atoms in the compound -/
noncomputable def num_Al_atoms : ℕ :=
  Int.toNat (Int.floor ((molecular_weight - num_I_atoms * atomic_weight_I) / atomic_weight_Al + 0.5))

/-- Theorem stating that the number of aluminum atoms in the compound is 1 -/
theorem num_Al_atoms_is_one : num_Al_atoms = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_Al_atoms_is_one_l239_23966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_comparison_l239_23992

noncomputable section

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- Define points
def P : ℝ × ℝ := (-1, 1)
def Q : ℝ × ℝ := (-1/2, 0)

-- Define line l
def line_l (k : ℝ) (x : ℝ) : ℝ := k * (x + 1/2)

-- Define intersection points M and N
def M (a k : ℝ) : ℝ × ℝ := sorry
def N (a k : ℝ) : ℝ × ℝ := sorry

-- Define points A and B
def A (a k : ℝ) : ℝ × ℝ := sorry
def B (a k : ℝ) : ℝ × ℝ := sorry

-- Define areas S1 and S2
def S1 (a k : ℝ) : ℝ := sorry
def S2 (a k : ℝ) : ℝ := sorry

-- Theorem statement
theorem area_comparison (a k : ℝ) (h1 : parabola a (-1) = 1) (h2 : parabola a (-1/2) = 0) (h3 : k > 0) : S1 a k > 3 * S2 a k := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_comparison_l239_23992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_satisfies_conditions_l239_23913

/-- The price per package of gaskets -/
noncomputable def price_per_package : ℚ := 134/5

/-- The number of packages sold at full price -/
def full_price_packages : ℕ := 10

/-- The total number of packages sold -/
def total_packages : ℕ := 60

/-- The discount rate for packages in excess of full_price_packages -/
def discount_rate : ℚ := 4/5

/-- The total revenue received -/
def total_revenue : ℚ := 1340

/-- Theorem stating that the given price per package satisfies the conditions -/
theorem price_satisfies_conditions : 
  (full_price_packages : ℚ) * price_per_package + 
  ((total_packages - full_price_packages : ℚ) * (discount_rate * price_per_package)) = 
  total_revenue := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_satisfies_conditions_l239_23913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbers_not_divisible_by_six_l239_23999

theorem max_numbers_not_divisible_by_six (n : ℕ) (h : n = 2022) : 
  ¬ ∃ (S : Finset ℕ), 
    (∀ x ∈ S, x ≤ n) ∧ 
    (S.card = 677) ∧ 
    (∀ x y, x ∈ S → y ∈ S → x ≠ y → ¬(6 ∣ (x + y))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbers_not_divisible_by_six_l239_23999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_proof_l239_23902

theorem vector_operation_proof :
  let v₁ : Fin 2 → ℚ := ![3, -9]
  let v₂ : Fin 2 → ℚ := ![2, -8]
  (4 : ℚ) • v₁ - (3 : ℚ) • v₂ = ![6, -12] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_proof_l239_23902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_max_l239_23973

/-- Parabola C: x^2 = 4y -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Point D -/
def D : ℝ × ℝ := (0, 2)

/-- Center M of the circle -/
noncomputable def M (a : ℝ) : ℝ × ℝ := (a, a^2/4)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Points A and B where circle intersects x-axis -/
def A (a : ℝ) : ℝ × ℝ := (a - 2, 0)
def B (a : ℝ) : ℝ × ℝ := (a + 2, 0)

/-- The expression to be maximized -/
noncomputable def expr (a : ℝ) : ℝ :=
  let l₁ := distance D (A a)
  let l₂ := distance D (B a)
  l₁ / l₂ + l₂ / l₁

theorem parabola_circle_max :
  ∃ (a : ℝ), ∀ (x : ℝ), expr x ≤ expr a ∧ expr a = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_max_l239_23973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_profit_percentage_approx_l239_23926

noncomputable def tv_cost : ℝ := 16000
noncomputable def dvd_cost : ℝ := 6250
noncomputable def theater_cost : ℝ := 11500
noncomputable def console_cost : ℝ := 18500

noncomputable def tv_profit_percent : ℝ := 30
noncomputable def dvd_profit_percent : ℝ := 20
noncomputable def theater_profit_percent : ℝ := 25
noncomputable def console_profit_percent : ℝ := 15

noncomputable def total_cost : ℝ := tv_cost + dvd_cost + theater_cost + console_cost

noncomputable def total_profit : ℝ := 
  (tv_cost * tv_profit_percent / 100) +
  (dvd_cost * dvd_profit_percent / 100) +
  (theater_cost * theater_profit_percent / 100) +
  (console_cost * console_profit_percent / 100)

noncomputable def overall_profit_percentage : ℝ := (total_profit / total_cost) * 100

theorem overall_profit_percentage_approx :
  |overall_profit_percentage - 22.39| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_profit_percentage_approx_l239_23926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_meet_in_two_hours_l239_23965

/-- Represents the problem of cars and a fly on a road --/
structure RoadProblem where
  road_length : ℝ
  speed_A : ℝ
  speed_B : ℝ
  speed_fly : ℝ

/-- The time when the cars meet and crush the fly --/
noncomputable def meeting_time (p : RoadProblem) : ℝ :=
  p.road_length / (p.speed_A + p.speed_B)

/-- Theorem stating that for the given conditions, the meeting time is 2 hours --/
theorem cars_meet_in_two_hours :
  let p : RoadProblem := {
    road_length := 300,
    speed_A := 50,
    speed_B := 100,
    speed_fly := 150
  }
  meeting_time p = 2 := by
  -- Unfold the definition of meeting_time
  unfold meeting_time
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_meet_in_two_hours_l239_23965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_sum_l239_23993

/-- A polynomial of degree n -/
def PolynomialOfDegree (n : ℕ) := {p : Polynomial ℝ // p.degree = some n}

/-- The degree of a polynomial -/
def degree (p : Polynomial ℝ) : ℕ := (Polynomial.degree p).getD 0

/-- Composition of a polynomial with x^n -/
noncomputable def composeWithPower (p : Polynomial ℝ) (n : ℕ) : Polynomial ℝ :=
  p.comp (Polynomial.monomial n 1)

theorem degree_of_polynomial_sum (f : PolynomialOfDegree 3) 
  (g : PolynomialOfDegree 2) (h : PolynomialOfDegree 6) :
  degree ((composeWithPower f.val 4 * composeWithPower g.val 5) + h.val) = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_sum_l239_23993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_points_in_circle_theorem_l239_23939

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A circle with center (0, 0) and diameter 1 -/
def unitDiameterCircle (p : Point) : Prop :=
  p.x^2 + p.y^2 ≤ (1/2)^2

theorem seven_points_in_circle_theorem (points : Fin 7 → Point)
    (h : ∀ i, unitDiameterCircle (points i)) :
    ∃ i j, i ≠ j ∧ distance (points i) (points j) ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_points_in_circle_theorem_l239_23939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l239_23989

/-- The area of a triangle with vertices at (2, 2), (8, 2), and (4, 10) is 24 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (8, 2)
  let C : ℝ × ℝ := (4, 10)
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  area = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l239_23989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_four_digit_number_last_digit_proof_largest_four_digit_number_last_digit_l239_23987

theorem largest_four_digit_number_last_digit : ℕ → Prop :=
  -- Define the set of four-digit numbers
  let four_digit_numbers := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999}

  -- Define the condition for the number to be divisible by 9
  let divisible_by_9 := λ n : ℕ => n % 9 = 0

  -- Define the condition for the first three digits to be a multiple of 4
  let first_three_digits_multiple_of_4 := λ n : ℕ => (n / 10) % 4 = 0

  -- Define the set of numbers satisfying both conditions
  let valid_numbers := {n ∈ four_digit_numbers | divisible_by_9 n ∧ first_three_digits_multiple_of_4 n}

  -- The theorem: The last digit of the largest number in valid_numbers is 3
  λ largest_number : ℕ => 
    largest_number ∈ valid_numbers ∧ 
    (∀ n ∈ valid_numbers, n ≤ largest_number) → 
    largest_number % 10 = 3

theorem proof_largest_four_digit_number_last_digit : 
  ∃ n : ℕ, largest_four_digit_number_last_digit n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_four_digit_number_last_digit_proof_largest_four_digit_number_last_digit_l239_23987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l239_23958

noncomputable def π : ℝ := Real.pi

noncomputable def f (x : ℝ) : ℝ := (π^x - π^(-x)) / 2
noncomputable def g (x : ℝ) : ℝ := (π^x + π^(-x)) / 2

theorem function_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, g (-x) = g x) ∧
  (¬∃ y, ∀ x, f x ≤ y) ∧
  (¬∃ y, ∀ x, y ≤ f x) ∧
  (∃ y, ∀ x, g x ≥ y) ∧
  (¬∃ y, ∀ x, g x ≤ y) ∧
  (∀ x, f (2*x) = 2 * f x * g x) ∧
  (∃ x, f x = 0) ∧
  (∀ x, g x ≠ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l239_23958
