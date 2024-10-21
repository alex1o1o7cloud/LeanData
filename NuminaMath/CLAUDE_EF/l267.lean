import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_expression_equals_eight_l267_26794

open Real

-- Define the logarithm expression
noncomputable def logarithm_expression : ℝ :=
  (log 32 / log 3) * (log 9 / log 4) - log (3/4) / log 2 + log 6 / log 2

-- Theorem statement
theorem logarithm_expression_equals_eight :
  logarithm_expression = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_expression_equals_eight_l267_26794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_ge_e_l267_26753

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log a + Real.log x) / x

-- State the theorem
theorem decreasing_f_implies_a_ge_e (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ → f a x₂ ≤ f a x₁) →
  a ≥ Real.exp 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_ge_e_l267_26753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l267_26757

noncomputable def cube_root (x : ℝ) : ℝ := Real.rpow x (1/3)

theorem equation_solutions :
  let S := {x : ℝ | cube_root (18*x - 2) + cube_root (16*x + 2) = 6 * cube_root x}
  S = {0, 1/Real.sqrt 501619, -1/Real.sqrt 501619} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l267_26757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jonys_speed_is_100_l267_26777

/-- Represents a walking route with start, turn, and end blocks -/
structure WalkingRoute where
  start : ℕ
  turn : ℕ
  stop : ℕ

/-- Calculates the total distance of a walking route in blocks -/
def totalDistanceInBlocks (route : WalkingRoute) : ℕ :=
  (route.turn - route.start) + (route.turn - route.stop)

/-- Represents Jony's walk -/
def jonysWalk : WalkingRoute := {
  start := 10,
  turn := 90,
  stop := 70
}

/-- The length of each block in meters -/
def blockLength : ℕ := 40

/-- The total time Jony spent walking in minutes -/
def totalWalkingTime : ℕ := 40

/-- Theorem: Jony's walking speed is 100 meters per minute -/
theorem jonys_speed_is_100 : 
  (totalDistanceInBlocks jonysWalk * blockLength) / totalWalkingTime = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jonys_speed_is_100_l267_26777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_cos_2x_l267_26787

open Real

-- Define the function f(x) = cos(2x)
noncomputable def f (x : ℝ) := cos (2 * x)

-- State the theorem
theorem derivative_of_cos_2x :
  ∀ x : ℝ, deriv f x = -2 * sin (2 * x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_cos_2x_l267_26787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_t_l267_26786

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the conditions
def is_valid_rectangle (rect : Rectangle) : Prop :=
  rect.A.1 = rect.D.1 ∧ rect.B.1 = rect.C.1 ∧
  rect.A.2 = rect.B.2 ∧ rect.D.2 = rect.C.2 ∧
  rect.B.1 - rect.A.1 = 1 ∧ rect.C.2 - rect.B.2 = 2

-- Define a point M inside the rectangle
def is_inside (M : ℝ × ℝ) (rect : Rectangle) : Prop :=
  rect.A.1 ≤ M.1 ∧ M.1 ≤ rect.B.1 ∧
  rect.A.2 ≤ M.2 ∧ M.2 ≤ rect.C.2

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the function t
noncomputable def t (M : ℝ × ℝ) (rect : Rectangle) : ℝ :=
  distance M rect.A * distance M rect.C +
  distance M rect.B * distance M rect.D

-- The theorem to prove
theorem min_value_of_t (rect : Rectangle) (M : ℝ × ℝ) :
  is_valid_rectangle rect → is_inside M rect → t M rect ≥ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_t_l267_26786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_list_mean_mode_l267_26750

theorem integer_list_mean_mode (x : ℕ) : 
  x ≤ 100 →
  x > 0 →
  let list := [31, 58, 98, x, x]
  let sum := 31 + 58 + 98 + x + x
  5 * (3 * x) = 2 * sum →
  x = 34 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_list_mean_mode_l267_26750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_rounds_to_52_37_l267_26720

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem not_rounds_to_52_37 : round_to_hundredth 52.375 ≠ 52.37 := by
  -- Convert the real numbers to rationals for exact comparison
  have h1 : (52375 : ℚ) / 1000 ≠ 5237 / 100 := by norm_num
  -- Use this fact to prove the theorem
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_rounds_to_52_37_l267_26720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l267_26730

/-- Fixed cost in million yuan -/
noncomputable def fixed_cost : ℝ := 20

/-- Variable cost function in million yuan -/
noncomputable def C (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10 * x^2 + 100 * x
  else if x ≥ 40 then 501 * x + 10000 / x - 4500
  else 0

/-- Selling price per hundred vehicles in million yuan -/
noncomputable def selling_price : ℝ := 5

/-- Profit function in million yuan -/
noncomputable def L (x : ℝ) : ℝ := selling_price * x - fixed_cost - C x

/-- Theorem stating the maximum profit and corresponding output -/
theorem max_profit :
  ∃ (x : ℝ), x = 100 ∧ L x = 2300 ∧ ∀ y, L y ≤ L x := by
  sorry

#check max_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l267_26730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_of_ln_is_exp_neg_l267_26743

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the rotation transformation
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)

-- Define the rotated function
noncomputable def g (x : ℝ) : ℝ := Real.exp (-x)

-- Theorem statement
theorem rotation_of_ln_is_exp_neg :
  ∀ x > 0, rotate90 (x, f x) = (-(f x), x) ∧ g (-(f x)) = x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_of_ln_is_exp_neg_l267_26743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_model2_regression_equation_and_prediction_l267_26766

-- Define the data types
structure DataSet where
  x_bar : ℝ
  y_bar : ℝ
  t_bar : ℝ
  sum_x_diff_sq : ℝ
  sum_t_diff_sq : ℝ
  sum_xy_diff : ℝ
  sum_ty_diff : ℝ

-- Define the models
noncomputable def Model1 (a b : ℝ) (x : ℝ) : ℝ := b * x + a
noncomputable def Model2 (c d : ℝ) (x : ℝ) : ℝ := d / x + c

-- Define the R-squared calculation
noncomputable def r_squared (y_pred y_actual : List ℝ) (y_mean : ℝ) : ℝ :=
  1 - (List.sum (List.map (λ (p : ℝ × ℝ) => (p.1 - p.2)^2) (List.zip y_pred y_actual))) /
      (List.sum (List.map (λ y => (y - y_mean)^2) y_actual))

-- Theorem statement
theorem model2_regression_equation_and_prediction 
  (data : DataSet) 
  (r_squared1 r_squared2 : ℝ) 
  (h_r_squared : r_squared2 > r_squared1) :
  ∃ (c d : ℝ), 
    (c = 2 ∧ d = 100) ∧ 
    (∀ x, Model2 c d x = 100 / x + 2) ∧
    (Model2 c d 25 = 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_model2_regression_equation_and_prediction_l267_26766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l267_26792

noncomputable def f (x : ℝ) : ℝ := (1/2)^x

theorem range_of_f :
  Set.range (fun x => f x) ∩ Set.Icc (f 1) (f (-3)) = Set.Icc (1/2) 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l267_26792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l267_26776

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 3 * x - 7 else 2 * (4 - x)

-- Theorem statement
theorem f_values :
  f (-4) = -19 ∧ f 5 = -2 := by
  -- Split the conjunction
  constructor
  -- Prove f(-4) = -19
  · simp [f]
    norm_num
  -- Prove f(5) = -2
  · simp [f]
    norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l267_26776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_correct_answer_count_l267_26764

/-- Represents a person's answers to the test questions -/
def Answer := Fin 15 → Bool

/-- The type of the test, containing 21 people's answers -/
def Test := Fin 21 → Answer

/-- Two people have at least one correct answer in common -/
def ShareCorrectAnswer (t : Test) (p1 p2 : Fin 21) : Prop :=
  ∃ q : Fin 15, t p1 q = true ∧ t p2 q = true

/-- Every two people have at least one correct answer in common -/
def EveryTwoShareCorrect (t : Test) : Prop :=
  ∀ p1 p2 : Fin 21, p1 ≠ p2 → ShareCorrectAnswer t p1 p2

/-- The number of people who correctly answered a given question -/
def CorrectCount (t : Test) (q : Fin 15) : Nat :=
  (Finset.filter (fun p => t p q = true) Finset.univ).card

/-- The maximum number of people who correctly answered any single question -/
def MaxCorrectCount (t : Test) : Nat :=
  Finset.sup Finset.univ (CorrectCount t)

theorem test_correct_answer_count
  (t : Test)
  (h : EveryTwoShareCorrect t) :
  5 ≤ MaxCorrectCount t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_correct_answer_count_l267_26764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l267_26774

/-- Definition of the line l: y = x - 8 -/
def line_l (x y : ℝ) : Prop := y = x - 8

/-- Definition of a direction vector -/
def is_direction_vector (l : ℝ → ℝ → Prop) (v : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), l (v.1 * t) (v.2 * t)

/-- Definition of the angle a line makes with the positive x-axis -/
noncomputable def line_angle (l : ℝ → ℝ → Prop) : ℝ :=
  Real.arctan (1 : ℝ)

theorem line_l_properties :
  is_direction_vector line_l (1, 1) ∧
  line_angle line_l = π / 4 := by
  sorry

#check line_l_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l267_26774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_used_l267_26734

/-- Represents a sampling method --/
inductive SamplingMethod
| Lottery
| Stratified
| RandomNumberTable
| Systematic

/-- Represents a class of students --/
structure StudentClass where
  size : Nat
  numbers : Finset Nat

/-- Represents a grade with multiple classes --/
structure Grade where
  classes : Finset StudentClass
  classCount : Nat

/-- Represents a sampling strategy --/
structure SamplingStrategy where
  grade : Grade
  selectedPosition : Nat

/-- Function to determine the sampling method based on the strategy --/
noncomputable def determineSamplingMethod (strategy : SamplingStrategy) : SamplingMethod :=
  sorry

theorem systematic_sampling_used (g : Grade) (s : SamplingStrategy) 
  (h1 : g.classCount = 12)
  (h2 : ∀ c ∈ g.classes, c.size = 50)
  (h3 : ∀ c ∈ g.classes, c.numbers = Finset.range 50)
  (h4 : s.selectedPosition = 40)
  (h5 : s.grade = g) :
  determineSamplingMethod s = SamplingMethod.Systematic := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_used_l267_26734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_zero_l267_26769

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 1 - Real.cos (x * Real.sin (1 / x))
  else 0

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_zero_l267_26769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_equal_half_l267_26759

noncomputable section

-- Define the expressions
def expr_A : ℝ → ℝ := λ x => (Real.tan x) / (1 - Real.tan x ^ 2)
def expr_B : ℝ → ℝ := λ x => (Real.tan x) * (Real.cos x ^ 2)
def expr_C : ℝ → ℝ := λ x => (Real.sqrt 3 / 3) * (Real.cos x ^ 2 - Real.sin x ^ 2)
def expr_D : ℝ → ℝ := λ x => (Real.tan x) / (1 - Real.tan x ^ 2)

-- Define the angles in radians
def angle_22_5 : ℝ := Real.pi / 8
def angle_15 : ℝ := Real.pi / 12
def angle_30 : ℝ := Real.pi / 6

-- State the theorem
theorem expressions_equal_half :
  expr_A angle_22_5 = 1/2 ∧ expr_C (Real.pi/12) = 1/2 ∧
  expr_B angle_15 ≠ 1/2 ∧ expr_D angle_30 ≠ 1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_equal_half_l267_26759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k4_existence_l267_26755

/-- A point in a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A line segment between two points --/
structure LineSegment where
  p1 : Point
  p2 : Point

/-- A graph represented by points and line segments --/
structure Graph where
  points : Finset Point
  edges : Finset LineSegment

/-- Three points are collinear if they lie on the same straight line --/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- A complete subgraph of order 4 (K₄) --/
def hasK4 (g : Graph) : Prop :=
  ∃ (p1 p2 p3 p4 : Point),
    p1 ∈ g.points ∧ p2 ∈ g.points ∧ p3 ∈ g.points ∧ p4 ∈ g.points ∧
    LineSegment.mk p1 p2 ∈ g.edges ∧ LineSegment.mk p1 p3 ∈ g.edges ∧ LineSegment.mk p1 p4 ∈ g.edges ∧
    LineSegment.mk p2 p3 ∈ g.edges ∧ LineSegment.mk p2 p4 ∈ g.edges ∧
    LineSegment.mk p3 p4 ∈ g.edges

theorem k4_existence
  (g : Graph)
  (h1 : g.points.card = 6)
  (h2 : g.edges.card = 13)
  (h3 : ∀ (p1 p2 p3 : Point), p1 ∈ g.points → p2 ∈ g.points → p3 ∈ g.points →
        p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬collinear p1 p2 p3) :
  hasK4 g :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k4_existence_l267_26755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_swaps_to_order_l267_26778

/-- Represents a collection of books on a shelf -/
structure BookShelf where
  books : Fin 100 → ℕ
  disordered : ∀ i j, i < j → books i > books j

/-- Represents a swap operation between two books -/
def swap (shelf : BookShelf) (i j : Fin 100) : BookShelf where
  books := fun k => if k = i then shelf.books j
                    else if k = j then shelf.books i
                    else shelf.books k
  disordered := by sorry

/-- Checks if two indices have different parity -/
def different_parity (i j : Fin 100) : Prop :=
  i.val % 2 ≠ j.val % 2

/-- Represents a sequence of valid swaps -/
def valid_swap_sequence (shelf : BookShelf) (swaps : List (Fin 100 × Fin 100)) : Prop :=
  ∀ p, p ∈ swaps → different_parity p.1 p.2

/-- Checks if a shelf is ordered -/
def is_ordered (shelf : BookShelf) : Prop :=
  ∀ i j, i < j → shelf.books i < shelf.books j

/-- The main theorem to be proved -/
theorem min_swaps_to_order (shelf : BookShelf) :
  ∃ (swaps : List (Fin 100 × Fin 100)),
    valid_swap_sequence shelf swaps ∧
    is_ordered (swaps.foldl (fun s p => swap s p.1 p.2) shelf) ∧
    swaps.length = 124 ∧
    (∀ (other_swaps : List (Fin 100 × Fin 100)),
      valid_swap_sequence shelf other_swaps →
      is_ordered (other_swaps.foldl (fun s p => swap s p.1 p.2) shelf) →
      other_swaps.length ≥ 124) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_swaps_to_order_l267_26778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_cos_sum_equals_two_l267_26797

theorem csc_cos_sum_equals_two : 
  1 / Real.sin (π / 18) + 4 * Real.cos (π / 9) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_cos_sum_equals_two_l267_26797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_xyz_equation_l267_26726

theorem unique_solution_xyz_equation :
  ∀ (x y z n : ℕ),
    n ≥ 2 →
    z ≤ 5 * 2^(2*n) →
    x^(2*n + 1) - y^(2*n + 1) = x*y*z + 2^(2*n + 1) →
    x = 3 ∧ y = 1 ∧ z = 70 ∧ n = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_xyz_equation_l267_26726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_between_50_and_300_l267_26789

theorem perfect_squares_between_50_and_300 : 
  (Finset.filter (fun n : ℕ => 50 < n * n ∧ n * n < 300) (Finset.range 18)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_between_50_and_300_l267_26789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_one_l267_26793

open Real

-- Define the curves in polar coordinates
noncomputable def C₁ (θ : ℝ) : ℝ := 4 * cos θ
def C₃ (θ : ℝ) : ℝ := 1

-- Define the angle of the ray
noncomputable def ray_angle : ℝ := π / 3

-- Define the intersection points
noncomputable def point_A : ℝ := C₁ ray_angle
noncomputable def point_B : ℝ := C₃ ray_angle

-- Theorem statement
theorem length_AB_is_one : abs (point_A - point_B) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_one_l267_26793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_KL_length_l267_26733

-- Define the square and points
noncomputable def Square (A B C D : ℝ × ℝ) : Prop :=
  A = (-1/2, 0) ∧ B = (-1/2, 1) ∧ C = (1/2, 1) ∧ D = (1/2, 0)

noncomputable def E : ℝ × ℝ := (-1/2, 1/3)
noncomputable def F : ℝ × ℝ := (-1/2, 2/3)
noncomputable def G : ℝ × ℝ := (-1/6, 1)
noncomputable def H : ℝ × ℝ := (1/6, 1)
noncomputable def I : ℝ × ℝ := (1/2, 1/2)
noncomputable def J : ℝ × ℝ := (0, 0)

-- Define the intersection points
noncomputable def K : ℝ × ℝ := (1/14, 3/7)
noncomputable def L : ℝ × ℝ := (-1/10, 3/5)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem square_KL_length (A B C D : ℝ × ℝ) :
  Square A B C D →
  distance K L = 6 * Real.sqrt 2 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_KL_length_l267_26733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_area_product_l267_26723

/-- 
Given a cube-shaped box with dimensions proportional to 3x, 4x, and 5x,
where x is a positive real number, this theorem proves that the product
of the bottom area (12x²), side area (15x²), and front area (20x²) is equal to 3600x⁶.
-/
theorem cube_area_product (x : ℝ) (h : x > 0) : 
  (12 * x^2) * (15 * x^2) * (20 * x^2) = 3600 * x^6 := by
  -- Expand the left-hand side
  calc (12 * x^2) * (15 * x^2) * (20 * x^2)
    = 12 * 15 * 20 * (x^2 * x^2 * x^2) := by ring
  -- Simplify the coefficient
  _ = 3600 * (x^2 * x^2 * x^2) := by norm_num
  -- Simplify the exponent
  _ = 3600 * x^6 := by ring

#check cube_area_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_area_product_l267_26723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_upper_bound_l267_26714

theorem lambda_upper_bound (l : ℝ) : 
  (∀ n : ℕ+, n.val^2 - n.val * (l + 1) + 7 ≥ l) → l ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_upper_bound_l267_26714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_relationships_l267_26706

-- Define the types for lines and planes
structure Line where

structure Plane where

-- Define the relations
axiom perpendicular : Line → Line → Prop
axiom parallel : Line → Line → Prop
axiom perpendicular_plane : Plane → Plane → Prop
axiom parallel_plane : Line → Plane → Prop
axiom intersect : Plane → Plane → Line → Prop
axiom contained_in : Line → Plane → Prop

-- Define the theorem
theorem spatial_relationships 
  (m n : Line) 
  (α β γ : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  (∃ (m n : Line) (α β : Plane), 
    perpendicular_plane α β ∧ 
    intersect α β m ∧ 
    perpendicular n m ∧ 
    ¬(perpendicular n m ∨ perpendicular n m)) ∧
  (∃ (m : Line) (α : Plane), 
    ¬perpendicular m m ∧ 
    ∃ (l : Set Line), (∀ l' ∈ l, perpendicular m l') ∧ Set.Infinite l) ∧
  (∀ (m n : Line) (α β : Plane), 
    intersect α β m → 
    parallel n m → 
    ¬contained_in n α → 
    ¬contained_in n β → 
    parallel_plane n α ∧ parallel_plane n β) ∧
  (∃ (m n : Line) (α β : Plane), 
    perpendicular_plane α β ∧ 
    parallel m n ∧ 
    perpendicular n m ∧ 
    ¬parallel_plane m α) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_relationships_l267_26706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_to_apple_value_l267_26744

-- Define the types for our goods
structure Commodity where
  value : ℕ

def Fish := Commodity
def Bread := Commodity
def Rice := Commodity
def Apple := Commodity

-- Define the exchange rates
def fish_to_bread (f : Fish) (b : Bread) : Prop := 3 * f.value = 2 * b.value
def bread_to_rice (b : Bread) (r : Rice) : Prop := b.value = 5 * r.value
def bread_to_apple (b : Bread) (a : Apple) : Prop := b.value = 3 * a.value

-- Theorem statement
theorem fish_to_apple_value : 
  ∀ (f : Fish) (b : Bread) (r : Rice) (a : Apple),
  fish_to_bread f b → bread_to_rice b r → bread_to_apple b a →
  f.value = 2 * a.value :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_to_apple_value_l267_26744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_ratio_l267_26796

open Matrix

def projection_matrix : Matrix (Fin 2) (Fin 2) ℚ := !![3/25, 4/25; 4/25, 12/25]

theorem projection_vector_ratio :
  ∀ (x y : ℚ), x ≠ 0 → y ≠ 0 →
  (projection_matrix.mulVec ![x, y] = ![x, y]) →
  (x / y = 2 / 11) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_ratio_l267_26796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l267_26710

noncomputable def f (x : ℝ) : ℝ := 3 * Real.tan (x / 2 + Real.pi / 3)

theorem min_positive_period_of_f :
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
  T = 2 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l267_26710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_144_l267_26749

/-- A function representing a single die roll --/
def dieRoll : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The probability of a specific outcome when rolling three dice --/
def threeDiceProb : ℚ := (1 : ℚ) / 216

/-- The set of all possible outcomes when rolling three dice --/
def threeDiceOutcomes : Finset (ℕ × ℕ × ℕ) :=
  Finset.product dieRoll (Finset.product dieRoll dieRoll)

/-- The favorable outcomes: triples whose product is 144 --/
def favorableOutcomes : Finset (ℕ × ℕ × ℕ) :=
  threeDiceOutcomes.filter (fun (a, b, c) => a * b * c = 144)

theorem probability_product_144 :
  (favorableOutcomes.card : ℚ) * threeDiceProb = 1 / 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_144_l267_26749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_set_size_l267_26708

def is_valid_set (S : Finset Nat) : Prop :=
  ∀ x ∈ S, Nat.Prime x ∧
  ∀ a b c, a ∈ S → b ∈ S → c ∈ S → Nat.Prime (a + b + c)

theorem max_prime_set_size :
  ∀ S : Finset Nat, is_valid_set S → S.card ≤ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_set_size_l267_26708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_right_trapezoid_l267_26721

/-- A right trapezoid is a trapezoid with one right angle. -/
def RightTrapezoid (A B C D : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let DC := (C.1 - D.1, C.2 - D.2)
  let AD := (D.1 - A.1, D.2 - A.2)
  (∃ (k : ℝ), AB = (k * DC.1, k * DC.2)) ∧  -- AB is parallel to DC
  (AB.1^2 + AB.2^2 = 4 * (DC.1^2 + DC.2^2)) ∧  -- AB = 2DC
  (AB.1 * AD.1 + AB.2 * AD.2 = 0)  -- AB is perpendicular to AD

/-- The quadrilateral ABCD with given vertices is a right trapezoid. -/
theorem quadrilateral_is_right_trapezoid :
  RightTrapezoid (1, 2) (3, 6) (0, 5) (-1, 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_right_trapezoid_l267_26721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_decreasing_bounded_l267_26761

noncomputable def f (x : ℝ) : ℝ := 2 - 1/x

noncomputable def sequence_a : ℝ → ℕ → ℝ
  | a₁, 0 => a₁
  | a₁, n + 1 => f (sequence_a a₁ n)

theorem sequence_a_decreasing_bounded (a₁ : ℝ) (h : 1 < a₁ ∧ a₁ < 2) :
  ∀ n : ℕ, 1 < sequence_a a₁ (n + 1) ∧ sequence_a a₁ (n + 1) < sequence_a a₁ n ∧ sequence_a a₁ n < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_decreasing_bounded_l267_26761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_even_function_l267_26756

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := Real.sin (2 * (x - m) + Real.pi / 3)

theorem smallest_m_for_even_function :
  ∀ m : ℝ, m > 0 →
  (∀ x : ℝ, g m x = g m (-x)) →
  m ≥ 5 * Real.pi / 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_even_function_l267_26756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_zero_z_is_purely_imaginary_z_equals_2_plus_5i_l267_26780

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := m * (m - 1) + (m^2 + 2*m - 3) * Complex.I

-- Theorem 1: z is zero iff m = 1
theorem z_is_zero (m : ℝ) : z m = 0 ↔ m = 1 := by sorry

-- Theorem 2: z is purely imaginary iff m = 0
theorem z_is_purely_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 0 := by sorry

-- Theorem 3: z = 2 + 5i iff m = 2
theorem z_equals_2_plus_5i (m : ℝ) : z m = 2 + 5*Complex.I ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_zero_z_is_purely_imaginary_z_equals_2_plus_5i_l267_26780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l267_26715

theorem arithmetic_sequence_sum (a₁ a₁₅ : ℂ) : 
  a₁^2 - 6*a₁ + 10 = 0 → 
  a₁₅^2 - 6*a₁₅ + 10 = 0 → 
  ∃ (d : ℂ), ∀ n, a₁ + (n-1)*d = a₁₅ + (15-n)*d →
  (17/2 : ℂ) * (a₁ + (a₁ + 16*d)) = 51 := by
  sorry

#check arithmetic_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l267_26715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_pill_cost_l267_26784

theorem green_pill_cost (total_days total_cost : ℕ) : ℕ :=
  let green_pill_cost := 20
  let pink_pill_cost := green_pill_cost - 1
  let daily_cost := green_pill_cost + pink_pill_cost
  have h1 : total_days = 14 := by sorry
  have h2 : total_cost = 546 := by sorry
  have h3 : total_cost = daily_cost * total_days := by sorry
  green_pill_cost

#check green_pill_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_pill_cost_l267_26784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_less_than_half_l267_26767

/-- The function f(x) = e^x - (1/2)x^2 + ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - (1/2) * x^2 + a * x

/-- Theorem: For a < 1 - e, the minimum value of f(x) on [1, +∞) is less than 1/2 -/
theorem f_min_less_than_half (a : ℝ) (h : a < 1 - Real.exp 1) :
  ∃ (x : ℝ), x ≥ 1 ∧ ∀ (y : ℝ), y ≥ 1 → f a x ≤ f a y ∧ f a x < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_less_than_half_l267_26767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_non_lucky_multiple_of_24_l267_26745

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ n % (sum_of_digits n) = 0

theorem least_non_lucky_multiple_of_24 :
  (∀ k : ℕ, k > 0 ∧ k < 120 ∧ 24 ∣ k → is_lucky k) ∧
  120 % 24 = 0 ∧
  ¬ is_lucky 120 :=
by
  sorry

#eval sum_of_digits 120  -- This should output 3
#eval 120 % 3  -- This should output 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_non_lucky_multiple_of_24_l267_26745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_smallest_positive_period_l267_26737

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + 2 * Real.cos (2 * x) + 3 * Real.tan (4 * x)

def D : Set ℝ := {x | ∀ k : ℤ, x ≠ (1 / 4 : ℝ) * k * Real.pi + Real.pi / 8}

def is_period (T : ℝ) : Prop := ∀ x ∈ D, f (x + T) = f x

def smallest_positive_period (T : ℝ) : Prop :=
  T > 0 ∧ is_period T ∧ ∀ S, 0 < S ∧ S < T → ¬ is_period S

theorem f_smallest_positive_period :
  smallest_positive_period Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_smallest_positive_period_l267_26737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_of_prism_with_inscribed_sphere_l267_26712

/-- The lateral edge of a regular triangular prism with base side 1 and an inscribed sphere -/
noncomputable def lateral_edge_of_prism : ℝ := Real.sqrt 3 / 3

/-- The base side of the prism -/
def base_side : ℝ := 1

/-- Theorem: The lateral edge of a regular triangular prism with base side 1 and an inscribed sphere is √3/3 -/
theorem lateral_edge_of_prism_with_inscribed_sphere :
  lateral_edge_of_prism = Real.sqrt 3 / 3 :=
by
  -- The proof is omitted for now
  sorry

#check lateral_edge_of_prism_with_inscribed_sphere

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_of_prism_with_inscribed_sphere_l267_26712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_2_neg1_polar_coordinates_l267_26782

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 && y ≥ 0 then Real.arctan (y / x) + Real.pi
           else if x < 0 && y < 0 then Real.arctan (y / x) - Real.pi
           else if x = 0 && y > 0 then Real.pi / 2
           else if x = 0 && y < 0 then -Real.pi / 2
           else 0  -- undefined for (0, 0)
  (r, if θ < 0 then θ + 2*Real.pi else θ)

theorem point_2_neg1_polar_coordinates :
  let (r, θ) := rectangular_to_polar 2 (-1)
  r = Real.sqrt 5 ∧ θ = 2*Real.pi - Real.arctan (1/2) ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_2_neg1_polar_coordinates_l267_26782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_weights_subset_sum_l267_26731

theorem seven_weights_subset_sum (weights : Finset ℕ) : 
  weights ⊆ Finset.range 27 →
  weights.card = 7 →
  ∃ (s t : Finset ℕ), s ⊆ weights ∧ t ⊆ weights ∧ s ≠ t ∧ s.Nonempty ∧ t.Nonempty ∧ 
    (s.sum id = t.sum id) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_weights_subset_sum_l267_26731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_planes_15_points_l267_26754

/-- The number of unique planes determined by n points in space, assuming no four points are coplanar -/
def uniquePlanes (n : ℕ) : ℕ := Nat.choose n 3

/-- No four points are coplanar -/
axiom no_four_coplanar (points : Finset (ℝ × ℝ × ℝ)) (h : points.card = 15) :
  ∀ p₁ p₂ p₃ p₄, p₁ ∈ points → p₂ ∈ points → p₃ ∈ points → p₄ ∈ points →
  p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ →
  ¬ Submodule.span ℝ {p₂ - p₁, p₃ - p₁, p₄ - p₁} = ⊤

theorem max_planes_15_points (points : Finset (ℝ × ℝ × ℝ)) (h : points.card = 15) :
  uniquePlanes 15 = 455 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_planes_15_points_l267_26754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_equals_4_has_three_solutions_l267_26740

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -3 then x^2 - 6 else x + 2

-- Theorem statement
theorem f_f_equals_4_has_three_solutions :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x : ℝ, x ∈ s ↔ f (f x) = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_equals_4_has_three_solutions_l267_26740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_circle_l267_26799

def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 - 1}

def CirclePQ (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X : ℝ × ℝ | (X.1 - (P.1 + Q.1)/2)^2 + (X.2 - (P.2 + Q.2)/2)^2 = ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) / 4}

theorem hyperbola_intersection_circle (k : ℝ) :
  let C := Hyperbola (Real.sqrt 2/2) 1
  let l := Line k
  ∃ P Q : ℝ × ℝ, P ∈ C ∩ l ∧ Q ∈ C ∩ l ∧ P ≠ Q ∧ (0, 0) ∈ CirclePQ P Q ↔ k = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_circle_l267_26799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_theorem_l267_26719

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi
  opposite_sides : a > 0 ∧ b > 0 ∧ c > 0

/-- The main theorem about the acute triangle ABC. -/
theorem acute_triangle_theorem (t : AcuteTriangle) 
  (h1 : Real.sqrt 3 * Real.sin t.C - Real.cos t.B = Real.cos (t.A - t.C))
  (h2 : t.a = 2 * Real.sqrt 3)
  (h3 : (1/2) * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 3) :
  t.A = Real.pi/3 ∧ t.b + t.c = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_theorem_l267_26719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_approx_l267_26724

/-- The atomic mass of Aluminum in g/mol -/
noncomputable def atomic_mass_Al : ℝ := 26.98

/-- The atomic mass of Oxygen in g/mol -/
noncomputable def atomic_mass_O : ℝ := 16.00

/-- The molar mass of Al2O3 in g/mol -/
noncomputable def molar_mass_Al2O3 : ℝ := 2 * atomic_mass_Al + 3 * atomic_mass_O

/-- The mass of Al in one mole of Al2O3 in g/mol -/
noncomputable def mass_Al_in_Al2O3 : ℝ := 2 * atomic_mass_Al

/-- The mass percentage of Al in Al2O3 -/
noncomputable def mass_percentage_Al : ℝ := (mass_Al_in_Al2O3 / molar_mass_Al2O3) * 100

theorem mass_percentage_Al_approx :
  abs (mass_percentage_Al - 52.91) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_approx_l267_26724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_is_12_l267_26705

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The polygon defined by the given vertices -/
def polygon : List Point := [
  ⟨0, 0⟩, ⟨2, 0⟩, ⟨2, 2⟩, ⟨4, 2⟩, ⟨4, 4⟩, ⟨2, 4⟩, ⟨0, 4⟩, ⟨0, 2⟩
]

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Calculate the area of a rectangle given two opposite corners -/
def rectangleArea (p1 p2 : Point) : ℝ :=
  abs ((p2.x - p1.x) * (p2.y - p1.y))

/-- Theorem: The area of the polygon is 12 square units -/
theorem polygon_area_is_12 : 
  triangleArea (polygon[0]) (polygon[1]) (polygon[7]) +
  rectangleArea (polygon[1]) (polygon[5]) +
  triangleArea (polygon[3]) (polygon[4]) (polygon[5]) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_is_12_l267_26705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_abs_plus_power_linear_function_through_points_l267_26746

-- Problem 1
theorem cube_root_plus_abs_plus_power : (8 : ℝ) ^ (1/3) + |(-5)| + (-1)^2023 = 6 := by sorry

-- Problem 2
theorem linear_function_through_points :
  ∀ k b : ℝ,
  (k * 0 + b = 1) →
  (k * 2 + b = 5) →
  ∀ x : ℝ, k * x + b = 2 * x + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_abs_plus_power_linear_function_through_points_l267_26746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_two_inequality_holds_iff_l267_26722

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + (a - 1) / x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x - g x

-- Theorem for part (1)
theorem min_value_when_a_is_two :
  ∃ (m : ℝ), ∀ (x : ℝ), x > 0 → h 2 x ≥ m ∧ ∃ (y : ℝ), y > 0 ∧ h 2 y = m ∧ m = 3 := by sorry

-- Theorem for part (2)
theorem inequality_holds_iff :
  ∀ (a : ℝ), a > 0 →
  (∀ (x : ℝ), x ≥ 1 → h a x ≥ 1) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_two_inequality_holds_iff_l267_26722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_withdrawal_approx_50_06_l267_26798

/-- Represents the bank account transactions over two months --/
structure BankAccount where
  initial_balance : ℚ
  first_deposit_rate : ℚ
  first_withdrawal_rate : ℚ
  second_deposit_rate : ℚ
  final_deposit_rate : ℚ
  final_balance_increase : ℚ

/-- Calculates the total withdrawal amount --/
def total_withdrawal (account : BankAccount) : ℚ :=
  let balance_after_first_deposit := account.initial_balance * (1 + account.first_deposit_rate)
  let balance_after_first_withdrawal := balance_after_first_deposit * (1 - account.first_withdrawal_rate)
  let balance_after_second_deposit := balance_after_first_withdrawal + account.initial_balance * account.second_deposit_rate
  let final_balance := account.initial_balance + account.final_balance_increase
  let balance_before_final_deposit := final_balance / (1 + account.final_deposit_rate)
  balance_after_second_deposit - balance_before_final_deposit

/-- Theorem stating that the total withdrawal is approximately 50.06 --/
theorem total_withdrawal_approx_50_06 (account : BankAccount)
  (h1 : account.initial_balance = 150)
  (h2 : account.first_deposit_rate = 1/10)
  (h3 : account.first_withdrawal_rate = 3/20)
  (h4 : account.second_deposit_rate = 1/4)
  (h5 : account.final_deposit_rate = 3/10)
  (h6 : account.final_balance_increase = 16) :
  ∃ ε > 0, |total_withdrawal account - 25003/500| < ε := by
  sorry

#eval total_withdrawal {
  initial_balance := 150,
  first_deposit_rate := 1/10,
  first_withdrawal_rate := 3/20,
  second_deposit_rate := 1/4,
  final_deposit_rate := 3/10,
  final_balance_increase := 16
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_withdrawal_approx_50_06_l267_26798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_proof_l267_26795

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  3 * x^2 - 12 * x + y^2 + 4 * y + 4 = 0

-- Define the area of the ellipse
noncomputable def ellipse_area : ℝ := 4 * Real.sqrt 3 * Real.pi

-- Theorem statement
theorem ellipse_area_proof :
  ∀ x y : ℝ, ellipse_equation x y → 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
  ((x - 2)^2 / (4 * a^2) + (y + 2)^2 / (4 * b^2) = 1) ∧
  (ellipse_area = Real.pi * a * b) := by
  sorry

#check ellipse_area_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_proof_l267_26795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_3_l267_26701

def sequence_a : ℕ → ℚ
  | 0 => -2
  | n + 1 => (1 + sequence_a n) / (1 - sequence_a n)

theorem a_2016_equals_3 : sequence_a 2015 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_3_l267_26701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_color_integer_coloring_l267_26717

/-- A coloring of integers using four colors. -/
def Coloring := ℤ → Fin 4

/-- Theorem statement for the four-color integer coloring problem. -/
theorem four_color_integer_coloring
  (f : Coloring) (x y : ℤ) 
  (h_odd_x : Odd x) (h_odd_y : Odd y) (h_distinct : |x| ≠ |y|) :
  ∃ (a b : ℤ), f a = f b ∧ (b - a ∈ ({x, y, x + y, x - y} : Set ℤ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_color_integer_coloring_l267_26717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_correct_l267_26728

/-- The equation of a circle in the form x^2 + y^2 + Dx + Ey + F = 0 -/
structure CircleEquation where
  D : ℚ
  E : ℚ
  F : ℚ

/-- The center of a circle -/
structure CircleCenter where
  x : ℚ
  y : ℚ

/-- Given a circle equation, compute its center -/
def computeCenter (eq : CircleEquation) : CircleCenter :=
  { x := -eq.D / 2, y := -eq.E / 2 }

theorem circle_center_correct (eq : CircleEquation) 
  (h : eq = { D := -10, E := 6, F := 25 }) : 
  computeCenter eq = { x := 5, y := -3 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_correct_l267_26728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_has_winning_strategy_l267_26725

/-- Represents a player in the game -/
inductive Player
| Petya
| Vasya

/-- Represents the game board -/
structure Board where
  rows : Nat
  cols : Nat

/-- Represents a strip placement -/
structure Strip where
  horizontal : Bool
  row : Nat
  col : Nat

/-- Represents the game state -/
structure GameState where
  board : Board
  currentPlayer : Player
  placedStrips : List Strip

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (strip : Strip) : Bool :=
  sorry

/-- Applies a move to the game state -/
def applyMove (state : GameState) (strip : Strip) : GameState :=
  sorry

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Determines the winner of the game -/
def getWinner (state : GameState) : Option Player :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Option Strip

/-- Theorem: Petya has a winning strategy -/
theorem petya_has_winning_strategy :
  ∃ (s : Strategy), ∀ (game : GameState),
    game.board.rows = 3 ∧ game.board.cols = 2021 ∧ game.currentPlayer = Player.Petya →
    ∀ (move : Strip), move ∈ s game →
    (getWinner (applyMove game move) = some Player.Petya) :=
  sorry

#check petya_has_winning_strategy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_has_winning_strategy_l267_26725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l267_26709

noncomputable def J : ℝ × ℝ := (-2, -3)
noncomputable def K : ℝ × ℝ := (-2, 1)
noncomputable def L : ℝ × ℝ := (6, 7)
noncomputable def M : ℝ × ℝ := (6, -3)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def perimeter : ℝ :=
  distance J K + distance K L + distance L M + distance M J

theorem trapezoid_perimeter :
  perimeter = 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l267_26709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_flow_maximum_traffic_flow_exceeds_ten_l267_26739

/-- The traffic flow function -/
noncomputable def traffic_flow (v : ℝ) : ℝ := 920 * v / (v^2 + 3*v + 1600)

theorem traffic_flow_maximum :
  ∃ (v_max : ℝ), v_max > 0 ∧
  (∀ (v : ℝ), v > 0 → traffic_flow v ≤ traffic_flow v_max) ∧
  v_max = 40 ∧ traffic_flow v_max = 920 / 83 := by
  sorry

theorem traffic_flow_exceeds_ten :
  ∀ (v : ℝ), 25 < v ∧ v < 64 ↔ traffic_flow v > 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_flow_maximum_traffic_flow_exceeds_ten_l267_26739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_eq_30_l267_26711

/-- An isosceles right triangle with legs of length 10 partitioned into 25 congruent triangles -/
structure PartitionedTriangle where
  leg_length : ℝ
  num_partitions : ℕ
  num_shaded : ℕ
  leg_length_eq : leg_length = 10
  num_partitions_eq : num_partitions = 25
  num_shaded_eq : num_shaded = 15

/-- The area of the shaded region in the partitioned triangle -/
noncomputable def shaded_area (t : PartitionedTriangle) : ℝ :=
  (t.leg_length ^ 2 / 2) * (t.num_shaded / t.num_partitions)

/-- Theorem stating that the shaded area is equal to 30 -/
theorem shaded_area_eq_30 (t : PartitionedTriangle) : shaded_area t = 30 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_eq_30_l267_26711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l267_26700

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(8*x + 4) * (4 : ℝ)^(4*x + 7) = (8 : ℝ)^(5*x + 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l267_26700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_graph_min_value_h_abs_not_equal_sqrt_squared_odd_not_always_through_origin_intermediate_value_not_always_true_l267_26716

-- Define the functions
def f (x : ℝ) := 3 * x^2
def g (x : ℝ) := 3 * (x - 1)^2
noncomputable def h (x : ℝ) := (2 : ℝ)^(|x|)

-- Theorem for proposition ③
theorem shift_graph : ∀ x : ℝ, g x = f (x - 1) := by sorry

-- Theorem for proposition ④
theorem min_value_h : ∀ x : ℝ, h x ≥ 1 ∧ ∃ y : ℝ, h y = 1 := by sorry

-- Theorem for falseness of proposition ①
theorem abs_not_equal_sqrt_squared : ∃ x : ℝ, |x| ≠ (Real.sqrt x)^2 := by sorry

-- Theorem for falseness of proposition ②
theorem odd_not_always_through_origin : 
  ∃ f : ℝ → ℝ, (∀ x : ℝ, f (-x) = -f x) ∧ f 0 ≠ 0 := by sorry

-- Theorem for falseness of proposition ⑤
theorem intermediate_value_not_always_true :
  ∃ f : ℝ → ℝ, f (-1) * f 3 < 0 ∧ ¬∃ x : ℝ, x ∈ Set.Icc (-1) 3 ∧ f x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_graph_min_value_h_abs_not_equal_sqrt_squared_odd_not_always_through_origin_intermediate_value_not_always_true_l267_26716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l267_26748

/-- An ellipse with semi-major axis a and semi-minor axis b. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0

/-- The eccentricity of an ellipse. -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2) / e.a

/-- The left vertex of the ellipse. -/
def Ellipse.leftVertex (e : Ellipse) : ℝ × ℝ := (-e.a, 0)

/-- The right vertex of the ellipse. -/
def Ellipse.rightVertex (e : Ellipse) : ℝ × ℝ := (e.a, 0)

/-- The top vertex of the ellipse. -/
def Ellipse.topVertex (e : Ellipse) : ℝ × ℝ := (0, e.b)

/-- The dot product of two 2D vectors. -/
def dotProduct (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- Vector from point p1 to point p2. -/
def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

/-- Theorem: If an ellipse with eccentricity 1/3 satisfies BA₁ · BA₂ = -1,
    then its equation is x²/9 + y²/8 = 1. -/
theorem ellipse_equation (e : Ellipse) 
    (h_ecc : e.eccentricity = 1/3)
    (h_dot : dotProduct 
      (vector e.topVertex e.leftVertex) 
      (vector e.topVertex e.rightVertex) = -1) :
    e.a^2 = 9 ∧ e.b^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l267_26748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l267_26758

-- Define the variables and conditions
variable (a b : ℝ)
variable (h1 : b > a)
variable (h2 : a > 1)
variable (h3 : 3 * (Real.log b / Real.log a) + 2 * (Real.log a / Real.log b) = 7)

-- Define the function to be minimized
noncomputable def f (a b : ℝ) : ℝ := a^2 + 3 / (b - 1)

-- State the theorem
theorem min_value_theorem :
  ∃ (min : ℝ), min = 2 * Real.sqrt 3 + 1 ∧
  ∀ (a b : ℝ), b > a ∧ a > 1 ∧ 3 * (Real.log b / Real.log a) + 2 * (Real.log a / Real.log b) = 7 →
  f a b ≥ min := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l267_26758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_travel_time_l267_26729

/-- Represents the time taken for a boat to travel upstream given the downstream travel time,
    distance, and river current speed. -/
noncomputable def upstreamTime (downstreamTime : ℝ) (distance : ℝ) (currentSpeed : ℝ) : ℝ :=
  let boatSpeed := distance / downstreamTime - currentSpeed
  distance / (boatSpeed - currentSpeed)

/-- Theorem stating that under the given conditions, the upstream travel time is 6 hours. -/
theorem upstream_travel_time :
  upstreamTime 4 24 1 = 6 := by
  -- Unfold the definition of upstreamTime
  unfold upstreamTime
  -- Simplify the expression
  simp
  -- The proof is completed using numerical computations
  norm_num

-- We can't use #eval for noncomputable functions, so we'll use #check instead
#check upstreamTime 4 24 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_travel_time_l267_26729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l267_26772

def S (n : ℕ) (a : ℕ → ℕ) : ℕ := n * a n - 3 * n * (n - 1)

theorem sequence_properties (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, S n a = n * a n - 3 * n * (n - 1)) :
  (∀ n : ℕ, a n = 6 * n - 5) ∧
  ∃ n : ℕ, (Finset.sum (Finset.range n) (λ i ↦ (S (i + 1) a : ℚ) / (i + 1))) - 3/2 * ((n : ℚ) - 1)^2 = 2016 ∧ n = 807 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l267_26772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l267_26783

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := (x + 1) * (x + 3) / (x - 1)^2

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | f x ≤ 0 ∧ x ≠ 1} = {x : ℝ | -3 ≤ x ∧ x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l267_26783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_residue_system_existence_l267_26747

theorem reduced_residue_system_existence (m n c : ℕ) (hm : m > 0) (hn : n > 0) 
  (hc : c > 0) (hφm : Nat.totient m = c) (hφn : Nat.totient n = c) : 
  ∃ (S : Finset ℕ), S.card = c ∧ 
    (∀ x ∈ S, Nat.Coprime x m ∧ Nat.Coprime x n) ∧
    (∀ y : ℕ, y < m → ∃ x ∈ S, x ≡ y [MOD m]) ∧
    (∀ y : ℕ, y < n → ∃ x ∈ S, x ≡ y [MOD n]) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_residue_system_existence_l267_26747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_d_l267_26703

/-- The function g(x) = 4x + c -/
noncomputable def g (c : ℤ) : ℝ → ℝ := fun x ↦ 4 * x + c

/-- The inverse function of g -/
noncomputable def g_inv (c : ℤ) : ℝ → ℝ := fun x ↦ (x - c) / 4

theorem intersection_point_d (c d : ℤ) :
  g c (-2) = d ∧ g_inv c d = -2 → d = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_d_l267_26703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l267_26732

/-- Calculates the length of a train given its speed and time to pass a fixed point. -/
noncomputable def train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time_s

/-- Theorem stating that a train traveling at 90 km/hr and passing a fixed point in 9 seconds has a length of 225 meters. -/
theorem train_length_calculation :
  train_length 90 9 = 225 := by
  unfold train_length
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_length 90 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l267_26732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l267_26742

/-- Given a train with speed in km/hr and time to pass a tree in seconds, 
    calculate its length in meters -/
noncomputable def trainLength (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (1000 / 3600) * time

/-- Theorem stating that a train with speed 63 km/hr passing a tree in 28 seconds 
    has a length of 490 meters -/
theorem train_length_calculation :
  trainLength 63 28 = 490 := by
  -- Unfold the definition of trainLength
  unfold trainLength
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- Check that the result is equal to 490
  norm_num

-- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l267_26742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AD_distance_range_l267_26762

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given conditions of the problem -/
axiom A : Point
axiom B : Point
axiom C : Point
axiom D : Point

/-- B is due east of A -/
axiom B_east_of_A : B.y = A.y

/-- C is due north of B -/
axiom C_north_of_B : C.x = B.x

/-- Distance between A and C is 20 -/
axiom AC_distance : Real.sqrt ((C.x - A.x)^2 + (C.y - A.y)^2) = 20

/-- Angle BAC is 30 degrees -/
axiom angle_BAC : Real.arccos ((B.x - A.x) / Real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)) = Real.pi / 6

/-- D is 40 meters northeast of C -/
axiom D_northeast_of_C : Real.sqrt ((D.x - C.x)^2 + (D.y - C.y)^2) = 40

/-- Angle between CD and north is 45 degrees -/
axiom angle_CD_north : Real.arctan ((D.x - C.x) / (D.y - C.y)) = Real.pi / 4

/-- Theorem: Distance AD is between 38 and 39 meters -/
theorem AD_distance_range :
  38 < Real.sqrt ((D.x - A.x)^2 + (D.y - A.y)^2) ∧ Real.sqrt ((D.x - A.x)^2 + (D.y - A.y)^2) < 39 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_AD_distance_range_l267_26762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_less_sqrt2_f_inequality_when_a_zero_l267_26768

/-- The function f(x) defined as e^(1-x)(-a + cos x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (1 - x) * (-a + Real.cos x)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 
  Real.exp (1 - x) * (a - (Real.sin x + Real.cos x))

theorem f_decreasing_iff_a_less_sqrt2 (a : ℝ) :
  (∃ x : ℝ, f_deriv a x < 0) ↔ a < Real.sqrt 2 := by sorry

theorem f_inequality_when_a_zero (x : ℝ) (h : x ∈ Set.Icc (-1) (1/2)) :
  f 0 (-x - 1) + 2 * f_deriv 0 x * Real.cos (x + 1) > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_less_sqrt2_f_inequality_when_a_zero_l267_26768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_covering_l267_26738

/-- Represents a square on the chessboard --/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the color of a square --/
inductive Color
  | Black
  | White

/-- A function to determine the color of a square --/
def squareColor (s : Square) : Color :=
  if (s.row + s.col) % 2 = 0 then Color.Black else Color.White

/-- Represents a chessboard with two squares removed --/
structure Chessboard where
  removed1 : Square
  removed2 : Square

/-- Represents a domino placement --/
structure Domino where
  square1 : Square
  square2 : Square

/-- Predicate to check if a domino placement is valid --/
def validDomino (d : Domino) : Prop :=
  (d.square1.row = d.square2.row ∧ d.square1.col + 1 = d.square2.col) ∨
  (d.square1.col = d.square2.col ∧ d.square1.row + 1 = d.square2.row)

/-- Predicate to check if a set of domino placements is a valid cover --/
def validCover (board : Chessboard) (cover : Set Domino) : Prop :=
  ∀ s : Square, s ≠ board.removed1 ∧ s ≠ board.removed2 →
    ∃! d, d ∈ cover ∧ (d.square1 = s ∨ d.square2 = s)

/-- The main theorem --/
theorem chessboard_covering (board : Chessboard) :
  (∃ cover : Set Domino, validCover board cover) ↔
  squareColor board.removed1 ≠ squareColor board.removed2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_covering_l267_26738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_value_l267_26790

noncomputable def f (x : ℝ) : ℝ := x - Real.pi - ⌊x / Real.pi⌋ - |Real.sin x|

theorem no_max_value : ¬∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (y : ℝ), f y = M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_value_l267_26790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_over_fifth_root_of_five_l267_26773

theorem sixth_root_over_fifth_root_of_five : 
  (5 : ℝ) ^ (1/6 : ℝ) / (5 : ℝ) ^ (1/5 : ℝ) = (5 : ℝ) ^ (-1/30 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_over_fifth_root_of_five_l267_26773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_BC_l267_26781

/-- Given vectors AB and n, and the dot product of n and AC, prove that the dot product of n and BC equals 2. -/
theorem dot_product_BC (AB n : ℝ × ℝ) (BC : ℝ × ℝ) (h1 : AB = (-1, 1)) (h2 : n = (1, 2)) 
    (h3 : n.1 * (AB.1 + BC.1) + n.2 * (AB.2 + BC.2) = 3) :
  n.1 * BC.1 + n.2 * BC.2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_BC_l267_26781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_railway_routes_l267_26736

/-- Represents a point in the railway station --/
inductive Point : Type where
  | A | B | C | D | E | F | G | H | I | J | K | L | M

/-- Represents a connection between two points --/
inductive Connection : Type where
  | connect : Point → Point → Connection

/-- The structure of the railway station --/
def station_structure : List Connection :=
  [Connection.connect Point.A Point.E,
   Connection.connect Point.A Point.F,
   Connection.connect Point.B Point.G,
   Connection.connect Point.B Point.H,
   Connection.connect Point.C Point.E,
   Connection.connect Point.C Point.F,
   Connection.connect Point.D Point.I,
   Connection.connect Point.E Point.I,
   Connection.connect Point.F Point.G,
   Connection.connect Point.F Point.H,
   Connection.connect Point.G Point.J,
   Connection.connect Point.H Point.J,
   Connection.connect Point.I Point.K,
   Connection.connect Point.I Point.L,
   Connection.connect Point.J Point.K,
   Connection.connect Point.J Point.L,
   Connection.connect Point.K Point.M,
   Connection.connect Point.L Point.M]

/-- Counts the number of routes from the left (Point A) to a given point --/
def count_routes : List Connection → Point → Nat
  | _, _ => sorry

/-- The main theorem stating that there are 18 routes through the station --/
theorem railway_routes :
  count_routes station_structure Point.M = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_railway_routes_l267_26736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_find_alpha_beta_l267_26770

noncomputable def a (α : Real) : Fin 2 → Real
  | 0 => Real.cos α
  | 1 => Real.sin α

noncomputable def b (β : Real) : Fin 2 → Real
  | 0 => Real.cos β
  | 1 => Real.sin β

def dot_product (v w : Fin 2 → Real) : Real :=
  (v 0) * (w 0) + (v 1) * (w 1)

def vector_diff_norm_squared (v w : Fin 2 → Real) : Real :=
  (v 0 - w 0)^2 + (v 1 - w 1)^2

theorem perpendicular_vectors (α β : Real) 
  (h1 : 0 < β) (h2 : β < α) (h3 : α < Real.pi) 
  (h4 : vector_diff_norm_squared (a α) (b β) = 2) : 
  dot_product (a α) (b β) = 0 := by
  sorry

def c : Fin 2 → Real
  | 0 => 0
  | 1 => 1

theorem find_alpha_beta :
  ∃ (α β : Real), 0 < β ∧ β < α ∧ α < Real.pi ∧
  (∀ (i : Fin 2), (a α i) + (b β i) = c i) ∧
  α = 5 * Real.pi / 6 ∧ β = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_find_alpha_beta_l267_26770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_l267_26735

/-- The acute angle formed by the line √3x + 3y + 4 = 0 with the positive x-axis is 5π/6 -/
theorem line_angle (x y : ℝ) : 
  (Real.sqrt 3 * x + 3 * y + 4 = 0) → 
  (∃ θ : ℝ, θ = 5 * Real.pi / 6 ∧ θ = Real.arctan (-Real.sqrt 3 / 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_l267_26735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l267_26702

noncomputable def f (x : ℝ) : ℝ := x * (3 * Real.log x + 1)

theorem tangent_line_at_one :
  let p : ℝ × ℝ := (1, 1)
  let m : ℝ := (deriv f) 1
  (λ x => m * (x - p.1) + p.2) = (λ x => 4 * x - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l267_26702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l267_26765

-- Define the function f based on the graph
noncomputable def f : ℝ → ℝ :=
  fun x => if x < -1 then -2 * x
           else if x < 3 then 2 * x + 3
           else -2 * x + 16

-- Define the property we want to prove
def satisfies_f_f_eq_4 (x : ℝ) : Prop :=
  f (f x) = 4

-- Theorem statement
theorem exactly_three_solutions :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x, x ∈ s ↔ satisfies_f_f_eq_4 x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l267_26765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_distribution_l267_26727

theorem apple_distribution (n : ℕ) (hn : n = 540) :
  (Finset.filter (fun d => d > 1 ∧ d < n) (Nat.divisors n)).card = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_distribution_l267_26727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_solution_l267_26788

-- Define the system of equations
def system (a b x : ℝ) : Prop :=
  Real.cos x = a * x + b ∧ Real.sin x + a = 0

-- State the theorem
theorem system_has_solution (a b : ℝ) 
  (h : ∃ x₁ x₂, x₁ ≠ x₂ ∧ Real.cos x₁ = a * x₁ + b ∧ Real.cos x₂ = a * x₂ + b) :
  ∃ x, system a b x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_solution_l267_26788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_less_than_cube_sin_over_x_l267_26785

theorem cos_less_than_cube_sin_over_x :
  ∀ x : ℝ, 0 < x → x < π / 2 → Real.cos x < (Real.sin x / x)^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_less_than_cube_sin_over_x_l267_26785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l267_26707

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 3 then (1/2) * x^2 + x
  else if -3 ≤ x ∧ x < 0 then -(1/2) * x^2 + x
  else 0  -- This case should never be reached given the domain

theorem f_properties :
  (∀ x ∈ Set.Icc (-3) 3, f (-x) = -f x) ∧  -- f is odd
  (∀ x ∈ Set.Ioo 0 3, f x = (1/2) * x^2 + x) ∧  -- Given definition for 0 < x ≤ 3
  (∀ x ∈ Set.Ico (-3) 0, f x = -(1/2) * x^2 + x) ∧  -- To be proved
  (∀ a ∈ Set.Ioo 0 2, f (a + 1) + f (2 * a - 1) > 0) ∧  -- To be proved
  (∀ a ∉ Set.Icc 0 2, f (a + 1) + f (2 * a - 1) ≤ 0)  -- To be proved
:= by sorry

-- No proof is required as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l267_26707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_product_l267_26751

-- Define the function f(x) = |log₂x - 1|
noncomputable def f (x : ℝ) : ℝ := |Real.log x / Real.log 2 - 1|

-- State the theorem
theorem roots_product (k : ℝ) (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hne : x₁ ≠ x₂) 
  (h₁ : f x₁ = k) (h₂ : f x₂ = k) : x₁ * x₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_product_l267_26751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_value_l267_26775

/-- The length of the side of an inscribed square in a rectangle --/
noncomputable def inscribed_square_side (rectangle_width rectangle_height : ℝ) (shaded_area_ratio : ℝ) : ℝ :=
  Real.sqrt ((1 - shaded_area_ratio) * rectangle_width * rectangle_height)

/-- Theorem: The length of the side of the inscribed square is 2√14 --/
theorem inscribed_square_side_value :
  inscribed_square_side 12 7 (1/3) = 2 * Real.sqrt 14 := by
  -- Unfold the definition of inscribed_square_side
  unfold inscribed_square_side
  -- Simplify the expression
  simp [Real.sqrt_mul, Real.sqrt_div]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_value_l267_26775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_calculation_l267_26763

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ := if a ≥ b then a else b

-- Theorem statement
theorem oplus_calculation : 
  (oplus 2.3 (7/3) + oplus (1/9) 0.1) / (oplus (4/9) 0.8) / 11 = 5/18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_calculation_l267_26763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l267_26718

/-- Line passing through (1,1) with inclination angle π/6 -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (1 + (Real.sqrt 3 / 2) * t, 1 + (1 / 2) * t)

/-- Curve C: y² = 8x -/
def curve_C (x y : ℝ) : Prop :=
  y^2 = 8 * x

/-- Point P -/
def point_P : ℝ × ℝ := (1, 1)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: Product of distances |PA| and |PB| is 28 -/
theorem intersection_distance_product :
  ∃ (t1 t2 : ℝ),
    curve_C (line_l t1).1 (line_l t1).2 ∧
    curve_C (line_l t2).1 (line_l t2).2 ∧
    t1 ≠ t2 ∧
    (distance point_P (line_l t1)) * (distance point_P (line_l t2)) = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l267_26718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_remainder_by_eight_l267_26713

theorem binary_remainder_by_eight : ∃ (n : ℕ), n = 3 := by
  -- Define the binary number 110110011011₂
  let binary_num : List Bool := [true, true, false, true, true, false, false, true, true, false, true, true]
  
  -- Function to convert binary to decimal
  let binary_to_decimal (bits : List Bool) : ℕ :=
    bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

  -- Function to get the last three bits
  let last_three_bits (bits : List Bool) : List Bool :=
    bits.reverse.take 3

  -- Calculate the remainder
  let remainder := binary_to_decimal (last_three_bits binary_num)

  -- Prove that the remainder is 3
  exists remainder
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_remainder_by_eight_l267_26713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l267_26760

-- Define A(x) as the smallest integer not less than x
noncomputable def A (x : ℝ) : ℤ := Int.ceil x

-- Theorem statement
theorem range_of_x (x : ℝ) (h_pos : x > 0) (h_eq : A (2 * x * A x) = 5) :
  1 < x ∧ x ≤ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l267_26760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_flips_value_l267_26752

/-- The probability of getting heads on a single flip --/
noncomputable def p_heads : ℝ := 1/3

/-- The probability of getting tails on a single flip --/
noncomputable def p_tails : ℝ := 2/3

/-- The number of players --/
def num_players : ℕ := 3

/-- The probability that all players flip the same number of times before getting heads --/
noncomputable def prob_same_flips : ℝ :=
  (∑' n : ℕ, (p_tails ^ (num_players * (n - 1))) * (p_heads ^ num_players))

theorem prob_same_flips_value :
  prob_same_flips = 1/19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_flips_value_l267_26752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_1544_l267_26741

/-- Represents the color of a t-shirt -/
inductive Color
| White
| Black

/-- Represents the size of a t-shirt -/
inductive Size
| Small
| Medium
| Large

/-- Represents the gender of an employee -/
inductive Gender
| Male
| Female

/-- The price of a men's t-shirt -/
def menPrice (c : Color) (s : Size) : Nat :=
  match c, s with
  | Color.White, Size.Small => 20
  | Color.White, Size.Medium => 24
  | Color.White, Size.Large => 28
  | Color.Black, Size.Small => 18
  | Color.Black, Size.Medium => 22
  | Color.Black, Size.Large => 26

/-- The price of a women's t-shirt -/
def womenPrice (c : Color) (s : Size) : Nat :=
  menPrice c s - 5

/-- The total number of employees -/
def totalEmployees : Nat := 40

/-- The number of employees for each size -/
def employeesPerSize (s : Size) : Nat :=
  match s with
  | Size.Small => totalEmployees * 50 / 100
  | Size.Medium => totalEmployees * 30 / 100
  | Size.Large => totalEmployees * 20 / 100

/-- The total cost of t-shirts -/
def totalCost : Nat :=
  let colors := [Color.White, Color.Black]
  let sizes := [Size.Small, Size.Medium, Size.Large]
  let genders := [Gender.Male, Gender.Female]
  List.sum (List.map (fun c =>
    List.sum (List.map (fun s =>
      List.sum (List.map (fun g =>
        let price := match g with
          | Gender.Male => menPrice c s
          | Gender.Female => womenPrice c s
        price * (employeesPerSize s / 2)
      ) genders)
    ) sizes)
  ) colors)

theorem total_cost_is_1544 : totalCost = 1544 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_1544_l267_26741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_five_terms_l267_26704

/-- Given that the expansion of (x+2)^n has exactly 5 terms, 
    prove that n = 4 and the constant term is 16 -/
theorem binomial_expansion_five_terms (n : ℕ) :
  (∃ (a₀ a₁ a₂ a₃ a₄ : ℝ), (X + 2 : Polynomial ℝ) ^ n = a₀ • 1 + a₁ • X + a₂ • X^2 + a₃ • X^3 + a₄ • X^4) →
  n = 4 ∧ (Polynomial.coeff ((X + 2 : Polynomial ℝ)^n) 0 = 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_five_terms_l267_26704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_butcher_packages_l267_26791

/-- Represents the number of packages delivered by a butcher -/
structure Packages where
  value : ℕ

/-- Represents the weight of beef in pounds -/
structure Pounds where
  value : ℕ

/-- The weight of each package in pounds -/
def package_weight : Pounds := ⟨4⟩

/-- The number of packages delivered by the first butcher -/
def first_butcher_packages : Packages := ⟨10⟩

/-- The number of packages delivered by the second butcher -/
def second_butcher_packages : Packages := ⟨7⟩

/-- The total weight of all delivered ground beef in pounds -/
def total_weight : Pounds := ⟨100⟩

/-- Calculates the weight of beef from a given number of packages -/
def weight_from_packages (p : Packages) : Pounds :=
  ⟨p.value * package_weight.value⟩

/-- Theorem: The third butcher delivered 8 packages -/
theorem third_butcher_packages : 
  ∃ (third_packages : Packages), 
    third_packages = ⟨8⟩ ∧
    (weight_from_packages first_butcher_packages).value + 
    (weight_from_packages second_butcher_packages).value + 
    (weight_from_packages third_packages).value = total_weight.value :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_butcher_packages_l267_26791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_odd_function_l267_26779

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem min_shift_for_odd_function :
  ∃ (φ : ℝ), φ > 0 ∧
  (∀ (x : ℝ), f (x + φ) = -f (-x - φ)) ∧
  (∀ (ψ : ℝ), 0 < ψ ∧ ψ < φ → ∃ (y : ℝ), f (y + ψ) ≠ -f (-y - ψ)) ∧
  φ = 3 * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_odd_function_l267_26779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_after_nine_years_l267_26771

/-- The price of a product after a certain number of three-year periods, given an initial price and a percentage decrease every three years. -/
noncomputable def price_after_periods (initial_price : ℝ) (percent_decrease : ℝ) (periods : ℕ) : ℝ :=
  initial_price * (1 - percent_decrease / 100) ^ periods

/-- Theorem stating that a product with an initial price of 640 yuan, decreasing by 25% every three years, will cost 270 yuan after 9 years. -/
theorem price_after_nine_years :
  let initial_price : ℝ := 640
  let percent_decrease : ℝ := 25
  let periods : ℕ := 3
  price_after_periods initial_price percent_decrease periods = 270 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_after_nine_years_l267_26771
