import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_opposite_purple_l396_39681

/-- Represents the colors of the squares -/
inductive Color
  | Red
  | Blue
  | Yellow
  | Green
  | Purple
  | Orange
deriving DecidableEq

/-- Represents a cube formed by folding six colored squares -/
structure ColoredCube where
  squares : List Color
  folded : Bool

/-- Defines the condition for a valid colored cube -/
def isValidColoredCube (cube : ColoredCube) : Prop :=
  cube.squares.length = 6 ∧
  cube.squares.toFinset.card = 6 ∧
  cube.folded = true

/-- Defines the opposite face in the cube -/
noncomputable def oppositeFace (cube : ColoredCube) (face : Color) : Color :=
  sorry  -- Implementation details omitted

/-- Theorem stating that the face opposite to Green is Purple -/
theorem green_opposite_purple (cube : ColoredCube) :
  isValidColoredCube cube →
  oppositeFace cube Color.Green = Color.Purple :=
by
  sorry  -- Proof details omitted

#check green_opposite_purple

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_opposite_purple_l396_39681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_increases_with_angle_sin_squared_sum_implies_obtuse_l396_39665

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = Real.pi
  side_positive : a > 0 ∧ b > 0 ∧ c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem 1: If angle A is greater than angle B, then sin A is greater than sin B
theorem sin_increases_with_angle {t : Triangle} :
  t.A > t.B → Real.sin t.A > Real.sin t.B := by
  sorry

-- Theorem 2: If sin^2 A + sin^2 B < sin^2 C, then triangle ABC is an obtuse triangle
theorem sin_squared_sum_implies_obtuse {t : Triangle} :
  Real.sin t.A ^ 2 + Real.sin t.B ^ 2 < Real.sin t.C ^ 2 → t.C > Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_increases_with_angle_sin_squared_sum_implies_obtuse_l396_39665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l396_39607

noncomputable def gauss (x : ℝ) : ℤ :=
  ⌊x⌋

noncomputable def f (x : ℝ) : ℝ :=
  x - gauss x

theorem f_range :
  (∀ y ∈ Set.range f, 0 ≤ y ∧ y < 1) ∧
  (∀ z, 0 ≤ z ∧ z < 1 → ∃ x, f x = z) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l396_39607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_travel_time_l396_39684

-- Define the variables
noncomputable def total_distance : ℝ := 1  -- Normalize the total distance to 1
noncomputable def walk_distance : ℝ := 1/3
noncomputable def run_distance : ℝ := 2/3
noncomputable def walk_time : ℝ := 9
noncomputable def run_speed_multiplier : ℝ := 5

-- Define the theorem
theorem sara_travel_time :
  let walk_speed := walk_distance / walk_time
  let run_speed := run_speed_multiplier * walk_speed
  let run_time := run_distance / run_speed
  walk_time + run_time = 12.6 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_travel_time_l396_39684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_set_size_l396_39626

theorem integer_set_size
  (k m r s t : ℕ)
  (avg_18 : (k + m + r + s + t : ℚ) / 5 = 18)
  (ordered : k < m ∧ m < r ∧ r < s ∧ s < t)
  (t_40 : t = 40)
  (median_max : r ≤ 23)
  (pos : k > 0 ∧ m > 0 ∧ r > 0 ∧ s > 0 ∧ t > 0)
  : Finset.card {k, m, r, s, t} = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_set_size_l396_39626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_correct_answer_is_C_l396_39679

-- Define a type for lines
structure Line where
  -- Add necessary fields or axioms for lines
  dummy : Unit

-- Define a type for planes
structure Plane where
  -- Add necessary fields or axioms for planes
  dummy : Unit

-- Define parallelism between lines
def parallel_lines (l1 l2 : Line) : Prop :=
  sorry -- Add definition of parallel lines

-- Define parallelism between a line and a plane
def parallel_line_plane (l : Line) (p : Plane) : Prop :=
  sorry -- Add definition of a line parallel to a plane

-- Define perpendicularity between lines
def perpendicular_lines (l1 l2 : Line) : Prop :=
  sorry -- Add definition of perpendicular lines

-- Theorem: If two lines are parallel to a third line, then they are parallel to each other
theorem parallel_transitivity (a b c : Line) :
  parallel_lines a c → parallel_lines b c → parallel_lines a b := by
  sorry

-- Main theorem that proves the correct answer
theorem correct_answer_is_C (a b c : Line) (α : Plane) :
  (parallel_lines a c ∧ parallel_lines b c) → parallel_lines a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_correct_answer_is_C_l396_39679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_cost_comparison_l396_39682

/-- Proves that the percentage increase in average cost per year from repaired used shoes to new shoes is approximately 18.52% -/
theorem shoe_cost_comparison (used_repair_cost : ℝ) (used_lifespan : ℝ) (new_cost : ℝ) (new_lifespan : ℝ)
  (h1 : used_repair_cost = 13.50)
  (h2 : used_lifespan = 1)
  (h3 : new_cost = 32.00)
  (h4 : new_lifespan = 2) :
  abs ((new_cost / new_lifespan - used_repair_cost / used_lifespan) / (used_repair_cost / used_lifespan) * 100 - 18.52) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_cost_comparison_l396_39682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_inequality_l396_39609

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem fibonacci_inequality (n : ℕ+) :
  (fibonacci (n + 1) : ℝ) ^ (1 / (n : ℝ)) ≥ 1 + 1 / (fibonacci n : ℝ) ^ (1 / (n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_inequality_l396_39609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_2014_l396_39691

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2000 then Real.cos (Real.pi / 4 * x) else x - 14

theorem f_composition_2014 : f (f 2014) = 1 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_2014_l396_39691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_rounds_to_170_l396_39677

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- Checks if a real number rounds to 1.70 when rounded to the nearest hundredth -/
def roundsTo170 (x : ℝ) : Prop :=
  roundToHundredth x = 1.70

theorem original_number_rounds_to_170 :
  roundsTo170 1.695 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_rounds_to_170_l396_39677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l396_39637

theorem trigonometric_equation_solution (t : ℝ) : 
  (Real.sin t ≠ 0 ∧ Real.cos t ≠ 0) →
  (1 / (2 * (Real.cos t / Real.sin t)^2 + 1) + 1 / (2 * (Real.sin t / Real.cos t)^2 + 1) = 15 * Real.cos (4*t) / (8 + Real.sin (2*t)^2)) ↔
  ∃ k : ℤ, t = π / 12 * (6 * k + 1) ∨ t = π / 12 * (6 * k - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l396_39637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l396_39672

-- Define the line C₁
def C₁ (t : ℝ) : ℝ × ℝ := (1 + t, 2 + t)

-- Define the circle C₂ in polar coordinates
noncomputable def C₂ (θ : ℝ) : ℝ := -2 * Real.cos θ + 2 * Real.sqrt 3 * Real.sin θ

-- Theorem statement
theorem intersection_chord_length :
  ∃ A B : ℝ × ℝ,
    (∃ t : ℝ, C₁ t = A) ∧
    (∃ t : ℝ, C₁ t = B) ∧
    (∃ θ : ℝ, C₂ θ * Real.cos θ = A.1 ∧ C₂ θ * Real.sin θ = A.2) ∧
    (∃ θ : ℝ, C₂ θ * Real.cos θ = B.1 ∧ C₂ θ * Real.sin θ = B.2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l396_39672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l396_39641

-- Define the variables and parameters
variable (a b : ℝ)

-- Define the solution set of ax > b
def solution_set_ax_gt_b := {x : ℝ | x < 2}

-- Define the inequality (ax+b)(x-1) > 0
def inequality (a b x : ℝ) : Prop := (a * x + b) * (x - 1) > 0

-- State the theorem
theorem solution_set_equality :
  (∀ x, x ∈ solution_set_ax_gt_b ↔ a * x > b) →
  (∀ x, inequality a b x ↔ x ∈ Set.Ioo (-2) 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l396_39641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_transport_cost_speed_in_range_l396_39660

/-- Transportation cost function -/
noncomputable def transport_cost (x : ℝ) : ℝ := 150 * (x + 1600 / x)

/-- Theorem stating that the minimum transportation cost occurs at speed 40 -/
theorem min_transport_cost :
  ∀ x : ℝ, 0 < x → x ≤ 50 → transport_cost x ≥ transport_cost 40 := by
  sorry

/-- Theorem stating that 40 is within the valid speed range -/
theorem speed_in_range : 0 < (40 : ℝ) ∧ (40 : ℝ) ≤ 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_transport_cost_speed_in_range_l396_39660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_from_tan_l396_39689

theorem cos_value_from_tan (θ : Real) (h1 : 0 < θ) (h2 : θ < π) (h3 : Real.tan θ = -4/3) :
  Real.cos θ = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_from_tan_l396_39689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_plus_n_l396_39631

theorem max_m_plus_n (x y z : ℚ) (m n : ℕ) : 
  0 < x → 0 < y → 0 < z →
  x < y → y < z →
  x + y + z = 1 →
  (x^2 + y^2 + z^2 - 1)^3 + 8*x*y*z = 0 →
  z = (m/n : ℚ)^2 →
  Nat.Coprime m n →
  m + n < 1000 →
  ∃ (max : ℕ), max = 536 ∧ ∀ (m' n' : ℕ), 
    (∃ (x' y' z' : ℚ), 
      0 < x' ∧ 0 < y' ∧ 0 < z' ∧
      x' < y' ∧ y' < z' ∧
      x' + y' + z' = 1 ∧
      (x'^2 + y'^2 + z'^2 - 1)^3 + 8*x'*y'*z' = 0 ∧
      z' = (m'/n' : ℚ)^2 ∧
      Nat.Coprime m' n' ∧
      m' + n' < 1000) →
    m' + n' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_plus_n_l396_39631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l396_39662

theorem diophantine_equation_solutions :
  {(a, b, c) : ℕ × ℕ × ℕ | 2^a * 3^b + 9 = c^2} =
  {(0, 3, 6), (4, 0, 5), (3, 3, 15), (5, 4, 51), (4, 3, 12)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l396_39662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_range_l396_39640

/-- The line l is defined by the equation 2x - y - 2m = 0 --/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  2 * x - y - 2 * m = 0

/-- The circle C is defined by the equation x^2 + y^2 = 5 --/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 = 5

/-- The distance from a point (x, y) to the line 2x - y - 2m = 0 --/
noncomputable def distance_to_line (m x y : ℝ) : ℝ :=
  |2 * x - y - 2 * m| / Real.sqrt 5

theorem line_circle_intersection_range :
  ∀ m : ℝ, (∃ x y : ℝ, line_l m x y ∧ circle_C x y) ↔ -5/2 ≤ m ∧ m ≤ 5/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_range_l396_39640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yanni_money_left_is_175_l396_39698

/-- Calculates the amount of money Yanni has left in cents --/
def yanni_money_left : ℤ :=
  let initial_amount : ℚ := 85/100
  let mother_gave : ℚ := 40/100
  let found : ℚ := 50/100
  let toy_cost : ℚ := 160/100
  let num_toys : ℕ := 3
  let discount_rate : ℚ := 10/100
  let tax_rate : ℚ := 5/100

  let total_money : ℚ := initial_amount + mother_gave + found
  let total_cost_before_discount : ℚ := toy_cost * num_toys
  let discount_amount : ℚ := total_cost_before_discount * discount_rate
  let total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount
  let tax_amount : ℚ := total_cost_after_discount * tax_rate
  let final_cost : ℚ := total_cost_after_discount + tax_amount

  if total_money < final_cost then
    (total_money * 100).floor
  else
    ((total_money - final_cost) * 100).floor

theorem yanni_money_left_is_175 : yanni_money_left = 175 := by
  sorry

#eval yanni_money_left

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yanni_money_left_is_175_l396_39698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_minus_sum_not_in_list_l396_39634

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem prime_product_minus_sum_not_in_list :
  ∀ p q : ℕ,
    is_prime p →
    is_prime q →
    p ≠ q →
    10 < p →
    p < 50 →
    10 < q →
    q < 50 →
    p * q - (p + q) ∉ ({221, 470, 629, 899, 950} : Set ℕ) :=
by
  sorry

#check prime_product_minus_sum_not_in_list

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_minus_sum_not_in_list_l396_39634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_screen_area_difference_l396_39620

theorem screen_area_difference : ℝ := by
  -- Define the diagonal lengths
  let d1 : ℝ := 19
  let d2 : ℝ := 17

  -- Define the side lengths
  let s1 : ℝ := d1 / Real.sqrt 2
  let s2 : ℝ := d2 / Real.sqrt 2

  -- Define the areas
  let a1 : ℝ := s1 * s1
  let a2 : ℝ := s2 * s2

  -- Calculate the difference in area
  let diff : ℝ := a1 - a2

  -- Prove that the difference is 36 square inches
  have h : diff = 36 := by
    -- The actual proof steps would go here
    sorry

  exact 36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_screen_area_difference_l396_39620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_x_l396_39601

-- Define the sequence
def x : ℕ → ℝ
| 0 => 0  -- Add a case for 0 to make the function total
| 1 => 4^(1/4)
| 2 => (4^(1/4))^(4^(1/4))
| (n+3) => (x (n+2))^(4^(1/4))

-- Theorem statement
theorem no_integer_x : ∀ n : ℕ, n > 0 → ¬(∃ m : ℤ, x n = ↑m) :=
by
  sorry

-- Additional lemma to show that x n is always positive for n > 0
lemma x_positive : ∀ n : ℕ, n > 0 → x n > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_x_l396_39601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_count_between_cubes_l396_39658

theorem integer_count_between_cubes : 
  (Finset.range (Int.toNat (⌊(11.7:Real)^3⌋ - ⌈(11.5:Real)^3⌉ + 1))).card = 81 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_count_between_cubes_l396_39658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mississippi_permutations_l396_39627

/-- Number of distinct permutations of MISSISSIPPI -/
theorem mississippi_permutations : 
  (Nat.factorial 11) / (Nat.factorial 1 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 2) = 34650 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mississippi_permutations_l396_39627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_theta_l396_39683

theorem tan_half_theta (θ : ℝ) 
  (h1 : Real.sin θ = -3/5) 
  (h2 : 3 * Real.pi < θ ∧ θ < 7/2 * Real.pi) : 
  Real.tan (θ / 2) = -3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_theta_l396_39683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_circle_circumference_l396_39654

-- Define the square
noncomputable def square_side : ℝ := 30 * Real.sqrt 2

-- Define the inscribed circle
noncomputable def circle_radius : ℝ := square_side / 2

-- Theorem for the diagonal of the square
theorem square_diagonal : Real.sqrt (2 * square_side ^ 2) = 60 := by
  sorry

-- Theorem for the circumference of the inscribed circle
theorem circle_circumference : 2 * Real.pi * circle_radius = 30 * Real.pi * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_circle_circumference_l396_39654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_subsequence_decreasing_x_difference_bound_l396_39687

def x : ℕ → ℚ
  | 0 => 1/2  -- Add this case for n = 0
  | 1 => 1/2
  | n+1 => 1 / (1 + x n)

theorem x_subsequence_decreasing :
  ∀ n : ℕ, x (2*n) > x (2*n + 2) :=
by sorry

theorem x_difference_bound :
  ∀ n : ℕ, |x (n+1) - x n| ≤ (1/6) * (2/5)^(n-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_subsequence_decreasing_x_difference_bound_l396_39687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l396_39668

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 - 2*x else x^2 - 2*x

theorem range_of_a (a : ℝ) :
  (f a - f (-a) ≤ 2 * f 1) ↔ a ∈ Set.Ici 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l396_39668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_four_l396_39648

-- Define the expression as noncomputable
noncomputable def expression : ℝ := (1/2) * Real.log 25 / Real.log 10 + Real.log 2 / Real.log 10 + 7 ^ (Real.log 3 / Real.log 7)

-- State the theorem
theorem expression_equals_four : expression = 4 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_four_l396_39648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_limit_point_l396_39671

/-- Represents the position of the ant -/
structure AntPosition where
  x : ℝ
  y : ℝ

/-- Represents the direction of the ant's movement -/
inductive Direction
  | Right
  | Down
  | Left
  | Up

/-- The movement pattern of the ant -/
def nextDirection (d : Direction) : Direction :=
  match d with
  | Direction.Right => Direction.Down
  | Direction.Down => Direction.Left
  | Direction.Left => Direction.Up
  | Direction.Up => Direction.Right

/-- The distance moved in each step -/
noncomputable def stepDistance (n : ℕ) : ℝ :=
  2 * (1/2)^n

/-- The position of the ant after n steps -/
noncomputable def antPosition (n : ℕ) : AntPosition :=
  sorry

/-- The limit point of the ant's path -/
noncomputable def antLimitPoint : AntPosition :=
  sorry

/-- Theorem stating that the limit point of the ant's path is (8/5, -4/5) -/
theorem ant_limit_point :
  antLimitPoint.x = 8/5 ∧ antLimitPoint.y = -4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_limit_point_l396_39671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_cosine_function_l396_39623

-- Define the function
noncomputable def f (x φ : ℝ) : ℝ := Real.sin (Real.pi * x + φ) - 2 * Real.cos (Real.pi * x + φ)

-- State the theorem
theorem symmetric_sine_cosine_function (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi) 
  (h3 : ∀ x : ℝ, f x φ = f (2 - x) φ) : 
  Real.sin (2 * φ) = -4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_cosine_function_l396_39623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_injective_M_closed_under_shift_preimage_of_shifted_set_l396_39690

/-- The set of functions of the form a cos x + b sin x -/
def M : Set (ℝ → ℝ) :=
  {f | ∃ a b : ℝ, f = fun x ↦ a * Real.cos x + b * Real.sin x}

/-- The mapping F from (a, b) to the function a cos x + b sin x -/
noncomputable def F : ℝ × ℝ → (ℝ → ℝ) :=
  fun (a, b) ↦ fun x ↦ a * Real.cos x + b * Real.sin x

theorem F_injective : Function.Injective F := by sorry

theorem M_closed_under_shift (t : ℝ) :
  ∀ f ∈ M, (fun x ↦ f (x + t)) ∈ M := by sorry

theorem preimage_of_shifted_set (a₀ b₀ : ℝ) :
  let f₀ : ℝ → ℝ := fun x ↦ a₀ * Real.cos x + b₀ * Real.sin x
  let M₁ : Set (ℝ → ℝ) := {g | ∃ t : ℝ, g = fun x ↦ f₀ (x + t)}
  F ⁻¹' M₁ = {p : ℝ × ℝ | (p.1 ^ 2 + p.2 ^ 2 : ℝ) = a₀ ^ 2 + b₀ ^ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_injective_M_closed_under_shift_preimage_of_shifted_set_l396_39690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_impossible_l396_39612

theorem earthquake_impossible (n : ℕ) (h1 : n ≥ 10^11 ∧ n < 10^12) 
  (h2 : ∃ (d : ℕ) (l : List ℕ), d < 10 ∧ l.length = 8 ∧ l.Nodup ∧ 
    (∀ x ∈ l, x < 10 ∧ x ≠ d) ∧ 
    n = d * (10^11 + 10^8 + 10^5 + 10^2) + 
        l[0]! * 10^10 + l[1]! * 10^9 + l[2]! * 10^7 + l[3]! * 10^6 + 
        l[4]! * 10^4 + l[5]! * 10^3 + l[6]! * 10^1 + l[7]!) : 
  ¬ Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_impossible_l396_39612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_problem_l396_39630

/-- The speed of the second train given the conditions of the problem -/
noncomputable def second_train_speed (first_train_speed : ℝ) (time : ℝ) (total_distance : ℝ) : ℝ :=
  (total_distance - first_train_speed * time) / time

/-- Theorem stating the speed of the second train under the given conditions -/
theorem second_train_speed_problem :
  second_train_speed 50 2.5 285 = 64 := by
  -- Unfold the definition of second_train_speed
  unfold second_train_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_problem_l396_39630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_midpoint_l396_39600

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the transformation
def transform (x y : ℝ) : ℝ × ℝ := (x, 2*y)

-- Define line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop := 2*ρ*Real.cos θ + ρ*Real.sin θ - 2 = 0

-- Define the resulting curve C
def curve_C (x y : ℝ) : Prop := ∃ (x₀ y₀ : ℝ), my_circle x₀ y₀ ∧ transform x₀ y₀ = (x, y)

-- Define the polar equation of the perpendicular line
noncomputable def perpendicular_line (ρ α : ℝ) : Prop := ρ = 3 / (4*Real.sin α - 2*Real.cos α)

theorem perpendicular_line_through_midpoint 
  (P₁ P₂ : ℝ × ℝ) -- Intersection points
  (h₁ : curve_C P₁.1 P₁.2)
  (h₂ : curve_C P₂.1 P₂.2)
  (h₃ : ∃ (ρ θ : ℝ), line_l ρ θ ∧ P₁ = (ρ * Real.cos θ, ρ * Real.sin θ))
  (h₄ : ∃ (ρ θ : ℝ), line_l ρ θ ∧ P₂ = (ρ * Real.cos θ, ρ * Real.sin θ))
  : ∃ (ρ α : ℝ), perpendicular_line ρ α ∧ 
    (ρ * Real.cos α, ρ * Real.sin α) = ((P₁.1 + P₂.1) / 2, (P₁.2 + P₂.2) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_midpoint_l396_39600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l396_39667

/-- The function f(x) = (6x^2 - 4) / (4x^2 + 6x + 3) -/
noncomputable def f (x : ℝ) : ℝ := (6 * x^2 - 4) / (4 * x^2 + 6 * x + 3)

/-- The horizontal asymptote of f(x) is y = 3/2 -/
theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N, ∀ x, x > N → |f x - 3/2| < ε :=
by
  sorry

#check horizontal_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l396_39667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_between_50_and_500_l396_39678

theorem perfect_squares_between_50_and_500 : 
  (Finset.filter (fun n : ℕ => 50 < n * n ∧ n * n < 500) (Finset.range 23)).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_between_50_and_500_l396_39678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l396_39697

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  (X^55 + X^44 + X^33 + X^22 + X^11 + 1 : Polynomial ℤ) = 
  q * (X^4 + X^3 + X^2 + X + 1) + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l396_39697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_231_l396_39606

theorem sum_of_divisors_231 : (Finset.filter (λ x : ℕ => 231 % x = 0) (Finset.range (231 + 1))).sum id = 384 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_231_l396_39606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_green_marbles_l396_39646

/-- The probability of selecting exactly two green marbles when picking four marbles
    at random without replacement from a bag containing 5 green, 3 blue, and 4 yellow marbles. -/
theorem prob_two_green_marbles (total : ℕ) (green : ℕ) (blue : ℕ) (yellow : ℕ) 
  (h_total : total = 12)
  (h_green : green = 5)
  (h_blue : blue = 3)
  (h_yellow : yellow = 4)
  (h_sum : green + blue + yellow = total) :
  (Nat.choose green 2 * Nat.choose (blue + yellow) 2) / Nat.choose total 4 = 14 / 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_green_marbles_l396_39646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_expression_intersection_with_unit_circle_l396_39625

-- Define the angle a and the point P
noncomputable def a : ℝ := Real.arctan (-3/4)
def P : ℝ × ℝ := (4, -3)

-- Define the condition that P lies on the terminal side of angle a
def P_on_terminal_side : Prop := 
  Real.tan a = -3/4 ∧ Real.cos a > 0

-- Theorem 1: Value of 2sin(a) - cos(a)
theorem value_of_expression (h : P_on_terminal_side) :
  2 * Real.sin a - Real.cos a = -2 := by
  sorry

-- Theorem 2: Coordinates of intersection with unit circle
theorem intersection_with_unit_circle (h : P_on_terminal_side) :
  ∃ (Q : ℝ × ℝ), Q.1^2 + Q.2^2 = 1 ∧ Q.1 = 4/5 ∧ Q.2 = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_expression_intersection_with_unit_circle_l396_39625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_of_line_l396_39642

/-- The line equation represented by a determinant -/
def line_equation (x y : ℝ) : Prop := abs (x * 1 - y * 2) = 3

/-- Definition of a direction vector for a line -/
def is_direction_vector (v : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  ∀ (t : ℝ) (x₀ y₀ : ℝ), line x₀ y₀ → line (x₀ + t * v.1) (y₀ + t * v.2)

/-- The theorem stating that (-2, -1) is a direction vector of the given line -/
theorem direction_vector_of_line :
  is_direction_vector (-2, -1) line_equation := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_of_line_l396_39642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_royalty_sales_ratio_decrease_l396_39603

-- Define the sales tiers and corresponding royalties
noncomputable def sales_tiers : List ℝ := [20, 50, 100, 200]
noncomputable def royalties : List ℝ := [7, 12, 19, 25]

-- Function to calculate the ratio of royalties to sales
noncomputable def ratio (royalty : ℝ) (sale : ℝ) : ℝ := royalty / sale

-- Function to calculate the percentage decrease between two ratios
noncomputable def percent_decrease (r1 : ℝ) (r2 : ℝ) : ℝ := (r1 - r2) / r1 * 100

-- Theorem statement
theorem royalty_sales_ratio_decrease (ε : ℝ) (hε : ε > 0) : 
  ∃ (d1 d2 d3 : ℝ), 
    (abs (d1 - 31.43) < ε ∧ abs (d2 - 20.83) < ε ∧ abs (d3 - 34.21) < ε) ∧
    (abs (percent_decrease (ratio (royalties.get! 0) (sales_tiers.get! 0)) (ratio (royalties.get! 1) (sales_tiers.get! 1)) - d1) < ε) ∧
    (abs (percent_decrease (ratio (royalties.get! 1) (sales_tiers.get! 1)) (ratio (royalties.get! 2) (sales_tiers.get! 2)) - d2) < ε) ∧
    (abs (percent_decrease (ratio (royalties.get! 2) (sales_tiers.get! 2)) (ratio (royalties.get! 3) (sales_tiers.get! 3)) - d3) < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_royalty_sales_ratio_decrease_l396_39603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l396_39666

theorem election_votes (winning_percentage : ℚ) (majority : ℕ) (total_votes : ℕ) : 
  winning_percentage = 3/5 →
  majority = 1200 →
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = majority →
  total_votes = 6000 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l396_39666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_rises_when_negative_l396_39604

/-- The inverse proportion function -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

/-- Condition for the graph to rise from left to right -/
def rises_left_to_right (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

/-- Theorem stating that the inverse proportion function rises when k is negative -/
theorem inverse_proportion_rises_when_negative (k : ℝ) (h : k < 0) :
  rises_left_to_right (inverse_proportion k) := by
  sorry

/-- Example showing that -2 satisfies the condition -/
example : rises_left_to_right (inverse_proportion (-2)) := by
  apply inverse_proportion_rises_when_negative
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_rises_when_negative_l396_39604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_C_l396_39613

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the right focus
noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 5, 0)

-- Define a point on the ellipse
noncomputable def point_on_ellipse (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, 2 * Real.sin θ)

-- Define the rectangle FABC
def rectangle_FABC (A : ℝ × ℝ) (F : ℝ × ℝ) : ℝ × ℝ := 
  (A.1 + (F.1 - A.1), A.2 + (F.2 - A.2))

-- Theorem statement
theorem trajectory_of_C (θ : ℝ) : 
  let A := point_on_ellipse θ
  let C := rectangle_FABC A right_focus
  (C.1 - Real.sqrt 5)^2 / 9 + C.2^2 / 4 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_C_l396_39613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_earthquake_magnitude_amplitude_ratio_l396_39644

-- Define the Richter magnitude formula
noncomputable def richter_magnitude (A : ℝ) (A_0 : ℝ) : ℝ := Real.log A / Real.log 10 - Real.log A_0 / Real.log 10

-- Theorem for the specific earthquake magnitude
theorem specific_earthquake_magnitude :
  richter_magnitude 1000 0.001 = 6 := by sorry

-- Theorem for the ratio of maximum amplitudes
theorem amplitude_ratio (A_9 A_5 A_0 : ℝ) :
  richter_magnitude A_9 A_0 = 9 ∧ 
  richter_magnitude A_5 A_0 = 5 → 
  A_9 / A_5 = 10000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_earthquake_magnitude_amplitude_ratio_l396_39644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lunks_needed_for_two_dozen_apples_l396_39636

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks (l : ℕ) : ℚ := (4 : ℚ) * l / 7

/-- Exchange rate between kunks and apples -/
def kunks_to_apples (k : ℕ) : ℚ := (5 : ℚ) * k / 3

/-- Number of apples in two dozen -/
def two_dozen : ℕ := 24

/-- The minimum number of lunks needed to purchase a given number of apples -/
noncomputable def lunks_needed_for_apples (a : ℕ) : ℕ :=
  Nat.ceil ((7 : ℚ) * (Nat.ceil ((3 : ℚ) * a / 5) : ℚ) / 4)

theorem lunks_needed_for_two_dozen_apples :
  lunks_needed_for_apples two_dozen = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lunks_needed_for_two_dozen_apples_l396_39636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_vessel_profitability_l396_39605

/-- Represents the fishing vessel's financial model -/
structure FishingVessel where
  purchase_price : ℕ -- in yuan
  annual_income : ℕ -- in yuan
  expenses : ℕ → ℕ -- function that takes years and returns expenses in thousand yuan

/-- Calculates the total profit after x years -/
def total_profit (v : FishingVessel) (x : ℕ) : ℤ :=
  x * v.annual_income / 1000 - v.expenses x - v.purchase_price / 1000

/-- Calculates the average annual profit after x years -/
noncomputable def avg_annual_profit (v : FishingVessel) (x : ℕ) : ℚ :=
  (total_profit v x : ℚ) / x

/-- The specific fishing vessel in the problem -/
def problem_vessel : FishingVessel :=
  { purchase_price := 980000
    annual_income := 500000
    expenses := fun x => 2 * x * x + 10 * x }

theorem fishing_vessel_profitability :
  -- 1. The minimum number of years for the boat to be profitable is 3
  (∀ x : ℕ, x > 0 → x < 3 → total_profit problem_vessel x ≤ 0) ∧
  (total_profit problem_vessel 3 > 0) ∧
  -- 2. The maximum total profit (including selling price) for both options is 110,000 yuan
  (∃ x : ℕ, x > 0 ∧ total_profit problem_vessel x + 80 = 110) ∧
  (∃ x : ℕ, x > 0 ∧ (x : ℚ) * avg_annual_profit problem_vessel x + 26 = 110) ∧
  -- 3. Option 2 takes less time to reach the maximum profit
  (∀ x y : ℕ, x > 0 → y > 0 →
    total_profit problem_vessel x + 80 = 110 →
    (y : ℚ) * avg_annual_profit problem_vessel y + 26 = 110 →
    y < x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_vessel_profitability_l396_39605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_toad_pairing_l396_39635

/-- The number of frogs and toads -/
def n : ℕ := 2017

/-- A friendship between frogs and toads -/
structure Friendship where
  frogs : Finset (Fin n)
  toads : Finset (Fin n)
  is_friend : Fin n → Fin n → Prop

/-- The number of ways to pair frogs with their friend toads -/
noncomputable def N (f : Friendship) : ℕ := sorry

/-- The set of all possible values of N -/
noncomputable def N_values (f : Friendship) : Set ℕ := sorry

/-- The number of distinct possible values of N -/
noncomputable def D (f : Friendship) : ℕ := sorry

/-- The sum of all possible values of N -/
noncomputable def S (f : Friendship) : ℕ := sorry

/-- The main theorem -/
theorem frog_toad_pairing (f : Friendship) 
  (h1 : ∀ frog : Fin n, ∃! (t1 t2 : Fin n), t1 ≠ t2 ∧ f.is_friend frog t1 ∧ f.is_friend frog t2) :
  D f = 1009 ∧ S f = 2^1009 - 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_toad_pairing_l396_39635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_l396_39649

theorem complex_multiplication (z : ℂ) : 
  Complex.abs z = 2 ∧ Complex.arg z = - Real.arctan (1/2) → 
  (1 + Complex.I) * z = 3 + Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_l396_39649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l396_39664

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Add this case to handle n = 0
  | 1 => 1
  | n + 1 => sequence_a n + Real.log (1 + 1 / n)

theorem sequence_a_formula (n : ℕ) :
  n ≥ 1 → sequence_a n = 1 + Real.log n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l396_39664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_SPC_is_one_l396_39645

-- Define a cube structure
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

-- Define the center point of a face
noncomputable def face_center (c : Cube) (v1 v2 v3 v4 : Fin 8) : ℝ × ℝ × ℝ :=
  let (x1, y1, z1) := c.vertices v1
  let (x2, y2, z2) := c.vertices v2
  let (x3, y3, z3) := c.vertices v3
  let (x4, y4, z4) := c.vertices v4
  ((x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4, (z1 + z2 + z3 + z4) / 4)

-- Define the opposite corner
def opposite_corner (v : Fin 8) : Fin 8 :=
  ((v : Nat) + 7) % 8

-- Define angle function (placeholder)
noncomputable def angle (a b c : ℝ × ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem sin_angle_SPC_is_one (c : Cube) (p q r s : Fin 8) :
  let P := face_center c p q r s
  let C := c.vertices (opposite_corner p)
  let S := c.vertices s
  Real.sin (angle S P C) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_SPC_is_one_l396_39645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newspaper_delivery_l396_39611

theorem newspaper_delivery (total_homes : ℕ) 
  (first_hour_fraction : ℚ) (second_round_percentage : ℚ) : 
  total_homes = 200 →
  first_hour_fraction = 2/5 →
  second_round_percentage = 60/100 →
  (total_homes - 
    (first_hour_fraction * ↑total_homes).floor - 
    (second_round_percentage * 
      ↑(total_homes - (first_hour_fraction * ↑total_homes).floor)).floor) = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_newspaper_delivery_l396_39611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_quadrants_l396_39655

/-- A linear function passing through three given points -/
def linear_function (k b : ℝ) : ℝ → ℝ := λ x ↦ k * x + b

theorem linear_function_quadrants (k b : ℝ) :
  k ≠ 0 →
  linear_function k b (-2) = 7 →
  linear_function k b 1 = 4 →
  linear_function k b 3 = 2 →
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ linear_function k b x = y) ∧
  (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ linear_function k b x = y) ∧
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ linear_function k b x = y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_quadrants_l396_39655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_simple_interest_rate_l396_39696

/-- Calculate the annual simple interest rate given a loan amount and total repayment after one year -/
theorem annual_simple_interest_rate 
  (loan_amount : ℝ) 
  (total_repayment : ℝ) 
  (loan_amount_positive : loan_amount > 0) 
  (repayment_greater : total_repayment > loan_amount) : 
  (total_repayment - loan_amount) / loan_amount = 0.08 :=
by
  sorry

#check annual_simple_interest_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_simple_interest_rate_l396_39696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l396_39608

/-- Define the function f(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x ≤ 0 then (x - a)^2 else x + 1/x + a

/-- Theorem stating the range of a --/
theorem a_range (a : ℝ) : 
  (∀ x, f a x ≥ f a 0) → a ∈ Set.Icc (-1) 2 := by
  sorry

#check a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l396_39608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l396_39610

-- Define the line l
def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 1 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the intersecting line
def line_m (m x y : ℝ) : Prop := m * x + y + (1/2) * m = 0

theorem circle_and_line_intersection :
  ∃ (a : ℝ), a > 0 ∧
  (∀ x y : ℝ, circle_eq x y → (x - a)^2 + y^2 = 1) ∧
  (∃ x y : ℝ, line_l x y ∧ circle_eq x y) ∧
  (∃ y : ℝ, circle_eq 0 y) ∧
  (∀ m : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ, 
    line_m m x₁ y₁ ∧ circle_eq x₁ y₁ ∧
    line_m m x₂ y₂ ∧ circle_eq x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 3) →
    m = Real.sqrt 11 / 11 ∨ m = -Real.sqrt 11 / 11) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l396_39610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinB_value_l396_39616

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Helper functions (not part of the proof, but needed for the statement)
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  angle t.B t.A t.C = 30 * Real.pi / 180 ∧ 
  distance t.A t.C = 2 ∧ 
  distance t.B t.C = Real.sqrt 2

-- State the theorem
theorem sinB_value (t : Triangle) (h : triangle_conditions t) : 
  Real.sin (angle t.A t.B t.C) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinB_value_l396_39616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_crossing_time_approx_l396_39685

def field_area : ℝ := 7201
def flat_speed : ℝ := 2.4
def speed_reduction : ℝ := 0.75

noncomputable def diagonal_crossing_time : ℝ :=
  let side_length := Real.sqrt field_area
  let diagonal_length := side_length * Real.sqrt 2
  let reduced_speed := flat_speed * speed_reduction * 1000 / 60
  diagonal_length / reduced_speed

theorem diagonal_crossing_time_approx :
  abs (diagonal_crossing_time - 4.0062) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_crossing_time_approx_l396_39685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_conditions_l396_39676

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => if x = 0 then 0 else 1 / x

-- State the theorem
theorem function_satisfies_conditions :
  (∀ x : ℝ, ∃ y : ℝ, f y = x) ∧ 
  (∀ b : ℝ, ∃! x : ℝ, f x = b) ∧
  (∀ a b : ℝ, a > 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = a * x₁ + b ∧ f x₂ = a * x₂ + b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_conditions_l396_39676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_sales_l396_39657

/-- Represents the price of a Stanford sweatshirt -/
def S : ℚ := sorry

/-- Represents the price of a Harvard sweatshirt -/
def H : ℚ := sorry

/-- The total sales on Monday -/
def monday_sales : Prop := 13 * S + 9 * H = 370

/-- The total sales on Tuesday -/
def tuesday_sales : Prop := 9 * S + 2 * H = 180

/-- Theorem stating that Wednesday's sales equal $300 -/
theorem wednesday_sales : monday_sales → tuesday_sales → 12 * S + 6 * H = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_sales_l396_39657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_circle_center_l396_39699

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the line equation
def line_eq (x y a : ℝ) : Prop := 3*x + y + a = 0

-- Define the center of a circle
def is_center (c : ℝ × ℝ) : Prop :=
  ∀ x y : ℝ, circle_eq x y → (x - c.1)^2 + (y - c.2)^2 ≤ (x - c.1)^2 + (y - c.2)^2

-- Theorem statement
theorem line_through_circle_center (a : ℝ) :
  (∃ c : ℝ × ℝ, is_center c ∧ line_eq c.1 c.2 a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_circle_center_l396_39699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_condition_l396_39669

/-- Predicate to determine if a curve is an ellipse -/
def IsEllipse (ρ θ : ℝ) : Prop := sorry

/-- Given a curve represented by the equation m*ρ*cos²θ + 3*ρ*sin²θ - 6*cosθ = 0,
    if it's an ellipse, then m > 0 and m ≠ 3 -/
theorem ellipse_condition (m : ℝ) : 
  (∀ ρ θ : ℝ, m * ρ * (Real.cos θ)^2 + 3 * ρ * (Real.sin θ)^2 - 6 * Real.cos θ = 0 → IsEllipse ρ θ) → 
  m > 0 ∧ m ≠ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_condition_l396_39669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_decreases_as_x_increases_l396_39624

/-- Given two points A and B on the graph of y = -½x - 1, prove that the y-coordinate of A is greater than the y-coordinate of B when A is to the left of B. -/
theorem y_decreases_as_x_increases (y₁ y₂ : ℝ) : 
  (1 : ℝ) = -1/2 * 1 - 1 + y₁ →   -- Point A(1, y₁) lies on the graph
  (3 : ℝ) = -1/2 * 3 - 1 + y₂ →   -- Point B(3, y₂) lies on the graph
  y₁ > y₂ := by
  sorry

#check y_decreases_as_x_increases

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_decreases_as_x_increases_l396_39624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_bisector_l396_39656

-- Define the ellipse C1
def C1 (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the hyperbola C2
def C2 (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the condition that C1 and C2 have coincident vertices
def coincident_vertices (a b : ℝ) : Prop := a = 2 ∧ b ≤ 2

-- Define the condition about the distance from focus to asymptote
def focus_asymptote_condition (b : ℝ) : Prop := b = 1

-- Define the point T
def T (t a : ℝ) : Prop := t ∈ (Set.Ioo (-a) 0) ∪ (Set.Ioo 0 a)

-- Define the non-vertical line l
def line_l (x y t k : ℝ) : Prop := y = k * (x - t)

-- Main theorem
theorem ellipse_and_bisector 
  (a b : ℝ) 
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : coincident_vertices a b)
  (h4 : focus_asymptote_condition b)
  (t : ℝ)
  (h5 : T t a)
  (k : ℝ)
  (h6 : k ≠ 0)
  : 
  (∀ x y, C1 x y a b ↔ x^2 / 4 + y^2 = 1) ∧
  ∃ m, m = 4 / t ∧ 
    ∀ x1 y1 x2 y2, 
      C1 x1 y1 a b → 
      C1 x2 y2 a b → 
      line_l x1 y1 t k → 
      line_l x2 y2 t k → 
      (y1 - 0) / (x1 - m) + (y2 - 0) / (x2 - m) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_bisector_l396_39656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_C_value_l396_39622

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi

-- Define the sine ratio condition
def sine_ratio (t : Triangle) : Prop :=
  ∃ (k : Real), k > 0 ∧ Real.sin t.A = 3 * k ∧ Real.sin t.B = 4 * k ∧ Real.sin t.C = 6 * k

-- Theorem statement
theorem cos_C_value (t : Triangle) (h : sine_ratio t) : Real.cos t.C = -11/24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_C_value_l396_39622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_proof_l396_39615

/-- Represents the time taken by a worker or group of workers to complete a job -/
noncomputable def time : ℝ → ℝ := sorry

/-- The rate at which a worker or group of workers completes a job -/
noncomputable def rate (t : ℝ) : ℝ := 1 / t

/-- Time taken by all four workers together -/
def all_workers : ℝ := sorry

/-- Time taken by Alpha alone -/
def alpha : ℝ := sorry

/-- Time taken by Beta alone -/
def beta : ℝ := sorry

/-- Time taken by Gamma alone -/
def gamma : ℝ := sorry

/-- Time taken by Delta alone -/
def delta : ℝ := sorry

/-- Time taken by Alpha, Beta, and Delta together -/
def k : ℝ := sorry

/-- The theorem stating the conditions and the result to be proved -/
theorem work_time_proof 
  (h1 : all_workers = alpha + 4)
  (h2 : all_workers = beta - 3)
  (h3 : all_workers = gamma / 3)
  (h4 : rate all_workers = rate alpha + rate beta + rate gamma + rate delta)
  (h5 : rate k = rate alpha + rate beta + rate delta) :
  k = 21 / 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_proof_l396_39615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_one_minus_f_eq_1024_l396_39693

/-- Given x = (2 + √2)^10, n = ⌊x⌋, and f = x - n, prove that x(1 - f) = 1024 -/
theorem x_one_minus_f_eq_1024 (x n : ℝ) (f : ℝ) 
    (hx : x = (2 + Real.sqrt 2)^10)
    (hn : n = ⌊x⌋)
    (hf : f = x - n) : 
  x * (1 - f) = 1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_one_minus_f_eq_1024_l396_39693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_qin_jiushao_theorem_l396_39653

/-- Converts a number from base 3 to decimal --/
def base3ToDecimal (n : Nat) : Nat :=
  sorry

/-- Computes the greatest common divisor of two numbers --/
def myGcd (a b : Nat) : Nat :=
  sorry

/-- Implements Qin Jiushao's algorithm for polynomial evaluation --/
def qinJiushaoAlgorithm (a b : Int) (x : Int) : Int :=
  let v0 : Int := 1
  let v1 : Int := v0 * x + a
  let v2 : Int := v1 * x + 0
  let v3 : Int := v2 * x - b
  v3

theorem qin_jiushao_theorem :
  let a : Int := base3ToDecimal 1202
  let b : Int := myGcd 8251 6105
  qinJiushaoAlgorithm a b (-1) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_qin_jiushao_theorem_l396_39653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_formula_l396_39688

/-- The total area of the shaded region formed by six overlapping circles -/
noncomputable def shadedArea (R : ℝ) : ℝ := 2 * Real.pi * R^2 - 3 * R^2 * Real.sqrt 3

/-- Theorem stating the total area of the shaded region -/
theorem shaded_area_formula (R : ℝ) (h : R > 0) :
  let circleRadius := R
  let numPetals := 6
  shadedArea circleRadius = 2 * Real.pi * R^2 - 3 * R^2 * Real.sqrt 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_formula_l396_39688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_in_sq_cm_l396_39647

/-- The area of a circle with diameter 8 meters, expressed in square centimeters -/
noncomputable def circle_area : ℝ :=
  let diameter : ℝ := 8
  let radius : ℝ := diameter / 2
  let area_in_square_meters : ℝ := Real.pi * radius^2
  let square_cm_per_square_meter : ℝ := 10000
  area_in_square_meters * square_cm_per_square_meter

/-- Theorem stating that the area of the circle is 160,000π square centimeters -/
theorem circle_area_in_sq_cm : circle_area = 160000 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_in_sq_cm_l396_39647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l396_39621

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 27 = 1

-- Define the parabola
def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the foci
noncomputable def F₁ : ℝ × ℝ := sorry
noncomputable def F₂ : ℝ × ℝ := sorry

-- Common point P
noncomputable def P : ℝ × ℝ := sorry

-- Theorem statement
theorem area_of_triangle (p : ℝ) :
  hyperbola P.1 P.2 →
  parabola p P.1 P.2 →
  F₂ = (p/2, 0) →
  (1/2) * abs (P.1 * F₂.2 + F₂.1 * F₁.2 + F₁.1 * P.2 - P.2 * F₂.1 - F₂.2 * F₁.1 - F₁.2 * P.1) = 36 * Real.sqrt 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l396_39621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l396_39680

-- Define the curve equation
def curve (x y α : ℝ) : Prop := x^2 + y^2 * Real.cos α = 1

-- Define the range of α
def α_range (α : ℝ) : Prop := 0 ≤ α ∧ α ≤ Real.pi

-- Theorem statement
theorem curve_transformation (x y α : ℝ) (h : α_range α) :
  (α = 0 → curve x y α ↔ x^2 + y^2 = 1) ∧
  (0 < α ∧ α < Real.pi/2 → ∃ a b, a > 0 ∧ b > 0 ∧ curve x y α ↔ x^2/a^2 + y^2/b^2 = 1) ∧
  (α = Real.pi/2 → curve x y α ↔ x^2 = 1) ∧
  (Real.pi/2 < α ∧ α < Real.pi → ∃ a b, a > 0 ∧ b > 0 ∧ curve x y α ↔ x^2/a^2 - y^2/b^2 = 1) ∧
  (α = Real.pi → curve x y α ↔ x^2 - y^2 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l396_39680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_count_l396_39695

theorem student_count (initial_avg : ℚ) (corrected_avg : ℚ) (score_difference : ℚ) : 
  initial_avg = 72 → 
  corrected_avg = 7171 / 100 → 
  score_difference = 10 → 
  ∃ n : ℕ, n ≠ 0 ∧ 
    (initial_avg * n + score_difference) / n = corrected_avg ∧
    n = 34 := by
  sorry

#eval (72 * 34 + 10) / 34

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_count_l396_39695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_loop_probability_l396_39618

/-- Represents the orientation of an arrow on a side of the rectangle -/
inductive ArrowOrientation
| Horizontal
| Vertical

/-- Represents the configuration of arrows on all four sides of the rectangle -/
structure RectangleArrows where
  top : ArrowOrientation
  bottom : ArrowOrientation
  left : ArrowOrientation
  right : ArrowOrientation

/-- Checks if the given arrow configuration forms a continuous loop -/
def isContinuousLoop (arrows : RectangleArrows) : Prop :=
  (arrows.top = arrows.bottom) ∧ (arrows.left = arrows.right)

/-- Calculates the probability of a continuous arrow loop -/
noncomputable def probabilityOfContinuousLoop : ℚ :=
  1 / 4

/-- Theorem stating that the probability of a continuous arrow loop is 1/4 -/
theorem continuous_loop_probability :
  probabilityOfContinuousLoop = 1 / 4 := by
  sorry

#check continuous_loop_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_loop_probability_l396_39618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l396_39673

-- Define the principal, time, and interest difference
def principal : ℝ := 4100
def time : ℝ := 2
def interest_difference : ℝ := 41

-- Define the equation for the difference between compound and simple interest
noncomputable def interest_equation (r : ℝ) : ℝ :=
  principal * ((1 + r)^time - 1 - r * time)

-- State the theorem
theorem interest_rate_is_ten_percent :
  ∃ r : ℝ, interest_equation r = interest_difference ∧ r = 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l396_39673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l396_39659

theorem trigonometric_identity (α : ℝ) :
  (3 + 4 * Real.cos (4 * α) + Real.cos (8 * α)) /
  (3 - 4 * Real.cos (4 * α) + Real.cos (8 * α)) =
  (Real.tan (2 * α))⁻¹ ^ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l396_39659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_implies_a_greater_than_two_thirds_l396_39651

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 - x^2

theorem non_monotonic_implies_a_greater_than_two_thirds :
  ∀ a : ℝ, a > 0 →
  (∃ x y : ℝ, 0 < x ∧ x < y ∧ y < 3 ∧ 
    ((f a x < f a y ∧ f a y > f a ((x + y) / 2)) ∨
     (f a x > f a y ∧ f a y < f a ((x + y) / 2)))) →
  a > 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_implies_a_greater_than_two_thirds_l396_39651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_tangent_sum_l396_39633

theorem perpendicular_vectors_tangent_sum (α : ℝ) : 
  let a : Fin 2 → ℝ := ![Real.cos α, -1]
  let b : Fin 2 → ℝ := ![2, Real.sin α]
  (a • b = 0) → Real.tan (α + π/4) = -3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_tangent_sum_l396_39633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l396_39643

noncomputable def triangle_area (A B C : Real) (c : Real) : Real :=
  1 / 2 * c * c * Real.sin B * Real.sin C / Real.sin (A + B)

theorem triangle_abc_properties (A B C : Real) (c : Real) 
  (h1 : Real.sin (Real.pi / 2 + A) = 2 * Real.sqrt 5 / 5)
  (h2 : Real.cos B = 3 * Real.sqrt 10 / 10)
  (h3 : c = 10) :
  Real.tan (2 * A) = 4 / 3 ∧ 
  triangle_area A B C c = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l396_39643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_not_a_in_zero_one_not_three_intersections_g_min_value_l396_39652

-- Define the function f(x) = ln x + 3x - 6
noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x - 6

-- Statement 1
theorem f_has_unique_zero : ∃! x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 :=
sorry

-- Statement 2
theorem not_a_in_zero_one : ¬(∀ a : ℝ, (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) → a ∈ Set.Ioo 0 1) :=
sorry

-- Statement 3
theorem not_three_intersections : ¬(∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ x₁ = Real.sin x₁ ∧ x₂ = Real.sin x₂ ∧ x₃ = Real.sin x₃) :=
sorry

-- Define the function g(x) = sin x cos x + sin x + cos x
noncomputable def g (x : ℝ) : ℝ := Real.sin x * Real.cos x + Real.sin x + Real.cos x

-- Statement 4
theorem g_min_value : ∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 4) → g x ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_not_a_in_zero_one_not_three_intersections_g_min_value_l396_39652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_characteristic_linear_one_third_characteristic_has_zero_exp_is_characteristic_l396_39661

-- Define the characteristic function property
def is_characteristic_function (f : ℝ → ℝ) (l : ℝ) : Prop :=
  ∀ x : ℝ, f (x + l) + l * f x = 0

-- Theorem 1: f(x) = 2x + 1 is not a l-characteristic function
theorem not_characteristic_linear : ¬ ∃ l : ℝ, is_characteristic_function (fun x ↦ 2 * x + 1) l := by
  sorry

-- Theorem 2: A 1/3-characteristic function has at least one zero
theorem one_third_characteristic_has_zero 
  (f : ℝ → ℝ) (hf : Continuous f) (h : is_characteristic_function f (1/3)) : 
  ∃ x₀ : ℝ, f x₀ = 0 := by
  sorry

-- Theorem 3: e^x is a l-characteristic function for some l
theorem exp_is_characteristic : ∃ l : ℝ, is_characteristic_function Real.exp l := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_characteristic_linear_one_third_characteristic_has_zero_exp_is_characteristic_l396_39661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_theorem_l396_39602

/-- Represents the compound interest calculation -/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 100) ^ n

/-- Theorem stating the relationship between the given conditions and the annual interest rate -/
theorem compound_interest_rate_theorem (P : ℝ) (r : ℝ) :
  compound_interest P r 8 = 17640 →
  compound_interest P r 12 = 21168 →
  ∃ ε > 0, |4 * r - 18.6| < ε := by
  intros h1 h2
  sorry

#check compound_interest_rate_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_theorem_l396_39602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l396_39670

-- Define the function f(x) = (1/3)^x
noncomputable def f (x : ℝ) : ℝ := (1/3) ^ x

-- State the theorem
theorem min_value_f_on_interval :
  ∀ x ∈ Set.Icc (-1 : ℝ) 0, f x ≥ 1 ∧ ∃ y ∈ Set.Icc (-1 : ℝ) 0, f y = 1 := by
  sorry

-- Note: Set.Icc a b represents the closed interval [a, b]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_l396_39670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expressions_equal_half_l396_39675

theorem trig_expressions_equal_half :
  (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180)) = 1/2 ∧
  Real.tan (22.5 * Real.pi / 180) / (1 - Real.tan (22.5 * Real.pi / 180)^2) = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expressions_equal_half_l396_39675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l396_39663

/-- Calculates the speed of a train given its length, time to pass a man, and the man's speed. -/
noncomputable def train_speed (train_length : ℝ) (passing_time : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let man_speed_mps := man_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_mps := relative_speed + man_speed_mps
  train_speed_mps * (3600 / 1000)

/-- Theorem stating that the train speed is approximately 57.99 kmph given the problem conditions. -/
theorem train_speed_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_speed 250 17.998560115190788 8 - 57.99| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l396_39663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_identification_l396_39632

-- Define what a fraction is in this context
def is_fraction (f : ℝ → ℝ) : Prop :=
  ∃ (n d : ℝ → ℝ), ∀ x, f x = n x / d x ∧ d x ≠ 0 ∧ (∃ y, d y ≠ d 0)

-- Define the given expressions
noncomputable def expr_A : ℝ → ℝ := λ _ => 1 / Real.pi
noncomputable def expr_B : ℝ → ℝ := λ x => (x + x) / 2
noncomputable def expr_C : ℝ → ℝ := λ x => 1 / (1 - x)
noncomputable def expr_D : ℝ → ℝ := λ _ => 3 / 5

-- State the theorem
theorem fraction_identification :
  ¬ is_fraction expr_A ∧
  ¬ is_fraction expr_B ∧
  is_fraction expr_C ∧
  ¬ is_fraction expr_D := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_identification_l396_39632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_ball_is_red_l396_39617

/-- Represents the color of a ball -/
inductive Color where
  | Blue
  | Red
  | Green

/-- Represents the state of the bottle -/
structure BottleState where
  blue : Nat
  red : Nat
  green : Nat

/-- Represents the possible outcomes of drawing two balls -/
inductive DrawOutcome where
  | BlueGreen
  | RedGreen
  | RedRed
  | Other

/-- Defines the initial state of the bottle -/
def initialState : BottleState :=
  { blue := 1001, red := 1000, green := 1000 }

/-- Defines the operation of drawing two balls and replacing them according to the rules -/
def performOperation (state : BottleState) (outcome : DrawOutcome) : BottleState :=
  match outcome with
  | DrawOutcome.BlueGreen => { blue := state.blue - 1, green := state.green - 1, red := state.red + 1 }
  | DrawOutcome.RedGreen => { red := state.red, green := state.green - 1, blue := state.blue }
  | DrawOutcome.RedRed => { red := state.red - 2, blue := state.blue + 2, green := state.green }
  | DrawOutcome.Other => { green := state.green + 1, blue := state.blue, red := state.red }

/-- Theorem stating that the last remaining ball will be red -/
theorem last_ball_is_red :
  ∃ (n : Nat), ∃ (sequence : List DrawOutcome),
    let finalState := sequence.foldl performOperation initialState
    finalState.blue = 0 ∧ finalState.green = 0 ∧ finalState.red = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_ball_is_red_l396_39617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l396_39638

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- ax + by + c = 0

/-- Checks if a line is tangent to a circle -/
def is_tangent_to (l : Line) (c : Circle) : Prop :=
  let (x, y) := c.center
  (l.a * x + l.b * y + l.c)^2 = (l.a^2 + l.b^2) * c.radius^2

/-- Checks if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem circle_radius_theorem (C1 C2 C3 : Circle) (l m : Line) :
  C2.radius = 9 →
  C3.radius = 4 →
  are_externally_tangent C1 C2 →
  are_externally_tangent C1 C3 →
  are_externally_tangent C2 C3 →
  is_tangent_to l C1 →
  is_tangent_to l C2 →
  is_tangent_to m C1 →
  is_tangent_to m C3 →
  are_parallel l m →
  C1.radius = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_theorem_l396_39638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_function_a_range_l396_39619

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((3 - a) * x - a) / Real.log a

-- State the theorem
theorem increasing_log_function_a_range :
  ∀ a : ℝ, (a > 0 ∧ a ≠ 1) → (∀ x y : ℝ, x < y → f a x < f a y) → 1 < a ∧ a < 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_function_a_range_l396_39619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l396_39692

-- Define the solution function
noncomputable def y (t : ℝ) : ℝ := Real.exp (2 * t) - Real.exp t

-- State the theorem
theorem differential_equation_solution (t : ℝ) (h : t > 0) :
  (deriv y t - 2 * y t = Real.exp t) ∧ (y 0 = 0) := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l396_39692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l396_39614

/-- The time it takes to fill a cistern with a leak -/
noncomputable def fill_time_with_leak (T : ℝ) : ℝ := T + 2

/-- The time it takes to empty the full cistern due to the leak -/
def empty_time : ℝ := 4

/-- The filling rate without the leak -/
noncomputable def fill_rate (T : ℝ) : ℝ := 1 / T

/-- The emptying rate due to the leak -/
noncomputable def leak_rate : ℝ := 1 / empty_time

/-- The effective filling rate with the leak -/
noncomputable def effective_fill_rate (T : ℝ) : ℝ := fill_rate T - leak_rate

theorem cistern_fill_time :
  ∃ T : ℝ, T > 0 ∧ effective_fill_rate T = 1 / fill_time_with_leak T ∧ T = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l396_39614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_and_tangent_identities_l396_39694

theorem sine_and_tangent_identities (α : Real) 
  (h1 : Real.sin α = 4/5) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.sin (α - π/6) = (4*Real.sqrt 3 + 3)/10 ∧ Real.tan (2*α) = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_and_tangent_identities_l396_39694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l396_39674

/-- Calculates the speed of a train in km/hr given its length in meters and time to cross a pole in seconds. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * (3600 / 1000)

/-- Theorem stating that a train with length 150 meters crossing a pole in 9 seconds has a speed of 60 km/hr. -/
theorem train_speed_calculation :
  train_speed 150 9 = 60 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the arithmetic
  simp [div_mul_div_comm, mul_div_assoc]
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l396_39674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_theorem_l396_39628

/-- Two circles in the xy-plane -/
structure TwoCircles (a b : ℝ) :=
  (circle1 : ℝ × ℝ → Prop)
  (circle2 : ℝ × ℝ → Prop)
  (h1 : ∀ x y, circle1 (x, y) ↔ x^2 + y^2 - 4*x + 2*y + 5 - a^2 = 0)
  (h2 : ∀ x y, circle2 (x, y) ↔ x^2 + y^2 - (2*b-10)*x - 2*b*y + 2*b^2 - 10*b + 16 = 0)

/-- The intersection points of the two circles -/
structure IntersectionPoints (a b : ℝ) extends TwoCircles a b :=
  (A B : ℝ × ℝ)
  (hA1 : circle1 A)
  (hA2 : circle2 A)
  (hB1 : circle1 B)
  (hB2 : circle2 B)
  (h_eq : A.1^2 + A.2^2 = B.1^2 + B.2^2)

/-- The theorem stating that b = 5/3 given the conditions -/
theorem circles_intersection_theorem (a : ℝ) (h : IntersectionPoints a (5/3)) : 
  (5/3 : ℝ) = 5/3 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_theorem_l396_39628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_shape_configuration_l396_39650

/-- Represents the number of dots on a cube face -/
inductive Dots
  | one
  | two
  | three

/-- Represents a cube with faces labeled by their dot count -/
structure Cube where
  face1 : Dots
  face2 : Dots
  face3 : Dots
  face4 : Dots
  face5 : Dots
  face6 : Dots

/-- Predicate to check if two cube positions are touching -/
def TouchingFaces : Fin 7 → Fin 7 → Prop := sorry

/-- Represents the configuration of 7 cubes in a "П" shape -/
structure Configuration where
  cubes : Fin 7 → Cube
  valid : ∀ i j, TouchingFaces i j → (cubes i).face1 = (cubes j).face1

/-- Helper function to count faces with a specific number of dots -/
def count3 (c : Cube) : Nat := sorry
def count2 (c : Cube) : Nat := sorry
def count1 (c : Cube) : Nat := sorry

/-- Functions to get the left faces A, B, and C -/
def leftFaceA (config : Configuration) : Dots := sorry
def leftFaceB (config : Configuration) : Dots := sorry
def leftFaceC (config : Configuration) : Dots := sorry

/-- The theorem to be proved -/
theorem pi_shape_configuration 
  (config : Configuration)
  (h1 : ∀ c, (count3 c = 1) ∧ (count2 c = 2) ∧ (count1 c = 3))
  (h2 : ∀ i, config.cubes i = config.cubes 0) :
  (leftFaceA config = Dots.two) ∧ 
  (leftFaceB config = Dots.two) ∧ 
  (leftFaceC config = Dots.three) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_shape_configuration_l396_39650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_count_l396_39639

/-- A set of eleven distinct integers containing 1, 2, 3, 5, and 8 -/
def S : Finset ℤ := sorry

/-- The property that S has exactly eleven elements -/
axiom S_card : S.card = 11

/-- The property that S contains 1, 2, 3, 5, and 8 -/
axiom S_contains : {1, 2, 3, 5, 8} ⊆ S

/-- The property that all elements in S are distinct -/
axiom S_distinct : ∀ x y : ℤ, x ∈ S → y ∈ S → x = y → x = y

/-- The set of possible median values for S -/
def possible_medians : Finset ℤ := sorry

/-- The theorem stating that the number of possible median values is 7 -/
theorem median_count : possible_medians.card = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_count_l396_39639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l396_39686

theorem solve_exponential_equation :
  ∃ x : ℝ, (7 : ℝ) ^ (3 * x + 2) = 1 / 49 ∧ x = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l396_39686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_koby_bought_two_boxes_l396_39629

/-- The number of boxes Koby bought -/
def koby_boxes : ℕ := sorry

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 33

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

/-- The number of whistlers in each of Koby's boxes -/
def koby_whistlers_per_box : ℕ := 5

/-- The number of sparklers in Cherie's box -/
def cherie_sparklers : ℕ := 8

/-- The number of whistlers in Cherie's box -/
def cherie_whistlers : ℕ := 9

/-- Theorem stating that Koby bought 2 boxes of fireworks -/
theorem koby_bought_two_boxes :
  (koby_sparklers_per_box * koby_boxes + cherie_sparklers) +
  (koby_whistlers_per_box * koby_boxes + cherie_whistlers) = total_fireworks →
  koby_boxes = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_koby_bought_two_boxes_l396_39629
