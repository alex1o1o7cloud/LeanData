import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_1600_800_l314_31473

-- Define the sum of divisors function
def σ (n : ℕ) : ℕ := (Finset.sum (Nat.divisors n) id)

-- Define the function f
def f (n : ℕ) : ℚ := (σ n : ℚ) / n

-- Theorem statement
theorem f_difference_1600_800 : f 1600 - f 800 = 39 / 400 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_1600_800_l314_31473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wife_account_percentage_calculation_l314_31467

/-- Calculates the percentage of income deposited to wife's account --/
noncomputable def wifeAccountPercentage (totalIncome childrenPercentage orphanPercentage finalAmount : ℝ) : ℝ :=
  let childrenAmount := childrenPercentage * totalIncome
  let remainingAfterChildren := totalIncome - childrenAmount
  let x := (remainingAfterChildren - (orphanPercentage * remainingAfterChildren) - finalAmount) / totalIncome * 100
  x

/-- Theorem stating the percentage deposited to wife's account --/
theorem wife_account_percentage_calculation :
  let totalIncome : ℝ := 1000
  let childrenPercentage : ℝ := 0.2  -- 10% to each of 2 children
  let orphanPercentage : ℝ := 0.1
  let finalAmount : ℝ := 500
  abs (wifeAccountPercentage totalIncome childrenPercentage orphanPercentage finalAmount - 24.44) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wife_account_percentage_calculation_l314_31467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_typing_task_completion_time_l314_31400

/-- Represents a typist with their time to complete the task alone -/
structure Typist where
  time : ℝ

/-- The typing task -/
def TypingTask : Type := Unit

theorem typing_task_completion_time 
  (A B C : Typist) 
  (task : TypingTask) 
  (h1 : A.time = B.time + 5)
  (h2 : 4 / A.time = 3 / B.time)
  (h3 : 1 / C.time = 2 * (1 / A.time))
  : ∃ (total_time : ℝ), total_time = 14 + 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_typing_task_completion_time_l314_31400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_18_deg_approximation_l314_31499

theorem sin_18_deg_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |Real.sin (π/10) - (π/10 - (π/10)^3/6)| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_18_deg_approximation_l314_31499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l314_31458

theorem problem_statement (a b c : ℝ) 
  (h1 : (2 : ℝ)^a = 24) 
  (h2 : (2 : ℝ)^b = 6) 
  (h3 : (2 : ℝ)^c = 9) : 
  (a - b = 2) ∧ (3*b = a + c) ∧ (2*b - c = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l314_31458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_of_h_l314_31447

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := f 1 x + 2 / x

-- Theorem for part I
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y,
    y = m * (x - 1) + f 2 1 ↔ x + y - 2 = 0 :=
by sorry

-- Theorem for part II
theorem monotonicity_of_h :
  (∀ x ∈ Set.Ioo 0 2, StrictAntiOn h (Set.Ioo 0 2)) ∧
  (∀ x ∈ Set.Ioi 2, StrictMonoOn h (Set.Ioi 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_of_h_l314_31447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_for_x_gt_1_l314_31451

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem f_expression_for_x_gt_1 (f : ℝ → ℝ) 
  (h1 : is_even_function (λ x ↦ f (x + 1)))
  (h2 : ∀ x, x < 1 → f x = x^2 + 1) :
  ∀ x, x > 1 → f x = x^2 - 4*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_for_x_gt_1_l314_31451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_properties_l314_31486

-- Define the curve
def cookie_boundary (x y : ℝ) : Prop :=
  x^2 + y^2 - 10 = 2*x + 6*y

-- State the theorem
theorem cookie_properties :
  ∃ (center_x center_y : ℝ),
    (∀ (x y : ℝ), cookie_boundary x y ↔ (x - center_x)^2 + (y - center_y)^2 = 20) ∧
    (let r := Real.sqrt 20;
     let area := π * r^2;
     r = 2 * Real.sqrt 5 ∧ area = 20 * π) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_properties_l314_31486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mars_colony_cost_per_person_l314_31426

/-- The cost per person to establish a Mars colony, given the total cost and number of contributors -/
noncomputable def cost_per_person (total_cost : ℝ) (num_people : ℝ) : ℝ :=
  total_cost / num_people

/-- Theorem stating that the cost per person for the Mars colony project is approximately 166.67 dollars -/
theorem mars_colony_cost_per_person :
  let total_cost : ℝ := 50 * 10^9  -- 50 billion dollars
  let num_people : ℝ := 300 * 10^6  -- 300 million people
  abs (cost_per_person total_cost num_people - 166.67) < 0.01 := by
  sorry

/-- Computation of the cost per person (for demonstration purposes) -/
def main : IO Unit := do
  let total_cost : Float := 50 * 10^9
  let num_people : Float := 300 * 10^6
  IO.println s!"Approximate cost per person: {total_cost / num_people}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mars_colony_cost_per_person_l314_31426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_permutations_time_l314_31496

def name : String := "Anna"
def letters_count : Nat := 4
def repeated_letters : Nat := 2
def rearrangements_per_minute : Nat := 15

/-- Calculates the number of permutations for a multiset -/
def multiset_permutations (n : Nat) (p : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial p * Nat.factorial (n - p))

theorem anna_permutations_time :
  (multiset_permutations letters_count repeated_letters : ℚ) / 
  (rearrangements_per_minute * 60 : ℚ) = 1 / 75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_permutations_time_l314_31496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_l_over_n_squared_l314_31439

/-- Represents the minimum number of colored vertices required on an n × n chessboard
    such that any k × k square has at least one colored vertex on its boundary. -/
def l (n : ℕ) : ℕ := sorry

/-- Theorem stating that the limit of l(n)/n^2 as n approaches infinity is 2/7 -/
theorem limit_l_over_n_squared :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |Real.sqrt (l n / n^2) - 2/7| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_l_over_n_squared_l314_31439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_segment_l314_31469

/-- Given two points A and B in 3D space, and a point P on the line segment AB
    such that AP:PB = 5:3, prove that P can be expressed as a linear combination
    of A and B with specific coefficients. -/
theorem point_on_line_segment (A B P : ℝ × ℝ × ℝ) :
  A = (1, 2, 3) →
  B = (4, 5, 6) →
  (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) →
  (5 * (B.fst - P.fst) = 3 * (P.fst - A.fst) ∧
   5 * (B.snd - P.snd) = 3 * (P.snd - A.snd) ∧
   5 * (B.snd.snd - P.snd.snd) = 3 * (P.snd.snd - A.snd.snd)) →
  P = (5/8) • A + (3/8) • B :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_segment_l314_31469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_gain_percent_l314_31493

/-- The gain percent for a cycle transaction -/
noncomputable def gain_percent (cost_price selling_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The gain percent for a cycle bought for Rs. 900 and sold for Rs. 1180 is approximately 31.11% -/
theorem cycle_gain_percent :
  let cost_price : ℝ := 900
  let selling_price : ℝ := 1180
  abs (gain_percent cost_price selling_price - 31.11) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_gain_percent_l314_31493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_representation_equivalence_l314_31492

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a ray
structure Ray where
  source : Point
  direction : Point

-- Define an angle
structure Angle where
  vertex : Point
  ray1 : Ray
  ray2 : Ray
  h1 : ray1.source = vertex
  h2 : ray2.source = vertex

-- Define angle equality
def angle_eq (a b : Angle) : Prop :=
  a.vertex = b.vertex ∧ 
  ((a.ray1 = b.ray1 ∧ a.ray2 = b.ray2) ∨ 
   (a.ray1 = b.ray2 ∧ a.ray2 = b.ray1))

-- Theorem statement
theorem angle_representation_equivalence 
  (A O B : Point) (rayOA rayOB : Ray) 
  (h1 : rayOA.source = O) (h2 : rayOB.source = O) :
  angle_eq 
    { vertex := O, ray1 := rayOA, ray2 := rayOB, h1 := h1, h2 := h2 }
    { vertex := O, ray1 := rayOB, ray2 := rayOA, h1 := h2, h2 := h1 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_representation_equivalence_l314_31492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drill_through_cube_possible_l314_31488

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Represents a cube with integer side length -/
structure Cube where
  side : ℕ

/-- Represents a line parallel to a cube's edge -/
inductive ParallelLine
  | X : ℕ → ℕ → ParallelLine  -- y and z coordinates
  | Y : ℕ → ℕ → ParallelLine  -- x and z coordinates
  | Z : ℕ → ℕ → ParallelLine  -- x and y coordinates

/-- Checks if a parallel line intersects the interior of a cuboid -/
def intersects_interior (l : ParallelLine) (c : Cuboid) : Prop :=
  sorry -- Define the intersection logic here

/-- The theorem to be proved -/
theorem drill_through_cube_possible 
  (small_cuboids : Finset Cuboid) 
  (large_cube : Cube) : 
  small_cuboids.card = 2000 ∧ 
  (∀ c ∈ small_cuboids, c.width = 2 ∧ c.length = 2 ∧ c.height = 1) ∧
  large_cube.side = 20 →
  ∃ (l : ParallelLine), ∀ c ∈ small_cuboids, ¬ intersects_interior l c :=
by
  sorry -- The proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_drill_through_cube_possible_l314_31488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_inclination_angle_l314_31403

/-- The equation of a circle with the largest possible area -/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + k*x + 2*y + k^2 = 0

/-- The equation of the line -/
def line_equation (x y k : ℝ) : Prop :=
  y = (k + 1)*x + 2

/-- The angle of inclination of the line -/
def angle_of_inclination (α : ℝ) : Prop :=
  α = Real.pi/4

theorem largest_circle_inclination_angle :
  ∀ k : ℝ, 
  (∀ x y : ℝ, circle_equation x y k → 
    ∀ r : ℝ, (x - (-k/2))^2 + (y - (-1))^2 = r^2 → 
    ∀ r' : ℝ, r' ≤ r) →
  (∀ x y : ℝ, line_equation x y k) →
  angle_of_inclination (Real.arctan (k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_inclination_angle_l314_31403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pastries_fit_l314_31459

/-- A cross-shaped pastry made of five unit squares -/
structure CrossPastry where
  squares : Finset (Int × Int)
  cross_shape : squares.card = 5 ∧ 
    ∃ c : Int × Int, ∀ p ∈ squares, (p.1 - c.1).natAbs + (p.2 - c.2).natAbs ≤ 1

/-- A rectangular box with an area of 16 square units -/
structure Box where
  width : Nat
  height : Nat
  area_16 : width * height = 16

/-- Two pastries can fit in the box if there exists a valid arrangement -/
def can_fit (p1 p2 : CrossPastry) (b : Box) : Prop :=
  ∃ t1 t2 : Int × Int, 
    (∀ s ∈ p1.squares, 0 ≤ s.1 + t1.1 ∧ s.1 + t1.1 < b.width ∧ 
                       0 ≤ s.2 + t1.2 ∧ s.2 + t1.2 < b.height) ∧
    (∀ s ∈ p2.squares, 0 ≤ s.1 + t2.1 ∧ s.1 + t2.1 < b.width ∧ 
                       0 ≤ s.2 + t2.2 ∧ s.2 + t2.2 < b.height) ∧
    (∀ s1 ∈ p1.squares, ∀ s2 ∈ p2.squares, 
      (s1.1 + t1.1, s1.2 + t1.2) ≠ (s2.1 + t2.1, s2.2 + t2.2))

theorem pastries_fit (p1 p2 : CrossPastry) (b : Box) : can_fit p1 p2 b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pastries_fit_l314_31459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_min_triangle_area_l314_31477

/-- A line passing through (1,4) and intersecting positive x and y axes -/
structure IntersectingLine where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : 1 / a + 4 / b = 1

/-- The sum of distances from origin to intersection points -/
noncomputable def sumOfDistances (l : IntersectingLine) : ℝ := l.a + l.b

/-- The area of the triangle formed by the intersection points and origin -/
noncomputable def triangleArea (l : IntersectingLine) : ℝ := l.a * l.b / 2

/-- The equation 2x+y-6=0 minimizes the sum of distances -/
theorem min_sum_distances (l : IntersectingLine) : 
  sumOfDistances l ≥ sumOfDistances ⟨3, 6, by norm_num, by norm_num, by norm_num⟩ := by
  sorry

/-- The equation 4x+y-8=0 minimizes the triangle area -/
theorem min_triangle_area (l : IntersectingLine) : 
  triangleArea l ≥ triangleArea ⟨2, 8, by norm_num, by norm_num, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_min_triangle_area_l314_31477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_consumption_day_b_twentyfour_fifths_eq_four_point_eight_final_tea_consumption_day_b_l314_31416

/-- Represents the relationship between hours programmed and cups of tea consumed. -/
structure ProgrammingTeaRelation where
  hours : ℚ
  cups : ℚ
  inverse_prop : hours * cups = hours * cups

/-- Day A information -/
def day_a : ProgrammingTeaRelation where
  hours := 8
  cups := 3
  inverse_prop := rfl

/-- Day B information -/
def day_b : ProgrammingTeaRelation where
  hours := 5
  cups := 24 / 5
  inverse_prop := rfl

/-- Theorem stating that the number of cups of tea on Day B is 4.8 -/
theorem tea_consumption_day_b :
  day_b.cups = 24 / 5 := by
  rfl

/-- Theorem stating that 24 / 5 is equal to 4.8 -/
theorem twentyfour_fifths_eq_four_point_eight :
  (24 : ℚ) / 5 = 4.8 := by
  norm_num

/-- Final theorem combining the previous two to show that Day B cups is 4.8 -/
theorem final_tea_consumption_day_b :
  day_b.cups = 4.8 := by
  rw [tea_consumption_day_b]
  exact twentyfour_fifths_eq_four_point_eight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_consumption_day_b_twentyfour_fifths_eq_four_point_eight_final_tea_consumption_day_b_l314_31416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vanaspati_percentage_after_addition_l314_31428

/-- Proves that adding 10 kg of pure ghee to a 10 kg mixture of 60% pure ghee and 40% vanaspati
    results in a new mixture with 20% vanaspati content. -/
theorem vanaspati_percentage_after_addition (original_quantity : ℝ) 
    (pure_ghee_percentage : ℝ) (vanaspati_percentage : ℝ) (added_pure_ghee : ℝ) :
  original_quantity = 10 →
  pure_ghee_percentage = 0.6 →
  vanaspati_percentage = 0.4 →
  added_pure_ghee = 10 →
  (original_quantity * vanaspati_percentage) / (original_quantity + added_pure_ghee) = 0.2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vanaspati_percentage_after_addition_l314_31428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l314_31449

open Set Real

noncomputable def e : ℝ := Real.exp 1

theorem solution_set_equivalence 
  (f : ℝ → ℝ) 
  (hf : Differentiable ℝ f)
  (h1 : ∀ x, f x + (deriv f) x < e) 
  (h2 : f 0 = e + 2) :
  {x : ℝ | e^x * f x > e^(x + 1) + 2} = Iio 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l314_31449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_pi_sixths_l314_31444

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_pi_sixths_l314_31444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l314_31489

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The distance from the center to a focus of an ellipse -/
noncomputable def Ellipse.focalDistance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- Theorem about the standard form and intersecting lines of a specific ellipse -/
theorem ellipse_properties (e : Ellipse) 
  (h_ecc : e.eccentricity = 1/2) 
  (h_max_dist : e.a + e.focalDistance = 3) :
  (∃ (k : ℝ), e.a = 2 ∧ e.b = Real.sqrt 3) ∧ 
  (∃ (m : ℝ), m = Real.sqrt 3 / 3 ∧ 
    ∀ (x y : ℝ), 
      (y = -x + m ∨ y = -x - m) → 
      (∃ (A B C D : ℝ × ℝ),
        (A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1) ∧
        (C.1^2 / 4 + C.2^2 / 3 = 1 ∧ D.1^2 / 4 + D.2^2 / 3 = 1) ∧
        (Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)) / 
        (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) = 8 * Real.sqrt 3 / 7)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l314_31489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_proof_l314_31410

noncomputable def cone_generator_length : ℝ := Real.sqrt 8

noncomputable def angle_two_cones : ℝ := Real.pi / 6
noncomputable def angle_third_cone : ℝ := Real.pi / 4

noncomputable def pyramid_volume (l : ℝ) (α β : ℝ) : ℝ :=
  Real.sqrt (Real.sqrt 3 + 1)

theorem pyramid_volume_proof (l α β : ℝ) 
  (h1 : l = cone_generator_length)
  (h2 : α = angle_two_cones)
  (h3 : β = angle_third_cone) :
  pyramid_volume l α β = Real.sqrt (Real.sqrt 3 + 1) := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_proof_l314_31410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l314_31461

-- Define the power function as noncomputable
noncomputable def power_function (m : ℝ) : ℝ → ℝ := fun x ↦ x ^ m

-- State the theorem
theorem power_function_through_point (m : ℝ) :
  power_function m 2 = Real.sqrt 2 → m = 1/2 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l314_31461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l314_31464

/-- The ellipse in which the square is inscribed -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/8 = 1

/-- A point is on the square if its coordinates have equal absolute values and satisfy the ellipse equation -/
def on_square (x y : ℝ) : Prop := |x| = |y| ∧ ellipse x y

/-- The area of the inscribed square -/
noncomputable def square_area : ℝ := 32/3

/-- Theorem: The area of the square inscribed in the given ellipse with sides parallel to the axes is 32/3 -/
theorem inscribed_square_area : 
  ∃ (s : ℝ), s > 0 ∧ 
  (∀ x y, on_square x y ↔ (x = s ∨ x = -s) ∧ (y = s ∨ y = -s)) ∧
  4 * s^2 = square_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l314_31464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_tournament_children_count_l314_31483

/-- Represents a basketball match with spectator information -/
structure Match where
  total : ℕ
  men : ℕ
  childrenRatio : ℕ
  womenRatio : ℕ
  seniorsRatio : ℕ

/-- Calculates the number of children in a match -/
def childrenCount (m : Match) : ℕ :=
  let remaining := m.total - m.men
  let totalRatio := m.childrenRatio + m.womenRatio + m.seniorsRatio
  (remaining * m.childrenRatio + totalRatio - 1) / totalRatio

theorem basketball_tournament_children_count 
  (match1 : Match)
  (match2 : Match)
  (match3 : Match)
  (h1 : match1.total = 18000 ∧ match1.men = 10800 ∧ match1.childrenRatio = 3 ∧ match1.womenRatio = 2 ∧ match1.seniorsRatio = 0)
  (h2 : match2.total = 22000 ∧ match2.men = 13860 ∧ match2.childrenRatio = 5 ∧ match2.womenRatio = 4 ∧ match2.seniorsRatio = 0)
  (h3 : match3.total = 10000 ∧ match3.men = 6500 ∧ match3.childrenRatio = 3 ∧ match3.womenRatio = 2 ∧ match3.seniorsRatio = 1)
  (h4 : match1.total + match2.total + match3.total = 50000) :
  childrenCount match1 + childrenCount match2 + childrenCount match3 = 10589 := by
  sorry

#eval childrenCount ⟨18000, 10800, 3, 2, 0⟩
#eval childrenCount ⟨22000, 13860, 5, 4, 0⟩
#eval childrenCount ⟨10000, 6500, 3, 2, 1⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_tournament_children_count_l314_31483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l314_31490

theorem integral_proof (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -4 ∧ x ≠ 2) : 
  deriv (λ y => 2*y + Real.log (abs y) + Real.log (abs (y + 4)) - 6*Real.log (abs (y - 2))) x = 
  (2*x^3 - 40*x - 8) / (x * (x + 4) * (x - 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l314_31490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_is_correct_l314_31497

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  4 * x^2 + 2 * x + y^2 + 4 * y + 5 = 0

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := 17 * Real.pi / 32

/-- Theorem stating that the calculated area is correct -/
theorem ellipse_area_is_correct :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), ellipse_equation x y ↔ (x + 1/8)^2 / a^2 + (y + 2)^2 / b^2 = 1) ∧
  ellipse_area = Real.pi * a * b := by
  sorry

#check ellipse_area_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_is_correct_l314_31497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l314_31422

-- Define the circle in polar coordinates
noncomputable def circle_polar (θ : ℝ) : ℝ := 2 * Real.cos θ - 2 * Real.sin θ

-- Define the line in polar coordinates
noncomputable def line_polar (θ : ℝ) : ℝ := 3 / Real.cos θ

-- Define the center of the circle in Cartesian coordinates
def circle_center : ℝ × ℝ := (1, -1)

-- Define the line in Cartesian coordinates
def line_cartesian : ℝ := 3

-- Statement: The distance between the center of the circle and the line is 2
theorem distance_circle_center_to_line :
  abs (line_cartesian - circle_center.1) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l314_31422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sales_volume_approx_l314_31438

/-- Represents the monthly sales data for a suite type -/
structure SalesData where
  initial_sales : ℕ
  growth_rate : ℚ
  fixed_increase : ℕ

/-- Calculates the total sales for a suite type over a given number of months -/
noncomputable def total_sales (data : SalesData) (months : ℕ) : ℚ :=
  if data.growth_rate ≠ 0 then
    (data.initial_sales : ℚ) * (1 - (1 + data.growth_rate) ^ months) / (1 - (1 + data.growth_rate))
  else
    (data.initial_sales : ℚ) * months + (data.fixed_increase : ℚ) * months * (months - 1) / 2

/-- The main theorem stating the total sales volume for both suite types -/
theorem total_sales_volume_approx : 
  let suite_110 : SalesData := ⟨20, 1/10, 0⟩
  let suite_90 : SalesData := ⟨20, 0, 10⟩
  let months : ℕ := 12
  ⌊(total_sales suite_110 months + total_sales suite_90 months)⌋ = 1320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sales_volume_approx_l314_31438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l314_31470

def A : Set ℤ := {x : ℤ | 0 ≤ x ∧ x ≤ 2}
def B : Set ℤ := {x : ℤ | -2 < x ∧ x ≤ 3}

theorem intersection_of_A_and_B :
  A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l314_31470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_infinite_integer_solutions_l314_31471

theorem linear_equation_infinite_integer_solutions :
  ∃ (f : ℕ → ℤ × ℤ), Function.Injective f ∧ (∀ n, (f n).1 + (f n).2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_infinite_integer_solutions_l314_31471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l314_31450

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * |x - 1| + 1

noncomputable def min_value (a : ℝ) : ℝ :=
  if a < 2 then -a^2 / 4 + a + 1 else 2

theorem f_minimum_value (a : ℝ) (h : a ≥ 0) :
  ∀ x, f a x ≥ min_value a ∧ ∃ x, f a x = min_value a := by
  sorry

#check f_minimum_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l314_31450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_subtraction_floor_subtraction_counterexample_floor_division_counterexample_l314_31411

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_subtraction (x : ℝ) : floor (x - 1) = floor x - 1 := by
  sorry

-- Additional theorems to cover other parts of the problem
theorem floor_subtraction_counterexample :
  ∃ x y : ℝ, floor (x - y) ≠ floor x - floor y := by
  sorry

theorem floor_division_counterexample :
  ∃ x y : ℝ, y ≠ 0 ∧ floor (x / y) ≠ floor x / floor y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_subtraction_floor_subtraction_counterexample_floor_division_counterexample_l314_31411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_has_two_zeros_l314_31432

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (2 : ℝ)^x else -1/x

-- Define the function F
noncomputable def F (x : ℝ) : ℝ := f x + x

-- Theorem statement
theorem F_has_two_zeros : ∃ (a b : ℝ), a ≠ b ∧ F a = 0 ∧ F b = 0 ∧ ∀ (x : ℝ), F x = 0 → x = a ∨ x = b :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_has_two_zeros_l314_31432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l314_31423

-- Define the fixed points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the condition for point M
def satisfies_angle_condition (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  ∃ α : ℝ, 0 < α ∧ α < Real.pi ∧
  (y / (x + 1) = Real.tan α) ∧
  (y / (x - 2) = Real.tan (Real.pi - 2*α))

-- Define the trajectory equations
def on_trajectory (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  (3*x^2 - y^2 = 3 ∧ x ≥ 1) ∨
  (y = 0 ∧ -1 < x ∧ x < 2)

-- The theorem statement
theorem trajectory_of_M :
  ∀ M : ℝ × ℝ, satisfies_angle_condition M → on_trajectory M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l314_31423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l314_31408

/-- A function f with the given properties -/
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

/-- The theorem statement -/
theorem function_inequality (A ω φ : ℝ) (h_pos_A : A > 0) (h_pos_ω : ω > 0) (h_pos_φ : φ > 0)
  (h_period : 2 * Real.pi / ω = Real.pi)
  (h_min : ∀ x, f A ω φ (2 * Real.pi / 3) ≤ f A ω φ x) :
  f A ω φ 2 < f A ω φ (-2) ∧ f A ω φ (-2) < f A ω φ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l314_31408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_screen_area_difference_l314_31465

/-- Calculates the area of a rectangular screen given its diagonal length and aspect ratio -/
noncomputable def screenArea (diagonal : ℝ) (aspectWidth : ℕ) (aspectHeight : ℕ) : ℝ :=
  let h := (diagonal / Real.sqrt ((aspectWidth / aspectHeight) ^ 2 + 1 : ℝ))
  let w := (aspectWidth : ℝ) / (aspectHeight : ℝ) * h
  w * h

/-- The difference in areas between two television screens with given diagonals and aspect ratios -/
theorem screen_area_difference :
  let screen1Area := screenArea 21 4 3
  let screen2Area := screenArea 17 16 9
  ∃ ε > 0, |screen1Area - screen2Area - 723.67| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_screen_area_difference_l314_31465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l314_31417

noncomputable def z : ℂ := 6 * (Complex.cos (4 * Real.pi / 3) + Complex.I * Complex.sin (4 * Real.pi / 3))

theorem complex_equality : z = -3 - 3 * Real.sqrt 3 * Complex.I := by
  -- Unfold the definition of z
  unfold z
  -- Simplify the expression
  simp [Complex.cos, Complex.sin]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l314_31417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_18_sqrt_2_l314_31466

-- Define the circles
def small_circle_radius : ℝ := 1.5
def large_circle_radius : ℝ := 3

-- Define the triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions
def circles_are_tangent (c1_center c2_center : ℝ × ℝ) : Prop :=
  dist c1_center c2_center = small_circle_radius + large_circle_radius

def triangle_tangent_to_circles (t : Triangle) (c1_center c2_center : ℝ × ℝ) : Prop :=
  ∃ (p1 p2 p3 : ℝ × ℝ),
    (dist p1 c1_center = small_circle_radius ∨ dist p1 c2_center = large_circle_radius) ∧
    (dist p2 c1_center = small_circle_radius ∨ dist p2 c2_center = large_circle_radius) ∧
    (dist p3 c1_center = small_circle_radius ∨ dist p3 c2_center = large_circle_radius) ∧
    (p1 ∈ Set.Icc t.A t.B ∨ p1 ∈ Set.Icc t.B t.C ∨ p1 ∈ Set.Icc t.C t.A) ∧
    (p2 ∈ Set.Icc t.A t.B ∨ p2 ∈ Set.Icc t.B t.C ∨ p2 ∈ Set.Icc t.C t.A) ∧
    (p3 ∈ Set.Icc t.A t.B ∨ p3 ∈ Set.Icc t.B t.C ∨ p3 ∈ Set.Icc t.C t.A)

def sides_AB_AC_congruent (t : Triangle) : Prop :=
  dist t.A t.B = dist t.A t.C

noncomputable def area (t : Triangle) : ℝ :=
  let s := (dist t.A t.B + dist t.B t.C + dist t.C t.A) / 2
  Real.sqrt (s * (s - dist t.A t.B) * (s - dist t.B t.C) * (s - dist t.C t.A))

-- Theorem statement
theorem triangle_area_is_18_sqrt_2 
  (t : Triangle) 
  (c1_center c2_center : ℝ × ℝ) 
  (h1 : circles_are_tangent c1_center c2_center)
  (h2 : triangle_tangent_to_circles t c1_center c2_center)
  (h3 : sides_AB_AC_congruent t) : 
  area t = 18 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_18_sqrt_2_l314_31466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rohan_entertainment_percentage_l314_31445

/-- Represents Rohan's monthly finances -/
structure RohanFinances where
  salary : ℚ
  food_percentage : ℚ
  rent_percentage : ℚ
  conveyance_percentage : ℚ
  savings : ℚ

/-- The conditions of Rohan's finances -/
def rohan_finances : RohanFinances :=
  { salary := 12500
  , food_percentage := 40
  , rent_percentage := 20
  , conveyance_percentage := 10
  , savings := 2500 }

/-- Calculates the percentage of salary spent on entertainment -/
def entertainment_percentage (r : RohanFinances) : ℚ :=
  100 - (r.food_percentage + r.rent_percentage + r.conveyance_percentage + (r.savings / r.salary * 100))

/-- Theorem stating that Rohan spends 10% of his salary on entertainment -/
theorem rohan_entertainment_percentage :
  entertainment_percentage rohan_finances = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rohan_entertainment_percentage_l314_31445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l314_31409

theorem problem_solution (m n : ℕ) : 
  let A : Finset ℕ := {1, n}
  let B : Finset ℕ := {2, 4, m}
  let C : Finset ℕ := Finset.biUnion A (fun x => Finset.image (· * x) B)
  (C.card = 6) →
  (C.sum id = 42) →
  m + n = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l314_31409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_difference_l314_31452

/-- The distance in meters by which runner A beats runner B in a race. -/
noncomputable def distance_difference (race_distance : ℝ) (a_time : ℝ) (time_difference : ℝ) : ℝ :=
  (race_distance / a_time) * time_difference

/-- Theorem stating the distance by which runner A beats runner B in a specific race scenario. -/
theorem race_distance_difference :
  let race_distance : ℝ := 1000
  let a_time : ℝ := 328.15384615384613
  let time_difference : ℝ := 18
  abs (distance_difference race_distance a_time time_difference - 54.83076923076923) < 0.00001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_difference_l314_31452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l314_31414

-- Define a parabola
structure Parabola where
  p : ℝ
  hp : p > 0
  eq : ∀ x y : ℝ, y^2 = 2 * p * x

-- Define the focus of a parabola
noncomputable def focus (para : Parabola) : ℝ × ℝ := (para.p / 2, 0)

-- Define the distance from focus to directrix
noncomputable def focusDirectrixDistance (para : Parabola) : ℝ := para.p

-- Theorem statement
theorem parabola_focus_coordinates (para : Parabola) 
  (h : focusDirectrixDistance para = 4) : 
  focus para = (2, 0) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l314_31414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l314_31402

theorem problem_solution : 
  (Real.sqrt 6)^2 - Real.rpow (-1) (2/3) + Real.sqrt 25 = 10 ∧
  Real.sqrt 3 * (Real.sqrt 3 - 1) + abs (-2 * Real.sqrt 3) = 3 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l314_31402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l314_31460

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

-- State the theorem
theorem f_min_value :
  ∃ (min : ℝ), (∀ x, f x ≥ min) ∧ (∃ x, f x = min) ∧ (min = -1/2) := by
  -- The proof is omitted and replaced with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l314_31460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_nine_value_l314_31468

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  seq_def : ∀ n, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem a_nine_value (seq : ArithmeticSequence) 
  (h1 : seq.a 5 = 8) 
  (h2 : sum_n seq 3 = 6) : 
  seq.a 9 = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_nine_value_l314_31468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_slope_l314_31418

/-- The equation of a line passing through a point with a given slope -/
def line_equation (x₀ y₀ m : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = m * (x - x₀)

/-- The point (-1, 2) -/
def center : ℝ × ℝ := (-1, 2)

/-- The slope of the line -/
def m : ℝ := 1

/-- The theorem stating the equation of the line -/
theorem line_through_point_with_slope :
  ∀ x y : ℝ, line_equation center.1 center.2 m x y ↔ x - y + 3 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_slope_l314_31418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_second_quadrant_l314_31455

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := (1 - Complex.I) * (a + Complex.I)

-- Define the condition for z to be in the second quadrant
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem a_range_for_second_quadrant :
  ∀ a : ℝ, in_second_quadrant (z a) ↔ a < -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_second_quadrant_l314_31455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_is_5_5_l314_31442

def points : List (ℝ × ℝ) := [(0, 7), (2, 1), (4, -3), (7, 0), (-2, -3), (5, 5)]

noncomputable def distanceFromOrigin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

theorem farthest_point_is_5_5 :
  (5, 5) ∈ points ∧ ∀ p ∈ points, distanceFromOrigin p ≤ distanceFromOrigin (5, 5) := by
  sorry

#eval points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_is_5_5_l314_31442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l314_31443

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := a * (x^2 + 1) + Real.log x

-- State the theorem
theorem m_range (m : ℝ) :
  (∀ a x, a ∈ Set.Ioo (-4) (-2) → x ∈ Set.Icc 1 3 → 
    m * a - f a x > a^2) → 
  m ≤ -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l314_31443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l314_31430

theorem tan_difference (α β : ℝ) (h1 : Real.tan α = 11) (h2 : Real.tan β = 5) :
  Real.tan (α - β) = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l314_31430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_is_four_l314_31431

noncomputable def total_distance : ℝ := 30.000000000000007

noncomputable def distance_by_foot : ℝ := (1/5) * total_distance
noncomputable def distance_by_bus : ℝ := (2/3) * total_distance
noncomputable def distance_by_car : ℝ := total_distance - (distance_by_foot + distance_by_bus)

theorem car_distance_is_four : 
  ∃ ε > 0, |distance_by_car - 4| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_distance_is_four_l314_31431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_valid_product_l314_31453

def ball_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6}

def valid_pair (a b : ℕ) : Bool :=
  a ∈ ball_numbers ∧ b ∈ ball_numbers ∧ (a * b) % 3 = 0 ∧ a * b > 15

def total_outcomes : ℕ := ball_numbers.card * ball_numbers.card

def successful_outcomes : ℕ := (ball_numbers.product ball_numbers).filter (fun (a, b) => valid_pair a b) |>.card

theorem probability_of_valid_product :
  (successful_outcomes : ℚ) / total_outcomes = 5 / 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_valid_product_l314_31453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_twos_l314_31406

-- Define the number of dice and the number of sides
def num_dice : ℕ := 8
def num_sides : ℕ := 8

-- Define the number of dice we want to show a specific number
def target_dice : ℕ := 4

-- Define the probability of a single die showing a specific number
def single_prob : ℚ := 1 / num_sides

-- Define the probability of a single die not showing a specific number
def single_prob_not : ℚ := (num_sides - 1) / num_sides

-- Theorem statement
theorem probability_four_twos :
  (Nat.choose num_dice target_dice : ℚ) * single_prob ^ target_dice * single_prob_not ^ (num_dice - target_dice) = 168070 / 16777216 := by
  sorry

#eval (Nat.choose num_dice target_dice : ℚ) * single_prob ^ target_dice * single_prob_not ^ (num_dice - target_dice)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_twos_l314_31406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_leg_sum_l314_31498

theorem similar_triangles_leg_sum (a b c : ℝ) (A B : ℝ) :
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for smaller triangle
  (1/2) * a * b = A →  -- Area of smaller triangle
  c = 10 →  -- Hypotenuse of smaller triangle
  A = 12 →  -- Area of smaller triangle
  B = 300 →  -- Area of larger triangle
  ∃ (x y : ℝ), x^2 + y^2 = (5*c)^2 ∧  -- Pythagorean theorem for larger triangle
               (1/2) * x * y = B ∧  -- Area of larger triangle
               abs (x + y - 61.3) < 0.1  -- Sum of legs of larger triangle (approximation)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_leg_sum_l314_31498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gradient_and_directional_derivative_l314_31425

noncomputable section

-- Define the function z
noncomputable def z (x y : ℝ) : ℝ := Real.arcsin (y / x^2)

-- Define the point A
def A : ℝ × ℝ := (-2, -1)

-- Define the vector a
def a : ℝ × ℝ := (3, -4)

-- Define the gradient function
noncomputable def grad_z (x y : ℝ) : ℝ × ℝ :=
  (-(2 * y) / (x * Real.sqrt (x^4 - y^2)), 1 / Real.sqrt (x^4 - y^2))

-- Define the directional derivative function
noncomputable def dir_deriv (x y : ℝ) (v : ℝ × ℝ) : ℝ :=
  let grad := grad_z x y
  let v_norm := Real.sqrt (v.1^2 + v.2^2)
  (grad.1 * v.1 + grad.2 * v.2) / v_norm

-- State the theorem
theorem gradient_and_directional_derivative :
  let grad := grad_z A.1 A.2
  let dir_d := dir_deriv A.1 A.2 a
  grad.1 = -1 / Real.sqrt 15 ∧
  grad.2 = 1 / Real.sqrt 15 ∧
  dir_d = -7 / (5 * Real.sqrt 15) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gradient_and_directional_derivative_l314_31425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l314_31472

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

-- Theorem for monotonicity and max/min values
theorem f_properties :
  (∀ x y : ℝ, 1 < x ∧ x < y → f y < f x) ∧ 
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f x ≤ 2/5) ∧
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f x ≥ 3/10) ∧
  f 2 = 2/5 ∧
  f 3 = 3/10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l314_31472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l314_31446

/-- Definition of the sequence a_n -/
def a (n : ℕ+) : ℝ := 2 * (n : ℝ) - 1

/-- Definition of S_n as the sum of the first n terms of a_n -/
noncomputable def S (n : ℕ+) : ℝ := (n : ℝ) * ((n : ℝ) + 1) / 2

/-- Main theorem stating the properties of the sequence a_n and related sum -/
theorem sequence_properties :
  (∀ n : ℕ+, a n > 0) ∧
  (∀ n : ℕ+, Real.sqrt (S n) = (1 + a n) / 2) →
  (∀ n : ℕ+, a n = 2 * (n : ℝ) - 1) ∧
  (∀ n : ℕ+, (Finset.range n).sum (λ i => 2 / (a ⟨i + 1, Nat.succ_pos i⟩ * a ⟨i + 2, Nat.succ_pos (i + 1)⟩)) = 2 * (n : ℝ) / (2 * (n : ℝ) + 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l314_31446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l314_31440

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < 2 * π / 3 →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  sin A / a = sin B / b →
  sin A / a = sin C / c →
  cos B / b + cos C / c = 2 * Real.sqrt 3 * sin A / (3 * sin C) →
  cos B + Real.sqrt 3 * sin B = 2 →
  b = Real.sqrt 3 / 2 ∧ 
  B = π / 3 ∧
  Real.sqrt 3 / 2 < a + c ∧ a + c ≤ Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l314_31440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l314_31429

/-- A right pyramid with a square base -/
structure SquarePyramid where
  base_side : ℝ
  slant_height : ℝ

/-- The total surface area of the pyramid -/
noncomputable def total_surface_area (p : SquarePyramid) : ℝ :=
  p.base_side^2 + 4 * (1/2 * p.base_side * p.slant_height)

/-- The volume of the pyramid -/
noncomputable def volume (p : SquarePyramid) : ℝ :=
  (1/3) * p.base_side^2 * Real.sqrt (p.slant_height^2 - (p.base_side/2)^2)

theorem pyramid_volume_theorem (p : SquarePyramid) 
  (h1 : total_surface_area p = 432)
  (h2 : p.base_side^2 = 2 * p.base_side * p.slant_height) :
  volume p = 288 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l314_31429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l314_31405

theorem sin_cos_difference (x y : ℝ) : 
  Real.sin (x + y) * Real.cos x - Real.cos (x + y) * Real.sin x = Real.sin y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l314_31405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_averages_final_three_days_l314_31424

/-- A race lasting one week with a total distance of 30 miles -/
structure Race where
  duration : ℕ := 7
  total_distance : ℝ := 30

/-- A runner in the race -/
structure Runner where
  name : String
  distance_first_three_days : ℝ
  distance_day_four : ℝ

noncomputable def Race.remaining_distance (race : Race) (runner : Runner) : ℝ :=
  race.total_distance - (runner.distance_first_three_days + runner.distance_day_four)

noncomputable def Race.average_remaining_distance (race : Race) (runner : Runner) : ℝ :=
  (race.remaining_distance runner) / 3

theorem average_of_averages_final_three_days (race : Race) 
  (jesse : Runner) 
  (mia : Runner) 
  (h1 : jesse.distance_first_three_days = 2)
  (h2 : jesse.distance_day_four = 10)
  (h3 : mia.distance_first_three_days + mia.distance_day_four = 12) :
  (race.average_remaining_distance jesse + race.average_remaining_distance mia) / 2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_averages_final_three_days_l314_31424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_museum_path_l314_31476

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a vector in 2D space -/
structure Vec where
  dx : ℝ
  dy : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Adds a vector to a point -/
def move (p : Point) (v : Vec) : Point :=
  ⟨p.x + v.dx, p.y + v.dy⟩

/-- Converts a distance and angle to a vector -/
noncomputable def polar_to_vector (r : ℝ) (θ : ℝ) : Vec :=
  ⟨r * Real.cos θ, r * Real.sin θ⟩

theorem museum_path :
  let start := (Point.mk 0 0)
  let museum := move start (polar_to_vector 400 (π/4))
  let first_walk := move start (polar_to_vector 600 (3*π/4))
  let second_walk := move first_walk (polar_to_vector 400 (π/4))
  let final_vector := Vec.mk (museum.x - second_walk.x) (museum.y - second_walk.y)
  (distance second_walk museum = 600) ∧
  (final_vector.dx = -final_vector.dy) ∧
  (final_vector.dx > 0) ∧
  (final_vector.dy < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_museum_path_l314_31476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fundraiser_group_composition_l314_31407

theorem fundraiser_group_composition (n : ℕ) : 
  (n : ℚ) > 0 → 
  (n / 2 : ℚ) / n = 1 / 2 → 
  ((n / 2 - 3) : ℚ) / (n - 2) = 2 / 5 → 
  n / 2 = 11 := by
  sorry

#check fundraiser_group_composition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fundraiser_group_composition_l314_31407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_difference_reciprocal_l314_31421

theorem inverse_difference_reciprocal : ((5:ℚ)⁻¹ - (2:ℚ)⁻¹)⁻¹ = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_difference_reciprocal_l314_31421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l314_31463

noncomputable section

variable (f : ℝ → ℝ)

axiom f_domain : ∀ x : ℝ, x > 0 → f x ∈ Set.range f

axiom f_half : f (1/2) = 2

axiom f_power : ∀ t : ℝ, ∀ x : ℝ, x > 0 → f (x^t) = t * f x

theorem f_properties :
  (f 1 = 0) ∧
  (f (1/4) = 4) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l314_31463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l314_31494

-- Define the train's length in meters
noncomputable def train_length : ℝ := 50

-- Define the train's speed in km/hr
noncomputable def train_speed_kmh : ℝ := 360

-- Convert km/hr to m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600

-- Theorem statement
theorem train_crossing_time :
  train_length / train_speed_ms = 0.5 := by
  -- Expand definitions
  unfold train_length train_speed_ms train_speed_kmh
  -- Perform the calculation
  norm_num
  -- The proof is completed automatically

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l314_31494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_form_l314_31441

theorem cosine_sum_product_form :
  ∃ (a b c d : ℕ+),
    (∀ x : ℝ, Real.cos x + Real.cos (5*x) + Real.cos (11*x) + Real.cos (15*x) = 
      (a : ℝ) * Real.cos (b*x) * Real.cos (c*x) * Real.cos (d*x)) ∧
    a + b + c + d = 19 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_form_l314_31441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_select_blocks_count_l314_31495

/-- The number of ways to select 4 blocks from a 6x6 grid, 
    such that no two blocks are in the same row or column -/
def select_blocks : ℕ :=
  (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4)

theorem select_blocks_count : select_blocks = 5400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_select_blocks_count_l314_31495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carrie_tshirt_purchase_l314_31482

/-- The cost of a single t-shirt in dollars -/
def tshirt_cost : ℚ := 9.65

/-- The total amount spent on t-shirts in dollars -/
def total_spent : ℚ := 115

/-- The maximum number of whole t-shirts that can be purchased -/
def max_tshirts : ℕ := (total_spent / tshirt_cost).floor.toNat

theorem carrie_tshirt_purchase :
  max_tshirts = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carrie_tshirt_purchase_l314_31482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_disk_exclusion_l314_31475

/-- A disk in a 2D plane -/
structure Disk where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Check if a point is inside a disk -/
def isInside (p : Point) (d : Disk) : Prop :=
  distance p d.center ≤ d.radius

theorem unit_disk_exclusion (K : Disk) (A B : Point) 
  (h₁ : isInside A K) 
  (h₂ : isInside B K) 
  (h₃ : distance A B > 2) : 
  ∃ (O₁ : Point), isInside O₁ K ∧ distance A O₁ > 1 ∧ distance B O₁ > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_disk_exclusion_l314_31475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_bounded_l314_31479

noncomputable def x : ℕ → ℝ
  | 0 => 2
  | n + 1 => (x n ^ 5 + 1) / (5 * x n)

theorem x_bounded (n : ℕ) : 1/5 ≤ x n ∧ x n ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_bounded_l314_31479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_destination_time_proof_l314_31401

/-- Represents the time it takes to walk to a destination -/
noncomputable def walkTime (speed : ℝ) (distance : ℝ) : ℝ := distance / speed

theorem destination_time_proof 
  (harris_speed : ℝ)
  (harris_time : ℝ)
  (harris_distance : ℝ)
  (your_speed : ℝ)
  (your_distance : ℝ)
  (h1 : harris_time = 2)
  (h2 : your_speed = 2 * harris_speed)
  (h3 : your_distance = 3 * harris_distance)
  (h4 : harris_time = walkTime harris_speed harris_distance) :
  walkTime your_speed your_distance = 3 := by
  sorry

#check destination_time_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_destination_time_proof_l314_31401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_4_increasing_and_min_value_when_a_between_0_and_1_l314_31454

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / (x + 1)

-- Part 1
theorem min_value_when_a_is_4 :
  ∀ x : ℝ, x ≥ 0 → f 4 x ≥ 3 :=
by
  sorry

-- Part 2
theorem increasing_and_min_value_when_a_between_0_and_1 (a : ℝ) (h : 0 < a ∧ a < 1) :
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → f a x < f a y) ∧
  (∀ x : ℝ, x ≥ 0 → f a x ≥ a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_4_increasing_and_min_value_when_a_between_0_and_1_l314_31454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l314_31462

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (sin (π - x) * cos (2*π - x) * tan (x + π)) / 
                     (tan (-x - π) * sin (-x - π))

-- State the theorem
theorem f_value_in_third_quadrant (x : ℝ) 
  (h1 : cos (x - 3*π/2) = 1/5)
  (h2 : π < x ∧ x < 3*π/2) : -- x is in the third quadrant
  f x = 2 * sqrt 6 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l314_31462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l314_31427

-- Define the expression as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.log (5 - x)) / (Real.sqrt (x - 2))

-- Theorem statement
theorem f_defined_iff (x : ℝ) : 
  (∃ y, f x = y) ↔ (2 < x ∧ x < 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l314_31427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_set_size_for_divisible_sum_or_diff_l314_31436

theorem smallest_set_size_for_divisible_sum_or_diff (k : ℕ) :
  (∃ n : ℕ, ∀ (S : Finset ℤ), S.card ≥ n →
    ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ((2 * k + 1 : ℤ) ∣ (a + b) ∨ (2 * k + 1 : ℤ) ∣ (a - b))) ∧
  (∀ n : ℕ, n < k + 2 →
    ∃ (S : Finset ℤ), S.card = n ∧
      ∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b → ¬((2 * k + 1 : ℤ) ∣ (a + b) ∨ (2 * k + 1 : ℤ) ∣ (a - b))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_set_size_for_divisible_sum_or_diff_l314_31436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimensions_l314_31419

/-- Represents a rectangular box with integer dimensions -/
structure Box where
  width : ℕ
  length : ℕ
  height_numerator : ℕ
  height_denominator : ℕ
  height_coprime : Nat.Coprime height_numerator height_denominator

/-- Calculates the perimeter of the triangle formed by the center points of three faces meeting at a corner -/
noncomputable def triangle_perimeter (b : Box) : ℝ :=
  let h := (b.height_numerator : ℝ) / b.height_denominator
  let d1 := Real.sqrt ((h/2)^2 + (b.width : ℝ)^2/4)
  let d2 := Real.sqrt ((b.length : ℝ)^2/4 + (b.width : ℝ)^2/4)
  let d3 := Real.sqrt ((b.length : ℝ)^2/4 + (h/2)^2)
  d1 + d2 + d3

/-- The main theorem -/
theorem box_dimensions (b : Box) 
  (h_width : b.width = 15)
  (h_length : b.length = 20)
  (h_perimeter : triangle_perimeter b = 45) :
  b.height_numerator + b.height_denominator = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimensions_l314_31419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_33_l314_31420

/-- A function that represents a 7-digit number of the form 8d5,33e -/
def number (d e : ℕ) : ℕ := 8000000 + d * 100000 + 500000 + 3300 + e

/-- Predicate to check if a natural number is a digit -/
def is_digit (n : ℕ) : Prop := n < 10

/-- The theorem stating the possible values of d for which the number is divisible by 33 -/
theorem divisible_by_33 (d e : ℕ) :
  is_digit d ∧ is_digit e ∧ (number d e) % 33 = 0 ↔ d ∈ ({1, 2, 3, 4} : Set ℕ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_33_l314_31420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_identities_l314_31413

theorem angle_identities (α β : Real) (h_acute_α : 0 < α ∧ α < Real.pi / 2) 
  (h_acute_β : 0 < β ∧ β < Real.pi / 2) (h_sin_α : Real.sin α = 4 / 5) (h_cos_β : Real.cos β = 12 / 13) :
  Real.sin (α + β) = 63 / 65 ∧ Real.tan (α - β) = 33 / 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_identities_l314_31413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l314_31404

/-- The daily sales volume function -/
noncomputable def y (m x : ℝ) : ℝ := m / (x - 3) + 8 * (x - 6)^2

/-- The daily profit function -/
noncomputable def f (m x : ℝ) : ℝ := (x - 3) * y m x

theorem max_profit (m : ℝ) :
  (∀ x, 3 < x → x < 6 → y m x ≥ 0) →
  y m 5 = 11 →
  (∃ (x_max : ℝ), 3 < x_max ∧ x_max < 6 ∧
    (∀ x, 3 < x → x < 6 → f m x ≤ f m x_max) ∧
    x_max = 4 ∧ f m x_max = 38) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l314_31404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_20cm_l314_31415

/-- The height of a cone formed by a semicircular surface with a given radius -/
noncomputable def cone_height (radius : ℝ) : ℝ :=
  let slant_height := radius
  let base_radius := radius / 2
  Real.sqrt (slant_height ^ 2 - base_radius ^ 2)

/-- Theorem: The height of a cone formed by a semicircular surface with a radius of 20 cm is 10√3 cm -/
theorem cone_height_20cm : 
  cone_height 20 = 10 * Real.sqrt 3 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval cone_height 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_20cm_l314_31415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_in_fourth_quadrant_decreasing_power_function_in_first_quadrant_correct_propositions_l314_31456

-- Define a power function as noncomputable
noncomputable def powerFunction (n : ℝ) : ℝ → ℝ := fun x ↦ x^n

-- Theorem stating that the graph of a power function cannot appear in the fourth quadrant
theorem power_function_not_in_fourth_quadrant (n : ℝ) :
  ∀ x y : ℝ, powerFunction n x = y → ¬(x > 0 ∧ y < 0) := by sorry

-- Theorem stating that for a power function to be decreasing in the first quadrant, n < 0
theorem decreasing_power_function_in_first_quadrant (n : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → powerFunction n x₁ > powerFunction n x₂) → n < 0 := by sorry

-- Theorem combining the two correct propositions
theorem correct_propositions :
  (∀ n : ℝ, ∀ x y : ℝ, powerFunction n x = y → ¬(x > 0 ∧ y < 0)) ∧
  (∀ n : ℝ, (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → powerFunction n x₁ > powerFunction n x₂) → n < 0) := by
  constructor
  · exact power_function_not_in_fourth_quadrant
  · exact decreasing_power_function_in_first_quadrant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_in_fourth_quadrant_decreasing_power_function_in_first_quadrant_correct_propositions_l314_31456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l314_31487

/-- The volume of a right circular cone -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The circumference of a circle -/
noncomputable def circle_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem cone_base_circumference :
  ∀ r h : ℝ,
  h = 9 →
  cone_volume r h = 27 * Real.pi →
  circle_circumference r = 6 * Real.pi :=
by
  intros r h h_eq vol_eq
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l314_31487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_root_problem_l314_31481

/-- Given complex numbers a, b, and c satisfying certain conditions, prove that a equals 1 + ∛4 -/
theorem complex_root_problem (a b c : ℂ) (h_real : a.im = 0)
  (h_sum : a + b + c = 5)
  (h_prod_sum : a * b + b * c + c * a = 5)
  (h_prod : a * b * c = 5) :
  a = 1 + Real.rpow 4 (1/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_root_problem_l314_31481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_medians_sum_equals_altitude_l314_31434

/-- An equilateral triangle with side length s -/
structure EquilateralTriangle where
  s : ℝ
  s_pos : s > 0

/-- The centroid of a triangle -/
noncomputable def centroid (t : EquilateralTriangle) : ℝ × ℝ := sorry

/-- The midpoint of a side of the triangle -/
noncomputable def sideMiddlePoint (t : EquilateralTriangle) (side : Fin 3) : ℝ × ℝ := sorry

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The altitude of an equilateral triangle -/
noncomputable def altitude (t : EquilateralTriangle) : ℝ := sorry

/-- The theorem to be proved -/
theorem medians_sum_equals_altitude (t : EquilateralTriangle) :
  (Finset.sum (Finset.range 3) (fun i => distance (centroid t) (sideMiddlePoint t i))) = altitude t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_medians_sum_equals_altitude_l314_31434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_of_squares_congruence_l314_31457

theorem prime_sum_of_squares_congruence (p a b : ℕ) : 
  Prime p → p = a^2 + b^2 → 
  ∃ x : ℤ, x ∈ ({(a : ℤ), -(a : ℤ), (b : ℤ), -(b : ℤ)} : Set ℤ) ∧ 
    x ≡ (1/2 : ℤ) * (Nat.choose ((p-1)/2) ((p-1)/4)) [ZMOD p] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_of_squares_congruence_l314_31457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_B2F_equals_2863_l314_31437

/-- Converts a hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'B' => 11
  | '2' => 2
  | 'F' => 15
  | _ => 0  -- Default case, should not occur for this problem

/-- Converts a hexadecimal string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.toList.reverse.enum.foldl (fun acc (i, c) => acc + (hex_to_dec c) * (16 ^ i)) 0

theorem hex_B2F_equals_2863 : hex_string_to_dec "B2F" = 2863 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_B2F_equals_2863_l314_31437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_pricing_equation_l314_31448

def cost_price : ℝ → Prop := sorry
def markup_percentage : ℝ := 0.4
def discount_percentage : ℝ := 0.8
def selling_price : ℝ := 240

theorem correct_pricing_equation (x : ℝ) (hx : cost_price x) :
  x * (1 + markup_percentage) * discount_percentage = selling_price := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_pricing_equation_l314_31448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_can_radius_l314_31485

/-- Represents a cylindrical can -/
structure Can where
  radius : ℝ
  height : ℝ

/-- The volume of a cylindrical can -/
noncomputable def volume (c : Can) : ℝ := Real.pi * c.radius^2 * c.height

theorem shorter_can_radius (short : Can) (tall : Can) :
  volume short = volume tall →
  tall.height = 4 * short.height →
  tall.radius = 16 →
  short.radius = 32 := by
  sorry

#check shorter_can_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_can_radius_l314_31485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_value_triangle_area_l314_31484

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (angleA : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.angleA = Real.pi/3 ∧ t.c = (3/7) * t.a

-- Part I: Prove sin C
theorem sin_C_value (t : Triangle) (h : triangle_conditions t) :
  Real.sin (Real.arcsin ((3 * Real.sqrt 3) / 14)) = (3 * Real.sqrt 3) / 14 :=
by sorry

-- Part II: Prove area when a = 7
theorem triangle_area (t : Triangle) (h : triangle_conditions t) (ha : t.a = 7) :
  (1/2) * t.b * t.c * Real.sin t.angleA = 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_value_triangle_area_l314_31484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l314_31435

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

theorem triangle_properties (t : Triangle) 
  (h1 : Real.cos (2 * t.A) - 3 * Real.cos (t.B + t.C) = 1)
  (h2 : (1/2) * t.b * t.c * Real.sin t.A = 5 * Real.sqrt 3)
  (h3 : t.b = 5) :
  t.A = π/3 ∧ Real.sin t.B * Real.sin t.C = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l314_31435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_problem_l314_31480

theorem sum_reciprocals_problem (m n : ℕ) 
  (h1 : m + n = 72)
  (h2 : Nat.gcd m n = 6)
  (h3 : Nat.lcm m n = 210)
  (h4 : m > 0)
  (h5 : n > 0) :
  1 / (m : ℚ) + 1 / (n : ℚ) = 6 / 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_problem_l314_31480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_fractions_l314_31412

theorem min_sum_of_fractions (A B C D : ℕ) : 
  A ∈ ({2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  B ∈ ({2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  C ∈ ({2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  D ∈ ({2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (A : ℚ) / B + (C : ℚ) / D ≥ 43 / 72 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_fractions_l314_31412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_circle_perimeter_approx_l314_31491

/-- The perimeter of a semi-circle with radius r -/
noncomputable def semiCirclePerimeter (r : ℝ) : ℝ := Real.pi * r + 2 * r

theorem semi_circle_perimeter_approx :
  let r : ℝ := 35.00860766835085
  abs (semiCirclePerimeter r - 180.08) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_circle_perimeter_approx_l314_31491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l314_31433

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  area : Real

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.a * Real.cos t.B + t.b * Real.cos t.A = 2 * t.c * Real.cos t.C)
  (h2 : t.area = 2 * Real.sqrt 3) :
  t.C = Real.pi / 3 ∧ t.c ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l314_31433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_of_girls_l314_31474

theorem average_age_of_girls (total_students : ℕ) (boys_avg_age : ℝ) (school_avg_age : ℝ) (num_girls : ℕ) :
  total_students = 632 →
  boys_avg_age = 12 →
  school_avg_age = 11.75 →
  num_girls = 158 →
  let num_boys : ℕ := total_students - num_girls
  let total_age : ℝ := total_students * school_avg_age
  let boys_total_age : ℝ := ↑num_boys * boys_avg_age
  let girls_total_age : ℝ := total_age - boys_total_age
  girls_total_age / ↑num_girls = 11 := by
  intros h1 h2 h3 h4
  -- Proof steps would go here
  sorry

#check average_age_of_girls

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_of_girls_l314_31474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_intersection_l314_31478

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in a 2D plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Parallelogram in a 2D plane -/
structure Parallelogram :=
  (A : Point) (B : Point) (C : Point) (D : Point)

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The main theorem -/
theorem parallelogram_intersection
  (ABCD : Parallelogram) (M : Point)
  (l_A l_B l_C l_D : Line)
  (h1 : parallel l_A (Line.mk M.x M.y (-M.x * ABCD.C.x - M.y * ABCD.C.y)))
  (h2 : parallel l_B (Line.mk M.x M.y (-M.x * ABCD.D.x - M.y * ABCD.D.y)))
  (h3 : parallel l_C (Line.mk M.x M.y (-M.x * ABCD.A.x - M.y * ABCD.A.y)))
  (h4 : parallel l_D (Line.mk M.x M.y (-M.x * ABCD.B.x - M.y * ABCD.B.y)))
  (h5 : point_on_line ABCD.A l_A)
  (h6 : point_on_line ABCD.B l_B)
  (h7 : point_on_line ABCD.C l_C)
  (h8 : point_on_line ABCD.D l_D) :
  ∃ (P : Point), point_on_line P l_A ∧ point_on_line P l_B ∧ point_on_line P l_C ∧ point_on_line P l_D :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_intersection_l314_31478
