import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_OAB_l383_38309

/-- The area of a parallelogram formed by two vectors in R³ -/
noncomputable def parallelogramArea (v w : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (
    (v.2.1 * w.2.2 - v.2.2 * w.2.1) ^ 2 +
    (v.1 * w.2.2 - v.2.2 * w.1) ^ 2 +
    (v.1 * w.2.1 - v.2.1 * w.1) ^ 2
  )

theorem parallelogram_area_OAB :
  let O : ℝ × ℝ × ℝ := (0, 0, 0)
  let A : ℝ × ℝ × ℝ := (1, Real.sqrt 3, 2)
  let B : ℝ × ℝ × ℝ := (Real.sqrt 3, -1, 2)
  let OA : ℝ × ℝ × ℝ := (A.1 - O.1, A.2.1 - O.2.1, A.2.2 - O.2.2)
  let OB : ℝ × ℝ × ℝ := (B.1 - O.1, B.2.1 - O.2.1, B.2.2 - O.2.2)
  parallelogramArea OA OB = 4 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_OAB_l383_38309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l383_38340

/-- Given a parabola y² = 2px (p > 0) with its focus on the circle x² + y² = 4,
    prove that the distance from the focus to the directrix is 4. -/
theorem parabola_focus_directrix_distance (p : ℝ) (h1 : p > 0) :
  (∃ (x y : ℝ), x^2 + y^2 = 4 ∧ y^2 = 2*p*x) →
  p = 4 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l383_38340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicularity_l383_38352

theorem vector_perpendicularity (a b : ℝ × ℝ) : 
  a = (5/13, 12/13) → b = (4/5, 3/5) → 
  (a.1^2 + a.2^2 = 1) → (b.1^2 + b.2^2 = 1) →
  (a.1 + b.1, a.2 + b.2).1 * (a.1 - b.1, a.2 - b.2).1 + 
  (a.1 + b.1, a.2 + b.2).2 * (a.1 - b.1, a.2 - b.2).2 = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicularity_l383_38352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_case_l383_38318

theorem right_triangle_special_case (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_special : c^2 = 2*a*b) : ∃ θ : ℝ, θ = 45 * (π / 180) ∧ (Real.sin θ = a / c ∨ Real.sin θ = b / c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_case_l383_38318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_contains_points_l383_38325

theorem ring_contains_points (points : Finset (EuclideanSpace ℝ (Fin 2))) : 
  (points.card = 650) →
  (∀ p ∈ points, dist (0 : EuclideanSpace ℝ (Fin 2)) p ≤ 16) →
  ∃ center : EuclideanSpace ℝ (Fin 2), 
    10 ≤ (points.filter (λ p => 2 ≤ dist center p ∧ dist center p ≤ 3)).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_contains_points_l383_38325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l383_38314

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := b + a * Real.sin x

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := Real.tan ((3 * a + b) * x)

theorem smallest_positive_period (a b : ℝ) (h1 : a < 0) 
  (h2 : ∀ x, f a b x ≤ -1) 
  (h3 : ∃ x, f a b x = -1)
  (h4 : ∀ x, f a b x ≥ -5) 
  (h5 : ∃ x, f a b x = -5) :
  (∀ t > 0, (∀ x, g a b (x + t) = g a b x) → t ≥ π / 9) :=
by
  sorry

#check smallest_positive_period

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l383_38314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABF_is_two_thirds_l383_38376

/-- A square with side length 2 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (2, 0) ∧ C = (2, 2) ∧ D = (0, 2))

/-- Point E on BC -/
def E : ℝ × ℝ := (2, 1)

/-- ABE is a right triangle -/
def is_right_triangle (S : Square) : Prop :=
  let ⟨A, B, _, _, _⟩ := S
  (B.1 - A.1) * (E.1 - A.1) + (B.2 - A.2) * (E.2 - A.2) = 0

/-- F is the intersection of BD and AE -/
noncomputable def F (S : Square) : ℝ × ℝ :=
  (4/3, 2/3)

/-- The area of triangle ABF -/
noncomputable def area_ABF (S : Square) : ℝ :=
  let ⟨A, B, _, _, _⟩ := S
  let F := F S
  1/2 * (B.1 - A.1) * F.2

theorem area_ABF_is_two_thirds (S : Square) :
  is_right_triangle S → area_ABF S = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABF_is_two_thirds_l383_38376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sqrt_minus_x_inverse_sqrt_l383_38338

theorem x_sqrt_minus_x_inverse_sqrt (x : ℝ) (h1 : 0 < x) (h2 : x < 1) (h3 : x + x⁻¹ = 3) :
  Real.sqrt x - (Real.sqrt x)⁻¹ = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sqrt_minus_x_inverse_sqrt_l383_38338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_backpack_profit_optimization_l383_38322

/-- Backpack profit optimization problem -/
theorem backpack_profit_optimization 
  (cost_price : ℝ) 
  (sales : ℝ → ℝ) 
  (x : ℝ) 
  (h1 : cost_price = 30)
  (h2 : ∀ x, sales x = -x + 60)
  (h3 : 30 ≤ x ∧ x ≤ 60) :
  let w := λ x ↦ (x - cost_price) * (sales x)
  (
    -- 1. Daily profit function
    w x = -x^2 + 90*x - 1800 ∧
    
    -- 2. Maximum profit
    (∃ max_profit : ℝ, max_profit = 225 ∧ 
     ∃ max_price : ℝ, max_price = 45 ∧
     ∀ y, 30 ≤ y ∧ y ≤ 60 → w y ≤ w max_price) ∧
    
    -- 3. Profit constraint
    (∃ constrained_price : ℝ, constrained_price = 40 ∧
     w constrained_price = 200 ∧
     constrained_price ≤ 48 ∧
     ∀ y, 30 ≤ y ∧ y ≤ 48 ∧ w y = 200 → y = constrained_price)
  ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_backpack_profit_optimization_l383_38322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_always_similar_l383_38346

-- Define the basic shapes
structure Rectangle where
  width : ℝ
  height : ℝ

structure Rhombus where
  side : ℝ
  angle : ℝ

structure Square where
  side : ℝ

structure RightTriangle where
  base : ℝ
  height : ℝ

-- Define similarity for shapes
def similar {α : Type*} (a b : α) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ ∀ (x y : α), true  -- Placeholder condition

-- Theorem statement
theorem squares_always_similar :
  (∀ (s1 s2 : Square), similar s1 s2) ∧
  (¬ ∀ (r1 r2 : Rectangle), similar r1 r2) ∧
  (¬ ∀ (rh1 rh2 : Rhombus), similar rh1 rh2) ∧
  (¬ ∀ (t1 t2 : RightTriangle), similar t1 t2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_always_similar_l383_38346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_length_l383_38365

/-- Given an ellipse C, a line l, and points A, B, and F, prove that |AF| = √2 -/
theorem ellipse_intersection_length (C l : Set (ℝ × ℝ)) (A B F : ℝ × ℝ) : 
  (∀ (x y : ℝ), ((x, y) : ℝ × ℝ) ∈ C ↔ x^2 / 2 + y^2 = 1) →  -- Ellipse equation
  (∀ (x y : ℝ), ((x, y) : ℝ × ℝ) ∈ l ↔ x = 2) →  -- Line equation
  F = (1, 0) →  -- Right focus coordinates
  A ∈ l →  -- A lies on l
  B ∈ C →  -- B lies on C
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ B = t • F + (1 - t) • A) →  -- B is between F and A
  (A.1 - F.1, A.2 - F.2) = 3 • (B.1 - F.1, B.2 - F.2) →  -- FA = 3FB
  Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_length_l383_38365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_l383_38330

/-- An even function that is monotonically decreasing on (0, +∞) and f(2) = 0 -/
noncomputable def f : ℝ → ℝ :=
  sorry

/-- f is an even function -/
axiom f_even : ∀ x : ℝ, f x = f (-x)

/-- f is monotonically decreasing on (0, +∞) -/
axiom f_decreasing : ∀ x y : ℝ, 0 < x → x < y → f y < f x

/-- f(2) = 0 -/
axiom f_2_eq_0 : f 2 = 0

/-- The solution set of the inequality -/
def solution_set : Set ℝ :=
  {x : ℝ | (f x + f (-x)) / (3 * x) < 0}

theorem solution_set_eq :
  solution_set = Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioi (2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_l383_38330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_domain_range_f_l383_38354

-- Define the function (marked as noncomputable due to Real.sqrt)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 - 2*x + 8)

-- Define the domain of f
def domain_f : Set ℝ := { x | -4 ≤ x ∧ x ≤ 2 }

-- Define the range of f
def range_f : Set ℝ := { y | 0 ≤ y ∧ y ≤ 3 }

-- Theorem statement
theorem intersection_domain_range_f :
  (domain_f ∩ range_f) = { x | 0 ≤ x ∧ x ≤ 2 } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_domain_range_f_l383_38354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identities_l383_38311

/-- Given a triangle ABC, prove two trigonometric identities -/
theorem triangle_trig_identities (a b c A B C : ℝ) (h_triangle : A + B + C = Real.pi) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) (h_law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C) :
  (a * Real.sin ((B - C) / 2)) / Real.sin (A / 2) + (b * Real.sin ((C - A) / 2)) / Real.sin (B / 2) + (c * Real.sin ((A - B) / 2)) / Real.sin (C / 2) = 0 ∧
  (a * Real.sin ((B - C) / 2)) / Real.cos (A / 2) + (b * Real.sin ((C - A) / 2)) / Real.cos (B / 2) + (c * Real.sin ((A - B) / 2)) / Real.cos (C / 2) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identities_l383_38311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_correctness_l383_38323

-- Define the circle and line for proposition 1
def circle1 (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4
def line1 (x y : ℝ) : Prop := x - 2*y = 0

-- Define the circle and line for proposition 2
def circle2 (x y θ : ℝ) : Prop := (x - Real.cos θ)^2 + (y - Real.sin θ)^2 = 1
def line2 (x y k : ℝ) : Prop := y = k*x

-- Define the lines for proposition 3
def line3a (x y a : ℝ) : Prop := a*x + 2*y = 0
def line3b (x y : ℝ) : Prop := x + y = 1

-- Define the regular tetrahedron and sphere for proposition 4
noncomputable def tetrahedron_edge : ℝ := Real.sqrt 2
noncomputable def sphere_volume : ℝ := (Real.sqrt 3 / 2) * Real.pi

theorem propositions_correctness :
  -- Proposition 1 is incorrect
  (∃ x y : ℝ, circle1 x y ∧ line1 x y) ∧
  (∀ x y : ℝ, circle1 x y ∧ line1 x y → (x - (-2))^2 + (y - (-1))^2 ≠ 1) ∧
  -- Proposition 2 is correct
  (∀ k θ : ℝ, ∃ x y : ℝ, circle2 x y θ ∧ line2 x y k) ∧
  -- Proposition 3 is incorrect
  (∃ a : ℝ, a ≠ 2 ∧ (∀ x y : ℝ, line3a x y a ↔ line3b x y)) ∧
  -- Proposition 4 is correct
  (sphere_volume = (4/3) * Real.pi * (Real.sqrt 3 / 2)^3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_correctness_l383_38323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_C_R_N_l383_38382

-- Define the set M
def M : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the complement of N in real numbers
def C_R_N : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem intersection_M_C_R_N : M ∩ C_R_N = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_C_R_N_l383_38382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l383_38396

noncomputable section

open Real

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a / sin A = b / sin B →
  a / sin A = c / sin C →
  a^2 = b^2 - c^2 + sqrt 3 * a * c →
  B = π / 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l383_38396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thief_distance_theorem_l383_38360

/-- The distance run by the thief before being overtaken by the policeman -/
noncomputable def distance_run_by_thief (initial_distance : ℝ) (thief_speed : ℝ) (policeman_speed : ℝ) : ℝ :=
  let thief_speed_ms := thief_speed * 1000 / 3600
  let policeman_speed_ms := policeman_speed * 1000 / 3600
  let relative_speed := policeman_speed_ms - thief_speed_ms
  let time := initial_distance / relative_speed
  thief_speed_ms * time

/-- Theorem stating the distance run by the thief before being overtaken -/
theorem thief_distance_theorem :
  ∃ ε > 0, |distance_run_by_thief 250 12 15 - 990.47| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_thief_distance_theorem_l383_38360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_relationship_l383_38347

/-- Represents the total distance to school -/
def total_distance : ℝ := sorry

/-- Represents the distance already traveled -/
def distance_traveled : ℝ := sorry

/-- Represents the remaining distance -/
def remaining_distance : ℝ := sorry

/-- The total distance is the sum of distance traveled and remaining distance -/
axiom distance_sum : total_distance = distance_traveled + remaining_distance

/-- Two quantities are related if they have a mathematical relationship -/
def are_related (a b : ℝ) : Prop := ∃ f : ℝ → ℝ → ℝ, f a b = total_distance

/-- Theorem stating that distance traveled and remaining distance are related -/
theorem distance_relationship : are_related distance_traveled remaining_distance := by
  sorry

#check distance_relationship

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_relationship_l383_38347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_intersection_equilateral_triangles_l383_38378

/-- The area of intersection of two equilateral triangles inscribed in a circle -/
theorem area_of_intersection_equilateral_triangles (R : ℝ) (R_pos : R > 0) :
  ∃ (area : ℝ), area = (Real.sqrt 3 * R^2) / 2 := by
  -- Define local variables
  let circle_radius := R
  let triangles_inscribed := 2
  let side_divisions := 3
  let area_of_intersection := (Real.sqrt 3 * R^2) / 2

  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_intersection_equilateral_triangles_l383_38378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_l383_38313

/-- Calculates the distance traveled given speed and time -/
noncomputable def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Calculates the speed given distance and time -/
noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) 
  (speed_increase : ℝ) :
  marguerite_distance = 150 →
  marguerite_time = 3 →
  sam_time = 4 →
  speed_increase = 1.2 →
  distance (speed marguerite_distance marguerite_time * speed_increase) sam_time = 240 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_l383_38313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percent_before_sale_is_25_percent_l383_38300

/-- Represents the pricing and profit details of an item sold by a shopkeeper -/
structure ItemSale where
  sale_price : ℚ
  discount_percent : ℚ
  sale_gain_percent : ℚ

/-- Calculates the gain percentage before a clearance sale -/
def gain_percent_before_sale (item : ItemSale) : ℚ :=
  let marked_price := item.sale_price / (1 - item.discount_percent / 100)
  let cost_price := item.sale_price / (1 + item.sale_gain_percent / 100)
  (marked_price - cost_price) / cost_price * 100

/-- Theorem stating that under given conditions, the gain percentage before the clearance sale is 25% -/
theorem gain_percent_before_sale_is_25_percent (item : ItemSale) 
  (h1 : item.sale_price = 30)
  (h2 : item.discount_percent = 10)
  (h3 : item.sale_gain_percent = (25/2)) :
  gain_percent_before_sale item = 25 := by
  sorry

#eval gain_percent_before_sale { sale_price := 30, discount_percent := 10, sale_gain_percent := 25/2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percent_before_sale_is_25_percent_l383_38300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_notes_problem_l383_38383

theorem algebra_notes_problem (total_sheets : ℕ) (total_pages : ℕ) (borrowed_sheets : ℕ) :
  total_sheets = 30 →
  total_pages = 60 →
  borrowed_sheets = 15 →
  let remaining_sheets := total_sheets - borrowed_sheets
  let remaining_pages := total_pages - 2 * borrowed_sheets
  let sum_remaining_pages := 
    (Finset.range remaining_sheets).sum (λ i => 2 * i + 1) +
    (Finset.range remaining_sheets).sum (λ i => total_pages - 2 * i)
  (sum_remaining_pages : ℚ) / remaining_pages = 25 := by
  sorry

#check algebra_notes_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_notes_problem_l383_38383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pasture_rent_calculation_l383_38332

/-- Calculates the total rent of a pasture given the usage details of three renters --/
theorem pasture_rent_calculation 
  (oxen_A : ℕ) (months_A : ℕ) 
  (oxen_B : ℕ) (months_B : ℕ) 
  (oxen_C : ℕ) (months_C : ℕ) 
  (share_C : ℚ) : 
  oxen_A = 10 → months_A = 7 →
  oxen_B = 12 → months_B = 5 →
  oxen_C = 15 → months_C = 3 →
  share_C = 72 →
  (oxen_A * months_A + oxen_B * months_B + oxen_C * months_C : ℚ) * 
  (share_C / (oxen_C * months_C : ℚ)) = 280 := by
  sorry

#eval (10 * 7 + 12 * 5 + 15 * 3 : ℕ)
#eval (72 : ℚ) / 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pasture_rent_calculation_l383_38332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z₁_div_z₂_l383_38358

def z₁ : ℂ := 1 + 3 * Complex.I
def z₂ : ℂ := 3 + Complex.I

theorem imaginary_part_of_z₁_div_z₂ : (z₁ / z₂).im = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z₁_div_z₂_l383_38358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l383_38362

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (x, -4)
  parallel a b → x = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l383_38362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l383_38337

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 8 * Real.cos θ

-- Define the line in parametric form
def line (t : ℝ) : ℝ × ℝ := (t + 2, t)

-- State the theorem
theorem curve_and_line_intersection :
  -- The Cartesian equation of curve C
  (∀ x y : ℝ, x^2 + y^2 = 8*x ↔ ∃ θ : ℝ, x = curve_C θ * Real.cos θ ∧ y = curve_C θ * Real.sin θ) ∧
  -- The length of the intersection segment AB
  (∃ A B : ℝ × ℝ, 
    (∃ t : ℝ, line t = A) ∧
    (∃ t : ℝ, line t = B) ∧
    A.1^2 + A.2^2 = 8*A.1 ∧
    B.1^2 + B.2^2 = 8*B.1 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 28) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l383_38337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l383_38333

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = Set.Iic (-4) ∪ Set.Ici 2 :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = Set.Ioi (-3/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l383_38333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_books_at_least_1_43_l383_38329

/-- Represents the number of books borrowed by a student -/
def BooksBorrowed := ℕ

/-- Represents a class of students and their borrowed books -/
structure ClassInfo where
  total_students : ℕ
  zero_books : ℕ
  one_book : ℕ
  two_books : ℕ
  at_least_three_books : ℕ
  max_books : ℕ

/-- Calculate the minimum total number of books borrowed -/
def min_total_books (c : ClassInfo) : ℕ :=
  c.one_book + 2 * c.two_books + 3 * c.at_least_three_books

/-- Calculate the average number of books borrowed per student -/
def average_books (c : ClassInfo) : ℚ :=
  (min_total_books c : ℚ) / c.total_students

/-- Theorem stating that the average number of books borrowed is at least 1.43 -/
theorem average_books_at_least_1_43 (c : ClassInfo) 
  (h1 : c.total_students = 30)
  (h2 : c.zero_books = 5)
  (h3 : c.one_book = 12)
  (h4 : c.two_books = 8)
  (h5 : c.at_least_three_books = c.total_students - (c.zero_books + c.one_book + c.two_books))
  (h6 : c.max_books = 20) :
  average_books c ≥ 1.43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_books_at_least_1_43_l383_38329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_proof_l383_38366

noncomputable section

open Real

theorem triangle_abc_proof (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Given conditions
  sqrt 3 * sin (2 * B) = 2 * (sin B) ^ 2 →
  a = 4 →
  b = 2 * sqrt 7 →
  -- Conclusions to prove
  B = π / 3 ∧ c = 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_proof_l383_38366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_divides_power_minus_one_l383_38397

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ, n > 0 → (n ∣ (2^n - 1) ↔ n = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_divides_power_minus_one_l383_38397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_routes_l383_38317

/-- Represents a point in a 2D grid --/
structure Point where
  x : Int
  y : Int

/-- Calculates the number of shortest paths between two points on a 2D grid --/
def shortestPaths (start finish : Point) : Nat :=
  Nat.choose (Int.natAbs (finish.x - start.x) + Int.natAbs (finish.y - start.y)) (Int.natAbs (finish.x - start.x))

/-- Samantha's home location relative to the western library entrance --/
def home : Point := { x := -3, y := 3 }

/-- Western library entrance location --/
def westEntrance : Point := { x := 0, y := 0 }

/-- Eastern library entrance location --/
def eastEntrance : Point := { x := 0, y := 0 }

/-- Samantha's school location relative to the eastern library entrance --/
def school : Point := { x := 3, y := -3 }

/-- Number of shortcut paths through the library --/
def libraryShortcuts : Nat := 2

/-- The total number of shortest routes Samantha can take --/
def totalRoutes : Nat :=
  shortestPaths home westEntrance * libraryShortcuts * shortestPaths eastEntrance school

theorem samantha_routes : totalRoutes = 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_routes_l383_38317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_theorem_l383_38301

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a cylinder with radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- The lateral surface area of a cylinder -/
noncomputable def cylinderLateralArea (c : Cylinder) : ℝ := 2 * Real.pi * c.radius * c.height

theorem rectangle_area_theorem (c1 c2 : Cylinder) (r : Rectangle) :
  cylinderVolume c1 / cylinderVolume c2 = 5 / 8 →
  cylinderLateralArea c1 = cylinderLateralArea c2 →
  cylinderLateralArea c1 = r.length * r.width →
  (r.length + 6) * (r.width + 6) - r.length * r.width = 114 →
  r.length * r.width = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_theorem_l383_38301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_heads_before_three_tails_l383_38391

/-- The probability of getting heads in a single fair coin flip -/
noncomputable def p_heads : ℝ := 1/2

/-- The probability of getting n consecutive heads -/
noncomputable def p_n_heads (n : ℕ) : ℝ := p_heads ^ n

/-- The probability of getting 1 to 3 heads in a block -/
noncomputable def p_block : ℝ := p_n_heads 1 + p_n_heads 2 + p_n_heads 3

/-- The probability of encountering a run of 4 heads before a run of 3 tails
    when repeatedly flipping a fair coin -/
noncomputable def q : ℝ := (p_n_heads 4) / (1 - p_block * p_heads)

theorem four_heads_before_three_tails : q = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_heads_before_three_tails_l383_38391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_eccentricity_l383_38312

/-- Given an ellipse and a hyperbola with coinciding foci, prove that the semi-major axis of the ellipse is greater than that of the hyperbola, and the product of their eccentricities is greater than 1. -/
theorem ellipse_hyperbola_eccentricity 
  (m n : ℝ) 
  (hm : m > 1) 
  (hn : n > 0) 
  (e₁ e₂ : ℝ) 
  (h_ellipse : ∀ x y : ℝ, x^2 / m^2 + y^2 = 1 → (x, y) ∈ Set.range (λ t : ℝ × ℝ ↦ (t.1, t.2)))
  (h_hyperbola : ∀ x y : ℝ, x^2 / n^2 - y^2 = 1 → (x, y) ∈ Set.range (λ t : ℝ × ℝ ↦ (t.1, t.2)))
  (h_foci : ∃ c : ℝ, c^2 = m^2 - 1 ∧ c^2 = n^2 + 1)
  (h_e₁ : e₁ = Real.sqrt (1 - 1/m^2))
  (h_e₂ : e₂ = Real.sqrt (1 + 1/n^2)) :
  m > n ∧ e₁ * e₂ > 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_eccentricity_l383_38312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_perimeter_ratio_l383_38356

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 is √3/2 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let s : ℝ := 6
  let area : ℝ := (Real.sqrt 3 / 4) * s^2
  let perimeter : ℝ := 3 * s
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_perimeter_ratio_l383_38356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_a_range_l383_38348

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x

-- Theorem for part (1)
theorem f_monotone_increasing (a : ℝ) (h : a > 0) :
  StrictMono (f a) := by sorry

-- Theorem for part (2)
theorem a_range (a : ℝ) (h : ∀ x > 1, f a x < x^2) :
  a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_a_range_l383_38348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_vector_dot_product_bound_l383_38357

-- Define the circle
def Circle (O : ℝ × ℝ) (l : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = l^2}

-- Define the dot product of 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the vector from one point to another
def vector (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

-- State the theorem
theorem circle_vector_dot_product_bound 
  (O : ℝ × ℝ) (l : ℝ) (A B C P : ℝ × ℝ) 
  (h_circle : A ∈ Circle O l ∧ B ∈ Circle O l ∧ C ∈ Circle O l)
  (h_diameter : vector O A = (-vector O B))
  (h_P_in_circle : (P.1 - O.1)^2 + (P.2 - O.2)^2 ≤ l^2) :
  -4/3 * l^2 ≤ 
    dot_product (vector P A) (vector P B) + 
    dot_product (vector P B) (vector P C) + 
    dot_product (vector P C) (vector P A) ∧
  dot_product (vector P A) (vector P B) + 
  dot_product (vector P B) (vector P C) + 
  dot_product (vector P C) (vector P A) ≤ 4 * l^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_vector_dot_product_bound_l383_38357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_running_time_problem_l383_38336

/-- Proves that the running time is 20 minutes given the conditions of the problem -/
theorem running_time_problem (running_speed walking_speed total_distance : ℝ) 
  (walking_time : ℝ) (h1 : running_speed = 6) (h2 : walking_speed = 2) 
  (h3 : walking_time = 0.5) (h4 : total_distance = 3) : 
  (total_distance - walking_speed * walking_time) / running_speed * 60 = 20 := by
  sorry

#check running_time_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_running_time_problem_l383_38336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l383_38315

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 1 - 2 * (Real.sin (x + π / 4))^2

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∃ p > 0, ∀ x, f (x + p) = f x) ∧  -- f is periodic
  (∀ q > 0, (∀ x, f (x + q) = f x) → q ≥ π) -- π is the smallest positive period
  := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l383_38315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_over_four_l383_38334

theorem cos_alpha_plus_pi_over_four 
  (α β : ℝ) 
  (h1 : α ∈ Set.Ioo (3 * Real.pi / 4) Real.pi) 
  (h2 : β ∈ Set.Ioo (3 * Real.pi / 4) Real.pi) 
  (h3 : Real.sin (α + β) = -3/5) 
  (h4 : Real.sin (β - Real.pi/4) = 12/13) : 
  Real.cos (α + Real.pi/4) = -56/65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_over_four_l383_38334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l383_38304

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is on the ellipse -/
def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  abs ((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2)

/-- The main theorem -/
theorem max_triangle_area (e : Ellipse) (A : Point) :
  on_ellipse e A →
  A.x = 2 ∧ A.y = Real.sqrt 2 →
  (e.a^2 - e.b^2).sqrt = 2 →
  (∀ B : Point, on_ellipse e B → triangle_area (Point.mk 0 0) A B ≤ 2 * Real.sqrt 2) ∧
  (∃ B : Point, on_ellipse e B ∧ triangle_area (Point.mk 0 0) A B = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l383_38304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radiansToDegrees_negative_eight_pi_thirds_l383_38303

/-- Conversion factor from radians to degrees -/
noncomputable def radiansToDegreesConversion : ℝ := 180 / Real.pi

/-- The angle in radians -/
noncomputable def angleInRadians : ℝ := -8 * Real.pi / 3

/-- Theorem: Converting -8π/3 radians to degrees yields -480° -/
theorem radiansToDegrees_negative_eight_pi_thirds :
  angleInRadians * radiansToDegreesConversion = -480 := by
  -- Expand the definitions
  unfold angleInRadians radiansToDegreesConversion
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radiansToDegrees_negative_eight_pi_thirds_l383_38303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_a_greater_than_half_l383_38384

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x + a^(2*x) - 2*a

theorem root_implies_a_greater_than_half (a : ℝ) :
  (∃ x ∈ Set.Ioo 0 1, f a x = 0) → a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_a_greater_than_half_l383_38384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jen_ate_eleven_suckers_l383_38355

-- Define the number of suckers at each stage
def total_suckers : ℕ := sorry
def jen_suckers : ℕ := sorry
def molly_suckers : ℕ := sorry
def harmony_suckers : ℕ := sorry
def taylor_suckers : ℕ := sorry
def callie_suckers : ℕ := sorry

-- Define the relationships based on the problem conditions
axiom jen_ate_half : jen_suckers = total_suckers / 2
axiom molly_received : molly_suckers = total_suckers - jen_suckers
axiom harmony_received : harmony_suckers = molly_suckers - 2
axiom taylor_received : taylor_suckers = harmony_suckers - 3
axiom callie_received : callie_suckers = taylor_suckers - 1
axiom callie_final : callie_suckers = 5

-- Theorem to prove
theorem jen_ate_eleven_suckers : jen_suckers = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jen_ate_eleven_suckers_l383_38355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_odd_and_increasing_l383_38326

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 10) - Real.exp (-x * Real.log 10)
def g (x : ℝ) : ℝ := x^3

-- Theorem statement
theorem f_and_g_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, g (-x) = -g x) ∧ 
  (∀ x y : ℝ, x < y → g x < g y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_odd_and_increasing_l383_38326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangement_l383_38381

theorem student_arrangement (n : ℕ) (h : n = 6) : 
  (Nat.factorial n) / 2 = 240 := by
  rw [h]
  norm_num
  ring
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangement_l383_38381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l383_38359

noncomputable section

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 2)

-- Define the circle equation
def on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y = 0

-- Define the triangle area function
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3))

-- State the theorem
theorem min_triangle_area :
  ∀ C : ℝ × ℝ, on_circle C.1 C.2 →
  ∀ area : ℝ, area = triangle_area A B C →
  area ≥ 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l383_38359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_properties_l383_38389

/-- Rectangle ABCD with area 2 -/
structure Rectangle (A B C D : ℝ × ℝ) where
  area : ℝ
  is_rectangle : Bool
  area_eq_two : area = 2

/-- Point P on side CD -/
def P (C D : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Point Q on side AB, tangency point of incircle of triangle PAB -/
def Q (A B : ℝ × ℝ) : ℝ × ℝ := sorry

/-- PA * PB is minimized -/
def is_min_PA_PB (P A B : ℝ × ℝ) : Prop := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem rectangle_properties (A B C D : ℝ × ℝ) (rect : Rectangle A B C D) 
  (h_min : is_min_PA_PB (P C D) A B) :
  (distance A B ≥ 2 * distance B C) ∧ 
  (distance A (Q A B) * distance B (Q A B) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_properties_l383_38389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_calculation_l383_38305

/-- Calculates the speed of a car in km/h given the tire's rotation speed and circumference -/
noncomputable def car_speed (revolutions_per_minute : ℝ) (tire_circumference : ℝ) : ℝ :=
  (revolutions_per_minute * tire_circumference * 60) / 1000

/-- Theorem: A car with a tire rotating at 400 revolutions per minute and a circumference of 6 meters travels at 144 km/h -/
theorem car_speed_calculation :
  car_speed 400 6 = 144 := by
  -- Unfold the definition of car_speed
  unfold car_speed
  -- Perform the calculation
  simp [mul_assoc, mul_comm, mul_div_assoc]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_calculation_l383_38305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l383_38327

-- Define the sequence
def our_sequence : ℕ → ℕ
| 0 => 2
| 1 => 3
| (n + 2) => (our_sequence n % 10) * (our_sequence (n + 1) % 10) % 10

-- Define the set of digits that never appear
def never_appear : Set ℕ := {0, 5, 7, 9}

-- Theorem statement
theorem sequence_properties :
  (∀ n : ℕ, our_sequence n ∉ never_appear) ∧
  (our_sequence 999 = 2) := by
  sorry

#eval our_sequence 999  -- This will evaluate the 1000th digit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l383_38327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_theorem_l383_38394

open Nat

/-- Number of positive divisors of k -/
def d (k : ℕ) : ℕ := sorry

/-- Sum of positive divisors of k -/
def σ (k : ℕ) : ℕ := sorry

theorem unique_function_theorem (f : ℕ → ℕ) :
  (∀ n : ℕ, f (d (n + 1)) = d (f n + 1)) ∧
  (∀ n : ℕ, f (σ (n + 1)) = σ (f n + 1)) →
  f = id :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_theorem_l383_38394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_cost_per_meter_l383_38328

/-- The cost price per meter of cloth, given the total cost and length. -/
noncomputable def cost_price_per_meter (total_cost : ℝ) (total_length : ℝ) : ℝ :=
  total_cost / total_length

/-- Theorem: The cost price per meter of cloth is $43, given the conditions. -/
theorem cloth_cost_per_meter :
  let total_cost : ℝ := 397.75
  let total_length : ℝ := 9.25
  cost_price_per_meter total_cost total_length = 43 := by
  -- Unfold the definition of cost_price_per_meter
  unfold cost_price_per_meter
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_cost_per_meter_l383_38328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_less_than_four_not_three_l383_38310

theorem unique_number_less_than_four_not_three : ∃! x : ℕ, x ∈ ({2, 3, 4, 5} : Set ℕ) ∧ x < 4 ∧ x ≠ 3 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_less_than_four_not_three_l383_38310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l383_38341

-- Define the complex number z
variable (z : ℂ)

-- State the theorem
theorem magnitude_of_z (h : z * (Complex.ofReal (Real.sqrt 2) + Complex.I) = Complex.I * 3) :
  Complex.abs z = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l383_38341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l383_38385

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) + 4

theorem problem_solution (h1 : f (Real.log (Real.log 10 / Real.log 2)) = 5)
  (h2 : Real.log (Real.log 10 / Real.log 2) + Real.log (Real.log 2 / Real.log 2) = 0) :
  f (Real.log (Real.log 2 / Real.log 2)) = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l383_38385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_required_rent_is_123_53_l383_38372

/-- Calculates the required monthly rent for a property investment --/
noncomputable def calculateRequiredRent (purchasePrice : ℝ) (annualReturnRate : ℝ) (maintenanceRate : ℝ) (annualTaxes : ℝ) : ℝ :=
  let annualReturn := purchasePrice * annualReturnRate
  let totalAnnualEarnings := annualReturn + annualTaxes
  let monthlyEarningsNeeded := totalAnnualEarnings / 12
  monthlyEarningsNeeded / (1 - maintenanceRate)

/-- Theorem stating the required monthly rent for the given problem --/
theorem required_rent_is_123_53 :
  let purchasePrice : ℝ := 15000
  let annualReturnRate : ℝ := 0.06
  let maintenanceRate : ℝ := 0.15
  let annualTaxes : ℝ := 360
  abs (calculateRequiredRent purchasePrice annualReturnRate maintenanceRate annualTaxes - 123.53) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_required_rent_is_123_53_l383_38372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducible_fraction_l383_38316

theorem irreducible_fraction (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducible_fraction_l383_38316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PQT_twice_OPQR_l383_38335

-- Define the square OPQR
noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def Q : ℝ × ℝ := (3, 3)
noncomputable def P : ℝ × ℝ := (3, 0)
noncomputable def R : ℝ × ℝ := (0, 3)

-- Define point T
noncomputable def T : ℝ × ℝ := (3, 12)

-- Function to calculate area of a square given side length
noncomputable def square_area (side : ℝ) : ℝ := side * side

-- Function to calculate area of a triangle given base and height
noncomputable def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

-- Theorem statement
theorem area_PQT_twice_OPQR :
  triangle_area (Q.1 - P.1) (T.2 - P.2) = 2 * square_area (Q.1 - O.1) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PQT_twice_OPQR_l383_38335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_squared_exceeds_critical_value_r_relationship_r_equals_six_l383_38375

/-- Represents the survey data --/
structure SurveyData where
  case_not_good : ℕ
  case_good : ℕ
  control_not_good : ℕ
  control_good : ℕ

/-- Calculates K² statistic --/
noncomputable def calculate_k_squared (data : SurveyData) : ℝ :=
  let n := (data.case_not_good + data.case_good + data.control_not_good + data.control_good : ℝ)
  let a := (data.case_not_good : ℝ)
  let b := (data.case_good : ℝ)
  let c := (data.control_not_good : ℝ)
  let d := (data.control_good : ℝ)
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The survey data from the problem --/
def survey_data : SurveyData := {
  case_not_good := 40,
  case_good := 60,
  control_not_good := 10,
  control_good := 90
}

/-- Theorem stating that K² > 6.635 for the given survey data --/
theorem k_squared_exceeds_critical_value : calculate_k_squared survey_data > 6.635 := by sorry

/-- Theorem proving the relationship for R --/
theorem r_relationship (P_A_B P_A_not_B : ℝ) :
  let P_not_A_B := 1 - P_A_B
  let P_not_A_not_B := 1 - P_A_not_B
  let R := (P_A_B / P_not_A_B) * (P_not_A_not_B / P_A_not_B)
  R = (P_A_B / P_not_A_B) * (P_not_A_not_B / P_A_not_B) := by sorry

/-- Theorem stating that R = 6 for the given survey data --/
theorem r_equals_six : 
  let P_A_B := 40 / 100
  let P_A_not_B := 10 / 100
  let P_not_A_B := 1 - P_A_B
  let P_not_A_not_B := 1 - P_A_not_B
  let R := (P_A_B / P_not_A_B) * (P_not_A_not_B / P_A_not_B)
  R = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_squared_exceeds_critical_value_r_relationship_r_equals_six_l383_38375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l383_38361

/-- Given a line and an ellipse with specific properties, prove the equation of the ellipse. -/
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let line := fun x : ℝ ↦ x + Real.sqrt 2
  let ellipse := fun (x y : ℝ) ↦ x^2 / a^2 + y^2 / b^2 = 1
  let O := (0 : ℝ × ℝ)
  ∃ (M N : ℝ × ℝ),
    ellipse M.1 M.2 ∧
    ellipse N.1 N.2 ∧
    M.2 = line M.1 ∧
    N.2 = line N.1 ∧
    (M.1 * N.1 + M.2 * N.2 = 0) ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = 6 →
    a^2 = 4 + 2 * Real.sqrt 2 ∧ b^2 = 4 - 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l383_38361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_iff_divides_equally_l383_38373

-- Define the parallel lines
structure ParallelLines where
  line1 : Set (ℝ × ℝ)
  line2 : Set (ℝ × ℝ)
  parallel : ∀ (x y : ℝ × ℝ), x ∈ line1 ∧ y ∈ line2 → (x.1 - y.1) * (x.2 - y.2) = 0

-- Define a point P
def P : ℝ × ℝ := sorry

-- Define a line through P
def lineThroughP (p : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define a segment on one of the parallel lines
def segmentOnLine (l : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

-- Define the property of dividing a segment into n equal parts
def divideIntoEqualParts (l : Set (ℝ × ℝ)) (s : Set (ℝ × ℝ)) (n : ℕ) : Prop := sorry

-- Define a custom parallelism relation for sets of points
def SetParallel (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ × ℝ), x ∈ s1 ∧ y ∈ s2 → (x.1 - y.1) * (x.2 - y.2) = 0

-- The main theorem
theorem line_parallel_iff_divides_equally (pl : ParallelLines) (n : ℕ) :
  ∀ l : Set (ℝ × ℝ), l = lineThroughP P →
    (SetParallel l pl.line1 ↔ divideIntoEqualParts l (segmentOnLine pl.line1) n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_iff_divides_equally_l383_38373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minimum_at_four_l383_38374

def a (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 8
  else a (n - 1) + (n - 1)

theorem a_minimum_at_four :
  ∀ n : ℕ, n ≥ 1 → a n / n ≥ a 4 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minimum_at_four_l383_38374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_trihedral_angle_sum_l383_38306

/-- Predicate to check if a point is the center of a cylinder. -/
def IsCylinderCenter (O : Point) : Prop := sorry

/-- Predicate to check if two points form a diameter of a cylinder base. -/
def IsDiameter (A B : Point) : Prop := sorry

/-- Predicate to check if a point is on the circle of a cylinder base. -/
def IsOnCircle (C : Point) : Prop := sorry

/-- Function to calculate the sum of dihedral angles of a trihedral angle. -/
def DihedralAngleSum (O A B C : Point) : ℝ := sorry

/-- Given a cylinder with center O, diameter AB of one base, and point C on the circle of the other base,
    the sum of the dihedral angles of the trihedral angle OABC with vertex at O is equal to 2π. -/
theorem cylinder_trihedral_angle_sum (O A B C : Point) 
  (hO : IsCylinderCenter O) (hAB : IsDiameter A B) (hC : IsOnCircle C) : 
  DihedralAngleSum O A B C = 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_trihedral_angle_sum_l383_38306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l383_38331

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = (e^x + ae^(-x))sin(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (Real.exp x + a * Real.exp (-x)) * Real.sin x

theorem odd_function_implies_a_equals_one :
  ∃ a : ℝ, IsOdd (f a) → a = 1 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l383_38331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_zero_l383_38387

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) * Real.exp x

-- State the theorem
theorem derivative_f_at_zero : 
  deriv f 0 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_zero_l383_38387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_incorrect_l383_38321

-- Define propositions
def prop1 : Prop := ∀ p q : Prop, ¬(p ∧ q) → (¬p ∧ ¬q)

def prop2 : Prop := 
  (∀ a b : ℝ, a > b → (2 : ℝ)^a > (2 : ℝ)^b - 1) ↔ 
  (∀ a b : ℝ, a > b ∧ (2 : ℝ)^a ≤ (2 : ℝ)^b - 1 → False)

def prop3 : Prop := 
  ∀ A B : ℝ, 
    (A > B ↔ Real.sin A > Real.sin B)

-- Theorem statement
theorem exactly_one_incorrect : 
  (¬prop1 ∧ prop2 ∧ prop3) ∨ 
  (prop1 ∧ ¬prop2 ∧ prop3) ∨ 
  (prop1 ∧ prop2 ∧ ¬prop3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_incorrect_l383_38321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cubes_for_views_l383_38308

/-- Represents a 3D configuration of unit cubes -/
structure CubeConfiguration where
  cubes : Finset (ℕ × ℕ × ℕ)

/-- Checks if the configuration satisfies the sharing face condition -/
def satisfies_sharing_condition (config : CubeConfiguration) : Prop :=
  ∀ c ∈ config.cubes, ∃ c' ∈ config.cubes, c ≠ c' ∧ (
    (c.1 = c'.1 ∧ c.2 = c'.2 ∧ (c.2.1 = c'.2.1 + 1 ∨ c.2.1 = c'.2.1 - 1)) ∨
    (c.1 = c'.1 ∧ c.2.2 = c'.2.2 ∧ (c.2.1 = c'.2.1 + 1 ∨ c.2.1 = c'.2.1 - 1)) ∨
    (c.2.1 = c'.2.1 ∧ c.2.2 = c'.2.2 ∧ (c.1 = c'.1 + 1 ∨ c.1 = c'.1 - 1))
  )

/-- Checks if the configuration matches the front view -/
def matches_front_view (config : CubeConfiguration) : Prop :=
  (∃ x y, (x, y, 0) ∈ config.cubes ∧ (x + 1, y, 0) ∈ config.cubes ∧ (x + 2, y, 0) ∈ config.cubes) ∧
  (∃ x y, (x, y, 1) ∈ config.cubes ∧ (x + 1, y, 1) ∈ config.cubes) ∧
  (∃ x y, (x, y, 2) ∈ config.cubes)

/-- Checks if the configuration matches the side view -/
def matches_side_view (config : CubeConfiguration) : Prop :=
  (∃ x z, (x, 0, z) ∈ config.cubes) ∧
  (∃ x z, (x, 1, z) ∈ config.cubes ∧ (x, 1, z + 1) ∈ config.cubes) ∧
  (∃ x z, (x, 2, z) ∈ config.cubes ∧ (x, 2, z + 1) ∈ config.cubes ∧ (x, 2, z + 2) ∈ config.cubes)

/-- The main theorem stating that the minimum number of cubes is 6 -/
theorem min_cubes_for_views : 
  ∃! (config : CubeConfiguration), 
    satisfies_sharing_condition config ∧ 
    matches_front_view config ∧ 
    matches_side_view config ∧ 
    config.cubes.card = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cubes_for_views_l383_38308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_increase_percentage_l383_38345

theorem rent_increase_percentage 
  (num_friends : ℕ) 
  (original_average : ℝ) 
  (new_average : ℝ) 
  (original_rent : ℝ) 
  (h1 : num_friends = 4)
  (h2 : original_average = 800)
  (h3 : new_average = 870)
  (h4 : original_rent = 1400) : 
  (((num_friends * new_average - num_friends * original_average) / original_rent) * 100 = 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_increase_percentage_l383_38345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_boys_count_l383_38395

theorem school_boys_count : ∃ B : ℕ, 
  let muslim_percent : ℚ := 46 / 100
  let hindu_percent : ℚ := 28 / 100
  let sikh_percent : ℚ := 10 / 100
  let other_boys : ℕ := 136
  muslim_percent + hindu_percent + sikh_percent + (other_boys : ℚ) / B = 1 ∧ B = 850 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_boys_count_l383_38395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_4_less_than_b_7_l383_38363

def b : ℕ → (ℕ → ℕ) → ℚ
  | 0, _ => 1  -- Add a case for 0
  | 1, α => 1 + 1 / (α 1 : ℚ)
  | n+1, α => 1 + 1 / (b n α + 1 / (α (n+1) : ℚ))

theorem b_4_less_than_b_7 (α : ℕ → ℕ) : b 4 α < b 7 α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_4_less_than_b_7_l383_38363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_division_theorem_l383_38342

/-- Represents a regular octagon -/
structure RegularOctagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Represents a straight cut on an octagon -/
structure StraightCut where
  start : ℝ × ℝ
  end_ : ℝ × ℝ

/-- Represents a set of pieces resulting from cuts on an octagon -/
structure OctagonPieces where
  pieces : Set (Set (ℝ × ℝ))

/-- Function to apply cuts to an octagon -/
noncomputable def apply_cuts (octagon : RegularOctagon) (cuts : List StraightCut) : OctagonPieces :=
  sorry

/-- Function to check if pieces can form a given number of congruent regular octagons -/
def can_form_octagons (pieces : OctagonPieces) (n : ℕ) : Prop :=
  sorry

/-- The main theorem statement -/
theorem octagon_division_theorem :
  ∀ (octagon1 octagon2 : RegularOctagon),
  ∃ (cuts : List StraightCut),
    let pieces1 := apply_cuts octagon1 cuts
    let pieces2 := apply_cuts octagon2 cuts
    (can_form_octagons pieces1 2 ∧ can_form_octagons pieces2 2) ∧
    (can_form_octagons pieces1 4 ∧ can_form_octagons pieces2 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_division_theorem_l383_38342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_squares_l383_38350

/-- A lattice point on a 2D grid. -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A square on a lattice grid. -/
structure LatticeSquare where
  vertices : Fin 4 → LatticePoint

/-- The size of the grid. -/
def gridSize : ℕ := 6

/-- Predicate to check if a LatticeSquare is valid on the grid. -/
def isValidSquare (s : LatticeSquare) : Prop :=
  ∀ i, 0 ≤ (s.vertices i).x ∧ (s.vertices i).x < gridSize ∧
       0 ≤ (s.vertices i).y ∧ (s.vertices i).y < gridSize

/-- Predicate to check if two LatticeSquares are congruent. -/
def areCongruent (s1 s2 : LatticeSquare) : Prop :=
  sorry  -- Definition of congruence between two squares

/-- The set of all valid LatticeSquares on the grid. -/
def allValidSquares : Set LatticeSquare :=
  { s | isValidSquare s }

/-- The set of non-congruent squares from allValidSquares. -/
noncomputable def nonCongruentSquares : Finset LatticeSquare :=
  sorry  -- Definition to select non-congruent squares from allValidSquares

/-- The main theorem stating the number of non-congruent squares. -/
theorem count_non_congruent_squares :
  Finset.card nonCongruentSquares = 155 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_squares_l383_38350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_distance_l383_38390

/-- Given a point A(1,2,3) in a three-dimensional Cartesian coordinate system,
    and its projection B onto the yOz plane, prove that the length of OB is √13. -/
theorem projection_distance (A B : ℝ × ℝ × ℝ) : 
  A = (1, 2, 3) → 
  B = (0, A.2.1, A.2.2) → 
  Real.sqrt ((B.1)^2 + (B.2.1)^2 + (B.2.2)^2) = Real.sqrt 13 := by
  sorry

#check projection_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_distance_l383_38390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_10_l383_38368

def factors_of_120 : Finset ℕ := Finset.filter (λ n => n > 0 ∧ 120 % n = 0) (Finset.range 121)

def factors_less_than_10 : Finset ℕ := Finset.filter (λ n => n < 10) factors_of_120

theorem probability_factor_less_than_10 : 
  (Finset.card factors_less_than_10 : ℚ) / (Finset.card factors_of_120 : ℚ) = 7 / 16 := by
  sorry

#eval Finset.card factors_of_120
#eval Finset.card factors_less_than_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_10_l383_38368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_is_1_a_range_for_f_increasing_l383_38343

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a/x

-- Part 1: Prove f(x) is increasing on [1, +∞) when a = 1
theorem f_increasing_when_a_is_1 :
  ∀ x1 x2 : ℝ, x1 ≥ 1 → x2 ≥ 1 → x1 > x2 → f 1 x1 > f 1 x2 := by
  sorry

-- Part 2: Prove the range of a for f(x) to be increasing on [2, +∞)
theorem a_range_for_f_increasing :
  ∀ a : ℝ, (∀ x1 x2 : ℝ, x1 ≥ 2 → x2 ≥ 2 → x1 > x2 → f a x1 > f a x2) ↔ a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_is_1_a_range_for_f_increasing_l383_38343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l383_38364

-- Define the points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (4, 3)

-- Define the parabola y^2 = 4x
def on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem min_distance_sum :
  ∀ P : ℝ × ℝ, on_parabola P → distance A P + distance B P ≥ 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l383_38364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l383_38377

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 300 →
  train_speed_kmh = 90 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 12 := by
  intro h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

-- The following line is commented out as it's not necessary for building
-- #eval train_crossing_time 300 90 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l383_38377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_income_percentage_is_six_percent_l383_38319

/-- Calculates the percentage of total annual income to entire investment for given investments -/
noncomputable def investment_income_percentage (initial_amount : ℝ) (initial_rate : ℝ) 
  (additional_amount : ℝ) (additional_rate : ℝ) : ℝ :=
  let total_investment := initial_amount + additional_amount
  let total_income := initial_amount * initial_rate + additional_amount * additional_rate
  (total_income / total_investment) * 100

/-- Theorem stating that for the given investments, the percentage of total annual income to entire investment is 6% -/
theorem investment_income_percentage_is_six_percent :
  investment_income_percentage 3000 0.05 1499.9999999999998 0.08 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_income_percentage_is_six_percent_l383_38319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_is_unique_l383_38353

/-- The intersection point of ρ=8sinθ and ρ=-8cosθ, where ρ > 0 and 0 ≤ θ < 2π -/
noncomputable def intersection_point : ℝ × ℝ := (4 * Real.sqrt 2, 3 * Real.pi / 4)

/-- First curve equation -/
noncomputable def curve1 (θ : ℝ) : ℝ := 8 * Real.sin θ

/-- Second curve equation -/
noncomputable def curve2 (θ : ℝ) : ℝ := -8 * Real.cos θ

theorem intersection_point_is_unique :
  ∃! p : ℝ × ℝ, 
    let (ρ, θ) := p
    0 < ρ ∧ 
    0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
    ρ = curve1 θ ∧ 
    ρ = curve2 θ ∧ 
    p = intersection_point := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_is_unique_l383_38353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_factor_proof_l383_38344

theorem other_factor_proof (w : ℕ) (h1 : ∃ k : ℕ, 936 * w = 27 * k)
  (h2 : ∃ k : ℕ, 936 * w = 100 * k) (h3 : w ≥ 120) :
  w = 120 * 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_factor_proof_l383_38344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ternary_2011_equals_58_l383_38380

/-- Converts a ternary (base-3) digit to its decimal (base-10) value based on its position. -/
def ternary_to_decimal (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (3 ^ position)

/-- The ternary representation of the number as a list of digits (least significant first). -/
def ternary_number : List ℕ := [1, 1, 0, 2]

/-- Theorem stating that the decimal representation of 2011₃ is 58. -/
theorem ternary_2011_equals_58 : 
  (List.zipWith ternary_to_decimal ternary_number (List.range ternary_number.length)).sum = 58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ternary_2011_equals_58_l383_38380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enrollment_difference_l383_38349

-- Define the schools and their enrollments
def schools : List (String × Nat) := [
  ("Varsity", 1500),
  ("Northwest", 1800),
  ("Central", 2400),
  ("Greenbriar", 2150),
  ("Maplewood", 1000)
]

-- Theorem statement
theorem enrollment_difference : 
  let enrollments := schools.map (λ (_, n) => n)
  (enrollments.maximum? ≠ none) ∧ 
  (enrollments.minimum? ≠ none) ∧ 
  ((enrollments.maximum?.getD 0) - (enrollments.minimum?.getD 0) = 1400) :=
by
  -- Introduce the local definition
  let enrollments := schools.map (λ (_, n) => n)
  
  -- Prove each part of the conjunction
  have h1 : enrollments.maximum? ≠ none := by sorry
  have h2 : enrollments.minimum? ≠ none := by sorry
  have h3 : (enrollments.maximum?.getD 0) - (enrollments.minimum?.getD 0) = 1400 := by sorry
  
  -- Combine the proofs
  exact ⟨h1, h2, h3⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enrollment_difference_l383_38349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_capacity_l383_38339

/-- Proves that the capacity of a train is 120 people given the conditions of bus capacities. -/
theorem train_capacity (bus_count : ℕ) (bus_capacity_ratio : ℚ) (total_bus_capacity : ℕ) : ℕ :=
  let train_capacity : ℕ := 120
  by
    have h1 : bus_count = 2 := by sorry
    have h2 : bus_capacity_ratio = 1 / 6 := by sorry
    have h3 : total_bus_capacity = 40 := by sorry
    have h4 : (↑bus_count : ℚ) * bus_capacity_ratio * ↑train_capacity = ↑total_bus_capacity := by
      calc
        (↑bus_count : ℚ) * bus_capacity_ratio * ↑train_capacity
          = 2 * (1 / 6) * 120 := by rw [h1, h2]; norm_cast
        _ = 40 := by norm_num
        _ = ↑total_bus_capacity := by rw [h3]; norm_cast
    exact train_capacity

#check train_capacity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_capacity_l383_38339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_circumscribed_sphere_triangular_pyramid_l383_38388

/-- The surface area of a sphere with radius R -/
noncomputable def surface_area_sphere (R : ℝ) : ℝ := 4 * Real.pi * R^2

/-- The surface area of the circumscribed sphere of a triangular pyramid with edge length a -/
noncomputable def surface_area_of_circumscribed_sphere_of_triangular_pyramid (a : ℝ) : ℝ :=
  surface_area_sphere ((Real.sqrt 6 / 4) * a)

/-- The surface area of a sphere circumscribing a triangular pyramid with equal edge lengths -/
theorem surface_area_circumscribed_sphere_triangular_pyramid (a : ℝ) (h : a > 0) :
  ∃ (S : ℝ), S = (3/2) * Real.pi * a^2 ∧ 
  S = surface_area_of_circumscribed_sphere_of_triangular_pyramid a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_circumscribed_sphere_triangular_pyramid_l383_38388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l383_38392

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := (sin (π * x) - cos (π * x) + 2) / sqrt x

/-- The theorem stating the minimum value of f(x) in the given interval -/
theorem min_value_of_f :
  ∀ x : ℝ, 1/4 ≤ x → x ≤ 5/4 → f x ≥ 4 * sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l383_38392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_ABC_l383_38386

open Real

-- Define the curve C₁
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (cos α, 1 + sin α)

-- Define the polar coordinates of a point
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

-- Define point A on curve C₁
noncomputable def A (θ : ℝ) : PolarPoint := { ρ := 2 * sin θ, θ := θ }

-- Define point B satisfying |OA| • |OB| = 6
noncomputable def B (θ : ℝ) : PolarPoint := { ρ := 3 / sin θ, θ := θ }

-- Define point C
def C : PolarPoint := { ρ := 2, θ := 0 }

-- Define the area of triangle ABC
noncomputable def area_ABC (θ : ℝ) : ℝ := |3 - 2 * (sin θ)^2|

-- Theorem statement
theorem min_area_ABC :
  ∃ (θ : ℝ), ∀ (θ' : ℝ), area_ABC θ ≤ area_ABC θ' ∧ area_ABC θ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_ABC_l383_38386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_shift_l383_38393

noncomputable def point := ℝ × ℝ

def shift (p : point) (s : point) : point :=
  (p.1 - s.1, p.2 - s.2)

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_after_shift :
  let origin_shift : point := (3, 4)
  let p1 : point := (-2, -6)
  let p2 : point := (10, -1)
  let shifted_p1 := shift p1 origin_shift
  let shifted_p2 := shift p2 origin_shift
  distance shifted_p1 shifted_p2 = 13 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_shift_l383_38393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_values_l383_38379

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2*x) + 2*a^x - 9

theorem max_value_implies_a_values (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≤ 6) ∧ (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = 6) →
  a = 3 ∨ a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_values_l383_38379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l383_38320

-- Define the parabola
def parabola (x : ℝ) : ℝ := -x^2 + 16

-- Theorem statement
theorem parabola_properties :
  -- X-intercepts
  (∃ x : ℝ, parabola x = 0 ∧ (x = 4 ∨ x = -4)) ∧
  -- Area of trapezoid
  (let A : ℝ × ℝ := (-4, 0)
   let B : ℝ × ℝ := (4, 0)
   let M : ℝ × ℝ := (-3, 7)
   let N : ℝ × ℝ := (3, 7)
   (1/2 : ℝ) * (6 + 8) * 7 = 49) ∧
  -- Area of quadrilateral
  (let V : ℝ × ℝ := (0, 16)
   let O : ℝ × ℝ := (0, 0)
   let P : ℝ × ℝ := (-7, -33)
   let Q : ℝ × ℝ := (7, -33)
   16 * 7 = 112) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l383_38320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l383_38399

noncomputable def power_function (m : ℤ) (x : ℝ) : ℝ := x^(m^2 - 2*m - 3)

theorem power_function_properties (m : ℤ) :
  (∀ x : ℝ, x ≠ 0 → power_function m x ≠ 0) ∧
  (∀ x : ℝ, power_function m x = power_function m (-x)) →
  m = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l383_38399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_AB_length_median_AM_l383_38369

-- Define the triangle ABC
noncomputable def A : ℝ × ℝ := (0, 5)
noncomputable def B : ℝ × ℝ := (-2, -1)
noncomputable def C : ℝ × ℝ := (4, 3)

-- Define M as the midpoint of BC
noncomputable def M : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Theorem for the equation of line AB
theorem line_equation_AB : 
  ∀ (x y : ℝ), (3 * x - y + 5 = 0) ↔ (∃ t : ℝ, (x, y) = (1 - t) • A + t • B) := by sorry

-- Theorem for the length of median AM
theorem length_median_AM : 
  Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) = Real.sqrt 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_AB_length_median_AM_l383_38369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_property_l383_38302

open Nat

-- Define a function to create a number with n nines
def nines (n : ℕ) : ℕ :=
  (10^n - 1)

-- Define the period of a fraction
noncomputable def period (p q : ℕ) : ℕ :=
  sorry

-- Define a function to split a number into two halves
def split_number (N : ℕ) : ℕ × ℕ :=
  sorry

theorem period_property (p q : ℕ) (h1 : Nat.Prime q) (h2 : q > 5) (h3 : p < q) :
  let N := period p q
  let (N₁, N₂) := split_number N
  let n := (Nat.log 10 N + 1) / 2
  N₁ + N₂ = nines n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_property_l383_38302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_alpha_l383_38398

-- Define the function f
noncomputable def f (α : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + Real.sin x
  else -x^2 + Real.cos (x + α)

-- State the theorem
theorem odd_function_alpha (α : ℝ) :
  (0 ≤ α ∧ α < 2 * Real.pi) →
  (∀ x, f α x = -(f α (-x))) →
  α = 3 * Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_alpha_l383_38398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_ace_position_l383_38370

/-- A well-shuffled pack of cards -/
structure CardPack where
  n : ℕ  -- Total number of cards
  ace_count : ℕ  -- Number of aces
  is_shuffled : Bool  -- Whether the pack is well shuffled

/-- The expected position of the second ace in a well-shuffled pack -/
def expected_position_second_ace (pack : CardPack) : ℚ :=
  (pack.n + 1 : ℚ) / 2

/-- Theorem stating the expected position of the second ace -/
theorem second_ace_position (pack : CardPack) 
  (h1 : pack.n > 0)
  (h2 : pack.ace_count = 3)
  (h3 : pack.is_shuffled = true) :
  expected_position_second_ace pack = (pack.n + 1 : ℚ) / 2 := by
  sorry

#eval expected_position_second_ace ⟨10, 3, true⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_ace_position_l383_38370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_array_sum_is_one_thirty_third_l383_38371

/-- Defines the entry of the 1/4-array at row r and column c -/
def arrayEntry (r c : ℕ) : ℚ := (1 / (3 * 4) ^ r) * (1 / 4 ^ c)

/-- The sum of all terms in the 1/4-array -/
noncomputable def arraySum : ℚ := ∑' r, ∑' c, arrayEntry r c

/-- Theorem stating that the sum of all terms in the 1/4-array is 1/33 -/
theorem array_sum_is_one_thirty_third : arraySum = 1 / 33 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_array_sum_is_one_thirty_third_l383_38371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bugs_next_meeting_time_l383_38307

noncomputable def circle1_radius : ℝ := 7
noncomputable def circle2_radius : ℝ := 3
noncomputable def bug1_speed : ℝ := 4 * Real.pi
noncomputable def bug2_speed : ℝ := 3 * Real.pi

noncomputable def time_to_complete_circle1 : ℝ := (2 * Real.pi * circle1_radius) / bug1_speed
noncomputable def time_to_complete_circle2 : ℝ := (2 * Real.pi * circle2_radius) / bug2_speed

def next_meeting_time : ℝ := 14

theorem bugs_next_meeting_time :
  (∃ n : ℤ, next_meeting_time / time_to_complete_circle1 = n) ∧
  (∃ m : ℤ, next_meeting_time / time_to_complete_circle2 = m) ∧
  ∀ t : ℝ, 0 < t → t < next_meeting_time →
    ¬((∃ k : ℤ, t / time_to_complete_circle1 = k) ∧ (∃ l : ℤ, t / time_to_complete_circle2 = l)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bugs_next_meeting_time_l383_38307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l383_38367

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := Real.exp x * (a * x + b) - x^2 - 4 * x

-- State the theorem
theorem f_properties (a b : ℝ) :
  (∀ x, (deriv (f a b)) x = Real.exp x * (a * x + a + b) - 2 * x - 4 ∧
        (deriv (f a b)) 0 = 4 ∧ 
        f a b 0 = 4) →
  (a = 4 ∧ b = 4) ∧
  (∃ x_max : ℝ, x_max = -2 ∧
    ∀ x, f 4 4 x ≤ f 4 4 x_max ∧
    f 4 4 x_max = 4 * (1 - Real.exp (-2))) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l383_38367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_each_activity_has_participant_l383_38351

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of activities -/
def num_activities : ℕ := 3

/-- The total number of possible outcomes -/
def total_outcomes : ℕ := num_activities ^ num_students

/-- The number of favorable outcomes where each activity has at least one student -/
def favorable_outcomes : ℕ := Nat.choose num_students 2 * Nat.factorial (num_activities - 1)

/-- The probability that each activity has at least one student participating -/
theorem probability_each_activity_has_participant : 
  (favorable_outcomes : ℚ) / total_outcomes = 4 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_each_activity_has_participant_l383_38351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l383_38324

theorem count_integer_pairs : ∃ (n : ℕ), n = 2^14 ∧ 
  n = Finset.card (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ 
    Nat.gcd p.1 p.2 = Nat.factorial 5 ∧ 
    Nat.lcm p.1 p.2 = Nat.factorial 50) (Finset.product (Finset.range (Nat.factorial 50 + 1)) (Finset.range (Nat.factorial 50 + 1)))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l383_38324
