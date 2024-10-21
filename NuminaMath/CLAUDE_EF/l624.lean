import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nellie_legos_l624_62445

/-- The number of legos Nellie has after a series of transactions -/
def remaining_legos (initial : ℕ) : ℕ :=
  let after_move := initial - (initial * 15 / 100)
  let after_sister := after_move - (after_move / 8)
  let after_cousin := after_sister - (after_sister * 20 / 100)
  after_cousin

/-- Theorem stating that Nellie ends up with 227 legos -/
theorem nellie_legos : remaining_legos 380 = 227 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nellie_legos_l624_62445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_distances_implies_parallelogram_l624_62425

/-- A convex quadrilateral with vertices A, B, C, and D. -/
structure ConvexQuadrilateral (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C D : V)

/-- The distance from a point to a line. -/
noncomputable def distanceToLine (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] (P : V) (l : Set V) : ℝ :=
  sorry

/-- The sum of distances from a point to all sides of a quadrilateral. -/
noncomputable def sumDistancesToSides (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (Q : ConvexQuadrilateral V) (P : V) : ℝ :=
  sorry

/-- A quadrilateral is a parallelogram if and only if its opposite sides are parallel. -/
def isParallelogram (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (Q : ConvexQuadrilateral V) : Prop :=
  sorry

/-- The main theorem: if the sum of distances from each vertex to all sides is constant,
    then the quadrilateral is a parallelogram. -/
theorem constant_sum_distances_implies_parallelogram 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (Q : ConvexQuadrilateral V) :
  (∃ (k : ℝ), ∀ (v : V), v ∈ ({Q.A, Q.B, Q.C, Q.D} : Set V) → sumDistancesToSides V Q v = k) →
  isParallelogram V Q :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_distances_implies_parallelogram_l624_62425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_solution_set_l624_62436

open Set Real

theorem function_inequality_solution_set
  (f : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (h_ineq : ∀ x, deriv f x > f x)
  (h_init : f 0 = 1) :
  {x | f x < exp x} = Iio 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_solution_set_l624_62436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_implies_perpendicular_parallel_lines_planes_equal_angles_l624_62473

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular_line_line : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (angle : Line → Plane → ℝ)

-- Proposition 2
theorem perpendicular_parallel_implies_perpendicular 
  (m n : Line) (α : Plane) :
  perpendicular_line_plane m α → parallel_line_plane n α → perpendicular_line_line m n :=
by
  sorry

-- Proposition 4
theorem parallel_lines_planes_equal_angles 
  (m n : Line) (α β : Plane) :
  parallel_lines m n → parallel_planes α β → angle m α = angle n β :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_implies_perpendicular_parallel_lines_planes_equal_angles_l624_62473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l624_62496

noncomputable section

-- Define the parabola C2
def C2 (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point M
def M : ℝ × ℝ := (4, 0)

-- Define the line l
def l (m : ℝ) (x y : ℝ) : Prop := x = m*y + 4

-- Define the symmetric point P
def P (m : ℝ) : ℝ × ℝ := (8/(1+m^2), -8*m/(1+m^2))

-- Define the ellipse C1
def C1 (a : ℝ) (x y : ℝ) : Prop := x^2/a^2 + y^2/(a^2-1) = 1

theorem ellipse_parabola_intersection :
  ∃ (m : ℝ), 
    (∀ x y : ℝ, l m x y → C2 x y → (x < 4 → y < 0) → 
      ∃ x' y' : ℝ, l m x' y' ∧ C2 x' y' ∧ x' > 4 ∧ 
      (x' - 4)^2 + y'^2 = 16 * ((4 - x)^2 + y^2)) ∧
    (C2 (P m).1 (P m).2) ∧
    (∃ a : ℝ, a > 1 ∧ ∀ x y : ℝ, l m x y → C1 a x y) →
  m = 3/2 ∧ 
  (∀ a : ℝ, (∃ x y : ℝ, l (3/2) x y ∧ C1 a x y) → a^2 ≥ 17/2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l624_62496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_satisfies_conditions_l624_62466

-- Define the function S(x)
noncomputable def S (x : ℝ) : ℝ := 7 - 4 * Real.sqrt (5 - x)

-- State the theorem
theorem S_satisfies_conditions :
  (∀ x, x < 5 → (deriv S) x = 2 / Real.sqrt (5 - x)) ∧
  S 1 = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_satisfies_conditions_l624_62466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_bailing_rate_l624_62437

/-- The minimum bailing rate problem -/
theorem minimum_bailing_rate
  (distance : ℝ)
  (leak_rate : ℝ)
  (max_water : ℝ)
  (rowing_speed : ℝ)
  (h1 : distance = 3)
  (h2 : leak_rate = 7)
  (h3 : max_water = 20)
  (h4 : rowing_speed = 6)
  : ∃ (bailing_rate : ℝ), 
    (bailing_rate ≥ 6.33 ∧ bailing_rate < 6.34) ∧ 
    (∀ (r : ℝ), r < bailing_rate → 
      (distance / rowing_speed) * 60 * (leak_rate - r) > max_water) ∧
    (distance / rowing_speed) * 60 * (leak_rate - bailing_rate) ≤ max_water :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_bailing_rate_l624_62437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sides_implies_a_range_l624_62472

/-- 
Given two points M(1, -a) and N(a, 1) that are on opposite sides of the line 2x - 3y + 1 = 0,
prove that -1 < a < 1.
-/
theorem opposite_sides_implies_a_range (a : ℝ) : 
  (let M : ℝ × ℝ := (1, -a);
   let N : ℝ × ℝ := (a, 1);
   let l (x y : ℝ) := 2*x - 3*y + 1;
   l M.1 M.2 * l N.1 N.2 < 0) →
  -1 < a ∧ a < 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sides_implies_a_range_l624_62472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_theorem_l624_62465

/-- A graph with the properties described in the problem -/
structure FriendshipGraph (n : ℕ) where
  vertices : Finset (Fin n)
  edges : Finset (Fin n × Fin n)
  h_n : n ≥ 6
  h_degree : ∀ v, v ∈ vertices → (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≥ n / 2
  h_connected : ∀ S : Finset (Fin n), S.card = n / 2 →
    (∃ a b, a ∈ S ∧ b ∈ S ∧ (a, b) ∈ edges) ∨
    (∃ a b, a ∈ vertices \ S ∧ b ∈ vertices \ S ∧ (a, b) ∈ edges)

/-- The main theorem: every FriendshipGraph contains a triangle -/
theorem friendship_theorem {n : ℕ} (G : FriendshipGraph n) :
  ∃ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a, b) ∈ G.edges ∧ (b, c) ∈ G.edges ∧ (a, c) ∈ G.edges :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_theorem_l624_62465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_orange_retail_price_l624_62406

/-- Calculates the retail price per kilogram for sugar oranges -/
theorem sugar_orange_retail_price
  (purchase_quantity : ℝ)
  (cost_price : ℝ)
  (weight_loss_percentage : ℝ)
  (profit_percentage : ℝ)
  (h1 : purchase_quantity = 500)
  (h2 : cost_price = 4.80)
  (h3 : weight_loss_percentage = 0.10)
  (h4 : profit_percentage = 0.20) :
  let total_cost := purchase_quantity * cost_price
  let effective_weight := purchase_quantity * (1 - weight_loss_percentage)
  let desired_revenue := total_cost * (1 + profit_percentage)
  let retail_price := desired_revenue / effective_weight
  retail_price = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_orange_retail_price_l624_62406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequalities_l624_62453

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.log x - 1) + 1 / x

-- Define the function g
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := (b - 1) * (Real.log x / Real.log b) - (x^2 - 1) / 2

-- State the theorem
theorem function_inequalities 
  (a : ℝ) (b : ℝ) (h_b : b > 1) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ = 0 ∧ (deriv (f a)) x₀ = 0) →
  (∀ x : ℝ, x > 0 → f a x ≤ (x - 1)^2 / x) ∧
  (∀ x : ℝ, 1 < x → x < Real.sqrt b → 0 < g b x ∧ g b x < (b - 1)^2 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequalities_l624_62453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_rectangle_area_of_rectangle_l624_62457

-- Define the rectangle
noncomputable def rectangle_length : ℝ := Real.sqrt 128
noncomputable def rectangle_width : ℝ := Real.sqrt 75

-- Theorem for the perimeter
theorem perimeter_of_rectangle :
  2 * (rectangle_length + rectangle_width) = 16 * Real.sqrt 2 + 10 * Real.sqrt 3 := by
  sorry

-- Theorem for the exact area
theorem area_of_rectangle :
  rectangle_length * rectangle_width = 40 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_rectangle_area_of_rectangle_l624_62457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cotangent_difference_identity_l624_62423

theorem cotangent_difference_identity (α β : ℝ) :
  (Real.tan α)⁻¹^2 - (Real.tan β)⁻¹^2 = (Real.cos α)^2 - (Real.cos β)^2 / ((Real.sin α)^2 * (Real.sin β)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cotangent_difference_identity_l624_62423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_remaining_money_correct_l624_62412

/-- Calculates the remaining money after John buys Christmas gifts for his family. -/
def johns_remaining_money
  (initial_budget : ℚ)
  (shoes_price : ℚ)
  (yoga_mat_price : ℚ)
  (watch_price : ℚ)
  (weights_price : ℚ)
  (sales_tax_rate : ℚ)
  (watch_discount_rate : ℚ)
  : ℚ :=
let total_before_discount := shoes_price + yoga_mat_price + watch_price + weights_price
let watch_discount := watch_price * watch_discount_rate
let total_after_discount := total_before_discount - watch_discount
let sales_tax := total_after_discount * sales_tax_rate
let total_cost := total_after_discount + sales_tax
initial_budget - total_cost

theorem johns_remaining_money_correct
  (initial_budget : ℚ)
  (shoes_price : ℚ)
  (yoga_mat_price : ℚ)
  (watch_price : ℚ)
  (weights_price : ℚ)
  (sales_tax_rate : ℚ)
  (watch_discount_rate : ℚ)
  (h1 : initial_budget = 999)
  (h2 : shoes_price = 165)
  (h3 : yoga_mat_price = 85)
  (h4 : watch_price = 215)
  (h5 : weights_price = 60)
  (h6 : sales_tax_rate = 7 / 100)
  (h7 : watch_discount_rate = 10 / 100)
  : johns_remaining_money initial_budget shoes_price yoga_mat_price watch_price weights_price sales_tax_rate watch_discount_rate = 460.25 :=
by
  -- Proof goes here
  sorry

#eval johns_remaining_money 999 165 85 215 60 (7/100) (10/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_remaining_money_correct_l624_62412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_area_ratio_l624_62460

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  verticalAngle : ℝ
  area : ℝ
  h_area : area = (1/2) * base * height

/-- Given two isosceles triangles with equal vertical angles, 
    if the ratio of their heights is 0.5714285714285714, 
    then the ratio of their areas is 0.3265306122448979 -/
theorem isosceles_triangles_area_ratio 
  (triangle1 triangle2 : IsoscelesTriangle) 
  (h_equal_angle : triangle1.verticalAngle = triangle2.verticalAngle)
  (h_height_ratio : triangle1.height / triangle2.height = 0.5714285714285714) : 
  triangle1.area / triangle2.area = 0.3265306122448979 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_area_ratio_l624_62460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_implies_inverse_l624_62418

theorem power_equation_implies_inverse (x : ℝ) : 
  (81 : ℝ)^6 = (27 : ℝ)^(x + 2) → (3 : ℝ)^(-x) = 1/729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_implies_inverse_l624_62418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l624_62430

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 2))

theorem equation_solutions :
  ∀ x : ℝ, x ≥ 2 → (f x = 2 ↔ x = 11 ∨ x = 27) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l624_62430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_intersection_l624_62498

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((x + 1) / (x - 2))
noncomputable def g (a x : ℝ) : ℝ := Real.sqrt (x^2 - (2*a + 1)*x + a^2 + a)

-- Define the domains A and B
def A : Set ℝ := {x : ℝ | x > 2 ∨ x ≤ -1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a + 1 ∨ x ≤ a}

-- State the theorem
theorem range_of_a_for_intersection (a : ℝ) : 
  (A ∩ B a = A) ↔ a ∈ Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_intersection_l624_62498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_relationship_l624_62486

-- Define the amount of soda Alice has
variable (A : ℝ)

-- Define Jacqueline's amount of soda relative to Alice's
def J (A : ℝ) : ℝ := 1.8 * A

-- Define Liliane's amount of soda relative to Alice's
def L (A : ℝ) : ℝ := 1.4 * A

-- Theorem stating the relationship between Jacqueline's and Liliane's soda amounts
theorem soda_relationship (A : ℝ) (A_pos : A > 0) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ abs ((J A - L A) / (L A) - 0.29) < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_relationship_l624_62486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l624_62469

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Indicates whether a rectangle is subdivided into n equal rectangles -/
def Rectangle.subdivided_into (r : Rectangle) (n : ℕ) : Prop := 
  ∃ (sub_width sub_height : ℝ), 
    r.width = n * sub_width ∧ 
    r.height = sub_height ∧ 
    sub_width < sub_height

/-- Given a rectangle ABCD divided into five equal rectangles with a perimeter of 20 cm,
    prove that its area is 750/121 cm^2. -/
theorem rectangle_area (ABCD : Rectangle) : 
  ABCD.subdivided_into 5 → 
  ABCD.perimeter = 20 → 
  ABCD.area = 750 / 121 :=
by
  intro h_subdivided h_perimeter
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l624_62469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_k_equals_one_l624_62431

-- Define the lines l₁ and l₂
def l₁ (k : ℝ) : ℝ → ℝ → Prop := λ x y ↦ x + (1 + k) * y = 2 - k
def l₂ (k : ℝ) : ℝ → ℝ → Prop := λ x y ↦ k * x + 2 * y + 8 = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop := 
  ∃ (a b c d : ℝ), ∀ (x y : ℝ), (f x y ↔ a * x + b * y = c) ∧ (g x y ↔ a * x + b * y = d)

-- Theorem statement
theorem parallel_lines_k_equals_one :
  ∀ k : ℝ, parallel (l₁ k) (l₂ k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_k_equals_one_l624_62431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_in_interval_l624_62434

noncomputable def f (x : ℝ) := (x^2 - 3*x + 3) / (2*x - 4)

theorem f_has_minimum_in_interval :
  ∃ (m : ℝ), ∀ (x : ℝ), -2 < x → x < 3 → f x ≥ m := by
  sorry

#check f_has_minimum_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_in_interval_l624_62434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_analysis_considerations_l624_62448

-- Define the types for our concepts
structure RegressionModel where

structure Population where

structure TimeRange where

structure SampleRange where

structure ForecastVariable where

-- Define the properties
def applicable_to_population (model : RegressionModel) (pop : Population) : Prop :=
  sorry

def has_time_validity (model : RegressionModel) (time : TimeRange) : Prop :=
  sorry

def affected_by_sample_range (model : RegressionModel) (range : SampleRange) : Prop :=
  sorry

def is_average_of_possible_values (forecast : ForecastVariable) : Prop :=
  sorry

-- Theorem representing the considerations in regression analysis
theorem regression_analysis_considerations 
  (model : RegressionModel) 
  (pop : Population) 
  (time : TimeRange) 
  (range : SampleRange) 
  (forecast : ForecastVariable) : 
  applicable_to_population model pop ∧ 
  has_time_validity model time ∧ 
  affected_by_sample_range model range ∧ 
  is_average_of_possible_values forecast := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_analysis_considerations_l624_62448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l624_62446

-- Define the ellipse parameters
def major_axis_length : ℝ := 4
def directrix_x : ℝ := -4

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the intercepting line
def is_on_line (x y : ℝ) : Prop :=
  y = x + 1

-- Define the chord length function
noncomputable def chord_length (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem ellipse_chord_length :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_on_ellipse x₁ y₁ ∧
    is_on_ellipse x₂ y₂ ∧
    is_on_line x₁ y₁ ∧
    is_on_line x₂ y₂ ∧
    chord_length x₁ y₁ x₂ y₂ = 24 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l624_62446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_from_altitudes_l624_62404

/-- Given a triangle ABC with sides a, b, c and corresponding altitudes h_a, h_b, h_c,
    prove that if h_a = 1/2, h_b = √2/2, and h_c = 1, then cos A = -√2/4 -/
theorem triangle_cosine_from_altitudes (a b c h_a h_b h_c A : ℝ) :
  h_a = 1/2 → h_b = Real.sqrt 2 / 2 → h_c = 1 →
  (1/2) * a * h_a = (1/2) * b * h_b →
  (1/2) * b * h_b = (1/2) * c * h_c →
  Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) →
  Real.cos A = -Real.sqrt 2 / 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_from_altitudes_l624_62404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yvettes_budget_calculation_l624_62467

/-- Represents Yvette's initial budget for the frame in dollars. -/
def initial_budget : ℝ := sorry

/-- Represents the price of the frame Yvette initially wanted in dollars. -/
def wanted_frame_price : ℝ := 1.2 * initial_budget

/-- Represents the price of the smaller frame Yvette actually bought in dollars. -/
def smaller_frame_price : ℝ := 0.75 * wanted_frame_price

theorem yvettes_budget_calculation :
  smaller_frame_price = initial_budget - 6 →
  initial_budget = 60 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yvettes_budget_calculation_l624_62467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_distance_along_stream_l624_62427

/-- Represents the speed of a boat in km/hr -/
noncomputable def boat_speed : ℝ := 4

/-- Represents the distance traveled against the stream in km -/
noncomputable def distance_against_stream : ℝ := 2

/-- Represents the time taken for the journey in hours -/
noncomputable def time : ℝ := 1

/-- Calculates the speed of the stream in km/hr -/
noncomputable def stream_speed : ℝ := boat_speed - distance_against_stream / time

/-- Calculates the distance traveled along the stream in km -/
noncomputable def distance_along_stream : ℝ := (boat_speed + stream_speed) * time

theorem boat_distance_along_stream :
  distance_along_stream = 6 := by
  -- Unfold definitions
  unfold distance_along_stream
  unfold stream_speed
  -- Simplify expressions
  simp [boat_speed, distance_against_stream, time]
  -- The proof is completed by normalization of real numbers
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_distance_along_stream_l624_62427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_rational_function_l624_62492

open Complex

noncomputable def f (z : ℂ) : ℂ := ∑' k, 1 / (log z + 2 * Real.pi * I * k : ℂ)^4

theorem f_equals_rational_function (z : ℂ) (hz : z ≠ 0 ∧ z ≠ 1) : 
  f z = (z^3 + 4*z^2 + z) / (6*(z-1)^4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_rational_function_l624_62492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_values_for_all_x₀_l624_62428

/-- The function f(x) = 5x - x^2 -/
def f (x : ℝ) : ℝ := 5 * x - x^2

/-- The sequence x_n defined by x_n = f(x_{n-1}) for n ≥ 1 -/
def x_seq (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => f (x_seq x₀ n)

/-- The set of values in the sequence starting from x₀ -/
def seq_values (x₀ : ℝ) : Set ℝ :=
  {x | ∃ n : ℕ, x_seq x₀ n = x}

theorem infinite_values_for_all_x₀ :
  ∀ x₀ : ℝ, ¬(Set.Finite (seq_values x₀)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_values_for_all_x₀_l624_62428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersecting_line_slope_l624_62424

/-- Proves that for a parabola y^2 = 2px with focus F, if a line with positive slope k
    passes through the intersection point of the parabola and x-axis, and intersects
    the parabola at two points N and Q such that ∠NFQ = 90°, then k = √2/2. -/
theorem parabola_intersecting_line_slope (p : ℝ) (k : ℝ) (hk : k > 0) :
  let C : ℝ → ℝ → Prop := fun x y ↦ y^2 = 2*p*x
  let M : ℝ × ℝ := (-p/2, 0)
  let F : ℝ × ℝ := (p/2, 0)
  let line : ℝ → ℝ := fun x ↦ k * (x + p/2)
  (∃ (N Q : ℝ × ℝ), C N.1 N.2 ∧ C Q.1 Q.2 ∧ 
    N.2 = line N.1 ∧ Q.2 = line Q.1 ∧
    (N.1 - F.1) * (Q.1 - F.1) + (N.2 - F.2) * (Q.2 - F.2) = 0) →
  k = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersecting_line_slope_l624_62424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinguishable_colorings_eq_15_l624_62429

/-- A color that can be used to paint a face of a tetrahedron -/
inductive Color
| Red
| White
| Blue

/-- A coloring of a tetrahedron is a function from its faces to colors -/
def Coloring := Fin 4 → Color

/-- Two colorings are equivalent if they can be rotated to look identical -/
def equivalent (c1 c2 : Coloring) : Prop := sorry

/-- A distinguishable coloring is an equivalence class of colorings -/
def DistinguishableColoring := Quotient (Setoid.mk equivalent sorry)

/-- The number of distinguishable colorings of a regular tetrahedron -/
def num_distinguishable_colorings : ℕ := sorry

theorem num_distinguishable_colorings_eq_15 :
  num_distinguishable_colorings = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinguishable_colorings_eq_15_l624_62429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l624_62488

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 7 = ∫ x in (0 : ℝ)..π, Real.sin x →           -- given condition
  a 4 + a 6 + a 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l624_62488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_is_110_l624_62455

-- Define Josanna's current test scores
def current_scores : List ℕ := [75, 85, 95, 65, 80]

-- Define the target increase in average
def target_increase : ℕ := 5

-- Function to calculate the average of a list of natural numbers
def average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

-- Function to calculate the minimum score needed for the next test
noncomputable def min_score_needed (scores : List ℕ) (target_increase : ℕ) : ℕ :=
  let current_avg := average scores
  let target_avg := current_avg + target_increase
  (Int.ceil ((target_avg * (scores.length + 1 : ℚ)) - scores.sum)).toNat

-- Theorem statement
theorem min_score_is_110 :
  min_score_needed current_scores target_increase = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_is_110_l624_62455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_eight_l624_62410

/-- A geometric sequence with integer common ratio -/
structure GeometricSequence where
  a : ℕ → ℚ
  q : ℤ
  geom_prop : ∀ n : ℕ, a (n + 1) = a n * (q : ℚ)

/-- Sum of the first n terms of a geometric sequence -/
def geometricSum (g : GeometricSequence) (n : ℕ) : ℚ :=
  if g.q = 1 then n * g.a 0
  else g.a 0 * (1 - (g.q : ℚ) ^ n) / (1 - (g.q : ℚ))

theorem geometric_sum_eight (g : GeometricSequence) 
  (h1 : g.a 0 + g.a 3 = 18)
  (h2 : g.a 1 + g.a 2 = 12) :
  geometricSum g 8 = 510 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_eight_l624_62410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_ten_l624_62444

theorem power_of_ten (x y : ℝ) (h1 : (10 : ℝ)^x = 3) (h2 : (10 : ℝ)^y = 4) : 
  (10 : ℝ)^(2*x - y) = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_ten_l624_62444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_equals_double_sine_sum_min_cos_B_is_half_l624_62419

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0
  angle_sum : A + B + C = π
  sine_law : a / (Real.sin A) = b / (Real.sin B)
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)

/-- Part 1: Arithmetic sequence condition -/
def is_arithmetic_sequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

/-- Part 2: Geometric sequence condition -/
def is_geometric_sequence (t : Triangle) : Prop :=
  t.b^2 = t.a * t.c

/-- Theorem 1: If sides form an arithmetic sequence, then sin A + sin C = 2sin(A+C) -/
theorem sine_sum_equals_double_sine_sum (t : Triangle) 
  (h : is_arithmetic_sequence t) : 
  Real.sin t.A + Real.sin t.C = 2 * Real.sin (t.A + t.C) := by sorry

/-- Theorem 2: If sides form a geometric sequence, then the minimum value of cos B is 1/2 -/
theorem min_cos_B_is_half (t : Triangle) 
  (h : is_geometric_sequence t) : 
  ∃ (min_cos_B : ℝ), (∀ (t' : Triangle), is_geometric_sequence t' → Real.cos t'.B ≥ min_cos_B) ∧ min_cos_B = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_equals_double_sine_sum_min_cos_B_is_half_l624_62419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_black_region_probability_l624_62415

/-- The probability of a coin covering part of the black region -/
noncomputable def black_region_probability : ℝ := (32 + 9 * Real.sqrt 2 + Real.pi) / 81

/-- The side length of the square -/
def square_side : ℝ := 10

/-- The leg length of the isosceles right triangles in the corners -/
def triangle_leg : ℝ := 3

/-- The side length of the central diamond -/
noncomputable def diamond_side : ℝ := 3 * Real.sqrt 2

/-- The side length of the small square in the center of the diamond -/
def small_square_side : ℝ := 1

/-- The diameter of the circular coin -/
def coin_diameter : ℝ := 1

/-- The number of black isosceles right triangles in the corners -/
def num_triangles : ℕ := 4

theorem coin_black_region_probability :
  let total_area := (square_side - coin_diameter) ^ 2
  let black_area := 
    -- Area of diamond and small square with boundary
    (2 * diamond_side ^ 2 + small_square_side ^ 2 + Real.pi / 2 + 3 * Real.sqrt 2 + 1) +
    -- Area of triangles with boundary
    (num_triangles : ℝ) * (triangle_leg ^ 2 / 2 + Real.pi / 16 + 3 * Real.sqrt 2 / 2)
  black_area / total_area = black_region_probability := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_black_region_probability_l624_62415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l624_62443

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then Real.log (1 - x) else 2 / (x - 1)

/-- The function g as defined in the problem -/
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := k / x^2

/-- The main theorem stating the maximum value of k -/
theorem max_k_value : 
  (∃ k : ℕ, k > 0 ∧ 
    (∀ p : ℝ, p > 1 → 
      ∃ m n : ℝ, m < 0 ∧ 0 < n ∧ n < p ∧ 
        f m = f p ∧ f p = g k n) ∧
    (∀ k' : ℕ, k' > k → 
      ¬(∀ p : ℝ, p > 1 → 
        ∃ m n : ℝ, m < 0 ∧ 0 < n ∧ n < p ∧ 
          f m = f p ∧ f p = g k' n))) ∧
  (∀ k : ℕ, 
    (k > 0 ∧ 
      (∀ p : ℝ, p > 1 → 
        ∃ m n : ℝ, m < 0 ∧ 0 < n ∧ n < p ∧ 
          f m = f p ∧ f p = g k n) ∧
      (∀ k' : ℕ, k' > k → 
        ¬(∀ p : ℝ, p > 1 → 
          ∃ m n : ℝ, m < 0 ∧ 0 < n ∧ n < p ∧ 
            f m = f p ∧ f p = g k' n))) →
    k = 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l624_62443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_empty_time_is_12_hours_l624_62421

/-- Represents the tank system with a leak and an inlet pipe -/
structure TankSystem where
  capacity : ℝ
  leak_empty_time : ℝ
  inlet_rate : ℝ

/-- Calculates the time it takes for the tank to become empty -/
noncomputable def empty_time (system : TankSystem) : ℝ :=
  let leak_rate := system.capacity / system.leak_empty_time
  let inlet_rate_hourly := system.inlet_rate * 60
  let net_rate := leak_rate - inlet_rate_hourly
  system.capacity / net_rate

/-- Theorem stating that for the given system, the empty time is 12 hours -/
theorem tank_empty_time_is_12_hours :
  let system : TankSystem := {
    capacity := 2160,
    leak_empty_time := 4,
    inlet_rate := 6
  }
  empty_time system = 12 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_empty_time_is_12_hours_l624_62421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_length_proof_l624_62499

/-- The length of a straight line segment on the grid -/
def straight_length : ℝ := 1

/-- The length of a diagonal line segment on the grid -/
noncomputable def diagonal_length : ℝ := Real.sqrt 2

/-- The number of straight line segments in the "XYZ" formation -/
def num_straight_segments : ℕ := 14

/-- The number of diagonal line segments in the "XYZ" formation -/
def num_diagonal_segments : ℕ := 5

/-- The total length of all line segments forming "XYZ" -/
noncomputable def total_length : ℝ := 
  (num_straight_segments : ℝ) * straight_length + 
  (num_diagonal_segments : ℝ) * diagonal_length

/-- Theorem stating that the total length of "XYZ" is 14 + 5 * sqrt(2) -/
theorem xyz_length_proof : 
  total_length = 14 + 5 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_length_proof_l624_62499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ryan_learned_two_days_l624_62411

/-- Ryan's learning schedule and total time spent --/
structure LearningSchedule where
  english_hours_per_day : ℚ
  chinese_hours_per_day : ℚ
  total_english_hours : ℚ

/-- Calculate the number of days Ryan learned --/
def days_learned (schedule : LearningSchedule) : ℚ :=
  schedule.total_english_hours / schedule.english_hours_per_day

/-- Theorem: Ryan learned for 2 days --/
theorem ryan_learned_two_days (schedule : LearningSchedule)
  (h1 : schedule.english_hours_per_day = 6)
  (h2 : schedule.chinese_hours_per_day = 5)
  (h3 : schedule.total_english_hours = 12) :
  days_learned schedule = 2 := by
  sorry

#check ryan_learned_two_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ryan_learned_two_days_l624_62411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_divisibility_l624_62484

noncomputable def T (n : ℕ) : ℝ := (3 + Real.sqrt 5)^n + (3 - Real.sqrt 5)^n

theorem smallest_integer_divisibility (n : ℕ) (hn : n > 0) :
  (∀ k, ∃ m : ℤ, T k = m) ∧
  (∀ x : ℝ, x > (3 + Real.sqrt 5)^(2*n) → ⌈x⌉ ≥ T (2*n)) ∧
  (∃ m : ℤ, T (2*n) = m ∧ 2^(n+1) ∣ m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_divisibility_l624_62484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_moves_terminate_l624_62401

/-- Represents the state of the apartment system -/
structure ApartmentSystem :=
  (num_apartments : ℕ)
  (num_people : ℕ)
  (occupancy : Fin num_apartments → ℕ)
  (overcrowd_threshold : ℕ)

/-- Defines an overcrowded apartment -/
def is_overcrowded (sys : ApartmentSystem) (apt : Fin sys.num_apartments) : Prop :=
  sys.occupancy apt ≥ sys.overcrowd_threshold

/-- Calculates the total number of handshakes in the system -/
def total_handshakes (sys : ApartmentSystem) : ℕ :=
  (Finset.univ.sum (λ apt => (sys.occupancy apt).choose 2))

/-- Represents a single day's movement -/
def daily_movement (sys : ApartmentSystem) (new_occupancy : Fin sys.num_apartments → ℕ) : ApartmentSystem :=
  { sys with occupancy := new_occupancy }

/-- The main theorem to prove -/
theorem apartment_moves_terminate (initial_sys : ApartmentSystem) 
  (h_apartments : initial_sys.num_apartments = 120)
  (h_people : initial_sys.num_people = 119)
  (h_overcrowd : initial_sys.overcrowd_threshold = 15)
  (h_valid_occupancy : ∀ apt, initial_sys.occupancy apt ≤ initial_sys.num_people) :
  ∃ (n : ℕ) (final_sys : ApartmentSystem), 
    (∀ apt, ¬is_overcrowded final_sys apt) ∧ 
    (total_handshakes final_sys < total_handshakes initial_sys) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_moves_terminate_l624_62401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_square_angle_l624_62456

-- Define the regular hexagon
structure RegularHexagon :=
  (vertices : Finset (ℝ × ℝ))
  (center : ℝ × ℝ)
  (is_regular : vertices.card = 6)

-- Define the square
structure Square :=
  (vertices : Finset (ℝ × ℝ))
  (is_square : vertices.card = 4)

-- Define the configuration
def HexagonSquareConfig (h : RegularHexagon) (s : Square) :=
  ∃ (A B : ℝ × ℝ), A ∈ h.vertices ∧ B ∈ h.vertices ∧ A ∈ s.vertices ∧
    (∃ (side_h : Finset (ℝ × ℝ)) (side_s : Finset (ℝ × ℝ)),
      side_h ⊆ h.vertices ∧ side_s ⊆ s.vertices ∧
      side_h.card = 2 ∧ side_s.card = 2 ∧
      A ∈ side_h ∧ A ∈ side_s ∧
      (∀ (p q : ℝ × ℝ), p ∈ side_h ∧ q ∈ side_s → p.1 = q.1 ∨ p.2 = q.2))

-- Define the angle function (this is a placeholder)
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- The theorem to prove
theorem hexagon_square_angle (h : RegularHexagon) (s : Square)
  (config : HexagonSquareConfig h s) :
  ∃ (A B : ℝ × ℝ), A ∈ h.vertices ∧ B ∈ h.vertices ∧ A ∈ s.vertices ∧
    angle A B h.center = 30 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_square_angle_l624_62456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l624_62449

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The conditions given in the problem --/
def TriangleConditions (t : Triangle) : Prop :=
  t.a - t.b = 2 ∧ t.c = 4 ∧ Real.sin t.A = 2 * Real.sin t.B

/-- The area of a triangle --/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  (1 / 2) * t.a * t.c * Real.sin t.B

/-- The main theorem to prove --/
theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  triangleArea t = Real.sqrt 15 ∧
  Real.sin (2 * t.A - t.B) = (7 * Real.sqrt 15) / 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l624_62449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_sqrt_radii_equals_target_l624_62474

/-- Represents a circle in the upper half-plane --/
structure Circle where
  radius : ℝ
  x_tangent : ℝ

/-- Represents a layer of circles --/
def Layer := List Circle

/-- Constructs the initial layer L₀ --/
noncomputable def initial_layer : Layer :=
  [{ radius := (50 : ℝ)^2, x_tangent := 0 }, { radius := (55 : ℝ)^2, x_tangent := 50^2 + 55^2 }]

/-- Constructs a new circle between two given circles --/
noncomputable def construct_circle (c1 c2 : Circle) : Circle :=
  { radius := (c1.radius * c2.radius) / ((c1.radius.sqrt + c2.radius.sqrt)^2),
    x_tangent := c1.x_tangent + 2 * (c1.radius * c2.radius).sqrt }

/-- Constructs the next layer given the previous layer --/
noncomputable def next_layer (prev : Layer) : Layer :=
  sorry

/-- Constructs all layers up to L₇ --/
noncomputable def construct_layers : List Layer :=
  sorry

/-- The set S as the union of all layers up to L₇ --/
noncomputable def S : List Circle :=
  sorry

/-- The sum of 1/√r(C) for all circles C in S --/
noncomputable def sum_inverse_sqrt_radii : ℝ :=
  sorry

/-- The main theorem to prove --/
theorem sum_inverse_sqrt_radii_equals_target : sum_inverse_sqrt_radii = 2688 / 550 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_sqrt_radii_equals_target_l624_62474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_not_always_true_l624_62402

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define perimeter
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Define circumradius (using the formula R = abc / (4 * area))
noncomputable def circumradius (t : Triangle) : ℝ :=
  (t.a * t.b * t.c) / (4 * (1/4 * Real.sqrt ((t.a + t.b + t.c) * (-t.a + t.b + t.c) * (t.a - t.b + t.c) * (t.a + t.b - t.c))))

-- Define inradius (using the formula r = area / s, where s is semi-perimeter)
noncomputable def inradius (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  (1/4 * Real.sqrt ((t.a + t.b + t.c) * (-t.a + t.b + t.c) * (t.a - t.b + t.c) * (t.a + t.b - t.c))) / s

-- Theorem statement
theorem triangle_inequalities_not_always_true :
  ¬(∀ t : Triangle,
    (perimeter t > circumradius t + inradius t) ∨
    (perimeter t ≤ circumradius t + inradius t) ∨
    (perimeter t / 6 < circumradius t + inradius t ∧ circumradius t + inradius t < 6 * perimeter t)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_not_always_true_l624_62402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l624_62464

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Define the set of natural numbers N
def N : Set ℝ := {x : ℝ | ∃ n : ℕ, x = n}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l624_62464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ones_sum_equals_product_l624_62426

theorem min_ones_sum_equals_product (n : ℕ) (h : n = 100) : 
  ∃ (r : ℕ) (a : ℕ → ℕ), 
    (∀ i, i ∈ Finset.range n → a i > 0) ∧ 
    (Finset.sum (Finset.range n) a = Finset.prod (Finset.range n) a) ∧
    ((Finset.range n).filter (λ i => a i = 1)).card = r ∧
    (∀ s : ℕ, s < r → 
      ¬∃ (b : ℕ → ℕ), 
        (∀ i, i ∈ Finset.range n → b i > 0) ∧
        (Finset.sum (Finset.range n) b = Finset.prod (Finset.range n) b) ∧
        ((Finset.range n).filter (λ i => b i = 1)).card = s) ∧
    r = 95 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ones_sum_equals_product_l624_62426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y1_less_than_y2_l624_62480

noncomputable def inverse_proportion (x : ℝ) : ℝ := -3 / x

theorem y1_less_than_y2 (y₁ y₂ : ℝ) 
  (h₁ : y₁ = inverse_proportion 1)
  (h₂ : y₂ = inverse_proportion 2) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y1_less_than_y2_l624_62480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_EG_GF_l624_62461

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define points D, E, F, G
noncomputable def D (triangle : Triangle) : ℝ × ℝ := sorry
noncomputable def E (triangle : Triangle) : ℝ × ℝ := sorry
noncomputable def F (triangle : Triangle) : ℝ × ℝ := sorry
noncomputable def G (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the conditions
def conditions (triangle : Triangle) : Prop :=
  let D := D triangle
  let E := E triangle
  let F := F triangle
  let G := G triangle
  -- BD:DC = 2:1
  ∃ (t : ℝ), D = ((2 * triangle.C.1 + triangle.B.1) / 3, (2 * triangle.C.2 + triangle.B.2) / 3)
  -- AB = 15
  ∧ Real.sqrt ((triangle.B.1 - triangle.A.1)^2 + (triangle.B.2 - triangle.A.2)^2) = 15
  -- AC = 18
  ∧ Real.sqrt ((triangle.C.1 - triangle.A.1)^2 + (triangle.C.2 - triangle.A.2)^2) = 18
  -- E is on AC
  ∧ ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 1 ∧ E = (s * triangle.C.1 + (1 - s) * triangle.A.1, s * triangle.C.2 + (1 - s) * triangle.A.2)
  -- F is on AB
  ∧ ∃ (r : ℝ), 0 ≤ r ∧ r ≤ 1 ∧ F = (r * triangle.B.1 + (1 - r) * triangle.A.1, r * triangle.B.2 + (1 - r) * triangle.A.2)
  -- G is the intersection of EF and AD
  ∧ ∃ (u v : ℝ), G = (u * E.1 + (1 - u) * F.1, u * E.2 + (1 - u) * F.2)
               ∧ G = (v * D.1 + (1 - v) * triangle.A.1, v * D.2 + (1 - v) * triangle.A.2)
  -- AE = 3AF
  ∧ Real.sqrt ((E.1 - triangle.A.1)^2 + (E.2 - triangle.A.2)^2) = 3 * Real.sqrt ((F.1 - triangle.A.1)^2 + (F.2 - triangle.A.2)^2)

-- Theorem statement
theorem ratio_EG_GF (triangle : Triangle) (h : conditions triangle) :
  let E := E triangle
  let G := G triangle
  let F := F triangle
  Real.sqrt ((E.1 - G.1)^2 + (E.2 - G.2)^2) / Real.sqrt ((G.1 - F.1)^2 + (G.2 - F.2)^2) = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_EG_GF_l624_62461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_on_line_l624_62400

/-- Helper function to calculate Euclidean distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Given points P, Q, and R in the xy-plane, prove that PR + RQ is minimized when R is on the line PQ -/
theorem min_distance_point_on_line (P Q R : ℝ × ℝ) :
  let px : ℝ := -1
  let py : ℝ := -2
  let qx : ℝ := 4
  let qy : ℝ := 2
  let rx : ℝ := 1
  P = (px, py) →
  Q = (qx, qy) →
  R = (rx, R.2) →
  (∀ m : ℝ, (distance P R + distance R Q) ≤ (distance P (rx, m) + distance (rx, m) Q)) ↔
  R.2 = -2/5
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_on_line_l624_62400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lake_crossing_time_l624_62438

/-- Proves the time to cross the width of a square lake given its area and the time to cross its length -/
theorem lake_crossing_time 
  (lake_area : ℝ) 
  (boat_speed : ℝ) 
  (length_crossing_time : ℝ) 
  (h1 : lake_area = 100) 
  (h2 : boat_speed = 10) 
  (h3 : length_crossing_time = 2) : 
  (Real.sqrt lake_area / boat_speed) * 60 = 60 := by
  sorry

#check lake_crossing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lake_crossing_time_l624_62438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_combined_set_l624_62462

def is_three_digit_prime (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ Nat.Prime n

def is_odd_multiple_of_11_less_than_200 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 11 * (2 * k + 1) ∧ n < 200

def combined_set : Set ℕ :=
  {n | is_three_digit_prime n ∨ is_odd_multiple_of_11_less_than_200 n}

theorem range_of_combined_set :
  ∃ max min : ℕ, max ∈ combined_set ∧ min ∈ combined_set ∧
    (∀ n ∈ combined_set, min ≤ n ∧ n ≤ max) ∧ max - min = 896 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_combined_set_l624_62462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_negative_sufficient_not_necessary_l624_62471

-- Define the function f(x) = m + log₂x
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m + Real.log x / Real.log 2

-- Define what it means for f to have a zero in its domain
def has_zero (m : ℝ) : Prop := ∃ x : ℝ, x ≥ 1 ∧ f m x = 0

-- Theorem statement
theorem m_negative_sufficient_not_necessary :
  (∀ m : ℝ, m < 0 → has_zero m) ∧
  ¬(∀ m : ℝ, has_zero m → m < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_negative_sufficient_not_necessary_l624_62471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_half_l624_62452

/-- If the terminal side of angle α passes through the point (1, -2),
    then cos(α + π/2) = 2√5/5 -/
theorem cos_alpha_plus_pi_half (α : ℝ) (h : ∃ (r : ℝ), r • (1, -2) = (Real.cos α, Real.sin α)) :
  Real.cos (α + π/2) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_half_l624_62452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_votes_difference_l624_62413

def total_votes : ℝ := 350

theorem votes_difference (votes_against : ℝ) (votes_for : ℝ) 
  (h1 : votes_against = 0.4 * total_votes)
  (h2 : votes_for > votes_against)
  (h3 : total_votes = votes_for + votes_against) :
  ∃ ε > 0, |votes_for - votes_against - 70| < ε := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_votes_difference_l624_62413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_days_l624_62475

noncomputable section

/-- The number of days A takes to complete the work alone -/
def a_days : ℝ := 12

/-- The number of days A works before B joins -/
def a_solo_days : ℝ := 3

/-- The number of days A and B work together to complete the remaining work -/
def ab_days : ℝ := 3

/-- The fraction of work A can complete in one day -/
noncomputable def a_work_per_day : ℝ := 1 / a_days

/-- The fraction of work completed by A before B joins -/
noncomputable def a_solo_work : ℝ := a_solo_days * a_work_per_day

/-- The fraction of work remaining when B joins -/
noncomputable def remaining_work : ℝ := 1 - a_solo_work

/-- The number of days B takes to complete the work alone -/
def b_days : ℝ := 6

theorem b_work_days : 
  a_days = 12 → 
  a_solo_days = 3 → 
  ab_days = 3 → 
  remaining_work = ab_days * (a_work_per_day + 1 / b_days) → 
  b_days = 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_days_l624_62475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_difference_approx_l624_62441

noncomputable section

-- Define the salaries and rates
def b_salary : ℝ := 100
def a_salary : ℝ := b_salary * 0.8
def tax_rate : ℝ := 0.1
def a_bonus_rate : ℝ := 0.05
def b_bonus_rate : ℝ := 0.07

-- Calculate after-tax salaries
def a_after_tax : ℝ := a_salary * (1 - tax_rate)
def b_after_tax : ℝ := b_salary * (1 - tax_rate)

-- Calculate final salaries (after tax and bonus)
def a_final : ℝ := a_after_tax * (1 + a_bonus_rate)
def b_final : ℝ := b_after_tax * (1 + b_bonus_rate)

-- Define the percentage difference
def percentage_difference : ℝ := (b_final - a_final) / a_final * 100

-- Theorem statement
theorem salary_difference_approx : 
  27.37 < percentage_difference ∧ percentage_difference < 27.39 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_difference_approx_l624_62441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_quintic_equation_l624_62417

theorem solutions_quintic_equation :
  let f : ℂ → ℂ := λ z => z^5 - 5*z^3 + 6*z
  {z : ℂ | f z = 0} = {0, -Complex.I * Real.sqrt 2, Complex.I * Real.sqrt 2, 
                         -Complex.I * Real.sqrt 3, Complex.I * Real.sqrt 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_quintic_equation_l624_62417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_catch_up_equation_problem_equation_correct_l624_62416

/-- Represents the speeds of horses and the head start time -/
structure HorseRace where
  fast_speed : ℝ
  slow_speed : ℝ
  head_start : ℝ

/-- The time it takes for the faster horse to catch up with the slower horse -/
noncomputable def catch_up_time (race : HorseRace) : ℝ :=
  (race.slow_speed * race.head_start) / (race.fast_speed - race.slow_speed)

/-- Theorem stating that the catch-up time satisfies the equation -/
theorem catch_up_equation (race : HorseRace) :
  (race.fast_speed - race.slow_speed) * (catch_up_time race) = race.slow_speed * race.head_start := by
  sorry

/-- The specific horse race from the problem -/
def problem_race : HorseRace :=
  { fast_speed := 240
    slow_speed := 150
    head_start := 12 }

/-- Theorem stating that the equation in option D is correct for the problem -/
theorem problem_equation_correct :
  240 * (catch_up_time problem_race) - 150 * (catch_up_time problem_race) = 150 * 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_catch_up_equation_problem_equation_correct_l624_62416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_age_calculation_l624_62485

theorem staff_age_calculation (num_students : ℕ) (avg_age : ℝ) (new_avg : ℝ) (staff_age : ℝ) :
  num_students = 32 →
  avg_age = 16 →
  new_avg = avg_age + 1 →
  (num_students * avg_age + staff_age) / (num_students + 1) = new_avg →
  staff_age = 49 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_age_calculation_l624_62485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_count_l624_62447

theorem marble_count (total non_pink non_yellow non_violet : ℝ) :
  non_pink = 10 →
  non_yellow = 12 →
  non_violet = 5 →
  total = (total - non_pink) + (total - non_yellow) + (total - non_violet) →
  total = 13.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_count_l624_62447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l624_62459

noncomputable section

-- Define the triangle ABC
def A : ℝ × ℝ := (-1, 4)
def B : ℝ × ℝ := (-2, -1)
def C : ℝ × ℝ := (2, 3)

-- Define the equation of a line
def is_on_line (x y a b c : ℝ) : Prop := a * x + b * y + c = 0

-- Define the area of a triangle
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ := 
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * abs ((x₁ - x₃) * (y₂ - y₁) - (x₁ - x₂) * (y₃ - y₁))

-- Theorem statement
theorem triangle_properties :
  (∀ x y, is_on_line x y 1 1 (-1) ↔ 
    is_on_line x y 1 (-1) 0 ∧ 
    ((x - A.1)^2 + (y - A.2)^2 = (x - C.1)^2 + (y - C.2)^2)) ∧
  triangle_area A B C = 8 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l624_62459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_weight_correlation_l624_62468

-- Define the relationships
def square_area_side_length : ℝ → ℝ := λ s => s^2

-- For height_weight_relation, we'll use a Prop instead of a function
def height_weight_relation : ℝ → ℝ → Prop := λ h w => True

def distance_time_relation : ℝ → ℝ → ℝ := λ v t => v * t

noncomputable def sphere_volume_radius : ℝ → ℝ := λ r => (4/3) * Real.pi * r^3

-- Define what it means for a relation to be functional
def is_functional (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y₁ y₂, f x y₁ ∧ f x y₂ → y₁ = y₂

-- Define what it means for a relation to be a correlation
def is_correlation (f : ℝ → ℝ → Prop) : Prop :=
  ∃ x₁ x₂ y₁ y₂, x₁ < x₂ ∧ y₁ < y₂ ∧ f x₁ y₁ ∧ f x₂ y₂

-- Theorem statement
theorem height_weight_correlation :
  is_correlation height_weight_relation ∧
  ¬is_functional height_weight_relation ∧
  is_functional (λ s a => a = square_area_side_length s) ∧
  is_functional (λ t d => d = distance_time_relation 1 t) ∧
  is_functional (λ r v => v = sphere_volume_radius r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_weight_correlation_l624_62468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l624_62435

noncomputable section

/-- A line passing through point P(m,0) with parametric equation x = m + (√3/2)t, y = (1/2)t -/
def line (m : ℝ) (t : ℝ) : ℝ × ℝ :=
  (m + (Real.sqrt 3 / 2) * t, (1 / 2) * t)

/-- Curve C with polar equation ρ = 2cosθ -/
def curve (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 = 2*p.1

/-- The intersection points of the line and the curve -/
def intersection (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line m t ∧ curve p}

/-- The product of the distances from P to the intersection points is 2 -/
def distance_product (m : ℝ) : Prop :=
  ∃ A B, A ∈ intersection m ∧ B ∈ intersection m ∧ A ≠ B ∧
    ((A.1 - m)^2 + A.2^2) * ((B.1 - m)^2 + B.2^2) = 2

/-- The theorem to be proved -/
theorem intersection_distance_product (m : ℝ) :
  distance_product m ↔ m = 2 ∨ m = -1 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l624_62435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rect_and_distance_range_l624_62420

open Real

noncomputable def A_polar : ℝ × ℝ := (sqrt 3, π / 6)
noncomputable def B_polar : ℝ × ℝ := (sqrt 3, π / 2)

noncomputable def C_polar (θ : ℝ) : ℝ := 2 * cos (θ - π / 3)

noncomputable def polar_to_rect (r θ : ℝ) : ℝ × ℝ := (r * cos θ, r * sin θ)

noncomputable def A_rect : ℝ × ℝ := polar_to_rect A_polar.1 A_polar.2
noncomputable def B_rect : ℝ × ℝ := polar_to_rect B_polar.1 B_polar.2

noncomputable def C_rect (θ : ℝ) : ℝ × ℝ := (1/2 + cos θ, sqrt 3 / 2 + sin θ)

def dist_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem polar_to_rect_and_distance_range :
  A_rect = (3/2, sqrt 3 / 2) ∧
  B_rect = (0, sqrt 3) ∧
  (∀ θ : ℝ, ∃ M : ℝ × ℝ, M = C_rect θ) ∧
  (∀ M : ℝ × ℝ, (∃ θ : ℝ, M = C_rect θ) →
    2 ≤ dist_squared M A_rect + dist_squared M B_rect ∧
    dist_squared M A_rect + dist_squared M B_rect ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rect_and_distance_range_l624_62420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_power_theorem_l624_62405

open Real MeasureTheory

/-- Recursive definition of fₙ -/
noncomputable def f (n : ℕ) (a : ℝ) (f₀ : ℝ → ℝ) : ℝ → ℝ :=
  match n with
  | 0 => f₀
  | n + 1 => fun x => ∫ t in a..x, f n a f₀ t

/-- The main theorem -/
theorem integral_power_theorem (n : ℕ) (a : ℝ) (f₀ : ℝ → ℝ) (x : ℝ) 
    (hf : Continuous f₀) (hn : n > 0) :
    f n a f₀ x = ((-1)^(n-1) / (n-1).factorial : ℝ) * 
      ∫ t in a..x, f₀ t * (t - x)^(n-1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_power_theorem_l624_62405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_count_l624_62483

def r (a N : ℕ) : ℕ := a % N

theorem remainder_count :
  let count := (Finset.range 1000000).filter (λ n => r (n + 1) 1000 > r (n + 1) 1001) |>.card
  count = 499500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_count_l624_62483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wealth_ratio_is_point_nine_l624_62403

/-- Represents the wealth distribution and taxation in two countries --/
structure WorldEconomy where
  total_population : ℝ
  total_wealth : ℝ
  x_population_ratio : ℝ
  x_wealth_ratio : ℝ
  y_population_ratio : ℝ
  y_wealth_ratio : ℝ
  x_tax_rate : ℝ

/-- Calculates the ratio of wealth per citizen between Country X and Country Y --/
noncomputable def wealth_ratio (w : WorldEconomy) : ℝ :=
  let x_wealth_after_tax := w.x_wealth_ratio * (1 - w.x_tax_rate) * w.total_wealth
  let x_wealth_per_citizen := x_wealth_after_tax / (w.x_population_ratio * w.total_population)
  let y_wealth_per_citizen := (w.y_wealth_ratio * w.total_wealth) / (w.y_population_ratio * w.total_population)
  x_wealth_per_citizen / y_wealth_per_citizen

/-- Theorem stating that the wealth ratio is 0.9 given the specified conditions --/
theorem wealth_ratio_is_point_nine (w : WorldEconomy)
  (h1 : w.x_population_ratio = 0.4)
  (h2 : w.x_wealth_ratio = 0.6)
  (h3 : w.y_population_ratio = 0.2)
  (h4 : w.y_wealth_ratio = 0.3)
  (h5 : w.x_tax_rate = 0.1) :
  wealth_ratio w = 0.9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wealth_ratio_is_point_nine_l624_62403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_equation_solutions_l624_62408

theorem quadruple_equation_solutions :
  {(x, y, z, n) : ℕ × ℕ × ℕ × ℕ | x^2 + y^2 + z^2 + 1 = 2^n} =
    {(0,0,0,0), (1,0,0,1), (0,1,0,1), (0,0,1,1), (1,1,1,2)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_equation_solutions_l624_62408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_denominator_of_factorial_div_power_l624_62414

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem denominator_of_factorial_div_power : 
  ∃ (k : ℕ), (factorial 100 / 6^100 : ℚ) = (k : ℚ) / (2^3 * 3^52 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_denominator_of_factorial_div_power_l624_62414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l624_62450

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / (2^x + a) - 1/2

theorem f_properties (a k : ℝ) :
  (∀ x, f a x = -f a (-x)) →  -- f is odd
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f a (2 - x) + f a (k - x^2) > 0) →
  (Set.range (f a) = Set.Ioo (-1/2 : ℝ) (1/2)) ∧
  (k < -9/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l624_62450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_bill_calculation_l624_62481

/-- Calculates the number of $10 bills needed to pay for groceries -/
theorem grocery_bill_calculation (soup_price soup_quantity bread_price bread_quantity
                                  cereal_price cereal_quantity milk_price milk_quantity : ℕ) : 
  soup_price = 2 →
  soup_quantity = 6 →
  bread_price = 5 →
  bread_quantity = 2 →
  cereal_price = 3 →
  cereal_quantity = 2 →
  milk_price = 4 →
  milk_quantity = 2 →
  (((soup_price * soup_quantity + bread_price * bread_quantity + 
     cereal_price * cereal_quantity + milk_price * milk_quantity : ℕ) + 9) / 10 : ℕ) = 4 :=
by
  sorry

#check grocery_bill_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_bill_calculation_l624_62481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_n_values_l624_62454

/-- A right triangle in the coordinate plane with legs parallel to the x and y axes. -/
structure RightTriangle where
  a : ℝ  -- x-coordinate of the right angle
  b : ℝ  -- y-coordinate of the right angle
  c : ℝ  -- half the length of the vertical leg
  d : ℝ  -- half the length of the horizontal leg

/-- The slopes of the medians to the midpoints of the legs of a right triangle. -/
noncomputable def median_slopes (t : RightTriangle) : ℝ × ℝ :=
  (t.c / (2 * t.d), 2 * t.c / t.d)

/-- The condition for the medians to lie on the given lines. -/
def medians_on_lines (t : RightTriangle) (n : ℝ) : Prop :=
  let (s1, s2) := median_slopes t
  (s1 = 4 ∧ s2 = n) ∨ (s1 = n ∧ s2 = 4)

/-- The main theorem stating that there are exactly two values of n for which
    a right triangle with the given properties exists. -/
theorem exactly_two_n_values :
  ∃! (s : Finset ℝ), s.card = 2 ∧
    (∀ n, n ∈ s ↔ ∃ t : RightTriangle, medians_on_lines t n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_n_values_l624_62454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_plane_and_passes_through_point_l624_62494

/-- A line in 3D space defined by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ
  z : ℝ → ℝ

/-- A plane in 3D space defined by three points -/
structure Plane where
  a : ℝ × ℝ × ℝ
  p : ℝ × ℝ × ℝ
  q : ℝ × ℝ × ℝ

/-- Check if a point lies on a parametric line -/
def pointOnLine (line : ParametricLine) (point : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, line.x t = point.1 ∧ line.y t = point.2.1 ∧ line.z t = point.2.2

/-- Calculate the normal vector of a plane -/
def normalVector (plane : Plane) : ℝ × ℝ × ℝ :=
  let ap := (plane.p.1 - plane.a.1, plane.p.2.1 - plane.a.2.1, plane.p.2.2 - plane.a.2.2)
  let aq := (plane.q.1 - plane.a.1, plane.q.2.1 - plane.a.2.1, plane.q.2.2 - plane.a.2.2)
  (ap.2.1 * aq.2.2 - ap.2.2 * aq.2.1, ap.2.2 * aq.1 - ap.1 * aq.2.2, ap.1 * aq.2.1 - ap.2.1 * aq.1)

/-- Check if a line is perpendicular to a plane -/
def linePerpendicularToPlane (line : ParametricLine) (plane : Plane) : Prop :=
  let n := normalVector plane
  let v := (line.x 1 - line.x 0, line.y 1 - line.y 0, line.z 1 - line.z 0)
  n.1 * v.1 + n.2.1 * v.2.1 + n.2.2 * v.2.2 = 0

theorem line_perpendicular_to_plane_and_passes_through_point :
  let line : ParametricLine := {
    x := λ t => -2 + 4 * t,
    y := λ t => 6 * t,
    z := λ t => 3 - 5 * t
  }
  let plane : Plane := {
    a := (-3, 0, 1),
    p := (-1, 2, 5),
    q := (3, -4, 1)
  }
  let m : ℝ × ℝ × ℝ := (-2, 0, 3)
  pointOnLine line m ∧ linePerpendicularToPlane line plane := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_plane_and_passes_through_point_l624_62494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l624_62490

/-- The speed of a train crossing a bridge -/
noncomputable def train_speed (train_length bridge_length crossing_time : ℝ) : ℝ :=
  (train_length + bridge_length) / crossing_time

/-- Theorem stating the speed of the train -/
theorem train_speed_calculation (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 100)
  (h2 : bridge_length = 300)
  (h3 : crossing_time = 12) :
  ∃ ε > 0, |train_speed train_length bridge_length crossing_time - 100/3| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l624_62490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_alpha_l624_62495

theorem sin_double_alpha (α : ℝ) (h : Real.sin (α + π/4) = Real.sqrt 2/3) : 
  Real.sin (2*α) = -5/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_alpha_l624_62495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l624_62440

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2)

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, ∃ x : ℝ, x ≠ 0 ∧ f x = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l624_62440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l624_62407

theorem simplify_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  4 * x^(1/4 : ℝ) * (-3 * x^(1/4 : ℝ) * y^(-(1/3) : ℝ)) / (-6 * (x^(-(1/2) : ℝ) * y^(-(2/3) : ℝ))) = 2 * x * y^(1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l624_62407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l624_62491

theorem solve_exponential_equation :
  ∃ x : ℝ, (5 : ℝ)^(x + 6) = (625 : ℝ)^x ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l624_62491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_4pi_l624_62470

-- Define the equation
def tanEquation (x : ℝ) : Prop := Real.tan x ^ 2 - 7 * Real.tan x + 2 = 0

-- Define the range of x
def inRange (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2 * Real.pi

-- Theorem statement
theorem sum_of_roots_is_4pi :
  ∃ (S : Finset ℝ), (∀ x ∈ S, tanEquation x ∧ inRange x) ∧ 
  (∀ x, tanEquation x ∧ inRange x → x ∈ S) ∧
  S.sum id = 4 * Real.pi :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_4pi_l624_62470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_larger_than_a_l624_62478

def harmonic_sum (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (λ i => 1 / (↑i + 1 : ℚ))

theorem b_larger_than_a :
  (1 / 2016 : ℚ) * harmonic_sum 2016 > (1 / 2015 : ℚ) * harmonic_sum 2015 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_larger_than_a_l624_62478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_arrangements_l624_62493

/-- The number of students -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of subjects -/
def subjects : ℕ := 4

/-- The number of subjects student A can participate in -/
def subjects_for_A : ℕ := 3

theorem competition_arrangements :
  let arrangements_without_A := Nat.factorial selected_students
  let arrangements_with_A := subjects_for_A * Nat.choose (total_students - 1) (selected_students - 1) * Nat.factorial (selected_students - 1)
  arrangements_without_A + arrangements_with_A = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_arrangements_l624_62493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sums_2015_l624_62439

def consecutive_sum (start : ℕ) (length : ℕ) : ℕ :=
  length * (2 * start + length - 1) / 2

theorem consecutive_sums_2015 :
  (Finset.filter (fun seq => seq.2 < 10 ∧ consecutive_sum seq.1 seq.2 = 2015) (Finset.product (Finset.range 2016) (Finset.range 10))).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sums_2015_l624_62439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_theorem_l624_62479

/-- Represents the loan details and repayment --/
structure LoanDetails where
  interest_rate : ℚ  -- Annual interest rate as a percentage
  duration : ℚ       -- Loan duration in years
  total_repaid : ℚ   -- Total amount repaid after the loan duration
  borrowed : ℚ       -- Amount originally borrowed

/-- Calculates the total amount to be repaid for a given loan --/
def calculate_total_repayment (loan : LoanDetails) : ℚ :=
  loan.borrowed * (1 + (loan.interest_rate / 100) * loan.duration)

/-- Theorem stating that given the loan conditions, the borrowed amount is approximately 5526 --/
theorem borrowed_amount_theorem (loan : LoanDetails) 
  (h1 : loan.interest_rate = 6)
  (h2 : loan.duration = 9)
  (h3 : loan.total_repaid = 8510)
  (h4 : calculate_total_repayment loan = loan.total_repaid) :
  ∃ ε > 0, |loan.borrowed - 5526| < ε := by
  sorry

#check borrowed_amount_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_theorem_l624_62479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_half_l624_62451

/-- A unit cube with opposite vertices A and B -/
structure UnitCube where
  A : Fin 3 → ℝ
  B : Fin 3 → ℝ
  is_unit : ∀ i, abs (B i - A i) = 1

/-- A sphere inside the unit cube -/
structure SphereInCube (cube : UnitCube) where
  center : Fin 3 → ℝ
  radius : ℝ
  center_in_cube : ∀ i, cube.A i ≤ center i ∧ center i ≤ cube.B i
  tangent_to_A_faces : ∀ i, center i = radius
  tangent_to_B_edges : ∀ i, abs (cube.B i - center i) = radius

/-- The radius of the sphere is 1/2 -/
theorem sphere_radius_is_half (cube : UnitCube) (sphere : SphereInCube cube) :
  sphere.radius = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_half_l624_62451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisible_by_primes_l624_62422

def consecutive_even_product (n : Nat) : Nat :=
  Finset.prod (Finset.range (n / 2 + 1)) (fun i => 2 * (i + 1))

theorem smallest_n_divisible_by_primes : 
  ∀ m : Nat, m ≤ 4056 → m % 2 = 0 → 
    (consecutive_even_product m % (1997 * 2011 * 2027) = 0 → m = 4056) ∧
    (m < 4056 → consecutive_even_product m % (1997 * 2011 * 2027) ≠ 0) :=
by
  sorry

#check smallest_n_divisible_by_primes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisible_by_primes_l624_62422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_are_odd_l624_62489

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x

noncomputable def h (a b c d : ℝ) (x : ℝ) : ℝ := (a * x^3) / (b * x^4 + c * x^2 + d)

-- Theorem stating that all functions are odd
theorem functions_are_odd (a b c d : ℝ) :
  (∀ x, f a (-x) = -(f a x)) ∧
  (∀ x, g a b (-x) = -(g a b x)) ∧
  (∀ x, h a b c d (-x) = -(h a b c d x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_are_odd_l624_62489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_path_exists_l624_62476

/-- Represents a square on a chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8
deriving Inhabited, DecidableEq

/-- Represents a chessboard -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Checks if two squares have the same color -/
def sameColor (s1 s2 : Square) : Bool :=
  (s1.row + s1.col) % 2 = (s2.row + s2.col) % 2

/-- Represents a path of a rook on a chessboard -/
def RookPath := List Square

/-- Checks if a path is valid for a rook -/
def isValidRookPath (path : RookPath) : Bool :=
  path.tail.all (fun s2 => 
    let s1 := path.head!
    s1.row = s2.row ∨ s1.col = s2.col)

/-- Checks if a path visits all squares exactly once, except for the end square which is visited twice -/
def visitsAllSquaresOnce (path : RookPath) : Bool :=
  let uniqueSquares := path.toFinset
  uniqueSquares.card = 64 ∧ path.length = 65

/-- Main theorem: For any two squares of the same color, there exists a valid rook path
    that visits all squares once and the second square twice -/
theorem rook_path_exists (s1 s2 : Square) (h : sameColor s1 s2) :
  ∃ (path : RookPath), path.head? = some s1 ∧ path.getLast? = some s2 ∧
    isValidRookPath path ∧ visitsAllSquaresOnce path := by
  sorry

#check rook_path_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_path_exists_l624_62476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lt_b_lt_c_l624_62482

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log 6
noncomputable def b : ℝ := Real.log 5 / Real.log 10
noncomputable def c : ℝ := 2 ^ (1/10 : ℝ)

-- State the theorem
theorem a_lt_b_lt_c : a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lt_b_lt_c_l624_62482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_i_equals_neg_one_plus_i_l624_62458

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the function f
noncomputable def f (x : ℂ) : ℂ := (x^5 + 2*x^3 + x) / (x + 1)

-- Theorem statement
theorem f_of_i_equals_neg_one_plus_i :
  i^2 = -1 → f i = -1 + i := by
  intro h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_i_equals_neg_one_plus_i_l624_62458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_table_l624_62477

/-- Represents a move on the table -/
inductive Move
  | DoubleRow (row : Fin 8)
  | SubtractColumn (col : Fin 5)

/-- Represents the state of the table -/
def Table := Fin 8 → Fin 5 → ℕ

/-- Applies a move to the table -/
def applyMove (t : Table) (m : Move) : Table :=
  match m with
  | Move.DoubleRow r => fun i j => if i = r then 2 * t i j else t i j
  | Move.SubtractColumn c => fun i j => if j = c then (t i j).pred else t i j

/-- Checks if all elements in the table are zero -/
def allZero (t : Table) : Prop :=
  ∀ i j, t i j = 0

theorem exists_zero_table (initial : Table) :
  ∃ (moves : List Move), allZero (moves.foldl applyMove initial) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_table_l624_62477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_winning_conditions_l624_62497

/-- Represents the possible moves in the game -/
inductive Move
| Add  : Move  -- represents x + 1
| Double : Move  -- represents 2x

/-- Represents a player in the game -/
inductive Player
| Ariane : Player
| Berenice : Player
deriving DecidableEq

/-- The game state -/
structure GameState (n : ℕ) where
  current : ℕ  -- current residue class
  turn : Player  -- whose turn it is

/-- Defines a valid move in the game -/
def validMove (n : ℕ) (state : GameState n) (move : Move) : Prop :=
  match move with
  | Move.Add => (state.current + 1) % n ≠ state.current % n
  | Move.Double => (2 * state.current) % n ≠ state.current % n

/-- Defines the winning condition for Ariane -/
def arianeWins (n : ℕ) (state : GameState n) : Prop :=
  state.current % n = 0

/-- Defines a winning strategy for a player -/
def hasWinningStrategy (n : ℕ) (player : Player) : Prop :=
  ∀ (state : GameState n),
    (state.turn = player) →
    (∃ (move : Move), validMove n state move ∧
      (if player = Player.Ariane
       then arianeWins n ⟨((match move with
         | Move.Add => state.current + 1
         | Move.Double => 2 * state.current) % n), Player.Berenice⟩
       else ¬∃ (m : Move), arianeWins n ⟨((match m with
         | Move.Add => (match move with
           | Move.Add => state.current + 1
           | Move.Double => 2 * state.current) + 1
         | Move.Double => 2 * (match move with
           | Move.Add => state.current + 1
           | Move.Double => 2 * state.current)) % n), Player.Ariane⟩))

/-- The main theorem stating the winning conditions for each player -/
theorem game_winning_conditions (n : ℕ) (h : n ≥ 2) :
  (hasWinningStrategy n Player.Ariane ↔ n ∈ ({2, 4, 8} : Finset ℕ)) ∧
  (hasWinningStrategy n Player.Berenice ↔ n ∉ ({2, 4, 8} : Finset ℕ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_winning_conditions_l624_62497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l624_62432

/-- The radius of a circle inscribed in a sector that is one-third of a circle --/
theorem inscribed_circle_radius (R : ℝ) (h : R = 5) : 
  ∃ r : ℝ, r > 0 ∧ 
     r + r * Real.sqrt 3 = R ∧
     r = (R * (Real.sqrt 3 - 1)) / (Real.sqrt 3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l624_62432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_ratio_theorem_l624_62463

theorem sphere_ratio_theorem (r₁ r₂ r₃ : ℝ) (h : r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) :
  (4 * Real.pi * r₁^2 : ℝ) / (4 * Real.pi * r₂^2) = 1 / 4 ∧
  (4 * Real.pi * r₂^2 : ℝ) / (4 * Real.pi * r₃^2) = 4 / 9 →
  ((4 / 3) * Real.pi * r₁^3 : ℝ) / ((4 / 3) * Real.pi * r₂^3) = 1 / 8 ∧
  ((4 / 3) * Real.pi * r₂^3 : ℝ) / ((4 / 3) * Real.pi * r₃^3) = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_ratio_theorem_l624_62463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pinedale_bus_distance_l624_62409

/-- Calculates the distance traveled by a bus given its speed, stop interval, and number of stops -/
noncomputable def bus_distance (speed : ℝ) (stop_interval : ℝ) (num_stops : ℕ) : ℝ :=
  speed * (stop_interval * (num_stops : ℝ) / 60)

/-- Theorem: A bus traveling at 60 km/h with stops every 5 minutes will cover 30 km in 6 stops -/
theorem pinedale_bus_distance :
  bus_distance 60 5 6 = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pinedale_bus_distance_l624_62409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_l624_62442

-- Define the floor dimensions
noncomputable def floor_length : ℝ := 12
noncomputable def floor_width : ℝ := 15

-- Define the tile dimensions
noncomputable def tile_side : ℝ := 2

-- Define the radius of the quarter circles
noncomputable def circle_radius : ℝ := 1

-- Calculate the number of tiles
noncomputable def num_tiles : ℝ := (floor_length / tile_side) * (floor_width / tile_side)

-- Calculate the area of a single tile
noncomputable def tile_area : ℝ := tile_side * tile_side

-- Calculate the white area in a single tile
noncomputable def white_area_per_tile : ℝ := Real.pi * circle_radius^2

-- Calculate the shaded area in a single tile
noncomputable def shaded_area_per_tile : ℝ := tile_area - white_area_per_tile

-- State the theorem
theorem total_shaded_area :
  num_tiles * shaded_area_per_tile = 180 - 45 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_l624_62442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_theorem_l624_62433

noncomputable def sum_odd_terms (x : ℝ) : ℝ := sorry
noncomputable def sum_even_terms (x : ℝ) : ℝ := sorry

theorem binomial_expansion_theorem (n : ℕ) (x : ℝ) 
  (P Q : ℝ) (h1 : P = sum_odd_terms ((1 + x)^n)) (h2 : Q = sum_even_terms ((1 + x)^n)) : 
  (1 - x^2)^n = P^2 - Q^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_theorem_l624_62433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_value_l624_62487

def letter_value (n : ℕ) : ℤ :=
  match n % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 1
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -1
  | _ => 0

def alphabet_position (c : Char) : ℕ :=
  (c.toNat - 'A'.toNat + 1)

def word_value (word : String) : ℤ :=
  word.toList.map (fun c => letter_value (alphabet_position c)) |>.sum

theorem algebra_value : word_value "ALGEBRA" = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_value_l624_62487
