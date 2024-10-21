import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l542_54220

/-- Definition of the ellipse E -/
def E (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Definition of the vector OP -/
def OP (x y : ℝ) : ℝ × ℝ := (x, 2*y)

/-- Definition of the vector OQ -/
def OQ (x y : ℝ) : ℝ × ℝ := (x, 2*y)

/-- Perpendicularity condition for OP and OQ -/
def perpendicular (p q : ℝ × ℝ) : Prop :=
  p.1 * q.1 + p.2 * q.2 = 0

/-- Area of a triangle given coordinates of its vertices -/
noncomputable def triangleArea (x1 y1 x2 y2 : ℝ) : ℝ :=
  (1/2) * abs (x1 * y2 - x2 * y1)

/-- Main theorem statement -/
theorem constant_triangle_area
  (x1 y1 x2 y2 : ℝ)
  (h1 : E x1 y1)
  (h2 : E x2 y2)
  (h3 : perpendicular (OP x1 y1) (OQ x2 y2)) :
  triangleArea x1 y1 x2 y2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l542_54220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sperner_theorem_l542_54274

/-- Given a finite set S with n elements and a collection F of subsets of S
    such that no element of F is a subset of another element in F,
    the cardinality of F is at most the binomial coefficient C(n, ⌊n/2⌋). -/
theorem sperner_theorem {α : Type*} (n : ℕ) (S : Finset α) (F : Finset (Finset α)) :
  S.card = n →
  (∀ A B : Finset α, A ∈ F → B ∈ F → A ⊆ B → A = B) →
  F.card ≤ Nat.choose n (n / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sperner_theorem_l542_54274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_with_equal_differences_l542_54247

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the altitude foot A₁
noncomputable def altitudeFoot (t : Triangle) : ℝ × ℝ := sorry

-- Define that A₁ is an internal point of BC
def isInternalPoint (t : Triangle) : Prop := sorry

-- Define the length of a segment
noncomputable def segmentLength (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem exists_triangle_with_equal_differences :
  ∃ (t : Triangle),
    isInternalPoint t ∧
    let A₁ := altitudeFoot t
    |segmentLength t.A t.B - segmentLength t.A t.C| ≥ |segmentLength A₁ t.B - segmentLength A₁ t.C| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_with_equal_differences_l542_54247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l542_54266

noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + Real.exp (-x * Real.log 2)

theorem f_properties : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂) ∧
  (∃! x : ℝ, x > 0 ∧ f x = 5 * Real.exp (-x * Real.log 2) + 3 ∧ x = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l542_54266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l542_54232

open Real

/-- The function f(x) = -2xlnx --/
noncomputable def f (x : ℝ) : ℝ := -2 * x * log x

/-- The function g(x) = -x^3 + 3xm --/
def g (x m : ℝ) : ℝ := -x^3 + 3 * x * m

/-- The theorem stating the range of m --/
theorem range_of_m (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (1/exp 1) (exp 1) ∧ x₂ ∈ Set.Icc (1/exp 1) (exp 1) ∧
    f x₁ = g x₁ m ∧ f x₂ = g x₂ m) →
  m ∈ Set.Ioo (1/3) (2/3 + 1/(3 * (exp 1)^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l542_54232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l542_54275

theorem simplify_expression : 
  (Real.rpow 32 (1/3) + Real.sqrt (4 + 1/4))^2 = (2 * Real.rpow 16 (1/3) * Real.sqrt 17 + 81) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l542_54275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_meets_requirements_l542_54216

/-- Represents the fruit supermarket's grape sales scenario -/
structure GrapeSales where
  initialPurchasePrice : ℚ
  initialSellingPrice : ℚ
  initialDailySales : ℚ
  salesIncreaseRate : ℚ
  priceDecreaseRate : ℚ
  profitTarget : ℚ
  minimumSales : ℚ

/-- Calculates the daily sales volume based on price reduction -/
def dailySalesVolume (g : GrapeSales) (priceReduction : ℚ) : ℚ :=
  g.initialDailySales + (priceReduction / g.priceDecreaseRate) * g.salesIncreaseRate

/-- Calculates the daily profit based on price reduction -/
def dailyProfit (g : GrapeSales) (priceReduction : ℚ) : ℚ :=
  (g.initialSellingPrice - g.initialPurchasePrice - priceReduction) *
  (dailySalesVolume g priceReduction)

/-- Theorem stating that a price reduction of 1 yuan meets the profit target and minimum sales requirement -/
theorem price_reduction_meets_requirements (g : GrapeSales) : 
  g.initialPurchasePrice = 3 ∧ 
  g.initialSellingPrice = 5 ∧ 
  g.initialDailySales = 100 ∧ 
  g.salesIncreaseRate = 20 ∧ 
  g.priceDecreaseRate = 1/10 ∧ 
  g.profitTarget = 300 ∧ 
  g.minimumSales = 220 →
  dailyProfit g 1 = g.profitTarget ∧ dailySalesVolume g 1 ≥ g.minimumSales := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_meets_requirements_l542_54216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stair_structure_area_l542_54238

/-- Represents a level in the stair-like structure -/
structure Level where
  cubes : ℕ
  exposed_faces : ℕ → ℕ

/-- Represents the entire stair-like structure -/
structure StairStructure where
  levels : List Level
  total_cubes : ℕ
  cube_edge : ℝ

/-- Calculates the total exposed surface area of the structure -/
def total_exposed_area (s : StairStructure) : ℝ :=
  s.levels.foldl
    (fun acc level =>
      acc + (List.range level.cubes).foldl
        (fun inner_acc i => inner_acc + (level.exposed_faces i : ℝ) * s.cube_edge ^ 2)
        0)
    0

/-- The main theorem stating the total exposed area is 45 square meters -/
theorem stair_structure_area :
  ∃ (s : StairStructure),
    s.total_cubes = 18 ∧
    s.cube_edge = 1 ∧
    s.levels = [
      ⟨7, fun _ => 1⟩,
      ⟨5, fun i => if i = 0 ∨ i = 4 then 4 else 2⟩,
      ⟨4, fun i => if i = 0 ∨ i = 3 then 4 else 3⟩,
      ⟨2, fun _ => 5⟩
    ] ∧
    total_exposed_area s = 45 := by
  sorry

#eval total_exposed_area ⟨[
  ⟨7, fun _ => 1⟩,
  ⟨5, fun i => if i = 0 ∨ i = 4 then 4 else 2⟩,
  ⟨4, fun i => if i = 0 ∨ i = 3 then 4 else 3⟩,
  ⟨2, fun _ => 5⟩
], 18, 1⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stair_structure_area_l542_54238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_y_between_3_and_6_l542_54236

open BigOperators

theorem sqrt_y_between_3_and_6 : 
  (Finset.filter (fun y : ℕ => 3 < Real.sqrt (y : ℝ) ∧ Real.sqrt (y : ℝ) < 6) (Finset.range 36)).card = 26 :=
by
  -- We use Finset.range 36 instead of Finset.univ for ℤ
  -- This covers all possible integers y where 9 < y < 36
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_y_between_3_and_6_l542_54236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_pi_over_4_l542_54276

open Real

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1 - sin (2 * x)) / ((π - 4 * x)^2)

-- State the theorem
theorem limit_of_f_at_pi_over_4 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - π/4| ∧ |x - π/4| < δ → |f x - 1/8| < ε := by
  sorry

#check limit_of_f_at_pi_over_4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_pi_over_4_l542_54276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diana_wins_with_probability_one_thirty_first_l542_54260

/-- Represents the probability of a player winning in a coin-flipping game -/
def win_probability (n : ℕ) : ℚ :=
  (1 / 2) ^ (4 * n + 1)

/-- The sum of the geometric series representing Diana's win probability -/
noncomputable def diana_win_probability : ℚ :=
  ∑' n, win_probability n

/-- Theorem stating that Diana's win probability is 1/31 -/
theorem diana_wins_with_probability_one_thirty_first :
  diana_win_probability = 1 / 31 := by
  sorry

#eval (1 : ℚ) / 31

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diana_wins_with_probability_one_thirty_first_l542_54260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_polynomial_l542_54240

theorem perfect_square_polynomial (n : ℕ) : 
  (∃ m : ℕ, n^4 + 2*n^3 + 6*n^2 + 12*n + 25 = m^2) ↔ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_polynomial_l542_54240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_focal_distances_l542_54251

/-- The ellipse C defined by x²/9 + y²/4 = 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 9 + p.2^2 / 4 = 1}

/-- The foci of the ellipse C -/
noncomputable def F₁ : ℝ × ℝ := sorry
noncomputable def F₂ : ℝ × ℝ := sorry

/-- Distance between two points in ℝ² -/
noncomputable def myDist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: The maximum value of |MF₁| * |MF₂| for any point M on the ellipse C is 9 -/
theorem max_product_focal_distances :
  ∃ M ∈ C, (∀ N ∈ C, myDist M F₁ * myDist M F₂ ≥ myDist N F₁ * myDist N F₂) ∧
  myDist M F₁ * myDist M F₂ = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_focal_distances_l542_54251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l542_54270

def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

theorem function_properties (a b : ℝ) 
  (h1 : f a b 1 = -3) 
  (h2 : deriv (f a b) 1 = 0) :
  ∃ (m : ℝ), 
    (∀ x > 0, f a b x + 2 * m^2 - m ≥ 0) ∧ 
    (m ≤ -1 ∨ m ≥ 3/2) ∧
    (∀ x < 0, deriv (f a b) x > 0) ∧
    (∀ x ∈ Set.Ioo 0 1, deriv (f a b) x < 0) ∧
    (∀ x > 1, deriv (f a b) x > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l542_54270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_five_triangles_l542_54286

/-- The area of the region covered by the union of five equilateral triangles -/
theorem area_of_five_triangles (side_length : ℝ) (h_side : side_length = 4) :
  let triangle_area := (Real.sqrt 3 / 4) * side_length^2
  let total_area_without_overlap := 5 * triangle_area
  let overlap_area := 4 * ((Real.sqrt 3 / 4) * (side_length / 2)^2)
  total_area_without_overlap - overlap_area = 16 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_five_triangles_l542_54286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_pieces_bound_l542_54295

/-- Represents a rectangle on a square sheet of paper -/
structure Rectangle where
  -- We don't need to define the exact properties of a rectangle for this statement

/-- Represents a square sheet of paper with rectangles drawn on it -/
structure Sheet where
  rectangles : List Rectangle
  noOverlap : ∀ r1 r2, r1 ∈ rectangles → r2 ∈ rectangles → r1 ≠ r2 → 
    True  -- Placeholder for the non-overlap condition

/-- The number of pieces remaining after cutting out rectangles -/
def remainingPieces (s : Sheet) : ℕ :=
  0  -- Placeholder for the actual calculation

/-- Theorem: The number of remaining pieces after cutting out rectangles is at most n + 1 -/
theorem remaining_pieces_bound (s : Sheet) : 
  remainingPieces s ≤ s.rectangles.length + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_pieces_bound_l542_54295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_1_differential_equation_solution_2_differential_equation_solution_3_l542_54212

-- Problem 1
theorem differential_equation_solution_1 (y : ℝ → ℝ) (C₁ C₂ C₃ : ℝ) :
  (∀ x, y x = x^5 + (C₁ * x^2) / 2 + C₂ * x + C₃) →
  (∀ x, (deriv (deriv (deriv y))) x = 60 * x^2) :=
sorry

-- Problem 2
theorem differential_equation_solution_2 (y : ℝ → ℝ) (C₁ C₂ : ℝ) :
  (∀ x, x ≠ 3 → y x = C₁ * Real.log (|x - 3|) + C₂) →
  (∀ x, x ≠ 3 → (x - 3) * (deriv (deriv y)) x + (deriv y) x = 0) :=
sorry

-- Problem 3
theorem differential_equation_solution_3 (y : ℝ → ℝ) :
  (∀ x, y x * (deriv (deriv y)) x - ((deriv y) x)^2 = (y x)^3) →
  (y 0 = -1/2) →
  ((deriv y) 0 = 0) →
  (∀ x, x = Real.log ((|1 - Real.sqrt (2 * y x + 1)|) / (1 + Real.sqrt (2 * y x + 1))) ∨
       x = -Real.log ((|1 - Real.sqrt (2 * y x + 1)|) / (1 + Real.sqrt (2 * y x + 1)))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_1_differential_equation_solution_2_differential_equation_solution_3_l542_54212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_proposition_correct_l542_54252

-- Define the basic geometric objects
variable (α β : Type) -- Planes
variable (a b : Type) -- Lines

-- Define the geometric relations
def intersects (p q : Type) : Prop := ∃ (x : Type), x = p ∧ x = q
def parallel (l p : Type) : Prop := ¬(intersects l p)
def subset (s t : Type) : Prop := s = t
def skew (l m : Type) : Prop := ¬(intersects l m) ∧ ¬(parallel l m)

-- State the theorem
theorem only_fourth_proposition_correct :
  ∀ (α β a b : Type),
  (skew a b ∧ 
   subset a α ∧ parallel a β ∧
   subset b β ∧ parallel b α) →
  parallel α β :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_proposition_correct_l542_54252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_dog_life_difference_l542_54244

theorem cat_dog_life_difference 
  (cat_lives : ℕ)
  (mouse_lives : ℕ)
  (mouse_dog_difference : ℕ)
  (h1 : cat_lives = 9)
  (h2 : mouse_lives = 13)
  (h3 : mouse_lives = mouse_dog_difference + (mouse_lives - mouse_dog_difference))
  : cat_lives - (mouse_lives - mouse_dog_difference) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_dog_life_difference_l542_54244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_leg_sum_l542_54294

theorem similar_triangles_leg_sum (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  -- Conditions for the smaller triangle
  a₁^2 + b₁^2 = c₁^2 →  -- Right triangle
  (1/2) * a₁ * b₁ = 8 →  -- Area is 8 square inches
  c₁ = 6 →  -- Hypotenuse is 6 inches
  -- Conditions for the larger triangle
  a₂^2 + b₂^2 = c₂^2 →  -- Right triangle
  (1/2) * a₂ * b₂ = 200 →  -- Area is 200 square inches
  -- Similarity condition
  a₂ / a₁ = b₂ / b₁ →
  a₂ / a₁ = c₂ / c₁ →
  -- Conclusion
  a₂ + b₂ = 40 := by
  sorry

#check similar_triangles_leg_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_leg_sum_l542_54294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_tangent_line_equation_f_lower_bound_l542_54287

noncomputable section

variable (a : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := x / a + a / x

theorem f_monotonicity (h : a ≠ 0) :
  (a > 0 → (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < a → f a x₁ > f a x₂)) ∧
  (a < 0 → (∀ x₁ x₂, -a < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂) ∧
           (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < -a → f a x₁ < f a x₂)) :=
sorry

theorem tangent_line_equation :
  let a := (1/2 : ℝ)
  let x₀ := (1 : ℝ)
  let y₀ := f a x₀
  let k := deriv (f a) x₀
  k * (-2) = 3 ∧ 3*x₀ - 2*y₀ + 2 = 0 :=
sorry

theorem f_lower_bound :
  let a := (1/2 : ℝ)
  ∀ x > 0, f a x > Real.log x + (1/2)*x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_tangent_line_equation_f_lower_bound_l542_54287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_percentage_l542_54278

/-- Given a mixture of wine and water, prove that the initial percentage of water is 20% -/
theorem initial_water_percentage
  (initial_volume : ℝ)
  (water_added : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_volume = 125)
  (h2 : water_added = 8.333333333333334)
  (h3 : final_water_percentage = 25)
  : ∃ initial_water_percentage : ℝ,
    initial_water_percentage = 20 ∧
    (initial_volume * (initial_water_percentage / 100) + water_added) / (initial_volume + water_added) = final_water_percentage / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_percentage_l542_54278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_intersection_l542_54213

-- Define set A
def A : Set (ℝ × ℝ) :=
  {p | ∃ α β : ℝ, p.1 = 2 * Real.sin α + 2 * Real.sin β ∧ 
                    p.2 = 2 * Real.cos α + 2 * Real.cos β}

-- Define set B
def B : Set (ℝ × ℝ) :=
  {p | Real.sin (p.1 + p.2) * Real.cos (p.1 + p.2) ≥ 0}

-- State the theorem
theorem area_of_intersection : 
  MeasureTheory.volume (A ∩ B) = 8 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_intersection_l542_54213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_approx_l542_54235

/-- The depth of a circular well -/
noncomputable def well_depth (diameter : ℝ) (volume : ℝ) : ℝ :=
  volume / (Real.pi * (diameter / 2)^2)

/-- Theorem: The depth of a circular well with diameter 2 meters and volume 43.982297150257104 cubic meters is approximately 14 meters -/
theorem well_depth_approx :
  let d := 2
  let v := 43.982297150257104
  ‖well_depth d v - 14‖ < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_approx_l542_54235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trace_length_75cm_l542_54297

/-- Represents the square and trace properties -/
structure SquareTrace where
  first_trace : ℝ  -- Length of the first trace
  num_flips : ℕ    -- Number of flips until coinciding with a vertex

/-- Calculates the total trace length for a given SquareTrace -/
noncomputable def total_trace_length (st : SquareTrace) : ℝ :=
  let d := st.first_trace
  (st.num_flips * (st.num_flips + 1) / 2 : ℝ) * d

/-- Theorem stating that for the given conditions, the total trace length is 75 cm -/
theorem trace_length_75cm (st : SquareTrace) 
  (h1 : st.first_trace = 5)
  (h2 : st.num_flips = 5) : 
  total_trace_length st = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trace_length_75cm_l542_54297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_tea_price_decreased_by_70_percent_l542_54281

/-- Represents the price change of green tea and coffee from June to July -/
structure PriceChange where
  june_price : ℚ  -- Price per pound of both green tea and coffee in June
  july_coffee_price : ℚ  -- Price per pound of coffee in July
  july_tea_price : ℚ  -- Price per pound of green tea in July
  july_mixture_price : ℚ  -- Price of 3 lbs mixture (1.5 lbs each) in July

/-- The percentage change in the price of green tea from June to July -/
def green_tea_price_change (p : PriceChange) : ℚ :=
  (p.july_tea_price - p.june_price) / p.june_price * 100

/-- Theorem stating the price change of green tea -/
theorem green_tea_price_decreased_by_70_percent (p : PriceChange) 
  (h1 : p.july_coffee_price = 2 * p.june_price)  -- Coffee price doubled in July
  (h2 : p.july_tea_price = 3/10)  -- Green tea price in July
  (h3 : p.july_mixture_price = 69/20)  -- Price of 3 lbs mixture in July
  : green_tea_price_change p = -70 := by
  sorry

#check green_tea_price_decreased_by_70_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_tea_price_decreased_by_70_percent_l542_54281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_l542_54285

theorem tan_half_sum (a b : ℝ) 
  (h1 : Real.cos a + Real.cos b = 1) 
  (h2 : Real.sin a + Real.sin b = 1/2) : 
  Real.tan ((a + b)/2) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_l542_54285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_is_correct_l542_54254

/-- Triangle ABC with given side lengths and median --/
structure TriangleABC where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  BM : ℝ
  h_AB : AB = 5
  h_BC : BC = 12
  h_AC : AC = 13
  h_BM_median : BM = (AC / 2)
  h_BM_sqrt : ∃ k, BM = k * Real.sqrt 2

/-- The value of k in the equation BM = k√2 for the given triangle --/
noncomputable def find_k (t : TriangleABC) : ℝ := 13 * Real.sqrt 2 / 4

/-- Theorem stating that the value of k is correct --/
theorem k_is_correct (t : TriangleABC) : 
  ∃ k, t.BM = k * Real.sqrt 2 ∧ k = find_k t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_is_correct_l542_54254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l542_54291

open Real

noncomputable def g (A : ℝ) : ℝ :=
  (cos A * (3 * sin A ^ 2 + sin A ^ 4 + 3 * cos A ^ 2 + sin A ^ 2 * cos A ^ 2)) /
  ((cos A / sin A) * (1 / sin A - cos A * (cos A / sin A)))

theorem g_range (A : ℝ) (h : ∀ n : ℤ, A ≠ n * π) :
  3 < g A ∧ g A < 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l542_54291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_a_l542_54298

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| - |x + a|

-- Theorem for the range of f(x)
theorem range_of_f (a : ℝ) (h : a > 0) :
  Set.range (f a) = Set.Icc (-a - 2) (a + 2) :=
sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ b c : ℝ, b^2 + c^2 + b*c = 1 →
    ∃ x : ℝ, f a x ≥ 3*(b + c)) →
  a ≥ 2*Real.sqrt 3 - 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_a_l542_54298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_center_perpendicular_to_given_l542_54224

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0

-- Define the given line
def given_line (x y : ℝ) : Prop := x + y = 0

-- Define the center of the circle
noncomputable def center : ℝ × ℝ := (-1, 0)

-- Define the slope of the given line
def given_line_slope : ℝ := -1

-- Define the slope of the perpendicular line
noncomputable def perpendicular_slope : ℝ := -1 / given_line_slope

-- Define the equation of the line we want to prove
def target_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem statement
theorem line_through_center_perpendicular_to_given :
  ∀ x y : ℝ, 
  (x, y) = center ∨ 
  (∃ t : ℝ, x = center.fst + t ∧ y = center.snd + t * perpendicular_slope) →
  target_line x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_center_perpendicular_to_given_l542_54224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_closed_form_l542_54228

/-- The sum of the first n terms of a specific sequence -/
def sequence_sum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun k => 2^(k+1) - 1)

/-- Theorem stating the closed form of the sequence sum -/
theorem sequence_sum_closed_form (n : ℕ) :
  sequence_sum n = 2^(n+1) - n - 2 := by
  sorry

#check sequence_sum_closed_form

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_closed_form_l542_54228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l542_54258

def a : ℕ → ℚ
  | 0 => 4  -- Add a case for 0 to avoid missing case error
  | 1 => 4
  | n + 2 => 4 - 4 / a (n + 1)

def b (n : ℕ) : ℚ := 1 / (a n - 2)

def is_arithmetic_sequence (f : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, f (n + 1) - f n = d

theorem sequence_properties :
  (is_arithmetic_sequence b) ∧
  (∀ n : ℕ, n ≠ 0 → a n = 2 / n + 2) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l542_54258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_cube_properties_l542_54283

/-- A cube made of 27 dice with 26 visible dice on the surface -/
structure DiceCube where
  total_dice : ℕ
  visible_dice : ℕ
  face_centered_prob : ℚ
  edge_centered_prob : ℚ
  vertex_centered_prob : ℚ

/-- Properties of the dice cube -/
def standard_dice_cube : DiceCube :=
  { total_dice := 27
  , visible_dice := 26
  , face_centered_prob := 1/6
  , edge_centered_prob := 1/3
  , vertex_centered_prob := 1/2
  }

/-- Function to calculate probability of exactly 25 sixes -/
noncomputable def prob_exactly_25_sixes (c : DiceCube) : ℚ := 31 / (2^13 * 3^18)

/-- Function to calculate probability of at least one one -/
noncomputable def prob_at_least_one_one (c : DiceCube) : ℚ := 1 - 5^6 / (2^2 * 3^18)

/-- Function to calculate expected number of sixes -/
noncomputable def expected_number_of_sixes (c : DiceCube) : ℚ := 9

/-- Function to calculate expected sum of numbers -/
noncomputable def expected_sum_of_numbers (c : DiceCube) : ℚ := 189

/-- Function to calculate expected number of different digits -/
noncomputable def expected_different_digits (c : DiceCube) : ℚ := 6 - 5^6 / (2 * 3^17)

/-- Theorems about the dice cube probabilities and expectations -/
theorem dice_cube_properties (c : DiceCube) (h : c = standard_dice_cube) :
  /- Probability of exactly 25 sixes on the surface -/
  (prob_exactly_25_sixes c = 31 / (2^13 * 3^18)) ∧
  /- Probability of at least one one on the surface -/
  (prob_at_least_one_one c = 1 - 5^6 / (2^2 * 3^18)) ∧
  /- Expected number of sixes on the surface -/
  (expected_number_of_sixes c = 9) ∧
  /- Expected sum of numbers on the surface -/
  (expected_sum_of_numbers c = 189) ∧
  /- Expected number of different digits on the surface -/
  (expected_different_digits c = 6 - 5^6 / (2 * 3^17)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_cube_properties_l542_54283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l542_54273

theorem sequence_inequality (a : ℕ → ℝ) (h : ∀ (m n : ℕ), a (m + n) ≤ a m + a n) 
  (nonneg : ∀ n, a n ≥ 0) :
  ∀ (m n : ℕ), m > 0 → n ≥ m → a n ≤ m * a 1 + (n / m - 1) * a m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l542_54273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l542_54241

-- Define the complex number z
noncomputable def z : ℂ := 1 + Complex.I

-- Define omega
noncomputable def ω : ℂ := z / (2 + Complex.I)

-- Theorem statement
theorem complex_number_problem :
  (∃ (b : ℝ), b > 0 ∧ z = 1 + Complex.I * b) → 
  (Complex.im ((z - 2)^2) ≠ 0 ∧ Complex.re ((z - 2)^2) = 0) →
  z = 1 + Complex.I ∧ Complex.abs ω = Real.sqrt 10 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l542_54241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_f_g_l542_54282

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 6)

noncomputable def g (x : ℝ) := f (x - Real.pi / 6)

theorem max_sum_f_g :
  ∃ (M : ℝ), M = Real.sqrt 3 ∧ ∀ (x : ℝ), f x + g x ≤ M :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_f_g_l542_54282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_inequality_greatest_constant_l542_54290

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the radii of the tangent circles
noncomputable def r (t : RightTriangle) : ℝ := (t.a * t.b) / (t.a + t.c)
noncomputable def t (t : RightTriangle) : ℝ := (t.a * t.b) / (t.b + t.c)

-- State the theorem
theorem tangent_circles_inequality (tri : RightTriangle) :
  (1 / r tri) + (1 / t tri) ≥ (Real.sqrt 2 + 1) * ((1 / tri.a) + (1 / tri.b)) := by
  sorry

-- State that √2 + 1 is the greatest possible value
theorem greatest_constant (p : ℝ) :
  (∀ (tri : RightTriangle), (1 / r tri) + (1 / t tri) ≥ p * ((1 / tri.a) + (1 / tri.b))) →
  p ≤ Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_inequality_greatest_constant_l542_54290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_30_degrees_l542_54229

theorem cot_30_degrees : Real.tan (π / 6)⁻¹ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_30_degrees_l542_54229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_center_of_mass_l542_54248

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a mass placed at a point -/
structure MassPoint where
  point : Point
  mass : ℝ

/-- Calculates the center of mass of a system of mass points -/
noncomputable def centerOfMass (points : List MassPoint) : Point :=
  let totalMass := points.map (·.mass) |>.sum
  let sumX := points.map (λ p => p.mass * p.point.x) |>.sum
  let sumY := points.map (λ p => p.mass * p.point.y) |>.sum
  { x := sumX / totalMass, y := sumY / totalMass }

theorem equilateral_triangle_center_of_mass :
  let A : Point := { x := 0, y := 0 }
  let B : Point := { x := 1, y := 0 }
  let C : Point := { x := 1/2, y := Real.sqrt 3 / 2 }
  let massPoints : List MassPoint := [
    { point := A, mass := 1 },
    { point := B, mass := 2 },
    { point := C, mass := 3 }
  ]
  let com := centerOfMass massPoints
  com.x = 7/12 ∧ com.y = Real.sqrt 3 / 4 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_center_of_mass_l542_54248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_sum_l542_54200

/-- Given points A(-3, -4) and B(6, 3) in the xy-plane, and C(1, m) chosen such that AC + CB is minimized, prove that m = -8/9 -/
theorem minimize_distance_sum (A B C : ℝ × ℝ) (m : ℝ) : 
  A = (-3, -4) → 
  B = (6, 3) → 
  C = (1, m) → 
  (∀ m' : ℝ, Real.sqrt ((1 - (-3))^2 + (m - (-4))^2) + Real.sqrt ((6 - 1)^2 + (3 - m)^2) 
    ≤ Real.sqrt ((1 - (-3))^2 + (m' - (-4))^2) + Real.sqrt ((6 - 1)^2 + (3 - m')^2)) →
  m = -8/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_sum_l542_54200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_b_shut_time_l542_54271

/-- Represents the capacity of a tank -/
noncomputable def TankCapacity : ℝ := 1

/-- The rate at which Pipe A fills the tank (in tanks per hour) -/
noncomputable def PipeARate : ℝ := 1 / 2

/-- The rate at which Pipe B fills the tank (in tanks per hour) -/
noncomputable def PipeBRate : ℝ := 1

/-- The time it takes for the tank to overflow (in hours) -/
noncomputable def OverflowTime : ℝ := 1 / 2

/-- Theorem: The time Pipe B is shut before the tank overflows is 30 minutes -/
theorem pipe_b_shut_time : 
  ∃ (t : ℝ), t * 60 = 30 ∧ 
  (PipeARate + PipeBRate) * (OverflowTime - t) + PipeARate * t = TankCapacity :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_b_shut_time_l542_54271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_value_l542_54267

/-- The function f(x) = kx^2 - kx -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - k * x

/-- The piecewise function g(x) -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.log x
  else -x^3 + (a + 1) * x^2 - a * x

/-- The theorem stating that a = 2 is the unique value satisfying f(x) ≥ g(x) for all positive x -/
theorem unique_a_value :
  ∃! a : ℝ, ∀ x : ℝ, x > 0 → f 1 x ≥ g a x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_value_l542_54267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_price_is_one_l542_54233

/-- The price of a shirt in dollars -/
noncomputable def shirt_price : ℚ := 1

/-- The number of shirts sold -/
def num_shirts : ℕ := 5

/-- The number of pants sold -/
def num_pants : ℕ := 5

/-- The price of a pair of pants in dollars -/
noncomputable def pants_price : ℚ := 3

/-- The fraction of money Kekai keeps for himself -/
noncomputable def kekai_fraction : ℚ := 1 / 2

/-- The amount of money Kekai has left in dollars -/
noncomputable def money_left : ℚ := 10

theorem shirt_price_is_one :
  shirt_price = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_price_is_one_l542_54233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_artist_birthday_same_day_of_week_l542_54292

/-- Determines if a year is a leap year --/
def isLeapYear (year : ℕ) : Bool :=
  (year % 400 == 0) || (year % 4 == 0 && year % 100 ≠ 0)

/-- Counts the number of leap years between two years, inclusive --/
def countLeapYears (startYear endYear : ℕ) : ℕ :=
  (List.range (endYear - startYear + 1)).map (fun i => startYear + i) |>.filter isLeapYear |>.length

/-- Calculates the number of days between two dates, considering leap years --/
def daysBetweenDates (startYear endYear : ℕ) : ℕ :=
  let totalYears := endYear - startYear + 1
  let leapYears := countLeapYears startYear endYear
  let regularYears := totalYears - leapYears
  regularYears * 365 + leapYears * 366

/-- Theorem: The artist's 130th birthday falls on the same day of the week as their birth date --/
theorem artist_birthday_same_day_of_week :
  let birthYear := 1896
  let birthdayYear := birthYear + 130
  daysBetweenDates birthYear birthdayYear % 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_artist_birthday_same_day_of_week_l542_54292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_depth_is_four_l542_54265

/-- Represents the properties of a river --/
structure River where
  width : ℝ
  flowRate : ℝ
  waterAmount : ℝ

/-- Calculates the depth of a river given its properties --/
noncomputable def riverDepth (r : River) : ℝ :=
  r.waterAmount / (r.width * r.flowRate * 1000 / 60)

/-- Theorem stating that a river with the given properties has a depth of 4 meters --/
theorem river_depth_is_four :
  let r : River := {
    width := 40,
    flowRate := 4,
    waterAmount := 10666.666666666666
  }
  riverDepth r = 4 := by
  -- Expand the definition of riverDepth
  unfold riverDepth
  -- Perform the calculation
  simp [River.width, River.flowRate, River.waterAmount]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_depth_is_four_l542_54265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_constant_on_open_interval_l542_54214

/-- A function f on (0,1) is constant if it's positive and satisfies the given inequality. -/
theorem function_constant_on_open_interval (f : ℝ → ℝ) :
  (∀ x, x ∈ Set.Ioo 0 1 → f x > 0) →
  (∀ x y, x ∈ Set.Ioo 0 1 → y ∈ Set.Ioo 0 1 → f x / f y + f (1 - x) / f (1 - y) ≤ 2) →
  ∃ c : ℝ, ∀ x, x ∈ Set.Ioo 0 1 → f x = c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_constant_on_open_interval_l542_54214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_part1_line_equation_part2_l542_54239

-- Part 1
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

def Line2D.contains_point (l : Line2D) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

def line_through_origin_and_intersection (l : Line2D) : Prop :=
  l.contains_point (0, 0) ∧
  ∃ p, l.contains_point p ∧
       (2 * p.1 + 3 * p.2 + 8 = 0) ∧
       (p.1 - p.2 - 1 = 0)

theorem line_equation_part1 (l : Line2D) :
  line_through_origin_and_intersection l →
  l.a = 2 ∧ l.b = -1 ∧ l.c = 0 :=
sorry

-- Part 2
def line_through_point_equal_intercepts (l : Line2D) : Prop :=
  l.contains_point (2, 3) ∧
  ∃ a, a ≠ 0 ∧ l.contains_point (a, 0) ∧ l.contains_point (0, a)

theorem line_equation_part2 (l : Line2D) :
  line_through_point_equal_intercepts l →
  (l.a = 1 ∧ l.b = 1 ∧ l.c = -5) ∨
  (l.a = 3 ∧ l.b = -2 ∧ l.c = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_part1_line_equation_part2_l542_54239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_l542_54261

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log (x - 1)

-- State the theorem
theorem tangent_line_at_2 :
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  let m : ℝ := (Real.log (x₀ - 1) + x₀ / (x₀ - 1))
  ∀ x y : ℝ, y = m * (x - x₀) + y₀ ↔ y = 2 * x - 4 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_l542_54261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_above_line_l542_54293

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 12*x + y^2 - 18*y + 89 = 0

/-- The line equation -/
def line_equation (y : ℝ) : Prop :=
  y = 4

/-- The area of the circle above the line -/
noncomputable def area_above_line : ℝ := 28 * Real.pi

/-- Theorem stating that the area of the circle above the line is 28π -/
theorem circle_area_above_line :
  ∀ x y : ℝ, circle_equation x y → area_above_line = 28 * Real.pi :=
by
  intros x y h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_above_line_l542_54293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_pairs_l542_54227

/-- The number of distinct pairs of integers (x, y) satisfying the given conditions -/
noncomputable def num_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    0 < p.1 ∧ p.1 < p.2 ∧ 
    Real.sqrt 3000 = Real.sqrt (p.1 : Real) + Real.sqrt (p.2 : Real))
    (Finset.product (Finset.range 3001) (Finset.range 3001))).card

/-- Theorem stating that there are exactly 4 such pairs -/
theorem four_pairs : num_pairs = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_pairs_l542_54227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_coefficient_relation_l542_54284

/-- Given vectors in a plane with specific properties, prove a relation between their coefficients -/
theorem vector_coefficient_relation (a b c : ℝ × ℝ) (l m : ℝ) : 
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- angle between a and b is 90°
  (a.1^2 + a.2^2 = 1) →  -- |a| = 1
  (b.1^2 + b.2^2 = 1) →  -- |b| = 1
  (c.1^2 + c.2^2 = 12) →  -- |c| = 2√3
  (c = (l * a.1 + m * b.1, l * a.2 + m * b.2)) →  -- c = la + mb
  l^2 + m^2 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_coefficient_relation_l542_54284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_as_function_of_pkd_l542_54279

/-- Given an equation of the form (y² - py) / (ky + d) = (n - 2) / (n + 2),
    where y has two distinct positive roots such that one root is twice the other,
    n can be expressed as a function of p, k, and d. -/
theorem n_as_function_of_pkd (p k d : ℝ) :
  ∃ (f : ℝ → ℝ → ℝ → ℝ) (y₁ y₂ : ℝ),
    y₁ > 0 ∧ y₂ > 0 ∧ y₁ ≠ y₂ ∧ y₁ = 2 * y₂ ∧
    (y₁^2 - p*y₁) / (k*y₁ + d) = (f p k d - 2) / (f p k d + 2) ∧
    (y₂^2 - p*y₂) / (k*y₂ + d) = (f p k d - 2) / (f p k d + 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_as_function_of_pkd_l542_54279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l542_54257

/-- Represents a cone with given base radius and slant height -/
structure Cone where
  baseRadius : ℝ
  slantHeight : ℝ

/-- Calculates the central angle (in degrees) of the sector in the unfolded diagram of a cone -/
noncomputable def centralAngle (cone : Cone) : ℝ :=
  (cone.baseRadius / cone.slantHeight) * 360

/-- Theorem: For a cone with base radius 3 cm and slant height 6 cm, 
    the central angle of the sector in its unfolded diagram is 180° -/
theorem cone_central_angle :
  let cone : Cone := { baseRadius := 3, slantHeight := 6 }
  centralAngle cone = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l542_54257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_filled_circles_l542_54210

/-- Represents the sequence of circles where the kth ● appears after k circles (○) -/
def circle_sequence (n : ℕ) : ℕ := 
  if n % 2 = 1 then 1 else n / 2

/-- The sum of the first n terms in the circle sequence -/
def circle_sum (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) circle_sequence

/-- The number of ● in the first 120 circles -/
theorem number_of_filled_circles : 
  (∃ n : ℕ, circle_sum n ≤ 120 ∧ circle_sum (n + 1) > 120) → 
  (∃ m : ℕ, m = 14 ∧ circle_sum m ≤ 120 ∧ circle_sum (m + 1) > 120) :=
by
  sorry

#eval circle_sum 14  -- To check the value
#eval circle_sum 15  -- To check the value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_filled_circles_l542_54210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_profit_percentage_l542_54230

/-- Calculates the profit percentage given the selling price and cost price -/
noncomputable def profit_percentage (selling_price cost_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- The profit percentage is approximately 19.99% when a book is sold for $260 with a cost price of $216.67 -/
theorem book_profit_percentage :
  let selling_price : ℝ := 260
  let cost_price : ℝ := 216.67
  abs (profit_percentage selling_price cost_price - 19.99) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_profit_percentage_l542_54230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_cap_area_for_given_sphere_l542_54299

/-- The volume of a sphere in terms of its radius -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The surface area of a spherical cap in terms of the sphere's radius and the cap's height -/
noncomputable def spherical_cap_area (r h : ℝ) : ℝ := 2 * Real.pi * r * h

theorem spherical_cap_area_for_given_sphere :
  ∃ (r : ℝ), 
    sphere_volume r = 288 * Real.pi ∧ 
    spherical_cap_area r 2 = 24 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_cap_area_for_given_sphere_l542_54299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_l542_54218

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := (4, 3)
def F₂ : ℝ × ℝ := (-6, 5)

theorem distance_between_foci :
  distance F₁.1 F₁.2 F₂.1 F₂.2 = 2 * Real.sqrt 26 := by
  sorry

#eval F₁
#eval F₂

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_l542_54218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jarry_secretary_or_treasurer_probability_l542_54264

/-- A club with 10 members, including Jarry -/
structure Club where
  members : Finset Nat
  jarry : Nat
  member_count : members.card = 10
  jarry_in_club : jarry ∈ members

/-- The probability of an event occurring -/
noncomputable def probability (event : Prop) : ℚ := sorry

/-- Choose n elements from a finite set -/
noncomputable def choose (s : Finset Nat) (n : Nat) : Finset Nat := sorry

theorem jarry_secretary_or_treasurer_probability (c : Club) :
  probability (
    let president := choose c.members 1
    let secretary := choose (c.members \ president) 1
    let treasurer := choose (c.members \ (president ∪ secretary)) 1
    c.jarry ∈ secretary ∨ c.jarry ∈ treasurer
  ) = 19/90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jarry_secretary_or_treasurer_probability_l542_54264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_property_l542_54202

theorem remainder_property (n : ℕ) : 
  (∀ q : ℕ, q > 0 → n % (q^2) < q^2 / 2) ↔ (n = 1 ∨ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_property_l542_54202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_imply_a_negative_intersection_points_not_one_l542_54263

noncomputable section

-- Define the quadratic equation
def quadratic_equation (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-3)*x + a

-- Define the piecewise function
def piecewise_function (x : ℝ) : ℝ :=
  if x ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) then
    3 - x^2
  else
    x^2 - 3

theorem quadratic_roots_imply_a_negative (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ quadratic_equation a x = 0 ∧ quadratic_equation a y = 0) →
  a < 0 :=
by sorry

theorem intersection_points_not_one (a : ℝ) :
  ¬(∃! x : ℝ, piecewise_function x = a) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_imply_a_negative_intersection_points_not_one_l542_54263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trail_length_is_48_l542_54237

/-- The length of the trail Shelly-Ann runs, in meters. -/
noncomputable def trail_length : ℝ := 48

/-- Shelly-Ann's normal running speed, in meters per second. -/
noncomputable def normal_speed : ℝ := 8

/-- The fraction of the trail covered in mud. -/
noncomputable def mud_fraction : ℝ := 1/3

/-- The fraction of normal speed at which Shelly-Ann runs through mud. -/
noncomputable def mud_speed_fraction : ℝ := 1/4

/-- The time it takes Shelly-Ann to run the entire trail, in seconds. -/
noncomputable def total_time : ℝ := 12

/-- Theorem stating that the trail length is 48 meters given the conditions. -/
theorem trail_length_is_48 : trail_length = 48 :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trail_length_is_48_l542_54237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_greater_than_1_1_l542_54203

noncomputable def numbers : List ℚ := [14/10, 9/10, 12/10, 5/10, 13/10]

theorem sum_of_numbers_greater_than_1_1 :
  (numbers.filter (λ x => x > 11/10)).sum = 39/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_greater_than_1_1_l542_54203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_decomposition_g_is_odd_l542_54277

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the properties of odd and even functions
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x
def is_even (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x

-- State the theorem
theorem exp_decomposition (g h : ℝ → ℝ) 
  (hf : ∀ x, f x = g x + h x) 
  (hg : is_odd g) 
  (hh : is_even h) : 
  ∀ x, g x = (Real.exp x - Real.exp (-x)) / 2 := by
  sorry

-- Define the specific g function
noncomputable def g (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

-- Prove that g is indeed odd
theorem g_is_odd : is_odd g := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_decomposition_g_is_odd_l542_54277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l542_54255

theorem range_of_m :
  ∃ (m : ℝ), ∀ (x : ℝ), (Real.sqrt 3 * Real.sin x + Real.cos x = 4 - m) → (2 ≤ m) ∧ (m ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l542_54255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l542_54222

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola : Set Point :=
  {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola y^2 = 4x -/
def focus : Point :=
  ⟨1, 0⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

theorem parabola_point_distance (M : Point) 
  (h1 : M ∈ Parabola) 
  (h2 : distance M focus = 10) : 
  distanceToYAxis M = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l542_54222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_coordinates_l542_54289

/-- Given a point M with rectangular coordinates (1, -√3), prove that its polar coordinates are (2, -π/3) -/
theorem rectangular_to_polar_coordinates :
  let x : ℝ := 1
  let y : ℝ := -Real.sqrt 3
  let ρ : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  ρ ≥ 0 ∧ -π ≤ θ ∧ θ < π →
  ρ = 2 ∧ θ = -π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_coordinates_l542_54289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_l542_54231

noncomputable def odot (x y : ℝ) : ℝ := (x * y + 4) / (x + y)

axiom odot_assoc (x y z : ℝ) : x > 0 → y > 0 → z > 0 → odot x (odot y z) = odot (odot x y) z

noncomputable def T : ℕ → ℝ
  | 0 => 3
  | n + 1 => odot (T n) (n + 4)

theorem not_perfect_square (n : ℕ) (h : n ≥ 4) : 
  ¬ ∃ (m : ℕ), (96 : ℝ) / (T n - 2) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_l542_54231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_half_l542_54296

-- Define the power function as noncomputable
noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

-- State the theorem
theorem alpha_value_half (α : ℝ) : 
  power_function α 2 = Real.sqrt 2 → α = 1/2 := by
  intro h
  -- The proof steps would go here
  sorry

#check alpha_value_half

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_half_l542_54296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_225_squared_l542_54201

open Complex

-- Define the theorem
theorem cos_sin_225_squared : (cos (225 * π / 180) + I * sin (225 * π / 180))^2 = I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_225_squared_l542_54201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_people_arrangement_l542_54207

def num_people : ℕ := 5

-- Define a function to calculate the number of arrangements
def arrangements (n : ℕ) : ℕ :=
  let adjacent_arrangements := Nat.factorial (n - 1)
  let non_adjacent_arrangements := Nat.factorial n - adjacent_arrangements
  adjacent_arrangements + non_adjacent_arrangements

-- Theorem statement
theorem five_people_arrangement :
  arrangements num_people = 48 := by
  -- Proof goes here
  sorry

#eval arrangements num_people

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_people_arrangement_l542_54207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_divisors_sum_l542_54217

theorem infinite_divisors_sum (k : ℕ) (h_k_odd : Odd k) (h_k_gt_3 : k > 3) :
  ∃ f : ℕ → ℕ, StrictMono f ∧
    ∀ (i : ℕ), ∃ (d₁ d₂ : ℕ),
      d₁ > 0 ∧ d₂ > 0 ∧
      (((f i)^2 + 1) / 2) % d₁ = 0 ∧
      (((f i)^2 + 1) / 2) % d₂ = 0 ∧
      d₁ + d₂ = f i + k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_divisors_sum_l542_54217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l542_54269

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x - f y) = 1 - x - y) : 
  ∀ x : ℝ, f x = 1/2 - x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l542_54269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_X_l542_54259

/-- The probability density function for the random variable X -/
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < Real.pi then (1/2) * Real.sin x else 0

/-- Theorem stating the variance of the random variable X -/
theorem variance_of_X : 
  ∫ x in Set.Ioo 0 Real.pi, (x - ∫ y in Set.Ioo 0 Real.pi, y * f y)^2 * f x = (Real.pi^2 - 8) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_X_l542_54259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_is_correct_l542_54205

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  3 * x^2 + 18 * x + 9 * y^2 - 27 * y + 27 = 0

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := 2.598 * Real.pi

/-- Theorem stating that the area of the ellipse is 2.598π -/
theorem ellipse_area_is_correct : 
  ∃ (a b : ℝ), (∀ (x y : ℝ), ellipse_equation x y ↔ (x + 3)^2 / a^2 + (y - 1.5)^2 / b^2 = 1) ∧
  ellipse_area = Real.pi * a * b := by
  sorry

#check ellipse_area_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_is_correct_l542_54205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PB_equation_l542_54268

-- Define the points A, B, and P
variable (A B P : ℝ × ℝ)

-- Define the conditions
axiom A_on_x_axis : A.2 = 0
axiom B_on_x_axis : B.2 = 0
axiom P_x_coord : P.1 = 1
axiom equal_distances : abs (P.1 - A.1) = abs (P.1 - B.1)
axiom PA_equation : ∀ (x y : ℝ), (x, y) ∈ {p | p.1 - p.2 + 1 = 0}

-- State the theorem
theorem PB_equation :
  ∀ (x y : ℝ), (x, y) ∈ {p | p.1 + p.2 - 3 = 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_PB_equation_l542_54268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_after_three_steps_l542_54256

/-- Represents a number line with equally spaced markings -/
structure NumberLine where
  start : ℚ
  end_ : ℚ
  steps : ℕ

/-- The value at a given step on the number line -/
def value_at_step (nl : NumberLine) (step : ℕ) : ℚ :=
  nl.start + (nl.end_ - nl.start) * (step : ℚ) / (nl.steps : ℚ)

/-- Theorem: On a number line from 0 to 20 with 5 equally spaced steps, 
    the value reached after 3 steps from 0 is 12 -/
theorem value_after_three_steps :
  let nl : NumberLine := { start := 0, end_ := 20, steps := 5 }
  value_at_step nl 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_after_three_steps_l542_54256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_theorem_octagon_diagonal_theorem_l542_54226

-- Define a convex hexagon
structure ConvexHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_convex : Convex ℝ (Set.range vertices)

-- Define the area of a hexagon
noncomputable def area (h : ConvexHexagon) : ℝ :=
  sorry

-- Define a diagonal of a hexagon
def diagonal (h : ConvexHexagon) (i j : Fin 6) : Set (ℝ × ℝ) :=
  sorry

-- Define the area of a triangle formed by a diagonal
noncomputable def triangle_area (h : ConvexHexagon) (i j : Fin 6) : ℝ :=
  sorry

-- Theorem statement
theorem hexagon_diagonal_theorem (h : ConvexHexagon) :
  ∃ (i j : Fin 6), i ≠ j ∧ triangle_area h i j ≤ (area h) / 6 := by
  sorry

-- Define a convex octagon
structure ConvexOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_convex : Convex ℝ (Set.range vertices)

-- Define the area of an octagon
noncomputable def octagon_area (o : ConvexOctagon) : ℝ :=
  sorry

-- Define the area of a triangle formed by a diagonal in an octagon
noncomputable def octagon_triangle_area (o : ConvexOctagon) (i j : Fin 8) : ℝ :=
  sorry

-- Theorem statement for octagon
theorem octagon_diagonal_theorem (o : ConvexOctagon) :
  ∃ (i j : Fin 8), i ≠ j ∧ octagon_triangle_area o i j ≤ (octagon_area o) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_theorem_octagon_diagonal_theorem_l542_54226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a4_value_l542_54249

/-- Represents an arithmetic sequence with its first term and common difference -/
structure ArithmeticSequence where
  a1 : ℝ  -- First term
  d : ℝ   -- Common difference

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * seq.a1 + ((n : ℝ) - 1) * seq.d)

/-- The fourth term of an arithmetic sequence -/
def a4 (seq : ArithmeticSequence) : ℝ :=
  seq.a1 + 3 * seq.d

/-- Theorem stating the maximum value of a4 given the conditions -/
theorem max_a4_value (seq : ArithmeticSequence) 
    (h1 : S seq 1 ≤ 13)
    (h4 : S seq 4 ≥ 10)
    (h5 : S seq 5 ≤ 15) :
    a4 seq ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a4_value_l542_54249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_value_l542_54221

/-- A polynomial of degree n satisfying P(k) = 1 / C(n+1, k) for k = 0, 1, ..., n -/
def SpecialPolynomial (n : ℕ) (P : ℝ → ℝ) : Prop :=
  (∃ (coeffs : Fin (n + 1) → ℝ), P = λ x ↦ (Finset.range (n + 1)).sum (λ i ↦ coeffs i * x ^ i)) ∧
  ∀ k : Fin (n + 1), P k = 1 / (Nat.choose (n + 1) k)

/-- The value of P(n+1) for a SpecialPolynomial P -/
theorem special_polynomial_value (n : ℕ) (P : ℝ → ℝ) 
  (h : SpecialPolynomial n P) : 
  P (n + 1) = if n % 2 = 0 then 1 else 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_value_l542_54221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_octahedron_volume_ratio_l542_54242

/-- The volume of a regular tetrahedron -/
noncomputable def volume_tetrahedron (side_length : ℝ) : ℝ :=
  side_length^3 * Real.sqrt 2 / 12

/-- The volume of a regular octahedron -/
noncomputable def volume_octahedron (side_length : ℝ) : ℝ :=
  side_length^3 * Real.sqrt 2 / 3

/-- The side length of the tetrahedron formed by joining the centers of adjoining faces of an octahedron -/
noncomputable def tetrahedron_side_length (octahedron_side_length : ℝ) : ℝ :=
  octahedron_side_length * Real.sqrt 3 / 3

/-- The theorem stating the ratio of volumes -/
theorem tetrahedron_octahedron_volume_ratio (s : ℝ) (h : s > 0) :
  volume_tetrahedron (tetrahedron_side_length s) / volume_octahedron s = Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_octahedron_volume_ratio_l542_54242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_east_northwest_l542_54246

/-- Represents the directions on a circular table with 12 equally spaced rays -/
inductive Direction
| North | Northeast | East | Southeast | South | Southwest | West | Northwest

/-- The number of rays on the circular table -/
def num_rays : ℕ := 12

/-- The angle between adjacent rays in degrees -/
noncomputable def angle_between_adjacent_rays : ℚ := 360 / num_rays

/-- Calculates the number of steps between two directions -/
def steps_between (d1 d2 : Direction) : ℕ :=
  match d1, d2 with
  | Direction.East, Direction.Northwest => 3
  | _, _ => 0  -- Default case, not relevant for this problem

/-- Theorem: The smaller angle between East and Northwest is 90 degrees -/
theorem angle_east_northwest : 
  angle_between_adjacent_rays * (steps_between Direction.East Direction.Northwest) = 90 := by
  sorry

#eval steps_between Direction.East Direction.Northwest

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_east_northwest_l542_54246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_372_in_first_quadrant_l542_54250

/-- The quadrant of an angle in degrees -/
inductive Quadrant
| First
| Second
| Third
| Fourth

/-- Determines the quadrant of an angle in degrees -/
noncomputable def angle_quadrant (angle : ℝ) : Quadrant :=
  let normalized_angle := angle % 360
  if 0 ≤ normalized_angle ∧ normalized_angle < 90 then Quadrant.First
  else if 90 ≤ normalized_angle ∧ normalized_angle < 180 then Quadrant.Second
  else if 180 ≤ normalized_angle ∧ normalized_angle < 270 then Quadrant.Third
  else Quadrant.Fourth

theorem angle_372_in_first_quadrant :
  angle_quadrant 372 = Quadrant.First :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_372_in_first_quadrant_l542_54250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_win_solution_l542_54215

def lottery_win_amount (x : ℝ) : Prop :=
  let federal_tax_rate : ℝ := 0.25
  let local_tax_rate : ℝ := 0.15
  let total_tax : ℝ := 18000
  let federal_tax : ℝ := federal_tax_rate * x
  let remaining_after_federal : ℝ := x - federal_tax
  let local_tax : ℝ := local_tax_rate * remaining_after_federal
  federal_tax + local_tax = total_tax

theorem lottery_win_solution :
  ∃ (x : ℝ), lottery_win_amount x ∧ Int.floor x = 49655 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_win_solution_l542_54215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_2_inequality_holds_iff_l542_54211

/-- The function f(x) = ln x + a/x - 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x - 1

theorem min_value_when_a_is_2 :
  ∃ (min_value : ℝ), min_value = Real.log 2 ∧
  ∀ x > 0, f 2 x ≥ min_value := by
  sorry

theorem inequality_holds_iff :
  ∀ a : ℝ, (∀ x ≥ 1, f a x ≤ (1/2) * x - 1) ↔ a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_2_inequality_holds_iff_l542_54211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l542_54272

/-- Point P in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Line l defined by parametric equations -/
structure Line where
  α : ℝ
  p : Point2D

/-- Intersection point of line with x-axis -/
noncomputable def intersectX (l : Line) : Point2D :=
  { x := l.p.x - l.p.y / Real.tan l.α, y := 0 }

/-- Intersection point of line with y-axis -/
noncomputable def intersectY (l : Line) : Point2D :=
  { x := 0, y := l.p.y - l.p.x * Real.tan l.α }

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem statement -/
theorem line_properties (l : Line) (h1 : l.p = { x := 2, y := 1 })
    (h2 : distance l.p (intersectX l) * distance l.p (intersectY l) = 4) :
    l.α = 3 * Real.pi / 4 ∧
    ∀ θ ρ, ρ * (Real.cos θ + Real.sin θ) = 3 ↔ 
      ρ * Real.cos θ = l.p.x - l.p.y + ρ * Real.sin θ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l542_54272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minor_axis_length_l542_54219

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The sum of distances from any point on the ellipse to the two foci -/
def Ellipse.focalSum (e : Ellipse) : ℝ := 2 * e.a

theorem ellipse_minor_axis_length
  (e : Ellipse)
  (h_eccentricity : e.eccentricity = Real.sqrt 5 / 3)
  (h_focal_sum : e.focalSum = 12) :
  2 * e.b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minor_axis_length_l542_54219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_five_l542_54223

-- Define the circle center
def circle_center (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define the points on the circle
def point1 : ℝ × ℝ := (2, 5)
def point2 : ℝ × ℝ := (3, 4)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem circle_radius_is_five :
  ∃ x : ℝ, distance (circle_center x) point1 = distance (circle_center x) point2 ∧
           distance (circle_center x) point1 = 5 := by
  -- Proof goes here
  sorry

#check circle_radius_is_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_five_l542_54223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l542_54209

noncomputable def original_function (x : ℝ) : ℝ := Real.cos (3 * x)

noncomputable def shift : ℝ := Real.pi / 12

noncomputable def shifted_function (x : ℝ) : ℝ := original_function (x - shift)

def is_center_of_symmetry (c : ℝ × ℝ) : Prop :=
  ∀ x, shifted_function (c.1 + x) = shifted_function (c.1 - x)

theorem center_of_symmetry :
  is_center_of_symmetry (-Real.pi/12, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l542_54209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_buffet_theorem_l542_54262

/-- Represents a buffet with dishes containing mangoes -/
structure MangoBuffet where
  total_dishes : ℕ
  mango_salsa_dishes : ℕ
  fresh_mango_ratio : ℚ
  mango_jelly_dishes : ℕ
  pickable_mango_dishes : ℕ

/-- Calculates the number of dishes available to someone who won't eat mangoes -/
def available_dishes (buffet : MangoBuffet) : ℕ :=
  buffet.total_dishes -
  (buffet.mango_salsa_dishes +
   (buffet.fresh_mango_ratio * buffet.total_dishes).floor.toNat +
   buffet.mango_jelly_dishes -
   buffet.pickable_mango_dishes)

/-- Theorem stating the number of available dishes for the given buffet conditions -/
theorem mango_buffet_theorem (buffet : MangoBuffet)
  (h1 : buffet.total_dishes = 36)
  (h2 : buffet.mango_salsa_dishes = 3)
  (h3 : buffet.fresh_mango_ratio = 1 / 6)
  (h4 : buffet.mango_jelly_dishes = 1)
  (h5 : buffet.pickable_mango_dishes = 2) :
  available_dishes buffet = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_buffet_theorem_l542_54262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_participants_is_six_l542_54243

/-- Represents a participant in the chess competition -/
structure Participant where
  wins : ℕ
  draws : ℕ
  losses : ℕ

/-- Calculates the total points for a participant -/
def points (p : Participant) : ℕ := 2 * p.wins + p.draws

/-- Represents the chess competition -/
structure ChessCompetition where
  participants : ℕ
  champion : Participant
  others : List Participant

/-- The competition follows single round-robin format -/
axiom single_round_robin (c : ChessCompetition) :
  ∀ p ∈ c.others, p.wins + p.draws + p.losses = c.participants - 1

/-- The champion has more points than any other participant -/
axiom champion_has_most_points (c : ChessCompetition) :
  ∀ p ∈ c.others, points c.champion > points p

/-- The champion has won fewer matches than any other participant -/
axiom champion_has_fewest_wins (c : ChessCompetition) :
  ∀ p ∈ c.others, c.champion.wins < p.wins

/-- The minimum number of participants is 6 -/
theorem min_participants_is_six :
  ∀ c : ChessCompetition,
    (∀ p ∈ c.others, points c.champion > points p) →
    (∀ p ∈ c.others, c.champion.wins < p.wins) →
    c.participants ≥ 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_participants_is_six_l542_54243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_parabola_min_distance_achievable_l542_54225

/-- The circle equation: x^2 + y^2 + 8x + 18 = 0 -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 + 8*x + 18 = 0

/-- The parabola equation: y^2 = 8x -/
def parabolaEq (x y : ℝ) : Prop := y^2 = 8*x

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The theorem stating the minimum distance between the circle and parabola -/
theorem min_distance_circle_parabola :
  ∀ x1 y1 x2 y2 : ℝ, circleEq x1 y1 → parabolaEq x2 y2 →
  distance x1 y1 x2 y2 ≥ 2 := by
  sorry

/-- The theorem stating that the minimum distance is achievable -/
theorem min_distance_achievable :
  ∃ x1 y1 x2 y2 : ℝ, circleEq x1 y1 ∧ parabolaEq x2 y2 ∧ distance x1 y1 x2 y2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_parabola_min_distance_achievable_l542_54225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2phi_value_l542_54253

-- Define the function f(x)
noncomputable def f (x φ : ℝ) : ℝ := Real.sin (Real.pi * x + φ) - 2 * Real.cos (Real.pi * x + φ)

-- Define the symmetry condition
def symmetric_about_1 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + x) = f (1 - x)

-- State the theorem
theorem sin_2phi_value (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi) 
  (h3 : symmetric_about_1 (f · φ)) : 
  Real.sin (2 * φ) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2phi_value_l542_54253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_C_l542_54234

/-- The conic section C defined by parametric equations -/
noncomputable def C : ℝ → ℝ × ℝ := fun t => (t + 1/t, t - 1/t)

/-- The eccentricity of a conic section -/
noncomputable def eccentricity (c : ℝ → ℝ × ℝ) : ℝ := sorry

/-- Theorem: The eccentricity of C is √2 -/
theorem eccentricity_of_C : eccentricity C = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_C_l542_54234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_interval_l542_54208

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 8)

-- State the theorem
theorem f_monotone_increasing_on_interval :
  MonotoneOn f (Set.Ioi 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_interval_l542_54208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_g_of_3_l542_54280

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 2 * x⁻¹ + (2 * x⁻¹) / (1 + 2 * x⁻¹)

-- State the theorem
theorem g_of_g_of_3 : g (g 3) = 585 / 368 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_g_of_3_l542_54280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_average_speed_approx_l542_54206

-- Define the driving segments
noncomputable def segment1_distance : ℝ := 360
noncomputable def segment1_speed : ℝ := 60
noncomputable def segment2_distance : ℝ := 200
noncomputable def segment2_speed : ℝ := 80
noncomputable def segment3_distance : ℝ := 150
noncomputable def segment3_speed : ℝ := 50
noncomputable def segment4_distance : ℝ := 90
noncomputable def segment4_speed : ℝ := 40

-- Calculate total distance
noncomputable def total_distance : ℝ :=
  segment1_distance + segment2_distance + segment3_distance + segment4_distance

-- Calculate total time
noncomputable def total_time : ℝ :=
  segment1_distance / segment1_speed +
  segment2_distance / segment2_speed +
  segment3_distance / segment3_speed +
  segment4_distance / segment4_speed

-- Define average speed
noncomputable def average_speed : ℝ := total_distance / total_time

-- Theorem to prove
theorem average_speed_calculation :
  average_speed = total_distance / total_time :=
by
  -- Unfold the definitions
  unfold average_speed
  -- The equality is now trivial
  rfl

-- Theorem to show that the average speed is approximately 58.18 mph
theorem average_speed_approx :
  ∃ ε > 0, |average_speed - 58.18| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_average_speed_approx_l542_54206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_fixed_line_l542_54245

/-- Represents an ellipse with parameter a -/
structure Ellipse (a : ℝ) where
  equation : ∀ x y : ℝ, x^2 / a^2 + y^2 / (1 - a^2) = 1

/-- Represents a point on the ellipse -/
structure PointOnEllipse (a : ℝ) where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / a^2 + y^2 / (1 - a^2) = 1
  in_first_quadrant : x > 0 ∧ y > 0

/-- The focal distance of the ellipse is 1 -/
def focal_distance_is_one (a : ℝ) : Prop :=
  a^2 - (1 - a^2) = (1/2)^2

/-- The foci of the ellipse -/
noncomputable def foci (a : ℝ) : ℝ × ℝ :=
  let c := Real.sqrt (2 * a^2 - 1)
  (-c, c)

/-- Q is the intersection of F₂P and the y-axis -/
noncomputable def Q (a : ℝ) (P : PointOnEllipse a) : ℝ × ℝ :=
  let (_, f₂) := foci a
  (0, P.y * f₂ / (f₂ - P.x))

/-- F₁P is perpendicular to F₁Q -/
def F₁P_perp_F₁Q (a : ℝ) (P : PointOnEllipse a) : Prop :=
  let (f₁, _) := foci a
  let q := Q a P
  (P.y / (P.x - f₁)) * (q.2 / (q.1 - f₁)) = -1

/-- The main theorem -/
theorem point_on_fixed_line (a : ℝ) (P : PointOnEllipse a) 
  (h₁ : focal_distance_is_one a) 
  (h₂ : F₁P_perp_F₁Q a P) : 
  P.x + P.y = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_fixed_line_l542_54245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_posters_made_l542_54204

-- Define the rate and time for Mario
def mario_rate : ℚ := 5
def mario_time : ℚ := 7

-- Define Samantha's rate relative to Mario's
def samantha_rate_multiplier : ℚ := 3/2
def samantha_time : ℚ := 9

-- Define Jonathan's rate relative to Samantha's and his time
def jonathan_rate_multiplier : ℚ := 2
def jonathan_time : ℚ := 6

-- Function to calculate the number of posters made
def posters_made (rate : ℚ) (time : ℚ) : ℕ :=
  (rate * time).floor.toNat

-- Theorem statement
theorem total_posters_made :
  posters_made mario_rate mario_time +
  posters_made (samantha_rate_multiplier * mario_rate) samantha_time +
  posters_made (jonathan_rate_multiplier * samantha_rate_multiplier * mario_rate) jonathan_time = 192 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_posters_made_l542_54204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisor_of_f_l542_54288

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem max_divisor_of_f :
  ∃ (m : ℕ), m = 36 ∧ 
  (∀ (n : ℕ), n > 0 → m ∣ f n) ∧
  (∀ (k : ℕ), k > 36 → ∃ (n : ℕ), n > 0 ∧ ¬(k ∣ f n)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisor_of_f_l542_54288
