import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1075_107552

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * (x - 1)

-- Define the minimum value function
noncomputable def min_value (a : ℝ) : ℝ :=
  if a ≤ 1 then 0
  else if a < 2 then a - Real.exp (a - 1)
  else a + Real.exp 1 - a * Real.exp 1

-- State the theorem
theorem f_min_value (a : ℝ) :
  ∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ min_value a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1075_107552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_f_g_inequality_implies_a_range_l1075_107525

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a * Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 1) / x - 1

-- Part I
theorem tangent_line_implies_a_value (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ = 0 ∧ (deriv (f a)) x₀ = 0) →
  a = -Real.exp 1 := by
  sorry

-- Part II
theorem f_g_inequality_implies_a_range (a : ℝ) :
  a > 0 →
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ici 3 → x₂ ∈ Set.Ici 3 → x₁ ≠ x₂ →
    |f a x₁ - f a x₂| < |g x₁ - g x₂|) →
  a ∈ Set.Ioo 0 ((2 * Real.exp 2) / 3 - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_f_g_inequality_implies_a_range_l1075_107525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_sqrt_at_4_l1075_107535

-- Define the function f as noncomputable due to its dependence on Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- Define f_inverse as a parameter instead of using it directly in the axiom
noncomputable def f_inverse : ℝ → ℝ := sorry

-- State that f_inverse is the inverse function of f
axiom f_inverse_is_inverse : 
  (∀ x, (f ∘ f_inverse) x = x) ∧ (∀ x, (f_inverse ∘ f) x = x)

-- Theorem to prove
theorem inverse_of_sqrt_at_4 : f_inverse 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_sqrt_at_4_l1075_107535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_circle_center_l1075_107529

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- Define the given line
def given_line (x y : ℝ) : Prop := 2*x + y = 0

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Theorem statement
theorem perpendicular_line_through_circle_center :
  ∀ x y : ℝ,
  perpendicular_line x y ↔ 
  (∃ c : ℝ × ℝ, (my_circle c.1 c.2 ∧ perpendicular_line c.1 c.2)) ∧
  (∀ x' y' : ℝ, given_line x' y' → (x - x') * (2*x' + y') + (y - y') * 1 = 0) :=
by
  sorry

#check perpendicular_line_through_circle_center

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_circle_center_l1075_107529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_length_l1075_107547

/-- The length of a diagonal in a 2x2 square grid -/
noncomputable def diagonal_2x2 : ℝ := Real.sqrt 8

/-- The length of a diagonal in a 1x1 square grid -/
noncomputable def diagonal_1x1 : ℝ := Real.sqrt 2

/-- The total length of segments forming the letter "X" -/
noncomputable def length_X : ℝ := 2 * diagonal_2x2

/-- The total length of segments forming the letter "Y" -/
noncomputable def length_Y : ℝ := 2 + 2 * diagonal_1x1

/-- The total length of segments forming the letter "Z" -/
noncomputable def length_Z : ℝ := 4 + diagonal_2x2

/-- The theorem stating the total length of the acronym "XYZ" -/
theorem xyz_length : length_X + length_Y + length_Z = 6 + 8 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_length_l1075_107547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1075_107574

theorem power_equality (x : ℝ) (h : (3 : ℝ)^(2*x) = 5) : (27 : ℝ)^(x + 0.5) = 5 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1075_107574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1075_107576

theorem sin_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) :
  Real.sin α = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1075_107576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_l1075_107575

noncomputable section

variable (x₁ x₂ x₃ x₄ x₅ : ℝ)
variable (c : ℝ)

noncomputable def mean (x₁ x₂ x₃ x₄ x₅ : ℝ) : ℝ := (x₁ + x₂ + x₃ + x₄ + x₅) / 5

noncomputable def variance (x₁ x₂ x₃ x₄ x₅ : ℝ) : ℝ :=
  ((x₁ - mean x₁ x₂ x₃ x₄ x₅)^2 + (x₂ - mean x₁ x₂ x₃ x₄ x₅)^2 + 
   (x₃ - mean x₁ x₂ x₃ x₄ x₅)^2 + (x₄ - mean x₁ x₂ x₃ x₄ x₅)^2 + 
   (x₅ - mean x₁ x₂ x₃ x₄ x₅)^2) / 5

theorem salary_increase 
  (h₁ : mean x₁ x₂ x₃ x₄ x₅ = 3500) 
  (h₂ : variance x₁ x₂ x₃ x₄ x₅ = 45) 
  (h₃ : c = 100) :
  mean (x₁ + c) (x₂ + c) (x₃ + c) (x₄ + c) (x₅ + c) = 3600 ∧
  variance (x₁ + c) (x₂ + c) (x₃ + c) (x₄ + c) (x₅ + c) = 45 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_l1075_107575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_relationship_total_weight_leo_current_weight_l1075_107599

/-- Leo's current weight in pounds -/
noncomputable def leo_weight : ℝ := sorry

/-- Kendra's current weight in pounds -/
noncomputable def kendra_weight : ℝ := sorry

/-- The combined weight of Leo and Kendra in pounds -/
def combined_weight : ℝ := 210

/-- Leo's weight after gaining 12 pounds -/
noncomputable def leo_weight_after_gain : ℝ := leo_weight + 12

/-- Theorem stating the relationship between Leo's and Kendra's weights -/
theorem weight_relationship : 
  leo_weight_after_gain = kendra_weight + 0.7 * kendra_weight := by sorry

/-- Theorem stating the combined weight of Leo and Kendra -/
theorem total_weight : 
  leo_weight + kendra_weight = combined_weight := by sorry

/-- Main theorem proving Leo's current weight -/
theorem leo_current_weight : 
  abs (leo_weight - 127.78) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_relationship_total_weight_leo_current_weight_l1075_107599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_m_c_equals_three_l1075_107539

/-- Two circles intersecting at points A and B, with centers on a line -/
structure IntersectingCircles where
  /-- x-coordinate of point B -/
  m : ℝ
  /-- Constant in the line equation x - y + c = 0 -/
  c : ℝ
  /-- The circles intersect at point A(1, 3) -/
  point_a : ℝ × ℝ := (1, 3)
  /-- The circles intersect at point B(m, -1) -/
  point_b : ℝ × ℝ := (m, -1)
  /-- The centers of both circles lie on the line x - y + c = 0 -/
  centers_on_line : ∃ (x y : ℝ), x - y + c = 0

/-- The sum of m and c is equal to 3 -/
theorem sum_m_c_equals_three (ic : IntersectingCircles) : ic.m + ic.c = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_m_c_equals_three_l1075_107539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_function_symmetry_l1075_107542

-- Define a type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define the properties of the three functions
theorem third_function_symmetry 
  (φ : RealFunction) 
  (inverse_φ : RealFunction)
  (third_func : RealFunction)
  (h1 : ∀ x, inverse_φ (φ x) = x)  -- inverse_φ is the inverse of φ
  (h2 : ∀ x y, third_func y = x ↔ inverse_φ (-x) = -y)  -- symmetry condition
  : third_func = λ x ↦ -φ (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_function_symmetry_l1075_107542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l1075_107526

def a : ℝ × ℝ := (2, 6)
def b (m : ℝ) : ℝ × ℝ := (m, -1)

theorem vector_relations (m : ℝ) :
  ((a.fst * (b m).fst + a.snd * (b m).snd = 0) → m = 3) ∧
  (∃ (k : ℝ), k ≠ 0 ∧ a.fst * k = (b m).fst ∧ a.snd * k = (b m).snd → m = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l1075_107526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_below_line_l1075_107558

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 10*y + 25 = 0

-- Define the line
def line_equation (y : ℝ) : Prop := y = 8

-- Define the area of the circle below the line
noncomputable def area_below_line : ℝ := 9 * Real.pi

-- Theorem statement
theorem circle_area_below_line :
  ∀ x y : ℝ, circle_equation x y → line_equation y → area_below_line = 9 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_below_line_l1075_107558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1075_107513

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2 else 
  if -1 < x ∧ x < 2 then x^2 else 0

-- State the theorem
theorem f_properties :
  (∀ x, f x ≠ 0 → x < 2) ∧
  (∀ y, ∃ x, f x = y ↔ y < 4 ∨ y ≤ 0) ∧
  (f 1 = 1) ∧
  (∀ x, f x = 3 ↔ x = Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1075_107513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l1075_107514

theorem cubic_equation_roots :
  let roots : Set ℝ := {1/8, -1/4, 1/2}
  let is_root (r : ℝ) := 64 * r^3 - 24 * r^2 - 6 * r + 1 = 0
  let is_geometric_progression (a b c : ℝ) := ∃ (q : ℝ), b = a * q ∧ c = b * q
  (∀ r ∈ roots, is_root r) ∧
  (∃ (a b c : ℝ), a ∈ roots ∧ b ∈ roots ∧ c ∈ roots ∧ is_geometric_progression a b c) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l1075_107514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l1075_107583

theorem complex_fraction_equality : 
  ((-1 + Complex.I) * (2 + Complex.I)) / (Complex.I^3) = (-1 - 3*Complex.I) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l1075_107583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_divisibility_l1075_107557

theorem prime_sum_divisibility (p : Fin 2021 → ℕ) 
  (prime_p : ∀ i, Nat.Prime (p i))
  (sum_divisible : (Finset.sum Finset.univ (λ i => (p i)^4)) % 6060 = 0) :
  4 ≤ (Finset.univ.filter (λ i => p i < 2021)).card := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_divisibility_l1075_107557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_color_is_blue_l1075_107522

def color_transform (n : ℕ) : ℕ :=
  if n ≤ 17 then 3 * n - 2 else Int.natAbs (129 - 2 * n)

def iterate_transform (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | m + 1 => color_transform (iterate_transform n m)

theorem final_color_is_blue :
  iterate_transform 5 2019 = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_color_is_blue_l1075_107522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_floor_product_72_l1075_107569

theorem unique_floor_product_72 :
  ∃! (x : ℝ), x > 0 ∧ (Int.floor x) * x = 72 ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_floor_product_72_l1075_107569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_ABO_l1075_107573

noncomputable section

-- Define the point P
def P : ℝ × ℝ := (37, 27)

-- Define the line passing through P
def line (k : ℝ) : ℝ → ℝ := λ x => k * (x - P.1) + P.2

-- Define the x-intercept (point A)
def A (k : ℝ) : ℝ × ℝ := (P.1 - P.2 / k, 0)

-- Define the y-intercept (point B)
def B (k : ℝ) : ℝ × ℝ := (0, P.2 - k * P.1)

-- Define the area of triangle ABO
def area (k : ℝ) : ℝ := (1 / 2) * (A k).1 * (B k).2

end noncomputable section

-- Theorem statement
theorem min_area_triangle_ABO :
  ∃ (min_area : ℝ), min_area = 1998 ∧ ∀ (k : ℝ), k ≠ 0 → area k ≥ min_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_ABO_l1075_107573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_sold_88_pens_gain_is_25_percent_of_total_cost_l1075_107562

/-- The number of pens sold by a trader -/
noncomputable def number_of_pens_sold (cost_per_pen : ℝ) : ℝ :=
  22 / 0.25

/-- Theorem: The trader sold 88 pens -/
theorem trader_sold_88_pens (cost_per_pen : ℝ) (cost_per_pen_pos : cost_per_pen > 0) :
  number_of_pens_sold cost_per_pen = 88 := by
  sorry

/-- The gain percentage as a decimal -/
def gain_percentage : ℝ := 0.25

/-- The number of pens whose cost equals the total gain -/
def pens_cost_equal_gain : ℝ := 22

/-- Theorem: The gain is 25% of the total cost -/
theorem gain_is_25_percent_of_total_cost (cost_per_pen : ℝ) (cost_per_pen_pos : cost_per_pen > 0) :
  pens_cost_equal_gain * cost_per_pen = gain_percentage * (number_of_pens_sold cost_per_pen * cost_per_pen) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_sold_88_pens_gain_is_25_percent_of_total_cost_l1075_107562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_18_or_50_l1075_107597

-- Define the trajectory M as a set of points (x, y) satisfying y^2 = x
def trajectoryM : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = p.1}

-- Define the line AB: y = x + 4
def lineAB : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 4}

-- Define the square ABCD
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  ab_on_lineAB : A ∈ lineAB ∧ B ∈ lineAB
  cd_on_trajectoryM : C ∈ trajectoryM ∧ D ∈ trajectoryM
  is_square : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - D.1)^2 + (A.2 - D.2)^2 ∧
              (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2 ∧
              (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

-- Helper function to calculate area (assuming square's side length is known)
def area (s : Square) : ℝ := 
  (s.A.1 - s.B.1)^2 + (s.A.2 - s.B.2)^2

-- Theorem statement
theorem square_area_is_18_or_50 (ABCD : Square) : 
  area ABCD = 18 ∨ area ABCD = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_18_or_50_l1075_107597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_average_is_61_l1075_107502

-- Define a function to represent the average of a list of numbers
noncomputable def average (xs : List ℝ) : ℝ := xs.sum / xs.length

-- Define the theorem
theorem transformed_average_is_61 
  (n : ℕ) 
  (xs : List ℝ) 
  (h : xs.length = n) 
  (h_avg : average xs = 30) : 
  average (xs.map (λ x => 2 * x + 1)) = 61 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_average_is_61_l1075_107502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_6_l1075_107545

/-- The set of numbers to draw from -/
def S : Finset ℕ := {1, 2, 3, 6}

/-- The set of pairs of numbers from S -/
def P : Finset (ℕ × ℕ) := S.product S

/-- Predicate for pairs whose product is 6 -/
def is_product_6 (pair : ℕ × ℕ) : Prop := pair.1 * pair.2 = 6

/-- The set of pairs whose product is 6 -/
def favorable_outcomes : Finset (ℕ × ℕ) := 
  P.filter (fun pair => pair.1 * pair.2 = 6)

theorem probability_product_6 :
  (favorable_outcomes.card : ℚ) / P.card = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_6_l1075_107545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_z_l1075_107507

-- Define the complex number z
noncomputable def z : ℂ := (2 + Complex.I) / (1 + Complex.I)^2

-- Theorem statement
theorem real_part_of_z : Complex.re z = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_z_l1075_107507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_online_store_commission_percentage_l1075_107509

/-- Calculates the commission percentage of an online store given the product cost,
    desired profit percentage, and observed online price. -/
theorem online_store_commission_percentage
  (cost : ℝ)
  (profit_percentage : ℝ)
  (observed_price : ℝ)
  (h1 : cost = 18)
  (h2 : profit_percentage = 20)
  (h3 : observed_price = 27) :
  (observed_price - cost * (1 + profit_percentage / 100)) / (cost * (1 + profit_percentage / 100)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_online_store_commission_percentage_l1075_107509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_satisfies_conditions_l1075_107577

-- Define the points
noncomputable def P₁ : ℝ × ℝ := (0, 2)
noncomputable def P₂ : ℝ × ℝ := (3, 0)
noncomputable def P : ℝ × ℝ := (2, 2/3)

-- Define vectors
def vector (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define a function to check if a point is on a line segment
def on_segment (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

-- Theorem statement
theorem point_satisfies_conditions : 
  on_segment P₁ P₂ P ∧ vector P₁ P = (2 * (vector P P₂).1, 2 * (vector P P₂).2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_satisfies_conditions_l1075_107577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_zero_f_solutions_l1075_107503

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2

-- Theorem 1: If f is even, then a = 0
theorem f_even_implies_a_zero (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 0 := by sorry

-- Theorem 2: If f(π/4) = √3 + 1, then the solutions to f(x) = 1 - √2 in [-π, π] are -5π/12 and 13π/12
theorem f_solutions (a : ℝ) :
  f a (Real.pi / 4) = Real.sqrt 3 + 1 →
  {x : ℝ | f a x = 1 - Real.sqrt 2 ∧ -Real.pi ≤ x ∧ x ≤ Real.pi} =
  {-5 * Real.pi / 12, 13 * Real.pi / 12} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_zero_f_solutions_l1075_107503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fifteen_terms_is_225_l1075_107570

/-- An arithmetic progression where the sum of the 4th and 12th terms is 30 -/
def ArithmeticProgression (a d : ℚ) : Prop :=
  (a + 3*d) + (a + 11*d) = 30

/-- The sum of the first n terms of an arithmetic progression -/
def SumArithmeticProgression (a d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2*a + (n - 1 : ℚ)*d)

/-- Theorem: The sum of the first 15 terms is 225 -/
theorem sum_fifteen_terms_is_225 (a d : ℚ) :
  ArithmeticProgression a d → SumArithmeticProgression a d 15 = 225 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fifteen_terms_is_225_l1075_107570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_classification_l1075_107596

/-- Represents a decimal number -/
structure Decimal where
  whole : Nat
  fractional : List Nat

/-- Checks if a decimal is infinite -/
def is_infinite (d : Decimal) : Prop :=
  d.fractional.length = Nat.succ Nat.zero

/-- Checks if a decimal is cyclic -/
def is_cyclic (d : Decimal) : Prop :=
  ∃ (n : Nat), n > 0 ∧ ∃ (cycle : List Nat), cycle.length > 0 ∧
    (∀ (i : Nat), i ≥ n → d.fractional[i]? = cycle[(i - n) % cycle.length]?)

def number_a : Decimal := ⟨1, [0, 7, 0, 7, 0, 7]⟩
def number_b : Decimal := ⟨6, [2, 8, 2, 8, 2, 8]⟩
def number_c : Decimal := ⟨3, [1, 4, 1, 5, 9, 2, 6]⟩

theorem decimal_classification :
  (is_cyclic number_a) ∧
  (is_infinite number_a ∧ is_infinite number_c) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_classification_l1075_107596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_b_value_l1075_107594

/-- An isosceles triangle with two equal sides and a base -/
structure IsoscelesTriangle where
  equal_side : ℝ
  base : ℝ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.equal_side + t.base

/-- The area of an isosceles triangle using Heron's formula -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  let s := perimeter t / 2
  Real.sqrt (s * (s - t.equal_side) * (s - t.equal_side) * (s - t.base))

/-- Two isosceles triangles are congruent if they have the same measurements -/
def is_congruent (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.equal_side = t2.equal_side ∧ t1.base = t2.base

theorem isosceles_triangle_b_value :
  ∀ (a b : ℝ),
  let t1 : IsoscelesTriangle := ⟨6, 10⟩
  let t2 : IsoscelesTriangle := ⟨a, b⟩
  perimeter t1 = perimeter t2 →
  area t1 = area t2 →
  ¬is_congruent t1 t2 →
  b = 5 := by
  sorry

#check isosceles_triangle_b_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_b_value_l1075_107594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gnome_fall_theorem_l1075_107546

/-- Gnome falling scenario -/
structure GnomeFallScenario where
  n : ℕ  -- number of gnomes
  p : ℝ  -- probability of a gnome falling
  hp : 0 < p ∧ p < 1  -- constraint on p

/-- Probability that exactly k gnomes fall -/
noncomputable def prob_k_fall (scenario : GnomeFallScenario) (k : ℕ) : ℝ :=
  scenario.p * (1 - scenario.p) ^ (scenario.n - k)

/-- Expected number of fallen gnomes -/
noncomputable def expected_fallen (scenario : GnomeFallScenario) : ℝ :=
  scenario.n + 1 - 1 / scenario.p + (1 - scenario.p) ^ (scenario.n + 1) / scenario.p

/-- Main theorem for gnome falling scenario -/
theorem gnome_fall_theorem (scenario : GnomeFallScenario) :
  (∀ k : ℕ, k ≤ scenario.n → prob_k_fall scenario k = scenario.p * (1 - scenario.p) ^ (scenario.n - k)) ∧
  expected_fallen scenario = scenario.n + 1 - 1 / scenario.p + (1 - scenario.p) ^ (scenario.n + 1) / scenario.p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gnome_fall_theorem_l1075_107546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_open_unit_interval_l1075_107533

open Set Real

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 1 / (x^k)

-- State the theorem
theorem range_of_f_on_open_unit_interval (k : ℝ) (h_k : k > 0) :
  range (f k ∘ (fun x ↦ x : Ioc 0 1 → ℝ)) = Ici 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_open_unit_interval_l1075_107533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1075_107551

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Ioo 0 2

-- Define the function composition g(x) = f(√(x+1))
noncomputable def g (x : ℝ) : ℝ := f (Real.sqrt (x + 1))

-- State the theorem
theorem domain_of_g :
  {x : ℝ | g x ∈ Set.range f} = Set.Ioc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1075_107551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1075_107511

-- Define the function f(x) = tan(x/2)
noncomputable def f (x : ℝ) : ℝ := Real.tan (x / 2)

-- State the theorem
theorem f_monotone_increasing :
  MonotoneOn f (Set.Ioo 0 Real.pi) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1075_107511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1075_107548

noncomputable def a : ℕ → ℝ
  | 0 => 3/2  -- Adding the base case for 0
  | 1 => 3/2
  | n + 1 => (1 + 1/(3^n)) * a n + 2/(n * (n + 1))

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) > a n) ∧
  (∀ n : ℕ, n ≥ 2 → a (n + 1) / a n ≤ 1 + 1/(3^n) + 2/(3*n*(n+1))) ∧
  (∀ n : ℕ, n ≥ 1 → a n < 3 * Real.sqrt (Real.exp 1)) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1075_107548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_implies_a_range_l1075_107519

theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → (x^2 + 2*(a-1)*x + 2) > (y^2 + 2*(a-1)*y + 2)) →
  a ∈ Set.Iic (-3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_implies_a_range_l1075_107519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_willey_farm_corn_cost_l1075_107521

/-- The Willey Farm Collective's land allocation and cost problem -/
theorem willey_farm_corn_cost 
  (total_land : ℕ) 
  (wheat_cost_per_acre : ℕ) 
  (total_capital : ℕ) 
  (wheat_acreage : ℕ) 
  (h1 : total_land = 4500)
  (h2 : wheat_cost_per_acre = 35)
  (h3 : total_capital = 165200)
  (h4 : wheat_acreage = 3400) : 
  (total_capital - wheat_cost_per_acre * wheat_acreage) / (total_land - wheat_acreage) = 42 := by
  sorry

#check willey_farm_corn_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_willey_farm_corn_cost_l1075_107521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_for_steve_and_leroy_l1075_107508

/-- Represents the fishing boat scenario -/
structure FishingBoat where
  distance_from_shore : ℝ  -- in miles
  water_intake_rate : ℝ    -- in gallons per minute
  sinking_threshold : ℝ    -- in gallons
  rowing_speed : ℝ         -- in miles per hour

/-- Calculates the minimum bailing rate required to reach shore without sinking -/
noncomputable def min_bailing_rate (boat : FishingBoat) : ℝ :=
  let time_to_shore := boat.distance_from_shore / boat.rowing_speed * 60  -- in minutes
  let total_water_intake := boat.water_intake_rate * time_to_shore
  (total_water_intake - boat.sinking_threshold) / time_to_shore

/-- Theorem stating the minimum bailing rate for the given scenario -/
theorem min_bailing_rate_for_steve_and_leroy :
  let boat := FishingBoat.mk 1.5 15 60 3
  min_bailing_rate boat = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_for_steve_and_leroy_l1075_107508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eldora_index_cards_l1075_107591

-- Define the cost of one box of paper clips
def paper_clip_cost : ℚ := 185 / 100

-- Define Eldora's purchase
def eldora_paper_clips : ℕ := 15
def eldora_total_cost : ℚ := 5540 / 100

-- Define Finn's purchase
def finn_paper_clips : ℕ := 12
def finn_index_cards : ℕ := 10
def finn_total_cost : ℚ := 6170 / 100

-- Theorem to prove
theorem eldora_index_cards : ∃ (n : ℕ), 
  (n : ℚ) * ((finn_total_cost - finn_paper_clips * paper_clip_cost) / finn_index_cards) + 
  eldora_paper_clips * paper_clip_cost = eldora_total_cost := by
  -- The number of index card packages Eldora bought should be 7
  use 7
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eldora_index_cards_l1075_107591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l1075_107501

/-- The slope angle of a line with equation ax + by + c = 0 -/
noncomputable def slope_angle (a b c : ℝ) : ℝ :=
  Real.arctan (-a / b) * (180 / Real.pi)

/-- Theorem: The slope angle of the line 3x + 3y + 1 = 0 is 135° -/
theorem slope_angle_of_line : slope_angle 3 3 1 = 135 := by
  -- Expand the definition of slope_angle
  unfold slope_angle
  -- Simplify the expression
  simp [Real.arctan_one, Real.pi]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l1075_107501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_no_nine_arithmetic_sequence_length_l1075_107510

/-- Checks if a natural number has the digit 9 in its decimal representation -/
def has_digit_nine (n : ℕ) : Prop :=
  ∃ k : ℕ, (n / 10^k) % 10 = 9

/-- An arithmetic sequence with positive integer first term and common difference,
    where no term contains the digit 9 in its decimal representation. -/
structure NoNineArithmeticSequence where
  first_term : ℕ+
  common_diff : ℕ+
  no_nine : ∀ n : ℕ, ¬ (has_digit_nine ((first_term : ℕ) + n * (common_diff : ℕ)))

/-- The maximum number of terms in a NoNineArithmeticSequence is 72 -/
theorem max_no_nine_arithmetic_sequence_length :
  ∀ seq : NoNineArithmeticSequence, ∃ m : ℕ, m ≤ 72 ∧
    ∀ n > m, has_digit_nine ((seq.first_term : ℕ) + n * (seq.common_diff : ℕ)) :=
by
  sorry

#check max_no_nine_arithmetic_sequence_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_no_nine_arithmetic_sequence_length_l1075_107510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_sufficient_condition_not_necessary_sufficient_but_not_necessary_l1075_107550

/-- A complex number z is purely imaginary if its real part is zero. -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

/-- The complex number z defined in terms of θ. -/
noncomputable def z (θ : ℝ) : ℂ := (Real.cos θ - Real.sin θ) * (1 + Complex.I)

/-- The condition θ = 3π/4 is sufficient for z to be purely imaginary. -/
theorem condition_sufficient :
  isPurelyImaginary (z (3 * Real.pi / 4)) := by sorry

/-- The condition θ = 3π/4 is not necessary for z to be purely imaginary. -/
theorem condition_not_necessary :
  ∃ θ, θ ≠ 3 * Real.pi / 4 ∧ isPurelyImaginary (z θ) := by sorry

/-- The main theorem combining the above results. -/
theorem sufficient_but_not_necessary :
  (∀ θ, θ = 3 * Real.pi / 4 → isPurelyImaginary (z θ)) ∧
  (∃ θ, θ ≠ 3 * Real.pi / 4 ∧ isPurelyImaginary (z θ)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_sufficient_condition_not_necessary_sufficient_but_not_necessary_l1075_107550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_not_always_equal_l1075_107589

/-- A rhombus is a quadrilateral with four equal sides, symmetry, and central symmetry -/
structure Rhombus where
  sides_equal : Bool
  symmetrical : Bool
  centrally_symmetrical : Bool

/-- Definition of a rhombus -/
def is_rhombus (q : Rhombus) : Prop :=
  q.sides_equal ∧ q.symmetrical ∧ q.centrally_symmetrical

/-- Property of equal diagonals -/
def has_equal_diagonals (q : Rhombus) : Prop := sorry

/-- Theorem: Not all rhombuses have equal diagonals -/
theorem rhombus_diagonals_not_always_equal :
  ∃ (r : Rhombus), is_rhombus r ∧ ¬has_equal_diagonals r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_not_always_equal_l1075_107589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_not_unique_l1075_107517

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  baseAngle : ℝ

/-- Given parts of an isosceles triangle -/
structure GivenParts where
  baseAngle : ℝ
  oppositeVertexSide : ℝ

/-- Predicate to check if two isosceles triangles are different -/
def areDifferentTriangles (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.leg ≠ t2.leg ∨ t1.base ≠ t2.base

/-- Theorem stating that the given parts do not uniquely determine an isosceles triangle -/
theorem isosceles_triangle_not_unique (g : GivenParts) :
  ∃ (t1 t2 : IsoscelesTriangle), 
    t1.baseAngle = g.baseAngle ∧
    2 * t1.leg * Real.sin t1.baseAngle = g.oppositeVertexSide ∧
    t2.baseAngle = g.baseAngle ∧
    2 * t2.leg * Real.sin t2.baseAngle = g.oppositeVertexSide ∧
    areDifferentTriangles t1 t2 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_not_unique_l1075_107517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_and_side_equations_l1075_107578

/-- A rhombus ABCD in the Cartesian coordinate system -/
structure Rhombus where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def is_on_line (p : ℝ × ℝ) (l : LineEquation) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

theorem rhombus_diagonal_and_side_equations 
  (ABCD : Rhombus)
  (h_A : ABCD.A = (-1, 2))
  (h_C : ABCD.C = (5, 4))
  (h_AB : is_on_line ABCD.A (LineEquation.mk 1 (-1) 3))
  (h_B : is_on_line ABCD.B (LineEquation.mk 1 (-1) 3))
  (h_D : is_on_line ABCD.D (LineEquation.mk 1 (-1) 3)) :
  ∃ (eq_BD eq_AD : LineEquation),
    eq_BD = LineEquation.mk 3 1 (-9) ∧  -- 3x + y - 9 = 0
    eq_AD = LineEquation.mk 1 7 (-13) ∧ -- x + 7y - 13 = 0
    is_on_line ABCD.B eq_BD ∧
    is_on_line ABCD.D eq_BD ∧
    is_on_line ABCD.A eq_AD ∧
    is_on_line ABCD.D eq_AD := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_and_side_equations_l1075_107578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_first_stop_distance_maria_distance_equation_l1075_107559

/-- The fraction of the total distance Maria traveled before her first stop -/
def x : ℚ := 1 / 2

/-- The total distance from Maria's starting point to her destination in miles -/
def total_distance : ℕ := 280

/-- The remaining distance after Maria's second stop in miles -/
def remaining_distance : ℕ := 105

theorem maria_first_stop_distance :
  x = 1 / 2 :=
by
  -- We use the definition of x directly
  rfl

theorem maria_distance_equation :
  (x : ℚ) * total_distance + (1 / 4 : ℚ) * (total_distance - x * total_distance) + remaining_distance = total_distance :=
by
  -- The proof of this equation is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_first_stop_distance_maria_distance_equation_l1075_107559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_percentage_is_16_percent_l1075_107579

/-- Calculates the dividend percentage given the purchase price, desired interest rate, and market value of a share -/
noncomputable def dividend_percentage (purchase_price : ℝ) (interest_rate : ℝ) (market_value : ℝ) : ℝ :=
  (purchase_price * interest_rate / market_value) * 100

/-- Theorem stating that under the given conditions, the dividend percentage is 16% -/
theorem dividend_percentage_is_16_percent 
  (purchase_price : ℝ) 
  (interest_rate : ℝ) 
  (market_value : ℝ) 
  (h1 : purchase_price = 56) 
  (h2 : interest_rate = 0.12) 
  (h3 : market_value = 42) : 
  dividend_percentage purchase_price interest_rate market_value = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_percentage_is_16_percent_l1075_107579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_range_l1075_107560

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The right focus of the hyperbola -/
noncomputable def right_focus (h : Hyperbola a b) : ℝ × ℝ :=
  (Real.sqrt (a^2 + b^2), 0)

/-- The left directrix of the hyperbola -/
noncomputable def left_directrix (h : Hyperbola a b) : ℝ := 
  -(a^2 / Real.sqrt (a^2 + b^2))

/-- A point on the hyperbola with x-coordinate 3a/2 -/
noncomputable def point_on_hyperbola (h : Hyperbola a b) : ℝ × ℝ :=
  (3 * a / 2, Real.sqrt ((9 * a^2 / 4 - a^2) * b^2 / a^2))

/-- The acute angle between the asymptotes of the hyperbola -/
noncomputable def asymptote_angle (h : Hyperbola a b) : ℝ :=
  Real.arctan (b / a)

theorem hyperbola_asymptote_angle_range {a b : ℝ} (h : Hyperbola a b) :
  let p := point_on_hyperbola h
  let f := right_focus h
  let d := left_directrix h
  (dist p f > |p.1 - d|) →
  (0 < asymptote_angle h ∧ asymptote_angle h < Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_range_l1075_107560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_profit_percentage_l1075_107564

/-- Calculate the selling price after markup and discount -/
noncomputable def sellingPrice (costPrice markup discount : ℝ) : ℝ :=
  costPrice * (1 + markup) * (1 - discount)

/-- Calculate the total revenue for an item -/
noncomputable def totalRevenue (units : ℕ) (sellingPrice : ℝ) : ℝ :=
  (units : ℝ) * sellingPrice

/-- Calculate the total cost for an item -/
noncomputable def totalCost (units : ℕ) (costPrice : ℝ) : ℝ :=
  (units : ℝ) * costPrice

/-- Calculate the profit percentage -/
noncomputable def profitPercentage (totalRevenue totalCost : ℝ) : ℝ :=
  (totalRevenue - totalCost) / totalCost * 100

theorem merchant_profit_percentage :
  let itemA_units : ℕ := 30
  let itemA_costPrice : ℝ := 10
  let itemA_markup : ℝ := 0.6
  let itemA_discount : ℝ := 0.2

  let itemB_units : ℕ := 20
  let itemB_costPrice : ℝ := 18
  let itemB_markup : ℝ := 0.8
  let itemB_discount : ℝ := 0.1

  let itemA_sellingPrice := sellingPrice itemA_costPrice itemA_markup itemA_discount
  let itemB_sellingPrice := sellingPrice itemB_costPrice itemB_markup itemB_discount

  let totalRevenue := totalRevenue itemA_units itemA_sellingPrice + totalRevenue itemB_units itemB_sellingPrice
  let totalCost := totalCost itemA_units itemA_costPrice + totalCost itemB_units itemB_costPrice

  let overallProfitPercentage := profitPercentage totalRevenue totalCost

  ∃ ε > 0, |overallProfitPercentage - 46.55| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_profit_percentage_l1075_107564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l1075_107592

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_properties (a b : ℝ) :
  f a b 1 = 2 ∧ f' a b (1/3) = 0 →
  (a = 4 ∧ b = -3) ∧
  (∀ x, x < -3 → (f' 4 (-3)) x > 0) ∧
  (∀ x, x > 1/3 → (f' 4 (-3)) x > 0) ∧
  (∀ x, -3 < x ∧ x < 1/3 → (f' 4 (-3)) x < 0) ∧
  (∀ x, x ∈ Set.Icc (-1) 1 → f 4 (-3) x ≤ 6) ∧
  (∀ x, x ∈ Set.Icc (-1) 1 → f 4 (-3) x ≥ -4/27) ∧
  (∃ x, x ∈ Set.Icc (-1) 1 ∧ f 4 (-3) x = 6) ∧
  (∃ x, x ∈ Set.Icc (-1) 1 ∧ f 4 (-3) x = -4/27) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l1075_107592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_approximation_l1075_107500

theorem log_sum_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |Real.log 3 / Real.log 10 + 3 * (Real.log 2 / Real.log 10) + 2 * (Real.log 5 / Real.log 10) + 
   4 * (Real.log 3 / Real.log 10) + Real.log 9 / Real.log 10 - 5.34| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_approximation_l1075_107500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_passing_mark_is_200_l1075_107534

/-- The passing mark for an exam -/
def passing_mark : ℚ := 200

/-- The total number of marks for the exam -/
def total_marks : ℚ := 500

/-- Theorem stating that the passing mark is 200 -/
theorem passing_mark_is_200 :
  (0.30 * total_marks = passing_mark - 50) ∧
  (0.45 * total_marks = passing_mark + 25) →
  passing_mark = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_passing_mark_is_200_l1075_107534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l1075_107520

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given side lengths can form a valid triangle -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Calculates the semi-perimeter of a triangle -/
noncomputable def semi_perimeter (t : Triangle) : ℝ :=
  (t.a + t.b + t.c) / 2

/-- Generates the next triangle in the sequence based on the current triangle -/
noncomputable def next_triangle (t : Triangle) : Triangle :=
  let s := semi_perimeter t
  { a := s - t.a
  , b := s - t.b
  , c := s - t.c }

/-- Represents the sequence of triangles -/
noncomputable def triangle_sequence : ℕ → Triangle
  | 0 => { a := 1010, b := 1011, c := 1012 }
  | n + 1 => next_triangle (triangle_sequence n)

/-- The theorem to be proved -/
theorem last_triangle_perimeter :
  ∃ n : ℕ,
    (is_valid_triangle (triangle_sequence n) ∧
     ¬is_valid_triangle (triangle_sequence (n + 1))) ∧
    (triangle_sequence n).a + (triangle_sequence n).b + (triangle_sequence n).c = 1526 / 128 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l1075_107520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermats_little_theorem_l1075_107588

theorem fermats_little_theorem (p : ℕ) (a : ℕ) 
  (hp : Nat.Prime p) (ha : ¬(p ∣ a)) : 
  a^(p-1) ≡ 1 [ZMOD p] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermats_little_theorem_l1075_107588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_root_l1075_107587

theorem modulus_of_complex_root (z : ℂ) (h : 3 * z^4 - 2 * Complex.I * z^3 - 2 * z + 3 * Complex.I = 0) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_root_l1075_107587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_theorem_l1075_107538

/-- Given points P, Q, R, S on a line, this function represents the conditions
    for forming a triangle by rotating PQ and RS. -/
def triangle_formation_conditions (a b c : ℝ) (angle_q : ℝ) : Prop :=
  2 * b > c ∧ c > a ∧ c > b ∧ angle_q < 2 * Real.pi / 3

/-- Theorem stating the necessary and sufficient conditions for triangle formation. -/
theorem triangle_formation_theorem
  (a b c : ℝ)
  (h_collinear : ∃ (P Q R S : ℝ × ℝ), P.1 < Q.1 ∧ Q.1 < R.1 ∧ R.1 < S.1)
  (h_distances : ∃ (P Q R S : ℝ × ℝ),
    ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt = a ∧
    ((P.1 - R.1)^2 + (P.2 - R.2)^2).sqrt = b ∧
    ((P.1 - S.1)^2 + (P.2 - S.2)^2).sqrt = c)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (angle_q : ℝ), triangle_formation_conditions a b c angle_q ↔
    ∃ (P' Q R S' : ℝ × ℝ),
      ((P'.1 - Q.1)^2 + (P'.2 - Q.2)^2).sqrt = a ∧
      ((Q.1 - R.1)^2 + (Q.2 - R.2)^2).sqrt = b - a ∧
      ((R.1 - S'.1)^2 + (R.2 - S'.2)^2).sqrt = c - b ∧
      (P' ≠ Q ∧ Q ≠ R ∧ R ≠ S') ∧
      angle_q = Real.arccos ((a^2 + (b-a)^2 - (c-b)^2) / (2*a*(b-a))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_theorem_l1075_107538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_sum_l1075_107524

-- Define the polynomial Q(z)
def Q (p q r : ℝ) (z : ℂ) : ℂ := z^3 + p*z^2 + q*z + r

-- Define the existence of w
def exists_w (p q r : ℝ) : Prop :=
  ∃ w : ℂ, ∀ z : ℂ, Q p q r z = 0 ↔ z = w + 4*Complex.I ∨ z = w + 10*Complex.I ∨ z = 3*w - 5

-- State the theorem
theorem cubic_roots_sum (p q r : ℝ) (h : exists_w p q r) : p + q + r = -150.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_sum_l1075_107524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_sample_size_l1075_107543

/-- Represents the total number of students in the sample -/
def total_students : ℕ := sorry

/-- Represents the number of freshman students -/
def freshmen : ℕ := sorry

/-- Represents the number of sophomore students -/
def sophomores : ℕ := sorry

/-- Represents the number of junior students -/
def juniors : ℕ := sorry

/-- Represents the number of senior students -/
def seniors : ℕ := sorry

theorem student_sample_size :
  (freshmen + sophomores + juniors + seniors = total_students) →
  (juniors = (22 : ℚ) / 100 * total_students) →
  (freshmen + juniors + seniors = (74 : ℚ) / 100 * total_students) →
  (seniors = 160) →
  (freshmen = sophomores + 48) →
  (total_students = 431) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_sample_size_l1075_107543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l1075_107572

open Real

noncomputable def diff_eq (y : ℝ → ℝ) (x : ℝ) : Prop :=
  deriv (deriv y) x + y x = x * sin x

noncomputable def general_solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  C₁ * cos x + C₂ * sin x - (x^2 / 4) * cos x + (x / 4) * sin x

theorem general_solution_satisfies_diff_eq (C₁ C₂ : ℝ) :
  ∀ x, diff_eq (general_solution C₁ C₂) x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l1075_107572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1075_107563

-- Define the function f(x) = (1/2)^(x^2 - 2x)
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (x^2 - 2*x)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1075_107563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_score_proof_l1075_107568

def min_correct_answers_for_score (total_questions : ℕ) 
  (correct_points : ℕ) (incorrect_deduction : ℕ) (target_score : ℕ) : ℕ :=
  let min_correct := 
    (((target_score + total_questions + incorrect_deduction - 1) : ℚ) / 
     (correct_points + incorrect_deduction : ℚ)).ceil.toNat
  min_correct

theorem competition_score_proof :
  min_correct_answers_for_score 30 5 1 100 = 22 := by
  unfold min_correct_answers_for_score
  -- The proof steps would go here
  sorry

#eval min_correct_answers_for_score 30 5 1 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_score_proof_l1075_107568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reeyas_weighted_average_l1075_107536

/-- Calculate the weighted average of scores given their weights -/
noncomputable def weighted_average (scores : List ℝ) (weights : List ℝ) : ℝ :=
  (List.sum (List.zipWith (· * ·) scores weights)) / (List.sum weights)

/-- The problem statement -/
theorem reeyas_weighted_average :
  let scores : List ℝ := [65, 67, 76, 82, 85]
  let weights : List ℝ := [2, 3, 1, 4, 1.5]
  weighted_average scores weights = 75 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reeyas_weighted_average_l1075_107536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_l1075_107506

noncomputable def f (x : ℝ) : ℝ := (3 * Real.exp (abs x) - x * Real.cos x) / Real.exp (abs x)

def interval : Set ℝ := Set.Icc (-Real.pi / 2) (Real.pi / 2)

theorem max_min_sum (p q : ℝ) 
  (hp : ∀ x, x ∈ interval → f x ≤ p) 
  (hq : ∀ x, x ∈ interval → q ≤ f x) 
  (hpq : ∃ x y, x ∈ interval ∧ y ∈ interval ∧ f x = p ∧ f y = q) : 
  p + q = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_l1075_107506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_reimbursement_correct_l1075_107561

/-- The amount Carla should receive from LeRoy and Bernardo to equally share expenses -/
noncomputable def carla_reimbursement (L B C X : ℝ) : ℝ := (L + B - 2*C - 2*X) / 3

/-- Theorem stating that carla_reimbursement is the correct amount for equal cost sharing -/
theorem carla_reimbursement_correct (L B C X : ℝ) :
  let total_cost := L + B + C + X
  let equal_share := total_cost / 3
  carla_reimbursement L B C X = equal_share - (C + X) :=
by
  -- Unfold the definition of carla_reimbursement
  unfold carla_reimbursement
  -- Simplify the expression
  simp
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_reimbursement_correct_l1075_107561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_weight_is_60_l1075_107556

/-- The weight of a round weight -/
def round_weight : ℝ := sorry

/-- The weight of a triangular weight -/
def triangular_weight : ℝ := sorry

/-- The weight of the rectangular weight -/
def rectangular_weight : ℝ := 90

/-- First balance condition: 1 round + 1 triangular = 3 round -/
axiom balance1 : round_weight + triangular_weight = 3 * round_weight

/-- Second balance condition: 4 round + 1 triangular = 1 triangular + 1 round + 1 rectangular -/
axiom balance2 : 4 * round_weight + triangular_weight = triangular_weight + round_weight + rectangular_weight

theorem triangular_weight_is_60 : triangular_weight = 60 := by
  sorry

#check triangular_weight_is_60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_weight_is_60_l1075_107556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_inequality_solution_l1075_107585

-- Define the function f(x) as described in the problem
noncomputable def f (x : ℝ) : ℝ := 
  if x^2 ≤ 4 then Real.sqrt (1 - x^2 / 4) else 0

-- Define the solution set
def solution_set : Set ℝ :=
  {x : ℝ | (-Real.sqrt 2 < x ∧ x < 0) ∨ (Real.sqrt 2 < x ∧ x ≤ 2)}

-- State the theorem
theorem ellipse_inequality_solution :
  ∀ x : ℝ, f x < f (-x) + x ↔ x ∈ solution_set :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_inequality_solution_l1075_107585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1075_107504

-- Define the vertices of the triangle
def P : ℝ × ℝ := (-3, 4)
def Q : ℝ × ℝ := (4, 3)
def R : ℝ × ℝ := (1, -6)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the perimeter of the triangle
noncomputable def perimeter : ℝ :=
  distance P Q + distance Q R + distance R P

-- Theorem statement
theorem triangle_perimeter :
  perimeter = Real.sqrt 50 + Real.sqrt 90 + Real.sqrt 116 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1075_107504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_draw_count_l1075_107595

/-- The number of ways to draw three tickets from a bag of 2015 tickets 
    numbered 1 to 2015, such that the drawn numbers a, b, c satisfy 
    1 ≤ a < b < c ≤ 2015 and a + b + c = 2018 -/
theorem ticket_draw_count : 
  (Finset.filter 
    (fun t : Fin 2015 × Fin 2015 × Fin 2015 => 
      let a := t.1.val + 1
      let b := t.2.1.val + 1
      let c := t.2.2.val + 1
      a < b ∧ b < c ∧ a + b + c = 2018)
    (Finset.univ.product (Finset.univ.product Finset.univ))).card = 338352 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_draw_count_l1075_107595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_R_equals_nine_l1075_107528

noncomputable def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 0, 3]

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := 
  !![(Real.sqrt 2) / 2, -(Real.sqrt 2) / 2; (Real.sqrt 2) / 2, (Real.sqrt 2) / 2]

noncomputable def R : Matrix (Fin 2) (Fin 2) ℝ := rotation_matrix * dilation_matrix

theorem det_R_equals_nine : Matrix.det R = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_R_equals_nine_l1075_107528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_sum_exists_l1075_107555

theorem three_sum_exists (n : ℤ) (X : Finset ℤ) 
  (h1 : X.card = n + 2)
  (h2 : ∀ x ∈ X, |x| ≤ n) : 
  ∃ a b c, a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ c = a + b := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_sum_exists_l1075_107555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l1075_107518

-- Define the point M
noncomputable def M : ℝ × ℝ := (-3, -3/2)

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the chord length
def chord_length : ℝ := 8

-- Define the line equations
def line1 (x : ℝ) : Prop := x + 3 = 0
def line2 (x y : ℝ) : Prop := 3*x + 4*y + 15 = 0

-- Theorem statement
theorem chord_line_equation :
  ∃ (x y : ℝ), 
    (x = M.1 ∧ y = M.2) ∨ 
    (circle_eq x y ∧ 
     (∃ (x' y' : ℝ), circle_eq x' y' ∧ 
      ((x - x')^2 + (y - y')^2 = chord_length^2/4) ∧
      (line1 x ∨ line2 x y))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l1075_107518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_division_l1075_107581

/-- Definition of a convex polygon -/
def ConvexPolygon (n : ℕ) : Type := sorry

/-- Definition of a triangle -/
def Triangle : Type := sorry

/-- Definition of colors -/
inductive Color : Type
| White : Color
| Black : Color

/-- Check if a triangle is on the perimeter of a polygon -/
def OnPerimeter {n : ℕ} (P : ConvexPolygon n) (t : Triangle) : Prop := sorry

/-- Check if two triangles share a segment -/
def ShareSegment (t1 t2 : Triangle) : Prop := sorry

/-- Check if a set of triangles covers a polygon -/
def UnionCovers {n : ℕ} (T : Set Triangle) (P : ConvexPolygon n) : Prop := sorry

/-- A convex n-gon can be divided into white and black triangles with specific conditions -/
theorem convex_polygon_division (n : ℕ) (h : n ≥ 3) :
  ∀ (P : ConvexPolygon n),
    ∃ (T : Set Triangle),
      (∀ t : Triangle, t ∈ T → (Color.White = White → ¬OnPerimeter P t)) ∧
      (∀ t1 t2 : Triangle, t1 ∈ T → t2 ∈ T → t1 ≠ t2 → Color.White = Color.White → ¬ShareSegment t1 t2) ∧
      (UnionCovers T P) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_division_l1075_107581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_of_p_l1075_107567

/-- A polynomial with only even powers of x -/
def EvenPowerPoly (p : Polynomial ℝ) : Prop :=
  ∀ (i : ℕ), Polynomial.coeff p i ≠ 0 → Even i

/-- The numerator of our rational function -/
noncomputable def numerator : Polynomial ℝ :=
  3 * Polynomial.X^8 + 4 * Polynomial.X^7 - 5 * Polynomial.X^3 - 2

/-- The rational function has a horizontal asymptote -/
def has_horizontal_asymptote (p : Polynomial ℝ) : Prop :=
  Polynomial.degree p ≥ Polynomial.degree numerator

theorem smallest_degree_of_p (p : Polynomial ℝ) 
  (h1 : EvenPowerPoly p) 
  (h2 : has_horizontal_asymptote p) : 
  Polynomial.degree p = 8 ∧ 
  ∀ (q : Polynomial ℝ), EvenPowerPoly q → has_horizontal_asymptote q → 
    Polynomial.degree q ≥ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_of_p_l1075_107567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_generic_tees_per_package_l1075_107540

/-- The number of generic golf tees in one package -/
def generic_tees_per_package : ℕ := 12

/-- The number of golfers -/
def num_golfers : ℕ := 4

/-- The minimum number of tees each golfer needs -/
def min_tees_per_golfer : ℕ := 20

/-- The maximum number of generic tee packages Bill can buy -/
def max_generic_packages : ℕ := 2

/-- The number of aero flight tee packages Bill must buy -/
def aero_packages : ℕ := 28

/-- The number of tees in each aero flight tee package -/
def aero_tees_per_package : ℕ := 2

theorem min_generic_tees_per_package :
  generic_tees_per_package = 12 :=
by
  -- The proof goes here
  sorry

#eval generic_tees_per_package

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_generic_tees_per_package_l1075_107540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_theorem_l1075_107531

/-- A quadratic function -/
class QuadraticFunction (f : ℝ → ℝ) : Prop where
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The x-coordinate of a point where a function intersects the x-axis -/
def XIntercept (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0

/-- The x-coordinate of the vertex of a quadratic function -/
def Vertex (f : ℝ → ℝ) (x : ℝ) : Prop :=
  QuadraticFunction f → ∀ y, f x ≤ f y ∨ f x ≥ f y

theorem quadratic_functions_theorem (f g : ℝ → ℝ) (x₁ x₂ x₃ x₄ : ℝ) :
  QuadraticFunction f →
  QuadraticFunction g →
  (∀ x, g x = -f (120 - x)) →
  (∃ v, Vertex f v ∧ g v = 0) →
  x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ →
  XIntercept f x₁ ∨ XIntercept g x₁ →
  XIntercept f x₂ ∨ XIntercept g x₂ →
  XIntercept f x₃ ∨ XIntercept g x₃ →
  XIntercept f x₄ ∨ XIntercept g x₄ →
  x₃ - x₂ = 180 →
  x₄ - x₁ = 540 + 360 * Real.sqrt 2 := by
  sorry

#check quadratic_functions_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_theorem_l1075_107531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cylinder_radius_for_given_crate_l1075_107590

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the maximum radius of a cylindrical tank that can fit in the crate -/
noncomputable def max_cylinder_radius (crate : CrateDimensions) : ℝ :=
  min (min crate.length crate.width) crate.height / 2

/-- Theorem stating that the maximum radius of a cylindrical tank in the given crate is 8 feet -/
theorem max_cylinder_radius_for_given_crate :
  let crate := CrateDimensions.mk 12 16 18
  max_cylinder_radius crate = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cylinder_radius_for_given_crate_l1075_107590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_left_faces_dots_l1075_107505

structure Die where
  faces : Fin 6 → Nat
  three_dot_face : ∃ f, faces f = 3
  two_dot_faces : ∃ f₁ f₂, f₁ ≠ f₂ ∧ faces f₁ = 2 ∧ faces f₂ = 2
  one_dot_faces : ∀ f, faces f ≠ 3 → faces f ≠ 2 → faces f = 1

def touching_faces_same_dots (d₁ d₂ : Die) (f₁ f₂ : Fin 6) : Prop :=
  d₁.faces f₁ = d₂.faces f₂

structure PiStructure where
  dice : Fin 7 → Die
  touching_faces : ∀ i j, i ≠ j → ∃ f₁ f₂, touching_faces_same_dots (dice i) (dice j) f₁ f₂

def left_faces (ps : PiStructure) : Fin 3 → Nat :=
  sorry

theorem left_faces_dots (ps : PiStructure) :
  left_faces ps 0 = 2 ∧ left_faces ps 1 = 2 ∧ left_faces ps 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_left_faces_dots_l1075_107505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_ratio_l1075_107523

-- Define the points
variable (A B C D O E : EuclideanSpace ℝ (Fin 2))

-- Define the properties of the parallelogram and other given conditions
def is_parallelogram (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def angle (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry
def is_center_of_circumcircle (O A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def line_intersects_angle_bisector (B O D E : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def exterior_angle_bisector (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- State the theorem
theorem parallelogram_ratio 
  (h1 : is_parallelogram A B C D)
  (h2 : angle A B C = 60)
  (h3 : is_center_of_circumcircle O A B C)
  (h4 : line_intersects_angle_bisector B O D E)
  (h5 : exterior_angle_bisector A B C D) :
  dist B O / dist O E = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_ratio_l1075_107523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elastic_collision_velocity_ratio_l1075_107537

/-- Represents the ratio of final to initial velocity in the elastic collision scenario -/
noncomputable def velocity_ratio : ℝ := 1 / Real.sqrt 6

theorem elastic_collision_velocity_ratio 
  (m : ℝ) 
  (v₀ : ℝ) 
  (T_max : ℝ) 
  (h₁ : m > 0) 
  (h₂ : v₀ > 0) 
  (h₃ : T_max > 0) :
  ∃ (v_f : ℝ), v_f / v₀ = velocity_ratio := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elastic_collision_velocity_ratio_l1075_107537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_other_x_coordinates_is_12_l1075_107527

-- Define a rectangle type
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ

-- Define the property of opposite vertices
def are_opposite_vertices (r : Rectangle) : Prop :=
  r.v1 = (2, 15) ∧ r.v2 = (10, -6)

-- Define the sum of x-coordinates of the other two vertices
noncomputable def sum_other_x_coordinates (r : Rectangle) : ℝ :=
  2 * ((r.v1.fst + r.v2.fst) / 2)

-- Theorem statement
theorem sum_other_x_coordinates_is_12 (r : Rectangle) 
  (h : are_opposite_vertices r) : 
  sum_other_x_coordinates r = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_other_x_coordinates_is_12_l1075_107527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_concentration_calculation_l1075_107541

-- Define the vessels
structure Vessel where
  capacity : ℚ
  concentration : ℚ

-- Define the problem parameters
def vessels : List Vessel := [
  ⟨2, 30/100⟩,
  ⟨6, 40/100⟩,
  ⟨4, 25/100⟩,
  ⟨3, 35/100⟩,
  ⟨5, 20/100⟩,
  ⟨7, 50/100⟩
]

def largeContainerCapacity : ℚ := 30

-- Calculate the total alcohol and total volume
def totalAlcohol : ℚ := vessels.foldl (fun acc v => acc + v.capacity * v.concentration) 0
def totalVesselVolume : ℚ := vessels.foldl (fun acc v => acc + v.capacity) 0

-- Define the theorem
theorem alcohol_concentration_calculation :
  let finalConcentration := totalAlcohol / largeContainerCapacity
  (finalConcentration * 10000).floor / 100 = 3183 / 100 := by
  sorry

#eval totalAlcohol
#eval totalVesselVolume
#eval (totalAlcohol / largeContainerCapacity)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_concentration_calculation_l1075_107541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l1075_107516

theorem size_relationship : 
  let a := ((-2023 : ℤ) ^ (0 : ℕ) : ℚ)
  let b := -(1 / 10 : ℚ)
  let c := (-5/3 : ℚ)^2
  c > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l1075_107516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_is_unit_circle_C₁_C₂_intersection_l1075_107571

-- Define the parametric curve C₁
noncomputable def C₁ (k : ℝ) : ℝ → ℝ × ℝ := fun t ↦ (Real.cos t ^ k, Real.sin t ^ k)

-- Define the polar curve C₂
noncomputable def C₂ : ℝ → ℝ × ℝ := fun θ ↦ 
  let ρ := 3 / (4 * Real.cos θ - 16 * Real.sin θ)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Theorem 1: C₁ is a unit circle when k = 1
theorem C₁_is_unit_circle : 
  ∀ t : ℝ, (C₁ 1 t).1 ^ 2 + (C₁ 1 t).2 ^ 2 = 1 := by sorry

-- Theorem 2: Intersection of C₁ and C₂ when k = 4
theorem C₁_C₂_intersection : 
  ∃ t : ℝ, C₁ 4 t = (1/4, 1/4) ∧ 4 * (1/4) - 16 * (1/4) + 3 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_is_unit_circle_C₁_C₂_intersection_l1075_107571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_technician_round_trip_completion_l1075_107553

/-- Represents the time required for a task at a stop or service center -/
structure TaskTime where
  required : ℚ
  completed : ℚ

/-- Represents the round-trip journey -/
structure RoundTrip where
  first_stop : TaskTime
  second_stop : TaskTime
  third_stop : TaskTime
  service_center : TaskTime

/-- Calculates the percentage of the round-trip completed based on task times -/
def round_trip_completion_percentage (trip : RoundTrip) : ℚ :=
  let total_required := trip.first_stop.required + trip.second_stop.required + 
                        trip.third_stop.required + trip.service_center.required
  let total_completed := trip.first_stop.completed + trip.second_stop.completed + 
                         trip.third_stop.completed + trip.service_center.completed
  (total_completed / total_required) * 100

/-- The main theorem stating that the technician has completed 80% of the round-trip -/
theorem technician_round_trip_completion : 
  let trip := RoundTrip.mk
    (TaskTime.mk 30 (30 * 1/2))
    (TaskTime.mk 45 (45 * 4/5))
    (TaskTime.mk 15 (15 * 3/5))
    (TaskTime.mk 60 60)
  round_trip_completion_percentage trip = 80 := by
  sorry

#eval round_trip_completion_percentage (RoundTrip.mk
  (TaskTime.mk 30 (30 * 1/2))
  (TaskTime.mk 45 (45 * 4/5))
  (TaskTime.mk 15 (15 * 3/5))
  (TaskTime.mk 60 60))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_technician_round_trip_completion_l1075_107553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1075_107544

-- Define the ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  eccentricity a b = Real.sqrt 2 / 2 →
  ellipse a b 0 1 →
  (∃ x y : ℝ, ellipse a b x y ∧ distance 0 1 x y = 4 * Real.sqrt 2 / 3) →
  (a = Real.sqrt 2 ∧ b = 1 ∧
   ∃ x y : ℝ, (x = 4/3 ∨ x = -4/3) ∧ y = -1/3 ∧
               ellipse (Real.sqrt 2) 1 x y ∧
               distance 0 1 x y = 4 * Real.sqrt 2 / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1075_107544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1075_107580

/-- The constant term in the expansion of (x^2 - 3x + 4/x)(1 - 1/√x)^5 is -25 -/
theorem constant_term_expansion : ∃ (f : ℝ → ℝ), 
  (∀ x : ℝ, x ≠ 0 → f x = (x^2 - 3*x + 4/x) * (1 - 1/Real.sqrt x)^5) ∧ 
  (∃ c : ℝ, c = -25 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |f x - c| < ε) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1075_107580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unsolvable_cube_l1075_107549

/-- Represents a 4x4x4 cube of integers -/
def Cube := Fin 4 → Fin 4 → Fin 4 → ℤ

/-- Checks if a position in the cube is valid -/
def is_valid_pos (i j k : Fin 4) : Prop :=
  i.val < 4 ∧ j.val < 4 ∧ k.val < 4

/-- Represents a move in the cube -/
def move (c : Cube) (i j k : Fin 4) : Cube := λ x y z ↦
  if (x = i ∧ y = j ∧ (z.val = k.val + 1 ∨ z.val = k.val - 1)) ∨
     (x = i ∧ (y.val = j.val + 1 ∨ y.val = j.val - 1) ∧ z = k) ∨
     ((x.val = i.val + 1 ∨ x.val = i.val - 1) ∧ y = j ∧ z = k)
  then c x y z + 1
  else c x y z

/-- Checks if all integers in the cube are divisible by 3 -/
def all_divisible_by_three (c : Cube) : Prop :=
  ∀ i j k, is_valid_pos i j k → (c i j k) % 3 = 0

/-- Main theorem: There exists an initial cube configuration such that
    it's impossible to reach a state where all integers are divisible by 3 -/
theorem exists_unsolvable_cube : 
  ∃ (initial : Cube), ¬∃ (moves : List (Fin 4 × Fin 4 × Fin 4)),
    all_divisible_by_three (moves.foldl (λ c m ↦ move c m.1 m.2.1 m.2.2) initial) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unsolvable_cube_l1075_107549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sugar_calculation_l1075_107530

/-- Represents the total amount of sugar in kilograms -/
def totalSugar : ℝ := 0

/-- Represents the amount of sugar sold at 8% profit in kilograms -/
def sugarAt8Percent : ℝ := 0

/-- The profit rate for the first part of sugar -/
def profitRate1 : ℝ := 0.08

/-- The profit rate for the second part of sugar -/
def profitRate2 : ℝ := 0.18

/-- The overall profit rate -/
def overallProfitRate : ℝ := 0.14

/-- The amount of sugar sold at 18% profit in kilograms -/
def sugarAt18Percent : ℝ := 600

theorem total_sugar_calculation :
  (profitRate1 * sugarAt8Percent + profitRate2 * sugarAt18Percent = overallProfitRate * totalSugar) →
  (sugarAt8Percent + sugarAt18Percent = totalSugar) →
  (totalSugar = 1000) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sugar_calculation_l1075_107530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2θ_value_l1075_107593

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x * (Real.cos x - Real.sin x) + 3

-- Define θ
noncomputable def θ : ℝ := Real.arccos (-(Real.sqrt 7 + 1) / 4) / 2

-- State the theorem
theorem cos_2θ_value :
  (∀ x ∈ Set.Icc 0 θ, f x ∈ Set.Icc 0 (2 * Real.sqrt 2 + 1)) →
  Real.cos (2 * θ) = -(Real.sqrt 7 + 1) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2θ_value_l1075_107593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_O_on_C_when_c_zero_O_inside_C_iff_c_positive_C_tangent_to_median_l1075_107515

-- Define the line l and circle C
def line_l (a b x y : ℝ) : Prop := x / a + y / b = 1

def circle_C (a b c x y : ℝ) : Prop := x^2 + y^2 - a*x - b*y - c = 0

-- Define point O
def point_O : ℝ × ℝ := (0, 0)

-- Define the median parallel to AB
def median_AB (a b : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ 3*x + 3*y + 4 = 0

-- Theorem 1: O lies on C when c = 0
theorem O_on_C_when_c_zero (a b : ℝ) :
  circle_C a b 0 (point_O.1) (point_O.2) := by sorry

-- Theorem 2: O inside C iff c > 0
theorem O_inside_C_iff_c_positive (a b c : ℝ) :
  (∃ ε > 0, ∀ x y, (x - point_O.1)^2 + (y - point_O.2)^2 < ε^2 →
    circle_C a b c x y) ↔ c > 0 := by sorry

-- Theorem 3: C tangent to median when a = b = c = -8/3
theorem C_tangent_to_median (a b c : ℝ) (h : a = -8/3 ∧ b = -8/3 ∧ c = -8/3) :
  ∃ x y, circle_C a b c x y ∧ median_AB a b x y ∧
    (∀ x' y', circle_C a b c x' y' ∧ median_AB a b x' y' → (x', y') = (x, y)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_O_on_C_when_c_zero_O_inside_C_iff_c_positive_C_tangent_to_median_l1075_107515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_savings_l1075_107584

def rent : ℝ := 5000
def milk : ℝ := 1500
def groceries : ℝ := 4500
def education : ℝ := 2500
def petrol : ℝ := 2000
def miscellaneous : ℝ := 6100

def total_expenses : ℝ := rent + milk + groceries + education + petrol + miscellaneous

def savings_rate : ℝ := 0.1

theorem kishore_savings :
  ∃ (salary : ℝ),
    salary > 0 ∧
    salary = total_expenses + savings_rate * salary ∧
    Int.floor (savings_rate * salary) = 2733 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_savings_l1075_107584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_equality_l1075_107598

variable (a b c d : ℝ)

def x₁ (a b c d : ℝ) : ℝ := b + c + d
def x₂ (a b c d : ℝ) : ℝ := -(a + b + c)
def x₃ (a b c d : ℝ) : ℝ := a - d
def y₁ (a b c d : ℝ) : ℝ := a + c + d
def y₂ (a b c d : ℝ) : ℝ := -(a + b + d)
def y₃ (a b c d : ℝ) : ℝ := b - c

def p₁ (a b c d : ℝ) : ℝ := x₁ a b c d * x₂ a b c d + x₂ a b c d * x₃ a b c d + x₃ a b c d * x₁ a b c d
def p₂ (a b c d : ℝ) : ℝ := y₁ a b c d * y₂ a b c d + y₂ a b c d * y₃ a b c d + y₃ a b c d * y₁ a b c d

theorem polynomial_roots_equality : p₁ a b c d = p₂ a b c d ↔ a * d = b * c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_equality_l1075_107598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1075_107565

-- Define the train's speed in km/hr
noncomputable def train_speed_kmh : ℝ := 54

-- Define the train's length in meters
noncomputable def train_length_m : ℝ := 135

-- Define the conversion factor from km/hr to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Theorem to prove
theorem train_crossing_time :
  (train_length_m / (train_speed_kmh * kmh_to_ms)) = 9 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1075_107565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1075_107566

theorem trigonometric_identity (α β γ : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : 0 < γ ∧ γ < π/2)
  (h4 : Real.sin β + Real.sin γ = Real.sin α)
  (h5 : Real.cos α + Real.cos γ = Real.cos β) :
  Real.cos (β - α) = 1/2 ∧ β - α = -π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1075_107566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_squares_l1075_107586

/-- Line l: 3x - 4y + m = 0 passes through point (-1, 2) -/
def line_l (x y : ℝ) (m : ℝ) : Prop :=
  3 * x - 4 * y + m = 0 ∧ 3 * (-1) - 4 * 2 + m = 0

/-- Curve G in polar coordinates: ρ = 2√2 sin(θ + π/4) -/
def curve_G_polar (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi/4)

/-- Curve G in Cartesian coordinates: (x-1)² + (y-1)² = 2 -/
def curve_G_cartesian (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

/-- Square OABC inscribed in curve G -/
def square_OABC (O A B C : ℝ × ℝ) : Prop :=
  O = (0, 0) ∧ A = (2, 0) ∧ B = (2, 2) ∧ C = (0, 2) ∧
  curve_G_cartesian A.1 A.2 ∧ curve_G_cartesian B.1 B.2 ∧
  curve_G_cartesian C.1 C.2

/-- The minimum value of PO² + PA² + PB² + PC² is 24 -/
theorem min_sum_of_squares (O A B C : ℝ × ℝ) (m : ℝ) :
  line_l (-1) 2 m →
  square_OABC O A B C →
  ∃ (min : ℝ), min = 24 ∧
    ∀ (P : ℝ × ℝ), line_l P.1 P.2 m →
      (P.1 - O.1)^2 + (P.2 - O.2)^2 +
      (P.1 - A.1)^2 + (P.2 - A.2)^2 +
      (P.1 - B.1)^2 + (P.2 - B.2)^2 +
      (P.1 - C.1)^2 + (P.2 - C.2)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_squares_l1075_107586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_length_l1075_107512

-- Define the points in ℝ²
variable (A B C D E : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h1 : ‖A - E‖ + ‖E - C‖ = ‖A - C‖)
variable (h2 : ‖B - D‖ + ‖D - C‖ = ‖B - C‖)
variable (h3 : ‖B - D‖ = 16)
variable (h4 : ‖A - B‖ = 9)
variable (h5 : ‖C - E‖ = 5)
variable (h6 : ‖D - E‖ = 3)

-- State the theorem
theorem AC_length : ‖A - C‖ = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_length_l1075_107512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1075_107532

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Axioms based on the given conditions
axiom has_derivative : ∀ x : ℝ, HasDerivAt f (f' x) x

axiom f_property : ∀ x : ℝ, f x = 4 * x^2 - f (-x)

axiom f'_inequality : ∀ x : ℝ, x < 0 → f' x + 1/2 < 4 * x

axiom f_inequality : ∀ m : ℝ, f (m + 1) ≤ f (-m) + 4 * m + 2

-- Theorem statement
theorem m_range : 
  {m : ℝ | f (m + 1) ≤ f (-m) + 4 * m + 2} = {m : ℝ | m ≥ -1/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1075_107532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l_l1075_107554

/-- Curve C in the xy-plane -/
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 4 * Real.sin θ)

/-- Line l in the xy-plane -/
noncomputable def line_l (α t : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 2 + t * Real.sin α)

/-- The slope of a line given its angle -/
noncomputable def line_slope (α : ℝ) : ℝ := Real.tan α

theorem slope_of_line_l (α : ℝ) :
  (∃ t₁ t₂ : ℝ, 
    curve_C t₁ = line_l α t₁ ∧ 
    curve_C t₂ = line_l α t₂ ∧
    ((line_l α t₁).1 + (line_l α t₂).1) / 2 = 1 ∧
    ((line_l α t₁).2 + (line_l α t₂).2) / 2 = 2) →
  line_slope α = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l_l1075_107554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_l1075_107582

-- Define the two functions as noncomputable
noncomputable def f (x : ℝ) := Real.sin (3 * x)
noncomputable def g (x : ℝ) := Real.sin (3 * x + 1)

-- State the theorem
theorem sin_shift (x : ℝ) : 
  f x = g (x - 1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_l1075_107582
