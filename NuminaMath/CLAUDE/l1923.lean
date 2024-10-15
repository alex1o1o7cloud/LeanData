import Mathlib

namespace NUMINAMATH_CALUDE_distance_on_line_l1923_192382

/-- Given two points (a, b) and (c, d) on the line x + y = px + q,
    prove that the distance between them is |a-c|√(1 + (p-1)²) -/
theorem distance_on_line (p q a b c d : ℝ) :
  (a + b = p * a + q) →
  (c + d = p * c + q) →
  Real.sqrt ((c - a)^2 + (d - b)^2) = |a - c| * Real.sqrt (1 + (p - 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_on_line_l1923_192382


namespace NUMINAMATH_CALUDE_complement_of_A_l1923_192317

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | |x - 1| > 2}

theorem complement_of_A : 
  Set.compl A = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1923_192317


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1923_192305

/-- An isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithInscribedCircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The diameter of the circle is along the base of the triangle -/
  diameter_along_base : Bool

/-- Theorem: The radius of the inscribed circle in the given isosceles triangle is 120/13 -/
theorem inscribed_circle_radius 
  (triangle : IsoscelesTriangleWithInscribedCircle) 
  (h1 : triangle.base = 20) 
  (h2 : triangle.height = 24) 
  (h3 : triangle.diameter_along_base = true) : 
  triangle.radius = 120 / 13 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1923_192305


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l1923_192336

def recipe_flour (total_sugar : ℕ) (added_sugar : ℕ) (flour_sugar_diff : ℕ) : ℕ :=
  (total_sugar - added_sugar) + flour_sugar_diff

theorem recipe_flour_amount : recipe_flour 6 4 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l1923_192336


namespace NUMINAMATH_CALUDE_pentagon_area_condition_l1923_192356

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Calculates the area of a pentagon given its vertices -/
def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Checks if a pentagon has a vertical line of symmetry -/
def hasVerticalSymmetry (p : Pentagon) : Prop := sorry

theorem pentagon_area_condition (y : ℝ) : 
  let p := Pentagon.mk (0, 0) (0, 5) (3, y) (6, 5) (6, 0)
  hasVerticalSymmetry p ∧ pentagonArea p = 50 → y = 35/3 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_condition_l1923_192356


namespace NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l1923_192387

/-- Given vectors a and b in ℝ², if a is perpendicular to b, then the magnitude of b is √5 -/
theorem perpendicular_vectors_magnitude (a b : ℝ × ℝ) : 
  a = (1, -2) → 
  b.1 = 2 → 
  a.1 * b.1 + a.2 * b.2 = 0 → 
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l1923_192387


namespace NUMINAMATH_CALUDE_interest_rate_beyond_five_years_l1923_192392

/-- Calculates the simple interest --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_beyond_five_years 
  (principal : ℝ)
  (rate_first_two_years : ℝ)
  (rate_next_three_years : ℝ)
  (total_interest : ℝ)
  (h1 : principal = 12000)
  (h2 : rate_first_two_years = 0.06)
  (h3 : rate_next_three_years = 0.09)
  (h4 : total_interest = 11400)
  : ∃ (rate_beyond_five_years : ℝ),
    rate_beyond_five_years = 0.14 ∧
    total_interest = 
      simple_interest principal rate_first_two_years 2 +
      simple_interest principal rate_next_three_years 3 +
      simple_interest principal rate_beyond_five_years 4 :=
sorry

end NUMINAMATH_CALUDE_interest_rate_beyond_five_years_l1923_192392


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1923_192327

theorem min_value_quadratic (x y : ℝ) :
  y = 3 * x^2 + 6 * x + 9 →
  ∀ z : ℝ, y ≥ 6 ∧ ∃ w : ℝ, 3 * w^2 + 6 * w + 9 = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1923_192327


namespace NUMINAMATH_CALUDE_vector_addition_l1923_192390

/-- Given two vectors OA and AB in R², prove that OB = OA + AB -/
theorem vector_addition (OA AB : ℝ × ℝ) (h1 : OA = (-2, 3)) (h2 : AB = (-1, -4)) :
  OA + AB = (-3, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l1923_192390


namespace NUMINAMATH_CALUDE_pentagon_perimeter_calculation_l1923_192394

/-- The perimeter of a pentagon with given side lengths -/
def pentagon_perimeter (FG GH HI IJ JF : ℝ) : ℝ := FG + GH + HI + IJ + JF

/-- Theorem: The perimeter of pentagon FGHIJ is 7 + 2√5 -/
theorem pentagon_perimeter_calculation :
  pentagon_perimeter 2 2 (Real.sqrt 5) (Real.sqrt 5) 3 = 7 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_calculation_l1923_192394


namespace NUMINAMATH_CALUDE_range_of_H_l1923_192360

/-- The function H defined as the difference of absolute values -/
def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

/-- Theorem stating the range of function H -/
theorem range_of_H :
  (∀ x : ℝ, H x ≥ -4 ∧ H x ≤ 4) ∧
  (∃ x : ℝ, H x = -4) ∧
  (∃ x : ℝ, H x = 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_H_l1923_192360


namespace NUMINAMATH_CALUDE_equation_solution_l1923_192370

theorem equation_solution :
  ∃ s : ℚ, (s - 60) / 3 = (6 - 3 * s) / 4 ∧ s = 258 / 13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1923_192370


namespace NUMINAMATH_CALUDE_magical_stack_size_l1923_192314

/-- Represents a stack of cards -/
structure CardStack :=
  (n : ℕ)  -- Half the total number of cards

/-- Checks if a card number is in its original position after restacking -/
def retains_position (stack : CardStack) (card : ℕ) : Prop :=
  card ≤ stack.n → card = 2 * card - 1
  ∧ card > stack.n → card = 2 * (card - stack.n)

/-- Defines a magical stack -/
def is_magical (stack : CardStack) : Prop :=
  ∃ (a b : ℕ), a ≤ stack.n ∧ b > stack.n ∧ retains_position stack a ∧ retains_position stack b

/-- Main theorem: A magical stack where card 161 retains its position has 482 cards -/
theorem magical_stack_size :
  ∀ (stack : CardStack),
    is_magical stack →
    retains_position stack 161 →
    2 * stack.n = 482 :=
by sorry

end NUMINAMATH_CALUDE_magical_stack_size_l1923_192314


namespace NUMINAMATH_CALUDE_distance_on_segment_triangle_inequality_l1923_192326

/-- Custom distance function for points in 2D space -/
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₂ - x₁| + |y₂ - y₁|

/-- Theorem: If C is on the line segment AB, then AC + CB = AB -/
theorem distance_on_segment (x₁ y₁ x₂ y₂ x y : ℝ) 
  (h_x : min x₁ x₂ ≤ x ∧ x ≤ max x₁ x₂) 
  (h_y : min y₁ y₂ ≤ y ∧ y ≤ max y₁ y₂) :
  distance x₁ y₁ x y + distance x y x₂ y₂ = distance x₁ y₁ x₂ y₂ := by sorry

/-- Theorem: For any triangle ABC, AC + CB > AB -/
theorem triangle_inequality (x₁ y₁ x₂ y₂ x y : ℝ) :
  distance x₁ y₁ x y + distance x y x₂ y₂ ≥ distance x₁ y₁ x₂ y₂ := by sorry

end NUMINAMATH_CALUDE_distance_on_segment_triangle_inequality_l1923_192326


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1923_192384

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_3 : a 3 = -3)
  (h_4 : a 4 = 6) :
  a 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1923_192384


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l1923_192313

theorem power_fraction_simplification :
  (1 : ℚ) / ((-5^4)^2) * (-5)^9 = -5 := by sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l1923_192313


namespace NUMINAMATH_CALUDE_product_modulo_25_l1923_192302

theorem product_modulo_25 (n : ℕ) : 
  65 * 74 * 89 ≡ n [ZMOD 25] → 0 ≤ n ∧ n < 25 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_modulo_25_l1923_192302


namespace NUMINAMATH_CALUDE_range_of_m_given_quadratic_inequality_l1923_192348

theorem range_of_m_given_quadratic_inequality (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*m*x + m + 2 ≥ 0) ↔ m ∈ Set.Icc (-1) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_given_quadratic_inequality_l1923_192348


namespace NUMINAMATH_CALUDE_remainder_theorem_l1923_192361

theorem remainder_theorem : 
  10002000400080016003200640128025605121024204840968192 % 100020004000800160032 = 40968192 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1923_192361


namespace NUMINAMATH_CALUDE_gum_sharing_theorem_l1923_192347

def john_gum : ℝ := 54.5
def cole_gum : ℝ := 45.75
def aubrey_gum : ℝ := 37
def maria_gum : ℝ := 70.25
def liam_gum : ℝ := 28.5
def emma_gum : ℝ := 32.5

def total_people : ℕ := 6

def total_gum : ℝ := 2 * (john_gum + cole_gum + aubrey_gum + maria_gum + liam_gum + emma_gum)

theorem gum_sharing_theorem : 
  total_gum / total_people = 89.5 := by
  sorry

end NUMINAMATH_CALUDE_gum_sharing_theorem_l1923_192347


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1923_192372

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a)
  (h_a2 : a 2 = Real.sqrt 2)
  (h_a3 : a 3 = Real.rpow 4 (1/3)) :
  (a 1 + a 15) / (a 7 + a 21) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1923_192372


namespace NUMINAMATH_CALUDE_basketball_pricing_solution_l1923_192386

/-- Represents the cost and pricing of basketballs --/
structure BasketballPricing where
  cost_a : ℝ  -- Cost of A brand basketball
  cost_b : ℝ  -- Cost of B brand basketball
  price_a : ℝ  -- Original price of A brand basketball
  markup_b : ℝ  -- Markup percentage for B brand basketball
  discount_a : ℝ  -- Discount percentage for remaining A brand basketballs

/-- Theorem stating the correct pricing and discount for the basketball problem --/
theorem basketball_pricing_solution (p : BasketballPricing) : 
  (40 * p.cost_a + 40 * p.cost_b = 7200) →
  (50 * p.cost_a + 30 * p.cost_b = 7400) →
  (p.price_a = 140) →
  (p.markup_b = 0.3) →
  (40 * (p.price_a - p.cost_a) + 10 * (p.price_a * (1 - p.discount_a / 100) - p.cost_a) + 30 * p.cost_b * p.markup_b = 2440) →
  (p.cost_a = 100 ∧ p.cost_b = 80 ∧ p.discount_a = 20) := by
  sorry

end NUMINAMATH_CALUDE_basketball_pricing_solution_l1923_192386


namespace NUMINAMATH_CALUDE_super_bowl_probability_sum_l1923_192346

theorem super_bowl_probability_sum :
  ∀ (p_play p_not_play : ℝ),
  p_play = 9 * p_not_play →
  p_play ≥ 0 →
  p_not_play ≥ 0 →
  p_play + p_not_play = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_super_bowl_probability_sum_l1923_192346


namespace NUMINAMATH_CALUDE_rebecca_eggs_count_l1923_192377

theorem rebecca_eggs_count (num_groups : ℕ) (eggs_per_group : ℕ) 
  (h1 : num_groups = 11) (h2 : eggs_per_group = 2) : 
  num_groups * eggs_per_group = 22 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_eggs_count_l1923_192377


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_abs_sum_l1923_192329

theorem sqrt_sum_equals_abs_sum (x : ℝ) :
  Real.sqrt (x^2 + 6*x + 9) + Real.sqrt (x^2 - 6*x + 9) = |x - 3| + |x + 3| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_abs_sum_l1923_192329


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l1923_192308

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l1923_192308


namespace NUMINAMATH_CALUDE_sqrt_simplification_l1923_192337

theorem sqrt_simplification :
  Real.sqrt 32 - Real.sqrt 18 + Real.sqrt 8 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l1923_192337


namespace NUMINAMATH_CALUDE_even_expression_l1923_192345

theorem even_expression (x : ℤ) (h : x = 3) : 
  ∃ k : ℤ, 2 * (x^2 + 9) = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_even_expression_l1923_192345


namespace NUMINAMATH_CALUDE_mod_difference_of_powers_l1923_192339

theorem mod_difference_of_powers (n : ℕ) : 45^1537 - 25^1537 ≡ 4 [MOD 8] := by
  sorry

end NUMINAMATH_CALUDE_mod_difference_of_powers_l1923_192339


namespace NUMINAMATH_CALUDE_decision_box_distinguishes_l1923_192343

-- Define the components of control structures
inductive ControlComponent
  | ProcessingBox
  | DecisionBox
  | StartEndBox
  | InputOutputBox

-- Define the types of control structures
structure ControlStructure where
  components : List ControlComponent

-- Define a selection structure
def SelectionStructure : ControlStructure := {
  components := [ControlComponent.ProcessingBox, ControlComponent.DecisionBox, 
                 ControlComponent.StartEndBox, ControlComponent.InputOutputBox]
}

-- Define a sequential structure
def SequentialStructure : ControlStructure := {
  components := [ControlComponent.ProcessingBox, ControlComponent.StartEndBox, 
                 ControlComponent.InputOutputBox]
}

-- Define the distinguishing feature
def isDistinguishingFeature (component : ControlComponent) 
                            (struct1 struct2 : ControlStructure) : Prop :=
  (component ∈ struct1.components) ∧ (component ∉ struct2.components)

-- Theorem stating that the decision box is the distinguishing feature
theorem decision_box_distinguishes :
  isDistinguishingFeature ControlComponent.DecisionBox SelectionStructure SequentialStructure :=
by
  sorry


end NUMINAMATH_CALUDE_decision_box_distinguishes_l1923_192343


namespace NUMINAMATH_CALUDE_circle_radius_increase_l1923_192359

/-- Given a circle with radius r, prove that when the radius is increased by 5 and the area is quadrupled, the original radius was 5 and the new perimeter is 20π. -/
theorem circle_radius_increase (r : ℝ) : 
  (π * (r + 5)^2 = 4 * π * r^2) → 
  (r = 5 ∧ 2 * π * (r + 5) = 20 * π) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_increase_l1923_192359


namespace NUMINAMATH_CALUDE_smallest_representable_numbers_l1923_192367

def is_representable (c : ℕ) : Prop :=
  ∃ m n : ℕ, c = 7 * m^2 - 11 * n^2

theorem smallest_representable_numbers :
  (is_representable 1 ∧ is_representable 5) ∧
  (∀ c : ℕ, c < 1 → ¬is_representable c) ∧
  (∀ c : ℕ, 1 < c → c < 5 → ¬is_representable c) :=
sorry

end NUMINAMATH_CALUDE_smallest_representable_numbers_l1923_192367


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1923_192355

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ 
    (x = -7 ∧ y = -99) ∨ 
    (x = -1 ∧ y = -9) ∨ 
    (x = 1 ∧ y = 5) ∨ 
    (x = 7 ∧ y = -97) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1923_192355


namespace NUMINAMATH_CALUDE_unique_square_friendly_l1923_192315

/-- A number is a perfect square if it's equal to some integer squared. -/
def IsPerfectSquare (n : ℤ) : Prop := ∃ k : ℤ, n = k^2

/-- An integer c is square-friendly if for all integers m, m^2 + 18m + c is a perfect square. -/
def IsSquareFriendly (c : ℤ) : Prop :=
  ∀ m : ℤ, IsPerfectSquare (m^2 + 18*m + c)

/-- Theorem: 81 is the only square-friendly integer. -/
theorem unique_square_friendly : ∃! c : ℤ, IsSquareFriendly c ∧ c = 81 := by
  sorry

end NUMINAMATH_CALUDE_unique_square_friendly_l1923_192315


namespace NUMINAMATH_CALUDE_all_x_greater_than_two_l1923_192389

theorem all_x_greater_than_two : ∀ x ∈ Set.Ioo 0 π, x + 1 / Real.sin x > 2 := by sorry

end NUMINAMATH_CALUDE_all_x_greater_than_two_l1923_192389


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1923_192331

-- Define the hyperbola
def is_hyperbola (a b : ℝ) (x y : ℝ → ℝ) : Prop :=
  ∀ t, x t^2 / a^2 - y t^2 / b^2 = 1

-- Define the focal length
def focal_length (c : ℝ) : Prop := c = 4 * Real.sqrt 5

-- Define the asymptotes
def asymptotes (x y : ℝ → ℝ) : Prop :=
  ∀ t, (2 * x t = y t) ∨ (2 * x t = -y t)

theorem hyperbola_equation (a b : ℝ) (x y : ℝ → ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (h3 : is_hyperbola a b x y)
  (h4 : focal_length (Real.sqrt (a^2 + b^2)))
  (h5 : asymptotes x y) :
  is_hyperbola 2 4 x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1923_192331


namespace NUMINAMATH_CALUDE_smallest_shift_for_scaled_function_l1923_192371

/-- A function with period 15 -/
def isPeriodic15 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 15) = f x

/-- The property we want to prove -/
def hasShiftProperty (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f ((x - b) / 3) = f (x / 3)

theorem smallest_shift_for_scaled_function 
  (f : ℝ → ℝ) (h : isPeriodic15 f) :
  (∃ b > 0, hasShiftProperty f b) ∧ 
  (∀ b > 0, hasShiftProperty f b → b ≥ 45) :=
sorry

end NUMINAMATH_CALUDE_smallest_shift_for_scaled_function_l1923_192371


namespace NUMINAMATH_CALUDE_equation_solution_l1923_192340

theorem equation_solution : ∃ x : ℝ, (x^2 + 3*x + 4) / (x^2 - 3*x + 2) = x + 6 := by
  use 1
  -- Proof goes here
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l1923_192340


namespace NUMINAMATH_CALUDE_min_value_on_line_l1923_192342

/-- The minimum value of ((a+1)^2 + b^2) is 3, given that (a,b) is on y = √3x - √3 -/
theorem min_value_on_line :
  ∀ a b : ℝ, b = Real.sqrt 3 * a - Real.sqrt 3 →
  (∀ x y : ℝ, y = Real.sqrt 3 * x - Real.sqrt 3 → (x + 1)^2 + y^2 ≥ 3) ∧
  ∃ x y : ℝ, y = Real.sqrt 3 * x - Real.sqrt 3 ∧ (x + 1)^2 + y^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_line_l1923_192342


namespace NUMINAMATH_CALUDE_fraction_addition_simplification_l1923_192311

theorem fraction_addition_simplification : (2 : ℚ) / 5 + (3 : ℚ) / 15 = (3 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_simplification_l1923_192311


namespace NUMINAMATH_CALUDE_cone_base_radius_l1923_192351

/-- Given a cone formed from a sector with arc length 8π, its base radius is 4. -/
theorem cone_base_radius (cone : Real) (sector : Real) :
  (sector = 8 * Real.pi) →    -- arc length of sector
  (sector = 2 * Real.pi * cone) →    -- arc length equals circumference of base
  (cone = 4) :=    -- radius of base
by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l1923_192351


namespace NUMINAMATH_CALUDE_mean_median_difference_l1923_192376

def frequency_distribution : List (ℕ × ℕ) := [
  (0, 2), (1, 3), (2, 4), (3, 5), (4, 3), (5, 1)
]

def total_students : ℕ := 18

def median (fd : List (ℕ × ℕ)) (total : ℕ) : ℚ :=
  sorry

def mean (fd : List (ℕ × ℕ)) (total : ℕ) : ℚ :=
  sorry

theorem mean_median_difference :
  let m := mean frequency_distribution total_students
  let med := median frequency_distribution total_students
  |m - med| = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l1923_192376


namespace NUMINAMATH_CALUDE_power_product_equals_negative_eighth_l1923_192373

theorem power_product_equals_negative_eighth (x : ℝ) (n : ℕ) :
  x = -0.125 → (x^(n+1) * 8^n = -0.125) := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_negative_eighth_l1923_192373


namespace NUMINAMATH_CALUDE_nine_more_knives_l1923_192396

/-- Represents the number of each type of cutlery -/
structure Cutlery where
  forks : ℕ
  knives : ℕ
  spoons : ℕ
  teaspoons : ℕ

/-- The initial state of the cutlery drawer -/
def initial : Cutlery :=
  { forks := 6
  , knives := 6 + 9  -- We're proving this 9
  , spoons := 2 * (6 + 9)
  , teaspoons := 6 / 2 }

/-- The state after adding 2 of each cutlery -/
def after_adding (c : Cutlery) : Cutlery :=
  { forks := c.forks + 2
  , knives := c.knives + 2
  , spoons := c.spoons + 2
  , teaspoons := c.teaspoons + 2 }

/-- The total number of cutlery pieces -/
def total (c : Cutlery) : ℕ :=
  c.forks + c.knives + c.spoons + c.teaspoons

/-- Main theorem: There are 9 more knives than forks initially -/
theorem nine_more_knives :
  initial.knives = initial.forks + 9 ∧
  initial.spoons = 2 * initial.knives ∧
  initial.teaspoons = initial.forks / 2 ∧
  total (after_adding initial) = 62 :=
by sorry

end NUMINAMATH_CALUDE_nine_more_knives_l1923_192396


namespace NUMINAMATH_CALUDE_triangle_inequality_bounds_l1923_192374

theorem triangle_inequality_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b ≤ c) (htri : a + b > c) :
  4 ≤ (a + b + c)^2 / (b * c) ∧ (a + b + c)^2 / (b * c) ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_bounds_l1923_192374


namespace NUMINAMATH_CALUDE_gas_price_calculation_l1923_192399

theorem gas_price_calculation (rental_cost mileage_rate total_miles total_expense gas_gallons : ℚ)
  (h1 : rental_cost = 150)
  (h2 : mileage_rate = 1/2)
  (h3 : total_miles = 320)
  (h4 : total_expense = 338)
  (h5 : gas_gallons = 8) :
  (total_expense - rental_cost - mileage_rate * total_miles) / gas_gallons = 7/2 := by
  sorry

#eval (338 : ℚ) - 150 - 1/2 * 320
#eval ((338 : ℚ) - 150 - 1/2 * 320) / 8

end NUMINAMATH_CALUDE_gas_price_calculation_l1923_192399


namespace NUMINAMATH_CALUDE_possible_values_of_2a_plus_b_l1923_192344

theorem possible_values_of_2a_plus_b (a b x y z : ℕ) :
  a^x = b^y ∧ 
  a^x = 1994^z ∧ 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / z →
  2*a + b = 1001 ∨ 2*a + b = 1996 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_2a_plus_b_l1923_192344


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1923_192383

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  startNumber : ℕ
  hTotalPositive : 0 < totalStudents
  hSamplePositive : 0 < sampleSize
  hStartValid : startNumber ≤ totalStudents

/-- Generates the sequence of selected student numbers -/
def generateSequence (s : SystematicSampling) : List ℕ :=
  List.range s.sampleSize |>.map (fun i => s.startNumber + i * (s.totalStudents / s.sampleSize))

/-- Theorem stating that the systematic sampling generates the expected sequence -/
theorem systematic_sampling_theorem (s : SystematicSampling)
  (h1 : s.totalStudents = 50)
  (h2 : s.sampleSize = 5)
  (h3 : s.startNumber = 3) :
  generateSequence s = [3, 13, 23, 33, 43] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1923_192383


namespace NUMINAMATH_CALUDE_symmetric_line_correct_l1923_192332

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-axis -/
def yAxis : Line := { a := 1, b := 0, c := 0 }

/-- Check if a line is symmetric to another line with respect to the y-axis -/
def isSymmetricToYAxis (l1 l2 : Line) : Prop :=
  l1.a = -l2.a ∧ l1.b = l2.b ∧ l1.c = l2.c

/-- The original line x - y + 1 = 0 -/
def originalLine : Line := { a := 1, b := -1, c := 1 }

/-- The symmetric line we want to prove -/
def symmetricLine : Line := { a := 1, b := 1, c := -1 }

theorem symmetric_line_correct : 
  isSymmetricToYAxis originalLine symmetricLine :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_correct_l1923_192332


namespace NUMINAMATH_CALUDE_sum_area_ABC_DEF_l1923_192353

-- Define the points and lengths
variable (A B C D E F G : ℝ × ℝ)
variable (AB BG GE DE : ℝ)

-- Define the areas of triangles
def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define the conditions
axiom AB_length : AB = 2
axiom BG_length : BG = 3
axiom GE_length : GE = 4
axiom DE_length : DE = 5

axiom sum_area_BCG_EFG : area_triangle B C G + area_triangle E F G = 24
axiom sum_area_AGF_CDG : area_triangle A G F + area_triangle C D G = 51

-- State the theorem
theorem sum_area_ABC_DEF :
  area_triangle A B C + area_triangle D E F = 23 :=
sorry

end NUMINAMATH_CALUDE_sum_area_ABC_DEF_l1923_192353


namespace NUMINAMATH_CALUDE_tip_calculation_correct_l1923_192304

/-- Calculates the tip amount for a family's salon visit -/
def calculate_tip (womens_haircut_price : ℚ) 
                  (childrens_haircut_price : ℚ) 
                  (teens_haircut_price : ℚ) 
                  (num_women : ℕ) 
                  (num_children : ℕ) 
                  (num_teens : ℕ) 
                  (hair_treatment_price : ℚ)
                  (tip_percentage : ℚ) : ℚ :=
  let total_cost := womens_haircut_price * num_women +
                    childrens_haircut_price * num_children +
                    teens_haircut_price * num_teens +
                    hair_treatment_price
  tip_percentage * total_cost

theorem tip_calculation_correct :
  calculate_tip 40 30 35 1 2 1 20 (1/4) = 155/4 :=
by sorry

end NUMINAMATH_CALUDE_tip_calculation_correct_l1923_192304


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l1923_192338

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (k * a.1 = b.1 ∧ k * a.2 = b.2)

/-- Theorem: If (1, 2) is parallel to (x, -4), then x = -2 -/
theorem parallel_vectors_imply_x_value :
  ∀ x : ℝ, parallel (1, 2) (x, -4) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l1923_192338


namespace NUMINAMATH_CALUDE_boxes_with_no_items_l1923_192395

/-- Given the following conditions:
  - There are 15 boxes in total
  - 8 boxes contain pencils
  - 5 boxes contain pens
  - 3 boxes contain markers
  - 4 boxes contain both pens and pencils
  - 1 box contains all three items (pencils, pens, and markers)
  Prove that the number of boxes containing neither pens, pencils, nor markers is 5. -/
theorem boxes_with_no_items (total : ℕ) (pencil : ℕ) (pen : ℕ) (marker : ℕ) 
  (pen_and_pencil : ℕ) (all_three : ℕ) :
  total = 15 →
  pencil = 8 →
  pen = 5 →
  marker = 3 →
  pen_and_pencil = 4 →
  all_three = 1 →
  total - (pen_and_pencil + (pencil - pen_and_pencil) + 
    (pen - pen_and_pencil) + (marker - all_three)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_boxes_with_no_items_l1923_192395


namespace NUMINAMATH_CALUDE_system_solution_l1923_192368

theorem system_solution (x y z : ℤ) : 
  (x^2 = y*z + 1 ∧ y^2 = z*x + 1 ∧ z^2 = x*y + 1) ↔ 
  ((x = 1 ∧ y = 0 ∧ z = -1) ∨
   (x = 1 ∧ y = -1 ∧ z = 0) ∨
   (x = 0 ∧ y = 1 ∧ z = -1) ∨
   (x = 0 ∧ y = -1 ∧ z = 1) ∨
   (x = -1 ∧ y = 1 ∧ z = 0) ∨
   (x = -1 ∧ y = 0 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1923_192368


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l1923_192378

/-- The probability of selecting 2 red balls from a bag containing 6 red, 4 blue, and 2 green balls -/
theorem probability_two_red_balls (red : ℕ) (blue : ℕ) (green : ℕ) 
  (h_red : red = 6) (h_blue : blue = 4) (h_green : green = 2) : 
  (Nat.choose red 2 : ℚ) / (Nat.choose (red + blue + green) 2) = 5 / 22 :=
sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l1923_192378


namespace NUMINAMATH_CALUDE_equation_solution_l1923_192330

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 5 ∧ x₂ = (4 * Real.sqrt 3) / 3 ∧ 
  (∀ x : ℝ, Real.sqrt 3 * x * (x - 5) + 4 * (5 - x) = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1923_192330


namespace NUMINAMATH_CALUDE_smallest_number_l1923_192352

-- Define a function to convert a number from base b to decimal
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

-- Define the given numbers
def num1 : Nat := to_decimal [8, 5] 9
def num2 : Nat := to_decimal [2, 1, 0] 6
def num3 : Nat := to_decimal [1, 0, 0, 0] 4
def num4 : Nat := to_decimal [1, 1, 1, 1, 1, 1] 2

-- Theorem statement
theorem smallest_number : num4 < num1 ∧ num4 < num2 ∧ num4 < num3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1923_192352


namespace NUMINAMATH_CALUDE_circle_whisper_game_l1923_192358

theorem circle_whisper_game (a b c d e f : ℕ) : 
  a + b + c + d + e + f = 18 →
  a + b = 16 →
  b + c = 12 →
  e + f = 8 →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_circle_whisper_game_l1923_192358


namespace NUMINAMATH_CALUDE_trig_identity_l1923_192362

theorem trig_identity (a : ℝ) (h : Real.sin (π * Real.cos a) = Real.cos (π * Real.sin a)) :
  35 * (Real.sin (2 * a))^2 + 84 * (Real.cos (4 * a))^2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1923_192362


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_l1923_192300

theorem simplify_sqrt_product : 
  Real.sqrt (3 * 5) * Real.sqrt (5^4 * 3^3) = 45 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_l1923_192300


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l1923_192354

/-- Proves that given two people moving in opposite directions for 4 hours,
    with one person moving at 3 km/hr and the distance between them after 4 hours being 20 km,
    the speed of the other person is 2 km/hr. -/
theorem opposite_direction_speed
  (time : ℝ)
  (pooja_speed : ℝ)
  (distance : ℝ)
  (h1 : time = 4)
  (h2 : pooja_speed = 3)
  (h3 : distance = 20) :
  ∃ (other_speed : ℝ), other_speed = 2 ∧ distance = (other_speed + pooja_speed) * time :=
sorry

end NUMINAMATH_CALUDE_opposite_direction_speed_l1923_192354


namespace NUMINAMATH_CALUDE_digits_of_two_power_fifteen_times_five_power_ten_l1923_192369

theorem digits_of_two_power_fifteen_times_five_power_ten : 
  (Nat.digits 10 (2^15 * 5^10)).length = 12 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_two_power_fifteen_times_five_power_ten_l1923_192369


namespace NUMINAMATH_CALUDE_complex_product_equals_112_l1923_192318

theorem complex_product_equals_112 (y : ℂ) (h : y = Complex.exp (2 * Real.pi * Complex.I / 9)) :
  (3 * y + y^3) * (3 * y^3 + y^9) * (3 * y^6 + y^18) * 
  (3 * y^2 + y^6) * (3 * y^5 + y^15) * (3 * y^7 + y^21) = 112 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_112_l1923_192318


namespace NUMINAMATH_CALUDE_min_value_theorem_l1923_192333

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 2) :
  2/x + 1/y ≥ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 2 ∧ 2/x₀ + 1/y₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1923_192333


namespace NUMINAMATH_CALUDE_lisa_marbles_problem_l1923_192388

def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let required_marbles := (num_friends * (num_friends + 1)) / 2
  if required_marbles > initial_marbles then
    required_marbles - initial_marbles
  else
    0

theorem lisa_marbles_problem (num_friends : ℕ) (initial_marbles : ℕ) 
  (h1 : num_friends = 12) (h2 : initial_marbles = 50) :
  min_additional_marbles num_friends initial_marbles = 28 := by
  sorry

end NUMINAMATH_CALUDE_lisa_marbles_problem_l1923_192388


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l1923_192349

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 40 → 
  a * b + b * c + c * d + d * a ≤ 800 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l1923_192349


namespace NUMINAMATH_CALUDE_union_and_subset_l1923_192366

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x < 1 + 3*m}

-- Define the complement of A
def A_complement : Set ℝ := {x | x ≤ -1 ∨ x > 3}

theorem union_and_subset :
  (∀ m : ℝ, m = 1 → A ∪ B m = {x | -1 < x ∧ x < 4}) ∧
  (∀ m : ℝ, B m ⊆ A_complement ↔ m ≤ -1/2 ∨ m > 3) :=
sorry

end NUMINAMATH_CALUDE_union_and_subset_l1923_192366


namespace NUMINAMATH_CALUDE_sam_fish_count_l1923_192306

/-- Represents the number of fish a person has -/
structure FishCount where
  goldfish : ℕ
  guppies : ℕ
  angelfish : ℕ

def Lilly : FishCount :=
  { goldfish := 10, guppies := 15, angelfish := 0 }

def Rosy : FishCount :=
  { goldfish := 12, guppies := 8, angelfish := 5 }

def Sam : FishCount :=
  { goldfish := Rosy.goldfish - 3, guppies := 2 * Lilly.guppies, angelfish := 0 }

def guppiesTransferred : ℕ := Lilly.guppies / 2

def LillyAfterTransfer : FishCount :=
  { Lilly with guppies := Lilly.guppies - guppiesTransferred }

def SamAfterTransfer : FishCount :=
  { Sam with guppies := Sam.guppies + guppiesTransferred }

def totalFish (fc : FishCount) : ℕ :=
  fc.goldfish + fc.guppies + fc.angelfish

theorem sam_fish_count :
  totalFish SamAfterTransfer = 46 := by sorry

end NUMINAMATH_CALUDE_sam_fish_count_l1923_192306


namespace NUMINAMATH_CALUDE_percentage_of_a_to_b_l1923_192310

theorem percentage_of_a_to_b (A B C D : ℝ) 
  (h1 : A = 0.125 * C)
  (h2 : B = 0.375 * D)
  (h3 : D = 1.225 * C)
  (h4 : C = 0.805 * B) :
  A = 0.100625 * B := by
sorry

end NUMINAMATH_CALUDE_percentage_of_a_to_b_l1923_192310


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l1923_192309

theorem imaginary_unit_sum (i : ℂ) (hi : i^2 = -1) : i^11 + i^111 + i^222 = -2*i - 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l1923_192309


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_integers_l1923_192365

theorem smallest_of_five_consecutive_even_integers : 
  ∃ (n : ℕ), 
    (5 * n + 20 = 30 * 31) ∧ 
    (∀ m : ℕ, m < n → ¬(5 * m + 20 = 30 * 31)) ∧
    (n % 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_integers_l1923_192365


namespace NUMINAMATH_CALUDE_expression_simplification_l1923_192312

theorem expression_simplification (x : ℝ) : 
  3 * x - 5 * (2 + x) + 6 * (2 - x) - 7 * (2 + 3 * x) = -29 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1923_192312


namespace NUMINAMATH_CALUDE_special_function_properties_l1923_192379

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) - f y = (x + 2*y + 2) * x

theorem special_function_properties
  (f : ℝ → ℝ) (h : special_function f) (h2 : f 2 = 12) :
  (f 0 = 4) ∧
  (∀ a : ℝ, (∃ x₀ : ℝ, 1 < x₀ ∧ x₀ < 4 ∧ f x₀ - 8 = a * x₀) ↔ -1 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l1923_192379


namespace NUMINAMATH_CALUDE_hcf_problem_l1923_192322

theorem hcf_problem (a b : ℕ) (h1 : a = 345) (h2 : b < a) 
  (h3 : Nat.lcm a b = Nat.gcd a b * 14 * 15) : Nat.gcd a b = 5 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l1923_192322


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1923_192363

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, (a + 1) * x - y + 2 = 0 ↔ x + (a - 1) * y - 1 = 0) → 
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l1923_192363


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_a_eq_two_root_implies_abs_z_eq_two_sqrt_two_l1923_192341

/-- Given a complex number z = (a^2 - 5a + 6) + (a - 3)i where a ∈ ℝ -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 5*a + 6) (a - 3)

/-- Part 1: If z is purely imaginary, then a = 2 -/
theorem purely_imaginary_implies_a_eq_two (a : ℝ) :
  z a = Complex.I * Complex.im (z a) → a = 2 := by sorry

/-- Part 2: If z is a root of the equation x^2 - 4x + 8 = 0, then |z| = 2√2 -/
theorem root_implies_abs_z_eq_two_sqrt_two (a : ℝ) :
  (z a)^2 - 4*(z a) + 8 = 0 → Complex.abs (z a) = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_a_eq_two_root_implies_abs_z_eq_two_sqrt_two_l1923_192341


namespace NUMINAMATH_CALUDE_common_factor_proof_l1923_192334

def p (x : ℝ) := x^2 - 2*x - 3
def q (x : ℝ) := x^2 - 6*x + 9
def common_factor (x : ℝ) := x - 3

theorem common_factor_proof :
  ∀ x : ℝ, (∃ k₁ k₂ : ℝ, p x = common_factor x * k₁ ∧ q x = common_factor x * k₂) :=
sorry

end NUMINAMATH_CALUDE_common_factor_proof_l1923_192334


namespace NUMINAMATH_CALUDE_tree_height_difference_l1923_192357

theorem tree_height_difference : 
  let pine_height : ℚ := 49/4
  let birch_height : ℚ := 37/2
  birch_height - pine_height = 25/4 := by sorry

#check tree_height_difference

end NUMINAMATH_CALUDE_tree_height_difference_l1923_192357


namespace NUMINAMATH_CALUDE_max_distance_complex_l1923_192316

theorem max_distance_complex (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (w : ℂ), Complex.abs (w + 2 - 2*I) = 1 ∧
             ∀ (v : ℂ), Complex.abs (v + 2 - 2*I) = 1 →
                        Complex.abs (v - 2 - 2*I) ≤ Complex.abs (w - 2 - 2*I) ∧
             Complex.abs (w - 2 - 2*I) = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_l1923_192316


namespace NUMINAMATH_CALUDE_edric_monthly_salary_l1923_192321

/-- Calculates the monthly salary given working hours per day, days per week, hourly rate, and weeks per month. -/
def monthly_salary (hours_per_day : ℝ) (days_per_week : ℝ) (hourly_rate : ℝ) (weeks_per_month : ℝ) : ℝ :=
  hours_per_day * days_per_week * hourly_rate * weeks_per_month

/-- Proves that Edric's monthly salary is approximately $623.52 given the specified working conditions. -/
theorem edric_monthly_salary :
  let hours_per_day : ℝ := 8
  let days_per_week : ℝ := 6
  let hourly_rate : ℝ := 3
  let weeks_per_month : ℝ := 52 / 12
  ∃ ε > 0, |monthly_salary hours_per_day days_per_week hourly_rate weeks_per_month - 623.52| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_edric_monthly_salary_l1923_192321


namespace NUMINAMATH_CALUDE_af_equals_kc_l1923_192364

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points
variable (O : ℝ × ℝ)  -- Center of the circle
variable (G H E D B A C F K : ℝ × ℝ)

-- Define the circle
variable (circle : Circle)

-- Define conditions
variable (gh_diameter : (G.1 - H.1)^2 + (G.2 - H.2)^2 = 4 * circle.radius^2)
variable (ed_diameter : (E.1 - D.1)^2 + (E.2 - D.2)^2 = 4 * circle.radius^2)
variable (perpendicular_diameters : (G.1 - H.1) * (E.1 - D.1) + (G.2 - H.2) * (E.2 - D.2) = 0)
variable (b_outside : (B.1 - circle.center.1)^2 + (B.2 - circle.center.2)^2 > circle.radius^2)
variable (a_on_circle : (A.1 - circle.center.1)^2 + (A.2 - circle.center.2)^2 = circle.radius^2)
variable (c_on_circle : (C.1 - circle.center.1)^2 + (C.2 - circle.center.2)^2 = circle.radius^2)
variable (a_on_gh : A.2 = G.2 ∧ A.2 = H.2)
variable (c_on_gh : C.2 = G.2 ∧ C.2 = H.2)
variable (f_on_gh : F.2 = G.2 ∧ F.2 = H.2)
variable (k_on_gh : K.2 = G.2 ∧ K.2 = H.2)
variable (ba_tangent : (B.1 - A.1) * (A.1 - circle.center.1) + (B.2 - A.2) * (A.2 - circle.center.2) = 0)
variable (bc_tangent : (B.1 - C.1) * (C.1 - circle.center.1) + (B.2 - C.2) * (C.2 - circle.center.2) = 0)
variable (be_intersects_gh_at_f : (B.1 - E.1) * (F.2 - B.2) = (F.1 - B.1) * (B.2 - E.2))
variable (bd_intersects_gh_at_k : (B.1 - D.1) * (K.2 - B.2) = (K.1 - B.1) * (B.2 - D.2))

-- Theorem statement
theorem af_equals_kc : (A.1 - F.1)^2 + (A.2 - F.2)^2 = (K.1 - C.1)^2 + (K.2 - C.2)^2 := by sorry

end NUMINAMATH_CALUDE_af_equals_kc_l1923_192364


namespace NUMINAMATH_CALUDE_shape_has_four_sides_l1923_192320

/-- The shape being fenced -/
structure Shape where
  sides : ℕ
  cost_per_side : ℕ
  total_cost : ℕ

/-- The shape satisfies the given conditions -/
def satisfies_conditions (s : Shape) : Prop :=
  s.cost_per_side = 69 ∧ s.total_cost = 276 ∧ s.total_cost = s.cost_per_side * s.sides

theorem shape_has_four_sides (s : Shape) (h : satisfies_conditions s) : s.sides = 4 := by
  sorry

end NUMINAMATH_CALUDE_shape_has_four_sides_l1923_192320


namespace NUMINAMATH_CALUDE_power_product_equals_eight_l1923_192397

theorem power_product_equals_eight (m n : ℤ) (h : 2 * m + n - 3 = 0) :
  (4 : ℝ) ^ m * (2 : ℝ) ^ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_eight_l1923_192397


namespace NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l1923_192350

theorem proposition_false_iff_a_in_range (a : ℝ) : 
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + (1/2 : ℝ) ≤ 0) ↔ -1 < a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l1923_192350


namespace NUMINAMATH_CALUDE_victors_hourly_rate_l1923_192301

theorem victors_hourly_rate (hours_worked : ℕ) (total_earned : ℕ) 
  (h1 : hours_worked = 10) 
  (h2 : total_earned = 60) : 
  total_earned / hours_worked = 6 := by
  sorry

end NUMINAMATH_CALUDE_victors_hourly_rate_l1923_192301


namespace NUMINAMATH_CALUDE_constant_molecular_weight_l1923_192381

/-- The molecular weight of an acid in g/mol -/
def molecular_weight : ℝ := 792

/-- The number of moles of the acid -/
def num_moles : ℝ := 9

/-- Theorem stating that the molecular weight remains constant regardless of the number of moles -/
theorem constant_molecular_weight : 
  ∀ n : ℝ, n > 0 → molecular_weight = molecular_weight := by sorry

end NUMINAMATH_CALUDE_constant_molecular_weight_l1923_192381


namespace NUMINAMATH_CALUDE_speed_ratio_theorem_l1923_192303

/-- Given two objects A and B with speeds v₁ and v₂ respectively, 
    if they meet after a hours when moving towards each other
    and A overtakes B after b hours when moving in the same direction,
    then the ratio of their speeds v₁/v₂ = (a + b) / (b - a). -/
theorem speed_ratio_theorem (v₁ v₂ a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : v₁ > v₂) :
  (∃ S : ℝ, S > 0 ∧ S = a * (v₁ + v₂) ∧ S = b * (v₁ - v₂)) →
  v₁ / v₂ = (a + b) / (b - a) :=
by sorry

end NUMINAMATH_CALUDE_speed_ratio_theorem_l1923_192303


namespace NUMINAMATH_CALUDE_sum_of_even_integers_l1923_192380

theorem sum_of_even_integers (first last : ℕ) (n : ℕ) (sum : ℕ) : 
  first = 202 →
  last = 300 →
  n = 50 →
  sum = 12550 →
  (last - first) / 2 + 1 = n →
  sum = n / 2 * (first + last) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_l1923_192380


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1923_192324

theorem necessary_not_sufficient_condition :
  (∀ x : ℝ, x^2 - x < 0 → -1 < x ∧ x < 1) ∧
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ ¬(x^2 - x < 0)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1923_192324


namespace NUMINAMATH_CALUDE_eggs_equal_rice_cost_l1923_192323

/-- The cost of a pound of rice in dollars -/
def rice_cost : ℝ := 0.36

/-- The cost of an egg in dollars -/
def egg_cost : ℝ := rice_cost

/-- The cost of half a liter of kerosene in dollars -/
def kerosene_cost : ℝ := 8 * egg_cost

/-- The number of eggs that cost the same as a pound of rice -/
def eggs_per_rice : ℕ := 1

theorem eggs_equal_rice_cost : eggs_per_rice = 1 := by
  sorry

end NUMINAMATH_CALUDE_eggs_equal_rice_cost_l1923_192323


namespace NUMINAMATH_CALUDE_janes_tulip_bulbs_l1923_192391

theorem janes_tulip_bulbs :
  ∀ (T : ℕ),
    (T + T / 2 + 30 + 90 = 150) →
    T = 20 := by
  sorry

end NUMINAMATH_CALUDE_janes_tulip_bulbs_l1923_192391


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l1923_192319

theorem solve_cubic_equation (x : ℝ) : 
  (x^3 * 6^3) / 432 = 864 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l1923_192319


namespace NUMINAMATH_CALUDE_race_time_proof_l1923_192375

/-- A runner completes a race -/
structure Runner where
  distance : ℝ  -- distance covered
  time : ℝ      -- time taken

/-- A race between two runners -/
structure Race where
  length : ℝ           -- race length
  runner_a : Runner    -- runner A
  runner_b : Runner    -- runner B

/-- Given a race satisfying the problem conditions, prove that runner A's time is 7 seconds -/
theorem race_time_proof (race : Race) 
  (h1 : race.length = 200)
  (h2 : race.runner_a.distance - race.runner_b.distance = 35)
  (h3 : race.runner_a.distance = race.length)
  (h4 : race.runner_a.time = 7) :
  race.runner_a.time = 7 := by sorry

end NUMINAMATH_CALUDE_race_time_proof_l1923_192375


namespace NUMINAMATH_CALUDE_subtraction_problem_l1923_192385

theorem subtraction_problem : 3.609 - 2.5 - 0.193 = 0.916 := by sorry

end NUMINAMATH_CALUDE_subtraction_problem_l1923_192385


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l1923_192307

theorem mean_proportional_problem (x : ℝ) :
  (56.5 : ℝ) = Real.sqrt (x * 64) → x = 3192.25 / 64 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l1923_192307


namespace NUMINAMATH_CALUDE_complex_modulus_product_l1923_192398

theorem complex_modulus_product : Complex.abs (5 - 3*I) * Complex.abs (5 + 3*I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l1923_192398


namespace NUMINAMATH_CALUDE_rachel_picked_four_apples_l1923_192325

/-- The number of apples Rachel picked from her tree -/
def apples_picked (initial_apples remaining_apples : ℕ) : ℕ :=
  initial_apples - remaining_apples

/-- Theorem: Rachel picked 4 apples -/
theorem rachel_picked_four_apples :
  apples_picked 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_rachel_picked_four_apples_l1923_192325


namespace NUMINAMATH_CALUDE_harrys_travel_time_l1923_192393

/-- Harry's travel time calculation -/
theorem harrys_travel_time :
  let bus_time_so_far : ℕ := 15
  let remaining_bus_time : ℕ := 25
  let total_bus_time : ℕ := bus_time_so_far + remaining_bus_time
  let walking_time : ℕ := total_bus_time / 2
  total_bus_time + walking_time = 60 := by sorry

end NUMINAMATH_CALUDE_harrys_travel_time_l1923_192393


namespace NUMINAMATH_CALUDE_point_in_quadrants_I_and_II_l1923_192335

-- Define the quadrants
def QuadrantI (x y : ℝ) : Prop := x > 0 ∧ y > 0
def QuadrantII (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Define the inequalities
def Inequality1 (x y : ℝ) : Prop := y > -3 * x
def Inequality2 (x y : ℝ) : Prop := y > x + 2

-- Theorem statement
theorem point_in_quadrants_I_and_II (x y : ℝ) :
  Inequality1 x y ∧ Inequality2 x y → QuadrantI x y ∨ QuadrantII x y :=
by sorry

end NUMINAMATH_CALUDE_point_in_quadrants_I_and_II_l1923_192335


namespace NUMINAMATH_CALUDE_impossible_all_positive_4x4_impossible_all_positive_8x8_l1923_192328

/-- Represents a grid of signs -/
def Grid (n : Nat) := Fin n → Fin n → Bool

/-- Represents a line (row, column, or diagonal) in the grid -/
inductive Line (n : Nat)
| Row : Fin n → Line n
| Col : Fin n → Line n
| Diag : Bool → Line n

/-- Flips the signs along a given line in the grid -/
def flipLine (n : Nat) (g : Grid n) (l : Line n) : Grid n :=
  sorry

/-- Checks if all signs in the grid are positive -/
def allPositive (n : Nat) (g : Grid n) : Prop :=
  ∀ i j, g i j = true

/-- Initial configuration for the 8x8 grid with one negative sign -/
def initialConfig : Grid 8 :=
  sorry

/-- Theorem for the 4x4 grid -/
theorem impossible_all_positive_4x4 (g : Grid 4) :
  ¬∃ (flips : List (Line 4)), allPositive 4 (flips.foldl (flipLine 4) g) :=
  sorry

/-- Theorem for the 8x8 grid -/
theorem impossible_all_positive_8x8 :
  ¬∃ (flips : List (Line 8)), allPositive 8 (flips.foldl (flipLine 8) initialConfig) :=
  sorry

end NUMINAMATH_CALUDE_impossible_all_positive_4x4_impossible_all_positive_8x8_l1923_192328
