import Mathlib

namespace solution_set_of_f_x_plus_one_gt_zero_l2141_214112

def symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + (1 - x)) = f x

def monotone_decreasing_from_one (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x < y → f y < f x

theorem solution_set_of_f_x_plus_one_gt_zero
  (f : ℝ → ℝ)
  (h_sym : symmetric_about_one f)
  (h_mono : monotone_decreasing_from_one f)
  (h_f_zero : f 0 = 0) :
  {x : ℝ | f (x + 1) > 0} = Set.Ioo (-1) 1 :=
sorry

end solution_set_of_f_x_plus_one_gt_zero_l2141_214112


namespace sum_remainder_mod_11_l2141_214178

theorem sum_remainder_mod_11 : 
  (101234 + 101235 + 101236 + 101237 + 101238 + 101239 + 101240) % 11 = 9 := by
  sorry

end sum_remainder_mod_11_l2141_214178


namespace solution_value_l2141_214182

theorem solution_value (a b : ℝ) : 
  (2 : ℝ) * a + (-1 : ℝ) * b = -1 → 2 * a - b + 2017 = 2016 := by
  sorry

end solution_value_l2141_214182


namespace perfect_linear_relationship_l2141_214120

-- Define a scatter plot as a list of points
def ScatterPlot := List (ℝ × ℝ)

-- Define a function to check if all points lie on a straight line
def allPointsOnLine (plot : ScatterPlot) : Prop := sorry

-- Define residuals
def residuals (plot : ScatterPlot) : List ℝ := sorry

-- Define sum of squares of residuals
def sumSquaresResiduals (plot : ScatterPlot) : ℝ := sorry

-- Define correlation coefficient
def correlationCoefficient (plot : ScatterPlot) : ℝ := sorry

-- Theorem statement
theorem perfect_linear_relationship (plot : ScatterPlot) :
  allPointsOnLine plot →
  (∀ r ∈ residuals plot, r = 0) ∧
  sumSquaresResiduals plot = 0 ∧
  |correlationCoefficient plot| = 1 := by sorry

end perfect_linear_relationship_l2141_214120


namespace vet_count_l2141_214172

theorem vet_count (total : ℕ) 
  (puppy_kibble : ℕ → ℕ) (yummy_kibble : ℕ → ℕ)
  (h1 : puppy_kibble total = (20 * total) / 100)
  (h2 : yummy_kibble total = (30 * total) / 100)
  (h3 : yummy_kibble total - puppy_kibble total = 100) :
  total = 1000 := by
sorry

end vet_count_l2141_214172


namespace expression_simplification_l2141_214142

theorem expression_simplification (x : ℝ) (h : x = (1/2)⁻¹ + (-3)^0) :
  ((x^2 - 1) / (x^2 - 2*x + 1) - 1 / (x - 1)) / (3 / (x - 1)) = 1 := by
  sorry

end expression_simplification_l2141_214142


namespace ellipse_with_same_foci_and_eccentricity_l2141_214103

/-- The standard equation of an ellipse with the same foci as another ellipse and a given eccentricity -/
theorem ellipse_with_same_foci_and_eccentricity 
  (a₁ b₁ : ℝ) 
  (h₁ : 0 < a₁ ∧ 0 < b₁) 
  (h₂ : a₁ > b₁) 
  (e : ℝ) 
  (he : e = Real.sqrt 5 / 5) :
  let c₁ := Real.sqrt (a₁^2 - b₁^2)
  let a := 5
  let b := Real.sqrt 20
  ∀ x y : ℝ, 
    (x^2 / a₁^2 + y^2 / b₁^2 = 1) → 
    (x^2 / a^2 + y^2 / b^2 = 1 ∧ 
     c₁ = Real.sqrt 5 ∧ 
     e = c₁ / a) := by
  sorry

end ellipse_with_same_foci_and_eccentricity_l2141_214103


namespace blue_marble_probability_l2141_214197

/-- Given odds for an event, calculates the probability of the event not occurring -/
def probability_of_not_occurring (favorable : ℕ) (unfavorable : ℕ) : ℚ :=
  unfavorable / (favorable + unfavorable)

/-- Theorem: If the odds for drawing a blue marble are 5:9, 
    the probability of not drawing a blue marble is 9/14 -/
theorem blue_marble_probability :
  probability_of_not_occurring 5 9 = 9 / 14 := by
sorry

end blue_marble_probability_l2141_214197


namespace ellipse_foci_distance_l2141_214129

/-- The distance between the foci of an ellipse with equation x^2 + 9y^2 = 576 is 32√2 -/
theorem ellipse_foci_distance : 
  let a : ℝ := Real.sqrt (576 / 1)
  let b : ℝ := Real.sqrt (576 / 9)
  2 * Real.sqrt (a^2 - b^2) = 32 * Real.sqrt 2 :=
by sorry

end ellipse_foci_distance_l2141_214129


namespace arithmetic_progression_squares_l2141_214154

theorem arithmetic_progression_squares (x : ℝ) : 
  ((x^2 - 2*x - 1)^2 + (x^2 + 2*x - 1)^2) / 2 = (x^2 + 1)^2 := by
  sorry

end arithmetic_progression_squares_l2141_214154


namespace largest_710_double_correct_l2141_214133

/-- Converts a base-10 number to its base-7 representation as a list of digits --/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Interprets a list of digits as a base-10 number --/
def fromDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if a number is a 7-10 double --/
def is710Double (n : ℕ) : Prop :=
  fromDigits (toBase7 n) = 2 * n

/-- The largest 7-10 double --/
def largest710Double : ℕ := 315

theorem largest_710_double_correct :
  is710Double largest710Double ∧
  ∀ n : ℕ, n > largest710Double → ¬is710Double n :=
sorry

end largest_710_double_correct_l2141_214133


namespace sum_of_integer_solutions_is_zero_l2141_214163

theorem sum_of_integer_solutions_is_zero : 
  ∃ (S : Finset Int), 
    (∀ x ∈ S, x^4 - 49*x^2 + 576 = 0) ∧ 
    (∀ x : Int, x^4 - 49*x^2 + 576 = 0 → x ∈ S) ∧ 
    (S.sum id = 0) := by
  sorry

end sum_of_integer_solutions_is_zero_l2141_214163


namespace quadratic_common_point_l2141_214181

theorem quadratic_common_point (a b c : ℝ) : 
  let f₁ := fun x => a * x^2 - b * x + c
  let f₂ := fun x => b * x^2 - c * x + a
  let f₃ := fun x => c * x^2 - a * x + b
  f₁ (-1) = f₂ (-1) ∧ f₂ (-1) = f₃ (-1) ∧ f₃ (-1) = a + b + c := by
sorry

end quadratic_common_point_l2141_214181


namespace number_of_boxes_l2141_214125

def total_oranges : ℕ := 45
def oranges_per_box : ℕ := 5

theorem number_of_boxes : total_oranges / oranges_per_box = 9 := by
  sorry

end number_of_boxes_l2141_214125


namespace absolute_value_inequality_l2141_214191

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x + a| ≥ 3) ↔ (a ≤ -5 ∨ a ≥ 1) := by
  sorry

end absolute_value_inequality_l2141_214191


namespace monica_savings_l2141_214105

def savings_pattern (week : ℕ) : ℕ :=
  let cycle := week % 20
  if cycle < 6 then 15 + 5 * cycle
  else if cycle < 12 then 40 - 5 * (cycle - 6)
  else if cycle < 18 then 15 + 5 * (cycle - 12)
  else 40 - 5 * (cycle - 18)

def total_savings : ℕ := (List.range 100).map savings_pattern |> List.sum

theorem monica_savings :
  total_savings = 1450 := by sorry

end monica_savings_l2141_214105


namespace product_of_roots_implies_k_l2141_214130

-- Define the polynomial P(X)
def P (k X : ℝ) : ℝ := X^4 - 18*X^3 + k*X^2 + 200*X - 1984

-- Define the theorem
theorem product_of_roots_implies_k (k : ℝ) :
  (∃ a b c d : ℝ, 
    P k a = 0 ∧ P k b = 0 ∧ P k c = 0 ∧ P k d = 0 ∧
    ((a * b = -32) ∨ (a * c = -32) ∨ (a * d = -32) ∨ 
     (b * c = -32) ∨ (b * d = -32) ∨ (c * d = -32))) →
  k = 86 := by
  sorry

end product_of_roots_implies_k_l2141_214130


namespace three_digit_number_difference_l2141_214158

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_range : hundreds ≥ 1 ∧ hundreds ≤ 9
  tens_range : tens ≥ 0 ∧ tens ≤ 9
  ones_range : ones ≥ 0 ∧ ones ≤ 9

/-- Calculates the value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Calculates the product of digits of a three-digit number -/
def ThreeDigitNumber.digitProduct (n : ThreeDigitNumber) : Nat :=
  n.hundreds * n.tens * n.ones

theorem three_digit_number_difference (a b c : ThreeDigitNumber) :
  a.digitProduct = 64 →
  b.digitProduct = 35 →
  c.digitProduct = 81 →
  a.hundreds + b.hundreds + c.hundreds = 24 →
  a.tens + b.tens + c.tens = 12 →
  a.ones + b.ones + c.ones = 6 →
  max (max a.value b.value) c.value - min (min a.value b.value) c.value = 182 := by
  sorry

end three_digit_number_difference_l2141_214158


namespace plates_used_l2141_214124

theorem plates_used (guests : ℕ) (meals_per_day : ℕ) (plates_per_meal : ℕ) (days : ℕ) : 
  guests = 5 →
  meals_per_day = 3 →
  plates_per_meal = 2 →
  days = 4 →
  (guests + 1) * meals_per_day * plates_per_meal * days = 144 :=
by sorry

end plates_used_l2141_214124


namespace at_least_four_boxes_same_items_l2141_214114

theorem at_least_four_boxes_same_items (boxes : Finset Nat) (items : Nat → Nat) : 
  boxes.card = 376 → 
  (∀ b ∈ boxes, items b ≤ 125) → 
  ∃ n : Nat, ∃ same_boxes : Finset Nat, same_boxes ⊆ boxes ∧ same_boxes.card ≥ 4 ∧ 
    ∀ b ∈ same_boxes, items b = n :=
by sorry

end at_least_four_boxes_same_items_l2141_214114


namespace unique_solution_sqrt_equation_l2141_214138

theorem unique_solution_sqrt_equation (m n : ℤ) :
  (5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n ↔ m = 0 ∧ n = 0 :=
by sorry

end unique_solution_sqrt_equation_l2141_214138


namespace sanchez_rope_theorem_l2141_214146

/-- The length of rope Mr. Sanchez bought last week in feet -/
def rope_last_week : ℕ := 6

/-- The difference in feet between last week's and this week's rope purchase -/
def rope_difference : ℕ := 4

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The total length of rope Mr. Sanchez bought in inches -/
def total_rope_inches : ℕ := (rope_last_week + (rope_last_week - rope_difference)) * inches_per_foot

theorem sanchez_rope_theorem : total_rope_inches = 96 := by
  sorry

end sanchez_rope_theorem_l2141_214146


namespace min_editors_l2141_214174

theorem min_editors (total : ℕ) (writers : ℕ) (x : ℕ) (both_max : ℕ) :
  total = 100 →
  writers = 40 →
  x ≤ both_max →
  both_max = 21 →
  total = writers + x + 2 * x →
  ∃ (editors : ℕ), editors ≥ 39 ∧ total = writers + editors + x :=
by sorry

end min_editors_l2141_214174


namespace rachel_father_age_at_25_l2141_214176

/-- Rachel's current age -/
def rachel_age : ℕ := 12

/-- Rachel's grandfather's age in terms of Rachel's age -/
def grandfather_age_factor : ℕ := 7

/-- Rachel's mother's age in terms of grandfather's age -/
def mother_age_factor : ℚ := 1/2

/-- Age difference between Rachel's father and mother -/
def father_mother_age_diff : ℕ := 5

/-- Rachel's target age -/
def rachel_target_age : ℕ := 25

/-- Theorem stating Rachel's father's age when Rachel is 25 -/
theorem rachel_father_age_at_25 : 
  rachel_age * grandfather_age_factor * mother_age_factor + father_mother_age_diff + 
  (rachel_target_age - rachel_age) = 60 := by
  sorry

end rachel_father_age_at_25_l2141_214176


namespace translation_proof_l2141_214170

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point in 2D -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem translation_proof :
  let P : Point := { x := -3, y := 2 }
  let translated_P := translate (translate P 2 0) 0 (-2)
  translated_P = { x := -1, y := 0 } := by sorry

end translation_proof_l2141_214170


namespace express_y_in_terms_of_x_l2141_214127

-- Define the variables x and y as real numbers
variable (x y : ℝ)

-- State the theorem
theorem express_y_in_terms_of_x (h : 2 * x + y = 1) : y = -2 * x + 1 := by
  sorry

end express_y_in_terms_of_x_l2141_214127


namespace paper_reams_for_haley_l2141_214166

theorem paper_reams_for_haley (total_reams sister_reams : ℕ) 
  (h1 : total_reams = 5)
  (h2 : sister_reams = 3) :
  total_reams - sister_reams = 2 := by
  sorry

end paper_reams_for_haley_l2141_214166


namespace lines_properties_l2141_214134

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x - y + 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 1 = 0

-- Theorem statement
theorem lines_properties (a : ℝ) :
  -- 1. The lines are always perpendicular
  (∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ → l₂ a x₂ y₂ → (x₁ - x₂) * (y₁ - y₂) = 0) ∧
  -- 2. l₁ passes through (0,1) and l₂ passes through (-1,0)
  l₁ a 0 1 ∧ l₂ a (-1) 0 ∧
  -- 3. The maximum distance from the intersection point to the origin is √2
  (∃ x y : ℝ, l₁ a x y ∧ l₂ a x y ∧
    ∀ x' y' : ℝ, l₁ a x' y' → l₂ a x' y' → x'^2 + y'^2 ≤ 2) ∧
  (∃ a₀ x₀ y₀ : ℝ, l₁ a₀ x₀ y₀ ∧ l₂ a₀ x₀ y₀ ∧ x₀^2 + y₀^2 = 2) :=
by sorry

end lines_properties_l2141_214134


namespace point_placement_on_line_l2141_214149

theorem point_placement_on_line : ∃ (a b c d : ℝ),
  |b - a| = 10 ∧
  |c - a| = 3 ∧
  |d - b| = 5 ∧
  |d - c| = 8 ∧
  a = 0 ∧ b = 10 ∧ c = -3 ∧ d = 5 := by
  sorry

end point_placement_on_line_l2141_214149


namespace range_of_a_l2141_214161

/-- The line y = x + 2 intersects the x-axis at point M and the y-axis at point N. -/
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (0, 2)

/-- Point P moves on the circle (x-a)^2 + y^2 = 2, where a > 0 -/
def circle_equation (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + y^2 = 2

/-- Angle MPN is always acute -/
def angle_MPN_acute (P : ℝ × ℝ) : Prop := sorry

theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ P : ℝ × ℝ, circle_equation a P.1 P.2 → angle_MPN_acute P) →
  a > Real.sqrt 7 - 1 :=
sorry

end range_of_a_l2141_214161


namespace range_of_linear_function_l2141_214150

def g (c d x : ℝ) : ℝ := c * x + d

theorem range_of_linear_function (c d : ℝ) (hc : c > 0) :
  Set.range (fun x => g c d x) = Set.Icc (-c + d) (2*c + d) :=
sorry

end range_of_linear_function_l2141_214150


namespace second_player_wins_l2141_214140

/-- A game on a circle with 2n + 1 equally spaced points -/
structure CircleGame where
  n : ℕ
  h : n ≥ 2

/-- A strategy for the second player -/
def SecondPlayerStrategy (game : CircleGame) : Type :=
  ℕ → ℕ

/-- Predicate to check if a triangle is obtuse -/
def IsObtuse (p1 p2 p3 : ℕ) : Prop :=
  sorry

/-- Predicate to check if all remaining triangles are obtuse -/
def AllTrianglesObtuse (remaining_points : List ℕ) : Prop :=
  sorry

/-- Predicate to check if a strategy is winning for the second player -/
def IsWinningStrategy (game : CircleGame) (strategy : SecondPlayerStrategy game) : Prop :=
  ∀ (first_player_moves : List ℕ),
    AllTrianglesObtuse (sorry) -- remaining points after applying the strategy

theorem second_player_wins (game : CircleGame) :
  ∃ (strategy : SecondPlayerStrategy game), IsWinningStrategy game strategy :=
sorry

end second_player_wins_l2141_214140


namespace two_digit_quadratic_equation_l2141_214108

theorem two_digit_quadratic_equation :
  ∃ (P : ℕ), 
    (P ≥ 10 ∧ P < 100) ∧ 
    (∀ x : ℝ, x^2 + P*x + 2001 = (x + 29) * (x + 69)) :=
sorry

end two_digit_quadratic_equation_l2141_214108


namespace bug_crawl_distance_l2141_214100

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ

/-- Calculates the shortest distance between two points on the surface of a cone -/
noncomputable def shortestDistance (c : Cone) (p1 p2 : ConePoint) : ℝ :=
  sorry

theorem bug_crawl_distance (c : Cone) (p1 p2 : ConePoint) :
  c.baseRadius = 500 →
  c.height = 250 * Real.sqrt 3 →
  p1.distanceFromVertex = 100 →
  p2.distanceFromVertex = 300 * Real.sqrt 3 →
  shortestDistance c p1 p2 = 100 * Real.sqrt 23 := by
  sorry

end bug_crawl_distance_l2141_214100


namespace product_pqr_l2141_214157

theorem product_pqr (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 30)
  (h3 : 1 / p + 1 / q + 1 / r + 240 / (p * q * r) = 1) :
  p * q * r = 1080 := by
sorry

end product_pqr_l2141_214157


namespace amount_difference_l2141_214119

def distribute_amount (total : ℝ) (p q r s t : ℝ) : Prop :=
  total = 25000 ∧
  p = 2 * q ∧
  s = 4 * r ∧
  q = r ∧
  p + q + r = (5/9) * total ∧
  s / (s + t) = 2/3 ∧
  s - p = 6944.4444

theorem amount_difference :
  ∀ (total p q r s t : ℝ),
  distribute_amount total p q r s t →
  s - p = 6944.4444 :=
by sorry

end amount_difference_l2141_214119


namespace total_nuts_weight_l2141_214117

def almonds : Real := 0.14
def pecans : Real := 0.38

theorem total_nuts_weight : almonds + pecans = 0.52 := by sorry

end total_nuts_weight_l2141_214117


namespace chess_team_arrangement_l2141_214144

/-- Represents the number of boys on the chess team -/
def num_boys : ℕ := 3

/-- Represents the number of girls on the chess team -/
def num_girls : ℕ := 2

/-- Represents the total number of team members -/
def total_members : ℕ := num_boys + num_girls

/-- Represents the number of ways to arrange the team members according to the specified conditions -/
def arrangements : ℕ := num_girls.factorial * num_boys.factorial

theorem chess_team_arrangement : arrangements = 12 := by
  sorry

end chess_team_arrangement_l2141_214144


namespace min_candies_pile_l2141_214156

theorem min_candies_pile : ∃ N : ℕ, N > 0 ∧ 
  (∃ k₁ : ℕ, N - 5 = 2 * k₁) ∧ 
  (∃ k₂ : ℕ, N - 2 = 3 * k₂) ∧ 
  (∃ k₃ : ℕ, N - 3 = 5 * k₃) ∧ 
  (∀ M : ℕ, M > 0 → 
    ((∃ m₁ : ℕ, M - 5 = 2 * m₁) ∧ 
     (∃ m₂ : ℕ, M - 2 = 3 * m₂) ∧ 
     (∃ m₃ : ℕ, M - 3 = 5 * m₃)) → M ≥ N) ∧
  N = 53 := by
sorry

end min_candies_pile_l2141_214156


namespace vector_sum_magnitude_l2141_214111

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 1) :
  Real.sqrt (((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2)) = 2 * Real.sqrt 3 := by
  sorry

end vector_sum_magnitude_l2141_214111


namespace solution_system_1_solution_system_2_l2141_214175

-- System (1)
theorem solution_system_1 (x y : ℝ) : 
  (4*x + 8*y = 12 ∧ 3*x - 2*y = 5) → (x = 2 ∧ y = 1/2) := by sorry

-- System (2)
theorem solution_system_2 (x y : ℝ) : 
  ((1/2)*x - (y+1)/3 = 1 ∧ 6*x + 2*y = 10) → (x = 2 ∧ y = -1) := by sorry

end solution_system_1_solution_system_2_l2141_214175


namespace isosceles_triangle_unique_range_l2141_214190

theorem isosceles_triangle_unique_range (a : ℝ) :
  (∃ (x y : ℝ), x^2 - 6*x + a = 0 ∧ y^2 - 6*y + a = 0 ∧ 
   x ≠ y ∧ 
   (x < y → 2*x ≤ y) ∧
   (y < x → 2*y ≤ x)) ↔
  (0 < a ∧ a ≤ 9) :=
sorry

end isosceles_triangle_unique_range_l2141_214190


namespace equation_roots_range_l2141_214180

theorem equation_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧
    3^(2*x + 1) + (m-1)*(3^(x+1) - 1) - (m-3)*3^x = 0 ∧
    3^(2*y + 1) + (m-1)*(3^(y+1) - 1) - (m-3)*3^y = 0) →
  m < (-3 - Real.sqrt 21) / 2 :=
by sorry

end equation_roots_range_l2141_214180


namespace mans_speed_with_stream_l2141_214143

/-- Given a man's rowing rate in still water and his speed against the stream,
    prove that his speed with the stream is equal to twice his rate in still water
    minus his speed against the stream. -/
theorem mans_speed_with_stream
  (rate_still_water : ℝ)
  (speed_against_stream : ℝ)
  (h1 : rate_still_water = 7)
  (h2 : speed_against_stream = 4) :
  rate_still_water + (rate_still_water - speed_against_stream) = 2 * rate_still_water - speed_against_stream :=
by sorry

end mans_speed_with_stream_l2141_214143


namespace tom_sleep_hours_l2141_214148

/-- Proves that Tom was getting 6 hours of sleep before increasing it by 1/3 to 8 hours --/
theorem tom_sleep_hours : 
  ∀ (x : ℝ), 
  (x + (1/3) * x = 8) → 
  x = 6 := by
sorry

end tom_sleep_hours_l2141_214148


namespace proportion_problem_l2141_214160

theorem proportion_problem (x y z v : ℤ) : 
  (x * v = y * z) →
  (x + v = y + z + 7) →
  (x^2 + v^2 = y^2 + z^2 + 21) →
  (x^4 + v^4 = y^4 + z^4 + 2625) →
  ((x = -3 ∧ v = 8 ∧ y = -6 ∧ z = 4) ∨ 
   (x = 8 ∧ v = -3 ∧ y = 4 ∧ z = -6)) := by
  sorry

end proportion_problem_l2141_214160


namespace book_prices_l2141_214199

theorem book_prices (book1_discounted : ℚ) (book2_discounted : ℚ)
  (h1 : book1_discounted = 8)
  (h2 : book2_discounted = 9)
  (h3 : book1_discounted = (1 / 8 : ℚ) * book1_discounted / (1 / 8 : ℚ))
  (h4 : book2_discounted = (1 / 9 : ℚ) * book2_discounted / (1 / 9 : ℚ)) :
  book1_discounted / (1 / 8 : ℚ) + book2_discounted / (1 / 9 : ℚ) = 145 := by
sorry

end book_prices_l2141_214199


namespace sunflower_rose_height_difference_l2141_214179

/-- The height difference between a sunflower and a rose bush -/
theorem sunflower_rose_height_difference :
  let sunflower_height : ℚ := 9 + 3/5
  let rose_height : ℚ := 5 + 4/5
  sunflower_height - rose_height = 3 + 4/5 := by sorry

end sunflower_rose_height_difference_l2141_214179


namespace train_crossing_time_l2141_214113

/-- Given a train crossing a platform, calculate the time it takes to cross a signal pole --/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 675)
  (h3 : platform_crossing_time = 39)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 12 := by
  sorry

end train_crossing_time_l2141_214113


namespace dividend_calculation_l2141_214167

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 18)
  (h2 : quotient = 9)
  (h3 : remainder = 4) :
  divisor * quotient + remainder = 166 := by
  sorry

end dividend_calculation_l2141_214167


namespace de_length_l2141_214194

/-- Triangle ABC with given side lengths and a line DE parallel to BC containing the incenter --/
structure TriangleWithParallelLine where
  -- Define the triangle
  AB : ℝ
  AC : ℝ
  BC : ℝ
  -- Ensure AB = 21, AC = 22, BC = 20
  h_AB : AB = 21
  h_AC : AC = 22
  h_BC : BC = 20
  -- Points D and E are on AB and AC respectively
  D : ℝ
  E : ℝ
  h_D : D ≥ 0 ∧ D ≤ AB
  h_E : E ≥ 0 ∧ E ≤ AC
  -- DE is parallel to BC
  h_parallel : True  -- We can't directly express parallelism here, so we assume it's true
  -- DE contains the incenter
  h_incenter : True  -- We can't directly express this, so we assume it's true

/-- The main theorem --/
theorem de_length (t : TriangleWithParallelLine) : ∃ (DE : ℝ), DE = 860 / 63 := by
  sorry

end de_length_l2141_214194


namespace inequality_proof_l2141_214183

theorem inequality_proof (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := by
  sorry

end inequality_proof_l2141_214183


namespace no_solution_exists_l2141_214159

theorem no_solution_exists : ¬∃ (x y : ℝ), 9^(y+1) / (1 + 4 / x^2) = 1 := by
  sorry

end no_solution_exists_l2141_214159


namespace quadratic_max_value_l2141_214122

-- Define the quadratic polynomial
def p (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_max_value (a b c : ℝ) (ha : a > 0) (h1 : p a b c 1 = 4) (h2 : p a b c 2 = 15) :
  (∃ (x : ℝ), ∀ (y : ℝ), p a b c y ≤ p a b c x) ∧
  (∀ (x : ℝ), p a b c x ≤ 4) ∧
  p a b c 1 = 4 := by
  sorry

end quadratic_max_value_l2141_214122


namespace degree_of_g_l2141_214185

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := -9 * x^5 + 4 * x^3 + 2 * x - 6

-- Define a proposition for the degree of a polynomial
def hasDegree (p : ℝ → ℝ) (n : ℕ) : Prop := sorry

-- State the theorem
theorem degree_of_g 
  (g : ℝ → ℝ) 
  (h : hasDegree (fun x => f x + g x) 2) : 
  hasDegree g 5 := by sorry

end degree_of_g_l2141_214185


namespace fred_balloon_count_l2141_214115

/-- The number of blue balloons Sally has -/
def sally_balloons : ℕ := 6

/-- The factor by which Fred has more balloons than Sally -/
def fred_factor : ℕ := 3

/-- The number of blue balloons Fred has -/
def fred_balloons : ℕ := sally_balloons * fred_factor

theorem fred_balloon_count : fred_balloons = 18 := by
  sorry

end fred_balloon_count_l2141_214115


namespace sqrt_two_power_2000_identity_l2141_214145

theorem sqrt_two_power_2000_identity : 
  (Real.sqrt 2 + 1)^2000 * (Real.sqrt 2 - 1)^2000 = 1 := by sorry

end sqrt_two_power_2000_identity_l2141_214145


namespace complex_sum_powers_l2141_214135

theorem complex_sum_powers (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^96 + z^97 + z^98 + z^99 + z^100 + z^101 = 0 := by
  sorry

end complex_sum_powers_l2141_214135


namespace chess_tournament_games_l2141_214147

theorem chess_tournament_games (n : ℕ) (h : n = 5) : 
  (n * (n - 1)) / 2 = 10 := by
  sorry

end chess_tournament_games_l2141_214147


namespace correct_swap_l2141_214186

def swap_values (a b : ℕ) : ℕ × ℕ :=
  let c := b
  let b' := a
  let a' := c
  (a', b')

theorem correct_swap :
  swap_values 6 5 = (5, 6) := by
sorry

end correct_swap_l2141_214186


namespace obtuse_triangle_area_l2141_214193

theorem obtuse_triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 8) (h2 : b = 12) (h3 : C = 150 * π / 180) :
  let area := (1/2) * a * b * Real.sin C
  area = 24 := by
sorry

end obtuse_triangle_area_l2141_214193


namespace not_surjective_product_of_injective_exists_surjective_factors_l2141_214152

-- Part a
theorem not_surjective_product_of_injective (f g : ℤ → ℤ) 
  (hf : Function.Injective f) (hg : Function.Injective g) :
  ¬ Function.Surjective (fun x ↦ f x * g x) := by
  sorry

-- Part b
theorem exists_surjective_factors (f : ℤ → ℤ) (hf : Function.Surjective f) :
  ∃ g h : ℤ → ℤ, Function.Surjective g ∧ Function.Surjective h ∧
    ∀ x, f x = g x * h x := by
  sorry

end not_surjective_product_of_injective_exists_surjective_factors_l2141_214152


namespace same_sign_range_l2141_214173

theorem same_sign_range (m : ℝ) : (2 - m) * (|m| - 3) > 0 ↔ m ∈ Set.Ioo 2 3 ∪ Set.Iio (-3) := by
  sorry

end same_sign_range_l2141_214173


namespace solve_exponential_equation_l2141_214141

theorem solve_exponential_equation :
  ∃ x : ℝ, (16 : ℝ) ^ x * (16 : ℝ) ^ x * (16 : ℝ) ^ x * (16 : ℝ) ^ x = (256 : ℝ) ^ 4 ∧ x = 2 := by
  sorry

end solve_exponential_equation_l2141_214141


namespace inequality_solution_set_l2141_214151

theorem inequality_solution_set (x : ℝ) :
  (x / (x - 1) + (x + 1) / (2 * x) ≥ 5 / 2) ↔ (x ≥ 1 / 2 ∧ x ≤ 1) :=
by sorry

end inequality_solution_set_l2141_214151


namespace replaced_person_weight_l2141_214196

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℚ) (new_person_weight : ℚ) : ℚ :=
  new_person_weight - (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the replaced person is 65 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 (5/2) 85 = 65 := by
  sorry

#eval weight_of_replaced_person 8 (5/2) 85

end replaced_person_weight_l2141_214196


namespace inscribed_square_area_l2141_214188

def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/8 = 1

def inscribed_square (s : ℝ) : Prop :=
  ellipse s s ∧ ellipse (-s) s ∧ ellipse s (-s) ∧ ellipse (-s) (-s)

theorem inscribed_square_area :
  ∃ s : ℝ, inscribed_square s ∧ (2*s)^2 = 32/3 := by sorry

end inscribed_square_area_l2141_214188


namespace sum_of_squares_equals_18_l2141_214132

/-- Right triangle ABC with hypotenuse AB -/
structure RightTriangle where
  AC : ℝ
  BC : ℝ
  AB : ℝ
  right_angle : AC^2 + BC^2 = AB^2

theorem sum_of_squares_equals_18 (triangle : RightTriangle) (h : triangle.AB = 3) :
  triangle.AB^2 + triangle.BC^2 + triangle.AC^2 = 18 := by
  sorry

end sum_of_squares_equals_18_l2141_214132


namespace unique_solution_cube_root_plus_square_root_l2141_214110

theorem unique_solution_cube_root_plus_square_root (x : ℝ) :
  (((x - 3) ^ (1/3 : ℝ)) + ((5 - x) ^ (1/2 : ℝ)) = 2) ↔ (x = 4) :=
sorry

end unique_solution_cube_root_plus_square_root_l2141_214110


namespace parabola_equation_l2141_214162

/-- A parabola with vertex at the origin and directrix x = 4 has the standard equation y^2 = -16x -/
theorem parabola_equation (y x : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ y^2 = -2*p*x) → -- Standard form of parabola equation
  (4 = p/2) →                        -- Condition for directrix at x = 4
  y^2 = -16*x :=                     -- Resulting equation
by sorry

end parabola_equation_l2141_214162


namespace intersection_M_N_l2141_214107

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x : ℕ | x - 1 ≥ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end intersection_M_N_l2141_214107


namespace quadratic_minimum_l2141_214195

theorem quadratic_minimum (x : ℝ) : 
  (∀ x, 4 * x^2 + 8 * x + 16 ≥ 12) ∧ 
  (∃ x, 4 * x^2 + 8 * x + 16 = 12) := by
  sorry

end quadratic_minimum_l2141_214195


namespace system_solution_l2141_214171

theorem system_solution :
  ∀ x y : ℝ,
  x^2 - 3*y - 88 ≥ 0 →
  x + 6*y ≥ 0 →
  (5 * Real.sqrt (x^2 - 3*y - 88) + Real.sqrt (x + 6*y) = 19 ∧
   3 * Real.sqrt (x^2 - 3*y - 88) = 1 + 2 * Real.sqrt (x + 6*y)) →
  ((x = 10 ∧ y = 1) ∨ (x = -21/2 ∧ y = 53/12)) :=
by sorry

end system_solution_l2141_214171


namespace perpendicular_lines_a_equals_one_l2141_214137

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_lines_a_equals_one :
  ∀ a : ℝ,
  perpendicular (-a/3) 3 →
  a = 1 := by
sorry

end perpendicular_lines_a_equals_one_l2141_214137


namespace license_plate_count_l2141_214139

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_count : ℕ := 10

/-- The number of even digits -/
def even_digit_count : ℕ := 5

/-- The number of odd digits -/
def odd_digit_count : ℕ := 5

/-- The number of letters in the license plate -/
def letter_count : ℕ := 3

/-- The number of digits in the license plate -/
def plate_digit_count : ℕ := 3

/-- The number of ways to arrange the odd, even, and any digit -/
def digit_arrangements : ℕ := 3

theorem license_plate_count :
  (alphabet_size ^ letter_count) *
  (even_digit_count * odd_digit_count * digit_count) *
  digit_arrangements = 13182000 := by
  sorry

end license_plate_count_l2141_214139


namespace parabola_segment_length_squared_l2141_214189

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y = 3x^2 - 4x + 5 -/
def onParabola (p : Point) : Prop :=
  p.y = 3 * p.x^2 - 4 * p.x + 5

/-- The origin (0, 0) is the midpoint of two points -/
def originIsMidpoint (p q : Point) : Prop :=
  p.x = -q.x ∧ p.y = -q.y

/-- The square of the distance between two points -/
def squareDistance (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- The main theorem -/
theorem parabola_segment_length_squared :
  ∀ p q : Point,
  onParabola p → onParabola q → originIsMidpoint p q →
  squareDistance p q = 8900 / 9 := by
  sorry

end parabola_segment_length_squared_l2141_214189


namespace divisible_by_10101_l2141_214155

/-- Given a two-digit number, returns the six-digit number formed by repeating it three times -/
def f (n : ℕ) : ℕ :=
  100000 * n + 1000 * n + 10 * n

/-- Theorem: For any two-digit number n, f(n) is divisible by 10101 -/
theorem divisible_by_10101 (n : ℕ) (h : 10 ≤ n ∧ n < 100) : 
  ∃ k : ℕ, f n = 10101 * k := by
  sorry

end divisible_by_10101_l2141_214155


namespace binomial_probability_two_l2141_214121

/-- The probability mass function for a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The problem statement -/
theorem binomial_probability_two (X : ℕ → ℝ) :
  (∀ k, X k = binomial_pmf 6 (1/3) k) →
  X 2 = 80/243 := by
  sorry

end binomial_probability_two_l2141_214121


namespace wood_cutting_problem_l2141_214128

theorem wood_cutting_problem : Nat.gcd 90 72 = 18 := by
  sorry

end wood_cutting_problem_l2141_214128


namespace min_product_equal_sum_l2141_214123

theorem min_product_equal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b) :
  a * b ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = a₀ + b₀ ∧ a₀ * b₀ = 4 :=
by sorry

end min_product_equal_sum_l2141_214123


namespace product_of_numbers_with_hcf_l2141_214136

theorem product_of_numbers_with_hcf (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  Nat.gcd a b = 11 ∧ 
  a = 33 ∧ 
  ∀ c, c > 0 ∧ Nat.gcd a c = 11 → b ≤ c →
  a * b = 363 := by
  sorry

end product_of_numbers_with_hcf_l2141_214136


namespace equation_holds_for_all_y_l2141_214192

theorem equation_holds_for_all_y (x : ℝ) : 
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := by
sorry

end equation_holds_for_all_y_l2141_214192


namespace equation_solution_l2141_214104

theorem equation_solution : 
  ∀ x : ℝ, (x + 1)^2 - 144 = 0 ↔ x = 11 ∨ x = -13 := by
sorry

end equation_solution_l2141_214104


namespace system_of_equations_solution_l2141_214187

theorem system_of_equations_solution 
  (x y z a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (eq1 : y * z / (y + z) = a)
  (eq2 : x * z / (x + z) = b)
  (eq3 : x * y / (x + y) = c) :
  x = 2 * a * b * c / (a * c + a * b - b * c) ∧
  y = 2 * a * b * c / (a * b + b * c - a * c) ∧
  z = 2 * a * b * c / (a * c + b * c - a * b) :=
by sorry

end system_of_equations_solution_l2141_214187


namespace gcd_4830_3289_l2141_214198

theorem gcd_4830_3289 : Nat.gcd 4830 3289 = 23 := by
  sorry

end gcd_4830_3289_l2141_214198


namespace intersection_circles_angle_relation_l2141_214164

/-- Given two intersecting circles with radius R and centers separated by a distance greater than R,
    prove that the angle β formed at one intersection point is three times the angle α formed at the other intersection point. -/
theorem intersection_circles_angle_relation (R : ℝ) (center_distance : ℝ) (α β : ℝ) :
  R > 0 →
  center_distance > R →
  α > 0 →
  β > 0 →
  β = 3 * α :=
by sorry

end intersection_circles_angle_relation_l2141_214164


namespace rectangle_width_l2141_214109

theorem rectangle_width (perimeter : ℝ) (length_difference : ℝ) (width : ℝ) : 
  perimeter = 48 →
  length_difference = 2 →
  perimeter = 2 * (width + length_difference) + 2 * width →
  width = 11 := by
sorry

end rectangle_width_l2141_214109


namespace max_angle_at_C_l2141_214118

/-- The line c given by the equation y = x + 1 -/
def line_c : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1}

/-- Point A with coordinates (1, 0) -/
def point_A : ℝ × ℝ := (1, 0)

/-- Point B with coordinates (3, 0) -/
def point_B : ℝ × ℝ := (3, 0)

/-- Point C with coordinates (1, 2) -/
def point_C : ℝ × ℝ := (1, 2)

/-- The angle between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating that C maximizes the angle ACB -/
theorem max_angle_at_C :
  point_C ∈ line_c ∧
  ∀ p ∈ line_c, angle point_A p point_B ≤ angle point_A point_C point_B :=
by sorry

end max_angle_at_C_l2141_214118


namespace polynomial_equality_l2141_214131

theorem polynomial_equality (x : ℝ) :
  let k : ℝ → ℝ := λ x => -5*x^5 + 7*x^4 - 7*x^3 - x + 2
  5*x^5 + 3*x^3 + x + k x = 7*x^4 - 4*x^3 + 2 := by
sorry

end polynomial_equality_l2141_214131


namespace integer_sum_problem_l2141_214184

theorem integer_sum_problem (x y : ℤ) : 
  x > 0 → y > 0 → x - y = 16 → x * y = 162 → x + y = 30 := by
  sorry

end integer_sum_problem_l2141_214184


namespace no_square_143_b_l2141_214177

theorem no_square_143_b : ¬ ∃ (b : ℤ), b > 4 ∧ ∃ (n : ℤ), b^2 + 4*b + 3 = n^2 := by
  sorry

end no_square_143_b_l2141_214177


namespace nickel_count_l2141_214126

/-- Given a purchase of 150 cents paid with 50 coins consisting of only pennies and nickels,
    prove that the number of nickels used is 25. -/
theorem nickel_count (p n : ℕ) : 
  p + n = 50 →  -- Total number of coins
  p + 5 * n = 150 →  -- Total value in cents
  n = 25 := by sorry

end nickel_count_l2141_214126


namespace largest_power_dividing_factorial_l2141_214116

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem largest_power_dividing_factorial :
  let n := 2520
  ∃ k : ℕ, k = 418 ∧
    (∀ m : ℕ, n^m ∣ factorial n → m ≤ k) ∧
    n^k ∣ factorial n :=
by sorry

end largest_power_dividing_factorial_l2141_214116


namespace spider_web_paths_l2141_214106

/-- The number of paths from (0, 0) to (m, n) on a grid, moving only right and up -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- The coordinates of the fly -/
def flyPosition : ℕ × ℕ := (5, 3)

theorem spider_web_paths :
  gridPaths flyPosition.1 flyPosition.2 = 56 := by
  sorry

end spider_web_paths_l2141_214106


namespace equilateral_triangle_area_l2141_214168

/-- The area of an equilateral triangle with altitude √15 is 5√3 square units. -/
theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = Real.sqrt 15) :
  let side : ℝ := 2 * Real.sqrt 5
  let area : ℝ := (side * h) / 2
  area = 5 * Real.sqrt 3 :=
by sorry

end equilateral_triangle_area_l2141_214168


namespace fifth_term_of_arithmetic_sequence_l2141_214101

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 5th term of an arithmetic sequence equals 8, given a₃ + a₇ = 16 -/
theorem fifth_term_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h : arithmetic_sequence a) (sum_eq : a 3 + a 7 = 16) : 
  a 5 = 8 := by sorry

end fifth_term_of_arithmetic_sequence_l2141_214101


namespace point_in_first_quadrant_l2141_214153

/-- Given points A and B with line AB parallel to y-axis, prove (-a, a+3) is in first quadrant --/
theorem point_in_first_quadrant (a : ℝ) : 
  (a - 1 = -2) →  -- Line AB parallel to y-axis implies x-coordinates are equal
  ((-a > 0) ∧ (a + 3 > 0)) := by
  sorry

end point_in_first_quadrant_l2141_214153


namespace extended_segment_endpoint_l2141_214165

/-- Given points A and B in 2D space, and a point C such that BC = 1/2 * AB,
    prove that C has specific coordinates. -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (-3, 5) → 
  B = (9, -1) → 
  C - B = (1/2 : ℝ) • (B - A) → 
  C = (15, -4) := by
  sorry

end extended_segment_endpoint_l2141_214165


namespace disney_banquet_attendees_l2141_214102

/-- The number of people who attended a Disney banquet -/
theorem disney_banquet_attendees :
  ∀ (resident_price non_resident_price total_revenue : ℚ) 
    (num_residents : ℕ) (total_attendees : ℕ),
  resident_price = 1295/100 →
  non_resident_price = 1795/100 →
  total_revenue = 942370/100 →
  num_residents = 219 →
  total_revenue = (num_residents : ℚ) * resident_price + 
    ((total_attendees - num_residents) : ℚ) * non_resident_price →
  total_attendees = 586 := by
sorry

end disney_banquet_attendees_l2141_214102


namespace stating_ball_338_position_l2141_214169

/-- 
Given a circular arrangement of 1000 cups where balls are placed in every 7th cup 
starting from cup 1, this function calculates the cup number for the nth ball.
-/
def ball_position (n : ℕ) : ℕ := 
  (1 + (n - 1) * 7) % 1000

/-- 
Theorem stating that the 338th ball will be placed in cup 359 
in the described arrangement.
-/
theorem ball_338_position : ball_position 338 = 359 := by
  sorry

#eval ball_position 338  -- This line is for verification purposes

end stating_ball_338_position_l2141_214169
