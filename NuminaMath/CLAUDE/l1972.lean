import Mathlib

namespace smallest_positive_angle_l1972_197254

open Real

-- Define the equation
def equation (x : ℝ) : Prop :=
  tan (5 * x * π / 180) = (1 - sin (x * π / 180)) / (1 + sin (x * π / 180))

-- State the theorem
theorem smallest_positive_angle :
  ∃ (x : ℝ), x > 0 ∧ x < 10 ∧ equation x ∧ ∀ (y : ℝ), 0 < y ∧ y < x → ¬(equation y) :=
by sorry

end smallest_positive_angle_l1972_197254


namespace circle_equation_l1972_197286

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) -/
theorem circle_equation (x y : ℝ) : 
  (∀ D E F : ℝ, (x^2 + y^2 + D*x + E*y + F = 0) → 
    (0^2 + 0^2 + D*0 + E*0 + F = 0 ∧ 
     4^2 + 0^2 + D*4 + E*0 + F = 0 ∧ 
     (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0)) →
  (x^2 + y^2 - 4*x - 6*y = 0 ↔ 
    (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
sorry

end circle_equation_l1972_197286


namespace geometric_sequence_common_ratio_l1972_197241

theorem geometric_sequence_common_ratio 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 27) 
  (h₂ : a₂ = 54) 
  (h₃ : a₃ = 108) 
  (h_geom : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) : 
  ∃ r : ℝ, r = 2 ∧ a₂ = a₁ * r ∧ a₃ = a₂ * r :=
sorry

end geometric_sequence_common_ratio_l1972_197241


namespace triangle_area_and_length_l1972_197272

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively, and point D as the midpoint of BC -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ × ℝ

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem triangle_area_and_length (t : Triangle) :
  (t.c * Real.cos t.B = Real.sqrt 3 * t.b * Real.sin t.C) →
  (t.a^2 * Real.sin t.C = 4 * Real.sqrt 3 * Real.sin t.A) →
  (area t = Real.sqrt 3) ∧
  (t.a = 2 * Real.sqrt 3 → t.b = Real.sqrt 7 → t.c > t.b →
   distance (0, 0) t.D = Real.sqrt 13) := by
  sorry

end triangle_area_and_length_l1972_197272


namespace new_average_after_22_innings_l1972_197217

def calculate_new_average (initial_innings : ℕ) (score_17th : ℕ) (average_increase : ℕ) 
  (additional_scores : List ℕ) : ℕ :=
  let total_innings := initial_innings + additional_scores.length
  let initial_average := (initial_innings - 1) * (average_increase + 1) / initial_innings
  let total_runs_17 := initial_innings * (initial_average + average_increase)
  let total_runs_22 := total_runs_17 + additional_scores.sum
  total_runs_22 / total_innings

theorem new_average_after_22_innings : 
  calculate_new_average 17 85 3 [100, 120, 45, 75, 65] = 47 := by
  sorry

end new_average_after_22_innings_l1972_197217


namespace friday_pushups_equal_total_l1972_197293

def monday_pushups : ℕ := 5
def tuesday_pushups : ℕ := 7
def wednesday_pushups : ℕ := 2 * tuesday_pushups
def thursday_pushups : ℕ := (monday_pushups + tuesday_pushups + wednesday_pushups) / 2
def total_monday_to_thursday : ℕ := monday_pushups + tuesday_pushups + wednesday_pushups + thursday_pushups

theorem friday_pushups_equal_total : total_monday_to_thursday = 39 := by
  sorry

end friday_pushups_equal_total_l1972_197293


namespace g_domain_l1972_197212

def f_domain : Set ℝ := Set.Icc 0 2

def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 1)

theorem g_domain (f : ℝ → ℝ) (h : ∀ x, x ∈ f_domain ↔ f x ≠ 0) :
  ∀ x, x ∈ Set.Icc (-1) 1 ↔ g f x ≠ 0 :=
sorry

end g_domain_l1972_197212


namespace ellipse_m_value_l1972_197210

/-- An ellipse with equation x^2 + my^2 = 1, where m is a positive real number -/
structure Ellipse (m : ℝ) : Type :=
  (eq : ∀ (x y : ℝ), x^2 + m*y^2 = 1)

/-- The foci of the ellipse are on the y-axis -/
def foci_on_y_axis (e : Ellipse m) : Prop :=
  ∃ (c : ℝ), c^2 = 1/m - 1

/-- The length of the major axis is twice the length of the minor axis -/
def major_axis_twice_minor (e : Ellipse m) : Prop :=
  2 * Real.sqrt 1 = Real.sqrt (1/m)

/-- The theorem stating that m = 1/4 for the given conditions -/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m)
  (h1 : m > 0)
  (h2 : foci_on_y_axis e)
  (h3 : major_axis_twice_minor e) :
  m = 1/4 := by
  sorry

end ellipse_m_value_l1972_197210


namespace bobby_candy_count_l1972_197218

/-- The total number of candy pieces Bobby ate -/
def total_candy (initial : ℕ) (more : ℕ) (chocolate : ℕ) : ℕ :=
  initial + more + chocolate

/-- Theorem stating that Bobby ate 133 pieces of candy in total -/
theorem bobby_candy_count :
  total_candy 28 42 63 = 133 := by
  sorry

end bobby_candy_count_l1972_197218


namespace circle_intersection_existence_l1972_197208

/-- Given a circle with diameter 2R and a line perpendicular to the diameter at distance a from one endpoint,
    this theorem states the conditions for the existence of points C on the circle and D on the perpendicular line
    such that CD = l. -/
theorem circle_intersection_existence (R a l : ℝ) : 
  (∃ (C D : ℝ × ℝ), 
    C.1^2 + C.2^2 = R^2 ∧ 
    D.1 = a ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = l^2) ↔ 
  ((0 < a ∧ a < 2*R ∧ l < 2*R - a) ∨
   (a > 2*R ∧ R > 0 ∧ l > a - 2*R) ∨
   (-2*R < a ∧ a < 0 ∧ l^2 ≥ -8*R*a ∧ l < 2*R - a) ∨
   (a < -2*R ∧ R < 0 ∧ l > 2*R - a)) :=
by sorry


end circle_intersection_existence_l1972_197208


namespace andrena_has_three_more_than_debelyn_l1972_197291

/-- Represents the number of dolls each person has -/
structure DollCounts where
  debelyn : ℕ
  christel : ℕ
  andrena : ℕ

/-- The initial state of doll ownership -/
def initial : DollCounts :=
  { debelyn := 20
  , christel := 24
  , andrena := 0 }

/-- The state after doll transfers -/
def final : DollCounts :=
  { debelyn := initial.debelyn - 2
  , christel := initial.christel - 5
  , andrena := initial.andrena + 2 + 5 }

theorem andrena_has_three_more_than_debelyn :
  final.andrena = final.christel + 2 →
  final.andrena - final.debelyn = 3 := by
  sorry

end andrena_has_three_more_than_debelyn_l1972_197291


namespace increasing_function_derivative_relation_l1972_197258

open Set
open Function
open Topology

theorem increasing_function_derivative_relation 
  {a b : ℝ} (hab : a < b) (f : ℝ → ℝ) (hf : DifferentiableOn ℝ f (Ioo a b)) :
  (∀ x ∈ Ioo a b, (deriv f) x > 0 → StrictMonoOn f (Ioo a b)) ∧
  ¬(StrictMonoOn f (Ioo a b) → ∀ x ∈ Ioo a b, (deriv f) x > 0) :=
sorry

end increasing_function_derivative_relation_l1972_197258


namespace factorial_equation_solution_l1972_197219

theorem factorial_equation_solution (m k : ℕ) (hm : m = 7) (hk : k = 12) :
  ∃ P : ℕ, (Nat.factorial 7) * (Nat.factorial 14) = 18 * P * (Nat.factorial 11) ∧ P = 54080 := by
  sorry

end factorial_equation_solution_l1972_197219


namespace sum_of_roots_quadratic_equation_l1972_197270

theorem sum_of_roots_quadratic_equation : 
  let f : ℝ → ℝ := λ x => x^2 + x - 2
  ∃ r₁ r₂ : ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = -1 :=
sorry

end sum_of_roots_quadratic_equation_l1972_197270


namespace cube_construction_proof_l1972_197226

/-- Represents a piece of cardboard with foldable and glueable edges -/
structure CardboardPiece where
  foldable_edges : Set (Nat × Nat)
  glueable_edges : Set (Nat × Nat)

/-- Represents a pair of cardboard pieces -/
structure CardboardOption where
  piece1 : CardboardPiece
  piece2 : CardboardPiece

/-- Checks if a CardboardOption can form a cube -/
def can_form_cube (option : CardboardOption) : Prop := sorry

/-- The set of all given options -/
def options : Set CardboardOption := sorry

/-- Option (e) from the given set -/
def option_e : CardboardOption := sorry

theorem cube_construction_proof :
  ∀ opt ∈ options, can_form_cube opt ↔ opt = option_e := by sorry

end cube_construction_proof_l1972_197226


namespace water_needed_for_bread_dough_bakery_recipe_water_needed_l1972_197211

theorem water_needed_for_bread_dough (water_per_portion : ℕ) (flour_per_portion : ℕ) (total_flour : ℕ) : ℕ :=
  let portions := total_flour / flour_per_portion
  portions * water_per_portion

theorem bakery_recipe_water_needed : water_needed_for_bread_dough 75 300 900 = 225 := by
  sorry

end water_needed_for_bread_dough_bakery_recipe_water_needed_l1972_197211


namespace perfect_square_prime_l1972_197207

theorem perfect_square_prime (p : ℕ) (n : ℕ) : 
  Nat.Prime p → (5^p + 4*p^4 = n^2) → p = 5 :=
by sorry

end perfect_square_prime_l1972_197207


namespace taco_salad_cost_correct_l1972_197203

/-- The cost of the Taco Salad at Wendy's -/
def taco_salad_cost : ℚ := 10

/-- The number of friends eating at Wendy's -/
def num_friends : ℕ := 5

/-- The cost of a Dave's Single hamburger -/
def hamburger_cost : ℚ := 5

/-- The number of Dave's Single hamburgers ordered -/
def num_hamburgers : ℕ := 5

/-- The cost of a set of french fries -/
def fries_cost : ℚ := 5/2

/-- The number of sets of french fries ordered -/
def num_fries : ℕ := 4

/-- The cost of a cup of peach lemonade -/
def lemonade_cost : ℚ := 2

/-- The number of cups of peach lemonade ordered -/
def num_lemonade : ℕ := 5

/-- The amount each friend pays when splitting the bill equally -/
def individual_payment : ℚ := 11

theorem taco_salad_cost_correct :
  taco_salad_cost + 
  (num_hamburgers * hamburger_cost) + 
  (num_fries * fries_cost) + 
  (num_lemonade * lemonade_cost) = 
  (num_friends * individual_payment) := by
  sorry

end taco_salad_cost_correct_l1972_197203


namespace product_of_sums_equals_difference_of_powers_l1972_197249

theorem product_of_sums_equals_difference_of_powers : 
  (4 + 3) * (4^2 + 3^2) * (4^4 + 3^4) * (4^8 + 3^8) * (4^16 + 3^16) * (4^32 + 3^32) * (4^64 + 3^64) = 3^128 - 4^128 := by
  sorry

end product_of_sums_equals_difference_of_powers_l1972_197249


namespace sum_of_f_negative_l1972_197200

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_property : ∀ x, f (-x) = -f (x + 4)
axiom f_increasing : ∀ x y, x > 2 → y > x → f y > f x

-- Define the theorem
theorem sum_of_f_negative (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ < 4) 
  (h2 : (x₁ - 2) * (x₂ - 2) < 0) : 
  f x₁ + f x₂ < 0 := by
sorry

end sum_of_f_negative_l1972_197200


namespace average_difference_l1972_197240

theorem average_difference : 
  let m := (12 + 15 + 9 + 14 + 10) / 5
  let n := (24 + 8 + 8 + 12) / 4
  n - m = 1 := by
sorry

end average_difference_l1972_197240


namespace last_three_digits_of_5_to_9000_l1972_197288

theorem last_three_digits_of_5_to_9000 (h : 5^300 ≡ 1 [MOD 800]) :
  5^9000 ≡ 1 [MOD 800] := by
sorry

end last_three_digits_of_5_to_9000_l1972_197288


namespace cayley_hamilton_for_B_l1972_197296

def B : Matrix (Fin 3) (Fin 3) ℝ := !![1, 2, 3; 2, 1, 2; 3, 2, 1]

theorem cayley_hamilton_for_B :
  ∃ (s t u : ℝ), 
    B^3 + s • B^2 + t • B + u • (1 : Matrix (Fin 3) (Fin 3) ℝ) = 0 ∧ 
    s = -7 ∧ t = 2 ∧ u = -9 := by
  sorry

end cayley_hamilton_for_B_l1972_197296


namespace truncated_pyramid_edges_and_height_l1972_197251

theorem truncated_pyramid_edges_and_height :
  ∃ (x y z u r s t : ℤ),
    x = 4 * r * t ∧
    y = 4 * s * t ∧
    z = (r - s)^2 - 2 * t^2 ∧
    u = (r - s)^2 + 2 * t^2 ∧
    (x - y)^2 + 2 * z^2 = 2 * u^2 :=
by sorry

end truncated_pyramid_edges_and_height_l1972_197251


namespace infinite_square_double_numbers_l1972_197265

/-- Definition of a double number -/
def is_double_number (x : ℕ) : Prop :=
  ∃ (d : ℕ), x = d * (10^(Nat.log 10 d + 1) + 1) ∧ d ≠ 0

/-- The main theorem -/
theorem infinite_square_double_numbers :
  ∀ k : ℕ, ∃ N : ℕ,
    let n := 21 * (1 + 14 * k)
    is_double_number (N * (10^n + 1)) ∧
    ∃ m : ℕ, N * (10^n + 1) = m^2 :=
by sorry

end infinite_square_double_numbers_l1972_197265


namespace mary_fruits_left_l1972_197221

/-- The number of apples Mary bought -/
def apples : Nat := 14

/-- The number of oranges Mary bought -/
def oranges : Nat := 9

/-- The number of blueberries Mary bought -/
def blueberries : Nat := 6

/-- The number of each type of fruit Mary ate -/
def eaten : Nat := 1

/-- The total number of fruits Mary has left -/
def fruits_left : Nat := (apples - eaten) + (oranges - eaten) + (blueberries - eaten)

theorem mary_fruits_left : fruits_left = 26 := by
  sorry

end mary_fruits_left_l1972_197221


namespace opposite_reciprocal_expression_value_l1972_197244

theorem opposite_reciprocal_expression_value (a b c d m : ℝ) :
  a + b = 0 →
  c * d = 1 →
  |m| = 4 →
  (a + b) / (3 * m) + m^2 - 5 * c * d + 6 * m = 35 ∨
  (a + b) / (3 * m) + m^2 - 5 * c * d + 6 * m = -13 :=
by sorry

end opposite_reciprocal_expression_value_l1972_197244


namespace find_y_l1972_197236

theorem find_y (x : ℝ) (y : ℝ) : 
  ((100 + 200 + 300 + x) / 4 = 250) →
  ((300 + 150 + 100 + x + y) / 5 = 200) →
  y = 50 := by
sorry

end find_y_l1972_197236


namespace equal_angle_point_exists_l1972_197274

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Three non-overlapping circles -/
structure ThreeCircles where
  c₁ : Circle
  c₂ : Circle
  c₃ : Circle
  non_overlapping : c₁.center ≠ c₂.center ∧ c₂.center ≠ c₃.center ∧ c₁.center ≠ c₃.center

/-- Distance between two points in 2D plane -/
def distance (p₁ p₂ : ℝ × ℝ) : ℝ := sorry

/-- The point from which all circles are seen at the same angle -/
def equal_angle_point (circles : ThreeCircles) (R : ℝ × ℝ) : Prop :=
  let O₁ := circles.c₁.center
  let O₂ := circles.c₂.center
  let O₃ := circles.c₃.center
  let r₁ := circles.c₁.radius
  let r₂ := circles.c₂.radius
  let r₃ := circles.c₃.radius
  (distance O₁ R / distance O₂ R = r₁ / r₂) ∧
  (distance O₂ R / distance O₃ R = r₂ / r₃) ∧
  (distance O₁ R / distance O₃ R = r₁ / r₃)

theorem equal_angle_point_exists (circles : ThreeCircles) :
  ∃ R : ℝ × ℝ, equal_angle_point circles R :=
sorry

end equal_angle_point_exists_l1972_197274


namespace line_equation_l1972_197298

/-- The curve y = 3x^2 - 4x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 6 * x - 4

/-- The point P -/
def P : ℝ × ℝ := (-1, 2)

/-- The point M -/
def M : ℝ × ℝ := (1, 1)

/-- The slope of the tangent line at M -/
def m : ℝ := f' M.1

theorem line_equation (x y : ℝ) :
  (2 * x - y + 4 = 0) ↔
  (y - P.2 = m * (x - P.1) ∧ 
   ∃ (t : ℝ), (x, y) = (t, f t) → y - f M.1 = m * (x - M.1)) := by
  sorry

end line_equation_l1972_197298


namespace consecutive_four_plus_one_is_square_l1972_197264

theorem consecutive_four_plus_one_is_square (a : ℕ) (h : a ≥ 1) :
  a * (a + 1) * (a + 2) * (a + 3) + 1 = (a^2 + 3*a + 1)^2 := by
  sorry

end consecutive_four_plus_one_is_square_l1972_197264


namespace square_greater_than_abs_l1972_197214

theorem square_greater_than_abs (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end square_greater_than_abs_l1972_197214


namespace total_money_l1972_197281

theorem total_money (a b : ℝ) (h1 : (4/15) * a = (2/5) * b) (h2 : b = 484) :
  a + b = 1210 := by
  sorry

end total_money_l1972_197281


namespace non_vegan_gluten_cupcakes_l1972_197213

/-- Given a set of cupcakes with specific properties, prove that the number of non-vegan cupcakes containing gluten is 28. -/
theorem non_vegan_gluten_cupcakes
  (total : ℕ)
  (gluten_free : ℕ)
  (vegan : ℕ)
  (vegan_gluten_free : ℕ)
  (h1 : total = 80)
  (h2 : gluten_free = total / 2)
  (h3 : vegan = 24)
  (h4 : vegan_gluten_free = vegan / 2)
  : total - gluten_free - (vegan - vegan_gluten_free) = 28 := by
  sorry

#check non_vegan_gluten_cupcakes

end non_vegan_gluten_cupcakes_l1972_197213


namespace count_ordered_pairs_l1972_197228

def prime_factorization : List (Nat × Nat) := [(2, 2), (3, 2), (7, 2)]

def n : Nat := 1764

theorem count_ordered_pairs : 
  (Finset.filter (fun p : Nat × Nat => p.1 * p.2 = n) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card = 27 := by
  sorry

end count_ordered_pairs_l1972_197228


namespace certain_number_equation_l1972_197285

theorem certain_number_equation : ∃ x : ℝ, 
  (3889 + x - 47.95000000000027 = 3854.002) ∧ 
  (x = 12.95200000000054) := by
  sorry

end certain_number_equation_l1972_197285


namespace inequality_relationship_l1972_197297

theorem inequality_relationship (a b : ℝ) : 
  (∀ x y : ℝ, x > y → x + 1 > y - 2) ∧ 
  (∃ x y : ℝ, x + 1 > y - 2 ∧ ¬(x > y)) :=
sorry

end inequality_relationship_l1972_197297


namespace thirty_sided_polygon_diagonals_l1972_197273

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem thirty_sided_polygon_diagonals :
  num_diagonals 30 = 405 := by sorry

end thirty_sided_polygon_diagonals_l1972_197273


namespace least_rectangle_area_for_two_squares_l1972_197255

theorem least_rectangle_area_for_two_squares :
  ∃ (A : ℝ), A = Real.sqrt 2 ∧
  (∀ (a b : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ a^2 + b^2 = 1 →
    ∃ (w h : ℝ), w ≥ 0 ∧ h ≥ 0 ∧ w * h = A ∧ a ≤ w ∧ b ≤ h) ∧
  (∀ (A' : ℝ), A' < A →
    ∃ (a b : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ a^2 + b^2 = 1 ∧
      ∀ (w h : ℝ), w ≥ 0 ∧ h ≥ 0 ∧ w * h = A' → (a > w ∨ b > h)) :=
by sorry

end least_rectangle_area_for_two_squares_l1972_197255


namespace function_property_implies_k_values_l1972_197278

-- Define the function type
def FunctionType := ℕ → ℤ

-- Define the property that the function must satisfy
def SatisfiesProperty (f : FunctionType) (k : ℤ) : Prop :=
  f 1995 = 1996 ∧
  ∀ x y : ℕ, f (x * y) = f x + f y + k * f (Nat.gcd x y)

-- Theorem statement
theorem function_property_implies_k_values :
  ∀ f : FunctionType, ∀ k : ℤ,
    SatisfiesProperty f k → (k = -1 ∨ k = 0) :=
sorry

end function_property_implies_k_values_l1972_197278


namespace complex_equation_solution_l1972_197231

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (a - Complex.I) * (1 + a * Complex.I) = -4 + 3 * Complex.I →
  a = -2 :=
by sorry

end complex_equation_solution_l1972_197231


namespace johns_fee_value_l1972_197280

/-- The one-time sitting fee for John's Photo World -/
def johns_fee : ℝ := sorry

/-- The price per sheet at John's Photo World -/
def johns_price_per_sheet : ℝ := 2.75

/-- The price per sheet at Sam's Picture Emporium -/
def sams_price_per_sheet : ℝ := 1.50

/-- The one-time sitting fee for Sam's Picture Emporium -/
def sams_fee : ℝ := 140

/-- The number of sheets being compared -/
def num_sheets : ℝ := 12

theorem johns_fee_value : johns_fee = 125 :=
  by
    have h : johns_price_per_sheet * num_sheets + johns_fee = sams_price_per_sheet * num_sheets + sams_fee :=
      sorry
    sorry

#check johns_fee_value

end johns_fee_value_l1972_197280


namespace special_circle_equation_l1972_197205

/-- A circle passing through the origin with center on the negative x-axis and radius 2 -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_negative_x_axis : center.1 < 0 ∧ center.2 = 0
  passes_through_origin : (center.1 ^ 2 + center.2 ^ 2) = radius ^ 2
  radius_is_two : radius = 2

/-- The equation of the special circle is (x+2)^2 + y^2 = 4 -/
theorem special_circle_equation (c : SpecialCircle) :
  ∀ (x y : ℝ), ((x + 2) ^ 2 + y ^ 2 = 4) ↔ 
  ((x - c.center.1) ^ 2 + (y - c.center.2) ^ 2 = c.radius ^ 2) :=
by sorry

end special_circle_equation_l1972_197205


namespace special_sequence_a10_l1972_197242

/-- A sequence with the property that for any p, q ∈ ℕ*, aₚ₊ₖ = aₚ · aₖ -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p * a q

theorem special_sequence_a10 (a : ℕ → ℕ) (h : SpecialSequence a) (h2 : a 2 = 4) :
  a 10 = 1024 := by
  sorry

end special_sequence_a10_l1972_197242


namespace even_digits_in_base7_of_528_l1972_197232

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Theorem: The number of even digits in the base-7 representation of 528₁₀ is 0 -/
theorem even_digits_in_base7_of_528 : 
  countEvenDigits (toBase7 528) = 0 := by
  sorry

end even_digits_in_base7_of_528_l1972_197232


namespace odot_calculation_l1972_197284

-- Define the ⊙ operation
def odot (a b : ℤ) : ℤ := a * b - (a + b)

-- State the theorem
theorem odot_calculation : odot 6 (odot 5 4) = 49 := by
  sorry

end odot_calculation_l1972_197284


namespace tank_capacity_l1972_197269

theorem tank_capacity : ∀ (x : ℚ), 
  (x / 8 + 90 = x / 2) → x = 240 := by
  sorry

end tank_capacity_l1972_197269


namespace division_remainder_proof_l1972_197252

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ)
  (h1 : dividend = 176)
  (h2 : divisor = 14)
  (h3 : quotient = 12)
  (h4 : dividend = divisor * quotient + remainder) :
  remainder = 8 := by
  sorry

end division_remainder_proof_l1972_197252


namespace largest_n_for_factorization_l1972_197229

/-- 
Theorem: The largest value of n for which 5x^2 + nx + 100 can be factored 
as the product of two linear factors with integer coefficients is 105.
-/
theorem largest_n_for_factorization : 
  (∃ (n : ℤ), ∀ (m : ℤ), 
    (∃ (a b : ℤ), ∀ (x : ℝ), 5 * x^2 + n * x + 100 = (5 * x + a) * (x + b)) ∧ 
    (∃ (a b : ℤ), ∀ (x : ℝ), 5 * x^2 + m * x + 100 = (5 * x + a) * (x + b) → m ≤ n)) ∧ 
  (∃ (a b : ℤ), ∀ (x : ℝ), 5 * x^2 + 105 * x + 100 = (5 * x + a) * (x + b)) :=
by sorry

#check largest_n_for_factorization

end largest_n_for_factorization_l1972_197229


namespace payment_difference_l1972_197224

/-- Represents the cost and distribution of a pizza -/
structure PizzaOrder where
  totalSlices : ℕ
  plainCost : ℚ
  mushroomCost : ℚ
  oliveCost : ℚ

/-- Calculates the total cost of the pizza -/
def totalCost (p : PizzaOrder) : ℚ :=
  p.plainCost + p.mushroomCost + p.oliveCost

/-- Calculates the cost per slice -/
def costPerSlice (p : PizzaOrder) : ℚ :=
  totalCost p / p.totalSlices

/-- Calculates the cost for Liam's portion -/
def liamCost (p : PizzaOrder) : ℚ :=
  costPerSlice p * (2 * p.totalSlices / 3 + 2)

/-- Calculates the cost for Emily's portion -/
def emilyCost (p : PizzaOrder) : ℚ :=
  costPerSlice p * 2

/-- The main theorem stating the difference in payment -/
theorem payment_difference (p : PizzaOrder) 
  (h1 : p.totalSlices = 12)
  (h2 : p.plainCost = 12)
  (h3 : p.mushroomCost = 3)
  (h4 : p.oliveCost = 4) :
  liamCost p - emilyCost p = 152 / 12 := by
  sorry

#eval (152 : ℚ) / 12  -- This should evaluate to 12.67

end payment_difference_l1972_197224


namespace cost_minimized_at_35_l1972_197263

/-- Represents the cost function for ordering hand sanitizers -/
def cost_function (x : ℝ) : ℝ := -2 * x^2 + 102 * x + 5000

/-- Represents the constraint on the number of boxes of type A sanitizer -/
def constraint (x : ℝ) : Prop := 15 ≤ x ∧ x ≤ 35

/-- Theorem stating that the cost function is minimized at x = 35 within the given constraints -/
theorem cost_minimized_at_35 :
  ∀ x : ℝ, constraint x → cost_function x ≥ cost_function 35 :=
sorry

end cost_minimized_at_35_l1972_197263


namespace purely_imaginary_complex_fraction_l1972_197299

theorem purely_imaginary_complex_fraction (a : ℝ) :
  let z : ℂ := (a + Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = 1 := by
  sorry

end purely_imaginary_complex_fraction_l1972_197299


namespace cone_volume_ratio_l1972_197276

/-- Two cones sharing a common base on a sphere -/
structure ConePair where
  R : ℝ  -- Radius of the sphere
  r : ℝ  -- Radius of the base of the cones
  h₁ : ℝ  -- Height of the first cone
  h₂ : ℝ  -- Height of the second cone

/-- The conditions of the problem -/
def ConePairConditions (cp : ConePair) : Prop :=
  cp.r^2 = 3 * cp.R^2 / 4 ∧  -- Area of base is 3/16 of sphere area
  cp.h₁ + cp.h₂ = 2 * cp.R ∧  -- Sum of heights equals diameter
  cp.r^2 + (cp.h₁ / 2)^2 = cp.R^2  -- Pythagorean theorem

/-- The theorem to be proved -/
theorem cone_volume_ratio (cp : ConePair) 
  (hc : ConePairConditions cp) : 
  cp.h₁ * cp.r^2 / (cp.h₂ * cp.r^2) = 1 / 3 :=
sorry

end cone_volume_ratio_l1972_197276


namespace greatest_q_minus_r_l1972_197275

theorem greatest_q_minus_r (q r : ℕ+) (h : 1001 = 17 * q + r) : 
  ∀ (q' r' : ℕ+), 1001 = 17 * q' + r' → q - r ≥ q' - r' := by
  sorry

end greatest_q_minus_r_l1972_197275


namespace euler_family_mean_age_l1972_197225

def euler_family_ages : List ℝ := [8, 8, 10, 10, 15, 12]

theorem euler_family_mean_age :
  (euler_family_ages.sum / euler_family_ages.length : ℝ) = 10.5 := by
  sorry

end euler_family_mean_age_l1972_197225


namespace pizza_toppings_l1972_197294

theorem pizza_toppings (total_slices ham_slices pineapple_slices : ℕ) 
  (h_total : total_slices = 15)
  (h_ham : ham_slices = 9)
  (h_pineapple : pineapple_slices = 12)
  (h_at_least_one : ∀ slice, slice ≤ total_slices → (slice ≤ ham_slices ∨ slice ≤ pineapple_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = ham_slices + pineapple_slices - total_slices ∧
    both_toppings = 6 := by
  sorry


end pizza_toppings_l1972_197294


namespace min_value_x_plus_2y_l1972_197250

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = x * y) :
  x + 2 * y ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_x_plus_2y_l1972_197250


namespace square_side_length_with_inscribed_circle_l1972_197260

theorem square_side_length_with_inscribed_circle (s : ℝ) : 
  (4 * s = π * (s / 2)^2) → s = 16 / π := by
  sorry

end square_side_length_with_inscribed_circle_l1972_197260


namespace relationship_equation_l1972_197238

theorem relationship_equation (x : ℝ) : 
  (2023 : ℝ) = (1/4 : ℝ) * x + 1 ↔ 
    (∃ A B : ℝ, A = 2023 ∧ B = x ∧ A = (1/4 : ℝ) * B + 1) :=
by sorry

end relationship_equation_l1972_197238


namespace complex_fraction_equals_962_l1972_197267

/-- Helper function to represent the factorization of x^4 + 400 --/
def factor (x : ℤ) : ℤ := (x * (x - 10) + 20) * (x * (x + 10) + 20)

/-- The main theorem stating that the given expression equals 962 --/
theorem complex_fraction_equals_962 : 
  (factor 10 * factor 26 * factor 42 * factor 58) / 
  (factor 2 * factor 18 * factor 34 * factor 50) = 962 := by
  sorry


end complex_fraction_equals_962_l1972_197267


namespace chord_length_l1972_197227

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the focus of the ellipse
def focus : ℝ × ℝ := (1, 0)

-- Define a chord through the focus and perpendicular to the major axis
def chord (y : ℝ) : Prop := ellipse 1 y

-- Theorem statement
theorem chord_length : 
  ∃ y₁ y₂ : ℝ, 
    chord y₁ ∧ 
    chord y₂ ∧ 
    y₁ ≠ y₂ ∧ 
    |y₁ - y₂| = 3 :=
sorry

end chord_length_l1972_197227


namespace no_exact_two_champions_l1972_197204

-- Define the tournament structure
structure Tournament where
  teams : Type
  plays : teams → teams → Prop
  beats : teams → teams → Prop

-- Define the superiority relation
def superior (t : Tournament) (a b : t.teams) : Prop :=
  t.beats a b ∨ ∃ c, t.beats a c ∧ t.beats c b

-- Define a champion
def is_champion (t : Tournament) (a : t.teams) : Prop :=
  ∀ b : t.teams, b ≠ a → superior t a b

-- Theorem statement
theorem no_exact_two_champions (t : Tournament) :
  ¬∃ (a b : t.teams), a ≠ b ∧
    is_champion t a ∧ is_champion t b ∧
    (∀ c : t.teams, is_champion t c → (c = a ∨ c = b)) :=
sorry

end no_exact_two_champions_l1972_197204


namespace job_completion_time_l1972_197233

/-- Represents the job completion scenario with changing number of workers -/
structure JobCompletion where
  initial_workers : ℕ
  initial_days : ℕ
  work_days_before_change : ℕ
  additional_workers : ℕ
  total_days : ℚ

/-- Theorem stating that under the given conditions, the job will be completed in 3.5 days -/
theorem job_completion_time (job : JobCompletion) :
  job.initial_workers = 6 ∧
  job.initial_days = 8 ∧
  job.work_days_before_change = 3 ∧
  job.additional_workers = 4 →
  job.total_days = 3.5 := by
  sorry

#check job_completion_time

end job_completion_time_l1972_197233


namespace sam_bought_one_lollipop_l1972_197266

/-- Calculates the number of lollipops Sam bought -/
def lollipops_bought (initial_dimes : ℕ) (initial_quarters : ℕ) (candy_bars : ℕ) 
  (dimes_per_candy : ℕ) (cents_per_lollipop : ℕ) (cents_left : ℕ) : ℕ :=
  let initial_cents := initial_dimes * 10 + initial_quarters * 25
  let candy_cost := candy_bars * dimes_per_candy * 10
  let cents_for_lollipops := initial_cents - candy_cost - cents_left
  cents_for_lollipops / cents_per_lollipop

theorem sam_bought_one_lollipop :
  lollipops_bought 19 6 4 3 25 195 = 1 := by
  sorry

end sam_bought_one_lollipop_l1972_197266


namespace first_stop_students_correct_l1972_197243

/-- The number of students who got on the bus at the first stop -/
def first_stop_students : ℕ := 39

/-- The number of students who got on the bus at the second stop -/
def second_stop_students : ℕ := 29

/-- The total number of students on the bus after the second stop -/
def total_students : ℕ := 68

/-- Theorem stating that the number of students who got on at the first stop is correct -/
theorem first_stop_students_correct :
  first_stop_students + second_stop_students = total_students := by
  sorry

end first_stop_students_correct_l1972_197243


namespace negation_of_universal_statement_l1972_197230

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - 3*x + 5 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^2 - 3*x₀ + 5 > 0) := by sorry

end negation_of_universal_statement_l1972_197230


namespace angle_calculation_l1972_197290

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℤ)
  (minutes : ℤ)

/-- Multiplication of an angle by an integer -/
def Angle.mul (a : Angle) (n : ℤ) : Angle :=
  ⟨a.degrees * n, a.minutes * n⟩

/-- Addition of two angles -/
def Angle.add (a b : Angle) : Angle :=
  ⟨a.degrees + b.degrees, a.minutes + b.minutes⟩

/-- Subtraction of two angles -/
def Angle.sub (a b : Angle) : Angle :=
  ⟨a.degrees - b.degrees, a.minutes - b.minutes⟩

/-- Normalize an angle by converting excess minutes to degrees -/
def Angle.normalize (a : Angle) : Angle :=
  let extraDegrees := a.minutes / 60
  let normalizedMinutes := a.minutes % 60
  ⟨a.degrees + extraDegrees, normalizedMinutes⟩

theorem angle_calculation :
  (Angle.normalize ((Angle.mul ⟨24, 31⟩ 4).sub ⟨62, 10⟩)) = ⟨35, 54⟩ := by
  sorry

end angle_calculation_l1972_197290


namespace probability_diamond_or_ace_l1972_197248

def standard_deck : ℕ := 52

def diamond_count : ℕ := 13

def ace_count : ℕ := 4

def favorable_outcomes : ℕ := diamond_count + ace_count - 1

theorem probability_diamond_or_ace :
  (favorable_outcomes : ℚ) / standard_deck = 4 / 13 :=
by sorry

end probability_diamond_or_ace_l1972_197248


namespace angela_beth_age_ratio_l1972_197246

theorem angela_beth_age_ratio :
  ∀ (angela_age beth_age : ℕ),
    (angela_age - 5 + beth_age - 5 = 45) →  -- Five years ago, sum of ages was 45
    (angela_age + 5 = 44) →                 -- In five years, Angela will be 44
    (angela_age : ℚ) / beth_age = 39 / 16   -- Ratio of current ages is 39:16
    :=
by
  sorry

end angela_beth_age_ratio_l1972_197246


namespace inverse_variation_problem_l1972_197292

/-- Given that x^2 and y vary inversely and are positive integers, 
    with y = 16 when x = 4, and z = x - y with z = 10 when y = 4, 
    prove that x = 1 when y = 256 -/
theorem inverse_variation_problem (x y z : ℕ+) (k : ℝ) : 
  (∀ (x y : ℕ+), (x:ℝ)^2 * y = k) →   -- x^2 and y vary inversely
  (4:ℝ)^2 * 16 = k →                  -- y = 16 when x = 4
  z = x - y →                         -- definition of z
  (∃ (x : ℕ+), z = 10 ∧ y = 4) →      -- z = 10 when y = 4
  (∃ (x : ℕ+), x = 1 ∧ y = 256) :=    -- to prove: x = 1 when y = 256
by sorry

end inverse_variation_problem_l1972_197292


namespace club_officer_selection_l1972_197253

/-- Represents the number of ways to choose officers in a club -/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  let president_vp_combinations := boys * girls * 2
  let secretary_choices := girls
  president_vp_combinations * secretary_choices

/-- Theorem stating the number of ways to choose officers under given conditions -/
theorem club_officer_selection :
  choose_officers 15 9 6 = 648 :=
by
  sorry


end club_officer_selection_l1972_197253


namespace ratio_problem_l1972_197283

theorem ratio_problem (N X : ℚ) (h1 : N / 2 = 150 / X) (h2 : N = 300) : X = 1 := by
  sorry

end ratio_problem_l1972_197283


namespace cost_of_melons_l1972_197277

/-- The cost of a single melon in dollars -/
def cost_per_melon : ℕ := 3

/-- The number of melons we want to calculate the cost for -/
def num_melons : ℕ := 6

/-- Theorem stating that the cost of 6 melons is $18 -/
theorem cost_of_melons : cost_per_melon * num_melons = 18 := by
  sorry

end cost_of_melons_l1972_197277


namespace geometric_sequence_sum_l1972_197262

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 1 + a 2 = 4/9 →
  a 3 + a 4 + a 5 + a 6 = 40 →
  (a 7 + a 8 + a 9) / 9 = 117 := by
  sorry

end geometric_sequence_sum_l1972_197262


namespace election_votes_theorem_l1972_197202

theorem election_votes_theorem (total_votes : ℕ) : 
  (∃ (winner_votes loser_votes : ℕ),
    winner_votes + loser_votes = total_votes ∧
    winner_votes = (70 * total_votes) / 100 ∧
    winner_votes - loser_votes = 360) →
  total_votes = 900 := by
sorry

end election_votes_theorem_l1972_197202


namespace rhombus_perimeter_l1972_197295

/-- A rhombus with a diagonal of length 6 and side length satisfying x^2 - 7x + 12 = 0 has a perimeter of 16 -/
theorem rhombus_perimeter (a b c d : ℝ) (h1 : a = b ∧ b = c ∧ c = d) 
  (h2 : ∃ (diag : ℝ), diag = 6) 
  (h3 : a^2 - 7*a + 12 = 0) : 
  a + b + c + d = 16 := by
  sorry

end rhombus_perimeter_l1972_197295


namespace inverse_variation_problem_l1972_197282

-- Define the inverse relationship between x and y
def inverse_relation (x y : ℝ) : Prop := ∃ k : ℝ, x * y^3 = k

-- Theorem statement
theorem inverse_variation_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : inverse_relation x₁ y₁)
  (h2 : inverse_relation x₂ y₂)
  (h3 : x₁ = 8)
  (h4 : y₁ = 1)
  (h5 : y₂ = 2) :
  x₂ = 1 := by
sorry

end inverse_variation_problem_l1972_197282


namespace min_distance_between_curves_l1972_197201

/-- The minimum distance between the curves y = e^(3x + 11) and y = (ln x - 11) / 3 -/
theorem min_distance_between_curves : ∃ d : ℝ, d > 0 ∧
  (∀ x y z : ℝ, y = Real.exp (3 * x + 11) ∧ z = (Real.log y - 11) / 3 →
    d ≤ Real.sqrt ((x - y)^2 + (y - z)^2)) ∧
  d = Real.sqrt 2 * (Real.log 3 + 12) / 3 :=
sorry

end min_distance_between_curves_l1972_197201


namespace hexagon_not_to_quadrilateral_other_polygons_to_quadrilateral_l1972_197220

-- Define a polygon type
inductive Polygon
| triangle : Polygon
| quadrilateral : Polygon
| pentagon : Polygon
| hexagon : Polygon

-- Define a function that represents cutting off one angle
def cutOffAngle (p : Polygon) : Polygon :=
  match p with
  | Polygon.triangle => Polygon.triangle  -- Assuming it remains a triangle
  | Polygon.quadrilateral => Polygon.triangle
  | Polygon.pentagon => Polygon.quadrilateral
  | Polygon.hexagon => Polygon.pentagon

-- Theorem stating that a hexagon cannot become a quadrilateral by cutting off one angle
theorem hexagon_not_to_quadrilateral :
  ∀ (p : Polygon), p = Polygon.hexagon → cutOffAngle p ≠ Polygon.quadrilateral :=
by sorry

-- Theorem stating that other polygons can potentially become a quadrilateral
theorem other_polygons_to_quadrilateral :
  ∃ (p : Polygon), p ≠ Polygon.hexagon ∧ (cutOffAngle p = Polygon.quadrilateral ∨ p = Polygon.quadrilateral) :=
by sorry

end hexagon_not_to_quadrilateral_other_polygons_to_quadrilateral_l1972_197220


namespace train_length_l1972_197209

/-- Proves that a train traveling at 45 km/hr crossing a 215-meter bridge in 30 seconds has a length of 160 meters -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 215 →
  crossing_time = 30 →
  train_speed * crossing_time - bridge_length = 160 := by
  sorry

end train_length_l1972_197209


namespace first_three_average_l1972_197271

theorem first_three_average (a b c d : ℝ) : 
  a = 33 →
  d = 18 →
  (b + c + d) / 3 = 15 →
  (a + b + c) / 3 = 20 := by
sorry

end first_three_average_l1972_197271


namespace vectors_orthogonal_l1972_197261

-- Define the vectors
def v1 : Fin 2 → ℝ := ![3, 7]
def v2 (x : ℝ) : Fin 2 → ℝ := ![x, -4]

-- Define orthogonality condition
def isOrthogonal (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

-- State the theorem
theorem vectors_orthogonal :
  isOrthogonal v1 (v2 (28/3)) := by sorry

end vectors_orthogonal_l1972_197261


namespace range_of_t_l1972_197239

-- Define the solution set of (a-1)^x > 1
def solution_set (a : ℝ) : Set ℝ := {x | x < 0}

-- Define the inequality q
def q (a t : ℝ) : Prop := a^2 - 2*t*a + t^2 - 1 < 0

-- Define the negation of p
def not_p (a : ℝ) : Prop := a ≤ 1 ∨ a ≥ 2

-- Define the negation of q
def not_q (a t : ℝ) : Prop := ¬(q a t)

-- Statement of the theorem
theorem range_of_t :
  (∀ a, solution_set a = {x | x < 0}) →
  (∀ a t, not_p a → not_q a t) →
  (∃ a t, not_p a ∧ q a t) →
  ∀ t, (∀ a, not_p a → not_q a t) → 1 ≤ t ∧ t ≤ 2 :=
sorry

end range_of_t_l1972_197239


namespace solve_cake_problem_l1972_197237

def cake_problem (cost_per_cake : ℕ) (john_payment : ℕ) : Prop :=
  ∃ (num_cakes : ℕ),
    cost_per_cake = 12 ∧
    john_payment = 18 ∧
    num_cakes * cost_per_cake = 2 * john_payment ∧
    num_cakes = 3

theorem solve_cake_problem :
  ∀ (cost_per_cake : ℕ) (john_payment : ℕ),
    cake_problem cost_per_cake john_payment :=
by
  sorry

end solve_cake_problem_l1972_197237


namespace inverse_sum_mod_thirteen_l1972_197259

theorem inverse_sum_mod_thirteen : 
  (((3⁻¹ : ZMod 13) + (4⁻¹ : ZMod 13) + (5⁻¹ : ZMod 13))⁻¹ : ZMod 13) = 1 := by
  sorry

end inverse_sum_mod_thirteen_l1972_197259


namespace root_implies_k_value_l1972_197206

theorem root_implies_k_value (k : ℝ) : 
  (2 * (4 : ℝ)^2 + 3 * 4 - k = 0) → k = 44 := by
  sorry

end root_implies_k_value_l1972_197206


namespace imaginary_part_of_z_l1972_197234

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (2 * i) / (1 - i)
  Complex.im z = 1 := by
  sorry

end imaginary_part_of_z_l1972_197234


namespace original_number_proof_l1972_197256

theorem original_number_proof (N : ℝ) (x y z : ℝ) : 
  (N * 1.2 = 480) →
  ((480 * 0.85) * x^2 = 5*x^3 + 24*x - 50) →
  ((N / y) * 0.75 = z) →
  (z = x * y) →
  N = 400 := by
sorry

end original_number_proof_l1972_197256


namespace max_value_a_l1972_197215

theorem max_value_a (a : ℝ) : 
  (∀ x k, x ∈ Set.Ioo 0 6 → k ∈ Set.Icc (-1) 1 → 
    6 * Real.log x + x^2 - 8*x + a ≤ k*x) → 
  a ≤ 6 - 6 * Real.log 6 :=
by sorry

end max_value_a_l1972_197215


namespace expression_equals_forty_times_ten_to_power_l1972_197287

theorem expression_equals_forty_times_ten_to_power : 
  (3^1506 + 7^1507)^2 - (3^1506 - 7^1507)^2 = 40 * 10^1506 := by
sorry

end expression_equals_forty_times_ten_to_power_l1972_197287


namespace power_zero_eq_one_l1972_197279

theorem power_zero_eq_one (n : ℤ) (h : n ≠ 0) : n^0 = 1 := by
  sorry

end power_zero_eq_one_l1972_197279


namespace distance_against_current_14km_l1972_197223

/-- Calculates the distance traveled against a current given swimming speed, current speed, and time. -/
def distanceAgainstCurrent (swimmingSpeed currentSpeed : ℝ) (time : ℝ) : ℝ :=
  (swimmingSpeed - currentSpeed) * time

/-- Proves that the distance traveled against the current is 14 km under given conditions. -/
theorem distance_against_current_14km
  (swimmingSpeed : ℝ)
  (currentSpeed : ℝ)
  (time : ℝ)
  (h1 : swimmingSpeed = 4)
  (h2 : currentSpeed = 2)
  (h3 : time = 7) :
  distanceAgainstCurrent swimmingSpeed currentSpeed time = 14 := by
  sorry

#eval distanceAgainstCurrent 4 2 7

end distance_against_current_14km_l1972_197223


namespace max_distance_C₁_intersections_l1972_197245

noncomputable section

-- Define the curves
def C₁ (t α : ℝ) : ℝ × ℝ := (t * Real.cos α, t * Real.sin α)

def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * Real.sqrt 3 * Real.sin θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

def C₃ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * Real.cos θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the set of valid parameters
def ValidParams : Set ℝ := {α | 0 ≤ α ∧ α ≤ Real.pi}

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem max_distance_C₁_intersections :
  ∃ (max_dist : ℝ), max_dist = 4 ∧
  ∀ (t₁ t₂ θ₁ θ₂ α : ℝ), α ∈ ValidParams →
    distance (C₁ t₁ α) (C₂ θ₁) = 0 →
    distance (C₁ t₂ α) (C₃ θ₂) = 0 →
    distance (C₁ t₁ α) (C₁ t₂ α) ≤ max_dist :=
sorry

end

end max_distance_C₁_intersections_l1972_197245


namespace percentage_difference_l1972_197289

theorem percentage_difference : 
  let sixty_percent_of_fifty : ℝ := (60 / 100) * 50
  let fifty_percent_of_thirty : ℝ := (50 / 100) * 30
  sixty_percent_of_fifty - fifty_percent_of_thirty = 15 := by
sorry

end percentage_difference_l1972_197289


namespace digit_sum_puzzle_l1972_197222

theorem digit_sum_puzzle (x y z w : ℕ) : 
  x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧ w ≤ 9 →
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
  x + z + x = 11 →
  y + z = 10 →
  x + w = 10 →
  x + y + z + w = 24 := by
sorry

end digit_sum_puzzle_l1972_197222


namespace sticks_per_matchbox_l1972_197257

/-- Given the following:
  * num_boxes: The number of boxes ordered
  * matchboxes_per_box: The number of matchboxes in each box
  * total_sticks: The total number of match sticks ordered

  Prove that the number of match sticks in each matchbox is 300.
-/
theorem sticks_per_matchbox
  (num_boxes : ℕ)
  (matchboxes_per_box : ℕ)
  (total_sticks : ℕ)
  (h1 : num_boxes = 4)
  (h2 : matchboxes_per_box = 20)
  (h3 : total_sticks = 24000) :
  total_sticks / (num_boxes * matchboxes_per_box) = 300 := by
  sorry

end sticks_per_matchbox_l1972_197257


namespace sum_product_equality_l1972_197247

theorem sum_product_equality (x y z : ℝ) (h : x + y + z = x * y * z) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z := by
  sorry

end sum_product_equality_l1972_197247


namespace door_height_is_eight_l1972_197235

/-- Represents the dimensions of a rectangular door and a pole satisfying specific conditions -/
structure DoorAndPole where
  pole_length : ℝ
  door_width : ℝ
  door_height : ℝ
  horizontal_condition : pole_length = door_width + 4
  vertical_condition : pole_length = door_height + 2
  diagonal_condition : pole_length^2 = door_width^2 + door_height^2

/-- Theorem stating that for any DoorAndPole structure, the door height is 8 -/
theorem door_height_is_eight (d : DoorAndPole) : d.door_height = 8 := by
  sorry

#check door_height_is_eight

end door_height_is_eight_l1972_197235


namespace first_sales_amount_l1972_197268

/-- Proves that the amount of the first sales is $10 million -/
theorem first_sales_amount (initial_royalties : ℝ) (subsequent_royalties : ℝ) 
  (subsequent_sales : ℝ) (royalty_rate_ratio : ℝ) :
  initial_royalties = 2 →
  subsequent_royalties = 8 →
  subsequent_sales = 100 →
  royalty_rate_ratio = 0.4 →
  ∃ (initial_sales : ℝ), initial_sales = 10 ∧ 
    (initial_royalties / initial_sales = subsequent_royalties / subsequent_sales / royalty_rate_ratio) :=
by sorry

end first_sales_amount_l1972_197268


namespace smallest_common_factor_l1972_197216

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m < 57 → Nat.gcd (15*m - 9) (11*m + 10) = 1) ∧ 
  Nat.gcd (15*57 - 9) (11*57 + 10) > 1 := by
  sorry

end smallest_common_factor_l1972_197216
