import Mathlib

namespace max_integer_k_no_real_roots_l2642_264288

theorem max_integer_k_no_real_roots (k : ℤ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k ≤ -2 ∧ ∀ m : ℤ, (∀ x : ℝ, x^2 - 2*x - m ≠ 0) → m ≤ k :=
by sorry

end max_integer_k_no_real_roots_l2642_264288


namespace common_chord_equation_l2642_264245

/-- Given two circles with equations x^2 + y^2 - 4x = 0 and x^2 + y^2 - 4y = 0,
    the equation of the line containing their common chord is x - y = 0 -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 - 4*x = 0) ∧ (x^2 + y^2 - 4*y = 0) → x - y = 0 := by
  sorry

end common_chord_equation_l2642_264245


namespace all_numbers_in_first_hundred_l2642_264285

/-- Represents the color of a number in the sequence -/
inductive Color
| Blue
| Red

/-- Represents the sequence of 200 numbers with their colors -/
def Sequence := Fin 200 → (ℕ × Color)

/-- The blue numbers form a sequence from 1 to 100 in ascending order -/
def blue_ascending (s : Sequence) : Prop :=
  ∀ i j : Fin 200, i < j →
    (s i).2 = Color.Blue → (s j).2 = Color.Blue →
      (s i).1 < (s j).1

/-- The red numbers form a sequence from 100 to 1 in descending order -/
def red_descending (s : Sequence) : Prop :=
  ∀ i j : Fin 200, i < j →
    (s i).2 = Color.Red → (s j).2 = Color.Red →
      (s i).1 > (s j).1

/-- The blue numbers are all natural numbers from 1 to 100 -/
def blue_range (s : Sequence) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 →
    ∃ i : Fin 200, (s i).1 = n ∧ (s i).2 = Color.Blue

/-- The red numbers are all natural numbers from 1 to 100 -/
def red_range (s : Sequence) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 →
    ∃ i : Fin 200, (s i).1 = n ∧ (s i).2 = Color.Red

theorem all_numbers_in_first_hundred (s : Sequence)
  (h1 : blue_ascending s)
  (h2 : red_descending s)
  (h3 : blue_range s)
  (h4 : red_range s) :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 →
    ∃ i : Fin 100, (s i).1 = n :=
by sorry

end all_numbers_in_first_hundred_l2642_264285


namespace function_difference_bound_l2642_264258

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - x * Real.log a

theorem function_difference_bound
  (a : ℝ) (ha : 1 < a ∧ a ≤ Real.exp 1) :
  ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 →
  |f a x₁ - f a x₂| ≤ Real.exp 1 - 2 :=
sorry

end function_difference_bound_l2642_264258


namespace quadratic_minimum_l2642_264280

theorem quadratic_minimum (b : ℝ) :
  let f : ℝ → ℝ := fun x ↦ 3 * x^2 - 18 * x + b
  ∃ (min_value : ℝ), (∀ x, f x ≥ min_value) ∧ (f 3 = min_value) ∧ (min_value = b - 27) := by
  sorry

end quadratic_minimum_l2642_264280


namespace concentric_circles_properties_l2642_264213

/-- Given two concentric circles where a chord is tangent to the smaller circle -/
structure ConcentricCircles where
  /-- Radius of the smaller circle -/
  r₁ : ℝ
  /-- Length of the chord tangent to the smaller circle -/
  chord_length : ℝ
  /-- The chord is tangent to the smaller circle -/
  tangent_chord : True

/-- Theorem about the radius of the larger circle and the area between the circles -/
theorem concentric_circles_properties (c : ConcentricCircles) 
  (h₁ : c.r₁ = 30)
  (h₂ : c.chord_length = 120) :
  ∃ (r₂ : ℝ) (area : ℝ),
    r₂ = 30 * Real.sqrt 5 ∧ 
    area = 3600 * Real.pi ∧
    r₂ > c.r₁ ∧
    area = Real.pi * (r₂^2 - c.r₁^2) := by
  sorry

end concentric_circles_properties_l2642_264213


namespace pyramid_missing_number_l2642_264242

/-- Represents a row in the pyramid -/
structure PyramidRow :=
  (left : ℚ) (middle : ℚ) (right : ℚ)

/-- Represents the pyramid structure -/
structure Pyramid :=
  (top_row : PyramidRow)
  (middle_row : PyramidRow)
  (bottom_row : PyramidRow)

/-- Checks if the pyramid satisfies the product rule -/
def is_valid_pyramid (p : Pyramid) : Prop :=
  p.middle_row.left = p.top_row.left * p.top_row.middle ∧
  p.middle_row.middle = p.top_row.middle * p.top_row.right ∧
  p.bottom_row.left = p.middle_row.left * p.middle_row.middle ∧
  p.bottom_row.middle = p.middle_row.middle * p.middle_row.right

/-- The main theorem -/
theorem pyramid_missing_number :
  ∀ (p : Pyramid),
    is_valid_pyramid p →
    p.middle_row = ⟨3, 2, 5⟩ →
    p.bottom_row.left = 10 →
    p.top_row.middle = 5/3 :=
by sorry

end pyramid_missing_number_l2642_264242


namespace tangent_line_equation_l2642_264265

/-- Given a function f(x) = x³ + ax² + bx + 1 with f'(1) = 2a and f'(2) = -b,
    where a and b are real constants, the equation of the tangent line
    to y = f(x) at (1, f(1)) is 6x + 2y - 1 = 0. -/
theorem tangent_line_equation (a b : ℝ) :
  let f (x : ℝ) := x^3 + a*x^2 + b*x + 1
  let f' (x : ℝ) := 3*x^2 + 2*a*x + b
  f' 1 = 2*a →
  f' 2 = -b →
  ∃ (m c : ℝ), m = -3 ∧ c = -1/2 ∧ ∀ x y, y - f 1 = m * (x - 1) ↔ 6*x + 2*y - 1 = 0 :=
by sorry

end tangent_line_equation_l2642_264265


namespace total_oranges_proof_l2642_264236

def initial_purchase : ℕ := 10
def additional_purchase : ℕ := 5
def weeks : ℕ := 3

def total_oranges : ℕ :=
  let week1_purchase := initial_purchase + additional_purchase
  let subsequent_weeks_purchase := 2 * week1_purchase
  week1_purchase + (weeks - 1) * subsequent_weeks_purchase

theorem total_oranges_proof :
  total_oranges = 75 :=
by sorry

end total_oranges_proof_l2642_264236


namespace light_path_in_cube_l2642_264272

/-- Represents a point on the face of a cube -/
structure FacePoint where
  x : ℝ
  y : ℝ

/-- Represents the path of a light beam in a cube -/
def LightPath (cube_side : ℝ) (p : FacePoint) : ℝ := sorry

theorem light_path_in_cube (cube_side : ℝ) (p : FacePoint) 
  (h_side : cube_side = 10)
  (h_p : p = ⟨3, 4⟩) :
  ∃ (r s : ℕ), LightPath cube_side p = r * Real.sqrt s ∧ r + s = 55 ∧ 
  ∀ (prime : ℕ), Nat.Prime prime → ¬(s.gcd (prime ^ 2) > 1) := by
  sorry

end light_path_in_cube_l2642_264272


namespace sqrt_500_simplification_l2642_264243

theorem sqrt_500_simplification : Real.sqrt 500 = 10 * Real.sqrt 5 := by sorry

end sqrt_500_simplification_l2642_264243


namespace real_number_classification_l2642_264218

theorem real_number_classification : 
  ∀ x : ℝ, x < 0 ∨ x ≥ 0 := by sorry

end real_number_classification_l2642_264218


namespace intersection_of_A_and_B_l2642_264279

def set_A : Set ℝ := {x | |x - 2| ≤ 1}
def set_B : Set ℝ := {x | (x - 5) / (2 - x) > 0}

theorem intersection_of_A_and_B :
  ∀ x : ℝ, x ∈ set_A ∩ set_B ↔ 2 < x ∧ x ≤ 3 := by sorry

end intersection_of_A_and_B_l2642_264279


namespace g_composition_of_3_l2642_264298

def g (x : ℤ) : ℤ :=
  if x % 3 = 0 then x / 3 else 4 * x - 1

theorem g_composition_of_3 : g (g (g (g 3))) = 3 := by
  sorry

end g_composition_of_3_l2642_264298


namespace max_cables_theorem_l2642_264244

/-- Represents the number of employees using each brand of computer -/
structure EmployeeCount where
  total : Nat
  brandX : Nat
  brandY : Nat

/-- Represents the constraints on connections -/
structure ConnectionConstraints where
  maxConnectionPercentage : Real

/-- Calculates the maximum number of cables that can be installed -/
def maxCables (employees : EmployeeCount) (constraints : ConnectionConstraints) : Nat :=
  sorry

/-- Theorem stating the maximum number of cables that can be installed -/
theorem max_cables_theorem (employees : EmployeeCount) (constraints : ConnectionConstraints) :
  employees.total = 50 →
  employees.brandX = 30 →
  employees.brandY = 20 →
  constraints.maxConnectionPercentage = 0.95 →
  maxCables employees constraints = 300 :=
  sorry

end max_cables_theorem_l2642_264244


namespace compound_molecular_weight_l2642_264277

/-- The atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Nitrogen atoms in the compound -/
def num_N : ℕ := 2

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := num_N * atomic_weight_N + num_O * atomic_weight_O

theorem compound_molecular_weight : molecular_weight = 76.02 := by
  sorry

end compound_molecular_weight_l2642_264277


namespace line_not_in_third_quadrant_l2642_264230

/-- Given a line ax + by + c = 0 where ac < 0 and bc < 0, prove that the line does not pass through the third quadrant. -/
theorem line_not_in_third_quadrant (a b c : ℝ) (h1 : a * c < 0) (h2 : b * c < 0) :
  ¬∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0 := by
  sorry

end line_not_in_third_quadrant_l2642_264230


namespace household_survey_total_l2642_264224

theorem household_survey_total (total : ℕ) (neither : ℕ) (only_a : ℕ) (both : ℕ) : 
  total = 180 ∧ 
  neither = 80 ∧ 
  only_a = 60 ∧ 
  both = 10 ∧ 
  (∃ (only_b : ℕ), only_b = 3 * both) →
  total = neither + only_a + both + (3 * both) :=
by sorry

end household_survey_total_l2642_264224


namespace sqrt_difference_equality_l2642_264223

theorem sqrt_difference_equality : 
  Real.sqrt (49 + 81) - Real.sqrt (36 - 25) = Real.sqrt 130 - Real.sqrt 11 := by
  sorry

end sqrt_difference_equality_l2642_264223


namespace rotated_point_coordinates_l2642_264266

def triangle_OAB (A : ℝ × ℝ) : Prop :=
  A.1 ≥ 0 ∧ A.2 > 0

def angle_ABO_is_right (A : ℝ × ℝ) : Prop :=
  (A.2 / A.1) * 10 = A.2

def angle_AOB_is_30 (A : ℝ × ℝ) : Prop :=
  A.2 / A.1 = Real.sqrt 3 / 3

def rotate_90_ccw (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

theorem rotated_point_coordinates (A : ℝ × ℝ) :
  triangle_OAB A →
  angle_ABO_is_right A →
  angle_AOB_is_30 A →
  rotate_90_ccw A = (-10 * Real.sqrt 3 / 3, 10) := by
  sorry

end rotated_point_coordinates_l2642_264266


namespace binomial_sum_equals_sixteen_l2642_264283

theorem binomial_sum_equals_sixteen (h : (Complex.exp (Complex.I * Real.pi / 4))^10 = Complex.I) :
  Nat.choose 10 1 - Nat.choose 10 3 + (Nat.choose 10 5 / 2) = 2^4 := by
  sorry

end binomial_sum_equals_sixteen_l2642_264283


namespace hyperbola_equation_l2642_264228

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  focal_distance : ℝ
  asymptote_slope : ℝ

-- Define the standard equation of the hyperbola
def standard_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, (y^2 / (8/5)) - (x^2 / (72/5)) = 1

-- Theorem statement
theorem hyperbola_equation (h : Hyperbola) 
  (h_foci : h.focal_distance = 8)
  (h_asymptote : h.asymptote_slope = 1/3) :
  standard_equation h :=
sorry

end hyperbola_equation_l2642_264228


namespace average_marks_proof_l2642_264267

/-- Given the marks for three subjects, proves that the average is 65 -/
theorem average_marks_proof (physics chemistry maths : ℝ) : 
  physics = 125 → 
  (physics + maths) / 2 = 90 → 
  (physics + chemistry) / 2 = 70 → 
  (physics + chemistry + maths) / 3 = 65 := by
sorry


end average_marks_proof_l2642_264267


namespace sector_max_area_angle_l2642_264263

/-- Given a sector with circumference 20, the radian measure of its central angle is 2 when the area of the sector is maximized. -/
theorem sector_max_area_angle (r : ℝ) (θ : ℝ) : 
  0 < r ∧ r < 10 →  -- Ensure r is in the valid range
  r * θ + 2 * r = 20 →  -- Circumference of sector is 20
  (∀ r' θ' : ℝ, 0 < r' ∧ r' < 10 → r' * θ' + 2 * r' = 20 → 
    r * θ / 2 ≥ r' * θ' / 2) →  -- Area is maximized
  θ = 2 := by
sorry

end sector_max_area_angle_l2642_264263


namespace cube_tetrahedrons_l2642_264249

/-- A cube has 8 vertices -/
def cube_vertices : ℕ := 8

/-- Number of vertices needed to form a tetrahedron -/
def tetrahedron_vertices : ℕ := 4

/-- Number of coplanar sets in a cube (faces + diagonal planes) -/
def coplanar_sets : ℕ := 12

/-- The number of different tetrahedrons that can be formed from the vertices of a cube -/
def num_tetrahedrons : ℕ := Nat.choose cube_vertices tetrahedron_vertices - coplanar_sets

theorem cube_tetrahedrons :
  num_tetrahedrons = Nat.choose cube_vertices tetrahedron_vertices - coplanar_sets :=
sorry

end cube_tetrahedrons_l2642_264249


namespace negation_of_proposition_l2642_264221

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), a = 1 → a + b = 1) ↔ (∃ (a b : ℝ), a = 1 ∧ a + b ≠ 1) :=
by sorry

end negation_of_proposition_l2642_264221


namespace rounding_inequality_l2642_264264

/-- The number of digits in a natural number -/
def num_digits (k : ℕ) : ℕ := sorry

/-- Rounds a natural number to the nearest power of 10 -/
def round_to_power_of_10 (k : ℕ) (power : ℕ) : ℕ := sorry

/-- Applies n-1 roundings to the nearest power of 10 -/
def apply_n_minus_1_roundings (k : ℕ) : ℕ := sorry

theorem rounding_inequality (k : ℕ) (h1 : k = 10 * 106) :
  let n := num_digits k
  let k_bar := apply_n_minus_1_roundings k
  k_bar < (18 : ℚ) / 13 * k := by sorry

end rounding_inequality_l2642_264264


namespace remainder_scaling_l2642_264251

theorem remainder_scaling (a b : ℕ) (c r : ℕ) (h : a = b * c + r) (hr : r = 7) :
  ∃ (c' : ℕ), 10 * a = 10 * b * c' + 70 :=
sorry

end remainder_scaling_l2642_264251


namespace unique_positive_solution_l2642_264293

theorem unique_positive_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∃! x : ℝ, x > 0 ∧ 
    (2 * (a + b) * x + 2 * a * b) / (4 * x + a + b) = ((a^(1/3) + b^(1/3)) / 2)^3 := by
  sorry

end unique_positive_solution_l2642_264293


namespace parabola_line_intersection_theorem_l2642_264204

-- Define the parabola C: y^2 = 3x
def C (x y : ℝ) : Prop := y^2 = 3*x

-- Define the line l: y = (3/2)x + b
def l (x y b : ℝ) : Prop := y = (3/2)*x + b

-- Define the intersection points E and F
def E (x y : ℝ) : Prop := C x y ∧ ∃ b, l x y b
def F (x y : ℝ) : Prop := C x y ∧ ∃ b, l x y b

-- Define point H on x-axis
def H (x : ℝ) : Prop := ∃ b, l x 0 b

-- Define the vector relationship
def vector_relationship (e_x e_y f_x f_y h_x k : ℝ) : Prop :=
  (h_x - e_x, -e_y) = k • (f_x - h_x, f_y)

-- Theorem statement
theorem parabola_line_intersection_theorem 
  (e_x e_y f_x f_y h_x : ℝ) (k : ℝ) :
  C e_x e_y → C f_x f_y →
  (∃ b, l e_x e_y b ∧ l f_x f_y b) →
  H h_x →
  vector_relationship e_x e_y f_x f_y h_x k →
  k > 1 →
  (f_x - e_x)^2 + (f_y - e_y)^2 = (4*Real.sqrt 13 / 3)^2 →
  k = 3 := by
  sorry

end parabola_line_intersection_theorem_l2642_264204


namespace median_equation_equal_intercepts_equation_l2642_264269

-- Define the vertices of triangle ABC
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (4, -6)
def C : ℝ × ℝ := (5, 1)

-- Define the equation of a line
def is_line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Theorem for the median equation
theorem median_equation :
  is_line_equation 1 (-2) (-3) (A.1 + (B.1 - A.1)/2) (A.2 + (B.2 - A.2)/2) ∧
  is_line_equation 1 (-2) (-3) C.1 C.2 :=
sorry

-- Theorem for the line with equal intercepts
theorem equal_intercepts_equation :
  is_line_equation 1 1 (-2) A.1 A.2 ∧
  ∃ (t : ℝ), is_line_equation 1 1 (-2) t 0 ∧ is_line_equation 1 1 (-2) 0 t :=
sorry

end median_equation_equal_intercepts_equation_l2642_264269


namespace fraction_equality_implies_numerator_equality_l2642_264241

theorem fraction_equality_implies_numerator_equality 
  (a b c : ℝ) (h : c ≠ 0) : a / c = b / c → a = b := by
  sorry

end fraction_equality_implies_numerator_equality_l2642_264241


namespace imaginary_unit_sum_l2642_264206

theorem imaginary_unit_sum (i : ℂ) : i^2 = -1 → i + i^2 + i^3 = -1 := by
  sorry

end imaginary_unit_sum_l2642_264206


namespace log_equation_solution_l2642_264215

theorem log_equation_solution (t : ℝ) (h : t > 0) :
  4 * (Real.log t / Real.log 3) = Real.log (4 * t) / Real.log 3 → t = (4 : ℝ) ^ (1 / 3) :=
by
  sorry

end log_equation_solution_l2642_264215


namespace negation_equivalence_l2642_264294

theorem negation_equivalence :
  (¬ ∀ x : ℝ, ∃ n : ℕ+, (n : ℝ) ≥ x^2) ↔ (∃ x : ℝ, ∀ n : ℕ+, (n : ℝ) < x^2) :=
by sorry

end negation_equivalence_l2642_264294


namespace equal_sum_sequence_definition_l2642_264222

/-- Definition of an equal sum sequence -/
def is_equal_sum_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = a (n + 1) + a (n + 2)

/-- Theorem stating the definition of an equal sum sequence -/
theorem equal_sum_sequence_definition (a : ℕ → ℝ) :
  is_equal_sum_sequence a ↔
    ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = a (n + 1) + a (n + 2) :=
by sorry

end equal_sum_sequence_definition_l2642_264222


namespace second_year_undeclared_fraction_l2642_264239

theorem second_year_undeclared_fraction :
  let total : ℚ := 1
  let first_year : ℚ := 1/4
  let second_year : ℚ := 1/2
  let third_year : ℚ := 1/6
  let fourth_year : ℚ := 1/12
  let first_year_undeclared : ℚ := 4/5
  let second_year_undeclared : ℚ := 3/4
  let third_year_undeclared : ℚ := 1/3
  let fourth_year_undeclared : ℚ := 1/6
  
  first_year + second_year + third_year + fourth_year = total →
  second_year * second_year_undeclared = 1/3
  := by sorry

end second_year_undeclared_fraction_l2642_264239


namespace sum_of_three_numbers_l2642_264282

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241) 
  (h2 : a*b + b*c + c*a = 100) : 
  a + b + c = 21 := by
sorry

end sum_of_three_numbers_l2642_264282


namespace slices_left_over_l2642_264233

-- Define the number of slices for each pizza size
def small_pizza_slices : ℕ := 4
def large_pizza_slices : ℕ := 8

-- Define the number of pizzas purchased
def small_pizzas_bought : ℕ := 3
def large_pizzas_bought : ℕ := 2

-- Define the number of slices each person eats
def george_slices : ℕ := 3
def bob_slices : ℕ := george_slices + 1
def susie_slices : ℕ := bob_slices / 2
def bill_slices : ℕ := 3
def fred_slices : ℕ := 3
def mark_slices : ℕ := 3

-- Calculate total slices and slices eaten
def total_slices : ℕ := small_pizza_slices * small_pizzas_bought + large_pizza_slices * large_pizzas_bought
def total_slices_eaten : ℕ := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

-- Theorem to prove
theorem slices_left_over : total_slices - total_slices_eaten = 10 := by
  sorry

end slices_left_over_l2642_264233


namespace weighted_average_markup_percentage_l2642_264200

-- Define the fruit types
inductive Fruit
| Apple
| Orange
| Banana

-- Define the properties for each fruit
def cost (f : Fruit) : ℝ :=
  match f with
  | Fruit.Apple => 30
  | Fruit.Orange => 40
  | Fruit.Banana => 50

def markup_percentage (f : Fruit) : ℝ :=
  match f with
  | Fruit.Apple => 0.10
  | Fruit.Orange => 0.15
  | Fruit.Banana => 0.20

def quantity (f : Fruit) : ℕ :=
  match f with
  | Fruit.Apple => 25
  | Fruit.Orange => 20
  | Fruit.Banana => 15

-- Calculate the markup amount for a fruit
def markup_amount (f : Fruit) : ℝ :=
  cost f * markup_percentage f

-- Calculate the selling price for a fruit
def selling_price (f : Fruit) : ℝ :=
  cost f + markup_amount f

-- Calculate the total selling price for all fruits
def total_selling_price : ℝ :=
  selling_price Fruit.Apple + selling_price Fruit.Orange + selling_price Fruit.Banana

-- Calculate the total cost for all fruits
def total_cost : ℝ :=
  cost Fruit.Apple + cost Fruit.Orange + cost Fruit.Banana

-- Calculate the total markup for all fruits
def total_markup : ℝ :=
  markup_amount Fruit.Apple + markup_amount Fruit.Orange + markup_amount Fruit.Banana

-- Theorem: The weighted average markup percentage is 15.83%
theorem weighted_average_markup_percentage :
  (total_markup / total_cost) * 100 = 15.83 := by
  sorry

end weighted_average_markup_percentage_l2642_264200


namespace unique_solution_fourth_power_equation_l2642_264255

theorem unique_solution_fourth_power_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (4 * x)^5 = (8 * x)^4 ∧ x = 4 := by
  sorry

end unique_solution_fourth_power_equation_l2642_264255


namespace x_interval_equivalence_l2642_264231

theorem x_interval_equivalence (x : ℝ) : 
  (2/3 < x ∧ x < 3/4) ↔ (2 < 3*x ∧ 3*x < 3) ∧ (2 < 4*x ∧ 4*x < 3) := by
sorry

end x_interval_equivalence_l2642_264231


namespace greatest_of_three_consecutive_integers_l2642_264262

theorem greatest_of_three_consecutive_integers (x : ℤ) :
  (x + (x + 1) + (x + 2) = 18) → (max x (max (x + 1) (x + 2)) = 7) :=
by sorry

end greatest_of_three_consecutive_integers_l2642_264262


namespace cube_surface_area_l2642_264296

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the squared distance between two points -/
def squaredDistance (p q : Point3D) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2

/-- The vertices of the cube -/
def A : Point3D := ⟨5, 7, 15⟩
def B : Point3D := ⟨6, 3, 6⟩
def C : Point3D := ⟨9, -2, 14⟩

/-- Theorem: The surface area of the cube with vertices A, B, and C is 294 -/
theorem cube_surface_area : ∃ (side : ℝ), 
  squaredDistance A B = 2 * side^2 ∧ 
  squaredDistance A C = 2 * side^2 ∧ 
  squaredDistance B C = 2 * side^2 ∧ 
  6 * side^2 = 294 := by
  sorry


end cube_surface_area_l2642_264296


namespace unique_paths_in_grid_l2642_264253

/-- The number of rows in the grid -/
def rows : ℕ := 6

/-- The number of columns in the grid -/
def cols : ℕ := 6

/-- The number of moves required to reach the bottom-right corner -/
def moves : ℕ := 5

/-- A function that calculates the number of unique paths in the grid -/
def uniquePaths (r : ℕ) (c : ℕ) (m : ℕ) : ℕ := 2^m

/-- Theorem stating that the number of unique paths in the given grid is 32 -/
theorem unique_paths_in_grid : uniquePaths rows cols moves = 32 := by
  sorry

end unique_paths_in_grid_l2642_264253


namespace inscribed_polygon_existence_l2642_264247

/-- Represents a line in a plane --/
structure Line where
  -- Add necessary fields for a line

/-- Represents a circle in a plane --/
structure Circle where
  -- Add necessary fields for a circle

/-- Represents a polygon --/
structure Polygon where
  -- Add necessary fields for a polygon

/-- Function to check if two lines are parallel --/
def are_parallel (l1 l2 : Line) : Prop :=
  sorry

/-- Function to check if a polygon is inscribed in a circle --/
def is_inscribed (p : Polygon) (c : Circle) : Prop :=
  sorry

/-- Function to check if a polygon side is parallel to a given line --/
def side_parallel_to_line (p : Polygon) (l : Line) : Prop :=
  sorry

/-- Main theorem statement --/
theorem inscribed_polygon_existence 
  (c : Circle) (lines : List Line) (n : Nat) 
  (h1 : lines.length = n)
  (h2 : ∀ (l1 l2 : Line), l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → ¬(are_parallel l1 l2)) :
  (n % 2 = 0 → 
    (∃ (p : Polygon), is_inscribed p c ∧ 
      (∀ (side : Line), side_parallel_to_line p side → side ∈ lines)) ∨
    (∀ (p : Polygon), ¬(is_inscribed p c ∧ 
      (∀ (side : Line), side_parallel_to_line p side → side ∈ lines)))) ∧
  (n % 2 = 1 → 
    ∃! (p1 p2 : Polygon), p1 ≠ p2 ∧ 
      is_inscribed p1 c ∧ is_inscribed p2 c ∧
      (∀ (side : Line), side_parallel_to_line p1 side → side ∈ lines) ∧
      (∀ (side : Line), side_parallel_to_line p2 side → side ∈ lines)) :=
sorry

end inscribed_polygon_existence_l2642_264247


namespace simplest_fraction_of_0375_l2642_264208

theorem simplest_fraction_of_0375 (c d : ℕ+) : 
  (c : ℚ) / (d : ℚ) = 0.375 ∧ 
  (∀ (m n : ℕ+), (m : ℚ) / (n : ℚ) = 0.375 → c ≤ m ∧ d ≤ n) →
  c + d = 11 := by
  sorry

end simplest_fraction_of_0375_l2642_264208


namespace intersection_point_l2642_264214

/-- The first curve -/
def curve1 (x : ℝ) : ℝ := 4 * x^2 + 3 * x - 2

/-- The second curve -/
def curve2 (x : ℝ) : ℝ := 2 * x^3 + x^2 + 7

/-- Theorem stating that (-1, -1) is the only intersection point of the two curves -/
theorem intersection_point : 
  ∃! p : ℝ × ℝ, p.1 = -1 ∧ p.2 = -1 ∧ curve1 p.1 = curve2 p.1 := by
  sorry

end intersection_point_l2642_264214


namespace equations_equivalence_l2642_264299

-- Define the equations
def equation1 (x : ℝ) : Prop := (-x - 2) / (x - 3) = (x + 1) / (x - 3)
def equation2 (x : ℝ) : Prop := -x - 2 = x + 1
def equation3 (x : ℝ) : Prop := (-x - 2) * (x - 3) = (x + 1) * (x - 3)

-- Theorem statement
theorem equations_equivalence :
  (∀ x : ℝ, x ≠ 3 → (equation1 x ↔ equation2 x)) ∧
  (¬ ∀ x : ℝ, equation2 x ↔ equation3 x) ∧
  (¬ ∀ x : ℝ, equation1 x ↔ equation3 x) :=
sorry

end equations_equivalence_l2642_264299


namespace f_neg_two_value_l2642_264268

def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + x^2

theorem f_neg_two_value (f : ℝ → ℝ) :
  (∀ x, F f x = -F f (-x)) →  -- F is an odd function
  f 2 = 1 →                   -- f(2) = 1
  f (-2) = -9 :=               -- Conclusion: f(-2) = -9
by
  sorry

end f_neg_two_value_l2642_264268


namespace paintball_cost_per_box_l2642_264261

/-- Calculates the cost per box of paintballs -/
def cost_per_box (plays_per_month : ℕ) (boxes_per_play : ℕ) (total_monthly_cost : ℕ) : ℚ :=
  total_monthly_cost / (plays_per_month * boxes_per_play)

/-- Theorem: Given the problem conditions, the cost per box of paintballs is $25 -/
theorem paintball_cost_per_box :
  let plays_per_month : ℕ := 3
  let boxes_per_play : ℕ := 3
  let total_monthly_cost : ℕ := 225
  cost_per_box plays_per_month boxes_per_play total_monthly_cost = 25 := by
  sorry


end paintball_cost_per_box_l2642_264261


namespace percentage_calculation_l2642_264205

theorem percentage_calculation (N : ℝ) (P : ℝ) : 
  N = 70 → (P / 100) * N - 10 = 25 → P = 50 := by
  sorry

end percentage_calculation_l2642_264205


namespace sin_eq_tan_sin_unique_solution_l2642_264237

theorem sin_eq_tan_sin_unique_solution :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ Real.arcsin (1/2) ∧ Real.sin x = Real.tan (Real.sin x) :=
by sorry

end sin_eq_tan_sin_unique_solution_l2642_264237


namespace smallest_n_with_conditions_l2642_264270

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem smallest_n_with_conditions : 
  ∀ n : ℕ, n > 0 → n % 3 = 0 → digit_product n = 882 → n ≥ 13677 := by
  sorry

#check smallest_n_with_conditions

end smallest_n_with_conditions_l2642_264270


namespace shaded_area_regular_octagon_l2642_264219

/-- The area of the shaded region in a regular octagon with side length 12 cm, 
    formed by connecting every other vertex (creating two squares) -/
theorem shaded_area_regular_octagon (side_length : ℝ) (h : side_length = 12) : 
  let octagon_area := 8 * (1/2 * side_length * (side_length / 2))
  octagon_area = 288 := by
  sorry

end shaded_area_regular_octagon_l2642_264219


namespace bicycle_profit_percentage_l2642_264250

theorem bicycle_profit_percentage 
  (final_price : ℝ) 
  (initial_cost : ℝ) 
  (intermediate_profit_percentage : ℝ) 
  (h1 : final_price = 225)
  (h2 : initial_cost = 112.5)
  (h3 : intermediate_profit_percentage = 25) :
  let intermediate_cost := final_price / (1 + intermediate_profit_percentage / 100)
  let initial_profit_percentage := (intermediate_cost - initial_cost) / initial_cost * 100
  initial_profit_percentage = 60 := by
sorry

end bicycle_profit_percentage_l2642_264250


namespace max_queens_on_chessboard_l2642_264260

/-- Represents a position on the chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Checks if two positions are on the same diagonal -/
def on_same_diagonal (p1 p2 : Position) : Prop :=
  (p1.row.val : Int) - (p1.col.val : Int) = (p2.row.val : Int) - (p2.col.val : Int) ∨
  (p1.row.val : Int) + (p1.col.val : Int) = (p2.row.val : Int) + (p2.col.val : Int)

/-- Checks if two queens attack each other -/
def queens_attack (p1 p2 : Position) : Prop :=
  p1.row = p2.row ∨ p1.col = p2.col ∨ on_same_diagonal p1 p2

/-- A valid placement of queens on the chessboard -/
structure QueenPlacement :=
  (black : List Position)
  (white : List Position)
  (black_valid : black.length ≤ 8)
  (white_valid : white.length ≤ 8)
  (no_attack : ∀ b ∈ black, ∀ w ∈ white, ¬queens_attack b w)

/-- The theorem to be proved -/
theorem max_queens_on_chessboard :
  ∃ (placement : QueenPlacement), placement.black.length = 8 ∧ placement.white.length = 8 ∧
  ∀ (other : QueenPlacement), other.black.length ≤ 8 ∧ other.white.length ≤ 8 :=
sorry

end max_queens_on_chessboard_l2642_264260


namespace number_of_friends_who_received_pebbles_l2642_264203

-- Define the given quantities
def total_weight_kg : ℕ := 36
def pebble_weight_g : ℕ := 250
def pebbles_per_friend : ℕ := 4

-- Define the conversion factor from kg to g
def kg_to_g : ℕ := 1000

-- Theorem to prove
theorem number_of_friends_who_received_pebbles :
  (total_weight_kg * kg_to_g) / (pebble_weight_g * pebbles_per_friend) = 36 := by
  sorry

end number_of_friends_who_received_pebbles_l2642_264203


namespace pascal_triangle_41st_number_l2642_264281

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of elements in a row of Pascal's triangle -/
def pascal_row_length (row : ℕ) : ℕ := row + 1

/-- The row number (0-indexed) of Pascal's triangle with 43 numbers -/
def target_row : ℕ := 42

/-- The index (0-indexed) of the target number in the row -/
def target_index : ℕ := 40

theorem pascal_triangle_41st_number : 
  binomial target_row target_index = 861 ∧ 
  pascal_row_length target_row = 43 := by
  sorry

end pascal_triangle_41st_number_l2642_264281


namespace certain_number_problem_l2642_264292

theorem certain_number_problem (x : ℚ) :
  (2 / 5 : ℚ) * 300 - (3 / 5 : ℚ) * x = 45 → x = 125 := by
  sorry

end certain_number_problem_l2642_264292


namespace sufficient_condition_range_l2642_264286

theorem sufficient_condition_range (x a : ℝ) :
  (∀ x, (|x - a| < 1 → x^2 + x - 2 > 0) ∧
   ∃ x, x^2 + x - 2 > 0 ∧ |x - a| ≥ 1) →
  a ≤ -3 ∨ a ≥ 2 :=
by sorry

end sufficient_condition_range_l2642_264286


namespace inverse_proposition_l2642_264212

theorem inverse_proposition (a b : ℝ) :
  (∀ x y : ℝ, x > y → x^3 > y^3) →
  (a^3 > b^3 → a > b) :=
by sorry

end inverse_proposition_l2642_264212


namespace positive_number_equation_solution_l2642_264289

theorem positive_number_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt ((10 * x) / 3) = x ∧ x = 10 / 3 := by
  sorry

end positive_number_equation_solution_l2642_264289


namespace exists_non_illuminating_rotation_l2642_264235

/-- Represents a three-dimensional cube --/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a projector that illuminates an octant --/
structure Projector where
  position : ℝ × ℝ × ℝ
  illumination : Set (ℝ × ℝ × ℝ)

/-- Represents a rotation in three-dimensional space --/
structure Rotation where
  matrix : Matrix (Fin 3) (Fin 3) ℝ

/-- Function to check if a point is illuminated by the projector --/
def is_illuminated (p : Projector) (point : ℝ × ℝ × ℝ) : Prop :=
  point ∈ p.illumination

/-- Function to apply a rotation to a projector --/
def rotate_projector (r : Rotation) (p : Projector) : Projector :=
  sorry

/-- Theorem stating that there exists a rotation such that no vertices are illuminated --/
theorem exists_non_illuminating_rotation (c : Cube) (p : Projector) :
  p.position = (0, 0, 0) →  -- Projector is at the center of the cube
  ∃ (r : Rotation), ∀ (v : Fin 8), ¬is_illuminated (rotate_projector r p) (c.vertices v) :=
sorry

end exists_non_illuminating_rotation_l2642_264235


namespace log_sqrt_45_l2642_264209

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_sqrt_45 (a b : ℝ) (h1 : log10 2 = a) (h2 : log10 3 = b) :
  log10 (Real.sqrt 45) = -a/2 + b + 1/2 := by sorry

end log_sqrt_45_l2642_264209


namespace kerosene_cost_is_44_cents_l2642_264211

/-- The cost of a liter of kerosene in cents -/
def kerosene_cost_cents (rice_cost_dollars : ℚ) : ℚ :=
  let egg_dozen_cost := rice_cost_dollars
  let egg_cost := egg_dozen_cost / 12
  let half_liter_kerosene_cost := 8 * egg_cost
  let liter_kerosene_cost_dollars := 2 * half_liter_kerosene_cost
  100 * liter_kerosene_cost_dollars

/-- Theorem stating that the cost of a liter of kerosene is 44 cents -/
theorem kerosene_cost_is_44_cents : 
  kerosene_cost_cents (33/100) = 44 := by
sorry

#eval kerosene_cost_cents (33/100)

end kerosene_cost_is_44_cents_l2642_264211


namespace cloves_discrepancy_l2642_264225

/-- Represents the number of creatures that can be repelled by 3 cloves of garlic -/
structure RepelRatio :=
  (vampires : ℚ)
  (wights : ℚ)
  (vampire_bats : ℚ)

/-- Represents the number of creatures to be repelled -/
structure CreaturesToRepel :=
  (vampires : ℕ)
  (wights : ℕ)
  (vampire_bats : ℕ)

/-- Calculates the number of cloves needed based on the repel ratio and creatures to repel -/
def cloves_needed (ratio : RepelRatio) (creatures : CreaturesToRepel) : ℚ :=
  3 * (creatures.vampires / ratio.vampires + 
       creatures.wights / ratio.wights + 
       creatures.vampire_bats / ratio.vampire_bats)

/-- The main theorem stating that the calculated cloves needed is not equal to 72 -/
theorem cloves_discrepancy (ratio : RepelRatio) (creatures : CreaturesToRepel) :
  ratio.vampires = 1 →
  ratio.wights = 3 →
  ratio.vampire_bats = 8 →
  creatures.vampires = 30 →
  creatures.wights = 12 →
  creatures.vampire_bats = 40 →
  cloves_needed ratio creatures ≠ 72 := by
  sorry


end cloves_discrepancy_l2642_264225


namespace arithmetic_mean_4_16_l2642_264229

theorem arithmetic_mean_4_16 (x : ℝ) : x = (4 + 16) / 2 → x = 10 := by sorry

end arithmetic_mean_4_16_l2642_264229


namespace or_propagation_l2642_264227

theorem or_propagation (p q r : Prop) (h1 : p ∨ q) (h2 : ¬p ∨ r) : q ∨ r := by
  sorry

end or_propagation_l2642_264227


namespace kids_difference_l2642_264216

theorem kids_difference (camp_kids home_kids : ℕ) 
  (h1 : camp_kids = 202958)
  (h2 : home_kids = 777622) :
  home_kids - camp_kids = 574664 := by
  sorry

end kids_difference_l2642_264216


namespace quadratic_roots_abs_less_than_one_l2642_264259

theorem quadratic_roots_abs_less_than_one (a b : ℝ) 
  (h1 : abs a + abs b < 1) 
  (h2 : a^2 - 4*b ≥ 0) : 
  ∀ x, x^2 + a*x + b = 0 → abs x < 1 := by
sorry

end quadratic_roots_abs_less_than_one_l2642_264259


namespace second_class_average_marks_l2642_264201

theorem second_class_average_marks (n1 n2 : ℕ) (avg1 avg_total : ℚ) :
  n1 = 35 →
  n2 = 45 →
  avg1 = 40 →
  avg_total = 51.25 →
  (n1 * avg1 + n2 * (n1 * avg1 + n2 * avg_total - n1 * avg1) / n2) / (n1 + n2) = avg_total →
  (n1 * avg1 + n2 * avg_total - n1 * avg1) / n2 = 60 :=
by sorry

end second_class_average_marks_l2642_264201


namespace generatrix_angle_is_60_degrees_l2642_264202

/-- A cone whose lateral surface unfolds into a semicircle -/
structure SemiCircleCone where
  /-- The radius of the semicircle (equal to the generatrix of the cone) -/
  radius : ℝ
  /-- Assumption that the lateral surface unfolds into a semicircle -/
  lateral_surface_is_semicircle : True

/-- The angle between the two generatrices in the axial section of a cone
    whose lateral surface unfolds into a semicircle is 60 degrees -/
theorem generatrix_angle_is_60_degrees (cone : SemiCircleCone) :
  let angle_rad := Real.pi / 3
  angle_rad = Real.arccos (1 / 2) :=
by sorry

end generatrix_angle_is_60_degrees_l2642_264202


namespace wise_men_hat_guesses_l2642_264217

/-- Represents the maximum number of guaranteed correct hat color guesses -/
def max_guaranteed_correct_guesses (n k : ℕ) : ℕ :=
  n - k - 1

/-- Theorem stating the maximum number of guaranteed correct hat color guesses -/
theorem wise_men_hat_guesses (n k : ℕ) (h1 : k < n) :
  max_guaranteed_correct_guesses n k = n - k - 1 :=
by sorry

end wise_men_hat_guesses_l2642_264217


namespace water_level_rise_l2642_264273

/-- The rise in water level when a sphere is fully immersed in a rectangular vessel --/
theorem water_level_rise (sphere_radius : ℝ) (vessel_length : ℝ) (vessel_width : ℝ) :
  sphere_radius = 10 →
  vessel_length = 30 →
  vessel_width = 25 →
  ∃ (water_rise : ℝ), abs (water_rise - 5.59) < 0.01 :=
by
  sorry

end water_level_rise_l2642_264273


namespace vector_angle_difference_l2642_264290

theorem vector_angle_difference (α β : Real) (a b : Real × Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π)
  (h4 : a = (Real.cos α, Real.sin α))
  (h5 : b = (Real.cos β, Real.sin β))
  (h6 : ‖3 • a + b‖ = ‖a - 2 • b‖) : 
  β - α = 2 * π / 3 := by
  sorry

end vector_angle_difference_l2642_264290


namespace platform_length_l2642_264271

/-- Given a train and two structures it passes through, calculate the length of the second structure. -/
theorem platform_length
  (train_length : ℝ)
  (tunnel_length : ℝ)
  (tunnel_time : ℝ)
  (platform_time : ℝ)
  (h1 : train_length = 330)
  (h2 : tunnel_length = 1200)
  (h3 : tunnel_time = 45)
  (h4 : platform_time = 15) :
  (tunnel_length + train_length) / tunnel_time * platform_time - train_length = 180 :=
by sorry

end platform_length_l2642_264271


namespace chloe_first_round_points_l2642_264226

/-- Represents the points scored in a trivia game. -/
structure TriviaProblem where
  first_round : ℤ
  second_round : ℤ
  last_round : ℤ
  total_points : ℤ

/-- The solution to Chloe's trivia game problem. -/
theorem chloe_first_round_points (game : TriviaProblem) 
  (h1 : game.second_round = 50)
  (h2 : game.last_round = -4)
  (h3 : game.total_points = 86)
  (h4 : game.first_round + game.second_round + game.last_round = game.total_points) :
  game.first_round = 40 := by
  sorry

end chloe_first_round_points_l2642_264226


namespace three_quantities_change_l2642_264278

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

-- Define the line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle type
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the problem setup
def setup (A B P Q : Point) (l1 l2 lPQ : Line) : Prop :=
  (P.x = (A.x + B.x) / 2) ∧ 
  (P.y = (A.y + B.y) / 2) ∧
  (l1.a = lPQ.a ∧ l1.b = lPQ.b) ∧
  (l2.a = lPQ.a ∧ l2.b = lPQ.b)

-- Define the four quantities
def lengthAB (A B : Point) : ℝ := sorry
def perimeterAPB (A B P : Point) : ℝ := sorry
def areaAPB (A B P : Point) : ℝ := sorry
def distancePtoAB (A B P : Point) : ℝ := sorry

-- Define a function that counts how many quantities change
def countChangingQuantities (A B P Q : Point) (l1 l2 lPQ : Line) : ℕ := sorry

-- The main theorem
theorem three_quantities_change 
  (A B P Q : Point) (l1 l2 lPQ : Line) 
  (h : setup A B P Q l1 l2 lPQ) : 
  countChangingQuantities A B P Q l1 l2 lPQ = 3 := sorry

end three_quantities_change_l2642_264278


namespace largest_consecutive_even_l2642_264246

theorem largest_consecutive_even : 
  ∀ (x : ℕ), 
  (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14) = 424) → 
  (x + 14 = 60) := by
sorry

end largest_consecutive_even_l2642_264246


namespace set_operations_l2642_264238

-- Define the universal set U
def U : Set ℕ := {x | x < 10}

-- Define set A
def A : Set ℕ := {x ∈ U | ∃ k, x = 2 * k}

-- Define set B
def B : Set ℕ := {x | x^2 - 3*x + 2 = 0}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {2}) ∧
  (A ∪ B = {0, 1, 2, 4, 6, 8}) ∧
  (U \ A = {1, 3, 5, 7, 9}) := by
  sorry

end set_operations_l2642_264238


namespace white_triangle_pairs_coincide_l2642_264287

/-- Represents the number of triangles of each color in each half -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of triangles -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ
  blue_white : ℕ

/-- Theorem stating that given the conditions in the problem, 
    the number of coinciding white triangle pairs is 3 -/
theorem white_triangle_pairs_coincide 
  (counts : TriangleCounts)
  (pairs : CoincidingPairs)
  (h1 : counts.red = 4)
  (h2 : counts.blue = 6)
  (h3 : counts.white = 9)
  (h4 : pairs.red_red = 3)
  (h5 : pairs.blue_blue = 4)
  (h6 : pairs.red_white = 3)
  (h7 : pairs.blue_white = 3) :
  ∃ (white_white_pairs : ℕ), white_white_pairs = 3 :=
sorry

end white_triangle_pairs_coincide_l2642_264287


namespace smallest_common_rose_purchase_l2642_264284

theorem smallest_common_rose_purchase : Nat.lcm 9 19 = 171 := by
  sorry

end smallest_common_rose_purchase_l2642_264284


namespace inequality_system_solution_l2642_264252

theorem inequality_system_solution (x : ℝ) : 
  (2 * x - 1 > 0 ∧ 3 * x > 2 * x + 2) ↔ x > 2 := by sorry

end inequality_system_solution_l2642_264252


namespace old_record_calculation_old_record_proof_l2642_264295

/-- Calculates the old record given James' performance and the points he beat the record by -/
theorem old_record_calculation (touchdowns_per_game : ℕ) (points_per_touchdown : ℕ) 
  (games_in_season : ℕ) (two_point_conversions : ℕ) (points_beat_record_by : ℕ) : ℕ :=
  let james_total_points := touchdowns_per_game * points_per_touchdown * games_in_season + 
                            two_point_conversions * 2
  james_total_points - points_beat_record_by

/-- Proves that the old record was 300 points given James' performance -/
theorem old_record_proof :
  old_record_calculation 4 6 15 6 72 = 300 := by
  sorry

end old_record_calculation_old_record_proof_l2642_264295


namespace intersection_point_parameter_l2642_264256

/-- Given three lines that intersect at a single point but do not form a triangle, 
    prove that the parameter 'a' in one of the lines must equal -1. -/
theorem intersection_point_parameter (a : ℝ) : 
  (∃ (x y : ℝ), ax + 2*y + 8 = 0 ∧ 4*x + 3*y = 10 ∧ 2*x - y = 10) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (ax₁ + 2*y₁ + 8 = 0 ∧ 4*x₁ + 3*y₁ = 10) →
    (ax₂ + 2*y₂ + 8 = 0 ∧ 2*x₂ - y₂ = 10) →
    x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  (∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (ax₁ + 2*y₁ + 8 = 0 ∧ 4*x₂ + 3*y₂ = 10 ∧ 2*x₃ - y₃ = 10) →
    ¬(Set.ncard {(x₁, y₁), (x₂, y₂), (x₃, y₃)} = 3)) →
  a = -1 :=
sorry

end intersection_point_parameter_l2642_264256


namespace prism_faces_count_l2642_264275

/-- A prism with n sides, where n is at least 3 --/
structure Prism (n : ℕ) where
  sides : n ≥ 3

/-- The number of faces in a prism --/
def num_faces (n : ℕ) (p : Prism n) : ℕ := n + 2

theorem prism_faces_count (n : ℕ) (p : Prism n) : 
  num_faces n p ≠ n :=
sorry

end prism_faces_count_l2642_264275


namespace inequality_range_l2642_264257

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ k ∈ Set.Ioc (-4) 0 := by
  sorry

end inequality_range_l2642_264257


namespace sum_of_roots_quadratic_l2642_264210

theorem sum_of_roots_quadratic (x : ℝ) : (x + 3) * (x - 2) = 15 → ∃ y : ℝ, (y + 3) * (y - 2) = 15 ∧ x + y = -1 := by
  sorry

end sum_of_roots_quadratic_l2642_264210


namespace convention_handshakes_correct_l2642_264297

/-- Represents the number of handshakes at a twins and triplets convention -/
def convention_handshakes (twin_sets : ℕ) (triplet_sets : ℕ) : ℕ :=
  let twins := twin_sets * 2
  let triplets := triplet_sets * 3
  let twin_handshakes := twins * (twins - 2)
  let triplet_handshakes := triplets * (triplets - 3)
  let cross_handshakes := twins * (triplets / 2) * 2
  (twin_handshakes + triplet_handshakes + cross_handshakes) / 2

theorem convention_handshakes_correct :
  convention_handshakes 9 6 = 441 := by sorry

end convention_handshakes_correct_l2642_264297


namespace circle_circumference_increase_l2642_264234

theorem circle_circumference_increase (d : ℝ) : 
  let original_circumference := π * d
  let new_circumference := π * (d + 2 * π)
  new_circumference - original_circumference = 2 * π^2 := by
sorry

end circle_circumference_increase_l2642_264234


namespace purple_probability_ten_sided_die_l2642_264220

/-- Represents a die with a specific number of sides and purple faces -/
structure Die :=
  (sides : ℕ)
  (purpleFaces : ℕ)
  (hPurple : purpleFaces ≤ sides)

/-- Calculates the probability of rolling a purple face on a given die -/
def probabilityPurple (d : Die) : ℚ :=
  d.purpleFaces / d.sides

/-- Theorem stating that for a 10-sided die with 2 purple faces, 
    the probability of rolling a purple face is 1/5 -/
theorem purple_probability_ten_sided_die :
  ∀ d : Die, d.sides = 10 → d.purpleFaces = 2 → probabilityPurple d = 1/5 := by
  sorry

end purple_probability_ten_sided_die_l2642_264220


namespace complex_coordinate_i_times_2_minus_i_l2642_264232

theorem complex_coordinate_i_times_2_minus_i : 
  (Complex.I * (2 - Complex.I)).re = 1 ∧ (Complex.I * (2 - Complex.I)).im = 2 := by
  sorry

end complex_coordinate_i_times_2_minus_i_l2642_264232


namespace bookshelf_problem_l2642_264207

theorem bookshelf_problem (initial_books : ℕ) 
  (day1_borrow day1_return day2_borrow day2_return : ℤ) : 
  initial_books = 20 →
  day1_borrow = -3 →
  day1_return = 1 →
  day2_borrow = -1 →
  day2_return = 2 →
  (initial_books : ℤ) + day1_borrow + day1_return + day2_borrow + day2_return = 19 :=
by sorry

end bookshelf_problem_l2642_264207


namespace right_triangle_inscribed_circle_l2642_264276

theorem right_triangle_inscribed_circle 
  (a b T C : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ T > 0 ∧ C > 0) 
  (h_right_triangle : ∃ (c h : ℝ), c = a + b ∧ T = (1/2) * c * h ∧ h^2 = a * b) 
  (h_inscribed : ∃ (R : ℝ), C = π * R^2 ∧ 2 * R = a + b) : 
  a * b = π * T^2 / C := by
sorry

end right_triangle_inscribed_circle_l2642_264276


namespace F_two_F_perfect_square_l2642_264240

def best_decomposition (n : ℕ+) : ℕ+ × ℕ+ :=
  sorry

def F (n : ℕ+) : ℚ :=
  let (p, q) := best_decomposition n
  (p : ℚ) / q

theorem F_two : F 2 = 1/2 := by sorry

theorem F_perfect_square (n : ℕ+) (h : ∃ m : ℕ+, n = m * m) : F n = 1 := by sorry

end F_two_F_perfect_square_l2642_264240


namespace suggestion_difference_l2642_264291

def student_count : ℕ := 1200

def food_suggestions : List ℕ := [408, 305, 137, 213, 137]

theorem suggestion_difference : 
  (List.maximum food_suggestions).get! - (List.minimum food_suggestions).get! = 271 := by
  sorry

end suggestion_difference_l2642_264291


namespace largest_n_for_product_1764_l2642_264274

/-- Two arithmetic sequences with integer terms -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem largest_n_for_product_1764 
  (a b : ℕ → ℤ) 
  (ha : ArithmeticSequence a) 
  (hb : ArithmeticSequence b) 
  (h_a1 : a 1 = 1) 
  (h_b1 : b 1 = 1) 
  (h_a2_le_b2 : a 2 ≤ b 2) 
  (h_product : ∃ n : ℕ, a n * b n = 1764) :
  (∀ m : ℕ, (∃ k : ℕ, a k * b k = 1764) → m ≤ 44) ∧ 
  (∃ n : ℕ, a n * b n = 1764 ∧ n = 44) :=
sorry

end largest_n_for_product_1764_l2642_264274


namespace angle_A_measure_triangle_area_l2642_264248

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def triangle_condition (t : Triangle) : Prop :=
  (t.a - t.b) / t.c = (Real.sin t.B + Real.sin t.C) / (Real.sin t.B + Real.sin t.A)

-- Theorem 1: Prove that angle A measures 2π/3
theorem angle_A_measure (t : Triangle) (h : triangle_condition t) : t.A = 2 * Real.pi / 3 := by
  sorry

-- Theorem 2: Prove the area of the triangle when a = √7 and b = 2c
theorem triangle_area (t : Triangle) (h1 : triangle_condition t) (h2 : t.a = Real.sqrt 7) (h3 : t.b = 2 * t.c) :
  (1 / 2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2 := by
  sorry

end angle_A_measure_triangle_area_l2642_264248


namespace white_shirts_count_l2642_264254

/-- The number of white t-shirts in each pack -/
def white_shirts_per_pack : ℕ := sorry

/-- The number of packs of white t-shirts bought -/
def white_packs : ℕ := 5

/-- The number of packs of blue t-shirts bought -/
def blue_packs : ℕ := 3

/-- The number of blue t-shirts in each pack -/
def blue_shirts_per_pack : ℕ := 9

/-- The total number of t-shirts bought -/
def total_shirts : ℕ := 57

theorem white_shirts_count : white_shirts_per_pack = 6 :=
  by sorry

end white_shirts_count_l2642_264254
