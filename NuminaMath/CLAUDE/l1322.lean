import Mathlib

namespace parallel_line_slope_l1322_132227

/-- Given a line with equation 3x - 6y = 12, prove that the slope of any parallel line is 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 12) → 
  ∃ m : ℝ, m = (1 : ℝ) / 2 ∧ ∀ (x₁ y₁ : ℝ), (3 * x₁ - 6 * y₁ = 12) → 
    ∃ b : ℝ, y₁ = m * x₁ + b :=
by sorry

end parallel_line_slope_l1322_132227


namespace f_at_3_l1322_132250

def f (x : ℝ) : ℝ := 9*x^3 - 5*x^2 - 3*x + 7

theorem f_at_3 : f 3 = 196 := by
  sorry

end f_at_3_l1322_132250


namespace smallest_taco_packages_l1322_132261

/-- The number of tacos in each package -/
def tacos_per_package : ℕ := 4

/-- The number of taco shells in each package -/
def shells_per_package : ℕ := 6

/-- The minimum number of tacos and taco shells required -/
def min_required : ℕ := 60

/-- Proposition: The smallest number of taco packages to buy is 15 -/
theorem smallest_taco_packages : 
  (∃ (taco_packages shell_packages : ℕ),
    taco_packages * tacos_per_package = shell_packages * shells_per_package ∧
    taco_packages * tacos_per_package ≥ min_required ∧
    shell_packages * shells_per_package ≥ min_required ∧
    ∀ (t s : ℕ), 
      t * tacos_per_package = s * shells_per_package →
      t * tacos_per_package ≥ min_required →
      s * shells_per_package ≥ min_required →
      t ≥ taco_packages) →
  (∃ (shell_packages : ℕ),
    15 * tacos_per_package = shell_packages * shells_per_package ∧
    15 * tacos_per_package ≥ min_required ∧
    shell_packages * shells_per_package ≥ min_required) :=
by sorry

end smallest_taco_packages_l1322_132261


namespace sqrt_product_equality_l1322_132206

theorem sqrt_product_equality : Real.sqrt 128 * Real.sqrt 50 * Real.sqrt 18 = 240 * Real.sqrt 2 := by
  sorry

end sqrt_product_equality_l1322_132206


namespace unique_assignment_l1322_132240

/-- Represents a valid assignment of digits to letters -/
structure Assignment where
  a : Fin 5
  m : Fin 5
  e : Fin 5
  h : Fin 5
  z : Fin 5
  different : a ≠ m ∧ a ≠ e ∧ a ≠ h ∧ a ≠ z ∧ m ≠ e ∧ m ≠ h ∧ m ≠ z ∧ e ≠ h ∧ e ≠ z ∧ h ≠ z

/-- The inequalities that must be satisfied -/
def satisfies_inequalities (assign : Assignment) : Prop :=
  3 > assign.a.val + 1 ∧
  assign.a.val + 1 > assign.m.val + 1 ∧
  assign.m.val + 1 < assign.e.val + 1 ∧
  assign.e.val + 1 < assign.h.val + 1 ∧
  assign.h.val + 1 < assign.a.val + 1

/-- The theorem stating that the only valid assignment results in ZAMENA = 541234 -/
theorem unique_assignment :
  ∀ (assign : Assignment),
    satisfies_inequalities assign →
    assign.z.val = 4 ∧
    assign.a.val = 3 ∧
    assign.m.val = 0 ∧
    assign.e.val = 1 ∧
    assign.h.val = 2 :=
by sorry

end unique_assignment_l1322_132240


namespace simplify_trig_expression_l1322_132200

theorem simplify_trig_expression (α : ℝ) :
  (1 - Real.cos (2 * α) + Real.sin (2 * α)) / (1 + Real.cos (2 * α) + Real.sin (2 * α)) = Real.tan α := by
  sorry

end simplify_trig_expression_l1322_132200


namespace simplify_nested_expression_l1322_132224

theorem simplify_nested_expression (x : ℝ) :
  2 * (1 - (2 * (1 - (1 + (2 * (1 - x)))))) = 8 * x - 10 := by
  sorry

end simplify_nested_expression_l1322_132224


namespace fraction_sum_and_multiply_l1322_132236

theorem fraction_sum_and_multiply :
  ((2 : ℚ) / 9 + 4 / 11) * 3 / 5 = 58 / 165 := by
  sorry

end fraction_sum_and_multiply_l1322_132236


namespace arithmetic_sequence_sum_property_l1322_132280

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 2 + a 8 = 10) :
  a 3 + a 7 = 10 :=
by
  sorry

end arithmetic_sequence_sum_property_l1322_132280


namespace race_distance_proof_l1322_132221

/-- The total distance of a race where:
    - A covers the distance in 45 seconds
    - B covers the distance in 60 seconds
    - A beats B by 50 meters
-/
def race_distance : ℝ := 150

theorem race_distance_proof :
  ∀ (a_time b_time : ℝ) (lead : ℝ),
  a_time = 45 ∧ 
  b_time = 60 ∧ 
  lead = 50 →
  race_distance = (lead * b_time) / (b_time / a_time - 1) :=
by sorry

end race_distance_proof_l1322_132221


namespace percentage_increase_l1322_132256

theorem percentage_increase (w : ℝ) (P : ℝ) : 
  w = 80 →
  (w + P / 100 * w) - (w - 25 / 100 * w) = 30 →
  P = 12.5 := by
sorry

end percentage_increase_l1322_132256


namespace clothing_sale_profit_l1322_132230

def initial_cost : ℕ := 400
def num_sets : ℕ := 8
def sale_price : ℕ := 55
def adjustments : List ℤ := [2, -3, 2, 1, -2, -1, 0, -2]

theorem clothing_sale_profit :
  (num_sets * sale_price : ℤ) + (adjustments.sum) - initial_cost = 37 := by
  sorry

end clothing_sale_profit_l1322_132230


namespace arg_ratio_of_unit_complex_l1322_132259

theorem arg_ratio_of_unit_complex (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 1)
  (h₂ : Complex.abs z₂ = 1)
  (h₃ : z₂ - z₁ = -1) :
  Complex.arg (z₁ / z₂) = π / 3 ∨ Complex.arg (z₁ / z₂) = 5 * π / 3 := by
  sorry

end arg_ratio_of_unit_complex_l1322_132259


namespace sum_of_specific_terms_l1322_132201

def sequence_sum (n : ℕ) : ℤ := n^2 - 1

def sequence_term (n : ℕ) : ℤ :=
  if n = 1 then 0
  else 2 * n - 2

theorem sum_of_specific_terms : 
  sequence_term 1 + sequence_term 3 + sequence_term 5 + sequence_term 7 + sequence_term 9 = 44 :=
by sorry

end sum_of_specific_terms_l1322_132201


namespace f_properties_l1322_132232

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2 + 1

theorem f_properties :
  let f := f
  ∃ (period : ℝ),
    (f (5 * Real.pi / 4) = Real.sqrt 3) ∧
    (period > 0 ∧ ∀ (x : ℝ), f (x + period) = f x) ∧
    (∀ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) → period ≤ p) ∧
    (f (-Real.pi / 5) < f (7 * Real.pi / 8)) := by
  sorry

end f_properties_l1322_132232


namespace smallest_m_no_real_roots_l1322_132284

theorem smallest_m_no_real_roots : ∃ (m : ℤ),
  (∀ (k : ℤ), k < m → ∃ (x : ℝ), 3 * x * (k * x - 6) - 2 * x^2 + 10 = 0) ∧
  (∀ (x : ℝ), 3 * x * (m * x - 6) - 2 * x^2 + 10 ≠ 0) ∧
  m = 4 := by
  sorry

end smallest_m_no_real_roots_l1322_132284


namespace intersection_distance_sum_l1322_132279

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (t, -1 + Real.sqrt 3 * t)

-- Define the curve C in polar coordinates
def curve_C (θ : ℝ) : ℝ := 2 * Real.sin θ + 2 * Real.cos θ

-- Define point P
def point_P : ℝ × ℝ := (0, -1)

-- Define the intersection points A and B
variable (A B : ℝ × ℝ)

-- Assume A and B are on both the line and the curve
axiom A_on_line : ∃ t : ℝ, line_l t = A
axiom B_on_line : ∃ t : ℝ, line_l t = B
axiom A_on_curve : ∃ θ : ℝ, (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ) = A
axiom B_on_curve : ∃ θ : ℝ, (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ) = B

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem intersection_distance_sum :
  1 / distance point_P A + 1 / distance point_P B = (2 * Real.sqrt 3 + 1) / 3 := by sorry

end intersection_distance_sum_l1322_132279


namespace perpendicular_line_equation_l1322_132289

/-- The equation of a line perpendicular to 2x+y-5=0 and passing through (2,3) is x-2y+4=0 -/
theorem perpendicular_line_equation : 
  ∀ (x y : ℝ), 
  (∃ (m : ℝ), (2 : ℝ) * x + y - 5 = 0 ↔ y = -2 * x + m) →
  (∃ (k : ℝ), k * (x - 2) + 3 = y ∧ k * 2 = -1) →
  x - 2 * y + 4 = 0 := by
sorry

end perpendicular_line_equation_l1322_132289


namespace smallest_positive_integer_with_remainders_l1322_132243

theorem smallest_positive_integer_with_remainders : ∃! x : ℕ+, 
  (x : ℤ) % 5 = 2 ∧ 
  (x : ℤ) % 7 = 3 ∧ 
  (x : ℤ) % 9 = 4 ∧
  ∀ y : ℕ+, ((y : ℤ) % 5 = 2 ∧ (y : ℤ) % 7 = 3 ∧ (y : ℤ) % 9 = 4) → x ≤ y :=
by
  -- Proof goes here
  sorry

end smallest_positive_integer_with_remainders_l1322_132243


namespace no_complete_non_self_intersecting_path_l1322_132263

/-- Represents the surface of a Rubik's cube -/
structure RubiksCubeSurface where
  squares : Nat
  diagonals : Nat
  vertices : Nat

/-- The surface of a standard Rubik's cube -/
def standardRubiksCube : RubiksCubeSurface :=
  { squares := 54
  , diagonals := 54
  , vertices := 56 }

/-- A path on the surface of a Rubik's cube -/
structure DiagonalPath (surface : RubiksCubeSurface) where
  length : Nat
  is_non_self_intersecting : Bool

/-- Theorem stating the impossibility of creating a non-self-intersecting path
    using all diagonals on the surface of a standard Rubik's cube -/
theorem no_complete_non_self_intersecting_path 
  (surface : RubiksCubeSurface) 
  (h_surface : surface = standardRubiksCube) :
  ¬∃ (path : DiagonalPath surface), 
    path.length = surface.diagonals ∧ 
    path.is_non_self_intersecting = true := by
  sorry


end no_complete_non_self_intersecting_path_l1322_132263


namespace fraction_sum_l1322_132255

theorem fraction_sum : (3 : ℚ) / 5 + (2 : ℚ) / 15 = (11 : ℚ) / 15 := by
  sorry

end fraction_sum_l1322_132255


namespace total_cost_shirt_and_shoes_l1322_132253

/-- The total cost of a shirt and shoes, given the shirt cost and the relationship between shirt and shoe costs -/
theorem total_cost_shirt_and_shoes (shirt_cost : ℕ) (h1 : shirt_cost = 97) :
  let shoe_cost := 2 * shirt_cost + 9
  shirt_cost + shoe_cost = 300 := by
sorry


end total_cost_shirt_and_shoes_l1322_132253


namespace car_travel_time_l1322_132296

theorem car_travel_time (speed_x speed_y distance_after_y : ℝ) 
  (hx : speed_x = 35)
  (hy : speed_y = 70)
  (hd : distance_after_y = 42)
  (h_same_distance : ∀ t : ℝ, speed_x * (t + (distance_after_y / speed_x)) = speed_y * t) :
  (distance_after_y / speed_x) * 60 = 72 := by
sorry

end car_travel_time_l1322_132296


namespace total_green_marbles_l1322_132290

/-- The number of green marbles Sara has -/
def sara_green : ℕ := 3

/-- The number of green marbles Tom has -/
def tom_green : ℕ := 4

/-- The total number of green marbles Sara and Tom have -/
def total_green : ℕ := sara_green + tom_green

theorem total_green_marbles : total_green = 7 := by
  sorry

end total_green_marbles_l1322_132290


namespace arrange_in_order_l1322_132223

def Ψ : ℤ := -(1006 : ℤ)

def Ω : ℤ := -(1007 : ℤ)

def Θ : ℤ := -(1008 : ℤ)

theorem arrange_in_order : Θ < Ω ∧ Ω < Ψ := by
  sorry

end arrange_in_order_l1322_132223


namespace det_A_eq_33_l1322_132220

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![5, 0, -2],
    ![1, 3,  4],
    ![0, -1, 1]]

theorem det_A_eq_33 : A.det = 33 := by
  sorry

end det_A_eq_33_l1322_132220


namespace max_b_in_box_l1322_132277

/-- Given a rectangular box with volume 360 cubic units and integer dimensions a, b, and c 
    where a > b > c > 2, the maximum value of b is 10. -/
theorem max_b_in_box (a b c : ℕ) : 
  a * b * c = 360 →
  a > b →
  b > c →
  c > 2 →
  b ≤ 10 ∧ ∃ (a' b' c' : ℕ), a' * b' * c' = 360 ∧ a' > b' ∧ b' > c' ∧ c' > 2 ∧ b' = 10 :=
by sorry

end max_b_in_box_l1322_132277


namespace zero_in_interval_implies_alpha_range_l1322_132267

theorem zero_in_interval_implies_alpha_range (α : ℝ) :
  (∃ x ∈ Set.Icc 0 1, x^2 + 2*α*x + 1 = 0) → α ≤ -1 := by
  sorry

end zero_in_interval_implies_alpha_range_l1322_132267


namespace annie_figurines_count_l1322_132214

def number_of_tvs : ℕ := 5
def cost_per_tv : ℕ := 50
def total_spent : ℕ := 260
def cost_per_figurine : ℕ := 1

theorem annie_figurines_count :
  (total_spent - number_of_tvs * cost_per_tv) / cost_per_figurine = 10 := by
  sorry

end annie_figurines_count_l1322_132214


namespace no_single_digit_A_with_integer_solutions_l1322_132234

theorem no_single_digit_A_with_integer_solutions : 
  ∀ A : ℕ, 1 ≤ A ∧ A ≤ 9 → 
  ¬∃ x : ℕ, x > 0 ∧ x^2 - 2*A*x + A*10 = 0 :=
by sorry

end no_single_digit_A_with_integer_solutions_l1322_132234


namespace frank_floor_l1322_132209

/-- Given information about the floors where Dennis, Charlie, and Frank live,
    prove that Frank lives on the 16th floor. -/
theorem frank_floor (dennis_floor charlie_floor frank_floor : ℕ) 
  (h1 : dennis_floor = charlie_floor + 2)
  (h2 : charlie_floor = frank_floor / 4)
  (h3 : dennis_floor = 6) :
  frank_floor = 16 := by
  sorry

end frank_floor_l1322_132209


namespace simplify_quadratic_expression_l1322_132286

/-- Simplification of a quadratic expression -/
theorem simplify_quadratic_expression (y : ℝ) :
  4 * y + 8 * y^2 + 6 - (3 - 4 * y - 8 * y^2) = 16 * y^2 + 8 * y + 3 := by
  sorry

end simplify_quadratic_expression_l1322_132286


namespace triangle_base_calculation_l1322_132216

theorem triangle_base_calculation (square_perimeter : ℝ) (triangle_area : ℝ) :
  square_perimeter = 60 →
  triangle_area = 150 →
  let square_side := square_perimeter / 4
  let triangle_height := square_side
  triangle_area = 1/2 * triangle_height * (triangle_base : ℝ) →
  triangle_base = 20 := by sorry

end triangle_base_calculation_l1322_132216


namespace red_card_events_l1322_132219

-- Define the set of cards
inductive Card : Type
| Black : Card
| Red : Card
| White : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the event "A gets the red card"
def A_gets_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "B gets the red card"
def B_gets_red (d : Distribution) : Prop := d Person.B = Card.Red

-- Define mutually exclusive events
def mutually_exclusive (P Q : Distribution → Prop) : Prop :=
  ∀ d : Distribution, ¬(P d ∧ Q d)

-- Define opposite events
def opposite_events (P Q : Distribution → Prop) : Prop :=
  ∀ d : Distribution, P d ↔ ¬Q d

-- Theorem statement
theorem red_card_events :
  (mutually_exclusive A_gets_red B_gets_red) ∧
  ¬(opposite_events A_gets_red B_gets_red) := by
  sorry

end red_card_events_l1322_132219


namespace jackie_apples_l1322_132268

theorem jackie_apples (adam_apples : ℕ) (difference : ℕ) (jackie_apples : ℕ) : 
  adam_apples = 14 → 
  adam_apples = jackie_apples + difference → 
  difference = 5 →
  jackie_apples = 9 := by
sorry

end jackie_apples_l1322_132268


namespace dogs_per_box_l1322_132258

theorem dogs_per_box (total_boxes : ℕ) (total_dogs : ℕ) (dogs_per_box : ℕ) :
  total_boxes = 7 →
  total_dogs = 28 →
  total_dogs = total_boxes * dogs_per_box →
  dogs_per_box = 4 := by
  sorry

end dogs_per_box_l1322_132258


namespace complex_integer_calculation_l1322_132276

theorem complex_integer_calculation : (-7)^7 / 7^4 + 2^6 - 8^2 = -343 := by
  sorry

end complex_integer_calculation_l1322_132276


namespace hyperbola_eccentricity_l1322_132278

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) : a > 0 → b > 0 →
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (∃ x y : ℝ, (x - c)^2 + y^2 = 4 * a^2) →  -- Circle equation
  (∃ x y : ℝ, (x - c)^2 + y^2 = 4 * a^2 ∧ b * x + a * y = 0 ∧ y^2 = b^2) →  -- Chord condition
  c^2 = a^2 * (1 + (c^2 / a^2 - 1)) →  -- Semi-latus rectum condition
  Real.sqrt ((c^2 / a^2) - 1) = Real.sqrt 3 :=
by sorry

end hyperbola_eccentricity_l1322_132278


namespace matrix_product_equals_C_l1322_132297

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, -1, 2; 1, 0, 5; 4, 1, -2]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![2, -3, 4; -1, 5, -2; 0, 2, 7]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![7, -10, 28; 2, 7, 39; 7, -11, 0]

theorem matrix_product_equals_C : A * B = C := by
  sorry

end matrix_product_equals_C_l1322_132297


namespace simplify_expression_l1322_132295

theorem simplify_expression (x : ℝ) : 2*x + 3 - 4*x - 5 + 6*x + 7 - 8*x - 9 = -4*x - 4 := by
  sorry

end simplify_expression_l1322_132295


namespace pat_calculation_l1322_132238

theorem pat_calculation (x : ℝ) : (x / 6) - 14 = 16 → (x * 6) + 14 > 1000 := by
  sorry

end pat_calculation_l1322_132238


namespace arithmetic_sequence_sum_l1322_132239

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : a 4 = 8) (h2 : a 5 = 12) (h3 : a 6 = 16) :
  a 1 + a 2 + a 3 = 0 :=
by sorry

end arithmetic_sequence_sum_l1322_132239


namespace problem_statement_l1322_132294

theorem problem_statement : (-5)^5 / 5^3 + 3^4 - 6^1 = 50 := by
  sorry

end problem_statement_l1322_132294


namespace min_sum_dimensions_l1322_132241

/-- Represents the dimensions of a rectangular box. -/
structure BoxDimensions where
  x : ℕ+
  y : ℕ+
  z : ℕ+
  y_eq_x_plus_3 : y = x + 3
  product_eq_2541 : x * y * z = 2541

/-- The sum of the dimensions of a box. -/
def sum_dimensions (d : BoxDimensions) : ℕ := d.x + d.y + d.z

/-- Theorem stating the minimum sum of dimensions for the given conditions. -/
theorem min_sum_dimensions :
  ∀ d : BoxDimensions, sum_dimensions d ≥ 38 := by sorry

end min_sum_dimensions_l1322_132241


namespace no_roots_of_third_trinomial_l1322_132217

theorem no_roots_of_third_trinomial (a b : ℤ) : 
  (∃ x : ℤ, x^2 + a*x + b = 0) → 
  (∃ y : ℤ, y^2 + a*y + (b + 1) = 0) → 
  ∀ z : ℝ, z^2 + a*z + (b + 2) ≠ 0 :=
by sorry

end no_roots_of_third_trinomial_l1322_132217


namespace part_one_part_two_l1322_132210

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := (k - 1) * x^2 - 4 * x + 3

-- Part 1
theorem part_one (k : ℝ) :
  (quadratic_equation k 1 = 0) → 
  (k = 2 ∧ ∃ x, x ≠ 1 ∧ quadratic_equation k x = 0 ∧ x = 3) :=
by sorry

-- Part 2
theorem part_two (k x₁ x₂ : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ quadratic_equation k x₁ = 0 ∧ quadratic_equation k x₂ = 0) →
  (x₁^2 * x₂ + x₁ * x₂^2 = 3) →
  (k = -1) :=
by sorry

end part_one_part_two_l1322_132210


namespace leah_chocolates_l1322_132228

theorem leah_chocolates (leah_chocolates max_chocolates : ℕ) : 
  leah_chocolates = max_chocolates + 8 →
  max_chocolates = leah_chocolates / 3 →
  leah_chocolates = 12 := by
sorry

end leah_chocolates_l1322_132228


namespace triangle_inequality_l1322_132205

open Real

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point M
def M : ℝ × ℝ := sorry

-- Define the semi-perimeter p
def semiPerimeter (t : Triangle) : ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_inequality (t : Triangle) :
  let p := semiPerimeter t
  distance M t.A * cos (angle t.B t.A t.C / 2) +
  distance M t.B * cos (angle t.A t.B t.C / 2) +
  distance M t.C * cos (angle t.A t.C t.B / 2) ≥ p := by
  sorry

end triangle_inequality_l1322_132205


namespace rotated_square_height_l1322_132298

theorem rotated_square_height :
  let square_side : ℝ := 1
  let rotation_angle : ℝ := 60 * (π / 180)  -- 60 degrees in radians
  let diagonal : ℝ := square_side * Real.sqrt 2
  let height_above_center : ℝ := (diagonal / 2) * Real.sin rotation_angle
  let original_center_height : ℝ := square_side / 2
  let total_height : ℝ := original_center_height + height_above_center
  total_height = (2 + Real.sqrt 6) / 4 := by sorry

end rotated_square_height_l1322_132298


namespace distance_between_specific_planes_l1322_132222

/-- Represents a plane in 3D space defined by ax + by + cz = d -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the distance between two planes -/
def distance_between_planes (p1 p2 : Plane) : ℝ :=
  sorry

/-- The first plane: 2x + 4y - 2z = 10 -/
def plane1 : Plane := ⟨2, 4, -2, 10⟩

/-- The second plane: x + 2y - z = -3 -/
def plane2 : Plane := ⟨1, 2, -1, -3⟩

theorem distance_between_specific_planes :
  distance_between_planes plane1 plane2 = Real.sqrt 6 / 6 := by
  sorry

end distance_between_specific_planes_l1322_132222


namespace pablo_stack_difference_l1322_132292

/-- The height of Pablo's toy block stacks -/
def PabloStacks : ℕ → ℕ
| 0 => 5  -- First stack
| 1 => PabloStacks 0 + 2  -- Second stack
| 2 => PabloStacks 1 - 5  -- Third stack
| 3 => 21 - (PabloStacks 0 + PabloStacks 1 + PabloStacks 2)  -- Last stack
| _ => 0  -- Any other index

theorem pablo_stack_difference : PabloStacks 3 - PabloStacks 2 = 5 := by
  sorry

end pablo_stack_difference_l1322_132292


namespace power_of_power_of_three_l1322_132231

theorem power_of_power_of_three : (3^3)^(3^3) = 7625597484987 := by sorry

end power_of_power_of_three_l1322_132231


namespace y_value_l1322_132245

theorem y_value (y : ℝ) (h : (9 : ℝ) / y^3 = y / 81) : y = 3 * Real.sqrt 3 := by
  sorry

end y_value_l1322_132245


namespace dogwood_tree_planting_l1322_132202

/-- The number of dogwood trees planted today -/
def trees_planted_today : ℕ := 41

/-- The initial number of trees in the park -/
def initial_trees : ℕ := 39

/-- The number of trees to be planted tomorrow -/
def trees_planted_tomorrow : ℕ := 20

/-- The final number of trees in the park -/
def final_trees : ℕ := 100

theorem dogwood_tree_planting :
  initial_trees + trees_planted_today + trees_planted_tomorrow = final_trees :=
by sorry

end dogwood_tree_planting_l1322_132202


namespace problem1_problem2_problem3_problem4_l1322_132248

theorem problem1 : 5 / 7 + (-5 / 6) - (-2 / 7) + 1 + 1 / 6 = 4 / 3 := by sorry

theorem problem2 : (1 / 2 - (1 + 1 / 3) + 3 / 8) / (-1 / 24) = 11 := by sorry

theorem problem3 : (-3)^3 + (-5)^2 - |(-3)| * 4 = -14 := by sorry

theorem problem4 : -(1^101) - (-0.5 - (1 - 3 / 5 * 0.7) / (-1 / 2)^2) = 91 / 50 := by sorry

end problem1_problem2_problem3_problem4_l1322_132248


namespace propositions_truth_l1322_132282

theorem propositions_truth :
  (∀ a b : ℝ, a > b ∧ 1/a > 1/b → a*b < 0) ∧
  (∃ a b : ℝ, a < b ∧ b < 0 ∧ ¬(a^2 < a*b ∧ a*b < b^2)) ∧
  (∃ c a b : ℝ, c > a ∧ a > b ∧ b > 0 ∧ ¬(a/(c-a) < b/(c-b))) ∧
  (∀ a b c : ℝ, a > b ∧ b > c ∧ c > 0 → a/b > (a+c)/(b+c)) :=
by
  sorry

end propositions_truth_l1322_132282


namespace least_band_members_l1322_132211

/-- Represents the target ratio for each instrument -/
def target_ratio : Vector ℕ 5 := ⟨[5, 3, 6, 2, 4], by rfl⟩

/-- Represents the minimum number of successful candidates for each instrument -/
def min_candidates : Vector ℕ 5 := ⟨[16, 15, 20, 2, 12], by rfl⟩

/-- Checks if a given number of band members satisfies the target ratio and minimum requirements -/
def satisfies_requirements (total_members : ℕ) : Prop :=
  ∃ (x : ℕ), x > 0 ∧
    (∀ i : Fin 5, 
      (target_ratio.get i) * x ≥ min_candidates.get i) ∧
    (target_ratio.get 0) * x + 
    (target_ratio.get 1) * x + 
    (target_ratio.get 2) * x + 
    (target_ratio.get 3) * x + 
    (target_ratio.get 4) * x = total_members

/-- The main theorem stating that 100 is the least number of total band members satisfying the requirements -/
theorem least_band_members : 
  satisfies_requirements 100 ∧ 
  (∀ n : ℕ, n < 100 → ¬satisfies_requirements n) :=
sorry

end least_band_members_l1322_132211


namespace lcm_gcd_ratio_540_360_l1322_132213

theorem lcm_gcd_ratio_540_360 : Nat.lcm 540 360 / Nat.gcd 540 360 = 6 := by
  sorry

end lcm_gcd_ratio_540_360_l1322_132213


namespace germination_probability_l1322_132299

/-- The probability of exactly k successes in n independent Bernoulli trials -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The germination rate of seeds -/
def germination_rate : ℝ := 0.9

/-- The number of seeds sown -/
def total_seeds : ℕ := 7

/-- The number of seeds expected to germinate -/
def germinated_seeds : ℕ := 5

theorem germination_probability :
  binomial_probability total_seeds germinated_seeds germination_rate =
  21 * (germination_rate^5) * ((1 - germination_rate)^2) :=
by sorry

end germination_probability_l1322_132299


namespace max_blocks_fit_l1322_132203

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- Calculates the number of blocks that can fit in one layer of the larger box -/
def blocksPerLayer (largeBox : BoxDimensions) (smallBox : BoxDimensions) : ℕ :=
  (largeBox.length / smallBox.length) * (largeBox.width / smallBox.width)

/-- The main theorem stating the maximum number of blocks that can fit -/
theorem max_blocks_fit (largeBox smallBox : BoxDimensions) :
  largeBox = BoxDimensions.mk 5 4 4 →
  smallBox = BoxDimensions.mk 3 2 1 →
  blocksPerLayer largeBox smallBox * (largeBox.height / smallBox.height) = 12 :=
sorry

end max_blocks_fit_l1322_132203


namespace salary_distribution_l1322_132285

theorem salary_distribution (total : ℝ) :
  ∃ (a b c d : ℝ),
    a + b + c + d = total ∧
    2 * b = 3 * a ∧
    4 * b = 6 * a ∧
    3 * c = 4 * b ∧
    d = c + 700 ∧
    b = 1050 := by
  sorry

end salary_distribution_l1322_132285


namespace algebraic_manipulation_l1322_132287

theorem algebraic_manipulation (a b : ℝ) :
  (-2 * a^2 * b)^2 * (3 * a * b^2 - 5 * a^2 * b) / (-a * b)^3 = -12 * a^2 * b + 20 * a^3 := by
  sorry

end algebraic_manipulation_l1322_132287


namespace least_three_digit_multiple_l1322_132274

theorem least_three_digit_multiple : ∃ n : ℕ,
  (n ≥ 100 ∧ n < 1000) ∧
  2 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 3 ∣ n ∧
  ∀ m : ℕ, (m ≥ 100 ∧ m < 1000 ∧ 2 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 3 ∣ m) → n ≤ m :=
by
  use 210
  sorry

end least_three_digit_multiple_l1322_132274


namespace alternating_work_completion_work_fully_completed_l1322_132229

/-- Represents the number of days it takes to complete the work when A and B work on alternate days, starting with B. -/
def alternating_work_days (a_days b_days : ℕ) : ℕ :=
  2 * (9 * b_days * a_days) / (b_days + 3 * a_days)

/-- Theorem stating that if A can complete the work in 12 days and B in 36 days,
    working on alternate days starting with B will complete the work in 18 days. -/
theorem alternating_work_completion :
  alternating_work_days 12 36 = 18 := by
  sorry

/-- Proof that the work is fully completed after 18 days. -/
theorem work_fully_completed (a_days b_days : ℕ) 
  (ha : a_days = 12) (hb : b_days = 36) :
  (9 : ℚ) * (1 / b_days + 1 / a_days) = 1 := by
  sorry

end alternating_work_completion_work_fully_completed_l1322_132229


namespace system_solution_condition_l1322_132271

/-- The system of equations has at least one solution if and only if -|a| ≤ b ≤ √2|a| -/
theorem system_solution_condition (a b : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 = a^2 ∧ x + |y| = b) ↔ -|a| ≤ b ∧ b ≤ Real.sqrt 2 * |a| :=
by sorry

end system_solution_condition_l1322_132271


namespace passing_mark_is_160_l1322_132225

/-- Represents an exam with a total number of marks and a passing mark. -/
structure Exam where
  total : ℕ
  passing : ℕ

/-- The condition that a candidate scoring 40% fails by 40 marks -/
def condition1 (e : Exam) : Prop :=
  (40 * e.total) / 100 = e.passing - 40

/-- The condition that a candidate scoring 60% passes by 20 marks -/
def condition2 (e : Exam) : Prop :=
  (60 * e.total) / 100 = e.passing + 20

/-- Theorem stating that given the conditions, the passing mark is 160 -/
theorem passing_mark_is_160 (e : Exam) 
  (h1 : condition1 e) (h2 : condition2 e) : e.passing = 160 := by
  sorry


end passing_mark_is_160_l1322_132225


namespace at_least_three_babies_speak_l1322_132265

def probability_baby_speaks : ℚ := 2/5

def number_of_babies : ℕ := 6

def probability_at_least_three_speak : ℚ := 7120/15625

theorem at_least_three_babies_speak :
  probability_at_least_three_speak =
    1 - (Nat.choose number_of_babies 0 * (1 - probability_baby_speaks)^number_of_babies +
         Nat.choose number_of_babies 1 * probability_baby_speaks * (1 - probability_baby_speaks)^(number_of_babies - 1) +
         Nat.choose number_of_babies 2 * probability_baby_speaks^2 * (1 - probability_baby_speaks)^(number_of_babies - 2)) :=
by sorry

end at_least_three_babies_speak_l1322_132265


namespace candy_mixture_price_l1322_132281

/-- Given two types of candy mixed to produce a mixture with known total weight and value,
    prove that the price of the second candy is $4.30 per pound. -/
theorem candy_mixture_price (x : ℝ) :
  x > 0 ∧
  x + 6.25 = 10 ∧
  3.5 * x + 6.25 * 4.3 = 4 * 10 →
  4.3 = (4 * 10 - 3.5 * x) / 6.25 :=
by sorry

end candy_mixture_price_l1322_132281


namespace unique_solution_condition_l1322_132262

/-- The system of equations has exactly one solution if and only if 
    (3 - √5)/2 < t < (3 + √5)/2 -/
theorem unique_solution_condition (t : ℝ) : 
  (∃! x y z v : ℝ, x + y + z + v = 0 ∧ 
    (x*y + y*z + z*v) + t*(x*z + x*v + y*v) = 0) ↔ 
  ((3 - Real.sqrt 5) / 2 < t ∧ t < (3 + Real.sqrt 5) / 2) := by
sorry

end unique_solution_condition_l1322_132262


namespace sqrt_sum_equals_sqrt_of_sum_sqrt_l1322_132291

theorem sqrt_sum_equals_sqrt_of_sum_sqrt (a b : ℚ) :
  Real.sqrt a + Real.sqrt b = Real.sqrt (2 + Real.sqrt 3) ↔
  ({a, b} : Set ℚ) = {1/2, 3/2} :=
sorry

end sqrt_sum_equals_sqrt_of_sum_sqrt_l1322_132291


namespace m_range_l1322_132254

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → m * x^3 - x^2 + 4*x + 3 ≥ 0) → 
  m ∈ Set.Icc (-6) (-2) := by
sorry

end m_range_l1322_132254


namespace no_solution_for_coin_problem_l1322_132273

theorem no_solution_for_coin_problem : 
  ¬∃ (x y z : ℕ), x + y + z = 13 ∧ x + 3*y + 5*z = 200 := by
sorry

end no_solution_for_coin_problem_l1322_132273


namespace range_sum_bounds_l1322_132244

/-- The function f(x) = -2x^2 + 4x -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x

/-- The range of f is [m, n] -/
def m : ℝ := -6
def n : ℝ := 2

theorem range_sum_bounds :
  ∀ x, m ≤ f x ∧ f x ≤ n →
  0 ≤ m + n ∧ m + n ≤ 4 := by
  sorry

#check range_sum_bounds

end range_sum_bounds_l1322_132244


namespace simplify_fraction_l1322_132249

theorem simplify_fraction : (2^5 + 2^3) / (2^4 - 2^2) = 10 / 3 := by
  sorry

end simplify_fraction_l1322_132249


namespace equation_solution_l1322_132270

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 4 ∧ x₂ = -2 ∧ 
  (∀ x : ℝ, (x + 1) * (x - 3) = 5 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end equation_solution_l1322_132270


namespace sports_club_overlap_l1322_132269

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h1 : total = 35)
  (h2 : badminton = 15)
  (h3 : tennis = 18)
  (h4 : neither = 5) :
  badminton + tennis - (total - neither) = 3 :=
by sorry

end sports_club_overlap_l1322_132269


namespace point_C_satisfies_condition_l1322_132288

/-- Given points A(-2, 1) and B(1, 4) in the plane, prove that C(-1, 2) satisfies AC = (1/2)CB -/
theorem point_C_satisfies_condition :
  let A : ℝ × ℝ := (-2, 1)
  let B : ℝ × ℝ := (1, 4)
  let C : ℝ × ℝ := (-1, 2)
  (C.1 - A.1, C.2 - A.2) = (1/2 : ℝ) • (B.1 - C.1, B.2 - C.2) := by
  sorry

#check point_C_satisfies_condition

end point_C_satisfies_condition_l1322_132288


namespace smallest_number_problem_l1322_132260

theorem smallest_number_problem (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a ≤ b ∧ b ≤ c →
  b = 29 →
  c = b + 7 →
  (a + b + c) / 3 = 30 →
  a = 25 := by
sorry

end smallest_number_problem_l1322_132260


namespace middle_term_coefficient_2x_plus_1_power_8_l1322_132235

theorem middle_term_coefficient_2x_plus_1_power_8 :
  let n : ℕ := 8
  let k : ℕ := n / 2
  let coeff : ℕ := Nat.choose n k * (2^k)
  coeff = 1120 :=
by sorry

end middle_term_coefficient_2x_plus_1_power_8_l1322_132235


namespace georgia_yellow_buttons_l1322_132208

/-- The number of yellow buttons Georgia has -/
def yellow_buttons : ℕ := sorry

/-- The number of black buttons Georgia has -/
def black_buttons : ℕ := 2

/-- The number of green buttons Georgia has -/
def green_buttons : ℕ := 3

/-- The number of buttons Georgia gives to Mary -/
def buttons_given : ℕ := 4

/-- The number of buttons Georgia has left after giving buttons to Mary -/
def buttons_left : ℕ := 5

/-- Theorem stating that Georgia has 4 yellow buttons -/
theorem georgia_yellow_buttons : yellow_buttons = 4 := by
  sorry

end georgia_yellow_buttons_l1322_132208


namespace carrots_grown_proof_l1322_132218

/-- The number of carrots Sandy grew -/
def sandy_carrots : ℕ := 8

/-- The number of carrots Mary grew -/
def mary_carrots : ℕ := 6

/-- The total number of carrots grown by Sandy and Mary -/
def total_carrots : ℕ := sandy_carrots + mary_carrots

theorem carrots_grown_proof : total_carrots = 14 := by
  sorry

end carrots_grown_proof_l1322_132218


namespace digit_ratio_l1322_132242

/-- Given a 3-digit integer x with hundreds digit a, tens digit b, and units digit c,
    where a > 0 and the difference between the two greatest possible values of x is 241,
    prove that the ratio of b to a is 5:7. -/
theorem digit_ratio (x a b c : ℕ) : 
  (100 ≤ x) ∧ (x < 1000) ∧  -- x is a 3-digit integer
  (x = 100 * a + 10 * b + c) ∧  -- x is composed of digits a, b, c
  (a > 0) ∧  -- a is positive
  (999 - x = 241) →  -- difference between greatest possible value and x is 241
  (b : ℚ) / a = 5 / 7 := by
sorry

end digit_ratio_l1322_132242


namespace sum_of_two_numbers_l1322_132257

theorem sum_of_two_numbers (s l : ℝ) : 
  s = 10.0 → 
  7 * s = 5 * l → 
  s + l = 24.0 := by
sorry

end sum_of_two_numbers_l1322_132257


namespace sector_max_area_l1322_132237

/-- Given a sector of a circle with perimeter c (c > 0), 
    prove that the maximum area is c^2/16 and occurs when the arc length is c/2 -/
theorem sector_max_area (c : ℝ) (hc : c > 0) :
  let area (L : ℝ) := (c - L) * L / 4
  ∃ (L : ℝ), L = c / 2 ∧ 
    (∀ x, area x ≤ area L) ∧
    area L = c^2 / 16 := by
  sorry

end sector_max_area_l1322_132237


namespace min_value_abs_sum_min_value_abs_sum_achieved_l1322_132226

theorem min_value_abs_sum (x : ℝ) : |x - 1| + |x - 3| ≥ 2 := by sorry

theorem min_value_abs_sum_achieved : ∃ x : ℝ, |x - 1| + |x - 3| = 2 := by sorry

end min_value_abs_sum_min_value_abs_sum_achieved_l1322_132226


namespace abc_inequality_l1322_132204

theorem abc_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) : 
  a + b + c + 2 * a * b * c > a * b + b * c + c * a + 2 * Real.sqrt (a * b * c) := by
  sorry

end abc_inequality_l1322_132204


namespace mountaineer_arrangement_count_l1322_132252

/-- The number of ways to arrange mountaineers -/
def arrange_mountaineers (total : ℕ) (familiar : ℕ) (group_size : ℕ) : ℕ :=
  -- Number of ways to divide familiar mountaineers
  (familiar.choose (group_size / 2) * (familiar - group_size / 2).choose (group_size / 2) / 2) *
  -- Number of ways to divide unfamiliar mountaineers
  ((total - familiar).choose ((total - familiar) / 2) * ((total - familiar) / 2).choose ((total - familiar) / 2) / 2) *
  -- Number of ways to pair groups
  2 *
  -- Number of ways to order the groups
  2

/-- The theorem stating the number of arrangements for the given problem -/
theorem mountaineer_arrangement_count : 
  arrange_mountaineers 10 4 2 = 120 := by sorry

end mountaineer_arrangement_count_l1322_132252


namespace machine_work_rate_l1322_132247

theorem machine_work_rate (x : ℝ) : 
  (1 / (x + 4) + 1 / (x + 3) + 1 / (x + 2) = 1 / x) → x = 1 / 2 := by
  sorry

end machine_work_rate_l1322_132247


namespace largest_prime_factor_of_1001_l1322_132246

theorem largest_prime_factor_of_1001 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1001 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1001 → q ≤ p :=
by sorry

end largest_prime_factor_of_1001_l1322_132246


namespace max_ab_line_tangent_circle_l1322_132264

/-- The maximum value of ab when a line is tangent to a circle -/
theorem max_ab_line_tangent_circle (a b : ℝ) : 
  -- Line equation: x + 2y = 0
  -- Circle equation: (x-a)² + (y-b)² = 5
  -- Line is tangent to circle
  (∃ x y : ℝ, x + 2*y = 0 ∧ (x-a)^2 + (y-b)^2 = 5 ∧ 
    ∀ x' y' : ℝ, x' + 2*y' = 0 → (x'-a)^2 + (y'-b)^2 ≥ 5) →
  -- Center of circle is above the line
  a + 2*b > 0 →
  -- The maximum value of ab is 25/8
  a * b ≤ 25/8 :=
by sorry

end max_ab_line_tangent_circle_l1322_132264


namespace phyllis_marble_count_l1322_132212

/-- The number of groups of marbles in Phyllis's collection -/
def num_groups : ℕ := 32

/-- The number of marbles in each group -/
def marbles_per_group : ℕ := 2

/-- The total number of marbles in Phyllis's collection -/
def total_marbles : ℕ := num_groups * marbles_per_group

theorem phyllis_marble_count : total_marbles = 64 := by
  sorry

end phyllis_marble_count_l1322_132212


namespace geometric_progression_solution_l1322_132272

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
structure GeometricProgression where
  firstTerm : ℚ
  commonRatio : ℚ

/-- The n-th term of a geometric progression. -/
def nthTerm (gp : GeometricProgression) (n : ℕ) : ℚ :=
  gp.firstTerm * gp.commonRatio ^ (n - 1)

theorem geometric_progression_solution :
  ∃ (gp : GeometricProgression),
    nthTerm gp 2 = 37 + 1/3 ∧
    nthTerm gp 6 = 2 + 1/3 ∧
    gp.firstTerm = 224/3 ∧
    gp.commonRatio = 1/2 :=
by
  sorry

end geometric_progression_solution_l1322_132272


namespace factorization_3m_squared_minus_12_l1322_132215

theorem factorization_3m_squared_minus_12 (m : ℝ) : 3 * m^2 - 12 = 3 * (m + 2) * (m - 2) := by
  sorry

end factorization_3m_squared_minus_12_l1322_132215


namespace sin_45_minus_sin_15_l1322_132275

theorem sin_45_minus_sin_15 : 
  Real.sin (45 * π / 180) - Real.sin (15 * π / 180) = (3 * Real.sqrt 2 - Real.sqrt 6) / 4 := by
sorry

end sin_45_minus_sin_15_l1322_132275


namespace basic_computer_printer_price_l1322_132233

/-- The total price of a basic computer and printer, given specific conditions -/
theorem basic_computer_printer_price : ∃ (printer_price : ℝ),
  let basic_computer_price : ℝ := 2000
  let enhanced_computer_price : ℝ := basic_computer_price + 500
  let total_price : ℝ := basic_computer_price + printer_price
  printer_price = (1 / 6) * (enhanced_computer_price + printer_price) →
  total_price = 2500 := by
sorry

end basic_computer_printer_price_l1322_132233


namespace line_intersects_circle_l1322_132251

theorem line_intersects_circle (a : ℝ) (h : a ≠ 0) :
  let d := |2*a| / Real.sqrt (a^2 + 1)
  d < 3 := by sorry

end line_intersects_circle_l1322_132251


namespace tournament_prize_total_l1322_132266

def prize_money (first_place : ℕ) (interval : ℕ) : ℕ :=
  let second_place := first_place - interval
  let third_place := second_place - interval
  first_place + second_place + third_place

theorem tournament_prize_total :
  prize_money 2000 400 = 4800 :=
by sorry

end tournament_prize_total_l1322_132266


namespace least_six_digit_divisible_by_198_l1322_132293

theorem least_six_digit_divisible_by_198 : ∃ n : ℕ, 
  (n ≥ 100000 ∧ n < 1000000) ∧  -- 6-digit number condition
  n % 198 = 0 ∧                 -- divisibility condition
  ∀ m : ℕ, (m ≥ 100000 ∧ m < 1000000) ∧ m % 198 = 0 → n ≤ m :=
by
  -- The proof would go here
  sorry

end least_six_digit_divisible_by_198_l1322_132293


namespace tim_morning_run_hours_l1322_132207

/-- Tim's running schedule -/
structure RunningSchedule where
  runs_per_week : ℕ
  total_hours_per_week : ℕ
  morning_equals_evening : Bool

/-- Calculate the number of hours Tim runs in the morning each day -/
def morning_run_hours (schedule : RunningSchedule) : ℚ :=
  if schedule.morning_equals_evening then
    (schedule.total_hours_per_week : ℚ) / (2 * schedule.runs_per_week)
  else
    0

/-- Theorem: Tim runs 1 hour in the morning each day -/
theorem tim_morning_run_hours :
  let tims_schedule : RunningSchedule := {
    runs_per_week := 5,
    total_hours_per_week := 10,
    morning_equals_evening := true
  }
  morning_run_hours tims_schedule = 1 := by sorry

end tim_morning_run_hours_l1322_132207


namespace total_pumpkin_pies_l1322_132283

theorem total_pumpkin_pies (pinky helen emily jake : ℕ)
  (h1 : pinky = 147)
  (h2 : helen = 56)
  (h3 : emily = 89)
  (h4 : jake = 122) :
  pinky + helen + emily + jake = 414 := by
  sorry

end total_pumpkin_pies_l1322_132283
