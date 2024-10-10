import Mathlib

namespace train_overtake_time_l801_80170

/-- The time (in seconds) for a faster train to overtake a slower train after they meet -/
def overtake_time (v1 v2 l : ℚ) : ℚ :=
  (2 * l) / ((v2 - v1) / 3600)

theorem train_overtake_time :
  let v1 : ℚ := 50  -- speed of slower train (mph)
  let v2 : ℚ := 70  -- speed of faster train (mph)
  let l : ℚ := 1/6  -- length of each train (miles)
  overtake_time v1 v2 l = 60 := by
sorry

end train_overtake_time_l801_80170


namespace monotone_sine_range_l801_80132

/-- The function f(x) = 2sin(ωx) is monotonically increasing on [-π/4, 2π/3] if and only if ω is in (0, 3/4] -/
theorem monotone_sine_range (ω : ℝ) (h : ω > 0) :
  StrictMonoOn (fun x => 2 * Real.sin (ω * x)) (Set.Icc (-π/4) (2*π/3)) ↔ ω ∈ Set.Ioo 0 (3/4) ∪ {3/4} := by
  sorry

end monotone_sine_range_l801_80132


namespace distance_to_point_l801_80154

theorem distance_to_point : Real.sqrt ((-12 - 0)^2 + (16 - 0)^2) = 20 := by
  sorry

end distance_to_point_l801_80154


namespace final_state_l801_80199

/-- Represents the state of variables a, b, and c --/
structure State where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Executes the program statements and returns the final state --/
def execute : State := 
  let s1 : State := ⟨1, 2, 3⟩  -- Initial assignment: a=1, b=2, c=3
  let s2 : State := ⟨s1.a, s1.b, s1.b⟩  -- c = b
  let s3 : State := ⟨s2.a, s2.a, s2.c⟩  -- b = a
  ⟨s3.c, s3.b, s3.c⟩  -- a = c

/-- The theorem stating the final values of a, b, and c --/
theorem final_state : execute = ⟨2, 1, 2⟩ := by
  sorry


end final_state_l801_80199


namespace not_divisible_by_1000_power_minus_1_l801_80100

theorem not_divisible_by_1000_power_minus_1 (m : ℕ) :
  ¬(1000^m - 1 ∣ 1978^m - 1) := by
sorry

end not_divisible_by_1000_power_minus_1_l801_80100


namespace trigonometric_identity_l801_80155

theorem trigonometric_identity (α : ℝ) : 
  Real.cos (4 * α) + Real.cos (3 * α) = 2 * Real.cos ((7 * α) / 2) * Real.cos (α / 2) := by
  sorry

end trigonometric_identity_l801_80155


namespace range_of_a_l801_80136

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (|4*x - 3| ≤ 1 → x^2 - (2*a + 1)*x + a^2 + a ≤ 0) ∧
   ¬(∀ x : ℝ, x^2 - (2*a + 1)*x + a^2 + a ≤ 0 → |4*x - 3| ≤ 1)) →
  0 ≤ a ∧ a ≤ 1/2 := by
sorry

end range_of_a_l801_80136


namespace translation_theorem_l801_80107

def original_function (x : ℝ) : ℝ := (x - 2)^2 + 1

def translated_function (x : ℝ) : ℝ := original_function (x + 2) - 2

theorem translation_theorem :
  ∀ x : ℝ, translated_function x = x^2 - 1 :=
by
  sorry

end translation_theorem_l801_80107


namespace transitivity_of_greater_than_l801_80193

theorem transitivity_of_greater_than {a b c : ℝ} (h1 : a > b) (h2 : b > c) : a > c := by
  sorry

end transitivity_of_greater_than_l801_80193


namespace slope_condition_l801_80171

/-- The slope of a line with y-intercept (0, 8) that intersects the ellipse 4x^2 + 25y^2 = 100 -/
def slope_intersecting_line_ellipse (m : ℝ) : Prop :=
  ∃ x y : ℝ, 
    y = m * x + 8 ∧ 
    4 * x^2 + 25 * y^2 = 100

/-- Theorem stating the condition for the slope of the intersecting line -/
theorem slope_condition : 
  ∀ m : ℝ, slope_intersecting_line_ellipse m ↔ m^2 ≥ 3/77 :=
by sorry

end slope_condition_l801_80171


namespace f_4_equals_24_l801_80117

-- Define the function f recursively
def f : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * f n

-- State the theorem
theorem f_4_equals_24 : f 4 = 24 := by
  sorry

end f_4_equals_24_l801_80117


namespace product_expansion_l801_80178

theorem product_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 4) * (8 / x^2 - 5 * x^3) = 6 / x^2 - 15 * x^3 / 4 := by
  sorry

end product_expansion_l801_80178


namespace initial_men_count_l801_80158

/-- Proves that the initial number of men is 1000, given the conditions of the problem. -/
theorem initial_men_count (initial_days : ℝ) (joined_days : ℝ) (joined_men : ℕ) : 
  initial_days = 20 →
  joined_days = 16.67 →
  joined_men = 200 →
  (∃ (initial_men : ℕ), initial_men * initial_days = (initial_men + joined_men) * joined_days ∧ initial_men = 1000) :=
by
  sorry

end initial_men_count_l801_80158


namespace quadratic_sum_theorem_l801_80173

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The x-coordinate of the vertex of a quadratic function -/
def vertex_x (f : QuadraticFunction) : ℚ := -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
def vertex_y (f : QuadraticFunction) : ℚ := f.c - f.b^2 / (4 * f.a)

/-- Theorem: For a quadratic function with integer coefficients and vertex at (2, -3),
    the sum a + b - c equals -4 -/
theorem quadratic_sum_theorem (f : QuadraticFunction) 
  (h1 : vertex_x f = 2)
  (h2 : vertex_y f = -3) :
  f.a + f.b - f.c = -4 := by sorry

end quadratic_sum_theorem_l801_80173


namespace rectangle_opposite_sides_l801_80141

/-- A parallelogram is a quadrilateral with opposite sides parallel and equal. -/
structure Parallelogram where
  opposite_sides_parallel : Bool
  opposite_sides_equal : Bool

/-- A rectangle is a special case of a parallelogram with right angles. -/
structure Rectangle extends Parallelogram where
  right_angles : Bool

/-- Deductive reasoning is a method of logical reasoning that uses general rules to reach a specific conclusion. -/
def DeductiveReasoning : Prop := True

/-- The reasoning method used in the given statement. -/
def reasoning_method : Prop := DeductiveReasoning

theorem rectangle_opposite_sides (p : Parallelogram) (r : Rectangle) :
  p.opposite_sides_parallel ∧ p.opposite_sides_equal →
  r.opposite_sides_parallel ∧ r.opposite_sides_equal →
  reasoning_method := by sorry

end rectangle_opposite_sides_l801_80141


namespace midpoint_segments_equal_l801_80160

/-- A structure representing a rectangle with a circle intersection --/
structure RectangleWithCircle where
  /-- The rectangle --/
  rectangle : Set (ℝ × ℝ)
  /-- The circle --/
  circle : Set (ℝ × ℝ)
  /-- The four right triangles formed by the intersection --/
  triangles : Fin 4 → Set (ℝ × ℝ)
  /-- The midpoints of the hypotenuses of the triangles --/
  midpoints : Fin 4 → ℝ × ℝ

/-- The theorem stating that A₀C₀ = B₀D₀ --/
theorem midpoint_segments_equal (rc : RectangleWithCircle) :
  dist (rc.midpoints 0) (rc.midpoints 2) = dist (rc.midpoints 1) (rc.midpoints 3) :=
sorry

end midpoint_segments_equal_l801_80160


namespace probability_four_white_balls_l801_80164

/-- The number of white balls in the box -/
def white_balls : ℕ := 7

/-- The number of black balls in the box -/
def black_balls : ℕ := 8

/-- The total number of balls in the box -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of balls drawn -/
def drawn_balls : ℕ := 5

/-- The number of white balls we want to draw -/
def target_white : ℕ := 4

/-- The number of black balls we want to draw -/
def target_black : ℕ := drawn_balls - target_white

theorem probability_four_white_balls : 
  (Nat.choose white_balls target_white * Nat.choose black_balls target_black : ℚ) / 
  Nat.choose total_balls drawn_balls = 280 / 3003 := by
sorry

end probability_four_white_balls_l801_80164


namespace smallest_n_ending_same_as_n_squared_l801_80189

theorem smallest_n_ending_same_as_n_squared : 
  ∃ (N : ℕ), 
    N > 0 ∧ 
    (N % 1000 = N^2 % 1000) ∧ 
    (N ≥ 100) ∧
    (∀ (M : ℕ), M > 0 ∧ M < N → (M % 1000 ≠ M^2 % 1000 ∨ M < 100)) ∧ 
    N = 376 :=
by sorry

end smallest_n_ending_same_as_n_squared_l801_80189


namespace income_comparison_l801_80139

/-- Given that Mary's income is 60% more than Tim's income, and Tim's income is 20% less than Juan's income, 
    prove that Mary's income is 128% of Juan's income. -/
theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.8)
  (h2 : mary = tim * 1.6) : 
  mary = juan * 1.28 := by
  sorry

end income_comparison_l801_80139


namespace smart_car_competition_probability_l801_80114

/-- The probability of selecting exactly 4 girls when randomly choosing 10 people
    from a group of 15 people (7 girls and 8 boys) -/
theorem smart_car_competition_probability :
  let total_members : ℕ := 15
  let girls : ℕ := 7
  let boys : ℕ := total_members - girls
  let selected : ℕ := 10
  let prob_four_girls := (Nat.choose girls 4 * Nat.choose boys 6 : ℚ) / Nat.choose total_members selected
  prob_four_girls = (Nat.choose girls 4 * Nat.choose boys 6 : ℚ) / Nat.choose total_members selected :=
by sorry

end smart_car_competition_probability_l801_80114


namespace frog_jump_difference_l801_80110

/-- The frog's jump distance in inches -/
def frog_jump : ℕ := 39

/-- The grasshopper's jump distance in inches -/
def grasshopper_jump : ℕ := 17

/-- Theorem: The frog jumped 22 inches farther than the grasshopper -/
theorem frog_jump_difference : frog_jump - grasshopper_jump = 22 := by
  sorry

end frog_jump_difference_l801_80110


namespace floor_area_K_l801_80127

/-- The number of circles in the ring -/
def n : ℕ := 7

/-- The radius of the larger circle C -/
def R : ℝ := 35

/-- The radius of each of the n congruent circles -/
noncomputable def r : ℝ := R * (Real.sqrt (2 - 2 * Real.cos (2 * Real.pi / n))) / 2

/-- The area K of the region inside circle C and outside all n circles -/
noncomputable def K : ℝ := Real.pi * (R^2 - n * r^2)

theorem floor_area_K : ⌊K⌋ = 1476 := by sorry

end floor_area_K_l801_80127


namespace wholesale_cost_calculation_l801_80187

/-- The wholesale cost of a sleeping bag -/
def wholesale_cost : ℝ := 24.56

/-- The selling price of a sleeping bag -/
def selling_price : ℝ := 28

/-- The gross profit percentage -/
def profit_percentage : ℝ := 0.14

theorem wholesale_cost_calculation :
  selling_price = wholesale_cost * (1 + profit_percentage) := by
  sorry

end wholesale_cost_calculation_l801_80187


namespace mrs_flannery_muffins_count_l801_80186

/-- The number of muffins baked by Mrs. Brier's class -/
def mrs_brier_muffins : ℕ := 18

/-- The number of muffins baked by Mrs. MacAdams's class -/
def mrs_macadams_muffins : ℕ := 20

/-- The total number of muffins baked by all first grade classes -/
def total_muffins : ℕ := 55

/-- The number of muffins baked by Mrs. Flannery's class -/
def mrs_flannery_muffins : ℕ := total_muffins - (mrs_brier_muffins + mrs_macadams_muffins)

theorem mrs_flannery_muffins_count : mrs_flannery_muffins = 17 := by
  sorry

end mrs_flannery_muffins_count_l801_80186


namespace polyhedron_20_faces_l801_80111

/-- A polyhedron with triangular faces -/
structure Polyhedron where
  faces : ℕ
  vertices : ℕ
  edges : ℕ
  triangular_faces : faces * 3 = edges * 2
  euler_formula : vertices - edges + faces = 2

/-- Theorem: A polyhedron with 20 triangular faces has 12 vertices and 30 edges -/
theorem polyhedron_20_faces (P : Polyhedron) (h : P.faces = 20) : 
  P.vertices = 12 ∧ P.edges = 30 := by
  sorry

end polyhedron_20_faces_l801_80111


namespace xyz_value_l801_80188

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 37)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) :
  x * y * z = 26 / 3 := by
sorry

end xyz_value_l801_80188


namespace age_difference_l801_80190

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 16) : a - c = 16 := by
  sorry

end age_difference_l801_80190


namespace janes_stick_length_l801_80135

-- Define the lengths of the sticks and other quantities
def pat_stick_length : ℕ := 30
def covered_length : ℕ := 7
def feet_to_inches : ℕ := 12

-- Define the theorem
theorem janes_stick_length :
  let uncovered_length : ℕ := pat_stick_length - covered_length
  let sarahs_stick_length : ℕ := 2 * uncovered_length
  let janes_stick_length : ℕ := sarahs_stick_length - 2 * feet_to_inches
  janes_stick_length = 22 := by
sorry

end janes_stick_length_l801_80135


namespace triangle_side_length_l801_80183

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = Real.pi / 3)
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 := by
sorry

end triangle_side_length_l801_80183


namespace point_coordinates_on_terminal_side_l801_80148

/-- Given a point P on the terminal side of -π/4 with |OP| = 2, prove its coordinates are (√2, -√2) -/
theorem point_coordinates_on_terminal_side (P : ℝ × ℝ) :
  (P.1 = Real.sqrt 2 ∧ P.2 = -Real.sqrt 2) ↔
  (∃ (r : ℝ), r > 0 ∧ P.1 = r * Real.cos (-π/4) ∧ P.2 = r * Real.sin (-π/4) ∧ r^2 = P.1^2 + P.2^2 ∧ r = 2) :=
by sorry

end point_coordinates_on_terminal_side_l801_80148


namespace quadratic_function_theorem_l801_80121

/-- A quadratic function satisfying certain conditions -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem statement -/
theorem quadratic_function_theorem (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, f a b c (x - 4) = f a b c (2 - x)) →
  (∀ x : ℝ, f a b c x ≥ x) →
  (∀ x ∈ Set.Ioo 0 2, f a b c x ≤ ((x + 1) / 2)^2) →
  (∃ x : ℝ, ∀ y : ℝ, f a b c x ≤ f a b c y) →
  (∃ x : ℝ, f a b c x = 0) →
  (∃ m : ℝ, m > 1 ∧ 
    (∀ m' : ℝ, m' > m → 
      ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f a b c (x + t) ≤ x) ∧
    (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x)) ∧
  (∀ m : ℝ, (m > 1 ∧ 
    (∀ m' : ℝ, m' > m → 
      ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f a b c (x + t) ≤ x) ∧
    (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x)) → m = 9) :=
by sorry

end quadratic_function_theorem_l801_80121


namespace alloy_gold_percentage_l801_80166

-- Define the weights and percentages
def total_weight : ℝ := 12.4
def metal_weight : ℝ := 6.2
def gold_percent_1 : ℝ := 0.60
def gold_percent_2 : ℝ := 0.40

-- Theorem statement
theorem alloy_gold_percentage :
  (metal_weight * gold_percent_1 + metal_weight * gold_percent_2) / total_weight = 0.50 := by
  sorry

end alloy_gold_percentage_l801_80166


namespace union_equals_A_l801_80119

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - a = 0}

-- State the theorem
theorem union_equals_A (a : ℝ) : (A ∪ B a = A) ↔ a < 0 := by
  sorry

end union_equals_A_l801_80119


namespace inequality_preserved_division_l801_80191

theorem inequality_preserved_division (x y a : ℝ) (h : x > y) :
  x / (a^2 + 1) > y / (a^2 + 1) := by sorry

end inequality_preserved_division_l801_80191


namespace height_comparison_l801_80167

theorem height_comparison (h_a h_b h_c : ℝ) 
  (h_a_def : h_a = 0.6 * h_b) 
  (h_c_def : h_c = 1.25 * h_a) : 
  (h_b - h_a) / h_a = 2/3 ∧ (h_c - h_a) / h_a = 1/4 := by
  sorry

end height_comparison_l801_80167


namespace ball_bearing_savings_ball_bearing_savings_correct_l801_80145

/-- Calculates the savings when buying ball bearings during a sale with a bulk discount -/
theorem ball_bearing_savings
  (num_machines : ℕ)
  (bearings_per_machine : ℕ)
  (regular_price : ℚ)
  (sale_price : ℚ)
  (bulk_discount : ℚ)
  (h1 : num_machines = 10)
  (h2 : bearings_per_machine = 30)
  (h3 : regular_price = 1)
  (h4 : sale_price = 3/4)
  (h5 : bulk_discount = 1/5)
  : ℚ :=
  let total_bearings := num_machines * bearings_per_machine
  let regular_cost := total_bearings * regular_price
  let sale_cost := total_bearings * sale_price
  let discounted_cost := sale_cost * (1 - bulk_discount)
  let savings := regular_cost - discounted_cost
  120

theorem ball_bearing_savings_correct : ball_bearing_savings 10 30 1 (3/4) (1/5) rfl rfl rfl rfl rfl = 120 := by
  sorry

end ball_bearing_savings_ball_bearing_savings_correct_l801_80145


namespace inscribed_square_area_l801_80106

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- A point is on the parabola if its y-coordinate equals f(x) -/
def on_parabola (p : ℝ × ℝ) : Prop := p.2 = f p.1

/-- A point is on the x-axis if its y-coordinate is 0 -/
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- A square is inscribed if its top vertices are on the parabola and bottom vertices are on the x-axis -/
def is_inscribed_square (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ), a < b ∧
    s = {(a, 0), (b, 0), (a, b-a), (b, b-a)} ∧
    on_parabola (a, b-a) ∧ on_parabola (b, b-a)

/-- The area of a square with side length s -/
def square_area (s : ℝ) : ℝ := s^2

theorem inscribed_square_area :
  ∀ s : Set (ℝ × ℝ), is_inscribed_square s → ∃ a : ℝ, square_area a = (3 - Real.sqrt 5) / 2 :=
sorry

end inscribed_square_area_l801_80106


namespace solve_equation_l801_80168

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.05) : x = 0.18 := by
  sorry

end solve_equation_l801_80168


namespace count_integers_satisfying_conditions_l801_80147

theorem count_integers_satisfying_conditions :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ 9 ∣ n ∧ Nat.lcm (Nat.factorial 6) n = 9 * Nat.gcd (Nat.factorial 9) n) ∧
    S.card = 30 ∧
    (∀ n : ℕ, n > 0 → 9 ∣ n → Nat.lcm (Nat.factorial 6) n = 9 * Nat.gcd (Nat.factorial 9) n → n ∈ S) :=
by sorry

end count_integers_satisfying_conditions_l801_80147


namespace union_of_A_and_B_l801_80109

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-1, 0, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by sorry

end union_of_A_and_B_l801_80109


namespace sum_of_roots_quadratic_l801_80197

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 - 5*x + 5 = 16) → (∃ y : ℝ, y^2 - 5*y + 5 = 16 ∧ x + y = 5) := by
  sorry

end sum_of_roots_quadratic_l801_80197


namespace square_perimeter_l801_80123

/-- The perimeter of a square with side length 13 centimeters is 52 centimeters. -/
theorem square_perimeter : ∀ (s : ℝ), s = 13 → 4 * s = 52 := by
  sorry

end square_perimeter_l801_80123


namespace distance_between_points_is_2_5_km_l801_80198

/-- Represents the running scenario with given parameters -/
structure RunningScenario where
  initialStandingTime : Real
  constantRunningRate : Real
  averageRate1 : Real
  averageRate2 : Real

/-- Calculates the distance run between two average rate points -/
def distanceBetweenPoints (scenario : RunningScenario) : Real :=
  sorry

/-- Theorem stating the distance run between the two average rate points -/
theorem distance_between_points_is_2_5_km (scenario : RunningScenario) 
  (h1 : scenario.initialStandingTime = 15 / 60) -- 15 seconds in minutes
  (h2 : scenario.constantRunningRate = 7)
  (h3 : scenario.averageRate1 = 7.5)
  (h4 : scenario.averageRate2 = 85 / 12) : -- 7 minutes 5 seconds in minutes
  distanceBetweenPoints scenario = 2.5 :=
  sorry

#check distance_between_points_is_2_5_km

end distance_between_points_is_2_5_km_l801_80198


namespace square_root_of_nine_l801_80184

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {-3, 3} := by sorry

end square_root_of_nine_l801_80184


namespace triangle_angle_calculation_l801_80152

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) : 
  C = π / 4 →  -- 45° in radians
  c = Real.sqrt 2 → 
  a = Real.sqrt 3 → 
  (A = π / 3 ∨ A = 2 * π / 3) -- 60° or 120° in radians
  :=
by sorry

end triangle_angle_calculation_l801_80152


namespace billy_coins_l801_80192

theorem billy_coins (quarter_piles dime_piles coins_per_pile : ℕ) 
  (h1 : quarter_piles = 2)
  (h2 : dime_piles = 3)
  (h3 : coins_per_pile = 4) :
  quarter_piles * coins_per_pile + dime_piles * coins_per_pile = 20 :=
by sorry

end billy_coins_l801_80192


namespace line_slope_l801_80162

theorem line_slope (t : ℝ) : 
  let x := 3 - (Real.sqrt 3 / 2) * t
  let y := 1 + (1 / 2) * t
  (y - 1) / (x - 3) = -Real.sqrt 3 / 3 :=
by sorry

end line_slope_l801_80162


namespace sum_bound_l801_80144

theorem sum_bound (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (h_sum : a + b + c = 1) :
  let S := 1 / (1 + a) + 1 / (1 + b) + 1 / (1 + c)
  9 / 4 ≤ S ∧ S ≤ 5 / 2 := by
  sorry

end sum_bound_l801_80144


namespace g_sum_symmetric_l801_80176

-- Define the function g
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^8 + e * x^6 - f * x^4 + 5

-- Theorem statement
theorem g_sum_symmetric (d e f : ℝ) :
  (∃ x, g d e f x = 7) → g d e f 2 + g d e f (-2) = 14 := by
  sorry

end g_sum_symmetric_l801_80176


namespace supporting_pillars_concrete_l801_80157

/-- The amount of concrete needed for a bridge construction --/
structure BridgeConcrete where
  roadwayDeck : ℕ
  oneAnchor : ℕ
  totalBridge : ℕ

/-- Calculates the amount of concrete needed for supporting pillars --/
def supportingPillarsAmount (b : BridgeConcrete) : ℕ :=
  b.totalBridge - (b.roadwayDeck + 2 * b.oneAnchor)

/-- Theorem stating the amount of concrete needed for supporting pillars --/
theorem supporting_pillars_concrete (b : BridgeConcrete) 
  (h1 : b.roadwayDeck = 1600)
  (h2 : b.oneAnchor = 700)
  (h3 : b.totalBridge = 4800) :
  supportingPillarsAmount b = 1800 := by
  sorry

#eval supportingPillarsAmount ⟨1600, 700, 4800⟩

end supporting_pillars_concrete_l801_80157


namespace spring_mass_for_length_30_l801_80142

def spring_length (mass : ℝ) : ℝ := 18 + 2 * mass

theorem spring_mass_for_length_30 :
  ∃ (mass : ℝ), spring_length mass = 30 ∧ mass = 6 :=
by sorry

end spring_mass_for_length_30_l801_80142


namespace max_ab_value_l801_80128

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃! x, x^2 + Real.sqrt a * x - b + 1/4 = 0) → 
  ∀ c, a * b ≤ c → c ≤ 1/16 :=
sorry

end max_ab_value_l801_80128


namespace book_ratio_is_three_l801_80169

/-- The number of books read last week -/
def books_last_week : ℕ := 5

/-- The number of pages in each book -/
def pages_per_book : ℕ := 300

/-- The total number of pages read this week -/
def pages_this_week : ℕ := 4500

/-- The ratio of books read this week to books read last week -/
def book_ratio : ℚ := (pages_this_week / pages_per_book) / books_last_week

theorem book_ratio_is_three : book_ratio = 3 := by
  sorry

end book_ratio_is_three_l801_80169


namespace deepak_age_l801_80194

theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 2 →
  rahul_age + 10 = 26 →
  deepak_age = 8 := by
sorry

end deepak_age_l801_80194


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_is_15_l801_80195

/-- An isosceles triangle with side lengths 3, 6, and 6 has a perimeter of 15 -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c p =>
    (a = 3 ∧ b = 6 ∧ c = 6) →  -- Two sides are 6, one side is 3
    (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
    (b = c) →  -- Isosceles condition
    (p = a + b + c) →  -- Definition of perimeter
    p = 15

theorem isosceles_triangle_perimeter_is_15 : 
  ∃ (a b c p : ℝ), isosceles_triangle_perimeter a b c p :=
sorry

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_is_15_l801_80195


namespace worker_travel_time_l801_80180

theorem worker_travel_time (T : ℝ) 
  (h1 : T > 0) 
  (h2 : (3/4 : ℝ) * T * (T + 12) = T * T) : 
  T = 36 := by
sorry

end worker_travel_time_l801_80180


namespace monotonic_increasing_sequence_l801_80130

/-- A sequence {a_n} with general term a_n = n^2 + bn is monotonically increasing if and only if b > -3 -/
theorem monotonic_increasing_sequence (b : ℝ) :
  (∀ n : ℕ, (n : ℝ)^2 + b * n < ((n + 1) : ℝ)^2 + b * (n + 1)) ↔ b > -3 :=
by sorry

end monotonic_increasing_sequence_l801_80130


namespace max_stamps_problem_l801_80137

/-- The maximum number of stamps that can be bought -/
def max_stamps (initial_money : ℕ) (bus_ticket_cost : ℕ) (stamp_price : ℕ) : ℕ :=
  ((initial_money * 100 - bus_ticket_cost) / stamp_price : ℕ)

/-- Theorem: Given $50 initial money, 180 cents bus ticket cost, and 45 cents stamp price,
    the maximum number of stamps that can be bought is 107 -/
theorem max_stamps_problem : max_stamps 50 180 45 = 107 := by
  sorry

end max_stamps_problem_l801_80137


namespace triangle_segment_length_l801_80124

/-- Given a triangle ABC with point D on AC and point E on AD, prove that FC = 10.125 -/
theorem triangle_segment_length 
  (DC CB : ℝ) 
  (h_DC : DC = 9)
  (h_CB : CB = 6)
  (AD AB ED : ℝ)
  (h_AB : AB = (1/3) * AD)
  (h_ED : ED = (3/4) * AD)
  : ∃ (FC : ℝ), FC = 10.125 := by
  sorry

end triangle_segment_length_l801_80124


namespace inverse_g_solution_l801_80104

noncomputable section

variables (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0)

def g (x : ℝ) : ℝ := 1 / (c * x + d)

theorem inverse_g_solution :
  let x := 1 / (-2 * c + d)
  g x = 1 / 2 := by sorry

end

end inverse_g_solution_l801_80104


namespace jays_savings_l801_80172

def savings_sequence (n : ℕ) : ℕ := 20 + 10 * n

def total_savings (weeks : ℕ) : ℕ :=
  (List.range weeks).map savings_sequence |> List.sum

theorem jays_savings : total_savings 4 = 140 := by
  sorry

end jays_savings_l801_80172


namespace closest_fraction_l801_80129

def medals_won : ℚ := 28 / 150

def options : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]

theorem closest_fraction : 
  ∃ (closest : ℚ), closest ∈ options ∧ 
  ∀ (x : ℚ), x ∈ options → |medals_won - closest| ≤ |medals_won - x| ∧
  closest = 1/5 := by sorry

end closest_fraction_l801_80129


namespace divisor_of_a_l801_80156

theorem divisor_of_a (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 24)
  (h2 : Nat.gcd b c = 36)
  (h3 : Nat.gcd c d = 54)
  (h4 : 70 < Nat.gcd d a ∧ Nat.gcd d a < 100) :
  13 ∣ a := by
  sorry

end divisor_of_a_l801_80156


namespace line_equation_transformation_l801_80125

/-- Given a line l: Ax + By + C = 0 and a point (x₀, y₀) on the line,
    prove that the line equation can be transformed to A(x - x₀) + B(y - y₀) = 0 -/
theorem line_equation_transformation 
  (A B C x₀ y₀ : ℝ) 
  (h1 : A ≠ 0 ∨ B ≠ 0) 
  (h2 : A * x₀ + B * y₀ + C = 0) :
  ∀ x y, A * x + B * y + C = 0 ↔ A * (x - x₀) + B * (y - y₀) = 0 :=
sorry

end line_equation_transformation_l801_80125


namespace heather_walk_distance_l801_80102

/-- The total distance Heather walked at the county fair -/
def total_distance (d1 d2 d3 : ℝ) : ℝ := d1 + d2 + d3

/-- Theorem stating the total distance Heather walked -/
theorem heather_walk_distance :
  let d1 : ℝ := 0.33  -- Distance from car to entrance
  let d2 : ℝ := 0.33  -- Distance to carnival rides
  let d3 : ℝ := 0.08  -- Distance from carnival rides back to car
  total_distance d1 d2 d3 = 0.74 := by
  sorry

end heather_walk_distance_l801_80102


namespace M_intersect_P_eq_y_geq_1_l801_80134

-- Define the sets M and P
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def P : Set ℝ := {y | ∃ x : ℝ, y = Real.log x}

-- State the theorem
theorem M_intersect_P_eq_y_geq_1 : M ∩ P = {y : ℝ | y ≥ 1} := by sorry

end M_intersect_P_eq_y_geq_1_l801_80134


namespace solution_set_of_inequality_l801_80140

noncomputable def f (x : ℝ) := Real.sin x - x

theorem solution_set_of_inequality (x : ℝ) :
  f (x + 2) + f (1 - 2*x) < 0 ↔ x < 3 :=
sorry

end solution_set_of_inequality_l801_80140


namespace not_necessarily_equal_distances_l801_80177

-- Define the points A_i and B_i in 3D space
variable (A B : ℕ → ℝ × ℝ × ℝ)

-- Define the radius of a circumscribed circle for a triangle
def circumradius (p q r : ℝ × ℝ × ℝ) : ℝ := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ × ℝ) : ℝ := sorry

-- A_1 is the circumcenter of a triangle
axiom A1_is_circumcenter : ∃ (p q r : ℝ × ℝ × ℝ), circumradius (A 1) p q = distance (A 1) p

-- The radii of circumscribed circles of triangles A_iA_jA_k and B_iB_jB_k are equal for any i, j, k
axiom equal_circumradii : ∀ (i j k : ℕ), circumradius (A i) (A j) (A k) = circumradius (B i) (B j) (B k)

-- The theorem to be proved
theorem not_necessarily_equal_distances :
  ¬(∀ (i j : ℕ), distance (A i) (A j) = distance (B i) (B j)) :=
sorry

end not_necessarily_equal_distances_l801_80177


namespace solve_for_m_l801_80149

theorem solve_for_m : ∃ m : ℤ, 5^2 + 7 = 4^3 + m ∧ m = -32 := by
  sorry

end solve_for_m_l801_80149


namespace trigonometric_expression_equals_one_l801_80182

theorem trigonometric_expression_equals_one : 
  (Real.sin (15 * π / 180) * Real.cos (25 * π / 180) + Real.cos (165 * π / 180) * Real.cos (115 * π / 180)) / 
  (Real.sin (35 * π / 180) * Real.cos (5 * π / 180) + Real.cos (145 * π / 180) * Real.cos (85 * π / 180)) = 1 := by
  sorry

end trigonometric_expression_equals_one_l801_80182


namespace part_one_part_two_l801_80163

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + a^2 - 2 * a + 2

-- Part 1
theorem part_one :
  let a := 2
  (∀ x ∈ Set.Icc (-1) 3, f a x ≥ -2) ∧
  (∃ x ∈ Set.Icc (-1) 3, f a x = -2) ∧
  (∀ x ∈ Set.Icc (-1) 3, f a x ≤ 14) ∧
  (∃ x ∈ Set.Icc (-1) 3, f a x = 14) :=
sorry

-- Part 2
theorem part_two :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 2, f a x ≥ 3) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 3) ↔
  a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10 :=
sorry

end part_one_part_two_l801_80163


namespace min_value_problem_l801_80161

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (x^2 + y^2 + x) / (x*y) ≥ 2*Real.sqrt 2 + 2 :=
by sorry

end min_value_problem_l801_80161


namespace train_meeting_distance_l801_80118

/-- Represents the distance traveled by a train given its speed and time -/
def distanceTraveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents the total initial distance between the trains -/
def totalDistance : ℝ := 350

/-- Represents the speed of Train A in miles per hour -/
def speedA : ℝ := 40

/-- Represents the speed of Train B in miles per hour -/
def speedB : ℝ := 30

/-- Theorem stating that Train A will have traveled 200 miles when the trains meet -/
theorem train_meeting_distance :
  ∃ (t : ℝ), t > 0 ∧ 
  distanceTraveled speedA t + distanceTraveled speedB t = totalDistance ∧
  distanceTraveled speedA t = 200 := by
  sorry

end train_meeting_distance_l801_80118


namespace alice_age_problem_l801_80105

theorem alice_age_problem :
  ∃! x : ℕ+, 
    (∃ n : ℕ+, (x : ℤ) - 4 = n^2) ∧ 
    (∃ m : ℕ+, (x : ℤ) + 2 = m^3) ∧ 
    x = 58 := by
  sorry

end alice_age_problem_l801_80105


namespace greatest_power_of_three_specific_case_l801_80112

theorem greatest_power_of_three (n : ℕ) : ∃ (k : ℕ), (3^n : ℤ) ∣ (6^n - 3^n) ∧ ¬(3^(n+1) : ℤ) ∣ (6^n - 3^n) :=
by
  sorry

theorem specific_case : ∃ (k : ℕ), (3^1503 : ℤ) ∣ (6^1503 - 3^1503) ∧ ¬(3^1504 : ℤ) ∣ (6^1503 - 3^1503) :=
by
  sorry

end greatest_power_of_three_specific_case_l801_80112


namespace max_guaranteed_rectangle_area_l801_80174

/-- Represents a chessboard with some squares removed -/
structure Chessboard :=
  (size : Nat)
  (removed : Finset (Nat × Nat))

/-- Represents a rectangle on the chessboard -/
structure Rectangle :=
  (top_left : Nat × Nat)
  (width : Nat)
  (height : Nat)

/-- Check if a rectangle fits on the chessboard without overlapping removed squares -/
def Rectangle.fits (board : Chessboard) (rect : Rectangle) : Prop :=
  rect.top_left.1 + rect.width ≤ board.size ∧
  rect.top_left.2 + rect.height ≤ board.size ∧
  ∀ x y, rect.top_left.1 ≤ x ∧ x < rect.top_left.1 + rect.width ∧
         rect.top_left.2 ≤ y ∧ y < rect.top_left.2 + rect.height →
         (x, y) ∉ board.removed

/-- The main theorem -/
theorem max_guaranteed_rectangle_area (board : Chessboard) 
  (h1 : board.size = 8) 
  (h2 : board.removed.card = 8) : 
  (∀ n > 8, ∃ rect : Rectangle, rect.width * rect.height = n → ¬rect.fits board) ∧ 
  (∃ rect : Rectangle, rect.width * rect.height = 8 ∧ rect.fits board) :=
sorry

end max_guaranteed_rectangle_area_l801_80174


namespace min_value_expression_l801_80138

theorem min_value_expression (x : ℝ) : 
  (∃ (m : ℝ), ∀ (y : ℝ), (15 - y) * (13 - y) * (15 + y) * (13 + y) + 200 * y^2 ≥ m) ∧ 
  (∃ (z : ℝ), (15 - z) * (13 - z) * (15 + z) * (13 + z) + 200 * z^2 = 33) :=
by sorry

end min_value_expression_l801_80138


namespace no_solution_equation_l801_80133

theorem no_solution_equation :
  ¬ ∃ x : ℝ, (x + 2) / (x - 2) - x / (x + 2) = 16 / (x^2 - 4) :=
by sorry

end no_solution_equation_l801_80133


namespace perfect_square_condition_l801_80150

/-- The polynomial in question -/
def P (x m : ℝ) : ℝ := (x-1)*(x+3)*(x-4)*(x-8) + m

/-- The polynomial is a perfect square -/
def is_perfect_square (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, ∀ x, f x = (g x)^2

theorem perfect_square_condition :
  ∃! m : ℝ, is_perfect_square (P · m) ∧ m = 196 :=
sorry

end perfect_square_condition_l801_80150


namespace smallest_six_digit_divisible_by_111_l801_80122

theorem smallest_six_digit_divisible_by_111 : ∃ n : ℕ, 
  (n ≥ 100000 ∧ n < 1000000) ∧ 
  n % 111 = 0 ∧
  ∀ m : ℕ, (m ≥ 100000 ∧ m < 1000000) ∧ m % 111 = 0 → n ≤ m :=
by sorry

end smallest_six_digit_divisible_by_111_l801_80122


namespace transform_to_successor_l801_80175

/-- Represents the allowed operations on natural numbers -/
inductive Operation
  | AddNine
  | EraseOne

/-- Applies a single operation to a natural number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.AddNine => n + 9
  | Operation.EraseOne => sorry  -- Implementation of erasing 1 is complex and not provided

/-- Applies a sequence of operations to a natural number -/
def applyOperations (n : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation n

/-- 
Theorem: For any natural number A, there exists a sequence of operations 
that transforms A into A+1
-/
theorem transform_to_successor (A : ℕ) : 
  ∃ (ops : List Operation), applyOperations A ops = A + 1 :=
sorry

end transform_to_successor_l801_80175


namespace monomial_product_l801_80115

theorem monomial_product (a b : ℤ) (x y : ℝ) (h1 : 4 * a - b = 2) (h2 : a + b = 3) :
  (-2 * x^(4*a-b) * y^3) * ((1/2) * x^2 * y^(a+b)) = -x^4 * y^6 := by
  sorry

end monomial_product_l801_80115


namespace family_movie_night_l801_80143

/-- Calculates the number of children in a family given ticket prices and payment information. -/
def number_of_children (regular_ticket_price : ℕ) (child_discount : ℕ) (total_payment : ℕ) (change : ℕ) (num_adults : ℕ) : ℕ :=
  let child_ticket_price := regular_ticket_price - child_discount
  let total_spent := total_payment - change
  let adult_tickets_cost := regular_ticket_price * num_adults
  let children_tickets_cost := total_spent - adult_tickets_cost
  children_tickets_cost / child_ticket_price

/-- Proves that the number of children in the family is 3 given the problem conditions. -/
theorem family_movie_night : number_of_children 9 2 40 1 2 = 3 := by
  sorry

end family_movie_night_l801_80143


namespace revenue_decrease_percent_l801_80179

/-- Calculates the decrease percent in revenue when tax is reduced and consumption is increased -/
theorem revenue_decrease_percent 
  (original_tax : ℝ) 
  (original_consumption : ℝ) 
  (tax_reduction_percent : ℝ) 
  (consumption_increase_percent : ℝ) 
  (h1 : tax_reduction_percent = 22) 
  (h2 : consumption_increase_percent = 9) 
  : (1 - (1 - tax_reduction_percent / 100) * (1 + consumption_increase_percent / 100)) * 100 = 15.02 := by
  sorry

end revenue_decrease_percent_l801_80179


namespace integer_equation_proof_l801_80196

theorem integer_equation_proof (m n : ℤ) (h : 3 * m * n + 3 * m = n + 2) : 3 * m + n = -2 := by
  sorry

end integer_equation_proof_l801_80196


namespace sum_of_differences_base7_l801_80120

/-- Converts a base 7 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 7 * acc) 0

/-- Converts a decimal number to its base 7 representation as a list of digits -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec go (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else go (m / 7) ((m % 7) :: acc)
  go n []

/-- Calculates the difference between two base 7 numbers -/
def diffBase7 (a b : List Nat) : List Nat :=
  toBase7 (toDecimal a - toDecimal b)

/-- Calculates the sum of two base 7 numbers -/
def sumBase7 (a b : List Nat) : List Nat :=
  toBase7 (toDecimal a + toDecimal b)

theorem sum_of_differences_base7 :
  let a := [5, 2, 4, 3]
  let b := [3, 1, 0, 5]
  let c := [6, 6, 6, 5]
  let d := [4, 3, 1, 2]
  let result := [4, 4, 5, 2]
  sumBase7 (diffBase7 a b) (diffBase7 c d) = result :=
by sorry

end sum_of_differences_base7_l801_80120


namespace rearrangement_theorem_l801_80151

/-- The number of ways to choose and rearrange 3 people from a group of 7 -/
def rearrangement_count : ℕ := 70

/-- The number of people in the class -/
def class_size : ℕ := 7

/-- The number of people to be rearranged -/
def rearrange_size : ℕ := 3

/-- The number of ways to derange 3 people -/
def derangement_3 : ℕ := 2

theorem rearrangement_theorem : 
  rearrangement_count = derangement_3 * (class_size.choose rearrange_size) := by
  sorry

end rearrangement_theorem_l801_80151


namespace quadratic_common_root_l801_80159

theorem quadratic_common_root (p1 p2 q1 q2 : ℂ) :
  (∃ x : ℂ, x^2 + p1*x + q1 = 0 ∧ x^2 + p2*x + q2 = 0) ↔
  (q2 - q1)^2 + (p1 - p2)*(p1*q2 - q1*p2) = 0 := by
  sorry

end quadratic_common_root_l801_80159


namespace parallelogram_revolution_surface_area_l801_80101

/-- The surface area of a solid of revolution formed by rotating a parallelogram -/
theorem parallelogram_revolution_surface_area
  (p d : ℝ)
  (perimeter_positive : p > 0)
  (diagonal_positive : d > 0) :
  let perimeter := 2 * p
  let diagonal := d
  let surface_area := 2 * Real.pi * d * p
  surface_area = 2 * Real.pi * diagonal * (perimeter / 2) :=
sorry

end parallelogram_revolution_surface_area_l801_80101


namespace candy_mixture_cost_per_pound_l801_80131

/-- Calculates the desired cost per pound of a candy mixture --/
theorem candy_mixture_cost_per_pound 
  (weight_expensive : ℝ) 
  (price_expensive : ℝ) 
  (weight_cheap : ℝ) 
  (price_cheap : ℝ) 
  (h1 : weight_expensive = 20) 
  (h2 : price_expensive = 10) 
  (h3 : weight_cheap = 80) 
  (h4 : price_cheap = 5) : 
  (weight_expensive * price_expensive + weight_cheap * price_cheap) / (weight_expensive + weight_cheap) = 6 := by
  sorry

end candy_mixture_cost_per_pound_l801_80131


namespace optimal_station_is_75km_l801_80108

/-- Represents a petrol station with its distance from a given point --/
structure PetrolStation :=
  (distance : ℝ)

/-- Represents a car with its fuel consumption rate --/
structure Car :=
  (consumption : ℝ)  -- litres per km

/-- Represents a journey with various parameters --/
structure Journey :=
  (totalDistance : ℝ)
  (initialFuel : ℝ)
  (initialDriven : ℝ)
  (stations : List PetrolStation)
  (tankCapacity : ℝ)

def Journey.optimalStation (j : Journey) (c : Car) : Option PetrolStation :=
  sorry

theorem optimal_station_is_75km 
  (j : Journey)
  (c : Car)
  (h1 : j.totalDistance = 520)
  (h2 : j.initialFuel = 14)
  (h3 : c.consumption = 0.1)
  (h4 : j.initialDriven = 55)
  (h5 : j.stations = [
    { distance := 35 },
    { distance := 45 },
    { distance := 55 },
    { distance := 75 },
    { distance := 95 }
  ])
  (h6 : j.tankCapacity = 40) :
  (Journey.optimalStation j c).map PetrolStation.distance = some 75 := by
  sorry

end optimal_station_is_75km_l801_80108


namespace particular_number_proof_l801_80146

theorem particular_number_proof (x : ℚ) : x / 4 + 3 = 5 → x = 8 := by
  sorry

end particular_number_proof_l801_80146


namespace minimum_value_theorem_l801_80113

theorem minimum_value_theorem (x : ℝ) (h : x > 4) :
  (x + 11) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 15 ∧
  (∃ x₀ > 4, (x₀ + 11) / Real.sqrt (x₀ - 4) = 2 * Real.sqrt 15 ∧ x₀ = 19) :=
by sorry

end minimum_value_theorem_l801_80113


namespace green_fish_count_l801_80165

theorem green_fish_count (T : ℕ) : ℕ := by
  -- Define the number of blue fish
  let blue : ℕ := T / 2

  -- Define the number of orange fish
  let orange : ℕ := blue - 15

  -- Define the number of green fish
  let green : ℕ := T - blue - orange

  -- Prove that green = 15
  sorry

end green_fish_count_l801_80165


namespace class_average_l801_80116

theorem class_average (total_students : ℕ) (top_scorers : ℕ) (top_score : ℕ) (zero_scorers : ℕ) (rest_average : ℕ) :
  total_students = 25 →
  top_scorers = 3 →
  top_score = 95 →
  zero_scorers = 5 →
  rest_average = 45 →
  (top_scorers * top_score + zero_scorers * 0 + (total_students - top_scorers - zero_scorers) * rest_average) / total_students = 42 := by
  sorry

end class_average_l801_80116


namespace triangle_problem_l801_80185

open Real

theorem triangle_problem (A B C : ℝ) (a b c S : ℝ) :
  (2 * sin B - 2 * sin B ^ 2 - cos (2 * B) = sqrt 3 - 1) →
  (B = π / 3 ∨ B = 2 * π / 3) ∧
  (B = π / 3 ∧ a = 6 ∧ S = 6 * sqrt 3 → b = 2 * sqrt 7) := by
  sorry


end triangle_problem_l801_80185


namespace smallest_label_on_1993_l801_80103

theorem smallest_label_on_1993 (n : ℕ) (h : n > 0) :
  (n * (n + 1) / 2) % 2000 = 1021 →
  ∀ m, 0 < m ∧ m < n → (m * (m + 1) / 2) % 2000 ≠ 1021 →
  n = 118 := by
sorry

end smallest_label_on_1993_l801_80103


namespace parallel_lines_k_l801_80181

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of k for which the lines y = 5x - 3 and y = (3k)x + 7 are parallel -/
theorem parallel_lines_k : ∃ k : ℝ, 
  (∀ x y : ℝ, y = 5 * x - 3 ↔ y = (3 * k) * x + 7) ↔ k = 5 / 3 := by
  sorry

end parallel_lines_k_l801_80181


namespace probability_prime_or_odd_l801_80126

/-- A function that determines if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that determines if a natural number is odd -/
def isOdd (n : ℕ) : Prop := sorry

/-- The set of balls numbered 1 through 8 -/
def ballSet : Finset ℕ := sorry

/-- The probability of selecting a ball with a number that is either prime or odd -/
def probabilityPrimeOrOdd : ℚ := sorry

/-- Theorem stating that the probability of selecting a ball with a number
    that is either prime or odd is 5/8 -/
theorem probability_prime_or_odd :
  probabilityPrimeOrOdd = 5 / 8 := by sorry

end probability_prime_or_odd_l801_80126


namespace sum_of_abs_values_l801_80153

theorem sum_of_abs_values (a b : ℝ) : 
  (abs a = 3) → (abs b = 4) → (a < b) → (a + b = 1 ∨ a + b = 7) := by
  sorry

end sum_of_abs_values_l801_80153
