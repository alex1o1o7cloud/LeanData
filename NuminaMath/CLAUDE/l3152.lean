import Mathlib

namespace correct_calculation_l3152_315297

theorem correct_calculation (m : ℝ) : 6*m + (-2 - 10*m) = -4*m - 2 := by
  sorry

end correct_calculation_l3152_315297


namespace rice_distribution_l3152_315251

theorem rice_distribution (R : ℚ) : 
  (7/10 : ℚ) * R - (3/10 : ℚ) * R = 20 → R = 50 := by
sorry

end rice_distribution_l3152_315251


namespace no_real_roots_for_equation_l3152_315276

theorem no_real_roots_for_equation : ¬∃ x : ℝ, x + Real.sqrt (2*x - 5) = 5 := by
  sorry

end no_real_roots_for_equation_l3152_315276


namespace f_value_at_3_l3152_315295

/-- Given a function f(x) = x^7 + ax^5 + bx - 5 where f(-3) = 5, prove that f(3) = -15 -/
theorem f_value_at_3 (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^7 + a*x^5 + b*x - 5)
    (h2 : f (-3) = 5) : 
  f 3 = -15 := by sorry

end f_value_at_3_l3152_315295


namespace perpendicular_lines_from_perpendicular_planes_l3152_315216

-- Define the types for planes and lines
variable (Plane : Type) (Line : Type)

-- Define the relationships between planes and lines
variable (is_perpendicular_line_plane : Line → Plane → Prop)
variable (is_perpendicular_plane_plane : Plane → Plane → Prop)
variable (is_perpendicular_line_line : Line → Line → Prop)
variable (are_distinct : Plane → Plane → Prop)
variable (are_non_intersecting : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_planes 
  (α β : Plane) (l m : Line)
  (h_distinct : are_distinct α β)
  (h_non_intersecting : are_non_intersecting l m)
  (h1 : is_perpendicular_line_plane l α)
  (h2 : is_perpendicular_line_plane m β)
  (h3 : is_perpendicular_plane_plane α β) :
  is_perpendicular_line_line l m :=
sorry

end perpendicular_lines_from_perpendicular_planes_l3152_315216


namespace breaks_required_correct_l3152_315291

/-- Represents a chocolate bar of dimensions m × n -/
structure ChocolateBar where
  m : ℕ+
  n : ℕ+

/-- The number of breaks required to separate all 1 × 1 squares in a chocolate bar -/
def breaks_required (bar : ChocolateBar) : ℕ :=
  bar.m.val * bar.n.val - 1

/-- Theorem stating that the number of breaks required is correct -/
theorem breaks_required_correct (bar : ChocolateBar) :
  breaks_required bar = bar.m.val * bar.n.val - 1 :=
by sorry

end breaks_required_correct_l3152_315291


namespace runner_problem_l3152_315247

/-- Proves that given the conditions of the runner's problem, the time taken for the second half is 10 hours -/
theorem runner_problem (v : ℝ) (h1 : v > 0) : 
  (40 / v = 20 / v + 5) → (40 / (v / 2) = 10) :=
by
  sorry

end runner_problem_l3152_315247


namespace third_number_in_set_l3152_315223

theorem third_number_in_set (x : ℝ) : 
  let set1 := [10, 70, 28]
  let set2 := [20, 40, x]
  (set2.sum / set2.length : ℝ) = (set1.sum / set1.length : ℝ) + 4 →
  x = 60 := by
sorry

end third_number_in_set_l3152_315223


namespace quarter_sector_area_l3152_315245

/-- The area of a quarter sector of a circle with diameter 10 meters -/
theorem quarter_sector_area (d : ℝ) (h : d = 10) : 
  (π * (d / 2)^2) / 4 = 6.25 * π := by
  sorry

end quarter_sector_area_l3152_315245


namespace inverse_variation_problem_l3152_315298

/-- Two quantities vary inversely if their product is constant -/
def VaryInversely (r s : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, r x * s x = k

theorem inverse_variation_problem (r s : ℝ → ℝ) 
  (h1 : VaryInversely r s)
  (h2 : r 1 = 1500)
  (h3 : s 1 = 0.4)
  (h4 : r 2 = 3000) :
  s 2 = 0.2 := by
sorry

end inverse_variation_problem_l3152_315298


namespace distinct_positive_factors_of_81_l3152_315255

theorem distinct_positive_factors_of_81 : 
  Finset.card (Nat.divisors 81) = 5 := by
  sorry

end distinct_positive_factors_of_81_l3152_315255


namespace trapezoid_perimeter_l3152_315284

/-- Trapezoid ABCD with given side lengths and angle -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  BC : ℝ
  DA : ℝ
  angleD : ℝ
  h_AB : AB = 40
  h_CD : CD = 60
  h_BC : BC = 50
  h_DA : DA = 70
  h_angleD : angleD = π / 3 -- 60° in radians

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.AB + t.BC + t.CD + t.DA

/-- Theorem: The perimeter of the given trapezoid is 220 units -/
theorem trapezoid_perimeter (t : Trapezoid) : perimeter t = 220 := by
  sorry

end trapezoid_perimeter_l3152_315284


namespace modified_cube_edges_l3152_315270

/-- A cube with equilateral triangular pyramids extended from its edge midpoints. -/
structure ModifiedCube where
  /-- The number of edges in the original cube. -/
  cube_edges : ℕ
  /-- The number of vertices in the original cube. -/
  cube_vertices : ℕ
  /-- The number of new edges added by each pyramid (excluding the base). -/
  pyramid_new_edges : ℕ
  /-- The number of new edges added to each original cube edge. -/
  new_edges_per_cube_edge : ℕ

/-- The total number of edges in the modified cube. -/
def total_edges (c : ModifiedCube) : ℕ :=
  c.cube_edges + 
  (c.cube_edges * c.new_edges_per_cube_edge) + 
  c.cube_edges

theorem modified_cube_edges :
  ∀ (c : ModifiedCube),
  c.cube_edges = 12 →
  c.cube_vertices = 8 →
  c.pyramid_new_edges = 4 →
  c.new_edges_per_cube_edge = 2 →
  total_edges c = 48 := by
  sorry

end modified_cube_edges_l3152_315270


namespace angle_KJG_measure_l3152_315266

-- Define the geometric configuration
structure GeometricConfig where
  -- JKL is a 45-45-90 right triangle
  JKL_is_45_45_90 : Bool
  -- GHIJ is a square
  GHIJ_is_square : Bool
  -- JKLK is a square
  JKLK_is_square : Bool

-- Define the theorem
theorem angle_KJG_measure (config : GeometricConfig) 
  (h1 : config.JKL_is_45_45_90 = true)
  (h2 : config.GHIJ_is_square = true)
  (h3 : config.JKLK_is_square = true) :
  ∃ (angle_KJG : ℝ), angle_KJG = 135 := by
  sorry


end angle_KJG_measure_l3152_315266


namespace m_value_l3152_315292

theorem m_value (m : ℝ) : 
  let A : Set ℝ := {0, m, m^2 - 3*m + 2}
  2 ∈ A → m = 3 := by
sorry

end m_value_l3152_315292


namespace legos_in_box_l3152_315206

theorem legos_in_box (total : ℕ) (used : ℕ) (missing : ℕ) (in_box : ℕ) : 
  total = 500 → 
  used = total / 2 → 
  missing = 5 → 
  in_box = total - used - missing → 
  in_box = 245 := by
sorry

end legos_in_box_l3152_315206


namespace specific_stick_displacement_l3152_315211

/-- Represents a uniform stick leaning against a support -/
structure LeaningStick where
  length : ℝ
  projection : ℝ

/-- Calculates the final horizontal displacement of a leaning stick after falling -/
def finalDisplacement (stick : LeaningStick) : ℝ :=
  sorry

/-- Theorem stating the final displacement of a specific stick configuration -/
theorem specific_stick_displacement :
  let stick : LeaningStick := { length := 120, projection := 70 }
  finalDisplacement stick = 25 := by sorry

end specific_stick_displacement_l3152_315211


namespace reciprocal_sum_l3152_315272

theorem reciprocal_sum (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + a = b / a + b) : 1 / a + 1 / b = -1 := by
  sorry

end reciprocal_sum_l3152_315272


namespace problem_statement_l3152_315271

theorem problem_statement (x y : ℝ) (h : 1 ≤ x^2 - x*y + y^2 ∧ x^2 - x*y + y^2 ≤ 2) :
  (2/9 ≤ x^4 + y^4 ∧ x^4 + y^4 ≤ 8) ∧
  (∀ n : ℕ, n ≥ 3 → x^(2*n) + y^(2*n) ≥ 2/3^n) := by
  sorry

end problem_statement_l3152_315271


namespace correct_seating_arrangements_l3152_315201

/-- The number of seats in the row -/
def num_seats : ℕ := 8

/-- The number of people to be seated -/
def num_people : ℕ := 3

/-- A function that calculates the number of valid seating arrangements -/
def seating_arrangements (seats : ℕ) (people : ℕ) : ℕ :=
  sorry  -- The actual implementation would go here

/-- Theorem stating that the number of seating arrangements is 24 -/
theorem correct_seating_arrangements :
  seating_arrangements num_seats num_people = 24 := by sorry


end correct_seating_arrangements_l3152_315201


namespace arithmetic_sequence_inequality_l3152_315246

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_inequality
  (a : ℕ → ℝ) (d : ℝ) (n : ℕ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_positive : d > 0)
  (h_n : n > 1) :
  a 1 * a (n + 1) < a 2 * a n :=
by
  sorry

end arithmetic_sequence_inequality_l3152_315246


namespace factorial_divides_theorem_l3152_315241

def divides (a b : ℤ) : Prop := ∃ k : ℤ, b = a * k

theorem factorial_divides_theorem (a : ℤ) :
  (∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, divides (n.factorial + a) ((2*n).factorial)) →
  a = 0 := by sorry

end factorial_divides_theorem_l3152_315241


namespace geometric_sequence_common_ratio_l3152_315218

/-- A geometric sequence with the given properties has a common ratio of 1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) 
  (h_sum_1_3 : a 1 + a 3 = 10) 
  (h_sum_4_6 : a 4 + a 6 = 5/4) : 
  a 2 / a 1 = 1/2 := by
sorry

end geometric_sequence_common_ratio_l3152_315218


namespace bennett_brothers_count_l3152_315244

theorem bennett_brothers_count :
  ∀ (aaron_brothers bennett_brothers : ℕ),
    aaron_brothers = 4 →
    bennett_brothers = 2 * aaron_brothers - 2 →
    bennett_brothers = 6 :=
by
  sorry

end bennett_brothers_count_l3152_315244


namespace tangent_points_satisfy_locus_l3152_315289

/-- A conic section with focus at the origin and directrix x - d = 0 -/
structure ConicSection (d : ℝ) where
  -- Point on the conic section
  x : ℝ
  y : ℝ
  -- Eccentricity
  e : ℝ
  -- Conic section equation
  eq : x^2 + y^2 = e^2 * (x - d)^2

/-- A point of tangency on the conic section -/
structure TangentPoint (d : ℝ) extends ConicSection d where
  -- Tangent line has slope 1 (parallel to y = x)
  tangent_slope : (1 - e^2) * x + y + e^2 * d = 0

/-- The locus of points of tangency -/
def locus_equation (d : ℝ) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  y^2 - x*y + d*(x + y) = 0

/-- The main theorem: points of tangency satisfy the locus equation -/
theorem tangent_points_satisfy_locus (d : ℝ) (p : TangentPoint d) :
  locus_equation d (p.x, p.y) := by
  sorry


end tangent_points_satisfy_locus_l3152_315289


namespace a_range_l3152_315228

def p (a : ℝ) : Prop := ∃ (x y : ℝ), x^2 + y^2 - a*x + y + 1 = 0

def q (a : ℝ) : Prop := ∃ (x y : ℝ), 2*a*x + (1-a)*y + 1 = 0 ∧ (2*a)/(a-1) > 1

def range_of_a (a : ℝ) : Prop := a ∈ Set.Icc (-Real.sqrt 3) (-1) ∪ Set.Ioc 1 (Real.sqrt 3)

theorem a_range (a : ℝ) :
  (∀ a, p a ∨ q a) ∧ (∀ a, ¬(p a ∧ q a)) →
  range_of_a a :=
sorry

end a_range_l3152_315228


namespace nut_mixture_weight_l3152_315278

/-- Given a mixture of nuts with 5 parts almonds to 2 parts walnuts by weight,
    and 250 pounds of almonds, the total weight of the mixture is 350 pounds. -/
theorem nut_mixture_weight (almond_parts : ℕ) (walnut_parts : ℕ) (almond_weight : ℝ) :
  almond_parts = 5 →
  walnut_parts = 2 →
  almond_weight = 250 →
  (almond_weight / almond_parts) * (almond_parts + walnut_parts) = 350 := by
  sorry

#check nut_mixture_weight

end nut_mixture_weight_l3152_315278


namespace inequality_proof_l3152_315213

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 
  3/2 + 1/4 * (a * (c - b)^2 / (c + b) + b * (c - a)^2 / (c + a) + c * (b - a)^2 / (b + a)) := by
  sorry

end inequality_proof_l3152_315213


namespace square_of_99_l3152_315214

theorem square_of_99 : (99 : ℕ) ^ 2 = 9801 := by
  sorry

end square_of_99_l3152_315214


namespace peanuts_lost_l3152_315233

def initial_peanuts : ℕ := 74
def final_peanuts : ℕ := 15

theorem peanuts_lost : initial_peanuts - final_peanuts = 59 := by
  sorry

end peanuts_lost_l3152_315233


namespace sugar_cube_weight_l3152_315253

/-- The weight of sugar cubes in the first group -/
def weight_first_group : ℝ := 10

/-- The number of ants in the first group -/
def ants_first : ℕ := 15

/-- The number of sugar cubes moved by the first group -/
def cubes_first : ℕ := 600

/-- The time taken by the first group (in hours) -/
def time_first : ℝ := 5

/-- The number of ants in the second group -/
def ants_second : ℕ := 20

/-- The number of sugar cubes moved by the second group -/
def cubes_second : ℕ := 960

/-- The time taken by the second group (in hours) -/
def time_second : ℝ := 3

/-- The weight of sugar cubes in the second group -/
def weight_second : ℝ := 5

theorem sugar_cube_weight :
  (ants_first : ℝ) * cubes_second * time_first * weight_second =
  (ants_second : ℝ) * cubes_first * time_second * weight_first_group :=
by sorry

end sugar_cube_weight_l3152_315253


namespace zeros_of_f_l3152_315207

def f (x : ℝ) : ℝ := x * (x^2 - 16)

theorem zeros_of_f :
  ∀ x : ℝ, f x = 0 ↔ x = -4 ∨ x = 0 ∨ x = 4 := by
  sorry

end zeros_of_f_l3152_315207


namespace harry_snakes_l3152_315279

/-- The number of snakes Harry owns -/
def num_snakes : ℕ := sorry

/-- The number of geckos Harry owns -/
def num_geckos : ℕ := 3

/-- The number of iguanas Harry owns -/
def num_iguanas : ℕ := 2

/-- Monthly feeding cost per snake in dollars -/
def snake_cost : ℕ := 10

/-- Monthly feeding cost per iguana in dollars -/
def iguana_cost : ℕ := 5

/-- Monthly feeding cost per gecko in dollars -/
def gecko_cost : ℕ := 15

/-- Total yearly feeding cost for all pets in dollars -/
def total_yearly_cost : ℕ := 1140

theorem harry_snakes :
  num_snakes = 4 ∧
  (12 * (num_geckos * gecko_cost + num_iguanas * iguana_cost + num_snakes * snake_cost) = total_yearly_cost) :=
by sorry

end harry_snakes_l3152_315279


namespace smallest_n_perfect_square_and_fifth_power_l3152_315281

theorem smallest_n_perfect_square_and_fifth_power : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^5) ∧
  (∀ (x : ℕ), x > 0 → 
    ((∃ (y : ℕ), 4 * x = y^2) ∧ (∃ (z : ℕ), 5 * x = z^5)) → 
    x ≥ 625) ∧
  n = 625 :=
sorry

end smallest_n_perfect_square_and_fifth_power_l3152_315281


namespace cubic_factor_sum_l3152_315263

/-- Given a cubic polynomial x^3 + ax^2 + bx + 8 with factors (x+1) and (x+2),
    prove that a + b = 21 -/
theorem cubic_factor_sum (a b : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, x^3 + a*x^2 + b*x + 8 = (x+1)*(x+2)*(x+c)) →
  a + b = 21 := by
sorry

end cubic_factor_sum_l3152_315263


namespace circle_C_equation_circle_C_fixed_point_l3152_315268

-- Define the circle C
def circle_C (t x y : ℝ) : Prop :=
  x^2 + y^2 - 2*t*x - 2*t^2*y + 4*t - 4 = 0

-- Define the line on which the center of C lies
def center_line (x y : ℝ) : Prop :=
  x - y + 2 = 0

-- Theorem 1: Equation of circle C
theorem circle_C_equation (t : ℝ) :
  (∃ x y : ℝ, circle_C t x y ∧ center_line x y) →
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 2*y - 8 = 0) ∨
  (∃ x y : ℝ, x^2 + y^2 - 4*x - 8*y + 4 = 0) :=
sorry

-- Theorem 2: Fixed point of circle C
theorem circle_C_fixed_point (t : ℝ) :
  circle_C t 2 0 :=
sorry

end circle_C_equation_circle_C_fixed_point_l3152_315268


namespace circular_garden_radius_increase_l3152_315286

theorem circular_garden_radius_increase (c₁ c₂ r₁ r₂ : ℝ) :
  c₁ = 30 →
  c₂ = 40 →
  c₁ = 2 * π * r₁ →
  c₂ = 2 * π * r₂ →
  r₂ - r₁ = 5 / π := by
sorry

end circular_garden_radius_increase_l3152_315286


namespace reflection_across_x_axis_l3152_315299

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point in the standard coordinate system -/
def original_point : ℝ × ℝ := (-2, 3)

theorem reflection_across_x_axis :
  reflect_x original_point = (-2, -3) := by sorry

end reflection_across_x_axis_l3152_315299


namespace right_triangle_hypotenuse_l3152_315208

theorem right_triangle_hypotenuse (PQ PR PS SQ PT TR QT SR : ℝ) :
  PS / SQ = 1 / 3 →
  PT / TR = 1 / 3 →
  QT = 20 →
  SR = 36 →
  PQ^2 + PR^2 = 1085.44 :=
by sorry

end right_triangle_hypotenuse_l3152_315208


namespace parabola_focus_distance_l3152_315265

/-- Theorem: For a parabola y² = 2px where p > 0, if the distance from its focus 
    to the line y = x + 1 is √2, then p = 2. -/
theorem parabola_focus_distance (p : ℝ) : 
  p > 0 → 
  (let focus : ℝ × ℝ := (p/2, 0)
   let distance_to_line (x y : ℝ) := |(-1:ℝ)*x + y - 1| / Real.sqrt 2
   distance_to_line (p/2) 0 = Real.sqrt 2) → 
  p = 2 := by
sorry

end parabola_focus_distance_l3152_315265


namespace sqrt_expression_simplification_l3152_315264

theorem sqrt_expression_simplification :
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end sqrt_expression_simplification_l3152_315264


namespace at_least_one_real_root_l3152_315277

theorem at_least_one_real_root (c : ℝ) : 
  ∃ x : ℝ, (x^2 + c*x + 2 = 0) ∨ (x^2 + 2*x + c = 2) := by
  sorry

end at_least_one_real_root_l3152_315277


namespace principal_amount_is_875_l3152_315250

/-- Calculates the principal amount given the interest rate, time, and total interest. -/
def calculate_principal (interest_rate : ℚ) (time : ℕ) (total_interest : ℚ) : ℚ :=
  (total_interest * 100) / (interest_rate * time)

/-- Theorem stating that given the specified conditions, the principal amount is 875. -/
theorem principal_amount_is_875 :
  let interest_rate : ℚ := 12
  let time : ℕ := 20
  let total_interest : ℚ := 2100
  calculate_principal interest_rate time total_interest = 875 := by sorry

end principal_amount_is_875_l3152_315250


namespace max_value_of_expression_l3152_315220

theorem max_value_of_expression :
  (∀ x : ℝ, |x - 1| - |x + 4| - 5 ≤ 0) ∧
  (∃ x : ℝ, |x - 1| - |x + 4| - 5 = 0) :=
by sorry

end max_value_of_expression_l3152_315220


namespace third_line_product_l3152_315248

/-- Given two positive real numbers a and b, prove that 
    x = -a/2 + √(a²/4 + b²) satisfies x(x + a) = b² -/
theorem third_line_product (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let x := -a/2 + Real.sqrt (a^2/4 + b^2)
  x * (x + a) = b^2 := by
  sorry

end third_line_product_l3152_315248


namespace election_winner_votes_l3152_315296

theorem election_winner_votes (total_votes : ℕ) : 
  (total_votes : ℝ) * 0.62 - (total_votes : ℝ) * 0.38 = 348 →
  (total_votes : ℝ) * 0.62 = 899 :=
by
  sorry

end election_winner_votes_l3152_315296


namespace technician_average_salary_l3152_315202

/-- Calculates the average salary of technicians in a workshop --/
theorem technician_average_salary
  (total_workers : ℕ)
  (total_average : ℚ)
  (num_technicians : ℕ)
  (non_technician_average : ℚ)
  (h1 : total_workers = 30)
  (h2 : total_average = 8000)
  (h3 : num_technicians = 10)
  (h4 : non_technician_average = 6000)
  : (total_average * total_workers - non_technician_average * (total_workers - num_technicians)) / num_technicians = 12000 := by
  sorry

end technician_average_salary_l3152_315202


namespace linear_function_values_l3152_315215

/-- A linear function y = kx + b passing through (-1, 0) and (2, 1/2) -/
def linear_function (x : ℚ) : ℚ :=
  let k : ℚ := 6
  let b : ℚ := -1
  k * x + b

theorem linear_function_values :
  linear_function 0 = -1 ∧
  linear_function (1/2) = 2 ∧
  linear_function (-1/2) = -4 := by
  sorry


end linear_function_values_l3152_315215


namespace gcd_lcm_sum_45_4095_l3152_315254

theorem gcd_lcm_sum_45_4095 : 
  (Nat.gcd 45 4095) + (Nat.lcm 45 4095) = 4140 := by
  sorry

end gcd_lcm_sum_45_4095_l3152_315254


namespace four_digit_cube_square_sum_multiple_of_seven_l3152_315239

theorem four_digit_cube_square_sum_multiple_of_seven :
  ∃ (x y : ℕ), 
    1000 ≤ x ∧ x < 10000 ∧ 
    7 ∣ x ∧
    x = (y^3 + y^2) / 7 ∧
    (x = 1386 ∨ x = 1200) :=
sorry

end four_digit_cube_square_sum_multiple_of_seven_l3152_315239


namespace new_person_age_l3152_315224

/-- Given a group of 10 persons where replacing a 46-year-old person with a new person
    decreases the average age by 3 years, prove that the age of the new person is 16 years. -/
theorem new_person_age (T : ℝ) (A : ℝ) : 
  (T / 10 = (T - 46 + A) / 10 + 3) → A = 16 := by
  sorry

end new_person_age_l3152_315224


namespace polynomial_root_sum_l3152_315235

theorem polynomial_root_sum (c d : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + c * (Complex.I * Real.sqrt 2 + 2) + d = 0 → 
  c + d = 14 := by
sorry

end polynomial_root_sum_l3152_315235


namespace intersecting_circles_equal_chords_l3152_315282

/-- Given two intersecting circles with radii 10 and 8 units, whose centers are 15 units apart,
    if a line is drawn through their intersection point P such that it creates equal chords QP and PR,
    then the square of the length of chord QP is 250. -/
theorem intersecting_circles_equal_chords (r₁ r₂ d : ℝ) (P Q R : ℝ × ℝ) :
  r₁ = 10 →
  r₂ = 8 →
  d = 15 →
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (P.1 - R.1)^2 + (P.2 - R.2)^2 →
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 250 :=
by sorry

end intersecting_circles_equal_chords_l3152_315282


namespace square_side_length_l3152_315203

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 1 / 9) (h2 : side * side = area) : 
  side = 1 / 3 := by
sorry

end square_side_length_l3152_315203


namespace square_side_length_l3152_315210

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 1/9 → side^2 = area → side = 1/3 := by
  sorry

end square_side_length_l3152_315210


namespace quadratic_points_relationship_l3152_315259

/-- A quadratic function f(x) = -x² + 2x + 3 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

/-- Point P₁ on the graph of f -/
def P₁ : ℝ × ℝ := (-1, f (-1))

/-- Point P₂ on the graph of f -/
def P₂ : ℝ × ℝ := (3, f 3)

/-- Point P₃ on the graph of f -/
def P₃ : ℝ × ℝ := (5, f 5)

theorem quadratic_points_relationship :
  P₁.2 = P₂.2 ∧ P₂.2 > P₃.2 := by sorry

end quadratic_points_relationship_l3152_315259


namespace sum_of_solutions_l3152_315209

theorem sum_of_solutions (y : ℝ) : (∃ y₁ y₂ : ℝ, y₁ + 16 / y₁ = 12 ∧ y₂ + 16 / y₂ = 12 ∧ y₁ ≠ y₂ ∧ y₁ + y₂ = 12) := by
  sorry

end sum_of_solutions_l3152_315209


namespace factorization_proof_l3152_315256

theorem factorization_proof (z : ℝ) :
  45 * z^12 + 180 * z^24 = 45 * z^12 * (1 + 4 * z^12) := by
  sorry

end factorization_proof_l3152_315256


namespace min_value_of_4a_plus_b_l3152_315229

theorem min_value_of_4a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (1 : ℝ) / a + (1 : ℝ) / b = 1) : 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → (1 : ℝ) / a' + (1 : ℝ) / b' = 1 → 4 * a' + b' ≥ 4 * a + b) ∧ 
  4 * a + b = 9 := by
  sorry

end min_value_of_4a_plus_b_l3152_315229


namespace linear_regression_point_difference_l3152_315269

theorem linear_regression_point_difference (x₀ y₀ : ℝ) : 
  let data_points : List (ℝ × ℝ) := [(1, 2), (3, 5), (6, 8), (x₀, y₀)]
  let x_mean : ℝ := (1 + 3 + 6 + x₀) / 4
  let y_mean : ℝ := (2 + 5 + 8 + y₀) / 4
  let regression_line (x : ℝ) : ℝ := x + 2
  regression_line x_mean = y_mean →
  x₀ - y₀ = -3 := by
sorry

end linear_regression_point_difference_l3152_315269


namespace quadratic_inequality_l3152_315212

theorem quadratic_inequality (a b c : ℝ) 
  (h : ∀ x : ℝ, a * x^2 + b * x + c < 0) : 
  b / a < c / a + 1 := by
  sorry

end quadratic_inequality_l3152_315212


namespace expression_is_equation_l3152_315257

/-- Definition of an equation -/
def is_equation (e : Prop) : Prop :=
  ∃ (x : ℝ), ∃ (f g : ℝ → ℝ), e = (f x = g x)

/-- The expression 2x - 1 = 3 is an equation -/
theorem expression_is_equation : is_equation (∃ x : ℝ, 2 * x - 1 = 3) := by
  sorry

end expression_is_equation_l3152_315257


namespace solve_equation_l3152_315267

theorem solve_equation (x y : ℚ) : y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end solve_equation_l3152_315267


namespace max_value_of_x_minus_x_squared_l3152_315273

theorem max_value_of_x_minus_x_squared (x : ℝ) :
  0 < x → x < 1 → ∃ (y : ℝ), y = 1/2 ∧ ∀ z, 0 < z → z < 1 → x * (1 - x) ≤ y * (1 - y) :=
by sorry

end max_value_of_x_minus_x_squared_l3152_315273


namespace solution_set_equivalence_l3152_315288

theorem solution_set_equivalence :
  ∀ (x y z : ℝ), x^2 - 9*y^2 = z^2 ↔ ∃ t : ℝ, x = 3*t ∧ y = t ∧ z = 0 :=
by sorry

end solution_set_equivalence_l3152_315288


namespace function_inequality_l3152_315294

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (π / 2), (deriv f x) / tan x < f x) →
  f (π / 3) < Real.sqrt 3 * f (π / 6) := by
sorry

end function_inequality_l3152_315294


namespace triangle_abc_properties_l3152_315280

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  B = π / 4 →
  c = Real.sqrt 6 →
  C = π / 3 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  A = 5 * π / 12 ∧
  a = 1 + Real.sqrt 3 ∧
  b = 2 := by
  sorry

end triangle_abc_properties_l3152_315280


namespace work_completion_time_l3152_315237

/-- The number of days it takes for A to complete the work alone -/
def days_for_A : ℕ := 6

/-- The number of days it takes for B to complete the work alone -/
def days_for_B : ℕ := 8

/-- The number of days it takes for A, B, and C to complete the work together -/
def days_for_ABC : ℕ := 3

/-- The total payment for the work in rupees -/
def total_payment : ℕ := 1200

/-- C's share of the payment in rupees -/
def C_share : ℕ := 150

theorem work_completion_time :
  (1 : ℚ) / days_for_A + (1 : ℚ) / days_for_B + 
  ((C_share : ℚ) / total_payment) * ((1 : ℚ) / days_for_ABC) = 
  (1 : ℚ) / days_for_ABC := by sorry

end work_completion_time_l3152_315237


namespace unfenced_length_l3152_315200

theorem unfenced_length 
  (field_side : ℝ) 
  (wire_cost : ℝ) 
  (budget : ℝ) 
  (h1 : field_side = 5000)
  (h2 : wire_cost = 30)
  (h3 : budget = 120000) : 
  field_side * 4 - (budget / wire_cost) = 1000 := by
  sorry

end unfenced_length_l3152_315200


namespace exam_contestants_l3152_315285

theorem exam_contestants :
  ∀ (x y : ℕ),
  (30 * (x - 1) + 26 = 26 * (y - 1) + 20) →
  (y = x + 9) →
  (30 * x - 4 = 1736) :=
by
  sorry

end exam_contestants_l3152_315285


namespace set_inclusion_implies_m_range_l3152_315205

theorem set_inclusion_implies_m_range (m : ℝ) :
  let P : Set ℝ := {x | -2 ≤ x ∧ x ≤ 10}
  let S : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}
  (S.Nonempty) → (S ⊆ P) → (0 ≤ m ∧ m ≤ 3) :=
by
  sorry

end set_inclusion_implies_m_range_l3152_315205


namespace range_of_a_l3152_315243

-- Define the line l: 2x - 3y + 1 = 0
def line (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0

-- Define the points M and N
def point_M (a : ℝ) : ℝ × ℝ := (1, -a)
def point_N (a : ℝ) : ℝ × ℝ := (a, 1)

-- Define the condition for points being on opposite sides of the line
def opposite_sides (a : ℝ) : Prop :=
  (2 * (point_M a).1 - 3 * (point_M a).2 + 1) * (2 * (point_N a).1 - 3 * (point_N a).2 + 1) < 0

-- Theorem statement
theorem range_of_a (a : ℝ) : opposite_sides a → -1 < a ∧ a < 1 := by
  sorry

end range_of_a_l3152_315243


namespace profit_percentage_invariant_l3152_315283

/-- Represents the profit percentage as a real number between 0 and 1 -/
def ProfitPercentage := { p : ℝ // 0 ≤ p ∧ p ≤ 1 }

/-- Represents the discount percentage as a real number between 0 and 1 -/
def DiscountPercentage := { d : ℝ // 0 ≤ d ∧ d ≤ 1 }

/-- The profit percentage remains the same regardless of the discount -/
theorem profit_percentage_invariant (profit_with_discount : ProfitPercentage) 
  (discount : DiscountPercentage) :
  ∃ (profit_without_discount : ProfitPercentage), 
  profit_without_discount = profit_with_discount :=
sorry

end profit_percentage_invariant_l3152_315283


namespace spade_calculation_l3152_315204

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_calculation : spade 5 (spade 6 7 + 2) = -96 := by
  sorry

end spade_calculation_l3152_315204


namespace total_answer_key_combinations_l3152_315287

/-- Represents the number of answer choices for a multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- Represents the number of true-false questions -/
def true_false_questions : ℕ := 4

/-- Represents the number of multiple-choice questions -/
def multiple_choice_questions : ℕ := 2

/-- Calculates the number of valid combinations for true-false questions -/
def valid_true_false_combinations : ℕ := 2^true_false_questions - 2

/-- Calculates the number of combinations for multiple-choice questions -/
def multiple_choice_combinations : ℕ := multiple_choice_options^multiple_choice_questions

/-- Theorem stating the total number of ways to create an answer key -/
theorem total_answer_key_combinations :
  valid_true_false_combinations * multiple_choice_combinations = 224 := by
  sorry

end total_answer_key_combinations_l3152_315287


namespace fixed_point_on_line_l3152_315252

theorem fixed_point_on_line (m : ℝ) : 
  (3 * m - 2) * (-3/4 : ℝ) - (m - 2) * (-13/4 : ℝ) - (m - 5) = 0 := by
  sorry

end fixed_point_on_line_l3152_315252


namespace expression1_eval_expression2_eval_l3152_315293

-- Part 1
def expression1 (x : ℝ) : ℝ := -3*x^2 + 5*x - 0.5*x^2 + x - 1

theorem expression1_eval : expression1 2 = -3 := by sorry

-- Part 2
def expression2 (a b : ℝ) : ℝ := (a^2*b + 3*a*b^2) - 3*(a^2*b + a*b^2 - 1)

theorem expression2_eval : expression2 (-2) 2 = -13 := by sorry

end expression1_eval_expression2_eval_l3152_315293


namespace polynomial_root_property_l3152_315227

/-- Given a polynomial x^3 + ax^2 + bx + 18b with nonzero integer coefficients a and b,
    if it has two coinciding integer roots and all three roots are integers,
    then |ab| = 1440 -/
theorem polynomial_root_property (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃ r s : ℤ, (∀ x : ℝ, x^3 + a*x^2 + b*x + 18*b = (x - r)^2 * (x - s)) ∧
              r ≠ s) →
  |a * b| = 1440 := by
  sorry

end polynomial_root_property_l3152_315227


namespace knights_and_knaves_solution_l3152_315238

-- Define the types for residents and their statements
inductive Resident : Type
| A
| B
| C

inductive Status : Type
| Knight
| Knave

-- Define the statement made by A
def statement_A (status : Resident → Status) : Prop :=
  status Resident.C = Status.Knight → status Resident.B = Status.Knave

-- Define the statement made by C
def statement_C (status : Resident → Status) : Prop :=
  status Resident.A ≠ status Resident.C ∧
  ((status Resident.A = Status.Knight ∧ status Resident.C = Status.Knave) ∨
   (status Resident.A = Status.Knave ∧ status Resident.C = Status.Knight))

-- Define the truthfulness of statements based on the speaker's status
def is_truthful (status : Resident → Status) (r : Resident) (stmt : Prop) : Prop :=
  (status r = Status.Knight ∧ stmt) ∨ (status r = Status.Knave ∧ ¬stmt)

-- Theorem stating the solution
theorem knights_and_knaves_solution :
  ∃ (status : Resident → Status),
    is_truthful status Resident.A (statement_A status) ∧
    is_truthful status Resident.C (statement_C status) ∧
    status Resident.A = Status.Knave ∧
    status Resident.B = Status.Knight ∧
    status Resident.C = Status.Knight :=
sorry

end knights_and_knaves_solution_l3152_315238


namespace solve_for_z_l3152_315258

theorem solve_for_z (x y z : ℤ) (h1 : x^2 = y - 4) (h2 : x = -6) (h3 : y = z + 2) : z = 38 := by
  sorry

end solve_for_z_l3152_315258


namespace isosceles_triangle_perimeter_l3152_315231

/-- An isosceles triangle with sides a, b, and c, where a = b -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : a = b
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

/-- Theorem: The perimeter of an isosceles triangle with sides 2 and 5 is 12 -/
theorem isosceles_triangle_perimeter :
  ∃ (t : IsoscelesTriangle), t.a = 2 ∧ t.c = 5 ∧ perimeter t = 12 :=
by sorry

end isosceles_triangle_perimeter_l3152_315231


namespace female_students_count_l3152_315236

theorem female_students_count (total_students sample_size : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : ∃ (sampled_girls sampled_boys : ℕ), 
    sampled_girls + sampled_boys = sample_size ∧ 
    sampled_boys = sampled_girls + 10) :
  ∃ (female_students : ℕ), female_students = 760 ∧ 
    female_students * sample_size = sampled_girls * total_students :=
by
  sorry


end female_students_count_l3152_315236


namespace x_plus_p_in_terms_of_p_l3152_315242

theorem x_plus_p_in_terms_of_p (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) : x + p = 2*p + 3 := by
  sorry

end x_plus_p_in_terms_of_p_l3152_315242


namespace interior_angles_ratio_l3152_315249

-- Define a triangle type
structure Triangle where
  -- Define exterior angles
  ext_angle1 : ℝ
  ext_angle2 : ℝ
  ext_angle3 : ℝ
  -- Condition: exterior angles sum to 360°
  sum_ext_angles : ext_angle1 + ext_angle2 + ext_angle3 = 360
  -- Condition: ratio of exterior angles is 3:4:5
  ratio_ext_angles : ∃ (x : ℝ), ext_angle1 = 3*x ∧ ext_angle2 = 4*x ∧ ext_angle3 = 5*x

-- Define interior angles
def interior_angle1 (t : Triangle) : ℝ := 180 - t.ext_angle1
def interior_angle2 (t : Triangle) : ℝ := 180 - t.ext_angle2
def interior_angle3 (t : Triangle) : ℝ := 180 - t.ext_angle3

-- Theorem statement
theorem interior_angles_ratio (t : Triangle) :
  ∃ (k : ℝ), interior_angle1 t = 3*k ∧ interior_angle2 t = 2*k ∧ interior_angle3 t = k := by
  sorry

end interior_angles_ratio_l3152_315249


namespace double_arccos_three_fifths_equals_arcsin_twentyfour_twentyfifths_l3152_315221

theorem double_arccos_three_fifths_equals_arcsin_twentyfour_twentyfifths :
  2 * Real.arccos (3/5) = Real.arcsin (24/25) := by
  sorry

end double_arccos_three_fifths_equals_arcsin_twentyfour_twentyfifths_l3152_315221


namespace race_distance_l3152_315225

theorem race_distance (a_time b_time : ℝ) (beat_distance : ℝ) : 
  a_time = 28 → b_time = 32 → beat_distance = 16 → 
  ∃ d : ℝ, d = 128 ∧ d / a_time * b_time = d - beat_distance :=
by
  sorry

end race_distance_l3152_315225


namespace claire_balloons_l3152_315261

theorem claire_balloons (initial : ℕ) : 
  initial - 12 - 9 + 11 = 39 → initial = 49 := by
  sorry

end claire_balloons_l3152_315261


namespace five_digit_number_divisibility_l3152_315290

theorem five_digit_number_divisibility (U : ℕ) : 
  U < 10 →
  (2018 * 10 + U) % 9 = 0 →
  (2018 * 10 + U) % 8 = 3 :=
by sorry

end five_digit_number_divisibility_l3152_315290


namespace scarlet_savings_l3152_315260

/-- The amount of money Scarlet saved initially -/
def initial_savings : ℕ := sorry

/-- The cost of the earrings Scarlet bought -/
def earrings_cost : ℕ := 23

/-- The cost of the necklace Scarlet bought -/
def necklace_cost : ℕ := 48

/-- The amount of money Scarlet has left -/
def money_left : ℕ := 9

/-- Theorem stating that Scarlet's initial savings equals the sum of her purchases and remaining money -/
theorem scarlet_savings : initial_savings = earrings_cost + necklace_cost + money_left :=
by sorry

end scarlet_savings_l3152_315260


namespace negation_theorem1_negation_theorem2_l3152_315262

-- Define a type for triangles
structure Triangle where
  -- You might add more properties here if needed
  interiorAngleSum : ℝ

-- Define the propositions
def proposition1 : Prop := ∃ t : Triangle, t.interiorAngleSum ≠ 180

def proposition2 : Prop := ∀ x : ℝ, |x| + x^2 ≥ 0

-- State the theorems
theorem negation_theorem1 : 
  (¬ proposition1) ↔ (∀ t : Triangle, t.interiorAngleSum = 180) :=
sorry

theorem negation_theorem2 : 
  (¬ proposition2) ↔ (∃ x : ℝ, |x| + x^2 < 0) :=
sorry

end negation_theorem1_negation_theorem2_l3152_315262


namespace negation_equivalence_l3152_315217

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
by sorry

end negation_equivalence_l3152_315217


namespace stable_poly_characterization_l3152_315230

-- Define the set K of positive integers not containing the digit 7
def K : Set Nat := {n : Nat | n > 0 ∧ ∀ d, d ∈ n.digits 10 → d ≠ 7}

-- Define a polynomial with nonnegative coefficients
def NonNegativePoly (f : Nat → Nat) : Prop :=
  ∃ (coeffs : List Nat), ∀ x, f x = (coeffs.enum.map (λ (i, a) => a * x^i)).sum

-- Define the stable property for a polynomial
def Stable (f : Nat → Nat) : Prop :=
  ∀ x, x ∈ K → f x ∈ K

-- Theorem statement
theorem stable_poly_characterization (f : Nat → Nat) 
  (h_nonneg : NonNegativePoly f) (h_stable : Stable f) :
  (∃ e k, k ∈ K ∧ ∀ x, f x = 10^e * x + k) ∨
  (∃ e, ∀ x, f x = 10^e * x) ∨
  (∃ k, k ∈ K ∧ ∀ x, f x = k) :=
sorry

end stable_poly_characterization_l3152_315230


namespace arithmetic_sequence_middle_term_l3152_315275

/-- Three real numbers form an arithmetic sequence if the middle term is the arithmetic mean of the other two terms -/
def is_arithmetic_sequence (a b c : ℝ) : Prop := b = (a + c) / 2

theorem arithmetic_sequence_middle_term :
  ∀ m : ℝ, is_arithmetic_sequence 2 m 6 → m = 4 := by
  sorry

end arithmetic_sequence_middle_term_l3152_315275


namespace max_product_of_radii_l3152_315240

/-- Two circles C₁ and C₂ are externally tangent -/
def externally_tangent (a b : ℝ) : Prop :=
  a + b = 3

/-- The equation of circle C₁ -/
def circle_C₁ (a : ℝ) (x y : ℝ) : Prop :=
  (x + a)^2 + (y - 2)^2 = 1

/-- The equation of circle C₂ -/
def circle_C₂ (b : ℝ) (x y : ℝ) : Prop :=
  (x - b)^2 + (y - 2)^2 = 4

theorem max_product_of_radii (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_tangent : externally_tangent a b) :
  a * b ≤ 9/4 ∧ ∃ (a₀ b₀ : ℝ), a₀ * b₀ = 9/4 ∧ externally_tangent a₀ b₀ := by
  sorry

end max_product_of_radii_l3152_315240


namespace hari_joined_after_five_months_l3152_315222

/-- Represents the business scenario with two partners --/
structure Business where
  praveen_investment : ℕ
  hari_investment : ℕ
  profit_ratio_praveen : ℕ
  profit_ratio_hari : ℕ
  total_duration : ℕ

/-- Calculates the number of months after which Hari joined the business --/
def months_until_hari_joined (b : Business) : ℕ :=
  let x := b.total_duration - (b.praveen_investment * b.total_duration * b.profit_ratio_hari) / 
           (b.hari_investment * b.profit_ratio_praveen)
  x

/-- Theorem stating that Hari joined 5 months after Praveen started the business --/
theorem hari_joined_after_five_months (b : Business) 
  (h1 : b.praveen_investment = 3220)
  (h2 : b.hari_investment = 8280)
  (h3 : b.profit_ratio_praveen = 2)
  (h4 : b.profit_ratio_hari = 3)
  (h5 : b.total_duration = 12) :
  months_until_hari_joined b = 5 := by
  sorry

#eval months_until_hari_joined ⟨3220, 8280, 2, 3, 12⟩

end hari_joined_after_five_months_l3152_315222


namespace pure_imaginary_fraction_l3152_315274

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ (y : ℝ), (Complex.I : ℂ) * y = (1 - a * Complex.I) / (1 + Complex.I)) → a = 1 := by
  sorry

end pure_imaginary_fraction_l3152_315274


namespace quadratic_always_nonnegative_implies_m_range_l3152_315234

theorem quadratic_always_nonnegative_implies_m_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + 2*m - 3 ≥ 0) → 2 ≤ m ∧ m ≤ 6 := by
  sorry

end quadratic_always_nonnegative_implies_m_range_l3152_315234


namespace triangle_abc_properties_l3152_315219

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_abc_properties (t : Triangle) 
  (h1 : t.a = t.b * Real.cos t.C + (Real.sqrt 3 / 3) * t.c * Real.sin t.B)
  (h2 : t.a + t.c = 6)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 2) : 
  t.B = π/3 ∧ t.b = 3 * Real.sqrt 2 := by
  sorry

end triangle_abc_properties_l3152_315219


namespace back_seat_capacity_is_twelve_l3152_315232

/-- Represents the seating arrangement and capacity of a bus -/
structure BusSeating where
  left_seats : ℕ
  right_seats : ℕ
  people_per_seat : ℕ
  total_capacity : ℕ

/-- Calculates the number of people who can sit in the back seat of the bus -/
def back_seat_capacity (bus : BusSeating) : ℕ :=
  bus.total_capacity - (bus.left_seats + bus.right_seats) * bus.people_per_seat

/-- Theorem stating the number of people who can sit in the back seat -/
theorem back_seat_capacity_is_twelve :
  ∀ (bus : BusSeating),
    bus.left_seats = 15 →
    bus.right_seats = bus.left_seats - 3 →
    bus.people_per_seat = 3 →
    bus.total_capacity = 93 →
    back_seat_capacity bus = 12 := by
  sorry


end back_seat_capacity_is_twelve_l3152_315232


namespace savings_calculation_l3152_315226

/-- Calculates the total savings given a daily savings amount and number of days -/
def totalSavings (dailySavings : ℕ) (days : ℕ) : ℕ :=
  dailySavings * days

/-- Theorem: If a person saves $24 every day for 365 days, the total savings will be $8,760 -/
theorem savings_calculation :
  totalSavings 24 365 = 8760 := by
  sorry

end savings_calculation_l3152_315226
