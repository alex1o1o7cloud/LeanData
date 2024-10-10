import Mathlib

namespace school_height_ratio_l3279_327948

theorem school_height_ratio (total_avg : ℝ) (female_avg : ℝ) (male_avg : ℝ)
  (h_total : total_avg = 180)
  (h_female : female_avg = 170)
  (h_male : male_avg = 182) :
  ∃ (m w : ℝ), m > 0 ∧ w > 0 ∧ m / w = 5 ∧
    male_avg * m + female_avg * w = total_avg * (m + w) :=
by
  sorry

end school_height_ratio_l3279_327948


namespace sum_of_roots_quadratic_l3279_327941

theorem sum_of_roots_quadratic (m n : ℝ) : 
  (m ^ 2 - 3 * m - 1 = 0) → (n ^ 2 - 3 * n - 1 = 0) → m + n = 3 := by
  sorry

end sum_of_roots_quadratic_l3279_327941


namespace max_value_x_plus_reciprocal_l3279_327901

theorem max_value_x_plus_reciprocal (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 13 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 13 := by
  sorry

end max_value_x_plus_reciprocal_l3279_327901


namespace mcdonald_accounting_error_l3279_327957

theorem mcdonald_accounting_error (x : ℝ) : x = 3.57 ↔ 9 * x = 32.13 := by sorry

end mcdonald_accounting_error_l3279_327957


namespace circle_point_distance_range_l3279_327984

/-- Given a circle C with equation (x-a)^2 + (y-a+2)^2 = 1 and a point A(0,2),
    if there exists a point M on C such that MA^2 + MO^2 = 10,
    then 0 ≤ a ≤ 3. -/
theorem circle_point_distance_range (a : ℝ) :
  (∃ x y : ℝ, (x - a)^2 + (y - a + 2)^2 = 1 ∧ x^2 + y^2 + x^2 + (y - 2)^2 = 10) →
  0 ≤ a ∧ a ≤ 3 := by
  sorry

end circle_point_distance_range_l3279_327984


namespace binomial_expansion_problem_l3279_327900

theorem binomial_expansion_problem (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) 
  (h : ∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) :
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2) ∧
  (a₁ + a₃ + a₅ + a₇ = -1094) ∧
  (a₀ + a₂ + a₄ + a₆ = 1093) := by
  sorry

end binomial_expansion_problem_l3279_327900


namespace cube_with_cut_corners_has_44_edges_l3279_327975

/-- A cube with cut corners is a polyhedron obtained by cutting off each corner of a cube
    such that no two cutting planes intersect within the cube, and each corner cut
    removes a vertex and replaces it with a quadrilateral face. -/
structure CubeWithCutCorners where
  /-- The number of vertices in the original cube -/
  original_vertices : ℕ
  /-- The number of edges in the original cube -/
  original_edges : ℕ
  /-- The number of new edges introduced by each corner cut -/
  new_edges_per_cut : ℕ
  /-- The condition that the original shape is a cube -/
  is_cube : original_vertices = 8 ∧ original_edges = 12
  /-- The condition that each corner cut introduces 4 new edges -/
  corner_cut : new_edges_per_cut = 4

/-- The number of edges in the resulting figure after cutting off all corners of a cube -/
def num_edges_after_cuts (c : CubeWithCutCorners) : ℕ :=
  c.original_edges + c.original_vertices * c.new_edges_per_cut

/-- Theorem stating that a cube with cut corners has 44 edges -/
theorem cube_with_cut_corners_has_44_edges (c : CubeWithCutCorners) :
  num_edges_after_cuts c = 44 := by
  sorry

end cube_with_cut_corners_has_44_edges_l3279_327975


namespace perpendicular_line_through_point_l3279_327947

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_through_point (A : Point) (l1 l2 : Line) :
  A.x = 2 ∧ A.y = -3 ∧
  l1.a = 1 ∧ l1.b = -2 ∧ l1.c = -3 ∧
  l2.a = 2 ∧ l2.b = 1 ∧ l2.c = -1 →
  A.liesOn l2 ∧ l1.perpendicular l2 := by
  sorry

end perpendicular_line_through_point_l3279_327947


namespace adam_apple_purchase_l3279_327983

/-- The quantity of apples Adam bought on Monday -/
def monday_apples : ℕ := 15

/-- The quantity of apples Adam bought on Tuesday -/
def tuesday_apples : ℕ := 3 * monday_apples

/-- The quantity of apples Adam bought on Wednesday -/
def wednesday_apples : ℕ := 4 * tuesday_apples

/-- The total quantity of apples Adam bought on these three days -/
def total_apples : ℕ := monday_apples + tuesday_apples + wednesday_apples

theorem adam_apple_purchase : total_apples = 240 := by
  sorry

end adam_apple_purchase_l3279_327983


namespace perimeter_difference_is_zero_l3279_327902

/-- A figure composed of unit squares -/
structure UnitSquareFigure where
  squares : ℕ
  perimeter : ℕ

/-- T-shaped figure with 5 unit squares -/
def t_shape : UnitSquareFigure :=
  { squares := 5,
    perimeter := 8 }

/-- Cross-shaped figure with 5 unit squares -/
def cross_shape : UnitSquareFigure :=
  { squares := 5,
    perimeter := 8 }

/-- The positive difference between the perimeters of the T-shape and cross-shape is 0 -/
theorem perimeter_difference_is_zero :
  (t_shape.perimeter : ℤ) - (cross_shape.perimeter : ℤ) = 0 := by
  sorry

#check perimeter_difference_is_zero

end perimeter_difference_is_zero_l3279_327902


namespace juanita_sunscreen_usage_l3279_327938

/-- Proves that Juanita uses 1 bottle of sunscreen per month -/
theorem juanita_sunscreen_usage
  (months_per_year : ℕ)
  (discount_rate : ℚ)
  (bottle_cost : ℚ)
  (total_discounted_cost : ℚ)
  (h1 : months_per_year = 12)
  (h2 : discount_rate = 30 / 100)
  (h3 : bottle_cost = 30)
  (h4 : total_discounted_cost = 252) :
  (total_discounted_cost / ((1 - discount_rate) * bottle_cost)) / months_per_year = 1 :=
sorry

end juanita_sunscreen_usage_l3279_327938


namespace matrix_inverse_scalar_multiple_l3279_327950

theorem matrix_inverse_scalar_multiple (d k : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 4; 6, d]
  (A⁻¹ = k • A) → d = -1 ∧ k = (1 : ℝ) / 25 := by
  sorry

end matrix_inverse_scalar_multiple_l3279_327950


namespace equation_solution_l3279_327920

theorem equation_solution : 
  {x : ℝ | x^2 + (x-1)*(x+3) = 3*x + 5} = {-2, 2} := by sorry

end equation_solution_l3279_327920


namespace problem_solution_l3279_327907

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 6 * x^3 + 18 * x^2 * y * z = 3 * x^4 + 6 * x^3 * y * z) : x = 2 := by
  sorry

end problem_solution_l3279_327907


namespace virtual_set_divisors_l3279_327994

def isVirtual (A : Finset ℕ) : Prop :=
  A.card = 5 ∧
  (∀ (a b c : ℕ), a ∈ A → b ∈ A → c ∈ A → a ≠ b → b ≠ c → a ≠ c → Nat.gcd a (Nat.gcd b c) > 1) ∧
  (∀ (a b c d : ℕ), a ∈ A → b ∈ A → c ∈ A → d ∈ A → a ≠ b → b ≠ c → c ≠ d → a ≠ c → a ≠ d → b ≠ d → Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 1)

theorem virtual_set_divisors (A : Finset ℕ) (h : isVirtual A) :
  (Finset.prod A id).divisors.card ≥ 2020 := by
  sorry

end virtual_set_divisors_l3279_327994


namespace exponent_multiplication_l3279_327945

theorem exponent_multiplication (a : ℝ) : a^6 * a^2 = a^8 := by
  sorry

end exponent_multiplication_l3279_327945


namespace sqrt_product_sqrt_l3279_327964

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_product_sqrt_l3279_327964


namespace solution_sets_equal_implies_alpha_value_l3279_327953

/-- The solution set of the inequality |2x-3| < 2 -/
def solution_set_1 : Set ℝ := {x : ℝ | |2*x - 3| < 2}

/-- The solution set of the inequality x^2 + αx + b < 0 -/
def solution_set_2 (α b : ℝ) : Set ℝ := {x : ℝ | x^2 + α*x + b < 0}

/-- Theorem stating that if the solution sets are equal, then α = -3 -/
theorem solution_sets_equal_implies_alpha_value (b : ℝ) :
  (∃ α, solution_set_1 = solution_set_2 α b) → 
  (∃ α, solution_set_1 = solution_set_2 α b ∧ α = -3) :=
by sorry

end solution_sets_equal_implies_alpha_value_l3279_327953


namespace price_reduction_percentage_l3279_327997

/-- Given a coat with an original price and a price reduction, 
    calculate the percentage reduction in price. -/
theorem price_reduction_percentage 
  (original_price : ℝ) 
  (price_reduction : ℝ) 
  (h1 : original_price = 500)
  (h2 : price_reduction = 350) : 
  (price_reduction / original_price) * 100 = 70 := by
sorry

end price_reduction_percentage_l3279_327997


namespace solve_for_y_l3279_327987

theorem solve_for_y (x y : ℤ) (h1 : x^2 - x + 6 = y + 2) (h2 : x = -5) : y = 34 := by
  sorry

end solve_for_y_l3279_327987


namespace union_of_sets_l3279_327908

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 3, 5}
  A ∪ B = {1, 2, 3, 5} := by
sorry

end union_of_sets_l3279_327908


namespace tangent_line_of_quartic_curve_l3279_327969

/-- The curve y = x^4 has a tangent line parallel to x + 2y - 8 = 0, 
    and this tangent line has the equation 8x + 16y + 3 = 0 -/
theorem tangent_line_of_quartic_curve (x y : ℝ) : 
  y = x^4 → 
  ∃ (x₀ y₀ : ℝ), y₀ = x₀^4 ∧ 
    (∀ (x' y' : ℝ), y' - y₀ = 4 * x₀^3 * (x' - x₀) → 
      ∃ (k : ℝ), y' - y₀ = k * (x' - x₀) ∧ k = -1/2) →
    8 * x₀ + 16 * y₀ + 3 = 0 :=
by sorry

end tangent_line_of_quartic_curve_l3279_327969


namespace reciprocal_of_sum_l3279_327972

theorem reciprocal_of_sum (y : ℚ) : y = 6 + 1/6 → 1/y = 6/37 := by
  sorry

end reciprocal_of_sum_l3279_327972


namespace geometry_propositions_l3279_327932

theorem geometry_propositions (p₁ p₂ p₃ p₄ : Prop) 
  (h₁ : p₁) (h₂ : ¬p₂) (h₃ : ¬p₃) (h₄ : p₄) :
  {p₁ ∧ p₄, ¬p₂ ∨ p₃, ¬p₃ ∨ ¬p₄} = 
  {q : Prop | q = (p₁ ∧ p₄) ∨ q = (¬p₂ ∨ p₃) ∨ q = (¬p₃ ∨ ¬p₄)} ∩ 
  {q : Prop | q} :=
by sorry

end geometry_propositions_l3279_327932


namespace max_projection_area_unit_cube_max_projection_area_unit_cube_proof_l3279_327952

/-- The maximum area of the orthogonal projection of a unit cube onto any plane -/
theorem max_projection_area_unit_cube : ℝ :=
  2 * Real.sqrt 3

/-- Theorem: The maximum area of the orthogonal projection of a unit cube onto any plane is 2√3 -/
theorem max_projection_area_unit_cube_proof :
  max_projection_area_unit_cube = 2 * Real.sqrt 3 := by
  sorry

#check max_projection_area_unit_cube_proof

end max_projection_area_unit_cube_max_projection_area_unit_cube_proof_l3279_327952


namespace range_of_a_l3279_327913

open Set

theorem range_of_a (p : ∀ x ∈ Icc 1 2, x^2 - a ≥ 0) 
                   (q : ∃ x₀ : ℝ, x₀ + 2*a*x₀ + 2 - a = 0) : 
  a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l3279_327913


namespace max_value_on_circle_l3279_327942

theorem max_value_on_circle (x y : ℝ) :
  x^2 + y^2 = 10 →
  ∃ (max : ℝ), max = 5 * Real.sqrt 10 ∧ ∀ (a b : ℝ), a^2 + b^2 = 10 → 3*a + 4*b ≤ max :=
sorry

end max_value_on_circle_l3279_327942


namespace vector_independence_l3279_327963

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b (m : ℝ) : Fin 2 → ℝ := ![m - 1, m + 3]

theorem vector_independence (m : ℝ) :
  LinearIndependent ℝ ![vector_a, vector_b m] ↔ m ≠ 5 := by
  sorry

end vector_independence_l3279_327963


namespace sqrt2_plus_sqrt3_irrational_l3279_327936

theorem sqrt2_plus_sqrt3_irrational : Irrational (Real.sqrt 2 + Real.sqrt 3) := by
  sorry

end sqrt2_plus_sqrt3_irrational_l3279_327936


namespace investment_schemes_count_l3279_327995

/-- The number of projects to invest in -/
def num_projects : ℕ := 3

/-- The number of candidate cities -/
def num_cities : ℕ := 4

/-- The maximum number of projects allowed in a single city -/
def max_projects_per_city : ℕ := 2

/-- Calculates the number of investment schemes -/
def num_investment_schemes : ℕ := sorry

/-- Theorem stating that the number of investment schemes is 60 -/
theorem investment_schemes_count :
  num_investment_schemes = 60 := by sorry

end investment_schemes_count_l3279_327995


namespace necessary_not_sufficient_condition_l3279_327909

theorem necessary_not_sufficient_condition :
  (∀ a b c d : ℝ, a + b < c + d → (a < c ∨ b < d)) ∧
  (∃ a b c d : ℝ, (a < c ∨ b < d) ∧ ¬(a + b < c + d)) := by
  sorry

end necessary_not_sufficient_condition_l3279_327909


namespace candidate_vote_percentage_l3279_327966

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (vote_difference : ℕ) 
  (candidate_percentage : ℚ) :
  total_votes = 7900 →
  vote_difference = 2370 →
  candidate_percentage = total_votes.cast / 100 * 
    ((total_votes.cast - vote_difference.cast) / (2 * total_votes.cast)) →
  candidate_percentage = 35 := by
sorry

end candidate_vote_percentage_l3279_327966


namespace expression_value_l3279_327976

theorem expression_value : 
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  x^2 + y^2 + z^2 + 2*x*y*z = 2 := by
sorry

end expression_value_l3279_327976


namespace problem_1_problem_2_l3279_327998

variable (a b : ℝ)

theorem problem_1 : 2 * a * (a^2 - 3*a - 1) = 2*a^3 - 6*a^2 - 2*a := by
  sorry

theorem problem_2 : (a^2*b - 2*a*b^2 + b^3) / b - (a + b)^2 = -4*a*b := by
  sorry

end problem_1_problem_2_l3279_327998


namespace base_10_to_base_6_l3279_327927

theorem base_10_to_base_6 : 
  (1 * 6^4 + 3 * 6^3 + 0 * 6^2 + 5 * 6^1 + 4 * 6^0 : ℕ) = 1978 := by
  sorry

#eval 1 * 6^4 + 3 * 6^3 + 0 * 6^2 + 5 * 6^1 + 4 * 6^0

end base_10_to_base_6_l3279_327927


namespace andrew_total_work_hours_l3279_327922

/-- The total hours Andrew worked on his Science report over three days -/
def total_hours (day1 day2 day3 : Real) : Real :=
  day1 + day2 + day3

/-- Theorem stating that Andrew worked 9.25 hours in total -/
theorem andrew_total_work_hours :
  let day1 : Real := 2.5
  let day2 : Real := day1 + 0.5
  let day3 : Real := 3.75
  total_hours day1 day2 day3 = 9.25 := by
  sorry

end andrew_total_work_hours_l3279_327922


namespace black_circle_area_black_circle_area_proof_l3279_327999

theorem black_circle_area (cube_edge : Real) (yellow_paint_area : Real) : Real :=
  let cube_face_area := cube_edge ^ 2
  let total_surface_area := 6 * cube_face_area
  let yellow_area_per_face := yellow_paint_area / 6
  let black_circle_area := cube_face_area - yellow_area_per_face
  
  black_circle_area

theorem black_circle_area_proof :
  black_circle_area 12 432 = 72 := by
  sorry

end black_circle_area_black_circle_area_proof_l3279_327999


namespace revenue_change_l3279_327911

/-- Given a projected revenue increase and the ratio of actual to projected revenue,
    calculate the actual percent change in revenue. -/
theorem revenue_change
  (projected_increase : ℝ)
  (actual_to_projected_ratio : ℝ)
  (h1 : projected_increase = 0.20)
  (h2 : actual_to_projected_ratio = 0.75) :
  (1 + projected_increase) * actual_to_projected_ratio - 1 = -0.10 := by
  sorry

#check revenue_change

end revenue_change_l3279_327911


namespace august_math_problems_l3279_327960

theorem august_math_problems (first_answer second_answer third_answer : ℕ) : 
  first_answer = 600 →
  second_answer = 2 * first_answer →
  third_answer = first_answer + second_answer - 400 →
  first_answer + second_answer + third_answer = 3200 := by
sorry

end august_math_problems_l3279_327960


namespace female_democrats_count_l3279_327917

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) 
  (h1 : female + male = total)
  (h2 : total = 720)
  (h3 : female / 2 + male / 4 = total / 3) :
  female / 2 = 120 := by
  sorry

end female_democrats_count_l3279_327917


namespace circle_line_bisection_implies_mn_range_l3279_327967

/-- The circle equation: x^2 + y^2 - 4x - 2y - 4 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 4 = 0

/-- The line equation: mx + 2ny - 4 = 0 -/
def line_equation (m n x y : ℝ) : Prop :=
  m*x + 2*n*y - 4 = 0

/-- The line bisects the perimeter of the circle -/
def line_bisects_circle (m n : ℝ) : Prop :=
  ∀ x y, circle_equation x y → line_equation m n x y

/-- The range of mn is (-∞, 1] -/
def mn_range (m n : ℝ) : Prop :=
  m * n ≤ 1

theorem circle_line_bisection_implies_mn_range :
  ∀ m n, line_bisects_circle m n → mn_range m n :=
sorry

end circle_line_bisection_implies_mn_range_l3279_327967


namespace hcf_problem_l3279_327910

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 2460) (h2 : Nat.lcm a b = 205) :
  Nat.gcd a b = 12 := by
  sorry

end hcf_problem_l3279_327910


namespace certain_number_proof_l3279_327986

theorem certain_number_proof : 
  ∃ x : ℝ, (15 / 100) * x = (2.5 / 100) * 450 ∧ x = 75 := by
  sorry

end certain_number_proof_l3279_327986


namespace inscribed_sphere_sum_l3279_327943

/-- A sphere inscribed in a right cone with base radius 15 cm and height 30 cm -/
structure InscribedSphere :=
  (b : ℝ)
  (d : ℝ)
  (radius : ℝ)
  (cone_base_radius : ℝ)
  (cone_height : ℝ)
  (radius_eq : radius = b * (Real.sqrt d - 1))
  (cone_base_radius_eq : cone_base_radius = 15)
  (cone_height_eq : cone_height = 30)

/-- Theorem stating that b + d = 12.5 for the inscribed sphere -/
theorem inscribed_sphere_sum (s : InscribedSphere) : s.b + s.d = 12.5 := by
  sorry

end inscribed_sphere_sum_l3279_327943


namespace molecular_weight_Al2S3_l3279_327974

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_S : ℝ := 32.06

-- Define the composition of Al2S3
def Al_atoms_in_Al2S3 : ℕ := 2
def S_atoms_in_Al2S3 : ℕ := 3

-- Define the number of moles
def moles_Al2S3 : ℝ := 3

-- Theorem statement
theorem molecular_weight_Al2S3 :
  let molecular_weight_one_mole := Al_atoms_in_Al2S3 * atomic_weight_Al + S_atoms_in_Al2S3 * atomic_weight_S
  moles_Al2S3 * molecular_weight_one_mole = 450.42 := by
  sorry

end molecular_weight_Al2S3_l3279_327974


namespace sleep_hours_calculation_l3279_327924

def hours_in_day : ℕ := 24
def work_hours : ℕ := 6
def chore_hours : ℕ := 5

theorem sleep_hours_calculation :
  hours_in_day - (work_hours + chore_hours) = 13 := by
  sorry

end sleep_hours_calculation_l3279_327924


namespace chef_apples_used_l3279_327935

/-- The number of apples the chef used to make pies -/
def applesUsed (initialApples remainingApples : ℕ) : ℕ :=
  initialApples - remainingApples

theorem chef_apples_used :
  let initialApples : ℕ := 43
  let remainingApples : ℕ := 2
  applesUsed initialApples remainingApples = 41 := by
  sorry

end chef_apples_used_l3279_327935


namespace binary_sum_equals_expected_l3279_327930

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

def b1 : List Bool := [false, true, true, false, true]  -- 10110₂
def b2 : List Bool := [false, true, true]               -- 110₂
def b3 : List Bool := [true]                            -- 1₂
def b4 : List Bool := [true, false, true]               -- 101₂

def expected_sum : List Bool := [false, false, false, false, true, true]  -- 110000₂

theorem binary_sum_equals_expected :
  nat_to_binary (binary_to_nat b1 + binary_to_nat b2 + binary_to_nat b3 + binary_to_nat b4) = expected_sum :=
by sorry

end binary_sum_equals_expected_l3279_327930


namespace geometric_sequence_a4_l3279_327959

/-- A geometric sequence with specified terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  geometric_sequence a → a 2 = -2 → a 6 = -32 → a 4 = -8 := by
  sorry

end geometric_sequence_a4_l3279_327959


namespace gain_amount_proof_l3279_327903

/-- Given an article sold at $180 with a 20% gain, prove that the gain amount is $30. -/
theorem gain_amount_proof (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 180)
  (h2 : gain_percentage = 0.20) : 
  let cost_price := selling_price / (1 + gain_percentage)
  selling_price - cost_price = 30 := by
sorry


end gain_amount_proof_l3279_327903


namespace fraction_reduction_l3279_327905

theorem fraction_reduction (a b d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hd : d ≠ 0) (hsum : a + b + d ≠ 0) :
  (a^2 + b^2 - d^2 + 2*a*b) / (a^2 + d^2 - b^2 + 2*a*d) = (a + b - d) / (a + d - b) := by
  sorry

end fraction_reduction_l3279_327905


namespace incorrect_statement_about_converses_l3279_327955

/-- A proposition in mathematics -/
structure Proposition where
  statement : Prop

/-- A theorem in mathematics -/
structure Theorem where
  statement : Prop
  proof : statement

/-- The converse of a proposition -/
def converse (p : Proposition) : Proposition :=
  ⟨¬p.statement⟩

theorem incorrect_statement_about_converses :
  ¬(∀ (p : Proposition), ∃ (c : Proposition), c = converse p ∧
     ∃ (t : Theorem), ¬∃ (c : Proposition), c = converse ⟨t.statement⟩) := by
  sorry

end incorrect_statement_about_converses_l3279_327955


namespace student_arrangement_theorem_l3279_327981

/-- The number of ways to arrange 3 male and 2 female students in a row -/
def total_arrangements : ℕ := 120

/-- The number of arrangements where exactly two male students are adjacent -/
def two_male_adjacent : ℕ := 72

/-- The number of arrangements where 3 male students of different heights 
    are arranged in descending order of height -/
def male_descending_height : ℕ := 20

/-- Given 3 male students and 2 female students, prove:
    1. The total number of arrangements
    2. The number of arrangements with exactly two male students adjacent
    3. The number of arrangements with male students in descending height order -/
theorem student_arrangement_theorem 
  (male_count : ℕ) 
  (female_count : ℕ) 
  (h1 : male_count = 3) 
  (h2 : female_count = 2) :
  (total_arrangements = 120) ∧ 
  (two_male_adjacent = 72) ∧ 
  (male_descending_height = 20) := by
  sorry

end student_arrangement_theorem_l3279_327981


namespace triangle_theorem_l3279_327933

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.c * Real.sin t.A = t.a * Real.cos t.C)
  (h2 : 0 < t.A ∧ t.A < Real.pi)
  (h3 : 0 < t.B ∧ t.B < Real.pi)
  (h4 : 0 < t.C ∧ t.C < Real.pi)
  (h5 : t.A + t.B + t.C = Real.pi) : 
  (t.C = Real.pi / 4) ∧ 
  (∃ (max : Real), ∀ (A B : Real), 
    (0 < A ∧ A < 3 * Real.pi / 4) → 
    (B = 3 * Real.pi / 4 - A) → 
    (Real.sqrt 3 * Real.sin A - Real.cos (B + Real.pi / 4) ≤ max) ∧
    (max = 2)) ∧
  (Real.sqrt 3 * Real.sin (Real.pi / 3) - Real.cos (5 * Real.pi / 12 + Real.pi / 4) = 2) := by
  sorry

end triangle_theorem_l3279_327933


namespace remaining_laps_after_break_l3279_327926

/-- The number of laps Jeff needs to swim over the weekend -/
def total_laps : ℕ := 98

/-- The number of laps Jeff swam on Saturday -/
def saturday_laps : ℕ := 27

/-- The number of laps Jeff swam on Sunday morning -/
def sunday_morning_laps : ℕ := 15

/-- Theorem stating the number of laps remaining after Jeff's break on Sunday -/
theorem remaining_laps_after_break : 
  total_laps - saturday_laps - sunday_morning_laps = 56 := by
  sorry

end remaining_laps_after_break_l3279_327926


namespace set_inclusion_condition_l3279_327970

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ a}
def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 2 * x + 3}
def C (a : ℝ) : Set ℝ := {z | ∃ x ∈ A a, z = x ^ 2}

-- State the theorem
theorem set_inclusion_condition (a : ℝ) :
  C a ⊆ B a ↔ (1/2 ≤ a ∧ a ≤ 2) ∨ (a ≥ 3) ∨ (a < -2) :=
sorry

end set_inclusion_condition_l3279_327970


namespace existence_of_irrational_powers_with_integer_result_l3279_327904

theorem existence_of_irrational_powers_with_integer_result :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ Irrational a ∧ Irrational b ∧ ∃ (n : ℤ), a^b = n := by
  sorry

end existence_of_irrational_powers_with_integer_result_l3279_327904


namespace store_profit_l3279_327990

/-- Prove that a store makes a profit when selling pens purchased from two markets -/
theorem store_profit (m n : ℝ) (h : m > n) : 
  let selling_price := (m + n) / 2
  let profit_A := 40 * (selling_price - m)
  let profit_B := 60 * (selling_price - n)
  profit_A + profit_B > 0 := by
  sorry


end store_profit_l3279_327990


namespace total_wheels_count_l3279_327925

def bicycle_count : Nat := 3
def tricycle_count : Nat := 4
def unicycle_count : Nat := 7

def bicycle_wheels : Nat := 2
def tricycle_wheels : Nat := 3
def unicycle_wheels : Nat := 1

theorem total_wheels_count : 
  bicycle_count * bicycle_wheels + 
  tricycle_count * tricycle_wheels + 
  unicycle_count * unicycle_wheels = 25 := by
  sorry

end total_wheels_count_l3279_327925


namespace first_day_over_threshold_l3279_327989

/-- The number of paperclips Max starts with on Monday -/
def initial_paperclips : ℕ := 3

/-- The factor by which the number of paperclips increases each day -/
def daily_increase_factor : ℕ := 4

/-- The threshold number of paperclips -/
def threshold : ℕ := 200

/-- The function that calculates the number of paperclips on day n -/
def paperclips (n : ℕ) : ℕ := initial_paperclips * daily_increase_factor^(n - 1)

/-- The theorem stating that the 5th day is the first day with more than 200 paperclips -/
theorem first_day_over_threshold :
  ∀ n : ℕ, n > 0 → (paperclips n > threshold ↔ n ≥ 5) :=
sorry

end first_day_over_threshold_l3279_327989


namespace smallest_part_of_proportional_division_l3279_327988

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let x := total / (a + b + c)
  let part1 := a * x
  let part2 := b * x
  let part3 := c * x
  (total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7) →
  min part1 (min part2 part3) = 24 :=
by sorry

end smallest_part_of_proportional_division_l3279_327988


namespace circle_not_in_second_quadrant_l3279_327993

/-- A circle in the xy-plane with center (a, 0) and radius 2 -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = 4}

/-- The second quadrant of the xy-plane -/
def SecondQuadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}

/-- The circle does not pass through the second quadrant -/
def NotInSecondQuadrant (a : ℝ) : Prop :=
  Circle a ∩ SecondQuadrant = ∅

theorem circle_not_in_second_quadrant (a : ℝ) :
  NotInSecondQuadrant a → a ≥ 2 := by
  sorry


end circle_not_in_second_quadrant_l3279_327993


namespace darren_tshirts_l3279_327971

/-- The number of packs of white t-shirts Darren bought -/
def white_packs : ℕ := 5

/-- The number of t-shirts in each pack of white t-shirts -/
def white_per_pack : ℕ := 6

/-- The number of packs of blue t-shirts Darren bought -/
def blue_packs : ℕ := 3

/-- The number of t-shirts in each pack of blue t-shirts -/
def blue_per_pack : ℕ := 9

/-- The total number of t-shirts Darren bought -/
def total_tshirts : ℕ := white_packs * white_per_pack + blue_packs * blue_per_pack

theorem darren_tshirts : total_tshirts = 57 := by
  sorry

end darren_tshirts_l3279_327971


namespace triangle_side_calculation_l3279_327991

theorem triangle_side_calculation (AB : ℝ) : 
  AB = 10 →
  ∃ (AC AD CD : ℝ),
    -- ABD is a 45-45-90 triangle
    AD = AB ∧
    -- ACD is a 30-60-90 triangle
    CD = 2 * AC ∧
    AD^2 = AC^2 + CD^2 ∧
    -- The result we want to prove
    CD = 10 * Real.sqrt 3 := by
  sorry

end triangle_side_calculation_l3279_327991


namespace reciprocal_roots_equation_l3279_327918

theorem reciprocal_roots_equation (m n : ℝ) (hn : n ≠ 0) :
  let original_eq := fun x => x^2 + m*x + n
  let reciprocal_eq := fun x => n*x^2 + m*x + 1
  ∀ x, original_eq x = 0 → reciprocal_eq (1/x) = 0 :=
sorry


end reciprocal_roots_equation_l3279_327918


namespace line_passes_through_fixed_point_l3279_327985

/-- A line that does not pass through the origin -/
structure Line where
  slope : ℝ
  intercept : ℝ
  not_through_origin : intercept ≠ 0

/-- A point on the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y = x^2 -/
def parabola (p : Point) : Prop :=
  p.y = p.x^2

/-- The line intersects the parabola at two points -/
def intersects_parabola (l : Line) (A B : Point) : Prop :=
  parabola A ∧ parabola B ∧
  A.y = l.slope * A.x + l.intercept ∧
  B.y = l.slope * B.x + l.intercept ∧
  A ≠ B

/-- The circle with diameter AB passes through the origin -/
def circle_through_origin (A B : Point) : Prop :=
  A.x * B.x + A.y * B.y = 0

/-- The main theorem -/
theorem line_passes_through_fixed_point (l : Line) (A B : Point)
  (h_intersects : intersects_parabola l A B)
  (h_circle : circle_through_origin A B) :
  ∃ (P : Point), P.x = 0 ∧ P.y = 1 ∧ P.y = l.slope * P.x + l.intercept :=
sorry

end line_passes_through_fixed_point_l3279_327985


namespace unique_triple_solution_l3279_327919

theorem unique_triple_solution (a b c : ℝ) : 
  a > 5 → b > 5 → c > 5 →
  (a + 3)^2 / (b + c - 3) + (b + 6)^2 / (c + a - 6) + (c + 9)^2 / (a + b - 9) = 81 →
  a = 15 ∧ b = 12 ∧ c = 9 := by
sorry

end unique_triple_solution_l3279_327919


namespace weight_difference_is_19_l3279_327956

/-- The combined weight difference between the lightest and heaviest individual -/
def weightDifference (john roy derek samantha : ℕ) : ℕ :=
  max john (max roy (max derek samantha)) - min john (min roy (min derek samantha))

/-- Theorem: The combined weight difference between the lightest and heaviest individual is 19 pounds -/
theorem weight_difference_is_19 :
  weightDifference 81 79 91 72 = 19 := by
  sorry

end weight_difference_is_19_l3279_327956


namespace product_of_distinct_solutions_l3279_327923

theorem product_of_distinct_solutions (x y : ℝ) : 
  x ≠ 0 → y ≠ 0 → x ≠ y → (x + 3 / x = y + 3 / y) → x * y = 3 := by
  sorry

end product_of_distinct_solutions_l3279_327923


namespace ending_number_is_300_l3279_327949

theorem ending_number_is_300 (ending_number : ℕ) : 
  (∃ (multiples : List ℕ), 
    multiples.length = 67 ∧ 
    (∀ n ∈ multiples, n % 3 = 0) ∧
    (∀ n ∈ multiples, 100 ≤ n ∧ n ≤ ending_number) ∧
    (∀ n, 100 ≤ n ∧ n ≤ ending_number ∧ n % 3 = 0 → n ∈ multiples)) →
  ending_number = 300 := by
sorry

end ending_number_is_300_l3279_327949


namespace least_sum_of_bases_l3279_327954

theorem least_sum_of_bases : ∃ (c d : ℕ+), 
  (∀ (c' d' : ℕ+), (2 * c' + 9 = 9 * d' + 2) → (c'.val + d'.val ≥ c.val + d.val)) ∧ 
  (2 * c + 9 = 9 * d + 2) ∧
  (c.val + d.val = 13) := by
  sorry

end least_sum_of_bases_l3279_327954


namespace basil_planter_problem_l3279_327912

theorem basil_planter_problem (total_seeds : ℕ) (num_large_planters : ℕ) (large_planter_capacity : ℕ) (small_planter_capacity : ℕ) 
  (h1 : total_seeds = 200)
  (h2 : num_large_planters = 4)
  (h3 : large_planter_capacity = 20)
  (h4 : small_planter_capacity = 4) :
  (total_seeds - num_large_planters * large_planter_capacity) / small_planter_capacity = 30 := by
  sorry

end basil_planter_problem_l3279_327912


namespace age_difference_proof_l3279_327937

/-- Given two persons with an age difference of 16 years, where the elder is currently 30 years old,
    this theorem proves that 6 years ago, the elder person was three times as old as the younger one. -/
theorem age_difference_proof :
  ∀ (younger_age elder_age : ℕ) (years_ago : ℕ),
    elder_age = 30 →
    elder_age = younger_age + 16 →
    elder_age - years_ago = 3 * (younger_age - years_ago) →
    years_ago = 6 := by
  sorry

end age_difference_proof_l3279_327937


namespace intersection_point_x_coordinate_l3279_327979

theorem intersection_point_x_coordinate 
  (a b : ℝ) 
  (h1 : a ≠ b) 
  (h2 : ∃! x y : ℝ, x^2 + 2*a*x + 6*b = x^2 + 2*b*x + 6*a) : 
  ∃ x y : ℝ, x^2 + 2*a*x + 6*b = x^2 + 2*b*x + 6*a ∧ x = 3 :=
by sorry

end intersection_point_x_coordinate_l3279_327979


namespace angle_difference_l3279_327928

theorem angle_difference (a β : ℝ) 
  (h1 : 3 * Real.sin a - Real.cos a = 0)
  (h2 : 7 * Real.sin β + Real.cos β = 0)
  (h3 : 0 < a) (h4 : a < Real.pi / 2)
  (h5 : Real.pi / 2 < β) (h6 : β < Real.pi) :
  2 * a - β = - 3 * Real.pi / 4 := by
  sorry

end angle_difference_l3279_327928


namespace fraction_equation_solution_l3279_327958

theorem fraction_equation_solution (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2/11 := by
  sorry

end fraction_equation_solution_l3279_327958


namespace power_two_greater_than_sum_of_powers_l3279_327982

theorem power_two_greater_than_sum_of_powers (n : ℕ) (x : ℝ) 
  (h1 : n ≥ 2) (h2 : |x| < 1) : 
  2^n > (1 - x)^n + (1 + x)^n := by
  sorry

end power_two_greater_than_sum_of_powers_l3279_327982


namespace product_digit_count_l3279_327934

theorem product_digit_count (k n : ℕ) (a b : ℕ) :
  (10^(k-1) ≤ a ∧ a < 10^k) →
  (10^(n-1) ≤ b ∧ b < 10^n) →
  (10^(k+n-1) ≤ a * b ∧ a * b < 10^(k+n+1)) :=
sorry

end product_digit_count_l3279_327934


namespace basketball_team_allocation_schemes_l3279_327915

theorem basketball_team_allocation_schemes (n : ℕ) (k : ℕ) (m : ℕ) 
  (h1 : n = 8)  -- number of classes
  (h2 : k = 10) -- total number of players
  (h3 : m = k - n) -- remaining spots after each class contributes one player
  : (n.choose 2) + (n.choose 1) = 36 := by
  sorry

end basketball_team_allocation_schemes_l3279_327915


namespace average_age_increase_l3279_327951

theorem average_age_increase (initial_men : ℕ) (replaced_men_ages : List ℕ) (women_avg_age : ℚ) : 
  initial_men = 8 →
  replaced_men_ages = [20, 10] →
  women_avg_age = 23 →
  (((initial_men : ℚ) * women_avg_age + (women_avg_age * 2 - replaced_men_ages.sum)) / initial_men) - women_avg_age = 2 := by
  sorry

end average_age_increase_l3279_327951


namespace large_font_pages_l3279_327914

/-- Represents the number of words per page for large font -/
def large_font_words_per_page : ℕ := 1800

/-- Represents the number of words per page for small font -/
def small_font_words_per_page : ℕ := 2400

/-- Represents the total number of pages allowed -/
def total_pages : ℕ := 21

/-- Represents the ratio of large font pages to small font pages -/
def font_ratio : Rat := 2 / 3

theorem large_font_pages : ℕ :=
  let large_pages : ℕ := 8
  let small_pages : ℕ := total_pages - large_pages
  have h1 : large_pages + small_pages = total_pages := by sorry
  have h2 : (large_pages : Rat) / (small_pages : Rat) = font_ratio := by sorry
  have h3 : large_pages * large_font_words_per_page + small_pages * small_font_words_per_page ≤ 48000 := by sorry
  large_pages

end large_font_pages_l3279_327914


namespace proposition_A_sufficient_not_necessary_l3279_327978

/-- Defines a geometric sequence of three real numbers -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  (b ^ 2 = a * c) ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

theorem proposition_A_sufficient_not_necessary :
  (∀ a b c : ℝ, b ^ 2 ≠ a * c → ¬ is_geometric_sequence a b c) ∧
  (∃ a b c : ℝ, ¬ is_geometric_sequence a b c ∧ b ^ 2 = a * c) :=
by sorry

end proposition_A_sufficient_not_necessary_l3279_327978


namespace group_size_proof_l3279_327968

theorem group_size_proof (total_spent : ℕ) (mango_price : ℕ) (pineapple_price : ℕ) (pineapple_spent : ℕ) :
  total_spent = 94 →
  mango_price = 5 →
  pineapple_price = 6 →
  pineapple_spent = 54 →
  ∃ (mango_count pineapple_count : ℕ),
    mango_count * mango_price + pineapple_count * pineapple_price = total_spent ∧
    pineapple_count * pineapple_price = pineapple_spent ∧
    mango_count + pineapple_count = 17 :=
by
  sorry

end group_size_proof_l3279_327968


namespace largest_intersection_point_l3279_327931

/-- The polynomial function -/
def polynomial (a : ℝ) (x : ℝ) : ℝ := x^6 - 8*x^5 + 22*x^4 + 6*x^3 + a*x^2

/-- The line function -/
def line (c : ℝ) (x : ℝ) : ℝ := 2*x + c

/-- The intersection function -/
def intersection (a c : ℝ) (x : ℝ) : ℝ := polynomial a x - line c x

theorem largest_intersection_point (a c : ℝ) :
  (∃ p q : ℝ, p ≠ q ∧ 
    (∀ x : ℝ, intersection a c x = 0 ↔ (x = p ∨ x = q)) ∧
    (∀ x : ℝ, (x - p)^3 * (x - q) = intersection a c x)) →
  (∀ x : ℝ, intersection a c x = 0 → x ≤ 7) ∧
  (∃ x : ℝ, intersection a c x = 0 ∧ x = 7) :=
by sorry

end largest_intersection_point_l3279_327931


namespace special_numbers_l3279_327961

def last_digit (n : ℕ) : ℕ := n % 10

theorem special_numbers : 
  {n : ℕ | (last_digit n) * 2016 = n} = {4032, 8064, 12096, 16128} :=
by sorry

end special_numbers_l3279_327961


namespace right_triangle_ratio_square_l3279_327940

theorem right_triangle_ratio_square (a c p : ℝ) (h1 : a > 0) (h2 : c > 0) (h3 : p > 0) : 
  (c / a = a / p) → (c^2 = a^2 + p^2) → ((c / a)^2 = (1 + Real.sqrt 5) / 2) := by
  sorry

end right_triangle_ratio_square_l3279_327940


namespace butterfat_mixture_l3279_327977

/-- Proves that adding 16 gallons of 10% butterfat milk to 8 gallons of 40% butterfat milk 
    results in a mixture with 20% butterfat. -/
theorem butterfat_mixture : 
  let initial_volume : ℝ := 8
  let initial_butterfat_percent : ℝ := 40
  let added_volume : ℝ := 16
  let added_butterfat_percent : ℝ := 10
  let final_butterfat_percent : ℝ := 20
  let total_volume := initial_volume + added_volume
  let total_butterfat := (initial_volume * initial_butterfat_percent / 100) + 
                         (added_volume * added_butterfat_percent / 100)
  (total_butterfat / total_volume) * 100 = final_butterfat_percent :=
by
  sorry

#check butterfat_mixture

end butterfat_mixture_l3279_327977


namespace expression_evaluation_l3279_327939

theorem expression_evaluation :
  100 + (120 / 15) + (18 * 20) - 250 - (360 / 12) = 188 := by
  sorry

end expression_evaluation_l3279_327939


namespace min_packs_for_event_l3279_327946

/-- Represents a pack of utensils -/
structure UtensilPack where
  total : Nat
  knife : Nat
  fork : Nat
  spoon : Nat
  equal_distribution : knife = fork ∧ fork = spoon
  pack_size : total = knife + fork + spoon

/-- Represents the required ratio of utensils -/
structure UtensilRatio where
  knife : Nat
  fork : Nat
  spoon : Nat

def min_packs_needed (pack : UtensilPack) (ratio : UtensilRatio) (min_spoons : Nat) : Nat :=
  sorry

theorem min_packs_for_event (pack : UtensilPack) (ratio : UtensilRatio) (min_spoons : Nat) :
  pack.total = 30 ∧
  ratio.knife = 2 ∧ ratio.fork = 3 ∧ ratio.spoon = 5 ∧
  min_spoons = 50 →
  min_packs_needed pack ratio min_spoons = 5 :=
sorry

end min_packs_for_event_l3279_327946


namespace subtract_negative_l3279_327929

theorem subtract_negative : -2 - (-3) = 1 := by
  sorry

end subtract_negative_l3279_327929


namespace circle_equation_l3279_327996

/-- A circle with center (a, 1) that is tangent to both lines x-y+1=0 and x-y-3=0 -/
structure TangentCircle where
  a : ℝ
  center : ℝ × ℝ
  tangent_line1 : ℝ → ℝ → ℝ
  tangent_line2 : ℝ → ℝ → ℝ
  center_def : center = (a, 1)
  tangent_line1_def : tangent_line1 = fun x y => x - y + 1
  tangent_line2_def : tangent_line2 = fun x y => x - y - 3
  is_tangent1 : ∃ (x y : ℝ), tangent_line1 x y = 0 ∧ (x - a)^2 + (y - 1)^2 = (x - center.1)^2 + (y - center.2)^2
  is_tangent2 : ∃ (x y : ℝ), tangent_line2 x y = 0 ∧ (x - a)^2 + (y - 1)^2 = (x - center.1)^2 + (y - center.2)^2

/-- The standard equation of the circle is (x-2)^2+(y-1)^2=2 -/
theorem circle_equation (c : TangentCircle) : 
  ∃ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 2 ∧ 
  (x - c.center.1)^2 + (y - c.center.2)^2 = (x - 2)^2 + (y - 1)^2 := by
  sorry

end circle_equation_l3279_327996


namespace magnified_diameter_is_0_3_l3279_327973

/-- The magnification factor of the electron microscope -/
def magnification : ℝ := 1000

/-- The actual diameter of the tissue in centimeters -/
def actual_diameter : ℝ := 0.0003

/-- The diameter of the magnified image in centimeters -/
def magnified_diameter : ℝ := actual_diameter * magnification

/-- Theorem stating that the magnified diameter is 0.3 centimeters -/
theorem magnified_diameter_is_0_3 : magnified_diameter = 0.3 := by
  sorry

end magnified_diameter_is_0_3_l3279_327973


namespace equation_C_is_linear_l3279_327944

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x + 3 = 7 is linear -/
theorem equation_C_is_linear : is_linear_equation (λ x => 2 * x + 3) :=
by
  sorry

#check equation_C_is_linear

end equation_C_is_linear_l3279_327944


namespace inequality_implies_bound_l3279_327962

theorem inequality_implies_bound (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 3, x^2 - a*x + 4 ≥ 0) → a ≤ 4 := by
  sorry

end inequality_implies_bound_l3279_327962


namespace mark_deck_project_cost_l3279_327980

/-- Calculates the total cost of a multi-layered deck project --/
def deck_project_cost (length width : ℝ) 
                      (material_a_cost material_b_cost material_c_cost : ℝ) 
                      (beam_cost sealant_cost : ℝ) 
                      (railing_cost_30 railing_cost_40 : ℝ) 
                      (tax_rate : ℝ) : ℝ :=
  let area := length * width
  let material_cost := area * (material_a_cost + material_b_cost + material_c_cost)
  let beam_cost_total := area * beam_cost * 2
  let sealant_cost_total := area * sealant_cost
  let railing_cost_total := 2 * (railing_cost_30 + railing_cost_40)
  let subtotal := material_cost + beam_cost_total + sealant_cost_total + railing_cost_total
  let tax := subtotal * tax_rate
  subtotal + tax

/-- The total cost of Mark's deck project is $25423.20 --/
theorem mark_deck_project_cost : 
  deck_project_cost 30 40 3 5 8 2 1 120 160 0.07 = 25423.20 := by
  sorry

end mark_deck_project_cost_l3279_327980


namespace wall_height_calculation_l3279_327965

/-- Calculates the height of a wall given its dimensions and the number and size of bricks used --/
theorem wall_height_calculation (brick_length brick_width brick_height : ℝ)
  (wall_length wall_width : ℝ) (num_bricks : ℕ) :
  brick_length = 20 →
  brick_width = 10 →
  brick_height = 7.5 →
  wall_length = 27 →
  wall_width = 2 →
  num_bricks = 27000 →
  ∃ (wall_height : ℝ), wall_height = 0.75 ∧
    wall_length * wall_width * wall_height = (brick_length * brick_width * brick_height * num_bricks) / 1000000 := by
  sorry

#check wall_height_calculation

end wall_height_calculation_l3279_327965


namespace workshop_workers_l3279_327992

/-- The total number of workers in a workshop given specific salary conditions -/
theorem workshop_workers (average_salary : ℚ) (technician_count : ℕ) (technician_salary : ℚ) (rest_salary : ℚ) : 
  average_salary = 850 ∧ 
  technician_count = 7 ∧ 
  technician_salary = 1000 ∧ 
  rest_salary = 780 →
  ∃ (total_workers : ℕ), total_workers = 22 ∧
    (technician_count : ℚ) * technician_salary + 
    (total_workers - technician_count : ℚ) * rest_salary = 
    (total_workers : ℚ) * average_salary :=
by
  sorry


end workshop_workers_l3279_327992


namespace properties_of_f_l3279_327921

noncomputable def f (x : ℝ) : ℝ := (3/2) ^ x

theorem properties_of_f (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  (f (x₁ + x₂) = f x₁ * f x₂) ∧
  ((f x₁ - f x₂) / (x₁ - x₂) > 0) ∧
  (1 < x₁ → x₁ < x₂ → f x₁ / (x₁ - 1) > f x₂ / (x₂ - 1)) :=
by sorry

end properties_of_f_l3279_327921


namespace similar_triangles_leg_sum_l3279_327906

theorem similar_triangles_leg_sum (a b c A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 → A > 0 → B > 0 → C > 0 →
  (1/2) * a * b = 6 →
  (1/2) * A * B = 150 →
  c = 5 →
  a^2 + b^2 = c^2 →
  A^2 + B^2 = C^2 →
  (a/A)^2 = (b/B)^2 →
  (a/A)^2 = (c/C)^2 →
  A + B = 35 :=
by sorry

end similar_triangles_leg_sum_l3279_327906


namespace complex_fraction_simplification_l3279_327916

theorem complex_fraction_simplification :
  1 + 3 / (2 + 5/6) = 35/17 := by sorry

end complex_fraction_simplification_l3279_327916
