import Mathlib

namespace slope_does_not_exist_for_vertical_line_l3128_312811

/-- A line is vertical if its equation can be written in the form x = constant -/
def IsVerticalLine (a b : ℝ) : Prop := a ≠ 0 ∧ ∀ x y : ℝ, a * x + b = 0 → x = -b / a

/-- The slope of a line does not exist if the line is vertical -/
def SlopeDoesNotExist (a b : ℝ) : Prop := IsVerticalLine a b

theorem slope_does_not_exist_for_vertical_line (a b : ℝ) :
  a * x + b = 0 → a ≠ 0 → SlopeDoesNotExist a b := by sorry

end slope_does_not_exist_for_vertical_line_l3128_312811


namespace zebra_stripes_l3128_312845

theorem zebra_stripes (w n b : ℕ) : 
  w + n = b + 1 →  -- Total black stripes = white stripes + 1
  b = w + 7 →      -- White stripes = wide black stripes + 7
  n = 8 :=         -- Number of narrow black stripes is 8
by sorry

end zebra_stripes_l3128_312845


namespace binomial_18_10_l3128_312812

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 45760 := by
  sorry

end binomial_18_10_l3128_312812


namespace max_value_of_f_on_interval_l3128_312880

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ f x = 5 ∧ ∀ y ∈ Set.Icc 0 3, f y ≤ f x :=
by sorry

end max_value_of_f_on_interval_l3128_312880


namespace no_valid_numbers_l3128_312869

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ (a x : ℕ), n = 1000 * a + x ∧ 100 ≤ x ∧ x < 1000 ∧ 8 * x = n

theorem no_valid_numbers : ¬∃ (n : ℕ), is_valid_number n := by
  sorry

end no_valid_numbers_l3128_312869


namespace choose_four_from_fifteen_l3128_312827

theorem choose_four_from_fifteen (n : ℕ) (k : ℕ) : n = 15 ∧ k = 4 → Nat.choose n k = 1365 := by
  sorry

end choose_four_from_fifteen_l3128_312827


namespace budget_research_development_l3128_312843

theorem budget_research_development (transportation utilities equipment supplies salaries research_development : ℝ) : 
  transportation = 20 →
  utilities = 5 →
  equipment = 4 →
  supplies = 2 →
  salaries = 216 / 360 * 100 →
  transportation + utilities + equipment + supplies + salaries + research_development = 100 →
  research_development = 9 := by
sorry

end budget_research_development_l3128_312843


namespace exist_distant_points_on_polyhedron_l3128_312895

/-- A sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- A polyhedron with a given number of faces -/
structure Polyhedron where
  faces : ℕ

/-- A polyhedron is circumscribed around a sphere -/
def is_circumscribed (p : Polyhedron) (s : Sphere) : Prop :=
  sorry

/-- The distance between two points on the surface of a polyhedron -/
def surface_distance (p : Polyhedron) (point1 point2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- The main theorem -/
theorem exist_distant_points_on_polyhedron (s : Sphere) (p : Polyhedron) 
  (h_radius : s.radius = 10)
  (h_faces : p.faces = 19)
  (h_circumscribed : is_circumscribed p s) :
  ∃ (point1 point2 : ℝ × ℝ × ℝ), surface_distance p point1 point2 > 21 :=
sorry

end exist_distant_points_on_polyhedron_l3128_312895


namespace sum_of_prime_factors_27000001_l3128_312867

theorem sum_of_prime_factors_27000001 :
  ∃ (p₁ p₂ p₃ p₄ : Nat),
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧
    p₁ * p₂ * p₃ * p₄ = 27000001 ∧
    p₁ + p₂ + p₃ + p₄ = 652 :=
by
  sorry

#check sum_of_prime_factors_27000001

end sum_of_prime_factors_27000001_l3128_312867


namespace average_seeds_per_grape_l3128_312841

/-- Theorem: Average number of seeds per grape -/
theorem average_seeds_per_grape 
  (total_seeds : ℕ) 
  (apple_seeds : ℕ) 
  (pear_seeds : ℕ) 
  (apples : ℕ) 
  (pears : ℕ) 
  (grapes : ℕ) 
  (seeds_needed : ℕ) 
  (h1 : total_seeds = 60)
  (h2 : apple_seeds = 6)
  (h3 : pear_seeds = 2)
  (h4 : apples = 4)
  (h5 : pears = 3)
  (h6 : grapes = 9)
  (h7 : seeds_needed = 3)
  : (total_seeds - (apples * apple_seeds + pears * pear_seeds) - seeds_needed) / grapes = 3 :=
by sorry

end average_seeds_per_grape_l3128_312841


namespace fuel_cost_savings_l3128_312847

theorem fuel_cost_savings (old_efficiency : ℝ) (old_fuel_cost : ℝ) 
  (trip_distance : ℝ) (efficiency_improvement : ℝ) (fuel_cost_increase : ℝ) :
  old_efficiency > 0 → old_fuel_cost > 0 → trip_distance > 0 →
  efficiency_improvement = 0.6 → fuel_cost_increase = 0.25 → trip_distance = 300 →
  let new_efficiency := old_efficiency * (1 + efficiency_improvement)
  let new_fuel_cost := old_fuel_cost * (1 + fuel_cost_increase)
  let old_trip_cost := (trip_distance / old_efficiency) * old_fuel_cost
  let new_trip_cost := (trip_distance / new_efficiency) * new_fuel_cost
  let savings_percentage := (old_trip_cost - new_trip_cost) / old_trip_cost * 100
  savings_percentage = 21.875 := by
sorry

end fuel_cost_savings_l3128_312847


namespace guard_arrangement_exists_l3128_312833

/-- Represents a guard with a position and direction of sight -/
structure Guard where
  position : ℝ × ℝ
  direction : ℝ × ℝ

/-- Represents the arrangement of guards around a point object -/
structure GuardArrangement where
  guards : List Guard
  object : ℝ × ℝ
  visibility_range : ℝ

/-- Predicate to check if a point is inside or on the boundary of a convex hull -/
def is_inside_or_on_convex_hull (point : ℝ × ℝ) (hull : List (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if a list of points forms a convex hull -/
def is_convex_hull (points : List (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if it's impossible to approach any point unnoticed -/
def is_approach_impossible (arrangement : GuardArrangement) : Prop :=
  sorry

/-- Theorem stating that it's possible to arrange guards to prevent unnoticed approach -/
theorem guard_arrangement_exists : ∃ (arrangement : GuardArrangement),
  arrangement.visibility_range = 100 ∧
  arrangement.guards.length ≥ 6 ∧
  is_convex_hull (arrangement.guards.map Guard.position) ∧
  is_inside_or_on_convex_hull arrangement.object (arrangement.guards.map Guard.position) ∧
  is_approach_impossible arrangement :=
by
  sorry

end guard_arrangement_exists_l3128_312833


namespace parallelogram_area_25_15_l3128_312859

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 25 cm and height 15 cm is 375 cm² -/
theorem parallelogram_area_25_15 :
  parallelogram_area 25 15 = 375 := by sorry

end parallelogram_area_25_15_l3128_312859


namespace sequence_properties_l3128_312814

/-- Given a sequence {a_n}, where S_n is the sum of the first n terms,
    a_1 = a (a ≠ 4), and a_{n+1} = 2S_n + 4^n for n ∈ ℕ* -/
def Sequence (a : ℝ) (a_n : ℕ+ → ℝ) (S_n : ℕ+ → ℝ) : Prop :=
  a ≠ 4 ∧
  a_n 1 = a ∧
  ∀ n : ℕ+, a_n (n + 1) = 2 * S_n n + 4^(n : ℕ)

/-- Definition of b_n -/
def b_n (S_n : ℕ+ → ℝ) : ℕ+ → ℝ :=
  λ n => S_n n - 4^(n : ℕ)

theorem sequence_properties {a : ℝ} {a_n : ℕ+ → ℝ} {S_n : ℕ+ → ℝ}
    (h : Sequence a a_n S_n) :
    /- 1. {b_n} forms a geometric progression with common ratio 3 -/
    (∀ n : ℕ+, b_n S_n (n + 1) = 3 * b_n S_n n) ∧
    /- 2. General formula for {a_n} -/
    (∀ n : ℕ+, n = 1 → a_n n = a) ∧
    (∀ n : ℕ+, n ≥ 2 → a_n n = 3 * 4^(n - 1 : ℕ) + 2 * (a - 4) * 3^(n - 2 : ℕ)) ∧
    /- 3. Range of a that satisfies a_{n+1} ≥ a_n for n ∈ ℕ* -/
    (∀ n : ℕ+, a_n (n + 1) ≥ a_n n ↔ a ∈ Set.Icc (-4 : ℝ) 4 ∪ Set.Ioi 4) :=
by sorry

end sequence_properties_l3128_312814


namespace seventh_term_of_geometric_sequence_l3128_312822

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem seventh_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_fourth : a 4 = 16)
  (h_tenth : a 10 = 2) :
  a 7 = 2 := by
  sorry

end seventh_term_of_geometric_sequence_l3128_312822


namespace smallest_six_digit_divisible_l3128_312878

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

theorem smallest_six_digit_divisible : 
  ∀ n : ℕ, 
    100000 ≤ n → 
    n < 1000000 → 
    (is_divisible_by n 25 ∧ 
     is_divisible_by n 35 ∧ 
     is_divisible_by n 45 ∧ 
     is_divisible_by n 15) → 
    n ≥ 100800 :=
sorry

end smallest_six_digit_divisible_l3128_312878


namespace factors_of_sixty_l3128_312865

theorem factors_of_sixty : Nat.card (Nat.divisors 60) = 12 := by
  sorry

end factors_of_sixty_l3128_312865


namespace algebraic_expression_equality_l3128_312874

theorem algebraic_expression_equality (x : ℝ) (h : x^2 - 4*x + 1 = 3) :
  3*x^2 - 12*x - 1 = 5 := by
  sorry

end algebraic_expression_equality_l3128_312874


namespace sum_equals_5186_l3128_312882

theorem sum_equals_5186 : 5 * 399 + 4 * 399 + 3 * 399 + 398 = 5186 := by
  sorry

end sum_equals_5186_l3128_312882


namespace race_distance_l3128_312831

/-- The race problem -/
theorem race_distance (time_A time_B : ℝ) (lead : ℝ) (distance : ℝ) : 
  time_A = 36 →
  time_B = 45 →
  lead = 20 →
  (distance / time_A) * time_B = distance + lead →
  distance = 80 := by
sorry

end race_distance_l3128_312831


namespace triangle_properties_l3128_312854

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * Real.sin t.B + t.b * Real.cos t.A = 0) 
  (h2 : 0 < t.A ∧ t.A < Real.pi) 
  (h3 : 0 < t.B ∧ t.B < Real.pi) 
  (h4 : 0 < t.C ∧ t.C < Real.pi) 
  (h5 : t.A + t.B + t.C = Real.pi) :
  t.A = 3 * Real.pi / 4 ∧ 
  (t.a = 2 * Real.sqrt 5 → t.b = 2 → 
    1/2 * t.b * t.c * Real.sin t.A = 2) := by
  sorry

end triangle_properties_l3128_312854


namespace tanα_eq_2_implies_reciprocal_sin2α_eq_5_4_l3128_312851

theorem tanα_eq_2_implies_reciprocal_sin2α_eq_5_4 (α : ℝ) (h : Real.tan α = 2) :
  1 / Real.sin (2 * α) = 5 / 4 := by
sorry

end tanα_eq_2_implies_reciprocal_sin2α_eq_5_4_l3128_312851


namespace fourteenth_root_of_unity_l3128_312807

theorem fourteenth_root_of_unity : 
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 13 ∧ 
  (Complex.tan (Real.pi / 7) + Complex.I) / (Complex.tan (Real.pi / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * Real.pi * n / 14)) := by
  sorry

end fourteenth_root_of_unity_l3128_312807


namespace S_31_composite_bound_l3128_312876

def S (k : ℕ+) (n : ℕ) : ℕ :=
  (n.digits k.val).sum

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

theorem S_31_composite_bound :
  ∃ (A : Finset ℕ), A.card ≤ 2 ∧
    ∀ p : ℕ, is_prime p → p < 20000 →
      is_composite (S 31 p) → S 31 p ∈ A :=
sorry

end S_31_composite_bound_l3128_312876


namespace arithmetic_expression_evaluation_l3128_312884

theorem arithmetic_expression_evaluation : 2 + 3 * 4^2 - 5 + 6 = 51 := by
  sorry

end arithmetic_expression_evaluation_l3128_312884


namespace polynomial_evaluation_l3128_312832

theorem polynomial_evaluation :
  ∃ x : ℝ, x > 0 ∧ x^2 - 2*x - 15 = 0 ∧ x^3 - 2*x^2 - 8*x + 16 = 51 := by
  sorry

end polynomial_evaluation_l3128_312832


namespace min_value_of_expression_l3128_312801

theorem min_value_of_expression (x : ℝ) :
  ∃ (min : ℝ), min = -4356 ∧ ∀ y : ℝ, (14 - y) * (8 - y) * (14 + y) * (8 + y) ≥ min :=
by sorry

end min_value_of_expression_l3128_312801


namespace exam_marks_l3128_312819

theorem exam_marks (full_marks : ℕ) (A B C D : ℕ) : 
  full_marks = 500 →
  A = B - B / 10 →
  B = C + C / 4 →
  C = D - D / 5 →
  D = full_marks * 4 / 5 →
  A = 360 := by sorry

end exam_marks_l3128_312819


namespace log_equality_implies_golden_ratio_l3128_312826

theorem log_equality_implies_golden_ratio (a b : ℝ) :
  a > 0 ∧ b > 0 →
  Real.log a / Real.log 8 = Real.log b / Real.log 18 ∧
  Real.log a / Real.log 8 = Real.log (a + b) / Real.log 32 →
  b / a = (1 + Real.sqrt 5) / 2 := by
sorry

end log_equality_implies_golden_ratio_l3128_312826


namespace max_candies_equals_complete_graph_edges_l3128_312821

/-- The number of ones initially on the board -/
def initial_ones : Nat := 30

/-- The number of minutes the process continues -/
def total_minutes : Nat := 30

/-- Represents the board state at any given time -/
structure Board where
  numbers : List Nat

/-- Represents a single operation of erasing two numbers and writing their sum -/
def erase_and_sum (b : Board) (i j : Nat) : Board := sorry

/-- The number of candies eaten in a single operation -/
def candies_eaten (b : Board) (i j : Nat) : Nat := sorry

/-- The maximum number of candies that can be eaten -/
def max_candies : Nat := (initial_ones * (initial_ones - 1)) / 2

/-- Theorem stating that the maximum number of candies eaten is equal to
    the number of edges in a complete graph with 'initial_ones' vertices -/
theorem max_candies_equals_complete_graph_edges :
  max_candies = (initial_ones * (initial_ones - 1)) / 2 := by sorry

end max_candies_equals_complete_graph_edges_l3128_312821


namespace company_shares_l3128_312897

theorem company_shares (K : ℝ) (P V S I : ℝ) 
  (h1 : P + V + S + I = K)
  (h2 : K + P = 1.25 * K)
  (h3 : K + V = 1.35 * K)
  (h4 : K + 2 * S = 1.4 * K)
  (h5 : I > 0) :
  ∃ x : ℝ, x > 2.5 ∧ x * I > 0.5 * K := by
  sorry

end company_shares_l3128_312897


namespace tetrahedron_cut_vertices_l3128_312820

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  vertices : Finset (Fin 4)

/-- The result of cutting off a vertex from a polyhedron -/
def cutVertex (p : RegularTetrahedron) (v : Fin 4) : ℕ := 3

/-- The number of vertices in the shape resulting from cutting off all vertices of a regular tetrahedron -/
def verticesAfterCutting (t : RegularTetrahedron) : ℕ :=
  t.vertices.sum (λ v => cutVertex t v)

/-- Theorem: Cutting off all vertices of a regular tetrahedron results in a shape with 12 vertices -/
theorem tetrahedron_cut_vertices (t : RegularTetrahedron) :
  verticesAfterCutting t = 12 := by sorry

end tetrahedron_cut_vertices_l3128_312820


namespace georgia_black_buttons_l3128_312872

theorem georgia_black_buttons
  (yellow_buttons : Nat)
  (green_buttons : Nat)
  (buttons_given : Nat)
  (buttons_left : Nat)
  (h1 : yellow_buttons = 4)
  (h2 : green_buttons = 3)
  (h3 : buttons_given = 4)
  (h4 : buttons_left = 5) :
  ∃ (black_buttons : Nat), black_buttons = 2 ∧
    yellow_buttons + black_buttons + green_buttons = buttons_left + buttons_given :=
by sorry

end georgia_black_buttons_l3128_312872


namespace factorization_3x2_minus_27y2_l3128_312853

theorem factorization_3x2_minus_27y2 (x y : ℝ) : 3 * x^2 - 27 * y^2 = 3 * (x + 3*y) * (x - 3*y) := by
  sorry

end factorization_3x2_minus_27y2_l3128_312853


namespace no_x_axis_intersection_implies_m_bound_l3128_312866

/-- A quadratic function of the form f(x) = x^2 - x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - x + m

/-- The discriminant of the quadratic function f(x) = x^2 - x + m -/
def discriminant (m : ℝ) : ℝ := 1 - 4*m

theorem no_x_axis_intersection_implies_m_bound (m : ℝ) :
  (∀ x : ℝ, f m x ≠ 0) → m > (1/4 : ℝ) := by
  sorry

end no_x_axis_intersection_implies_m_bound_l3128_312866


namespace sum_of_squares_l3128_312844

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_cubes_eq_sum_fifth : a^3 + b^3 + c^3 = a^5 + b^5 + c^5) :
  a^2 + b^2 + c^2 = 6/5 := by
  sorry

end sum_of_squares_l3128_312844


namespace sum_of_squares_of_roots_l3128_312842

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (8 * x₁^2 + 12 * x₁ - 14 = 0) → 
  (8 * x₂^2 + 12 * x₂ - 14 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 23/4) := by
  sorry

end sum_of_squares_of_roots_l3128_312842


namespace twenty_is_least_pieces_l3128_312813

/-- The number of expected guests -/
def expected_guests : Set Nat := {10, 11}

/-- A function to check if a number of pieces can be equally divided among a given number of guests -/
def can_divide_equally (pieces : Nat) (guests : Nat) : Prop :=
  ∃ (share : Nat), pieces = guests * share

/-- The proposition that a given number of pieces is the least number that can be equally divided among either 10 or 11 guests -/
def is_least_pieces (pieces : Nat) : Prop :=
  (∀ g ∈ expected_guests, can_divide_equally pieces g) ∧
  (∀ p < pieces, ∃ g ∈ expected_guests, ¬can_divide_equally p g)

/-- Theorem stating that 20 is the least number of pieces that can be equally divided among either 10 or 11 guests -/
theorem twenty_is_least_pieces : is_least_pieces 20 := by
  sorry

end twenty_is_least_pieces_l3128_312813


namespace expand_product_l3128_312863

theorem expand_product (x : ℝ) : (x + 3) * (x - 2) * (x + 4) = x^3 + 5*x^2 - 2*x - 24 := by
  sorry

end expand_product_l3128_312863


namespace complex_addition_simplification_l3128_312896

theorem complex_addition_simplification :
  (-5 : ℂ) + 3*I + (2 : ℂ) - 7*I = -3 - 4*I :=
by sorry

end complex_addition_simplification_l3128_312896


namespace petya_vasya_divisibility_l3128_312828

theorem petya_vasya_divisibility (n m : ℕ) (h : ∀ k ∈ Finset.range 100, ∃ j ∈ Finset.range 99, (m - j) ∣ (n + k)) :
  m > n^3 / 10000000 := by
  sorry

end petya_vasya_divisibility_l3128_312828


namespace double_papers_double_time_l3128_312846

/-- Represents the time taken to check exam papers under different conditions -/
def exam_check_time (men : ℕ) (days : ℕ) (hours_per_day : ℕ) (papers : ℕ) : ℕ :=
  men * days * hours_per_day

/-- Theorem stating the relationship between different exam checking scenarios -/
theorem double_papers_double_time (men₁ days₁ hours₁ men₂ days₂ papers₁ : ℕ) :
  exam_check_time men₁ days₁ hours₁ papers₁ = 160 →
  men₁ = 4 →
  days₁ = 8 →
  hours₁ = 5 →
  men₂ = 2 →
  days₂ = 20 →
  exam_check_time men₂ days₂ 8 (2 * papers₁) = 320 := by
  sorry

#check double_papers_double_time

end double_papers_double_time_l3128_312846


namespace complement_of_M_in_U_l3128_312816

def U : Set Int := {1, -2, 3, -4, 5, -6}
def M : Set Int := {1, -2, 3, -4}

theorem complement_of_M_in_U : Mᶜ = {5, -6} := by sorry

end complement_of_M_in_U_l3128_312816


namespace sqrt_equation_solution_l3128_312818

theorem sqrt_equation_solution :
  ∀ a b : ℕ+,
  a < b →
  Real.sqrt (3 + Real.sqrt (45 + 20 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b →
  a = 3 ∧ b = 5 := by
  sorry

end sqrt_equation_solution_l3128_312818


namespace intersection_condition_l3128_312817

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 1 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

-- State the theorem
theorem intersection_condition (a : ℝ) : A ∩ B a = B a → a = 1 ∨ a = -1 := by
  sorry

end intersection_condition_l3128_312817


namespace square_difference_l3128_312898

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end square_difference_l3128_312898


namespace johns_gym_time_l3128_312887

/-- Represents the number of times John goes to the gym per week -/
def gym_visits_per_week : ℕ := 3

/-- Represents the number of hours John spends weightlifting each gym visit -/
def weightlifting_hours : ℚ := 1

/-- Represents the fraction of weightlifting time spent on warming up and cardio -/
def warmup_cardio_fraction : ℚ := 1 / 3

/-- Calculates the total hours John spends at the gym per week -/
def total_gym_hours : ℚ :=
  gym_visits_per_week * (weightlifting_hours + warmup_cardio_fraction * weightlifting_hours)

theorem johns_gym_time : total_gym_hours = 4 := by
  sorry

end johns_gym_time_l3128_312887


namespace percentage_good_fruits_l3128_312886

/-- Calculates the percentage of fruits in good condition given the number of oranges and bananas and their respective rotten percentages. -/
theorem percentage_good_fruits (oranges bananas : ℕ) (rotten_oranges_percent rotten_bananas_percent : ℚ) :
  oranges = 600 →
  bananas = 400 →
  rotten_oranges_percent = 15 / 100 →
  rotten_bananas_percent = 3 / 100 →
  (((oranges + bananas : ℚ) - (oranges * rotten_oranges_percent + bananas * rotten_bananas_percent)) / (oranges + bananas) * 100 : ℚ) = 89.8 := by
  sorry

end percentage_good_fruits_l3128_312886


namespace student_number_problem_l3128_312861

theorem student_number_problem (x : ℝ) : 4 * x - 142 = 110 → x = 63 := by
  sorry

end student_number_problem_l3128_312861


namespace isabella_hair_growth_l3128_312802

def monthly_growth : List Float := [0.5, 1, 0.75, 1.25, 1, 0.5, 1.5, 1, 0.25, 1.5, 1.25, 0.75]

theorem isabella_hair_growth :
  monthly_growth.sum = 11.25 := by
  sorry

end isabella_hair_growth_l3128_312802


namespace tangent_line_to_circle_C_internal_tangency_of_circles_l3128_312839

-- Define the circle C
def circle_C (m : ℝ) (x y : ℝ) : Prop :=
  (x - m)^2 + (y - 2*m)^2 = m^2

-- Define the circle E
def circle_E (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 16

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop :=
  y = (3/4) * x ∨ x = 0

-- Theorem for part (I)
theorem tangent_line_to_circle_C :
  ∀ x y : ℝ, circle_C 2 x y → tangent_line x y → (x = 0 ∧ y = 0) ∨ (x ≠ 0 ∧ y ≠ 0) :=
sorry

-- Theorem for part (II)
theorem internal_tangency_of_circles :
  ∃ x y : ℝ, circle_C ((Real.sqrt 29 - 1) / 4) x y ∧ circle_E x y :=
sorry

end tangent_line_to_circle_C_internal_tangency_of_circles_l3128_312839


namespace willies_stickers_l3128_312829

theorem willies_stickers (initial : ℕ) (given : ℕ) (remaining : ℕ) : 
  given = 7 → remaining = 29 → initial = remaining + given :=
by
  sorry

end willies_stickers_l3128_312829


namespace midpoint_coordinate_sum_l3128_312805

/-- The sum of the coordinates of the midpoint of a segment with endpoints (10, 3) and (4, -3) is 7. -/
theorem midpoint_coordinate_sum : 
  let p₁ : ℝ × ℝ := (10, 3)
  let p₂ : ℝ × ℝ := (4, -3)
  let midpoint := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  (midpoint.1 + midpoint.2 : ℝ) = 7 := by
sorry

end midpoint_coordinate_sum_l3128_312805


namespace differentials_of_z_l3128_312877

noncomputable section

variables (x y : ℝ) (dx dy : ℝ)

def z : ℝ := x^5 * y^3

def dz : ℝ := 5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy

def d2z : ℝ := 20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2

def d3z : ℝ := 60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3

theorem differentials_of_z :
  (dz x y dx dy = 5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy) ∧
  (d2z x y dx dy = 20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2) ∧
  (d3z x y dx dy = 60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3) :=
by sorry

end differentials_of_z_l3128_312877


namespace marks_remaining_money_l3128_312862

def initial_amount : ℕ := 85
def books_seven_dollars : ℕ := 3
def books_five_dollars : ℕ := 4
def books_nine_dollars : ℕ := 2

def cost_seven_dollars : ℕ := 7
def cost_five_dollars : ℕ := 5
def cost_nine_dollars : ℕ := 9

theorem marks_remaining_money :
  initial_amount - 
  (books_seven_dollars * cost_seven_dollars + 
   books_five_dollars * cost_five_dollars + 
   books_nine_dollars * cost_nine_dollars) = 26 := by
  sorry

end marks_remaining_money_l3128_312862


namespace cube_difference_factorization_l3128_312879

theorem cube_difference_factorization (a b : ℝ) :
  a^3 - 8*b^3 = (a - 2*b) * (a^2 + 2*a*b + 4*b^2) := by
  sorry

end cube_difference_factorization_l3128_312879


namespace det_3_4_1_2_l3128_312824

-- Define the determinant function for a 2x2 matrix
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Theorem statement
theorem det_3_4_1_2 : det2x2 3 4 1 2 = 2 := by
  sorry

end det_3_4_1_2_l3128_312824


namespace derivative_e_cubed_l3128_312837

-- e is the base of the natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- Statement: The derivative of e^3 is e^3
theorem derivative_e_cubed : 
  deriv (fun x : ℝ => e^3) = fun x : ℝ => e^3 :=
sorry

end derivative_e_cubed_l3128_312837


namespace smallest_enclosing_sphere_radius_l3128_312864

/-- The radius of the smallest sphere centered at the origin that contains
    ten spheres of radius 2 positioned at the corners of a cube with side length 4 -/
theorem smallest_enclosing_sphere_radius (r : ℝ) (s : ℝ) : r = 2 ∧ s = 4 →
  (2 * Real.sqrt 3 + 2 : ℝ) = 
    (s * Real.sqrt 3 / 2 + r : ℝ) := by sorry

end smallest_enclosing_sphere_radius_l3128_312864


namespace milk_tea_sales_l3128_312849

-- Define the relationship between cups of milk tea and total sales price
def sales_price (x : ℕ) : ℕ := 10 * x + 2

-- Theorem stating the conditions and the result to be proved
theorem milk_tea_sales :
  (sales_price 1 = 12) →
  (sales_price 2 = 22) →
  (∃ x : ℕ, sales_price x = 822) →
  (∃ x : ℕ, sales_price x = 822 ∧ x = 82) :=
by sorry

end milk_tea_sales_l3128_312849


namespace rectangle_side_ratio_l3128_312891

/-- Represents the configuration of rectangles around a square -/
structure RectangleSquareConfig where
  inner_square_side : ℝ
  rectangle_short_side : ℝ
  rectangle_long_side : ℝ

/-- The theorem stating the ratio of rectangle sides given the square configuration -/
theorem rectangle_side_ratio
  (config : RectangleSquareConfig)
  (h1 : config.inner_square_side + 2 * config.rectangle_short_side = 2 * config.inner_square_side)
  (h2 : config.rectangle_long_side + config.inner_square_side = 2 * config.inner_square_side)
  (h3 : (2 * config.inner_square_side) ^ 2 = 4 * config.inner_square_side ^ 2) :
  config.rectangle_long_side / config.rectangle_short_side = 2 := by
  sorry

end rectangle_side_ratio_l3128_312891


namespace shaded_region_area_l3128_312806

/-- Given a shaded region consisting of congruent squares, proves that the total area is 40 cm² --/
theorem shaded_region_area (n : ℕ) (d : ℝ) (A : ℝ) :
  n = 20 →  -- Total number of congruent squares
  d = 8 →   -- Diagonal of the square formed by 16 smaller squares
  A = d^2 / 2 →  -- Area of the square formed by 16 smaller squares
  A / 16 * n = 40 :=
by sorry

end shaded_region_area_l3128_312806


namespace quadratic_two_distinct_roots_l3128_312830

theorem quadratic_two_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ + 0 = 0 ∧ x₂^2 - 3*x₂ + 0 = 0 := by
  sorry

end quadratic_two_distinct_roots_l3128_312830


namespace smallest_positive_linear_combination_l3128_312889

theorem smallest_positive_linear_combination : 
  (∃ (k : ℕ+), k = Nat.gcd 3003 60606 ∧ 
   (∀ (x : ℕ+), (∃ (m n : ℤ), x.val = 3003 * m + 60606 * n) → k ≤ x) ∧
   (∃ (m n : ℤ), k.val = 3003 * m + 60606 * n)) ∧
  Nat.gcd 3003 60606 = 273 := by
sorry

end smallest_positive_linear_combination_l3128_312889


namespace perfect_squares_among_options_l3128_312848

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def option_a : ℕ := 3^3 * 4^4 * 5^5
def option_b : ℕ := 3^4 * 4^5 * 5^6
def option_c : ℕ := 3^6 * 4^4 * 5^6
def option_d : ℕ := 3^5 * 4^6 * 5^5
def option_e : ℕ := 3^6 * 4^6 * 5^4

theorem perfect_squares_among_options :
  (¬ is_perfect_square option_a) ∧
  (is_perfect_square option_b) ∧
  (is_perfect_square option_c) ∧
  (¬ is_perfect_square option_d) ∧
  (is_perfect_square option_e) := by
  sorry

end perfect_squares_among_options_l3128_312848


namespace unique_solution_for_prime_equation_l3128_312800

theorem unique_solution_for_prime_equation (p : ℕ) (hp : Prime p) :
  ∀ x y : ℕ+, p * (x - y) = x * y → x = p^2 - p ∧ y = p + 1 := by
  sorry

end unique_solution_for_prime_equation_l3128_312800


namespace partner_a_investment_l3128_312858

/-- Represents the investment and profit distribution scenario described in the problem -/
structure BusinessScenario where
  a_investment : ℚ  -- Investment of partner a
  b_investment : ℚ  -- Investment of partner b
  total_profit : ℚ  -- Total profit
  a_total_received : ℚ  -- Total amount received by partner a
  management_fee_percent : ℚ  -- Percentage of profit for management

/-- The main theorem representing the problem -/
theorem partner_a_investment (scenario : BusinessScenario) : 
  scenario.b_investment = 2500 ∧ 
  scenario.total_profit = 9600 ∧
  scenario.a_total_received = 6000 ∧
  scenario.management_fee_percent = 1/10 →
  scenario.a_investment = 3500 := by
sorry


end partner_a_investment_l3128_312858


namespace sum_of_cumulative_sums_geometric_sequence_l3128_312804

/-- The sum of cumulative sums of a geometric sequence -/
theorem sum_of_cumulative_sums_geometric_sequence (a₁ q : ℝ) (h : |q| < 1) :
  ∃ (S : ℕ → ℝ), (∀ n, S n = a₁ * (1 - q^n) / (1 - q)) ∧
  (∑' n, S n) = a₁ / (1 - q)^2 := by
sorry

end sum_of_cumulative_sums_geometric_sequence_l3128_312804


namespace contractor_engagement_days_l3128_312850

/-- Represents the daily wage in rupees --/
def daily_wage : ℚ := 25

/-- Represents the daily fine in rupees --/
def daily_fine : ℚ := 7.5

/-- Represents the total amount received in rupees --/
def total_amount : ℚ := 685

/-- Represents the number of days absent --/
def days_absent : ℕ := 2

/-- Proves that the contractor was engaged for 28 days --/
theorem contractor_engagement_days : 
  ∃ (days_worked : ℕ), 
    (daily_wage * days_worked - daily_fine * days_absent = total_amount) ∧ 
    (days_worked + days_absent = 28) := by
  sorry

end contractor_engagement_days_l3128_312850


namespace smallest_three_digit_divisible_by_4_and_5_l3128_312835

theorem smallest_three_digit_divisible_by_4_and_5 :
  ∃ n : ℕ, n = 100 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 4 ∣ m ∧ 5 ∣ m → n ≤ m) ∧
  4 ∣ n ∧ 5 ∣ n :=
by sorry

end smallest_three_digit_divisible_by_4_and_5_l3128_312835


namespace rhombus_other_diagonal_l3128_312840

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ  -- Length of the first diagonal
  d2 : ℝ  -- Length of the second diagonal
  area : ℝ -- Area of the rhombus

/-- Theorem: In a rhombus with one diagonal of 25 m and an area of 625 m², the other diagonal is 50 m -/
theorem rhombus_other_diagonal (r : Rhombus) (h1 : r.d1 = 25) (h2 : r.area = 625) : r.d2 = 50 := by
  sorry

end rhombus_other_diagonal_l3128_312840


namespace complement_intersection_when_a_is_3_range_of_a_when_union_equals_B_range_of_a_when_intersection_is_empty_l3128_312838

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 * a + 1}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}

-- Theorem 1: When a=3, ℂᴿ(A∩B) = {x | x < 2 or x > 4}
theorem complement_intersection_when_a_is_3 :
  (Set.univ \ (A 3 ∩ B)) = {x | x < 2 ∨ x > 4} := by sorry

-- Theorem 2: When A∪B=B, the range of a is (-∞,-2)∪[-1,3/2]
theorem range_of_a_when_union_equals_B :
  (∀ a, A a ∪ B = B) ↔ (∀ a, a < -2 ∨ (-1 ≤ a ∧ a ≤ 3/2)) := by sorry

-- Theorem 3: When A∩B=∅, the range of a is (-∞,-3/2)∪(5,+∞)
theorem range_of_a_when_intersection_is_empty :
  (∀ a, A a ∩ B = ∅) ↔ (∀ a, a < -3/2 ∨ a > 5) := by sorry

end complement_intersection_when_a_is_3_range_of_a_when_union_equals_B_range_of_a_when_intersection_is_empty_l3128_312838


namespace quadratic_solution_property_l3128_312856

theorem quadratic_solution_property (a b : ℝ) : 
  (3 * a^2 - 9 * a + 21 = 0) ∧ 
  (3 * b^2 - 9 * b + 21 = 0) →
  (3 * a - 4) * (6 * b - 8) = 50 := by
sorry

end quadratic_solution_property_l3128_312856


namespace scientific_notation_correct_l3128_312885

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 5500

/-- The scientific notation representation of the number -/
def scientificForm : ScientificNotation := {
  coefficient := 5.5
  exponent := 3
  h_coefficient := by sorry
}

/-- Theorem stating that the scientific notation is correct -/
theorem scientific_notation_correct : 
  (scientificForm.coefficient * (10 : ℝ) ^ scientificForm.exponent) = number := by sorry

end scientific_notation_correct_l3128_312885


namespace balloon_difference_l3128_312892

theorem balloon_difference (your_balloons friend_balloons : ℕ) 
  (h1 : your_balloons = 7) 
  (h2 : friend_balloons = 5) : 
  your_balloons - friend_balloons = 2 := by
sorry

end balloon_difference_l3128_312892


namespace triangle_inequality_from_sum_product_l3128_312888

theorem triangle_inequality_from_sum_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
  c < a + b ∧ a < b + c ∧ b < c + a :=
by sorry

end triangle_inequality_from_sum_product_l3128_312888


namespace temporary_wall_area_l3128_312834

theorem temporary_wall_area : 
  let width : Real := 5.4
  let length : Real := 2.5
  width * length = 13.5 := by
sorry

end temporary_wall_area_l3128_312834


namespace expression_evaluation_l3128_312890

theorem expression_evaluation (x y : ℚ) (hx : x = 1/3) (hy : y = -2) :
  (x * (x + y) - (x - y)^2) / y = 3 := by
  sorry

end expression_evaluation_l3128_312890


namespace largest_quotient_and_smallest_product_l3128_312899

def S : Set ℤ := {-25, -4, -1, 3, 5, 9}

theorem largest_quotient_and_smallest_product (a b : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hb_nonzero : b ≠ 0) :
  (∃ (x y : ℤ) (hx : x ∈ S) (hy : y ∈ S) (hy_nonzero : y ≠ 0), (a / b : ℚ) ≤ (x / y : ℚ)) ∧
  (∃ (x y : ℤ) (hx : x ∈ S) (hy : y ∈ S), a * b ≥ x * y) ∧
  (∃ (x y : ℤ) (hx : x ∈ S) (hy : y ∈ S) (hy_nonzero : y ≠ 0), (x / y : ℚ) = 3) ∧
  (∃ (x y : ℤ) (hx : x ∈ S) (hy : y ∈ S), x * y = -225) := by
  sorry

end largest_quotient_and_smallest_product_l3128_312899


namespace equation_solutions_l3128_312809

theorem equation_solutions (k : ℤ) (x₁ x₂ x₃ x₄ y₁ : ℤ) :
  (y₁^2 - k = x₁^3) ∧
  ((y₁ - 1)^2 - k = x₂^3) ∧
  ((y₁ - 2)^2 - k = x₃^3) ∧
  ((y₁ - 3)^2 - k = x₄^3) →
  k ≡ 17 [ZMOD 63] :=
by sorry

end equation_solutions_l3128_312809


namespace fixed_point_of_exponential_translation_l3128_312836

theorem fixed_point_of_exponential_translation (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 + a^(x - 1)
  f 1 = 3 := by sorry

end fixed_point_of_exponential_translation_l3128_312836


namespace no_two_digit_factors_of_1729_l3128_312825

theorem no_two_digit_factors_of_1729 : 
  ¬ ∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ a * b = 1729 := by
  sorry

end no_two_digit_factors_of_1729_l3128_312825


namespace sum_of_digits_62_l3128_312808

def is_two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def reverse_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem sum_of_digits_62 :
  ∀ n : ℕ,
  is_two_digit_number n →
  n = 62 →
  reverse_digits n + 36 = n →
  digit_sum n = 8 := by
sorry

end sum_of_digits_62_l3128_312808


namespace sum_b_formula_l3128_312893

def a (n : ℕ) : ℕ := 2 * n + 1

def b (n : ℕ) : ℚ :=
  (Finset.sum (Finset.range n) (fun i => a (i + 1))) / n

def sum_b (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (fun i => b (i + 1))

theorem sum_b_formula (n : ℕ) : sum_b n = (n * (n + 5) : ℚ) / 2 := by
  sorry

end sum_b_formula_l3128_312893


namespace expand_product_l3128_312883

theorem expand_product (x : ℝ) : (5 * x + 7) * (3 * x^2 + 2 * x + 4) = 15 * x^3 + 31 * x^2 + 34 * x + 28 := by
  sorry

end expand_product_l3128_312883


namespace tenth_term_of_specific_sequence_l3128_312855

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  a₁ : ℚ  -- First term
  d : ℚ   -- Common difference

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a₁ + (n - 1 : ℚ) * seq.d

theorem tenth_term_of_specific_sequence :
  ∃ (seq : ArithmeticSequence),
    seq.nthTerm 1 = 5/6 ∧
    seq.nthTerm 16 = 7/8 ∧
    seq.nthTerm 10 = 103/120 := by
  sorry

end tenth_term_of_specific_sequence_l3128_312855


namespace p_iff_m_gt_2_p_xor_q_iff_m_range_l3128_312875

/-- Proposition p: The equation x^2 + mx + 1 = 0 has two distinct negative real roots -/
def p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ 
  x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- Proposition q: The equation 4x^2 + 4(m-2)x + 1 = 0 has no real roots -/
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem p_iff_m_gt_2 (m : ℝ) : p m ↔ m > 2 :=
sorry

theorem p_xor_q_iff_m_range (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ≥ 3 ∨ (1 < m ∧ m ≤ 2) :=
sorry

end p_iff_m_gt_2_p_xor_q_iff_m_range_l3128_312875


namespace smallest_n_for_equation_l3128_312868

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem smallest_n_for_equation : 
  ∃ (n : ℕ), n > 0 ∧ 2 * n * factorial n + 3 * factorial n = 5040 ∧ 
  ∀ (m : ℕ), m > 0 ∧ m < n → 2 * m * factorial m + 3 * factorial m ≠ 5040 :=
by sorry

end smallest_n_for_equation_l3128_312868


namespace fast_food_cost_correct_l3128_312815

/-- The cost of fast food given the number of servings of each type -/
def fast_food_cost (a b : ℕ) : ℕ := 30 * a + 20 * b

/-- Theorem stating that the cost of fast food is calculated correctly -/
theorem fast_food_cost_correct (a b : ℕ) : 
  fast_food_cost a b = 30 * a + 20 * b := by
  sorry

end fast_food_cost_correct_l3128_312815


namespace quadratic_inequality_solution_set_l3128_312870

theorem quadratic_inequality_solution_set 
  (a b c α β : ℝ) 
  (h1 : α > 0) 
  (h2 : β > α) 
  (h3 : ∀ x, ax^2 + b*x + c > 0 ↔ α < x ∧ x < β) :
  ∀ x, c*x^2 + b*x + a > 0 ↔ 1/β < x ∧ x < 1/α :=
by sorry

end quadratic_inequality_solution_set_l3128_312870


namespace cookie_store_spending_l3128_312823

theorem cookie_store_spending : ∀ (ben david : ℝ),
  (david = 0.6 * ben) →  -- For every dollar Ben spent, David spent 40 cents less
  (ben = david + 16) →   -- Ben paid $16.00 more than David
  (ben + david = 64) :=  -- The total amount they spent together
by
  sorry

end cookie_store_spending_l3128_312823


namespace product_of_four_six_seven_fourteen_l3128_312881

theorem product_of_four_six_seven_fourteen : 4 * 6 * 7 * 14 = 2352 := by
  sorry

end product_of_four_six_seven_fourteen_l3128_312881


namespace peters_savings_l3128_312803

/-- Peter's vacation savings problem -/
theorem peters_savings (total_needed : ℕ) (monthly_savings : ℕ) (months_to_goal : ℕ) 
  (h1 : total_needed = 5000)
  (h2 : monthly_savings = 700)
  (h3 : months_to_goal = 3)
  (h4 : total_needed = monthly_savings * months_to_goal + current_savings) :
  current_savings = 2900 :=
by
  sorry

end peters_savings_l3128_312803


namespace geometric_sum_7_terms_l3128_312857

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_7_terms :
  let a : ℚ := 1/2
  let r : ℚ := -1/2
  let n : ℕ := 7
  geometric_sum a r n = 129/384 := by
sorry

end geometric_sum_7_terms_l3128_312857


namespace square_equation_proof_l3128_312852

theorem square_equation_proof (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ (k : ℚ), (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = k^2 ∧ 
              a^2 + b^2 - c^2 = k^2 ∧
              k = (4*a - 3*b : ℚ) / 5 := by
  sorry

end square_equation_proof_l3128_312852


namespace solution_difference_l3128_312860

def is_solution (x : ℝ) : Prop :=
  (4 * x - 12) / (x^2 + 2*x - 15) = x + 2

theorem solution_difference (p q : ℝ) 
  (hp : is_solution p) 
  (hq : is_solution q) 
  (hdistinct : p ≠ q) 
  (horder : p > q) : 
  p - q = 5 := by
  sorry

end solution_difference_l3128_312860


namespace largest_number_proof_l3128_312871

theorem largest_number_proof (a b c d e : ℝ) 
  (ha : a = 0.997) (hb : b = 0.979) (hc : c = 0.99) (hd : d = 0.9709) (he : e = 0.999) :
  e = max a (max b (max c (max d e))) :=
by sorry

end largest_number_proof_l3128_312871


namespace second_division_percentage_l3128_312894

/-- Proves that the percentage of students who got second division is 54% -/
theorem second_division_percentage
  (total_students : ℕ)
  (first_division_percentage : ℚ)
  (just_passed : ℕ)
  (h_total : total_students = 300)
  (h_first : first_division_percentage = 26 / 100)
  (h_passed : just_passed = 60)
  (h_all_passed : total_students = 
    (first_division_percentage * total_students).floor + 
    (total_students - (first_division_percentage * total_students).floor - just_passed) + 
    just_passed) :
  (total_students - (first_division_percentage * total_students).floor - just_passed : ℚ) / 
  total_students * 100 = 54 := by
  sorry

end second_division_percentage_l3128_312894


namespace quadratic_order_l3128_312873

/-- Given m < -2 and points on a quadratic function, prove y3 < y2 < y1 -/
theorem quadratic_order (m : ℝ) (y1 y2 y3 : ℝ)
  (h_m : m < -2)
  (h_y1 : y1 = (m - 1)^2 + 2*(m - 1))
  (h_y2 : y2 = m^2 + 2*m)
  (h_y3 : y3 = (m + 1)^2 + 2*(m + 1)) :
  y3 < y2 ∧ y2 < y1 := by
  sorry

end quadratic_order_l3128_312873


namespace geometric_series_sum_l3128_312810

/-- Sum of a geometric series with n terms -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The given geometric series -/
def givenSeries : List ℚ := [1/4, -1/16, 1/64, -1/256, 1/1024]

theorem geometric_series_sum :
  let a₁ : ℚ := 1/4
  let r : ℚ := -1/4
  let n : ℕ := 5
  geometricSum a₁ r n = 205/1024 ∧ givenSeries.sum = 205/1024 := by
  sorry

end geometric_series_sum_l3128_312810
