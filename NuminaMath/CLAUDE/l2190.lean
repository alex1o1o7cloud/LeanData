import Mathlib

namespace NUMINAMATH_CALUDE_miles_driven_equals_365_l2190_219085

/-- Calculates the total miles driven given car efficiencies and gas usage --/
def total_miles_driven (highway_mpg city_mpg : ℚ) (total_gas : ℚ) (highway_city_diff : ℚ) : ℚ :=
  let city_miles := (total_gas * highway_mpg * city_mpg - city_mpg * highway_city_diff) / (highway_mpg + city_mpg)
  let highway_miles := city_miles + highway_city_diff
  city_miles + highway_miles

/-- Theorem stating the total miles driven under given conditions --/
theorem miles_driven_equals_365 :
  total_miles_driven 37 30 11 5 = 365 := by
  sorry

end NUMINAMATH_CALUDE_miles_driven_equals_365_l2190_219085


namespace NUMINAMATH_CALUDE_group_count_divisible_by_27_l2190_219031

/-- Represents the number of groups of each size -/
structure GroupCounts where
  size2 : ℕ
  size5 : ℕ
  size11 : ℕ

/-- The mean size of a group is 4 -/
def mean_size_condition (g : GroupCounts) : Prop :=
  (2 * g.size2 + 5 * g.size5 + 11 * g.size11) / (g.size2 + g.size5 + g.size11) = 4

/-- The mean of answers when each person is asked how many others are in their group is 6 -/
def mean_answer_condition (g : GroupCounts) : Prop :=
  (2 * g.size2 * 1 + 5 * g.size5 * 4 + 11 * g.size11 * 10) / (2 * g.size2 + 5 * g.size5 + 11 * g.size11) = 6

/-- The main theorem to prove -/
theorem group_count_divisible_by_27 (g : GroupCounts) 
  (h1 : mean_size_condition g) (h2 : mean_answer_condition g) : 
  ∃ k : ℕ, g.size2 + g.size5 + g.size11 = 27 * k := by
  sorry

end NUMINAMATH_CALUDE_group_count_divisible_by_27_l2190_219031


namespace NUMINAMATH_CALUDE_circle_chord_intersection_l2190_219004

theorem circle_chord_intersection (r : ℝ) (chord_length : ℝ) :
  r = 7 →
  chord_length = 10 →
  let segment_length := r - 2 * Real.sqrt 6
  ∃ (AK KB : ℝ),
    AK = segment_length ∧
    KB = 2 * r - segment_length ∧
    AK + KB = 2 * r ∧
    AK * KB = (chord_length / 2) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_chord_intersection_l2190_219004


namespace NUMINAMATH_CALUDE_integer_solutions_count_l2190_219054

theorem integer_solutions_count : 
  ∃ (S : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ S ↔ x^2 - y^2 = 1988) ∧ 
    Finset.card S = 8 := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_count_l2190_219054


namespace NUMINAMATH_CALUDE_age_equality_time_l2190_219016

/-- Given two people a and b, where a is 5 years older than b and their present ages sum to 13,
    this theorem proves that it will take 11 years for thrice a's age to equal 4 times b's age. -/
theorem age_equality_time (a b : ℕ) : 
  a = b + 5 → 
  a + b = 13 → 
  ∃ x : ℕ, x = 11 ∧ 3 * (a + x) = 4 * (b + x) :=
by sorry

end NUMINAMATH_CALUDE_age_equality_time_l2190_219016


namespace NUMINAMATH_CALUDE_two_consistent_faces_l2190_219010

/-- A graph representing a convex polyhedron -/
structure ConvexPolyhedronGraph where
  V : Type* -- Vertices
  E : Type* -- Edges
  F : Type* -- Faces
  adj : V → List E -- Adjacent edges for each vertex
  face_edges : F → List E -- Edges for each face
  orientation : E → Bool -- Edge orientation (True for outgoing, False for incoming)

/-- Number of changes in edge orientation around a vertex -/
def vertex_orientation_changes (G : ConvexPolyhedronGraph) (v : G.V) : Nat :=
  sorry

/-- Number of changes in edge orientation around a face -/
def face_orientation_changes (G : ConvexPolyhedronGraph) (f : G.F) : Nat :=
  sorry

/-- Main theorem -/
theorem two_consistent_faces (G : ConvexPolyhedronGraph)
  (h1 : ∀ v : G.V, ∃ e1 e2 : G.E, e1 ∈ G.adj v ∧ e2 ∈ G.adj v ∧ G.orientation e1 ≠ G.orientation e2) :
  ∃ f1 f2 : G.F, f1 ≠ f2 ∧ face_orientation_changes G f1 = 0 ∧ face_orientation_changes G f2 = 0 :=
sorry

end NUMINAMATH_CALUDE_two_consistent_faces_l2190_219010


namespace NUMINAMATH_CALUDE_no_solution_a_no_solution_b_no_solution_c_no_solution_d_no_solution_e_no_solution_f_no_solution_g_no_solution_h_l2190_219012

-- a) √(x+2) = -2
theorem no_solution_a : ¬∃ x : ℝ, Real.sqrt (x + 2) = -2 := by sorry

-- b) √(2x+3) + √(x+3) = 0
theorem no_solution_b : ¬∃ x : ℝ, Real.sqrt (2*x + 3) + Real.sqrt (x + 3) = 0 := by sorry

-- c) √(4-x) - √(x-6) = 2
theorem no_solution_c : ¬∃ x : ℝ, Real.sqrt (4 - x) - Real.sqrt (x - 6) = 2 := by sorry

-- d) √(-1-x) = ∛(x-5)
theorem no_solution_d : ¬∃ x : ℝ, Real.sqrt (-1 - x) = (x - 5) ^ (1/3 : ℝ) := by sorry

-- e) 5√x - 3√(-x) + 17/x = 4
theorem no_solution_e : ¬∃ x : ℝ, 5 * Real.sqrt x - 3 * Real.sqrt (-x) + 17 / x = 4 := by sorry

-- f) √(x-3) - √(x+9) = √(x-2)
theorem no_solution_f : ¬∃ x : ℝ, Real.sqrt (x - 3) - Real.sqrt (x + 9) = Real.sqrt (x - 2) := by sorry

-- g) √x + √(x+9) = 2
theorem no_solution_g : ¬∃ x : ℝ, Real.sqrt x + Real.sqrt (x + 9) = 2 := by sorry

-- h) ∛(x + 1/x) = √(-x) - 1
theorem no_solution_h : ¬∃ x : ℝ, (x + 1/x) ^ (1/3 : ℝ) = Real.sqrt (-x) - 1 := by sorry

end NUMINAMATH_CALUDE_no_solution_a_no_solution_b_no_solution_c_no_solution_d_no_solution_e_no_solution_f_no_solution_g_no_solution_h_l2190_219012


namespace NUMINAMATH_CALUDE_roots_equation_sum_l2190_219082

theorem roots_equation_sum (α β : ℝ) : 
  α^2 - 3*α + 1 = 0 → β^2 - 3*β + 1 = 0 → 3*α^5 + 7*β^4 = 817 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_sum_l2190_219082


namespace NUMINAMATH_CALUDE_sequence_term_correct_l2190_219002

def sequence_sum (n : ℕ) : ℕ := 2^n + 3

def sequence_term (n : ℕ) : ℕ :=
  match n with
  | 1 => 5
  | _ => 2^(n-1)

theorem sequence_term_correct :
  ∀ n : ℕ, n ≥ 1 → sequence_term n = 
    if n = 1 
    then sequence_sum 1
    else sequence_sum n - sequence_sum (n-1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_term_correct_l2190_219002


namespace NUMINAMATH_CALUDE_min_teachers_for_our_school_l2190_219092

/-- Represents a school with math, physics, and chemistry teachers -/
structure School where
  mathTeachers : ℕ
  physicsTeachers : ℕ
  chemistryTeachers : ℕ
  maxSubjectsPerTeacher : ℕ

/-- The minimum number of teachers required for a given school -/
def minTeachersRequired (s : School) : ℕ := sorry

/-- Our specific school configuration -/
def ourSchool : School :=
  { mathTeachers := 4
    physicsTeachers := 3
    chemistryTeachers := 3
    maxSubjectsPerTeacher := 2 }

/-- Theorem stating that the minimum number of teachers required for our school is 6 -/
theorem min_teachers_for_our_school :
  minTeachersRequired ourSchool = 6 := by sorry

end NUMINAMATH_CALUDE_min_teachers_for_our_school_l2190_219092


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2190_219081

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 3 / b₀ = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2190_219081


namespace NUMINAMATH_CALUDE_matildas_father_chocolates_l2190_219045

/-- Calculates the number of chocolate bars Matilda's father had left -/
def fathersRemainingChocolates (initialBars : ℕ) (people : ℕ) (givenToMother : ℕ) (eaten : ℕ) : ℕ :=
  let barsPerPerson := initialBars / people
  let givenToFather := people * (barsPerPerson / 2)
  givenToFather - givenToMother - eaten

/-- Proves that Matilda's father had 5 chocolate bars left -/
theorem matildas_father_chocolates :
  fathersRemainingChocolates 20 5 3 2 = 5 := by
  sorry

#eval fathersRemainingChocolates 20 5 3 2

end NUMINAMATH_CALUDE_matildas_father_chocolates_l2190_219045


namespace NUMINAMATH_CALUDE_comparison_square_and_power_l2190_219018

theorem comparison_square_and_power (n : ℕ) (h : n ≥ 3) : (n + 1)^2 < 3^n := by
  sorry

end NUMINAMATH_CALUDE_comparison_square_and_power_l2190_219018


namespace NUMINAMATH_CALUDE_house_cost_proof_l2190_219093

def land_acres : ℕ := 30
def land_cost_per_acre : ℕ := 20
def num_cows : ℕ := 20
def cost_per_cow : ℕ := 1000
def num_chickens : ℕ := 100
def cost_per_chicken : ℕ := 5
def solar_install_hours : ℕ := 6
def solar_install_cost_per_hour : ℕ := 100
def solar_equipment_cost : ℕ := 6000
def total_cost : ℕ := 147700

def land_cost : ℕ := land_acres * land_cost_per_acre
def cows_cost : ℕ := num_cows * cost_per_cow
def chickens_cost : ℕ := num_chickens * cost_per_chicken
def solar_install_cost : ℕ := solar_install_hours * solar_install_cost_per_hour
def total_solar_cost : ℕ := solar_install_cost + solar_equipment_cost

theorem house_cost_proof :
  total_cost - (land_cost + cows_cost + chickens_cost + total_solar_cost) = 120000 := by
  sorry

end NUMINAMATH_CALUDE_house_cost_proof_l2190_219093


namespace NUMINAMATH_CALUDE_shell_difference_l2190_219017

theorem shell_difference (perfect_shells broken_shells non_spiral_perfect : ℕ) 
  (h1 : perfect_shells = 17)
  (h2 : broken_shells = 52)
  (h3 : non_spiral_perfect = 12) : 
  (broken_shells / 2) - (perfect_shells - non_spiral_perfect) = 21 := by
  sorry

end NUMINAMATH_CALUDE_shell_difference_l2190_219017


namespace NUMINAMATH_CALUDE_part_one_part_two_l2190_219065

-- Define proposition p
def p (k : ℝ) : Prop := ∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 + 2*x - k ≤ 0

-- Define proposition q
def q (k : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*k*x + 3*k + 4 = 0

-- Theorem for part 1
theorem part_one (k : ℝ) : p k → k ∈ Set.Ici 3 := by sorry

-- Theorem for part 2
theorem part_two (k : ℝ) : 
  (p k ∧ ¬q k) ∨ (¬p k ∧ q k) → k ∈ Set.Iic (-1) ∪ Set.Ico 3 4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2190_219065


namespace NUMINAMATH_CALUDE_mod_thirteen_problem_l2190_219097

theorem mod_thirteen_problem (a : ℤ) 
  (h1 : 0 < a) (h2 : a < 13) 
  (h3 : (53^2017 + a) % 13 = 0) : 
  a = 12 := by
  sorry

end NUMINAMATH_CALUDE_mod_thirteen_problem_l2190_219097


namespace NUMINAMATH_CALUDE_parabola_intersection_sum_l2190_219030

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (x y : ℝ) : Prop :=
  ∃ (t : ℝ), x = 1 + t ∧ y = t

-- Define the intersection points
structure IntersectionPoint where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y
  on_line : line_through_focus x y

-- Theorem statement
theorem parabola_intersection_sum (A B : IntersectionPoint) :
  (A.x - (-1))^2 + (A.y)^2 + (B.x - (-1))^2 + (B.y)^2 = 64 →
  A.x + B.x = 6 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_sum_l2190_219030


namespace NUMINAMATH_CALUDE_odd_function_inequality_l2190_219068

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x / 3 - 2^x
  else if x < 0 then x / 3 + 2^(-x)
  else 0

-- State the theorem
theorem odd_function_inequality (k : ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ t, f (t^2 - 2*t) + f (2*t^2 - k) < 0) →
  k < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_inequality_l2190_219068


namespace NUMINAMATH_CALUDE_initial_student_count_l2190_219075

/-- Given the initial average weight and the new average weight after admitting a new student,
    prove that the initial number of students is 29. -/
theorem initial_student_count
  (initial_avg : ℝ)
  (new_avg : ℝ)
  (new_student_weight : ℝ)
  (h1 : initial_avg = 28)
  (h2 : new_avg = 27.1)
  (h3 : new_student_weight = 1)
  : ∃ n : ℕ, n = 29 ∧ 
    n * initial_avg + new_student_weight = (n + 1) * new_avg :=
by
  sorry


end NUMINAMATH_CALUDE_initial_student_count_l2190_219075


namespace NUMINAMATH_CALUDE_equal_cubic_expressions_l2190_219083

theorem equal_cubic_expressions (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 3)
  (sum_squares_eq : a^2 + b^2 + c^2 + d^2 = 3)
  (sum_triple_products_eq : a*b*c + b*c*d + c*d*a + d*a*b = 1) :
  a*(1-a)^3 = b*(1-b)^3 ∧ b*(1-b)^3 = c*(1-c)^3 ∧ c*(1-c)^3 = d*(1-d)^3 := by
  sorry

end NUMINAMATH_CALUDE_equal_cubic_expressions_l2190_219083


namespace NUMINAMATH_CALUDE_reflection_maps_points_l2190_219003

/-- Reflects a point across the line y = x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem reflection_maps_points :
  let A : ℝ × ℝ := (-3, 2)
  let B : ℝ × ℝ := (-2, 5)
  let A' : ℝ × ℝ := (2, -3)
  let B' : ℝ × ℝ := (5, -2)
  reflect_y_eq_x A = A' ∧ reflect_y_eq_x B = B' := by
  sorry


end NUMINAMATH_CALUDE_reflection_maps_points_l2190_219003


namespace NUMINAMATH_CALUDE_range_of_a_l2190_219047

-- Define propositions p and q
def p (a : ℝ) : Prop := -2 < a ∧ a ≤ 2
def q (a : ℝ) : Prop := 0 < a ∧ a < 1

-- Define the set of valid a values
def valid_a_set : Set ℝ := {a | (1 ≤ a ∧ a ≤ 2) ∨ (-2 < a ∧ a ≤ 0)}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ valid_a_set :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2190_219047


namespace NUMINAMATH_CALUDE_simplify_fraction_l2190_219086

theorem simplify_fraction (b : ℝ) (h : b = 5) : 15 * b^4 / (75 * b^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2190_219086


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2190_219061

/-- The equation of a hyperbola -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

/-- The equations of the asymptotes -/
def asymptotes (x y : ℝ) : Prop := (Real.sqrt 3 * x + 2 * y = 0) ∨ (Real.sqrt 3 * x - 2 * y = 0)

/-- Theorem: The equations of the asymptotes of the given hyperbola -/
theorem hyperbola_asymptotes : ∀ x y : ℝ, hyperbola x y → asymptotes x y := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2190_219061


namespace NUMINAMATH_CALUDE_stacy_brother_growth_l2190_219022

/-- Proves that Stacy's brother grew 1 inch last year -/
theorem stacy_brother_growth (stacy_initial_height stacy_final_height stacy_growth_difference : ℕ) 
  (h1 : stacy_initial_height = 50)
  (h2 : stacy_final_height = 57)
  (h3 : stacy_growth_difference = 6) :
  stacy_final_height - stacy_initial_height - stacy_growth_difference = 1 := by
  sorry

end NUMINAMATH_CALUDE_stacy_brother_growth_l2190_219022


namespace NUMINAMATH_CALUDE_triangle_abc_is_right_l2190_219013

/-- Given three points in a 2D plane, determines if they form a right triangle --/
def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let ab_squared := (B.1 - A.1)^2 + (B.2 - A.2)^2
  let bc_squared := (C.1 - B.1)^2 + (C.2 - B.2)^2
  let ca_squared := (A.1 - C.1)^2 + (A.2 - C.2)^2
  (ab_squared = bc_squared + ca_squared) ∨
  (bc_squared = ab_squared + ca_squared) ∨
  (ca_squared = ab_squared + bc_squared)

/-- The triangle formed by points A(5, -2), B(1, 5), and C(-1, 2) is a right triangle --/
theorem triangle_abc_is_right :
  is_right_triangle (5, -2) (1, 5) (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_is_right_l2190_219013


namespace NUMINAMATH_CALUDE_chessboard_decomposition_l2190_219063

/-- Represents a square on a chessboard -/
structure Square where
  side : ℕ
  area : ℕ
  area_eq : area = side * side

/-- Represents a chessboard -/
structure Chessboard where
  side : ℕ
  area : ℕ
  area_eq : area = side * side

/-- Represents a decomposition of a chessboard into squares -/
structure Decomposition (board : Chessboard) where
  squares : List Square
  piece_count : ℕ
  valid : piece_count = 6 ∧ squares.length = 3
  area_sum : (squares.map (·.area)).sum = board.area

/-- The main theorem: A 7x7 chessboard can be decomposed into 6 pieces 
    that form three squares of sizes 6x6, 3x3, and 2x2 -/
theorem chessboard_decomposition :
  ∃ (d : Decomposition ⟨7, 49, rfl⟩),
    d.squares = [⟨6, 36, rfl⟩, ⟨3, 9, rfl⟩, ⟨2, 4, rfl⟩] := by
  sorry

end NUMINAMATH_CALUDE_chessboard_decomposition_l2190_219063


namespace NUMINAMATH_CALUDE_spider_dressing_theorem_l2190_219024

/-- The number of legs of the spider -/
def num_legs : ℕ := 10

/-- The number of items per leg -/
def items_per_leg : ℕ := 3

/-- The total number of items -/
def total_items : ℕ := num_legs * items_per_leg

/-- The number of possible arrangements of items for one leg -/
def arrangements_per_leg : ℕ := Nat.factorial items_per_leg

theorem spider_dressing_theorem :
  (Nat.factorial total_items) / (arrangements_per_leg ^ num_legs) = 
  (Nat.factorial (num_legs * items_per_leg)) / (Nat.factorial items_per_leg ^ num_legs) := by
  sorry

end NUMINAMATH_CALUDE_spider_dressing_theorem_l2190_219024


namespace NUMINAMATH_CALUDE_triangle_geometric_sequence_l2190_219095

theorem triangle_geometric_sequence (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧
  -- a, b, c form a geometric sequence
  (b ^ 2 = a * c) ∧
  -- Given trigonometric ratios
  (Real.sin B = 5 / 13) ∧
  (Real.cos B = 12 / (a * c)) →
  -- Conclusion
  a + c = 3 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_geometric_sequence_l2190_219095


namespace NUMINAMATH_CALUDE_jack_grassy_time_is_six_l2190_219072

/-- Represents the race up the hill -/
structure HillRace where
  jackSandyTime : ℝ
  jackSpeedIncrease : ℝ
  jillTotalTime : ℝ
  jillFinishDifference : ℝ

/-- Calculates Jack's time on the grassy second half of the hill -/
def jackGrassyTime (race : HillRace) : ℝ :=
  race.jillTotalTime - race.jillFinishDifference - race.jackSandyTime

/-- Theorem stating that Jack's time on the grassy second half is 6 seconds -/
theorem jack_grassy_time_is_six (race : HillRace) 
  (h1 : race.jackSandyTime = 19)
  (h2 : race.jackSpeedIncrease = 0.25)
  (h3 : race.jillTotalTime = 32)
  (h4 : race.jillFinishDifference = 7) :
  jackGrassyTime race = 6 := by
  sorry

#check jack_grassy_time_is_six

end NUMINAMATH_CALUDE_jack_grassy_time_is_six_l2190_219072


namespace NUMINAMATH_CALUDE_sum_of_series_equals_25_16_l2190_219021

theorem sum_of_series_equals_25_16 : 
  (∑' n, n / 5^n) + (∑' n, (1 / 5)^n) = 25 / 16 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_25_16_l2190_219021


namespace NUMINAMATH_CALUDE_statement_c_not_always_true_l2190_219043

theorem statement_c_not_always_true :
  ¬ ∀ (a b c : ℝ), a > b → a * c^2 > b * c^2 := by sorry

end NUMINAMATH_CALUDE_statement_c_not_always_true_l2190_219043


namespace NUMINAMATH_CALUDE_calculations_correctness_l2190_219006

theorem calculations_correctness : 
  (-3 - 1 ≠ -2) ∧ 
  ((-3/4) - (3/4) ≠ 0) ∧ 
  (-8 / (-2) ≠ -4) ∧ 
  ((-3)^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_calculations_correctness_l2190_219006


namespace NUMINAMATH_CALUDE_distribution_difference_l2190_219051

theorem distribution_difference (total : ℕ) (p q r s : ℕ) : 
  total = 1000 →
  p = 2 * q →
  s = 4 * r →
  q = r →
  p + q + r + s = total →
  s - p = 250 := by
  sorry

end NUMINAMATH_CALUDE_distribution_difference_l2190_219051


namespace NUMINAMATH_CALUDE_prob_A_wins_3_1_l2190_219042

/-- The probability of Team A winning a single game -/
def prob_A_win : ℚ := 1/2

/-- The probability of Team B winning a single game -/
def prob_B_win : ℚ := 1/2

/-- The number of games in a best-of-five series where one team wins 3-1 -/
def games_played : ℕ := 4

/-- The number of ways to arrange 3 wins in 4 games -/
def winning_scenarios : ℕ := 3

/-- The probability of Team A winning with a score of 3:1 in a best-of-five series -/
theorem prob_A_wins_3_1 : 
  (prob_A_win ^ 3 * prob_B_win) * winning_scenarios = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_wins_3_1_l2190_219042


namespace NUMINAMATH_CALUDE_joans_balloons_l2190_219034

theorem joans_balloons (initial_balloons final_balloons : ℕ) 
  (h1 : initial_balloons = 8)
  (h2 : final_balloons = 10) :
  final_balloons - initial_balloons = 2 := by
  sorry

end NUMINAMATH_CALUDE_joans_balloons_l2190_219034


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2190_219028

/-- Given planar vectors a and b, prove that m = 1 makes ma + b perpendicular to a -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (-1, 3)) (h2 : b = (4, -2)) :
  ∃ m : ℝ, m = 1 ∧ (m • a + b) • a = 0 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2190_219028


namespace NUMINAMATH_CALUDE_profit_maximized_at_25_l2190_219037

/-- Profit function for the commodity -/
def profit (x : ℤ) : ℤ := (x - 20) * (30 - x)

/-- Theorem stating that profit is maximized at x = 25 -/
theorem profit_maximized_at_25 :
  ∀ x : ℤ, 20 ≤ x → x ≤ 30 → profit x ≤ profit 25 :=
by
  sorry

#check profit_maximized_at_25

end NUMINAMATH_CALUDE_profit_maximized_at_25_l2190_219037


namespace NUMINAMATH_CALUDE_sum_of_digits_plus_two_l2190_219025

/-- T(n) represents the sum of the digits of a positive integer n -/
def T (n : ℕ+) : ℕ := sorry

/-- For a certain positive integer n, T(n) = 1598 implies T(n+2) = 1600 -/
theorem sum_of_digits_plus_two (n : ℕ+) (h : T n = 1598) : T (n + 2) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_plus_two_l2190_219025


namespace NUMINAMATH_CALUDE_zeros_of_cosine_minus_one_l2190_219039

theorem zeros_of_cosine_minus_one (ω : ℝ) : 
  (ω > 0) →
  (∃ (x₁ x₂ x₃ : ℝ), 
    (x₁ ∈ Set.Icc 0 (2 * Real.pi)) ∧ 
    (x₂ ∈ Set.Icc 0 (2 * Real.pi)) ∧ 
    (x₃ ∈ Set.Icc 0 (2 * Real.pi)) ∧ 
    (x₁ ≠ x₂) ∧ (x₂ ≠ x₃) ∧ (x₁ ≠ x₃) ∧
    (Real.cos (ω * x₁) = 1) ∧ 
    (Real.cos (ω * x₂) = 1) ∧ 
    (Real.cos (ω * x₃) = 1) ∧
    (∀ x ∈ Set.Icc 0 (2 * Real.pi), Real.cos (ω * x) = 1 → (x = x₁ ∨ x = x₂ ∨ x = x₃))) ↔
  (2 ≤ ω ∧ ω < 3) :=
by sorry

end NUMINAMATH_CALUDE_zeros_of_cosine_minus_one_l2190_219039


namespace NUMINAMATH_CALUDE_louise_needs_30_boxes_l2190_219058

/-- Represents the number of pencils each box can hold for different colors --/
structure BoxCapacity where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- Represents the number of pencils Louise has for each color --/
structure PencilCount where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the number of boxes needed for a given color --/
def boxesNeeded (capacity : ℕ) (count : ℕ) : ℕ :=
  (count + capacity - 1) / capacity

/-- Calculates the total number of boxes Louise needs --/
def totalBoxesNeeded (capacity : BoxCapacity) (count : PencilCount) : ℕ :=
  boxesNeeded capacity.red count.red +
  boxesNeeded capacity.blue count.blue +
  boxesNeeded capacity.yellow count.yellow +
  boxesNeeded capacity.green count.green

/-- The main theorem stating that Louise needs 30 boxes --/
theorem louise_needs_30_boxes :
  let capacity := BoxCapacity.mk 15 25 10 30
  let redCount := 45
  let blueCount := 3 * redCount + 6
  let yellowCount := 80
  let greenCount := 2 * (redCount + blueCount)
  let count := PencilCount.mk redCount blueCount yellowCount greenCount
  totalBoxesNeeded capacity count = 30 := by
  sorry


end NUMINAMATH_CALUDE_louise_needs_30_boxes_l2190_219058


namespace NUMINAMATH_CALUDE_percentage_less_than_l2190_219076

theorem percentage_less_than (x y z : ℝ) :
  x = 1.2 * y →
  x = 0.84 * z →
  y = 0.7 * z :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_less_than_l2190_219076


namespace NUMINAMATH_CALUDE_complex_magnitude_l2190_219099

theorem complex_magnitude (z : ℂ) (h : z / (1 - Complex.I)^2 = (1 + Complex.I) / 2) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2190_219099


namespace NUMINAMATH_CALUDE_michelangelo_painting_l2190_219044

theorem michelangelo_painting (total : ℕ) (left : ℕ) (this_week : ℕ) : 
  total = 28 → 
  left = 13 → 
  total - left = this_week + this_week / 4 →
  this_week = 12 := by
  sorry

end NUMINAMATH_CALUDE_michelangelo_painting_l2190_219044


namespace NUMINAMATH_CALUDE_max_value_expression_l2190_219011

theorem max_value_expression (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  2 * x * y * Real.sqrt 10 + 10 * y * z ≤ 5 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l2190_219011


namespace NUMINAMATH_CALUDE_circle_radius_l2190_219036

/-- The radius of the circle defined by x^2 + y^2 + 2x + 6y = 0 is √10 -/
theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 + 2*x + 6*y = 0) → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l2190_219036


namespace NUMINAMATH_CALUDE_erasers_given_to_doris_l2190_219005

def initial_erasers : ℕ := 81
def final_erasers : ℕ := 47

theorem erasers_given_to_doris : initial_erasers - final_erasers = 34 := by
  sorry

end NUMINAMATH_CALUDE_erasers_given_to_doris_l2190_219005


namespace NUMINAMATH_CALUDE_ad_fraction_of_page_l2190_219059

theorem ad_fraction_of_page 
  (page_width : ℝ) 
  (page_height : ℝ) 
  (cost_per_sq_inch : ℝ) 
  (total_cost : ℝ) : 
  page_width = 9 → 
  page_height = 12 → 
  cost_per_sq_inch = 8 → 
  total_cost = 432 → 
  (total_cost / cost_per_sq_inch) / (page_width * page_height) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ad_fraction_of_page_l2190_219059


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_l2190_219032

theorem triangle_cosine_sum (A B C : ℝ) (h1 : A + B + C = π) (h2 : A = 3 * B) (h3 : A = 9 * C) :
  Real.cos A * Real.cos B + Real.cos B * Real.cos C + Real.cos C * Real.cos A = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_l2190_219032


namespace NUMINAMATH_CALUDE_average_speed_two_segment_trip_l2190_219048

/-- Given a trip with two segments, prove that the average speed is as calculated -/
theorem average_speed_two_segment_trip (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : distance1 = 360)
  (h2 : speed1 = 60)
  (h3 : distance2 = 120)
  (h4 : speed2 = 40) :
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 480 / 9 := by
sorry

end NUMINAMATH_CALUDE_average_speed_two_segment_trip_l2190_219048


namespace NUMINAMATH_CALUDE_constant_remainder_condition_l2190_219064

-- Define the polynomials
def dividend (a : ℝ) (x : ℝ) : ℝ := 12 * x^3 - 9 * x^2 + a * x + 8
def divisor (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the theorem
theorem constant_remainder_condition (a : ℝ) :
  (∃ (r : ℝ), ∀ (x : ℝ), ∃ (q : ℝ), dividend a x = divisor x * q + r) ↔ a = -7 := by
  sorry

end NUMINAMATH_CALUDE_constant_remainder_condition_l2190_219064


namespace NUMINAMATH_CALUDE_compound_formula_l2190_219088

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00

-- Define the number of oxygen atoms
def num_O : ℕ := 3

-- Define the total molecular weight
def total_molecular_weight : ℝ := 102

-- Define the molecular formula
structure MolecularFormula where
  num_Al : ℕ
  num_O : ℕ

-- Theorem to prove
theorem compound_formula :
  ∃ (formula : MolecularFormula),
    formula.num_O = num_O ∧
    formula.num_Al * atomic_weight_Al + formula.num_O * atomic_weight_O = total_molecular_weight ∧
    formula = MolecularFormula.mk 2 3 := by
  sorry


end NUMINAMATH_CALUDE_compound_formula_l2190_219088


namespace NUMINAMATH_CALUDE_distance_minimization_l2190_219089

theorem distance_minimization (t : ℝ) (h : t > 0) :
  let f (x : ℝ) := x^2 + 1
  let g (x : ℝ) := Real.log x
  let distance_squared (x : ℝ) := (f x - g x)^2
  (∀ x > 0, distance_squared t ≤ distance_squared x) →
  t = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_minimization_l2190_219089


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2190_219046

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + 3 * a 8 + a 15 = 100) :
  2 * a 9 - a 10 = 20 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2190_219046


namespace NUMINAMATH_CALUDE_least_number_of_cans_l2190_219069

def maaza_liters : ℕ := 157
def pepsi_liters : ℕ := 173
def sprite_liters : ℕ := 389

def total_cans : ℕ := maaza_liters + pepsi_liters + sprite_liters

theorem least_number_of_cans :
  ∀ (can_size : ℕ),
    can_size > 0 →
    can_size ∣ maaza_liters →
    can_size ∣ pepsi_liters →
    can_size ∣ sprite_liters →
    total_cans ≤ maaza_liters / can_size + pepsi_liters / can_size + sprite_liters / can_size :=
by sorry

end NUMINAMATH_CALUDE_least_number_of_cans_l2190_219069


namespace NUMINAMATH_CALUDE_perpendicular_lines_relationship_l2190_219057

-- Define a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define perpendicularity between a line and a vector
def perpendicular (l : Line3D) (v : ℝ × ℝ × ℝ) : Prop :=
  let (dx, dy, dz) := l.direction
  let (vx, vy, vz) := v
  dx * vx + dy * vy + dz * vz = 0

-- Define the relationships between two lines
inductive LineRelationship
  | Parallel
  | Intersecting
  | Skew

-- State the theorem
theorem perpendicular_lines_relationship (l1 l2 l3 : Line3D) 
  (h1 : perpendicular l1 l3.direction) 
  (h2 : perpendicular l2 l3.direction) :
  ∃ r : LineRelationship, true :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_relationship_l2190_219057


namespace NUMINAMATH_CALUDE_part_to_third_ratio_l2190_219079

theorem part_to_third_ratio (N P : ℝ) (h1 : (1/4) * (1/3) * P = 20) (h2 : 0.40 * N = 240) :
  P / ((1/3) * N) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_part_to_third_ratio_l2190_219079


namespace NUMINAMATH_CALUDE_sector_max_area_l2190_219091

/-- Given a sector of a circle with perimeter 30 cm, prove that the maximum area is 225/4 cm² and the corresponding central angle is 2 radians. -/
theorem sector_max_area (l R : ℝ) (h1 : l + 2*R = 30) (h2 : l > 0) (h3 : R > 0) :
  ∃ (S : ℝ), S ≤ 225/4 ∧
  (S = 225/4 → l = 15 ∧ R = 15/2 ∧ l / R = 2) :=
sorry

end NUMINAMATH_CALUDE_sector_max_area_l2190_219091


namespace NUMINAMATH_CALUDE_perfect_squares_identification_l2190_219050

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def option_A : ℕ := 3^4 * 4^5 * 7^7
def option_B : ℕ := 3^6 * 4^4 * 7^6
def option_C : ℕ := 3^5 * 4^6 * 7^5
def option_D : ℕ := 3^4 * 4^7 * 7^4
def option_E : ℕ := 3^6 * 4^6 * 7^6

theorem perfect_squares_identification :
  ¬(is_perfect_square option_A) ∧
  (is_perfect_square option_B) ∧
  ¬(is_perfect_square option_C) ∧
  (is_perfect_square option_D) ∧
  (is_perfect_square option_E) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_identification_l2190_219050


namespace NUMINAMATH_CALUDE_monica_savings_l2190_219052

/-- Calculates the total amount saved given the weekly savings, number of weeks, and number of repetitions. -/
def total_savings (weekly_savings : ℕ) (weeks : ℕ) (repetitions : ℕ) : ℕ :=
  weekly_savings * weeks * repetitions

/-- Proves that saving $15 per week for 60 weeks, repeated 5 times, results in a total savings of $4500. -/
theorem monica_savings : total_savings 15 60 5 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_monica_savings_l2190_219052


namespace NUMINAMATH_CALUDE_percentage_calculation_l2190_219078

theorem percentage_calculation (part whole : ℝ) (h1 : part = 375.2) (h2 : whole = 12546.8) :
  (part / whole) * 100 = 2.99 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2190_219078


namespace NUMINAMATH_CALUDE_pamphlet_cost_l2190_219020

theorem pamphlet_cost : ∃ p : ℝ, p = 1.10 ∧ 8 * p < 9 ∧ 11 * p > 12 := by
  sorry

end NUMINAMATH_CALUDE_pamphlet_cost_l2190_219020


namespace NUMINAMATH_CALUDE_job_completion_days_l2190_219014

/-- The number of days initially planned for a job to be completed, given:
  * 6 workers start the job
  * After 3 days, 4 more workers join
  * With 10 workers, the job is finished in 3 more days
  * Each worker has the same efficiency -/
def initial_days : ℕ := 6

/-- The total amount of work to be done -/
def total_work : ℝ := 1

/-- The number of workers that start the job -/
def initial_workers : ℕ := 6

/-- The number of days worked before additional workers join -/
def days_before_join : ℕ := 3

/-- The number of additional workers that join -/
def additional_workers : ℕ := 4

/-- The number of days needed to finish the job after additional workers join -/
def days_after_join : ℕ := 3

theorem job_completion_days :
  let work_rate := total_work / initial_days
  let work_done_before_join := days_before_join * work_rate
  let remaining_work := total_work - work_done_before_join
  let total_workers := initial_workers + additional_workers
  remaining_work / days_after_join = total_workers * work_rate
  → initial_days = 6 := by sorry

end NUMINAMATH_CALUDE_job_completion_days_l2190_219014


namespace NUMINAMATH_CALUDE_unique_non_negative_one_result_l2190_219008

theorem unique_non_negative_one_result :
  (-1 * 1 = -1) ∧
  ((-1) / (-1) ≠ -1) ∧
  (-2015 / 2015 = -1) ∧
  ((-1)^9 * (-1)^2 = -1) :=
by sorry

end NUMINAMATH_CALUDE_unique_non_negative_one_result_l2190_219008


namespace NUMINAMATH_CALUDE_sticker_distribution_l2190_219041

/-- The number of ways to distribute n identical objects into k distinct groups,
    where each group must have at least one object -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- The number of stickers -/
def num_stickers : ℕ := 10

/-- The number of sheets of paper -/
def num_sheets : ℕ := 5

theorem sticker_distribution :
  distribute num_stickers num_sheets = 126 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2190_219041


namespace NUMINAMATH_CALUDE_solution_triplets_l2190_219096

theorem solution_triplets (x y z : ℝ) : 
  x^2 * y + y^2 * z = 1040 ∧
  x^2 * z + z^2 * y = 260 ∧
  (x - y) * (y - z) * (z - x) = -540 →
  (x = 16 ∧ y = 4 ∧ z = 1) ∨ (x = 1 ∧ y = 16 ∧ z = 4) :=
by sorry

end NUMINAMATH_CALUDE_solution_triplets_l2190_219096


namespace NUMINAMATH_CALUDE_vector_BC_proof_l2190_219023

def A : ℝ × ℝ := (0, 0)  -- Assuming A as the origin for simplicity
def B : ℝ × ℝ := (2, 4)
def C : ℝ × ℝ := (1, 3)

def vector_AB : ℝ × ℝ := B
def vector_AC : ℝ × ℝ := C

theorem vector_BC_proof :
  (C.1 - B.1, C.2 - B.2) = (-1, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_BC_proof_l2190_219023


namespace NUMINAMATH_CALUDE_complex_square_root_of_negative_four_l2190_219094

theorem complex_square_root_of_negative_four :
  ∀ z : ℂ, z^2 = -4 ↔ z = 2*I ∨ z = -2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_root_of_negative_four_l2190_219094


namespace NUMINAMATH_CALUDE_total_books_read_is_36sc_l2190_219070

/-- The number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  let books_per_month : ℕ := 3
  let months_per_year : ℕ := 12
  let books_per_student_per_year : ℕ := books_per_month * months_per_year
  let total_students : ℕ := c * s
  books_per_student_per_year * total_students

/-- Theorem stating that the total number of books read is 36 * s * c -/
theorem total_books_read_is_36sc (c s : ℕ) :
  total_books_read c s = 36 * s * c := by
  sorry

end NUMINAMATH_CALUDE_total_books_read_is_36sc_l2190_219070


namespace NUMINAMATH_CALUDE_sum_of_pairwise_products_of_roots_l2190_219000

theorem sum_of_pairwise_products_of_roots (p q r : ℂ) : 
  2 * p^3 - 4 * p^2 + 8 * p - 5 = 0 →
  2 * q^3 - 4 * q^2 + 8 * q - 5 = 0 →
  2 * r^3 - 4 * r^2 + 8 * r - 5 = 0 →
  p * q + q * r + p * r = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_pairwise_products_of_roots_l2190_219000


namespace NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l2190_219040

theorem halfway_between_one_eighth_and_one_third : 
  (1 / 8 : ℚ) + ((1 / 3 : ℚ) - (1 / 8 : ℚ)) / 2 = 11 / 48 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l2190_219040


namespace NUMINAMATH_CALUDE_complex_sum_equality_l2190_219049

theorem complex_sum_equality : 
  12 * Complex.exp (3 * Real.pi * Complex.I / 13) + 
  12 * Complex.exp (7 * Real.pi * Complex.I / 26) = 
  24 * Real.cos (Real.pi / 26) * Complex.exp (19 * Real.pi * Complex.I / 52) := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l2190_219049


namespace NUMINAMATH_CALUDE_quadruplet_babies_l2190_219074

theorem quadruplet_babies (total : ℕ) (twins triplets quadruplets : ℕ) : 
  total = 1500 →
  2 * twins + 3 * triplets + 4 * quadruplets = total →
  triplets = 3 * quadruplets →
  twins = 2 * triplets →
  4 * quadruplets = 240 := by
sorry

end NUMINAMATH_CALUDE_quadruplet_babies_l2190_219074


namespace NUMINAMATH_CALUDE_equality_of_sets_l2190_219098

theorem equality_of_sets (x y a : ℝ) : 
  (3 * x^2 = x^2 + x^2 + x^2) ∧ 
  ((x - y)^2 = (y - x)^2) ∧ 
  ((a^2)^3 = (a^3)^2) := by
sorry

end NUMINAMATH_CALUDE_equality_of_sets_l2190_219098


namespace NUMINAMATH_CALUDE_cpu_sales_count_l2190_219029

/-- Represents the sales data for a hardware store for one week -/
structure HardwareSales where
  graphics_cards : Nat
  hard_drives : Nat
  cpus : Nat
  ram_pairs : Nat
  graphics_card_price : Nat
  hard_drive_price : Nat
  cpu_price : Nat
  ram_pair_price : Nat
  total_earnings : Nat

/-- Theorem stating that given the sales data, the number of CPUs sold is 8 -/
theorem cpu_sales_count (sales : HardwareSales) : 
  sales.graphics_cards = 10 ∧
  sales.hard_drives = 14 ∧
  sales.ram_pairs = 4 ∧
  sales.graphics_card_price = 600 ∧
  sales.hard_drive_price = 80 ∧
  sales.cpu_price = 200 ∧
  sales.ram_pair_price = 60 ∧
  sales.total_earnings = 8960 →
  sales.cpus = 8 := by
  sorry

#check cpu_sales_count

end NUMINAMATH_CALUDE_cpu_sales_count_l2190_219029


namespace NUMINAMATH_CALUDE_day_of_week_p_minus_one_l2190_219056

-- Define a type for days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the day of the week for a given day number
def dayOfWeek (dayNumber : Nat) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

-- Define the theorem
theorem day_of_week_p_minus_one (P : Nat) :
  dayOfWeek 250 = DayOfWeek.Sunday →
  dayOfWeek 150 = DayOfWeek.Sunday →
  dayOfWeek 50 = DayOfWeek.Sunday :=
by
  sorry

-- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_day_of_week_p_minus_one_l2190_219056


namespace NUMINAMATH_CALUDE_minimum_packages_for_equal_shipment_l2190_219027

theorem minimum_packages_for_equal_shipment (sarah_capacity : Nat) (ryan_capacity : Nat) (emily_capacity : Nat)
  (h1 : sarah_capacity = 18)
  (h2 : ryan_capacity = 11)
  (h3 : emily_capacity = 15) :
  Nat.lcm (Nat.lcm sarah_capacity ryan_capacity) emily_capacity = 990 :=
by sorry

end NUMINAMATH_CALUDE_minimum_packages_for_equal_shipment_l2190_219027


namespace NUMINAMATH_CALUDE_ratio_sum_to_y_l2190_219001

theorem ratio_sum_to_y (w x y : ℝ) (hw_x : w / x = 1 / 3) (hw_y : w / y = 2 / 3) :
  (x + y) / y = 3 := by sorry

end NUMINAMATH_CALUDE_ratio_sum_to_y_l2190_219001


namespace NUMINAMATH_CALUDE_tan_C_in_special_triangle_l2190_219026

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_of_angles : A + B + C = Real.pi

-- Define the theorem
theorem tan_C_in_special_triangle (t : Triangle) (h1 : Real.tan t.A = 1) (h2 : Real.tan t.B = 2) : 
  Real.tan t.C = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_C_in_special_triangle_l2190_219026


namespace NUMINAMATH_CALUDE_existence_of_close_pairs_l2190_219066

theorem existence_of_close_pairs :
  ∀ (a b : Fin 7 → ℝ),
  (∀ i, 0 ≤ a i) →
  (∀ i, 0 ≤ b i) →
  (∀ i, a i + b i ≤ 2) →
  ∃ k m, k ≠ m ∧ |a k - a m| + |b k - b m| ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_close_pairs_l2190_219066


namespace NUMINAMATH_CALUDE_soccer_campers_count_l2190_219033

/-- The number of soccer campers at a summer sports camp -/
def soccer_campers (total : ℕ) (basketball : ℕ) (football : ℕ) : ℕ :=
  total - (basketball + football)

/-- Theorem stating the number of soccer campers given the conditions -/
theorem soccer_campers_count :
  soccer_campers 88 24 32 = 32 := by
  sorry

end NUMINAMATH_CALUDE_soccer_campers_count_l2190_219033


namespace NUMINAMATH_CALUDE_three_tangent_lines_l2190_219060

/-- A line passing through (-4,1) that intersects y² = 4x at one point -/
structure TangentLine where
  /-- The slope of the line -/
  k : ℝ
  /-- The line passes through (-4,1) -/
  passes_through_point : (k * (-4) + 1) = 1
  /-- The line intersects the parabola at only one point -/
  single_intersection : (2 * k^2 - k + 1) = 0

/-- The number of distinct tangent lines -/
def count_tangent_lines : ℕ := sorry

/-- Theorem stating there are exactly 3 tangent lines -/
theorem three_tangent_lines : count_tangent_lines = 3 := by sorry

end NUMINAMATH_CALUDE_three_tangent_lines_l2190_219060


namespace NUMINAMATH_CALUDE_unit_circle_point_movement_l2190_219090

theorem unit_circle_point_movement (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 2) :
  let P₀ : ℝ × ℝ := (1, 0)
  let P₁ : ℝ × ℝ := (Real.cos α, Real.sin α)
  let P₂ : ℝ × ℝ := (Real.cos (α + Real.pi / 4), Real.sin (α + Real.pi / 4))
  P₂.1 = -3/5 → P₁.2 = 7 * Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_unit_circle_point_movement_l2190_219090


namespace NUMINAMATH_CALUDE_team_wins_l2190_219055

theorem team_wins (current_percentage : ℚ) (future_wins future_games : ℕ) 
  (new_percentage : ℚ) (h1 : current_percentage = 45/100) 
  (h2 : future_wins = 6) (h3 : future_games = 8) (h4 : new_percentage = 1/2) : 
  ∃ (total_games : ℕ) (current_wins : ℕ), 
    (current_wins : ℚ) / total_games = current_percentage ∧
    ((current_wins + future_wins) : ℚ) / (total_games + future_games) = new_percentage ∧
    current_wins = 18 := by
  sorry

end NUMINAMATH_CALUDE_team_wins_l2190_219055


namespace NUMINAMATH_CALUDE_intersection_M_N_l2190_219073

def M : Set ℤ := {-1, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = x^2}

theorem intersection_M_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2190_219073


namespace NUMINAMATH_CALUDE_more_boys_than_girls_is_two_l2190_219080

/-- The number of more boys than girls in a field day competition -/
def more_boys_than_girls : ℕ :=
  let fourth_grade_class1_girls := 12
  let fourth_grade_class1_boys := 13
  let fourth_grade_class2_girls := 15
  let fourth_grade_class2_boys := 11
  let fifth_grade_class1_girls := 9
  let fifth_grade_class1_boys := 13
  let fifth_grade_class2_girls := 10
  let fifth_grade_class2_boys := 11

  let total_girls := fourth_grade_class1_girls + fourth_grade_class2_girls +
                     fifth_grade_class1_girls + fifth_grade_class2_girls
  let total_boys := fourth_grade_class1_boys + fourth_grade_class2_boys +
                    fifth_grade_class1_boys + fifth_grade_class2_boys

  total_boys - total_girls

theorem more_boys_than_girls_is_two :
  more_boys_than_girls = 2 := by
  sorry

end NUMINAMATH_CALUDE_more_boys_than_girls_is_two_l2190_219080


namespace NUMINAMATH_CALUDE_math_test_problem_l2190_219077

theorem math_test_problem (total_questions word_problems steve_answers difference : ℕ)
  (h1 : total_questions = 45)
  (h2 : word_problems = 17)
  (h3 : steve_answers = 38)
  (h4 : difference = total_questions - steve_answers)
  (h5 : difference = 7) :
  total_questions - word_problems - steve_answers = 21 :=
by sorry

end NUMINAMATH_CALUDE_math_test_problem_l2190_219077


namespace NUMINAMATH_CALUDE_max_value_of_f_in_interval_l2190_219007

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

theorem max_value_of_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-4 : ℝ) (4 : ℝ) ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-4 : ℝ) (4 : ℝ) → f x ≤ f c ∧ f c = 10 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_in_interval_l2190_219007


namespace NUMINAMATH_CALUDE_toms_age_ratio_l2190_219071

theorem toms_age_ratio (T M : ℝ) : T > 0 → M > 0 → T - M = 3 * (T - 4 * M) → T / M = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l2190_219071


namespace NUMINAMATH_CALUDE_min_value_of_f_l2190_219053

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 4

-- State the theorem
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2190_219053


namespace NUMINAMATH_CALUDE_clothing_price_problem_l2190_219038

theorem clothing_price_problem (total_spent : ℕ) (num_pieces : ℕ) (price1 : ℕ) (price2 : ℕ) 
  (h1 : total_spent = 610)
  (h2 : num_pieces = 7)
  (h3 : price1 = 49)
  (h4 : price2 = 81)
  : (total_spent - price1 - price2) / (num_pieces - 2) = 96 := by
  sorry

end NUMINAMATH_CALUDE_clothing_price_problem_l2190_219038


namespace NUMINAMATH_CALUDE_selection_schemes_count_l2190_219084

def number_of_people : ℕ := 6
def number_of_places : ℕ := 4
def number_of_restricted_people : ℕ := 2
def number_of_restricted_places : ℕ := 1

theorem selection_schemes_count :
  (number_of_people.choose number_of_places) *
  (number_of_places - number_of_restricted_places).choose 1 *
  ((number_of_people - number_of_restricted_people).choose (number_of_places - 1)) *
  (number_of_places - 1).factorial = 240 := by
  sorry

end NUMINAMATH_CALUDE_selection_schemes_count_l2190_219084


namespace NUMINAMATH_CALUDE_leahs_garden_darker_tiles_l2190_219067

/-- Represents a square garden with a symmetrical tile pattern -/
structure SymmetricalGarden where
  -- The size of the repeating block
  block_size : ℕ
  -- The size of the center square in each block
  center_size : ℕ
  -- The number of darker tiles in the center square
  darker_tiles_in_center : ℕ

/-- The fraction of darker tiles in the garden -/
def fraction_of_darker_tiles (g : SymmetricalGarden) : ℚ :=
  (g.darker_tiles_in_center * (g.block_size / g.center_size)^2 : ℚ) / g.block_size^2

/-- Theorem stating the fraction of darker tiles in Leah's garden -/
theorem leahs_garden_darker_tiles :
  ∃ (g : SymmetricalGarden), 
    g.block_size = 4 ∧ 
    g.center_size = 2 ∧ 
    g.darker_tiles_in_center = 3 ∧ 
    fraction_of_darker_tiles g = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_leahs_garden_darker_tiles_l2190_219067


namespace NUMINAMATH_CALUDE_same_last_digit_count_l2190_219019

def has_same_last_digit (x : ℕ) : Bool :=
  x % 10 = (64 - x) % 10

def count_same_last_digit : ℕ :=
  (List.range 63).filter (λ x => has_same_last_digit (x + 1)) |>.length

theorem same_last_digit_count : count_same_last_digit = 13 := by
  sorry

end NUMINAMATH_CALUDE_same_last_digit_count_l2190_219019


namespace NUMINAMATH_CALUDE_scout_camp_chocolate_cost_l2190_219035

/-- The cost of chocolate bars for a scout camp out --/
def chocolate_cost (bar_cost : ℚ) (sections_per_bar : ℕ) (num_scouts : ℕ) (smores_per_scout : ℕ) : ℚ :=
  let total_smores := num_scouts * smores_per_scout
  let bars_needed := (total_smores + sections_per_bar - 1) / sections_per_bar
  bars_needed * bar_cost

/-- Theorem: The cost of chocolate bars for the given scout camp out is $15.00 --/
theorem scout_camp_chocolate_cost :
  chocolate_cost (3/2) 3 15 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_scout_camp_chocolate_cost_l2190_219035


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2190_219062

/-- The area of a triangle with sides 28, 26, and 10 is 130 -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    let a : ℝ := 28
    let b : ℝ := 26
    let c : ℝ := 10
    let s : ℝ := (a + b + c) / 2
    area = Real.sqrt (s * (s - a) * (s - b) * (s - c)) ∧ area = 130

/-- Proof of the theorem -/
theorem triangle_area_proof : triangle_area 130 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2190_219062


namespace NUMINAMATH_CALUDE_book_price_increase_l2190_219087

theorem book_price_increase (final_price : ℝ) (increase_percentage : ℝ) 
  (h1 : final_price = 360)
  (h2 : increase_percentage = 20) :
  let original_price := final_price / (1 + increase_percentage / 100)
  original_price = 300 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l2190_219087


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2190_219015

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 7) % 24 = 0 ∧
  (n + 7) % 36 = 0 ∧
  (n + 7) % 50 = 0 ∧
  (n + 7) % 56 = 0 ∧
  (n + 7) % 81 = 0

theorem smallest_number_divisible : 
  is_divisible_by_all 113393 ∧ 
  ∀ m : ℕ, m < 113393 → ¬is_divisible_by_all m :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l2190_219015


namespace NUMINAMATH_CALUDE_consecutive_product_theorem_l2190_219009

theorem consecutive_product_theorem (n : ℕ) : 
  (∃ m : ℕ, 9*n^2 + 5*n - 26 = m * (m + 1)) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_theorem_l2190_219009
