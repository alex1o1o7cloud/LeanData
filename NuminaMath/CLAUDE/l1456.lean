import Mathlib

namespace right_triangle_sin_value_l1456_145686

/-- Given a right triangle DEF with �angle E = 90° and 4 sin D = 5 cos D, prove that sin D = (5√41) / 41 -/
theorem right_triangle_sin_value (D E F : ℝ) (h_right_angle : E = 90) 
  (h_sin_cos_relation : 4 * Real.sin D = 5 * Real.cos D) : 
  Real.sin D = (5 * Real.sqrt 41) / 41 := by
  sorry

end right_triangle_sin_value_l1456_145686


namespace helmet_store_theorem_l1456_145601

/-- Represents the sales data for a single day -/
structure DailySales where
  helmetA : ℕ
  helmetB : ℕ
  totalAmount : ℕ

/-- Represents the helmet store problem -/
structure HelmetStore where
  day1 : DailySales
  day2 : DailySales
  costPriceA : ℕ
  costPriceB : ℕ
  totalHelmets : ℕ
  budget : ℕ
  profitGoal : ℕ

/-- The main theorem for the helmet store problem -/
theorem helmet_store_theorem (store : HelmetStore)
  (h1 : store.day1 = ⟨10, 15, 1150⟩)
  (h2 : store.day2 = ⟨6, 12, 810⟩)
  (h3 : store.costPriceA = 40)
  (h4 : store.costPriceB = 30)
  (h5 : store.totalHelmets = 100)
  (h6 : store.budget = 3400)
  (h7 : store.profitGoal = 1300) :
  ∃ (priceA priceB maxA : ℕ),
    priceA = 55 ∧
    priceB = 40 ∧
    maxA = 40 ∧
    ¬∃ (numA : ℕ), numA ≤ maxA ∧ 
      (priceA - store.costPriceA) * numA + 
      (priceB - store.costPriceB) * (store.totalHelmets - numA) ≥ store.profitGoal :=
sorry

end helmet_store_theorem_l1456_145601


namespace tetrahedron_faces_tetrahedron_has_four_faces_l1456_145631

/-- A tetrahedron is a three-dimensional geometric shape with four triangular faces. -/
structure Tetrahedron where
  -- We don't need to define the internal structure for this problem

/-- The number of faces in a tetrahedron is 4. -/
theorem tetrahedron_faces (t : Tetrahedron) : Nat :=
  4

#check tetrahedron_faces

/-- Proof that a tetrahedron has 4 faces. -/
theorem tetrahedron_has_four_faces (t : Tetrahedron) : tetrahedron_faces t = 4 := by
  sorry

end tetrahedron_faces_tetrahedron_has_four_faces_l1456_145631


namespace ralph_peanuts_l1456_145602

def initial_peanuts : ℕ := 74
def lost_peanuts : ℕ := 59

theorem ralph_peanuts : initial_peanuts - lost_peanuts = 15 := by
  sorry

end ralph_peanuts_l1456_145602


namespace num_tetrahedrons_in_cube_l1456_145604

/-- The number of vertices in a cube. -/
def cube_vertices : ℕ := 8

/-- The number of vertices required to form a tetrahedron. -/
def tetrahedron_vertices : ℕ := 4

/-- The number of coplanar combinations in a cube (faces and diagonals). -/
def coplanar_combinations : ℕ := 12

/-- Calculates the number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of tetrahedrons that can be formed using the vertices of a cube. -/
theorem num_tetrahedrons_in_cube : 
  choose cube_vertices tetrahedron_vertices - coplanar_combinations = 58 := by
  sorry

end num_tetrahedrons_in_cube_l1456_145604


namespace max_candies_eaten_l1456_145600

/-- Represents the board with numbers -/
structure Board :=
  (numbers : List Nat)

/-- Represents Karlson's candy-eating process -/
def process (b : Board) : Nat :=
  let n := b.numbers.length
  n * (n - 1) / 2

/-- The initial board with 37 ones -/
def initial_board : Board :=
  { numbers := List.replicate 37 1 }

/-- The theorem stating the maximum number of candies Karlson can eat -/
theorem max_candies_eaten (b : Board := initial_board) : 
  process b = 666 := by
  sorry

end max_candies_eaten_l1456_145600


namespace transportation_cost_optimization_l1456_145605

/-- Transportation cost optimization problem -/
theorem transportation_cost_optimization 
  (distance : ℝ) 
  (max_speed : ℝ) 
  (fixed_cost : ℝ) 
  (variable_cost_factor : ℝ) :
  distance = 1000 →
  max_speed = 80 →
  fixed_cost = 400 →
  variable_cost_factor = 1/4 →
  ∃ (optimal_speed : ℝ),
    optimal_speed > 0 ∧ 
    optimal_speed ≤ max_speed ∧
    optimal_speed = 40 ∧
    ∀ (speed : ℝ), 
      speed > 0 → 
      speed ≤ max_speed → 
      distance * (variable_cost_factor * speed + fixed_cost / speed) ≥ 
      distance * (variable_cost_factor * optimal_speed + fixed_cost / optimal_speed) :=
by sorry


end transportation_cost_optimization_l1456_145605


namespace dinner_cakes_l1456_145637

def total_cakes : ℕ := 15
def lunch_cakes : ℕ := 6

theorem dinner_cakes : total_cakes - lunch_cakes = 9 := by
  sorry

end dinner_cakes_l1456_145637


namespace tea_blend_gain_percent_l1456_145629

/-- Represents the cost and quantity of a tea variety -/
structure TeaVariety where
  cost : ℚ
  quantity : ℚ

/-- Calculates the gain percent for a tea blend -/
def gainPercent (tea1 : TeaVariety) (tea2 : TeaVariety) (sellingPrice : ℚ) : ℚ :=
  let totalCost := tea1.cost * tea1.quantity + tea2.cost * tea2.quantity
  let totalQuantity := tea1.quantity + tea2.quantity
  let costPrice := totalCost / totalQuantity
  ((sellingPrice - costPrice) / costPrice) * 100

/-- Theorem stating that the gain percent for the given tea blend is 12% -/
theorem tea_blend_gain_percent :
  let tea1 := TeaVariety.mk 18 5
  let tea2 := TeaVariety.mk 20 3
  let sellingPrice := 21
  gainPercent tea1 tea2 sellingPrice = 12 := by
  sorry

#eval gainPercent (TeaVariety.mk 18 5) (TeaVariety.mk 20 3) 21

end tea_blend_gain_percent_l1456_145629


namespace complex_trajectory_l1456_145645

theorem complex_trajectory (x y : ℝ) (z : ℂ) (h1 : x ≥ 1/2) (h2 : z = x + y * I) (h3 : Complex.abs (z - 1) = x) :
  y^2 = 2*x - 1 :=
sorry

end complex_trajectory_l1456_145645


namespace square_sum_inequality_l1456_145678

theorem square_sum_inequality (a b : ℝ) 
  (h1 : a^3 - b^3 = 2) 
  (h2 : a^5 - b^5 ≥ 4) : 
  a^2 + b^2 ≥ 2 := by
  sorry

end square_sum_inequality_l1456_145678


namespace negation_of_universal_negation_of_proposition_l1456_145664

theorem negation_of_universal (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) := by sorry

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, 2*x - 1 > 0) ↔ (∃ x : ℝ, 2*x - 1 ≤ 0) := by sorry

end negation_of_universal_negation_of_proposition_l1456_145664


namespace max_digit_sum_l1456_145626

theorem max_digit_sum (d e f z : ℕ) : 
  d ≤ 9 → e ≤ 9 → f ≤ 9 →
  (d * 100 + e * 10 + f : ℚ) / 1000 = 1 / z →
  0 < z → z ≤ 9 →
  d + e + f ≤ 8 := by
sorry

end max_digit_sum_l1456_145626


namespace work_completion_time_l1456_145615

theorem work_completion_time (x_days y_days : ℕ) (x_remaining : ℕ) (y_worked : ℕ) : 
  x_days = 24 →
  y_worked = 10 →
  x_remaining = 9 →
  (y_worked : ℚ) / y_days + (x_remaining : ℚ) / x_days = 1 →
  y_days = 16 := by
sorry

end work_completion_time_l1456_145615


namespace adams_fair_expense_l1456_145688

def fair_problem (initial_tickets : ℕ) (ferris_wheel_cost : ℕ) (roller_coaster_cost : ℕ) 
  (remaining_tickets : ℕ) (ticket_price : ℕ) (snack_price : ℕ) : Prop :=
  let used_tickets := initial_tickets - remaining_tickets
  let ride_cost := used_tickets * ticket_price
  let total_spent := ride_cost + snack_price
  total_spent = 99

theorem adams_fair_expense :
  fair_problem 13 2 3 4 9 18 := by
  sorry

end adams_fair_expense_l1456_145688


namespace union_of_A_and_B_l1456_145658

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 1 < 0}
def B : Set ℝ := {x | ∃ y, y = Real.log x}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x | x > -1} := by sorry

end union_of_A_and_B_l1456_145658


namespace box_dimensions_l1456_145685

theorem box_dimensions (x : ℕ+) : 
  (((x : ℝ) + 3) * ((x : ℝ) - 4) * ((x : ℝ)^2 + 16) < 800 ∧ 
   (x : ℝ)^2 + 16 > 30) ↔ 
  (x = 4 ∨ x = 5) :=
sorry

end box_dimensions_l1456_145685


namespace triangle_count_segment_count_l1456_145666

/-- Represents a convex polygon divided into triangles -/
structure TriangulatedPolygon where
  p : ℕ  -- number of triangles
  n : ℕ  -- number of vertices on the boundary
  m : ℕ  -- number of vertices inside

/-- The number of triangles in a triangulated polygon satisfies p = n + 2m - 2 -/
theorem triangle_count (poly : TriangulatedPolygon) :
  poly.p = poly.n + 2 * poly.m - 2 := by sorry

/-- The number of segments that are sides of the resulting triangles is 2n + 3m - 3 -/
theorem segment_count (poly : TriangulatedPolygon) :
  2 * poly.n + 3 * poly.m - 3 = poly.p + poly.n + poly.m - 1 := by sorry

end triangle_count_segment_count_l1456_145666


namespace intersection_M_N_l1456_145623

def M : Set ℝ := {x : ℝ | (x + 2) * (x - 2) > 0}

def N : Set ℝ := {-3, -2, 2, 3, 4}

theorem intersection_M_N : M ∩ N = {-3, 3, 4} := by sorry

end intersection_M_N_l1456_145623


namespace range_of_m_for_function_equality_l1456_145639

theorem range_of_m_for_function_equality (m : ℝ) : 
  (∀ x₁ ∈ (Set.Icc (-1 : ℝ) 2), ∃ x₀ ∈ (Set.Icc (-1 : ℝ) 2), 
    m * x₁ + 2 = x₀^2 - 2*x₀) → 
  m ∈ Set.Icc (-1 : ℝ) (1/2) := by
  sorry

end range_of_m_for_function_equality_l1456_145639


namespace function_minimum_implies_a_less_than_one_l1456_145676

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a

-- State the theorem
theorem function_minimum_implies_a_less_than_one :
  ∀ a : ℝ, (∃ m : ℝ, ∀ x < 1, f a x ≥ f a m) → a < 1 := by
  sorry

end function_minimum_implies_a_less_than_one_l1456_145676


namespace problem_distribution_l1456_145693

def distribute_problems (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) * n^(m - 2)

theorem problem_distribution :
  distribute_problems 12 5 = 228096 := by
  sorry

end problem_distribution_l1456_145693


namespace infinitely_many_divisors_of_2_pow_n_plus_1_l1456_145644

theorem infinitely_many_divisors_of_2_pow_n_plus_1 (m : ℕ) :
  (3 ^ m) ∣ (2 ^ (3 ^ m) + 1) :=
sorry

end infinitely_many_divisors_of_2_pow_n_plus_1_l1456_145644


namespace perpendicular_parallel_implies_perpendicular_l1456_145698

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Perpendicular relation between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between a line and a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicular relation between two lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

theorem perpendicular_parallel_implies_perpendicular 
  (l1 l2 : Line3D) (α : Plane3D) :
  perpendicular_line_plane l1 α → parallel_line_plane l2 α → 
  perpendicular_lines l1 l2 :=
sorry

end perpendicular_parallel_implies_perpendicular_l1456_145698


namespace smallest_integer_solution_l1456_145642

theorem smallest_integer_solution (x : ℤ) : 
  (∀ y : ℤ, 2*y + 5 < 3*y - 10 → y ≥ 16) ∧ (2*16 + 5 < 3*16 - 10) := by
  sorry

end smallest_integer_solution_l1456_145642


namespace board_cut_theorem_l1456_145619

theorem board_cut_theorem (total_length : ℝ) (short_length : ℝ) : 
  total_length = 6 →
  short_length + 2 * short_length = total_length →
  short_length = 2 := by
sorry

end board_cut_theorem_l1456_145619


namespace shared_bikes_theorem_l1456_145663

def a (n : ℕ+) : ℕ :=
  if n ≤ 3 then 5 * n^4 + 15 else 470 - 10 * n

def b (n : ℕ+) : ℕ := n + 5

def S (n : ℕ+) : ℕ := 8800 - 4 * (n - 46)^2

def remaining_bikes (n : ℕ+) : ℕ := 
  (Finset.range n).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩) - 
  (Finset.range n).sum (λ i => b ⟨i + 1, Nat.succ_pos i⟩)

theorem shared_bikes_theorem :
  remaining_bikes 4 = 945 ∧
  remaining_bikes 42 = 8782 ∧
  S 42 = 8736 ∧
  remaining_bikes 42 > S 42 :=
sorry

end shared_bikes_theorem_l1456_145663


namespace outlet_pipe_emptying_time_l1456_145634

/-- Given an outlet pipe that empties 1/3 of a cistern in 8 minutes,
    prove that it takes 16 minutes to empty 2/3 of the cistern. -/
theorem outlet_pipe_emptying_time
  (emptying_rate : ℝ → ℝ)
  (h1 : emptying_rate 8 = 1/3)
  (h2 : ∀ t : ℝ, emptying_rate t = (t/8) * (1/3)) :
  ∃ t : ℝ, emptying_rate t = 2/3 ∧ t = 16 :=
sorry

end outlet_pipe_emptying_time_l1456_145634


namespace cosine_product_inequality_l1456_145618

theorem cosine_product_inequality (a b c x : ℝ) :
  -(Real.sin ((b - c) / 2))^2 ≤ Real.cos (a * x + b) * Real.cos (a * x + c) ∧
  Real.cos (a * x + b) * Real.cos (a * x + c) ≤ (Real.cos ((b - c) / 2))^2 := by
  sorry

end cosine_product_inequality_l1456_145618


namespace right_triangle_with_incircle_l1456_145652

theorem right_triangle_with_incircle (r c a b : ℝ) : 
  r = 15 →  -- radius of incircle
  c = 73 →  -- hypotenuse
  r = (a + b - c) / 2 →  -- incircle radius formula
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  ((a = 55 ∧ b = 48) ∨ (a = 48 ∧ b = 55)) := by sorry

end right_triangle_with_incircle_l1456_145652


namespace apples_handed_out_to_students_l1456_145622

/-- Proves that the number of apples handed out to students is 42 -/
theorem apples_handed_out_to_students (initial_apples : ℕ) (pies : ℕ) (apples_per_pie : ℕ) 
  (h1 : initial_apples = 96)
  (h2 : pies = 9)
  (h3 : apples_per_pie = 6) : 
  initial_apples - pies * apples_per_pie = 42 := by
  sorry

#check apples_handed_out_to_students

end apples_handed_out_to_students_l1456_145622


namespace cosine_sum_theorem_l1456_145627

theorem cosine_sum_theorem : 
  12 * (Real.cos (π / 8)) ^ 4 + 
  (Real.cos (3 * π / 8)) ^ 4 + 
  (Real.cos (5 * π / 8)) ^ 4 + 
  (Real.cos (7 * π / 8)) ^ 4 = 3 / 2 := by
sorry

end cosine_sum_theorem_l1456_145627


namespace fraction_equation_solution_l1456_145630

theorem fraction_equation_solution :
  ∀ x : ℚ, (1 : ℚ) / 3 + (1 : ℚ) / 4 = 1 / x → x = 12 / 7 := by
  sorry

end fraction_equation_solution_l1456_145630


namespace manager_salary_calculation_l1456_145609

/-- The daily salary of a manager in a grocery store -/
def manager_salary : ℝ := 5

/-- The daily salary of a clerk in a grocery store -/
def clerk_salary : ℝ := 2

/-- The number of managers in the grocery store -/
def num_managers : ℕ := 2

/-- The number of clerks in the grocery store -/
def num_clerks : ℕ := 3

/-- The total daily salary of all employees in the grocery store -/
def total_salary : ℝ := 16

theorem manager_salary_calculation :
  manager_salary * num_managers + clerk_salary * num_clerks = total_salary :=
by sorry

end manager_salary_calculation_l1456_145609


namespace mel_age_is_21_l1456_145638

/-- Katherine's age in years -/
def katherine_age : ℕ := 24

/-- The age difference between Katherine and Mel in years -/
def age_difference : ℕ := 3

/-- Mel's age in years -/
def mel_age : ℕ := katherine_age - age_difference

theorem mel_age_is_21 : mel_age = 21 := by
  sorry

end mel_age_is_21_l1456_145638


namespace washington_party_handshakes_l1456_145681

/-- Represents a party with married couples -/
structure Party where
  couples : Nat
  men : Nat
  women : Nat

/-- Calculates the number of handshakes in the party -/
def handshakes (p : Party) : Nat :=
  -- Handshakes among men
  (p.men.choose 2) +
  -- Handshakes between men and women (excluding spouses)
  p.men * (p.women - 1)

/-- Theorem stating the number of handshakes at George Washington's party -/
theorem washington_party_handshakes :
  ∃ (p : Party),
    p.couples = 13 ∧
    p.men = p.couples ∧
    p.women = p.couples ∧
    handshakes p = 234 := by
  sorry

end washington_party_handshakes_l1456_145681


namespace proof_by_contradiction_assumption_l1456_145665

theorem proof_by_contradiction_assumption (a b : ℝ) : 
  (a ≤ b → False) → a > b :=
sorry

end proof_by_contradiction_assumption_l1456_145665


namespace trampoline_jumps_l1456_145655

theorem trampoline_jumps (ronald_jumps rupert_extra_jumps : ℕ) 
  (h1 : ronald_jumps = 157)
  (h2 : rupert_extra_jumps = 86) : 
  ronald_jumps + (ronald_jumps + rupert_extra_jumps) = 400 := by
  sorry

end trampoline_jumps_l1456_145655


namespace problem_statement_l1456_145661

theorem problem_statement :
  -- Part 1
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 → 1/a + 1/b ≥ 4) ∧
  -- Part 2
  (∃ min : ℝ, min = 1/14 ∧
    ∀ x y z : ℝ, x + 2*y + 3*z = 1 → x^2 + y^2 + z^2 ≥ min ∧
    ∃ x₀ y₀ z₀ : ℝ, x₀ + 2*y₀ + 3*z₀ = 1 ∧ x₀^2 + y₀^2 + z₀^2 = min) :=
by sorry

end problem_statement_l1456_145661


namespace johns_shower_duration_l1456_145628

/-- Proves that John's shower duration is 10 minutes given the conditions --/
theorem johns_shower_duration :
  let days_in_four_weeks : ℕ := 28
  let shower_frequency : ℕ := 2  -- every other day
  let water_usage_per_minute : ℚ := 2  -- gallons per minute
  let total_water_usage : ℚ := 280  -- gallons in 4 weeks
  
  let num_showers : ℕ := days_in_four_weeks / shower_frequency
  let water_per_shower : ℚ := total_water_usage / num_showers
  let shower_duration : ℚ := water_per_shower / water_usage_per_minute
  
  shower_duration = 10 := by sorry

end johns_shower_duration_l1456_145628


namespace equation_represents_hyperbola_l1456_145632

/-- The equation (x+y)^2 = x^2 + y^2 + 1 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b : ℝ) (k : ℝ), k ≠ 0 ∧
  (∀ x y : ℝ, (x + y)^2 = x^2 + y^2 + 1 ↔ (x * y = k ∧ (x / a)^2 - (y / b)^2 = 1)) :=
by sorry

end equation_represents_hyperbola_l1456_145632


namespace marilyn_shared_bottle_caps_l1456_145669

/-- The number of bottle caps Marilyn shared with Nancy -/
def shared_bottle_caps (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem marilyn_shared_bottle_caps :
  shared_bottle_caps 51 15 = 36 :=
by sorry

end marilyn_shared_bottle_caps_l1456_145669


namespace parabola_point_distance_l1456_145656

theorem parabola_point_distance (x y : ℝ) : 
  x^2 = 4*y →                             -- P is on the parabola x^2 = 4y
  (x^2 + (y - 1)^2 = 4) →                 -- Distance from P to A(0,1) is 2
  y = 1 :=                                -- Distance from P to x-axis is 1
by sorry

end parabola_point_distance_l1456_145656


namespace solution_set_of_inequality_l1456_145621

theorem solution_set_of_inequality (x : ℝ) :
  (x - 3) / (x + 2) < 0 ↔ -2 < x ∧ x < 3 := by sorry

end solution_set_of_inequality_l1456_145621


namespace regular_polygon_150_deg_interior_has_12_sides_l1456_145608

/-- A regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_150_deg_interior_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ angle : ℝ, angle = 150 → n * angle = (n - 2) * 180) →
  n = 12 := by
sorry

end regular_polygon_150_deg_interior_has_12_sides_l1456_145608


namespace f_6_l1456_145624

/-- A function satisfying f(x) = f(x - 2) + 3 for all real x, with f(2) = 4 -/
def f : ℝ → ℝ :=
  sorry

/-- The functional equation for f -/
axiom f_eq (x : ℝ) : f x = f (x - 2) + 3

/-- The initial condition for f -/
axiom f_2 : f 2 = 4

/-- Theorem: f(6) = 10 -/
theorem f_6 : f 6 = 10 := by
  sorry

end f_6_l1456_145624


namespace probability_same_color_l1456_145699

/-- The number of marbles of each color -/
def marbles_per_color : ℕ := 3

/-- The number of colors -/
def num_colors : ℕ := 3

/-- The total number of marbles -/
def total_marbles : ℕ := marbles_per_color * num_colors

/-- The number of marbles Cheryl picks -/
def picked_marbles : ℕ := 3

/-- The probability of picking 3 marbles of the same color -/
theorem probability_same_color :
  (num_colors * Nat.choose marbles_per_color picked_marbles * 
   Nat.choose (total_marbles - picked_marbles) (total_marbles - 2 * picked_marbles)) /
  Nat.choose total_marbles picked_marbles = 1 / 28 := by sorry

end probability_same_color_l1456_145699


namespace existence_of_odd_powers_sum_l1456_145614

theorem existence_of_odd_powers_sum (m : ℤ) :
  ∃ (a b k : ℤ), 
    Odd a ∧ 
    Odd b ∧ 
    k > 0 ∧ 
    2 * m = a^19 + b^99 + k * 2^1999 := by
  sorry

end existence_of_odd_powers_sum_l1456_145614


namespace set_S_satisfies_conditions_l1456_145653

def S : Finset Nat := {2, 3, 11, 23, 31}

def P : Nat := S.prod id

theorem set_S_satisfies_conditions :
  (∀ x ∈ S, x > 1) ∧
  (∀ x ∈ S, x ∣ (P / x + 1) ∧ x ≠ (P / x + 1)) := by
  sorry

end set_S_satisfies_conditions_l1456_145653


namespace first_company_fixed_cost_l1456_145651

/-- The fixed amount charged by the first rental company -/
def F : ℝ := 38.95

/-- The cost per mile for the first rental company -/
def cost_per_mile_first : ℝ := 0.31

/-- The fixed amount charged by Safety Rent A Truck -/
def fixed_cost_safety : ℝ := 41.95

/-- The cost per mile for Safety Rent A Truck -/
def cost_per_mile_safety : ℝ := 0.29

/-- The number of miles driven -/
def miles : ℝ := 150.0

theorem first_company_fixed_cost :
  F + cost_per_mile_first * miles = fixed_cost_safety + cost_per_mile_safety * miles :=
by sorry

end first_company_fixed_cost_l1456_145651


namespace evaluate_expression_l1456_145613

theorem evaluate_expression : -25 - 7 * (4 + 2) = -67 := by
  sorry

end evaluate_expression_l1456_145613


namespace fraction_division_proof_l1456_145649

theorem fraction_division_proof : (5 / 4) / (8 / 15) = 75 / 32 := by
  sorry

end fraction_division_proof_l1456_145649


namespace vector_equation_and_parallel_condition_l1456_145682

/-- Vector in R² -/
structure Vec2 where
  x : ℝ
  y : ℝ

/-- Scalar multiplication for Vec2 -/
def scalarMul (r : ℝ) (v : Vec2) : Vec2 :=
  ⟨r * v.x, r * v.y⟩

/-- Addition for Vec2 -/
def add (v w : Vec2) : Vec2 :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Two Vec2 are parallel if their cross product is zero -/
def isParallel (v w : Vec2) : Prop :=
  v.x * w.y = v.y * w.x

theorem vector_equation_and_parallel_condition :
  let a : Vec2 := ⟨3, 2⟩
  let b : Vec2 := ⟨-1, 2⟩
  let c : Vec2 := ⟨4, 1⟩
  
  /- Part 1: Vector equation -/
  (a = add (scalarMul (5/9) b) (scalarMul (8/9) c)) ∧
  
  /- Part 2: Parallel condition -/
  (isParallel (add a (scalarMul (-16/13) c)) (add (scalarMul 2 b) (scalarMul (-1) a))) :=
by
  sorry

end vector_equation_and_parallel_condition_l1456_145682


namespace cos_sin_cos_bounds_l1456_145620

theorem cos_sin_cos_bounds (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ (2 + Real.sqrt 3) / 8 := by
  sorry

end cos_sin_cos_bounds_l1456_145620


namespace female_employees_percentage_l1456_145633

/-- The percentage of female employees in an office -/
def percentage_female_employees (total_employees : ℕ) 
  (percent_computer_literate : ℚ) 
  (female_computer_literate : ℕ) 
  (percent_male_computer_literate : ℚ) : ℚ :=
  sorry

/-- Theorem stating the percentage of female employees is 60% -/
theorem female_employees_percentage 
  (h1 : total_employees = 1500)
  (h2 : percent_computer_literate = 62 / 100)
  (h3 : female_computer_literate = 630)
  (h4 : percent_male_computer_literate = 1 / 2) :
  percentage_female_employees total_employees percent_computer_literate 
    female_computer_literate percent_male_computer_literate = 60 / 100 :=
by sorry

end female_employees_percentage_l1456_145633


namespace johns_growth_per_month_l1456_145650

/-- Proves that John's growth per month is 2 inches given his original height, new height, and growth period. -/
theorem johns_growth_per_month 
  (original_height : ℕ) 
  (new_height_feet : ℕ) 
  (growth_period : ℕ) 
  (h1 : original_height = 66)
  (h2 : new_height_feet = 6)
  (h3 : growth_period = 3) :
  (new_height_feet * 12 - original_height) / growth_period = 2 := by
  sorry

#check johns_growth_per_month

end johns_growth_per_month_l1456_145650


namespace total_groom_time_in_minutes_l1456_145660

/-- The time in hours it takes to groom a dog -/
def dog_groom_time : ℝ := 2.5

/-- The time in hours it takes to groom a cat -/
def cat_groom_time : ℝ := 0.5

/-- The number of dogs to be groomed -/
def num_dogs : ℕ := 5

/-- The number of cats to be groomed -/
def num_cats : ℕ := 3

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem stating that the total time to groom 5 dogs and 3 cats is 840 minutes -/
theorem total_groom_time_in_minutes : 
  (dog_groom_time * num_dogs + cat_groom_time * num_cats) * minutes_per_hour = 840 := by
  sorry

end total_groom_time_in_minutes_l1456_145660


namespace coefficient_x3_in_2x_plus_1_power_5_l1456_145657

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (2x+1)^5
def coefficient_x3 : ℕ := binomial 5 2 * 2^3

-- Theorem statement
theorem coefficient_x3_in_2x_plus_1_power_5 : coefficient_x3 = 80 := by sorry

end coefficient_x3_in_2x_plus_1_power_5_l1456_145657


namespace plum_jelly_sales_l1456_145667

/-- Represents the number of jars sold for each type of jelly -/
structure JellySales where
  grape : ℕ
  strawberry : ℕ
  raspberry : ℕ
  plum : ℕ

/-- Defines the conditions for jelly sales -/
def validJellySales (sales : JellySales) : Prop :=
  sales.grape = 2 * sales.strawberry ∧
  sales.raspberry = 2 * sales.plum ∧
  sales.raspberry = sales.grape / 3 ∧
  sales.strawberry = 18

/-- Theorem stating that given the conditions, 6 jars of plum jelly were sold -/
theorem plum_jelly_sales (sales : JellySales) (h : validJellySales sales) : sales.plum = 6 := by
  sorry

end plum_jelly_sales_l1456_145667


namespace kaleb_restaurant_bill_l1456_145640

/-- Calculates the total bill for a group at Kaleb's Restaurant -/
def total_bill (num_adults : ℕ) (num_children : ℕ) (adult_meal_cost : ℕ) (child_meal_cost : ℕ) (soda_cost : ℕ) : ℕ :=
  num_adults * adult_meal_cost + num_children * child_meal_cost + (num_adults + num_children) * soda_cost

/-- Theorem: The total bill for a group of 6 adults and 2 children at Kaleb's Restaurant is $60 -/
theorem kaleb_restaurant_bill :
  total_bill 6 2 6 4 2 = 60 := by
  sorry

end kaleb_restaurant_bill_l1456_145640


namespace bubble_pass_probability_specific_l1456_145694

/-- The probability that in a sequence of n distinct terms,
    the kth term ends up in the mth position after one bubble pass -/
def bubble_pass_probability (n k m : ℕ) : ℚ :=
  if k ≤ m ∧ m < n then
    (1 : ℚ) / k * (1 : ℚ) / (m - k + 1) * (1 : ℚ) / (n - m)
  else 0

/-- The main theorem stating the probability for the specific case -/
theorem bubble_pass_probability_specific :
  bubble_pass_probability 50 20 40 = 1 / 4000 := by
  sorry

#eval bubble_pass_probability 50 20 40

end bubble_pass_probability_specific_l1456_145694


namespace intersection_M_N_l1456_145662

-- Define the sets M and N
def M : Set ℝ := {x | 1 + x ≥ 0}
def N : Set ℝ := {x | (4 : ℝ) / (1 - x) > 0}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end intersection_M_N_l1456_145662


namespace range_of_a_l1456_145689

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 * x^2 - 3 * x + 1 ≤ 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : (A ∩ (Set.compl (B a)) = ∅) → (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end range_of_a_l1456_145689


namespace transistors_2010_l1456_145672

/-- Moore's law: Number of transistors doubles every 18 months -/
def moores_law_doubling_period : ℕ := 18

/-- Number of transistors in 1995 -/
def transistors_1995 : ℕ := 2500000

/-- Calculate the number of transistors after a given number of months -/
def transistors_after (initial_transistors : ℕ) (months : ℕ) : ℕ :=
  initial_transistors * 2^(months / moores_law_doubling_period)

/-- Theorem: Number of transistors in 2010 according to Moore's law -/
theorem transistors_2010 :
  transistors_after transistors_1995 ((2010 - 1995) * 12) = 2560000000 := by
  sorry

end transistors_2010_l1456_145672


namespace smallest_result_l1456_145671

def S : Set Nat := {2, 3, 5, 7, 11, 13}

theorem smallest_result (a b c : Nat) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (∀ x y z : Nat, x ∈ S → y ∈ S → z ∈ S → x ≠ y ∧ y ≠ z ∧ x ≠ z → 
    22 ≤ (x + x + y) * z) ∧ (∃ x y z : Nat, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (x + x + y) * z = 22) := by
  sorry

end smallest_result_l1456_145671


namespace sons_age_l1456_145692

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 30 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 28 := by
sorry

end sons_age_l1456_145692


namespace arithmetic_sequence_general_term_l1456_145648

/-- An arithmetic sequence satisfying given conditions has one of two specific general terms -/
theorem arithmetic_sequence_general_term 
  (a : ℕ → ℚ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : a 2 + a 7 + a 12 = 12) 
  (h_product : a 2 * a 7 * a 12 = 28) :
  (∃ C : ℚ, ∀ n : ℕ, a n = 3/5 * n - 1/5 + C) ∨ 
  (∃ C : ℚ, ∀ n : ℕ, a n = -3/5 * n + 41/5 + C) :=
sorry

end arithmetic_sequence_general_term_l1456_145648


namespace min_value_z_l1456_145674

theorem min_value_z (x y : ℝ) : 3 * x^2 + 5 * y^2 + 6 * x - 4 * y + 3 * x^3 + 15 ≥ 8.2 := by
  sorry

end min_value_z_l1456_145674


namespace seating_arrangements_l1456_145646

def total_people : ℕ := 12
def people_per_table : ℕ := 6
def num_tables : ℕ := 2

def arrange_people (n : ℕ) (k : ℕ) : ℕ := (n.factorial * 14400) / (k.factorial * k.factorial)

def arrange_couples (n : ℕ) (k : ℕ) : ℕ := (n.factorial * 14400 * 4096) / (k.factorial * k.factorial)

theorem seating_arrangements :
  (arrange_people total_people people_per_table = (total_people.factorial * 14400) / (people_per_table.factorial * people_per_table.factorial)) ∧
  (arrange_couples total_people people_per_table = (total_people.factorial * 14400 * 4096) / (people_per_table.factorial * people_per_table.factorial)) :=
by sorry

end seating_arrangements_l1456_145646


namespace tangent_slope_angle_l1456_145691

/-- The slope angle of the tangent line to y = x^3 forming an isosceles triangle -/
theorem tangent_slope_angle (x₀ : ℝ) : 
  let B : ℝ × ℝ := (x₀, x₀^3)
  let slope : ℝ := 3 * x₀^2
  let A : ℝ × ℝ := ((2/3) * x₀, 0)
  (x₀^4 = 1/3) →  -- This ensures OAB is isosceles
  (slope = Real.sqrt 3) →
  Real.arctan slope = π/3 :=
by sorry

end tangent_slope_angle_l1456_145691


namespace agnes_current_age_l1456_145610

/-- Agnes's current age -/
def agnes_age : ℕ := 25

/-- Jane's current age -/
def jane_age : ℕ := 6

/-- Years into the future when Agnes will be twice Jane's age -/
def years_future : ℕ := 13

theorem agnes_current_age :
  agnes_age = 25 ∧
  jane_age = 6 ∧
  agnes_age + years_future = 2 * (jane_age + years_future) :=
by sorry

end agnes_current_age_l1456_145610


namespace count_non_divisible_is_30_l1456_145687

/-- g(n) is the product of the proper positive integer divisors of n -/
def g (n : ℕ) : ℕ := sorry

/-- Counts the numbers n between 2 and 100 (inclusive) for which n does not divide g(n) -/
def count_non_divisible : ℕ := sorry

theorem count_non_divisible_is_30 : count_non_divisible = 30 := by sorry

end count_non_divisible_is_30_l1456_145687


namespace problem_solution_l1456_145684

theorem problem_solution : 
  (1/2 - 2/3 - 3/4) * 12 = -11 ∧ 
  -(1^6) + |-2/3| - (1 - 5/9) + 2/3 = -1/9 := by sorry

end problem_solution_l1456_145684


namespace sin_2alpha_plus_2pi_3_l1456_145603

/-- Given an angle α in a Cartesian coordinate system with its vertex at the origin,
    its initial side on the non-negative x-axis, and its terminal side passing through (-1, 2),
    prove that sin(2α + 2π/3) = (4 - 3√3) / 10 -/
theorem sin_2alpha_plus_2pi_3 (α : Real) :
  (∃ (x y : Real), x = -1 ∧ y = 2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin (2 * α + 2 * Real.pi / 3) = (4 - 3 * Real.sqrt 3) / 10 := by
  sorry

end sin_2alpha_plus_2pi_3_l1456_145603


namespace range_of_y₂_l1456_145617

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define line l₁
def l₁ (x : ℝ) : Prop := x = -1

-- Define line l₂
def l₂ (y t : ℝ) : Prop := y = t

-- Define point P
def P (t : ℝ) : ℝ × ℝ := (-1, t)

-- Define curve C₂
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

-- Define points A, B, and C on C₂
def A : ℝ × ℝ := (1, 2)
def B (x₁ y₁ : ℝ) : Prop := C₂ x₁ y₁ ∧ (x₁, y₁) ≠ A
def C (x₂ y₂ : ℝ) : Prop := C₂ x₂ y₂ ∧ (x₂, y₂) ≠ A

-- AB perpendicular to BC
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - 1) * (x₂ - x₁) + (y₁ - 2) * (y₂ - y₁) = 0

-- Theorem statement
theorem range_of_y₂ (x₁ y₁ x₂ y₂ : ℝ) :
  B x₁ y₁ → C x₂ y₂ → perpendicular x₁ y₁ x₂ y₂ →
  y₂ ∈ (Set.Iic (-6) \ {-6}) ∪ Set.Ici 10 :=
sorry

end range_of_y₂_l1456_145617


namespace ten_machines_four_minutes_l1456_145635

/-- The number of bottles produced by a given number of machines in a given time -/
def bottles_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  let bottles_per_minute := (270 * machines) / 5
  bottles_per_minute * minutes

/-- Theorem stating that 10 machines produce 2160 bottles in 4 minutes -/
theorem ten_machines_four_minutes :
  bottles_produced 10 4 = 2160 := by
  sorry

end ten_machines_four_minutes_l1456_145635


namespace total_time_outside_class_l1456_145680

def recess_break_1 : ℕ := 15
def recess_break_2 : ℕ := 15
def lunch_break : ℕ := 30
def additional_recess : ℕ := 20

theorem total_time_outside_class : 
  2 * recess_break_1 + lunch_break + additional_recess = 80 := by
  sorry

end total_time_outside_class_l1456_145680


namespace dog_escape_ways_l1456_145607

def base_7_to_10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₀ * 7^0 + d₁ * 7^1 + d₂ * 7^2

theorem dog_escape_ways : base_7_to_10 2 3 1 = 120 := by
  sorry

end dog_escape_ways_l1456_145607


namespace even_function_property_l1456_145641

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of being an even function
def isEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Theorem statement
theorem even_function_property 
  (h1 : isEven f) 
  (h2 : ∀ x ∈ Set.Icc (-5 : ℝ) 5, ∃ y, f x = y)
  (h3 : f 3 > f 1) : 
  f (-1) < f 3 := by
  sorry

end even_function_property_l1456_145641


namespace inequality_proof_l1456_145625

theorem inequality_proof (m n : ℕ) (h : m < Real.sqrt 2 * n) :
  (m : ℝ) / n < Real.sqrt 2 * (1 - 1 / (4 * n^2)) := by
  sorry

end inequality_proof_l1456_145625


namespace questions_per_exam_l1456_145679

theorem questions_per_exam
  (num_classes : ℕ)
  (students_per_class : ℕ)
  (total_questions : ℕ)
  (h1 : num_classes = 5)
  (h2 : students_per_class = 35)
  (h3 : total_questions = 1750) :
  total_questions / (num_classes * students_per_class) = 10 := by
sorry

end questions_per_exam_l1456_145679


namespace imaginary_part_of_complex_fraction_l1456_145683

theorem imaginary_part_of_complex_fraction (a : ℝ) : 
  Complex.im ((1 + a * Complex.I) / Complex.I) = -1 := by
  sorry

end imaginary_part_of_complex_fraction_l1456_145683


namespace sugar_solution_replacement_l1456_145606

theorem sugar_solution_replacement (W : ℝ) (x : ℝ) : 
  (W > 0) → 
  (0 ≤ x) → (x ≤ 1) →
  ((1 - x) * (0.22 * W) + x * (0.74 * W) = 0.35 * W) ↔ 
  (x = 1/4) :=
by sorry

end sugar_solution_replacement_l1456_145606


namespace mango_rice_flour_cost_l1456_145670

/-- Given the cost relationships between mangos, rice, and flour, 
    prove that the total cost of 4 kg of mangos, 3 kg of rice, and 5 kg of flour is $1027.2 -/
theorem mango_rice_flour_cost 
  (mango_cost rice_cost flour_cost : ℝ) 
  (h1 : 10 * mango_cost = 24 * rice_cost) 
  (h2 : 6 * flour_cost = 2 * rice_cost) 
  (h3 : flour_cost = 24) : 
  4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 1027.2 := by
sorry

end mango_rice_flour_cost_l1456_145670


namespace problem_statement_l1456_145697

theorem problem_statement (a b : ℕ+) :
  (18 ^ a.val) * (9 ^ (3 * a.val - 1)) = (2 ^ 6) * (3 ^ b.val) →
  a.val = 6 := by
sorry

end problem_statement_l1456_145697


namespace constant_term_binomial_expansion_l1456_145659

theorem constant_term_binomial_expansion (x : ℝ) : 
  (∃ c : ℝ, c = 1120 ∧ 
   ∃ f : ℝ → ℝ, 
   (∀ y, f y = (y - 2/y)^8) ∧
   (∃ g : ℝ → ℝ, (∀ y, f y = g y + c + y * (g y)))) :=
by sorry

end constant_term_binomial_expansion_l1456_145659


namespace condition_necessary_not_sufficient_l1456_145690

theorem condition_necessary_not_sufficient (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b, a > b ∧ b > 0 → 1 / a < 1 / b) ∧
  (∃ a b, 1 / a < 1 / b ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end condition_necessary_not_sufficient_l1456_145690


namespace range_of_a_l1456_145695

theorem range_of_a (x y a : ℝ) (h1 : x > y) (h2 : (a + 3) * x < (a + 3) * y) : a < -3 := by
  sorry

end range_of_a_l1456_145695


namespace intersection_when_a_is_3_intersection_equals_A_iff_l1456_145675

def A (a : ℝ) := { x : ℝ | a ≤ x ∧ x ≤ a + 3 }
def B := { x : ℝ | x < -1 ∨ x > 5 }

theorem intersection_when_a_is_3 :
  A 3 ∩ B = { x : ℝ | 5 < x ∧ x ≤ 6 } := by sorry

theorem intersection_equals_A_iff (a : ℝ) :
  A a ∩ B = A a ↔ a < -4 ∨ a > 5 := by sorry

end intersection_when_a_is_3_intersection_equals_A_iff_l1456_145675


namespace minor_arc_probability_l1456_145668

/-- The probability that the length of the minor arc is less than 1 on a circle
    with circumference 3, given a fixed point A and a randomly selected point B. -/
theorem minor_arc_probability (circle_circumference : ℝ) (arc_length : ℝ) :
  circle_circumference = 3 →
  arc_length = 1 →
  (2 * arc_length) / circle_circumference = 2/3 := by
  sorry

end minor_arc_probability_l1456_145668


namespace eliza_almonds_l1456_145654

theorem eliza_almonds (eliza_almonds daniel_almonds : ℕ) : 
  eliza_almonds = daniel_almonds + 8 →
  daniel_almonds = eliza_almonds / 3 →
  eliza_almonds = 12 := by
sorry

end eliza_almonds_l1456_145654


namespace average_of_abc_l1456_145647

theorem average_of_abc (A B C : ℚ) 
  (eq1 : 2002 * C - 3003 * A = 6006)
  (eq2 : 2002 * B + 4004 * A = 8008)
  (eq3 : B - C = A + 1) :
  (A + B + C) / 3 = 7 / 3 := by
  sorry

end average_of_abc_l1456_145647


namespace correct_average_after_errors_l1456_145673

theorem correct_average_after_errors (n : ℕ) (initial_avg : ℚ) 
  (error1 : ℚ) (error2 : ℚ) (error3 : ℚ) : 
  n = 15 → 
  initial_avg = 24 → 
  error1 = 65 - 45 → 
  error2 = 42 - 28 → 
  error3 = 75 - 55 → 
  (n : ℚ) * initial_avg + error1 + error2 + error3 = n * (27.6 : ℚ) := by
  sorry

end correct_average_after_errors_l1456_145673


namespace pencils_remaining_l1456_145677

theorem pencils_remaining (x : ℕ) : ℕ :=
  let initial_pencils_per_child : ℕ := 2
  let number_of_children : ℕ := 15
  let total_initial_pencils : ℕ := initial_pencils_per_child * number_of_children
  let pencils_given_away : ℕ := number_of_children * x
  total_initial_pencils - pencils_given_away

#check pencils_remaining

end pencils_remaining_l1456_145677


namespace smallest_x_value_l1456_145616

theorem smallest_x_value (x : ℝ) : 
  ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20 → x ≥ 0 :=
by sorry

end smallest_x_value_l1456_145616


namespace repeating_decimal_fraction_sum_l1456_145611

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n.gcd d = 1) ∧ 
  (n : ℚ) / (d : ℚ) = 7 + 47 / 99 ∧ 
  n + d = 839 := by
  sorry

end repeating_decimal_fraction_sum_l1456_145611


namespace min_transportation_cost_l1456_145643

/-- Represents the transportation problem between cities A, B, C, and D. -/
structure TransportationProblem where
  inventory_A : ℕ := 12
  inventory_B : ℕ := 8
  demand_C : ℕ := 10
  demand_D : ℕ := 10
  cost_A_to_C : ℕ := 300
  cost_A_to_D : ℕ := 500
  cost_B_to_C : ℕ := 400
  cost_B_to_D : ℕ := 800

/-- The total cost function for the transportation problem. -/
def total_cost (tp : TransportationProblem) (x : ℕ) : ℕ :=
  200 * x + 8400

/-- The theorem stating that the minimum total transportation cost is 8800 yuan. -/
theorem min_transportation_cost (tp : TransportationProblem) :
  ∃ (x : ℕ), 2 ≤ x ∧ x ≤ 10 ∧ (∀ (y : ℕ), 2 ≤ y ∧ y ≤ 10 → total_cost tp x ≤ total_cost tp y) ∧
  total_cost tp x = 8800 :=
sorry

#check min_transportation_cost

end min_transportation_cost_l1456_145643


namespace consecutive_powers_divisibility_l1456_145636

theorem consecutive_powers_divisibility (a : ℝ) (n : ℕ) :
  ∃ k : ℤ, a^n + a^(n+1) = k * a * (a + 1) := by sorry

end consecutive_powers_divisibility_l1456_145636


namespace john_walks_to_school_l1456_145612

/-- The distance Nina walks to school in miles -/
def nina_distance : ℝ := 0.4

/-- The additional distance John walks compared to Nina in miles -/
def additional_distance : ℝ := 0.3

/-- John's distance to school in miles -/
def john_distance : ℝ := nina_distance + additional_distance

/-- Theorem stating that John walks 0.7 miles to school -/
theorem john_walks_to_school : john_distance = 0.7 := by
  sorry

end john_walks_to_school_l1456_145612


namespace arithmetic_sequence_common_difference_l1456_145696

/-- An arithmetic sequence with 10 terms where the sum of odd-numbered terms is 15
    and the sum of even-numbered terms is 30 has a common difference of 3. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) -- The arithmetic sequence
  (h1 : a 1 + a 3 + a 5 + a 7 + a 9 = 15) -- Sum of odd-numbered terms
  (h2 : a 2 + a 4 + a 6 + a 8 + a 10 = 30) -- Sum of even-numbered terms
  (h3 : ∀ n : ℕ, n < 10 → a (n + 1) - a n = a 2 - a 1) -- Definition of arithmetic sequence
  : a 2 - a 1 = 3 := by
  sorry

end arithmetic_sequence_common_difference_l1456_145696
