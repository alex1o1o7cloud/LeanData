import Mathlib

namespace NUMINAMATH_CALUDE_point_coordinates_l3738_373811

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance to x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance to y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem statement -/
theorem point_coordinates (P : Point) 
  (h1 : isInSecondQuadrant P) 
  (h2 : distanceToXAxis P = 7) 
  (h3 : distanceToYAxis P = 3) : 
  P.x = -3 ∧ P.y = 7 := by
  sorry


end NUMINAMATH_CALUDE_point_coordinates_l3738_373811


namespace NUMINAMATH_CALUDE_two_books_different_subjects_l3738_373839

theorem two_books_different_subjects (math_books : ℕ) (chinese_books : ℕ) (english_books : ℕ) :
  math_books = 10 → chinese_books = 9 → english_books = 8 →
  (math_books * chinese_books) + (math_books * english_books) + (chinese_books * english_books) = 242 :=
by sorry

end NUMINAMATH_CALUDE_two_books_different_subjects_l3738_373839


namespace NUMINAMATH_CALUDE_product_evaluation_l3738_373889

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3738_373889


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_fourth_l3738_373858

theorem cos_alpha_plus_pi_fourth (α β : Real) : 
  (3 * Real.pi / 4 < α) ∧ (α < Real.pi) ∧
  (3 * Real.pi / 4 < β) ∧ (β < Real.pi) ∧
  (Real.sin (α + β) = -4/5) ∧
  (Real.sin (β - Real.pi/4) = 12/13) →
  Real.cos (α + Real.pi/4) = -63/65 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_fourth_l3738_373858


namespace NUMINAMATH_CALUDE_world_cup_tickets_system_correct_l3738_373898

/-- Represents the World Cup ticket reservation problem -/
structure WorldCupTickets where
  x : ℕ  -- number of group stage tickets
  y : ℕ  -- number of final tickets
  total_tickets : x + y = 20
  total_price : 2800 * x + 6400 * y = 74000

/-- The system of equations correctly represents the World Cup ticket situation -/
theorem world_cup_tickets_system_correct (tickets : WorldCupTickets) :
  (tickets.x + tickets.y = 20) ∧ (2800 * tickets.x + 6400 * tickets.y = 74000) :=
by sorry

end NUMINAMATH_CALUDE_world_cup_tickets_system_correct_l3738_373898


namespace NUMINAMATH_CALUDE_train_length_calculation_second_train_length_l3738_373830

/-- Calculates the length of the second train given the conditions of the problem -/
theorem train_length_calculation (length1 : ℝ) (speed1 speed2 : ℝ) (crossing_time : ℝ) : ℝ :=
  let km_per_hr_to_m_per_s : ℝ := 1000 / 3600
  let speed1_m_per_s : ℝ := speed1 * km_per_hr_to_m_per_s
  let speed2_m_per_s : ℝ := speed2 * km_per_hr_to_m_per_s
  let relative_speed : ℝ := speed1_m_per_s + speed2_m_per_s
  let total_distance : ℝ := relative_speed * crossing_time
  total_distance - length1

/-- The length of the second train is approximately 160 meters -/
theorem second_train_length :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |train_length_calculation 140 60 40 10.799136069114471 - 160| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_calculation_second_train_length_l3738_373830


namespace NUMINAMATH_CALUDE_vector_projection_on_x_axis_l3738_373822

theorem vector_projection_on_x_axis (a : ℝ) (φ : ℝ) :
  a = 5 →
  φ = Real.pi / 3 →
  a * Real.cos φ = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_on_x_axis_l3738_373822


namespace NUMINAMATH_CALUDE_domino_grid_side_divisible_by_four_l3738_373866

/-- A rectangular grid that can be cut into 1x2 dominoes with the property that any straight line
    along the grid lines intersects a multiple of four dominoes. -/
structure DominoCoveredGrid where
  a : ℕ  -- length of the grid
  b : ℕ  -- width of the grid
  is_valid : (a * b) % 2 = 0  -- ensures the grid can be covered by 1x2 dominoes
  line_cuts_multiple_of_four : ∀ (line : ℕ), line ≤ a ∨ line ≤ b → (line * 2) % 4 = 0

/-- If a rectangular grid can be covered by 1x2 dominoes such that any straight line along
    the grid lines intersects a multiple of four dominoes, then one of its sides is divisible by 4. -/
theorem domino_grid_side_divisible_by_four (grid : DominoCoveredGrid) :
  4 ∣ grid.a ∨ 4 ∣ grid.b :=
sorry

end NUMINAMATH_CALUDE_domino_grid_side_divisible_by_four_l3738_373866


namespace NUMINAMATH_CALUDE_work_earnings_theorem_l3738_373804

/-- Given the following conditions:
  - I worked t+2 hours
  - I earned 4t-4 dollars per hour
  - Bob worked 4t-6 hours
  - Bob earned t+3 dollars per hour
  - I earned three dollars more than Bob
Prove that t = 7/2 -/
theorem work_earnings_theorem (t : ℚ) : 
  (t + 2) * (4 * t - 4) = (4 * t - 6) * (t + 3) + 3 → t = 7/2 := by
sorry

end NUMINAMATH_CALUDE_work_earnings_theorem_l3738_373804


namespace NUMINAMATH_CALUDE_plains_routes_count_l3738_373851

/-- Represents the number of routes between two types of cities -/
structure RouteCount where
  total : ℕ
  mountain : ℕ
  plain : ℕ

/-- Calculates the number of routes between plains cities -/
def plainsRoutes (cities : ℕ × ℕ) (routes : RouteCount) : ℕ :=
  routes.total - routes.mountain - (cities.1 * 3 - 2 * routes.mountain) / 2

/-- Theorem stating the number of routes between plains cities -/
theorem plains_routes_count 
  (cities : ℕ × ℕ) 
  (routes : RouteCount) 
  (h1 : cities.1 + cities.2 = 100)
  (h2 : cities.1 = 30)
  (h3 : cities.2 = 70)
  (h4 : routes.total = 150)
  (h5 : routes.mountain = 21) :
  plainsRoutes cities routes = 81 := by
sorry

end NUMINAMATH_CALUDE_plains_routes_count_l3738_373851


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3738_373824

/-- The distance between vertices of a hyperbola with equation x^2/16 - y^2/9 = 1 is 8 -/
theorem hyperbola_vertex_distance :
  let h : ℝ × ℝ → ℝ := fun (x, y) ↦ x^2/16 - y^2/9 - 1
  ∃ v₁ v₂ : ℝ × ℝ, h v₁ = 0 ∧ h v₂ = 0 ∧ ‖v₁ - v₂‖ = 8 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3738_373824


namespace NUMINAMATH_CALUDE_place_two_after_two_digit_number_l3738_373820

theorem place_two_after_two_digit_number (a b : ℕ) (h1 : a ≥ 1) (h2 : a ≤ 9) (h3 : b ≤ 9) : 
  (10 * a + b) * 10 + 2 = 100 * a + 10 * b + 2 := by
  sorry

end NUMINAMATH_CALUDE_place_two_after_two_digit_number_l3738_373820


namespace NUMINAMATH_CALUDE_sine_theorem_trihedral_angle_first_cosine_theorem_trihedral_angle_second_cosine_theorem_trihedral_angle_l3738_373845

/-- A trihedral angle with face angles α, β, γ and opposite dihedral angles A, B, C. -/
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real
  A : Real
  B : Real
  C : Real

/-- The sine theorem for a trihedral angle holds. -/
theorem sine_theorem_trihedral_angle (t : TrihedralAngle) :
  (Real.sin t.α) / (Real.sin t.A) = (Real.sin t.β) / (Real.sin t.B) ∧
  (Real.sin t.β) / (Real.sin t.B) = (Real.sin t.γ) / (Real.sin t.C) :=
sorry

/-- The first cosine theorem for a trihedral angle holds. -/
theorem first_cosine_theorem_trihedral_angle (t : TrihedralAngle) :
  Real.cos t.α = Real.cos t.β * Real.cos t.γ + Real.sin t.β * Real.sin t.γ * Real.cos t.A :=
sorry

/-- The second cosine theorem for a trihedral angle holds. -/
theorem second_cosine_theorem_trihedral_angle (t : TrihedralAngle) :
  Real.cos t.A = -Real.cos t.B * Real.cos t.C + Real.sin t.B * Real.sin t.C * Real.cos t.α :=
sorry

end NUMINAMATH_CALUDE_sine_theorem_trihedral_angle_first_cosine_theorem_trihedral_angle_second_cosine_theorem_trihedral_angle_l3738_373845


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3738_373857

theorem inequality_and_equality_condition (x y z : ℝ) (h : x^2 + y^2 + z^2 = 3) :
  x^3 - (y^2 + y*z + z^2)*x + y^2*z + y*z^2 ≤ 3 * Real.sqrt 3 ∧
  (x^3 - (y^2 + y*z + z^2)*x + y^2*z + y*z^2 = 3 * Real.sqrt 3 ↔
    ((x = Real.sqrt 3 ∧ y = 0 ∧ z = 0) ∨
     (x = -Real.sqrt 3 / 3 ∧ y = 2 * Real.sqrt 3 / 3 ∧ z = 2 * Real.sqrt 3 / 3))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3738_373857


namespace NUMINAMATH_CALUDE_total_allocation_schemes_l3738_373885

def num_classes : ℕ := 4
def total_spots : ℕ := 5
def min_spots_class_a : ℕ := 2

def allocation_schemes (n c m : ℕ) : ℕ :=
  -- n: total spots
  -- c: number of classes
  -- m: minimum spots for Class A
  sorry

theorem total_allocation_schemes :
  allocation_schemes total_spots num_classes min_spots_class_a = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_allocation_schemes_l3738_373885


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3738_373887

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 2) * a n = (a (n + 1))^2) 
  (h_a1 : a 1 = 1) 
  (h_product : a 1 * a 2 * a 3 = -8) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = -2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3738_373887


namespace NUMINAMATH_CALUDE_soccer_cards_l3738_373888

theorem soccer_cards (total_players : ℕ) (no_caution_players : ℕ) (yellow_to_red : ℕ) : 
  total_players = 11 →
  no_caution_players = 5 →
  yellow_to_red = 2 →
  (total_players - no_caution_players) / yellow_to_red = 3 := by
  sorry

end NUMINAMATH_CALUDE_soccer_cards_l3738_373888


namespace NUMINAMATH_CALUDE_a_values_l3738_373843

def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

theorem a_values (a : ℝ) : A ∪ B a = A → a = 0 ∨ a = -1 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_a_values_l3738_373843


namespace NUMINAMATH_CALUDE_max_rented_trucks_24_l3738_373812

/-- Represents the truck rental scenario for a week -/
structure TruckRental where
  total_trucks : ℕ
  returned_ratio : ℚ
  saturday_trucks : ℕ

/-- The maximum number of trucks that could have been rented out during the week -/
def max_rented_trucks (rental : TruckRental) : ℕ :=
  min rental.total_trucks (2 * rental.saturday_trucks)

/-- Theorem stating the maximum number of rented trucks for the given scenario -/
theorem max_rented_trucks_24 (rental : TruckRental) 
    (h1 : rental.total_trucks = 24)
    (h2 : rental.returned_ratio = 1/2)
    (h3 : rental.saturday_trucks ≥ 12) :
  max_rented_trucks rental = 24 := by
  sorry

#eval max_rented_trucks ⟨24, 1/2, 12⟩

end NUMINAMATH_CALUDE_max_rented_trucks_24_l3738_373812


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3738_373892

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →
  a = Real.sqrt 7 →
  (1 / 2) * b * c * Real.sin A = 3 * Real.sqrt 3 / 2 →
  a + b + c = 5 + Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3738_373892


namespace NUMINAMATH_CALUDE_basketball_distribution_l3738_373850

theorem basketball_distribution (total : ℕ) (left : ℕ) (classes : ℕ) : 
  total = 54 → left = 5 → classes * ((total - left) / classes) = total - left → classes = 7 := by
  sorry

end NUMINAMATH_CALUDE_basketball_distribution_l3738_373850


namespace NUMINAMATH_CALUDE_same_parity_smallest_largest_l3738_373803

/-- A set with certain properties related to positioning in a function or polynomial -/
def A_P : Set ℤ := sorry

/-- The smallest element of A_P -/
def smallest (A : Set ℤ) : ℤ := sorry

/-- The largest element of A_P -/
def largest (A : Set ℤ) : ℤ := sorry

/-- A function to determine if a number is even -/
def isEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem same_parity_smallest_largest : 
  isEven (smallest A_P) ↔ isEven (largest A_P) := by sorry

end NUMINAMATH_CALUDE_same_parity_smallest_largest_l3738_373803


namespace NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_equals_five_halves_l3738_373877

theorem sqrt_a_div_sqrt_b_equals_five_halves (a b : ℝ) 
  (h : (1/3)^2 + (1/4)^2 = (25*a / 61*b) * ((1/5)^2 + (1/6)^2)) : 
  Real.sqrt a / Real.sqrt b = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_equals_five_halves_l3738_373877


namespace NUMINAMATH_CALUDE_value_of_N_l3738_373872

theorem value_of_N : ∃ N : ℝ, (0.25 * N = 0.55 * 5000) ∧ (N = 11000) := by
  sorry

end NUMINAMATH_CALUDE_value_of_N_l3738_373872


namespace NUMINAMATH_CALUDE_specific_rectangle_measurements_l3738_373884

/-- A rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculate the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculate the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem stating the area and perimeter of a specific rectangle -/
theorem specific_rectangle_measurements :
  let r : Rectangle := { length := 0.5, width := 0.36 }
  area r = 0.18 ∧ perimeter r = 1.72 := by
  sorry

end NUMINAMATH_CALUDE_specific_rectangle_measurements_l3738_373884


namespace NUMINAMATH_CALUDE_overtime_pay_ratio_l3738_373886

/-- Calculates the ratio of overtime pay rate to regular pay rate -/
theorem overtime_pay_ratio (regular_rate : ℚ) (regular_hours : ℚ) (total_pay : ℚ) (overtime_hours : ℚ)
  (h1 : regular_rate = 3)
  (h2 : regular_hours = 40)
  (h3 : total_pay = 186)
  (h4 : overtime_hours = 11) :
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_pay - regular_pay
  let overtime_rate := overtime_pay / overtime_hours
  overtime_rate / regular_rate = 2 := by
  sorry


end NUMINAMATH_CALUDE_overtime_pay_ratio_l3738_373886


namespace NUMINAMATH_CALUDE_solution_ratio_l3738_373876

/-- Given a system of linear equations with a non-zero solution (x, y, z) and parameter k:
    x + k*y + 4*z = 0
    4*x + k*y + z = 0
    3*x + 5*y - 2*z = 0
    Prove that xz/y^2 = 25 -/
theorem solution_ratio (k x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : x + k*y + 4*z = 0)
  (eq2 : 4*x + k*y + z = 0)
  (eq3 : 3*x + 5*y - 2*z = 0) :
  x*z / (y^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_solution_ratio_l3738_373876


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3738_373853

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 + m * x + 16 = 0) ↔ (m = 8 * Real.sqrt 3 ∨ m = -8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3738_373853


namespace NUMINAMATH_CALUDE_equation_solution_l3738_373829

theorem equation_solution : ∃ x : ℝ, -200 * x = 1600 ∧ x = -8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3738_373829


namespace NUMINAMATH_CALUDE_max_tax_revenue_l3738_373838

-- Define the market conditions
def supply_function (P : ℝ) : ℝ := 6 * P - 312
def demand_slope : ℝ := 4
def tax_rate : ℝ := 30
def consumer_price : ℝ := 118

-- Define the demand function
def demand_function (P : ℝ) : ℝ := 688 - demand_slope * P

-- Define the tax revenue function
def tax_revenue (t : ℝ) : ℝ := (288 - 2.4 * t) * t

-- Theorem statement
theorem max_tax_revenue :
  ∃ (t : ℝ), ∀ (t' : ℝ), tax_revenue t ≥ tax_revenue t' ∧ tax_revenue t = 8640 := by
  sorry


end NUMINAMATH_CALUDE_max_tax_revenue_l3738_373838


namespace NUMINAMATH_CALUDE_smallest_k_for_convergence_l3738_373827

def u : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 3 * u n - 3 * (u n)^3

def L : ℚ := 1/3

theorem smallest_k_for_convergence :
  ∀ k : ℕ, k ≥ 1 → |u k - L| ≤ 1 / 3^300 ∧
  ∀ m : ℕ, m < k → |u m - L| > 1 / 3^300 →
  k = 1 := by sorry

end NUMINAMATH_CALUDE_smallest_k_for_convergence_l3738_373827


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_5_l3738_373818

theorem smallest_positive_integer_ending_in_3_divisible_by_5 : ∃ n : ℕ,
  (n % 10 = 3) ∧ 
  (n % 5 = 0) ∧ 
  (∀ m : ℕ, m < n → (m % 10 = 3 → m % 5 ≠ 0)) ∧
  n = 53 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_5_l3738_373818


namespace NUMINAMATH_CALUDE_first_pipe_fill_time_l3738_373878

/-- The time it takes for the first pipe to fill the cistern -/
def T : ℝ := 10

/-- The time it takes for the second pipe to fill the cistern -/
def second_pipe_time : ℝ := 12

/-- The time it takes for the third pipe to empty the cistern -/
def third_pipe_time : ℝ := 25

/-- The time it takes to fill the cistern when all pipes are opened simultaneously -/
def combined_time : ℝ := 6.976744186046512

theorem first_pipe_fill_time :
  (1 / T + 1 / second_pipe_time - 1 / third_pipe_time) * combined_time = 1 :=
sorry

end NUMINAMATH_CALUDE_first_pipe_fill_time_l3738_373878


namespace NUMINAMATH_CALUDE_mod_equivalence_2021_l3738_373869

theorem mod_equivalence_2021 :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -2021 [ZMOD 13] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_2021_l3738_373869


namespace NUMINAMATH_CALUDE_smallest_cube_ending_112_l3738_373868

theorem smallest_cube_ending_112 : ∃ n : ℕ+, (
  n^3 ≡ 112 [ZMOD 1000] ∧
  ∀ m : ℕ+, m^3 ≡ 112 [ZMOD 1000] → n ≤ m
) ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_112_l3738_373868


namespace NUMINAMATH_CALUDE_exam_pupils_count_l3738_373833

theorem exam_pupils_count :
  ∀ (n : ℕ) (total_marks : ℕ),
    n > 4 →
    total_marks = 39 * n →
    (total_marks - 71) / (n - 4) = 44 →
    n = 21 := by
  sorry

end NUMINAMATH_CALUDE_exam_pupils_count_l3738_373833


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l3738_373841

/-- Given a salary that increases to $812 with a 16% raise, 
    prove that a 10% raise results in $770.0000000000001 -/
theorem salary_increase_percentage (S : ℝ) 
  (h1 : S + 0.16 * S = 812) 
  (h2 : S + 0.1 * S = 770.0000000000001) : 
  ∃ (P : ℝ), S + P * S = 770.0000000000001 ∧ P = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l3738_373841


namespace NUMINAMATH_CALUDE_largest_d_for_g_range_contains_two_l3738_373810

/-- The function g(x) defined as x^2 - 6x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + d

/-- The theorem stating that the largest value of d such that 2 is in the range of g(x) is 11 -/
theorem largest_d_for_g_range_contains_two :
  ∃ (d_max : ℝ), d_max = 11 ∧
  (∀ d : ℝ, (∃ x : ℝ, g d x = 2) → d ≤ d_max) ∧
  (∃ x : ℝ, g d_max x = 2) :=
sorry

end NUMINAMATH_CALUDE_largest_d_for_g_range_contains_two_l3738_373810


namespace NUMINAMATH_CALUDE_seaweed_for_livestock_l3738_373848

def total_seaweed : ℝ := 500

def fire_percentage : ℝ := 0.4
def medicinal_percentage : ℝ := 0.2
def food_and_feed_percentage : ℝ := 0.4

def human_consumption_ratio : ℝ := 0.3

theorem seaweed_for_livestock (total : ℝ) (fire_pct : ℝ) (med_pct : ℝ) (food_feed_pct : ℝ) (human_ratio : ℝ) 
    (h1 : total = total_seaweed)
    (h2 : fire_pct = fire_percentage)
    (h3 : med_pct = medicinal_percentage)
    (h4 : food_feed_pct = food_and_feed_percentage)
    (h5 : human_ratio = human_consumption_ratio)
    (h6 : fire_pct + med_pct + food_feed_pct = 1) :
  food_feed_pct * total * (1 - human_ratio) = 140 := by
  sorry

end NUMINAMATH_CALUDE_seaweed_for_livestock_l3738_373848


namespace NUMINAMATH_CALUDE_triangle_area_is_correct_l3738_373825

/-- The slope of the first line -/
def m₁ : ℚ := 1/3

/-- The slope of the second line -/
def m₂ : ℚ := 3

/-- The point of intersection of the first two lines -/
def A : ℚ × ℚ := (3, 3)

/-- The equation of the third line: x + y = 12 -/
def line3 (x y : ℚ) : Prop := x + y = 12

/-- The area of the triangle formed by the three lines -/
noncomputable def triangle_area : ℚ := sorry

theorem triangle_area_is_correct : triangle_area = 8625/1000 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_correct_l3738_373825


namespace NUMINAMATH_CALUDE_garden_length_theorem_l3738_373836

/-- Represents a rectangular garden with given dimensions and area allocations. -/
structure Garden where
  length : ℝ
  width : ℝ
  tilled_ratio : ℝ
  trellised_ratio : ℝ
  raised_bed_area : ℝ

/-- Theorem stating the conditions and conclusion about the garden's length. -/
theorem garden_length_theorem (g : Garden) : 
  g.width = 120 ∧ 
  g.tilled_ratio = 1/2 ∧ 
  g.trellised_ratio = 1/3 ∧ 
  g.raised_bed_area = 8800 →
  g.length = 220 := by
  sorry

#check garden_length_theorem

end NUMINAMATH_CALUDE_garden_length_theorem_l3738_373836


namespace NUMINAMATH_CALUDE_min_M_is_two_thirds_l3738_373895

-- Define the set of quadratic polynomials satisfying the conditions
def QuadraticPolynomials : Set (ℝ → ℝ) :=
  {p | ∀ x ∈ Set.Icc (-1 : ℝ) 1,
       (∃ a b c : ℝ, p = fun x ↦ a * x^2 + b * x + c) ∧
       p x ≥ 0 ∧
       (∫ x in Set.Icc (-1 : ℝ) 1, p x) = 1}

-- Define M(x) as the maximum value of polynomials in QuadraticPolynomials at x
noncomputable def M (x : ℝ) : ℝ :=
  ⨆ (p ∈ QuadraticPolynomials), p x

theorem min_M_is_two_thirds :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, M x ≥ 2/3) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, M x = 2/3) :=
sorry

end NUMINAMATH_CALUDE_min_M_is_two_thirds_l3738_373895


namespace NUMINAMATH_CALUDE_book_cost_l3738_373826

theorem book_cost (initial_amount : ℕ) (num_books : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 79)
  (h2 : num_books = 9)
  (h3 : remaining_amount = 16) :
  (initial_amount - remaining_amount) / num_books = 7 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_l3738_373826


namespace NUMINAMATH_CALUDE_inequality_solution_comparison_l3738_373842

theorem inequality_solution_comparison (m n : ℝ) 
  (hm : 5 * m - 2 ≥ 3) 
  (hn : ¬(5 * n - 2 ≥ 3)) : 
  m > n :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_comparison_l3738_373842


namespace NUMINAMATH_CALUDE_exists_complete_list_l3738_373837

-- Define the tournament structure
structure Tournament where
  players : Type
  played : players → players → Prop
  winner : players → players → Prop
  no_draw : ∀ (a b : players), played a b → (winner a b ∨ winner b a)
  all_play : ∀ (a b : players), a ≠ b → played a b
  no_self_play : ∀ (a : players), ¬played a a

-- Define the list of beaten players for each player
def beaten_list (t : Tournament) (a : t.players) : Set t.players :=
  {b | t.winner a b ∨ ∃ c, t.winner a c ∧ t.winner c b}

-- Theorem statement
theorem exists_complete_list (t : Tournament) :
  ∃ a : t.players, ∀ b : t.players, b ≠ a → b ∈ beaten_list t a :=
sorry

end NUMINAMATH_CALUDE_exists_complete_list_l3738_373837


namespace NUMINAMATH_CALUDE_nine_ones_squared_l3738_373875

def nine_ones : ℕ := 111111111

theorem nine_ones_squared :
  nine_ones ^ 2 = 12345678987654321 := by sorry

end NUMINAMATH_CALUDE_nine_ones_squared_l3738_373875


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3738_373800

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (3 + n) = 8 → n = 61 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3738_373800


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l3738_373849

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (large_cube_edge : ℝ) (h : large_cube_edge = 12) :
  let sphere_diameter := large_cube_edge
  let small_cube_diagonal := sphere_diameter
  let small_cube_edge := small_cube_diagonal / Real.sqrt 3
  let small_cube_volume := small_cube_edge ^ 3
  small_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l3738_373849


namespace NUMINAMATH_CALUDE_constant_sum_of_powers_l3738_373874

theorem constant_sum_of_powers (n : ℕ+) :
  (∀ x y z : ℝ, x + y + z = 0 → x * y * z = 1 → 
    ∃ c : ℝ, ∀ a b d : ℝ, a + b + d = 0 → a * b * d = 1 → 
      a^(n : ℕ) + b^(n : ℕ) + d^(n : ℕ) = c) ↔ n = 1 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_constant_sum_of_powers_l3738_373874


namespace NUMINAMATH_CALUDE_estimate_population_characteristic_l3738_373865

/-- Given a population and a sample, estimate the total number with a certain characteristic -/
theorem estimate_population_characteristic
  (total_population : ℕ)
  (sample_size : ℕ)
  (sample_with_characteristic : ℕ)
  (sample_size_positive : sample_size > 0)
  (sample_size_le_total : sample_size ≤ total_population)
  (sample_with_characteristic_le_sample : sample_with_characteristic ≤ sample_size) :
  let estimated_total := (total_population * sample_with_characteristic) / sample_size
  estimated_total = 6000 ∧ 
  estimated_total ≤ total_population ∧
  (sample_with_characteristic : ℚ) / (sample_size : ℚ) = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_estimate_population_characteristic_l3738_373865


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3738_373882

theorem cubic_root_sum (a b c : ℝ) : 
  0 < a ∧ a < 1 ∧
  0 < b ∧ b < 1 ∧
  0 < c ∧ c < 1 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  24 * a^3 - 36 * a^2 + 14 * a - 1 = 0 ∧
  24 * b^3 - 36 * b^2 + 14 * b - 1 = 0 ∧
  24 * c^3 - 36 * c^2 + 14 * c - 1 = 0 →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3738_373882


namespace NUMINAMATH_CALUDE_complement_of_M_l3738_373847

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 < 2*x}

theorem complement_of_M : Set.compl M = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l3738_373847


namespace NUMINAMATH_CALUDE_darryl_melon_sales_l3738_373813

/-- Calculates the total money made from selling melons given the initial quantities,
    prices, dropped/rotten melons, and remaining melons. -/
def total_money_made (initial_cantaloupes initial_honeydews : ℕ)
                     (price_cantaloupe price_honeydew : ℕ)
                     (dropped_cantaloupes rotten_honeydews : ℕ)
                     (remaining_cantaloupes remaining_honeydews : ℕ) : ℕ :=
  let sold_cantaloupes := initial_cantaloupes - remaining_cantaloupes - dropped_cantaloupes
  let sold_honeydews := initial_honeydews - remaining_honeydews - rotten_honeydews
  sold_cantaloupes * price_cantaloupe + sold_honeydews * price_honeydew

/-- Theorem stating that Darryl made $85 from selling melons. -/
theorem darryl_melon_sales : 
  total_money_made 30 27 2 3 2 3 8 9 = 85 := by
  sorry


end NUMINAMATH_CALUDE_darryl_melon_sales_l3738_373813


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l3738_373864

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n.gcd d = 1) ∧ 
  (n : ℚ) / (d : ℚ) = 3 + 834 / 999 ∧
  n + d = 4830 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l3738_373864


namespace NUMINAMATH_CALUDE_distance_between_trees_l3738_373809

/-- Given a yard of length 400 meters with 26 trees planted at equal distances,
    including one tree at each end, the distance between consecutive trees is 16 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 400 ∧ num_trees = 26 →
  (yard_length / (num_trees - 1 : ℝ)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l3738_373809


namespace NUMINAMATH_CALUDE_square_roots_problem_l3738_373859

theorem square_roots_problem (x : ℝ) (h1 : x > 0) 
  (h2 : ∃ a : ℝ, (2 - a)^2 = x ∧ (2*a + 1)^2 = x) :
  ∃ a : ℝ, a = -3 ∧ (17 - x)^(1/3 : ℝ) = -2 := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l3738_373859


namespace NUMINAMATH_CALUDE_function_value_problem_l3738_373852

theorem function_value_problem (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x, f ((x / 2) - 1) = 2 * x + 3)
  (h2 : f m = 6) : 
  m = -1/4 := by sorry

end NUMINAMATH_CALUDE_function_value_problem_l3738_373852


namespace NUMINAMATH_CALUDE_range_of_a_l3738_373802

-- Define the sets P and M
def P : Set ℝ := {x | x^2 ≤ 1}
def M (a : ℝ) : Set ℝ := {a}

-- State the theorem
theorem range_of_a (a : ℝ) : P ∪ M a = P → a ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3738_373802


namespace NUMINAMATH_CALUDE_no_return_after_2020_rounds_l3738_373897

/-- Represents the movement of a ball in a circular arrangement of boxes. -/
def ballMovement (N : ℕ+) (n₀ : ℕ) (k : ℕ) : ℕ :=
  (k * (2 * n₀ + k - 1) / 2) % N

/-- Represents the number on the ball after k rounds. -/
def ballNumber (N : ℕ+) (n₀ : ℕ) (k : ℕ) : ℕ :=
  (n₀ + k - 1) % N + 1

theorem no_return_after_2020_rounds (N : ℕ+) (n₀ : ℕ) (h : n₀ ≥ 1 ∧ n₀ ≤ N) :
  ¬(ballMovement N n₀ 2020 = 0 ∧ ballNumber N n₀ 2020 = n₀) :=
sorry

end NUMINAMATH_CALUDE_no_return_after_2020_rounds_l3738_373897


namespace NUMINAMATH_CALUDE_calculate_expression_l3738_373893

theorem calculate_expression : 15 * (2/3) * 45 + 15 * (1/3) * 90 = 900 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3738_373893


namespace NUMINAMATH_CALUDE_granola_bar_cost_l3738_373890

/-- Calculates the total cost of granola bars after applying a discount --/
def total_cost_after_discount (
  oatmeal_quantity : ℕ)
  (oatmeal_price : ℚ)
  (peanut_quantity : ℕ)
  (peanut_price : ℚ)
  (chocolate_quantity : ℕ)
  (chocolate_price : ℚ)
  (mixed_quantity : ℕ)
  (mixed_price : ℚ)
  (discount_percentage : ℚ) : ℚ :=
  let total_before_discount := 
    oatmeal_quantity * oatmeal_price +
    peanut_quantity * peanut_price +
    chocolate_quantity * chocolate_price +
    mixed_quantity * mixed_price
  let discount_amount := (discount_percentage / 100) * total_before_discount
  total_before_discount - discount_amount

/-- Theorem stating the total cost after discount for the given problem --/
theorem granola_bar_cost : 
  total_cost_after_discount 6 (25/20) 8 (3/2) 5 (7/4) 3 2 15 = 2911/100 :=
sorry

end NUMINAMATH_CALUDE_granola_bar_cost_l3738_373890


namespace NUMINAMATH_CALUDE_triangle_properties_l3738_373883

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  2 * t.b * Real.cos t.A + t.a = 2 * t.c ∧
  t.c = 8 ∧
  Real.sin t.A = (3 * Real.sqrt 3) / 14

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.B = π / 3 ∧ 
  (1 / 2 * t.a * t.c * Real.sin t.B) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3738_373883


namespace NUMINAMATH_CALUDE_initial_points_count_l3738_373856

/-- The number of points after one operation -/
def points_after_one_op (n : ℕ) : ℕ := 2 * n - 1

/-- The number of points after two operations -/
def points_after_two_ops (n : ℕ) : ℕ := 2 * (points_after_one_op n) - 1

/-- The number of points after three operations -/
def points_after_three_ops (n : ℕ) : ℕ := 2 * (points_after_two_ops n) - 1

/-- 
Theorem: If we start with n points on a line, perform the operation of adding a point 
between each pair of neighboring points three times, and end up with 65 points, 
then n must be equal to 9.
-/
theorem initial_points_count : points_after_three_ops 9 = 65 ∧ 
  (∀ m : ℕ, points_after_three_ops m = 65 → m = 9) := by
  sorry

end NUMINAMATH_CALUDE_initial_points_count_l3738_373856


namespace NUMINAMATH_CALUDE_range_of_function_l3738_373814

theorem range_of_function (x : ℝ) (h : x ≥ -1) :
  let y := (12 * Real.sqrt (x + 1)) / (3 * x + 4)
  0 ≤ y ∧ y ≤ 2 * Real.sqrt 3 :=
sorry


end NUMINAMATH_CALUDE_range_of_function_l3738_373814


namespace NUMINAMATH_CALUDE_steves_oranges_l3738_373891

/-- Given that Steve shares some oranges and has a certain number left, 
    this theorem proves how many oranges Steve had initially. -/
theorem steves_oranges (shared : ℕ) (left : ℕ) (initial : ℕ) : 
  shared = 4 → left = 42 → initial = shared + left → initial = 46 := by
  sorry

end NUMINAMATH_CALUDE_steves_oranges_l3738_373891


namespace NUMINAMATH_CALUDE_triangle_properties_l3738_373867

theorem triangle_properties (a b c A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  Real.sqrt 3 * b = 2 * a * Real.sin B * Real.cos C + 2 * c * Real.sin B * Real.cos A →
  a = 3 →
  c = 4 →
  B = π/3 ∧ b = Real.sqrt 13 ∧ Real.cos (2 * A + B) = -23/26 := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l3738_373867


namespace NUMINAMATH_CALUDE_exponent_equality_l3738_373871

theorem exponent_equality (n : ℕ) : 4^8 = 4^n → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l3738_373871


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3738_373807

def p (x : ℝ) : ℝ := 4*x^8 - 2*x^6 + 5*x^4 - x^3 + 3*x - 15

theorem remainder_theorem (p : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - a) * q x + p a := sorry

theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x, p x = (2*x - 6) * q x + 25158 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3738_373807


namespace NUMINAMATH_CALUDE_yellow_paint_cans_l3738_373899

theorem yellow_paint_cans (yellow green : ℕ) (total : ℕ) : 
  yellow + green = total → 
  yellow = 4 * green / 3 → 
  total = 42 → 
  yellow = 24 := by sorry

end NUMINAMATH_CALUDE_yellow_paint_cans_l3738_373899


namespace NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_sum_l3738_373817

-- Define a convex polygon
structure ConvexPolygon where
  sides : ℕ
  is_convex : sides ≥ 3

-- Define the sum of interior angles for a polygon
def sum_interior_angles (p : ConvexPolygon) : ℝ :=
  180 * (p.sides - 2 : ℝ)

-- Theorem statement
theorem polygon_sides_from_interior_angle_sum (p : ConvexPolygon) 
  (h : sum_interior_angles p - x = 2190)
  (hx : 0 < x ∧ x < 180) : p.sides = 15 := by
  sorry

#check polygon_sides_from_interior_angle_sum

end NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_sum_l3738_373817


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l3738_373819

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∃ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l3738_373819


namespace NUMINAMATH_CALUDE_morse_code_symbols_l3738_373816

theorem morse_code_symbols : 
  (Finset.range 5).sum (fun i => 2^(i+1)) = 62 :=
sorry

end NUMINAMATH_CALUDE_morse_code_symbols_l3738_373816


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3738_373860

theorem absolute_value_inequality (x : ℝ) : 
  x ≠ 0 → (|((x - 2) / x)| > ((x - 2) / x) ↔ 0 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3738_373860


namespace NUMINAMATH_CALUDE_max_value_of_f_l3738_373855

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the interval
def I : Set ℝ := Set.Icc (-4) 3

-- State the theorem
theorem max_value_of_f : 
  ∃ (x : ℝ), x ∈ I ∧ f x = 15 ∧ ∀ (y : ℝ), y ∈ I → f y ≤ f x :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3738_373855


namespace NUMINAMATH_CALUDE_vector_calculation_l3738_373862

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_calculation (a b : V) : 
  (1 / 2 : ℝ) • ((2 : ℝ) • a - (4 : ℝ) • b) + (2 : ℝ) • b = a := by
  sorry

end NUMINAMATH_CALUDE_vector_calculation_l3738_373862


namespace NUMINAMATH_CALUDE_perfect_square_with_three_or_fewer_swaps_l3738_373873

/-- Represents a permutation of digits --/
def Permutation := List Nat

/-- Checks if a permutation represents a perfect square --/
def is_perfect_square (p : Permutation) : Prop :=
  ∃ n : Nat, n * n = p.foldl (fun acc d => acc * 10 + d) 0

/-- Counts the number of swaps needed to transform one permutation into another --/
def swap_count (p1 p2 : Permutation) : Nat :=
  sorry

/-- The initial permutation of digits from 1 to 9 --/
def initial_permutation : Permutation := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- Theorem: There exists a permutation of digits 1-9 that forms a perfect square 
    and can be achieved with 3 or fewer swaps from the initial permutation --/
theorem perfect_square_with_three_or_fewer_swaps :
  ∃ (final_perm : Permutation), 
    is_perfect_square final_perm ∧ 
    swap_count initial_permutation final_perm ≤ 3 :=
  sorry

end NUMINAMATH_CALUDE_perfect_square_with_three_or_fewer_swaps_l3738_373873


namespace NUMINAMATH_CALUDE_adam_current_age_l3738_373844

/-- Adam's current age -/
def adam_age : ℕ := sorry

/-- Tom's current age -/
def tom_age : ℕ := 12

/-- Years into the future -/
def years_future : ℕ := 12

/-- Combined age in the future -/
def combined_future_age : ℕ := 44

theorem adam_current_age :
  adam_age = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_adam_current_age_l3738_373844


namespace NUMINAMATH_CALUDE_snug_fit_circles_l3738_373805

/-- Given a circle of diameter 3 inches containing two circles of diameters 2 inches and 1 inch,
    the diameter of two additional identical circles that fit snugly within the larger circle
    is 12/7 inches. -/
theorem snug_fit_circles (R : ℝ) (r₁ : ℝ) (r₂ : ℝ) (d : ℝ) :
  R = 3/2 ∧ r₁ = 1 ∧ r₂ = 1/2 →
  d > 0 →
  (R - d)^2 + (R - d)^2 = (2*d)^2 →
  d = 6/7 :=
by sorry

end NUMINAMATH_CALUDE_snug_fit_circles_l3738_373805


namespace NUMINAMATH_CALUDE_cos_330_degrees_l3738_373846

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l3738_373846


namespace NUMINAMATH_CALUDE_nina_widget_problem_l3738_373840

theorem nina_widget_problem (x : ℝ) (h1 : 6 * x = 8 * (x - 1)) : 6 * x = 24 := by
  sorry

end NUMINAMATH_CALUDE_nina_widget_problem_l3738_373840


namespace NUMINAMATH_CALUDE_plan_A_fixed_charge_l3738_373881

/-- The fixed charge for the first 4 minutes in Plan A -/
def fixed_charge : ℝ := sorry

/-- The per-minute rate after the first 4 minutes in Plan A -/
def rate_A : ℝ := 0.06

/-- The per-minute rate for Plan B -/
def rate_B : ℝ := 0.08

/-- The duration at which both plans charge the same amount -/
def equal_duration : ℝ := 18

theorem plan_A_fixed_charge :
  fixed_charge = 0.60 :=
by
  have h1 : fixed_charge + rate_A * (equal_duration - 4) = rate_B * equal_duration :=
    sorry
  sorry


end NUMINAMATH_CALUDE_plan_A_fixed_charge_l3738_373881


namespace NUMINAMATH_CALUDE_ryan_pages_theorem_l3738_373894

/-- The number of books Ryan got from the library -/
def ryan_books : ℕ := 5

/-- The number of pages in each of Ryan's brother's books -/
def brother_book_pages : ℕ := 200

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The additional pages Ryan reads per day compared to his brother -/
def ryan_extra_pages_per_day : ℕ := 100

/-- The total number of pages in Ryan's books -/
def ryan_total_pages : ℕ := days_in_week * (brother_book_pages + ryan_extra_pages_per_day)

theorem ryan_pages_theorem :
  ryan_total_pages = 2100 :=
by sorry

end NUMINAMATH_CALUDE_ryan_pages_theorem_l3738_373894


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l3738_373854

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x

theorem minimum_value_implies_a (a : ℝ) :
  a > 0 →
  (∀ x, x > 0 → x ≤ Real.exp 1 → f a x ≥ 3/2) →
  (∃ x, 1 ≤ x ∧ x ≤ Real.exp 1 ∧ f a x = 3/2) →
  a = -Real.sqrt (Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l3738_373854


namespace NUMINAMATH_CALUDE_probability_sum_not_less_than_6_l3738_373832

/-- Represents a tetrahedral die with faces numbered 1, 2, 3, 5 -/
def TetrahedralDie : Type := Fin 4

/-- The possible face values of the tetrahedral die -/
def face_values : List ℕ := [1, 2, 3, 5]

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 16

/-- Predicate to check if the sum of two face values is not less than 6 -/
def sum_not_less_than_6 (a b : ℕ) : Prop := a + b ≥ 6

/-- The number of favorable outcomes (sum not less than 6) -/
def favorable_outcomes : ℕ := 8

/-- Theorem stating that the probability of the sum being not less than 6 is 1/2 -/
theorem probability_sum_not_less_than_6 :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_sum_not_less_than_6_l3738_373832


namespace NUMINAMATH_CALUDE_inequality_proof_l3738_373861

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3738_373861


namespace NUMINAMATH_CALUDE_probability_top_given_not_female_is_one_eighth_l3738_373808

/-- Represents the probability of selecting a top student given the student is not female -/
def probability_top_given_not_female (total_students : ℕ) (female_students : ℕ) (top_fraction : ℚ) (top_female_fraction : ℚ) : ℚ :=
  let male_students := total_students - female_students
  let top_students := (total_students : ℚ) * top_fraction
  let male_top_students := top_students * (1 - top_female_fraction)
  male_top_students / male_students

/-- Theorem stating the probability of selecting a top student given the student is not female -/
theorem probability_top_given_not_female_is_one_eighth :
  probability_top_given_not_female 60 20 (1/6) (1/2) = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_probability_top_given_not_female_is_one_eighth_l3738_373808


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_l3738_373815

noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then c * x + 2
  else if x ≥ -1 then 2 * x - 4
  else 3 * x - d

theorem continuous_piecewise_function (c d : ℝ) :
  Continuous (f c d) → c + d = -7 := by
  sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_l3738_373815


namespace NUMINAMATH_CALUDE_snail_meets_minute_hand_l3738_373828

/-- Represents the position on a clock face in minutes past 12 -/
def ClockPosition := ℕ

/-- Calculates the position of the snail at a given time -/
def snail_position (time : ℕ) : ClockPosition :=
  (3 * time) % 60

/-- Calculates the position of the minute hand at a given time -/
def minute_hand_position (time : ℕ) : ClockPosition :=
  time % 60

/-- Checks if the snail and minute hand meet at a given time -/
def meets_at (time : ℕ) : Prop :=
  snail_position time = minute_hand_position time

theorem snail_meets_minute_hand :
  meets_at 40 ∧ meets_at 80 :=
sorry

end NUMINAMATH_CALUDE_snail_meets_minute_hand_l3738_373828


namespace NUMINAMATH_CALUDE_phone_number_probability_l3738_373870

def first_three_digits : ℕ := 3
def last_five_digits_arrangements : ℕ := 300

theorem phone_number_probability :
  let total_possibilities := first_three_digits * last_five_digits_arrangements
  (1 : ℚ) / total_possibilities = (1 : ℚ) / 900 :=
by sorry

end NUMINAMATH_CALUDE_phone_number_probability_l3738_373870


namespace NUMINAMATH_CALUDE_puppies_bought_l3738_373834

/-- The total number of puppies bought by Arven -/
def total_puppies : ℕ := 5

/-- The cost of each puppy on sale -/
def sale_price : ℕ := 150

/-- The cost of each puppy not on sale -/
def regular_price : ℕ := 175

/-- The number of puppies on sale -/
def sale_puppies : ℕ := 3

/-- The total cost of all puppies -/
def total_cost : ℕ := 800

theorem puppies_bought :
  total_puppies = sale_puppies + (total_cost - sale_puppies * sale_price) / regular_price :=
by sorry

end NUMINAMATH_CALUDE_puppies_bought_l3738_373834


namespace NUMINAMATH_CALUDE_jason_oranges_l3738_373831

/-- 
Given that Mary picked 122 oranges and the total number of oranges picked by Mary and Jason is 227,
prove that Jason picked 105 oranges.
-/
theorem jason_oranges :
  let mary_oranges : ℕ := 122
  let total_oranges : ℕ := 227
  let jason_oranges : ℕ := total_oranges - mary_oranges
  jason_oranges = 105 := by sorry

end NUMINAMATH_CALUDE_jason_oranges_l3738_373831


namespace NUMINAMATH_CALUDE_det_sine_matrix_zero_l3738_373879

theorem det_sine_matrix_zero :
  let M : Matrix (Fin 3) (Fin 3) ℝ := λ i j ↦ 
    Real.sin (((i : ℕ) * 3 + (j : ℕ) + 2) : ℝ)
  Matrix.det M = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_sine_matrix_zero_l3738_373879


namespace NUMINAMATH_CALUDE_leading_coefficient_of_p_l3738_373896

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := -5*(x^5 - x^4 + x^3) + 8*(x^5 + 3) - 3*(2*x^5 + x^3 + 2)

/-- The leading coefficient of a polynomial -/
def leadingCoefficient (f : ℝ → ℝ) : ℝ :=
  sorry -- Definition of leading coefficient

theorem leading_coefficient_of_p :
  leadingCoefficient p = -3 := by sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_p_l3738_373896


namespace NUMINAMATH_CALUDE_dave_tickets_used_l3738_373806

/-- Given that Dave had 13 tickets initially and has 7 tickets left after buying toys,
    prove that he used 6 tickets to buy toys. -/
theorem dave_tickets_used (initial : ℕ) (left : ℕ) (used : ℕ) 
    (h1 : initial = 13) 
    (h2 : left = 7) 
    (h3 : used = initial - left) : 
  used = 6 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_used_l3738_373806


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3738_373823

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) + 2
  f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3738_373823


namespace NUMINAMATH_CALUDE_square_difference_divided_l3738_373801

theorem square_difference_divided : (111^2 - 99^2) / 12 = 210 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_l3738_373801


namespace NUMINAMATH_CALUDE_A_sufficient_not_necessary_l3738_373880

-- Define propositions A and B
def A (x y : ℝ) : Prop := x + y ≠ 8
def B (x y : ℝ) : Prop := x ≠ 2 ∨ y ≠ 6

-- Theorem statement
theorem A_sufficient_not_necessary :
  (∀ x y : ℝ, A x y → B x y) ∧
  ¬(∀ x y : ℝ, B x y → A x y) :=
by sorry

end NUMINAMATH_CALUDE_A_sufficient_not_necessary_l3738_373880


namespace NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l3738_373835

theorem gcd_lcm_product_90_150 : 
  (Nat.gcd 90 150) * (Nat.lcm 90 150) = 13500 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_90_150_l3738_373835


namespace NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l3738_373863

/-- Given a total sum of 2769 divided into two parts, if the interest on the first part
    for 8 years at 3% per annum equals the interest on the second part for 3 years at 5% per annum,
    then the second part is 1704. -/
theorem interest_equality_implies_second_sum (total : ℝ) (first_part second_part : ℝ) :
  total = 2769 →
  first_part + second_part = total →
  (first_part * 3 * 8) / 100 = (second_part * 5 * 3) / 100 →
  second_part = 1704 :=
by sorry

end NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l3738_373863


namespace NUMINAMATH_CALUDE_round_trip_ticket_percentage_l3738_373821

theorem round_trip_ticket_percentage (total_passengers : ℝ) 
  (h1 : (0.2 : ℝ) * total_passengers = (passengers_with_roundtrip_and_car : ℝ))
  (h2 : (0.5 : ℝ) * (passengers_with_roundtrip : ℝ) = passengers_with_roundtrip - passengers_with_roundtrip_and_car) :
  (passengers_with_roundtrip : ℝ) / total_passengers = (0.4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_round_trip_ticket_percentage_l3738_373821
