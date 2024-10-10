import Mathlib

namespace arcsin_one_eq_pi_div_two_l1378_137890

-- Define arcsin function
noncomputable def arcsin (x : ℝ) : ℝ :=
  Real.arcsin x

-- State the theorem
theorem arcsin_one_eq_pi_div_two :
  arcsin 1 = π / 2 :=
sorry

end arcsin_one_eq_pi_div_two_l1378_137890


namespace unique_quadratic_function_l1378_137818

-- Define the property that f should satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = ⨆ y : ℝ, (2 * x * y - f y)

-- State the theorem
theorem unique_quadratic_function :
  ∃! f : ℝ → ℝ, SatisfiesProperty f ∧ ∀ x : ℝ, f x = x^2 :=
sorry

end unique_quadratic_function_l1378_137818


namespace exterior_angle_regular_octagon_l1378_137865

theorem exterior_angle_regular_octagon : 
  ∀ (n : ℕ) (sum_exterior_angles : ℝ),
  n = 8 → 
  sum_exterior_angles = 360 →
  (sum_exterior_angles / n : ℝ) = 45 := by
sorry

end exterior_angle_regular_octagon_l1378_137865


namespace m_greater_than_n_l1378_137869

theorem m_greater_than_n (x y : ℝ) : x^2 + y^2 + 1 > 2*(x + y - 1) := by
  sorry

end m_greater_than_n_l1378_137869


namespace trig_identity_l1378_137899

theorem trig_identity (α : ℝ) : 
  Real.sin (9 * α) + Real.sin (10 * α) + Real.sin (11 * α) + Real.sin (12 * α) = 
  4 * Real.cos (α / 2) * Real.cos α * Real.sin ((21 * α) / 2) := by
  sorry

end trig_identity_l1378_137899


namespace field_trip_vans_l1378_137842

theorem field_trip_vans (total_people : ℕ) (num_buses : ℕ) (people_per_bus : ℕ) (people_per_van : ℕ) :
  total_people = 180 →
  num_buses = 8 →
  people_per_bus = 18 →
  people_per_van = 6 →
  ∃ (num_vans : ℕ), num_vans = 6 ∧ total_people = num_buses * people_per_bus + num_vans * people_per_van :=
by sorry

end field_trip_vans_l1378_137842


namespace right_triangle_squares_area_l1378_137834

theorem right_triangle_squares_area (x : ℝ) :
  let triangle_area := (1/2) * (3*x) * (4*x)
  let square1_area := (3*x)^2
  let square2_area := (4*x)^2
  let total_area := triangle_area + square1_area + square2_area
  (total_area = 1000) → (x = 10 * Real.sqrt 31 / 31) := by
  sorry

end right_triangle_squares_area_l1378_137834


namespace inequality_proof_l1378_137888

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) ≥ -17 + 12 * Real.sqrt 2 :=
by sorry

end inequality_proof_l1378_137888


namespace joshua_bottle_caps_l1378_137836

/-- The initial number of bottle caps Joshua had -/
def initial_caps : ℕ := 40

/-- The number of bottle caps Joshua bought -/
def bought_caps : ℕ := 7

/-- The total number of bottle caps Joshua has after buying more -/
def total_caps : ℕ := 47

theorem joshua_bottle_caps : initial_caps + bought_caps = total_caps := by
  sorry

end joshua_bottle_caps_l1378_137836


namespace magic_square_y_value_l1378_137867

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a : ℕ) (b : ℕ) (c : ℕ)
  (d : ℕ) (e : ℕ) (f : ℕ)
  (g : ℕ) (h : ℕ) (i : ℕ)

/-- Checks if a given 3x3 square is a magic square -/
def is_magic_square (s : MagicSquare) : Prop :=
  let sum := s.a + s.b + s.c
  sum = s.d + s.e + s.f ∧
  sum = s.g + s.h + s.i ∧
  sum = s.a + s.d + s.g ∧
  sum = s.b + s.e + s.h ∧
  sum = s.c + s.f + s.i ∧
  sum = s.a + s.e + s.i ∧
  sum = s.c + s.e + s.g

theorem magic_square_y_value (y : ℕ) :
  ∃ (s : MagicSquare), 
    is_magic_square s ∧ 
    s.a = y ∧ s.b = 25 ∧ s.c = 70 ∧ 
    s.d = 5 → 
    y = 90 := by sorry

end magic_square_y_value_l1378_137867


namespace minimum_nickels_needed_l1378_137827

def shoe_cost : ℚ := 45.50
def ten_dollar_bills : ℕ := 4
def quarters : ℕ := 5
def quarter_value : ℚ := 0.25
def nickel_value : ℚ := 0.05

theorem minimum_nickels_needed :
  ∃ n : ℕ, 
    (n : ℚ) * nickel_value + (ten_dollar_bills * 10 : ℚ) + (quarters : ℚ) * quarter_value ≥ shoe_cost ∧
    ∀ m : ℕ, m < n → (m : ℚ) * nickel_value + (ten_dollar_bills * 10 : ℚ) + (quarters : ℚ) * quarter_value < shoe_cost :=
by
  sorry

end minimum_nickels_needed_l1378_137827


namespace number_puzzle_solution_l1378_137833

theorem number_puzzle_solution : ∃ x : ℝ, 12 * x = x + 198 ∧ x = 18 := by
  sorry

end number_puzzle_solution_l1378_137833


namespace complement_intersection_equals_set_l1378_137848

-- Define the universal set U
def U : Finset Nat := {1,2,3,4,5,6,7,8}

-- Define set A
def A : Finset Nat := {1,4,6}

-- Define set B
def B : Finset Nat := {4,5,7}

-- Theorem statement
theorem complement_intersection_equals_set :
  (U \ A) ∩ (U \ B) = {2,3,8} := by sorry

end complement_intersection_equals_set_l1378_137848


namespace cone_lateral_surface_area_l1378_137858

/-- The lateral surface area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_surface_area : 
  ∀ (r h l : ℝ) (S : ℝ),
    r = 3 →
    h = 4 →
    l^2 = r^2 + h^2 →
    S = π * r * l →
    S = 15 * π := by
  sorry

end cone_lateral_surface_area_l1378_137858


namespace nine_possible_H_values_l1378_137846

/-- A function that represents the number formed by digits E, F, G, G, F --/
def EFGGF (E F G : Nat) : Nat := 10000 * E + 1000 * F + 100 * G + 10 * G + F

/-- A function that represents the number formed by digits F, G, E, E, H --/
def FGEEH (F G E H : Nat) : Nat := 10000 * F + 1000 * G + 100 * E + 10 * E + H

/-- A function that represents the number formed by digits H, F, H, H, H --/
def HFHHH (H F : Nat) : Nat := 10000 * H + 1000 * F + 100 * H + 10 * H + H

/-- The main theorem stating that there are exactly 9 possible values for H --/
theorem nine_possible_H_values (E F G H : Nat) :
  (E < 10 ∧ F < 10 ∧ G < 10 ∧ H < 10) →  -- E, F, G, H are digits
  (E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H) →  -- E, F, G, H are distinct
  (EFGGF E F G + FGEEH F G E H = HFHHH H F) →  -- The addition equation
  (∃! (s : Finset Nat), s.card = 9 ∧ ∀ h, h ∈ s ↔ ∃ E F G, EFGGF E F G + FGEEH F G E h = HFHHH h F) :=
by sorry


end nine_possible_H_values_l1378_137846


namespace base_number_proof_l1378_137849

theorem base_number_proof (n : ℕ) (x : ℝ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = x^18) 
  (h2 : n = 17) : 
  x = 2 := by
sorry

end base_number_proof_l1378_137849


namespace checkers_games_theorem_l1378_137826

theorem checkers_games_theorem (games_friend1 games_friend2 : ℕ) 
  (h1 : games_friend1 = 25) 
  (h2 : games_friend2 = 17) : 
  (∃ (x y z : ℕ), x + z = games_friend1 ∧ y + z = games_friend2 ∧ x + y = 34) ∧ 
  (¬ ∃ (x y z : ℕ), x + z = games_friend1 ∧ y + z = games_friend2 ∧ x + y = 35) ∧
  (¬ ∃ (x y z : ℕ), x + z = games_friend1 ∧ y + z = games_friend2 ∧ x + y = 56) :=
by sorry

#check checkers_games_theorem

end checkers_games_theorem_l1378_137826


namespace finite_solutions_of_system_l1378_137839

theorem finite_solutions_of_system (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ∃ (S : Finset (ℤ × ℤ × ℤ × ℤ)), ∀ (x y z w : ℤ),
    x * y + z * w = a ∧ x * z + y * w = b → (x, y, z, w) ∈ S :=
sorry

end finite_solutions_of_system_l1378_137839


namespace solution_range_l1378_137803

theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ (2 * x + m) / (x - 1) = 1) → m > -1 := by
  sorry

end solution_range_l1378_137803


namespace domino_arrangements_count_l1378_137805

/-- Represents a grid with width and height -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a domino with length and width -/
structure Domino :=
  (length : ℕ)
  (width : ℕ)

/-- Calculates the number of distinct arrangements of dominoes on a grid -/
def count_arrangements (g : Grid) (d : Domino) (num_dominoes : ℕ) : ℕ :=
  Nat.choose (g.width + g.height - 2) (g.width - 1)

/-- Theorem stating the number of distinct arrangements -/
theorem domino_arrangements_count (g : Grid) (d : Domino) (num_dominoes : ℕ) :
  g.width = 6 →
  g.height = 4 →
  d.length = 2 →
  d.width = 1 →
  num_dominoes = 5 →
  count_arrangements g d num_dominoes = 56 :=
by sorry

end domino_arrangements_count_l1378_137805


namespace rectangular_box_problem_l1378_137889

theorem rectangular_box_problem (m n r : ℕ) (hm : m > 0) (hn : n > 0) (hr : r > 0)
  (h_order : m ≤ n ∧ n ≤ r) (h_equation : (m-2)*(n-2)*(r-2) + 4*((m-2) + (n-2) + (r-2)) - 
  2*((m-2)*(n-2) + (m-2)*(r-2) + (n-2)*(r-2)) = 1985) :
  (m = 5 ∧ n = 7 ∧ r = 663) ∨
  (m = 5 ∧ n = 5 ∧ r = 1981) ∨
  (m = 3 ∧ n = 3 ∧ r = 1981) ∨
  (m = 1 ∧ n = 7 ∧ r = 399) ∨
  (m = 1 ∧ n = 3 ∧ r = 1987) := by
  sorry

end rectangular_box_problem_l1378_137889


namespace greatest_divisor_four_consecutive_integers_l1378_137806

theorem greatest_divisor_four_consecutive_integers :
  ∃ (d : ℕ), d = 12 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (m * (m + 1) * (m + 2) * (m + 3)))) :=
by sorry

end greatest_divisor_four_consecutive_integers_l1378_137806


namespace polynomial_coefficient_B_l1378_137801

theorem polynomial_coefficient_B (A : ℤ) :
  ∃ (r₁ r₂ r₃ r₄ : ℕ+),
    (r₁ : ℤ) + r₂ + r₃ + r₄ = 7 ∧
    ∀ (z : ℂ), z^4 - 7*z^3 + A*z^2 + (-12)*z + 24 = (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) := by
  sorry

end polynomial_coefficient_B_l1378_137801


namespace midpoint_linear_combination_l1378_137844

/-- Given two points A and B in the plane, prove that if C is their midpoint,
    then a specific linear combination of C's coordinates equals -21. -/
theorem midpoint_linear_combination (A B : ℝ × ℝ) (h : A = (20, 9) ∧ B = (4, 6)) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  2 * C.1 - 6 * C.2 = -21 := by
  sorry

#check midpoint_linear_combination

end midpoint_linear_combination_l1378_137844


namespace sum_in_base7_l1378_137896

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- The sum of 666₇, 66₇, and 6₇ in base 7 is 1400₇ -/
theorem sum_in_base7 : 
  base10ToBase7 (base7ToBase10 666 + base7ToBase10 66 + base7ToBase10 6) = 1400 := by
  sorry

end sum_in_base7_l1378_137896


namespace max_ab_max_expression_min_sum_l1378_137871

-- Define the conditions
def is_valid_pair (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 1

-- Theorem 1: Maximum value of ab
theorem max_ab (a b : ℝ) (h : is_valid_pair a b) :
  a * b ≤ 1/4 ∧ ∃ (a₀ b₀ : ℝ), is_valid_pair a₀ b₀ ∧ a₀ * b₀ = 1/4 :=
sorry

-- Theorem 2: Maximum value of 4a - 1/(4b)
theorem max_expression (a b : ℝ) (h : is_valid_pair a b) :
  4*a - 1/(4*b) ≤ 2 ∧ ∃ (a₀ b₀ : ℝ), is_valid_pair a₀ b₀ ∧ 4*a₀ - 1/(4*b₀) = 2 :=
sorry

-- Theorem 3: Minimum value of 1/a + 2/b
theorem min_sum (a b : ℝ) (h : is_valid_pair a b) :
  1/a + 2/b ≥ 3 + 2*Real.sqrt 2 ∧ ∃ (a₀ b₀ : ℝ), is_valid_pair a₀ b₀ ∧ 1/a₀ + 2/b₀ = 3 + 2*Real.sqrt 2 :=
sorry

end max_ab_max_expression_min_sum_l1378_137871


namespace line_intersecting_ellipse_l1378_137840

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define a line by its slope and y-intercept
def line_equation (k b : ℝ) (x y : ℝ) : Prop := y = k * x + b

-- Define what it means for a point to be a midpoint of two other points
def is_midpoint (x₁ y₁ x₂ y₂ x y : ℝ) : Prop := 
  x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2

theorem line_intersecting_ellipse (x₁ y₁ x₂ y₂ : ℝ) :
  is_on_ellipse x₁ y₁ → 
  is_on_ellipse x₂ y₂ → 
  is_midpoint x₁ y₁ x₂ y₂ 1 (1/2) →
  ∃ k b, line_equation k b x₁ y₁ ∧ line_equation k b x₂ y₂ ∧ k = -1 ∧ b = 2 :=
sorry

end line_intersecting_ellipse_l1378_137840


namespace sum_greater_than_four_l1378_137845

theorem sum_greater_than_four (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (h : 1/a + 1/b = 1) :
  a + b > 4 := by
sorry

end sum_greater_than_four_l1378_137845


namespace perpendicular_vector_scalar_l1378_137860

/-- Given two vectors a and b in ℝ², if a + x*b is perpendicular to b, then x = -2/5 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (x : ℝ) 
  (ha : a = (3, 4))
  (hb : b = (2, -1))
  (h_perp : (a.1 + x * b.1, a.2 + x * b.2) • b = 0) :
  x = -2/5 := by
  sorry

end perpendicular_vector_scalar_l1378_137860


namespace vertical_angles_equal_l1378_137802

-- Define a line as a type
def Line := ℝ → ℝ → Prop

-- Define an angle as a pair of lines
def Angle := Line × Line

-- Define vertical angles
def VerticalAngles (a b : Angle) : Prop :=
  ∃ (l1 l2 : Line), l1 ≠ l2 ∧ 
    ((a.1 = l1 ∧ a.2 = l2) ∨ (a.1 = l2 ∧ a.2 = l1)) ∧
    ((b.1 = l1 ∧ b.2 = l2) ∨ (b.1 = l2 ∧ b.2 = l1))

-- Define angle measure
def AngleMeasure (a : Angle) : ℝ := sorry

-- Theorem: Vertical angles are always equal
theorem vertical_angles_equal (a b : Angle) :
  VerticalAngles a b → AngleMeasure a = AngleMeasure b := by
  sorry

end vertical_angles_equal_l1378_137802


namespace quadratic_equation_has_real_root_l1378_137825

theorem quadratic_equation_has_real_root (a b : ℝ) : 
  ∃ x : ℝ, x^2 + a*x + b = 0 := by
sorry

end quadratic_equation_has_real_root_l1378_137825


namespace quadratic_roots_condition_l1378_137830

theorem quadratic_roots_condition (p q : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 2 ∧ 
   x₁^2 + 2*p*x₁ + q = 0 ∧ x₂^2 + 2*p*x₂ + q = 0) ↔ 
  (q > 0 ∧ p < -2) :=
sorry

end quadratic_roots_condition_l1378_137830


namespace no_solution_to_inequalities_l1378_137851

theorem no_solution_to_inequalities :
  ¬ ∃ x : ℝ, (4 * x + 2 < (x + 3)^2) ∧ ((x + 3)^2 < 8 * x + 1) := by
  sorry

end no_solution_to_inequalities_l1378_137851


namespace incorrect_conjunction_falsehood_l1378_137885

theorem incorrect_conjunction_falsehood : ¬(∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)) := by
  sorry

end incorrect_conjunction_falsehood_l1378_137885


namespace det_E_l1378_137861

/-- A 2x2 matrix representing a dilation centered at the origin with scale factor 5 -/
def E : Matrix (Fin 2) (Fin 2) ℝ := !![5, 0; 0, 5]

/-- Theorem: The determinant of E is 25 -/
theorem det_E : Matrix.det E = 25 := by sorry

end det_E_l1378_137861


namespace sqrt_x_plus_one_real_l1378_137841

theorem sqrt_x_plus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by sorry

end sqrt_x_plus_one_real_l1378_137841


namespace complement_intersection_A_B_union_B_C_implies_a_bound_l1378_137895

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | 2*x + a > 0}

-- Theorem for part (1)
theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {x | x < 2 ∨ x ≥ 3} := by sorry

-- Theorem for part (2)
theorem union_B_C_implies_a_bound (a : ℝ) :
  B ∪ C a = C a → a > -4 := by sorry

end complement_intersection_A_B_union_B_C_implies_a_bound_l1378_137895


namespace students_per_table_unchanged_l1378_137820

/-- Proves that the number of students per table remains the same when evenly dividing the total number of students across all tables. -/
theorem students_per_table_unchanged 
  (initial_students_per_table : ℝ) 
  (num_tables : ℝ) 
  (h1 : initial_students_per_table = 6.0)
  (h2 : num_tables = 34.0) :
  let total_students := initial_students_per_table * num_tables
  total_students / num_tables = initial_students_per_table := by
  sorry

end students_per_table_unchanged_l1378_137820


namespace car_speed_percentage_increase_l1378_137859

/-- Proves that given two cars driving toward each other, with the first car traveling at 100 km/h,
    a distance of 720 km between them, and meeting after 4 hours, the percentage increase in the
    speed of the first car compared to the second car is 25%. -/
theorem car_speed_percentage_increase
  (speed_first : ℝ)
  (distance : ℝ)
  (time : ℝ)
  (h1 : speed_first = 100)
  (h2 : distance = 720)
  (h3 : time = 4)
  (h4 : speed_first * time + (distance / time) * time = distance) :
  (speed_first - (distance / time)) / (distance / time) * 100 = 25 :=
by sorry

end car_speed_percentage_increase_l1378_137859


namespace point_in_fourth_quadrant_l1378_137855

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the original point A -/
def A (m : ℝ) : Point :=
  { x := -5 * m, y := 2 * m - 1 }

/-- Moves a point up by a given amount -/
def moveUp (p : Point) (amount : ℝ) : Point :=
  { x := p.x, y := p.y + amount }

/-- Checks if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Main theorem -/
theorem point_in_fourth_quadrant (m : ℝ) :
  (moveUp (A m) 3).y = 0 → isInFourthQuadrant (A m) :=
by sorry

end point_in_fourth_quadrant_l1378_137855


namespace no_real_roots_when_m_is_one_m_range_for_specified_root_intervals_l1378_137864

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x + 2*m + 1

-- Theorem 1: When m = 1, the equation has no real roots
theorem no_real_roots_when_m_is_one :
  ∀ x : ℝ, f 1 x ≠ 0 := by sorry

-- Theorem 2: Range of m when roots are in specified intervals
theorem m_range_for_specified_root_intervals :
  (∃ x y : ℝ, x ∈ Set.Ioo (-1) 0 ∧ y ∈ Set.Ioo 1 2 ∧ f m x = 0 ∧ f m y = 0) ↔
  m ∈ Set.Ioo (-5/6) (-1/2) := by sorry

end no_real_roots_when_m_is_one_m_range_for_specified_root_intervals_l1378_137864


namespace y_intercept_of_line_l1378_137875

/-- The y-intercept of the line 3x - 4y = 12 is -3 -/
theorem y_intercept_of_line (x y : ℝ) : 3 * x - 4 * y = 12 → x = 0 → y = -3 := by
  sorry

end y_intercept_of_line_l1378_137875


namespace min_value_absolute_sum_l1378_137852

theorem min_value_absolute_sum (x : ℝ) : 
  ∃ (m : ℝ), (∀ x, |x - 1| + |x + 2| ≥ m) ∧ (∃ x, |x - 1| + |x + 2| = m) ∧ m = 3 := by
  sorry

end min_value_absolute_sum_l1378_137852


namespace quadratic_roots_property_l1378_137870

theorem quadratic_roots_property (a b : ℝ) : 
  a ≠ b ∧ 
  a^2 + 3*a - 5 = 0 ∧ 
  b^2 + 3*b - 5 = 0 → 
  a^2 + 3*a*b + a - 2*b = -4 := by sorry

end quadratic_roots_property_l1378_137870


namespace cricket_team_matches_l1378_137808

/-- Proves that the total number of matches played by a cricket team in August is 250,
    given the initial and final winning percentages and the number of matches won during a winning streak. -/
theorem cricket_team_matches : 
  ∀ (initial_win_percent : ℝ) (final_win_percent : ℝ) (streak_wins : ℕ),
    initial_win_percent = 0.20 →
    final_win_percent = 0.52 →
    streak_wins = 80 →
    ∃ (total_matches : ℕ),
      total_matches = 250 ∧
      (initial_win_percent * total_matches + streak_wins) / total_matches = final_win_percent :=
by sorry

end cricket_team_matches_l1378_137808


namespace arithmetic_computation_l1378_137884

theorem arithmetic_computation : 1 + (6 * 2 - 3 + 5) * 4 / 2 = 29 := by sorry

end arithmetic_computation_l1378_137884


namespace parabola_unique_values_l1378_137807

/-- Parabola passing through (1, 1) and tangent to y = x - 3 at (2, -1) -/
def parabola_conditions (a b c : ℝ) : Prop :=
  -- Passes through (1, 1)
  a + b + c = 1 ∧
  -- Passes through (2, -1)
  4*a + 2*b + c = -1 ∧
  -- Derivative at x = 2 equals slope of y = x - 3
  4*a + b = 1

/-- Theorem stating the unique values of a, b, and c satisfying the conditions -/
theorem parabola_unique_values :
  ∃! (a b c : ℝ), parabola_conditions a b c ∧ a = 3 ∧ b = -11 ∧ c = 9 :=
by sorry

end parabola_unique_values_l1378_137807


namespace shari_walk_distance_l1378_137828

/-- Calculates the distance walked given a constant walking speed, total time, and break time. -/
def distance_walked (speed : ℝ) (total_time : ℝ) (break_time : ℝ) : ℝ :=
  speed * (total_time - break_time)

/-- Proves that walking at 4 miles per hour for 2 hours with a 30-minute break results in 6 miles walked. -/
theorem shari_walk_distance :
  let speed : ℝ := 4
  let total_time : ℝ := 2
  let break_time : ℝ := 0.5
  distance_walked speed total_time break_time = 6 := by sorry

end shari_walk_distance_l1378_137828


namespace time_after_duration_sum_l1378_137800

/-- Represents time on a 12-hour digital clock -/
structure Time12 where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds a duration to a given time and returns the resulting time on a 12-hour clock -/
def addDuration (start : Time12) (hours minutes seconds : Nat) : Time12 :=
  sorry

/-- Converts a Time12 to the sum of its components -/
def timeSum (t : Time12) : Nat :=
  t.hours + t.minutes + t.seconds

theorem time_after_duration_sum :
  let start := Time12.mk 3 0 0
  let result := addDuration start 307 58 59
  timeSum result = 127 := by
  sorry

end time_after_duration_sum_l1378_137800


namespace limit_of_rational_function_at_four_l1378_137824

theorem limit_of_rational_function_at_four :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 4| ∧ |x - 4| < δ →
    |((x^2 - 2*x - 8) / (x - 4)) - 6| < ε := by
  sorry

end limit_of_rational_function_at_four_l1378_137824


namespace apple_boxes_theorem_l1378_137809

/-- Calculates the number of boxes of apples after removing rotten ones -/
def calculate_apple_boxes (apples_per_crate : ℕ) (num_crates : ℕ) (rotten_apples : ℕ) (apples_per_box : ℕ) : ℕ :=
  ((apples_per_crate * num_crates) - rotten_apples) / apples_per_box

/-- Theorem: Given the problem conditions, the number of boxes of apples is 100 -/
theorem apple_boxes_theorem :
  calculate_apple_boxes 180 12 160 20 = 100 := by
  sorry

end apple_boxes_theorem_l1378_137809


namespace magazine_circulation_ratio_l1378_137814

/-- Given a magazine's circulation data, proves the ratio of circulation in 1961 to total circulation from 1961-1970 -/
theorem magazine_circulation_ratio 
  (avg_circulation : ℝ) -- Average yearly circulation for 1962-1970
  (h1 : avg_circulation > 0) -- Assumption that average circulation is positive
  : (4 * avg_circulation) / (4 * avg_circulation + 9 * avg_circulation) = 4 / 13 := by
  sorry

end magazine_circulation_ratio_l1378_137814


namespace octagon_area_l1378_137897

/-- The area of a regular octagon inscribed in a circle with area 256π -/
theorem octagon_area (circle_area : ℝ) (h : circle_area = 256 * Real.pi) :
  ∃ (octagon_area : ℝ), octagon_area = 1024 * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

end octagon_area_l1378_137897


namespace correct_annual_take_home_pay_l1378_137837

def annual_take_home_pay (hourly_rate : ℝ) (regular_hours_per_week : ℝ) (weeks_per_year : ℝ)
  (overtime_hours_per_quarter : ℝ) (overtime_rate_multiplier : ℝ)
  (federal_tax_rate_1 : ℝ) (federal_tax_threshold_1 : ℝ)
  (federal_tax_rate_2 : ℝ) (federal_tax_threshold_2 : ℝ)
  (state_tax_rate : ℝ) (unemployment_insurance_rate : ℝ)
  (unemployment_insurance_threshold : ℝ) (social_security_rate : ℝ)
  (social_security_threshold : ℝ) : ℝ :=
  sorry

theorem correct_annual_take_home_pay :
  annual_take_home_pay 15 40 52 20 1.5 0.1 10000 0.12 30000 0.05 0.01 7000 0.062 142800 = 25474 :=
by sorry

end correct_annual_take_home_pay_l1378_137837


namespace not_always_true_from_false_l1378_137812

-- Define a proposition
variable (P Q R : Prop)

-- Define a logical argument
def logical_argument (premises : Prop) (conclusion : Prop) : Prop :=
  premises → conclusion

-- Define soundness of logical derivation
def sound_derivation (arg : Prop → Prop) : Prop :=
  ∀ (X Y : Prop), (X → Y) → (arg X → arg Y)

-- Theorem statement
theorem not_always_true_from_false :
  ∃ (premises conclusion : Prop) (arg : Prop → Prop),
    (¬premises) ∧ 
    (sound_derivation arg) ∧
    (logical_argument premises conclusion) ∧
    (¬conclusion) :=
sorry

end not_always_true_from_false_l1378_137812


namespace system_solution_exists_l1378_137856

theorem system_solution_exists : ∃ (x y : ℝ), 
  (x * Real.sqrt (x * y) + y * Real.sqrt (x * y) = 10) ∧ 
  (x^2 + y^2 = 17) := by
sorry

end system_solution_exists_l1378_137856


namespace exists_points_on_hyperbola_with_midpoint_l1378_137829

/-- The hyperbola equation --/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

/-- Definition of a midpoint --/
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem exists_points_on_hyperbola_with_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    hyperbola x₁ y₁ ∧ 
    hyperbola x₂ y₂ ∧ 
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ :=
sorry

end exists_points_on_hyperbola_with_midpoint_l1378_137829


namespace polynomial_division_theorem_l1378_137813

theorem polynomial_division_theorem (x : ℝ) :
  x^5 - 21*x^3 + 8*x^2 - 17*x + 12 = (x - 3)*(x^4 + 3*x^3 - 12*x^2 - 28*x - 101) + (-201) := by
  sorry

end polynomial_division_theorem_l1378_137813


namespace parallel_condition_l1378_137887

/-- Two lines are parallel if they have the same slope -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of the line (a^2-a)x+y=0 -/
def slope1 (a : ℝ) : ℝ := a^2 - a

/-- The slope of the line 2x+y+1=0 -/
def slope2 : ℝ := 2

theorem parallel_condition (a : ℝ) :
  (a = 2 → parallel (slope1 a) slope2) ∧
  (∃ b : ℝ, b ≠ 2 ∧ parallel (slope1 b) slope2) :=
sorry

end parallel_condition_l1378_137887


namespace f_composition_negative_eight_l1378_137843

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then -x^(1/3)
  else x + 2/x - 7

-- State the theorem
theorem f_composition_negative_eight : f (f (-8)) = -4 := by
  sorry

end f_composition_negative_eight_l1378_137843


namespace longer_subsegment_length_l1378_137817

/-- Triangle with sides in ratio 3:4:5 -/
structure Triangle :=
  (a b c : ℝ)
  (ratio : a / b = 3 / 4 ∧ b / c = 4 / 5)

/-- Angle bisector theorem -/
axiom angle_bisector_theorem {t : Triangle} (d : ℝ) :
  d / (t.c - d) = t.a / t.b

/-- Main theorem -/
theorem longer_subsegment_length (t : Triangle) (h : t.c = 15) :
  let d := t.c * (t.a / (t.a + t.b))
  d = 75 / 8 := by sorry

end longer_subsegment_length_l1378_137817


namespace jellybean_count_l1378_137835

/-- The number of blue jellybeans in a jar -/
def blue_jellybeans (total purple orange red : ℕ) : ℕ :=
  total - (purple + orange + red)

/-- Theorem: In a jar with 200 total jellybeans, 26 purple, 40 orange, and 120 red jellybeans,
    there are 14 blue jellybeans. -/
theorem jellybean_count : blue_jellybeans 200 26 40 120 = 14 := by
  sorry

end jellybean_count_l1378_137835


namespace sin_2theta_value_l1378_137847

theorem sin_2theta_value (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π/2) 
  (h3 : Real.cos (π/4 - θ) * Real.cos (π/4 + θ) = Real.sqrt 2 / 6) : 
  Real.sin (2 * θ) = Real.sqrt 7 / 3 := by
sorry

end sin_2theta_value_l1378_137847


namespace z_profit_share_l1378_137853

/-- Calculates the share of profit for a partner in a business --/
def calculate_profit_share (
  x_capital y_capital z_capital : ℕ)  -- Initial capitals
  (x_months y_months z_months : ℕ)    -- Months of investment
  (total_profit : ℕ)                  -- Total annual profit
  : ℕ :=
  let x_share := x_capital * x_months
  let y_share := y_capital * y_months
  let z_share := z_capital * z_months
  let total_share := x_share + y_share + z_share
  (z_share * total_profit) / total_share

/-- Theorem statement for Z's profit share --/
theorem z_profit_share :
  calculate_profit_share 20000 25000 30000 12 12 7 50000 = 14000 := by
  sorry

end z_profit_share_l1378_137853


namespace adult_admission_price_l1378_137857

/-- Proves that the adult admission price was 60 cents given the conditions -/
theorem adult_admission_price (total_attendance : ℕ) (child_ticket_price : ℕ) 
  (children_attended : ℕ) (total_revenue : ℕ) : ℕ :=
  sorry

end adult_admission_price_l1378_137857


namespace hexagon_largest_angle_l1378_137854

theorem hexagon_largest_angle (a b c d e f : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  b / a = 3 / 2 →
  c / a = 3 / 2 →
  d / a = 2 →
  e / a = 2 →
  f / a = 3 →
  a + b + c + d + e + f = 720 →
  f = 2160 / 11 := by
sorry

end hexagon_largest_angle_l1378_137854


namespace paper_recycling_trees_saved_l1378_137878

theorem paper_recycling_trees_saved 
  (trees_per_tonne : ℕ) 
  (schools : ℕ) 
  (paper_per_school : ℚ) 
  (h1 : trees_per_tonne = 24)
  (h2 : schools = 4)
  (h3 : paper_per_school = 3/4) : 
  ↑schools * paper_per_school * trees_per_tonne = 72 := by
  sorry

end paper_recycling_trees_saved_l1378_137878


namespace solve_equation_l1378_137894

theorem solve_equation (x : ℝ) : 
  5 * x^(1/3) - 3 * (x / x^(2/3)) = 9 + x^(1/3) ↔ x = 729 := by
  sorry

end solve_equation_l1378_137894


namespace f_of_3_eq_3_l1378_137892

/-- The exponent in the function definition -/
def n : ℕ := 2008

/-- The function f(x) is defined implicitly by this equation -/
def f_equation (f : ℝ → ℝ) (x : ℝ) : Prop :=
  (x^(3^n - 1) - 1) * f x = 
    (List.range n).foldl (λ acc i => acc * (x^(3^i) + 1)) (x + 1) + (x^2 - 1) - 1

/-- The theorem stating that f(3) = 3 -/
theorem f_of_3_eq_3 (f : ℝ → ℝ) (hf : ∀ x, f_equation f x) : f 3 = 3 := by
  sorry

end f_of_3_eq_3_l1378_137892


namespace die_roll_prime_probability_l1378_137868

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def sum_is_prime (x y : ℕ) : Prop := is_prime (x + y)

def count_prime_sums : ℕ := 22

def total_outcomes : ℕ := 48

theorem die_roll_prime_probability :
  (count_prime_sums : ℚ) / total_outcomes = 11 / 24 := by sorry

end die_roll_prime_probability_l1378_137868


namespace same_terminal_side_l1378_137823

theorem same_terminal_side (k : ℤ) : 
  let angles : List ℝ := [-5*π/3, 2*π/3, 4*π/3, 5*π/3]
  let target : ℝ := -π/3
  let same_side (α : ℝ) : Prop := ∃ n : ℤ, α = 2*π*n + target
  ∀ α ∈ angles, same_side α ↔ α = 5*π/3 :=
by sorry

end same_terminal_side_l1378_137823


namespace expression_value_l1378_137877

theorem expression_value (a : ℝ) (h : a = 1/3) : 
  (3 * a⁻¹ + a⁻¹ / 3) / (2 * a) = 15 := by sorry

end expression_value_l1378_137877


namespace problem_1_and_2_l1378_137882

theorem problem_1_and_2 :
  (1/2 * Real.sqrt 24 - Real.sqrt 3 * Real.sqrt 2 = 0) ∧
  ((2 * Real.sqrt 3 + 3 * Real.sqrt 2)^2 = 30 + 12 * Real.sqrt 6) := by
  sorry

end problem_1_and_2_l1378_137882


namespace custom_operation_equality_l1378_137832

/-- The custom operation ⊗ -/
def otimes (a b : ℝ) : ℝ := (a - b) ^ 2

/-- Theorem stating the equality for the given expression -/
theorem custom_operation_equality (x y z : ℝ) : 
  otimes ((x + z) ^ 2) ((z + y) ^ 2) = (x^2 + 2*x*z - y^2 - 2*z*y) ^ 2 := by
  sorry

end custom_operation_equality_l1378_137832


namespace employee_payment_l1378_137838

/-- Given two employees X and Y with a total payment of 550 units,
    where X is paid 120% of Y's payment, prove that Y is paid 250 units. -/
theorem employee_payment (x y : ℝ) 
  (total : x + y = 550)
  (ratio : x = 1.2 * y) : 
  y = 250 := by
  sorry

end employee_payment_l1378_137838


namespace ara_height_ara_current_height_ara_height_is_59_l1378_137804

theorem ara_height (shea_initial : ℝ) (shea_final : ℝ) (ara_initial : ℝ) : ℝ :=
  let shea_growth := shea_final - shea_initial
  let ara_growth := shea_growth / 3
  ara_initial + ara_growth

theorem ara_current_height : ℝ :=
  let shea_final := 64
  let shea_initial := shea_final / 1.25
  let ara_initial := shea_initial + 4
  ara_height shea_initial shea_final ara_initial

theorem ara_height_is_59 : ⌊ara_current_height⌋ = 59 := by
  sorry

end ara_height_ara_current_height_ara_height_is_59_l1378_137804


namespace opposite_of_nine_l1378_137822

theorem opposite_of_nine : -(9 : ℤ) = -9 := by sorry

end opposite_of_nine_l1378_137822


namespace oil_production_fraction_l1378_137893

def initial_concentration : ℝ := 0.02
def first_replacement : ℝ := 0.03
def second_replacement : ℝ := 0.015

theorem oil_production_fraction (x : ℝ) 
  (hx_pos : x > 0)
  (hx_le_one : x ≤ 1)
  (h_first_replacement : initial_concentration * (1 - x) + first_replacement * x = initial_concentration + x * (first_replacement - initial_concentration))
  (h_second_replacement : (initial_concentration + x * (first_replacement - initial_concentration)) * (1 - x) + second_replacement * x = initial_concentration) :
  x = 1/2 := by
sorry

end oil_production_fraction_l1378_137893


namespace jacks_books_l1378_137880

/-- Calculates the number of books in a stack given the stack thickness,
    pages per inch, and pages per book. -/
def number_of_books (stack_thickness : ℕ) (pages_per_inch : ℕ) (pages_per_book : ℕ) : ℕ :=
  (stack_thickness * pages_per_inch) / pages_per_book

/-- Theorem stating that Jack's stack of 12 inches with 80 pages per inch
    and 160 pages per book contains 6 books. -/
theorem jacks_books :
  number_of_books 12 80 160 = 6 := by
  sorry

end jacks_books_l1378_137880


namespace diluted_vinegar_concentration_diluted_vinegar_concentration_proof_l1378_137810

/-- Calculates the concentration of a diluted vinegar solution -/
theorem diluted_vinegar_concentration 
  (original_volume : ℝ) 
  (original_concentration : ℝ) 
  (water_added : ℝ) : ℝ :=
  let vinegar_amount := original_volume * (original_concentration / 100)
  let total_volume := original_volume + water_added
  let diluted_concentration := (vinegar_amount / total_volume) * 100
  diluted_concentration

/-- Proves that the diluted vinegar concentration is approximately 7% -/
theorem diluted_vinegar_concentration_proof 
  (original_volume : ℝ) 
  (original_concentration : ℝ) 
  (water_added : ℝ) 
  (h1 : original_volume = 12) 
  (h2 : original_concentration = 36.166666666666664) 
  (h3 : water_added = 50) :
  ∃ ε > 0, |diluted_vinegar_concentration original_volume original_concentration water_added - 7| < ε :=
sorry

end diluted_vinegar_concentration_diluted_vinegar_concentration_proof_l1378_137810


namespace three_roots_implies_a_range_l1378_137883

theorem three_roots_implies_a_range (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ x : ℝ, x^2 = a * Real.exp x ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  0 < a ∧ a < 4 / Real.exp 2 :=
by sorry

end three_roots_implies_a_range_l1378_137883


namespace orangeade_ratio_l1378_137886

def orangeade_problem (orange_juice water_day1 : ℝ) : Prop :=
  let water_day2 := 2 * water_day1
  let price_day1 := 0.30
  let price_day2 := 0.20
  let volume_day1 := orange_juice + water_day1
  let volume_day2 := orange_juice + water_day2
  (volume_day1 * price_day1 = volume_day2 * price_day2) →
  (orange_juice = water_day1)

theorem orangeade_ratio :
  ∀ (orange_juice water_day1 : ℝ),
  orangeade_problem orange_juice water_day1 :=
sorry

end orangeade_ratio_l1378_137886


namespace unique_line_through_point_l1378_137866

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem unique_line_through_point :
  ∃! (a b : ℕ), 
    a > 0 ∧ 
    is_prime b ∧ 
    (6 : ℚ) / a + (5 : ℚ) / b = 1 := by
  sorry

end unique_line_through_point_l1378_137866


namespace eighth_number_with_digit_sum_13_l1378_137874

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that returns whether a natural number has a digit sum of 13 -/
def has_digit_sum_13 (n : ℕ) : Prop := digit_sum n = 13

/-- A function that returns the nth positive integer with digit sum 13 -/
def nth_digit_sum_13 (n : ℕ) : ℕ := sorry

theorem eighth_number_with_digit_sum_13 : nth_digit_sum_13 8 = 148 := by sorry

end eighth_number_with_digit_sum_13_l1378_137874


namespace sum_product_reciprocal_sum_squared_inequality_l1378_137815

theorem sum_product_reciprocal_sum_squared_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a*b + b*c + c*a) * (1/(a+b)^2 + 1/(b+c)^2 + 1/(c+a)^2) ≥ 9/4 := by
  sorry

end sum_product_reciprocal_sum_squared_inequality_l1378_137815


namespace alice_bob_race_difference_l1378_137873

/-- The difference in finish times between two runners in a race. -/
def finish_time_difference (alice_speed bob_speed race_distance : ℝ) : ℝ :=
  bob_speed * race_distance - alice_speed * race_distance

/-- Theorem stating the difference in finish times for Alice and Bob in a 12-mile race. -/
theorem alice_bob_race_difference :
  finish_time_difference 5 7 12 = 24 := by
  sorry

end alice_bob_race_difference_l1378_137873


namespace square_tablecloth_side_length_l1378_137879

-- Define a square tablecloth
structure SquareTablecloth where
  side : ℝ
  area : ℝ
  is_square : area = side * side

-- Theorem statement
theorem square_tablecloth_side_length 
  (tablecloth : SquareTablecloth) 
  (h : tablecloth.area = 5) : 
  tablecloth.side = Real.sqrt 5 := by
  sorry

end square_tablecloth_side_length_l1378_137879


namespace permutation_equality_l1378_137821

-- Define the permutation function
def A (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

-- State the theorem
theorem permutation_equality (n : ℕ) :
  A (2 * n) ^ 3 = 9 * (A n) ^ 3 → n = 14 := by
  sorry

end permutation_equality_l1378_137821


namespace mandy_med_school_acceptances_l1378_137850

theorem mandy_med_school_acceptances
  (total_researched : ℕ)
  (applied_fraction : ℚ)
  (accepted_fraction : ℚ)
  (h1 : total_researched = 96)
  (h2 : applied_fraction = 5 / 8)
  (h3 : accepted_fraction = 3 / 5)
  : ℕ :=
by
  sorry

end mandy_med_school_acceptances_l1378_137850


namespace polar_coordinate_equivalence_l1378_137891

def standard_polar_form (r : ℝ) (θ : ℝ) : Prop :=
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

theorem polar_coordinate_equivalence :
  ∀ (r₁ r₂ θ₁ θ₂ : ℝ),
  r₁ = -3 ∧ θ₁ = 5 * Real.pi / 6 →
  r₂ = 3 ∧ θ₂ = 11 * Real.pi / 6 →
  standard_polar_form r₂ θ₂ →
  (r₁ * (Real.cos θ₁), r₁ * (Real.sin θ₁)) = (r₂ * (Real.cos θ₂), r₂ * (Real.sin θ₂)) :=
by sorry

end polar_coordinate_equivalence_l1378_137891


namespace possible_values_of_a_l1378_137863

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 13*x^3) 
  (h3 : a - b = 2*x) : 
  (a = x + (Real.sqrt 66 * x) / 6) ∨ (a = x - (Real.sqrt 66 * x) / 6) :=
sorry

end possible_values_of_a_l1378_137863


namespace right_triangle_one_two_sqrt_three_l1378_137862

theorem right_triangle_one_two_sqrt_three :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  a = 1 ∧ b = Real.sqrt 3 ∧ c = 2 ∧
  a^2 + b^2 = c^2 ∧
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end right_triangle_one_two_sqrt_three_l1378_137862


namespace shopping_cost_theorem_l1378_137831

/-- Calculates the total cost of Fabian's shopping trip --/
def calculate_shopping_cost (
  apple_price : ℝ)
  (walnut_price : ℝ)
  (orange_price : ℝ)
  (pasta_price : ℝ)
  (sugar_discount : ℝ)
  (orange_discount : ℝ)
  (sales_tax : ℝ) : ℝ :=
  let apple_cost := 5 * apple_price
  let sugar_cost := 3 * (apple_price - sugar_discount)
  let walnut_cost := 0.5 * walnut_price
  let orange_cost := 2 * orange_price * (1 - orange_discount)
  let pasta_cost := 3 * pasta_price
  let total_before_tax := apple_cost + sugar_cost + walnut_cost + orange_cost + pasta_cost
  total_before_tax * (1 + sales_tax)

/-- The theorem stating the total cost of Fabian's shopping --/
theorem shopping_cost_theorem :
  calculate_shopping_cost 2 6 3 1.5 1 0.1 0.05 = 27.20 := by sorry

end shopping_cost_theorem_l1378_137831


namespace square_side_length_l1378_137876

/-- Given a rectangle with sides 9 cm and 16 cm and a square with the same area,
    prove that the side length of the square is 12 cm. -/
theorem square_side_length (rectangle_width : ℝ) (rectangle_length : ℝ) (square_side : ℝ) :
  rectangle_width = 9 →
  rectangle_length = 16 →
  rectangle_width * rectangle_length = square_side * square_side →
  square_side = 12 := by
sorry

end square_side_length_l1378_137876


namespace train_speed_train_speed_is_60_kmph_l1378_137819

/-- The speed of a train given its length, time to pass a person, and the person's speed in the opposite direction -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let man_speed_mps := man_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_mps := relative_speed - man_speed_mps
  let train_speed_kmph := train_speed_mps * (3600 / 1000)
  train_speed_kmph

/-- The speed of the train is 60 kmph given the specified conditions -/
theorem train_speed_is_60_kmph : 
  train_speed 55 3 6 = 60 := by
  sorry

end train_speed_train_speed_is_60_kmph_l1378_137819


namespace unique_solution_condition_l1378_137898

theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, a * (3 : ℝ)^x + (3 : ℝ)^(-x) = 3) ↔ 
  a ∈ Set.Iic (0 : ℝ) ∪ {9/4} :=
sorry

end unique_solution_condition_l1378_137898


namespace probability_is_one_third_l1378_137816

/-- Line represented by slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The region of interest in the first quadrant -/
def Region (p q r : Line) : Set Point :=
  {pt : Point | 0 ≤ pt.x ∧ 0 ≤ pt.y ∧ 
                pt.y ≤ p.slope * pt.x + p.intercept ∧
                r.intercept < pt.y ∧ 
                q.slope * pt.x + q.intercept < pt.y ∧ 
                pt.y < p.slope * pt.x + p.intercept}

/-- The area of the region of interest -/
noncomputable def areaOfRegion (p q r : Line) : ℝ := sorry

/-- The total area under line p and above x-axis in the first quadrant -/
noncomputable def totalArea (p : Line) : ℝ := sorry

/-- The main theorem stating the probability -/
theorem probability_is_one_third 
  (p : Line) 
  (q : Line) 
  (r : Line) 
  (hp : p.slope = -2 ∧ p.intercept = 8) 
  (hq : q.slope = -3 ∧ q.intercept = 8) 
  (hr : r.slope = 0 ∧ r.intercept = 4) : 
  areaOfRegion p q r / totalArea p = 1/3 := by sorry

end probability_is_one_third_l1378_137816


namespace perfect_squares_difference_99_l1378_137811

theorem perfect_squares_difference_99 :
  ∃! (l : List ℕ), 
    (∀ x ∈ l, ∃ a : ℕ, x = a^2 ∧ ∃ b : ℕ, x + 99 = b^2) ∧ 
    (∀ x : ℕ, (∃ a : ℕ, x = a^2 ∧ ∃ b : ℕ, x + 99 = b^2) → x ∈ l) ∧
    l.length = 3 :=
by sorry

end perfect_squares_difference_99_l1378_137811


namespace nap_time_is_three_hours_nap_time_in_hours_l1378_137872

/-- Calculates the remaining time for a nap given flight duration and time spent on activities -/
def remaining_nap_time (flight_duration : ℕ) (reading_time : ℕ) (movie_time : ℕ)
  (dinner_time : ℕ) (radio_time : ℕ) (game_time : ℕ) : ℕ :=
  flight_duration - (reading_time + movie_time + dinner_time + radio_time + game_time)

/-- Theorem stating that the remaining time for a nap is 3 hours -/
theorem nap_time_is_three_hours :
  remaining_nap_time 680 120 240 30 40 70 = 180 := by
  sorry

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℕ) : ℕ :=
  minutes / 60

/-- Theorem stating that 180 minutes is equal to 3 hours -/
theorem nap_time_in_hours :
  minutes_to_hours (remaining_nap_time 680 120 240 30 40 70) = 3 := by
  sorry

end nap_time_is_three_hours_nap_time_in_hours_l1378_137872


namespace parabola_equation_l1378_137881

/-- A parabola with vertex at the origin and directrix x = -2 has the equation y^2 = 8x -/
theorem parabola_equation (x y : ℝ) : 
  (∀ p : ℝ, p > 0 → 
    (x - p)^2 + y^2 = (x + p)^2 ∧ 
    p = 2) → 
  y^2 = 8*x := by
  sorry

end parabola_equation_l1378_137881
