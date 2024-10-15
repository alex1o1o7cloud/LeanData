import Mathlib

namespace NUMINAMATH_CALUDE_triangle_reciprocal_sum_l3009_300941

/-- For any triangle, the sum of reciprocals of altitudes equals the sum of reciprocals of exradii, which equals the reciprocal of the inradius. -/
theorem triangle_reciprocal_sum (a b c h_a h_b h_c r_a r_b r_c r A p : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ h_a > 0 ∧ h_b > 0 ∧ h_c > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ r > 0 ∧ A > 0 ∧ p > 0)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_area : A = p * r)
  (h_altitude_a : h_a = 2 * A / a)
  (h_altitude_b : h_b = 2 * A / b)
  (h_altitude_c : h_c = 2 * A / c)
  (h_exradius_a : r_a = A / (p - a))
  (h_exradius_b : r_b = A / (p - b))
  (h_exradius_c : r_c = A / (p - c)) :
  1 / h_a + 1 / h_b + 1 / h_c = 1 / r_a + 1 / r_b + 1 / r_c ∧
  1 / h_a + 1 / h_b + 1 / h_c = 1 / r := by sorry

end NUMINAMATH_CALUDE_triangle_reciprocal_sum_l3009_300941


namespace NUMINAMATH_CALUDE_M_subset_N_l3009_300987

def M : Set ℚ := {x | ∃ k : ℤ, x = k / 3 + 1 / 6}
def N : Set ℚ := {x | ∃ k : ℤ, x = k / 6 + 1 / 3}

theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l3009_300987


namespace NUMINAMATH_CALUDE_twice_x_minus_three_l3009_300963

theorem twice_x_minus_three (x : ℝ) : 2 * x - 3 = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_twice_x_minus_three_l3009_300963


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l3009_300955

theorem defective_shipped_percentage 
  (total_units : ℝ) 
  (defective_rate : ℝ) 
  (shipped_rate : ℝ) 
  (h1 : defective_rate = 0.1) 
  (h2 : shipped_rate = 0.05) : 
  defective_rate * shipped_rate * 100 = 0.5 := by
sorry

end NUMINAMATH_CALUDE_defective_shipped_percentage_l3009_300955


namespace NUMINAMATH_CALUDE_negation_existential_quadratic_l3009_300990

theorem negation_existential_quadratic (x : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_existential_quadratic_l3009_300990


namespace NUMINAMATH_CALUDE_arthur_summer_reading_l3009_300982

theorem arthur_summer_reading (book1_pages book2_pages : ℕ) 
  (book1_read_percent : ℚ) (book2_read_fraction : ℚ) (pages_left : ℕ) : 
  book1_pages = 500 → 
  book2_pages = 1000 → 
  book1_read_percent = 80 / 100 → 
  book2_read_fraction = 1 / 5 → 
  pages_left = 200 → 
  (book1_pages * book1_read_percent).floor + 
  (book2_pages * book2_read_fraction).floor + 
  pages_left = 800 := by
  sorry

end NUMINAMATH_CALUDE_arthur_summer_reading_l3009_300982


namespace NUMINAMATH_CALUDE_rug_overlap_problem_l3009_300973

theorem rug_overlap_problem (total_rug_area : ℝ) (covered_floor_area : ℝ) (two_layer_area : ℝ)
  (h1 : total_rug_area = 200)
  (h2 : covered_floor_area = 140)
  (h3 : two_layer_area = 24) :
  ∃ (three_layer_area : ℝ),
    three_layer_area = 18 ∧
    total_rug_area - covered_floor_area = two_layer_area + 2 * three_layer_area :=
by sorry

end NUMINAMATH_CALUDE_rug_overlap_problem_l3009_300973


namespace NUMINAMATH_CALUDE_min_distance_to_point_l3009_300958

theorem min_distance_to_point (x y : ℝ) (h : 6 * x + 8 * y - 1 = 0) :
  ∃ (min : ℝ), min = 7 / 10 ∧ ∀ (x' y' : ℝ), 6 * x' + 8 * y' - 1 = 0 →
    Real.sqrt (x' ^ 2 + y' ^ 2 - 2 * y' + 1) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_point_l3009_300958


namespace NUMINAMATH_CALUDE_percentage_difference_l3009_300940

theorem percentage_difference (T : ℝ) (h1 : T > 0) : 
  let F := 0.70 * T
  let S := 0.90 * F
  (T - S) / T = 0.37 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l3009_300940


namespace NUMINAMATH_CALUDE_largest_three_digit_square_base_7_l3009_300952

/-- The largest integer whose square has exactly 3 digits in base 7 -/
def M : ℕ := 18

/-- Conversion of a natural number to its base 7 representation -/
def to_base_7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem largest_three_digit_square_base_7 :
  (M * M ≥ 7^2) ∧ 
  (M * M < 7^3) ∧ 
  (∀ n : ℕ, n > M → n * n ≥ 7^3) ∧
  (to_base_7 M = [2, 4]) :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_square_base_7_l3009_300952


namespace NUMINAMATH_CALUDE_parabola_two_axis_intersections_l3009_300917

/-- A parabola has only two common points with the coordinate axes if and only if m is 0 or 8 --/
theorem parabola_two_axis_intersections (m : ℝ) : 
  (∃! x y : ℝ, (y = 2*x^2 + 8*x + m ∧ (x = 0 ∨ y = 0)) ∧ 
   (∃ x' y' : ℝ, (y' = 2*x'^2 + 8*x' + m ∧ (x' = 0 ∨ y' = 0)) ∧ (x ≠ x' ∨ y ≠ y'))) ↔ 
  (m = 0 ∨ m = 8) :=
sorry

end NUMINAMATH_CALUDE_parabola_two_axis_intersections_l3009_300917


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_ellipse_standard_equation_l3009_300978

-- Problem 1: Hyperbola
def hyperbola_equation (e : ℝ) (vertex_distance : ℝ) : Prop :=
  e = 5/3 ∧ vertex_distance = 6 →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2/9 - y^2/16 = 1)

theorem hyperbola_standard_equation :
  hyperbola_equation (5/3) 6 :=
sorry

-- Problem 2: Ellipse
def ellipse_equation (major_minor_ratio : ℝ) (point : ℝ × ℝ) : Prop :=
  major_minor_ratio = 3 ∧ point = (3, 0) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/9 + y^2 = 1)) ∨
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, y^2/a^2 + x^2/b^2 = 1 ↔ y^2/81 + x^2/9 = 1))

theorem ellipse_standard_equation :
  ellipse_equation 3 (3, 0) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_ellipse_standard_equation_l3009_300978


namespace NUMINAMATH_CALUDE_neil_initial_games_l3009_300921

theorem neil_initial_games (henry_initial : ℕ) (henry_gave : ℕ) (henry_neil_ratio : ℕ) :
  henry_initial = 33 →
  henry_gave = 5 →
  henry_neil_ratio = 4 →
  henry_initial - henry_gave = henry_neil_ratio * (2 + henry_gave) :=
by
  sorry

end NUMINAMATH_CALUDE_neil_initial_games_l3009_300921


namespace NUMINAMATH_CALUDE_domain_of_function_1_l3009_300951

theorem domain_of_function_1 (x : ℝ) : 
  Set.univ = {x : ℝ | ∃ y : ℝ, y = (2 * x^2 - 1) / (x^2 + 3)} :=
sorry

end NUMINAMATH_CALUDE_domain_of_function_1_l3009_300951


namespace NUMINAMATH_CALUDE_sum_product_ratio_theorem_l3009_300934

theorem sum_product_ratio_theorem (x y z : ℝ) (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) (hsum : x + y + z = 12) :
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = (144 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) := by
  sorry

end NUMINAMATH_CALUDE_sum_product_ratio_theorem_l3009_300934


namespace NUMINAMATH_CALUDE_assembly_line_theorem_l3009_300965

/-- Represents the number of tasks in the assembly line -/
def num_tasks : ℕ := 6

/-- Represents the number of freely arrangeable tasks -/
def num_free_tasks : ℕ := 5

/-- The number of ways to arrange the assembly line -/
def assembly_line_arrangements : ℕ := Nat.factorial num_free_tasks

/-- Theorem stating the number of ways to arrange the assembly line -/
theorem assembly_line_theorem : 
  assembly_line_arrangements = 120 := by sorry

end NUMINAMATH_CALUDE_assembly_line_theorem_l3009_300965


namespace NUMINAMATH_CALUDE_inequality_proof_l3009_300909

theorem inequality_proof (a b : ℝ) (h1 : b > a) (h2 : a > 0) : 2 * a + b / 2 ≥ 2 * Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3009_300909


namespace NUMINAMATH_CALUDE_music_listening_time_l3009_300997

/-- Given a music tempo and total beats heard per week, calculate the hours of music listened to per day. -/
theorem music_listening_time (tempo : ℕ) (total_beats_per_week : ℕ) : 
  tempo = 200 → total_beats_per_week = 168000 → 
  (total_beats_per_week / 7 / tempo * 60) / 60 = 2 := by
  sorry

#check music_listening_time

end NUMINAMATH_CALUDE_music_listening_time_l3009_300997


namespace NUMINAMATH_CALUDE_square_decomposition_l3009_300953

theorem square_decomposition (a b c k : ℕ) (n : ℕ) (h1 : c^2 = n * a^2 + n * b^2) 
  (h2 : (5*k)^2 = (4*k)^2 + (3*k)^2) (h3 : n = k^2) (h4 : n = 9) : c = 15 :=
sorry

end NUMINAMATH_CALUDE_square_decomposition_l3009_300953


namespace NUMINAMATH_CALUDE_complex_expression_equality_l3009_300919

theorem complex_expression_equality : (2 - Complex.I)^2 - (1 + 3 * Complex.I) = 2 - 7 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l3009_300919


namespace NUMINAMATH_CALUDE_peanut_butter_cans_l3009_300996

theorem peanut_butter_cans (n : ℕ) (initial_avg_price remaining_avg_price returned_avg_price : ℚ)
  (h1 : initial_avg_price = 365/10)
  (h2 : remaining_avg_price = 30/1)
  (h3 : returned_avg_price = 495/10)
  (h4 : n * initial_avg_price = (n - 2) * remaining_avg_price + 2 * returned_avg_price) :
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_cans_l3009_300996


namespace NUMINAMATH_CALUDE_first_term_of_sequence_l3009_300918

/-- Given a sequence of points scored in a game where the second to sixth terms
    are 3, 5, 8, 12, and 17, and the differences between consecutive terms
    form an arithmetic sequence, prove that the first term of the sequence is 2. -/
theorem first_term_of_sequence (a : ℕ → ℕ) : 
  a 2 = 3 ∧ a 3 = 5 ∧ a 4 = 8 ∧ a 5 = 12 ∧ a 6 = 17 ∧ 
  (∃ d : ℕ, ∀ n : ℕ, n ≥ 2 → a (n+1) - a n = d + n - 2) →
  a 1 = 2 :=
by sorry

end NUMINAMATH_CALUDE_first_term_of_sequence_l3009_300918


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_l3009_300988

theorem infinite_geometric_series_sum : 
  let a : ℝ := 1/4  -- first term
  let r : ℝ := 1/2  -- common ratio
  let S : ℝ := ∑' n, a * r^n  -- infinite sum
  S = 1/2 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_l3009_300988


namespace NUMINAMATH_CALUDE_triangle_circle_area_ratio_l3009_300968

-- Define a right-angled isosceles triangle
structure RightIsoscelesTriangle where
  leg : ℝ
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = Real.sqrt 2 * leg

-- Define a circle
structure Circle where
  radius : ℝ

-- Define the theorem
theorem triangle_circle_area_ratio 
  (t : RightIsoscelesTriangle) 
  (c : Circle) 
  (perimeter_eq : 2 * t.leg + t.hypotenuse = 2 * Real.pi * c.radius) : 
  (t.leg^2 / 2) / (Real.pi * c.radius^2) = Real.pi * (3 - 2 * Real.sqrt 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_circle_area_ratio_l3009_300968


namespace NUMINAMATH_CALUDE_opponents_team_points_l3009_300912

-- Define the points for each player
def max_points : ℕ := 5
def dulce_points : ℕ := 3

-- Define Val's points as twice the combined points of Max and Dulce
def val_points : ℕ := 2 * (max_points + dulce_points)

-- Define the total points of their team
def team_points : ℕ := max_points + dulce_points + val_points

-- Define the point difference between the teams
def point_difference : ℕ := 16

-- Theorem to prove
theorem opponents_team_points : 
  team_points + point_difference = 40 := by sorry

end NUMINAMATH_CALUDE_opponents_team_points_l3009_300912


namespace NUMINAMATH_CALUDE_nancy_wednesday_pots_l3009_300930

/-- The number of clay pots Nancy created on each day of the week --/
structure ClayPots where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- The conditions of Nancy's clay pot creation --/
def nancy_pots : ClayPots where
  monday := 12
  tuesday := 2 * 12
  wednesday := 50 - (12 + 2 * 12)

/-- Theorem stating that Nancy created 14 clay pots on Wednesday --/
theorem nancy_wednesday_pots : nancy_pots.wednesday = 14 := by
  sorry

#eval nancy_pots.wednesday

end NUMINAMATH_CALUDE_nancy_wednesday_pots_l3009_300930


namespace NUMINAMATH_CALUDE_partnership_investment_ratio_l3009_300999

theorem partnership_investment_ratio 
  (x : ℝ) 
  (m : ℝ) 
  (total_gain : ℝ) 
  (a_share : ℝ) 
  (h1 : total_gain = 27000) 
  (h2 : a_share = 9000) 
  (h3 : a_share = (1/3) * total_gain) 
  (h4 : (12*x) / (12*x + 12*x + 4*m*x) = 1/3) : 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_partnership_investment_ratio_l3009_300999


namespace NUMINAMATH_CALUDE_solve_system_l3009_300905

theorem solve_system (x y : ℚ) (h1 : x/2 - 2*y = 2) (h2 : x/2 + 2*y = 12) : y = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3009_300905


namespace NUMINAMATH_CALUDE_star_associativity_l3009_300933

-- Define the universal set
variable {U : Type}

-- Define the * operation
def star (X Y : Set U) : Set U := (X ∩ Y)ᶜ

-- State the theorem
theorem star_associativity (X Y Z : Set U) : 
  star (star X Y) Z = (Xᶜ ∩ Yᶜ) ∪ Z := by sorry

end NUMINAMATH_CALUDE_star_associativity_l3009_300933


namespace NUMINAMATH_CALUDE_reflect_M_y_axis_l3009_300913

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The original point M -/
def M : ℝ × ℝ := (3, 2)

/-- Theorem: Reflecting M(3,2) across the y-axis results in (-3,2) -/
theorem reflect_M_y_axis : reflect_y M = (-3, 2) := by
  sorry

end NUMINAMATH_CALUDE_reflect_M_y_axis_l3009_300913


namespace NUMINAMATH_CALUDE_greatest_common_divisor_620_180_under_100_l3009_300972

theorem greatest_common_divisor_620_180_under_100 :
  ∃ (d : ℕ), d = Nat.gcd 620 180 ∧ d < 100 ∧ d ∣ 620 ∧ d ∣ 180 ∧
  ∀ (x : ℕ), x < 100 → x ∣ 620 → x ∣ 180 → x ≤ d :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_620_180_under_100_l3009_300972


namespace NUMINAMATH_CALUDE_negation_distribution_l3009_300904

theorem negation_distribution (x : ℝ) : -(3*x - 2) = -3*x + 2 := by sorry

end NUMINAMATH_CALUDE_negation_distribution_l3009_300904


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3009_300979

theorem complex_number_in_first_quadrant : 
  let z : ℂ := 1 / (2 - Complex.I)
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3009_300979


namespace NUMINAMATH_CALUDE_or_sufficient_not_necessary_for_and_l3009_300980

theorem or_sufficient_not_necessary_for_and (p q : Prop) :
  (∃ (h : p ∨ q → p ∧ q), ¬(p ∧ q → p ∨ q)) := by sorry

end NUMINAMATH_CALUDE_or_sufficient_not_necessary_for_and_l3009_300980


namespace NUMINAMATH_CALUDE_science_club_neither_math_nor_physics_l3009_300948

theorem science_club_neither_math_nor_physics 
  (total : ℕ) 
  (math : ℕ) 
  (physics : ℕ) 
  (both : ℕ) 
  (h1 : total = 100) 
  (h2 : math = 65) 
  (h3 : physics = 43) 
  (h4 : both = 10) : 
  total - (math + physics - both) = 2 :=
by sorry

end NUMINAMATH_CALUDE_science_club_neither_math_nor_physics_l3009_300948


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3009_300966

theorem complex_magnitude_problem (z : ℂ) : z = 1 + 2 * I + I ^ 3 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3009_300966


namespace NUMINAMATH_CALUDE_skittles_and_erasers_grouping_l3009_300932

theorem skittles_and_erasers_grouping :
  let skittles : ℕ := 4502
  let erasers : ℕ := 4276
  let total_items : ℕ := skittles + erasers
  let num_groups : ℕ := 154
  total_items / num_groups = 57 := by
  sorry

end NUMINAMATH_CALUDE_skittles_and_erasers_grouping_l3009_300932


namespace NUMINAMATH_CALUDE_factoring_expression_l3009_300969

theorem factoring_expression (x : ℝ) : 2 * x * (x + 3) + 4 * (x + 3) = 2 * (x + 2) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l3009_300969


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3009_300911

theorem square_sum_geq_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3009_300911


namespace NUMINAMATH_CALUDE_condition_iff_prime_l3009_300981

def satisfies_condition (n : ℕ) : Prop :=
  (n = 2) ∨ (n > 2 ∧ ∀ k : ℕ, 2 ≤ k → k < n → ¬(k ∣ n))

theorem condition_iff_prime (n : ℕ) : satisfies_condition n ↔ Nat.Prime n :=
  sorry

end NUMINAMATH_CALUDE_condition_iff_prime_l3009_300981


namespace NUMINAMATH_CALUDE_angle_beta_proof_l3009_300939

theorem angle_beta_proof (α β : Real) (h1 : π / 2 < β) (h2 : β < π)
  (h3 : Real.tan (α + β) = 9 / 19) (h4 : Real.tan α = -4) :
  β = π - Real.arctan 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_beta_proof_l3009_300939


namespace NUMINAMATH_CALUDE_joy_pencil_count_l3009_300915

/-- The number of pencils Colleen has -/
def colleen_pencils : ℕ := 50

/-- The cost of each pencil in dollars -/
def pencil_cost : ℕ := 4

/-- The difference in dollars between what Colleen and Joy paid -/
def payment_difference : ℕ := 80

/-- The number of pencils Joy has -/
def joy_pencils : ℕ := 30

theorem joy_pencil_count :
  colleen_pencils * pencil_cost = joy_pencils * pencil_cost + payment_difference :=
sorry

end NUMINAMATH_CALUDE_joy_pencil_count_l3009_300915


namespace NUMINAMATH_CALUDE_second_part_speed_l3009_300995

/-- Represents a bicycle trip with three parts -/
structure BicycleTrip where
  total_distance : ℝ
  time_per_part : ℝ
  speed_first_part : ℝ
  speed_last_part : ℝ

/-- Theorem stating the speed of the second part of the trip -/
theorem second_part_speed (trip : BicycleTrip)
  (h_distance : trip.total_distance = 12)
  (h_time : trip.time_per_part = 0.25)
  (h_speed1 : trip.speed_first_part = 16)
  (h_speed3 : trip.speed_last_part = 20) :
  let distance1 := trip.speed_first_part * trip.time_per_part
  let distance3 := trip.speed_last_part * trip.time_per_part
  let distance2 := trip.total_distance - (distance1 + distance3)
  distance2 / trip.time_per_part = 12 := by
  sorry

#check second_part_speed

end NUMINAMATH_CALUDE_second_part_speed_l3009_300995


namespace NUMINAMATH_CALUDE_parallel_lines_b_value_l3009_300962

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ : ℝ} : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of b for which the given lines are parallel -/
theorem parallel_lines_b_value (b : ℝ) : 
  (∃ c₁ c₂ : ℝ, ∀ x y : ℝ, 3 * y - 3 * b = 9 * x + c₁ ↔ y + 2 = (b + 9) * x + c₂) → 
  b = -6 := by
sorry


end NUMINAMATH_CALUDE_parallel_lines_b_value_l3009_300962


namespace NUMINAMATH_CALUDE_sine_sum_greater_cosine_sum_increasing_geometric_sequence_l3009_300954

-- Define an acute-angled triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute_A : 0 < A ∧ A < π / 2
  acute_B : 0 < B ∧ B < π / 2
  acute_C : 0 < C ∧ C < π / 2
  sum_to_pi : A + B + C = π

-- Define a geometric sequence
def GeometricSequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- Statement for proposition ③
theorem sine_sum_greater_cosine_sum (t : AcuteTriangle) :
  Real.sin t.A + Real.sin t.B + Real.sin t.C > Real.cos t.A + Real.cos t.B + Real.cos t.C :=
sorry

-- Statement for proposition ④
theorem increasing_geometric_sequence (a : ℕ → ℝ) :
  (GeometricSequence a ∧ (∃ q > 1, ∀ n : ℕ, a (n + 1) = q * a n)) →
  (∀ n : ℕ, a (n + 1) > a n) ∧
  ¬((∀ n : ℕ, a (n + 1) > a n) → (∃ q > 1, ∀ n : ℕ, a (n + 1) = q * a n)) :=
sorry

end NUMINAMATH_CALUDE_sine_sum_greater_cosine_sum_increasing_geometric_sequence_l3009_300954


namespace NUMINAMATH_CALUDE_cassette_price_proof_l3009_300964

def total_money : ℕ := 37
def cd_price : ℕ := 14

theorem cassette_price_proof :
  ∃ (cassette_price : ℕ),
    2 * cd_price + cassette_price = total_money ∧
    cd_price + 2 * cassette_price = total_money - 5 ∧
    cassette_price = 9 := by
  sorry

end NUMINAMATH_CALUDE_cassette_price_proof_l3009_300964


namespace NUMINAMATH_CALUDE_expression_lower_bound_l3009_300945

theorem expression_lower_bound (n : ℤ) (L : ℤ) :
  (∃! (S : Finset ℤ), S.card = 25 ∧ ∀ m ∈ S, L < 4*m + 7 ∧ 4*m + 7 < 100) →
  L = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_lower_bound_l3009_300945


namespace NUMINAMATH_CALUDE_solution_value_l3009_300970

theorem solution_value (x y : ℚ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) : 
  (x + y) / 3 = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3009_300970


namespace NUMINAMATH_CALUDE_boat_distance_calculation_l3009_300977

theorem boat_distance_calculation
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (total_time : ℝ)
  (h1 : boat_speed = 8)
  (h2 : stream_speed = 2)
  (h3 : total_time = 56)
  : ∃ (distance : ℝ),
    distance = 210 ∧
    total_time = distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) :=
by sorry

end NUMINAMATH_CALUDE_boat_distance_calculation_l3009_300977


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3009_300946

theorem simplify_sqrt_expression : 
  Real.sqrt 10 - Real.sqrt 40 + Real.sqrt 90 = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3009_300946


namespace NUMINAMATH_CALUDE_insect_growth_theorem_l3009_300900

/-- An insect that doubles in size daily -/
structure GrowingInsect where
  initialSize : ℝ
  daysToReach10cm : ℕ

/-- The number of days it takes for the insect to reach 2.5 cm -/
def daysToReach2_5cm (insect : GrowingInsect) : ℕ :=
  sorry

theorem insect_growth_theorem (insect : GrowingInsect) 
  (h1 : insect.daysToReach10cm = 10) 
  (h2 : 2 ^ insect.daysToReach10cm * insect.initialSize = 10) :
  daysToReach2_5cm insect = 8 :=
sorry

end NUMINAMATH_CALUDE_insect_growth_theorem_l3009_300900


namespace NUMINAMATH_CALUDE_expression_value_l3009_300986

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 4)  -- absolute value of m is 4
  : a + b - (c * d) ^ 2021 - 3 * m = -13 ∨ a + b - (c * d) ^ 2021 - 3 * m = 11 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3009_300986


namespace NUMINAMATH_CALUDE_triangles_in_hexagon_with_center_l3009_300983

/-- The number of triangles formed by 7 points of a regular hexagon (including center) --/
def num_triangles_hexagon : ℕ :=
  Nat.choose 7 3 - 3

theorem triangles_in_hexagon_with_center :
  num_triangles_hexagon = 32 := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_hexagon_with_center_l3009_300983


namespace NUMINAMATH_CALUDE_right_triangle_leg_divisible_by_three_l3009_300936

theorem right_triangle_leg_divisible_by_three 
  (a b c : ℕ) -- a, b are legs, c is hypotenuse
  (h_right : a^2 + b^2 = c^2) -- Pythagorean theorem
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) -- Positive sides
  : 3 ∣ a ∨ 3 ∣ b := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_divisible_by_three_l3009_300936


namespace NUMINAMATH_CALUDE_final_result_depends_on_blue_l3009_300908

/-- Represents the color of a sprite -/
inductive SpriteColor
| Red
| Blue

/-- Represents the state of the game -/
structure GameState where
  red : ℕ  -- number of red sprites
  blue : ℕ  -- number of blue sprites

/-- Represents the result of the game -/
def GameResult := SpriteColor

/-- The game rules for sprite collision -/
def collide (c1 c2 : SpriteColor) : SpriteColor :=
  match c1, c2 with
  | SpriteColor.Red, SpriteColor.Red => SpriteColor.Red
  | SpriteColor.Blue, SpriteColor.Blue => SpriteColor.Red
  | _, _ => SpriteColor.Blue

/-- The final result of the game -/
def finalResult (initial : GameState) : GameResult :=
  if initial.blue % 2 = 0 then SpriteColor.Red else SpriteColor.Blue

/-- The main theorem: the final result depends only on the initial number of blue sprites -/
theorem final_result_depends_on_blue (m n : ℕ) :
  finalResult { red := m, blue := n } = 
    if n % 2 = 0 then SpriteColor.Red else SpriteColor.Blue :=
by sorry

end NUMINAMATH_CALUDE_final_result_depends_on_blue_l3009_300908


namespace NUMINAMATH_CALUDE_division_simplification_l3009_300989

theorem division_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y :=
by sorry

end NUMINAMATH_CALUDE_division_simplification_l3009_300989


namespace NUMINAMATH_CALUDE_square_ratio_side_length_sum_l3009_300901

theorem square_ratio_side_length_sum (area_ratio : ℚ) : 
  area_ratio = 50 / 98 →
  ∃ (a b c : ℕ), 
    (a * Real.sqrt b / c : ℝ) = Real.sqrt (area_ratio) ∧ 
    a = 5 ∧ b = 1 ∧ c = 7 ∧
    a + b + c = 13 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_sum_l3009_300901


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3009_300994

theorem triangle_angle_measure
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Sides of the triangle opposite to A, B, C respectively
  (h1 : a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = (1/2) * b)
  (h2 : a > b)
  (h3 : 0 < A ∧ A < π) -- Ensuring A is a valid angle measure
  (h4 : 0 < B ∧ B < π) -- Ensuring B is a valid angle measure
  (h5 : 0 < C ∧ C < π) -- Ensuring C is a valid angle measure
  (h6 : A + B + C = π) -- Sum of angles in a triangle
  : B = π/6 := by
  sorry

#check triangle_angle_measure

end NUMINAMATH_CALUDE_triangle_angle_measure_l3009_300994


namespace NUMINAMATH_CALUDE_range_of_m_l3009_300976

-- Define propositions p and q
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the set A (negation of q)
def A (m : ℝ) : Set ℝ := {x | x > 1 + m ∨ x < 1 - m}

-- Define the set B (negation of p)
def B : Set ℝ := {x | x > 10 ∨ x < -2}

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, x ∈ A m → x ∈ B) →
  (∃ x, x ∈ B ∧ x ∉ A m) →
  m ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3009_300976


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3009_300959

theorem min_value_of_expression (a b c : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  4 * (a^2 + b^2 + c^2) - (a + b + c)^2 ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3009_300959


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3009_300923

theorem line_segment_endpoint (x : ℝ) :
  (((x - 3)^2 + (4 + 2)^2).sqrt = 17) →
  (x < 0) →
  (x = 3 - Real.sqrt 253) := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3009_300923


namespace NUMINAMATH_CALUDE_team_organization_theorem_l3009_300942

/-- The number of ways to organize a team of 13 members into a specific hierarchy -/
def team_organization_count : ℕ := 4804800

/-- The total number of team members -/
def total_members : ℕ := 13

/-- The number of project managers -/
def project_managers : ℕ := 3

/-- The number of subordinates per project manager -/
def subordinates_per_manager : ℕ := 3

/-- Theorem stating the correct number of ways to organize the team -/
theorem team_organization_theorem :
  team_organization_count = 
    total_members * 
    (Nat.choose (total_members - 1) project_managers) * 
    (Nat.choose (total_members - 1 - project_managers) subordinates_per_manager) * 
    (Nat.choose (total_members - 1 - project_managers - subordinates_per_manager) subordinates_per_manager) * 
    (Nat.choose (total_members - 1 - project_managers - 2 * subordinates_per_manager) subordinates_per_manager) :=
by
  sorry

#eval team_organization_count

end NUMINAMATH_CALUDE_team_organization_theorem_l3009_300942


namespace NUMINAMATH_CALUDE_circle_M_properties_l3009_300925

-- Define the circle M
def circle_M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 4}

-- Define points A and B
def point_A : ℝ × ℝ := (1, -1)
def point_B : ℝ × ℝ := (-1, 1)

-- Define the line on which the center of M lies
def center_line (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem circle_M_properties :
  (point_A ∈ circle_M) ∧
  (point_B ∈ circle_M) ∧
  (∃ c : ℝ × ℝ, c ∈ circle_M ∧ center_line c.1 c.2) ∧
  (∀ p : ℝ × ℝ, p ∈ circle_M →
    (4 - Real.sqrt 7) / 3 ≤ (p.2 + 3) / (p.1 + 3) ∧
    (p.2 + 3) / (p.1 + 3) ≤ (4 + Real.sqrt 7) / 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_M_properties_l3009_300925


namespace NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_l3009_300985

-- Problem 1
theorem factorization_problem1 (m : ℝ) : 
  m * (m - 5) - 2 * (5 - m)^2 = -(m - 5) * (m - 10) := by sorry

-- Problem 2
theorem factorization_problem2 (x : ℝ) : 
  -4 * x^3 + 8 * x^2 - 4 * x = -4 * x * (x - 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_l3009_300985


namespace NUMINAMATH_CALUDE_function_identification_l3009_300916

/-- A first-degree function -/
def first_degree_function (f : ℝ → ℝ) : Prop :=
  ∃ k m : ℝ, ∀ x, f x = k * x + m

/-- A second-degree function -/
def second_degree_function (g : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, g x = a * x^2 + b * x + c

/-- Function composition equality -/
def composition_equality (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = g (f x)

/-- Tangent to x-axis -/
def tangent_to_x_axis (g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, y ≠ x → g y > 0

/-- Tangent to another function -/
def tangent_to_function (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = g x ∧ ∀ y : ℝ, y ≠ x → f y ≠ g y

theorem function_identification
  (f g : ℝ → ℝ)
  (h1 : first_degree_function f)
  (h2 : second_degree_function g)
  (h3 : composition_equality f g)
  (h4 : tangent_to_x_axis g)
  (h5 : tangent_to_function f g)
  (h6 : g 0 = 1/16) :
  (∀ x, f x = x) ∧ (∀ x, g x = x^2 + 1/2 * x + 1/16) := by
  sorry

end NUMINAMATH_CALUDE_function_identification_l3009_300916


namespace NUMINAMATH_CALUDE_survey_result_l3009_300992

/-- The number of households that used neither brand E nor brand B soap -/
def neither : ℕ := 80

/-- The number of households that used only brand E soap -/
def only_E : ℕ := 60

/-- The number of households that used both brands of soap -/
def both : ℕ := 40

/-- The ratio of households that used only brand B soap to those that used both brands -/
def B_to_both_ratio : ℕ := 3

/-- The total number of households surveyed -/
def total_households : ℕ := neither + only_E + both + B_to_both_ratio * both

theorem survey_result : total_households = 300 := by
  sorry

end NUMINAMATH_CALUDE_survey_result_l3009_300992


namespace NUMINAMATH_CALUDE_total_egg_rolls_l3009_300935

theorem total_egg_rolls (omar_rolls karen_rolls : ℕ) 
  (h1 : omar_rolls = 219) 
  (h2 : karen_rolls = 229) : 
  omar_rolls + karen_rolls = 448 := by
sorry

end NUMINAMATH_CALUDE_total_egg_rolls_l3009_300935


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3009_300960

/-- Given that y varies inversely as the square of x, and y = 15 when x = 5,
    prove that y = 375/9 when x = 3. -/
theorem inverse_variation_problem (y : ℝ → ℝ) (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 → y x = k / (x^2)) →  -- y varies inversely as the square of x
  y 5 = 15 →                           -- y = 15 when x = 5
  y 3 = 375 / 9 :=                     -- y = 375/9 when x = 3
by
  sorry


end NUMINAMATH_CALUDE_inverse_variation_problem_l3009_300960


namespace NUMINAMATH_CALUDE_cubic_integer_root_l3009_300903

theorem cubic_integer_root
  (a b c : ℚ)
  (h1 : ∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ↔ (x = 3 - Real.sqrt 5 ∨ x = 3 + Real.sqrt 5 ∨ (∃ n : ℤ, x = n)))
  (h2 : ∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 3 - Real.sqrt 5)
  (h3 : ∃ n : ℤ, (n : ℝ)^3 + a*(n : ℝ)^2 + b*(n : ℝ) + c = 0) :
  ∃ n : ℤ, n^3 + a*n^2 + b*n + c = 0 ∧ n = -6 :=
sorry

end NUMINAMATH_CALUDE_cubic_integer_root_l3009_300903


namespace NUMINAMATH_CALUDE_power_of_five_mod_ten_thousand_l3009_300943

theorem power_of_five_mod_ten_thousand :
  5^2023 ≡ 8125 [ZMOD 10000] := by sorry

end NUMINAMATH_CALUDE_power_of_five_mod_ten_thousand_l3009_300943


namespace NUMINAMATH_CALUDE_area_of_triangle_perimeter_of_triangle_l3009_300944

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.b^2 + t.c^2 - t.a^2 = t.b * t.c ∧ t.b * t.c = 1

-- Define the additional condition for part II
def satisfiesAdditionalCondition (t : Triangle) : Prop :=
  4 * Real.cos t.B * Real.cos t.C - 1 = 0

-- Theorem for part I
theorem area_of_triangle (t : Triangle) (h : satisfiesConditions t) :
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 4 := by
  sorry

-- Theorem for part II
theorem perimeter_of_triangle (t : Triangle) 
  (h1 : satisfiesConditions t) (h2 : satisfiesAdditionalCondition t) :
  t.a + t.b + t.c = 3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_perimeter_of_triangle_l3009_300944


namespace NUMINAMATH_CALUDE_randy_initial_amount_l3009_300906

/-- Proves that Randy's initial amount is $6166.67 given the problem conditions --/
theorem randy_initial_amount :
  ∀ (initial : ℝ),
  (3/4 : ℝ) * (initial + 2900) - 1300 = 5500 →
  initial = 6166.67 := by
sorry

end NUMINAMATH_CALUDE_randy_initial_amount_l3009_300906


namespace NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l3009_300984

-- Problem 1
theorem calculation_proof (a b : ℝ) (h : a ≠ b ∧ a ≠ -b ∧ a ≠ 0) :
  (a - b) / (a + b) - (a^2 - 2*a*b + b^2) / (a^2 - b^2) / ((a - b) / a) = -b / (a + b) := by
  sorry

-- Problem 2
theorem inequality_system_solution (x : ℝ) :
  (x - 3*(x - 2) ≥ 4 ∧ (2*x - 1) / 5 > (x + 1) / 2) ↔ x < -7 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l3009_300984


namespace NUMINAMATH_CALUDE_pathway_area_is_196_l3009_300902

/-- Represents the farm layout --/
structure FarmLayout where
  rows : Nat
  columns : Nat
  bed_width : Nat
  bed_height : Nat
  pathway_width : Nat

/-- Calculates the total area of pathways in the farm --/
def pathway_area (farm : FarmLayout) : Nat :=
  let total_width := farm.columns * farm.bed_width + (farm.columns + 1) * farm.pathway_width
  let total_height := farm.rows * farm.bed_height + (farm.rows + 1) * farm.pathway_width
  let total_area := total_width * total_height
  let beds_area := farm.rows * farm.columns * farm.bed_width * farm.bed_height
  total_area - beds_area

/-- Theorem stating that the pathway area for the given farm layout is 196 square feet --/
theorem pathway_area_is_196 (farm : FarmLayout) 
    (h1 : farm.rows = 4)
    (h2 : farm.columns = 3)
    (h3 : farm.bed_width = 4)
    (h4 : farm.bed_height = 3)
    (h5 : farm.pathway_width = 2) : 
  pathway_area farm = 196 := by
  sorry

end NUMINAMATH_CALUDE_pathway_area_is_196_l3009_300902


namespace NUMINAMATH_CALUDE_article_cost_l3009_300993

/-- The cost of an article given specific selling conditions --/
theorem article_cost : ∃ (C : ℝ), 
  (450 - C = 1.1 * (380 - C)) ∧ 
  (C > 0) ∧ 
  (C = 320) := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l3009_300993


namespace NUMINAMATH_CALUDE_expression_evaluation_l3009_300961

theorem expression_evaluation :
  -2^3 + 36 / 3^2 * (-1/2 : ℝ) + |(-5 : ℝ)| = -5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3009_300961


namespace NUMINAMATH_CALUDE_rent_increase_for_tax_change_l3009_300922

/-- Proves that a 12.5% rent increase maintains the same net income when tax increases from 10% to 20% -/
theorem rent_increase_for_tax_change (a : ℝ) (h : a > 0) :
  let initial_net_income := a * (1 - 0.1)
  let new_rent := a * (1 + 0.125)
  let new_net_income := new_rent * (1 - 0.2)
  initial_net_income = new_net_income :=
by sorry

#check rent_increase_for_tax_change

end NUMINAMATH_CALUDE_rent_increase_for_tax_change_l3009_300922


namespace NUMINAMATH_CALUDE_not_divisible_by_2020_l3009_300974

theorem not_divisible_by_2020 (k : ℕ) : ¬(2020 ∣ (k^3 - 3*k^2 + 2*k + 2)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_2020_l3009_300974


namespace NUMINAMATH_CALUDE_jelly_bean_problem_l3009_300928

def jelly_bean_piles (initial_amount : ℕ) (amount_eaten : ℕ) (pile_weight : ℕ) : ℕ :=
  (initial_amount - amount_eaten) / pile_weight

theorem jelly_bean_problem :
  jelly_bean_piles 36 6 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_problem_l3009_300928


namespace NUMINAMATH_CALUDE_expression_bounds_l3009_300937

theorem expression_bounds (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  2 * Real.sqrt 2 ≤ 
    Real.sqrt ((a^2)^2 + (b^2 - b^2)^2) + 
    Real.sqrt ((b^2)^2 + (c^2 - b^2)^2) + 
    Real.sqrt ((c^2)^2 + (d^2 - c^2)^2) + 
    Real.sqrt ((d^2)^2 + (a^2 - d^2)^2) ∧
  Real.sqrt ((a^2)^2 + (b^2 - b^2)^2) + 
    Real.sqrt ((b^2)^2 + (c^2 - b^2)^2) + 
    Real.sqrt ((c^2)^2 + (d^2 - c^2)^2) + 
    Real.sqrt ((d^2)^2 + (a^2 - d^2)^2) ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l3009_300937


namespace NUMINAMATH_CALUDE_unique_solution_l3009_300929

/-- Define the function f(x, y) = (x + y)(x^2 + y^2) -/
def f (x y : ℝ) : ℝ := (x + y) * (x^2 + y^2)

/-- Theorem stating that the only solution to the system of equations is (0, 0, 0, 0) -/
theorem unique_solution (a b c d : ℝ) :
  f a b = f c d ∧ f a c = f b d ∧ f a d = f b c →
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  sorry


end NUMINAMATH_CALUDE_unique_solution_l3009_300929


namespace NUMINAMATH_CALUDE_floor_tile_equations_l3009_300971

/-- Represents the floor tile purchase scenario -/
structure FloorTilePurchase where
  x : ℕ  -- number of colored floor tiles
  y : ℕ  -- number of single-color floor tiles
  colored_cost : ℕ := 24  -- cost of colored tiles in yuan
  single_cost : ℕ := 12   -- cost of single-color tiles in yuan
  total_cost : ℕ := 2220  -- total cost in yuan

/-- The system of equations correctly represents the floor tile purchase scenario -/
theorem floor_tile_equations (purchase : FloorTilePurchase) : 
  (purchase.colored_cost * purchase.x + purchase.single_cost * purchase.y = purchase.total_cost) ∧
  (purchase.y = 2 * purchase.x - 15) := by
  sorry

end NUMINAMATH_CALUDE_floor_tile_equations_l3009_300971


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3009_300907

/-- Represents the age of a person -/
structure Age :=
  (years : ℕ)

/-- Represents the ratio between two numbers -/
structure Ratio :=
  (numerator : ℕ)
  (denominator : ℕ)

/-- Given two people p and q, their ages 6 years ago, and their current total age,
    proves that the ratio of their current ages is 3:4 -/
theorem age_ratio_proof 
  (p q : Age) 
  (h1 : p.years + 6 = (q.years + 6) / 2)  -- 6 years ago, p was half of q in age
  (h2 : (p.years + 6) + (q.years + 6) = 21)  -- The total of their present ages is 21
  : Ratio.mk 3 4 = Ratio.mk (p.years + 6) (q.years + 6) :=
by
  sorry


end NUMINAMATH_CALUDE_age_ratio_proof_l3009_300907


namespace NUMINAMATH_CALUDE_chocolate_count_l3009_300920

/-- The number of chocolates in each bag -/
def chocolates_per_bag : ℕ := 156

/-- The number of bags bought -/
def bags_bought : ℕ := 20

/-- The total number of chocolates -/
def total_chocolates : ℕ := chocolates_per_bag * bags_bought

theorem chocolate_count : total_chocolates = 3120 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_count_l3009_300920


namespace NUMINAMATH_CALUDE_survey_is_sample_l3009_300910

/-- Represents the total number of students in the population -/
def population_size : ℕ := 32000

/-- Represents the number of students surveyed -/
def survey_size : ℕ := 1600

/-- Represents a student's weight -/
structure Weight where
  value : ℝ

/-- Represents the population of all students' weights -/
def population : Finset Weight := sorry

/-- Represents the surveyed students' weights -/
def survey : Finset Weight := sorry

/-- Theorem stating that the survey is a sample of the population -/
theorem survey_is_sample : survey ⊆ population ∧ survey.card = survey_size := by sorry

end NUMINAMATH_CALUDE_survey_is_sample_l3009_300910


namespace NUMINAMATH_CALUDE_intersection_range_l3009_300956

/-- The function f(x) = x^3 - 3x - 1 -/
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

/-- Predicate to check if a line y = m intersects f at three distinct points -/
def has_three_distinct_intersections (m : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m

/-- Theorem stating the range of m for which y = m intersects f at three distinct points -/
theorem intersection_range :
  ∀ m : ℝ, has_three_distinct_intersections m ↔ m > -3 ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_range_l3009_300956


namespace NUMINAMATH_CALUDE_variations_formula_l3009_300991

/-- The number of r-class variations from n elements where the first s elements occur -/
def variations (n r s : ℕ) : ℕ :=
  (Nat.factorial (n - s) * Nat.factorial r) / (Nat.factorial (r - s) * Nat.factorial (n - r))

/-- Theorem stating the number of r-class variations from n elements where the first s elements occur -/
theorem variations_formula (n r s : ℕ) (h1 : s < r) (h2 : r ≤ n) :
  variations n r s = (Nat.factorial (n - s) * Nat.factorial r) / (Nat.factorial (r - s) * Nat.factorial (n - r)) :=
by sorry

end NUMINAMATH_CALUDE_variations_formula_l3009_300991


namespace NUMINAMATH_CALUDE_jack_remaining_plates_l3009_300967

def initial_flower_plates : ℕ := 6
def initial_checked_plates : ℕ := 9
def initial_striped_plates : ℕ := 3
def smashed_flower_plates : ℕ := 2
def smashed_striped_plates : ℕ := 1

def remaining_plates : ℕ :=
  (initial_flower_plates - smashed_flower_plates) +
  initial_checked_plates +
  (initial_striped_plates - smashed_striped_plates) +
  (initial_checked_plates * initial_checked_plates)

theorem jack_remaining_plates :
  remaining_plates = 96 := by
  sorry

end NUMINAMATH_CALUDE_jack_remaining_plates_l3009_300967


namespace NUMINAMATH_CALUDE_total_phones_sold_l3009_300924

/-- Calculates the total number of cell phones sold given the initial and final inventories, and the number of damaged/defective phones. -/
def cellPhonesSold (initialSamsung : ℕ) (finalSamsung : ℕ) (initialIPhone : ℕ) (finalIPhone : ℕ) (damagedSamsung : ℕ) (defectiveIPhone : ℕ) : ℕ :=
  (initialSamsung - damagedSamsung - finalSamsung) + (initialIPhone - defectiveIPhone - finalIPhone)

/-- Theorem stating that the total number of cell phones sold is 4 given the specific inventory and damage numbers. -/
theorem total_phones_sold :
  cellPhonesSold 14 10 8 5 2 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_phones_sold_l3009_300924


namespace NUMINAMATH_CALUDE_smallest_constant_degenerate_triangle_l3009_300927

/-- A degenerate triangle is represented by three non-negative real numbers a, b, and c,
    where a + b = c --/
structure DegenerateTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  non_neg_a : 0 ≤ a
  non_neg_b : 0 ≤ b
  non_neg_c : 0 ≤ c
  sum_eq_c : a + b = c

/-- The smallest constant N such that (a^2 + b^2) / c^2 < N for all degenerate triangles
    is 1/2 --/
theorem smallest_constant_degenerate_triangle :
  ∃ N : ℝ, (∀ t : DegenerateTriangle, (t.a^2 + t.b^2) / t.c^2 < N) ∧
  (∀ ε > 0, ∃ t : DegenerateTriangle, (t.a^2 + t.b^2) / t.c^2 > N - ε) ∧
  N = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_constant_degenerate_triangle_l3009_300927


namespace NUMINAMATH_CALUDE_loss_equivalent_pencils_proof_l3009_300950

/-- The number of pencils Patrick purchased -/
def total_pencils : ℕ := 60

/-- The ratio of cost to selling price for 60 pencils -/
def cost_to_sell_ratio : ℚ := 1.3333333333333333

/-- The number of pencils whose selling price equals the loss -/
def loss_equivalent_pencils : ℕ := 20

theorem loss_equivalent_pencils_proof :
  ∃ (selling_price : ℚ) (cost : ℚ),
    cost = cost_to_sell_ratio * selling_price ∧
    loss_equivalent_pencils * (selling_price / total_pencils) = cost - selling_price :=
by sorry

end NUMINAMATH_CALUDE_loss_equivalent_pencils_proof_l3009_300950


namespace NUMINAMATH_CALUDE_root_condition_implies_m_range_l3009_300947

theorem root_condition_implies_m_range (m : ℝ) :
  (∀ x : ℝ, (m / (2 * x - 4) = (1 - x) / (2 - x) - 2) → x > 0) →
  m < 6 ∧ m ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_root_condition_implies_m_range_l3009_300947


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3009_300975

/-- Given a hyperbola with equation (x-2)^2/144 - (y+3)^2/81 = 1, 
    the slope of its asymptotes is 3/4 -/
theorem hyperbola_asymptote_slope :
  ∀ (x y : ℝ), 
  ((x - 2)^2 / 144 - (y + 3)^2 / 81 = 1) →
  (∃ m : ℝ, m = 3/4 ∧ 
   (∀ ε > 0, ∃ x₀ y₀ : ℝ, 
    ((x₀ - 2)^2 / 144 - (y₀ + 3)^2 / 81 = 1) ∧
    abs (y₀ - (m * x₀ - 9/2)) < ε)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3009_300975


namespace NUMINAMATH_CALUDE_factorial_square_root_theorem_l3009_300957

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_square_root_theorem :
  (((factorial 5 * factorial 4).sqrt) ^ 2 : ℕ) = 2880 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_theorem_l3009_300957


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_squared_factorization_of_2025_l3009_300938

/-- The largest integer k such that 2025^k divides (2025!)^2 is 505. -/
theorem largest_power_dividing_factorial_squared : ∃ k : ℕ, k = 505 ∧ 
  (∀ m : ℕ, (2025 ^ m : ℕ) ∣ (Nat.factorial 2025)^2 → m ≤ k) ∧
  (2025 ^ k : ℕ) ∣ (Nat.factorial 2025)^2 := by
  sorry

/-- 2025 is equal to 3^4 * 5^2 -/
theorem factorization_of_2025 : 2025 = 3^4 * 5^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_squared_factorization_of_2025_l3009_300938


namespace NUMINAMATH_CALUDE_distribute_7_balls_3_boxes_l3009_300926

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_7_balls_3_boxes : distribute 7 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_7_balls_3_boxes_l3009_300926


namespace NUMINAMATH_CALUDE_storks_on_fence_storks_count_l3009_300931

theorem storks_on_fence (initial_birds : ℕ) (additional_birds : ℕ) (bird_stork_difference : ℕ) : ℕ :=
  let total_birds := initial_birds + additional_birds
  let storks := total_birds - bird_stork_difference
  storks

theorem storks_count : storks_on_fence 3 4 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_storks_on_fence_storks_count_l3009_300931


namespace NUMINAMATH_CALUDE_polygon_with_720_degree_sum_is_hexagon_l3009_300998

/-- A polygon with interior angles summing to 720° has 6 sides -/
theorem polygon_with_720_degree_sum_is_hexagon :
  ∀ (n : ℕ), (n - 2) * 180 = 720 → n = 6 :=
by sorry

end NUMINAMATH_CALUDE_polygon_with_720_degree_sum_is_hexagon_l3009_300998


namespace NUMINAMATH_CALUDE_division_inequality_quotient_invariance_l3009_300914

theorem division_inequality : 0.056 / 0.08 ≠ 0.56 / 0.08 := by
  -- The proof goes here
  sorry

-- Property of invariance of quotient
theorem quotient_invariance (a b c : ℝ) (hc : c ≠ 0) :
  a / b = (a * c) / (b * c) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_division_inequality_quotient_invariance_l3009_300914


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3009_300949

theorem solution_set_inequality (x : ℝ) : 
  (1 / x < 1 / 3) ↔ (x < 0 ∨ x > 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3009_300949
