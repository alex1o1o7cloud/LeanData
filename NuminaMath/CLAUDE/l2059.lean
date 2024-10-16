import Mathlib

namespace NUMINAMATH_CALUDE_orange_distribution_l2059_205945

/-- The number of ways to distribute distinct oranges to sons. -/
def distribute_oranges (num_oranges : ℕ) (num_sons : ℕ) : ℕ :=
  (num_sons.choose num_oranges) * num_oranges.factorial

/-- Theorem: The number of ways to distribute 5 distinct oranges to 8 sons is 6720. -/
theorem orange_distribution :
  distribute_oranges 5 8 = 6720 := by
  sorry

#eval distribute_oranges 5 8

end NUMINAMATH_CALUDE_orange_distribution_l2059_205945


namespace NUMINAMATH_CALUDE_butter_cheese_ratio_l2059_205916

/-- Represents the prices of items bought by Ursula -/
structure Prices where
  butter : ℝ
  bread : ℝ
  cheese : ℝ
  tea : ℝ

/-- The conditions of Ursula's shopping trip -/
def shopping_conditions (p : Prices) : Prop :=
  p.tea = 10 ∧
  p.tea = 2 * p.cheese ∧
  p.bread = p.butter / 2 ∧
  p.butter + p.bread + p.cheese + p.tea = 21

/-- The theorem stating that under the given conditions, 
    the price of butter is 80% of the price of cheese -/
theorem butter_cheese_ratio (p : Prices) 
  (h : shopping_conditions p) : p.butter / p.cheese = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_butter_cheese_ratio_l2059_205916


namespace NUMINAMATH_CALUDE_fraction_equality_l2059_205958

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + y) / (x - 4 * y) = -3) : 
  (x + 3 * y) / (3 * x - y) = 16 / 13 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2059_205958


namespace NUMINAMATH_CALUDE_rectangle_perimeter_is_48_l2059_205979

/-- A rectangle can be cut into two squares with side length 8 cm -/
structure Rectangle where
  length : ℝ
  width : ℝ
  is_cut_into_squares : length = 2 * width
  square_side : ℝ
  square_side_eq : square_side = width

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: The perimeter of the rectangle is 48 cm -/
theorem rectangle_perimeter_is_48 (r : Rectangle) (h : r.square_side = 8) : perimeter r = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_is_48_l2059_205979


namespace NUMINAMATH_CALUDE_power_seven_150_mod_12_l2059_205985

theorem power_seven_150_mod_12 : 7^150 ≡ 1 [ZMOD 12] := by sorry

end NUMINAMATH_CALUDE_power_seven_150_mod_12_l2059_205985


namespace NUMINAMATH_CALUDE_cosine_equation_solutions_l2059_205914

theorem cosine_equation_solutions :
  ∃! (n : ℕ), ∃ (S : Finset ℝ),
    (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi) ∧
    (∀ x ∈ S, 3 * (Real.cos x)^3 - 7 * (Real.cos x)^2 + 3 * Real.cos x = 0) ∧
    Finset.card S = n ∧
    n = 4 :=
by sorry

end NUMINAMATH_CALUDE_cosine_equation_solutions_l2059_205914


namespace NUMINAMATH_CALUDE_news_spread_theorem_l2059_205977

/-- Represents the spread of news in a village -/
structure NewsSpread where
  residents : ℕ
  start_date : ℕ
  current_date : ℕ
  informed_residents : Finset ℕ

/-- The number of days since the news started spreading -/
def days_passed (ns : NewsSpread) : ℕ :=
  ns.current_date - ns.start_date

/-- Predicate to check if all residents are informed -/
def all_informed (ns : NewsSpread) : Prop :=
  ns.informed_residents.card = ns.residents

theorem news_spread_theorem (ns : NewsSpread) 
  (h_residents : ns.residents = 20)
  (h_start : ns.start_date = 1) :
  (∃ d₁ d₂, d₁ ≤ 15 ∧ d₂ ≥ 18 ∧ days_passed {ns with current_date := ns.start_date + d₁} < ns.residents ∧
            all_informed {ns with current_date := ns.start_date + d₂}) ∧
  (∀ d, d > 20 → all_informed {ns with current_date := ns.start_date + d}) :=
by sorry

end NUMINAMATH_CALUDE_news_spread_theorem_l2059_205977


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_squares_l2059_205915

theorem largest_common_divisor_of_consecutive_squares (n : ℤ) (h : Even n) :
  (∃ (k : ℤ), k > 1 ∧ ∀ (b : ℤ), k ∣ ((n + 1)^2 - n^2)) → False :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_squares_l2059_205915


namespace NUMINAMATH_CALUDE_extreme_values_and_roots_l2059_205940

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_values_and_roots (a b c : ℝ) :
  (∀ x : ℝ, f' a b x = 0 ↔ x = 1 ∨ x = 3) →
  (a = -6 ∧ b = 9) ∧
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f (-6) 9 c x = 0 ∧ f (-6) 9 c y = 0 ∧ f (-6) 9 c z = 0) →
  -4 < c ∧ c < 0 :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_roots_l2059_205940


namespace NUMINAMATH_CALUDE_sqrt_three_properties_l2059_205913

theorem sqrt_three_properties : ∃ x : ℝ, Irrational x ∧ 0 < x ∧ x < 3 :=
  by
  use Real.sqrt 3
  sorry

end NUMINAMATH_CALUDE_sqrt_three_properties_l2059_205913


namespace NUMINAMATH_CALUDE_circle_radius_l2059_205956

theorem circle_radius (x y : ℝ) (h : x + 2*y = 100*Real.pi) : 
  x = Real.pi * 8^2 ∧ y = 2 * Real.pi * 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l2059_205956


namespace NUMINAMATH_CALUDE_sine_inequality_l2059_205942

theorem sine_inequality (x : ℝ) : 
  (9.2894 * Real.sin x * Real.sin (2 * x) * Real.sin (3 * x) > Real.sin (4 * x)) ↔ 
  (∃ n : ℤ, (-π/8 + π * n < x ∧ x < π * n) ∨ 
            (π/8 + π * n < x ∧ x < 3*π/8 + π * n) ∨ 
            (π/2 + π * n < x ∧ x < 5*π/8 + π * n)) := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l2059_205942


namespace NUMINAMATH_CALUDE_perpendicular_to_third_not_implies_perpendicular_l2059_205988

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Predicate for two lines being perpendicular -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  -- Definition of perpendicular lines
  sorry

/-- Theorem stating that the perpendicularity of two lines to a third line
    does not imply their perpendicularity to each other -/
theorem perpendicular_to_third_not_implies_perpendicular :
  ∃ (l1 l2 l3 : Line3D),
    perpendicular l1 l3 ∧ perpendicular l2 l3 ∧ ¬perpendicular l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_to_third_not_implies_perpendicular_l2059_205988


namespace NUMINAMATH_CALUDE_equal_cake_distribution_l2059_205912

theorem equal_cake_distribution (total_cakes : ℕ) (num_children : ℕ) (cakes_per_child : ℕ) :
  total_cakes = 18 →
  num_children = 3 →
  total_cakes = num_children * cakes_per_child →
  cakes_per_child = 6 := by
sorry

end NUMINAMATH_CALUDE_equal_cake_distribution_l2059_205912


namespace NUMINAMATH_CALUDE_turtle_race_times_l2059_205954

/-- The time it took for Greta's turtle to finish the race -/
def greta_time : ℕ := sorry

/-- The time it took for George's turtle to finish the race -/
def george_time : ℕ := sorry

/-- The time it took for Gloria's turtle to finish the race -/
def gloria_time : ℕ := 8

theorem turtle_race_times :
  (george_time = greta_time - 2) ∧
  (gloria_time = 2 * george_time) ∧
  (greta_time = 6) := by sorry

end NUMINAMATH_CALUDE_turtle_race_times_l2059_205954


namespace NUMINAMATH_CALUDE_opposite_of_three_l2059_205963

theorem opposite_of_three : 
  (-(3 : ℤ) : ℤ) = -3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_three_l2059_205963


namespace NUMINAMATH_CALUDE_tan_pi_minus_alpha_eq_two_implies_ratio_eq_one_fourth_l2059_205911

theorem tan_pi_minus_alpha_eq_two_implies_ratio_eq_one_fourth
  (α : ℝ) (h : Real.tan (π - α) = 2) :
  (Real.sin (π/2 + α) + Real.sin (π - α)) /
  (Real.cos (3*π/2 + α) + 2 * Real.cos (π + α)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_minus_alpha_eq_two_implies_ratio_eq_one_fourth_l2059_205911


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2059_205961

theorem solution_set_inequality (x : ℝ) : 
  (((1 - x) / (x + 1) ≤ 0) ↔ (x ∈ Set.Iic (-1) ∪ Set.Ici 1)) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2059_205961


namespace NUMINAMATH_CALUDE_sleep_ratio_theorem_l2059_205919

/-- Represents Billy's sleep pattern over four nights -/
structure SleepPattern where
  first_night : ℝ
  second_night : ℝ
  third_night : ℝ
  fourth_night : ℝ

/-- Theorem stating the ratio of the fourth night's sleep to the third night's sleep -/
theorem sleep_ratio_theorem (s : SleepPattern) 
  (h1 : s.first_night = 6)
  (h2 : s.second_night = s.first_night + 2)
  (h3 : s.third_night = s.second_night / 2)
  (h4 : s.fourth_night = s.third_night * (s.fourth_night / s.third_night))
  (h5 : s.first_night + s.second_night + s.third_night + s.fourth_night = 30) :
  s.fourth_night / s.third_night = 3 := by
  sorry

end NUMINAMATH_CALUDE_sleep_ratio_theorem_l2059_205919


namespace NUMINAMATH_CALUDE_josh_marbles_remaining_l2059_205960

def initial_marbles : ℕ := 16
def lost_marbles : ℕ := 7

theorem josh_marbles_remaining : initial_marbles - lost_marbles = 9 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_remaining_l2059_205960


namespace NUMINAMATH_CALUDE_average_and_differences_l2059_205970

theorem average_and_differences (x : ℝ) : 
  (45 + x) / 2 = 38 → |x - 45| + |x - 30| = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_and_differences_l2059_205970


namespace NUMINAMATH_CALUDE_percentage_prefer_x_is_zero_l2059_205997

def total_employees : ℕ := 200
def relocated_to_x : ℚ := 30 / 100
def relocated_to_y : ℚ := 70 / 100
def prefer_y : ℚ := 40 / 100
def max_satisfied : ℕ := 140

theorem percentage_prefer_x_is_zero :
  ∃ (prefer_x : ℚ),
    prefer_x ≥ 0 ∧
    prefer_x + prefer_y = 1 ∧
    (prefer_x * total_employees).floor + (prefer_y * total_employees).floor ≤ max_satisfied ∧
    prefer_x = 0 := by sorry

end NUMINAMATH_CALUDE_percentage_prefer_x_is_zero_l2059_205997


namespace NUMINAMATH_CALUDE_kaplan_bobby_slice_ratio_l2059_205917

/-- Represents the number of pizzas Bobby has -/
def bobby_pizzas : ℕ := 2

/-- Represents the number of slices per pizza -/
def slices_per_pizza : ℕ := 6

/-- Represents the number of slices Mrs. Kaplan has -/
def kaplan_slices : ℕ := 3

/-- Calculates the total number of slices Bobby has -/
def bobby_slices : ℕ := bobby_pizzas * slices_per_pizza

/-- Represents the ratio of Mrs. Kaplan's slices to Bobby's slices -/
def slice_ratio : Rat := kaplan_slices / bobby_slices

theorem kaplan_bobby_slice_ratio :
  slice_ratio = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_kaplan_bobby_slice_ratio_l2059_205917


namespace NUMINAMATH_CALUDE_function_value_at_negative_one_l2059_205922

/-- Given a function f(x) = a*sin(x) + b*tan(x) + 3 where a and b are real numbers,
    if f(1) = 1, then f(-1) = 5. -/
theorem function_value_at_negative_one 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * Real.sin x + b * Real.tan x + 3) 
  (h2 : f 1 = 1) : 
  f (-1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_one_l2059_205922


namespace NUMINAMATH_CALUDE_walkway_area_is_116_l2059_205981

/-- Represents the garden layout --/
structure GardenLayout where
  bed_width : ℕ := 8
  bed_height : ℕ := 3
  beds_per_row : ℕ := 2
  num_rows : ℕ := 3
  walkway_width : ℕ := 1
  has_central_walkway : Bool := true

/-- Calculates the total area of walkways in the garden --/
def walkway_area (garden : GardenLayout) : ℕ :=
  let total_width := garden.bed_width * garden.beds_per_row + 
                     (garden.beds_per_row + 1) * garden.walkway_width + 
                     (if garden.has_central_walkway then garden.walkway_width else 0)
  let total_height := garden.bed_height * garden.num_rows + 
                      (garden.num_rows + 1) * garden.walkway_width
  let total_area := total_width * total_height
  let beds_area := garden.bed_width * garden.bed_height * garden.beds_per_row * garden.num_rows
  total_area - beds_area

/-- Theorem stating that the walkway area for the given garden layout is 116 square feet --/
theorem walkway_area_is_116 (garden : GardenLayout) : walkway_area garden = 116 := by
  sorry

end NUMINAMATH_CALUDE_walkway_area_is_116_l2059_205981


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2059_205941

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Define the slope of the line parallel to 3x + y = 0
def m : ℝ := -3

-- Define the point of tangency
def a : ℝ := 1
def b : ℝ := f a

-- State the theorem
theorem tangent_line_equation :
  ∃ (c : ℝ), ∀ x y : ℝ,
    (y - b = m * (x - a)) ↔ (y = -3*x + c) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2059_205941


namespace NUMINAMATH_CALUDE_sphere_in_cone_l2059_205932

theorem sphere_in_cone (b d g : ℝ) : 
  let cone_base_radius : ℝ := 15
  let cone_height : ℝ := 30
  let sphere_radius : ℝ := b * Real.sqrt d - g
  g = b + 6 →
  sphere_radius = (cone_height * cone_base_radius) / (cone_base_radius + Real.sqrt (cone_base_radius^2 + cone_height^2)) →
  b + d = 12.5 := by
sorry

end NUMINAMATH_CALUDE_sphere_in_cone_l2059_205932


namespace NUMINAMATH_CALUDE_stereo_system_trade_in_john_stereo_trade_in_l2059_205938

theorem stereo_system_trade_in (old_cost : ℝ) (trade_in_percentage : ℝ) 
  (new_cost : ℝ) (discount_percentage : ℝ) : ℝ :=
  let trade_in_value := old_cost * trade_in_percentage
  let discounted_new_cost := new_cost * (1 - discount_percentage)
  discounted_new_cost - trade_in_value

theorem john_stereo_trade_in :
  stereo_system_trade_in 250 0.8 600 0.25 = 250 := by
  sorry

end NUMINAMATH_CALUDE_stereo_system_trade_in_john_stereo_trade_in_l2059_205938


namespace NUMINAMATH_CALUDE_farm_problem_solution_l2059_205959

/-- Represents the farm field ploughing problem -/
structure FarmField where
  planned_daily_rate : ℕ  -- Planned hectares per day
  actual_daily_rate : ℕ   -- Actual hectares per day
  extra_days : ℕ          -- Additional days worked
  remaining_area : ℕ      -- Hectares left to plough

/-- Calculates the total area and initially planned days for a given farm field problem -/
def solve_farm_problem (field : FarmField) : ℕ × ℕ :=
  sorry

/-- Theorem stating the solution to the specific farm field problem -/
theorem farm_problem_solution :
  let field := FarmField.mk 90 85 2 40
  solve_farm_problem field = (3780, 42) :=
sorry

end NUMINAMATH_CALUDE_farm_problem_solution_l2059_205959


namespace NUMINAMATH_CALUDE_nathaniel_best_friends_l2059_205933

def initial_tickets : ℕ := 11
def remaining_tickets : ℕ := 3
def tickets_per_friend : ℕ := 2

def number_of_friends : ℕ := (initial_tickets - remaining_tickets) / tickets_per_friend

theorem nathaniel_best_friends : number_of_friends = 4 := by
  sorry

end NUMINAMATH_CALUDE_nathaniel_best_friends_l2059_205933


namespace NUMINAMATH_CALUDE_complex_modulus_l2059_205967

theorem complex_modulus (z : ℂ) : z = Complex.mk (Real.sin (π / 3)) (-Real.cos (π / 6)) → Complex.abs z = Real.sqrt (3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2059_205967


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2059_205955

/-- The complex number -2i/(1+i) is equal to -1-i -/
theorem complex_fraction_equality : ((-2 * Complex.I) / (1 + Complex.I)) = (-1 - Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2059_205955


namespace NUMINAMATH_CALUDE_power_evaluation_l2059_205929

theorem power_evaluation : (2 ^ 2) ^ (2 ^ (2 + 1)) = 65536 := by sorry

end NUMINAMATH_CALUDE_power_evaluation_l2059_205929


namespace NUMINAMATH_CALUDE_some_board_game_masters_enjoy_logic_puzzles_l2059_205920

-- Define the universe
variable (U : Type)

-- Define predicates
variable (M : U → Prop)  -- M x means x is a mathematics enthusiast
variable (B : U → Prop)  -- B x means x is a board game master
variable (L : U → Prop)  -- L x means x enjoys logic puzzles

-- State the theorem
theorem some_board_game_masters_enjoy_logic_puzzles
  (h1 : ∀ x, M x → L x)  -- All mathematics enthusiasts enjoy logic puzzles
  (h2 : ∃ x, B x ∧ M x)  -- Some board game masters are mathematics enthusiasts
  : ∃ x, B x ∧ L x :=    -- Some board game masters enjoy logic puzzles
by
  sorry


end NUMINAMATH_CALUDE_some_board_game_masters_enjoy_logic_puzzles_l2059_205920


namespace NUMINAMATH_CALUDE_max_points_on_ellipse_l2059_205939

/-- Represents an ellipse with semi-major axis a and focal distance c -/
structure Ellipse where
  a : ℝ
  c : ℝ

/-- Represents a sequence of points on an ellipse -/
structure PointSequence where
  n : ℕ
  d : ℝ

theorem max_points_on_ellipse (e : Ellipse) (seq : PointSequence) :
  e.a - e.c = 1 →
  e.a + e.c = 3 →
  seq.d > 1/100 →
  (∀ i : ℕ, i < seq.n → 1 + i * seq.d ≤ 3) →
  seq.n ≤ 200 := by
  sorry

end NUMINAMATH_CALUDE_max_points_on_ellipse_l2059_205939


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l2059_205925

theorem fraction_ratio_equality (x : ℚ) : (2 / 3 : ℚ) / x = (3 / 5 : ℚ) / (7 / 15 : ℚ) → x = 14 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l2059_205925


namespace NUMINAMATH_CALUDE_quadratic_point_relationship_l2059_205975

/-- Given a quadratic function f(x) = -3x² + 2, prove that for points
    A(-1, y₁), B(1, y₂), and C(2, y₃) on its graph, y₁ = y₂ > y₃ holds. -/
theorem quadratic_point_relationship : ∀ (y₁ y₂ y₃ : ℝ),
  (y₁ = -3 * (-1)^2 + 2) →
  (y₂ = -3 * 1^2 + 2) →
  (y₃ = -3 * 2^2 + 2) →
  (y₁ = y₂ ∧ y₁ > y₃) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_point_relationship_l2059_205975


namespace NUMINAMATH_CALUDE_divisibility_condition_l2059_205990

theorem divisibility_condition (n p : ℕ) (h_prime : Nat.Prime p) (h_range : 0 < n ∧ n ≤ 2*p) :
  (n^(p-1) ∣ ((p-1)^n + 1)) ↔ (n = 1 ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2059_205990


namespace NUMINAMATH_CALUDE_triangle_not_right_angle_l2059_205949

theorem triangle_not_right_angle (a b c : ℝ) (h_sum : a + b + c = 180) (h_ratio : ∃ k : ℝ, a = 3*k ∧ b = 4*k ∧ c = 5*k) : ¬(a = 90 ∨ b = 90 ∨ c = 90) := by
  sorry

end NUMINAMATH_CALUDE_triangle_not_right_angle_l2059_205949


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l2059_205931

/-- A regular nonagon is a 9-sided polygon with all sides and angles equal -/
structure RegularNonagon where
  -- We don't need to define the structure explicitly for this problem

/-- The number of diagonals in a regular nonagon -/
def num_diagonals (n : RegularNonagon) : ℕ := 27

/-- The number of ways to choose 4 vertices from 9 vertices -/
def num_four_vertices_choices (n : RegularNonagon) : ℕ := Nat.choose 9 4

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def num_diagonal_pairs (n : RegularNonagon) : ℕ := Nat.choose (num_diagonals n) 2

/-- The probability of two randomly chosen diagonals intersecting inside the nonagon -/
def intersection_probability (n : RegularNonagon) : ℚ :=
  (num_four_vertices_choices n : ℚ) / (num_diagonal_pairs n : ℚ)

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  intersection_probability n = 14 / 39 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l2059_205931


namespace NUMINAMATH_CALUDE_odot_composition_l2059_205994

/-- Custom operation ⊙ -/
def odot (x y : ℝ) : ℝ := x^2 + x*y - y^2

/-- Theorem stating that h ⊙ (h ⊙ h) = -4 when h = 2 -/
theorem odot_composition (h : ℝ) (h_eq : h = 2) : odot h (odot h h) = -4 := by
  sorry

end NUMINAMATH_CALUDE_odot_composition_l2059_205994


namespace NUMINAMATH_CALUDE_h_is_even_l2059_205995

-- Define g as an odd function
def g : ℝ → ℝ := sorry

-- Axiom stating that g is an odd function
axiom g_odd : ∀ x : ℝ, g (-x) = -g x

-- Define h using g
def h (x : ℝ) : ℝ := |g (x^4)|

-- Theorem stating that h is an even function
theorem h_is_even : ∀ x : ℝ, h (-x) = h x := by
  sorry

end NUMINAMATH_CALUDE_h_is_even_l2059_205995


namespace NUMINAMATH_CALUDE_roots_of_g_are_cubes_of_roots_of_f_l2059_205905

/-- The original polynomial f(x) -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 1

/-- The polynomial g(x) whose roots are the cubes of the roots of f(x) -/
def g (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

/-- Theorem stating that the roots of g are the cubes of the roots of f -/
theorem roots_of_g_are_cubes_of_roots_of_f :
  ∀ r : ℝ, f r = 0 → g (r^3) = 0 := by sorry

end NUMINAMATH_CALUDE_roots_of_g_are_cubes_of_roots_of_f_l2059_205905


namespace NUMINAMATH_CALUDE_koschei_coins_theorem_l2059_205907

theorem koschei_coins_theorem :
  ∃! n : ℕ, 300 ≤ n ∧ n ≤ 400 ∧ n % 10 = 7 ∧ n % 12 = 9 :=
by sorry

end NUMINAMATH_CALUDE_koschei_coins_theorem_l2059_205907


namespace NUMINAMATH_CALUDE_lunch_percentage_l2059_205978

theorem lunch_percentage (total_students : ℕ) (total_students_pos : total_students > 0) :
  let boys := (6 : ℚ) / 10 * total_students
  let girls := (4 : ℚ) / 10 * total_students
  let boys_lunch := (60 : ℚ) / 100 * boys
  let girls_lunch := (40 : ℚ) / 100 * girls
  let total_lunch := boys_lunch + girls_lunch
  (total_lunch / total_students) * 100 = 52 := by
  sorry

end NUMINAMATH_CALUDE_lunch_percentage_l2059_205978


namespace NUMINAMATH_CALUDE_wheel_configuration_theorem_l2059_205926

/-- Represents a wheel with spokes -/
structure Wheel :=
  (spokes : ℕ)
  (spokes_le_three : spokes ≤ 3)

/-- Represents a configuration of wheels -/
def WheelConfiguration := List Wheel

/-- The total number of spokes in a configuration -/
def total_spokes (config : WheelConfiguration) : ℕ :=
  config.map Wheel.spokes |>.sum

/-- Theorem stating that 3 wheels are possible and 2 wheels are not possible -/
theorem wheel_configuration_theorem 
  (config : WheelConfiguration) 
  (total_spokes_ge_seven : total_spokes config ≥ 7) : 
  (∃ (three_wheel_config : WheelConfiguration), three_wheel_config.length = 3 ∧ total_spokes three_wheel_config ≥ 7) ∧
  (¬ ∃ (two_wheel_config : WheelConfiguration), two_wheel_config.length = 2 ∧ total_spokes two_wheel_config ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_wheel_configuration_theorem_l2059_205926


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2059_205903

theorem sum_of_cubes (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x + y + x^2*y + x*y^2 = 24) : 
  x^3 + y^3 = 68 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2059_205903


namespace NUMINAMATH_CALUDE_wheel_speed_problem_l2059_205937

theorem wheel_speed_problem (circumference : ℝ) (time_decrease : ℝ) (speed_increase : ℝ) :
  circumference = 15 →
  time_decrease = 1 / 3 →
  speed_increase = 10 →
  ∃ (original_speed : ℝ),
    original_speed * (circumference / 5280) = circumference / 5280 ∧
    (original_speed + speed_increase) * ((circumference / (5280 * original_speed)) - time_decrease / 3600) = circumference / 5280 ∧
    original_speed = 15 :=
by sorry

end NUMINAMATH_CALUDE_wheel_speed_problem_l2059_205937


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l2059_205983

/-- Given four positive numbers in sequence where the first is 4 and the last is 16,
    with two numbers inserted between them such that the first three form a geometric progression
    and the last three form a harmonic progression, prove that the sum of the inserted numbers is 8. -/
theorem inserted_numbers_sum (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧  -- x and y are positive
  (∃ r : ℝ, r > 0 ∧ x = 4 * r ∧ y = 4 * r^2) ∧  -- geometric progression
  2 / y = 1 / x + 1 / 16 →  -- harmonic progression
  x + y = 8 := by
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l2059_205983


namespace NUMINAMATH_CALUDE_find_number_l2059_205904

theorem find_number : ∃ x : ℝ, x - 2.95 - 2.95 = 9.28 ∧ x = 15.18 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2059_205904


namespace NUMINAMATH_CALUDE_decimal_to_fraction_sum_l2059_205965

theorem decimal_to_fraction_sum (m n : ℕ+) : 
  (m : ℚ) / (n : ℚ) = 1824 / 10000 → 
  ∀ (a b : ℕ+), (a : ℚ) / (b : ℚ) = 1824 / 10000 → 
  (a : ℕ) ≤ (m : ℕ) ∧ (b : ℕ) ≤ (n : ℕ) →
  m + n = 739 := by
sorry


end NUMINAMATH_CALUDE_decimal_to_fraction_sum_l2059_205965


namespace NUMINAMATH_CALUDE_cupcakes_brought_is_correct_l2059_205972

/-- The number of cupcakes Dani brought to her 2nd-grade class. -/
def cupcakes_brought : ℕ := 30

/-- The total number of students in the class, including Dani. -/
def total_students : ℕ := 27

/-- The number of teachers in the class. -/
def teachers : ℕ := 1

/-- The number of teacher's aids in the class. -/
def teacher_aids : ℕ := 1

/-- The number of students who called in sick. -/
def sick_students : ℕ := 3

/-- The number of cupcakes left after distribution. -/
def leftover_cupcakes : ℕ := 4

/-- Theorem stating that the number of cupcakes Dani brought is correct. -/
theorem cupcakes_brought_is_correct :
  cupcakes_brought = 
    (total_students - sick_students + teachers + teacher_aids) + leftover_cupcakes :=
by
  sorry

end NUMINAMATH_CALUDE_cupcakes_brought_is_correct_l2059_205972


namespace NUMINAMATH_CALUDE_stack_map_a_front_view_l2059_205991

/-- Represents a column of stacks in the Stack Map --/
def Column := List Nat

/-- Represents the Stack Map A --/
def StackMapA : List Column := [
  [3, 1],       -- First column
  [2, 2, 1],    -- Second column
  [1, 4, 2],    -- Third column
  [5]           -- Fourth column
]

/-- Calculates the front view of a Stack Map --/
def frontView (map : List Column) : List Nat :=
  map.map (List.foldl max 0)

/-- Theorem: The front view of Stack Map A is [3, 2, 4, 5] --/
theorem stack_map_a_front_view :
  frontView StackMapA = [3, 2, 4, 5] := by
  sorry

end NUMINAMATH_CALUDE_stack_map_a_front_view_l2059_205991


namespace NUMINAMATH_CALUDE_rectangle_cut_theorem_l2059_205957

/-- In a rectangle with a line cutting it, if the area of the resulting quadrilateral
    is 40% of the total area, then the height of this quadrilateral is 0.8 times
    the length of the rectangle. -/
theorem rectangle_cut_theorem (L W x : ℝ) : 
  L > 0 → W > 0 → x > 0 →
  x * W / 2 = 0.4 * L * W →
  x = 0.8 * L := by
sorry

end NUMINAMATH_CALUDE_rectangle_cut_theorem_l2059_205957


namespace NUMINAMATH_CALUDE_points_three_units_from_negative_two_l2059_205918

theorem points_three_units_from_negative_two : 
  ∀ x : ℝ, (abs (x - (-2)) = 3) ↔ (x = -5 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_points_three_units_from_negative_two_l2059_205918


namespace NUMINAMATH_CALUDE_smallest_positive_root_l2059_205962

theorem smallest_positive_root (b c d : ℝ) (hb : |b| ≤ 1) (hc : |c| ≤ 3) (hd : |d| ≤ 2) :
  ∃ (s : ℝ), s > 0 ∧ s^3 + b*s^2 + c*s + d = 0 ∧
  ∀ (x : ℝ), x > 0 ∧ x^3 + b*x^2 + c*x + d = 0 → s ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_root_l2059_205962


namespace NUMINAMATH_CALUDE_expected_digits_is_one_point_six_l2059_205944

/-- A fair 20-sided die -/
def icosahedralDie : Finset ℕ := Finset.range 20

/-- The probability of rolling any specific number on the die -/
def prob (n : ℕ) : ℚ := if n ∈ icosahedralDie then 1 / 20 else 0

/-- The number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else 3

/-- The expected number of digits when rolling the die -/
def expectedDigits : ℚ :=
  (icosahedralDie.sum fun n => prob n * numDigits n)

theorem expected_digits_is_one_point_six :
  expectedDigits = 8/5 := by sorry

end NUMINAMATH_CALUDE_expected_digits_is_one_point_six_l2059_205944


namespace NUMINAMATH_CALUDE_water_addition_proof_l2059_205928

/-- Proves that adding 23 litres of water to a 45-litre mixture with initial milk to water ratio of 4:1 results in a new mixture with milk to water ratio of 1.125 -/
theorem water_addition_proof (initial_volume : ℝ) (initial_ratio : ℚ) (water_added : ℝ) (final_ratio : ℚ) : 
  initial_volume = 45 ∧ 
  initial_ratio = 4/1 ∧ 
  water_added = 23 ∧ 
  final_ratio = 1125/1000 →
  let initial_milk := (initial_ratio / (initial_ratio + 1)) * initial_volume
  let initial_water := (1 / (initial_ratio + 1)) * initial_volume
  let final_water := initial_water + water_added
  initial_milk / final_water = final_ratio :=
by sorry

end NUMINAMATH_CALUDE_water_addition_proof_l2059_205928


namespace NUMINAMATH_CALUDE_fraction_inequality_l2059_205953

theorem fraction_inequality (a b m : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : m > 0) :
  (a + m) / (b + m) > a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2059_205953


namespace NUMINAMATH_CALUDE_no_nontrivial_solutions_l2059_205924

theorem no_nontrivial_solutions (x y z t : ℤ) :
  x^2 = 2*y^2 ∧ x^4 + 3*y^4 + 27*z^4 = 9*t^4 → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nontrivial_solutions_l2059_205924


namespace NUMINAMATH_CALUDE_mean_equality_implies_x_value_l2059_205908

theorem mean_equality_implies_x_value :
  let mean1 := (3 + 7 + 15) / 3
  let mean2 := (x + 10) / 2
  mean1 = mean2 → x = 20 / 3 :=
by sorry

end NUMINAMATH_CALUDE_mean_equality_implies_x_value_l2059_205908


namespace NUMINAMATH_CALUDE_stream_current_rate_l2059_205987

/-- Proves that the rate of a stream's current is 4 kmph given the conditions of a boat's travel --/
theorem stream_current_rate (distance_one_way : ℝ) (total_time : ℝ) (still_water_speed : ℝ) 
  (h1 : distance_one_way = 6)
  (h2 : total_time = 2)
  (h3 : still_water_speed = 8) :
  ∃ c : ℝ, c = 4 ∧ 
    (distance_one_way / (still_water_speed - c) + distance_one_way / (still_water_speed + c) = total_time) :=
by sorry

end NUMINAMATH_CALUDE_stream_current_rate_l2059_205987


namespace NUMINAMATH_CALUDE_multiply_powers_of_a_l2059_205976

theorem multiply_powers_of_a (a : ℝ) : 5 * a^3 * (3 * a^3) = 15 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_of_a_l2059_205976


namespace NUMINAMATH_CALUDE_residue_products_l2059_205996

theorem residue_products (n k : ℕ+) : 
  (∃ (a : Fin n → ℤ) (b : Fin k → ℤ), 
    ∀ (i j i' j' : ℕ) (hi : i < n) (hj : j < k) (hi' : i' < n) (hj' : j' < k),
      (i ≠ i' ∨ j ≠ j') → 
      (a ⟨i, hi⟩ * b ⟨j, hj⟩) % (n * k : ℕ) ≠ (a ⟨i', hi'⟩ * b ⟨j', hj'⟩) % (n * k : ℕ)) ↔ 
  Nat.gcd n k = 1 :=
sorry

end NUMINAMATH_CALUDE_residue_products_l2059_205996


namespace NUMINAMATH_CALUDE_probability_at_most_one_first_class_l2059_205900

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of products -/
def total_products : ℕ := 5

/-- The number of first-class products -/
def first_class_products : ℕ := 3

/-- The number of second-class products -/
def second_class_products : ℕ := 2

/-- The number of products to be selected -/
def selected_products : ℕ := 2

theorem probability_at_most_one_first_class :
  (choose first_class_products 1 * choose second_class_products 1 + choose second_class_products 2) /
  choose total_products selected_products = 7 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_at_most_one_first_class_l2059_205900


namespace NUMINAMATH_CALUDE_circle_radius_range_l2059_205980

-- Define the circle C
def Circle (r : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define points A and B
def A : ℝ × ℝ := (6, 0)
def B : ℝ × ℝ := (0, 8)

-- Define the line segment AB
def LineSegmentAB := {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • A + t • B}

-- Define the condition for points M and N
def ExistsMN (r : ℝ) (P : ℝ × ℝ) :=
  ∃ M N : ℝ × ℝ, M ∈ Circle r ∧ N ∈ Circle r ∧ P.1 - M.1 = N.1 - M.1 ∧ P.2 - M.2 = N.2 - M.2

-- State the theorem
theorem circle_radius_range :
  ∀ r : ℝ, (∀ P ∈ LineSegmentAB, ExistsMN r P) ↔ (8/3 ≤ r ∧ r < 12/5) :=
sorry

end NUMINAMATH_CALUDE_circle_radius_range_l2059_205980


namespace NUMINAMATH_CALUDE_smallest_valid_number_l2059_205969

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧
  ∃ k, n = k * (lcm 3 (lcm 4 (lcm 5 (lcm 6 7)))) + 1

theorem smallest_valid_number : 
  is_valid_number 61 ∧ ∀ m, is_valid_number m → m ≥ 61 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l2059_205969


namespace NUMINAMATH_CALUDE_genevieve_errors_fixed_l2059_205993

/-- Represents the number of errors fixed by a programmer -/
def errors_fixed (total_lines : ℕ) (debug_interval : ℕ) (errors_per_debug : ℕ) : ℕ :=
  (total_lines / debug_interval) * errors_per_debug

/-- Theorem stating the number of errors fixed by Genevieve -/
theorem genevieve_errors_fixed :
  errors_fixed 4300 100 3 = 129 := by
  sorry

end NUMINAMATH_CALUDE_genevieve_errors_fixed_l2059_205993


namespace NUMINAMATH_CALUDE_inner_circle_to_triangle_ratio_l2059_205934

/-- The ratio of the area of the innermost circle to the area of the equilateral triangle --/
theorem inner_circle_to_triangle_ratio (s : ℝ) (h : s = 10) :
  let R := s * Real.sqrt 3 / 6
  let a := 2 * R
  let r := a / 2
  let A_triangle := Real.sqrt 3 / 4 * s^2
  let A_circle := Real.pi * r^2
  A_circle / A_triangle = Real.pi * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_inner_circle_to_triangle_ratio_l2059_205934


namespace NUMINAMATH_CALUDE_team_selection_with_girl_l2059_205906

theorem team_selection_with_girl (n m k : ℕ) (hn : n = 5) (hm : m = 5) (hk : k = 3) :
  Nat.choose (n + m) k - Nat.choose n k = 110 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_with_girl_l2059_205906


namespace NUMINAMATH_CALUDE_game_cost_l2059_205986

def initial_money : ℕ := 63
def toy_price : ℕ := 3
def toys_affordable : ℕ := 5

def remaining_money : ℕ := toy_price * toys_affordable

theorem game_cost : initial_money - remaining_money = 48 := by
  sorry

end NUMINAMATH_CALUDE_game_cost_l2059_205986


namespace NUMINAMATH_CALUDE_abc_system_property_l2059_205902

theorem abc_system_property (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (eq1 : a^2 + a = b^2)
  (eq2 : b^2 + b = c^2)
  (eq3 : c^2 + c = a^2) :
  (a - b) * (b - c) * (c - a) = 1 := by
sorry

end NUMINAMATH_CALUDE_abc_system_property_l2059_205902


namespace NUMINAMATH_CALUDE_tan_alpha_values_l2059_205951

theorem tan_alpha_values (α : Real) 
  (h : 2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + 5 * Real.cos α ^ 2 = 3) : 
  Real.tan α = 1 ∨ Real.tan α = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_values_l2059_205951


namespace NUMINAMATH_CALUDE_estimated_value_reasonable_l2059_205921

/-- The lower bound of the scale -/
def lower_bound : ℝ := 9.80

/-- The upper bound of the scale -/
def upper_bound : ℝ := 10.0

/-- The estimated value -/
def estimated_value : ℝ := 9.95

/-- Theorem stating that the estimated value is a reasonable approximation -/
theorem estimated_value_reasonable :
  lower_bound < estimated_value ∧
  estimated_value < upper_bound ∧
  (estimated_value - lower_bound) > (upper_bound - estimated_value) :=
by sorry

end NUMINAMATH_CALUDE_estimated_value_reasonable_l2059_205921


namespace NUMINAMATH_CALUDE_complex_subtraction_l2059_205946

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + I) :
  a - 3*b = -1 - 6*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l2059_205946


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2059_205982

theorem quadratic_factorization (c d : ℤ) : 
  (∀ x, 25*x^2 - 85*x - 90 = (5*x + c) * (5*x + d)) → c + 2*d = -24 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2059_205982


namespace NUMINAMATH_CALUDE_rhinoceros_grazing_area_l2059_205923

theorem rhinoceros_grazing_area 
  (initial_population : ℕ) 
  (watering_area : ℕ) 
  (population_increase_rate : ℚ) 
  (total_preserve_area : ℕ) 
  (h1 : initial_population = 8000)
  (h2 : watering_area = 10000)
  (h3 : population_increase_rate = 1/10)
  (h4 : total_preserve_area = 890000) :
  let final_population := initial_population + initial_population * population_increase_rate
  let grazing_area := total_preserve_area - watering_area
  grazing_area / final_population = 100 := by
sorry

end NUMINAMATH_CALUDE_rhinoceros_grazing_area_l2059_205923


namespace NUMINAMATH_CALUDE_line_PB_equation_l2059_205989

-- Define the points A, B, and P
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (5, 0)
def P : ℝ × ℝ := (2, 3)

-- Define the equations of lines PA and PB
def line_PA (x y : ℝ) : Prop := x - y + 1 = 0
def line_PB (x y : ℝ) : Prop := x + y - 5 = 0

-- State the theorem
theorem line_PB_equation :
  (A.1 = -1 ∧ A.2 = 0) →  -- A is on x-axis
  (B.1 = 5 ∧ B.2 = 0) →   -- B is on x-axis
  P.1 = 2 →               -- x-coordinate of P is 2
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 →  -- PA = PB
  (∀ x y, line_PA x y ↔ x - y + 1 = 0) →  -- Equation of PA
  (∀ x y, line_PB x y ↔ x + y - 5 = 0) :=  -- Equation of PB
by sorry

end NUMINAMATH_CALUDE_line_PB_equation_l2059_205989


namespace NUMINAMATH_CALUDE_perimeter_of_8x4_formation_l2059_205974

/-- A rectangular formation of students -/
structure Formation :=
  (rows : ℕ)
  (columns : ℕ)

/-- The number of elements on the perimeter of a formation -/
def perimeter_count (f : Formation) : ℕ :=
  2 * (f.rows + f.columns) - 4

theorem perimeter_of_8x4_formation :
  let f : Formation := ⟨8, 4⟩
  perimeter_count f = 20 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_8x4_formation_l2059_205974


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l2059_205966

theorem right_triangle_ratio (x y : ℝ) (h1 : x > y) (h2 : y > 0) 
  (h3 : (x - y)^2 + x^2 = (x + y)^2) : x = 4 * y := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l2059_205966


namespace NUMINAMATH_CALUDE_f_properties_l2059_205998

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def has_max_value (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f x ≤ c

-- State the theorem
theorem f_properties (hsym : symmetric_about_origin f)
                     (hinc : increasing_on f 3 7)
                     (hmax : has_max_value f 3 7 5) :
  increasing_on f (-7) (-3) ∧ has_max_value f (-7) (-3) (-5) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2059_205998


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l2059_205947

theorem fraction_sum_theorem (a b c : ℝ) 
  (h : a / (35 - a) + b / (55 - b) + c / (70 - c) = 8) :
  7 / (35 - a) + 11 / (55 - b) + 14 / (70 - c) = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l2059_205947


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2059_205968

/-- A complex number ω such that ω^2 + ω + 1 = 0 -/
noncomputable def ω : ℂ := sorry

/-- The property that ω^2 + ω + 1 = 0 -/
axiom ω_property : ω^2 + ω + 1 = 0

/-- The polynomial x^104 + Ax^3 + Bx -/
def polynomial (A B : ℝ) (x : ℂ) : ℂ := x^104 + A * x^3 + B * x

/-- The divisibility condition -/
def is_divisible (A B : ℝ) : Prop :=
  polynomial A B ω = 0

theorem polynomial_divisibility (A B : ℝ) :
  is_divisible A B → A + B = 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2059_205968


namespace NUMINAMATH_CALUDE_max_value_implies_ratio_l2059_205952

/-- The function f(x) = x³ + ax² + bx - a² - 7a -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

/-- f(x) reaches a maximum value of 10 at x = 1 -/
def max_at_one (a b : ℝ) : Prop :=
  (∀ x, f a b x ≤ f a b 1) ∧ f a b 1 = 10

theorem max_value_implies_ratio (a b : ℝ) (h : max_at_one a b) : a / b = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_ratio_l2059_205952


namespace NUMINAMATH_CALUDE_outfit_combinations_l2059_205909

theorem outfit_combinations (short_sleeve : ℕ) (long_sleeve : ℕ) (jeans : ℕ) (formal_trousers : ℕ) :
  short_sleeve = 5 →
  long_sleeve = 3 →
  jeans = 6 →
  formal_trousers = 2 →
  (short_sleeve + long_sleeve) * (jeans + formal_trousers) = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2059_205909


namespace NUMINAMATH_CALUDE_fraction_power_four_l2059_205930

theorem fraction_power_four : (5 / 3 : ℚ) ^ 4 = 625 / 81 := by sorry

end NUMINAMATH_CALUDE_fraction_power_four_l2059_205930


namespace NUMINAMATH_CALUDE_seaweed_for_fires_l2059_205999

theorem seaweed_for_fires (total_seaweed livestock_feed : ℝ)
  (h1 : total_seaweed = 400)
  (h2 : livestock_feed = 150)
  (h3 : livestock_feed = 0.75 * (1 - fire_percentage / 100) * total_seaweed) :
  fire_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_seaweed_for_fires_l2059_205999


namespace NUMINAMATH_CALUDE_book_sale_profit_percentage_l2059_205901

/-- Calculates the profit percentage for a book sale with given parameters. -/
theorem book_sale_profit_percentage
  (purchase_price : ℝ)
  (purchase_tax_rate : ℝ)
  (shipping_fee : ℝ)
  (selling_price : ℝ)
  (trading_tax_rate : ℝ)
  (h1 : purchase_price = 32)
  (h2 : purchase_tax_rate = 0.05)
  (h3 : shipping_fee = 2.5)
  (h4 : selling_price = 56)
  (h5 : trading_tax_rate = 0.07)
  : ∃ (profit_percentage : ℝ), abs (profit_percentage - 44.26) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_book_sale_profit_percentage_l2059_205901


namespace NUMINAMATH_CALUDE_solve_equation_l2059_205992

theorem solve_equation (n : ℤ) : (n + 1999) / 2 = -1 → n = -2001 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2059_205992


namespace NUMINAMATH_CALUDE_smallest_hot_dog_packages_l2059_205927

theorem smallest_hot_dog_packages : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → 5 * m % 7 = 0 → m ≥ n) ∧ 5 * n % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_hot_dog_packages_l2059_205927


namespace NUMINAMATH_CALUDE_toys_per_rabbit_l2059_205910

-- Define the number of rabbits
def num_rabbits : ℕ := 16

-- Define the number of toys bought on Monday
def monday_toys : ℕ := 6

-- Define the number of toys bought on Wednesday
def wednesday_toys : ℕ := 2 * monday_toys

-- Define the number of toys bought on Friday
def friday_toys : ℕ := 4 * monday_toys

-- Define the number of toys bought on Saturday
def saturday_toys : ℕ := wednesday_toys / 2

-- Define the total number of toys
def total_toys : ℕ := monday_toys + wednesday_toys + friday_toys + saturday_toys

-- Theorem statement
theorem toys_per_rabbit : total_toys / num_rabbits = 3 := by
  sorry

end NUMINAMATH_CALUDE_toys_per_rabbit_l2059_205910


namespace NUMINAMATH_CALUDE_maries_socks_l2059_205984

theorem maries_socks (x y z : ℕ) : 
  x + y + z = 15 →
  2 * x + 3 * y + 5 * z = 36 →
  x ≥ 1 →
  y ≥ 1 →
  z ≥ 1 →
  x = 11 := by
sorry

end NUMINAMATH_CALUDE_maries_socks_l2059_205984


namespace NUMINAMATH_CALUDE_sequence_decreasing_equivalence_l2059_205973

def IsDecreasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a (n + 1) < a n

theorem sequence_decreasing_equivalence (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) ↔ IsDecreasing a :=
sorry

end NUMINAMATH_CALUDE_sequence_decreasing_equivalence_l2059_205973


namespace NUMINAMATH_CALUDE_first_group_size_first_group_size_proof_l2059_205948

/-- Given two groups of workers building walls, this theorem proves that the number of workers in the first group is 20, based on the given conditions. -/
theorem first_group_size : ℕ :=
  let wall_length_1 : ℝ := 66
  let days_1 : ℕ := 4
  let wall_length_2 : ℝ := 567.6
  let days_2 : ℕ := 8
  let workers_2 : ℕ := 86
  let workers_1 := (wall_length_1 * days_2 * workers_2) / (wall_length_2 * days_1)
  20

/-- Proof of the theorem -/
theorem first_group_size_proof : first_group_size = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_first_group_size_proof_l2059_205948


namespace NUMINAMATH_CALUDE_inner_rectangle_area_l2059_205936

theorem inner_rectangle_area (a b : ℕ) : 
  a > 2 → 
  b > 2 → 
  (3 * a + 4) * (b + 3) = 65 → 
  a * b = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_inner_rectangle_area_l2059_205936


namespace NUMINAMATH_CALUDE_mixture_ratio_after_replacement_l2059_205971

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents the state of the liquid mixture -/
structure LiquidMixture where
  ratioAB : Ratio
  volumeA : ℝ
  totalVolume : ℝ

def initialMixture : LiquidMixture :=
  { ratioAB := { numerator := 4, denominator := 1 }
  , volumeA := 32
  , totalVolume := 40 }

def replacementVolume : ℝ := 20

/-- Calculates the new ratio after replacing some mixture with liquid B -/
def newRatio (initial : LiquidMixture) (replace : ℝ) : Ratio :=
  { numerator := 2
  , denominator := 3 }

theorem mixture_ratio_after_replacement :
  newRatio initialMixture replacementVolume = { numerator := 2, denominator := 3 } :=
sorry

end NUMINAMATH_CALUDE_mixture_ratio_after_replacement_l2059_205971


namespace NUMINAMATH_CALUDE_onion_basket_change_l2059_205943

/-- The net change in the number of onions in Sara's basket -/
def net_change (sara_added : ℤ) (sally_removed : ℤ) (fred_added : ℤ) : ℤ :=
  sara_added - sally_removed + fred_added

/-- Theorem stating that the net change in onions is 8 -/
theorem onion_basket_change :
  net_change 4 5 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_onion_basket_change_l2059_205943


namespace NUMINAMATH_CALUDE_sequence_parity_l2059_205950

def T : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => T (n + 2) + T (n + 1) - T n

theorem sequence_parity :
  (T 2021 % 2 = 1) ∧ (T 2022 % 2 = 0) ∧ (T 2023 % 2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_parity_l2059_205950


namespace NUMINAMATH_CALUDE_fixed_ray_exists_l2059_205935

/-- Represents a circle with a color -/
structure ColoredCircle where
  center : ℝ × ℝ
  radius : ℝ
  color : Bool

/-- Represents an angle with colored sides -/
structure ColoredAngle where
  vertex : ℝ × ℝ
  side1 : ℝ × ℝ → Prop
  side2 : ℝ × ℝ → Prop
  color1 : Bool
  color2 : Bool

/-- Represents a configuration of circles and an angle -/
structure Configuration where
  circle1 : ColoredCircle
  circle2 : ColoredCircle
  angle : ColoredAngle

/-- Predicate to check if circles are non-overlapping -/
def non_overlapping (c1 c2 : ColoredCircle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 > (c1.radius + c2.radius) ^ 2

/-- Predicate to check if a point is outside an angle -/
def outside_angle (p : ℝ × ℝ) (a : ColoredAngle) : Prop :=
  ¬a.side1 p ∧ ¬a.side2 p

/-- Predicate to check if a side touches a circle -/
def touches (side : ℝ × ℝ → Prop) (c : ColoredCircle) : Prop :=
  ∃ p : ℝ × ℝ, side p ∧ (p.1 - c.center.1) ^ 2 + (p.2 - c.center.2) ^ 2 = c.radius ^ 2

/-- Main theorem statement -/
theorem fixed_ray_exists (config : Configuration) 
  (h1 : non_overlapping config.circle1 config.circle2)
  (h2 : config.circle1.color ≠ config.circle2.color)
  (h3 : config.angle.color1 = config.circle1.color)
  (h4 : config.angle.color2 = config.circle2.color)
  (h5 : outside_angle config.circle1.center config.angle)
  (h6 : outside_angle config.circle2.center config.angle)
  (h7 : touches config.angle.side1 config.circle1)
  (h8 : touches config.angle.side2 config.circle2)
  (h9 : config.angle.vertex ≠ config.circle1.center)
  (h10 : config.angle.vertex ≠ config.circle2.center) :
  ∃ (ray : ℝ × ℝ → Prop), ∀ (config' : Configuration), 
    (config'.circle1 = config.circle1 ∧ 
     config'.circle2 = config.circle2 ∧
     config'.angle.vertex = config.angle.vertex ∧
     touches config'.angle.side1 config'.circle1 ∧
     touches config'.angle.side2 config'.circle2) →
    ∃ p : ℝ × ℝ, ray p ∧ 
      (∃ t : ℝ, t > 0 ∧ p = (config'.angle.vertex.1 + t * (p.1 - config'.angle.vertex.1),
                             config'.angle.vertex.2 + t * (p.2 - config'.angle.vertex.2))) :=
sorry

end NUMINAMATH_CALUDE_fixed_ray_exists_l2059_205935


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2059_205964

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]

theorem sum_first_six_primes_mod_seventh_prime : 
  (List.sum (List.take 6 first_seven_primes)) % (List.get! first_seven_primes 6) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2059_205964
