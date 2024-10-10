import Mathlib

namespace simplify_calculations_l1929_192959

theorem simplify_calculations :
  ((999 : ℕ) * 999 + 1999 = 1000000) ∧
  ((9 : ℕ) * 72 * 125 = 81000) ∧
  ((416 : ℤ) - 327 + 184 - 273 = 0) := by
  sorry

end simplify_calculations_l1929_192959


namespace marbles_choice_count_l1929_192938

/-- The number of ways to choose marbles under specific conditions -/
def choose_marbles (total : ℕ) (red green blue : ℕ) (choose : ℕ) : ℕ :=
  let other := total - (red + green + blue)
  let color_pairs := (red * green + red * blue + green * blue)
  let remaining_choices := Nat.choose (other + red - 1 + green - 1 + blue - 1) (choose - 2)
  color_pairs * remaining_choices

/-- Theorem stating the number of ways to choose marbles under given conditions -/
theorem marbles_choice_count :
  choose_marbles 15 2 2 2 5 = 495 := by sorry

end marbles_choice_count_l1929_192938


namespace product_of_eight_consecutive_integers_divisible_by_80_l1929_192993

theorem product_of_eight_consecutive_integers_divisible_by_80 (n : ℕ) : 
  80 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) * (n + 7)) := by
  sorry

end product_of_eight_consecutive_integers_divisible_by_80_l1929_192993


namespace pens_per_student_l1929_192982

theorem pens_per_student (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ) :
  total_pens = 100 →
  total_pencils = 50 →
  max_students = 50 →
  total_pens / max_students = 2 :=
by sorry

end pens_per_student_l1929_192982


namespace hair_cut_total_l1929_192947

theorem hair_cut_total (first_cut second_cut : ℝ) 
  (h1 : first_cut = 0.375)
  (h2 : second_cut = 0.5) : 
  first_cut + second_cut = 0.875 := by
  sorry

end hair_cut_total_l1929_192947


namespace tangent_product_inequality_l1929_192984

theorem tangent_product_inequality (α β γ : Real) (h : α + β + γ = Real.pi) :
  Real.tan (α/2) * Real.tan (β/2) * Real.tan (γ/2) ≤ Real.sqrt 3 / 9 := by
  sorry

end tangent_product_inequality_l1929_192984


namespace polynomial_equality_solutions_l1929_192985

theorem polynomial_equality_solutions : 
  ∀ (a b c : ℤ), 
  (∀ x : ℤ, (x - a) * (x - 8) + 4 = (x + b) * (x + c)) → 
  ((a = 20 ∧ b = -6 ∧ c = -6) ∨ (a = 29 ∧ b = -9 ∧ c = -12)) :=
by sorry

end polynomial_equality_solutions_l1929_192985


namespace bicycle_speed_problem_l1929_192933

theorem bicycle_speed_problem (distance : ℝ) (speed_ratio : ℝ) (time_diff : ℝ) :
  distance = 12 →
  speed_ratio = 1.2 →
  time_diff = 1/6 →
  ∃ (speed_B : ℝ),
    distance / speed_B - distance / (speed_ratio * speed_B) = time_diff ∧
    speed_B = 12 := by
  sorry

end bicycle_speed_problem_l1929_192933


namespace system_of_inequalities_solution_l1929_192998

theorem system_of_inequalities_solution :
  ∀ x y : ℤ,
    (2 * x - y > 3 ∧ 3 - 2 * x + y > 0) ↔ ((x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1)) :=
by sorry

end system_of_inequalities_solution_l1929_192998


namespace binary_encodes_to_032239_l1929_192910

/-- Represents a mapping from characters to digits -/
def EncodeMap := Char → Nat

/-- The encoding scheme based on "MONITOR KEYBOARD" -/
def monitorKeyboardEncode : EncodeMap :=
  fun c => match c with
  | 'M' => 0
  | 'O' => 1
  | 'N' => 2
  | 'I' => 3
  | 'T' => 4
  | 'R' => 6
  | 'K' => 7
  | 'E' => 8
  | 'Y' => 9
  | 'B' => 0
  | 'A' => 2
  | 'D' => 4
  | _ => 0  -- Default case, should not be reached for valid inputs

/-- Encodes a string to a list of digits using the given encoding map -/
def encodeString (encode : EncodeMap) (s : String) : List Nat :=
  s.data.map encode

/-- Converts a list of digits to a natural number -/
def digitsToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

/-- The main theorem: BINARY encodes to 032239 -/
theorem binary_encodes_to_032239 :
  digitsToNat (encodeString monitorKeyboardEncode "BINARY") = 032239 := by
  sorry


end binary_encodes_to_032239_l1929_192910


namespace min_matchsticks_removal_theorem_l1929_192913

/-- Represents a configuration of matchsticks forming triangles -/
structure MatchstickConfiguration where
  total_matchsticks : ℕ
  total_triangles : ℕ

/-- Represents the minimum number of matchsticks to remove -/
def min_matchsticks_to_remove (config : MatchstickConfiguration) : ℕ := sorry

/-- The theorem to be proved -/
theorem min_matchsticks_removal_theorem (config : MatchstickConfiguration) 
  (h1 : config.total_matchsticks = 42)
  (h2 : config.total_triangles = 38) :
  min_matchsticks_to_remove config ≥ 12 := by sorry

end min_matchsticks_removal_theorem_l1929_192913


namespace parabola_transformation_l1929_192916

def original_parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 + 2

def shift_left (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f (x + k)

def shift_up (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f x + k

def transformed_parabola : ℝ → ℝ :=
  shift_up (shift_left original_parabola 3) 2

theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = 2 * x^2 + 4 := by
  sorry

end parabola_transformation_l1929_192916


namespace line_symmetry_l1929_192979

/-- Given a line l₁: y = 2x + 1 and a point p: (1, 1), 
    the line l₂: y = 2x - 3 is symmetric to l₁ about p -/
theorem line_symmetry (x y : ℝ) : 
  (y = 2*x + 1) → 
  (∃ (x' y' : ℝ), y' = 2*x' - 3 ∧ 
    ((x + x') / 2 = 1 ∧ (y + y') / 2 = 1)) :=
by sorry

end line_symmetry_l1929_192979


namespace arcsin_sin_eq_solution_l1929_192958

theorem arcsin_sin_eq_solution (x : ℝ) : 
  Real.arcsin (Real.sin x) = (3 * x) / 4 ∧ 
  -(π / 2) ≤ (3 * x) / 4 ∧ 
  (3 * x) / 4 ≤ π / 2 → 
  x = 0 := by
sorry

end arcsin_sin_eq_solution_l1929_192958


namespace min_sum_squares_l1929_192911

-- Define the points A, B, C, D, E as real numbers representing their positions on a line
def A : ℝ := 0
def B : ℝ := 1
def C : ℝ := 3
def D : ℝ := 6
def E : ℝ := 10

-- Define the function to be minimized
def f (x : ℝ) : ℝ := (x - A)^2 + (x - B)^2 + (x - C)^2 + (x - D)^2 + (x - E)^2

-- State the theorem
theorem min_sum_squares :
  ∃ (min : ℝ), min = 60 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

end min_sum_squares_l1929_192911


namespace basketball_handshakes_l1929_192944

/-- Calculates the total number of handshakes in a basketball game scenario -/
theorem basketball_handshakes (team_size : ℕ) (referee_count : ℕ) : team_size = 7 ∧ referee_count = 3 →
  team_size * team_size + 2 * team_size * referee_count = 91 := by
  sorry

#check basketball_handshakes

end basketball_handshakes_l1929_192944


namespace distance_to_y_axis_l1929_192970

/-- Given two perpendicular lines, prove that the distance from (m, 1) to the y-axis is 0 or 5 -/
theorem distance_to_y_axis (m : ℝ) : 
  (∃ x y, mx - (x + 2) * y + 2 = 0 ∧ 3 * x - m * y - 1 = 0) →  -- Lines exist
  (∀ x₁ y₁ x₂ y₂, mx₁ - (x₁ + 2) * y₁ + 2 = 0 ∧ 3 * x₂ - m * y₂ - 1 = 0 → 
    (m * 3 + m * (m + 2) = 0)) →  -- Lines are perpendicular
  (abs m = 0 ∨ abs m = 5) :=  -- Distance from (m, 1) to y-axis is 0 or 5
by sorry

end distance_to_y_axis_l1929_192970


namespace product_mod_seventeen_l1929_192945

theorem product_mod_seventeen :
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 14 := by
  sorry

end product_mod_seventeen_l1929_192945


namespace sequence_property_l1929_192977

theorem sequence_property (a : ℕ → ℕ) 
  (h_bijective : Function.Bijective a) 
  (h_positive : ∀ n, a n > 0) : 
  ∃ ℓ m : ℕ, 1 < ℓ ∧ ℓ < m ∧ a 1 + a m = 2 * a ℓ := by
  sorry

end sequence_property_l1929_192977


namespace profit_growth_rate_l1929_192949

/-- The average monthly growth rate of profit from March to May -/
def average_growth_rate : ℝ := 0.2

/-- The profit in March -/
def march_profit : ℝ := 5000

/-- The profit in May -/
def may_profit : ℝ := 7200

/-- The number of months between March and May -/
def months_between : ℕ := 2

theorem profit_growth_rate :
  march_profit * (1 + average_growth_rate) ^ months_between = may_profit :=
sorry

end profit_growth_rate_l1929_192949


namespace quadratic_propositions_l1929_192943

/-- Proposition p: The equation x^2 - 4mx + 1 = 0 has real solutions -/
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 4*m*x + 1 = 0

/-- Proposition q: There exists some x₀ ∈ ℝ such that mx₀² - 2x₀ - 1 > 0 -/
def q (m : ℝ) : Prop := ∃ x₀ : ℝ, m*x₀^2 - 2*x₀ - 1 > 0

theorem quadratic_propositions (m : ℝ) :
  (p m ↔ (m ≥ 1/2 ∨ m ≤ -1/2)) ∧
  (q m ↔ m > -1) ∧
  ((p m ↔ ¬q m) → (-1 < m ∧ m < 1/2)) :=
sorry

end quadratic_propositions_l1929_192943


namespace total_cost_is_21_l1929_192953

/-- The cost of a single carnation in dollars -/
def single_carnation_cost : ℚ := 0.50

/-- The cost of a dozen carnations in dollars -/
def dozen_carnation_cost : ℚ := 4.00

/-- The number of teachers Georgia is sending carnations to -/
def number_of_teachers : ℕ := 5

/-- The number of friends Georgia is buying carnations for -/
def number_of_friends : ℕ := 14

/-- The total cost of Georgia's carnation purchases -/
def total_cost : ℚ := dozen_carnation_cost * number_of_teachers + 
  single_carnation_cost * (number_of_friends % 12)

/-- Theorem stating that the total cost is $21.00 -/
theorem total_cost_is_21 : total_cost = 21 := by sorry

end total_cost_is_21_l1929_192953


namespace sum_of_reciprocal_pair_l1929_192902

theorem sum_of_reciprocal_pair (a b : ℝ) : 
  a > 0 → b > 0 → a * b = 1 → (3 * a + 2 * b) * (3 * b + 2 * a) = 295 → a + b = 7 := by
sorry

end sum_of_reciprocal_pair_l1929_192902


namespace bridge_length_l1929_192924

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time_s : ℝ)
  (h1 : train_length = 130)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time_s = 30) :
  train_speed_kmh * (1000 / 3600) * crossing_time_s - train_length = 245 :=
by sorry

end bridge_length_l1929_192924


namespace wenlock_years_ago_l1929_192969

/-- The year when the Wenlock Olympian Games were first held -/
def wenlock_first_year : ℕ := 1850

/-- The reference year (when the Olympic Games mascot 'Wenlock' was named) -/
def reference_year : ℕ := 2012

/-- The number of years between the first Wenlock Olympian Games and the reference year -/
def years_difference : ℕ := reference_year - wenlock_first_year

theorem wenlock_years_ago : years_difference = 162 := by
  sorry

end wenlock_years_ago_l1929_192969


namespace exactly_two_solutions_l1929_192994

-- Define the system of equations
def satisfies_system (x y : ℝ) : Prop :=
  x + 2*y = 2 ∧ |abs x - 2*(abs y)| = 2

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {pair | satisfies_system pair.1 pair.2}

-- Theorem statement
theorem exactly_two_solutions :
  ∃ (a b c d : ℝ), 
    solution_set = {(a, b), (c, d)} ∧
    (a, b) ≠ (c, d) ∧
    ∀ (x y : ℝ), (x, y) ∈ solution_set → (x, y) = (a, b) ∨ (x, y) = (c, d) :=
sorry

end exactly_two_solutions_l1929_192994


namespace sum_of_squares_of_roots_l1929_192999

theorem sum_of_squares_of_roots (x y : ℝ) : 
  (3 * x^2 - 7 * x + 5 = 0) → 
  (3 * y^2 - 7 * y + 5 = 0) → 
  (x^2 + y^2 = 19/9) := by
sorry

end sum_of_squares_of_roots_l1929_192999


namespace range_of_m_l1929_192951

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 10}

theorem range_of_m (m : ℝ) :
  (A m ∪ B = B) ↔ m ≤ 11/3 := by sorry

end range_of_m_l1929_192951


namespace longest_side_of_triangle_l1929_192930

theorem longest_side_of_triangle (y : ℝ) : 
  10 + (y + 6) + (3*y + 2) = 45 →
  max 10 (max (y + 6) (3*y + 2)) = 22.25 := by
sorry

end longest_side_of_triangle_l1929_192930


namespace magical_points_on_quadratic_unique_magical_point_condition_l1929_192941

/-- Definition of a magical point -/
def is_magical_point (x y : ℝ) : Prop := y = 2 * x

/-- The quadratic function y = x^2 - x - 4 -/
def quadratic_function (x : ℝ) : ℝ := x^2 - x - 4

/-- The generalized quadratic function y = tx^2 + (t-2)x - 4 -/
def generalized_quadratic_function (t x : ℝ) : ℝ := t * x^2 + (t - 2) * x - 4

theorem magical_points_on_quadratic :
  ∀ x y : ℝ, is_magical_point x y ∧ y = quadratic_function x ↔ (x = -1 ∧ y = -2) ∨ (x = 4 ∧ y = 8) :=
sorry

theorem unique_magical_point_condition :
  ∀ t : ℝ, t ≠ 0 →
  (∃! x y : ℝ, is_magical_point x y ∧ y = generalized_quadratic_function t x) ↔ t = -4 :=
sorry

end magical_points_on_quadratic_unique_magical_point_condition_l1929_192941


namespace f_monotone_range_l1929_192919

/-- The function f(x) defined as x^2 + a|x-1| -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * abs (x - 1)

/-- The theorem stating the range of 'a' for which f is monotonically increasing on [0, +∞) -/
theorem f_monotone_range (a : ℝ) :
  (∀ x y, 0 ≤ x ∧ x ≤ y → f a x ≤ f a y) ↔ -2 ≤ a ∧ a ≤ 0 := by
  sorry

end f_monotone_range_l1929_192919


namespace lilies_per_centerpiece_is_six_l1929_192926

/-- Calculates the number of lilies per centerpiece given the following conditions:
  * There are 6 centerpieces
  * Each centerpiece uses 8 roses
  * Each centerpiece uses twice as many orchids as roses
  * The total budget is $2700
  * Each flower costs $15
-/
def lilies_per_centerpiece (num_centerpieces : ℕ) (roses_per_centerpiece : ℕ) 
  (orchid_ratio : ℕ) (total_budget : ℕ) (flower_cost : ℕ) : ℕ :=
  let total_roses := num_centerpieces * roses_per_centerpiece
  let total_orchids := num_centerpieces * roses_per_centerpiece * orchid_ratio
  let rose_orchid_cost := (total_roses + total_orchids) * flower_cost
  let remaining_budget := total_budget - rose_orchid_cost
  let total_lilies := remaining_budget / flower_cost
  total_lilies / num_centerpieces

/-- Theorem stating that given the specific conditions, the number of lilies per centerpiece is 6 -/
theorem lilies_per_centerpiece_is_six :
  lilies_per_centerpiece 6 8 2 2700 15 = 6 := by
  sorry

end lilies_per_centerpiece_is_six_l1929_192926


namespace ice_cream_sundaes_l1929_192992

/-- The number of ice cream flavors -/
def n : ℕ := 8

/-- The number of scoops in each sundae -/
def k : ℕ := 2

/-- The number of unique two scoop sundaes -/
def unique_sundaes : ℕ := Nat.choose n k

theorem ice_cream_sundaes :
  unique_sundaes = 28 := by sorry

end ice_cream_sundaes_l1929_192992


namespace madeline_unused_crayons_l1929_192973

theorem madeline_unused_crayons :
  let box1to3 := 3 * 30 * (1/2 : ℚ)
  let box4to5 := 2 * 36 * (3/4 : ℚ)
  let box6to7 := 2 * 40 * (2/5 : ℚ)
  let box8 := 1 * 45 * (5/9 : ℚ)
  let box9to10 := 2 * 48 * (7/8 : ℚ)
  let box11 := 1 * 27 * (5/6 : ℚ)
  let box12 := 1 * 54 * (1/2 : ℚ)
  let total_unused := box1to3 + box4to5 + box6to7 + box8 + box9to10 + box11 + box12
  ⌊total_unused⌋ = 289 :=
by sorry

end madeline_unused_crayons_l1929_192973


namespace max_m_value_l1929_192935

def f (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem max_m_value : 
  (∃ (m : ℝ), m = 4 ∧ 
    (∃ (t : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 1 m → f (x + t) ≤ x)) ∧
  (∀ (m : ℝ), m > 4 → 
    (∀ (t : ℝ), ∃ (x : ℝ), x ∈ Set.Icc 1 m ∧ f (x + t) > x)) :=
by sorry

end max_m_value_l1929_192935


namespace total_carrots_l1929_192989

theorem total_carrots (sally_carrots fred_carrots : ℕ) 
  (h1 : sally_carrots = 6) 
  (h2 : fred_carrots = 4) : 
  sally_carrots + fred_carrots = 10 := by
  sorry

end total_carrots_l1929_192989


namespace rectangular_prism_sum_l1929_192997

/-- A rectangular prism is a three-dimensional shape with 6 faces, 12 edges, and 8 vertices. -/
structure RectangularPrism where
  faces : Nat
  edges : Nat
  vertices : Nat
  faces_eq : faces = 6
  edges_eq : edges = 12
  vertices_eq : vertices = 8

/-- The sum of faces, edges, and vertices of a rectangular prism is 26. -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  rp.faces + rp.edges + rp.vertices = 26 := by
  sorry

end rectangular_prism_sum_l1929_192997


namespace point_on_transformed_plane_l1929_192907

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (pt : Point3D) (p : Plane) : Prop :=
  p.a * pt.x + p.b * pt.y + p.c * pt.z + p.d = 0

/-- The main theorem to prove -/
theorem point_on_transformed_plane :
  let A : Point3D := { x := -2, y := -1, z := 1 }
  let a : Plane := { a := 1, b := -2, c := 6, d := -10 }
  let k : ℝ := 3/5
  pointOnPlane A (transformPlane a k) := by sorry

end point_on_transformed_plane_l1929_192907


namespace ralph_weekly_tv_hours_l1929_192940

/-- Represents Ralph's TV watching habits for a week -/
structure TVWatchingHabits where
  weekday_hours : ℕ
  weekend_hours : ℕ
  weekdays : ℕ
  weekend_days : ℕ

/-- Calculates the total hours of TV watched in a week -/
def total_weekly_hours (habits : TVWatchingHabits) : ℕ :=
  habits.weekday_hours * habits.weekdays + habits.weekend_hours * habits.weekend_days

/-- Theorem stating that Ralph watches 32 hours of TV in a week -/
theorem ralph_weekly_tv_hours :
  let habits : TVWatchingHabits := {
    weekday_hours := 4,
    weekend_hours := 6,
    weekdays := 5,
    weekend_days := 2
  }
  total_weekly_hours habits = 32 := by
  sorry

end ralph_weekly_tv_hours_l1929_192940


namespace rebecca_groups_l1929_192929

def egg_count : Nat := 75
def banana_count : Nat := 99
def marble_count : Nat := 48
def apple_count : Nat := 6 * 12  -- 6 dozen
def orange_count : Nat := 6  -- half dozen

def egg_group_size : Nat := 4
def banana_group_size : Nat := 5
def marble_group_size : Nat := 6
def apple_group_size : Nat := 12
def orange_group_size : Nat := 2

def total_groups : Nat :=
  (egg_count + egg_group_size - 1) / egg_group_size +
  (banana_count + banana_group_size - 1) / banana_group_size +
  marble_count / marble_group_size +
  apple_count / apple_group_size +
  orange_count / orange_group_size

theorem rebecca_groups : total_groups = 54 := by
  sorry

end rebecca_groups_l1929_192929


namespace laurens_mail_problem_l1929_192939

/-- Lauren's mail sending problem -/
theorem laurens_mail_problem (monday tuesday wednesday thursday : ℕ) :
  monday = 65 ∧
  tuesday > monday ∧
  wednesday = tuesday - 5 ∧
  thursday = wednesday + 15 ∧
  monday + tuesday + wednesday + thursday = 295 →
  tuesday - monday = 10 := by
sorry

end laurens_mail_problem_l1929_192939


namespace horner_method_V3_l1929_192995

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + x^3 - x^2 + 3*x - 5

def horner_V3 (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  let a₅ := 1
  let a₄ := 2
  let a₃ := 1
  let a₂ := -1
  let a₁ := 3
  let a₀ := -5
  let V₀ := a₅
  let V₁ := V₀ * x + a₄
  let V₂ := V₁ * x + a₃
  V₂ * x + a₂

theorem horner_method_V3 :
  horner_V3 f 5 = 179 := by sorry

end horner_method_V3_l1929_192995


namespace min_a_value_l1929_192963

-- Define the conditions
def p (x : ℝ) : Prop := |x + 1| ≤ 2
def q (x a : ℝ) : Prop := x ≤ a

-- Define what it means for p to be sufficient but not necessary for q
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ ¬(∀ x, q x a → p x)

-- State the theorem
theorem min_a_value :
  ∀ a : ℝ, sufficient_not_necessary a → a ≥ 1 :=
by sorry

end min_a_value_l1929_192963


namespace solve_equation_l1929_192965

theorem solve_equation (x t : ℝ) : 
  (3 * (x + 5)) / 4 = t + (3 - 3 * x) / 2 → x = 3 := by
  sorry

end solve_equation_l1929_192965


namespace integral_f_minus_one_to_pi_l1929_192988

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x - 1 else Real.cos x

theorem integral_f_minus_one_to_pi :
  ∫ x in (-1)..(Real.pi), f x = 1 := by sorry

end integral_f_minus_one_to_pi_l1929_192988


namespace vitya_wins_l1929_192956

/-- Represents a point on the infinite grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents the game state --/
structure GameState where
  marked_points : List GridPoint
  current_player : Bool  -- true for Kolya, false for Vitya

/-- Checks if a list of points forms a convex polygon --/
def is_convex_polygon (points : List GridPoint) : Prop :=
  sorry

/-- Checks if a move is valid according to the game rules --/
def is_valid_move (state : GameState) (new_point : GridPoint) : Prop :=
  is_convex_polygon (new_point :: state.marked_points)

/-- Represents a strategy for playing the game --/
def Strategy := GameState → Option GridPoint

/-- Checks if a strategy is winning for a player --/
def is_winning_strategy (strategy : Strategy) (player : Bool) : Prop :=
  sorry

theorem vitya_wins :
  ∃ (strategy : Strategy), is_winning_strategy strategy false :=
sorry

end vitya_wins_l1929_192956


namespace barbell_cost_l1929_192912

theorem barbell_cost (number_of_barbells : ℕ) (amount_paid : ℕ) (change_received : ℕ) :
  number_of_barbells = 3 ∧ amount_paid = 850 ∧ change_received = 40 →
  (amount_paid - change_received) / number_of_barbells = 270 :=
by
  sorry

end barbell_cost_l1929_192912


namespace f_increasing_iff_three_solutions_iff_l1929_192986

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * |x - a| + 2 * x

-- Theorem 1: f(x) is increasing on ℝ iff -2 ≤ a ≤ 2
theorem f_increasing_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ -2 ≤ a ∧ a ≤ 2 :=
sorry

-- Theorem 2: f(x) = bf(a) has three distinct real solutions iff b ∈ (1, 9/8)
theorem three_solutions_iff (a : ℝ) (h : -2 ≤ a ∧ a ≤ 4) :
  (∃ b : ℝ, 1 < b ∧ b < 9/8 ∧ ∃ x y z : ℝ, x < y ∧ y < z ∧
    f a x = b * f a a ∧ f a y = b * f a a ∧ f a z = b * f a a) ↔
  (2 < a ∧ a ≤ 4) :=
sorry

end

end f_increasing_iff_three_solutions_iff_l1929_192986


namespace mary_garden_apples_l1929_192976

/-- The number of pies Mary wants to bake -/
def num_pies : ℕ := 10

/-- The number of apples required for each pie -/
def apples_per_pie : ℕ := 8

/-- The number of additional apples Mary needs to buy -/
def apples_to_buy : ℕ := 30

/-- The number of apples Mary harvested from her garden -/
def apples_from_garden : ℕ := num_pies * apples_per_pie - apples_to_buy

theorem mary_garden_apples : apples_from_garden = 50 := by
  sorry

end mary_garden_apples_l1929_192976


namespace max_pyramid_volume_l1929_192966

/-- The maximum volume of a pyramid SABC with given conditions -/
theorem max_pyramid_volume (AB AC : ℝ) (sin_BAC : ℝ) (h : ℝ) :
  AB = 5 →
  AC = 8 →
  sin_BAC = 4/5 →
  h ≤ (5 * Real.sqrt 137 * Real.sqrt 3) / 8 →
  (1/3 : ℝ) * (1/2 * AB * AC * sin_BAC) * h ≤ 10 * Real.sqrt (137/3) :=
by sorry

end max_pyramid_volume_l1929_192966


namespace ana_salary_calculation_l1929_192901

/-- Calculates the final salary after a raise, pay cut, and bonus -/
def final_salary (initial_salary : ℝ) (raise_percent : ℝ) (cut_percent : ℝ) (bonus : ℝ) : ℝ :=
  initial_salary * (1 + raise_percent) * (1 - cut_percent) + bonus

theorem ana_salary_calculation :
  final_salary 2500 0.25 0.25 200 = 2543.75 := by
  sorry

end ana_salary_calculation_l1929_192901


namespace odd_function_value_at_negative_one_l1929_192960

-- Define the function f
noncomputable def f (c : ℝ) : ℝ → ℝ := fun x =>
  if x ≥ 0 then 3^x - 2*x + c else -(3^(-x) - 2*(-x) + c)

-- State the theorem
theorem odd_function_value_at_negative_one (c : ℝ) :
  (∀ x, f c x = -(f c (-x))) → f c (-1) = 0 := by
  sorry

end odd_function_value_at_negative_one_l1929_192960


namespace chess_tournament_games_l1929_192981

theorem chess_tournament_games (num_players : ℕ) (total_games : ℕ) (games_per_pair : ℕ) : 
  num_players = 8 →
  total_games = 56 →
  total_games = (num_players * (num_players - 1) * games_per_pair) / 2 →
  games_per_pair = 2 := by
sorry

end chess_tournament_games_l1929_192981


namespace tangerine_tree_count_prove_tangerine_tree_count_l1929_192946

theorem tangerine_tree_count : ℕ → ℕ → ℕ → Prop :=
  fun pear_trees apple_trees tangerine_trees =>
    (pear_trees = 56) →
    (pear_trees = apple_trees + 18) →
    (tangerine_trees = apple_trees - 12) →
    (tangerine_trees = 26)

-- Proof
theorem prove_tangerine_tree_count :
  ∃ (pear_trees apple_trees tangerine_trees : ℕ),
    tangerine_tree_count pear_trees apple_trees tangerine_trees :=
by
  sorry

end tangerine_tree_count_prove_tangerine_tree_count_l1929_192946


namespace workshop_average_salary_l1929_192971

theorem workshop_average_salary 
  (total_workers : ℕ) 
  (num_technicians : ℕ) 
  (avg_salary_technicians : ℕ) 
  (avg_salary_others : ℕ) 
  (h1 : total_workers = 28) 
  (h2 : num_technicians = 7) 
  (h3 : avg_salary_technicians = 14000) 
  (h4 : avg_salary_others = 6000) :
  (num_technicians * avg_salary_technicians + 
   (total_workers - num_technicians) * avg_salary_others) / total_workers = 8000 :=
by sorry

end workshop_average_salary_l1929_192971


namespace simplify_trig_ratio_l1929_192952

theorem simplify_trig_ratio : 
  (Real.sin (30 * π / 180) + Real.sin (40 * π / 180)) / 
  (Real.cos (30 * π / 180) + Real.cos (40 * π / 180)) = 
  Real.tan (35 * π / 180) := by sorry

end simplify_trig_ratio_l1929_192952


namespace fourth_student_guess_l1929_192962

def jellybean_guess (first_guess : ℕ) : ℕ :=
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let average := (first_guess + second_guess + third_guess) / 3
  average + 25

theorem fourth_student_guess :
  jellybean_guess 100 = 525 := by
  sorry

end fourth_student_guess_l1929_192962


namespace line_slope_intercept_sum_l1929_192955

/-- Given a line with slope 5 passing through the point (-2, 4), 
    prove that the sum of its slope and y-intercept is 19. -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℝ), 
    m = 5 →                  -- The slope is 5
    4 = m * (-2) + b →       -- The line passes through (-2, 4)
    m + b = 19 :=            -- The sum of slope and y-intercept is 19
by
  sorry


end line_slope_intercept_sum_l1929_192955


namespace probability_at_least_one_B_l1929_192954

/-- The probability of selecting at least one question of type B when randomly choosing 2 questions out of 5, where 2 are of type A and 3 are of type B -/
theorem probability_at_least_one_B (total : Nat) (type_A : Nat) (type_B : Nat) (select : Nat) : 
  total = 5 → type_A = 2 → type_B = 3 → select = 2 →
  (Nat.choose total select - Nat.choose type_A select) / Nat.choose total select = 9 / 10 := by
sorry


end probability_at_least_one_B_l1929_192954


namespace max_gross_profit_l1929_192961

/-- The gross profit function L(p) for a store selling goods --/
def L (p : ℝ) : ℝ := (8300 - 170*p - p^2)*(p - 20)

/-- The statement that L(p) achieves its maximum at p = 30 with a value of 23000 --/
theorem max_gross_profit :
  ∃ (p : ℝ), p > 0 ∧ L p = 23000 ∧ ∀ (q : ℝ), q > 0 → L q ≤ L p :=
sorry

end max_gross_profit_l1929_192961


namespace sam_pennies_washing_l1929_192922

/-- The number of pennies Sam got for washing clothes -/
def pennies_from_washing (total_cents : ℕ) (num_quarters : ℕ) : ℕ :=
  total_cents - (num_quarters * 25)

/-- Theorem stating that Sam got 9 pennies for washing clothes -/
theorem sam_pennies_washing : 
  pennies_from_washing 184 7 = 9 := by sorry

end sam_pennies_washing_l1929_192922


namespace shyne_plants_l1929_192906

/-- The number of eggplants that can be grown from one seed packet -/
def eggplants_per_packet : ℕ := 14

/-- The number of sunflowers that can be grown from one seed packet -/
def sunflowers_per_packet : ℕ := 10

/-- The number of eggplant seed packets Shyne bought -/
def eggplant_packets : ℕ := 4

/-- The number of sunflower seed packets Shyne bought -/
def sunflower_packets : ℕ := 6

/-- The total number of plants Shyne can grow -/
def total_plants : ℕ := eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets

theorem shyne_plants : total_plants = 116 := by
  sorry

end shyne_plants_l1929_192906


namespace regular_polygon_perimeter_l1929_192996

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (s : ℝ) (e : ℝ) : 
  s = 7 → e = 90 → (360 / e : ℝ) * s = 28 := by
  sorry

end regular_polygon_perimeter_l1929_192996


namespace xyz_value_l1929_192918

theorem xyz_value (a b c x y z : ℂ)
  (eq1 : a = (b + c) / (x - 2))
  (eq2 : b = (c + a) / (y - 2))
  (eq3 : c = (a + b) / (z - 2))
  (sum_prod : x * y + y * z + z * x = 67)
  (sum : x + y + z = 2010) :
  x * y * z = -5892 := by
sorry

end xyz_value_l1929_192918


namespace estimate_fish_population_l1929_192964

/-- Estimates the number of fish in a pond using the capture-recapture method. -/
theorem estimate_fish_population (initial_marked : ℕ) (second_sample : ℕ) (marked_in_second : ℕ) :
  initial_marked = 200 →
  second_sample = 100 →
  marked_in_second = 20 →
  (initial_marked * second_sample) / marked_in_second = 1000 :=
by
  sorry

#check estimate_fish_population

end estimate_fish_population_l1929_192964


namespace math_team_combinations_l1929_192937

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of girls in the math club -/
def num_girls : ℕ := 4

/-- The number of boys in the math club -/
def num_boys : ℕ := 6

/-- The number of girls to be chosen for the team -/
def girls_in_team : ℕ := 3

/-- The number of boys to be chosen for the team -/
def boys_in_team : ℕ := 2

theorem math_team_combinations :
  (choose num_girls girls_in_team) * (choose num_boys boys_in_team) = 60 := by
  sorry

end math_team_combinations_l1929_192937


namespace difference_of_squares_535_465_l1929_192904

theorem difference_of_squares_535_465 : 535^2 - 465^2 = 70000 := by
  sorry

end difference_of_squares_535_465_l1929_192904


namespace prob_three_heads_in_eight_tosses_l1929_192928

/-- A fair coin is tossed 8 times. -/
def num_tosses : ℕ := 8

/-- The number of heads we're looking for. -/
def target_heads : ℕ := 3

/-- The probability of getting heads on a single toss of a fair coin. -/
def prob_heads : ℚ := 1/2

/-- The probability of getting exactly 'target_heads' heads in 'num_tosses' tosses of a fair coin. -/
def probability_exact_heads : ℚ :=
  (Nat.choose num_tosses target_heads : ℚ) * prob_heads^target_heads * (1 - prob_heads)^(num_tosses - target_heads)

/-- Theorem stating that the probability of getting exactly 3 heads in 8 tosses of a fair coin is 7/32. -/
theorem prob_three_heads_in_eight_tosses : probability_exact_heads = 7/32 := by
  sorry

end prob_three_heads_in_eight_tosses_l1929_192928


namespace intersection_A_B_union_A_B_range_of_a_l1929_192968

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 < x ∧ x ≤ 8}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 8} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | 1 < x ∧ x < 9} := by sorry

-- Theorem for the range of a when A ∩ C is non-empty
theorem range_of_a (a : ℝ) : (A ∩ C a).Nonempty → a ≤ 8 := by sorry

end intersection_A_B_union_A_B_range_of_a_l1929_192968


namespace contractor_absent_days_l1929_192942

/-- Proves the number of absent days for a contractor under specific conditions -/
theorem contractor_absent_days 
  (total_days : ℕ) 
  (daily_pay : ℚ) 
  (daily_fine : ℚ) 
  (total_received : ℚ) : 
  total_days = 30 → 
  daily_pay = 25 → 
  daily_fine = 7.5 → 
  total_received = 620 → 
  ∃ (days_worked : ℕ) (days_absent : ℕ), 
    days_worked + days_absent = total_days ∧ 
    (daily_pay * days_worked : ℚ) - (daily_fine * days_absent : ℚ) = total_received ∧ 
    days_absent = 8 := by
  sorry

end contractor_absent_days_l1929_192942


namespace frog_jump_probability_l1929_192983

-- Define the square
def Square := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6}

-- Define a valid jump
def ValidJump (p q : ℝ × ℝ) : Prop :=
  (p.1 = q.1 ∧ |p.2 - q.2| = 1) ∨ (p.2 = q.2 ∧ |p.1 - q.1| = 1)

-- Define the boundary of the square
def Boundary (p : ℝ × ℝ) : Prop :=
  p.1 = 0 ∨ p.1 = 6 ∨ p.2 = 0 ∨ p.2 = 6

-- Define vertical sides
def VerticalSide (p : ℝ × ℝ) : Prop :=
  (p.1 = 0 ∨ p.1 = 6) ∧ 0 ≤ p.2 ∧ p.2 ≤ 6

-- Define the probability function
noncomputable def P (p : ℝ × ℝ) : ℝ :=
  sorry -- The actual implementation would go here

-- State the theorem
theorem frog_jump_probability :
  P (2, 3) = 3/5 :=
sorry

end frog_jump_probability_l1929_192983


namespace triangle_properties_l1929_192978

-- Define the triangle ABC
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 4)

-- Define the equations of median and altitude
def median_eq (x y : ℝ) : Prop := 2 * x + y - 7 = 0
def altitude_eq (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define vertex C
def C : ℝ × ℝ := (3, 1)

-- Define the area of the triangle
def triangle_area : ℝ := 3

-- Theorem statement
theorem triangle_properties :
  median_eq (C.1) (C.2) ∧ 
  altitude_eq (C.1) (C.2) →
  C = (3, 1) ∧ 
  triangle_area = 3 := by sorry

end triangle_properties_l1929_192978


namespace all_positive_integers_are_valid_l1929_192915

-- Define a coloring of the infinite grid
def Coloring := ℤ → ℤ → Bool

-- Define a rectangle on the grid
structure Rectangle where
  x : ℤ
  y : ℤ
  width : ℕ+
  height : ℕ+

-- Count the number of red cells in a rectangle
def countRedCells (c : Coloring) (r : Rectangle) : ℕ :=
  sorry

-- Define the property that all n-cell rectangles have an odd number of red cells
def validColoring (n : ℕ+) (c : Coloring) : Prop :=
  ∀ r : Rectangle, r.width * r.height = n → Odd (countRedCells c r)

-- The main theorem
theorem all_positive_integers_are_valid :
  ∀ n : ℕ+, ∃ c : Coloring, validColoring n c :=
sorry

end all_positive_integers_are_valid_l1929_192915


namespace triathlete_average_speed_l1929_192990

/-- Proves that the average speed of a triathlete is 0.125 miles per minute
    given specific conditions for running and swimming. -/
theorem triathlete_average_speed
  (run_distance : ℝ)
  (swim_distance : ℝ)
  (run_speed : ℝ)
  (swim_speed : ℝ)
  (h1 : run_distance = 3)
  (h2 : swim_distance = 3)
  (h3 : run_speed = 10)
  (h4 : swim_speed = 6) :
  (run_distance + swim_distance) / ((run_distance / run_speed + swim_distance / swim_speed) * 60) = 0.125 := by
  sorry

#check triathlete_average_speed

end triathlete_average_speed_l1929_192990


namespace initial_bananas_count_l1929_192975

/-- The number of bananas in each package -/
def package_size : ℕ := 13

/-- The number of bananas added to the pile -/
def bananas_added : ℕ := 7

/-- The total number of bananas after adding -/
def total_bananas : ℕ := 9

/-- The initial number of bananas on the desk -/
def initial_bananas : ℕ := total_bananas - bananas_added

theorem initial_bananas_count : initial_bananas = 2 := by
  sorry

end initial_bananas_count_l1929_192975


namespace five_cubic_yards_to_cubic_feet_l1929_192914

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℝ := 3

/-- The number of cubic yards we want to convert -/
def cubic_yards : ℝ := 5

/-- The theorem states that 5 cubic yards is equal to 135 cubic feet -/
theorem five_cubic_yards_to_cubic_feet : 
  cubic_yards * (yards_to_feet ^ 3) = 135 := by
  sorry

end five_cubic_yards_to_cubic_feet_l1929_192914


namespace rectangle_max_area_l1929_192972

theorem rectangle_max_area (x y P D : ℝ) (h1 : P = 2*x + 2*y) (h2 : D^2 = x^2 + y^2) 
  (h3 : P = 14) (h4 : D = 5) :
  ∃ (A : ℝ), A = x * y ∧ A ≤ 49/4 ∧ ∃ (x₀ y₀ : ℝ), x₀ * y₀ = 49/4 := by
  sorry

#check rectangle_max_area

end rectangle_max_area_l1929_192972


namespace irene_worked_50_hours_l1929_192908

/-- Calculates the total hours worked given the regular hours, overtime hours, regular pay, overtime pay rate, and total income. -/
def total_hours_worked (regular_hours : ℕ) (regular_pay : ℕ) (overtime_rate : ℕ) (total_income : ℕ) : ℕ :=
  regular_hours + (total_income - regular_pay) / overtime_rate

/-- Proves that given the problem conditions, Irene worked 50 hours. -/
theorem irene_worked_50_hours (regular_hours : ℕ) (regular_pay : ℕ) (overtime_rate : ℕ) (total_income : ℕ)
  (h1 : regular_hours = 40)
  (h2 : regular_pay = 500)
  (h3 : overtime_rate = 20)
  (h4 : total_income = 700) :
  total_hours_worked regular_hours regular_pay overtime_rate total_income = 50 := by
  sorry

#eval total_hours_worked 40 500 20 700

end irene_worked_50_hours_l1929_192908


namespace longest_leg_of_smallest_triangle_l1929_192974

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse_def : hypotenuse = 2 * shorter_leg
  longer_leg_def : longer_leg = shorter_leg * Real.sqrt 3

/-- Represents a sequence of three 30-60-90 triangles -/
structure TriangleSequence where
  largest : Triangle30_60_90
  middle : Triangle30_60_90
  smallest : Triangle30_60_90
  sequence_property : 
    largest.longer_leg = middle.hypotenuse ∧
    middle.longer_leg = smallest.hypotenuse

theorem longest_leg_of_smallest_triangle 
  (seq : TriangleSequence) 
  (h : seq.largest.hypotenuse = 16) : 
  seq.smallest.longer_leg = 6 * Real.sqrt 3 := by
  sorry

end longest_leg_of_smallest_triangle_l1929_192974


namespace three_numbers_sum_l1929_192920

theorem three_numbers_sum (a b c m : ℕ) : 
  a + b + c = 2015 →
  a + b = m + 1 →
  b + c = m + 2011 →
  c + a = m + 2012 →
  m = 2 := by
sorry

end three_numbers_sum_l1929_192920


namespace min_value_sum_squared_ratios_l1929_192936

theorem min_value_sum_squared_ratios (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 9) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 := by
  sorry

end min_value_sum_squared_ratios_l1929_192936


namespace floor_sum_equals_negative_one_l1929_192925

theorem floor_sum_equals_negative_one : ⌊(18.7 : ℝ)⌋ + ⌊(-18.7 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_equals_negative_one_l1929_192925


namespace hair_length_after_growth_and_cut_l1929_192987

theorem hair_length_after_growth_and_cut (x : ℝ) : 
  let initial_length : ℝ := 14
  let growth : ℝ := x
  let cut_length : ℝ := 20
  let final_length : ℝ := initial_length + growth - cut_length
  final_length = x - 6 := by sorry

end hair_length_after_growth_and_cut_l1929_192987


namespace binomial_expansion_sum_l1929_192905

theorem binomial_expansion_sum (m : ℝ) : 
  (∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ, 
    (∀ x : ℝ, (1 + m * x)^6 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6) ∧
    (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 64)) → 
  m = 1 ∨ m = -3 := by
sorry

end binomial_expansion_sum_l1929_192905


namespace choose_four_from_thirteen_l1929_192948

theorem choose_four_from_thirteen : Nat.choose 13 4 = 715 := by sorry

end choose_four_from_thirteen_l1929_192948


namespace remainder_4059_div_32_l1929_192957

theorem remainder_4059_div_32 : 4059 % 32 = 27 := by
  sorry

end remainder_4059_div_32_l1929_192957


namespace class_fraction_problem_l1929_192927

theorem class_fraction_problem (G : ℕ) (B : ℕ) :
  B = (5 * G) / 3 →
  (2 * G) / 3 = (1 / 4) * (B + G) :=
by sorry

end class_fraction_problem_l1929_192927


namespace power_of_two_minus_one_as_power_l1929_192921

theorem power_of_two_minus_one_as_power (n : ℕ) : 
  (∃ (a k : ℕ), k ≥ 2 ∧ 2^n - 1 = a^k) ↔ n = 0 ∨ n = 1 := by
  sorry

end power_of_two_minus_one_as_power_l1929_192921


namespace equation_solutions_l1929_192917

theorem equation_solutions :
  (∀ x : ℝ, (x - 2) * (x - 3) = x - 2 ↔ x = 2 ∨ x = 4) ∧
  (∀ x : ℝ, 2 * x^2 - 5 * x + 1 = 0 ↔ x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4) :=
by sorry

end equation_solutions_l1929_192917


namespace arithmetic_sequence_divisibility_l1929_192967

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_divisibility
  (a : ℕ → ℕ)
  (h_arith : is_arithmetic_sequence a)
  (h_div : ∀ n : ℕ, 2005 ∣ a n * a (n + 31)) :
  ∀ n : ℕ, 2005 ∣ a n :=
sorry

end arithmetic_sequence_divisibility_l1929_192967


namespace ice_cream_volume_l1929_192931

/-- The volume of ice cream in a cone with a cylinder on top -/
theorem ice_cream_volume (cone_height : ℝ) (cone_radius : ℝ) (cylinder_height : ℝ) : 
  cone_height = 12 → 
  cone_radius = 3 → 
  cylinder_height = 2 → 
  (1/3 * π * cone_radius^2 * cone_height) + (π * cone_radius^2 * cylinder_height) = 54 * π := by
  sorry


end ice_cream_volume_l1929_192931


namespace base9_725_to_base3_l1929_192980

/-- Converts a base-9 digit to its two-digit base-3 representation -/
def base9_to_base3_digit (d : ℕ) : ℕ × ℕ :=
  (d / 3, d % 3)

/-- Converts a base-9 number to its base-3 representation -/
def base9_to_base3 (n : ℕ) : List ℕ :=
  let digits := n.digits 9
  List.join (digits.map (fun d => let (q, r) := base9_to_base3_digit d; [q, r]))

theorem base9_725_to_base3 :
  base9_to_base3 725 = [2, 1, 0, 2, 1, 2] := by
  sorry

end base9_725_to_base3_l1929_192980


namespace harkamal_payment_l1929_192950

/-- The amount Harkamal paid to the shopkeeper -/
def total_amount_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Proof that Harkamal paid 1145 to the shopkeeper -/
theorem harkamal_payment : total_amount_paid 8 70 9 65 = 1145 := by
  sorry

end harkamal_payment_l1929_192950


namespace problem_1_l1929_192991

theorem problem_1 (x y : ℝ) (h : x^2 + y^2 = 1) :
  x^6 + 3*x^2*y^2 + y^6 = 1 := by
  sorry

end problem_1_l1929_192991


namespace find_a_and_m_l1929_192932

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + (a-1) = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

-- State the theorem
theorem find_a_and_m :
  ∃ (a m : ℝ),
    (A ∪ B a = A) ∧
    (A ∩ B a = C m) ∧
    (a = 3) ∧
    (m = 3) := by
  sorry

end find_a_and_m_l1929_192932


namespace congruence_solution_l1929_192923

theorem congruence_solution (n : ℤ) : 
  (13 * n) % 47 = 8 % 47 → n % 47 = 29 % 47 := by
sorry

end congruence_solution_l1929_192923


namespace equal_points_per_round_l1929_192909

-- Define the total points and number of rounds
def total_points : ℕ := 300
def num_rounds : ℕ := 5

-- Define the points per round
def points_per_round : ℕ := total_points / num_rounds

-- Theorem to prove
theorem equal_points_per_round :
  (total_points = num_rounds * points_per_round) ∧ (points_per_round = 60) := by
  sorry

end equal_points_per_round_l1929_192909


namespace m_value_l1929_192934

/-- The function f(x) = 4x^2 + 5x + 6 -/
def f (x : ℝ) : ℝ := 4 * x^2 + 5 * x + 6

/-- The function g(x) = 2x^3 - mx + 8, parameterized by m -/
def g (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - m * x + 8

/-- Theorem stating that if f(5) - g(5) = 15, then m = 28.4 -/
theorem m_value (m : ℝ) : f 5 - g m 5 = 15 → m = 28.4 := by
  sorry

end m_value_l1929_192934


namespace time_to_fill_cistern_l1929_192903

/-- Given a cistern that can be partially filled by a pipe, this theorem proves
    the time required to fill the entire cistern. -/
theorem time_to_fill_cistern (partial_fill_time : ℝ) (partial_fill_fraction : ℝ) 
  (h1 : partial_fill_time = 7)
  (h2 : partial_fill_fraction = 1 / 11) : 
  partial_fill_time / partial_fill_fraction = 77 := by
  sorry

end time_to_fill_cistern_l1929_192903


namespace geometry_theorem_l1929_192900

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations
def contains (p : Plane) (l : Line) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def intersection (p1 p2 : Plane) (l : Line) : Prop := sorry

-- State the theorem
theorem geometry_theorem 
  (m n l : Line) 
  (α β γ : Plane) 
  (hm : m ≠ n ∧ m ≠ l ∧ n ≠ l) 
  (hα : α ≠ β ∧ α ≠ γ ∧ β ≠ γ) :
  (contains α m ∧ contains β n ∧ perpendicular_planes α β ∧ 
   intersection α β l ∧ perpendicular_lines m l → perpendicular_lines m n) ∧
  (parallel_planes α γ ∧ parallel_planes β γ → parallel_planes α β) := by
  sorry

end geometry_theorem_l1929_192900
