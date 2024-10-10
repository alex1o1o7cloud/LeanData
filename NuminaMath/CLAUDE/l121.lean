import Mathlib

namespace sumata_vacation_l121_12183

/-- The Sumata family vacation problem -/
theorem sumata_vacation (miles_per_day : ℕ) (total_miles : ℕ) (h1 : miles_per_day = 250) (h2 : total_miles = 1250) :
  total_miles / miles_per_day = 5 := by
  sorry

end sumata_vacation_l121_12183


namespace min_value_of_two_plus_y_l121_12118

theorem min_value_of_two_plus_y (x y : ℝ) (h1 : y > 0) (h2 : x^2 + y - 3 = 0) :
  ∀ z, z = 2 + y → z ≥ 2 :=
sorry

end min_value_of_two_plus_y_l121_12118


namespace circle_radius_with_perpendicular_chords_l121_12157

/-- Given a circle with two perpendicular chords intersecting at the center,
    if two parallel sides of the formed quadrilateral have length 2,
    then the radius of the circle is √2. -/
theorem circle_radius_with_perpendicular_chords 
  (O : ℝ × ℝ) -- Center of the circle
  (K L M N : ℝ × ℝ) -- Points on the circle
  (h1 : (K.1 - M.1) * (L.2 - N.2) = 0) -- KM ⊥ LN
  (h2 : (K.2 - L.2) = (M.2 - N.2)) -- KL ∥ MN
  (h3 : Real.sqrt ((K.1 - L.1)^2 + (K.2 - L.2)^2) = 2) -- KL = 2
  (h4 : Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 2) -- MN = 2
  (h5 : O = (0, 0)) -- Center at origin
  (h6 : K.2 = 0 ∧ M.2 = 0) -- K and M on x-axis
  (h7 : L.1 = 0 ∧ N.1 = 0) -- L and N on y-axis
  : Real.sqrt (K.1^2 + N.2^2) = Real.sqrt 2 := by
  sorry


end circle_radius_with_perpendicular_chords_l121_12157


namespace boys_from_clay_l121_12191

/-- Represents the number of students from each school and gender --/
structure StudentCounts where
  total : Nat
  boys : Nat
  girls : Nat
  jonas : Nat
  clay : Nat
  pine : Nat
  jonasGirls : Nat
  pineBoys : Nat

/-- Theorem stating that the number of boys from Clay Middle School is 40 --/
theorem boys_from_clay (s : StudentCounts)
  (h_total : s.total = 120)
  (h_boys : s.boys = 70)
  (h_girls : s.girls = 50)
  (h_jonas : s.jonas = 50)
  (h_clay : s.clay = 40)
  (h_pine : s.pine = 30)
  (h_jonasGirls : s.jonasGirls = 30)
  (h_pineBoys : s.pineBoys = 10)
  : s.clay - (s.girls - s.jonasGirls - (s.pine - s.pineBoys)) = 40 := by
  sorry

end boys_from_clay_l121_12191


namespace ellipse_line_intersection_collinearity_l121_12149

-- Define the ellipse Γ
def Γ (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 12

-- Define the line l
def l (m q y : ℝ) : ℝ := m * y + q

-- Define the right focus of the ellipse
def F : ℝ × ℝ := (1, 0)

-- Define the condition for A₁, F, and B to be collinear
def collinear (A₁ F B : ℝ × ℝ) : Prop :=
  (F.2 - A₁.2) * (B.1 - F.1) = (B.2 - F.2) * (F.1 - A₁.1)

-- Main theorem
theorem ellipse_line_intersection_collinearity 
  (m q : ℝ) 
  (hm : m ≠ 0) 
  (A B : ℝ × ℝ) 
  (hA : Γ A.1 A.2 ∧ A.1 = l m q A.2)
  (hB : Γ B.1 B.2 ∧ B.1 = l m q B.2)
  (hAB : A ≠ B)
  (A₁ : ℝ × ℝ)
  (hA₁ : A₁ = (A.1, -A.2)) :
  (collinear A₁ F B ↔ q = 4) :=
sorry

end ellipse_line_intersection_collinearity_l121_12149


namespace abs_eq_sqrt_square_abs_sqrt_square_domain_l121_12122

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

theorem abs_sqrt_square_domain : Set.univ = {x : ℝ | ∃ y, y = Real.sqrt (x^2)} := by sorry

end abs_eq_sqrt_square_abs_sqrt_square_domain_l121_12122


namespace one_correct_description_l121_12164

/-- Represents an experimental description --/
structure ExperimentalDescription where
  id : Nat
  isCorrect : Bool

/-- The set of all experimental descriptions --/
def experimentDescriptions : Finset ExperimentalDescription := sorry

/-- Theorem stating that there is exactly one correct experimental description --/
theorem one_correct_description :
  (experimentDescriptions.filter (λ d => d.isCorrect)).card = 1 := by sorry

end one_correct_description_l121_12164


namespace movie_replay_count_l121_12163

theorem movie_replay_count (movie_length : Real) (ad_length : Real) (theater_hours : Real) :
  movie_length = 1.5 ∧ ad_length = 1/3 ∧ theater_hours = 11 →
  ⌊theater_hours * 60 / (movie_length * 60 + ad_length)⌋ = 6 := by
sorry

end movie_replay_count_l121_12163


namespace meet_once_l121_12151

/-- Represents the movement of Michael and the garbage truck --/
structure Movement where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between Michael and the garbage truck --/
def number_of_meetings (m : Movement) : ℕ :=
  sorry

/-- The theorem stating that Michael and the garbage truck meet exactly once --/
theorem meet_once (m : Movement) 
  (h1 : m.michael_speed = 3)
  (h2 : m.truck_speed = 6)
  (h3 : m.pail_distance = 100)
  (h4 : m.truck_stop_time = 20)
  (h5 : m.initial_distance = 100) : 
  number_of_meetings m = 1 :=
sorry

end meet_once_l121_12151


namespace bridge_problem_l121_12173

/-- A graph representing the bridge system. -/
structure BridgeGraph where
  /-- The set of nodes (islands) in the graph. -/
  nodes : Finset (Fin 4)
  /-- The set of edges (bridges) in the graph. -/
  edges : Finset (Fin 4 × Fin 4)
  /-- The degree of each node. -/
  degree : Fin 4 → Nat
  /-- Condition that node 0 (A) has degree 3. -/
  degree_A : degree 0 = 3
  /-- Condition that node 1 (B) has degree 5. -/
  degree_B : degree 1 = 5
  /-- Condition that node 2 (C) has degree 3. -/
  degree_C : degree 2 = 3
  /-- Condition that node 3 (D) has degree 3. -/
  degree_D : degree 3 = 3
  /-- The total number of edges is 9. -/
  edge_count : edges.card = 9

/-- The number of Eulerian paths in the bridge graph. -/
def countEulerianPaths (g : BridgeGraph) : Nat :=
  sorry

/-- Theorem stating that the number of Eulerian paths is 132. -/
theorem bridge_problem (g : BridgeGraph) : countEulerianPaths g = 132 :=
  sorry

end bridge_problem_l121_12173


namespace amc10_min_correct_problems_l121_12194

/-- The AMC 10 scoring system and Sarah's strategy -/
structure AMC10 where
  total_problems : Nat
  attempted_problems : Nat
  correct_points : Nat
  unanswered_points : Nat
  target_score : Nat

/-- The minimum number of correctly solved problems to reach the target score -/
def min_correct_problems (amc : AMC10) : Nat :=
  let unanswered := amc.total_problems - amc.attempted_problems
  let unanswered_score := unanswered * amc.unanswered_points
  let required_score := amc.target_score - unanswered_score
  (required_score + amc.correct_points - 1) / amc.correct_points

/-- Theorem stating that for the given AMC 10 configuration, 
    the minimum number of correctly solved problems is 20 -/
theorem amc10_min_correct_problems :
  let amc : AMC10 := {
    total_problems := 30,
    attempted_problems := 25,
    correct_points := 7,
    unanswered_points := 2,
    target_score := 150
  }
  min_correct_problems amc = 20 := by
  sorry

end amc10_min_correct_problems_l121_12194


namespace function_value_sum_l121_12195

def nondecreasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_value_sum (f : ℝ → ℝ) :
  nondecreasing f 0 1 →
  f 0 = 0 →
  (∀ x, f (x / 3) = (1 / 2) * f x) →
  (∀ x, f (1 - x) = 1 - f x) →
  f 1 + f (1 / 2) + f (1 / 3) + f (1 / 6) + f (1 / 7) + f (1 / 8) = 11 / 4 := by
sorry

end function_value_sum_l121_12195


namespace principal_amount_proof_l121_12107

/-- Proves that given specific interest conditions, the principal amount is 6400 --/
theorem principal_amount_proof (rate : ℝ) (time : ℝ) (difference : ℝ) : rate = 0.05 → time = 2 → difference = 16 → 
  ∃ (principal : ℝ), principal * ((1 + rate)^time - 1 - rate * time) = difference ∧ principal = 6400 := by
  sorry

end principal_amount_proof_l121_12107


namespace soccer_team_lineup_combinations_l121_12133

-- Define the total number of players
def total_players : ℕ := 16

-- Define the number of quadruplets
def num_quadruplets : ℕ := 4

-- Define the number of starters to choose
def num_starters : ℕ := 7

-- Define the number of quadruplets that must be in the starting lineup
def quadruplets_in_lineup : ℕ := 3

-- Theorem statement
theorem soccer_team_lineup_combinations :
  (Nat.choose num_quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - num_quadruplets) (num_starters - quadruplets_in_lineup)) = 1980 :=
by sorry

end soccer_team_lineup_combinations_l121_12133


namespace log_meaningful_implies_t_range_p_sufficient_for_q_implies_a_range_l121_12187

-- Define the propositions
def p (a t : ℝ) : Prop := -2 * t^2 + 7 * t - 5 > 0
def q (a t : ℝ) : Prop := t^2 - (a + 3) * t + (a + 2) < 0

theorem log_meaningful_implies_t_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ t : ℝ, p a t → 1 < t ∧ t < 5/2 :=
sorry

theorem p_sufficient_for_q_implies_a_range :
  ∀ a : ℝ, (∀ t : ℝ, 1 < t ∧ t < 5/2 → q a t) → a ≥ 1/2 :=
sorry

end log_meaningful_implies_t_range_p_sufficient_for_q_implies_a_range_l121_12187


namespace division_relation_l121_12138

theorem division_relation (D : ℝ) (h : D > 0) :
  let d := D / 35
  let q := D / 5
  q = D / 5 ∧ q = 7 * d := by sorry

end division_relation_l121_12138


namespace triangle_inequality_condition_l121_12168

theorem triangle_inequality_condition (k l : ℝ) :
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b →
    k * a^2 + l * b^2 > c^2) ↔
  k * l ≥ k + l ∧ k > 1 ∧ l > 1 := by
  sorry

end triangle_inequality_condition_l121_12168


namespace complex_condition_implies_a_value_l121_12152

theorem complex_condition_implies_a_value (a : ℝ) :
  (((a : ℂ) + Complex.I) * (2 * Complex.I)).re > 0 → a = -1 := by
  sorry

end complex_condition_implies_a_value_l121_12152


namespace robin_cupcakes_l121_12114

/-- The number of cupcakes with chocolate sauce Robin ate -/
def chocolate_cupcakes : ℕ := sorry

/-- The number of cupcakes with buttercream frosting Robin ate -/
def buttercream_cupcakes : ℕ := sorry

/-- The total number of cupcakes Robin ate -/
def total_cupcakes : ℕ := 12

theorem robin_cupcakes :
  chocolate_cupcakes + buttercream_cupcakes = total_cupcakes ∧
  buttercream_cupcakes = 2 * chocolate_cupcakes →
  chocolate_cupcakes = 4 :=
by sorry

end robin_cupcakes_l121_12114


namespace alligator_journey_time_l121_12134

/-- The combined time of Paul's journey to the Nile Delta and back -/
def combined_journey_time (initial_time : ℕ) (additional_return_time : ℕ) : ℕ :=
  initial_time + (initial_time + additional_return_time)

/-- Theorem stating that the combined journey time is 10 hours -/
theorem alligator_journey_time : combined_journey_time 4 2 = 10 := by
  sorry

end alligator_journey_time_l121_12134


namespace arrangement_count_is_120_l121_12145

/-- The number of ways to arrange 4 distinct objects and 1 empty space in 5 positions -/
def arrangement_count : ℕ := 5 * 4 * 3 * 2 * 1

/-- Theorem: The number of ways to arrange 4 distinct objects and 1 empty space in 5 positions is 120 -/
theorem arrangement_count_is_120 : arrangement_count = 120 := by
  sorry

end arrangement_count_is_120_l121_12145


namespace duty_roster_theorem_l121_12192

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of arrangements where two specific people are adjacent -/
def adjacent_arrangements (n : ℕ) : ℕ := 2 * permutations (n - 1)

/-- The number of arrangements where both pairs of specific people are adjacent -/
def both_adjacent_arrangements (n : ℕ) : ℕ := 2 * 2 * permutations (n - 2)

/-- The number of valid arrangements for the duty roster problem -/
def duty_roster_arrangements (n : ℕ) : ℕ :=
  permutations n - 2 * adjacent_arrangements n + both_adjacent_arrangements n

theorem duty_roster_theorem :
  duty_roster_arrangements 6 = 336 := by sorry

end duty_roster_theorem_l121_12192


namespace cookie_distribution_theorem_l121_12125

/-- Represents the distribution of cookies in boxes -/
def CookieDistribution := List Nat

/-- Represents the process of taking cookies from boxes and placing them on plates -/
def distributeCookies (boxes : CookieDistribution) : List Nat :=
  let maxCookies := boxes.foldl max 0
  List.range maxCookies |>.map (fun i => boxes.filter (· > i) |>.length)

theorem cookie_distribution_theorem (boxes : CookieDistribution) :
  (boxes.toFinset |>.card) = ((distributeCookies boxes).toFinset |>.card) := by
  sorry

end cookie_distribution_theorem_l121_12125


namespace parallel_planes_sufficient_not_necessary_l121_12139

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation between planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_plane_parallel : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_sufficient_not_necessary
  (α β : Plane) (a : Line)
  (h_a_in_α : line_in_plane a α) :
  (∀ α β a, plane_parallel α β → line_plane_parallel a β) ∧
  (∃ α β a, line_plane_parallel a β ∧ ¬ plane_parallel α β) :=
sorry

end parallel_planes_sufficient_not_necessary_l121_12139


namespace square_floor_tiles_l121_12142

/-- A square floor covered with congruent square tiles -/
structure SquareFloor :=
  (side_length : ℕ)

/-- The number of tiles along the diagonals of a square floor -/
def diagonal_tiles (floor : SquareFloor) : ℕ :=
  2 * floor.side_length - 1

/-- The total number of tiles covering a square floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length ^ 2

/-- Theorem: If the total number of tiles along the two diagonals is 49,
    then the number of tiles covering the entire floor is 625 -/
theorem square_floor_tiles (floor : SquareFloor) :
  diagonal_tiles floor = 49 → total_tiles floor = 625 :=
by
  sorry


end square_floor_tiles_l121_12142


namespace intersection_when_m_is_5_subset_condition_l121_12185

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 10}

-- Theorem for part 1
theorem intersection_when_m_is_5 :
  A 5 ∩ B = {x | 6 ≤ x ∧ x ≤ 10} := by sorry

-- Theorem for part 2
theorem subset_condition :
  ∀ m : ℝ, A m ⊆ B ↔ m ≤ 11/3 := by sorry

end intersection_when_m_is_5_subset_condition_l121_12185


namespace polygon_properties_l121_12150

/-- The number of diagonals from a vertex in a polygon with n sides -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- The number of triangles formed by diagonals in a polygon with n sides -/
def triangles_formed (n : ℕ) : ℕ := n - 2

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- The sum of exterior angles of any polygon -/
def sum_exterior_angles : ℕ := 360

theorem polygon_properties :
  (diagonals_from_vertex 6 = 3) ∧
  (triangles_formed 6 = 4) ∧
  (sum_interior_angles 6 = 720) ∧
  (∃ n : ℕ, sum_interior_angles n = 2 * sum_exterior_angles - 180 ∧ n = 5) :=
sorry

end polygon_properties_l121_12150


namespace log_equality_implies_n_fifth_power_l121_12116

theorem log_equality_implies_n_fifth_power (n : ℝ) :
  n > 0 →
  (Real.log (675 * Real.sqrt 3)) / (Real.log (3 * n)) = (Real.log 75) / (Real.log n) →
  n^5 = 5625 := by
  sorry

end log_equality_implies_n_fifth_power_l121_12116


namespace smallest_sum_of_reciprocals_l121_12158

theorem smallest_sum_of_reciprocals (x y : ℕ+) :
  x ≠ y →
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 18 →
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 18 →
  (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ) →
  (x : ℕ) + (y : ℕ) = 75 :=
by sorry

end smallest_sum_of_reciprocals_l121_12158


namespace rhombus_perimeter_l121_12137

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := by
  sorry

#check rhombus_perimeter

end rhombus_perimeter_l121_12137


namespace special_bin_op_property_l121_12108

/-- A binary operation on a set S satisfying (a * b) * a = b for all a, b ∈ S -/
class SpecialBinOp (S : Type) where
  op : S → S → S
  identity : ∀ a b : S, op (op a b) a = b

/-- 
If S has a binary operation satisfying (a * b) * a = b for all a, b ∈ S,
then a * (b * a) = b for all a, b ∈ S
-/
theorem special_bin_op_property {S : Type} [SpecialBinOp S] :
  ∀ a b : S, SpecialBinOp.op a (SpecialBinOp.op b a) = b :=
by sorry

end special_bin_op_property_l121_12108


namespace last_two_digits_product_l121_12102

theorem last_two_digits_product (n : ℤ) : 
  (n % 100 ≥ 10) →  -- Ensure n has at least two digits
  (n % 8 = 0) →     -- n is divisible by 8
  ((n % 100) / 10 + n % 10 = 12) →  -- Sum of last two digits is 12
  ((n % 100) / 10 * (n % 10) = 32) :=  -- Product of last two digits is 32
by sorry

end last_two_digits_product_l121_12102


namespace negative_abs_negative_three_l121_12154

theorem negative_abs_negative_three : -|-3| = -3 := by
  sorry

end negative_abs_negative_three_l121_12154


namespace base_with_five_digits_l121_12128

theorem base_with_five_digits : ∃! b : ℕ+, b ≥ 2 ∧ b ^ 4 ≤ 500 ∧ 500 < b ^ 5 := by sorry

end base_with_five_digits_l121_12128


namespace total_flower_cost_l121_12117

-- Define the promenade perimeter in meters
def promenade_perimeter : ℕ := 1500

-- Define the planting interval in meters
def planting_interval : ℕ := 30

-- Define the cost per flower in won
def cost_per_flower : ℕ := 5000

-- Theorem to prove
theorem total_flower_cost : 
  (promenade_perimeter / planting_interval) * cost_per_flower = 250000 := by
sorry

end total_flower_cost_l121_12117


namespace wednesday_occurs_five_times_in_august_l121_12124

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- July of year N -/
def july : Month := { days := 31, firstDay := DayOfWeek.Tuesday }

/-- August of year N -/
def august : Month := { days := 31, firstDay := sorry }

/-- Counts the occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat := sorry

/-- The main theorem -/
theorem wednesday_occurs_five_times_in_august :
  (countDayOccurrences july DayOfWeek.Tuesday = 5) →
  (countDayOccurrences august DayOfWeek.Wednesday = 5) := by
  sorry

end wednesday_occurs_five_times_in_august_l121_12124


namespace circle_radius_is_5_l121_12131

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*x + y^2 - 21 = 0

/-- The radius of the circle defined by circle_equation -/
def circle_radius : ℝ := 5

/-- Theorem: The radius of the circle defined by circle_equation is 5 -/
theorem circle_radius_is_5 : 
  ∀ x y : ℝ, circle_equation x y → 
  ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = circle_radius^2 :=
sorry

end circle_radius_is_5_l121_12131


namespace tangent_line_intercept_l121_12184

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + a*x + 2

-- Define the derivative of f(x)
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 6*x + a

theorem tangent_line_intercept (a : ℝ) : 
  (f a 0 = 2) ∧ 
  (∃ m : ℝ, ∀ x : ℝ, m*x + 2 = f_prime a 0 * x + 2) ∧
  (∃ t : ℝ, t = -2 ∧ f_prime a 0 * t + 2 = 0) →
  a = 1 := by sorry

end tangent_line_intercept_l121_12184


namespace evaluate_expression_l121_12101

theorem evaluate_expression (x : ℝ) (h : 3 * x - 2 = 13) : 6 * x - 4 = 26 := by
  sorry

end evaluate_expression_l121_12101


namespace cat_and_mouse_positions_l121_12110

def cat_cycle_length : ℕ := 4
def mouse_cycle_length : ℕ := 8
def total_moves : ℕ := 247

theorem cat_and_mouse_positions :
  (total_moves % cat_cycle_length = 3) ∧
  (total_moves % mouse_cycle_length = 7) := by
  sorry

end cat_and_mouse_positions_l121_12110


namespace sqrt_18_times_sqrt_32_l121_12190

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end sqrt_18_times_sqrt_32_l121_12190


namespace square_difference_formula_l121_12129

theorem square_difference_formula (x y : ℚ) 
  (sum_eq : x + y = 8/15)
  (diff_eq : x - y = 2/15) :
  x^2 - y^2 = 16/225 := by
sorry

end square_difference_formula_l121_12129


namespace smallest_odd_number_l121_12161

theorem smallest_odd_number (x : ℤ) : 
  (x % 2 = 1) →  -- x is odd
  ((x + 2) % 2 = 1) →  -- x + 2 is odd
  ((x + 4) % 2 = 1) →  -- x + 4 is odd
  (x + (x + 2) + (x + 4) = (x + 4) + 28) →  -- sum condition
  (x = 13) :=  -- conclusion
by sorry

end smallest_odd_number_l121_12161


namespace fraction_equality_l121_12180

theorem fraction_equality (a b : ℚ) (h : (a - 2*b) / b = 3/5) : a / b = 13/5 := by
  sorry

end fraction_equality_l121_12180


namespace smallest_base_for_fourth_power_l121_12136

/-- Given an integer N represented as 777 in base b, 
    18 is the smallest b for which N is a fourth power -/
theorem smallest_base_for_fourth_power (N : ℤ) (b : ℤ) : 
  N = 7 * b^2 + 7 * b + 7 → -- N's representation in base b is 777
  (∃ (a : ℤ), N = a^4) →    -- N is a fourth power
  b ≥ 18 :=                 -- 18 is the smallest such b
by sorry

end smallest_base_for_fourth_power_l121_12136


namespace excircle_geometric_mean_implies_side_relation_l121_12160

/-- 
Given a triangle with sides a, b, and c, and excircle radii ra, rb, and rc opposite to sides a, b, and c respectively,
if rc is the geometric mean of ra and rb, then c = (a^2 + b^2) / (a + b).
-/
theorem excircle_geometric_mean_implies_side_relation 
  {a b c ra rb rc : ℝ} 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ ra > 0 ∧ rb > 0 ∧ rc > 0)
  (h_triangle : c < a + b)
  (h_geometric_mean : rc^2 = ra * rb) :
  c = (a^2 + b^2) / (a + b) := by
sorry

end excircle_geometric_mean_implies_side_relation_l121_12160


namespace simplify_nested_roots_l121_12171

theorem simplify_nested_roots (b : ℝ) (hb : b > 0) :
  (((b^16)^(1/8))^(1/4))^3 * (((b^16)^(1/4))^(1/8))^3 = b^3 := by
  sorry

end simplify_nested_roots_l121_12171


namespace functional_equation_solutions_l121_12100

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + x * y) = f x * f (y + 1)

theorem functional_equation_solutions :
  ∀ f : ℝ → ℝ, functional_equation f →
    (∀ x, f x = 0) ∨ (∀ x, f x = 1) ∨ (∀ x, f x = x) := by
  sorry

end functional_equation_solutions_l121_12100


namespace discount_profit_calculation_l121_12106

-- Define the discount percentage
def discount : ℝ := 0.05

-- Define the profit percentage without discount
def profit_without_discount : ℝ := 0.29

-- Define the function to calculate profit percentage with discount
def profit_with_discount (d : ℝ) (p : ℝ) : ℝ :=
  (1 - d) * (1 + p) - 1

-- Theorem statement
theorem discount_profit_calculation :
  abs (profit_with_discount discount profit_without_discount - 0.2255) < 0.0001 := by
  sorry

end discount_profit_calculation_l121_12106


namespace greatest_integer_with_gcf_five_exists_185_solution_is_185_l121_12146

theorem greatest_integer_with_gcf_five (n : ℕ) : n < 200 ∧ Nat.gcd n 30 = 5 → n ≤ 185 :=
by
  sorry

theorem exists_185 : 185 < 200 ∧ Nat.gcd 185 30 = 5 :=
by
  sorry

theorem solution_is_185 : ∃ (n : ℕ), n = 185 ∧ n < 200 ∧ Nat.gcd n 30 = 5 ∧
  ∀ (m : ℕ), m < 200 ∧ Nat.gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_integer_with_gcf_five_exists_185_solution_is_185_l121_12146


namespace intersects_x_axis_once_iff_a_eq_zero_or_one_or_nine_l121_12199

/-- A function f(x) intersects the x-axis at only one point if and only if
    it has exactly one real root or it is a non-constant linear function. -/
def intersects_x_axis_once (f : ℝ → ℝ) : Prop :=
  (∃! x, f x = 0) ∨ (∃ m b, m ≠ 0 ∧ ∀ x, f x = m * x + b)

/-- The quadratic function f(x) = ax² + (a-3)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 3) * x + 1

theorem intersects_x_axis_once_iff_a_eq_zero_or_one_or_nine :
  ∀ a : ℝ, intersects_x_axis_once (f a) ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end intersects_x_axis_once_iff_a_eq_zero_or_one_or_nine_l121_12199


namespace fair_number_exists_l121_12182

/-- Represents a digit as a natural number between 0 and 9 -/
def Digit : Type := { n : ℕ // n < 10 }

/-- Represents a number as a list of digits -/
def Number := List Digit

/-- Checks if a digit is even -/
def isEven (d : Digit) : Bool :=
  d.val % 2 = 0

/-- Counts the number of even digits at odd positions and even positions -/
def countEvenDigits (n : Number) : ℕ × ℕ :=
  let rec count (digits : List Digit) (isOddPosition : Bool) (evenOdd evenEven : ℕ) : ℕ × ℕ :=
    match digits with
    | [] => (evenOdd, evenEven)
    | d :: ds =>
      if isEven d then
        if isOddPosition then
          count ds (not isOddPosition) (evenOdd + 1) evenEven
        else
          count ds (not isOddPosition) evenOdd (evenEven + 1)
      else
        count ds (not isOddPosition) evenOdd evenEven
  count n true 0 0

/-- Checks if a number is fair (equal number of even digits at odd and even positions) -/
def isFair (n : Number) : Bool :=
  let (evenOdd, evenEven) := countEvenDigits n
  evenOdd = evenEven

/-- Main theorem: For any number with an odd number of digits, 
    there exists a way to remove one digit to make it fair -/
theorem fair_number_exists (n : Number) (h : n.length % 2 = 1) :
  ∃ (i : Fin n.length), isFair (n.removeNth i) := by
  sorry

end fair_number_exists_l121_12182


namespace money_division_l121_12198

theorem money_division (a b c : ℝ) (h1 : a = (1/2) * b) (h2 : b = (1/2) * c) (h3 : c = 208) :
  a + b + c = 364 := by sorry

end money_division_l121_12198


namespace algebraic_expression_simplification_l121_12166

theorem algebraic_expression_simplification :
  let x : ℝ := (Real.sqrt 3) / 2 + 1 / 2
  (1 / x + (x + 1) / x) / ((x + 2) / (x^2 + x)) = (Real.sqrt 3 + 3) / 2 :=
by sorry

end algebraic_expression_simplification_l121_12166


namespace max_sum_on_circle_l121_12155

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 36) : x + y ≤ 8 := by
  sorry

end max_sum_on_circle_l121_12155


namespace inequality_problem_l121_12197

theorem inequality_problem :
  (∀ x : ℝ, |x + 7| + |x - 1| ≥ 8) ∧
  (¬ ∃ m : ℝ, m > 8 ∧ ∀ x : ℝ, |x + 7| + |x - 1| ≥ m) ∧
  (∀ x : ℝ, |x - 3| - 2*x ≤ 4 ↔ x ≥ -1/3) :=
by sorry

end inequality_problem_l121_12197


namespace sphere_cylinder_volumes_l121_12176

/-- Given a sphere with surface area 144π cm² that fits exactly inside a cylinder
    with height equal to the sphere's diameter, prove that the volume of the sphere
    is 288π cm³ and the volume of the cylinder is 432π cm³. -/
theorem sphere_cylinder_volumes (r : ℝ) (h : 4 * Real.pi * r^2 = 144 * Real.pi) :
  (4/3 : ℝ) * Real.pi * r^3 = 288 * Real.pi ∧
  Real.pi * r^2 * (2*r) = 432 * Real.pi := by
  sorry

end sphere_cylinder_volumes_l121_12176


namespace rational_numbers_composition_l121_12186

-- Define the set of integers
def Integers : Set ℚ := {x : ℚ | ∃ n : ℤ, x = n}

-- Define the set of fractions
def Fractions : Set ℚ := {x : ℚ | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}

-- Theorem statement
theorem rational_numbers_composition :
  Set.univ = Integers ∪ Fractions :=
sorry

end rational_numbers_composition_l121_12186


namespace apple_mango_equivalence_l121_12177

theorem apple_mango_equivalence (apple_value mango_value : ℝ) :
  (5 / 4 * 16 * apple_value = 10 * mango_value) →
  (3 / 4 * 12 * apple_value = 4.5 * mango_value) := by
  sorry

end apple_mango_equivalence_l121_12177


namespace binary_product_theorem_l121_12130

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binaryToNat (bits : List Bool) : Nat :=
  bits.reverse.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number to its binary representation. -/
def natToBinary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
  toBinary n

theorem binary_product_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [true, true, true, true, false, false, true]  -- 1001111₂
  binaryToNat a * binaryToNat b = binaryToNat c := by
  sorry

end binary_product_theorem_l121_12130


namespace cube_volume_from_surface_area_l121_12159

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 1734 → volume = (((surface_area / 6) ^ (1/2 : ℝ)) ^ 3) → volume = 4913 := by
  sorry

end cube_volume_from_surface_area_l121_12159


namespace cube_distance_to_plane_l121_12189

theorem cube_distance_to_plane (cube_side : ℝ) (height1 height2 height3 : ℝ) 
  (r s t : ℕ+) (d : ℝ) :
  cube_side = 15 →
  height1 = 15 ∧ height2 = 16 ∧ height3 = 17 →
  d = (r : ℝ) - Real.sqrt s / (t : ℝ) →
  d = (48 - Real.sqrt 224) / 3 →
  r + s + t = 275 := by
sorry

end cube_distance_to_plane_l121_12189


namespace max_value_of_expression_l121_12169

def S : Set Int := {-1, -2, -3, -4, -5}

theorem max_value_of_expression (a b c d : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  (∀ (w x y z : Int), w ∈ S → x ∈ S → y ∈ S → z ∈ S →
    w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    (w^x + y^z : Rat) ≤ 10/9) ∧
  (∃ (w x y z : Int), w ∈ S ∧ x ∈ S ∧ y ∈ S ∧ z ∈ S ∧
    w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    (w^x + y^z : Rat) = 10/9) :=
by sorry

end max_value_of_expression_l121_12169


namespace largest_integer_before_zero_l121_12156

noncomputable def f (x : ℝ) := Real.log x + 2 * x - 6

theorem largest_integer_before_zero (x₀ : ℝ) (h : f x₀ = 0) :
  ∃ k : ℤ, k = 2 ∧ k ≤ ⌊x₀⌋ ∧ ∀ m : ℤ, m > k → m > ⌊x₀⌋ :=
sorry

end largest_integer_before_zero_l121_12156


namespace first_term_of_arithmetic_sequence_l121_12121

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem first_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a5 : a 5 = 9) 
  (h_a3_a2 : 2 * a 3 = a 2 + 6) : 
  a 1 = -3 := by
sorry

end first_term_of_arithmetic_sequence_l121_12121


namespace unique_relation_sum_l121_12193

theorem unique_relation_sum (a b c : ℕ) : 
  ({a, b, c} : Set ℕ) = {1, 2, 3} →
  (((a ≠ 3 ∧ b ≠ 3 ∧ c = 3) ∨ (a ≠ 3 ∧ b = 3 ∧ c ≠ 3) ∨ (a = 3 ∧ b ≠ 3 ∧ c ≠ 3)) ∧
   ¬((a ≠ 3 ∧ b ≠ 3 ∧ c = 3) ∧ (a ≠ 3 ∧ b = 3 ∧ c ≠ 3)) ∧
   ¬((a ≠ 3 ∧ b ≠ 3 ∧ c = 3) ∧ (a = 3 ∧ b ≠ 3 ∧ c ≠ 3)) ∧
   ¬((a ≠ 3 ∧ b = 3 ∧ c ≠ 3) ∧ (a = 3 ∧ b ≠ 3 ∧ c ≠ 3))) →
  100 * a + 10 * b + c = 312 := by
sorry

end unique_relation_sum_l121_12193


namespace perfect_square_swap_l121_12105

theorem perfect_square_swap (a b : ℕ) (ha : a > b) (hb : b > 0) 
  (hA : ∃ k : ℕ, a^2 + 4*b + 1 = k^2) 
  (hB : ∃ m : ℕ, b^2 + 4*a + 1 = m^2) : 
  a = 8 ∧ b = 4 := by sorry

end perfect_square_swap_l121_12105


namespace milk_price_is_three_l121_12103

/-- Represents the milk and butter selling scenario --/
structure MilkButterScenario where
  num_cows : ℕ
  milk_per_cow : ℕ
  num_customers : ℕ
  milk_per_customer : ℕ
  butter_sticks_per_gallon : ℕ
  butter_price_per_stick : ℚ
  total_earnings : ℚ

/-- Calculates the price per gallon of milk --/
def price_per_gallon (scenario : MilkButterScenario) : ℚ :=
  let total_milk := scenario.num_cows * scenario.milk_per_cow
  let sold_milk := scenario.num_customers * scenario.milk_per_customer
  let remaining_milk := total_milk - sold_milk
  let butter_sticks := remaining_milk * scenario.butter_sticks_per_gallon
  let butter_earnings := butter_sticks * scenario.butter_price_per_stick
  let milk_earnings := scenario.total_earnings - butter_earnings
  milk_earnings / sold_milk

/-- Theorem stating that the price per gallon of milk is $3 --/
theorem milk_price_is_three (scenario : MilkButterScenario) 
  (h1 : scenario.num_cows = 12)
  (h2 : scenario.milk_per_cow = 4)
  (h3 : scenario.num_customers = 6)
  (h4 : scenario.milk_per_customer = 6)
  (h5 : scenario.butter_sticks_per_gallon = 2)
  (h6 : scenario.butter_price_per_stick = 3/2)
  (h7 : scenario.total_earnings = 144) :
  price_per_gallon scenario = 3 := by
  sorry

end milk_price_is_three_l121_12103


namespace kingfisher_to_warbler_ratio_l121_12123

/-- Represents the composition of bird species in the Goshawk-Eurasian Nature Reserve -/
structure BirdPopulation where
  hawks : ℝ
  paddyfieldWarblers : ℝ
  kingfishers : ℝ
  others : ℝ

/-- The conditions of the bird population in the nature reserve -/
def validBirdPopulation (bp : BirdPopulation) : Prop :=
  bp.hawks = 0.3 ∧
  bp.paddyfieldWarblers = 0.4 * (1 - bp.hawks) ∧
  bp.others = 0.35 ∧
  bp.hawks + bp.paddyfieldWarblers + bp.kingfishers + bp.others = 1

/-- The theorem stating the relationship between kingfishers and paddyfield-warblers -/
theorem kingfisher_to_warbler_ratio (bp : BirdPopulation) 
  (h : validBirdPopulation bp) : 
  bp.kingfishers / bp.paddyfieldWarblers = 0.25 := by
  sorry

end kingfisher_to_warbler_ratio_l121_12123


namespace square_sum_equals_one_l121_12162

theorem square_sum_equals_one (a b : ℝ) 
  (h : a * Real.sqrt (1 - b^2) + b * Real.sqrt (1 - a^2) = 1) : 
  a^2 + b^2 = 1 := by
sorry

end square_sum_equals_one_l121_12162


namespace triangle_sine_identity_l121_12135

theorem triangle_sine_identity (A B C : ℝ) (h₁ : 0 < A) (h₂ : 0 < B) (h₃ : 0 < C) 
  (h₄ : A + B + C = π) (h₅ : 4 * Real.sin A * Real.sin B * Real.cos C = Real.sin A ^ 2 + Real.sin B ^ 2) : 
  Real.sin A ^ 2 + Real.sin B ^ 2 = 2 * Real.sin C ^ 2 := by
  sorry

end triangle_sine_identity_l121_12135


namespace compute_expression_l121_12141

theorem compute_expression : 8 * (2 / 3)^4 = 128 / 81 := by
  sorry

end compute_expression_l121_12141


namespace geometric_series_sum_l121_12126

theorem geometric_series_sum (a b : ℝ) (h : ∑' n, a / b^n = 5) : 
  ∑' n, a / (a + b)^n = 5/6 := by
sorry

end geometric_series_sum_l121_12126


namespace optimal_shelf_arrangement_l121_12175

def math_books : ℕ := 130
def portuguese_books : ℕ := 195

theorem optimal_shelf_arrangement :
  ∃ (n : ℕ), n > 0 ∧
  n ∣ math_books ∧
  n ∣ portuguese_books ∧
  (∀ m : ℕ, m > n → ¬(m ∣ math_books ∧ m ∣ portuguese_books)) ∧
  n = 65 := by
  sorry

end optimal_shelf_arrangement_l121_12175


namespace unique_number_l121_12153

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2*k + 1

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9*k

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem unique_number : 
  ∃! n : ℕ, is_two_digit n ∧ 
            is_odd n ∧ 
            is_multiple_of_9 n ∧ 
            is_perfect_square (digit_product n) ∧
            n = 99 := by
  sorry

end unique_number_l121_12153


namespace cone_volume_l121_12111

/-- The volume of a cone with given slant height and height --/
theorem cone_volume (slant_height height : ℝ) (h1 : slant_height = 15) (h2 : height = 9) :
  (1 / 3 : ℝ) * Real.pi * (slant_height ^ 2 - height ^ 2) * height = 432 * Real.pi := by
  sorry

end cone_volume_l121_12111


namespace number_problem_l121_12120

theorem number_problem (A B C : ℝ) 
  (h1 : A - B = 1620)
  (h2 : 0.075 * A = 0.125 * B)
  (h3 : 0.06 * B = 0.10 * C) :
  A = 4050 ∧ B = 2430 ∧ C = 1458 := by
  sorry

end number_problem_l121_12120


namespace inequality_solutions_l121_12172

theorem inequality_solutions :
  (∀ x : ℝ, x^2 - 2*x - 1 > 0 ↔ (x > Real.sqrt 2 + 1 ∨ x < -Real.sqrt 2 + 1)) ∧
  (∀ x : ℝ, x ≠ 3 → ((2*x - 1) / (x - 3) ≥ 3 ↔ 3 < x ∧ x ≤ 8)) :=
by sorry

end inequality_solutions_l121_12172


namespace christen_peeled_24_potatoes_l121_12181

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  totalPotatoes : ℕ
  homerRate : ℕ
  christenRate : ℕ
  timeBeforeChristenJoins : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christenPeeledPotatoes (scenario : PotatoPeeling) : ℕ :=
  let potatoesLeftAfterHomer := scenario.totalPotatoes - scenario.homerRate * scenario.timeBeforeChristenJoins
  let combinedRate := scenario.homerRate + scenario.christenRate
  let timeForRemaining := potatoesLeftAfterHomer / combinedRate
  scenario.christenRate * timeForRemaining

/-- Theorem stating that Christen peeled 24 potatoes -/
theorem christen_peeled_24_potatoes (scenario : PotatoPeeling) 
  (h1 : scenario.totalPotatoes = 60)
  (h2 : scenario.homerRate = 3)
  (h3 : scenario.christenRate = 4)
  (h4 : scenario.timeBeforeChristenJoins = 6) :
  christenPeeledPotatoes scenario = 24 := by
  sorry

end christen_peeled_24_potatoes_l121_12181


namespace negation_equivalence_l121_12174

theorem negation_equivalence (a b : ℝ) :
  ¬(((a - 2) * (b - 3) = 0) → (a = 2 ∨ b = 3)) ↔
  (((a - 2) * (b - 3) ≠ 0) → (a ≠ 2 ∧ b ≠ 3)) := by sorry

end negation_equivalence_l121_12174


namespace unique_prime_solution_l121_12143

theorem unique_prime_solution : ∃! (p : ℕ), Prime p ∧ (p^4 + 2*p^3 + 4*p^2 + 2*p + 1)^5 = 418195493 := by
  sorry

end unique_prime_solution_l121_12143


namespace sum_of_sixth_powers_l121_12109

theorem sum_of_sixth_powers (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 14) :
  ζ₁^6 + ζ₂^6 + ζ₃^6 = 128.75 := by
  sorry

end sum_of_sixth_powers_l121_12109


namespace morning_orange_sales_l121_12115

/-- Proves the number of oranges sold in the morning given fruit prices and sales data --/
theorem morning_orange_sales
  (apple_price : ℚ)
  (orange_price : ℚ)
  (morning_apples : ℕ)
  (afternoon_apples : ℕ)
  (afternoon_oranges : ℕ)
  (total_sales : ℚ)
  (h1 : apple_price = 3/2)
  (h2 : orange_price = 1)
  (h3 : morning_apples = 40)
  (h4 : afternoon_apples = 50)
  (h5 : afternoon_oranges = 40)
  (h6 : total_sales = 205) :
  ∃ morning_oranges : ℕ,
    morning_oranges = 30 ∧
    total_sales = apple_price * (morning_apples + afternoon_apples : ℚ) +
                  orange_price * (morning_oranges + afternoon_oranges : ℚ) :=
by sorry

end morning_orange_sales_l121_12115


namespace abs_a_minus_b_ge_four_l121_12165

theorem abs_a_minus_b_ge_four (a b : ℝ) : 
  (∀ x : ℝ, |x - a| < 1 → |x - b| > 3) → 
  |a - b| ≥ 4 := by
  sorry

end abs_a_minus_b_ge_four_l121_12165


namespace traffic_light_theorem_l121_12112

/-- Represents the probability of different traffic light combinations -/
structure TrafficLightProbabilities where
  p1 : ℝ  -- Both lights green
  p2 : ℝ  -- First green, second red
  p3 : ℝ  -- First red, second green
  p4 : ℝ  -- Both lights red

/-- The conditions of the traffic light problem -/
def traffic_light_conditions (p : TrafficLightProbabilities) : Prop :=
  0 ≤ p.p1 ∧ 0 ≤ p.p2 ∧ 0 ≤ p.p3 ∧ 0 ≤ p.p4 ∧  -- Probabilities are non-negative
  p.p1 + p.p2 + p.p3 + p.p4 = 1 ∧  -- Sum of probabilities is 1
  p.p1 + p.p2 = 2/3 ∧  -- First light is green 2/3 of the time
  p.p1 + p.p3 = 2/3 ∧  -- Second light is green 2/3 of the time
  p.p1 / (p.p1 + p.p2) = 3/4  -- Given first is green, second is green 3/4 of the time

/-- The theorem to be proved -/
theorem traffic_light_theorem (p : TrafficLightProbabilities) 
  (h : traffic_light_conditions p) : 
  p.p4 / (p.p3 + p.p4) = 1/2 := by
  sorry

end traffic_light_theorem_l121_12112


namespace sum_of_unknown_numbers_l121_12179

def known_numbers : List ℕ := [690, 744, 745, 747, 748, 749, 752, 752, 753, 755, 760, 769]

theorem sum_of_unknown_numbers 
  (total_count : ℕ) 
  (average : ℕ) 
  (h1 : total_count = 15) 
  (h2 : average = 750) 
  (h3 : known_numbers.length = 12) : 
  (total_count * average) - known_numbers.sum = 2336 := by
  sorry

end sum_of_unknown_numbers_l121_12179


namespace prime_equation_l121_12140

theorem prime_equation (a b : ℕ) : 
  Prime a → Prime b → a^11 + b = 2089 → 49*b - a = 2007 := by sorry

end prime_equation_l121_12140


namespace no_solution_when_m_equals_negative_one_l121_12119

theorem no_solution_when_m_equals_negative_one :
  ∀ x : ℝ, (3 - 2*x) / (x - 3) + (2 + (-1)*x) / (3 - x) ≠ -1 :=
by
  sorry

end no_solution_when_m_equals_negative_one_l121_12119


namespace adoption_fee_calculation_l121_12167

theorem adoption_fee_calculation (james_payment : ℝ) (friend_percentage : ℝ) : 
  james_payment = 150 → friend_percentage = 0.25 → 
  ∃ (total_fee : ℝ), total_fee = 200 ∧ james_payment = (1 - friend_percentage) * total_fee :=
sorry

end adoption_fee_calculation_l121_12167


namespace triangle_area_theorem_l121_12148

/-- Given a triangle with vertices at (0, 0), (x, 3x), and (x, 0), where x > 0,
    if the area of the triangle is 81 square units, then x = 3√6. -/
theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 81 → x = 3 * Real.sqrt 6 := by
sorry

end triangle_area_theorem_l121_12148


namespace solution_exists_l121_12178

-- Define the vector type
def Vec2 := Fin 2 → ℝ

-- Define the constants a and b
variable (a b : ℝ)

-- Define the vectors
def v1 : Vec2 := ![1, 4]
def v2 : Vec2 := ![3, -2]
def result : Vec2 := ![5, 6]

-- Define vector addition and scalar multiplication
def add (u v : Vec2) : Vec2 := λ i => u i + v i
def smul (c : ℝ) (v : Vec2) : Vec2 := λ i => c * v i

-- State the theorem
theorem solution_exists :
  ∃ a b : ℝ, add (smul a v1) (smul b v2) = result ∧ a = 2 ∧ b = 1 := by
  sorry

end solution_exists_l121_12178


namespace complex_subtraction_simplification_l121_12144

theorem complex_subtraction_simplification :
  (5 - 7 * Complex.I) - (3 - 2 * Complex.I) = 2 - 5 * Complex.I :=
by sorry

end complex_subtraction_simplification_l121_12144


namespace roller_alignment_l121_12196

/-- The number of rotations needed for alignment of two rollers -/
def alignmentRotations (r1 r2 : ℕ) : ℕ :=
  (Nat.lcm r1 r2) / r1

/-- Theorem: The number of rotations for alignment of rollers with radii 105 and 90 is 6 -/
theorem roller_alignment :
  alignmentRotations 105 90 = 6 := by
  sorry

end roller_alignment_l121_12196


namespace system_solution_l121_12132

theorem system_solution : 
  ∀ (a b c d : ℝ), 
    a + c = -7 ∧ 
    a * c + b + d = 18 ∧ 
    a * d + b * c = -22 ∧ 
    b * d = 12 → 
    ((a = -5 ∧ b = 6 ∧ c = -2 ∧ d = 2) ∨ 
     (a = -2 ∧ b = 2 ∧ c = -5 ∧ d = 6)) := by
  sorry

end system_solution_l121_12132


namespace fertility_rate_not_valid_indicator_other_indicators_are_valid_l121_12147

-- Define the type for population growth indicators
inductive PopulationGrowthIndicator
  | BirthRate
  | MortalityRate
  | NaturalIncreaseRate
  | FertilityRate

-- Define the set of valid indicators
def validIndicators : Set PopulationGrowthIndicator :=
  {PopulationGrowthIndicator.BirthRate,
   PopulationGrowthIndicator.MortalityRate,
   PopulationGrowthIndicator.NaturalIncreaseRate}

-- Theorem: Fertility rate is not a valid indicator
theorem fertility_rate_not_valid_indicator :
  PopulationGrowthIndicator.FertilityRate ∉ validIndicators :=
by
  sorry

-- Theorem: All other indicators are valid
theorem other_indicators_are_valid :
  PopulationGrowthIndicator.BirthRate ∈ validIndicators ∧
  PopulationGrowthIndicator.MortalityRate ∈ validIndicators ∧
  PopulationGrowthIndicator.NaturalIncreaseRate ∈ validIndicators :=
by
  sorry

end fertility_rate_not_valid_indicator_other_indicators_are_valid_l121_12147


namespace set_equality_implies_difference_l121_12113

theorem set_equality_implies_difference (a b : ℝ) :
  ({0, b/a, b} : Set ℝ) = ({1, a+b, a} : Set ℝ) →
  b - a = 2 := by
sorry

end set_equality_implies_difference_l121_12113


namespace unique_two_digit_integer_l121_12104

theorem unique_two_digit_integer (s : ℕ) : 
  (10 ≤ s ∧ s < 100) ∧ (13 * s) % 100 = 52 ↔ s = 4 := by
  sorry

end unique_two_digit_integer_l121_12104


namespace two_digit_number_theorem_l121_12127

/-- Represents a two-digit integer with its tens and units digits. -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Theorem stating the relationship between j and k for a two-digit number. -/
theorem two_digit_number_theorem (n : TwoDigitNumber) (k j : ℚ) :
  (10 * n.tens + n.units : ℚ) = k * (n.tens + n.units) →
  (20 * n.units + n.tens : ℚ) = j * (n.tens + n.units) →
  j = (199 + k) / 10 := by
  sorry

end two_digit_number_theorem_l121_12127


namespace train_length_l121_12170

/-- Given a train crossing a bridge, calculate its length. -/
theorem train_length
  (train_speed : ℝ)
  (crossing_time : ℝ)
  (bridge_length : ℝ)
  (h1 : train_speed = 45)  -- km/hr
  (h2 : crossing_time = 30 / 3600)  -- convert seconds to hours
  (h3 : bridge_length = 220 / 1000)  -- convert meters to kilometers
  : (train_speed * crossing_time - bridge_length) * 1000 = 155 :=
by sorry

end train_length_l121_12170


namespace polynomial_remainder_theorem_l121_12188

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^4 + 2*x^2 + 2) % (x - 2) = 26 := by
  sorry

end polynomial_remainder_theorem_l121_12188
