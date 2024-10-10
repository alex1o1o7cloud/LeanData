import Mathlib

namespace min_value_of_expression_l1999_199999

theorem min_value_of_expression (a : ℝ) (h : a > 1) : 
  a + 1 / (a - 1) ≥ 3 ∧ ∃ a₀ > 1, a₀ + 1 / (a₀ - 1) = 3 :=
sorry

end min_value_of_expression_l1999_199999


namespace smallest_distance_to_2i_l1999_199945

theorem smallest_distance_to_2i (z : ℂ) (h : Complex.abs (z^2 + 3 + Complex.I) = Complex.abs (z * (z + 1 + 3 * Complex.I))) :
  Complex.abs (z - 2 * Complex.I) ≥ (1 : ℝ) / 2 ∧
  ∃ w : ℂ, Complex.abs (w^2 + 3 + Complex.I) = Complex.abs (w * (w + 1 + 3 * Complex.I)) ∧
           Complex.abs (w - 2 * Complex.I) = (1 : ℝ) / 2 :=
by sorry

end smallest_distance_to_2i_l1999_199945


namespace geometric_sequence_product_l1999_199931

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 8 * a 9 * a 10 = -a 13 ^ 2) →
  (a 8 * a 9 * a 10 = -1000) →
  a 10 * a 12 = 100 * Real.sqrt 10 := by
  sorry

end geometric_sequence_product_l1999_199931


namespace inverse_proportion_points_l1999_199957

/-- Given that (2,3) lies on the graph of y = k/x (k ≠ 0), prove that (1,6) also lies on the same graph. -/
theorem inverse_proportion_points : ∀ k : ℝ, k ≠ 0 → (3 = k / 2) → (6 = k / 1) := by
  sorry

end inverse_proportion_points_l1999_199957


namespace ball_distribution_problem_l1999_199970

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes,
    with at least one empty box -/
def distributeWithEmpty (n : ℕ) (k : ℕ) : ℕ := k * (k-1)^n

theorem ball_distribution_problem :
  distribute 6 3 - distributeWithEmpty 6 3 = 537 := by
  sorry

end ball_distribution_problem_l1999_199970


namespace ratio_transformation_l1999_199943

theorem ratio_transformation (x y : ℝ) (h : x / y = 7 / 3) : 
  (x + y) / (x - y) = 5 / 2 := by
sorry

end ratio_transformation_l1999_199943


namespace kangaroo_cant_reach_far_l1999_199942

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a valid jump for the kangaroo -/
def validJump (p q : Point) : Prop :=
  (q.x = p.x + 1 ∧ q.y = p.y - 1) ∨ (q.x = p.x - 5 ∧ q.y = p.y + 7)

/-- Defines if a point is in the first quadrant -/
def inFirstQuadrant (p : Point) : Prop :=
  p.x ≥ 0 ∧ p.y ≥ 0

/-- Defines if a point is at least 1000 units away from the origin -/
def farFromOrigin (p : Point) : Prop :=
  p.x^2 + p.y^2 ≥ 1000000

/-- Defines if a point can be reached through a sequence of valid jumps -/
def canReach (start target : Point) : Prop :=
  ∃ (n : ℕ) (path : ℕ → Point), 
    path 0 = start ∧ 
    path n = target ∧ 
    ∀ i < n, validJump (path i) (path (i+1)) ∧ inFirstQuadrant (path (i+1))

/-- The main theorem to be proved -/
theorem kangaroo_cant_reach_far (p : Point) 
  (h1 : inFirstQuadrant p) 
  (h2 : p.x + p.y ≤ 4) : 
  ¬∃ q : Point, canReach p q ∧ farFromOrigin q :=
sorry

end kangaroo_cant_reach_far_l1999_199942


namespace sample_size_equals_selected_students_l1999_199962

/-- Represents the sample size of a survey -/
def sample_size : ℕ := 1200

/-- Represents the number of students selected for the investigation -/
def selected_students : ℕ := 1200

/-- Theorem stating that the sample size is equal to the number of selected students -/
theorem sample_size_equals_selected_students : sample_size = selected_students := by
  sorry

end sample_size_equals_selected_students_l1999_199962


namespace juan_stamp_cost_l1999_199922

/-- Represents the cost of stamps for a given country -/
structure StampCost where
  country : String
  cost : Float

/-- Represents the number of stamps for a country in a specific decade -/
structure StampCount where
  country : String
  decade : String
  count : Nat

def brazil_cost : StampCost := ⟨"Brazil", 0.07⟩
def peru_cost : StampCost := ⟨"Peru", 0.05⟩

def brazil_70s : StampCount := ⟨"Brazil", "70s", 12⟩
def brazil_80s : StampCount := ⟨"Brazil", "80s", 15⟩
def peru_70s : StampCount := ⟨"Peru", "70s", 6⟩
def peru_80s : StampCount := ⟨"Peru", "80s", 12⟩

def total_cost (costs : List StampCost) (counts : List StampCount) : Float :=
  sorry

theorem juan_stamp_cost :
  total_cost [brazil_cost, peru_cost] [brazil_70s, brazil_80s, peru_70s, peru_80s] = 2.79 :=
sorry

end juan_stamp_cost_l1999_199922


namespace trajectory_equation_l1999_199936

/-- The trajectory of a point P satisfying |PM| - |PN| = 2√2, where M(-2, 0) and N(2, 0) are fixed points -/
def trajectory (x y : ℝ) : Prop :=
  x > 0 ∧ x^2 / 2 - y^2 / 2 = 1

/-- The distance condition for point P -/
def distance_condition (x y : ℝ) : Prop :=
  Real.sqrt ((x + 2)^2 + y^2) - Real.sqrt ((x - 2)^2 + y^2) = 2 * Real.sqrt 2

theorem trajectory_equation :
  ∀ x y : ℝ, distance_condition x y → trajectory x y :=
sorry

end trajectory_equation_l1999_199936


namespace boat_return_time_boat_return_time_example_l1999_199907

/-- The time taken for a boat to return upstream along a riverbank, given its downstream travel details and river flow speeds. -/
theorem boat_return_time (downstream_speed : ℝ) (downstream_time : ℝ) (downstream_distance : ℝ)
  (main_flow_speed : ℝ) (bank_flow_speed : ℝ) : ℝ :=
  let boat_speed := downstream_speed - main_flow_speed
  let upstream_speed := boat_speed - bank_flow_speed
  downstream_distance / upstream_speed

/-- The boat's return time is 20 hours given the specified conditions. -/
theorem boat_return_time_example : 
  boat_return_time 36 10 360 10 8 = 20 := by
  sorry

end boat_return_time_boat_return_time_example_l1999_199907


namespace cos_is_omega_2_on_0_1_sin_omega_t_characterization_sin_sum_range_for_omega_functions_l1999_199950

/-- Definition of Ω(t) function -/
def is_omega_t_function (f : ℝ → ℝ) (t a b : ℝ) : Prop :=
  a < b ∧ t > 0 ∧
  ((∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y) ∨ (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y)) ∧
  ((∀ x y, a + t ≤ x ∧ x < y ∧ y ≤ b + t → f x ≤ f y) ∨ (∀ x y, a + t ≤ x ∧ x < y ∧ y ≤ b + t → f x ≥ f y))

/-- Theorem: cos x is an Ω(2) function on [0,1] -/
theorem cos_is_omega_2_on_0_1 : is_omega_t_function Real.cos 2 0 1 := by sorry

/-- Theorem: Characterization of t for sin x to be an Ω(t) function on [-π/2, π/2] -/
theorem sin_omega_t_characterization (t : ℝ) : 
  is_omega_t_function Real.sin t (-π/2) (π/2) ↔ ∃ k : ℤ, t = 2 * π * k ∧ k > 0 := by sorry

/-- Theorem: Range of sin α + sin β for Ω functions -/
theorem sin_sum_range_for_omega_functions (α β : ℝ) :
  (∃ a B, is_omega_t_function Real.sin β a (α + B) ∧ is_omega_t_function Real.sin α B (α + β)) →
  (0 < Real.sin α + Real.sin β ∧ Real.sin α + Real.sin β ≤ 1) ∨ Real.sin α + Real.sin β = 2 := by sorry

end cos_is_omega_2_on_0_1_sin_omega_t_characterization_sin_sum_range_for_omega_functions_l1999_199950


namespace plumber_pipe_cost_l1999_199986

/-- The total cost of pipes bought by a plumber -/
def total_cost (copper_length plastic_length price_per_meter : ℕ) : ℕ :=
  (copper_length + plastic_length) * price_per_meter

/-- Theorem stating the total cost for the plumber's purchase -/
theorem plumber_pipe_cost :
  let copper_length : ℕ := 10
  let plastic_length : ℕ := copper_length + 5
  let price_per_meter : ℕ := 4
  total_cost copper_length plastic_length price_per_meter = 100 := by
  sorry

end plumber_pipe_cost_l1999_199986


namespace sufficient_not_necessary_l1999_199960

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x + 3 > 2 → -x < 6) ∧ 
  (∃ x : ℝ, -x < 6 ∧ ¬(x + 3 > 2)) := by
  sorry

end sufficient_not_necessary_l1999_199960


namespace lake_pleasant_excursion_l1999_199988

theorem lake_pleasant_excursion (total_kids : ℕ) 
  (h1 : total_kids = 40)
  (h2 : ∃ tubing_kids : ℕ, 4 * tubing_kids = total_kids)
  (h3 : ∃ rafting_kids : ℕ, 2 * rafting_kids = tubing_kids) :
  rafting_kids = 5 := by
  sorry

end lake_pleasant_excursion_l1999_199988


namespace students_disliking_menu_l1999_199975

theorem students_disliking_menu (total : ℕ) (liked : ℕ) (h1 : total = 400) (h2 : liked = 235) :
  total - liked = 165 := by
  sorry

end students_disliking_menu_l1999_199975


namespace cookie_sheet_length_l1999_199963

/-- Given a rectangle with width 10 inches and perimeter 24 inches, prove its length is 2 inches. -/
theorem cookie_sheet_length (width : ℝ) (perimeter : ℝ) (length : ℝ) : 
  width = 10 → perimeter = 24 → perimeter = 2 * (length + width) → length = 2 := by
  sorry

end cookie_sheet_length_l1999_199963


namespace sum_of_cubes_l1999_199992

theorem sum_of_cubes (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a * b + a * c + b * c = 5) 
  (h3 : a * b * c = -6) : 
  a^3 + b^3 + c^3 = -40 := by
sorry

end sum_of_cubes_l1999_199992


namespace intersection_when_a_is_one_subset_condition_l1999_199979

-- Define set A
def A : Set ℝ := {x | |x - 1| ≤ 1}

-- Define set B with parameter a
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem for part 1
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for part 2
theorem subset_condition (a : ℝ) :
  A ⊆ B a ↔ a ≤ 0 := by sorry

end intersection_when_a_is_one_subset_condition_l1999_199979


namespace inequality_theorem_l1999_199967

theorem inequality_theorem (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  a / (a - c) > b / (b - c) := by
  sorry

end inequality_theorem_l1999_199967


namespace tripled_base_and_exponent_l1999_199989

theorem tripled_base_and_exponent (a b x : ℝ) (h1 : b ≠ 0) :
  let r := (3*a)^(3*b)
  r = a^b * x^b → x = 27 * a^2 := by
sorry

end tripled_base_and_exponent_l1999_199989


namespace min_value_f_l1999_199929

open Real

noncomputable def f (x : ℝ) : ℝ := (sin x + 1) * (cos x + 1) / (sin x * cos x)

theorem min_value_f :
  ∀ x ∈ Set.Ioo 0 (π/2), f x ≥ 3 + 2 * sqrt 2 ∧
  ∃ x₀ ∈ Set.Ioo 0 (π/2), f x₀ = 3 + 2 * sqrt 2 :=
sorry

end min_value_f_l1999_199929


namespace lcm_812_3214_l1999_199951

theorem lcm_812_3214 : Nat.lcm 812 3214 = 1304124 := by
  sorry

end lcm_812_3214_l1999_199951


namespace lightning_rod_height_l1999_199998

/-- Given a lightning rod that breaks twice under strong wind conditions, 
    this theorem proves the height of the rod. -/
theorem lightning_rod_height (h : ℝ) (x₁ : ℝ) (x₂ : ℝ) : 
  h > 0 → 
  x₁ > 0 → 
  x₂ > 0 → 
  h^2 - x₁^2 = 400 → 
  h^2 - x₂^2 = 900 → 
  x₂ = x₁ - 5 → 
  h = Real.sqrt 3156.25 := by
sorry

end lightning_rod_height_l1999_199998


namespace arithmetic_sequence_problem_l1999_199905

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 4 + a 5 + a 13 + a 14 + a 15 = 8) :
  5 * a 7 - 2 * a 4 = 4 := by
  sorry

end arithmetic_sequence_problem_l1999_199905


namespace combined_mean_of_three_sets_l1999_199973

theorem combined_mean_of_three_sets (set1_count : ℕ) (set1_mean : ℚ)
                                    (set2_count : ℕ) (set2_mean : ℚ)
                                    (set3_count : ℕ) (set3_mean : ℚ) :
  set1_count = 7 ∧ set1_mean = 15 ∧
  set2_count = 8 ∧ set2_mean = 20 ∧
  set3_count = 5 ∧ set3_mean = 12 →
  (set1_count * set1_mean + set2_count * set2_mean + set3_count * set3_mean) /
  (set1_count + set2_count + set3_count) = 325 / 20 := by
sorry

end combined_mean_of_three_sets_l1999_199973


namespace sum_of_fractions_l1999_199920

theorem sum_of_fractions : 
  (1 / (1 * 2 : ℚ)) + (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + 
  (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 6 / 7 := by
  sorry

end sum_of_fractions_l1999_199920


namespace min_river_width_for_race_l1999_199953

/-- The width of a river that can accommodate a boat race -/
def river_width (num_boats : ℕ) (boat_width : ℕ) (space_between : ℕ) : ℕ :=
  num_boats * boat_width + (num_boats - 1) * space_between + 2 * space_between

/-- Theorem stating the minimum width of the river for the given conditions -/
theorem min_river_width_for_race : river_width 8 3 2 = 42 := by
  sorry

end min_river_width_for_race_l1999_199953


namespace rays_dog_walks_66_blocks_per_day_l1999_199917

/-- Represents the number of blocks Ray walks in each segment of his route -/
structure RouteSegments where
  toPark : ℕ
  toHighSchool : ℕ
  toHome : ℕ

/-- Calculates the total number of blocks walked in one complete route -/
def totalBlocksPerWalk (route : RouteSegments) : ℕ :=
  route.toPark + route.toHighSchool + route.toHome

/-- Represents Ray's daily dog walking routine -/
structure DailyWalk where
  route : RouteSegments
  frequency : ℕ

/-- Calculates the total number of blocks walked per day -/
def totalBlocksPerDay (daily : DailyWalk) : ℕ :=
  (totalBlocksPerWalk daily.route) * daily.frequency

/-- Theorem: Ray's dog walks 66 blocks each day -/
theorem rays_dog_walks_66_blocks_per_day :
  ∀ (daily : DailyWalk),
    daily.route.toPark = 4 →
    daily.route.toHighSchool = 7 →
    daily.route.toHome = 11 →
    daily.frequency = 3 →
    totalBlocksPerDay daily = 66 := by
  sorry


end rays_dog_walks_66_blocks_per_day_l1999_199917


namespace unique_positive_x_exists_l1999_199935

/-- Given a > b > 0, there exists a unique positive x such that 
    f(x) = ((a^(1/3) + b^(1/3)) / 2)^3, where f(x) = (2(a+b)x + 2ab) / (4x + a + b) -/
theorem unique_positive_x_exists (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃! x : ℝ, x > 0 ∧ (2 * (a + b) * x + 2 * a * b) / (4 * x + a + b) = ((a^(1/3) + b^(1/3)) / 2)^3 := by
  sorry

end unique_positive_x_exists_l1999_199935


namespace container_volume_ratio_l1999_199916

theorem container_volume_ratio (volume_first volume_second : ℚ) : 
  volume_first > 0 →
  volume_second > 0 →
  (4 / 5 : ℚ) * volume_first = (2 / 3 : ℚ) * volume_second →
  volume_first / volume_second = 5 / 6 := by
  sorry

end container_volume_ratio_l1999_199916


namespace apple_price_36kg_l1999_199918

/-- The price of apples with a two-tier pricing system -/
def apple_price (l q : ℚ) (kg : ℚ) : ℚ :=
  if kg ≤ 30 then l * kg
  else l * 30 + q * (kg - 30)

theorem apple_price_36kg (l q : ℚ) :
  (apple_price l q 33 = 360) →
  (apple_price l q 25 = 250) →
  (apple_price l q 36 = 420) :=
by sorry

end apple_price_36kg_l1999_199918


namespace ac_plus_one_lt_a_plus_c_l1999_199968

theorem ac_plus_one_lt_a_plus_c (a c : ℝ) (ha : 0 < a ∧ a < 1) (hc : c > 1) :
  a * c + 1 < a + c := by
  sorry

end ac_plus_one_lt_a_plus_c_l1999_199968


namespace sum_interior_angles_convex_polygon_l1999_199901

theorem sum_interior_angles_convex_polygon (n : ℕ) (h : n = 10) :
  (∃ (v : ℕ), v + 3 = n ∧ v = 7) →
  (n - 2) * 180 = 1440 :=
sorry

end sum_interior_angles_convex_polygon_l1999_199901


namespace sin_405_degrees_l1999_199997

theorem sin_405_degrees : Real.sin (405 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_degrees_l1999_199997


namespace arithmetic_calculation_l1999_199932

theorem arithmetic_calculation : 5 * 6 - 2 * 3 + 7 * 4 + 9 * 2 = 70 := by
  sorry

end arithmetic_calculation_l1999_199932


namespace square_equality_solution_l1999_199902

theorem square_equality_solution : ∃ (N : ℕ+), (36 ^ 2 * 72 ^ 2 : ℕ) = 12 ^ 2 * N ^ 2 ∧ N = 216 := by
  sorry

end square_equality_solution_l1999_199902


namespace overlaid_triangles_result_l1999_199904

/-- Represents a transparent sheet with shaded triangles -/
structure Sheet :=
  (total_triangles : Nat)
  (shaded_triangles : Nat)

/-- Calculates the number of visible shaded triangles when sheets are overlaid -/
def visible_shaded_triangles (sheets : List Sheet) : Nat :=
  sorry

/-- Theorem stating the result for the specific problem -/
theorem overlaid_triangles_result :
  let sheets := [
    { total_triangles := 49, shaded_triangles := 16 },
    { total_triangles := 49, shaded_triangles := 16 },
    { total_triangles := 49, shaded_triangles := 16 }
  ]
  visible_shaded_triangles sheets = 31 := by
  sorry

end overlaid_triangles_result_l1999_199904


namespace min_value_quadratic_l1999_199911

theorem min_value_quadratic (s : ℝ) :
  -8 * s^2 + 64 * s + 20 ≥ 148 ∧ ∃ t : ℝ, -8 * t^2 + 64 * t + 20 = 148 := by
  sorry

end min_value_quadratic_l1999_199911


namespace five_items_three_categories_l1999_199937

/-- The number of ways to distribute n distinct items among k distinct categories,
    where each item must be used exactly once. -/
def distributionCount (n k : ℕ) : ℕ :=
  k^n - (k * 1 + k.choose 2 * (2^n - 2))

/-- Theorem stating that there are 150 ways to distribute 5 distinct items
    among 3 distinct categories, where each item must be used exactly once. -/
theorem five_items_three_categories :
  distributionCount 5 3 = 150 := by
  sorry

end five_items_three_categories_l1999_199937


namespace chess_player_win_loss_difference_l1999_199969

theorem chess_player_win_loss_difference
  (total_games : ℕ)
  (total_points : ℚ)
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)
  (h1 : total_games = 40)
  (h2 : total_points = 25)
  (h3 : wins + draws + losses = total_games)
  (h4 : wins + (1/2 : ℚ) * draws = total_points) :
  wins - losses = 10 := by
  sorry

end chess_player_win_loss_difference_l1999_199969


namespace books_on_cart_l1999_199987

/-- The number of books on a cart -/
theorem books_on_cart 
  (fiction : ℕ) 
  (non_fiction : ℕ) 
  (autobiographies : ℕ) 
  (picture : ℕ) 
  (h1 : fiction = 5)
  (h2 : non_fiction = fiction + 4)
  (h3 : autobiographies = 2 * fiction)
  (h4 : picture = 11) :
  fiction + non_fiction + autobiographies + picture = 35 := by
sorry

end books_on_cart_l1999_199987


namespace range_of_a_l1999_199981

/-- The function g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-1)*x + a - 2*a^2

/-- The function h(x) -/
def h (x : ℝ) : ℝ := (x-1)^2

/-- The set A -/
def A (a : ℝ) : Set ℝ := {x | g a x > 0}

/-- The set B -/
def B : Set ℝ := {x | h x < 1}

/-- The function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x * g a x

/-- The set C -/
def C (a : ℝ) : Set ℝ := {x | f a x > 0}

/-- The theorem stating the range of a -/
theorem range_of_a :
  ∀ a : ℝ, (A a ∩ B).Nonempty ∧ (C a ∩ B).Nonempty ↔ 
    (1/3 < a ∧ a < 2) ∨ (-1/2 < a ∧ a < 1/3) :=
sorry

end range_of_a_l1999_199981


namespace youngest_child_age_l1999_199909

theorem youngest_child_age (n : ℕ) 
  (h1 : ∃ x : ℕ, x + (x + 2) + (x + 4) = 48)
  (h2 : ∃ y : ℕ, y + (y + 3) + (y + 6) = 60)
  (h3 : ∃ z : ℕ, z + (z + 4) = 30)
  (h4 : n = 8) :
  ∃ w : ℕ, (w = 13 ∧ w ≤ x ∧ w ≤ y ∧ w ≤ z) :=
by
  sorry

end youngest_child_age_l1999_199909


namespace school_gymnastics_ratio_l1999_199990

theorem school_gymnastics_ratio (total_students : ℕ) 
  (h_total : total_students = 120) : 
  ¬ ∃ (boys girls : ℕ), boys + girls = total_students ∧ 9 * girls = 2 * boys := by
  sorry

end school_gymnastics_ratio_l1999_199990


namespace max_time_digit_sum_l1999_199954

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours ≤ 23
  minute_valid : minutes ≤ 59

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a given time -/
def timeDigitSum (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits in a 24-hour digital watch display is 24 -/
theorem max_time_digit_sum : 
  ∃ (t : Time24), ∀ (t' : Time24), timeDigitSum t' ≤ timeDigitSum t ∧ timeDigitSum t = 24 :=
sorry

end max_time_digit_sum_l1999_199954


namespace max_pairs_correct_l1999_199933

def max_pairs (n : ℕ) : ℕ :=
  let k := (8037 : ℕ) / 5
  k

theorem max_pairs_correct (n : ℕ) (h : n = 4019) :
  ∀ (k : ℕ) (pairs : List (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2 ∧ p.1 ∈ Finset.range n ∧ p.2 ∈ Finset.range n) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
    pairs.length ≤ max_pairs n :=
by sorry

end max_pairs_correct_l1999_199933


namespace complex_circle_theorem_l1999_199958

def complex_circle_problem (a₁ a₂ a₃ a₄ a₅ : ℂ) (s : ℝ) : Prop :=
  (a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₃ ≠ 0 ∧ a₄ ≠ 0 ∧ a₅ ≠ 0) ∧
  (a₂ / a₁ = a₃ / a₂) ∧ (a₃ / a₂ = a₄ / a₃) ∧ (a₄ / a₃ = a₅ / a₄) ∧
  (a₁ + a₂ + a₃ + a₄ + a₅ = 4 * (1 / a₁ + 1 / a₂ + 1 / a₃ + 1 / a₄ + 1 / a₅)) ∧
  (a₁ + a₂ + a₃ + a₄ + a₅ = s) ∧
  (Complex.abs s ≤ 2) →
  Complex.abs a₁ = 2 ∧ Complex.abs a₂ = 2 ∧ Complex.abs a₃ = 2 ∧ Complex.abs a₄ = 2 ∧ Complex.abs a₅ = 2

theorem complex_circle_theorem (a₁ a₂ a₃ a₄ a₅ : ℂ) (s : ℝ) :
  complex_circle_problem a₁ a₂ a₃ a₄ a₅ s := by
  sorry

end complex_circle_theorem_l1999_199958


namespace john_initial_money_l1999_199939

theorem john_initial_money (spent : ℕ) (left : ℕ) : 
  left = 500 → 
  spent = left + 600 → 
  spent + left = 1600 :=
by
  sorry

end john_initial_money_l1999_199939


namespace smallest_four_digit_multiple_l1999_199914

theorem smallest_four_digit_multiple : ∃ (n : ℕ), 
  (n = 1119) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < n → ¬(((m + 1) % 5 = 0) ∧ ((m + 1) % 7 = 0) ∧ ((m + 1) % 8 = 0))) ∧
  ((n + 1) % 5 = 0) ∧ ((n + 1) % 7 = 0) ∧ ((n + 1) % 8 = 0) :=
by
  sorry

end smallest_four_digit_multiple_l1999_199914


namespace rabbit_carrot_consumption_l1999_199971

theorem rabbit_carrot_consumption :
  ∀ (rabbit_days deer_days : ℕ) (total_food : ℕ),
    rabbit_days = deer_days + 2 →
    5 * rabbit_days = total_food →
    6 * deer_days = total_food →
    5 * rabbit_days = 60 :=
by
  sorry

end rabbit_carrot_consumption_l1999_199971


namespace razorback_tshirt_revenue_l1999_199924

/-- The total money made by selling a given number of t-shirts at a fixed price -/
def total_money_made (num_shirts : ℕ) (price_per_shirt : ℕ) : ℕ :=
  num_shirts * price_per_shirt

/-- Theorem stating that selling 45 t-shirts at $16 each results in $720 total -/
theorem razorback_tshirt_revenue : total_money_made 45 16 = 720 := by
  sorry

end razorback_tshirt_revenue_l1999_199924


namespace geometric_sequence_property_l1999_199994

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem geometric_sequence_property 
  (b : ℕ → ℝ) (m n p : ℕ) 
  (h_geometric : GeometricSequence b)
  (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p)
  (h_positive : 0 < m ∧ 0 < n ∧ 0 < p) :
  (b p) ^ (m - n) * (b m) ^ (n - p) * (b n) ^ (p - m) = 1 := by
  sorry

end geometric_sequence_property_l1999_199994


namespace series_rationality_characterization_l1999_199955

/-- Represents a sequence of coefficients for the series -/
def CoefficientSequence := ℕ → ℕ

/-- The series sum for a given coefficient sequence -/
noncomputable def SeriesSum (a : CoefficientSequence) : ℝ :=
  ∑' n, (a n : ℝ) / n.factorial

/-- Condition that all coefficients from N onwards are zero -/
def AllZeroFrom (a : CoefficientSequence) (N : ℕ) : Prop :=
  ∀ n ≥ N, a n = 0

/-- Condition that all coefficients from N onwards are n-1 -/
def AllNMinusOneFrom (a : CoefficientSequence) (N : ℕ) : Prop :=
  ∀ n ≥ N, a n = n - 1

/-- The main theorem statement -/
theorem series_rationality_characterization (a : CoefficientSequence) 
  (h : ∀ n ≥ 2, 0 ≤ a n ∧ a n ≤ n - 1) :
  (∃ q : ℚ, SeriesSum a = q) ↔ 
  (∃ N : ℕ, AllZeroFrom a N ∨ AllNMinusOneFrom a N) := by
  sorry

end series_rationality_characterization_l1999_199955


namespace sons_age_l1999_199919

theorem sons_age (man daughter son : ℕ) 
  (h1 : man = son + 30)
  (h2 : daughter = son - 8)
  (h3 : man + 2 = 3 * (son + 2))
  (h4 : man + 2 = 2 * (daughter + 2)) :
  son = 13 := by
  sorry

end sons_age_l1999_199919


namespace direct_variation_problem_l1999_199983

/-- A function representing the relationship between x and y -/
def f (k : ℝ) (y : ℝ) : ℝ := k * y^2

theorem direct_variation_problem (k : ℝ) :
  f k 1 = 6 → f k 4 = 96 := by
  sorry

end direct_variation_problem_l1999_199983


namespace four_teacher_proctoring_l1999_199926

/-- Represents the number of teachers and classes -/
def n : ℕ := 4

/-- The number of ways to arrange n teachers to proctor n classes, where no teacher proctors their own class -/
def derangement (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of ways to arrange 4 teachers to proctor 4 classes, where no teacher proctors their own class, is equal to 9 -/
theorem four_teacher_proctoring : derangement n = 9 := by sorry

end four_teacher_proctoring_l1999_199926


namespace fish_population_estimate_l1999_199923

/-- Estimates the number of fish in a lake using the capture-recapture technique -/
theorem fish_population_estimate (tagged_april : ℕ) (captured_august : ℕ) (tagged_recaptured : ℕ)
  (tagged_survival_rate : ℝ) (original_fish_rate : ℝ) :
  tagged_april = 100 →
  captured_august = 100 →
  tagged_recaptured = 5 →
  tagged_survival_rate = 0.7 →
  original_fish_rate = 0.8 →
  ∃ (estimated_population : ℕ), estimated_population = 1120 :=
by sorry

end fish_population_estimate_l1999_199923


namespace no_prime_generating_pair_l1999_199906

theorem no_prime_generating_pair : 
  ¬ ∃ (a b : ℕ+), ∀ (p q : ℕ), 
    1000 < p ∧ 1000 < q ∧ 
    Nat.Prime p ∧ Nat.Prime q ∧ 
    p ≠ q → 
    Nat.Prime (a * p + b * q) := by
  sorry

end no_prime_generating_pair_l1999_199906


namespace negation_of_proposition_l1999_199927

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → 3 * x^2 - x - 2 > 0) ↔ 
  (∃ x : ℝ, x > 0 ∧ 3 * x^2 - x - 2 ≤ 0) := by
  sorry

end negation_of_proposition_l1999_199927


namespace shopkeeper_discount_l1999_199978

theorem shopkeeper_discount (cost_price : ℝ) (h_positive : cost_price > 0) : 
  let labeled_price := cost_price * (1 + 0.4)
  let selling_price := cost_price * (1 + 0.33)
  let discount := labeled_price - selling_price
  let discount_percentage := (discount / labeled_price) * 100
  discount_percentage = 5 := by
sorry

end shopkeeper_discount_l1999_199978


namespace interest_period_calculation_l1999_199985

theorem interest_period_calculation 
  (initial_amount : ℝ) 
  (rate_A : ℝ) 
  (rate_B : ℝ) 
  (gain_B : ℝ) 
  (h1 : initial_amount = 2800)
  (h2 : rate_A = 0.15)
  (h3 : rate_B = 0.185)
  (h4 : gain_B = 294) :
  ∃ t : ℝ, t = 3 ∧ initial_amount * (rate_B - rate_A) * t = gain_B :=
sorry

end interest_period_calculation_l1999_199985


namespace students_taking_no_subjects_l1999_199900

theorem students_taking_no_subjects (total : ℕ) (music art sports : ℕ) 
  (music_and_art music_and_sports art_and_sports : ℕ) (all_three : ℕ) : 
  total = 1200 →
  music = 60 →
  art = 80 →
  sports = 30 →
  music_and_art = 25 →
  music_and_sports = 15 →
  art_and_sports = 20 →
  all_three = 10 →
  total - (music + art + sports - music_and_art - music_and_sports - art_and_sports + all_three) = 1080 := by
  sorry

#check students_taking_no_subjects

end students_taking_no_subjects_l1999_199900


namespace sum_of_fractions_equals_one_l1999_199977

theorem sum_of_fractions_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_sum : a + b + c = 1) : 
  (a^2 * b^2 / ((a^2 + b*c) * (b^2 + a*c))) + 
  (a^2 * c^2 / ((a^2 + b*c) * (c^2 + a*b))) + 
  (b^2 * c^2 / ((b^2 + a*c) * (c^2 + a*b))) = 1 := by
  sorry

end sum_of_fractions_equals_one_l1999_199977


namespace employee_pay_l1999_199964

theorem employee_pay (total : ℝ) (ratio : ℝ) (lower_pay : ℝ) : 
  total = 580 →
  ratio = 1.5 →
  total = lower_pay + ratio * lower_pay →
  lower_pay = 232 := by
sorry

end employee_pay_l1999_199964


namespace percentage_equivalence_l1999_199912

theorem percentage_equivalence : 
  (75 / 100) * 600 = (50 / 100) * 900 := by sorry

end percentage_equivalence_l1999_199912


namespace essay_word_count_l1999_199949

theorem essay_word_count 
  (intro_length : ℕ) 
  (body_section_length : ℕ) 
  (num_body_sections : ℕ) 
  (h1 : intro_length = 450)
  (h2 : num_body_sections = 4)
  (h3 : body_section_length = 800) : 
  intro_length + 3 * intro_length + num_body_sections * body_section_length = 5000 :=
by sorry

end essay_word_count_l1999_199949


namespace absolute_value_equality_l1999_199915

theorem absolute_value_equality (y : ℝ) : |y| = |y - 3| → y = 3/2 := by
  sorry

end absolute_value_equality_l1999_199915


namespace selection_methods_l1999_199930

theorem selection_methods (female_students male_students : ℕ) 
  (h1 : female_students = 3) 
  (h2 : male_students = 2) : 
  female_students + male_students = 5 := by
  sorry

end selection_methods_l1999_199930


namespace tan_sum_simplification_l1999_199947

theorem tan_sum_simplification : 
  Real.tan (π / 8) + Real.tan (5 * π / 24) = 
    2 * Real.sin (13 * π / 24) / Real.sqrt ((2 + Real.sqrt 2) * (2 + Real.sqrt 3)) := by
  sorry

end tan_sum_simplification_l1999_199947


namespace f_integer_values_l1999_199948

def f (a b : ℕ+) : ℚ :=
  (a.val^2 + a.val * b.val + b.val^2) / (a.val * b.val - 1)

theorem f_integer_values (a b : ℕ+) (h : a.val * b.val ≠ 1) :
  ∃ (n : ℤ), n ∈ ({4, 7} : Set ℤ) ∧ f a b = n := by
  sorry

end f_integer_values_l1999_199948


namespace min_filtration_processes_correct_l1999_199976

/-- The reduction rate of impurities for each filtration process -/
def reduction_rate : ℝ := 0.20

/-- The target percentage of impurities after filtration -/
def target_percentage : ℝ := 0.05

/-- Approximation of log₂ -/
def log2_approx : ℝ := 0.3010

/-- The minimum number of filtration processes required -/
def min_filtration_processes : ℕ := 14

/-- Theorem stating the minimum number of filtration processes required -/
theorem min_filtration_processes_correct :
  ∀ n : ℕ,
  (1 - reduction_rate) ^ n < target_percentage →
  n ≥ min_filtration_processes :=
sorry

end min_filtration_processes_correct_l1999_199976


namespace dogwood_trees_in_park_l1999_199961

theorem dogwood_trees_in_park (current : ℕ) (planted : ℕ) (total : ℕ) : 
  planted = 49 → total = 83 → current + planted = total → current = 34 := by
  sorry

end dogwood_trees_in_park_l1999_199961


namespace basketball_players_l1999_199980

theorem basketball_players (total : ℕ) (hockey : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 25)
  (h2 : hockey = 15)
  (h3 : neither = 4)
  (h4 : both = 10) :
  ∃ basketball : ℕ, basketball = 16 :=
by sorry

end basketball_players_l1999_199980


namespace elderly_arrangement_count_l1999_199910

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem elderly_arrangement_count :
  let volunteers : ℕ := 5
  let elderly : ℕ := 2
  let total_units : ℕ := volunteers + 1  -- Treating elderly as one unit
  let total_arrangements : ℕ := factorial total_units * factorial elderly
  let end_arrangements : ℕ := 2 * factorial (total_units - 1) * factorial elderly
  total_arrangements - end_arrangements = 960 := by
  sorry

end elderly_arrangement_count_l1999_199910


namespace equal_color_distribution_l1999_199934

/-- The number of balls -/
def n : ℕ := 8

/-- The probability of a ball being painted black or white -/
def p : ℚ := 1/2

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of having exactly 4 black and 4 white balls -/
def prob_four_black_four_white : ℚ :=
  (choose n (n/2) : ℚ) * p^n

theorem equal_color_distribution :
  prob_four_black_four_white = 35/128 :=
sorry

end equal_color_distribution_l1999_199934


namespace a_decreasing_l1999_199965

open BigOperators

def a (n : ℕ) : ℚ := ∑ k in Finset.range n, 1 / (k * (n + 1 - k))

theorem a_decreasing (n : ℕ) (h : n ≥ 2) : a (n + 1) < a n := by
  sorry

end a_decreasing_l1999_199965


namespace consecutive_product_not_perfect_power_l1999_199908

theorem consecutive_product_not_perfect_power (n : ℕ) :
  ∀ m : ℕ, m ≥ 2 → ¬∃ k : ℕ, (n - 1) * n * (n + 1) = k^m :=
by sorry

end consecutive_product_not_perfect_power_l1999_199908


namespace problem_statement_l1999_199993

theorem problem_statement (a b c : ℝ) 
  (h1 : a^2 - b^2 = 10) 
  (h2 : a * b = 5) 
  (h3 : a^2 + b^2 + c^2 = 20) : 
  a^4 + b^4 + c^4 = 150 := by
  sorry

end problem_statement_l1999_199993


namespace evelyns_remaining_bottle_caps_l1999_199903

/-- The number of bottle caps Evelyn has left after losing some -/
def bottle_caps_left (initial : ℝ) (lost : ℝ) : ℝ := initial - lost

/-- Theorem: Evelyn's remaining bottle caps -/
theorem evelyns_remaining_bottle_caps :
  bottle_caps_left 63.75 18.36 = 45.39 := by
  sorry

end evelyns_remaining_bottle_caps_l1999_199903


namespace sequence_integer_count_l1999_199991

def sequence_term (n : ℕ) : ℚ :=
  9720 / 3^n

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem sequence_integer_count :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    ¬is_integer (sequence_term k)) →
  (∃! (k : ℕ), k = 6 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    ¬is_integer (sequence_term k)) :=
by sorry

end sequence_integer_count_l1999_199991


namespace square_equation_solution_l1999_199996

theorem square_equation_solution : ∃ (M : ℕ), M > 0 ∧ 12^2 * 30^2 = 15^2 * M^2 ∧ M = 24 := by sorry

end square_equation_solution_l1999_199996


namespace fraction_equation_solution_l1999_199952

theorem fraction_equation_solution : ∃ x : ℚ, (1 / 3 - 1 / 4 : ℚ) = 1 / x ∧ x = 12 := by
  sorry

end fraction_equation_solution_l1999_199952


namespace trig_expression_equals_sqrt2_over_2_l1999_199938

theorem trig_expression_equals_sqrt2_over_2 :
  (Real.sin (20 * π / 180)) * Real.sqrt (1 + Real.cos (40 * π / 180)) / (Real.cos (50 * π / 180)) = Real.sqrt 2 / 2 := by
  sorry

end trig_expression_equals_sqrt2_over_2_l1999_199938


namespace unique_multiplication_with_repeated_digit_l1999_199913

theorem unique_multiplication_with_repeated_digit :
  ∃! (a b c d e f g h i j z : ℕ),
    (0 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧
    (0 ≤ d ∧ d ≤ 9) ∧ (0 ≤ e ∧ e ≤ 9) ∧ (0 ≤ f ∧ f ≤ 9) ∧
    (0 ≤ g ∧ g ≤ 9) ∧ (0 ≤ h ∧ h ≤ 9) ∧ (0 ≤ i ∧ i ≤ 9) ∧
    (0 ≤ j ∧ j ≤ 9) ∧ (0 ≤ z ∧ z ≤ 9) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
    h ≠ i ∧ h ≠ j ∧
    i ≠ j ∧
    (a * 1000000 + b * 100000 + z * 10000 + c * 1000 + d * 100 + e * 10 + z) *
    (f * 100000 + g * 10000 + h * 1000 + i * 100 + z * 10 + j) =
    423416204528 :=
by sorry

end unique_multiplication_with_repeated_digit_l1999_199913


namespace polynomial_evaluation_l1999_199972

theorem polynomial_evaluation :
  let f : ℝ → ℝ := λ x ↦ x^4 + x^3 + x^2 + x + 2
  f 2 = 32 := by sorry

end polynomial_evaluation_l1999_199972


namespace president_and_vice_president_choices_l1999_199928

/-- The number of ways to choose a President and a Vice-President from a group of people -/
def choosePresidentAndVicePresident (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: There are 20 ways to choose a President and a Vice-President from a group of 5 people -/
theorem president_and_vice_president_choices :
  choosePresidentAndVicePresident 5 = 20 := by
  sorry

#eval choosePresidentAndVicePresident 5

end president_and_vice_president_choices_l1999_199928


namespace machine_parts_replacement_l1999_199956

theorem machine_parts_replacement (num_machines : ℕ) (parts_per_machine : ℕ)
  (fail_rate_week1 : ℚ) (fail_rate_week2 : ℚ) (fail_rate_week3 : ℚ) :
  num_machines = 500 →
  parts_per_machine = 6 →
  fail_rate_week1 = 1/10 →
  fail_rate_week2 = 3/10 →
  fail_rate_week3 = 6/10 →
  (fail_rate_week1 + fail_rate_week2 + fail_rate_week3 = 1) →
  (num_machines * parts_per_machine * fail_rate_week3 +
   (num_machines * parts_per_machine * fail_rate_week2 * fail_rate_week3) +
   (num_machines * parts_per_machine * fail_rate_week1 * fail_rate_week2 * fail_rate_week3) : ℚ) = 1983 := by
  sorry


end machine_parts_replacement_l1999_199956


namespace tangent_line_and_monotonicity_l1999_199982

noncomputable def f (a b x : ℝ) : ℝ := (1/2) * a * x^2 - (a^2 + b) * x + a * Real.log x

theorem tangent_line_and_monotonicity :
  ∀ (a b : ℝ),
  (a = -1 ∧ b = 0 →
    ∃ (m c : ℝ), m = -3 ∧ c = -3/2 ∧
    ∀ x y, y = m * (x - 1) + c ↔ 6 * x + 2 * y - 3 = 0) ∧
  (b = 1 →
    ((a ≤ 0 → ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a b x₁ > f a b x₂) ∧
    (a > 1 →
      (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/a → f a b x₁ < f a b x₂) ∧
      (∀ x₁ x₂, 1/a < x₁ ∧ x₁ < x₂ ∧ x₂ < a → f a b x₁ > f a b x₂) ∧
      (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f a b x₁ < f a b x₂)))) :=
by sorry

end tangent_line_and_monotonicity_l1999_199982


namespace foreign_language_ratio_l1999_199995

theorem foreign_language_ratio (M F : ℕ) (h1 : M > 0) (h2 : F > 0) : 
  (3 * M + 4 * F : ℚ) / (5 * M + 6 * F) = 19 / 30 → M = F :=
by sorry

end foreign_language_ratio_l1999_199995


namespace hyperbola_asymptotes_l1999_199974

theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 4 - y^2 / 9 = 0) ↔ (y = (3/2) * x ∨ y = -(3/2) * x) := by
  sorry

end hyperbola_asymptotes_l1999_199974


namespace inscribed_square_diagonal_l1999_199941

theorem inscribed_square_diagonal (length width : ℝ) (h1 : length = 8) (h2 : width = 6) :
  let inscribed_square_side := width
  let inscribed_square_area := inscribed_square_side ^ 2
  let third_square_area := 9 * inscribed_square_area
  let third_square_side := Real.sqrt third_square_area
  let third_square_diagonal := third_square_side * Real.sqrt 2
  third_square_diagonal = 18 * Real.sqrt 2 := by
  sorry

end inscribed_square_diagonal_l1999_199941


namespace unique_prime_in_range_l1999_199940

theorem unique_prime_in_range : 
  ∃! n : ℕ, 30 < n ∧ n ≤ 43 ∧ Prime n ∧ n % 9 = 7 :=
by
  sorry

end unique_prime_in_range_l1999_199940


namespace quadratic_factorization_l1999_199925

theorem quadratic_factorization (y : ℝ) (a b : ℤ) 
  (h : ∀ y, 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) : 
  a - b = -7 := by
sorry

end quadratic_factorization_l1999_199925


namespace weight_2019_is_9_5_l1999_199959

/-- The weight of a single stick in kilograms -/
def stick_weight : ℝ := 0.5

/-- The number of sticks used to form each digit -/
def sticks_per_digit : Fin 10 → ℕ
  | 0 => 6
  | 1 => 2
  | 2 => 5
  | 9 => 6
  | _ => 0  -- We only care about digits 0, 1, 2, and 9 for this problem

/-- The weight of the number 2019 in kilograms -/
def weight_2019 : ℝ :=
  (sticks_per_digit 2 + sticks_per_digit 0 + sticks_per_digit 1 + sticks_per_digit 9) * stick_weight

/-- The theorem stating that the weight of 2019 is 9.5 kg -/
theorem weight_2019_is_9_5 : weight_2019 = 9.5 := by
  sorry

#eval weight_2019

end weight_2019_is_9_5_l1999_199959


namespace linear_function_condition_passes_through_origin_l1999_199984

/-- A linear function of x with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := (2*m + 1)*x + m - 3

theorem linear_function_condition (m : ℝ) :
  (∀ x, ∃ y, f m x = y) ↔ m ≠ -1/2 :=
sorry

theorem passes_through_origin (m : ℝ) :
  f m 0 = 0 ↔ m = 3 :=
sorry

end linear_function_condition_passes_through_origin_l1999_199984


namespace prob_first_second_win_eq_three_tenths_l1999_199944

/-- Represents a lottery with winning and non-winning tickets -/
structure Lottery where
  total_tickets : ℕ
  winning_tickets : ℕ
  people : ℕ
  h_winning_le_total : winning_tickets ≤ total_tickets
  h_people_le_total : people ≤ total_tickets

/-- The probability of drawing a winning ticket -/
def prob_win (L : Lottery) : ℚ :=
  L.winning_tickets / L.total_tickets

/-- The probability of both the first and second person drawing a winning ticket -/
def prob_first_second_win (L : Lottery) : ℚ :=
  (L.winning_tickets / L.total_tickets) * ((L.winning_tickets - 1) / (L.total_tickets - 1))

/-- Theorem stating the probability of both first and second person drawing a winning ticket -/
theorem prob_first_second_win_eq_three_tenths (L : Lottery) 
    (h_total : L.total_tickets = 5)
    (h_winning : L.winning_tickets = 3)
    (h_people : L.people = 5) :
    prob_first_second_win L = 3 / 10 := by
  sorry


end prob_first_second_win_eq_three_tenths_l1999_199944


namespace sum_of_extrema_l1999_199966

theorem sum_of_extrema (a b c : ℝ) (h1 : a + b + c = 5) (h2 : a^2 + b^2 + c^2 = 7) : 
  ∃ (n N : ℝ), (∀ x, (∃ y z, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 7) → n ≤ x ∧ x ≤ N) ∧ n + N = 10/3 :=
by sorry

end sum_of_extrema_l1999_199966


namespace vector_dot_product_theorem_l1999_199921

def orthogonal_unit_vectors (i j : ℝ × ℝ) : Prop :=
  i.1 * j.1 + i.2 * j.2 = 0 ∧ i.1^2 + i.2^2 = 1 ∧ j.1^2 + j.2^2 = 1

def vector_a (i j : ℝ × ℝ) : ℝ × ℝ := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)

def vector_b (i j : ℝ × ℝ) (m : ℝ) : ℝ × ℝ := (i.1 - m * j.1, i.2 - m * j.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_theorem (i j : ℝ × ℝ) (m : ℝ) :
  orthogonal_unit_vectors i j →
  dot_product (vector_a i j) (vector_b i j m) = 1 →
  m = 1/3 := by sorry

end vector_dot_product_theorem_l1999_199921


namespace total_tires_in_parking_lot_l1999_199946

def num_cars : ℕ := 30
def regular_tires_per_car : ℕ := 4
def spare_tires_per_car : ℕ := 1

theorem total_tires_in_parking_lot :
  (num_cars * (regular_tires_per_car + spare_tires_per_car)) = 150 := by
  sorry

end total_tires_in_parking_lot_l1999_199946
