import Mathlib

namespace NUMINAMATH_CALUDE_first_player_wins_l3365_336500

/-- Represents a game played on a regular polygon -/
structure PolygonGame where
  sides : ℕ
  is_regular : sides > 2

/-- Represents a move in the game -/
inductive Move
| connect (v1 v2 : ℕ)

/-- Represents the state of the game -/
structure GameState where
  game : PolygonGame
  moves : List Move

/-- Checks if a move is valid given the current game state -/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  sorry

/-- Checks if the game is over (no more valid moves) -/
def is_game_over (state : GameState) : Prop :=
  sorry

/-- Determines the winner of the game -/
def winner (state : GameState) : Option Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- Checks if a strategy is winning for the first player -/
def is_winning_strategy (strat : Strategy) (game : PolygonGame) : Prop :=
  sorry

/-- The main theorem: there exists a winning strategy for the first player in a 1968-sided polygon game -/
theorem first_player_wins :
  ∃ (strat : Strategy), is_winning_strategy strat ⟨1968, by norm_num⟩ :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l3365_336500


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3365_336505

/-- 
Given an ellipse b^2 x^2 + a^2 y^2 = a^2 b^2 and a line y = px + q,
this theorem states the condition for the line to be tangent to the ellipse
and provides the coordinates of the tangency point.
-/
theorem line_tangent_to_ellipse 
  (a b p q : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hq : q ≠ 0) :
  (∀ x y : ℝ, b^2 * x^2 + a^2 * y^2 = a^2 * b^2 ∧ y = p * x + q) →
  (a^2 * p^2 + b^2 = q^2 ∧ 
   ∃ x y : ℝ, x = -a^2 * p / q ∧ y = b^2 / q ∧ 
   b^2 * x^2 + a^2 * y^2 = a^2 * b^2 ∧ y = p * x + q) :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3365_336505


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l3365_336542

theorem rectangular_field_dimensions : ∃ m : ℝ, m > 3 ∧ (3*m + 8)*(m - 3) = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l3365_336542


namespace NUMINAMATH_CALUDE_factor_implies_s_value_l3365_336543

theorem factor_implies_s_value (m s : ℤ) : 
  (∃ k : ℤ, m^2 - s*m - 24 = (m - 8) * k) → s = 5 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_s_value_l3365_336543


namespace NUMINAMATH_CALUDE_alpha_is_two_thirds_l3365_336586

theorem alpha_is_two_thirds (α : ℚ) (h1 : 0 < α) (h2 : α < 1) 
  (h3 : Real.cos (3 * Real.pi * α) + 2 * Real.cos (2 * Real.pi * α) = 0) : 
  α = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_alpha_is_two_thirds_l3365_336586


namespace NUMINAMATH_CALUDE_S_31_primes_less_than_20000_l3365_336546

/-- Sum of digits in base k -/
def S (k : ℕ) (n : ℕ) : ℕ := sorry

/-- The theorem to be proved -/
theorem S_31_primes_less_than_20000 (p : ℕ) (h_prime : Nat.Prime p) (h_bound : p < 20000) :
  S 31 p = 49 ∨ S 31 p = 77 := by sorry

end NUMINAMATH_CALUDE_S_31_primes_less_than_20000_l3365_336546


namespace NUMINAMATH_CALUDE_partnership_profit_share_l3365_336530

/-- Given a partnership with three investors A, B, and C, where A invests 3 times as much as B
    and 2/3 of what C invests, prove that C's share of a total profit of 11000 is (9/17) * 11000. -/
theorem partnership_profit_share (a b c : ℝ) (profit : ℝ) : 
  a = 3 * b → 
  a = (2/3) * c → 
  profit = 11000 → 
  c * profit / (a + b + c) = (9/17) * 11000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_share_l3365_336530


namespace NUMINAMATH_CALUDE_product_xyz_is_negative_one_l3365_336525

theorem product_xyz_is_negative_one 
  (x y z : ℝ) 
  (h1 : x + 1/y = 1) 
  (h2 : y + 1/z = 1) : 
  x * y * z = -1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_is_negative_one_l3365_336525


namespace NUMINAMATH_CALUDE_hammond_discarded_marble_l3365_336547

/-- The weight of discarded marble after carving statues -/
def discarded_marble (initial_block : ℕ) (statue1 statue2 statue3 statue4 : ℕ) : ℕ :=
  initial_block - (statue1 + statue2 + statue3 + statue4)

/-- Theorem stating the amount of discarded marble for Hammond's statues -/
theorem hammond_discarded_marble :
  discarded_marble 80 10 18 15 15 = 22 := by
  sorry

end NUMINAMATH_CALUDE_hammond_discarded_marble_l3365_336547


namespace NUMINAMATH_CALUDE_rooster_count_l3365_336576

theorem rooster_count (total_birds : ℕ) (rooster_ratio hen_ratio chick_ratio duck_ratio goose_ratio : ℕ) 
  (h1 : total_birds = 9000)
  (h2 : rooster_ratio = 4)
  (h3 : hen_ratio = 2)
  (h4 : chick_ratio = 6)
  (h5 : duck_ratio = 3)
  (h6 : goose_ratio = 1) :
  (total_birds * rooster_ratio) / (rooster_ratio + hen_ratio + chick_ratio + duck_ratio + goose_ratio) = 2250 := by
  sorry

end NUMINAMATH_CALUDE_rooster_count_l3365_336576


namespace NUMINAMATH_CALUDE_tiles_required_to_cover_floor_l3365_336553

-- Define the dimensions
def floor_length : ℚ := 10
def floor_width : ℚ := 15
def tile_length : ℚ := 5 / 12  -- 5 inches in feet
def tile_width : ℚ := 2 / 3    -- 8 inches in feet

-- Theorem statement
theorem tiles_required_to_cover_floor :
  (floor_length * floor_width) / (tile_length * tile_width) = 540 := by
  sorry

end NUMINAMATH_CALUDE_tiles_required_to_cover_floor_l3365_336553


namespace NUMINAMATH_CALUDE_problem_proof_l3365_336559

theorem problem_proof : |Real.sqrt 3 - 2| - Real.sqrt ((-3)^2) + 2 * Real.sqrt 9 = 5 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l3365_336559


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3365_336541

def M : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}
def N : Set ℝ := {x : ℝ | x ≤ -3}

theorem union_of_M_and_N : M ∪ N = {x : ℝ | x < 1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3365_336541


namespace NUMINAMATH_CALUDE_parallelepiped_plane_ratio_l3365_336587

/-- A parallelepiped in 3D space -/
structure Parallelepiped where
  -- Add necessary fields here
  mk :: -- Constructor

/-- The number of distinct planes passing through any three vertices of a parallelepiped -/
def num_distinct_planes (V : Parallelepiped) : ℕ := sorry

/-- The number of planes that bisect the volume of a parallelepiped -/
def num_bisecting_planes (V : Parallelepiped) : ℕ := sorry

/-- Theorem stating the ratio of bisecting planes to total distinct planes -/
theorem parallelepiped_plane_ratio (V : Parallelepiped) : 
  (num_bisecting_planes V : ℚ) / (num_distinct_planes V : ℚ) = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_plane_ratio_l3365_336587


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3365_336521

theorem point_in_second_quadrant (a : ℤ) : 
  (2*a + 1 < 0) ∧ (2 + a > 0) → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3365_336521


namespace NUMINAMATH_CALUDE_marble_weight_l3365_336575

theorem marble_weight (marble_weight : ℚ) (car_weight : ℚ) : 
  (9 * marble_weight = 4 * car_weight) →
  (3 * car_weight = 36) →
  marble_weight = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_marble_weight_l3365_336575


namespace NUMINAMATH_CALUDE_valid_three_digit_count_l3365_336510

/-- The count of three-digit numbers without exactly two identical adjacent digits -/
def valid_three_digit_numbers : ℕ := 738

/-- The total count of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers with exactly two identical adjacent digits -/
def invalid_three_digit_numbers : ℕ := 162

theorem valid_three_digit_count :
  valid_three_digit_numbers = total_three_digit_numbers - invalid_three_digit_numbers :=
by sorry

end NUMINAMATH_CALUDE_valid_three_digit_count_l3365_336510


namespace NUMINAMATH_CALUDE_oil_price_reduction_l3365_336593

theorem oil_price_reduction (original_price : ℝ) : 
  (∃ (quantity : ℝ), 
    original_price * quantity = 800 ∧ 
    (0.75 * original_price) * (quantity + 5) = 800) →
  0.75 * original_price = 30 := by
sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l3365_336593


namespace NUMINAMATH_CALUDE_marble_distribution_l3365_336549

theorem marble_distribution (total_marbles : ℕ) (group_size : ℕ) : 
  total_marbles = 364 →
  (total_marbles / group_size : ℚ) - (total_marbles / (group_size + 2) : ℚ) = 1 →
  group_size = 26 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l3365_336549


namespace NUMINAMATH_CALUDE_canoe_downstream_speed_l3365_336526

/-- The speed of a canoe rowing downstream, given its upstream speed against a stream -/
theorem canoe_downstream_speed
  (upstream_speed : ℝ)
  (stream_speed : ℝ)
  (h1 : upstream_speed = 4)
  (h2 : stream_speed = 4) :
  upstream_speed + 2 * stream_speed = 12 :=
by sorry

end NUMINAMATH_CALUDE_canoe_downstream_speed_l3365_336526


namespace NUMINAMATH_CALUDE_peters_extra_pictures_l3365_336556

theorem peters_extra_pictures (randy_pictures : ℕ) (peter_pictures : ℕ) (quincy_pictures : ℕ) :
  randy_pictures = 5 →
  quincy_pictures = peter_pictures + 20 →
  randy_pictures + peter_pictures + quincy_pictures = 41 →
  peter_pictures - randy_pictures = 3 := by
  sorry

end NUMINAMATH_CALUDE_peters_extra_pictures_l3365_336556


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_pow_2018_l3365_336509

theorem last_two_digits_of_7_pow_2018 : 7^2018 % 100 = 49 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_7_pow_2018_l3365_336509


namespace NUMINAMATH_CALUDE_fraction_transformation_l3365_336515

theorem fraction_transformation (n : ℚ) : (4 + n) / (7 + n) = 2 / 3 → n = 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_transformation_l3365_336515


namespace NUMINAMATH_CALUDE_sphere_packing_radius_l3365_336552

/-- A configuration of spheres packed in a cube. -/
structure SpherePacking where
  cube_side : ℝ
  num_spheres : ℕ
  sphere_radius : ℝ

/-- The specific sphere packing configuration described in the problem. -/
def problem_packing : SpherePacking where
  cube_side := 2
  num_spheres := 16
  sphere_radius := 1  -- This is what we want to prove

/-- Predicate to check if a sphere packing configuration is valid according to the problem description. -/
def is_valid_packing (p : SpherePacking) : Prop :=
  p.cube_side = 2 ∧
  p.num_spheres = 16 ∧
  -- One sphere at the center, others tangent to it and three faces
  2 * p.sphere_radius = p.cube_side / 2

theorem sphere_packing_radius : 
  is_valid_packing problem_packing ∧ 
  problem_packing.sphere_radius = 1 :=
by sorry

end NUMINAMATH_CALUDE_sphere_packing_radius_l3365_336552


namespace NUMINAMATH_CALUDE_errand_time_is_110_minutes_l3365_336583

def driving_time_one_way : ℕ := 20
def parent_teacher_night_time : ℕ := 70

def total_errand_time : ℕ :=
  2 * driving_time_one_way + parent_teacher_night_time

theorem errand_time_is_110_minutes :
  total_errand_time = 110 := by
  sorry

end NUMINAMATH_CALUDE_errand_time_is_110_minutes_l3365_336583


namespace NUMINAMATH_CALUDE_ninas_toys_l3365_336581

theorem ninas_toys (toy_price : ℕ) (card_packs : ℕ) (card_price : ℕ) (shirts : ℕ) (shirt_price : ℕ) (total_spent : ℕ) : 
  toy_price = 10 →
  card_packs = 2 →
  card_price = 5 →
  shirts = 5 →
  shirt_price = 6 →
  total_spent = 70 →
  ∃ (num_toys : ℕ), num_toys * toy_price + card_packs * card_price + shirts * shirt_price = total_spent ∧ num_toys = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ninas_toys_l3365_336581


namespace NUMINAMATH_CALUDE_ellipse_circle_intersection_l3365_336551

/-- The ellipse C defined by x^2 + 16y^2 = 16 -/
def ellipse_C (x y : ℝ) : Prop := x^2 + 16 * y^2 = 16

/-- The circle Γ with center (0, h) and radius r -/
def circle_Γ (h r : ℝ) (x y : ℝ) : Prop := x^2 + (y - h)^2 = r^2

/-- The foci of ellipse C -/
def foci : Set (ℝ × ℝ) := {(-Real.sqrt 15, 0), (Real.sqrt 15, 0)}

theorem ellipse_circle_intersection (a b : ℝ) :
  (∃ r h, r ∈ Set.Icc a b ∧
    (∀ (f : ℝ × ℝ), f ∈ foci → circle_Γ h r f.1 f.2) ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄,
      ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧ ellipse_C x₃ y₃ ∧ ellipse_C x₄ y₄ ∧
      circle_Γ h r x₁ y₁ ∧ circle_Γ h r x₂ y₂ ∧ circle_Γ h r x₃ y₃ ∧ circle_Γ h r x₄ y₄ ∧
      (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
      (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧ (x₃, y₃) ≠ (x₄, y₄))) →
  a + b = Real.sqrt 15 + 8 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_circle_intersection_l3365_336551


namespace NUMINAMATH_CALUDE_right_triangle_construction_l3365_336516

/-- Given a length b (representing one leg) and a length c (representing the projection of the other leg onto the hypotenuse), a right triangle can be constructed. -/
theorem right_triangle_construction (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  ∃ (a x : ℝ), a > 0 ∧ x > 0 ∧ x + c = a ∧ b^2 = x * a :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_construction_l3365_336516


namespace NUMINAMATH_CALUDE_faster_train_speed_l3365_336529

-- Define the parameters
def train_length : ℝ := 500  -- in meters
def slower_train_speed : ℝ := 30  -- in km/hr
def passing_time : ℝ := 47.99616030717543  -- in seconds

-- Define the theorem
theorem faster_train_speed :
  ∃ (faster_speed : ℝ),
    faster_speed > slower_train_speed ∧
    faster_speed = 45 ∧
    (faster_speed + slower_train_speed) * (passing_time / 3600) = 2 * train_length / 1000 :=
by sorry

end NUMINAMATH_CALUDE_faster_train_speed_l3365_336529


namespace NUMINAMATH_CALUDE_equation_solution_l3365_336533

theorem equation_solution : ∃ y : ℚ, (3 * (y + 1) / 4 - (1 - y) / 8 = 1) ∧ (y = 3 / 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3365_336533


namespace NUMINAMATH_CALUDE_no_common_solution_l3365_336579

theorem no_common_solution : ¬∃ (x y : ℝ), x^2 + y^2 = 25 ∧ x^2 + 3*y = 45 := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l3365_336579


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3365_336540

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  a 5 + a 8 = 5 →                                   -- given condition
  a 2 + a 11 = 5 :=                                 -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3365_336540


namespace NUMINAMATH_CALUDE_equation_root_l3365_336567

theorem equation_root (m : ℝ) : 
  (∃ x : ℝ, x^2 + 5*x + m = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 + 5*y + m = 0 ∧ y = -4) :=
sorry

end NUMINAMATH_CALUDE_equation_root_l3365_336567


namespace NUMINAMATH_CALUDE_angle_457_properties_l3365_336562

-- Define the set of angles with the same terminal side as -457°
def same_terminal_side (β : ℝ) : Prop :=
  ∃ k : ℤ, β = k * 360 - 457

-- Define the third quadrant
def third_quadrant (θ : ℝ) : Prop :=
  180 < θ % 360 ∧ θ % 360 < 270

-- Theorem statement
theorem angle_457_properties :
  (∀ β, same_terminal_side β ↔ ∃ k : ℤ, β = k * 360 - 457) ∧
  third_quadrant (-457) := by
  sorry

end NUMINAMATH_CALUDE_angle_457_properties_l3365_336562


namespace NUMINAMATH_CALUDE_lcm_gcd_relation_l3365_336536

theorem lcm_gcd_relation (a b : ℕ) : 
  (Nat.lcm a b + Nat.gcd a b = a * b / 5) ↔ 
  ((a = 10 ∧ b = 10) ∨ (a = 6 ∧ b = 30) ∨ (a = 30 ∧ b = 6)) :=
sorry

end NUMINAMATH_CALUDE_lcm_gcd_relation_l3365_336536


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_exist_l3365_336548

/-- Definition of triangular numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating the existence of infinitely many pairs (a, b) satisfying the property -/
theorem infinitely_many_pairs_exist :
  ∃ f : ℕ → ℕ × ℕ, ∀ k : ℕ,
    let (a, b) := f k
    ∀ n : ℕ, (∃ m : ℕ, a * triangular_number n + b = triangular_number m) ↔
              (∃ l : ℕ, triangular_number n = triangular_number l) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_exist_l3365_336548


namespace NUMINAMATH_CALUDE_balloon_arrangements_l3365_336591

def word_length : ℕ := 7
def repeated_letters : ℕ := 2
def single_letters : ℕ := 3

theorem balloon_arrangements :
  (word_length.factorial) / 
  ((2 : ℕ).factorial * (2 : ℕ).factorial * 
   (1 : ℕ).factorial * (1 : ℕ).factorial * (1 : ℕ).factorial) = 1260 :=
by sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l3365_336591


namespace NUMINAMATH_CALUDE_boys_passed_exam_l3365_336520

/-- Proves the number of boys who passed an examination given specific conditions -/
theorem boys_passed_exam (total_boys : ℕ) (overall_avg : ℚ) (pass_avg : ℚ) (fail_avg : ℚ) :
  total_boys = 120 →
  overall_avg = 36 →
  pass_avg = 39 →
  fail_avg = 15 →
  ∃ (passed_boys : ℕ),
    passed_boys = 105 ∧
    passed_boys ≤ total_boys ∧
    (passed_boys : ℚ) * pass_avg + (total_boys - passed_boys : ℚ) * fail_avg = (total_boys : ℚ) * overall_avg :=
by sorry

end NUMINAMATH_CALUDE_boys_passed_exam_l3365_336520


namespace NUMINAMATH_CALUDE_mikes_apples_l3365_336577

theorem mikes_apples (nancy_apples keith_ate_apples apples_left : ℝ) 
  (h1 : nancy_apples = 3.0)
  (h2 : keith_ate_apples = 6.0)
  (h3 : apples_left = 4.0) :
  ∃ mike_apples : ℝ, mike_apples = 7.0 ∧ mike_apples + nancy_apples - keith_ate_apples = apples_left :=
by sorry

end NUMINAMATH_CALUDE_mikes_apples_l3365_336577


namespace NUMINAMATH_CALUDE_ways_to_express_114_l3365_336599

/-- Represents the number of ways to express a given number as the sum of ones and threes with a minimum number of ones -/
def waysToExpress (total : ℕ) (minOnes : ℕ) : ℕ :=
  (total - minOnes) / 3 + 1

/-- The theorem stating that there are 35 ways to express 114 as the sum of ones and threes with at least 10 ones -/
theorem ways_to_express_114 : waysToExpress 114 10 = 35 := by
  sorry

#eval waysToExpress 114 10

end NUMINAMATH_CALUDE_ways_to_express_114_l3365_336599


namespace NUMINAMATH_CALUDE_total_games_is_295_l3365_336573

/-- The number of games won by the Chicago Bulls -/
def bulls_games : ℕ := 70

/-- The number of games won by the Miami Heat -/
def heat_games : ℕ := bulls_games + 5

/-- The number of games won by the New York Knicks -/
def knicks_games : ℕ := 2 * heat_games

/-- The total number of games won by all three teams -/
def total_games : ℕ := bulls_games + heat_games + knicks_games

theorem total_games_is_295 : total_games = 295 := by sorry

end NUMINAMATH_CALUDE_total_games_is_295_l3365_336573


namespace NUMINAMATH_CALUDE_power_expression_evaluation_l3365_336527

theorem power_expression_evaluation (b : ℕ) (h : b = 4) : b^3 * b^6 / b^2 = 16384 := by
  sorry

end NUMINAMATH_CALUDE_power_expression_evaluation_l3365_336527


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l3365_336566

theorem rectangular_prism_volume (a b c : ℕ) : 
  a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 →
  2 * ((a - 2) * (b - 2) + (b - 2) * (c - 2) + (a - 2) * (c - 2)) = 24 →
  4 * ((a - 2) + (b - 2) + (c - 2)) = 28 →
  a * b * c = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l3365_336566


namespace NUMINAMATH_CALUDE_lukas_average_points_l3365_336568

/-- Lukas's average points per game in basketball -/
def average_points (total_points : ℕ) (num_games : ℕ) : ℚ :=
  (total_points : ℚ) / (num_games : ℚ)

/-- Theorem: Lukas averages 12 points per game -/
theorem lukas_average_points :
  average_points 60 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_lukas_average_points_l3365_336568


namespace NUMINAMATH_CALUDE_math_problems_l3365_336538

theorem math_problems :
  (32 * 3 = 96) ∧
  (43 / 9 = 4 ∧ 43 % 9 = 7) ∧
  (630 / 9 = 70) ∧
  (125 * 47 * 8 = 125 * 8 * 47) := by
  sorry

end NUMINAMATH_CALUDE_math_problems_l3365_336538


namespace NUMINAMATH_CALUDE_ellipse_max_value_l3365_336524

theorem ellipse_max_value (x y : ℝ) : 
  x^2 + 4*y^2 = 4 → 
  ∃ (M : ℝ), M = 7 ∧ ∀ (a b : ℝ), a^2 + 4*b^2 = 4 → (3/4)*a^2 + 2*a - b^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ellipse_max_value_l3365_336524


namespace NUMINAMATH_CALUDE_ordered_pairs_satisfying_equation_l3365_336595

theorem ordered_pairs_satisfying_equation : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      let a := p.1
      let b := p.2
      a > 0 ∧ b > 0 ∧ 
      a * b + 83 = 24 * Nat.lcm a b + 17 * Nat.gcd a b)
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_ordered_pairs_satisfying_equation_l3365_336595


namespace NUMINAMATH_CALUDE_twenty_multi_painted_cubes_l3365_336522

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : ℕ
  top_painted : Bool
  sides_painted : Bool
  bottom_painted : Bool

/-- Counts the number of unit cubes with at least two painted faces -/
def count_multi_painted_cubes (cube : PaintedCube) : ℕ :=
  sorry

/-- The main theorem -/
theorem twenty_multi_painted_cubes :
  let cube : PaintedCube := {
    size := 4,
    top_painted := true,
    sides_painted := true,
    bottom_painted := false
  }
  count_multi_painted_cubes cube = 20 := by
  sorry

end NUMINAMATH_CALUDE_twenty_multi_painted_cubes_l3365_336522


namespace NUMINAMATH_CALUDE_rhinestone_problem_l3365_336513

theorem rhinestone_problem (total : ℕ) (bought_fraction : ℚ) (found_fraction : ℚ) : 
  total = 45 → 
  bought_fraction = 1/3 → 
  found_fraction = 1/5 → 
  total - (total * bought_fraction).floor - (total * found_fraction).floor = 21 := by
  sorry

end NUMINAMATH_CALUDE_rhinestone_problem_l3365_336513


namespace NUMINAMATH_CALUDE_sum_of_ages_proof_l3365_336503

/-- Proves that the sum of ages of a mother and daughter is 70 years,
    given the daughter's age and the age difference. -/
theorem sum_of_ages_proof (daughter_age mother_daughter_diff : ℕ) : 
  daughter_age = 19 →
  mother_daughter_diff = 32 →
  daughter_age + (daughter_age + mother_daughter_diff) = 70 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_proof_l3365_336503


namespace NUMINAMATH_CALUDE_family_spent_36_dollars_l3365_336550

/-- The cost of a movie ticket in dollars -/
def ticket_cost : ℚ := 5

/-- The cost of popcorn as a fraction of the ticket cost -/
def popcorn_ratio : ℚ := 4/5

/-- The cost of soda as a fraction of the popcorn cost -/
def soda_ratio : ℚ := 1/2

/-- The number of tickets bought -/
def num_tickets : ℕ := 4

/-- The number of popcorn sets bought -/
def num_popcorn : ℕ := 2

/-- The number of soda cans bought -/
def num_soda : ℕ := 4

/-- Theorem: The total amount spent by the family is $36 -/
theorem family_spent_36_dollars :
  let popcorn_cost := ticket_cost * popcorn_ratio
  let soda_cost := popcorn_cost * soda_ratio
  let total_cost := (num_tickets : ℚ) * ticket_cost +
                    (num_popcorn : ℚ) * popcorn_cost +
                    (num_soda : ℚ) * soda_cost
  total_cost = 36 := by sorry

end NUMINAMATH_CALUDE_family_spent_36_dollars_l3365_336550


namespace NUMINAMATH_CALUDE_negation_of_proposition_quadratic_inequality_negation_l3365_336560

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬∀ x, p x) ↔ ∃ x, ¬(p x) := by sorry

theorem quadratic_inequality_negation :
  (¬∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_quadratic_inequality_negation_l3365_336560


namespace NUMINAMATH_CALUDE_pizza_price_proof_l3365_336512

/-- The standard price of a pizza at Piazzanos Pizzeria -/
def standard_price : ℚ := 5

/-- The number of triple cheese pizzas purchased -/
def triple_cheese_count : ℕ := 10

/-- The number of meat lovers pizzas purchased -/
def meat_lovers_count : ℕ := 9

/-- The total cost of the purchase -/
def total_cost : ℚ := 55

theorem pizza_price_proof :
  (triple_cheese_count / 2 + meat_lovers_count * 2 / 3) * standard_price = total_cost := by
  sorry

end NUMINAMATH_CALUDE_pizza_price_proof_l3365_336512


namespace NUMINAMATH_CALUDE_three_from_nine_combination_l3365_336589

theorem three_from_nine_combination : (Nat.choose 9 3) = 84 := by
  sorry

end NUMINAMATH_CALUDE_three_from_nine_combination_l3365_336589


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l3365_336572

theorem least_three_digit_multiple_of_eight :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 8 ∣ n → n ≥ 104 :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l3365_336572


namespace NUMINAMATH_CALUDE_exists_sequence_expectation_sum_neq_sum_expectation_l3365_336558

/-- A sequence of random variables -/
def RandomSequence := ℕ → MeasurableSpace ℝ

/-- Expected value of a random variable -/
noncomputable def expectation (X : MeasurableSpace ℝ) : ℝ := sorry

/-- Infinite sum of random variables -/
noncomputable def infiniteSum (ξ : RandomSequence) : MeasurableSpace ℝ := sorry

/-- Theorem: There exists a sequence of random variables where the expectation of the sum
    is not equal to the sum of expectations -/
theorem exists_sequence_expectation_sum_neq_sum_expectation :
  ∃ ξ : RandomSequence,
    expectation (infiniteSum ξ) ≠ ∑' n, expectation (ξ n) := by sorry

end NUMINAMATH_CALUDE_exists_sequence_expectation_sum_neq_sum_expectation_l3365_336558


namespace NUMINAMATH_CALUDE_bread_in_pond_l3365_336507

theorem bread_in_pond (total_bread : ℕ) (duck1_bread : ℕ) (duck2_bread : ℕ) (duck3_bread : ℕ) 
  (h1 : total_bread = 100)
  (h2 : duck1_bread = total_bread / 2)
  (h3 : duck2_bread = 13)
  (h4 : duck3_bread = 7) :
  total_bread - (duck1_bread + duck2_bread + duck3_bread) = 30 := by
  sorry

end NUMINAMATH_CALUDE_bread_in_pond_l3365_336507


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l3365_336539

theorem unique_positive_integer_solution : ∃! (x : ℕ), x > 0 ∧ (4 * x)^2 - 2 * x = 8066 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l3365_336539


namespace NUMINAMATH_CALUDE_ratio_constraint_l3365_336508

theorem ratio_constraint (a b : ℝ) (h1 : 0 ≤ a) (h2 : a < b) 
  (h3 : ∀ x : ℝ, a + b * Real.cos x + (b / (2 * Real.sqrt 2)) * Real.cos (2 * x) ≥ 0) :
  (b + a) / (b - a) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_constraint_l3365_336508


namespace NUMINAMATH_CALUDE_smallest_A_with_triple_factors_l3365_336537

def number_of_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_A_with_triple_factors : 
  ∃ (A : ℕ), A > 0 ∧ 
  (∀ (k : ℕ), k > 0 ∧ k < A → number_of_factors (6 * k) ≠ 3 * number_of_factors k) ∧
  number_of_factors (6 * A) = 3 * number_of_factors A ∧
  A = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_A_with_triple_factors_l3365_336537


namespace NUMINAMATH_CALUDE_negative_majority_sequence_l3365_336588

theorem negative_majority_sequence :
  ∃ (x : Fin 2004 → ℤ),
    (∀ k : Fin 2001, x (k + 3) = x (k + 2) + x k * x (k + 1)) ∧
    (∃ n : ℕ, 2 * n > 2004 ∧ (∃ S : Finset (Fin 2004), S.card = n ∧ ∀ i ∈ S, x i < 0)) := by
  sorry

end NUMINAMATH_CALUDE_negative_majority_sequence_l3365_336588


namespace NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l3365_336531

theorem parametric_to_ordinary_equation 
  (t : ℝ) (x y : ℝ) 
  (h1 : t ≥ 0) 
  (h2 : x = Real.sqrt t + 1) 
  (h3 : y = 2 * Real.sqrt t - 1) : 
  y = 2 * x - 3 ∧ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l3365_336531


namespace NUMINAMATH_CALUDE_percentage_sum_proof_l3365_336597

theorem percentage_sum_proof : (0.08 * 24) + (0.10 * 40) = 5.92 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_proof_l3365_336597


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3365_336501

/-- Given a principal amount P, an unknown interest rate R, and a 10-year period,
    if increasing the interest rate by 5% results in Rs. 400 more interest,
    then P must equal Rs. 800. -/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (h_R : R > 0) :
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 400 →
  P = 800 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3365_336501


namespace NUMINAMATH_CALUDE_triangle_inequality_ortho_segments_inequality_not_always_true_l3365_336598

/-- A triangle with sides a ≥ b ≥ c and corresponding altitudes m_a ≤ m_b ≤ m_c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  h_sides : a ≥ b ∧ b ≥ c
  h_altitudes : m_a ≤ m_b ∧ m_b ≤ m_c

/-- Lengths of segments from vertex to orthocenter along corresponding altitudes -/
structure OrthoSegments where
  m_a_star : ℝ
  m_b_star : ℝ
  m_c_star : ℝ

/-- Theorem stating the inequality for sides and altitudes -/
theorem triangle_inequality (t : Triangle) : t.a + t.m_a ≥ t.b + t.m_b ∧ t.b + t.m_b ≥ t.c + t.m_c :=
  sorry

/-- Statement that the inequality for orthocenter segments is not always true -/
theorem ortho_segments_inequality_not_always_true : 
  ¬ ∀ (t : Triangle) (o : OrthoSegments), t.a + o.m_a_star ≥ t.b + o.m_b_star ∧ t.b + o.m_b_star ≥ t.c + o.m_c_star :=
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_ortho_segments_inequality_not_always_true_l3365_336598


namespace NUMINAMATH_CALUDE_book_pages_calculation_l3365_336511

theorem book_pages_calculation (pages_per_night : ℝ) (nights : ℝ) (h1 : pages_per_night = 120.0) (h2 : nights = 10.0) :
  pages_per_night * nights = 1200.0 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l3365_336511


namespace NUMINAMATH_CALUDE_circumscribed_parallelepiped_surface_area_l3365_336563

/-- A right parallelepiped circumscribed by a sphere -/
structure CircumscribedParallelepiped where
  /-- The first base diagonal of the parallelepiped -/
  a : ℝ
  /-- The second base diagonal of the parallelepiped -/
  b : ℝ
  /-- The parallelepiped is circumscribed by a sphere -/
  is_circumscribed : True

/-- The surface area of a circumscribed parallelepiped -/
def surface_area (p : CircumscribedParallelepiped) : ℝ :=
  6 * p.a * p.b

/-- Theorem: The surface area of a right parallelepiped circumscribed by a sphere,
    with base diagonals a and b, is equal to 6ab -/
theorem circumscribed_parallelepiped_surface_area
  (p : CircumscribedParallelepiped) :
  surface_area p = 6 * p.a * p.b := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_parallelepiped_surface_area_l3365_336563


namespace NUMINAMATH_CALUDE_custom_ops_theorem_l3365_336519

/-- Custom addition operation for natural numbers -/
def customAdd (a b : ℕ) : ℕ := a + b + 1

/-- Custom multiplication operation for natural numbers -/
def customMul (a b : ℕ) : ℕ := a * b - 1

/-- Theorem stating that (5 ⊕ 7) ⊕ (2 ⊗ 4) = 21 -/
theorem custom_ops_theorem : customAdd (customAdd 5 7) (customMul 2 4) = 21 := by
  sorry

end NUMINAMATH_CALUDE_custom_ops_theorem_l3365_336519


namespace NUMINAMATH_CALUDE_ellipse_equation_l3365_336564

/-- The standard equation of an ellipse with given properties -/
theorem ellipse_equation (c a b : ℝ) (h1 : c = 3) (h2 : a = 5) (h3 : b = 4) 
  (h4 : c / a = 3 / 5) (h5 : a^2 = b^2 + c^2) :
  ∀ x y : ℝ, (x^2 / 25 + y^2 / 16 = 1) ↔ 
  (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3365_336564


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l3365_336578

theorem right_triangle_acute_angles (α β : ℝ) : 
  α > 0 ∧ β > 0 ∧  -- Acute angles are positive
  α + β = 90 ∧     -- Sum of acute angles in a right triangle is 90°
  α = 4 * β →      -- Ratio of acute angles is 4:1
  (min α β = 18 ∧ max α β = 72) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l3365_336578


namespace NUMINAMATH_CALUDE_amount_to_leave_in_till_l3365_336594

/-- Represents the number of bills of each denomination in the till -/
structure TillContents where
  hundred_bills : Nat
  fifty_bills : Nat
  twenty_bills : Nat
  ten_bills : Nat
  five_bills : Nat
  one_bills : Nat

/-- Calculates the total value of bills in the till -/
def total_in_notes (till : TillContents) : Nat :=
  till.hundred_bills * 100 +
  till.fifty_bills * 50 +
  till.twenty_bills * 20 +
  till.ten_bills * 10 +
  till.five_bills * 5 +
  till.one_bills

/-- Calculates the amount to leave in the till -/
def amount_to_leave (till : TillContents) (amount_to_hand_in : Nat) : Nat :=
  total_in_notes till - amount_to_hand_in

/-- Jack's till contents -/
def jacks_till : TillContents :=
  { hundred_bills := 2
  , fifty_bills := 1
  , twenty_bills := 5
  , ten_bills := 3
  , five_bills := 7
  , one_bills := 27 }

theorem amount_to_leave_in_till :
  amount_to_leave jacks_till 142 = 300 := by
  sorry

end NUMINAMATH_CALUDE_amount_to_leave_in_till_l3365_336594


namespace NUMINAMATH_CALUDE_four_meetings_theorem_l3365_336590

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  direction : Bool -- True for clockwise, False for counterclockwise

/-- Calculates the number of meetings between two runners on a circular track -/
def number_of_meetings (runner1 runner2 : Runner) : ℕ :=
  sorry

/-- Theorem stating that two runners with speeds 2 m/s and 3 m/s in opposite directions meet 4 times -/
theorem four_meetings_theorem (track_length : ℝ) (h : track_length > 0) :
  let runner1 : Runner := ⟨2, true⟩
  let runner2 : Runner := ⟨3, false⟩
  number_of_meetings runner1 runner2 = 4 :=
sorry

end NUMINAMATH_CALUDE_four_meetings_theorem_l3365_336590


namespace NUMINAMATH_CALUDE_min_broken_pastries_correct_l3365_336592

/-- The number of different fillings -/
def num_fillings : ℕ := 10

/-- The total number of pastries -/
def total_pastries : ℕ := 45

/-- Predicate to check if a number of broken pastries is sufficient for the trick -/
def is_sufficient (n : ℕ) : Prop :=
  ∀ (remaining : Finset (Fin 2 → Fin num_fillings)),
    remaining.card = total_pastries - n →
    ∀ pastry ∈ remaining, ∃ filling, ∀ other ∈ remaining, other ≠ pastry → pastry filling ≠ other filling

/-- The smallest number of pastries that need to be broken for the trick to work -/
def min_broken_pastries : ℕ := 36

/-- Theorem stating that min_broken_pastries is the smallest number for which the trick works -/
theorem min_broken_pastries_correct :
  is_sufficient min_broken_pastries ∧ ∀ k < min_broken_pastries, ¬is_sufficient k := by sorry

end NUMINAMATH_CALUDE_min_broken_pastries_correct_l3365_336592


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l3365_336571

/-- The number of revolutions needed for the second horse to travel the same distance as the first horse on a merry-go-round -/
theorem merry_go_round_revolutions (r₁ r₂ : ℝ) (n₁ : ℕ) (h₁ : r₁ = 15) (h₂ : r₂ = 5) (h₃ : n₁ = 20) :
  ∃ n₂ : ℕ, n₁ * r₁ = n₂ * r₂ ∧ n₂ = 60 := by
  sorry

#check merry_go_round_revolutions

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l3365_336571


namespace NUMINAMATH_CALUDE_john_profit_l3365_336582

def calculate_profit (woodburning_qty : ℕ) (woodburning_price : ℚ)
                     (metal_qty : ℕ) (metal_price : ℚ)
                     (painting_qty : ℕ) (painting_price : ℚ)
                     (glass_qty : ℕ) (glass_price : ℚ)
                     (wood_cost : ℚ) (metal_cost : ℚ)
                     (paint_cost : ℚ) (glass_cost : ℚ)
                     (woodburning_discount : ℚ) (glass_discount : ℚ)
                     (sales_tax : ℚ) : ℚ :=
  sorry

theorem john_profit :
  calculate_profit 20 15 15 25 10 40 5 30 100 150 120 90 (10/100) (15/100) (5/100) = 771.13 :=
sorry

end NUMINAMATH_CALUDE_john_profit_l3365_336582


namespace NUMINAMATH_CALUDE_exists_divisible_term_l3365_336555

/-- Sequence defined by a₀ = 5 and aₙ₊₁ = 2aₙ + 1 -/
def a : ℕ → ℕ
  | 0 => 5
  | n + 1 => 2 * a n + 1

/-- For every natural number n, there exists a different k such that a_n divides a_k -/
theorem exists_divisible_term (n : ℕ) : ∃ k : ℕ, k ≠ n ∧ a n ∣ a k := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_term_l3365_336555


namespace NUMINAMATH_CALUDE_inequality_proof_l3365_336518

theorem inequality_proof (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  x^2*y + x*y^2 + 1 ≤ x^2*y^2 + x + y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3365_336518


namespace NUMINAMATH_CALUDE_hcf_problem_l3365_336596

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 1800) (h2 : Nat.lcm a b = 200) :
  Nat.gcd a b = 9 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l3365_336596


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l3365_336544

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The function f(x) = x(x+1)(x+2)...(x+n) -/
def f (n : ℕ) (x : ℝ) : ℝ := (List.range (n + 1)).foldl (fun acc i => acc * (x + i)) x

/-- Theorem: The derivative of f(x) at x = 0 is equal to n! -/
theorem derivative_f_at_zero (n : ℕ) : 
  deriv (f n) 0 = factorial n := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l3365_336544


namespace NUMINAMATH_CALUDE_problem_solution_l3365_336561

theorem problem_solution (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc : a^2 + b^2 + c^2 = 49)
  (sum_xyz : x^2 + y^2 + z^2 = 64)
  (sum_prod : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3365_336561


namespace NUMINAMATH_CALUDE_hemisphere_volume_calculation_l3365_336585

/-- Given a total volume of water and the number of hemisphere containers,
    calculate the volume of each hemisphere container. -/
def hemisphere_volume (total_volume : ℚ) (num_containers : ℕ) : ℚ :=
  total_volume / num_containers

/-- Theorem stating that the volume of each hemisphere container is 4 L
    when 2735 containers are used to hold 10940 L of water. -/
theorem hemisphere_volume_calculation :
  hemisphere_volume 10940 2735 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_volume_calculation_l3365_336585


namespace NUMINAMATH_CALUDE_machine_C_time_l3365_336523

/-- Time for Machine A to finish the job -/
def timeA : ℝ := 4

/-- Time for Machine B to finish the job -/
def timeB : ℝ := 12

/-- Time for all machines together to finish the job -/
def timeAll : ℝ := 2

/-- Time for Machine C to finish the job -/
def timeC : ℝ := 6

/-- Theorem stating that given the conditions, Machine C takes 6 hours to finish the job alone -/
theorem machine_C_time : 
  1 / timeA + 1 / timeB + 1 / timeC = 1 / timeAll := by sorry

end NUMINAMATH_CALUDE_machine_C_time_l3365_336523


namespace NUMINAMATH_CALUDE_f_composition_range_l3365_336569

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 else 2^x

theorem f_composition_range : 
  {a : ℝ | f (f a) = 2^(f a)} = {a : ℝ | a ≥ 2/3} := by sorry

end NUMINAMATH_CALUDE_f_composition_range_l3365_336569


namespace NUMINAMATH_CALUDE_equation_solution_sum_l3365_336504

theorem equation_solution_sum (a : ℝ) (h : a ≥ 1) :
  ∃ x : ℝ, x ≥ 0 ∧ Real.sqrt (a - Real.sqrt (a + x)) = x ∧
  (∀ y : ℝ, y ≥ 0 ∧ Real.sqrt (a - Real.sqrt (a + y)) = y → y = x) ∧
  x = (Real.sqrt (4 * a - 3) - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_sum_l3365_336504


namespace NUMINAMATH_CALUDE_kevins_food_spending_l3365_336554

theorem kevins_food_spending (total_budget : ℕ) (samuels_ticket : ℕ) (samuels_food_drinks : ℕ)
  (kevins_ticket : ℕ) (kevins_drinks : ℕ) (kevins_food : ℕ)
  (h1 : total_budget = 20)
  (h2 : samuels_ticket = 14)
  (h3 : samuels_food_drinks = 6)
  (h4 : kevins_ticket = 14)
  (h5 : kevins_drinks = 2)
  (h6 : samuels_ticket + samuels_food_drinks = total_budget)
  (h7 : kevins_ticket + kevins_drinks + kevins_food = total_budget) :
  kevins_food = 4 := by
  sorry

end NUMINAMATH_CALUDE_kevins_food_spending_l3365_336554


namespace NUMINAMATH_CALUDE_english_only_students_l3365_336535

/-- Represents the number of students in each language class -/
structure LanguageClasses where
  english : ℕ
  french : ℕ
  spanish : ℕ

/-- The conditions of the problem -/
def language_class_conditions (c : LanguageClasses) : Prop :=
  c.english + c.french + c.spanish = 40 ∧
  c.english = 3 * c.french ∧
  c.english = 2 * c.spanish

/-- The theorem to prove -/
theorem english_only_students (c : LanguageClasses) 
  (h : language_class_conditions c) : 
  c.english - (c.french + c.spanish) = 30 := by
  sorry


end NUMINAMATH_CALUDE_english_only_students_l3365_336535


namespace NUMINAMATH_CALUDE_tangent_sphere_radius_l3365_336570

/-- A truncated cone with horizontal bases of radii 12 and 4, and height 15 -/
structure TruncatedCone where
  largeRadius : ℝ := 12
  smallRadius : ℝ := 4
  height : ℝ := 15

/-- A sphere tangent to the inside surfaces of a truncated cone -/
structure TangentSphere (cone : TruncatedCone) where
  radius : ℝ

/-- The radius of the tangent sphere is √161/2 -/
theorem tangent_sphere_radius (cone : TruncatedCone) (sphere : TangentSphere cone) :
  sphere.radius = Real.sqrt 161 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sphere_radius_l3365_336570


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3365_336517

theorem chess_tournament_games (n : ℕ) (h : n = 10) : 
  (n * (n - 1)) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3365_336517


namespace NUMINAMATH_CALUDE_probability_two_red_one_blue_l3365_336532

/-- Represents a cube composed of smaller unit cubes -/
structure Cube where
  edge_length : ℕ

/-- Represents the painting state of a smaller cube -/
inductive PaintState
  | Unpainted
  | Red
  | Blue
  | RedAndBlue

/-- Represents a painted cube -/
structure PaintedCube where
  cube : Cube
  paint : Cube → PaintState

/-- Calculates the number of cubes with exactly two red faces and one blue face -/
def cubes_with_two_red_one_blue (c : PaintedCube) : ℕ := sorry

/-- Calculates the total number of unit cubes in a larger cube -/
def total_unit_cubes (c : Cube) : ℕ := c.edge_length ^ 3

/-- Theorem stating the probability of selecting a cube with two red faces and one blue face -/
theorem probability_two_red_one_blue (c : PaintedCube) 
  (h1 : c.cube.edge_length = 8)
  (h2 : ∀ (x : Cube), x.edge_length = 1 → c.paint x ≠ PaintState.Unpainted) 
  (h3 : ∃ (layer : ℕ), layer < c.cube.edge_length ∧ 
    ∀ (x : Cube), x.edge_length = 1 → 
      (∃ (i j : ℕ), i < c.cube.edge_length ∧ j < c.cube.edge_length ∧
        (i = layer ∨ i = c.cube.edge_length - 1 - layer ∨
         j = layer ∨ j = c.cube.edge_length - 1 - layer)) →
      c.paint x = PaintState.Blue) :
  (cubes_with_two_red_one_blue c : ℚ) / (total_unit_cubes c.cube : ℚ) = 3 / 32 := by sorry

end NUMINAMATH_CALUDE_probability_two_red_one_blue_l3365_336532


namespace NUMINAMATH_CALUDE_part_a_part_b_l3365_336506

def solution_set_a : Set (ℤ × ℤ) := {(6, -21), (-13, -2), (4, 15), (23, -4), (7, -12), (-4, -1), (3, 6), (14, -5), (8, -9), (-1, 0), (2, 3), (11, -6)}

def equation_set_a : Set (ℤ × ℤ) := {(x, y) | x * y + 3 * x - 5 * y = -3}

theorem part_a : equation_set_a = solution_set_a := by sorry

def solution_set_b : Set (ℤ × ℤ) := {(4, 2)}

def equation_set_b : Set (ℤ × ℤ) := {(x, y) | x - y = x / y}

theorem part_b : equation_set_b = solution_set_b := by sorry

end NUMINAMATH_CALUDE_part_a_part_b_l3365_336506


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_nonpositive_l3365_336502

theorem quadratic_inequality_always_nonpositive :
  ∀ x : ℝ, -8 * x^2 + 4 * x - 7 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_nonpositive_l3365_336502


namespace NUMINAMATH_CALUDE_correct_polynomial_result_l3365_336545

/-- Given a polynomial P, prove that if subtracting P from a^2 - 5a + 7 results in 2a^2 - 3a + 5,
    then adding P to 2a^2 - 3a + 5 yields 5a^2 - 11a + 17. -/
theorem correct_polynomial_result (P : Polynomial ℚ) : 
  (a^2 - 5*a + 7 : Polynomial ℚ) - P = 2*a^2 - 3*a + 5 →
  P + (2*a^2 - 3*a + 5 : Polynomial ℚ) = 5*a^2 - 11*a + 17 := by
  sorry

end NUMINAMATH_CALUDE_correct_polynomial_result_l3365_336545


namespace NUMINAMATH_CALUDE_chord_length_l3365_336580

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length (l : Real → Real × Real) (C₁ : Real → Real × Real) : 
  (∀ t, l t = (1 + 3/5 * t, 4/5 * t)) →
  (∀ θ, C₁ θ = (Real.cos θ, Real.sin θ)) →
  (∃ A B, A ≠ B ∧ (∃ t₁ t₂, l t₁ = A ∧ l t₂ = B) ∧ (∃ θ₁ θ₂, C₁ θ₁ = A ∧ C₁ θ₂ = B)) →
  ∃ A B, A ≠ B ∧ (∃ t₁ t₂, l t₁ = A ∧ l t₂ = B) ∧ (∃ θ₁ θ₂, C₁ θ₁ = A ∧ C₁ θ₂ = B) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6/5 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l3365_336580


namespace NUMINAMATH_CALUDE_train_length_l3365_336528

/-- The length of a train given its speed and time to pass a stationary object -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 63 → time = 36 → speed * time * (1000 / 3600) = 630 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l3365_336528


namespace NUMINAMATH_CALUDE_donation_difference_l3365_336574

def total_donation : ℕ := 1000
def treetown_forest_donation : ℕ := 570

theorem donation_difference : 
  treetown_forest_donation - (total_donation - treetown_forest_donation) = 140 := by
  sorry

end NUMINAMATH_CALUDE_donation_difference_l3365_336574


namespace NUMINAMATH_CALUDE_A_inter_B_eq_B_l3365_336584

-- Define set A
def A : Set ℝ := {y | ∃ x, y = |x| - 1}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem A_inter_B_eq_B : A ∩ B = B := by sorry

end NUMINAMATH_CALUDE_A_inter_B_eq_B_l3365_336584


namespace NUMINAMATH_CALUDE_frog_jump_theorem_l3365_336557

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Function to calculate the number of final frog positions -/
def numFrogPositions (ℓ : ℕ) : ℕ :=
  ((ℓ + 2) / 2) * ((ℓ + 4) / 2) * (((ℓ + 1) / 2) * ((ℓ + 3) / 2))^2 / 8

/-- Main theorem statement -/
theorem frog_jump_theorem (abc : Triangle) (ℓ : ℕ) (m n : Point) :
  (abc.a.x = 0 ∧ abc.a.y = 0) →  -- A at origin
  (abc.b.x = 1 ∧ abc.b.y = 0) →  -- B at (1,0)
  (abc.c.x = 1/2 ∧ abc.c.y = Real.sqrt 3 / 2) →  -- C at (1/2, √3/2)
  (m.x = ℓ ∧ m.y = 0) →  -- M on AB
  (n.x = ℓ/2 ∧ n.y = ℓ * Real.sqrt 3 / 2) →  -- N on AC
  ∃ (finalPositions : ℕ), finalPositions = numFrogPositions ℓ :=
by sorry

end NUMINAMATH_CALUDE_frog_jump_theorem_l3365_336557


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l3365_336514

theorem complementary_angles_difference (x y : ℝ) : 
  x + y = 90 →  -- angles are complementary
  x = 4 * y →   -- ratio of angles is 4:1
  |x - y| = 54  -- absolute difference between angles is 54°
  := by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l3365_336514


namespace NUMINAMATH_CALUDE_definite_integral_equality_l3365_336565

theorem definite_integral_equality : ∫ x in (1 : ℝ)..3, (2 * x - 1 / x^2) = 22 / 3 := by sorry

end NUMINAMATH_CALUDE_definite_integral_equality_l3365_336565


namespace NUMINAMATH_CALUDE_restaurant_change_l3365_336534

/-- Calculates the change received after a restaurant meal -/
theorem restaurant_change 
  (lee_money : ℕ) 
  (friend_money : ℕ) 
  (wings_cost : ℕ) 
  (salad_cost : ℕ) 
  (soda_cost : ℕ) 
  (soda_quantity : ℕ) 
  (tax : ℕ) 
  (h1 : lee_money = 10) 
  (h2 : friend_money = 8) 
  (h3 : wings_cost = 6) 
  (h4 : salad_cost = 4) 
  (h5 : soda_cost = 1) 
  (h6 : soda_quantity = 2) 
  (h7 : tax = 3) : 
  lee_money + friend_money - (wings_cost + salad_cost + soda_cost * soda_quantity + tax) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_change_l3365_336534
