import Mathlib

namespace NUMINAMATH_CALUDE_store_profits_l3946_394612

theorem store_profits (profit_a profit_b : ℝ) 
  (h : profit_a * 1.2 = profit_b * 0.9) : 
  profit_a = 0.75 * profit_b := by
sorry

end NUMINAMATH_CALUDE_store_profits_l3946_394612


namespace NUMINAMATH_CALUDE_virginia_eggs_l3946_394609

theorem virginia_eggs (initial_eggs : ℕ) (taken_eggs : ℕ) (final_eggs : ℕ) : 
  initial_eggs = 96 → taken_eggs = 3 → final_eggs = initial_eggs - taken_eggs → final_eggs = 93 := by
  sorry

end NUMINAMATH_CALUDE_virginia_eggs_l3946_394609


namespace NUMINAMATH_CALUDE_probability_no_distinct_roots_l3946_394610

-- Define the range of b and c
def valid_range (n : Int) : Prop := -7 ≤ n ∧ n ≤ 7

-- Define the condition for not having distinct real roots
def no_distinct_roots (b c : Int) : Prop := b^2 - 4*c ≤ 0

-- Define the total number of possible pairs
def total_pairs : Nat := (15 * 15 : Nat)

-- Define the number of pairs that don't have distinct roots
def pairs_without_distinct_roots : Nat := 180

-- Theorem statement
theorem probability_no_distinct_roots :
  (pairs_without_distinct_roots : Rat) / total_pairs = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_distinct_roots_l3946_394610


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l3946_394696

/-- The range of m for which the point P(1-1/3m, m-5) is in the third quadrant --/
theorem point_in_third_quadrant (m : ℝ) : 
  (1 - 1/3*m < 0 ∧ m - 5 < 0) ↔ (3 < m ∧ m < 5) := by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l3946_394696


namespace NUMINAMATH_CALUDE_smallest_power_divisible_by_240_l3946_394681

theorem smallest_power_divisible_by_240 (n : ℕ) : 
  (∀ k : ℕ, k < n → ¬(240 ∣ 60^k)) ∧ (240 ∣ 60^n) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_power_divisible_by_240_l3946_394681


namespace NUMINAMATH_CALUDE_exam_score_calculation_l3946_394678

theorem exam_score_calculation (total_questions : ℕ) (correct_score : ℕ) (total_score : ℕ) (correct_answers : ℕ) 
  (h1 : total_questions = 50)
  (h2 : correct_score = 4)
  (h3 : total_score = 130)
  (h4 : correct_answers = 36) :
  (correct_score * correct_answers - total_score) / (total_questions - correct_answers) = 1 := by
sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l3946_394678


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3946_394645

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence satisfying certain conditions, prove that a₇ + a₁₀ = 27/2 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h1 : a 3 + a 6 = 6) 
  (h2 : a 5 + a 8 = 9) : 
  a 7 + a 10 = 27/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3946_394645


namespace NUMINAMATH_CALUDE_runners_meeting_point_l3946_394624

/-- Represents the marathon track setup and runners' speeds -/
structure MarathonTrack where
  totalLength : ℝ
  uphillLength : ℝ
  jackHeadStart : ℝ
  jackUphillSpeed : ℝ
  jackDownhillSpeed : ℝ
  jillUphillSpeed : ℝ
  jillDownhillSpeed : ℝ

/-- Calculates the distance from the top of the hill where runners meet -/
def distanceFromTop (track : MarathonTrack) : ℝ :=
  sorry

/-- Theorem stating the distance from the top where runners meet -/
theorem runners_meeting_point (track : MarathonTrack)
  (h1 : track.totalLength = 16)
  (h2 : track.uphillLength = 8)
  (h3 : track.jackHeadStart = 0.25)
  (h4 : track.jackUphillSpeed = 12)
  (h5 : track.jackDownhillSpeed = 18)
  (h6 : track.jillUphillSpeed = 14)
  (h7 : track.jillDownhillSpeed = 20) :
  distanceFromTop track = 511 / 32 := by
  sorry

end NUMINAMATH_CALUDE_runners_meeting_point_l3946_394624


namespace NUMINAMATH_CALUDE_specific_pentagon_area_l3946_394620

/-- Pentagon PQRST with given side lengths and angles -/
structure Pentagon where
  PQ : ℝ
  QR : ℝ
  ST : ℝ
  perimeter : ℝ
  angle_QRS : ℝ
  angle_RST : ℝ
  angle_STP : ℝ

/-- The area of a pentagon with given properties -/
def pentagon_area (p : Pentagon) : ℝ :=
  sorry

/-- Theorem stating the area of the specific pentagon -/
theorem specific_pentagon_area :
  ∀ (p : Pentagon),
    p.PQ = 13 ∧
    p.QR = 18 ∧
    p.ST = 30 ∧
    p.perimeter = 82 ∧
    p.angle_QRS = 90 ∧
    p.angle_RST = 90 ∧
    p.angle_STP = 90 →
    pentagon_area p = 270 :=
  sorry

end NUMINAMATH_CALUDE_specific_pentagon_area_l3946_394620


namespace NUMINAMATH_CALUDE_gas_station_candy_boxes_l3946_394639

theorem gas_station_candy_boxes : 3 + 5 + 2 + 4 + 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_gas_station_candy_boxes_l3946_394639


namespace NUMINAMATH_CALUDE_unique_solution_system_l3946_394626

theorem unique_solution_system (x y : ℝ) : 
  x ≥ 0 ∧ y ≥ 0 ∧
  y * Real.sqrt (2 * x) - x * Real.sqrt (2 * y) = 6 ∧
  x * y^2 - x^2 * y = 30 →
  x = 1/2 ∧ y = 8 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3946_394626


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3946_394608

theorem quadratic_roots_relation (b c : ℝ) : 
  (∃ r s : ℝ, 2 * r^2 - 4 * r - 5 = 0 ∧ 
               2 * s^2 - 4 * s - 5 = 0 ∧
               ∀ x : ℝ, x^2 + b * x + c = 0 ↔ (x = r - 3 ∨ x = s - 3)) →
  c = 1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3946_394608


namespace NUMINAMATH_CALUDE_sealant_cost_per_square_foot_l3946_394618

/-- Calculates the cost per square foot of sealant for a deck -/
theorem sealant_cost_per_square_foot
  (length : ℝ)
  (width : ℝ)
  (construction_cost_per_sqft : ℝ)
  (total_paid : ℝ)
  (h1 : length = 30)
  (h2 : width = 40)
  (h3 : construction_cost_per_sqft = 3)
  (h4 : total_paid = 4800) :
  (total_paid - construction_cost_per_sqft * length * width) / (length * width) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sealant_cost_per_square_foot_l3946_394618


namespace NUMINAMATH_CALUDE_construction_material_total_l3946_394625

theorem construction_material_total (gravel sand : ℝ) 
  (h1 : gravel = 5.91) (h2 : sand = 8.11) : 
  gravel + sand = 14.02 := by
  sorry

end NUMINAMATH_CALUDE_construction_material_total_l3946_394625


namespace NUMINAMATH_CALUDE_inequality_solution_count_l3946_394672

theorem inequality_solution_count : 
  ∃! (x : ℕ), x > 0 ∧ 15 < -2 * (x : ℤ) + 17 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_count_l3946_394672


namespace NUMINAMATH_CALUDE_andrews_family_size_l3946_394667

/-- Given the conditions of Andrew's family mask usage, prove the number of family members excluding Andrew. -/
theorem andrews_family_size (total_masks : ℕ) (change_interval : ℕ) (total_days : ℕ) :
  total_masks = 100 →
  change_interval = 4 →
  total_days = 80 →
  ∃ (family_size : ℕ), family_size = 4 ∧ 
    (family_size + 1) * (total_days / change_interval) = total_masks :=
by sorry

end NUMINAMATH_CALUDE_andrews_family_size_l3946_394667


namespace NUMINAMATH_CALUDE_square_root_fraction_simplification_l3946_394688

theorem square_root_fraction_simplification :
  (Real.sqrt (8^2 + 15^2)) / (Real.sqrt (25 + 16)) = (17 * Real.sqrt 41) / 41 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_simplification_l3946_394688


namespace NUMINAMATH_CALUDE_team_selection_with_twins_l3946_394617

-- Define the total number of players
def total_players : ℕ := 18

-- Define the number of players to be chosen
def chosen_players : ℕ := 7

-- Define the number of twins
def num_twins : ℕ := 2

-- Theorem statement
theorem team_selection_with_twins :
  (Nat.choose total_players chosen_players) - 
  (Nat.choose (total_players - num_twins) chosen_players) = 20384 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_with_twins_l3946_394617


namespace NUMINAMATH_CALUDE_decimal_calculation_l3946_394619

theorem decimal_calculation : (3.15 * 2.5) - 1.75 = 6.125 := by
  sorry

end NUMINAMATH_CALUDE_decimal_calculation_l3946_394619


namespace NUMINAMATH_CALUDE_fourth_root_equality_exp_power_equality_cube_root_equality_sqrt_product_inequality_l3946_394689

-- Define π as a real number greater than 3
variable (π : ℝ) [Fact (π > 3)]

-- Theorem for option A
theorem fourth_root_equality : ∀ π : ℝ, π > 3 → (((3 - π) ^ 4) ^ (1/4 : ℝ)) = π - 3 := by sorry

-- Theorem for option B
theorem exp_power_equality : ∀ x : ℝ, Real.exp (2 * x) = (Real.exp x) ^ 2 := by sorry

-- Theorem for option C
theorem cube_root_equality : ∀ a b : ℝ, ((a - b) ^ 3) ^ (1/3 : ℝ) = a - b := by sorry

-- Theorem for option D (showing it's not always true)
theorem sqrt_product_inequality : ∃ a b : ℝ, (a * b) ^ (1/2 : ℝ) ≠ (a ^ (1/2 : ℝ)) * (b ^ (1/2 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_fourth_root_equality_exp_power_equality_cube_root_equality_sqrt_product_inequality_l3946_394689


namespace NUMINAMATH_CALUDE_circle_radius_l3946_394646

theorem circle_radius (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 2*y + k = 0 → 
   ∃ h a : ℝ, (x - h)^2 + (y - a)^2 = 5^2) ↔ 
  k = -8 := by sorry

end NUMINAMATH_CALUDE_circle_radius_l3946_394646


namespace NUMINAMATH_CALUDE_special_function_properties_l3946_394611

/-- A function satisfying the given properties -/
def SpecialFunction (g : ℝ → ℝ) : Prop :=
  (∀ x, g x > 0) ∧ (∀ a b, g a * g b = g (a * b))

theorem special_function_properties (g : ℝ → ℝ) (h : SpecialFunction g) :
  (g 1 = 1) ∧ (∀ a, g (1 / a) = 1 / g a) := by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l3946_394611


namespace NUMINAMATH_CALUDE_rectangle_to_parallelogram_perimeter_l3946_394687

/-- A rectangle is transformed into a parallelogram while maintaining the same perimeter -/
theorem rectangle_to_parallelogram_perimeter (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :
  let rectangle_perimeter := 2 * (a + b)
  let parallelogram_perimeter := 2 * (a + b)
  rectangle_perimeter = parallelogram_perimeter :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_parallelogram_perimeter_l3946_394687


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3946_394605

theorem cube_root_equation_solution (y : ℝ) :
  (6 - 2 / y) ^ (1/3 : ℝ) = -3 → y = 2/33 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3946_394605


namespace NUMINAMATH_CALUDE_yellow_white_flowers_l3946_394614

theorem yellow_white_flowers (total : ℕ) (red_yellow : ℕ) (red_white : ℕ) (red_minus_white : ℕ) :
  total = 44 →
  red_yellow = 17 →
  red_white = 14 →
  red_minus_white = 4 →
  red_yellow + red_white - (red_white + (total - red_yellow - red_white)) = red_minus_white →
  total - red_yellow - red_white = 13 := by
sorry

end NUMINAMATH_CALUDE_yellow_white_flowers_l3946_394614


namespace NUMINAMATH_CALUDE_cookies_ratio_l3946_394658

/-- Proves the ratio of cookies eaten by Monica's mother to her father -/
theorem cookies_ratio :
  ∀ (total mother_cookies father_cookies brother_cookies left : ℕ),
  total = 30 →
  father_cookies = 10 →
  brother_cookies = mother_cookies + 2 →
  left = 8 →
  total = mother_cookies + father_cookies + brother_cookies + left →
  (mother_cookies : ℚ) / father_cookies = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cookies_ratio_l3946_394658


namespace NUMINAMATH_CALUDE_work_hours_ratio_l3946_394663

theorem work_hours_ratio (amber_hours : ℕ) (total_hours : ℕ) : 
  amber_hours = 12 →
  total_hours = 40 →
  ∃ (ella_hours : ℕ),
    (ella_hours + amber_hours + amber_hours / 3 = total_hours) ∧
    (ella_hours : ℚ) / amber_hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_work_hours_ratio_l3946_394663


namespace NUMINAMATH_CALUDE_sum_remainder_mod_nine_l3946_394666

theorem sum_remainder_mod_nine (n : ℤ) : (8 - n + (n + 5)) % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_nine_l3946_394666


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l3946_394695

theorem greatest_integer_inequality : ∀ x : ℤ, (7 : ℚ) / 9 > (x : ℚ) / 15 ↔ x ≤ 11 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l3946_394695


namespace NUMINAMATH_CALUDE_simplify_expression_l3946_394693

theorem simplify_expression (x y : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 20 + 4*y = 45*x + 20 + 4*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3946_394693


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l3946_394657

theorem right_triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b < c) (right_triangle : a^2 + b^2 = c^2) :
  (1/a + 1/b + 1/c) ≥ (5 + 3 * Real.sqrt 2) / (a + b + c) ∧
  ∀ M > 5 + 3 * Real.sqrt 2, ∃ a' b' c' : ℝ,
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' ≤ b' ∧ b' < c' ∧ a'^2 + b'^2 = c'^2 ∧
    (1/a' + 1/b' + 1/c') < M / (a' + b' + c') := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l3946_394657


namespace NUMINAMATH_CALUDE_complex_fourth_power_l3946_394692

theorem complex_fourth_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l3946_394692


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3946_394637

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) :
  (∀ x : ℚ, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₇ + a₆ + a₅ + a₄ + a₃ + a₂ + a₁ = 129 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3946_394637


namespace NUMINAMATH_CALUDE_cone_volume_l3946_394654

theorem cone_volume (s : ℝ) (θ : ℝ) (h : s = 6 ∧ θ = 2 * π / 3) :
  ∃ (V : ℝ), V = (16 * Real.sqrt 2 / 3) * π ∧
  V = (1 / 3) * π * (s * θ / (2 * π))^2 * Real.sqrt (s^2 - (s * θ / (2 * π))^2) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l3946_394654


namespace NUMINAMATH_CALUDE_village_burn_time_l3946_394602

/-- Represents the number of cottages remaining after n intervals -/
def A : ℕ → ℕ
| 0 => 90
| n + 1 => 2 * A n - 96

/-- The time it takes Trodgor to burn down the village -/
def burnTime : ℕ := 1920

theorem village_burn_time : 
  ∀ n : ℕ, A n = 0 → n * 480 = burnTime := by
  sorry

#check village_burn_time

end NUMINAMATH_CALUDE_village_burn_time_l3946_394602


namespace NUMINAMATH_CALUDE_ant_return_probability_60_l3946_394673

/-- The probability of an ant returning to its starting vertex on a tetrahedron after n random edge traversals -/
def ant_return_probability (n : ℕ) : ℚ :=
  (3^(n-1) + 1) / (4 * 3^(n-1))

/-- The theorem stating the probability of an ant returning to its starting vertex on a tetrahedron after 60 random edge traversals -/
theorem ant_return_probability_60 :
  ant_return_probability 60 = (3^59 + 1) / (4 * 3^59) := by
  sorry

end NUMINAMATH_CALUDE_ant_return_probability_60_l3946_394673


namespace NUMINAMATH_CALUDE_burrito_cost_burrito_cost_is_six_l3946_394650

/-- Calculates the cost of burritos given the following conditions:
  * There are 10 burritos with 120 calories each
  * 5 burgers with 400 calories each cost $8
  * Burgers provide 50 more calories per dollar than burritos
-/
theorem burrito_cost : ℝ → Prop :=
  fun cost : ℝ =>
    let burrito_count : ℕ := 10
    let burrito_calories : ℕ := 120
    let burger_count : ℕ := 5
    let burger_calories : ℕ := 400
    let burger_cost : ℝ := 8
    let calorie_difference : ℝ := 50

    let total_burrito_calories : ℕ := burrito_count * burrito_calories
    let total_burger_calories : ℕ := burger_count * burger_calories
    let burger_calories_per_dollar : ℝ := total_burger_calories / burger_cost
    let burrito_calories_per_dollar : ℝ := burger_calories_per_dollar - calorie_difference

    cost = total_burrito_calories / burrito_calories_per_dollar ∧
    cost = 6

theorem burrito_cost_is_six : burrito_cost 6 := by
  sorry

end NUMINAMATH_CALUDE_burrito_cost_burrito_cost_is_six_l3946_394650


namespace NUMINAMATH_CALUDE_shaded_area_is_eight_l3946_394680

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a semicircle -/
structure Semicircle where
  center : Point
  radius : ℝ

/-- The geometric configuration -/
structure GeometricLayout where
  semicircle_ADB : Semicircle
  semicircle_BEC : Semicircle
  semicircle_DFE : Semicircle
  point_D : Point
  point_E : Point
  point_F : Point

/-- Conditions of the geometric layout -/
def validGeometricLayout (layout : GeometricLayout) : Prop :=
  layout.semicircle_ADB.radius = 2 ∧
  layout.semicircle_BEC.radius = 2 ∧
  layout.semicircle_DFE.radius = 1 ∧
  -- D is midpoint of ADB
  layout.point_D = { x := layout.semicircle_ADB.center.x, y := layout.semicircle_ADB.center.y + layout.semicircle_ADB.radius } ∧
  -- E is midpoint of BEC
  layout.point_E = { x := layout.semicircle_BEC.center.x, y := layout.semicircle_BEC.center.y + layout.semicircle_BEC.radius } ∧
  -- F is midpoint of DFE
  layout.point_F = { x := layout.semicircle_DFE.center.x, y := layout.semicircle_DFE.center.y + layout.semicircle_DFE.radius }

/-- Calculate the area of the shaded region -/
def shadedArea (layout : GeometricLayout) : ℝ :=
  -- Placeholder for the actual calculation
  8

/-- Theorem stating that the shaded area is 8 square units -/
theorem shaded_area_is_eight (layout : GeometricLayout) (h : validGeometricLayout layout) :
  shadedArea layout = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_shaded_area_is_eight_l3946_394680


namespace NUMINAMATH_CALUDE_fish_pond_area_increase_l3946_394613

/-- Proves that the increase in area of a rectangular fish pond is (20x-4) square meters
    when both length and width are increased by 2 meters. -/
theorem fish_pond_area_increase (x : ℝ) :
  let original_length : ℝ := 5 * x
  let original_width : ℝ := 5 * x - 4
  let new_length : ℝ := original_length + 2
  let new_width : ℝ := original_width + 2
  let original_area : ℝ := original_length * original_width
  let new_area : ℝ := new_length * new_width
  new_area - original_area = 20 * x - 4 := by
sorry

end NUMINAMATH_CALUDE_fish_pond_area_increase_l3946_394613


namespace NUMINAMATH_CALUDE_propositions_truth_l3946_394643

theorem propositions_truth :
  (∀ a b : ℝ, a * b > 0 → a > b → 1 / a < 1 / b) ∧
  (∀ a b : ℝ, a > abs b → a^2 > b^2) ∧
  (¬ ∀ a b c d : ℝ, a > b → c > d → a - c > b - d) ∧
  (∀ a b m : ℝ, 0 < a → a < b → m > 0 → a / b < (a + m) / (b + m)) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l3946_394643


namespace NUMINAMATH_CALUDE_xiaoqiang_games_l3946_394691

/-- Represents a player in the chess tournament -/
inductive Player : Type
| Jia : Player
| Yi : Player
| Bing : Player
| Ding : Player
| Xiaoqiang : Player

/-- The number of games played by each player -/
def games_played (p : Player) : ℕ :=
  match p with
  | Player.Jia => 4
  | Player.Yi => 3
  | Player.Bing => 2
  | Player.Ding => 1
  | Player.Xiaoqiang => 2  -- This is what we want to prove

/-- The total number of games in a round-robin tournament -/
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem xiaoqiang_games :
  games_played Player.Xiaoqiang = 2 :=
by sorry

end NUMINAMATH_CALUDE_xiaoqiang_games_l3946_394691


namespace NUMINAMATH_CALUDE_cloth_price_calculation_l3946_394699

theorem cloth_price_calculation (quantity : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) (total_cost : ℝ) :
  quantity = 9.25 →
  discount_rate = 0.12 →
  tax_rate = 0.05 →
  total_cost = 397.75 →
  ∃ P : ℝ, (quantity * (P - discount_rate * P)) * (1 + tax_rate) = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_cloth_price_calculation_l3946_394699


namespace NUMINAMATH_CALUDE_scientific_notation_of_2590000_l3946_394644

theorem scientific_notation_of_2590000 :
  2590000 = 2.59 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2590000_l3946_394644


namespace NUMINAMATH_CALUDE_ice_cream_revenue_l3946_394616

/-- Calculate the total revenue from ice cream sales with discounts --/
theorem ice_cream_revenue : 
  let chocolate : ℕ := 50
  let mango : ℕ := 54
  let vanilla : ℕ := 80
  let strawberry : ℕ := 40
  let price : ℚ := 2
  let chocolate_sold : ℚ := 3 / 5 * chocolate
  let mango_sold : ℚ := 2 / 3 * mango
  let vanilla_sold : ℚ := 75 / 100 * vanilla
  let strawberry_sold : ℚ := 5 / 8 * strawberry
  let discount : ℚ := 1 / 2
  let apply_discount (x : ℚ) : ℚ := if x ≥ 10 then x * discount else 0

  let total_revenue : ℚ := 
    (chocolate_sold + mango_sold + vanilla_sold + strawberry_sold) * price - 
    (apply_discount chocolate_sold + apply_discount mango_sold + 
     apply_discount vanilla_sold + apply_discount strawberry_sold)

  total_revenue = 226.5 := by sorry

end NUMINAMATH_CALUDE_ice_cream_revenue_l3946_394616


namespace NUMINAMATH_CALUDE_cyclists_meet_time_l3946_394665

/-- Two cyclists meet at the starting point on a circular track -/
theorem cyclists_meet_time (circumference : ℝ) (speed1 speed2 : ℝ) 
  (h_circumference : circumference = 600)
  (h_speed1 : speed1 = 7)
  (h_speed2 : speed2 = 8) :
  (circumference / (speed1 + speed2)) = 40 := by
  sorry

#check cyclists_meet_time

end NUMINAMATH_CALUDE_cyclists_meet_time_l3946_394665


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3946_394634

/-- 
For a quadratic equation kx^2 + 2x + 1 = 0, where k is a real number,
this theorem states that the equation has real roots if and only if k ≤ 1 and k ≠ 0.
-/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3946_394634


namespace NUMINAMATH_CALUDE_min_value_expression_l3946_394675

theorem min_value_expression (a b : ℝ) (h1 : ab - 4*a - b + 1 = 0) (h2 : a > 1) :
  ∀ x y : ℝ, x * y - 4*x - y + 1 = 0 → x > 1 → (a + 1) * (b + 2) ≤ (x + 1) * (y + 2) ∧
  ∃ a₀ b₀ : ℝ, a₀ * b₀ - 4*a₀ - b₀ + 1 = 0 ∧ a₀ > 1 ∧ (a₀ + 1) * (b₀ + 2) = 27 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3946_394675


namespace NUMINAMATH_CALUDE_overtime_increase_is_25_percent_l3946_394635

/-- Calculates the percentage increase for overtime pay given basic pay and total wage information. -/
def overtime_percentage_increase (basic_pay : ℚ) (total_wage : ℚ) (basic_hours : ℕ) (total_hours : ℕ) : ℚ :=
  let basic_rate : ℚ := basic_pay / basic_hours
  let overtime_hours : ℕ := total_hours - basic_hours
  let overtime_pay : ℚ := total_wage - basic_pay
  let overtime_rate : ℚ := overtime_pay / overtime_hours
  ((overtime_rate - basic_rate) / basic_rate) * 100

/-- Theorem stating that given the specified conditions, the overtime percentage increase is 25%. -/
theorem overtime_increase_is_25_percent :
  overtime_percentage_increase 20 25 40 48 = 25 := by
  sorry

end NUMINAMATH_CALUDE_overtime_increase_is_25_percent_l3946_394635


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l3946_394600

theorem divisibility_by_eleven (B : Nat) : 
  (B = 5 → 11 ∣ 15675) → 
  (∀ n : Nat, n < 10 → (11 ∣ (15670 + n) ↔ n = B)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l3946_394600


namespace NUMINAMATH_CALUDE_cylinder_base_area_l3946_394698

theorem cylinder_base_area (S : ℝ) (h : S > 0) :
  let cross_section_area := 4 * S
  let cross_section_is_square := true
  let base_area := π * S
  cross_section_is_square ∧ cross_section_area = 4 * S → base_area = π * S :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_base_area_l3946_394698


namespace NUMINAMATH_CALUDE_divisible_by_nine_l3946_394603

theorem divisible_by_nine (x y : ℕ) (h : x < 10 ∧ y < 10) :
  (300000 + 10000 * x + 5700 + 70 * y + 2) % 9 = 0 →
  x + y = 1 ∨ x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l3946_394603


namespace NUMINAMATH_CALUDE_average_speed_approx_202_l3946_394621

/-- Calculates the average speed given initial and final odometer readings and total driving time -/
def average_speed (initial_reading final_reading : ℕ) (total_time : ℚ) : ℚ :=
  (final_reading - initial_reading : ℚ) / total_time

theorem average_speed_approx_202 (initial_reading final_reading : ℕ) (total_time : ℚ) :
  initial_reading = 12321 →
  final_reading = 14741 →
  total_time = 12 →
  ∃ ε > 0, |average_speed initial_reading final_reading total_time - 202| < ε :=
by
  sorry

#eval average_speed 12321 14741 12

end NUMINAMATH_CALUDE_average_speed_approx_202_l3946_394621


namespace NUMINAMATH_CALUDE_consecutive_integers_product_272_sum_33_l3946_394628

theorem consecutive_integers_product_272_sum_33 :
  ∀ x y : ℕ,
  x > 0 →
  y = x + 1 →
  x * y = 272 →
  x + y = 33 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_272_sum_33_l3946_394628


namespace NUMINAMATH_CALUDE_angle_c_measure_l3946_394601

theorem angle_c_measure (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  A + B = 80 →       -- Given condition
  C = 100            -- Conclusion to prove
:= by sorry

end NUMINAMATH_CALUDE_angle_c_measure_l3946_394601


namespace NUMINAMATH_CALUDE_cow_value_increase_l3946_394636

def starting_weight : ℝ := 732
def weight_increase_factor : ℝ := 1.35
def price_per_pound : ℝ := 2.75

theorem cow_value_increase : 
  let new_weight := starting_weight * weight_increase_factor
  let value_at_new_weight := new_weight * price_per_pound
  let value_at_starting_weight := starting_weight * price_per_pound
  value_at_new_weight - value_at_starting_weight = 704.55 := by sorry

end NUMINAMATH_CALUDE_cow_value_increase_l3946_394636


namespace NUMINAMATH_CALUDE_expression_value_l3946_394652

theorem expression_value : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3946_394652


namespace NUMINAMATH_CALUDE_rounding_down_less_than_exact_sum_l3946_394642

def fraction_a : ℚ := 2 / 3
def fraction_b : ℚ := 5 / 4

def round_down (q : ℚ) : ℤ := ⌊q⌋

theorem rounding_down_less_than_exact_sum :
  (round_down fraction_a : ℚ) + (round_down fraction_b : ℚ) ≤ fraction_a + fraction_b := by
  sorry

end NUMINAMATH_CALUDE_rounding_down_less_than_exact_sum_l3946_394642


namespace NUMINAMATH_CALUDE_exists_special_number_l3946_394676

def is_ten_digit (n : ℕ) : Prop :=
  10^9 ≤ n ∧ n < 10^10

def all_digits_distinct (n : ℕ) : Prop :=
  ∀ d₁ d₂, 0 ≤ d₁ ∧ d₁ < 10 ∧ 0 ≤ d₂ ∧ d₂ < 10 →
    (n / 10^d₁ % 10 = n / 10^d₂ % 10) → d₁ = d₂

theorem exists_special_number :
  ∃ (a b : ℕ), a ≠ b ∧ a < 10 ∧ b < 10 ∧
    is_ten_digit ((10000 * a + 1111 * b)^2 - 1) ∧
    all_digits_distinct ((10000 * a + 1111 * b)^2 - 1) :=
sorry

end NUMINAMATH_CALUDE_exists_special_number_l3946_394676


namespace NUMINAMATH_CALUDE_zero_in_interval_l3946_394674

-- Define the function f
def f (x : ℝ) : ℝ := -|x - 5| + 2*x - 1

-- State the theorem
theorem zero_in_interval : 
  ∃ x ∈ Set.Ioo 2 3, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3946_394674


namespace NUMINAMATH_CALUDE_triangle_area_l3946_394670

-- Define the plane region
def PlaneRegion (k : ℝ) := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 2 * p.1 ∧ k * p.1 - p.2 + 1 ≥ 0}

-- Define a right triangle
def IsRightTriangle (r : Set (ℝ × ℝ)) : Prop := sorry

-- Define the area of a set in ℝ²
noncomputable def Area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem triangle_area (k : ℝ) :
  IsRightTriangle (PlaneRegion k) →
  Area (PlaneRegion k) = 1/5 ∨ Area (PlaneRegion k) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3946_394670


namespace NUMINAMATH_CALUDE_find_other_number_l3946_394651

theorem find_other_number (A B : ℕ+) (h1 : A = 24) (h2 : Nat.gcd A B = 14) (h3 : Nat.lcm A B = 312) :
  B = 182 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l3946_394651


namespace NUMINAMATH_CALUDE_isosceles_triangle_on_parabola_l3946_394671

/-- Given two points P and Q on the parabola y = -x^2 that form an isosceles triangle POQ with the origin O,
    prove that the distance between P and Q is twice the x-coordinate of P. -/
theorem isosceles_triangle_on_parabola (p : ℝ) :
  let P : ℝ × ℝ := (p, -p^2)
  let Q : ℝ × ℝ := (-p, -p^2)
  let O : ℝ × ℝ := (0, 0)
  (P.1^2 + P.2^2 = Q.1^2 + Q.2^2) →  -- PO = OQ (isosceles condition)
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * p  -- PQ = 2p
:= by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_on_parabola_l3946_394671


namespace NUMINAMATH_CALUDE_tangent_line_sum_l3946_394631

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that the tangent line at (1, f(1)) has equation y = 1/2 * x + 2
def has_tangent_line (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), m = (1/2 : ℝ) ∧ b = 2 ∧ 
  ∀ (x : ℝ), f 1 + m * (x - 1) = m * x + b

-- State the theorem
theorem tangent_line_sum (f : ℝ → ℝ) (h : has_tangent_line f) :
  f 1 + deriv f 1 = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l3946_394631


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l3946_394669

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l3946_394669


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3946_394623

/-- Represents a chess tournament with players of two ranks -/
structure ChessTournament where
  a_players : Nat
  b_players : Nat

/-- Calculates the total number of games in a chess tournament -/
def total_games (t : ChessTournament) : Nat :=
  t.a_players * t.b_players

/-- Theorem: In a chess tournament with 3 'A' players and 3 'B' players, 
    where each 'A' player faces all 'B' players, the total number of games is 9 -/
theorem chess_tournament_games :
  ∀ (t : ChessTournament), 
  t.a_players = 3 → t.b_players = 3 → total_games t = 9 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_games_l3946_394623


namespace NUMINAMATH_CALUDE_centroid_property_l3946_394649

/-- The centroid of a triangle divides each median in the ratio 2:1 and creates three equal-area subtriangles. -/
structure Centroid (xA yA xB yB xC yC : ℚ) where
  x : ℚ
  y : ℚ
  is_centroid : x = (xA + xB + xC) / 3 ∧ y = (yA + yB + yC) / 3

/-- Given a triangle ABC with vertices A(5,8), B(3,-2), and C(6,1),
    if D(m,n) is the centroid of the triangle, then 10m + n = 49. -/
theorem centroid_property :
  let d : Centroid 5 8 3 (-2) 6 1 := ⟨(14/3), (7/3), by sorry⟩
  10 * d.x + d.y = 49 := by sorry

end NUMINAMATH_CALUDE_centroid_property_l3946_394649


namespace NUMINAMATH_CALUDE_gift_card_value_l3946_394690

theorem gift_card_value (original_value : ℝ) : 
  (3 / 8 : ℝ) * original_value = 75 → original_value = 200 := by
  sorry

end NUMINAMATH_CALUDE_gift_card_value_l3946_394690


namespace NUMINAMATH_CALUDE_probability_of_circle_l3946_394630

theorem probability_of_circle (total_figures : ℕ) (circles : ℕ) 
  (h1 : total_figures = 10)
  (h2 : circles = 4) :
  (circles : ℚ) / total_figures = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_circle_l3946_394630


namespace NUMINAMATH_CALUDE_min_value_theorem_l3946_394606

theorem min_value_theorem (a b : ℝ) (h : a^2 + 2*b^2 = 6) :
  ∃ (m : ℝ), m = -2*Real.sqrt 3 ∧ ∀ x y : ℝ, x^2 + 2*y^2 = 6 → m ≤ x + Real.sqrt 2 * y :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3946_394606


namespace NUMINAMATH_CALUDE_equation_solution_l3946_394615

theorem equation_solution :
  ∃! x : ℚ, 7 + 3.5 * x = 2.1 * x - 30 * 1.5 ∧ x = -520 / 14 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3946_394615


namespace NUMINAMATH_CALUDE_nabla_calculation_l3946_394640

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := a + b^a

-- Theorem statement
theorem nabla_calculation : nabla (nabla 3 2) 2 = 2059 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l3946_394640


namespace NUMINAMATH_CALUDE_win_sector_area_l3946_394638

/-- Given a circular spinner with radius 10 cm and a probability of winning 2/5,
    the area of the WIN sector is 40π square centimeters. -/
theorem win_sector_area (r : ℝ) (p : ℝ) (A_win : ℝ) :
  r = 10 →
  p = 2 / 5 →
  A_win = p * π * r^2 →
  A_win = 40 * π :=
by sorry

end NUMINAMATH_CALUDE_win_sector_area_l3946_394638


namespace NUMINAMATH_CALUDE_f_bounds_l3946_394685

def a : ℤ := 2001

def A : Set (ℤ × ℤ) :=
  {p | p.2 ≠ 0 ∧ 
       p.1 < 2 * a ∧ 
       (2 * p.2) ∣ (2 * a * p.1 - p.1^2 + p.2^2) ∧ 
       p.2^2 - p.1^2 + 2 * p.1 * p.2 ≤ 2 * a * (p.2 - p.1)}

def f (p : ℤ × ℤ) : ℚ :=
  (2 * a * p.1 - p.1^2 - p.1 * p.2) / p.2

theorem f_bounds :
  ∃ (min max : ℚ), min = 2 ∧ max = 3750 ∧
  ∀ p ∈ A, min ≤ f p ∧ f p ≤ max :=
sorry

end NUMINAMATH_CALUDE_f_bounds_l3946_394685


namespace NUMINAMATH_CALUDE_min_b_value_l3946_394641

/-- The function f(x) = x^2 + 2bx -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*b*x

/-- The function g(x) = |x-1| -/
def g (x : ℝ) : ℝ := |x - 1|

/-- The theorem stating the minimum value of b -/
theorem min_b_value :
  ∀ b : ℝ,
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 →
    f b x₁ - f b x₂ < g x₁ - g x₂) →
  b ≥ -1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_b_value_l3946_394641


namespace NUMINAMATH_CALUDE_prime_minister_stays_l3946_394662

/-- Represents the message on a piece of paper -/
inductive Message
| stay
| leave

/-- Represents a piece of paper with a message -/
structure Paper :=
  (message : Message)

/-- The portfolio containing two papers -/
structure Portfolio :=
  (paper1 : Paper)
  (paper2 : Paper)

/-- The state of the game after the prime minister's action -/
structure GameState :=
  (destroyed : Paper)
  (revealed : Paper)

/-- The prime minister's strategy -/
def primeMinisterStrategy (portfolio : Portfolio) : GameState :=
  { destroyed := portfolio.paper1,
    revealed := portfolio.paper2 }

/-- The king's claim about the portfolio -/
def kingsClaim (p : Portfolio) : Prop :=
  (p.paper1.message = Message.stay ∧ p.paper2.message = Message.leave) ∨
  (p.paper1.message = Message.leave ∧ p.paper2.message = Message.stay)

/-- The actual content of the portfolio -/
def actualPortfolio : Portfolio :=
  { paper1 := { message := Message.leave },
    paper2 := { message := Message.leave } }

theorem prime_minister_stays :
  ∀ (state : GameState),
  state = primeMinisterStrategy actualPortfolio →
  state.revealed.message = Message.leave →
  ∃ (claim : Paper), claim.message = Message.stay ∧ 
    (claim = state.destroyed ∨ kingsClaim actualPortfolio = False) :=
by sorry

end NUMINAMATH_CALUDE_prime_minister_stays_l3946_394662


namespace NUMINAMATH_CALUDE_amy_music_files_l3946_394627

theorem amy_music_files (total : ℕ) (video picture : ℝ) (h1 : total = 48) (h2 : video = 21.0) (h3 : picture = 23.0) :
  total - (video + picture) = 4 := by
sorry

end NUMINAMATH_CALUDE_amy_music_files_l3946_394627


namespace NUMINAMATH_CALUDE_max_δ_is_seven_l3946_394655

/-- The sequence a_n = 1 + n^3 -/
def a (n : ℕ) : ℕ := 1 + n^3

/-- The greatest common divisor of consecutive terms in the sequence -/
def δ (n : ℕ) : ℕ := Nat.gcd (a (n + 1)) (a n)

/-- The maximum value of δ_n is 7 -/
theorem max_δ_is_seven : ∃ (n : ℕ), δ n = 7 ∧ ∀ (m : ℕ), δ m ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_max_δ_is_seven_l3946_394655


namespace NUMINAMATH_CALUDE_circular_seating_arrangement_l3946_394683

/-- 
Given a circular arrangement of students, if the 6th position 
is exactly opposite to the 16th position, then there are 22 students in total.
-/
theorem circular_seating_arrangement (n : ℕ) : 
  (6 + n / 2 ≡ 16 [MOD n]) → n = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_circular_seating_arrangement_l3946_394683


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3946_394659

theorem isosceles_triangle_base_length 
  (eq_perimeter : ℝ) 
  (is_perimeter : ℝ) 
  (eq_side : ℝ) 
  (is_equal_side : ℝ) 
  (is_base : ℝ) 
  (vertex_angle : ℝ) 
  (h1 : eq_perimeter = 45) 
  (h2 : is_perimeter = 40) 
  (h3 : 3 * eq_side = eq_perimeter) 
  (h4 : 2 * is_equal_side + is_base = is_perimeter) 
  (h5 : is_equal_side = eq_side) 
  (h6 : 100 < vertex_angle ∧ vertex_angle < 120) : 
  is_base = 10 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3946_394659


namespace NUMINAMATH_CALUDE_distinct_polygons_count_l3946_394697

/-- The number of points marked on the circle -/
def n : ℕ := 15

/-- The total number of possible subsets of n points -/
def total_subsets : ℕ := 2^n

/-- The number of subsets with 0 elements -/
def subsets_0 : ℕ := Nat.choose n 0

/-- The number of subsets with 1 element -/
def subsets_1 : ℕ := Nat.choose n 1

/-- The number of subsets with 2 elements -/
def subsets_2 : ℕ := Nat.choose n 2

/-- The number of distinct convex polygons with 3 or more sides -/
def num_polygons : ℕ := total_subsets - subsets_0 - subsets_1 - subsets_2

theorem distinct_polygons_count : num_polygons = 32647 := by
  sorry

end NUMINAMATH_CALUDE_distinct_polygons_count_l3946_394697


namespace NUMINAMATH_CALUDE_eva_marks_difference_l3946_394633

/-- Represents Eva's marks in a single semester -/
structure SemesterMarks where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Represents Eva's marks for the entire year -/
structure YearMarks where
  first : SemesterMarks
  second : SemesterMarks

def total_marks (year : YearMarks) : ℕ :=
  year.first.maths + year.first.arts + year.first.science +
  year.second.maths + year.second.arts + year.second.science

theorem eva_marks_difference (eva : YearMarks) : 
  eva.second.maths = 80 →
  eva.second.arts = 90 →
  eva.second.science = 90 →
  eva.first.maths = eva.second.maths + 10 →
  eva.first.science = eva.second.science - (eva.second.science / 3) →
  total_marks eva = 485 →
  eva.second.arts - eva.first.arts = 75 := by
  sorry

end NUMINAMATH_CALUDE_eva_marks_difference_l3946_394633


namespace NUMINAMATH_CALUDE_remainder_problem_l3946_394647

theorem remainder_problem (greatest_divisor remainder_4521 : ℕ) 
  (h1 : greatest_divisor = 88)
  (h2 : remainder_4521 = 33)
  (h3 : ∃ q1 : ℕ, 3815 = greatest_divisor * q1 + (3815 % greatest_divisor))
  (h4 : ∃ q2 : ℕ, 4521 = greatest_divisor * q2 + remainder_4521) :
  3815 % greatest_divisor = 31 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l3946_394647


namespace NUMINAMATH_CALUDE_sum_of_digits_equation_l3946_394629

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- State the theorem
theorem sum_of_digits_equation : 
  ∃ (n : ℕ), n + sum_of_digits n = 2018 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_equation_l3946_394629


namespace NUMINAMATH_CALUDE_games_calculation_l3946_394679

def football_games : List Nat := [29, 35, 48, 43, 56, 36]
def baseball_games : List Nat := [15, 19, 23, 14, 18, 17]
def basketball_games : List Nat := [17, 21, 14, 32, 22, 27]

def total_games : Nat := football_games.sum + baseball_games.sum + basketball_games.sum

def average_games : Nat := total_games / 6

theorem games_calculation :
  total_games = 486 ∧ average_games = 81 := by
  sorry

end NUMINAMATH_CALUDE_games_calculation_l3946_394679


namespace NUMINAMATH_CALUDE_solve_for_s_l3946_394632

theorem solve_for_s (s t : ℤ) (eq1 : 8 * s + 7 * t = 156) (eq2 : s = t - 3) : s = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_s_l3946_394632


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l3946_394604

theorem baseball_card_value_decrease (initial_value : ℝ) (h_initial_positive : initial_value > 0) : 
  let first_year_value := initial_value * (1 - 0.5)
  let second_year_decrease_percent := (0.55 * initial_value - 0.5 * initial_value) / first_year_value
  second_year_decrease_percent = 0.1 := by
sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l3946_394604


namespace NUMINAMATH_CALUDE_same_terminal_side_negative_pi_sixth_same_terminal_side_l3946_394653

theorem same_terminal_side (θ₁ θ₂ : ℝ) : ∃ k : ℤ, θ₂ = θ₁ + 2 * π * k → 
  (θ₁.cos = θ₂.cos ∧ θ₁.sin = θ₂.sin) :=
by sorry

theorem negative_pi_sixth_same_terminal_side : 
  ∃ k : ℤ, (11 * π / 6 : ℝ) = -π / 6 + 2 * π * k :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_negative_pi_sixth_same_terminal_side_l3946_394653


namespace NUMINAMATH_CALUDE_max_product_with_851_l3946_394607

def digits : Finset Nat := {1, 5, 6, 8, 9}

def is_valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def three_digit_number (a b c : Nat) : Nat := 100 * a + 10 * b + c

def two_digit_number (d e : Nat) : Nat := 10 * d + e

theorem max_product_with_851 :
  ∀ a b c d e : Nat,
    is_valid_combination a b c d e →
    three_digit_number a b c * two_digit_number d e ≤ three_digit_number 8 5 1 * two_digit_number 9 6 :=
sorry

end NUMINAMATH_CALUDE_max_product_with_851_l3946_394607


namespace NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_l3946_394668

-- Define the propositions
def p (m : ℝ) : Prop := m = -1
def q (m : ℝ) : Prop := ∀ (x y : ℝ), (x - 1 = 0) ∧ (x + m^2 * y = 0) → 
  (∃ (k : ℝ), k ≠ 0 ∧ (1 * k = m^2) ∨ (1 * m^2 = -k))

-- Theorem statement
theorem p_neither_sufficient_nor_necessary :
  (∃ m : ℝ, p m ∧ ¬q m) ∧ (∃ m : ℝ, q m ∧ ¬p m) :=
sorry

end NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_l3946_394668


namespace NUMINAMATH_CALUDE_frannie_jumped_less_jump_difference_l3946_394694

/-- The number of times Frannie jumped -/
def frannies_jumps : ℕ := 53

/-- The number of times Meg jumped -/
def megs_jumps : ℕ := 71

/-- Frannie jumped fewer times than Meg -/
theorem frannie_jumped_less : frannies_jumps < megs_jumps := by sorry

/-- The difference in jumps between Meg and Frannie is 18 -/
theorem jump_difference : megs_jumps - frannies_jumps = 18 := by sorry

end NUMINAMATH_CALUDE_frannie_jumped_less_jump_difference_l3946_394694


namespace NUMINAMATH_CALUDE_hidden_primes_sum_l3946_394677

/-- A card with two numbers -/
structure Card where
  visible : Nat
  hidden : Nat

/-- Predicate to check if a number is prime -/
def isPrime (n : Nat) : Prop := sorry

/-- The sum of numbers on a card -/
def cardSum (c : Card) : Nat := c.visible + c.hidden

theorem hidden_primes_sum (c1 c2 c3 : Card) : 
  c1.visible = 17 →
  c2.visible = 26 →
  c3.visible = 41 →
  isPrime c1.hidden →
  isPrime c2.hidden →
  isPrime c3.hidden →
  cardSum c1 = cardSum c2 →
  cardSum c2 = cardSum c3 →
  c1.hidden + c2.hidden + c3.hidden = 198 := by
  sorry

end NUMINAMATH_CALUDE_hidden_primes_sum_l3946_394677


namespace NUMINAMATH_CALUDE_triangle_tangent_difference_bound_l3946_394664

theorem triangle_tangent_difference_bound (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Triangle is acute
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  b^2 - a^2 = a*c →  -- Given condition
  1 < (1 / Real.tan A - 1 / Real.tan B) ∧ (1 / Real.tan A - 1 / Real.tan B) < 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_tangent_difference_bound_l3946_394664


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3946_394684

def vector_a (m : ℝ) : Fin 2 → ℝ := ![3, -2*m]
def vector_b (m : ℝ) : Fin 2 → ℝ := ![8, 3*m]

def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

theorem perpendicular_vectors_m_value :
  ∀ m : ℝ, dot_product (vector_a m) (vector_b m) = 0 → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3946_394684


namespace NUMINAMATH_CALUDE_golf_cost_calculation_l3946_394660

/-- Proves that given the cost of one round of golf and the number of rounds that can be played,
    the total amount of money is correctly calculated. -/
theorem golf_cost_calculation (cost_per_round : ℕ) (num_rounds : ℕ) (total_money : ℕ) :
  cost_per_round = 80 →
  num_rounds = 5 →
  total_money = cost_per_round * num_rounds →
  total_money = 400 := by
sorry

end NUMINAMATH_CALUDE_golf_cost_calculation_l3946_394660


namespace NUMINAMATH_CALUDE_difference_of_squares_l3946_394648

theorem difference_of_squares (x y : ℝ) : 
  x + y = 15 → x - y = 10 → x^2 - y^2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3946_394648


namespace NUMINAMATH_CALUDE_triangle_inequality_l3946_394682

theorem triangle_inequality (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  2 < (a + b) / c + (b + c) / a + (c + a) / b - (a^3 + b^3 + c^3) / (a * b * c) ∧
  (a + b) / c + (b + c) / a + (c + a) / b - (a^3 + b^3 + c^3) / (a * b * c) ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3946_394682


namespace NUMINAMATH_CALUDE_speed_conversion_l3946_394686

theorem speed_conversion (speed_kmh : ℝ) (speed_ms : ℝ) : 
  speed_kmh = 1.2 → speed_ms = 1/3 → speed_kmh * (1000 / 3600) = speed_ms :=
by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l3946_394686


namespace NUMINAMATH_CALUDE_largest_non_representable_amount_l3946_394622

/-- Represents the denominations of coins in Limonia -/
def coin_denominations (n : ℕ) : List ℕ :=
  List.range (n + 1) |>.map (λ i => 5^(n - i) * 7^i)

/-- Determines if a number is representable using the given coin denominations -/
def is_representable (s n : ℕ) : Prop :=
  ∃ (coeffs : List ℕ), s = List.sum (List.zipWith (·*·) coeffs (coin_denominations n))

/-- The main theorem stating the largest non-representable amount -/
theorem largest_non_representable_amount (n : ℕ) :
  ∀ s : ℕ, s > 2 * 7^(n+1) - 3 * 5^(n+1) → is_representable s n ∧
  ¬is_representable (2 * 7^(n+1) - 3 * 5^(n+1)) n :=
sorry

end NUMINAMATH_CALUDE_largest_non_representable_amount_l3946_394622


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l3946_394661

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 108 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l3946_394661


namespace NUMINAMATH_CALUDE_polygon_sides_and_diagonals_l3946_394656

theorem polygon_sides_and_diagonals :
  ∀ n : ℕ,
  (180 * (n - 2) = 3 * 360 + 180) →
  (n = 9 ∧ (n * (n - 3)) / 2 = 27) :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_and_diagonals_l3946_394656
