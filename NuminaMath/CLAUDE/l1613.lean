import Mathlib

namespace bundle_sheets_value_l1613_161355

/-- The number of sheets in a bundle -/
def bundle_sheets : ℕ := 2

/-- The number of bundles of colored paper -/
def colored_bundles : ℕ := 3

/-- The number of bunches of white paper -/
def white_bunches : ℕ := 2

/-- The number of heaps of scrap paper -/
def scrap_heaps : ℕ := 5

/-- The number of sheets in a bunch -/
def bunch_sheets : ℕ := 4

/-- The number of sheets in a heap -/
def heap_sheets : ℕ := 20

/-- The total number of sheets removed -/
def total_sheets : ℕ := 114

theorem bundle_sheets_value :
  colored_bundles * bundle_sheets + white_bunches * bunch_sheets + scrap_heaps * heap_sheets = total_sheets :=
by sorry

end bundle_sheets_value_l1613_161355


namespace diamond_royalty_sanity_undetermined_l1613_161395

/-- Represents the sanity status of a person -/
inductive SanityStatus
  | Sane
  | Insane
  | Unknown

/-- Represents a royal person -/
structure RoyalPerson where
  name : String
  status : SanityStatus

/-- Represents a rumor about a person's sanity -/
structure Rumor where
  subject : RoyalPerson
  content : SanityStatus

/-- Represents the reliability of information -/
inductive Reliability
  | Reliable
  | Unreliable
  | Unknown

/-- The problem setup -/
def diamondRoyalty : Prop := ∃ (king queen : RoyalPerson) 
  (rumor : Rumor) (rumorReliability : Reliability),
  king.name = "King of Diamonds" ∧
  queen.name = "Queen of Diamonds" ∧
  rumor.subject = queen ∧
  rumor.content = SanityStatus.Insane ∧
  rumorReliability = Reliability.Unknown ∧
  (king.status = SanityStatus.Unknown ∨ 
   king.status = SanityStatus.Insane) ∧
  queen.status = SanityStatus.Unknown

/-- The theorem to be proved -/
theorem diamond_royalty_sanity_undetermined : 
  diamondRoyalty → 
  ∃ (king queen : RoyalPerson), 
    king.name = "King of Diamonds" ∧
    queen.name = "Queen of Diamonds" ∧
    king.status = SanityStatus.Unknown ∧
    queen.status = SanityStatus.Unknown :=
by
  sorry

end diamond_royalty_sanity_undetermined_l1613_161395


namespace simultaneous_equations_solution_l1613_161332

theorem simultaneous_equations_solution (m : ℝ) :
  (∃ (x y z : ℝ), y = m * x + z + 2 ∧ y = (3 * m - 2) * x + z + 5) ↔ m ≠ 1 := by
  sorry

end simultaneous_equations_solution_l1613_161332


namespace money_distribution_exists_l1613_161349

/-- Represents the money distribution problem with Ram, Gopal, Krishan, and Shekhar -/
def MoneyDistribution (x : ℚ) : Prop :=
  let ram_share := 7
  let gopal_share := 17
  let krishan_share := 17
  let shekhar_share := x
  let ram_money := 490
  let unit_value := ram_money / ram_share
  let gopal_shekhar_ratio := 2 / 1
  (gopal_share / shekhar_share = gopal_shekhar_ratio) ∧
  (shekhar_share * unit_value = 595)

/-- Theorem stating that there exists a valid money distribution satisfying all conditions -/
theorem money_distribution_exists : ∃ x, MoneyDistribution x := by
  sorry

end money_distribution_exists_l1613_161349


namespace express_y_in_terms_of_x_l1613_161359

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 := by
  sorry

end express_y_in_terms_of_x_l1613_161359


namespace solve_F_equation_l1613_161365

-- Define the function F
def F (a b c : ℚ) : ℚ := a * b^3 + c

-- Theorem statement
theorem solve_F_equation :
  ∃ a : ℚ, F a 3 10 = F a 5 20 ∧ a = -5/49 := by
  sorry

end solve_F_equation_l1613_161365


namespace cricket_team_win_percentage_l1613_161380

theorem cricket_team_win_percentage (matches_in_august : ℕ) (total_wins : ℕ) (overall_win_rate : ℚ) :
  matches_in_august = 120 →
  total_wins = 75 →
  overall_win_rate = 52/100 →
  (total_wins : ℚ) / (matches_in_august + (total_wins / overall_win_rate - matches_in_august)) = overall_win_rate →
  (matches_in_august * overall_win_rate - (total_wins - matches_in_august)) / matches_in_august = 17/40 :=
by sorry

end cricket_team_win_percentage_l1613_161380


namespace leftover_coins_value_l1613_161318

def quarters_per_roll : ℕ := 30
def dimes_per_roll : ℕ := 40
def sally_quarters : ℕ := 101
def sally_dimes : ℕ := 173
def ben_quarters : ℕ := 150
def ben_dimes : ℕ := 195
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

theorem leftover_coins_value :
  let total_quarters := sally_quarters + ben_quarters
  let total_dimes := sally_dimes + ben_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  let leftover_value := (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value
  leftover_value = 3.55 := by sorry

end leftover_coins_value_l1613_161318


namespace initial_overs_played_l1613_161326

/-- Proves the number of overs played initially in a cricket game -/
theorem initial_overs_played (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) (remaining_overs : ℝ) :
  target = 282 →
  initial_rate = 3.2 →
  required_rate = 8.333333333333334 →
  remaining_overs = 30 →
  ∃ (initial_overs : ℝ), 
    initial_overs * initial_rate + remaining_overs * required_rate = target ∧
    initial_overs = 10 :=
by sorry

end initial_overs_played_l1613_161326


namespace solution_set_solution_characterization_solution_equals_intervals_l1613_161370

theorem solution_set : Set ℝ := by
  sorry

theorem solution_characterization :
  solution_set = {x : ℝ | 2 ≤ |x - 3| ∧ |x - 3| ≤ 5 ∧ (x - 3)^2 ≤ 16} := by
  sorry

theorem solution_equals_intervals :
  solution_set = Set.Icc (-1) 1 ∪ Set.Icc 5 7 := by
  sorry

end solution_set_solution_characterization_solution_equals_intervals_l1613_161370


namespace number_added_to_55_l1613_161311

theorem number_added_to_55 : ∃ x : ℤ, 55 + x = 88 ∧ x = 33 := by sorry

end number_added_to_55_l1613_161311


namespace max_product_constraint_l1613_161350

theorem max_product_constraint (a b : ℝ) : 
  a > 0 → b > 0 → a + 2 * b = 10 → ∃ (m : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → x + 2 * y = 10 → x * y ≤ m ∧ a * b = m :=
by sorry

end max_product_constraint_l1613_161350


namespace min_distance_line_to_log_curve_l1613_161367

/-- The minimum distance between a point on y = x and a point on y = ln x is √2/2 -/
theorem min_distance_line_to_log_curve : 
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧ 
  ∀ (P Q : ℝ × ℝ), 
    (P.2 = P.1) → 
    (Q.2 = Real.log Q.1) → 
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by sorry

end min_distance_line_to_log_curve_l1613_161367


namespace minimum_cost_for_boxes_l1613_161352

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the problem parameters -/
structure ProblemParams where
  boxDims : BoxDimensions
  costPerBox : ℝ
  totalVolumeNeeded : ℝ

theorem minimum_cost_for_boxes (p : ProblemParams)
  (h1 : p.boxDims.length = 20)
  (h2 : p.boxDims.width = 20)
  (h3 : p.boxDims.height = 12)
  (h4 : p.costPerBox = 0.4)
  (h5 : p.totalVolumeNeeded = 2160000) :
  ⌈p.totalVolumeNeeded / boxVolume p.boxDims⌉ * p.costPerBox = 180 := by
  sorry

#check minimum_cost_for_boxes

end minimum_cost_for_boxes_l1613_161352


namespace sum_of_solutions_eq_five_halves_l1613_161388

theorem sum_of_solutions_eq_five_halves :
  let f : ℝ → ℝ := λ x => (4*x + 6)*(3*x - 12)
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 5/2 :=
by sorry

end sum_of_solutions_eq_five_halves_l1613_161388


namespace possible_r_value_l1613_161387

-- Define sets A and B
def A (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 * (p.1 - 1) + p.2 * (p.2 - 1) ≤ r}

def B (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ r^2}

-- Theorem statement
theorem possible_r_value :
  ∃ (r : ℝ), (A r ⊆ B r) ∧ (r = Real.sqrt 2 + 1) := by
  sorry

end possible_r_value_l1613_161387


namespace polygon_distance_inequality_l1613_161361

/-- A polygon in a plane -/
structure Polygon where
  vertices : List (Real × Real)
  is_closed : vertices.length > 2

/-- Calculate the perimeter of a polygon -/
def perimeter (F : Polygon) : Real :=
  sorry

/-- Calculate the sum of distances from a point to the vertices of a polygon -/
def sum_distances_to_vertices (X : Real × Real) (F : Polygon) : Real :=
  sorry

/-- Calculate the sum of distances from a point to the sidelines of a polygon -/
def sum_distances_to_sidelines (X : Real × Real) (F : Polygon) : Real :=
  sorry

/-- The main theorem -/
theorem polygon_distance_inequality (X : Real × Real) (F : Polygon) :
  let p := perimeter F
  let d := sum_distances_to_vertices X F
  let h := sum_distances_to_sidelines X F
  d^2 - h^2 ≥ p^2 / 4 := by
    sorry

end polygon_distance_inequality_l1613_161361


namespace root_range_l1613_161356

theorem root_range (k : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ x^2 + (k-5)*x + 9 = 0) ↔ 
  (-5 < k ∧ k < -3/2) := by
sorry

end root_range_l1613_161356


namespace line_segment_param_sum_squares_l1613_161366

/-- Given a line segment connecting (1,3) and (4,9) parameterized by x = pt + q and y = rt + s,
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1,3), prove that p^2 + q^2 + r^2 + s^2 = 55 -/
theorem line_segment_param_sum_squares (p q r s : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = p * t + q ∧ y = r * t + s) → -- parameterization
  (q = 1 ∧ s = 3) → -- t = 0 corresponds to (1,3)
  (p + q = 4 ∧ r + s = 9) → -- endpoint (4,9)
  p^2 + q^2 + r^2 + s^2 = 55 := by
  sorry

end line_segment_param_sum_squares_l1613_161366


namespace equality_of_fractions_l1613_161351

theorem equality_of_fractions (x y z l : ℝ) :
  (9 / (x + y + 1) = l / (x + z - 1)) ∧
  (l / (x + z - 1) = 13 / (z - y + 2)) →
  l = 22 := by
sorry

end equality_of_fractions_l1613_161351


namespace equation_solution_difference_l1613_161399

theorem equation_solution_difference : ∃ x₁ x₂ : ℝ,
  (x₁ + 3)^2 / (2*x₁ + 15) = 3 ∧
  (x₂ + 3)^2 / (2*x₂ + 15) = 3 ∧
  x₁ ≠ x₂ ∧
  x₂ - x₁ = 12 := by
  sorry

end equation_solution_difference_l1613_161399


namespace tree_planting_campaign_l1613_161316

theorem tree_planting_campaign (february_trees : ℕ) (planned_trees : ℕ) : 
  february_trees = planned_trees * 19 / 20 →
  (planned_trees * 11 / 10 : ℚ) = 
    (planned_trees * 11 / 10 : ℕ) ∧ planned_trees > 0 →
  ∃ (total_trees : ℕ), total_trees = planned_trees * 11 / 10 :=
by sorry

end tree_planting_campaign_l1613_161316


namespace pizza_price_proof_l1613_161308

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

end pizza_price_proof_l1613_161308


namespace remaining_problems_to_grade_l1613_161347

theorem remaining_problems_to_grade 
  (problems_per_paper : ℕ) 
  (total_papers : ℕ) 
  (graded_papers : ℕ) 
  (h1 : problems_per_paper = 15)
  (h2 : total_papers = 45)
  (h3 : graded_papers = 18)
  : (total_papers - graded_papers) * problems_per_paper = 405 :=
by sorry

end remaining_problems_to_grade_l1613_161347


namespace solution_replacement_concentration_l1613_161322

theorem solution_replacement_concentration 
  (initial_concentration : ℝ) 
  (final_concentration : ℝ) 
  (replaced_fraction : ℝ) 
  (replacement_concentration : ℝ) : 
  initial_concentration = 45 ∧ 
  final_concentration = 35 ∧ 
  replaced_fraction = 0.5 → 
  replacement_concentration = 25 := by
  sorry

end solution_replacement_concentration_l1613_161322


namespace skew_diagonals_properties_l1613_161336

/-- A cube with edge length 1 -/
structure UnitCube where
  edge_length : ℝ
  edge_length_eq_one : edge_length = 1

/-- Skew diagonals of two adjacent faces of a unit cube -/
structure SkewDiagonals (cube : UnitCube) where
  angle : ℝ
  distance : ℝ

/-- Theorem about the properties of skew diagonals in a unit cube -/
theorem skew_diagonals_properties (cube : UnitCube) :
  ∃ (sd : SkewDiagonals cube),
    sd.angle = Real.pi / 3 ∧ sd.distance = 1 / Real.sqrt 3 :=
sorry

end skew_diagonals_properties_l1613_161336


namespace hockey_games_in_season_l1613_161390

theorem hockey_games_in_season (games_per_month : ℕ) (months_in_season : ℕ) 
  (h1 : games_per_month = 13) (h2 : months_in_season = 14) : 
  games_per_month * months_in_season = 182 :=
by sorry

end hockey_games_in_season_l1613_161390


namespace complex_power_sum_l1613_161357

/-- Given that z = (i - 1) / √2, prove that z^100 + z^50 + 1 = -i -/
theorem complex_power_sum (z : ℂ) : z = (Complex.I - 1) / Real.sqrt 2 → z^100 + z^50 + 1 = -Complex.I := by
  sorry

end complex_power_sum_l1613_161357


namespace partition_equal_product_l1613_161358

def numbers : List Nat := [21, 22, 34, 39, 44, 45, 65, 76, 133, 153]

def target_product : Nat := 349188840

theorem partition_equal_product :
  ∃ (A B : List Nat),
    A.length = 5 ∧
    B.length = 5 ∧
    A ∪ B = numbers ∧
    A ∩ B = [] ∧
    A.prod = target_product ∧
    B.prod = target_product :=
by
  sorry

end partition_equal_product_l1613_161358


namespace ternary_1021_is_34_l1613_161334

def ternary_to_decimal (t : List Nat) : Nat :=
  List.foldl (fun acc d => acc * 3 + d) 0 t.reverse

theorem ternary_1021_is_34 :
  ternary_to_decimal [1, 0, 2, 1] = 34 := by
  sorry

end ternary_1021_is_34_l1613_161334


namespace trihedral_angle_inequality_l1613_161381

/-- Represents a trihedral angle with three planar angles -/
structure TrihedralAngle where
  α : ℝ
  β : ℝ
  γ : ℝ
  α_pos : 0 < α
  β_pos : 0 < β
  γ_pos : 0 < γ
  α_lt_pi : α < π
  β_lt_pi : β < π
  γ_lt_pi : γ < π

/-- The sum of any two planar angles in a trihedral angle is greater than the third angle -/
theorem trihedral_angle_inequality (t : TrihedralAngle) :
  t.α + t.β > t.γ ∧ t.α + t.γ > t.β ∧ t.β + t.γ > t.α := by
  sorry

end trihedral_angle_inequality_l1613_161381


namespace remainder_theorem_l1613_161382

/-- The polynomial of degree 2023 --/
def f (z : ℂ) : ℂ := z^2023 + 2

/-- The cubic polynomial --/
def g (z : ℂ) : ℂ := z^3 + z^2 + 1

/-- The theorem statement --/
theorem remainder_theorem :
  ∃ (P S : ℂ → ℂ), (∀ z, f z = g z * P z + S z) ∧
                   (∃ a b c, ∀ z, S z = a * z^2 + b * z + c) →
  ∀ z, S z = z + 2 :=
sorry

end remainder_theorem_l1613_161382


namespace painted_cells_20210_1505_l1613_161304

/-- The number of unique cells painted by two diagonals in a rectangle --/
def painted_cells (width height : ℕ) : ℕ :=
  let gcd := width.gcd height
  let subrect_width := width / gcd
  let subrect_height := height / gcd
  let cells_per_subrect := subrect_width + subrect_height - 1
  let total_cells := 2 * gcd * cells_per_subrect
  let overlap_cells := gcd
  total_cells - overlap_cells

/-- Theorem stating the number of painted cells in a 20210 × 1505 rectangle --/
theorem painted_cells_20210_1505 :
  painted_cells 20210 1505 = 42785 := by
  sorry

end painted_cells_20210_1505_l1613_161304


namespace email_difference_l1613_161345

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 9

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 10

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 7

/-- The difference between the number of emails Jack received in the morning and evening -/
theorem email_difference : morning_emails - evening_emails = 2 := by
  sorry

end email_difference_l1613_161345


namespace line_tangent_to_ellipse_l1613_161307

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

end line_tangent_to_ellipse_l1613_161307


namespace product_of_sum_and_cube_sum_l1613_161392

theorem product_of_sum_and_cube_sum (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (cube_sum_eq : x^3 + y^3 = 172) : 
  x * y = 41.4 := by
sorry

end product_of_sum_and_cube_sum_l1613_161392


namespace faster_train_speed_l1613_161337

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

end faster_train_speed_l1613_161337


namespace stratified_sampling_male_count_l1613_161384

theorem stratified_sampling_male_count 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (female_in_sample : ℕ) 
  (h1 : total_students = 1200) 
  (h2 : sample_size = 30) 
  (h3 : female_in_sample = 14) :
  let male_in_sample := sample_size - female_in_sample
  let male_in_grade := (male_in_sample : ℚ) / sample_size * total_students
  male_in_grade = 640 := by
sorry

end stratified_sampling_male_count_l1613_161384


namespace circle_equation_l1613_161385

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def isInFirstQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def intersectsXAxisAt (c : Circle) (p1 p2 : ℝ × ℝ) : Prop :=
  p1.2 = 0 ∧ p2.2 = 0 ∧
  (c.center.1 - p1.1)^2 + (c.center.2 - p1.2)^2 = c.radius^2 ∧
  (c.center.1 - p2.1)^2 + (c.center.2 - p2.2)^2 = c.radius^2

def isTangentToLine (c : Circle) : Prop :=
  let d := |c.center.1 - c.center.2 + 1| / Real.sqrt 2
  d = c.radius

-- Theorem statement
theorem circle_equation (c : Circle) :
  isInFirstQuadrant c.center →
  intersectsXAxisAt c (1, 0) (3, 0) →
  isTangentToLine c →
  ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 2 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
sorry

end circle_equation_l1613_161385


namespace least_possible_area_of_square_l1613_161375

/-- Given a square with sides measured to the nearest centimeter as 7 cm,
    the least possible actual area of the square is 42.25 cm². -/
theorem least_possible_area_of_square (side_length : ℝ) : 
  (6.5 ≤ side_length) ∧ (side_length < 7.5) → side_length ^ 2 ≥ 42.25 := by
  sorry

end least_possible_area_of_square_l1613_161375


namespace team_score_l1613_161398

/-- Given a basketball team where each person scores 2 points and there are 9 people playing,
    the total points scored by the team is 18. -/
theorem team_score (points_per_person : ℕ) (num_players : ℕ) (total_points : ℕ) :
  points_per_person = 2 →
  num_players = 9 →
  total_points = points_per_person * num_players →
  total_points = 18 := by
  sorry

end team_score_l1613_161398


namespace divisibility_proof_l1613_161333

theorem divisibility_proof :
  ∃ (n : ℕ), (425897 + n) % 456 = 0 ∧ n = 47 := by
  sorry

end divisibility_proof_l1613_161333


namespace complement_intersection_theorem_l1613_161319

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 5}

-- Define set A
def A : Set ℝ := {x | -3 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}

-- Define the result set
def result : Set ℝ := {x | -5 ≤ x ∧ x ≤ -3}

-- Theorem statement
theorem complement_intersection_theorem :
  (Set.compl A ∩ B) = result := by sorry

end complement_intersection_theorem_l1613_161319


namespace triangle_inequality_l1613_161376

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  let s := (a + b + c) / 2
  (2*a*(2*a - s))/(b + c) + (2*b*(2*b - s))/(c + a) + (2*c*(2*c - s))/(a + b) ≥ s := by
  sorry

end triangle_inequality_l1613_161376


namespace sqrt_sum_equals_five_sqrt_two_over_two_l1613_161300

theorem sqrt_sum_equals_five_sqrt_two_over_two :
  Real.sqrt 8 + Real.sqrt (1/2) = (5 * Real.sqrt 2) / 2 := by
  sorry

end sqrt_sum_equals_five_sqrt_two_over_two_l1613_161300


namespace negation_of_existence_negation_of_proposition_l1613_161329

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x > 0, p x) ↔ ∀ x > 0, ¬ p x := by sorry

theorem negation_of_proposition :
  (¬ ∃ x > 0, Real.log x > x - 1) ↔ (∀ x > 0, Real.log x ≤ x - 1) := by sorry

end negation_of_existence_negation_of_proposition_l1613_161329


namespace right_triangle_construction_l1613_161341

/-- Given a length b (representing one leg) and a length c (representing the projection of the other leg onto the hypotenuse), a right triangle can be constructed. -/
theorem right_triangle_construction (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  ∃ (a x : ℝ), a > 0 ∧ x > 0 ∧ x + c = a ∧ b^2 = x * a :=
by sorry

end right_triangle_construction_l1613_161341


namespace fraction_simplification_and_division_l1613_161309

theorem fraction_simplification_and_division (a x : ℝ) : 
  (a = -2 → (a^2 + a) / (a^2 - 3*a) / ((a^2 - 1) / (a - 3)) - 1 / (a + 1) = 2/3) ∧
  ((x^2 - 1) / (x - 4) / ((x + 1) / (4 - x)) = 1 - x) :=
by sorry

end fraction_simplification_and_division_l1613_161309


namespace acute_angles_subset_first_quadrant_l1613_161323

-- Define the sets M, N, and P
def M : Set ℝ := {θ | 0 < θ ∧ θ < 90}
def N : Set ℝ := {θ | θ < 90}
def P : Set ℝ := {θ | ∃ k : ℤ, k * 360 < θ ∧ θ < k * 360 + 90}

-- Theorem to prove
theorem acute_angles_subset_first_quadrant : M ⊆ P := by
  sorry

end acute_angles_subset_first_quadrant_l1613_161323


namespace at_least_one_third_l1613_161374

theorem at_least_one_third (a b c : ℝ) (h : a + b + c = 1) :
  (a ≥ 1/3) ∨ (b ≥ 1/3) ∨ (c ≥ 1/3) := by
  sorry

end at_least_one_third_l1613_161374


namespace apple_difference_l1613_161371

theorem apple_difference (martha_apples harry_apples : ℕ) 
  (h1 : martha_apples = 68)
  (h2 : harry_apples = 19)
  (h3 : ∃ tim_apples : ℕ, tim_apples = 2 * harry_apples ∧ tim_apples < martha_apples) :
  martha_apples - (2 * harry_apples) = 30 := by
sorry

end apple_difference_l1613_161371


namespace square_sum_geq_product_sum_l1613_161306

theorem square_sum_geq_product_sum (a b c : ℝ) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end square_sum_geq_product_sum_l1613_161306


namespace fractional_equation_solution_range_l1613_161369

theorem fractional_equation_solution_range (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x ≠ 1 ∧ x ≠ 1/2 ∧ 2/(x-1) = m/(2*x-1)) ↔ 
  (m > 4 ∨ m < 2) ∧ m ≠ 0 := by
sorry

end fractional_equation_solution_range_l1613_161369


namespace line_plane_parallelism_l1613_161383

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallelLines : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)
variable (containedIn : Line → Plane → Prop)

-- State the theorem
theorem line_plane_parallelism 
  (a b : Line) (α : Plane) 
  (h1 : parallelLinePlane a α) 
  (h2 : parallelLines a b) 
  (h3 : ¬ containedIn b α) : 
  parallelLinePlane b α :=
sorry

end line_plane_parallelism_l1613_161383


namespace average_weight_of_children_l1613_161396

theorem average_weight_of_children (num_boys num_girls : ℕ) 
  (avg_weight_boys avg_weight_girls : ℚ) :
  num_boys = 8 →
  num_girls = 5 →
  avg_weight_boys = 160 →
  avg_weight_girls = 130 →
  (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 148 :=
by sorry

end average_weight_of_children_l1613_161396


namespace eight_holes_when_unfolded_l1613_161313

/-- Represents a fold on the triangular paper -/
structure Fold where
  vertex : ℕ
  midpoint : ℕ

/-- Represents the triangular paper with its folds and holes -/
structure TriangularPaper where
  folds : List Fold
  holes : ℕ

/-- Performs a fold on the triangular paper -/
def applyFold (paper : TriangularPaper) (fold : Fold) : TriangularPaper :=
  { folds := fold :: paper.folds, holes := paper.holes }

/-- Punches holes in the folded paper -/
def punchHoles (paper : TriangularPaper) (n : ℕ) : TriangularPaper :=
  { folds := paper.folds, holes := paper.holes + n }

/-- Unfolds the paper and calculates the total number of holes -/
def unfold (paper : TriangularPaper) : ℕ :=
  match paper.folds.length with
  | 0 => paper.holes
  | 1 => 2 * paper.holes
  | _ => 4 * paper.holes

/-- Theorem stating that folding an equilateral triangle twice and punching two holes results in eight holes when unfolded -/
theorem eight_holes_when_unfolded (initialPaper : TriangularPaper) : 
  let firstFold := Fold.mk 1 2
  let secondFold := Fold.mk 3 4
  let foldedPaper := applyFold (applyFold initialPaper firstFold) secondFold
  let punchedPaper := punchHoles foldedPaper 2
  unfold punchedPaper = 8 := by
  sorry


end eight_holes_when_unfolded_l1613_161313


namespace paula_and_olive_spend_twenty_l1613_161373

/-- The total amount spent by Paula and Olive at the kiddy gift shop -/
def total_spent (bracelet_price keychain_price coloring_book_price : ℕ)
  (paula_bracelets paula_keychains : ℕ)
  (olive_coloring_books olive_bracelets : ℕ) : ℕ :=
  (paula_bracelets * bracelet_price + paula_keychains * keychain_price) +
  (olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price)

/-- Theorem stating that Paula and Olive spend $20 in total -/
theorem paula_and_olive_spend_twenty :
  total_spent 4 5 3 2 1 1 1 = 20 := by
  sorry

end paula_and_olive_spend_twenty_l1613_161373


namespace prime_implication_l1613_161315

theorem prime_implication (p : ℕ) (hp : Prime p) (h8p2_1 : Prime (8 * p^2 + 1)) :
  Prime (8 * p^2 - p + 2) := by
  sorry

end prime_implication_l1613_161315


namespace profit_equation_store_profit_equation_l1613_161325

/-- Represents the profit equation for a store selling goods -/
theorem profit_equation (initial_price initial_cost initial_volume : ℕ) 
                        (price_increase : ℕ) (volume_decrease_rate : ℕ) 
                        (profit_increase : ℕ) : Prop :=
  let new_price := initial_price + price_increase
  let new_volume := initial_volume - volume_decrease_rate * price_increase
  let profit_per_unit := new_price - initial_cost
  profit_per_unit * new_volume = initial_volume * (initial_price - initial_cost) + profit_increase

/-- The specific profit equation for the given problem -/
theorem store_profit_equation (x : ℕ) : 
  profit_equation 36 20 200 x 5 1200 ↔ (x + 16) * (200 - 5 * x) = 1200 :=
sorry

end profit_equation_store_profit_equation_l1613_161325


namespace constant_zero_arithmetic_not_geometric_l1613_161327

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def constant_zero_sequence : ℕ → ℝ :=
  λ _ => 0

theorem constant_zero_arithmetic_not_geometric :
  is_arithmetic_sequence constant_zero_sequence ∧
  ¬ is_geometric_sequence constant_zero_sequence :=
by sorry

end constant_zero_arithmetic_not_geometric_l1613_161327


namespace tomato_seeds_planted_l1613_161386

/-- The total number of tomato seeds planted by Mike, Ted, and Sarah -/
def total_seeds (mike_morning mike_afternoon ted_morning ted_afternoon sarah_morning sarah_afternoon : ℕ) : ℕ :=
  mike_morning + mike_afternoon + ted_morning + ted_afternoon + sarah_morning + sarah_afternoon

theorem tomato_seeds_planted :
  ∃ (mike_morning mike_afternoon ted_morning ted_afternoon sarah_morning sarah_afternoon : ℕ),
    mike_morning = 50 ∧
    ted_morning = 2 * mike_morning ∧
    sarah_morning = mike_morning + 30 ∧
    mike_afternoon = 60 ∧
    ted_afternoon = mike_afternoon - 20 ∧
    sarah_afternoon = sarah_morning + 20 ∧
    total_seeds mike_morning mike_afternoon ted_morning ted_afternoon sarah_morning sarah_afternoon = 430 :=
by sorry

end tomato_seeds_planted_l1613_161386


namespace coin_count_proof_l1613_161397

/-- Represents the total number of coins given the following conditions:
  - There are coins of 20 paise and 25 paise denominations
  - The total value of all coins is 7100 paise (71 Rs)
  - There are 200 coins of 20 paise denomination
-/
def totalCoins (totalValue : ℕ) (value20p : ℕ) (value25p : ℕ) (count20p : ℕ) : ℕ :=
  count20p + (totalValue - count20p * value20p) / value25p

theorem coin_count_proof :
  totalCoins 7100 20 25 200 = 324 := by
  sorry

end coin_count_proof_l1613_161397


namespace circular_fortress_volume_l1613_161343

theorem circular_fortress_volume : 
  let base_circumference : ℝ := 48
  let height : ℝ := 11
  let π : ℝ := 3
  let radius := base_circumference / (2 * π)
  let volume := π * radius^2 * height
  volume = 2112 := by
  sorry

end circular_fortress_volume_l1613_161343


namespace roots_quadratic_equation_l1613_161321

theorem roots_quadratic_equation (m n : ℝ) : 
  (m^2 - 2*m + 1 = 0) → (n^2 - 2*n + 1 = 0) → 
  (m + n) / (m^2 - 2*m) = -2 := by
  sorry

end roots_quadratic_equation_l1613_161321


namespace postcard_selection_ways_l1613_161389

theorem postcard_selection_ways : 
  let total_teachers : ℕ := 4
  let type_a_cards : ℕ := 2
  let type_b_cards : ℕ := 3
  let total_cards_to_select : ℕ := 4
  ∃ (ways : ℕ), ways = 10 ∧ 
    ways = (Nat.choose total_teachers type_a_cards) + 
           (Nat.choose total_teachers (total_teachers - type_a_cards)) :=
by sorry

end postcard_selection_ways_l1613_161389


namespace caffeine_in_coffee_l1613_161377

/-- The amount of caffeine in a cup of coffee -/
def caffeine_per_cup : ℝ := 80

/-- Lisa's daily caffeine limit in milligrams -/
def daily_limit : ℝ := 200

/-- The number of cups Lisa drinks -/
def cups_drunk : ℝ := 3

/-- The amount Lisa exceeds her limit by in milligrams -/
def excess_amount : ℝ := 40

theorem caffeine_in_coffee :
  caffeine_per_cup * cups_drunk = daily_limit + excess_amount :=
by sorry

end caffeine_in_coffee_l1613_161377


namespace percentage_sum_proof_l1613_161353

theorem percentage_sum_proof : (0.08 * 24) + (0.10 * 40) = 5.92 := by
  sorry

end percentage_sum_proof_l1613_161353


namespace man_son_age_difference_l1613_161394

/-- Represents the age difference between a man and his son -/
def ageDifference (manAge sonAge : ℕ) : ℕ := manAge - sonAge

theorem man_son_age_difference :
  ∀ (manAge sonAge : ℕ),
  sonAge = 14 →
  manAge + 2 = 2 * (sonAge + 2) →
  ageDifference manAge sonAge = 16 := by
sorry

end man_son_age_difference_l1613_161394


namespace blue_balls_count_l1613_161344

theorem blue_balls_count (red green : ℕ) (p : ℚ) (blue : ℕ) : 
  red = 4 → 
  green = 2 → 
  p = 4/30 → 
  (red / (red + blue + green : ℚ)) * ((red - 1) / (red + blue + green - 1 : ℚ)) = p → 
  blue = 4 :=
sorry

end blue_balls_count_l1613_161344


namespace complementary_angles_difference_l1613_161339

theorem complementary_angles_difference (x y : ℝ) : 
  x + y = 90 →  -- angles are complementary
  x = 4 * y →   -- ratio of angles is 4:1
  |x - y| = 54  -- absolute difference between angles is 54°
  := by sorry

end complementary_angles_difference_l1613_161339


namespace polynomial_factor_coefficients_l1613_161317

theorem polynomial_factor_coefficients :
  ∀ (a b : ℤ),
  (∃ (c d : ℤ), ∀ (x : ℚ),
    a * x^4 + b * x^3 + 32 * x^2 - 16 * x + 6 = (3 * x^2 - 2 * x + 1) * (c * x^2 + d * x + 6)) →
  a = 18 ∧ b = -24 := by
  sorry

end polynomial_factor_coefficients_l1613_161317


namespace sqrt_3_sum_square_l1613_161363

theorem sqrt_3_sum_square (x y : ℝ) : x = Real.sqrt 3 + 1 → y = Real.sqrt 3 - 1 → x^2 + x*y + y^2 = 10 := by
  sorry

end sqrt_3_sum_square_l1613_161363


namespace fuel_in_truck_is_38_l1613_161372

/-- Calculates the amount of fuel already in a truck given the total capacity,
    amount spent, change received, and cost per liter. -/
def fuel_already_in_truck (total_capacity : ℕ) (amount_spent : ℕ) (change : ℕ) (cost_per_liter : ℕ) : ℕ :=
  total_capacity - (amount_spent - change) / cost_per_liter

/-- Proves that given the specific conditions, the amount of fuel already in the truck is 38 liters. -/
theorem fuel_in_truck_is_38 :
  fuel_already_in_truck 150 350 14 3 = 38 := by
  sorry

#eval fuel_already_in_truck 150 350 14 3

end fuel_in_truck_is_38_l1613_161372


namespace double_age_in_three_years_l1613_161354

/-- The number of years from now when Tully will be twice as old as Kate -/
def years_until_double_age (tully_age_last_year : ℕ) (kate_age_now : ℕ) : ℕ :=
  3

theorem double_age_in_three_years (tully_age_last_year kate_age_now : ℕ) 
  (h1 : tully_age_last_year = 60) (h2 : kate_age_now = 29) :
  years_until_double_age tully_age_last_year kate_age_now = 3 := by
  sorry

end double_age_in_three_years_l1613_161354


namespace geometric_sequence_increasing_condition_l1613_161378

/-- A geometric sequence with first term a₁ and common ratio q -/
def GeometricSequence (a₁ q : ℝ) : ℕ → ℝ :=
  fun n ↦ a₁ * q^(n - 1)

theorem geometric_sequence_increasing_condition (a₁ q : ℝ) :
  (a₁ < 0 ∧ 0 < q ∧ q < 1 →
    ∀ n : ℕ, n > 0 → GeometricSequence a₁ q (n + 1) > GeometricSequence a₁ q n) ∧
  (∃ a₁' q' : ℝ, (∀ n : ℕ, n > 0 → GeometricSequence a₁' q' (n + 1) > GeometricSequence a₁' q' n) ∧
    ¬(a₁' < 0 ∧ 0 < q' ∧ q' < 1)) :=
by sorry

end geometric_sequence_increasing_condition_l1613_161378


namespace rectangle_area_l1613_161328

theorem rectangle_area (c d w : ℝ) (h1 : w > 0) (h2 : w + 3 > w) 
  (h3 : (c + d)^2 = w^2 + (w + 3)^2) : w * (w + 3) = w^2 + 3*w := by
  sorry

end rectangle_area_l1613_161328


namespace sequence_errors_l1613_161362

-- Part (a)
def sequence_a (x y z : ℝ) : Prop :=
  (225 / 25 + 75 = 100 - 16) ∧
  (25 * (9 / (1 + 3)) = 84) ∧
  (25 * 12 = 84) ∧
  (25 = 7)

-- Part (b)
def sequence_b (x y z : ℝ) : Prop :=
  (5005 - 2002 = 35 * 143 - 143 * 14) ∧
  (5005 - 35 * 143 = 2002 - 143 * 14) ∧
  (5 * (1001 - 7 * 143) = 2 * (1001 - 7 * 143)) ∧
  (5 = 2)

theorem sequence_errors :
  ¬(∃ x y z : ℝ, sequence_a x y z) ∧
  ¬(∃ x y z : ℝ, sequence_b x y z) :=
sorry

end sequence_errors_l1613_161362


namespace min_tiles_needed_l1613_161360

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- The dimensions of the tile -/
def tileDimensions : Dimensions := ⟨2, 5⟩

/-- The dimensions of the floor in feet -/
def floorDimensionsFeet : Dimensions := ⟨3, 4⟩

/-- The dimensions of the floor in inches -/
def floorDimensionsInches : Dimensions :=
  ⟨feetToInches floorDimensionsFeet.length, feetToInches floorDimensionsFeet.width⟩

/-- Calculates the number of tiles needed to cover the floor -/
def tilesNeeded : ℕ :=
  (area floorDimensionsInches + area tileDimensions - 1) / area tileDimensions

theorem min_tiles_needed :
  tilesNeeded = 173 := by
  sorry

end min_tiles_needed_l1613_161360


namespace inequality_proof_l1613_161302

theorem inequality_proof (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  x^2*y + x*y^2 + 1 ≤ x^2*y^2 + x + y := by
  sorry

end inequality_proof_l1613_161302


namespace line_segment_both_symmetric_l1613_161305

-- Define the shapes
inductive Shape
| EquilateralTriangle
| IsoscelesTriangle
| Parallelogram
| LineSegment

-- Define symmetry properties
def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => true
  | Shape.LineSegment => true
  | _ => false

def isAxiallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => true
  | Shape.IsoscelesTriangle => true
  | Shape.LineSegment => true
  | _ => false

-- Theorem statement
theorem line_segment_both_symmetric :
  ∀ s : Shape, (isCentrallySymmetric s ∧ isAxiallySymmetric s) ↔ s = Shape.LineSegment :=
by sorry

end line_segment_both_symmetric_l1613_161305


namespace student_council_committees_l1613_161312

theorem student_council_committees (x : ℕ) : 
  (x.choose 3 = 20) → (x.choose 4 = 15) := by sorry

end student_council_committees_l1613_161312


namespace james_ali_difference_l1613_161301

def total_amount : ℕ := 250
def james_amount : ℕ := 145

theorem james_ali_difference :
  ∀ (ali_amount : ℕ),
  ali_amount + james_amount = total_amount →
  james_amount > ali_amount →
  james_amount - ali_amount = 40 :=
by sorry

end james_ali_difference_l1613_161301


namespace smallest_with_twelve_divisors_l1613_161346

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_with_twelve_divisors : 
  ∀ n : ℕ, n > 0 → divisor_count n = 12 → n ≥ 96 :=
by sorry

end smallest_with_twelve_divisors_l1613_161346


namespace ninth_term_of_sequence_l1613_161368

def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem ninth_term_of_sequence (a d : ℝ) :
  arithmetic_sequence a d 3 = 20 →
  arithmetic_sequence a d 6 = 26 →
  arithmetic_sequence a d 9 = 32 := by
sorry

end ninth_term_of_sequence_l1613_161368


namespace characterize_f_l1613_161310

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ a b c : ℕ, a ≥ 2 → b ≥ 2 → c ≥ 2 →
    (f^[a*b*c - a] (a*b*c)) + (f^[a*b*c - b] (a*b*c)) + (f^[a*b*c - c] (a*b*c)) = a + b + c

theorem characterize_f (f : ℕ → ℕ) (h : is_valid_f f) :
  ∀ n : ℕ, n ≥ 3 → f n = n - 1 :=
sorry

end characterize_f_l1613_161310


namespace andrea_sod_rectangles_l1613_161364

/-- Calculates the number of sod rectangles needed for a given area -/
def sodRectanglesNeeded (length width : ℕ) : ℕ :=
  (length * width + 11) / 12

/-- The total number of sod rectangles needed for Andrea's backyard -/
def totalSodRectangles : ℕ :=
  sodRectanglesNeeded 35 42 +
  sodRectanglesNeeded 55 86 +
  sodRectanglesNeeded 20 50 +
  sodRectanglesNeeded 48 66

theorem andrea_sod_rectangles :
  totalSodRectangles = 866 := by
  sorry

end andrea_sod_rectangles_l1613_161364


namespace inverse_f_at_neg_1_l1613_161338

-- Define f as a function with an inverse
variable (f : ℝ → ℝ)
variable (hf : Function.Bijective f)

-- Define the condition that f(2) = -1
axiom f_at_2 : f 2 = -1

-- State the theorem to be proved
theorem inverse_f_at_neg_1 : Function.invFun f (-1) = 2 := by sorry

end inverse_f_at_neg_1_l1613_161338


namespace pencil_distribution_l1613_161314

/-- The number of ways to distribute n identical objects among k people,
    where each person must receive at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

theorem pencil_distribution :
  distribute 7 4 = 52 := by sorry

end pencil_distribution_l1613_161314


namespace average_difference_l1613_161379

theorem average_difference (x : ℝ) : (10 + x + 50) / 3 = (20 + 40 + 6) / 3 + 8 ↔ x = 30 := by
  sorry

end average_difference_l1613_161379


namespace hyperbola_parameters_l1613_161324

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Theorem: Given a hyperbola with specific asymptote and focus, determine its parameters -/
theorem hyperbola_parameters 
  (h : Hyperbola) 
  (h_asymptote : b / a = 2) 
  (h_focus : Real.sqrt (a^2 + b^2) = Real.sqrt 5) : 
  h.a = 1 ∧ h.b = 2 := by
  sorry

end hyperbola_parameters_l1613_161324


namespace propositions_p_and_q_true_l1613_161342

theorem propositions_p_and_q_true : (∃ x₀ : ℝ, x₀^2 < x₀) ∧ (∀ x : ℝ, x^2 - x + 1 > 0) := by
  sorry

end propositions_p_and_q_true_l1613_161342


namespace fraction_transformation_l1613_161340

theorem fraction_transformation (n : ℚ) : (4 + n) / (7 + n) = 2 / 3 → n = 2 :=
by sorry

end fraction_transformation_l1613_161340


namespace ellipse_foci_distance_l1613_161320

/-- Given an ellipse with equation 9x^2 + 16y^2 = 144, the distance between its foci is 2√7 -/
theorem ellipse_foci_distance (x y : ℝ) :
  (9 * x^2 + 16 * y^2 = 144) → (∃ f₁ f₂ : ℝ × ℝ, 
    f₁ ≠ f₂ ∧ 
    (∀ p : ℝ × ℝ, 9 * p.1^2 + 16 * p.2^2 = 144 → 
      Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + 
      Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = 8) ∧
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 2 * Real.sqrt 7) :=
by sorry

end ellipse_foci_distance_l1613_161320


namespace max_quadrilateral_area_l1613_161393

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Points on the x-axis that the ellipse passes through -/
def P : ℝ × ℝ := (1, 0)
def Q : ℝ × ℝ := (-1, 0)

/-- A function representing parallel lines passing through a point with slope k -/
def parallelLine (p : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - p.1) + p.2

/-- The quadrilateral formed by the intersection of parallel lines and the ellipse -/
def quadrilateral (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, ellipse x y ∧ (parallelLine P k x y ∨ parallelLine Q k x y)}

/-- The area of the quadrilateral as a function of the slope k -/
noncomputable def quadrilateralArea (k : ℝ) : ℝ :=
  sorry  -- Actual computation of area

theorem max_quadrilateral_area :
  ∃ (max_area : ℝ), max_area = 2 * Real.sqrt 3 ∧
    ∀ k, quadrilateralArea k ≤ max_area :=
  sorry

#check max_quadrilateral_area

end max_quadrilateral_area_l1613_161393


namespace combination_equations_l1613_161348

def A (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem combination_equations :
  (∃! x : ℕ+, 3 * (A x.val 3) = 2 * (A (x.val + 1) 2) + 6 * (A x.val 2)) ∧
  (∃ x : ℕ+, x = 1 ∨ x = 2) ∧ (∀ x : ℕ+, A 8 x.val = A 8 (5 * x.val - 4) → x = 1 ∨ x = 2) :=
by sorry

end combination_equations_l1613_161348


namespace custom_ops_theorem_l1613_161303

/-- Custom addition operation for natural numbers -/
def customAdd (a b : ℕ) : ℕ := a + b + 1

/-- Custom multiplication operation for natural numbers -/
def customMul (a b : ℕ) : ℕ := a * b - 1

/-- Theorem stating that (5 ⊕ 7) ⊕ (2 ⊗ 4) = 21 -/
theorem custom_ops_theorem : customAdd (customAdd 5 7) (customMul 2 4) = 21 := by
  sorry

end custom_ops_theorem_l1613_161303


namespace locker_count_l1613_161330

/-- The cost of a single digit in dollars -/
def digit_cost : ℚ := 0.03

/-- The total cost of labeling all lockers in dollars -/
def total_cost : ℚ := 206.91

/-- Calculates the cost of labeling lockers from 1 to n -/
def labeling_cost (n : ℕ) : ℚ :=
  let one_digit := min n 9
  let two_digit := min (n - 9) 90
  let three_digit := min (n - 99) 900
  let four_digit := max (n - 999) 0
  digit_cost * (one_digit + 2 * two_digit + 3 * three_digit + 4 * four_digit)

/-- The theorem stating that 2001 lockers can be labeled with the given total cost -/
theorem locker_count : labeling_cost 2001 = total_cost := by
  sorry

end locker_count_l1613_161330


namespace min_value_expression_min_value_achievable_l1613_161331

theorem min_value_expression (x y : ℝ) (h1 : y > 0) (h2 : y = -1/x + 1) :
  2*x + 1/y ≥ 2*Real.sqrt 2 + 3 :=
by
  sorry

theorem min_value_achievable :
  ∃ (x y : ℝ), y > 0 ∧ y = -1/x + 1 ∧ 2*x + 1/y = 2*Real.sqrt 2 + 3 :=
by
  sorry

end min_value_expression_min_value_achievable_l1613_161331


namespace triangle_altitude_tangent_relation_l1613_161391

-- Define a triangle with its properties
structure Triangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles
  A : ℝ
  B : ℝ
  C : ℝ
  -- Altitudes
  DA' : ℝ
  EB' : ℝ
  FC' : ℝ
  -- Conditions
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_altitudes : 0 < DA' ∧ 0 < EB' ∧ 0 < FC'
  angle_sum : A + B + C = π
  -- Additional conditions may be needed to fully define a valid triangle

-- Theorem statement
theorem triangle_altitude_tangent_relation (t : Triangle) :
  t.a / t.DA' + t.b / t.EB' + t.c / t.FC' = 2 * Real.tan t.A * Real.tan t.B * Real.tan t.C := by
  sorry


end triangle_altitude_tangent_relation_l1613_161391


namespace remainder_theorem_l1613_161335

theorem remainder_theorem (n : ℤ) (h : n % 5 = 3) : (7 * n + 4) % 5 = 0 := by
  sorry

end remainder_theorem_l1613_161335
