import Mathlib

namespace inequality_range_l2814_281464

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x + 1| > a) → a < 4 := by
sorry

end inequality_range_l2814_281464


namespace intersection_of_A_and_B_l2814_281454

open Set

def A : Set ℝ := {x | 1 < x^2 ∧ x^2 < 4}
def B : Set ℝ := {x | x - 1 ≥ 0}

theorem intersection_of_A_and_B : A ∩ B = Ioo 1 2 := by sorry

end intersection_of_A_and_B_l2814_281454


namespace prime_extension_l2814_281443

theorem prime_extension (n : ℕ+) (h : ∀ k : ℕ, 0 ≤ k ∧ k < Real.sqrt ((n + 2) / 3) → Nat.Prime (k^2 + k + n + 2)) :
  ∀ k : ℕ, Real.sqrt ((n + 2) / 3) ≤ k ∧ k ≤ n → Nat.Prime (k^2 + k + n + 2) := by
  sorry

end prime_extension_l2814_281443


namespace lcm_of_9_12_15_l2814_281472

theorem lcm_of_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end lcm_of_9_12_15_l2814_281472


namespace only_7_3_1_wins_for_second_player_l2814_281416

/-- Represents a wall configuration in the game --/
structure WallConfig :=
  (walls : List Nat)

/-- Calculates the nim-value of a single wall --/
def nimValue (wall : Nat) : Nat :=
  sorry

/-- Calculates the nim-sum of a list of nim-values --/
def nimSum (values : List Nat) : Nat :=
  sorry

/-- Determines if a given configuration is a winning position for the second player --/
def isWinningForSecondPlayer (config : WallConfig) : Prop :=
  nimSum (config.walls.map nimValue) = 0

/-- The main theorem stating that (7,3,1) is the only winning configuration for the second player --/
theorem only_7_3_1_wins_for_second_player :
  let configs := [
    WallConfig.mk [7, 1, 1],
    WallConfig.mk [7, 2, 1],
    WallConfig.mk [7, 2, 2],
    WallConfig.mk [7, 3, 1],
    WallConfig.mk [7, 3, 2]
  ]
  ∀ config ∈ configs, isWinningForSecondPlayer config ↔ config = WallConfig.mk [7, 3, 1] :=
  sorry

end only_7_3_1_wins_for_second_player_l2814_281416


namespace chest_contents_l2814_281496

/-- Represents the types of coins that can be in a chest -/
inductive CoinType
  | Gold
  | Silver
  | Copper

/-- Represents a chest with its inscription and actual content -/
structure Chest where
  inscription : CoinType → Prop
  content : CoinType

/-- The problem setup -/
def chestProblem (c1 c2 c3 : Chest) : Prop :=
  -- Inscriptions
  c1.inscription = fun t => t = CoinType.Gold ∧
  c2.inscription = fun t => t = CoinType.Silver ∧
  c3.inscription = fun t => t = CoinType.Gold ∨ t = CoinType.Silver ∧
  -- All inscriptions are incorrect
  ¬c1.inscription c1.content ∧
  ¬c2.inscription c2.content ∧
  ¬c3.inscription c3.content ∧
  -- One of each type of coin
  c1.content ≠ c2.content ∧
  c2.content ≠ c3.content ∧
  c3.content ≠ c1.content

/-- The theorem to prove -/
theorem chest_contents (c1 c2 c3 : Chest) :
  chestProblem c1 c2 c3 →
  c1.content = CoinType.Silver ∧
  c2.content = CoinType.Gold ∧
  c3.content = CoinType.Copper :=
by
  sorry

end chest_contents_l2814_281496


namespace katie_baked_18_cupcakes_l2814_281460

/-- The number of cupcakes Todd ate -/
def todd_ate : ℕ := 8

/-- The number of packages Katie could make after Todd ate some cupcakes -/
def packages : ℕ := 5

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 2

/-- The initial number of cupcakes Katie baked -/
def initial_cupcakes : ℕ := todd_ate + packages * cupcakes_per_package

theorem katie_baked_18_cupcakes : initial_cupcakes = 18 := by
  sorry

end katie_baked_18_cupcakes_l2814_281460


namespace min_reciprocal_sum_l2814_281441

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 ∧ (1/a + 1/b = 2 ↔ a = 1 ∧ b = 1) :=
by sorry

end min_reciprocal_sum_l2814_281441


namespace broken_cone_height_l2814_281467

/-- Theorem: New height of a broken cone -/
theorem broken_cone_height (r : ℝ) (l : ℝ) (l_new : ℝ) (H : ℝ) :
  r = 6 →
  l = 13 →
  l_new = l - 2 →
  H^2 + r^2 = l_new^2 →
  H = Real.sqrt 85 := by
  sorry

end broken_cone_height_l2814_281467


namespace no_winning_strategy_l2814_281455

-- Define the game board
def GameBoard := Fin 99

-- Define a piece
structure Piece where
  number : Fin 99
  position : GameBoard

-- Define a player
inductive Player
| Jia
| Yi

-- Define the game state
structure GameState where
  board : List Piece
  currentPlayer : Player

-- Define a winning condition
def isWinningState (state : GameState) : Prop :=
  ∃ (i j k : GameBoard),
    (i.val + 1) % 99 = j.val ∧
    (j.val + 1) % 99 = k.val ∧
    ∃ (pi pj pk : Piece),
      pi ∈ state.board ∧ pj ∈ state.board ∧ pk ∈ state.board ∧
      pi.position = i ∧ pj.position = j ∧ pk.position = k ∧
      pj.number.val - pi.number.val = pk.number.val - pj.number.val

-- Define a strategy
def Strategy := GameState → Option Piece

-- Define the theorem
theorem no_winning_strategy :
  ¬∃ (s : Strategy), ∀ (opponent_strategy : Strategy),
    (∃ (n : ℕ) (final_state : GameState),
      final_state.currentPlayer = Player.Yi ∧
      isWinningState final_state) ∨
    (∃ (n : ℕ) (final_state : GameState),
      final_state.currentPlayer = Player.Jia ∧
      isWinningState final_state) :=
sorry

end no_winning_strategy_l2814_281455


namespace intersection_of_A_and_B_l2814_281494

def A : Set ℝ := {-1, 0, (1/2 : ℝ), 3}
def B : Set ℝ := {x : ℝ | x^2 ≥ 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 3} := by
  sorry

end intersection_of_A_and_B_l2814_281494


namespace investment_interest_theorem_l2814_281453

/-- Calculates the total interest earned from two investments -/
def totalInterest (totalAmount : ℝ) (amount1 : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount2 := totalAmount - amount1
  let interest1 := amount1 * rate1
  let interest2 := amount2 * rate2
  interest1 + interest2

/-- Proves that investing $9000 with $4000 at 8% and the rest at 9% yields $770 in interest -/
theorem investment_interest_theorem :
  totalInterest 9000 4000 0.08 0.09 = 770 := by
  sorry

end investment_interest_theorem_l2814_281453


namespace real_solution_implies_m_positive_l2814_281439

theorem real_solution_implies_m_positive (x m : ℝ) : 
  (∃ x : ℝ, 3^x - m = 0) → m > 0 := by
sorry

end real_solution_implies_m_positive_l2814_281439


namespace largest_integer_m_l2814_281419

theorem largest_integer_m (x m : ℝ) : 
  (3 : ℝ) / 3 + 2 * m < -3 → 
  ∀ k : ℤ, (k : ℝ) > m → k ≤ -3 :=
sorry

end largest_integer_m_l2814_281419


namespace range_of_a_for_always_positive_quadratic_l2814_281425

theorem range_of_a_for_always_positive_quadratic :
  {a : ℝ | ∀ x : ℝ, x^2 + 2*a*x + a > 0} = Set.Ioo 0 1 := by
  sorry

end range_of_a_for_always_positive_quadratic_l2814_281425


namespace intersection_of_S_and_T_l2814_281482

def S : Set ℕ := {1, 4, 5}
def T : Set ℕ := {2, 3, 4}

theorem intersection_of_S_and_T : S ∩ T = {4} := by sorry

end intersection_of_S_and_T_l2814_281482


namespace stratified_sampling_sophomores_l2814_281429

/-- Given a school with 2000 students, of which 700 are sophomores,
    and a stratified sample of 100 students, the number of sophomores
    in the sample should be 35. -/
theorem stratified_sampling_sophomores :
  ∀ (total_students sample_size num_sophomores : ℕ),
    total_students = 2000 →
    sample_size = 100 →
    num_sophomores = 700 →
    (num_sophomores * sample_size) / total_students = 35 :=
by
  sorry

#check stratified_sampling_sophomores

end stratified_sampling_sophomores_l2814_281429


namespace sphere_hemisphere_volume_ratio_l2814_281423

/-- The ratio of the volume of a sphere with radius 3q to the volume of a hemisphere with radius q is 54 -/
theorem sphere_hemisphere_volume_ratio (q : ℝ) (q_pos : 0 < q) : 
  (4 / 3 * Real.pi * (3 * q)^3) / ((1 / 2) * (4 / 3 * Real.pi * q^3)) = 54 := by
  sorry

end sphere_hemisphere_volume_ratio_l2814_281423


namespace absolute_value_nonnegative_l2814_281414

theorem absolute_value_nonnegative (a : ℝ) : ¬(|a| < 0) := by
  sorry

end absolute_value_nonnegative_l2814_281414


namespace rectangular_field_area_l2814_281456

/-- Represents the ratio of the sides of a rectangular field -/
def side_ratio : ℚ := 3 / 4

/-- Cost of fencing per metre in paise -/
def fencing_cost_per_metre : ℚ := 25

/-- Total cost of fencing in rupees -/
def total_fencing_cost : ℚ := 101.5

/-- Conversion factor from rupees to paise -/
def rupees_to_paise : ℕ := 100

theorem rectangular_field_area (length width : ℚ) :
  length / width = side_ratio →
  2 * (length + width) * fencing_cost_per_metre = total_fencing_cost * rupees_to_paise →
  length * width = 10092 := by sorry

end rectangular_field_area_l2814_281456


namespace sin_50_plus_sqrt3_tan_10_eq_1_l2814_281433

theorem sin_50_plus_sqrt3_tan_10_eq_1 :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end sin_50_plus_sqrt3_tan_10_eq_1_l2814_281433


namespace petya_max_win_margin_l2814_281422

theorem petya_max_win_margin :
  ∀ (p1 p2 v1 v2 : ℕ),
    p1 + p2 + v1 + v2 = 27 →
    p1 = v1 + 9 →
    v2 = p2 + 9 →
    p1 + p2 > v1 + v2 →
    p1 + p2 - (v1 + v2) ≤ 9 :=
by
  sorry

end petya_max_win_margin_l2814_281422


namespace quadratic_roots_ratio_l2814_281478

theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x ≠ y ∧
   x^2 + 12*x + k = 0 ∧ 
   y^2 + 12*y + k = 0 ∧
   x / y = 3) → k = 27 := by
  sorry

end quadratic_roots_ratio_l2814_281478


namespace current_speed_calculation_l2814_281404

theorem current_speed_calculation (boat_speed : ℝ) (upstream_time : ℝ) (downstream_time : ℝ)
  (h1 : boat_speed = 16)
  (h2 : upstream_time = 20 / 60)
  (h3 : downstream_time = 15 / 60) :
  ∃ (current_speed : ℝ),
    (boat_speed - current_speed) * upstream_time = (boat_speed + current_speed) * downstream_time ∧
    current_speed = 16 / 7 := by
  sorry

end current_speed_calculation_l2814_281404


namespace preimage_of_one_seven_l2814_281442

/-- The mapping function from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

/-- The theorem stating that (4, -3) is the preimage of (1, 7) under f -/
theorem preimage_of_one_seven :
  f (4, -3) = (1, 7) ∧ 
  ∀ p : ℝ × ℝ, f p = (1, 7) → p = (4, -3) := by
  sorry

end preimage_of_one_seven_l2814_281442


namespace sum_of_solutions_quadratic_l2814_281435

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 10*x - 24) → (∃ y : ℝ, y^2 = 10*y - 24 ∧ x + y = 10) := by
  sorry

end sum_of_solutions_quadratic_l2814_281435


namespace equation_solution_l2814_281424

theorem equation_solution : 
  ∃! x : ℝ, (2 : ℝ) / (x + 3) + (3 * x) / (x + 3) - (5 : ℝ) / (x + 3) = 4 ∧ x = -15 := by
  sorry

end equation_solution_l2814_281424


namespace point_coordinates_sum_l2814_281477

theorem point_coordinates_sum (X Y Z : ℝ × ℝ) : 
  (X.1 - Z.1) / (X.1 - Y.1) = 1/2 →
  (X.2 - Z.2) / (X.2 - Y.2) = 1/2 →
  (Z.1 - Y.1) / (X.1 - Y.1) = 1/2 →
  (Z.2 - Y.2) / (X.2 - Y.2) = 1/2 →
  Y = (2, 5) →
  Z = (1, -3) →
  X.1 + X.2 = -11 := by
sorry

end point_coordinates_sum_l2814_281477


namespace find_k_n_l2814_281417

theorem find_k_n : ∃ (k n : ℕ), k * n^2 - k * n - n^2 + n = 94 ∧ k = 48 ∧ n = 2 := by
  sorry

end find_k_n_l2814_281417


namespace perpendicular_vectors_x_value_l2814_281415

/-- Given two vectors a and b in ℝ², where a = (x - 1, 2) and b = (2, 1),
    if a is perpendicular to b, then x = 0. -/
theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (x - 1, 2)
  let b : ℝ × ℝ := (2, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 0 := by
  sorry

end perpendicular_vectors_x_value_l2814_281415


namespace smallest_battleship_board_l2814_281489

/-- Represents the types of ships in Battleship -/
inductive ShipType
  | OneByFour
  | OneByThree
  | OneByTwo
  | OneByOne

/-- The set of ships in a standard Battleship game -/
def battleshipSet : List ShipType :=
  [ShipType.OneByFour] ++
  List.replicate 2 ShipType.OneByThree ++
  List.replicate 3 ShipType.OneByTwo ++
  List.replicate 4 ShipType.OneByOne

/-- Calculates the number of nodes a ship occupies, including its surrounding space -/
def nodesOccupied (ship : ShipType) : Nat :=
  match ship with
  | ShipType.OneByFour => 10
  | ShipType.OneByThree => 8
  | ShipType.OneByTwo => 6
  | ShipType.OneByOne => 4

/-- The smallest square board size for Battleship -/
def smallestBoardSize : Nat := 7

theorem smallest_battleship_board :
  (∀ n : Nat, n < smallestBoardSize → 
    (List.sum (List.map nodesOccupied battleshipSet) > (n + 1)^2)) ∧
  (List.sum (List.map nodesOccupied battleshipSet) ≤ (smallestBoardSize + 1)^2) := by
  sorry

#eval smallestBoardSize  -- Should output 7

end smallest_battleship_board_l2814_281489


namespace ice_cream_volume_specific_ice_cream_volume_l2814_281476

/-- The volume of ice cream in a right circular cone with a hemisphere on top -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1/3) * π * r^2 * h
  let hemisphere_volume := (2/3) * π * r^3
  cone_volume + hemisphere_volume = (320/3) * π :=
by
  sorry

/-- The specific case with h = 12 and r = 4 -/
theorem specific_ice_cream_volume :
  let h : ℝ := 12
  let r : ℝ := 4
  let cone_volume := (1/3) * π * r^2 * h
  let hemisphere_volume := (2/3) * π * r^3
  cone_volume + hemisphere_volume = (320/3) * π :=
by
  sorry

end ice_cream_volume_specific_ice_cream_volume_l2814_281476


namespace a_5_of_1034_is_5_l2814_281491

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Factorial base representation -/
def factorial_base_rep (n : ℕ) : List ℕ :=
  sorry

/-- The 5th coefficient in the factorial base representation -/
def a_5 (n : ℕ) : ℕ :=
  match factorial_base_rep n with
  | b₁ :: b₂ :: b₃ :: b₄ :: b₅ :: _ => b₅
  | _ => 0  -- Default case if the list is too short

/-- Theorem stating that the 5th coefficient of 1034 in factorial base is 5 -/
theorem a_5_of_1034_is_5 : a_5 1034 = 5 := by
  sorry

end a_5_of_1034_is_5_l2814_281491


namespace star_arrangement_count_l2814_281461

/-- The number of symmetries of a regular six-pointed star -/
def star_symmetries : ℕ := 12

/-- The number of distinct shells to be placed -/
def num_shells : ℕ := 12

/-- The number of distinct arrangements of shells on a regular six-pointed star,
    considering rotational and reflectional symmetries -/
def distinct_arrangements : ℕ := Nat.factorial num_shells / star_symmetries

theorem star_arrangement_count :
  distinct_arrangements = 39916800 := by sorry

end star_arrangement_count_l2814_281461


namespace inequality_chain_l2814_281406

theorem inequality_chain (x : ℝ) (h : 1 < x ∧ x < 2) :
  ((Real.log x) / x) ^ 2 < (Real.log x) / x ∧ (Real.log x) / x < (Real.log (x^2)) / (x^2) := by
  sorry

end inequality_chain_l2814_281406


namespace spare_time_is_five_hours_l2814_281444

/-- Calculates the spare time for painting a room given the following conditions:
  * The room has 5 walls
  * Each wall is 2 meters by 3 meters
  * The painter can paint 1 square meter every 10 minutes
  * The painter has 10 hours to paint everything
-/
def spare_time_for_painting : ℕ :=
  let num_walls : ℕ := 5
  let wall_width : ℕ := 2
  let wall_height : ℕ := 3
  let painting_rate : ℕ := 10  -- minutes per square meter
  let total_time : ℕ := 10 * 60  -- total time in minutes

  let wall_area : ℕ := wall_width * wall_height
  let total_area : ℕ := num_walls * wall_area
  let painting_time : ℕ := total_area * painting_rate
  let spare_time_minutes : ℕ := total_time - painting_time
  spare_time_minutes / 60

theorem spare_time_is_five_hours : spare_time_for_painting = 5 := by
  sorry

end spare_time_is_five_hours_l2814_281444


namespace selection_theorem_l2814_281408

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def total_people : ℕ := num_boys + num_girls
def num_to_select : ℕ := 4

theorem selection_theorem : 
  (Nat.choose total_people num_to_select) - (Nat.choose num_boys num_to_select) = 34 := by
  sorry

end selection_theorem_l2814_281408


namespace jeans_pricing_l2814_281457

theorem jeans_pricing (C : ℝ) (R : ℝ) :
  (C > 0) →
  (R > C) →
  (1.40 * R = 1.96 * C) →
  ((R - C) / C * 100 = 40) :=
by sorry

end jeans_pricing_l2814_281457


namespace min_value_theorem_l2814_281450

theorem min_value_theorem (a b : ℝ) (hb : b > 0) (h : a + 2*b = 1) :
  (3/b) + (1/a) ≥ 7 + 2*Real.sqrt 6 := by
  sorry

end min_value_theorem_l2814_281450


namespace fourth_term_of_geometric_sequence_l2814_281403

/-- Given a geometric sequence where:
    - The first term is 4
    - The second term is 12y
    - The third term is 36y^3
    Prove that the fourth term is 108y^5 -/
theorem fourth_term_of_geometric_sequence (y : ℝ) :
  let a₁ : ℝ := 4
  let a₂ : ℝ := 12 * y
  let a₃ : ℝ := 36 * y^3
  let a₄ : ℝ := 108 * y^5
  (∃ (r : ℝ), a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r) :=
by
  sorry

end fourth_term_of_geometric_sequence_l2814_281403


namespace evaluate_expression_l2814_281431

theorem evaluate_expression : (16 ^ 24) / (64 ^ 8) = 16 ^ 12 := by
  sorry

end evaluate_expression_l2814_281431


namespace complex_equation_solution_l2814_281436

theorem complex_equation_solution (z : ℂ) : (2 * Complex.I) / z = 1 - Complex.I → z = -1 + Complex.I := by
  sorry

end complex_equation_solution_l2814_281436


namespace right_triangles_shared_hypotenuse_l2814_281432

theorem right_triangles_shared_hypotenuse (a : ℝ) (h : a ≥ Real.sqrt 7) :
  let BC : ℝ := 3
  let AC : ℝ := a
  let AD : ℝ := 4
  let AB : ℝ := Real.sqrt (AC^2 + BC^2)
  let BD : ℝ := Real.sqrt (AB^2 - AD^2)
  BD = Real.sqrt (a^2 - 7) :=
by sorry

end right_triangles_shared_hypotenuse_l2814_281432


namespace sum_of_fractions_equals_41_20_l2814_281426

theorem sum_of_fractions_equals_41_20 : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) + (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 41 / 20 := by
  sorry

end sum_of_fractions_equals_41_20_l2814_281426


namespace fountain_water_after_25_days_l2814_281470

def fountain_water_volume (initial_volume : ℝ) (evaporation_rate : ℝ) (rain_interval : ℕ) (rain_amount : ℝ) (days : ℕ) : ℝ :=
  let total_evaporation := evaporation_rate * days
  let rain_events := days / rain_interval
  let total_rain := rain_events * rain_amount
  initial_volume + total_rain - total_evaporation

theorem fountain_water_after_25_days :
  fountain_water_volume 120 0.8 5 5 25 = 125 := by sorry

end fountain_water_after_25_days_l2814_281470


namespace max_value_of_f_l2814_281473

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x - Real.cos x

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = Real.sqrt 5 := by
  sorry

end max_value_of_f_l2814_281473


namespace specific_figure_perimeter_l2814_281466

/-- A figure composed of unit squares arranged in a specific pattern -/
structure UnitSquareFigure where
  horizontalSegments : ℕ
  verticalSegments : ℕ

/-- The perimeter of a UnitSquareFigure -/
def perimeter (figure : UnitSquareFigure) : ℕ :=
  figure.horizontalSegments + figure.verticalSegments

/-- The specific figure from the problem -/
def specificFigure : UnitSquareFigure :=
  { horizontalSegments := 16, verticalSegments := 10 }

/-- Theorem stating that the perimeter of the specific figure is 26 -/
theorem specific_figure_perimeter :
  perimeter specificFigure = 26 := by sorry

end specific_figure_perimeter_l2814_281466


namespace stating_non_parallel_necessary_not_sufficient_l2814_281438

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Represents a system of two linear equations -/
structure LinearSystem where
  line1 : Line
  line2 : Line

/-- Checks if a linear system has a unique solution -/
def has_unique_solution (sys : LinearSystem) : Prop :=
  sys.line1.a * sys.line2.b ≠ sys.line1.b * sys.line2.a

/-- 
Theorem stating that non-parallel lines are a necessary but insufficient condition
for a system of two linear equations to have a unique solution
-/
theorem non_parallel_necessary_not_sufficient (sys : LinearSystem) :
  has_unique_solution sys → ¬(are_parallel sys.line1 sys.line2) ∧
  ¬(¬(are_parallel sys.line1 sys.line2) → has_unique_solution sys) :=
by sorry

end stating_non_parallel_necessary_not_sufficient_l2814_281438


namespace article_sale_loss_percent_l2814_281484

/-- Theorem: Given an article with a 35% gain at its original selling price,
    when sold at 2/3 of the original price, the loss percent is 10%. -/
theorem article_sale_loss_percent (cost_price : ℝ) (original_price : ℝ) :
  original_price = cost_price * (1 + 35 / 100) →
  let new_price := (2 / 3) * original_price
  let loss := cost_price - new_price
  let loss_percent := (loss / cost_price) * 100
  loss_percent = 10 := by
sorry

end article_sale_loss_percent_l2814_281484


namespace circle_center_l2814_281413

def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

theorem circle_center : 
  ∃ (h k : ℝ), (∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 5) ∧ h = 1 ∧ k = 2 :=
sorry

end circle_center_l2814_281413


namespace balloon_arrangements_l2814_281495

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : ℕ) (repeatedTwice : ℕ) (appearOnce : ℕ) : ℕ :=
  Nat.factorial totalLetters / (2^repeatedTwice * Nat.factorial appearOnce)

/-- Theorem: The number of distinct arrangements of letters in a word with 7 letters,
    where two letters each appear twice and three letters appear once, is equal to 1260 -/
theorem balloon_arrangements :
  distinctArrangements 7 2 3 = 1260 := by
  sorry

end balloon_arrangements_l2814_281495


namespace sum_of_divisors_450_has_three_prime_factors_l2814_281479

/-- The sum of positive divisors function -/
noncomputable def sigma (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors function -/
noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_450_has_three_prime_factors :
  let n : ℕ := 450
  let sum_of_divisors : ℕ := sigma n
  num_distinct_prime_factors sum_of_divisors = 3 := by
  sorry

end sum_of_divisors_450_has_three_prime_factors_l2814_281479


namespace smallest_value_in_different_bases_l2814_281462

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

theorem smallest_value_in_different_bases :
  let base_9 := to_decimal [8, 5] 9
  let base_6 := to_decimal [2, 1, 0] 6
  let base_4 := to_decimal [1, 0, 0, 0] 4
  let base_2 := to_decimal [1, 1, 1, 1, 1, 1] 2
  base_2 = min base_9 (min base_6 (min base_4 base_2)) :=
by sorry

end smallest_value_in_different_bases_l2814_281462


namespace product_sum_equality_l2814_281434

theorem product_sum_equality : 15 * 35 + 45 * 15 = 1200 := by
  sorry

end product_sum_equality_l2814_281434


namespace figure_area_bound_l2814_281412

-- Define the unit square
def UnitSquare : Set (Real × Real) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the property of the figure
def ValidFigure (F : Set (Real × Real)) : Prop :=
  F ⊆ UnitSquare ∧
  ∀ p q : Real × Real, p ∈ F → q ∈ F → dist p q ≠ 0.001

-- Define the area of a set
noncomputable def area (S : Set (Real × Real)) : Real :=
  sorry

-- State the theorem
theorem figure_area_bound {F : Set (Real × Real)} (hF : ValidFigure F) :
  area F ≤ 0.34 ∧ area F ≤ 0.287 :=
sorry

end figure_area_bound_l2814_281412


namespace symmetric_function_max_value_l2814_281469

/-- Given a function f(x) = (1-x^2)(x^2 + ax + b) that is symmetric about x = -2,
    prove that its maximum value is 16. -/
theorem symmetric_function_max_value
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_def : ∀ x, f x = (1 - x^2) * (x^2 + a*x + b))
  (h_sym : ∀ x, f (x + (-2)) = f ((-2) - x)) :
  ∃ x, f x = 16 ∧ ∀ y, f y ≤ 16 := by
  sorry

end symmetric_function_max_value_l2814_281469


namespace postcard_problem_l2814_281468

theorem postcard_problem (initial_postcards : ℕ) : 
  (initial_postcards / 2 + (initial_postcards / 2) * 3 = 36) → 
  initial_postcards = 18 := by
  sorry

end postcard_problem_l2814_281468


namespace new_year_money_distribution_l2814_281407

/-- Represents the distribution of money to three grandsons --/
structure MoneyDistribution :=
  (grandson1 : ℕ)
  (grandson2 : ℕ)
  (grandson3 : ℕ)

/-- Checks if a distribution is valid according to the problem conditions --/
def is_valid_distribution (d : MoneyDistribution) : Prop :=
  -- Total sum is 300
  d.grandson1 + d.grandson2 + d.grandson3 = 300 ∧
  -- Each amount is divisible by 10 (smallest denomination)
  d.grandson1 % 10 = 0 ∧ d.grandson2 % 10 = 0 ∧ d.grandson3 % 10 = 0 ∧
  -- Each amount is one of the allowed denominations (50, 20, or 10)
  (d.grandson1 % 50 = 0 ∨ d.grandson1 % 20 = 0 ∨ d.grandson1 % 10 = 0) ∧
  (d.grandson2 % 50 = 0 ∨ d.grandson2 % 20 = 0 ∨ d.grandson2 % 10 = 0) ∧
  (d.grandson3 % 50 = 0 ∨ d.grandson3 % 20 = 0 ∨ d.grandson3 % 10 = 0) ∧
  -- Number of bills condition
  (d.grandson1 / 10 = (d.grandson2 / 20) * (d.grandson3 / 50) ∨
   d.grandson2 / 20 = (d.grandson1 / 10) * (d.grandson3 / 50) ∨
   d.grandson3 / 50 = (d.grandson1 / 10) * (d.grandson2 / 20))

/-- The theorem to be proved --/
theorem new_year_money_distribution :
  ∀ d : MoneyDistribution,
    is_valid_distribution d →
    (d = ⟨100, 100, 100⟩ ∨ d = ⟨90, 60, 150⟩ ∨ d = ⟨90, 150, 60⟩ ∨
     d = ⟨60, 90, 150⟩ ∨ d = ⟨60, 150, 90⟩ ∨ d = ⟨150, 60, 90⟩ ∨
     d = ⟨150, 90, 60⟩) :=
by sorry


end new_year_money_distribution_l2814_281407


namespace scientific_notation_120_million_l2814_281400

theorem scientific_notation_120_million :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 120000000 = a * (10 : ℝ) ^ n ∧ a = 1.2 ∧ n = 7 :=
by sorry

end scientific_notation_120_million_l2814_281400


namespace triangle_third_side_validity_l2814_281451

theorem triangle_third_side_validity (a b c : ℝ) : 
  a = 4 → b = 10 → c = 11 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) ∧ 
  (c < a + b ∧ a < b + c ∧ b < c + a) := by
  sorry

end triangle_third_side_validity_l2814_281451


namespace philips_farm_animals_l2814_281428

/-- Represents the number of animals on Philip's farm -/
structure FarmAnimals where
  cows : ℕ
  ducks : ℕ
  horses : ℕ
  pigs : ℕ
  chickens : ℕ

/-- Calculates the total number of animals on the farm -/
def total_animals (farm : FarmAnimals) : ℕ :=
  farm.cows + farm.ducks + farm.horses + farm.pigs + farm.chickens

/-- Theorem stating the total number of animals on Philip's farm -/
theorem philips_farm_animals :
  ∃ (farm : FarmAnimals),
    farm.cows = 20 ∧
    farm.ducks = farm.cows + farm.cows / 2 ∧
    farm.horses = (farm.cows + farm.ducks) / 5 ∧
    farm.pigs = (farm.cows + farm.ducks + farm.horses) / 5 ∧
    farm.chickens = 3 * (farm.cows - farm.horses) ∧
    total_animals farm = 102 := by
  sorry


end philips_farm_animals_l2814_281428


namespace imaginary_root_cubic_equation_l2814_281471

theorem imaginary_root_cubic_equation (a b q r : ℝ) :
  b ≠ 0 →
  (∃ (x : ℂ), x^3 + q*x + r = 0 ∧ x = a + b*Complex.I) →
  q = b^2 - 3*a^2 := by
  sorry

end imaginary_root_cubic_equation_l2814_281471


namespace smallest_prime_factor_of_1953_l2814_281410

theorem smallest_prime_factor_of_1953 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1953 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1953 → p ≤ q :=
sorry

end smallest_prime_factor_of_1953_l2814_281410


namespace fraction_simplification_l2814_281483

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b :=
by sorry

end fraction_simplification_l2814_281483


namespace pi_irrational_in_set_l2814_281427

theorem pi_irrational_in_set : ∃ x ∈ ({-2, 0, Real.sqrt 9, Real.pi} : Set ℝ), Irrational x :=
by sorry

end pi_irrational_in_set_l2814_281427


namespace dan_marbles_l2814_281445

theorem dan_marbles (initial_marbles given_marbles : ℕ) 
  (h1 : initial_marbles = 64)
  (h2 : given_marbles = 14) :
  initial_marbles - given_marbles = 50 := by
  sorry

end dan_marbles_l2814_281445


namespace greatest_x_value_l2814_281449

theorem greatest_x_value : 
  let f (x : ℝ) := ((5*x - 20) / (4*x - 5))^2 + (5*x - 20) / (4*x - 5)
  ∃ (x_max : ℝ), x_max = 50/29 ∧ 
    (∀ (x : ℝ), f x = 18 → x ≤ x_max) ∧
    (f x_max = 18) :=
by sorry

end greatest_x_value_l2814_281449


namespace double_time_double_discount_l2814_281480

/-- Represents the true discount for a bill over a given time period. -/
structure TrueDiscount where
  bill : ℝ  -- Face value of the bill
  discount : ℝ  -- Amount of discount
  time : ℝ  -- Time period

/-- Calculates the true discount for a doubled time period. -/
def double_time_discount (td : TrueDiscount) : ℝ :=
  2 * td.discount

/-- Theorem stating that doubling the time period doubles the true discount. -/
theorem double_time_double_discount (td : TrueDiscount) 
  (h1 : td.bill = 110) 
  (h2 : td.discount = 10) :
  double_time_discount td = 20 := by
  sorry

#check double_time_double_discount

end double_time_double_discount_l2814_281480


namespace quadratic_root_values_l2814_281465

theorem quadratic_root_values : 
  (Real.sqrt (9 - 8 * 0) = 3) ∧ 
  (Real.sqrt (9 - 8 * (1/2)) = Real.sqrt 5) ∧ 
  (Real.sqrt (9 - 8 * (-2)) = 5) := by
  sorry

end quadratic_root_values_l2814_281465


namespace digits_of_product_l2814_281481

theorem digits_of_product : ∃ (n : ℕ), n = 3^4 * 6^8 ∧ (Nat.log 10 n + 1 = 9) := by sorry

end digits_of_product_l2814_281481


namespace arrangements_with_three_together_eq_36_l2814_281475

/-- The number of different arrangements of five students in a row,
    where three specific students must be together. -/
def arrangements_with_three_together : ℕ :=
  (3 : ℕ).factorial * (3 : ℕ).factorial

theorem arrangements_with_three_together_eq_36 :
  arrangements_with_three_together = 36 := by
  sorry

end arrangements_with_three_together_eq_36_l2814_281475


namespace triangle_area_l2814_281485

theorem triangle_area (base height : ℝ) (h1 : base = 3) (h2 : height = 4) :
  (base * height) / 2 = 6 := by
  sorry

end triangle_area_l2814_281485


namespace value_calculation_l2814_281418

theorem value_calculation (n : ℝ) (v : ℝ) (h : n = 50) : 0.20 * n - 4 = v → v = 6 := by
  sorry

end value_calculation_l2814_281418


namespace cosine_function_property_l2814_281459

/-- Given a cosine function with specific properties, prove that its angular frequency is 2. -/
theorem cosine_function_property (f : ℝ → ℝ) (ω φ : ℝ) (h_ω_pos : ω > 0) (h_φ_bound : |φ| ≤ π/2) 
  (h_f_def : ∀ x, f x = Real.sqrt 2 * Real.cos (ω * x + φ)) 
  (h_product : ∃ x₁ x₂ : ℝ, f x₁ * f x₂ = -2)
  (h_min_diff : ∃ x₁ x₂ : ℝ, f x₁ * f x₂ = -2 ∧ |x₁ - x₂| = π/2) : ω = 2 := by
  sorry

end cosine_function_property_l2814_281459


namespace ratio_subtraction_l2814_281437

theorem ratio_subtraction (x : ℕ) : 
  x = 3 ∧ 
  (6 - x : ℚ) / (7 - x) < 16 / 21 ∧ 
  ∀ y : ℕ, y < x → (6 - y : ℚ) / (7 - y) ≥ 16 / 21 → 
  6 = 6 :=
sorry

end ratio_subtraction_l2814_281437


namespace percentage_less_than_l2814_281497

theorem percentage_less_than (w x y z : ℝ) 
  (hw : w = 0.60 * x) 
  (hz1 : z = 0.54 * y) 
  (hz2 : z = 1.50 * w) : 
  x = 0.60 * y := by
sorry

end percentage_less_than_l2814_281497


namespace root_relationship_l2814_281447

theorem root_relationship (m n a b : ℝ) : 
  (∀ x, 3 - (x - m) * (x - n) = 0 ↔ x = a ∨ x = b) →
  a < m ∧ m < n ∧ n < b :=
sorry

end root_relationship_l2814_281447


namespace complex_modulus_problem_l2814_281499

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l2814_281499


namespace binomial_coefficient_sum_squared_difference_l2814_281487

theorem binomial_coefficient_sum_squared_difference (a₀ a₁ a₂ a₃ : ℝ) : 
  (∀ x, (Real.sqrt 5 * x - 1)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -64 := by
  sorry

end binomial_coefficient_sum_squared_difference_l2814_281487


namespace max_product_of_primes_sum_l2814_281493

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem max_product_of_primes_sum (p : List ℕ) (h : p = primes) :
  ∃ (a b c d e f g h : ℕ),
    a ∈ p ∧ b ∈ p ∧ c ∈ p ∧ d ∈ p ∧ e ∈ p ∧ f ∈ p ∧ g ∈ p ∧ h ∈ p ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h ∧
    (a + b + c + d) * (e + f + g + h) = 1480 ∧
    ∀ (a' b' c' d' e' f' g' h' : ℕ),
      a' ∈ p → b' ∈ p → c' ∈ p → d' ∈ p → e' ∈ p → f' ∈ p → g' ∈ p → h' ∈ p →
      a' ≠ b' → a' ≠ c' → a' ≠ d' → a' ≠ e' → a' ≠ f' → a' ≠ g' → a' ≠ h' →
      b' ≠ c' → b' ≠ d' → b' ≠ e' → b' ≠ f' → b' ≠ g' → b' ≠ h' →
      c' ≠ d' → c' ≠ e' → c' ≠ f' → c' ≠ g' → c' ≠ h' →
      d' ≠ e' → d' ≠ f' → d' ≠ g' → d' ≠ h' →
      e' ≠ f' → e' ≠ g' → e' ≠ h' →
      f' ≠ g' → f' ≠ h' →
      g' ≠ h' →
      (a' + b' + c' + d') * (e' + f' + g' + h') ≤ 1480 :=
by sorry

end max_product_of_primes_sum_l2814_281493


namespace equation_solution_l2814_281402

theorem equation_solution :
  ∃ x : ℝ, 4 * (x - 2) * (x + 5) = (2 * x - 3) * (2 * x + 11) + 11 ∧ x = 4.5 := by
  sorry

end equation_solution_l2814_281402


namespace inequality_proof_l2814_281490

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0) 
  (h5 : a + b + c + d = 1) : 
  (a + 2*b + 3*c + 4*d) * (a^a * b^b * c^c * d^d) < 1 := by
  sorry

end inequality_proof_l2814_281490


namespace flood_probability_l2814_281474

theorem flood_probability (p_30 p_40 : ℝ) 
  (h1 : p_30 = 0.8) 
  (h2 : p_40 = 0.85) : 
  (p_40 - p_30) / (1 - p_30) = 0.25 := by
  sorry

end flood_probability_l2814_281474


namespace coordinate_difference_of_P_l2814_281488

/-- Triangle ABC with vertices A(0,10), B(4,0), C(12,0) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {⟨0, 10⟩, ⟨4, 0⟩, ⟨12, 0⟩}

/-- Point P on line AC -/
def P : ℝ × ℝ := ⟨6, 5⟩

/-- Point Q on line BC -/
def Q : ℝ × ℝ := ⟨6, 0⟩

/-- Area of triangle PQC -/
def area_PQC : ℝ := 16

/-- Theorem: The positive difference between x and y coordinates of P is 1 -/
theorem coordinate_difference_of_P :
  P ∈ Set.Icc (0 : ℝ) 12 ×ˢ Set.Icc (0 : ℝ) 10 →
  Q.1 = P.1 →
  Q.2 = 0 →
  area_PQC = 16 →
  |P.1 - P.2| = 1 := by sorry

end coordinate_difference_of_P_l2814_281488


namespace exponential_equation_and_inequality_l2814_281446

/-- Given a > 0 and a ≠ 1, this theorem proves the conditions for equality and inequality
    between a^(3x+1) and a^(-2x) -/
theorem exponential_equation_and_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, a^(3*x + 1) = a^(-2*x) ↔ x = 1/5) ∧
  (∀ x : ℝ, (a > 1 → (a^(3*x + 1) < a^(-2*x) ↔ x < 1/5)) ∧
            (a < 1 → (a^(3*x + 1) < a^(-2*x) ↔ x > 1/5))) :=
by sorry

end exponential_equation_and_inequality_l2814_281446


namespace propositions_p_and_q_l2814_281405

theorem propositions_p_and_q : 
  (∃ a b c : ℝ, a < b ∧ a * c^2 ≥ b * c^2) ∧ 
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 - Real.log x₀ = 0) := by
  sorry

end propositions_p_and_q_l2814_281405


namespace edward_spent_thirteen_l2814_281430

/-- The amount of money Edward spent -/
def amount_spent (initial_amount current_amount : ℕ) : ℕ :=
  initial_amount - current_amount

/-- Theorem: Edward spent $13 -/
theorem edward_spent_thirteen : amount_spent 19 6 = 13 := by
  sorry

end edward_spent_thirteen_l2814_281430


namespace total_farm_tax_collected_l2814_281452

/-- Theorem: Total farm tax collected from a village
Given:
- Farm tax is levied on 75% of the cultivated land
- Mr. William paid $480 as farm tax
- Mr. William's land represents 16.666666666666668% of the total taxable land in the village

Prove: The total amount collected through farm tax from the village is $2880 -/
theorem total_farm_tax_collected (william_tax : ℝ) (william_land_percentage : ℝ) 
  (h1 : william_tax = 480)
  (h2 : william_land_percentage = 16.666666666666668) :
  william_tax / (william_land_percentage / 100) = 2880 := by
  sorry

#check total_farm_tax_collected

end total_farm_tax_collected_l2814_281452


namespace sum_even_not_square_or_cube_l2814_281498

theorem sum_even_not_square_or_cube (n : ℕ+) :
  ∀ k m : ℕ+, (n : ℕ) * (n + 1) ≠ k ^ 2 ∧ (n : ℕ) * (n + 1) ≠ m ^ 3 := by
  sorry

end sum_even_not_square_or_cube_l2814_281498


namespace league_teams_count_league_teams_count_proof_l2814_281421

theorem league_teams_count : ℕ → Prop :=
  fun n => (n * (n - 1) / 2 = 45) → n = 10

-- The proof is omitted
theorem league_teams_count_proof : league_teams_count 10 := by
  sorry

end league_teams_count_league_teams_count_proof_l2814_281421


namespace cat_kittens_count_l2814_281401

def animal_shelter_problem (initial_cats : ℕ) (new_cats : ℕ) (adopted_cats : ℕ) (final_cats : ℕ) : ℕ :=
  let total_before_events := initial_cats + new_cats
  let after_adoption := total_before_events - adopted_cats
  final_cats - after_adoption + 1

theorem cat_kittens_count : animal_shelter_problem 6 12 3 19 = 5 := by
  sorry

end cat_kittens_count_l2814_281401


namespace train_cost_XY_is_900_l2814_281448

/-- Represents the cost of a train journey in dollars -/
def train_cost (distance : ℝ) : ℝ := 0.20 * distance

/-- The cities and their distances -/
structure Cities where
  XY : ℝ
  XZ : ℝ

/-- The problem setup -/
def piravena_journey : Cities where
  XY := 4500
  XZ := 4000

theorem train_cost_XY_is_900 :
  train_cost piravena_journey.XY = 900 := by sorry

end train_cost_XY_is_900_l2814_281448


namespace geometric_sequence_sum_l2814_281458

/-- A geometric sequence with positive first term and a specific condition on its terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n, a (n + 1) = a n * q) ∧ 
  a 1 > 0 ∧
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25

/-- The sum of the third and fifth terms of the geometric sequence is 5 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : geometric_sequence a) : 
  a 3 + a 5 = 5 := by
  sorry

end geometric_sequence_sum_l2814_281458


namespace dog_barking_problem_l2814_281463

/-- Given the barking patterns of two dogs and the owner's hushing behavior, 
    calculate the number of times the owner said "hush". -/
theorem dog_barking_problem (poodle_barks terrier_barks owner_hushes : ℕ) : 
  poodle_barks = 24 →
  poodle_barks = 2 * terrier_barks →
  owner_hushes * 2 = terrier_barks →
  owner_hushes = 6 := by
  sorry

end dog_barking_problem_l2814_281463


namespace power_calculation_l2814_281409

theorem power_calculation : (16^4 * 8^6) / 4^14 = 2^6 := by
  sorry

end power_calculation_l2814_281409


namespace mikes_net_spent_l2814_281411

/-- The net amount Mike spent at the music store -/
def net_amount (trumpet_cost song_book_price : ℚ) : ℚ :=
  trumpet_cost - song_book_price

/-- Theorem stating the net amount Mike spent at the music store -/
theorem mikes_net_spent :
  let trumpet_cost : ℚ := 145.16
  let song_book_price : ℚ := 5.84
  net_amount trumpet_cost song_book_price = 139.32 := by
  sorry

end mikes_net_spent_l2814_281411


namespace rectangle_areas_sum_l2814_281420

theorem rectangle_areas_sum : 
  let widths : List ℕ := [2, 3, 4, 5, 6, 7, 8]
  let lengths : List ℕ := [5, 8, 11, 14, 17, 20, 23]
  (widths.zip lengths).map (fun (w, l) => w * l) |>.sum = 574 := by
  sorry

end rectangle_areas_sum_l2814_281420


namespace quadratic_inequality_solution_l2814_281440

theorem quadratic_inequality_solution (x : ℝ) : 
  -9 * x^2 + 6 * x + 1 < 0 ↔ (1 - Real.sqrt 2) / 3 < x ∧ x < (1 + Real.sqrt 2) / 3 :=
by sorry

end quadratic_inequality_solution_l2814_281440


namespace seven_x_plus_four_is_odd_l2814_281486

theorem seven_x_plus_four_is_odd (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) := by
  sorry

end seven_x_plus_four_is_odd_l2814_281486


namespace function_periodicity_l2814_281492

variable (a : ℝ)
variable (f : ℝ → ℝ)

theorem function_periodicity
  (h : ∀ x, f (x + a) = (1 + f x) / (1 - f x)) :
  ∀ x, f (x + 4 * a) = f x :=
by sorry

end function_periodicity_l2814_281492
