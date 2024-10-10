import Mathlib

namespace inverse_of_i_minus_three_inverse_i_l607_60718

-- Define i as a complex number with i^2 = -1
def i : ℂ := Complex.I

-- State the theorem
theorem inverse_of_i_minus_three_inverse_i (h : i^2 = -1) :
  (i - 3 * i⁻¹)⁻¹ = -i/4 := by
  sorry

end inverse_of_i_minus_three_inverse_i_l607_60718


namespace sufficient_not_necessary_l607_60773

theorem sufficient_not_necessary :
  (∀ x : ℝ, x > 3 → x^2 - 5*x + 6 > 0) ∧
  (∃ x : ℝ, x^2 - 5*x + 6 > 0 ∧ ¬(x > 3)) := by
  sorry

end sufficient_not_necessary_l607_60773


namespace defective_units_shipped_l607_60702

theorem defective_units_shipped (total_units : ℝ) (h1 : total_units > 0) : 
  let defective_units := 0.07 * total_units
  let defective_shipped := 0.0035 * total_units
  (defective_shipped / defective_units) * 100 = 5 := by
sorry

end defective_units_shipped_l607_60702


namespace fraction_meaningful_l607_60760

theorem fraction_meaningful (x : ℝ) : (∃ y : ℝ, y = 1 / (x + 1)) ↔ x ≠ -1 := by sorry

end fraction_meaningful_l607_60760


namespace sum_of_extremes_even_integers_l607_60715

theorem sum_of_extremes_even_integers (n : ℕ) (z : ℚ) (h_odd : Odd n) (h_pos : 0 < n) :
  ∃ b : ℤ,
    (∀ i : ℕ, i < n → Even (b + 2 * ↑i)) ∧
    z = (↑n * b + ↑n * (↑n - 1)) / ↑n →
    (b + (b + 2 * ↑(n - 1))) = 2 * ↑⌊z⌋ :=
by sorry

end sum_of_extremes_even_integers_l607_60715


namespace factor_expression_l607_60772

theorem factor_expression (m n : ℝ) : 3 * m^2 - 6 * m * n + 3 * n^2 = 3 * (m - n)^2 := by
  sorry

end factor_expression_l607_60772


namespace sleep_stats_l607_60725

def sleep_times : List ℕ := [7, 8, 9, 10]
def frequencies : List ℕ := [6, 9, 11, 4]

def mode (times : List ℕ) (freqs : List ℕ) : ℕ := sorry

def median (times : List ℕ) (freqs : List ℕ) : ℚ := sorry

theorem sleep_stats :
  mode sleep_times frequencies = 9 ∧ 
  median sleep_times frequencies = 17/2 := by sorry

end sleep_stats_l607_60725


namespace plane_through_points_l607_60764

/-- The plane equation coefficients -/
def A : ℤ := 1
def B : ℤ := 2
def C : ℤ := -2
def D : ℤ := -10

/-- The three points on the plane -/
def p : ℝ × ℝ × ℝ := (-2, 3, -3)
def q : ℝ × ℝ × ℝ := (2, 3, -1)
def r : ℝ × ℝ × ℝ := (4, 1, -2)

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem plane_through_points :
  plane_equation p.1 p.2.1 p.2.2 ∧
  plane_equation q.1 q.2.1 q.2.2 ∧
  plane_equation r.1 r.2.1 r.2.2 ∧
  A > 0 ∧
  Nat.gcd (Int.natAbs A) (Int.natAbs B) = 1 ∧
  Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C) = 1 ∧
  Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1 :=
by sorry

end plane_through_points_l607_60764


namespace ratio_problem_l607_60740

theorem ratio_problem (a b c : ℝ) (h1 : b / a = 4) (h2 : c / b = 5) : 
  (a + b) / (b + c) = 5 / 24 := by
sorry

end ratio_problem_l607_60740


namespace average_marks_combined_classes_l607_60711

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) : 
  n1 = 22 →
  n2 = 28 →
  avg1 = 40 →
  avg2 = 60 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 51.2 := by
  sorry

end average_marks_combined_classes_l607_60711


namespace sqrt_square_negative_two_l607_60752

theorem sqrt_square_negative_two : Real.sqrt ((-2)^2) = 2 := by sorry

end sqrt_square_negative_two_l607_60752


namespace bacteria_population_growth_l607_60794

/-- The final population of a bacteria culture after doubling for a given time -/
def finalPopulation (initialPopulation : ℕ) (doublingTime : ℕ) (totalTime : ℕ) : ℕ :=
  initialPopulation * 2^(totalTime / doublingTime)

/-- Theorem: The bacteria population doubles from 1000 to 1,024,000 in 20 minutes -/
theorem bacteria_population_growth :
  finalPopulation 1000 2 20 = 1024000 := by
  sorry

end bacteria_population_growth_l607_60794


namespace det2_trig_equality_l607_60735

-- Define the second-order determinant
def det2 (a b d c : ℝ) : ℝ := a * c - b * d

-- State the theorem
theorem det2_trig_equality : det2 (Real.sin (50 * π / 180)) (Real.cos (40 * π / 180)) (-Real.sqrt 3 * Real.tan (10 * π / 180)) 1 = 1 := by sorry

end det2_trig_equality_l607_60735


namespace largest_prime_fermat_like_under_300_l607_60756

def fermat_like (n : ℕ) : ℕ := 2^n + 1

theorem largest_prime_fermat_like_under_300 :
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    (∃ (n : ℕ), Nat.Prime n ∧ fermat_like n = p) ∧
    p < 300 ∧
    (∀ (q : ℕ), 
      Nat.Prime q → 
      (∃ (m : ℕ), Nat.Prime m ∧ fermat_like m = q) → 
      q < 300 → 
      q ≤ p) ∧
    p = 5 :=
by sorry

end largest_prime_fermat_like_under_300_l607_60756


namespace young_pioneers_tree_planting_l607_60749

theorem young_pioneers_tree_planting (x : ℕ) : 
  (5 * x + 3 = 6 * x - 7) → 
  (x = 10 ∧ 5 * x + 3 = 53) := by
  sorry

end young_pioneers_tree_planting_l607_60749


namespace constant_geometric_sequence_l607_60727

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = q * b n

theorem constant_geometric_sequence
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_geom : is_geometric_sequence b)
  (h_relation : ∀ n : ℕ, a (n + 1) / a n = b n) :
  ∀ n : ℕ, b n = 1 := by
sorry

end constant_geometric_sequence_l607_60727


namespace large_circle_radius_l607_60765

/-- Given two circles A and B with radii 3 and 2 respectively, internally tangent to a larger circle
    at different points, and the distance between the centers of circles A and B is 6,
    the radius of the large circle is (5 + √33) / 2. -/
theorem large_circle_radius (r : ℝ) : r > 0 →
  (r - 3) ^ 2 + (r - 2) ^ 2 + 2 * (r - 3) * (r - 2) = 36 →
  r = (5 + Real.sqrt 33) / 2 := by
  sorry

end large_circle_radius_l607_60765


namespace expected_cells_theorem_l607_60741

/-- A grid of height 2 stretching infinitely in one direction -/
def Grid := ℕ × Fin 2

/-- Probability of a door being locked -/
def p_locked : ℚ := 1/2

/-- Philip's starting position -/
def start : Grid := (0, 0)

/-- Expected number of reachable cells -/
noncomputable def expected_reachable_cells : ℚ := 32/7

/-- Main theorem: The expected number of cells Philip can reach is 32/7 -/
theorem expected_cells_theorem :
  let grid : Type := Grid
  let p_lock : ℚ := p_locked
  let start_pos : grid := start
  expected_reachable_cells = 32/7 := by
  sorry

end expected_cells_theorem_l607_60741


namespace tenth_term_is_21_l607_60703

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_first_three : a 1 + a 2 + a 3 = 15
  geometric : (a 2 + 5)^2 = (a 1 + 2) * (a 3 + 13)

/-- The 10th term of the arithmetic sequence is 21 -/
theorem tenth_term_is_21 (seq : ArithmeticSequence) : seq.a 10 = 21 := by
  sorry

#check tenth_term_is_21

end tenth_term_is_21_l607_60703


namespace six_balls_two_boxes_l607_60787

/-- The number of ways to distribute n indistinguishable balls into 2 indistinguishable boxes -/
def distribution_count (n : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 6 indistinguishable balls into 2 indistinguishable boxes is 4 -/
theorem six_balls_two_boxes : distribution_count 6 = 4 := by sorry

end six_balls_two_boxes_l607_60787


namespace inequality_proof_l607_60721

theorem inequality_proof (a b : ℝ) (ha : -4 < a ∧ a < 0) (hb : -4 < b ∧ b < 0) :
  2 * |a - b| < |a * b + 2 * a + 2 * b| := by
  sorry

end inequality_proof_l607_60721


namespace square_area_equal_perimeter_l607_60783

/-- Given a triangle with sides 7.5 cm, 9.5 cm, and 12 cm, and a square with the same perimeter as this triangle, the area of the square is 52.5625 square centimeters. -/
theorem square_area_equal_perimeter (a b c s : ℝ) : 
  a = 7.5 → b = 9.5 → c = 12 → s * 4 = a + b + c → s^2 = 52.5625 := by
  sorry

end square_area_equal_perimeter_l607_60783


namespace hyperbola_and_line_properties_l607_60724

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the line MN
def line_MN (x y : ℝ) : Prop := 3 * x - 2 * y - 18 = 0

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := 3 * x + 4 * y = 0

-- Theorem statement
theorem hyperbola_and_line_properties :
  ∃ (x y : ℝ), 
    -- The hyperbola passes through (8, 3√3)
    hyperbola 8 (3 * Real.sqrt 3) ∧
    -- The point (8, 3) bisects a chord MN on the hyperbola
    ∃ (x1 y1 x2 y2 : ℝ),
      hyperbola x1 y1 ∧ 
      hyperbola x2 y2 ∧
      line_MN x1 y1 ∧
      line_MN x2 y2 ∧
      (x1 + x2) / 2 = 8 ∧
      (y1 + y2) / 2 = 3 ∧
    -- The asymptotes are given by 3x + 4y = 0
    (∀ (x y : ℝ), asymptotes x y ↔ (x = y ∨ x = -y)) →
    -- The equation of the hyperbola is x²/16 - y²/9 = 1
    (∀ (x y : ℝ), hyperbola x y ↔ x^2 / 16 - y^2 / 9 = 1) ∧
    -- The equation of the line containing MN is 3x - 2y - 18 = 0
    (∀ (x y : ℝ), line_MN x y ↔ 3 * x - 2 * y - 18 = 0) :=
by sorry

end hyperbola_and_line_properties_l607_60724


namespace fruit_group_sizes_l607_60770

def bananas : ℕ := 527
def oranges : ℕ := 386
def apples : ℕ := 319

def banana_groups : ℕ := 11
def orange_groups : ℕ := 103
def apple_groups : ℕ := 17

def banana_group_size : ℕ := bananas / banana_groups
def orange_group_size : ℕ := oranges / orange_groups
def apple_group_size : ℕ := apples / apple_groups

theorem fruit_group_sizes :
  banana_group_size = 47 ∧ orange_group_size = 3 ∧ apple_group_size = 18 :=
sorry

end fruit_group_sizes_l607_60770


namespace c_profit_share_l607_60781

theorem c_profit_share (total_investment : ℕ) (total_profit : ℕ) (c_investment : ℕ) : 
  total_investment = 180000 → 
  total_profit = 60000 → 
  c_investment = 72000 → 
  (c_investment : ℚ) / total_investment * total_profit = 24000 := by
  sorry

end c_profit_share_l607_60781


namespace total_pictures_l607_60776

/-- 
Given that:
- Randy drew 5 pictures
- Peter drew 3 more pictures than Randy
- Quincy drew 20 more pictures than Peter

Prove that the total number of pictures drawn by Randy, Peter, and Quincy is 41.
-/
theorem total_pictures (randy peter quincy : ℕ) : 
  randy = 5 → 
  peter = randy + 3 → 
  quincy = peter + 20 → 
  randy + peter + quincy = 41 := by
  sorry

end total_pictures_l607_60776


namespace equation_solution_l607_60705

theorem equation_solution : 
  ∃ r : ℚ, r = -7/4 ∧ 
  (r^2 - 6*r + 8) / (r^2 - 9*r + 20) = (r^2 - 3*r - 18) / (r^2 - 2*r - 24) :=
by sorry

end equation_solution_l607_60705


namespace mutually_exclusive_events_l607_60708

-- Define the sample space
def Ω : Type := List Bool

-- Define the event "at most one hit"
def at_most_one_hit (ω : Ω) : Prop :=
  ω.length = 2 ∧ (ω.count true ≤ 1)

-- Define the event "two hits"
def two_hits (ω : Ω) : Prop :=
  ω.length = 2 ∧ (ω.count true = 2)

-- Theorem statement
theorem mutually_exclusive_events :
  ∀ ω : Ω, ¬(at_most_one_hit ω ∧ two_hits ω) :=
sorry

end mutually_exclusive_events_l607_60708


namespace time_per_bone_l607_60706

def total_analysis_time : ℕ := 206
def total_bones : ℕ := 206

theorem time_per_bone : 
  (total_analysis_time : ℚ) / (total_bones : ℚ) = 1 := by sorry

end time_per_bone_l607_60706


namespace used_car_lot_vehicles_l607_60774

theorem used_car_lot_vehicles (total_vehicles : ℕ) : 
  (total_vehicles / 3 : ℚ) * 2 + -- tires from motorcycles
  (total_vehicles * 2 / 3 * 3 / 4 : ℚ) * 4 + -- tires from cars without spare
  (total_vehicles * 2 / 3 * 1 / 4 : ℚ) * 5 = 84 → -- tires from cars with spare
  total_vehicles = 24 := by
sorry

end used_car_lot_vehicles_l607_60774


namespace grasshopper_trap_l607_60797

/-- Represents the position of the grasshopper on the number line -/
structure Position :=
  (value : ℚ)

/-- Represents a move of the grasshopper -/
inductive Move
  | Left  : ℕ+ → Move
  | Right : ℕ+ → Move

/-- Defines the result of applying a move to a position -/
def apply_move (p : Position) (m : Move) : Position :=
  match m with
  | Move.Left n  => ⟨p.value - n.val⟩
  | Move.Right n => ⟨p.value + n.val⟩

/-- Theorem stating that for any binary-rational position between 0 and 1,
    there exists a sequence of moves that leads to either 0 or 1 -/
theorem grasshopper_trap (a k : ℕ) (h1 : 0 < a) (h2 : a < 2^k) :
  ∃ (moves : List Move), 
    let final_pos := (moves.foldl apply_move ⟨a / 2^k⟩).value
    final_pos = 0 ∨ final_pos = 1 :=
sorry

end grasshopper_trap_l607_60797


namespace wax_for_feathers_l607_60766

/-- The total amount of wax required for feathers given the current amount and additional amount needed -/
theorem wax_for_feathers (current_amount additional_amount : ℕ) : 
  current_amount = 20 → additional_amount = 146 → 
  current_amount + additional_amount = 166 := by
  sorry

end wax_for_feathers_l607_60766


namespace average_carnations_l607_60793

def bouquet1 : ℕ := 9
def bouquet2 : ℕ := 14
def bouquet3 : ℕ := 13
def total_bouquets : ℕ := 3

theorem average_carnations :
  (bouquet1 + bouquet2 + bouquet3) / total_bouquets = 12 := by
  sorry

end average_carnations_l607_60793


namespace ball_arrangements_l607_60704

def num_balls : ℕ := 5
def num_black_balls : ℕ := 2
def num_colored_balls : ℕ := 3
def balls_in_row : ℕ := 4

theorem ball_arrangements :
  (Nat.factorial balls_in_row) + 
  (num_colored_balls * (Nat.factorial balls_in_row) / 2) = 60 := by
  sorry

end ball_arrangements_l607_60704


namespace dmv_waiting_time_solution_l607_60775

/-- Represents the waiting time problem at the DMV. -/
def DMVWaitingTime (x : ℚ) : Prop :=
  let timeToTakeNumber : ℚ := 20
  let timeWaitingForCall : ℚ := 20 * x + 14
  let totalWaitingTime : ℚ := 114
  (timeToTakeNumber + timeWaitingForCall = totalWaitingTime) ∧
  (timeWaitingForCall / timeToTakeNumber = 47 / 10)

/-- The theorem stating that there exists a solution to the DMV waiting time problem. -/
theorem dmv_waiting_time_solution : ∃ x : ℚ, DMVWaitingTime x := by
  sorry

end dmv_waiting_time_solution_l607_60775


namespace day_one_sales_is_86_l607_60780

/-- The number of cups sold on day one -/
def day_one_sales : ℕ := sorry

/-- The number of cups sold on each of the next 11 days -/
def daily_sales : ℕ := 50

/-- The total number of days -/
def total_days : ℕ := 12

/-- The average daily sales over the 12-day period -/
def average_sales : ℕ := 53

/-- Theorem: The number of cups sold on day one is 86 -/
theorem day_one_sales_is_86 :
  day_one_sales = 86 :=
by
  sorry

end day_one_sales_is_86_l607_60780


namespace root_lines_are_tangents_l607_60767

/-- The Opq plane -/
structure Opq_plane where
  p : ℝ
  q : ℝ

/-- The line given by a^2 + ap + q = 0 for a real number a -/
def root_line (a : ℝ) : Set Opq_plane :=
  {point : Opq_plane | a^2 + a * point.p + point.q = 0}

/-- The discriminant parabola p^2 - 4q = 0 -/
def discriminant_parabola : Set Opq_plane :=
  {point : Opq_plane | point.p^2 - 4 * point.q = 0}

/-- A line is tangent to the parabola if it intersects the parabola at exactly one point -/
def is_tangent (line : Set Opq_plane) : Prop :=
  ∃! point : Opq_plane, point ∈ line ∩ discriminant_parabola

/-- The set of all tangents to the discriminant parabola -/
def all_tangents : Set (Set Opq_plane) :=
  {line : Set Opq_plane | is_tangent line}

/-- The theorem to be proved -/
theorem root_lines_are_tangents :
  {line | ∃ a : ℝ, line = root_line a} = all_tangents :=
sorry

end root_lines_are_tangents_l607_60767


namespace power_of_seven_mod_thousand_l607_60799

theorem power_of_seven_mod_thousand : 7^1984 ≡ 401 [ZMOD 1000] := by sorry

end power_of_seven_mod_thousand_l607_60799


namespace division_problem_l607_60761

theorem division_problem (L S q : ℕ) : 
  L - S = 1335 → 
  L = 1584 → 
  L = S * q + 15 → 
  q = 6 := by
sorry

end division_problem_l607_60761


namespace line_intersects_y_axis_l607_60768

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The y-axis intersection point of a line -/
def yAxisIntersection (l : Line) : ℝ × ℝ := sorry

/-- The specific line passing through (5, 25) and (-5, 5) -/
def specificLine : Line := { x₁ := 5, y₁ := 25, x₂ := -5, y₂ := 5 }

theorem line_intersects_y_axis :
  yAxisIntersection specificLine = (0, 15) := by sorry

end line_intersects_y_axis_l607_60768


namespace triangle_theorem_l607_60782

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  h1 : 0 < a ∧ 0 < b ∧ 0 < c
  h2 : 0 < A ∧ 0 < B ∧ 0 < C
  h3 : A + B + C = π

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) (h : t.a + t.a * Real.cos t.C = Real.sqrt 3 * t.c * Real.sin t.A) :
  t.C = π / 3 ∧
  ∃ (D : ℝ), (D > 0 ∧ 2 * Real.sqrt 3 = D ∧
    ∀ (a b : ℝ), (a > 0 ∧ b > 0 → 2 * a + b ≥ 6 + 4 * Real.sqrt 2)) := by
  sorry

end triangle_theorem_l607_60782


namespace linear_function_third_quadrant_l607_60754

/-- A linear function y = (m-2)x + m-1 does not pass through the third quadrant
    if and only if 1 ≤ m < 2. -/
theorem linear_function_third_quadrant (m : ℝ) :
  (∀ x y : ℝ, y = (m - 2) * x + m - 1 → (x < 0 ∧ y < 0 → False)) ↔ 1 ≤ m ∧ m < 2 :=
sorry

end linear_function_third_quadrant_l607_60754


namespace smallest_max_sum_l607_60784

theorem smallest_max_sum (a b c d e : ℕ+) (sum_eq : a + b + c + d + e = 2023) :
  let M := max (a + b) (max (b + c) (max (c + d) (d + e)))
  405 ≤ M ∧ ∃ (a' b' c' d' e' : ℕ+), a' + b' + c' + d' + e' = 2023 ∧
    max (a' + b') (max (b' + c') (max (c' + d') (d' + e'))) = 405 := by
  sorry

end smallest_max_sum_l607_60784


namespace min_value_sqrt_sum_equality_condition_l607_60712

theorem min_value_sqrt_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 + b^2) / c + (b^2 + c^2) / a + (c^2 + a^2) / b) ≥ Real.sqrt 6 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 + b^2) / c + (b^2 + c^2) / a + (c^2 + a^2) / b) = Real.sqrt 6 ↔ a = b ∧ b = c :=
by sorry

end min_value_sqrt_sum_equality_condition_l607_60712


namespace expression_evaluation_l607_60736

theorem expression_evaluation :
  let x : ℤ := -2
  let y : ℤ := 1
  5 * x^2 - 2*x*y + 3*(x*y + 2) - 1 = 23 :=
by sorry

end expression_evaluation_l607_60736


namespace hexagon_c_x_coordinate_l607_60707

/-- A hexagon with vertices A, B, C, D, E, F in 2D space -/
structure Hexagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- The area of a hexagon -/
def hexagonArea (h : Hexagon) : ℝ := sorry

/-- Check if a hexagon has a horizontal line of symmetry -/
def hasHorizontalSymmetry (h : Hexagon) : Prop := sorry

/-- Given a hexagon with specified properties, prove that the x-coordinate of vertex C is 22 -/
theorem hexagon_c_x_coordinate (h : Hexagon) 
  (sym : hasHorizontalSymmetry h)
  (area : hexagonArea h = 30)
  (vA : h.A = (0, 0))
  (vB : h.B = (0, 2))
  (vD : h.D = (5, 2))
  (vE : h.E = (5, 0))
  (vF : h.F = (2, 0)) :
  h.C.1 = 22 := by sorry

end hexagon_c_x_coordinate_l607_60707


namespace students_not_enrolled_l607_60742

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 69)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 15 := by
  sorry

end students_not_enrolled_l607_60742


namespace not_integer_negative_nine_tenths_l607_60757

theorem not_integer_negative_nine_tenths : ¬ (∃ (n : ℤ), (n : ℚ) = -9/10) := by
  sorry

end not_integer_negative_nine_tenths_l607_60757


namespace support_area_l607_60778

/-- The area of a support satisfying specific mass and area change conditions -/
theorem support_area : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (50 / (x - 5) = 60 / x + 1) ∧ 
  (x = 15) := by
  sorry

end support_area_l607_60778


namespace first_class_average_l607_60785

/-- Proves that given two classes with specified student counts and averages,
    the average of the first class can be determined. -/
theorem first_class_average
  (n1 : ℕ)  -- number of students in the first class
  (n2 : ℕ)  -- number of students in the second class
  (a2 : ℝ)  -- average marks of the second class
  (a_total : ℝ)  -- average marks of all students
  (h1 : n1 = 35)
  (h2 : n2 = 45)
  (h3 : a2 = 60)
  (h4 : a_total = 51.25)
  (h5 : n1 + n2 > 0)  -- to ensure division by zero is avoided
  : ∃ (a1 : ℝ), a1 = 40 ∧ (n1 * a1 + n2 * a2) / (n1 + n2) = a_total :=
sorry

end first_class_average_l607_60785


namespace min_distance_curve_to_line_l607_60792

theorem min_distance_curve_to_line :
  let C₁ := {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1 ∧ p.2 ≠ 0}
  let C₂ := {p : ℝ × ℝ | p.1 + p.2 - 8 = 0}
  (∀ p ∈ C₁, ∃ q ∈ C₂, ∀ r ∈ C₂, dist p q ≤ dist p r) →
  (∃ p ∈ C₁, ∃ q ∈ C₂, dist p q = 3 * Real.sqrt 2) →
  ∀ p ∈ C₁, ∀ q ∈ C₂, dist p q ≥ 3 * Real.sqrt 2 :=
by sorry


end min_distance_curve_to_line_l607_60792


namespace inequality_proof_l607_60713

theorem inequality_proof (x y : ℝ) (h1 : x^2 ≥ y) (h2 : y^2 ≥ x) :
  x / (y^2 + 1) + y / (x^2 + 1) ≤ 1 := by
  sorry

end inequality_proof_l607_60713


namespace max_soap_boxes_in_carton_l607_60750

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 48, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 8, width := 6, height := 5 }

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 300 := by
  sorry

end max_soap_boxes_in_carton_l607_60750


namespace parallelogram_base_l607_60701

/-- The base of a parallelogram with area 960 square cm and height 16 cm is 60 cm -/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 960 ∧ height = 16 ∧ area = base * height → base = 60 := by
  sorry

end parallelogram_base_l607_60701


namespace complex_equation_solution_l607_60723

/-- Given a positive real number a and complex numbers z and ω defined as follows:
    z = (a - i) / (1 - i)
    ω = z(z + i)
    If the imaginary part of ω minus its real part equals 3/2, then a = 2. -/
theorem complex_equation_solution (a : ℝ) (z ω : ℂ) (h1 : a > 0) 
  (h2 : z = (a - Complex.I) / (1 - Complex.I))
  (h3 : ω = z * (z + Complex.I))
  (h4 : ω.im - ω.re = 3/2) : 
  a = 2 := by sorry

end complex_equation_solution_l607_60723


namespace f_shifted_l607_60759

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 + 4*(x + 1) - 5

-- State the theorem
theorem f_shifted (x : ℝ) : f (x + 2) = x^2 + 8*x + 7 := by
  sorry

end f_shifted_l607_60759


namespace unique_three_digit_sum_l607_60795

/-- A three-digit integer with all digits different -/
def ThreeDigitDistinct : Type := { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) }

/-- The theorem stating that there exists a unique three-digit integer with all digits different that, when summed 9 times, equals 2331 -/
theorem unique_three_digit_sum :
  ∃! (n : ThreeDigitDistinct), 9 * n.val = 2331 :=
sorry

end unique_three_digit_sum_l607_60795


namespace democrat_ratio_l607_60751

/-- Proves that the ratio of democrats to total participants is 1:3 given the specified conditions -/
theorem democrat_ratio (total_participants : ℕ) (female_democrats : ℕ) :
  total_participants = 750 →
  female_democrats = 125 →
  (∃ (female_participants male_participants : ℕ),
    female_participants + male_participants = total_participants ∧
    2 * female_democrats = female_participants ∧
    4 * female_democrats = male_participants) →
  (3 * (2 * female_democrats) : ℚ) / total_participants = 1 := by
  sorry

end democrat_ratio_l607_60751


namespace inequality_proof_l607_60719

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^2 - b^2) / (a^2 + b^2) > (a - b) / (a + b) := by
  sorry

end inequality_proof_l607_60719


namespace inequality_equivalence_l607_60710

theorem inequality_equivalence (y : ℝ) : 
  3/20 + |y - 7/40| < 1/4 ↔ 3/40 < y ∧ y < 11/40 := by
  sorry

end inequality_equivalence_l607_60710


namespace curve_point_coordinates_l607_60738

theorem curve_point_coordinates (θ : Real) (x y : Real) :
  0 ≤ θ ∧ θ ≤ π →
  x = 3 * Real.cos θ →
  y = 4 * Real.sin θ →
  y / x = 1 →
  x = 12 / 5 ∧ y = 12 / 5 := by
  sorry

end curve_point_coordinates_l607_60738


namespace inverse_A_cubed_l607_60791

theorem inverse_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) : 
  A⁻¹ = !![3, 7; -2, -4] → (A^3)⁻¹ = !![11, 17; 2, 6] := by
  sorry

end inverse_A_cubed_l607_60791


namespace percentage_relationship_l607_60729

theorem percentage_relationship (x y : ℝ) (h : y = x * 1.6) :
  x = y * (1 - 0.375) :=
by sorry

end percentage_relationship_l607_60729


namespace city_budget_properties_l607_60762

/-- CityBudget represents the budget allocation for a city's rail transit line project over three years. -/
structure CityBudget where
  total : ℝ  -- Total budget in billion yuan
  track_laying : ℝ → ℝ  -- Investment in track laying for each year
  relocation : ℝ → ℝ  -- Investment in relocation for each year
  auxiliary : ℝ → ℝ  -- Investment in auxiliary facilities for each year
  b : ℝ  -- Annual increase in track laying investment

/-- Investment ratios and conditions for the city budget -/
def budget_conditions (budget : CityBudget) : Prop :=
  ∃ x : ℝ,
    budget.track_laying 0 = 2 * x ∧
    budget.relocation 0 = 4 * x ∧
    budget.auxiliary 0 = x ∧
    (∀ t : ℝ, t ≥ 0 ∧ t < 3 → budget.track_laying (t + 1) = budget.track_laying t + budget.b) ∧
    (budget.track_laying 0 + budget.track_laying 1 + budget.track_laying 2 = 54) ∧
    (∃ y : ℝ, y > 0 ∧ y < 1 ∧ 
      budget.relocation 1 = budget.relocation 0 * (1 - y) ∧
      budget.relocation 2 = budget.relocation 1 * (1 - y) ∧
      budget.relocation 2 = 5) ∧
    (budget.auxiliary 1 = budget.auxiliary 0 * (1 + 1.5 * budget.b / (2 * x))) ∧
    (budget.auxiliary 2 = budget.auxiliary 0 + budget.auxiliary 1 + 4) ∧
    (budget.track_laying 0 + budget.track_laying 1 + budget.track_laying 2) / 
    (budget.auxiliary 0 + budget.auxiliary 1 + budget.auxiliary 2) = 3 / 2

/-- Main theorem stating the properties of the city budget -/
theorem city_budget_properties (budget : CityBudget) (h : budget_conditions budget) :
  (budget.auxiliary 0 + budget.auxiliary 1 + budget.auxiliary 2 = 36) ∧
  (budget.track_laying 0 + budget.relocation 0 + budget.auxiliary 0 = 35) ∧
  (∃ y : ℝ, y = 0.5 ∧ 
    budget.relocation 1 = budget.relocation 0 * (1 - y) ∧
    budget.relocation 2 = budget.relocation 1 * (1 - y)) := by
  sorry

end city_budget_properties_l607_60762


namespace at_least_one_geq_six_l607_60726

theorem at_least_one_geq_six (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 4 / b ≥ 6) ∨ (b + 9 / c ≥ 6) ∨ (c + 16 / a ≥ 6) := by
  sorry

end at_least_one_geq_six_l607_60726


namespace inverse_of_A_cubed_l607_60732

theorem inverse_of_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A⁻¹ = !![5, -2; 1, 3] →
  (A^3)⁻¹ = !![111, -34; 47, 5] := by
sorry

end inverse_of_A_cubed_l607_60732


namespace midpoint_coordinate_product_l607_60755

/-- The product of the coordinates of the midpoint of a line segment
    with endpoints (4, -1) and (-8, 7) is -6. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 4
  let y1 : ℝ := -1
  let x2 : ℝ := -8
  let y2 : ℝ := 7
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = -6 := by
  sorry

end midpoint_coordinate_product_l607_60755


namespace jack_needs_five_rocks_l607_60796

/-- The number of rocks needed to equalize weights on a see-saw -/
def rocks_needed (jack_weight anna_weight rock_weight : ℕ) : ℕ :=
  (jack_weight - anna_weight) / rock_weight

/-- Theorem: Jack needs 5 rocks to equalize weights with Anna -/
theorem jack_needs_five_rocks :
  rocks_needed 60 40 4 = 5 := by
  sorry

end jack_needs_five_rocks_l607_60796


namespace x_powers_sum_l607_60700

theorem x_powers_sum (x : ℝ) (h : 47 = x^4 + 1/x^4) : 
  (x^2 + 1/x^2 = 7) ∧ (x^8 + 1/x^8 = -433) := by
  sorry

end x_powers_sum_l607_60700


namespace reciprocal_inequality_l607_60748

theorem reciprocal_inequality (a b : ℝ) (h1 : a > 0) (h2 : 0 > b) : 1 / a > 1 / b := by
  sorry

end reciprocal_inequality_l607_60748


namespace color_combinations_l607_60798

/-- Represents the number of color choices for each dot -/
def num_colors : ℕ := 3

/-- Represents the number of ways to color a single triangle -/
def ways_per_triangle : ℕ := 6

/-- Represents the number of triangles in the figure -/
def num_triangles : ℕ := 3

/-- Represents the total number of dots in the figure -/
def total_dots : ℕ := 10

/-- Theorem stating the number of ways to color the figure -/
theorem color_combinations : 
  (ways_per_triangle ^ num_triangles : ℕ) = 216 :=
sorry

end color_combinations_l607_60798


namespace fifty_black_reachable_l607_60731

/-- Represents the number of marbles of each color in the urn -/
structure UrnState where
  white : ℕ
  black : ℕ

/-- The initial state of the urn -/
def initial_state : UrnState := ⟨50, 150⟩

/-- Applies one of the four possible operations to the urn -/
def apply_operation (state : UrnState) : UrnState :=
  match state with
  | ⟨w, b⟩ => 
    if b ≥ 3 then ⟨w, b - 1⟩  -- Operation 1
    else if b ≥ 2 ∧ w ≥ 1 then ⟨w, b - 1⟩  -- Operation 2
    else if b ≥ 1 ∧ w ≥ 2 then state  -- Operation 3
    else if w ≥ 3 then ⟨w - 2, b⟩  -- Operation 4
    else state  -- No operation possible

/-- Checks if the given state is reachable from the initial state -/
def is_reachable (state : UrnState) : Prop :=
  ∃ n : ℕ, (n.iterate apply_operation initial_state).black = state.black

/-- The theorem to be proven -/
theorem fifty_black_reachable :
  ∃ w : ℕ, is_reachable ⟨w, 50⟩ :=
sorry

end fifty_black_reachable_l607_60731


namespace constant_term_equality_l607_60709

theorem constant_term_equality (a : ℝ) : 
  (∃ k : ℕ, (Nat.choose 9 k) * 2^k = 84 * 64 ∧ 
   18 - 3 * k = 0) →
  (∃ r : ℕ, (Nat.choose 9 r) * a^r = 84 * 64 ∧
   9 - 3 * r = 0) →
  a = 4 := by sorry

end constant_term_equality_l607_60709


namespace probability_one_absent_one_present_l607_60777

/-- The probability of a student being absent on any given day -/
def p_absent : ℚ := 1 / 20

/-- The probability of a student being present on any given day -/
def p_present : ℚ := 1 - p_absent

/-- The probability that among any two randomly selected students, one is absent and the other present on a given day -/
def p_one_absent_one_present : ℚ := 2 * p_absent * p_present

theorem probability_one_absent_one_present :
  p_one_absent_one_present = 19 / 200 :=
sorry

end probability_one_absent_one_present_l607_60777


namespace rectangle_area_l607_60745

/-- The area of a rectangle with an inscribed circle of radius 5 and length-to-width ratio of 2:1 -/
theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 5 → ratio = 2 → 2 * r * ratio * r = 200 := by
  sorry

end rectangle_area_l607_60745


namespace seven_cupcakes_left_l607_60743

/-- The number of cupcakes left after eating some from multiple packages -/
def cupcakes_left (packages : ℕ) (cupcakes_per_package : ℕ) (eaten : ℕ) : ℕ :=
  packages * cupcakes_per_package - eaten

/-- Proof that 7 cupcakes are left given the initial conditions -/
theorem seven_cupcakes_left :
  cupcakes_left 3 4 5 = 7 := by
  sorry

end seven_cupcakes_left_l607_60743


namespace temperature_conversion_l607_60717

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 75 → k = 167 := by
  sorry

end temperature_conversion_l607_60717


namespace triangle_properties_l607_60753

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  C = π / 3 →
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 →
  (∃ (R : ℝ), 2 * R = c / Real.sin C ∧ 2 * R = b / Real.sin B ∧ 2 * R = a / Real.sin A) →
  (a = 2 ∧ b = 2) ∧
  (∀ (a' b' : ℝ), b' / 2 + a' ≤ 2 * Real.sqrt 21 / 3) :=
by sorry

end triangle_properties_l607_60753


namespace factorial_100_trailing_zeros_l607_60779

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- 100! has 24 trailing zeros -/
theorem factorial_100_trailing_zeros :
  trailingZeros 100 = 24 := by
  sorry

end factorial_100_trailing_zeros_l607_60779


namespace number_of_girls_l607_60737

theorem number_of_girls (total_children happy_children sad_children neutral_children boys happy_boys sad_girls : ℕ) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  boys = 18 →
  happy_boys = 6 →
  sad_girls = 4 →
  happy_children + sad_children + neutral_children = total_children →
  total_children - boys = 42 := by
  sorry

end number_of_girls_l607_60737


namespace banana_permutations_l607_60716

-- Define the word and its properties
def banana_length : ℕ := 6
def banana_a_count : ℕ := 3
def banana_n_count : ℕ := 2
def banana_b_count : ℕ := 1

-- Define the function to calculate permutations with repetition
def permutations_with_repetition (n : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial n / (repetitions.map Nat.factorial).prod

-- Theorem statement
theorem banana_permutations :
  permutations_with_repetition banana_length [banana_a_count, banana_n_count, banana_b_count] = 60 := by
  sorry


end banana_permutations_l607_60716


namespace power_mod_theorem_l607_60789

theorem power_mod_theorem : 2^2001 ≡ 64 [MOD (2^7 - 1)] := by sorry

end power_mod_theorem_l607_60789


namespace total_weekend_hours_l607_60763

-- Define the working hours for Saturday and Sunday
def saturday_hours : ℕ := 6
def sunday_hours : ℕ := 4

-- Theorem to prove
theorem total_weekend_hours :
  saturday_hours + sunday_hours = 10 := by
  sorry

end total_weekend_hours_l607_60763


namespace handshaking_arrangements_mod_1000_l607_60722

/-- A handshaking arrangement for a group of people -/
structure HandshakingArrangement (n : ℕ) where
  shakes : Fin n → Finset (Fin n)
  shake_count : ∀ i, (shakes i).card = 3
  symmetry : ∀ i j, i ∈ shakes j ↔ j ∈ shakes i

/-- The number of distinct handshaking arrangements for 12 people -/
def num_arrangements : ℕ := sorry

theorem handshaking_arrangements_mod_1000 :
  num_arrangements % 1000 = 250 := by sorry

end handshaking_arrangements_mod_1000_l607_60722


namespace cos_minus_sin_nine_pi_fourths_l607_60790

theorem cos_minus_sin_nine_pi_fourths : 
  Real.cos (-9 * Real.pi / 4) - Real.sin (-9 * Real.pi / 4) = Real.sqrt 2 := by
  sorry

end cos_minus_sin_nine_pi_fourths_l607_60790


namespace function_composition_l607_60788

/-- Given f(x) = 2x + 3 and g(x + 2) = f(x), prove that g(x) = 2x - 1 -/
theorem function_composition (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 2 * x + 3)
  (hg : ∀ x, g (x + 2) = f x) :
  ∀ x, g x = 2 * x - 1 := by sorry

end function_composition_l607_60788


namespace bert_pencil_usage_l607_60746

/-- The number of words Bert writes to use up a pencil -/
def words_per_pencil (puzzles_per_day : ℕ) (days_per_pencil : ℕ) (words_per_puzzle : ℕ) : ℕ :=
  puzzles_per_day * days_per_pencil * words_per_puzzle

/-- Theorem stating that Bert writes 1050 words to use up a pencil -/
theorem bert_pencil_usage :
  words_per_pencil 1 14 75 = 1050 := by
  sorry

end bert_pencil_usage_l607_60746


namespace birdhouse_nails_count_l607_60786

/-- The number of planks required to build one birdhouse -/
def planks_per_birdhouse : ℕ := 7

/-- The cost of one nail in dollars -/
def nail_cost : ℚ := 0.05

/-- The cost of one plank in dollars -/
def plank_cost : ℕ := 3

/-- The total cost to build 4 birdhouses in dollars -/
def total_cost_4_birdhouses : ℕ := 88

/-- The number of birdhouses built -/
def num_birdhouses : ℕ := 4

/-- The number of nails required to build one birdhouse -/
def nails_per_birdhouse : ℕ := 20

theorem birdhouse_nails_count :
  nails_per_birdhouse * num_birdhouses * nail_cost +
  planks_per_birdhouse * num_birdhouses * plank_cost =
  total_cost_4_birdhouses := by sorry

end birdhouse_nails_count_l607_60786


namespace black_area_after_changes_l607_60758

/-- Represents the fraction of area that remains black after each change -/
def remaining_black_fraction : ℚ := 8 / 9

/-- The number of times the process is repeated -/
def num_changes : ℕ := 6

/-- The fraction of the original area that remains black after all changes -/
def final_black_fraction : ℚ := remaining_black_fraction ^ num_changes

theorem black_area_after_changes : 
  final_black_fraction = 262144 / 531441 := by sorry

end black_area_after_changes_l607_60758


namespace max_score_calculation_l607_60769

/-- Given the percentages scored by three students and their average score,
    calculate the maximum possible score in the exam. -/
theorem max_score_calculation (p1 p2 p3 avg : ℝ) 
    (h1 : p1 = 64)
    (h2 : p2 = 36)
    (h3 : p3 = 44)
    (h4 : avg = 432)
    (h5 : (p1 + p2 + p3) / 300 * max_score = avg) :
  max_score = 900 := by
  sorry

#check max_score_calculation

end max_score_calculation_l607_60769


namespace cylinder_surface_area_l607_60730

/-- The total surface area of a cylinder formed by rolling a rectangle -/
theorem cylinder_surface_area (rectangle_length : ℝ) (rectangle_width : ℝ) 
  (h1 : rectangle_length = 4 * Real.pi)
  (h2 : rectangle_width = 2) : 
  2 * Real.pi * (rectangle_width / 2)^2 + rectangle_length * rectangle_width = 16 * Real.pi := by
  sorry

end cylinder_surface_area_l607_60730


namespace total_work_seconds_l607_60733

/-- The total number of seconds worked by four people given their work hours relationships -/
theorem total_work_seconds 
  (bianca_hours : ℝ) 
  (h1 : bianca_hours = 12.5)
  (h2 : ∃ celeste_hours : ℝ, celeste_hours = 2 * bianca_hours)
  (h3 : ∃ mcclain_hours : ℝ, mcclain_hours = celeste_hours - 8.5)
  (h4 : ∃ omar_hours : ℝ, omar_hours = bianca_hours + 3)
  : ∃ total_seconds : ℝ, total_seconds = 250200 := by
  sorry


end total_work_seconds_l607_60733


namespace evaluate_expression_l607_60720

theorem evaluate_expression : (-3)^4 - (-3)^3 + (-3)^2 - 3^2 + 3^3 - 3^4 = 9 := by
  sorry

end evaluate_expression_l607_60720


namespace quadratic_equation_range_l607_60728

/-- The range of k for which the quadratic equation (k-1)x^2 - 2x + 1 = 0 has two distinct real roots -/
theorem quadratic_equation_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (k - 1) * x₁^2 - 2 * x₁ + 1 = 0 ∧ 
   (k - 1) * x₂^2 - 2 * x₂ + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ 1) :=
by sorry

end quadratic_equation_range_l607_60728


namespace sum_of_solutions_quadratic_l607_60744

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 - 6*x - 22 = 2*x + 18) → 
  (∃ x₁ x₂ : ℝ, (x₁ + x₂ = 8) ∧ (x₁ * x₂ = x₁ + x₂ - 8)) :=
by sorry

end sum_of_solutions_quadratic_l607_60744


namespace greatest_difference_l607_60714

theorem greatest_difference (n m : ℕ+) (h : 1023 = 17 * n + m) : 
  ∃ (n' m' : ℕ+), 1023 = 17 * n' + m' ∧ ∀ (a b : ℕ+), 1023 = 17 * a + b → (n' : ℤ) - m' ≥ (a : ℤ) - b :=
by sorry

end greatest_difference_l607_60714


namespace number_division_problem_l607_60747

theorem number_division_problem (x y : ℝ) : 
  (x - 5) / y = 7 → (x - 24) / 10 = 3 → y = 7 := by
  sorry

end number_division_problem_l607_60747


namespace remaining_cooking_time_l607_60771

theorem remaining_cooking_time (recommended_time_minutes : ℕ) (cooked_time_seconds : ℕ) : 
  recommended_time_minutes = 5 → cooked_time_seconds = 45 → 
  recommended_time_minutes * 60 - cooked_time_seconds = 255 := by
  sorry

end remaining_cooking_time_l607_60771


namespace reading_calculation_l607_60739

/-- Calculates the number of characters read in a given number of days -/
def charactersRead (dailyReading : ℕ) (days : ℕ) : ℕ :=
  dailyReading * days

/-- Approximates a number to the nearest ten-thousand -/
def approximateToTenThousand (n : ℕ) : ℕ :=
  n / 10000

theorem reading_calculation (dailyReading : ℕ) 
  (h : dailyReading = 800) : 
  let weeklyReading := charactersRead dailyReading 7
  let twentyWeeksReading := charactersRead weeklyReading 20
  weeklyReading = 5600 ∧ 
  twentyWeeksReading = 112000 ∧ 
  approximateToTenThousand twentyWeeksReading = 11 := by
  sorry

end reading_calculation_l607_60739


namespace alcohol_mixture_percentage_l607_60734

/-- Proves that mixing 8 liters of 25% alcohol solution with 2 liters of 12% alcohol solution results in a 22.4% alcohol solution -/
theorem alcohol_mixture_percentage :
  let volume1 : ℝ := 8
  let concentration1 : ℝ := 0.25
  let volume2 : ℝ := 2
  let concentration2 : ℝ := 0.12
  let total_volume : ℝ := volume1 + volume2
  let total_alcohol : ℝ := volume1 * concentration1 + volume2 * concentration2
  total_alcohol / total_volume = 0.224 := by
sorry


end alcohol_mixture_percentage_l607_60734
