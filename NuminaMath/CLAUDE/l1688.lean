import Mathlib

namespace cubic_value_l1688_168825

theorem cubic_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2010 = 2011 := by
  sorry

end cubic_value_l1688_168825


namespace line_passes_through_point_l1688_168871

/-- The line equation passes through the point (3, 1) for all values of m -/
theorem line_passes_through_point :
  ∀ (m : ℝ), (2 * m + 1) * 3 + (m + 1) * 1 - 7 * m - 4 = 0 := by
  sorry

end line_passes_through_point_l1688_168871


namespace tan_105_degrees_l1688_168853

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_degrees_l1688_168853


namespace real_root_of_cubic_l1688_168815

theorem real_root_of_cubic (a b c : ℂ) (h_a_real : a.im = 0)
  (h_sum : a + b + c = 5)
  (h_sum_prod : a * b + b * c + c * a = 7)
  (h_prod : a * b * c = 2) :
  a = 2 := by
sorry

end real_root_of_cubic_l1688_168815


namespace no_cover_with_changed_tiles_l1688_168840

/-- Represents a rectangular floor -/
structure Floor :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a tile configuration -/
structure TileConfig :=
  (twoBytwo : ℕ)  -- number of 2x2 tiles
  (fourByOne : ℕ) -- number of 4x1 tiles

/-- Predicate to check if a floor can be covered by a given tile configuration -/
def canCover (f : Floor) (tc : TileConfig) : Prop :=
  4 * tc.twoBytwo + 4 * tc.fourByOne = f.length * f.width

/-- Main theorem: If a floor can be covered by a tile configuration,
    it cannot be covered by changing the number of tiles by ±1 for each type -/
theorem no_cover_with_changed_tiles (f : Floor) (tc : TileConfig) :
  canCover f tc →
  ¬(canCover f { twoBytwo := tc.twoBytwo + 1, fourByOne := tc.fourByOne - 1 } ∨
    canCover f { twoBytwo := tc.twoBytwo - 1, fourByOne := tc.fourByOne + 1 }) :=
by
  sorry

#check no_cover_with_changed_tiles

end no_cover_with_changed_tiles_l1688_168840


namespace continued_fraction_solution_l1688_168829

/-- The solution to the equation x = 3 + 9 / (2 + 9 / x) -/
theorem continued_fraction_solution :
  ∃ x : ℝ, x = 3 + 9 / (2 + 9 / x) ∧ x = (3 + 3 * Real.sqrt 7) / 2 := by
  sorry

end continued_fraction_solution_l1688_168829


namespace quadratic_always_positive_triangle_angle_sum_product_zero_implies_factor_zero_factors_nonzero_implies_x_not_roots_l1688_168822

-- Statement 1
theorem quadratic_always_positive : ∀ x : ℝ, x^2 - x + 1 > 0 := by sorry

-- Statement 2
theorem triangle_angle_sum : ∀ a b c : ℝ, 
  0 < a ∧ 0 < b ∧ 0 < c → a + b + c = 180 := by sorry

-- Statement 3
theorem product_zero_implies_factor_zero : ∀ a b c : ℝ, 
  a * b * c = 0 → a = 0 ∨ b = 0 ∨ c = 0 := by sorry

-- Statement 4
theorem factors_nonzero_implies_x_not_roots : ∀ x : ℝ, 
  (x - 1) * (x - 2) ≠ 0 → x ≠ 1 ∧ x ≠ 2 := by sorry

end quadratic_always_positive_triangle_angle_sum_product_zero_implies_factor_zero_factors_nonzero_implies_x_not_roots_l1688_168822


namespace line_through_two_points_l1688_168875

theorem line_through_two_points (m n p : ℝ) :
  (m = 3 * n + 5) ∧ (m + 2 = 3 * (n + p) + 5) → p = 2 / 3 := by
  sorry

end line_through_two_points_l1688_168875


namespace arithmetic_sequence_sum_l1688_168868

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 2 = 3 →
  a 7 + a 8 = 27 →
  a 9 + a 10 = 35 := by
  sorry

end arithmetic_sequence_sum_l1688_168868


namespace isabel_bouquets_l1688_168898

/-- Given the total number of flowers, flowers per bouquet, and wilted flowers,
    calculate the maximum number of full bouquets that can be made. -/
def max_bouquets (total : ℕ) (per_bouquet : ℕ) (wilted : ℕ) : ℕ :=
  (total - wilted) / per_bouquet

/-- Theorem stating that given 132 total flowers, 11 flowers per bouquet,
    and 16 wilted flowers, the maximum number of full bouquets is 10. -/
theorem isabel_bouquets :
  max_bouquets 132 11 16 = 10 := by
  sorry

end isabel_bouquets_l1688_168898


namespace min_value_of_expression_min_value_achieved_l1688_168888

theorem min_value_of_expression (x : ℝ) : 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) ≥ -6452.25 :=
by sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) < -6452.25 + ε :=
by sorry

end min_value_of_expression_min_value_achieved_l1688_168888


namespace absolute_value_equation_l1688_168893

theorem absolute_value_equation (x z : ℝ) 
  (h : |2*x - Real.log z| = 2*x + Real.log z) : x * (z - 1) = 0 := by
  sorry

end absolute_value_equation_l1688_168893


namespace geometric_sequence_increasing_condition_l1688_168837

/-- A geometric sequence with first term a and common ratio q -/
def geometricSequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

/-- A sequence is monotonically increasing -/
def MonotonicallyIncreasing (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, s n ≤ s (n + 1)

/-- The condition q > 1 is neither necessary nor sufficient for a geometric sequence to be monotonically increasing -/
theorem geometric_sequence_increasing_condition (a q : ℝ) :
  ¬(((q > 1) ↔ MonotonicallyIncreasing (geometricSequence a q))) :=
sorry

end geometric_sequence_increasing_condition_l1688_168837


namespace oliver_final_balance_l1688_168808

def oliver_money_problem (initial_amount : ℝ) (allowance_savings : ℝ) (chore_earnings : ℝ)
  (frisbee_cost : ℝ) (puzzle_cost : ℝ) (sticker_cost : ℝ)
  (movie_ticket_price : ℝ) (movie_discount_percent : ℝ)
  (snack_price : ℝ) (snack_coupon : ℝ)
  (birthday_gift : ℝ) : Prop :=
  let total_expenses := frisbee_cost + puzzle_cost + sticker_cost
  let discounted_movie_price := movie_ticket_price * (1 - movie_discount_percent / 100)
  let snack_cost := snack_price - snack_coupon
  let final_balance := initial_amount + allowance_savings + chore_earnings - 
                       total_expenses - discounted_movie_price - snack_cost + birthday_gift
  final_balance = 9

theorem oliver_final_balance :
  oliver_money_problem 9 5 6 4 3 2 10 20 3 1 8 :=
by sorry

end oliver_final_balance_l1688_168808


namespace area_equality_l1688_168814

-- Define the points
variable (A B C D E F G H : ℝ × ℝ)

-- Define the convex quadrilateral ABCD
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

-- Define midpoints
def is_midpoint (M X Y : ℝ × ℝ) : Prop := sorry

-- Define intersection of lines
def is_intersection (P X₁ Y₁ X₂ Y₂ : ℝ × ℝ) : Prop := sorry

-- Define area of a triangle
def area_triangle (X Y Z : ℝ × ℝ) : ℝ := sorry

-- Define area of a quadrilateral
def area_quadrilateral (W X Y Z : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_equality 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_E_midpoint : is_midpoint E A B)
  (h_F_midpoint : is_midpoint F C D)
  (h_G_intersection : is_intersection G A F D E)
  (h_H_intersection : is_intersection H B F C E) :
  area_triangle A G D + area_triangle B H C = area_quadrilateral E H F G := 
sorry

end area_equality_l1688_168814


namespace cubic_function_properties_l1688_168890

def f (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem cubic_function_properties (b c d : ℝ) :
  f b c d 0 = 2 ∧ 
  (∀ x, (6:ℝ)*x - f b c d (-1) + 7 = 0 ↔ x = -1) →
  f b c d (-1) = 1 ∧
  ∀ x, f b c d x = x^3 - 3*x^2 - 3*x + 2 :=
by sorry

end cubic_function_properties_l1688_168890


namespace complex_sum_reciprocal_imag_part_l1688_168844

theorem complex_sum_reciprocal_imag_part :
  let z : ℂ := 2 + I
  (z + z⁻¹).im = 4/5 := by sorry

end complex_sum_reciprocal_imag_part_l1688_168844


namespace average_weight_after_student_leaves_l1688_168859

/-- Represents the class of students with their weights before and after a student leaves -/
structure ClassWeights where
  totalStudents : Nat
  maleWeightSum : ℝ
  femaleWeightSum : ℝ
  leavingStudentWeight : ℝ
  weightIncreaseAfterLeaving : ℝ

/-- Theorem stating the average weight of remaining students after one leaves -/
theorem average_weight_after_student_leaves (c : ClassWeights)
  (h1 : c.totalStudents = 60)
  (h2 : c.leavingStudentWeight = 45)
  (h3 : c.weightIncreaseAfterLeaving = 0.2) :
  (c.maleWeightSum + c.femaleWeightSum - c.leavingStudentWeight) / (c.totalStudents - 1) = 57 := by
  sorry


end average_weight_after_student_leaves_l1688_168859


namespace equation_solution_l1688_168862

theorem equation_solution (x : ℝ) : 
  (5 * x^2 - 3) / (x + 3) - 5 / (x + 3) = 6 / (x + 3) → 
  x = Real.sqrt 70 / 5 ∨ x = -Real.sqrt 70 / 5 :=
by sorry

end equation_solution_l1688_168862


namespace age_problem_l1688_168838

theorem age_problem (age1 age2 : ℕ) : 
  age1 + age2 = 63 →
  age1 = 2 * (age2 - (age1 - age2)) →
  (age1 = 36 ∧ age2 = 27) :=
by sorry

end age_problem_l1688_168838


namespace simplify_and_rationalize_l1688_168850

theorem simplify_and_rationalize (x : ℝ) :
  x = 1 / (2 - 1 / (Real.sqrt 5 + 2)) →
  x = (4 + Real.sqrt 5) / 11 := by
sorry

end simplify_and_rationalize_l1688_168850


namespace old_manufacturing_cost_l1688_168867

/-- Proves that the old manufacturing cost was $65 given the conditions of the problem -/
theorem old_manufacturing_cost (selling_price : ℝ) (new_manufacturing_cost : ℝ) : 
  selling_price = 100 →
  new_manufacturing_cost = 50 →
  (selling_price - new_manufacturing_cost) / selling_price = 0.5 →
  (selling_price - 0.65 * selling_price) = 65 :=
by sorry

end old_manufacturing_cost_l1688_168867


namespace daves_initial_apps_l1688_168800

theorem daves_initial_apps (initial_files : ℕ) (apps_left : ℕ) (files_left : ℕ) : 
  initial_files = 24 →
  apps_left = 21 →
  files_left = 4 →
  apps_left = files_left + 17 →
  ∃ initial_apps : ℕ, initial_apps = 41 :=
by
  sorry

end daves_initial_apps_l1688_168800


namespace hyperbola_vertex_distance_l1688_168818

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

-- Define the distance between vertices
def vertex_distance : ℝ := 8

-- Theorem statement
theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ), hyperbola_equation x y → vertex_distance = 8 := by
  sorry

end hyperbola_vertex_distance_l1688_168818


namespace train_passing_time_l1688_168849

/-- Proves that the time for train A to pass train B is 7.5 seconds given the conditions -/
theorem train_passing_time (length_A length_B : ℝ) (time_B_passes_A : ℝ) 
  (h1 : length_A = 150)
  (h2 : length_B = 200)
  (h3 : time_B_passes_A = 10) :
  (length_A / (length_B / time_B_passes_A)) = 7.5 := by
  sorry

end train_passing_time_l1688_168849


namespace speech_competition_proof_l1688_168832

def scores : List ℝ := [91, 89, 88, 92, 90]

theorem speech_competition_proof :
  let n : ℕ := 5
  let avg : ℝ := 90
  let variance : ℝ := (1 : ℝ) / n * (scores.map (λ x => (x - avg)^2)).sum
  (scores.sum / n = avg) ∧ (variance = 2) :=
by sorry

end speech_competition_proof_l1688_168832


namespace M_intersect_N_equals_zero_one_half_open_l1688_168892

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

-- Define set N
def N : Set ℝ := {x : ℝ | x < 1}

-- Theorem statement
theorem M_intersect_N_equals_zero_one_half_open : M ∩ N = Set.Icc 0 1 ∩ Set.Iio 1 := by sorry

end M_intersect_N_equals_zero_one_half_open_l1688_168892


namespace circular_track_length_l1688_168843

/-- Represents a circular track with two runners --/
structure CircularTrack where
  length : ℝ
  runner1_speed : ℝ
  runner2_speed : ℝ

/-- Represents a meeting point of the runners --/
structure MeetingPoint where
  distance1 : ℝ  -- Distance run by runner 1
  distance2 : ℝ  -- Distance run by runner 2

/-- The theorem to be proved --/
theorem circular_track_length
  (track : CircularTrack)
  (first_meeting : MeetingPoint)
  (second_meeting : MeetingPoint) :
  (first_meeting.distance1 = 100) →
  (second_meeting.distance2 - first_meeting.distance2 = 150) →
  (track.runner1_speed > 0) →
  (track.runner2_speed > 0) →
  (track.length = 500) :=
by sorry

end circular_track_length_l1688_168843


namespace fraction_simplification_l1688_168878

theorem fraction_simplification : (3 * 4) / 6 = 2 := by
  sorry

end fraction_simplification_l1688_168878


namespace rotating_triangle_path_length_l1688_168865

/-- The total path length of point A in a rotating triangle -/
theorem rotating_triangle_path_length (α : ℝ) (h1 : 0 < α) (h2 : α < π / 3) :
  let triangle_rotation := (2 / 3 * π * (1 + Real.sin α) - 2 * α)
  (100 - 1) / 3 * triangle_rotation = 22 * π * (1 + Real.sin α) - 66 * α :=
by sorry

end rotating_triangle_path_length_l1688_168865


namespace rectangle_perimeter_l1688_168835

theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : w = 6) : 
  let triangle_area := (1/2) * a * b
  let rectangle_length := triangle_area / w
  2 * (w + rectangle_length) = 30 := by sorry

end rectangle_perimeter_l1688_168835


namespace swimmers_speed_l1688_168848

/-- Proves that a swimmer's speed in still water is 12 km/h given the conditions -/
theorem swimmers_speed (v s : ℝ) (h1 : s = 4) (h2 : (v - s)⁻¹ = 2 * (v + s)⁻¹) : v = 12 := by
  sorry

end swimmers_speed_l1688_168848


namespace sector_perimeter_l1688_168806

theorem sector_perimeter (r : ℝ) (S : ℝ) (h1 : r = 2) (h2 : S = 8) :
  let α := 2 * S / r^2
  let L := r * α
  r + r + L = 12 := by sorry

end sector_perimeter_l1688_168806


namespace coin_and_die_probability_l1688_168877

theorem coin_and_die_probability : 
  let n : ℕ := 12  -- number of coin flips
  let k : ℕ := 9   -- number of heads
  let d : ℕ := 6   -- number of sides on the die
  let p_coin : ℚ := 1/2  -- probability of heads on a fair coin
  let p_die : ℚ := 1/d  -- probability of rolling a 6 on a fair die
  (Nat.choose n k * p_coin^k * (1 - p_coin)^(n - k)) * p_die = 55/6144 :=
by sorry

end coin_and_die_probability_l1688_168877


namespace bouncy_balls_per_package_l1688_168839

theorem bouncy_balls_per_package (red_packs green_packs yellow_packs total_balls : ℕ) 
  (h1 : red_packs = 4)
  (h2 : yellow_packs = 8)
  (h3 : green_packs = 4)
  (h4 : total_balls = 160) :
  ∃ (balls_per_pack : ℕ), 
    balls_per_pack * (red_packs + yellow_packs + green_packs) = total_balls ∧ 
    balls_per_pack = 10 := by
  sorry

end bouncy_balls_per_package_l1688_168839


namespace specific_building_height_l1688_168805

/-- The height of a building with varying story heights -/
def building_height (total_stories : ℕ) (base_height : ℕ) (height_increase : ℕ) : ℕ :=
  let first_half := total_stories / 2
  let second_half := total_stories - first_half
  (first_half * base_height) + (second_half * (base_height + height_increase))

/-- Theorem stating the height of the specific building described in the problem -/
theorem specific_building_height :
  building_height 20 12 3 = 270 := by
  sorry

end specific_building_height_l1688_168805


namespace undergrads_playing_sports_l1688_168807

theorem undergrads_playing_sports (total_students : ℕ) 
  (grad_percent : ℚ) (grad_not_playing : ℚ) (undergrad_not_playing : ℚ) 
  (total_not_playing : ℚ) :
  total_students = 800 →
  grad_percent = 1/4 →
  grad_not_playing = 1/2 →
  undergrad_not_playing = 1/5 →
  total_not_playing = 3/10 →
  (total_students : ℚ) * (1 - grad_percent) * (1 - undergrad_not_playing) = 480 :=
by sorry

end undergrads_playing_sports_l1688_168807


namespace random_events_count_l1688_168826

theorem random_events_count (total_events : ℕ) 
  (prob_certain : ℚ) (prob_impossible : ℚ) :
  total_events = 10 →
  prob_certain = 2 / 10 →
  prob_impossible = 3 / 10 →
  (total_events : ℚ) * prob_certain + 
  (total_events : ℚ) * prob_impossible + 
  (total_events - 
    (total_events * prob_certain).floor - 
    (total_events * prob_impossible).floor : ℚ) = total_events →
  total_events - 
    (total_events * prob_certain).floor - 
    (total_events * prob_impossible).floor = 5 := by
  sorry

#check random_events_count

end random_events_count_l1688_168826


namespace sin_330_degrees_l1688_168804

theorem sin_330_degrees :
  Real.sin (330 * π / 180) = -(1 / 2) := by sorry

end sin_330_degrees_l1688_168804


namespace cos_sin_shift_l1688_168881

theorem cos_sin_shift (x : ℝ) : 
  Real.cos (x + 2 * Real.pi / 3) = Real.sin (Real.pi / 3 - (x + Real.pi / 2)) := by
  sorry

end cos_sin_shift_l1688_168881


namespace expression_simplification_l1688_168872

theorem expression_simplification 
  (p q : ℝ) (x : ℝ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hx_pos : x > 0) (hx_neq_one : x ≠ 1) :
  (x^(3/p) - x^(3/q)) / ((x^(1/p) + x^(1/q))^2 - 2*x^(1/q)*(x^(1/q) + x^(1/p))) + 
  x^(1/p) / (x^((q-p)/(p*q)) + 1) = x^(1/p) + x^(1/q) := by
  sorry

end expression_simplification_l1688_168872


namespace cafeteria_pies_l1688_168884

theorem cafeteria_pies (initial_apples handed_out apples_per_pie : ℕ) 
  (h1 : initial_apples = 86)
  (h2 : handed_out = 30)
  (h3 : apples_per_pie = 8) :
  (initial_apples - handed_out) / apples_per_pie = 7 :=
by sorry

end cafeteria_pies_l1688_168884


namespace abc_inequality_l1688_168813

theorem abc_inequality (a b c : ℝ) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (sum_eq : a + b + c = 6) 
  (prod_sum_eq : a * b + b * c + c * a = 9) : 
  0 < a * b * c ∧ a * b * c < 4 := by
sorry

end abc_inequality_l1688_168813


namespace no_rational_solution_l1688_168819

theorem no_rational_solution : ¬∃ (x y z : ℚ), (x + y + z = 0) ∧ (x^2 + y^2 + z^2 = 100) := by
  sorry

end no_rational_solution_l1688_168819


namespace angle_sum_theorem_l1688_168847

theorem angle_sum_theorem (x : ℝ) : 
  (6 * x + 7 * x + 3 * x + 4 * x) * (π / 180) = 2 * π → x = 18 := by
  sorry

end angle_sum_theorem_l1688_168847


namespace four_by_four_cube_unpainted_l1688_168830

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_units : Nat
  painted_per_face : Nat

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpainted_cubes (cube : PaintedCube) : Nat :=
  cube.total_units - (cube.size * cube.size * 6 - (cube.size - 2) * (cube.size - 2) * 6)

/-- Theorem stating that a 4x4x4 cube with 4 painted squares per face has 52 unpainted unit cubes -/
theorem four_by_four_cube_unpainted :
  let cube : PaintedCube := { size := 4, total_units := 64, painted_per_face := 4 }
  unpainted_cubes cube = 52 := by
  sorry

end four_by_four_cube_unpainted_l1688_168830


namespace cost_price_per_meter_l1688_168836

/-- The cost price of one meter of cloth given the selling price and profit per meter -/
theorem cost_price_per_meter 
  (total_meters : ℕ) 
  (selling_price : ℚ) 
  (profit_per_meter : ℚ) : 
  total_meters = 80 → 
  selling_price = 6900 → 
  profit_per_meter = 20 → 
  (selling_price - total_meters * profit_per_meter) / total_meters = 66.25 := by
sorry

end cost_price_per_meter_l1688_168836


namespace integer_fraction_l1688_168897

theorem integer_fraction (a : ℕ+) : 
  (↑(2 * a.val + 8) / ↑(a.val + 1) : ℚ).isInt ↔ a.val = 1 ∨ a.val = 2 ∨ a.val = 5 := by
  sorry

end integer_fraction_l1688_168897


namespace rabbit_can_escape_l1688_168817

/-- Represents a point in 2D space -/
structure Point where
  x : Real
  y : Real

/-- Represents a square with side length 1 -/
structure Square where
  center : Point
  side_length : Real := 1

/-- Represents an entity (rabbit or wolf) with a position and speed -/
structure Entity where
  position : Point
  speed : Real

/-- Theorem stating that the rabbit can escape the square -/
theorem rabbit_can_escape (s : Square) (rabbit : Entity) (wolves : Finset Entity) :
  rabbit.position = s.center →
  wolves.card = 4 →
  (∀ w ∈ wolves, w.speed = 1.4 * rabbit.speed) →
  (∀ w ∈ wolves, w.position.x = 0 ∨ w.position.x = 1) →
  (∀ w ∈ wolves, w.position.y = 0 ∨ w.position.y = 1) →
  ∃ (escape_path : Real → Point),
    (escape_path 0 = rabbit.position) ∧
    (∃ t : Real, t > 0 ∧ (escape_path t).x = 0 ∨ (escape_path t).x = 1 ∨ (escape_path t).y = 0 ∨ (escape_path t).y = 1) ∧
    (∀ w ∈ wolves, ∀ t : Real, t ≥ 0 → 
      (escape_path t).x ≠ w.position.x ∨ (escape_path t).y ≠ w.position.y) :=
sorry

end rabbit_can_escape_l1688_168817


namespace x_cube_between_x_and_x_square_l1688_168845

theorem x_cube_between_x_and_x_square :
  let x : ℚ := -2/5
  x < x^3 ∧ x^3 < x^2 := by sorry

end x_cube_between_x_and_x_square_l1688_168845


namespace martha_lasagna_cost_l1688_168886

/-- The cost of ingredients for Martha's lasagna -/
def lasagna_cost (cheese_price meat_price pasta_price tomato_price : ℝ) : ℝ :=
  1.5 * cheese_price + 0.55 * meat_price + 0.28 * pasta_price + 2.2 * tomato_price

/-- Theorem stating the total cost of ingredients for Martha's lasagna -/
theorem martha_lasagna_cost :
  lasagna_cost 6.30 8.55 2.40 1.79 = 18.76 := by
  sorry


end martha_lasagna_cost_l1688_168886


namespace christine_travel_time_l1688_168841

/-- Given Christine's travel scenario, prove the time she wandered. -/
theorem christine_travel_time (speed : ℝ) (distance : ℝ) (h1 : speed = 20) (h2 : distance = 80) :
  distance / speed = 4 := by
  sorry

end christine_travel_time_l1688_168841


namespace intersection_of_M_and_N_l1688_168802

-- Define the universal set U as the real numbers
def U := ℝ

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = 2^(|x|)}

-- Define set N
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.log (3 - x)}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = {t : ℝ | 1 ≤ t ∧ t < 3} := by sorry

end intersection_of_M_and_N_l1688_168802


namespace polynomial_factor_coefficients_l1688_168811

theorem polynomial_factor_coefficients :
  ∀ (a b : ℤ),
  (∃ (c d : ℤ),
    (∀ x : ℝ, a * x^4 + b * x^3 + 40 * x^2 - 20 * x + 8 = (3 * x^2 - 2 * x + 2) * (c * x^2 + d * x + 4))) →
  a = -51 ∧ b = 25 := by
sorry

end polynomial_factor_coefficients_l1688_168811


namespace exam_time_allocation_l1688_168857

/-- Represents the time spent on type A problems in an exam -/
def time_on_type_A (total_time minutes : ℕ) (total_questions type_A_questions : ℕ) : ℕ :=
  let type_B_questions := total_questions - type_A_questions
  let time_ratio := 2  -- Type A takes twice as long as Type B
  let total_time_units := type_A_questions * time_ratio + type_B_questions
  (total_time * minutes * type_A_questions * time_ratio) / total_time_units

/-- Theorem: Given the exam conditions, the time spent on type A problems is 120 minutes -/
theorem exam_time_allocation :
  time_on_type_A 3 60 200 100 = 120 := by
  sorry

end exam_time_allocation_l1688_168857


namespace kamal_physics_marks_l1688_168831

/-- Represents a student's marks in various subjects -/
structure StudentMarks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average marks for a student -/
def average (marks : StudentMarks) : ℚ :=
  (marks.english + marks.mathematics + marks.physics + marks.chemistry + marks.biology) / 5

theorem kamal_physics_marks :
  ∀ (marks : StudentMarks),
    marks.english = 76 →
    marks.mathematics = 60 →
    marks.chemistry = 67 →
    marks.biology = 85 →
    average marks = 74 →
    marks.physics = 82 :=
by
  sorry

end kamal_physics_marks_l1688_168831


namespace x_squared_ge_one_necessary_not_sufficient_l1688_168885

theorem x_squared_ge_one_necessary_not_sufficient :
  (∀ x : ℝ, x ≥ 1 → x^2 ≥ 1) ∧
  (∃ x : ℝ, x^2 ≥ 1 ∧ x < 1) :=
by sorry

end x_squared_ge_one_necessary_not_sufficient_l1688_168885


namespace fraction_inequality_l1688_168828

theorem fraction_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  1 / a + 4 / (1 - a) ≥ 9 := by
  sorry

end fraction_inequality_l1688_168828


namespace smallest_integer_satisfying_inequality_l1688_168873

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3*x - 7 → x ≥ 4 ∧ 4 < 3*4 - 7 := by sorry

end smallest_integer_satisfying_inequality_l1688_168873


namespace symmetric_line_equation_l1688_168824

/-- Given two lines in a 2D plane:
    1. y = 3x + 4
    2. y = x
    This theorem states that the line symmetric to y = 3x + 4
    with respect to y = x has the equation y = (1/3)x - (4/3) -/
theorem symmetric_line_equation :
  let line1 : ℝ → ℝ := λ x => 3 * x + 4
  let line2 : ℝ → ℝ := λ x => x
  let symmetric_line : ℝ → ℝ := λ x => (1/3) * x - (4/3)
  ∀ x y : ℝ,
    (y = line1 x ∧ 
     ∃ x' y', x' = y ∧ y' = x ∧ y' = line2 x') →
    y = symmetric_line x :=
by sorry

end symmetric_line_equation_l1688_168824


namespace smallest_dual_palindrome_l1688_168846

/-- Checks if a natural number is a palindrome in the given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ n : ℕ, n > 20 → 
  (isPalindrome n 2 ∧ isPalindrome n 4) → 
  n ≥ 21 :=
sorry

end smallest_dual_palindrome_l1688_168846


namespace statement_S_holds_for_options_l1688_168899

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def options : List ℕ := [90, 99, 108, 117]

theorem statement_S_holds_for_options : ∀ n ∈ options, 
  (sum_of_digits n) % 9 = 0 → n % 3 = 0 := by sorry

end statement_S_holds_for_options_l1688_168899


namespace remaining_shoes_l1688_168895

theorem remaining_shoes (large medium small sold : ℕ) 
  (h_large : large = 22)
  (h_medium : medium = 50)
  (h_small : small = 24)
  (h_sold : sold = 83) :
  large + medium + small - sold = 13 := by
  sorry

end remaining_shoes_l1688_168895


namespace tangent_circle_equation_l1688_168869

/-- A circle tangent to both coordinate axes with its center on the line 5x - 3y = 8 -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_axes : center.1 = radius ∧ center.2 = radius
  center_on_line : 5 * center.1 - 3 * center.2 = 8

/-- The equation of the circle is either (x-4)² + (y-4)² = 16 or (x-1)² + (y+1)² = 1 -/
theorem tangent_circle_equation (c : TangentCircle) :
  (∀ x y : ℝ, (x - 4)^2 + (y - 4)^2 = 16) ∨
  (∀ x y : ℝ, (x - 1)^2 + (y + 1)^2 = 1) :=
sorry

end tangent_circle_equation_l1688_168869


namespace max_intersections_circle_ellipse_triangle_l1688_168882

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- An ellipse in a 2D plane -/
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ -- semi-major axis
  b : ℝ -- semi-minor axis

/-- A triangle in a 2D plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- The maximum number of intersection points between a circle and a triangle -/
def max_intersections_circle_triangle : ℕ := 6

/-- The maximum number of intersection points between an ellipse and a triangle -/
def max_intersections_ellipse_triangle : ℕ := 6

/-- The maximum number of intersection points between a circle and an ellipse -/
def max_intersections_circle_ellipse : ℕ := 4

/-- Theorem: The maximum number of intersection points among a circle, an ellipse, and a triangle is 16 -/
theorem max_intersections_circle_ellipse_triangle :
  ∀ (c : Circle) (e : Ellipse) (t : Triangle),
  max_intersections_circle_triangle +
  max_intersections_ellipse_triangle +
  max_intersections_circle_ellipse = 16 :=
by
  sorry

end max_intersections_circle_ellipse_triangle_l1688_168882


namespace adjacent_negative_product_l1688_168894

def a (n : ℕ) : ℤ := 2 * n - 17

theorem adjacent_negative_product :
  ∀ n : ℕ, (a n * a (n + 1) < 0) ↔ n = 8 := by sorry

end adjacent_negative_product_l1688_168894


namespace additional_students_l1688_168816

theorem additional_students (initial_students : ℕ) (students_per_computer : ℕ) (target_computers : ℕ) : 
  initial_students = 82 →
  students_per_computer = 2 →
  target_computers = 49 →
  (initial_students + (target_computers - initial_students / students_per_computer) * students_per_computer) - initial_students = 16 := by
sorry

end additional_students_l1688_168816


namespace absolute_difference_of_roots_l1688_168856

theorem absolute_difference_of_roots (p q : ℝ) : 
  p^2 - 6*p + 8 = 0 → q^2 - 6*q + 8 = 0 → |p - q| = 2 := by
  sorry

end absolute_difference_of_roots_l1688_168856


namespace a_gt_b_iff_a_plus_ln_a_gt_b_plus_ln_b_l1688_168821

theorem a_gt_b_iff_a_plus_ln_a_gt_b_plus_ln_b 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a > b) ↔ (a + Real.log a > b + Real.log b) := by
  sorry

end a_gt_b_iff_a_plus_ln_a_gt_b_plus_ln_b_l1688_168821


namespace trigonometric_inequality_l1688_168861

theorem trigonometric_inequality (α β γ : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : 0 < γ ∧ γ < π/2)
  (h4 : Real.sin α ^ 3 + Real.sin β ^ 3 + Real.sin γ ^ 3 = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3 * Real.sqrt 3 / 2 := by
  sorry

end trigonometric_inequality_l1688_168861


namespace inequality_proof_l1688_168858

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_3 : a + b + c = 3) : 
  (a / (a + b)) + (b / (b + c)) + (c / (c + a)) ≤ 3 / (1 + Real.sqrt (a * b * c)) := by
  sorry

end inequality_proof_l1688_168858


namespace work_problem_solution_l1688_168860

def work_problem (a_days b_days b_worked_days : ℕ) : ℕ :=
  let b_work_rate := 1 / b_days
  let b_completed := b_work_rate * b_worked_days
  let remaining_work := 1 - b_completed
  let a_work_rate := 1 / a_days
  Nat.ceil (remaining_work / a_work_rate)

theorem work_problem_solution :
  work_problem 18 15 10 = 6 :=
by sorry

end work_problem_solution_l1688_168860


namespace min_value_of_sum_of_squares_l1688_168866

theorem min_value_of_sum_of_squares (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → 
  (∀ c d : ℝ, (∃ x : ℝ, x^4 + c*x^3 + d*x^2 + c*x + 1 = 0) → a^2 + b^2 ≤ c^2 + d^2) →
  a^2 + b^2 = 4/5 := by
sorry

end min_value_of_sum_of_squares_l1688_168866


namespace x_plus_y_equals_three_l1688_168883

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

variable (a b c p : V)
variable (x y : ℝ)

-- {a, b, c} is a basis of the space
variable (h1 : LinearIndependent ℝ ![a, b, c])
variable (h2 : Submodule.span ℝ {a, b, c} = ⊤)

-- p = 3a + b + c
variable (h3 : p = 3 • a + b + c)

-- {a+b, a-b, c} is another basis of the space
variable (h4 : LinearIndependent ℝ ![a + b, a - b, c])
variable (h5 : Submodule.span ℝ {a + b, a - b, c} = ⊤)

-- p = x(a+b) + y(a-b) + c
variable (h6 : p = x • (a + b) + y • (a - b) + c)

theorem x_plus_y_equals_three : x + y = 3 := by sorry

end x_plus_y_equals_three_l1688_168883


namespace projection_coordinates_l1688_168851

/-- The plane equation ax + by + cz + d = 0 -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The projection of a point onto a plane -/
def projection (p : Point3D) (plane : Plane) : Point3D :=
  sorry

theorem projection_coordinates :
  let p := Point3D.mk 1 2 (-1)
  let plane := Plane.mk 3 (-1) 2 (-4)
  let proj := projection p plane
  proj.x = 29 / 14 ∧ proj.y = 23 / 14 ∧ proj.z = -2 / 7 := by
  sorry

end projection_coordinates_l1688_168851


namespace polynomial_factorization_l1688_168820

theorem polynomial_factorization (k : ℝ) : 
  (∀ x : ℝ, x^2 - k*x - 6 = (x - 2)*(x + 3)) → k = -1 := by
  sorry

end polynomial_factorization_l1688_168820


namespace min_cyclic_fraction_sum_l1688_168855

theorem min_cyclic_fraction_sum (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b + b / c + c / d + d / a) ≥ 4 ∧ 
  ((a / b + b / c + c / d + d / a) = 4 ↔ a = b ∧ b = c ∧ c = d) := by
  sorry

end min_cyclic_fraction_sum_l1688_168855


namespace complex_division_equality_l1688_168809

theorem complex_division_equality : (3 + Complex.I) / (1 + Complex.I) = 2 - Complex.I := by
  sorry

end complex_division_equality_l1688_168809


namespace decimal_50_to_ternary_l1688_168810

/-- Converts a natural number to its ternary (base-3) representation -/
def to_ternary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- Checks if a list of digits is a valid ternary number -/
def is_valid_ternary (l : List ℕ) : Prop :=
  l.all (λ d => d < 3)

theorem decimal_50_to_ternary :
  let ternary := to_ternary 50
  is_valid_ternary ternary ∧ ternary = [1, 2, 1, 2] := by sorry

end decimal_50_to_ternary_l1688_168810


namespace encounters_for_2015_trips_relative_speeds_main_encounters_theorem_l1688_168870

/-- The number of encounters between two people traveling between two points -/
def encounters (a_trips b_trips : ℕ) : ℕ :=
  let full_cycles := a_trips / 2
  let remainder := a_trips % 2
  3 * full_cycles + if remainder = 0 then 0 else 2

/-- The theorem stating the number of encounters when A reaches point B 2015 times -/
theorem encounters_for_2015_trips : encounters 2015 2015 = 3023 := by
  sorry

/-- The theorem stating the relative speeds of A and B -/
theorem relative_speeds : ∀ (va vb : ℝ), 
  5 * va = 9 * vb → vb = (18/5) * va := by
  sorry

/-- The main theorem proving the number of encounters -/
theorem main_encounters_theorem : 
  ∃ (va vb : ℝ), va > 0 ∧ vb > 0 ∧ 5 * va = 9 * vb ∧ encounters 2015 2015 = 3023 := by
  sorry

end encounters_for_2015_trips_relative_speeds_main_encounters_theorem_l1688_168870


namespace expansion_equality_l1688_168812

theorem expansion_equality (x : ℝ) : 
  (x - 2)^5 + 5*(x - 2)^4 + 10*(x - 2)^3 + 10*(x - 2)^2 + 5*(x - 2) + 1 = (x - 1)^5 := by
sorry

end expansion_equality_l1688_168812


namespace smallest_unrepresentable_odd_number_l1688_168863

theorem smallest_unrepresentable_odd_number :
  ∀ n : ℕ, n > 0 → n % 2 = 1 →
    (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ n = 7^x - 3 * 2^y) ∨ n ≥ 3 :=
by sorry

end smallest_unrepresentable_odd_number_l1688_168863


namespace sin_double_angle_for_point_l1688_168842

theorem sin_double_angle_for_point (a : ℝ) (θ : ℝ) (h : a > 0) :
  let P : ℝ × ℝ := (-4 * a, 3 * a)
  (∃ r : ℝ, r > 0 ∧ P.1 = r * Real.cos θ ∧ P.2 = r * Real.sin θ) →
  Real.sin (2 * θ) = -24 / 25 := by
  sorry

end sin_double_angle_for_point_l1688_168842


namespace min_value_inequality_l1688_168896

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a^2 - 1) * (1 / b^2 - 1) ≥ 9 := by
  sorry

end min_value_inequality_l1688_168896


namespace tomatoes_for_sale_tuesday_l1688_168876

/-- Calculates the amount of tomatoes ready for sale on Tuesday given specific conditions --/
theorem tomatoes_for_sale_tuesday 
  (initial_shipment : ℝ)
  (saturday_selling_rate : ℝ)
  (sunday_spoilage_rate : ℝ)
  (monday_shipment_multiplier : ℝ)
  (monday_selling_rate : ℝ)
  (tuesday_spoilage_rate : ℝ)
  (h1 : initial_shipment = 1000)
  (h2 : saturday_selling_rate = 0.6)
  (h3 : sunday_spoilage_rate = 0.2)
  (h4 : monday_shipment_multiplier = 1.5)
  (h5 : monday_selling_rate = 0.4)
  (h6 : tuesday_spoilage_rate = 0.15) :
  ∃ (tomatoes_tuesday : ℝ), tomatoes_tuesday = 928.2 := by
  sorry

end tomatoes_for_sale_tuesday_l1688_168876


namespace intersection_range_l1688_168864

theorem intersection_range (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    x₁ > 0 ∧ x₂ > 0 ∧
    y₁ = k * x₁ + 2 ∧
    y₂ = k * x₂ + 2 ∧
    x₁^2 - y₁^2 = 6 ∧
    x₂^2 - y₂^2 = 6) →
  -Real.sqrt 15 / 3 < k ∧ k < -1 :=
by sorry

end intersection_range_l1688_168864


namespace rachel_envelope_stuffing_l1688_168823

/-- Rachel's envelope stuffing problem -/
theorem rachel_envelope_stuffing
  (total_hours : ℕ)
  (total_envelopes : ℕ)
  (second_hour_envelopes : ℕ)
  (required_rate : ℕ)
  (h1 : total_hours = 8)
  (h2 : total_envelopes = 1500)
  (h3 : second_hour_envelopes = 141)
  (h4 : required_rate = 204) :
  total_envelopes - (required_rate * (total_hours - 2)) - second_hour_envelopes = 135 := by
  sorry

#check rachel_envelope_stuffing

end rachel_envelope_stuffing_l1688_168823


namespace investment_problem_l1688_168879

/-- The investment problem --/
theorem investment_problem (vishal trishul raghu : ℝ) : 
  vishal = 1.1 * trishul →  -- Vishal invested 10% more than Trishul
  raghu = 2500 →  -- Raghu invested Rs. 2500
  vishal + trishul + raghu = 7225 →  -- Total sum of investments
  trishul < raghu →  -- Trishul invested less than Raghu
  (raghu - trishul) / raghu * 100 = 10 :=  -- Percentage difference
by sorry

end investment_problem_l1688_168879


namespace line_extraction_theorem_l1688_168889

-- Define a structure for a line in a plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a structure for a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if two lines are parallel
def areParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

-- Function to check if a line intersects all given lines
def intersectsAllLines (l : Line) (l1 l2 l3 l4 : Line) : Prop :=
  ∃ p1 p2 p3 p4 : Point,
    pointOnLine p1 l ∧ pointOnLine p1 l1 ∧
    pointOnLine p2 l ∧ pointOnLine p2 l2 ∧
    pointOnLine p3 l ∧ pointOnLine p3 l3 ∧
    pointOnLine p4 l ∧ pointOnLine p4 l4

-- Function to check if segments are in given ratios
def segmentsInRatio (l : Line) (l1 l2 l3 : Line) (r1 r2 : ℝ) : Prop :=
  ∃ p1 p2 p3 : Point,
    pointOnLine p1 l ∧ pointOnLine p1 l1 ∧
    pointOnLine p2 l ∧ pointOnLine p2 l2 ∧
    pointOnLine p3 l ∧ pointOnLine p3 l3 ∧
    (p2.x - p1.x)^2 + (p2.y - p1.y)^2 = r1 * ((p3.x - p2.x)^2 + (p3.y - p2.y)^2) ∧
    (p3.x - p2.x)^2 + (p3.y - p2.y)^2 = r2 * ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Theorem statement
theorem line_extraction_theorem (l1 l2 l3 l4 : Line) (r1 r2 : ℝ) :
  (∃ l : Line, intersectsAllLines l l1 l2 l3 l4 ∧ segmentsInRatio l l1 l2 l3 r1 r2) ∨
  (∃ m : Line, segmentsInRatio m l1 l2 l3 r1 r2 ∧ (areParallel m l4 ∨ m = l4)) :=
sorry

end line_extraction_theorem_l1688_168889


namespace lisa_tommy_earnings_difference_l1688_168833

-- Define the earnings for each person
def sophia_earnings : ℕ := 10 + 15 + 25
def sarah_earnings : ℕ := 15 + 10 + 20 + 20
def lisa_earnings : ℕ := 20 + 30
def jack_earnings : ℕ := 10 + 10 + 10 + 15 + 15
def tommy_earnings : ℕ := 5 + 5 + 10 + 10

-- Define the total earnings
def total_earnings : ℕ := 180

-- Theorem statement
theorem lisa_tommy_earnings_difference :
  lisa_earnings - tommy_earnings = 20 :=
sorry

end lisa_tommy_earnings_difference_l1688_168833


namespace vacation_cost_division_l1688_168827

theorem vacation_cost_division (total_cost : ℕ) (cost_difference : ℕ) : 
  (total_cost = 360) →
  (total_cost / 4 + cost_difference = total_cost / 3) →
  (cost_difference = 30) →
  3 = total_cost / (total_cost / 3) :=
by
  sorry

end vacation_cost_division_l1688_168827


namespace quadratic_properties_l1688_168854

def f (x : ℝ) := -2 * x^2 + 4 * x + 1

theorem quadratic_properties :
  (∃ (a : ℝ), ∀ (x : ℝ), f x = f (2 - x)) ∧
  (f 1 = 3 ∧ ∀ (x : ℝ), f x ≤ f 1) := by
  sorry

end quadratic_properties_l1688_168854


namespace smallest_qnnn_l1688_168891

def is_two_digit_with_equal_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 11 = 0

def is_one_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

def satisfies_condition (nn : ℕ) (n : ℕ) (qnnn : ℕ) : Prop :=
  is_two_digit_with_equal_digits nn ∧
  is_one_digit n ∧
  nn * n = qnnn ∧
  1000 ≤ qnnn ∧ qnnn ≤ 9999 ∧
  qnnn % 1000 % 100 % 10 = n ∧
  qnnn % 1000 % 100 / 10 = n ∧
  qnnn % 1000 / 100 = n

theorem smallest_qnnn :
  ∀ qnnn : ℕ, (∃ nn n : ℕ, satisfies_condition nn n qnnn) →
  2555 ≤ qnnn :=
sorry

end smallest_qnnn_l1688_168891


namespace quadruple_work_time_l1688_168803

-- Define the work rates for A and B
def work_rate_A : ℚ := 1 / 45
def work_rate_B : ℚ := 1 / 30

-- Define the combined work rate
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Define the time to complete 4 times the work
def time_for_quadruple_work : ℚ := 4 / combined_work_rate

-- Theorem statement
theorem quadruple_work_time : time_for_quadruple_work = 9/2 := by sorry

end quadruple_work_time_l1688_168803


namespace book_pages_theorem_l1688_168852

/-- A book with two chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ

/-- The total number of pages in a book -/
def total_pages (b : Book) : ℕ := b.chapter1_pages + b.chapter2_pages

/-- Theorem: A book with 48 pages in the first chapter and 46 pages in the second chapter has 94 pages in total -/
theorem book_pages_theorem :
  ∀ (b : Book), b.chapter1_pages = 48 → b.chapter2_pages = 46 → total_pages b = 94 := by
  sorry

end book_pages_theorem_l1688_168852


namespace allocation_schemes_13_4_l1688_168887

/-- The number of ways to allocate outstanding member quotas to classes. -/
def allocationSchemes (totalMembers : ℕ) (numClasses : ℕ) : ℕ :=
  Nat.choose (totalMembers - numClasses + numClasses - 1) (numClasses - 1)

/-- Theorem stating the number of allocation schemes for 13 members to 4 classes. -/
theorem allocation_schemes_13_4 :
  allocationSchemes 13 4 = 220 := by
  sorry

#eval allocationSchemes 13 4

end allocation_schemes_13_4_l1688_168887


namespace cubic_decreasing_iff_a_leq_neg_three_l1688_168874

/-- A cubic function f(x) = a x^3 + 3 x^2 - x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 1

/-- A function is decreasing on ℝ if for all x, y in ℝ, x < y implies f(x) > f(y) -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

theorem cubic_decreasing_iff_a_leq_neg_three (a : ℝ) :
  IsDecreasing (f a) ↔ a ≤ -3 := by sorry

end cubic_decreasing_iff_a_leq_neg_three_l1688_168874


namespace transformations_correct_l1688_168880

theorem transformations_correct (a b c : ℝ) (h1 : a = b) (h2 : c ≠ 0) (h3 : a / c = b / c) (h4 : -2 * a = -2 * b) : 
  (a + 6 = b + 6) ∧ 
  (a / 9 = b / 9) ∧ 
  (a = b) ∧ 
  (a = b) := by
  sorry

end transformations_correct_l1688_168880


namespace complex_equation_solution_l1688_168834

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = 3 + Complex.I → z = 1 + 2 * Complex.I := by
  sorry

end complex_equation_solution_l1688_168834


namespace octal_sum_equals_2351_l1688_168801

/-- Converts a base-8 number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a decimal number to its base-8 representation as a list of digits -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

/-- Theorem stating that the sum of 1457₈ and 672₈ in base 8 is 2351₈ -/
theorem octal_sum_equals_2351 :
  let a := octal_to_decimal [7, 5, 4, 1]  -- 1457₈
  let b := octal_to_decimal [2, 7, 6]     -- 672₈
  decimal_to_octal (a + b) = [1, 5, 3, 2] -- 2351₈
  := by sorry

end octal_sum_equals_2351_l1688_168801
