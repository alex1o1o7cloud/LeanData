import Mathlib

namespace ratio_equality_l187_18744

theorem ratio_equality (x y : ℝ) (h : 1.5 * x = 0.04 * y) :
  (y - x) / (y + x) = 73 / 77 := by
  sorry

end ratio_equality_l187_18744


namespace monotonic_f_implies_a_range_l187_18768

def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

theorem monotonic_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y ∨ ∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end monotonic_f_implies_a_range_l187_18768


namespace sum_of_four_numbers_l187_18701

theorem sum_of_four_numbers : 1357 + 7531 + 3175 + 5713 = 17776 := by
  sorry

end sum_of_four_numbers_l187_18701


namespace second_number_is_22_l187_18764

theorem second_number_is_22 (x y : ℝ) (h1 : x + y = 33) (h2 : y = 2 * x) : y = 22 := by
  sorry

end second_number_is_22_l187_18764


namespace stratified_sample_size_l187_18748

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  totalPopulation : ℕ
  stratumPopulation : ℕ
  stratumSample : ℕ
  totalSample : ℕ

/-- The stratified sampling is proportional if the ratio of the stratum in the population
    equals the ratio of the stratum in the sample -/
def isProportional (s : StratifiedSample) : Prop :=
  s.stratumPopulation * s.totalSample = s.totalPopulation * s.stratumSample

theorem stratified_sample_size 
  (s : StratifiedSample) 
  (h1 : s.totalPopulation = 12000)
  (h2 : s.stratumPopulation = 3600)
  (h3 : s.stratumSample = 60)
  (h4 : isProportional s) :
  s.totalSample = 200 := by
  sorry

end stratified_sample_size_l187_18748


namespace theatre_seating_l187_18707

theorem theatre_seating (total_seats : ℕ) (row_size : ℕ) (expected_attendance : ℕ) : 
  total_seats = 225 → 
  row_size = 15 → 
  expected_attendance = 160 → 
  (total_seats - (((expected_attendance + row_size - 1) / row_size) * row_size)) = 60 :=
by sorry

end theatre_seating_l187_18707


namespace four_letter_word_count_l187_18751

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The length of the word -/
def word_length : ℕ := 4

theorem four_letter_word_count : 
  alphabet_size * vowel_count * alphabet_size = 3380 := by
  sorry

#check four_letter_word_count

end four_letter_word_count_l187_18751


namespace task_completion_time_l187_18789

/-- Given that m men can complete a task in d days, 
    prove that m + r² men will complete the same task in md / (m + r²) days -/
theorem task_completion_time 
  (m d r : ℕ) -- m, d, and r are natural numbers
  (m_pos : 0 < m) -- m is positive
  (d_pos : 0 < d) -- d is positive
  (total_work : ℕ := m * d) -- total work in man-days
  : (↑total_work : ℚ) / (m + r^2 : ℚ) = (↑m * ↑d : ℚ) / (↑m + ↑r^2 : ℚ) := by
  sorry


end task_completion_time_l187_18789


namespace multiplicative_inverse_modulo_million_l187_18780

theorem multiplicative_inverse_modulo_million : ∃ N : ℕ, 
  N > 0 ∧ 
  N < 1000000 ∧ 
  (N * ((222222 : ℕ) * 476190)) % 1000000 = 1 := by
  sorry

end multiplicative_inverse_modulo_million_l187_18780


namespace same_color_marble_probability_l187_18726

/-- The probability of drawing four marbles of the same color from a box containing
    3 orange, 7 purple, and 5 green marbles, without replacement. -/
theorem same_color_marble_probability :
  let total_marbles : ℕ := 3 + 7 + 5
  let orange_marbles : ℕ := 3
  let purple_marbles : ℕ := 7
  let green_marbles : ℕ := 5
  let draw_count : ℕ := 4
  
  (orange_marbles.choose draw_count +
   purple_marbles.choose draw_count +
   green_marbles.choose draw_count : ℚ) /
  (total_marbles.choose draw_count : ℚ) = 210 / 1369 :=
by sorry

end same_color_marble_probability_l187_18726


namespace box_volume_l187_18756

theorem box_volume (x : ℕ+) :
  (5 * x) * (5 * (x + 1)) * (5 * (x + 2)) = 25 * x^3 + 50 * x^2 + 125 * x :=
by sorry

end box_volume_l187_18756


namespace binomial_divisibility_sequence_l187_18762

theorem binomial_divisibility_sequence :
  ∃ n : ℕ, n > 2003 ∧ ∀ i j : ℕ, 0 ≤ i → i < j → j ≤ 2003 → (n.choose i ∣ n.choose j) := by
  sorry

end binomial_divisibility_sequence_l187_18762


namespace john_needs_29_planks_l187_18785

/-- The number of large planks John uses for the house wall. -/
def large_planks : ℕ := 12

/-- The number of small planks John uses for the house wall. -/
def small_planks : ℕ := 17

/-- The total number of planks John needs for the house wall. -/
def total_planks : ℕ := large_planks + small_planks

/-- Theorem stating that the total number of planks John needs is 29. -/
theorem john_needs_29_planks : total_planks = 29 := by
  sorry

end john_needs_29_planks_l187_18785


namespace perpendicular_line_equation_l187_18736

/-- Given a point and a line, this theorem proves that the equation
    x - 2y + 7 = 0 represents a line passing through the given point
    and perpendicular to the given line. -/
theorem perpendicular_line_equation (x y : ℝ) :
  let point : ℝ × ℝ := (-1, 3)
  let given_line := {(x, y) : ℝ × ℝ | 2 * x + y + 3 = 0}
  let perpendicular_line := {(x, y) : ℝ × ℝ | x - 2 * y + 7 = 0}
  (point ∈ perpendicular_line) ∧
  (∀ (v w : ℝ × ℝ), v ∈ given_line → w ∈ given_line → v ≠ w →
    let slope_given := (w.2 - v.2) / (w.1 - v.1)
    let slope_perp := (y - 3) / (x - (-1))
    slope_given * slope_perp = -1) :=
by sorry

end perpendicular_line_equation_l187_18736


namespace actual_time_greater_than_planned_l187_18753

/-- Proves that the actual running time is greater than the planned time under given conditions -/
theorem actual_time_greater_than_planned (a V : ℝ) (h1 : a > 0) (h2 : V > 0) : 
  (a / (1.25 * V) / 2 + a / (0.8 * V) / 2) > a / V := by
  sorry

#check actual_time_greater_than_planned

end actual_time_greater_than_planned_l187_18753


namespace sqrt_equation_solution_l187_18774

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (4 * x + 9) = 12 → x = 33.75 := by
  sorry

end sqrt_equation_solution_l187_18774


namespace circle_angle_sum_l187_18729

/-- Given a circle with points X, Y, and Z, where arc XY = 50°, arc YZ = 45°, arc ZX = 90°,
    angle α = (arc XZ - arc YZ) / 2, and angle β = arc YZ / 2,
    prove that the sum of angles α and β equals 47.5°. -/
theorem circle_angle_sum (arcXY arcYZ arcZX : Real) (α β : Real) :
  arcXY = 50 ∧ arcYZ = 45 ∧ arcZX = 90 ∧
  α = (arcXY + arcYZ - arcYZ) / 2 ∧
  β = arcYZ / 2 →
  α + β = 47.5 := by
sorry


end circle_angle_sum_l187_18729


namespace negation_of_implication_l187_18795

-- Define a triangle type
structure Triangle where
  -- Add any necessary fields here
  mk :: -- Constructor

-- Define properties for triangles
def isEquilateral (t : Triangle) : Prop := sorry
def interiorAnglesEqual (t : Triangle) : Prop := sorry

-- State the theorem
theorem negation_of_implication :
  (¬(∀ t : Triangle, isEquilateral t → interiorAnglesEqual t)) ↔
  (∀ t : Triangle, ¬isEquilateral t → ¬interiorAnglesEqual t) :=
sorry

end negation_of_implication_l187_18795


namespace maximize_quadrilateral_area_l187_18740

/-- Given a rectangle ABCD with length 2 and width 1, and points E on AB and F on AD
    such that AE = 2AF, the area of quadrilateral CDFE is maximized when AF = 3/4,
    and the maximum area is 7/8 square units. -/
theorem maximize_quadrilateral_area (A B C D E F : ℝ × ℝ) :
  let rectangle_length : ℝ := 2
  let rectangle_width : ℝ := 1
  let ABCD_is_rectangle := 
    (A.1 = B.1 - rectangle_length) ∧ 
    (A.2 = D.2) ∧ 
    (B.2 = C.2) ∧ 
    (C.1 = D.1 + rectangle_length) ∧ 
    (A.2 = B.2 + rectangle_width)
  let E_on_AB := E.2 = A.2
  let F_on_AD := F.1 = A.1
  let AE_equals_2AF := E.1 - A.1 = 2 * (F.2 - A.2)
  let area_CDFE (x : ℝ) := 2 * x^2 - 3 * x + 2
  ABCD_is_rectangle → E_on_AB → F_on_AD → AE_equals_2AF →
    (∃ (x : ℝ), x = 3/4 ∧ 
      (∀ (y : ℝ), 0 ≤ y ∧ y ≤ 1 → area_CDFE y ≤ area_CDFE x) ∧
      area_CDFE x = 7/8) := by
  sorry

end maximize_quadrilateral_area_l187_18740


namespace twelve_solutions_for_quadratic_diophantine_l187_18745

theorem twelve_solutions_for_quadratic_diophantine (n : ℕ) (x y : ℕ+) 
  (h1 : x^2 - x*y + y^2 = n)
  (h2 : x ≠ y)
  (h3 : x ≠ 2*y)
  (h4 : y ≠ 2*x) :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), p ∈ S → p.1^2 - p.1*p.2 + p.2^2 = n) ∧ S.card ≥ 12 :=
sorry

end twelve_solutions_for_quadratic_diophantine_l187_18745


namespace container_capacity_l187_18742

theorem container_capacity (C : ℝ) : 0.40 * C + 14 = 0.75 * C → C = 40 := by
  sorry

end container_capacity_l187_18742


namespace quadratic_one_solution_l187_18797

theorem quadratic_one_solution (m : ℚ) : 
  (∃! x : ℚ, 3 * x^2 - 7 * x + m = 0) → m = 49/12 := by
  sorry

end quadratic_one_solution_l187_18797


namespace number_problem_l187_18779

theorem number_problem : ∃ x : ℝ, 0.65 * x - 25 = 90 ∧ abs (x - 176.92) < 0.01 := by
  sorry

end number_problem_l187_18779


namespace dark_tile_fraction_is_one_third_l187_18763

-- Define the properties of the tiled floor
structure TiledFloor :=
  (section_width : ℕ)
  (section_height : ℕ)
  (dark_tiles_per_section : ℕ)

-- Define the fraction of dark tiles
def dark_tile_fraction (floor : TiledFloor) : ℚ :=
  floor.dark_tiles_per_section / (floor.section_width * floor.section_height)

-- Theorem statement
theorem dark_tile_fraction_is_one_third 
  (floor : TiledFloor) 
  (h1 : floor.section_width = 6) 
  (h2 : floor.section_height = 4) 
  (h3 : floor.dark_tiles_per_section = 8) : 
  dark_tile_fraction floor = 1/3 := by
  sorry

end dark_tile_fraction_is_one_third_l187_18763


namespace interest_rate_calculation_l187_18755

theorem interest_rate_calculation (principal : ℝ) (time : ℝ) (lending_rate : ℝ) (gain_per_year : ℝ) 
  (h1 : principal = 20000)
  (h2 : time = 6)
  (h3 : lending_rate = 0.09)
  (h4 : gain_per_year = 200) :
  let interest_received := principal * lending_rate * time
  let total_gain := gain_per_year * time
  let interest_paid := interest_received - total_gain
  let borrowing_rate := interest_paid / (principal * time)
  borrowing_rate = 0.08 := by
sorry

end interest_rate_calculation_l187_18755


namespace shortest_altitude_right_triangle_l187_18750

theorem shortest_altitude_right_triangle :
  ∀ (a b c h : ℝ),
    a = 9 ∧ b = 12 ∧ c = 15 →
    a^2 + b^2 = c^2 →
    (1/2) * a * b = (1/2) * c * h →
    h = 7.2 :=
by
  sorry

end shortest_altitude_right_triangle_l187_18750


namespace inscribed_squares_area_l187_18713

/-- Given three squares inscribed in right triangles with areas A, M, and N,
    where M = 5 and N = 12, prove that A = 17 + 4√15 -/
theorem inscribed_squares_area (A M N : ℝ) (hM : M = 5) (hN : N = 12) :
  A = (Real.sqrt M + Real.sqrt N) ^ 2 →
  A = 17 + 4 * Real.sqrt 15 := by
sorry

end inscribed_squares_area_l187_18713


namespace product_sum_difference_theorem_l187_18718

theorem product_sum_difference_theorem (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x * y = 2688) 
  (h4 : x = 84) : 
  (x + y) - (x - y) = 64 := by
  sorry

end product_sum_difference_theorem_l187_18718


namespace multiplication_subtraction_equality_l187_18702

theorem multiplication_subtraction_equality : 154 * 1836 - 54 * 1836 = 183600 := by
  sorry

end multiplication_subtraction_equality_l187_18702


namespace circle_and_line_tangency_l187_18732

-- Define the line l
def line (x y a : ℝ) : Prop := Real.sqrt 3 * x - y - a = 0

-- Define the circle C in polar form
def circle_polar (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

-- Define the circle C in Cartesian form
def circle_cartesian (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Theorem statement
theorem circle_and_line_tangency :
  -- Part I: Equivalence of polar and Cartesian forms of circle C
  (∀ x y ρ θ : ℝ, circle_polar ρ θ ↔ circle_cartesian x y) ∧
  -- Part II: Tangency condition
  (∀ a : ℝ, (∃ x y : ℝ, line x y a ∧ circle_cartesian x y ∧
    (∀ x' y' : ℝ, line x' y' a ∧ circle_cartesian x' y' → x = x' ∧ y = y'))
    ↔ (a = -3 ∨ a = 1)) := by
  sorry

end circle_and_line_tangency_l187_18732


namespace power_division_result_l187_18712

theorem power_division_result : 8^15 / 64^7 = 8 := by sorry

end power_division_result_l187_18712


namespace minimum_m_value_l187_18741

noncomputable def f (x : ℝ) : ℝ := Real.log x + (2*x + 1) / x

theorem minimum_m_value (m : ℤ) :
  (∃ x : ℝ, x > 1 ∧ f x < (m * (x - 1) + 2) / x) →
  m ≥ 5 :=
by sorry

end minimum_m_value_l187_18741


namespace prob_one_white_correct_prob_red_given_red_correct_l187_18777

-- Define the number of red and white balls
def red_balls : ℕ := 4
def white_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := red_balls + white_balls

-- Define the number of balls drawn
def balls_drawn : ℕ := 3

-- Define the probability of drawing exactly one white ball
def prob_one_white : ℚ := 3/5

-- Define the probability of drawing a red ball on the second draw given a red ball was drawn on the first draw
def prob_red_given_red : ℚ := 3/5

-- Theorem 1: Probability of drawing exactly one white ball
theorem prob_one_white_correct :
  (Nat.choose white_balls 1 * Nat.choose red_balls (balls_drawn - 1)) / 
  Nat.choose total_balls balls_drawn = prob_one_white := by sorry

-- Theorem 2: Probability of drawing a red ball on the second draw given a red ball was drawn on the first draw
theorem prob_red_given_red_correct :
  (red_balls - 1) / (total_balls - 1) = prob_red_given_red := by sorry

end prob_one_white_correct_prob_red_given_red_correct_l187_18777


namespace m_range_for_inequality_l187_18798

theorem m_range_for_inequality (m : ℝ) : 
  (∀ x : ℝ, x ≤ -1 → (m - m^2) * 4^x + 2^x + 1 > 0) → 
  -2 < m ∧ m < 3 := by
sorry

end m_range_for_inequality_l187_18798


namespace A_intersect_B_l187_18775

def A : Set ℝ := {-2, 0, 1, 2}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem A_intersect_B : A ∩ B = {-2, 0, 1} := by sorry

end A_intersect_B_l187_18775


namespace running_time_ratio_l187_18767

theorem running_time_ratio (danny_time steve_time : ℝ) 
  (h1 : danny_time = 25)
  (h2 : steve_time / 2 + 12.5 = danny_time) : 
  danny_time / steve_time = 1 / 2 := by
sorry

end running_time_ratio_l187_18767


namespace circle_area_ratio_l187_18791

-- Define the circles and angles
def circle_small : Real → Real → Real := sorry
def circle_large : Real → Real → Real := sorry
def circle_sum : Real → Real → Real := sorry

def angle_small : Real := 60
def angle_large : Real := 48
def angle_sum : Real := 108

-- Define the radii
def radius_small : Real := sorry
def radius_large : Real := sorry
def radius_sum : Real := radius_small + radius_large

-- Define arc lengths
def arc_length (circle : Real → Real → Real) (angle : Real) : Real := sorry

-- State the theorem
theorem circle_area_ratio :
  let arc_small := arc_length circle_small angle_small
  let arc_large := arc_length circle_large angle_large
  let arc_sum := arc_length circle_sum angle_sum
  arc_small = arc_large ∧
  arc_sum = arc_small + arc_large →
  (circle_small radius_small 0) / (circle_large radius_large 0) = 16 / 25 := by
  sorry

end circle_area_ratio_l187_18791


namespace ellipse_and_intersection_l187_18716

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Define the line
def line (x y : ℝ) : Prop := Real.sqrt 3 * y - 2 * x - 2 = 0

theorem ellipse_and_intersection :
  -- Given conditions
  (ellipse_C 0 2) ∧ 
  (ellipse_C (1/2) (Real.sqrt 3)) ∧
  -- Prove the following
  (∀ x y : ℝ, ellipse_C x y ↔ x^2 + y^2/4 = 1) ∧ 
  (ellipse_C (-1) 0 ∧ line (-1) 0) ∧
  (ellipse_C (1/2) (Real.sqrt 3) ∧ line (1/2) (Real.sqrt 3)) :=
by sorry

end ellipse_and_intersection_l187_18716


namespace square_less_than_triple_l187_18781

theorem square_less_than_triple (n : ℤ) : n^2 < 3*n ↔ n = 1 ∨ n = 2 := by
  sorry

end square_less_than_triple_l187_18781


namespace prob_no_consecutive_heads_10_l187_18776

/-- Number of valid sequences without two consecutive heads for n coin tosses -/
def f (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | n + 2 => f (n + 1) + f n

/-- Probability of no two consecutive heads in n coin tosses -/
def prob_no_consecutive_heads (n : ℕ) : ℚ :=
  (f n : ℚ) / (2^n : ℚ)

/-- Theorem: The probability of no two heads appearing consecutively in 10 coin tosses is 9/64 -/
theorem prob_no_consecutive_heads_10 : prob_no_consecutive_heads 10 = 9/64 := by
  sorry

#eval prob_no_consecutive_heads 10

end prob_no_consecutive_heads_10_l187_18776


namespace system_equation_solution_l187_18746

theorem system_equation_solution (x y : ℝ) 
  (eq1 : 7 * x + y = 19) 
  (eq2 : x + 3 * y = 1) : 
  2 * x + y = 5 := by
  sorry

end system_equation_solution_l187_18746


namespace lines_perpendicular_to_plane_are_parallel_planes_perpendicular_to_line_are_parallel_l187_18735

-- Define the necessary structures
structure Line3D where
  -- Add necessary fields

structure Plane3D where
  -- Add necessary fields

-- Define perpendicularity and parallelism
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

def perpendicular_plane_line (p : Plane3D) (l : Line3D) : Prop :=
  sorry

def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

-- Theorem 1: Two lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_plane_are_parallel
  (l1 l2 : Line3D) (p : Plane3D)
  (h1 : perpendicular_line_plane l1 p)
  (h2 : perpendicular_line_plane l2 p) :
  parallel_lines l1 l2 :=
sorry

-- Theorem 2: Two planes perpendicular to the same line are parallel
theorem planes_perpendicular_to_line_are_parallel
  (p1 p2 : Plane3D) (l : Line3D)
  (h1 : perpendicular_plane_line p1 l)
  (h2 : perpendicular_plane_line p2 l) :
  parallel_planes p1 p2 :=
sorry

end lines_perpendicular_to_plane_are_parallel_planes_perpendicular_to_line_are_parallel_l187_18735


namespace renatas_final_balance_l187_18714

/-- Represents the balance and transactions of Renata's day --/
def renatas_day (initial_amount : ℚ) (charity_donation : ℚ) (prize_pounds : ℚ) 
  (slot_loss_euros : ℚ) (slot_loss_pounds : ℚ) (slot_loss_dollars : ℚ)
  (sunglasses_euros : ℚ) (water_pounds : ℚ) (lottery_ticket : ℚ) (lottery_prize : ℚ)
  (meal_euros : ℚ) (coffee_euros : ℚ) : ℚ :=
  let pound_to_dollar : ℚ := 1.35
  let euro_to_dollar : ℚ := 1.10
  let sunglasses_discount : ℚ := 0.20
  let meal_discount : ℚ := 0.30
  
  let balance1 := initial_amount - charity_donation
  let balance2 := balance1 + prize_pounds * pound_to_dollar
  let balance3 := balance2 - slot_loss_euros * euro_to_dollar
  let balance4 := balance3 - slot_loss_pounds * pound_to_dollar
  let balance5 := balance4 - slot_loss_dollars
  let balance6 := balance5 - sunglasses_euros * (1 - sunglasses_discount) * euro_to_dollar
  let balance7 := balance6 - water_pounds * pound_to_dollar
  let balance8 := balance7 - lottery_ticket
  let balance9 := balance8 + lottery_prize
  let lunch_cost := (meal_euros * (1 - meal_discount) + coffee_euros) * euro_to_dollar
  balance9 - lunch_cost / 2

/-- Theorem stating that Renata's final balance is $35.95 --/
theorem renatas_final_balance :
  renatas_day 50 10 50 30 20 15 15 1 1 30 10 3 = 35.95 := by sorry

end renatas_final_balance_l187_18714


namespace retiree_benefit_theorem_l187_18765

/-- Represents a bank customer --/
structure Customer where
  repayment_rate : ℝ
  monthly_income_stability : ℝ
  preferred_deposit_term : ℝ

/-- Represents a bank's financial metrics --/
structure BankMetrics where
  loan_default_risk : ℝ
  deposit_stability : ℝ
  long_term_liquidity : ℝ

/-- Calculates the benefit for a bank based on customer characteristics --/
def calculate_bank_benefit (c : Customer) : ℝ :=
  c.repayment_rate + c.monthly_income_stability + c.preferred_deposit_term

/-- Represents a retiree customer --/
def retiree : Customer where
  repayment_rate := 0.95
  monthly_income_stability := 0.9
  preferred_deposit_term := 5

/-- Represents an average customer --/
def average_customer : Customer where
  repayment_rate := 0.8
  monthly_income_stability := 0.7
  preferred_deposit_term := 2

/-- Theorem stating that offering special rates to retirees is beneficial for banks --/
theorem retiree_benefit_theorem :
  calculate_bank_benefit retiree > calculate_bank_benefit average_customer :=
by sorry

end retiree_benefit_theorem_l187_18765


namespace series_term_equals_original_term_l187_18788

/-- The n-th term of the series -4+7-4+7-4+7-... -/
def seriesTerm (n : ℕ) : ℝ :=
  1.5 + 5.5 * (-1)^n

/-- The original series terms -/
def originalTerm (n : ℕ) : ℝ :=
  if n % 2 = 1 then -4 else 7

theorem series_term_equals_original_term (n : ℕ) :
  seriesTerm n = originalTerm n := by
  sorry

#check series_term_equals_original_term

end series_term_equals_original_term_l187_18788


namespace sphere_volume_circumscribing_rectangular_solid_l187_18794

/-- The volume of a sphere circumscribing a rectangular solid with dimensions 1, 2, and 3 -/
theorem sphere_volume_circumscribing_rectangular_solid :
  let l : Real := 1  -- length
  let w : Real := 2  -- width
  let h : Real := 3  -- height
  let diagonal := Real.sqrt (l^2 + w^2 + h^2)
  let radius := diagonal / 2
  let volume := (4/3) * Real.pi * radius^3
  volume = (7 * Real.sqrt 14 / 3) * Real.pi := by
sorry


end sphere_volume_circumscribing_rectangular_solid_l187_18794


namespace train_speed_theorem_l187_18720

/-- Theorem: Given two trains moving in opposite directions, with one train's speed being 100 kmph,
    lengths of 500 m and 700 m, and a crossing time of 19.6347928529354 seconds,
    the speed of the faster train is 100 kmph. -/
theorem train_speed_theorem (v_slow v_fast : ℝ) (length_slow length_fast : ℝ) (crossing_time : ℝ) :
  v_fast = 100 ∧
  length_slow = 500 ∧
  length_fast = 700 ∧
  crossing_time = 19.6347928529354 ∧
  (length_slow + length_fast) / 1000 / (crossing_time / 3600) = v_slow + v_fast →
  v_fast = 100 := by
  sorry

#check train_speed_theorem

end train_speed_theorem_l187_18720


namespace stock_price_change_l187_18708

def down_limit : ℝ := 0.9
def up_limit : ℝ := 1.1
def num_limits : ℕ := 3

theorem stock_price_change (initial_price : ℝ) (initial_price_pos : initial_price > 0) :
  initial_price * (down_limit ^ num_limits) * (up_limit ^ num_limits) < initial_price :=
by sorry

end stock_price_change_l187_18708


namespace modulus_of_z_l187_18722

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for z
def z_condition (z : ℂ) : Prop := z * (1 + i) = 2

-- State the theorem
theorem modulus_of_z (z : ℂ) (h : z_condition z) : Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_z_l187_18722


namespace women_meeting_point_l187_18733

/-- Represents the distance walked by the woman starting from point B -/
def distance_B (h : ℕ) : ℚ :=
  h * (h + 3) / 2

/-- Represents the total distance walked by both women -/
def total_distance (h : ℕ) : ℚ :=
  3 * h + distance_B h

theorem women_meeting_point :
  ∃ (h : ℕ), h > 0 ∧ total_distance h = 60 ∧ distance_B h - 3 * h = 6 := by
  sorry

end women_meeting_point_l187_18733


namespace sum_product_plus_one_positive_l187_18737

theorem sum_product_plus_one_positive (a b c : ℝ) 
  (ha : abs a < 1) (hb : abs b < 1) (hc : abs c < 1) : 
  a * b + b * c + c * a + 1 > 0 := by
  sorry

end sum_product_plus_one_positive_l187_18737


namespace map_scale_l187_18761

theorem map_scale (map_length : ℝ) (real_distance : ℝ) (query_length : ℝ) :
  map_length > 0 →
  real_distance > 0 →
  query_length > 0 →
  (15 : ℝ) * real_distance = 45 * map_length →
  25 * real_distance = 75 * map_length := by
  sorry

end map_scale_l187_18761


namespace focus_directrix_distance_l187_18772

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Theorem statement
theorem focus_directrix_distance :
  let focus_y := 1 / 16
  let directrix_y := -1 / 16
  |focus_y - directrix_y| = 1 / 8 := by sorry

end focus_directrix_distance_l187_18772


namespace square_area_from_octagon_l187_18730

theorem square_area_from_octagon (side_length : ℝ) (octagon_area : ℝ) : 
  side_length > 0 →
  octagon_area = 7 * (side_length / 3)^2 →
  octagon_area = 105 →
  side_length^2 = 135 :=
by
  sorry

#check square_area_from_octagon

end square_area_from_octagon_l187_18730


namespace number_of_boys_in_school_l187_18719

theorem number_of_boys_in_school (total : ℕ) (boys : ℕ) :
  total = 1150 →
  (total - boys : ℚ) = (boys : ℚ) * total / 100 →
  boys = 92 := by
sorry

end number_of_boys_in_school_l187_18719


namespace brads_balloons_l187_18782

/-- Brad's balloon count problem -/
theorem brads_balloons (red : ℕ) (green : ℕ) 
  (h1 : red = 8) 
  (h2 : green = 9) : 
  red + green = 17 := by
  sorry

end brads_balloons_l187_18782


namespace sufficient_not_necessary_condition_l187_18783

-- Define the condition p
def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

-- Define the condition q
def q (x a : ℝ) : Prop := x ≤ a

-- State the theorem
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x) → a ≥ 2 := by
  sorry

end sufficient_not_necessary_condition_l187_18783


namespace car_rental_per_mile_rate_l187_18731

theorem car_rental_per_mile_rate (daily_rate : ℝ) (daily_budget : ℝ) (distance : ℝ) :
  daily_rate = 30 →
  daily_budget = 76 →
  distance = 200 →
  (daily_budget - daily_rate) / distance * 100 = 23 := by
sorry

end car_rental_per_mile_rate_l187_18731


namespace largest_number_l187_18792

def hcf (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem largest_number (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (hcf_cond : hcf a b c = 23)
  (lcm_cond : lcm a b c = 23 * 13 * 19 * 17) :
  max a (max b c) = 437 := by
  sorry

end largest_number_l187_18792


namespace polynomial_division_remainder_l187_18749

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ,
  3 * X^4 - 8 * X^3 + 20 * X^2 - 7 * X + 13 = 
  (X^2 + 5 * X - 3) * q + (168 * X^2 + 44 * X + 85) := by
  sorry

end polynomial_division_remainder_l187_18749


namespace car_price_difference_l187_18752

/-- Proves the difference in price between the old and new car -/
theorem car_price_difference
  (sale_percentage : ℝ)
  (additional_amount : ℝ)
  (new_car_price : ℝ)
  (h1 : sale_percentage = 0.8)
  (h2 : additional_amount = 4000)
  (h3 : new_car_price = 30000)
  (h4 : sale_percentage * (new_car_price - additional_amount) + additional_amount = new_car_price) :
  (new_car_price - additional_amount) / sale_percentage - new_car_price = 2500 := by
sorry

end car_price_difference_l187_18752


namespace cost_45_roses_l187_18778

/-- The cost of a bouquet is directly proportional to the number of roses it contains -/
axiom price_proportional_to_roses (n : ℕ) (price : ℚ) : n > 0 → price > 0 → ∃ k : ℚ, k > 0 ∧ price = k * n

/-- The cost of a bouquet with 15 roses -/
def cost_15 : ℚ := 25

/-- The number of roses in the first bouquet -/
def roses_15 : ℕ := 15

/-- The number of roses in the second bouquet -/
def roses_45 : ℕ := 45

/-- The theorem to prove -/
theorem cost_45_roses : 
  ∃ (k : ℚ), k > 0 ∧ cost_15 = k * roses_15 → k * roses_45 = 75 :=
sorry

end cost_45_roses_l187_18778


namespace least_addition_for_divisibility_least_addition_for_51234_div_9_least_addition_is_3_l187_18784

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
by sorry

theorem least_addition_for_51234_div_9 :
  ∃ (x : ℕ), x < 9 ∧ (51234 + x) % 9 = 0 ∧ ∀ (y : ℕ), y < x → (51234 + y) % 9 ≠ 0 :=
by
  apply least_addition_for_divisibility 51234 9
  norm_num

theorem least_addition_is_3 :
  ∃! (x : ℕ), x < 9 ∧ (51234 + x) % 9 = 0 ∧ ∀ (y : ℕ), y < x → (51234 + y) % 9 ≠ 0 ∧ x = 3 :=
by sorry

end least_addition_for_divisibility_least_addition_for_51234_div_9_least_addition_is_3_l187_18784


namespace triangle_ABC_properties_l187_18743

theorem triangle_ABC_properties (A B C : Real) (a b c : Real) 
  (m n : Fin 2 → Real) :
  m 0 = Real.cos A ∧ m 1 = Real.sin A ∧
  n 0 = Real.sqrt 2 - Real.sin A ∧ n 1 = Real.cos A ∧
  (m 0 * n 0 + m 1 * n 1 = 1) ∧
  b = 4 * Real.sqrt 2 ∧
  c = Real.sqrt 2 * a →
  A = π / 4 ∧ 
  (1/2 : Real) * b * c * Real.sin A = 16 := by
  sorry

end triangle_ABC_properties_l187_18743


namespace binomial_30_3_l187_18758

theorem binomial_30_3 : (30 : ℕ).choose 3 = 4060 := by sorry

end binomial_30_3_l187_18758


namespace log_sum_equals_two_l187_18787

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_sum_equals_two : 2 * lg 5 + lg 4 = 2 := by
  sorry

end log_sum_equals_two_l187_18787


namespace divisibility_proof_l187_18766

theorem divisibility_proof (n : ℕ) : 
  (∃ k : ℤ, 32^(3*n) - 1312^n = 1966 * k) ∧ 
  (∃ m : ℤ, 843^(2*n+1) - 1099^(2*n+1) + 16^(4*n+2) = 1967 * m) := by
  sorry

end divisibility_proof_l187_18766


namespace factors_of_30_to_4th_l187_18771

theorem factors_of_30_to_4th (h : 30 = 2 * 3 * 5) :
  (Finset.filter (fun d => d ≠ 1 ∧ d ≠ 30^4) (Nat.divisors (30^4))).card = 123 := by
  sorry

end factors_of_30_to_4th_l187_18771


namespace union_of_A_and_B_l187_18709

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x | x > 0} := by sorry

end union_of_A_and_B_l187_18709


namespace hyperbola_imaginary_axis_length_l187_18769

/-- Given a hyperbola x^2 + my^2 = 1 passing through the point (-√2, 2),
    the length of its imaginary axis is 4. -/
theorem hyperbola_imaginary_axis_length 
  (m : ℝ) 
  (h : (-Real.sqrt 2)^2 + m * 2^2 = 1) : 
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ↔ x^2 + m*y^2 = 1) ∧
    2*b = 4 := by
  sorry

end hyperbola_imaginary_axis_length_l187_18769


namespace train_length_l187_18715

/-- The length of a train given its crossing times over two platforms -/
theorem train_length (t1 t2 p1 p2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) (h3 : p1 > 0) (h4 : p2 > 0)
  (h5 : (L + p1) / t1 = (L + p2) / t2) : L = 100 :=
by
  sorry

#check train_length

end train_length_l187_18715


namespace max_ratio_two_digit_integers_with_mean_70_l187_18703

theorem max_ratio_two_digit_integers_with_mean_70 :
  ∀ x y : ℕ,
  10 ≤ x ∧ x ≤ 99 →
  10 ≤ y ∧ y ≤ 99 →
  (x + y) / 2 = 70 →
  ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →
  10 ≤ b ∧ b ≤ 99 →
  (a + b) / 2 = 70 →
  x / y ≤ 99 / 41 :=
by sorry

end max_ratio_two_digit_integers_with_mean_70_l187_18703


namespace brothers_to_madelines_money_ratio_l187_18739

theorem brothers_to_madelines_money_ratio (madelines_money : ℕ) (total_money : ℕ) : 
  madelines_money = 48 →
  total_money = 72 →
  (total_money - madelines_money) * 2 = madelines_money :=
by
  sorry

end brothers_to_madelines_money_ratio_l187_18739


namespace parabola_line_intersection_properties_l187_18705

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Theorem about properties of intersections between a line through the focus and a parabola -/
theorem parabola_line_intersection_properties (par : Parabola) 
  (A B : ParabolaPoint) (h_on_parabola : A.y^2 = 2*par.p*A.x ∧ B.y^2 = 2*par.p*B.x) 
  (h_through_focus : ∃ (k : ℝ), A.y = k*(A.x - par.p/2) ∧ B.y = k*(B.x - par.p/2)) :
  A.x * B.x = (par.p^2)/4 ∧ 
  1/(A.x + par.p/2) + 1/(B.x + par.p/2) = 2/par.p := by
  sorry

end parabola_line_intersection_properties_l187_18705


namespace exists_valid_surname_l187_18721

/-- Represents the positions of letters in a 6-letter Russian surname --/
structure SurnameLetter where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ
  sixth : ℕ

/-- Conditions for the Russian writer's surname --/
def is_valid_surname (s : SurnameLetter) : Prop :=
  s.first = s.third ∧
  s.second = s.fourth ∧
  s.fifth = s.first + 9 ∧
  s.sixth = s.second + s.fourth - 2 ∧
  3 * s.first = s.second - 4 ∧
  s.first + s.second + s.third + s.fourth + s.fifth + s.sixth = 83

/-- The theorem stating the existence of a valid surname --/
theorem exists_valid_surname : ∃ (s : SurnameLetter), is_valid_surname s :=
sorry

end exists_valid_surname_l187_18721


namespace inverse_of_A_l187_18728

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 7; 2, 3]

theorem inverse_of_A :
  A⁻¹ = !![-(3/2), 7/2; 1, -2] := by
  sorry

end inverse_of_A_l187_18728


namespace arithmetic_progression_log_range_l187_18725

theorem arithmetic_progression_log_range (x y : ℝ) : 
  (∃ k : ℝ, Real.log 2 - k = Real.log (Real.sin x - 1/3) ∧ 
             Real.log (Real.sin x - 1/3) - k = Real.log (1 - y)) →
  (y ≥ 7/9 ∧ ∀ M : ℝ, ∃ y' ≥ M, 
    ∃ k' : ℝ, Real.log 2 - k' = Real.log (Real.sin x - 1/3) ∧ 
             Real.log (Real.sin x - 1/3) - k' = Real.log (1 - y')) :=
by sorry

end arithmetic_progression_log_range_l187_18725


namespace unique_solution_ABCD_l187_18734

/-- Represents a base-5 number with two digits --/
def Base5TwoDigit (a b : Nat) : Nat := 5 * a + b

/-- Represents a base-5 number with one digit --/
def Base5OneDigit (a : Nat) : Nat := a

/-- Represents a base-5 number with two identical digits --/
def Base5TwoSameDigit (a : Nat) : Nat := 5 * a + a

theorem unique_solution_ABCD :
  ∀ A B C D : Nat,
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0) →
  (A < 5 ∧ B < 5 ∧ C < 5 ∧ D < 5) →
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (Base5TwoDigit A B + Base5OneDigit C = Base5TwoDigit D 0) →
  (Base5TwoDigit A B + Base5TwoDigit B A = Base5TwoSameDigit D) →
  A = 4 ∧ B = 1 ∧ C = 4 ∧ D = 4 := by
  sorry

#check unique_solution_ABCD

end unique_solution_ABCD_l187_18734


namespace fraction_equality_l187_18747

theorem fraction_equality (x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (2 * x - 8 * y) = -1) :
  (2 * x + 8 * y) / (4 * x - 2 * y) = 5 := by
  sorry

end fraction_equality_l187_18747


namespace bakery_doughnuts_given_away_l187_18790

/-- Given a bakery scenario, prove the number of doughnuts given away -/
theorem bakery_doughnuts_given_away
  (total_doughnuts : ℕ)
  (doughnuts_per_box : ℕ)
  (boxes_sold : ℕ)
  (h1 : total_doughnuts = 300)
  (h2 : doughnuts_per_box = 10)
  (h3 : boxes_sold = 27)
  : (total_doughnuts - boxes_sold * doughnuts_per_box) = 30 :=
by sorry

end bakery_doughnuts_given_away_l187_18790


namespace distance_city_A_to_B_distance_city_A_to_B_value_l187_18786

/-- Proves that the distance between city A and city B is 450 km given the problem conditions -/
theorem distance_city_A_to_B : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (time_eddy : ℝ) (time_freddy : ℝ) (speed_ratio : ℝ) (known_distance : ℝ) =>
    time_eddy = 3 ∧ 
    time_freddy = 4 ∧ 
    speed_ratio = 2 ∧ 
    known_distance = 300 →
    ∃ (distance_AB distance_AC : ℝ),
      distance_AB / time_eddy = speed_ratio * (distance_AC / time_freddy) ∧
      (distance_AB = known_distance ∨ distance_AC = known_distance) ∧
      distance_AB = 450

theorem distance_city_A_to_B_value : distance_city_A_to_B 3 4 2 300 := by
  sorry

end distance_city_A_to_B_distance_city_A_to_B_value_l187_18786


namespace binary_to_base_4_conversion_l187_18704

-- Define the binary number
def binary_num : ℕ := 11011011

-- Define the base 4 number
def base_4_num : ℕ := 3123

-- Theorem stating the equality of the binary and base 4 representations
theorem binary_to_base_4_conversion :
  (binary_num.digits 2).foldl (λ acc d => 2 * acc + d) 0 =
  (base_4_num.digits 4).foldl (λ acc d => 4 * acc + d) 0 :=
by sorry

end binary_to_base_4_conversion_l187_18704


namespace square_plus_reciprocal_squared_l187_18727

theorem square_plus_reciprocal_squared (x : ℝ) (h : x ≠ 0) :
  x^2 + (1/x^2) = 7 → x^4 + (1/x^4) = 47 := by
sorry

end square_plus_reciprocal_squared_l187_18727


namespace product_coefficient_sum_l187_18770

theorem product_coefficient_sum (a b c d : ℝ) : 
  (∀ x, (4 * x^2 - 6 * x + 5) * (9 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  12 * a + 6 * b + 3 * c + d = -27 := by
sorry

end product_coefficient_sum_l187_18770


namespace total_spent_equals_sum_l187_18793

/-- The total amount Jason spent on clothing -/
def total_spent : ℚ := 19.02

/-- The amount Jason spent on shorts -/
def shorts_cost : ℚ := 14.28

/-- The amount Jason spent on a jacket -/
def jacket_cost : ℚ := 4.74

/-- Theorem stating that the total amount spent is the sum of the costs of shorts and jacket -/
theorem total_spent_equals_sum : total_spent = shorts_cost + jacket_cost := by
  sorry

end total_spent_equals_sum_l187_18793


namespace quadratic_sum_l187_18773

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (8 * x^2 + 40 * x + 160 = a * (x + b)^2 + c) ∧ (a + b + c = 120.5) := by
  sorry

end quadratic_sum_l187_18773


namespace cube_sphere_surface_area_ratio_l187_18738

-- Define a cube with an inscribed sphere
structure CubeWithInscribedSphere where
  edge_length : ℝ
  sphere_radius : ℝ
  h_diameter : sphere_radius * 2 = edge_length

-- Theorem statement
theorem cube_sphere_surface_area_ratio 
  (c : CubeWithInscribedSphere) : 
  (6 * c.edge_length^2) / (4 * Real.pi * c.sphere_radius^2) = 6 / Real.pi := by
  sorry

end cube_sphere_surface_area_ratio_l187_18738


namespace johann_mail_delivery_l187_18757

theorem johann_mail_delivery (total : ℕ) (friend_delivery : ℕ) (num_friends : ℕ) :
  total = 180 →
  friend_delivery = 41 →
  num_friends = 2 →
  total - (friend_delivery * num_friends) = 98 :=
by sorry

end johann_mail_delivery_l187_18757


namespace part_one_part_two_l187_18717

-- Define the function f
def f (x a b : ℝ) : ℝ := x^2 - (a + 1) * x + b

-- Part 1
theorem part_one (a : ℝ) :
  (∃ x ∈ Set.Icc 2 3, f x a (-1) = 0) →
  (1/2 : ℝ) ≤ a ∧ a ≤ 5/3 := by sorry

-- Part 2
theorem part_two (x : ℝ) :
  (∀ a ∈ Set.Icc 2 3, f x a a < 0) →
  1 < x ∧ x < 2 := by sorry

end part_one_part_two_l187_18717


namespace triangle_side_length_l187_18700

theorem triangle_side_length (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  a + c = 2 * b →          -- Given condition
  a * c = 6 →              -- Given condition
  Real.cos (60 * π / 180) = (a^2 + c^2 - b^2) / (2 * a * c) →  -- Cosine theorem for 60°
  b = Real.sqrt 6 := by
sorry

-- Note: We use Real.cos and Real.sqrt to represent cosine and square root functions

end triangle_side_length_l187_18700


namespace rocky_day1_miles_l187_18759

def rocky_training (day1 : ℝ) : Prop :=
  let day2 := 2 * day1
  let day3 := 3 * day2
  day1 + day2 + day3 = 36

theorem rocky_day1_miles : ∃ (x : ℝ), rocky_training x ∧ x = 4 := by sorry

end rocky_day1_miles_l187_18759


namespace age_ratio_in_five_years_l187_18723

/-- Represents the ages of Sam and Dan -/
structure Ages where
  sam : ℕ
  dan : ℕ

/-- The conditions given in the problem -/
def age_conditions (a : Ages) : Prop :=
  (a.sam - 3 = 2 * (a.dan - 3)) ∧ 
  (a.sam - 7 = 3 * (a.dan - 7))

/-- The future condition we want to prove -/
def future_ratio (a : Ages) (years : ℕ) : Prop :=
  3 * (a.dan + years) = 2 * (a.sam + years)

/-- The main theorem to prove -/
theorem age_ratio_in_five_years (a : Ages) :
  age_conditions a → future_ratio a 5 := by
  sorry

end age_ratio_in_five_years_l187_18723


namespace points_on_unit_circle_l187_18724

theorem points_on_unit_circle (t : ℝ) :
  let x := (2 - t^2) / (2 + t^2)
  let y := 3 * t / (2 + t^2)
  x^2 + y^2 = 1 := by
sorry

end points_on_unit_circle_l187_18724


namespace one_right_angled_triangle_l187_18711

/-- A triangle with side lengths 15, 20, and x has exactly one right angle -/
def has_one_right_angle (x : ℤ) : Prop :=
  (x ^ 2 = 15 ^ 2 + 20 ^ 2) ∨ 
  (15 ^ 2 = x ^ 2 + 20 ^ 2) ∨ 
  (20 ^ 2 = 15 ^ 2 + x ^ 2)

/-- The triangle inequality is satisfied -/
def satisfies_triangle_inequality (x : ℤ) : Prop :=
  x > 0 ∧ 15 + 20 > x ∧ 15 + x > 20 ∧ 20 + x > 15

/-- There exists exactly one integer x that satisfies the conditions -/
theorem one_right_angled_triangle : 
  ∃! x : ℤ, satisfies_triangle_inequality x ∧ has_one_right_angle x :=
sorry

end one_right_angled_triangle_l187_18711


namespace reseating_twelve_women_l187_18760

def reseating_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1  -- Consider the empty case as 1
  | 1 => 1
  | 2 => 3
  | n + 3 => reseating_ways (n + 2) + reseating_ways (n + 1) + reseating_ways n

theorem reseating_twelve_women :
  reseating_ways 12 = 1201 := by
  sorry

end reseating_twelve_women_l187_18760


namespace touching_values_are_zero_and_neg_four_l187_18796

/-- Two linear functions with parallel, non-vertical graphs -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b ∧ g x = a * x + c
  not_vertical : ∃ (a : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + (f 0)

/-- Condition that (f x)^2 touches 4(g x) -/
def touches_squared_to_scaled (p : ParallelLinearFunctions) : Prop :=
  ∃! x, (p.f x)^2 = 4 * (p.g x)

/-- Values of A for which (g x)^2 touches A(f x) -/
def touching_values (p : ParallelLinearFunctions) : Set ℝ :=
  {A | ∃! x, (p.g x)^2 = A * (p.f x)}

/-- Main theorem -/
theorem touching_values_are_zero_and_neg_four 
    (p : ParallelLinearFunctions) 
    (h : touches_squared_to_scaled p) : 
    touching_values p = {0, -4} := by
  sorry


end touching_values_are_zero_and_neg_four_l187_18796


namespace rob_travel_time_l187_18799

/-- The time it takes Rob to get to the national park -/
def rob_time : ℝ := 1

/-- The time it takes Mark to get to the national park -/
def mark_time : ℝ := 3 * rob_time

/-- The head start time Mark has -/
def head_start : ℝ := 2

theorem rob_travel_time : 
  head_start + rob_time = mark_time ∧ rob_time = 1 := by sorry

end rob_travel_time_l187_18799


namespace min_value_ab_min_value_is_2sqrt2_exists_min_value_l187_18706

theorem min_value_ab (a b : ℝ) (h : a > 0 ∧ b > 0) (eq : 1/a + 2/b = Real.sqrt (a*b)) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 2/y = Real.sqrt (x*y) → a*b ≤ x*y :=
by
  sorry

theorem min_value_is_2sqrt2 (a b : ℝ) (h : a > 0 ∧ b > 0) (eq : 1/a + 2/b = Real.sqrt (a*b)) :
  a*b ≥ 2*Real.sqrt 2 :=
by
  sorry

theorem exists_min_value (a b : ℝ) (h : a > 0 ∧ b > 0) (eq : 1/a + 2/b = Real.sqrt (a*b)) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 2/y = Real.sqrt (x*y) ∧ x*y = 2*Real.sqrt 2 :=
by
  sorry

end min_value_ab_min_value_is_2sqrt2_exists_min_value_l187_18706


namespace antonov_remaining_packs_l187_18710

/-- The number of packs Antonov has after giving one pack to his sister -/
def remaining_packs (initial_candies : ℕ) (candies_per_pack : ℕ) : ℕ :=
  (initial_candies - candies_per_pack) / candies_per_pack

/-- Theorem stating that Antonov has 2 packs remaining -/
theorem antonov_remaining_packs :
  remaining_packs 60 20 = 2 := by
  sorry

end antonov_remaining_packs_l187_18710


namespace yannas_baking_problem_l187_18754

/-- Yanna's baking problem -/
theorem yannas_baking_problem (morning_butter_cookies morning_biscuits afternoon_butter_cookies afternoon_biscuits : ℕ) 
  (h1 : morning_butter_cookies = 20)
  (h2 : afternoon_butter_cookies = 10)
  (h3 : afternoon_biscuits = 20)
  (h4 : morning_biscuits + afternoon_biscuits = morning_butter_cookies + afternoon_butter_cookies + 30) :
  morning_biscuits = 40 := by
  sorry

end yannas_baking_problem_l187_18754
