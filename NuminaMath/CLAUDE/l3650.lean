import Mathlib

namespace quarter_circle_roll_path_length_l3650_365031

/-- The length of the path traveled by point F when rolling a quarter-circle region -/
theorem quarter_circle_roll_path_length 
  (EF : ℝ) -- Length of EF (radius of the quarter-circle)
  (h_EF : EF = 3 / Real.pi) -- Given condition that EF = 3/π cm
  : (2 * Real.pi * EF) = 6 := by
  sorry

end quarter_circle_roll_path_length_l3650_365031


namespace range_of_a_l3650_365050

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0 → |x - 3| > 1) ∧ 
  (∃ x : ℝ, |x - 3| > 1 ∧ x^2 - 4*a*x + 3*a^2 ≥ 0) ∧
  (a > 0) →
  a ≥ 4 ∨ (0 < a ∧ a ≤ 2/3) :=
by sorry

end range_of_a_l3650_365050


namespace smallest_n_for_divisibility_l3650_365081

/-- Given a positive odd number m, find the smallest natural number n 
    such that 2^1989 divides m^n - 1 -/
theorem smallest_n_for_divisibility (m : ℕ) (h_m_pos : 0 < m) (h_m_odd : Odd m) :
  ∃ (k : ℕ), ∃ (n : ℕ),
    (∀ (i : ℕ), i ≤ k → m % (2^i) = 1) ∧
    (m % (2^(k+1)) ≠ 1) ∧
    (n = 2^(1989 - k)) ∧
    (2^1989 ∣ m^n - 1) ∧
    (∀ (j : ℕ), j < n → ¬(2^1989 ∣ m^j - 1)) :=
by sorry

end smallest_n_for_divisibility_l3650_365081


namespace probability_one_from_each_group_l3650_365040

theorem probability_one_from_each_group :
  ∀ (total : ℕ) (group1 : ℕ) (group2 : ℕ),
    total = group1 + group2 →
    group1 > 0 →
    group2 > 0 →
    (group1 : ℚ) / total * group2 / (total - 1) +
    (group2 : ℚ) / total * group1 / (total - 1) = 5 / 9 :=
by
  sorry

end probability_one_from_each_group_l3650_365040


namespace bushes_needed_bushes_needed_proof_l3650_365092

/-- The number of containers of blueberries yielded by each bush -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries that can be traded for zucchinis -/
def containers_for_trade : ℕ := 6

/-- The number of zucchinis received in trade for containers_for_trade -/
def zucchinis_from_trade : ℕ := 3

/-- The target number of zucchinis -/
def target_zucchinis : ℕ := 60

/-- Theorem: The number of bushes needed to obtain the target number of zucchinis -/
theorem bushes_needed : ℕ := 12

/-- Proof that bushes_needed is correct -/
theorem bushes_needed_proof : 
  bushes_needed * containers_per_bush * zucchinis_from_trade = 
  target_zucchinis * containers_for_trade :=
by sorry

end bushes_needed_bushes_needed_proof_l3650_365092


namespace john_writing_years_l3650_365090

/-- Represents the number of months in a year -/
def months_per_year : ℕ := 12

/-- Represents the number of months it takes John to write a book -/
def months_per_book : ℕ := 2

/-- Represents the average earnings per book in dollars -/
def earnings_per_book : ℕ := 30000

/-- Represents the total earnings from writing in dollars -/
def total_earnings : ℕ := 3600000

/-- Calculates the number of years John has been writing -/
def years_writing : ℚ :=
  (total_earnings / earnings_per_book) / (months_per_year / months_per_book)

theorem john_writing_years :
  years_writing = 20 := by sorry

end john_writing_years_l3650_365090


namespace x_gt_1_sufficient_not_necessary_for_x_sq_gt_x_l3650_365012

theorem x_gt_1_sufficient_not_necessary_for_x_sq_gt_x :
  (∀ x : ℝ, x > 1 → x^2 > x) ∧ 
  (∃ x : ℝ, x^2 > x ∧ x ≤ 1) := by
  sorry

end x_gt_1_sufficient_not_necessary_for_x_sq_gt_x_l3650_365012


namespace perpendicular_from_line_perpendicular_and_parallel_perpendicular_from_perpendicular_and_parallel_l3650_365097

-- Define the types for planes and lines
variable {Point : Type*}
variable {Line : Type*}
variable {Plane : Type*}

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- Theorem 1
theorem perpendicular_from_line_perpendicular_and_parallel
  (l : Line) (α β : Plane) :
  line_perpendicular l α → line_parallel l β → perpendicular α β := by sorry

-- Theorem 2
theorem perpendicular_from_perpendicular_and_parallel
  (α β γ : Plane) :
  perpendicular α β → parallel α γ → perpendicular γ β := by sorry

end perpendicular_from_line_perpendicular_and_parallel_perpendicular_from_perpendicular_and_parallel_l3650_365097


namespace point_in_region_l3650_365029

def in_region (x y : ℝ) : Prop := 2 * x + y - 6 ≤ 0

theorem point_in_region :
  in_region 0 6 ∧
  ¬in_region 0 7 ∧
  ¬in_region 5 0 ∧
  ¬in_region 2 3 :=
by sorry

end point_in_region_l3650_365029


namespace power_division_equality_l3650_365034

theorem power_division_equality : (10^8 : ℝ) / (2 * 10^6) = 50 := by sorry

end power_division_equality_l3650_365034


namespace objects_meet_distance_l3650_365091

/-- The distance traveled by object A when it meets object B -/
def distance_A_traveled (t : ℝ) : ℝ := t^2 - t

/-- The distance traveled by object B when it meets object A -/
def distance_B_traveled (t : ℝ) : ℝ := t + 4 * t^2

/-- The initial distance between objects A and B -/
def initial_distance : ℝ := 405

theorem objects_meet_distance (t : ℝ) (h : t > 0) 
  (h1 : distance_A_traveled t + distance_B_traveled t = initial_distance) : 
  distance_A_traveled t = 72 := by
  sorry

end objects_meet_distance_l3650_365091


namespace problem_solution_l3650_365076

/-- The number of people initially working on the problem -/
def initial_people : ℕ := 1

/-- The initial working time in hours -/
def initial_time : ℕ := 10

/-- The working time after adding one person, in hours -/
def reduced_time : ℕ := 5

theorem problem_solution :
  initial_people * initial_time = (initial_people + 1) * reduced_time :=
by sorry

end problem_solution_l3650_365076


namespace negation_of_divisible_by_two_is_even_l3650_365077

theorem negation_of_divisible_by_two_is_even :
  (¬ ∀ n : ℤ, 2 ∣ n → Even n) ↔ (∃ n : ℤ, 2 ∣ n ∧ ¬Even n) := by
  sorry

end negation_of_divisible_by_two_is_even_l3650_365077


namespace line_tangent_to_log_curve_l3650_365053

/-- A line y = x + 1 is tangent to the curve y = ln(x + a) if and only if a = 2 -/
theorem line_tangent_to_log_curve (a : ℝ) : 
  (∃ x : ℝ, x + 1 = Real.log (x + a) ∧ 
   ∀ y : ℝ, y ≠ x → y + 1 ≠ Real.log (y + a) ∧
   (1 : ℝ) = 1 / (x + a)) ↔ 
  a = 2 := by
  sorry

end line_tangent_to_log_curve_l3650_365053


namespace sum_of_divisors_3600_l3650_365087

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the sum of divisors of 2^i * 3^j * 5^k is 3600, then i + j + k = 7 -/
theorem sum_of_divisors_3600 (i j k : ℕ) : 
  sum_of_divisors (2^i * 3^j * 5^k) = 3600 → i + j + k = 7 := by
  sorry

end sum_of_divisors_3600_l3650_365087


namespace arithmetic_sequence_fifth_term_l3650_365093

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 2 + a 4 = 16)
  (h_first : a 1 = 1) :
  a 5 = 15 := by
sorry

end arithmetic_sequence_fifth_term_l3650_365093


namespace absolute_value_equality_l3650_365094

theorem absolute_value_equality (a : ℝ) : 
  |a| = |5 + 1/3| → a = 5 + 1/3 ∨ a = -(5 + 1/3) := by
  sorry

end absolute_value_equality_l3650_365094


namespace triangle_side_range_l3650_365079

theorem triangle_side_range :
  ∀ m : ℝ,
  (3 > 0 ∧ 1 - 2*m > 0 ∧ 8 > 0) →
  (3 + (1 - 2*m) > 8 ∧ 3 + 8 > 1 - 2*m ∧ (1 - 2*m) + 8 > 3) →
  (-5 < m ∧ m < -2) :=
by sorry

end triangle_side_range_l3650_365079


namespace vasya_meeting_time_l3650_365086

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Calculates the difference in minutes between two times -/
def timeDifference (t1 t2 : Time) : Int :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

theorem vasya_meeting_time :
  let normalArrival : Time := ⟨18, 0, by norm_num, by norm_num⟩
  let earlyArrival : Time := ⟨17, 0, by norm_num, by norm_num⟩
  let meetingTime : Time := ⟨17, 50, by norm_num, by norm_num⟩
  let normalHomeArrival : Time := ⟨19, 0, by norm_num, by norm_num⟩  -- Assuming normal home arrival is at 19:00
  let earlyHomeArrival : Time := ⟨18, 40, by norm_num, by norm_num⟩  -- 20 minutes earlier than normal

  -- Vasya arrives 1 hour early
  timeDifference normalArrival earlyArrival = 60 →
  -- They arrive home 20 minutes earlier than usual
  timeDifference normalHomeArrival earlyHomeArrival = 20 →
  -- The meeting time is 10 minutes before the normal arrival time
  timeDifference normalArrival meetingTime = 10 →
  -- The meeting time is 50 minutes after the early arrival time
  timeDifference meetingTime earlyArrival = 50 →
  meetingTime = ⟨17, 50, by norm_num, by norm_num⟩ :=
by
  sorry


end vasya_meeting_time_l3650_365086


namespace beach_visitors_beach_visitors_proof_l3650_365057

theorem beach_visitors (initial_people : ℕ) (people_left : ℕ) (total_if_stayed : ℕ) : ℕ :=
  let total_before_leaving := total_if_stayed + people_left
  total_before_leaving - initial_people

#check beach_visitors 3 40 63 = 100

/- Proof
theorem beach_visitors_proof :
  beach_visitors 3 40 63 = 100 := by
  sorry
-/

end beach_visitors_beach_visitors_proof_l3650_365057


namespace qq_level_difference_l3650_365051

/-- Represents the QQ level system -/
structure QQLevel where
  activedays : ℕ
  stars : ℕ
  moons : ℕ
  suns : ℕ

/-- Calculates the total number of stars for a given level -/
def totalStars (level : ℕ) : ℕ := level

/-- Calculates the number of active days required for a given level -/
def activeDaysForLevel (level : ℕ) : ℕ := level * (level + 4)

/-- Converts stars to an equivalent QQ level -/
def starsToLevel (stars : ℕ) : ℕ := stars

/-- Theorem: The difference in active days between 1 sun and 2 moons 1 star is 203 -/
theorem qq_level_difference : 
  let sunLevel := starsToLevel (4 * 4)
  let currentLevel := starsToLevel (2 * 4 + 1)
  activeDaysForLevel sunLevel - activeDaysForLevel currentLevel = 203 := by
  sorry


end qq_level_difference_l3650_365051


namespace sqrt_undefined_range_l3650_365038

theorem sqrt_undefined_range (a : ℝ) : ¬ (∃ x : ℝ, x ^ 2 = 2 * a - 1) → a < 1 / 2 := by
  sorry

end sqrt_undefined_range_l3650_365038


namespace medicine_parts_for_child_l3650_365063

/-- Calculates the number of equal parts a medicine dose should be divided into -/
def medicine_parts (weight : ℕ) (dosage_per_kg : ℕ) (mg_per_part : ℕ) : ℕ :=
  (weight * dosage_per_kg * 1000) / mg_per_part

/-- Theorem: For a 30 kg child, with 5 ml/kg dosage and 50 mg parts, the dose divides into 3000 parts -/
theorem medicine_parts_for_child : medicine_parts 30 5 50 = 3000 := by
  sorry

end medicine_parts_for_child_l3650_365063


namespace recreation_area_tents_l3650_365044

/-- Represents the number of tents in different parts of the campsite -/
structure Campsite where
  north : ℕ
  east : ℕ
  center : ℕ
  south : ℕ

/-- Calculates the total number of tents in the campsite -/
def total_tents (c : Campsite) : ℕ :=
  c.north + c.east + c.center + c.south

/-- Theorem stating the total number of tents in the recreation area -/
theorem recreation_area_tents : ∃ (c : Campsite), 
  c.north = 100 ∧ 
  c.east = 2 * c.north ∧ 
  c.center = 4 * c.north ∧ 
  c.south = 200 ∧ 
  total_tents c = 900 := by
  sorry


end recreation_area_tents_l3650_365044


namespace z_is_real_z_is_pure_imaginary_z_in_third_quadrant_l3650_365030

-- Define the complex number z as a function of real m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 3*m) (m^2 - m - 6)

-- Theorem for when z is a real number
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = 3 ∨ m = -2 := by sorry

-- Theorem for when z is a pure imaginary number
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 0 := by sorry

-- Theorem for when z is in the third quadrant
theorem z_in_third_quadrant (m : ℝ) : (z m).re < 0 ∧ (z m).im < 0 ↔ 0 < m ∧ m < 3 := by sorry

end z_is_real_z_is_pure_imaginary_z_in_third_quadrant_l3650_365030


namespace world_grain_ratio_2010_l3650_365055

theorem world_grain_ratio_2010 : 
  let supply : ℕ := 1800000
  let demand : ℕ := 2400000
  (supply : ℚ) / demand = 3 / 4 := by sorry

end world_grain_ratio_2010_l3650_365055


namespace frog_jumped_farther_l3650_365022

/-- The frog's jump distance in inches -/
def frog_jump : ℕ := 39

/-- The grasshopper's jump distance in inches -/
def grasshopper_jump : ℕ := 17

/-- The difference in jump distance between the frog and the grasshopper -/
def jump_difference : ℕ := frog_jump - grasshopper_jump

theorem frog_jumped_farther : jump_difference = 22 := by
  sorry

end frog_jumped_farther_l3650_365022


namespace finite_solutions_except_two_l3650_365027

/-- The set of positive integer solutions x for the equation xn+1 | n^2+kn+1 -/
def S (n k : ℕ+) : Set ℕ+ :=
  {x | ∃ m : ℕ+, (x * n + 1) * m = n^2 + k * n + 1}

/-- The set of positive integers n for which S n k has at least two elements -/
def P (k : ℕ+) : Set ℕ+ :=
  {n | ∃ x y : ℕ+, x ≠ y ∧ x ∈ S n k ∧ y ∈ S n k}

theorem finite_solutions_except_two :
  ∀ k : ℕ+, k ≠ 2 → Set.Finite (P k) :=
sorry

end finite_solutions_except_two_l3650_365027


namespace square_mod_four_l3650_365098

theorem square_mod_four (n : ℤ) : (n^2) % 4 = 0 ∨ (n^2) % 4 = 1 := by
  sorry

end square_mod_four_l3650_365098


namespace total_tile_cost_l3650_365000

def courtyard_length : ℝ := 10
def courtyard_width : ℝ := 25
def tiles_per_sqft : ℝ := 4
def green_tile_percentage : ℝ := 0.4
def green_tile_cost : ℝ := 3
def red_tile_cost : ℝ := 1.5

theorem total_tile_cost : 
  let area := courtyard_length * courtyard_width
  let total_tiles := area * tiles_per_sqft
  let green_tiles := total_tiles * green_tile_percentage
  let red_tiles := total_tiles * (1 - green_tile_percentage)
  green_tiles * green_tile_cost + red_tiles * red_tile_cost = 2100 := by
sorry

end total_tile_cost_l3650_365000


namespace equal_slopes_iff_parallel_l3650_365020

-- Define the concept of a line in 2D space
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define what it means for two lines to be parallel
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define what it means for two lines to be distinct
def are_distinct (l1 l2 : Line) : Prop :=
  l1 ≠ l2

-- Theorem statement
theorem equal_slopes_iff_parallel (l1 l2 : Line) :
  are_distinct l1 l2 → (l1.slope = l2.slope ↔ are_parallel l1 l2) :=
sorry

end equal_slopes_iff_parallel_l3650_365020


namespace clothing_distribution_l3650_365062

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (remaining_loads : ℕ) 
  (h1 : total = 47)
  (h2 : first_load = 17)
  (h3 : remaining_loads = 5)
  : (total - first_load) / remaining_loads = 6 := by
  sorry

end clothing_distribution_l3650_365062


namespace first_player_wins_l3650_365045

/-- Represents a digit (0-9) -/
def Digit : Type := Fin 10

/-- Represents an operation (addition or multiplication) -/
inductive Operation
| add : Operation
| mul : Operation

/-- Represents a game state -/
structure GameState :=
  (digits : List Digit)
  (operations : List Operation)

/-- Represents a game move -/
structure Move :=
  (digit : Digit)
  (operation : Option Operation)

/-- Evaluates the final result of the game -/
def evaluateGame (state : GameState) : ℕ :=
  sorry

/-- Checks if a number is even -/
def isEven (n : ℕ) : Prop :=
  ∃ k, n = 2 * k

/-- Theorem: The first player can always win with optimal play -/
theorem first_player_wins :
  ∀ (initial_digit : Digit),
    isEven initial_digit.val →
    ∃ (strategy : List Move),
      ∀ (opponent_moves : List Move),
        let final_state := sorry
        isEven (evaluateGame final_state) :=
by sorry

end first_player_wins_l3650_365045


namespace find_y_l3650_365037

theorem find_y : ∃ y : ℝ, y > 0 ∧ 0.02 * y * y = 18 ∧ y = 30 := by sorry

end find_y_l3650_365037


namespace polynomial_property_l3650_365004

/-- Given a polynomial P(x) = x^4 + ax^3 + bx^2 + cx + d where a, b, c, d are constants,
    if P(1) = 1993, P(2) = 3986, and P(3) = 5979, then 1/4[P(11) + P(-7)] = 4693. -/
theorem polynomial_property (a b c d : ℝ) (P : ℝ → ℝ) 
    (h1 : ∀ x, P x = x^4 + a*x^3 + b*x^2 + c*x + d)
    (h2 : P 1 = 1993)
    (h3 : P 2 = 3986)
    (h4 : P 3 = 5979) :
    (1/4) * (P 11 + P (-7)) = 4693 := by
  sorry

end polynomial_property_l3650_365004


namespace flowers_in_vase_l3650_365019

theorem flowers_in_vase (roses : ℕ) (carnations : ℕ) : 
  roses = 5 → carnations = 5 → roses + carnations = 10 := by
  sorry

end flowers_in_vase_l3650_365019


namespace perpendicular_line_plane_condition_l3650_365064

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and between a line and a plane
variable (perp_line : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_plane_condition 
  (a l : Line) (α : Plane) (h_subset : subset a α) :
  (perp_plane l α → perp_line l a) ∧ 
  ∃ (l' : Line), perp_line l' a ∧ ¬perp_plane l' α :=
sorry

end perpendicular_line_plane_condition_l3650_365064


namespace midpoint_segment_length_is_three_l3650_365088

/-- A trapezoid with specific properties -/
structure Trapezoid where
  /-- The sum of the two base angles is 90° -/
  base_angles_sum : ℝ
  /-- The length of the upper base -/
  upper_base : ℝ
  /-- The length of the lower base -/
  lower_base : ℝ
  /-- The sum of the two base angles is 90° -/
  base_angles_sum_eq : base_angles_sum = 90
  /-- The length of the upper base is 5 -/
  upper_base_eq : upper_base = 5
  /-- The length of the lower base is 11 -/
  lower_base_eq : lower_base = 11

/-- The length of the segment connecting the midpoints of the two bases -/
def midpoint_segment_length (t : Trapezoid) : ℝ := 3

/-- Theorem: The length of the segment connecting the midpoints of the two bases is 3 -/
theorem midpoint_segment_length_is_three (t : Trapezoid) :
  midpoint_segment_length t = 3 := by
  sorry

#check midpoint_segment_length_is_three

end midpoint_segment_length_is_three_l3650_365088


namespace building_height_l3650_365085

/-- Given a flagpole and a building casting shadows under similar conditions,
    prove that the height of the building is 20 meters. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole_height : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 50)
  : (flagpole_height / flagpole_shadow) * building_shadow = 20 :=
by sorry

end building_height_l3650_365085


namespace quadrilateral_area_is_77_over_6_l3650_365074

/-- Represents the arrangement of three squares -/
structure SquareArrangement where
  small_side : ℝ
  medium_side : ℝ
  large_side : ℝ
  coplanar : Prop
  side_by_side : Prop

/-- Calculates the area of the quadrilateral formed in the square arrangement -/
def quadrilateral_area (arr : SquareArrangement) : ℝ :=
  sorry

/-- The main theorem stating that the quadrilateral area is 77/6 -/
theorem quadrilateral_area_is_77_over_6 (arr : SquareArrangement) 
  (h1 : arr.small_side = 3)
  (h2 : arr.medium_side = 5)
  (h3 : arr.large_side = 7)
  (h4 : arr.coplanar)
  (h5 : arr.side_by_side) :
  quadrilateral_area arr = 77 / 6 :=
sorry

end quadrilateral_area_is_77_over_6_l3650_365074


namespace tan_thirteen_pi_sixths_l3650_365035

theorem tan_thirteen_pi_sixths : Real.tan (13 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end tan_thirteen_pi_sixths_l3650_365035


namespace two_times_three_plus_two_times_three_l3650_365041

theorem two_times_three_plus_two_times_three : 2 * 3 + 2 * 3 = 12 := by
  sorry

end two_times_three_plus_two_times_three_l3650_365041


namespace heart_then_club_probability_l3650_365033

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (hearts : ℕ)
  (clubs : ℕ)

/-- Calculates the probability of drawing a heart first and a club second from a standard deck -/
def probability_heart_then_club (d : Deck) : ℚ :=
  (d.hearts : ℚ) / d.total_cards * d.clubs / (d.total_cards - 1)

/-- Theorem stating the probability of drawing a heart first and a club second from a standard 52-card deck -/
theorem heart_then_club_probability :
  let standard_deck : Deck := ⟨52, 13, 13⟩
  probability_heart_then_club standard_deck = 13 / 204 := by
  sorry

end heart_then_club_probability_l3650_365033


namespace range_of_expressions_l3650_365084

-- Define variables a and b with given constraints
theorem range_of_expressions (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : 2 < b ∧ b < 8) :
  -- (1) Range of a/b
  (1/8 : ℝ) < a/b ∧ a/b < 2 ∧
  -- (2) Range of 2a + 3b
  8 < 2*a + 3*b ∧ 2*a + 3*b < 32 ∧
  -- (3) Range of a - b
  -7 < a - b ∧ a - b < 2 := by
  sorry

end range_of_expressions_l3650_365084


namespace thirty_people_handshakes_l3650_365066

/-- The number of handshakes in a group of n people where each person shakes hands
    with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that in a group of 30 people, the total number of handshakes is 435. -/
theorem thirty_people_handshakes :
  handshakes 30 = 435 := by sorry

end thirty_people_handshakes_l3650_365066


namespace triangle_reconstruction_unique_l3650_365061

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle on a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents an acute triangle -/
structure AcuteTriangle where
  A : Point
  B : Point
  C : Point

/-- Represents the given information for triangle reconstruction -/
structure ReconstructionData where
  circumcircle : Circle
  C₀ : Point  -- Intersection of angle bisector from C with circumcircle
  A₁ : Point  -- Intersection of altitude from A with circumcircle
  B₁ : Point  -- Intersection of altitude from B with circumcircle

/-- Function to reconstruct the triangle from the given data -/
def reconstructTriangle (data : ReconstructionData) : AcuteTriangle :=
  sorry

/-- Theorem stating that the triangle can be uniquely reconstructed -/
theorem triangle_reconstruction_unique (data : ReconstructionData) :
  ∃! (triangle : AcuteTriangle),
    (Circle.center data.circumcircle).x ^ 2 + (Circle.center data.circumcircle).y ^ 2 = 
      data.circumcircle.radius ^ 2 ∧
    (data.C₀.x - triangle.C.x) ^ 2 + (data.C₀.y - triangle.C.y) ^ 2 = 
      data.circumcircle.radius ^ 2 ∧
    (data.A₁.x - triangle.A.x) ^ 2 + (data.A₁.y - triangle.A.y) ^ 2 = 
      data.circumcircle.radius ^ 2 ∧
    (data.B₁.x - triangle.B.x) ^ 2 + (data.B₁.y - triangle.B.y) ^ 2 = 
      data.circumcircle.radius ^ 2 :=
  sorry

end triangle_reconstruction_unique_l3650_365061


namespace correct_ball_placement_count_l3650_365018

/-- The number of ways to place four distinct balls into three boxes, leaving exactly one box empty -/
def ball_placement_count : ℕ := 42

/-- The number of distinct balls -/
def num_balls : ℕ := 4

/-- The number of boxes -/
def num_boxes : ℕ := 3

/-- Theorem stating that the number of ways to place the balls is correct -/
theorem correct_ball_placement_count :
  (∃ (f : Fin num_balls → Fin num_boxes),
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∃ (empty_box : Fin num_boxes), ∀ i, f i ≠ empty_box) ∧
    (∀ box : Fin num_boxes, box ≠ empty_box → ∃ i, f i = box)) →
  ball_placement_count = 42 :=
by sorry

end correct_ball_placement_count_l3650_365018


namespace number_value_l3650_365070

theorem number_value (x : ℝ) (number : ℝ) 
  (h1 : 5 - 5/x = number + 4/x) 
  (h2 : x = 9) : 
  number = 4 := by
sorry

end number_value_l3650_365070


namespace smallest_prime_twelve_less_than_square_l3650_365078

theorem smallest_prime_twelve_less_than_square : ∃ n : ℕ, 
  (∀ m : ℕ, m < n → ¬(Nat.Prime m ∧ ∃ k : ℕ, m = k^2 - 12)) ∧
  Nat.Prime n ∧ ∃ k : ℕ, n = k^2 - 12 := by
  sorry

end smallest_prime_twelve_less_than_square_l3650_365078


namespace cubic_increasing_and_odd_l3650_365021

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem cubic_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry


end cubic_increasing_and_odd_l3650_365021


namespace trigonometric_expression_value_l3650_365024

theorem trigonometric_expression_value (α : Real) (h : α = -35 * Real.pi / 6) :
  (2 * Real.sin (Real.pi + α) * Real.cos (Real.pi - α) - Real.sin (3 * Real.pi / 2 + α)) /
  (1 + Real.sin α ^ 2 - Real.cos (Real.pi / 2 + α) - Real.cos (Real.pi + α) ^ 2) = -Real.sqrt 3 := by
  sorry

end trigonometric_expression_value_l3650_365024


namespace limit_proof_l3650_365068

theorem limit_proof (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧
  ∀ x : ℝ, 0 < |x - 11| ∧ |x - 11| < δ →
    |(2 * x^2 - 21 * x - 11) / (x - 11) - 23| < ε := by
  sorry

end limit_proof_l3650_365068


namespace oranges_per_sack_l3650_365026

/-- Proves that the number of oranges per sack is 50, given the harvest conditions --/
theorem oranges_per_sack (total_sacks : ℕ) (discarded_sacks : ℕ) (total_oranges : ℕ)
  (h1 : total_sacks = 76)
  (h2 : discarded_sacks = 64)
  (h3 : total_oranges = 600) :
  total_oranges / (total_sacks - discarded_sacks) = 50 := by
  sorry

#check oranges_per_sack

end oranges_per_sack_l3650_365026


namespace greatest_missed_problems_to_pass_l3650_365006

/-- The number of problems on the test -/
def total_problems : ℕ := 50

/-- The minimum percentage required to pass the test -/
def passing_percentage : ℚ := 85 / 100

/-- The greatest number of problems that can be missed while still passing -/
def max_missed_problems : ℕ := 7

theorem greatest_missed_problems_to_pass :
  max_missed_problems = 
    (total_problems - Int.floor (passing_percentage * total_problems : ℚ)) := by
  sorry

end greatest_missed_problems_to_pass_l3650_365006


namespace room_equation_l3650_365082

/-- 
Theorem: For a positive integer x representing the number of rooms, 
if accommodating 6 people per room leaves exactly one room vacant, 
and accommodating 5 people per room leaves 4 people unaccommodated, 
then the equation 6(x-1) = 5x + 4 holds true.
-/
theorem room_equation (x : ℕ+) 
  (h1 : 6 * (x - 1) = 6 * x - 6)  -- With 6 people per room, one room is vacant
  (h2 : 5 * x + 4 = 6 * x - 6)    -- With 5 people per room, 4 people are unaccommodated
  : 6 * (x - 1) = 5 * x + 4 := by
  sorry


end room_equation_l3650_365082


namespace congruence_problem_l3650_365002

theorem congruence_problem (x : ℤ) : 
  (5 * x + 9) % 18 = 3 → (3 * x + 14) % 18 = 14 := by
  sorry

end congruence_problem_l3650_365002


namespace range_of_a_l3650_365072

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x, a*x^2 - x + a > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) →
  (0 < a ∧ a ≤ 1/2) ∨ a ≥ 1 :=
by sorry

end range_of_a_l3650_365072


namespace dogs_in_center_l3650_365058

/-- Represents the number of dogs that can perform a specific combination of tricks -/
structure DogTricks where
  jump : ℕ
  fetch : ℕ
  shake : ℕ
  jumpFetch : ℕ
  fetchShake : ℕ
  jumpShake : ℕ
  allThree : ℕ
  none : ℕ

/-- The total number of dogs in the center -/
def totalDogs (d : DogTricks) : ℕ :=
  d.allThree +
  (d.jumpFetch - d.allThree) +
  (d.fetchShake - d.allThree) +
  (d.jumpShake - d.allThree) +
  (d.jump - d.jumpFetch - d.jumpShake + d.allThree) +
  (d.fetch - d.jumpFetch - d.fetchShake + d.allThree) +
  (d.shake - d.jumpShake - d.fetchShake + d.allThree) +
  d.none

/-- Theorem stating that the total number of dogs in the center is 115 -/
theorem dogs_in_center (d : DogTricks)
  (h_jump : d.jump = 70)
  (h_fetch : d.fetch = 40)
  (h_shake : d.shake = 50)
  (h_jumpFetch : d.jumpFetch = 30)
  (h_fetchShake : d.fetchShake = 20)
  (h_jumpShake : d.jumpShake = 25)
  (h_allThree : d.allThree = 15)
  (h_none : d.none = 15) :
  totalDogs d = 115 := by
  sorry

end dogs_in_center_l3650_365058


namespace college_sports_participation_l3650_365080

/-- The total number of students who play at least one sport (cricket or basketball) -/
def total_students (cricket_players basketball_players both_players : ℕ) : ℕ :=
  cricket_players + basketball_players - both_players

/-- Theorem stating the total number of students playing at least one sport -/
theorem college_sports_participation : 
  total_students 500 600 220 = 880 := by
  sorry

end college_sports_participation_l3650_365080


namespace compound_interest_principal_exists_l3650_365099

theorem compound_interest_principal_exists : ∃ (P r : ℝ), 
  P > 0 ∧ r > 0 ∧ 
  P * (1 + r)^2 = 8800 ∧ 
  P * (1 + r)^3 = 9261 := by
sorry

end compound_interest_principal_exists_l3650_365099


namespace complex_square_on_negative_imaginary_axis_l3650_365054

/-- A complex number z lies on the negative half of the imaginary axis if its real part is 0 and its imaginary part is negative -/
def lies_on_negative_imaginary_axis (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im < 0

theorem complex_square_on_negative_imaginary_axis (a : ℝ) :
  lies_on_negative_imaginary_axis ((a + Complex.I) ^ 2) → a = -1 := by
  sorry

end complex_square_on_negative_imaginary_axis_l3650_365054


namespace tan_alpha_plus_pi_fourth_l3650_365007

theorem tan_alpha_plus_pi_fourth (α : Real) (h : Real.tan α = Real.sqrt 3) :
  Real.tan (α + π/4) = -2 - Real.sqrt 3 := by
  sorry

end tan_alpha_plus_pi_fourth_l3650_365007


namespace not_sufficient_nor_necessary_l3650_365065

theorem not_sufficient_nor_necessary (p q : Prop) :
  ¬(((p ∧ q) → ¬p) ∧ (¬p → (p ∧ q))) := by sorry

end not_sufficient_nor_necessary_l3650_365065


namespace supremum_of_expression_l3650_365016

theorem supremum_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ s : ℝ, s = -9/2 ∧ (- 1/(2*a) - 2/b ≤ s) ∧ 
  ∀ t : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → - 1/(2*x) - 2/y ≤ t) → s ≤ t :=
by sorry

end supremum_of_expression_l3650_365016


namespace product_divisible_by_17_l3650_365025

theorem product_divisible_by_17 : 
  17 ∣ (2002 + 3) * (2003 + 3) * (2004 + 3) * (2005 + 3) * (2006 + 3) * (2007 + 3) := by
  sorry

end product_divisible_by_17_l3650_365025


namespace percent_relation_l3650_365073

theorem percent_relation (x y z : ℝ) (h1 : 0.45 * z = 0.39 * y) (h2 : z = 0.65 * x) :
  y = 0.75 * x := by
  sorry

end percent_relation_l3650_365073


namespace stream_speed_l3650_365014

/-- Proves that the speed of the stream is 6 kmph given the conditions --/
theorem stream_speed (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 18 →
  (1 / (boat_speed - stream_speed)) = (2 * (1 / (boat_speed + stream_speed))) →
  stream_speed = 6 := by
sorry

end stream_speed_l3650_365014


namespace power_identity_l3650_365028

theorem power_identity (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) : 
  a^(3*m + 2*n) = 72 := by
sorry

end power_identity_l3650_365028


namespace inequality_proof_l3650_365075

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x < y) :
  x + Real.sqrt (y^2 + 2) < y + Real.sqrt (x^2 + 2) := by
  sorry

end inequality_proof_l3650_365075


namespace planar_figure_division_l3650_365036

/-- A planar figure with diameter 1 -/
structure PlanarFigure where
  diam : ℝ
  diam_eq_one : diam = 1

/-- The minimum diameter of n parts that a planar figure can be divided into -/
noncomputable def δ₂ (n : ℕ) (F : PlanarFigure) : ℝ := sorry

/-- Main theorem about division of planar figures -/
theorem planar_figure_division (F : PlanarFigure) : 
  (δ₂ 3 F ≤ Real.sqrt 3 / 2) ∧ 
  (δ₂ 4 F ≤ Real.sqrt 2 / 2) ∧ 
  (δ₂ 7 F ≤ 1 / 2) := by sorry

end planar_figure_division_l3650_365036


namespace exponent_addition_l3650_365069

theorem exponent_addition (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end exponent_addition_l3650_365069


namespace same_color_probability_l3650_365005

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def total_plates : ℕ := red_plates + blue_plates
def plates_selected : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates plates_selected + Nat.choose blue_plates plates_selected) / 
  Nat.choose total_plates plates_selected = 2 / 11 := by
  sorry

end same_color_probability_l3650_365005


namespace video_game_expenditure_l3650_365048

theorem video_game_expenditure (total : ℝ) (books toys snacks : ℝ) : 
  total = 45 →
  books = (1/4) * total →
  toys = (1/3) * total →
  snacks = (2/9) * total →
  total - (books + toys + snacks) = 8.75 :=
by sorry

end video_game_expenditure_l3650_365048


namespace men_earnings_l3650_365047

/-- Represents the total earnings of workers over a week. -/
structure Earnings where
  men : ℝ
  women : ℝ
  boys : ℝ

/-- Represents the work rates and hours of different groups of workers. -/
structure WorkData where
  X : ℝ  -- Number of women equivalent to 5 men
  M : ℝ  -- Hours worked by men
  W : ℝ  -- Hours worked by women
  B : ℝ  -- Hours worked by boys
  rm : ℝ  -- Wage rate for men per hour
  rw : ℝ  -- Wage rate for women per hour
  rb : ℝ  -- Wage rate for boys per hour

/-- Theorem stating the total earnings for men given the problem conditions. -/
theorem men_earnings (data : WorkData) (total : Earnings) :
  (5 : ℝ) * data.X * data.W * data.rw = (8 : ℝ) * data.B * data.rb →
  total.men + total.women + total.boys = 180 →
  total.men = (5 : ℝ) * data.M * data.rm :=
by sorry

end men_earnings_l3650_365047


namespace point_on_angle_bisector_l3650_365089

/-- 
Given a point M with coordinates (3n-2, 2n+7) that lies on the angle bisector 
of the second and fourth quadrants, prove that n = -1.
-/
theorem point_on_angle_bisector (n : ℝ) : 
  (∃ M : ℝ × ℝ, M.1 = 3*n - 2 ∧ M.2 = 2*n + 7 ∧ 
   M.1 + M.2 = 0) → n = -1 := by
sorry

end point_on_angle_bisector_l3650_365089


namespace log_drift_theorem_l3650_365039

/-- The time it takes for a log to drift downstream -/
def log_drift_time (downstream_time upstream_time : ℝ) : ℝ :=
  6 * (upstream_time - downstream_time)

/-- Theorem: Given the downstream and upstream travel times of a boat, 
    the time for a log to drift downstream is 12 hours -/
theorem log_drift_theorem (downstream_time upstream_time : ℝ) 
  (h1 : downstream_time = 2)
  (h2 : upstream_time = 3) : 
  log_drift_time downstream_time upstream_time = 12 := by
  sorry

end log_drift_theorem_l3650_365039


namespace negation_of_universal_proposition_l3650_365043

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l3650_365043


namespace nine_possible_values_for_D_l3650_365056

-- Define the type for digits (0-9)
def Digit := Fin 10

-- Define the addition equation
def addition_equation (A B C D : Digit) : Prop :=
  10000 * A.val + 1000 * B.val + 100 * A.val + 10 * C.val + A.val +
  10000 * C.val + 1000 * A.val + 100 * D.val + 10 * A.val + B.val =
  10000 * D.val + 1000 * C.val + 100 * D.val + 10 * D.val + D.val

-- Define the distinct digits condition
def distinct_digits (A B C D : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Theorem statement
theorem nine_possible_values_for_D :
  ∃ (s : Finset Digit),
    s.card = 9 ∧
    (∀ D ∈ s, ∃ A B C, distinct_digits A B C D ∧ addition_equation A B C D) ∧
    (∀ D, D ∉ s → ¬∃ A B C, distinct_digits A B C D ∧ addition_equation A B C D) :=
sorry

end nine_possible_values_for_D_l3650_365056


namespace jays_family_percentage_l3650_365042

theorem jays_family_percentage (total_guests : ℕ) (female_percentage : ℚ) (jays_family_females : ℕ) : 
  total_guests = 240 → 
  female_percentage = 60 / 100 → 
  jays_family_females = 72 → 
  (jays_family_females : ℚ) / (female_percentage * total_guests) = 1 / 2 := by
  sorry

end jays_family_percentage_l3650_365042


namespace reflection_line_equation_l3650_365010

/-- The equation of the reflection line for a triangle. -/
def reflection_line (D E F D' E' F' : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

/-- Theorem stating the equation of the reflection line for the given triangle. -/
theorem reflection_line_equation
  (D : ℝ × ℝ) (E : ℝ × ℝ) (F : ℝ × ℝ)
  (D' : ℝ × ℝ) (E' : ℝ × ℝ) (F' : ℝ × ℝ)
  (hD : D = (1, 2)) (hE : E = (6, 3)) (hF : F = (-3, 4))
  (hD' : D' = (1, -2)) (hE' : E' = (6, -3)) (hF' : F' = (-3, -4)) :
  reflection_line D E F D' E' F' = {p : ℝ × ℝ | p.2 = 0} :=
sorry

end reflection_line_equation_l3650_365010


namespace correct_setup_is_valid_l3650_365003

-- Define the structure for an experimental setup
structure ExperimentalSetup :=
  (num_plates : Nat)
  (bacteria_counts : List Nat)
  (average_count : Nat)

-- Define the conditions for a valid experimental setup
def is_valid_setup (setup : ExperimentalSetup) : Prop :=
  setup.num_plates ≥ 3 ∧
  setup.bacteria_counts.length = setup.num_plates ∧
  setup.average_count = setup.bacteria_counts.sum / setup.num_plates ∧
  setup.bacteria_counts.all (λ count => 
    setup.bacteria_counts.all (λ other_count => 
      (count : Int) - other_count ≤ 50 ∧ other_count - count ≤ 50))

-- Define the correct setup (option D)
def correct_setup : ExperimentalSetup :=
  { num_plates := 3,
    bacteria_counts := [210, 240, 250],
    average_count := 233 }

-- Theorem to prove
theorem correct_setup_is_valid :
  is_valid_setup correct_setup :=
sorry

end correct_setup_is_valid_l3650_365003


namespace even_sum_condition_l3650_365095

theorem even_sum_condition (m n : ℤ) :
  (∀ m n : ℤ, Even m ∧ Even n → Even (m + n)) ∧
  (∃ m n : ℤ, Even (m + n) ∧ (¬Even m ∨ ¬Even n)) :=
sorry

end even_sum_condition_l3650_365095


namespace conference_married_men_fraction_l3650_365023

theorem conference_married_men_fraction
  (total_women : ℕ)
  (single_women : ℕ)
  (married_women : ℕ)
  (married_men : ℕ)
  (h1 : single_women + married_women = total_women)
  (h2 : married_women = married_men)
  (h3 : (single_women : ℚ) / total_women = 3 / 7) :
  (married_men : ℚ) / (total_women + married_men) = 4 / 11 := by
  sorry

end conference_married_men_fraction_l3650_365023


namespace vector_subtraction_l3650_365067

theorem vector_subtraction (a b : ℝ × ℝ) : 
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by
  sorry

end vector_subtraction_l3650_365067


namespace smallest_number_l3650_365060

theorem smallest_number (a b c d e : ℚ) : 
  a = 0.803 → b = 0.8003 → c = 0.8 → d = 0.8039 → e = 0.809 →
  c ≤ a ∧ c ≤ b ∧ c ≤ d ∧ c ≤ e := by
  sorry

end smallest_number_l3650_365060


namespace skill_player_water_consumption_l3650_365017

/-- Proves that skill position players drink 6 ounces each given the conditions of the football team's water consumption problem. -/
theorem skill_player_water_consumption
  (total_water : ℕ)
  (num_linemen : ℕ)
  (num_skill_players : ℕ)
  (lineman_consumption : ℕ)
  (num_skill_players_before_refill : ℕ)
  (h1 : total_water = 126)
  (h2 : num_linemen = 12)
  (h3 : num_skill_players = 10)
  (h4 : lineman_consumption = 8)
  (h5 : num_skill_players_before_refill = 5)
  : (total_water - num_linemen * lineman_consumption) / num_skill_players_before_refill = 6 := by
  sorry

#check skill_player_water_consumption

end skill_player_water_consumption_l3650_365017


namespace hiker_first_day_distance_l3650_365052

/-- A hiker's three-day journey --/
def HikersJourney (h : ℝ) : Prop :=
  let d1 := 3 * h  -- Distance on day 1
  let d2 := 4 * (h - 1)  -- Distance on day 2
  let d3 := 5 * 6  -- Distance on day 3
  d1 + d2 + d3 = 68  -- Total distance

/-- The hiker walked 18 miles on the first day --/
theorem hiker_first_day_distance :
  ∃ h : ℝ, HikersJourney h ∧ 3 * h = 18 :=
sorry

end hiker_first_day_distance_l3650_365052


namespace not_all_five_digit_extendable_all_one_digit_extendable_minimal_extension_l3650_365011

/-- Represents a natural number with a specified number of digits -/
def NDigitNumber (n : ℕ) := { x : ℕ // x ≥ 10^(n-1) ∧ x < 10^n }

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- Appends k digits to an n-digit number -/
def append_digits (x : NDigitNumber n) (y : ℕ) (k : ℕ) : ℕ :=
  x.val * 10^k + y

theorem not_all_five_digit_extendable : ∃ x : NDigitNumber 6, 
  x.val ≥ 5 * 10^5 ∧ x.val < 6 * 10^5 ∧ 
  ¬∃ y : ℕ, y < 10^6 ∧ is_perfect_square (append_digits x y 6) :=
sorry

theorem all_one_digit_extendable : ∀ x : NDigitNumber 6, 
  x.val ≥ 10^5 ∧ x.val < 2 * 10^5 → 
  ∃ y : ℕ, y < 10^6 ∧ is_perfect_square (append_digits x y 6) :=
sorry

theorem minimal_extension (n : ℕ) : 
  (∀ x : NDigitNumber n, ∃ y : ℕ, y < 10^(n+1) ∧ is_perfect_square (append_digits x y (n+1))) ∧
  (∃ x : NDigitNumber n, ∀ y : ℕ, y < 10^n → ¬is_perfect_square (append_digits x y n)) :=
sorry

end not_all_five_digit_extendable_all_one_digit_extendable_minimal_extension_l3650_365011


namespace equation_represents_point_l3650_365049

/-- The equation x^2 + 36y^2 - 12x - 72y + 36 = 0 represents a single point (6, 1) in the xy-plane -/
theorem equation_represents_point :
  ∀ x y : ℝ, x^2 + 36*y^2 - 12*x - 72*y + 36 = 0 ↔ x = 6 ∧ y = 1 := by
  sorry

end equation_represents_point_l3650_365049


namespace sheets_in_stack_l3650_365096

/-- Given that 400 sheets of paper are 4 centimeters thick, 
    prove that a 14-inch high stack contains 3556 sheets. -/
theorem sheets_in_stack (sheets_in_4cm : ℕ) (thickness_4cm : ℝ) 
  (stack_height_inches : ℝ) (cm_per_inch : ℝ) :
  sheets_in_4cm = 400 →
  thickness_4cm = 4 →
  stack_height_inches = 14 →
  cm_per_inch = 2.54 →
  (stack_height_inches * cm_per_inch) / (thickness_4cm / sheets_in_4cm) = 3556 := by
  sorry

end sheets_in_stack_l3650_365096


namespace fifth_score_for_average_85_l3650_365001

/-- Given four test scores and a desired average, calculate the required fifth score -/
def calculate_fifth_score (score1 score2 score3 score4 : ℕ) (desired_average : ℚ) : ℚ :=
  5 * desired_average - (score1 + score2 + score3 + score4)

/-- Theorem: The fifth score needed to achieve an average of 85 given the first four scores -/
theorem fifth_score_for_average_85 :
  calculate_fifth_score 85 79 92 84 85 = 85 := by sorry

end fifth_score_for_average_85_l3650_365001


namespace katies_games_l3650_365013

theorem katies_games (friends_games : ℕ) (katies_extra_games : ℕ) 
  (h1 : friends_games = 59)
  (h2 : katies_extra_games = 22) : 
  friends_games + katies_extra_games = 81 :=
by sorry

end katies_games_l3650_365013


namespace runner_stops_in_third_quarter_l3650_365032

theorem runner_stops_in_third_quarter 
  (track_circumference : ℝ) 
  (total_distance : ℝ) 
  (quarter_length : ℝ) :
  track_circumference = 50 →
  total_distance = 5280 →
  quarter_length = track_circumference / 4 →
  ∃ (n : ℕ) (remaining_distance : ℝ),
    total_distance = n * track_circumference + remaining_distance ∧
    remaining_distance > 2 * quarter_length ∧
    remaining_distance ≤ 3 * quarter_length :=
by sorry

end runner_stops_in_third_quarter_l3650_365032


namespace min_perimeter_of_rectangle_l3650_365083

theorem min_perimeter_of_rectangle (l w : ℕ) : 
  l * w = 50 → 2 * (l + w) ≥ 30 := by
  sorry

end min_perimeter_of_rectangle_l3650_365083


namespace arithmetic_sequence_property_l3650_365008

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 3 + 3 * a 8 + a 13 = 120 → a 8 = 24 := by
  sorry

end arithmetic_sequence_property_l3650_365008


namespace divisibility_proof_l3650_365046

theorem divisibility_proof : (2 ∣ 32) ∧ (20 ∣ 320) := by
  sorry

end divisibility_proof_l3650_365046


namespace candy_difference_l3650_365015

/-- Theorem about the difference in candy counts in a box of rainbow nerds -/
theorem candy_difference (purple : ℕ) (yellow : ℕ) (green : ℕ) (total : ℕ) : 
  purple = 10 →
  yellow = purple + 4 →
  total = 36 →
  purple + yellow + green = total →
  yellow - green = 2 := by
sorry

end candy_difference_l3650_365015


namespace soda_price_ratio_l3650_365059

/-- The ratio of unit prices between two soda brands -/
theorem soda_price_ratio 
  (volume_A : ℝ) (volume_B : ℝ) (price_A : ℝ) (price_B : ℝ)
  (h_volume : volume_A = 1.25 * volume_B)
  (h_price : price_A = 0.85 * price_B)
  (h_positive : volume_B > 0 ∧ price_B > 0) :
  (price_A / volume_A) / (price_B / volume_B) = 17 / 25 := by
sorry

end soda_price_ratio_l3650_365059


namespace movie_session_duration_l3650_365009

/-- Represents the start time of a movie session -/
structure SessionTime where
  hour : ℕ
  minute : ℕ
  h_valid : hour < 24
  m_valid : minute < 60

/-- Represents the duration of a movie session -/
structure SessionDuration where
  hours : ℕ
  minutes : ℕ
  m_valid : minutes < 60

/-- Checks if a given SessionTime is consistent with the known session times -/
def is_consistent (st : SessionTime) (duration : SessionDuration) : Prop :=
  let next_session := SessionTime.mk 
    ((st.hour + duration.hours + (st.minute + duration.minutes) / 60) % 24)
    ((st.minute + duration.minutes) % 60)
    sorry
    sorry
  (st.hour = 12 ∧ next_session.hour = 13) ∨
  (st.hour = 13 ∧ next_session.hour = 14) ∨
  (st.hour = 23 ∧ next_session.hour = 24) ∨
  (st.hour = 24 ∧ next_session.hour = 1)

theorem movie_session_duration : 
  ∃ (start : SessionTime) (duration : SessionDuration),
    duration.hours = 1 ∧ 
    duration.minutes = 50 ∧
    is_consistent start duration ∧
    (∀ (other_duration : SessionDuration),
      is_consistent start other_duration → 
      other_duration = duration) := by
  sorry

end movie_session_duration_l3650_365009


namespace x_gt_3_sufficient_not_necessary_for_x_sq_gt_4_l3650_365071

theorem x_gt_3_sufficient_not_necessary_for_x_sq_gt_4 :
  (∀ x : ℝ, x > 3 → x^2 > 4) ∧
  ¬(∀ x : ℝ, x^2 > 4 → x > 3) :=
by sorry

end x_gt_3_sufficient_not_necessary_for_x_sq_gt_4_l3650_365071
