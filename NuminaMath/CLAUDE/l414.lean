import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_k_range_l414_41429

/-- The equation of an ellipse in terms of parameter k -/
def is_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (3 + k) + y^2 / (2 - k) = 1 ∧ 
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

/-- The range of k for which the equation represents an ellipse -/
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ k ∈ Set.Ioo (-3 : ℝ) (-1/2) ∪ Set.Ioo (-1/2 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l414_41429


namespace NUMINAMATH_CALUDE_sandy_comic_books_l414_41404

theorem sandy_comic_books (x : ℕ) : 
  (x / 2 : ℚ) - 3 + 6 = 13 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_sandy_comic_books_l414_41404


namespace NUMINAMATH_CALUDE_double_time_double_discount_l414_41490

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

end NUMINAMATH_CALUDE_double_time_double_discount_l414_41490


namespace NUMINAMATH_CALUDE_intersection_of_S_and_T_l414_41492

def S : Set ℕ := {1, 4, 5}
def T : Set ℕ := {2, 3, 4}

theorem intersection_of_S_and_T : S ∩ T = {4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_S_and_T_l414_41492


namespace NUMINAMATH_CALUDE_elevator_time_l414_41489

/-- Represents the number of floors in the building -/
def num_floors : ℕ := 9

/-- Represents the number of steps per floor -/
def steps_per_floor : ℕ := 30

/-- Represents the number of steps Jake descends per second -/
def jake_steps_per_second : ℕ := 3

/-- Represents the time difference (in seconds) between Jake and the elevator reaching the ground floor -/
def time_difference : ℕ := 30

/-- Calculates the total number of steps Jake needs to descend -/
def total_steps : ℕ := (num_floors - 1) * steps_per_floor

/-- Calculates the time (in seconds) it takes Jake to reach the ground floor -/
def jake_time : ℕ := total_steps / jake_steps_per_second

/-- Theorem stating that the elevator takes 50 seconds to reach the ground level -/
theorem elevator_time : jake_time - time_difference = 50 := by sorry

end NUMINAMATH_CALUDE_elevator_time_l414_41489


namespace NUMINAMATH_CALUDE_wedge_volume_l414_41414

/-- The volume of a wedge cut from a cylindrical log -/
theorem wedge_volume (d h r : ℝ) (θ : ℝ) : 
  d = 18 →                           -- diameter of the log
  h = d →                            -- height of the cylindrical section
  r = d / 2 →                        -- radius of the log
  θ = 60 →                           -- angle between cuts in degrees
  (π * r^2 * h) / 2 = 729 * π := by
  sorry

#check wedge_volume

end NUMINAMATH_CALUDE_wedge_volume_l414_41414


namespace NUMINAMATH_CALUDE_imaginary_part_of_square_l414_41426

theorem imaginary_part_of_square : Complex.im ((1 - 4 * Complex.I) ^ 2) = -8 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_square_l414_41426


namespace NUMINAMATH_CALUDE_jacket_price_change_l414_41406

theorem jacket_price_change (P : ℝ) (x : ℝ) (h : x > 0) :
  P * (1 - (x / 100)^2) * 0.9 = 0.75 * P →
  x = 100 * Real.sqrt (1 / 6) := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_change_l414_41406


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l414_41475

theorem tangent_line_to_circle (r : ℝ) (h_pos : r > 0) :
  (∀ x y : ℝ, x + y = r → x^2 + y^2 = 4*r → 
    ∀ ε > 0, ∃ x' y' : ℝ, x' + y' = r ∧ (x' - x)^2 + (y' - y)^2 < ε^2 ∧ x'^2 + y'^2 ≠ 4*r) →
  r = 8 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l414_41475


namespace NUMINAMATH_CALUDE_coffee_consumption_l414_41402

theorem coffee_consumption (x : ℝ) : 
  x > 0 → -- Tom's coffee size is positive
  (2/3 * x + (5/48 * x + 3) = 5/4 * (2/3 * x) - (5/48 * x + 3)) → -- They drink the same amount
  x + 1.25 * x = 36 -- Total coffee consumed is 36 ounces
  := by sorry

end NUMINAMATH_CALUDE_coffee_consumption_l414_41402


namespace NUMINAMATH_CALUDE_thumbtacks_total_l414_41470

/-- Given 3 cans of thumbtacks, where 120 thumbtacks are used from each can
    and 30 thumbtacks remain in each can after use, prove that the total
    number of thumbtacks in the three full cans initially was 450. -/
theorem thumbtacks_total (cans : Nat) (used_per_can : Nat) (remaining_per_can : Nat)
    (h1 : cans = 3)
    (h2 : used_per_can = 120)
    (h3 : remaining_per_can = 30) :
    cans * (used_per_can + remaining_per_can) = 450 := by
  sorry

end NUMINAMATH_CALUDE_thumbtacks_total_l414_41470


namespace NUMINAMATH_CALUDE_xy_greater_than_xz_l414_41471

theorem xy_greater_than_xz (x y z : ℝ) 
  (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 1) : x * y > x * z := by
  sorry

end NUMINAMATH_CALUDE_xy_greater_than_xz_l414_41471


namespace NUMINAMATH_CALUDE_trigonometric_identity_l414_41469

theorem trigonometric_identity (α β γ : ℝ) : 
  (Real.sin α + Real.sin β + Real.sin γ - Real.sin (α + β + γ)) / 
  (Real.cos α + Real.cos β + Real.cos γ + Real.cos (α + β + γ)) = 
  Real.tan ((α + β) / 2) * Real.tan ((β + γ) / 2) * Real.tan ((γ + α) / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l414_41469


namespace NUMINAMATH_CALUDE_race_distance_p_l414_41474

/-- The distance P runs in a race where:
  1. P's speed is 20% faster than Q's speed
  2. Q starts 300 meters ahead of P
  3. P and Q finish the race at the same time
-/
theorem race_distance_p (vq : ℝ) : ∃ dp : ℝ,
  let vp := 1.2 * vq
  let dq := dp - 300
  dp / vp = dq / vq ∧ dp = 1800 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_p_l414_41474


namespace NUMINAMATH_CALUDE_no_solutions_in_interval_l414_41453

theorem no_solutions_in_interval (x : Real) : 
  x ∈ Set.Icc 0 Real.pi → 
  ¬(Real.sin (Real.pi * Real.cos x) = Real.cos (Real.pi * Real.sin x)) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_in_interval_l414_41453


namespace NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_three_l414_41427

/-- Two lines in the form Ax + By + C = 0 are parallel if and only if their slopes are equal -/
def parallel_lines (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ A1 / B1 = A2 / B2

/-- The first line: 3x + ay + 1 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  3 * x + a * y + 1 = 0

/-- The second line: (a+2)x + y + a = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  (a + 2) * x + y + a = 0

/-- The theorem stating that the lines are parallel if and only if a = -3 -/
theorem lines_parallel_iff_a_eq_neg_three :
  ∃ (a : ℝ), parallel_lines 3 a 1 (a + 2) 1 a ↔ a = -3 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_three_l414_41427


namespace NUMINAMATH_CALUDE_residue_mod_37_l414_41401

theorem residue_mod_37 : ∃ k : ℤ, -927 = 37 * k + 35 ∧ (35 : ℤ) ∈ Set.range (fun i => i : Fin 37 → ℤ) := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_37_l414_41401


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l414_41403

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 24) :
  let r := d / 4
  (4 / 3) * π * r^3 = 2304 * π := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l414_41403


namespace NUMINAMATH_CALUDE_ellipse_major_axis_l414_41457

/-- The equation of an ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

/-- The length of the major axis of the ellipse -/
def major_axis_length : ℝ := 6

/-- Theorem: The length of the major axis of the ellipse x^2 + 9y^2 = 9 is 6 -/
theorem ellipse_major_axis :
  ∀ x y : ℝ, ellipse_equation x y → major_axis_length = 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_l414_41457


namespace NUMINAMATH_CALUDE_division_problem_l414_41477

theorem division_problem :
  ∃ (dividend : ℕ), 
    dividend = 11889708 ∧ 
    dividend / 12 = 990809 ∧ 
    dividend % 12 = 0 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l414_41477


namespace NUMINAMATH_CALUDE_max_value_of_linear_combination_l414_41423

theorem max_value_of_linear_combination (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 6 * y ≤ 12) : 
  x + 2 * y ≤ 4 ∧ ∃ (x₀ y₀ : ℝ), 4 * x₀ + 3 * y₀ ≤ 10 ∧ 3 * x₀ + 6 * y₀ ≤ 12 ∧ x₀ + 2 * y₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_linear_combination_l414_41423


namespace NUMINAMATH_CALUDE_tight_sequence_x_range_l414_41479

/-- Definition of a tight sequence -/
def is_tight_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (1/2 : ℝ) ≤ a (n+1) / a n ∧ a (n+1) / a n ≤ 2

/-- Theorem about the range of x in a specific tight sequence -/
theorem tight_sequence_x_range (a : ℕ → ℝ) (x : ℝ) 
  (h_tight : is_tight_sequence a)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 3/2)
  (h_a3 : a 3 = x)
  (h_a4 : a 4 = 4) :
  2 ≤ x ∧ x ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_tight_sequence_x_range_l414_41479


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l414_41438

theorem arithmetic_sequence_sum : 
  ∀ (a d n : ℕ) (last : ℕ),
    a = 2 →
    d = 2 →
    last = 20 →
    last = a + (n - 1) * d →
    (n : ℕ) * (a + last) / 2 = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l414_41438


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l414_41487

theorem sqrt_expression_equality : 
  Real.sqrt 2 * Real.sqrt 6 - 4 * Real.sqrt (1/2) - (1 - Real.sqrt 3)^2 = 4 * Real.sqrt 3 - 2 * Real.sqrt 2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l414_41487


namespace NUMINAMATH_CALUDE_bicycle_trip_average_speed_l414_41409

/-- Proves that for a bicycle trip with two parts:
    1. 10 km at 12 km/hr
    2. 12 km at 10 km/hr
    The average speed for the entire trip is 660/61 km/hr. -/
theorem bicycle_trip_average_speed :
  let distance1 : ℝ := 10
  let speed1 : ℝ := 12
  let distance2 : ℝ := 12
  let speed2 : ℝ := 10
  let total_distance : ℝ := distance1 + distance2
  let total_time : ℝ := distance1 / speed1 + distance2 / speed2
  let average_speed : ℝ := total_distance / total_time
  average_speed = 660 / 61 := by
sorry

end NUMINAMATH_CALUDE_bicycle_trip_average_speed_l414_41409


namespace NUMINAMATH_CALUDE_tournament_points_l414_41467

-- Define the type for teams
inductive Team : Type
  | A | B | C | D | E

-- Define the function for points
def points : Team → ℕ
  | Team.A => 7
  | Team.B => 6
  | Team.C => 4
  | Team.D => 5
  | Team.E => 2

-- Define the properties of the tournament
axiom different_points : ∀ t1 t2 : Team, t1 ≠ t2 → points t1 ≠ points t2
axiom a_most_points : ∀ t : Team, t ≠ Team.A → points Team.A > points t
axiom b_beat_a : points Team.B > points Team.A
axiom b_no_loss : ∀ t : Team, t ≠ Team.B → points Team.B ≥ points t
axiom c_no_loss : ∀ t : Team, t ≠ Team.C → points Team.C ≥ points t
axiom d_more_than_c : points Team.D > points Team.C

-- Theorem to prove
theorem tournament_points : 
  (points Team.A = 7 ∧ 
   points Team.B = 6 ∧ 
   points Team.C = 4 ∧ 
   points Team.D = 5 ∧ 
   points Team.E = 2) := by
  sorry

end NUMINAMATH_CALUDE_tournament_points_l414_41467


namespace NUMINAMATH_CALUDE_wednesdays_temperature_l414_41452

theorem wednesdays_temperature (monday tuesday wednesday : ℤ) : 
  tuesday = monday + 4 →
  wednesday = monday - 6 →
  tuesday = 22 →
  wednesday = 12 := by
sorry

end NUMINAMATH_CALUDE_wednesdays_temperature_l414_41452


namespace NUMINAMATH_CALUDE_original_mean_l414_41430

theorem original_mean (n : ℕ) (decrement : ℝ) (updated_mean : ℝ) (h1 : n = 50) (h2 : decrement = 34) (h3 : updated_mean = 166) : 
  (n : ℝ) * updated_mean + n * decrement = n * 200 := by
  sorry

end NUMINAMATH_CALUDE_original_mean_l414_41430


namespace NUMINAMATH_CALUDE_dvd_average_price_l414_41419

/-- Calculates the average price of DVDs bought from two boxes with different prices -/
theorem dvd_average_price (box1_count : ℕ) (box1_price : ℚ) (box2_count : ℕ) (box2_price : ℚ) :
  box1_count = 10 →
  box1_price = 2 →
  box2_count = 5 →
  box2_price = 5 →
  (box1_count * box1_price + box2_count * box2_price) / (box1_count + box2_count : ℚ) = 3 := by
  sorry

#check dvd_average_price

end NUMINAMATH_CALUDE_dvd_average_price_l414_41419


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l414_41472

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_1 + 3a_8 + a_15 = 60,
    prove that 2a_9 - a_10 = 12 -/
theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + 3 * a 8 + a 15 = 60) :
  2 * a 9 - a 10 = 12 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l414_41472


namespace NUMINAMATH_CALUDE_lori_earnings_l414_41481

/-- Represents the earnings from Lori's carsharing company -/
def carsharing_earnings (num_red_cars num_white_cars : ℕ) 
  (red_car_rate white_car_rate : ℚ) (rental_hours : ℕ) : ℚ :=
  let total_minutes := rental_hours * 60
  let red_car_earnings := num_red_cars * red_car_rate * total_minutes
  let white_car_earnings := num_white_cars * white_car_rate * total_minutes
  red_car_earnings + white_car_earnings

/-- Theorem stating that Lori's earnings are $2340 given the problem conditions -/
theorem lori_earnings : 
  carsharing_earnings 3 2 3 2 3 = 2340 := by
  sorry

#eval carsharing_earnings 3 2 3 2 3

end NUMINAMATH_CALUDE_lori_earnings_l414_41481


namespace NUMINAMATH_CALUDE_mike_practice_hours_l414_41463

/-- Calculates the total practice hours for a goalkeeper before a game -/
def total_practice_hours (weekday_hours : ℕ) (saturday_hours : ℕ) (weeks_until_game : ℕ) : ℕ :=
  (weekday_hours * 5 + saturday_hours) * weeks_until_game

/-- Theorem: Mike's total practice hours before the next game -/
theorem mike_practice_hours :
  total_practice_hours 3 5 3 = 60 := by
  sorry

#eval total_practice_hours 3 5 3

end NUMINAMATH_CALUDE_mike_practice_hours_l414_41463


namespace NUMINAMATH_CALUDE_coin_flip_probability_l414_41473

theorem coin_flip_probability :
  let n : ℕ := 6  -- total number of coins
  let k : ℕ := 3  -- number of specific coins we're interested in
  let total_outcomes : ℕ := 2^n
  let favorable_outcomes : ℕ := 2^(n - k)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l414_41473


namespace NUMINAMATH_CALUDE_tomatoes_eaten_by_birds_l414_41480

theorem tomatoes_eaten_by_birds 
  (initial_tomatoes : ℕ) 
  (remaining_tomatoes : ℕ) 
  (h1 : initial_tomatoes = 21)
  (h2 : remaining_tomatoes = 14) :
  initial_tomatoes - remaining_tomatoes = 7 := by
sorry

end NUMINAMATH_CALUDE_tomatoes_eaten_by_birds_l414_41480


namespace NUMINAMATH_CALUDE_complement_of_M_l414_41408

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 - 4 ≤ 0}

-- State the theorem
theorem complement_of_M :
  (U \ M) = {x : ℝ | x < -2 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l414_41408


namespace NUMINAMATH_CALUDE_ellipse_focal_length_specific_ellipse_focal_length_l414_41449

/-- The focal length of an ellipse with equation x²/a² + y²/b² = 1 is 2c, where c² = a² - b² -/
theorem ellipse_focal_length (a b : ℝ) (h : 0 < b ∧ b < a) :
  let c := Real.sqrt (a^2 - b^2)
  let focal_length := 2 * c
  focal_length = 2 → a^2 = 2 ∧ b^2 = 1 := by sorry

/-- The focal length of the ellipse x²/2 + y² = 1 is 2 -/
theorem specific_ellipse_focal_length :
  let a := Real.sqrt 2
  let b := 1
  let c := Real.sqrt (a^2 - b^2)
  let focal_length := 2 * c
  focal_length = 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_specific_ellipse_focal_length_l414_41449


namespace NUMINAMATH_CALUDE_profit_percentage_60_percent_l414_41442

/-- Profit percentage for 60% of apples given total apples, profit percentages, and sales distribution --/
theorem profit_percentage_60_percent (total_apples : ℝ) (profit_40_percent : ℝ) (total_profit_percent : ℝ) :
  total_apples = 280 →
  profit_40_percent = 10 →
  total_profit_percent = 22.000000000000007 →
  let profit_60_percent := 
    (total_profit_percent * total_apples - profit_40_percent * (0.4 * total_apples)) / (0.6 * total_apples) * 100
  profit_60_percent = 30 := by sorry

end NUMINAMATH_CALUDE_profit_percentage_60_percent_l414_41442


namespace NUMINAMATH_CALUDE_sixth_grade_count_l414_41448

/-- The number of students in the sixth grade -/
def sixth_grade_students : ℕ := 108

/-- The total number of students in fifth and sixth grades -/
def total_students : ℕ := 200

/-- The number of fifth grade students who went to the celebration -/
def fifth_grade_celebration : ℕ := 11

/-- The percentage of sixth grade students who went to the celebration -/
def sixth_grade_celebration_percent : ℚ := 1/4

theorem sixth_grade_count : 
  sixth_grade_students = 108 ∧
  total_students = 200 ∧
  fifth_grade_celebration = 11 ∧
  sixth_grade_celebration_percent = 1/4 ∧
  (total_students - sixth_grade_students - fifth_grade_celebration) = 
  (sixth_grade_students * (1 - sixth_grade_celebration_percent)) :=
by sorry

end NUMINAMATH_CALUDE_sixth_grade_count_l414_41448


namespace NUMINAMATH_CALUDE_ants_in_park_l414_41495

-- Define the dimensions of the park in meters
def park_width : ℝ := 100
def park_length : ℝ := 130

-- Define the ant density per square centimeter
def ants_per_sq_cm : ℝ := 1.2

-- Define the conversion factor from meters to centimeters
def cm_per_meter : ℝ := 100

-- Theorem statement
theorem ants_in_park :
  let park_area_sq_cm := park_width * park_length * cm_per_meter^2
  let total_ants := park_area_sq_cm * ants_per_sq_cm
  total_ants = 156000000 := by
  sorry

end NUMINAMATH_CALUDE_ants_in_park_l414_41495


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l414_41431

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_expression :
  3 * (4 - 2*i) + 2*i * (3 + 2*i) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l414_41431


namespace NUMINAMATH_CALUDE_rice_trader_problem_l414_41488

/-- A rice trader problem -/
theorem rice_trader_problem (initial_stock restocked final_stock : ℕ) 
  (h1 : initial_stock = 55)
  (h2 : restocked = 132)
  (h3 : final_stock = 164) :
  ∃ (sold : ℕ), initial_stock - sold + restocked = final_stock ∧ sold = 23 := by
  sorry

end NUMINAMATH_CALUDE_rice_trader_problem_l414_41488


namespace NUMINAMATH_CALUDE_interest_group_signup_ways_l414_41418

theorem interest_group_signup_ways (num_students : ℕ) (num_groups : ℕ) : 
  num_students = 4 → num_groups = 3 → (num_groups ^ num_students : ℕ) = 81 := by
  sorry

end NUMINAMATH_CALUDE_interest_group_signup_ways_l414_41418


namespace NUMINAMATH_CALUDE_correct_members_taken_course_not_passed_l414_41468

/-- Represents a swim club with members and their test status -/
structure SwimClub where
  total_members : ℕ
  passed_test : ℕ
  not_taken_course : ℕ

/-- The number of members who have taken the preparatory course but not passed the test -/
def members_taken_course_not_passed (club : SwimClub) : ℕ :=
  club.total_members - club.passed_test - club.not_taken_course

/-- Theorem stating the correct number of members who have taken the course but not passed -/
theorem correct_members_taken_course_not_passed (club : SwimClub) 
    (h1 : club.total_members = 100)
    (h2 : club.passed_test = 30)
    (h3 : club.not_taken_course = 30) : 
  members_taken_course_not_passed club = 40 := by
  sorry

#eval members_taken_course_not_passed ⟨100, 30, 30⟩

end NUMINAMATH_CALUDE_correct_members_taken_course_not_passed_l414_41468


namespace NUMINAMATH_CALUDE_curve_points_difference_l414_41432

theorem curve_points_difference (e a b : ℝ) : 
  e > 0 →
  (a^2 + e^2 = 2*e*a + 1) →
  (b^2 + e^2 = 2*e*b + 1) →
  a ≠ b →
  |a - b| = 2 := by
sorry

end NUMINAMATH_CALUDE_curve_points_difference_l414_41432


namespace NUMINAMATH_CALUDE_ellipse_constant_product_l414_41416

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def l (k m x y : ℝ) : Prop := y = k*x + m

-- Define the intersection point Q
def Q (k m : ℝ) : ℝ × ℝ := (-4, k*(-4) + m)

-- Define the left focus F
def F : ℝ × ℝ := (-1, 0)

-- State the theorem
theorem ellipse_constant_product (k m : ℝ) (A B P : ℝ × ℝ) :
  E A.1 A.2 →
  E B.1 B.2 →
  E P.1 P.2 →
  l k m A.1 A.2 →
  l k m B.1 B.2 →
  P = (A.1 + B.1, A.2 + B.2) →
  (P.1 - F.1) * (Q k m).1 + (P.2 - F.2) * (Q k m).2 = 3/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_constant_product_l414_41416


namespace NUMINAMATH_CALUDE_jake_brought_one_balloon_l414_41460

/-- The number of balloons Allan and Jake brought to the park in total -/
def total_balloons : ℕ := 3

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 2

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := total_balloons - allan_balloons

/-- Theorem stating that Jake brought 1 balloon to the park -/
theorem jake_brought_one_balloon : jake_balloons = 1 := by
  sorry

end NUMINAMATH_CALUDE_jake_brought_one_balloon_l414_41460


namespace NUMINAMATH_CALUDE_limit_at_negative_four_l414_41412

/-- The limit of (2x^2 + 6x - 8)/(x + 4) as x approaches -4 is -10 -/
theorem limit_at_negative_four :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x + 4| ∧ |x + 4| < δ →
    |(2*x^2 + 6*x - 8)/(x + 4) + 10| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_at_negative_four_l414_41412


namespace NUMINAMATH_CALUDE_sichuan_peppercorn_transport_l414_41499

/-- Represents the capacity of a truck type -/
structure TruckCapacity where
  a : ℕ
  b : ℕ
  h : a = b + 20

/-- Represents the number of trucks needed for each type -/
structure TruckCount where
  a : ℕ
  b : ℕ

theorem sichuan_peppercorn_transport 
  (cap : TruckCapacity) 
  (h1 : 1000 / cap.a = 800 / cap.b)
  (count : TruckCount)
  (h2 : count.a + count.b = 18)
  (h3 : cap.a * count.a + cap.b * (count.b - 1) + 65 = 1625) :
  cap.a = 100 ∧ cap.b = 80 ∧ count.a = 10 ∧ count.b = 8 := by
  sorry

#check sichuan_peppercorn_transport

end NUMINAMATH_CALUDE_sichuan_peppercorn_transport_l414_41499


namespace NUMINAMATH_CALUDE_min_value_of_f_l414_41455

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

-- State the theorem
theorem min_value_of_f (a : ℝ) : 
  ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), f a x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l414_41455


namespace NUMINAMATH_CALUDE_remaining_candy_l414_41450

/-- Given a group of people who collected candy and ate some, calculate the remaining candy. -/
theorem remaining_candy (total_candy : ℕ) (num_people : ℕ) (candy_eaten_per_person : ℕ) :
  total_candy = 120 →
  num_people = 3 →
  candy_eaten_per_person = 6 →
  total_candy - (num_people * candy_eaten_per_person) = 102 := by
  sorry

end NUMINAMATH_CALUDE_remaining_candy_l414_41450


namespace NUMINAMATH_CALUDE_problem_statement_l414_41494

theorem problem_statement :
  (∀ x : ℝ, x ≠ 0 → x^2 + 1/x^2 ≥ 2) ∧
  (¬ ∃ x : ℝ, x^2 + 1/x^2 ≤ 2) ∧
  ((∃ x : ℝ, x^2 + 1/x^2 ≤ 2) ∨ (∀ x : ℝ, x ≠ 0 → x^2 + 1/x^2 > 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l414_41494


namespace NUMINAMATH_CALUDE_binomial_thirteen_eleven_times_two_l414_41440

theorem binomial_thirteen_eleven_times_two : 2 * (Nat.choose 13 11) = 156 := by
  sorry

end NUMINAMATH_CALUDE_binomial_thirteen_eleven_times_two_l414_41440


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l414_41498

theorem log_sum_equals_two :
  2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l414_41498


namespace NUMINAMATH_CALUDE_pens_in_drawer_l414_41422

/-- The number of pens in Maria's desk drawer -/
def total_pens (red_pens black_pens blue_pens : ℕ) : ℕ :=
  red_pens + black_pens + blue_pens

/-- Theorem stating the total number of pens in Maria's desk drawer -/
theorem pens_in_drawer : 
  let red_pens : ℕ := 8
  let black_pens : ℕ := red_pens + 10
  let blue_pens : ℕ := red_pens + 7
  total_pens red_pens black_pens blue_pens = 41 := by
  sorry

end NUMINAMATH_CALUDE_pens_in_drawer_l414_41422


namespace NUMINAMATH_CALUDE_function_equality_condition_l414_41400

theorem function_equality_condition (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x
  ({x : ℝ | f x = 0} = {x : ℝ | f (f x) = 0} ∧ {x : ℝ | f x = 0}.Nonempty) ↔ 0 ≤ a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_condition_l414_41400


namespace NUMINAMATH_CALUDE_digit_sum_of_special_number_l414_41415

theorem digit_sum_of_special_number (N : ℕ) : 
  100 ≤ N ∧ N < 1000 ∧ 
  N % 10 = 7 ∧ N % 11 = 7 ∧ N % 12 = 7 → 
  (N / 100 + (N / 10) % 10 + N % 10) = 19 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_of_special_number_l414_41415


namespace NUMINAMATH_CALUDE_sin_cos_product_positive_implies_quadrant_I_or_III_l414_41484

def is_in_quadrant_I_or_III (θ : Real) : Prop :=
  (0 < θ ∧ θ < Real.pi / 2) ∨ (Real.pi < θ ∧ θ < 3 * Real.pi / 2)

theorem sin_cos_product_positive_implies_quadrant_I_or_III (θ : Real) :
  Real.sin θ * Real.cos θ > 0 → is_in_quadrant_I_or_III θ :=
by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_positive_implies_quadrant_I_or_III_l414_41484


namespace NUMINAMATH_CALUDE_balanced_numbers_count_l414_41451

/-- A four-digit number abcd is balanced if a + b = c + d -/
def is_balanced (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a + b = c + d

/-- Count of balanced four-digit numbers with sum 8 -/
def balanced_sum_8_count : ℕ := 72

/-- Count of balanced four-digit numbers with sum 16 -/
def balanced_sum_16_count : ℕ := 9

/-- Total count of balanced four-digit numbers -/
def total_balanced_count : ℕ := 615

/-- Theorem stating the counts of balanced numbers -/
theorem balanced_numbers_count :
  (balanced_sum_8_count = 72) ∧
  (balanced_sum_16_count = 9) ∧
  (total_balanced_count = 615) :=
sorry

end NUMINAMATH_CALUDE_balanced_numbers_count_l414_41451


namespace NUMINAMATH_CALUDE_correct_third_grade_sample_l414_41417

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  third_grade_students : ℕ
  sample_size : ℕ

/-- Calculates the number of students to be sampled from the third grade -/
def third_grade_sample (s : School) : ℕ :=
  s.sample_size * s.third_grade_students / s.total_students

/-- Theorem stating the correct number of third grade students in the sample -/
theorem correct_third_grade_sample (s : School) 
  (h1 : s.total_students = 1600)
  (h2 : s.third_grade_students = 400)
  (h3 : s.sample_size = 80) :
  third_grade_sample s = 20 := by
  sorry

#eval third_grade_sample ⟨1600, 400, 80⟩

end NUMINAMATH_CALUDE_correct_third_grade_sample_l414_41417


namespace NUMINAMATH_CALUDE_infinite_chessboard_rightlines_l414_41437

-- Define a rightline as a sequence of natural numbers
def Rightline := ℕ → ℕ

-- A rightline without multiples of 3
def NoMultiplesOfThree (r : Rightline) : Prop :=
  ∀ n : ℕ, r n % 3 ≠ 0

-- Pairwise disjoint rightlines
def PairwiseDisjoint (rs : ℕ → Rightline) : Prop :=
  ∀ i j : ℕ, i ≠ j → (∀ n : ℕ, rs i n ≠ rs j n)

theorem infinite_chessboard_rightlines :
  (∃ r : Rightline, NoMultiplesOfThree r) ∧
  (∃ rs : ℕ → Rightline, PairwiseDisjoint rs ∧ (∀ i : ℕ, NoMultiplesOfThree (rs i))) :=
sorry

end NUMINAMATH_CALUDE_infinite_chessboard_rightlines_l414_41437


namespace NUMINAMATH_CALUDE_total_legs_farmer_brown_l414_41465

/-- The number of legs for a chicken -/
def chicken_legs : ℕ := 2

/-- The number of legs for a sheep -/
def sheep_legs : ℕ := 4

/-- The number of chickens Farmer Brown fed -/
def num_chickens : ℕ := 7

/-- The number of sheep Farmer Brown fed -/
def num_sheep : ℕ := 5

/-- Theorem stating the total number of legs among the animals Farmer Brown fed -/
theorem total_legs_farmer_brown : 
  num_chickens * chicken_legs + num_sheep * sheep_legs = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_farmer_brown_l414_41465


namespace NUMINAMATH_CALUDE_payment_of_A_l414_41456

theorem payment_of_A (a b c : ℕ) : 
  a + b = 67 → b + c = 64 → a + c = 63 → a = 33 := by
  sorry

end NUMINAMATH_CALUDE_payment_of_A_l414_41456


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l414_41462

theorem subtraction_of_fractions : (8 : ℚ) / 19 - (5 : ℚ) / 57 = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l414_41462


namespace NUMINAMATH_CALUDE_mario_blossoms_l414_41445

/-- The number of hibiscus plants Mario has -/
def num_plants : ℕ := 3

/-- The number of flowers on the first hibiscus plant -/
def flowers_first : ℕ := 2

/-- The number of flowers on the second hibiscus plant -/
def flowers_second : ℕ := 2 * flowers_first

/-- The number of flowers on the third hibiscus plant -/
def flowers_third : ℕ := 4 * flowers_second

/-- The total number of blossoms Mario has -/
def total_blossoms : ℕ := flowers_first + flowers_second + flowers_third

theorem mario_blossoms : total_blossoms = 22 := by
  sorry

end NUMINAMATH_CALUDE_mario_blossoms_l414_41445


namespace NUMINAMATH_CALUDE_smallest_y_abs_eq_l414_41420

theorem smallest_y_abs_eq (y : ℝ) : (|2 * y + 6| = 18) → (∃ (z : ℝ), |2 * z + 6| = 18 ∧ z ≤ y) → y = -12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_abs_eq_l414_41420


namespace NUMINAMATH_CALUDE_urn_theorem_l414_41428

/-- Represents the state of the two urns -/
structure UrnState where
  urn1 : ℕ
  urn2 : ℕ

/-- Represents the transfer rule between urns -/
def transfer (state : UrnState) : UrnState :=
  if state.urn1 % 2 = 0 then
    UrnState.mk (state.urn1 / 2) (state.urn2 + state.urn1 / 2)
  else if state.urn2 % 2 = 0 then
    UrnState.mk (state.urn1 + state.urn2 / 2) (state.urn2 / 2)
  else
    state

theorem urn_theorem (p k : ℕ) (h1 : Prime p) (h2 : Prime (2 * p + 1)) (h3 : k < 2 * p + 1) :
  ∃ (n : ℕ) (state : UrnState),
    state.urn1 + state.urn2 = 2 * p + 1 ∧
    (transfer^[n] state).urn1 = k ∨ (transfer^[n] state).urn2 = k :=
  sorry

end NUMINAMATH_CALUDE_urn_theorem_l414_41428


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l414_41461

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > 2 → 1/x < 1/2) ∧
  (∃ x, 1/x < 1/2 ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l414_41461


namespace NUMINAMATH_CALUDE_exam_day_percentage_l414_41444

/-- Represents the percentage of students who took the exam on the assigned day -/
def assigned_day_percentage : ℝ := 70

/-- Represents the total number of students in the class -/
def total_students : ℕ := 100

/-- Represents the average score of students who took the exam on the assigned day -/
def assigned_day_score : ℝ := 60

/-- Represents the average score of students who took the exam on the make-up date -/
def makeup_day_score : ℝ := 80

/-- Represents the average score for the entire class -/
def class_average_score : ℝ := 66

theorem exam_day_percentage :
  assigned_day_percentage * assigned_day_score / 100 +
  (100 - assigned_day_percentage) * makeup_day_score / 100 =
  class_average_score :=
sorry

end NUMINAMATH_CALUDE_exam_day_percentage_l414_41444


namespace NUMINAMATH_CALUDE_mcdonalds_coupon_value_l414_41424

/-- Proves that given an original cost of $7.50, a senior citizen discount of 20%,
    and a final payment of $4, the coupon value that makes this possible is $2.50. -/
theorem mcdonalds_coupon_value :
  let original_cost : ℝ := 7.50
  let senior_discount : ℝ := 0.20
  let final_payment : ℝ := 4.00
  let coupon_value : ℝ := 2.50
  (1 - senior_discount) * (original_cost - coupon_value) = final_payment := by
sorry

end NUMINAMATH_CALUDE_mcdonalds_coupon_value_l414_41424


namespace NUMINAMATH_CALUDE_basic_astrophysics_degrees_l414_41476

/-- Represents the allocation of a budget in a circle graph --/
def BudgetAllocation (total : ℝ) (allocated : ℝ) (degreesPerPercent : ℝ) : Prop :=
  total = 100 ∧ 
  allocated = 95 ∧ 
  degreesPerPercent = 360 / 100

/-- Theorem: The number of degrees representing the remaining budget (basic astrophysics) is 18 --/
theorem basic_astrophysics_degrees 
  (total allocated remaining : ℝ) 
  (degreesPerPercent : ℝ) 
  (h : BudgetAllocation total allocated degreesPerPercent) :
  remaining = 18 :=
sorry

end NUMINAMATH_CALUDE_basic_astrophysics_degrees_l414_41476


namespace NUMINAMATH_CALUDE_digits_of_product_l414_41491

theorem digits_of_product : ∃ (n : ℕ), n = 3^4 * 6^8 ∧ (Nat.log 10 n + 1 = 9) := by sorry

end NUMINAMATH_CALUDE_digits_of_product_l414_41491


namespace NUMINAMATH_CALUDE_series_sum_l414_41446

/-- The sum of the infinite series Σ(n=1 to ∞) of n/(3^n) equals 9/4 -/
theorem series_sum : ∑' n : ℕ, (n : ℝ) / (3 : ℝ) ^ n = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l414_41446


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l414_41435

/-- The eccentricity of an ellipse with equation x²/16 + y²/12 = 1 is 1/2 -/
theorem ellipse_eccentricity : ∃ e : ℝ,
  (∀ x y : ℝ, x^2/16 + y^2/12 = 1 → 
    ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧
      x^2/a^2 + y^2/b^2 = 1 ∧
      c^2 = a^2 - b^2 ∧
      e = c/a) ∧
  e = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l414_41435


namespace NUMINAMATH_CALUDE_inequality_equivalence_l414_41413

theorem inequality_equivalence (x y : ℝ) : 
  y - x < Real.sqrt (4 * x^2) ↔ (x ≥ 0 ∧ y < 3 * x) ∨ (x < 0 ∧ y < -x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l414_41413


namespace NUMINAMATH_CALUDE_reciprocal_power_2006_l414_41436

theorem reciprocal_power_2006 (a : ℚ) : 
  (a ≠ 0 ∧ a = 1 / a) → a^2006 = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_power_2006_l414_41436


namespace NUMINAMATH_CALUDE_team_b_average_points_l414_41439

theorem team_b_average_points (average_first_two : ℝ) : 
  (2 * average_first_two + 47 + 330 > 500) → average_first_two > 61.5 := by
  sorry

end NUMINAMATH_CALUDE_team_b_average_points_l414_41439


namespace NUMINAMATH_CALUDE_coral_population_decline_l414_41466

/-- The yearly decrease rate of the coral population -/
def decrease_rate : ℝ := 0.25

/-- The threshold below which we consider the population critically low -/
def critical_threshold : ℝ := 0.05

/-- The number of years it takes for the population to fall below the critical threshold -/
def years_to_critical : ℕ := 9

/-- The remaining population after n years -/
def population_after (n : ℕ) : ℝ := (1 - decrease_rate) ^ n

theorem coral_population_decline :
  population_after years_to_critical < critical_threshold :=
sorry

end NUMINAMATH_CALUDE_coral_population_decline_l414_41466


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l414_41407

/-- The perimeter of a semi-circle with radius 6.83 cm is approximately 35.12 cm. -/
theorem semicircle_perimeter_approx : 
  let r : ℝ := 6.83
  let π : ℝ := Real.pi
  let perimeter : ℝ := π * r + 2 * r
  ∃ ε > 0, abs (perimeter - 35.12) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l414_41407


namespace NUMINAMATH_CALUDE_binary_1101_equals_13_l414_41421

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- The binary representation of 1101₂ -/
def binary_1101 : List Bool := [true, true, false, true]

theorem binary_1101_equals_13 : binary_to_decimal binary_1101 = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_equals_13_l414_41421


namespace NUMINAMATH_CALUDE_base_9_minus_b_multiple_of_7_l414_41447

def base_9_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9^i)) 0

def is_multiple_of (a b : Int) : Prop :=
  ∃ k : Int, a = b * k

theorem base_9_minus_b_multiple_of_7 (b : Int) :
  (0 ≤ b) →
  (b ≤ 9) →
  (is_multiple_of (base_9_to_decimal [2, 7, 6, 4, 5, 1, 3] - b) 7) →
  b = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_9_minus_b_multiple_of_7_l414_41447


namespace NUMINAMATH_CALUDE_factor_of_valid_Z_l414_41497

def is_valid_Z (n : ℕ) : Prop :=
  10000000 ≤ n ∧ n < 100000000 ∧
  ∃ (a b c d : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    n = 10000000 * a + 1000000 * b + 100000 * c + 10000 * d +
        1000 * a + 100 * b + 10 * c + d

theorem factor_of_valid_Z (Z : ℕ) (h : is_valid_Z Z) : 
  10001 ∣ Z :=
sorry

end NUMINAMATH_CALUDE_factor_of_valid_Z_l414_41497


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l414_41464

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 34 →
  (1/2) * a * b = 24 →
  a^2 + b^2 = c^2 →
  c = 62/4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l414_41464


namespace NUMINAMATH_CALUDE_polynomial_relationship_l414_41493

def f (x : ℝ) : ℝ := x^2 + x

theorem polynomial_relationship : 
  (f 1 = 2) ∧ 
  (f 2 = 6) ∧ 
  (f 3 = 12) ∧ 
  (f 4 = 20) ∧ 
  (f 5 = 30) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_relationship_l414_41493


namespace NUMINAMATH_CALUDE_midpoint_quadrilateral_area_l414_41441

/-- The area of the quadrilateral formed by connecting the midpoints of a rectangle -/
theorem midpoint_quadrilateral_area (w l : ℝ) (hw : w = 10) (hl : l = 14) :
  let midpoint_quad_area := (w / 2) * (l / 2)
  midpoint_quad_area = 35 := by sorry

end NUMINAMATH_CALUDE_midpoint_quadrilateral_area_l414_41441


namespace NUMINAMATH_CALUDE_logarithm_sum_equality_l414_41411

-- Define the logarithm function (base 2)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem logarithm_sum_equality (a b : ℝ) 
  (h1 : a > 1) (h2 : b > 1) (h3 : lg (a + b) = lg a + lg b) : 
  lg (a - 1) + lg (b - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_equality_l414_41411


namespace NUMINAMATH_CALUDE_fraction_enlargement_l414_41483

theorem fraction_enlargement (x y : ℝ) (h : x + y ≠ 0) :
  (2 * (3 * x) * (3 * y)) / ((3 * x) + (3 * y)) = 3 * ((2 * x * y) / (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_enlargement_l414_41483


namespace NUMINAMATH_CALUDE_harvest_season_duration_l414_41433

theorem harvest_season_duration (regular_earnings overtime_earnings total_earnings : ℕ) 
  (h1 : regular_earnings = 28)
  (h2 : overtime_earnings = 939)
  (h3 : total_earnings = 1054997) : 
  total_earnings / (regular_earnings + overtime_earnings) = 1091 := by
  sorry

end NUMINAMATH_CALUDE_harvest_season_duration_l414_41433


namespace NUMINAMATH_CALUDE_common_difference_is_two_l414_41458

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a₁ : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The sum of the 4th and 6th terms is 10 -/
  sum_4_6 : a₁ + 3*d + (a₁ + 5*d) = 10
  /-- The sum of the first 5 terms is 5 -/
  sum_5 : 5*a₁ + 10*d = 5

/-- The common difference of the arithmetic sequence is 2 -/
theorem common_difference_is_two (seq : ArithmeticSequence) : seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l414_41458


namespace NUMINAMATH_CALUDE_bank_transfer_problem_l414_41459

theorem bank_transfer_problem (X : ℝ) :
  (0.8 * X = 30000) → X = 37500 := by
  sorry

end NUMINAMATH_CALUDE_bank_transfer_problem_l414_41459


namespace NUMINAMATH_CALUDE_fifth_odd_multiple_of_five_under_hundred_fifth_odd_multiple_of_five_under_hundred_proof_l414_41410

theorem fifth_odd_multiple_of_five_under_hundred : ℕ → Prop :=
  fun n =>
    (∃ k, n = 5 * (2 * k + 1)) ∧  -- n is odd and a multiple of 5
    n < 100 ∧  -- n is less than 100
    (∃ m, m = 5 ∧  -- m is the count of numbers satisfying the conditions
      ∀ i, i < n →
        (∃ j, i = 5 * (2 * j + 1)) ∧ i < 100 →
        i ≤ m * 9) →  -- there are exactly 4 numbers before n satisfying the conditions
    n = 45  -- the fifth such number is 45

-- The proof of this theorem is omitted
theorem fifth_odd_multiple_of_five_under_hundred_proof : fifth_odd_multiple_of_five_under_hundred 45 := by
  sorry

end NUMINAMATH_CALUDE_fifth_odd_multiple_of_five_under_hundred_fifth_odd_multiple_of_five_under_hundred_proof_l414_41410


namespace NUMINAMATH_CALUDE_inequality_proof_l414_41434

theorem inequality_proof (a b x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : x / a < y / b) :
  (1 / 2) * (x / a + y / b) > (x + y) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l414_41434


namespace NUMINAMATH_CALUDE_customers_left_second_time_l414_41486

theorem customers_left_second_time 
  (initial_customers : ℝ)
  (first_group_left : ℝ)
  (final_customers : ℝ)
  (h1 : initial_customers = 36.0)
  (h2 : first_group_left = 19.0)
  (h3 : final_customers = 3) :
  initial_customers - first_group_left - final_customers = 14.0 :=
by sorry

end NUMINAMATH_CALUDE_customers_left_second_time_l414_41486


namespace NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l414_41425

theorem shortest_side_of_right_triangle (a b c : ℝ) :
  a = 5 →
  b = 12 →
  c^2 = a^2 + b^2 →
  c ≥ a ∧ c ≥ b →
  a = min a b := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l414_41425


namespace NUMINAMATH_CALUDE_sum_of_squares_divisible_by_seven_l414_41485

theorem sum_of_squares_divisible_by_seven (a b : ℤ) (h : 7 ∣ (a^2 + b^2)) : 7 ∣ a ∧ 7 ∣ b := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisible_by_seven_l414_41485


namespace NUMINAMATH_CALUDE_unique_negative_zero_implies_a_gt_two_l414_41478

/-- The function f(x) = ax³ - 3x² + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The unique zero point of f(x) -/
noncomputable def x₀ (a : ℝ) : ℝ := sorry

theorem unique_negative_zero_implies_a_gt_two (a : ℝ) :
  (∃! x, f a x = 0) ∧ (x₀ a < 0) → a > 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_negative_zero_implies_a_gt_two_l414_41478


namespace NUMINAMATH_CALUDE_girl_pairs_in_circular_arrangement_l414_41496

/-- 
Given a circular arrangement of boys and girls:
- n_boys: number of boys
- n_girls: number of girls
- boy_pairs: number of pairs of boys sitting next to each other
- girl_pairs: number of pairs of girls sitting next to each other
-/
def circular_arrangement (n_boys n_girls boy_pairs girl_pairs : ℕ) : Prop :=
  n_boys + n_girls > 0 ∧ boy_pairs ≤ n_boys ∧ girl_pairs ≤ n_girls

theorem girl_pairs_in_circular_arrangement 
  (n_boys n_girls boy_pairs girl_pairs : ℕ) 
  (h_arrangement : circular_arrangement n_boys n_girls boy_pairs girl_pairs)
  (h_boys : n_boys = 10)
  (h_girls : n_girls = 15)
  (h_boy_pairs : boy_pairs = 5) :
  girl_pairs = 10 := by
  sorry

end NUMINAMATH_CALUDE_girl_pairs_in_circular_arrangement_l414_41496


namespace NUMINAMATH_CALUDE_max_value_of_a_l414_41443

theorem max_value_of_a (a b c : ℝ) (h1 : a + b + c = 7) (h2 : a * b + a * c + b * c = 12) :
  a ≤ (7 + Real.sqrt 46) / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l414_41443


namespace NUMINAMATH_CALUDE_inequality_solution_l414_41405

-- Define the function f
def f (x : ℝ) := x^2 - x - 6

-- State the theorem
theorem inequality_solution :
  (∀ x : ℝ, f x = 0 ↔ x = -2 ∨ x = 3) →
  (∀ x : ℝ, -6 * f (-x) > 0 ↔ -3 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l414_41405


namespace NUMINAMATH_CALUDE_max_eggs_l414_41454

theorem max_eggs (x : ℕ) : 
  x < 200 ∧ 
  x % 3 = 2 ∧ 
  x % 4 = 3 ∧ 
  x % 5 = 4 ∧
  (∀ y : ℕ, y < 200 ∧ y % 3 = 2 ∧ y % 4 = 3 ∧ y % 5 = 4 → y ≤ x) →
  x = 179 :=
by sorry

end NUMINAMATH_CALUDE_max_eggs_l414_41454


namespace NUMINAMATH_CALUDE_range_of_a_for_subset_l414_41482

-- Define the sets A and B
def A (a : ℝ) := {x : ℝ | 3 + a ≤ x ∧ x ≤ 4 + 3*a}
def B := {x : ℝ | (x + 4) / (5 - x) ≥ 0 ∧ x ≠ 5}

-- State the theorem
theorem range_of_a_for_subset : 
  {a : ℝ | ∀ x, x ∈ A a → x ∈ B} = {a : ℝ | -1/2 ≤ a ∧ a < 1/3} := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_subset_l414_41482
