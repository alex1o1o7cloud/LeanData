import Mathlib

namespace percentage_of_absent_students_l680_68043

theorem percentage_of_absent_students (total : ℕ) (present : ℕ) : 
  total = 50 → present = 43 → (((total - present : ℚ) / total) * 100 = 14) := by sorry

end percentage_of_absent_students_l680_68043


namespace min_value_geometric_sequence_l680_68063

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem min_value_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_pos : ∀ n, a n > 0)
  (h_mean : Real.sqrt (a 4 * a 14) = 2 * Real.sqrt 2) :
  (2 * a 7 + a 11 ≥ 8) ∧ ∃ x, 2 * x + (a 11) = 8 := by
  sorry

end min_value_geometric_sequence_l680_68063


namespace escalator_walking_rate_l680_68047

/-- Proves that given an escalator moving upward at 10 ft/sec with a length of 112 feet,
    if a person takes 8 seconds to cover the entire length,
    then the person's walking rate on the escalator is 4 ft/sec. -/
theorem escalator_walking_rate
  (escalator_speed : ℝ)
  (escalator_length : ℝ)
  (time_taken : ℝ)
  (h1 : escalator_speed = 10)
  (h2 : escalator_length = 112)
  (h3 : time_taken = 8)
  : ∃ (walking_rate : ℝ),
    walking_rate = 4 ∧
    escalator_length = (walking_rate + escalator_speed) * time_taken :=
by sorry

end escalator_walking_rate_l680_68047


namespace polynomial_division_remainder_l680_68099

/-- The remainder when x^3 + 3 is divided by x^2 + 2 is -2x + 3 -/
theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, x^3 + 3 = (x^2 + 2) * q + (-2*x + 3) := by
  sorry

end polynomial_division_remainder_l680_68099


namespace unique_solution_l680_68045

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem unique_solution :
  ∀ k : ℕ, (factorial (k / 2) * (k / 4) = 2016 + k^2) ↔ k = 12 :=
by sorry

end unique_solution_l680_68045


namespace sphere_cube_volume_constant_l680_68088

/-- The value of K when a sphere has the same surface area as a cube with side length 3
    and its volume is expressed as (K * sqrt(6)) / sqrt(π) -/
theorem sphere_cube_volume_constant (cube_side : ℝ) (sphere_volume : ℝ → ℝ) : 
  cube_side = 3 →
  (4 * π * (sphere_volume K / ((4 / 3) * π))^(2/3) = 6 * cube_side^2) →
  sphere_volume K = K * Real.sqrt 6 / Real.sqrt π →
  K = 27 * Real.sqrt 6 / Real.sqrt 2 := by
sorry

end sphere_cube_volume_constant_l680_68088


namespace x_minus_y_equals_eight_l680_68006

theorem x_minus_y_equals_eight (x y : ℝ) 
  (hx : 3 = 0.15 * x) 
  (hy : 3 = 0.25 * y) : 
  x - y = 8 := by
sorry

end x_minus_y_equals_eight_l680_68006


namespace other_jelly_correct_l680_68052

/-- Given a total amount of jelly and the amount of one type, 
    calculate the amount of the other type -/
def other_jelly_amount (total : ℕ) (one_type : ℕ) : ℕ :=
  total - one_type

/-- Theorem: The amount of the other type of jelly is the difference
    between the total amount and the amount of one type -/
theorem other_jelly_correct (total : ℕ) (one_type : ℕ) 
  (h : one_type ≤ total) : 
  other_jelly_amount total one_type = total - one_type :=
by
  sorry

#eval other_jelly_amount 6310 4518

end other_jelly_correct_l680_68052


namespace quadratic_vertex_x_coordinate_l680_68092

/-- Given a quadratic function f(x) = ax^2 + bx + c that passes through
    the points (2, 5), (8, 5), and (9, 11), prove that the x-coordinate
    of its vertex is 5. -/
theorem quadratic_vertex_x_coordinate
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^2 + b * x + c)
  (h_point1 : f 2 = 5)
  (h_point2 : f 8 = 5)
  (h_point3 : f 9 = 11) :
  ∃ (vertex_x : ℝ), vertex_x = 5 ∧ ∀ x, f x ≤ f vertex_x :=
sorry

end quadratic_vertex_x_coordinate_l680_68092


namespace expression_value_l680_68021

theorem expression_value (x y : ℝ) : 
  x^4 + 6*x^2*y + 9*y^2 + 2*x^2 + 6*y + 4 = 7 → 
  x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = -2 ∨ 
  x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = 14 :=
by sorry

end expression_value_l680_68021


namespace cos_sum_square_75_15_l680_68087

theorem cos_sum_square_75_15 :
  Real.cos (75 * π / 180) ^ 2 + Real.cos (15 * π / 180) ^ 2 + 
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) = 5 / 4 := by
  sorry

end cos_sum_square_75_15_l680_68087


namespace power_subtraction_l680_68028

theorem power_subtraction (x a b : ℝ) (ha : x^a = 3) (hb : x^b = 5) : x^(a - b) = 3/5 := by
  sorry

end power_subtraction_l680_68028


namespace club_members_theorem_l680_68062

theorem club_members_theorem (total : ℕ) (left_handed : ℕ) (rock_fans : ℕ) (right_handed_non_fans : ℕ) 
  (h1 : total = 25)
  (h2 : left_handed = 10)
  (h3 : rock_fans = 18)
  (h4 : right_handed_non_fans = 3)
  (h5 : left_handed + (total - left_handed) = total) :
  ∃ (left_handed_rock_fans : ℕ),
    left_handed_rock_fans = 6 ∧
    left_handed_rock_fans ≤ left_handed ∧
    left_handed_rock_fans ≤ rock_fans ∧
    left_handed_rock_fans + (left_handed - left_handed_rock_fans) + 
    (rock_fans - left_handed_rock_fans) + right_handed_non_fans = total :=
by
  sorry

end club_members_theorem_l680_68062


namespace corrected_mean_calculation_l680_68001

/-- Calculates the corrected mean of a set of observations after fixing recording errors -/
theorem corrected_mean_calculation (n : ℕ) (original_mean : ℝ) 
  (error1_recorded error1_actual : ℝ)
  (error2_recorded error2_actual : ℝ)
  (error3_recorded error3_actual : ℝ)
  (h1 : n = 50)
  (h2 : original_mean = 41)
  (h3 : error1_recorded = 23 ∧ error1_actual = 48)
  (h4 : error2_recorded = 42 ∧ error2_actual = 36)
  (h5 : error3_recorded = 28 ∧ error3_actual = 55) :
  let corrected_sum := n * original_mean + 
    (error1_actual - error1_recorded) + 
    (error2_actual - error2_recorded) + 
    (error3_actual - error3_recorded)
  (corrected_sum / n) = 41.92 := by
  sorry

end corrected_mean_calculation_l680_68001


namespace parametric_to_cartesian_l680_68048

/-- Given parametric equations x = 1 + 2cosθ and y = 2sinθ, 
    prove they are equivalent to the Cartesian equation (x-1)² + y² = 4 -/
theorem parametric_to_cartesian :
  ∀ (x y θ : ℝ), 
  x = 1 + 2 * Real.cos θ ∧ 
  y = 2 * Real.sin θ → 
  (x - 1)^2 + y^2 = 4 := by
sorry

end parametric_to_cartesian_l680_68048


namespace pi_approximation_proof_l680_68082

theorem pi_approximation_proof :
  let π := 4 * Real.sin (52 * π / 180)
  (2 * π * Real.sqrt (16 - π^2) - 8 * Real.sin (44 * π / 180)) /
  (Real.sqrt 3 - 2 * Real.sqrt 3 * Real.sin (22 * π / 180)^2) = 8 * Real.sqrt 3 := by
  sorry

end pi_approximation_proof_l680_68082


namespace game_winning_probability_l680_68016

/-- A game with consecutive integers from 2 to 2020 -/
def game_range : Set ℕ := {n | 2 ≤ n ∧ n ≤ 2020}

/-- The total number of integers in the game -/
def total_numbers : ℕ := 2019

/-- Two numbers are coprime if their greatest common divisor is 1 -/
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The number of coprime pairs in the game range -/
def coprime_pairs : ℕ := 1010

/-- The probability of winning is the number of coprime pairs divided by the total numbers -/
theorem game_winning_probability :
  (coprime_pairs : ℚ) / total_numbers = 1010 / 2019 := by sorry

end game_winning_probability_l680_68016


namespace rhombus_perimeter_l680_68040

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := by
  sorry

end rhombus_perimeter_l680_68040


namespace negative_cube_squared_l680_68013

theorem negative_cube_squared (x : ℝ) : (-x^3)^2 = x^6 := by
  sorry

end negative_cube_squared_l680_68013


namespace olyas_numbers_proof_l680_68009

def first_number : ℕ := 929
def second_number : ℕ := 20
def third_number : ℕ := 2

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem olyas_numbers_proof :
  (100 ≤ first_number ∧ first_number < 1000) ∧
  (second_number = sum_of_digits first_number) ∧
  (third_number = sum_of_digits second_number) ∧
  (∃ (a b : ℕ), first_number = 100 * a + 10 * b + a ∧
                second_number = 10 * b + 0 ∧
                third_number = b) :=
by sorry

end olyas_numbers_proof_l680_68009


namespace arithmetic_progression_common_difference_l680_68066

/-- Proves that in an arithmetic progression with first term 2, last term 62, and 31 terms, the common difference is 2. -/
theorem arithmetic_progression_common_difference 
  (first_term : ℕ) 
  (last_term : ℕ) 
  (num_terms : ℕ) 
  (h1 : first_term = 2) 
  (h2 : last_term = 62) 
  (h3 : num_terms = 31) : 
  (last_term - first_term) / (num_terms - 1) = 2 := by
  sorry

end arithmetic_progression_common_difference_l680_68066


namespace crafts_club_members_crafts_club_members_proof_l680_68079

theorem crafts_club_members : ℕ → Prop :=
  fun n =>
    let necklaces_per_member : ℕ := 2
    let beads_per_necklace : ℕ := 50
    let total_beads : ℕ := 900
    n * (necklaces_per_member * beads_per_necklace) = total_beads →
    n = 9

-- Proof
theorem crafts_club_members_proof : crafts_club_members 9 := by
  sorry

end crafts_club_members_crafts_club_members_proof_l680_68079


namespace championship_games_l680_68027

theorem championship_games (n : ℕ) (n_ge_2 : n ≥ 2) : 
  (n * (n - 1)) / 2 = (Finset.sum (Finset.range (n - 1)) (λ i => n - 1 - i)) :=
by sorry

end championship_games_l680_68027


namespace max_tickets_jane_can_buy_l680_68011

theorem max_tickets_jane_can_buy (ticket_price : ℚ) (budget : ℚ) : 
  ticket_price = 27/2 → budget = 100 → 
  (∃ n : ℕ, n * ticket_price ≤ budget ∧ 
    ∀ m : ℕ, m * ticket_price ≤ budget → m ≤ n) → 
  (∃ n : ℕ, n * ticket_price ≤ budget ∧ 
    ∀ m : ℕ, m * ticket_price ≤ budget → m ≤ n) ∧ n = 7 :=
by
  sorry

end max_tickets_jane_can_buy_l680_68011


namespace triangle_square_side_ratio_l680_68010

theorem triangle_square_side_ratio (perimeter : ℝ) (triangle_side square_side : ℝ) : 
  perimeter > 0 → 
  triangle_side * 3 = perimeter → 
  square_side * 4 = perimeter → 
  triangle_side / square_side = 4 / 3 :=
by
  sorry

end triangle_square_side_ratio_l680_68010


namespace sin_pi_sixth_minus_two_alpha_l680_68025

theorem sin_pi_sixth_minus_two_alpha (α : ℝ) 
  (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.sin (π / 6 - 2 * α) = -7 / 9 := by
  sorry

end sin_pi_sixth_minus_two_alpha_l680_68025


namespace car_repair_cost_john_car_repair_cost_l680_68022

/-- Calculates the amount spent on car repairs given savings information -/
theorem car_repair_cost (monthly_savings : ℕ) (savings_months : ℕ) (remaining_amount : ℕ) : ℕ :=
  let total_savings := monthly_savings * savings_months
  total_savings - remaining_amount

/-- Proves that John spent $400 on car repairs -/
theorem john_car_repair_cost : 
  car_repair_cost 25 24 200 = 400 := by
  sorry

end car_repair_cost_john_car_repair_cost_l680_68022


namespace existence_of_powers_of_seven_with_difference_divisible_by_2021_l680_68071

theorem existence_of_powers_of_seven_with_difference_divisible_by_2021 :
  ∃ (n m : ℕ), n > m ∧ (7^n - 7^m) % 2021 = 0 := by
  sorry

end existence_of_powers_of_seven_with_difference_divisible_by_2021_l680_68071


namespace bisection_method_structures_l680_68039

-- Define the function for which we're finding the root
def f (x : ℝ) := x^2 - 2

-- Define the bisection method structure
structure BisectionMethod where
  sequential : Bool
  conditional : Bool
  loop : Bool

-- Theorem statement
theorem bisection_method_structures :
  ∀ (ε : ℝ) (a b : ℝ), 
    ε > 0 → a < b → f a * f b < 0 →
    ∃ (m : BisectionMethod),
      m.sequential ∧ m.conditional ∧ m.loop ∧
      ∃ (x : ℝ), a ≤ x ∧ x ≤ b ∧ |f x| < ε :=
sorry

end bisection_method_structures_l680_68039


namespace no_ab_term_l680_68083

/-- The polynomial does not contain the term ab if and only if m = -2 -/
theorem no_ab_term (a b m : ℝ) : 
  2 * (a^2 + a*b - 5*b^2) - (a^2 - m*a*b + 2*b^2) = a^2 - 12*b^2 ↔ m = -2 :=
by sorry

end no_ab_term_l680_68083


namespace perimeter_ratio_is_one_l680_68046

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents the original paper --/
def original_paper : Rectangle := { width := 8, height := 12 }

/-- Represents one of the rectangles after folding and cutting --/
def folded_cut_rectangle : Rectangle := { width := 4, height := 6 }

/-- Theorem stating that the ratio of perimeters is 1 --/
theorem perimeter_ratio_is_one : 
  perimeter folded_cut_rectangle / perimeter folded_cut_rectangle = 1 := by sorry

end perimeter_ratio_is_one_l680_68046


namespace adam_chocolate_boxes_l680_68089

/-- The number of boxes of chocolate candy Adam bought -/
def chocolate_boxes : ℕ := sorry

/-- The number of boxes of caramel candy Adam bought -/
def caramel_boxes : ℕ := 5

/-- The number of pieces of candy in each box -/
def pieces_per_box : ℕ := 4

/-- The total number of candies Adam had -/
def total_candies : ℕ := 28

theorem adam_chocolate_boxes :
  chocolate_boxes = 2 :=
by sorry

end adam_chocolate_boxes_l680_68089


namespace square_root_pattern_l680_68029

theorem square_root_pattern (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h2 : Real.sqrt (2 + 2/3) = 2 * Real.sqrt (2/3))
  (h3 : Real.sqrt (3 + 3/8) = 3 * Real.sqrt (3/8))
  (h4 : Real.sqrt (4 + 4/15) = 4 * Real.sqrt (4/15))
  (h7 : Real.sqrt (7 + a/b) = 7 * Real.sqrt (a/b)) :
  a + b = 55 := by sorry

end square_root_pattern_l680_68029


namespace cosine_sine_equivalence_l680_68081

theorem cosine_sine_equivalence (θ : ℝ) : 
  Real.cos (3 * Real.pi / 2 - θ) = Real.sin (Real.pi + θ) ∧ 
  Real.cos (3 * Real.pi / 2 - θ) = Real.cos (Real.pi / 2 + θ) := by
  sorry

end cosine_sine_equivalence_l680_68081


namespace cos_inequality_solution_set_l680_68061

theorem cos_inequality_solution_set (x : ℝ) : 
  (Real.cos x + 1/2 ≤ 0) ↔ 
  (∃ k : ℤ, 2*k*Real.pi + 2*Real.pi/3 ≤ x ∧ x ≤ 2*k*Real.pi + 4*Real.pi/3) :=
by sorry

end cos_inequality_solution_set_l680_68061


namespace monkey_climb_theorem_l680_68093

/-- The height of the tree that the monkey climbs -/
def tree_height : ℕ := 20

/-- The height the monkey climbs in one hour during the first 17 hours -/
def hourly_climb : ℕ := 3

/-- The height the monkey slips back in one hour during the first 17 hours -/
def hourly_slip : ℕ := 2

/-- The number of hours it takes the monkey to reach the top of the tree -/
def total_hours : ℕ := 18

/-- The height the monkey climbs in the last hour -/
def final_climb : ℕ := 3

theorem monkey_climb_theorem :
  tree_height = (total_hours - 1) * (hourly_climb - hourly_slip) + final_climb :=
by sorry

end monkey_climb_theorem_l680_68093


namespace first_group_number_is_9_l680_68060

/-- Represents a systematic sampling method -/
structure SystematicSampling where
  population : ℕ
  sample_size : ℕ
  group_number : ℕ → ℕ
  h_population : population > 0
  h_sample_size : sample_size > 0
  h_sample_size_le_population : sample_size ≤ population

/-- The number drawn by the first group in a systematic sampling -/
def first_group_number (s : SystematicSampling) : ℕ :=
  s.group_number 1

/-- Theorem stating that the first group number is 9 given the problem conditions -/
theorem first_group_number_is_9 (s : SystematicSampling)
    (h_population : s.population = 960)
    (h_sample_size : s.sample_size = 32)
    (h_fifth_group : s.group_number 5 = 129) :
    first_group_number s = 9 := by
  sorry

end first_group_number_is_9_l680_68060


namespace min_value_theorem_l680_68096

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  2 * x + 18 / x ≥ 12 ∧ (2 * x + 18 / x = 12 ↔ x = 3) :=
by sorry

end min_value_theorem_l680_68096


namespace prime_count_inequality_l680_68015

/-- p_n denotes the nth prime number -/
def p (n : ℕ) : ℕ := sorry

/-- π(x) denotes the number of primes less than or equal to x -/
def π (x : ℝ) : ℕ := sorry

/-- The product of the first n primes -/
def primeProduct (n : ℕ) : ℕ := sorry

theorem prime_count_inequality (n : ℕ) (h : n ≥ 6) :
  π (Real.sqrt (primeProduct n : ℝ)) > 2 * n := by
  sorry

end prime_count_inequality_l680_68015


namespace quadratic_root_proof_l680_68036

theorem quadratic_root_proof (x : ℝ) : 
  x = (-31 - Real.sqrt 481) / 12 → 6 * x^2 + 31 * x + 20 = 0 := by
  sorry

end quadratic_root_proof_l680_68036


namespace tangent_line_at_one_l680_68030

/-- The function f(x) = x^4 - 2x^3 -/
def f (x : ℝ) : ℝ := x^4 - 2*x^3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4*x^3 - 6*x^2

theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -2*x + 1 :=
sorry

end tangent_line_at_one_l680_68030


namespace water_level_drop_l680_68050

/-- The drop in water level when removing a partially submerged spherical ball from a prism-shaped glass -/
theorem water_level_drop (a r h : ℝ) (ha : a > 0) (hr : r > 0) (hh : h > 0) (hhr : h < r) :
  let base_area := (3 * Real.sqrt 3 * a^2) / 2
  let submerged_height := r - h
  let submerged_volume := π * submerged_height^2 * (3*r - submerged_height) / 3
  submerged_volume / base_area = (6 * π * Real.sqrt 3) / 25 :=
by sorry

end water_level_drop_l680_68050


namespace cube_sum_and_reciprocal_l680_68078

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := by
  sorry

end cube_sum_and_reciprocal_l680_68078


namespace sin_cos_difference_21_81_l680_68007

theorem sin_cos_difference_21_81 :
  Real.sin (21 * π / 180) * Real.cos (81 * π / 180) -
  Real.cos (21 * π / 180) * Real.sin (81 * π / 180) =
  -Real.sqrt 3 / 2 := by
  sorry

end sin_cos_difference_21_81_l680_68007


namespace abs_neg_x_eq_five_implies_x_plus_minus_five_l680_68095

theorem abs_neg_x_eq_five_implies_x_plus_minus_five (x : ℝ) : 
  |(-x)| = 5 → x = -5 ∨ x = 5 := by
sorry

end abs_neg_x_eq_five_implies_x_plus_minus_five_l680_68095


namespace inscribed_cylinder_radius_l680_68059

/-- Represents a right circular cylinder inscribed in a right circular cone. -/
structure InscribedCylinder where
  /-- Radius of the inscribed cylinder -/
  radius : ℝ
  /-- Height of the inscribed cylinder -/
  height : ℝ
  /-- Diameter of the cone -/
  cone_diameter : ℝ
  /-- Altitude of the cone -/
  cone_altitude : ℝ
  /-- The cylinder's diameter is equal to its height -/
  cylinder_property : height = 2 * radius
  /-- The cone has a diameter of 20 -/
  cone_diameter_value : cone_diameter = 20
  /-- The cone has an altitude of 24 -/
  cone_altitude_value : cone_altitude = 24

/-- Theorem stating that the radius of the inscribed cylinder is 60/11 -/
theorem inscribed_cylinder_radius (c : InscribedCylinder) : c.radius = 60 / 11 := by
  sorry

end inscribed_cylinder_radius_l680_68059


namespace line_shift_theorem_l680_68041

/-- Represents a line in the 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shifts a line horizontally -/
def shift_horizontal (l : Line) (units : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept + l.slope * units }

/-- Shifts a line vertically -/
def shift_vertical (l : Line) (units : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept - units }

/-- The theorem stating that shifting the line y = 2x - 1 left by 3 units
    and then down by 4 units results in the line y = 2x + 1 -/
theorem line_shift_theorem :
  let initial_line := Line.mk 2 (-1)
  let shifted_left := shift_horizontal initial_line 3
  let final_line := shift_vertical shifted_left 4
  final_line = Line.mk 2 1 := by
  sorry


end line_shift_theorem_l680_68041


namespace parallel_vectors_x_coord_l680_68019

/-- Given vectors a and b in ℝ², if a + b is parallel to a - 2b, then the x-coordinate of b is 4. -/
theorem parallel_vectors_x_coord (a b : ℝ × ℝ) (h : a.1 = 2 ∧ a.2 = 1 ∧ b.2 = 2) :
  (∃ k : ℝ, (a.1 + b.1, a.2 + b.2) = k • (a.1 - 2 * b.1, a.2 - 2 * b.2)) →
  b.1 = 4 := by
sorry

end parallel_vectors_x_coord_l680_68019


namespace base5_subtraction_l680_68003

/-- Converts a base 5 number represented as a list of digits to its decimal equivalent -/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 5 * acc + d) 0

/-- Converts a decimal number to its base 5 representation as a list of digits -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The first number in base 5 -/
def num1 : List Nat := [1, 2, 3, 4]

/-- The second number in base 5 -/
def num2 : List Nat := [2, 3, 4]

/-- The expected difference in base 5 -/
def expected_diff : List Nat := [1, 0, 0, 0]

theorem base5_subtraction :
  decimalToBase5 (base5ToDecimal num1 - base5ToDecimal num2) = expected_diff := by
  sorry

end base5_subtraction_l680_68003


namespace quadratic_form_sum_l680_68085

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 2 * x^2 - 8 * x + 2 = a * (x - h)^2 + k) → a + h + k = -2 := by
  sorry

end quadratic_form_sum_l680_68085


namespace line_intercept_ratio_l680_68031

theorem line_intercept_ratio (b : ℝ) (u v : ℝ) 
  (h1 : b ≠ 0)
  (h2 : 0 = 8 * u + b)
  (h3 : 0 = 4 * v + b) :
  u / v = 1 / 2 := by
  sorry

end line_intercept_ratio_l680_68031


namespace thirty_percent_of_hundred_l680_68008

theorem thirty_percent_of_hundred : (30 : ℝ) = (30 / 100) * 100 := by
  sorry

end thirty_percent_of_hundred_l680_68008


namespace lab_workstations_l680_68094

theorem lab_workstations (total_students : ℕ) (two_student_stations : ℕ) (three_student_stations : ℕ) :
  total_students = 38 →
  two_student_stations = 10 →
  two_student_stations * 2 + three_student_stations * 3 = total_students →
  two_student_stations + three_student_stations = 16 :=
by
  sorry

end lab_workstations_l680_68094


namespace five_ruble_coins_count_l680_68024

/-- Given the total number of coins and the number of coins that are not of each other denomination,
    prove that the number of five-ruble coins is 5. -/
theorem five_ruble_coins_count
  (total_coins : ℕ)
  (not_two_ruble : ℕ)
  (not_ten_ruble : ℕ)
  (not_one_ruble : ℕ)
  (h1 : total_coins = 25)
  (h2 : not_two_ruble = 19)
  (h3 : not_ten_ruble = 20)
  (h4 : not_one_ruble = 16) :
  total_coins - ((total_coins - not_two_ruble) + (total_coins - not_ten_ruble) + (total_coins - not_one_ruble)) = 5 :=
by sorry

end five_ruble_coins_count_l680_68024


namespace perpendicular_lines_parallel_l680_68056

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end perpendicular_lines_parallel_l680_68056


namespace not_necessarily_right_triangle_l680_68032

theorem not_necessarily_right_triangle (A B C : ℝ) (h : A / B = 3 / 4 ∧ B / C = 4 / 5) :
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end not_necessarily_right_triangle_l680_68032


namespace binomial_variance_specific_case_l680_68035

-- Define the parameters
def n : ℕ := 10
def p : ℝ := 0.02

-- Define the variance function for a binomial distribution
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Theorem statement
theorem binomial_variance_specific_case :
  binomial_variance n p = 0.196 := by
  sorry

end binomial_variance_specific_case_l680_68035


namespace centroids_form_equilateral_triangle_l680_68051

/-- Given a triangle ABC with vertices z₁, z₂, z₃ in the complex plane,
    the centroids of equilateral triangles constructed externally on its sides
    form an equilateral triangle. -/
theorem centroids_form_equilateral_triangle (z₁ z₂ z₃ : ℂ) : 
  let g₁ := (z₁ * (1 + Complex.exp (Real.pi * Complex.I / 3)) + z₂ * (2 - Complex.exp (Real.pi * Complex.I / 3))) / 3
  let g₂ := (z₂ * (1 + Complex.exp (Real.pi * Complex.I / 3)) + z₃ * (2 - Complex.exp (Real.pi * Complex.I / 3))) / 3
  let g₃ := (z₃ * (1 + Complex.exp (Real.pi * Complex.I / 3)) + z₁ * (2 - Complex.exp (Real.pi * Complex.I / 3))) / 3
  (g₂ - g₁) = (g₃ - g₁) * Complex.exp ((2 * Real.pi * Complex.I) / 3) :=
by sorry


end centroids_form_equilateral_triangle_l680_68051


namespace parabola_vertex_l680_68034

/-- The vertex of the parabola y = 2(x-5)^2 + 3 has coordinates (5, 3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 2*(x - 5)^2 + 3 → (5, 3) = (x, y) := by sorry

end parabola_vertex_l680_68034


namespace can_display_sequence_l680_68084

/-- 
Given a sequence where:
- The first term is 2
- Each subsequent term increases by 3
- The 9th term is 26
Prove that this sequence exists and satisfies these conditions.
-/
theorem can_display_sequence : 
  ∃ (a : ℕ → ℕ), 
    a 1 = 2 ∧ 
    (∀ n, a (n + 1) = a n + 3) ∧ 
    a 9 = 26 := by
  sorry

end can_display_sequence_l680_68084


namespace midpoint_octagon_area_ratio_l680_68037

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- The octagon formed by connecting midpoints of a regular octagon's sides -/
def midpointOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem stating that the area of the midpoint octagon is 1/4 of the original octagon -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpointOctagon o) = (1 / 4) * area o :=
sorry

end midpoint_octagon_area_ratio_l680_68037


namespace matts_working_ratio_l680_68070

/-- Matt's working schedule problem -/
theorem matts_working_ratio :
  let monday_minutes : ℕ := 450
  let wednesday_minutes : ℕ := 300
  let tuesday_minutes : ℕ := wednesday_minutes - 75
  tuesday_minutes * 2 = monday_minutes := by sorry

end matts_working_ratio_l680_68070


namespace metal_square_weight_relation_l680_68000

/-- Represents the properties of a square metal slab -/
structure MetalSquare where
  side_length : ℝ
  weight : ℝ

/-- Theorem stating the relationship between two metal squares of the same material and thickness -/
theorem metal_square_weight_relation 
  (uniformDensity : ℝ → ℝ → ℝ) -- Function representing uniform density
  (square1 : MetalSquare) 
  (square2 : MetalSquare) 
  (h1 : square1.side_length = 4) 
  (h2 : square1.weight = 16) 
  (h3 : square2.side_length = 6) 
  (h4 : ∀ s w, uniformDensity s w = w / (s * s)) -- Density is weight divided by area
  : square2.weight = 36 := by
  sorry

end metal_square_weight_relation_l680_68000


namespace total_diagonals_total_internal_angles_l680_68069

/-- Number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

/-- Number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Calculate the number of diagonals in a polygon -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Calculate the sum of internal angles in a polygon -/
def internal_angles_sum (n : ℕ) : ℕ := (n - 2) * 180

/-- The total number of diagonals in a pentagon and an octagon is 25 -/
theorem total_diagonals : 
  diagonals pentagon_sides + diagonals octagon_sides = 25 := by sorry

/-- The sum of internal angles of a pentagon and an octagon is 1620° -/
theorem total_internal_angles : 
  internal_angles_sum pentagon_sides + internal_angles_sum octagon_sides = 1620 := by sorry

end total_diagonals_total_internal_angles_l680_68069


namespace min_value_and_inequality_l680_68038

theorem min_value_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∃ (min : ℝ), min = 9/4 ∧ ∀ x y, x > 0 → y > 0 → x + y = 2 → 1/(1+x) + 4/(1+y) ≥ min) ∧
  a^2*b^2 + a^2 + b^2 ≥ a*b*(a+b+1) := by
sorry

end min_value_and_inequality_l680_68038


namespace inequality_proof_l680_68075

theorem inequality_proof (x y : ℝ) (h : x ≠ y) : x^4 + y^4 > x^3*y + x*y^3 := by
  sorry

end inequality_proof_l680_68075


namespace rectangular_solid_volume_l680_68020

/-- The volume of a rectangular solid with given face areas -/
theorem rectangular_solid_volume
  (side_face_area front_face_area bottom_face_area : ℝ)
  (h_side : side_face_area = 18)
  (h_front : front_face_area = 15)
  (h_bottom : bottom_face_area = 10) :
  ∃ (x y z : ℝ),
    x * y = side_face_area ∧
    y * z = front_face_area ∧
    z * x = bottom_face_area ∧
    x * y * z = 30 * Real.sqrt 3 :=
by sorry

end rectangular_solid_volume_l680_68020


namespace alicia_wages_l680_68049

/- Define the hourly wage in dollars -/
def hourly_wage : ℚ := 25

/- Define the local tax rate as a percentage -/
def tax_rate : ℚ := 2.5

/- Define the conversion rate from dollars to cents -/
def cents_per_dollar : ℕ := 100

/- Theorem statement -/
theorem alicia_wages :
  let wage_in_cents := hourly_wage * cents_per_dollar
  let tax_amount := (tax_rate / 100) * wage_in_cents
  let after_tax_earnings := wage_in_cents - tax_amount
  (tax_amount = 62.5 ∧ after_tax_earnings = 2437.5) := by
  sorry

end alicia_wages_l680_68049


namespace find_m_l680_68086

theorem find_m : ∃ m : ℝ, 10^m = 10^2 * Real.sqrt (10^90 / 0.0001) ∧ m = 49 := by
  sorry

end find_m_l680_68086


namespace simplest_quadratic_radical_l680_68053

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), x = Real.sqrt y ∧ 
  (∀ (z : ℚ), x ≠ (z : ℝ)) ∧
  (∀ (a b : ℕ), x ≠ Real.sqrt (a / b))

theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (Real.sqrt 3) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 9) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/2)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 0.1) :=
sorry

end simplest_quadratic_radical_l680_68053


namespace problem_statement_l680_68067

theorem problem_statement (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^9 + a^6 = 2 := by
  sorry

end problem_statement_l680_68067


namespace unique_operation_equals_one_l680_68012

theorem unique_operation_equals_one : 
  (-3 + (-3) ≠ 1) ∧ 
  (-3 - (-3) ≠ 1) ∧ 
  (-3 / (-3) = 1) ∧ 
  (-3 * (-3) ≠ 1) :=
by sorry

end unique_operation_equals_one_l680_68012


namespace bobby_candy_count_l680_68005

theorem bobby_candy_count (initial : ℕ) (additional : ℕ) : 
  initial = 26 → additional = 17 → initial + additional = 43 := by
  sorry

end bobby_candy_count_l680_68005


namespace floor_to_total_ratio_example_l680_68004

/-- The ratio of students sitting on the floor to the total number of students -/
def floor_to_total_ratio (total_students floor_students : ℕ) : ℚ :=
  floor_students / total_students

/-- Proof that the ratio of students sitting on the floor to the total number of students is 11/26 -/
theorem floor_to_total_ratio_example : 
  floor_to_total_ratio 26 11 = 11 / 26 := by
  sorry

end floor_to_total_ratio_example_l680_68004


namespace soccer_team_goals_l680_68080

theorem soccer_team_goals (total_players : ℕ) (total_goals : ℕ) (games_played : ℕ) 
  (h1 : total_players = 24)
  (h2 : total_goals = 150)
  (h3 : games_played = 15)
  (h4 : (total_players / 3) * games_played = total_goals - 30) : 
  30 = total_goals - (total_players / 3) * games_played := by
sorry

end soccer_team_goals_l680_68080


namespace plot_length_l680_68073

/-- Proves that the length of a rectangular plot is 65 meters given the specified conditions -/
theorem plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 30 →
  perimeter = 2 * length + 2 * breadth →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  perimeter = total_cost / cost_per_meter →
  length = 65 := by
sorry


end plot_length_l680_68073


namespace simplify_expression_l680_68076

theorem simplify_expression (y : ℝ) : 7*y - 3 + 2*y + 15 = 9*y + 12 := by
  sorry

end simplify_expression_l680_68076


namespace absolute_value_inequality_l680_68090

theorem absolute_value_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : abs a > abs b := by
  sorry

end absolute_value_inequality_l680_68090


namespace exponent_negative_product_squared_l680_68044

theorem exponent_negative_product_squared (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := by
  sorry

end exponent_negative_product_squared_l680_68044


namespace power_four_congruence_l680_68097

theorem power_four_congruence (n : ℕ) (a : ℤ) (hn : n > 0) (ha : a^3 ≡ 1 [ZMOD n]) :
  a^4 ≡ a [ZMOD n] := by
  sorry

end power_four_congruence_l680_68097


namespace f_properties_l680_68058

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x) ^ 2 + a

theorem f_properties (a : ℝ) :
  (∃ (T : ℝ), ∀ (x : ℝ), f a x = f a (x + T)) ∧ 
  (∃ (min_val : ℝ), min_val = 0 → 
    (a = 1 ∧ 
     (∃ (max_val : ℝ), max_val = 4 ∧ ∀ (x : ℝ), f a x ≤ max_val) ∧
     (∃ (k : ℤ), ∀ (x : ℝ), f a x = f a (↑k * Real.pi / 2 + Real.pi / 6 - x)))) :=
by sorry

end f_properties_l680_68058


namespace range_of_a_l680_68014

theorem range_of_a (a : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2*y + 2*z) →
  (a ≤ -2 ∨ a ≥ 4) :=
sorry

end range_of_a_l680_68014


namespace quadratic_two_distinct_roots_l680_68077

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x - k = 0 ∧ y^2 + 2*y - k = 0) ↔ k > -1 := by
  sorry

end quadratic_two_distinct_roots_l680_68077


namespace angle_rotation_l680_68033

theorem angle_rotation (initial_angle rotation : ℝ) (h1 : initial_angle = 60) (h2 : rotation = 420) :
  (initial_angle - (rotation % 360)) % 360 = 0 :=
by sorry

end angle_rotation_l680_68033


namespace piano_lesson_rate_piano_rate_is_28_l680_68026

/-- Calculates the hourly rate for piano lessons given the conditions -/
theorem piano_lesson_rate (clarinet_rate : ℝ) (clarinet_hours : ℝ) (piano_hours : ℝ) 
  (extra_piano_cost : ℝ) (weeks_per_year : ℕ) : ℝ :=
  let annual_clarinet_cost := clarinet_rate * clarinet_hours * weeks_per_year
  let annual_piano_cost := annual_clarinet_cost + extra_piano_cost
  annual_piano_cost / (piano_hours * weeks_per_year)

/-- The hourly rate for piano lessons is $28 -/
theorem piano_rate_is_28 : 
  piano_lesson_rate 40 3 5 1040 52 = 28 := by
sorry

end piano_lesson_rate_piano_rate_is_28_l680_68026


namespace complex_equation_solution_l680_68091

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := (1 + i) * z = 2 * i

-- Theorem statement
theorem complex_equation_solution :
  ∃ (z : ℂ), equation z ∧ z = 1 + i :=
sorry

end complex_equation_solution_l680_68091


namespace system_of_equations_sum_l680_68065

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) : 
  a / (a - 17) + b / (b - 29) + c / (c - 53) = 1 := by
  sorry

end system_of_equations_sum_l680_68065


namespace regular_polygon_with_150_degree_angles_l680_68055

theorem regular_polygon_with_150_degree_angles (n : ℕ) : 
  n > 2 →                                 -- n is the number of sides, must be greater than 2
  (180 * (n - 2) : ℝ) = (150 * n : ℝ) →   -- sum of interior angles formula
  n = 12 :=                               -- conclusion: the polygon has 12 sides
by sorry

end regular_polygon_with_150_degree_angles_l680_68055


namespace triangle_reconstruction_from_altitude_feet_l680_68074

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle with vertices A, B, C -/
structure Triangle :=
  (A B C : Point)

/-- Represents the feet of altitudes of a triangle -/
structure AltitudeFeet :=
  (A1 B1 C1 : Point)

/-- Predicate to check if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop :=
  sorry

/-- Function to get the feet of altitudes of a triangle -/
def getAltitudeFeet (t : Triangle) : AltitudeFeet :=
  sorry

/-- Predicate to check if a triangle can be reconstructed from given altitude feet -/
def canReconstructTriangle (feet : AltitudeFeet) : Prop :=
  sorry

/-- Theorem stating that an acute-angled triangle can be reconstructed from its altitude feet -/
theorem triangle_reconstruction_from_altitude_feet
  (t : Triangle) (h : isAcuteAngled t) :
  canReconstructTriangle (getAltitudeFeet t) :=
sorry

end triangle_reconstruction_from_altitude_feet_l680_68074


namespace unit_circle_image_l680_68057

def unit_circle_mapping (z : ℂ) : Prop := Complex.abs z = 1

theorem unit_circle_image :
  ∀ z : ℂ, unit_circle_mapping z → Complex.abs (z^2) = 1 := by
sorry

end unit_circle_image_l680_68057


namespace calculation_proof_l680_68017

theorem calculation_proof :
  ((-1 : ℚ)^2 + 27/4 * (-4) / (-3)^2 = -4) ∧
  ((-36 : ℚ) * (3/4 - 5/6 + 7/9) = -25) := by
  sorry

end calculation_proof_l680_68017


namespace company_assets_and_price_l680_68018

theorem company_assets_and_price (A B P : ℝ) 
  (h1 : P = 1.5 * A) 
  (h2 : P = 0.8571428571428571 * (A + B)) : 
  P = 2 * B := by
sorry

end company_assets_and_price_l680_68018


namespace percentage_calculation_l680_68042

theorem percentage_calculation (number : ℝ) (result : ℝ) (P : ℝ) : 
  number = 4400 → 
  result = 99 → 
  P * number = result → 
  P = 0.0225 := by
  sorry

end percentage_calculation_l680_68042


namespace intersection_value_l680_68072

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The equation of the first line -/
def line1 (a c : ℝ) (x y : ℝ) : Prop := a * x - 3 * y = c

/-- The equation of the second line -/
def line2 (b c : ℝ) (x y : ℝ) : Prop := 3 * x + b * y = -c

/-- The theorem stating that c = 39 given the conditions -/
theorem intersection_value (a b c : ℝ) : 
  perpendicular (a / 3) (-3 / b) →
  line1 a c 2 (-3) →
  line2 b c 2 (-3) →
  c = 39 := by sorry

end intersection_value_l680_68072


namespace last_date_divisible_by_101_in_2011_l680_68023

def is_valid_date (year month day : ℕ) : Prop :=
  year = 2011 ∧ 
  1 ≤ month ∧ month ≤ 12 ∧
  1 ≤ day ∧ day ≤ 31

def date_to_number (year month day : ℕ) : ℕ :=
  year * 10000 + month * 100 + day

theorem last_date_divisible_by_101_in_2011 :
  ∀ (month day : ℕ),
    is_valid_date 2011 month day →
    date_to_number 2011 month day ≤ 20111221 ∨
    ¬(date_to_number 2011 month day % 101 = 0) :=
sorry

end last_date_divisible_by_101_in_2011_l680_68023


namespace geometric_sequence_cars_below_threshold_l680_68068

/- Define the sequence of ordinary cars -/
def a : ℕ → ℝ
  | 0 => 300  -- Initial value for 2020
  | n + 1 => 0.9 * a n + 8

/- Define the transformed sequence -/
def b (n : ℕ) : ℝ := a n - 80

/- Theorem statement -/
theorem geometric_sequence : ∀ n : ℕ, b (n + 1) = 0.9 * b n := by
  sorry

/- Additional theorem to show the year when cars are less than 1.5 million -/
theorem cars_below_threshold (n : ℕ) : a n < 150 → n ≥ 12 := by
  sorry

end geometric_sequence_cars_below_threshold_l680_68068


namespace cubic_equation_solutions_l680_68002

theorem cubic_equation_solutions :
  ∀ x : ℝ, (x ^ (1/3) = 15 / (8 - x ^ (1/3))) ↔ (x = 27 ∨ x = 125) := by
  sorry

end cubic_equation_solutions_l680_68002


namespace expression_evaluation_l680_68064

theorem expression_evaluation :
  let x : ℤ := -1
  (x + 1) * (x - 2) + 2 * (x + 4) * (x - 4) = -30 :=
by sorry

end expression_evaluation_l680_68064


namespace lamp_height_difference_l680_68098

/-- The height difference between two lamps -/
theorem lamp_height_difference (old_height new_height : ℝ) 
  (h1 : old_height = 1) 
  (h2 : new_height = 2.3333333333333335) : 
  new_height - old_height = 1.3333333333333335 := by
  sorry

end lamp_height_difference_l680_68098


namespace inverse_matrices_solution_l680_68054

theorem inverse_matrices_solution :
  ∀ (a b : ℚ),
  let A : Matrix (Fin 2) (Fin 2) ℚ := ![![a, 3], ![2, 5]]
  let B : Matrix (Fin 2) (Fin 2) ℚ := ![![b, -1/5], ![1/2, 1/10]]
  let I : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![0, 1]]
  A * B = I → a = 3/2 ∧ b = -5/4 := by sorry

end inverse_matrices_solution_l680_68054
