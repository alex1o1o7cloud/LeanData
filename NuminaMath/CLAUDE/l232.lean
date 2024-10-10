import Mathlib

namespace mans_age_twice_sons_age_l232_23212

/-- Proves that the number of years until a man's age is twice his son's age is 2 -/
theorem mans_age_twice_sons_age (man_age son_age years : ℕ) : 
  man_age = son_age + 28 →
  son_age = 26 →
  (man_age + years) = 2 * (son_age + years) →
  years = 2 := by
sorry

end mans_age_twice_sons_age_l232_23212


namespace intersection_equals_N_l232_23267

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x > 0, y = x^2}
def N : Set ℝ := {y | ∃ x > 0, y = x + 2}

-- State the theorem
theorem intersection_equals_N : M ∩ N = N := by
  sorry

end intersection_equals_N_l232_23267


namespace selection_theorem_l232_23257

/-- The number of ways to select one person from a department with n employees -/
def selectOne (n : ℕ) : ℕ := n

/-- The total number of ways to select one person from three departments -/
def totalWays (deptA deptB deptC : ℕ) : ℕ :=
  selectOne deptA + selectOne deptB + selectOne deptC

theorem selection_theorem :
  totalWays 2 4 3 = 9 := by sorry

end selection_theorem_l232_23257


namespace circle_equation_from_parabola_l232_23227

/-- Given a parabola x² = 16y, prove that a circle centered at its focus
    and tangent to its directrix has the equation x² + (y - 4)² = 64 -/
theorem circle_equation_from_parabola (x y : ℝ) :
  (x^2 = 16*y) →  -- Parabola equation
  ∃ (h k r : ℝ),
    (h = 0 ∧ k = 4) →  -- Focus (circle center) at (0, 4)
    ((x - h)^2 + (y - k)^2 = r^2) →  -- General circle equation
    (y = -4 → (x - h)^2 + (y - k)^2 = r^2)  -- Circle tangent to directrix y = -4
    →
    ((x - 0)^2 + (y - 4)^2 = 64)  -- Specific circle equation
    :=
by sorry

end circle_equation_from_parabola_l232_23227


namespace second_division_percentage_l232_23233

theorem second_division_percentage 
  (total_students : ℕ) 
  (first_division_percentage : ℚ) 
  (just_passed : ℕ) 
  (h1 : total_students = 300)
  (h2 : first_division_percentage = 28/100)
  (h3 : just_passed = 54)
  : (↑(total_students - (total_students * first_division_percentage).floor - just_passed) / total_students : ℚ) = 54/100 := by
  sorry

end second_division_percentage_l232_23233


namespace brick_length_proof_l232_23286

/-- Given a courtyard and bricks with specific dimensions, prove the length of each brick -/
theorem brick_length_proof (courtyard_length : ℝ) (courtyard_breadth : ℝ) 
  (brick_breadth : ℝ) (total_bricks : ℕ) :
  courtyard_length = 20 →
  courtyard_breadth = 16 →
  brick_breadth = 0.1 →
  total_bricks = 16000 →
  (courtyard_length * courtyard_breadth * 10000) / (total_bricks * brick_breadth * 100) = 20 := by
  sorry

end brick_length_proof_l232_23286


namespace max_fruits_is_34_l232_23270

/-- Represents the weight of an apple in grams -/
def apple_weight : ℕ := 300

/-- Represents the weight of a pear in grams -/
def pear_weight : ℕ := 200

/-- Represents the maximum weight Ana's bag can hold in grams -/
def bag_capacity : ℕ := 7000

/-- Represents the constraint on the number of apples and pears -/
def weight_constraint (m p : ℕ) : Prop :=
  m * apple_weight + p * pear_weight ≤ bag_capacity

/-- Represents the total number of fruits -/
def total_fruits (m p : ℕ) : ℕ := m + p

/-- Theorem stating that the maximum number of fruits Ana can buy is 34 -/
theorem max_fruits_is_34 : 
  ∃ (m p : ℕ), weight_constraint m p ∧ m > 0 ∧ p > 0 ∧
  total_fruits m p = 34 ∧
  ∀ (m' p' : ℕ), weight_constraint m' p' ∧ m' > 0 ∧ p' > 0 → 
    total_fruits m' p' ≤ 34 :=
sorry

end max_fruits_is_34_l232_23270


namespace farmer_pumpkin_seeds_per_row_l232_23254

/-- Represents the farmer's planting scenario -/
structure FarmerPlanting where
  bean_seedlings : ℕ
  bean_per_row : ℕ
  pumpkin_seeds : ℕ
  radishes : ℕ
  radish_per_row : ℕ
  rows_per_bed : ℕ
  plant_beds : ℕ

/-- Calculates the number of pumpkin seeds per row -/
def pumpkin_seeds_per_row (fp : FarmerPlanting) : ℕ :=
  fp.pumpkin_seeds / (fp.plant_beds * fp.rows_per_bed - fp.bean_seedlings / fp.bean_per_row - fp.radishes / fp.radish_per_row)

/-- Theorem stating that given the specific planting scenario, the farmer plants 7 pumpkin seeds per row -/
theorem farmer_pumpkin_seeds_per_row :
  let fp : FarmerPlanting := {
    bean_seedlings := 64,
    bean_per_row := 8,
    pumpkin_seeds := 84,
    radishes := 48,
    radish_per_row := 6,
    rows_per_bed := 2,
    plant_beds := 14
  }
  pumpkin_seeds_per_row fp = 7 := by
  sorry

end farmer_pumpkin_seeds_per_row_l232_23254


namespace polynomial_never_33_l232_23237

theorem polynomial_never_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end polynomial_never_33_l232_23237


namespace photo_arrangements_3_3_1_l232_23223

/-- The number of possible arrangements for a group photo -/
def photo_arrangements (num_boys num_girls : ℕ) : ℕ :=
  let adjacent_choices := num_boys * num_girls
  let remaining_boys := num_boys - 1
  let remaining_girls := num_girls - 1
  let remaining_arrangements := (remaining_boys * (remaining_boys - 1)) * 
                                (remaining_girls * (remaining_girls - 1) * 
                                 (remaining_boys + remaining_girls) * 
                                 (remaining_boys + remaining_girls - 1))
  2 * adjacent_choices * remaining_arrangements

/-- Theorem stating the number of arrangements for 3 boys, 3 girls, and 1 teacher -/
theorem photo_arrangements_3_3_1 :
  photo_arrangements 3 3 = 432 := by
  sorry

#eval photo_arrangements 3 3

end photo_arrangements_3_3_1_l232_23223


namespace motorcycle_theorem_l232_23249

def motorcycle_problem (k r t : ℝ) (h1 : k > 0) (h2 : r > 0) (h3 : t > 0) : Prop :=
  ∃ (v1 v2 : ℝ), v1 > v2 ∧ v2 > 0 ∧
    r * (v1 - v2) = k ∧
    t * (v1 + v2) = k ∧
    v1 / v2 = |r + t| / |r - t|

theorem motorcycle_theorem (k r t : ℝ) (h1 : k > 0) (h2 : r > 0) (h3 : t > 0) :
  motorcycle_problem k r t h1 h2 h3 :=
sorry

end motorcycle_theorem_l232_23249


namespace largest_two_digit_prime_factor_of_binomial_150_75_l232_23275

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem largest_two_digit_prime_factor_of_binomial_150_75 :
  (∃ (p : ℕ), p.Prime ∧ p ∣ binomial 150 75 ∧ 10 ≤ p ∧ p < 100) ∧
  (∀ (q : ℕ), q.Prime → q ∣ binomial 150 75 → 10 ≤ q → q < 100 → q ≤ 73) :=
by sorry

end largest_two_digit_prime_factor_of_binomial_150_75_l232_23275


namespace spurs_team_size_l232_23294

/-- The number of basketballs each player has -/
def basketballs_per_player : ℕ := 11

/-- The total number of basketballs -/
def total_basketballs : ℕ := 242

/-- The number of players on the team -/
def number_of_players : ℕ := total_basketballs / basketballs_per_player

theorem spurs_team_size : number_of_players = 22 := by
  sorry

end spurs_team_size_l232_23294


namespace distance_is_13_l232_23299

/-- The distance between two villages Yolkino and Palkino. -/
def distance_between_villages : ℝ := 13

/-- A point on the highway between Yolkino and Palkino. -/
structure HighwayPoint where
  distance_to_yolkino : ℝ
  distance_to_palkino : ℝ
  sum_is_13 : distance_to_yolkino + distance_to_palkino = 13

/-- The theorem stating that the distance between Yolkino and Palkino is 13 km. -/
theorem distance_is_13 : 
  ∀ (p : HighwayPoint), distance_between_villages = p.distance_to_yolkino + p.distance_to_palkino :=
by
  sorry

end distance_is_13_l232_23299


namespace ashok_pyarelal_capital_ratio_l232_23282

/-- Given a business investment scenario where:
  * The total loss is 1000
  * Pyarelal's loss is 900
  * Ashok's loss is the remaining amount
  * The ratio of losses is proportional to the ratio of investments

  This theorem proves that the ratio of Ashok's capital to Pyarelal's capital is 1:9.
-/
theorem ashok_pyarelal_capital_ratio 
  (total_loss : ℕ) 
  (pyarelal_loss : ℕ) 
  (h_total_loss : total_loss = 1000)
  (h_pyarelal_loss : pyarelal_loss = 900)
  (ashok_loss : ℕ := total_loss - pyarelal_loss)
  (ashok_capital pyarelal_capital : ℚ)
  (h_loss_ratio : ashok_loss / pyarelal_loss = ashok_capital / pyarelal_capital) :
  ashok_capital / pyarelal_capital = 1 / 9 := by
  sorry

end ashok_pyarelal_capital_ratio_l232_23282


namespace max_sphere_radius_squared_l232_23246

/-- Two congruent right circular cones with a sphere inside -/
structure ConeFigure where
  /-- Base radius of each cone -/
  base_radius : ℝ
  /-- Height of each cone -/
  cone_height : ℝ
  /-- Distance from base to intersection point of axes -/
  intersection_distance : ℝ
  /-- Radius of the sphere -/
  sphere_radius : ℝ
  /-- Condition: base radius is 5 -/
  base_radius_eq : base_radius = 5
  /-- Condition: cone height is 10 -/
  cone_height_eq : cone_height = 10
  /-- Condition: intersection distance is 5 -/
  intersection_eq : intersection_distance = 5
  /-- Condition: sphere lies within both cones -/
  sphere_within_cones : sphere_radius ≤ intersection_distance

/-- The maximum possible value of r^2 is 80 -/
theorem max_sphere_radius_squared (cf : ConeFigure) : 
  ∃ (max_r : ℝ), ∀ (r : ℝ), cf.sphere_radius = r → r^2 ≤ max_r^2 ∧ max_r^2 = 80 := by
  sorry

end max_sphere_radius_squared_l232_23246


namespace f_of_4_6_l232_23208

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

theorem f_of_4_6 : f (4, 6) = (10, -2) := by
  sorry

end f_of_4_6_l232_23208


namespace art_gallery_display_ratio_l232_23279

theorem art_gallery_display_ratio :
  let total_pieces : ℕ := 2700
  let sculptures_not_displayed : ℕ := 1200
  let paintings_not_displayed : ℕ := sculptures_not_displayed / 3
  let pieces_not_displayed : ℕ := sculptures_not_displayed + paintings_not_displayed
  let pieces_displayed : ℕ := total_pieces - pieces_not_displayed
  let sculptures_displayed : ℕ := pieces_displayed / 6
  pieces_displayed / total_pieces = 11 / 27 :=
by
  sorry

end art_gallery_display_ratio_l232_23279


namespace sin_squared_alpha_minus_pi_fourth_l232_23232

theorem sin_squared_alpha_minus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 2/3) :
  Real.sin (α - Real.pi/4)^2 = 1/6 := by
  sorry

end sin_squared_alpha_minus_pi_fourth_l232_23232


namespace ab_values_l232_23240

theorem ab_values (a b : ℝ) (h : a^2*b^2 + a^2 + b^2 + 1 = 4*a*b) :
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) := by
  sorry

end ab_values_l232_23240


namespace perfect_square_trinomial_l232_23245

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + (m+1)*x + 16 = (x + a)^2) → m = 7 ∨ m = -9 := by
  sorry

end perfect_square_trinomial_l232_23245


namespace point_on_line_l232_23209

/-- A point on a line in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Defines a line y = mx + c -/
structure Line2D where
  m : ℝ
  c : ℝ

/-- Theorem: If a point P(a,b) lies on the line y = -3x - 4, then b + 3a + 4 = 0 -/
theorem point_on_line (P : Point2D) (L : Line2D) 
  (h1 : L.m = -3)
  (h2 : L.c = -4)
  (h3 : P.y = L.m * P.x + L.c) :
  P.y + 3 * P.x + 4 = 0 := by
  sorry


end point_on_line_l232_23209


namespace gcd_7163_209_l232_23259

theorem gcd_7163_209 : Nat.gcd 7163 209 = 19 := by
  have h1 : 7163 = 209 * 34 + 57 := by sorry
  have h2 : 209 = 57 * 3 + 38 := by sorry
  have h3 : 57 = 38 * 1 + 19 := by sorry
  have h4 : 38 = 19 * 2 := by sorry
  sorry

end gcd_7163_209_l232_23259


namespace bowling_money_theorem_l232_23272

/-- The cost of renting bowling shoes for a day -/
def shoe_rental_cost : ℚ := 0.50

/-- The cost of bowling one game -/
def game_cost : ℚ := 1.75

/-- The maximum number of complete games the person can bowl -/
def max_games : ℕ := 7

/-- The total amount of money the person has -/
def total_money : ℚ := shoe_rental_cost + max_games * game_cost

theorem bowling_money_theorem :
  total_money = 12.75 := by sorry

end bowling_money_theorem_l232_23272


namespace angelinas_speed_l232_23244

/-- Proves that Angelina's speed from grocery to gym is 24 meters per second -/
theorem angelinas_speed (v : ℝ) : 
  v > 0 →  -- Assume positive speed
  720 / v - 480 / (2 * v) = 40 →  -- Time difference condition
  2 * v = 24 := by
sorry

end angelinas_speed_l232_23244


namespace worker_a_time_proof_l232_23289

/-- The time it takes for Worker A to complete the job alone -/
def worker_a_time : ℝ := 8.4

/-- The time it takes for Worker B to complete the job alone -/
def worker_b_time : ℝ := 6

/-- The time it takes for both workers to complete the job together -/
def combined_time : ℝ := 3.428571428571429

theorem worker_a_time_proof :
  (1 / worker_a_time) + (1 / worker_b_time) = (1 / combined_time) :=
sorry

end worker_a_time_proof_l232_23289


namespace vector_parallel_solution_l232_23241

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, 1)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem vector_parallel_solution :
  ∃ (x : ℝ), parallel (a.1 + x * b.1, a.2 + x * b.2) (a.1 - b.1, a.2 - b.2) ∧ x = -1 :=
by sorry

end vector_parallel_solution_l232_23241


namespace carol_cupcakes_l232_23201

/-- Calculates the total number of cupcakes Carol has after selling some and making more. -/
def total_cupcakes (initial : ℕ) (sold : ℕ) (new_made : ℕ) : ℕ :=
  initial - sold + new_made

/-- Proves that Carol has 49 cupcakes in total given the initial conditions. -/
theorem carol_cupcakes : total_cupcakes 30 9 28 = 49 := by
  sorry

end carol_cupcakes_l232_23201


namespace cubic_root_sum_l232_23205

theorem cubic_root_sum (a b c d : ℝ) (h1 : a ≠ 0) 
  (h2 : a * 4^3 + b * 4^2 + c * 4 + d = 0)
  (h3 : a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) :
  (b + c) / a = -13 := by
sorry

end cubic_root_sum_l232_23205


namespace chocolate_box_problem_l232_23207

theorem chocolate_box_problem (B : ℕ) : 
  (B : ℚ) - ((1/4 : ℚ) * B - 5) - ((1/4 : ℚ) * B - 10) = 110 → B = 190 := by
  sorry

end chocolate_box_problem_l232_23207


namespace folded_rectangle_perimeter_l232_23253

/-- The perimeter of a rectangular sheet folded along its diagonal -/
theorem folded_rectangle_perimeter (length width : ℝ) (h1 : length = 20) (h2 : width = 12) :
  2 * (length + width) = 64 := by
  sorry

end folded_rectangle_perimeter_l232_23253


namespace cosine_sum_twenty_degrees_l232_23229

theorem cosine_sum_twenty_degrees : 
  Real.cos (20 * π / 180) + Real.cos (60 * π / 180) + 
  Real.cos (100 * π / 180) + Real.cos (140 * π / 180) = 1/2 := by
  sorry

end cosine_sum_twenty_degrees_l232_23229


namespace dennis_initial_amount_l232_23265

theorem dennis_initial_amount (shirt_cost change_received : ℕ) : 
  shirt_cost = 27 → change_received = 23 → shirt_cost + change_received = 50 := by
  sorry

end dennis_initial_amount_l232_23265


namespace john_miles_conversion_l232_23214

/-- Converts a base-7 number represented as a list of digits to its base-10 equivalent -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base-7 representation of the number of miles John cycled -/
def johnMilesBase7 : List Nat := [6, 1, 5, 3]

theorem john_miles_conversion :
  base7ToBase10 johnMilesBase7 = 1287 := by
  sorry

end john_miles_conversion_l232_23214


namespace arithmetic_sequence_sum_l232_23291

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying the condition,
    prove that the sum of specific terms equals 2502.5. -/
theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_condition : a 3 + a 4 + a 10 + a 11 = 2002) :
  a 1 + a 5 + a 7 + a 9 + a 13 = 2502.5 := by
  sorry

end arithmetic_sequence_sum_l232_23291


namespace teal_color_perception_l232_23206

theorem teal_color_perception (total : ℕ) (kinda_blue : ℕ) (both : ℕ) (neither : ℕ) :
  total = 150 →
  kinda_blue = 90 →
  both = 35 →
  neither = 25 →
  ∃ kinda_green : ℕ, kinda_green = 70 ∧ 
    kinda_green + kinda_blue - both + neither = total :=
by sorry

end teal_color_perception_l232_23206


namespace positive_integer_solutions_l232_23222

theorem positive_integer_solutions :
  ∀ (a b c x y z : ℕ+),
    (a + b + c = x * y * z ∧ x + y + z = a * b * c) ↔
    ((x = 3 ∧ y = 2 ∧ z = 1 ∧ a = 3 ∧ b = 2 ∧ c = 1) ∨
     (x = 3 ∧ y = 3 ∧ z = 1 ∧ a = 5 ∧ b = 2 ∧ c = 1) ∨
     (x = 5 ∧ y = 2 ∧ z = 1 ∧ a = 3 ∧ b = 3 ∧ c = 1)) :=
by sorry

end positive_integer_solutions_l232_23222


namespace stratified_sample_theorem_l232_23287

/-- Represents the number of households selected in a stratum -/
structure StratumSample where
  total : ℕ
  selected : ℕ

/-- Represents the stratified sample of households -/
structure StratifiedSample where
  high_income : StratumSample
  middle_income : StratumSample
  low_income : StratumSample

def total_households (s : StratifiedSample) : ℕ :=
  s.high_income.total + s.middle_income.total + s.low_income.total

def total_selected (s : StratifiedSample) : ℕ :=
  s.high_income.selected + s.middle_income.selected + s.low_income.selected

theorem stratified_sample_theorem (s : StratifiedSample) :
  s.high_income.total = 120 →
  s.middle_income.total = 200 →
  s.low_income.total = 160 →
  s.high_income.selected = 6 →
  total_households s = 480 →
  total_selected s = 24 := by
  sorry

#check stratified_sample_theorem

end stratified_sample_theorem_l232_23287


namespace chord_length_sqrt3_line_l232_23239

/-- A line in the form y = mx --/
structure Line where
  m : ℝ

/-- A circle in the form (x - h)^2 + (y - k)^2 = r^2 --/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The length of a chord formed by the intersection of a line and a circle --/
def chordLength (l : Line) (c : Circle) : ℝ :=
  sorry

theorem chord_length_sqrt3_line (c : Circle) :
  c.h = 2 ∧ c.k = 0 ∧ c.r = 2 →
  chordLength { m := Real.sqrt 3 } c = 2 := by
  sorry

end chord_length_sqrt3_line_l232_23239


namespace red_paint_percentage_l232_23234

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  blue : ℝ
  red : ℝ
  white : ℝ
  total : ℝ
  blue_percentage : ℝ
  red_percentage : ℝ
  white_percentage : ℝ

/-- Given a paint mixture with 70% blue paint, 140 ounces of blue paint, 
    and 20 ounces of white paint, prove that 20% of the mixture is red paint -/
theorem red_paint_percentage 
  (mixture : PaintMixture) 
  (h1 : mixture.blue_percentage = 0.7) 
  (h2 : mixture.blue = 140) 
  (h3 : mixture.white = 20) 
  (h4 : mixture.total = mixture.blue + mixture.red + mixture.white) 
  (h5 : mixture.blue_percentage + mixture.red_percentage + mixture.white_percentage = 1) :
  mixture.red_percentage = 0.2 := by
  sorry

end red_paint_percentage_l232_23234


namespace ball_placement_count_l232_23273

theorem ball_placement_count :
  let n_balls : ℕ := 4
  let n_boxes : ℕ := 3
  let placement_options := n_boxes ^ n_balls
  placement_options = 81 :=
by sorry

end ball_placement_count_l232_23273


namespace polynomial_intersection_l232_23242

-- Define the polynomials f and h
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def h (p q x : ℝ) : ℝ := x^2 + p*x + q

-- Define the theorem
theorem polynomial_intersection (a b p q : ℝ) : 
  -- f and h are distinct polynomials
  (∃ x, f a b x ≠ h p q x) →
  -- The x-coordinate of the vertex of f is a root of h
  h p q (-a/2) = 0 →
  -- The x-coordinate of the vertex of h is a root of f
  f a b (-p/2) = 0 →
  -- Both f and h have the same minimum value
  (∃ y, f a b (-a/2) = y ∧ h p q (-p/2) = y) →
  -- The graphs of f and h intersect at the point (50, -50)
  f a b 50 = -50 ∧ h p q 50 = -50 →
  -- Conclusion: a + p = 0
  a + p = 0 := by
sorry

end polynomial_intersection_l232_23242


namespace intersection_point_of_function_and_inverse_l232_23262

-- Define the function f
def f (b : ℤ) : ℝ → ℝ := λ x => 4 * x + b

-- State the theorem
theorem intersection_point_of_function_and_inverse
  (b : ℤ) (a : ℤ) :
  (∃ (x : ℝ), f b x = x ∧ f b (-4) = a) →
  a = -4 :=
by sorry

end intersection_point_of_function_and_inverse_l232_23262


namespace tip_percentage_is_22_percent_l232_23219

/-- Calculates the tip percentage given the total amount spent, food price, and sales tax rate. -/
def calculate_tip_percentage (total_spent : ℚ) (food_price : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let sales_tax := food_price * sales_tax_rate
  let tip := total_spent - (food_price + sales_tax)
  (tip / food_price) * 100

/-- Theorem stating that under the given conditions, the tip percentage is 22%. -/
theorem tip_percentage_is_22_percent :
  let total_spent : ℚ := 132
  let food_price : ℚ := 100
  let sales_tax_rate : ℚ := 10 / 100
  calculate_tip_percentage total_spent food_price sales_tax_rate = 22 := by
  sorry

end tip_percentage_is_22_percent_l232_23219


namespace expired_bottle_probability_l232_23297

theorem expired_bottle_probability (total_bottles : ℕ) (expired_bottles : ℕ) 
  (prob_both_unexpired : ℚ) :
  total_bottles = 30 →
  expired_bottles = 3 →
  prob_both_unexpired = 351 / 435 →
  (1 - prob_both_unexpired : ℚ) = 28 / 145 :=
by sorry

end expired_bottle_probability_l232_23297


namespace ellipse_equation_from_conditions_l232_23221

/-- An ellipse with foci on the y-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < b ∧ b < a
  h_c : c^2 = a^2 - b^2

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The condition that a point on the short axis and the two foci form an equilateral triangle -/
def equilateral_triangle_condition (e : Ellipse) : Prop :=
  e.a = 2 * e.c

/-- The condition that the shortest distance from the foci to the endpoints of the major axis is √3 -/
def shortest_distance_condition (e : Ellipse) : Prop :=
  e.a - e.c = Real.sqrt 3

theorem ellipse_equation_from_conditions (e : Ellipse)
  (h_triangle : equilateral_triangle_condition e)
  (h_distance : shortest_distance_condition e) :
  ∀ x y : ℝ, ellipse_equation e x y ↔ x^2 / 12 + y^2 / 9 = 1 :=
sorry

end ellipse_equation_from_conditions_l232_23221


namespace x0_value_l232_23231

open Real

noncomputable def f (x : ℝ) : ℝ := x * (2016 + log x)

theorem x0_value (x0 : ℝ) (h : deriv f x0 = 2017) : x0 = 1 := by
  sorry

end x0_value_l232_23231


namespace marker_distance_l232_23250

theorem marker_distance (k : ℝ) (h_pos : k > 0) : 
  (∀ n : ℕ, ∀ m : ℕ, m - n = 4 → 
    Real.sqrt ((m - n)^2 + (m*k - n*k)^2) = 31) →
  Real.sqrt ((19 - 7)^2 + (19*k - 7*k)^2) = 93 :=
by sorry

end marker_distance_l232_23250


namespace walking_rate_problem_l232_23296

/-- Proves that given the conditions of the problem, the walking rate when missing the train is 4 kmph -/
theorem walking_rate_problem (distance : ℝ) (early_rate : ℝ) (early_time : ℝ) (late_time : ℝ) :
  distance = 4 →
  early_rate = 5 →
  early_time = 6 →
  late_time = 6 →
  ∃ (late_rate : ℝ),
    (distance / early_rate) * 60 + early_time = (distance / late_rate) * 60 - late_time ∧
    late_rate = 4 :=
by sorry

end walking_rate_problem_l232_23296


namespace function_equality_implies_sum_l232_23298

theorem function_equality_implies_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 4) = 4 * x^2 + 9 * x + 5) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 14 := by
  sorry

end function_equality_implies_sum_l232_23298


namespace axis_of_symmetry_l232_23217

/-- The quadratic function f(x) = 3(x-1)^2 + 2 -/
def f (x : ℝ) : ℝ := 3 * (x - 1)^2 + 2

/-- The axis of symmetry for f(x) is x = 1 -/
theorem axis_of_symmetry (x : ℝ) : f (1 + x) = f (1 - x) := by sorry

end axis_of_symmetry_l232_23217


namespace sum_diff_parity_l232_23252

theorem sum_diff_parity (a b : ℤ) : Even (a + b) ↔ Even (a - b) := by
  sorry

end sum_diff_parity_l232_23252


namespace square_root_of_one_fourth_l232_23258

theorem square_root_of_one_fourth :
  {x : ℚ | x^2 = (1 : ℚ) / 4} = {(1 : ℚ) / 2, -(1 : ℚ) / 2} := by
  sorry

end square_root_of_one_fourth_l232_23258


namespace inequality_proof_l232_23235

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2/3 :=
by sorry

end inequality_proof_l232_23235


namespace perfect_cube_condition_l232_23280

/-- A polynomial x^3 + px^2 + qx + n is a perfect cube if and only if q = p^2 / 3 and n = p^3 / 27 -/
theorem perfect_cube_condition (p q n : ℝ) :
  (∃ a : ℝ, ∀ x : ℝ, x^3 + p*x^2 + q*x + n = (x + a)^3) ↔ 
  (q = p^2 / 3 ∧ n = p^3 / 27) :=
by sorry

end perfect_cube_condition_l232_23280


namespace unique_solution_linear_system_l232_23225

theorem unique_solution_linear_system :
  ∃! (x y : ℝ), x + y = 5 ∧ x + 2*y = 8 :=
by
  sorry

end unique_solution_linear_system_l232_23225


namespace positive_integer_solutions_count_l232_23269

theorem positive_integer_solutions_count : ∃ (n : ℕ), n = 10 ∧ 
  n = (Finset.filter (λ (x : ℕ × ℕ × ℕ) => 
    x.1 + x.2.1 + x.2.2 = 6 ∧ x.1 > 0 ∧ x.2.1 > 0 ∧ x.2.2 > 0) 
    (Finset.product (Finset.range 7) (Finset.product (Finset.range 7) (Finset.range 7)))).card :=
by
  sorry

end positive_integer_solutions_count_l232_23269


namespace sufficient_not_necessary_l232_23268

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, x > 1 → x^2 + x - 2 > 0) ∧ 
  (∃ x, x^2 + x - 2 > 0 ∧ x ≤ 1) :=
sorry

end sufficient_not_necessary_l232_23268


namespace oliver_candy_boxes_l232_23220

def candy_problem (morning_boxes afternoon_multiplier given_away : ℕ) : ℕ :=
  morning_boxes + afternoon_multiplier * morning_boxes - given_away

theorem oliver_candy_boxes :
  candy_problem 8 3 10 = 22 := by
  sorry

end oliver_candy_boxes_l232_23220


namespace ratio_average_l232_23203

theorem ratio_average (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b / a = 5 / 4 →
  c / a = 6 / 4 →
  c = 24 →
  (a + b + c) / 3 = 20 := by
sorry

end ratio_average_l232_23203


namespace circle_radius_range_l232_23211

/-- Given two circles O₁ and O₂, with O₁ having radius 1 and O₂ having radius r,
    and the distance between their centers being 5,
    if there exists a point P on O₂ such that PO₁ = 2,
    then the radius r of O₂ is between 3 and 7 inclusive. -/
theorem circle_radius_range (r : ℝ) :
  let O₁ : ℝ × ℝ := (0, 0)  -- Assuming O₁ is at the origin for simplicity
  let O₂ : ℝ × ℝ := (5, 0)  -- Assuming O₂ is on the x-axis
  ∃ (P : ℝ × ℝ), 
    (P.1 - O₂.1)^2 + P.2^2 = r^2 ∧  -- P is on circle O₂
    (P.1 - O₁.1)^2 + P.2^2 = 4      -- PO₁ = 2
  → 3 ≤ r ∧ r ≤ 7 :=
by sorry

end circle_radius_range_l232_23211


namespace lavender_candles_count_l232_23283

theorem lavender_candles_count (almond coconut lavender : ℕ) : 
  almond = 10 →
  coconut = (3 * almond) / 2 →
  lavender = 2 * coconut →
  lavender = 30 := by
sorry

end lavender_candles_count_l232_23283


namespace rectangle_tiling_tiling_count_lower_bound_l232_23260

/-- Represents a tiling of a rectangle -/
structure Tiling (m n : ℕ) :=
  (pieces : ℕ)
  (is_valid : Bool)

/-- The number of ways to tile a 5 × 2k rectangle with 2k pieces -/
def tiling_count (k : ℕ) : ℕ := sorry

theorem rectangle_tiling (n : ℕ) (t : Tiling 5 n) :
  t.pieces = n ∧ t.is_valid → Even n := by sorry

theorem tiling_count_lower_bound (k : ℕ) :
  k ≥ 3 → tiling_count k > 2 * 3^(k-1) := by sorry

end rectangle_tiling_tiling_count_lower_bound_l232_23260


namespace cube_sum_prime_power_l232_23204

theorem cube_sum_prime_power (a b p n : ℕ) : 
  0 < a ∧ 0 < b ∧ 0 < p ∧ 0 < n ∧ Nat.Prime p ∧ a^3 + b^3 = p^n →
  (∃ k : ℕ, 0 < k ∧
    ((a = 2^(k-1) ∧ b = 2^(k-1) ∧ p = 2 ∧ n = 3*k - 2) ∨
     (a = 2 * 3^(k-1) ∧ b = 3^(k-1) ∧ p = 3 ∧ n = 3*k - 1) ∨
     (a = 3^(k-1) ∧ b = 2 * 3^(k-1) ∧ p = 3 ∧ n = 3*k - 1))) :=
by
  sorry

end cube_sum_prime_power_l232_23204


namespace jason_car_count_l232_23218

theorem jason_car_count (purple : ℕ) (red : ℕ) (green : ℕ) : 
  purple = 47 →
  red = purple + 6 →
  green = 4 * red →
  purple + red + green = 312 := by
sorry

end jason_car_count_l232_23218


namespace probability_exactly_two_ones_equals_fraction_l232_23202

def num_dice : ℕ := 12
def num_sides : ℕ := 6
def target_outcome : ℕ := 1
def target_count : ℕ := 2

def probability_exactly_two_ones : ℚ :=
  (num_dice.choose target_count : ℚ) * 
  (1 / num_sides) ^ target_count * 
  ((num_sides - 1) / num_sides) ^ (num_dice - target_count)

theorem probability_exactly_two_ones_equals_fraction :
  probability_exactly_two_ones = (66 * 5^10 : ℚ) / (36 * 6^10) := by
  sorry

end probability_exactly_two_ones_equals_fraction_l232_23202


namespace planned_speed_calculation_l232_23215

theorem planned_speed_calculation (distance : ℝ) (speed_multiplier : ℝ) (time_saved : ℝ) 
  (h1 : distance = 180)
  (h2 : speed_multiplier = 1.2)
  (h3 : time_saved = 0.5)
  : ∃ v : ℝ, v > 0 ∧ distance / v = distance / (speed_multiplier * v) + time_saved ∧ v = 60 := by
  sorry

end planned_speed_calculation_l232_23215


namespace wheat_purchase_proof_l232_23278

/-- The cost of wheat in cents per pound -/
def wheat_cost : ℚ := 72

/-- The cost of oats in cents per pound -/
def oats_cost : ℚ := 36

/-- The total amount of wheat and oats bought in pounds -/
def total_amount : ℚ := 30

/-- The total amount spent in cents -/
def total_spent : ℚ := 1620

/-- The amount of wheat bought in pounds -/
def wheat_amount : ℚ := 15

theorem wheat_purchase_proof :
  ∃ (oats_amount : ℚ),
    wheat_amount + oats_amount = total_amount ∧
    wheat_cost * wheat_amount + oats_cost * oats_amount = total_spent :=
by sorry

end wheat_purchase_proof_l232_23278


namespace e_power_necessary_not_sufficient_for_ln_l232_23213

theorem e_power_necessary_not_sufficient_for_ln (x : ℝ) :
  (∃ y, (Real.exp y > 1 ∧ Real.log y ≥ 0)) ∧
  (∀ z, Real.log z < 0 → Real.exp z > 1) :=
sorry

end e_power_necessary_not_sufficient_for_ln_l232_23213


namespace tripod_height_is_2_sqrt_5_l232_23274

/-- A tripod with two legs of length 6 and one leg of length 4 -/
structure Tripod :=
  (leg1 : ℝ)
  (leg2 : ℝ)
  (leg3 : ℝ)
  (h : leg1 = 6)
  (i : leg2 = 6)
  (j : leg3 = 4)

/-- The height of the tripod when fully extended -/
def tripod_height (t : Tripod) : ℝ := sorry

/-- Theorem stating that the height of the tripod is 2√5 -/
theorem tripod_height_is_2_sqrt_5 (t : Tripod) : tripod_height t = 2 * Real.sqrt 5 := by sorry

end tripod_height_is_2_sqrt_5_l232_23274


namespace division_problem_l232_23266

theorem division_problem (x y q : ℕ) : 
  y - x = 1375 →
  y = 1632 →
  y = q * x + 15 →
  q = 6 := by
sorry

end division_problem_l232_23266


namespace different_meal_combinations_l232_23216

theorem different_meal_combinations (n : Nat) (h : n = 12) :
  (n * (n - 1) : Nat) = 132 := by
  sorry

end different_meal_combinations_l232_23216


namespace second_car_speed_l232_23200

/-- 
Given two cars starting from opposite ends of a 105-mile highway, 
with one car traveling at 15 mph and both meeting after 3 hours, 
prove that the speed of the second car is 20 mph.
-/
theorem second_car_speed 
  (highway_length : ℝ) 
  (first_car_speed : ℝ) 
  (meeting_time : ℝ) 
  (h1 : highway_length = 105) 
  (h2 : first_car_speed = 15) 
  (h3 : meeting_time = 3) : 
  ∃ (second_car_speed : ℝ), 
    first_car_speed * meeting_time + second_car_speed * meeting_time = highway_length ∧ 
    second_car_speed = 20 := by
  sorry

end second_car_speed_l232_23200


namespace unique_solution_implies_a_eq_one_g_monotone_increasing_sum_of_zeros_lt_two_l232_23276

noncomputable section

variables (x : ℝ) (p : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := a - 1/x - Real.log x

def g (x : ℝ) (p : ℝ) : ℝ := Real.log x - 2*(x-p)/(x+p) - Real.log p

theorem unique_solution_implies_a_eq_one :
  (∃! x, f 1 x = 0) → 1 = 1 := by sorry

theorem g_monotone_increasing (hp : p > 0) :
  Monotone (g · p) := by sorry

theorem sum_of_zeros_lt_two :
  ∃ x₁ x₂, x₁ < x₂ ∧ f 1 x₁ = 0 ∧ f 1 x₂ = 0 → x₁ + x₂ < 2 := by sorry

end

end unique_solution_implies_a_eq_one_g_monotone_increasing_sum_of_zeros_lt_two_l232_23276


namespace unique_divisible_by_nine_l232_23281

theorem unique_divisible_by_nine : ∃! x : ℕ, 
  x ≥ 0 ∧ x ≤ 9 ∧ (13800 + x * 10 + 6) % 9 = 0 := by sorry

end unique_divisible_by_nine_l232_23281


namespace conference_duration_theorem_l232_23255

/-- The duration of the conference in minutes -/
def conference_duration (first_session_hours : ℕ) (first_session_minutes : ℕ) 
  (second_session_hours : ℕ) (second_session_minutes : ℕ) : ℕ :=
  (first_session_hours * 60 + first_session_minutes) + 
  (second_session_hours * 60 + second_session_minutes)

/-- Theorem stating the total duration of the conference -/
theorem conference_duration_theorem : 
  conference_duration 8 15 3 40 = 715 := by sorry

end conference_duration_theorem_l232_23255


namespace quadratic_solution_sum_l232_23263

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 3 * x^2 + 4 = 5 * x - 7 ↔ x = a + b * I ∨ x = a - b * I) →
  a + b^2 = 137 / 36 := by
sorry

end quadratic_solution_sum_l232_23263


namespace cashback_less_profitable_l232_23248

structure Bank where
  forecasted_cashback : ℝ
  actual_cashback : ℝ

structure Customer where
  is_savvy : Bool
  cashback_optimized : ℝ

def cashback_program (b : Bank) (customers : List Customer) : Prop :=
  b.actual_cashback > b.forecasted_cashback

def growing_savvy_customers (customers : List Customer) : Prop :=
  (customers.filter (λ c => c.is_savvy)).length > 
  (customers.filter (λ c => !c.is_savvy)).length

theorem cashback_less_profitable 
  (b : Bank) 
  (customers : List Customer) 
  (h1 : growing_savvy_customers customers) 
  (h2 : ∀ c ∈ customers, c.is_savvy → c.cashback_optimized > 0) :
  cashback_program b customers :=
by
  sorry

#check cashback_less_profitable

end cashback_less_profitable_l232_23248


namespace parallel_lines_corresponding_angles_not_always_equal_l232_23284

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of a transversal
def transversal (t l1 l2 : Line) : Prop := sorry

-- Define alternate interior angles
def alternateInteriorAngles (a1 a2 : Angle) (l1 l2 t : Line) : Prop := sorry

-- Define corresponding angles
def correspondingAngles (a1 a2 : Angle) (l1 l2 t : Line) : Prop := sorry

-- Theorem statement
theorem parallel_lines_corresponding_angles_not_always_equal :
  ∃ (l1 l2 t : Line) (a1 a2 : Angle),
    parallel l1 l2 ∧
    transversal t l1 l2 ∧
    correspondingAngles a1 a2 l1 l2 t ∧
    a1 ≠ a2 :=
  sorry

end parallel_lines_corresponding_angles_not_always_equal_l232_23284


namespace first_platform_length_is_150_l232_23256

/-- The length of a train in meters -/
def train_length : ℝ := 150

/-- The length of the second platform in meters -/
def second_platform_length : ℝ := 250

/-- The time taken to cross the first platform in seconds -/
def time_first_platform : ℝ := 15

/-- The time taken to cross the second platform in seconds -/
def time_second_platform : ℝ := 20

/-- The length of the first platform in meters -/
def first_platform_length : ℝ := 150

theorem first_platform_length_is_150 :
  (train_length + first_platform_length) / time_first_platform =
  (train_length + second_platform_length) / time_second_platform :=
sorry

end first_platform_length_is_150_l232_23256


namespace petya_wins_l232_23261

/-- Represents the game state -/
structure GameState where
  stones : ℕ
  playerTurn : Bool  -- true for Petya, false for Vasya

/-- Defines a valid move in the game -/
def validMove (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : ℕ) : GameState :=
  { stones := state.stones - move, playerTurn := ¬state.playerTurn }

/-- Determines if the game is over -/
def gameOver (state : GameState) : Prop := state.stones = 0

/-- Defines a winning strategy for the first player -/
def winningStrategy (strategy : GameState → ℕ) : Prop :=
  ∀ (state : GameState), 
    validMove (strategy state) ∧ 
    (gameOver (applyMove state (strategy state)) ∨ 
     ∀ (opponentMove : ℕ), validMove opponentMove → 
       ¬gameOver (applyMove (applyMove state (strategy state)) opponentMove))

/-- Theorem: The first player (Petya) has a winning strategy -/
theorem petya_wins : 
  ∃ (strategy : GameState → ℕ), winningStrategy strategy ∧ 
    (∀ (state : GameState), state.stones = 111 ∧ state.playerTurn = true → 
      gameOver (applyMove state (strategy state)) ∨ 
      ∀ (opponentMove : ℕ), validMove opponentMove → 
        ¬gameOver (applyMove (applyMove state (strategy state)) opponentMove)) :=
sorry

end petya_wins_l232_23261


namespace smallest_covering_segment_l232_23288

/-- Represents an equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_unit : side_length = 1

/-- Represents a sliding segment in the triangle -/
structure SlidingSegment where
  length : ℝ
  covers_triangle : Prop

/-- The smallest sliding segment that covers the triangle has length 2/3 -/
theorem smallest_covering_segment (triangle : EquilateralTriangle) :
  ∃ (d : ℝ), d = 2/3 ∧ 
  (∀ (s : SlidingSegment), s.covers_triangle → s.length ≥ d) ∧
  (∃ (s : SlidingSegment), s.covers_triangle ∧ s.length = d) :=
sorry

end smallest_covering_segment_l232_23288


namespace cos_alpha_cos_2alpha_distinct_digits_l232_23271

/-- Represents a repeating decimal of the form 0.aḃ -/
def repeating_decimal (a b : ℕ) : ℚ :=
  (10 * a + b) / 90

theorem cos_alpha_cos_2alpha_distinct_digits :
  ∃! (a b c d : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    repeating_decimal a b = Real.cos α ∧
    repeating_decimal c d = -Real.cos (2 * α) ∧
    a = 1 ∧ b = 6 ∧ c = 9 ∧ d = 4 :=
by
  sorry

end cos_alpha_cos_2alpha_distinct_digits_l232_23271


namespace gas_price_calculation_l232_23264

/-- Proves that the actual cost of gas per gallon is $1.80 given the problem conditions -/
theorem gas_price_calculation (expected_price : ℝ) : 
  (12 * expected_price = 10 * (expected_price + 0.3)) → 
  (expected_price + 0.3 = 1.8) := by
  sorry

#check gas_price_calculation

end gas_price_calculation_l232_23264


namespace geometric_sequence_sixth_term_l232_23210

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_roots : 3 * (a 3)^2 - 11 * (a 3) + 9 = 0 ∧ 3 * (a 9)^2 - 11 * (a 9) + 9 = 0) :
  (a 6)^2 = 3 :=
sorry

end geometric_sequence_sixth_term_l232_23210


namespace c_share_correct_l232_23277

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_share (total_profit : ℕ) (investments : List ℕ) (partner_index : ℕ) : ℕ :=
  sorry

theorem c_share_correct (total_profit : ℕ) (investments : List ℕ) :
  total_profit = 90000 →
  investments = [30000, 45000, 50000] →
  calculate_share total_profit investments 2 = 36000 :=
by sorry

end c_share_correct_l232_23277


namespace triangle_positive_number_placement_l232_23236

theorem triangle_positive_number_placement 
  (A B C : ℝ × ℝ) -- Vertices of the triangle
  (AB BC CA : ℝ)  -- Lengths of the sides
  (h_pos_AB : AB > 0)
  (h_pos_BC : BC > 0)
  (h_pos_CA : CA > 0)
  (h_triangle : AB + BC > CA ∧ BC + CA > AB ∧ CA + AB > BC) -- Triangle inequality
  : ∃ x y z : ℝ, 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    AB = x + y ∧
    BC = y + z ∧
    CA = z + x :=
sorry

end triangle_positive_number_placement_l232_23236


namespace perimeter_of_figure_l232_23251

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A B C : Point)

/-- A square defined by four points -/
structure Square :=
  (E H I J : Point)

/-- The figure ABCDEFGHIJ -/
structure Figure :=
  (A B C D E F G H I J : Point)

/-- Definition of an equilateral triangle -/
def isEquilateral (t : Triangle) : Prop :=
  sorry

/-- Definition of a midpoint -/
def isMidpoint (M A B : Point) : Prop :=
  sorry

/-- Definition of a square -/
def isSquare (s : Square) : Prop :=
  sorry

/-- Distance between two points -/
def distance (A B : Point) : ℝ :=
  sorry

/-- Perimeter of the figure -/
def perimeter (fig : Figure) : ℝ :=
  sorry

/-- Main theorem -/
theorem perimeter_of_figure (fig : Figure) :
  isEquilateral ⟨fig.A, fig.B, fig.C⟩ →
  isEquilateral ⟨fig.A, fig.D, fig.E⟩ →
  isEquilateral ⟨fig.E, fig.F, fig.G⟩ →
  isMidpoint fig.D fig.A fig.C →
  isMidpoint fig.G fig.A fig.E →
  isSquare ⟨fig.E, fig.H, fig.I, fig.J⟩ →
  distance fig.E fig.J = distance fig.D fig.E →
  distance fig.A fig.B = 6 →
  perimeter fig = 37.5 := by
  sorry

end perimeter_of_figure_l232_23251


namespace unique_solution_l232_23224

/-- The polynomial P(x) = 3x^4 + ax^3 + bx^2 - 16x + 55 -/
def P (a b x : ℝ) : ℝ := 3 * x^4 + a * x^3 + b * x^2 - 16 * x + 55

/-- The first divisibility condition -/
def condition1 (a b : ℝ) : Prop :=
  P a b (-4/3) = 23

/-- The second divisibility condition -/
def condition2 (a b : ℝ) : Prop :=
  P a b 3 = 10

theorem unique_solution :
  ∃! (a b : ℝ), condition1 a b ∧ condition2 a b ∧ a = -29 ∧ b = 7 := by
  sorry

end unique_solution_l232_23224


namespace square_side_length_range_l232_23293

theorem square_side_length_range (area : ℝ) (h : area = 15) :
  ∃ x : ℝ, x > 3 ∧ x < 4 ∧ x^2 = area := by
  sorry

end square_side_length_range_l232_23293


namespace female_workers_l232_23290

/-- Represents the number of workers in a company --/
structure Company where
  male : ℕ
  female : ℕ
  male_no_plan : ℕ
  female_no_plan : ℕ

/-- The conditions of the company --/
def company_conditions (c : Company) : Prop :=
  c.male = 112 ∧
  c.male_no_plan = (40 * c.male) / 100 ∧
  c.female_no_plan = (25 * c.female) / 100 ∧
  (30 * (c.male_no_plan + c.female_no_plan)) / 100 = c.male_no_plan ∧
  (60 * (c.male - c.male_no_plan + c.female - c.female_no_plan)) / 100 = (c.male - c.male_no_plan)

/-- The theorem to be proved --/
theorem female_workers (c : Company) : company_conditions c → c.female = 420 := by
  sorry

end female_workers_l232_23290


namespace rocket_altitude_time_rocket_minimum_time_l232_23226

/-- Represents the distance covered by the rocket in a given second -/
def distance_at_second (n : ℕ) : ℝ := 2 + 2 * (n - 1)

/-- Represents the total distance covered by the rocket after n seconds -/
def total_distance (n : ℕ) : ℝ := (n : ℝ) * 2 + (n : ℝ) * ((n : ℝ) - 1)

/-- The theorem stating that the rocket reaches 240 km altitude in 15 seconds -/
theorem rocket_altitude_time : total_distance 15 = 240 := by
  sorry

/-- The theorem stating that 15 is the minimum time to reach 240 km -/
theorem rocket_minimum_time (n : ℕ) : 
  n < 15 → total_distance n < 240 := by
  sorry

end rocket_altitude_time_rocket_minimum_time_l232_23226


namespace chess_tournament_solution_l232_23230

/-- Chess tournament with n women and 2n men -/
structure ChessTournament (n : ℕ) where
  women : Fin n
  men : Fin (2 * n)

/-- The number of games played in the tournament -/
def total_games (n : ℕ) : ℕ :=
  n * (3 * n - 1) / 2

/-- The number of games won by women -/
def women_wins (n : ℕ) : ℚ :=
  (n * (n - 1) / 2) + (17 * n^2 - 3 * n) / 8

/-- The number of games won by men -/
def men_wins (n : ℕ) : ℚ :=
  (n * (2 * n - 1)) + (3 * n / 8)

/-- The theorem stating that n must equal 3 -/
theorem chess_tournament_solution : 
  ∃ (n : ℕ), n > 0 ∧ 
  7 * (men_wins n) = 5 * (women_wins n) ∧
  (women_wins n).isInt ∧ (men_wins n).isInt :=
by
  sorry

end chess_tournament_solution_l232_23230


namespace compare_fractions_l232_23285

theorem compare_fractions : -8 / 21 > -3 / 7 := by
  sorry

end compare_fractions_l232_23285


namespace slower_train_speed_l232_23238

theorem slower_train_speed
  (train_length : ℝ)
  (faster_train_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 250)
  (h2 : faster_train_speed = 45)
  (h3 : passing_time = 23.998080153587715) :
  ∃ (slower_train_speed : ℝ),
    slower_train_speed = 30 ∧
    slower_train_speed + faster_train_speed = (2 * train_length / 1000) / (passing_time / 3600) :=
by sorry

end slower_train_speed_l232_23238


namespace johns_average_speed_l232_23295

/-- Proves that John's average speed was 40 miles per hour given the conditions of the problem -/
theorem johns_average_speed 
  (john_time : ℝ) 
  (beth_route_difference : ℝ)
  (beth_time_difference : ℝ)
  (beth_speed : ℝ)
  (h1 : john_time = 30 / 60) -- John's time in hours
  (h2 : beth_route_difference = 5) -- Beth's route was 5 miles longer
  (h3 : beth_time_difference = 20 / 60) -- Beth's additional time in hours
  (h4 : beth_speed = 30) -- Beth's average speed in miles per hour
  : (beth_speed * (john_time + beth_time_difference) - beth_route_difference) / john_time = 40 :=
by sorry

end johns_average_speed_l232_23295


namespace dinners_sold_in_four_days_l232_23228

/-- Calculates the total number of dinners sold over 4 days given specific sales patterns. -/
def total_dinners_sold (monday : ℕ) : ℕ :=
  let tuesday := monday + 40
  let wednesday := tuesday / 2
  let thursday := wednesday + 3
  monday + tuesday + wednesday + thursday

/-- Theorem stating that given the specific sales pattern, 203 dinners were sold over 4 days. -/
theorem dinners_sold_in_four_days : total_dinners_sold 40 = 203 := by
  sorry

end dinners_sold_in_four_days_l232_23228


namespace problem_statement_l232_23243

theorem problem_statement (x y : ℝ) (h : x + y = 1) :
  (x^2 + 3*y^2 ≥ 3/4) ∧
  (x*y > 0 → ∀ a : ℝ, a ≤ 5/2 → 1/x + 1/y ≥ |a - 2| + |a + 1|) ∧
  (∀ a : ℝ, (x*y > 0 → 1/x + 1/y ≥ |a - 2| + |a + 1|) ↔ a ≤ 5/2) := by
  sorry

end problem_statement_l232_23243


namespace line_tangent_to_parabola_l232_23247

/-- A line y = 3x + d is tangent to the parabola y^2 = 12x if and only if d = 1 -/
theorem line_tangent_to_parabola (d : ℝ) : 
  (∃! x : ℝ, (3 * x + d)^2 = 12 * x) ↔ d = 1 :=
sorry

end line_tangent_to_parabola_l232_23247


namespace maximize_product_l232_23292

theorem maximize_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 40) :
  x^3 * y^4 ≤ (160/7)^3 * (120/7)^4 ∧
  (x^3 * y^4 = (160/7)^3 * (120/7)^4 ↔ x = 160/7 ∧ y = 120/7) :=
by sorry

end maximize_product_l232_23292
