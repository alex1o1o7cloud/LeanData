import Mathlib

namespace range_of_m_l3585_358526

-- Define p and q as predicates on real numbers
def p (x : ℝ) : Prop := |x - 3| ≤ 2
def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

-- Define the theorem
theorem range_of_m :
  (∀ x m : ℝ, (¬(p x) → ¬(q x m)) ∧ ¬(¬(q x m) → ¬(p x))) →
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ ∀ m : ℝ, a ≤ m ∧ m ≤ b :=
by sorry

end range_of_m_l3585_358526


namespace solution_set_of_inequality_l3585_358561

theorem solution_set_of_inequality (x : ℝ) :
  (1 ≤ |x + 2| ∧ |x + 2| ≤ 5) ↔ ((-7 ≤ x ∧ x ≤ -3) ∨ (-1 ≤ x ∧ x ≤ 3)) :=
by sorry

end solution_set_of_inequality_l3585_358561


namespace algebraic_expression_value_l3585_358501

theorem algebraic_expression_value
  (m n p q x : ℝ)
  (h1 : m = -n)
  (h2 : p * q = 1)
  (h3 : |x| = 2) :
  (m + n) / 2022 + 2023 * p * q + x^2 = 2027 :=
by sorry

end algebraic_expression_value_l3585_358501


namespace flower_production_percentage_l3585_358556

theorem flower_production_percentage
  (daisy_seeds sunflower_seeds : ℕ)
  (daisy_germination_rate sunflower_germination_rate : ℚ)
  (flowering_plants : ℕ)
  (h1 : daisy_seeds = 25)
  (h2 : sunflower_seeds = 25)
  (h3 : daisy_germination_rate = 0.6)
  (h4 : sunflower_germination_rate = 0.8)
  (h5 : flowering_plants = 28)
  : (flowering_plants : ℚ) / ((daisy_germination_rate * daisy_seeds + sunflower_germination_rate * sunflower_seeds) : ℚ) = 0.8 := by
  sorry

end flower_production_percentage_l3585_358556


namespace area_of_rectangle_with_squares_l3585_358558

/-- A rectangle divided into four identical squares with a given perimeter -/
structure RectangleWithSquares where
  side_length : ℝ
  perimeter : ℝ
  perimeter_eq : perimeter = 8 * side_length

/-- The area of a rectangle divided into four identical squares -/
def area (r : RectangleWithSquares) : ℝ :=
  4 * r.side_length^2

/-- Theorem: A rectangle divided into four identical squares with a perimeter of 160 has an area of 1600 -/
theorem area_of_rectangle_with_squares (r : RectangleWithSquares) (h : r.perimeter = 160) :
  area r = 1600 := by
  sorry

#check area_of_rectangle_with_squares

end area_of_rectangle_with_squares_l3585_358558


namespace sum_of_powers_of_i_l3585_358534

theorem sum_of_powers_of_i : 
  let i : ℂ := Complex.I
  (i + 2 * i^2 + 3 * i^3 + 4 * i^4 + 5 * i^5 + 6 * i^6 + 7 * i^7 + 8 * i^8) = (4 : ℂ) - 4 * i :=
by sorry

end sum_of_powers_of_i_l3585_358534


namespace total_eggs_collected_l3585_358555

/-- The number of dozen eggs collected by Benjamin, Carla, Trisha, and David -/
def total_eggs (benjamin carla trisha david : ℕ) : ℕ :=
  benjamin + carla + trisha + david

/-- Theorem stating the total number of dozen eggs collected -/
theorem total_eggs_collected :
  ∃ (benjamin carla trisha david : ℕ),
    benjamin = 6 ∧
    carla = 3 * benjamin ∧
    trisha = benjamin - 4 ∧
    david = 2 * trisha ∧
    total_eggs benjamin carla trisha david = 30 := by
  sorry

end total_eggs_collected_l3585_358555


namespace point_transformation_difference_l3585_358527

-- Define the rotation and reflection transformations
def rotate90CounterClockwise (center x : ℝ × ℝ) : ℝ × ℝ :=
  let (cx, cy) := center
  let (px, py) := x
  (cx - (py - cy), cy + (px - cx))

def reflectAboutYEqualNegX (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-y, -x)

-- State the theorem
theorem point_transformation_difference (a b : ℝ) :
  let p : ℝ × ℝ := (a, b)
  let rotated := rotate90CounterClockwise (2, 6) p
  let final := reflectAboutYEqualNegX rotated
  final = (-5, 2) → b - a = 15 := by
  sorry


end point_transformation_difference_l3585_358527


namespace library_visitors_l3585_358551

theorem library_visitors (non_sunday_avg : ℕ) (monthly_avg : ℕ) (month_days : ℕ) (sundays : ℕ) :
  non_sunday_avg = 240 →
  monthly_avg = 285 →
  month_days = 30 →
  sundays = 5 →
  (sundays * (monthly_avg * month_days - non_sunday_avg * (month_days - sundays))) / sundays = 510 := by
  sorry

end library_visitors_l3585_358551


namespace stewart_farm_sheep_count_l3585_358596

theorem stewart_farm_sheep_count :
  -- Definitions
  let sheep_horse_cow_ratio : Fin 3 → ℕ := ![4, 7, 5]
  let food_per_animal : Fin 3 → ℕ := ![150, 230, 300]
  let total_food : Fin 3 → ℕ := ![9750, 12880, 15000]

  -- Conditions
  ∀ (num_animals : Fin 3 → ℕ),
    (∀ i : Fin 3, num_animals i * food_per_animal i = total_food i) →
    (∀ i j : Fin 3, num_animals i * sheep_horse_cow_ratio j = num_animals j * sheep_horse_cow_ratio i) →

  -- Conclusion
  num_animals 0 = 98 :=
by
  sorry

end stewart_farm_sheep_count_l3585_358596


namespace olympiad_participants_impossibility_l3585_358554

theorem olympiad_participants_impossibility : ¬ ∃ (x : ℕ), x + (x + 43) = 1000 := by
  sorry

end olympiad_participants_impossibility_l3585_358554


namespace arithmetic_sequence_30th_term_l3585_358513

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 30th term of the given arithmetic sequence is 351. -/
theorem arithmetic_sequence_30th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_a1 : a 1 = 3)
  (h_a2 : a 2 = 15)
  (h_a3 : a 3 = 27) :
  a 30 = 351 :=
sorry

end arithmetic_sequence_30th_term_l3585_358513


namespace car_speed_time_relations_l3585_358584

/-- Represents the speed and time of a car --/
structure CarData where
  speed : ℝ
  time : ℝ

/-- Given conditions and proof goals for the car problem --/
theorem car_speed_time_relations 
  (x y z : CarData) 
  (h1 : y.speed = 3 * x.speed) 
  (h2 : z.speed = (x.speed + y.speed) / 2) 
  (h3 : x.speed * x.time = y.speed * y.time) 
  (h4 : x.speed * x.time = z.speed * z.time) : 
  z.speed = 2 * x.speed ∧ 
  y.time = x.time / 3 ∧ 
  z.time = x.time / 2 := by
  sorry


end car_speed_time_relations_l3585_358584


namespace paint_coverage_l3585_358524

/-- Given a cube with 10-foot edges, if it costs $16 to paint the entire outside surface
    of the cube and paint costs $3.20 per quart, then one quart of paint covers 120 square feet. -/
theorem paint_coverage (cube_edge : Real) (total_cost : Real) (paint_cost_per_quart : Real) :
  cube_edge = 10 ∧ total_cost = 16 ∧ paint_cost_per_quart = 3.2 →
  (6 * cube_edge^2) / (total_cost / paint_cost_per_quart) = 120 := by
  sorry

end paint_coverage_l3585_358524


namespace f_5_equals_102_l3585_358536

def f (x y : ℝ) : ℝ := 2 * x^2 + y

theorem f_5_equals_102 (y : ℝ) (some_value : ℝ) :
  f some_value y = 60 →
  f 5 y = 102 →
  f 5 y = 102 := by
  sorry

end f_5_equals_102_l3585_358536


namespace first_duck_ate_half_l3585_358583

/-- The fraction of bread eaten by the first duck -/
def first_duck_fraction (total_bread pieces_left second_duck_pieces third_duck_pieces : ℕ) : ℚ :=
  let eaten := total_bread - pieces_left
  let first_duck_pieces := eaten - (second_duck_pieces + third_duck_pieces)
  first_duck_pieces / total_bread

/-- Theorem stating the fraction of bread eaten by the first duck -/
theorem first_duck_ate_half :
  first_duck_fraction 100 30 13 7 = 1/2 := by
  sorry

#eval first_duck_fraction 100 30 13 7

end first_duck_ate_half_l3585_358583


namespace min_value_of_angle_sum_l3585_358567

theorem min_value_of_angle_sum (α β : Real) : 
  α > 0 → β > 0 → α + β = π / 2 → (4 / α + 1 / β ≥ 18 / π) :=
by sorry

end min_value_of_angle_sum_l3585_358567


namespace mississippi_permutations_count_l3585_358537

/-- The number of unique permutations of MISSISSIPPI -/
def mississippi_permutations : ℕ :=
  Nat.factorial 11 / (Nat.factorial 1 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 2)

/-- Theorem stating that the number of unique permutations of MISSISSIPPI is 34650 -/
theorem mississippi_permutations_count : mississippi_permutations = 34650 := by
  sorry

end mississippi_permutations_count_l3585_358537


namespace power_subtraction_division_l3585_358514

theorem power_subtraction_division (n : ℕ) : 1^567 - 3^8 / 3^5 = -26 := by
  sorry

end power_subtraction_division_l3585_358514


namespace p_necessary_not_sufficient_for_q_l3585_358516

-- Define the conditions
def p (x : ℝ) : Prop := x < -1 ∨ x > 1
def q (x : ℝ) : Prop := x < -2

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by
  sorry

end p_necessary_not_sufficient_for_q_l3585_358516


namespace book_and_painting_participants_l3585_358525

/-- Represents the number of participants in various activities and their intersections --/
structure ActivityParticipants where
  total : ℕ
  book_club : ℕ
  fun_sports : ℕ
  env_painting : ℕ
  book_and_sports : ℕ
  sports_and_painting : ℕ

/-- Theorem stating the number of participants in both Book Club and Environmental Theme Painting --/
theorem book_and_painting_participants (ap : ActivityParticipants)
  (h_total : ap.total = 120)
  (h_book : ap.book_club = 80)
  (h_sports : ap.fun_sports = 50)
  (h_painting : ap.env_painting = 40)
  (h_book_sports : ap.book_and_sports = 20)
  (h_sports_painting : ap.sports_and_painting = 10)
  (h_max_two : ∀ p, p ≤ 2) :
  ap.book_club + ap.fun_sports + ap.env_painting - ap.total - ap.book_and_sports - ap.sports_and_painting = 20 :=
by sorry

end book_and_painting_participants_l3585_358525


namespace sqrt_four_squared_times_five_to_sixth_l3585_358529

theorem sqrt_four_squared_times_five_to_sixth : Real.sqrt (4^2 * 5^6) = 500 := by
  sorry

end sqrt_four_squared_times_five_to_sixth_l3585_358529


namespace min_games_for_20_teams_l3585_358506

/-- Represents a football tournament --/
structure Tournament where
  num_teams : ℕ
  num_games : ℕ

/-- Checks if a tournament satisfies the condition that among any three teams, 
    two have played against each other --/
def satisfies_condition (t : Tournament) : Prop :=
  ∀ (a b c : Fin t.num_teams), a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    (∃ (x y : Fin t.num_teams), x ≠ y ∧ ((x = a ∧ y = b) ∨ (x = b ∧ y = c) ∨ (x = a ∧ y = c)))

/-- The main theorem stating the minimum number of games required --/
theorem min_games_for_20_teams : 
  ∃ (t : Tournament), t.num_teams = 20 ∧ t.num_games = 90 ∧ 
    satisfies_condition t ∧ 
    (∀ (t' : Tournament), t'.num_teams = 20 ∧ satisfies_condition t' → t'.num_games ≥ 90) :=
sorry

end min_games_for_20_teams_l3585_358506


namespace find_A_value_l3585_358515

theorem find_A_value : ∃! A : ℕ, ∃ B : ℕ, 
  (A < 10 ∧ B < 10) ∧ 
  (500 + 10 * A + 8) - (100 * B + 14) = 364 :=
by
  -- The proof would go here
  sorry

end find_A_value_l3585_358515


namespace binomial_expansion_theorem_l3585_358500

def binomial_coefficient (n k : ℕ) : ℕ := sorry

def binomial_expansion_coefficient (a : ℝ) : ℝ :=
  (-a) * binomial_coefficient 9 1

theorem binomial_expansion_theorem (a : ℝ) :
  binomial_expansion_coefficient a = 36 → a = -4 := by
  sorry

end binomial_expansion_theorem_l3585_358500


namespace sum_of_distances_is_ten_l3585_358585

/-- Given a circle tangent to the sides of an angle at points A and B, with a point C on the circle,
    this structure represents the distances and conditions of the problem. -/
structure CircleTangentProblem where
  -- Distance from C to line AB
  h : ℝ
  -- Distance from C to the side of the angle passing through A
  h_A : ℝ
  -- Distance from C to the side of the angle passing through B
  h_B : ℝ
  -- Condition: h is equal to 4
  h_eq_four : h = 4
  -- Condition: One distance is four times the other
  one_distance_four_times_other : h_B = 4 * h_A

/-- The theorem states that under the given conditions, the sum of distances h_A and h_B is 10. -/
theorem sum_of_distances_is_ten (p : CircleTangentProblem) : p.h_A + p.h_B = 10 := by
  sorry

end sum_of_distances_is_ten_l3585_358585


namespace unique_congruence_solution_l3585_358548

theorem unique_congruence_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n < 19 ∧ -200 ≡ n [ZMOD 19] ∧ n = 9 := by
  sorry

end unique_congruence_solution_l3585_358548


namespace total_skips_five_throws_l3585_358542

-- Define the skip function
def S (n : ℕ) : ℕ := n^2 + n

-- Define the sum of skips from 1 to n
def sum_skips (n : ℕ) : ℕ :=
  (List.range n).map S |> List.sum

-- Theorem statement
theorem total_skips_five_throws :
  sum_skips 5 = 70 := by sorry

end total_skips_five_throws_l3585_358542


namespace siblings_total_weight_l3585_358569

def total_weight (weight1 weight2 backpack1 backpack2 : ℕ) : ℕ :=
  weight1 + weight2 + backpack1 + backpack2

theorem siblings_total_weight :
  ∀ (antonio_weight antonio_sister_weight : ℕ),
    antonio_weight = 50 →
    antonio_sister_weight = antonio_weight - 12 →
    total_weight antonio_weight antonio_sister_weight 5 3 = 96 := by
  sorry

end siblings_total_weight_l3585_358569


namespace value_of_expression_l3585_358544

theorem value_of_expression (x y z : ℝ) 
  (h1 : 3 * x - 4 * y - 2 * z = 0)
  (h2 : x - 2 * y - 8 * z = 0)
  (h3 : z ≠ 0) :
  (x^2 + 3*x*y) / (y^2 + z^2) = 329/61 := by
  sorry

end value_of_expression_l3585_358544


namespace function_value_2023_l3585_358557

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The main theorem -/
theorem function_value_2023 (f : ℝ → ℝ) 
    (h_even : IsEven f)
    (h_not_zero : ∃ x, f x ≠ 0)
    (h_equation : ∀ x, x * f (x + 2) = (x + 2) * f x + 2) :
  f 2023 = -1 := by
sorry

end function_value_2023_l3585_358557


namespace plants_given_to_friend_l3585_358512

def initial_plants : ℕ := 3
def months : ℕ := 3
def remaining_plants : ℕ := 20

def plants_after_doubling (initial : ℕ) (months : ℕ) : ℕ :=
  initial * (2 ^ months)

theorem plants_given_to_friend :
  plants_after_doubling initial_plants months - remaining_plants = 4 := by
  sorry

end plants_given_to_friend_l3585_358512


namespace project_completion_time_l3585_358559

/-- Represents the project completion time given the conditions -/
theorem project_completion_time 
  (total_man_days : ℝ) 
  (initial_workers : ℕ) 
  (workers_left : ℕ) 
  (h1 : total_man_days = 200)
  (h2 : initial_workers = 10)
  (h3 : workers_left = 4)
  : ∃ (x : ℝ), x > 0 ∧ x + (total_man_days - initial_workers * x) / (initial_workers - workers_left) = 40 := by
  sorry

#check project_completion_time

end project_completion_time_l3585_358559


namespace function_continuity_l3585_358523

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that f(x) + f(ax) is continuous for any a > 1
def condition (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, a > 1 → Continuous (fun x ↦ f x + f (a * x))

-- State the theorem
theorem function_continuity (hf : condition f) : Continuous f := by
  sorry

end function_continuity_l3585_358523


namespace minimum_cubes_required_l3585_358589

/-- Represents a 3D grid of unit cubes -/
def CubeGrid := List (List (List Bool))

/-- Checks if a cube in the grid shares at least one face with another cube -/
def sharesface (grid : CubeGrid) : Bool :=
  sorry

/-- Generates the front view of the grid -/
def frontView (grid : CubeGrid) : List (List Bool) :=
  sorry

/-- Generates the side view of the grid -/
def sideView (grid : CubeGrid) : List (List Bool) :=
  sorry

/-- The given front view -/
def givenFrontView : List (List Bool) :=
  [[true, true, false],
   [true, true, false],
   [true, false, false]]

/-- The given side view -/
def givenSideView : List (List Bool) :=
  [[true, true, true, false],
   [false, true, false, false],
   [false, false, true, false]]

/-- Counts the number of cubes in the grid -/
def countCubes (grid : CubeGrid) : Nat :=
  sorry

theorem minimum_cubes_required :
  ∃ (grid : CubeGrid),
    sharesface grid ∧
    frontView grid = givenFrontView ∧
    sideView grid = givenSideView ∧
    countCubes grid = 5 ∧
    (∀ (other : CubeGrid),
      sharesface other →
      frontView other = givenFrontView →
      sideView other = givenSideView →
      countCubes other ≥ 5) :=
  sorry

end minimum_cubes_required_l3585_358589


namespace toucan_count_l3585_358571

theorem toucan_count (initial_toucans : ℕ) (joining_toucans : ℕ) : 
  initial_toucans = 2 → joining_toucans = 1 → initial_toucans + joining_toucans = 3 := by
  sorry

end toucan_count_l3585_358571


namespace unique_digit_arrangement_l3585_358552

/-- A type representing digits from 1 to 9 -/
inductive Digit : Type
  | one : Digit
  | two : Digit
  | three : Digit
  | four : Digit
  | five : Digit
  | six : Digit
  | seven : Digit
  | eight : Digit
  | nine : Digit

/-- Convert a Digit to a natural number -/
def digit_to_nat (d : Digit) : ℕ :=
  match d with
  | Digit.one => 1
  | Digit.two => 2
  | Digit.three => 3
  | Digit.four => 4
  | Digit.five => 5
  | Digit.six => 6
  | Digit.seven => 7
  | Digit.eight => 8
  | Digit.nine => 9

/-- Convert three Digits to a natural number -/
def three_digit_to_nat (e f g : Digit) : ℕ :=
  100 * (digit_to_nat e) + 10 * (digit_to_nat f) + (digit_to_nat g)

/-- The main theorem -/
theorem unique_digit_arrangement :
  ∃! (a b c d e f g h : Digit),
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧ (a ≠ h) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧ (b ≠ h) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧ (c ≠ h) ∧
    (d ≠ e) ∧ (d ≠ f) ∧ (d ≠ g) ∧ (d ≠ h) ∧
    (e ≠ f) ∧ (e ≠ g) ∧ (e ≠ h) ∧
    (f ≠ g) ∧ (f ≠ h) ∧
    (g ≠ h) ∧
    ((digit_to_nat a) / (digit_to_nat b) = (digit_to_nat c) / (digit_to_nat d)) ∧
    ((digit_to_nat c) / (digit_to_nat d) = (three_digit_to_nat e f g) / (10 * (digit_to_nat h) + 9)) :=
by sorry


end unique_digit_arrangement_l3585_358552


namespace triangle_base_measurement_l3585_358568

/-- Given a triangular shape with height 20 cm, if the total area of three similar such shapes is 1200 cm², then the base of each triangle is 40 cm. -/
theorem triangle_base_measurement (height : ℝ) (total_area : ℝ) : 
  height = 20 → total_area = 1200 → ∃ (base : ℝ), base = 40 ∧ 3 * (base * height / 2) = total_area := by
  sorry

end triangle_base_measurement_l3585_358568


namespace triangle_properties_l3585_358521

/-- An acute triangle with side lengths a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

/-- The main theorem about the specific triangle -/
theorem triangle_properties (t : AcuteTriangle)
    (h1 : t.a = 2)
    (h2 : 2 * Real.sin t.A = Real.sin t.C)
    (h3 : Real.cos t.C = 1/4) :
    t.c = 4 ∧ (1/2 * t.a * t.b * Real.sin t.C) = Real.sqrt 15 := by
  sorry

end triangle_properties_l3585_358521


namespace question_one_l3585_358504

theorem question_one (a b : ℚ) : |a| = 3 ∧ |b| = 1 ∧ a < b → a + b = -2 ∨ a + b = -4 := by
  sorry

end question_one_l3585_358504


namespace parabola_smallest_a_l3585_358564

/-- Given a parabola with vertex (1/2, -5/4), equation y = ax^2 + bx + c,
    a > 0, and directrix y = -2, prove that the smallest possible value of a is 2/3 -/
theorem parabola_smallest_a (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Equation of parabola
  (a > 0) →                               -- a is positive
  (∀ x : ℝ, a * (x - 1/2)^2 - 5/4 = a * x^2 + b * x + c) →  -- Vertex form
  (∀ x : ℝ, -2 = a * x^2 + b * x + c - 3/4 * (1/a)) →       -- Directrix equation
  a = 2/3 :=
by sorry

end parabola_smallest_a_l3585_358564


namespace lab_capacity_l3585_358598

theorem lab_capacity (total_capacity : ℕ) (total_stations : ℕ) (two_student_stations : ℕ) 
  (h1 : total_capacity = 38)
  (h2 : total_stations = 16)
  (h3 : two_student_stations = 10) :
  total_capacity - (2 * two_student_stations) = 18 := by
  sorry

end lab_capacity_l3585_358598


namespace sum_of_numbers_greater_than_point_four_l3585_358590

theorem sum_of_numbers_greater_than_point_four : 
  let numbers : List ℚ := [0.8, 1/2, 0.9]
  let sum_of_greater : ℚ := (numbers.filter (λ x => x > 0.4)).sum
  sum_of_greater = 2.2 := by
  sorry

end sum_of_numbers_greater_than_point_four_l3585_358590


namespace total_students_proof_l3585_358511

/-- Represents the total number of senior students -/
def total_students : ℕ := 300

/-- Represents the number of students who didn't receive scholarships -/
def no_scholarship_students : ℕ := 255

/-- Percentage of students who received full merit scholarships -/
def full_scholarship_percent : ℚ := 5 / 100

/-- Percentage of students who received half merit scholarships -/
def half_scholarship_percent : ℚ := 10 / 100

/-- Theorem stating that the total number of students is 300 given the scholarship distribution -/
theorem total_students_proof :
  (1 - full_scholarship_percent - half_scholarship_percent) * total_students = no_scholarship_students :=
sorry

end total_students_proof_l3585_358511


namespace remaining_distance_calculation_l3585_358575

/-- The remaining distance to travel after four people have traveled part of the way. -/
def remaining_distance (total_distance : ℝ) 
  (amoli_speed1 amoli_time1 amoli_speed2 amoli_time2 : ℝ)
  (anayet_speed1 anayet_time1 anayet_speed2 anayet_time2 : ℝ)
  (bimal_speed1 bimal_time1 bimal_speed2 bimal_time2 : ℝ)
  (chandni_distance : ℝ) : ℝ :=
  total_distance - 
  (amoli_speed1 * amoli_time1 + amoli_speed2 * amoli_time2 +
   anayet_speed1 * anayet_time1 + anayet_speed2 * anayet_time2 +
   bimal_speed1 * bimal_time1 + bimal_speed2 * bimal_time2 +
   chandni_distance)

/-- Theorem stating the remaining distance to travel. -/
theorem remaining_distance_calculation : 
  remaining_distance 1475 42 3.5 38 2 61 2.5 75 1.5 55 4 30 2 35 = 672 := by
  sorry

end remaining_distance_calculation_l3585_358575


namespace sum_of_square_roots_geq_one_l3585_358594

theorem sum_of_square_roots_geq_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a / (a + 3 * b)) + Real.sqrt (b / (b + 3 * a)) ≥ 1 := by
  sorry

end sum_of_square_roots_geq_one_l3585_358594


namespace sum_of_roots_bound_l3585_358510

/-- Given a quadratic equation x^2 - 2(1-k)x + k^2 = 0 with real roots α and β, 
    the sum of these roots α + β is greater than or equal to 1. -/
theorem sum_of_roots_bound (k : ℝ) (α β : ℝ) : 
  (∀ x, x^2 - 2*(1-k)*x + k^2 = 0 ↔ x = α ∨ x = β) →
  α + β ≥ 1 := by
  sorry

end sum_of_roots_bound_l3585_358510


namespace larger_number_l3585_358517

theorem larger_number (x y : ℝ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : y = 33 := by
  sorry

end larger_number_l3585_358517


namespace f_derivative_l3585_358531

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) * Real.cos x

theorem f_derivative :
  deriv f = fun x => Real.exp (2 * x) * (2 * Real.cos x - Real.sin x) := by
  sorry

end f_derivative_l3585_358531


namespace cone_sphere_volume_ratio_l3585_358565

/-- Given a sphere and a right circular cone where:
    1. The cone's height is twice the sphere's radius
    2. The cone's base radius is equal to the sphere's radius
    Prove that the ratio of the cone's volume to the sphere's volume is 1/2 -/
theorem cone_sphere_volume_ratio (r : ℝ) (h : ℝ) (h_pos : r > 0) (h_eq : h = 2 * r) :
  (1 / 3 * π * r^2 * h) / (4 / 3 * π * r^3) = 1 / 2 := by
  sorry

end cone_sphere_volume_ratio_l3585_358565


namespace garden_perimeter_l3585_358519

theorem garden_perimeter (garden_width playground_length : ℝ) 
  (h1 : garden_width = 4)
  (h2 : playground_length = 16)
  (h3 : ∃ (garden_length playground_width : ℝ), 
    garden_width * garden_length = playground_length * playground_width)
  (h4 : ∃ (garden_length : ℝ), 2 * (garden_width + garden_length) = 104) :
  ∃ (garden_length : ℝ), 2 * (garden_width + garden_length) = 104 :=
by
  sorry

#check garden_perimeter

end garden_perimeter_l3585_358519


namespace calculation_one_l3585_358582

theorem calculation_one : (-2) + (-7) + 9 - (-12) = 12 := by sorry

end calculation_one_l3585_358582


namespace winning_strategy_works_l3585_358508

/-- Represents a player in the coin game -/
inductive Player : Type
| One : Player
| Two : Player

/-- The game state -/
structure GameState :=
  (coins : ℕ)
  (currentPlayer : Player)

/-- Valid moves for each player -/
def validMove (player : Player) (n : ℕ) : Prop :=
  match player with
  | Player.One => n % 2 = 1 ∧ 1 ≤ n ∧ n ≤ 99
  | Player.Two => n % 2 = 0 ∧ 2 ≤ n ∧ n ≤ 100

/-- The winning strategy function -/
def winningStrategy (state : GameState) : Option ℕ :=
  match state.currentPlayer with
  | Player.One => 
    if state.coins > 95 then some 95
    else if state.coins % 101 ≠ 0 then some (state.coins % 101)
    else none
  | Player.Two => none

/-- The main theorem -/
theorem winning_strategy_works : 
  ∀ (state : GameState), 
    state.coins = 2015 → 
    state.currentPlayer = Player.One → 
    ∃ (move : ℕ), 
      validMove Player.One move ∧ 
      move = 95 ∧
      ∀ (opponentMove : ℕ), 
        validMove Player.Two opponentMove → 
        ∃ (nextMove : ℕ), 
          validMove Player.One nextMove ∧ 
          state.coins - move - opponentMove - nextMove ≡ 0 [MOD 101] :=
sorry

#check winning_strategy_works

end winning_strategy_works_l3585_358508


namespace rectangular_prism_surface_area_l3585_358587

/-- The surface area of a rectangular prism with dimensions 1, 2, and 2 is 16 -/
theorem rectangular_prism_surface_area :
  let length : ℝ := 1
  let width : ℝ := 2
  let height : ℝ := 2
  let surface_area := 2 * (length * width + length * height + width * height)
  surface_area = 16 := by
  sorry

end rectangular_prism_surface_area_l3585_358587


namespace digital_earth_properties_l3585_358528

-- Define the properties of Digital Earth
structure DigitalEarth where
  is_digitized : Bool
  meets_current_needs : Bool
  main_feature_virtual_reality : Bool
  uses_centralized_storage : Bool

-- Define the correct properties of Digital Earth
def correct_digital_earth : DigitalEarth := {
  is_digitized := true,
  meets_current_needs := false,
  main_feature_virtual_reality := true,
  uses_centralized_storage := false
}

-- Theorem stating the correct properties of Digital Earth
theorem digital_earth_properties :
  correct_digital_earth.is_digitized ∧
  correct_digital_earth.main_feature_virtual_reality ∧
  ¬correct_digital_earth.meets_current_needs ∧
  ¬correct_digital_earth.uses_centralized_storage :=
by sorry


end digital_earth_properties_l3585_358528


namespace cistern_filling_time_l3585_358543

theorem cistern_filling_time (x : ℝ) 
  (h1 : x > 0)
  (h2 : 1/x + 1/12 - 1/15 = 7/60) : x = 10 := by
  sorry

end cistern_filling_time_l3585_358543


namespace cylinder_unique_non_identical_views_l3585_358532

-- Define the types of solid objects
inductive SolidObject
  | Sphere
  | TriangularPyramid
  | Cube
  | Cylinder

-- Define a function that checks if all views are identical
def hasIdenticalViews (obj : SolidObject) : Prop :=
  match obj with
  | SolidObject.Sphere => True
  | SolidObject.TriangularPyramid => False
  | SolidObject.Cube => True
  | SolidObject.Cylinder => False

-- Theorem statement
theorem cylinder_unique_non_identical_views :
  ∀ (obj : SolidObject), ¬(hasIdenticalViews obj) ↔ obj = SolidObject.Cylinder :=
by sorry

end cylinder_unique_non_identical_views_l3585_358532


namespace perpendicular_transitivity_perpendicular_parallel_planes_l3585_358507

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Theorem 1
theorem perpendicular_transitivity 
  (m n : Line) (α : Plane) 
  (h1 : parallel m n) (h2 : perpendicular m α) : 
  perpendicular n α :=
sorry

-- Theorem 2
theorem perpendicular_parallel_planes 
  (m n : Line) (α β : Plane)
  (h1 : plane_parallel α β) (h2 : parallel m n) (h3 : perpendicular m α) :
  perpendicular n β :=
sorry

end perpendicular_transitivity_perpendicular_parallel_planes_l3585_358507


namespace garrison_size_l3585_358572

/-- Represents the initial number of men in the garrison -/
def initial_men : ℕ := 150

/-- Represents the number of days the provisions were initially meant to last -/
def initial_days : ℕ := 31

/-- Represents the number of days that passed before reinforcements arrived -/
def days_before_reinforcement : ℕ := 16

/-- Represents the number of reinforcement men that arrived -/
def reinforcement_men : ℕ := 300

/-- Represents the number of days the provisions lasted after reinforcements arrived -/
def remaining_days : ℕ := 5

theorem garrison_size :
  initial_men * initial_days = 
  initial_men * (initial_days - days_before_reinforcement) ∧
  initial_men * (initial_days - days_before_reinforcement) = 
  (initial_men + reinforcement_men) * remaining_days :=
by sorry

end garrison_size_l3585_358572


namespace parabola_y_comparison_l3585_358597

/-- Parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 2

/-- Theorem stating that y₁ < y₂ for the given parabola -/
theorem parabola_y_comparison :
  ∀ (y₁ y₂ : ℝ), f 1 = y₁ → f 3 = y₂ → y₁ < y₂ := by
  sorry

end parabola_y_comparison_l3585_358597


namespace rectangle_perimeter_l3585_358576

theorem rectangle_perimeter (a b : ℕ) : 
  b = 3 * a →                   -- One side is three times as long as the other
  a * b = 2 * (a + b) + 12 →    -- Area equals perimeter plus 12
  2 * (a + b) = 32              -- Perimeter is 32 units
  := by sorry

end rectangle_perimeter_l3585_358576


namespace union_of_A_and_B_l3585_358578

-- Define the sets A and B
def A : Set ℝ := {x | 2 / x > 1}
def B : Set ℝ := {x | Real.log x < 0}

-- Define the union of A and B
def AunionB : Set ℝ := A ∪ B

-- Theorem statement
theorem union_of_A_and_B : AunionB = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end union_of_A_and_B_l3585_358578


namespace combinatorial_identity_l3585_358562

theorem combinatorial_identity (n : ℕ) : 
  (n.choose 2) * 2 = 42 → n.choose 3 = 35 := by
  sorry

end combinatorial_identity_l3585_358562


namespace solution_to_linear_equation_l3585_358560

theorem solution_to_linear_equation :
  ∃ x : ℝ, 3 * x - 6 = 0 ∧ x = 2 := by
  sorry

end solution_to_linear_equation_l3585_358560


namespace triangle_area_l3585_358503

theorem triangle_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2) 
  (h_angle : a / c = 1 / 2) (h_hypotenuse : c = 40) : 
  (a * b) / 2 = 200 * Real.sqrt 3 := by
  sorry

end triangle_area_l3585_358503


namespace inequality_proof_l3585_358553

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z ≥ x * y + y * z + z * x) : 
  Real.sqrt (x * y * z) ≥ Real.sqrt x + Real.sqrt y + Real.sqrt z := by
  sorry

end inequality_proof_l3585_358553


namespace cody_book_series_l3585_358541

/-- The number of books in Cody's favorite book series -/
def books_in_series (first_week : ℕ) (second_week : ℕ) (subsequent_weeks : ℕ) (total_weeks : ℕ) : ℕ :=
  first_week + second_week + (subsequent_weeks * (total_weeks - 2))

/-- Theorem stating the number of books in Cody's series -/
theorem cody_book_series : books_in_series 6 3 9 7 = 54 := by
  sorry

end cody_book_series_l3585_358541


namespace race_time_calculation_l3585_358546

/-- Given that Prejean's speed is three-quarters of Rickey's speed and Rickey took 40 minutes to finish a race, 
    prove that the total time taken by both runners is 40 + 40 * (4/3) minutes. -/
theorem race_time_calculation (rickey_speed rickey_time prejean_speed : ℝ) 
    (h1 : rickey_time = 40)
    (h2 : prejean_speed = 3/4 * rickey_speed) : 
  rickey_time + (rickey_time / (prejean_speed / rickey_speed)) = 40 + 40 * (4/3) := by
  sorry

end race_time_calculation_l3585_358546


namespace sphere_cap_cone_volume_equality_l3585_358509

theorem sphere_cap_cone_volume_equality (R : ℝ) (R_pos : R > 0) :
  ∃ x : ℝ, x > 0 ∧ x < R ∧
  (2 / 3 * R^2 * π * (R - x) = 1 / 3 * (R^2 - x^2) * π * x) ∧
  x = R * (Real.sqrt 5 - 1) / 2 :=
by sorry

end sphere_cap_cone_volume_equality_l3585_358509


namespace gcd_4050_12150_l3585_358595

theorem gcd_4050_12150 : Nat.gcd 4050 12150 = 450 := by
  sorry

end gcd_4050_12150_l3585_358595


namespace special_collection_books_l3585_358538

/-- The number of books in a special collection at the beginning of a month,
    given the number of books loaned, returned, and remaining at the end of the month. -/
theorem special_collection_books
  (loaned : ℕ)
  (return_rate : ℚ)
  (end_count : ℕ)
  (h1 : loaned = 30)
  (h2 : return_rate = 7/10)
  (h3 : end_count = 66)
  : ℕ := by
  sorry

end special_collection_books_l3585_358538


namespace leftover_bolts_count_l3585_358502

theorem leftover_bolts_count :
  let bolt_boxes : Nat := 7
  let bolts_per_box : Nat := 11
  let nut_boxes : Nat := 3
  let nuts_per_box : Nat := 15
  let used_bolts_and_nuts : Nat := 113
  let leftover_nuts : Nat := 6
  
  let total_bolts : Nat := bolt_boxes * bolts_per_box
  let total_nuts : Nat := nut_boxes * nuts_per_box
  let total_bolts_and_nuts : Nat := total_bolts + total_nuts
  let leftover_bolts_and_nuts : Nat := total_bolts_and_nuts - used_bolts_and_nuts
  let leftover_bolts : Nat := leftover_bolts_and_nuts - leftover_nuts

  leftover_bolts = 3 := by sorry

end leftover_bolts_count_l3585_358502


namespace arithmetic_sequence_problem_l3585_358540

/-- An arithmetic sequence -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₂ = 2 and a₆ = 10, a₁₀ = 18 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : arithmeticSequence a) 
    (h_a2 : a 2 = 2) 
    (h_a6 : a 6 = 10) : 
  a 10 = 18 := by
sorry

end arithmetic_sequence_problem_l3585_358540


namespace exists_q_no_zeros_in_decimal_l3585_358535

theorem exists_q_no_zeros_in_decimal : ∃ q : ℚ, ∃ a : ℕ, (
  (q * 2^1000 = a) ∧
  (∀ d : ℕ, d < 10 → (a.digits 10).contains d → (d = 1 ∨ d = 2))
) := by sorry

end exists_q_no_zeros_in_decimal_l3585_358535


namespace fishing_trip_theorem_l3585_358518

def is_small_fish (weight : ℕ) : Bool := 1 ≤ weight ∧ weight ≤ 5
def is_medium_fish (weight : ℕ) : Bool := 6 ≤ weight ∧ weight ≤ 12
def is_large_fish (weight : ℕ) : Bool := weight > 12

def brendan_morning_catch : List ℕ := [1, 3, 4, 7, 7, 13, 15, 17]
def brendan_afternoon_catch : List ℕ := [2, 8, 8, 18, 20]
def emily_catch : List ℕ := [5, 6, 9, 11, 14, 20]
def dad_catch : List ℕ := [3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21]

def brendan_morning_keep (weight : ℕ) : Bool := is_medium_fish weight ∨ is_large_fish weight
def brendan_afternoon_keep (weight : ℕ) : Bool := is_medium_fish weight ∨ (is_large_fish weight ∧ weight > 15)
def emily_keep (weight : ℕ) : Bool := is_large_fish weight ∨ weight = 5
def dad_keep (weight : ℕ) : Bool := (is_medium_fish weight ∧ weight ≥ 8 ∧ weight ≤ 11) ∨ (is_large_fish weight ∧ weight > 15 ∧ weight ≠ 21)

theorem fishing_trip_theorem :
  (brendan_morning_catch.filter brendan_morning_keep).length +
  (brendan_afternoon_catch.filter brendan_afternoon_keep).length +
  (emily_catch.filter emily_keep).length +
  (dad_catch.filter dad_keep).length = 18 := by
  sorry

end fishing_trip_theorem_l3585_358518


namespace maria_papers_left_l3585_358520

/-- The number of papers Maria has left after giving away some papers -/
def papers_left (desk : ℕ) (backpack : ℕ) (given_away : ℕ) : ℕ :=
  desk + backpack - given_away

/-- Theorem stating that Maria has 91 - x papers left after giving away x papers -/
theorem maria_papers_left (x : ℕ) :
  papers_left 50 41 x = 91 - x :=
by sorry

end maria_papers_left_l3585_358520


namespace international_call_rate_l3585_358530

/-- Represents the cost and duration of phone calls -/
structure PhoneCall where
  localRate : ℚ
  localDuration : ℚ
  internationalDuration : ℚ
  totalCost : ℚ

/-- Calculates the cost per minute of an international call -/
def internationalRate (call : PhoneCall) : ℚ :=
  (call.totalCost - call.localRate * call.localDuration) / call.internationalDuration

/-- Theorem: Given the specified conditions, the international call rate is 25 cents per minute -/
theorem international_call_rate (call : PhoneCall) 
  (h1 : call.localRate = 5/100)
  (h2 : call.localDuration = 45)
  (h3 : call.internationalDuration = 31)
  (h4 : call.totalCost = 10) :
  internationalRate call = 25/100 := by
  sorry


end international_call_rate_l3585_358530


namespace large_rectangle_perimeter_l3585_358566

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Calculates the perimeter of a square -/
def Square.perimeter (s : Square) : ℝ := 4 * s.side

/-- Theorem: The perimeter of a large rectangle composed of three identical squares
    and three identical small rectangles is 52, given the conditions. -/
theorem large_rectangle_perimeter : 
  ∀ (s : Square) (r : Rectangle),
    s.perimeter = 24 →
    r.perimeter = 16 →
    (3 * s.side = r.length) →
    (s.side + r.width = 3 * r.length + 3 * r.width) →
    Rectangle.perimeter { length := 3 * s.side, width := s.side + r.width } = 52 := by
  sorry

end large_rectangle_perimeter_l3585_358566


namespace pyramid_height_l3585_358599

theorem pyramid_height (p q : ℝ) : 
  p > 0 ∧ q > 0 →
  3^2 + p^2 = 5^2 →
  (1/3) * (1/2 * 3 * p) * q = 12 →
  q = 6 :=
by sorry

end pyramid_height_l3585_358599


namespace investment_scientific_notation_l3585_358577

-- Define the total investment in yuan
def total_investment : ℝ := 7.7e9

-- Define the scientific notation representation
def scientific_notation : ℝ := 7.7 * (10 ^ 9)

-- Theorem statement
theorem investment_scientific_notation : total_investment = scientific_notation := by
  sorry

end investment_scientific_notation_l3585_358577


namespace inverse_proportion_comparison_l3585_358588

/-- Given two points A(1, y₁) and B(2, y₂) on the graph of y = 2/x, prove that y₁ > y₂ -/
theorem inverse_proportion_comparison (y₁ y₂ : ℝ) :
  y₁ = 2 / 1 → y₂ = 2 / 2 → y₁ > y₂ := by
  sorry

end inverse_proportion_comparison_l3585_358588


namespace north_pond_duck_count_l3585_358579

/-- The number of ducks at Lake Michigan -/
def lake_michigan_ducks : ℕ := 100

/-- The number of ducks at North Pond -/
def north_pond_ducks : ℕ := 2 * lake_michigan_ducks + 6

theorem north_pond_duck_count : north_pond_ducks = 206 := by
  sorry

end north_pond_duck_count_l3585_358579


namespace y_share_is_63_l3585_358593

/-- Represents the share of each person in rupees -/
structure Share where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The total amount to be divided -/
def total_amount : ℝ := 245

/-- The ratio of y's share to x's share -/
def y_ratio : ℝ := 0.45

/-- The ratio of z's share to x's share -/
def z_ratio : ℝ := 0.30

/-- The share satisfies the given conditions -/
def is_valid_share (s : Share) : Prop :=
  s.x + s.y + s.z = total_amount ∧
  s.y = y_ratio * s.x ∧
  s.z = z_ratio * s.x

theorem y_share_is_63 :
  ∃ (s : Share), is_valid_share s ∧ s.y = 63 := by
  sorry

end y_share_is_63_l3585_358593


namespace truck_toll_calculation_l3585_358547

/-- Calculates the total toll for a truck crossing a bridge --/
def calculate_total_toll (B A1 A2 X1 X2 w : ℚ) (is_peak_hour : Bool) : ℚ :=
  let T := B + A1 * (X1 - 2) + A2 * X2
  let F := if w > 10000 then 0.1 * (w - 10000) else 0
  let total_without_surcharge := T + F
  let S := if is_peak_hour then 0.02 * total_without_surcharge else 0
  total_without_surcharge + S

theorem truck_toll_calculation :
  let B : ℚ := 0.50
  let A1 : ℚ := 0.75
  let A2 : ℚ := 0.50
  let X1 : ℚ := 1  -- One axle with 2 wheels
  let X2 : ℚ := 4  -- Four axles with 4 wheels each
  let w : ℚ := 12000
  let is_peak_hour : Bool := true  -- 9 AM is during peak hours
  calculate_total_toll B A1 A2 X1 X2 w is_peak_hour = 205.79 := by
  sorry


end truck_toll_calculation_l3585_358547


namespace intersection_points_form_hyperbola_l3585_358563

theorem intersection_points_form_hyperbola :
  ∀ (t x y : ℝ), 
    (2 * t * x - 3 * y - 4 * t = 0) → 
    (x - 3 * t * y + 4 = 0) → 
    (x^2 / 16 - y^2 = 1) :=
by sorry

end intersection_points_form_hyperbola_l3585_358563


namespace min_overlap_social_media_l3585_358545

/-- The minimum percentage of adults using both Facebook and Instagram -/
theorem min_overlap_social_media (facebook_users instagram_users : ℝ) 
  (h1 : facebook_users = 85)
  (h2 : instagram_users = 75) :
  (facebook_users + instagram_users) - 100 = 60 := by
  sorry

end min_overlap_social_media_l3585_358545


namespace max_b_value_l3585_358522

theorem max_b_value (a b : ℤ) : 
  (a + b)^2 + a*(a + b) + b = 0 →
  b ≤ 9 ∧ ∃ (a₀ b₀ : ℤ), (a₀ + b₀)^2 + a₀*(a₀ + b₀) + b₀ = 0 ∧ b₀ = 9 :=
by sorry

end max_b_value_l3585_358522


namespace multiple_is_two_l3585_358580

/-- The grading method used by a teacher for a test -/
structure GradingMethod where
  totalQuestions : ℕ
  studentScore : ℕ
  correctAnswers : ℕ
  scoreCalculation : ℕ → ℕ → ℕ → ℕ → ℕ

/-- The multiple used for incorrect responses in the grading method -/
def incorrectResponseMultiple (gm : GradingMethod) : ℕ :=
  let incorrectAnswers := gm.totalQuestions - gm.correctAnswers
  (gm.correctAnswers - gm.studentScore) / incorrectAnswers

/-- Theorem stating that the multiple used for incorrect responses is 2 -/
theorem multiple_is_two (gm : GradingMethod) 
  (h1 : gm.totalQuestions = 100)
  (h2 : gm.studentScore = 76)
  (h3 : gm.correctAnswers = 92)
  (h4 : gm.scoreCalculation = fun total correct incorrect multiple => 
    correct - multiple * incorrect) :
  incorrectResponseMultiple gm = 2 := by
  sorry


end multiple_is_two_l3585_358580


namespace bush_spacing_l3585_358574

theorem bush_spacing (yard_side_length : ℕ) (num_sides : ℕ) (num_bushes : ℕ) :
  yard_side_length = 16 →
  num_sides = 3 →
  num_bushes = 12 →
  (yard_side_length * num_sides) / num_bushes = 4 := by
  sorry

end bush_spacing_l3585_358574


namespace geometric_series_first_term_l3585_358533

theorem geometric_series_first_term 
  (a r : ℝ) 
  (h1 : 0 ≤ r ∧ r < 1) -- Condition for convergence of geometric series
  (h2 : a / (1 - r) = 20) -- Sum of terms is 20
  (h3 : a^2 / (1 - r^2) = 80) -- Sum of squares of terms is 80
  : a = 20 / 3 := by
sorry

end geometric_series_first_term_l3585_358533


namespace six_digit_palindrome_divisible_by_11_l3585_358549

theorem six_digit_palindrome_divisible_by_11 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  let W := 100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a
  11 ∣ W :=
by sorry

end six_digit_palindrome_divisible_by_11_l3585_358549


namespace triangle_proof_l3585_358539

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def scalene_triangle (t : Triangle) : Prop :=
  t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.c ≠ t.a

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 3 ∧ t.c = 4 ∧ t.C = 2 * t.A

-- Theorem statement
theorem triangle_proof (t : Triangle) 
  (h1 : scalene_triangle t) 
  (h2 : triangle_conditions t) : 
  Real.cos t.A = 2/3 ∧ 
  t.b = 7/3 ∧ 
  Real.cos (2 * t.A + Real.pi/6) = -(Real.sqrt 3 + 4 * Real.sqrt 5)/18 := by
  sorry

end triangle_proof_l3585_358539


namespace cards_distribution_theorem_l3585_358591

/-- Given a total number of cards and people, calculate how many people receive fewer cards when dealt as evenly as possible. -/
def people_with_fewer_cards (total_cards : ℕ) (num_people : ℕ) (threshold : ℕ) : ℕ :=
  let cards_per_person := total_cards / num_people
  let extra_cards := total_cards % num_people
  if cards_per_person + 1 < threshold then num_people
  else num_people - extra_cards

/-- Theorem stating that when 60 cards are dealt to 9 people as evenly as possible, 3 people will have fewer than 7 cards. -/
theorem cards_distribution_theorem :
  people_with_fewer_cards 60 9 7 = 3 := by
  sorry

end cards_distribution_theorem_l3585_358591


namespace z_plus_one_is_pure_imaginary_l3585_358573

theorem z_plus_one_is_pure_imaginary : 
  let z : ℂ := (-2 * Complex.I) / (1 + Complex.I)
  ∃ (y : ℝ), z + 1 = y * Complex.I :=
by
  sorry

end z_plus_one_is_pure_imaginary_l3585_358573


namespace ellipse_intersection_area_and_ratio_l3585_358570

/-- The ellipse C: (x^2/4) + (y^2/b^2) = 1 with 0 < b < 2 -/
def ellipse (b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / b^2) = 1 ∧ 0 < b ∧ b < 2

/-- The point M(1,1) is on the ellipse -/
def M_on_ellipse (b : ℝ) : Prop := ellipse b 1 1

/-- The line l: x + ty - 1 = 0 -/
def line (t : ℝ) (x y : ℝ) : Prop := x + t * y - 1 = 0

/-- Points A and B are the intersections of the line and the ellipse -/
def intersections (b t : ℝ) (A B : ℝ × ℝ) : Prop :=
  ellipse b A.1 A.2 ∧ ellipse b B.1 B.2 ∧ 
  line t A.1 A.2 ∧ line t B.1 B.2

/-- The area of triangle ABM -/
noncomputable def area_ABM (A B : ℝ × ℝ) : ℝ := sorry

/-- Points P and Q are intersections of AM and BM with x = 4 -/
def P_Q_intersections (A B : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  P.1 = 4 ∧ Q.1 = 4 ∧ 
  (∃ k₁ k₂ : ℝ, P = k₁ • A + (1 - k₁) • (1, 1)) ∧
  (∃ k₃ k₄ : ℝ, Q = k₃ • B + (1 - k₃) • (1, 1))

/-- The area of triangle PQM -/
noncomputable def area_PQM (P Q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_intersection_area_and_ratio 
  (b : ℝ) (h_M : M_on_ellipse b) :
  (∃ A B : ℝ × ℝ, intersections b 1 A B ∧ area_ABM A B = Real.sqrt 13 / 4) ∧
  (∃ t : ℝ, ∃ A B P Q : ℝ × ℝ, 
    intersections b t A B ∧ 
    P_Q_intersections A B P Q ∧
    area_PQM P Q = 5 * area_ABM A B ∧
    t = 3 * Real.sqrt 2 / 2 ∨ t = -3 * Real.sqrt 2 / 2) :=
sorry

end ellipse_intersection_area_and_ratio_l3585_358570


namespace rob_quarters_l3585_358581

def quarters : ℕ → ℚ
  | n => (n : ℚ) * (1 / 4)

def dimes : ℕ → ℚ
  | n => (n : ℚ) * (1 / 10)

def nickels : ℕ → ℚ
  | n => (n : ℚ) * (1 / 20)

def pennies : ℕ → ℚ
  | n => (n : ℚ) * (1 / 100)

theorem rob_quarters (x : ℕ) :
  quarters x + dimes 3 + nickels 5 + pennies 12 = 242 / 100 →
  x = 7 := by
sorry

end rob_quarters_l3585_358581


namespace multiply_decimals_l3585_358586

theorem multiply_decimals : (4.8 : ℝ) * 0.25 * 0.1 = 0.12 := by
  sorry

end multiply_decimals_l3585_358586


namespace u_plus_v_value_l3585_358505

theorem u_plus_v_value (u v : ℚ) 
  (eq1 : 3 * u + 7 * v = 17)
  (eq2 : 5 * u - 3 * v = 9) :
  u + v = 43 / 11 := by
sorry

end u_plus_v_value_l3585_358505


namespace quadratic_inequality_properties_l3585_358550

/-- Given a quadratic inequality ax^2 + bx + c < 0 with solution set (1/t, t) where t > 0,
    prove certain properties about a, b, c, and related equations. -/
theorem quadratic_inequality_properties
  (a b c t : ℝ)
  (h_solution_set : ∀ x : ℝ, a * x^2 + b * x + c < 0 ↔ 1/t < x ∧ x < t)
  (h_t_pos : t > 0) :
  abc < 0 ∧
  2*a + b < 0 ∧
  (∀ x₁ x₂ : ℝ, (a * x₁ + b * Real.sqrt x₁ + c = 0 ∧
                 a * x₂ + b * Real.sqrt x₂ + c = 0) →
                x₁ + x₂ > t + 1/t) :=
by sorry

end quadratic_inequality_properties_l3585_358550


namespace perpendicular_vectors_imply_m_eq_neg_two_l3585_358592

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The theorem stating that if vectors (2, 3m+2) and (m, -1) are perpendicular, then m = -2 -/
theorem perpendicular_vectors_imply_m_eq_neg_two (m : ℝ) :
  perpendicular (2, 3*m+2) (m, -1) → m = -2 := by
  sorry

end perpendicular_vectors_imply_m_eq_neg_two_l3585_358592
