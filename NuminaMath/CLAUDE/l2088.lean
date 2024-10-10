import Mathlib

namespace platform_crossing_time_l2088_208825

def train_speed : Real := 36  -- km/h
def time_to_cross_pole : Real := 12  -- seconds
def time_to_cross_platform : Real := 44.99736021118311  -- seconds

theorem platform_crossing_time :
  time_to_cross_platform = 44.99736021118311 :=
by sorry

end platform_crossing_time_l2088_208825


namespace parabola_conditions_imply_a_range_l2088_208863

theorem parabola_conditions_imply_a_range (a : ℝ) : 
  (a - 1 > 0) →  -- parabola y=(a-1)x^2 opens upwards
  (2*a - 3 < 0) →  -- parabola y=(2a-3)x^2 opens downwards
  (|a - 1| > |2*a - 3|) →  -- parabola y=(a-1)x^2 has a wider opening than y=(2a-3)x^2
  (4/3 < a ∧ a < 3/2) :=
by sorry

end parabola_conditions_imply_a_range_l2088_208863


namespace kindergartners_with_orange_shirts_l2088_208854

-- Define the constants from the problem
def first_graders : ℕ := 113
def second_graders : ℕ := 107
def third_graders : ℕ := 108

def yellow_shirt_cost : ℚ := 5
def blue_shirt_cost : ℚ := 56/10
def green_shirt_cost : ℚ := 21/4
def orange_shirt_cost : ℚ := 29/5

def total_spent : ℚ := 2317

-- Theorem to prove
theorem kindergartners_with_orange_shirts :
  (total_spent -
   (first_graders * yellow_shirt_cost +
    second_graders * blue_shirt_cost +
    third_graders * green_shirt_cost)) / orange_shirt_cost = 101 := by
  sorry

end kindergartners_with_orange_shirts_l2088_208854


namespace guest_cars_count_l2088_208808

/-- Calculates the number of guest cars given the total number of wheels and parent cars -/
def guest_cars (total_wheels : ℕ) (parent_cars : ℕ) : ℕ :=
  (total_wheels - 4 * parent_cars) / 4

/-- Theorem: Given 48 total wheels and 2 parent cars, the number of guest cars is 10 -/
theorem guest_cars_count : guest_cars 48 2 = 10 := by
  sorry

end guest_cars_count_l2088_208808


namespace inner_circle_radius_l2088_208897

theorem inner_circle_radius : 
  ∀ r : ℝ,
  (r > 0) →
  (π * (9^2) - π * ((0.75 * r)^2) = 3.6 * (π * 6^2 - π * r^2)) →
  r = 4 := by
sorry

end inner_circle_radius_l2088_208897


namespace angle_is_three_pi_over_four_l2088_208879

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_is_three_pi_over_four (a b : ℝ × ℝ) 
  (h1 : a.fst * (a.fst - 2 * b.fst) + a.snd * (a.snd - 2 * b.snd) = 3)
  (h2 : a.fst^2 + a.snd^2 = 1)
  (h3 : b = (1, 1)) :
  angle_between_vectors a b = 3 * π / 4 := by sorry

end angle_is_three_pi_over_four_l2088_208879


namespace max_temperature_range_l2088_208893

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem max_temperature_range 
  (T1 T2 T3 T4 T5 : ℕ) 
  (avg_temp : (T1 + T2 + T3 + T4 + T5) / 5 = 60)
  (lowest_temp : T1 = 50 ∧ T2 = 50)
  (consecutive : ∃ n : ℕ, T3 = n ∧ T4 = n + 1 ∧ T5 = n + 2)
  (ordered : T3 ≤ T4 ∧ T4 ≤ T5)
  (prime_exists : is_prime T3 ∨ is_prime T4 ∨ is_prime T5) :
  T5 - T1 = 18 :=
sorry

end max_temperature_range_l2088_208893


namespace complex_power_magnitude_l2088_208843

theorem complex_power_magnitude : Complex.abs ((2 + Complex.I * Real.sqrt 11) ^ 4) = 225 := by
  sorry

end complex_power_magnitude_l2088_208843


namespace x_value_l2088_208845

theorem x_value : ∃ x : ℚ, (3 * x) / 7 = 21 ∧ x = 49 := by sorry

end x_value_l2088_208845


namespace checkers_inequality_l2088_208850

theorem checkers_inequality (n : ℕ) (A B : ℕ) : A ≤ 3 * B :=
  by
  -- Assume n is the number of black checkers (equal to the number of white checkers)
  -- A is the number of triples with white majority
  -- B is the number of triples with black majority
  sorry

end checkers_inequality_l2088_208850


namespace quadratic_form_constant_l2088_208890

theorem quadratic_form_constant (x : ℝ) :
  ∃ (a h k : ℝ), x^2 - 7*x = a*(x - h)^2 + k ∧ k = -49/4 := by
  sorry

end quadratic_form_constant_l2088_208890


namespace angle_MDN_is_acute_l2088_208831

/-- The parabola y^2 = 2x -/
def parabola (x y : ℝ) : Prop := y^2 = 2*x

/-- A line passing through point (2,0) -/
def line_through_P (k : ℝ) (x y : ℝ) : Prop := x = k*y + 2

/-- The vertical line x = -1/2 -/
def vertical_line (x : ℝ) : Prop := x = -1/2

/-- The dot product of two 2D vectors -/
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1*x2 + y1*y2

theorem angle_MDN_is_acute (k t : ℝ) (xM yM xN yN : ℝ) :
  parabola xM yM →
  parabola xN yN →
  line_through_P k xM yM →
  line_through_P k xN yN →
  vertical_line (-1/2) →
  xM ≠ xN ∨ yM ≠ yN →
  dot_product (xM + 1/2) (yM - t) (xN + 1/2) (yN - t) > 0 :=
sorry

end angle_MDN_is_acute_l2088_208831


namespace function_minimum_value_l2088_208842

/-- Given a function f(x) = (ax + b) / (x^2 + 4) that attains a maximum value of 1 at x = -1,
    prove that the minimum value of f(x) is -1/4 -/
theorem function_minimum_value (a b : ℝ) :
  let f := fun x : ℝ => (a * x + b) / (x^2 + 4)
  (f (-1) = 1) →
  (∃ x₀, ∀ x, f x ≥ f x₀) →
  (∃ x₁, f x₁ = -1/4 ∧ ∀ x, f x ≥ -1/4) :=
by sorry

end function_minimum_value_l2088_208842


namespace set_difference_M_N_l2088_208844

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 3, 4}

theorem set_difference_M_N : M \ N = {1, 5} := by sorry

end set_difference_M_N_l2088_208844


namespace fraction_equality_l2088_208803

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) :
  1 / 7 + (2 * q - p) / (2 * q + p) = 4 / 7 := by
  sorry

end fraction_equality_l2088_208803


namespace cube_minus_square_equals_zero_l2088_208835

theorem cube_minus_square_equals_zero : 4^3 - 8^2 = 0 :=
by
  -- Given conditions (not used in the proof, but included for completeness)
  have h1 : 2^3 - 7^2 = 1 := by sorry
  have h2 : 3^3 - 6^2 = 9 := by sorry
  have h3 : 5^3 - 9^2 = 16 := by sorry
  
  -- Proof
  sorry

end cube_minus_square_equals_zero_l2088_208835


namespace collision_count_l2088_208852

/-- The number of collisions between two groups of balls moving in opposite directions in a trough with a wall -/
def totalCollisions (n m : ℕ) : ℕ :=
  n * m + n * (n - 1) / 2

/-- Theorem stating that the total number of collisions for 20 balls moving towards a wall
    and 16 balls moving in the opposite direction is 510 -/
theorem collision_count : totalCollisions 20 16 = 510 := by
  sorry

end collision_count_l2088_208852


namespace statements_classification_correct_l2088_208809

-- Define the type of statement
inductive StatementType
  | Universal
  | Existential

-- Define a structure to represent a statement
structure Statement where
  text : String
  type : StatementType
  isTrue : Bool

-- Define the four statements
def statement1 : Statement :=
  { text := "The diagonals of a square are perpendicular bisectors of each other"
  , type := StatementType.Universal
  , isTrue := true }

def statement2 : Statement :=
  { text := "All Chinese people speak Chinese"
  , type := StatementType.Universal
  , isTrue := false }

def statement3 : Statement :=
  { text := "Some numbers are greater than their squares"
  , type := StatementType.Existential
  , isTrue := true }

def statement4 : Statement :=
  { text := "Some real numbers have irrational square roots"
  , type := StatementType.Existential
  , isTrue := true }

-- Theorem to prove the correctness of the statements' classifications
theorem statements_classification_correct :
  statement1.type = StatementType.Universal ∧ statement1.isTrue = true ∧
  statement2.type = StatementType.Universal ∧ statement2.isTrue = false ∧
  statement3.type = StatementType.Existential ∧ statement3.isTrue = true ∧
  statement4.type = StatementType.Existential ∧ statement4.isTrue = true :=
by sorry

end statements_classification_correct_l2088_208809


namespace car_distance_covered_l2088_208818

/-- Proves that a car traveling at 97.5 km/h for 4 hours covers a distance of 390 km -/
theorem car_distance_covered (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 97.5 → time = 4 → distance = speed * time → distance = 390 := by
  sorry

end car_distance_covered_l2088_208818


namespace product_is_square_l2088_208807

theorem product_is_square (g : ℕ) (h : g = 14) : ∃ n : ℕ, 3150 * g = n^2 := by
  sorry

end product_is_square_l2088_208807


namespace inequality_proof_l2088_208828

theorem inequality_proof (n : ℕ) (x y z : ℝ) 
  (h_n : n ≥ 3) 
  (h_x : x > 0) (h_y : y > 0) (h_z : z > 0)
  (h_sum : x + y + z = 1) :
  (1 / x^(n-1) - x) * (1 / y^(n-1) - y) * (1 / z^(n-1) - z) ≥ ((3^n - 1) / 3)^3 := by
  sorry

end inequality_proof_l2088_208828


namespace square_rectangle_area_difference_l2088_208805

theorem square_rectangle_area_difference :
  let square_side : ℝ := 2
  let rect_length : ℝ := 2
  let rect_width : ℝ := 2
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  square_area - rect_area = 0 := by
sorry

end square_rectangle_area_difference_l2088_208805


namespace triangle_cosine_relation_l2088_208833

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and area S, if S + a² = (b + c)², then cos A = -15/17 -/
theorem triangle_cosine_relation (a b c S : ℝ) (A : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < S →  -- positive area
  S = (1/2) * b * c * Real.sin A →  -- area formula
  a^2 + b^2 - 2 * a * b * Real.cos A = c^2 →  -- cosine law
  S + a^2 = (b + c)^2 →  -- given condition
  Real.cos A = -15/17 := by
sorry

end triangle_cosine_relation_l2088_208833


namespace expression_is_perfect_square_l2088_208847

/-- The expression is a perfect square when p equals 0.28 -/
theorem expression_is_perfect_square : 
  ∃ (x : ℝ), (12.86 * 12.86 + 12.86 * 0.28 + 0.14 * 0.14) = x^2 := by
  sorry

end expression_is_perfect_square_l2088_208847


namespace equation_has_real_root_l2088_208836

theorem equation_has_real_root (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) := by
  sorry

end equation_has_real_root_l2088_208836


namespace problem_solution_l2088_208824

theorem problem_solution :
  ∀ (a b c : ℕ+) (x y z : ℤ),
    x = -2272 →
    y = 1000 + 100 * c.val + 10 * b.val + a.val →
    z = 1 →
    a.val * x + b.val * y + c.val * z = 1 →
    a < b →
    b < c →
    y = 1987 := by
  sorry

end problem_solution_l2088_208824


namespace fraction_comparison_l2088_208827

theorem fraction_comparison : 
  let original := -15 / 12
  let a := -30 / 24
  let b := -1 - 3 / 12
  let c := -1 - 9 / 36
  let d := -1 - 5 / 15
  let e := -1 - 25 / 100
  (a = original ∧ b = original ∧ c = original ∧ e = original) ∧ d ≠ original :=
by sorry

end fraction_comparison_l2088_208827


namespace arithmetic_expression_equality_l2088_208800

theorem arithmetic_expression_equality : 5 * 7 - 6 * 8 + 9 * 2 + 7 * 3 = 26 := by
  sorry

end arithmetic_expression_equality_l2088_208800


namespace jenny_lasagna_profit_l2088_208815

/-- Calculates Jenny's profit from selling lasagna pans -/
def jennys_profit (cost_per_pan : ℝ) (num_pans : ℕ) (price_per_pan : ℝ) : ℝ :=
  (price_per_pan * num_pans) - (cost_per_pan * num_pans)

theorem jenny_lasagna_profit :
  let cost_per_pan : ℝ := 10
  let num_pans : ℕ := 20
  let price_per_pan : ℝ := 25
  jennys_profit cost_per_pan num_pans price_per_pan = 300 := by
  sorry

end jenny_lasagna_profit_l2088_208815


namespace imaginary_unit_power_l2088_208876

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_unit_power (n : ℤ) : i^n = 1 ↔ 4 ∣ n :=
sorry

end imaginary_unit_power_l2088_208876


namespace isosceles_triangle_base_range_l2088_208848

/-- An isosceles triangle with perimeter 20 and base length x -/
structure IsoscelesTriangle where
  base : ℝ
  perimeter : ℝ
  is_isosceles : perimeter = 20
  base_definition : base > 0

/-- The range of possible base lengths for an isosceles triangle with perimeter 20 -/
theorem isosceles_triangle_base_range (t : IsoscelesTriangle) :
  5 < t.base ∧ t.base < 10 := by
  sorry


end isosceles_triangle_base_range_l2088_208848


namespace percent_women_non_union_part_time_l2088_208856

/-- Represents the percentage of employees who are men -/
def percentMen : ℝ := 54

/-- Represents the percentage of employees who are women -/
def percentWomen : ℝ := 46

/-- Represents the percentage of men who work full-time -/
def percentMenFullTime : ℝ := 70

/-- Represents the percentage of men who work part-time -/
def percentMenPartTime : ℝ := 30

/-- Represents the percentage of women who work full-time -/
def percentWomenFullTime : ℝ := 60

/-- Represents the percentage of women who work part-time -/
def percentWomenPartTime : ℝ := 40

/-- Represents the percentage of full-time employees who are unionized -/
def percentFullTimeUnionized : ℝ := 60

/-- Represents the percentage of part-time employees who are unionized -/
def percentPartTimeUnionized : ℝ := 50

/-- The main theorem stating that given the conditions, 
    approximately 52.94% of non-union part-time employees are women -/
theorem percent_women_non_union_part_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧
  abs ((9 : ℝ) / 17 * 100 - 52.94) < ε := by
  sorry


end percent_women_non_union_part_time_l2088_208856


namespace martins_trip_distance_l2088_208881

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that Martin's trip distance is 72.0 miles -/
theorem martins_trip_distance :
  let speed : ℝ := 12.0
  let time : ℝ := 6.0
  distance speed time = 72.0 := by
sorry

end martins_trip_distance_l2088_208881


namespace equation_solution_l2088_208857

theorem equation_solution (x : ℝ) : (x + 6) / (x - 3) = 4 → x = 6 := by
  sorry

end equation_solution_l2088_208857


namespace cd_length_calculation_l2088_208849

theorem cd_length_calculation : 
  let first_cd_length : ℝ := 1.5
  let second_cd_length : ℝ := 1.5
  let third_cd_length : ℝ := 2 * first_cd_length
  first_cd_length + second_cd_length + third_cd_length = 6 := by sorry

end cd_length_calculation_l2088_208849


namespace fifth_power_complex_equality_l2088_208811

theorem fifth_power_complex_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + b * Complex.I) ^ 5 = (a - b * Complex.I) ^ 5) : 
  b / a = Real.sqrt 5 := by
sorry

end fifth_power_complex_equality_l2088_208811


namespace basketball_lineup_count_l2088_208896

def number_of_lineups (total_players : ℕ) (captain_count : ℕ) (regular_players : ℕ) : ℕ :=
  total_players * (Nat.choose (total_players - 1) regular_players)

theorem basketball_lineup_count :
  number_of_lineups 12 1 5 = 5544 := by
sorry

end basketball_lineup_count_l2088_208896


namespace container_volume_ratio_l2088_208820

theorem container_volume_ratio :
  ∀ (volume_container1 volume_container2 : ℚ),
  volume_container1 > 0 →
  volume_container2 > 0 →
  (2 : ℚ) / 3 * volume_container1 = (1 : ℚ) / 2 * volume_container2 →
  volume_container1 / volume_container2 = (3 : ℚ) / 4 := by
sorry

end container_volume_ratio_l2088_208820


namespace solution_set_abs_inequality_l2088_208822

theorem solution_set_abs_inequality (x : ℝ) :
  |2 - x| ≥ 1 ↔ x ≤ 1 ∨ x ≥ 3 := by sorry

end solution_set_abs_inequality_l2088_208822


namespace gorillas_sent_to_different_zoo_l2088_208873

theorem gorillas_sent_to_different_zoo :
  let initial_animals : ℕ := 68
  let hippopotamus : ℕ := 1
  let rhinos : ℕ := 3
  let lion_cubs : ℕ := 8
  let meerkats : ℕ := 2 * lion_cubs
  let final_animals : ℕ := 90
  let gorillas_sent : ℕ := initial_animals + hippopotamus + rhinos + lion_cubs + meerkats - final_animals
  gorillas_sent = 6 :=
by sorry

end gorillas_sent_to_different_zoo_l2088_208873


namespace trees_to_plant_l2088_208862

/-- The number of trees chopped down in the first half of the year -/
def first_half_trees : ℕ := 200

/-- The number of trees chopped down in the second half of the year -/
def second_half_trees : ℕ := 300

/-- The number of trees to be planted for each tree chopped down -/
def trees_to_plant_ratio : ℕ := 3

/-- Theorem stating the number of trees the company needs to plant -/
theorem trees_to_plant : 
  (first_half_trees + second_half_trees) * trees_to_plant_ratio = 1500 := by
  sorry

end trees_to_plant_l2088_208862


namespace f_value_at_2_l2088_208892

def f (a b x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem f_value_at_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 := by
  sorry

end f_value_at_2_l2088_208892


namespace trig_simplification_l2088_208810

theorem trig_simplification :
  (Real.sin (35 * π / 180))^2 / Real.sin (20 * π / 180) - 1 / (2 * Real.sin (20 * π / 180)) = -1/2 := by
  sorry

end trig_simplification_l2088_208810


namespace blue_balls_in_jar_l2088_208877

theorem blue_balls_in_jar (total_balls : ℕ) (removed_blue : ℕ) (prob_after : ℚ) : 
  total_balls = 25 → 
  removed_blue = 5 → 
  prob_after = 1/5 → 
  ∃ initial_blue : ℕ, 
    initial_blue = 9 ∧ 
    (initial_blue - removed_blue : ℚ) / (total_balls - removed_blue : ℚ) = prob_after :=
by sorry

end blue_balls_in_jar_l2088_208877


namespace kitchen_width_proof_l2088_208866

/-- Proves that the width of a rectangular kitchen floor is 8 inches, given the specified conditions. -/
theorem kitchen_width_proof (tile_area : ℝ) (kitchen_length : ℝ) (total_tiles : ℕ) 
  (h1 : tile_area = 6)
  (h2 : kitchen_length = 72)
  (h3 : total_tiles = 96) :
  (tile_area * total_tiles) / kitchen_length = 8 := by
  sorry

end kitchen_width_proof_l2088_208866


namespace sideline_time_l2088_208812

def game_duration : ℕ := 90
def first_play_time : ℕ := 20
def second_play_time : ℕ := 35

theorem sideline_time :
  game_duration - (first_play_time + second_play_time) = 35 := by
  sorry

end sideline_time_l2088_208812


namespace probability_two_white_balls_is_one_fifth_l2088_208860

/-- The probability of drawing two white balls from a box containing 7 white balls
    and 8 black balls, when drawing two balls at random without replacement. -/
def probability_two_white_balls : ℚ :=
  let total_balls : ℕ := 7 + 8
  let white_balls : ℕ := 7
  (Nat.choose white_balls 2 : ℚ) / (Nat.choose total_balls 2 : ℚ)

/-- Theorem stating that the probability of drawing two white balls is 1/5. -/
theorem probability_two_white_balls_is_one_fifth :
  probability_two_white_balls = 1 / 5 := by
  sorry

end probability_two_white_balls_is_one_fifth_l2088_208860


namespace infinitely_many_even_floor_squares_l2088_208841

theorem infinitely_many_even_floor_squares (α : ℝ) (h : α > 0) :
  Set.Infinite {n : ℕ+ | Even ⌊(n : ℝ)^2 * α⌋} := by
  sorry

end infinitely_many_even_floor_squares_l2088_208841


namespace inequality_proof_l2088_208819

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * Real.log x + a * x^2 + 1

theorem inequality_proof (a : ℝ) (h : a ≤ -2) :
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → (f a x₁ - f a x₂) / (x₂ - x₁) ≥ 4 := by
  sorry

end inequality_proof_l2088_208819


namespace train_length_l2088_208804

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 58 → time_s = 9 → 
  ∃ length_m : ℝ, abs (length_m - 144.99) < 0.01 := by
  sorry

#check train_length

end train_length_l2088_208804


namespace remainder_is_zero_l2088_208887

-- Define the given binary number
def binary_num : Nat := 857  -- 1101011001₂ in decimal

-- Theorem statement
theorem remainder_is_zero : (binary_num + 3) % 4 = 0 := by
  sorry

end remainder_is_zero_l2088_208887


namespace base_2_representation_of_125_l2088_208806

theorem base_2_representation_of_125 :
  ∃ (b : List Bool), 
    (b.length = 7) ∧
    (b.foldl (λ acc x => 2 * acc + if x then 1 else 0) 0 = 125) ∧
    (b = [true, true, true, true, true, false, true]) := by
  sorry

end base_2_representation_of_125_l2088_208806


namespace pentagon_triangle_angle_sum_l2088_208891

/-- The measure of an interior angle of a regular pentagon in degrees -/
def regular_pentagon_angle : ℝ := 108

/-- The measure of an interior angle of a regular triangle in degrees -/
def regular_triangle_angle : ℝ := 60

/-- Theorem: The sum of angles formed by two adjacent sides of a regular pentagon 
    and one side of a regular triangle that share a vertex is 168 degrees -/
theorem pentagon_triangle_angle_sum : 
  regular_pentagon_angle + regular_triangle_angle = 168 := by
  sorry

end pentagon_triangle_angle_sum_l2088_208891


namespace symmetric_line_theorem_l2088_208813

/-- The equation of a line symmetric to another line with respect to a vertical line. -/
def symmetric_line_equation (a b c : ℝ) (k : ℝ) : ℝ → ℝ → Prop :=
  fun x y => 2 * (k - a) * x + b * y + (c - 2 * k * (k - a)) = 0

/-- The original line equation. -/
def original_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

/-- The line of symmetry. -/
def symmetry_line (x : ℝ) : Prop := x = 3

theorem symmetric_line_theorem :
  symmetric_line_equation 1 (-2) 1 3 = fun x y => 2 * x + y - 8 = 0 :=
sorry

end symmetric_line_theorem_l2088_208813


namespace fraction_sum_lower_bound_sum_lower_bound_l2088_208882

-- Part 1
theorem fraction_sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  1 / a + 1 / (b + 1) ≥ 4 / 5 := by sorry

-- Part 2
theorem sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b + a * b = 8) :
  a + b ≥ 4 := by sorry

end fraction_sum_lower_bound_sum_lower_bound_l2088_208882


namespace triangle_theorem_l2088_208826

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle)
  (h1 : t.a * Real.cos t.B + t.b * Real.cos t.A = 2 * Real.cos t.C)
  (h2 : t.a + t.b = 4)
  (h3 : t.c = 2) :
  t.C = Real.pi / 3 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 :=
by sorry

end triangle_theorem_l2088_208826


namespace price_reduction_theorem_l2088_208874

def price_reduction_problem (original_price : ℝ) : Prop :=
  let reduced_price := 0.75 * original_price
  let original_amount := 1100 / original_price
  let new_amount := 1100 / reduced_price
  (new_amount - original_amount = 5) ∧ (reduced_price = 55)

theorem price_reduction_theorem :
  ∃ (original_price : ℝ), price_reduction_problem original_price :=
sorry

end price_reduction_theorem_l2088_208874


namespace obtuse_triangles_in_17gon_l2088_208859

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A triangle formed by three vertices of a regular polygon -/
structure PolygonTriangle (n : ℕ) (polygon : RegularPolygon n) where
  v1 : Fin n
  v2 : Fin n
  v3 : Fin n

/-- Predicate to determine if a triangle is obtuse -/
def isObtuseTriangle (n : ℕ) (polygon : RegularPolygon n) (triangle : PolygonTriangle n polygon) : Prop :=
  sorry

/-- Count the number of obtuse triangles in a regular polygon -/
def countObtuseTriangles (n : ℕ) (polygon : RegularPolygon n) : ℕ :=
  sorry

theorem obtuse_triangles_in_17gon :
  ∀ (polygon : RegularPolygon 17),
  countObtuseTriangles 17 polygon = 476 :=
sorry

end obtuse_triangles_in_17gon_l2088_208859


namespace cos_15_cos_45_minus_cos_75_sin_45_l2088_208801

theorem cos_15_cos_45_minus_cos_75_sin_45 :
  Real.cos (15 * π / 180) * Real.cos (45 * π / 180) - 
  Real.cos (75 * π / 180) * Real.sin (45 * π / 180) = 1 / 2 := by
  sorry

end cos_15_cos_45_minus_cos_75_sin_45_l2088_208801


namespace inverse_composition_l2088_208853

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv g_inv : ℝ → ℝ)

-- Condition: f⁻¹(g(x)) = 7x - 4
axiom condition : ∀ x, f_inv (g x) = 7 * x - 4

-- Theorem to prove
theorem inverse_composition :
  g_inv (f 2) = 6 / 7 :=
sorry

end inverse_composition_l2088_208853


namespace function_nonpositive_implies_a_geq_3_l2088_208895

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 3 - a

-- State the theorem
theorem function_nonpositive_implies_a_geq_3 :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ 0) → a ≥ 3 := by
  sorry

end function_nonpositive_implies_a_geq_3_l2088_208895


namespace total_boxes_sold_l2088_208834

def boxes_sold (friday saturday sunday monday : ℕ) : ℕ :=
  friday + saturday + sunday + monday

theorem total_boxes_sold :
  ∀ (friday saturday sunday monday : ℕ),
    friday = 40 →
    saturday = 2 * friday - 10 →
    sunday = saturday / 2 →
    monday = sunday + (sunday / 4 + 1) →
    boxes_sold friday saturday sunday monday = 189 :=
by sorry

end total_boxes_sold_l2088_208834


namespace arithmetic_sequence_sum_l2088_208814

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → a 1 + a 2 = -1 → a 3 = 4 → a 4 + a 5 = 17 := by
  sorry

#check arithmetic_sequence_sum

end arithmetic_sequence_sum_l2088_208814


namespace number_problem_l2088_208858

theorem number_problem (x : ℝ) : (0.6 * (3/5) * x = 36) → x = 100 := by
  sorry

end number_problem_l2088_208858


namespace num_tetrahedrons_in_cube_l2088_208855

/-- A cube is represented by its 8 vertices -/
structure Cube :=
  (vertices : Fin 8 → Point)

/-- A tetrahedron is represented by its 4 vertices -/
structure Tetrahedron :=
  (vertices : Fin 4 → Point)

/-- Function to check if a set of 4 vertices forms a valid tetrahedron -/
def is_valid_tetrahedron (c : Cube) (t : Tetrahedron) : Prop :=
  sorry

/-- The number of valid tetrahedrons that can be formed from the vertices of a cube -/
def num_tetrahedrons (c : Cube) : ℕ :=
  sorry

/-- Theorem stating that the number of tetrahedrons formed from a cube's vertices is 58 -/
theorem num_tetrahedrons_in_cube (c : Cube) : num_tetrahedrons c = 58 :=
  sorry

end num_tetrahedrons_in_cube_l2088_208855


namespace smallest_n_is_25_l2088_208898

/-- Represents a student's answers as a 5-tuple of integers from 1 to 4 -/
def Answer := Fin 5 → Fin 4

/-- The set of all possible answer patterns satisfying the modular constraint -/
def S : Set Answer :=
  {a | (a 0).val + (a 1).val + (a 2).val + (a 3).val + (a 4).val ≡ 0 [MOD 4]}

/-- The number of students -/
def num_students : ℕ := 2000

/-- The function that checks if two answers differ in at least two positions -/
def differ_in_two (a b : Answer) : Prop :=
  ∃ i j, i ≠ j ∧ a i ≠ b i ∧ a j ≠ b j

/-- The theorem to be proved -/
theorem smallest_n_is_25 :
  ∀ f : Fin num_students → Answer,
  ∃ n : ℕ, n = 25 ∧
  (∀ subset : Fin n → Fin num_students,
   ∃ a b c d : Fin n,
   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   differ_in_two (f (subset a)) (f (subset b)) ∧
   differ_in_two (f (subset a)) (f (subset c)) ∧
   differ_in_two (f (subset a)) (f (subset d)) ∧
   differ_in_two (f (subset b)) (f (subset c)) ∧
   differ_in_two (f (subset b)) (f (subset d)) ∧
   differ_in_two (f (subset c)) (f (subset d))) ∧
  (∀ m : ℕ, m < 25 →
   ∃ f : Fin num_students → Answer,
   ∀ subset : Fin m → Fin num_students,
   ¬∃ a b c d : Fin m,
   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   differ_in_two (f (subset a)) (f (subset b)) ∧
   differ_in_two (f (subset a)) (f (subset c)) ∧
   differ_in_two (f (subset a)) (f (subset d)) ∧
   differ_in_two (f (subset b)) (f (subset c)) ∧
   differ_in_two (f (subset b)) (f (subset d)) ∧
   differ_in_two (f (subset c)) (f (subset d))) :=
by sorry

end smallest_n_is_25_l2088_208898


namespace container_capacity_l2088_208846

theorem container_capacity : ∀ (C : ℝ), 
  (0.3 * C + 18 = 0.75 * C) → C = 40 :=
fun C h => by
  sorry

end container_capacity_l2088_208846


namespace pictures_in_first_album_l2088_208802

def total_pictures : ℕ := 25
def num_other_albums : ℕ := 5
def pics_per_other_album : ℕ := 3

theorem pictures_in_first_album :
  total_pictures - (num_other_albums * pics_per_other_album) = 10 := by
  sorry

end pictures_in_first_album_l2088_208802


namespace class_size_l2088_208894

/-- The position of Xiao Ming from the front of the line -/
def position_from_front : ℕ := 23

/-- The position of Xiao Ming from the back of the line -/
def position_from_back : ℕ := 23

/-- The total number of students in the class -/
def total_students : ℕ := position_from_front + position_from_back - 1

theorem class_size :
  total_students = 45 :=
sorry

end class_size_l2088_208894


namespace cosine_in_right_triangle_l2088_208872

theorem cosine_in_right_triangle (D E F : Real) (h1 : 0 < D) (h2 : 0 < E) (h3 : 0 < F) : 
  D^2 + E^2 = F^2 → D = 8 → F = 17 → Real.cos (Real.arccos (D / F)) = 15 / 17 := by
sorry

end cosine_in_right_triangle_l2088_208872


namespace tim_weekly_earnings_l2088_208864

/-- Calculates Tim's total weekly earnings including bonuses -/
def timWeeklyEarnings (tasksPerDay : ℕ) (workDaysPerWeek : ℕ) 
  (tasksPay1 tasksPay2 tasksPay3 : ℕ) (rate1 rate2 rate3 : ℚ) 
  (bonusThreshold : ℕ) (bonusAmount : ℚ) 
  (performanceBonusThreshold : ℕ) (performanceBonusAmount : ℚ) : ℚ :=
  let dailyEarnings := tasksPay1 * rate1 + tasksPay2 * rate2 + tasksPay3 * rate3
  let dailyBonuses := (tasksPerDay / bonusThreshold) * bonusAmount
  let weeklyEarnings := (dailyEarnings + dailyBonuses) * workDaysPerWeek
  let performanceBonus := if tasksPerDay ≥ performanceBonusThreshold then performanceBonusAmount else 0
  weeklyEarnings + performanceBonus

/-- Tim's total weekly earnings are $1058 -/
theorem tim_weekly_earnings :
  timWeeklyEarnings 100 6 40 30 30 (6/5) (3/2) 2 50 10 90 20 = 1058 := by
  sorry

end tim_weekly_earnings_l2088_208864


namespace train_length_train_length_proof_l2088_208816

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_s

/-- Proof that a train's length is approximately 119.97 meters -/
theorem train_length_proof (speed_kmh : ℝ) (time_s : ℝ) 
  (h1 : speed_kmh = 48) 
  (h2 : time_s = 9) : 
  ∃ ε > 0, |train_length speed_kmh time_s - 119.97| < ε :=
sorry

end train_length_train_length_proof_l2088_208816


namespace prasanna_speed_l2088_208821

-- Define the speeds and distance
def laxmi_speed : ℝ := 40
def total_distance : ℝ := 78
def time : ℝ := 1

-- Theorem to prove Prasanna's speed
theorem prasanna_speed : 
  ∃ (prasanna_speed : ℝ), 
    laxmi_speed * time + prasanna_speed * time = total_distance ∧ 
    prasanna_speed = 38 := by
  sorry

end prasanna_speed_l2088_208821


namespace intersection_right_triangle_l2088_208867

/-- Given a line and a circle that intersect, and the triangle formed by the
    intersection points and the circle's center is right-angled, prove the value of a. -/
theorem intersection_right_triangle (a : ℝ) : 
  -- Line equation
  (∃ x y : ℝ, a * x - y + 6 = 0) →
  -- Circle equation
  (∃ x y : ℝ, (x + 1)^2 + (y - a)^2 = 16) →
  -- Circle center
  let C : ℝ × ℝ := (-1, a)
  -- Intersection points exist
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (a * A.1 - A.2 + 6 = 0) ∧ ((A.1 + 1)^2 + (A.2 - a)^2 = 16) ∧
    (a * B.1 - B.2 + 6 = 0) ∧ ((B.1 + 1)^2 + (B.2 - a)^2 = 16)) →
  -- Triangle ABC is right-angled
  (∃ A B : ℝ × ℝ, (A - C) • (B - C) = 0) →
  -- Conclusion
  a = 3 - Real.sqrt 2 :=
by sorry

end intersection_right_triangle_l2088_208867


namespace sqrt_two_irrational_in_set_l2088_208888

-- Define the set of numbers
def number_set : Set ℝ := {0, 1.414, Real.sqrt 2, 1/3}

-- Define irrationality
def is_irrational (x : ℝ) : Prop := ∀ p q : ℤ, q ≠ 0 → x ≠ (p : ℝ) / (q : ℝ)

-- Theorem statement
theorem sqrt_two_irrational_in_set : 
  ∃ x ∈ number_set, is_irrational x ∧ x = Real.sqrt 2 := by
sorry

end sqrt_two_irrational_in_set_l2088_208888


namespace sibling_height_l2088_208870

/-- Given information about Eliza and her siblings' heights, prove the height of the sibling with unknown height -/
theorem sibling_height (total_height : ℕ) (eliza_height : ℕ) (sibling1_height : ℕ) (sibling2_height : ℕ) :
  total_height = 330 ∧
  eliza_height = 68 ∧
  sibling1_height = 66 ∧
  sibling2_height = 66 ∧
  ∃ (unknown_sibling_height last_sibling_height : ℕ),
    unknown_sibling_height = eliza_height + 2 ∧
    total_height = eliza_height + sibling1_height + sibling2_height + unknown_sibling_height + last_sibling_height →
  ∃ (unknown_sibling_height : ℕ), unknown_sibling_height = 70 :=
by sorry

end sibling_height_l2088_208870


namespace unique_zero_of_f_l2088_208817

noncomputable def f (x : ℝ) := 2^x + x^3 - 2

theorem unique_zero_of_f :
  ∃! x : ℝ, f x = 0 :=
sorry

end unique_zero_of_f_l2088_208817


namespace total_donation_is_375_l2088_208884

/- Define the donation amounts for each company -/
def foster_farms_donation : ℕ := 45
def american_summits_donation : ℕ := 2 * foster_farms_donation
def hormel_donation : ℕ := 3 * foster_farms_donation
def boudin_butchers_donation : ℕ := hormel_donation / 3
def del_monte_foods_donation : ℕ := american_summits_donation - 30

/- Define the total donation -/
def total_donation : ℕ := 
  foster_farms_donation + 
  american_summits_donation + 
  hormel_donation + 
  boudin_butchers_donation + 
  del_monte_foods_donation

/- Theorem stating that the total donation is 375 -/
theorem total_donation_is_375 : total_donation = 375 := by
  sorry

end total_donation_is_375_l2088_208884


namespace total_amount_l2088_208851

-- Define the amounts received by A, B, and C
variable (A B C : ℝ)

-- Define the conditions
axiom condition1 : A = (1/3) * (B + C)
axiom condition2 : B = (2/7) * (A + C)
axiom condition3 : A = B + 20

-- Define the total amount
def total : ℝ := A + B + C

-- Theorem statement
theorem total_amount : total A B C = 720 := by
  sorry

end total_amount_l2088_208851


namespace m_range_l2088_208868

-- Define the propositions p and q
def p (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) < 0
def q (x : ℝ) : Prop := 1/2 < x ∧ x < 2/3

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x)

-- State the theorem
theorem m_range :
  ∀ m : ℝ, (necessary_but_not_sufficient (p m) q) ↔ (-1/3 ≤ m ∧ m ≤ 3/2) :=
sorry

end m_range_l2088_208868


namespace vector_properties_l2088_208832

/-- Given points A and B in a 2D Cartesian coordinate system, prove properties of vectors AB and OA·OB --/
theorem vector_properties (A B : ℝ × ℝ) (h1 : A = (-3, -4)) (h2 : B = (5, -12)) :
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let OA : ℝ × ℝ := A
  let OB : ℝ × ℝ := B
  (AB = (8, -8)) ∧
  (Real.sqrt ((AB.1)^2 + (AB.2)^2) = 8 * Real.sqrt 2) ∧
  (OA.1 * OB.1 + OA.2 * OB.2 = 33) := by
sorry

end vector_properties_l2088_208832


namespace crayons_per_pack_l2088_208885

theorem crayons_per_pack (num_packs : ℕ) (extra_crayons : ℕ) (total_crayons : ℕ) 
  (h1 : num_packs = 4)
  (h2 : extra_crayons = 6)
  (h3 : total_crayons = 46) :
  ∃ (crayons_per_pack : ℕ), 
    crayons_per_pack * num_packs + extra_crayons = total_crayons ∧ 
    crayons_per_pack = 10 := by
  sorry

end crayons_per_pack_l2088_208885


namespace crab_fishing_income_l2088_208839

/-- Calculate weekly income from crab fishing --/
theorem crab_fishing_income 
  (num_buckets : ℕ) 
  (crabs_per_bucket : ℕ) 
  (price_per_crab : ℕ) 
  (days_per_week : ℕ) 
  (h1 : num_buckets = 8)
  (h2 : crabs_per_bucket = 12)
  (h3 : price_per_crab = 5)
  (h4 : days_per_week = 7) :
  num_buckets * crabs_per_bucket * price_per_crab * days_per_week = 3360 := by
  sorry

end crab_fishing_income_l2088_208839


namespace equal_sides_from_tangent_sum_l2088_208829

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively
  (sum_angles : A + B + C = π)  -- Sum of angles in a triangle is π
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)  -- Sides are positive

-- State the theorem
theorem equal_sides_from_tangent_sum (t : Triangle) :
  t.a * Real.tan t.A + t.b * Real.tan t.B = (t.a + t.b) * Real.tan ((t.A + t.B) / 2) →
  t.a = t.b :=
by sorry

end equal_sides_from_tangent_sum_l2088_208829


namespace fraction_equality_l2088_208861

theorem fraction_equality (a b : ℚ) (h : a / b = 2 / 5) : (a - b) / b = -3 / 5 := by
  sorry

end fraction_equality_l2088_208861


namespace ball_ratio_l2088_208875

theorem ball_ratio (R B x : ℕ) : 
  R > 0 → B > 0 → x > 0 →
  R = (R + B + x) / 4 →
  R + x = (B + x) / 2 →
  R / B = 2 / 5 := by
  sorry

end ball_ratio_l2088_208875


namespace lcm_factor_problem_l2088_208889

theorem lcm_factor_problem (A B : ℕ+) (X : ℕ) : 
  Nat.gcd A B = 23 →
  A = 230 →
  Nat.lcm A B = 23 * X * 10 →
  X = 1 := by
sorry

end lcm_factor_problem_l2088_208889


namespace polygon_diagonals_l2088_208883

theorem polygon_diagonals (n : ℕ) (h1 : n * 10 = 360) : n * (n - 3) / 2 = 594 := by
  sorry

end polygon_diagonals_l2088_208883


namespace transformed_roots_l2088_208886

theorem transformed_roots (p : ℝ) (α β : ℝ) : 
  (3 * α^2 + 4 * α + p = 0) → 
  (3 * β^2 + 4 * β + p = 0) → 
  ((α / 3 - 2)^2 + 16 * (α / 3 - 2) + (60 + 3 * p) = 0) ∧
  ((β / 3 - 2)^2 + 16 * (β / 3 - 2) + (60 + 3 * p) = 0) := by
  sorry

end transformed_roots_l2088_208886


namespace resort_tips_l2088_208880

theorem resort_tips (total_months : ℕ) (other_months : ℕ) (avg_other_tips : ℝ) (aug_tips : ℝ) :
  total_months = other_months + 1 →
  aug_tips = 0.5 * (aug_tips + other_months * avg_other_tips) →
  aug_tips = 6 * avg_other_tips :=
by
  sorry

end resort_tips_l2088_208880


namespace organize_four_men_five_women_l2088_208823

/-- The number of ways to organize men and women into groups -/
def organize_groups (num_men : ℕ) (num_women : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the correct number of ways to organize the groups -/
theorem organize_four_men_five_women :
  organize_groups 4 5 = 180 := by
  sorry

end organize_four_men_five_women_l2088_208823


namespace work_fraction_proof_l2088_208840

theorem work_fraction_proof (total_payment : ℚ) (b_payment : ℚ) 
  (h1 : total_payment = 529)
  (h2 : b_payment = 12) :
  (total_payment - b_payment) / total_payment = 517 / 529 := by
  sorry

end work_fraction_proof_l2088_208840


namespace race_time_proof_l2088_208838

/-- 
Proves that in a 1000-meter race where runner A finishes 200 meters ahead of runner B, 
and the time difference between their finishes is 10 seconds, 
the time taken by runner A to complete the race is 50 seconds.
-/
theorem race_time_proof (race_length : ℝ) (distance_diff : ℝ) (time_diff : ℝ) 
  (h1 : race_length = 1000)
  (h2 : distance_diff = 200)
  (h3 : time_diff = 10) : 
  ∃ (time_A : ℝ), time_A = 50 ∧ 
    race_length / time_A = (race_length - distance_diff) / (time_A + time_diff) :=
by
  sorry

#check race_time_proof

end race_time_proof_l2088_208838


namespace chocolate_box_pieces_l2088_208869

theorem chocolate_box_pieces (initial_boxes : ℕ) (given_away : ℕ) (remaining_pieces : ℕ) :
  initial_boxes = 7 →
  given_away = 3 →
  remaining_pieces = 16 →
  ∃ (pieces_per_box : ℕ), 
    pieces_per_box * (initial_boxes - given_away) = remaining_pieces ∧
    pieces_per_box = 4 :=
by sorry

end chocolate_box_pieces_l2088_208869


namespace largest_prime_divisor_to_test_l2088_208899

theorem largest_prime_divisor_to_test (n : ℕ) (h : 800 ≤ n ∧ n ≤ 850) :
  (∀ p : ℕ, p ≤ 29 → Prime p → ¬(p ∣ n)) → Prime n :=
sorry

end largest_prime_divisor_to_test_l2088_208899


namespace exist_four_distinct_numbers_perfect_squares_l2088_208871

theorem exist_four_distinct_numbers_perfect_squares : 
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (m n : ℕ), 
      a^2 + 2*c*d + b^2 = m^2 ∧
      c^2 + 2*a*b + d^2 = n^2 :=
by sorry

end exist_four_distinct_numbers_perfect_squares_l2088_208871


namespace perpendicular_line_equation_l2088_208878

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The general form of a line equation: ax + by + c = 0 -/
structure GeneralLineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- Convert a line from slope-intercept form to general form -/
def toGeneralForm (l : Line) : GeneralLineEquation :=
  { a := l.slope, b := -1, c := l.y_intercept }

/-- The main theorem -/
theorem perpendicular_line_equation 
  (l : Line) 
  (h1 : l.y_intercept = 2) 
  (h2 : perpendicular l { slope := -1, y_intercept := 3 }) : 
  toGeneralForm l = { a := 1, b := -1, c := 2 } := by
  sorry

end perpendicular_line_equation_l2088_208878


namespace principal_amount_proof_l2088_208865

-- Define the interest rates for each year
def r1 : ℝ := 0.08
def r2 : ℝ := 0.10
def r3 : ℝ := 0.12
def r4 : ℝ := 0.09
def r5 : ℝ := 0.11

-- Define the total compound interest
def total_interest : ℝ := 4016.25

-- Define the compound interest factor
def compound_factor : ℝ := (1 + r1) * (1 + r2) * (1 + r3) * (1 + r4) * (1 + r5)

-- State the theorem
theorem principal_amount_proof :
  ∃ P : ℝ, P * (compound_factor - 1) = total_interest ∧ 
  abs (P - 7065.84) < 0.01 := by
sorry

end principal_amount_proof_l2088_208865


namespace sam_spent_pennies_l2088_208837

/-- Given that Sam initially had 98 pennies and now has 5 pennies left,
    prove that he spent 93 pennies. -/
theorem sam_spent_pennies (initial : Nat) (remaining : Nat) (spent : Nat)
    (h1 : initial = 98)
    (h2 : remaining = 5)
    (h3 : spent = initial - remaining) :
  spent = 93 := by
  sorry

end sam_spent_pennies_l2088_208837


namespace intersection_M_N_l2088_208830

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x : ℝ | x^2 ≤ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end intersection_M_N_l2088_208830
