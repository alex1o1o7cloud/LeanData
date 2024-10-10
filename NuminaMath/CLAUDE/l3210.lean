import Mathlib

namespace more_boys_than_girls_boy_girl_difference_l3210_321063

/-- The number of girls in the school -/
def num_girls : ℕ := 635

/-- The number of boys in the school -/
def num_boys : ℕ := 1145

/-- There are more boys than girls -/
theorem more_boys_than_girls : num_boys > num_girls := by sorry

/-- The difference between the number of boys and girls is 510 -/
theorem boy_girl_difference : num_boys - num_girls = 510 := by sorry

end more_boys_than_girls_boy_girl_difference_l3210_321063


namespace smallest_integer_fraction_thirteen_satisfies_smallest_integer_is_thirteen_l3210_321021

theorem smallest_integer_fraction (y : ℤ) : (8 : ℚ) / 11 < (y : ℚ) / 17 → y ≥ 13 :=
by
  sorry

theorem thirteen_satisfies (y : ℤ) : (8 : ℚ) / 11 < (13 : ℚ) / 17 :=
by
  sorry

theorem smallest_integer_is_thirteen : ∃ y : ℤ, ((8 : ℚ) / 11 < (y : ℚ) / 17) ∧ (∀ z : ℤ, (8 : ℚ) / 11 < (z : ℚ) / 17 → z ≥ y) ∧ y = 13 :=
by
  sorry

end smallest_integer_fraction_thirteen_satisfies_smallest_integer_is_thirteen_l3210_321021


namespace arithmetic_sequence_general_term_l3210_321003

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The general term of an arithmetic sequence. -/
def arithmetic_general_term (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a 1 + (n - 1) * (a 2 - a 1)

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum1 : a 2 + a 6 = 8)
  (h_sum2 : a 3 + a 4 = 3) :
  ∀ n : ℕ, arithmetic_general_term a n = 5 * n - 16 :=
sorry

end arithmetic_sequence_general_term_l3210_321003


namespace find_divisor_l3210_321070

theorem find_divisor (N : ℝ) (D : ℝ) (h1 : (N - 6) / D = 2) (h2 : N = 32) : D = 13 := by
  sorry

end find_divisor_l3210_321070


namespace spinner_points_north_l3210_321016

/-- Represents the four cardinal directions -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a rotation of the spinner -/
def rotate (initial : Direction) (revolutions : ℚ) : Direction :=
  sorry

/-- Theorem stating that after the described rotations, the spinner points north -/
theorem spinner_points_north :
  let initial_direction := Direction.North
  let clockwise_rotation := 7/2
  let counterclockwise_rotation := 5/2
  rotate (rotate initial_direction clockwise_rotation) (-counterclockwise_rotation) = Direction.North :=
by sorry

end spinner_points_north_l3210_321016


namespace diana_weekly_earnings_l3210_321055

/-- Represents Diana's work schedule and earnings --/
structure DianaWork where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  hourly_rate : ℕ

/-- Calculates Diana's weekly earnings based on her work schedule --/
def weekly_earnings (d : DianaWork) : ℕ :=
  (d.monday_hours + d.tuesday_hours + d.wednesday_hours + d.thursday_hours + d.friday_hours) * d.hourly_rate

/-- Diana's actual work schedule --/
def diana : DianaWork :=
  { monday_hours := 10
    tuesday_hours := 15
    wednesday_hours := 10
    thursday_hours := 15
    friday_hours := 10
    hourly_rate := 30 }

/-- Theorem stating that Diana's weekly earnings are $1800 --/
theorem diana_weekly_earnings :
  weekly_earnings diana = 1800 := by
  sorry


end diana_weekly_earnings_l3210_321055


namespace min_intersection_size_l3210_321020

theorem min_intersection_size (U B P : Finset ℕ) 
  (h1 : U.card = 25)
  (h2 : B ⊆ U)
  (h3 : P ⊆ U)
  (h4 : B.card = 15)
  (h5 : P.card = 18) :
  (B ∩ P).card ≥ 8 := by
sorry

end min_intersection_size_l3210_321020


namespace percentage_kindergarten_combined_l3210_321014

/-- Percentage of Kindergarten students in combined schools -/
theorem percentage_kindergarten_combined (pinegrove_total : ℕ) (maplewood_total : ℕ)
  (pinegrove_k_percent : ℚ) (maplewood_k_percent : ℚ)
  (h1 : pinegrove_total = 150)
  (h2 : maplewood_total = 250)
  (h3 : pinegrove_k_percent = 18/100)
  (h4 : maplewood_k_percent = 14/100) :
  (pinegrove_k_percent * pinegrove_total + maplewood_k_percent * maplewood_total) /
  (pinegrove_total + maplewood_total) = 155/1000 := by
  sorry

#check percentage_kindergarten_combined

end percentage_kindergarten_combined_l3210_321014


namespace inequality_equivalence_l3210_321098

theorem inequality_equivalence (x : ℝ) : 3 * x^2 + x < 8 ↔ -2 < x ∧ x < 4/3 := by
  sorry

end inequality_equivalence_l3210_321098


namespace sample_data_properties_l3210_321060

theorem sample_data_properties (x : Fin 6 → ℝ) 
  (h_ordered : ∀ i j, i < j → x i ≤ x j) : 
  (((x 2 + x 3) / 2 = (x 3 + x 4) / 2) ∧ 
  (x 5 - x 2 ≤ x 6 - x 1)) := by sorry

end sample_data_properties_l3210_321060


namespace fourth_grade_students_l3210_321009

theorem fourth_grade_students (initial : ℕ) (left : ℕ) (new : ℕ) : 
  initial = 10 → left = 4 → new = 42 → initial - left + new = 48 := by
sorry

end fourth_grade_students_l3210_321009


namespace extreme_point_condition_monotonicity_for_maximum_two_solutions_condition_l3210_321036

noncomputable section

-- Define the function f
def f (c b x : ℝ) : ℝ := c * Real.log x + 0.5 * x^2 + b * x

-- Define the derivative of f
def f' (c b x : ℝ) : ℝ := c / x + x + b

theorem extreme_point_condition (c b : ℝ) (hc : c ≠ 0) : 
  f' c b 1 = 0 ↔ b + c + 1 = 0 :=
sorry

theorem monotonicity_for_maximum (c b : ℝ) (hc : c ≠ 0) (hmax : c > 1) :
  (∀ x, 0 < x → x < 1 → (f' c b x > 0)) ∧ 
  (∀ x, 1 < x → x < c → (f' c b x < 0)) ∧
  (∀ x, x > c → (f' c b x > 0)) :=
sorry

theorem two_solutions_condition (c : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f c (-1-c) x₁ = 0 ∧ f c (-1-c) x₂ = 0 ∧ 
   (∀ x, f c (-1-c) x = 0 → x = x₁ ∨ x = x₂)) ↔ 
  (-0.5 < c ∧ c < 0) :=
sorry

end extreme_point_condition_monotonicity_for_maximum_two_solutions_condition_l3210_321036


namespace city_mpg_is_32_l3210_321083

/-- Represents the fuel efficiency of a car in different driving conditions -/
structure CarFuelEfficiency where
  highway_miles_per_tank : ℝ
  city_miles_per_tank : ℝ
  highway_city_mpg_difference : ℝ

/-- Calculates the city miles per gallon given the car's fuel efficiency data -/
def city_mpg (car : CarFuelEfficiency) : ℝ :=
  sorry

/-- Theorem stating that for the given car data, the city MPG is 32 -/
theorem city_mpg_is_32 (car : CarFuelEfficiency)
  (h1 : car.highway_miles_per_tank = 462)
  (h2 : car.city_miles_per_tank = 336)
  (h3 : car.highway_city_mpg_difference = 12) :
  city_mpg car = 32 := by
  sorry

end city_mpg_is_32_l3210_321083


namespace min_voters_for_tall_giraffe_win_l3210_321085

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingStructure where
  total_voters : ℕ
  num_districts : ℕ
  precincts_per_district : ℕ
  voters_per_precinct : ℕ

/-- Calculates the minimum number of voters required to win -/
def min_voters_to_win (vs : VotingStructure) : ℕ :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let precincts_to_win_per_district := (vs.precincts_per_district + 1) / 2
  let voters_to_win_per_precinct := (vs.voters_per_precinct + 1) / 2
  districts_to_win * precincts_to_win_per_district * voters_to_win_per_precinct

/-- The theorem stating the minimum number of voters for the Tall giraffe to win -/
theorem min_voters_for_tall_giraffe_win (vs : VotingStructure) 
  (h1 : vs.total_voters = 135)
  (h2 : vs.num_districts = 5)
  (h3 : vs.precincts_per_district = 9)
  (h4 : vs.voters_per_precinct = 3)
  (h5 : vs.total_voters = vs.num_districts * vs.precincts_per_district * vs.voters_per_precinct) :
  min_voters_to_win vs = 30 := by
  sorry

end min_voters_for_tall_giraffe_win_l3210_321085


namespace train_speed_proof_l3210_321086

/-- Proves that a train crossing a bridge has a speed of approximately 36 kmph given specific conditions. -/
theorem train_speed_proof (train_length bridge_length time_to_cross : ℝ) 
  (h1 : train_length = 140)
  (h2 : bridge_length = 150)
  (h3 : time_to_cross = 28.997680185585153) : 
  ∃ (speed : ℝ), abs (speed - 36) < 0.1 := by
  sorry

end train_speed_proof_l3210_321086


namespace unique_solution_to_equation_l3210_321058

theorem unique_solution_to_equation :
  ∃! y : ℝ, y ≠ 2 ∧ y ≠ -2 ∧
  (-12 * y) / (y^2 - 4) = (3 * y) / (y + 2) - 9 / (y - 2) ∧
  y = 3 := by
  sorry

end unique_solution_to_equation_l3210_321058


namespace carpet_cost_proof_l3210_321010

theorem carpet_cost_proof (floor_length floor_width carpet_side_length carpet_cost : ℝ) 
  (h1 : floor_length = 24)
  (h2 : floor_width = 64)
  (h3 : carpet_side_length = 8)
  (h4 : carpet_cost = 24) : 
  (floor_length * floor_width) / (carpet_side_length * carpet_side_length) * carpet_cost = 576 := by
  sorry

end carpet_cost_proof_l3210_321010


namespace class_average_mark_l3210_321068

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) (excluded_average : ℝ) (remaining_average : ℝ) : 
  total_students = 13 →
  excluded_students = 5 →
  excluded_average = 40 →
  remaining_average = 92 →
  (total_students : ℝ) * (total_students * (remaining_average : ℝ) - excluded_students * excluded_average) / 
    (total_students * (total_students - excluded_students)) = 72 := by
  sorry


end class_average_mark_l3210_321068


namespace constant_term_binomial_expansion_l3210_321099

/-- The constant term in the binomial expansion of (3x^2 - 2/x^3)^5 is 1080 -/
theorem constant_term_binomial_expansion :
  let f (x : ℝ) := (3 * x^2 - 2 / x^3)^5
  ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 0 → f x = c + x * (f x - c) / x ∧ c = 1080 :=
sorry

end constant_term_binomial_expansion_l3210_321099


namespace reuleaux_triangle_fits_all_holes_l3210_321048

-- Define a Reuleaux Triangle
structure ReuleauxTriangle where
  -- Add necessary properties of a Reuleaux Triangle
  constant_width : ℝ

-- Define the types of holes
inductive HoleType
  | Triangular
  | Square
  | Circular

-- Define a function to check if a shape fits into a hole
def fits_into (shape : ReuleauxTriangle) (hole : HoleType) : Prop :=
  match hole with
  | HoleType.Triangular => true -- Assume it fits into triangular hole
  | HoleType.Square => true     -- Assume it fits into square hole
  | HoleType.Circular => true   -- Assume it fits into circular hole

-- Theorem statement
theorem reuleaux_triangle_fits_all_holes (r : ReuleauxTriangle) :
  (∀ (h : HoleType), fits_into r h) :=
sorry

end reuleaux_triangle_fits_all_holes_l3210_321048


namespace fourth_term_is_two_l3210_321025

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  a_1_eq : a 1 = 16
  a_6_eq : a 6 = 2 * a 5 * a 7

/-- The fourth term of the geometric sequence is 2 -/
theorem fourth_term_is_two (seq : GeometricSequence) : seq.a 4 = 2 := by
  sorry


end fourth_term_is_two_l3210_321025


namespace door_probability_l3210_321019

/-- The probability of exactly k successes in n independent trials 
    with probability p for each trial -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

theorem door_probability : 
  binomial_probability 5 2 (1/2) = 5/16 := by
  sorry

end door_probability_l3210_321019


namespace min_sum_m_n_l3210_321011

theorem min_sum_m_n (m n : ℕ+) (h : 98 * m = n^3) : 
  (∀ (m' n' : ℕ+), 98 * m' = n'^3 → m' + n' ≥ m + n) → m + n = 42 :=
by sorry

end min_sum_m_n_l3210_321011


namespace circle_equation_radius_8_l3210_321007

/-- The equation x^2 + 14x + y^2 + 10y - k = 0 represents a circle of radius 8 if and only if k = 10 -/
theorem circle_equation_radius_8 (x y k : ℝ) : 
  (∃ h₁ h₂ : ℝ, ∀ x y : ℝ, x^2 + 14*x + y^2 + 10*y - k = 0 ↔ (x - h₁)^2 + (y - h₂)^2 = 64) ↔ 
  k = 10 := by
sorry

end circle_equation_radius_8_l3210_321007


namespace january_salary_l3210_321029

/-- Given the average salaries for two four-month periods and the salary for May,
    prove that the salary for January is 5700. -/
theorem january_salary
  (avg_jan_to_apr : (jan + feb + mar + apr) / 4 = 8000)
  (avg_feb_to_may : (feb + mar + apr + may) / 4 = 8200)
  (may_salary : may = 6500)
  : jan = 5700 := by
  sorry

end january_salary_l3210_321029


namespace ladder_problem_l3210_321018

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end ladder_problem_l3210_321018


namespace bernardo_silvia_game_l3210_321054

theorem bernardo_silvia_game (N : ℕ) : N = 24 ↔ 
  (N ≤ 999) ∧ 
  (3 * N < 800) ∧ 
  (3 * N - 30 < 800) ∧ 
  (9 * N - 90 < 800) ∧ 
  (9 * N - 120 < 800) ∧ 
  (27 * N - 360 < 800) ∧ 
  (27 * N - 390 < 800) ∧ 
  (81 * N - 1170 ≥ 800) ∧ 
  (∀ m : ℕ, m < N → 
    (3 * m < 800) ∧ 
    (3 * m - 30 < 800) ∧ 
    (9 * m - 90 < 800) ∧ 
    (9 * m - 120 < 800) ∧ 
    (27 * m - 360 < 800) ∧ 
    (27 * m - 390 < 800) ∧ 
    (81 * m - 1170 < 800)) := by
  sorry

end bernardo_silvia_game_l3210_321054


namespace expand_product_l3210_321008

theorem expand_product (x : ℝ) : (x - 3) * (x + 4) = x^2 + x - 12 := by
  sorry

end expand_product_l3210_321008


namespace inequality_solution_set_l3210_321038

theorem inequality_solution_set :
  {x : ℝ | (1 : ℝ) / (x - 1) < -1} = Set.Ioo 0 1 := by sorry

end inequality_solution_set_l3210_321038


namespace sphere_surface_area_from_volume_l3210_321091

theorem sphere_surface_area_from_volume (V : ℝ) (r : ℝ) (S : ℝ) : 
  V = 72 * Real.pi → 
  V = (4 / 3) * Real.pi * r^3 → 
  S = 4 * Real.pi * r^2 → 
  S = 36 * Real.pi * 2^(2/3) := by
  sorry

end sphere_surface_area_from_volume_l3210_321091


namespace benny_missed_games_l3210_321052

/-- The number of baseball games Benny missed -/
def games_missed (total_games attended_games : ℕ) : ℕ :=
  total_games - attended_games

/-- Proof that Benny missed 25 games -/
theorem benny_missed_games :
  let total_games : ℕ := 39
  let attended_games : ℕ := 14
  games_missed total_games attended_games = 25 := by
  sorry

end benny_missed_games_l3210_321052


namespace remainder_14_div_5_l3210_321023

theorem remainder_14_div_5 : 14 % 5 = 4 := by
  sorry

end remainder_14_div_5_l3210_321023


namespace four_coin_stacking_methods_l3210_321045

/-- Represents a coin with two sides -/
inductive Coin
| Head
| Tail

/-- Represents a stack of coins -/
def CoinStack := List Coin

/-- Checks if a given coin stack is valid (no adjacent heads) -/
def is_valid_stack (stack : CoinStack) : Bool :=
  match stack with
  | [] => true
  | [_] => true
  | Coin.Head :: Coin.Head :: _ => false
  | _ :: rest => is_valid_stack rest

/-- Generates all possible coin stacks of a given length -/
def generate_stacks (n : Nat) : List CoinStack :=
  if n = 0 then [[]]
  else
    let prev_stacks := generate_stacks (n - 1)
    prev_stacks.bind (fun stack => [Coin.Head :: stack, Coin.Tail :: stack])

/-- Counts the number of valid coin stacks of a given length -/
def count_valid_stacks (n : Nat) : Nat :=
  (generate_stacks n).filter is_valid_stack |>.length

/-- The main theorem to be proved -/
theorem four_coin_stacking_methods :
  count_valid_stacks 4 = 5 := by
  sorry

end four_coin_stacking_methods_l3210_321045


namespace cookie_distribution_l3210_321001

theorem cookie_distribution (total : ℚ) (blue green red : ℚ) : 
  blue + green + red = total →
  blue + green = 2/3 * total →
  blue = 1/4 * total →
  green / (blue + green) = 5/8 := by
sorry

end cookie_distribution_l3210_321001


namespace round_trip_speed_calculation_l3210_321062

/-- Proves that given a round trip of 240 miles with a total travel time of 5.4 hours,
    where the return trip speed is 50 miles per hour, the outbound trip speed is 40 miles per hour. -/
theorem round_trip_speed_calculation (total_distance : ℝ) (total_time : ℝ) (return_speed : ℝ) :
  total_distance = 240 →
  total_time = 5.4 →
  return_speed = 50 →
  ∃ (outbound_speed : ℝ),
    outbound_speed = 40 ∧
    total_time = (total_distance / 2) / outbound_speed + (total_distance / 2) / return_speed :=
by sorry

end round_trip_speed_calculation_l3210_321062


namespace shortest_side_length_l3210_321076

theorem shortest_side_length (a b c : ℝ) : 
  a + b + c = 15 ∧ a = 2 * c ∧ b = 2 * c → c = 3 :=
by
  sorry

end shortest_side_length_l3210_321076


namespace women_at_gathering_l3210_321035

/-- The number of women at a social gathering --/
def number_of_women (number_of_men : ℕ) (dances_per_man : ℕ) (dances_per_woman : ℕ) : ℕ :=
  (number_of_men * dances_per_man) / dances_per_woman

/-- Theorem: At a social gathering with the given conditions, 20 women attended --/
theorem women_at_gathering :
  let number_of_men : ℕ := 15
  let dances_per_man : ℕ := 4
  let dances_per_woman : ℕ := 3
  number_of_women number_of_men dances_per_man dances_per_woman = 20 := by
sorry

#eval number_of_women 15 4 3

end women_at_gathering_l3210_321035


namespace sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l3210_321022

theorem sqrt_twelve_minus_sqrt_three_equals_sqrt_three : 
  Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l3210_321022


namespace eight_coin_flips_sequences_l3210_321006

/-- The number of distinct sequences for n coin flips -/
def coin_sequences (n : ℕ) : ℕ := 2^n

/-- Theorem stating that the number of distinct sequences for 8 coin flips is 256 -/
theorem eight_coin_flips_sequences : coin_sequences 8 = 256 := by
  sorry

end eight_coin_flips_sequences_l3210_321006


namespace sin_2theta_third_quadrant_l3210_321041

theorem sin_2theta_third_quadrant (θ : Real) :
  (π < θ ∧ θ < 3*π/2) →  -- θ is in the third quadrant
  (Real.sin θ)^4 + (Real.cos θ)^4 = 5/9 →
  Real.sin (2*θ) = -2*Real.sqrt 2/3 :=
by
  sorry

end sin_2theta_third_quadrant_l3210_321041


namespace root_of_equations_l3210_321053

theorem root_of_equations (a b c d e k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0)
  (eq1 : a * k^4 + b * k^3 + c * k^2 + d * k + e = 0)
  (eq2 : b * k^4 + c * k^3 + d * k^2 + e * k + a = 0) :
  k^5 = 1 :=
sorry

end root_of_equations_l3210_321053


namespace geometric_sequence_product_roots_product_geometric_sequence_problem_l3210_321095

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) (h : geometric_sequence a) :
  ∀ i j k l : ℕ, i + j = k + l → a i * a j = a k * a l :=
sorry

theorem roots_product (p q r : ℝ) (x y : ℝ) (hx : p * x^2 + q * x + r = 0) (hy : p * y^2 + q * y + r = 0) :
  x * y = r / p :=
sorry

theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a)
  (h_roots : 3 * a 1^2 - 2 * a 1 - 6 = 0 ∧ 3 * a 10^2 - 2 * a 10 - 6 = 0) :
  a 4 * a 7 = -2 :=
sorry

end geometric_sequence_product_roots_product_geometric_sequence_problem_l3210_321095


namespace product_and_sum_of_factors_l3210_321084

theorem product_and_sum_of_factors : ∃ a b : ℕ, 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8775 ∧ 
  a + b = 110 := by
sorry

end product_and_sum_of_factors_l3210_321084


namespace xyz_product_l3210_321073

theorem xyz_product (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h1 : x * (y + z) = 198)
  (h2 : y * (z + x) = 216)
  (h3 : z * (x + y) = 234) :
  x * y * z = 1080 := by
sorry

end xyz_product_l3210_321073


namespace exponent_rules_l3210_321033

theorem exponent_rules :
  (∀ x : ℝ, x^5 * x^2 = x^7) ∧
  (∀ m : ℝ, (m^2)^4 = m^8) ∧
  (∀ x y : ℝ, (-2*x*y^2)^3 = -8*x^3*y^6) := by
  sorry

end exponent_rules_l3210_321033


namespace blake_bought_six_chocolate_packs_l3210_321051

/-- The number of lollipops Blake bought -/
def lollipops : ℕ := 4

/-- The cost of one lollipop in dollars -/
def lollipop_cost : ℕ := 2

/-- The number of $10 bills Blake gave to the cashier -/
def bills_given : ℕ := 6

/-- The amount of change Blake received in dollars -/
def change_received : ℕ := 4

/-- The cost of one pack of chocolate in terms of lollipops -/
def chocolate_pack_cost : ℕ := 4 * lollipop_cost

/-- The total amount Blake spent in dollars -/
def total_spent : ℕ := bills_given * 10 - change_received

/-- Theorem stating that Blake bought 6 packs of chocolate -/
theorem blake_bought_six_chocolate_packs : 
  (total_spent - lollipops * lollipop_cost) / chocolate_pack_cost = 6 := by
  sorry

end blake_bought_six_chocolate_packs_l3210_321051


namespace douglas_weight_is_52_l3210_321031

/-- Anne's weight in pounds -/
def anne_weight : ℕ := 67

/-- The difference in weight between Anne and Douglas in pounds -/
def weight_difference : ℕ := 15

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := anne_weight - weight_difference

/-- Theorem stating Douglas's weight -/
theorem douglas_weight_is_52 : douglas_weight = 52 := by
  sorry

end douglas_weight_is_52_l3210_321031


namespace power_of_power_l3210_321065

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l3210_321065


namespace mary_screw_sections_l3210_321024

def number_of_sections (initial_screws : ℕ) (multiplier : ℕ) (screws_per_section : ℕ) : ℕ :=
  (initial_screws + initial_screws * multiplier) / screws_per_section

theorem mary_screw_sections :
  number_of_sections 8 2 6 = 4 := by
  sorry

end mary_screw_sections_l3210_321024


namespace edge_stop_probability_l3210_321064

-- Define the grid size
def gridSize : Nat := 4

-- Define a position on the grid
structure Position where
  x : Nat
  y : Nat
  deriving Repr

-- Define the possible directions
inductive Direction
  | Up
  | Down
  | Left
  | Right

-- Define whether a position is on the edge
def isEdge (pos : Position) : Bool :=
  pos.x == 1 || pos.x == gridSize || pos.y == 1 || pos.y == gridSize

-- Define the next position after a move, with wrap-around
def nextPosition (pos : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.Up => ⟨pos.x, if pos.y == gridSize then 1 else pos.y + 1⟩
  | Direction.Down => ⟨pos.x, if pos.y == 1 then gridSize else pos.y - 1⟩
  | Direction.Left => ⟨if pos.x == 1 then gridSize else pos.x - 1, pos.y⟩
  | Direction.Right => ⟨if pos.x == gridSize then 1 else pos.x + 1, pos.y⟩

-- Define the probability of stopping at an edge within n hops
def probStopAtEdge (start : Position) (n : Nat) : Real :=
  sorry

-- Theorem statement
theorem edge_stop_probability :
  probStopAtEdge ⟨2, 1⟩ 5 =
    probStopAtEdge ⟨2, 1⟩ 1 +
    probStopAtEdge ⟨2, 1⟩ 2 +
    probStopAtEdge ⟨2, 1⟩ 3 +
    probStopAtEdge ⟨2, 1⟩ 4 +
    probStopAtEdge ⟨2, 1⟩ 5 :=
  sorry

end edge_stop_probability_l3210_321064


namespace prove_initial_person_count_l3210_321072

/-- The initial number of persons in a group where:
  - The average weight increase is 4.2 kg when a new person replaces one of the original group.
  - The weight of the person leaving is 65 kg.
  - The weight of the new person is 98.6 kg.
-/
def initialPersonCount : ℕ := 8

theorem prove_initial_person_count :
  let avgWeightIncrease : ℚ := 21/5
  let oldPersonWeight : ℚ := 65
  let newPersonWeight : ℚ := 493/5
  (newPersonWeight - oldPersonWeight) / avgWeightIncrease = initialPersonCount := by
  sorry

end prove_initial_person_count_l3210_321072


namespace al_sandwich_options_l3210_321074

-- Define the types of ingredients
structure Ingredients :=
  (bread : Nat)
  (meat : Nat)
  (cheese : Nat)

-- Define the restrictions
structure Restrictions :=
  (turkey_swiss : Nat)
  (rye_roast_beef : Nat)

-- Define the function to calculate the number of sandwiches
def calculate_sandwiches (i : Ingredients) (r : Restrictions) : Nat :=
  i.bread * i.meat * i.cheese - r.turkey_swiss - r.rye_roast_beef

-- Theorem statement
theorem al_sandwich_options (i : Ingredients) (r : Restrictions) 
  (h1 : i.bread = 5)
  (h2 : i.meat = 7)
  (h3 : i.cheese = 6)
  (h4 : r.turkey_swiss = 5)
  (h5 : r.rye_roast_beef = 6) :
  calculate_sandwiches i r = 199 := by
  sorry


end al_sandwich_options_l3210_321074


namespace sum_of_second_progression_l3210_321089

/-- Given two arithmetic progressions with specific conditions, prove that the sum of the terms of the second progression is 14. -/
theorem sum_of_second_progression (a₁ a₅ b₁ bₙ : ℚ) (N : ℕ) : 
  a₁ = 7 →
  a₅ = -5 →
  b₁ = 0 →
  bₙ = 7/2 →
  N > 1 →
  (∃ d D : ℚ, a₁ + 2*d = b₁ + 2*D ∧ a₅ = a₁ + 4*d ∧ bₙ = b₁ + (N-1)*D) →
  (N/2 : ℚ) * (b₁ + bₙ) = 14 := by
  sorry

end sum_of_second_progression_l3210_321089


namespace belt_and_road_population_scientific_notation_l3210_321093

theorem belt_and_road_population_scientific_notation :
  let billion : ℝ := 10^9
  4.4 * billion = 4.4 * 10^9 := by
  sorry

end belt_and_road_population_scientific_notation_l3210_321093


namespace school_average_difference_l3210_321013

theorem school_average_difference : 
  let total_students : ℕ := 120
  let total_teachers : ℕ := 6
  let class_sizes : List ℕ := [60, 30, 15, 10, 3, 2]
  let t : ℚ := (total_students : ℚ) / total_teachers
  let s : ℚ := (class_sizes.map (λ size => (size : ℚ) * size / total_students)).sum
  t - s = -20316 / 1000 := by
sorry

end school_average_difference_l3210_321013


namespace domestic_tourists_scientific_notation_l3210_321005

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem domestic_tourists_scientific_notation :
  toScientificNotation 274000000 =
    ScientificNotation.mk 2.74 8 (by norm_num) :=
by sorry

end domestic_tourists_scientific_notation_l3210_321005


namespace point_A_coordinates_l3210_321092

/-- Given a point A(m, 2) on the line y = 2x - 4, prove that its coordinates are (3, 2) -/
theorem point_A_coordinates :
  ∀ m : ℝ, (2 : ℝ) = 2 * m - 4 → m = 3 ∧ (3, 2) = (m, 2) := by
  sorry

end point_A_coordinates_l3210_321092


namespace tank_weight_calculation_l3210_321057

def tank_capacity : ℝ := 200
def empty_tank_weight : ℝ := 80
def fill_percentage : ℝ := 0.8
def water_weight_per_gallon : ℝ := 8

theorem tank_weight_calculation : 
  let water_volume : ℝ := tank_capacity * fill_percentage
  let water_weight : ℝ := water_volume * water_weight_per_gallon
  let total_weight : ℝ := empty_tank_weight + water_weight
  total_weight = 1360 := by sorry

end tank_weight_calculation_l3210_321057


namespace quadratic_root_difference_l3210_321000

theorem quadratic_root_difference : 
  let a : ℝ := 6 + 3 * Real.sqrt 5
  let b : ℝ := -(3 + Real.sqrt 5)
  let c : ℝ := 1
  let discriminant := b^2 - 4*a*c
  let root_difference := Real.sqrt discriminant / a
  root_difference = (Real.sqrt 6 - Real.sqrt 5) / 3 := by
  sorry

end quadratic_root_difference_l3210_321000


namespace ratio_A_B_between_zero_and_one_l3210_321050

def A : ℕ := 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28
def B : ℕ := 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20

theorem ratio_A_B_between_zero_and_one : 0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 := by
  sorry

end ratio_A_B_between_zero_and_one_l3210_321050


namespace min_value_theorem_max_value_theorem_l3210_321040

-- Part 1
theorem min_value_theorem (x : ℝ) (hx : x > 0) : 12 / x + 3 * x ≥ 12 := by
  sorry

-- Part 2
theorem max_value_theorem (x : ℝ) (hx1 : x > 0) (hx2 : x < 1/3) : x * (1 - 3 * x) ≤ 1/12 := by
  sorry

end min_value_theorem_max_value_theorem_l3210_321040


namespace race_solution_l3210_321030

/-- A race between two runners A and B -/
structure Race where
  /-- The total distance of the race in meters -/
  distance : ℝ
  /-- The time it takes runner A to complete the race in seconds -/
  time_A : ℝ
  /-- The difference in distance between A and B at the finish line in meters -/
  distance_diff : ℝ
  /-- The difference in time between A and B at the finish line in seconds -/
  time_diff : ℝ

/-- The theorem stating the properties of the race and its solution -/
theorem race_solution (race : Race)
  (h1 : race.time_A = 23)
  (h2 : race.distance_diff = 56 ∨ race.time_diff = 7) :
  race.distance = 56 := by
  sorry


end race_solution_l3210_321030


namespace congruence_solution_l3210_321097

theorem congruence_solution (n : ℤ) : 
  0 ≤ n ∧ n < 203 ∧ (150 * n) % 203 = 95 % 203 → n = 144 := by sorry

end congruence_solution_l3210_321097


namespace courtyard_length_l3210_321094

/-- The length of a rectangular courtyard given its width and paving details. -/
theorem courtyard_length (width : ℝ) (brick_length brick_width : ℝ) (total_bricks : ℕ) : 
  width = 16 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  total_bricks = 16000 →
  width * (total_bricks : ℝ) * brick_length * brick_width / width = 20 := by
  sorry

#check courtyard_length

end courtyard_length_l3210_321094


namespace music_stand_cost_l3210_321067

/-- The cost of Jason's music stand, given his total spending and the costs of other items. -/
theorem music_stand_cost (total_spent flute_cost book_cost : ℚ) 
  (h1 : total_spent = 158.35)
  (h2 : flute_cost = 142.46)
  (h3 : book_cost = 7) :
  total_spent - (flute_cost + book_cost) = 8.89 := by
  sorry

end music_stand_cost_l3210_321067


namespace diophantine_equation_solutions_l3210_321049

theorem diophantine_equation_solutions :
  let S : Set (ℤ × ℤ) := {(4, -1), (-26, -9), (-16, -9), (-6, -1), (50, 15), (-72, -25)}
  ∀ (x y : ℤ), (x^2 - 5*x*y + 6*y^2 - 3*x + 5*y - 25 = 0) ↔ (x, y) ∈ S :=
by sorry

end diophantine_equation_solutions_l3210_321049


namespace units_digit_of_m_squared_plus_two_to_m_l3210_321066

/-- The units digit of m^2 + 2^m is 7, where m = 2021^2 + 3^2021 -/
theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : m = 2021^2 + 3^2021 → (m^2 + 2^m) % 10 = 7 := by
  sorry

end units_digit_of_m_squared_plus_two_to_m_l3210_321066


namespace division_equality_l3210_321071

theorem division_equality : (124 : ℚ) / (8 + 14 * 3) = 62 / 25 := by
  sorry

end division_equality_l3210_321071


namespace cubic_equation_solution_l3210_321039

theorem cubic_equation_solution (x : ℝ) : 
  x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ↔ x = 6 := by sorry

end cubic_equation_solution_l3210_321039


namespace stratified_sampling_third_year_l3210_321080

theorem stratified_sampling_third_year 
  (total_students : ℕ) 
  (third_year_students : ℕ) 
  (total_sample : ℕ) 
  (h1 : total_students = 1200)
  (h2 : third_year_students = 300)
  (h3 : total_sample = 100) :
  (third_year_students : ℚ) / total_students * total_sample = 25 := by
  sorry

end stratified_sampling_third_year_l3210_321080


namespace apples_ordered_per_month_l3210_321087

def chandler_initial : ℕ := 23
def lucy_initial : ℕ := 19
def ross_initial : ℕ := 15
def chandler_increase : ℕ := 2
def lucy_decrease : ℕ := 1
def weeks_per_month : ℕ := 4

def total_apples_month : ℕ :=
  (chandler_initial + (chandler_initial + chandler_increase) + 
   (chandler_initial + 2 * chandler_increase) + 
   (chandler_initial + 3 * chandler_increase)) +
  (lucy_initial + (lucy_initial - lucy_decrease) + 
   (lucy_initial - 2 * lucy_decrease) + 
   (lucy_initial - 3 * lucy_decrease)) +
  (ross_initial * weeks_per_month)

theorem apples_ordered_per_month : 
  total_apples_month = 234 := by sorry

end apples_ordered_per_month_l3210_321087


namespace third_month_sale_l3210_321043

/-- Calculates the missing sale amount given the average sale and other known sales. -/
def calculate_missing_sale (average : ℕ) (num_months : ℕ) (known_sales : List ℕ) : ℕ :=
  average * num_months - known_sales.sum

/-- The problem statement -/
theorem third_month_sale (average : ℕ) (num_months : ℕ) (known_sales : List ℕ) :
  average = 5600 ∧ 
  num_months = 6 ∧ 
  known_sales = [5266, 5768, 5678, 6029, 4937] →
  calculate_missing_sale average num_months known_sales = 5922 := by
  sorry

end third_month_sale_l3210_321043


namespace sum_of_products_l3210_321044

-- Define the problem statement
theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 2)
  (eq2 : y^2 + y*z + z^2 = 5)
  (eq3 : z^2 + x*z + x^2 = 3) :
  x*y + y*z + x*z = 2 * Real.sqrt 2 := by
sorry

end sum_of_products_l3210_321044


namespace quadratic_completion_l3210_321012

theorem quadratic_completion (x : ℝ) :
  ∃ (d e : ℝ), x^2 - 24*x + 45 = (x + d)^2 + e ∧ d + e = -111 := by
  sorry

end quadratic_completion_l3210_321012


namespace circle_tangent_to_lines_l3210_321056

/-- The circle with center (1, 1) and radius √5 is tangent to both lines 2x - y + 4 = 0 and 2x - y - 6 = 0 -/
theorem circle_tangent_to_lines :
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + (y - 1)^2 = 5}
  let line1 := {(x, y) : ℝ × ℝ | 2*x - y + 4 = 0}
  let line2 := {(x, y) : ℝ × ℝ | 2*x - y - 6 = 0}
  (∃ p ∈ circle ∩ line1, ∀ q ∈ circle, q ∉ line1 ∨ q = p) ∧
  (∃ p ∈ circle ∩ line2, ∀ q ∈ circle, q ∉ line2 ∨ q = p) :=
by
  sorry

end circle_tangent_to_lines_l3210_321056


namespace initial_oranges_count_l3210_321037

/-- The number of apples in the box -/
def num_apples : ℕ := 14

/-- The number of oranges to be removed -/
def oranges_removed : ℕ := 6

/-- The percentage of apples after removing oranges -/
def apple_percentage : ℚ := 70 / 100

theorem initial_oranges_count : 
  ∃ (initial_oranges : ℕ), 
    (num_apples : ℚ) / ((num_apples : ℚ) + (initial_oranges - oranges_removed : ℚ)) = apple_percentage ∧ 
    initial_oranges = 12 := by
  sorry

end initial_oranges_count_l3210_321037


namespace average_shirts_sold_per_day_l3210_321027

theorem average_shirts_sold_per_day 
  (morning_day1 : ℕ) 
  (afternoon_day1 : ℕ) 
  (day2 : ℕ) 
  (h1 : morning_day1 = 250) 
  (h2 : afternoon_day1 = 20) 
  (h3 : day2 = 320) : 
  (morning_day1 + afternoon_day1 + day2) / 2 = 295 := by
sorry

end average_shirts_sold_per_day_l3210_321027


namespace maximum_marks_correct_l3210_321042

/-- The maximum marks in an exam where:
    1. The passing threshold is 33% of the maximum marks.
    2. A student got 92 marks.
    3. The student failed by 40 marks (i.e., needed 40 more marks to pass). -/
def maximum_marks : ℕ := 400

/-- The passing threshold as a fraction of the maximum marks -/
def passing_threshold : ℚ := 33 / 100

/-- The marks obtained by the student -/
def obtained_marks : ℕ := 92

/-- The additional marks needed to pass -/
def additional_marks_needed : ℕ := 40

theorem maximum_marks_correct :
  maximum_marks * (passing_threshold : ℚ) = obtained_marks + additional_marks_needed := by
  sorry

end maximum_marks_correct_l3210_321042


namespace unchanged_temperature_count_is_219_l3210_321061

/-- The count of integer Fahrenheit temperatures between 32 and 2000 (inclusive) 
    that remain unchanged after the specified conversion process -/
def unchangedTemperatureCount : ℕ :=
  let minTemp := 32
  let maxTemp := 2000
  (maxTemp - minTemp) / 9 + 1

theorem unchanged_temperature_count_is_219 : 
  unchangedTemperatureCount = 219 := by
  sorry

end unchanged_temperature_count_is_219_l3210_321061


namespace product_469158_9999_l3210_321046

theorem product_469158_9999 : 469158 * 9999 = 4690872842 := by
  sorry

end product_469158_9999_l3210_321046


namespace pyramid_theorem_l3210_321032

/-- A regular triangular pyramid with an inscribed sphere -/
structure RegularPyramidWithSphere where
  /-- The side length of the base triangle -/
  base_side : ℝ
  /-- The radius of the inscribed sphere -/
  sphere_radius : ℝ
  /-- The sphere is inscribed at the midpoint of the pyramid's height -/
  sphere_at_midpoint : True
  /-- The sphere touches the lateral faces of the pyramid -/
  sphere_touches_faces : True
  /-- A hemisphere supported by the inscribed circle in the base touches the sphere externally -/
  hemisphere_touches_sphere : True

/-- Properties of the regular triangular pyramid with inscribed sphere -/
def pyramid_properties (p : RegularPyramidWithSphere) : Prop :=
  p.sphere_radius = 1 ∧
  p.base_side = 2 * Real.sqrt 3 * (Real.sqrt 5 + 1)

/-- The lateral surface area of the pyramid -/
noncomputable def lateral_surface_area (p : RegularPyramidWithSphere) : ℝ :=
  3 * Real.sqrt 15 * (Real.sqrt 5 + 1)

/-- The angle between lateral faces of the pyramid -/
noncomputable def lateral_face_angle (p : RegularPyramidWithSphere) : ℝ :=
  Real.arccos (1 / Real.sqrt 5)

/-- Theorem stating the properties of the pyramid -/
theorem pyramid_theorem (p : RegularPyramidWithSphere) 
  (h : pyramid_properties p) :
  lateral_surface_area p = 3 * Real.sqrt 15 * (Real.sqrt 5 + 1) ∧
  lateral_face_angle p = Real.arccos (1 / Real.sqrt 5) := by
  sorry

end pyramid_theorem_l3210_321032


namespace volleyball_lineup_combinations_l3210_321078

def volleyball_team_size : ℕ := 10
def lineup_size : ℕ := 5

theorem volleyball_lineup_combinations :
  (volleyball_team_size.factorial) / ((volleyball_team_size - lineup_size).factorial) = 30240 := by
  sorry

end volleyball_lineup_combinations_l3210_321078


namespace valid_a_values_l3210_321034

def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2 - a + 1}

theorem valid_a_values :
  ∀ a : ℝ, (A a ⊇ B a) → (a = -1 ∨ a = 2) :=
by sorry

end valid_a_values_l3210_321034


namespace closest_to_zero_l3210_321002

def integers : List Int := [-1101, 1011, -1010, -1001, 1110]

theorem closest_to_zero (n : Int) (h : n ∈ integers) : 
  ∀ m ∈ integers, |n| ≤ |m| ↔ n = -1001 :=
by
  sorry

#check closest_to_zero

end closest_to_zero_l3210_321002


namespace twentieth_base4_is_110_l3210_321075

/-- Converts a decimal number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The 20th number in the base-4 system -/
def twentieth_base4 : List ℕ := toBase4 20

theorem twentieth_base4_is_110 : twentieth_base4 = [1, 1, 0] := by
  sorry

end twentieth_base4_is_110_l3210_321075


namespace original_class_size_l3210_321069

theorem original_class_size (original_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) : 
  original_avg = 50 →
  new_students = 12 →
  new_avg = 32 →
  avg_decrease = 4 →
  ∃ (original_size : ℕ), 
    (original_size : ℝ) * original_avg + (new_students : ℝ) * new_avg = 
    (original_size + new_students : ℝ) * (original_avg - avg_decrease) ∧
    original_size = 42 :=
by sorry


end original_class_size_l3210_321069


namespace integer_root_of_cubic_l3210_321082

theorem integer_root_of_cubic (b c : ℚ) :
  (∃ x : ℤ, x^3 + b*x + c = 0) →
  (Complex.exp (3 - Real.sqrt 3))^3 + b*(Complex.exp (3 - Real.sqrt 3)) + c = 0 →
  (∃ x : ℤ, x^3 + b*x + c = 0 ∧ x = -6) :=
by sorry

end integer_root_of_cubic_l3210_321082


namespace triangle_area_l3210_321081

theorem triangle_area (a b c : ℝ) (h1 : a = 14) (h2 : b = 48) (h3 : c = 50) :
  (1/2) * a * b = 336 := by
  sorry

end triangle_area_l3210_321081


namespace no_positive_integer_solutions_l3210_321015

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (x^2 * y^2)^2 - 14 * (x^2 * y^2) + 49 = 0 :=
by sorry

end no_positive_integer_solutions_l3210_321015


namespace sum_of_fractions_l3210_321017

theorem sum_of_fractions : 
  (251 : ℚ) / (2008 * 2009) + (251 : ℚ) / (2009 * 2010) = -1 / 8040 := by
  sorry

end sum_of_fractions_l3210_321017


namespace zeroPointThreeBarSix_eq_elevenThirties_l3210_321026

/-- Represents a repeating decimal with a non-repeating part and a repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℚ
  repeating : ℚ
  repeatingLessThanOne : repeating < 1

/-- The value of a repeating decimal as a rational number -/
def RepeatingDecimal.toRational (d : RepeatingDecimal) : ℚ :=
  d.nonRepeating + d.repeating / (1 - (1/10)^(d.repeating.den))

/-- 0.3̄6 as a RepeatingDecimal -/
def zeroPointThreeBarSix : RepeatingDecimal :=
  { nonRepeating := 3/10
    repeating := 6/10
    repeatingLessThanOne := by sorry }

theorem zeroPointThreeBarSix_eq_elevenThirties : 
  zeroPointThreeBarSix.toRational = 11/30 := by sorry

end zeroPointThreeBarSix_eq_elevenThirties_l3210_321026


namespace inverse_variation_proof_l3210_321028

/-- Given that y^4 varies inversely with z^2 and y = 3 when z = 1, prove that y = √3 when z = 3 -/
theorem inverse_variation_proof (y z : ℝ) (h1 : ∃ k : ℝ, ∀ y z, y^4 * z^2 = k) 
  (h2 : ∃ y₀ z₀, y₀ = 3 ∧ z₀ = 1 ∧ y₀^4 * z₀^2 = (3 : ℝ)^4 * 1^2) :
  ∃ y₁, y₁^4 * 3^2 = 3^4 * 1^2 ∧ y₁ = Real.sqrt 3 := by
  sorry

end inverse_variation_proof_l3210_321028


namespace refill_count_is_three_l3210_321090

/-- Calculates the number of daily water bottle refills given the parameters. -/
def daily_refills (bottle_capacity : ℕ) (days : ℕ) (spill1 : ℕ) (spill2 : ℕ) (total_drunk : ℕ) : ℕ :=
  ((total_drunk + spill1 + spill2) / (bottle_capacity * days) : ℕ)

/-- Proves that given the specified parameters, the number of daily refills is 3. -/
theorem refill_count_is_three :
  daily_refills 20 7 5 8 407 = 3 := by
  sorry

end refill_count_is_three_l3210_321090


namespace asymptote_of_hyperbola_l3210_321079

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 25 - y^2 / 16 = 1

/-- The equation of an asymptote -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = (4/5) * x

/-- Theorem: The given equation is an asymptote of the hyperbola -/
theorem asymptote_of_hyperbola :
  ∀ x y : ℝ, asymptote_equation x y → (∃ ε > 0, ∀ δ > ε, 
    ∃ x' y' : ℝ, hyperbola_equation x' y' ∧ 
    ((x' - x)^2 + (y' - y)^2 < δ^2)) :=
sorry

end asymptote_of_hyperbola_l3210_321079


namespace isosceles_triangle_l3210_321047

theorem isosceles_triangle (A B C : Real) (h : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) : A = B := by
  sorry

end isosceles_triangle_l3210_321047


namespace problem_statement_l3210_321077

theorem problem_statement (a b : ℤ) (h1 : a = -5) (h2 : b = 3) : 
  -a - b^4 + a*b = -91 := by sorry

end problem_statement_l3210_321077


namespace f_analytical_expression_k_range_for_monotonicity_l3210_321059

-- Part 1
def f₁ (x : ℝ) := x^2 - 3*x + 2

theorem f_analytical_expression :
  ∀ x, f₁ (x + 1) = x^2 - 3*x + 2 →
  ∃ g : ℝ → ℝ, (∀ x, g x = x^2 - 6*x + 6) ∧ (∀ x, g x = f₁ x) :=
sorry

-- Part 2
def f₂ (k : ℝ) (x : ℝ) := x^2 - 2*k*x - 8

theorem k_range_for_monotonicity :
  ∀ k, (∀ x ∈ Set.Icc 1 4, Monotone (f₂ k)) →
  k ≥ 4 ∨ k ≤ 1 :=
sorry

end f_analytical_expression_k_range_for_monotonicity_l3210_321059


namespace no_odd_multiples_of_18_24_36_between_1500_3000_l3210_321004

theorem no_odd_multiples_of_18_24_36_between_1500_3000 :
  ∀ n : ℕ, 1500 < n ∧ n < 3000 ∧ n % 2 = 1 →
    ¬(18 ∣ n ∧ 24 ∣ n ∧ 36 ∣ n) :=
by sorry

end no_odd_multiples_of_18_24_36_between_1500_3000_l3210_321004


namespace flour_measurement_l3210_321088

theorem flour_measurement (flour_needed : ℚ) (cup_capacity : ℚ) : 
  flour_needed = 4 + 3 / 4 →
  cup_capacity = 1 / 2 →
  ⌈flour_needed / cup_capacity⌉ = 10 := by
  sorry

end flour_measurement_l3210_321088


namespace hausdorff_dim_countable_union_l3210_321096

open MeasureTheory

-- Define a countable collection of sets
variable {α : Type*} [MeasurableSpace α]
variable (A : ℕ → Set α)

-- Define Hausdorff dimension
noncomputable def hausdorffDim (S : Set α) : ℝ := sorry

-- State the theorem
theorem hausdorff_dim_countable_union :
  hausdorffDim (⋃ i, A i) = ⨆ i, hausdorffDim (A i) := by sorry

end hausdorff_dim_countable_union_l3210_321096
