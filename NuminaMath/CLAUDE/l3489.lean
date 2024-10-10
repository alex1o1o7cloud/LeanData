import Mathlib

namespace complex_product_given_pure_imaginary_sum_l3489_348983

theorem complex_product_given_pure_imaginary_sum (a : ℝ) : 
  let z₁ : ℂ := a - 2*I
  let z₂ : ℂ := -1 + a*I
  (∃ (b : ℝ), z₁ + z₂ = b*I) → z₁ * z₂ = 1 + 3*I :=
by sorry

end complex_product_given_pure_imaginary_sum_l3489_348983


namespace train_speed_difference_l3489_348991

theorem train_speed_difference (v : ℝ) 
  (cattle_speed : ℝ) (head_start : ℝ) (diesel_time : ℝ) (total_distance : ℝ)
  (h1 : v < cattle_speed)
  (h2 : cattle_speed = 56)
  (h3 : head_start = 6)
  (h4 : diesel_time = 12)
  (h5 : total_distance = 1284)
  (h6 : cattle_speed * head_start + cattle_speed * diesel_time + v * diesel_time = total_distance) :
  cattle_speed - v = 33 := by
sorry

end train_speed_difference_l3489_348991


namespace five_dice_probability_l3489_348996

/-- A die is represented as a number from 1 to 6 -/
def Die := Fin 6

/-- A roll of five dice -/
def FiveDiceRoll := Fin 5 → Die

/-- The probability space of rolling five fair six-sided dice -/
def Ω : Type := FiveDiceRoll

/-- The probability measure on Ω -/
noncomputable def P : Set Ω → ℝ := sorry

/-- The event that at least three dice show the same value -/
def AtLeastThreeSame (roll : Ω) : Prop := sorry

/-- The sum of the values shown on all dice -/
def DiceSum (roll : Ω) : ℕ := sorry

/-- The event that the sum of all dice is greater than 20 -/
def SumGreaterThan20 (roll : Ω) : Prop := DiceSum roll > 20

/-- The main theorem to be proved -/
theorem five_dice_probability : 
  P {roll : Ω | AtLeastThreeSame roll ∧ SumGreaterThan20 roll} = 31 / 432 := by sorry

end five_dice_probability_l3489_348996


namespace integral_sqrt_4_minus_x_squared_l3489_348990

theorem integral_sqrt_4_minus_x_squared : ∫ x in (-2)..2, Real.sqrt (4 - x^2) = 2 * Real.pi := by sorry

end integral_sqrt_4_minus_x_squared_l3489_348990


namespace september_births_percentage_l3489_348903

theorem september_births_percentage 
  (total_people : ℕ) 
  (september_births : ℕ) 
  (h1 : total_people = 120) 
  (h2 : september_births = 12) : 
  (september_births : ℚ) / total_people * 100 = 10 := by
sorry

end september_births_percentage_l3489_348903


namespace runner_speed_ratio_l3489_348944

theorem runner_speed_ratio (u₁ u₂ : ℝ) (h₁ : u₁ > u₂) (h₂ : u₁ + u₂ = 5) (h₃ : u₁ - u₂ = 5 / 3) :
  u₁ / u₂ = 2 := by
sorry

end runner_speed_ratio_l3489_348944


namespace clock_hands_opposite_period_l3489_348928

/-- The number of times clock hands are in opposite directions in 12 hours -/
def opposite_directions_per_12_hours : ℕ := 11

/-- The number of hours on a clock -/
def hours_on_clock : ℕ := 12

/-- The number of minutes between opposite directions -/
def minutes_between_opposite : ℕ := 30

/-- The observed number of times the hands are in opposite directions -/
def observed_opposite_directions : ℕ := 22

/-- The period in which the hands show opposite directions 22 times -/
def period : ℕ := 24

theorem clock_hands_opposite_period :
  opposite_directions_per_12_hours * 2 = observed_opposite_directions →
  period = 24 := by sorry

end clock_hands_opposite_period_l3489_348928


namespace fraction_simplification_l3489_348907

theorem fraction_simplification (x : ℝ) (h : x = 3) : 
  (x^8 + 18*x^4 + 81) / (x^4 + 9) = 90 := by
  sorry

end fraction_simplification_l3489_348907


namespace t_shaped_area_concrete_t_shaped_area_l3489_348974

/-- The area of a T-shaped region formed by subtracting three smaller rectangles from a larger rectangle -/
theorem t_shaped_area (a b c d e f : ℕ) : 
  a * b - (c * d + e * f + c * (b - f)) = 24 :=
by
  sorry

/-- Concrete instance of the T-shaped area theorem -/
theorem concrete_t_shaped_area : 
  8 * 6 - (2 * 2 + 4 * 2 + 2 * 6) = 24 :=
by
  sorry

end t_shaped_area_concrete_t_shaped_area_l3489_348974


namespace system_solution_l3489_348962

theorem system_solution (x y z u : ℚ) : 
  x + y = 12 ∧ 
  x / z = 3 / 2 ∧ 
  z + u = 10 ∧ 
  y * u = 36 →
  x = 6 ∧ y = 6 ∧ z = 4 ∧ u = 6 := by
sorry

end system_solution_l3489_348962


namespace kg_to_lb_conversion_rate_l3489_348985

/-- Conversion rate from kilograms to pounds -/
def kg_to_lb_rate : ℝ := 2.2

/-- Initial weight in kilograms -/
def initial_weight_kg : ℝ := 80

/-- Weight loss in pounds per hour of exercise -/
def weight_loss_per_hour : ℝ := 1.5

/-- Hours of exercise per day -/
def exercise_hours_per_day : ℝ := 2

/-- Number of days of exercise -/
def exercise_days : ℝ := 14

/-- Final weight in pounds after exercise period -/
def final_weight_lb : ℝ := 134

theorem kg_to_lb_conversion_rate :
  kg_to_lb_rate * initial_weight_kg =
    final_weight_lb + weight_loss_per_hour * exercise_hours_per_day * exercise_days :=
by sorry

end kg_to_lb_conversion_rate_l3489_348985


namespace h_3_value_l3489_348953

-- Define the functions
def f (x : ℝ) : ℝ := 2 * x + 9
def g (x : ℝ) : ℝ := (f x) ^ (1/3) - 3
def h (x : ℝ) : ℝ := f (g x)

-- State the theorem
theorem h_3_value : h 3 = 2 * 15^(1/3) + 3 := by sorry

end h_3_value_l3489_348953


namespace square_sum_factorial_solutions_l3489_348992

theorem square_sum_factorial_solutions :
  ∀ (a b n : ℕ+),
    n < 14 →
    a ≤ b →
    a ^ 2 + b ^ 2 = n! →
    ((n = 2 ∧ a = 1 ∧ b = 1) ∨ (n = 6 ∧ a = 12 ∧ b = 24)) :=
by sorry

end square_sum_factorial_solutions_l3489_348992


namespace rectangle_area_l3489_348984

/-- The area of a rectangle with length 2x and width 2x-1 is 4x^2 - 2x -/
theorem rectangle_area (x : ℝ) : 
  let length : ℝ := 2 * x
  let width : ℝ := 2 * x - 1
  length * width = 4 * x^2 - 2 * x := by
sorry

end rectangle_area_l3489_348984


namespace two_questions_suffice_l3489_348969

-- Define the possible types of siblings
inductive SiblingType
  | Truthful
  | Unpredictable

-- Define a sibling
structure Sibling :=
  (type : SiblingType)

-- Define the farm setup
structure Farm :=
  (siblings : Fin 3 → Sibling)
  (correct_path : Nat)

-- Define the possible answers to a question
inductive Answer
  | Yes
  | No

-- Define a question as a function from a sibling to an answer
def Question := Sibling → Answer

-- Define the theorem
theorem two_questions_suffice (farm : Farm) :
  ∃ (q1 q2 : Question), ∀ (i j : Fin 3),
    (farm.siblings i).type = SiblingType.Truthful →
    (farm.siblings j).type = SiblingType.Truthful →
    i ≠ j →
    ∃ (f : Answer → Answer → Nat),
      f (q1 (farm.siblings i)) (q2 (farm.siblings j)) = farm.correct_path :=
sorry


end two_questions_suffice_l3489_348969


namespace divisibility_by_six_divisibility_by_120_divisibility_by_48_divisibility_by_1152_not_always_divisible_by_720_l3489_348932

-- Part (a)
theorem divisibility_by_six (a b c : ℤ) (h : 6 ∣ (a + b + c)) : 6 ∣ (a^5 + b^3 + c) := by sorry

-- Part (b)
theorem divisibility_by_120 (n : ℤ) : 120 ∣ (n^5 - 5*n^3 + 4*n) := by sorry

-- Part (c)
theorem divisibility_by_48 (n : ℤ) (h : Odd n) : 48 ∣ (n^3 + 3*n^2 - n - 3) := by sorry

-- Part (d)
theorem divisibility_by_1152 (n : ℤ) (h : Odd n) : 1152 ∣ (n^8 - n^6 - n^4 + n^2) := by sorry

-- Part (e)
theorem not_always_divisible_by_720 : ∃ n : ℤ, ¬(720 ∣ (n*(n^2 - 1)*(n^2 - 4))) := by sorry

end divisibility_by_six_divisibility_by_120_divisibility_by_48_divisibility_by_1152_not_always_divisible_by_720_l3489_348932


namespace integer_pairs_satisfying_equation_l3489_348917

theorem integer_pairs_satisfying_equation : 
  {(x, y) : ℤ × ℤ | x^2 + y^2 = x + y + 2} = 
  {(-1, 0), (-1, 1), (0, -1), (0, 2), (1, -1), (1, 2), (2, 0), (2, 1)} := by
  sorry

end integer_pairs_satisfying_equation_l3489_348917


namespace parabola_intersection_length_l3489_348967

/-- Parabola type representing y^2 = 4x --/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line type --/
structure Line where
  passes_through : ℝ × ℝ → Prop

/-- Represents a point on the parabola --/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

theorem parabola_intersection_length 
  (p : Parabola) 
  (l : Line) 
  (A B : ParabolaPoint) 
  (h1 : p.equation = fun x y => y^2 = 4*x)
  (h2 : p.focus = (1, 0))
  (h3 : l.passes_through p.focus)
  (h4 : p.equation A.x A.y)
  (h5 : p.equation B.x B.y)
  (h6 : l.passes_through (A.x, A.y))
  (h7 : l.passes_through (B.x, B.y))
  (h8 : (A.x + B.x) / 2 = 3)
  : Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 8 := by
  sorry

end parabola_intersection_length_l3489_348967


namespace range_of_b_l3489_348941

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = Real.sqrt (9 - p.1^2)}
def N (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + b}

-- State the theorem
theorem range_of_b (b : ℝ) : M ∩ N b = ∅ ↔ b > 3 * Real.sqrt 2 ∨ b < -3 * Real.sqrt 2 := by
  sorry

end range_of_b_l3489_348941


namespace star_transformation_l3489_348940

theorem star_transformation (a b c d : ℕ) :
  a ∈ Finset.range 17 → b ∈ Finset.range 17 → c ∈ Finset.range 17 → d ∈ Finset.range 17 →
  a + b + c + d = 34 →
  (17 - a) + (17 - b) + (17 - c) + (17 - d) = 34 := by
sorry

end star_transformation_l3489_348940


namespace adult_average_age_l3489_348976

theorem adult_average_age
  (total_members : ℕ)
  (total_average_age : ℚ)
  (num_girls : ℕ)
  (num_boys : ℕ)
  (num_adults : ℕ)
  (girls_average_age : ℚ)
  (boys_average_age : ℚ)
  (h1 : total_members = 50)
  (h2 : total_average_age = 18)
  (h3 : num_girls = 25)
  (h4 : num_boys = 20)
  (h5 : num_adults = 5)
  (h6 : girls_average_age = 16)
  (h7 : boys_average_age = 17)
  (h8 : total_members = num_girls + num_boys + num_adults) :
  (total_members * total_average_age - num_girls * girls_average_age - num_boys * boys_average_age) / num_adults = 32 := by
  sorry

end adult_average_age_l3489_348976


namespace debate_team_girls_l3489_348925

theorem debate_team_girls (boys : ℕ) (groups : ℕ) (group_size : ℕ) (girls : ℕ) : 
  boys = 26 → 
  groups = 8 → 
  group_size = 9 → 
  groups * group_size = boys + girls → 
  girls = 46 :=
by
  sorry

end debate_team_girls_l3489_348925


namespace truck_travel_distance_l3489_348978

/-- Given a truck that travels 300 miles on 10 gallons of gas, 
    prove that it will travel 450 miles on 15 gallons of gas, 
    assuming a constant rate of fuel consumption. -/
theorem truck_travel_distance (initial_distance : ℝ) (initial_gas : ℝ) (new_gas : ℝ) : 
  initial_distance = 300 ∧ initial_gas = 10 ∧ new_gas = 15 →
  (new_gas * initial_distance) / initial_gas = 450 := by
  sorry

end truck_travel_distance_l3489_348978


namespace parallel_vectors_imply_x_equals_four_l3489_348951

/-- Given two vectors a and b in ℝ², where a = (2,1) and b = (x,2),
    if a + b is parallel to a - 2b, then x = 4 -/
theorem parallel_vectors_imply_x_equals_four (x : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, 2)
  (∃ (k : ℝ), k ≠ 0 ∧ (a.1 + b.1, a.2 + b.2) = k • (a.1 - 2*b.1, a.2 - 2*b.2)) →
  x = 4 :=
by sorry

end parallel_vectors_imply_x_equals_four_l3489_348951


namespace square_area_on_parabola_l3489_348924

/-- The area of a square with one side on y = 7 and endpoints on y = x^2 + 2x + 1 is 28 -/
theorem square_area_on_parabola : 
  ∃ (x₁ x₂ : ℝ),
    (x₁^2 + 2*x₁ + 1 = 7) ∧ 
    (x₂^2 + 2*x₂ + 1 = 7) ∧ 
    ((x₂ - x₁)^2 = 28) := by
  sorry

end square_area_on_parabola_l3489_348924


namespace cole_gum_count_l3489_348904

/-- The number of people sharing the gum -/
def num_people : ℕ := 3

/-- The number of pieces of gum John has -/
def john_gum : ℕ := 54

/-- The number of pieces of gum Aubrey has -/
def aubrey_gum : ℕ := 0

/-- The number of pieces each person gets after sharing -/
def shared_gum : ℕ := 33

/-- Cole's initial number of pieces of gum -/
def cole_gum : ℕ := num_people * shared_gum - john_gum - aubrey_gum

theorem cole_gum_count : cole_gum = 45 := by
  sorry

end cole_gum_count_l3489_348904


namespace work_completion_time_l3489_348910

/-- Given that two workers A and B can complete a work together in a certain number of days,
    and worker A can complete the work alone in a certain number of days,
    this function calculates the number of days worker B would take to complete the work alone. -/
def days_for_B (days_together days_A_alone : ℚ) : ℚ :=
  1 / (1 / days_together - 1 / days_A_alone)

/-- Theorem stating that if A and B can complete a work in 12 days, and A alone can complete
    the work in 20 days, then B alone will complete the work in 30 days. -/
theorem work_completion_time :
  days_for_B 12 20 = 30 := by
  sorry

end work_completion_time_l3489_348910


namespace contrapositive_equivalence_l3489_348947

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 + x - 6 > 0 ↔ (x < -3 ∨ x > 2)) ↔
  (∀ x : ℝ, (x ≥ -3 ∧ x ≤ 2) → x^2 + x - 6 ≤ 0) :=
sorry

end contrapositive_equivalence_l3489_348947


namespace equation_solution_pairs_l3489_348979

theorem equation_solution_pairs : 
  ∀ x y : ℕ+, x^(y : ℕ) - y^(x : ℕ) = 1 ↔ (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) := by
sorry

end equation_solution_pairs_l3489_348979


namespace min_xy_value_l3489_348914

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 8/y = 1) :
  xy ≥ 64 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2/x + 8/y = 1 ∧ x*y = 64 :=
sorry

end min_xy_value_l3489_348914


namespace bread_left_l3489_348968

theorem bread_left (total : ℕ) (bomi_ate : ℕ) (yejun_ate : ℕ) 
  (h1 : total = 1000)
  (h2 : bomi_ate = 350)
  (h3 : yejun_ate = 500) :
  total - (bomi_ate + yejun_ate) = 150 := by
  sorry

end bread_left_l3489_348968


namespace daisy_solution_l3489_348927

def daisy_problem (day1 day2 day3 day4 total : ℕ) : Prop :=
  day1 = 45 ∧
  day2 = day1 + 20 ∧
  day3 = 2 * day2 - 10 ∧
  day1 + day2 + day3 + day4 = total ∧
  total = 350

theorem daisy_solution :
  ∃ day1 day2 day3 day4 total, daisy_problem day1 day2 day3 day4 total ∧ day4 = 120 := by
  sorry

end daisy_solution_l3489_348927


namespace units_digit_of_j_squared_plus_three_to_j_l3489_348998

theorem units_digit_of_j_squared_plus_three_to_j (j : ℕ) : 
  j = 2023^3 + 3^2023 + 2023 → (j^2 + 3^j) % 10 = 6 := by
sorry

end units_digit_of_j_squared_plus_three_to_j_l3489_348998


namespace f_sum_positive_l3489_348912

def f (x : ℝ) : ℝ := x^2015

theorem f_sum_positive (a b : ℝ) (h : a + b > 0) : f a + f b > 0 := by
  sorry

end f_sum_positive_l3489_348912


namespace trig_function_equality_l3489_348958

/-- Given two functions f and g defined on real numbers, prove that g(x) equals f(π/4 + x) for all real x. -/
theorem trig_function_equality (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = Real.sin (2 * x + π / 3))
  (hg : ∀ x, g x = Real.cos (2 * x + π / 3)) :
  ∀ x, g x = f (π / 4 + x) := by
  sorry

end trig_function_equality_l3489_348958


namespace marlon_lollipops_l3489_348919

theorem marlon_lollipops (initial_lollipops : ℕ) (kept_lollipops : ℕ) (lou_lollipops : ℕ) :
  initial_lollipops = 42 →
  kept_lollipops = 4 →
  lou_lollipops = 10 →
  (initial_lollipops - kept_lollipops - lou_lollipops : ℚ) / initial_lollipops = 2/3 :=
by sorry

end marlon_lollipops_l3489_348919


namespace average_listening_time_is_55_minutes_l3489_348911

/-- Represents the distribution of audience members and their listening times --/
structure AudienceDistribution where
  total_audience : ℕ
  lecture_duration : ℕ
  full_listeners_percent : ℚ
  sleepers_percent : ℚ
  quarter_listeners_percent : ℚ
  half_listeners_percent : ℚ
  three_quarter_listeners_percent : ℚ

/-- Calculates the average listening time for the given audience distribution --/
def average_listening_time (dist : AudienceDistribution) : ℚ :=
  sorry

/-- The theorem to be proved --/
theorem average_listening_time_is_55_minutes 
  (dist : AudienceDistribution)
  (h1 : dist.total_audience = 200)
  (h2 : dist.lecture_duration = 90)
  (h3 : dist.full_listeners_percent = 30 / 100)
  (h4 : dist.sleepers_percent = 15 / 100)
  (h5 : dist.quarter_listeners_percent = (1 - dist.full_listeners_percent - dist.sleepers_percent) / 4)
  (h6 : dist.half_listeners_percent = (1 - dist.full_listeners_percent - dist.sleepers_percent) / 4)
  (h7 : dist.three_quarter_listeners_percent = 1 - dist.full_listeners_percent - dist.sleepers_percent - dist.quarter_listeners_percent - dist.half_listeners_percent)
  : average_listening_time dist = 55 := by
  sorry

end average_listening_time_is_55_minutes_l3489_348911


namespace smallest_number_with_remainder_two_l3489_348988

theorem smallest_number_with_remainder_two : ∃! n : ℕ,
  n > 1 ∧
  (∀ d ∈ ({3, 4, 5, 6, 7} : Set ℕ), n % d = 2) ∧
  (∀ m : ℕ, m > 1 ∧ (∀ d ∈ ({3, 4, 5, 6, 7} : Set ℕ), m % d = 2) → m ≥ n) ∧
  n = 422 :=
by sorry

end smallest_number_with_remainder_two_l3489_348988


namespace hyperbola_eccentricity_l3489_348975

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) 
    and asymptote equations y = ±x, its eccentricity is √2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : ∀ (x y : ℝ), (y = x ∨ y = -x) → (x^2 / a^2 - y^2 / b^2 = 1)) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 2 := by sorry

end hyperbola_eccentricity_l3489_348975


namespace two_different_color_balls_probability_two_different_color_balls_probability_proof_l3489_348963

theorem two_different_color_balls_probability 
  (total_balls : ℕ) 
  (red_balls yellow_balls white_balls : ℕ) 
  (h1 : total_balls = red_balls + yellow_balls + white_balls)
  (h2 : red_balls = 2)
  (h3 : yellow_balls = 2)
  (h4 : white_balls = 1)
  : ℚ :=
4/5

theorem two_different_color_balls_probability_proof 
  (total_balls : ℕ) 
  (red_balls yellow_balls white_balls : ℕ) 
  (h1 : total_balls = red_balls + yellow_balls + white_balls)
  (h2 : red_balls = 2)
  (h3 : yellow_balls = 2)
  (h4 : white_balls = 1)
  : two_different_color_balls_probability total_balls red_balls yellow_balls white_balls h1 h2 h3 h4 = 4/5 := by
  sorry

end two_different_color_balls_probability_two_different_color_balls_probability_proof_l3489_348963


namespace droid_coffee_usage_l3489_348936

/-- The number of bags of coffee beans Droid uses in a week -/
def weekly_coffee_usage : ℕ :=
  let morning_usage := 3
  let afternoon_usage := 3 * morning_usage
  let evening_usage := 2 * morning_usage
  let daily_usage := morning_usage + afternoon_usage + evening_usage
  7 * daily_usage

/-- Theorem stating that Droid uses 126 bags of coffee beans in a week -/
theorem droid_coffee_usage : weekly_coffee_usage = 126 := by
  sorry

end droid_coffee_usage_l3489_348936


namespace no_valid_replacements_l3489_348942

theorem no_valid_replacements :
  ∀ z : ℕ, z < 10 → ¬(35000 + 100 * z + 45) % 4 = 0 := by
sorry

end no_valid_replacements_l3489_348942


namespace zeros_in_Q_l3489_348982

def R (k : ℕ) : ℚ := (10^k - 1) / 9

def Q : ℚ := R 25 / R 5

def count_zeros (q : ℚ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 16 := by sorry

end zeros_in_Q_l3489_348982


namespace alphaBetaArrangementsCount_l3489_348956

/-- The number of distinct arrangements of 9 letters, where one letter appears 4 times
    and six other letters appear once each. -/
def alphaBetaArrangements : ℕ :=
  Nat.factorial 9 / (Nat.factorial 4 * (Nat.factorial 1)^6)

/-- Theorem stating that the number of distinct arrangements of letters in "alpha beta"
    under the given conditions is 15120. -/
theorem alphaBetaArrangementsCount : alphaBetaArrangements = 15120 := by
  sorry

end alphaBetaArrangementsCount_l3489_348956


namespace trolleybus_problem_l3489_348921

/-- Trolleybus Problem -/
theorem trolleybus_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  (∀ z : ℝ, z > 0 → y * z = 6 * (y - x) ∧ y * z = 3 * (y + x)) →
  (∃ z : ℝ, z = 4 ∧ x = y / 3) :=
by sorry

end trolleybus_problem_l3489_348921


namespace female_math_only_result_l3489_348997

/-- The number of female students who participated in the math competition but not in the English competition -/
def female_math_only (male_math female_math female_eng male_eng total male_both : ℕ) : ℕ :=
  let male_total := male_math + male_eng - male_both
  let female_total := total - male_total
  let female_both := female_math + female_eng - female_total
  female_math - female_both

/-- Theorem stating the result of the problem -/
theorem female_math_only_result : 
  female_math_only 120 80 120 80 260 75 = 15 := by
  sorry

end female_math_only_result_l3489_348997


namespace expression_evaluation_l3489_348900

theorem expression_evaluation : 
  (3 - 4 * (3 - 5)⁻¹)⁻¹ = (1 : ℚ) / 5 := by sorry

end expression_evaluation_l3489_348900


namespace elevator_occupancy_l3489_348950

/-- Proves that the total number of people in the elevator is 7 after a new person enters --/
theorem elevator_occupancy (initial_people : ℕ) (initial_avg_weight : ℝ) (new_avg_weight : ℝ) :
  initial_people = 6 →
  initial_avg_weight = 160 →
  new_avg_weight = 151 →
  initial_people + 1 = 7 :=
by sorry

end elevator_occupancy_l3489_348950


namespace science_marks_calculation_l3489_348929

def average_marks : ℝ := 75
def num_subjects : ℕ := 5
def math_marks : ℝ := 76
def social_marks : ℝ := 82
def english_marks : ℝ := 67
def biology_marks : ℝ := 85

theorem science_marks_calculation :
  ∃ (science_marks : ℝ),
    (math_marks + social_marks + english_marks + biology_marks + science_marks) / num_subjects = average_marks ∧
    science_marks = 65 := by
  sorry

end science_marks_calculation_l3489_348929


namespace mary_nickels_l3489_348994

/-- The number of nickels Mary has after receiving some from her dad -/
def total_nickels (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Proof that Mary has 12 nickels after receiving 5 from her dad -/
theorem mary_nickels : total_nickels 7 5 = 12 := by
  sorry

end mary_nickels_l3489_348994


namespace lunchroom_students_l3489_348915

/-- The number of students sitting at each table -/
def students_per_table : ℕ := 6

/-- The number of tables in the lunchroom -/
def number_of_tables : ℕ := 34

/-- The total number of students in the lunchroom -/
def total_students : ℕ := students_per_table * number_of_tables

theorem lunchroom_students : total_students = 204 := by
  sorry

end lunchroom_students_l3489_348915


namespace olives_price_per_pound_l3489_348993

/-- Calculates the price per pound of olives given Teresa's shopping list and total spent --/
theorem olives_price_per_pound (sandwich_price : ℝ) (salami_price : ℝ) (olive_weight : ℝ) 
  (feta_weight : ℝ) (feta_price_per_pound : ℝ) (bread_price : ℝ) (total_spent : ℝ) 
  (h1 : sandwich_price = 7.75)
  (h2 : salami_price = 4)
  (h3 : olive_weight = 1/4)
  (h4 : feta_weight = 1/2)
  (h5 : feta_price_per_pound = 8)
  (h6 : bread_price = 2)
  (h7 : total_spent = 40) :
  (total_spent - (2 * sandwich_price + salami_price + 3 * salami_price + 
  feta_weight * feta_price_per_pound + bread_price)) / olive_weight = 10 := by
  sorry

end olives_price_per_pound_l3489_348993


namespace quadratic_inequality_solutions_l3489_348970

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * x + 6 * k

-- Define the solution set types
inductive SolutionSet
  | Interval
  | AllReals
  | Empty

-- State the theorem
theorem quadratic_inequality_solutions (k : ℝ) (h : k ≠ 0) :
  (∀ x, f k x < 0 ↔ x < -3 ∨ x > -2) → k = -2/5 ∧
  (∀ x, f k x < 0) → k < -Real.sqrt 6 / 6 ∧
  (∀ x, f k x ≥ 0) → k ≥ Real.sqrt 6 / 6 := by
  sorry

end quadratic_inequality_solutions_l3489_348970


namespace room_tiling_theorem_l3489_348957

/-- Calculates the number of tiles needed to cover a rectangular room with a border of larger tiles -/
def tilesNeeded (roomLength roomWidth borderTileSize innerTileSize : ℕ) : ℕ :=
  let borderTiles := 2 * (roomLength / borderTileSize + roomWidth / borderTileSize) - 4
  let innerLength := roomLength - 2 * borderTileSize
  let innerWidth := roomWidth - 2 * borderTileSize
  let innerTiles := (innerLength / innerTileSize) * (innerWidth / innerTileSize)
  borderTiles + innerTiles

/-- The theorem stating that 310 tiles are needed for the given room specifications -/
theorem room_tiling_theorem :
  tilesNeeded 24 18 2 1 = 310 := by
  sorry

end room_tiling_theorem_l3489_348957


namespace sin_30_cos_60_plus_cos_30_sin_60_l3489_348971

theorem sin_30_cos_60_plus_cos_30_sin_60 : 
  Real.sin (30 * π / 180) * Real.cos (60 * π / 180) + 
  Real.cos (30 * π / 180) * Real.sin (60 * π / 180) = 1 := by
  sorry

end sin_30_cos_60_plus_cos_30_sin_60_l3489_348971


namespace find_B_l3489_348960

theorem find_B : ∃ B : ℕ, 
  (632 - 591 = 41) ∧ 
  (∃ (AB1 : ℕ), AB1 = 500 + 90 + B ∧ AB1 < 1000) → 
  B = 9 := by
  sorry

end find_B_l3489_348960


namespace perpendicular_line_equation_l3489_348909

/-- A line passing through the point (-1, 2) and perpendicular to 2x - 3y + 4 = 0 has the equation 3x + 2y - 1 = 0 -/
theorem perpendicular_line_equation :
  let point : ℝ × ℝ := (-1, 2)
  let given_line (x y : ℝ) := 2 * x - 3 * y + 4 = 0
  let perpendicular_line (x y : ℝ) := 3 * x + 2 * y - 1 = 0
  (∀ x y : ℝ, perpendicular_line x y ↔ 
    (x = point.1 ∧ y = point.2 ∨ 
     ∃ t : ℝ, x = point.1 + 3 * t ∧ y = point.2 + 2 * t)) ∧
  (∀ x y : ℝ, given_line x y → 
    ∀ x' y' : ℝ, perpendicular_line x' y' → 
      (x - x') * 2 + (y - y') * 3 = 0) := by
  sorry


end perpendicular_line_equation_l3489_348909


namespace m_plus_abs_m_nonnegative_l3489_348987

theorem m_plus_abs_m_nonnegative (m : ℚ) : m + |m| ≥ 0 := by sorry

end m_plus_abs_m_nonnegative_l3489_348987


namespace smallest_sum_of_three_l3489_348977

def S : Set Int := {7, 25, -1, 12, -3}

theorem smallest_sum_of_three (s : Set Int) (h : s = S) :
  (∃ (a b c : Int), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a + b + c = 3 ∧
    ∀ (x y z : Int), x ∈ s → y ∈ s → z ∈ s → x ≠ y → y ≠ z → x ≠ z →
      x + y + z ≥ 3) :=
by sorry

end smallest_sum_of_three_l3489_348977


namespace division_algorithm_l3489_348946

theorem division_algorithm (x y : ℤ) (hx : x ≥ 0) (hy : y > 0) :
  ∃! (q r : ℤ), x = q * y + r ∧ 0 ≤ r ∧ r < y := by
  sorry

end division_algorithm_l3489_348946


namespace cakes_served_yesterday_l3489_348926

theorem cakes_served_yesterday (lunch_today dinner_today total : ℕ) 
  (h1 : lunch_today = 5)
  (h2 : dinner_today = 6)
  (h3 : total = 14) :
  total - (lunch_today + dinner_today) = 3 := by
  sorry

end cakes_served_yesterday_l3489_348926


namespace max_m_value_max_m_is_optimal_l3489_348959

-- Define the quadratic function
def f (x : ℝ) := x^2 - 4*x

-- State the theorem
theorem max_m_value :
  (∀ x ∈ Set.Ioo 0 1, f x ≥ m) → m ≤ -3 :=
by sorry

-- Define the maximum value of m
def max_m : ℝ := -3

-- Prove that this is indeed the maximum value
theorem max_m_is_optimal :
  (∀ x ∈ Set.Ioo 0 1, f x ≥ max_m) ∧
  ∀ ε > 0, ∃ x ∈ Set.Ioo 0 1, f x < max_m + ε :=
by sorry

end max_m_value_max_m_is_optimal_l3489_348959


namespace smallest_base_for_90_in_three_digits_l3489_348918

theorem smallest_base_for_90_in_three_digits : 
  ∀ b : ℕ, b > 0 → (b^2 ≤ 90 ∧ 90 < b^3) → b ≥ 5 :=
by sorry

end smallest_base_for_90_in_three_digits_l3489_348918


namespace negation_of_negation_l3489_348955

theorem negation_of_negation : -(-2023) = 2023 := by
  sorry

end negation_of_negation_l3489_348955


namespace conditional_statement_b_is_content_when_met_l3489_348913

/-- Represents the structure of a conditional statement -/
structure ConditionalStatement where
  condition : Prop
  contentWhenMet : Prop
  contentWhenNotMet : Prop

/-- Theorem stating that B in a conditional statement represents the content executed when the condition is met -/
theorem conditional_statement_b_is_content_when_met (stmt : ConditionalStatement) :
  stmt.contentWhenMet = stmt.contentWhenMet := by sorry

end conditional_statement_b_is_content_when_met_l3489_348913


namespace function_symmetry_l3489_348905

/-- Given a function f: ℝ → ℝ, if the graph of f(x-1) is symmetric to the curve y = e^x 
    with respect to the y-axis, then f(x) = e^(-x-1) -/
theorem function_symmetry (f : ℝ → ℝ) : 
  (∀ x : ℝ, f (x - 1) = Real.exp (-x)) → 
  (∀ x : ℝ, f x = Real.exp (-x - 1)) := by
  sorry

end function_symmetry_l3489_348905


namespace cubic_yards_to_cubic_feet_l3489_348989

-- Define the conversion factor
def yards_to_feet : ℝ := 3

-- Define the volume in cubic yards
def cubic_yards : ℝ := 5

-- Theorem to prove
theorem cubic_yards_to_cubic_feet :
  cubic_yards * yards_to_feet^3 = 135 := by
  sorry

end cubic_yards_to_cubic_feet_l3489_348989


namespace roots_sum_reciprocals_l3489_348908

theorem roots_sum_reciprocals (p q : ℝ) (x₁ x₂ : ℝ) (hx₁ : x₁^2 + p*x₁ + q = 0) (hx₂ : x₂^2 + p*x₂ + q = 0) (hq : q ≠ 0) :
  x₁/x₂ + x₂/x₁ = (p^2 - 2*q) / q :=
by sorry

end roots_sum_reciprocals_l3489_348908


namespace relationship_between_exponents_l3489_348952

theorem relationship_between_exponents (a b c d x y q z : ℝ) 
  (h1 : a^(2*x) = c^(3*q)) 
  (h2 : a^(2*x) = b) 
  (h3 : c^(4*y) = a^(3*z)) 
  (h4 : c^(4*y) = d) 
  (h5 : a ≠ 0) 
  (h6 : c ≠ 0) : 
  9*q*z = 8*x*y := by
sorry

end relationship_between_exponents_l3489_348952


namespace point_not_in_quadrants_III_IV_l3489_348933

theorem point_not_in_quadrants_III_IV (m : ℝ) : 
  let A : ℝ × ℝ := (m, m^2 + 1)
  ¬(A.1 ≤ 0 ∧ A.2 ≤ 0) ∧ ¬(A.1 ≥ 0 ∧ A.2 ≤ 0) := by
  sorry

end point_not_in_quadrants_III_IV_l3489_348933


namespace marathon_remainder_l3489_348906

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a distance in miles and yards -/
structure Distance where
  miles : ℕ
  yards : ℕ

def marathon_length : Marathon :=
  { miles := 26, yards := 385 }

def yards_per_mile : ℕ := 1760

def number_of_marathons : ℕ := 5

theorem marathon_remainder (m : ℕ) (y : ℕ) 
  (h : Distance.mk m y = 
    { miles := number_of_marathons * marathon_length.miles + (number_of_marathons * marathon_length.yards) / yards_per_mile,
      yards := (number_of_marathons * marathon_length.yards) % yards_per_mile }) 
  (h_range : y < yards_per_mile) : 
  y = 165 := by
  sorry

end marathon_remainder_l3489_348906


namespace range_of_m_l3489_348916

theorem range_of_m (P S : Set ℝ) (m : ℝ) : 
  P = {x : ℝ | x^2 - 8*x - 20 ≤ 0} →
  S = {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m} →
  S.Nonempty →
  (∀ x, x ∉ P → x ∉ S) →
  (∃ x, x ∉ P ∧ x ∈ S) →
  m ≥ 9 ∧ ∀ k ≥ 9, ∃ S', S' = {x : ℝ | 1 - k ≤ x ∧ x ≤ 1 + k} ∧
    S'.Nonempty ∧
    (∀ x, x ∉ P → x ∉ S') ∧
    (∃ x, x ∉ P ∧ x ∈ S') :=
by sorry

end range_of_m_l3489_348916


namespace log_equation_solution_l3489_348931

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 = 5 →
  x = 3^(10/3) := by
sorry

end log_equation_solution_l3489_348931


namespace range_of_m_l3489_348945

theorem range_of_m : 
  (∀ x : ℝ, ∃ m : ℝ, 4^x - 2^(x+1) + m = 0) → 
  (∀ m : ℝ, (∃ x : ℝ, 4^x - 2^(x+1) + m = 0) → m ≤ 1) :=
by sorry

end range_of_m_l3489_348945


namespace tangent_circles_radius_l3489_348961

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  Real.sqrt ((c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2) = r1 + r2

theorem tangent_circles_radius (r : ℝ) :
  r > 0 →
  externally_tangent (0, 0) (3, 0) 1 r →
  r = 2 := by
sorry

end tangent_circles_radius_l3489_348961


namespace arrangements_equal_42_l3489_348920

/-- The number of departments in the unit -/
def num_departments : ℕ := 3

/-- The number of people returning after training -/
def num_returning : ℕ := 2

/-- The maximum number of people that can be accommodated in each department -/
def max_per_department : ℕ := 1

/-- A function that calculates the number of different arrangements -/
def num_arrangements (n d r m : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the number of arrangements is 42 -/
theorem arrangements_equal_42 : 
  num_arrangements num_departments num_departments num_returning max_per_department = 42 :=
sorry

end arrangements_equal_42_l3489_348920


namespace intersection_empty_iff_m_nonnegative_l3489_348930

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*x + m = 0}
def B : Set ℝ := {x : ℝ | x > 0}

-- State the theorem
theorem intersection_empty_iff_m_nonnegative (m : ℝ) :
  A m ∩ B = ∅ ↔ m ∈ Set.Ici (0 : ℝ) := by sorry

end intersection_empty_iff_m_nonnegative_l3489_348930


namespace max_z_value_l3489_348986

theorem max_z_value (x y z : ℝ) (sum_eq : x + y + z = 5) (prod_eq : x*y + y*z + z*x = 3) :
  z ≤ 13/3 := by
  sorry

end max_z_value_l3489_348986


namespace morse_code_symbols_l3489_348972

/-- The number of possible symbols for a given sequence length in Morse code -/
def morse_combinations (n : ℕ) : ℕ := 2^n

/-- The total number of distinct Morse code symbols for sequences up to length 5 -/
def total_morse_symbols : ℕ :=
  (morse_combinations 1) + (morse_combinations 2) + (morse_combinations 3) +
  (morse_combinations 4) + (morse_combinations 5)

theorem morse_code_symbols :
  total_morse_symbols = 62 :=
by sorry

end morse_code_symbols_l3489_348972


namespace watch_time_calculation_l3489_348939

/-- The total watching time for two shows, where the second is 4 times longer than the first -/
def total_watching_time (first_show_duration : ℕ) : ℕ :=
  first_show_duration + 4 * first_show_duration

/-- Theorem stating that given a 30-minute show and another 4 times longer, the total watching time is 150 minutes -/
theorem watch_time_calculation : total_watching_time 30 = 150 := by
  sorry

end watch_time_calculation_l3489_348939


namespace largest_prime_factor_of_expression_l3489_348935

theorem largest_prime_factor_of_expression : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (12^3 + 15^4 - 6^5) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (12^3 + 15^4 - 6^5) → q ≤ p :=
by
  -- The proof would go here
  sorry

end largest_prime_factor_of_expression_l3489_348935


namespace ball_return_to_start_l3489_348965

def circle_size : ℕ := 14
def step_size : ℕ := 3

theorem ball_return_to_start :
  ∀ (start : ℕ),
  start < circle_size →
  (∃ (n : ℕ), n > 0 ∧ (start + n * step_size) % circle_size = start) →
  (∀ (m : ℕ), 0 < m → m < circle_size → (start + m * step_size) % circle_size ≠ start) →
  (start + circle_size * step_size) % circle_size = start :=
by sorry

#check ball_return_to_start

end ball_return_to_start_l3489_348965


namespace translation_down_three_units_l3489_348949

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically -/
def translateLine (l : Line) (dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - dy }

theorem translation_down_three_units :
  let originalLine : Line := { slope := 1/2, intercept := 0 }
  let translatedLine : Line := translateLine originalLine 3
  translatedLine = { slope := 1/2, intercept := -3 } := by
  sorry

end translation_down_three_units_l3489_348949


namespace complex_number_in_second_quadrant_l3489_348981

def i : ℂ := Complex.I

theorem complex_number_in_second_quadrant :
  let z : ℂ := (1 + 2*i) / (1 + 2*i^3)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end complex_number_in_second_quadrant_l3489_348981


namespace square_area_error_l3489_348966

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * 1.04
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error_percentage := (calculated_area - actual_area) / actual_area * 100
  area_error_percentage = 8.16 := by
    sorry

end square_area_error_l3489_348966


namespace only_131_not_in_second_column_l3489_348999

def second_column (n : ℕ) : ℕ := 3 * n + 1

theorem only_131_not_in_second_column :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 400 →
    (31 = second_column n ∨
     94 = second_column n ∨
     331 = second_column n ∨
     907 = second_column n) ∧
    ¬(131 = second_column n) := by
  sorry

end only_131_not_in_second_column_l3489_348999


namespace arithmetic_sequence_exists_geometric_sequence_not_exists_l3489_348980

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral :=
  (a b c d : ℝ)
  (sum_opposite : a + c = 180 ∧ b + d = 180)
  (angle_bounds : 0 < a ∧ a < 180 ∧ 0 < b ∧ b < 180 ∧ 0 < c ∧ c < 180 ∧ 0 < d ∧ d < 180)

-- Theorem for arithmetic sequence
theorem arithmetic_sequence_exists (q : CyclicQuadrilateral) :
  ∃ (α d : ℝ), d ≠ 0 ∧
    q.a = α ∧ q.b = α + d ∧ q.c = α + 2*d ∧ q.d = α + 3*d :=
sorry

-- Theorem for geometric sequence
theorem geometric_sequence_not_exists (q : CyclicQuadrilateral) :
  ¬∃ (α r : ℝ), r ≠ 1 ∧ r > 0 ∧
    q.a = α ∧ q.b = α * r ∧ q.c = α * r^2 ∧ q.d = α * r^3 :=
sorry

end arithmetic_sequence_exists_geometric_sequence_not_exists_l3489_348980


namespace chips_bought_l3489_348954

/-- Given three friends paying $5 each for bags of chips costing $3 per bag,
    prove that they can buy 5 bags of chips. -/
theorem chips_bought (num_friends : ℕ) (payment_per_friend : ℕ) (cost_per_bag : ℕ) :
  num_friends = 3 →
  payment_per_friend = 5 →
  cost_per_bag = 3 →
  (num_friends * payment_per_friend) / cost_per_bag = 5 :=
by sorry

end chips_bought_l3489_348954


namespace teacher_in_middle_girls_not_adjacent_teacher_flanked_by_girls_l3489_348973

-- Define the class composition
def num_boys : ℕ := 4
def num_girls : ℕ := 2
def num_teacher : ℕ := 1

-- Define the total number of people
def total_people : ℕ := num_boys + num_girls + num_teacher

-- Theorem for scenario 1
theorem teacher_in_middle :
  (Nat.factorial (total_people - 1)) = 720 := by sorry

-- Theorem for scenario 2
theorem girls_not_adjacent :
  (Nat.factorial (total_people - num_girls)) * (Nat.factorial (total_people - num_girls - 1)) = 2400 := by sorry

-- Theorem for scenario 3
theorem teacher_flanked_by_girls :
  (Nat.factorial (total_people - num_girls)) * (Nat.factorial num_girls) = 240 := by sorry

end teacher_in_middle_girls_not_adjacent_teacher_flanked_by_girls_l3489_348973


namespace shaded_area_equals_1150_l3489_348901

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- The main theorem stating the area of the shaded region -/
theorem shaded_area_equals_1150 :
  let square_side : ℝ := 40
  let p1 : Point := ⟨0, 0⟩
  let p2 : Point := ⟨20, 0⟩
  let p3 : Point := ⟨40, 30⟩
  let p4 : Point := ⟨40, 40⟩
  let p5 : Point := ⟨10, 40⟩
  let p6 : Point := ⟨0, 10⟩
  let square_area := square_side * square_side
  let triangle1_area := triangleArea p2 ⟨40, 0⟩ p3
  let triangle2_area := triangleArea p6 ⟨0, 40⟩ p5
  square_area - (triangle1_area + triangle2_area) = 1150 := by
  sorry

end shaded_area_equals_1150_l3489_348901


namespace polynomial_factorization_l3489_348964

theorem polynomial_factorization (p q : ℝ) :
  ∃ (a b c d e f : ℝ), ∀ (x : ℝ),
    x^4 + p*x^2 + q = (a*x^2 + b*x + c) * (d*x^2 + e*x + f) := by
  sorry

end polynomial_factorization_l3489_348964


namespace states_fraction_1840_to_1849_l3489_348938

theorem states_fraction_1840_to_1849 (total_states : ℕ) (joined_1840_to_1849 : ℕ) :
  total_states = 33 →
  joined_1840_to_1849 = 6 →
  (joined_1840_to_1849 : ℚ) / total_states = 2 / 11 := by
  sorry

end states_fraction_1840_to_1849_l3489_348938


namespace locus_of_center_C_l3489_348934

/-- Circle C₁ with equation x² + y² + 4y + 3 = 0 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 4*p.2 + 3 = 0}

/-- Circle C₂ with equation x² + y² - 4y - 77 = 0 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.2 - 77 = 0}

/-- The locus of the center of circle C -/
def locus_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 / 25 + p.1^2 / 21 = 1}

/-- Theorem stating that the locus of the center of circle C forms an ellipse
    given the tangency conditions with C₁ and C₂ -/
theorem locus_of_center_C (C : Set (ℝ × ℝ)) :
  (∃ r : ℝ, ∀ p ∈ C, ∃ q ∈ C₁, ‖p - q‖ = r) →  -- C is externally tangent to C₁
  (∃ R : ℝ, ∀ p ∈ C, ∃ q ∈ C₂, ‖p - q‖ = R) →  -- C is internally tangent to C₂
  C = locus_C :=
sorry

end locus_of_center_C_l3489_348934


namespace least_product_of_distinct_primes_above_20_l3489_348937

theorem least_product_of_distinct_primes_above_20 :
  ∃ (p q : ℕ), 
    p.Prime ∧ q.Prime ∧ 
    p > 20 ∧ q > 20 ∧ 
    p ≠ q ∧
    p * q = 667 ∧
    ∀ (r s : ℕ), r.Prime → s.Prime → r > 20 → s > 20 → r ≠ s → r * s ≥ 667 :=
by sorry

end least_product_of_distinct_primes_above_20_l3489_348937


namespace airport_walk_probability_l3489_348948

/-- Represents an airport with a given number of gates and distance between adjacent gates -/
structure Airport where
  num_gates : ℕ
  distance_between_gates : ℕ

/-- Calculates the number of gate pairs within a given distance -/
def count_pairs_within_distance (a : Airport) (max_distance : ℕ) : ℕ :=
  sorry

/-- The probability of walking at most a given distance between two random gates -/
def probability_within_distance (a : Airport) (max_distance : ℕ) : ℚ :=
  sorry

theorem airport_walk_probability :
  let a : Airport := ⟨15, 90⟩
  probability_within_distance a 360 = 59 / 105 := by
  sorry

end airport_walk_probability_l3489_348948


namespace complex_modulus_l3489_348923

theorem complex_modulus (z : ℂ) (h : z * (1 - Complex.I) = 2 * Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_l3489_348923


namespace inverse_direct_proportionality_l3489_348922

/-- Given two real numbers are inversely proportional -/
def inversely_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

/-- Given two real numbers are directly proportional -/
def directly_proportional (z y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ z = k * y

/-- Main theorem -/
theorem inverse_direct_proportionality
  (x y z : ℝ → ℝ)
  (h_inv : ∀ t, inversely_proportional (x t) (y t))
  (h_dir : ∀ t, directly_proportional (z t) (y t))
  (h_x : x 9 = 40)
  (h_z : z 10 = 45) :
  x 20 = 18 ∧ z 20 = 90 := by
  sorry


end inverse_direct_proportionality_l3489_348922


namespace water_volume_in_first_solution_l3489_348943

/-- The cost per liter of a spirit-water solution is directly proportional to the fraction of spirit by volume. -/
axiom cost_proportional_to_spirit_fraction (cost spirit_vol total_vol : ℝ) : 
  cost = (spirit_vol / total_vol) * (cost * total_vol / spirit_vol)

/-- The cost of the first solution with 1 liter of spirit and an unknown amount of water -/
def first_solution_cost : ℝ := 0.50

/-- The cost of the second solution with 1 liter of spirit and 2 liters of water -/
def second_solution_cost : ℝ := 0.50

/-- The volume of spirit in both solutions -/
def spirit_volume : ℝ := 1

/-- The volume of water in the second solution -/
def second_solution_water_volume : ℝ := 2

/-- The volume of water in the first solution -/
def first_solution_water_volume : ℝ := 2

theorem water_volume_in_first_solution : 
  first_solution_water_volume = 2 := by sorry

end water_volume_in_first_solution_l3489_348943


namespace no_real_roots_l3489_348902

/-- Given a function f and constants a and b, prove that f(ax + b) has no real roots -/
theorem no_real_roots (f : ℝ → ℝ) (a b : ℝ) : 
  (∀ x, f x = x^2 + 2*x + a) →
  (∀ x, f (b*x) = 9*x - 6*x + 2) →
  (∀ x, f (a*x + b) ≠ 0) :=
by sorry

end no_real_roots_l3489_348902


namespace lulu_blueberry_pies_count_l3489_348995

/-- The number of blueberry pies Lulu baked -/
def lulu_blueberry_pies : ℕ := 73 - (13 + 10 + 8 + 16 + 12)

theorem lulu_blueberry_pies_count :
  lulu_blueberry_pies = 14 := by
  sorry

end lulu_blueberry_pies_count_l3489_348995
