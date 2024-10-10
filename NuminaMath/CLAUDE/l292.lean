import Mathlib

namespace hotel_nights_calculation_l292_29265

theorem hotel_nights_calculation (total_value car_value hotel_cost_per_night : ℕ) 
  (h1 : total_value = 158000)
  (h2 : car_value = 30000)
  (h3 : hotel_cost_per_night = 4000) :
  (total_value - (car_value + 4 * car_value)) / hotel_cost_per_night = 2 := by
  sorry

end hotel_nights_calculation_l292_29265


namespace horror_movie_tickets_l292_29245

theorem horror_movie_tickets (romance_tickets horror_tickets : ℕ) : 
  romance_tickets = 25 →
  horror_tickets = 3 * romance_tickets + 18 →
  horror_tickets = 93 := by
sorry

end horror_movie_tickets_l292_29245


namespace apples_in_basket_l292_29204

/-- The number of apples left in a basket after removals --/
def applesLeft (initial : ℕ) (rickiRemoves : ℕ) : ℕ :=
  initial - rickiRemoves - (2 * rickiRemoves)

/-- Theorem stating the number of apples left in the basket --/
theorem apples_in_basket : applesLeft 74 14 = 32 := by
  sorry

end apples_in_basket_l292_29204


namespace tree_height_proof_l292_29219

/-- Proves that a tree with current height 180 inches, which is 50% taller than its original height, had an original height of 10 feet. -/
theorem tree_height_proof (current_height : ℝ) (height_increase_percent : ℝ) 
  (h1 : current_height = 180)
  (h2 : height_increase_percent = 50)
  (h3 : current_height = (1 + height_increase_percent / 100) * (12 * 10)) : 
  ∃ (original_height_feet : ℝ), original_height_feet = 10 :=
by
  sorry

#check tree_height_proof

end tree_height_proof_l292_29219


namespace milk_water_mixture_l292_29201

theorem milk_water_mixture (milk water : ℝ) : 
  milk / water = 2 →
  milk / (water + 10) = 6 / 5 →
  milk = 30 := by
sorry

end milk_water_mixture_l292_29201


namespace parabola_focus_coordinates_l292_29293

/-- The focus of the parabola y = 3x^2 has coordinates (0, 1/12) -/
theorem parabola_focus_coordinates :
  ∀ (x y : ℝ), y = 3 * x^2 → ∃ (p : ℝ), p > 0 ∧ x^2 = (1/(4*p)) * y ∧ (0, p) = (0, 1/12) :=
by sorry

end parabola_focus_coordinates_l292_29293


namespace city_population_l292_29202

theorem city_population (population_percentage : Real) (partial_population : ℕ) (total_population : ℕ) : 
  population_percentage = 0.85 →
  partial_population = 85000 →
  population_percentage * (total_population : Real) = partial_population →
  total_population = 100000 :=
by
  sorry

end city_population_l292_29202


namespace new_person_weight_l292_29236

/-- Given 4 people, with one weighing 95 kg, if the average weight increases by 8.5 kg
    when a new person replaces the 95 kg person, then the new person weighs 129 kg. -/
theorem new_person_weight (initial_count : Nat) (replaced_weight : Real) (avg_increase : Real) :
  initial_count = 4 →
  replaced_weight = 95 →
  avg_increase = 8.5 →
  (initial_count : Real) * avg_increase + replaced_weight = 129 :=
by sorry

end new_person_weight_l292_29236


namespace binary_subtraction_result_l292_29208

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 111111111₂ -/
def binary_111111111 : List Bool := [true, true, true, true, true, true, true, true, true]

/-- The binary representation of 111111₂ -/
def binary_111111 : List Bool := [true, true, true, true, true, true]

/-- The theorem stating that the difference between the decimal representations
    of 111111111₂ and 111111₂ is equal to 448 -/
theorem binary_subtraction_result :
  binary_to_decimal binary_111111111 - binary_to_decimal binary_111111 = 448 := by
  sorry

end binary_subtraction_result_l292_29208


namespace call_duration_l292_29290

def calls_per_year : ℕ := 52
def cost_per_minute : ℚ := 5 / 100
def total_cost_per_year : ℚ := 78

theorem call_duration :
  (total_cost_per_year / cost_per_minute) / calls_per_year = 30 := by
  sorry

end call_duration_l292_29290


namespace tourist_count_l292_29296

theorem tourist_count : 
  ∃ (n : ℕ), 
    (1/2 : ℚ) * n + (1/3 : ℚ) * n + (1/4 : ℚ) * n = 39 ∧ 
    n = 36 := by
  sorry

end tourist_count_l292_29296


namespace number_divided_by_two_equals_number_minus_five_l292_29257

theorem number_divided_by_two_equals_number_minus_five : ∃! x : ℝ, x / 2 = x - 5 := by
  sorry

end number_divided_by_two_equals_number_minus_five_l292_29257


namespace sum_quotient_reciprocal_l292_29286

theorem sum_quotient_reciprocal (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 45) (h4 : x * y = 500) : 
  (x / y) + (1 / x) + (1 / y) = 1.34 := by
  sorry

end sum_quotient_reciprocal_l292_29286


namespace lines_perpendicular_imply_parallel_l292_29269

-- Define a type for lines in 3D space
structure Line3D where
  -- You might want to add more properties here, but for this problem, we only need the line itself
  line : Type

-- Define perpendicularity and parallelism for lines
def perpendicular (l1 l2 : Line3D) : Prop := sorry

def parallel (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem lines_perpendicular_imply_parallel (a b c d : Line3D) 
  (h1 : perpendicular a c)
  (h2 : perpendicular b c)
  (h3 : perpendicular a d)
  (h4 : perpendicular b d) :
  parallel a b ∨ parallel c d := by
  sorry

end lines_perpendicular_imply_parallel_l292_29269


namespace investment_problem_l292_29275

/-- Proves that given the investment conditions, the amount invested at Speedy Growth Bank is $300 --/
theorem investment_problem (total_investment : ℝ) (speedy_rate : ℝ) (safe_rate : ℝ) (final_amount : ℝ)
  (h1 : total_investment = 1500)
  (h2 : speedy_rate = 0.04)
  (h3 : safe_rate = 0.06)
  (h4 : final_amount = 1584)
  (h5 : ∀ x : ℝ, x * (1 + speedy_rate) + (total_investment - x) * (1 + safe_rate) = final_amount) :
  ∃ x : ℝ, x = 300 ∧ x * (1 + speedy_rate) + (total_investment - x) * (1 + safe_rate) = final_amount :=
by sorry

end investment_problem_l292_29275


namespace boxes_per_case_l292_29276

theorem boxes_per_case (total_boxes : ℕ) (total_cases : ℕ) (boxes_per_case : ℕ) : 
  total_boxes = 20 → total_cases = 5 → total_boxes = total_cases * boxes_per_case → boxes_per_case = 4 := by
  sorry

end boxes_per_case_l292_29276


namespace number_multiplied_by_48_l292_29277

theorem number_multiplied_by_48 : ∃ x : ℤ, x * 48 = 173 * 240 ∧ x = 865 := by
  sorry

end number_multiplied_by_48_l292_29277


namespace percentage_female_officers_on_duty_l292_29242

def total_officers_on_duty : ℕ := 204
def female_ratio_on_duty : ℚ := 1/2
def total_female_officers : ℕ := 600

theorem percentage_female_officers_on_duty :
  (total_officers_on_duty * female_ratio_on_duty) / total_female_officers * 100 = 17 := by
  sorry

end percentage_female_officers_on_duty_l292_29242


namespace sqrt_equation_solution_l292_29258

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (8 + 3 * z) = 14 :=
by
  -- Proof goes here
  sorry

end sqrt_equation_solution_l292_29258


namespace football_team_handedness_l292_29283

theorem football_team_handedness (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ)
  (h1 : total_players = 70)
  (h2 : throwers = 28)
  (h3 : right_handed = 56)
  (h4 : throwers ≤ right_handed) :
  (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 := by
sorry

end football_team_handedness_l292_29283


namespace correct_answer_l292_29238

theorem correct_answer (x : ℤ) (h : x - 8 = 32) : x + 8 = 48 := by
  sorry

end correct_answer_l292_29238


namespace ellipse_hyperbola_same_directrix_l292_29239

/-- An ellipse with equation x^2 + k*y^2 = 1 -/
def ellipse (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + k * p.2^2 = 1}

/-- A hyperbola with equation x^2/4 - y^2/5 = 1 -/
def hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 - p.2^2 / 5 = 1}

/-- The directrix of a conic section -/
def directrix (c : Set (ℝ × ℝ)) : Set ℝ := sorry

theorem ellipse_hyperbola_same_directrix (k : ℝ) :
  directrix (ellipse k) = directrix hyperbola → k = 16/7 := by
  sorry

end ellipse_hyperbola_same_directrix_l292_29239


namespace necessary_sufficient_condition_l292_29229

theorem necessary_sufficient_condition (a b : ℝ) :
  a * |a + b| < |a| * (a + b) ↔ a < 0 ∧ b > -a := by
  sorry

end necessary_sufficient_condition_l292_29229


namespace no_solution_for_functional_equation_l292_29244

theorem no_solution_for_functional_equation :
  ¬∃ (f : ℕ → ℕ), ∀ (x : ℕ), f (f x) = x + 1 := by
  sorry

end no_solution_for_functional_equation_l292_29244


namespace systematic_sampling_fourth_student_l292_29298

/-- Systematic sampling function that returns the nth sample given a starting point and interval -/
def systematic_sample (start : Nat) (interval : Nat) (n : Nat) : Nat :=
  start + (n - 1) * interval

theorem systematic_sampling_fourth_student 
  (total_students : Nat) 
  (sample_size : Nat) 
  (sample1 sample2 sample3 : Nat) :
  total_students = 36 →
  sample_size = 4 →
  sample1 = 6 →
  sample2 = 24 →
  sample3 = 33 →
  ∃ (start interval : Nat),
    systematic_sample start interval 1 = sample1 ∧
    systematic_sample start interval 2 = sample2 ∧
    systematic_sample start interval 3 = sample3 ∧
    systematic_sample start interval 4 = 15 :=
by
  sorry

#check systematic_sampling_fourth_student

end systematic_sampling_fourth_student_l292_29298


namespace modulo_17_residue_l292_29225

theorem modulo_17_residue : (305 + 7 * 51 + 11 * 187 + 6 * 23) % 17 = 3 := by
  sorry

end modulo_17_residue_l292_29225


namespace bike_speed_l292_29297

/-- Proves that a bike moving at constant speed, covering 32 meters in 8 seconds, has a speed of 4 meters per second -/
theorem bike_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 32 → time = 8 → speed = distance / time → speed = 4 := by
  sorry

end bike_speed_l292_29297


namespace class_gender_ratio_l292_29210

/-- Proves that given the boys' average score of 90, girls' average score of 96,
    and overall class average of 94, the ratio of boys to girls in the class is 1:2. -/
theorem class_gender_ratio (B G : ℕ) (B_pos : B > 0) (G_pos : G > 0) : 
  (90 * B + 96 * G) / (B + G) = 94 → B = G / 2 := by
  sorry

end class_gender_ratio_l292_29210


namespace melissas_total_points_l292_29252

/-- Calculates the total points scored in multiple games -/
def totalPoints (gamesPlayed : ℕ) (pointsPerGame : ℕ) : ℕ :=
  gamesPlayed * pointsPerGame

/-- Proves that Melissa's total points is 81 -/
theorem melissas_total_points :
  let gamesPlayed : ℕ := 3
  let pointsPerGame : ℕ := 27
  totalPoints gamesPlayed pointsPerGame = 81 := by
  sorry

end melissas_total_points_l292_29252


namespace remainder_theorem_l292_29228

/-- The polynomial to be divided -/
def f (x : ℝ) : ℝ := x^5 - 2*x^4 - x^3 + 2*x^2 + x

/-- The divisor polynomial -/
def g (x : ℝ) : ℝ := (x^2 - 9) * (x - 1)

/-- The proposed remainder -/
def r (x : ℝ) : ℝ := 9*x^2 + 73*x - 81

/-- Theorem stating that r is the remainder when f is divided by g -/
theorem remainder_theorem : ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x := by sorry

end remainder_theorem_l292_29228


namespace hyperbola_equation_l292_29288

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the line
def line (x y : ℝ) : Prop :=
  y = 2*x - 4

-- Define the right focus
def right_focus (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 = a^2 + b^2 ∧ x > 0 ∧ y = 0

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (x y : ℝ), right_focus a b x y ∧ line x y) →
  (∃! (p : ℝ × ℝ), hyperbola a b p.1 p.2 ∧ line p.1 p.2) →
  (∀ (x y : ℝ), hyperbola a b x y ↔ 5*x^2/4 - 5*y^2/16 = 1) :=
by sorry

end hyperbola_equation_l292_29288


namespace right_triangle_arithmetic_sides_ratio_l292_29233

-- Define a right-angled triangle with sides forming an arithmetic sequence
structure RightTriangleArithmeticSides where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2
  arithmetic_sequence : ∃ d : ℝ, b = a + d ∧ c = b + d

-- Theorem statement
theorem right_triangle_arithmetic_sides_ratio 
  (t : RightTriangleArithmeticSides) : 
  ∃ k : ℝ, t.a = 3*k ∧ t.b = 4*k ∧ t.c = 5*k := by
sorry

end right_triangle_arithmetic_sides_ratio_l292_29233


namespace church_seating_capacity_l292_29294

theorem church_seating_capacity (chairs_per_row : ℕ) (num_rows : ℕ) (total_people : ℕ) :
  chairs_per_row = 6 →
  num_rows = 20 →
  total_people = 600 →
  total_people / (chairs_per_row * num_rows) = 5 :=
by sorry

end church_seating_capacity_l292_29294


namespace digit_difference_in_base_d_l292_29267

/-- Given two digits X and Y in base d > 8, if XY_d + XX_d = 234_d, then X_d - Y_d = -2_d. -/
theorem digit_difference_in_base_d (d : ℕ) (X Y : ℕ) (h_d : d > 8) 
  (h_digits : X < d ∧ Y < d) 
  (h_sum : X * d + Y + X * d + X = 2 * d * d + 3 * d + 4) :
  X - Y = d - 2 := by
  sorry

end digit_difference_in_base_d_l292_29267


namespace square_of_107_l292_29218

theorem square_of_107 : (107 : ℕ)^2 = 11449 := by
  sorry

end square_of_107_l292_29218


namespace value_of_expression_l292_29231

theorem value_of_expression (x : ℝ) (h : 6 * x^2 - 4 * x - 3 = 0) :
  (x - 1)^2 + x * (x + 2/3) = 2 := by
  sorry

end value_of_expression_l292_29231


namespace stating_chess_team_arrangements_l292_29216

/-- Represents the number of boys on the chess team -/
def num_boys : Nat := 3

/-- Represents the number of girls on the chess team -/
def num_girls : Nat := 3

/-- Represents the total number of students on the chess team -/
def total_students : Nat := num_boys + num_girls

/-- 
Represents the number of ways to arrange the chess team in a row 
such that all boys are at the ends and exactly one boy is in the middle
-/
def num_arrangements : Nat := 36

/-- 
Theorem stating that the number of arrangements of the chess team
satisfying the given conditions is equal to 36
-/
theorem chess_team_arrangements : 
  (num_boys = 3 ∧ num_girls = 3) → num_arrangements = 36 := by
  sorry

end stating_chess_team_arrangements_l292_29216


namespace water_balloon_count_l292_29274

-- Define the number of water balloons for each person
def sarah_balloons : ℕ := 5
def janice_balloons : ℕ := 6

-- Define the relationships between the number of water balloons
theorem water_balloon_count :
  ∀ (tim_balloons randy_balloons cynthia_balloons : ℕ),
  (tim_balloons = 2 * sarah_balloons) →
  (tim_balloons + 3 = janice_balloons) →
  (2 * randy_balloons = janice_balloons) →
  (cynthia_balloons = 4 * randy_balloons) →
  cynthia_balloons = 12 := by
  sorry

end water_balloon_count_l292_29274


namespace solutions_count_l292_29299

theorem solutions_count : 
  (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ 3 * p.1 + p.2 = 100) 
    (Finset.product (Finset.range 101) (Finset.range 101))).card = 33 := by
  sorry

end solutions_count_l292_29299


namespace concentric_circles_area_ratio_l292_29273

/-- Two concentric circles with center Q -/
structure ConcentricCircles where
  center : Point
  radius₁ : ℝ
  radius₂ : ℝ
  h : radius₁ < radius₂

/-- The length of an arc given its central angle and the circle's radius -/
def arcLength (angle : ℝ) (radius : ℝ) : ℝ := angle * radius

theorem concentric_circles_area_ratio 
  (circles : ConcentricCircles) 
  (h : arcLength (π/3) circles.radius₁ = arcLength (π/6) circles.radius₂) : 
  (circles.radius₁^2) / (circles.radius₂^2) = 1/4 := by
  sorry

end concentric_circles_area_ratio_l292_29273


namespace robie_chocolates_l292_29248

/-- The number of bags of chocolates Robie has after her purchases and giveaway. -/
def final_bags (initial : ℕ) (given_away : ℕ) (bought_later : ℕ) : ℕ :=
  initial - given_away + bought_later

/-- Theorem stating that Robie ends up with 4 bags of chocolates. -/
theorem robie_chocolates : final_bags 3 2 3 = 4 := by
  sorry

end robie_chocolates_l292_29248


namespace tour_budget_l292_29271

/-- Given a tour scenario, proves that the total budget for the original tour is 360 units -/
theorem tour_budget (original_days : ℕ) (extension_days : ℕ) (expense_reduction : ℕ) : 
  original_days = 20 → 
  extension_days = 4 → 
  expense_reduction = 3 →
  (original_days * (original_days + extension_days)) / extension_days = 360 :=
by
  sorry

#check tour_budget

end tour_budget_l292_29271


namespace smallest_integer_y_l292_29227

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def expression (y : ℤ) : ℚ := (y^2 - 3*y + 11) / (y - 5)

theorem smallest_integer_y : 
  (∀ y : ℤ, y < 6 → ¬(is_integer (expression y))) ∧ 
  (is_integer (expression 6)) := by sorry

end smallest_integer_y_l292_29227


namespace hawthorn_box_maximum_l292_29264

theorem hawthorn_box_maximum (N : ℕ) : 
  N > 100 ∧
  N % 3 = 1 ∧
  N % 4 = 2 ∧
  N % 5 = 3 ∧
  N % 6 = 4 →
  N ≤ 178 ∧ ∃ (M : ℕ), M = 178 ∧ 
    M > 100 ∧
    M % 3 = 1 ∧
    M % 4 = 2 ∧
    M % 5 = 3 ∧
    M % 6 = 4 := by
  sorry

end hawthorn_box_maximum_l292_29264


namespace mrs_hilt_animal_legs_l292_29249

theorem mrs_hilt_animal_legs :
  let num_dogs : ℕ := 2
  let num_chickens : ℕ := 2
  let dog_legs : ℕ := 4
  let chicken_legs : ℕ := 2
  num_dogs * dog_legs + num_chickens * chicken_legs = 12 :=
by sorry

end mrs_hilt_animal_legs_l292_29249


namespace line_circle_separate_trajectory_of_P_l292_29255

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the line l
def line_l (a b α t : ℝ) (x y : ℝ) : Prop :=
  x = a + t * Real.cos α ∧ y = b + t * Real.sin α

-- Part 1: Line and circle are separate
theorem line_circle_separate :
  ∀ x y t : ℝ,
  line_l 8 0 (π/3) t x y →
  ¬ circle_C x y :=
sorry

-- Part 2: Trajectory of point P
theorem trajectory_of_P :
  ∀ a b x y : ℝ,
  (∃ α t₁ t₂ : ℝ,
    circle_C (a + t₁ * Real.cos α) (b + t₁ * Real.sin α) ∧
    circle_C (a + t₂ * Real.cos α) (b + t₂ * Real.sin α) ∧
    t₁ ≠ t₂ ∧
    (a^2 + b^2) * ((a + t₁ * Real.cos α)^2 + (b + t₁ * Real.sin α)^2) =
    ((a + t₂ * Real.cos α)^2 + (b + t₂ * Real.sin α)^2) * a^2 + b^2) →
  x^2 + y^2 = 8 :=
sorry

end line_circle_separate_trajectory_of_P_l292_29255


namespace remaining_distance_to_hotel_l292_29285

/-- Calculates the remaining distance to the hotel given Samuel's journey conditions --/
theorem remaining_distance_to_hotel : 
  let total_distance : ℕ := 600
  let speed1 : ℕ := 50
  let time1 : ℕ := 3
  let speed2 : ℕ := 80
  let time2 : ℕ := 4
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let traveled_distance := distance1 + distance2
  total_distance - traveled_distance = 130 := by
  sorry

end remaining_distance_to_hotel_l292_29285


namespace complex_fraction_sum_l292_29214

theorem complex_fraction_sum : 
  let S := 1 / (2 - Real.sqrt 3) - 1 / (Real.sqrt 3 - Real.sqrt 2) + 
           1 / (Real.sqrt 2 - 1) - 1 / (1 - Real.sqrt 3 + Real.sqrt 2)
  S = 2 := by
  sorry

end complex_fraction_sum_l292_29214


namespace asymptote_coincidence_l292_29243

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the parabola
def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the asymptote of the hyperbola
def hyperbola_asymptote (x : ℝ) : Prop := x = -3/2 ∨ x = 3/2

-- Define the asymptote of the parabola
def parabola_asymptote (p x : ℝ) : Prop := x = -p/2

-- State the theorem
theorem asymptote_coincidence (p : ℝ) :
  (p > 0) →
  (∃ x : ℝ, hyperbola_asymptote x ∧ parabola_asymptote p x) →
  p = 3 :=
sorry

end asymptote_coincidence_l292_29243


namespace age_problem_l292_29209

/-- The problem of finding when B was half the age A will be in 10 years -/
theorem age_problem (B_age : ℕ) (A_age : ℕ) (x : ℕ) : 
  B_age = 37 →
  A_age = B_age + 7 →
  B_age - x = (A_age + 10) / 2 →
  x = 10 := by
  sorry

end age_problem_l292_29209


namespace nine_zeros_in_binary_representation_l292_29237

/-- The number of zeros in the binary representation of a natural number -/
def countZeros (n : ℕ) : ℕ := sorry

/-- An unknown non-negative integer -/
def someNumber : ℕ := sorry

/-- The main expression: 6 * 1024 + 4 * 64 + someNumber -/
def mainExpression : ℕ := 6 * 1024 + 4 * 64 + someNumber

theorem nine_zeros_in_binary_representation :
  countZeros mainExpression = 9 := by sorry

end nine_zeros_in_binary_representation_l292_29237


namespace z_in_fourth_quadrant_l292_29284

theorem z_in_fourth_quadrant (z : ℂ) (h : z * Complex.I = 2 + Complex.I) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end z_in_fourth_quadrant_l292_29284


namespace number_exceeding_fraction_l292_29222

theorem number_exceeding_fraction (x : ℚ) : x = (3 / 8) * x + 35 → x = 56 := by
  sorry

end number_exceeding_fraction_l292_29222


namespace enrico_earnings_l292_29220

/-- Calculates the earnings from selling roosters -/
def rooster_earnings (price_per_kg : ℚ) (weights : List ℚ) : ℚ :=
  (weights.map (· * price_per_kg)).sum

/-- Proves that Enrico's earnings from selling two roosters are $35 -/
theorem enrico_earnings : 
  let price_per_kg : ℚ := 1/2
  let weights : List ℚ := [30, 40]
  rooster_earnings price_per_kg weights = 35 := by
sorry

#eval rooster_earnings (1/2) [30, 40]

end enrico_earnings_l292_29220


namespace intersection_of_M_and_N_l292_29247

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end intersection_of_M_and_N_l292_29247


namespace only_paintable_integer_l292_29241

/-- Represents a painting pattern for the fence. -/
structure PaintingPattern where
  start : ℕ
  interval : ℕ

/-- Checks if a given triple (h, t, u) results in a valid painting pattern. -/
def isValidPainting (h t u : ℕ) : Prop :=
  let harold := PaintingPattern.mk 4 h
  let tanya := PaintingPattern.mk 5 (2 * t)
  let ulysses := PaintingPattern.mk 6 (3 * u)
  ∀ n : ℕ, n ≥ 1 →
    (∃! painter, painter ∈ [harold, tanya, ulysses] ∧
      ∃ k, n = painter.start + painter.interval * k)

/-- Calculates the paintable integer for a given triple (h, t, u). -/
def paintableInteger (h t u : ℕ) : ℕ :=
  100 * h + 20 * t + 2 * u

/-- The main theorem stating that 390 is the only paintable integer. -/
theorem only_paintable_integer :
  ∀ h t u : ℕ, h > 0 ∧ t > 0 ∧ u > 0 →
    isValidPainting h t u ↔ paintableInteger h t u = 390 :=
sorry

end only_paintable_integer_l292_29241


namespace tabitha_honey_nights_l292_29291

/-- Calculates the number of nights Tabitha can enjoy honey in her tea before running out. -/
def honey_nights (servings_per_cup : ℕ) (cups_per_night : ℕ) (container_size : ℕ) (servings_per_ounce : ℕ) : ℕ :=
  let total_servings := container_size * servings_per_ounce
  let servings_per_night := servings_per_cup * cups_per_night
  total_servings / servings_per_night

/-- Proves that Tabitha can enjoy honey in her tea for 48 nights before running out. -/
theorem tabitha_honey_nights : 
  honey_nights 1 2 16 6 = 48 := by
  sorry

end tabitha_honey_nights_l292_29291


namespace negation_disjunction_true_l292_29259

theorem negation_disjunction_true (p q : Prop) : 
  (p ∧ q) = False → (¬p ∨ ¬q) = True := by sorry

end negation_disjunction_true_l292_29259


namespace hannah_strawberries_l292_29253

theorem hannah_strawberries (x : ℕ) : 
  (30 * x - 20 - 30 = 100) → x = 5 := by
  sorry

end hannah_strawberries_l292_29253


namespace complex_calculation_l292_29278

theorem complex_calculation : 
  let z : ℂ := 1 - Complex.I
  2 / z + z^2 = 1 - Complex.I := by sorry

end complex_calculation_l292_29278


namespace fraction_to_decimal_l292_29207

theorem fraction_to_decimal : (15 : ℚ) / 625 = (24 : ℚ) / 1000 := by sorry

end fraction_to_decimal_l292_29207


namespace original_number_proof_l292_29203

theorem original_number_proof : ∃ N : ℕ, 
  (∀ m : ℕ, m < N → ¬(m - 6 ≡ 3 [MOD 5] ∧ m - 6 ≡ 3 [MOD 11] ∧ m - 6 ≡ 3 [MOD 13])) ∧
  (N - 6 ≡ 3 [MOD 5] ∧ N - 6 ≡ 3 [MOD 11] ∧ N - 6 ≡ 3 [MOD 13]) ∧
  N = 724 :=
by sorry

#check original_number_proof

end original_number_proof_l292_29203


namespace roots_sum_of_squares_l292_29282

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 2*a - 3 = 0) → (b^2 - 2*b - 3 = 0) → (a ≠ b) → a^2 + b^2 = 10 := by
  sorry

end roots_sum_of_squares_l292_29282


namespace perpendicular_vectors_k_value_l292_29279

def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)
def a : ℝ × ℝ := (2 * i.1 + 0 * i.2, 0 * j.1 + 3 * j.2)
def b (k : ℝ) : ℝ × ℝ := (k * i.1 + 0 * i.2, 0 * j.1 - 4 * j.2)

theorem perpendicular_vectors_k_value :
  ∀ k : ℝ, (a.1 * (b k).1 + a.2 * (b k).2 = 0) → k = 6 :=
by sorry

end perpendicular_vectors_k_value_l292_29279


namespace expansion_coefficient_x_squared_l292_29246

/-- The coefficient of x^2 in the expansion of (1 + x + x^(1/2018))^10 -/
def coefficient_x_squared : ℕ :=
  Nat.choose 10 2

theorem expansion_coefficient_x_squared :
  coefficient_x_squared = 45 := by sorry

end expansion_coefficient_x_squared_l292_29246


namespace square_of_sum_twice_x_plus_y_l292_29226

theorem square_of_sum_twice_x_plus_y (x y : ℝ) : (2*x + y)^2 = (2*x + y)^2 := by
  sorry

end square_of_sum_twice_x_plus_y_l292_29226


namespace point_a_coordinates_l292_29240

/-- A point on the x-axis at a distance of 3 units from the origin -/
structure PointA where
  x : ℝ
  y : ℝ
  on_x_axis : y = 0
  distance_from_origin : x^2 + y^2 = 3^2

theorem point_a_coordinates (A : PointA) : (A.x = 3 ∧ A.y = 0) ∨ (A.x = -3 ∧ A.y = 0) := by
  sorry

end point_a_coordinates_l292_29240


namespace symmetry_implies_sum_power_l292_29230

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other. -/
def symmetric_x_axis (P Q : ℝ × ℝ) : Prop :=
  P.1 = Q.1 ∧ P.2 = -Q.2

theorem symmetry_implies_sum_power (a b : ℝ) :
  symmetric_x_axis (a, 3) (4, b) → (a + b)^2021 = 1 := by
  sorry

end symmetry_implies_sum_power_l292_29230


namespace textbook_page_ratio_l292_29234

/-- Proves the ratio of math textbook pages to the sum of history and geography textbook pages -/
theorem textbook_page_ratio : ∀ (history geography math science : ℕ) (total : ℕ),
  history = 160 →
  geography = history + 70 →
  science = 2 * history →
  total = history + geography + math + science →
  total = 905 →
  (math : ℚ) / (history + geography : ℚ) = 1 / 2 := by
  sorry

end textbook_page_ratio_l292_29234


namespace sufficient_not_necessary_l292_29281

theorem sufficient_not_necessary (p q : Prop) :
  (¬(p ∨ q) → ¬p) ∧ ¬(¬p → ¬(p ∨ q)) := by sorry

end sufficient_not_necessary_l292_29281


namespace power_sum_theorem_l292_29223

theorem power_sum_theorem (a : ℝ) (m : ℕ) (h : a^m = 2) : a^(2*m) + a^(3*m) = 12 := by
  sorry

end power_sum_theorem_l292_29223


namespace problem_1_problem_2_problem_3_problem_4_l292_29272

-- Problem 1
theorem problem_1 (m n : ℝ) : 2 * m * n^2 * (1/4 * m * n) = 1/2 * m^2 * n^3 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : (2 * a^3 * b^2 + a^2 * b) / (a * b) = 2 * a^2 * b + a := by sorry

-- Problem 3
theorem problem_3 (x : ℝ) : (2 * x + 3) * (x - 1) = 2 * x^2 + x - 3 := by sorry

-- Problem 4
theorem problem_4 (x y : ℝ) : (x + y)^2 - 2 * y * (x - y) = x^2 + 3 * y^2 := by sorry

end problem_1_problem_2_problem_3_problem_4_l292_29272


namespace chess_tournament_players_l292_29292

/-- A chess tournament with specific point distribution rules -/
structure ChessTournament where
  n : ℕ  -- Number of players not in the lowest-scoring group
  total_players : ℕ := n + 15
  
  -- Each player plays exactly one game against every other player
  games_played : ℕ := total_players * (total_players - 1) / 2
  
  -- Points from games between non-lowest scoring players
  points_upper : ℕ := n * (n - 1) / 2
  
  -- Points from games within lowest scoring group
  points_lower : ℕ := 105
  
  -- Conditions on point distribution
  point_distribution : Prop := 
    2 * points_upper + 2 * points_lower = games_played

/-- The theorem stating that the total number of players is 36 -/
theorem chess_tournament_players : 
  ∀ t : ChessTournament, t.total_players = 36 := by
  sorry

end chess_tournament_players_l292_29292


namespace min_type1_figures_l292_29211

/-- The side length of the equilateral triangle T -/
def side_length : ℕ := 2022

/-- The total number of unit triangles in T -/
def total_triangles : ℕ := side_length * (side_length + 1) / 2

/-- The number of upward-pointing unit triangles in T -/
def upward_triangles : ℕ := (total_triangles + side_length) / 2

/-- The number of downward-pointing unit triangles in T -/
def downward_triangles : ℕ := (total_triangles - side_length) / 2

/-- The excess of upward-pointing unit triangles -/
def excess_upward : ℕ := upward_triangles - downward_triangles

/-- A figure consisting of 4 equilateral unit triangles -/
inductive Figure
| Type1 : Figure  -- Has an excess of ±2 upward-pointing unit triangles
| Type2 : Figure  -- Has equal number of upward and downward-pointing unit triangles
| Type3 : Figure  -- Has equal number of upward and downward-pointing unit triangles
| Type4 : Figure  -- Has equal number of upward and downward-pointing unit triangles

/-- A covering of the triangle T with figures -/
def Covering := List Figure

/-- Predicate to check if a covering is valid -/
def is_valid_covering (c : Covering) : Prop := sorry

/-- The number of Type1 figures in a covering -/
def count_type1 (c : Covering) : ℕ := sorry

theorem min_type1_figures :
  ∃ (c : Covering), is_valid_covering c ∧
  count_type1 c = 1011 ∧
  ∀ (c' : Covering), is_valid_covering c' → count_type1 c' ≥ 1011 := by sorry

end min_type1_figures_l292_29211


namespace pizza_distribution_l292_29289

theorem pizza_distribution (total_pizzas : ℕ) (slices_per_pizza : ℕ) (num_students : ℕ)
  (leftover_cheese : ℕ) (leftover_onion : ℕ) (onion_per_student : ℕ)
  (h1 : total_pizzas = 6)
  (h2 : slices_per_pizza = 18)
  (h3 : num_students = 32)
  (h4 : leftover_cheese = 8)
  (h5 : leftover_onion = 4)
  (h6 : onion_per_student = 1) :
  (total_pizzas * slices_per_pizza - leftover_cheese - leftover_onion - num_students * onion_per_student) / num_students = 2 := by
  sorry

#check pizza_distribution

end pizza_distribution_l292_29289


namespace two_black_cards_selection_l292_29260

/-- The number of cards in each suit of a standard deck -/
def cards_per_suit : ℕ := 13

/-- The number of black suits in a standard deck -/
def black_suits : ℕ := 2

/-- The total number of black cards in a standard deck -/
def total_black_cards : ℕ := black_suits * cards_per_suit

/-- The number of ways to select two different black cards from a standard deck, where order matters -/
def ways_to_select_two_black_cards : ℕ := total_black_cards * (total_black_cards - 1)

theorem two_black_cards_selection :
  ways_to_select_two_black_cards = 650 := by
  sorry

end two_black_cards_selection_l292_29260


namespace remaining_money_l292_29221

-- Define the plant sales
def orchid_sales : ℕ := 30
def orchid_price : ℕ := 50
def money_plant_sales : ℕ := 25
def money_plant_price : ℕ := 30
def bonsai_sales : ℕ := 15
def bonsai_price : ℕ := 75
def cacti_sales : ℕ := 20
def cacti_price : ℕ := 20

-- Define the expenses
def num_workers : ℕ := 4
def worker_pay : ℕ := 60
def new_pots_cost : ℕ := 250
def utility_bill : ℕ := 200
def tax : ℕ := 500

-- Calculate total earnings
def total_earnings : ℕ := 
  orchid_sales * orchid_price + 
  money_plant_sales * money_plant_price + 
  bonsai_sales * bonsai_price + 
  cacti_sales * cacti_price

-- Calculate total expenses
def total_expenses : ℕ := 
  num_workers * worker_pay + 
  new_pots_cost + 
  utility_bill + 
  tax

-- Theorem to prove
theorem remaining_money : 
  total_earnings - total_expenses = 2585 := by
  sorry

end remaining_money_l292_29221


namespace roots_sum_magnitude_l292_29215

theorem roots_sum_magnitude (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ →
  r₁^2 + p*r₁ + 18 = 0 →
  r₂^2 + p*r₂ + 18 = 0 →
  |r₁ + r₂| > 6 := by
sorry

end roots_sum_magnitude_l292_29215


namespace circle_center_is_three_halves_thirty_seven_fourths_l292_29270

/-- A circle passes through (0, 9) and is tangent to y = x^2 at (3, 9) -/
def CircleTangentToParabola (center : ℝ × ℝ) : Prop :=
  let (a, b) := center
  -- Circle passes through (0, 9)
  (a^2 + (b - 9)^2 = a^2 + (b - 9)^2) ∧
  -- Circle is tangent to y = x^2 at (3, 9)
  ((a - 3)^2 + (b - 9)^2 = (a - 0)^2 + (b - 9)^2) ∧
  -- Tangent line to parabola at (3, 9) is perpendicular to line from (3, 9) to center
  ((b - 9) / (a - 3) = -1 / (2 * 3))

/-- The center of the circle is (3/2, 37/4) -/
theorem circle_center_is_three_halves_thirty_seven_fourths :
  CircleTangentToParabola (3/2, 37/4) :=
by sorry

end circle_center_is_three_halves_thirty_seven_fourths_l292_29270


namespace loot_box_average_loss_l292_29224

/-- Represents the expected value calculation for a loot box system -/
def loot_box_expected_value (standard_value : ℝ) (rare_a_prob : ℝ) (rare_a_value : ℝ)
  (rare_b_prob : ℝ) (rare_b_value : ℝ) (rare_c_prob : ℝ) (rare_c_value : ℝ) : ℝ :=
  let standard_prob := 1 - (rare_a_prob + rare_b_prob + rare_c_prob)
  standard_prob * standard_value + rare_a_prob * rare_a_value +
  rare_b_prob * rare_b_value + rare_c_prob * rare_c_value

/-- Calculates the average loss per loot box -/
def average_loss_per_loot_box (box_cost : ℝ) (expected_value : ℝ) : ℝ :=
  box_cost - expected_value

/-- Theorem stating the average loss per loot box in the given scenario -/
theorem loot_box_average_loss :
  let box_cost : ℝ := 5
  let standard_value : ℝ := 3.5
  let rare_a_prob : ℝ := 0.05
  let rare_a_value : ℝ := 10
  let rare_b_prob : ℝ := 0.03
  let rare_b_value : ℝ := 15
  let rare_c_prob : ℝ := 0.02
  let rare_c_value : ℝ := 20
  let expected_value := loot_box_expected_value standard_value rare_a_prob rare_a_value
    rare_b_prob rare_b_value rare_c_prob rare_c_value
  average_loss_per_loot_box box_cost expected_value = 0.5 := by
  sorry

end loot_box_average_loss_l292_29224


namespace expression_evaluation_l292_29263

theorem expression_evaluation : (2^10 * 3^3) / (6 * 2^5) = 144 := by
  sorry

end expression_evaluation_l292_29263


namespace exam_results_l292_29280

theorem exam_results (total : ℕ) (failed_hindi : ℕ) (failed_english : ℕ) (failed_both : ℕ)
  (h1 : failed_hindi = total / 4)
  (h2 : failed_english = total / 2)
  (h3 : failed_both = total / 4)
  : (total - (failed_hindi + failed_english - failed_both)) = total / 2 := by
  sorry

end exam_results_l292_29280


namespace sqrt_sum_squared_l292_29261

theorem sqrt_sum_squared (x : ℝ) :
  (Real.sqrt (10 + x) + Real.sqrt (30 - x) = 8) →
  ((10 + x) * (30 - x) = 144) :=
by sorry

end sqrt_sum_squared_l292_29261


namespace crimson_valley_skirts_l292_29235

theorem crimson_valley_skirts 
  (azure_skirts : ℕ) 
  (seafoam_skirts : ℕ) 
  (purple_skirts : ℕ) 
  (crimson_skirts : ℕ) 
  (h1 : azure_skirts = 90)
  (h2 : seafoam_skirts = 2 * azure_skirts / 3)
  (h3 : purple_skirts = seafoam_skirts / 4)
  (h4 : crimson_skirts = purple_skirts / 3) :
  crimson_skirts = 5 := by
  sorry

end crimson_valley_skirts_l292_29235


namespace f_less_than_g_max_l292_29254

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (2*a + 1) * x + 2 * log x

def g (x : ℝ) : ℝ := x^2 - 2*x

theorem f_less_than_g_max (a : ℝ) :
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Ioo 0 2, f a x₁ < g x₂) →
  a > log 2 - 1 := by
sorry

end f_less_than_g_max_l292_29254


namespace closest_integer_to_two_plus_sqrt_six_l292_29212

theorem closest_integer_to_two_plus_sqrt_six (x : ℝ) : 
  x = 2 + Real.sqrt 6 → 
  ∃ (n : ℕ), n = 4 ∧ ∀ (m : ℕ), m ≠ 4 → |x - ↑n| < |x - ↑m| := by
  sorry

end closest_integer_to_two_plus_sqrt_six_l292_29212


namespace intersection_theorem_l292_29262

def P : Set ℝ := {x | (x - 1)^2 > 16}
def Q (a : ℝ) : Set ℝ := {x | x^2 + (a - 8) * x - 8 * a ≤ 0}

theorem intersection_theorem (a : ℝ) :
  (∃ a, a = 3 → P ∩ Q a = {x | 5 < x ∧ x ≤ 8}) ∧
  (P ∩ Q a = {x | 5 < x ∧ x ≤ 8} ↔ a ∈ Set.Icc (-5) 3) ∧
  (∀ a,
    (a > 3 → P ∩ Q a = {x | -a ≤ x ∧ x < -3 ∨ 5 < x ∧ x ≤ 8}) ∧
    (-5 ≤ a ∧ a ≤ 3 → P ∩ Q a = {x | 5 < x ∧ x ≤ 8}) ∧
    (-8 ≤ a ∧ a < -5 → P ∩ Q a = {x | -a ≤ x ∧ x ≤ 8}) ∧
    (a < -8 → P ∩ Q a = {x | 8 ≤ x ∧ x ≤ -a})) :=
sorry

end intersection_theorem_l292_29262


namespace second_next_perfect_square_l292_29205

theorem second_next_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ m : ℕ, m^2 = x + 4 * (x : ℝ).sqrt + 4 :=
sorry

end second_next_perfect_square_l292_29205


namespace base7_to_base10_54231_l292_29266

/-- Converts a base 7 number to base 10 -/
def base7_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- The base 7 representation of the number -/
def base7_num : List Nat := [1, 3, 2, 4, 5]

theorem base7_to_base10_54231 :
  base7_to_base10 base7_num = 13497 := by sorry

end base7_to_base10_54231_l292_29266


namespace kabadi_kho_kho_intersection_no_players_in_both_games_l292_29250

/-- Proves that the number of people playing both kabadi and kho kho is 0 -/
theorem kabadi_kho_kho_intersection (total : ℕ) (kabadi : ℕ) (kho_kho_only : ℕ)
  (h_total : total = 30)
  (h_kabadi : kabadi = 10)
  (h_kho_kho_only : kho_kho_only = 20) :
  total = kabadi + kho_kho_only :=
by sorry

/-- The number of people playing both kabadi and kho kho -/
def both_games (total : ℕ) (kabadi : ℕ) (kho_kho_only : ℕ) : ℕ :=
  kabadi - (total - kho_kho_only)

theorem no_players_in_both_games (total : ℕ) (kabadi : ℕ) (kho_kho_only : ℕ)
  (h_total : total = 30)
  (h_kabadi : kabadi = 10)
  (h_kho_kho_only : kho_kho_only = 20) :
  both_games total kabadi kho_kho_only = 0 :=
by sorry

end kabadi_kho_kho_intersection_no_players_in_both_games_l292_29250


namespace circle_diameter_from_area_l292_29287

/-- Given a circle with area π/4 square units, its diameter is 1 unit. -/
theorem circle_diameter_from_area :
  ∀ (r : ℝ), π * r^2 = π / 4 → 2 * r = 1 := by
  sorry

end circle_diameter_from_area_l292_29287


namespace gear_speed_proportion_l292_29200

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Represents a system of four meshed gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear
  x : ℕ
  y : ℕ
  z : ℕ
  w : ℕ
  mesh_correctly : A.teeth * A.speed = B.teeth * B.speed ∧
                   B.teeth * B.speed = C.teeth * C.speed ∧
                   C.teeth * C.speed = D.teeth * D.speed

/-- Theorem stating the proportion of angular speeds in a gear system -/
theorem gear_speed_proportion (gs : GearSystem)
  (hA : gs.A.teeth = 10 * gs.x)
  (hB : gs.B.teeth = 15 * gs.y)
  (hC : gs.C.teeth = 12 * gs.z)
  (hD : gs.D.teeth = 20 * gs.w) :
  ∃ (k : ℝ), k > 0 ∧
    gs.A.speed = k * (12 * gs.y * gs.z * gs.w : ℝ) ∧
    gs.B.speed = k * (8 * gs.x * gs.z * gs.w : ℝ) ∧
    gs.C.speed = k * (10 * gs.x * gs.y * gs.w : ℝ) ∧
    gs.D.speed = k * (6 * gs.x * gs.y * gs.z : ℝ) :=
sorry

end gear_speed_proportion_l292_29200


namespace min_button_presses_to_escape_l292_29217

/-- Represents the state of the room with doors and mines -/
structure RoomState where
  armed_mines : ℕ
  closed_doors : ℕ

/-- Represents the actions of pressing buttons -/
structure ButtonPresses where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the final state of the room after pressing buttons -/
def final_state (initial : RoomState) (presses : ButtonPresses) : RoomState :=
  { armed_mines := initial.armed_mines + presses.red - 2 * presses.yellow,
    closed_doors := initial.closed_doors + presses.yellow - 2 * presses.green }

/-- Checks if all mines are disarmed and all doors are open -/
def is_solved (state : RoomState) : Prop :=
  state.armed_mines = 0 ∧ state.closed_doors = 0

/-- The main theorem to prove -/
theorem min_button_presses_to_escape : 
  ∃ (presses : ButtonPresses),
    is_solved (final_state { armed_mines := 3, closed_doors := 3 } presses) ∧
    presses.red + presses.yellow + presses.green = 9 ∧
    ∀ (other_presses : ButtonPresses),
      is_solved (final_state { armed_mines := 3, closed_doors := 3 } other_presses) →
      other_presses.red + other_presses.yellow + other_presses.green ≥ 9 :=
by sorry

end min_button_presses_to_escape_l292_29217


namespace committee_permutations_count_l292_29251

/-- The number of distinct permutations of the letters in "COMMITTEE" -/
def committee_permutations : ℕ := sorry

/-- The total number of letters in "COMMITTEE" -/
def total_letters : ℕ := 8

/-- The number of occurrences of each letter in "COMMITTEE" -/
def letter_occurrences : List ℕ := [2, 2, 3, 1, 1]

theorem committee_permutations_count : 
  committee_permutations = (total_letters.factorial) / (letter_occurrences.map Nat.factorial).prod := by
  sorry

end committee_permutations_count_l292_29251


namespace bulb_probability_l292_29295

/-- The probability that a bulb from factory X works for over 4000 hours -/
def prob_x : ℝ := 0.59

/-- The probability that a bulb from factory Y works for over 4000 hours -/
def prob_y : ℝ := 0.65

/-- The probability that a bulb from factory Z works for over 4000 hours -/
def prob_z : ℝ := 0.70

/-- The proportion of bulbs supplied by factory X -/
def supply_x : ℝ := 0.5

/-- The proportion of bulbs supplied by factory Y -/
def supply_y : ℝ := 0.3

/-- The proportion of bulbs supplied by factory Z -/
def supply_z : ℝ := 0.2

/-- The overall probability that a randomly selected bulb will work for over 4000 hours -/
def overall_prob : ℝ := supply_x * prob_x + supply_y * prob_y + supply_z * prob_z

theorem bulb_probability : overall_prob = 0.63 := by sorry

end bulb_probability_l292_29295


namespace cubic_fraction_zero_l292_29256

theorem cubic_fraction_zero (a b c : ℝ) (h : a + b + c ≠ 0) :
  ((a^3 - b^3)^2 + (b^3 - c^3)^2 + (c^3 - a^3)^2) / (a^3 + b^3 + c^3 - 3*a*b*c) = 0 :=
by sorry

end cubic_fraction_zero_l292_29256


namespace partnership_investment_l292_29213

/-- A partnership business problem -/
theorem partnership_investment (b_investment c_investment c_profit total_profit : ℕ) 
  (hb : b_investment = 72000)
  (hc : c_investment = 81000)
  (hcp : c_profit = 36000)
  (htp : total_profit = 80000) :
  ∃ a_investment : ℕ, 
    (c_profit : ℚ) / (total_profit : ℚ) = (c_investment : ℚ) / ((a_investment : ℚ) + (b_investment : ℚ) + (c_investment : ℚ)) ∧ 
    a_investment = 27000 := by
  sorry

end partnership_investment_l292_29213


namespace original_number_proof_l292_29268

theorem original_number_proof (x : ℝ) : ((x - 3) / 6) * 12 = 8 → x = 7 := by
  sorry

end original_number_proof_l292_29268


namespace three_statements_true_l292_29232

open Function

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def isPeriodic (f : ℝ → ℝ) : Prop := ∃ T ≠ 0, ∀ x, f (x + T) = f x
def isMonoDecreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y
def hasInverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, LeftInverse g f ∧ RightInverse g f

-- The main theorem
theorem three_statements_true (f : ℝ → ℝ) : 
  (isOdd f → isOdd (f ∘ f)) ∧
  (isPeriodic f → isPeriodic (f ∘ f)) ∧
  ¬(isMonoDecreasing f → isMonoDecreasing (f ∘ f)) ∧
  (hasInverse f → (∃ x, f x = x)) :=
sorry

end three_statements_true_l292_29232


namespace yearly_salary_calculation_l292_29206

/-- Proves that the yearly salary excluding turban is 160 rupees given the problem conditions --/
theorem yearly_salary_calculation (partial_payment : ℕ) (turban_value : ℕ) (months_worked : ℕ) (total_months : ℕ) :
  partial_payment = 50 →
  turban_value = 70 →
  months_worked = 9 →
  total_months = 12 →
  (partial_payment + turban_value : ℚ) / (months_worked : ℚ) * (total_months : ℚ) = 160 := by
sorry

end yearly_salary_calculation_l292_29206
