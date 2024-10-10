import Mathlib

namespace root_sum_product_l1122_112243

theorem root_sum_product (c d : ℝ) : 
  (c^4 - 6*c^3 - 4*c - 1 = 0) → 
  (d^4 - 6*d^3 - 4*d - 1 = 0) → 
  (c ≠ d) →
  (cd + c + d = 4) := by
sorry

end root_sum_product_l1122_112243


namespace ellipse_equation_proof_l1122_112248

/-- An ellipse with given properties -/
structure Ellipse where
  /-- First focus of the ellipse -/
  F₁ : ℝ × ℝ
  /-- Second focus of the ellipse -/
  F₂ : ℝ × ℝ
  /-- Length of the chord AB passing through F₂ and perpendicular to x-axis -/
  AB_length : ℝ
  /-- The first focus is at (-1, 0) -/
  F₁_constraint : F₁ = (-1, 0)
  /-- The second focus is at (1, 0) -/
  F₂_constraint : F₂ = (1, 0)
  /-- The length of AB is 3 -/
  AB_length_constraint : AB_length = 3

/-- The equation of the ellipse -/
def ellipse_equation (C : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- Theorem stating that the given ellipse satisfies the equation x²/4 + y²/3 = 1 -/
theorem ellipse_equation_proof (C : Ellipse) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | ellipse_equation C p.1 p.2} ↔ 
  (x, y) ∈ {p : ℝ × ℝ | ∃ t : ℝ, p = (C.F₂.1, t) ∧ 
  abs (2 * t) = C.AB_length ∧ 
  (p.1 - C.F₁.1)^2 + p.2^2 + (p.1 - C.F₂.1)^2 + p.2^2 = 
  ((p.1 - C.F₁.1)^2 + p.2^2)^(1/2) + ((p.1 - C.F₂.1)^2 + p.2^2)^(1/2)} := by
  sorry

end ellipse_equation_proof_l1122_112248


namespace t_of_f_6_l1122_112274

noncomputable def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 2)

noncomputable def f (x : ℝ) : ℝ := 6 - t x

theorem t_of_f_6 : t (f 6) = Real.sqrt 26 - 2 := by
  sorry

end t_of_f_6_l1122_112274


namespace tammy_climbing_speed_l1122_112273

/-- Tammy's mountain climbing problem -/
theorem tammy_climbing_speed :
  ∀ (v : ℝ), -- v represents the speed on the first day
  v > 0 →
  v * 7 + (v + 0.5) * 5 + (v + 1.5) * 8 = 85 →
  7 + 5 + 8 = 20 →
  (v + 0.5) = 4.025 :=
by
  sorry

end tammy_climbing_speed_l1122_112273


namespace sufficient_but_not_necessary_l1122_112265

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  (∃ b : ℝ, b ≠ 2 ∧ (b - 1) * (b - 2) = 0) := by
  sorry

end sufficient_but_not_necessary_l1122_112265


namespace simon_received_stamps_l1122_112252

/-- The number of stamps Simon received from his friends -/
def stamps_received (initial_stamps current_stamps : ℕ) : ℕ :=
  current_stamps - initial_stamps

/-- Theorem stating that Simon received 27 stamps from his friends -/
theorem simon_received_stamps :
  stamps_received 34 61 = 27 := by
  sorry

end simon_received_stamps_l1122_112252


namespace edge_sum_is_144_l1122_112287

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  -- The three dimensions of the solid
  a : ℝ
  b : ℝ
  c : ℝ
  -- Volume is 432 cm³
  volume_eq : a * b * c = 432
  -- Surface area is 432 cm²
  surface_area_eq : 2 * (a * b + b * c + a * c) = 432
  -- Dimensions are in geometric progression
  geometric_progression : b * b = a * c

/-- The sum of the lengths of all edges of the rectangular solid is 144 cm -/
theorem edge_sum_is_144 (solid : RectangularSolid) :
  4 * (solid.a + solid.b + solid.c) = 144 := by
  sorry

end edge_sum_is_144_l1122_112287


namespace jacqueline_guavas_l1122_112217

theorem jacqueline_guavas (plums apples given_away left : ℕ) (guavas : ℕ) : 
  plums = 16 → 
  apples = 21 → 
  given_away = 40 → 
  left = 15 → 
  plums + guavas + apples = given_away + left → 
  guavas = 18 :=
by
  sorry

end jacqueline_guavas_l1122_112217


namespace relay_race_total_time_l1122_112205

/-- The total time for a relay race with four athletes -/
def relay_race_time (athlete1_time athlete2_extra athlete3_less athlete4_less : ℕ) : ℕ :=
  let athlete2_time := athlete1_time + athlete2_extra
  let athlete3_time := athlete2_time - athlete3_less
  let athlete4_time := athlete1_time - athlete4_less
  athlete1_time + athlete2_time + athlete3_time + athlete4_time

/-- Theorem stating that the total time for the given relay race is 200 seconds -/
theorem relay_race_total_time :
  relay_race_time 55 10 15 25 = 200 := by
  sorry

end relay_race_total_time_l1122_112205


namespace esports_gender_related_prob_select_male_expected_like_esports_l1122_112231

-- Define the survey data
def total_students : ℕ := 400
def male_like : ℕ := 120
def male_dislike : ℕ := 80
def female_like : ℕ := 100
def female_dislike : ℕ := 100

-- Define the critical value for α = 0.05
def critical_value : ℚ := 3841/1000

-- Define the chi-square statistic function
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Theorem 1: Chi-square statistic is greater than critical value
theorem esports_gender_related :
  chi_square male_like male_dislike female_like female_dislike > critical_value := by
  sorry

-- Theorem 2: Probability of selecting at least one male student
theorem prob_select_male :
  1 - (Nat.choose 5 3 : ℚ) / (Nat.choose 9 3) = 37/42 := by
  sorry

-- Theorem 3: Expected number of students who like esports
theorem expected_like_esports :
  (10 : ℚ) * (male_like + female_like) / total_students = 11/2 := by
  sorry

end esports_gender_related_prob_select_male_expected_like_esports_l1122_112231


namespace polynomial_factors_l1122_112229

theorem polynomial_factors (x : ℝ) : 
  ∃ (a b c : ℝ), 8*x^3 + 14*x^2 - 17*x + 6 = (x + 1/2) * (x - 2) * (a*x + b) ∧ c ≠ 0 := by
  sorry

end polynomial_factors_l1122_112229


namespace set_B_equals_l1122_112270

def A : Set Int := {-2, -1, 1, 2, 3, 4}

def B : Set Int := {x | ∃ t ∈ A, x = t^2}

theorem set_B_equals : B = {1, 4, 9, 16} := by sorry

end set_B_equals_l1122_112270


namespace sum_of_x_and_y_is_four_l1122_112232

/-- Two-digit number represented as a pair of digits -/
def TwoDigitNumber := Nat × Nat

/-- Sum of digits of two two-digit numbers -/
def sumOfDigits (n1 n2 : TwoDigitNumber) : Nat :=
  n1.1 + n1.2 + n2.1 + n2.2

/-- Result of adding two two-digit numbers -/
def addTwoDigitNumbers (n1 n2 : TwoDigitNumber) : Nat × Nat × Nat :=
  let sum := n1.1 * 10 + n1.2 + n2.1 * 10 + n2.2
  (sum / 100, (sum / 10) % 10, sum % 10)

theorem sum_of_x_and_y_is_four (n1 n2 : TwoDigitNumber) :
  sumOfDigits n1 n2 = 22 →
  (addTwoDigitNumbers n1 n2).2.2 = 9 →
  (addTwoDigitNumbers n1 n2).1 + (addTwoDigitNumbers n1 n2).2.1 = 4 := by
  sorry

end sum_of_x_and_y_is_four_l1122_112232


namespace interpolation_polynomial_existence_and_uniqueness_l1122_112213

theorem interpolation_polynomial_existence_and_uniqueness
  (n : ℕ) (x y : Fin n → ℝ) (h : ∀ i j : Fin n, i < j → x i < x j) :
  ∃! f : ℝ → ℝ,
    (∀ i : Fin n, f (x i) = y i) ∧
    ∃ p : Polynomial ℝ, (∀ t, f t = p.eval t) ∧ p.degree < n :=
sorry

end interpolation_polynomial_existence_and_uniqueness_l1122_112213


namespace trig_expression_value_l1122_112277

theorem trig_expression_value (α : Real) 
  (h1 : π/2 < α) 
  (h2 : α < π) 
  (h3 : Real.sin α + Real.cos α = 1/5) : 
  2 / (Real.cos α - Real.sin α) = -10/7 := by
  sorry

end trig_expression_value_l1122_112277


namespace car_costs_theorem_l1122_112233

def cost_of_old_car : ℝ := 1800
def cost_of_second_oldest_car : ℝ := 900
def cost_of_new_car : ℝ := 2 * cost_of_old_car
def sale_price_old_car : ℝ := 1800
def sale_price_second_oldest_car : ℝ := 900
def loan_amount : ℝ := cost_of_new_car - (sale_price_old_car + sale_price_second_oldest_car)
def annual_interest_rate : ℝ := 0.05
def years_passed : ℝ := 2
def remaining_debt : ℝ := 2000

theorem car_costs_theorem :
  cost_of_old_car = 1800 ∧
  cost_of_second_oldest_car = 900 ∧
  cost_of_new_car = 2 * cost_of_old_car ∧
  cost_of_new_car = 4 * cost_of_second_oldest_car ∧
  sale_price_old_car = 1800 ∧
  sale_price_second_oldest_car = 900 ∧
  loan_amount = cost_of_new_car - (sale_price_old_car + sale_price_second_oldest_car) ∧
  remaining_debt = 2000 :=
by sorry

end car_costs_theorem_l1122_112233


namespace best_marksman_score_l1122_112258

theorem best_marksman_score (team_size : ℕ) (hypothetical_score : ℕ) (hypothetical_average : ℕ) (actual_total : ℕ) : 
  team_size = 6 → 
  hypothetical_score = 92 →
  hypothetical_average = 84 →
  actual_total = 497 →
  ∃ (best_score : ℕ), best_score = 85 ∧ 
    actual_total = (team_size - 1) * hypothetical_average + best_score := by
  sorry

end best_marksman_score_l1122_112258


namespace triangle_angles_l1122_112203

/-- Given a triangle with sides a, b, and c, where a = b = 3 and c = √7 - √3,
    prove that the angles of the triangle are as follows:
    - Angle C (opposite side c) = arccos((4 + √21) / 9)
    - Angles A and B = (180° - arccos((4 + √21) / 9)) / 2 -/
theorem triangle_angles (a b c : ℝ) (h1 : a = 3) (h2 : b = 3) (h3 : c = Real.sqrt 7 - Real.sqrt 3) :
  let angle_c := Real.arccos ((4 + Real.sqrt 21) / 9)
  let angle_a := (π - angle_c) / 2
  ∃ (A B C : ℝ),
    A = angle_a ∧
    B = angle_a ∧
    C = angle_c ∧
    A + B + C = π :=
by sorry

end triangle_angles_l1122_112203


namespace log_inequality_l1122_112255

theorem log_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) :
  Real.log (Real.sqrt (x₁ * x₂)) = (Real.log x₁ + Real.log x₂) / 2 ∧
  Real.log (Real.sqrt (x₁ * x₂)) < Real.log ((x₁ + x₂) / 2) := by
  sorry

end log_inequality_l1122_112255


namespace spheres_in_base_of_pyramid_l1122_112236

/-- The number of spheres in a regular triangular pyramid with n levels -/
def triangular_pyramid (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem spheres_in_base_of_pyramid (total_spheres : ℕ) (h : total_spheres = 165) :
  ∃ n : ℕ, triangular_pyramid n = total_spheres ∧ triangular_number n = 45 := by
  sorry

end spheres_in_base_of_pyramid_l1122_112236


namespace joe_initial_cars_l1122_112271

/-- The number of cars Joe will have after getting more cars -/
def total_cars : ℕ := 62

/-- The number of additional cars Joe will get -/
def additional_cars : ℕ := 12

/-- Theorem: Joe's initial number of cars is 50 -/
theorem joe_initial_cars : 
  total_cars - additional_cars = 50 := by
  sorry

end joe_initial_cars_l1122_112271


namespace hockey_league_games_l1122_112227

/-- The number of games played in a hockey league season -/
def number_of_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

/-- Theorem: In a hockey league with 19 teams, where each team faces every other team 10 times, 
    the total number of games played in the season is 1710 -/
theorem hockey_league_games : number_of_games 19 10 = 1710 := by
  sorry

end hockey_league_games_l1122_112227


namespace candy_sales_average_l1122_112292

/-- The average of candy sales for five months -/
def average_candy_sales (jan feb mar apr may : ℕ) : ℚ :=
  (jan + feb + mar + apr + may) / 5

/-- Theorem stating that the average candy sales is 96 dollars -/
theorem candy_sales_average :
  average_candy_sales 110 80 70 130 90 = 96 := by sorry

end candy_sales_average_l1122_112292


namespace journey_distance_l1122_112280

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 5)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  ∃ (distance : ℝ), 
    distance / (2 * speed1) + distance / (2 * speed2) = total_time ∧ 
    distance = 112 := by
  sorry

end journey_distance_l1122_112280


namespace count_flippy_divisible_by_18_l1122_112276

def is_flippy (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ 
    n = a * 100000 + b * 10000 + a * 1000 + b * 100 + a * 10 + b

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

theorem count_flippy_divisible_by_18 :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, is_flippy n ∧ is_six_digit n ∧ n % 18 = 0) ∧
    s.card = 4 := by
  sorry

end count_flippy_divisible_by_18_l1122_112276


namespace unique_prime_product_l1122_112220

theorem unique_prime_product (p q r : Nat) : 
  Prime p ∧ Prime q ∧ Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  p * q * r = 7802 ∧
  p + q + r = 1306 →
  ∀ (p1 p2 p3 : Nat), 
    Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ 
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    p1 * p2 * p3 ≠ 7802 ∧
    p1 + p2 + p3 = 1306 →
    False :=
by sorry

#check unique_prime_product

end unique_prime_product_l1122_112220


namespace odometer_sum_squares_l1122_112289

/-- Represents the odometer reading as a triple of natural numbers -/
structure OdometerReading where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : b ≥ 1
  h2 : a + b + c ≤ 9

/-- Represents Liam's car trip -/
structure CarTrip where
  speed : ℕ
  hours : ℕ
  initial : OdometerReading
  final : OdometerReading
  h1 : speed = 60
  h2 : final.a = initial.b
  h3 : final.b = initial.c
  h4 : final.c = initial.a
  h5 : 100 * final.b + 10 * final.c + final.a - (100 * initial.a + 10 * initial.b + initial.c) = speed * hours

theorem odometer_sum_squares (trip : CarTrip) : 
  trip.initial.a^2 + trip.initial.b^2 + trip.initial.c^2 = 29 := by
  sorry

end odometer_sum_squares_l1122_112289


namespace landscape_breadth_l1122_112214

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  length : ℝ
  breadth : ℝ
  playground_area : ℝ

/-- The breadth is 6 times the length -/
def breadth_length_relation (l : Landscape) : Prop :=
  l.breadth = 6 * l.length

/-- The playground occupies 1/7th of the total landscape area -/
def playground_proportion (l : Landscape) : Prop :=
  l.playground_area = (1 / 7) * l.length * l.breadth

/-- The playground area is 4200 square meters -/
def playground_area_value (l : Landscape) : Prop :=
  l.playground_area = 4200

/-- Theorem: The breadth of the landscape is 420 meters -/
theorem landscape_breadth (l : Landscape) 
  (h1 : breadth_length_relation l)
  (h2 : playground_proportion l)
  (h3 : playground_area_value l) : 
  l.breadth = 420 := by sorry

end landscape_breadth_l1122_112214


namespace parabola_translation_l1122_112244

/-- Given two parabolas, prove that one is a translation of the other -/
theorem parabola_translation (x : ℝ) :
  (x^2 + 4*x + 5) = ((x + 2)^2 + 1) := by sorry

end parabola_translation_l1122_112244


namespace units_digit_of_5_to_4_l1122_112228

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_5_to_4 : unitsDigit (5^4) = 5 := by
  sorry

end units_digit_of_5_to_4_l1122_112228


namespace enemies_left_undefeated_video_game_enemies_l1122_112210

theorem enemies_left_undefeated 
  (points_per_enemy : ℕ) 
  (total_enemies : ℕ) 
  (points_earned : ℕ) : ℕ :=
  let enemies_defeated := points_earned / points_per_enemy
  total_enemies - enemies_defeated

theorem video_game_enemies 
  (points_per_enemy : ℕ) 
  (total_enemies : ℕ) 
  (points_earned : ℕ) 
  (h1 : points_per_enemy = 8) 
  (h2 : total_enemies = 7) 
  (h3 : points_earned = 40) : 
  enemies_left_undefeated points_per_enemy total_enemies points_earned = 2 := by
  sorry

end enemies_left_undefeated_video_game_enemies_l1122_112210


namespace solution_to_system_l1122_112279

theorem solution_to_system :
  ∃ (x y : ℝ), x + y = (6 - x) + (6 - y) ∧ x - y = (x - 2) + (y - 2) ∧ x = 2 ∧ y = 4 := by
  sorry

end solution_to_system_l1122_112279


namespace max_value_of_sum_products_l1122_112295

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a + b + c + d = 200 → 
  a * b + b * c + c * d ≤ 10000 ∧ 
  ∃ a b c d, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
             a + b + c + d = 200 ∧ 
             a * b + b * c + c * d = 10000 :=
by sorry

end max_value_of_sum_products_l1122_112295


namespace total_travel_time_l1122_112266

def distance_washington_idaho : ℝ := 640
def distance_idaho_nevada : ℝ := 550
def speed_washington_idaho : ℝ := 80
def speed_idaho_nevada : ℝ := 50

theorem total_travel_time :
  (distance_washington_idaho / speed_washington_idaho) +
  (distance_idaho_nevada / speed_idaho_nevada) = 19 := by
  sorry

end total_travel_time_l1122_112266


namespace expected_draws_for_specific_box_l1122_112298

/-- A box containing red and white balls -/
structure Box where
  red : ℕ
  white : ℕ

/-- The expected number of draws needed to pick a white ball -/
def expectedDraws (b : Box) : ℚ :=
  -- Definition to be proved
  11/9

/-- Theorem stating the expected number of draws for a specific box configuration -/
theorem expected_draws_for_specific_box :
  let b : Box := ⟨2, 8⟩
  expectedDraws b = 11/9 := by
  sorry


end expected_draws_for_specific_box_l1122_112298


namespace max_distance_for_given_tires_l1122_112275

/-- Represents the maximum distance a car can travel by switching tires -/
def max_distance (front_tire_life : ℕ) (rear_tire_life : ℕ) : ℕ :=
  let switch_point := front_tire_life / 2
  switch_point + min (front_tire_life - switch_point) (rear_tire_life - switch_point)

/-- Theorem stating the maximum distance a car can travel with given tire lives -/
theorem max_distance_for_given_tires :
  max_distance 21000 28000 = 24000 :=
by sorry

end max_distance_for_given_tires_l1122_112275


namespace sum_of_factorials_perfect_square_mod_5_l1122_112288

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def is_perfect_square_mod_5 (n : ℕ) : Prop :=
  ∃ m : ℕ, n ≡ m^2 [ZMOD 5]

theorem sum_of_factorials_perfect_square_mod_5 (n : ℕ+) :
  is_perfect_square_mod_5 (sum_of_factorials n) ↔ n = 1 ∨ n = 3 := by
  sorry

end sum_of_factorials_perfect_square_mod_5_l1122_112288


namespace tau_phi_sum_equation_l1122_112237

/-- τ(n) represents the number of positive divisors of n -/
def tau (n : ℕ) : ℕ := sorry

/-- φ(n) represents the number of positive integers less than n and relatively prime to n -/
def phi (n : ℕ) : ℕ := sorry

/-- A predicate to check if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

theorem tau_phi_sum_equation (n : ℕ) (h : n > 1) :
  tau n + phi n = n + 1 ↔ n = 4 ∨ isPrime n := by sorry

end tau_phi_sum_equation_l1122_112237


namespace three_by_five_rectangle_triangles_l1122_112296

/-- Represents a rectangle divided into a grid with diagonal lines. -/
structure GridRectangle where
  horizontal_divisions : Nat
  vertical_divisions : Nat

/-- Counts the number of triangles in a GridRectangle. -/
def count_triangles (rect : GridRectangle) : Nat :=
  sorry

/-- Theorem stating that a 3x5 GridRectangle contains 76 triangles. -/
theorem three_by_five_rectangle_triangles :
  count_triangles ⟨3, 5⟩ = 76 := by
  sorry

end three_by_five_rectangle_triangles_l1122_112296


namespace cubic_function_m_value_l1122_112202

theorem cubic_function_m_value (d e f g m : ℤ) :
  let g : ℝ → ℝ := λ x => (d : ℝ) * x^3 + (e : ℝ) * x^2 + (f : ℝ) * x + (g : ℝ)
  g 1 = 0 ∧
  70 < g 5 ∧ g 5 < 80 ∧
  120 < g 6 ∧ g 6 < 130 ∧
  10000 * (m : ℝ) < g 50 ∧ g 50 < 10000 * ((m + 1) : ℝ) →
  m = 12 := by
sorry

end cubic_function_m_value_l1122_112202


namespace same_row_both_shows_l1122_112224

/-- Represents a seating arrangement for a show -/
def SeatingArrangement := Fin 50 → Fin 7

/-- The number of rows in the cinema -/
def num_rows : Nat := 7

/-- The number of children attending the shows -/
def num_children : Nat := 50

/-- Theorem: There exist at least two children who sat in the same row during both shows -/
theorem same_row_both_shows (morning_seating evening_seating : SeatingArrangement) :
  ∃ (i j : Fin 50), i ≠ j ∧
    morning_seating i = morning_seating j ∧
    evening_seating i = evening_seating j :=
sorry

end same_row_both_shows_l1122_112224


namespace function_root_implies_a_range_l1122_112225

theorem function_root_implies_a_range (a : ℝ) :
  (∃ x₀ : ℝ, -1 < x₀ ∧ x₀ < 1 ∧ 3 * a * x₀ - 2 * a + 1 = 0) →
  a < -1 ∨ a > 1/5 :=
by sorry

end function_root_implies_a_range_l1122_112225


namespace not_p_sufficient_not_necessary_for_not_q_l1122_112272

-- Define the conditions p and q
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 > x^2

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that ¬p is sufficient but not necessary for ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, not_p x → not_q x) ∧ 
  (∃ x : ℝ, not_q x ∧ ¬(not_p x)) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l1122_112272


namespace farmer_land_ownership_l1122_112263

/-- The total land owned by the farmer in acres -/
def total_land : ℝ := 6000

/-- The proportion of land cleared for planting -/
def cleared_proportion : ℝ := 0.90

/-- The proportion of cleared land planted with soybeans -/
def soybean_proportion : ℝ := 0.30

/-- The proportion of cleared land planted with wheat -/
def wheat_proportion : ℝ := 0.60

/-- The amount of cleared land planted with corn in acres -/
def corn_land : ℝ := 540

theorem farmer_land_ownership :
  total_land * cleared_proportion * (1 - soybean_proportion - wheat_proportion) = corn_land :=
by sorry

end farmer_land_ownership_l1122_112263


namespace residue_of_7_1234_mod_19_l1122_112208

theorem residue_of_7_1234_mod_19 : 7^1234 % 19 = 9 := by
  sorry

end residue_of_7_1234_mod_19_l1122_112208


namespace surface_area_of_circumscribed_sphere_l1122_112223

/-- A regular tetrahedron with edge length √2 -/
structure RegularTetrahedron where
  edgeLength : ℝ
  isRegular : edgeLength = Real.sqrt 2

/-- A sphere circumscribing a regular tetrahedron -/
structure CircumscribedSphere (t : RegularTetrahedron) where
  radius : ℝ
  containsVertices : True  -- This is a placeholder for the condition that all vertices are on the sphere

/-- The surface area of a sphere circumscribing a regular tetrahedron with edge length √2 is 3π -/
theorem surface_area_of_circumscribed_sphere (t : RegularTetrahedron) (s : CircumscribedSphere t) :
  4 * Real.pi * s.radius ^ 2 = 3 * Real.pi :=
sorry

end surface_area_of_circumscribed_sphere_l1122_112223


namespace linear_func_not_in_M_exp_func_in_M_sin_func_in_M_iff_l1122_112204

-- Define the property for a function to be in set M
def in_set_M (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = T * f x

-- Part 1
theorem linear_func_not_in_M : ¬ in_set_M (λ x : ℝ ↦ x) := by sorry

-- Part 2
theorem exp_func_in_M (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  (∃ T : ℝ, T > 0 ∧ a^T = T) → in_set_M (λ x : ℝ ↦ a^x) := by sorry

-- Part 3
theorem sin_func_in_M_iff (k : ℝ) :
  in_set_M (λ x : ℝ ↦ Real.sin (k * x)) ↔ ∃ m : ℤ, k = m * Real.pi := by sorry

end linear_func_not_in_M_exp_func_in_M_sin_func_in_M_iff_l1122_112204


namespace digit_sum_property_l1122_112299

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem digit_sum_property (n : ℕ) 
  (h1 : sum_of_digits n = 100)
  (h2 : sum_of_digits (44 * n) = 800) : 
  sum_of_digits (3 * n) = 300 := by sorry

end digit_sum_property_l1122_112299


namespace marys_nickels_l1122_112268

/-- Given that Mary initially had 7 nickels and now has 12 nickels,
    prove that Mary's dad gave her 5 nickels. -/
theorem marys_nickels (initial : ℕ) (final : ℕ) (given : ℕ) :
  initial = 7 → final = 12 → given = final - initial → given = 5 := by
  sorry

end marys_nickels_l1122_112268


namespace modular_inverse_11_mod_1021_l1122_112249

theorem modular_inverse_11_mod_1021 : ∃ x : ℕ, x ∈ Finset.range 1021 ∧ (11 * x) % 1021 = 1 := by
  sorry

end modular_inverse_11_mod_1021_l1122_112249


namespace fraction_simplification_l1122_112284

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) :
  (x^2 - x*y) / ((x - y)^2) = x / (x - y) := by
  sorry

end fraction_simplification_l1122_112284


namespace portfolio_annual_yield_is_correct_l1122_112215

structure Security where
  quantity : ℕ
  initialPrice : ℝ
  priceAfter180Days : ℝ

def Portfolio : List Security := [
  ⟨1000, 95.3, 98.6⟩,
  ⟨1000, 89.5, 93.4⟩,
  ⟨1000, 92.1, 96.2⟩,
  ⟨1, 100000, 104300⟩,
  ⟨1, 200000, 209420⟩,
  ⟨40, 3700, 3900⟩,
  ⟨500, 137, 142⟩
]

def calculateAnnualYield (portfolio : List Security) : ℝ :=
  sorry

theorem portfolio_annual_yield_is_correct :
  abs (calculateAnnualYield Portfolio - 9.21) < 0.01 := by
  sorry

end portfolio_annual_yield_is_correct_l1122_112215


namespace f_inequality_l1122_112218

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

theorem f_inequality : f (Real.sqrt 3 / 2) > f (Real.sqrt 6 / 2) ∧ f (Real.sqrt 6 / 2) > f (Real.sqrt 2 / 2) := by
  sorry

end f_inequality_l1122_112218


namespace passing_marks_l1122_112267

/-- Given an exam with total marks T and passing marks P, prove that P = 240 -/
theorem passing_marks (T : ℝ) (P : ℝ) : 
  (0.30 * T = P - 60) →  -- Condition 1: 30% fails by 60 marks
  (0.45 * T = P + 30) →  -- Condition 2: 45% passes by 30 marks
  P = 240 := by
sorry

end passing_marks_l1122_112267


namespace quadratic_function_properties_l1122_112294

-- Define the quadratic function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 3

-- State the theorem
theorem quadratic_function_properties (a : ℝ) :
  (∀ x : ℝ, f a x = f a (4 - x)) →
  (a = 4) ∧
  (Set.Icc 0 3).image (f a) = Set.Icc (-1) 3 ∧
  ∃ (g : ℝ → ℝ), (∀ x : ℝ, f a x = (x - 2)^2 + 1) :=
by
  sorry


end quadratic_function_properties_l1122_112294


namespace absolute_sum_inequality_l1122_112262

theorem absolute_sum_inequality (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x + 9| > a) → a < 8 := by
  sorry

end absolute_sum_inequality_l1122_112262


namespace largest_integer_product_12_l1122_112247

theorem largest_integer_product_12 (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a * b * c * d * e = 12 →
  max a (max b (max c (max d e))) = 3 := by
sorry

end largest_integer_product_12_l1122_112247


namespace blake_change_l1122_112234

/-- The amount Blake spends on oranges -/
def orange_cost : ℕ := 40

/-- The amount Blake spends on apples -/
def apple_cost : ℕ := 50

/-- The amount Blake spends on mangoes -/
def mango_cost : ℕ := 60

/-- The initial amount Blake has -/
def initial_amount : ℕ := 300

/-- The change given to Blake -/
def change : ℕ := initial_amount - (orange_cost + apple_cost + mango_cost)

theorem blake_change : change = 150 := by
  sorry

end blake_change_l1122_112234


namespace rainville_total_rainfall_2007_l1122_112206

/-- Calculates the total rainfall for a year given the average monthly rainfall -/
def total_rainfall (average_monthly_rainfall : ℝ) : ℝ :=
  average_monthly_rainfall * 12

/-- Represents the rainfall data for Rainville from 2005 to 2007 -/
structure RainvilleRainfall where
  rainfall_2005 : ℝ
  rainfall_increase_2006 : ℝ
  rainfall_increase_2007 : ℝ

/-- Theorem stating the total rainfall in Rainville for 2007 -/
theorem rainville_total_rainfall_2007 (data : RainvilleRainfall) 
  (h1 : data.rainfall_2005 = 50)
  (h2 : data.rainfall_increase_2006 = 3)
  (h3 : data.rainfall_increase_2007 = 5) :
  total_rainfall (data.rainfall_2005 + data.rainfall_increase_2006 + data.rainfall_increase_2007) = 696 :=
by sorry

end rainville_total_rainfall_2007_l1122_112206


namespace spatial_relationships_l1122_112260

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (parallelLines : Line → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem spatial_relationships 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : contains β m) :
  (parallel α β → perpendicularLines l m) ∧ 
  (parallelLines l m → perpendicularPlanes α β) := by
  sorry

end spatial_relationships_l1122_112260


namespace work_done_by_resultant_force_l1122_112200

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Calculates the dot product of two 2D vectors -/
def dot_product (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Adds two 2D vectors -/
def add_vectors (v1 v2 : Vector2D) : Vector2D :=
  ⟨v1.x + v2.x, v1.y + v2.y⟩

theorem work_done_by_resultant_force : 
  let f1 : Vector2D := ⟨3, -4⟩
  let f2 : Vector2D := ⟨2, -5⟩
  let f3 : Vector2D := ⟨3, 1⟩
  let a : Vector2D := ⟨1, 1⟩
  let b : Vector2D := ⟨0, 5⟩
  let resultant_force := add_vectors (add_vectors f1 f2) f3
  let displacement := ⟨b.x - a.x, b.y - a.y⟩
  dot_product resultant_force displacement = -40 := by
  sorry

end work_done_by_resultant_force_l1122_112200


namespace inequality_solution_set_l1122_112242

theorem inequality_solution_set (x : ℝ) :
  (x^2 / (x + 1) ≥ 3 / (x - 2) + 9 / 4) ↔ (x < -3/4 ∨ (x > 2 ∧ x < 5)) :=
sorry

end inequality_solution_set_l1122_112242


namespace pentagon_angles_count_l1122_112226

/-- Represents a sequence of 5 interior angles of a convex pentagon --/
structure PentagonAngles where
  angles : Fin 5 → ℕ
  sum_540 : (angles 0) + (angles 1) + (angles 2) + (angles 3) + (angles 4) = 540
  increasing : ∀ i j, i < j → angles i < angles j
  smallest_ge_60 : angles 0 ≥ 60
  largest_lt_150 : angles 4 < 150
  arithmetic : ∃ d : ℕ, ∀ i : Fin 4, angles (i + 1) = angles i + d
  not_equiangular : ¬ (∀ i j, angles i = angles j)

/-- The number of valid PentagonAngles --/
def validPentagonAnglesCount : ℕ := 5

theorem pentagon_angles_count :
  {s : Finset PentagonAngles | s.card = validPentagonAnglesCount} ≠ ∅ :=
sorry

end pentagon_angles_count_l1122_112226


namespace wendy_miles_walked_l1122_112259

def pedometer_max : ℕ := 49999
def flips : ℕ := 60
def final_reading : ℕ := 25000
def steps_per_mile : ℕ := 1500

def total_steps : ℕ := (pedometer_max + 1) * flips + final_reading

def miles_walked : ℚ := total_steps / steps_per_mile

theorem wendy_miles_walked :
  ⌊(miles_walked + 50) / 100⌋ * 100 = 2000 :=
sorry

end wendy_miles_walked_l1122_112259


namespace sonnys_cookies_l1122_112281

/-- Given an initial number of cookie boxes and the number of boxes given to brother, sister, and cousin,
    calculate the number of boxes left for Sonny. -/
def cookies_left (initial : ℕ) (to_brother : ℕ) (to_sister : ℕ) (to_cousin : ℕ) : ℕ :=
  initial - (to_brother + to_sister + to_cousin)

/-- Theorem stating that given 45 initial boxes of cookies, after giving away 12 to brother,
    9 to sister, and 7 to cousin, the number of boxes left for Sonny is 17. -/
theorem sonnys_cookies : cookies_left 45 12 9 7 = 17 := by
  sorry

end sonnys_cookies_l1122_112281


namespace arrangement_count_l1122_112238

def number_of_arrangements (n_male n_female : ℕ) : ℕ :=
  sorry

theorem arrangement_count :
  let n_male := 2
  let n_female := 3
  number_of_arrangements n_male n_female = 48 :=
by
  sorry

end arrangement_count_l1122_112238


namespace balloon_problem_solution_l1122_112221

/-- The total number of balloons Brooke and Tracy have after Tracy pops half of hers -/
def total_balloons (brooke_initial : ℕ) (brooke_added : ℕ) (tracy_initial : ℕ) (tracy_added : ℕ) : ℕ :=
  let brooke_total := brooke_initial + brooke_added
  let tracy_before_popping := tracy_initial + tracy_added
  let tracy_after_popping := tracy_before_popping / 2
  brooke_total + tracy_after_popping

/-- Theorem stating that the total number of balloons is 35 given the problem conditions -/
theorem balloon_problem_solution :
  total_balloons 12 8 6 24 = 35 := by
  sorry

end balloon_problem_solution_l1122_112221


namespace police_departments_female_officers_l1122_112261

/-- Represents a police department with female officers -/
structure Department where
  totalOfficers : ℕ
  femaleOfficersOnDuty : ℕ
  femaleOfficerPercentage : ℚ

/-- Calculates the total number of female officers in a department -/
def totalFemaleOfficers (d : Department) : ℕ :=
  (d.femaleOfficersOnDuty : ℚ) / d.femaleOfficerPercentage |>.ceil.toNat

theorem police_departments_female_officers 
  (deptA : Department)
  (deptB : Department)
  (deptC : Department)
  (hA : deptA = { totalOfficers := 180, femaleOfficersOnDuty := 90, femaleOfficerPercentage := 18/100 })
  (hB : deptB = { totalOfficers := 200, femaleOfficersOnDuty := 60, femaleOfficerPercentage := 25/100 })
  (hC : deptC = { totalOfficers := 150, femaleOfficersOnDuty := 40, femaleOfficerPercentage := 30/100 }) :
  totalFemaleOfficers deptA = 500 ∧
  totalFemaleOfficers deptB = 240 ∧
  totalFemaleOfficers deptC = 133 ∧
  totalFemaleOfficers deptA + totalFemaleOfficers deptB + totalFemaleOfficers deptC = 873 := by
  sorry

end police_departments_female_officers_l1122_112261


namespace min_value_geometric_sequence_l1122_112246

/-- Given a geometric sequence with first term a₁ = 1, 
    the minimum value of 3a₂ + 7a₃ is -27/28 -/
theorem min_value_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 1 = 1 →                     -- first term is 1
  ∃ m : ℝ, m = -27/28 ∧ ∀ r : ℝ, 3 * (a 2) + 7 * (a 3) ≥ m :=
by sorry

end min_value_geometric_sequence_l1122_112246


namespace A_simplest_form_l1122_112209

/-- The complex expression A -/
def A : ℚ :=
  (0.375 * 2.6) / (2.5 * 1.2) +
  (0.625 * 1.6) / (3 * 1.2 * 4.1666666666666666) +
  6.666666666666667 * 0.12 +
  28 +
  (1 / 9) / 7 +
  0.2 / (9 * 22)

/-- Theorem stating that A, when expressed as a fraction in simplest form, has numerator 1901 and denominator 360 -/
theorem A_simplest_form :
  let (n, d) := (A.num, A.den)
  (n.gcd d = 1) ∧ (n = 1901) ∧ (d = 360) := by sorry

end A_simplest_form_l1122_112209


namespace speed_calculation_l1122_112219

/-- Given a distance of 240 km and a travel time of 6 hours, prove that the speed is 40 km/hr. -/
theorem speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 240) (h2 : time = 6) :
  distance / time = 40 := by
  sorry

end speed_calculation_l1122_112219


namespace circle_equation_implies_m_lt_5_l1122_112257

/-- A circle in the xy-plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- The equation of a circle given by x^2 + y^2 - 4x - 2y + m = 0 --/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + m = 0

/-- Theorem: If x^2 + y^2 - 4x - 2y + m = 0 represents a circle, then m < 5 --/
theorem circle_equation_implies_m_lt_5 :
  ∀ m : ℝ, (∃ c : Circle, ∀ x y : ℝ, circle_equation x y m ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) → m < 5 :=
by sorry

end circle_equation_implies_m_lt_5_l1122_112257


namespace value_added_to_half_l1122_112241

theorem value_added_to_half : ∃ v : ℝ, (1/2 : ℝ) * 16 + v = 13 ∧ v = 5 := by
  sorry

end value_added_to_half_l1122_112241


namespace trees_planted_l1122_112201

def road_length : ℕ := 2575
def tree_interval : ℕ := 25

theorem trees_planted (n : ℕ) : 
  n = road_length / tree_interval + 1 → n = 104 := by
  sorry

end trees_planted_l1122_112201


namespace odd_function_theorem_l1122_112269

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- Definition of function g in terms of f -/
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

/-- Theorem: If f is odd, g(x) = f(x) + 2, and g(1) = 1, then g(-1) = 3 -/
theorem odd_function_theorem (f : ℝ → ℝ) (h1 : IsOdd f) (h2 : g f 1 = 1) : g f (-1) = 3 := by
  sorry


end odd_function_theorem_l1122_112269


namespace initial_players_count_l1122_112222

theorem initial_players_count (players_quit : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  let initial_players := 8
  let remaining_players := initial_players - players_quit
  have h1 : players_quit = 3 := by sorry
  have h2 : lives_per_player = 3 := by sorry
  have h3 : total_lives = 15 := by sorry
  have h4 : remaining_players * lives_per_player = total_lives := by sorry
  initial_players

#check initial_players_count

end initial_players_count_l1122_112222


namespace shortest_chord_length_l1122_112256

/-- Circle C with center (1,2) and radius 5 -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

/-- Line l passing through point M(3,1) -/
def line_l (x y m : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

/-- Point M(3,1) -/
def point_M : ℝ × ℝ := (3, 1)

/-- M is inside circle C -/
axiom M_inside_C : circle_C point_M.1 point_M.2

/-- The shortest chord theorem -/
theorem shortest_chord_length :
  ∃ (m : ℝ), line_l point_M.1 point_M.2 m →
  (∀ (x y : ℝ), line_l x y m → circle_C x y →
  ∃ (x' y' : ℝ), line_l x' y' m ∧ circle_C x' y' ∧
  ((x - x')^2 + (y - y')^2)^(1/2) ≤ 4 * 5^(1/2)) ∧
  (∃ (x y x' y' : ℝ), line_l x y m ∧ circle_C x y ∧
  line_l x' y' m ∧ circle_C x' y' ∧
  ((x - x')^2 + (y - y')^2)^(1/2) = 4 * 5^(1/2)) :=
sorry

end shortest_chord_length_l1122_112256


namespace point_transformation_l1122_112240

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the initial point A
def A : Point2D := ⟨5, 4⟩

-- Define the transformation function
def transform (p : Point2D) : Point2D :=
  ⟨p.x - 4, p.y - 3⟩

-- State the theorem
theorem point_transformation :
  transform A = Point2D.mk 1 1 := by sorry

end point_transformation_l1122_112240


namespace original_number_exists_l1122_112297

theorem original_number_exists : ∃ x : ℝ, 3 * (2 * x + 5) = 117 := by
  sorry

end original_number_exists_l1122_112297


namespace garage_sale_items_count_l1122_112207

theorem garage_sale_items_count (prices : Finset ℕ) (radio_price : ℕ) : 
  radio_price ∈ prices →
  (prices.filter (λ x => x > radio_price)).card = 15 →
  (prices.filter (λ x => x < radio_price)).card = 22 →
  prices.card = 38 := by
  sorry

end garage_sale_items_count_l1122_112207


namespace james_pistachio_expenditure_l1122_112245

/-- Calculates the weekly expenditure on pistachios given the cost per can, ounces per can, daily consumption, and days of consumption. -/
def weekly_pistachio_expenditure (cost_per_can : ℚ) (ounces_per_can : ℚ) (ounces_consumed : ℚ) (days_consumed : ℕ) : ℚ :=
  let weekly_consumption := (7 : ℚ) / days_consumed * ounces_consumed
  let cans_needed := (weekly_consumption / ounces_per_can).ceil
  cans_needed * cost_per_can

/-- Theorem stating that James' weekly expenditure on pistachios is $90. -/
theorem james_pistachio_expenditure :
  weekly_pistachio_expenditure 10 5 30 5 = 90 := by
  sorry

end james_pistachio_expenditure_l1122_112245


namespace fixed_point_of_exponential_shift_l1122_112293

/-- For any constant a > 0 and a ≠ 1, the function f(x) = a^(x-1) - 1 passes through the point (1, 0) -/
theorem fixed_point_of_exponential_shift (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) - 1
  f 1 = 0 := by
  sorry

end fixed_point_of_exponential_shift_l1122_112293


namespace catch_up_time_l1122_112283

-- Define the velocities of objects A and B
def v_A (t : ℝ) : ℝ := 3 * t^2 + 1
def v_B (t : ℝ) : ℝ := 10 * t

-- Define the distances traveled by objects A and B
def d_A (t : ℝ) : ℝ := t^3 + t
def d_B (t : ℝ) : ℝ := 5 * t^2 + 5

-- Theorem: Object A catches up with object B at t = 5 seconds
theorem catch_up_time : 
  ∃ t : ℝ, t = 5 ∧ d_A t = d_B t :=
sorry

end catch_up_time_l1122_112283


namespace john_spends_more_l1122_112212

/-- Calculates the difference in annual cost between John's new and former living arrangements -/
def annual_cost_difference (former_rent_per_sqft : ℚ) (former_size : ℚ) 
  (new_rent_first_half : ℚ) (new_rent_increase_percent : ℚ) 
  (winter_utilities : ℚ) (other_utilities : ℚ) : ℚ :=
  let former_annual_cost := former_rent_per_sqft * former_size * 12
  let new_rent_second_half := new_rent_first_half * (1 + new_rent_increase_percent)
  let new_annual_rent := new_rent_first_half * 6 + new_rent_second_half * 6
  let new_annual_utilities := winter_utilities * 3 + other_utilities * 9
  let new_total_cost := new_annual_rent + new_annual_utilities
  let john_new_cost := new_total_cost / 2
  john_new_cost - former_annual_cost

/-- Theorem stating that John spends $195 more annually in the new arrangement -/
theorem john_spends_more : 
  annual_cost_difference 2 750 2800 (5/100) 200 150 = 195 := by
  sorry

end john_spends_more_l1122_112212


namespace sqrt_inequality_l1122_112216

theorem sqrt_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  Real.sqrt (a / d) > Real.sqrt (b / c) := by
  sorry

end sqrt_inequality_l1122_112216


namespace fraction_calculation_l1122_112235

theorem fraction_calculation (x y : ℚ) (hx : x = 4/7) (hy : y = 5/8) :
  (7*x + 5*y) / (70*x*y) = 57/400 := by
sorry

end fraction_calculation_l1122_112235


namespace book_page_numbering_l1122_112254

/-- The total number of digits used to number pages in a book -/
def total_digits (n : ℕ) : ℕ :=
  let single_digit := min n 9
  let double_digit := max 0 (min n 99 - 9)
  let triple_digit := max 0 (n - 99)
  single_digit + 2 * double_digit + 3 * triple_digit

/-- Theorem stating that a book with 266 pages uses 690 digits for page numbering -/
theorem book_page_numbering :
  total_digits 266 = 690 := by
  sorry

end book_page_numbering_l1122_112254


namespace intersection_M_N_l1122_112290

def M : Set ℝ := {0, 2}
def N : Set ℝ := {x | 0 ≤ x ∧ x < 2}

theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end intersection_M_N_l1122_112290


namespace apple_pie_apples_apple_pie_theorem_l1122_112251

theorem apple_pie_apples (total_greg_sarah : ℕ) (susan_multiplier : ℕ) (mark_difference : ℕ) (mom_leftover : ℕ) : ℕ :=
  let greg_apples := total_greg_sarah / 2
  let susan_apples := greg_apples * susan_multiplier
  let mark_apples := susan_apples - mark_difference
  let pie_apples := susan_apples - mom_leftover
  pie_apples

theorem apple_pie_theorem :
  apple_pie_apples 18 2 5 9 = 9 := by
  sorry

end apple_pie_apples_apple_pie_theorem_l1122_112251


namespace peri_arrival_day_l1122_112230

def travel_pattern (day : ℕ) : ℕ :=
  if day % 10 = 0 then 0 else 1

def total_distance (n : ℕ) : ℕ :=
  (List.range n).map travel_pattern |> List.sum

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed - 1) % 7 + 1

theorem peri_arrival_day :
  ∃ (n : ℕ), total_distance n = 90 ∧ day_of_week 1 n = 2 :=
sorry

end peri_arrival_day_l1122_112230


namespace smallest_non_factor_product_of_48_l1122_112211

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_non_factor_product_of_48 (x y : ℕ) : 
  x ≠ y → 
  x > 0 → 
  y > 0 → 
  is_factor x 48 → 
  is_factor y 48 → 
  ¬ is_factor (x * y) 48 → 
  ∀ a b : ℕ, a ≠ b → a > 0 → b > 0 → is_factor a 48 → is_factor b 48 → ¬ is_factor (a * b) 48 → x * y ≤ a * b →
  x * y = 32 :=
sorry

end smallest_non_factor_product_of_48_l1122_112211


namespace decompose_6058_l1122_112239

theorem decompose_6058 : 6058 = 6 * 1000 + 5 * 10 + 8 * 1 := by
  sorry

end decompose_6058_l1122_112239


namespace intersection_empty_implies_a_ge_5_not_p_sufficient_not_necessary_implies_a_le_2_l1122_112286

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | x ≥ 1 + a ∨ x ≤ 1 - a}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

-- Theorem 1: If A ∩ B = ∅, then a ≥ 5
theorem intersection_empty_implies_a_ge_5 (a : ℝ) (h : a > 0) :
  A ∩ B a = ∅ → a ≥ 5 := by sorry

-- Theorem 2: If ¬p is a sufficient but not necessary condition for q, then 0 < a ≤ 2
theorem not_p_sufficient_not_necessary_implies_a_le_2 (a : ℝ) (h : a > 0) :
  (∀ x, ¬p x → q a x) ∧ (∃ x, q a x ∧ p x) → a ≤ 2 := by sorry

end intersection_empty_implies_a_ge_5_not_p_sufficient_not_necessary_implies_a_le_2_l1122_112286


namespace prob_even_after_removal_l1122_112291

/-- Probability of selecting a dot from a face with n dots -/
def probSelectDot (n : ℕ) : ℚ := n / 21

/-- Probability that a face with n dots remains even after removing two dots -/
def probRemainsEven (n : ℕ) : ℚ :=
  if n % 2 = 0
  then 1 - probSelectDot n * ((n - 1) / 20)
  else probSelectDot n * ((n - 1) / 20)

/-- The probability of rolling an even number of dots after removing two random dots -/
def probEvenAfterRemoval : ℚ :=
  (1 / 6) * (probRemainsEven 1 + probRemainsEven 2 + probRemainsEven 3 +
             probRemainsEven 4 + probRemainsEven 5 + probRemainsEven 6)

theorem prob_even_after_removal :
  probEvenAfterRemoval = 167 / 630 := by
  sorry

end prob_even_after_removal_l1122_112291


namespace room_width_calculation_l1122_112278

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  area : ℝ
  length : ℝ
  width : ℝ

/-- Theorem: Given a room with area 12.0 sq ft and length 1.5 ft, its width is 8.0 ft -/
theorem room_width_calculation (room : RoomDimensions) 
  (h_area : room.area = 12.0) 
  (h_length : room.length = 1.5) : 
  room.width = 8.0 := by
  sorry

end room_width_calculation_l1122_112278


namespace largest_n_for_factorization_l1122_112285

/-- 
Given a quadratic expression of the form 3x^2 + nx + 108, 
this theorem states that the largest value of n for which 
the expression can be factored as the product of two linear 
factors with integer coefficients is 325.
-/
theorem largest_n_for_factorization : 
  (∃ (n : ℤ), ∀ (A B : ℤ), 
    (3 * A = 1 ∧ B = 108) → 
    (∀ (x : ℤ), 3 * x^2 + n * x + 108 = (3 * x + A) * (x + B)) ∧
    (∀ (m : ℤ), (∃ (C D : ℤ), ∀ (x : ℤ), 
      3 * x^2 + m * x + 108 = (3 * x + C) * (x + D)) → 
      m ≤ n)) ∧
  (∀ (n : ℤ), (∀ (A B : ℤ), 
    (3 * A = 1 ∧ B = 108) → 
    (∀ (x : ℤ), 3 * x^2 + n * x + 108 = (3 * x + A) * (x + B)) ∧
    (∀ (m : ℤ), (∃ (C D : ℤ), ∀ (x : ℤ), 
      3 * x^2 + m * x + 108 = (3 * x + C) * (x + D)) → 
      m ≤ n)) → 
  n = 325) :=
by sorry

end largest_n_for_factorization_l1122_112285


namespace recurring_decimal_subtraction_l1122_112253

theorem recurring_decimal_subtraction : 
  (246 : ℚ) / 999 - 135 / 999 - 579 / 999 = -52 / 111 := by
  sorry

end recurring_decimal_subtraction_l1122_112253


namespace hadley_walk_l1122_112250

/-- Hadley's walk problem -/
theorem hadley_walk (x : ℝ) :
  (x ≥ 0) →
  (x - 1 ≥ 0) →
  (x + (x - 1) + 3 = 6) →
  x = 2 := by
  sorry

end hadley_walk_l1122_112250


namespace geometric_sequence_fifth_term_l1122_112264

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_eq : a 4 * a 6 + a 5 ^ 2 = 50) :
  a 5 = 5 := by
  sorry

end geometric_sequence_fifth_term_l1122_112264


namespace triangle_side_calculation_l1122_112282

open Real

theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Condition 1: sin C + 2sin C cos B = sin A
  sin C + 2 * sin C * cos B = sin A →
  -- Condition 2: C ∈ (0, π/2)
  0 < C ∧ C < π / 2 →
  -- Condition 3: a = √6
  a = Real.sqrt 6 →
  -- Condition 4: cos B = 1/3
  cos B = 1 / 3 →
  -- Conclusion: b = 12/5
  b = 12 / 5 := by
sorry

end triangle_side_calculation_l1122_112282
