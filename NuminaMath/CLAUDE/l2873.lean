import Mathlib

namespace incorrect_observation_value_l2873_287315

theorem incorrect_observation_value 
  (n : ℕ) 
  (initial_mean : ℝ) 
  (correct_value : ℝ) 
  (new_mean : ℝ) 
  (h1 : n = 40)
  (h2 : initial_mean = 100)
  (h3 : correct_value = 50)
  (h4 : new_mean = 99.075) :
  (n : ℝ) * initial_mean - (n : ℝ) * new_mean + correct_value = 87 :=
by sorry

end incorrect_observation_value_l2873_287315


namespace reception_friends_l2873_287322

def wedding_reception (total_attendees : ℕ) (bride_couples : ℕ) (groom_couples : ℕ) : Prop :=
  let family_members := 2 * (bride_couples + groom_couples)
  let friends := total_attendees - family_members
  friends = 100

theorem reception_friends :
  wedding_reception 180 20 20 := by
  sorry

end reception_friends_l2873_287322


namespace function_composition_value_l2873_287367

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := 4 * x - 6
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem function_composition_value (a b : ℝ) :
  (∀ x : ℝ, h a b x = (x - 14) / 2) → a - 2 * b = 101 / 8 := by
  sorry

end function_composition_value_l2873_287367


namespace gcd_2048_2101_l2873_287341

theorem gcd_2048_2101 : Nat.gcd 2048 2101 = 1 := by
  sorry

end gcd_2048_2101_l2873_287341


namespace final_silver_count_l2873_287358

/-- Represents the number of tokens Alex has -/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents a token exchange booth -/
structure Booth where
  redIn : ℕ
  blueIn : ℕ
  redOut : ℕ
  blueOut : ℕ
  silverOut : ℕ

/-- Checks if an exchange is possible at a given booth -/
def canExchange (tokens : TokenCount) (booth : Booth) : Prop :=
  tokens.red ≥ booth.redIn ∧ tokens.blue ≥ booth.blueIn

/-- Performs an exchange at a given booth -/
def exchange (tokens : TokenCount) (booth : Booth) : TokenCount :=
  { red := tokens.red - booth.redIn + booth.redOut,
    blue := tokens.blue - booth.blueIn + booth.blueOut,
    silver := tokens.silver + booth.silverOut }

/-- Checks if any exchanges are possible -/
def exchangesPossible (tokens : TokenCount) (booths : List Booth) : Prop :=
  ∃ b ∈ booths, canExchange tokens b

/-- The main theorem to prove -/
theorem final_silver_count 
  (initialTokens : TokenCount)
  (booth1 booth2 : Booth)
  (h_initial : initialTokens = ⟨75, 75, 0⟩)
  (h_booth1 : booth1 = ⟨2, 0, 0, 1, 1⟩)
  (h_booth2 : booth2 = ⟨0, 3, 1, 0, 1⟩)
  : ∃ (finalTokens : TokenCount), 
    (¬ exchangesPossible finalTokens [booth1, booth2]) ∧ 
    finalTokens.silver = 103 := by
  sorry

end final_silver_count_l2873_287358


namespace sum_of_A_and_B_is_31_l2873_287311

theorem sum_of_A_and_B_is_31 (A B : ℝ) :
  (∀ x : ℝ, x ≠ 3 → A / (x - 3) + B * (x + 2) = (-4 * x^2 + 11 * x + 35) / (x - 3)) →
  A + B = 31 := by
  sorry

end sum_of_A_and_B_is_31_l2873_287311


namespace center_square_area_ratio_l2873_287347

/-- Represents a square flag with a symmetric cross -/
structure SquareFlag where
  side : ℝ
  cross_width : ℝ
  center_side : ℝ
  cross_area_ratio : ℝ
  cross_symmetric : Bool
  cross_uniform : Bool

/-- The theorem stating that if a symmetric cross occupies 50% of a square flag's area, 
    the center square occupies 6.25% of the total area -/
theorem center_square_area_ratio (flag : SquareFlag) 
  (h1 : flag.cross_area_ratio = 0.5)
  (h2 : flag.cross_symmetric = true)
  (h3 : flag.cross_uniform = true)
  (h4 : flag.center_side = flag.side / 4) :
  (flag.center_side ^ 2) / (flag.side ^ 2) = 0.0625 := by
  sorry

end center_square_area_ratio_l2873_287347


namespace tom_initial_money_l2873_287339

/-- Tom's initial amount of money -/
def initial_money : ℕ := sorry

/-- Cost of the game Tom bought -/
def game_cost : ℕ := 49

/-- Cost of each toy -/
def toy_cost : ℕ := 4

/-- Number of toys Tom could buy after purchasing the game -/
def num_toys : ℕ := 2

/-- Theorem stating that Tom's initial money was $57 -/
theorem tom_initial_money : 
  initial_money = game_cost + num_toys * toy_cost :=
sorry

end tom_initial_money_l2873_287339


namespace perimeter_decreases_to_convex_hull_outer_perimeter_not_smaller_l2873_287387

-- Define a polygon as a list of points in 2D space
def Polygon := List (ℝ × ℝ)

-- Define a function to calculate the perimeter of a polygon
def perimeter (p : Polygon) : ℝ := sorry

-- Define a predicate to check if a polygon is convex
def is_convex (p : Polygon) : Prop := sorry

-- Define the convex hull of a polygon
def convex_hull (p : Polygon) : Polygon := sorry

-- Define a predicate to check if one polygon is completely inside another
def is_inside (a b : Polygon) : Prop := sorry

theorem perimeter_decreases_to_convex_hull (p : Polygon) : 
  perimeter (convex_hull p) < perimeter p := sorry

theorem outer_perimeter_not_smaller (a b : Polygon) 
  (h1 : is_convex a) (h2 : is_convex b) (h3 : is_inside a b) : 
  perimeter b ≥ perimeter a := sorry

end perimeter_decreases_to_convex_hull_outer_perimeter_not_smaller_l2873_287387


namespace jean_spot_ratio_l2873_287320

/-- Represents the number of spots on different parts of Jean's body -/
structure SpotCount where
  upperTorso : ℕ
  sides : ℕ

/-- The ratio of spots on the upper torso to total spots -/
def spotRatio (s : SpotCount) : ℚ :=
  s.upperTorso / (s.upperTorso + s.sides)

/-- Theorem stating that the ratio of spots on the upper torso to total spots is 3/4 -/
theorem jean_spot_ratio :
  ∀ (s : SpotCount), s.upperTorso = 30 → s.sides = 10 → spotRatio s = 3/4 := by
  sorry


end jean_spot_ratio_l2873_287320


namespace proj_equals_v_l2873_287378

/-- Given two 2D vectors v and w, prove that the projection of v onto w is equal to v itself. -/
theorem proj_equals_v (v w : Fin 2 → ℝ) (hv : v = ![- 3, 2]) (hw : w = ![4, - 2]) :
  (v • w / (w • w)) • w = v := by sorry

end proj_equals_v_l2873_287378


namespace tara_book_sales_l2873_287333

/-- The number of books Tara needs to sell to reach her goal -/
def books_to_sell (initial_savings : ℕ) (clarinet_cost : ℕ) (book_price : ℕ) (accessory_cost : ℕ) : ℕ :=
  let goal := clarinet_cost - initial_savings
  let halfway := goal / 2
  let books_to_halfway := halfway / book_price
  let new_goal := goal + accessory_cost
  let books_after_loss := new_goal / book_price
  books_to_halfway + books_after_loss

/-- Theorem stating that Tara needs to sell 35 books in total -/
theorem tara_book_sales :
  books_to_sell 10 90 4 20 = 35 := by
  sorry

end tara_book_sales_l2873_287333


namespace equal_area_division_l2873_287370

/-- The value of c that divides the area of five unit squares into two equal regions -/
def c : ℝ := 1.75

/-- The total area of the five unit squares -/
def total_area : ℝ := 5

/-- The equation of the line passing through (c, 0) and (3, 4) -/
def line_equation (x y : ℝ) : Prop := y = (4 / (3 - c)) * (x - c)

/-- The area of the triangle formed by the line and the x-axis -/
def triangle_area : ℝ := 2 * (3 - c)

theorem equal_area_division :
  triangle_area = total_area / 2 ↔ c = 1.75 :=
sorry

end equal_area_division_l2873_287370


namespace megan_carrots_second_day_l2873_287361

/-- Calculates the number of carrots Megan picked on the second day -/
def carrots_picked_second_day (initial_carrots : ℕ) (thrown_out : ℕ) (total_carrots : ℕ) : ℕ :=
  total_carrots - (initial_carrots - thrown_out)

/-- Proves that Megan picked 46 carrots on the second day -/
theorem megan_carrots_second_day : 
  carrots_picked_second_day 19 4 61 = 46 := by
  sorry

end megan_carrots_second_day_l2873_287361


namespace unanswered_completion_count_l2873_287345

/-- A structure representing a multiple choice test -/
structure MultipleChoiceTest where
  total_questions : Nat
  choices_per_question : Nat
  single_answer_questions : Nat
  multi_select_questions : Nat
  correct_choices_per_multi : Nat

/-- The number of ways to complete the test with all questions unanswered -/
def ways_to_complete_unanswered (test : MultipleChoiceTest) : Nat :=
  1

/-- Theorem stating that there is only one way to complete the test with all questions unanswered -/
theorem unanswered_completion_count (test : MultipleChoiceTest)
  (h1 : test.total_questions = 10)
  (h2 : test.choices_per_question = 8)
  (h3 : test.single_answer_questions = 6)
  (h4 : test.multi_select_questions = 4)
  (h5 : test.correct_choices_per_multi = 2)
  (h6 : test.total_questions = test.single_answer_questions + test.multi_select_questions) :
  ways_to_complete_unanswered test = 1 := by
  sorry

end unanswered_completion_count_l2873_287345


namespace smallest_angle_satisfying_equation_l2873_287309

theorem smallest_angle_satisfying_equation :
  let f : ℝ → ℝ := λ x => 9 * Real.sin x * Real.cos x ^ 4 - 9 * Real.sin x ^ 4 * Real.cos x
  ∃ x : ℝ, x > 0 ∧ f x = 1/2 ∧ ∀ y : ℝ, y > 0 → f y = 1/2 → x ≤ y ∧ x = π/6 :=
by sorry

end smallest_angle_satisfying_equation_l2873_287309


namespace positive_y_squared_geq_2y_minus_1_l2873_287392

theorem positive_y_squared_geq_2y_minus_1 :
  ∀ y : ℝ, y > 0 → y^2 ≥ 2*y - 1 := by
  sorry

end positive_y_squared_geq_2y_minus_1_l2873_287392


namespace mariela_get_well_cards_l2873_287375

theorem mariela_get_well_cards (hospital_cards : ℕ) (home_cards : ℕ) 
  (h1 : hospital_cards = 403) (h2 : home_cards = 287) : 
  hospital_cards + home_cards = 690 := by
  sorry

end mariela_get_well_cards_l2873_287375


namespace balloon_count_theorem_l2873_287365

theorem balloon_count_theorem (fred sam dan total : ℕ) 
  (h1 : fred = 10)
  (h2 : sam = 46)
  (h3 : dan = 16)
  (h4 : total = 72) :
  fred + sam + dan = total :=
sorry

end balloon_count_theorem_l2873_287365


namespace subtraction_and_divisibility_imply_sum_l2873_287346

/-- A number is divisible by 11 if and only if the alternating sum of its digits is divisible by 11 -/
def divisible_by_11 (n : ℕ) : Prop :=
  (n % 11 = 0)

/-- Returns the hundreds digit of a three-digit number -/
def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

/-- Returns the tens digit of a three-digit number -/
def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- Returns the ones digit of a three-digit number -/
def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem subtraction_and_divisibility_imply_sum (c d : ℕ) :
  (745 - (300 + c * 10 + 4) = 400 + d * 10 + 1) →
  divisible_by_11 (400 + d * 10 + 1) →
  c + d = 14 := by
  sorry

end subtraction_and_divisibility_imply_sum_l2873_287346


namespace integral_equals_ln_80_over_23_l2873_287356

open Real MeasureTheory

theorem integral_equals_ln_80_over_23 :
  ∫ x in (1 : ℝ)..2, (9 * x + 4) / (x^5 + 3 * x^2 + x) = Real.log (80 / 23) := by
  sorry

end integral_equals_ln_80_over_23_l2873_287356


namespace max_speed_theorem_l2873_287319

/-- Represents a data point of (speed, defective items) -/
structure DataPoint where
  speed : ℝ
  defective : ℝ

/-- The set of observed data points -/
def observed_data : List DataPoint := [
  ⟨8, 5⟩, ⟨12, 8⟩, ⟨14, 9⟩, ⟨16, 11⟩
]

/-- Calculates the slope of the regression line -/
def calculate_slope (data : List DataPoint) : ℝ := sorry

/-- Calculates the y-intercept of the regression line -/
def calculate_intercept (data : List DataPoint) (slope : ℝ) : ℝ := sorry

/-- The maximum number of defective items allowed per hour -/
def max_defective : ℝ := 10

theorem max_speed_theorem (data : List DataPoint) 
    (h_linear : ∃ (m b : ℝ), ∀ point ∈ data, point.defective = m * point.speed + b) :
  let slope := calculate_slope data
  let intercept := calculate_intercept data slope
  let max_speed := (max_defective - intercept) / slope
  ⌊max_speed⌋ = 15 := by sorry

end max_speed_theorem_l2873_287319


namespace median_and_midline_projection_l2873_287353

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a parallel projection
def ParallelProjection := (ℝ × ℝ) → (ℝ × ℝ)

-- Define a median of a triangle
def median (t : Triangle) : (ℝ × ℝ) := sorry

-- Define a midline of a triangle
def midline (t : Triangle) : (ℝ × ℝ) := sorry

-- Theorem statement
theorem median_and_midline_projection 
  (t : Triangle) 
  (p : ParallelProjection) 
  (h : ∃ t', t' = Triangle.mk (p t.A) (p t.B) (p t.C)) :
  (p (median t) = median (Triangle.mk (p t.A) (p t.B) (p t.C))) ∧
  (p (midline t) = midline (Triangle.mk (p t.A) (p t.B) (p t.C))) := by
  sorry

end median_and_midline_projection_l2873_287353


namespace cos_B_in_triangle_l2873_287321

theorem cos_B_in_triangle (A B C : ℝ) (AC BC : ℝ) (angle_A : ℝ) :
  AC = Real.sqrt 2 →
  BC = Real.sqrt 3 →
  angle_A = π / 3 →
  Real.cos B = Real.sqrt 2 / 2 :=
by sorry

end cos_B_in_triangle_l2873_287321


namespace pears_minus_apples_equals_two_l2873_287335

/-- Represents a bowl of fruit containing apples, pears, and bananas. -/
structure FruitBowl where
  apples : ℕ
  pears : ℕ
  bananas : ℕ

/-- A bowl of fruit satisfying the given conditions. -/
def specialBowl : FruitBowl :=
  { apples := 0,  -- Placeholder value, will be constrained by theorem
    pears := 0,   -- Placeholder value, will be constrained by theorem
    bananas := 9 }

theorem pears_minus_apples_equals_two (bowl : FruitBowl) :
  bowl.apples + bowl.pears + bowl.bananas = 19 →
  bowl.bananas = 9 →
  bowl.bananas = bowl.pears + 3 →
  bowl.pears > bowl.apples →
  bowl.pears - bowl.apples = 2 := by
  sorry

#check pears_minus_apples_equals_two specialBowl

end pears_minus_apples_equals_two_l2873_287335


namespace expression_simplification_l2873_287397

theorem expression_simplification (x y m : ℝ) 
  (h1 : (x - 5)^2 + |m - 1| = 0)
  (h2 : y + 1 = 5) :
  (2*x^2 - 3*x*y - 4*y^2) - m*(3*x^2 - x*y + 9*y^2) = -273 := by
  sorry

end expression_simplification_l2873_287397


namespace sum_of_two_equals_third_l2873_287342

theorem sum_of_two_equals_third (a b c : ℝ) 
  (h1 : |a - b| ≥ |c|) 
  (h2 : |b - c| ≥ |a|) 
  (h3 : |c - a| ≥ |b|) : 
  a = b + c ∨ b = c + a ∨ c = a + b := by
  sorry

end sum_of_two_equals_third_l2873_287342


namespace eva_apple_count_l2873_287394

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks Eva needs to eat an apple -/
def weeks : ℕ := 2

/-- The number of apples Eva should buy -/
def apples_to_buy : ℕ := days_per_week * weeks

theorem eva_apple_count : apples_to_buy = 14 := by
  sorry

end eva_apple_count_l2873_287394


namespace g_value_at_5_l2873_287352

def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 3

theorem g_value_at_5 (a b c : ℝ) (h : g a b c (-5) = -3) : g a b c 5 = 9 := by
  sorry

end g_value_at_5_l2873_287352


namespace floor_sqrt_19_squared_l2873_287369

theorem floor_sqrt_19_squared : ⌊Real.sqrt 19⌋^2 = 16 := by
  sorry

end floor_sqrt_19_squared_l2873_287369


namespace square_of_sum_fifteen_seven_l2873_287386

theorem square_of_sum_fifteen_seven : 15^2 + 2*(15*7) + 7^2 = 484 := by
  sorry

end square_of_sum_fifteen_seven_l2873_287386


namespace find_q_l2873_287350

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p * q = 8) :
  q = 4 + 2 * Real.sqrt 2 := by
  sorry

end find_q_l2873_287350


namespace toms_video_game_spending_l2873_287318

/-- The cost of the Batman game in dollars -/
def batman_cost : ℚ := 13.60

/-- The cost of the Superman game in dollars -/
def superman_cost : ℚ := 5.06

/-- The number of games Tom already owns -/
def existing_games : ℕ := 2

/-- The total amount Tom spent on video games -/
def total_spent : ℚ := batman_cost + superman_cost

theorem toms_video_game_spending :
  total_spent = 18.66 := by sorry

end toms_video_game_spending_l2873_287318


namespace trevor_taxi_cost_l2873_287344

/-- Calculates the total cost of Trevor's taxi ride downtown -/
def total_taxi_cost (uber_cost lyft_cost taxi_cost detour_rate tip_rate : ℚ) : ℚ :=
  let detour_cost := taxi_cost * detour_rate
  let tip := taxi_cost * tip_rate
  taxi_cost + detour_cost + tip

/-- Proves that the total cost of Trevor's taxi ride downtown is $20.25 -/
theorem trevor_taxi_cost :
  let uber_cost : ℚ := 22
  let lyft_cost : ℚ := uber_cost - 3
  let taxi_cost : ℚ := lyft_cost - 4
  let detour_rate : ℚ := 15 / 100
  let tip_rate : ℚ := 20 / 100
  total_taxi_cost uber_cost lyft_cost taxi_cost detour_rate tip_rate = 8100 / 400 := by
  sorry

#eval total_taxi_cost 22 19 15 (15/100) (20/100)

end trevor_taxi_cost_l2873_287344


namespace living_space_increase_l2873_287329

/-- Proves that the average annual increase in living space needed is approximately 12.05 ten thousand m² --/
theorem living_space_increase (initial_population : ℝ) (initial_space_per_person : ℝ)
  (target_space_per_person : ℝ) (growth_rate : ℝ) (years : ℕ)
  (h1 : initial_population = 20) -- in ten thousands
  (h2 : initial_space_per_person = 8)
  (h3 : target_space_per_person = 10)
  (h4 : growth_rate = 0.01)
  (h5 : years = 4) :
  ∃ x : ℝ, abs (x - 12.05) < 0.01 ∧ 
  x * years = target_space_per_person * (initial_population * (1 + growth_rate) ^ years) - 
              initial_space_per_person * initial_population :=
by sorry


end living_space_increase_l2873_287329


namespace complex_fraction_simplification_l2873_287368

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (3 + 4 * Complex.I) = 43/25 + (1/25) * Complex.I :=
by sorry

end complex_fraction_simplification_l2873_287368


namespace fair_payment_division_l2873_287308

/-- Represents the payment for digging the trench -/
def total_payment : ℚ := 2

/-- Represents Abraham's digging rate relative to Benjamin's soil throwing rate -/
def abraham_dig_rate : ℚ := 1

/-- Represents Benjamin's digging rate relative to Abraham's soil throwing rate -/
def benjamin_dig_rate : ℚ := 4

/-- Represents the ratio of Abraham's payment to the total payment -/
def abraham_payment_ratio : ℚ := 1/3

/-- Represents the ratio of Benjamin's payment to the total payment -/
def benjamin_payment_ratio : ℚ := 2/3

/-- Theorem stating the fair division of payment between Abraham and Benjamin -/
theorem fair_payment_division :
  abraham_payment_ratio * total_payment = 2/3 ∧
  benjamin_payment_ratio * total_payment = 4/3 :=
by sorry

end fair_payment_division_l2873_287308


namespace half_abs_diff_squares_l2873_287316

theorem half_abs_diff_squares : (1/2 : ℝ) * |25^2 - 20^2| = 112.5 := by
  sorry

end half_abs_diff_squares_l2873_287316


namespace basketball_court_equation_rewrite_l2873_287324

theorem basketball_court_equation_rewrite :
  ∃ (a b c : ℤ), a > 0 ∧
  (∀ x : ℝ, 16 * x^2 + 32 * x - 40 = 0 ↔ (a * x + b)^2 = c) ∧
  a + b + c = 64 := by
  sorry

end basketball_court_equation_rewrite_l2873_287324


namespace bicycle_profit_calculation_l2873_287340

theorem bicycle_profit_calculation (profit_A profit_B final_price : ℝ) :
  profit_A = 0.60 ∧ profit_B = 0.25 ∧ final_price = 225 →
  ∃ cost_price_A : ℝ,
    cost_price_A * (1 + profit_A) * (1 + profit_B) = final_price ∧
    cost_price_A = 112.5 := by
  sorry

end bicycle_profit_calculation_l2873_287340


namespace gyeong_hun_climb_l2873_287393

/-- Gyeong-hun's mountain climbing problem -/
theorem gyeong_hun_climb (uphill_speed downhill_speed : ℝ)
                         (downhill_extra_distance total_time : ℝ)
                         (h1 : uphill_speed = 3)
                         (h2 : downhill_speed = 4)
                         (h3 : downhill_extra_distance = 2)
                         (h4 : total_time = 4) :
  ∃ (distance : ℝ),
    distance / uphill_speed + (distance + downhill_extra_distance) / downhill_speed = total_time ∧
    distance = 6 := by
  sorry

end gyeong_hun_climb_l2873_287393


namespace fifteenth_term_ratio_l2873_287396

-- Define the sums of arithmetic sequences
def S (a d n : ℚ) : ℚ := n * (2 * a + (n - 1) * d) / 2
def T (b e n : ℚ) : ℚ := n * (2 * b + (n - 1) * e) / 2

-- Define the ratio condition
def ratio_condition (a d b e n : ℚ) : Prop :=
  S a d n / T b e n = (5 * n + 3) / (3 * n + 17)

-- Define the 15th term of each sequence
def term_15 (a d : ℚ) : ℚ := a + 14 * d

-- Theorem statement
theorem fifteenth_term_ratio 
  (a d b e : ℚ) 
  (h : ∀ n : ℚ, ratio_condition a d b e n) : 
  term_15 a d / term_15 b e = 44 / 95 := by
  sorry

end fifteenth_term_ratio_l2873_287396


namespace polynomial_A_and_difference_l2873_287351

/-- Given polynomials A and B where B = 4x² - 3y - 1 and A + B = 6x² - y -/
def B (x y : ℝ) : ℝ := 4 * x^2 - 3 * y - 1

/-- Definition of A based on the given condition A + B = 6x² - y -/
def A (x y : ℝ) : ℝ := 6 * x^2 - y - B x y

theorem polynomial_A_and_difference (x y : ℝ) :
  A x y = 2 * x^2 + 2 * y + 1 ∧
  (|x - 1| + (y + 1)^2 = 0 → A x y - B x y = -5) :=
by sorry

end polynomial_A_and_difference_l2873_287351


namespace chocolate_bar_cost_is_six_l2873_287357

/-- The cost of each chocolate bar, given the total number of bars, 
    the number of unsold bars, and the total revenue from sales. -/
def chocolate_bar_cost (total_bars : ℕ) (unsold_bars : ℕ) (revenue : ℕ) : ℚ :=
  revenue / (total_bars - unsold_bars)

/-- Theorem stating that the cost of each chocolate bar is $6 under the given conditions. -/
theorem chocolate_bar_cost_is_six : 
  chocolate_bar_cost 13 6 42 = 6 := by
  sorry

end chocolate_bar_cost_is_six_l2873_287357


namespace geric_initial_bills_l2873_287398

/-- The number of bills Jessa had initially -/
def jessa_initial : ℕ := 7 + 3

/-- The number of bills Kyla had -/
def kyla : ℕ := jessa_initial - 2

/-- The number of bills Geric had initially -/
def geric_initial : ℕ := 2 * kyla

theorem geric_initial_bills : geric_initial = 16 := by
  sorry

end geric_initial_bills_l2873_287398


namespace min_value_theorem_l2873_287313

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 3 → 1 / (a + 1) + 2 / b ≥ 1 / (x + 1) + 2 / y) →
  1 / (x + 1) + 2 / y = (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end min_value_theorem_l2873_287313


namespace quadratic_real_roots_l2873_287364

/-- The quadratic equation ax^2 - x + 1 = 0 has real roots if and only if a ≤ 1/4 and a ≠ 0 -/
theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - x + 1 = 0) ↔ (a ≤ 1/4 ∧ a ≠ 0) :=
sorry

end quadratic_real_roots_l2873_287364


namespace brother_catchup_l2873_287373

/-- The time it takes for the older brother to catch up with the younger brother -/
def catchup_time (older_time younger_time delay : ℚ) : ℚ :=
  let relative_speed := 1 / older_time - 1 / younger_time
  let distance_covered := delay / younger_time
  delay + distance_covered / relative_speed

theorem brother_catchup :
  let older_time : ℚ := 12
  let younger_time : ℚ := 20
  let delay : ℚ := 5
  catchup_time older_time younger_time delay = 25/2 := by sorry

end brother_catchup_l2873_287373


namespace problem_solution_l2873_287302

theorem problem_solution (a b c : ℝ) (h1 : a < 0) (h2 : 0 < b) (h3 : b < c) :
  (a^2 * b < a^2 * c) ∧ (a^3 < a^2 * b) ∧ (a + b < b + c) := by
  sorry

end problem_solution_l2873_287302


namespace prob_two_gold_given_at_least_one_gold_l2873_287343

/-- The probability of selecting two gold medals given that at least one gold medal is selected -/
theorem prob_two_gold_given_at_least_one_gold 
  (total_medals : ℕ) 
  (gold_medals : ℕ) 
  (silver_medals : ℕ) 
  (bronze_medals : ℕ) 
  (h1 : total_medals = gold_medals + silver_medals + bronze_medals)
  (h2 : total_medals = 10)
  (h3 : gold_medals = 5)
  (h4 : silver_medals = 3)
  (h5 : bronze_medals = 2) :
  (Nat.choose gold_medals 2 : ℚ) / (Nat.choose total_medals 2 - Nat.choose (silver_medals + bronze_medals) 2) = 2/7 :=
sorry

end prob_two_gold_given_at_least_one_gold_l2873_287343


namespace farm_tree_sub_branches_l2873_287399

/-- Proves that the number of sub-branches per branch is 40, given the conditions from the farm tree problem -/
theorem farm_tree_sub_branches :
  let branches_per_tree : ℕ := 10
  let leaves_per_sub_branch : ℕ := 60
  let total_trees : ℕ := 4
  let total_leaves : ℕ := 96000
  ∃ (sub_branches_per_branch : ℕ),
    sub_branches_per_branch = 40 ∧
    total_leaves = total_trees * branches_per_tree * leaves_per_sub_branch * sub_branches_per_branch :=
by sorry

end farm_tree_sub_branches_l2873_287399


namespace decimal_power_division_l2873_287303

theorem decimal_power_division : (0.4 : ℝ)^4 / (0.04 : ℝ)^3 = 400 := by
  sorry

end decimal_power_division_l2873_287303


namespace boat_speed_in_still_water_l2873_287383

/-- Proves that the speed of a boat in still water is 12 km/hr given the conditions -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : current_speed = 4) 
  (h2 : downstream_distance = 4.8) 
  (h3 : downstream_time = 18 / 60) : 
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 12 ∧ 
    downstream_distance = (still_water_speed + current_speed) * downstream_time :=
by sorry

end boat_speed_in_still_water_l2873_287383


namespace quadrilateral_angles_not_always_form_triangle_l2873_287366

theorem quadrilateral_angles_not_always_form_triangle : ∃ (α β γ δ : ℝ),
  α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0 ∧
  α + β + γ + δ = 360 ∧
  ¬(α + β > γ ∧ β + γ > α ∧ γ + α > β) ∧
  ¬(α + β > δ ∧ β + δ > α ∧ δ + α > β) ∧
  ¬(α + γ > δ ∧ γ + δ > α ∧ δ + α > γ) ∧
  ¬(β + γ > δ ∧ γ + δ > β ∧ δ + β > γ) :=
sorry

end quadrilateral_angles_not_always_form_triangle_l2873_287366


namespace plane_properties_l2873_287355

structure Plane

-- Define parallel and perpendicular relations for planes
def parallel (p q : Plane) : Prop := sorry
def perpendicular (p q : Plane) : Prop := sorry

-- Define line as intersection of two planes
def line_intersection (p q : Plane) : Type := sorry

-- Define parallel relation for lines
def line_parallel (l m : Type) : Prop := sorry

theorem plane_properties (α β γ : Plane) (hd : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  (parallel α β ∧ parallel β γ → parallel α γ) ∧
  (parallel α β → ∀ (a b : Type), a = line_intersection α γ → b = line_intersection β γ → line_parallel a b) ∧
  (parallel α β ∧ perpendicular β γ → perpendicular α γ) ∧
  ¬(∀ α β γ : Plane, perpendicular α β ∧ perpendicular β γ → perpendicular α γ) := by
  sorry

end plane_properties_l2873_287355


namespace arithmetic_sequences_ratio_l2873_287354

/-- Two arithmetic sequences and their sums -/
def arithmetic_sequences (a b : ℕ → ℚ) (S T : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → S n / T n = (2 * n + 1 : ℚ) / (n + 2 : ℚ)

theorem arithmetic_sequences_ratio 
  (a b : ℕ → ℚ) (S T : ℕ → ℚ) 
  (h : arithmetic_sequences a b S T) : 
  a 7 / b 7 = 9 / 5 := by
  sorry

end arithmetic_sequences_ratio_l2873_287354


namespace place_mat_length_l2873_287377

/-- The radius of the round table -/
def table_radius : ℝ := 5

/-- The width of each place mat -/
def mat_width : ℝ := 1.5

/-- The number of place mats -/
def num_mats : ℕ := 4

/-- The theorem stating the length of each place mat -/
theorem place_mat_length :
  ∃ y : ℝ,
    y > 0 ∧
    y = 0.75 ∧
    (y + 2.5 * Real.sqrt 2 - mat_width / 2)^2 + (mat_width / 2)^2 = table_radius^2 ∧
    ∀ (i : Fin num_mats),
      ∃ (x y : ℝ),
        x^2 + y^2 = table_radius^2 ∧
        (x - mat_width / 2)^2 + (y - (y + 2.5 * Real.sqrt 2 - mat_width / 2))^2 = table_radius^2 :=
by
  sorry

end place_mat_length_l2873_287377


namespace find_A_l2873_287381

theorem find_A (A B : ℕ) (h1 : A + B = 1149) (h2 : A = 8 * B + 24) : A = 1024 := by
  sorry

end find_A_l2873_287381


namespace boy_position_in_line_l2873_287389

/-- The position of a boy in a line of boys, where he is equidistant from both ends -/
def midPosition (n : ℕ) : ℕ :=
  (n + 1) / 2

/-- Theorem: In a line of 37 boys, the boy who is equidistant from both ends is in position 19 -/
theorem boy_position_in_line :
  midPosition 37 = 19 := by
  sorry

end boy_position_in_line_l2873_287389


namespace inequality_proof_l2873_287390

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c = 1 → a^2 + b^2 + c^2 ≥ 1/3) ∧
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end inequality_proof_l2873_287390


namespace smallest_m_is_correct_l2873_287384

/-- The smallest positive value of m for which 10x^2 - mx + 180 = 0 has integral solutions -/
def smallest_m : ℕ := 90

/-- A function representing the quadratic equation 10x^2 - mx + 180 = 0 -/
def quadratic (m : ℕ) (x : ℤ) : ℤ := 10 * x^2 - m * x + 180

theorem smallest_m_is_correct : 
  (∃ x y : ℤ, x ≠ y ∧ quadratic smallest_m x = 0 ∧ quadratic smallest_m y = 0) ∧ 
  (∀ m : ℕ, m < smallest_m → 
    ¬∃ x y : ℤ, x ≠ y ∧ quadratic m x = 0 ∧ quadratic m y = 0) :=
sorry

end smallest_m_is_correct_l2873_287384


namespace largest_coin_distribution_l2873_287379

theorem largest_coin_distribution (n : ℕ) : n ≤ 111 ↔ 
  (∃ (k : ℕ), n = 12 * k + 3 ∧ n < 120) :=
sorry

end largest_coin_distribution_l2873_287379


namespace drawing_set_prices_and_quantity_l2873_287325

/-- Represents the cost and selling prices of drawing tool sets from two brands -/
structure DrawingSetPrices where
  costA : ℝ
  costB : ℝ
  sellA : ℝ
  sellB : ℝ

/-- Theorem stating the properties of the drawing set prices and minimum purchase quantity -/
theorem drawing_set_prices_and_quantity (p : DrawingSetPrices)
  (h1 : p.costA = p.costB + 2.5)
  (h2 : 200 / p.costA = 2 * (75 / p.costB))
  (h3 : p.sellA = 13)
  (h4 : p.sellB = 9.5) :
  p.costA = 10 ∧ p.costB = 7.5 ∧
  (∀ a : ℕ, (p.sellA - p.costA) * a + (p.sellB - p.costB) * (2 * a + 4) > 120 → a ≥ 17) :=
sorry

end drawing_set_prices_and_quantity_l2873_287325


namespace surf_festival_problem_l2873_287331

/-- The Rip Curl Myrtle Beach Surf Festival problem -/
theorem surf_festival_problem (total_surfers : ℝ) (S1 : ℝ) :
  total_surfers = 15000 ∧
  S1 + 0.9 * S1 + 1.5 * S1 + (S1 + 0.9 * S1) + 0.5 * (S1 + 0.9 * S1) = total_surfers →
  S1 = 2400 ∧ total_surfers / 5 = 3000 := by
  sorry

end surf_festival_problem_l2873_287331


namespace venus_speed_conversion_l2873_287310

/-- Converts a speed from miles per second to miles per hour -/
def miles_per_second_to_miles_per_hour (speed_mps : ℝ) : ℝ :=
  speed_mps * 3600

/-- The speed of Venus around the sun in miles per second -/
def venus_speed_mps : ℝ := 21.9

theorem venus_speed_conversion :
  miles_per_second_to_miles_per_hour venus_speed_mps = 78840 := by
  sorry

end venus_speed_conversion_l2873_287310


namespace complex_magnitude_sum_squared_l2873_287376

theorem complex_magnitude_sum_squared : (Complex.abs (3 - 6*Complex.I) + Complex.abs (3 + 6*Complex.I))^2 = 180 := by
  sorry

end complex_magnitude_sum_squared_l2873_287376


namespace savings_calculation_l2873_287326

theorem savings_calculation (savings : ℚ) (tv_cost : ℚ) : 
  (1 / 4 : ℚ) * savings = tv_cost → 
  tv_cost = 300 → 
  savings = 1200 :=
by sorry

end savings_calculation_l2873_287326


namespace ellipse_b_squared_value_l2873_287385

/-- The squared semi-minor axis of an ellipse with equation (x^2/25) + (y^2/b^2) = 1,
    which has the same foci as a hyperbola with equation (x^2/225) - (y^2/144) = 1/36 -/
def ellipse_b_squared : ℝ := 14.75

/-- The equation of the ellipse -/
def is_on_ellipse (x y b : ℝ) : Prop :=
  x^2 / 25 + y^2 / b^2 = 1

/-- The equation of the hyperbola -/
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 225 - y^2 / 144 = 1 / 36

/-- The foci of the ellipse and hyperbola coincide -/
axiom foci_coincide : ∃ c : ℝ,
  c^2 = 25 - ellipse_b_squared ∧
  c^2 = 225 / 36 - 144 / 36

theorem ellipse_b_squared_value :
  ellipse_b_squared = 14.75 := by sorry

end ellipse_b_squared_value_l2873_287385


namespace sqrt_16_equals_4_l2873_287359

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_16_equals_4_l2873_287359


namespace genetic_material_distribution_l2873_287362

/-- Represents a cell in a diploid organism -/
structure DiploidCell where
  genetic_material : Set (α : Type)
  cytoplasm : Set (α : Type)

/-- Represents the process of cell division -/
def cell_division (parent : DiploidCell) : DiploidCell × DiploidCell :=
  sorry

/-- Predicate to check if the distribution is random and unequal -/
def is_random_unequal_distribution (parent : DiploidCell) (daughter1 daughter2 : DiploidCell) : Prop :=
  sorry

/-- Theorem stating that genetic material in cytoplasm is distributed randomly and unequally during cell division -/
theorem genetic_material_distribution (cell : DiploidCell) :
  let (daughter1, daughter2) := cell_division cell
  is_random_unequal_distribution cell daughter1 daughter2 := by
  sorry

end genetic_material_distribution_l2873_287362


namespace fraction_transformation_l2873_287306

theorem fraction_transformation (a b : ℝ) (h : a * b > 0) :
  (3 * a + 2 * (3 * b)) / (2 * (3 * a) * (3 * b)) = (1 / 3) * ((a + 2 * b) / (2 * a * b)) := by
  sorry

end fraction_transformation_l2873_287306


namespace square_fitting_theorem_l2873_287363

theorem square_fitting_theorem :
  ∃ (N : ℕ), N > 0 ∧ (N : ℝ) * (1 / N) ≤ 1 ∧ 4 * N = 1992 := by
  sorry

end square_fitting_theorem_l2873_287363


namespace point_transformation_l2873_287304

/-- Rotate a point (x, y) counterclockwise by 90° around (h, k) -/
def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

/-- Reflect a point (x, y) about the line y = x -/
def reflectYeqX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let p := (a, b)
  let rotated := rotate90 a b 2 3
  let reflected := reflectYeqX rotated.1 rotated.2
  reflected = (-4, 2) → b - a = -6 := by
  sorry

end point_transformation_l2873_287304


namespace geometric_sequence_problem_l2873_287327

/-- Given a geometric sequence {a_n} where a_2 and a_6 are roots of x^2 - 34x + 64 = 0, a_4 = 8 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- a is a geometric sequence
  (a 2 * a 2 - 34 * a 2 + 64 = 0) →  -- a_2 is a root of x^2 - 34x + 64 = 0
  (a 6 * a 6 - 34 * a 6 + 64 = 0) →  -- a_6 is a root of x^2 - 34x + 64 = 0
  (a 4 = 8) := by
sorry


end geometric_sequence_problem_l2873_287327


namespace distance_from_origin_l2873_287338

theorem distance_from_origin (x y : ℝ) (n : ℝ) : 
  y = 15 →
  (x - 5)^2 + (y - 8)^2 = 13^2 →
  x > 5 →
  n^2 = x^2 + y^2 →
  n = Real.sqrt (370 + 20 * Real.sqrt 30) :=
by sorry

end distance_from_origin_l2873_287338


namespace ear_muffs_bought_in_december_l2873_287312

theorem ear_muffs_bought_in_december (before_december : ℕ) (total : ℕ) 
  (h1 : before_december = 1346)
  (h2 : total = 7790) :
  total - before_december = 6444 :=
by sorry

end ear_muffs_bought_in_december_l2873_287312


namespace quadratic_root_existence_l2873_287388

theorem quadratic_root_existence (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (a^2 - 4*b ≥ 0) ∨ (c^2 - 4*d ≥ 0) := by
  sorry

end quadratic_root_existence_l2873_287388


namespace solve_exponential_equation_l2873_287360

theorem solve_exponential_equation :
  ∃ x : ℝ, (64 : ℝ)^(3*x) = (16 : ℝ)^(4*x - 5) ∧ x = -10 :=
by
  sorry

end solve_exponential_equation_l2873_287360


namespace fifty_ring_squares_l2873_287374

/-- Calculate the number of squares in the nth ring around a 2x1 rectangle --/
def ring_squares (n : ℕ) : ℕ :=
  let outer_width := 2 + 2 * n
  let outer_height := 1 + 2 * n
  let inner_width := 2 + 2 * (n - 1)
  let inner_height := 1 + 2 * (n - 1)
  outer_width * outer_height - inner_width * inner_height

/-- The 50th ring around a 2x1 rectangle contains 402 unit squares --/
theorem fifty_ring_squares : ring_squares 50 = 402 := by
  sorry

#eval ring_squares 50  -- This line is optional, for verification purposes

end fifty_ring_squares_l2873_287374


namespace mans_rowing_rate_l2873_287330

/-- Proves that a man's rowing rate in still water is 11 km/h given his speeds with and against the stream. -/
theorem mans_rowing_rate (with_stream : ℝ) (against_stream : ℝ)
  (h_with : with_stream = 18)
  (h_against : against_stream = 4) :
  (with_stream + against_stream) / 2 = 11 := by
  sorry

end mans_rowing_rate_l2873_287330


namespace pie_sale_profit_l2873_287332

/-- Calculates the profit from selling pies given the number of pies, costs, and selling price -/
theorem pie_sale_profit
  (num_pumpkin : ℕ)
  (num_cherry : ℕ)
  (cost_pumpkin : ℕ)
  (cost_cherry : ℕ)
  (selling_price : ℕ)
  (h1 : num_pumpkin = 10)
  (h2 : num_cherry = 12)
  (h3 : cost_pumpkin = 3)
  (h4 : cost_cherry = 5)
  (h5 : selling_price = 5) :
  (num_pumpkin + num_cherry) * selling_price - (num_pumpkin * cost_pumpkin + num_cherry * cost_cherry) = 20 :=
by sorry

end pie_sale_profit_l2873_287332


namespace circles_externally_tangent_l2873_287301

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

/-- The first circle: x^2 + y^2 = 4 -/
def circle1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- The second circle: x^2 + y^2 - 2mx + m^2 - 1 = 0 -/
def circle2 (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*m*p.1 + m^2 - 1 = 0}

theorem circles_externally_tangent :
  ∀ m : ℝ, (∃ p : ℝ × ℝ, p ∈ circle1 ∩ circle2 m) ∧
           externally_tangent (0, 0) (m, 0) 2 1 ↔ m = 3 ∨ m = -3 :=
sorry

end circles_externally_tangent_l2873_287301


namespace circle_intersection_range_l2873_287300

-- Define the circle C
def circle_C (a x y : ℝ) : Prop :=
  (x - a - 1)^2 + (y - Real.sqrt 3 * a)^2 = 1

-- Define the condition |MA| = 2|MO|
def condition_M (x y : ℝ) : Prop :=
  (x + 3)^2 + y^2 = 4 * (x^2 + y^2)

-- Define the range of a
def range_a (a : ℝ) : Prop :=
  (1/2 ≤ a ∧ a ≤ 3/2) ∨ (-3/2 ≤ a ∧ a ≤ -1/2)

theorem circle_intersection_range :
  ∀ a : ℝ, (∃ x y : ℝ, circle_C a x y ∧ condition_M x y) ↔ range_a a :=
by sorry

end circle_intersection_range_l2873_287300


namespace function_inequality_solution_l2873_287307

-- Define the function f
noncomputable def f (x : ℝ) (p q a b : ℝ) (h : ℝ → ℝ) : ℝ :=
  if q > 0 then
    (a * x) / (1 - q) + h x * q^x + b / (1 - q) - a * p / ((1 - q)^2)
  else
    (a * x) / (1 - q) + h x * (-q)^x + b / (1 - q) - a * p / ((1 - q)^2)

-- State the theorem
theorem function_inequality_solution (p q a b : ℝ) (h : ℝ → ℝ) :
  q ≠ 1 →
  (∀ x, q > 0 → h (x + p) ≥ h x) →
  (∀ x, q < 0 → h (x + p) ≥ -h x) →
  (∀ x, f (x + p) p q a b h - q * f x p q a b h ≥ a * x + b) ↔
  (∀ x, f x p q a b h = if q > 0 then
    (a * x) / (1 - q) + h x * q^x + b / (1 - q) - a * p / ((1 - q)^2)
  else
    (a * x) / (1 - q) + h x * (-q)^x + b / (1 - q) - a * p / ((1 - q)^2)) :=
by sorry

end function_inequality_solution_l2873_287307


namespace isbn_check_digit_l2873_287382

/-- Calculates the sum S for an ISBN --/
def calculate_S (A B C D E F G H I : ℕ) : ℕ :=
  10 * A + 9 * B + 8 * C + 7 * D + 6 * E + 5 * F + 4 * G + 3 * H + 2 * I

/-- Determines the check digit J based on the remainder r --/
def determine_J (r : ℕ) : ℕ :=
  if r = 0 then 0
  else if r = 1 then 10  -- Representing 'x' as 10
  else 11 - r

/-- Theorem: For the ISBN 962y707015, y = 7 --/
theorem isbn_check_digit (y : ℕ) (hy : y < 10) :
  let S := calculate_S 9 6 2 y 7 0 7 0 1
  let r := S % 11
  determine_J r = 5 → y = 7 := by
  sorry

end isbn_check_digit_l2873_287382


namespace A_simplified_A_value_when_x_plus_one_squared_is_six_l2873_287395

-- Define the polynomial A
def A (x : ℝ) : ℝ := (x + 2)^2 + (1 - x) * (2 + x) - 3

-- Theorem for the simplified form of A
theorem A_simplified (x : ℝ) : A x = 3 * x + 3 := by sorry

-- Theorem for the value of A when (x+1)^2 = 6
theorem A_value_when_x_plus_one_squared_is_six :
  ∃ x : ℝ, (x + 1)^2 = 6 ∧ (A x = 3 * Real.sqrt 6 ∨ A x = -3 * Real.sqrt 6) := by sorry

end A_simplified_A_value_when_x_plus_one_squared_is_six_l2873_287395


namespace divisibility_problem_l2873_287305

def solution_set : Set Int :=
  {-21, -9, -5, -3, -1, 1, 2, 4, 5, 6, 7, 9, 11, 15, 27}

theorem divisibility_problem :
  ∀ x : Int, x ≠ 3 ∧ (x - 3 ∣ x^3 - 3) ↔ x ∈ solution_set := by
  sorry

end divisibility_problem_l2873_287305


namespace car_journey_digit_squares_sum_l2873_287349

/-- Represents a car journey with specific odometer conditions -/
structure CarJourney where
  a : ℕ
  b : ℕ
  c : ℕ
  hours : ℕ
  initialReading : ℕ
  finalReading : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem car_journey_digit_squares_sum
  (journey : CarJourney)
  (h1 : journey.a ≥ 1)
  (h2 : journey.a + journey.b + journey.c = 9)
  (h3 : journey.initialReading = 100 * journey.a + 10 * journey.b + journey.c)
  (h4 : journey.finalReading = 100 * journey.c + 10 * journey.b + journey.a)
  (h5 : journey.finalReading - journey.initialReading = 65 * journey.hours) :
  journey.a^2 + journey.b^2 + journey.c^2 = 53 :=
sorry

end car_journey_digit_squares_sum_l2873_287349


namespace angle_A_value_max_area_l2873_287336

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  t.a * Real.cos t.C + (t.c - 2 * t.b) * Real.cos t.A = 0

-- Theorem 1: If the condition is satisfied, then A = π/3
theorem angle_A_value (t : Triangle) (h : satisfies_condition t) : t.A = π / 3 :=
sorry

-- Theorem 2: If a = 2 and the condition is satisfied, the maximum area is √3
theorem max_area (t : Triangle) (h1 : satisfies_condition t) (h2 : t.a = 2) :
  (∀ t' : Triangle, satisfies_condition t' → t'.a = 2 → 
    1 / 2 * t'.b * t'.c * Real.sin t'.A ≤ Real.sqrt 3) ∧
  (∃ t' : Triangle, satisfies_condition t' ∧ t'.a = 2 ∧
    1 / 2 * t'.b * t'.c * Real.sin t'.A = Real.sqrt 3) :=
sorry

end angle_A_value_max_area_l2873_287336


namespace triangle_angle_relation_l2873_287334

theorem triangle_angle_relation (A B C : ℝ) (h1 : Real.cos A + Real.sin B = 1) 
  (h2 : Real.sin A + Real.cos B = Real.sqrt 3) : Real.cos (A - C) = Real.sqrt 3 / 2 := by
  sorry

end triangle_angle_relation_l2873_287334


namespace lilys_calculation_l2873_287314

theorem lilys_calculation (a b c : ℝ) 
  (h1 : a - 2 * (b - 3 * c) = 14) 
  (h2 : a - 2 * b - 3 * c = 2) : 
  a - 2 * b = 6 := by sorry

end lilys_calculation_l2873_287314


namespace lunch_packing_ratio_l2873_287371

def school_days : ℕ := 180
def aliyah_lunch_days : ℕ := school_days / 2
def becky_lunch_days : ℕ := 45

theorem lunch_packing_ratio :
  becky_lunch_days / aliyah_lunch_days = 1 / 2 := by
  sorry

end lunch_packing_ratio_l2873_287371


namespace perfect_cube_prime_factor_addition_l2873_287323

theorem perfect_cube_prime_factor_addition (x : ℕ) : ∃ x, 
  (27 = 3^3) ∧ 
  (∃ p : ℕ, Prime p ∧ p = 3 + x) ∧ 
  x = 2 := by
sorry

end perfect_cube_prime_factor_addition_l2873_287323


namespace find_S_l2873_287328

theorem find_S : ∃ S : ℝ, 
  (∀ a b c d : ℝ, 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧ d ≠ 1 ∧
    a + b + c + d = S ∧
    1/a + 1/b + 1/c + 1/d = S ∧
    1/(a-1) + 1/(b-1) + 1/(c-1) + 1/(d-1) = S) →
  S = -2 :=
by sorry

end find_S_l2873_287328


namespace smallest_n_for_pencil_paradox_l2873_287380

theorem smallest_n_for_pencil_paradox : ∃ n : ℕ, n = 100 ∧ 
  (∀ m : ℕ, m < n → 
    ¬(∃ a b c d : ℕ, 
      6 * a + 10 * b = m ∧ 
      6 * c + 10 * d = m + 2 ∧ 
      7 * a + 12 * b > 7 * c + 12 * d)) ∧
  (∃ a b c d : ℕ, 
    6 * a + 10 * b = n ∧ 
    6 * c + 10 * d = n + 2 ∧ 
    7 * a + 12 * b > 7 * c + 12 * d) := by
  sorry

end smallest_n_for_pencil_paradox_l2873_287380


namespace opposite_of_negative_sqrt_three_l2873_287317

theorem opposite_of_negative_sqrt_three : -(-(Real.sqrt 3)) = Real.sqrt 3 := by
  sorry

end opposite_of_negative_sqrt_three_l2873_287317


namespace circle_radius_range_l2873_287337

/-- The set of points (x, y) satisfying the given equation -/
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.sin (3 * p.1 + 4 * p.2) = Real.sin (3 * p.1) + Real.sin (4 * p.2)}

/-- A circle with center c and radius r -/
def Circle (c : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}

/-- The theorem stating the range of possible radii for non-intersecting circles -/
theorem circle_radius_range (c : ℝ × ℝ) (r : ℝ) :
  (∀ p ∈ M, p ∉ Circle c r) → 0 < r ∧ r < Real.pi / 6 := by
  sorry


end circle_radius_range_l2873_287337


namespace r_fourth_plus_inverse_l2873_287348

theorem r_fourth_plus_inverse (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end r_fourth_plus_inverse_l2873_287348


namespace expression_simplification_l2873_287391

theorem expression_simplification (x y : ℝ) :
  let P := 2 * x + 3 * y
  let Q := 3 * x + 2 * y
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = (24 * x^2 + 52 * x * y + 24 * y^2) / (5 * x * y - 5 * y^2) :=
by sorry

end expression_simplification_l2873_287391


namespace part_one_part_two_l2873_287372

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x - m < 0}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Part I
theorem part_one (m : ℝ) (h : m = 3) : A ∩ (U \ B m) = Set.Icc 3 4 := by sorry

-- Part II
theorem part_two (m : ℝ) (h : A ∩ B m = ∅) : m ≤ -2 := by sorry

end part_one_part_two_l2873_287372
