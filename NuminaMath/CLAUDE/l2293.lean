import Mathlib

namespace NUMINAMATH_CALUDE_lunch_to_reading_time_ratio_l2293_229311

theorem lunch_to_reading_time_ratio
  (book_pages : ℕ)
  (pages_per_hour : ℕ)
  (lunch_time : ℕ)
  (h1 : book_pages = 4000)
  (h2 : pages_per_hour = 250)
  (h3 : lunch_time = 4) :
  (lunch_time : ℚ) / ((book_pages : ℚ) / (pages_per_hour : ℚ)) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_lunch_to_reading_time_ratio_l2293_229311


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l2293_229377

theorem average_of_remaining_numbers
  (total : ℕ)
  (avg_all : ℚ)
  (avg_first_two : ℚ)
  (avg_next_two : ℚ)
  (h_total : total = 6)
  (h_avg_all : avg_all = 5/2)
  (h_avg_first_two : avg_first_two = 11/10)
  (h_avg_next_two : avg_next_two = 14/10) :
  (total * avg_all - 2 * avg_first_two - 2 * avg_next_two) / 2 = 5 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l2293_229377


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2293_229398

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x ≥ 1 → |x + 1| + |x - 1| = 2 * |x|) ∧
  ¬(|x + 1| + |x - 1| = 2 * |x| → x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2293_229398


namespace NUMINAMATH_CALUDE_line_intercept_theorem_l2293_229333

/-- Given a line ax - 6y - 12a = 0 where a ≠ 0, if its x-intercept is three times its y-intercept, then a = -2 -/
theorem line_intercept_theorem (a : ℝ) (h1 : a ≠ 0) : 
  (∃ x y : ℝ, a * x - 6 * y - 12 * a = 0 ∧ 
   x = 3 * y ∧ 
   (∀ x' y' : ℝ, a * x' - 6 * y' - 12 * a = 0 → (x' = 0 ∨ y' = 0) → (x' = x ∨ y' = y))) → 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_line_intercept_theorem_l2293_229333


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_16_l2293_229336

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_16 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 16 → n ≤ 82 :=
sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_16_l2293_229336


namespace NUMINAMATH_CALUDE_lemonade_sales_profit_difference_l2293_229362

/-- Lemonade sales problem -/
theorem lemonade_sales_profit_difference : 
  let katya_glasses : ℕ := 8
  let katya_price : ℚ := 3/2
  let katya_cost : ℚ := 1/2
  let ricky_glasses : ℕ := 9
  let ricky_price : ℚ := 2
  let ricky_cost : ℚ := 3/4
  let tina_price : ℚ := 3
  let tina_cost : ℚ := 1
  
  let katya_revenue := katya_glasses * katya_price
  let ricky_revenue := ricky_glasses * ricky_price
  let combined_revenue := katya_revenue + ricky_revenue
  let tina_target := 2 * combined_revenue
  
  let katya_profit := katya_revenue - (katya_glasses : ℚ) * katya_cost
  let tina_glasses := tina_target / tina_price
  let tina_profit := tina_target - tina_glasses * tina_cost
  
  tina_profit - katya_profit = 32
  := by sorry

end NUMINAMATH_CALUDE_lemonade_sales_profit_difference_l2293_229362


namespace NUMINAMATH_CALUDE_isosceles_triangle_l2293_229355

theorem isosceles_triangle (A B C : ℝ) (hsum : A + B + C = π) 
  (h : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) :
  A = B :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l2293_229355


namespace NUMINAMATH_CALUDE_mix_paint_intensity_theorem_l2293_229353

/-- Calculates the intensity of a paint mixture after replacing a portion of the original paint with a different intensity paint. -/
def mixPaintIntensity (originalIntensity replacementIntensity fractionReplaced : ℚ) : ℚ :=
  (1 - fractionReplaced) * originalIntensity + fractionReplaced * replacementIntensity

/-- Theorem stating that mixing 10% intensity paint with 20% intensity paint in equal proportions results in 15% intensity. -/
theorem mix_paint_intensity_theorem :
  mixPaintIntensity (1/10) (1/5) (1/2) = (3/20) := by
  sorry

#eval mixPaintIntensity (1/10) (1/5) (1/2)

end NUMINAMATH_CALUDE_mix_paint_intensity_theorem_l2293_229353


namespace NUMINAMATH_CALUDE_pin_permutations_l2293_229392

theorem pin_permutations : 
  let n : ℕ := 4
  ∀ (digits : Finset ℕ), Finset.card digits = n → Finset.card (Finset.powersetCard n digits) = n.factorial :=
by
  sorry

end NUMINAMATH_CALUDE_pin_permutations_l2293_229392


namespace NUMINAMATH_CALUDE_future_age_difference_l2293_229340

/-- Represents the age difference between Kaylee and Matt in the future -/
def AgeDifference (x : ℕ) : Prop :=
  (8 + x) = 3 * 5

/-- Proves that the number of years into the future when Kaylee will be 3 times as old as Matt is now is 7 years -/
theorem future_age_difference : ∃ (x : ℕ), AgeDifference x ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_future_age_difference_l2293_229340


namespace NUMINAMATH_CALUDE_non_square_sequence_250th_term_l2293_229325

/-- The sequence of positive integers omitting perfect squares -/
def non_square_sequence : ℕ → ℕ := sorry

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- The 250th term of the non-square sequence -/
def term_250 : ℕ := non_square_sequence 250

theorem non_square_sequence_250th_term :
  term_250 = 265 := by sorry

end NUMINAMATH_CALUDE_non_square_sequence_250th_term_l2293_229325


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_reciprocal_l2293_229304

theorem cubic_roots_sum_of_squares_reciprocal (a b c : ℝ) 
  (sum_eq : a + b + c = 12)
  (sum_prod_eq : a * b + b * c + c * a = 20)
  (prod_eq : a * b * c = -5) :
  1 / a^2 + 1 / b^2 + 1 / c^2 = 20.8 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_reciprocal_l2293_229304


namespace NUMINAMATH_CALUDE_abc_cba_divisibility_l2293_229345

theorem abc_cba_divisibility (a : ℕ) (h : a ≤ 7) :
  ∃ k : ℕ, 100 * a + 10 * (a + 1) + (a + 2) + 100 * (a + 2) + 10 * (a + 1) + a = 212 * k := by
  sorry

end NUMINAMATH_CALUDE_abc_cba_divisibility_l2293_229345


namespace NUMINAMATH_CALUDE_clock_shows_four_fifty_l2293_229331

/-- Represents a clock hand --/
inductive ClockHand
| A
| B
| C

/-- Represents the position of a clock hand --/
structure HandPosition where
  hand : ClockHand
  exactHourMarker : Bool

/-- Represents a clock with three hands --/
structure Clock where
  hands : List HandPosition
  handsEqualLength : Bool

/-- Theorem stating that given the specific clock configuration, the time shown is 4:50 --/
theorem clock_shows_four_fifty (c : Clock) 
  (h1 : c.handsEqualLength = true)
  (h2 : c.hands.length = 3)
  (h3 : ∃ h ∈ c.hands, h.hand = ClockHand.A ∧ h.exactHourMarker = true)
  (h4 : ∃ h ∈ c.hands, h.hand = ClockHand.B ∧ h.exactHourMarker = true)
  (h5 : ∃ h ∈ c.hands, h.hand = ClockHand.C ∧ h.exactHourMarker = false) :
  ∃ (hour : Nat) (minute : Nat), hour = 4 ∧ minute = 50 := by
  sorry


end NUMINAMATH_CALUDE_clock_shows_four_fifty_l2293_229331


namespace NUMINAMATH_CALUDE_min_mozart_bach_not_beethoven_l2293_229397

def Universe := 200
def Mozart := 150
def Bach := 120
def Beethoven := 90

theorem min_mozart_bach_not_beethoven :
  ∃ (m b e mb mbe : ℕ),
    m ≤ Mozart ∧
    b ≤ Bach ∧
    e ≤ Beethoven ∧
    mb ≤ m ∧
    mb ≤ b ∧
    mbe ≤ mb ∧
    mbe ≤ e ∧
    m + b - mb ≤ Universe ∧
    m + b + e - mb - mbe ≤ Universe ∧
    mb - mbe ≥ 10 :=
  sorry

end NUMINAMATH_CALUDE_min_mozart_bach_not_beethoven_l2293_229397


namespace NUMINAMATH_CALUDE_watch_correction_l2293_229379

/-- The number of days from April 1 at 12 noon to April 10 at 6 P.M. -/
def days_passed : ℚ := 9 + 6 / 24

/-- The rate at which the watch loses time, in minutes per day -/
def loss_rate : ℚ := 3

/-- The positive correction in minutes to be added to the watch -/
def correction (d : ℚ) (r : ℚ) : ℚ := d * r

theorem watch_correction :
  correction days_passed loss_rate = 27.75 := by
  sorry

end NUMINAMATH_CALUDE_watch_correction_l2293_229379


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2293_229367

/-- Given a geometric sequence {a_n} with common ratio q,
    if a_1 + a_3 = 10 and a_4 + a_6 = 5/4, then q = 1/2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  a 1 + a 3 = 10 →                  -- first given condition
  a 4 + a 6 = 5/4 →                 -- second given condition
  q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2293_229367


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2293_229342

theorem pure_imaginary_fraction (b : ℝ) : 
  (Complex.I * (((1 : ℂ) + b * Complex.I) / ((2 : ℂ) - Complex.I))).re = 0 → 
  (((1 : ℂ) + b * Complex.I) / ((2 : ℂ) - Complex.I)).im ≠ 0 → 
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2293_229342


namespace NUMINAMATH_CALUDE_x_value_proof_l2293_229301

theorem x_value_proof (x : ℝ) (h : (x / 6) / 3 = 9 / (x / 3)) : x = 3 * Real.sqrt 54 ∨ x = -3 * Real.sqrt 54 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2293_229301


namespace NUMINAMATH_CALUDE_player_A_can_win_l2293_229323

/-- Represents a game board with three rows --/
structure GameBoard :=
  (row1 : List ℤ)
  (row2 : List ℤ)
  (row3 : List ℤ)

/-- Represents a player in the game --/
inductive Player
  | A
  | B

/-- Defines a valid game board configuration --/
def ValidBoard (board : GameBoard) : Prop :=
  Odd board.row1.length ∧ 
  Odd board.row2.length ∧ 
  Odd board.row3.length

/-- Defines the game state --/
structure GameState :=
  (board : GameBoard)
  (currentPlayer : Player)

/-- Defines a game strategy for player A --/
def Strategy := GameState → ℕ → ℤ → GameState

/-- Theorem: Player A can always achieve the desired row sums --/
theorem player_A_can_win (initialBoard : GameBoard) (targetSum1 targetSum2 targetSum3 : ℤ) :
  ValidBoard initialBoard →
  ∃ (strategy : Strategy),
    (∀ (finalBoard : GameBoard),
      (finalBoard.row1.sum = targetSum1) ∧
      (finalBoard.row2.sum = targetSum2) ∧
      (finalBoard.row3.sum = targetSum3)) :=
sorry

end NUMINAMATH_CALUDE_player_A_can_win_l2293_229323


namespace NUMINAMATH_CALUDE_igloo_construction_l2293_229393

def igloo_bricks (n : ℕ) : ℕ :=
  if n ≤ 6 then
    14 + 2 * (n - 1)
  else
    24 - 3 * (n - 6)

def total_bricks : ℕ := (List.range 10).map (λ i => igloo_bricks (i + 1)) |>.sum

theorem igloo_construction :
  total_bricks = 170 := by
  sorry

end NUMINAMATH_CALUDE_igloo_construction_l2293_229393


namespace NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l2293_229382

/-- The ratio of the volume of water in a cone filled to 2/3 of its height to the total volume of the cone is 8/27. -/
theorem water_volume_ratio_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let water_height := (2 : ℝ) / 3 * h
  let water_radius := (2 : ℝ) / 3 * r
  let cone_volume := (1 : ℝ) / 3 * π * r^2 * h
  let water_volume := (1 : ℝ) / 3 * π * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 := by
sorry


end NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l2293_229382


namespace NUMINAMATH_CALUDE_house_rent_percentage_l2293_229365

def monthly_salary : ℝ := 12500
def food_percentage : ℝ := 40
def entertainment_percentage : ℝ := 10
def conveyance_percentage : ℝ := 10
def savings : ℝ := 2500

theorem house_rent_percentage :
  let total_percentage : ℝ := food_percentage + entertainment_percentage + conveyance_percentage
  let spent_amount : ℝ := monthly_salary - savings
  let savings_percentage : ℝ := (savings / monthly_salary) * 100
  let remaining_percentage : ℝ := 100 - total_percentage - savings_percentage
  remaining_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_house_rent_percentage_l2293_229365


namespace NUMINAMATH_CALUDE_g_neg_four_l2293_229314

def g (x : ℝ) : ℝ := 5 * x + 2

theorem g_neg_four : g (-4) = -18 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_four_l2293_229314


namespace NUMINAMATH_CALUDE_armands_guessing_game_l2293_229380

theorem armands_guessing_game : ∃ x : ℕ, x = 33 ∧ 3 * x = 2 * 51 - 3 := by
  sorry

end NUMINAMATH_CALUDE_armands_guessing_game_l2293_229380


namespace NUMINAMATH_CALUDE_city_rentals_rate_proof_l2293_229354

/-- The daily rate for Safety Rent-a-Car in dollars -/
def safety_daily_rate : ℝ := 21.95

/-- The per-mile rate for Safety Rent-a-Car in dollars -/
def safety_mile_rate : ℝ := 0.19

/-- The daily rate for City Rentals in dollars -/
def city_daily_rate : ℝ := 18.95

/-- The number of miles driven -/
def miles_driven : ℝ := 150

/-- The per-mile rate for City Rentals in dollars -/
def city_mile_rate : ℝ := 0.21

theorem city_rentals_rate_proof :
  safety_daily_rate + safety_mile_rate * miles_driven =
  city_daily_rate + city_mile_rate * miles_driven :=
by sorry

end NUMINAMATH_CALUDE_city_rentals_rate_proof_l2293_229354


namespace NUMINAMATH_CALUDE_commute_time_difference_l2293_229387

/-- Proves that the difference in commute time between walking and taking the train is 25 minutes -/
theorem commute_time_difference
  (distance : Real)
  (walking_speed : Real)
  (train_speed : Real)
  (additional_train_time : Real)
  (h1 : distance = 1.5)
  (h2 : walking_speed = 3)
  (h3 : train_speed = 20)
  (h4 : additional_train_time = 0.5 / 60) :
  (distance / walking_speed - (distance / train_speed + additional_train_time)) * 60 = 25 := by
  sorry

end NUMINAMATH_CALUDE_commute_time_difference_l2293_229387


namespace NUMINAMATH_CALUDE_spinner_probabilities_l2293_229329

theorem spinner_probabilities :
  ∀ (p_C : ℚ),
  (1 / 4 : ℚ) + (1 / 3 : ℚ) + p_C + p_C = 1 →
  p_C = (5 / 24 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_spinner_probabilities_l2293_229329


namespace NUMINAMATH_CALUDE_probability_of_event_B_l2293_229391

theorem probability_of_event_B 
  (A B : Set ℝ) 
  (P : Set ℝ → ℝ) 
  (h1 : P (A ∩ B) = 0.25)
  (h2 : P (A ∪ B) = 0.6)
  (h3 : P A = 0.45) :
  P B = 0.4 := by
sorry

end NUMINAMATH_CALUDE_probability_of_event_B_l2293_229391


namespace NUMINAMATH_CALUDE_hiking_team_participants_l2293_229374

/-- The number of gloves needed for the hiking team -/
def total_gloves : ℕ := 126

/-- The number of gloves each participant needs -/
def gloves_per_participant : ℕ := 2

/-- The number of participants in the hiking team -/
def num_participants : ℕ := total_gloves / gloves_per_participant

theorem hiking_team_participants : num_participants = 63 := by
  sorry

end NUMINAMATH_CALUDE_hiking_team_participants_l2293_229374


namespace NUMINAMATH_CALUDE_project_time_allocation_l2293_229341

theorem project_time_allocation (total_time research_time proposal_time : ℕ) 
  (h1 : total_time = 20)
  (h2 : research_time = 10)
  (h3 : proposal_time = 2) :
  total_time - (research_time + proposal_time) = 8 := by
  sorry

end NUMINAMATH_CALUDE_project_time_allocation_l2293_229341


namespace NUMINAMATH_CALUDE_correct_observation_value_l2293_229373

theorem correct_observation_value
  (n : ℕ)
  (original_mean : ℚ)
  (incorrect_value : ℚ)
  (corrected_mean : ℚ)
  (h1 : n = 50)
  (h2 : original_mean = 36)
  (h3 : incorrect_value = 23)
  (h4 : corrected_mean = 36.5) :
  (n : ℚ) * corrected_mean - ((n : ℚ) * original_mean - incorrect_value) = 48 :=
by sorry

end NUMINAMATH_CALUDE_correct_observation_value_l2293_229373


namespace NUMINAMATH_CALUDE_number_division_problem_l2293_229360

theorem number_division_problem (n : ℕ) : 
  n % 8 = 2 ∧ n / 8 = 156 → n / 5 - 3 = 247 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2293_229360


namespace NUMINAMATH_CALUDE_solve_for_m_l2293_229338

/-- Given that (x, y) = (2, -3) is a solution of the equation mx + 3y = 1, prove that m = 5 -/
theorem solve_for_m (x y m : ℝ) (h1 : x = 2) (h2 : y = -3) (h3 : m * x + 3 * y = 1) : m = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l2293_229338


namespace NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_65_l2293_229384

theorem right_triangle_with_hypotenuse_65 :
  ∃ (a b : ℕ), 
    a < b ∧ 
    a^2 + b^2 = 65^2 ∧ 
    a = 16 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_65_l2293_229384


namespace NUMINAMATH_CALUDE_polynomial_value_l2293_229376

/-- A polynomial of degree 5 with integer coefficients -/
def polynomial (a₁ a₂ a₃ a₄ a₅ : ℤ) (x : ℝ) : ℝ :=
  x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅

theorem polynomial_value (a₁ a₂ a₃ a₄ a₅ : ℤ) :
  let f := polynomial a₁ a₂ a₃ a₄ a₅
  (f (Real.sqrt 3 + Real.sqrt 2) = 0) →
  (f 1 + f 3 = 0) →
  (f (-1) = 24) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l2293_229376


namespace NUMINAMATH_CALUDE_five_is_solution_l2293_229371

/-- The equation we're working with -/
def equation (x : ℝ) : Prop :=
  x^3 + 2*(x+1)^3 + 3*(x+2)^3 = 3*(x+3)^3

/-- Theorem stating that 5 is a solution to the equation -/
theorem five_is_solution : equation 5 := by
  sorry

end NUMINAMATH_CALUDE_five_is_solution_l2293_229371


namespace NUMINAMATH_CALUDE_cover_cost_is_77_l2293_229306

/-- Represents the cost of printing a book in kopecks -/
def book_cost (cover_cost : ℕ) (page_cost : ℕ) (num_pages : ℕ) : ℕ :=
  (cover_cost * 100 + page_cost * num_pages + 99) / 100 * 100

/-- The problem statement -/
theorem cover_cost_is_77 : 
  ∃ (cover_cost page_cost : ℕ),
    (∀ n, book_cost cover_cost page_cost n = ((cover_cost * 100 + page_cost * n + 99) / 100) * 100) ∧
    book_cost cover_cost page_cost 104 = 134 * 100 ∧
    book_cost cover_cost page_cost 192 = 181 * 100 ∧
    cover_cost = 77 :=
by
  sorry

end NUMINAMATH_CALUDE_cover_cost_is_77_l2293_229306


namespace NUMINAMATH_CALUDE_root_sum_theorem_l2293_229302

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x

-- Define the theorem
theorem root_sum_theorem (h k : ℝ) 
  (h_root : p h = 1) 
  (k_root : p k = 5) : 
  h + k = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2293_229302


namespace NUMINAMATH_CALUDE_least_n_for_inequality_l2293_229320

theorem least_n_for_inequality : 
  ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, k > 0 → (1 / k - 1 / (k + 1) < 1 / 15) → k ≥ n) ∧ (1 / n - 1 / (n + 1) < 1 / 15) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_n_for_inequality_l2293_229320


namespace NUMINAMATH_CALUDE_triangle_tangent_range_l2293_229388

theorem triangle_tangent_range (a b c : ℝ) (A B C : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 0 < A ∧ A < π) (h5 : 0 < B ∧ B < π) (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π) (h8 : a^2 + b^2 + Real.sqrt 2 * a * b = c^2) :
  0 < Real.tan A * Real.tan (2 * B) ∧ Real.tan A * Real.tan (2 * B) < 1/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_tangent_range_l2293_229388


namespace NUMINAMATH_CALUDE_max_value_a_l2293_229351

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 4 * c)
  (h3 : c < 5 * d)
  (h4 : d < 50) :
  a ≤ 2924 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 2924 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 50 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l2293_229351


namespace NUMINAMATH_CALUDE_inches_per_foot_l2293_229363

theorem inches_per_foot (rope_first : ℕ) (rope_difference : ℕ) (total_inches : ℕ) : 
  rope_first = 6 →
  rope_difference = 4 →
  total_inches = 96 →
  (total_inches / (rope_first + (rope_first - rope_difference))) = 12 :=
by sorry

end NUMINAMATH_CALUDE_inches_per_foot_l2293_229363


namespace NUMINAMATH_CALUDE_square_difference_305_295_l2293_229399

theorem square_difference_305_295 : 305^2 - 295^2 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_305_295_l2293_229399


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_is_180_l2293_229352

-- Define a triangle in Euclidean space
structure EuclideanTriangle where
  -- We don't need to specify the exact properties of a triangle here
  -- as we're focusing on the angle sum property

-- Define the concept of interior angles of a triangle
def interior_angles (t : EuclideanTriangle) : ℝ := sorry

-- State the theorem about the sum of interior angles
theorem sum_of_interior_angles_is_180 (t : EuclideanTriangle) :
  interior_angles t = 180 := by sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_is_180_l2293_229352


namespace NUMINAMATH_CALUDE_prime_divisors_inequality_l2293_229343

-- Define the variables
variable (x y z : ℕ)
variable (p q : ℕ)

-- Define the conditions
variable (h1 : x > 2)
variable (h2 : y > 1)
variable (h3 : z > 0)
variable (h4 : x^y + 1 = z^2)

-- Define p and q
variable (hp : p = (Nat.factors x).card)
variable (hq : q = (Nat.factors y).card)

-- State the theorem
theorem prime_divisors_inequality : p ≥ q + 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisors_inequality_l2293_229343


namespace NUMINAMATH_CALUDE_jamal_storage_solution_l2293_229326

/-- Represents the storage problem with given file sizes and constraints -/
structure StorageProblem where
  total_files : ℕ
  disk_capacity : ℚ
  files_085 : ℕ
  files_075 : ℕ
  files_045 : ℕ
  no_mix_constraint : Bool

/-- Calculates the minimum number of disks needed for the given storage problem -/
def min_disks_needed (p : StorageProblem) : ℕ :=
  sorry

/-- The specific storage problem instance -/
def jamal_storage : StorageProblem :=
  { total_files := 36
  , disk_capacity := 1.44
  , files_085 := 5
  , files_075 := 15
  , files_045 := 16
  , no_mix_constraint := true }

/-- Theorem stating that the minimum number of disks needed for Jamal's storage problem is 24 -/
theorem jamal_storage_solution :
  min_disks_needed jamal_storage = 24 :=
  sorry

end NUMINAMATH_CALUDE_jamal_storage_solution_l2293_229326


namespace NUMINAMATH_CALUDE_exists_odd_64digit_no_zeros_div_101_l2293_229368

/-- A 64-digit natural number -/
def Digit64 : Type := { n : ℕ // n ≥ 10^63 ∧ n < 10^64 }

/-- Predicate for numbers not containing zeros -/
def NoZeros (n : ℕ) : Prop := ∀ d : ℕ, d < 64 → (n / 10^d) % 10 ≠ 0

/-- Theorem stating the existence of an odd 64-digit number without zeros that is divisible by 101 -/
theorem exists_odd_64digit_no_zeros_div_101 :
  ∃ (n : Digit64), NoZeros n.val ∧ n.val % 101 = 0 ∧ n.val % 2 = 1 := by sorry

end NUMINAMATH_CALUDE_exists_odd_64digit_no_zeros_div_101_l2293_229368


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_solution_set_is_real_l2293_229372

-- Define the inequality function
def f (a : ℝ) (x : ℝ) : ℝ := |a*x - 2| + |a*x - a|

-- Theorem 1: Solution set when a = 1
theorem solution_set_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 2 ↔ x ≥ 2.5 ∨ x ≤ 0.5 := by sorry

-- Theorem 2: Range of a when solution set is ℝ
theorem range_of_a_when_solution_set_is_real :
  (∀ a : ℝ, a > 0 → (∀ x : ℝ, f a x ≥ 2)) → (∀ a : ℝ, a > 0 → a ≥ 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_solution_set_is_real_l2293_229372


namespace NUMINAMATH_CALUDE_jamie_peeled_24_l2293_229324

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  sylvia_rate : ℕ
  jamie_rate : ℕ
  sylvia_solo_time : ℕ

/-- Calculates the number of potatoes Jamie peeled -/
def jamie_peeled (scenario : PotatoPeeling) : ℕ :=
  let sylvia_solo := scenario.sylvia_rate * scenario.sylvia_solo_time
  let remaining := scenario.total_potatoes - sylvia_solo
  let combined_rate := scenario.sylvia_rate + scenario.jamie_rate
  let combined_time := remaining / combined_rate
  scenario.jamie_rate * combined_time

/-- Theorem stating that Jamie peeled 24 potatoes -/
theorem jamie_peeled_24 (scenario : PotatoPeeling) 
    (h1 : scenario.total_potatoes = 60)
    (h2 : scenario.sylvia_rate = 4)
    (h3 : scenario.jamie_rate = 6)
    (h4 : scenario.sylvia_solo_time = 5) : 
  jamie_peeled scenario = 24 := by
  sorry

end NUMINAMATH_CALUDE_jamie_peeled_24_l2293_229324


namespace NUMINAMATH_CALUDE_expression_value_l2293_229349

theorem expression_value (x : ℤ) (h : x = -2) : 4 * x - 5 = -13 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2293_229349


namespace NUMINAMATH_CALUDE_product_sum_bounds_l2293_229350

def pairProductSum (pairs : List (ℕ × ℕ)) : ℕ :=
  (pairs.map (λ (a, b) => a * b)).sum

theorem product_sum_bounds :
  ∀ (pairs : List (ℕ × ℕ)),
    pairs.length = 50 ∧
    (pairs.map Prod.fst ++ pairs.map Prod.snd).toFinset = Finset.range 100
    →
    85850 ≤ pairProductSum pairs ∧ pairProductSum pairs ≤ 169150 :=
by sorry

end NUMINAMATH_CALUDE_product_sum_bounds_l2293_229350


namespace NUMINAMATH_CALUDE_f_properties_l2293_229364

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define g' as the derivative of g
variable (g' : ℝ → ℝ)

-- State the conditions
axiom cond1 : ∀ x, f x + g' x - 10 = 0
axiom cond2 : ∀ x, f x - g' (4 - x) - 10 = 0
axiom g_even : ∀ x, g (-x) = g x

-- State the theorem
theorem f_properties :
  (f 1 + f 3 = 20) ∧ (f 4 = 10) ∧ (f 2022 = 10) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2293_229364


namespace NUMINAMATH_CALUDE_flea_misses_point_l2293_229332

/-- The number of points on the circle -/
def n : ℕ := 300

/-- The set of all points on the circle -/
def Circle := Fin n

/-- The jumping pattern of the flea -/
def jump (k : ℕ) : ℕ := k * (k + 1) / 2

/-- The set of points visited by the flea -/
def VisitedPoints : Set Circle :=
  {p | ∃ k : ℕ, p = ⟨jump k % n, sorry⟩}

/-- Theorem stating that there exists a point the flea never visits -/
theorem flea_misses_point : ∃ p : Circle, p ∉ VisitedPoints := by
  sorry

end NUMINAMATH_CALUDE_flea_misses_point_l2293_229332


namespace NUMINAMATH_CALUDE_factorization_equality_l2293_229327

theorem factorization_equality (a b : ℝ) : (a - b)^2 - (b - a) = (a - b) * ((a - b) + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2293_229327


namespace NUMINAMATH_CALUDE_hosing_time_is_10_minutes_l2293_229357

def dog_cleaning_time (num_shampoos : ℕ) (time_per_shampoo : ℕ) (total_cleaning_time : ℕ) : ℕ :=
  total_cleaning_time - (num_shampoos * time_per_shampoo)

theorem hosing_time_is_10_minutes :
  dog_cleaning_time 3 15 55 = 10 := by
  sorry

end NUMINAMATH_CALUDE_hosing_time_is_10_minutes_l2293_229357


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_primes_l2293_229308

/-- A function that returns true if a number is composite, false otherwise -/
def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that returns true if a number has no prime factors less than 20, false otherwise -/
def no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 20 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_primes : 
  (is_composite 529 ∧ no_small_prime_factors 529) ∧ 
  (∀ m : ℕ, m < 529 → ¬(is_composite m ∧ no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_primes_l2293_229308


namespace NUMINAMATH_CALUDE_min_distance_line_parabola_l2293_229395

/-- The minimum distance between a point on the line y = (12/5)x - 5 and a point on the parabola y = x^2 is 89/65 -/
theorem min_distance_line_parabola :
  let line := fun (x : ℝ) => (12/5) * x - 5
  let parabola := fun (x : ℝ) => x^2
  let distance := fun (x₁ x₂ : ℝ) => 
    Real.sqrt ((x₂ - x₁)^2 + (parabola x₂ - line x₁)^2)
  (∃ (x₁ x₂ : ℝ), ∀ (y₁ y₂ : ℝ), distance x₁ x₂ ≤ distance y₁ y₂) ∧
  (∃ (x₁ x₂ : ℝ), distance x₁ x₂ = 89/65) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_line_parabola_l2293_229395


namespace NUMINAMATH_CALUDE_total_marbles_relation_l2293_229389

/-- Represents the number of marbles of each color -/
structure MarbleCollection where
  red : ℝ
  blue : ℝ
  green : ℝ

/-- Conditions for the marble collection -/
def validCollection (c : MarbleCollection) : Prop :=
  c.red = 1.4 * c.blue ∧ c.green = 1.5 * c.red

/-- Total number of marbles in the collection -/
def totalMarbles (c : MarbleCollection) : ℝ :=
  c.red + c.blue + c.green

/-- Theorem stating the relationship between total marbles and red marbles -/
theorem total_marbles_relation (c : MarbleCollection) (h : validCollection c) :
    totalMarbles c = 3.21 * c.red := by
  sorry

#check total_marbles_relation

end NUMINAMATH_CALUDE_total_marbles_relation_l2293_229389


namespace NUMINAMATH_CALUDE_smallest_stairs_l2293_229315

theorem smallest_stairs (n : ℕ) : 
  (n > 15) ∧ 
  (n % 6 = 4) ∧ 
  (n % 7 = 3) ∧ 
  (∀ m : ℕ, m > 15 ∧ m % 6 = 4 ∧ m % 7 = 3 → m ≥ n) → 
  n = 52 := by
sorry

end NUMINAMATH_CALUDE_smallest_stairs_l2293_229315


namespace NUMINAMATH_CALUDE_chocolate_count_l2293_229396

/-- The number of large boxes in the massive crate -/
def large_boxes : ℕ := 54

/-- The number of small boxes in each large box -/
def small_boxes_per_large : ℕ := 24

/-- The number of chocolate bars in each small box -/
def chocolates_per_small : ℕ := 37

/-- The total number of chocolate bars in the massive crate -/
def total_chocolates : ℕ := large_boxes * small_boxes_per_large * chocolates_per_small

theorem chocolate_count : total_chocolates = 47952 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_count_l2293_229396


namespace NUMINAMATH_CALUDE_height_on_longest_side_of_6_8_10_triangle_l2293_229359

theorem height_on_longest_side_of_6_8_10_triangle :
  ∃ (a b c h : ℝ),
    a = 6 ∧ b = 8 ∧ c = 10 ∧
    a^2 + b^2 = c^2 ∧
    c > a ∧ c > b ∧
    h = 4.8 ∧
    (1/2) * c * h = (1/2) * a * b :=
sorry

end NUMINAMATH_CALUDE_height_on_longest_side_of_6_8_10_triangle_l2293_229359


namespace NUMINAMATH_CALUDE_tangent_parallel_condition_extreme_values_and_intersections_l2293_229316

/-- The cubic function f(x) = x^3 - ax^2 + bx + c -/
def f (a b c x : ℝ) : ℝ := x^3 - a*x^2 + b*x + c

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + b

theorem tangent_parallel_condition (a b c : ℝ) :
  (∃ x₀ : ℝ, f' a b x₀ = 0) → a^2 ≥ 3*b :=
sorry

theorem extreme_values_and_intersections (c : ℝ) :
  (f' 3 (-9) (-1) = 0 ∧ f' 3 (-9) 3 = 0) →
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f 3 (-9) c x₁ = 0 ∧ f 3 (-9) c x₂ = 0 ∧ f 3 (-9) c x₃ = 0 ∧
    (∀ x : ℝ, f 3 (-9) c x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  -5 < c ∧ c < 27 :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_condition_extreme_values_and_intersections_l2293_229316


namespace NUMINAMATH_CALUDE_shopping_mall_profit_l2293_229344

/-- Represents the cost and selling prices of items A and B, and the minimum number of type B items to purchase for a profit exceeding $380 -/
theorem shopping_mall_profit (cost_A cost_B sell_A sell_B : ℚ) (min_B : ℕ) : 
  cost_A = cost_B - 2 →
  80 / cost_A = 100 / cost_B →
  sell_A = 12 →
  sell_B = 15 →
  cost_A = 8 →
  cost_B = 10 →
  (∀ y : ℕ, y ≥ min_B → 
    (sell_A - cost_A) * (3 * y - 5 : ℚ) + (sell_B - cost_B) * y > 380) →
  min_B = 24 :=
by sorry

end NUMINAMATH_CALUDE_shopping_mall_profit_l2293_229344


namespace NUMINAMATH_CALUDE_t_shaped_area_l2293_229339

/-- The area of a T-shaped region formed by subtracting three smaller rectangles
    from a larger rectangle -/
theorem t_shaped_area (total_width total_height : ℝ)
                      (rect1_width rect1_height : ℝ)
                      (rect2_width rect2_height : ℝ)
                      (rect3_width rect3_height : ℝ)
                      (h1 : total_width = 6)
                      (h2 : total_height = 5)
                      (h3 : rect1_width = 1)
                      (h4 : rect1_height = 4)
                      (h5 : rect2_width = 1)
                      (h6 : rect2_height = 4)
                      (h7 : rect3_width = 1)
                      (h8 : rect3_height = 3) :
  total_width * total_height - 
  (rect1_width * rect1_height + rect2_width * rect2_height + rect3_width * rect3_height) = 19 := by
  sorry

end NUMINAMATH_CALUDE_t_shaped_area_l2293_229339


namespace NUMINAMATH_CALUDE_sequence_increasing_l2293_229313

theorem sequence_increasing (n : ℕ) (h : n ≥ 1) : 
  let a : ℕ → ℚ := fun k => (2 * k : ℚ) / (3 * k + 1)
  a (n + 1) > a n := by
sorry

end NUMINAMATH_CALUDE_sequence_increasing_l2293_229313


namespace NUMINAMATH_CALUDE_prize_winning_beverage_probabilities_l2293_229381

/-- The probability of success for each independent event -/
def p : ℚ := 1 / 6

/-- The probability of failure for each independent event -/
def q : ℚ := 1 - p

theorem prize_winning_beverage_probabilities :
  let prob_all_fail := q ^ 3
  let prob_at_least_two_fail := 1 - (3 * p^2 * q + p^3)
  (prob_all_fail = 125 / 216) ∧ (prob_at_least_two_fail = 25 / 27) := by
  sorry

end NUMINAMATH_CALUDE_prize_winning_beverage_probabilities_l2293_229381


namespace NUMINAMATH_CALUDE_third_player_win_probability_is_one_fifteenth_l2293_229386

/-- Represents the probability of the third player winning in a four-player coin-flipping game -/
def third_player_win_probability : ℚ := 1 / 15

/-- The game has four players taking turns -/
def number_of_players : ℕ := 4

/-- The position of the player we're calculating the probability for -/
def target_player_position : ℕ := 3

/-- Theorem stating that the probability of the third player winning is 1/15 -/
theorem third_player_win_probability_is_one_fifteenth :
  third_player_win_probability = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_third_player_win_probability_is_one_fifteenth_l2293_229386


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l2293_229318

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  A = 385 →
  B = 180 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l2293_229318


namespace NUMINAMATH_CALUDE_max_triangles_in_7x7_grid_triangle_l2293_229330

/-- Represents a right-angled triangle on a grid -/
structure GridTriangle where
  leg_length : ℕ
  is_right_angled : Bool

/-- Counts the maximum number of triangles in a grid triangle -/
def count_max_triangles (t : GridTriangle) : ℕ := sorry

/-- The main theorem stating the maximum number of triangles in a 7x7 grid triangle -/
theorem max_triangles_in_7x7_grid_triangle :
  ∀ (t : GridTriangle),
    t.leg_length = 7 →
    t.is_right_angled = true →
    count_max_triangles t = 28 := by sorry

end NUMINAMATH_CALUDE_max_triangles_in_7x7_grid_triangle_l2293_229330


namespace NUMINAMATH_CALUDE_particle_speed_l2293_229369

/-- A particle moves in a 2D plane. Its position at time t is given by (3t + 1, -2t + 5). 
    The theorem states that the speed of the particle is √13 units of distance per unit of time. -/
theorem particle_speed (t : ℝ) : 
  let position := fun (t : ℝ) => (3 * t + 1, -2 * t + 5)
  let velocity := fun (t : ℝ) => (3, -2)
  let speed := Real.sqrt (3^2 + (-2)^2)
  speed = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_particle_speed_l2293_229369


namespace NUMINAMATH_CALUDE_allan_initial_balloons_l2293_229310

def balloons_problem (initial_balloons : ℕ) : Prop :=
  let total_balloons := initial_balloons + 3
  6 = total_balloons + 1

theorem allan_initial_balloons : 
  ∃ (initial_balloons : ℕ), balloons_problem initial_balloons ∧ initial_balloons = 2 := by
  sorry

end NUMINAMATH_CALUDE_allan_initial_balloons_l2293_229310


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2293_229346

theorem geometric_series_sum : 
  let series := [2, 6, 18, 54, 162, 486, 1458, 4374]
  series.sum = 6560 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2293_229346


namespace NUMINAMATH_CALUDE_regular_polygon_with_45_degree_exterior_angles_l2293_229356

theorem regular_polygon_with_45_degree_exterior_angles (n : ℕ) 
  (h1 : n > 2) 
  (h2 : (360 : ℝ) / n = 45) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_45_degree_exterior_angles_l2293_229356


namespace NUMINAMATH_CALUDE_gcd_abc_plus_cba_l2293_229300

def is_consecutive (a b c : ℕ) : Prop := b = a + 1 ∧ c = a + 2

def abc_plus_cba (a b c : ℕ) : ℕ := 100 * a + 10 * b + c + 100 * c + 10 * b + a

theorem gcd_abc_plus_cba :
  ∀ a b c : ℕ,
  0 ≤ a ∧ a ≤ 7 →
  is_consecutive a b c →
  (∃ k : ℕ, abc_plus_cba a b c = 2 * k) ∧
  (∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℕ,
    0 ≤ a₁ ∧ a₁ ≤ 7 ∧
    0 ≤ a₂ ∧ a₂ ≤ 7 ∧
    is_consecutive a₁ b₁ c₁ ∧
    is_consecutive a₂ b₂ c₂ ∧
    Nat.gcd (abc_plus_cba a₁ b₁ c₁) (abc_plus_cba a₂ b₂ c₂) = 2) :=
by sorry

end NUMINAMATH_CALUDE_gcd_abc_plus_cba_l2293_229300


namespace NUMINAMATH_CALUDE_set_equality_implies_m_value_l2293_229309

theorem set_equality_implies_m_value (m : ℝ) :
  let A : Set ℝ := {1, 3, m^2}
  let B : Set ℝ := {1, m}
  A ∪ B = A →
  m = 0 ∨ m = 3 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_m_value_l2293_229309


namespace NUMINAMATH_CALUDE_exists_polygon_with_n_triangulations_l2293_229312

/-- A polygon is a closed planar figure with straight sides. -/
structure Polygon where
  -- We don't need to define the full structure of a polygon for this statement
  mk :: (dummy : Unit)

/-- The number of triangulations of a polygon. -/
def numTriangulations (p : Polygon) : ℕ := sorry

/-- For any positive integer n, there exists a polygon with exactly n triangulations. -/
theorem exists_polygon_with_n_triangulations :
  ∀ n : ℕ, n > 0 → ∃ p : Polygon, numTriangulations p = n := by sorry

end NUMINAMATH_CALUDE_exists_polygon_with_n_triangulations_l2293_229312


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2293_229328

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that the perimeter is 6√2 under the given conditions. -/
theorem triangle_perimeter (a b c : ℝ) (A : ℝ) : 
  A = π / 3 →
  b + c = 2 * a →
  (1 / 2) * b * c * Real.sin A = 2 * Real.sqrt 3 →
  a + b + c = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2293_229328


namespace NUMINAMATH_CALUDE_largest_x_value_l2293_229375

theorem largest_x_value : 
  let f : ℝ → ℝ := λ x => 7 * (9 * x^2 + 8 * x + 12) - x * (9 * x - 45)
  ∃ (x : ℝ), f x = 0 ∧ ∀ (y : ℝ), f y = 0 → y ≤ x ∧ x = -7/6 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l2293_229375


namespace NUMINAMATH_CALUDE_rower_round_trip_time_l2293_229348

/-- Proves that the total time to row to Big Rock and back is 1 hour -/
theorem rower_round_trip_time
  (rower_speed : ℝ)
  (river_speed : ℝ)
  (distance : ℝ)
  (h1 : rower_speed = 7)
  (h2 : river_speed = 2)
  (h3 : distance = 3.2142857142857144)
  : (distance / (rower_speed - river_speed)) + (distance / (rower_speed + river_speed)) = 1 := by
  sorry


end NUMINAMATH_CALUDE_rower_round_trip_time_l2293_229348


namespace NUMINAMATH_CALUDE_sum_of_abs_values_l2293_229383

theorem sum_of_abs_values (a b : ℝ) (ha : |a| = 4) (hb : |b| = 5) :
  (a + b = 9) ∨ (a + b = -9) ∨ (a + b = 1) ∨ (a + b = -1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abs_values_l2293_229383


namespace NUMINAMATH_CALUDE_yans_distance_ratio_l2293_229366

theorem yans_distance_ratio :
  ∀ (w x y : ℝ),
    w > 0 →  -- walking speed is positive
    x > 0 →  -- distance from Yan to home is positive
    y > 0 →  -- distance from Yan to stadium is positive
    y / w = x / w + (x + y) / (10 * w) →  -- time equality condition
    x / y = 9 / 11 := by
  sorry

end NUMINAMATH_CALUDE_yans_distance_ratio_l2293_229366


namespace NUMINAMATH_CALUDE_special_sum_eq_1010_l2293_229347

/-- Double factorial of a natural number -/
def doubleFac : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * doubleFac n

/-- The sum from i=1 to 1010 of ((2i)!! / (2i+1)!!) * ((2i+1)! / (2i)!) -/
def specialSum : ℚ :=
  (Finset.range 1010).sum (fun i =>
    let i' := i + 1
    (doubleFac (2 * i') : ℚ) / (doubleFac (2 * i' + 1)) *
    (Nat.factorial (2 * i' + 1) : ℚ) / (Nat.factorial (2 * i')))

/-- The sum is equal to 1010 -/
theorem special_sum_eq_1010 : specialSum = 1010 := by sorry

end NUMINAMATH_CALUDE_special_sum_eq_1010_l2293_229347


namespace NUMINAMATH_CALUDE_function_zero_point_implies_a_range_l2293_229317

theorem function_zero_point_implies_a_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioo 0 1 ∧ 4 * |a| * x₀ - 2 * a + 1 = 0) →
  a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_zero_point_implies_a_range_l2293_229317


namespace NUMINAMATH_CALUDE_vertically_opposite_angles_equal_l2293_229334

/-- Two angles are vertically opposite if they are formed by two intersecting lines
    and are not adjacent to each other. -/
def vertically_opposite (α β : Real) : Prop := sorry

theorem vertically_opposite_angles_equal {α β : Real} (h : vertically_opposite α β) : α = β := by
  sorry

end NUMINAMATH_CALUDE_vertically_opposite_angles_equal_l2293_229334


namespace NUMINAMATH_CALUDE_negation_of_statement_l2293_229378

theorem negation_of_statement :
  (¬ ∀ x : ℝ, x > 0 → x - Real.log x > 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - Real.log x₀ ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_statement_l2293_229378


namespace NUMINAMATH_CALUDE_average_of_first_three_l2293_229335

theorem average_of_first_three (A B C D : ℝ) : 
  (B + C + D) / 3 = 5 → 
  A + D = 11 → 
  D = 4 → 
  (A + B + C) / 3 = 6 := by
sorry

end NUMINAMATH_CALUDE_average_of_first_three_l2293_229335


namespace NUMINAMATH_CALUDE_geometric_sequence_a11_l2293_229321

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

-- Define the theorem
theorem geometric_sequence_a11
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a2a5 : a 2 * a 5 = 20)
  (h_a1a6 : a 1 + a 6 = 9) :
  a 11 = 25 / 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a11_l2293_229321


namespace NUMINAMATH_CALUDE_election_result_l2293_229361

/-- Represents the total number of votes in the election --/
def total_votes : ℕ := sorry

/-- Represents the initial percentage of votes for Candidate A --/
def initial_votes_A : ℚ := 65 / 100

/-- Represents the initial percentage of votes for Candidate B --/
def initial_votes_B : ℚ := 50 / 100

/-- Represents the initial percentage of votes for Candidate C --/
def initial_votes_C : ℚ := 45 / 100

/-- Represents the number of votes that change from A to B --/
def votes_A_to_B : ℕ := 1000

/-- Represents the number of votes that change from C to B --/
def votes_C_to_B : ℕ := 500

/-- Represents the final percentage of votes for Candidate B --/
def final_votes_B : ℚ := 70 / 100

theorem election_result : 
  initial_votes_B * total_votes + votes_A_to_B + votes_C_to_B = final_votes_B * total_votes ∧
  total_votes = 7500 := by sorry

end NUMINAMATH_CALUDE_election_result_l2293_229361


namespace NUMINAMATH_CALUDE_largest_two_digit_divisible_by_6_ending_in_4_l2293_229303

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∀ n : ℕ, is_two_digit n → n % 6 = 0 → ends_in_4 n → n ≤ 84 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_divisible_by_6_ending_in_4_l2293_229303


namespace NUMINAMATH_CALUDE_log_function_passes_through_point_l2293_229390

theorem log_function_passes_through_point 
  (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ Real.log x / Real.log a - 2
  f 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_function_passes_through_point_l2293_229390


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2293_229370

-- First expression
theorem simplify_expression_1 (a b : ℝ) : (1 : ℝ) * (4 * a - 2 * b) - (5 * a - 3 * b) = -a + b := by
  sorry

-- Second expression
theorem simplify_expression_2 (x : ℝ) : 2 * (2 * x^2 + 3 * x - 1) - (4 * x^2 + 2 * x - 2) = 4 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2293_229370


namespace NUMINAMATH_CALUDE_gretchen_earnings_l2293_229394

/-- Gretchen's earnings from drawing caricatures over a weekend -/
def weekend_earnings (price_per_drawing : ℕ) (saturday_sales : ℕ) (sunday_sales : ℕ) : ℕ :=
  price_per_drawing * (saturday_sales + sunday_sales)

/-- Theorem stating Gretchen's earnings for the given weekend -/
theorem gretchen_earnings :
  weekend_earnings 20 24 16 = 800 := by
  sorry

end NUMINAMATH_CALUDE_gretchen_earnings_l2293_229394


namespace NUMINAMATH_CALUDE_parallel_lines_b_value_l2293_229337

/-- Given two lines in the xy-plane, this theorem proves that if they are parallel,
    then the value of b must be 6. -/
theorem parallel_lines_b_value (b : ℝ) :
  (∀ x y, 3 * y - 3 * b = 9 * x) →
  (∀ x y, y - 2 = (b - 3) * x) →
  (∃ k : ℝ, ∀ x y, 3 * y - 3 * b = 9 * x ↔ y - 2 = k * (x - 0)) →
  b = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_b_value_l2293_229337


namespace NUMINAMATH_CALUDE_remainder_101_pow_37_mod_100_l2293_229307

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_pow_37_mod_100_l2293_229307


namespace NUMINAMATH_CALUDE_solutions_when_a_is_one_two_distinct_solutions_inequality_holds_for_all_x_l2293_229358

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 1
def g (a : ℝ) (x : ℝ) : ℝ := a * |x - 1|

-- Theorem 1
theorem solutions_when_a_is_one :
  {x : ℝ | |f x| = g 1 x} = {-2, 0, 1} := by sorry

-- Theorem 2
theorem two_distinct_solutions (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ |f x| = g a x ∧ |f y| = g a y) ↔ (a = 0 ∨ a = 2) := by sorry

-- Theorem 3
theorem inequality_holds_for_all_x (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_solutions_when_a_is_one_two_distinct_solutions_inequality_holds_for_all_x_l2293_229358


namespace NUMINAMATH_CALUDE_empty_solution_set_iff_k_ge_one_l2293_229305

-- Define the function representing the left side of the inequality
def f (k x : ℝ) : ℝ := k * x^2 - 2 * |x - 1| + 3 * k

-- Define the property of having an empty solution set
def has_empty_solution_set (k : ℝ) : Prop :=
  ∀ x : ℝ, f k x ≥ 0

-- State the theorem
theorem empty_solution_set_iff_k_ge_one :
  ∀ k : ℝ, has_empty_solution_set k ↔ k ≥ 1 := by sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_k_ge_one_l2293_229305


namespace NUMINAMATH_CALUDE_car_average_speed_l2293_229385

/-- Proves that the average speed of a car is 36 km/hr given specific uphill and downhill conditions -/
theorem car_average_speed :
  let uphill_speed : ℝ := 30
  let downhill_speed : ℝ := 60
  let uphill_distance : ℝ := 100
  let downhill_distance : ℝ := 50
  let total_distance : ℝ := uphill_distance + downhill_distance
  let uphill_time : ℝ := uphill_distance / uphill_speed
  let downhill_time : ℝ := downhill_distance / downhill_speed
  let total_time : ℝ := uphill_time + downhill_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 36 := by
sorry

end NUMINAMATH_CALUDE_car_average_speed_l2293_229385


namespace NUMINAMATH_CALUDE_unique_solution_l2293_229322

/-- Represents the ages of two sons given specific conditions --/
structure SonsAges where
  elder : ℕ
  younger : ℕ
  doubled_elder_exceeds_sum : 2 * elder = elder + younger + 18
  younger_less_than_difference : younger = elder - younger - 6

/-- The unique solution to the SonsAges problem --/
def solution : SonsAges := { 
  elder := 30,
  younger := 12,
  doubled_elder_exceeds_sum := by sorry,
  younger_less_than_difference := by sorry
}

/-- Proves that the solution is unique --/
theorem unique_solution (s : SonsAges) : s = solution := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2293_229322


namespace NUMINAMATH_CALUDE_det_circulant_matrix_l2293_229319

def circulant_matrix (n : ℕ) (h : n > 1) (h_odd : Odd n) : Matrix (Fin n) (Fin n) ℤ :=
  λ i j => if i = j then 2
           else if (i - j) % n = 2 ∨ (i - j) % n = n - 2 then 1
           else 0

theorem det_circulant_matrix (n : ℕ) (h : n > 1) (h_odd : Odd n) :
  let A := circulant_matrix n h h_odd
  Matrix.det A = 4 := by
  sorry

end NUMINAMATH_CALUDE_det_circulant_matrix_l2293_229319
