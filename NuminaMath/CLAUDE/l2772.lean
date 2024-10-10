import Mathlib

namespace right_triangle_perimeter_l2772_277295

theorem right_triangle_perimeter (a b c : ℝ) : 
  a = 10 ∧ b = 24 ∧ c = 26 →
  a + b > c ∧ a + c > b ∧ b + c > a →
  a^2 + b^2 = c^2 →
  a + b + c = 60 :=
by sorry

end right_triangle_perimeter_l2772_277295


namespace semicircle_to_cone_volume_l2772_277214

/-- The volume of a cone formed by rolling up a semicircle -/
theorem semicircle_to_cone_volume (R : ℝ) (R_pos : R > 0) :
  let r := R / 2
  let h := R * (Real.sqrt 3) / 2
  (1 / 3) * Real.pi * r^2 * h = (Real.pi * R^3 * Real.sqrt 3) / 24 :=
by sorry

end semicircle_to_cone_volume_l2772_277214


namespace square_area_error_l2772_277237

theorem square_area_error (a : ℝ) (h : a > 0) : 
  let measured_side := a * 1.05
  let actual_area := a ^ 2
  let calculated_area := measured_side ^ 2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.1025 := by sorry

end square_area_error_l2772_277237


namespace sprint_distance_l2772_277298

/-- Given a constant speed and a duration, calculates the distance traveled. -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that sprinting at 6 miles per hour for 4 hours results in a distance of 24 miles. -/
theorem sprint_distance : distance_traveled 6 4 = 24 := by
  sorry

end sprint_distance_l2772_277298


namespace prism_surface_area_l2772_277255

/-- A rectangular prism formed by unit cubes -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The volume of a rectangular prism -/
def volume (p : RectangularPrism) : ℕ :=
  p.length * p.width * p.height

/-- The surface area of a rectangular prism -/
def surfaceArea (p : RectangularPrism) : ℕ :=
  2 * (p.length * p.width + p.width * p.height + p.height * p.length)

/-- The number of unpainted cubes in a prism -/
def unpaintedCubes (p : RectangularPrism) : ℕ :=
  (p.length - 2) * (p.width - 2) * (p.height - 2)

theorem prism_surface_area :
  ∃ (p : RectangularPrism),
    volume p = 120 ∧
    unpaintedCubes p = 24 ∧
    surfaceArea p = 148 := by
  sorry

end prism_surface_area_l2772_277255


namespace system_solution_and_simplification_l2772_277267

-- Define the system of equations
def system (x y m : ℝ) : Prop :=
  x + y = m + 2 ∧ 4 * x + 5 * y = 6 * m + 3

-- Define the positivity condition for x and y
def positive_solution (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

-- Theorem statement
theorem system_solution_and_simplification (m : ℝ) :
  (∃ x y, system x y m ∧ positive_solution x y) →
  (5/2 < m ∧ m < 7) ∧
  (|2*m - 5| - |m - 7| = 3*m - 12) :=
by sorry

end system_solution_and_simplification_l2772_277267


namespace john_gave_one_third_l2772_277254

/-- The fraction of burritos John gave to his friend -/
def fraction_given_away (boxes : ℕ) (burritos_per_box : ℕ) (days : ℕ) (burritos_per_day : ℕ) (burritos_left : ℕ) : ℚ :=
  let total_bought := boxes * burritos_per_box
  let total_eaten := days * burritos_per_day
  let total_before_eating := total_eaten + burritos_left
  let given_away := total_bought - total_before_eating
  given_away / total_bought

/-- Theorem stating that John gave away 1/3 of the burritos -/
theorem john_gave_one_third :
  fraction_given_away 3 20 10 3 10 = 1 / 3 := by
  sorry


end john_gave_one_third_l2772_277254


namespace units_digit_problem_l2772_277250

theorem units_digit_problem : ∃ n : ℕ, (6 * 16 * 1986 - 6^4) % 10 = 0 := by
  sorry

end units_digit_problem_l2772_277250


namespace octagon_perimeter_l2772_277229

/-- The perimeter of a regular octagon with side length 2 units is 16 units. -/
theorem octagon_perimeter : ℕ → ℕ → ℕ
  | 8, 2 => 16
  | _, _ => 0

#check octagon_perimeter

end octagon_perimeter_l2772_277229


namespace smallest_four_digit_multiple_of_112_l2772_277220

theorem smallest_four_digit_multiple_of_112 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 112 ∣ n → 1008 ≤ n :=
by
  sorry

end smallest_four_digit_multiple_of_112_l2772_277220


namespace johns_money_proof_l2772_277253

/-- Calculates John's initial amount of money given his purchases and remaining money -/
def johns_initial_money (roast_cost vegetables_cost remaining_money : ℕ) : ℕ :=
  roast_cost + vegetables_cost + remaining_money

theorem johns_money_proof (roast_cost vegetables_cost remaining_money : ℕ) 
  (h1 : roast_cost = 17)
  (h2 : vegetables_cost = 11)
  (h3 : remaining_money = 72) :
  johns_initial_money roast_cost vegetables_cost remaining_money = 100 := by
  sorry

end johns_money_proof_l2772_277253


namespace lottery_probabilities_l2772_277247

def total_numbers : ℕ := 10
def numbers_per_ticket : ℕ := 5
def numbers_drawn : ℕ := 4

def probability_four_match : ℚ := 1 / 21
def probability_two_match : ℚ := 10 / 21

theorem lottery_probabilities :
  (total_numbers = 10) →
  (numbers_per_ticket = 5) →
  (numbers_drawn = 4) →
  (probability_four_match = 1 / 21) ∧
  (probability_two_match = 10 / 21) := by
  sorry

end lottery_probabilities_l2772_277247


namespace smallest_term_is_fifth_l2772_277264

def a (n : ℕ) : ℤ := 3 * n^2 - 28 * n

theorem smallest_term_is_fifth : 
  ∀ k : ℕ, k ≠ 0 → a 5 ≤ a k :=
sorry

end smallest_term_is_fifth_l2772_277264


namespace max_value_of_f_on_interval_l2772_277271

def f (x : ℝ) := 2 * x^2 + 4 * x - 1

theorem max_value_of_f_on_interval : 
  ∃ (c : ℝ), c ∈ Set.Icc (-2) 2 ∧ 
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≤ f c) ∧
  f c = 15 :=
sorry

end max_value_of_f_on_interval_l2772_277271


namespace june_upload_ratio_l2772_277293

/-- Represents the video upload scenario for a YouTuber in June --/
structure VideoUpload where
  totalDays : Nat
  halfMonth : Nat
  firstHalfDailyHours : Nat
  totalHours : Nat

/-- Calculates the ratio of daily video hours in the second half to the first half of the month --/
def uploadRatio (v : VideoUpload) : Rat :=
  let firstHalfTotal := v.firstHalfDailyHours * v.halfMonth
  let secondHalfTotal := v.totalHours - firstHalfTotal
  let secondHalfDaily := secondHalfTotal / v.halfMonth
  secondHalfDaily / v.firstHalfDailyHours

/-- The main theorem stating the upload ratio for the given scenario --/
theorem june_upload_ratio (v : VideoUpload) 
    (h1 : v.totalDays = 30)
    (h2 : v.halfMonth = 15)
    (h3 : v.firstHalfDailyHours = 10)
    (h4 : v.totalHours = 450) :
  uploadRatio v = 2 := by
  sorry

#eval uploadRatio { totalDays := 30, halfMonth := 15, firstHalfDailyHours := 10, totalHours := 450 }

end june_upload_ratio_l2772_277293


namespace bee_speed_is_11_5_l2772_277284

/-- Represents the bee's journey with given conditions -/
structure BeeJourney where
  v : ℝ  -- Bee's constant actual speed
  t_dr : ℝ := 10  -- Time from daisy to rose
  t_rp : ℝ := 6   -- Time from rose to poppy
  t_pt : ℝ := 8   -- Time from poppy to tulip
  slow : ℝ := 2   -- Speed reduction due to crosswind
  boost : ℝ := 3  -- Speed increase due to crosswind

  d_dr : ℝ := t_dr * (v - slow)  -- Distance from daisy to rose
  d_rp : ℝ := t_rp * (v + boost) -- Distance from rose to poppy
  d_pt : ℝ := t_pt * (v - slow)  -- Distance from poppy to tulip

  h_distance_diff : d_dr = d_rp + 8  -- Distance condition
  h_distance_equal : d_pt = d_dr     -- Distance equality condition

/-- Theorem stating that the bee's speed is 11.5 m/s given the conditions -/
theorem bee_speed_is_11_5 (j : BeeJourney) : j.v = 11.5 := by
  sorry

end bee_speed_is_11_5_l2772_277284


namespace amoeba_count_10_days_l2772_277275

/-- The number of amoebas in the petri dish after n days -/
def amoeba_count (n : ℕ) : ℕ := 3^n

/-- Theorem stating that the number of amoebas after 10 days is 59049 -/
theorem amoeba_count_10_days : amoeba_count 10 = 59049 := by
  sorry

end amoeba_count_10_days_l2772_277275


namespace unique_triples_l2772_277263

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem unique_triples : 
  ∀ a b c : ℕ, 
    (is_prime (a^2 - 23)) → 
    (is_prime (b^2 - 23)) → 
    ((a^2 - 23) * (b^2 - 23) = c^2 - 23) → 
    ((a = 5 ∧ b = 6 ∧ c = 7) ∨ (a = 6 ∧ b = 5 ∧ c = 7)) :=
by sorry

end unique_triples_l2772_277263


namespace subset_sum_theorem_l2772_277212

theorem subset_sum_theorem (a₁ a₂ a₃ a₄ : ℝ) 
  (h : (a₁ + a₂) + (a₁ + a₃) + (a₁ + a₄) + (a₂ + a₃) + (a₂ + a₄) + (a₃ + a₄) + 
       (a₁ + a₂ + a₃) + (a₁ + a₂ + a₄) + (a₁ + a₃ + a₄) + (a₂ + a₃ + a₄) = 28) :
  a₁ + a₂ + a₃ + a₄ = 4 := by
sorry

end subset_sum_theorem_l2772_277212


namespace triangle_cosine_problem_l2772_277244

theorem triangle_cosine_problem (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  b = 3 →
  C = 2 * A →
  0 < A →
  A < π →
  0 < B →
  B < π →
  0 < C →
  C < π →
  a = 2 * Real.sin B * Real.sin (C / 2) →
  b = 2 * Real.sin A * Real.sin (C / 2) →
  c = 2 * Real.sin A * Real.sin B →
  Real.cos C = 1 / 4 := by
sorry

end triangle_cosine_problem_l2772_277244


namespace smallest_integer_absolute_value_l2772_277235

theorem smallest_integer_absolute_value (x : ℤ) :
  (∀ y : ℤ, |3 * y - 4| ≤ 22 → x ≤ y) ↔ x = -6 := by
  sorry

end smallest_integer_absolute_value_l2772_277235


namespace python_to_boa_ratio_l2772_277261

/-- The ratio of pythons to boa constrictors in a park -/
theorem python_to_boa_ratio :
  let total_snakes : ℕ := 200
  let boa_constrictors : ℕ := 40
  let rattlesnakes : ℕ := 40
  let pythons : ℕ := total_snakes - (boa_constrictors + rattlesnakes)
  (pythons : ℚ) / boa_constrictors = 3 := by
  sorry

end python_to_boa_ratio_l2772_277261


namespace factorization_of_36x_squared_minus_4_l2772_277278

theorem factorization_of_36x_squared_minus_4 (x : ℝ) :
  36 * x^2 - 4 = 4 * (3*x + 1) * (3*x - 1) := by
  sorry

end factorization_of_36x_squared_minus_4_l2772_277278


namespace quadratic_inequality_solution_l2772_277205

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 40*x + 350 ≤ 0} = Set.Icc 10 30 :=
by sorry

end quadratic_inequality_solution_l2772_277205


namespace right_triangle_hypotenuse_l2772_277232

/-- Given a right triangle with one leg of length 15 and the angle opposite that leg measuring 30°, 
    the hypotenuse has length 30. -/
theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) (h1 : leg = 15) (h2 : angle = 30) :
  let hypotenuse := 2 * leg
  hypotenuse = 30 := by
  sorry

end right_triangle_hypotenuse_l2772_277232


namespace g_behavior_l2772_277243

def g (x : ℝ) := -3 * x^4 + 5 * x^3 - 2

theorem g_behavior :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x < M) :=
sorry

end g_behavior_l2772_277243


namespace complex_expression_equals_negative_two_l2772_277231

theorem complex_expression_equals_negative_two :
  let A := (Real.sqrt 6 + Real.sqrt 2) * (Real.sqrt 3 - 2) * Real.sqrt (Real.sqrt 3 + 2)
  A = -2 := by
  sorry

end complex_expression_equals_negative_two_l2772_277231


namespace unique_divisible_by_792_l2772_277281

/-- Represents a 7-digit number in the form 13xy45z -/
def number (x y z : Nat) : Nat :=
  1300000 + x * 10000 + y * 1000 + 450 + z

/-- Checks if a number is of the form 13xy45z where x, y, z are single digits -/
def isValidForm (n : Nat) : Prop :=
  ∃ x y z, x < 10 ∧ y < 10 ∧ z < 10 ∧ n = number x y z

theorem unique_divisible_by_792 :
  ∃! n, isValidForm n ∧ n % 792 = 0 ∧ n = 1380456 :=
sorry

end unique_divisible_by_792_l2772_277281


namespace total_swim_time_l2772_277218

def freestyle : ℕ := 48

def backstroke (f : ℕ) : ℕ := f + 4

def butterfly (b : ℕ) : ℕ := b + 3

def breaststroke (t : ℕ) : ℕ := t + 2

theorem total_swim_time :
  freestyle + backstroke freestyle + butterfly (backstroke freestyle) + breaststroke (butterfly (backstroke freestyle)) = 212 := by
  sorry

end total_swim_time_l2772_277218


namespace mans_age_twice_sons_l2772_277265

/-- 
Proves that the number of years until a man's age is twice his son's age is 2,
given that the man is 24 years older than his son and the son's present age is 22 years.
-/
theorem mans_age_twice_sons (
  son_age : ℕ) (man_age : ℕ) (years : ℕ) : 
  son_age = 22 →
  man_age = son_age + 24 →
  man_age + years = 2 * (son_age + years) →
  years = 2 :=
by sorry

end mans_age_twice_sons_l2772_277265


namespace sum_of_two_squares_l2772_277266

theorem sum_of_two_squares (P : ℤ) (a b : ℤ) (h : P = a^2 + b^2) :
  ∃ x y : ℤ, 2*P = x^2 + y^2 := by
sorry

end sum_of_two_squares_l2772_277266


namespace marc_tv_watching_l2772_277208

/-- Given the number of episodes Marc watches per day and the total number of episodes,
    prove the relationship between x, y, and z. -/
theorem marc_tv_watching
  (friends_total : ℕ)
  (seinfeld_total : ℕ)
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (h1 : friends_total = 50)
  (h2 : seinfeld_total = 75)
  (h3 : x * z = friends_total)
  (h4 : y * z = seinfeld_total) :
  y = (3 / 2) * x ∧ z = 50 / x :=
by sorry

end marc_tv_watching_l2772_277208


namespace final_amount_correct_l2772_277256

def total_income : ℝ := 1000000

def children_share : ℝ := 0.2
def num_children : ℕ := 3
def wife_share : ℝ := 0.3
def orphan_donation_rate : ℝ := 0.05

def amount_left : ℝ := 
  total_income * (1 - children_share * num_children - wife_share) * (1 - orphan_donation_rate)

theorem final_amount_correct : amount_left = 95000 := by
  sorry

end final_amount_correct_l2772_277256


namespace largest_prime_factor_l2772_277292

def expression : ℤ := 17^4 + 3 * 17^2 + 2 - 16^4

theorem largest_prime_factor :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expression.natAbs ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ expression.natAbs → q ≤ p ∧
  p = 34087 :=
sorry

end largest_prime_factor_l2772_277292


namespace prob_multiple_13_eq_l2772_277289

/-- Represents a standard deck of 54 cards with 4 suits (1-13) and 2 jokers -/
def Deck : Type := Fin 54

/-- Represents the rank of a card (1-13 for regular cards, 0 for jokers) -/
def rank (card : Deck) : ℕ :=
  if card.val < 52 then
    (card.val % 13) + 1
  else
    0

/-- Shuffles the deck uniformly randomly -/
def shuffle (deck : Deck → α) : Deck → α :=
  sorry

/-- Calculates the score based on the shuffled deck -/
def score (shuffled_deck : Deck → Deck) : ℕ :=
  sorry

/-- Probability that the score is a multiple of 13 -/
def prob_multiple_13 : ℚ :=
  sorry

/-- Main theorem: The probability of the score being a multiple of 13 is 77/689 -/
theorem prob_multiple_13_eq : prob_multiple_13 = 77 / 689 :=
  sorry

end prob_multiple_13_eq_l2772_277289


namespace third_butcher_delivery_l2772_277223

theorem third_butcher_delivery (package_weight : ℕ) (first_butcher : ℕ) (second_butcher : ℕ) (total_weight : ℕ) : 
  package_weight = 4 →
  first_butcher = 10 →
  second_butcher = 7 →
  total_weight = 100 →
  ∃ third_butcher : ℕ, 
    third_butcher * package_weight + first_butcher * package_weight + second_butcher * package_weight = total_weight ∧
    third_butcher = 8 :=
by sorry

end third_butcher_delivery_l2772_277223


namespace football_cost_l2772_277221

theorem football_cost (total_cost marbles_cost baseball_cost : ℚ)
  (h1 : total_cost = 20.52)
  (h2 : marbles_cost = 9.05)
  (h3 : baseball_cost = 6.52) :
  total_cost - marbles_cost - baseball_cost = 4.95 := by
  sorry

end football_cost_l2772_277221


namespace compute_fraction_square_l2772_277283

theorem compute_fraction_square : 6 * (3 / 7)^2 = 54 / 49 := by
  sorry

end compute_fraction_square_l2772_277283


namespace ball_max_height_l2772_277270

-- Define the function representing the ball's height
def f (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

-- Theorem stating that the maximum height is 40 feet
theorem ball_max_height :
  ∃ (max : ℝ), max = 40 ∧ ∀ (t : ℝ), f t ≤ max :=
sorry

end ball_max_height_l2772_277270


namespace set_equality_l2772_277215

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end set_equality_l2772_277215


namespace unique_solution_in_interval_l2772_277282

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 1

-- State the theorem
theorem unique_solution_in_interval :
  ∃! a : ℝ, 0 < a ∧ a < 3 ∧ f a = 7 ∧ a = 2 := by sorry

end unique_solution_in_interval_l2772_277282


namespace magazine_circulation_ratio_l2772_277272

/-- The circulation ratio problem for magazine P -/
theorem magazine_circulation_ratio 
  (avg_circulation : ℝ) -- Average yearly circulation for 1962-1970
  (h : avg_circulation > 0) -- Assumption that circulation is positive
  : (4 * avg_circulation) / (4 * avg_circulation + 9 * avg_circulation) = 4 / 13 := by
  sorry

end magazine_circulation_ratio_l2772_277272


namespace addition_problem_base_5_l2772_277279

def base_5_to_10 (n : ℕ) : ℕ := sorry

def base_10_to_5 (n : ℕ) : ℕ := sorry

theorem addition_problem_base_5 (X Y : ℕ) : 
  base_10_to_5 (3 * 25 + X * 5 + Y) + base_10_to_5 (3 * 5 + 2) = 
  base_10_to_5 (4 * 25 + 2 * 5 + X) →
  X + Y = 6 := by sorry

end addition_problem_base_5_l2772_277279


namespace algebraic_simplification_l2772_277209

theorem algebraic_simplification (a b : ℝ) : -a^2 * (-2*a*b) + 3*a * (a^2*b - 1) = 5*a^3*b - 3*a := by
  sorry

end algebraic_simplification_l2772_277209


namespace corrected_mean_l2772_277285

theorem corrected_mean (n : ℕ) (original_mean : ℝ) (incorrect_value correct_value : ℝ) :
  n = 40 ∧ original_mean = 36 ∧ incorrect_value = 20 ∧ correct_value = 34 →
  (n : ℝ) * original_mean - incorrect_value + correct_value = n * 36.35 :=
by sorry

end corrected_mean_l2772_277285


namespace train_speed_l2772_277200

/-- Calculates the speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length time_to_pass : ℝ) :
  train_length = 300 →
  bridge_length = 115 →
  time_to_pass = 42.68571428571429 →
  ∃ (speed : ℝ), abs (speed - 35.01) < 0.01 ∧ 
    speed = (train_length + bridge_length) / time_to_pass * 3.6 := by
  sorry

end train_speed_l2772_277200


namespace square_root_of_nine_l2772_277225

theorem square_root_of_nine :
  ∃ x : ℝ, x^2 = 9 ∧ (x = 3 ∨ x = -3) :=
by sorry

end square_root_of_nine_l2772_277225


namespace factorization_equality_l2772_277268

theorem factorization_equality (a b : ℝ) : a^2 * b - 6 * a * b + 9 * b = b * (a - 3)^2 := by
  sorry

end factorization_equality_l2772_277268


namespace age_difference_proof_l2772_277260

/-- Proves the number of years ago when the elder person was twice as old as the younger person -/
theorem age_difference_proof (younger_age elder_age years_ago : ℕ) : 
  younger_age = 35 →
  elder_age - younger_age = 20 →
  elder_age - years_ago = 2 * (younger_age - years_ago) →
  years_ago = 15 := by
sorry

end age_difference_proof_l2772_277260


namespace sin_period_2x_minus_pi_div_6_l2772_277206

/-- The minimum positive period of y = sin(2x - π/6) is π -/
theorem sin_period_2x_minus_pi_div_6 (x : ℝ) :
  let f := fun x => Real.sin (2 * x - π / 6)
  ∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧ 
  (∀ q, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q) ∧
  p = π :=
sorry

end sin_period_2x_minus_pi_div_6_l2772_277206


namespace sum_of_coefficients_l2772_277226

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 + (1 + x)^6 + (1 + x)^7 + (1 + x)^8
           = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 502 := by
sorry

end sum_of_coefficients_l2772_277226


namespace unique_sum_value_l2772_277251

theorem unique_sum_value (n m : ℤ) 
  (h1 : 3 * n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3 * m - 2 * n < 46) :
  2 * n + m = 36 := by
sorry

end unique_sum_value_l2772_277251


namespace subset_relation_l2772_277252

universe u

theorem subset_relation (A B : Set α) :
  (∃ x, x ∈ B) →
  (∀ y, y ∈ A → y ∈ B) →
  B ⊆ A :=
by sorry

end subset_relation_l2772_277252


namespace shaded_area_is_36_l2772_277274

/-- Given a rectangle and a right triangle with the following properties:
    - Rectangle: width 12, height 12, lower right vertex at (12, 0)
    - Triangle: base 12, height 12, lower left vertex at (12, 0)
    - Line passing through (0, 12) and (24, 0)
    Prove that the area of the triangle formed by this line, the vertical line x = 12,
    and the x-axis is 36 square units. -/
theorem shaded_area_is_36 (rectangle_width rectangle_height triangle_base triangle_height : ℝ)
  (h_rect_width : rectangle_width = 12)
  (h_rect_height : rectangle_height = 12)
  (h_tri_base : triangle_base = 12)
  (h_tri_height : triangle_height = 12) :
  let line := fun x => -1/2 * x + 12
  let intersection_x := 12
  let intersection_y := line intersection_x
  let shaded_area := 1/2 * intersection_y * triangle_base
  shaded_area = 36 := by
  sorry

end shaded_area_is_36_l2772_277274


namespace fifty_eighth_digit_of_one_seventeenth_l2772_277249

def decimal_representation (n : ℕ) : List ℕ := sorry

def is_periodic (l : List ℕ) : Prop := sorry

def nth_digit (l : List ℕ) (n : ℕ) : ℕ := sorry

theorem fifty_eighth_digit_of_one_seventeenth (h : is_periodic (decimal_representation 17)) :
  nth_digit (decimal_representation 17) 58 = 4 := by sorry

end fifty_eighth_digit_of_one_seventeenth_l2772_277249


namespace mans_speed_against_current_l2772_277219

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
theorem mans_speed_against_current 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_with_current = 12)
  (h2 : current_speed = 2) :
  speed_with_current - 2 * current_speed = 8 :=
by sorry

end mans_speed_against_current_l2772_277219


namespace min_value_ab_l2772_277234

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b + 3 = a * b) :
  a * b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ + 3 = a₀ * b₀ ∧ a₀ * b₀ = 9 :=
sorry

end min_value_ab_l2772_277234


namespace inequality_proof_l2772_277227

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a * (3 * a - 1)) / (1 + a^2) + 
  (b * (3 * b - 1)) / (1 + b^2) + 
  (c * (3 * c - 1)) / (1 + c^2) ≥ 0 :=
by sorry

end inequality_proof_l2772_277227


namespace tea_shop_problem_l2772_277203

/-- Tea shop problem -/
theorem tea_shop_problem 
  (cost_A : ℝ) 
  (cost_B : ℝ) 
  (num_B_more : ℕ) 
  (cost_ratio : ℝ) 
  (total_boxes : ℕ) 
  (sell_price_A : ℝ) 
  (sell_price_B : ℝ) 
  (discount : ℝ) 
  (profit : ℝ)
  (h1 : cost_A = 4000)
  (h2 : cost_B = 8400)
  (h3 : num_B_more = 10)
  (h4 : cost_ratio = 1.4)
  (h5 : total_boxes = 100)
  (h6 : sell_price_A = 300)
  (h7 : sell_price_B = 400)
  (h8 : discount = 0.3)
  (h9 : profit = 5800) :
  ∃ (cost_per_A cost_per_B : ℝ) (num_A num_B : ℕ),
    cost_per_A = 200 ∧ 
    cost_per_B = 280 ∧ 
    num_A = 40 ∧ 
    num_B = 60 ∧
    cost_B / cost_per_B - cost_A / cost_per_A = num_B_more ∧
    cost_per_B = cost_ratio * cost_per_A ∧
    num_A + num_B = total_boxes ∧
    (sell_price_A - cost_per_A) * (num_A / 2) + 
    (sell_price_A * (1 - discount) - cost_per_A) * (num_A / 2) +
    (sell_price_B - cost_per_B) * (num_B / 2) + 
    (sell_price_B * (1 - discount) - cost_per_B) * (num_B / 2) = profit :=
by
  sorry

end tea_shop_problem_l2772_277203


namespace coin_toss_experiment_l2772_277238

theorem coin_toss_experiment (total_tosses : ℕ) (heads_frequency : ℚ) 
  (h1 : total_tosses = 100)
  (h2 : heads_frequency = 49/100) :
  total_tosses - (total_tosses * heads_frequency).num = 51 := by
  sorry

end coin_toss_experiment_l2772_277238


namespace max_2012_gons_less_than_1006_l2772_277245

/-- The number of sides in each polygon -/
def n : ℕ := 2012

/-- The maximum number of different n-gons that can be drawn with all vertices shared 
    and no sides shared between any two polygons -/
def max_polygons (n : ℕ) : ℕ := (n - 1) / 2

/-- Theorem: The maximum number of different 2012-gons that can be drawn with all vertices shared 
    and no sides shared between any two polygons is less than 1006 -/
theorem max_2012_gons_less_than_1006 : max_polygons n < 1006 := by
  sorry

end max_2012_gons_less_than_1006_l2772_277245


namespace video_votes_l2772_277248

theorem video_votes (score : ℤ) (like_percentage : ℚ) : 
  score = 120 ∧ 
  like_percentage = 72 / 100 →
  ∃ (total_votes : ℕ), 
    (like_percentage * total_votes : ℚ) - ((1 - like_percentage) * total_votes : ℚ) = score ∧
    total_votes = 273 := by
sorry

end video_votes_l2772_277248


namespace inequality_proof_l2772_277241

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) :
  x^2 * y^2 + |x^2 - y^2| ≤ π/2 := by
  sorry

end inequality_proof_l2772_277241


namespace min_value_of_sequence_sequence_satisfies_conditions_l2772_277210

def sequence_a (n : ℕ) : ℝ :=
  if n = 0 then 0
  else if n = 1 then 98
  else 102 + (n - 2) * (2 * n + 2)

theorem min_value_of_sequence (n : ℕ) (h : n > 0) :
  sequence_a n / n ≥ 26 ∧ ∃ m : ℕ, m > 0 ∧ sequence_a m / m = 26 :=
by
  sorry

theorem sequence_satisfies_conditions :
  sequence_a 2 = 102 ∧
  ∀ n : ℕ, n > 0 → sequence_a (n + 1) - sequence_a n = 4 * n :=
by
  sorry

end min_value_of_sequence_sequence_satisfies_conditions_l2772_277210


namespace square_perimeter_l2772_277290

theorem square_perimeter (rectangle_perimeter : ℝ) (square_side : ℝ) : 
  (rectangle_perimeter + 4 * square_side) - rectangle_perimeter = 17 →
  4 * square_side = 34 := by
  sorry

end square_perimeter_l2772_277290


namespace nonagon_diagonals_count_l2772_277211

/-- The number of sides in a nonagon -/
def nonagon_sides : ℕ := 9

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := (nonagon_sides * (nonagon_sides - 3)) / 2

theorem nonagon_diagonals_count : nonagon_diagonals = 27 := by
  sorry

end nonagon_diagonals_count_l2772_277211


namespace division_count_is_eight_l2772_277213

/-- Represents an L-shaped piece consisting of three cells -/
structure LPiece where
  -- Define properties of an L-shaped piece if needed

/-- Represents a 3 × 6 rectangle -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Represents a division of the rectangle into L-shaped pieces -/
structure Division where
  pieces : List LPiece

/-- Function to count valid divisions of the rectangle into L-shaped pieces -/
def countValidDivisions (rect : Rectangle) : Nat :=
  sorry

/-- Theorem stating that the number of ways to divide a 3 × 6 rectangle 
    into L-shaped pieces of three cells is 8 -/
theorem division_count_is_eight :
  let rect : Rectangle := { width := 6, height := 3 }
  countValidDivisions rect = 8 := by
  sorry

end division_count_is_eight_l2772_277213


namespace inequality_proof_l2772_277216

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (a^2 * b + a + b^2) * (a * b^2 + a^2 + b) > 9 * a^2 * b^2 := by
  sorry

end inequality_proof_l2772_277216


namespace man_mass_on_boat_l2772_277230

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sink_height : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sink_height * 1000

/-- Theorem stating that a man who causes a 7m x 3m boat to sink by 1cm has a mass of 210 kg. -/
theorem man_mass_on_boat : 
  mass_of_man 7 3 0.01 = 210 := by sorry

end man_mass_on_boat_l2772_277230


namespace garden_pool_perimeter_l2772_277296

/-- Represents a rectangular garden with square plots and a pool -/
structure Garden where
  plot_area : ℝ
  garden_length : ℝ
  num_plots : ℕ

/-- Calculates the perimeter of the pool in the garden -/
def pool_perimeter (g : Garden) : ℝ :=
  2 * g.garden_length

/-- Theorem stating the perimeter of the pool in the given garden configuration -/
theorem garden_pool_perimeter (g : Garden) 
  (h1 : g.plot_area = 20)
  (h2 : g.garden_length = 9)
  (h3 : g.num_plots = 4) : 
  pool_perimeter g = 18 := by
  sorry

#check garden_pool_perimeter

end garden_pool_perimeter_l2772_277296


namespace kims_candy_bars_l2772_277242

/-- The number of candy bars Kim's dad buys her each week -/
def candy_bars_per_week : ℕ := 2

/-- The number of weeks in the problem -/
def total_weeks : ℕ := 16

/-- The number of candy bars Kim eats in the given period -/
def candy_bars_eaten : ℕ := total_weeks / 4

/-- The number of candy bars Kim has saved after the given period -/
def candy_bars_saved : ℕ := 28

theorem kims_candy_bars : 
  candy_bars_per_week * total_weeks - candy_bars_eaten = candy_bars_saved :=
by sorry

end kims_candy_bars_l2772_277242


namespace tan_sum_product_equals_sqrt_three_l2772_277202

theorem tan_sum_product_equals_sqrt_three : 
  Real.tan (17 * π / 180) + Real.tan (43 * π / 180) + 
  Real.sqrt 3 * Real.tan (17 * π / 180) * Real.tan (43 * π / 180) = Real.sqrt 3 := by
  sorry

end tan_sum_product_equals_sqrt_three_l2772_277202


namespace function_shift_l2772_277207

/-- Given a function f(x) = (x(x+3))/2, prove that f(x-1) = (x^2 + x - 2)/2 -/
theorem function_shift (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x * (x + 3)) / 2
  f (x - 1) = (x^2 + x - 2) / 2 := by
  sorry

end function_shift_l2772_277207


namespace premium_probability_option2_higher_price_probability_relationship_l2772_277288

-- Define the grades of oranges
inductive Grade : Type
| Premium : Grade
| Special : Grade
| Superior : Grade
| FirstGrade : Grade

-- Define the distribution of boxes
def total_boxes : ℕ := 100
def premium_boxes : ℕ := 40
def special_boxes : ℕ := 30
def superior_boxes : ℕ := 10
def first_grade_boxes : ℕ := 20

-- Define the pricing options
def option1_price : ℚ := 27
def premium_price : ℚ := 36
def special_price : ℚ := 30
def superior_price : ℚ := 24
def first_grade_price : ℚ := 18

-- Theorem 1: Probability of selecting a premium grade box
theorem premium_probability : 
  (premium_boxes : ℚ) / total_boxes = 2 / 5 := by sorry

-- Theorem 2: Average price of Option 2 is higher than Option 1
theorem option2_higher_price :
  (premium_price * premium_boxes + special_price * special_boxes + 
   superior_price * superior_boxes + first_grade_price * first_grade_boxes) / 
  total_boxes > option1_price := by sorry

-- Define probabilities for selecting 3 boxes with different grades
def p₁ : ℚ := 1465 / 1617  -- from 100 boxes
def p₂ : ℚ := 53 / 57      -- from 20 boxes in stratified sampling

-- Theorem 3: Relationship between p₁ and p₂
theorem probability_relationship : p₁ < p₂ := by sorry

end premium_probability_option2_higher_price_probability_relationship_l2772_277288


namespace distance_to_y_axis_l2772_277239

/-- The distance from a point P(2-a, -5) to the y-axis is |2-a| -/
theorem distance_to_y_axis (a : ℝ) : 
  let P : ℝ × ℝ := (2 - a, -5)
  abs (P.1) = abs (2 - a) := by sorry

end distance_to_y_axis_l2772_277239


namespace gcd_lcm_product_18_42_l2772_277217

theorem gcd_lcm_product_18_42 : Nat.gcd 18 42 * Nat.lcm 18 42 = 756 := by
  sorry

end gcd_lcm_product_18_42_l2772_277217


namespace two_true_statements_l2772_277277

theorem two_true_statements 
  (x y a b : ℝ) 
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ a ≠ 0 ∧ b ≠ 0) 
  (h_x_lt_a : x < a) 
  (h_y_lt_b : y < b) 
  (h_positive : x > 0 ∧ y > 0 ∧ a > 0 ∧ b > 0) : 
  ∃! n : ℕ, n = 2 ∧ n = (
    (if x + y < a + b then 1 else 0) +
    (if x - y < a - b then 1 else 0) +
    (if x * y < a * b then 1 else 0) +
    (if (x / y < a / b → x / y < a / b) then 1 else 0)
  ) := by sorry

end two_true_statements_l2772_277277


namespace probability_one_black_ball_l2772_277204

def total_balls : ℕ := 4
def black_balls : ℕ := 2
def white_balls : ℕ := 2
def drawn_balls : ℕ := 2

theorem probability_one_black_ball :
  (Nat.choose black_balls 1 * Nat.choose white_balls 1) / Nat.choose total_balls drawn_balls = 2 / 3 :=
by sorry

end probability_one_black_ball_l2772_277204


namespace problem_statement_l2772_277273

theorem problem_statement (a b x y : ℝ) 
  (h1 : a*x + b*y = 2)
  (h2 : a*x^2 + b*y^2 = 5)
  (h3 : a*x^3 + b*y^3 = 10)
  (h4 : a*x^4 + b*y^4 = 30) :
  a*x^5 + b*y^5 = 40 := by sorry

end problem_statement_l2772_277273


namespace work_completion_time_l2772_277299

/-- The time taken for A, B, and C to complete the work together -/
def time_together (time_A time_B time_C : ℚ) : ℚ :=
  1 / (1 / time_A + 1 / time_B + 1 / time_C)

/-- Theorem stating that A, B, and C can complete the work together in 2 days -/
theorem work_completion_time :
  time_together 4 6 12 = 2 := by sorry

end work_completion_time_l2772_277299


namespace three_digit_two_digit_operations_l2772_277269

theorem three_digit_two_digit_operations (a b : ℕ) 
  (ha : 100 ≤ a ∧ a ≤ 999) (hb : 10 ≤ b ∧ b ≤ 99) : 
  (∀ x y, 100 ≤ x ∧ x ≤ 999 ∧ 10 ≤ y ∧ y ≤ 99 → x + y ≥ a + b) → a + b = 110 ∧
  (∀ x y, 100 ≤ x ∧ x ≤ 999 ∧ 10 ≤ y ∧ y ≤ 99 → x + y ≤ a + b) → a + b = 1098 ∧
  (∀ x y, 100 ≤ x ∧ x ≤ 999 ∧ 10 ≤ y ∧ y ≤ 99 → x - y ≥ a - b) → a - b = 1 ∧
  (∀ x y, 100 ≤ x ∧ x ≤ 999 ∧ 10 ≤ y ∧ y ≤ 99 → x - y ≤ a - b) → a - b = 989 :=
by sorry

end three_digit_two_digit_operations_l2772_277269


namespace integral_tan_cos_equality_l2772_277258

open Real MeasureTheory Interval

theorem integral_tan_cos_equality : 
  ∫ x in (-1 : ℝ)..1, (tan x)^11 + (cos x)^21 = 2 * ∫ x in (0 : ℝ)..1, (cos x)^21 := by
  sorry

end integral_tan_cos_equality_l2772_277258


namespace bat_wings_area_is_four_l2772_277222

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the "bat wings" shape -/
structure BatWings where
  rect : Rectangle
  quarterCircleRadius : ℝ

/-- Calculate the area of the "bat wings" -/
noncomputable def batWingsArea (bw : BatWings) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem bat_wings_area_is_four :
  ∀ (bw : BatWings),
    bw.rect.width = 4 ∧
    bw.rect.height = 5 ∧
    bw.quarterCircleRadius = 2 →
    batWingsArea bw = 4 := by
  sorry

end bat_wings_area_is_four_l2772_277222


namespace banana_mush_proof_l2772_277297

theorem banana_mush_proof (flour_ratio : ℝ) (total_bananas : ℝ) (total_flour : ℝ)
  (h1 : flour_ratio = 3)
  (h2 : total_bananas = 20)
  (h3 : total_flour = 15) :
  (total_bananas * flour_ratio) / total_flour = 4 := by
  sorry

end banana_mush_proof_l2772_277297


namespace positive_expression_l2772_277291

theorem positive_expression (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : -2 < y ∧ y < 0) 
  (hz : 0 < z ∧ z < 1) : 
  x + y^2 > 0 := by
  sorry

end positive_expression_l2772_277291


namespace sum_side_lengths_eq_66_l2772_277276

/-- Represents a convex quadrilateral with specific properties -/
structure ConvexQuadrilateral where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ
  -- Angle A
  angle_a : ℝ
  -- Parallel sides condition
  parallel_ab_cd : Prop
  -- Arithmetic progression condition
  arithmetic_progression : Prop
  -- AB is maximum length
  ab_max : Prop

/-- The sum of all possible values for a side length other than AB -/
def sum_possible_side_lengths (q : ConvexQuadrilateral) : ℝ := sorry

/-- Main theorem statement -/
theorem sum_side_lengths_eq_66 (q : ConvexQuadrilateral) 
  (h1 : q.ab = 18)
  (h2 : q.angle_a = 60 * π / 180)
  (h3 : q.parallel_ab_cd)
  (h4 : q.arithmetic_progression)
  (h5 : q.ab_max) :
  sum_possible_side_lengths q = 66 := by sorry

end sum_side_lengths_eq_66_l2772_277276


namespace no_real_roots_l2772_277286

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 2) + 2 = 0 := by
  sorry

end no_real_roots_l2772_277286


namespace ship_length_observation_l2772_277287

/-- The length of a ship observed from shore --/
theorem ship_length_observation (same_direction : ℝ) (opposite_direction : ℝ) :
  same_direction = 200 →
  opposite_direction = 40 →
  (∃ ship_length : ℝ, (ship_length = 100 ∨ ship_length = 200 / 3)) :=
by sorry

end ship_length_observation_l2772_277287


namespace infinite_series_sum_l2772_277233

/-- The sum of the infinite series ∑_{k=1}^∞ (k^2 / 3^k) is equal to 7/8 -/
theorem infinite_series_sum : 
  ∑' k : ℕ+, (k : ℝ)^2 / 3^(k : ℝ) = 7/8 := by sorry

end infinite_series_sum_l2772_277233


namespace debby_zoo_pictures_l2772_277224

theorem debby_zoo_pictures : ∀ (zoo_pics museum_pics deleted_pics remaining_pics : ℕ),
  museum_pics = 12 →
  deleted_pics = 14 →
  remaining_pics = 22 →
  zoo_pics + museum_pics - deleted_pics = remaining_pics →
  zoo_pics = 24 := by
sorry

end debby_zoo_pictures_l2772_277224


namespace function_value_at_negative_a_l2772_277262

/-- Given a function f(x) = ax² + bx, if f(a) = 8, then f(-a) = 8 - 2ab -/
theorem function_value_at_negative_a (a b : ℝ) :
  let f : ℝ → ℝ := fun x ↦ a * x^2 + b * x
  f a = 8 → f (-a) = 8 - 2 * a * b := by
  sorry

end function_value_at_negative_a_l2772_277262


namespace modulus_constraint_implies_range_l2772_277240

theorem modulus_constraint_implies_range (a : ℝ) :
  (∀ θ : ℝ, Complex.abs ((a + Real.cos θ) + (2 * a - Real.sin θ) * Complex.I) ≤ 2) →
  a ∈ Set.Icc (-Real.sqrt 5 / 5) (Real.sqrt 5 / 5) :=
by sorry

end modulus_constraint_implies_range_l2772_277240


namespace min_soldiers_to_add_l2772_277280

theorem min_soldiers_to_add (N : ℕ) : 
  N % 7 = 2 → N % 12 = 2 → (84 - N % 84) = 82 := by
  sorry

end min_soldiers_to_add_l2772_277280


namespace sum_of_a_and_b_l2772_277246

theorem sum_of_a_and_b (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 - b^2 = -12) : a + b = -4 := by
  sorry

end sum_of_a_and_b_l2772_277246


namespace solve_scooter_problem_l2772_277257

def scooter_problem (C : ℝ) (repair_percentage : ℝ) (profit_percentage : ℝ) (profit : ℝ) : Prop :=
  let repair_cost := repair_percentage * C
  let selling_price := (1 + profit_percentage) * C
  selling_price - C = profit ∧ 
  repair_cost = 550

theorem solve_scooter_problem :
  ∃ C : ℝ, scooter_problem C 0.1 0.2 1100 :=
sorry

end solve_scooter_problem_l2772_277257


namespace smallest_sum_of_identical_numbers_l2772_277236

theorem smallest_sum_of_identical_numbers : ∃ (a b c : ℕ), 
  (6036 = 2010 * a) ∧ 
  (6036 = 2012 * b) ∧ 
  (6036 = 2013 * c) ∧ 
  (∀ (n : ℕ) (x y z : ℕ), 
    n > 0 ∧ n < 6036 → 
    ¬(n = 2010 * x ∧ n = 2012 * y ∧ n = 2013 * z)) :=
by sorry

end smallest_sum_of_identical_numbers_l2772_277236


namespace zero_in_set_A_l2772_277294

theorem zero_in_set_A : 
  let A : Set ℕ := {0, 1, 2}
  0 ∈ A := by
sorry

end zero_in_set_A_l2772_277294


namespace same_remainder_difference_divisible_l2772_277201

theorem same_remainder_difference_divisible (a m b : ℤ) : 
  (∃ r : ℤ, a % b = r ∧ m % b = r) → b ∣ (a - m) := by sorry

end same_remainder_difference_divisible_l2772_277201


namespace cube_sum_ge_triple_product_l2772_277259

theorem cube_sum_ge_triple_product (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a^3 + b^3 + c^3 ≥ 3 * a * b * c := by
  sorry

end cube_sum_ge_triple_product_l2772_277259


namespace elijah_card_decks_l2772_277228

theorem elijah_card_decks (total_cards : ℕ) (cards_per_deck : ℕ) (h1 : total_cards = 312) (h2 : cards_per_deck = 52) :
  total_cards / cards_per_deck = 6 :=
by sorry

end elijah_card_decks_l2772_277228
