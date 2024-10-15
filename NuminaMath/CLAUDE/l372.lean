import Mathlib

namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l372_37273

theorem binomial_expansion_problem (n : ℕ) : 
  ((-2 : ℤ) ^ n = 64) →
  (n = 6 ∧ Nat.choose n 2 * 9 = 135) := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l372_37273


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l372_37224

theorem cos_x_plus_2y_equals_one 
  (x y a : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4))
  (h2 : y ∈ Set.Icc (-π/4) (π/4))
  (h3 : x^3 + Real.sin x - 2*a = 0)
  (h4 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l372_37224


namespace NUMINAMATH_CALUDE_both_in_picture_probability_l372_37295

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ  -- Time to complete one lap in seconds
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Calculates the probability of both runners being in the picture -/
def probabilityBothInPicture (sarah sam : Runner) (pictureWidth : ℝ) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem both_in_picture_probability 
  (sarah : Runner) 
  (sam : Runner) 
  (sarah_laptime : sarah.lapTime = 120)
  (sam_laptime : sam.lapTime = 75)
  (sarah_direction : sarah.direction = true)
  (sam_direction : sam.direction = false)
  (picture_width : ℝ)
  (picture_covers_third : picture_width = sarah.lapTime / 3) :
  probabilityBothInPicture sarah sam picture_width = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_both_in_picture_probability_l372_37295


namespace NUMINAMATH_CALUDE_matrix_product_is_zero_l372_37220

open Matrix

/-- Given two 3x3 matrices A and B, prove that their product is the zero matrix --/
theorem matrix_product_is_zero (a b c : ℝ) :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 2*c, -2*b; -2*c, 0, 2*a; 2*b, -2*a, 0]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![2*a^2, a^2+b^2, a^2+c^2; a^2+b^2, 2*b^2, b^2+c^2; a^2+c^2, b^2+c^2, 2*c^2]
  A * B = 0 := by
  sorry

#check matrix_product_is_zero

end NUMINAMATH_CALUDE_matrix_product_is_zero_l372_37220


namespace NUMINAMATH_CALUDE_fraction_simplification_l372_37240

theorem fraction_simplification :
  5 / (Real.sqrt 75 + 3 * Real.sqrt 5 + 2 * Real.sqrt 45) = (25 * Real.sqrt 3 - 45 * Real.sqrt 5) / 330 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l372_37240


namespace NUMINAMATH_CALUDE_abc_sum_product_l372_37233

theorem abc_sum_product (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c > 0) :
  a * b + b * c + c * a < 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_product_l372_37233


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l372_37276

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(4*x + 2) * (4 : ℝ)^(3*x + 7) = (8 : ℝ)^(5*x + 6) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l372_37276


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l372_37249

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 2*x - 3 = 0

-- Define the roots of the quadratic equation
def roots (a b : ℝ) : Prop := quadratic_eq a ∧ quadratic_eq b ∧ a ≠ b

-- Define the linear function
def linear_function (x : ℝ) (a b : ℝ) : ℝ := (a*b - 1)*x + a + b

-- Theorem: The linear function does not pass through the third quadrant
theorem linear_function_not_in_third_quadrant (a b : ℝ) (h : roots a b) :
  ∀ x y : ℝ, y = linear_function x a b → ¬(x < 0 ∧ y < 0) :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l372_37249


namespace NUMINAMATH_CALUDE_woogle_threshold_l372_37206

/-- The score for dropping n woogles -/
def drop_score (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The score for eating n woogles -/
def eat_score (n : ℕ) : ℕ := 15 * n

/-- 30 is the smallest positive integer for which dropping woogles scores more than eating them -/
theorem woogle_threshold : ∀ k : ℕ, k < 30 → drop_score k ≤ eat_score k ∧ drop_score 30 > eat_score 30 := by
  sorry

end NUMINAMATH_CALUDE_woogle_threshold_l372_37206


namespace NUMINAMATH_CALUDE_total_games_played_l372_37208

theorem total_games_played (total_teams : Nat) (rivalry_groups : Nat) (teams_per_group : Nat) (additional_games_per_team : Nat) : 
  total_teams = 50 → 
  rivalry_groups = 10 → 
  teams_per_group = 5 → 
  additional_games_per_team = 2 → 
  (total_teams * (total_teams - 1) / 2) + (rivalry_groups * teams_per_group * additional_games_per_team / 2) = 1325 := by
  sorry

end NUMINAMATH_CALUDE_total_games_played_l372_37208


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l372_37292

/-- 
Calculates the number of games in a single-elimination tournament.
num_teams: The number of teams in the tournament.
-/
def num_games (num_teams : ℕ) : ℕ :=
  num_teams - 1

theorem single_elimination_tournament_games :
  num_games 16 = 15 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l372_37292


namespace NUMINAMATH_CALUDE_problem_solution_l372_37291

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 6) 
  (h3 : x = -6) : 
  y = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l372_37291


namespace NUMINAMATH_CALUDE_basketball_win_rate_l372_37200

theorem basketball_win_rate (games_won : ℕ) (first_games : ℕ) (total_games : ℕ) (remaining_games : ℕ) (win_rate : ℚ) : 
  games_won = 25 ∧ 
  first_games = 35 ∧ 
  total_games = 60 ∧ 
  remaining_games = 25 ∧ 
  win_rate = 4/5 →
  (games_won + remaining_games : ℚ) / total_games = win_rate ↔ 
  remaining_games = 23 := by
sorry

end NUMINAMATH_CALUDE_basketball_win_rate_l372_37200


namespace NUMINAMATH_CALUDE_son_is_eighteen_l372_37211

theorem son_is_eighteen (father_age son_age : ℕ) : 
  father_age + son_age = 55 →
  ∃ (y : ℕ), father_age + y + (son_age + y) = 93 ∧ son_age + y = father_age →
  (father_age = 18 ∨ son_age = 18) →
  son_age = 18 := by
sorry

end NUMINAMATH_CALUDE_son_is_eighteen_l372_37211


namespace NUMINAMATH_CALUDE_correct_propositions_count_l372_37254

-- Define the types of events
inductive EventType
  | Certain
  | Impossible
  | Random

-- Define the propositions
def proposition1 : EventType := EventType.Certain
def proposition2 : EventType := EventType.Impossible
def proposition3 : EventType := EventType.Certain
def proposition4 : EventType := EventType.Random

-- Define a function to check if a proposition is correct
def is_correct (prop : EventType) : Bool :=
  match prop with
  | EventType.Certain => true
  | EventType.Impossible => true
  | EventType.Random => true

-- Theorem: The number of correct propositions is 3
theorem correct_propositions_count :
  (is_correct proposition1).toNat +
  (is_correct proposition2).toNat +
  (is_correct proposition3).toNat +
  (is_correct proposition4).toNat = 3 := by
  sorry


end NUMINAMATH_CALUDE_correct_propositions_count_l372_37254


namespace NUMINAMATH_CALUDE_benny_total_spend_l372_37267

def total_cost (soft_drink_cost : ℕ) (candy_bar_cost : ℕ) (num_candy_bars : ℕ) : ℕ :=
  soft_drink_cost + candy_bar_cost * num_candy_bars

theorem benny_total_spend :
  let soft_drink_cost : ℕ := 2
  let candy_bar_cost : ℕ := 5
  let num_candy_bars : ℕ := 5
  total_cost soft_drink_cost candy_bar_cost num_candy_bars = 27 := by
sorry

end NUMINAMATH_CALUDE_benny_total_spend_l372_37267


namespace NUMINAMATH_CALUDE_rationalize_sqrt_sum_l372_37259

def rationalize_and_simplify (x y z : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  sorry

theorem rationalize_sqrt_sum : 
  let (A, B, C, D, E, F) := rationalize_and_simplify 5 2 7
  A + B + C + D + E + F = 84 := by sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_sum_l372_37259


namespace NUMINAMATH_CALUDE_outfit_combinations_l372_37232

def num_shirts : ℕ := 5
def num_pants : ℕ := 6
def restricted_combinations : ℕ := 2

theorem outfit_combinations :
  let total_combinations := num_shirts * num_pants
  let restricted_shirt_combinations := num_pants - restricted_combinations
  let unrestricted_combinations := (num_shirts - 1) * num_pants
  unrestricted_combinations + restricted_shirt_combinations = 28 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l372_37232


namespace NUMINAMATH_CALUDE_largest_M_has_property_l372_37207

/-- The property that for any 10 distinct real numbers in [1, M], 
    there exist three that form a quadratic with no real roots -/
def has_property (M : ℝ) : Prop :=
  ∀ (a : Fin 10 → ℝ), (∀ i j, i ≠ j → a i ≠ a j) → 
  (∀ i, 1 ≤ a i ∧ a i ≤ M) →
  ∃ i j k, i < j ∧ j < k ∧ a i < a j ∧ a j < a k ∧
  (a j)^2 < 4 * (a i) * (a k)

/-- The largest integer M > 1 with the property -/
def largest_M : ℕ := 4^255

theorem largest_M_has_property :
  (has_property (largest_M : ℝ)) ∧
  ∀ n : ℕ, n > largest_M → ¬(has_property (n : ℝ)) :=
by sorry


end NUMINAMATH_CALUDE_largest_M_has_property_l372_37207


namespace NUMINAMATH_CALUDE_altitude_of_triangle_on_rectangle_diagonal_l372_37202

theorem altitude_of_triangle_on_rectangle_diagonal (l : ℝ) (h : l > 0) :
  let w := l * Real.sqrt 2 / 2
  let diagonal := Real.sqrt (l^2 + w^2)
  let rectangle_area := l * w
  let triangle_area := diagonal * altitude / 2
  triangle_area = rectangle_area →
  altitude = l * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_altitude_of_triangle_on_rectangle_diagonal_l372_37202


namespace NUMINAMATH_CALUDE_students_painting_l372_37280

theorem students_painting (green red both : ℕ) 
  (h1 : green = 52)
  (h2 : red = 56)
  (h3 : both = 38) :
  green + red - both = 70 := by
  sorry

end NUMINAMATH_CALUDE_students_painting_l372_37280


namespace NUMINAMATH_CALUDE_find_p_l372_37251

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 20

-- State the theorem
theorem find_p : ∃ p : ℝ, f (f (f p)) = 6 ∧ p = 18.25 := by
  sorry

end NUMINAMATH_CALUDE_find_p_l372_37251


namespace NUMINAMATH_CALUDE_point_satisfies_conditions_l372_37266

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of the line y = -2x + 3 -/
def on_line (p : Point) : Prop :=
  p.y = -2 * p.x + 3

/-- The condition for a point to be in the first quadrant -/
def in_first_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The point (1, 1) -/
def point : Point :=
  { x := 1, y := 1 }

theorem point_satisfies_conditions :
  in_first_quadrant point ∧ on_line point :=
by sorry

end NUMINAMATH_CALUDE_point_satisfies_conditions_l372_37266


namespace NUMINAMATH_CALUDE_rice_distribution_l372_37260

theorem rice_distribution (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_weight = 25 / 4 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight * ounces_per_pound) / num_containers = 25 := by
  sorry

end NUMINAMATH_CALUDE_rice_distribution_l372_37260


namespace NUMINAMATH_CALUDE_lottery_winnings_l372_37201

/-- Calculates the total winnings for lottery tickets -/
theorem lottery_winnings
  (num_tickets : ℕ)
  (winning_numbers_per_ticket : ℕ)
  (value_per_winning_number : ℕ)
  (h1 : num_tickets = 3)
  (h2 : winning_numbers_per_ticket = 5)
  (h3 : value_per_winning_number = 20) :
  num_tickets * winning_numbers_per_ticket * value_per_winning_number = 300 :=
by sorry

end NUMINAMATH_CALUDE_lottery_winnings_l372_37201


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l372_37265

theorem sufficient_not_necessary (x : ℝ) : 
  (x ≥ (1/2) → 2*x^2 + x - 1 ≥ 0) ∧ 
  ¬(2*x^2 + x - 1 ≥ 0 → x ≥ (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l372_37265


namespace NUMINAMATH_CALUDE_james_oreos_count_l372_37231

/-- The number of Oreos Jordan has -/
def jordan_oreos : ℕ := sorry

/-- The number of Oreos James has -/
def james_oreos : ℕ := 4 * jordan_oreos + 7

/-- The total number of Oreos -/
def total_oreos : ℕ := 52

theorem james_oreos_count : james_oreos = 43 := by
  sorry

end NUMINAMATH_CALUDE_james_oreos_count_l372_37231


namespace NUMINAMATH_CALUDE_square_sum_equation_l372_37226

theorem square_sum_equation (n m : ℕ) (h : n ^ 2 = (Finset.range (m - 99)).sum (λ i => i + 100)) : n + m = 497 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equation_l372_37226


namespace NUMINAMATH_CALUDE_original_slices_count_l372_37297

/-- The number of slices in the original loaf of bread -/
def S : ℕ := 27

/-- The number of slices Andy ate -/
def slices_andy_ate : ℕ := 6

/-- The number of slices Emma used for toast -/
def slices_for_toast : ℕ := 20

/-- The number of slices left after making toast -/
def slices_left : ℕ := 1

/-- Theorem stating that the original number of slices equals the sum of slices eaten,
    used for toast, and left over -/
theorem original_slices_count : S = slices_andy_ate + slices_for_toast + slices_left :=
by sorry

end NUMINAMATH_CALUDE_original_slices_count_l372_37297


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_28_l372_37219

def digit_sum (n : ℕ) : ℕ := sorry

def is_prime (n : ℕ) : Prop := sorry

theorem smallest_prime_with_digit_sum_28 :
  (is_prime 1999) ∧ 
  (digit_sum 1999 = 28) ∧ 
  (∀ m : ℕ, m < 1999 → (is_prime m ∧ digit_sum m = 28) → False) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_28_l372_37219


namespace NUMINAMATH_CALUDE_maria_cookies_l372_37241

theorem maria_cookies (cookies_per_bag : ℕ) (chocolate_chip : ℕ) (baggies : ℕ) 
  (h1 : cookies_per_bag = 8)
  (h2 : chocolate_chip = 5)
  (h3 : baggies = 3) :
  cookies_per_bag * baggies - chocolate_chip = 19 := by
  sorry

end NUMINAMATH_CALUDE_maria_cookies_l372_37241


namespace NUMINAMATH_CALUDE_two_numbers_difference_l372_37216

theorem two_numbers_difference (a b : ℕ) : 
  a + b = 20000 →
  b % 5 = 0 →
  b / 10 = a →
  (b % 10 = 0 ∨ b % 10 = 5) →
  b - a = 16358 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l372_37216


namespace NUMINAMATH_CALUDE_inequality_proof_l372_37215

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  (x^3 + 2*y^2 + 3*z) * (4*y^3 + 5*z^2 + 6*x) * (7*z^3 + 8*x^2 + 9*y) ≥ 720 * (x*y + y*z + x*z) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l372_37215


namespace NUMINAMATH_CALUDE_rectangle_y_value_l372_37203

theorem rectangle_y_value (y : ℝ) : 
  let vertices : List (ℝ × ℝ) := [(-2, y), (6, y), (-2, 2), (6, 2)]
  let length : ℝ := 6 - (-2)
  let height : ℝ := y - 2
  let area : ℝ := 64
  (length * height = area) → y = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l372_37203


namespace NUMINAMATH_CALUDE_carrie_leftover_money_l372_37238

/-- Calculates the amount of money Carrie has left after purchasing a bike, helmet, and accessories --/
theorem carrie_leftover_money 
  (hourly_rate : ℝ)
  (hours_per_week : ℝ)
  (weeks_worked : ℝ)
  (bike_cost : ℝ)
  (sales_tax_rate : ℝ)
  (helmet_cost : ℝ)
  (accessories_cost : ℝ)
  (h1 : hourly_rate = 8)
  (h2 : hours_per_week = 35)
  (h3 : weeks_worked = 4)
  (h4 : bike_cost = 400)
  (h5 : sales_tax_rate = 0.06)
  (h6 : helmet_cost = 50)
  (h7 : accessories_cost = 30) :
  hourly_rate * hours_per_week * weeks_worked - 
  (bike_cost * (1 + sales_tax_rate) + helmet_cost + accessories_cost) = 616 :=
by sorry

end NUMINAMATH_CALUDE_carrie_leftover_money_l372_37238


namespace NUMINAMATH_CALUDE_lanas_final_pages_l372_37228

def lanas_pages (initial_pages : ℕ) (duanes_pages : ℕ) : ℕ :=
  initial_pages + duanes_pages / 2

theorem lanas_final_pages :
  lanas_pages 8 42 = 29 := by sorry

end NUMINAMATH_CALUDE_lanas_final_pages_l372_37228


namespace NUMINAMATH_CALUDE_bob_profit_l372_37274

/-- Calculates the profit from breeding and selling show dogs -/
def dog_breeding_profit (num_dogs : ℕ) (cost_per_dog : ℚ) (num_puppies : ℕ) (price_per_puppy : ℚ) : ℚ :=
  num_puppies * price_per_puppy - num_dogs * cost_per_dog

/-- Bob's profit from breeding and selling show dogs -/
theorem bob_profit : 
  dog_breeding_profit 2 250 6 350 = 1600 :=
by sorry

end NUMINAMATH_CALUDE_bob_profit_l372_37274


namespace NUMINAMATH_CALUDE_rhombus_prism_lateral_area_l372_37282

/-- The lateral surface area of a right prism with a rhombus base and given dimensions. -/
theorem rhombus_prism_lateral_area (d1 d2 h : ℝ) (hd1 : d1 = 9) (hd2 : d2 = 15) (hh : h = 5) :
  4 * (((d1 ^ 2 / 4 + d2 ^ 2 / 4) : ℝ).sqrt) * h = 160 :=
by sorry

#check rhombus_prism_lateral_area

end NUMINAMATH_CALUDE_rhombus_prism_lateral_area_l372_37282


namespace NUMINAMATH_CALUDE_dealer_articles_purchased_l372_37258

theorem dealer_articles_purchased
  (total_purchase_price : ℝ)
  (num_articles_sold : ℕ)
  (total_selling_price : ℝ)
  (profit_percentage : ℝ)
  (h1 : total_purchase_price = 25)
  (h2 : num_articles_sold = 12)
  (h3 : total_selling_price = 33)
  (h4 : profit_percentage = 0.65)
  : ∃ (num_articles_purchased : ℕ),
    (num_articles_purchased : ℝ) * (total_selling_price / num_articles_sold) =
    total_purchase_price * (1 + profit_percentage) ∧
    num_articles_purchased = 15 :=
by sorry

end NUMINAMATH_CALUDE_dealer_articles_purchased_l372_37258


namespace NUMINAMATH_CALUDE_pauls_remaining_crayons_l372_37286

/-- Given that Paul initially had 479 crayons and lost or gave away 345 crayons,
    prove that he has 134 crayons left. -/
theorem pauls_remaining_crayons (initial : ℕ) (lost : ℕ) (remaining : ℕ) 
    (h1 : initial = 479) 
    (h2 : lost = 345) 
    (h3 : remaining = initial - lost) : 
  remaining = 134 := by
  sorry

end NUMINAMATH_CALUDE_pauls_remaining_crayons_l372_37286


namespace NUMINAMATH_CALUDE_joel_peppers_l372_37239

/-- Represents the number of peppers picked each day of the week -/
structure WeeklyPeppers where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the number of non-hot peppers given the weekly pepper count -/
def nonHotPeppers (w : WeeklyPeppers) : ℕ :=
  let total := w.sunday + w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday
  (total * 4) / 5

theorem joel_peppers :
  let w : WeeklyPeppers := {
    sunday := 7,
    monday := 12,
    tuesday := 14,
    wednesday := 12,
    thursday := 5,
    friday := 18,
    saturday := 12
  }
  nonHotPeppers w = 64 := by
  sorry

end NUMINAMATH_CALUDE_joel_peppers_l372_37239


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l372_37269

/-- A point is in the second quadrant if its x-coordinate is negative and its y-coordinate is positive -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The point P(-1, m^2+1) is in the second quadrant for any real number m -/
theorem point_in_second_quadrant (m : ℝ) : is_in_second_quadrant (-1) (m^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l372_37269


namespace NUMINAMATH_CALUDE_certain_number_proof_l372_37284

/-- Given that 213 * 16 = 3408, prove that the number x satisfying x * 2.13 = 0.03408 is equal to 0.016 -/
theorem certain_number_proof (h : 213 * 16 = 3408) : 
  ∃ x : ℝ, x * 2.13 = 0.03408 ∧ x = 0.016 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l372_37284


namespace NUMINAMATH_CALUDE_fraction_problem_l372_37296

theorem fraction_problem (f : ℚ) : 
  (f * 20 + 5 = 15) → f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l372_37296


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l372_37277

theorem condition_sufficient_not_necessary (a b c : ℝ) :
  (∀ a b c : ℝ, a > b ∧ c > 0 → a * c > b * c) ∧
  ¬(∀ a b c : ℝ, a * c > b * c → a > b ∧ c > 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l372_37277


namespace NUMINAMATH_CALUDE_additional_gas_needed_l372_37218

/-- Calculates the additional gallons of gas needed for a truck to reach its destination. -/
theorem additional_gas_needed
  (miles_per_gallon : ℝ)
  (total_distance : ℝ)
  (current_gas : ℝ)
  (h1 : miles_per_gallon = 3)
  (h2 : total_distance = 90)
  (h3 : current_gas = 12) :
  (total_distance - current_gas * miles_per_gallon) / miles_per_gallon = 18 := by
  sorry

end NUMINAMATH_CALUDE_additional_gas_needed_l372_37218


namespace NUMINAMATH_CALUDE_ferry_speed_proof_l372_37223

/-- The speed of ferry P in km/h -/
def speed_P : ℝ := 6

/-- The speed of ferry Q in km/h -/
def speed_Q : ℝ := speed_P + 3

/-- The time taken by ferry P in hours -/
def time_P : ℝ := 3

/-- The time taken by ferry Q in hours -/
def time_Q : ℝ := time_P + 3

/-- The distance traveled by ferry P in km -/
def distance_P : ℝ := speed_P * time_P

/-- The distance traveled by ferry Q in km -/
def distance_Q : ℝ := 3 * distance_P

theorem ferry_speed_proof :
  speed_P = 6 ∧
  speed_Q = speed_P + 3 ∧
  time_Q = time_P + 3 ∧
  distance_Q = 3 * distance_P ∧
  distance_P = speed_P * time_P ∧
  distance_Q = speed_Q * time_Q :=
by sorry

end NUMINAMATH_CALUDE_ferry_speed_proof_l372_37223


namespace NUMINAMATH_CALUDE_last_locker_opened_l372_37227

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Toggles the state of a locker -/
def toggleLocker (state : LockerState) : LockerState :=
  match state with
  | LockerState.Open => LockerState.Closed
  | LockerState.Closed => LockerState.Open

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

/-- The main theorem stating that the last locker opened is 509 -/
theorem last_locker_opened :
  ∀ (n : Nat), n ≤ 512 →
    (isPerfectSquare n ↔ (
      ∀ (k : Nat), k ≤ 512 →
        (n % k = 0 → toggleLocker (
          if k = 1 then LockerState.Closed
          else if k < n then toggleLocker LockerState.Closed
          else LockerState.Closed
        ) = LockerState.Open)
    )) →
  (∀ m : Nat, m > 509 ∧ m ≤ 512 →
    ¬(∀ (k : Nat), k ≤ 512 →
      (m % k = 0 → toggleLocker (
        if k = 1 then LockerState.Closed
        else if k < m then toggleLocker LockerState.Closed
        else LockerState.Closed
      ) = LockerState.Open))) →
  isPerfectSquare 509 :=
by
  sorry


end NUMINAMATH_CALUDE_last_locker_opened_l372_37227


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l372_37229

theorem fraction_to_decimal :
  (7 : ℚ) / 12 = 0.5833333333333333333333333333333333 :=
sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l372_37229


namespace NUMINAMATH_CALUDE_five_workers_required_l372_37285

/-- Represents the project parameters and progress -/
structure ProjectStatus :=
  (total_days : ℕ)
  (elapsed_days : ℕ)
  (initial_workers : ℕ)
  (completed_fraction : ℚ)

/-- Calculates the minimum number of workers required to complete the project on schedule -/
def min_workers_required (status : ProjectStatus) : ℕ :=
  sorry

/-- Theorem stating that for the given project status, 5 workers are required -/
theorem five_workers_required (status : ProjectStatus) 
  (h1 : status.total_days = 20)
  (h2 : status.elapsed_days = 5)
  (h3 : status.initial_workers = 10)
  (h4 : status.completed_fraction = 1/4) :
  min_workers_required status = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_workers_required_l372_37285


namespace NUMINAMATH_CALUDE_sarah_skateboard_speed_l372_37205

/-- Given the following conditions:
1. Pete walks backwards three times faster than Susan walks forwards.
2. Tracy does one-handed cartwheels twice as fast as Susan walks forwards.
3. Mike swims eight times faster than Tracy does cartwheels.
4. Pete can walk on his hands at only one quarter of the speed that Tracy can do cartwheels.
5. Pete can ride his bike five times faster than Mike swims.
6. Pete walks on his hands at 2 miles per hour.
7. Patty can row three times faster than Pete walks backwards.
8. Sarah can skateboard six times faster than Patty rows.

Prove that Sarah can skateboard at 216 miles per hour. -/
theorem sarah_skateboard_speed :
  ∀ (pete_backward_speed pete_hand_speed pete_bike_speed susan_speed tracy_speed
     mike_speed patty_speed sarah_speed : ℝ),
  pete_backward_speed = 3 * susan_speed →
  tracy_speed = 2 * susan_speed →
  mike_speed = 8 * tracy_speed →
  pete_hand_speed = 1/4 * tracy_speed →
  pete_bike_speed = 5 * mike_speed →
  pete_hand_speed = 2 →
  patty_speed = 3 * pete_backward_speed →
  sarah_speed = 6 * patty_speed →
  sarah_speed = 216 := by
  sorry

end NUMINAMATH_CALUDE_sarah_skateboard_speed_l372_37205


namespace NUMINAMATH_CALUDE_linear_functions_intersection_l372_37230

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Evaluate a linear function at a given x -/
def LinearFunction.eval (f : LinearFunction) (x : ℝ) : ℝ :=
  f.slope * x + f.intercept

theorem linear_functions_intersection (f₁ f₂ : LinearFunction) :
  (f₁.eval 2 = f₂.eval 2) →
  (|f₁.eval 8 - f₂.eval 8| = 8) →
  ((f₁.eval 20 = 100) ∨ (f₂.eval 20 = 100)) →
  ((f₁.eval 20 = 76 ∧ f₂.eval 20 = 100) ∨ (f₁.eval 20 = 100 ∧ f₂.eval 20 = 124) ∨
   (f₁.eval 20 = 100 ∧ f₂.eval 20 = 76) ∨ (f₁.eval 20 = 124 ∧ f₂.eval 20 = 100)) := by
  sorry

end NUMINAMATH_CALUDE_linear_functions_intersection_l372_37230


namespace NUMINAMATH_CALUDE_spade_nested_calculation_l372_37294

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_nested_calculation : spade 3 (spade 4 5) = -72 := by
  sorry

end NUMINAMATH_CALUDE_spade_nested_calculation_l372_37294


namespace NUMINAMATH_CALUDE_wade_sandwich_cost_l372_37237

def sandwich_cost (total_spent : ℚ) (num_sandwiches : ℕ) (num_drinks : ℕ) (drink_cost : ℚ) : ℚ :=
  (total_spent - (num_drinks : ℚ) * drink_cost) / (num_sandwiches : ℚ)

theorem wade_sandwich_cost :
  sandwich_cost 26 3 2 4 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_wade_sandwich_cost_l372_37237


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l372_37245

theorem magnitude_of_complex_power : 
  Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 6) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l372_37245


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l372_37244

-- System 1
theorem system_one_solution (x : ℝ) :
  (2 * x + 10 ≤ 5 * x + 1 ∧ 3 * (x - 1) > 9) ↔ x > 4 := by sorry

-- System 2
theorem system_two_solution (x : ℝ) :
  (3 * (x + 2) ≥ 2 * x + 5 ∧ 2 * x - (3 * x + 1) / 2 < 1) ↔ -1 ≤ x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l372_37244


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l372_37250

theorem square_area_equal_perimeter_triangle (a b c s : ℝ) : 
  a = 7.5 ∧ b = 9.3 ∧ c = 12.2 → -- triangle side lengths
  s * 4 = a + b + c →           -- equal perimeters
  s * s = 52.5625 :=            -- square area
by sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l372_37250


namespace NUMINAMATH_CALUDE_angle_three_times_complement_l372_37272

theorem angle_three_times_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_three_times_complement_l372_37272


namespace NUMINAMATH_CALUDE_whiteboard_washing_time_l372_37281

theorem whiteboard_washing_time 
  (kids : ℕ) 
  (whiteboards : ℕ) 
  (time : ℕ) 
  (h1 : kids = 4) 
  (h2 : whiteboards = 3) 
  (h3 : time = 20) :
  (1 : ℕ) * 6 * time = kids * whiteboards * 160 :=
sorry

end NUMINAMATH_CALUDE_whiteboard_washing_time_l372_37281


namespace NUMINAMATH_CALUDE_exists_valid_numbering_9_not_exists_valid_numbering_10_l372_37264

/-- A convex n-gon with a point inside -/
structure ConvexNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  inner_point : ℝ × ℝ
  is_convex : sorry -- Add convexity condition

/-- Numbering of sides and segments -/
def Numbering (n : ℕ) := Fin n → Fin n

/-- Sum of numbers in a triangle -/
def triangle_sum (n : ℕ) (polygon : ConvexNGon n) (numbering : Numbering n) (i : Fin n) : ℕ := sorry

/-- Existence of a valid numbering for n = 9 -/
theorem exists_valid_numbering_9 :
  ∃ (polygon : ConvexNGon 9) (numbering : Numbering 9),
    ∀ (i j : Fin 9), triangle_sum 9 polygon numbering i = triangle_sum 9 polygon numbering j :=
sorry

/-- Non-existence of a valid numbering for n = 10 -/
theorem not_exists_valid_numbering_10 :
  ¬ ∃ (polygon : ConvexNGon 10) (numbering : Numbering 10),
    ∀ (i j : Fin 10), triangle_sum 10 polygon numbering i = triangle_sum 10 polygon numbering j :=
sorry

end NUMINAMATH_CALUDE_exists_valid_numbering_9_not_exists_valid_numbering_10_l372_37264


namespace NUMINAMATH_CALUDE_job_fair_theorem_l372_37287

/-- Represents a candidate in the job fair --/
structure Candidate where
  correct_answers : ℕ
  prob_correct : ℚ

/-- The job fair scenario --/
structure JobFair where
  total_questions : ℕ
  selected_questions : ℕ
  candidate_a : Candidate
  candidate_b : Candidate

/-- Calculates the probability of a specific sequence of answers for candidate A --/
def prob_sequence (jf : JobFair) : ℚ :=
  (1 - jf.candidate_a.correct_answers / jf.total_questions) *
  (jf.candidate_a.correct_answers / (jf.total_questions - 1)) *
  ((jf.candidate_a.correct_answers - 1) / (jf.total_questions - 2))

/-- Calculates the variance of correct answers for candidate A --/
def variance_a (jf : JobFair) : ℚ := sorry

/-- Calculates the variance of correct answers for candidate B --/
def variance_b (jf : JobFair) : ℚ := sorry

/-- The main theorem to be proved --/
theorem job_fair_theorem (jf : JobFair)
    (h1 : jf.total_questions = 8)
    (h2 : jf.selected_questions = 3)
    (h3 : jf.candidate_a.correct_answers = 6)
    (h4 : jf.candidate_b.prob_correct = 3/4) :
    prob_sequence jf = 5/7 ∧ variance_a jf < variance_b jf := by
  sorry

end NUMINAMATH_CALUDE_job_fair_theorem_l372_37287


namespace NUMINAMATH_CALUDE_belinda_passed_twenty_percent_l372_37270

/-- The percentage of flyers Belinda passed out -/
def belinda_percentage (total flyers : ℕ) (ryan alyssa scott : ℕ) : ℚ :=
  (total - (ryan + alyssa + scott)) / total * 100

/-- Theorem stating that Belinda passed out 20% of the flyers -/
theorem belinda_passed_twenty_percent :
  belinda_percentage 200 42 67 51 = 20 := by
  sorry

end NUMINAMATH_CALUDE_belinda_passed_twenty_percent_l372_37270


namespace NUMINAMATH_CALUDE_power_of_power_l372_37209

theorem power_of_power (x : ℝ) : (x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l372_37209


namespace NUMINAMATH_CALUDE_product_of_roots_l372_37248

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 5) = 24 → ∃ y : ℝ, (x + 3) * (x - 5) = 24 ∧ (x * y = -39) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l372_37248


namespace NUMINAMATH_CALUDE_can_display_total_l372_37288

def triangle_display (n : ℕ) (first_row : ℕ) (increment : ℕ) : ℕ :=
  (n * (2 * first_row + (n - 1) * increment)) / 2

theorem can_display_total :
  let n := 9  -- number of rows
  let seventh_row := 19  -- number of cans in the seventh row
  let increment := 3  -- difference in cans between adjacent rows
  let first_row := seventh_row - 6 * increment  -- number of cans in the first row
  triangle_display n first_row increment = 117 :=
by
  sorry

end NUMINAMATH_CALUDE_can_display_total_l372_37288


namespace NUMINAMATH_CALUDE_car_cost_proof_l372_37268

def down_payment : ℕ := 8000
def num_payments : ℕ := 48
def monthly_payment : ℕ := 525
def interest_rate : ℚ := 5 / 100

def total_car_cost : ℕ := 34460

theorem car_cost_proof :
  down_payment +
  num_payments * monthly_payment +
  num_payments * (interest_rate * monthly_payment).floor = total_car_cost := by
  sorry

end NUMINAMATH_CALUDE_car_cost_proof_l372_37268


namespace NUMINAMATH_CALUDE_even_decreasing_inequality_l372_37204

-- Define an even function f: ℝ → ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define a decreasing function on [0, +∞)
def decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f y < f x

-- Theorem statement
theorem even_decreasing_inequality (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_decreasing : decreasing_on_nonneg f) : 
  f 3 < f (-2) ∧ f (-2) < f 1 :=
sorry

end NUMINAMATH_CALUDE_even_decreasing_inequality_l372_37204


namespace NUMINAMATH_CALUDE_inequality_proof_l372_37261

theorem inequality_proof (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_condition : x * y + y * z + z * x = 1) : 
  (1 / (x + y)) + (1 / (y + z)) + (1 / (z + x)) ≥ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l372_37261


namespace NUMINAMATH_CALUDE_impossible_to_reach_target_l372_37255

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- The initial grid state with all zeros -/
def initial_grid : Grid := fun _ _ => 0

/-- Represents a 2x2 subgrid position in the 3x3 grid -/
inductive SubgridPos
| TopLeft
| TopRight
| BottomLeft
| BottomRight

/-- Applies a 2x2 increment operation to the grid at the specified position -/
def apply_operation (g : Grid) (pos : SubgridPos) : Grid :=
  fun i j =>
    match pos with
    | SubgridPos.TopLeft => if i < 2 && j < 2 then g i j + 1 else g i j
    | SubgridPos.TopRight => if i < 2 && j > 0 then g i j + 1 else g i j
    | SubgridPos.BottomLeft => if i > 0 && j < 2 then g i j + 1 else g i j
    | SubgridPos.BottomRight => if i > 0 && j > 0 then g i j + 1 else g i j

/-- The target grid state we want to prove is impossible to reach -/
def target_grid : Grid :=
  fun i j => if i = 1 && j = 1 then 4 else 1

/-- Theorem stating that it's impossible to reach the target grid from the initial grid
    using any sequence of 2x2 increment operations -/
theorem impossible_to_reach_target :
  ∀ (ops : List SubgridPos),
    (ops.foldl apply_operation initial_grid) ≠ target_grid :=
sorry

end NUMINAMATH_CALUDE_impossible_to_reach_target_l372_37255


namespace NUMINAMATH_CALUDE_apple_purchase_theorem_l372_37235

/-- Represents the cost in cents for a pack of apples --/
structure ApplePack where
  count : ℕ
  cost : ℕ

/-- Represents a purchase of apple packs --/
structure Purchase where
  pack : ApplePack
  quantity : ℕ

def total_apples (purchases : List Purchase) : ℕ :=
  purchases.foldl (fun acc p => acc + p.pack.count * p.quantity) 0

def total_cost (purchases : List Purchase) : ℕ :=
  purchases.foldl (fun acc p => acc + p.pack.cost * p.quantity) 0

def average_cost (purchases : List Purchase) : ℚ :=
  (total_cost purchases : ℚ) / (total_apples purchases : ℚ)

theorem apple_purchase_theorem (scheme1 scheme2 : ApplePack) 
  (purchase1 purchase2 : Purchase) : 
  scheme1.count = 4 → 
  scheme1.cost = 15 → 
  scheme2.count = 7 → 
  scheme2.cost = 28 → 
  purchase1.pack = scheme2 → 
  purchase1.quantity = 4 → 
  purchase2.pack = scheme1 → 
  purchase2.quantity = 2 → 
  total_cost [purchase1, purchase2] = 142 ∧ 
  average_cost [purchase1, purchase2] = 5.0714 := by
  sorry

end NUMINAMATH_CALUDE_apple_purchase_theorem_l372_37235


namespace NUMINAMATH_CALUDE_no_solutions_in_interval_l372_37298

theorem no_solutions_in_interval (x : ℝ) :
  -π ≤ x ∧ x ≤ 3*π →
  ¬(1 / Real.sin x + 1 / Real.cos x = 4) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_in_interval_l372_37298


namespace NUMINAMATH_CALUDE_blue_pill_cost_proof_l372_37263

def treatment_duration : ℕ := 3 * 7 -- 3 weeks in days

def daily_blue_pills : ℕ := 1
def daily_yellow_pills : ℕ := 1

def total_cost : ℚ := 735

def blue_pill_cost : ℚ := 18.5
def yellow_pill_cost : ℚ := blue_pill_cost - 2

theorem blue_pill_cost_proof :
  blue_pill_cost * (treatment_duration * daily_blue_pills) +
  yellow_pill_cost * (treatment_duration * daily_yellow_pills) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_blue_pill_cost_proof_l372_37263


namespace NUMINAMATH_CALUDE_lyceum_students_count_l372_37217

theorem lyceum_students_count :
  ∀ n : ℕ,
  (1000 < n ∧ n < 2000) →
  (n * 76 % 100 = 0) →
  (n * 5 % 37 = 0) →
  n = 1850 :=
by
  sorry

end NUMINAMATH_CALUDE_lyceum_students_count_l372_37217


namespace NUMINAMATH_CALUDE_fraction_product_l372_37271

theorem fraction_product : (2 : ℚ) / 9 * 5 / 14 = 5 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l372_37271


namespace NUMINAMATH_CALUDE_difference_in_tickets_l372_37221

def tickets_for_toys : ℕ := 31
def tickets_for_clothes : ℕ := 14

theorem difference_in_tickets : tickets_for_toys - tickets_for_clothes = 17 := by
  sorry

end NUMINAMATH_CALUDE_difference_in_tickets_l372_37221


namespace NUMINAMATH_CALUDE_rafting_and_tubing_count_l372_37293

theorem rafting_and_tubing_count (total_kids : ℕ) 
  (h1 : total_kids = 40) 
  (tubing_fraction : ℚ) 
  (h2 : tubing_fraction = 1/4) 
  (rafting_fraction : ℚ) 
  (h3 : rafting_fraction = 1/2) : ℕ :=
  let tubing_kids := (total_kids : ℚ) * tubing_fraction
  let rafting_and_tubing_kids := tubing_kids * rafting_fraction
  5

#check rafting_and_tubing_count

end NUMINAMATH_CALUDE_rafting_and_tubing_count_l372_37293


namespace NUMINAMATH_CALUDE_abs_product_of_neg_two_and_four_l372_37213

theorem abs_product_of_neg_two_and_four :
  ∀ x y : ℤ, x = -2 → y = 4 → |x * y| = 8 := by
  sorry

end NUMINAMATH_CALUDE_abs_product_of_neg_two_and_four_l372_37213


namespace NUMINAMATH_CALUDE_min_value_g_l372_37257

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points -/
def distance (p q : Point3D) : ℝ := sorry

/-- Definition of the tetrahedron EFGH -/
def Tetrahedron (E F G H : Point3D) : Prop :=
  distance E H = 30 ∧
  distance F G = 30 ∧
  distance E G = 40 ∧
  distance F H = 40 ∧
  distance E F = 48 ∧
  distance G H = 48

/-- Function g(Y) as defined in the problem -/
def g (E F G H Y : Point3D) : ℝ :=
  distance E Y + distance F Y + distance G Y + distance H Y

/-- Theorem stating the minimum value of g(Y) -/
theorem min_value_g (E F G H : Point3D) :
  Tetrahedron E F G H →
  ∃ (min : ℝ), min = 4 * Real.sqrt 578 ∧
    ∀ (Y : Point3D), g E F G H Y ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_g_l372_37257


namespace NUMINAMATH_CALUDE_mortgage_payment_sum_l372_37214

theorem mortgage_payment_sum (a₁ : ℝ) (r : ℝ) (n : ℕ) (h1 : a₁ = 400) (h2 : r = 2) (h3 : n = 11) :
  a₁ * (1 - r^n) / (1 - r) = 819400 := by
  sorry

end NUMINAMATH_CALUDE_mortgage_payment_sum_l372_37214


namespace NUMINAMATH_CALUDE_parabola_equation_l372_37247

/-- A parabola with a vertical axis of symmetry -/
structure VerticalParabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The equation of a vertical parabola in vertex form -/
def VerticalParabola.equation (p : VerticalParabola) (x y : ℝ) : Prop :=
  y = p.a * (x - p.h)^2 + p.k

/-- The equation of a vertical parabola in standard form -/
def VerticalParabola.standardForm (p : VerticalParabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + (-2 * p.a * p.h) * x + (p.a * p.h^2 + p.k)

theorem parabola_equation (p : VerticalParabola) (h_vertex : p.h = 3 ∧ p.k = -2)
    (h_point : p.equation 5 6) :
  p.standardForm x y ↔ y = 2 * x^2 - 12 * x + 16 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l372_37247


namespace NUMINAMATH_CALUDE_perfect_squares_solution_l372_37246

theorem perfect_squares_solution (x y : ℤ) :
  (∃ a : ℤ, x + y = a^2) →
  (∃ b : ℤ, 2*x + 3*y = b^2) →
  (∃ c : ℤ, 3*x + y = c^2) →
  x = 0 ∧ y = 0 := by
sorry

end NUMINAMATH_CALUDE_perfect_squares_solution_l372_37246


namespace NUMINAMATH_CALUDE_workshop_average_salary_l372_37225

/-- Given a workshop with workers, prove that the average salary of all workers is 8000 Rs. -/
theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (tech_salary : ℕ)
  (non_tech_salary : ℕ)
  (h1 : total_workers = 21)
  (h2 : technicians = 7)
  (h3 : tech_salary = 12000)
  (h4 : non_tech_salary = 6000) :
  (technicians * tech_salary + (total_workers - technicians) * non_tech_salary) / total_workers = 8000 := by
  sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l372_37225


namespace NUMINAMATH_CALUDE_colored_isosceles_triangle_exists_l372_37275

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A colored vertex in a polygon -/
def ColoredVertex (n : ℕ) (p : RegularPolygon n) := Fin n

/-- Three vertices form an isosceles triangle -/
def IsIsoscelesTriangle (n : ℕ) (p : RegularPolygon n) (v1 v2 v3 : Fin n) : Prop := sorry

theorem colored_isosceles_triangle_exists 
  (p : RegularPolygon 5000) 
  (colored : Finset (ColoredVertex 5000 p)) 
  (h : colored.card = 2001) : 
  ∃ (v1 v2 v3 : ColoredVertex 5000 p), 
    v1 ∈ colored ∧ v2 ∈ colored ∧ v3 ∈ colored ∧ 
    IsIsoscelesTriangle 5000 p v1 v2 v3 :=
  sorry

end NUMINAMATH_CALUDE_colored_isosceles_triangle_exists_l372_37275


namespace NUMINAMATH_CALUDE_remy_water_usage_l372_37243

theorem remy_water_usage (roman_usage : ℕ) 
  (h1 : roman_usage + (3 * roman_usage + 1) = 33) : 
  3 * roman_usage + 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_remy_water_usage_l372_37243


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l372_37256

/-- The complex number i(1+i) corresponds to a point in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant : 
  let z : ℂ := Complex.I * (1 + Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l372_37256


namespace NUMINAMATH_CALUDE_parallel_lines_solution_l372_37278

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (a b c d e f : ℝ) : Prop :=
  a * e = b * d

/-- The first line: ax + 2y + 6 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 6 = 0

/-- The second line: x + (a - 1)y + (a^2 - 1) = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  x + (a - 1) * y + (a^2 - 1) = 0

theorem parallel_lines_solution :
  ∀ a : ℝ, parallel_lines a 2 1 (a - 1) 1 1 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_solution_l372_37278


namespace NUMINAMATH_CALUDE_range_of_f_domain1_range_of_f_domain2_l372_37236

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 9

-- Define the domains
def domain1 : Set ℝ := {x | 3 < x ∧ x ≤ 8}
def domain2 : Set ℝ := {x | -3 < x ∧ x ≤ 2}

-- State the theorems
theorem range_of_f_domain1 :
  f '' domain1 = Set.Ioc 12 57 := by sorry

theorem range_of_f_domain2 :
  f '' domain2 = Set.Ico 8 24 := by sorry

end NUMINAMATH_CALUDE_range_of_f_domain1_range_of_f_domain2_l372_37236


namespace NUMINAMATH_CALUDE_lindsay_doll_difference_l372_37222

/-- The number of dolls Lindsay has with different hair colors -/
structure DollCounts where
  blonde : ℕ
  brown : ℕ
  black : ℕ

/-- Lindsay's doll collection satisfying the given conditions -/
def lindsay_dolls : DollCounts where
  blonde := 4
  brown := 4 * 4
  black := 4 * 4 - 2

/-- The difference between the number of dolls with black and brown hair combined
    and the number of dolls with blonde hair -/
def hair_color_difference (d : DollCounts) : ℕ :=
  d.brown + d.black - d.blonde

theorem lindsay_doll_difference :
  hair_color_difference lindsay_dolls = 26 := by
  sorry

end NUMINAMATH_CALUDE_lindsay_doll_difference_l372_37222


namespace NUMINAMATH_CALUDE_natural_number_solution_xy_l372_37242

theorem natural_number_solution_xy : 
  ∀ (x y : ℕ), x + y = x * y ↔ x = 2 ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_natural_number_solution_xy_l372_37242


namespace NUMINAMATH_CALUDE_five_digit_sum_l372_37252

def is_valid_digit (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 9

theorem five_digit_sum (x : ℕ) (h1 : is_valid_digit x) 
  (h2 : x ≠ 1 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 6) 
  (h3 : 120 * (1 + 3 + 4 + 6 + x) = 2640) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_sum_l372_37252


namespace NUMINAMATH_CALUDE_isosceles_triangle_altitude_l372_37289

theorem isosceles_triangle_altitude (a : ℝ) : 
  let r : ℝ := 7
  let circle_x_circumference : ℝ := 14 * Real.pi
  let circle_y_radius : ℝ := 2 * a
  (circle_x_circumference = 2 * Real.pi * r) →
  (circle_y_radius = r) →
  let h : ℝ := Real.sqrt 3 * a
  (h^2 + a^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_altitude_l372_37289


namespace NUMINAMATH_CALUDE_number_plus_273_l372_37283

theorem number_plus_273 (x : ℤ) : x - 477 = 273 → x + 273 = 1023 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_273_l372_37283


namespace NUMINAMATH_CALUDE_problem_statement_l372_37234

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ x * y ≤ a * b) ∧
  (a^2 + b^2 ≥ 1/2) ∧
  (4/a + 1/b ≥ 9) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ Real.sqrt x + Real.sqrt y < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l372_37234


namespace NUMINAMATH_CALUDE_slope_angle_range_l372_37212

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + (3 - Real.sqrt 3)*x + 3/4

def is_on_curve (p : ℝ × ℝ) : Prop := p.2 = f p.1

theorem slope_angle_range (p q : ℝ × ℝ) (hp : is_on_curve p) (hq : is_on_curve q) :
  let α := Real.arctan ((q.2 - p.2) / (q.1 - p.1))
  α ∈ Set.union (Set.Ico 0 (Real.pi / 2)) (Set.Icc (2 * Real.pi / 3) Real.pi) :=
sorry

end NUMINAMATH_CALUDE_slope_angle_range_l372_37212


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l372_37262

/-- The product of two repeating decimals 0.151515... and 0.353535... is equal to 175/3267 -/
theorem product_of_repeating_decimals : 
  (15 : ℚ) / 99 * (35 : ℚ) / 99 = (175 : ℚ) / 3267 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l372_37262


namespace NUMINAMATH_CALUDE_remainder_theorem_l372_37290

theorem remainder_theorem (n : ℕ) : (3^(2*n) + 8) % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l372_37290


namespace NUMINAMATH_CALUDE_max_quotient_value_l372_37253

theorem max_quotient_value (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 500) (hb : 400 ≤ b ∧ b ≤ 1000) :
  (∀ x y, 100 ≤ x ∧ x ≤ 500 → 400 ≤ y ∧ y ≤ 1000 → b / a ≥ y / x) ∧ b / a ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_quotient_value_l372_37253


namespace NUMINAMATH_CALUDE_transport_speed_problem_l372_37210

/-- Proves that given two transports traveling in opposite directions, with one traveling at 60 mph,
    if they are 348 miles apart after 2.71875 hours, then the speed of the second transport is 68 mph. -/
theorem transport_speed_problem (speed_a speed_b : ℝ) (time : ℝ) (distance : ℝ) : 
  speed_a = 60 →
  time = 2.71875 →
  distance = 348 →
  (speed_a + speed_b) * time = distance →
  speed_b = 68 := by
  sorry

#check transport_speed_problem

end NUMINAMATH_CALUDE_transport_speed_problem_l372_37210


namespace NUMINAMATH_CALUDE_decagon_triangles_l372_37299

/-- The number of vertices in a regular decagon -/
def decagon_vertices : ℕ := 10

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def triangles_from_decagon : ℕ := Nat.choose decagon_vertices triangle_vertices

theorem decagon_triangles :
  triangles_from_decagon = 120 := by sorry

end NUMINAMATH_CALUDE_decagon_triangles_l372_37299


namespace NUMINAMATH_CALUDE_journey_duration_l372_37279

/-- Represents the journey of a spaceship --/
structure SpaceshipJourney where
  initial_travel : ℕ
  initial_break : ℕ
  second_travel : ℕ
  second_break : ℕ
  subsequent_travel : ℕ
  subsequent_break : ℕ
  total_non_moving : ℕ

/-- Calculates the total journey time for a spaceship --/
def total_journey_time (j : SpaceshipJourney) : ℕ :=
  let remaining_break := j.total_non_moving - j.initial_break - j.second_break
  let subsequent_segments := remaining_break / j.subsequent_break
  j.initial_travel + j.initial_break + j.second_travel + j.second_break +
  subsequent_segments * (j.subsequent_travel + j.subsequent_break)

/-- Theorem stating that the journey takes 72 hours --/
theorem journey_duration (j : SpaceshipJourney)
  (h1 : j.initial_travel = 10)
  (h2 : j.initial_break = 3)
  (h3 : j.second_travel = 10)
  (h4 : j.second_break = 1)
  (h5 : j.subsequent_travel = 11)
  (h6 : j.subsequent_break = 1)
  (h7 : j.total_non_moving = 8) :
  total_journey_time j = 72 := by
  sorry


end NUMINAMATH_CALUDE_journey_duration_l372_37279
