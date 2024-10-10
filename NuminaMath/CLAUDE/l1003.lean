import Mathlib

namespace smallest_integer_with_16_divisors_l1003_100360

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a given positive integer has exactly 16 positive divisors -/
def has_16_divisors (n : ℕ+) : Prop := num_divisors n = 16

theorem smallest_integer_with_16_divisors :
  (∃ (n : ℕ+), has_16_divisors n) ∧
  (∀ (m : ℕ+), has_16_divisors m → 384 ≤ m) ∧
  has_16_divisors 384 := by sorry

end smallest_integer_with_16_divisors_l1003_100360


namespace quadratic_coefficient_sum_l1003_100375

/-- A quadratic function with roots at -3 and 5, and a minimum value of 36 -/
def quadratic (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_coefficient_sum (a b c : ℝ) :
  (∀ x, quadratic a b c x ≥ 36) ∧ 
  quadratic a b c (-3) = 0 ∧ 
  quadratic a b c 5 = 0 →
  a + b + c = 36 := by
sorry

end quadratic_coefficient_sum_l1003_100375


namespace money_split_proof_l1003_100369

/-- 
Given two people splitting money in a 2:3 ratio where the smaller share is $50,
prove that the total amount shared is $125.
-/
theorem money_split_proof (smaller_share : ℕ) (total : ℕ) : 
  smaller_share = 50 → 
  2 * total = 5 * smaller_share →
  total = 125 := by
sorry

end money_split_proof_l1003_100369


namespace hemisphere_cylinder_surface_area_l1003_100393

/-- The total surface area of a shape consisting of a hemisphere attached to a cylindrical segment -/
theorem hemisphere_cylinder_surface_area (r : ℝ) (h : r = 10) :
  let hemisphere_area := 2 * π * r^2
  let cylinder_base_area := π * r^2
  let cylinder_lateral_area := 2 * π * r * (r / 2)
  hemisphere_area + cylinder_base_area + cylinder_lateral_area = 40 * π * r^2 :=
by sorry

end hemisphere_cylinder_surface_area_l1003_100393


namespace smallest_root_of_equation_l1003_100373

theorem smallest_root_of_equation (x : ℚ) : 
  (x - 5/6)^2 + (x - 5/6)*(x - 2/3) = 0 ∧ x^2 - 2*x + 1 ≥ 0 → 
  x ≥ 5/6 ∧ (∀ y : ℚ, y < 5/6 → (y - 5/6)^2 + (y - 5/6)*(y - 2/3) ≠ 0 ∨ y^2 - 2*y + 1 < 0) :=
by sorry

end smallest_root_of_equation_l1003_100373


namespace minimum_parents_needed_minimum_parents_for_tour_l1003_100320

theorem minimum_parents_needed (num_children : ℕ) (car_capacity : ℕ) : ℕ :=
  let total_people := num_children
  let drivers_needed := (total_people + car_capacity - 1) / car_capacity
  drivers_needed

theorem minimum_parents_for_tour :
  minimum_parents_needed 50 6 = 10 :=
by sorry

end minimum_parents_needed_minimum_parents_for_tour_l1003_100320


namespace find_k_value_l1003_100364

theorem find_k_value (x y z k : ℝ) 
  (eq1 : 9 / (x + y) = k / (x + 2*z))
  (eq2 : k / (x + 2*z) = 14 / (z - y))
  (cond1 : y = 2*x)
  (cond2 : x + z = 10) :
  k = 46 := by
sorry

end find_k_value_l1003_100364


namespace linear_system_solution_l1003_100396

theorem linear_system_solution (x y : ℝ) 
  (eq1 : x + 3*y = 20) 
  (eq2 : x + y = 10) : 
  x = 5 ∧ y = 5 := by
sorry

end linear_system_solution_l1003_100396


namespace weight_gain_theorem_l1003_100347

def weight_gain_problem (initial_weight first_month_gain second_month_gain : ℕ) : Prop :=
  initial_weight + first_month_gain + second_month_gain = 120

theorem weight_gain_theorem : 
  weight_gain_problem 70 20 30 := by sorry

end weight_gain_theorem_l1003_100347


namespace score_difference_is_1_25_l1003_100392

-- Define the score distribution
def score_distribution : List (ℝ × ℝ) := [
  (0.15, 60),
  (0.20, 75),
  (0.25, 85),
  (0.30, 95),
  (0.10, 100)
]

-- Calculate the mean score
def mean_score : ℝ := 
  (score_distribution.map (λ p => p.1 * p.2)).sum

-- Define the median score
def median_score : ℝ := 85

-- Theorem statement
theorem score_difference_is_1_25 : 
  median_score - mean_score = 1.25 := by sorry

end score_difference_is_1_25_l1003_100392


namespace whitney_book_cost_l1003_100302

/-- Proves that given the conditions of Whitney's purchase, each book costs $11. -/
theorem whitney_book_cost (num_books num_magazines : ℕ) (magazine_cost total_cost book_cost : ℚ) : 
  num_books = 16 →
  num_magazines = 3 →
  magazine_cost = 1 →
  total_cost = 179 →
  total_cost = num_books * book_cost + num_magazines * magazine_cost →
  book_cost = 11 := by
sorry

end whitney_book_cost_l1003_100302


namespace prob_two_hearts_is_one_seventeenth_l1003_100316

/-- A standard deck of cards. -/
structure Deck :=
  (cards : Finset Nat)
  (size : cards.card = 52)
  (suits : Finset Nat)
  (suit_size : suits.card = 4)

/-- The number of hearts in a standard deck. -/
def hearts_count : Nat := 13

/-- The probability of drawing two hearts from a well-shuffled standard deck. -/
def prob_two_hearts (d : Deck) : ℚ :=
  (hearts_count * (hearts_count - 1)) / (d.cards.card * (d.cards.card - 1))

/-- Theorem: The probability of drawing two hearts from a well-shuffled standard deck is 1/17. -/
theorem prob_two_hearts_is_one_seventeenth (d : Deck) :
  prob_two_hearts d = 1 / 17 := by
  sorry

end prob_two_hearts_is_one_seventeenth_l1003_100316


namespace angle_sum_at_point_l1003_100313

theorem angle_sum_at_point (y : ℝ) : 
  (170 : ℝ) + y + y = 360 → y = 95 := by
  sorry

end angle_sum_at_point_l1003_100313


namespace girls_in_class_l1003_100340

theorem girls_in_class (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 56 → 
  4 * boys = 3 * girls → 
  total = girls + boys → 
  girls = 32 := by
sorry

end girls_in_class_l1003_100340


namespace ordered_pairs_satisfying_conditions_l1003_100333

theorem ordered_pairs_satisfying_conditions :
  ∀ a b : ℕ+,
  (a.val^2 + b.val^2 + 25 = 15 * a.val * b.val) ∧
  (Nat.Prime (a.val^2 + a.val * b.val + b.val^2)) →
  ((a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1)) :=
by sorry

end ordered_pairs_satisfying_conditions_l1003_100333


namespace line_and_circle_proof_l1003_100384

-- Define the lines and circles
def l₁ (x y : ℝ) : Prop := x - 2*y + 2 = 0
def l₂ (x y : ℝ) : Prop := 2*x - y - 2 = 0
def c₁ (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def c₂ (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line that we want to prove
def target_line (x y : ℝ) : Prop := y = x

-- Define the circle that we want to prove
def target_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 11 = 0

-- Define the line on which the center of the target circle should lie
def center_line (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

theorem line_and_circle_proof :
  -- Part 1: The target line passes through the origin and the intersection of l₁ and l₂
  (∃ x y : ℝ, l₁ x y ∧ l₂ x y ∧ target_line x y) ∧
  target_line 0 0 ∧
  -- Part 2: The target circle has its center on the center_line and passes through
  -- the intersection points of c₁ and c₂
  (∃ x y : ℝ, center_line x y ∧ 
    ∀ a b : ℝ, (c₁ a b ∧ c₂ a b) → target_circle a b) :=
sorry

end line_and_circle_proof_l1003_100384


namespace radius_of_larger_circle_l1003_100346

/-- Given two identical circles touching each other from the inside of a third circle,
    prove that the radius of the larger circle is 9 when the perimeter of the triangle
    formed by connecting the three centers is 18. -/
theorem radius_of_larger_circle (r R : ℝ) : r > 0 → R > r →
  (R - r) + (R - r) + 2 * r = 18 → R = 9 := by
  sorry

end radius_of_larger_circle_l1003_100346


namespace sum_remainder_mod_seven_l1003_100342

theorem sum_remainder_mod_seven :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 7 = 2 := by
  sorry

end sum_remainder_mod_seven_l1003_100342


namespace base3_sum_theorem_l1003_100315

/-- Converts a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 3 * acc + d) 0

/-- Converts a decimal number to its base 3 representation as a list of digits -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- The main theorem stating the sum of the given base 3 numbers -/
theorem base3_sum_theorem :
  let a := base3ToDecimal [2, 1, 2, 1]
  let b := base3ToDecimal [1, 2, 1, 2]
  let c := base3ToDecimal [2, 1, 2]
  let d := base3ToDecimal [2]
  decimalToBase3 (a + b + c + d) = [2, 2, 0, 1] := by sorry

end base3_sum_theorem_l1003_100315


namespace research_team_probabilities_l1003_100389

/-- Represents a research team member -/
structure Member where
  gender : Bool  -- true for male, false for female
  speaksEnglish : Bool

/-- Represents a research team -/
def ResearchTeam : Type := List Member

/-- Creates a research team with the given specifications -/
def createTeam : ResearchTeam :=
  [
    { gender := true, speaksEnglish := false },   -- non-English speaking male
    { gender := true, speaksEnglish := true },    -- English speaking male
    { gender := true, speaksEnglish := true },    -- English speaking male
    { gender := true, speaksEnglish := true },    -- English speaking male
    { gender := false, speaksEnglish := false },  -- non-English speaking female
    { gender := false, speaksEnglish := true }    -- English speaking female
  ]

/-- Calculates the probability of selecting two members with a given property -/
def probabilityOfSelection (team : ResearchTeam) (property : Member → Member → Bool) : Rat :=
  sorry

theorem research_team_probabilities (team : ResearchTeam) 
  (h1 : team = createTeam) : 
  (probabilityOfSelection team (fun m1 m2 => m1.gender = m2.gender) = 7/15) ∧ 
  (probabilityOfSelection team (fun m1 m2 => m1.speaksEnglish ∨ m2.speaksEnglish) = 14/15) ∧
  (probabilityOfSelection team (fun m1 m2 => m1.gender ≠ m2.gender ∧ (m1.speaksEnglish ∨ m2.speaksEnglish)) = 7/15) :=
by sorry

end research_team_probabilities_l1003_100389


namespace inequality_proof_l1003_100339

theorem inequality_proof (a : ℝ) : 
  (a^2 + 5)^2 + 4*a*(10 - a) - 8*a^3 ≥ 0 ∧ 
  ((a^2 + 5)^2 + 4*a*(10 - a) - 8*a^3 = 0 ↔ a = 5 ∨ a = -1) :=
by sorry

end inequality_proof_l1003_100339


namespace infinite_product_a_l1003_100381

noncomputable def a : ℕ → ℚ
  | 0 => 2/3
  | n+1 => 1 + (a n - 1)^2

theorem infinite_product_a : ∏' n, a n = 1/2 := by sorry

end infinite_product_a_l1003_100381


namespace fair_coin_five_tosses_l1003_100363

/-- A fair coin is a coin with equal probability of landing on either side. -/
def fair_coin (p : ℝ) : Prop := p = 1 / 2

/-- The probability of a specific sequence of n tosses for a fair coin. -/
def prob_sequence (n : ℕ) (p : ℝ) : ℝ := p ^ n

/-- The probability of landing on the same side for n tosses of a fair coin. -/
def prob_same_side (n : ℕ) (p : ℝ) : ℝ := 2 * (prob_sequence n p)

theorem fair_coin_five_tosses (p : ℝ) (h : fair_coin p) :
  prob_same_side 5 p = 1 / 16 := by
  sorry

end fair_coin_five_tosses_l1003_100363


namespace no_real_roots_quadratic_l1003_100325

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 := by
  sorry

end no_real_roots_quadratic_l1003_100325


namespace circle_center_and_radius_l1003_100350

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of the circle (x-1)^2 + y^2 = 3 -/
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 3

theorem circle_center_and_radius :
  ∃ (c : Circle), (∀ x y, circle_equation x y ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
                   c.center = (1, 0) ∧
                   c.radius = Real.sqrt 3 := by
  sorry

end circle_center_and_radius_l1003_100350


namespace darkest_cell_value_l1003_100326

/-- Represents the grid structure -/
structure Grid :=
  (white1 white2 white3 white4 : Nat)
  (gray1 gray2 : Nat)
  (dark : Nat)

/-- The grid satisfies the problem conditions -/
def valid_grid (g : Grid) : Prop :=
  g.white1 > 1 ∧ g.white2 > 1 ∧ g.white3 > 1 ∧ g.white4 > 1 ∧
  g.white1 * g.white2 = 55 ∧
  g.white3 * g.white4 = 55 ∧
  g.gray1 = g.white1 * g.white3 ∧
  g.gray2 = g.white2 * g.white4 ∧
  g.dark = g.gray1 * g.gray2

theorem darkest_cell_value (g : Grid) :
  valid_grid g → g.dark = 245025 := by
  sorry

#check darkest_cell_value

end darkest_cell_value_l1003_100326


namespace divisibility_property_l1003_100328

theorem divisibility_property (n : ℕ) (hn : n > 0) :
  ∃ (a b : ℤ), (n : ℤ) ∣ (4 * a^2 + 9 * b^2 - 1) := by
  sorry

end divisibility_property_l1003_100328


namespace expand_binomials_l1003_100327

theorem expand_binomials (a : ℝ) : (a + 3) * (-a + 1) = -a^2 - 2*a + 3 := by
  sorry

end expand_binomials_l1003_100327


namespace floor_sqrt_50_l1003_100388

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
  sorry

end floor_sqrt_50_l1003_100388


namespace gcd_4557_5115_l1003_100359

theorem gcd_4557_5115 : Nat.gcd 4557 5115 = 93 := by
  sorry

end gcd_4557_5115_l1003_100359


namespace digit_120_is_1_l1003_100319

/-- Represents the decimal number formed by concatenating integers 1 to 51 -/
def x : ℚ :=
  0.123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051

/-- Returns the nth digit after the decimal point in a rational number -/
def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem digit_120_is_1 : nthDigitAfterDecimal x 120 = 1 := by
  sorry

end digit_120_is_1_l1003_100319


namespace students_playing_both_sports_l1003_100352

/-- Given a school with 460 students, where 325 play football, 175 play cricket, 
    and 50 play neither, prove that 90 students play both sports. -/
theorem students_playing_both_sports (total : ℕ) (football : ℕ) (cricket : ℕ) (neither : ℕ) 
  (h1 : total = 460)
  (h2 : football = 325)
  (h3 : cricket = 175)
  (h4 : neither = 50)
  : total = football + cricket - 90 + neither := by
  sorry

end students_playing_both_sports_l1003_100352


namespace product_expansion_l1003_100332

theorem product_expansion (x : ℝ) : 5*(x-6)*(x+9) + 3*x = 5*x^2 + 18*x - 270 := by
  sorry

end product_expansion_l1003_100332


namespace quadratic_roots_theorem_l1003_100371

/-- A quadratic function f(x) with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + a

/-- The function g(x) represents f(x) - x -/
def g (a : ℝ) (x : ℝ) : ℝ := f a x - x

theorem quadratic_roots_theorem (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : g a x₁ = 0) (h₂ : g a x₂ = 0) (h₃ : 0 < x₁) (h₄ : x₁ < x₂) (h₅ : x₂ < 1) :
  (0 < a ∧ a < 3 - Real.sqrt 2) ∧ f a 0 * f a 1 - f a 0 < 1/4 := by
  sorry

end quadratic_roots_theorem_l1003_100371


namespace expression_equality_implies_square_l1003_100334

theorem expression_equality_implies_square (x y : ℕ) 
  (h : (1 : ℚ) / x + 1 / y + 1 / (x * y) = 1 / (x + 4) + 1 / (y - 4) + 1 / ((x + 4) * (y - 4))) :
  ∃ n : ℕ, x * y + 4 = n^2 := by
  sorry

end expression_equality_implies_square_l1003_100334


namespace average_questions_correct_l1003_100322

def dongwoos_group : List Nat := [16, 22, 30, 26, 18, 20]

theorem average_questions_correct : 
  (List.sum dongwoos_group) / (List.length dongwoos_group) = 22 := by
  sorry

end average_questions_correct_l1003_100322


namespace simplify_expression_l1003_100397

theorem simplify_expression (x y : ℝ) (h : y ≠ 0) :
  ((x + y)^2 - (x + y)*(x - y)) / (2*y) = y + x := by
  sorry

end simplify_expression_l1003_100397


namespace range_of_a_minus_b_l1003_100300

theorem range_of_a_minus_b (a b : ℝ) (θ : ℝ) 
  (h1 : |a - Real.sin θ ^ 2| ≤ 1) 
  (h2 : |b + Real.cos θ ^ 2| ≤ 1) : 
  -1 ≤ a - b ∧ a - b ≤ 3 := by
  sorry

end range_of_a_minus_b_l1003_100300


namespace trapezium_height_l1003_100394

theorem trapezium_height (a b h : ℝ) : 
  a > 0 → b > 0 → h > 0 →
  a = 20 → b = 18 → 
  (1/2) * (a + b) * h = 190 →
  h = 10 := by
sorry

end trapezium_height_l1003_100394


namespace lenora_points_scored_l1003_100385

-- Define the types of shots
inductive ShotType
| ThreePoint
| FreeThrow

-- Define the game parameters
def total_shots : ℕ := 40
def three_point_success_rate : ℚ := 1/4
def free_throw_success_rate : ℚ := 1/2

-- Define the point values for each shot type
def point_value (shot : ShotType) : ℕ :=
  match shot with
  | ShotType.ThreePoint => 3
  | ShotType.FreeThrow => 1

-- Define the function to calculate points scored
def points_scored (three_point_attempts : ℕ) (free_throw_attempts : ℕ) : ℚ :=
  (three_point_attempts : ℚ) * three_point_success_rate * (point_value ShotType.ThreePoint) +
  (free_throw_attempts : ℚ) * free_throw_success_rate * (point_value ShotType.FreeThrow)

-- Theorem statement
theorem lenora_points_scored :
  ∀ (three_point_attempts free_throw_attempts : ℕ),
    three_point_attempts + free_throw_attempts = total_shots →
    points_scored three_point_attempts free_throw_attempts = 30 :=
by sorry

end lenora_points_scored_l1003_100385


namespace blue_section_probability_l1003_100379

def bernoulli_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

theorem blue_section_probability : 
  bernoulli_probability 7 7 (2/7) = 128/823543 := by
  sorry

end blue_section_probability_l1003_100379


namespace solution_set_characterization_l1003_100376

/-- A function f: ℝ → ℝ is even -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The set of real numbers x where xf(x) > 0 -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ := {x | x * f x > 0}

/-- The open interval (-∞, -1) ∪ (1, +∞) -/
def TargetSet : Set ℝ := {x | x < -1 ∨ x > 1}

theorem solution_set_characterization (f : ℝ → ℝ) 
  (h_even : IsEven f)
  (h_positive : ∀ x > 0, f x + x * (deriv f x) > 0)
  (h_zero : f 1 = 0) :
  SolutionSet f = TargetSet := by sorry

end solution_set_characterization_l1003_100376


namespace sum_from_difference_and_squares_l1003_100303

theorem sum_from_difference_and_squares (m n : ℤ) 
  (h1 : m^2 - n^2 = 18) 
  (h2 : m - n = 9) : 
  m + n = 2 := by
sorry

end sum_from_difference_and_squares_l1003_100303


namespace pony_speed_l1003_100367

/-- The average speed of a pony given specific conditions of a chase scenario. -/
theorem pony_speed (horse_speed : ℝ) (head_start : ℝ) (chase_time : ℝ) : 
  horse_speed = 35 → head_start = 3 → chase_time = 4 → 
  ∃ (pony_speed : ℝ), pony_speed = 20 ∧ 
  horse_speed * chase_time = pony_speed * (head_start + chase_time) := by
sorry

end pony_speed_l1003_100367


namespace church_distance_l1003_100390

theorem church_distance (horse_speed : ℝ) (hourly_rate : ℝ) (flat_fee : ℝ) (total_paid : ℝ) 
  (h1 : horse_speed = 10)
  (h2 : hourly_rate = 30)
  (h3 : flat_fee = 20)
  (h4 : total_paid = 80) : 
  (total_paid - flat_fee) / hourly_rate * horse_speed = 20 := by
  sorry

#check church_distance

end church_distance_l1003_100390


namespace shower_water_usage_l1003_100301

theorem shower_water_usage (roman remy riley ronan : ℝ) : 
  remy = 3 * roman + 1 →
  riley = roman + remy - 2 →
  ronan = riley / 2 →
  roman + remy + riley + ronan = 60 →
  remy = 18.85 := by
sorry

end shower_water_usage_l1003_100301


namespace equation_d_is_linear_l1003_100399

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants, and at least one of a or b is non-zero. --/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y ↔ a * x + b * y = c

/-- The equation x = y + 1 --/
def EquationD (x y : ℝ) : Prop := x = y + 1

theorem equation_d_is_linear : IsLinearEquationInTwoVariables EquationD := by
  sorry

#check equation_d_is_linear

end equation_d_is_linear_l1003_100399


namespace original_number_l1003_100343

theorem original_number (x : ℝ) : 1 + 1 / x = 9 / 4 → x = 4 / 5 := by
  sorry

end original_number_l1003_100343


namespace sum_of_ages_l1003_100380

/-- Given the ages of a father and son, prove that their sum is 55 years. -/
theorem sum_of_ages (father_age son_age : ℕ) 
  (h1 : father_age = 37) 
  (h2 : son_age = 18) : 
  father_age + son_age = 55 := by
  sorry

end sum_of_ages_l1003_100380


namespace thirty_five_power_ab_equals_R_power_b_times_S_power_a_l1003_100345

theorem thirty_five_power_ab_equals_R_power_b_times_S_power_a
  (a b : ℤ) (R S : ℝ) (hR : R = 5^a) (hS : S = 7^b) :
  35^(a*b) = R^b * S^a := by
  sorry

end thirty_five_power_ab_equals_R_power_b_times_S_power_a_l1003_100345


namespace solve_quadratic_equation_l1003_100324

theorem solve_quadratic_equation (x : ℝ) : 2 * (x - 1)^2 = 8 → x = 3 ∨ x = -1 := by
  sorry

end solve_quadratic_equation_l1003_100324


namespace range_of_m_l1003_100349

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

def q (x m : ℝ) : Prop := (x - 1 - m) * (x - 1 + m) ≤ 0

-- Define the sufficient condition relationship
def sufficient_condition (m : ℝ) : Prop :=
  ∀ x, q x m → p x

-- Define the not necessary condition relationship
def not_necessary_condition (m : ℝ) : Prop :=
  ∃ x, p x ∧ ¬(q x m)

-- Main theorem
theorem range_of_m :
  ∀ m : ℝ, m > 0 ∧ sufficient_condition m ∧ not_necessary_condition m
  → m ≤ 3 :=
sorry

end range_of_m_l1003_100349


namespace at_least_one_equation_has_two_distinct_roots_l1003_100329

theorem at_least_one_equation_has_two_distinct_roots
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (4 * b^2 - 4 * a * c > 0) ∨ (4 * c^2 - 4 * a * b > 0) ∨ (4 * a^2 - 4 * b * c > 0) :=
sorry

end at_least_one_equation_has_two_distinct_roots_l1003_100329


namespace factorial_sum_equality_l1003_100310

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + 2 * Nat.factorial 6 = 41040 := by
  sorry

end factorial_sum_equality_l1003_100310


namespace f_3_range_l1003_100353

/-- Given a quadratic function f(x) = ax^2 - c with specific constraints on f(1) and f(2),
    we prove that f(3) lies within a certain range. -/
theorem f_3_range (a c : ℝ) (h1 : -4 ≤ a - c ∧ a - c ≤ -1) (h2 : -1 ≤ 4*a - c ∧ 4*a - c ≤ 5) :
  -1 ≤ 9*a - c ∧ 9*a - c ≤ 20 := by
  sorry

#check f_3_range

end f_3_range_l1003_100353


namespace sequence_uniqueness_l1003_100330

theorem sequence_uniqueness (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, n ≥ 1 → (a (n + 1))^2 = 1 + (n + 2021) * a n) :
  ∀ n : ℕ, n ≥ 1 → a n = n + 2019 := by
  sorry

end sequence_uniqueness_l1003_100330


namespace photograph_perimeter_l1003_100354

/-- 
Given a rectangular photograph with a border, this theorem proves that 
if the total area with a 1-inch border is m square inches, and 
the total area with a 3-inch border is (m + 52) square inches, 
then the perimeter of the photograph is 10 inches.
-/
theorem photograph_perimeter 
  (w l m : ℝ) 
  (h1 : (w + 2) * (l + 2) = m) 
  (h2 : (w + 6) * (l + 6) = m + 52) : 
  2 * (w + l) = 10 :=
sorry

end photograph_perimeter_l1003_100354


namespace parabola_intersection_theorem_l1003_100358

/-- Parabola with equation y^2 = 2x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line passing through a point -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- Intersection points of a line and a parabola -/
def Intersection (p : Parabola) (l : Line) : Set (ℝ × ℝ) :=
  {pt | p.equation pt.1 pt.2 ∧ pt.2 = l.slope * (pt.1 - l.point.1) + l.point.2}

theorem parabola_intersection_theorem (p : Parabola) (l : Line) 
    (A B : ℝ × ℝ) (hA : A ∈ Intersection p l) (hB : B ∈ Intersection p l) :
  p.equation 0.5 0 →  -- Focus is on the parabola
  l.point = p.focus →  -- Line passes through the focus
  ‖A - B‖ = 25/12 →  -- Distance between A and B
  ‖A - p.focus‖ < ‖B - p.focus‖ →  -- AF < BF
  ‖A - p.focus‖ = 5/6 :=  -- |AF| = 5/6
by sorry

end parabola_intersection_theorem_l1003_100358


namespace baron_munchausen_claim_l1003_100311

theorem baron_munchausen_claim (weights : Finset ℕ) : 
  weights.card = 8 ∧ weights = Finset.range 8 →
  ∃ (A B C : Finset ℕ), 
    A ∪ B ∪ C = weights ∧
    A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
    A.card = 2 ∧ B.card = 5 ∧ C.card = 1 ∧
    (A.sum id = B.sum id) ∧
    (∀ w ∈ C, w = A.sum id - B.sum id) :=
by sorry

end baron_munchausen_claim_l1003_100311


namespace vector_sum_length_l1003_100341

/-- Given vectors a and b in ℝ², prove that |a + 2b| = √61 under specific conditions. -/
theorem vector_sum_length (a b : ℝ × ℝ) : 
  a = (3, -4) → 
  ‖b‖ = 2 → 
  (a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖) = 1/2 →
  ‖a + 2 • b‖ = Real.sqrt 61 := by
  sorry

end vector_sum_length_l1003_100341


namespace power_sum_equals_eight_l1003_100374

theorem power_sum_equals_eight :
  (-2)^3 + (-2)^2 + (-2)^1 + 2^1 + 2^2 + 2^3 = 8 := by
  sorry

end power_sum_equals_eight_l1003_100374


namespace harmonic_mean_of_three_fourths_and_five_sixths_l1003_100304

theorem harmonic_mean_of_three_fourths_and_five_sixths :
  let a : ℚ := 3/4
  let b : ℚ := 5/6
  let harmonic_mean (x y : ℚ) := 2 * x * y / (x + y)
  harmonic_mean a b = 15/19 := by
  sorry

end harmonic_mean_of_three_fourths_and_five_sixths_l1003_100304


namespace problem_statement_l1003_100391

/-- The equation x^2 - x + a^2 - 6a = 0 has one positive root and one negative root. -/
def p (a : ℝ) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - x + a^2 - 6*a = 0 ∧ y^2 - y + a^2 - 6*a = 0

/-- The graph of y = x^2 + (a-3)x + 1 has no common points with the x-axis. -/
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + (a-3)*x + 1 ≠ 0

/-- The range of values for a is 0 < a ≤ 1 or 5 ≤ a < 6. -/
def range_of_a (a : ℝ) : Prop :=
  (0 < a ∧ a ≤ 1) ∨ (5 ≤ a ∧ a < 6)

theorem problem_statement (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_of_a a := by
  sorry

end problem_statement_l1003_100391


namespace three_halves_equals_one_point_five_l1003_100362

theorem three_halves_equals_one_point_five : (3 : ℚ) / 2 = 1.5 := by
  sorry

end three_halves_equals_one_point_five_l1003_100362


namespace largest_non_odd_units_digit_proof_l1003_100387

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

def units_digit (n : ℕ) : ℕ := n % 10

def largest_non_odd_units_digit : ℕ := 8

theorem largest_non_odd_units_digit_proof :
  ∀ d : ℕ, d ≤ 9 →
    (d > largest_non_odd_units_digit →
      ∃ n : ℕ, is_odd n ∧ units_digit n = d) ∧
    (d ≤ largest_non_odd_units_digit →
      d = largest_non_odd_units_digit ∨
      ∀ n : ℕ, is_odd n → units_digit n ≠ d) :=
sorry

end largest_non_odd_units_digit_proof_l1003_100387


namespace negation_of_existence_negation_of_quadratic_inequality_l1003_100366

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 > 0) :=
by sorry

end negation_of_existence_negation_of_quadratic_inequality_l1003_100366


namespace arithmetic_sequence_12th_term_l1003_100351

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : ∀ n, a (n + 1) - a n = 2)
  (h3 : a 3 = 4) :
  a 12 = 22 := by
sorry

end arithmetic_sequence_12th_term_l1003_100351


namespace bananas_left_l1003_100335

/-- The number of bananas in a dozen -/
def dozen : ℕ := 12

/-- The number of bananas Anthony ate -/
def eaten : ℕ := 2

/-- Theorem: The number of bananas left is 10 -/
theorem bananas_left : dozen - eaten = 10 := by
  sorry

end bananas_left_l1003_100335


namespace triangle_max_area_l1003_100309

/-- The maximum area of a triangle with one side of length 3 and the sum of the other two sides equal to 5 is 3. -/
theorem triangle_max_area :
  ∀ (b c : ℝ),
  b > 0 → c > 0 →
  b + c = 5 →
  let a := 3
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area ≤ 3 ∧ ∃ (b₀ c₀ : ℝ), b₀ > 0 ∧ c₀ > 0 ∧ b₀ + c₀ = 5 ∧
    let s₀ := (a + b₀ + c₀) / 2
    Real.sqrt (s₀ * (s₀ - a) * (s₀ - b₀) * (s₀ - c₀)) = 3 :=
by
  sorry

end triangle_max_area_l1003_100309


namespace sum_ge_sum_of_sqrt_products_l1003_100361

theorem sum_ge_sum_of_sqrt_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) := by
  sorry

end sum_ge_sum_of_sqrt_products_l1003_100361


namespace sample_size_is_300_l1003_100370

/-- Represents the population ratios of the districts -/
def district_ratios : List ℕ := [2, 3, 5, 2, 6]

/-- The number of individuals contributed by the largest district -/
def largest_district_contribution : ℕ := 100

/-- Calculates the total sample size based on the district ratios and the contribution of the largest district -/
def calculate_sample_size (ratios : List ℕ) (largest_contribution : ℕ) : ℕ :=
  let total_ratio := ratios.sum
  let largest_ratio := ratios.maximum?
  match largest_ratio with
  | some max_ratio => (total_ratio * largest_contribution) / max_ratio
  | none => 0

/-- Theorem stating that the calculated sample size is 300 -/
theorem sample_size_is_300 :
  calculate_sample_size district_ratios largest_district_contribution = 300 := by
  sorry

#eval calculate_sample_size district_ratios largest_district_contribution

end sample_size_is_300_l1003_100370


namespace point_b_coordinates_l1003_100356

/-- Given a circle and two points A and B, if the squared distance from any point on the circle to A
    is twice the squared distance to B, then B has coordinates (1, 1). -/
theorem point_b_coordinates (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4 → (x - 2)^2 + (y - 2)^2 = 2*((x - a)^2 + (y - b)^2)) →
  a = 1 ∧ b = 1 := by
sorry

end point_b_coordinates_l1003_100356


namespace lottery_possibility_l1003_100383

theorem lottery_possibility (win_chance : ℝ) (h : win_chance = 0.01) : 
  ∃ (outcome : Bool), outcome = true :=
sorry

end lottery_possibility_l1003_100383


namespace sin_cos_roots_l1003_100398

theorem sin_cos_roots (θ : Real) (a : Real) 
  (h1 : x^2 - 2 * Real.sqrt 2 * a * x + a = 0 ↔ x = Real.sin θ ∨ x = Real.cos θ)
  (h2 : -π/2 < θ ∧ θ < 0) : 
  a = -1/4 ∧ Real.sin θ - Real.cos θ = -Real.sqrt 6 / 2 := by
  sorry

end sin_cos_roots_l1003_100398


namespace promotion_savings_difference_l1003_100395

/-- Represents the cost of a pair of shoes in dollars -/
def shoe_cost : ℝ := 50

/-- Calculates the cost of two pairs of shoes using Promotion X -/
def cost_promotion_x : ℝ :=
  shoe_cost + (shoe_cost * (1 - 0.4))

/-- Calculates the cost of two pairs of shoes using Promotion Y -/
def cost_promotion_y : ℝ :=
  shoe_cost + (shoe_cost - 15)

/-- Theorem: The difference in cost between Promotion Y and Promotion X is $5 -/
theorem promotion_savings_difference :
  cost_promotion_y - cost_promotion_x = 5 := by
  sorry

end promotion_savings_difference_l1003_100395


namespace triangle_properties_l1003_100377

noncomputable section

/-- Triangle ABC with given properties -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle given two sides and the included angle -/
def area (t : Triangle) : ℝ := (1/2) * t.a * t.b * Real.sin t.C

theorem triangle_properties (t : Triangle) 
  (h1 : t.c = 2)
  (h2 : t.C = π/3) :
  (area t = Real.sqrt 3 → t.a = 2 ∧ t.b = 2) ∧
  (Real.sin t.B = 2 * Real.sin t.A → area t = 4 * Real.sqrt 3 / 3) := by
  sorry

end

end triangle_properties_l1003_100377


namespace alternate_arrangements_count_l1003_100368

/-- The number of ways to arrange two men and two women alternately in a row -/
def alternateArrangements : ℕ :=
  let menCount := 2
  let womenCount := 2
  let manFirstArrangements := menCount * womenCount
  let womanFirstArrangements := womenCount * menCount
  manFirstArrangements + womanFirstArrangements

theorem alternate_arrangements_count :
  alternateArrangements = 8 := by
  sorry

end alternate_arrangements_count_l1003_100368


namespace largest_number_bound_l1003_100314

theorem largest_number_bound (a b c : ℝ) (sum_zero : a + b + c = 0) (product_eight : a * b * c = 8) :
  max a (max b c) ≥ 2 * Real.rpow 4 (1/3) := by
  sorry

end largest_number_bound_l1003_100314


namespace lcm_of_135_and_468_l1003_100323

theorem lcm_of_135_and_468 : Nat.lcm 135 468 = 7020 := by
  sorry

end lcm_of_135_and_468_l1003_100323


namespace number_of_distributions_l1003_100382

/-- The number of ways to distribute 5 students into 3 groups with constraints -/
def distribution_schemes : ℕ :=
  -- The actual calculation would go here, but we don't have the solution steps
  80

/-- Theorem stating the number of distribution schemes -/
theorem number_of_distributions :
  distribution_schemes = 80 :=
by
  -- The proof would go here
  sorry

#check number_of_distributions

end number_of_distributions_l1003_100382


namespace consecutive_page_numbers_sum_l1003_100344

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 287 ∧ 
  ∀ m : ℕ, m > 0 ∧ m * (m + 1) = 20412 → m + (m + 1) ≥ 287 := by
  sorry

end consecutive_page_numbers_sum_l1003_100344


namespace rectangle_measurement_error_l1003_100308

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) (h_positive : L > 0 ∧ W > 0) :
  (1.10 * L) * (W * (1 - x / 100)) = L * W * 1.045 → x = 5 := by
  sorry

end rectangle_measurement_error_l1003_100308


namespace auto_finance_to_total_auto_ratio_l1003_100372

def total_consumer_credit : ℝ := 855
def auto_finance_credit : ℝ := 57
def auto_credit_percentage : ℝ := 0.20

theorem auto_finance_to_total_auto_ratio :
  let total_auto_credit := total_consumer_credit * auto_credit_percentage
  auto_finance_credit / total_auto_credit = 1/3 := by
sorry

end auto_finance_to_total_auto_ratio_l1003_100372


namespace complement_of_union_is_four_l1003_100312

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {2, 3}

-- Theorem to prove
theorem complement_of_union_is_four :
  (M ∪ N)ᶜ = {4} := by sorry

end complement_of_union_is_four_l1003_100312


namespace sara_quarters_l1003_100305

theorem sara_quarters (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 21 → additional = 49 → total = initial + additional → total = 70 := by
  sorry

end sara_quarters_l1003_100305


namespace max_k_inequality_l1003_100321

theorem max_k_inequality (k : ℝ) : 
  (∀ (x y : ℤ), 4 * x^2 + y^2 + 1 ≥ k * x * (y + 1)) ↔ k ≤ 3 := by sorry

end max_k_inequality_l1003_100321


namespace age_of_seventh_person_l1003_100338

-- Define the ages and age differences
variable (A1 A2 A3 A4 A5 A6 A7 D1 D2 D3 D4 D5 : ℕ)

-- Define the conditions
axiom age_order : A1 < A2 ∧ A2 < A3 ∧ A3 < A4 ∧ A4 < A5 ∧ A5 < A6

axiom age_differences : 
  A2 = A1 + D1 ∧
  A3 = A2 + D2 ∧
  A4 = A3 + D3 ∧
  A5 = A4 + D4 ∧
  A6 = A5 + D5

axiom sum_of_six : A1 + A2 + A3 + A4 + A5 + A6 = 246

axiom sum_of_seven : A1 + A2 + A3 + A4 + A5 + A6 + A7 = 315

-- The theorem to prove
theorem age_of_seventh_person : A7 = 69 := by
  sorry

end age_of_seventh_person_l1003_100338


namespace female_average_score_l1003_100306

theorem female_average_score (total_average : ℝ) (male_average : ℝ) (male_count : ℕ) (female_count : ℕ) :
  total_average = 90 →
  male_average = 82 →
  male_count = 8 →
  female_count = 32 →
  (male_count * male_average + female_count * ((male_count + female_count) * total_average - male_count * male_average) / female_count) / (male_count + female_count) = 90 →
  ((male_count + female_count) * total_average - male_count * male_average) / female_count = 92 := by
sorry

end female_average_score_l1003_100306


namespace minimum_value_theorem_l1003_100337

theorem minimum_value_theorem (a b m n : ℝ) : 
  (∀ x, (x + 2) / (x + 1) < 0 ↔ a < x ∧ x < b) →
  m * a + n * b + 1 = 0 →
  m * n > 0 →
  (∀ m' n', m' * n' > 0 → m' * a + n' * b + 1 = 0 → 2 / m' + 1 / n' ≥ 2 / m + 1 / n) →
  2 / m + 1 / n = 9 :=
sorry

end minimum_value_theorem_l1003_100337


namespace some_birds_are_white_l1003_100317

-- Define our universe
variable (U : Type)

-- Define our predicates
variable (Swan : U → Prop)
variable (Bird : U → Prop)
variable (White : U → Prop)

-- State our theorem
theorem some_birds_are_white
  (h1 : ∀ x, Swan x → White x)  -- All swans are white
  (h2 : ∃ x, Bird x ∧ Swan x)   -- Some birds are swans
  : ∃ x, Bird x ∧ White x :=    -- Conclusion: Some birds are white
by sorry

end some_birds_are_white_l1003_100317


namespace only_A_scored_full_marks_l1003_100386

/-- Represents the three students -/
inductive Student
  | A
  | B
  | C

/-- Represents whether a statement is true or false -/
def Statement := Bool

/-- Represents whether a student scored full marks -/
def ScoredFullMarks := Student → Bool

/-- Represents whether a student told the truth -/
def ToldTruth := Student → Bool

/-- A's statement: C did not score full marks -/
def statementA (s : ScoredFullMarks) : Statement :=
  !s Student.C

/-- B's statement: I scored full marks -/
def statementB (s : ScoredFullMarks) : Statement :=
  s Student.B

/-- C's statement: A is telling the truth -/
def statementC (t : ToldTruth) : Statement :=
  t Student.A

theorem only_A_scored_full_marks :
  ∀ (s : ScoredFullMarks) (t : ToldTruth),
    (∃! x : Student, s x = true) →
    (∃! x : Student, t x = false) →
    (t Student.A = (statementA s)) →
    (t Student.B = (statementB s)) →
    (t Student.C = (statementC t)) →
    s Student.A = true :=
sorry

end only_A_scored_full_marks_l1003_100386


namespace stratified_sampling_population_size_l1003_100348

theorem stratified_sampling_population_size 
  (total_male : ℕ) 
  (sample_size : ℕ) 
  (female_in_sample : ℕ) 
  (h1 : total_male = 570) 
  (h2 : sample_size = 110) 
  (h3 : female_in_sample = 53) :
  let male_in_sample := sample_size - female_in_sample
  let total_population := (total_male * sample_size) / male_in_sample
  total_population = 1100 := by
sorry

end stratified_sampling_population_size_l1003_100348


namespace calories_in_one_bar_l1003_100357

/-- The number of calories in 3 candy bars -/
def total_calories : ℕ := 24

/-- The number of candy bars -/
def num_bars : ℕ := 3

/-- The number of calories in one candy bar -/
def calories_per_bar : ℕ := total_calories / num_bars

theorem calories_in_one_bar : calories_per_bar = 8 := by
  sorry

end calories_in_one_bar_l1003_100357


namespace sequence_sum_lower_bound_l1003_100365

theorem sequence_sum_lower_bound (n : ℕ) (a : ℕ → ℝ) 
  (h1 : a 1 = 0)
  (h2 : ∀ i ∈ Finset.range n, i ≥ 2 → |a i| = |a (i-1) + 1|) :
  (Finset.range n).sum a ≥ -n / 2 := by
  sorry

end sequence_sum_lower_bound_l1003_100365


namespace max_segments_proof_l1003_100336

/-- Given n consecutive points on a line with total length 1, 
    this function returns the maximum number of segments with length ≥ a,
    where 0 ≤ a ≤ 1/(n-1) -/
def max_segments (n : ℕ) (a : ℝ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem stating that for n consecutive points on a line with total length 1,
    and 0 ≤ a ≤ 1/(n-1), the maximum number of segments with length ≥ a is n(n-1)/2 -/
theorem max_segments_proof (n : ℕ) (a : ℝ) 
    (h1 : n > 1) 
    (h2 : 0 ≤ a) 
    (h3 : a ≤ 1 / (n - 1)) : 
  max_segments n a = n * (n - 1) / 2 := by
  sorry

end max_segments_proof_l1003_100336


namespace quaternary_201_equals_33_l1003_100307

/-- Converts a quaternary (base 4) number to its decimal equivalent -/
def quaternary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The quaternary representation of the number -/
def quaternary_201 : List Nat := [1, 0, 2]

theorem quaternary_201_equals_33 :
  quaternary_to_decimal quaternary_201 = 33 := by
  sorry

end quaternary_201_equals_33_l1003_100307


namespace log_sum_equals_ten_l1003_100355

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sum_equals_ten :
  log 3 243 - log 3 (1/27) + log 3 9 = 10 := by
  sorry

end log_sum_equals_ten_l1003_100355


namespace unique_shapes_count_l1003_100318

-- Define a rectangle
structure Rectangle where
  vertices : Fin 4 → Point

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define an ellipse
structure Ellipse where
  foci : Point × Point
  major_axis : ℝ

-- Function to count unique shapes
def count_unique_shapes (R : Rectangle) : ℕ :=
  let circles := sorry
  let ellipses := sorry
  circles + ellipses

-- Theorem statement
theorem unique_shapes_count (R : Rectangle) :
  count_unique_shapes R = 6 :=
sorry

end unique_shapes_count_l1003_100318


namespace circle_tangency_and_chord_properties_l1003_100378

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define point P
def point_P (t : ℝ) : ℝ × ℝ := (-1, t)

-- Define circle N
def circle_N (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 1

-- Define the external tangency condition
def external_tangent (a b : ℝ) : Prop := (a - 2)^2 + b^2 = 9

-- Define the chord length condition
def chord_length (t : ℝ) : Prop := ∃ (k : ℝ), 8*k^2 + 6*t*k + t^2 - 1 = 0

-- Define the ST distance condition
def ST_distance (t : ℝ) : Prop := (t^2 + 8) / 16 = 9/16

theorem circle_tangency_and_chord_properties :
  ∀ t : ℝ,
  (∃ a b : ℝ, circle_N (-1) 1 a b ∧ external_tangent a b) →
  (chord_length t ∧ ST_distance t) →
  ((circle_N x y (-1) 0 ∨ circle_N x y (-2/5) (9/5)) ∧ (t = 1 ∨ t = -1)) :=
sorry

end circle_tangency_and_chord_properties_l1003_100378


namespace polynomial_simplification_l1003_100331

theorem polynomial_simplification (p : ℝ) :
  (5 * p^4 + 4 * p^3 - 7 * p^2 + 9 * p - 3) + (-8 * p^4 + 2 * p^3 - p^2 - 3 * p + 4) =
  -3 * p^4 + 6 * p^3 - 8 * p^2 + 6 * p + 1 := by
  sorry

end polynomial_simplification_l1003_100331
