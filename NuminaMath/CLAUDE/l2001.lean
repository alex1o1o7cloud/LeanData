import Mathlib

namespace optimal_price_l2001_200105

/-- Revenue function -/
def revenue (p : ℝ) : ℝ := 150 * p - 6 * p^2

/-- Constraint: price is at most 30 -/
def price_constraint (p : ℝ) : Prop := p ≤ 30

/-- Constraint: at least 40 books sold per month -/
def sales_constraint (p : ℝ) : Prop := 150 - 6 * p ≥ 40

/-- The optimal price is an integer -/
def integer_price (p : ℝ) : Prop := ∃ n : ℤ, p = n ∧ n > 0

/-- Theorem: The price of 13 maximizes revenue under given constraints -/
theorem optimal_price :
  ∀ p : ℝ, 
  price_constraint p → 
  sales_constraint p → 
  integer_price p → 
  revenue p ≤ revenue 13 :=
sorry

end optimal_price_l2001_200105


namespace max_fourth_power_sum_l2001_200152

theorem max_fourth_power_sum (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) :
  ∃ (m : ℝ), (∀ x y z w : ℝ, x^3 + y^3 + z^3 + w^3 = 4 → x^4 + y^4 + z^4 + w^4 ≤ m) ∧
             (a^4 + b^4 + c^4 + d^4 = m) ∧
             m = 16 :=
sorry

end max_fourth_power_sum_l2001_200152


namespace sarahs_flour_purchase_l2001_200179

/-- Sarah's flour purchase problem -/
theorem sarahs_flour_purchase
  (rye : ℝ)
  (chickpea : ℝ)
  (pastry : ℝ)
  (total : ℝ)
  (h_rye : rye = 5)
  (h_chickpea : chickpea = 3)
  (h_pastry : pastry = 2)
  (h_total : total = 20)
  : total - (rye + chickpea + pastry) = 10 := by
  sorry

end sarahs_flour_purchase_l2001_200179


namespace cubic_equation_solution_l2001_200180

theorem cubic_equation_solution : 
  ∀ x y : ℕ+, 
  (x : ℝ)^3 + (y : ℝ)^3 = 4 * ((x : ℝ)^2 * (y : ℝ) + (x : ℝ) * (y : ℝ)^2 - 5) → 
  ((x = 1 ∧ y = 3) ∨ (x = 3 ∧ y = 1)) := by
sorry

end cubic_equation_solution_l2001_200180


namespace smallest_prime_after_seven_nonprimes_l2001_200129

def is_prime (n : ℕ) : Prop := sorry

def consecutive_nonprimes (start : ℕ) (count : ℕ) : Prop :=
  ∀ k, k ≥ start ∧ k < start + count → ¬ is_prime k

theorem smallest_prime_after_seven_nonprimes :
  ∃ n : ℕ, 
    (consecutive_nonprimes n 7) ∧ 
    (is_prime (n + 7)) ∧
    (∀ m : ℕ, m < n → ¬(consecutive_nonprimes m 7 ∧ is_prime (m + 7))) ∧
    (n + 7 = 97) :=
sorry

end smallest_prime_after_seven_nonprimes_l2001_200129


namespace power_calculation_l2001_200149

theorem power_calculation : (8^5 / 8^2) * 4^6 = 2^21 := by
  sorry

end power_calculation_l2001_200149


namespace train_seats_theorem_l2001_200135

/-- The total number of seats on the train -/
def total_seats : ℕ := 180

/-- The number of seats in Standard Class -/
def standard_seats : ℕ := 36

/-- The fraction of total seats in Comfort Class -/
def comfort_fraction : ℚ := 1/5

/-- The fraction of total seats in Premium Class -/
def premium_fraction : ℚ := 3/5

/-- Theorem stating that the total number of seats is 180 -/
theorem train_seats_theorem :
  (standard_seats : ℚ) + comfort_fraction * total_seats + premium_fraction * total_seats = total_seats := by
  sorry

end train_seats_theorem_l2001_200135


namespace circle_properties_l2001_200119

/-- Given that this equation represents a circle for real m, prove the statements about m, r, and the circle's center -/
theorem circle_properties (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*(m + 3)*x + 2*(1 - 4*m^2)*y + 16*m^4 + 9 = 0 →
    ∃ r : ℝ, (x - (m + 3))^2 + (y - (4*m^2 - 1))^2 = r^2) →
  (-1 < m ∧ m < 1) ∧
  (∃ r : ℝ, 0 < r ∧ r ≤ Real.sqrt 2 ∧
    ∀ x y : ℝ, x^2 + y^2 - 2*(m + 3)*x + 2*(1 - 4*m^2)*y + 16*m^4 + 9 = 0 →
      (x - (m + 3))^2 + (y - (4*m^2 - 1))^2 = r^2) ∧
  (∃ x y : ℝ, -1 < x ∧ x < 4 ∧ y = 4*(x - 3)^2 - 1 ∧
    x^2 + y^2 - 2*(m + 3)*x + 2*(1 - 4*m^2)*y + 16*m^4 + 9 = 0) :=
by sorry

end circle_properties_l2001_200119


namespace shifted_parabola_passes_through_point_l2001_200102

/-- The original parabola equation -/
def original_parabola (x : ℝ) : ℝ := -x^2 - 2*x + 3

/-- The shifted parabola equation -/
def shifted_parabola (x : ℝ) : ℝ := -x^2 + 2

/-- Theorem stating that the shifted parabola passes through (-1, 1) -/
theorem shifted_parabola_passes_through_point :
  shifted_parabola (-1) = 1 := by sorry

end shifted_parabola_passes_through_point_l2001_200102


namespace even_function_inequality_l2001_200146

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f: ℝ → ℝ is increasing on [0, +∞) if
    for all x, y ∈ [0, +∞), x < y implies f(x) < f(y) -/
def IncreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem even_function_inequality (f : ℝ → ℝ) 
    (heven : EvenFunction f) (hinc : IncreasingOnNonnegative f) :
    f π > f (-2) ∧ f (-2) > f (-1) := by
  sorry

end even_function_inequality_l2001_200146


namespace count_two_digit_numbers_tens_less_than_ones_eq_36_l2001_200170

/-- The count of two-digit numbers where the tens digit is less than the ones digit -/
def count_two_digit_numbers_tens_less_than_ones : ℕ :=
  (Finset.range 9).sum (λ t => (Finset.range (10 - t)).card)

/-- Theorem stating that the count of two-digit numbers where the tens digit is less than the ones digit is 36 -/
theorem count_two_digit_numbers_tens_less_than_ones_eq_36 :
  count_two_digit_numbers_tens_less_than_ones = 36 := by
  sorry

end count_two_digit_numbers_tens_less_than_ones_eq_36_l2001_200170


namespace gillians_phone_bill_l2001_200193

theorem gillians_phone_bill (x : ℝ) : 
  (12 * (x * 1.1) = 660) → x = 50 := by
  sorry

end gillians_phone_bill_l2001_200193


namespace cuatro_cuinte_equation_l2001_200100

/-- Represents a mapping from letters to digits -/
def LetterToDigit := Char → Nat

/-- Check if a mapping is valid (each letter maps to a unique digit) -/
def is_valid_mapping (m : LetterToDigit) : Prop :=
  ∀ c₁ c₂, c₁ ≠ c₂ → m c₁ ≠ m c₂

/-- Convert a string to a number using the given mapping -/
def string_to_number (s : String) (m : LetterToDigit) : Nat :=
  s.foldl (fun acc c => 10 * acc + m c) 0

/-- The main theorem to prove -/
theorem cuatro_cuinte_equation (m : LetterToDigit) 
  (h_valid : is_valid_mapping m)
  (h_cuatro : string_to_number "CUATRO" m = 170349)
  (h_cuaatro : string_to_number "CUAATRO" m = 1700349)
  (h_cuinte : string_to_number "CUINTE" m = 3852345) :
  170349 + 170349 + 1700349 + 1700349 + 170349 = 3852345 := by
  sorry

/-- Lemma: The mapping satisfies the equation -/
lemma mapping_satisfies_equation (m : LetterToDigit) 
  (h_valid : is_valid_mapping m)
  (h_cuatro : string_to_number "CUATRO" m = 170349)
  (h_cuaatro : string_to_number "CUAATRO" m = 1700349)
  (h_cuinte : string_to_number "CUINTE" m = 3852345) :
  string_to_number "CUATRO" m + string_to_number "CUATRO" m + 
  string_to_number "CUAATRO" m + string_to_number "CUAATRO" m + 
  string_to_number "CUATRO" m = string_to_number "CUINTE" m := by
  sorry

end cuatro_cuinte_equation_l2001_200100


namespace rectangle_same_color_l2001_200132

-- Define the color type
def Color := Fin

-- Define a point on the plane
structure Point where
  x : Int
  y : Int

-- Define a coloring function
def coloring (p : Nat) : Point → Color p :=
  sorry

-- The main theorem
theorem rectangle_same_color (p : Nat) :
  ∃ (a b c d : Point), 
    (a.x < b.x ∧ a.y < c.y) ∧ 
    (b.x - a.x = d.x - c.x) ∧ 
    (c.y - a.y = d.y - b.y) ∧
    (coloring p a = coloring p b) ∧
    (coloring p b = coloring p c) ∧
    (coloring p c = coloring p d) :=
  sorry

end rectangle_same_color_l2001_200132


namespace charity_raffle_winnings_l2001_200186

theorem charity_raffle_winnings (winnings : ℝ) : 
  (winnings / 2 - 2 = 55) → winnings = 114 := by
  sorry

end charity_raffle_winnings_l2001_200186


namespace olivias_albums_l2001_200125

/-- Given a total number of pictures and a number of albums, 
    calculate the number of pictures in each album. -/
def pictures_per_album (total_pictures : ℕ) (num_albums : ℕ) : ℕ :=
  total_pictures / num_albums

/-- Prove that given 40 total pictures and 8 albums, 
    there are 5 pictures in each album. -/
theorem olivias_albums : 
  let total_pictures : ℕ := 40
  let num_albums : ℕ := 8
  pictures_per_album total_pictures num_albums = 5 := by
  sorry

end olivias_albums_l2001_200125


namespace gcd_lcm_product_24_60_l2001_200192

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end gcd_lcm_product_24_60_l2001_200192


namespace count_solutions_correct_l2001_200181

/-- The number of integer solutions to x^2 - y^2 = 45 -/
def count_solutions : ℕ := 12

/-- A pair of integers (x, y) is a solution if x^2 - y^2 = 45 -/
def is_solution (x y : ℤ) : Prop := x^2 - y^2 = 45

/-- The theorem stating that there are exactly 12 integer solutions to x^2 - y^2 = 45 -/
theorem count_solutions_correct :
  (∃ (s : Finset (ℤ × ℤ)), s.card = count_solutions ∧ 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ is_solution p.1 p.2)) :=
sorry


end count_solutions_correct_l2001_200181


namespace work_completion_smaller_group_l2001_200155

/-- Given that 22 men complete a work in 55 days, and another group completes
    the same work in 121 days, prove that the number of men in the second group is 10. -/
theorem work_completion_smaller_group : 
  ∀ (work : ℕ) (group1_size group1_days group2_days : ℕ),
    group1_size = 22 →
    group1_days = 55 →
    group2_days = 121 →
    group1_size * group1_days = work →
    ∃ (group2_size : ℕ), 
      group2_size * group2_days = work ∧
      group2_size = 10 :=
by
  sorry

#check work_completion_smaller_group

end work_completion_smaller_group_l2001_200155


namespace circle_tangency_l2001_200109

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c1_center : ℝ × ℝ) (c1_radius : ℝ) 
                       (c2_center : ℝ × ℝ) (c2_radius : ℝ) : Prop :=
  (c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2 = (c1_radius + c2_radius)^2

theorem circle_tangency (a : ℝ) (h : a > 0) :
  externally_tangent (a, 0) 2 (0, Real.sqrt 5) 3 → a = 2 * Real.sqrt 5 := by
  sorry

end circle_tangency_l2001_200109


namespace normal_lemon_tree_production_l2001_200121

/-- The number of lemons produced by a normal lemon tree per year. -/
def normal_lemon_production : ℕ := 60

/-- The number of trees in Jim's grove. -/
def jims_trees : ℕ := 1500

/-- The number of lemons Jim's grove produces per year. -/
def jims_production : ℕ := 135000

/-- Jim's trees produce 50% more lemons than normal trees. -/
def jims_tree_efficiency : ℚ := 3/2

theorem normal_lemon_tree_production :
  normal_lemon_production * jims_trees * jims_tree_efficiency = jims_production :=
by sorry

end normal_lemon_tree_production_l2001_200121


namespace largest_factorial_divisor_l2001_200134

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem largest_factorial_divisor :
  ∀ m : ℕ, m > 98 → ¬(factorial m ∣ factorial 100 + factorial 99 + factorial 98) ∧
  (factorial 98 ∣ factorial 100 + factorial 99 + factorial 98) := by
  sorry

end largest_factorial_divisor_l2001_200134


namespace inequality_solution_range_l2001_200162

/-- The solution set of the inequality |x - 1| < kx contains exactly three integers -/
def has_three_integer_solutions (k : ℝ) : Prop :=
  ∃ (a b c : ℤ), a < b ∧ b < c ∧
  (∀ x : ℝ, |x - 1| < k * x ↔ (x > a ∧ x < c)) ∧
  (∀ n : ℤ, |n - 1| < k * n ↔ (n = a + 1 ∨ n = b ∨ n = c - 1))

/-- The main theorem -/
theorem inequality_solution_range (k : ℝ) :
  has_three_integer_solutions k → k ∈ Set.Ioo (2/3) (3/4) := by
  sorry

end inequality_solution_range_l2001_200162


namespace darias_savings_correct_l2001_200168

/-- Calculates the weekly savings amount needed to reach a target --/
def weekly_savings (total_cost : ℕ) (initial_savings : ℕ) (weeks : ℕ) : ℕ :=
  (total_cost - initial_savings) / weeks

/-- Proves that Daria's weekly savings amount is correct --/
theorem darias_savings_correct (total_cost initial_savings weeks : ℕ)
  (h1 : total_cost = 120)
  (h2 : initial_savings = 20)
  (h3 : weeks = 10) :
  weekly_savings total_cost initial_savings weeks = 10 := by
  sorry

end darias_savings_correct_l2001_200168


namespace typing_problem_l2001_200143

/-- Represents the typing speed of a typist in pages per hour -/
structure TypingSpeed :=
  (speed : ℝ)

/-- Represents the length of a chapter in pages -/
structure ChapterLength :=
  (pages : ℝ)

/-- Represents the time taken to type a chapter in hours -/
structure TypingTime :=
  (hours : ℝ)

theorem typing_problem (x y : TypingSpeed) (c1 c2 c3 : ChapterLength) (t1 t2 : TypingTime) :
  -- First chapter is twice as short as the second
  c1.pages = c2.pages / 2 →
  -- First chapter is three times longer than the third
  c1.pages = 3 * c3.pages →
  -- Typists retyped first chapter together in 3 hours and 36 minutes
  t1.hours = 3.6 →
  c1.pages / (x.speed + y.speed) = t1.hours →
  -- Second chapter was retyped in 8 hours
  t2.hours = 8 →
  -- First typist worked alone for 2 hours on second chapter
  2 * x.speed + 6 * (x.speed + y.speed) = c2.pages →
  -- Time for second typist to retype third chapter alone
  c3.pages / y.speed = 3 := by
sorry

end typing_problem_l2001_200143


namespace calculation_proof_l2001_200165

theorem calculation_proof :
  (5.42 - (3.75 - 0.58) = 2.25) ∧
  ((4/5) * 7.7 + 0.8 * 3.3 - (4/5) = 8) := by
  sorry

end calculation_proof_l2001_200165


namespace divisibility_condition_l2001_200151

theorem divisibility_condition (a n : ℕ+) : 
  n ∣ ((a + 1)^n.val - a^n.val) ↔ n = 1 := by
  sorry

end divisibility_condition_l2001_200151


namespace max_checkers_theorem_l2001_200108

/-- Represents a chessboard configuration -/
structure ChessboardConfig where
  size : Nat
  white_checkers : Nat
  black_checkers : Nat

/-- Checks if a chessboard configuration is valid -/
def is_valid_config (c : ChessboardConfig) : Prop :=
  c.size = 8 ∧
  c.white_checkers = 2 * c.black_checkers ∧
  c.white_checkers + c.black_checkers ≤ c.size * c.size

/-- The maximum number of checkers that can be placed -/
def max_checkers : Nat := 48

/-- Theorem: The maximum number of checkers that can be placed on an 8x8 chessboard,
    such that each row and column contains twice as many white checkers as black ones, is 48 -/
theorem max_checkers_theorem (c : ChessboardConfig) :
  is_valid_config c → c.white_checkers + c.black_checkers ≤ max_checkers :=
by
  sorry

#check max_checkers_theorem

end max_checkers_theorem_l2001_200108


namespace count_digit_nine_to_thousand_l2001_200127

/-- The number of occurrences of a digit in a specific place (units, tens, or hundreds) for numbers from 1 to 1000 -/
def occurrences_in_place : ℕ := 100

/-- The number of places (units, tens, hundreds) in numbers from 1 to 1000 -/
def num_places : ℕ := 3

/-- The digit we're counting -/
def target_digit : ℕ := 9

/-- Theorem: The number of occurrences of the digit 9 in the list of integers from 1 to 1000 is equal to 300 -/
theorem count_digit_nine_to_thousand : 
  occurrences_in_place * num_places = 300 :=
sorry

end count_digit_nine_to_thousand_l2001_200127


namespace nell_remaining_cards_l2001_200116

/-- Proves that Nell has 276 cards after giving away 28 cards from her initial 304 cards. -/
theorem nell_remaining_cards (initial_cards : ℕ) (cards_given : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 304 → cards_given = 28 → remaining_cards = initial_cards - cards_given → remaining_cards = 276 := by
  sorry

end nell_remaining_cards_l2001_200116


namespace car_distance_theorem_l2001_200176

theorem car_distance_theorem (initial_time : ℝ) (new_speed : ℝ) :
  initial_time = 6 →
  new_speed = 56 →
  ∃ (distance : ℝ),
    distance = new_speed * (3/2 * initial_time) ∧
    distance = 504 := by
  sorry

end car_distance_theorem_l2001_200176


namespace composition_equality_l2001_200166

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 4
def g (x : ℝ) : ℝ := 5 * x + 2

-- State the theorem
theorem composition_equality : f (g (f 3)) = 108 := by
  sorry

end composition_equality_l2001_200166


namespace intersection_of_M_and_N_l2001_200169

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end intersection_of_M_and_N_l2001_200169


namespace fourth_term_is_negative_eight_l2001_200110

/-- A geometric sequence with a quadratic equation and specific conditions -/
structure GeometricSequence where
  -- The quadratic equation coefficients
  a : ℝ
  b : ℝ
  c : ℝ
  -- The condition that the quadratic equation holds for the sequence
  quad_eq : a * t^2 + b * t + c = 0
  -- The conditions given in the problem
  sum_condition : a1 + a2 = -1
  diff_condition : a1 - a3 = -3
  -- The general term of the sequence
  a_n : ℕ → ℝ

/-- The theorem stating that the fourth term of the sequence is -8 -/
theorem fourth_term_is_negative_eight (seq : GeometricSequence) :
  seq.a = 1 ∧ seq.b = -36 ∧ seq.c = 288 →
  seq.a_n 4 = -8 := by
  sorry

end fourth_term_is_negative_eight_l2001_200110


namespace remainder_of_repeated_12_l2001_200120

def repeated_digit_number (n : ℕ) : ℕ := 
  -- Function to generate the number with n repetitions of "12"
  -- Implementation details omitted for brevity
  sorry

theorem remainder_of_repeated_12 (n : ℕ) :
  repeated_digit_number 150 % 99 = 18 := by
  sorry

end remainder_of_repeated_12_l2001_200120


namespace michael_chicken_count_l2001_200117

/-- Calculates the number of chickens after a given number of years -/
def chickenCount (initialCount : ℕ) (annualIncrease : ℕ) (years : ℕ) : ℕ :=
  initialCount + annualIncrease * years

/-- Theorem stating that Michael will have 1900 chickens after 9 years -/
theorem michael_chicken_count :
  chickenCount 550 150 9 = 1900 := by
  sorry

end michael_chicken_count_l2001_200117


namespace pentagon_area_bound_l2001_200164

-- Define the pentagon ABCDE
variable (A B C D E : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def is_convex_pentagon (A B C D E : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def angle (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

def distance (P Q : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

def area (A B C D E : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Theorem statement
theorem pentagon_area_bound 
  (h_convex : is_convex_pentagon A B C D E)
  (h_angle_EAB : angle E A B = 2 * π / 3)
  (h_angle_ABC : angle A B C = 2 * π / 3)
  (h_angle_ADB : angle A D B = π / 6)
  (h_angle_CDE : angle C D E = π / 3)
  (h_side_AB : distance A B = 1) :
  area A B C D E < Real.sqrt 3 := by
  sorry

end pentagon_area_bound_l2001_200164


namespace shooting_challenge_sequences_l2001_200126

theorem shooting_challenge_sequences : ℕ := by
  -- Define the total number of targets
  let total_targets : ℕ := 10

  -- Define the number of targets in each column
  let targets_A : ℕ := 4
  let targets_B : ℕ := 4
  let targets_C : ℕ := 2

  -- Assert that the sum of targets in all columns equals the total targets
  have h1 : targets_A + targets_B + targets_C = total_targets := by sorry

  -- Define the number of different sequences
  let num_sequences : ℕ := (Nat.factorial total_targets) / 
    ((Nat.factorial targets_A) * (Nat.factorial targets_B) * (Nat.factorial targets_C))

  -- Prove that the number of sequences equals 3150
  have h2 : num_sequences = 3150 := by sorry

  -- Return the result
  exact 3150

end shooting_challenge_sequences_l2001_200126


namespace gunther_typing_capacity_l2001_200124

/-- Given Gunther's typing speed and work day length, prove the number of words he can type in a day --/
theorem gunther_typing_capacity (words_per_set : ℕ) (minutes_per_set : ℕ) (minutes_per_day : ℕ) 
  (h1 : words_per_set = 160)
  (h2 : minutes_per_set = 3)
  (h3 : minutes_per_day = 480) :
  (minutes_per_day / minutes_per_set) * words_per_set = 25600 := by
  sorry

#eval (480 / 3) * 160

end gunther_typing_capacity_l2001_200124


namespace smallest_four_digit_divisible_by_3_and_8_l2001_200113

theorem smallest_four_digit_divisible_by_3_and_8 :
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n < 10000) ∧ 
    n % 3 = 0 ∧ 
    n % 8 = 0 ∧
    (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) → m % 3 = 0 → m % 8 = 0 → n ≤ m) ∧
    n = 1008 :=
by
  sorry

end smallest_four_digit_divisible_by_3_and_8_l2001_200113


namespace not_center_of_symmetry_l2001_200145

/-- Given that the centers of symmetry for tan(x) are of the form (kπ/2, 0) where k is any integer,
    prove that (-π/18, 0) is not a center of symmetry for the function t = tan(3x + π/3) -/
theorem not_center_of_symmetry :
  ¬ (∃ (k : ℤ), -π/18 = k*π/6 - π/9) := by sorry

end not_center_of_symmetry_l2001_200145


namespace town_population_male_count_l2001_200178

theorem town_population_male_count (total_population : ℕ) (num_groups : ℕ) (male_groups : ℕ) : 
  total_population = 480 →
  num_groups = 4 →
  male_groups = 2 →
  (total_population / num_groups) * male_groups = 240 := by
sorry

end town_population_male_count_l2001_200178


namespace tangent_and_trig_identity_l2001_200114

theorem tangent_and_trig_identity (α : Real) (h : Real.tan α = -2) : 
  Real.tan (α - 7 * Real.pi) = -2 ∧ 
  (2 * Real.sin (Real.pi - α) * Real.sin (α - Real.pi / 2)) / 
  (Real.sin α ^ 2 - 2 * Real.cos α ^ 2) = 2 := by
  sorry

end tangent_and_trig_identity_l2001_200114


namespace compute_expression_l2001_200197

theorem compute_expression : 15 * (30 / 6)^2 = 375 := by sorry

end compute_expression_l2001_200197


namespace probability_two_black_balls_is_three_tenths_l2001_200148

def total_balls : ℕ := 16
def black_balls : ℕ := 9

def probability_two_black_balls : ℚ :=
  (black_balls.choose 2) / (total_balls.choose 2)

theorem probability_two_black_balls_is_three_tenths :
  probability_two_black_balls = 3 / 10 := by
  sorry

end probability_two_black_balls_is_three_tenths_l2001_200148


namespace xiao_wei_wears_five_l2001_200142

/-- Represents the five people in the line -/
inductive Person : Type
  | XiaoWang
  | XiaoZha
  | XiaoTian
  | XiaoYan
  | XiaoWei

/-- Represents the hat numbers -/
inductive HatNumber : Type
  | One
  | Two
  | Three
  | Four
  | Five

/-- Function that assigns a hat number to each person -/
def hatAssignment : Person → HatNumber := sorry

/-- Function that determines if a person can see another person's hat -/
def canSee : Person → Person → Prop := sorry

/-- The hat numbers are all different -/
axiom all_different : ∀ p q : Person, p ≠ q → hatAssignment p ≠ hatAssignment q

/-- Xiao Wang cannot see any hats -/
axiom xiao_wang_sees_none : ∀ p : Person, ¬(canSee Person.XiaoWang p)

/-- Xiao Zha can only see hat 4 -/
axiom xiao_zha_sees_four : ∃! p : Person, canSee Person.XiaoZha p ∧ hatAssignment p = HatNumber.Four

/-- Xiao Tian does not see hat 3, but can see hat 1 -/
axiom xiao_tian_condition : (∃ p : Person, canSee Person.XiaoTian p ∧ hatAssignment p = HatNumber.One) ∧
                            (∀ p : Person, canSee Person.XiaoTian p → hatAssignment p ≠ HatNumber.Three)

/-- Xiao Yan can see three hats, but not hat 3 -/
axiom xiao_yan_condition : (∃ p q r : Person, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
                            canSee Person.XiaoYan p ∧ canSee Person.XiaoYan q ∧ canSee Person.XiaoYan r) ∧
                           (∀ p : Person, canSee Person.XiaoYan p → hatAssignment p ≠ HatNumber.Three)

/-- Xiao Wei can see hat 3 and hat 2 -/
axiom xiao_wei_condition : (∃ p : Person, canSee Person.XiaoWei p ∧ hatAssignment p = HatNumber.Three) ∧
                           (∃ q : Person, canSee Person.XiaoWei q ∧ hatAssignment q = HatNumber.Two)

/-- Theorem: Xiao Wei is wearing hat number 5 -/
theorem xiao_wei_wears_five : hatAssignment Person.XiaoWei = HatNumber.Five := by sorry

end xiao_wei_wears_five_l2001_200142


namespace sum_of_digits_B_is_seven_l2001_200123

def sum_of_digits (n : ℕ) : ℕ := sorry

def A : ℕ := sum_of_digits (4444^4444)

def B : ℕ := sum_of_digits A

theorem sum_of_digits_B_is_seven : sum_of_digits B = 7 := by sorry

end sum_of_digits_B_is_seven_l2001_200123


namespace cot_sixty_degrees_l2001_200106

theorem cot_sixty_degrees : Real.cos (π / 3) / Real.sin (π / 3) = Real.sqrt 3 / 3 := by
  sorry

end cot_sixty_degrees_l2001_200106


namespace chosen_number_l2001_200195

theorem chosen_number (x : ℝ) : x / 8 - 100 = 6 → x = 848 := by
  sorry

end chosen_number_l2001_200195


namespace fundamental_theorem_of_algebra_l2001_200185

-- Define a polynomial with complex coefficients
def ComplexPolynomial := ℂ → ℂ

-- State the fundamental theorem of algebra
theorem fundamental_theorem_of_algebra :
  ∀ (P : ComplexPolynomial), ∃ (z : ℂ), P z = 0 :=
sorry

end fundamental_theorem_of_algebra_l2001_200185


namespace math_problem_proof_l2001_200139

theorem math_problem_proof :
  let expression1 := -3^2 + 2^2023 * (-1/2)^2022 + (-2024)^0
  let x : ℚ := -1/2
  let y : ℚ := 1
  let expression2 := ((x + 2*y)^2 - (2*x + y)*(2*x - y) - 5*(x^2 + y^2)) / (2*x)
  expression1 = -6 ∧ expression2 = 4 := by sorry

end math_problem_proof_l2001_200139


namespace no_five_consecutive_divisible_by_2005_l2001_200140

/-- The sequence a_n defined as 1 + 2^n + 3^n + 4^n + 5^n -/
def a (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

/-- Theorem stating that there are no 5 consecutive terms in the sequence a_n all divisible by 2005 -/
theorem no_five_consecutive_divisible_by_2005 :
  ∀ m : ℕ, ¬(∀ k : Fin 5, 2005 ∣ a (m + k)) :=
by sorry

end no_five_consecutive_divisible_by_2005_l2001_200140


namespace coplanar_points_l2001_200183

/-- The points (0,0,0), (1,a,0), (0,1,a), and (a,0,1) are coplanar if and only if a = -1 -/
theorem coplanar_points (a : ℝ) : 
  (Matrix.det
    ![![1, 0, a],
      ![a, 1, 0],
      ![0, a, 1]] = 0) ↔ a = -1 := by
  sorry

end coplanar_points_l2001_200183


namespace arrangement_from_combination_l2001_200187

theorem arrangement_from_combination (n : ℕ) (h1 : n ≥ 2) (h2 : Nat.choose n 2 = 15) : 
  n * (n - 1) = 30 := by
  sorry

end arrangement_from_combination_l2001_200187


namespace estimate_sqrt_difference_l2001_200173

theorem estimate_sqrt_difference (ε : Real) (h : ε > 0) : 
  |Real.sqrt 58 - Real.sqrt 55 - 0.20| < ε :=
sorry

end estimate_sqrt_difference_l2001_200173


namespace f_properties_l2001_200112

/-- The function f(x) that attains an extremum of 0 at x = -1 -/
def f (a b x : ℝ) : ℝ := x^3 + 2*a*x^2 + b*x + a - 1

/-- Theorem stating the properties of f(x) -/
theorem f_properties (a b : ℝ) :
  (f a b (-1) = 0 ∧ (deriv (f a b)) (-1) = 0) →
  (a = 1 ∧ b = 1 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, f 1 1 x ≤ 4) :=
by sorry

end f_properties_l2001_200112


namespace three_number_sum_l2001_200177

theorem three_number_sum : ∀ a b c : ℝ,
  a ≤ b → b ≤ c →
  b = 7 →
  (a + b + c) / 3 = a + 8 →
  (a + b + c) / 3 = c - 20 →
  a + b + c = 57 := by
sorry

end three_number_sum_l2001_200177


namespace catherine_pencil_distribution_l2001_200107

theorem catherine_pencil_distribution (initial_pens : ℕ) (initial_pencils : ℕ) 
  (friends : ℕ) (pens_per_friend : ℕ) (total_left : ℕ) :
  initial_pens = 60 →
  initial_pencils = initial_pens →
  friends = 7 →
  pens_per_friend = 8 →
  total_left = 22 →
  ∃ (pencils_per_friend : ℕ),
    pencils_per_friend * friends = initial_pencils - (total_left - (initial_pens - pens_per_friend * friends)) ∧
    pencils_per_friend = 6 :=
by sorry

end catherine_pencil_distribution_l2001_200107


namespace oliver_tickets_used_l2001_200175

theorem oliver_tickets_used (ferris_rides bumper_rides ticket_cost : ℕ) 
  (h1 : ferris_rides = 5)
  (h2 : bumper_rides = 4)
  (h3 : ticket_cost = 7) :
  (ferris_rides + bumper_rides) * ticket_cost = 63 := by
  sorry

end oliver_tickets_used_l2001_200175


namespace floor_sum_example_l2001_200101

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_example_l2001_200101


namespace multiply_add_distribute_compute_expression_l2001_200141

theorem multiply_add_distribute (a b c : ℕ) : a * b + c * a = a * (b + c) := by sorry

theorem compute_expression : 45 * 27 + 73 * 45 = 4500 := by sorry

end multiply_add_distribute_compute_expression_l2001_200141


namespace canDisplay_totalCans_l2001_200191

/-- The number of cans in each layer forms an arithmetic sequence -/
def canSequence (n : ℕ) : ℕ := 35 - 3 * n

/-- The total number of layers in the display -/
def numLayers : ℕ := 12

/-- The total number of cans in the display -/
def totalCans : ℕ := (numLayers * (canSequence 0 + canSequence (numLayers - 1))) / 2

theorem canDisplay_totalCans : totalCans = 216 := by
  sorry

end canDisplay_totalCans_l2001_200191


namespace book_problem_solution_l2001_200161

/-- Represents the cost and quantity relationships between two types of books -/
structure BookProblem where
  cost_diff : ℕ             -- Cost difference between type B and type A
  total_cost_A : ℕ          -- Total cost for type A books
  total_cost_B : ℕ          -- Total cost for type B books
  total_books : ℕ           -- Total number of books to purchase
  max_total_cost : ℕ        -- Maximum total cost allowed

/-- Calculates the cost of type A books given the problem parameters -/
def cost_A (p : BookProblem) : ℕ :=
  p.total_cost_A * p.total_cost_B / (p.total_cost_B - p.total_cost_A * p.cost_diff)

/-- Calculates the cost of type B books given the problem parameters -/
def cost_B (p : BookProblem) : ℕ :=
  cost_A p + p.cost_diff

/-- Calculates the minimum number of type A books to purchase -/
def min_books_A (p : BookProblem) : ℕ :=
  (p.total_books * cost_B p - p.max_total_cost) / (cost_B p - cost_A p)

/-- Theorem stating the solution to the book purchasing problem -/
theorem book_problem_solution (p : BookProblem) 
  (h : p = { cost_diff := 20, total_cost_A := 540, total_cost_B := 780, 
             total_books := 70, max_total_cost := 3550 }) : 
  cost_A p = 45 ∧ cost_B p = 65 ∧ min_books_A p = 50 := by
  sorry

end book_problem_solution_l2001_200161


namespace total_toys_cost_is_20_74_l2001_200118

/-- The amount spent on toy cars -/
def toy_cars_cost : ℚ := 14.88

/-- The amount spent on toy trucks -/
def toy_trucks_cost : ℚ := 5.86

/-- The total amount spent on toys -/
def total_toys_cost : ℚ := toy_cars_cost + toy_trucks_cost

/-- Theorem stating that the total amount spent on toys is $20.74 -/
theorem total_toys_cost_is_20_74 : total_toys_cost = 20.74 := by sorry

end total_toys_cost_is_20_74_l2001_200118


namespace binomial_coefficient_n_choose_n_minus_one_l2001_200158

theorem binomial_coefficient_n_choose_n_minus_one (n : ℕ+) : 
  Nat.choose n.val (n.val - 1) = n.val := by
  sorry

end binomial_coefficient_n_choose_n_minus_one_l2001_200158


namespace sum_of_squares_nonzero_implies_one_nonzero_l2001_200157

theorem sum_of_squares_nonzero_implies_one_nonzero (a b : ℝ) : 
  a^2 + b^2 ≠ 0 → a ≠ 0 ∨ b ≠ 0 := by
  sorry

end sum_of_squares_nonzero_implies_one_nonzero_l2001_200157


namespace interior_diagonal_sum_for_specific_box_l2001_200167

/-- Represents a rectangular box with given surface area and edge length sum -/
structure RectangularBox where
  surface_area : ℝ
  edge_length_sum : ℝ

/-- Calculates the sum of lengths of all interior diagonals of a rectangular box -/
def interior_diagonal_sum (box : RectangularBox) : ℝ :=
  sorry

/-- Theorem: For a rectangular box with surface area 130 and edge length sum 56,
    the sum of interior diagonal lengths is 4√66 -/
theorem interior_diagonal_sum_for_specific_box :
  let box : RectangularBox := { surface_area := 130, edge_length_sum := 56 }
  interior_diagonal_sum box = 4 * Real.sqrt 66 := by
  sorry

end interior_diagonal_sum_for_specific_box_l2001_200167


namespace same_solution_implies_c_equals_6_l2001_200160

theorem same_solution_implies_c_equals_6 (x : ℝ) (c : ℝ) : 
  (3 * x + 6 = 0) → (c * x + 15 = 3) → c = 6 := by
  sorry

end same_solution_implies_c_equals_6_l2001_200160


namespace exists_valid_strategy_365_l2001_200130

/-- A strategy for sorting n elements using 3-way comparisons -/
def SortingStrategy (n : ℕ) := ℕ

/-- The number of 3-way comparisons needed to sort n elements using a given strategy -/
def comparisons (n : ℕ) (s : SortingStrategy n) : ℕ := sorry

/-- A strategy is valid if it correctly sorts n elements -/
def is_valid_strategy (n : ℕ) (s : SortingStrategy n) : Prop := sorry

/-- The main theorem: there exists a valid strategy for 365 elements using at most 1691 comparisons -/
theorem exists_valid_strategy_365 :
  ∃ (s : SortingStrategy 365), is_valid_strategy 365 s ∧ comparisons 365 s ≤ 1691 := by sorry

end exists_valid_strategy_365_l2001_200130


namespace largest_solution_of_equation_l2001_200104

theorem largest_solution_of_equation (c : ℝ) : 
  (3 * c + 6) * (c - 2) = 9 * c → c ≤ 4 :=
by sorry

end largest_solution_of_equation_l2001_200104


namespace rectangular_field_width_l2001_200174

/-- 
Given a rectangular field where the length is 7/5 of its width and the perimeter is 360 meters,
prove that the width of the field is 75 meters.
-/
theorem rectangular_field_width (width length perimeter : ℝ) : 
  length = (7/5) * width → 
  perimeter = 2 * length + 2 * width → 
  perimeter = 360 → 
  width = 75 := by sorry

end rectangular_field_width_l2001_200174


namespace quadratic_discriminant_nonnegative_l2001_200154

theorem quadratic_discriminant_nonnegative (x : ℤ) :
  x^2 * (49 - 40*x^2) ≥ 0 ↔ x = 0 ∨ x = 1 ∨ x = -1 := by
  sorry

end quadratic_discriminant_nonnegative_l2001_200154


namespace amy_earnings_l2001_200133

theorem amy_earnings (hourly_wage : ℝ) (hours_worked : ℝ) (tips : ℝ) :
  hourly_wage = 2 → hours_worked = 7 → tips = 9 →
  hourly_wage * hours_worked + tips = 23 := by
  sorry

end amy_earnings_l2001_200133


namespace initial_overs_calculation_l2001_200136

/-- Proves the number of initial overs in a cricket game given specific conditions --/
theorem initial_overs_calculation (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) 
  (remaining_overs : ℝ) (h1 : target = 272) (h2 : initial_rate = 3.2) (h3 : required_rate = 6) 
  (h4 : remaining_overs = 40) :
  ∃ (initial_overs : ℝ), initial_overs = 10 ∧ 
  target = initial_rate * initial_overs + required_rate * remaining_overs :=
by
  sorry


end initial_overs_calculation_l2001_200136


namespace sin_squared_plus_sin_double_l2001_200171

theorem sin_squared_plus_sin_double (α : Real) (h : Real.tan α = 1/2) :
  Real.sin α ^ 2 + Real.sin (2 * α) = 1 := by sorry

end sin_squared_plus_sin_double_l2001_200171


namespace greatest_multiple_of_6_and_5_less_than_1000_l2001_200147

theorem greatest_multiple_of_6_and_5_less_than_1000 : ∃ n : ℕ, 
  n = 990 ∧ 
  6 ∣ n ∧ 
  5 ∣ n ∧ 
  n < 1000 ∧ 
  ∀ m : ℕ, (6 ∣ m ∧ 5 ∣ m ∧ m < 1000) → m ≤ n :=
by sorry

end greatest_multiple_of_6_and_5_less_than_1000_l2001_200147


namespace f_properties_l2001_200172

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x) + Real.log (1 + x) + x^4 - 2*x^2

theorem f_properties :
  (∀ x, f x ≠ 0 → -1 < x ∧ x < 1) ∧
  (∀ x, f (-x) = f x) ∧
  (∀ y, (∃ x, f x = y) → y ≤ 0) :=
by sorry

end f_properties_l2001_200172


namespace pipe_crate_height_difference_l2001_200184

/-- The height difference between two crates of cylindrical pipes -/
theorem pipe_crate_height_difference (pipe_diameter : ℝ) (crate_a_rows : ℕ) (crate_b_rows : ℕ) :
  pipe_diameter = 20 →
  crate_a_rows = 10 →
  crate_b_rows = 9 →
  let crate_a_height := crate_a_rows * pipe_diameter
  let crate_b_height := crate_b_rows * pipe_diameter + (crate_b_rows - 1) * pipe_diameter * Real.sqrt 3
  crate_a_height - crate_b_height = 20 - 160 * Real.sqrt 3 := by
sorry


end pipe_crate_height_difference_l2001_200184


namespace complement_A_intersect_B_l2001_200122

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {-2, -1, 0}
def B : Finset Int := {0, 1, 2}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {1, 2} := by sorry

end complement_A_intersect_B_l2001_200122


namespace coin_fraction_missing_l2001_200103

theorem coin_fraction_missing (x : ℚ) : x > 0 → 
  let lost := x / 2
  let found := (3 / 4) * lost
  x - (x - lost + found) = x / 8 := by
sorry

end coin_fraction_missing_l2001_200103


namespace red_apples_count_l2001_200189

def basket_problem (total_apples green_apples : ℕ) : Prop :=
  total_apples = 9 ∧ green_apples = 2 → total_apples - green_apples = 7

theorem red_apples_count : basket_problem 9 2 := by
  sorry

end red_apples_count_l2001_200189


namespace parallelogram_xy_product_l2001_200153

/-- A parallelogram with side lengths specified by parameters -/
structure Parallelogram (x y : ℝ) where
  ef : ℝ
  fg : ℝ
  gh : ℝ
  he : ℝ
  ef_eq : ef = 42
  fg_eq : fg = 4 * y^2
  gh_eq : gh = 3 * x + 6
  he_eq : he = 32
  opposite_sides_equal : ef = gh ∧ fg = he

/-- The product of x and y in the specified parallelogram is 24√2 -/
theorem parallelogram_xy_product (x y : ℝ) (p : Parallelogram x y) :
  x * y = 24 * Real.sqrt 2 := by
  sorry

end parallelogram_xy_product_l2001_200153


namespace f_difference_960_480_l2001_200150

def sum_of_divisors (n : ℕ) : ℕ := sorry

def f (n : ℕ) : ℚ := (sum_of_divisors n : ℚ) / n

theorem f_difference_960_480 : f 960 - f 480 = 1 / 40 := by sorry

end f_difference_960_480_l2001_200150


namespace parabola_zeros_difference_l2001_200196

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
def Parabola.y (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The zeros of the parabola -/
def Parabola.zeros (p : Parabola) : Set ℝ :=
  {x : ℝ | p.y x = 0}

theorem parabola_zeros_difference (p : Parabola) :
  p.y 1 = -2 →  -- Vertex at (1, -2)
  p.y 3 = 10 →  -- Point (3, 10) on the parabola
  ∃ m n : ℝ,
    m ∈ p.zeros ∧
    n ∈ p.zeros ∧
    m > n ∧
    m - n = 2 * Real.sqrt 6 / 3 := by
  sorry

end parabola_zeros_difference_l2001_200196


namespace lcm_problem_l2001_200111

theorem lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 16) (h2 : a * b = 2560) :
  Nat.lcm a b = 160 := by
  sorry

end lcm_problem_l2001_200111


namespace problem_statement_l2001_200144

theorem problem_statement (m n : ℤ) (h : 2*m - 3*n = 7) : 8 - 2*m + 3*n = 1 := by
  sorry

end problem_statement_l2001_200144


namespace ratio_problem_l2001_200115

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 ∧ B / C = 1/6) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 5/8 := by sorry

end ratio_problem_l2001_200115


namespace knights_archery_skill_l2001_200159

theorem knights_archery_skill (total : ℕ) (total_pos : total > 0) : 
  let gold := (3 * total) / 8
  let silver := total - gold
  let skilled := total / 4
  ∃ (gold_skilled silver_skilled : ℕ),
    gold_skilled + silver_skilled = skilled ∧
    gold_skilled * silver = 3 * silver_skilled * gold ∧
    gold_skilled * 7 = gold * 3 := by
  sorry

end knights_archery_skill_l2001_200159


namespace enemies_left_proof_l2001_200198

def enemies_left_undefeated (total_enemies : ℕ) (points_per_enemy : ℕ) (total_points : ℕ) : ℕ :=
  total_enemies - (total_points / points_per_enemy)

theorem enemies_left_proof (total_enemies : ℕ) (points_per_enemy : ℕ) (total_points : ℕ)
  (h1 : total_enemies = 11)
  (h2 : points_per_enemy = 9)
  (h3 : total_points = 72) :
  enemies_left_undefeated total_enemies points_per_enemy total_points = 3 :=
by
  sorry

#eval enemies_left_undefeated 11 9 72

end enemies_left_proof_l2001_200198


namespace inverse_sum_product_l2001_200131

theorem inverse_sum_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h_sum : 3*x + y/3 ≠ 0) : 
  (3*x + y/3)⁻¹ * ((3*x)⁻¹ + (y/3)⁻¹) = (x*y)⁻¹ := by
  sorry

end inverse_sum_product_l2001_200131


namespace inequality_system_solution_l2001_200128

theorem inequality_system_solution (x : ℝ) :
  (3 * x - 5 ≤ x + 1 ∧ (x - 1) / 2 > x - 4) ↔ x < 3 := by
sorry

end inequality_system_solution_l2001_200128


namespace diplomats_not_speaking_russian_l2001_200188

theorem diplomats_not_speaking_russian (total : ℕ) (french : ℕ) (both_percent : ℚ) (neither_percent : ℚ) 
  (h_total : total = 70)
  (h_french : french = 25)
  (h_both : both_percent = 1/10)
  (h_neither : neither_percent = 1/5) : 
  total - (total : ℚ) * (1 - neither_percent) + french - total * both_percent = 39 := by
  sorry

end diplomats_not_speaking_russian_l2001_200188


namespace line_intercepts_sum_l2001_200137

/-- Given a line with equation y - 3 = -3(x - 5), 
    the sum of its x-intercept and y-intercept is 24 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y - 3 = -3 * (x - 5)) → 
  ∃ (x_int y_int : ℝ), 
    (y_int - 3 = -3 * (x_int - 5)) ∧ 
    (0 - 3 = -3 * (x_int - 5)) ∧ 
    (y_int - 3 = -3 * (0 - 5)) ∧ 
    (x_int + y_int = 24) :=
by sorry

end line_intercepts_sum_l2001_200137


namespace joan_video_game_spending_l2001_200156

/-- The cost of the basketball game Joan purchased -/
def basketball_cost : ℚ := 5.2

/-- The cost of the racing game Joan purchased -/
def racing_cost : ℚ := 4.23

/-- The total amount Joan spent on video games -/
def total_spent : ℚ := basketball_cost + racing_cost

/-- Theorem stating that the total amount Joan spent on video games is $9.43 -/
theorem joan_video_game_spending : total_spent = 9.43 := by sorry

end joan_video_game_spending_l2001_200156


namespace binomial_expansion_equal_terms_l2001_200194

theorem binomial_expansion_equal_terms (p q : ℝ) (hp : p > 0) (hq : q > 0) : 
  10 * p^9 * q = 45 * p^8 * q^2 → p + 2*q = 1 → p = 9/13 := by sorry

end binomial_expansion_equal_terms_l2001_200194


namespace matrix_product_AB_l2001_200190

def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, -2; 4, 0]
def B : Matrix (Fin 2) (Fin 1) ℝ := !![5; -1]

theorem matrix_product_AB :
  A * B = !![17; 20] := by sorry

end matrix_product_AB_l2001_200190


namespace special_point_properties_l2001_200199

/-- A point in the second quadrant with coordinate product -10 -/
def special_point : ℝ × ℝ := (-2, 5)

theorem special_point_properties :
  let (x, y) := special_point
  x < 0 ∧ y > 0 ∧ x * y = -10 := by
  sorry

end special_point_properties_l2001_200199


namespace sum_of_tens_and_units_digits_of_8_pow_2003_l2001_200163

/-- The sum of the tens digit and the units digit in the decimal representation of 8^2003 is 2 -/
theorem sum_of_tens_and_units_digits_of_8_pow_2003 : ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ a + b = 2 ∧ 
  (∃ (k : ℕ), 8^2003 = k * 100 + a * 10 + b) := by
  sorry

end sum_of_tens_and_units_digits_of_8_pow_2003_l2001_200163


namespace calculate_total_income_person_total_income_l2001_200182

/-- Calculates a person's total income based on given distributions and remaining amount. -/
theorem calculate_total_income (children_percentage : ℝ) (wife_percentage : ℝ) 
  (orphan_donation_percentage : ℝ) (remaining_amount : ℝ) : ℝ :=
  let total_distributed_percentage := 3 * children_percentage + wife_percentage
  let remaining_percentage := 1 - total_distributed_percentage
  let final_remaining_percentage := remaining_percentage * (1 - orphan_donation_percentage)
  remaining_amount / final_remaining_percentage

/-- Proves that the person's total income is $1,000,000 given the conditions. -/
theorem person_total_income : 
  calculate_total_income 0.2 0.3 0.05 50000 = 1000000 := by
  sorry

end calculate_total_income_person_total_income_l2001_200182


namespace family_egg_count_l2001_200138

/-- Calculates the final number of eggs a family has after using some and chickens laying new ones. -/
def finalEggCount (initialEggs usedEggs chickens eggsPerChicken : ℕ) : ℕ :=
  initialEggs - usedEggs + chickens * eggsPerChicken

/-- Proves that for the given scenario, the family ends up with 11 eggs. -/
theorem family_egg_count : finalEggCount 10 5 2 3 = 11 := by
  sorry

end family_egg_count_l2001_200138
