import Mathlib

namespace smallest_gcd_bc_l1254_125449

theorem smallest_gcd_bc (a b c : ℕ+) 
  (hab : Nat.gcd a b = 360)
  (hac : Nat.gcd a c = 1170)
  (hb : 5 ∣ b)
  (hc : 13 ∣ c) :
  ∃ (k : ℕ+), Nat.gcd b c = k ∧ 
  ∀ (m : ℕ+), Nat.gcd b c ≤ m → k ≤ m :=
by sorry

end smallest_gcd_bc_l1254_125449


namespace sector_perimeter_l1254_125473

theorem sector_perimeter (θ : Real) (r : Real) (h1 : θ = 54) (h2 : r = 20) :
  let l := (θ / 360) * (2 * Real.pi * r)
  l + 2 * r = 6 * Real.pi + 40 := by
  sorry

end sector_perimeter_l1254_125473


namespace consecutive_integers_equation_l1254_125463

theorem consecutive_integers_equation (x y z : ℤ) : 
  (y = x - 1) →
  (z = x - 2) →
  (x > y) →
  (y > z) →
  (2 * x + 3 * y + 3 * z = 5 * y + 11) →
  z = 3 := by
sorry

end consecutive_integers_equation_l1254_125463


namespace coefficient_x4_in_product_l1254_125404

def p (x : ℝ) : ℝ := x^5 - 2*x^4 + 3*x^3 - 4*x^2 + x - 1
def q (x : ℝ) : ℝ := 3*x^4 - 4*x^3 + 5*x^2 - 2*x + 6

theorem coefficient_x4_in_product :
  ∃ (a b c d e f g h i j : ℝ),
    p x * q x = a*x^9 + b*x^8 + c*x^7 + d*x^6 + e*x^5 + (-38)*x^4 + f*x^3 + g*x^2 + h*x + i :=
by sorry

end coefficient_x4_in_product_l1254_125404


namespace max_silver_tokens_l1254_125454

/-- Represents the state of Alex's tokens -/
structure TokenState where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth -/
structure Booth where
  redIn : ℕ
  blueIn : ℕ
  redOut : ℕ
  blueOut : ℕ
  silverOut : ℕ

/-- Checks if an exchange is possible at a given booth -/
def canExchange (state : TokenState) (booth : Booth) : Prop :=
  state.red ≥ booth.redIn ∧ state.blue ≥ booth.blueIn

/-- Performs an exchange at a given booth -/
def exchange (state : TokenState) (booth : Booth) : TokenState :=
  { red := state.red - booth.redIn + booth.redOut,
    blue := state.blue - booth.blueIn + booth.blueOut,
    silver := state.silver + booth.silverOut }

/-- The theorem to be proved -/
theorem max_silver_tokens : ∃ (finalState : TokenState),
  let initialState : TokenState := { red := 90, blue := 65, silver := 0 }
  let booth1 : Booth := { redIn := 3, blueIn := 0, redOut := 0, blueOut := 2, silverOut := 1 }
  let booth2 : Booth := { redIn := 0, blueIn := 4, redOut := 2, blueOut := 0, silverOut := 1 }
  (∀ state, (canExchange state booth1 ∨ canExchange state booth2) → 
    (finalState.silver ≥ state.silver)) ∧
  (¬ canExchange finalState booth1 ∧ ¬ canExchange finalState booth2) ∧
  finalState.silver = 67 :=
sorry

end max_silver_tokens_l1254_125454


namespace alcohol_mixture_concentration_l1254_125444

/-- Represents an alcohol solution with a volume and concentration -/
structure AlcoholSolution where
  volume : ℝ
  concentration : ℝ

/-- Represents a mixture of two alcohol solutions -/
def AlcoholMixture (s1 s2 : AlcoholSolution) (finalConcentration : ℝ) :=
  s1.volume * s1.concentration + s2.volume * s2.concentration = 
    (s1.volume + s2.volume) * finalConcentration

theorem alcohol_mixture_concentration 
  (s1 s2 : AlcoholSolution) (finalConcentration : ℝ) :
  s1.volume = 75 →
  s2.volume = 125 →
  s2.concentration = 0.12 →
  finalConcentration = 0.15 →
  AlcoholMixture s1 s2 finalConcentration →
  s1.concentration = 0.20 := by
sorry

end alcohol_mixture_concentration_l1254_125444


namespace lucy_disproves_tom_l1254_125483

-- Define the visible sides of the cards
def visible_numbers : List ℕ := [2, 4, 5, 7]
def visible_letters : List Char := ['B', 'C', 'D']

-- Define primality
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define consonant
def is_consonant (c : Char) : Prop := c ∈ ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

-- Tom's claim
def toms_claim (n : ℕ) (c : Char) : Prop := is_prime n → is_consonant c

-- Lucy's action
def lucy_flips_5 : Prop := ∃ (c : Char), c ∉ visible_letters ∧ ¬(is_consonant c)

-- Theorem to prove
theorem lucy_disproves_tom : 
  (∀ n ∈ visible_numbers, is_prime n → n ≠ 5 → ∃ c ∈ visible_letters, toms_claim n c) →
  lucy_flips_5 →
  ¬(∀ n c, toms_claim n c) :=
by sorry

end lucy_disproves_tom_l1254_125483


namespace negation_existential_to_universal_l1254_125460

theorem negation_existential_to_universal :
  (¬ ∃ x : ℝ, 2 * x - 3 > 1) ↔ (∀ x : ℝ, 2 * x - 3 ≤ 1) := by
  sorry

end negation_existential_to_universal_l1254_125460


namespace gamma_max_success_ratio_l1254_125410

theorem gamma_max_success_ratio 
  (alpha_day1_score alpha_day1_total : ℕ)
  (alpha_day2_score alpha_day2_total : ℕ)
  (gamma_day1_score gamma_day1_total : ℕ)
  (gamma_day2_score gamma_day2_total : ℕ)
  (h1 : alpha_day1_score = 170)
  (h2 : alpha_day1_total = 280)
  (h3 : alpha_day2_score = 150)
  (h4 : alpha_day2_total = 220)
  (h5 : gamma_day1_total < alpha_day1_total)
  (h6 : gamma_day1_score > 0)
  (h7 : gamma_day2_score > 0)
  (h8 : (gamma_day1_score : ℚ) / gamma_day1_total < (alpha_day1_score : ℚ) / alpha_day1_total)
  (h9 : (gamma_day2_score : ℚ) / gamma_day2_total < (alpha_day2_score : ℚ) / alpha_day2_total)
  (h10 : gamma_day1_total + gamma_day2_total = 500)
  (h11 : (alpha_day1_score + alpha_day2_score : ℚ) / (alpha_day1_total + alpha_day2_total) = 320 / 500) :
  (gamma_day1_score + gamma_day2_score : ℚ) / 500 ≤ 170 / 500 :=
by sorry

end gamma_max_success_ratio_l1254_125410


namespace even_sum_probability_l1254_125482

-- Define the possible outcomes for each spinner
def X : Finset ℕ := {2, 5, 7}
def Y : Finset ℕ := {2, 4, 6}
def Z : Finset ℕ := {1, 2, 3, 4}

-- Define a function to check if a number is even
def isEven (n : ℕ) : Bool := n % 2 = 0

-- Define the probability of getting an even sum
def probEvenSum : ℚ := sorry

-- Theorem statement
theorem even_sum_probability :
  probEvenSum = 1/2 := by sorry

end even_sum_probability_l1254_125482


namespace allan_initial_balloons_l1254_125416

theorem allan_initial_balloons :
  ∀ (allan_initial jake_balloons allan_bought total : ℕ),
    jake_balloons = 5 →
    allan_bought = 2 →
    total = 10 →
    allan_initial + allan_bought + jake_balloons = total →
    allan_initial = 3 := by
  sorry

end allan_initial_balloons_l1254_125416


namespace largest_prime_divisor_of_17_squared_plus_40_squared_l1254_125490

theorem largest_prime_divisor_of_17_squared_plus_40_squared :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (17^2 + 40^2) ∧ ∀ q : ℕ, Nat.Prime q → q ∣ (17^2 + 40^2) → q ≤ p := by
  sorry

end largest_prime_divisor_of_17_squared_plus_40_squared_l1254_125490


namespace factorization_equality_l1254_125491

theorem factorization_equality (a : ℝ) : 2*a^2 + 4*a + 2 = 2*(a + 1)^2 := by
  sorry

end factorization_equality_l1254_125491


namespace problem_solution_l1254_125417

/-- A function representing the relationship between x and y -/
def f (x : ℝ) : ℝ := sorry

/-- The constant of proportionality -/
def k : ℝ := sorry

theorem problem_solution :
  (∀ x : ℝ, f x - 4 = k * (2 * x + 1)) →  -- y-4 is directly proportional to 2x+1
  (f (-1) = 6) →                          -- When x = -1, y = 6
  (∀ x : ℝ, f x = -4 * x + 2) ∧           -- Functional expression
  (f (3/2) = -4)                          -- When y = -4, x = 3/2
  := by sorry

end problem_solution_l1254_125417


namespace problem_1_l1254_125443

theorem problem_1 (x : ℝ) (h : x^2 = 2) : (3*x)^2 - 4*(x^3)^2 = -14 := by
  sorry

end problem_1_l1254_125443


namespace quintic_integer_root_count_l1254_125413

/-- Represents a polynomial of degree 5 with integer coefficients -/
structure QuinticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- The set of possible numbers of integer roots for a quintic polynomial with integer coefficients -/
def PossibleRootCounts : Set ℕ := {0, 1, 2, 3, 5}

/-- Counts the number of integer roots of a quintic polynomial, including multiplicity -/
def countIntegerRoots (p : QuinticPolynomial) : ℕ := sorry

/-- Theorem stating that the number of integer roots of a quintic polynomial with integer coefficients
    can only be 0, 1, 2, 3, or 5 -/
theorem quintic_integer_root_count (p : QuinticPolynomial) :
  countIntegerRoots p ∈ PossibleRootCounts := by sorry

end quintic_integer_root_count_l1254_125413


namespace mobile_phone_purchase_price_l1254_125478

/-- The purchase price of the refrigerator in rupees -/
def refrigerator_price : ℝ := 15000

/-- The loss percentage on the refrigerator sale -/
def refrigerator_loss_percent : ℝ := 0.05

/-- The profit percentage on the mobile phone sale -/
def mobile_profit_percent : ℝ := 0.10

/-- The overall profit in rupees -/
def overall_profit : ℝ := 50

/-- The purchase price of the mobile phone in rupees -/
def mobile_price : ℝ := 8000

theorem mobile_phone_purchase_price :
  ∃ (x : ℝ),
    x = mobile_price ∧
    refrigerator_price * (1 - refrigerator_loss_percent) +
    x * (1 + mobile_profit_percent) =
    refrigerator_price + x + overall_profit :=
by sorry

end mobile_phone_purchase_price_l1254_125478


namespace unique_sums_count_l1254_125438

def bagA : Finset ℕ := {2, 3, 4}
def bagB : Finset ℕ := {3, 4, 5}

def possibleSums : Finset ℕ := (bagA.product bagB).image (fun p => p.1 + p.2)

theorem unique_sums_count : possibleSums.card = 5 := by sorry

end unique_sums_count_l1254_125438


namespace new_players_count_new_players_proof_l1254_125492

theorem new_players_count (returning_players : ℕ) (players_per_group : ℕ) (num_groups : ℕ) : ℕ :=
  let total_players := num_groups * players_per_group
  total_players - returning_players

theorem new_players_proof :
  new_players_count 6 6 9 = 48 := by
  sorry

end new_players_count_new_players_proof_l1254_125492


namespace polynomial_decomposition_l1254_125401

theorem polynomial_decomposition (x : ℝ) :
  x^3 - 2*x^2 + 3*x + 5 = 11 + 7*(x - 2) + 4*(x - 2)^2 + (x - 2)^3 := by
  sorry

end polynomial_decomposition_l1254_125401


namespace problem_solution_l1254_125450

theorem problem_solution :
  ∀ (a b m n : ℝ),
  (m = (a + 4) ^ (1 / (b - 1))) →
  (n = (3 * b - 1) ^ (1 / (a - 2))) →
  ((b - 1) = 2) →
  ((a - 2) = 3) →
  ((m - 2 * n) ^ (1 / 3) = -1) ∧
  (∀ (m' n' : ℝ),
    (m' = Real.sqrt (1 - a) + Real.sqrt (a - 1) + 1) →
    (n' = 25) →
    (Real.sqrt (3 * n' + 6 * m') = 9 ∨ Real.sqrt (3 * n' + 6 * m') = -9)) :=
by sorry

end problem_solution_l1254_125450


namespace point_not_on_graph_l1254_125465

/-- The function f(x) = x^2 / (x + 1) -/
def f (x : ℚ) : ℚ := x^2 / (x + 1)

/-- The point (-1/2, 1/6) -/
def point : ℚ × ℚ := (-1/2, 1/6)

/-- Theorem: The point (-1/2, 1/6) is not on the graph of f(x) = x^2 / (x + 1) -/
theorem point_not_on_graph : f point.1 ≠ point.2 := by sorry

end point_not_on_graph_l1254_125465


namespace inequality_proof_l1254_125418

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : a + b + c = 3) :
  a * b^2 + b * c^2 + c * a^2 ≤ 27/8 := by
  sorry

end inequality_proof_l1254_125418


namespace min_value_theorem_l1254_125468

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hmin : ∀ x, |x + a| + |x - b| ≥ 4) : 
  (a + b = 4) ∧ (∀ a' b' : ℝ, a' > 0 → b' > 0 → (1/4) * a'^2 + (1/9) * b'^2 ≥ 16/13) := by
  sorry

end min_value_theorem_l1254_125468


namespace greatest_possible_award_l1254_125474

/-- The greatest possible individual award in a prize distribution problem --/
theorem greatest_possible_award (total_prize : ℝ) (num_winners : ℕ) (min_award : ℝ)
  (h1 : total_prize = 2000)
  (h2 : num_winners = 50)
  (h3 : min_award = 25)
  (h4 : (3 / 4 : ℝ) * total_prize = (2 / 5 : ℝ) * (num_winners : ℝ) * (greatest_award : ℝ)) :
  greatest_award = 775 := by
  sorry

end greatest_possible_award_l1254_125474


namespace number_difference_l1254_125412

theorem number_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : |x - y| = 3 := by
  sorry

end number_difference_l1254_125412


namespace quartic_comparison_l1254_125459

noncomputable def Q (x : ℝ) : ℝ := x^4 - 2*x^3 + 3*x^2 - 4*x + 5

def sum_of_zeros (f : ℝ → ℝ) : ℝ := 2 -- From Vieta's formula for quartic polynomial

def product_of_zeros (f : ℝ → ℝ) : ℝ := f 0

def sum_of_coefficients (f : ℝ → ℝ) : ℝ := 3 -- 1 - 2 + 3 - 4 + 5

theorem quartic_comparison :
  (sum_of_zeros Q)^2 ≤ Q (-1) ∧
  (sum_of_zeros Q)^2 ≤ product_of_zeros Q ∧
  (sum_of_zeros Q)^2 ≤ sum_of_coefficients Q :=
sorry

end quartic_comparison_l1254_125459


namespace h_k_equality_implies_m_value_l1254_125446

/-- The function h(x) = x^2 - 3x + m -/
def h (x m : ℝ) : ℝ := x^2 - 3*x + m

/-- The function k(x) = x^2 - 3x + 5m -/
def k (x m : ℝ) : ℝ := x^2 - 3*x + 5*m

/-- Theorem stating that if 3h(5) = 2k(5), then m = 10/7 -/
theorem h_k_equality_implies_m_value :
  ∀ m : ℝ, 3 * (h 5 m) = 2 * (k 5 m) → m = 10/7 := by
  sorry

end h_k_equality_implies_m_value_l1254_125446


namespace twenty_first_term_is_4641_l1254_125487

/-- The nth term of the sequence is the sum of n consecutive integers starting from n(n-1)/2 + 1 -/
def sequence_term (n : ℕ) : ℕ :=
  let start := n * (n - 1) / 2 + 1
  (n * (2 * start + n - 1)) / 2

theorem twenty_first_term_is_4641 : sequence_term 21 = 4641 := by sorry

end twenty_first_term_is_4641_l1254_125487


namespace pet_store_count_l1254_125462

/-- Given the ratios of cats to dogs and dogs to birds, and the number of cats,
    prove the number of dogs and birds -/
theorem pet_store_count (cats : ℕ) (dogs : ℕ) (birds : ℕ) : 
  cats = 20 →                   -- There are 20 cats
  5 * cats = 4 * dogs →         -- Ratio of cats to dogs is 4:5
  7 * dogs = 3 * birds →        -- Ratio of dogs to birds is 3:7
  dogs = 25 ∧ birds = 56 :=     -- Prove dogs = 25 and birds = 56
by sorry

end pet_store_count_l1254_125462


namespace complex_equality_implication_l1254_125485

theorem complex_equality_implication (x y : ℝ) : 
  (Complex.I * x + 2 = y - Complex.I) → (x - y = -3) := by
  sorry

end complex_equality_implication_l1254_125485


namespace combined_cube_volume_l1254_125497

theorem combined_cube_volume : 
  let lily_cubes := 4
  let lily_side_length := 3
  let mark_cubes := 3
  let mark_side_length := 4
  let zoe_cubes := 2
  let zoe_side_length := 5
  lily_cubes * lily_side_length^3 + 
  mark_cubes * mark_side_length^3 + 
  zoe_cubes * zoe_side_length^3 = 550 := by
sorry

end combined_cube_volume_l1254_125497


namespace volume_of_specific_box_l1254_125498

/-- The volume of a rectangular box -/
def box_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a box with dimensions 20 cm, 15 cm, and 10 cm is 3000 cm³ -/
theorem volume_of_specific_box : box_volume 20 15 10 = 3000 := by
  sorry

end volume_of_specific_box_l1254_125498


namespace solutions_of_absolute_value_equation_l1254_125489

theorem solutions_of_absolute_value_equation :
  {x : ℝ | |x - 2| + |x - 3| = 1} = Set.Icc 2 3 := by sorry

end solutions_of_absolute_value_equation_l1254_125489


namespace set_intersection_problem_l1254_125456

theorem set_intersection_problem (M N : Set ℕ) : 
  M = {1, 2, 3} → N = {2, 3, 4} → M ∩ N = {2, 3} := by
  sorry

end set_intersection_problem_l1254_125456


namespace cookie_bags_count_l1254_125466

theorem cookie_bags_count (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 703) (h2 : cookies_per_bag = 19) :
  total_cookies / cookies_per_bag = 37 := by
  sorry

end cookie_bags_count_l1254_125466


namespace square_difference_l1254_125452

theorem square_difference (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3 * x + y = 18) : 
  x^2 - y^2 = -72 := by
sorry

end square_difference_l1254_125452


namespace sum_reciprocal_inequality_l1254_125422

theorem sum_reciprocal_inequality (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c ≤ 3) : 
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) ≥ 3 / 2 := by
sorry

end sum_reciprocal_inequality_l1254_125422


namespace single_draw_probability_triple_draw_probability_l1254_125428

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Black

/-- Represents the outcome of drawing a single ball -/
def SingleDrawOutcome := BallColor

/-- Represents the outcome of drawing three balls -/
def TripleDrawOutcome := (BallColor × BallColor × BallColor)

/-- The total number of balls in the box -/
def totalBalls : Nat := 5 + 2

/-- The number of white balls in the box -/
def whiteBalls : Nat := 5

/-- The number of black balls in the box -/
def blackBalls : Nat := 2

/-- A function that simulates drawing a single ball -/
noncomputable def simulateSingleDraw : SingleDrawOutcome := sorry

/-- A function that simulates drawing three balls -/
noncomputable def simulateTripleDraw : TripleDrawOutcome := sorry

/-- Checks if a single draw outcome is favorable (white ball) -/
def isFavorableSingleDraw (outcome : SingleDrawOutcome) : Bool :=
  match outcome with
  | BallColor.White => true
  | BallColor.Black => false

/-- Checks if a triple draw outcome is favorable (all white balls) -/
def isFavorableTripleDraw (outcome : TripleDrawOutcome) : Bool :=
  match outcome with
  | (BallColor.White, BallColor.White, BallColor.White) => true
  | _ => false

/-- Theorem: The probability of drawing a white ball is equal to the ratio of favorable outcomes to total outcomes in a random simulation -/
theorem single_draw_probability (n : Nat) (m : Nat) :
  m ≤ n →
  (m : ℚ) / n = whiteBalls / totalBalls :=
sorry

/-- Theorem: The probability of drawing three white balls is equal to the ratio of favorable outcomes to total outcomes in a random simulation -/
theorem triple_draw_probability (n : Nat) (m : Nat) :
  m ≤ n →
  (m : ℚ) / n = (whiteBalls / totalBalls) * ((whiteBalls - 1) / (totalBalls - 1)) * ((whiteBalls - 2) / (totalBalls - 2)) :=
sorry

end single_draw_probability_triple_draw_probability_l1254_125428


namespace metallic_sheet_length_l1254_125440

/-- Given a rectangular metallic sheet from which squares are cut at corners to form a box,
    this theorem proves the length of the original sheet. -/
theorem metallic_sheet_length
  (square_side : ℝ)
  (sheet_width : ℝ)
  (box_volume : ℝ)
  (h_square : square_side = 6)
  (h_width : sheet_width = 36)
  (h_volume : box_volume = 5184)
  (h_box : box_volume = (sheet_length - 2 * square_side) * (sheet_width - 2 * square_side) * square_side) :
  sheet_length = 48 :=
by
  sorry


end metallic_sheet_length_l1254_125440


namespace no_prime_multiple_chain_l1254_125495

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def primes_1_to_12 : Set ℕ := {n : ℕ | is_prime n ∧ n ≤ 12}

theorem no_prime_multiple_chain :
  ∀ a b c : ℕ, a ∈ primes_1_to_12 → b ∈ primes_1_to_12 → c ∈ primes_1_to_12 →
  a ≠ b → b ≠ c → a ≠ c →
  ¬(a ∣ b ∧ b ∣ c) :=
sorry

end no_prime_multiple_chain_l1254_125495


namespace yoongi_has_smallest_points_l1254_125467

def jungkook_points : ℕ := 9
def yoongi_points : ℕ := 4
def yuna_points : ℕ := 5

theorem yoongi_has_smallest_points : 
  yoongi_points ≤ jungkook_points ∧ yoongi_points ≤ yuna_points :=
by sorry

end yoongi_has_smallest_points_l1254_125467


namespace complex_equation_proof_l1254_125445

theorem complex_equation_proof (x : ℂ) (h : x - 1/x = 3*I) : x^12 - 1/x^12 = 103682 := by
  sorry

end complex_equation_proof_l1254_125445


namespace cubic_root_sum_squares_reciprocal_l1254_125475

theorem cubic_root_sum_squares_reciprocal (α β γ : ℂ) : 
  α^3 - 6*α^2 + 11*α - 6 = 0 →
  β^3 - 6*β^2 + 11*β - 6 = 0 →
  γ^3 - 6*γ^2 + 11*γ - 6 = 0 →
  α ≠ β → β ≠ γ → γ ≠ α →
  1/α^2 + 1/β^2 + 1/γ^2 = 49/36 := by
sorry

end cubic_root_sum_squares_reciprocal_l1254_125475


namespace reunion_attendance_l1254_125477

/-- The number of people attending a family reunion. -/
def n : ℕ := sorry

/-- The age of the youngest person at the reunion. -/
def youngest_age : ℕ := sorry

/-- The age of the oldest person at the reunion. -/
def oldest_age : ℕ := sorry

/-- The sum of ages of all people at the reunion. -/
def total_age_sum : ℕ := sorry

/-- The average age of members excluding the oldest person is 18 years old. -/
axiom avg_without_oldest : (total_age_sum - oldest_age) / (n - 1) = 18

/-- The average age of members excluding the youngest person is 20 years old. -/
axiom avg_without_youngest : (total_age_sum - youngest_age) / (n - 1) = 20

/-- The age difference between the oldest and youngest person is 40 years. -/
axiom age_difference : oldest_age - youngest_age = 40

/-- The number of people attending the reunion is 21. -/
theorem reunion_attendance : n = 21 := by sorry

end reunion_attendance_l1254_125477


namespace symmetric_points_difference_l1254_125414

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = -x₁ ∧ y₂ = -y₁

/-- Given A(a,1) and B(5,b) are symmetric with respect to the origin, prove a - b = -4 -/
theorem symmetric_points_difference (a b : ℝ) 
  (h : symmetric_wrt_origin a 1 5 b) : a - b = -4 := by
  sorry

end symmetric_points_difference_l1254_125414


namespace inequality_implies_max_a_l1254_125436

theorem inequality_implies_max_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 1 ≥ a * |x - 1|) → a ≤ -2 :=
by sorry

end inequality_implies_max_a_l1254_125436


namespace equations_solutions_l1254_125457

-- Define the equations
def equation1 (x : ℝ) : Prop :=
  (x - 3) / (x - 2) + 1 = 3 / (2 - x)

def equation2 (x : ℝ) : Prop :=
  (x - 2) / (x + 2) - (x + 2) / (x - 2) = 16 / (x^2 - 4)

-- Theorem statement
theorem equations_solutions :
  equation1 1 ∧ equation2 (-4) :=
by sorry

end equations_solutions_l1254_125457


namespace equation_solution_inequality_solution_l1254_125434

-- Definition of permutation
def A (n : ℕ) (m : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - m)

-- Theorem for the equation
theorem equation_solution :
  ∃! x : ℕ, 3 * A 8 x = 4 * A 9 (x - 1) ∧ x = 6 :=
sorry

-- Theorem for the inequality
theorem inequality_solution :
  ∀ x : ℕ, x ≥ 4 ↔ A (x - 2) 2 + x ≥ 2 :=
sorry

end equation_solution_inequality_solution_l1254_125434


namespace x_range_l1254_125403

-- Define the inequality condition
def inequality_condition (x m : ℝ) : Prop :=
  2 * x - 1 > m * (x^2 - 1)

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  |m| ≤ 2

-- Theorem statement
theorem x_range :
  (∀ x m : ℝ, m_range m → inequality_condition x m) →
  ∃ a b : ℝ, a = (Real.sqrt 7 - 1) / 2 ∧ b = (Real.sqrt 3 + 1) / 2 ∧
    ∀ x : ℝ, (∀ m : ℝ, m_range m → inequality_condition x m) → a < x ∧ x < b :=
sorry

end x_range_l1254_125403


namespace invitations_per_package_l1254_125435

theorem invitations_per_package 
  (total_packs : ℕ) 
  (total_invitations : ℕ) 
  (h1 : total_packs = 5)
  (h2 : total_invitations = 45) :
  total_invitations / total_packs = 9 :=
by sorry

end invitations_per_package_l1254_125435


namespace equal_sum_parallel_segments_l1254_125427

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB : ℝ) (BC : ℝ) (CA : ℝ)

/-- Points on the sides of the triangle -/
structure Points (t : Triangle) :=
  (D : ℝ) (E : ℝ) (F : ℝ) (G : ℝ)
  (h₁ : 0 ≤ D ∧ D ≤ E ∧ E ≤ t.AB)
  (h₂ : 0 ≤ F ∧ F ≤ G ∧ G ≤ t.CA)

/-- Perimeter of triangle ADF -/
def perim_ADF (t : Triangle) (p : Points t) : ℝ :=
  p.D + (p.F - p.D) + p.F

/-- Perimeter of trapezoid DEFG -/
def perim_DEFG (t : Triangle) (p : Points t) : ℝ :=
  (p.E - p.D) + (p.G - p.F) + p.G + (p.F - p.D)

/-- Perimeter of trapezoid EBCG -/
def perim_EBCG (t : Triangle) (p : Points t) : ℝ :=
  (t.AB - p.E) + t.BC + (t.CA - p.G) + (p.G - p.F)

theorem equal_sum_parallel_segments (t : Triangle) (p : Points t) 
    (h_sides : t.AB = 2 ∧ t.BC = 3 ∧ t.CA = 4)
    (h_parallel : (p.E - p.D) / t.BC = (p.G - p.F) / t.BC)
    (h_perims : perim_ADF t p = perim_DEFG t p ∧ perim_DEFG t p = perim_EBCG t p) :
    (p.E - p.D) + (p.G - p.F) = 9/2 := by
  sorry

end equal_sum_parallel_segments_l1254_125427


namespace calculation_proof_expression_equivalence_l1254_125426

-- First part of the problem
theorem calculation_proof : 28 + 72 + (9 - 8) = 172 := by sorry

-- Second part of the problem
def original_expression : ℚ := 4600 / 23 - 19 * 10

def reordered_expression : ℚ := (4600 / 23) - (19 * 10)

theorem expression_equivalence : original_expression = reordered_expression := by sorry

end calculation_proof_expression_equivalence_l1254_125426


namespace stream_speed_l1254_125480

/-- Proves that the speed of a stream is 20 km/h given the conditions of the rowing problem -/
theorem stream_speed (boat_speed : ℝ) (upstream_time downstream_time : ℝ) 
  (h1 : boat_speed = 60)
  (h2 : upstream_time = 2 * downstream_time)
  (h3 : upstream_time > 0)
  (h4 : downstream_time > 0) :
  let stream_speed := (boat_speed - boat_speed * downstream_time / upstream_time) / 2
  stream_speed = 20 := by
  sorry


end stream_speed_l1254_125480


namespace marble_problem_l1254_125421

theorem marble_problem (r b : ℕ) : 
  ((r - 3 : ℚ) / (r + b - 3) = 1 / 10) →
  ((r : ℚ) / (r + b - 3) = 1 / 4) →
  r + b = 13 := by
  sorry

end marble_problem_l1254_125421


namespace min_value_of_y_l1254_125481

theorem min_value_of_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 4/b = 2) :
  ∀ y : ℝ, y = 4*a + b → y ≥ 8 :=
by sorry

end min_value_of_y_l1254_125481


namespace rectangular_plot_longer_side_l1254_125455

theorem rectangular_plot_longer_side 
  (width : ℝ) 
  (pole_distance : ℝ) 
  (num_poles : ℕ) 
  (h1 : width = 30)
  (h2 : pole_distance = 5)
  (h3 : num_poles = 32) :
  let perimeter := pole_distance * (num_poles - 1 : ℝ)
  let length := (perimeter / 2) - width
  length = 47.5 := by
sorry

end rectangular_plot_longer_side_l1254_125455


namespace linear_equation_implies_a_value_l1254_125488

/-- Given that (a-2)x^(|a|-1) + 3y = 1 is a linear equation in x and y, prove that a = -2 --/
theorem linear_equation_implies_a_value (a : ℝ) : 
  (∀ x y : ℝ, ∃ k m : ℝ, (a - 2) * x^(|a| - 1) + 3 * y = k * x + m * y + 1) → 
  a = -2 :=
by sorry

end linear_equation_implies_a_value_l1254_125488


namespace candy_expenditure_l1254_125470

theorem candy_expenditure (initial : ℕ) (oranges apples left : ℕ) 
  (h1 : initial = 95)
  (h2 : oranges = 14)
  (h3 : apples = 25)
  (h4 : left = 50) :
  initial - (oranges + apples) - left = 6 := by
  sorry

end candy_expenditure_l1254_125470


namespace hunter_score_theorem_l1254_125408

def math_test_scores (grant_score john_score hunter_score : ℕ) : Prop :=
  (grant_score = 100) ∧
  (grant_score = john_score + 10) ∧
  (john_score = 2 * hunter_score)

theorem hunter_score_theorem :
  ∀ grant_score john_score hunter_score : ℕ,
  math_test_scores grant_score john_score hunter_score →
  hunter_score = 45 :=
by
  sorry

end hunter_score_theorem_l1254_125408


namespace percentage_problem_l1254_125400

theorem percentage_problem (x p : ℝ) (h1 : 0.25 * x = (p/100) * 500 - 5) (h2 : x = 180) : p = 10 := by
  sorry

end percentage_problem_l1254_125400


namespace derivative_x_over_one_minus_cos_l1254_125431

/-- The derivative of x / (1 - cos x) is (1 - cos x - x * sin x) / (1 - cos x)^2 -/
theorem derivative_x_over_one_minus_cos (x : ℝ) :
  deriv (fun x => x / (1 - Real.cos x)) x = (1 - Real.cos x - x * Real.sin x) / (1 - Real.cos x)^2 :=
sorry

end derivative_x_over_one_minus_cos_l1254_125431


namespace soup_cans_bought_soup_cans_received_johns_soup_cans_l1254_125420

theorem soup_cans_bought (normal_price : ℝ) (total_paid : ℝ) : ℝ :=
  total_paid / normal_price

theorem soup_cans_received (cans_bought : ℝ) : ℝ :=
  2 * cans_bought

theorem johns_soup_cans (normal_price : ℝ) (total_paid : ℝ) : 
  soup_cans_received (soup_cans_bought normal_price total_paid) = 30 :=
by
  -- Assuming normal_price = 0.60 and total_paid = 9
  have h1 : normal_price = 0.60 := by sorry
  have h2 : total_paid = 9 := by sorry
  
  -- Calculate the number of cans bought
  have cans_bought : ℝ := soup_cans_bought normal_price total_paid
  
  -- Calculate the total number of cans received
  have total_cans : ℝ := soup_cans_received cans_bought
  
  -- Prove that the total number of cans is 30
  sorry

end soup_cans_bought_soup_cans_received_johns_soup_cans_l1254_125420


namespace beetles_eaten_per_day_l1254_125402

/-- The number of beetles eaten by one bird per day -/
def beetles_per_bird : ℕ := 12

/-- The number of birds eaten by one snake per day -/
def birds_per_snake : ℕ := 3

/-- The number of snakes eaten by one jaguar per day -/
def snakes_per_jaguar : ℕ := 5

/-- The number of jaguars in the forest -/
def jaguars_in_forest : ℕ := 6

/-- The total number of beetles eaten per day in the forest -/
def total_beetles_eaten : ℕ := 
  jaguars_in_forest * snakes_per_jaguar * birds_per_snake * beetles_per_bird

theorem beetles_eaten_per_day :
  total_beetles_eaten = 1080 := by
  sorry

end beetles_eaten_per_day_l1254_125402


namespace max_four_digit_divisible_by_36_11_l1254_125464

def digit_reverse (n : Nat) : Nat :=
  -- Implementation of digit reversal (not provided)
  sorry

theorem max_four_digit_divisible_by_36_11 :
  ∃ (m : Nat),
    1000 ≤ m ∧ m ≤ 9999 ∧
    m % 36 = 0 ∧
    (digit_reverse m) % 36 = 0 ∧
    m % 11 = 0 ∧
    ∀ (k : Nat), 1000 ≤ k ∧ k ≤ 9999 ∧
      k % 36 = 0 ∧ (digit_reverse k) % 36 = 0 ∧ k % 11 = 0 →
      k ≤ m ∧
    m = 9504 :=
by
  sorry

end max_four_digit_divisible_by_36_11_l1254_125464


namespace train_crossing_time_l1254_125442

/-- The time taken for a train to cross a platform of equal length -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 1500 →
  train_speed_kmh = 180 →
  (2 * train_length) / (train_speed_kmh * 1000 / 3600) = 60 := by
  sorry

#check train_crossing_time

end train_crossing_time_l1254_125442


namespace sixth_root_of_24414062515625_l1254_125471

theorem sixth_root_of_24414062515625 : 
  (24414062515625 : ℝ) ^ (1/6 : ℝ) = 51 := by sorry

end sixth_root_of_24414062515625_l1254_125471


namespace marathon_finishers_l1254_125493

/-- Proves that 563 people finished the marathon given the conditions --/
theorem marathon_finishers :
  ∀ (finished : ℕ),
  (finished + (finished + 124) = 1250) →
  finished = 563 := by
  sorry

end marathon_finishers_l1254_125493


namespace oven_temperature_l1254_125411

theorem oven_temperature (required_temp increase_needed : ℕ) 
  (h1 : required_temp = 546)
  (h2 : increase_needed = 396) :
  required_temp - increase_needed = 150 := by
sorry

end oven_temperature_l1254_125411


namespace intersection_A_complement_B_l1254_125407

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x : ℝ | x < -1 ∨ 4 < x}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end intersection_A_complement_B_l1254_125407


namespace total_chips_calculation_l1254_125494

/-- The total number of chips Viviana and Susana have together -/
def total_chips (viviana_chocolate viviana_vanilla susana_chocolate susana_vanilla : ℕ) : ℕ :=
  viviana_chocolate + viviana_vanilla + susana_chocolate + susana_vanilla

/-- Theorem stating the total number of chips Viviana and Susana have together -/
theorem total_chips_calculation :
  ∀ (viviana_chocolate viviana_vanilla susana_chocolate susana_vanilla : ℕ),
  viviana_chocolate = susana_chocolate + 5 →
  susana_vanilla = (3 * viviana_vanilla) / 4 →
  viviana_vanilla = 20 →
  susana_chocolate = 25 →
  total_chips viviana_chocolate viviana_vanilla susana_chocolate susana_vanilla = 90 :=
by sorry

end total_chips_calculation_l1254_125494


namespace trigonometric_identity_l1254_125469

theorem trigonometric_identity (A B C : Real) 
  (h : A + B + C = Real.pi) : 
  (Real.sin A + Real.sin B - Real.sin C) / (Real.sin A + Real.sin B + Real.sin C) = 
  Real.tan (A/2) * Real.tan (B/2) := by
  sorry

end trigonometric_identity_l1254_125469


namespace fourth_person_height_l1254_125458

theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℕ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- heights in increasing order
  h₂ = h₁ + 2 →                 -- difference between 1st and 2nd
  h₃ = h₂ + 2 →                 -- difference between 2nd and 3rd
  h₄ = h₃ + 6 →                 -- difference between 3rd and 4th
  (h₁ + h₂ + h₃ + h₄) / 4 = 77  -- average height
  → h₄ = 83 := by
sorry

end fourth_person_height_l1254_125458


namespace angle_C_is_pi_over_three_sum_a_b_range_l1254_125424

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the condition given in the problem
def satisfiesCondition (t : Triangle) : Prop :=
  (t.a + t.c) * (Real.sin t.A - Real.sin t.C) = Real.sin t.B * (t.a - t.b)

-- Theorem for part I
theorem angle_C_is_pi_over_three (t : Triangle) 
  (h : satisfiesCondition t) : t.C = π / 3 := by sorry

-- Theorem for part II
theorem sum_a_b_range (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : t.c = 2) : 
  2 < t.a + t.b ∧ t.a + t.b ≤ 4 := by sorry

end angle_C_is_pi_over_three_sum_a_b_range_l1254_125424


namespace sqrt_equation_solution_l1254_125486

theorem sqrt_equation_solution (x : ℝ) :
  x > 9 →
  (Real.sqrt (x - 5 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 5 * Real.sqrt (x - 9)) - 3) ↔
  x ≥ 25 / 2 :=
by sorry

end sqrt_equation_solution_l1254_125486


namespace valid_param_iff_l1254_125419

/-- A structure representing a vector parameterization of a line -/
structure VectorParam where
  x₀ : ℝ
  y₀ : ℝ
  a : ℝ
  b : ℝ

/-- The line equation y = 2x + 6 -/
def line_equation (x y : ℝ) : Prop := y = 2 * x + 6

/-- Predicate to check if a vector parameterization is valid for the line y = 2x + 6 -/
def is_valid_param (p : VectorParam) : Prop :=
  line_equation p.x₀ p.y₀ ∧ p.b = 2 * p.a

/-- Theorem stating the condition for a valid vector parameterization -/
theorem valid_param_iff (p : VectorParam) :
  is_valid_param p ↔
    (∀ t : ℝ, line_equation (p.x₀ + t * p.a) (p.y₀ + t * p.b)) :=
by sorry

end valid_param_iff_l1254_125419


namespace peaches_after_seven_days_l1254_125439

def peaches_after_days (initial_total : ℕ) (initial_ripe : ℕ) (days : ℕ) : ℕ × ℕ :=
  sorry

theorem peaches_after_seven_days :
  let initial_total := 18
  let initial_ripe := 4
  let ripen_pattern (d : ℕ) := d + 1
  let eat_pattern (d : ℕ) := d
  let (ripe, unripe) := peaches_after_days initial_total initial_ripe 7
  ripe = 0 ∧ unripe = 0 :=
sorry

end peaches_after_seven_days_l1254_125439


namespace range_of_a_l1254_125461

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + Real.exp (-x) - a

def range_subset (f : ℝ → ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x a ≥ 0

theorem range_of_a (a : ℝ) :
  (range_subset f a) ↔ a ≤ 2 :=
sorry

end range_of_a_l1254_125461


namespace alternatingArrangements_4_3_l1254_125409

/-- The number of ways to arrange 4 men and 3 women in a row, such that no two men or two women are adjacent -/
def alternatingArrangements (numMen : Nat) (numWomen : Nat) : Nat :=
  Nat.factorial numMen * 
  (Nat.choose (numMen + 1) numWomen) * 
  Nat.factorial numWomen

/-- Theorem stating that the number of alternating arrangements of 4 men and 3 women is 1440 -/
theorem alternatingArrangements_4_3 : 
  alternatingArrangements 4 3 = 1440 := by
  sorry

#eval alternatingArrangements 4 3

end alternatingArrangements_4_3_l1254_125409


namespace odd_factors_of_450_l1254_125447

/-- The number of odd factors of a natural number n -/
def num_odd_factors (n : ℕ) : ℕ := sorry

/-- 450 has exactly 9 odd factors -/
theorem odd_factors_of_450 : num_odd_factors 450 = 9 := by sorry

end odd_factors_of_450_l1254_125447


namespace truth_values_of_p_and_q_l1254_125499

theorem truth_values_of_p_and_q (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬(¬p ∨ q)) : 
  p ∧ ¬q := by
  sorry

end truth_values_of_p_and_q_l1254_125499


namespace school_survey_l1254_125479

theorem school_survey (total_students : ℕ) (sample_size : ℕ) (girl_boy_diff : ℕ) :
  total_students = 1600 →
  sample_size = 200 →
  girl_boy_diff = 20 →
  (sample_size - girl_boy_diff) / 2 / sample_size * total_students = 720 :=
by sorry

end school_survey_l1254_125479


namespace first_negative_term_position_l1254_125425

def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem first_negative_term_position
  (a₁ : ℤ)
  (d : ℤ)
  (h₁ : a₁ = 1031)
  (h₂ : d = -3) :
  (∀ k < 345, arithmeticSequence a₁ d k ≥ 0) ∧
  arithmeticSequence a₁ d 345 < 0 :=
sorry

end first_negative_term_position_l1254_125425


namespace new_ratio_after_refill_l1254_125441

def initial_ratio_a : ℚ := 7
def initial_ratio_b : ℚ := 5
def initial_volume_a : ℚ := 21
def volume_drawn : ℚ := 9

theorem new_ratio_after_refill :
  let total_volume := initial_volume_a * (initial_ratio_a + initial_ratio_b) / initial_ratio_a
  let removed_a := volume_drawn * initial_ratio_a / (initial_ratio_a + initial_ratio_b)
  let removed_b := volume_drawn * initial_ratio_b / (initial_ratio_a + initial_ratio_b)
  let remaining_a := initial_volume_a - removed_a
  let remaining_b := total_volume - initial_volume_a - removed_b
  let new_b := remaining_b + volume_drawn
  (remaining_a : ℚ) / new_b = 21 / 27 :=
sorry

end new_ratio_after_refill_l1254_125441


namespace image_of_one_two_l1254_125405

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (2 * p.1 - p.2, p.1 - 2 * p.2)

theorem image_of_one_two :
  f (1, 2) = (0, -3) := by sorry

end image_of_one_two_l1254_125405


namespace treasure_in_fourth_bag_l1254_125415

/-- Given four bags A, B, C, and D, prove that D is the heaviest bag. -/
theorem treasure_in_fourth_bag (A B C D : ℝ) 
  (h1 : A + B < C)
  (h2 : A + C = D)
  (h3 : A + D > B + C) :
  D > A ∧ D > B ∧ D > C := by
  sorry

end treasure_in_fourth_bag_l1254_125415


namespace darcys_walking_speed_l1254_125429

/-- Proves that Darcy's walking speed is 3 miles per hour given the problem conditions -/
theorem darcys_walking_speed 
  (distance_to_work : ℝ) 
  (train_speed : ℝ) 
  (additional_train_time : ℝ) 
  (time_difference : ℝ) 
  (h1 : distance_to_work = 1.5)
  (h2 : train_speed = 20)
  (h3 : additional_train_time = 23.5 / 60)
  (h4 : time_difference = 2 / 60)
  (h5 : distance_to_work / train_speed + additional_train_time + time_difference = distance_to_work / 3) :
  3 = 3 := by
  sorry

#check darcys_walking_speed

end darcys_walking_speed_l1254_125429


namespace households_with_car_l1254_125472

theorem households_with_car (total : ℕ) (neither : ℕ) (both : ℕ) (bike_only : ℕ) 
  (h1 : total = 90)
  (h2 : neither = 11)
  (h3 : both = 14)
  (h4 : bike_only = 35) :
  ∃ (car : ℕ), car = 44 ∧ 
    car + bike_only + both + neither = total ∧
    car + bike_only + neither = total - both :=
by
  sorry

#check households_with_car

end households_with_car_l1254_125472


namespace tablet_savings_l1254_125448

/-- The savings when buying a tablet in cash versus installment -/
theorem tablet_savings : 
  let cash_price : ℕ := 450
  let down_payment : ℕ := 100
  let first_four_months : ℕ := 4 * 40
  let next_four_months : ℕ := 4 * 35
  let last_four_months : ℕ := 4 * 30
  let total_installment : ℕ := down_payment + first_four_months + next_four_months + last_four_months
  total_installment - cash_price = 70 := by
  sorry

end tablet_savings_l1254_125448


namespace count_divisible_by_11_eq_36_l1254_125432

/-- The number obtained by concatenating integers from 1 to n -/
def a (n : ℕ) : ℕ := sorry

/-- The count of k values in [1, 200] for which a_k is divisible by 11 -/
def count_divisible_by_11 : ℕ := sorry

/-- Theorem stating that the count of k values in [1, 200] for which a_k is divisible by 11 is 36 -/
theorem count_divisible_by_11_eq_36 : count_divisible_by_11 = 36 := by sorry

end count_divisible_by_11_eq_36_l1254_125432


namespace complement_A_intersect_B_l1254_125406

def I : Finset Nat := {1,2,3,4,5,6}
def A : Finset Nat := {1,3,5}
def B : Finset Nat := {2,3,6}

theorem complement_A_intersect_B : 
  (I \ A) ∩ B = {2,6} := by sorry

end complement_A_intersect_B_l1254_125406


namespace rectangle_diagonal_l1254_125451

theorem rectangle_diagonal (l w : ℝ) (h1 : l + w = 15) (h2 : l = 2 * w) :
  Real.sqrt (l^2 + w^2) = Real.sqrt 125 := by
  sorry

end rectangle_diagonal_l1254_125451


namespace dinosaur_weight_theorem_l1254_125437

def regular_dinosaur_weight : ℕ := 800
def number_of_regular_dinosaurs : ℕ := 5
def barney_weight_difference : ℕ := 1500

def total_weight : ℕ :=
  (regular_dinosaur_weight * number_of_regular_dinosaurs) +
  (regular_dinosaur_weight * number_of_regular_dinosaurs + barney_weight_difference)

theorem dinosaur_weight_theorem :
  total_weight = 9500 :=
by sorry

end dinosaur_weight_theorem_l1254_125437


namespace function_not_satisfying_condition_l1254_125484

theorem function_not_satisfying_condition :
  ∃ f : ℝ → ℝ, (∀ x, f x = x + 1) ∧ (∃ x, f (2 * x) ≠ 2 * f x) := by
  sorry

end function_not_satisfying_condition_l1254_125484


namespace gcd_power_minus_identity_gcd_power_minus_identity_general_l1254_125423

theorem gcd_power_minus_identity (a : ℕ) (h : a ≥ 2) : 
  13530 ∣ a^41 - a :=
sorry

/- More general version for any natural number n -/
theorem gcd_power_minus_identity_general (n : ℕ) (a : ℕ) (h : a ≥ 2) : 
  ∃ k : ℕ, k ∣ a^n - a :=
sorry

end gcd_power_minus_identity_gcd_power_minus_identity_general_l1254_125423


namespace bonus_remainder_l1254_125430

theorem bonus_remainder (P : ℕ) (h : P % 5 = 2) : (3 * P) % 5 = 1 := by
  sorry

end bonus_remainder_l1254_125430


namespace carla_fish_count_l1254_125476

/-- Given that Carla, Kyle, and Tasha caught a total of 36 fish, Kyle caught 14 fish,
    and Kyle and Tasha caught the same number of fish, prove that Carla caught 8 fish. -/
theorem carla_fish_count (total : ℕ) (kyle_fish : ℕ) (h1 : total = 36) (h2 : kyle_fish = 14)
    (h3 : ∃ (tasha_fish : ℕ), tasha_fish = kyle_fish ∧ total = kyle_fish + tasha_fish + (total - kyle_fish - tasha_fish)) :
  total - kyle_fish - kyle_fish = 8 := by
  sorry

end carla_fish_count_l1254_125476


namespace five_spheres_max_regions_l1254_125453

/-- The maximum number of regions into which n spheres can divide three-dimensional space -/
def max_regions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => max_regions n + 2 + n + n * (n + 1) / 2

/-- The maximum number of regions into which five spheres can divide three-dimensional space is 47 -/
theorem five_spheres_max_regions :
  max_regions 5 = 47 := by sorry

end five_spheres_max_regions_l1254_125453


namespace square_diagonal_l1254_125433

theorem square_diagonal (A : ℝ) (s : ℝ) (d : ℝ) (h1 : A = 9/16) (h2 : A = s^2) (h3 : d = s * Real.sqrt 2) :
  d = 3/4 * Real.sqrt 2 := by
  sorry

end square_diagonal_l1254_125433


namespace spherical_to_rectangular_conversion_l1254_125496

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 3
  let θ : ℝ := 3 * π / 2
  let φ : ℝ := π / 3
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry


end spherical_to_rectangular_conversion_l1254_125496
