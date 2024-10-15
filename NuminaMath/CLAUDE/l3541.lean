import Mathlib

namespace NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l3541_354161

theorem binomial_coefficient_n_minus_two (n : ℕ) (h : n > 3) :
  Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l3541_354161


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3541_354192

theorem quadratic_no_real_roots :
  ∀ (x : ℝ), x^2 + 2*x + 4 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3541_354192


namespace NUMINAMATH_CALUDE_internal_borders_length_l3541_354169

/-- Represents a square garden bed with integer side length -/
structure SquareBed where
  side : ℕ

/-- Represents a rectangular garden divided into square beds -/
structure Garden where
  width : ℕ
  height : ℕ
  beds : List SquareBed

/-- Calculates the total area of the garden -/
def Garden.area (g : Garden) : ℕ := g.width * g.height

/-- Calculates the total area covered by the beds -/
def Garden.bedArea (g : Garden) : ℕ := g.beds.map (fun b => b.side * b.side) |>.sum

/-- Calculates the perimeter of the garden -/
def Garden.perimeter (g : Garden) : ℕ := 2 * (g.width + g.height)

/-- Calculates the sum of perimeters of all beds -/
def Garden.bedPerimeters (g : Garden) : ℕ := g.beds.map (fun b => 4 * b.side) |>.sum

/-- Theorem stating the length of internal borders in a specific garden configuration -/
theorem internal_borders_length (g : Garden) : 
  g.width = 6 ∧ 
  g.height = 7 ∧ 
  g.beds.length = 5 ∧ 
  g.area = g.bedArea →
  (g.bedPerimeters - g.perimeter) / 2 = 15 := by
  sorry


end NUMINAMATH_CALUDE_internal_borders_length_l3541_354169


namespace NUMINAMATH_CALUDE_train_length_l3541_354158

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (crossing_time : ℝ) : 
  speed_kmh = 144 → crossing_time = 5 → 
  speed_kmh * (1000 / 3600) * crossing_time = 200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3541_354158


namespace NUMINAMATH_CALUDE_solution_is_816_div_5_l3541_354162

/-- The function g(y) = ∛(30y + ∛(30y + 17)) is increasing --/
axiom g_increasing (y : ℝ) : 
  Monotone (fun y => Real.rpow (30 * y + Real.rpow (30 * y + 17) (1/3 : ℝ)) (1/3 : ℝ))

/-- The equation ∛(30y + ∛(30y + 17)) = 17 has a unique solution --/
axiom unique_solution : ∃! y : ℝ, Real.rpow (30 * y + Real.rpow (30 * y + 17) (1/3 : ℝ)) (1/3 : ℝ) = 17

theorem solution_is_816_div_5 : 
  ∃! y : ℝ, Real.rpow (30 * y + Real.rpow (30 * y + 17) (1/3 : ℝ)) (1/3 : ℝ) = 17 ∧ y = 816 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_816_div_5_l3541_354162


namespace NUMINAMATH_CALUDE_train_length_calculation_l3541_354137

/-- Calculates the length of a train given the speeds of a jogger and the train,
    the initial distance between them, and the time it takes for the train to pass the jogger. -/
theorem train_length_calculation (jogger_speed train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (5 / 18) →
  train_speed = 45 * (5 / 18) →
  initial_distance = 190 →
  passing_time = 31 →
  (train_speed - jogger_speed) * passing_time - initial_distance = 120 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3541_354137


namespace NUMINAMATH_CALUDE_letter_drawing_probabilities_l3541_354181

/-- Represents a set of letters -/
def LetterSet := Finset Char

/-- Represents a word as a list of characters -/
def Word := List Char

/-- Calculate the number of ways to arrange n items from a set of k items -/
def arrangements (n k : ℕ) : ℕ := 
  (k - n + 1).factorial / (k - n).factorial

/-- Calculate the probability of drawing a specific sequence -/
def probability_specific_sequence (total_letters : ℕ) (word_length : ℕ) : ℚ :=
  1 / arrangements word_length total_letters

/-- Calculate the probability of drawing a sequence with repeated letters -/
def probability_repeated_sequence (total_letters : ℕ) (word_length : ℕ) (permutations : ℕ) : ℚ :=
  permutations / arrangements word_length total_letters

/-- The main theorem to prove -/
theorem letter_drawing_probabilities 
  (s1 : LetterSet) 
  (w1 : Word)
  (s2 : LetterSet)
  (w2 : Word) :
  s1.card = 6 →
  w1.length = 4 →
  s2.card = 6 →
  w2.length = 4 →
  probability_specific_sequence 6 4 = 1 / 360 ∧
  probability_repeated_sequence 6 4 12 = 1 / 30 :=
by sorry

end NUMINAMATH_CALUDE_letter_drawing_probabilities_l3541_354181


namespace NUMINAMATH_CALUDE_smallest_interesting_number_l3541_354156

def is_interesting (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 2 * n = a^2 ∧ 15 * n = b^3

theorem smallest_interesting_number : 
  (is_interesting 1800) ∧ (∀ m : ℕ, m < 1800 → ¬(is_interesting m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_interesting_number_l3541_354156


namespace NUMINAMATH_CALUDE_no_rational_roots_l3541_354191

theorem no_rational_roots :
  ∀ (q : ℚ), 3 * q^4 - 2 * q^3 - 15 * q^2 + 6 * q + 3 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_l3541_354191


namespace NUMINAMATH_CALUDE_probability_of_rerolling_two_is_one_over_144_l3541_354106

/-- Represents the outcome of rolling a single die -/
inductive DieOutcome
| One | Two | Three | Four | Five | Six

/-- Represents the state of a die (original or rerolled) -/
inductive DieState
| Original (outcome : DieOutcome)
| Rerolled (original : DieOutcome) (new : DieOutcome)

/-- Represents the game state after Jason's decision -/
structure GameState :=
(dice : Fin 3 → DieState)
(rerolledCount : Nat)

/-- Determines if a game state is winning -/
def isWinningState (state : GameState) : Bool :=
  sorry

/-- Calculates the probability of a given game state -/
def probabilityOfState (state : GameState) : ℚ :=
  sorry

/-- Calculates the probability of Jason choosing to reroll exactly two dice -/
def probabilityOfRerollingTwo : ℚ :=
  sorry

theorem probability_of_rerolling_two_is_one_over_144 :
  probabilityOfRerollingTwo = 1 / 144 :=
sorry

end NUMINAMATH_CALUDE_probability_of_rerolling_two_is_one_over_144_l3541_354106


namespace NUMINAMATH_CALUDE_your_bill_before_tax_friend_order_equation_l3541_354123

/-- The cost of a taco in dollars -/
def taco_cost : ℝ := sorry

/-- The cost of an enchilada in dollars -/
def enchilada_cost : ℝ := 2

/-- The cost of 3 tacos and 5 enchiladas in dollars -/
def friend_order_cost : ℝ := 12.70

theorem your_bill_before_tax :
  2 * taco_cost + 3 * enchilada_cost = 7.80 :=
by
  sorry

/-- The friend's order cost equation -/
theorem friend_order_equation :
  3 * taco_cost + 5 * enchilada_cost = friend_order_cost :=
by
  sorry

end NUMINAMATH_CALUDE_your_bill_before_tax_friend_order_equation_l3541_354123


namespace NUMINAMATH_CALUDE_profit_percentage_l3541_354178

theorem profit_percentage (cost_price selling_price : ℚ) : 
  cost_price = 32 → 
  selling_price = 56 → 
  (selling_price - cost_price) / cost_price * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l3541_354178


namespace NUMINAMATH_CALUDE_average_of_solutions_is_zero_l3541_354195

theorem average_of_solutions_is_zero :
  let solutions := {x : ℝ | Real.sqrt (3 * x^2 + 4) = Real.sqrt 49}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
    (x₁ + x₂) / 2 = 0 ∧
    ∀ (x : ℝ), x ∈ solutions → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_average_of_solutions_is_zero_l3541_354195


namespace NUMINAMATH_CALUDE_power_sum_integer_l3541_354111

theorem power_sum_integer (a : ℝ) (h : ∃ k : ℤ, a + 1 / a = k) :
  ∀ n : ℕ, ∃ m : ℤ, a^n + 1 / a^n = m :=
by sorry

end NUMINAMATH_CALUDE_power_sum_integer_l3541_354111


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3541_354176

theorem binomial_coefficient_equality (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3541_354176


namespace NUMINAMATH_CALUDE_sum_of_squared_sums_l3541_354138

theorem sum_of_squared_sums (a b c : ℝ) : 
  (a^3 - 15*a^2 + 17*a - 8 = 0) →
  (b^3 - 15*b^2 + 17*b - 8 = 0) →
  (c^3 - 15*c^2 + 17*c - 8 = 0) →
  (a+b)^2 + (b+c)^2 + (c+a)^2 = 416 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squared_sums_l3541_354138


namespace NUMINAMATH_CALUDE_jeff_storage_usage_l3541_354182

theorem jeff_storage_usage (total_storage : ℝ) (storage_per_song : ℝ) (num_songs : ℕ) (mb_per_gb : ℕ) :
  total_storage = 16 →
  storage_per_song = 30 / 1000 →
  num_songs = 400 →
  mb_per_gb = 1000 →
  total_storage - (↑num_songs * storage_per_song) = 4 :=
by sorry

end NUMINAMATH_CALUDE_jeff_storage_usage_l3541_354182


namespace NUMINAMATH_CALUDE_remainder_problem_l3541_354183

theorem remainder_problem (n : ℕ) 
  (h1 : n^2 % 7 = 2) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l3541_354183


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l3541_354170

theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  r > 0 ∧ d > 0 ∧ 
  40 * r^2 = 10 * d^2 + 16000 ∧ 
  36 * r^2 = 81 * d^2 + 11664 →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l3541_354170


namespace NUMINAMATH_CALUDE_anna_baking_trays_l3541_354153

/-- The number of cupcakes per tray -/
def cupcakes_per_tray : ℕ := 20

/-- The price of each cupcake in dollars -/
def cupcake_price : ℚ := 2

/-- The fraction of cupcakes sold -/
def fraction_sold : ℚ := 3/5

/-- The total earnings in dollars -/
def total_earnings : ℚ := 96

/-- The number of baking trays Anna used -/
def num_trays : ℕ := 4

theorem anna_baking_trays :
  (cupcakes_per_tray : ℚ) * num_trays * fraction_sold * cupcake_price = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_anna_baking_trays_l3541_354153


namespace NUMINAMATH_CALUDE_square_diagonals_equal_l3541_354174

-- Define the necessary structures
structure Rectangle where
  diagonals_equal : Bool

structure Square extends Rectangle

-- Define the theorem
theorem square_diagonals_equal (h1 : ∀ r : Rectangle, r.diagonals_equal) 
  (h2 : Square → Rectangle) : 
  ∀ s : Square, (h2 s).diagonals_equal :=
by
  sorry


end NUMINAMATH_CALUDE_square_diagonals_equal_l3541_354174


namespace NUMINAMATH_CALUDE_quadratic_has_two_real_roots_roots_difference_three_l3541_354152

/-- The quadratic equation x^2 - (m-1)x + (m-2) = 0 -/
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - (m-1)*x + (m-2)

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := (m-1)^2 - 4*(m-2)

theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 :=
sorry

theorem roots_difference_three (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ |x₁ - x₂| = 3) →
  m = 0 ∨ m = 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_real_roots_roots_difference_three_l3541_354152


namespace NUMINAMATH_CALUDE_base_conversion_2025_to_octal_l3541_354117

theorem base_conversion_2025_to_octal :
  (2025 : ℕ) = (3 * 8^3 + 7 * 8^2 + 5 * 8^1 + 1 * 8^0 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_2025_to_octal_l3541_354117


namespace NUMINAMATH_CALUDE_factorial_sum_mod_20_l3541_354131

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_sum_mod_20 : (factorial 1 + factorial 2 + factorial 3 + factorial 4 + factorial 5 + factorial 6) % 20 = 13 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_mod_20_l3541_354131


namespace NUMINAMATH_CALUDE_value_difference_l3541_354139

theorem value_difference (n : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) :
  n = 80 ∧ increase_percent = 0.125 ∧ decrease_percent = 0.25 →
  n * (1 + increase_percent) - n * (1 - decrease_percent) = 30 :=
by sorry

end NUMINAMATH_CALUDE_value_difference_l3541_354139


namespace NUMINAMATH_CALUDE_orange_count_theorem_l3541_354140

/-- The number of oranges initially in the box -/
def initial_oranges : ℝ := 55.0

/-- The number of oranges Susan adds to the box -/
def added_oranges : ℝ := 35.0

/-- The total number of oranges in the box after Susan adds more -/
def total_oranges : ℝ := 90.0

/-- Theorem stating that the initial number of oranges plus the added oranges equals the total oranges -/
theorem orange_count_theorem : initial_oranges + added_oranges = total_oranges := by
  sorry

end NUMINAMATH_CALUDE_orange_count_theorem_l3541_354140


namespace NUMINAMATH_CALUDE_amanda_ticket_sales_l3541_354184

/-- Amanda's ticket sales problem -/
theorem amanda_ticket_sales 
  (total_goal : ℕ) 
  (friends : ℕ) 
  (tickets_per_friend : ℕ) 
  (second_day_sales : ℕ) 
  (h1 : total_goal = 80)
  (h2 : friends = 5)
  (h3 : tickets_per_friend = 4)
  (h4 : second_day_sales = 32) :
  total_goal - (friends * tickets_per_friend + second_day_sales) = 28 := by
sorry


end NUMINAMATH_CALUDE_amanda_ticket_sales_l3541_354184


namespace NUMINAMATH_CALUDE_shelves_needed_l3541_354112

theorem shelves_needed (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 12) (h2 : books_per_shelf = 4) :
  total_books / books_per_shelf = 3 := by
  sorry

end NUMINAMATH_CALUDE_shelves_needed_l3541_354112


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3541_354155

theorem complex_fraction_equality : (2 : ℂ) / (1 + Complex.I) = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3541_354155


namespace NUMINAMATH_CALUDE_smallest_prime_average_l3541_354172

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a list contains five different prime numbers
def isFiveDifferentPrimes (list : List ℕ) : Prop :=
  list.length = 5 ∧ list.Nodup ∧ ∀ n ∈ list, isPrime n

-- Define a function to calculate the average of a list of numbers
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem smallest_prime_average :
  ∀ list : List ℕ, isFiveDifferentPrimes list → (average list).isInt → average list ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_average_l3541_354172


namespace NUMINAMATH_CALUDE_smaller_bill_value_l3541_354128

theorem smaller_bill_value (total_bills : ℕ) (total_value : ℕ) (small_bills : ℕ) (ten_bills : ℕ) 
  (h1 : total_bills = 12)
  (h2 : total_value = 100)
  (h3 : small_bills = 4)
  (h4 : ten_bills = 8)
  (h5 : total_bills = small_bills + ten_bills)
  (h6 : total_value = small_bills * x + ten_bills * 10) :
  x = 5 := by
  sorry

#check smaller_bill_value

end NUMINAMATH_CALUDE_smaller_bill_value_l3541_354128


namespace NUMINAMATH_CALUDE_agno3_mass_fraction_l3541_354135

/-- Given the number of moles, molar mass, and total solution mass of AgNO₃,
    prove that its mass fraction in the solution is 8%. -/
theorem agno3_mass_fraction :
  ∀ (n M m_total : ℝ),
  n = 0.12 →
  M = 170 →
  m_total = 255 →
  let m := n * M
  let ω := m * 100 / m_total
  ω = 8 := by
sorry

end NUMINAMATH_CALUDE_agno3_mass_fraction_l3541_354135


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l3541_354199

/-- Given two circular pulleys with an uncrossed belt, prove the distance between their centers. -/
theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (h₁ : r₁ = 14)
  (h₂ : r₂ = 4)
  (h₃ : contact_distance = 24) :
  Real.sqrt ((r₁ - r₂)^2 + contact_distance^2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l3541_354199


namespace NUMINAMATH_CALUDE_min_values_ab_l3541_354190

theorem min_values_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  (ab ≥ 8) ∧ (a + b ≥ 3 + 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_values_ab_l3541_354190


namespace NUMINAMATH_CALUDE_tan_fifteen_to_sqrt_three_l3541_354109

theorem tan_fifteen_to_sqrt_three : (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_to_sqrt_three_l3541_354109


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l3541_354186

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_11th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_2 : a 2 = 3)
  (h_6 : a 6 = 7) :
  a 11 = 12 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l3541_354186


namespace NUMINAMATH_CALUDE_equation_solutions_l3541_354134

theorem equation_solutions :
  (∀ x y : ℤ, y^4 + 2*x^4 + 1 = 4*x^2*y ↔ (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = 1)) ∧
  (∀ x y z : ℕ+, 5*(x*y + y*z + z*x) = 4*x*y*z ↔
    ((x = 5 ∧ y = 10 ∧ z = 2) ∨ (x = 5 ∧ y = 2 ∧ z = 10) ∨
     (x = 10 ∧ y = 5 ∧ z = 2) ∨ (x = 10 ∧ y = 2 ∧ z = 5) ∨
     (x = 2 ∧ y = 10 ∧ z = 5) ∨ (x = 2 ∧ y = 5 ∧ z = 10) ∨
     (x = 4 ∧ y = 20 ∧ z = 2) ∨ (x = 4 ∧ y = 2 ∧ z = 20) ∨
     (x = 2 ∧ y = 4 ∧ z = 20) ∨ (x = 2 ∧ y = 20 ∧ z = 4) ∨
     (x = 20 ∧ y = 2 ∧ z = 4) ∨ (x = 20 ∧ y = 4 ∧ z = 2))) := by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l3541_354134


namespace NUMINAMATH_CALUDE_candy_left_is_49_l3541_354188

/-- The number of pieces of candy Brent has left after trick-or-treating and giving some to his sister -/
def candy_left : ℕ :=
  let kit_kats := 5
  let hershey_kisses := 3 * kit_kats
  let nerds := 8
  let lollipops := 11
  let baby_ruths := 10
  let reese_cups := baby_ruths / 2
  let total_candy := kit_kats + hershey_kisses + nerds + lollipops + baby_ruths + reese_cups
  let given_away := 5
  total_candy - given_away

theorem candy_left_is_49 : candy_left = 49 := by
  sorry

end NUMINAMATH_CALUDE_candy_left_is_49_l3541_354188


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3541_354196

-- Problem 1
theorem problem_1 : 8 * 77 * 125 = 77000 := by sorry

-- Problem 2
theorem problem_2 : 12 * 98 = 1176 := by sorry

-- Problem 3
theorem problem_3 : 6 * 321 + 6 * 179 = 3000 := by sorry

-- Problem 4
theorem problem_4 : 56 * 101 - 56 = 5600 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3541_354196


namespace NUMINAMATH_CALUDE_exists_special_sequence_l3541_354127

/-- An infinite increasing sequence of natural numbers -/
def IncreasingSeq : ℕ → ℕ := sorry

/-- The property that the sequence is increasing -/
axiom seq_increasing : ∀ n : ℕ, IncreasingSeq n < IncreasingSeq (n + 1)

/-- The coprimality property of the sequence -/
axiom seq_coprime : ∀ i j p q r : ℕ, 
  i ≠ j → i ≠ p → i ≠ q → i ≠ r → j ≠ p → j ≠ q → j ≠ r → p ≠ q → p ≠ r → q ≠ r →
  Nat.gcd (IncreasingSeq i + IncreasingSeq j) (IncreasingSeq p + IncreasingSeq q + IncreasingSeq r) = 1

/-- The main theorem: existence of the sequence with the required properties -/
theorem exists_special_sequence : 
  ∃ (seq : ℕ → ℕ), 
    (∀ n : ℕ, seq n < seq (n + 1)) ∧ 
    (∀ i j p q r : ℕ, 
      i ≠ j → i ≠ p → i ≠ q → i ≠ r → j ≠ p → j ≠ q → j ≠ r → p ≠ q → p ≠ r → q ≠ r →
      Nat.gcd (seq i + seq j) (seq p + seq q + seq r) = 1) :=
sorry

end NUMINAMATH_CALUDE_exists_special_sequence_l3541_354127


namespace NUMINAMATH_CALUDE_sqrt_a_minus_4_real_l3541_354133

theorem sqrt_a_minus_4_real (a : ℝ) : (∃ x : ℝ, x^2 = a - 4) ↔ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_a_minus_4_real_l3541_354133


namespace NUMINAMATH_CALUDE_damages_cost_l3541_354104

def tire_cost_1 : ℕ := 230
def tire_cost_2 : ℕ := 250
def tire_cost_3 : ℕ := 280
def window_cost_1 : ℕ := 700
def window_cost_2 : ℕ := 800
def window_cost_3 : ℕ := 900

def total_damages : ℕ := 
  2 * tire_cost_1 + 2 * tire_cost_2 + 2 * tire_cost_3 +
  window_cost_1 + window_cost_2 + window_cost_3

theorem damages_cost : total_damages = 3920 := by
  sorry

end NUMINAMATH_CALUDE_damages_cost_l3541_354104


namespace NUMINAMATH_CALUDE_orchestra_size_l3541_354164

def percussion_count : ℕ := 3
def brass_count : ℕ := 5 + 4 + 2 + 2
def strings_count : ℕ := 7 + 5 + 4 + 2
def woodwinds_count : ℕ := 3 + 4 + 2 + 1
def keyboards_harp_count : ℕ := 1 + 1
def conductor_count : ℕ := 1

theorem orchestra_size :
  percussion_count + brass_count + strings_count + woodwinds_count + keyboards_harp_count + conductor_count = 47 := by
  sorry

end NUMINAMATH_CALUDE_orchestra_size_l3541_354164


namespace NUMINAMATH_CALUDE_tracy_has_two_dogs_l3541_354102

/-- The number of dogs Tracy has -/
def num_dogs : ℕ :=
  let cups_per_meal : ℚ := 3/2  -- 1.5 cups per meal
  let meals_per_day : ℕ := 3
  let total_pounds : ℕ := 4
  let cups_per_pound : ℚ := 9/4  -- 2.25 cups per pound

  let total_cups : ℚ := total_pounds * cups_per_pound
  let cups_per_dog_per_day : ℚ := cups_per_meal * meals_per_day

  (total_cups / cups_per_dog_per_day).num.toNat

/-- Theorem stating that Tracy has 2 dogs -/
theorem tracy_has_two_dogs : num_dogs = 2 := by
  sorry

end NUMINAMATH_CALUDE_tracy_has_two_dogs_l3541_354102


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l3541_354126

theorem product_of_sums_equals_difference_of_powers : 
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l3541_354126


namespace NUMINAMATH_CALUDE_part_one_part_two_l3541_354125

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ :=
  if a ≥ b then b^2 else 2*a - b

-- Theorem for part (1)
theorem part_one : triangle (-3) (-4) = 16 := by sorry

-- Theorem for part (2)
theorem part_two : triangle (triangle (-2) 3) (-8) = 64 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3541_354125


namespace NUMINAMATH_CALUDE_min_value_theorem_l3541_354177

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 27) :
  3 * x + 2 * y + z ≥ 18 * Real.rpow 2 (1/3) ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 27 ∧
    3 * x₀ + 2 * y₀ + z₀ = 18 * Real.rpow 2 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3541_354177


namespace NUMINAMATH_CALUDE_fair_special_savings_l3541_354122

/-- Calculates the percentage saved when buying three pairs of sandals under the "fair special" --/
theorem fair_special_savings : 
  let regular_price : ℝ := 60
  let second_pair_discount : ℝ := 0.4
  let third_pair_discount : ℝ := 0.25
  let total_regular_price : ℝ := 3 * regular_price
  let discounted_price : ℝ := regular_price + 
                              (1 - second_pair_discount) * regular_price + 
                              (1 - third_pair_discount) * regular_price
  let savings : ℝ := total_regular_price - discounted_price
  let percentage_saved : ℝ := (savings / total_regular_price) * 100
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |percentage_saved - 22| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_fair_special_savings_l3541_354122


namespace NUMINAMATH_CALUDE_light_flash_duration_l3541_354129

/-- Proves that if a light flashes every 15 seconds and flashes 240 times, the time taken is exactly one hour. -/
theorem light_flash_duration (flash_interval : ℕ) (total_flashes : ℕ) (seconds_per_hour : ℕ) : 
  flash_interval = 15 → 
  total_flashes = 240 → 
  seconds_per_hour = 3600 → 
  flash_interval * total_flashes = seconds_per_hour :=
by
  sorry

#check light_flash_duration

end NUMINAMATH_CALUDE_light_flash_duration_l3541_354129


namespace NUMINAMATH_CALUDE_symphony_orchestra_members_l3541_354114

theorem symphony_orchestra_members : ∃! n : ℕ,
  200 < n ∧ n < 300 ∧
  n % 6 = 2 ∧
  n % 8 = 3 ∧
  n % 9 = 4 ∧
  n = 260 := by
sorry

end NUMINAMATH_CALUDE_symphony_orchestra_members_l3541_354114


namespace NUMINAMATH_CALUDE_apple_juice_distribution_l3541_354116

/-- Given a total amount of apple juice and the difference between two people's consumption,
    calculate the amount consumed by the person who drinks more. -/
theorem apple_juice_distribution (total : ℝ) (difference : ℝ) (kyu_yeon_amount : ℝ) : 
  total = 12.4 ∧ difference = 2.6 → kyu_yeon_amount = 7.5 := by
  sorry

#check apple_juice_distribution

end NUMINAMATH_CALUDE_apple_juice_distribution_l3541_354116


namespace NUMINAMATH_CALUDE_rope_cutting_l3541_354113

theorem rope_cutting (total_length : ℕ) (long_piece_length : ℕ) (num_short_pieces : ℕ) 
  (h1 : total_length = 27)
  (h2 : long_piece_length = 4)
  (h3 : num_short_pieces = 3) :
  ∃ (num_long_pieces : ℕ) (short_piece_length : ℕ),
    num_long_pieces * long_piece_length + num_short_pieces * short_piece_length = total_length ∧
    num_long_pieces = total_length / long_piece_length ∧
    short_piece_length = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_l3541_354113


namespace NUMINAMATH_CALUDE_final_student_count_l3541_354149

/-- Represents the arrangement of students in the photo. -/
structure StudentArrangement where
  rows : ℕ
  columns : ℕ

/-- The initial arrangement of students before any changes. -/
def initial_arrangement : StudentArrangement := { rows := 0, columns := 0 }

/-- The arrangement after moving one student from each row. -/
def first_adjustment (a : StudentArrangement) : StudentArrangement :=
  { rows := a.rows + 1, columns := a.columns - 1 }

/-- The arrangement after moving a second student from each row. -/
def second_adjustment (a : StudentArrangement) : StudentArrangement :=
  { rows := a.rows + 1, columns := a.columns - 1 }

/-- Calculates the total number of students in the arrangement. -/
def total_students (a : StudentArrangement) : ℕ := a.rows * a.columns

/-- The theorem stating the final number of students in the photo. -/
theorem final_student_count :
  ∃ (a : StudentArrangement),
    (first_adjustment a).columns = (first_adjustment a).rows + 4 ∧
    (second_adjustment (first_adjustment a)).columns = (second_adjustment (first_adjustment a)).rows ∧
    total_students (second_adjustment (first_adjustment a)) = 24 :=
  sorry

end NUMINAMATH_CALUDE_final_student_count_l3541_354149


namespace NUMINAMATH_CALUDE_new_average_after_multipliers_l3541_354154

theorem new_average_after_multipliers (original_list : List ℝ) 
  (h1 : original_list.length = 7)
  (h2 : original_list.sum / original_list.length = 20)
  (multipliers : List ℝ := [2, 3, 4, 5, 6, 7, 8]) :
  (List.zipWith (· * ·) original_list multipliers).sum / original_list.length = 100 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_multipliers_l3541_354154


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l3541_354144

theorem parabola_point_coordinates :
  ∀ (x y : ℝ),
  y^2 = 4*x →                             -- Point P(x, y) lies on the parabola y^2 = 4x
  (x - 1)^2 + y^2 = 100 →                 -- Distance from P to focus F(1, 0) is 10
  x = 9 ∧ (y = 6 ∨ y = -6) :=             -- Conclusion: x = 9 and y = ±6
by
  sorry


end NUMINAMATH_CALUDE_parabola_point_coordinates_l3541_354144


namespace NUMINAMATH_CALUDE_multiply_72519_by_9999_l3541_354185

theorem multiply_72519_by_9999 : 72519 * 9999 = 725117481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72519_by_9999_l3541_354185


namespace NUMINAMATH_CALUDE_concert_ticket_cost_l3541_354194

theorem concert_ticket_cost (num_tickets : ℕ) (discount_rate : ℚ) (discount_threshold : ℕ) (total_paid : ℚ) :
  num_tickets = 12 →
  discount_rate = 5 / 100 →
  discount_threshold = 10 →
  total_paid = 476 →
  ∃ (original_cost : ℚ), 
    original_cost * (num_tickets - discount_rate * (num_tickets - discount_threshold)) = total_paid ∧
    original_cost = 40 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_l3541_354194


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_of_powers_l3541_354160

theorem cube_root_unity_sum_of_powers : 
  let ω : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  (ω ^ 3 = 1) → (ω ≠ 1) →
  (ω ^ 8 + (ω ^ 2) ^ 8 = -2) := by
sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_of_powers_l3541_354160


namespace NUMINAMATH_CALUDE_union_A_B_union_complement_A_B_l3541_354130

-- Define the universe set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 7}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x : ℝ | 2 ≤ x ∧ x ≤ 7} := by sorry

-- Theorem for (∁ₓA) ∪ (∁ₓB)
theorem union_complement_A_B : (Set.compl A) ∪ (Set.compl B) = {x : ℝ | x < 3 ∨ x ≥ 5} := by sorry

end NUMINAMATH_CALUDE_union_A_B_union_complement_A_B_l3541_354130


namespace NUMINAMATH_CALUDE_non_holiday_rate_correct_l3541_354157

/-- The number of customers per hour during the non-holiday season -/
def non_holiday_rate : ℕ := 175

/-- The number of customers per hour during the holiday season -/
def holiday_rate : ℕ := non_holiday_rate * 2

/-- The total number of customers during the holiday season -/
def total_customers : ℕ := 2800

/-- The number of hours observed during the holiday season -/
def observation_hours : ℕ := 8

/-- Theorem stating that the non-holiday rate is correct given the conditions -/
theorem non_holiday_rate_correct : 
  holiday_rate * observation_hours = total_customers ∧
  non_holiday_rate = 175 := by
  sorry

end NUMINAMATH_CALUDE_non_holiday_rate_correct_l3541_354157


namespace NUMINAMATH_CALUDE_books_written_proof_l3541_354132

def total_books (zig_books flo_books : ℕ) : ℕ :=
  zig_books + flo_books

theorem books_written_proof (zig_books flo_books : ℕ) 
  (h1 : zig_books = 60) 
  (h2 : zig_books = 4 * flo_books) : 
  total_books zig_books flo_books = 75 := by
  sorry

end NUMINAMATH_CALUDE_books_written_proof_l3541_354132


namespace NUMINAMATH_CALUDE_pretty_numbers_characterization_l3541_354108

def is_pretty (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ k ℓ : ℕ, 0 < k ∧ k < n ∧ 0 < ℓ ∧ ℓ < n ∧ k ∣ n ∧ ℓ ∣ n →
    (2 * k - ℓ) ∣ n ∨ (2 * ℓ - k) ∣ n

theorem pretty_numbers_characterization (n : ℕ) :
  is_pretty n ↔ Nat.Prime n ∨ n = 6 ∨ n = 9 ∨ n = 15 :=
sorry

end NUMINAMATH_CALUDE_pretty_numbers_characterization_l3541_354108


namespace NUMINAMATH_CALUDE_odd_m_triple_g_35_l3541_354124

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5
  else if n % 3 = 0 then n / 3
  else n  -- This case is not specified in the original problem, but needed for completeness

theorem odd_m_triple_g_35 (m : ℤ) (h_odd : m % 2 = 1) (h_triple_g : g (g (g m)) = 35) : m = 85 := by
  sorry

end NUMINAMATH_CALUDE_odd_m_triple_g_35_l3541_354124


namespace NUMINAMATH_CALUDE_train_platform_time_l3541_354120

/-- Given a train of length 1200 meters that takes 120 seconds to pass a tree,
    this theorem proves that the time required for the train to pass a platform
    of length 800 meters is 200 seconds. -/
theorem train_platform_time (train_length : ℝ) (tree_pass_time : ℝ) (platform_length : ℝ)
  (h1 : train_length = 1200)
  (h2 : tree_pass_time = 120)
  (h3 : platform_length = 800) :
  (train_length + platform_length) / (train_length / tree_pass_time) = 200 :=
by sorry

end NUMINAMATH_CALUDE_train_platform_time_l3541_354120


namespace NUMINAMATH_CALUDE_function_properties_l3541_354121

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x
noncomputable def g (x : ℝ) : ℝ := 2 / x

-- Define the combined function h
noncomputable def h (x : ℝ) : ℝ := f x + g x

-- Theorem statement
theorem function_properties :
  (∃ k : ℝ, ∀ x : ℝ, x > 0 → f x = k * x) ∧
  (∃ c : ℝ, ∀ x : ℝ, x > 0 → g x = c / x) ∧
  f 1 = 1 ∧
  g 1 = 2 →
  (∀ x : ℝ, x > 0 → f x = x ∧ g x = 2 / x) ∧
  (∀ x : ℝ, x ≠ 0 → h x = -h (-x)) ∧
  (∀ x : ℝ, 0 < x → x ≤ Real.sqrt 2 → h x ≥ 2 * Real.sqrt 2) ∧
  h (Real.sqrt 2) = 2 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_function_properties_l3541_354121


namespace NUMINAMATH_CALUDE_quadratic_function_bound_l3541_354146

theorem quadratic_function_bound (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x^2 + b * x + c| ≤ 1) →
  (a + b) * c ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bound_l3541_354146


namespace NUMINAMATH_CALUDE_meaningful_set_equiv_range_expression_meaningful_iff_in_set_l3541_354105

-- Define the set of real numbers for which the expression is meaningful
def MeaningfulSet : Set ℝ :=
  {x : ℝ | x ≥ -2/3 ∧ x ≠ 0}

-- Theorem stating that the MeaningfulSet is equivalent to the given range
theorem meaningful_set_equiv_range :
  MeaningfulSet = Set.Icc (-2/3) 0 ∪ Set.Ioi 0 :=
sorry

-- Theorem proving that the expression is meaningful if and only if x is in MeaningfulSet
theorem expression_meaningful_iff_in_set (x : ℝ) :
  (3 * x + 2 ≥ 0 ∧ x ≠ 0) ↔ x ∈ MeaningfulSet :=
sorry

end NUMINAMATH_CALUDE_meaningful_set_equiv_range_expression_meaningful_iff_in_set_l3541_354105


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l3541_354193

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 18

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 36

theorem bowling_ball_weight_proof :
  (8 * bowling_ball_weight = 4 * canoe_weight) ∧
  (3 * canoe_weight = 108) →
  bowling_ball_weight = 18 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l3541_354193


namespace NUMINAMATH_CALUDE_craft_store_optimal_solution_l3541_354103

/-- Represents the craft store problem -/
structure CraftStore where
  profit_per_item : ℝ
  cost_50_items : ℝ
  revenue_40_items : ℝ
  initial_daily_sales : ℕ
  sales_increase_per_yuan : ℕ

/-- Theorem stating the optimal solution for the craft store problem -/
theorem craft_store_optimal_solution (store : CraftStore) 
  (h1 : store.profit_per_item = 45)
  (h2 : store.cost_50_items = store.revenue_40_items)
  (h3 : store.initial_daily_sales = 100)
  (h4 : store.sales_increase_per_yuan = 4) :
  ∃ (cost_price marked_price optimal_reduction max_profit : ℝ),
    cost_price = 180 ∧
    marked_price = 225 ∧
    optimal_reduction = 10 ∧
    max_profit = 4900 := by
  sorry

end NUMINAMATH_CALUDE_craft_store_optimal_solution_l3541_354103


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l3541_354118

theorem cylinder_volume_relation (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let volume_C := π * h^2 * r
  let volume_D := π * r^2 * h
  (volume_D = 3 * volume_C) → (volume_D = 9 * π * h^3) := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l3541_354118


namespace NUMINAMATH_CALUDE_defective_percentage_is_3_6_percent_l3541_354171

/-- Represents the percentage of products manufactured by each machine -/
structure MachineProduction where
  m1 : ℝ
  m2 : ℝ
  m3 : ℝ

/-- Represents the percentage of defective products for each machine -/
structure DefectivePercentage where
  m1 : ℝ
  m2 : ℝ
  m3 : ℝ

/-- Calculates the percentage of defective products in the stockpile -/
def calculateDefectivePercentage (prod : MachineProduction) (defect : DefectivePercentage) : ℝ :=
  prod.m1 * defect.m1 + prod.m2 * defect.m2 + prod.m3 * defect.m3

theorem defective_percentage_is_3_6_percent 
  (prod : MachineProduction)
  (defect : DefectivePercentage)
  (h1 : prod.m1 = 0.4)
  (h2 : prod.m2 = 0.3)
  (h3 : prod.m3 = 0.3)
  (h4 : defect.m1 = 0.03)
  (h5 : defect.m2 = 0.01)
  (h6 : defect.m3 = 0.07) :
  calculateDefectivePercentage prod defect = 0.036 := by
  sorry

#eval calculateDefectivePercentage 
  { m1 := 0.4, m2 := 0.3, m3 := 0.3 } 
  { m1 := 0.03, m2 := 0.01, m3 := 0.07 }

end NUMINAMATH_CALUDE_defective_percentage_is_3_6_percent_l3541_354171


namespace NUMINAMATH_CALUDE_quadratic_other_x_intercept_l3541_354167

/-- Given a quadratic function f(x) = ax^2 + bx + c with vertex (5, 10) and
    one x-intercept at (1, 0), the x-coordinate of the other x-intercept is 9. -/
theorem quadratic_other_x_intercept
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h_quad : ∀ x, f x = a * x^2 + b * x + c)
  (h_vertex : f 5 = 10 ∧ ∀ x, f x ≤ f 5)
  (h_intercept : f 1 = 0) :
  ∃ x, x ≠ 1 ∧ f x = 0 ∧ x = 9 :=
sorry

end NUMINAMATH_CALUDE_quadratic_other_x_intercept_l3541_354167


namespace NUMINAMATH_CALUDE_hillary_friday_reading_time_l3541_354166

/-- The total assigned reading time in minutes -/
def total_assigned_time : ℕ := 60

/-- The number of minutes Hillary read on Saturday -/
def saturday_reading_time : ℕ := 28

/-- The number of minutes Hillary needs to read on Sunday -/
def sunday_reading_time : ℕ := 16

/-- The number of minutes Hillary read on Friday night -/
def friday_reading_time : ℕ := total_assigned_time - (saturday_reading_time + sunday_reading_time)

theorem hillary_friday_reading_time :
  friday_reading_time = 16 := by sorry

end NUMINAMATH_CALUDE_hillary_friday_reading_time_l3541_354166


namespace NUMINAMATH_CALUDE_cutting_process_ends_at_1998_l3541_354142

/-- Represents a shape with points on its boundary -/
structure Shape :=
  (points : ℕ)

/-- Represents the state of the cutting process -/
structure CuttingState :=
  (shape : Shape)
  (cuts : ℕ)

/-- Checks if a shape is a polygon -/
def is_polygon (s : Shape) : Prop :=
  s.points ≤ 3

/-- Checks if a shape can become a polygon with further cutting -/
def can_become_polygon (s : Shape) : Prop :=
  s.points > 3

/-- Performs a cut on the shape -/
def cut (state : CuttingState) : CuttingState :=
  { shape := { points := state.shape.points - 1 },
    cuts := state.cuts + 1 }

/-- The main theorem to be proved -/
theorem cutting_process_ends_at_1998 :
  ∀ (initial_state : CuttingState),
    initial_state.shape.points = 1001 →
    ∀ (n : ℕ),
      n ≤ 1998 →
      ¬(is_polygon (cut^[n] initial_state).shape) ∧
      can_become_polygon (cut^[n] initial_state).shape →
      ¬(∃ (m : ℕ),
        m > 1998 ∧
        ¬(is_polygon (cut^[m] initial_state).shape) ∧
        can_become_polygon (cut^[m] initial_state).shape) :=
sorry

end NUMINAMATH_CALUDE_cutting_process_ends_at_1998_l3541_354142


namespace NUMINAMATH_CALUDE_bags_found_next_day_l3541_354165

theorem bags_found_next_day 
  (initial_bags : ℕ) 
  (total_bags : ℕ) 
  (h : initial_bags ≤ total_bags) :
  total_bags - initial_bags = total_bags - initial_bags :=
by sorry

end NUMINAMATH_CALUDE_bags_found_next_day_l3541_354165


namespace NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_geq_3_l3541_354147

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 6

theorem decreasing_quadratic_implies_a_geq_3 (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 3 → f a x₁ > f a x₂) →
  a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_geq_3_l3541_354147


namespace NUMINAMATH_CALUDE_mickey_horses_per_week_l3541_354145

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of horses Minnie mounts per day -/
def minnie_horses_per_day : ℕ := days_in_week + 3

/-- The number of horses Mickey mounts per day -/
def mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6

/-- Theorem: Mickey mounts 98 horses per week -/
theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end NUMINAMATH_CALUDE_mickey_horses_per_week_l3541_354145


namespace NUMINAMATH_CALUDE_average_book_width_l3541_354148

theorem average_book_width :
  let book_widths : List ℚ := [5, 3/4, 3/2, 3, 29/4, 12]
  (book_widths.sum / book_widths.length : ℚ) = 59/12 := by
sorry

end NUMINAMATH_CALUDE_average_book_width_l3541_354148


namespace NUMINAMATH_CALUDE_inequality_proof_l3541_354150

theorem inequality_proof (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 6)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 12) :
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧ 
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3541_354150


namespace NUMINAMATH_CALUDE_sunzi_car_problem_l3541_354151

theorem sunzi_car_problem (x : ℕ) : 
  (3 * (x - 2) = 2 * x + 9) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_sunzi_car_problem_l3541_354151


namespace NUMINAMATH_CALUDE_unique_a_value_l3541_354141

-- Define the set A
def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

-- State the theorem
theorem unique_a_value : ∃! a : ℝ, 1 ∈ A a ∧ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l3541_354141


namespace NUMINAMATH_CALUDE_unique_prime_product_sum_l3541_354179

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def distinct_primes (p q r : ℕ) : Prop := is_prime p ∧ is_prime q ∧ is_prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r

theorem unique_prime_product_sum (p q r : ℕ) : 
  5401 = p * q * r → 
  distinct_primes p q r →
  ∃! n : ℕ, ∃ p1 p2 p3 : ℕ, 
    n = p1 * p2 * p3 ∧ 
    distinct_primes p1 p2 p3 ∧ 
    p1 + p2 + p3 = p + q + r ∧
    n ≠ 5401 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_product_sum_l3541_354179


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3541_354175

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- a_1, a_2, and a_4 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℝ) : Prop :=
  a 2 ^ 2 = a 1 * a 4

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
  (h1 : arithmetic_seq a) (h2 : geometric_subseq a) : a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3541_354175


namespace NUMINAMATH_CALUDE_prime_square_minus_cube_eq_one_l3541_354173

theorem prime_square_minus_cube_eq_one (p q : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ p > 0 ∧ q > 0 → (p^2 - q^3 = 1 ↔ p = 3 ∧ q = 2) := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_cube_eq_one_l3541_354173


namespace NUMINAMATH_CALUDE_largest_common_divisor_l3541_354189

theorem largest_common_divisor : ∃ (n : ℕ), n = 45 ∧ 
  n ∣ 540 ∧ n < 60 ∧ n ∣ 180 ∧ 
  ∀ (m : ℕ), m ∣ 540 ∧ m < 60 ∧ m ∣ 180 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_l3541_354189


namespace NUMINAMATH_CALUDE_area_comparison_l3541_354159

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ := sorry

-- Define a function to check if a triangle is inscribed in a circle
def isInscribed (t : Triangle) (c : Circle) : Prop := sorry

-- Define a function to find the points where angle bisectors meet the circle
def angleBisectorPoints (t : Triangle) (c : Circle) : Triangle := sorry

-- Theorem statement
theorem area_comparison 
  (t : Triangle) (c : Circle) 
  (h : isInscribed t c) : 
  let t' := angleBisectorPoints t c
  triangleArea t ≤ triangleArea t' := by sorry

end NUMINAMATH_CALUDE_area_comparison_l3541_354159


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l3541_354180

theorem wire_cut_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  (a^2 / 16 = b^2 / (4 * Real.pi)) → a / b = 2 / Real.sqrt Real.pi := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l3541_354180


namespace NUMINAMATH_CALUDE_inverse_proportion_order_l3541_354187

/-- Given points A(-2,a), B(-1,b), and C(3,c) on the graph of y = 4/x, prove b < a < c -/
theorem inverse_proportion_order (a b c : ℝ) : 
  ((-2 : ℝ) * a = 4) → ((-1 : ℝ) * b = 4) → ((3 : ℝ) * c = 4) → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_order_l3541_354187


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3541_354163

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 1| ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3541_354163


namespace NUMINAMATH_CALUDE_concave_function_triangle_inequality_l3541_354115

def f (x : ℝ) := x^2 - 2*x + 2

theorem concave_function_triangle_inequality (m : ℝ) : 
  (∀ a b c : ℝ, 1/3 ≤ a ∧ a < b ∧ b < c ∧ c ≤ m^2 - m + 2 → 
    f a + f b > f c ∧ f b + f c > f a ∧ f c + f a > f b) ↔ 
  0 ≤ m ∧ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_concave_function_triangle_inequality_l3541_354115


namespace NUMINAMATH_CALUDE_interest_difference_l3541_354168

/-- Calculate the difference between compound interest and simple interest -/
theorem interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) (compounding_frequency : ℕ) :
  principal = 1200 →
  rate = 0.1 →
  time = 1 →
  compounding_frequency = 2 →
  let simple_interest := principal * rate * time
  let compound_interest := principal * ((1 + rate / compounding_frequency) ^ (compounding_frequency * time) - 1)
  compound_interest - simple_interest = 3 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l3541_354168


namespace NUMINAMATH_CALUDE_existence_of_same_sum_opposite_signs_l3541_354198

theorem existence_of_same_sum_opposite_signs :
  ∃ (y : ℝ) (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ > 0 ∧ x₁^4 + x₁^5 = y ∧ x₂^4 + x₂^5 = y :=
by sorry

end NUMINAMATH_CALUDE_existence_of_same_sum_opposite_signs_l3541_354198


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3541_354107

/-- Given that the inequality |x+a|+|x-1|+a>2009 (where a is a constant) has a non-empty set of solutions, 
    the range of values for a is (-∞, 1004) -/
theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + a| + |x - 1| + a > 2009) → 
  a ∈ Set.Iio 1004 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3541_354107


namespace NUMINAMATH_CALUDE_brother_difference_is_two_l3541_354101

/-- The number of Aaron's brothers -/
def aaron_brothers : ℕ := 4

/-- The number of Bennett's brothers -/
def bennett_brothers : ℕ := 6

/-- The difference between twice the number of Aaron's brothers and the number of Bennett's brothers -/
def brother_difference : ℕ := 2 * aaron_brothers - bennett_brothers

/-- Theorem stating that the difference between twice the number of Aaron's brothers
    and the number of Bennett's brothers is 2 -/
theorem brother_difference_is_two : brother_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_brother_difference_is_two_l3541_354101


namespace NUMINAMATH_CALUDE_power_values_l3541_354136

-- Define variables
variable (a m n : ℝ)

-- State the theorem
theorem power_values (h1 : a^m = 2) (h2 : a^n = 3) :
  a^(4*m + 3*n) = 432 ∧ a^(5*m - 2*n) = 32/9 := by
  sorry

end NUMINAMATH_CALUDE_power_values_l3541_354136


namespace NUMINAMATH_CALUDE_tickets_per_candy_l3541_354100

def whack_a_mole_tickets : ℕ := 2
def skee_ball_tickets : ℕ := 13
def candies_bought : ℕ := 5

def total_tickets : ℕ := whack_a_mole_tickets + skee_ball_tickets

theorem tickets_per_candy : total_tickets / candies_bought = 3 := by
  sorry

end NUMINAMATH_CALUDE_tickets_per_candy_l3541_354100


namespace NUMINAMATH_CALUDE_suspension_days_per_instance_l3541_354143

/-- The number of fingers and toes a typical person has -/
def typical_fingers_and_toes : ℕ := 20

/-- The number of instances of bullying Kris is responsible for -/
def bullying_instances : ℕ := 20

/-- The total number of days Kris was suspended -/
def total_suspension_days : ℕ := 3 * typical_fingers_and_toes

/-- The number of days Kris was suspended for each instance of bullying -/
def days_per_instance : ℚ := total_suspension_days / bullying_instances

theorem suspension_days_per_instance :
  days_per_instance = 3 := by sorry

end NUMINAMATH_CALUDE_suspension_days_per_instance_l3541_354143


namespace NUMINAMATH_CALUDE_not_all_equilateral_triangles_have_same_perimeter_l3541_354197

-- Define an equilateral triangle
structure EquilateralTriangle where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

-- Properties of equilateral triangles
def EquilateralTriangle.isEquiangular (t : EquilateralTriangle) : Prop :=
  -- All angles are 60 degrees
  true

def EquilateralTriangle.isIsosceles (t : EquilateralTriangle) : Prop :=
  -- At least two sides are equal (all sides are equal in this case)
  true

def EquilateralTriangle.isRegularPolygon (t : EquilateralTriangle) : Prop :=
  -- All sides equal and all angles equal
  true

def EquilateralTriangle.isSimilarTo (t1 t2 : EquilateralTriangle) : Prop :=
  -- All equilateral triangles are similar
  true

def EquilateralTriangle.perimeter (t : EquilateralTriangle) : ℝ :=
  3 * t.sideLength

-- Theorem to prove
theorem not_all_equilateral_triangles_have_same_perimeter :
  ∃ t1 t2 : EquilateralTriangle, t1.perimeter ≠ t2.perimeter ∧
    t1.isEquiangular ∧ t2.isEquiangular ∧
    t1.isIsosceles ∧ t2.isIsosceles ∧
    t1.isRegularPolygon ∧ t2.isRegularPolygon ∧
    t1.isSimilarTo t2 :=
  sorry

end NUMINAMATH_CALUDE_not_all_equilateral_triangles_have_same_perimeter_l3541_354197


namespace NUMINAMATH_CALUDE_system_equations_properties_l3541_354110

theorem system_equations_properties (a x y : ℝ) 
  (eq1 : x + y = 1 - a) 
  (eq2 : x - y = 3 * a + 5) 
  (x_pos : x > 0) 
  (y_nonneg : y ≥ 0) : 
  (a = -5/3 → x = y) ∧ 
  (a = -2 → x + y = 5 + a) ∧ 
  (0 < x ∧ x ≤ 1 → 2 ≤ y ∧ y < 4) := by
  sorry

end NUMINAMATH_CALUDE_system_equations_properties_l3541_354110


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3541_354119

theorem quadratic_solution_sum (c d : ℝ) : 
  (5 * (c + d * I) ^ 2 + 4 * (c + d * I) + 20 = 0) ∧ 
  (5 * (c - d * I) ^ 2 + 4 * (c - d * I) + 20 = 0) →
  c + d ^ 2 = 86 / 25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3541_354119
