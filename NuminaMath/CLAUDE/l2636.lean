import Mathlib

namespace nine_digit_sum_l2636_263683

-- Define the type for digits 1-9
def Digit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

-- Define the structure for the nine-digit number
structure NineDigitNumber where
  digits : Fin 9 → Digit
  distinct : ∀ i j, i ≠ j → digits i ≠ digits j

-- Define the property that each two-digit segment is a product of two single-digit numbers
def validSegments (n : NineDigitNumber) : Prop :=
  ∀ i : Fin 8, ∃ (x y : Digit), 
    (n.digits i).val * 10 + (n.digits (i + 1)).val = x.val * y.val

-- Define the function to calculate the sum of ABC + DEF + GHI
def sumSegments (n : NineDigitNumber) : ℕ :=
  ((n.digits 0).val * 100 + (n.digits 1).val * 10 + (n.digits 2).val) +
  ((n.digits 3).val * 100 + (n.digits 4).val * 10 + (n.digits 5).val) +
  ((n.digits 6).val * 100 + (n.digits 7).val * 10 + (n.digits 8).val)

-- State the theorem
theorem nine_digit_sum (n : NineDigitNumber) (h : validSegments n) : 
  sumSegments n = 1440 :=
sorry

end nine_digit_sum_l2636_263683


namespace greatest_integer_solution_two_satisfies_inequality_three_exceeds_inequality_greatest_integer_value_l2636_263630

theorem greatest_integer_solution (x : ℤ) : x^2 + 5*x < 30 → x ≤ 2 :=
by
  sorry

theorem two_satisfies_inequality : 2^2 + 5*2 < 30 :=
by
  sorry

theorem three_exceeds_inequality : ¬(3^2 + 5*3 < 30) :=
by
  sorry

theorem greatest_integer_value : ∃ (x : ℤ), x^2 + 5*x < 30 ∧ ∀ (y : ℤ), y^2 + 5*y < 30 → y ≤ x :=
by
  sorry

end greatest_integer_solution_two_satisfies_inequality_three_exceeds_inequality_greatest_integer_value_l2636_263630


namespace earnings_difference_l2636_263604

/-- Represents the delivery areas --/
inductive DeliveryArea
  | A
  | B
  | C

/-- Represents a delivery worker --/
structure DeliveryWorker where
  name : String
  deliveries : DeliveryArea → Nat

/-- Get the fee for a specific delivery area --/
def areaFee (area : DeliveryArea) : Nat :=
  match area with
  | DeliveryArea.A => 100
  | DeliveryArea.B => 125
  | DeliveryArea.C => 150

/-- Calculate the total earnings for a worker --/
def totalEarnings (worker : DeliveryWorker) : Nat :=
  (worker.deliveries DeliveryArea.A * areaFee DeliveryArea.A) +
  (worker.deliveries DeliveryArea.B * areaFee DeliveryArea.B) +
  (worker.deliveries DeliveryArea.C * areaFee DeliveryArea.C)

/-- Oula's delivery data --/
def oula : DeliveryWorker :=
  { name := "Oula"
    deliveries := fun
      | DeliveryArea.A => 48
      | DeliveryArea.B => 32
      | DeliveryArea.C => 16 }

/-- Tona's delivery data --/
def tona : DeliveryWorker :=
  { name := "Tona"
    deliveries := fun
      | DeliveryArea.A => 27
      | DeliveryArea.B => 18
      | DeliveryArea.C => 9 }

/-- The main theorem to prove --/
theorem earnings_difference : totalEarnings oula - totalEarnings tona = 4900 := by
  sorry


end earnings_difference_l2636_263604


namespace fifth_term_zero_l2636_263656

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fifth_term_zero
  (a : ℕ → ℚ)
  (x y : ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 0 = x + 2*y)
  (h_second : a 1 = x - 2*y)
  (h_third : a 2 = x^2 - 4*y^2)
  (h_fourth : a 3 = x / (2*y))
  (h_x : x = 1/2)
  (h_y : y = 1/4)
  : a 4 = 0 := by
  sorry

end fifth_term_zero_l2636_263656


namespace pokemon_card_collection_l2636_263681

def cards_needed (michael_cards : ℕ) (mark_diff : ℕ) (lloyd_ratio : ℕ) (total_goal : ℕ) : ℕ :=
  let mark_cards := michael_cards - mark_diff
  let lloyd_cards := mark_cards / lloyd_ratio
  total_goal - (michael_cards + mark_cards + lloyd_cards)

theorem pokemon_card_collection : 
  cards_needed 100 10 3 300 = 80 := by
  sorry

end pokemon_card_collection_l2636_263681


namespace age_ratio_change_l2636_263631

theorem age_ratio_change (father_age : ℕ) (man_age : ℕ) (years : ℕ) : 
  father_age = 60 → 
  man_age = (2 * father_age) / 5 → 
  (man_age + years) * 2 = father_age + years → 
  years = 12 := by
sorry

end age_ratio_change_l2636_263631


namespace other_diagonal_length_l2636_263618

/-- A trapezoid with diagonals intersecting at a right angle -/
structure RightAngleDiagonalTrapezoid where
  midline : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  diagonals_perpendicular : diagonal1 * diagonal2 = midline * midline * 4

/-- Theorem: In a trapezoid with diagonals intersecting at a right angle,
    if the midline is 6.5 and one diagonal is 12, then the other diagonal is 5 -/
theorem other_diagonal_length
  (t : RightAngleDiagonalTrapezoid)
  (h1 : t.midline = 6.5)
  (h2 : t.diagonal1 = 12) :
  t.diagonal2 = 5 := by
  sorry

end other_diagonal_length_l2636_263618


namespace partial_fraction_decomposition_l2636_263639

theorem partial_fraction_decomposition (N₁ N₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 3 → (42 * x - 37) / (x^2 - 4*x + 3) = N₁ / (x - 1) + N₂ / (x - 3)) →
  N₁ * N₂ = -445/4 := by
sorry

end partial_fraction_decomposition_l2636_263639


namespace complex_equality_modulus_l2636_263607

theorem complex_equality_modulus (x y : ℝ) (i : ℂ) : 
  i * i = -1 →
  (2 + i) * (3 - x * i) = 3 + (y + 5) * i →
  Complex.abs (x + y * i) = 5 := by
  sorry

end complex_equality_modulus_l2636_263607


namespace sqrt_x_minus_3_real_l2636_263626

theorem sqrt_x_minus_3_real (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 3) ↔ x ≥ 3 := by sorry

end sqrt_x_minus_3_real_l2636_263626


namespace sons_age_l2636_263655

/-- Proves that the son's current age is 16 years given the specified conditions -/
theorem sons_age (son_age father_age : ℕ) : 
  father_age = 4 * son_age →
  (son_age - 10) + (father_age - 10) = 60 →
  son_age = 16 := by
  sorry

end sons_age_l2636_263655


namespace solution_replacement_l2636_263680

theorem solution_replacement (initial_conc : ℚ) (replacing_conc : ℚ) (final_conc : ℚ) 
  (h1 : initial_conc = 70/100)
  (h2 : replacing_conc = 25/100)
  (h3 : final_conc = 35/100) :
  ∃ (x : ℚ), x = 7/9 ∧ initial_conc * (1 - x) + replacing_conc * x = final_conc :=
by sorry

end solution_replacement_l2636_263680


namespace distance_to_SFL_is_81_miles_l2636_263613

/-- The distance to Super Fun-tastic Land -/
def distance_to_SFL (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: The distance to Super Fun-tastic Land is 81 miles -/
theorem distance_to_SFL_is_81_miles :
  distance_to_SFL 27 3 = 81 := by
  sorry

end distance_to_SFL_is_81_miles_l2636_263613


namespace fraction_problem_l2636_263623

theorem fraction_problem (N : ℝ) (F : ℝ) (h : F * (1/3 * N) = 30) :
  ∃ G : ℝ, G * N = 75 ∧ G = 5/6 := by
  sorry

end fraction_problem_l2636_263623


namespace euler_line_parallel_l2636_263642

/-- Triangle ABC with vertices A(-3,0), B(3,0), and C(3,3) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {⟨-3, 0⟩, ⟨3, 0⟩, ⟨3, 3⟩}

/-- The Euler line of a triangle -/
def euler_line (t : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry

/-- A line with equation ax + (a^2 - 3)y - 9 = 0 -/
def line_l (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + (a^2 - 3) * p.2 - 9 = 0}

/-- Two lines are parallel -/
def parallel (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  sorry

theorem euler_line_parallel :
  ∀ a : ℝ, parallel (line_l a) (euler_line triangle_ABC) ↔ a = -1 :=
by sorry

end euler_line_parallel_l2636_263642


namespace max_value_product_l2636_263648

theorem max_value_product (a b c x y z : ℝ) 
  (non_neg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z)
  (sum_abc : a + b + c = 1)
  (sum_xyz : x + y + z = 1) :
  (a - x^2) * (b - y^2) * (c - z^2) ≤ 1/16 :=
by sorry

end max_value_product_l2636_263648


namespace cube_64_sqrt_is_plus_minus_2_l2636_263602

theorem cube_64_sqrt_is_plus_minus_2 (x : ℝ) (h : x^3 = 64) : 
  Real.sqrt x = 2 ∨ Real.sqrt x = -2 := by
sorry

end cube_64_sqrt_is_plus_minus_2_l2636_263602


namespace amy_remaining_money_l2636_263619

theorem amy_remaining_money (initial_amount : ℝ) 
  (doll_cost doll_quantity : ℝ)
  (board_game_cost board_game_quantity : ℝ)
  (comic_book_cost comic_book_quantity : ℝ) :
  initial_amount = 100 ∧
  doll_cost = 1.25 ∧
  doll_quantity = 3 ∧
  board_game_cost = 12.75 ∧
  board_game_quantity = 2 ∧
  comic_book_cost = 3.50 ∧
  comic_book_quantity = 4 →
  initial_amount - (doll_cost * doll_quantity + board_game_cost * board_game_quantity + comic_book_cost * comic_book_quantity) = 56.75 := by
sorry

end amy_remaining_money_l2636_263619


namespace adam_received_one_smiley_l2636_263670

/-- Represents the number of smileys each friend received -/
structure SmileyCounts where
  adam : ℕ
  mojmir : ℕ
  petr : ℕ
  pavel : ℕ

/-- The conditions of the problem -/
def validSmileyCounts (counts : SmileyCounts) : Prop :=
  counts.adam + counts.mojmir + counts.petr + counts.pavel = 52 ∧
  counts.adam ≥ 1 ∧
  counts.mojmir ≥ 1 ∧
  counts.petr ≥ 1 ∧
  counts.pavel ≥ 1 ∧
  counts.petr + counts.pavel = 33 ∧
  counts.mojmir > counts.adam ∧
  counts.mojmir > counts.petr ∧
  counts.mojmir > counts.pavel

theorem adam_received_one_smiley (counts : SmileyCounts) 
  (h : validSmileyCounts counts) : counts.adam = 1 := by
  sorry

end adam_received_one_smiley_l2636_263670


namespace newspaper_prices_l2636_263611

theorem newspaper_prices :
  ∃ (x y : ℕ) (k : ℚ),
    x < 30 ∧ y < 30 ∧ 0 < k ∧ k < 1 ∧
    k * 30 = y ∧ k * x = 15 ∧
    ((x = 25 ∧ y = 18) ∨ (x = 18 ∧ y = 25)) ∧
    ∀ (x' y' : ℕ) (k' : ℚ),
      x' < 30 → y' < 30 → 0 < k' → k' < 1 →
      k' * 30 = y' → k' * x' = 15 →
      ((x' = 25 ∧ y' = 18) ∨ (x' = 18 ∧ y' = 25)) :=
by sorry

end newspaper_prices_l2636_263611


namespace profit_is_27000_l2636_263652

/-- Represents the profit sharing problem between Tom and Jose -/
structure ProfitSharing where
  tom_investment : ℕ
  tom_months : ℕ
  jose_investment : ℕ
  jose_months : ℕ
  jose_profit : ℕ

/-- Calculates the total profit earned by Tom and Jose -/
def total_profit (ps : ProfitSharing) : ℕ :=
  let tom_total := ps.tom_investment * ps.tom_months
  let jose_total := ps.jose_investment * ps.jose_months
  let ratio_sum := (tom_total / (tom_total.gcd jose_total)) + (jose_total / (tom_total.gcd jose_total))
  (ratio_sum * ps.jose_profit) / (jose_total / (tom_total.gcd jose_total))

/-- Theorem stating that the total profit is 27000 for the given conditions -/
theorem profit_is_27000 (ps : ProfitSharing)
  (h1 : ps.tom_investment = 30000)
  (h2 : ps.tom_months = 12)
  (h3 : ps.jose_investment = 45000)
  (h4 : ps.jose_months = 10)
  (h5 : ps.jose_profit = 15000) :
  total_profit ps = 27000 := by
  sorry

#eval total_profit { tom_investment := 30000, tom_months := 12, jose_investment := 45000, jose_months := 10, jose_profit := 15000 }

end profit_is_27000_l2636_263652


namespace distinct_collections_count_l2636_263665

/-- Represents the number of each letter in BIOLOGY --/
structure LetterCount where
  o : Nat
  i : Nat
  y : Nat
  b : Nat
  g : Nat

/-- The initial count of letters in BIOLOGY --/
def initial_count : LetterCount :=
  { o := 2, i := 1, y := 1, b := 1, g := 2 }

/-- A collection of letters that can be put in the bag --/
structure BagCollection where
  vowels : Nat
  consonants : Nat

/-- Check if a collection is valid (3 vowels and 2 consonants) --/
def is_valid_collection (c : BagCollection) : Prop :=
  c.vowels = 3 ∧ c.consonants = 2

/-- Count the number of distinct vowel combinations --/
def count_vowel_combinations (lc : LetterCount) : Nat :=
  sorry

/-- Count the number of distinct consonant combinations --/
def count_consonant_combinations (lc : LetterCount) : Nat :=
  sorry

/-- The main theorem: there are 12 distinct possible collections --/
theorem distinct_collections_count :
  count_vowel_combinations initial_count * count_consonant_combinations initial_count = 12 :=
sorry

end distinct_collections_count_l2636_263665


namespace always_possible_to_sell_tickets_l2636_263622

/-- Represents the amount a child pays (5 or 10 yuan) -/
inductive Payment
| five : Payment
| ten : Payment

/-- A queue of children represented by their payments -/
def Queue := List Payment

/-- Counts the number of each type of payment in a queue -/
def countPayments (q : Queue) : ℕ × ℕ :=
  q.foldl (λ (five, ten) p => match p with
    | Payment.five => (five + 1, ten)
    | Payment.ten => (five, ten + 1)
  ) (0, 0)

/-- Checks if it's possible to give change at each step -/
def canGiveChange (q : Queue) : Prop :=
  q.foldl (λ acc p => match p with
    | Payment.five => acc + 1
    | Payment.ten => acc - 1
  ) 0 ≥ 0

/-- The main theorem stating that it's always possible to sell tickets without running out of change -/
theorem always_possible_to_sell_tickets (q : Queue) :
  let (fives, tens) := countPayments q
  fives = tens → q.length = 2 * fives → canGiveChange q :=
sorry

#check always_possible_to_sell_tickets

end always_possible_to_sell_tickets_l2636_263622


namespace smallest_divisible_by_1_to_10_l2636_263684

theorem smallest_divisible_by_1_to_10 : ∃ (n : ℕ), n > 0 ∧ (∀ i : ℕ, i ∈ Finset.range 10 → i.succ ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ i : ℕ, i ∈ Finset.range 10 → i.succ ∣ m) → n ≤ m) :=
  sorry

end smallest_divisible_by_1_to_10_l2636_263684


namespace divisibility_condition_l2636_263694

theorem divisibility_condition (n : ℤ) : (3 * n + 7) ∣ (5 * n + 13) ↔ n ∈ ({-3, -2, -1} : Set ℤ) := by
  sorry

end divisibility_condition_l2636_263694


namespace expand_product_l2636_263633

theorem expand_product (x : ℝ) : (2*x + 3) * (4*x - 5) = 8*x^2 + 2*x - 15 := by
  sorry

end expand_product_l2636_263633


namespace triangle_area_is_16_l2636_263674

/-- The area of a triangle formed by three lines in a 2D plane --/
def triangleArea (line1 line2 line3 : ℝ → ℝ) : ℝ :=
  sorry

/-- The first line: y = 6 --/
def line1 (x : ℝ) : ℝ := 6

/-- The second line: y = 2 + x --/
def line2 (x : ℝ) : ℝ := 2 + x

/-- The third line: y = 2 - x --/
def line3 (x : ℝ) : ℝ := 2 - x

theorem triangle_area_is_16 : triangleArea line1 line2 line3 = 16 := by
  sorry

end triangle_area_is_16_l2636_263674


namespace tangent_line_to_parabola_l2636_263696

theorem tangent_line_to_parabola (x y : ℝ) :
  let f : ℝ → ℝ := λ t => t^2
  let tangent_point : ℝ × ℝ := (1, 1)
  let slope : ℝ := 2 * tangent_point.1
  2 * x - y - 1 = 0 ↔ y = slope * (x - tangent_point.1) + tangent_point.2 :=
by sorry

end tangent_line_to_parabola_l2636_263696


namespace double_inequality_proof_l2636_263616

theorem double_inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let f := (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1)))
  (0 < f) ∧ 
  (f ≤ 1/8) ∧ 
  (f = 1/8 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry


end double_inequality_proof_l2636_263616


namespace next_birthday_age_is_56_l2636_263697

/-- Represents a person's age in years, months, weeks, and days -/
structure Age where
  years : ℕ
  months : ℕ
  weeks : ℕ
  days : ℕ

/-- Calculates the age on the next birthday given a current age -/
def nextBirthdayAge (currentAge : Age) : ℕ :=
  sorry

/-- Theorem stating that given the specific age, the next birthday age will be 56 -/
theorem next_birthday_age_is_56 :
  let currentAge : Age := { years := 50, months := 50, weeks := 50, days := 50 }
  nextBirthdayAge currentAge = 56 := by
  sorry

end next_birthday_age_is_56_l2636_263697


namespace second_rectangle_perimeter_l2636_263605

theorem second_rectangle_perimeter (a b : ℝ) : 
  (a + 3) * (b + 3) - a * b = 48 →
  2 * ((a + 3) + (b + 3)) = 38 := by
sorry

end second_rectangle_perimeter_l2636_263605


namespace five_digit_divisibility_l2636_263664

def is_valid_digit (d : ℕ) : Prop := d ∈ ({2, 3, 4, 5, 6} : Set ℕ)

def digits_to_number (p q r s t : ℕ) : ℕ := p * 10000 + q * 1000 + r * 100 + s * 10 + t

theorem five_digit_divisibility (p q r s t : ℕ) :
  is_valid_digit p ∧ is_valid_digit q ∧ is_valid_digit r ∧ is_valid_digit s ∧ is_valid_digit t →
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t →
  (p * 100 + q * 10 + r) % 6 = 0 →
  (q * 100 + r * 10 + s) % 3 = 0 →
  (r * 100 + s * 10 + t) % 9 = 0 →
  p = 3 := by
  sorry

end five_digit_divisibility_l2636_263664


namespace point_on_x_axis_l2636_263658

/-- 
A point M with coordinates (m-1, 2m) lies on the x-axis if and only if 
its coordinates are (-1, 0).
-/
theorem point_on_x_axis (m : ℝ) : 
  (m - 1, 2 * m) ∈ {p : ℝ × ℝ | p.2 = 0} ↔ (m - 1, 2 * m) = (-1, 0) := by
  sorry

end point_on_x_axis_l2636_263658


namespace tourists_escape_theorem_l2636_263685

/-- Represents the color of a hat -/
inductive HatColor
  | Black
  | White

/-- Represents a tourist in the line -/
structure Tourist where
  position : Nat
  hatColor : HatColor

/-- Represents the line of tourists -/
def TouristLine := List Tourist

/-- A strategy is a function that takes the visible hats and previous guesses
    and returns a guess for the current tourist's hat color -/
def Strategy := (visibleHats : List HatColor) → (previousGuesses : List HatColor) → HatColor

/-- Applies the strategy to a line of tourists and returns the number of correct guesses -/
def applyStrategy (line : TouristLine) (strategy : Strategy) : Nat :=
  sorry

/-- There exists a strategy that guarantees at least 9 out of 10 tourists can correctly guess their hat color -/
theorem tourists_escape_theorem :
  ∃ (strategy : Strategy),
    ∀ (line : TouristLine),
      line.length = 10 →
      applyStrategy line strategy ≥ 9 :=
sorry

end tourists_escape_theorem_l2636_263685


namespace sequence_general_term_l2636_263614

theorem sequence_general_term (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  a 1 = 2 →
  (∀ n, a (n + 1)^2 = (a n)^2 + 2) →
  ∀ n, a n = Real.sqrt (2 * n + 2) :=
by sorry

end sequence_general_term_l2636_263614


namespace initial_men_count_l2636_263628

/-- Represents a road construction project --/
structure RoadProject where
  length : ℝ  -- Length of the road in km
  duration : ℝ  -- Total duration of the project in days
  initialProgress : ℝ  -- Length of road completed after 10 days
  initialDays : ℝ  -- Number of days for initial progress
  extraMen : ℕ  -- Number of extra men needed to finish on time

/-- Calculates the initial number of men employed in the project --/
def initialMen (project : RoadProject) : ℕ :=
  sorry

/-- Theorem stating the initial number of men for the given project --/
theorem initial_men_count (project : RoadProject) 
  (h1 : project.length = 10)
  (h2 : project.duration = 30)
  (h3 : project.initialProgress = 2)
  (h4 : project.initialDays = 10)
  (h5 : project.extraMen = 30) :
  initialMen project = 75 :=
sorry

end initial_men_count_l2636_263628


namespace system1_solution_system2_solution_l2636_263649

-- System 1
theorem system1_solution :
  ∃ (x y : ℝ), x + y = 2 ∧ 5 * x - 2 * (x + y) = 6 ∧ x = 2 ∧ y = 0 := by sorry

-- System 2
theorem system2_solution :
  ∃ (a b c : ℝ), a + b = 3 ∧ 5 * a + 3 * c = 1 ∧ a + b + c = 0 ∧ a = 2 ∧ b = 1 ∧ c = -3 := by sorry

end system1_solution_system2_solution_l2636_263649


namespace sandwich_combinations_count_l2636_263699

/-- Represents the number of toppings available -/
def num_toppings : ℕ := 7

/-- Represents the number of bread types available -/
def num_bread_types : ℕ := 3

/-- Represents the number of filling types available -/
def num_filling_types : ℕ := 3

/-- Represents the maximum number of filling layers -/
def max_filling_layers : ℕ := 2

/-- Calculates the total number of sandwich combinations -/
def total_sandwich_combinations : ℕ :=
  (2^num_toppings) * num_bread_types * (num_filling_types + num_filling_types^2)

/-- Theorem stating that the total number of sandwich combinations is 4608 -/
theorem sandwich_combinations_count :
  total_sandwich_combinations = 4608 := by sorry

end sandwich_combinations_count_l2636_263699


namespace xiao_ming_arrival_time_l2636_263688

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : minutes < 60 := by sorry

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60
  , minutes := totalMinutes % 60 }

theorem xiao_ming_arrival_time 
  (departure_time : Time)
  (journey_duration : Nat)
  (h1 : departure_time.hours = 6)
  (h2 : departure_time.minutes = 55)
  (h3 : journey_duration = 30) :
  addMinutes departure_time journey_duration = { hours := 7, minutes := 25 } := by
  sorry

end xiao_ming_arrival_time_l2636_263688


namespace max_silver_tokens_l2636_263617

/-- Represents the state of tokens Alex has -/
structure TokenState where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules -/
inductive ExchangeRule
  | RedToSilver : ExchangeRule  -- 3 red for 1 silver and 2 blue
  | BlueToSilver : ExchangeRule -- 4 blue for 1 silver and 2 red

/-- Applies an exchange rule to a token state -/
def applyExchange (state : TokenState) (rule : ExchangeRule) : Option TokenState :=
  match rule with
  | ExchangeRule.RedToSilver =>
      if state.red ≥ 3 then
        some ⟨state.red - 3, state.blue + 2, state.silver + 1⟩
      else
        none
  | ExchangeRule.BlueToSilver =>
      if state.blue ≥ 4 then
        some ⟨state.red + 2, state.blue - 4, state.silver + 1⟩
      else
        none

/-- Checks if any exchange is possible -/
def canExchange (state : TokenState) : Bool :=
  state.red ≥ 3 ∨ state.blue ≥ 4

/-- The main theorem to prove -/
theorem max_silver_tokens (initialState : TokenState) 
    (h_initial : initialState = ⟨100, 100, 0⟩) :
    ∃ (finalState : TokenState), 
      (¬canExchange finalState) ∧ 
      (finalState.silver = 88) ∧
      (∃ (exchanges : List ExchangeRule), 
        finalState = exchanges.foldl (λ s r => (applyExchange s r).getD s) initialState) :=
  sorry


end max_silver_tokens_l2636_263617


namespace no_quadratic_trinomials_with_integer_roots_l2636_263627

theorem no_quadratic_trinomials_with_integer_roots : 
  ¬ ∃ (a b c x₁ x₂ x₃ x₄ : ℤ), 
    (∀ x : ℤ, a * x^2 + b * x + c = a * (x - x₁) * (x - x₂)) ∧ 
    (∀ x : ℤ, (a + 1) * x^2 + (b + 1) * x + (c + 1) = (a + 1) * (x - x₃) * (x - x₄)) := by
  sorry

end no_quadratic_trinomials_with_integer_roots_l2636_263627


namespace problem_solution_l2636_263603

theorem problem_solution : 
  (((Real.sqrt 48) / (Real.sqrt 3) - (Real.sqrt (1/2)) * (Real.sqrt 12) + (Real.sqrt 24)) = 4 + Real.sqrt 6) ∧
  ((Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) + (Real.sqrt 3 - 1)^2 = 7 - 2 * Real.sqrt 3) := by
  sorry

end problem_solution_l2636_263603


namespace min_value_quadratic_function_l2636_263645

/-- Given a quadratic function f(x) = ax² - 4x + c with range [0, +∞),
    prove that the minimum value of 1/c + 9/a is 3 -/
theorem min_value_quadratic_function (a c : ℝ) (h_pos_a : a > 0) (h_pos_c : c > 0)
  (h_range : Set.range (fun x => a * x^2 - 4*x + c) = Set.Ici 0)
  (h_ac : a * c = 4) :
  (∀ y, 1/c + 9/a ≥ y) ∧ (∃ y, 1/c + 9/a = y) ∧ y = 3 := by
  sorry

end min_value_quadratic_function_l2636_263645


namespace product_equality_l2636_263600

theorem product_equality : 2.5 * 8.5 * (5.2 - 0.2) = 106.25 := by
  sorry

end product_equality_l2636_263600


namespace intersection_of_M_and_N_l2636_263615

def M : Set ℝ := {x | (x + 2) * (x - 1) < 0}
def N : Set ℝ := {x | x + 1 < 0}

theorem intersection_of_M_and_N : M ∩ N = Set.Ioo (-2) (-1) := by sorry

end intersection_of_M_and_N_l2636_263615


namespace remaining_rolls_to_sell_l2636_263676

/-- Calculates the remaining rolls of gift wrap Nellie needs to sell -/
theorem remaining_rolls_to_sell 
  (total_rolls : ℕ) 
  (sold_to_grandmother : ℕ) 
  (sold_to_uncle : ℕ) 
  (sold_to_neighbor : ℕ) 
  (h1 : total_rolls = 45)
  (h2 : sold_to_grandmother = 1)
  (h3 : sold_to_uncle = 10)
  (h4 : sold_to_neighbor = 6) :
  total_rolls - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 28 := by
  sorry

end remaining_rolls_to_sell_l2636_263676


namespace quadratic_vertex_form_h_l2636_263678

/-- Given a quadratic expression 3x^2 + 9x + 20, prove that when written in the form a(x - h)^2 + k, the value of h is -3/2. -/
theorem quadratic_vertex_form_h (x : ℝ) :
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k :=
sorry

end quadratic_vertex_form_h_l2636_263678


namespace median_triangle_inequalities_l2636_263637

-- Define a structure for a triangle with angles
structure Triangle where
  α : Real
  β : Real
  γ : Real

-- Define a structure for a triangle formed from medians
structure MedianTriangle where
  α_m : Real
  β_m : Real
  γ_m : Real

-- Main theorem
theorem median_triangle_inequalities (T : Triangle) (M : MedianTriangle)
  (h1 : T.α > T.β)
  (h2 : T.β > T.γ)
  : T.α > M.α_m ∧
    T.α > M.β_m ∧
    M.γ_m > T.β ∧
    T.β > M.α_m ∧
    M.β_m > T.γ ∧
    M.γ_m > T.γ := by
  sorry

end median_triangle_inequalities_l2636_263637


namespace alexandra_rearrangement_time_l2636_263601

/-- The number of letters in Alexandra's name -/
def name_length : ℕ := 8

/-- The number of rearrangements Alexandra can write per minute -/
def rearrangements_per_minute : ℕ := 16

/-- Calculate the time required to write all rearrangements in hours -/
def time_to_write_all_rearrangements : ℕ :=
  (Nat.factorial name_length / rearrangements_per_minute) / 60

theorem alexandra_rearrangement_time :
  time_to_write_all_rearrangements = 42 := by sorry

end alexandra_rearrangement_time_l2636_263601


namespace intersection_A_B_l2636_263650

-- Define set A
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 6}

-- Define set B
def B : Set ℝ := {x | 3 * x^2 + x - 8 ≤ 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 4/3} := by
  sorry

end intersection_A_B_l2636_263650


namespace tree_planting_event_l2636_263662

theorem tree_planting_event (boys girls : ℕ) : 
  girls = boys + 400 →
  girls > boys →
  (60 : ℚ) / 100 * (boys + girls : ℚ) = 960 →
  boys = 600 := by
sorry

end tree_planting_event_l2636_263662


namespace greatest_integer_value_l2636_263621

theorem greatest_integer_value (x : ℤ) : 
  (∀ y : ℤ, y > x → ¬(∃ z : ℤ, (y^2 + 5*y + 6) / (y - 2) = z)) →
  (∃ z : ℤ, (x^2 + 5*x + 6) / (x - 2) = z) →
  x = 22 :=
by sorry

end greatest_integer_value_l2636_263621


namespace shaded_area_division_l2636_263689

/-- Represents a grid in the first quadrant -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)
  (shaded_squares : ℕ)

/-- Represents a line passing through (0,0) and (8,c) -/
structure Line :=
  (c : ℝ)

/-- Checks if a line divides the shaded area of a grid into two equal parts -/
def divides_equally (g : Grid) (l : Line) : Prop :=
  ∃ (area : ℝ), area > 0 ∧ area * 2 = g.shaded_squares

theorem shaded_area_division (g : Grid) (l : Line) :
  g.width = 8 ∧ g.height = 6 ∧ g.shaded_squares = 32 →
  divides_equally g l ↔ l.c = 4 :=
sorry

end shaded_area_division_l2636_263689


namespace hotdog_cost_l2636_263666

/-- The total cost of hot dogs given the number of hot dogs and the price per hot dog. -/
def total_cost (num_hotdogs : ℕ) (price_per_hotdog : ℚ) : ℚ :=
  num_hotdogs * price_per_hotdog

/-- Theorem stating that the total cost of 6 hot dogs at 50 cents each is $3.00 -/
theorem hotdog_cost : total_cost 6 (50 / 100) = 3 := by
  sorry

end hotdog_cost_l2636_263666


namespace circle_E_equation_l2636_263687

-- Define the circle E
def circle_E (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the condition that E passes through A(0,0) and B(1,1)
def passes_through_A_and_B (E : Set (ℝ × ℝ)) : Prop :=
  (0, 0) ∈ E ∧ (1, 1) ∈ E

-- Define the three additional conditions
def condition_1 (E : Set (ℝ × ℝ)) : Prop :=
  (2, 0) ∈ E

def condition_2 (E : Set (ℝ × ℝ)) : Prop :=
  ∀ m : ℝ, ∃ p q : ℝ × ℝ, p ∈ E ∧ q ∈ E ∧
    p.2 = m * (p.1 - 1) ∧ q.2 = m * (q.1 - 1) ∧
    p ≠ q

def condition_3 (E : Set (ℝ × ℝ)) : Prop :=
  ∃ y : ℝ, (0, y) ∈ E ∧ ∀ t : ℝ, t ≠ y → (0, t) ∉ E

-- The main theorem
theorem circle_E_equation :
  ∀ E : Set (ℝ × ℝ),
  passes_through_A_and_B E →
  (condition_1 E ∨ condition_2 E ∨ condition_3 E) →
  E = circle_E (1, 0) 1 :=
sorry

end circle_E_equation_l2636_263687


namespace quadratic_max_value_l2636_263646

theorem quadratic_max_value (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ -3 * x^2 + 15 * x + 9
  ∃ (max : ℝ), max = (111 : ℝ) / 4 ∧ ∀ y, f y ≤ max :=
by sorry

end quadratic_max_value_l2636_263646


namespace total_distance_walked_and_run_l2636_263632

/-- Calculates the total distance traveled when walking and running at given rates and times. -/
theorem total_distance_walked_and_run
  (walking_time : ℝ) (walking_rate : ℝ) (running_time : ℝ) (running_rate : ℝ) :
  walking_time = 30 / 60 →
  walking_rate = 3.5 →
  running_time = 45 / 60 →
  running_rate = 8 →
  walking_time * walking_rate + running_time * running_rate = 7.75 := by
  sorry

#check total_distance_walked_and_run

end total_distance_walked_and_run_l2636_263632


namespace intersection_M_N_l2636_263673

open Set

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the complement of N in U
def complementN : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := U \ complementN

-- Theorem statement
theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 0} := by
  sorry

end intersection_M_N_l2636_263673


namespace not_prime_p_l2636_263679

theorem not_prime_p (x k : ℕ) (p : ℕ) (h : x^5 + 2*x + 3 = p*k) : ¬ Nat.Prime p := by
  sorry

end not_prime_p_l2636_263679


namespace incorrect_statement_proof_l2636_263672

structure VisionSurvey where
  total_students : Nat
  sample_size : Nat
  is_about_vision : Bool

def is_correct_statement (s : VisionSurvey) (statement : String) : Prop :=
  match statement with
  | "The sample size is correct" => s.sample_size = 40
  | "The sample is about vision of selected students" => s.is_about_vision
  | "The population is about vision of all students" => s.is_about_vision
  | "The individual refers to each student" => false
  | _ => false

theorem incorrect_statement_proof (s : VisionSurvey) 
  (h1 : s.total_students = 400) 
  (h2 : s.sample_size = 40) 
  (h3 : s.is_about_vision = true) :
  ¬(is_correct_statement s "The individual refers to each student") := by
  sorry

end incorrect_statement_proof_l2636_263672


namespace largest_last_digit_is_two_l2636_263686

/-- A string of digits satisfying the given conditions -/
structure SpecialString :=
  (digits : Fin 1003 → Nat)
  (first_digit : digits 0 = 2)
  (consecutive_divisible : ∀ i : Fin 1002, 
    (digits i * 10 + digits (i.succ)) % 17 = 0 ∨ 
    (digits i * 10 + digits (i.succ)) % 23 = 0)

/-- The largest possible last digit in the special string -/
def largest_last_digit : Nat := 2

/-- Theorem stating that the largest possible last digit is 2 -/
theorem largest_last_digit_is_two :
  ∀ s : SpecialString, s.digits 1002 ≤ largest_last_digit :=
sorry

end largest_last_digit_is_two_l2636_263686


namespace power_equation_solution_l2636_263693

theorem power_equation_solution (x y : ℕ) :
  (3 : ℝ) ^ x * (4 : ℝ) ^ y = 59049 ∧ x = 10 → x - y = 10 := by
  sorry

end power_equation_solution_l2636_263693


namespace bus_ride_difference_l2636_263608

-- Define the lengths of the bus rides
def oscar_ride : ℝ := 0.75
def charlie_ride : ℝ := 0.25

-- Theorem statement
theorem bus_ride_difference : oscar_ride - charlie_ride = 0.50 := by
  sorry

end bus_ride_difference_l2636_263608


namespace union_when_a_is_neg_two_intersection_equals_B_iff_l2636_263638

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 6}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a - 1 ≤ x ∧ x ≤ a + 1}

-- Theorem 1: When a = -2, A ∪ B = {x | -5 ≤ x ≤ 6}
theorem union_when_a_is_neg_two :
  A ∪ B (-2) = {x : ℝ | -5 ≤ x ∧ x ≤ 6} := by sorry

-- Theorem 2: A ∩ B = B if and only if a ≥ -1
theorem intersection_equals_B_iff (a : ℝ) :
  A ∩ B a = B a ↔ a ≥ -1 := by sorry

end union_when_a_is_neg_two_intersection_equals_B_iff_l2636_263638


namespace perpendicular_line_x_intercept_l2636_263657

/-- Given a line L1 with equation 4x + 5y = 10 and a perpendicular line L2 with y-intercept -3,
    the x-intercept of L2 is 12/5 -/
theorem perpendicular_line_x_intercept :
  ∀ (L1 L2 : Set (ℝ × ℝ)),
  (∀ x y, (x, y) ∈ L1 ↔ 4 * x + 5 * y = 10) →
  (∃ m : ℝ, ∀ x y, (x, y) ∈ L2 ↔ y = m * x - 3) →
  (∀ x y₁ y₂, (x, y₁) ∈ L1 ∧ (x, y₂) ∈ L2 → (y₂ - y₁) * (4 * (x + 1) + 5 * y₁ - 10) = 0) →
  (0, -3) ∈ L2 →
  (12/5, 0) ∈ L2 :=
by sorry

end perpendicular_line_x_intercept_l2636_263657


namespace two_false_propositions_l2636_263659

-- Define the original proposition
def original_prop (a : ℝ) : Prop := a > -3 → a > -6

-- Define the converse proposition
def converse_prop (a : ℝ) : Prop := a > -6 → a > -3

-- Define the inverse proposition
def inverse_prop (a : ℝ) : Prop := a ≤ -3 → a ≤ -6

-- Define the contrapositive proposition
def contrapositive_prop (a : ℝ) : Prop := a ≤ -6 → a ≤ -3

-- Theorem statement
theorem two_false_propositions :
  ∃ (f : Fin 4 → Prop), 
    (∀ a : ℝ, f 0 = original_prop a ∧ 
              f 1 = converse_prop a ∧ 
              f 2 = inverse_prop a ∧ 
              f 3 = contrapositive_prop a) ∧
    (∃! (i j : Fin 4), i ≠ j ∧ ¬(f i) ∧ ¬(f j) ∧ 
      ∀ (k : Fin 4), k ≠ i ∧ k ≠ j → f k) :=
by
  sorry

end two_false_propositions_l2636_263659


namespace triangle_perimeter_when_area_equals_four_inradius_l2636_263695

/-- Given a triangle with an inscribed circle, if the area of the triangle is numerically equal to
    four times the radius of the inscribed circle, then the perimeter of the triangle is 8. -/
theorem triangle_perimeter_when_area_equals_four_inradius (A r s p : ℝ) :
  A > 0 → r > 0 → s > 0 → p > 0 →
  A = r * s →  -- Area formula using inradius and semiperimeter
  A = 4 * r →  -- Given condition
  p = 2 * s →  -- Perimeter is twice the semiperimeter
  p = 8 := by sorry

end triangle_perimeter_when_area_equals_four_inradius_l2636_263695


namespace cherry_pits_sprouted_percentage_l2636_263698

theorem cherry_pits_sprouted_percentage (total_pits : ℕ) (saplings_sold : ℕ) (saplings_left : ℕ) :
  total_pits = 80 →
  saplings_sold = 6 →
  saplings_left = 14 →
  (((saplings_sold + saplings_left : ℚ) / total_pits) * 100 : ℚ) = 25 := by
  sorry

end cherry_pits_sprouted_percentage_l2636_263698


namespace infinite_solutions_iff_b_eq_neg_twelve_l2636_263651

/-- The equation 4(3x-b) = 3(4x + 16) has infinitely many solutions x if and only if b = -12 -/
theorem infinite_solutions_iff_b_eq_neg_twelve (b : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end infinite_solutions_iff_b_eq_neg_twelve_l2636_263651


namespace quadratic_inequality_solution_set_l2636_263606

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo (-3 : ℝ) 2 = {x : ℝ | a * x^2 + 5 * x + b < 0}) : 
  Set.Ioo (-1/3 : ℝ) (1/2 : ℝ) = {x : ℝ | b * x^2 + 5 * x + a > 0} :=
by sorry

end quadratic_inequality_solution_set_l2636_263606


namespace savings_calculation_l2636_263609

theorem savings_calculation (income : ℕ) (ratio_income : ℕ) (ratio_expenditure : ℕ) 
  (h1 : income = 16000)
  (h2 : ratio_income = 5)
  (h3 : ratio_expenditure = 4) :
  income - (income * ratio_expenditure / ratio_income) = 3200 := by
  sorry

end savings_calculation_l2636_263609


namespace candy_bar_sales_ratio_l2636_263644

theorem candy_bar_sales_ratio :
  ∀ (price : ℚ) (marvin_sales : ℕ) (tina_extra_earnings : ℚ),
    price = 2 →
    marvin_sales = 35 →
    tina_extra_earnings = 140 →
    ∃ (tina_sales : ℕ),
      tina_sales * price = marvin_sales * price + tina_extra_earnings ∧
      tina_sales = 3 * marvin_sales :=
by sorry

end candy_bar_sales_ratio_l2636_263644


namespace complex_polynomial_root_l2636_263677

theorem complex_polynomial_root (a b c d : ℤ) : 
  (a * (Complex.I + 3) ^ 5 + b * (Complex.I + 3) ^ 4 + c * (Complex.I + 3) ^ 3 + 
   d * (Complex.I + 3) ^ 2 + b * (Complex.I + 3) + a = 0) →
  (Int.gcd a (Int.gcd b (Int.gcd c d)) = 1) →
  d.natAbs = 167 := by
sorry

end complex_polynomial_root_l2636_263677


namespace sequence_a_formula_l2636_263641

def sequence_a (n : ℕ+) : ℝ := sorry

def sum_S (n : ℕ+) : ℝ := sorry

axiom sum_S_2 : sum_S 2 = 4

axiom sequence_a_next (n : ℕ+) : sequence_a (n + 1) = 2 * sum_S n + 1

theorem sequence_a_formula (n : ℕ+) : sequence_a n = 3^(n.val - 1) := by sorry

end sequence_a_formula_l2636_263641


namespace sum_equals_5000_minus_N_l2636_263660

theorem sum_equals_5000_minus_N (N : ℕ) : 
  988 + 990 + 992 + 994 + 996 = 5000 - N → N = 40 := by
  sorry

end sum_equals_5000_minus_N_l2636_263660


namespace equation_solution_l2636_263675

theorem equation_solution (x : ℝ) (h : x ≠ 3) :
  (x^2 - 9) / (x - 3) = 3 * x ↔ x = 3/2 := by
  sorry

end equation_solution_l2636_263675


namespace negation_of_universal_statement_l2636_263691

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) := by
  sorry

end negation_of_universal_statement_l2636_263691


namespace math_book_cost_l2636_263671

theorem math_book_cost (total_books : ℕ) (math_books : ℕ) (history_book_cost : ℕ) (total_cost : ℕ) :
  total_books = 80 →
  math_books = 32 →
  history_book_cost = 5 →
  total_cost = 368 →
  ∃ (math_book_cost : ℕ),
    math_book_cost * math_books + (total_books - math_books) * history_book_cost = total_cost ∧
    math_book_cost = 4 :=
by sorry

end math_book_cost_l2636_263671


namespace angle_terminal_side_trig_sum_l2636_263624

theorem angle_terminal_side_trig_sum (α : Real) :
  (∃ (P : Real × Real), P = (-4/5, 3/5) ∧ P.1 = -4/5 ∧ P.2 = 3/5 ∧ 
   P.1^2 + P.2^2 = 1 ∧ 
   P.1 = Real.cos α ∧ P.2 = Real.sin α) →
  2 * Real.sin α + Real.cos α = 2/5 := by
sorry

end angle_terminal_side_trig_sum_l2636_263624


namespace expansion_properties_l2636_263634

/-- Given n, returns the sum of the binomial coefficients of the last three terms in (1-3x)^n -/
def sumLastThreeCoefficients (n : ℕ) : ℕ :=
  Nat.choose n (n-2) + Nat.choose n (n-1) + Nat.choose n n

/-- Returns the coefficient of the (r+1)-th term in the expansion of (1-3x)^n -/
def coefficientOfTerm (n : ℕ) (r : ℕ) : ℤ :=
  (Nat.choose n r : ℤ) * (-3) ^ r

/-- Returns the absolute value of the coefficient of the (r+1)-th term in the expansion of (1-3x)^n -/
def absCoefficient (n : ℕ) (r : ℕ) : ℕ :=
  Nat.choose n r * 3 ^ r

/-- The main theorem about the expansion of (1-3x)^n -/
theorem expansion_properties (n : ℕ) (h : sumLastThreeCoefficients n = 121) :
  (∃ r : ℕ, r = 12 ∧ ∀ k : ℕ, absCoefficient n k ≤ absCoefficient n r) ∧
  (∀ k : ℕ, Nat.choose n k ≤ Nat.choose n 7 ∧ Nat.choose n k ≤ Nat.choose n 8) :=
sorry

end expansion_properties_l2636_263634


namespace jet_ski_time_to_dock_b_l2636_263668

/-- Represents the scenario of a jet ski and a canoe traveling on a river --/
structure RiverTravel where
  distance : ℝ  -- Distance between dock A and dock B
  speed_difference : ℝ  -- Speed difference between jet ski and current
  total_time : ℝ  -- Total time until jet ski meets canoe

/-- 
Calculates the time taken by the jet ski to reach dock B.
Returns the time in hours.
-/
def time_to_dock_b (rt : RiverTravel) : ℝ :=
  sorry

/-- Theorem stating that the time taken by the jet ski to reach dock B is 3 hours --/
theorem jet_ski_time_to_dock_b (rt : RiverTravel) 
  (h1 : rt.distance = 60) 
  (h2 : rt.speed_difference = 10) 
  (h3 : rt.total_time = 8) : 
  time_to_dock_b rt = 3 :=
  sorry

end jet_ski_time_to_dock_b_l2636_263668


namespace certain_number_is_negative_eleven_l2636_263640

def binary_op (n : ℤ) : ℤ := n - (n * 5)

theorem certain_number_is_negative_eleven :
  ∃ (certain_number : ℤ),
    (binary_op 3 < certain_number) ∧
    (certain_number ≤ binary_op 4) ∧
    (∀ m : ℤ, (binary_op 3 < m) ∧ (m ≤ binary_op 4) → certain_number ≤ m) ∧
    certain_number = -11 :=
by
  sorry

end certain_number_is_negative_eleven_l2636_263640


namespace strawberry_picking_l2636_263635

theorem strawberry_picking (total strawberries_JM strawberries_Z : ℕ) 
  (h1 : total = 550)
  (h2 : strawberries_JM = 350)
  (h3 : strawberries_Z = 200) :
  total - (strawberries_JM - strawberries_Z) = 400 := by
  sorry

#check strawberry_picking

end strawberry_picking_l2636_263635


namespace cafeteria_total_l2636_263636

/-- The total number of people in a cafeteria with checkered, horizontal, and vertical striped shirts -/
def total_people (checkered : ℕ) (horizontal : ℕ) (vertical : ℕ) : ℕ :=
  checkered + horizontal + vertical

/-- Theorem: The total number of people in the cafeteria is 40 -/
theorem cafeteria_total : 
  ∃ (checkered horizontal vertical : ℕ),
    checkered = 7 ∧ 
    horizontal = 4 * checkered ∧ 
    vertical = 5 ∧ 
    total_people checkered horizontal vertical = 40 := by
  sorry

end cafeteria_total_l2636_263636


namespace camden_swim_count_l2636_263667

/-- The number of weeks in March -/
def weeks_in_march : ℕ := 4

/-- The number of times Susannah went swimming in March -/
def susannah_swims : ℕ := 24

/-- The difference in weekly swims between Susannah and Camden -/
def weekly_swim_difference : ℕ := 2

/-- Camden's total number of swims in March -/
def camden_swims : ℕ := 16

theorem camden_swim_count :
  (susannah_swims / weeks_in_march - weekly_swim_difference) * weeks_in_march = camden_swims := by
  sorry

end camden_swim_count_l2636_263667


namespace tiger_distance_is_160_l2636_263661

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- The total distance traveled by the escaped tiger -/
def tiger_distance : ℝ :=
  distance 25 1 + distance 35 2 + distance 20 1.5 + distance 10 1 + distance 50 0.5

/-- Theorem stating that the tiger traveled 160 miles -/
theorem tiger_distance_is_160 : tiger_distance = 160 := by
  sorry

end tiger_distance_is_160_l2636_263661


namespace awake_cats_l2636_263669

theorem awake_cats (total : ℕ) (asleep : ℕ) (awake : ℕ) : 
  total = 98 → asleep = 92 → awake = total - asleep → awake = 6 := by
  sorry

end awake_cats_l2636_263669


namespace divya_age_l2636_263643

theorem divya_age (divya_age nacho_age : ℝ) : 
  nacho_age + 5 = 3 * (divya_age + 5) →
  nacho_age + divya_age = 40 →
  divya_age = 7.5 := by
sorry

end divya_age_l2636_263643


namespace tangerines_taken_l2636_263682

/-- Represents the contents of Tina's fruit bag -/
structure FruitBag where
  apples : Nat
  oranges : Nat
  tangerines : Nat

/-- Represents the number of fruits taken away -/
structure FruitsTaken where
  oranges : Nat
  tangerines : Nat

def initialBag : FruitBag := { apples := 9, oranges := 5, tangerines := 17 }

def orangesTaken : Nat := 2

theorem tangerines_taken (bag : FruitBag) (taken : FruitsTaken) : 
  bag.oranges - taken.oranges + 4 = bag.tangerines - taken.tangerines →
  taken.tangerines = 10 := by
  sorry

#check tangerines_taken initialBag { oranges := orangesTaken, tangerines := 10 }

end tangerines_taken_l2636_263682


namespace ratio_proof_l2636_263654

theorem ratio_proof (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 2) : (a + b) / (b + c) = 4/9 := by
  sorry

end ratio_proof_l2636_263654


namespace ball_ratio_problem_l2636_263612

theorem ball_ratio_problem (white_balls red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 3 / 2 →
  white_balls = 9 →
  red_balls = 6 := by
sorry

end ball_ratio_problem_l2636_263612


namespace quadratic_properties_l2636_263620

-- Define the quadratic function
def quadratic (a b x : ℝ) : ℝ := a * x^2 - b * x

-- State the theorem
theorem quadratic_properties
  (a b m n : ℝ)
  (h_a : a ≠ 0)
  (h_point : quadratic a b m = 2)
  (h_range : ∀ x, quadratic a b x ≥ -2/3 → x ≤ n - 1 ∨ x ≥ -3 - n) :
  (∃ x, ∀ y, quadratic a b y = quadratic a b x → y = x ∨ y = -4 - x) ∧
  (quadratic a b 1 = 2) :=
sorry

end quadratic_properties_l2636_263620


namespace triangle_division_exists_l2636_263653

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  convex : sorry

/-- A division of a triangle into four convex shapes -/
structure TriangleDivision where
  original : ConvexPolygon 3
  triangle : ConvexPolygon 3
  quadrilateral : ConvexPolygon 4
  pentagon : ConvexPolygon 5
  hexagon : ConvexPolygon 6
  valid_division : sorry

/-- Any triangle can be divided into a triangle, quadrilateral, pentagon, and hexagon -/
theorem triangle_division_exists : ∀ (t : ConvexPolygon 3), ∃ (d : TriangleDivision), d.original = t :=
sorry

end triangle_division_exists_l2636_263653


namespace pure_imaginary_complex_number_l2636_263629

theorem pure_imaginary_complex_number (m : ℝ) : 
  (((m^2 - 5*m + 6) : ℂ) + (m^2 - 3*m)*I = (0 : ℂ) + (m^2 - 3*m)*I) → m = 2 := by
  sorry

end pure_imaginary_complex_number_l2636_263629


namespace rectangle_with_equal_sums_l2636_263663

/-- A regular polygon with 2004 sides -/
structure RegularPolygon2004 where
  vertices : Fin 2004 → ℕ
  vertex_range : ∀ i, 1 ≤ vertices i ∧ vertices i ≤ 501

/-- Four vertices form a rectangle in a regular 2004-sided polygon -/
def isRectangle (p : RegularPolygon2004) (a b c d : Fin 2004) : Prop :=
  (b - a) % 2004 = (d - c) % 2004 ∧ (c - b) % 2004 = (a - d) % 2004

/-- The sums of numbers assigned to opposite vertices are equal -/
def equalOppositeSums (p : RegularPolygon2004) (a b c d : Fin 2004) : Prop :=
  p.vertices a + p.vertices c = p.vertices b + p.vertices d

/-- Main theorem: There exist four vertices forming a rectangle with equal opposite sums -/
theorem rectangle_with_equal_sums (p : RegularPolygon2004) :
  ∃ a b c d : Fin 2004, isRectangle p a b c d ∧ equalOppositeSums p a b c d := by
  sorry


end rectangle_with_equal_sums_l2636_263663


namespace symmetry_y_axis_values_l2636_263692

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are equal -/
def symmetric_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = y₂

theorem symmetry_y_axis_values :
  ∀ a b : ℝ, symmetric_y_axis a (-3) 2 b → a = -2 ∧ b = -3 :=
by
  sorry

end symmetry_y_axis_values_l2636_263692


namespace max_xy_value_l2636_263647

theorem max_xy_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : 6 * x + 8 * y = 72) :
  x * y ≤ 27 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 6 * x₀ + 8 * y₀ = 72 ∧ x₀ * y₀ = 27 :=
by sorry

end max_xy_value_l2636_263647


namespace percent_of_percent_l2636_263690

theorem percent_of_percent (y : ℝ) : (21 / 100) * y = (30 / 100) * ((70 / 100) * y) := by
  sorry

end percent_of_percent_l2636_263690


namespace multiply_whole_and_mixed_number_l2636_263610

theorem multiply_whole_and_mixed_number :
  7 * (9 + 2 / 5) = 65 + 4 / 5 := by sorry

end multiply_whole_and_mixed_number_l2636_263610


namespace intersection_A_complement_B_l2636_263625

def U : Set Nat := {1,2,3,4,5,6,7,8}
def A : Set Nat := {2,3,5,6}
def B : Set Nat := {1,3,4,6,7}

theorem intersection_A_complement_B : A ∩ (U \ B) = {2,5} := by
  sorry

end intersection_A_complement_B_l2636_263625
