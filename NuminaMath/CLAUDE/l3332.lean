import Mathlib

namespace amys_pencils_l3332_333223

/-- Amy's pencil counting problem -/
theorem amys_pencils (initial : ℕ) (bought : ℕ) (total : ℕ) : 
  initial = 3 → bought = 7 → total = initial + bought → total = 10 := by
  sorry

end amys_pencils_l3332_333223


namespace polygon_sides_count_l3332_333239

theorem polygon_sides_count (n : ℕ) : n > 2 →
  (n - 2) * 180 = 4 * 360 → n = 10 := by
  sorry

end polygon_sides_count_l3332_333239


namespace divide_algebraic_expression_l3332_333278

theorem divide_algebraic_expression (a b : ℝ) (h : a ≠ 0) :
  (8 * a * b) / (2 * a) = 4 * b := by
  sorry

end divide_algebraic_expression_l3332_333278


namespace candy_shipment_proof_l3332_333229

/-- Represents the number of cases of each candy type in a shipment -/
structure CandyShipment where
  chocolate : ℕ
  lollipops : ℕ
  gummy_bears : ℕ

/-- The ratio of chocolate bars to lollipops to gummy bears -/
def candy_ratio : CandyShipment := ⟨3, 2, 1⟩

/-- The actual shipment received -/
def actual_shipment : CandyShipment := ⟨36, 48, 24⟩

theorem candy_shipment_proof :
  (actual_shipment.chocolate / candy_ratio.chocolate = 
   actual_shipment.lollipops / candy_ratio.lollipops) ∧
  (actual_shipment.gummy_bears = 
   actual_shipment.chocolate / candy_ratio.chocolate * candy_ratio.gummy_bears) ∧
  (actual_shipment.chocolate + actual_shipment.lollipops + actual_shipment.gummy_bears = 108) :=
by sorry

#check candy_shipment_proof

end candy_shipment_proof_l3332_333229


namespace even_times_odd_is_odd_l3332_333261

variable (f g : ℝ → ℝ)

-- Define even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statement
theorem even_times_odd_is_odd (hf : IsEven f) (hg : IsOdd g) : IsOdd (fun x ↦ f x * g x) := by
  sorry

end even_times_odd_is_odd_l3332_333261


namespace intersecting_chords_theorem_l3332_333203

/-- Given two intersecting chords in a circle, where one chord is divided into segments
    of 12 cm and 18 cm, and the other chord is divided in the ratio 3:8,
    prove that the length of the second chord is 33 cm. -/
theorem intersecting_chords_theorem (chord1_seg1 chord1_seg2 : ℝ)
  (chord2_ratio1 chord2_ratio2 : ℕ) :
  chord1_seg1 = 12 →
  chord1_seg2 = 18 →
  chord2_ratio1 = 3 →
  chord2_ratio2 = 8 →
  chord1_seg1 * chord1_seg2 = (chord2_ratio1 : ℝ) * (chord2_ratio2 : ℝ) * ((33 : ℝ) / (chord2_ratio1 + chord2_ratio2))^2 :=
by sorry

end intersecting_chords_theorem_l3332_333203


namespace triangle_with_arithmetic_progression_sides_and_perimeter_15_l3332_333227

def is_arithmetic_progression (a b c : ℕ) : Prop :=
  b - a = c - b

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_with_arithmetic_progression_sides_and_perimeter_15 :
  ∀ a b c : ℕ,
    a + b + c = 15 →
    is_arithmetic_progression a b c →
    is_valid_triangle a b c →
    ((a = 5 ∧ b = 5 ∧ c = 5) ∨
     (a = 4 ∧ b = 5 ∧ c = 6) ∨
     (a = 3 ∧ b = 5 ∧ c = 7)) :=
by sorry

end triangle_with_arithmetic_progression_sides_and_perimeter_15_l3332_333227


namespace product_of_square_roots_l3332_333296

theorem product_of_square_roots (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p^3) * Real.sqrt (25 * p^2) * Real.sqrt (2 * p^5) = 25 * p^5 * Real.sqrt 6 := by
  sorry

end product_of_square_roots_l3332_333296


namespace sam_puppies_count_l3332_333265

def final_puppies (initial bought given_away sold : ℕ) : ℕ :=
  initial - given_away + bought - sold

theorem sam_puppies_count : final_puppies 72 25 18 13 = 66 := by
  sorry

end sam_puppies_count_l3332_333265


namespace experiment_is_conditional_control_l3332_333286

-- Define the types of control experiments
inductive ControlType
  | Blank
  | Standard
  | Mutual
  | Conditional

-- Define the components of a culture medium
structure CultureMedium where
  urea : Bool
  nitrate : Bool
  otherComponents : Set String

-- Define an experimental group
structure ExperimentalGroup where
  medium : CultureMedium

-- Define the experiment
structure Experiment where
  groupA : ExperimentalGroup
  groupB : ExperimentalGroup
  sameOtherConditions : Bool

def isConditionalControl (exp : Experiment) : Prop :=
  exp.groupA.medium.urea = true ∧
  exp.groupA.medium.nitrate = false ∧
  exp.groupB.medium.urea = true ∧
  exp.groupB.medium.nitrate = true ∧
  exp.groupA.medium.otherComponents = exp.groupB.medium.otherComponents ∧
  exp.sameOtherConditions = true

theorem experiment_is_conditional_control (exp : Experiment) 
  (h1 : exp.groupA.medium.urea = true)
  (h2 : exp.groupA.medium.nitrate = false)
  (h3 : exp.groupB.medium.urea = true)
  (h4 : exp.groupB.medium.nitrate = true)
  (h5 : exp.groupA.medium.otherComponents = exp.groupB.medium.otherComponents)
  (h6 : exp.sameOtherConditions = true) :
  isConditionalControl exp :=
by sorry

end experiment_is_conditional_control_l3332_333286


namespace mean_temperature_l3332_333232

def temperatures : List Int := [-8, -6, -3, -3, 0, 4, -1]

theorem mean_temperature :
  (temperatures.sum : ℚ) / temperatures.length = -17 / 7 := by sorry

end mean_temperature_l3332_333232


namespace abs_z_equals_2_sqrt_5_l3332_333299

open Complex

theorem abs_z_equals_2_sqrt_5 (z : ℂ) 
  (h1 : ∃ (r : ℝ), z + 2*I = r)
  (h2 : ∃ (s : ℝ), z / (2 - I) = s) : 
  abs z = 2 * Real.sqrt 5 := by
sorry

end abs_z_equals_2_sqrt_5_l3332_333299


namespace prob_two_queens_or_two_aces_value_l3332_333260

-- Define the deck
def total_cards : ℕ := 52
def num_aces : ℕ := 4
def num_queens : ℕ := 4

-- Define the probability function
noncomputable def prob_two_queens_or_two_aces : ℚ :=
  let two_queens := (num_queens.choose 2) * ((total_cards - num_queens).choose 1)
  let two_aces := (num_aces.choose 2) * ((total_cards - num_aces).choose 1)
  let three_aces := num_aces.choose 3
  (two_queens + two_aces + three_aces) / (total_cards.choose 3)

-- State the theorem
theorem prob_two_queens_or_two_aces_value : 
  prob_two_queens_or_two_aces = 29 / 1105 := by sorry

end prob_two_queens_or_two_aces_value_l3332_333260


namespace last_day_sales_l3332_333251

/-- The number of packs sold by Lucy and Robyn on their last day -/
def total_packs_sold (lucy_packs robyn_packs : ℕ) : ℕ :=
  lucy_packs + robyn_packs

/-- Theorem stating that the total number of packs sold by Lucy and Robyn is 35 -/
theorem last_day_sales : total_packs_sold 19 16 = 35 := by
  sorry

end last_day_sales_l3332_333251


namespace exponential_is_self_derivative_l3332_333236

theorem exponential_is_self_derivative : 
  ∃ f : ℝ → ℝ, (∀ x, f x = Real.exp x) ∧ (∀ x, deriv f x = f x) :=
sorry

end exponential_is_self_derivative_l3332_333236


namespace calculate_net_profit_l3332_333294

/-- Given a purchase price, overhead percentage, and markup, calculate the net profit -/
theorem calculate_net_profit (purchase_price overhead_percentage markup : ℝ) :
  purchase_price = 48 →
  overhead_percentage = 0.20 →
  markup = 45 →
  let overhead := purchase_price * overhead_percentage
  let total_cost := purchase_price + overhead
  let selling_price := total_cost + markup
  let net_profit := selling_price - total_cost
  net_profit = 45 := by
  sorry

end calculate_net_profit_l3332_333294


namespace box_two_three_neg_two_l3332_333256

-- Define the box operation for integers a, b, and c
def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

-- Theorem statement
theorem box_two_three_neg_two :
  box 2 3 (-2) = 107 / 9 := by sorry

end box_two_three_neg_two_l3332_333256


namespace min_both_composers_l3332_333215

theorem min_both_composers (total : ℕ) (beethoven : ℕ) (chopin : ℕ) 
  (h1 : total = 130) 
  (h2 : beethoven = 110) 
  (h3 : chopin = 90) 
  (h4 : beethoven ≤ total) 
  (h5 : chopin ≤ total) : 
  (beethoven + chopin - total : ℤ) ≥ 70 := by
  sorry

end min_both_composers_l3332_333215


namespace range_of_half_difference_l3332_333214

theorem range_of_half_difference (α β : Real) 
  (h1 : -π/2 ≤ α) (h2 : α < β) (h3 : β ≤ π/2) :
  ∀ x, x ∈ Set.Icc (-π/2) 0 ↔ ∃ α' β', -π/2 ≤ α' ∧ α' < β' ∧ β' ≤ π/2 ∧ x = (α' - β') / 2 :=
by sorry

end range_of_half_difference_l3332_333214


namespace degree_of_specific_polynomial_l3332_333274

/-- The degree of a polynomial of the form (aᵏ * bⁿ) where a and b are polynomials -/
def degree_product_power (deg_a deg_b k n : ℕ) : ℕ := k * deg_a + n * deg_b

/-- The degree of the polynomial (x³ + x + 1)⁵ * (x⁴ + x² + 1)² -/
def degree_specific_polynomial : ℕ :=
  degree_product_power 3 4 5 2

theorem degree_of_specific_polynomial :
  degree_specific_polynomial = 23 := by sorry

end degree_of_specific_polynomial_l3332_333274


namespace exponent_multiplication_l3332_333258

theorem exponent_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end exponent_multiplication_l3332_333258


namespace max_value_x_plus_2y_l3332_333238

theorem max_value_x_plus_2y (x y : ℝ) (h : x^2 + 4*y^2 - 2*x*y = 4) :
  ∃ (M : ℝ), M = 4 ∧ ∀ (z : ℝ), z = x + 2*y → z ≤ M :=
by sorry

end max_value_x_plus_2y_l3332_333238


namespace sum_of_squares_l3332_333209

theorem sum_of_squares (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 13) : a^2 + b^2 = 35 := by
  sorry

end sum_of_squares_l3332_333209


namespace quadratic_always_positive_l3332_333271

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + x + 2 > 0 := by sorry

end quadratic_always_positive_l3332_333271


namespace set_relationship_l3332_333237

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem set_relationship :
  (¬(B (1/5) ⊆ A)) ∧
  (∀ a : ℝ, (B a ⊆ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5)) :=
sorry

end set_relationship_l3332_333237


namespace flag_arrangements_count_l3332_333235

/-- The number of distinguishable arrangements of flags on two flagpoles -/
def M : ℕ :=
  /- Number of ways to choose 11 positions out of 13 for red flags -/
  let red_positions := Nat.choose 13 11
  /- Number of ways to place the divider between flagpoles -/
  let divider_positions := 13
  /- Total number of arrangements -/
  let total_arrangements := red_positions * divider_positions
  /- Number of invalid arrangements (where one pole gets no flag) -/
  let invalid_arrangements := 2 * red_positions
  /- Final number of valid arrangements -/
  total_arrangements - invalid_arrangements

/-- Theorem stating that M is equal to 858 -/
theorem flag_arrangements_count : M = 858 := by sorry

end flag_arrangements_count_l3332_333235


namespace tribe_leadership_combinations_l3332_333282

theorem tribe_leadership_combinations (n : ℕ) (h : n = 15) : 
  (n) *                             -- Choose the chief
  (Nat.choose (n - 1) 2) *          -- Choose 2 supporting chiefs
  (Nat.choose (n - 3) 2) *          -- Choose 2 inferior officers for chief A
  (Nat.choose (n - 5) 2) *          -- Choose 2 assistants for A's officers
  (Nat.choose (n - 7) 2) *          -- Choose 2 inferior officers for chief B
  (Nat.choose (n - 9) 2) *          -- Choose 2 assistants for B's officers
  (Nat.choose (n - 11) 2) *         -- Choose 2 assistants for B's officers
  (Nat.choose (n - 13) 2) = 400762320000 := by
sorry

end tribe_leadership_combinations_l3332_333282


namespace line_equation_equivalence_l3332_333243

def line_equation (x y : ℝ) : Prop :=
  2 * (x - 3) + (-1) * (y - (-4)) = 6

theorem line_equation_equivalence :
  ∀ x y : ℝ, line_equation x y ↔ y = 2 * x - 16 :=
by sorry

end line_equation_equivalence_l3332_333243


namespace train_speed_l3332_333285

/-- Proves that the speed of a train is 36 km/hr given specific conditions -/
theorem train_speed (jogger_speed : ℝ) (initial_distance : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 →
  initial_distance = 240 →
  train_length = 120 →
  passing_time = 36 →
  (initial_distance + train_length) / passing_time * 3.6 = 36 := by
  sorry

#check train_speed

end train_speed_l3332_333285


namespace f_plus_g_at_one_l3332_333200

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_plus_g_at_one
  (h_even : is_even f)
  (h_odd : is_odd g)
  (h_diff : ∀ x, f x - g x = x^3 + x^2 + 1) :
  f 1 + g 1 = 1 := by
  sorry

end f_plus_g_at_one_l3332_333200


namespace don_tiles_per_minute_l3332_333266

/-- The number of tiles Don can paint per minute -/
def D : ℕ := sorry

/-- The number of tiles Ken can paint per minute -/
def ken_tiles : ℕ := D + 2

/-- The number of tiles Laura can paint per minute -/
def laura_tiles : ℕ := 2 * (D + 2)

/-- The number of tiles Kim can paint per minute -/
def kim_tiles : ℕ := 2 * (D + 2) - 3

/-- The total number of tiles painted by all four people in 15 minutes -/
def total_tiles : ℕ := 375

theorem don_tiles_per_minute :
  D + ken_tiles + laura_tiles + kim_tiles = total_tiles / 15 ∧ D = 3 := by sorry

end don_tiles_per_minute_l3332_333266


namespace divisor_condition_l3332_333216

def satisfies_condition (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k l : ℕ, k ∣ n → l ∣ n → k < n → l < n →
    (2*k - l) ∣ n ∨ (2*l - k) ∣ n

theorem divisor_condition (n : ℕ) :
  satisfies_condition n ↔ Nat.Prime n ∨ n ∈ ({6, 9, 15} : Set ℕ) :=
sorry

end divisor_condition_l3332_333216


namespace max_product_representation_l3332_333204

def representation_sum (n : ℕ) : List ℕ → Prop :=
  λ l => l.sum = n ∧ l.all (· > 0)

theorem max_product_representation (n : ℕ) :
  ∃ (l : List ℕ), representation_sum 2015 l ∧
    ∀ (m : List ℕ), representation_sum 2015 m →
      l.prod ≥ m.prod :=
by
  sorry

#check max_product_representation 2015

end max_product_representation_l3332_333204


namespace matildas_chocolate_bars_l3332_333218

/-- Proves that Matilda initially had 4 chocolate bars given the problem conditions -/
theorem matildas_chocolate_bars (total_people : ℕ) (sisters : ℕ) (fathers_remaining : ℕ) 
  (mothers_share : ℕ) (fathers_eaten : ℕ) :
  total_people = sisters + 1 →
  sisters = 4 →
  fathers_remaining = 5 →
  mothers_share = 3 →
  fathers_eaten = 2 →
  ∃ (initial_bars : ℕ),
    initial_bars = (fathers_remaining + mothers_share + fathers_eaten) * 2 / total_people ∧
    initial_bars = 4 :=
by sorry

end matildas_chocolate_bars_l3332_333218


namespace unique_solution_system_l3332_333289

theorem unique_solution_system (x y z : ℝ) : 
  (x + y = 3 * x + 4) ∧ 
  (2 * y + 3 + z = 6 * y + 6) ∧ 
  (3 * z + 3 + x = 9 * z + 8) ↔ 
  (x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

end unique_solution_system_l3332_333289


namespace paint_calculation_l3332_333201

/-- Represents the number of rooms that can be painted with a given amount of paint -/
structure PaintCoverage where
  totalRooms : ℕ
  cansUsed : ℕ

/-- Calculates the number of cans used for a given number of rooms -/
def cansForRooms (initialCoverage finalCoverage : PaintCoverage) (roomsToPaint : ℕ) : ℕ :=
  let roomsPerCan := (initialCoverage.totalRooms - finalCoverage.totalRooms) / 
                     (initialCoverage.cansUsed - finalCoverage.cansUsed)
  roomsToPaint / roomsPerCan

theorem paint_calculation (initialCoverage finalCoverage : PaintCoverage) 
  (h1 : initialCoverage.totalRooms = 45)
  (h2 : finalCoverage.totalRooms = 36)
  (h3 : initialCoverage.cansUsed - finalCoverage.cansUsed = 4) :
  cansForRooms initialCoverage finalCoverage 36 = 16 := by
  sorry

end paint_calculation_l3332_333201


namespace total_sequences_is_288_l3332_333249

/-- Represents a team in the tournament -/
inductive Team : Type
  | A | B | C | D | E | F

/-- Represents a match between two teams -/
structure Match where
  team1 : Team
  team2 : Team

/-- Represents the tournament structure -/
structure Tournament where
  day1_matches : List Match
  no_ties : Bool

/-- Calculates the number of possible outcomes for a given number of matches -/
def possible_outcomes (num_matches : Nat) : Nat :=
  2^num_matches

/-- Calculates the number of possible arrangements for the winners' group on day 2 -/
def winners_arrangements (num_winners : Nat) : Nat :=
  Nat.factorial num_winners

/-- Calculates the number of possible outcomes for the losers' match on day 2 -/
def losers_match_outcomes (num_losers : Nat) : Nat :=
  num_losers * 2

/-- Calculates the total number of possible ranking sequences -/
def total_sequences (t : Tournament) : Nat :=
  possible_outcomes t.day1_matches.length *
  winners_arrangements 3 *
  losers_match_outcomes 3 *
  possible_outcomes 1

/-- The theorem stating that the total number of possible ranking sequences is 288 -/
theorem total_sequences_is_288 (t : Tournament) 
  (h1 : t.day1_matches.length = 3)
  (h2 : t.no_ties = true) :
  total_sequences t = 288 := by
  sorry

end total_sequences_is_288_l3332_333249


namespace math_book_cost_l3332_333252

/-- Proves that the cost of each math book is $4 given the conditions of the book purchase problem. -/
theorem math_book_cost (total_books : ℕ) (history_book_cost : ℕ) (total_cost : ℕ) (math_books : ℕ) :
  total_books = 80 →
  history_book_cost = 5 →
  total_cost = 390 →
  math_books = 10 →
  (total_books - math_books) * history_book_cost + math_books * 4 = total_cost :=
by
  sorry

#check math_book_cost

end math_book_cost_l3332_333252


namespace factorial_product_not_square_l3332_333288

theorem factorial_product_not_square (n : ℕ) : 
  ∃ (m : ℕ), (n.factorial ^ 2 * (n + 1).factorial * (2 * n + 9).factorial * (2 * n + 10).factorial) ≠ m ^ 2 := by
  sorry

end factorial_product_not_square_l3332_333288


namespace largest_divisor_of_n_l3332_333202

theorem largest_divisor_of_n (n : ℕ+) (h : 650 ∣ n^3) : 130 ∣ n := by
  sorry

end largest_divisor_of_n_l3332_333202


namespace sqrt_equation_solution_l3332_333272

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 + Real.sqrt (x^2)) = 4 ↔ x = 13 ∨ x = -13 :=
by sorry

end sqrt_equation_solution_l3332_333272


namespace simplify_sqrt_sum_l3332_333268

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_sum_l3332_333268


namespace sequence_sum_product_l3332_333253

def sequence_property (α β γ : ℕ) (a b : ℕ → ℕ) : Prop :=
  (α < γ) ∧
  (α * γ = β^2 + 1) ∧
  (a 0 = 1) ∧ (b 0 = 1) ∧
  (∀ n, a (n + 1) = α * a n + β * b n) ∧
  (∀ n, b (n + 1) = β * a n + γ * b n)

theorem sequence_sum_product (α β γ : ℕ) (a b : ℕ → ℕ) 
  (h : sequence_property α β γ a b) :
  ∀ m n, a (m + n) + b (m + n) = a m * a n + b m * b n :=
by sorry

end sequence_sum_product_l3332_333253


namespace no_number_with_digit_product_1560_l3332_333224

/-- The product of the decimal digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- Theorem stating that no natural number has a digit product of 1560 -/
theorem no_number_with_digit_product_1560 : 
  ¬ ∃ (n : ℕ), digit_product n = 1560 := by sorry

end no_number_with_digit_product_1560_l3332_333224


namespace inequality_proof_l3332_333297

theorem inequality_proof (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h1 : a₁ ≥ a₂) (h2 : a₂ ≥ a₃) (h3 : a₃ > 0)
  (h4 : b₁ ≥ b₂) (h5 : b₂ ≥ b₃) (h6 : b₃ > 0)
  (h7 : a₁ * a₂ * a₃ = b₁ * b₂ * b₃)
  (h8 : a₁ - a₃ ≤ b₁ - b₃) :
  a₁ + a₂ + a₃ ≤ 2 * (b₁ + b₂ + b₃) := by
sorry

end inequality_proof_l3332_333297


namespace correct_option_is_valid_print_statement_l3332_333276

-- Define an enum for the options
inductive ProgramOption
| A
| B
| C
| D

-- Define a function to check if an option is a valid print statement
def isValidPrintStatement (option : ProgramOption) : Prop :=
  match option with
  | ProgramOption.A => True  -- PRINT 4*x is valid
  | _ => False               -- Other options are not valid print statements

-- Theorem statement
theorem correct_option_is_valid_print_statement :
  ∃ (option : ProgramOption), isValidPrintStatement option :=
by
  sorry


end correct_option_is_valid_print_statement_l3332_333276


namespace negation_equivalence_l3332_333234

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0) := by
  sorry

end negation_equivalence_l3332_333234


namespace arrangement_remainder_l3332_333275

/-- The number of green marbles -/
def green_marbles : ℕ := 5

/-- The maximum number of red marbles that satisfies the condition -/
def max_red_marbles : ℕ := 16

/-- The total number of marbles -/
def total_marbles : ℕ := green_marbles + max_red_marbles

/-- The number of ways to arrange the marbles -/
def num_arrangements : ℕ := Nat.choose (green_marbles + max_red_marbles) green_marbles

/-- The theorem to be proved -/
theorem arrangement_remainder : num_arrangements % 1000 = 3 := by
  sorry

end arrangement_remainder_l3332_333275


namespace square_side_length_l3332_333279

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 64 → side * side = area → side = 8 := by
  sorry

end square_side_length_l3332_333279


namespace doughnuts_served_l3332_333284

theorem doughnuts_served (staff : ℕ) (doughnuts_per_staff : ℕ) (doughnuts_left : ℕ) : 
  staff = 19 → doughnuts_per_staff = 2 → doughnuts_left = 12 →
  staff * doughnuts_per_staff + doughnuts_left = 50 := by
  sorry

end doughnuts_served_l3332_333284


namespace sixth_term_is_geometric_mean_l3332_333226

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d

/-- The second term is the geometric mean of the first and fourth terms -/
def SecondTermIsGeometricMean (a : ℕ → ℝ) : Prop :=
  a 2 = Real.sqrt (a 1 * a 4)

theorem sixth_term_is_geometric_mean
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_second : SecondTermIsGeometricMean a) :
  a 6 = Real.sqrt (a 4 * a 9) :=
by sorry

end sixth_term_is_geometric_mean_l3332_333226


namespace range_of_a_l3332_333295

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∨ 
  (∃ x₀ : ℝ, x₀^2 - x₀ + a = 0) ∧ 
  ¬((∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ 
    (∃ x₀ : ℝ, x₀^2 - x₀ + a = 0)) →
  a < 0 ∨ (1/4 < a ∧ a < 4) :=
sorry

end range_of_a_l3332_333295


namespace cylinder_volume_l3332_333254

/-- The volume of a cylinder with base radius 1 cm and height 2 cm is 2π cm³ -/
theorem cylinder_volume : 
  let r : ℝ := 1  -- base radius in cm
  let h : ℝ := 2  -- height in cm
  let V : ℝ := π * r^2 * h  -- volume formula
  V = 2 * π := by
  sorry

end cylinder_volume_l3332_333254


namespace school_children_count_l3332_333270

theorem school_children_count :
  let absent_children : ℕ := 160
  let total_bananas : ℕ → ℕ → ℕ := λ present absent => 2 * present + 2 * absent
  let extra_bananas : ℕ → ℕ → ℕ := λ present absent => 2 * present
  let boys_bananas : ℕ → ℕ := λ total => 3 * (total / 4)
  let girls_bananas : ℕ → ℕ := λ total => total / 4
  ∃ (present_children : ℕ),
    total_bananas present_children absent_children = 
      total_bananas present_children present_children + extra_bananas present_children absent_children ∧
    boys_bananas (total_bananas present_children absent_children) + 
      girls_bananas (total_bananas present_children absent_children) = 
      total_bananas present_children absent_children ∧
    present_children + absent_children = 6560 :=
by sorry

end school_children_count_l3332_333270


namespace fifth_place_votes_l3332_333263

theorem fifth_place_votes (total_votes : ℕ) (num_candidates : ℕ) 
  (diff1 diff2 diff3 diff4 : ℕ) :
  total_votes = 3567 →
  num_candidates = 5 →
  diff1 = 143 →
  diff2 = 273 →
  diff3 = 329 →
  diff4 = 503 →
  ∃ (winner_votes : ℕ),
    winner_votes + (winner_votes - diff1) + (winner_votes - diff2) + 
    (winner_votes - diff3) + (winner_votes - diff4) = total_votes ∧
    winner_votes - diff4 = 700 :=
by sorry

end fifth_place_votes_l3332_333263


namespace percentage_to_full_amount_l3332_333250

theorem percentage_to_full_amount (amount : ℝ) : 
  (25 / 100) * amount = 200 → amount = 800 := by
  sorry

end percentage_to_full_amount_l3332_333250


namespace valid_sequence_count_l3332_333210

def word : String := "EQUALS"

def valid_sequence (s : String) : Prop :=
  s.length = 4 ∧
  s.toList.toFinset ⊆ word.toList.toFinset ∧
  s.front = 'L' ∧
  s.back = 'Q' ∧
  s.toList.toFinset.card = 4

def count_valid_sequences : ℕ :=
  (word.toList.toFinset.filter (λ c => c ≠ 'L' ∧ c ≠ 'Q')).card *
  ((word.toList.toFinset.filter (λ c => c ≠ 'L' ∧ c ≠ 'Q')).card - 1)

theorem valid_sequence_count :
  count_valid_sequences = 12 :=
sorry

end valid_sequence_count_l3332_333210


namespace prob_rain_all_days_l3332_333242

def prob_rain_friday : ℝ := 0.4
def prob_rain_saturday : ℝ := 0.5
def prob_rain_sunday : ℝ := 0.2

theorem prob_rain_all_days :
  prob_rain_friday * prob_rain_saturday * prob_rain_sunday = 0.04 := by
  sorry

end prob_rain_all_days_l3332_333242


namespace car_speed_proof_l3332_333290

/-- 
Proves that a car traveling at a constant speed v km/h takes 2 seconds longer 
to travel 1 kilometer than it would at 450 km/h if and only if v = 360 km/h.
-/
theorem car_speed_proof (v : ℝ) : v > 0 → (
  (1 / v) * 3600 = (1 / 450) * 3600 + 2 ↔ v = 360
) := by sorry

end car_speed_proof_l3332_333290


namespace benton_school_earnings_l3332_333221

/-- Represents the total earnings of students from a school -/
def school_earnings (students : ℕ) (days : ℕ) (daily_wage : ℚ) : ℚ :=
  students * days * daily_wage

/-- Calculates the daily wage per student given the total amount and total student-days -/
def calculate_daily_wage (total_amount : ℚ) (total_student_days : ℕ) : ℚ :=
  total_amount / total_student_days

theorem benton_school_earnings :
  let adams_students : ℕ := 4
  let adams_days : ℕ := 4
  let benton_students : ℕ := 5
  let benton_days : ℕ := 6
  let camden_students : ℕ := 6
  let camden_days : ℕ := 7
  let total_amount : ℚ := 780

  let total_student_days : ℕ := 
    adams_students * adams_days + 
    benton_students * benton_days + 
    camden_students * camden_days

  let daily_wage : ℚ := calculate_daily_wage total_amount total_student_days

  let benton_earnings : ℚ := school_earnings benton_students benton_days daily_wage

  ⌊benton_earnings⌋ = 266 :=
by sorry

end benton_school_earnings_l3332_333221


namespace coin_flip_probability_difference_l3332_333277

def fair_coin_probability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ k * (1 / 2) ^ (n - k)

theorem coin_flip_probability_difference :
  fair_coin_probability 4 3 - fair_coin_probability 4 4 = 7 / 16 := by
  sorry

end coin_flip_probability_difference_l3332_333277


namespace exposed_sides_is_21_l3332_333244

/-- Represents a polygon with a specific number of sides -/
structure Polygon where
  sides : ℕ
  sides_positive : sides > 0

/-- Represents the configuration of polygons -/
structure PolygonConfiguration where
  triangle : Polygon
  square : Polygon
  pentagon : Polygon
  hexagon : Polygon
  heptagon : Polygon
  triangle_is_equilateral : triangle.sides = 3
  square_is_square : square.sides = 4
  pentagon_is_pentagon : pentagon.sides = 5
  hexagon_is_hexagon : hexagon.sides = 6
  heptagon_is_heptagon : heptagon.sides = 7

/-- The number of shared sides in the configuration -/
def shared_sides : ℕ := 4

/-- Theorem stating that the number of exposed sides in the configuration is 21 -/
theorem exposed_sides_is_21 (config : PolygonConfiguration) : 
  config.triangle.sides + config.square.sides + config.pentagon.sides + 
  config.hexagon.sides + config.heptagon.sides - shared_sides = 21 := by
  sorry

end exposed_sides_is_21_l3332_333244


namespace second_tank_volume_l3332_333220

/-- Represents the capacity of each tank in liters -/
def tank_capacity : ℝ := 1000

/-- Represents the volume of water in the first tank in liters -/
def first_tank_volume : ℝ := 300

/-- Represents the fraction of the second tank that is filled -/
def second_tank_fill_ratio : ℝ := 0.45

/-- Represents the additional water needed to fill both tanks in liters -/
def additional_water_needed : ℝ := 1250

/-- Theorem stating that the second tank contains 450 liters of water -/
theorem second_tank_volume :
  let second_tank_volume := second_tank_fill_ratio * tank_capacity
  second_tank_volume = 450 := by sorry

end second_tank_volume_l3332_333220


namespace sqrt_inequality_l3332_333228

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) :
  Real.sqrt a - Real.sqrt (a - 2) < Real.sqrt (a - 1) - Real.sqrt (a - 3) := by
  sorry

end sqrt_inequality_l3332_333228


namespace sharmila_hourly_wage_l3332_333208

/-- Represents Sharmila's work schedule and earnings -/
structure WorkSchedule where
  monday_hours : ℕ
  wednesday_hours : ℕ
  friday_hours : ℕ
  tuesday_hours : ℕ
  thursday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week -/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.monday_hours + 2 * schedule.tuesday_hours

/-- Calculates the hourly wage given a work schedule -/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sharmila's work schedule -/
def sharmila_schedule : WorkSchedule := {
  monday_hours := 10,
  wednesday_hours := 10,
  friday_hours := 10,
  tuesday_hours := 8,
  thursday_hours := 8,
  weekly_earnings := 460
}

/-- Theorem stating that Sharmila's hourly wage is $10 -/
theorem sharmila_hourly_wage :
  hourly_wage sharmila_schedule = 10 := by sorry

end sharmila_hourly_wage_l3332_333208


namespace king_middle_school_teachers_l3332_333217

theorem king_middle_school_teachers :
  let total_students : ℕ := 1500
  let classes_per_student : ℕ := 5
  let regular_class_size : ℕ := 30
  let specialized_classes : ℕ := 10
  let specialized_class_size : ℕ := 15
  let classes_per_teacher : ℕ := 3

  let total_class_instances : ℕ := total_students * classes_per_student
  let specialized_class_instances : ℕ := specialized_classes * specialized_class_size
  let regular_class_instances : ℕ := total_class_instances - specialized_class_instances
  let regular_classes : ℕ := regular_class_instances / regular_class_size
  let total_classes : ℕ := regular_classes + specialized_classes
  let number_of_teachers : ℕ := total_classes / classes_per_teacher

  number_of_teachers = 85 := by sorry

end king_middle_school_teachers_l3332_333217


namespace mushroom_collection_l3332_333231

theorem mushroom_collection (a b v g : ℚ) 
  (eq1 : a / 2 + 2 * b = v + g) 
  (eq2 : a + b = v / 2 + 2 * g) : 
  v = 2 * b ∧ a = 2 * g := by
  sorry

end mushroom_collection_l3332_333231


namespace hundredth_count_is_twelve_l3332_333246

/-- Represents the circular arrangement of stones with the given counting pattern. -/
def StoneCircle :=
  { n : ℕ // n > 0 ∧ n ≤ 12 }

/-- The label assigned to a stone after a certain number of counts. -/
def label (count : ℕ) : StoneCircle → ℕ :=
  sorry

/-- The original stone number corresponding to a given label. -/
def originalStone (label : ℕ) : StoneCircle :=
  sorry

/-- Theorem stating that the 100th count corresponds to the original stone number 12. -/
theorem hundredth_count_is_twelve :
  originalStone 100 = ⟨12, by norm_num⟩ :=
sorry

end hundredth_count_is_twelve_l3332_333246


namespace min_sum_of_product_72_l3332_333230

theorem min_sum_of_product_72 (a b : ℤ) (h : a * b = 72) : 
  (∀ x y : ℤ, x * y = 72 → a + b ≤ x + y) ∧ (∃ x y : ℤ, x * y = 72 ∧ x + y = -17) := by
  sorry

end min_sum_of_product_72_l3332_333230


namespace temperature_drop_per_tree_l3332_333262

/-- Proves that the temperature drop per tree is 0.1 degrees -/
theorem temperature_drop_per_tree 
  (cost_per_tree : ℝ) 
  (initial_temp : ℝ) 
  (final_temp : ℝ) 
  (total_cost : ℝ) 
  (h1 : cost_per_tree = 6)
  (h2 : initial_temp = 80)
  (h3 : final_temp = 78.2)
  (h4 : total_cost = 108) :
  (initial_temp - final_temp) / (total_cost / cost_per_tree) = 0.1 := by
  sorry

end temperature_drop_per_tree_l3332_333262


namespace B_cannot_be_possible_l3332_333212

-- Define the set A
def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}

-- Define the set B (the one we want to prove cannot be possible)
def B : Set ℝ := {x : ℝ | x ≥ -1}

-- Theorem statement
theorem B_cannot_be_possible : A ∩ B = ∅ → False := by
  sorry

end B_cannot_be_possible_l3332_333212


namespace fixed_point_and_parabola_l3332_333207

/-- The fixed point P that the line passes through for all values of a -/
def P : ℝ × ℝ := (2, -8)

/-- The line equation for any real number a -/
def line_equation (a x y : ℝ) : Prop :=
  (2*a + 3)*x + y - 4*a + 2 = 0

/-- The parabola equation with y-axis as the axis of symmetry -/
def parabola_equation_y (x y : ℝ) : Prop :=
  y^2 = 32*x

/-- The parabola equation with x-axis as the axis of symmetry -/
def parabola_equation_x (x y : ℝ) : Prop :=
  x^2 = -1/2*y

theorem fixed_point_and_parabola :
  (∀ a : ℝ, line_equation a P.1 P.2) ∧
  (parabola_equation_y P.1 P.2 ∨ parabola_equation_x P.1 P.2) :=
sorry

end fixed_point_and_parabola_l3332_333207


namespace function_property_l3332_333287

/-- Given a function f(x) = ax^4 + bx^2 - x + 1 where a and b are real numbers,
    if f(2) = 9, then f(-2) = 13 -/
theorem function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^4 + b * x^2 - x + 1
  f 2 = 9 → f (-2) = 13 := by
  sorry

end function_property_l3332_333287


namespace necessary_not_sufficient_l3332_333241

/-- The function f(x) = 3x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := 3 * x + m

/-- m is not in the open interval (-3, -1) -/
def not_in_interval (m : ℝ) : Prop := m ≤ -3 ∨ m ≥ -1

/-- f has no zero in the interval [0, 1] -/
def no_zero_in_interval (m : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, f m x ≠ 0

theorem necessary_not_sufficient :
  (∀ m : ℝ, no_zero_in_interval m → not_in_interval m) ∧
  (∃ m : ℝ, not_in_interval m ∧ ¬(no_zero_in_interval m)) :=
by sorry

end necessary_not_sufficient_l3332_333241


namespace smallest_multiple_36_with_digit_sum_multiple_9_l3332_333281

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

theorem smallest_multiple_36_with_digit_sum_multiple_9 :
  ∃ (k : ℕ), k > 0 ∧ 36 * k = 36 ∧
  (∀ m : ℕ, m > 0 ∧ m < k → ¬(∃ n : ℕ, 36 * m = 36 * n ∧ 9 ∣ sumOfDigits (36 * n))) ∧
  (9 ∣ sumOfDigits 36) :=
sorry

end smallest_multiple_36_with_digit_sum_multiple_9_l3332_333281


namespace probability_triangle_or_circle_l3332_333247

def total_figures : ℕ := 12
def num_triangles : ℕ := 4
def num_circles : ℕ := 3
def num_squares : ℕ := 5

theorem probability_triangle_or_circle :
  (num_triangles + num_circles : ℚ) / total_figures = 7 / 12 := by
  sorry

end probability_triangle_or_circle_l3332_333247


namespace min_additional_squares_for_symmetry_l3332_333245

/-- A point in the grid --/
structure Point where
  x : Nat
  y : Nat

/-- The grid dimensions --/
def gridWidth : Nat := 4
def gridHeight : Nat := 5

/-- The initially colored squares --/
def initialColoredSquares : List Point := [
  { x := 1, y := 4 },
  { x := 2, y := 1 },
  { x := 4, y := 2 }
]

/-- A function to check if a point is within the grid --/
def isInGrid (p : Point) : Prop :=
  1 ≤ p.x ∧ p.x ≤ gridWidth ∧ 1 ≤ p.y ∧ p.y ≤ gridHeight

/-- A function to check if two points are symmetrical about the vertical line --/
def isVerticallySymmetric (p1 p2 : Point) : Prop :=
  p1.x + p2.x = gridWidth + 1 ∧ p1.y = p2.y

/-- A function to check if two points are symmetrical about the horizontal line --/
def isHorizontallySymmetric (p1 p2 : Point) : Prop :=
  p1.x = p2.x ∧ p1.y + p2.y = gridHeight + 1

/-- A function to check if two points are rotationally symmetric --/
def isRotationallySymmetric (p1 p2 : Point) : Prop :=
  p1.x + p2.x = gridWidth + 1 ∧ p1.y + p2.y = gridHeight + 1

/-- The main theorem --/
theorem min_additional_squares_for_symmetry :
  ∃ (additionalSquares : List Point),
    (∀ p ∈ additionalSquares, isInGrid p) ∧
    (∀ p ∈ initialColoredSquares ++ additionalSquares,
      (∃ q ∈ initialColoredSquares ++ additionalSquares, isVerticallySymmetric p q) ∧
      (∃ q ∈ initialColoredSquares ++ additionalSquares, isHorizontallySymmetric p q) ∧
      (∃ q ∈ initialColoredSquares ++ additionalSquares, isRotationallySymmetric p q)) ∧
    additionalSquares.length = 9 ∧
    (∀ (otherSquares : List Point),
      (∀ p ∈ otherSquares, isInGrid p) →
      (∀ p ∈ initialColoredSquares ++ otherSquares,
        (∃ q ∈ initialColoredSquares ++ otherSquares, isVerticallySymmetric p q) ∧
        (∃ q ∈ initialColoredSquares ++ otherSquares, isHorizontallySymmetric p q) ∧
        (∃ q ∈ initialColoredSquares ++ otherSquares, isRotationallySymmetric p q)) →
      otherSquares.length ≥ 9) :=
sorry

end min_additional_squares_for_symmetry_l3332_333245


namespace shorter_tank_radius_l3332_333269

/-- Given two cylindrical tanks with equal volumes, where one tank is twice as tall as the other,
    and the radius of the taller tank is 10 units, the radius of the shorter tank is 10√2 units. -/
theorem shorter_tank_radius (h : ℝ) (h_pos : h > 0) : 
  let v := π * (10^2) * (2*h)  -- Volume of the taller tank
  let r := Real.sqrt 200       -- Radius of the shorter tank
  v = π * r^2 * h              -- Volumes are equal
  → r = 10 * Real.sqrt 2       -- Radius of the shorter tank is 10√2
  := by sorry

end shorter_tank_radius_l3332_333269


namespace average_difference_l3332_333213

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : (b + c) / 2 = 170) : 
  a - c = -120 := by
sorry

end average_difference_l3332_333213


namespace triangle_inequality_with_median_l3332_333206

/-- 
For any triangle with side lengths a, b, and c, and median length m_a 
from vertex A to the midpoint of side BC, the inequality a^2 + 4m_a^2 ≤ (b+c)^2 holds.
-/
theorem triangle_inequality_with_median 
  (a b c m_a : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < m_a) 
  (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b) 
  (h_median : m_a > 0) : 
  a^2 + 4 * m_a^2 ≤ (b + c)^2 := by
  sorry

end triangle_inequality_with_median_l3332_333206


namespace quadratic_inequality_range_l3332_333293

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + a > 0) → a > 1 := by
  sorry

end quadratic_inequality_range_l3332_333293


namespace grey_area_ratio_l3332_333205

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square piece of paper -/
structure Square where
  sideLength : ℝ
  a : Point
  b : Point
  c : Point
  d : Point

/-- Represents a kite shape -/
structure Kite where
  a : Point
  e : Point
  c : Point
  f : Point

/-- Function to fold the paper along a line -/
def foldPaper (s : Square) (p : Point) : Kite :=
  sorry

/-- Theorem stating the ratio of grey area to total area of the kite -/
theorem grey_area_ratio (s : Square) (e f : Point) :
  let k := foldPaper s e
  let k' := foldPaper s f
  let greyArea := sorry
  let totalArea := sorry
  greyArea / totalArea = 1 / Real.sqrt 2 := by
  sorry

end grey_area_ratio_l3332_333205


namespace remainder_sum_l3332_333280

theorem remainder_sum (x y : ℤ) : 
  x % 80 = 75 → y % 120 = 115 → (x + y) % 40 = 30 := by sorry

end remainder_sum_l3332_333280


namespace set_M_equals_one_two_three_l3332_333267

def M : Set ℤ := {a | 0 < 2*a - 1 ∧ 2*a - 1 ≤ 5}

theorem set_M_equals_one_two_three : M = {1, 2, 3} := by
  sorry

end set_M_equals_one_two_three_l3332_333267


namespace savings_double_l3332_333291

/-- Represents the financial situation of a man over two years -/
structure FinancialSituation where
  first_year_income : ℝ
  first_year_savings_rate : ℝ
  income_increase_rate : ℝ
  expenditure_ratio : ℝ

/-- Calculates the percentage increase in savings -/
def savings_increase_percentage (fs : FinancialSituation) : ℝ :=
  -- The actual calculation will be implemented in the proof
  sorry

/-- Theorem stating that the savings increase by 100% -/
theorem savings_double (fs : FinancialSituation) 
  (h1 : fs.first_year_savings_rate = 0.35)
  (h2 : fs.income_increase_rate = 0.35)
  (h3 : fs.expenditure_ratio = 2)
  : savings_increase_percentage fs = 100 := by
  sorry

end savings_double_l3332_333291


namespace hyperbolas_same_asymptotes_l3332_333225

/-- Given two hyperbolas with equations x²/9 - y²/16 = 1 and y²/25 - x²/M = 1,
    prove that if they have the same asymptotes, then M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ y^2/25 - x^2/M = 1) →
  (∀ k : ℝ, (∃ x y : ℝ, y = k*x ∧ x^2/9 - y^2/16 = 1) ↔
            (∃ x y : ℝ, y = k*x ∧ y^2/25 - x^2/M = 1)) →
  M = 225/16 :=
by sorry

end hyperbolas_same_asymptotes_l3332_333225


namespace imaginary_part_of_2_minus_i_l3332_333255

theorem imaginary_part_of_2_minus_i :
  let z : ℂ := 2 - I
  Complex.im z = -1 := by sorry

end imaginary_part_of_2_minus_i_l3332_333255


namespace croissant_mix_time_l3332_333222

/-- The time it takes to make croissants -/
def croissant_making : Prop :=
  let fold_count : ℕ := 4
  let fold_time : ℕ := 5
  let rest_count : ℕ := 4
  let rest_time : ℕ := 75
  let bake_time : ℕ := 30
  let total_time : ℕ := 6 * 60

  let fold_total : ℕ := fold_count * fold_time
  let rest_total : ℕ := rest_count * rest_time
  let known_time : ℕ := fold_total + rest_total + bake_time

  let mix_time : ℕ := total_time - known_time

  mix_time = 10

theorem croissant_mix_time : croissant_making := by
  sorry

end croissant_mix_time_l3332_333222


namespace vinnie_tips_l3332_333283

theorem vinnie_tips (paul_tips : ℕ) (vinnie_more : ℕ) : 
  paul_tips = 14 → vinnie_more = 16 → paul_tips + vinnie_more = 30 := by
  sorry

end vinnie_tips_l3332_333283


namespace unique_solution_l3332_333257

theorem unique_solution : ∃! (m n : ℕ), 
  m > 0 ∧ n > 0 ∧ 14 * m * n = 55 - 7 * m - 2 * n :=
by
  -- The proof would go here
  sorry

end unique_solution_l3332_333257


namespace largest_divisor_of_consecutive_sum_l3332_333211

theorem largest_divisor_of_consecutive_sum (a : ℤ) : 
  ∃ (d : ℤ), d > 0 ∧ d ∣ (a - 1 + a + a + 1) ∧ 
  ∀ (k : ℤ), k > d → ∃ (n : ℤ), ¬(k ∣ (n - 1 + n + n + 1)) :=
by sorry

end largest_divisor_of_consecutive_sum_l3332_333211


namespace smallest_n_with_right_triangle_l3332_333298

/-- A function that checks if three numbers can form a right triangle --/
def isRightTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The set S containing numbers from 1 to 50 --/
def S : Finset ℕ := Finset.range 50

/-- A property that checks if a subset of size n always contains a right triangle --/
def hasRightTriangle (n : ℕ) : Prop :=
  ∀ (T : Finset ℕ), T ⊆ S → T.card = n →
    ∃ (a b c : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ isRightTriangle a b c

/-- The main theorem stating that 42 is the smallest n satisfying the property --/
theorem smallest_n_with_right_triangle :
  hasRightTriangle 42 ∧ ∀ m < 42, ¬(hasRightTriangle m) :=
sorry

end smallest_n_with_right_triangle_l3332_333298


namespace circle_area_ratio_l3332_333233

theorem circle_area_ratio (R : ℝ) (R_pos : R > 0) : 
  (π * (R/3)^2) / (π * R^2) = 1/9 := by
  sorry

end circle_area_ratio_l3332_333233


namespace pairwise_ratio_sum_geq_three_halves_l3332_333264

theorem pairwise_ratio_sum_geq_three_halves
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ 3 / 2 := by
  sorry

end pairwise_ratio_sum_geq_three_halves_l3332_333264


namespace number_problem_l3332_333273

theorem number_problem (x : ℝ) : 0.60 * x - 40 = 50 → x = 150 := by
  sorry

end number_problem_l3332_333273


namespace students_in_row_l3332_333292

theorem students_in_row (S R : ℕ) : 
  S = 5 * R + 6 →
  S = 6 * (R - 3) →
  6 = S / R - 18 := by
sorry

end students_in_row_l3332_333292


namespace choose_two_from_three_l3332_333219

theorem choose_two_from_three : Nat.choose 3 2 = 3 := by
  sorry

end choose_two_from_three_l3332_333219


namespace max_advancing_players_16_10_l3332_333240

/-- Represents a chess tournament -/
structure ChessTournament where
  players : ℕ
  points_to_advance : ℕ

/-- Calculates the total number of games in a round-robin tournament -/
def total_games (t : ChessTournament) : ℕ :=
  t.players * (t.players - 1) / 2

/-- Calculates the total points awarded in the tournament -/
def total_points (t : ChessTournament) : ℕ :=
  total_games t

/-- Defines the maximum number of players that can advance -/
def max_advancing_players (t : ChessTournament) : ℕ :=
  11

/-- Theorem: In a 16-player tournament where players need at least 10 points to advance,
    the maximum number of players who can advance is 11 -/
theorem max_advancing_players_16_10 :
  ∀ t : ChessTournament,
    t.players = 16 →
    t.points_to_advance = 10 →
    max_advancing_players t = 11 :=
by sorry


end max_advancing_players_16_10_l3332_333240


namespace book_arrangement_theorem_l3332_333248

/-- The number of ways to arrange n unique objects --/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange two groups of books, where each group stays together --/
def arrange_book_groups : ℕ := permutations 2

/-- The number of ways to arrange 4 unique math books within their group --/
def arrange_math_books : ℕ := permutations 4

/-- The number of ways to arrange 4 unique English books within their group --/
def arrange_english_books : ℕ := permutations 4

/-- The total number of ways to arrange 4 unique math books and 4 unique English books on a shelf,
    with all math books staying together and all English books staying together --/
def total_arrangements : ℕ := arrange_book_groups * arrange_math_books * arrange_english_books

theorem book_arrangement_theorem : total_arrangements = 1152 := by
  sorry

end book_arrangement_theorem_l3332_333248


namespace square_sum_equals_90_l3332_333259

theorem square_sum_equals_90 (x y : ℝ) (h1 : x + 3*y = 6) (h2 : x*y = -9) : 
  x^2 + 9*y^2 = 90 := by
sorry

end square_sum_equals_90_l3332_333259
