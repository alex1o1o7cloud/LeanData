import Mathlib

namespace pirate_treasure_division_l1666_166612

theorem pirate_treasure_division (N : ℕ) (h : 3000 ≤ N ∧ N ≤ 4000) :
  let remaining1 := (3 * N - 6) / 4
  let remaining2 := (9 * N - 42) / 16
  let remaining3 := (108 * N - 888) / 256
  let remaining4 := (82944 * N - 876400) / 262144
  let share1 := (N + 6) / 4
  let share2 := (3 * N + 18) / 16
  let share3 := (9 * N + 54) / 64
  let share4 := (108 * N + 648) / 1024
  let final_share := remaining4 / 4
  (share1 + final_share = 1178) ∧
  (share2 + final_share = 954) ∧
  (share3 + final_share = 786) ∧
  (share4 + final_share = 660) := by
sorry

end pirate_treasure_division_l1666_166612


namespace tangent_line_equation_l1666_166616

theorem tangent_line_equation (x y : ℝ) : 
  y = 2 * x * Real.tan x →
  (2 + Real.pi / 2) * (Real.pi / 4) - (Real.pi / 2) - Real.pi^2 / 4 = 0 →
  (2 + Real.pi / 2) * x - y - Real.pi^2 / 4 = 0 := by
  sorry

end tangent_line_equation_l1666_166616


namespace min_value_xy_l1666_166695

theorem min_value_xy (x y : ℕ+) (h1 : x^2 + y^2 - 2017*(x:ℕ)*(y:ℕ) > 0) 
  (h2 : ∃ (z : ℕ), z^2 ≠ x^2 + y^2 - 2017*(x:ℕ)*(y:ℕ)) : 
  x^2 + y^2 - 2017*(x:ℕ)*(y:ℕ) ≥ 2019 := by
sorry

end min_value_xy_l1666_166695


namespace simplify_square_roots_l1666_166635

theorem simplify_square_roots : 
  Real.sqrt (5 * 3) * Real.sqrt (3^4 * 5^2) = 45 * Real.sqrt 15 := by
  sorry

end simplify_square_roots_l1666_166635


namespace alyssa_grape_cost_l1666_166611

/-- The amount Alyssa paid for cherries in dollars -/
def cherry_cost : ℚ := 9.85

/-- The total amount Alyssa spent in dollars -/
def total_spent : ℚ := 21.93

/-- The amount Alyssa paid for grapes in dollars -/
def grape_cost : ℚ := total_spent - cherry_cost

theorem alyssa_grape_cost : grape_cost = 12.08 := by
  sorry

end alyssa_grape_cost_l1666_166611


namespace alice_most_dogs_l1666_166633

-- Define the number of cats and dogs for each person
variable (Kc Ac Bc Kd Ad Bd : ℕ)

-- Define the conditions
axiom kathy_more_cats : Kc > Ac
axiom kathy_more_dogs : Kd > Bd
axiom alice_more_dogs : Ad > Kd
axiom bruce_more_cats : Bc > Ac

-- Theorem to prove
theorem alice_most_dogs : Ad > Kd ∧ Ad > Bd :=
sorry

end alice_most_dogs_l1666_166633


namespace remainder_theorem_l1666_166654

theorem remainder_theorem (n : ℕ) (h1 : n > 0) (h2 : (n + 1) % 6 = 4) : n % 2 = 1 := by
  sorry

end remainder_theorem_l1666_166654


namespace incorrect_number_calculation_l1666_166677

theorem incorrect_number_calculation (n : ℕ) (initial_avg correct_avg correct_num : ℝ) :
  n = 10 ∧ initial_avg = 19 ∧ correct_avg = 24 ∧ correct_num = 76 →
  ∃ (incorrect_num : ℝ),
    n * initial_avg + (correct_num - incorrect_num) = n * correct_avg ∧
    incorrect_num = 26 := by
  sorry

end incorrect_number_calculation_l1666_166677


namespace black_pens_removed_l1666_166693

/-- Proves that 7 black pens were removed from a jar given the initial and final conditions -/
theorem black_pens_removed (initial_blue : ℕ) (initial_black : ℕ) (initial_red : ℕ)
  (blue_removed : ℕ) (final_count : ℕ)
  (h1 : initial_blue = 9)
  (h2 : initial_black = 21)
  (h3 : initial_red = 6)
  (h4 : blue_removed = 4)
  (h5 : final_count = 25) :
  initial_black - (initial_blue + initial_black + initial_red - blue_removed - final_count) = 7 := by
  sorry

#check black_pens_removed

end black_pens_removed_l1666_166693


namespace claudia_earnings_l1666_166631

def class_price : ℕ := 10
def saturday_attendance : ℕ := 20
def sunday_attendance : ℕ := saturday_attendance / 2

def total_earnings : ℕ := class_price * (saturday_attendance + sunday_attendance)

theorem claudia_earnings : total_earnings = 300 := by
  sorry

end claudia_earnings_l1666_166631


namespace sandy_jacket_return_l1666_166697

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℝ := 13.99

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℝ := 12.14

/-- The net amount Sandy spent on clothes -/
def net_spent : ℝ := 18.70

/-- The amount Sandy received for returning the jacket -/
def jacket_return : ℝ := shorts_cost + shirt_cost - net_spent

theorem sandy_jacket_return : jacket_return = 7.43 := by
  sorry

end sandy_jacket_return_l1666_166697


namespace same_row_twice_l1666_166618

theorem same_row_twice (num_rows : Nat) (num_people : Nat) :
  num_rows = 7 →
  num_people = 50 →
  ∃ (p1 p2 : Nat) (r : Nat),
    p1 ≠ p2 ∧
    p1 < num_people ∧
    p2 < num_people ∧
    r < num_rows ∧
    (∃ (morning_seating afternoon_seating : Nat → Nat),
      morning_seating p1 = r ∧
      morning_seating p2 = r ∧
      afternoon_seating p1 = r ∧
      afternoon_seating p2 = r) :=
by sorry

end same_row_twice_l1666_166618


namespace max_k_for_arithmetic_sequences_l1666_166686

/-- An arithmetic sequence -/
def ArithmeticSequence (a d : ℝ) : ℕ → ℝ := fun n ↦ a + (n - 1) * d

theorem max_k_for_arithmetic_sequences (a₁ a₂ d₁ d₂ : ℝ) (k : ℕ) :
  k > 1 →
  (ArithmeticSequence a₁ d₁ (k - 1)) * (ArithmeticSequence a₂ d₂ (k - 1)) = 42 →
  (ArithmeticSequence a₁ d₁ k) * (ArithmeticSequence a₂ d₂ k) = 30 →
  (ArithmeticSequence a₁ d₁ (k + 1)) * (ArithmeticSequence a₂ d₂ (k + 1)) = 16 →
  a₁ = a₂ →
  k ≤ 14 ∧ ∃ (a d₁ d₂ : ℝ), k = 14 ∧
    (ArithmeticSequence a d₁ 13) * (ArithmeticSequence a d₂ 13) = 42 ∧
    (ArithmeticSequence a d₁ 14) * (ArithmeticSequence a d₂ 14) = 30 ∧
    (ArithmeticSequence a d₁ 15) * (ArithmeticSequence a d₂ 15) = 16 := by
  sorry

end max_k_for_arithmetic_sequences_l1666_166686


namespace share_distribution_l1666_166640

theorem share_distribution (total : ℝ) (share_a : ℝ) (share_b : ℝ) (share_c : ℝ) :
  total = 246 →
  share_b = 0.65 →
  share_c = 48 →
  share_a + share_b + share_c = 1 →
  share_c / total = 48 / 246 :=
by sorry

end share_distribution_l1666_166640


namespace root_implies_k_value_l1666_166637

theorem root_implies_k_value (k : ℝ) : 
  (6 * ((-25 - Real.sqrt 409) / 12)^2 + 25 * ((-25 - Real.sqrt 409) / 12) + k = 0) → k = 9 := by
  sorry

end root_implies_k_value_l1666_166637


namespace tournament_games_count_l1666_166659

/-- Represents a basketball tournament with a preliminary round and main tournament. -/
structure BasketballTournament where
  preliminaryTeams : Nat
  preliminarySpots : Nat
  mainTournamentTeams : Nat

/-- Calculates the number of games in the preliminary round. -/
def preliminaryGames (t : BasketballTournament) : Nat :=
  t.preliminarySpots

/-- Calculates the number of games in the main tournament. -/
def mainTournamentGames (t : BasketballTournament) : Nat :=
  t.mainTournamentTeams - 1

/-- Calculates the total number of games in the entire tournament. -/
def totalGames (t : BasketballTournament) : Nat :=
  preliminaryGames t + mainTournamentGames t

/-- Theorem stating that the total number of games in the specific tournament setup is 17. -/
theorem tournament_games_count :
  ∃ (t : BasketballTournament),
    t.preliminaryTeams = 4 ∧
    t.preliminarySpots = 2 ∧
    t.mainTournamentTeams = 16 ∧
    totalGames t = 17 := by
  sorry

end tournament_games_count_l1666_166659


namespace initial_quarters_l1666_166628

def quarters_after_events (initial : ℕ) : ℕ :=
  let after_doubling := initial * 2
  let after_second_year := after_doubling + 3 * 12
  let after_third_year := after_second_year + 4
  let before_loss := after_third_year
  (before_loss * 3) / 4

theorem initial_quarters (initial : ℕ) : 
  quarters_after_events initial = 105 ↔ initial = 50 := by
  sorry

#eval quarters_after_events 50  -- Should output 105

end initial_quarters_l1666_166628


namespace tricycle_count_l1666_166634

theorem tricycle_count (total_children : ℕ) (total_wheels : ℕ) 
  (h1 : total_children = 10) 
  (h2 : total_wheels = 26) : ∃ (bicycles tricycles : ℕ),
  bicycles + tricycles = total_children ∧ 
  2 * bicycles + 3 * tricycles = total_wheels ∧
  tricycles = 6 := by
  sorry

end tricycle_count_l1666_166634


namespace factorial_ratio_l1666_166625

theorem factorial_ratio : (12 : ℕ).factorial / (11 : ℕ).factorial = 12 := by
  sorry

end factorial_ratio_l1666_166625


namespace distinct_values_of_d_l1666_166679

theorem distinct_values_of_d (d : ℂ) (u v w x : ℂ) 
  (h_distinct : u ≠ v ∧ u ≠ w ∧ u ≠ x ∧ v ≠ w ∧ v ≠ x ∧ w ≠ x)
  (h_eq : ∀ z : ℂ, (z - u) * (z - v) * (z - w) * (z - x) = 
                   (z - d*u) * (z - d*v) * (z - d*w) * (z - d*x)) :
  ∃! (S : Finset ℂ), S.card = 4 ∧ ∀ d' : ℂ, d' ∈ S ↔ 
    (∀ z : ℂ, (z - u) * (z - v) * (z - w) * (z - x) = 
              (z - d'*u) * (z - d'*v) * (z - d'*w) * (z - d'*x)) :=
by sorry

end distinct_values_of_d_l1666_166679


namespace min_value_ratio_l1666_166674

-- Define the arithmetic and geometric sequence properties
def is_arithmetic_sequence (x a b y : ℝ) : Prop :=
  a + b = x + y

def is_geometric_sequence (x c d y : ℝ) : Prop :=
  c * d = x * y

-- State the theorem
theorem min_value_ratio (x y a b c d : ℝ) 
  (hx : x > 0) (hy : y > 0)
  (ha : is_arithmetic_sequence x a b y)
  (hg : is_geometric_sequence x c d y) :
  (a + b)^2 / (c * d) ≥ 4 ∧ 
  ∃ (a b c d : ℝ), (a + b)^2 / (c * d) = 4 :=
sorry

end min_value_ratio_l1666_166674


namespace equation_solution_l1666_166629

theorem equation_solution :
  ∃ (x : ℝ), 
    (3*x - 1 ≥ 0) ∧ 
    (x + 4 > 0) ∧ 
    (Real.sqrt ((3*x - 1) / (x + 4)) + 3 - 4 * Real.sqrt ((x + 4) / (3*x - 1)) = 0) ∧
    (x = 5/2) := by
  sorry

end equation_solution_l1666_166629


namespace corrected_mean_problem_l1666_166613

/-- Calculates the corrected mean of observations after fixing an error --/
def corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * original_mean + (correct_value - incorrect_value)) / n

/-- Theorem stating the corrected mean for the given problem --/
theorem corrected_mean_problem :
  corrected_mean 50 36 23 34 = 36.22 := by
  sorry

#eval corrected_mean 50 36 23 34

end corrected_mean_problem_l1666_166613


namespace probability_both_from_c_l1666_166644

structure Workshop where
  name : String
  quantity : Nat

def total_quantity (workshops : List Workshop) : Nat :=
  workshops.foldl (fun acc w => acc + w.quantity) 0

def sample_size : Nat := 6

def stratified_sample (w : Workshop) (total : Nat) : Nat :=
  w.quantity * sample_size / total

theorem probability_both_from_c (workshops : List Workshop) :
  let total := total_quantity workshops
  let c_workshop := workshops.find? (fun w => w.name = "C")
  match c_workshop with
  | some c =>
    let c_samples := stratified_sample c total
    (c_samples.choose 2) / (sample_size.choose 2) = 1 / 5
  | none => False
  := by sorry

end probability_both_from_c_l1666_166644


namespace gems_per_dollar_l1666_166664

/-- Proves that the number of gems per dollar is 100 given the problem conditions -/
theorem gems_per_dollar (total_spent : ℝ) (bonus_rate : ℝ) (final_gems : ℝ) :
  total_spent = 250 →
  bonus_rate = 0.2 →
  final_gems = 30000 →
  (final_gems / (total_spent * (1 + bonus_rate))) = 100 := by
sorry

end gems_per_dollar_l1666_166664


namespace limit_fraction_is_two_l1666_166645

theorem limit_fraction_is_two : ∀ ε > 0, ∃ N : ℕ, ∀ n > N,
  |((2 * n - 3 : ℝ) / (n + 2 : ℝ)) - 2| < ε :=
by sorry

end limit_fraction_is_two_l1666_166645


namespace decimal_524_to_octal_l1666_166653

-- Define a function to convert decimal to octal
def decimalToOctal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec helper (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else helper (m / 8) ((m % 8) :: acc)
    helper n []

-- Theorem statement
theorem decimal_524_to_octal :
  decimalToOctal 524 = [1, 0, 1, 4] := by sorry

end decimal_524_to_octal_l1666_166653


namespace chocolate_difference_l1666_166638

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 10

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- Theorem stating the difference in chocolate consumption between Robert and Nickel -/
theorem chocolate_difference : robert_chocolates - nickel_chocolates = 5 := by
  sorry

end chocolate_difference_l1666_166638


namespace open_set_classification_l1666_166672

-- Define the concept of an open set in R²
def is_open_set (A : Set (ℝ × ℝ)) : Prop :=
  ∀ (p : ℝ × ℝ), p ∈ A → ∃ (r : ℝ), r > 0 ∧ 
    {q : ℝ × ℝ | Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2) < r} ⊆ A

-- Define the sets
def set1 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def set2 : Set (ℝ × ℝ) := {p | p.1 + p.2 + 2 > 0}
def set3 : Set (ℝ × ℝ) := {p | |p.1 + p.2| ≤ 6}
def set4 : Set (ℝ × ℝ) := {p | 0 < p.1^2 + (p.2 - Real.sqrt 2)^2 ∧ p.1^2 + (p.2 - Real.sqrt 2)^2 < 1}

-- State the theorem
theorem open_set_classification :
  ¬(is_open_set set1) ∧
  (is_open_set set2) ∧
  ¬(is_open_set set3) ∧
  (is_open_set set4) :=
sorry

end open_set_classification_l1666_166672


namespace edge_probability_is_three_nineteenths_l1666_166605

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  vertex_degree : ∀ v : Fin 20, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of selecting two vertices that form an edge in a regular dodecahedron -/
def edge_probability (d : RegularDodecahedron) : ℚ :=
  d.edges.card / Nat.choose 20 2

/-- Theorem stating the probability of selecting two vertices that form an edge -/
theorem edge_probability_is_three_nineteenths (d : RegularDodecahedron) :
  edge_probability d = 3 / 19 := by
  sorry

end edge_probability_is_three_nineteenths_l1666_166605


namespace maple_trees_planted_l1666_166690

theorem maple_trees_planted (initial_maple : ℕ) (final_maple : ℕ) :
  initial_maple = 2 →
  final_maple = 11 →
  final_maple - initial_maple = 9 :=
by sorry

end maple_trees_planted_l1666_166690


namespace min_packs_for_126_cans_l1666_166692

/-- Represents the number of cans in each pack size --/
inductive PackSize
| small : PackSize
| medium : PackSize
| large : PackSize

/-- Returns the number of cans for a given pack size --/
def cansInPack (p : PackSize) : ℕ :=
  match p with
  | PackSize.small => 15
  | PackSize.medium => 18
  | PackSize.large => 36

/-- Represents a combination of packs --/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans in a pack combination --/
def totalCans (c : PackCombination) : ℕ :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Calculates the total number of packs in a combination --/
def totalPacks (c : PackCombination) : ℕ :=
  c.small + c.medium + c.large

/-- Defines what it means for a pack combination to be valid --/
def isValidCombination (c : PackCombination) : Prop :=
  totalCans c = 126

/-- Theorem: The minimum number of packs needed to buy exactly 126 cans is 4 --/
theorem min_packs_for_126_cans :
  ∃ (c : PackCombination), isValidCombination c ∧
    totalPacks c = 4 ∧
    (∀ (c' : PackCombination), isValidCombination c' → totalPacks c ≤ totalPacks c') :=
sorry

end min_packs_for_126_cans_l1666_166692


namespace equation_solution_l1666_166699

theorem equation_solution (x y z k : ℝ) :
  (9 / (x - y) = k / (x + z)) ∧ (k / (x + z) = 16 / (z + y)) → k = 25 := by
  sorry

end equation_solution_l1666_166699


namespace total_children_count_l1666_166687

/-- The number of toy cars given to boys -/
def toy_cars : ℕ := 134

/-- The number of dolls given to girls -/
def dolls : ℕ := 269

/-- The number of board games given to both boys and girls -/
def board_games : ℕ := 87

/-- Every child received only one toy -/
axiom one_toy_per_child : True

/-- The total number of children attending the event -/
def total_children : ℕ := toy_cars + dolls

theorem total_children_count : total_children = 403 := by sorry

end total_children_count_l1666_166687


namespace expression_evaluation_l1666_166656

theorem expression_evaluation (b c : ℕ) (h1 : b = 2) (h2 : c = 3) : 
  (b^3 * b^4) + c^2 = 137 := by
sorry

end expression_evaluation_l1666_166656


namespace complement_union_A_B_complement_A_inter_B_l1666_166651

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Theorem for the first part
theorem complement_union_A_B : 
  (Aᶜ ∪ Bᶜ) = {x | x ≤ 2 ∨ x ≥ 10} := by sorry

-- Theorem for the second part
theorem complement_A_inter_B : 
  (Aᶜ ∩ B) = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

end complement_union_A_B_complement_A_inter_B_l1666_166651


namespace line_through_parabola_focus_l1666_166682

/-- The value of 'a' for a line ax - y + 1 = 0 passing through the focus of the parabola y^2 = 4x -/
theorem line_through_parabola_focus (a : ℝ) : 
  (∃ x y : ℝ, y^2 = 4*x ∧ a*x - y + 1 = 0 ∧ x = 1 ∧ y = 0) → a = -1 :=
by sorry

end line_through_parabola_focus_l1666_166682


namespace unit_square_tiling_l1666_166600

/-- A rectangle is considered "good" if it can be tiled by rectangles similar to 1 × (3 + ∛3) -/
def is_good (a b : ℝ) : Prop := sorry

/-- The scaling property of good rectangles -/
axiom good_scale (a b c : ℝ) (h : c > 0) :
  is_good a b → is_good (a * c) (b * c)

/-- The integer multiple property of good rectangles -/
axiom good_int_multiple (m n : ℝ) (j : ℕ) (h : j > 0) :
  is_good m n → is_good m (n * j)

/-- The main theorem: the unit square can be tiled with rectangles similar to 1 × (3 + ∛3) -/
theorem unit_square_tiling :
  ∃ (tiling : Set (ℝ × ℝ)), 
    (∀ (rect : ℝ × ℝ), rect ∈ tiling → is_good rect.1 rect.2) ∧
    (∃ (f : ℝ × ℝ → ℝ × ℝ), 
      (∀ x y, f (x, y) = (x, y)) ∧
      (∀ (rect : ℝ × ℝ), rect ∈ tiling → 
        ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ f (rect.1, rect.2) = (a, b) ∧ 
        b / a = 3 + Real.rpow 3 (1/3 : ℝ))) :=
sorry

end unit_square_tiling_l1666_166600


namespace rational_sum_and_sum_of_squares_coprime_to_six_l1666_166698

theorem rational_sum_and_sum_of_squares_coprime_to_six (a b : ℚ) :
  let S := a + b
  (S = a + b) → (S = a^2 + b^2) → ∃ (m k : ℤ), S = m / k ∧ k ≠ 0 ∧ Nat.Coprime k.natAbs 6 := by
  sorry

end rational_sum_and_sum_of_squares_coprime_to_six_l1666_166698


namespace bryans_book_collection_l1666_166652

theorem bryans_book_collection (books_per_continent : ℕ) (total_books : ℕ) 
  (h1 : books_per_continent = 122) 
  (h2 : total_books = 488) : 
  total_books / books_per_continent = 4 := by
sorry

end bryans_book_collection_l1666_166652


namespace number_division_problem_l1666_166647

theorem number_division_problem (x : ℚ) : x / 5 = 80 + x / 6 → x = 2400 := by
  sorry

end number_division_problem_l1666_166647


namespace smallest_lcm_for_four_digit_gcd_five_l1666_166627

theorem smallest_lcm_for_four_digit_gcd_five (m n : ℕ) : 
  m ≥ 1000 ∧ m ≤ 9999 ∧ n ≥ 1000 ∧ n ≤ 9999 ∧ Nat.gcd m n = 5 →
  Nat.lcm m n ≥ 203010 :=
by sorry

end smallest_lcm_for_four_digit_gcd_five_l1666_166627


namespace pear_apple_equivalence_l1666_166680

/-- The cost of fruits at Joe's Fruit Stand -/
structure FruitCost where
  pear : ℕ
  grape : ℕ
  apple : ℕ

/-- The relation between pears and grapes -/
def pear_grape_relation (c : FruitCost) : Prop :=
  4 * c.pear = 3 * c.grape

/-- The relation between grapes and apples -/
def grape_apple_relation (c : FruitCost) : Prop :=
  9 * c.grape = 6 * c.apple

/-- Theorem stating the cost equivalence of 24 pears and 12 apples -/
theorem pear_apple_equivalence (c : FruitCost) 
  (h1 : pear_grape_relation c) 
  (h2 : grape_apple_relation c) : 
  24 * c.pear = 12 * c.apple :=
by
  sorry

end pear_apple_equivalence_l1666_166680


namespace expression_value_l1666_166642

theorem expression_value (a b c : ℤ) :
  a = 18 ∧ b = 20 ∧ c = 22 →
  (a - (b - c)) - ((a - b) - c) = 44 := by
sorry

end expression_value_l1666_166642


namespace arithmetic_sequence_nth_term_l1666_166691

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_nth_term
  (a : ℕ → ℝ)
  (d : ℝ)
  (h : ArithmeticSequence a d)
  (h1 : a 1 = 11)
  (h2 : d = 2)
  (h3 : ∃ n : ℕ, a n = 2009) :
  ∃ n : ℕ, n = 1000 ∧ a n = 2009 :=
sorry

end arithmetic_sequence_nth_term_l1666_166691


namespace systematic_sampling_l1666_166614

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Systematic
  | Stratified

/-- Represents an auditorium -/
structure Auditorium where
  totalSeats : Nat
  seatsPerRow : Nat

/-- Represents a selection of students -/
structure Selection where
  seatNumber : Nat
  count : Nat

/-- Determines the sampling method based on the auditorium and selection -/
def determineSamplingMethod (a : Auditorium) (s : Selection) : SamplingMethod :=
  sorry

/-- Theorem: Selecting students with seat number 15 from the given auditorium is a systematic sampling method -/
theorem systematic_sampling (a : Auditorium) (s : Selection) :
  a.totalSeats = 25 →
  a.seatsPerRow = 20 →
  s.seatNumber = 15 →
  s.count = 25 →
  determineSamplingMethod a s = SamplingMethod.Systematic :=
  sorry

end systematic_sampling_l1666_166614


namespace sara_sent_nine_letters_in_february_l1666_166604

/-- The number of letters Sara sent in February -/
def letters_in_february : ℕ := 33 - (6 + 3 * 6)

/-- Proof that Sara sent 9 letters in February -/
theorem sara_sent_nine_letters_in_february :
  letters_in_february = 9 := by
  sorry

#eval letters_in_february

end sara_sent_nine_letters_in_february_l1666_166604


namespace import_export_scientific_notation_l1666_166683

def billion : ℝ := 1000000000

theorem import_export_scientific_notation (volume : ℝ) (h : volume = 214.7 * billion) :
  ∃ (a : ℝ) (n : ℤ), volume = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = 10 := by
  sorry

end import_export_scientific_notation_l1666_166683


namespace a_2016_value_l1666_166658

def sequence_sum (n : ℕ) : ℕ := n ^ 2

theorem a_2016_value :
  let a : ℕ → ℕ := fun n => sequence_sum n - sequence_sum (n - 1)
  a 2016 = 4031 := by
  sorry

end a_2016_value_l1666_166658


namespace sought_hyperbola_satisfies_conditions_l1666_166663

/-- Given hyperbola equation -/
def given_hyperbola (x y : ℝ) : Prop :=
  x^2 / 5 - y^2 / 4 = 1

/-- Asymptotes of the given hyperbola -/
def given_asymptotes (x y : ℝ) : Prop :=
  y = (2 / Real.sqrt 5) * x ∨ y = -(2 / Real.sqrt 5) * x

/-- The equation of the sought hyperbola -/
def sought_hyperbola (x y : ℝ) : Prop :=
  5 * y^2 / 4 - x^2 = 1

/-- Theorem stating that the sought hyperbola satisfies the required conditions -/
theorem sought_hyperbola_satisfies_conditions :
  (∀ x y : ℝ, given_asymptotes x y ↔ (y = (2 / Real.sqrt 5) * x ∨ y = -(2 / Real.sqrt 5) * x)) ∧
  sought_hyperbola 2 2 :=
sorry


end sought_hyperbola_satisfies_conditions_l1666_166663


namespace inequality_proof_l1666_166602

theorem inequality_proof (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + 2) * (b + 2) ≥ c * d := by
  sorry

end inequality_proof_l1666_166602


namespace f_always_above_y_l1666_166626

/-- The function f(x) = mx^2 - 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 3

/-- The line y = mx - m -/
def y (m : ℝ) (x : ℝ) : ℝ := m * x - m

/-- Theorem stating that f(x) is always above y for all real x if and only if m > 4 -/
theorem f_always_above_y (m : ℝ) : 
  (∀ x : ℝ, f m x > y m x) ↔ m > 4 := by
  sorry

end f_always_above_y_l1666_166626


namespace good_set_closed_under_addition_l1666_166641

-- Define a "good set"
def is_good_set (A : Set ℚ) : Prop :=
  (0 ∈ A) ∧ (1 ∈ A) ∧
  (∀ x y, x ∈ A → y ∈ A → (x - y) ∈ A) ∧
  (∀ x, x ∈ A → x ≠ 0 → (1 / x) ∈ A)

-- Theorem statement
theorem good_set_closed_under_addition (A : Set ℚ) (h : is_good_set A) :
  ∀ x y, x ∈ A → y ∈ A → (x + y) ∈ A :=
by sorry

end good_set_closed_under_addition_l1666_166641


namespace f_derivative_at_zero_l1666_166675

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x

theorem f_derivative_at_zero : 
  deriv f 0 = 2 := by sorry

end f_derivative_at_zero_l1666_166675


namespace odot_ten_five_l1666_166646

-- Define the ⊙ operation
def odot (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

-- Theorem statement
theorem odot_ten_five : odot 10 5 = 38 / 3 := by
  sorry

end odot_ten_five_l1666_166646


namespace parabola_triangle_property_l1666_166661

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line y = x + 3
def line (x y : ℝ) : Prop := y = x + 3

-- Define a point on the parabola
structure PointOnParabola (p : ℝ) where
  x : ℝ
  y : ℝ
  on_parabola : parabola p x y

-- Define the theorem
theorem parabola_triangle_property (p : ℝ) :
  parabola p 1 2 →  -- The parabola passes through (1, 2)
  ∀ (A : PointOnParabola p),
    A.x ≠ 1 ∨ A.y ≠ 2 →  -- A is different from (1, 2)
    ∃ (P B : ℝ × ℝ),
      -- P is on the line AC and y = x + 3
      (∃ t : ℝ, P.1 = (1 - t) * 1 + t * A.x ∧ P.2 = (1 - t) * 2 + t * A.y) ∧
      line P.1 P.2 ∧
      -- B is on the parabola and has the same y-coordinate as P
      parabola p B.1 B.2 ∧ B.2 = P.2 →
      -- 1. AB passes through (3, 2)
      (∃ s : ℝ, 3 = (1 - s) * A.x + s * B.1 ∧ 2 = (1 - s) * A.y + s * B.2) ∧
      -- 2. The minimum area of triangle ABC is 4√2
      (∀ (area : ℝ), area ≥ 0 ∧ area * area = 32 → 
        ∃ (A' : PointOnParabola p) (P' B' : ℝ × ℝ),
          A'.x ≠ 1 ∨ A'.y ≠ 2 ∧
          (∃ t : ℝ, P'.1 = (1 - t) * 1 + t * A'.x ∧ P'.2 = (1 - t) * 2 + t * A'.y) ∧
          line P'.1 P'.2 ∧
          parabola p B'.1 B'.2 ∧ B'.2 = P'.2 ∧
          area = (1/2) * Real.sqrt ((A'.x - 1)^2 + (A'.y - 2)^2) * Real.sqrt ((B'.1 - 1)^2 + (B'.2 - 2)^2)) :=
by sorry

end parabola_triangle_property_l1666_166661


namespace function_equivalence_l1666_166671

theorem function_equivalence (f : ℝ → ℝ) (h : ∀ x, f (2 * x + 1) = 2 * x^2 + 1) :
  ∀ x, f x = 1/2 * x^2 - x + 3/2 := by
sorry

end function_equivalence_l1666_166671


namespace shopping_ratio_l1666_166655

theorem shopping_ratio : 
  let emma_spent : ℕ := 58
  let elsa_spent : ℕ := 2 * emma_spent
  let total_spent : ℕ := 638
  let elizabeth_spent : ℕ := total_spent - (emma_spent + elsa_spent)
  (elizabeth_spent : ℚ) / (elsa_spent : ℚ) = 4 / 1 := by
sorry

end shopping_ratio_l1666_166655


namespace rectangular_prism_width_l1666_166636

theorem rectangular_prism_width 
  (l h w d : ℝ) 
  (h_def : h = 2 * l)
  (l_val : l = 5)
  (diagonal : d = 17)
  (diag_eq : d^2 = l^2 + w^2 + h^2) :
  w = 2 * Real.sqrt 41 := by
sorry

end rectangular_prism_width_l1666_166636


namespace plot_length_is_61_l1666_166678

/-- Proves that the length of a rectangular plot is 61 meters given the specified conditions. -/
theorem plot_length_is_61 (breadth : ℝ) (length : ℝ) (fencing_cost_per_meter : ℝ) (total_fencing_cost : ℝ) :
  length = breadth + 22 →
  fencing_cost_per_meter = 26.5 →
  total_fencing_cost = 5300 →
  fencing_cost_per_meter * (2 * length + 2 * breadth) = total_fencing_cost →
  length = 61 := by
  sorry

end plot_length_is_61_l1666_166678


namespace sum_four_characterization_l1666_166666

/-- Represents the outcome of rolling a single die -/
def DieOutcome := Fin 6

/-- Represents the outcome of rolling two dice -/
def TwoDiceOutcome := DieOutcome × DieOutcome

/-- The sum of points obtained when rolling two dice -/
def sumPoints (outcome : TwoDiceOutcome) : Nat :=
  outcome.1.val + 1 + outcome.2.val + 1

/-- The event where the sum of points is 4 -/
def sumIsFour (outcome : TwoDiceOutcome) : Prop :=
  sumPoints outcome = 4

/-- The event where one die shows 3 and the other shows 1 -/
def threeAndOne (outcome : TwoDiceOutcome) : Prop :=
  (outcome.1.val = 2 ∧ outcome.2.val = 0) ∨ (outcome.1.val = 0 ∧ outcome.2.val = 2)

/-- The event where both dice show 2 -/
def bothTwo (outcome : TwoDiceOutcome) : Prop :=
  outcome.1.val = 1 ∧ outcome.2.val = 1

theorem sum_four_characterization (outcome : TwoDiceOutcome) :
  sumIsFour outcome ↔ threeAndOne outcome ∨ bothTwo outcome := by
  sorry

end sum_four_characterization_l1666_166666


namespace coplanar_vectors_lambda_l1666_166667

/-- Given three vectors a, b, and c in R³, if they are coplanar and have specific coordinates,
    then the third coordinate of c is 65/7. -/
theorem coplanar_vectors_lambda (a b c : ℝ × ℝ × ℝ) :
  a = (2, -1, 3) →
  b = (-1, 4, -2) →
  c.1 = 7 ∧ c.2.1 = 5 →
  (∃ (p q : ℝ), c = p • a + q • b) →
  c.2.2 = 65 / 7 := by
  sorry

end coplanar_vectors_lambda_l1666_166667


namespace john_photos_count_l1666_166657

/-- The number of photos each person brings and the total slots in the album --/
def photo_problem (cristina_photos sarah_photos clarissa_photos total_slots : ℕ) : Prop :=
  ∃ john_photos : ℕ,
    john_photos = total_slots - (cristina_photos + sarah_photos + clarissa_photos)

/-- Theorem stating that John brings 10 photos given the problem conditions --/
theorem john_photos_count :
  photo_problem 7 9 14 40 → ∃ john_photos : ℕ, john_photos = 10 :=
by
  sorry

end john_photos_count_l1666_166657


namespace not_in_range_iff_c_in_interval_l1666_166639

/-- The function g(x) defined in terms of c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + c*x + 3

/-- Theorem stating that -3 is not in the range of g(x) iff c ∈ (-2√6, 2√6) -/
theorem not_in_range_iff_c_in_interval (c : ℝ) : 
  (∀ x : ℝ, g c x ≠ -3) ↔ c ∈ Set.Ioo (-2 * Real.sqrt 6) (2 * Real.sqrt 6) :=
by sorry

end not_in_range_iff_c_in_interval_l1666_166639


namespace problem_solution_l1666_166688

/-- Represents the contents of a box of colored balls -/
structure Box where
  red : Nat
  yellow : Nat
  blue : Nat

/-- Calculates the probability of Person B winning given the contents of both boxes -/
def probability_b_wins (box_a box_b : Box) : Rat :=
  let total_a := box_a.red + box_a.yellow + box_a.blue
  let total_b := box_b.red + box_b.yellow + box_b.blue
  ((box_a.red * box_b.red) + (box_a.yellow * box_b.yellow) + (box_a.blue * box_b.blue)) / (total_a * total_b)

/-- Calculates the average score for Person B given the contents of both boxes -/
def average_score_b (box_a box_b : Box) : Rat :=
  let total_a := box_a.red + box_a.yellow + box_a.blue
  let total_b := box_b.red + box_b.yellow + box_b.blue
  ((box_a.red * box_b.red * 1) + (box_a.yellow * box_b.yellow * 2) + (box_a.blue * box_b.blue * 3)) / (total_a * total_b)

theorem problem_solution :
  let box_a : Box := ⟨3, 2, 1⟩
  let box_b1 : Box := ⟨1, 2, 3⟩
  let box_b2 : Box := ⟨1, 4, 1⟩
  (probability_b_wins box_a box_b1 = 5/18) ∧
  (average_score_b box_a box_b2 = 11/18) ∧
  (∀ (x y z : Nat), x + y + z = 6 → average_score_b box_a ⟨x, y, z⟩ ≤ 11/18) := by
  sorry

end problem_solution_l1666_166688


namespace contribution_increase_l1666_166696

theorem contribution_increase (initial_contributions : ℕ) (initial_average : ℚ) (new_contribution : ℚ) :
  initial_contributions = 3 →
  initial_average = 75 →
  new_contribution = 150 →
  let total_initial := initial_contributions * initial_average
  let new_total := total_initial + new_contribution
  let new_average := new_total / (initial_contributions + 1)
  let increase := new_average - initial_average
  let percentage_increase := (increase / initial_average) * 100
  percentage_increase = 25 := by
  sorry

end contribution_increase_l1666_166696


namespace minimum_soldiers_to_add_l1666_166649

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) :
  (84 - N % 84) = 82 :=
sorry

end minimum_soldiers_to_add_l1666_166649


namespace solve_equation_for_x_l1666_166643

theorem solve_equation_for_x :
  ∃ X : ℝ, 1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * X)) = 1200.0000000000002 ∧ X = 0.5 := by
  sorry

end solve_equation_for_x_l1666_166643


namespace parallelogram_altitude_base_ratio_l1666_166676

theorem parallelogram_altitude_base_ratio 
  (area : ℝ) (base : ℝ) (altitude : ℝ) 
  (h_area : area = 288) 
  (h_base : base = 12) 
  (h_area_formula : area = base * altitude) : 
  altitude / base = 2 := by
sorry

end parallelogram_altitude_base_ratio_l1666_166676


namespace bridge_length_l1666_166669

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 235 :=
by sorry

end bridge_length_l1666_166669


namespace ratio_equality_l1666_166665

theorem ratio_equality (x : ℝ) : (0.6 / x = 5 / 8) → x = 0.96 := by
  sorry

end ratio_equality_l1666_166665


namespace M_is_graph_of_square_function_l1666_166648

def M : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

theorem M_is_graph_of_square_function :
  M = {p : ℝ × ℝ | p.2 = p.1^2} := by sorry

end M_is_graph_of_square_function_l1666_166648


namespace amanda_purchase_cost_l1666_166632

def dress_price : ℚ := 50
def shoes_price : ℚ := 75
def dress_discount : ℚ := 0.30
def shoes_discount : ℚ := 0.25
def tax_rate : ℚ := 0.05

def total_cost : ℚ :=
  let dress_discounted := dress_price * (1 - dress_discount)
  let shoes_discounted := shoes_price * (1 - shoes_discount)
  let subtotal := dress_discounted + shoes_discounted
  let tax := subtotal * tax_rate
  subtotal + tax

theorem amanda_purchase_cost : total_cost = 95.81 := by
  sorry

end amanda_purchase_cost_l1666_166632


namespace computer_on_time_l1666_166673

/-- Represents days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Represents time of day in hours and minutes -/
structure Time where
  hour : ℕ
  minute : ℕ
  h_valid : hour < 24
  m_valid : minute < 60

/-- Represents a specific moment (day and time) -/
structure Moment where
  day : Day
  time : Time

def computer_on_duration : ℕ := 100

def computer_off_moment : Moment :=
  { day := Day.Friday
  , time := { hour := 17, minute := 0, h_valid := by norm_num, m_valid := by norm_num } }

theorem computer_on_time (on_moment off_moment : Moment) 
  (h : off_moment = computer_off_moment) 
  (duration : ℕ) (h_duration : duration = computer_on_duration) :
  on_moment = 
    { day := Day.Monday
    , time := { hour := 13, minute := 0, h_valid := by norm_num, m_valid := by norm_num } } :=
  sorry

end computer_on_time_l1666_166673


namespace percentage_calculation_l1666_166681

theorem percentage_calculation (P : ℝ) : 
  (P / 100) * 200 - 30 = 50 → P = 40 :=
by sorry

end percentage_calculation_l1666_166681


namespace gcd_g_10_g_13_l1666_166619

def g (x : ℤ) : ℤ := x^3 - 3*x^2 + x + 2050

theorem gcd_g_10_g_13 : Int.gcd (g 10) (g 13) = 1 := by sorry

end gcd_g_10_g_13_l1666_166619


namespace sixteenth_root_of_sixteen_l1666_166660

theorem sixteenth_root_of_sixteen (n : ℝ) : (16 : ℝ) ^ (1/4 : ℝ) = 2^n → n = 1 := by
  sorry

end sixteenth_root_of_sixteen_l1666_166660


namespace exclusive_movies_count_l1666_166621

/-- Given two movie collections belonging to Andrew and John, this theorem proves
    the number of movies that are in either collection but not both. -/
theorem exclusive_movies_count
  (total_andrew : ℕ)
  (shared : ℕ)
  (john_exclusive : ℕ)
  (h1 : total_andrew = 25)
  (h2 : shared = 15)
  (h3 : john_exclusive = 8) :
  total_andrew - shared + john_exclusive = 18 :=
by sorry

end exclusive_movies_count_l1666_166621


namespace shooting_competition_l1666_166620

theorem shooting_competition (hit_rate_A hit_rate_B prob_total_2 : ℚ) : 
  hit_rate_A = 3/5 →
  prob_total_2 = 9/20 →
  hit_rate_A * (1 - hit_rate_B) + (1 - hit_rate_A) * hit_rate_B = prob_total_2 →
  hit_rate_B = 3/4 := by
  sorry

end shooting_competition_l1666_166620


namespace club_members_count_l1666_166606

/-- The number of members in the club -/
def n : ℕ := sorry

/-- The age of the old (replaced) member -/
def O : ℕ := sorry

/-- The age of the new member -/
def N : ℕ := sorry

/-- The average age remains unchanged after replacement and 3 years -/
axiom avg_unchanged : (n * O + 3 * n) / n = (n * N + 3 * n) / n

/-- The difference between the ages of the replaced and new member is 15 -/
axiom age_difference : O - N = 15

/-- Theorem: The number of members in the club is 5 -/
theorem club_members_count : n = 5 := by sorry

end club_members_count_l1666_166606


namespace timothy_cows_l1666_166603

def total_cost : ℕ := 147700
def land_cost : ℕ := 30 * 20
def house_cost : ℕ := 120000
def cow_cost : ℕ := 1000
def chicken_cost : ℕ := 100 * 5
def solar_installation_cost : ℕ := 6 * 100
def solar_equipment_cost : ℕ := 6000

def other_costs : ℕ := land_cost + house_cost + chicken_cost + solar_installation_cost + solar_equipment_cost

theorem timothy_cows :
  (total_cost - other_costs) / cow_cost = 20 := by sorry

end timothy_cows_l1666_166603


namespace chocolate_bar_pieces_l1666_166662

theorem chocolate_bar_pieces :
  ∀ (total : ℕ),
  (total / 2 : ℕ) + (total / 4 : ℕ) + 15 = total →
  total = 60 :=
by
  sorry

end chocolate_bar_pieces_l1666_166662


namespace complement_A_union_B_A_inter_complement_B_l1666_166685

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 1 < x ∧ x < 9}

-- Theorem for the first part
theorem complement_A_union_B : 
  (Set.univ \ A) ∪ B = {x | x < 0 ∨ x > 1} := by sorry

-- Theorem for the second part
theorem A_inter_complement_B : 
  A ∩ (Set.univ \ B) = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end complement_A_union_B_A_inter_complement_B_l1666_166685


namespace imaginary_part_of_complex_fraction_l1666_166689

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (1 + 3*I) / (1 - I) → z.im = 2 := by
  sorry

end imaginary_part_of_complex_fraction_l1666_166689


namespace certain_number_solution_l1666_166601

theorem certain_number_solution : ∃ x : ℚ, (40 * 30 + (12 + 8) * x) / 5 = 1212 ∧ x = 3 := by
  sorry

end certain_number_solution_l1666_166601


namespace tangent_slope_three_cubic_l1666_166684

theorem tangent_slope_three_cubic (x : ℝ) : 
  (∃ y : ℝ, y = x^3 ∧ (3 * x^2 = 3)) ↔ (x = 1 ∨ x = -1) := by
  sorry

end tangent_slope_three_cubic_l1666_166684


namespace paving_stone_width_l1666_166694

/-- Theorem: Width of paving stones in a rectangular courtyard -/
theorem paving_stone_width
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (stone_length : ℝ)
  (stone_count : ℕ)
  (h1 : courtyard_length = 20)
  (h2 : courtyard_width = 16.5)
  (h3 : stone_length = 2.5)
  (h4 : stone_count = 66)
  : ∃ (stone_width : ℝ),
    courtyard_length * courtyard_width = stone_count * (stone_length * stone_width) ∧
    stone_width = 2 :=
by
  sorry

end paving_stone_width_l1666_166694


namespace cleaning_frequency_in_year_l1666_166617

/-- The number of times a person cleans themselves in 52 weeks, given they take
    a bath twice a week and a shower once a week. -/
def cleaningFrequency (bathsPerWeek showerPerWeek weeksInYear : ℕ) : ℕ :=
  (bathsPerWeek + showerPerWeek) * weeksInYear

/-- Theorem stating that a person who takes a bath twice a week and a shower once a week
    cleans themselves 156 times in 52 weeks. -/
theorem cleaning_frequency_in_year :
  cleaningFrequency 2 1 52 = 156 := by
  sorry

end cleaning_frequency_in_year_l1666_166617


namespace sum_of_digits_9ab_l1666_166624

def a : ℕ := 10^2023 - 1

def b : ℕ := 7 * (10^2023 - 1) / 9

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_9ab : sum_of_digits (9 * a * b) = 36410 := by
  sorry

end sum_of_digits_9ab_l1666_166624


namespace sum_of_segments_is_81_l1666_166610

/-- Represents the structure of triangles within the larger triangle -/
structure TriangleStructure where
  large_perimeter : ℝ
  small_side_length : ℝ
  small_triangle_count : ℕ

/-- The specific triangle structure from the problem -/
def problem_structure : TriangleStructure where
  large_perimeter := 24
  small_side_length := 1
  small_triangle_count := 27

/-- Calculates the sum of all segment lengths in the structure -/
def sum_of_segments (ts : TriangleStructure) : ℝ :=
  ts.small_triangle_count * (3 * ts.small_side_length)

/-- Theorem stating the sum of all segments in the given structure is 81 -/
theorem sum_of_segments_is_81 :
  sum_of_segments problem_structure = 81 := by sorry

end sum_of_segments_is_81_l1666_166610


namespace commonly_used_charts_characterization_l1666_166622

/-- A type representing different types of charts -/
inductive Chart
  | ContingencyTable
  | ThreeDimensionalBarChart
  | TwoDimensionalBarChart
  | OtherChart

/-- The set of charts commonly used for analyzing relationships between two categorical variables -/
def commonly_used_charts : Set Chart := sorry

/-- The theorem stating that the commonly used charts are exactly the contingency tables,
    three-dimensional bar charts, and two-dimensional bar charts -/
theorem commonly_used_charts_characterization :
  commonly_used_charts = {Chart.ContingencyTable, Chart.ThreeDimensionalBarChart, Chart.TwoDimensionalBarChart} := by sorry

end commonly_used_charts_characterization_l1666_166622


namespace prime_sum_theorem_l1666_166670

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem prime_sum_theorem (a b c : ℕ) 
  (ha : isPrime a) (hb : isPrime b) (hc : isPrime c)
  (h1 : b + c = 13) (h2 : c^2 - a^2 = 72) : a + b + c = 15 := by
  sorry

end prime_sum_theorem_l1666_166670


namespace silver_division_problem_l1666_166608

theorem silver_division_problem (x y : ℤ) : 
  y = 7 * x + 4 ∧ y = 9 * x - 8 → y = 46 := by
sorry

end silver_division_problem_l1666_166608


namespace milk_cost_verify_milk_cost_l1666_166609

/-- Proves that the cost of a gallon of milk is $4 given the conditions about coffee consumption and costs --/
theorem milk_cost (cups_per_day : ℕ) (oz_per_cup : ℚ) (bag_cost : ℚ) (oz_per_bag : ℚ) 
  (milk_usage : ℚ) (total_cost : ℚ) : ℚ :=
by
  -- Define the conditions
  have h1 : cups_per_day = 2 := by sorry
  have h2 : oz_per_cup = 3/2 := by sorry
  have h3 : bag_cost = 8 := by sorry
  have h4 : oz_per_bag = 21/2 := by sorry
  have h5 : milk_usage = 1/2 := by sorry
  have h6 : total_cost = 18 := by sorry

  -- Calculate the cost of a gallon of milk
  sorry

/-- The cost of a gallon of milk --/
def gallon_milk_cost : ℚ := 4

/-- Proves that the calculated cost matches the expected cost --/
theorem verify_milk_cost : 
  milk_cost 2 (3/2) 8 (21/2) (1/2) 18 = gallon_milk_cost := by sorry

end milk_cost_verify_milk_cost_l1666_166609


namespace prove_ball_size_ratio_l1666_166668

def ball_size_ratio (first_ball : ℝ) (second_ball : ℝ) (third_ball : ℝ) : Prop :=
  first_ball = second_ball / 2 ∧ 
  second_ball = 18 ∧ 
  third_ball = 27 ∧ 
  third_ball / first_ball = 3

theorem prove_ball_size_ratio : 
  ∃ (first_ball second_ball third_ball : ℝ), 
    ball_size_ratio first_ball second_ball third_ball :=
sorry

end prove_ball_size_ratio_l1666_166668


namespace quadratic_root_problem_l1666_166630

theorem quadratic_root_problem (k : ℝ) : 
  (2 : ℝ)^2 + 2 - k = 0 → (-3 : ℝ)^2 + (-3) - k = 0 := by
  sorry

end quadratic_root_problem_l1666_166630


namespace statement_B_statement_D_l1666_166607

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (contained : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- Statement B
theorem statement_B :
  perpendicular m n →
  perpendicular_plane m α →
  perpendicular_plane n β →
  perpendicular_planes α β :=
sorry

-- Statement D
theorem statement_D :
  parallel_planes α β →
  perpendicular_plane m α →
  parallel n β →
  perpendicular m n :=
sorry

end statement_B_statement_D_l1666_166607


namespace arrive_at_beths_house_time_l1666_166623

/-- The time it takes for Tom and Beth to meet and return to Beth's house -/
def meeting_and_return_time (tom_speed beth_speed : ℚ) : ℚ :=
  let meeting_time := 1 / (tom_speed + beth_speed)
  let return_time := (1 / 2) / beth_speed
  meeting_time + return_time

/-- Theorem stating that Tom and Beth will arrive at Beth's house 78 minutes after noon -/
theorem arrive_at_beths_house_time :
  let tom_speed : ℚ := 1 / 63
  let beth_speed : ℚ := 1 / 84
  meeting_and_return_time tom_speed beth_speed = 78 / 1 := by
  sorry

#eval meeting_and_return_time (1 / 63) (1 / 84)

end arrive_at_beths_house_time_l1666_166623


namespace polynomial_factorization_l1666_166650

theorem polynomial_factorization : 
  ∀ x : ℤ, x^15 + x^10 + 1 = (x^3 + x^2 + 1) * (x^12 - x^11 + x^9 - x^8 + x^6 - x^4 + x^2 - x + 1) := by
  sorry

end polynomial_factorization_l1666_166650


namespace fixed_point_of_exponential_function_l1666_166615

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 7 + a^(x - 1)
  f 1 = 8 := by
  sorry

end fixed_point_of_exponential_function_l1666_166615
