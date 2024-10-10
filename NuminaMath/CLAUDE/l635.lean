import Mathlib

namespace sum_of_fractions_l635_63544

theorem sum_of_fractions : (1 : ℚ) / 2 + (1 : ℚ) / 4 = (3 : ℚ) / 4 := by
  sorry

end sum_of_fractions_l635_63544


namespace sum_of_squared_roots_l635_63506

theorem sum_of_squared_roots (p q r : ℝ) : 
  (3 * p^3 + 2 * p^2 - 3 * p - 8 = 0) →
  (3 * q^3 + 2 * q^2 - 3 * q - 8 = 0) →
  (3 * r^3 + 2 * r^2 - 3 * r - 8 = 0) →
  p^2 + q^2 + r^2 = -14/9 := by
sorry

end sum_of_squared_roots_l635_63506


namespace average_adjacent_pairs_l635_63501

/-- Represents a row of people --/
structure Row where
  boys : ℕ
  girls : ℕ

/-- Calculates the expected number of boy-girl or girl-boy pairs in a row --/
def expectedPairs (r : Row) : ℚ :=
  let total := r.boys + r.girls
  let prob := (r.boys : ℚ) * r.girls / (total * (total - 1))
  2 * prob * (total - 1)

/-- The problem statement --/
theorem average_adjacent_pairs (row1 row2 : Row)
  (h1 : row1 = ⟨10, 12⟩)
  (h2 : row2 = ⟨15, 5⟩) :
  expectedPairs row1 + expectedPairs row2 = 2775 / 154 := by
  sorry

#eval expectedPairs ⟨10, 12⟩ + expectedPairs ⟨15, 5⟩

end average_adjacent_pairs_l635_63501


namespace limestone_cost_proof_l635_63507

/-- The cost of limestone per pound -/
def limestone_cost : ℝ := 3

/-- The total weight of the compound in pounds -/
def total_weight : ℝ := 100

/-- The total cost of the compound in dollars -/
def total_cost : ℝ := 425

/-- The weight of limestone used in the compound in pounds -/
def limestone_weight : ℝ := 37.5

/-- The weight of shale mix used in the compound in pounds -/
def shale_weight : ℝ := 62.5

/-- The cost of shale mix per pound in dollars -/
def shale_cost_per_pound : ℝ := 5

/-- The total cost of shale mix in the compound in dollars -/
def total_shale_cost : ℝ := 312.5

theorem limestone_cost_proof :
  limestone_cost * limestone_weight + total_shale_cost = total_cost ∧
  limestone_weight + shale_weight = total_weight ∧
  shale_cost_per_pound * shale_weight = total_shale_cost :=
by sorry

end limestone_cost_proof_l635_63507


namespace kid_tickets_sold_l635_63573

/-- Proves the number of kid tickets sold given ticket prices, total tickets, and profit -/
theorem kid_tickets_sold 
  (adult_price : ℕ) 
  (kid_price : ℕ) 
  (total_tickets : ℕ) 
  (total_profit : ℕ) 
  (h1 : adult_price = 6)
  (h2 : kid_price = 2)
  (h3 : total_tickets = 175)
  (h4 : total_profit = 750) :
  ∃ (adult_tickets kid_tickets : ℕ), 
    adult_tickets + kid_tickets = total_tickets ∧
    adult_price * adult_tickets + kid_price * kid_tickets = total_profit ∧
    kid_tickets = 75 :=
by sorry

end kid_tickets_sold_l635_63573


namespace expression_simplification_l635_63581

theorem expression_simplification (a : ℝ) (ha : a = 2018) :
  (a^2 - 3*a) / (a^2 + a) / ((a - 3) / (a^2 - 1)) * ((a + 1) / (a - 1)) = a := by
  sorry

end expression_simplification_l635_63581


namespace proportional_function_quadrants_l635_63512

theorem proportional_function_quadrants (k : ℝ) :
  let f : ℝ → ℝ := λ x => (-k^2 - 2) * x
  (∀ x y, f x = y → (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by sorry

end proportional_function_quadrants_l635_63512


namespace factorial_equation_l635_63595

theorem factorial_equation (x : ℕ) : 6 * 8 * 3 * x = Nat.factorial 10 → x = 75600 := by
  sorry

end factorial_equation_l635_63595


namespace power_quotient_plus_five_l635_63502

theorem power_quotient_plus_five : 23^12 / 23^5 + 5 = 148035894 := by
  sorry

end power_quotient_plus_five_l635_63502


namespace stating_judge_assignment_count_l635_63555

/-- Represents the number of judges from each grade -/
def judges_per_grade : ℕ := 2

/-- Represents the number of grades -/
def num_grades : ℕ := 3

/-- Represents the number of courts -/
def num_courts : ℕ := 3

/-- Represents the number of judges per court -/
def judges_per_court : ℕ := 2

/-- 
Theorem stating that the number of ways to assign judges to courts 
under the given conditions is 48
-/
theorem judge_assignment_count : 
  (judges_per_grade ^ num_courts) * (Nat.factorial num_courts) = 48 := by
  sorry


end stating_judge_assignment_count_l635_63555


namespace sum_formula_and_difference_l635_63543

def f (n : ℕ) : ℕ := n + (n + 1) + (n + 2) + (3 * n - 2)

theorem sum_formula_and_difference (n k : ℕ) (h : n > 0) (h' : k > 0) : 
  f n = (2 * n - 1)^2 ∧ f (k + 1) - f k = 8 * k := by sorry

end sum_formula_and_difference_l635_63543


namespace proportional_relationship_l635_63589

-- Define the proportionality constant
def k : ℝ := 2

-- Define the functional relationship
def f (x : ℝ) : ℝ := k * x + 3

-- State the theorem
theorem proportional_relationship (x y : ℝ) :
  (∀ x, y - 3 = k * x) →  -- (y-3) is directly proportional to x
  (f 2 = 7) →             -- when x=2, y=7
  (∀ x, f x = 2 * x + 3) ∧ -- functional relationship
  (f 4 = 11) ∧            -- when x=4, y=11
  (f⁻¹ 4 = 1/2)           -- when y=4, x=1/2
  := by sorry

end proportional_relationship_l635_63589


namespace expression_value_l635_63513

theorem expression_value : (-1/2)^2023 * 2^2024 = -2 := by sorry

end expression_value_l635_63513


namespace absolute_value_and_exponentiation_calculation_l635_63565

theorem absolute_value_and_exponentiation_calculation : 
  |1 - 3| * ((-12) - 2^3) = -40 := by
sorry

end absolute_value_and_exponentiation_calculation_l635_63565


namespace skew_lines_and_tetrahedron_l635_63557

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the relation for a point lying on a line
variable (lies_on : Point → Line → Prop)

-- Define the property of lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of lines being perpendicular
variable (perpendicular : Line → Line → Prop)

-- Define the property of points forming a regular tetrahedron
variable (form_regular_tetrahedron : Point → Point → Point → Point → Prop)

-- State the theorem
theorem skew_lines_and_tetrahedron 
  (A B C D : Point) (a b : Line) :
  lies_on A a → lies_on B a → lies_on C b → lies_on D b →
  skew a b →
  ¬perpendicular a b →
  (∃ (AC BD : Line), lies_on A AC ∧ lies_on C AC ∧ lies_on B BD ∧ lies_on D BD ∧ skew AC BD) ∧
  ¬form_regular_tetrahedron A B C D :=
sorry

end skew_lines_and_tetrahedron_l635_63557


namespace lab_coat_uniform_ratio_l635_63532

theorem lab_coat_uniform_ratio :
  ∀ (num_uniforms num_lab_coats num_total : ℕ),
    num_uniforms = 12 →
    num_lab_coats = 6 * num_uniforms →
    num_total = num_lab_coats + num_uniforms →
    num_total % 14 = 0 →
    num_lab_coats / num_uniforms = 6 :=
by
  sorry

end lab_coat_uniform_ratio_l635_63532


namespace sequence_formula_l635_63567

theorem sequence_formula (a : ℕ → ℝ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 2 → a n - a (n-1) = (1/3)^(n-1)) →
  ∀ n : ℕ, n ≥ 1 → a n = (3/2) * (1 - (1/3)^n) :=
by sorry

end sequence_formula_l635_63567


namespace nine_team_league_games_l635_63541

/-- The number of games played in a league where each team plays every other team once -/
def numGames (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league with 9 teams, where each team plays every other team exactly once,
    the total number of games played is 36. -/
theorem nine_team_league_games :
  numGames 9 = 36 := by
  sorry


end nine_team_league_games_l635_63541


namespace total_holidays_in_year_l635_63598

def holidays : List Nat := [4, 3, 5, 3, 4, 2, 5, 3, 4, 3, 5, 4]

theorem total_holidays_in_year : holidays.sum = 45 := by
  sorry

end total_holidays_in_year_l635_63598


namespace smallest_common_shelving_count_l635_63552

theorem smallest_common_shelving_count : Nat.lcm 6 17 = 102 := by
  sorry

end smallest_common_shelving_count_l635_63552


namespace solve_chicken_problem_l635_63533

def chicken_problem (chicken_cost total_spent potato_cost : ℕ) : Prop :=
  chicken_cost > 0 ∧
  total_spent > potato_cost ∧
  (total_spent - potato_cost) % chicken_cost = 0 ∧
  (total_spent - potato_cost) / chicken_cost = 3

theorem solve_chicken_problem :
  chicken_problem 3 15 6 := by
  sorry

end solve_chicken_problem_l635_63533


namespace product_inequality_l635_63596

theorem product_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (a + b) * (a + c) * (b + c) ≥ 8 * a * b * c :=
by sorry

end product_inequality_l635_63596


namespace prob_red_then_king_diamonds_standard_deck_l635_63580

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (red_cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- Probability of drawing a red card first and then the King of Diamonds second -/
def prob_red_then_king_diamonds (d : Deck) : Rat :=
  if d.total_cards = 52 ∧ d.red_cards = 26 ∧ d.ranks = 13 ∧ d.suits = 4 then
    1 / 102
  else
    0

/-- Theorem stating the probability of drawing a red card first and then the King of Diamonds second -/
theorem prob_red_then_king_diamonds_standard_deck :
  ∃ (d : Deck), prob_red_then_king_diamonds d = 1 / 102 :=
sorry

end prob_red_then_king_diamonds_standard_deck_l635_63580


namespace expression_defined_iff_l635_63576

-- Define the set of real numbers for which the expression is defined
def valid_x : Set ℝ := {x | x ∈ Set.Ioo (-Real.sqrt 5) 1 ∪ Set.Ioo 3 (Real.sqrt 5)}

-- Define the conditions for the expression to be defined
def conditions (x : ℝ) : Prop :=
  x^2 - 4*x + 3 > 0 ∧ 5 - x^2 > 0

-- Theorem statement
theorem expression_defined_iff (x : ℝ) :
  conditions x ↔ x ∈ valid_x := by sorry

end expression_defined_iff_l635_63576


namespace video_call_cost_proof_l635_63558

/-- Calculates the cost of a video call given the charge rate and duration. -/
def video_call_cost (charge_rate : ℕ) (charge_interval : ℕ) (duration : ℕ) : ℕ :=
  (duration / charge_interval) * charge_rate

/-- Proves that a 2 minute and 40 second video call costs 480 won at a rate of 30 won per 10 seconds. -/
theorem video_call_cost_proof :
  let charge_rate : ℕ := 30
  let charge_interval : ℕ := 10
  let duration : ℕ := 2 * 60 + 40
  video_call_cost charge_rate charge_interval duration = 480 := by
  sorry

#eval video_call_cost 30 10 (2 * 60 + 40)

end video_call_cost_proof_l635_63558


namespace problem_solution_l635_63575

theorem problem_solution (x y z w : ℕ+) 
  (h1 : x^3 = y^4)
  (h2 : z^5 = w^2)
  (h3 : z - x = 31) :
  (w : ℤ) - y = -759439 := by
  sorry

end problem_solution_l635_63575


namespace twice_x_minus_y_negative_l635_63551

theorem twice_x_minus_y_negative (x y : ℝ) : 
  (2 * x - y < 0) ↔ (∃ z : ℝ, z < 0 ∧ 2 * x - y = z) :=
sorry

end twice_x_minus_y_negative_l635_63551


namespace total_weekly_egg_supply_l635_63578

/-- Represents the daily egg supply to a store -/
structure DailySupply where
  oddDays : ℕ
  evenDays : ℕ

/-- Calculates the total eggs supplied to a store in a week -/
def weeklySupply (supply : DailySupply) : ℕ :=
  4 * supply.oddDays + 3 * supply.evenDays

/-- Converts dozens to individual eggs -/
def dozensToEggs (dozens : ℕ) : ℕ :=
  dozens * 12

theorem total_weekly_egg_supply :
  let store1 := DailySupply.mk (dozensToEggs 5) (dozensToEggs 5)
  let store2 := DailySupply.mk 30 30
  let store3 := DailySupply.mk (dozensToEggs 25) (dozensToEggs 15)
  weeklySupply store1 + weeklySupply store2 + weeklySupply store3 = 2370 := by
  sorry

end total_weekly_egg_supply_l635_63578


namespace cubic_equation_one_root_implies_a_range_l635_63525

theorem cubic_equation_one_root_implies_a_range (a : ℝ) : 
  (∃! x : ℝ, x^3 + (1-a)*x^2 - 2*a*x + a^2 = 0) → a < -1/4 := by
  sorry

end cubic_equation_one_root_implies_a_range_l635_63525


namespace not_odd_function_iff_exists_neq_l635_63515

theorem not_odd_function_iff_exists_neq (f : ℝ → ℝ) :
  (¬ ∀ x, f (-x) = -f x) ↔ ∃ x₀, f (-x₀) ≠ -f x₀ :=
sorry

end not_odd_function_iff_exists_neq_l635_63515


namespace expression_equality_l635_63560

theorem expression_equality : 
  Real.sqrt 12 + |Real.sqrt 3 - 2| + 3 - (Real.pi - 3.14)^0 = Real.sqrt 3 + 4 := by
  sorry

end expression_equality_l635_63560


namespace ant_return_probability_l635_63559

/-- A modified lattice with an extra horizontal connection --/
structure ModifiedLattice :=
  (extra_connection : Bool)

/-- An ant on the modified lattice --/
structure Ant :=
  (position : ℤ × ℤ)
  (moves : ℕ)

/-- The probability of the ant returning to its starting point --/
def return_probability (l : ModifiedLattice) (a : Ant) : ℚ :=
  sorry

/-- Theorem stating the probability of returning to the starting point after 6 moves --/
theorem ant_return_probability (l : ModifiedLattice) (a : Ant) : 
  l.extra_connection = true →
  a.moves = 6 →
  return_probability l a = 1 / 64 :=
sorry

end ant_return_probability_l635_63559


namespace black_duck_count_l635_63574

/-- Represents the number of fish per duck of each color --/
structure FishPerDuck where
  white : ℕ
  black : ℕ
  multicolor : ℕ

/-- Represents the number of ducks of each color --/
structure DuckCounts where
  white : ℕ
  black : ℕ
  multicolor : ℕ

/-- The theorem stating the number of black ducks --/
theorem black_duck_count 
  (fish_per_duck : FishPerDuck)
  (duck_counts : DuckCounts)
  (total_fish : ℕ)
  (h1 : fish_per_duck.white = 5)
  (h2 : fish_per_duck.black = 10)
  (h3 : fish_per_duck.multicolor = 12)
  (h4 : duck_counts.white = 3)
  (h5 : duck_counts.multicolor = 6)
  (h6 : total_fish = 157)
  (h7 : total_fish = 
    fish_per_duck.white * duck_counts.white + 
    fish_per_duck.black * duck_counts.black + 
    fish_per_duck.multicolor * duck_counts.multicolor) :
  duck_counts.black = 7 := by
  sorry

end black_duck_count_l635_63574


namespace annual_decrease_rate_l635_63588

/-- Proves that the annual decrease rate is 20% for a town with given population changes. -/
theorem annual_decrease_rate (initial_population : ℝ) (population_after_two_years : ℝ) 
  (h1 : initial_population = 15000)
  (h2 : population_after_two_years = 9600) :
  ∃ (r : ℝ), r = 20 ∧ population_after_two_years = initial_population * (1 - r / 100)^2 := by
  sorry

end annual_decrease_rate_l635_63588


namespace outfit_combinations_l635_63585

def number_of_shirts : ℕ := 5
def number_of_pants : ℕ := 6
def number_of_belts : ℕ := 2

theorem outfit_combinations : 
  number_of_shirts * number_of_pants * number_of_belts = 60 := by
  sorry

end outfit_combinations_l635_63585


namespace no_perfect_square_19xx99_l635_63504

theorem no_perfect_square_19xx99 : ¬ ∃ (n : ℕ), 
  (n * n ≥ 1900000) ∧ 
  (n * n < 2000000) ∧ 
  (n * n % 100 = 99) := by
sorry

end no_perfect_square_19xx99_l635_63504


namespace students_with_all_pets_l635_63539

theorem students_with_all_pets (total_students : ℕ) 
  (dog_fraction : ℚ) (cat_fraction : ℚ)
  (other_pet_count : ℕ) (no_pet_count : ℕ)
  (only_dog_count : ℕ) (only_other_count : ℕ)
  (cat_and_other_count : ℕ) :
  total_students = 40 →
  dog_fraction = 5 / 8 →
  cat_fraction = 1 / 4 →
  other_pet_count = 8 →
  no_pet_count = 6 →
  only_dog_count = 12 →
  only_other_count = 3 →
  cat_and_other_count = 10 →
  (∃ (all_pets_count : ℕ),
    all_pets_count = 0 ∧
    total_students * dog_fraction = only_dog_count + all_pets_count + cat_and_other_count ∧
    total_students * cat_fraction = cat_and_other_count + all_pets_count ∧
    other_pet_count = only_other_count + all_pets_count + cat_and_other_count ∧
    total_students - no_pet_count = only_dog_count + only_other_count + all_pets_count + cat_and_other_count) :=
by
  sorry

end students_with_all_pets_l635_63539


namespace smallest_median_l635_63524

def number_set (x : ℤ) : Finset ℤ := {x, 2*x, 4, 3, 6}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (· ≤ m)).card ≥ s.card ∧
  2 * (s.filter (· ≥ m)).card ≥ s.card

theorem smallest_median :
  ∀ x : ℤ, ∃ m : ℤ, is_median m (number_set x) ∧ 
  (∀ m' : ℤ, is_median m' (number_set x) → m ≤ m') ∧
  m = 3 :=
sorry

end smallest_median_l635_63524


namespace fraction_simplification_l635_63542

theorem fraction_simplification :
  (252 : ℚ) / 18 * 7 / 189 * 9 / 4 = 7 / 6 := by
  sorry

end fraction_simplification_l635_63542


namespace inequality_proof_l635_63553

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hab : a + b ≥ 1) 
  (hbc : b + c ≥ 1) 
  (hca : c + a ≥ 1) : 
  1 ≤ (1 - a)^2 + (1 - b)^2 + (1 - c)^2 + (2 * Real.sqrt 2 * a * b * c) / Real.sqrt (a^2 + b^2 + c^2) :=
sorry

end inequality_proof_l635_63553


namespace count_without_one_between_1_and_2000_l635_63528

/-- Count of numbers without digit 1 in a given range -/
def count_without_digit_one (lower : Nat) (upper : Nat) : Nat :=
  sorry

/-- The main theorem -/
theorem count_without_one_between_1_and_2000 :
  count_without_digit_one 1 2000 = 1457 := by sorry

end count_without_one_between_1_and_2000_l635_63528


namespace largest_three_digit_divisible_by_six_l635_63577

theorem largest_three_digit_divisible_by_six :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 6 = 0 → n ≤ 996 ∧ 996 % 6 = 0 :=
by sorry

end largest_three_digit_divisible_by_six_l635_63577


namespace extended_triangles_similarity_l635_63579

/-- Represents a point in the complex plane -/
structure ComplexPoint where
  x : ℝ
  y : ℝ

/-- Represents a triangle in the complex plane -/
structure Triangle where
  A : ComplexPoint
  B : ComplexPoint
  C : ComplexPoint

/-- Extends a side of a triangle by a factor k -/
def extendSide (A B : ComplexPoint) (k : ℝ) : ComplexPoint :=
  { x := A.x + k * (B.x - A.x),
    y := A.y + k * (B.y - A.y) }

/-- Extends an altitude of a triangle by a factor k -/
def extendAltitude (A B C : ComplexPoint) (k : ℝ) : ComplexPoint :=
  { x := A.x + k * (C.y - B.y),
    y := A.y - k * (C.x - B.x) }

/-- Checks if two triangles are similar -/
def areSimilar (T1 T2 : Triangle) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
    (T1.B.x - T1.A.x)^2 + (T1.B.y - T1.A.y)^2 = r * ((T2.B.x - T2.A.x)^2 + (T2.B.y - T2.A.y)^2) ∧
    (T1.C.x - T1.B.x)^2 + (T1.C.y - T1.B.y)^2 = r * ((T2.C.x - T2.B.x)^2 + (T2.C.y - T2.B.y)^2) ∧
    (T1.A.x - T1.C.x)^2 + (T1.A.y - T1.C.y)^2 = r * ((T2.A.x - T2.C.x)^2 + (T2.A.y - T2.C.y)^2)

theorem extended_triangles_similarity (ABC : Triangle) :
  ∃ (k : ℝ), k > 1 ∧
    let P := extendSide ABC.A ABC.B k
    let Q := extendSide ABC.B ABC.C k
    let R := extendSide ABC.C ABC.A k
    let A' := extendAltitude ABC.A ABC.B ABC.C k
    let B' := extendAltitude ABC.B ABC.C ABC.A k
    let C' := extendAltitude ABC.C ABC.A ABC.B k
    areSimilar
      { A := P, B := Q, C := R }
      { A := A', B := B', C := C' } :=
by sorry

end extended_triangles_similarity_l635_63579


namespace added_number_after_doubling_l635_63592

theorem added_number_after_doubling (initial_number : ℕ) (x : ℕ) : 
  initial_number = 8 → 
  3 * (2 * initial_number + x) = 75 → 
  x = 9 := by
sorry

end added_number_after_doubling_l635_63592


namespace area_triangle_PAB_l635_63530

/-- Given points A(-1, 2), B(3, 4), and P on the x-axis such that |PA| = |PB|,
    the area of triangle PAB is 15/2. -/
theorem area_triangle_PAB :
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (3, 4)
  ∀ P : ℝ × ℝ,
    P.2 = 0 →  -- P is on the x-axis
    (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 →  -- |PA| = |PB|
    abs ((B.1 - A.1) * (P.2 - A.2) - (B.2 - A.2) * (P.1 - A.1)) / 2 = 15/2 :=
by sorry

end area_triangle_PAB_l635_63530


namespace tan_4290_degrees_l635_63508

theorem tan_4290_degrees : Real.tan (4290 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_4290_degrees_l635_63508


namespace specific_pyramid_volume_l635_63538

/-- A pyramid with a parallelogram base and specific dimensions --/
structure Pyramid where
  base_side1 : ℝ
  base_side2 : ℝ
  base_diagonal : ℝ
  lateral_edge : ℝ

/-- The volume of the pyramid --/
def pyramid_volume (p : Pyramid) : ℝ := sorry

/-- Theorem stating the volume of the specific pyramid is 200 --/
theorem specific_pyramid_volume :
  let p : Pyramid := {
    base_side1 := 9,
    base_side2 := 10,
    base_diagonal := 11,
    lateral_edge := Real.sqrt 10
  }
  pyramid_volume p = 200 := by sorry

end specific_pyramid_volume_l635_63538


namespace log_comparison_l635_63564

theorem log_comparison : Real.log 80 / Real.log 20 < Real.log 640 / Real.log 80 := by
  sorry

end log_comparison_l635_63564


namespace cereal_box_cups_l635_63550

/-- Calculates the total number of cups in a cereal box -/
def total_cups (servings : ℕ) (cups_per_serving : ℕ) : ℕ :=
  servings * cups_per_serving

/-- Theorem: A cereal box with 9 servings and 2 cups per serving contains 18 cups of cereal -/
theorem cereal_box_cups : total_cups 9 2 = 18 := by
  sorry

end cereal_box_cups_l635_63550


namespace base_89_multiple_of_13_l635_63536

theorem base_89_multiple_of_13 (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 20) 
  (h3 : (142536472 : ℤ) ≡ b [ZMOD 13]) : b = 8 := by
  sorry

end base_89_multiple_of_13_l635_63536


namespace inequality_and_equality_condition_l635_63569

theorem inequality_and_equality_condition (n : ℕ+) (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) : 
  (1 + a / b)^(n : ℕ) + (1 + b / a)^(n : ℕ) ≥ 2^((n : ℕ) + 1) ∧ 
  ((1 + a / b)^(n : ℕ) + (1 + b / a)^(n : ℕ) = 2^((n : ℕ) + 1) ↔ a = b) :=
by sorry

end inequality_and_equality_condition_l635_63569


namespace tangent_line_at_x_1_l635_63572

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (2 * x - y + 1 = 0) :=
by sorry

end tangent_line_at_x_1_l635_63572


namespace shooter_probabilities_l635_63556

def hit_probability : ℚ := 4/5

def exactly_eight_hits (n k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1-p)^(n-k)

def at_least_eight_hits (n : ℕ) (p : ℚ) : ℚ :=
  exactly_eight_hits n 8 p + exactly_eight_hits n 9 p + p^n

theorem shooter_probabilities :
  (exactly_eight_hits 10 8 hit_probability = 
    Nat.choose 10 8 * (4/5)^8 * (1/5)^2) ∧
  (at_least_eight_hits 10 hit_probability = 
    Nat.choose 10 8 * (4/5)^8 * (1/5)^2 + 
    Nat.choose 10 9 * (4/5)^9 * (1/5) + 
    (4/5)^10) := by
  sorry

end shooter_probabilities_l635_63556


namespace function_properties_l635_63534

def f (a x : ℝ) : ℝ := |2*x + a| + |x - 1|

theorem function_properties :
  (∀ x : ℝ, f 3 x < 6 ↔ -8/3 < x ∧ x < 4/3) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x + f a (-x) ≥ 5) ↔ a ≤ -3/2 ∨ a ≥ 3/2) :=
by sorry

end function_properties_l635_63534


namespace min_magnitude_vector_sum_l635_63546

/-- The minimum magnitude of the vector sum of two specific unit vectors -/
theorem min_magnitude_vector_sum :
  let a : ℝ × ℝ := (Real.cos (25 * π / 180), Real.sin (25 * π / 180))
  let b : ℝ × ℝ := (Real.sin (20 * π / 180), Real.cos (20 * π / 180))
  ∃ (min_val : ℝ), min_val = Real.sqrt 2 / 2 ∧
    ∀ (t : ℝ), Real.sqrt ((a.1 + t * b.1)^2 + (a.2 + t * b.2)^2) ≥ min_val :=
by sorry

end min_magnitude_vector_sum_l635_63546


namespace x_range_for_inequality_l635_63535

-- Define the function f(m, x)
def f (m x : ℝ) : ℝ := m * (x^2 - 1) - 1 - 8*x

-- State the theorem
theorem x_range_for_inequality :
  (∀ x : ℝ, (∀ m : ℝ, -1 ≤ m ∧ m ≤ 4 → f m x < 0) ↔ 0 < x ∧ x < 5/2) :=
sorry

end x_range_for_inequality_l635_63535


namespace track_length_l635_63566

/-- The length of a circular track given specific meeting conditions of two runners -/
theorem track_length : ∀ (x : ℝ), 
  (∃ (v_brenda v_sally : ℝ), v_brenda > 0 ∧ v_sally > 0 ∧
    -- First meeting condition
    80 / v_brenda = (x/2 - 80) / v_sally ∧
    -- Second meeting condition
    (x/2 - 100) / v_brenda = (x/2 + 100) / v_sally) →
  x = 520 :=
by sorry

end track_length_l635_63566


namespace four_children_probability_l635_63593

theorem four_children_probability (p_boy p_girl : ℚ) : 
  p_boy = 2/3 → 
  p_girl = 1/3 → 
  (1 - (p_boy^4 + p_girl^4)) = 64/81 := by
sorry

end four_children_probability_l635_63593


namespace smallest_multiple_l635_63537

theorem smallest_multiple (x : ℕ) : x = 32 ↔ 
  (x > 0 ∧ 
   1152 ∣ (900 * x) ∧ 
   ∀ y : ℕ, (y > 0 ∧ y < x) → ¬(1152 ∣ (900 * y))) := by
  sorry

end smallest_multiple_l635_63537


namespace inscribed_circle_square_area_l635_63554

/-- Given a circle inscribed in a square, if the circle's area is 314 square inches,
    then the square's area is 400 square inches. -/
theorem inscribed_circle_square_area :
  ∀ (circle_radius square_side : ℝ),
  circle_radius > 0 →
  square_side > 0 →
  circle_radius * 2 = square_side →
  π * circle_radius^2 = 314 →
  square_side^2 = 400 :=
by sorry

end inscribed_circle_square_area_l635_63554


namespace characterize_superinvariant_sets_l635_63511

/-- A set S is superinvariant if for any stretching A, there exists a translation B
    such that the images of S under A and B agree -/
def IsSuperinvariant (S : Set ℝ) : Prop :=
  ∀ (x₀ a : ℝ) (ha : a > 0),
    ∃ (b : ℝ),
      (∀ x ∈ S, ∃ y ∈ S, x₀ + a * (x - x₀) = y + b) ∧
      (∀ t ∈ S, ∃ u ∈ S, t + b = x₀ + a * (u - x₀))

/-- The set of all superinvariant subsets of ℝ -/
def SuperinvariantSets : Set (Set ℝ) :=
  {S | IsSuperinvariant S}

theorem characterize_superinvariant_sets :
  SuperinvariantSets =
    {∅} ∪ {Set.univ} ∪ {{p} | p : ℝ} ∪ {Set.univ \ {p} | p : ℝ} ∪
    {Set.Ioi p | p : ℝ} ∪ {Set.Ici p | p : ℝ} ∪
    {Set.Iio p | p : ℝ} ∪ {Set.Iic p | p : ℝ} :=
  sorry

#check characterize_superinvariant_sets

end characterize_superinvariant_sets_l635_63511


namespace sqrt_product_equals_sqrt_of_product_l635_63519

theorem sqrt_product_equals_sqrt_of_product : 
  Real.sqrt 2 * Real.sqrt 5 = Real.sqrt 10 := by
  sorry

end sqrt_product_equals_sqrt_of_product_l635_63519


namespace philips_banana_groups_l635_63597

/-- Given Philip's fruit collection, prove the number of banana groups -/
theorem philips_banana_groups
  (total_oranges : ℕ) (total_bananas : ℕ)
  (orange_groups : ℕ) (oranges_per_group : ℕ)
  (h1 : total_oranges = 384)
  (h2 : total_bananas = 192)
  (h3 : orange_groups = 16)
  (h4 : oranges_per_group = 24)
  (h5 : total_oranges = orange_groups * oranges_per_group)
  : total_bananas / oranges_per_group = 8 := by
  sorry

end philips_banana_groups_l635_63597


namespace exponential_function_first_quadrant_l635_63594

theorem exponential_function_first_quadrant (m : ℝ) :
  (∀ x : ℝ, x > 0 → (1/5)^(x+1) + m ≤ 0) ↔ m ≤ -1/5 := by sorry

end exponential_function_first_quadrant_l635_63594


namespace max_subway_employees_l635_63562

theorem max_subway_employees (total_employees : ℕ) 
  (h_total : total_employees = 48) 
  (part_time full_time : ℕ) 
  (h_sum : part_time + full_time = total_employees)
  (subway_part_time subway_full_time : ℕ)
  (h_part_time : subway_part_time * 3 ≤ part_time)
  (h_full_time : subway_full_time * 4 ≤ full_time) :
  subway_part_time + subway_full_time ≤ 15 :=
sorry

end max_subway_employees_l635_63562


namespace extremum_implies_a_equals_five_l635_63587

/-- Given a function f(x) = 4x^3 - ax^2 - 2x + b with an extremum at x = 1, prove that a = 5 --/
theorem extremum_implies_a_equals_five (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => 4*x^3 - a*x^2 - 2*x + b
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  a = 5 := by
sorry

end extremum_implies_a_equals_five_l635_63587


namespace simple_interest_principal_l635_63529

/-- Simple interest calculation -/
theorem simple_interest_principal
  (interest : ℝ)
  (time : ℝ)
  (rate : ℝ)
  (h1 : interest = 2500)
  (h2 : time = 5)
  (h3 : rate = 10)
  : interest = (5000 * rate * time) / 100 :=
by sorry

end simple_interest_principal_l635_63529


namespace chess_games_count_l635_63582

/-- The number of combinations of k items from a set of n items -/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of players in the chess group -/
def num_players : ℕ := 50

/-- The number of players in each game -/
def players_per_game : ℕ := 2

theorem chess_games_count : combinations num_players players_per_game = 1225 := by
  sorry

end chess_games_count_l635_63582


namespace return_trip_duration_l635_63545

/-- Represents the flight scenario between two cities --/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- speed of the plane in still air
  w : ℝ  -- speed of the wind
  outbound_time : ℝ  -- time for outbound trip (against wind)
  still_air_time : ℝ  -- time for return trip in still air

/-- The theorem stating the return trip duration --/
theorem return_trip_duration (fs : FlightScenario) : 
  fs.outbound_time = 120 →  -- Condition 1
  fs.d = fs.outbound_time * (fs.p - fs.w) →  -- Derived from Condition 1
  fs.still_air_time = fs.d / fs.p →  -- Definition of still air time
  fs.d / (fs.p + fs.w) = fs.still_air_time - 15 →  -- Condition 3
  (fs.d / (fs.p + fs.w) = 15 ∨ fs.d / (fs.p + fs.w) = 85) :=
by sorry

#check return_trip_duration

end return_trip_duration_l635_63545


namespace sum_of_squares_of_solutions_l635_63503

theorem sum_of_squares_of_solutions : ∃ (s₁ s₂ : ℝ), 
  (s₁^2 - 17*s₁ + 22 = 0) ∧ 
  (s₂^2 - 17*s₂ + 22 = 0) ∧ 
  (s₁^2 + s₂^2 = 245) := by
sorry

end sum_of_squares_of_solutions_l635_63503


namespace factorial_equality_l635_63583

theorem factorial_equality : 6 * 8 * 3 * 280 = Nat.factorial 8 := by
  sorry

end factorial_equality_l635_63583


namespace chairs_to_remove_l635_63526

theorem chairs_to_remove (initial_chairs : ℕ) (chairs_per_row : ℕ) (expected_participants : ℕ)
  (h1 : initial_chairs = 196)
  (h2 : chairs_per_row = 14)
  (h3 : expected_participants = 120)
  (h4 : chairs_per_row > 0) :
  let remaining_chairs := ((expected_participants + chairs_per_row - 1) / chairs_per_row) * chairs_per_row
  initial_chairs - remaining_chairs = 70 := by
sorry

end chairs_to_remove_l635_63526


namespace four_digit_numbers_with_sum_counts_l635_63505

/-- A function that returns the number of four-digit natural numbers with a given digit sum -/
def countFourDigitNumbersWithSum (sum : Nat) : Nat :=
  sorry

/-- The theorem stating the correct counts for digit sums 5, 6, and 7 -/
theorem four_digit_numbers_with_sum_counts :
  (countFourDigitNumbersWithSum 5 = 35) ∧
  (countFourDigitNumbersWithSum 6 = 56) ∧
  (countFourDigitNumbersWithSum 7 = 84) := by
  sorry

end four_digit_numbers_with_sum_counts_l635_63505


namespace coefficient_of_minus_five_ab_l635_63547

/-- The coefficient of a monomial is the numerical factor multiplying the variables. -/
def coefficient (m : ℤ) (x : String) : ℤ :=
  m

/-- A monomial is represented as an integer multiplied by a string of variables. -/
def Monomial := ℤ × String

theorem coefficient_of_minus_five_ab :
  let m : Monomial := (-5, "ab")
  coefficient m.1 m.2 = -5 := by
  sorry

end coefficient_of_minus_five_ab_l635_63547


namespace opposite_signs_for_positive_solution_l635_63522

theorem opposite_signs_for_positive_solution (a b : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ x : ℝ, x > 0 ∧ a * x + b = 0) : a * b < 0 := by
  sorry

end opposite_signs_for_positive_solution_l635_63522


namespace geometric_sequence_first_term_l635_63549

/-- Given a geometric sequence where the fifth term is 45 and the sixth term is 60,
    prove that the first term is 1215/256. -/
theorem geometric_sequence_first_term
  (a : ℚ)  -- First term of the sequence
  (r : ℚ)  -- Common ratio of the sequence
  (h1 : a * r^4 = 45)  -- Fifth term is 45
  (h2 : a * r^5 = 60)  -- Sixth term is 60
  : a = 1215 / 256 := by
  sorry

end geometric_sequence_first_term_l635_63549


namespace min_box_height_l635_63518

/-- Represents the side length of the square base of the box -/
def base_side : ℝ → ℝ := λ x => x

/-- Represents the height of the box -/
def box_height : ℝ → ℝ := λ x => x + 5

/-- Calculates the surface area of the box -/
def surface_area : ℝ → ℝ := λ x => 2 * x^2 + 4 * x * (x + 5)

/-- States that the surface area is at least 150 square units -/
def surface_area_constraint : ℝ → Prop := λ x => surface_area x ≥ 150

theorem min_box_height :
  ∃ x : ℝ, x > 0 ∧ surface_area_constraint x ∧
    box_height x = 10 ∧
    ∀ y : ℝ, y > 0 ∧ surface_area_constraint y → surface_area x ≤ surface_area y :=
by sorry

end min_box_height_l635_63518


namespace jelly_bean_problem_l635_63531

/-- The number of jelly beans remaining in the container after distribution --/
def remaining_jelly_beans (initial : ℕ) (people : ℕ) (first_group : ℕ) (last_group : ℕ) (last_group_takes : ℕ) : ℕ :=
  initial - (first_group * (2 * last_group_takes) + last_group * last_group_takes)

/-- Theorem stating the number of remaining jelly beans --/
theorem jelly_bean_problem :
  remaining_jelly_beans 8000 10 6 4 400 = 1600 := by
  sorry

end jelly_bean_problem_l635_63531


namespace divisibility_cycle_l635_63500

theorem divisibility_cycle (a b c : ℕ+) : 
  (∃ k₁ : ℕ, (2^(a:ℕ) - 1) = k₁ * (b:ℕ)) ∧
  (∃ k₂ : ℕ, (2^(b:ℕ) - 1) = k₂ * (c:ℕ)) ∧
  (∃ k₃ : ℕ, (2^(c:ℕ) - 1) = k₃ * (a:ℕ)) →
  a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end divisibility_cycle_l635_63500


namespace total_present_age_l635_63540

-- Define the present ages of p and q
def p : ℕ := sorry
def q : ℕ := sorry

-- Define the conditions
axiom age_relation : p - 12 = (q - 12) / 2
axiom present_ratio : p * 4 = q * 3

-- Theorem to prove
theorem total_present_age : p + q = 42 := by sorry

end total_present_age_l635_63540


namespace bridge_length_bridge_length_problem_l635_63570

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let distance_covered := train_speed_ms * crossing_time
  distance_covered - train_length

/-- The length of the bridge is 215 meters -/
theorem bridge_length_problem : bridge_length 160 45 30 = 215 := by
  sorry

end bridge_length_bridge_length_problem_l635_63570


namespace power_of_product_l635_63586

theorem power_of_product (a b : ℝ) : (a * b^2)^3 = a^3 * b^6 := by sorry

end power_of_product_l635_63586


namespace mothers_age_l635_63584

/-- Proves that the mother's age this year is 39 years old -/
theorem mothers_age (sons_age : ℕ) (mothers_age : ℕ) : mothers_age = 39 :=
  by
  -- Define the son's current age
  have h1 : sons_age = 12 := by sorry
  
  -- Define the relationship between mother's and son's ages three years ago
  have h2 : mothers_age - 3 = 4 * (sons_age - 3) := by sorry
  
  -- Prove that the mother's age is 39
  sorry


end mothers_age_l635_63584


namespace zachary_pushups_count_l635_63571

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 51

/-- The number of push-ups David did -/
def david_pushups : ℕ := zachary_pushups + 22

/-- The number of push-ups John did -/
def john_pushups : ℕ := 69

theorem zachary_pushups_count : zachary_pushups = 51 := by
  have h1 : david_pushups = zachary_pushups + 22 := rfl
  have h2 : john_pushups = david_pushups - 4 := by sorry
  have h3 : john_pushups = 69 := rfl
  sorry

end zachary_pushups_count_l635_63571


namespace meaningful_reciprocal_l635_63599

theorem meaningful_reciprocal (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := by
  sorry

end meaningful_reciprocal_l635_63599


namespace min_value_theorem_equality_exists_l635_63568

theorem min_value_theorem (x : ℝ) (h : x > 0) : 2*x + 1/(2*x) + 1 ≥ 3 := by
  sorry

theorem equality_exists : ∃ x : ℝ, x > 0 ∧ 2*x + 1/(2*x) + 1 = 3 := by
  sorry

end min_value_theorem_equality_exists_l635_63568


namespace right_triangle_properties_l635_63563

/-- A triangle with side lengths 13, 84, and 85 is a right triangle with area 546, semiperimeter 91, and inradius 6 -/
theorem right_triangle_properties : ∃ (a b c : ℝ), 
  a = 13 ∧ b = 84 ∧ c = 85 ∧
  a^2 + b^2 = c^2 ∧
  (1/2 * a * b : ℝ) = 546 ∧
  ((a + b + c) / 2 : ℝ) = 91 ∧
  (546 / 91 : ℝ) = 6 := by
sorry


end right_triangle_properties_l635_63563


namespace circle_in_square_l635_63510

theorem circle_in_square (r : ℝ) (h : r = 6) :
  let square_side := 2 * r
  let square_area := square_side ^ 2
  let smaller_square_side := square_side - 2
  let smaller_square_area := smaller_square_side ^ 2
  (square_area = 144 ∧ square_area - smaller_square_area = 44) := by
  sorry

end circle_in_square_l635_63510


namespace equation_one_solution_l635_63548

theorem equation_one_solution (x : ℝ) : 
  (x - 1)^2 - 4 = 0 ↔ x = -1 ∨ x = 3 := by sorry

end equation_one_solution_l635_63548


namespace like_terms_exponent_difference_l635_63514

theorem like_terms_exponent_difference (m n : ℤ) : 
  (∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ 4 * x^(2*m+2) * y^(n-1) = -3 * x^(3*m+1) * y^(3*n-5)) → 
  m - n = -1 := by
sorry

end like_terms_exponent_difference_l635_63514


namespace y_equation_solution_l635_63590

theorem y_equation_solution (y : ℝ) (c d : ℕ+) 
  (h1 : y^2 + 4*y + 4/y + 1/y^2 = 30)
  (h2 : y = c + Real.sqrt d) :
  c + d = 5 := by
  sorry

end y_equation_solution_l635_63590


namespace five_cubic_yards_equals_135_cubic_feet_l635_63591

/-- Converts cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet (yards : ℝ) : ℝ :=
  yards * 27

theorem five_cubic_yards_equals_135_cubic_feet :
  cubic_yards_to_cubic_feet 5 = 135 := by
  sorry

end five_cubic_yards_equals_135_cubic_feet_l635_63591


namespace complex_fraction_simplification_l635_63561

theorem complex_fraction_simplification (x : ℝ) 
  (h1 : x ≠ 2) (h2 : x ≠ 4) (h3 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) = 
  1 / ((x - 2) * (x - 4)) := by
  sorry

end complex_fraction_simplification_l635_63561


namespace quadratic_roots_when_positive_discriminant_l635_63516

theorem quadratic_roots_when_positive_discriminant
  (a b c : ℝ) (ha : a ≠ 0) (h_disc : b^2 - 4*a*c > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + b*x₁ + c = 0 ∧ a*x₂^2 + b*x₂ + c = 0 :=
sorry

end quadratic_roots_when_positive_discriminant_l635_63516


namespace square_number_divisible_by_nine_between_40_and_90_l635_63509

theorem square_number_divisible_by_nine_between_40_and_90 :
  ∃ x : ℕ, x^2 = x ∧ x % 9 = 0 ∧ 40 < x ∧ x < 90 → x = 81 :=
by sorry

end square_number_divisible_by_nine_between_40_and_90_l635_63509


namespace pentagon_area_ratio_l635_63527

-- Define the pentagon
structure Pentagon where
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  I : ℝ × ℝ
  J : ℝ × ℝ

def is_convex (p : Pentagon) : Prop := sorry

-- Define parallel lines
def parallel (a b c d : ℝ × ℝ) : Prop := sorry

-- Define angle measurement
def angle (a b c : ℝ × ℝ) : ℝ := sorry

-- Define distance between points
def distance (a b : ℝ × ℝ) : ℝ := sorry

-- Define area of a triangle
def triangle_area (a b c : ℝ × ℝ) : ℝ := sorry

theorem pentagon_area_ratio (p : Pentagon) :
  is_convex p →
  parallel p.F p.G p.I p.J →
  parallel p.G p.H p.F p.I →
  parallel p.G p.I p.H p.J →
  angle p.F p.G p.H = 120 * π / 180 →
  distance p.F p.G = 4 →
  distance p.G p.H = 6 →
  distance p.H p.J = 18 →
  (triangle_area p.F p.G p.H) / (triangle_area p.G p.H p.I) = 16 / 171 := by
  sorry

end pentagon_area_ratio_l635_63527


namespace distance_between_P_and_Q_l635_63517

theorem distance_between_P_and_Q : ∀ (pq : ℝ),
  (∃ (x : ℝ),
    -- A walks 30 km each day
    30 * x = pq ∧
    -- B starts after A has walked 72 km
    72 + 30 * (pq / 80) = x ∧
    -- B walks 1/10 of the total distance each day
    (pq / 10) * (pq / 80) = pq - x ∧
    -- B meets A after walking for 1/8 of the daily km covered
    (pq / 10) * (1 / 8) = pq / 80) →
  pq = 320 ∨ pq = 180 := by
sorry

end distance_between_P_and_Q_l635_63517


namespace positive_A_value_l635_63523

theorem positive_A_value : ∃ A : ℕ+, A^2 - 1 = 3577 * 3579 ∧ A = 3578 := by
  sorry

end positive_A_value_l635_63523


namespace linear_equation_natural_solution_l635_63520

theorem linear_equation_natural_solution (m : ℤ) : 
  (∃ x : ℕ, m * (x : ℤ) - 6 = x) ↔ m ∈ ({2, 3, 4, 7} : Set ℤ) := by sorry

end linear_equation_natural_solution_l635_63520


namespace min_sum_of_squares_l635_63521

theorem min_sum_of_squares (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 := by
  sorry

end min_sum_of_squares_l635_63521
