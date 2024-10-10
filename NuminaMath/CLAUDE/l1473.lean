import Mathlib

namespace sqrt_sum_equals_thirteen_sixths_l1473_147365

theorem sqrt_sum_equals_thirteen_sixths : 
  Real.sqrt (9 / 4) + Real.sqrt (4 / 9) = 13 / 6 := by sorry

end sqrt_sum_equals_thirteen_sixths_l1473_147365


namespace amount_decreased_l1473_147387

theorem amount_decreased (x y : ℝ) (h1 : x = 50.0) (h2 : 0.20 * x - y = 6) : y = 4 := by
  sorry

end amount_decreased_l1473_147387


namespace power_fraction_simplification_l1473_147381

theorem power_fraction_simplification : (3^100 + 3^98) / (3^100 - 3^98) = 5/4 := by sorry

end power_fraction_simplification_l1473_147381


namespace power_function_m_value_l1473_147350

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a : ℝ) (n : ℝ), ∀ x, f x = a * x^n

theorem power_function_m_value (m : ℝ) :
  is_power_function (λ x => (3*m - 1) * x^m) → m = 2/3 :=
by
  sorry

end power_function_m_value_l1473_147350


namespace tangent_line_at_point_one_l1473_147341

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_line_at_point_one (x y : ℝ) :
  f 1 = -1 →
  f' 1 = 1 →
  (y - f 1 = f' 1 * (x - 1)) ↔ x - y - 2 = 0 :=
by sorry

end tangent_line_at_point_one_l1473_147341


namespace complex_expression_equals_two_l1473_147396

theorem complex_expression_equals_two :
  (Complex.I * (1 - Complex.I)^2 : ℂ) = 2 := by sorry

end complex_expression_equals_two_l1473_147396


namespace evelyn_found_caps_l1473_147366

/-- The number of bottle caps Evelyn started with -/
def starting_caps : ℕ := 18

/-- The number of bottle caps Evelyn ended up with -/
def total_caps : ℕ := 81

/-- The number of bottle caps Evelyn found -/
def found_caps : ℕ := total_caps - starting_caps

theorem evelyn_found_caps : found_caps = 63 := by
  sorry

end evelyn_found_caps_l1473_147366


namespace f_properties_l1473_147363

def f (x : ℝ) : ℝ := x^3 + x^2 - 8*x + 6

theorem f_properties :
  (∀ x, deriv f x = 3*x^2 + 2*x - 8) ∧
  deriv f (-2) = 0 ∧
  deriv f 1 = -3 ∧
  f 1 = 0 ∧
  (∀ x, x < -2 ∨ x > 4/3 → deriv f x > 0) ∧
  (∀ x, -2 < x ∧ x < 4/3 → deriv f x < 0) :=
by sorry

end f_properties_l1473_147363


namespace largest_prime_in_equation_l1473_147329

theorem largest_prime_in_equation (x : ℤ) (n : ℕ) (p : ℕ) 
  (hp : Nat.Prime p) (heq : 7 * x^2 - 44 * x + 12 = p^n) :
  p ≤ 47 :=
sorry

end largest_prime_in_equation_l1473_147329


namespace expression_equality_l1473_147398

theorem expression_equality (x : ℝ) :
  3 * (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 
  ((Real.sqrt 3 - 1) * x + 5 + 2 * Real.sqrt 3)^2 := by
  sorry

end expression_equality_l1473_147398


namespace smallest_n_value_l1473_147360

/-- Represents the dimensions of a rectangular block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in a block given its dimensions -/
def totalCubes (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of invisible cubes when three faces are shown -/
def invisibleCubes (d : BlockDimensions) : ℕ :=
  (d.length - 1) * (d.width - 1) * (d.height - 1)

/-- Theorem stating the smallest possible value of N -/
theorem smallest_n_value (d : BlockDimensions) : 
  invisibleCubes d = 143 → totalCubes d ≥ 336 ∧ ∃ d', invisibleCubes d' = 143 ∧ totalCubes d' = 336 := by
  sorry

end smallest_n_value_l1473_147360


namespace coin_flip_probability_l1473_147397

theorem coin_flip_probability :
  let n : ℕ := 12  -- number of coin flips
  let k : ℕ := 9   -- number of heads we want
  let p : ℚ := 1/2 -- probability of heads for a fair coin
  Nat.choose n k * p^k * (1-p)^(n-k) = 55/1024 := by
sorry

end coin_flip_probability_l1473_147397


namespace commute_days_theorem_l1473_147313

theorem commute_days_theorem (x : ℕ) 
  (morning_bus : ℕ) 
  (afternoon_train : ℕ) 
  (bike_commute : ℕ) 
  (h1 : morning_bus = 12) 
  (h2 : afternoon_train = 20) 
  (h3 : bike_commute = 10) 
  (h4 : x = morning_bus + afternoon_train - bike_commute) : x = 30 := by
  sorry

#check commute_days_theorem

end commute_days_theorem_l1473_147313


namespace second_store_unload_percentage_l1473_147328

def initial_load : ℝ := 50000
def first_unload_percent : ℝ := 0.1
def remaining_after_deliveries : ℝ := 36000

theorem second_store_unload_percentage :
  let remaining_after_first := initial_load * (1 - first_unload_percent)
  let unloaded_at_second := remaining_after_first - remaining_after_deliveries
  (unloaded_at_second / remaining_after_first) * 100 = 20 := by
  sorry

end second_store_unload_percentage_l1473_147328


namespace fraction_doubling_l1473_147353

theorem fraction_doubling (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1.4 * x) / (0.7 * y) = 2 * (x / y) :=
by sorry

end fraction_doubling_l1473_147353


namespace no_real_solutions_l1473_147393

theorem no_real_solutions :
  ∀ x : ℝ, x ≠ 2 → (3 * x^2) / (x - 2) - (3 * x + 9) / 4 + (5 - 9 * x) / (x - 2) + 2 ≠ 0 := by
  sorry

end no_real_solutions_l1473_147393


namespace kozel_garden_problem_l1473_147321

theorem kozel_garden_problem (x : ℕ) (y : ℕ) : 
  (y = 3 * x + 1) → 
  (y = 4 * (x - 1)) → 
  (x = 5 ∧ y = 16) :=
by sorry

end kozel_garden_problem_l1473_147321


namespace perpendicular_chords_sum_bounds_l1473_147386

/-- Given a circle with radius R and an interior point P at distance kR from the center,
    where 0 ≤ k ≤ 1, the sum of the lengths of two perpendicular chords passing through P
    is bounded above by 2R√(2(1 - k²)) and below by 0. -/
theorem perpendicular_chords_sum_bounds (R k : ℝ) (h_R_pos : R > 0) (h_k_range : 0 ≤ k ∧ k ≤ 1) :
  ∃ (chord_sum : ℝ), 0 ≤ chord_sum ∧ chord_sum ≤ 2 * R * Real.sqrt (2 * (1 - k^2)) := by
  sorry

end perpendicular_chords_sum_bounds_l1473_147386


namespace parallel_vectors_m_value_l1473_147302

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ (m : ℝ),
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (m, -1)
  are_parallel a b → m = 1/2 := by
  sorry

end parallel_vectors_m_value_l1473_147302


namespace valentino_farm_birds_l1473_147367

/-- The number of birds on Mr. Valentino's farm -/
def total_birds (chickens : ℕ) : ℕ :=
  let ducks := 2 * chickens
  let turkeys := 3 * ducks
  chickens + ducks + turkeys

/-- Theorem stating the total number of birds on Mr. Valentino's farm -/
theorem valentino_farm_birds :
  total_birds 200 = 1800 := by
  sorry

end valentino_farm_birds_l1473_147367


namespace product_as_sum_of_squares_l1473_147355

theorem product_as_sum_of_squares : 
  85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^ 2 := by
  sorry

end product_as_sum_of_squares_l1473_147355


namespace sum_difference_theorem_l1473_147389

def jo_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_ten (x : ℕ) : ℕ :=
  let r := x % 10
  if r < 5 then x - r else x + (10 - r)

def kate_sum (n : ℕ) : ℕ :=
  List.range n |> List.map (λ x => round_to_nearest_ten (x + 1)) |> List.sum

theorem sum_difference_theorem :
  jo_sum 60 - kate_sum 60 = 1530 := by sorry

end sum_difference_theorem_l1473_147389


namespace preceding_number_in_base_three_l1473_147352

/-- Converts a base-3 number to decimal --/
def baseThreeToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Converts a decimal number to base-3 --/
def decimalToBaseThree (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
    aux n []

theorem preceding_number_in_base_three (M : List Nat) (h : M = [2, 1, 0, 2, 1]) :
  decimalToBaseThree (baseThreeToDecimal M - 1) = [2, 1, 0, 2, 0] := by
  sorry

end preceding_number_in_base_three_l1473_147352


namespace hiltons_marbles_l1473_147348

theorem hiltons_marbles (initial_marbles : ℕ) : 
  (initial_marbles + 6 - 10 + 2 * 10 = 42) → initial_marbles = 26 := by
  sorry

end hiltons_marbles_l1473_147348


namespace exists_integer_fifth_power_less_than_one_l1473_147383

theorem exists_integer_fifth_power_less_than_one :
  ∃ x : ℤ, x^5 < 1 := by sorry

end exists_integer_fifth_power_less_than_one_l1473_147383


namespace lateral_edge_length_l1473_147364

-- Define the regular triangular pyramid
structure RegularTriangularPyramid where
  baseEdge : ℝ
  lateralEdge : ℝ

-- Define the property of medians not intersecting and lying on cube edges
def mediansPropertyHolds (pyramid : RegularTriangularPyramid) : Prop :=
  -- This is a placeholder for the complex geometric condition
  -- In a real implementation, this would involve more detailed geometric definitions
  sorry

-- Theorem statement
theorem lateral_edge_length
  (pyramid : RegularTriangularPyramid)
  (h1 : pyramid.baseEdge = 1)
  (h2 : mediansPropertyHolds pyramid) :
  pyramid.lateralEdge = Real.sqrt 6 / 2 :=
sorry

end lateral_edge_length_l1473_147364


namespace students_left_on_bus_l1473_147399

theorem students_left_on_bus (initial_students : ℕ) (students_off : ℕ) : 
  initial_students = 10 → students_off = 3 → initial_students - students_off = 7 := by
  sorry

end students_left_on_bus_l1473_147399


namespace antonio_hamburger_usage_l1473_147395

/-- Calculates the total amount of hamburger used for meatballs given the number of family members,
    meatballs per person, and amount of hamburger per meatball. -/
def hamburger_used (family_members : ℕ) (meatballs_per_person : ℕ) (hamburger_per_meatball : ℚ) : ℚ :=
  (family_members * meatballs_per_person : ℚ) * hamburger_per_meatball

/-- Proves that given the conditions in the problem, Antonio used 4 pounds of hamburger. -/
theorem antonio_hamburger_usage :
  let family_members : ℕ := 8
  let meatballs_per_person : ℕ := 4
  let hamburger_per_meatball : ℚ := 1/8
  hamburger_used family_members meatballs_per_person hamburger_per_meatball = 4 := by
  sorry


end antonio_hamburger_usage_l1473_147395


namespace max_value_of_min_expression_l1473_147312

theorem max_value_of_min_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  min x (min (-1/y) (y + 1/x)) ≤ Real.sqrt 2 ∧
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ min x (min (-1/y) (y + 1/x)) = Real.sqrt 2 :=
by sorry

end max_value_of_min_expression_l1473_147312


namespace complex_calculation_equality_l1473_147309

theorem complex_calculation_equality : 
  (2 * (15^2 + 35^2 + 21^2) - (3^4 + 5^4 + 7^4)) / (3 + 5 + 7) = 45 := by
  sorry

end complex_calculation_equality_l1473_147309


namespace freddy_travel_time_l1473_147335

/-- Represents the travel details of a person --/
structure TravelDetails where
  startCity : String
  endCity : String
  distance : Real
  time : Real

/-- The problem setup --/
def problem : Prop := ∃ (eddySpeed freddySpeed : Real),
  let eddy : TravelDetails := ⟨"A", "B", 900, 3⟩
  let freddy : TravelDetails := ⟨"A", "C", 300, freddySpeed / 300⟩
  eddySpeed = eddy.distance / eddy.time ∧
  eddySpeed / freddySpeed = 4 ∧
  freddy.time = 4

/-- The theorem to be proved --/
theorem freddy_travel_time : problem := by sorry

end freddy_travel_time_l1473_147335


namespace john_travel_distance_l1473_147370

/-- Calculates the total distance traveled given a constant speed and two driving periods -/
def totalDistance (speed : ℝ) (time1 : ℝ) (time2 : ℝ) : ℝ :=
  speed * (time1 + time2)

/-- Proves that the total distance traveled is 225 miles -/
theorem john_travel_distance :
  let speed := 45
  let time1 := 2
  let time2 := 3
  totalDistance speed time1 time2 = 225 := by
sorry

end john_travel_distance_l1473_147370


namespace tangent_line_at_minus_one_l1473_147317

def curve (x : ℝ) : ℝ := x^3

theorem tangent_line_at_minus_one : 
  let p : ℝ × ℝ := (-1, -1)
  let m : ℝ := 3 * p.1^2
  let tangent_line (x : ℝ) : ℝ := m * (x - p.1) + p.2
  ∀ x, tangent_line x = 3 * x + 2 := by
  sorry

end tangent_line_at_minus_one_l1473_147317


namespace magic_deck_price_is_two_l1473_147375

/-- The price of a magic card deck given initial and final quantities and total earnings -/
def magic_deck_price (initial : ℕ) (final : ℕ) (earnings : ℕ) : ℚ :=
  earnings / (initial - final)

/-- Theorem: The price of each magic card deck is 2 dollars -/
theorem magic_deck_price_is_two :
  magic_deck_price 5 3 4 = 2 := by
  sorry

end magic_deck_price_is_two_l1473_147375


namespace exam_maximum_marks_l1473_147394

theorem exam_maximum_marks 
  (passing_percentage : ℝ)
  (student_score : ℕ)
  (failing_margin : ℕ)
  (h1 : passing_percentage = 0.45)
  (h2 : student_score = 40)
  (h3 : failing_margin = 40) :
  ∃ (max_marks : ℕ), max_marks = 180 ∧ 
    (passing_percentage * max_marks : ℝ) = (student_score + failing_margin) :=
by sorry

end exam_maximum_marks_l1473_147394


namespace profit_doubling_l1473_147334

theorem profit_doubling (cost : ℝ) (original_price : ℝ) :
  original_price = cost * 1.6 →
  let double_price := 2 * original_price
  (double_price - cost) / cost * 100 = 220 := by
sorry

end profit_doubling_l1473_147334


namespace gain_percent_calculation_l1473_147327

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h : 50 * cost_price = 46 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = 100 / 11.5 := by
  sorry

end gain_percent_calculation_l1473_147327


namespace chord_equation_l1473_147351

/-- Given positive real numbers m, n, s, t satisfying certain conditions,
    prove that the equation of the line containing a chord of the hyperbola
    x²/4 - y²/2 = 1 with midpoint (m, n) is x - 2y + 1 = 0. -/
theorem chord_equation (m n s t : ℝ) 
    (hm : m > 0) (hn : n > 0) (hs : s > 0) (ht : t > 0)
    (h1 : m + n = 2)
    (h2 : m / s + n / t = 9)
    (h3 : s + t = 4 / 9)
    (h4 : ∀ s' t' : ℝ, s' > 0 → t' > 0 → m / s' + n / t' = 9 → s' + t' ≥ 4 / 9)
    (h5 : ∃ x₁ y₁ x₂ y₂ : ℝ, 
      x₁^2 / 4 - y₁^2 / 2 = 1 ∧
      x₂^2 / 4 - y₂^2 / 2 = 1 ∧
      (x₁ + x₂) / 2 = m ∧
      (y₁ + y₂) / 2 = n) :
  ∃ a b c : ℝ, a * m + b * n + c = 0 ∧
             ∀ x y : ℝ, x^2 / 4 - y^2 / 2 = 1 →
               (∃ t : ℝ, x = m + t * a ∧ y = n + t * b) →
               a * x + b * y + c = 0 ∧
               a = 1 ∧ b = -2 ∧ c = 1 :=
by sorry

end chord_equation_l1473_147351


namespace f_no_extreme_points_l1473_147390

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + 3*x - a

-- Theorem stating that f has no extreme points for any real a
theorem f_no_extreme_points (a : ℝ) : 
  ∀ x : ℝ, ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f a y ≠ f a x ∨ (f a y < f a x ∧ y < x) ∨ (f a y > f a x ∧ y > x) :=
sorry

end f_no_extreme_points_l1473_147390


namespace candies_problem_l1473_147307

theorem candies_problem (n : ℕ) (a : ℕ) (h1 : n > 0) (h2 : a > 1) 
  (h3 : ∀ i : Fin n, a = n * a - a - 7) : n * a = 21 := by
  sorry

end candies_problem_l1473_147307


namespace arithmetic_sequence_ninth_term_l1473_147332

/-- Given an arithmetic sequence where the third term is 23 and the sixth term is 29,
    prove that the ninth term is 35. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)  -- a is the arithmetic sequence
  (h1 : a 3 = 23)  -- third term is 23
  (h2 : a 6 = 29)  -- sixth term is 29
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- definition of arithmetic sequence
  : a 9 = 35 := by
  sorry

end arithmetic_sequence_ninth_term_l1473_147332


namespace f_value_at_3_l1473_147316

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 2

-- State the theorem
theorem f_value_at_3 (a b : ℝ) (h : f a b (-3) = -1) : f a b 3 = 5 := by
  sorry

end f_value_at_3_l1473_147316


namespace constant_sum_of_powers_l1473_147325

/-- S_n is constant for real x, y, z with xyz = 1 and x + y + z = 0 iff n = 1 or n = 3 -/
theorem constant_sum_of_powers (n : ℕ+) :
  (∀ x y z : ℝ, x * y * z = 1 → x + y + z = 0 → 
    ∃ c : ℝ, ∀ x' y' z' : ℝ, x' * y' * z' = 1 → x' + y' + z' = 0 → 
      x'^(n : ℕ) + y'^(n : ℕ) + z'^(n : ℕ) = c) ↔ 
  n = 1 ∨ n = 3 := by
sorry

end constant_sum_of_powers_l1473_147325


namespace chess_tournament_games_l1473_147357

/-- The number of games in a chess tournament --/
def num_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem stating the number of games in the specific tournament --/
theorem chess_tournament_games :
  num_games 20 3 = 570 := by sorry

end chess_tournament_games_l1473_147357


namespace rolles_theorem_application_l1473_147300

-- Define the function f(x) = x^2 + 2x + 7
def f (x : ℝ) : ℝ := x^2 + 2*x + 7

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 2*x + 2

-- Theorem statement
theorem rolles_theorem_application :
  ∃ c ∈ Set.Ioo (-6 : ℝ) 4, f' c = 0 :=
by
  -- Proof goes here
  sorry

end rolles_theorem_application_l1473_147300


namespace sqrt_difference_equals_one_l1473_147345

theorem sqrt_difference_equals_one : 
  Real.sqrt 9 - Real.sqrt ((-2)^2) = 1 := by sorry

end sqrt_difference_equals_one_l1473_147345


namespace inequality_holds_for_all_reals_l1473_147314

theorem inequality_holds_for_all_reals (a b : ℝ) (h : |a - b| > 2) :
  ∀ x : ℝ, |x - a| + |x - b| > 2 := by
sorry

end inequality_holds_for_all_reals_l1473_147314


namespace banana_orange_equivalence_l1473_147333

/-- Given that 3/4 of 12 bananas are worth as much as 9 oranges,
    prove that 3/5 of 15 bananas are worth as much as 9 oranges. -/
theorem banana_orange_equivalence (banana_value : ℚ) :
  (3/4 : ℚ) * 12 * banana_value = 9 →
  (3/5 : ℚ) * 15 * banana_value = 9 :=
by sorry

end banana_orange_equivalence_l1473_147333


namespace solve_batting_problem_l1473_147371

def batting_problem (pitches_per_token : ℕ) (macy_tokens : ℕ) (piper_tokens : ℕ) 
  (macy_hits : ℕ) (total_misses : ℕ) : Prop :=
  let total_pitches := pitches_per_token * (macy_tokens + piper_tokens)
  let total_hits := total_pitches - total_misses
  let piper_hits := total_hits - macy_hits
  piper_hits = 55

theorem solve_batting_problem :
  batting_problem 15 11 17 50 315 := by
  sorry

end solve_batting_problem_l1473_147371


namespace sylvester_theorem_l1473_147346

-- Define coprimality
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the theorem
theorem sylvester_theorem (a b : ℕ) (h : coprime a b) :
  -- Part 1: Unique solution in the strip
  (∀ c : ℕ, ∃! p : ℕ × ℕ, p.1 < b ∧ a * p.1 + b * p.2 = c) ∧
  -- Part 2: Largest value without non-negative solutions
  (∀ c : ℕ, c > a * b - a - b → ∃ x y : ℕ, a * x + b * y = c) ∧
  (¬∃ x y : ℕ, a * x + b * y = a * b - a - b) := by
  sorry

end sylvester_theorem_l1473_147346


namespace percent_increase_decrease_l1473_147358

theorem percent_increase_decrease (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hq_bound : q < 100) (hM : M > 0) :
  (M * (1 + p/100) * (1 - q/100) > M) ↔ (p > 100*q / (100 - q)) := by
  sorry

end percent_increase_decrease_l1473_147358


namespace ab_value_l1473_147361

theorem ab_value (a b : ℝ) (h : (a + 3)^2 + (b - 3)^2 = 0) : a^b = -27 := by
  sorry

end ab_value_l1473_147361


namespace rhombus_construction_exists_l1473_147308

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry

-- Define a rhombus
structure Rhombus where
  vertices : Fin 4 → ℝ × ℝ
  is_rhombus : sorry

-- Define the property of sides being parallel to diagonals
def parallel_to_diagonals (r : Rhombus) (q : ConvexQuadrilateral) : Prop :=
  sorry

-- Define the property of vertices lying on the sides of the quadrilateral
def vertices_on_sides (r : Rhombus) (q : ConvexQuadrilateral) : Prop :=
  sorry

theorem rhombus_construction_exists (q : ConvexQuadrilateral) :
  ∃ (r : Rhombus), vertices_on_sides r q ∧ parallel_to_diagonals r q :=
sorry

end rhombus_construction_exists_l1473_147308


namespace not_divides_2007_l1473_147392

theorem not_divides_2007 : ¬(2007 ∣ (2009^3 - 2009)) := by sorry

end not_divides_2007_l1473_147392


namespace bubble_pass_probability_specific_l1473_147322

def bubble_pass_probability (n : ℕ) (initial_pos : ℕ) (final_pos : ℕ) : ℚ :=
  if initial_pos < final_pos ∧ final_pos ≤ n then
    1 / (initial_pos * (final_pos - 1))
  else
    0

theorem bubble_pass_probability_specific :
  bubble_pass_probability 50 25 35 = 1 / 850 := by
  sorry

end bubble_pass_probability_specific_l1473_147322


namespace unique_solution_g_equals_g_inv_l1473_147362

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x - 12

-- Define the inverse function g⁻¹
noncomputable def g_inv (x : ℝ) : ℝ := (x + 12) / 5

-- Theorem statement
theorem unique_solution_g_equals_g_inv :
  ∃! x : ℝ, g x = g_inv x :=
by
  sorry

end unique_solution_g_equals_g_inv_l1473_147362


namespace square_of_binomial_coefficient_l1473_147338

theorem square_of_binomial_coefficient (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 - 18*x + a = (3*x + b)^2) → a = 9 := by
  sorry

end square_of_binomial_coefficient_l1473_147338


namespace ellipse_equation_l1473_147303

/-- The standard equation of an ellipse with given properties -/
theorem ellipse_equation (e : ℝ) (P : ℝ × ℝ) : 
  e = Real.sqrt 5 / 5 →
  P = (-5, 4) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 45 + y^2 / 36 = 1) :=
by
  sorry

#check ellipse_equation

end ellipse_equation_l1473_147303


namespace laran_sells_five_posters_l1473_147311

/-- Represents the poster business model for Laran -/
structure PosterBusiness where
  large_posters_per_day : ℕ
  large_poster_price : ℕ
  large_poster_cost : ℕ
  small_poster_price : ℕ
  small_poster_cost : ℕ
  weekly_profit : ℕ
  school_days_per_week : ℕ

/-- Calculates the total number of posters sold per day -/
def total_posters_per_day (b : PosterBusiness) : ℕ :=
  b.large_posters_per_day + 
  ((b.weekly_profit / b.school_days_per_week - 
    (b.large_posters_per_day * (b.large_poster_price - b.large_poster_cost))) / 
   (b.small_poster_price - b.small_poster_cost))

/-- Theorem stating that Laran sells 5 posters per day -/
theorem laran_sells_five_posters (b : PosterBusiness) 
  (h1 : b.large_posters_per_day = 2)
  (h2 : b.large_poster_price = 10)
  (h3 : b.large_poster_cost = 5)
  (h4 : b.small_poster_price = 6)
  (h5 : b.small_poster_cost = 3)
  (h6 : b.weekly_profit = 95)
  (h7 : b.school_days_per_week = 5) :
  total_posters_per_day b = 5 := by
  sorry

end laran_sells_five_posters_l1473_147311


namespace beadshop_profit_l1473_147379

theorem beadshop_profit (monday_profit_ratio : ℚ) (tuesday_profit_ratio : ℚ) (wednesday_profit : ℚ) 
  (h1 : monday_profit_ratio = 1/3)
  (h2 : tuesday_profit_ratio = 1/4)
  (h3 : wednesday_profit = 500) :
  ∃ total_profit : ℚ, 
    total_profit * (1 - monday_profit_ratio - tuesday_profit_ratio) = wednesday_profit ∧
    total_profit = 1200 := by
sorry

end beadshop_profit_l1473_147379


namespace simple_interest_principal_l1473_147305

/-- Simple interest calculation --/
theorem simple_interest_principal
  (interest : ℚ)
  (rate : ℚ)
  (time : ℚ)
  (h1 : interest = 160)
  (h2 : rate = 4 / 100)
  (h3 : time = 5) :
  interest = (800 * rate * time) :=
by sorry

end simple_interest_principal_l1473_147305


namespace task_completion_time_l1473_147384

/-- The time required for Sumin and Junwoo to complete a task together, given their individual work rates -/
theorem task_completion_time (sumin_rate junwoo_rate : ℚ) 
  (h_sumin : sumin_rate = 1 / 10)
  (h_junwoo : junwoo_rate = 1 / 15) :
  (1 : ℚ) / (sumin_rate + junwoo_rate) = 6 := by
  sorry

end task_completion_time_l1473_147384


namespace barry_total_amount_l1473_147391

/-- Calculates the total amount Barry needs to pay for his purchase --/
def calculate_total_amount (shirt_price pants_price tie_price : ℝ)
  (shirt_discount pants_discount coupon_discount sales_tax : ℝ) : ℝ :=
  let discounted_shirt := shirt_price * (1 - shirt_discount)
  let discounted_pants := pants_price * (1 - pants_discount)
  let subtotal := discounted_shirt + discounted_pants + tie_price
  let after_coupon := subtotal * (1 - coupon_discount)
  let total := after_coupon * (1 + sales_tax)
  total

/-- Theorem stating that the total amount Barry needs to pay is $201.27 --/
theorem barry_total_amount : 
  calculate_total_amount 80 100 40 0.15 0.10 0.05 0.07 = 201.27 := by
  sorry

end barry_total_amount_l1473_147391


namespace currency_notes_total_l1473_147342

theorem currency_notes_total (total_notes : ℕ) (denom_1 denom_2 : ℕ) (amount_denom_2 : ℕ) : 
  total_notes = 100 → 
  denom_1 = 70 → 
  denom_2 = 50 → 
  amount_denom_2 = 100 →
  ∃ (notes_denom_1 notes_denom_2 : ℕ),
    notes_denom_1 + notes_denom_2 = total_notes ∧
    notes_denom_2 * denom_2 = amount_denom_2 ∧
    notes_denom_1 * denom_1 + notes_denom_2 * denom_2 = 6960 :=
by sorry

end currency_notes_total_l1473_147342


namespace number_of_people_l1473_147347

/-- Given a group of people, prove that there are 5 people based on the given conditions. -/
theorem number_of_people (n : ℕ) (total_age : ℕ) : n = 5 :=
  by
  /- Define the average age of all people -/
  have avg_age : total_age = n * 30 := by sorry
  
  /- Define the total age when the youngest was born -/
  have prev_total_age : total_age - 6 = (n - 1) * 24 := by sorry
  
  /- The main proof -/
  sorry

end number_of_people_l1473_147347


namespace four_numbers_product_sum_prime_l1473_147359

theorem four_numbers_product_sum_prime :
  ∃ (a b c d : ℕ), a < b ∧ b < c ∧ c < d ∧
  Nat.Prime (a * b + c * d) ∧
  Nat.Prime (a * c + b * d) ∧
  Nat.Prime (a * d + b * c) := by
  sorry

end four_numbers_product_sum_prime_l1473_147359


namespace A_minus_3B_equals_x_cubed_plus_y_cubed_l1473_147339

variable (x y : ℝ)

def A : ℝ := x^3 + 3*x^2*y + y^3 - 3*x*y^2
def B : ℝ := x^2*y - x*y^2

theorem A_minus_3B_equals_x_cubed_plus_y_cubed :
  A x y - 3 * B x y = x^3 + y^3 := by sorry

end A_minus_3B_equals_x_cubed_plus_y_cubed_l1473_147339


namespace trigonometric_expression_equality_l1473_147378

theorem trigonometric_expression_equality : 
  (Real.sin (92 * π / 180) - Real.sin (32 * π / 180) * Real.cos (60 * π / 180)) / Real.cos (32 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end trigonometric_expression_equality_l1473_147378


namespace xy_value_l1473_147326

theorem xy_value (x y : ℝ) (h : Real.sqrt (x - 3) + |y + 2| = 0) : x * y = -6 := by
  sorry

end xy_value_l1473_147326


namespace trigonometric_equation_solution_l1473_147374

theorem trigonometric_equation_solution (θ : ℝ) :
  3 * Real.sin (-3 * Real.pi + θ) + Real.cos (Real.pi - θ) = 0 →
  (Real.sin θ * Real.cos θ) / Real.cos (2 * θ) = -3/8 := by
  sorry

end trigonometric_equation_solution_l1473_147374


namespace cubic_equation_solutions_l1473_147376

theorem cubic_equation_solutions :
  let f : ℂ → ℂ := λ x => (x^3 + 3*x^2*Real.sqrt 2 + 6*x + 2*Real.sqrt 2) + (x + Real.sqrt 2)
  ∀ x : ℂ, f x = 0 ↔ x = -Real.sqrt 2 ∨ x = -Real.sqrt 2 + Complex.I ∨ x = -Real.sqrt 2 - Complex.I :=
by sorry

end cubic_equation_solutions_l1473_147376


namespace birch_tree_spacing_probability_l1473_147315

def total_trees : ℕ := 15
def pine_trees : ℕ := 4
def maple_trees : ℕ := 5
def birch_trees : ℕ := 6

theorem birch_tree_spacing_probability :
  let non_birch_trees := pine_trees + maple_trees
  let total_arrangements := (total_trees.choose birch_trees : ℚ)
  let valid_arrangements := ((non_birch_trees + 1).choose birch_trees : ℚ)
  valid_arrangements / total_arrangements = 2 / 95 := by
sorry

end birch_tree_spacing_probability_l1473_147315


namespace x_in_open_interval_one_two_l1473_147356

/-- A monotonically increasing function on (0,+∞) -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ 0 < y ∧ x < y → f x < f y

theorem x_in_open_interval_one_two
  (f : ℝ → ℝ)
  (h_mono : MonoIncreasing f)
  (h_gt : ∀ x, 0 < x → f x > f (2 - x)) :
  ∃ x, 1 < x ∧ x < 2 :=
sorry

end x_in_open_interval_one_two_l1473_147356


namespace alien_mineral_collection_l1473_147354

/-- Converts a base-6 number represented as a list of digits to its base-10 equivalent -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The base-6 representation of the number -/
def alienCollection : List Nat := [5, 3, 2]

theorem alien_mineral_collection :
  base6ToBase10 alienCollection = 95 := by
  sorry

end alien_mineral_collection_l1473_147354


namespace unique_symmetric_solutions_l1473_147369

theorem unique_symmetric_solutions (a b α β : ℝ) :
  (α * β = a ∧ α + β = b) →
  (∀ x y : ℝ, x * y = a ∧ x + y = b ↔ (x = α ∧ y = β) ∨ (x = β ∧ y = α)) :=
sorry

end unique_symmetric_solutions_l1473_147369


namespace problem_solution_l1473_147330

def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 1|

theorem problem_solution (a : ℝ) :
  (∀ x, f a x ≤ 2 ↔ -1/2 ≤ x ∧ x ≤ 3/2) →
  (a = 2 ∧
   ∀ x, f 2 x + f 2 (x/2 - 1) ≥ 5 ↔ x ≥ 3 ∨ x ≤ -1/3) :=
by sorry

end problem_solution_l1473_147330


namespace acid_dilution_l1473_147306

theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 50 → 
  initial_concentration = 0.4 → 
  final_concentration = 0.25 → 
  water_added = 30 → 
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration := by
  sorry

#check acid_dilution

end acid_dilution_l1473_147306


namespace polynomial_transformation_c_values_l1473_147344

/-- The number of distinct possible values of c in a polynomial transformation. -/
theorem polynomial_transformation_c_values
  (a b r s t : ℂ)
  (h_distinct : r ≠ s ∧ s ≠ t ∧ r ≠ t)
  (h_transform : ∀ z, (z - r) * (z - s) * (z - t) =
                      ((a * z + b) - c * r) * ((a * z + b) - c * s) * ((a * z + b) - c * t)) :
  ∃! (values : Finset ℂ), values.card = 4 ∧ ∀ c, c ∈ values ↔ 
    ∃ z, (z - r) * (z - s) * (z - t) =
         ((a * z + b) - c * r) * ((a * z + b) - c * s) * ((a * z + b) - c * t) :=
by sorry

end polynomial_transformation_c_values_l1473_147344


namespace divisibility_of_quadratic_l1473_147323

theorem divisibility_of_quadratic (n : ℤ) : 
  (∀ n, ¬(8 ∣ (n^2 - 6*n - 2))) ∧ 
  (∀ n, ¬(9 ∣ (n^2 - 6*n - 2))) ∧ 
  (∀ n, (11 ∣ (n^2 - 6*n - 2)) ↔ (n ≡ 3 [ZMOD 11])) ∧ 
  (∀ n, ¬(121 ∣ (n^2 - 6*n - 2))) := by
  sorry

end divisibility_of_quadratic_l1473_147323


namespace division_problem_l1473_147301

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2/5) : 
  c / a = 5/6 := by sorry

end division_problem_l1473_147301


namespace chocolate_distribution_l1473_147373

theorem chocolate_distribution (minho : ℕ) (taemin kibum : ℕ) : 
  taemin = 5 * minho →
  kibum = 3 * minho →
  taemin + kibum = 160 →
  minho = 20 := by
sorry

end chocolate_distribution_l1473_147373


namespace division_simplification_l1473_147337

theorem division_simplification (a : ℝ) (h : a ≠ 0) :
  (a - 1/a) / ((a - 1)/a) = a + 1 := by
sorry

end division_simplification_l1473_147337


namespace smallest_number_with_remainders_l1473_147343

theorem smallest_number_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 3 = 2) ∧ 
  (n % 5 = 3) ∧ 
  (∀ m : ℕ, m > 0 → m % 3 = 2 → m % 5 = 3 → n ≤ m) ∧
  n = 8 := by
sorry

end smallest_number_with_remainders_l1473_147343


namespace intersection_line_equation_l1473_147331

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2 (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 20

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (A B : ℝ × ℝ),
  (circle1 A.1 A.2 ∧ circle2 A.1 A.2) →
  (circle1 B.1 B.2 ∧ circle2 B.1 B.2) →
  A ≠ B →
  line A.1 A.2 ∧ line B.1 B.2 :=
sorry

end intersection_line_equation_l1473_147331


namespace rhombus_sides_equal_l1473_147349

/-- A rhombus is a quadrilateral with all sides equal -/
structure Rhombus where
  sides : Fin 4 → ℝ
  is_quadrilateral : True
  all_sides_equal : ∀ (i j : Fin 4), sides i = sides j

/-- All four sides of a rhombus are equal -/
theorem rhombus_sides_equal (r : Rhombus) : 
  ∀ (i j : Fin 4), r.sides i = r.sides j := by
  sorry

end rhombus_sides_equal_l1473_147349


namespace odd_decreasing_function_properties_l1473_147382

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_decreasing : ∀ x y, x < y → f x > f y)

-- Define the theorem
theorem odd_decreasing_function_properties
  (a b : ℝ)
  (h_sum_neg : a + b < 0) :
  f (a + b) > 0 ∧ f a + f b > 0 := by
sorry


end odd_decreasing_function_properties_l1473_147382


namespace noah_in_middle_chair_l1473_147320

/- Define the friends as an enumeration -/
inductive Friend
| Liam
| Noah
| Olivia
| Emma
| Sophia

/- Define the seating arrangement as a function from chair number to Friend -/
def Seating := Fin 5 → Friend

def is_valid_seating (s : Seating) : Prop :=
  /- Sophia sits in the first chair -/
  s 1 = Friend.Sophia ∧
  /- Emma sits directly in front of Liam -/
  (∃ i : Fin 4, s i = Friend.Emma ∧ s (i + 1) = Friend.Liam) ∧
  /- Noah sits somewhere in front of Emma -/
  (∃ i j : Fin 5, i < j ∧ s i = Friend.Noah ∧ s j = Friend.Emma) ∧
  /- At least one person sits between Noah and Olivia -/
  (∃ i j k : Fin 5, i < j ∧ j < k ∧ s i = Friend.Noah ∧ s k = Friend.Olivia) ∧
  /- All friends are seated -/
  (∃ i : Fin 5, s i = Friend.Liam) ∧
  (∃ i : Fin 5, s i = Friend.Noah) ∧
  (∃ i : Fin 5, s i = Friend.Olivia) ∧
  (∃ i : Fin 5, s i = Friend.Emma) ∧
  (∃ i : Fin 5, s i = Friend.Sophia)

theorem noah_in_middle_chair (s : Seating) (h : is_valid_seating s) :
  s 3 = Friend.Noah :=
by sorry

end noah_in_middle_chair_l1473_147320


namespace proposition_truth_l1473_147340

theorem proposition_truth (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬¬q) : 
  (¬p ∧ q) := by
  sorry

end proposition_truth_l1473_147340


namespace max_eel_coverage_l1473_147380

/-- An eel is a polyomino formed by a path of unit squares which makes two turns in opposite directions -/
def Eel : Type := Unit

/-- A configuration of non-overlapping eels on a grid -/
def EelConfiguration (n : ℕ) : Type := Unit

/-- The area covered by a configuration of eels -/
def coveredArea (n : ℕ) (config : EelConfiguration n) : ℕ := sorry

theorem max_eel_coverage :
  ∃ (config : EelConfiguration 1000),
    coveredArea 1000 config = 999998 ∧
    ∀ (other_config : EelConfiguration 1000),
      coveredArea 1000 other_config ≤ 999998 := by sorry

end max_eel_coverage_l1473_147380


namespace red_peppers_weight_l1473_147318

/-- The weight of red peppers at Dale's Vegetarian Restaurant -/
def weight_red_peppers : ℝ := 5.666666666666667 - 2.8333333333333335

/-- Theorem: The weight of red peppers is equal to the total weight of peppers minus the weight of green peppers -/
theorem red_peppers_weight :
  weight_red_peppers = 5.666666666666667 - 2.8333333333333335 := by
  sorry

end red_peppers_weight_l1473_147318


namespace chemistry_books_count_l1473_147372

def number_of_biology_books : ℕ := 13
def total_combinations : ℕ := 2184

theorem chemistry_books_count (C : ℕ) : 
  (number_of_biology_books.choose 2) * (C.choose 2) = total_combinations → C = 8 := by
  sorry

end chemistry_books_count_l1473_147372


namespace parallel_vectors_t_value_l1473_147336

/-- Two vectors in ℝ² -/
def Vector2 := ℝ × ℝ

/-- Check if two vectors are parallel -/
def are_parallel (v w : Vector2) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_t_value :
  ∀ (t : ℝ),
  let a : Vector2 := (t, -6)
  let b : Vector2 := (-3, 2)
  are_parallel a b → t = 9 := by
sorry

end parallel_vectors_t_value_l1473_147336


namespace peter_bought_nine_kilos_of_tomatoes_l1473_147385

/-- Represents the purchase of groceries by Peter -/
structure Groceries where
  initialMoney : ℕ
  potatoPrice : ℕ
  potatoKilos : ℕ
  tomatoPrice : ℕ
  cucumberPrice : ℕ
  cucumberKilos : ℕ
  bananaPrice : ℕ
  bananaKilos : ℕ
  remainingMoney : ℕ

/-- Calculates the number of kilos of tomatoes bought -/
def tomatoKilos (g : Groceries) : ℕ :=
  (g.initialMoney - g.remainingMoney - 
   (g.potatoPrice * g.potatoKilos + 
    g.cucumberPrice * g.cucumberKilos + 
    g.bananaPrice * g.bananaKilos)) / g.tomatoPrice

/-- Theorem stating that Peter bought 9 kilos of tomatoes -/
theorem peter_bought_nine_kilos_of_tomatoes (g : Groceries) 
  (h1 : g.initialMoney = 500)
  (h2 : g.potatoPrice = 2)
  (h3 : g.potatoKilos = 6)
  (h4 : g.tomatoPrice = 3)
  (h5 : g.cucumberPrice = 4)
  (h6 : g.cucumberKilos = 5)
  (h7 : g.bananaPrice = 5)
  (h8 : g.bananaKilos = 3)
  (h9 : g.remainingMoney = 426) :
  tomatoKilos g = 9 := by
  sorry

end peter_bought_nine_kilos_of_tomatoes_l1473_147385


namespace clique_six_and_best_degree_l1473_147377

/-- A graph with 1991 points where every point has degree at least 1593 -/
structure Graph1991 where
  vertices : Finset (Fin 1991)
  edges : Finset (Fin 1991 × Fin 1991)
  degree_condition : ∀ v : Fin 1991, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≥ 1593

/-- A clique is a subset of vertices where every two distinct vertices are adjacent -/
def is_clique (G : Graph1991) (S : Finset (Fin 1991)) : Prop :=
  ∀ u v : Fin 1991, u ∈ S → v ∈ S → u ≠ v → (u, v) ∈ G.edges ∨ (v, u) ∈ G.edges

/-- The main theorem stating that there exists a clique of size 6 and 1593 is the best possible -/
theorem clique_six_and_best_degree (G : Graph1991) :
  (∃ S : Finset (Fin 1991), S.card = 6 ∧ is_clique G S) ∧
  ∀ d < 1593, ∃ H : Graph1991, ¬∃ S : Finset (Fin 1991), S.card = 6 ∧ is_clique H S :=
sorry

end clique_six_and_best_degree_l1473_147377


namespace min_vertices_for_quadrilateral_l1473_147319

theorem min_vertices_for_quadrilateral (n : ℕ) (hn : n ≥ 10) :
  let k := ⌊(3 * n : ℝ) / 4⌋ + 1
  ∀ S : Finset (Fin n),
    S.card ≥ k →
    ∃ (a b c d : Fin n), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
      ((b - a) % n = 1 ∨ (b - a) % n = n - 1) ∧
      ((c - b) % n = 1 ∨ (c - b) % n = n - 1) ∧
      ((d - c) % n = 1 ∨ (d - c) % n = n - 1) :=
by sorry

#check min_vertices_for_quadrilateral

end min_vertices_for_quadrilateral_l1473_147319


namespace base7_to_base10_conversion_l1473_147304

-- Define a function to convert a base 7 number to base 10
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

-- Define the given base 7 number
def base7Number : List Nat := [1, 2, 3, 5, 4]

-- Theorem to prove
theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 11481 := by
  sorry

end base7_to_base10_conversion_l1473_147304


namespace prob_girl_from_E_expected_value_X_l1473_147388

/-- Represents a family with a number of boys and girls -/
structure Family :=
  (boys : Nat)
  (girls : Nat)

/-- The set of all families -/
def Families : Finset (Fin 5) := Finset.univ

/-- The number of boys and girls in each family -/
def familyData : Fin 5 → Family
  | ⟨0, _⟩ => ⟨0, 0⟩  -- Family A
  | ⟨1, _⟩ => ⟨1, 0⟩  -- Family B
  | ⟨2, _⟩ => ⟨0, 1⟩  -- Family C
  | ⟨3, _⟩ => ⟨1, 1⟩  -- Family D
  | ⟨4, _⟩ => ⟨1, 2⟩  -- Family E

/-- The total number of children -/
def totalChildren : Nat := Finset.sum Families (λ i => (familyData i).boys + (familyData i).girls)

/-- The total number of girls -/
def totalGirls : Nat := Finset.sum Families (λ i => (familyData i).girls)

/-- Probability of selecting a girl from family E given that a girl is selected -/
theorem prob_girl_from_E : 
  (familyData 4).girls / totalGirls = 1 / 2 := by sorry

/-- Probability distribution of X when selecting 3 families -/
def probDistX (x : Fin 3) : Rat :=
  match x with
  | ⟨0, _⟩ => 1 / 10
  | ⟨1, _⟩ => 3 / 5
  | ⟨2, _⟩ => 3 / 10

/-- Expected value of X -/
def expectedX : Rat := Finset.sum (Finset.range 3) (λ i => i * probDistX i)

theorem expected_value_X : expectedX = 6 / 5 := by sorry

end prob_girl_from_E_expected_value_X_l1473_147388


namespace pizza_fraction_l1473_147310

theorem pizza_fraction (total_slices : ℕ) (whole_slices : ℕ) (shared_slice_fraction : ℚ) :
  total_slices = 16 →
  whole_slices = 2 →
  shared_slice_fraction = 1/6 →
  (whole_slices : ℚ) / total_slices + shared_slice_fraction / total_slices = 13/96 := by
  sorry

end pizza_fraction_l1473_147310


namespace crayon_selection_theorem_l1473_147324

def total_crayons : ℕ := 15
def red_crayons : ℕ := 3
def non_red_crayons : ℕ := total_crayons - red_crayons
def selection_size : ℕ := 5

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem crayon_selection_theorem :
  choose total_crayons selection_size - choose non_red_crayons selection_size = 2211 :=
by sorry

end crayon_selection_theorem_l1473_147324


namespace investment_calculation_l1473_147368

/-- Given a total investment split between a savings account and mutual funds,
    where the investment in mutual funds is 6 times the investment in the savings account,
    calculate the total investment in mutual funds. -/
theorem investment_calculation (total : ℝ) (savings : ℝ) (mutual_funds : ℝ)
    (h1 : total = 320000)
    (h2 : mutual_funds = 6 * savings)
    (h3 : total = savings + mutual_funds) :
  mutual_funds = 274285.74 := by
  sorry

end investment_calculation_l1473_147368
