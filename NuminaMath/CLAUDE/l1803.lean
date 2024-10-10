import Mathlib

namespace power_of_two_equality_l1803_180317

theorem power_of_two_equality : ∃ x : ℕ, 8^12 + 8^12 + 8^12 = 2^x ∧ x = 38 := by
  sorry

end power_of_two_equality_l1803_180317


namespace ceiling_minus_fractional_part_l1803_180364

theorem ceiling_minus_fractional_part (x : ℝ) : ⌈x⌉ - (x - ⌊x⌋) = 1 := by
  sorry

end ceiling_minus_fractional_part_l1803_180364


namespace inequality_system_solution_l1803_180338

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (2*x - a < 1 ∧ x - 2*b > 3) ↔ (-1 < x ∧ x < 1)) →
  (a + 1) * (b - 1) = -6 := by
  sorry

end inequality_system_solution_l1803_180338


namespace exist_valid_subgrid_l1803_180368

/-- Represents a grid of 0s and 1s -/
def Grid := Matrix (Fin 100) (Fin 2018) Bool

/-- A predicate that checks if a grid satisfies the condition of having at least 75 ones in each column -/
def ValidGrid (g : Grid) : Prop :=
  ∀ j : Fin 2018, (Finset.filter (fun i => g i j) Finset.univ).card ≥ 75

/-- A predicate that checks if a 5-row subgrid has at most one all-zero column -/
def ValidSubgrid (g : Grid) (rows : Finset (Fin 100)) : Prop :=
  rows.card = 5 ∧
  (Finset.filter (fun j : Fin 2018 => ∀ i ∈ rows, ¬g i j) Finset.univ).card ≤ 1

/-- The main theorem to be proved -/
theorem exist_valid_subgrid (g : Grid) (h : ValidGrid g) :
  ∃ rows : Finset (Fin 100), ValidSubgrid g rows := by
  sorry

end exist_valid_subgrid_l1803_180368


namespace total_distance_is_410_l1803_180399

-- Define bird types and their speeds
structure Bird where
  name : String
  speed : ℝ
  flightTime : ℝ

-- Define constants
def headwind : ℝ := 5
def totalBirds : ℕ := 6

-- Define the list of birds
def birds : List Bird := [
  { name := "eagle", speed := 15, flightTime := 2.5 },
  { name := "falcon", speed := 46, flightTime := 2.5 },
  { name := "pelican", speed := 33, flightTime := 2.5 },
  { name := "hummingbird", speed := 30, flightTime := 2.5 },
  { name := "hawk", speed := 45, flightTime := 3 },
  { name := "swallow", speed := 25, flightTime := 1.5 }
]

-- Calculate actual distance traveled by a bird
def actualDistance (bird : Bird) : ℝ :=
  (bird.speed - headwind) * bird.flightTime

-- Calculate total distance traveled by all birds
def totalDistance : ℝ :=
  (birds.map actualDistance).sum

-- Theorem to prove
theorem total_distance_is_410 : totalDistance = 410 := by
  sorry

end total_distance_is_410_l1803_180399


namespace ratio_of_distances_l1803_180397

/-- Given five consecutive points on a line, prove the ratio of two specific distances -/
theorem ratio_of_distances (E F G H I : ℝ) (hEF : |E - F| = 3) (hFG : |F - G| = 6) 
  (hGH : |G - H| = 4) (hHI : |H - I| = 2) (hOrder : E < F ∧ F < G ∧ G < H ∧ H < I) : 
  |E - G| / |H - I| = 9/2 := by
  sorry

end ratio_of_distances_l1803_180397


namespace quadratic_properties_l1803_180375

def quadratic_function (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

theorem quadratic_properties (a h k : ℝ) :
  quadratic_function a h k (-2) = 0 →
  quadratic_function a h k 4 = 0 →
  quadratic_function a h k 1 = -9/2 →
  (a = 1/2 ∧ h = 1 ∧ k = -9/2 ∧ ∀ x, quadratic_function a h k x ≥ -9/2) :=
by sorry

end quadratic_properties_l1803_180375


namespace aquarium_fish_count_l1803_180334

def aquarium (num_stingrays : ℕ) : ℕ → Prop :=
  fun total_fish =>
    ∃ num_sharks : ℕ,
      num_sharks = 2 * num_stingrays ∧
      total_fish = num_sharks + num_stingrays

theorem aquarium_fish_count : aquarium 28 84 := by
  sorry

end aquarium_fish_count_l1803_180334


namespace stamp_collection_duration_l1803_180341

/-- Proves the collection duration for two stamp collectors given their collection rates and total stamps --/
theorem stamp_collection_duration (total_stamps : ℕ) (rate1 rate2 : ℕ) (extra_weeks : ℕ) : 
  total_stamps = 300 →
  rate1 = 5 →
  rate2 = 3 →
  extra_weeks = 20 →
  ∃ (weeks1 weeks2 : ℕ), 
    weeks1 = 30 ∧
    weeks2 = 50 ∧
    weeks2 = weeks1 + extra_weeks ∧
    total_stamps = rate1 * weeks1 + rate2 * weeks2 :=
by sorry


end stamp_collection_duration_l1803_180341


namespace geometric_progression_sum_ratio_l1803_180301

theorem geometric_progression_sum_ratio (a : ℝ) (n : ℕ) : 
  let r : ℝ := 3
  let S_n := a * (1 - r^n) / (1 - r)
  let S_3 := a * (1 - r^3) / (1 - r)
  S_n / S_3 = 28 → n = 6 := by
sorry

end geometric_progression_sum_ratio_l1803_180301


namespace quincy_age_l1803_180388

/-- Given the ages of several people and their relationships, calculate Quincy's age -/
theorem quincy_age (kiarra bea job figaro quincy : ℝ) : 
  kiarra = 2 * bea →
  job = 3 * bea →
  figaro = job + 7 →
  kiarra = 30 →
  quincy = (job + figaro) / 2 →
  quincy = 48.5 := by
sorry

end quincy_age_l1803_180388


namespace equation_solution_l1803_180380

theorem equation_solution : ∃ x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) := by
  sorry

end equation_solution_l1803_180380


namespace sqrt_equation_solution_l1803_180379

theorem sqrt_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt x + Real.sqrt (x + 4) = 12 ∧ x = 1225 / 36 := by
  sorry

end sqrt_equation_solution_l1803_180379


namespace union_equals_A_l1803_180314

def A : Set ℤ := {-1, 0, 1}
def B (a : ℤ) : Set ℤ := {a, a^2}

theorem union_equals_A (a : ℤ) : A ∪ B a = A ↔ a = -1 := by
  sorry

end union_equals_A_l1803_180314


namespace prime_power_sum_condition_l1803_180340

theorem prime_power_sum_condition (n : ℕ) :
  Nat.Prime (2^n + n^2016) ↔ n = 1 := by
  sorry

end prime_power_sum_condition_l1803_180340


namespace unique_expansion_terms_l1803_180331

def expansion_terms (N : ℕ) : ℕ := Nat.choose N 5

theorem unique_expansion_terms : 
  ∃! N : ℕ, N > 0 ∧ expansion_terms N = 231 :=
by sorry

end unique_expansion_terms_l1803_180331


namespace total_paths_XZ_l1803_180394

-- Define the number of paths between points
def paths_XY : ℕ := 2
def paths_YZ : ℕ := 2
def direct_paths_XZ : ℕ := 2

-- Theorem statement
theorem total_paths_XZ : paths_XY * paths_YZ + direct_paths_XZ = 6 := by
  sorry

end total_paths_XZ_l1803_180394


namespace parabolas_intersection_l1803_180384

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ :=
  {x | x = 2 - Real.sqrt 26 ∨ x = 2 + Real.sqrt 26}

/-- The y-coordinate of the intersection points of two parabolas -/
def intersection_y : ℝ := 48

/-- The first parabola function -/
def f (x : ℝ) : ℝ := 3 * x^2 - 12 * x - 18

/-- The second parabola function -/
def g (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 4

theorem parabolas_intersection :
  ∀ x y : ℝ, (f x = y ∧ g x = y) ↔ (x ∈ intersection_x ∧ y = intersection_y) :=
sorry

end parabolas_intersection_l1803_180384


namespace sum_pascal_row_21st_triangular_l1803_180363

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of entries in the n-th row of Pascal's triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^n

theorem sum_pascal_row_21st_triangular : 
  pascal_row_sum (triangular_number 21 - 1) = 2^230 := by sorry

end sum_pascal_row_21st_triangular_l1803_180363


namespace f_has_minimum_at_one_point_five_l1803_180362

def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 2

theorem f_has_minimum_at_one_point_five :
  ∃ (y : ℝ), ∀ (x : ℝ), f x ≥ f (3/2) := by
  sorry

end f_has_minimum_at_one_point_five_l1803_180362


namespace mode_of_shoe_sizes_l1803_180308

def shoe_sizes : List ℝ := [24, 24.5, 25, 25.5, 26]
def sales : List ℕ := [2, 5, 3, 6, 4]

def mode (sizes : List ℝ) (counts : List ℕ) : ℝ :=
  let pairs := List.zip sizes counts
  let max_count := pairs.map Prod.snd |>.maximum?
  match max_count with
  | none => 0  -- Default value if the list is empty
  | some mc => (pairs.filter (fun p => p.2 = mc)).head!.1

theorem mode_of_shoe_sizes :
  mode shoe_sizes sales = 25.5 := by
  sorry

end mode_of_shoe_sizes_l1803_180308


namespace initial_erasers_count_l1803_180343

/-- The number of scissors initially in the drawer -/
def initial_scissors : ℕ := 118

/-- The number of erasers Jason placed in the drawer -/
def erasers_added : ℕ := 131

/-- The total number of erasers after Jason added some -/
def total_erasers : ℕ := 270

/-- The initial number of erasers in the drawer -/
def initial_erasers : ℕ := total_erasers - erasers_added

theorem initial_erasers_count : initial_erasers = 139 := by
  sorry

end initial_erasers_count_l1803_180343


namespace simplify_square_roots_l1803_180324

theorem simplify_square_roots : 
  Real.sqrt 726 / Real.sqrt 242 + Real.sqrt 484 / Real.sqrt 121 = Real.sqrt 3 + 2 := by
  sorry

end simplify_square_roots_l1803_180324


namespace pie_slices_remaining_l1803_180396

theorem pie_slices_remaining (total_slices : ℕ) 
  (joe_fraction darcy_fraction carl_fraction emily_fraction : ℚ) : 
  total_slices = 24 →
  joe_fraction = 1/3 →
  darcy_fraction = 1/4 →
  carl_fraction = 1/6 →
  emily_fraction = 1/8 →
  total_slices - (total_slices * joe_fraction + total_slices * darcy_fraction + 
    total_slices * carl_fraction + total_slices * emily_fraction) = 3 := by
  sorry

end pie_slices_remaining_l1803_180396


namespace projection_a_onto_b_l1803_180335

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-4, 7)

theorem projection_a_onto_b :
  let proj := (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2)
  proj = Real.sqrt 65 / 5 := by sorry

end projection_a_onto_b_l1803_180335


namespace solve_system_of_equations_l1803_180336

theorem solve_system_of_equations (a b : ℝ) 
  (eq1 : 3 * a + 2 * b = 18) 
  (eq2 : 5 * a + 4 * b = 31) : 
  2 * a + b = 11.5 := by
sorry

end solve_system_of_equations_l1803_180336


namespace consecutive_digits_pattern_l1803_180325

def consecutive_digits (n : Nat) : Nat :=
  if n = 0 then 0 else
  let rec aux (k : Nat) (acc : Nat) : Nat :=
    if k = 0 then acc else aux (k - 1) (acc * 10 + k)
  aux n 0

def reverse_consecutive_digits (n : Nat) : Nat :=
  if n = 0 then 0 else
  let rec aux (k : Nat) (acc : Nat) : Nat :=
    if k = 0 then acc else aux (k - 1) (acc * 10 + (10 - k))
  aux n 0

theorem consecutive_digits_pattern (n : Nat) (h : n > 0 ∧ n ≤ 9) :
  consecutive_digits n * 8 + n = reverse_consecutive_digits n := by
  sorry

end consecutive_digits_pattern_l1803_180325


namespace max_value_theorem_l1803_180367

/-- A function satisfying the given recurrence relation -/
def RecurrenceFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1) = 1 + Real.sqrt (2 * f x - f x ^ 2)

/-- The theorem stating the maximum value of f(1) + f(2020) -/
theorem max_value_theorem (f : ℝ → ℝ) (h : RecurrenceFunction f) :
    ∃ M : ℝ, M = 2 + Real.sqrt 2 ∧ f 1 + f 2020 ≤ M ∧ 
    ∃ g : ℝ → ℝ, RecurrenceFunction g ∧ g 1 + g 2020 = M :=
  sorry

end max_value_theorem_l1803_180367


namespace consecutive_squares_sum_l1803_180381

theorem consecutive_squares_sum (n : ℤ) : 
  n^2 + (n + 1)^2 = 452 → n + 1 = 15 := by
  sorry

end consecutive_squares_sum_l1803_180381


namespace percentage_same_grade_l1803_180361

/-- Represents the grade a student can receive -/
inductive Grade
| A
| B
| C
| D
| E

/-- Represents the grade distribution for a single test -/
structure GradeDistribution :=
  (A : Nat)
  (B : Nat)
  (C : Nat)
  (D : Nat)
  (E : Nat)

/-- The total number of students in the class -/
def totalStudents : Nat := 50

/-- The grade distribution for the first test -/
def firstTestDistribution : GradeDistribution := {
  A := 7,
  B := 12,
  C := 19,
  D := 8,
  E := 4
}

/-- The grade distribution for the second test -/
def secondTestDistribution : GradeDistribution := {
  A := 8,
  B := 16,
  C := 14,
  D := 7,
  E := 5
}

/-- The number of students who received the same grade on both tests -/
def sameGradeCount : Nat := 20

/-- Theorem: The percentage of students who received the same grade on both tests is 40% -/
theorem percentage_same_grade :
  (sameGradeCount : ℚ) / (totalStudents : ℚ) * 100 = 40 := by sorry

end percentage_same_grade_l1803_180361


namespace exists_tricolor_right_triangle_l1803_180376

/-- A color type with three possible values -/
inductive Color
| Red
| Green
| Blue

/-- A point in the plane with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each point -/
def Coloring := Point → Color

/-- Predicate to check if a triangle is right-angled -/
def isRightTriangle (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p3.x - p2.x)^2 + (p3.y - p2.y)^2 = 
  (p3.x - p1.x)^2 + (p3.y - p1.y)^2

/-- Main theorem -/
theorem exists_tricolor_right_triangle (coloring : Coloring) 
  (h1 : ∃ p : Point, coloring p = Color.Red)
  (h2 : ∃ p : Point, coloring p = Color.Green)
  (h3 : ∃ p : Point, coloring p = Color.Blue) :
  ∃ p1 p2 p3 : Point, 
    isRightTriangle p1 p2 p3 ∧ 
    coloring p1 ≠ coloring p2 ∧ 
    coloring p2 ≠ coloring p3 ∧ 
    coloring p3 ≠ coloring p1 :=
sorry

end exists_tricolor_right_triangle_l1803_180376


namespace investment_problem_l1803_180351

theorem investment_problem (T : ℝ) :
  (0.10 * (T - 700) - 0.08 * 700 = 74) →
  T = 2000 := by
sorry

end investment_problem_l1803_180351


namespace circle_center_polar_coordinates_l1803_180333

theorem circle_center_polar_coordinates :
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + (y - 1)^2 = 1}
  let center := (1, 1)
  let r := Real.sqrt 2
  let θ := Real.pi / 4
  (center ∈ circle) ∧
  (r * Real.cos θ = center.1 - 0) ∧
  (r * Real.sin θ = center.2 - 0) :=
by sorry

end circle_center_polar_coordinates_l1803_180333


namespace bus_ticket_probability_l1803_180366

/-- Represents the lottery game with given parameters -/
structure LotteryGame where
  initialAmount : ℝ
  ticketCost : ℝ
  winProbability : ℝ
  prizeAmount : ℝ
  targetAmount : ℝ

/-- Calculates the probability of winning enough money to reach the target amount -/
noncomputable def winProbability (game : LotteryGame) : ℝ :=
  let p := game.winProbability
  let q := 1 - p
  (p^2 * (1 + 2*q)) / (1 - 2*p*q^2)

/-- Theorem stating the probability of winning the bus ticket -/
theorem bus_ticket_probability (game : LotteryGame) 
  (h1 : game.initialAmount = 20)
  (h2 : game.ticketCost = 10)
  (h3 : game.winProbability = 0.1)
  (h4 : game.prizeAmount = 30)
  (h5 : game.targetAmount = 45) :
  ∃ ε > 0, |winProbability game - 0.033| < ε :=
sorry

end bus_ticket_probability_l1803_180366


namespace lily_break_time_l1803_180391

/-- Represents Lily's typing scenario -/
structure TypingScenario where
  words_per_minute : ℕ
  minutes_before_break : ℕ
  total_minutes : ℕ
  total_words : ℕ

/-- Calculates the break time in minutes for a given typing scenario -/
def calculate_break_time (scenario : TypingScenario) : ℕ :=
  sorry

/-- Theorem stating that Lily's break time is 2 minutes -/
theorem lily_break_time :
  let lily_scenario : TypingScenario := {
    words_per_minute := 15,
    minutes_before_break := 10,
    total_minutes := 19,
    total_words := 255
  }
  calculate_break_time lily_scenario = 2 :=
by sorry

end lily_break_time_l1803_180391


namespace negative_two_squared_l1803_180339

theorem negative_two_squared : -2^2 = -4 := by
  sorry

end negative_two_squared_l1803_180339


namespace jack_total_miles_driven_l1803_180320

/-- Calculates the total miles driven given the number of years and miles driven per four-month period -/
def total_miles_driven (years : ℕ) (miles_per_period : ℕ) : ℕ :=
  let months : ℕ := years * 12
  let periods : ℕ := months / 4
  periods * miles_per_period

/-- Proves that given 9 years of driving and 37,000 miles driven every four months, the total miles driven is 999,000 -/
theorem jack_total_miles_driven :
  total_miles_driven 9 37000 = 999000 := by
  sorry

#eval total_miles_driven 9 37000

end jack_total_miles_driven_l1803_180320


namespace distinct_collections_count_l1803_180309

/-- Represents the number of occurrences of each letter in "CALCULATIONS" --/
def letterCounts : Fin 26 → ℕ
| 0 => 3  -- A
| 2 => 3  -- C
| 8 => 1  -- I
| 11 => 3 -- L
| 13 => 1 -- N
| 14 => 1 -- O
| 18 => 1 -- S
| 19 => 1 -- T
| 20 => 1 -- U
| _ => 0

/-- Predicate for vowels --/
def isVowel (n : Fin 26) : Bool :=
  n = 0 ∨ n = 4 ∨ n = 8 ∨ n = 14 ∨ n = 20

/-- The number of distinct collections of three vowels and three consonants --/
def distinctCollections : ℕ := sorry

/-- Theorem stating that the number of distinct collections is 126 --/
theorem distinct_collections_count :
  distinctCollections = 126 := by sorry

end distinct_collections_count_l1803_180309


namespace integer_pair_theorem_l1803_180329

/-- Given positive integers a and b where a > b, prove that a²b - ab² = 30 
    if and only if (a, b) is one of (5, 2), (5, 3), (6, 1), or (6, 5) -/
theorem integer_pair_theorem (a b : ℕ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) :
  a^2 * b - a * b^2 = 30 ↔ 
  ((a = 5 ∧ b = 2) ∨ (a = 5 ∧ b = 3) ∨ (a = 6 ∧ b = 1) ∨ (a = 6 ∧ b = 5)) :=
by sorry

end integer_pair_theorem_l1803_180329


namespace range_of_H_l1803_180315

-- Define the function H
def H (x : ℝ) : ℝ := |x + 3| - |x - 2|

-- State the theorem about the range of H
theorem range_of_H :
  Set.range H = Set.Iic 5 :=
sorry

end range_of_H_l1803_180315


namespace leon_order_proof_l1803_180322

def toy_organizer_sets : ℕ := 3
def toy_organizer_price : ℚ := 78
def gaming_chair_price : ℚ := 83
def delivery_fee_rate : ℚ := 0.05
def total_payment : ℚ := 420

def gaming_chairs_ordered : ℕ := 2

theorem leon_order_proof :
  ∃ (g : ℕ), 
    (toy_organizer_sets * toy_organizer_price + g * gaming_chair_price) * (1 + delivery_fee_rate) = total_payment ∧
    g = gaming_chairs_ordered :=
by sorry

end leon_order_proof_l1803_180322


namespace sector_area_l1803_180342

/-- Given a circular sector with central angle 3 radians and perimeter 5, its area is 3/2. -/
theorem sector_area (θ : Real) (p : Real) (S : Real) : 
  θ = 3 → p = 5 → S = (θ * (p - θ)) / (2 * (2 + θ)) → S = 3/2 := by sorry

end sector_area_l1803_180342


namespace new_average_after_modification_l1803_180377

def consecutive_integers (start : ℤ) : List ℤ :=
  List.range 10 |>.map (λ i => start + i)

def modified_sequence (start : ℤ) : List ℤ :=
  List.range 10 |>.map (λ i => start + i - (9 - i))

theorem new_average_after_modification (start : ℤ) :
  (consecutive_integers start).sum / 10 = 20 →
  (modified_sequence start).sum / 10 = 15 := by
  sorry

end new_average_after_modification_l1803_180377


namespace intersection_point_of_lines_l1803_180323

/-- The intersection point of two lines with given angles of inclination -/
theorem intersection_point_of_lines (m n : ℝ → ℝ) (k₁ k₂ : ℝ) :
  (∀ x, m x = k₁ * x + 2) →
  (∀ x, n x = k₂ * x + Real.sqrt 3 + 1) →
  k₁ = Real.tan (π / 4) →
  k₂ = Real.tan (π / 3) →
  ∃ x y, m x = n x ∧ m x = y ∧ x = -1 ∧ y = 1 := by
  sorry

end intersection_point_of_lines_l1803_180323


namespace inverse_proportion_problem_l1803_180346

/-- Given that α is inversely proportional to β, prove that when α = 5 and β = 20, 
    then α = 10 when β = 10 -/
theorem inverse_proportion_problem (α β : ℝ) (k : ℝ) 
    (h1 : α * β = k)  -- α is inversely proportional to β
    (h2 : 5 * 20 = k) -- α = 5 when β = 20
    : 10 * 10 = k :=  -- α = 10 when β = 10
  sorry

end inverse_proportion_problem_l1803_180346


namespace h_greater_than_two_l1803_180371

theorem h_greater_than_two (x : ℝ) (hx : x > 0) : Real.exp x - Real.log x > 2 := by
  sorry

end h_greater_than_two_l1803_180371


namespace problem_1_problem_2_problem_3_problem_4_l1803_180344

-- Problem 1
theorem problem_1 : -3 - (-10) + (-9) - 10 = -12 := by sorry

-- Problem 2
theorem problem_2 : (1/4 : ℚ) + (-1/8) + (-7/8) - (3/4) = -3/2 := by sorry

-- Problem 3
theorem problem_3 : -25 * (-18) + (-25) * 12 + 25 * (-10) = -100 := by sorry

-- Problem 4
theorem problem_4 : -48 * (-1/6 + 3/4 - 1/24) = -26 := by sorry

end problem_1_problem_2_problem_3_problem_4_l1803_180344


namespace complex_division_result_l1803_180387

theorem complex_division_result : (1 + 2*I : ℂ) / I = 2 - I := by sorry

end complex_division_result_l1803_180387


namespace seeds_per_medium_row_is_twenty_l1803_180348

/-- Represents the garden setup with large and medium beds -/
structure GardenSetup where
  largeBeds : Nat
  mediumBeds : Nat
  largeRowsPerBed : Nat
  mediumRowsPerBed : Nat
  seedsPerLargeRow : Nat
  totalSeeds : Nat

/-- Calculates the number of seeds per row in the medium bed -/
def seedsPerMediumRow (setup : GardenSetup) : Nat :=
  let largeSeeds := setup.largeBeds * setup.largeRowsPerBed * setup.seedsPerLargeRow
  let mediumSeeds := setup.totalSeeds - largeSeeds
  let totalMediumRows := setup.mediumBeds * setup.mediumRowsPerBed
  mediumSeeds / totalMediumRows

/-- Theorem stating that the number of seeds per row in the medium bed is 20 -/
theorem seeds_per_medium_row_is_twenty :
  let setup : GardenSetup := {
    largeBeds := 2,
    mediumBeds := 2,
    largeRowsPerBed := 4,
    mediumRowsPerBed := 3,
    seedsPerLargeRow := 25,
    totalSeeds := 320
  }
  seedsPerMediumRow setup = 20 := by sorry

end seeds_per_medium_row_is_twenty_l1803_180348


namespace blue_shirt_percentage_l1803_180318

/-- Proves that the percentage of students wearing blue shirts is 44% -/
theorem blue_shirt_percentage
  (total_students : ℕ)
  (red_shirt_percentage : ℚ)
  (green_shirt_percentage : ℚ)
  (other_colors_count : ℕ)
  (h_total : total_students = 900)
  (h_red : red_shirt_percentage = 28/100)
  (h_green : green_shirt_percentage = 10/100)
  (h_other : other_colors_count = 162) :
  (total_students : ℚ) - (red_shirt_percentage + green_shirt_percentage + (other_colors_count : ℚ) / total_students) * total_students = 44/100 * total_students :=
by sorry

end blue_shirt_percentage_l1803_180318


namespace ruble_payment_l1803_180389

theorem ruble_payment (n : ℕ) (h : n > 7) : ∃ x y : ℕ, 3 * x + 5 * y = n := by
  sorry

end ruble_payment_l1803_180389


namespace cone_height_l1803_180311

theorem cone_height (r : ℝ) (h : ℝ) :
  r = 1 →
  (2 * Real.pi * r = (2 * Real.pi / 3) * 3) →
  h = Real.sqrt (3^2 - r^2) →
  h = 2 * Real.sqrt 2 :=
by sorry

end cone_height_l1803_180311


namespace smallest_number_l1803_180330

/-- Converts a number from base 6 to decimal -/
def base6ToDecimal (n : Nat) : Nat :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number from base 4 to decimal -/
def base4ToDecimal (n : Nat) : Nat :=
  (n / 1000) * 64 + ((n / 100) % 10) * 16 + ((n / 10) % 10) * 4 + (n % 10)

/-- Converts a number from base 2 to decimal -/
def base2ToDecimal (n : Nat) : Nat :=
  (n / 100000) * 32 + ((n / 10000) % 10) * 16 + ((n / 1000) % 10) * 8 +
  ((n / 100) % 10) * 4 + ((n / 10) % 10) * 2 + (n % 10)

theorem smallest_number (n1 n2 n3 : Nat) 
  (h1 : n1 = 210)
  (h2 : n2 = 1000)
  (h3 : n3 = 111111) :
  base2ToDecimal n3 < base6ToDecimal n1 ∧ base2ToDecimal n3 < base4ToDecimal n2 := by
  sorry

end smallest_number_l1803_180330


namespace first_expression_value_l1803_180302

theorem first_expression_value (E a : ℝ) (h1 : (E + (3 * a - 8)) / 2 = 89) (h2 : a = 34) : E = 84 := by
  sorry

end first_expression_value_l1803_180302


namespace expression_value_l1803_180378

theorem expression_value (a b c d m : ℝ) : 
  (a = -b) → (c * d = 1) → (abs m = 2) → 
  (3 * (a + b - 1) + (-c * d) ^ 2023 - 2 * m = -8 ∨ 
   3 * (a + b - 1) + (-c * d) ^ 2023 - 2 * m = 0) := by
sorry

end expression_value_l1803_180378


namespace root_range_l1803_180365

/-- Given that the equation |x-k| = (√2/2)k√x has two unequal real roots in the interval [k-1, k+1], prove that the range of k is 0 < k ≤ 1. -/
theorem root_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    k - 1 ≤ x₁ ∧ x₁ ≤ k + 1 ∧
    k - 1 ≤ x₂ ∧ x₂ ≤ k + 1 ∧
    |x₁ - k| = (Real.sqrt 2 / 2) * k * Real.sqrt x₁ ∧
    |x₂ - k| = (Real.sqrt 2 / 2) * k * Real.sqrt x₂) →
  0 < k ∧ k ≤ 1 :=
by sorry

end root_range_l1803_180365


namespace book_distribution_theorem_l1803_180373

/-- The number of ways to select k items from n distinct items, where order matters. -/
def permutations (n k : ℕ) : ℕ := (n.factorial) / ((n - k).factorial)

/-- The number of ways to select 2 books from 5 different books and give them to 2 students. -/
def book_distribution_ways : ℕ := permutations 5 2

theorem book_distribution_theorem : book_distribution_ways = 20 := by
  sorry

end book_distribution_theorem_l1803_180373


namespace complement_of_A_l1803_180386

def U : Set ℤ := {-2, -1, 0, 1, 2}

def A : Set ℤ := {x : ℤ | x^2 < 3}

theorem complement_of_A : 
  (U \ A) = {-2, 2} := by sorry

end complement_of_A_l1803_180386


namespace intersection_of_A_and_B_l1803_180345

def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def B : Set ℝ := {-2, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end intersection_of_A_and_B_l1803_180345


namespace decreasing_function_characterization_l1803_180332

open Set

/-- A function f is decreasing on an open interval (a, b) if for all x₁, x₂ in (a, b),
    x₁ < x₂ implies f(x₁) > f(x₂) -/
def DecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ Ioo a b → x₂ ∈ Ioo a b → x₁ < x₂ → f x₁ > f x₂

theorem decreasing_function_characterization
  (f : ℝ → ℝ) (a b : ℝ) (h : a < b)
  (h_domain : ∀ x, x ∈ Ioo a b → f x ∈ range f)
  (h_inequality : ∀ x₁ x₂, x₁ ∈ Ioo a b → x₂ ∈ Ioo a b →
    (x₁ - x₂) * (f x₁ - f x₂) < 0) :
  DecreasingOn f a b := by
  sorry

end decreasing_function_characterization_l1803_180332


namespace cubic_factorization_l1803_180357

theorem cubic_factorization (a : ℝ) : a^3 - 3*a = a*(a + Real.sqrt 3)*(a - Real.sqrt 3) := by
  sorry

end cubic_factorization_l1803_180357


namespace triangle_abc_properties_l1803_180316

noncomputable def triangle_abc (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a - b) * (Real.sin A + Real.sin B) = c * (Real.sin A - Real.sin C) ∧
  b = 2 ∧
  a = 2 * Real.sqrt 6 / 3

theorem triangle_abc_properties {a b c A B C : ℝ} 
  (h : triangle_abc a b c A B C) : 
  (∃ (R : ℝ), 2 * R = 4 * Real.sqrt 3 / 3) ∧ 
  (∃ (area : ℝ), area = Real.sqrt 3 / 3 + 1) :=
sorry

end triangle_abc_properties_l1803_180316


namespace mileage_scientific_notation_equality_l1803_180369

-- Define the original mileage
def original_mileage : ℝ := 42000

-- Define the scientific notation representation
def scientific_notation : ℝ := 4.2 * (10^4)

-- Theorem to prove the equality
theorem mileage_scientific_notation_equality :
  original_mileage = scientific_notation :=
by sorry

end mileage_scientific_notation_equality_l1803_180369


namespace sum_of_roots_greater_than_two_l1803_180355

theorem sum_of_roots_greater_than_two (x₁ x₂ : ℝ) 
  (h₁ : 5 * x₁^3 - 6 = 0) 
  (h₂ : 6 * x₂^3 - 5 = 0) : 
  x₁ + x₂ > 2 := by
sorry

end sum_of_roots_greater_than_two_l1803_180355


namespace y_value_l1803_180307

theorem y_value (y : ℚ) (h : (1 / 4) - (1 / 6) = 2 / y) : y = 24 := by
  sorry

end y_value_l1803_180307


namespace total_interest_compound_linh_investment_interest_l1803_180350

/-- Calculate the total interest earned on an investment with compound interest -/
theorem total_interest_compound (P : ℝ) (r : ℝ) (n : ℕ) :
  let A := P * (1 + r) ^ n
  A - P = P * ((1 + r) ^ n - 1) := by
  sorry

/-- Prove the total interest earned for Linh's investment -/
theorem linh_investment_interest :
  let P : ℝ := 1200  -- Initial investment
  let r : ℝ := 0.08  -- Annual interest rate
  let n : ℕ := 4     -- Number of years
  let A := P * (1 + r) ^ n
  A - P = 1200 * ((1 + 0.08) ^ 4 - 1) := by
  sorry

end total_interest_compound_linh_investment_interest_l1803_180350


namespace quadratic_roots_l1803_180312

/-- Given a quadratic function f(x) = ax² - 2ax + c where a ≠ 0,
    if f(3) = 0, then the solutions to f(x) = 0 are x₁ = -1 and x₂ = 3 -/
theorem quadratic_roots (a c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 2 * a * x + c
  f 3 = 0 → (∀ x, f x = 0 ↔ x = -1 ∨ x = 3) :=
by sorry

end quadratic_roots_l1803_180312


namespace discount_rate_example_l1803_180310

/-- Given a bag with a marked price and a selling price, calculate the discount rate. -/
def discount_rate (marked_price selling_price : ℚ) : ℚ :=
  (marked_price - selling_price) / marked_price * 100

/-- Theorem: The discount rate for a bag marked at $80 and sold for $68 is 15%. -/
theorem discount_rate_example : discount_rate 80 68 = 15 := by
  sorry

end discount_rate_example_l1803_180310


namespace quadratic_standard_form_l1803_180393

theorem quadratic_standard_form :
  ∀ x : ℝ, (x + 3) * (2 * x - 1) = -4 ↔ 2 * x^2 + 5 * x + 1 = 0 :=
by sorry

end quadratic_standard_form_l1803_180393


namespace longest_side_of_triangle_l1803_180305

/-- The length of the longest side of a triangle with vertices at (3,3), (8,9), and (9,3) is √61 units. -/
theorem longest_side_of_triangle : ∃ (a b c : ℝ × ℝ),
  a = (3, 3) ∧ b = (8, 9) ∧ c = (9, 3) ∧
  (max (dist a b) (max (dist b c) (dist c a)))^2 = 61 :=
by
  sorry

end longest_side_of_triangle_l1803_180305


namespace food_storage_temperature_l1803_180319

-- Define the temperature range
def temp_center : ℝ := -2
def temp_range : ℝ := 3

-- Define the function to check if a temperature is within the range
def is_within_range (temp : ℝ) : Prop :=
  temp ≥ temp_center - temp_range ∧ temp ≤ temp_center + temp_range

-- State the theorem
theorem food_storage_temperature :
  is_within_range (-1) ∧
  ¬is_within_range 2 ∧
  ¬is_within_range (-6) ∧
  ¬is_within_range 4 :=
sorry

end food_storage_temperature_l1803_180319


namespace increasing_f_implies_a_range_f_below_g_implies_a_range_l1803_180321

-- Define the function f
def f (a x : ℝ) : ℝ := x * |x - a| + 3 * x

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 1

-- Theorem 1
theorem increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → -3 ≤ a ∧ a ≤ 3 :=
sorry

-- Theorem 2
theorem f_below_g_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → f a x < g x) → 3/2 < a ∧ a < 2 :=
sorry

end increasing_f_implies_a_range_f_below_g_implies_a_range_l1803_180321


namespace max_surface_area_rectangular_solid_in_sphere_l1803_180358

theorem max_surface_area_rectangular_solid_in_sphere :
  ∀ (a b c : ℝ),
  (a > 0) → (b > 0) → (c > 0) →
  (a^2 + b^2 + c^2 = 4) →
  2 * (a * b + a * c + b * c) ≤ 8 :=
sorry

end max_surface_area_rectangular_solid_in_sphere_l1803_180358


namespace no_solution_for_sock_problem_l1803_180349

theorem no_solution_for_sock_problem : ¬∃ (n m : ℕ), 
  n + m = 2009 ∧ 
  (n * (n - 1) + m * (m - 1)) / ((n + m) * (n + m - 1)) = 1 / 2 := by
  sorry

end no_solution_for_sock_problem_l1803_180349


namespace polynomial_division_theorem_l1803_180352

theorem polynomial_division_theorem (x : ℚ) : 
  let dividend := 10 * x^4 - 3 * x^3 + 2 * x^2 - x + 6
  let divisor := 3 * x + 4
  let quotient := 10/3 * x^3 - 49/9 * x^2 + 427/27 * x - 287/54
  let remainder := 914/27
  dividend = divisor * quotient + remainder := by sorry

end polynomial_division_theorem_l1803_180352


namespace complex_equality_sum_l1803_180337

theorem complex_equality_sum (a b : ℕ) (h : a > 0 ∧ b > 0) :
  Complex.abs ((a : ℂ) + Complex.I) * Complex.abs (2 + Complex.I) =
  Complex.abs ((b : ℂ) - Complex.I) / Complex.abs (2 - Complex.I) →
  a + b = 8 := by
  sorry

end complex_equality_sum_l1803_180337


namespace triangle_area_fraction_l1803_180360

/-- The area of triangle ABC with vertices A(1,3), B(5,1), and C(4,4) is 1/6 of the area of a 6 × 5 rectangle. -/
theorem triangle_area_fraction (A B C : ℝ × ℝ) (h_A : A = (1, 3)) (h_B : B = (5, 1)) (h_C : C = (4, 4)) :
  let triangle_area := abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2)) / 2
  let rectangle_area := 6 * 5
  triangle_area / rectangle_area = 1 / 6 := by sorry

end triangle_area_fraction_l1803_180360


namespace pet_store_dogs_l1803_180326

theorem pet_store_dogs (cat_count : ℕ) (cat_ratio dog_ratio : ℕ) : 
  cat_count = 18 → cat_ratio = 3 → dog_ratio = 4 → 
  (cat_count * dog_ratio) / cat_ratio = 24 := by
sorry

end pet_store_dogs_l1803_180326


namespace march_total_distance_l1803_180304

/-- Represents Emberly's walking distance for a single day -/
structure DailyWalk where
  day : Nat
  distance : Float

/-- Emberly's walking pattern for March -/
def marchWalks : List DailyWalk := [
  ⟨1, 4⟩, ⟨2, 3⟩, ⟨3, 4⟩, ⟨4, 3⟩, ⟨5, 4⟩, ⟨6, 0⟩, ⟨7, 0⟩,
  ⟨8, 5⟩, ⟨9, 2.5⟩, ⟨10, 5⟩, ⟨11, 5⟩, ⟨12, 2.5⟩, ⟨13, 2.5⟩, ⟨14, 0⟩,
  ⟨15, 6⟩, ⟨16, 6⟩, ⟨17, 0⟩, ⟨18, 0⟩, ⟨19, 0⟩, ⟨20, 4⟩, ⟨21, 4⟩, ⟨22, 3.5⟩,
  ⟨23, 4.5⟩, ⟨24, 0⟩, ⟨25, 4.5⟩, ⟨26, 0⟩, ⟨27, 4.5⟩, ⟨28, 0⟩, ⟨29, 4.5⟩, ⟨30, 0⟩, ⟨31, 0⟩
]

/-- Calculate the total distance walked in March -/
def totalDistance (walks : List DailyWalk) : Float :=
  walks.foldl (fun acc walk => acc + walk.distance) 0

/-- Theorem: The total distance Emberly walked in March is 82 miles -/
theorem march_total_distance : totalDistance marchWalks = 82 := by
  sorry


end march_total_distance_l1803_180304


namespace geometric_sequence_product_l1803_180390

theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ n : ℕ, a n > 0) →  -- positive sequence
  (∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = r * a n) →  -- geometric sequence
  (a 1)^2 - 10*(a 1) + 16 = 0 →  -- a_1 is a root
  (a 19)^2 - 10*(a 19) + 16 = 0 →  -- a_19 is a root
  a 8 * a 10 * a 12 = 64 := by
sorry

end geometric_sequence_product_l1803_180390


namespace shopkeeper_cloth_sale_l1803_180385

/-- Calculates the total selling amount for cloth given the length, cost price, and loss per metre -/
def total_selling_amount (cloth_length : ℕ) (cost_price_per_metre : ℕ) (loss_per_metre : ℕ) : ℕ :=
  let selling_price_per_metre := cost_price_per_metre - loss_per_metre
  cloth_length * selling_price_per_metre

/-- Proves that the total selling amount for 200 metres of cloth with a cost price of 95 Rs per metre 
    and a loss of 5 Rs per metre is 18000 Rs -/
theorem shopkeeper_cloth_sale : 
  total_selling_amount 200 95 5 = 18000 := by
  sorry

end shopkeeper_cloth_sale_l1803_180385


namespace games_not_working_l1803_180395

theorem games_not_working (friend_games garage_games good_games : ℕ) : 
  friend_games = 2 → garage_games = 2 → good_games = 2 →
  friend_games + garage_games - good_games = 2 := by
sorry

end games_not_working_l1803_180395


namespace cubic_expression_equal_sixty_times_ten_power_l1803_180372

theorem cubic_expression_equal_sixty_times_ten_power : 
  (2^1501 + 5^1502)^3 - (2^1501 - 5^1502)^3 = 60 * 10^1501 := by sorry

end cubic_expression_equal_sixty_times_ten_power_l1803_180372


namespace train_problem_l1803_180306

/-- 
Given two trains departing simultaneously from points A and B towards each other,
this theorem proves the speeds of the trains and the distance between A and B.
-/
theorem train_problem (p q t : ℝ) (hp : p > 0) (hq : q > 0) (ht : t > 0) :
  ∃ (x y z : ℝ),
    x > 0 ∧ y > 0 ∧ z > 0 ∧  -- Speeds and distance are positive
    (p / y = (z - p) / x) ∧  -- Trains meet at distance p from B
    (t * y = q + z - p) ∧   -- Second train's position after t hours
    (t * (x + y) = 2 * z) ∧ -- Total distance traveled by both trains after t hours
    (x = (4 * p - 2 * q) / t) ∧ -- Speed of first train
    (y = 2 * p / t) ∧           -- Speed of second train
    (z = 3 * p - q)             -- Distance between A and B
  := by sorry

end train_problem_l1803_180306


namespace M_equality_l1803_180347

theorem M_equality : 
  let M := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2) - Real.sqrt (4 - 2 * Real.sqrt 3)
  M = (5/2) * Real.sqrt 2 - Real.sqrt 3 + 3/2 := by
  sorry

end M_equality_l1803_180347


namespace sarees_with_six_shirts_l1803_180328

/-- The price of a saree in dollars -/
def saree_price : ℕ := sorry

/-- The price of a shirt in dollars -/
def shirt_price : ℕ := sorry

/-- The number of sarees bought with 6 shirts -/
def num_sarees : ℕ := sorry

theorem sarees_with_six_shirts :
  (2 * saree_price + 4 * shirt_price = 1600) →
  (12 * shirt_price = 2400) →
  (num_sarees * saree_price + 6 * shirt_price = 1600) →
  num_sarees = 1 := by
  sorry

end sarees_with_six_shirts_l1803_180328


namespace f_monotone_increasing_max_a_value_l1803_180374

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 1) / (Real.exp x)

theorem f_monotone_increasing :
  ∀ x y, 1 < x ∧ x < y ∧ y < 2 → f x < f y :=
sorry

theorem max_a_value :
  (∀ x, x > 0 → Real.exp x * f x ≥ a + Real.log x) → a ≤ 1 :=
sorry

end f_monotone_increasing_max_a_value_l1803_180374


namespace sum_of_three_numbers_l1803_180353

theorem sum_of_three_numbers (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a + b * c = (a + b) * (a + c)) : a + b + c = 1 := by
  sorry

end sum_of_three_numbers_l1803_180353


namespace complex_number_equation_l1803_180300

theorem complex_number_equation : ∃ z : ℂ, z / (1 + Complex.I) = Complex.I ^ 2015 + Complex.I ^ 2016 ∧ z = 2 := by
  sorry

end complex_number_equation_l1803_180300


namespace terry_more_stickers_than_steven_l1803_180356

/-- Given the number of stickers each person has, prove that Terry has 20 more stickers than Steven -/
theorem terry_more_stickers_than_steven 
  (ryan_stickers : ℕ) 
  (steven_stickers : ℕ) 
  (terry_stickers : ℕ) 
  (total_stickers : ℕ) 
  (h1 : ryan_stickers = 30)
  (h2 : steven_stickers = 3 * ryan_stickers)
  (h3 : terry_stickers > steven_stickers)
  (h4 : ryan_stickers + steven_stickers + terry_stickers = total_stickers)
  (h5 : total_stickers = 230) :
  terry_stickers - steven_stickers = 20 := by
sorry

end terry_more_stickers_than_steven_l1803_180356


namespace sum_of_powers_l1803_180354

theorem sum_of_powers (ω : ℂ) (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^10 + ω^12 + ω^14 + ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 = 1 := by
  sorry

end sum_of_powers_l1803_180354


namespace terrell_lifting_equivalence_l1803_180398

/-- The number of times Terrell lifts the weights initially -/
def initial_lifts : ℕ := 10

/-- The weight of each dumbbell in the initial setup (in pounds) -/
def initial_weight : ℕ := 25

/-- The weight of each dumbbell in the new setup (in pounds) -/
def new_weight : ℕ := 20

/-- The number of dumbbells used in each lift -/
def num_dumbbells : ℕ := 2

/-- The number of times Terrell must lift the new weights to achieve the same total weight -/
def required_lifts : ℚ := 12.5

theorem terrell_lifting_equivalence :
  (num_dumbbells * initial_weight * initial_lifts : ℚ) = 
  (num_dumbbells * new_weight * required_lifts) :=
by sorry

end terrell_lifting_equivalence_l1803_180398


namespace angle_system_solutions_l1803_180359

theorem angle_system_solutions :
  ∀ x y : ℝ,
  0 ≤ x ∧ x < 2 * Real.pi ∧ 0 ≤ y ∧ y < 2 * Real.pi →
  Real.sin x + Real.cos y = 0 ∧ Real.cos x * Real.sin y = -1/2 →
  (x = Real.pi/4 ∧ y = 5*Real.pi/4) ∨
  (x = 3*Real.pi/4 ∧ y = 3*Real.pi/4) ∨
  (x = 5*Real.pi/4 ∧ y = Real.pi/4) ∨
  (x = 7*Real.pi/4 ∧ y = 7*Real.pi/4) :=
by sorry

end angle_system_solutions_l1803_180359


namespace donut_theft_ratio_l1803_180383

theorem donut_theft_ratio (initial_donuts : ℕ) (bill_eaten : ℕ) (secretary_taken : ℕ) (final_donuts : ℕ)
  (h1 : initial_donuts = 50)
  (h2 : bill_eaten = 2)
  (h3 : secretary_taken = 4)
  (h4 : final_donuts = 22) :
  (initial_donuts - bill_eaten - secretary_taken - final_donuts) / (initial_donuts - bill_eaten - secretary_taken) = 1 / 2 := by
sorry

end donut_theft_ratio_l1803_180383


namespace imaginary_part_of_complex_fraction_l1803_180303

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 2 * I / (1 + I)
  Complex.im z = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l1803_180303


namespace inequality_system_solution_l1803_180370

theorem inequality_system_solution :
  let S := {x : ℝ | 2*x > x + 1 ∧ 4*x - 1 > 7}
  S = {x : ℝ | x > 2} := by sorry

end inequality_system_solution_l1803_180370


namespace student_height_probability_l1803_180392

theorem student_height_probability (p_less_160 p_between_160_175 : ℝ) :
  p_less_160 = 0.2 →
  p_between_160_175 = 0.5 →
  1 - p_less_160 - p_between_160_175 = 0.3 :=
by sorry

end student_height_probability_l1803_180392


namespace geometric_sequence_property_l1803_180382

/-- A geometric sequence with the given property -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  arithmetic_property : 3 * a 1 + 2 * a 2 = a 3

/-- The main theorem -/
theorem geometric_sequence_property (seq : GeometricSequence) :
  (seq.a 9 + seq.a 10) / (seq.a 7 + seq.a 8) = 9 := by
  sorry

end geometric_sequence_property_l1803_180382


namespace arithmetic_sequence_2016th_term_l1803_180313

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_2016th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_incr : ∀ n : ℕ, a (n + 1) > a n) 
  (h_first : a 1 = 1) 
  (h_geom : (a 4)^2 = a 2 * a 8) :
  a 2016 = 2016 := by
sorry

end arithmetic_sequence_2016th_term_l1803_180313


namespace three_white_marbles_possible_l1803_180327

/-- Represents the state of the urn -/
structure UrnState where
  white : ℕ
  black : ℕ

/-- Represents the possible operations on the urn -/
inductive Operation
  | op1 | op2 | op3 | op4 | op5

/-- Applies an operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.op1 => UrnState.mk state.white (state.black - 2)
  | Operation.op2 => UrnState.mk (state.white - 1) (state.black - 2)
  | Operation.op3 => UrnState.mk state.white (state.black - 1)
  | Operation.op4 => UrnState.mk state.white (state.black - 1)
  | Operation.op5 => UrnState.mk (state.white - 3) (state.black + 2)

/-- Applies a sequence of operations to the urn state -/
def applyOperations (initial : UrnState) (ops : List Operation) : UrnState :=
  ops.foldl applyOperation initial

/-- The theorem to be proved -/
theorem three_white_marbles_possible :
  ∃ (ops : List Operation),
    let final := applyOperations (UrnState.mk 150 50) ops
    final.white = 3 ∧ final.black ≥ 0 := by
  sorry

end three_white_marbles_possible_l1803_180327
