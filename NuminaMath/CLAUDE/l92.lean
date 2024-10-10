import Mathlib

namespace vector_problem_l92_9256

-- Define the vectors
def OA (k : ℝ) : ℝ × ℝ := (k, 12)
def OB : ℝ × ℝ := (4, 5)
def OC (k : ℝ) : ℝ × ℝ := (-k, 10)

-- Define collinearity
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, B - A = t • (C - A)

-- State the theorem
theorem vector_problem (k : ℝ) :
  collinear (OA k) OB (OC k) → k = -2/3 := by
  sorry

end vector_problem_l92_9256


namespace chocolate_bar_cost_l92_9246

/-- The cost of chocolate bars for a scout camp -/
theorem chocolate_bar_cost (chocolate_bar_cost : ℝ) (sections_per_bar : ℕ) (num_scouts : ℕ) (smores_per_scout : ℕ) : 
  chocolate_bar_cost = 1.5 →
  sections_per_bar = 3 →
  num_scouts = 15 →
  smores_per_scout = 2 →
  (num_scouts * smores_per_scout : ℝ) / sections_per_bar * chocolate_bar_cost = 15 := by
  sorry

#check chocolate_bar_cost

end chocolate_bar_cost_l92_9246


namespace intersection_sum_l92_9226

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- The cubic equation y = x³ - 3x - 4 -/
def cubic (p : Point) : Prop :=
  p.y = p.x^3 - 3*p.x - 4

/-- The linear equation x + 3y = 3 -/
def linear (p : Point) : Prop :=
  p.x + 3*p.y = 3

theorem intersection_sum :
  ∃ (p₁ p₂ p₃ : Point),
    (cubic p₁ ∧ linear p₁) ∧
    (cubic p₂ ∧ linear p₂) ∧
    (cubic p₃ ∧ linear p₃) ∧
    (p₁.x + p₂.x + p₃.x = 8/3) ∧
    (p₁.y + p₂.y + p₃.y = 19/9) := by
  sorry

end intersection_sum_l92_9226


namespace arithmetic_sequence_sum_l92_9241

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →
  a 3 + a 4 + a 5 + a 6 + a 7 = 45 →
  a 5 = 9 := by
sorry

end arithmetic_sequence_sum_l92_9241


namespace composite_equal_if_same_greatest_divisors_l92_9251

/-- The set of greatest divisors of a natural number, excluding the number itself -/
def greatestDivisors (n : ℕ) : Set ℕ :=
  {d | d ∣ n ∧ d ≠ n ∧ ∀ k, k ∣ n ∧ k ≠ n → k ≤ d}

/-- Two natural numbers are composite if they are greater than 1 and not prime -/
def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

theorem composite_equal_if_same_greatest_divisors (a b : ℕ) 
    (ha : isComposite a) (hb : isComposite b) 
    (h : greatestDivisors a = greatestDivisors b) : 
  a = b := by
  sorry

end composite_equal_if_same_greatest_divisors_l92_9251


namespace complex_number_calculation_l92_9294

theorem complex_number_calculation (z : ℂ) : z = 1 + I → (2 / z) + z^2 = 1 + I := by
  sorry

end complex_number_calculation_l92_9294


namespace smallest_q_for_five_in_range_l92_9280

/-- The function g(x) defined as x^2 - 4x + q -/
def g (q : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + q

/-- 5 is within the range of g(x) -/
def in_range (q : ℝ) : Prop := ∃ x, g q x = 5

/-- The smallest value of q such that 5 is within the range of g(x) is 9 -/
theorem smallest_q_for_five_in_range : 
  (∃ q₀, in_range q₀ ∧ ∀ q, in_range q → q₀ ≤ q) ∧ 
  (∀ q, in_range q ↔ 9 ≤ q) :=
sorry

end smallest_q_for_five_in_range_l92_9280


namespace earliest_meeting_time_l92_9225

def anna_lap_time : ℕ := 5
def stephanie_lap_time : ℕ := 8
def james_lap_time : ℕ := 10

theorem earliest_meeting_time :
  let lap_times := [anna_lap_time, stephanie_lap_time, james_lap_time]
  Nat.lcm (Nat.lcm anna_lap_time stephanie_lap_time) james_lap_time = 40 := by
  sorry

end earliest_meeting_time_l92_9225


namespace enrollment_increase_l92_9276

theorem enrollment_increase (e1991 e1992 e1993 : ℝ) 
  (h1 : e1993 = e1991 * (1 + 0.38))
  (h2 : e1993 = e1992 * (1 + 0.15)) :
  e1992 = e1991 * (1 + 0.2) := by
  sorry

end enrollment_increase_l92_9276


namespace valid_assignment_d_plus_5_l92_9247

/-- Represents a programming language variable --/
structure Variable where
  name : String

/-- Represents a programming language expression --/
inductive Expression where
  | Var : Variable → Expression
  | Const : Int → Expression
  | Add : Expression → Expression → Expression

/-- Represents an assignment statement --/
structure Assignment where
  lhs : Variable
  rhs : Expression

/-- Predicate to check if an assignment is valid --/
def is_valid_assignment (a : Assignment) : Prop :=
  ∃ (d : Variable), a.lhs = d ∧ 
    a.rhs = Expression.Add (Expression.Var d) (Expression.Const 5)

/-- Theorem stating that "d = d + 5" is a valid assignment --/
theorem valid_assignment_d_plus_5 :
  ∃ (a : Assignment), is_valid_assignment a :=
sorry

end valid_assignment_d_plus_5_l92_9247


namespace num_cubes_5_peaks_num_cubes_2014_peaks_painted_area_2014_peaks_l92_9227

/-- Represents a wall made of unit cubes with a given number of peaks -/
structure Wall where
  peaks : ℕ

/-- The number of cubes needed to construct a wall with n peaks -/
def num_cubes (w : Wall) : ℕ := 3 * w.peaks - 1

/-- The painted surface area of a wall with n peaks, excluding the base -/
def painted_area (w : Wall) : ℕ := 10 * w.peaks + 9

/-- Theorem stating the number of cubes for a wall with 5 peaks -/
theorem num_cubes_5_peaks : num_cubes { peaks := 5 } = 14 := by sorry

/-- Theorem stating the number of cubes for a wall with 2014 peaks -/
theorem num_cubes_2014_peaks : num_cubes { peaks := 2014 } = 6041 := by sorry

/-- Theorem stating the painted area for a wall with 2014 peaks -/
theorem painted_area_2014_peaks : painted_area { peaks := 2014 } = 20139 := by sorry

end num_cubes_5_peaks_num_cubes_2014_peaks_painted_area_2014_peaks_l92_9227


namespace consecutive_integers_square_sum_l92_9273

theorem consecutive_integers_square_sum (n : ℕ) : 
  (n > 0) → 
  (n^2 + (n + 1)^2 = n * (n + 1) + 91) → 
  n = 9 := by
sorry

end consecutive_integers_square_sum_l92_9273


namespace train_crossing_time_l92_9228

theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 100 ∧ train_speed_kmh = 90 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 4 := by
sorry

end train_crossing_time_l92_9228


namespace root_one_when_sum_zero_reciprocal_roots_l92_9288

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Theorem 1: If a + b + c = 0, then x = 1 is a root
theorem root_one_when_sum_zero (a b c : ℝ) (ha : a ≠ 0) (hsum : a + b + c = 0) :
  quadratic a b c 1 := by sorry

-- Theorem 2: If x1 and x2 are roots of ax^2 + bx + c = 0 where x1 ≠ x2 ≠ 0,
-- then 1/x1 and 1/x2 are roots of cx^2 + bx + a = 0 (c ≠ 0)
theorem reciprocal_roots (a b c x1 x2 : ℝ) (ha : a ≠ 0) (hc : c ≠ 0)
  (hx1 : x1 ≠ 0) (hx2 : x2 ≠ 0) (hx1x2 : x1 ≠ x2)
  (hroot1 : quadratic a b c x1) (hroot2 : quadratic a b c x2) :
  quadratic c b a (1/x1) ∧ quadratic c b a (1/x2) := by sorry

end root_one_when_sum_zero_reciprocal_roots_l92_9288


namespace no_simultaneous_solution_l92_9258

theorem no_simultaneous_solution : ¬∃ x : ℝ, (5 * x^2 - 7 * x + 1 < 0) ∧ (x^2 - 9 * x + 30 < 0) := by
  sorry

end no_simultaneous_solution_l92_9258


namespace eliza_height_difference_l92_9205

/-- Given the heights of Eliza and her siblings, prove that Eliza is 2 inches shorter than the tallest sibling -/
theorem eliza_height_difference (total_height : ℕ) (sibling1_height sibling2_height sibling3_height eliza_height : ℕ) :
  total_height = 330 ∧
  sibling1_height = 66 ∧
  sibling2_height = 66 ∧
  sibling3_height = 60 ∧
  eliza_height = 68 →
  ∃ (tallest_sibling_height : ℕ),
    tallest_sibling_height + sibling1_height + sibling2_height + sibling3_height + eliza_height = total_height ∧
    tallest_sibling_height - eliza_height = 2 :=
by sorry

end eliza_height_difference_l92_9205


namespace boys_share_l92_9206

theorem boys_share (total_amount : ℕ) (total_children : ℕ) (num_boys : ℕ) (amount_per_girl : ℕ) 
  (h1 : total_amount = 460)
  (h2 : total_children = 41)
  (h3 : num_boys = 33)
  (h4 : amount_per_girl = 8) :
  (total_amount - (total_children - num_boys) * amount_per_girl) / num_boys = 12 := by
  sorry

end boys_share_l92_9206


namespace union_M_N_complement_N_P_subset_M_iff_l92_9264

-- Define the sets M, N, and P
def M : Set ℝ := {x | (x + 4) * (x - 6) < 0}
def N : Set ℝ := {x | x - 5 < 0}
def P (t : ℝ) : Set ℝ := {x | |x| = t}

-- Theorem 1: M ∪ N = {x | x < 6}
theorem union_M_N : M ∪ N = {x | x < 6} := by sorry

-- Theorem 2: N̄ₘ = {x | x ≥ 5}
theorem complement_N : (Nᶜ : Set ℝ) = {x | x ≥ 5} := by sorry

-- Theorem 3: P ⊆ M if and only if t ∈ (-∞, 4)
theorem P_subset_M_iff (t : ℝ) : P t ⊆ M ↔ t < 4 := by sorry

end union_M_N_complement_N_P_subset_M_iff_l92_9264


namespace tan_2y_value_l92_9208

theorem tan_2y_value (x y : ℝ) 
  (h : Real.sin (x - y) * Real.cos x - Real.cos (x - y) * Real.sin x = 3/5) : 
  Real.tan (2 * y) = 24/7 ∨ Real.tan (2 * y) = -24/7 := by
  sorry

end tan_2y_value_l92_9208


namespace x_value_when_y_is_one_l92_9234

theorem x_value_when_y_is_one (x y : ℝ) : 
  y = 2 / (4 * x + 2) → y = 1 → x = 0 := by
  sorry

end x_value_when_y_is_one_l92_9234


namespace waste_after_ten_years_l92_9209

/-- Calculates the amount of waste after n years given an initial amount and growth rate -/
def wasteAmount (a : ℝ) (b : ℝ) (n : ℕ) : ℝ :=
  a * (1 + b) ^ n

/-- Theorem: The amount of waste after 10 years is a(1+b)^10 -/
theorem waste_after_ten_years (a b : ℝ) :
  wasteAmount a b 10 = a * (1 + b) ^ 10 := by
  sorry

#check waste_after_ten_years

end waste_after_ten_years_l92_9209


namespace complement_of_P_in_U_l92_9272

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define the set P
def P : Set ℝ := {x : ℝ | x^2 - 5*x - 6 ≥ 0}

-- State the theorem
theorem complement_of_P_in_U : 
  Set.compl P = Set.Ioo (-1 : ℝ) (6 : ℝ) :=
sorry

end complement_of_P_in_U_l92_9272


namespace max_coprime_partition_l92_9265

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def valid_partition (A B : Finset ℕ) : Prop :=
  (∀ a ∈ A, 2 ≤ a ∧ a ≤ 20) ∧
  (∀ b ∈ B, 2 ≤ b ∧ b ≤ 20) ∧
  (∀ a ∈ A, ∀ b ∈ B, is_coprime a b) ∧
  A ∩ B = ∅ ∧
  A ∪ B ⊆ Finset.range 19 ∪ {20}

theorem max_coprime_partition :
  ∃ A B : Finset ℕ,
    valid_partition A B ∧
    A.card * B.card = 49 ∧
    ∀ C D : Finset ℕ, valid_partition C D → C.card * D.card ≤ 49 := by
  sorry

end max_coprime_partition_l92_9265


namespace radford_distance_at_finish_l92_9202

/-- Represents the race between Radford and Peter -/
structure Race where
  radford_initial_lead : ℝ
  peter_lead_after_3min : ℝ
  race_duration : ℝ
  peter_speed_advantage : ℝ

/-- Calculates the distance between Radford and Peter at the end of the race -/
def final_distance (race : Race) : ℝ :=
  race.peter_lead_after_3min + race.peter_speed_advantage * (race.race_duration - 3)

/-- Theorem stating that Radford is 82 meters behind Peter at the end of the race -/
theorem radford_distance_at_finish (race : Race) 
  (h1 : race.radford_initial_lead = 30)
  (h2 : race.peter_lead_after_3min = 18)
  (h3 : race.race_duration = 7)
  (h4 : race.peter_speed_advantage = 16) :
  final_distance race = 82 := by
  sorry

end radford_distance_at_finish_l92_9202


namespace bird_on_time_speed_l92_9261

/-- Represents the problem of Mr. Bird's commute --/
structure BirdCommute where
  distance : ℝ
  time_on_time : ℝ
  speed_late : ℝ
  speed_early : ℝ
  late_time : ℝ
  early_time : ℝ

/-- The theorem stating the correct speed for Mr. Bird to arrive on time --/
theorem bird_on_time_speed (b : BirdCommute) 
  (h1 : b.speed_late = 30)
  (h2 : b.speed_early = 50)
  (h3 : b.late_time = 5 / 60)
  (h4 : b.early_time = 5 / 60)
  (h5 : b.distance = b.speed_late * (b.time_on_time + b.late_time))
  (h6 : b.distance = b.speed_early * (b.time_on_time - b.early_time)) :
  b.distance / b.time_on_time = 37.5 := by
  sorry

end bird_on_time_speed_l92_9261


namespace contrapositive_equivalence_l92_9270

theorem contrapositive_equivalence (x : ℝ) :
  (x^2 < 1 → -1 < x ∧ x < 1) ↔ (x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1) := by sorry

end contrapositive_equivalence_l92_9270


namespace vector_sum_proof_l92_9285

def vector1 : Fin 2 → ℝ := ![5, -3]
def vector2 : Fin 2 → ℝ := ![-4, 6]
def vector3 : Fin 2 → ℝ := ![2, -8]

theorem vector_sum_proof :
  vector1 + vector2 + vector3 = ![3, -5] := by
  sorry

end vector_sum_proof_l92_9285


namespace frustum_smaller_radius_l92_9231

/-- A circular frustum with the given properties -/
structure CircularFrustum where
  r : ℝ  -- radius of the smaller base
  slant_height : ℝ
  lateral_area : ℝ

/-- The theorem statement -/
theorem frustum_smaller_radius (f : CircularFrustum) 
  (h1 : f.slant_height = 3)
  (h2 : f.lateral_area = 84 * Real.pi)
  (h3 : 2 * Real.pi * (3 * f.r) = 3 * (2 * Real.pi * f.r)) :
  f.r = 7 := by
  sorry

end frustum_smaller_radius_l92_9231


namespace unique_quadratic_solution_l92_9284

theorem unique_quadratic_solution (m : ℝ) : 
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) → m = 0 ∨ m = -1 := by
  sorry

end unique_quadratic_solution_l92_9284


namespace factory_working_days_l92_9217

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 4340

/-- The number of toys produced per day -/
def toys_per_day : ℕ := 2170

/-- The number of working days per week -/
def working_days : ℕ := toys_per_week / toys_per_day

theorem factory_working_days : working_days = 2 := by
  sorry

end factory_working_days_l92_9217


namespace average_weight_of_group_l92_9240

theorem average_weight_of_group (girls_count boys_count : ℕ) 
  (girls_avg_weight boys_avg_weight : ℝ) :
  girls_count = 5 →
  boys_count = 5 →
  girls_avg_weight = 45 →
  boys_avg_weight = 55 →
  let total_count := girls_count + boys_count
  let total_weight := girls_count * girls_avg_weight + boys_count * boys_avg_weight
  (total_weight / total_count : ℝ) = 50 := by
  sorry

end average_weight_of_group_l92_9240


namespace amelia_half_money_left_l92_9263

/-- Represents the fraction of money Amelia has left after buying all books -/
def amelia_money_left (total_money : ℝ) (book_cost : ℝ) (num_books : ℕ) : ℝ :=
  total_money - (book_cost * num_books)

/-- Theorem stating that Amelia will have half of her money left after buying all books -/
theorem amelia_half_money_left 
  (total_money : ℝ) (book_cost : ℝ) (num_books : ℕ) 
  (h1 : total_money > 0) 
  (h2 : book_cost > 0) 
  (h3 : num_books > 0)
  (h4 : (1/4) * total_money = (1/2) * (book_cost * num_books)) :
  amelia_money_left total_money book_cost num_books = (1/2) * total_money := by
  sorry

#check amelia_half_money_left

end amelia_half_money_left_l92_9263


namespace always_three_same_color_sum_zero_l92_9242

-- Define a type for colors
inductive Color
| White
| Black

-- Define a function type for coloring integers
def Coloring := Int → Color

-- Define the property that 2016 and 2017 are different colors
def DifferentColors (c : Coloring) : Prop :=
  c 2016 ≠ c 2017

-- Define the property of three integers having the same color and summing to zero
def ThreeSameColorSumZero (c : Coloring) : Prop :=
  ∃ x y z : Int, (c x = c y ∧ c y = c z) ∧ x + y + z = 0

-- State the theorem
theorem always_three_same_color_sum_zero (c : Coloring) :
  DifferentColors c → ThreeSameColorSumZero c := by
  sorry

end always_three_same_color_sum_zero_l92_9242


namespace complement_union_theorem_l92_9282

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 5}

theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {0, 1, 3, 4, 5} := by sorry

end complement_union_theorem_l92_9282


namespace tricycle_count_l92_9239

/-- Represents the number of vehicles of each type -/
structure VehicleCounts where
  bicycles : ℕ
  tricycles : ℕ
  scooters : ℕ

/-- The total number of children -/
def totalChildren : ℕ := 10

/-- The total number of wheels -/
def totalWheels : ℕ := 25

/-- Calculates the total number of children given the vehicle counts -/
def countChildren (v : VehicleCounts) : ℕ :=
  v.bicycles + v.tricycles + v.scooters

/-- Calculates the total number of wheels given the vehicle counts -/
def countWheels (v : VehicleCounts) : ℕ :=
  2 * v.bicycles + 3 * v.tricycles + v.scooters

/-- Theorem stating that the number of tricycles is 5 -/
theorem tricycle_count :
  ∃ (v : VehicleCounts),
    countChildren v = totalChildren ∧
    countWheels v = totalWheels ∧
    v.tricycles = 5 := by
  sorry

end tricycle_count_l92_9239


namespace remaining_red_cards_l92_9295

theorem remaining_red_cards (total_cards : ℕ) (red_cards : ℕ) (removed_cards : ℕ) : 
  total_cards = 52 → 
  red_cards = total_cards / 2 →
  removed_cards = 10 →
  red_cards - removed_cards = 16 := by
  sorry

end remaining_red_cards_l92_9295


namespace people_visited_neither_l92_9218

theorem people_visited_neither (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) :
  total = 100 →
  iceland = 55 →
  norway = 43 →
  both = 61 →
  total - (iceland + norway - both) = 63 := by
  sorry

end people_visited_neither_l92_9218


namespace polynomial_division_theorem_remainder_is_z_minus_one_l92_9253

/-- The polynomial division theorem for this specific case -/
theorem polynomial_division_theorem (z : ℂ) :
  ∃ (Q R : ℂ → ℂ), z^2023 + 1 = (z^2 - z + 1) * Q z + R z ∧ 
  (∀ x, ∃ (a b : ℂ), R x = a * x + b) := by sorry

/-- The main theorem proving R(z) = z - 1 -/
theorem remainder_is_z_minus_one :
  ∃ (Q R : ℂ → ℂ), 
    (∀ z, z^2023 + 1 = (z^2 - z + 1) * Q z + R z) ∧
    (∀ x, ∃ (a b : ℂ), R x = a * x + b) ∧
    (∀ z, R z = z - 1) := by sorry

end polynomial_division_theorem_remainder_is_z_minus_one_l92_9253


namespace smallest_prime_factor_of_1821_l92_9229

theorem smallest_prime_factor_of_1821 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1821 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1821 → p ≤ q :=
  sorry

end smallest_prime_factor_of_1821_l92_9229


namespace range_of_a_for_false_proposition_l92_9289

theorem range_of_a_for_false_proposition :
  {a : ℝ | ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + 2*a*x₀ + 2*a + 3 < 0} = Set.Ioi (-1) := by
  sorry

end range_of_a_for_false_proposition_l92_9289


namespace group_contains_perfect_square_diff_l92_9268

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem group_contains_perfect_square_diff :
  ∀ (partition : Fin 3 → Set ℕ),
    (∀ n : ℕ, n ≤ 46 → ∃ i : Fin 3, n ∈ partition i) →
    (∀ i j : Fin 3, i ≠ j → partition i ∩ partition j = ∅) →
    (∀ i : Fin 3, partition i ⊆ Finset.range 47) →
    ∃ (i : Fin 3) (a b : ℕ), 
      a ∈ partition i ∧ 
      b ∈ partition i ∧ 
      a ≠ b ∧ 
      is_perfect_square (max a b - min a b) :=
by
  sorry

#check group_contains_perfect_square_diff

end group_contains_perfect_square_diff_l92_9268


namespace house_number_unit_digit_l92_9281

def is_divisible_by (n m : ℕ) : Prop := ∃ k, n = m * k

def hundred_digit (n : ℕ) : ℕ := (n / 100) % 10

def unit_digit (n : ℕ) : ℕ := n % 10

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem house_number_unit_digit (n : ℕ) 
  (three_digit : 100 ≤ n ∧ n < 1000)
  (exactly_three_true : ∃ (s1 s2 s3 s4 s5 : Prop), 
    (s1 = is_divisible_by n 9) ∧
    (s2 = is_even n) ∧
    (s3 = (hundred_digit n = 3)) ∧
    (s4 = is_odd (unit_digit n)) ∧
    (s5 = is_divisible_by n 5) ∧
    ((s1 ∧ s2 ∧ s3 ∧ ¬s4 ∧ ¬s5) ∨
     (s1 ∧ s2 ∧ s3 ∧ ¬s4 ∧ s5) ∨
     (s1 ∧ s2 ∧ ¬s3 ∧ ¬s4 ∧ s5) ∨
     (s1 ∧ ¬s2 ∧ s3 ∧ ¬s4 ∧ s5) ∨
     (¬s1 ∧ s2 ∧ s3 ∧ ¬s4 ∧ s5))) :
  unit_digit n = 0 := by sorry

end house_number_unit_digit_l92_9281


namespace helen_lawn_gas_consumption_l92_9255

/-- Represents the number of months with 2 cuts per month -/
def low_frequency_months : ℕ := 4

/-- Represents the number of months with 4 cuts per month -/
def high_frequency_months : ℕ := 4

/-- Represents the number of cuts per month in low frequency months -/
def low_frequency_cuts : ℕ := 2

/-- Represents the number of cuts per month in high frequency months -/
def high_frequency_cuts : ℕ := 4

/-- Represents the number of cuts before needing to refuel -/
def cuts_per_refuel : ℕ := 4

/-- Represents the number of gallons used per refuel -/
def gallons_per_refuel : ℕ := 2

/-- Theorem stating that Helen will need 12 gallons of gas for lawn cutting from March through October -/
theorem helen_lawn_gas_consumption : 
  (low_frequency_months * low_frequency_cuts + high_frequency_months * high_frequency_cuts) / cuts_per_refuel * gallons_per_refuel = 12 :=
by sorry

end helen_lawn_gas_consumption_l92_9255


namespace abs_negative_2023_l92_9260

theorem abs_negative_2023 : |(-2023 : ℝ)| = 2023 := by
  sorry

end abs_negative_2023_l92_9260


namespace quiz_homework_difference_l92_9271

/-- Represents the points distribution in Paul's biology class -/
structure PointsDistribution where
  total : ℕ
  homework : ℕ
  quiz : ℕ
  test : ℕ

/-- The conditions for Paul's point distribution -/
def paulsDistribution (p : PointsDistribution) : Prop :=
  p.total = 265 ∧
  p.homework = 40 ∧
  p.test = 4 * p.quiz ∧
  p.total = p.homework + p.quiz + p.test

/-- Theorem stating the difference between quiz and homework points -/
theorem quiz_homework_difference (p : PointsDistribution) 
  (h : paulsDistribution p) : p.quiz - p.homework = 5 := by
  sorry

end quiz_homework_difference_l92_9271


namespace inequality_proof_l92_9213

theorem inequality_proof (n : ℕ+) (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  (1 - x + x^2 / 2)^(n : ℝ) - (1 - x)^(n : ℝ) ≤ x / 2 := by
  sorry

end inequality_proof_l92_9213


namespace function_value_at_symmetry_point_l92_9259

/-- Given a function f(x) = 3cos(ωx + φ) that satisfies f(π/6 + x) = f(π/6 - x) for all x,
    prove that f(π/6) equals either 3 or -3 -/
theorem function_value_at_symmetry_point 
  (ω φ : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = 3 * Real.cos (ω * x + φ))
  (h2 : ∀ x, f (π/6 + x) = f (π/6 - x)) :
  f (π/6) = 3 ∨ f (π/6) = -3 :=
sorry

end function_value_at_symmetry_point_l92_9259


namespace factorial_equation_solutions_l92_9215

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_equation_solutions :
  ∀ a b c : ℕ+,
    (factorial a.val + factorial b.val = 2^(factorial c.val)) ↔
    ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (2, 2, 2)) :=
by sorry

end factorial_equation_solutions_l92_9215


namespace star_running_back_yards_l92_9262

/-- Represents the yardage statistics for a football player -/
structure PlayerStats where
  total_yards : ℕ
  pass_yards : ℕ
  run_yards : ℕ

/-- Calculates the running yards for a player given total yards and pass yards -/
def calculate_run_yards (total : ℕ) (pass : ℕ) : ℕ :=
  total - pass

/-- Theorem stating that the star running back's running yards is 90 -/
theorem star_running_back_yards (player : PlayerStats)
    (h1 : player.total_yards = 150)
    (h2 : player.pass_yards = 60)
    (h3 : player.run_yards = calculate_run_yards player.total_yards player.pass_yards) :
    player.run_yards = 90 := by
  sorry

end star_running_back_yards_l92_9262


namespace dining_bill_proof_l92_9297

theorem dining_bill_proof (num_people : ℕ) (individual_payment : ℚ) (tip_percentage : ℚ) 
  (h1 : num_people = 7)
  (h2 : individual_payment = 21.842857142857145)
  (h3 : tip_percentage = 1/10) :
  (num_people : ℚ) * individual_payment / (1 + tip_percentage) = 139 := by
  sorry

end dining_bill_proof_l92_9297


namespace profit_maximization_l92_9299

/-- Profit function given price x -/
def profit (x : ℝ) : ℝ := (x - 40) * (300 - (x - 60) * 10)

/-- The price that maximizes profit -/
def optimal_price : ℝ := 65

/-- The maximum profit achieved -/
def max_profit : ℝ := 6250

theorem profit_maximization :
  (∀ x : ℝ, profit x ≤ profit optimal_price) ∧
  profit optimal_price = max_profit := by
  sorry

#check profit_maximization

end profit_maximization_l92_9299


namespace container_capacity_l92_9249

/-- Given that 8 liters is 20% of a container's capacity, prove that 40 such containers have a total capacity of 1600 liters. -/
theorem container_capacity (container_capacity : ℝ) 
  (h1 : 8 = 0.2 * container_capacity) 
  (h2 : container_capacity > 0) : 
  40 * container_capacity = 1600 := by
  sorry

end container_capacity_l92_9249


namespace equation_solution_l92_9235

theorem equation_solution :
  ∃ x : ℝ, (2*x + 1)/3 - (5*x - 1)/6 = 1 ∧ x = -3 :=
by sorry

end equation_solution_l92_9235


namespace remainder_of_2615_base12_div_9_l92_9232

/-- Converts a base-12 digit to its decimal equivalent -/
def base12ToDecimal (digit : ℕ) : ℕ := digit

/-- Calculates the decimal value of a base-12 number given its digits -/
def base12Value (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  base12ToDecimal d₃ * 12^3 + base12ToDecimal d₂ * 12^2 + 
  base12ToDecimal d₁ * 12^1 + base12ToDecimal d₀ * 12^0

/-- The base-12 number 2615₁₂ -/
def num : ℕ := base12Value 2 6 1 5

theorem remainder_of_2615_base12_div_9 :
  num % 9 = 8 := by sorry

end remainder_of_2615_base12_div_9_l92_9232


namespace quotient_of_powers_l92_9200

theorem quotient_of_powers (a b c : ℕ) (ha : a = 50) (hb : b = 25) (hc : c = 100) :
  (a ^ 50) / (b ^ 25) = c ^ 25 := by
  sorry

end quotient_of_powers_l92_9200


namespace solution_set_range_l92_9207

-- Define the function f(x) for a given a
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 1) * x^2 - (a - 1) * x - 1

-- Define the property that f(x) < 0 for all real x
def always_negative (a : ℝ) : Prop := ∀ x : ℝ, f a x < 0

-- Define the set of a for which f(x) < 0 for all real x
def solution_set : Set ℝ := {a : ℝ | always_negative a}

-- State the theorem
theorem solution_set_range : solution_set = Set.Ioc (-3/5) 1 := by sorry

end solution_set_range_l92_9207


namespace rent_increase_new_mean_l92_9211

theorem rent_increase_new_mean 
  (num_friends : ℕ) 
  (initial_average : ℝ) 
  (increased_rent : ℝ) 
  (increase_percentage : ℝ) : 
  num_friends = 4 → 
  initial_average = 800 → 
  increased_rent = 800 → 
  increase_percentage = 0.25 → 
  (num_friends * initial_average + increased_rent * increase_percentage) / num_friends = 850 := by
  sorry

end rent_increase_new_mean_l92_9211


namespace alpha_not_rational_l92_9248

theorem alpha_not_rational (α : ℝ) (h : Real.cos (α * π / 180) = 1/3) : ¬ (∃ (m n : ℤ), α = m / n) := by
  sorry

end alpha_not_rational_l92_9248


namespace equal_sums_exist_l92_9252

/-- Represents a 3x3 grid with values from {-1, 0, 1} -/
def Grid := Matrix (Fin 3) (Fin 3) (Fin 3)

/-- Computes the sum of a row in the grid -/
def rowSum (g : Grid) (i : Fin 3) : ℤ := sorry

/-- Computes the sum of a column in the grid -/
def colSum (g : Grid) (j : Fin 3) : ℤ := sorry

/-- Computes the sum of the main diagonal -/
def mainDiagSum (g : Grid) : ℤ := sorry

/-- Computes the sum of the anti-diagonal -/
def antiDiagSum (g : Grid) : ℤ := sorry

/-- All possible sums in the grid -/
def allSums (g : Grid) : List ℤ := 
  [rowSum g 0, rowSum g 1, rowSum g 2, 
   colSum g 0, colSum g 1, colSum g 2, 
   mainDiagSum g, antiDiagSum g]

theorem equal_sums_exist (g : Grid) : 
  ∃ (i j : Fin 8), i ≠ j ∧ (allSums g).get i = (allSums g).get j := by sorry

end equal_sums_exist_l92_9252


namespace matchsticks_left_six_matchsticks_left_l92_9219

/-- Calculates the number of matchsticks left after Elvis and Ralph create their squares --/
theorem matchsticks_left (total : ℕ) (elvis_max : ℕ) (ralph_max : ℕ) 
  (elvis_per_square : ℕ) (ralph_per_square : ℕ) : ℕ :=
  let elvis_squares := elvis_max / elvis_per_square
  let ralph_squares := ralph_max / ralph_per_square
  let elvis_used := elvis_squares * elvis_per_square
  let ralph_used := ralph_squares * ralph_per_square
  total - (elvis_used + ralph_used)

/-- Proves that 6 matchsticks are left under the given conditions --/
theorem six_matchsticks_left : 
  matchsticks_left 50 20 30 4 8 = 6 := by
  sorry

end matchsticks_left_six_matchsticks_left_l92_9219


namespace tan_geq_one_range_l92_9278

open Set
open Real

theorem tan_geq_one_range (f : ℝ → ℝ) (h : ∀ x ∈ Ioo (-π/2) (π/2), f x = tan x) :
  {x ∈ Ioo (-π/2) (π/2) | f x ≥ 1} = Ico (π/4) (π/2) := by
  sorry

end tan_geq_one_range_l92_9278


namespace inequality_proof_l92_9245

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 3) : 
  a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2 := by
sorry

end inequality_proof_l92_9245


namespace cubic_sum_problem_l92_9250

theorem cubic_sum_problem (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 2) 
  (h3 : a^3 + b^3 + c^3 = 3) : 
  a * b * c = 1/6 ∧ a^4 + b^4 + c^4 = 25/6 := by
  sorry

end cubic_sum_problem_l92_9250


namespace max_winner_number_l92_9243

/-- Represents a wrestler in the tournament -/
structure Wrestler :=
  (number : ℕ)

/-- The tournament setup -/
def Tournament :=
  { wrestlers : Finset Wrestler // wrestlers.card = 512 }

/-- Predicate for the winning condition in a match -/
def wins (w1 w2 : Wrestler) : Prop :=
  w1.number < w2.number ∧ w2.number - w1.number > 2

/-- The winner of the tournament -/
def tournamentWinner (t : Tournament) : Wrestler :=
  sorry

/-- Theorem stating the maximum possible qualification number of the winner -/
theorem max_winner_number (t : Tournament) : 
  (tournamentWinner t).number ≤ 18 :=
sorry

end max_winner_number_l92_9243


namespace max_sum_red_green_balls_l92_9222

theorem max_sum_red_green_balls :
  ∀ (total red green blue : ℕ),
    total = 28 →
    green = 12 →
    red + green + blue = total →
    red ≤ 11 →
    red + green ≤ 23 ∧ ∃ (red' : ℕ), red' ≤ 11 ∧ red' + green = 23 :=
by sorry

end max_sum_red_green_balls_l92_9222


namespace volume_to_surface_area_ratio_l92_9277

/-- A structure made of unit cubes -/
structure CubeStructure where
  /-- The number of unit cubes in the structure -/
  num_cubes : ℕ
  /-- The volume of the structure in cubic units -/
  volume : ℕ
  /-- The surface area of the structure in square units -/
  surface_area : ℕ
  /-- The structure has a central cube surrounded symmetrically on all faces except the bottom -/
  has_central_cube : Prop
  /-- The structure forms a large plus sign when viewed from the top -/
  is_plus_shaped : Prop

/-- The specific cube structure described in the problem -/
def plus_structure : CubeStructure :=
  { num_cubes := 9
  , volume := 9
  , surface_area := 31
  , has_central_cube := True
  , is_plus_shaped := True }

/-- The theorem stating that the ratio of volume to surface area for the plus_structure is 9/31 -/
theorem volume_to_surface_area_ratio (s : CubeStructure) (h1 : s = plus_structure) :
  (s.volume : ℚ) / s.surface_area = 9 / 31 := by
  sorry

end volume_to_surface_area_ratio_l92_9277


namespace consecutive_even_ages_l92_9221

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem consecutive_even_ages (a b c : ℕ) 
  (h1 : is_even a)
  (h2 : is_even b)
  (h3 : is_even c)
  (h4 : b = a + 2)
  (h5 : c = b + 2)
  (h6 : a + b + c = 48) :
  a = 14 ∧ c = 18 := by
sorry

end consecutive_even_ages_l92_9221


namespace corrected_mean_l92_9238

theorem corrected_mean (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 ∧ initial_mean = 32 ∧ incorrect_value = 23 ∧ correct_value = 48 →
  (n * initial_mean - incorrect_value + correct_value) / n = 32.5 := by
  sorry

end corrected_mean_l92_9238


namespace series_sum_equals_one_l92_9244

open Real

noncomputable def seriesSum : ℝ := ∑' k, (k^2 : ℝ) / 3^k

theorem series_sum_equals_one : seriesSum = 1 := by sorry

end series_sum_equals_one_l92_9244


namespace function_value_range_l92_9269

theorem function_value_range (a : ℝ) : 
  (∃ x y : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ∧ y ∈ Set.Icc (-1 : ℝ) 1 ∧ 
   (a * x + 2 * a + 1) * (a * y + 2 * a + 1) < 0) ↔ 
  a ∈ Set.Ioo (-(1/3) : ℝ) (-1) :=
by sorry

end function_value_range_l92_9269


namespace quadratic_equation_real_roots_m_range_l92_9290

theorem quadratic_equation_real_roots_m_range 
  (m : ℝ) 
  (has_real_roots : ∃ x : ℝ, (m - 2) * x^2 + 2 * m * x + m + 3 = 0) :
  m ≤ 6 ∧ m ≠ 2 :=
by sorry

end quadratic_equation_real_roots_m_range_l92_9290


namespace lung_cancer_probability_l92_9293

theorem lung_cancer_probability (overall_prob : ℝ) (smoker_ratio : ℝ) (smoker_prob : ℝ) :
  overall_prob = 0.001 →
  smoker_ratio = 0.2 →
  smoker_prob = 0.004 →
  ∃ (nonsmoker_prob : ℝ),
    nonsmoker_prob = 0.00025 ∧
    overall_prob = smoker_ratio * smoker_prob + (1 - smoker_ratio) * nonsmoker_prob :=
by sorry

end lung_cancer_probability_l92_9293


namespace parallel_vectors_dot_product_l92_9216

/-- Given vectors a and b in ℝ², where a is parallel to b, prove their dot product is -5 -/
theorem parallel_vectors_dot_product (x : ℝ) :
  let a : Fin 2 → ℝ := ![x, x - 1]
  let b : Fin 2 → ℝ := ![1, 2]
  (∃ (k : ℝ), a = k • b) →
  (a 0 * b 0 + a 1 * b 1 = -5) :=
by sorry

end parallel_vectors_dot_product_l92_9216


namespace star_cell_is_one_l92_9292

/-- Represents a 4x4 grid of natural numbers -/
def Grid := Fin 4 → Fin 4 → Nat

/-- Check if all numbers in the grid are nonzero -/
def all_nonzero (g : Grid) : Prop :=
  ∀ i j, g i j ≠ 0

/-- Calculate the product of a row -/
def row_product (g : Grid) (i : Fin 4) : Nat :=
  (g i 0) * (g i 1) * (g i 2) * (g i 3)

/-- Calculate the product of a column -/
def col_product (g : Grid) (j : Fin 4) : Nat :=
  (g 0 j) * (g 1 j) * (g 2 j) * (g 3 j)

/-- Calculate the product of the main diagonal -/
def main_diag_product (g : Grid) : Nat :=
  (g 0 0) * (g 1 1) * (g 2 2) * (g 3 3)

/-- Calculate the product of the anti-diagonal -/
def anti_diag_product (g : Grid) : Nat :=
  (g 0 3) * (g 1 2) * (g 2 1) * (g 3 0)

/-- Check if all products are equal -/
def all_products_equal (g : Grid) : Prop :=
  let p := row_product g 0
  (∀ i, row_product g i = p) ∧
  (∀ j, col_product g j = p) ∧
  (main_diag_product g = p) ∧
  (anti_diag_product g = p)

/-- The main theorem -/
theorem star_cell_is_one (g : Grid) 
  (h1 : all_nonzero g)
  (h2 : all_products_equal g)
  (h3 : g 1 1 = 2)
  (h4 : g 1 2 = 16)
  (h5 : g 2 1 = 8)
  (h6 : g 2 2 = 32) :
  g 1 3 = 1 := by
  sorry

end star_cell_is_one_l92_9292


namespace complex_division_negative_l92_9204

theorem complex_division_negative (m : ℝ) : 
  let z₁ : ℂ := 2 + 3*I
  let z₂ : ℂ := m - I
  (z₁ / z₂).re < 0 ∧ (z₁ / z₂).im = 0 → m = -2/3 :=
by sorry

end complex_division_negative_l92_9204


namespace cone_no_rectangular_cross_section_cone_unique_no_rectangular_cross_section_l92_9287

-- Define the types of geometric shapes we're considering
inductive GeometricShape
| Cone
| Cylinder
| TriangularPrism
| RectangularPrism

-- Define a function that determines if a shape can have a rectangular cross-section
def canHaveRectangularCrossSection (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Cone => False
  | _ => True

-- Theorem statement
theorem cone_no_rectangular_cross_section :
  ∀ (shape : GeometricShape),
    ¬(canHaveRectangularCrossSection shape) ↔ shape = GeometricShape.Cone :=
by
  sorry

-- Alternative formulation focusing on the unique property of the cone
theorem cone_unique_no_rectangular_cross_section :
  ∃! (shape : GeometricShape), ¬(canHaveRectangularCrossSection shape) :=
by
  sorry

end cone_no_rectangular_cross_section_cone_unique_no_rectangular_cross_section_l92_9287


namespace vector_b_coordinates_l92_9236

theorem vector_b_coordinates (a b : ℝ × ℝ) :
  a = (Real.sqrt 3, Real.sqrt 5) →
  (a.1 * b.1 + a.2 * b.2 = 0) →
  (b.1^2 + b.2^2 = 4) →
  (b = (-Real.sqrt 10 / 2, Real.sqrt 6 / 2) ∨ b = (Real.sqrt 10 / 2, -Real.sqrt 6 / 2)) :=
by sorry

end vector_b_coordinates_l92_9236


namespace f_ordering_l92_9296

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem f_ordering : f (-π/3) > f (-1) ∧ f (-1) > f (π/11) := by
  sorry

end f_ordering_l92_9296


namespace sector_forms_cone_l92_9286

/-- Represents a circular sector -/
structure CircularSector where
  radius : ℝ
  angle : ℝ

/-- Represents a cone -/
structure Cone where
  baseRadius : ℝ
  slantHeight : ℝ

/-- Given a circular sector, returns the cone formed by aligning its straight sides -/
def sectorToCone (sector : CircularSector) : Cone :=
  sorry

theorem sector_forms_cone :
  let sector : CircularSector := ⟨12, 270 * π / 180⟩
  let cone : Cone := sectorToCone sector
  cone.baseRadius = 9 ∧ cone.slantHeight = 12 := by
  sorry

end sector_forms_cone_l92_9286


namespace fraction_of_5000_l92_9230

theorem fraction_of_5000 : 
  ∃ (f : ℚ), (f * (1/2 * (2/5 * 5000)) = 750.0000000000001) ∧ (f = 3/4) := by
  sorry

end fraction_of_5000_l92_9230


namespace least_phrases_to_learn_l92_9203

theorem least_phrases_to_learn (total_phrases : ℕ) (min_grade : ℚ) : 
  total_phrases = 600 → min_grade = 90 / 100 → 
  ∃ (least_phrases : ℕ), 
    (least_phrases : ℚ) / total_phrases ≥ min_grade ∧
    ∀ (n : ℕ), (n : ℚ) / total_phrases ≥ min_grade → n ≥ least_phrases ∧
    least_phrases = 540 :=
by sorry

end least_phrases_to_learn_l92_9203


namespace point_division_ratios_l92_9274

/-- Given two points A and B on a line, there exist points M and N such that
    AM:MB = 2:1 and AN:NB = 1:3 respectively. -/
theorem point_division_ratios (A B : ℝ) : 
  (∃ M : ℝ, |A - M| / |M - B| = 2) ∧ 
  (∃ N : ℝ, |A - N| / |N - B| = 1/3) := by
  sorry

end point_division_ratios_l92_9274


namespace sum_of_multiples_l92_9267

theorem sum_of_multiples (m n : ℝ) : 2 * m + 3 * n = 2*m + 3*n := by sorry

end sum_of_multiples_l92_9267


namespace no_k_for_all_positive_quadratic_l92_9291

theorem no_k_for_all_positive_quadratic : ¬∃ k : ℝ, ∀ x : ℝ, x^2 - (k - 4)*x - (k + 2) > 0 := by
  sorry

end no_k_for_all_positive_quadratic_l92_9291


namespace arithmetic_mean_problem_l92_9233

theorem arithmetic_mean_problem (a₁ a₂ a₃ a₄ a₅ a₆ A : ℝ) 
  (h_mean : (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 = A)
  (h_first_four : (a₁ + a₂ + a₃ + a₄) / 4 = A + 10)
  (h_last_four : (a₃ + a₄ + a₅ + a₆) / 4 = A - 7) :
  (a₁ + a₂ + a₅ + a₆) / 4 = A - 3 := by
  sorry

end arithmetic_mean_problem_l92_9233


namespace die_rolls_for_most_likely_32_twos_l92_9224

/-- The number of rolls needed for the most likely number of twos to be 32 -/
theorem die_rolls_for_most_likely_32_twos :
  ∃ n : ℕ, 191 ≤ n ∧ n ≤ 197 ∧
  (∀ k : ℕ, (Nat.choose n k * (1/6)^k * (5/6)^(n-k)) ≤ (Nat.choose n 32 * (1/6)^32 * (5/6)^(n-32))) :=
by sorry

end die_rolls_for_most_likely_32_twos_l92_9224


namespace special_triangle_ac_length_l92_9298

/-- A triangle ABC with a point D on side AC, satisfying specific conditions -/
structure SpecialTriangle where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point D on side AC -/
  D : ℝ × ℝ
  /-- AB is greater than BC -/
  ab_gt_bc : dist A B > dist B C
  /-- BC equals 6 -/
  bc_eq_six : dist B C = 6
  /-- BD equals 7 -/
  bd_eq_seven : dist B D = 7
  /-- Triangle ABD is isosceles -/
  abd_isosceles : dist A B = dist A D ∨ dist A B = dist B D
  /-- Triangle BCD is isosceles -/
  bcd_isosceles : dist B C = dist C D ∨ dist B D = dist C D
  /-- D lies on AC -/
  d_on_ac : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = ((1 - t) • A.1 + t • C.1, (1 - t) • A.2 + t • C.2)

/-- The length of AC in the special triangle is 13 -/
theorem special_triangle_ac_length (t : SpecialTriangle) : dist t.A t.C = 13 := by
  sorry

end special_triangle_ac_length_l92_9298


namespace new_pressure_is_two_l92_9257

/-- Represents the pressure-volume relationship at constant temperature -/
structure GasState where
  pressure : ℝ
  volume : ℝ
  constant : ℝ

/-- The pressure-volume relationship is inversely proportional -/
axiom pressure_volume_constant (state : GasState) : state.pressure * state.volume = state.constant

/-- Initial state of the gas -/
def initial_state : GasState :=
  { pressure := 4
    volume := 3
    constant := 4 * 3 }

/-- New state of the gas after transfer -/
def new_state : GasState :=
  { pressure := 2  -- This is what we want to prove
    volume := 6
    constant := initial_state.constant }

/-- Theorem stating that the new pressure is 2 kPa -/
theorem new_pressure_is_two :
  new_state.pressure = 2 := by sorry

end new_pressure_is_two_l92_9257


namespace simple_interest_rate_calculation_l92_9201

/-- Calculate the interest rate given the principal, time, and total interest for a simple interest loan. -/
theorem simple_interest_rate_calculation
  (principal : ℝ)
  (time : ℝ)
  (total_interest : ℝ)
  (h_principal : principal = 5000)
  (h_time : time = 10)
  (h_total_interest : total_interest = 2000) :
  (total_interest * 100) / (principal * time) = 4 := by
  sorry

end simple_interest_rate_calculation_l92_9201


namespace max_dot_product_l92_9283

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (2, -1)

-- Define the moving point M
def M : Set (ℝ × ℝ) := {p | -2 ≤ p.1 ∧ p.1 ≤ 2 ∧ -2 ≤ p.2 ∧ p.2 ≤ 2}

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem max_dot_product :
  ∃ (max : ℝ), max = 4 ∧ ∀ m ∈ M, dot_product (m.1 - O.1, m.2 - O.2) (C.1 - O.1, C.2 - O.2) ≤ max :=
sorry

end max_dot_product_l92_9283


namespace simplify_expression_l92_9212

theorem simplify_expression :
  (Real.sqrt 2 + 1) ^ (1 - Real.sqrt 3) / (Real.sqrt 2 - 1) ^ (1 + Real.sqrt 3) = 3 + 2 * Real.sqrt 2 := by
  sorry

end simplify_expression_l92_9212


namespace polyline_segment_bound_l92_9275

/-- Represents a grid paper with square side length 1 -/
structure GridPaper where
  -- Additional structure properties can be added if needed

/-- Represents a point on the grid paper -/
structure GridPoint where
  -- Additional point properties can be added if needed

/-- Represents a polyline segment on the grid paper -/
structure PolylineSegment where
  start : GridPoint
  length : ℕ
  -- Additional segment properties can be added if needed

/-- 
  P_k denotes the number of different polyline segments of length k 
  starting from a fixed point O on a grid paper, where each segment 
  lies along the grid lines
-/
def P (grid : GridPaper) (O : GridPoint) (k : ℕ) : ℕ :=
  sorry -- Definition of P_k

/-- 
  Theorem: For all natural numbers k, the number of different polyline 
  segments of length k starting from a fixed point O on a grid paper 
  with square side length 1, where each segment lies along the grid lines, 
  is less than 2 × 3^k
-/
theorem polyline_segment_bound 
  (grid : GridPaper) (O : GridPoint) : 
  ∀ k : ℕ, P grid O k < 2 * 3^k := by
  sorry


end polyline_segment_bound_l92_9275


namespace mothers_carrots_l92_9254

theorem mothers_carrots (faye_carrots good_carrots bad_carrots : ℕ) 
  (h1 : faye_carrots = 23)
  (h2 : good_carrots = 12)
  (h3 : bad_carrots = 16) :
  good_carrots + bad_carrots - faye_carrots = 5 :=
by sorry

end mothers_carrots_l92_9254


namespace negation_of_forall_leq_is_exists_gt_l92_9237

theorem negation_of_forall_leq_is_exists_gt (p : (n : ℕ) → n^2 ≤ 2^n → Prop) :
  (¬ ∀ n : ℕ, n^2 ≤ 2^n) ↔ (∃ n : ℕ, n^2 > 2^n) := by
  sorry

end negation_of_forall_leq_is_exists_gt_l92_9237


namespace region_area_theorem_l92_9214

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bounded by two circles and the x-axis -/
def areaRegion (c1 c2 : Circle) : ℝ :=
  sorry

theorem region_area_theorem (c1 c2 : Circle) 
  (h1 : c1.center = (3, 5) ∧ c1.radius = 5)
  (h2 : c2.center = (13, 5) ∧ c2.radius = 5) : 
  areaRegion c1 c2 = 50 - 12.5 * Real.pi :=
sorry

end region_area_theorem_l92_9214


namespace product_equals_zero_l92_9266

theorem product_equals_zero : (3 * 5 * 7 + 4 * 6 * 8) * (2 * 12 * 5 - 20 * 3 * 2) = 0 := by
  sorry

end product_equals_zero_l92_9266


namespace insurance_coverage_percentage_l92_9223

def mri_cost : ℝ := 1200
def doctor_rate : ℝ := 300
def doctor_time : ℝ := 0.5
def fee_for_seen : ℝ := 150
def tim_payment : ℝ := 300

def total_cost : ℝ := mri_cost + doctor_rate * doctor_time + fee_for_seen

def insurance_coverage : ℝ := total_cost - tim_payment

theorem insurance_coverage_percentage : 
  insurance_coverage / total_cost * 100 = 80 := by sorry

end insurance_coverage_percentage_l92_9223


namespace path_area_is_775_l92_9279

/-- Represents the dimensions of a rectangular field with a surrounding path -/
structure FieldWithPath where
  field_length : ℝ
  field_width : ℝ
  path_width : ℝ

/-- Calculates the area of the path surrounding a rectangular field -/
def path_area (f : FieldWithPath) : ℝ :=
  (f.field_length + 2 * f.path_width) * (f.field_width + 2 * f.path_width) -
  f.field_length * f.field_width

/-- Theorem stating that the area of the path for the given field dimensions is 775 sq m -/
theorem path_area_is_775 :
  let f : FieldWithPath := {
    field_length := 95,
    field_width := 55,
    path_width := 2.5
  }
  path_area f = 775 := by sorry

end path_area_is_775_l92_9279


namespace determinant_transformation_l92_9210

theorem determinant_transformation (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = 9 →
  Matrix.det !![2*p, 5*p + 4*q; 2*r, 5*r + 4*s] = 72 := by
  sorry

end determinant_transformation_l92_9210


namespace polynomial_remainder_l92_9220

theorem polynomial_remainder (x : ℝ) : 
  (5 * x^3 - 9 * x^2 + 3 * x + 17) % (x - 2) = 27 := by
  sorry

end polynomial_remainder_l92_9220
