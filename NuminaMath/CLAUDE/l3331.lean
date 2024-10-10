import Mathlib

namespace train_length_l3331_333103

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 56 → time_s = 9 → ∃ (length_m : ℝ), 
  (abs (length_m - (speed_kmh * 1000 / 3600 * time_s)) < 0.01) ∧ 
  (abs (length_m - 140) < 0.01) := by
  sorry

end train_length_l3331_333103


namespace bus_driver_overtime_pay_increase_l3331_333110

/-- Calculates the percentage increase in overtime pay rate for a bus driver -/
theorem bus_driver_overtime_pay_increase 
  (regular_rate : ℝ) 
  (regular_hours : ℝ) 
  (total_compensation : ℝ) 
  (total_hours : ℝ) : 
  regular_rate = 16 →
  regular_hours = 40 →
  total_compensation = 920 →
  total_hours = 50 →
  ((total_compensation - regular_rate * regular_hours) / (total_hours - regular_hours) - regular_rate) / regular_rate * 100 = 75 := by
  sorry


end bus_driver_overtime_pay_increase_l3331_333110


namespace newberg_airport_passengers_l3331_333149

theorem newberg_airport_passengers : 
  let on_time_passengers : ℕ := 14507
  let late_passengers : ℕ := 213
  on_time_passengers + late_passengers = 14720 := by sorry

end newberg_airport_passengers_l3331_333149


namespace prop_a_necessary_not_sufficient_for_prop_b_l3331_333178

theorem prop_a_necessary_not_sufficient_for_prop_b :
  (∀ (a b : ℝ), (1 / b < 1 / a ∧ 1 / a < 0) → a * b > b ^ 2) ∧
  (∃ (a b : ℝ), a * b > b ^ 2 ∧ ¬(1 / b < 1 / a ∧ 1 / a < 0)) := by
  sorry

end prop_a_necessary_not_sufficient_for_prop_b_l3331_333178


namespace exists_subsequences_forming_2520_l3331_333169

def infinite_sequence : ℕ → ℕ
  | n => match n % 6 with
         | 0 => 2
         | 1 => 0
         | 2 => 1
         | 3 => 5
         | 4 => 2
         | 5 => 0
         | _ => 0  -- This case should never occur

def is_subsequence (s : List ℕ) : Prop :=
  ∃ start : ℕ, ∀ i : ℕ, i < s.length → s.get ⟨i, by sorry⟩ = infinite_sequence (start + i)

def concatenate_to_number (s1 s2 : List ℕ) : ℕ :=
  (s1 ++ s2).foldl (λ acc d => acc * 10 + d) 0

theorem exists_subsequences_forming_2520 :
  ∃ (s1 s2 : List ℕ),
    s1 ≠ [] ∧ s2 ≠ [] ∧
    is_subsequence s1 ∧
    is_subsequence s2 ∧
    concatenate_to_number s1 s2 = 2520 ∧
    2520 % 45 = 0 := by
  sorry

end exists_subsequences_forming_2520_l3331_333169


namespace blueberry_picking_difference_l3331_333183

theorem blueberry_picking_difference (annie kathryn ben : ℕ) : 
  annie = 8 →
  kathryn = annie + 2 →
  ben < kathryn →
  annie + kathryn + ben = 25 →
  kathryn - ben = 3 :=
by sorry

end blueberry_picking_difference_l3331_333183


namespace quadratic_equation_solution_l3331_333146

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 1 + Real.sqrt 6 ∧ x₁^2 - 2*x₁ - 5 = 0) ∧
  (x₂ = 1 - Real.sqrt 6 ∧ x₂^2 - 2*x₂ - 5 = 0) := by
  sorry

end quadratic_equation_solution_l3331_333146


namespace initial_average_age_l3331_333119

/-- Given a group of people with an unknown initial average age, 
    prove that when a new person joins and changes the average, 
    we can determine the initial average age. -/
theorem initial_average_age 
  (n : ℕ) 
  (new_person_age : ℕ) 
  (new_average : ℝ) 
  (h1 : n = 10)
  (h2 : new_person_age = 37)
  (h3 : new_average = 17) :
  ∃ (initial_average : ℝ),
    n * initial_average + new_person_age = (n + 1) * new_average ∧ 
    initial_average = 15 := by
  sorry

end initial_average_age_l3331_333119


namespace congruence_problem_l3331_333114

theorem congruence_problem (x : ℤ) : 
  (4 * x + 5) ≡ 3 [ZMOD 17] → (2 * x + 8) ≡ 7 [ZMOD 17] := by
  sorry

end congruence_problem_l3331_333114


namespace diophantine_equation_solutions_l3331_333106

theorem diophantine_equation_solutions (x y : ℕ) : 
  2^(2*x + 1) + 2^x + 1 = y^2 ↔ (x = 4 ∧ y = 23) ∨ (x = 0 ∧ y = 2) := by
  sorry

end diophantine_equation_solutions_l3331_333106


namespace factorial_difference_l3331_333123

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end factorial_difference_l3331_333123


namespace dihedral_angle_bounds_l3331_333189

/-- A regular pyramid with an n-sided polygonal base -/
structure RegularPyramid where
  n : ℕ
  base_sides : n > 2

/-- The dihedral angle between two adjacent lateral faces of a regular pyramid -/
def dihedral_angle (p : RegularPyramid) : ℝ :=
  sorry

/-- Theorem: The dihedral angle in a regular pyramid is bounded -/
theorem dihedral_angle_bounds (p : RegularPyramid) :
  (((p.n - 2) / p.n : ℝ) * Real.pi) < dihedral_angle p ∧ dihedral_angle p < Real.pi :=
sorry

end dihedral_angle_bounds_l3331_333189


namespace divisible_by_64_l3331_333163

theorem divisible_by_64 (n : ℕ+) : ∃ k : ℤ, (5 : ℤ)^n.val - 8*n.val^2 + 4*n.val - 1 = 64*k := by
  sorry

end divisible_by_64_l3331_333163


namespace danny_fish_tank_theorem_l3331_333162

/-- Represents the fish tank contents and sales --/
structure FishTank where
  initialGuppies : Nat
  initialAngelfish : Nat
  initialTigerSharks : Nat
  initialOscarFish : Nat
  soldGuppies : Nat
  soldAngelfish : Nat
  soldTigerSharks : Nat
  soldOscarFish : Nat

/-- Calculates the remaining fish in the tank --/
def remainingFish (tank : FishTank) : Nat :=
  (tank.initialGuppies + tank.initialAngelfish + tank.initialTigerSharks + tank.initialOscarFish) -
  (tank.soldGuppies + tank.soldAngelfish + tank.soldTigerSharks + tank.soldOscarFish)

/-- Theorem stating that the remaining fish in Danny's tank is 198 --/
theorem danny_fish_tank_theorem (tank : FishTank) 
  (h1 : tank.initialGuppies = 94)
  (h2 : tank.initialAngelfish = 76)
  (h3 : tank.initialTigerSharks = 89)
  (h4 : tank.initialOscarFish = 58)
  (h5 : tank.soldGuppies = 30)
  (h6 : tank.soldAngelfish = 48)
  (h7 : tank.soldTigerSharks = 17)
  (h8 : tank.soldOscarFish = 24) :
  remainingFish tank = 198 := by
  sorry

end danny_fish_tank_theorem_l3331_333162


namespace parallelogram_area_theorem_l3331_333186

/-- A point with integer coordinates -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A parallelogram defined by four lattice points -/
structure Parallelogram where
  v1 : LatticePoint
  v2 : LatticePoint
  v3 : LatticePoint
  v4 : LatticePoint

/-- Checks if a point is inside or on the edges of a parallelogram (excluding vertices) -/
def isInsideOrOnEdge (p : LatticePoint) (para : Parallelogram) : Prop :=
  sorry

/-- Calculates the area of a parallelogram -/
def area (para : Parallelogram) : ℚ :=
  sorry

theorem parallelogram_area_theorem (para : Parallelogram) :
  (∃ p : LatticePoint, isInsideOrOnEdge p para) → area para > 1 :=
by sorry

end parallelogram_area_theorem_l3331_333186


namespace max_value_at_two_l3331_333187

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

-- State the theorem
theorem max_value_at_two (c : ℝ) :
  (∀ x : ℝ, f c x ≤ f c 2) → c = 6 := by
  sorry

end max_value_at_two_l3331_333187


namespace ninth_row_fourth_number_l3331_333141

/-- The start of the i-th row in the sequence -/
def row_start (i : ℕ) : ℕ := 2 + 4 * 6 * (i - 1)

/-- The n-th number in the i-th row of the sequence -/
def seq_number (i n : ℕ) : ℕ := row_start i + 4 * (n - 1)

theorem ninth_row_fourth_number : seq_number 9 4 = 206 := by
  sorry

end ninth_row_fourth_number_l3331_333141


namespace carnival_game_ratio_l3331_333158

/-- The ratio of winners to losers in a carnival game -/
def carnival_ratio (winners losers : ℕ) : ℚ :=
  winners / losers

/-- Simplify a ratio by dividing both numerator and denominator by their GCD -/
def simplify_ratio (n d : ℕ) : ℚ :=
  (n / Nat.gcd n d) / (d / Nat.gcd n d)

theorem carnival_game_ratio :
  simplify_ratio 28 7 = 4 / 1 := by
  sorry

end carnival_game_ratio_l3331_333158


namespace prudence_weekend_sleep_l3331_333102

/-- Represents Prudence's sleep schedule over 4 weeks -/
structure SleepSchedule where
  weekdayNightSleep : ℕ  -- Hours of sleep on weeknights (Sun-Thu)
  weekendNapHours : ℕ    -- Hours of nap on weekend days
  totalSleepHours : ℕ    -- Total hours of sleep in 4 weeks
  weekdayNights : ℕ      -- Number of weekday nights in 4 weeks
  weekendNights : ℕ      -- Number of weekend nights in 4 weeks
  weekendDays : ℕ        -- Number of weekend days in 4 weeks

/-- Calculates the hours of sleep per night on weekends given Prudence's sleep schedule -/
def weekendNightSleep (s : SleepSchedule) : ℚ :=
  let weekdaySleep := s.weekdayNightSleep * s.weekdayNights
  let weekendNapSleep := s.weekendNapHours * s.weekendDays
  let remainingSleep := s.totalSleepHours - weekdaySleep - weekendNapSleep
  remainingSleep / s.weekendNights

/-- Theorem stating that Prudence sleeps 9 hours per night on weekends -/
theorem prudence_weekend_sleep (s : SleepSchedule)
  (h1 : s.weekdayNightSleep = 6)
  (h2 : s.weekendNapHours = 1)
  (h3 : s.totalSleepHours = 200)
  (h4 : s.weekdayNights = 20)
  (h5 : s.weekendNights = 8)
  (h6 : s.weekendDays = 8) :
  weekendNightSleep s = 9 := by
  sorry

#eval weekendNightSleep {
  weekdayNightSleep := 6,
  weekendNapHours := 1,
  totalSleepHours := 200,
  weekdayNights := 20,
  weekendNights := 8,
  weekendDays := 8
}

end prudence_weekend_sleep_l3331_333102


namespace sum_of_100th_terms_l3331_333171

/-- Given two arithmetic sequences {a_n} and {b_n} satisfying certain conditions,
    prove that the sum of their 100th terms is 383. -/
theorem sum_of_100th_terms (a b : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- a_n is arithmetic
  (∀ n m : ℕ, b (n + 1) - b n = b (m + 1) - b m) →  -- b_n is arithmetic
  a 5 + b 5 = 3 →
  a 9 + b 9 = 19 →
  a 100 + b 100 = 383 := by
sorry

end sum_of_100th_terms_l3331_333171


namespace dorothy_found_57_pieces_l3331_333188

/-- The number of sea glass pieces found by Dorothy -/
def dorothy_total (blanche_green blanche_red rose_red rose_blue : ℕ) : ℕ :=
  let dorothy_red := 2 * (blanche_red + rose_red)
  let dorothy_blue := 3 * rose_blue
  dorothy_red + dorothy_blue

/-- Theorem stating that Dorothy found 57 pieces of sea glass -/
theorem dorothy_found_57_pieces :
  dorothy_total 12 3 9 11 = 57 := by
  sorry

#eval dorothy_total 12 3 9 11

end dorothy_found_57_pieces_l3331_333188


namespace Q_not_subset_P_l3331_333192

-- Define set P
def P : Set ℝ := {y | y ≥ 0}

-- Define set Q
def Q : Set ℝ := {y | ∃ x, y = Real.log x}

-- Theorem statement
theorem Q_not_subset_P : ¬(Q ⊆ P ∧ P ∩ Q = Q) := by
  sorry

end Q_not_subset_P_l3331_333192


namespace diagonal_to_larger_base_ratio_l3331_333184

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of the smaller base -/
  smaller_base : ℝ
  /-- The length of the larger base -/
  larger_base : ℝ
  /-- The length of the diagonal -/
  diagonal : ℝ
  /-- The height of the trapezoid -/
  altitude : ℝ
  /-- The smaller base is positive -/
  smaller_base_pos : 0 < smaller_base
  /-- The larger base is greater than the smaller base -/
  base_order : smaller_base < larger_base
  /-- The smaller base equals half the diagonal -/
  smaller_base_eq_half_diagonal : smaller_base = diagonal / 2
  /-- The altitude equals two-thirds of the smaller base -/
  altitude_eq_two_thirds_smaller_base : altitude = 2 / 3 * smaller_base

/-- The ratio of the diagonal to the larger base in the specific isosceles trapezoid -/
theorem diagonal_to_larger_base_ratio (t : IsoscelesTrapezoid) : 
  t.diagonal / t.larger_base = 4 / 9 := by
  sorry

end diagonal_to_larger_base_ratio_l3331_333184


namespace parallelogram_count_is_392_l3331_333194

/-- Represents a parallelogram PQRS with the given properties -/
structure Parallelogram where
  q : ℕ+  -- x-coordinate of Q (also y-coordinate since Q is on y = x)
  s : ℕ+  -- x-coordinate of S
  m : ℕ   -- slope of line y = mx where S lies
  h_m_gt_one : m > 1
  h_area : (m - 1) * q * s = 250000

/-- Counts the number of valid parallelograms -/
def count_parallelograms : ℕ := sorry

/-- The main theorem stating that the count of valid parallelograms is 392 -/
theorem parallelogram_count_is_392 : count_parallelograms = 392 := by sorry

end parallelogram_count_is_392_l3331_333194


namespace max_sum_of_two_max_sum_is_zero_l3331_333134

def number_set : Finset Int := {1, -1, -2}

theorem max_sum_of_two (a b : Int) (ha : a ∈ number_set) (hb : b ∈ number_set) (hab : a ≠ b) :
  ∃ (x y : Int), x ∈ number_set ∧ y ∈ number_set ∧ x ≠ y ∧ x + y ≥ a + b :=
sorry

theorem max_sum_is_zero :
  ∃ (a b : Int), a ∈ number_set ∧ b ∈ number_set ∧ a ≠ b ∧
  (∀ (x y : Int), x ∈ number_set → y ∈ number_set → x ≠ y → a + b ≥ x + y) ∧
  a + b = 0 :=
sorry

end max_sum_of_two_max_sum_is_zero_l3331_333134


namespace max_intersections_for_given_points_l3331_333120

/-- The maximum number of intersection points in the first quadrant -/
def max_intersection_points (x_points y_points : ℕ) : ℕ :=
  (x_points * y_points * (x_points - 1) * (y_points - 1)) / 4

/-- Theorem stating the maximum number of intersection points for the given conditions -/
theorem max_intersections_for_given_points :
  max_intersection_points 5 3 = 30 := by sorry

end max_intersections_for_given_points_l3331_333120


namespace min_distance_exp_ln_curves_l3331_333198

/-- The minimum distance between a point on y = e^x and a point on y = ln x is √2 -/
theorem min_distance_exp_ln_curves : ∃ (d : ℝ),
  d = Real.sqrt 2 ∧
  ∀ (x₁ x₂ : ℝ),
    let P := (x₁, Real.exp x₁)
    let Q := (x₂, Real.log x₂)
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by sorry

end min_distance_exp_ln_curves_l3331_333198


namespace total_fruits_l3331_333167

theorem total_fruits (total_baskets : ℕ) 
                     (apple_baskets orange_baskets : ℕ) 
                     (apples_per_basket oranges_per_basket pears_per_basket : ℕ) : 
  total_baskets = 127 →
  apple_baskets = 79 →
  orange_baskets = 30 →
  apples_per_basket = 75 →
  oranges_per_basket = 143 →
  pears_per_basket = 56 →
  (apple_baskets * apples_per_basket + 
   orange_baskets * oranges_per_basket + 
   (total_baskets - apple_baskets - orange_baskets) * pears_per_basket) = 11223 :=
by
  sorry

#check total_fruits

end total_fruits_l3331_333167


namespace average_income_proof_l3331_333104

def income_days : Nat := 5

def daily_incomes : List ℝ := [400, 250, 650, 400, 500]

theorem average_income_proof :
  (daily_incomes.sum / income_days : ℝ) = 440 := by
  sorry

end average_income_proof_l3331_333104


namespace sam_seashells_l3331_333159

/-- The number of seashells Sam has after giving some to Joan -/
def remaining_seashells (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

theorem sam_seashells : remaining_seashells 35 18 = 17 := by
  sorry

end sam_seashells_l3331_333159


namespace plan1_more_cost_effective_when_sessions_gt_8_l3331_333142

/-- Represents the cost of a fitness plan based on the number of sessions -/
structure FitnessPlan where
  fixedFee : ℕ
  perSessionFee : ℕ

/-- Calculates the total cost for a given plan and number of sessions -/
def totalCost (plan : FitnessPlan) (sessions : ℕ) : ℕ :=
  plan.fixedFee + plan.perSessionFee * sessions

/-- Theorem: Plan 1 is more cost-effective than Plan 2 when sessions > 8 -/
theorem plan1_more_cost_effective_when_sessions_gt_8
  (plan1 : FitnessPlan)
  (plan2 : FitnessPlan)
  (h1 : plan1.fixedFee = 80 ∧ plan1.perSessionFee = 10)
  (h2 : plan2.fixedFee = 0 ∧ plan2.perSessionFee = 20)
  : ∀ sessions, sessions > 8 → totalCost plan1 sessions < totalCost plan2 sessions := by
  sorry

#check plan1_more_cost_effective_when_sessions_gt_8

end plan1_more_cost_effective_when_sessions_gt_8_l3331_333142


namespace inequality_proofs_l3331_333122

theorem inequality_proofs 
  (h : ∀ x > 0, 1 / (1 + x) < Real.log (1 + 1 / x) ∧ Real.log (1 + 1 / x) < 1 / x) :
  (1 + 1/2 + 1/3 + 1/4 + 1/5 + 1/6 + 1/7 > Real.log 8) ∧
  (1/2 + 1/3 + 1/4 + 1/5 + 1/6 + 1/7 + 1/8 < Real.log 8) ∧
  ((1 : ℝ) / 1 + 8 / 8 + 28 / 64 + 56 / 512 + 70 / 4096 + 56 / 32768 + 28 / 262144 + 8 / 2097152 + 1 / 16777216 < Real.exp 1) :=
by sorry

end inequality_proofs_l3331_333122


namespace rabbit_area_l3331_333156

theorem rabbit_area (ear_area : ℝ) (total_area : ℝ) : 
  ear_area = 10 → ear_area = (1/8) * total_area → total_area = 80 := by
  sorry

end rabbit_area_l3331_333156


namespace origin_outside_circle_l3331_333151

theorem origin_outside_circle (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 + 2*a*x + 2*y + (a-1)^2 = 0}
  (0, 0) ∉ circle ∧ ∃ (p : ℝ × ℝ), p ∈ circle ∧ dist p (0, 0) < dist (0, 0) p := by
  sorry

end origin_outside_circle_l3331_333151


namespace remainder_problem_l3331_333165

theorem remainder_problem : (98 * 103 + 7) % 12 = 1 := by sorry

end remainder_problem_l3331_333165


namespace greatest_divisor_with_remainders_l3331_333129

theorem greatest_divisor_with_remainders (a b r1 r2 : ℕ) (ha : a = 6215) (hb : b = 7373) (hr1 : r1 = 23) (hr2 : r2 = 29) :
  Nat.gcd (a - r1) (b - r2) = 96 := by
  sorry

end greatest_divisor_with_remainders_l3331_333129


namespace max_m_value_l3331_333168

theorem max_m_value (m : ℝ) : 
  let A := {x : ℝ | m + 1 ≤ x ∧ x ≤ 3 * m - 1}
  let B := {x : ℝ | 1 ≤ x ∧ x ≤ 10}
  A ⊆ B → m ≤ 11 / 3 :=
by sorry

end max_m_value_l3331_333168


namespace denominator_problem_l3331_333177

theorem denominator_problem (numerator denominator : ℤ) : 
  denominator = numerator - 4 →
  numerator + 6 = 3 * denominator →
  denominator = 5 := by
sorry

end denominator_problem_l3331_333177


namespace monotonicity_and_tangent_line_and_max_k_l3331_333191

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 2

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a

theorem monotonicity_and_tangent_line_and_max_k :
  -- Part 1: Monotonicity of f(x)
  (∀ a : ℝ, a ≤ 0 → StrictMono (f a)) ∧
  (∀ a : ℝ, a > 0 → 
    (∀ x y : ℝ, x < y → y < Real.log a → f a y < f a x) ∧
    (∀ x y : ℝ, Real.log a < x → x < y → f a x < f a y)) ∧
  
  -- Part 2: Tangent line condition
  (∀ a : ℝ, (∃ x₀ : ℝ, f_deriv a x₀ = Real.exp 1 ∧ 
    f a x₀ = Real.exp x₀ - 2) → a = 0) ∧
  
  -- Part 3: Maximum value of k
  (∀ k : ℤ, (∀ x : ℝ, x > 0 → (x - ↑k) * (f_deriv 1 x) + x + 1 > 0) → 
    k ≤ 2) ∧
  (∃ x : ℝ, x > 0 ∧ (x - 2) * (f_deriv 1 x) + x + 1 > 0)
  := by sorry

end monotonicity_and_tangent_line_and_max_k_l3331_333191


namespace lines_parallel_iff_a_eq_zero_l3331_333150

/-- Two lines in the form of x-2ay=1 and 2x-2ay=1 are parallel if and only if a=0 -/
theorem lines_parallel_iff_a_eq_zero (a : ℝ) :
  (∀ x y : ℝ, x - 2*a*y = 1 ↔ 2*x - 2*a*y = 1) ↔ a = 0 := by
  sorry

end lines_parallel_iff_a_eq_zero_l3331_333150


namespace other_number_proof_l3331_333133

theorem other_number_proof (A B : ℕ) : 
  A > 0 → B > 0 →
  Nat.lcm A B = 9699690 →
  Nat.gcd A B = 385 →
  A = 44530 →
  B = 83891 := by
sorry

end other_number_proof_l3331_333133


namespace hyperbola_asymptote_l3331_333154

/-- The asymptote of a hyperbola with specific properties -/
theorem hyperbola_asymptote (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let C : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  let asymptote_parallel : ℝ → ℝ → Prop := λ x y => y = (b / a) * (x - c)
  ∃ (P : ℝ × ℝ), 
    C P.1 P.2 ∧ 
    asymptote_parallel P.1 P.2 ∧
    ((P.1 + c) * (P.1 - c) + P.2^2 = 0) →
    (∀ (x y : ℝ), y = 2 * x ∨ y = -2 * x ↔ x^2 / a^2 - y^2 / b^2 = 0) :=
by sorry

end hyperbola_asymptote_l3331_333154


namespace bank_teller_problem_l3331_333124

theorem bank_teller_problem (total_bills : ℕ) (total_value : ℕ) 
  (h1 : total_bills = 54)
  (h2 : total_value = 780) :
  ∃ (five_dollar_bills twenty_dollar_bills : ℕ),
    five_dollar_bills + twenty_dollar_bills = total_bills ∧
    5 * five_dollar_bills + 20 * twenty_dollar_bills = total_value ∧
    five_dollar_bills = 20 := by
  sorry

end bank_teller_problem_l3331_333124


namespace rectangle_area_perimeter_relation_l3331_333121

theorem rectangle_area_perimeter_relation (x : ℝ) :
  let length : ℝ := 4 * x
  let width : ℝ := x + 10
  let area : ℝ := length * width
  let perimeter : ℝ := 2 * (length + width)
  (area = 2 * perimeter) → x = (Real.sqrt 41 - 1) / 2 := by
  sorry

end rectangle_area_perimeter_relation_l3331_333121


namespace probability_of_valid_sequence_l3331_333137

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def validSequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of possible sequences of length n -/
def totalSequences (n : ℕ) : ℕ := 2^n

/-- The probability of a valid sequence of length 8 -/
def probability : ℚ := validSequences 8 / totalSequences 8

theorem probability_of_valid_sequence :
  probability = 55 / 256 := by sorry

end probability_of_valid_sequence_l3331_333137


namespace octal_minus_base9_equals_152294_l3331_333130

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

theorem octal_minus_base9_equals_152294 :
  let octal_num := [5, 4, 3, 2, 1, 0]
  let base9_num := [4, 3, 2, 1, 0]
  base_to_decimal octal_num 8 - base_to_decimal base9_num 9 = 152294 := by
  sorry

end octal_minus_base9_equals_152294_l3331_333130


namespace set_difference_equals_interval_l3331_333117

def M : Set ℝ := {x | x^2 + x - 12 ≤ 0}

def N : Set ℝ := {y | ∃ x, y = 3^x ∧ x ≤ 1}

theorem set_difference_equals_interval :
  {x | x ∈ M ∧ x ∉ N} = Set.Ico (-4) 0 := by sorry

end set_difference_equals_interval_l3331_333117


namespace complex_number_problem_l3331_333139

/-- Given complex numbers z₁ and z₂ satisfying certain conditions, prove that z₂ = 6 + 2i -/
theorem complex_number_problem (z₁ z₂ : ℂ) : 
  ((z₁ - 2) * Complex.I = 1 + Complex.I) →
  (z₂.im = 2) →
  ((z₁ * z₂).im = 0) →
  z₂ = 6 + 2 * Complex.I :=
by
  sorry


end complex_number_problem_l3331_333139


namespace power_plus_one_not_divisible_by_power_minus_one_l3331_333100

theorem power_plus_one_not_divisible_by_power_minus_one (x y : ℕ) (h : y > 2) :
  ¬ (2^y - 1 ∣ 2^x + 1) := by
  sorry

end power_plus_one_not_divisible_by_power_minus_one_l3331_333100


namespace passes_through_fixed_point_not_in_fourth_quadrant_min_area_and_equation_l3331_333126

/-- Definition of the line l with parameter k -/
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0

/-- The fixed point that the line passes through -/
def fixed_point : ℝ × ℝ := (-2, 1)

/-- Theorem 1: The line passes through the fixed point for all real k -/
theorem passes_through_fixed_point (k : ℝ) :
  line_l k (fixed_point.1) (fixed_point.2) := by sorry

/-- Theorem 2: The line does not pass through the fourth quadrant iff k ≥ 0 -/
theorem not_in_fourth_quadrant (k : ℝ) :
  (∀ x y, x > 0 → y < 0 → ¬line_l k x y) ↔ k ≥ 0 := by sorry

/-- Function to calculate the area of the triangle formed by the line's intersections -/
noncomputable def triangle_area (k : ℝ) : ℝ :=
  if k ≠ 0 then
    (1 + 2 * k) * ((1 + 2 * k) / k) / 2
  else 0

/-- Theorem 3: The minimum area of the triangle is 4, occurring when k = 1/2 -/
theorem min_area_and_equation :
  (∀ k, k > 0 → triangle_area k ≥ 4) ∧
  triangle_area (1/2) = 4 ∧
  line_l (1/2) x y ↔ x - 2 * y + 4 = 0 := by sorry

end passes_through_fixed_point_not_in_fourth_quadrant_min_area_and_equation_l3331_333126


namespace quadratic_equation_solution_l3331_333101

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x - 2
  ∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 3 ∧ x₂ = 1 - Real.sqrt 3 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end quadratic_equation_solution_l3331_333101


namespace sum_of_odd_coefficients_l3331_333144

theorem sum_of_odd_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (∀ x, (2*x + 1)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₃ + a₅ = 122 := by
sorry

end sum_of_odd_coefficients_l3331_333144


namespace geometric_progression_proof_l3331_333107

theorem geometric_progression_proof (x : ℝ) :
  (30 + x)^2 = (10 + x) * (90 + x) →
  x = 0 ∧ (30 + x) / (10 + x) = 3 := by
  sorry

end geometric_progression_proof_l3331_333107


namespace boys_on_playground_l3331_333193

/-- The number of boys on a playground, given the total number of children and the number of girls. -/
def number_of_boys (total_children : ℕ) (number_of_girls : ℕ) : ℕ :=
  total_children - number_of_girls

/-- Theorem stating that the number of boys on the playground is 40. -/
theorem boys_on_playground : number_of_boys 117 77 = 40 := by
  sorry

end boys_on_playground_l3331_333193


namespace unique_age_sum_of_digits_l3331_333176

theorem unique_age_sum_of_digits : ∃! y : ℕ,
  1900 ≤ y ∧ y < 2000 ∧
  1988 - y = 22 ∧
  1988 - y = (y / 1000) + ((y / 100) % 10) + ((y / 10) % 10) + (y % 10) :=
by sorry

end unique_age_sum_of_digits_l3331_333176


namespace spinner_probability_l3331_333175

theorem spinner_probability (p_D p_E p_FG : ℚ) : 
  p_D = 1/4 → p_E = 1/3 → p_D + p_E + p_FG = 1 → p_FG = 5/12 := by
  sorry

end spinner_probability_l3331_333175


namespace spade_operation_l3331_333113

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_operation : (5 : ℝ) * (spade 2 (spade 6 9)) = 5 := by
  sorry

end spade_operation_l3331_333113


namespace wedge_volume_l3331_333182

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (log_diameter : ℝ) (cut_angle : ℝ) : 
  log_diameter = 12 →
  cut_angle = 45 →
  (π * (log_diameter / 2)^2 * log_diameter) / 2 = 216 * π := by
  sorry

end wedge_volume_l3331_333182


namespace subtract_inequality_l3331_333128

theorem subtract_inequality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : a - 3 < b - 3 := by
  sorry

end subtract_inequality_l3331_333128


namespace chris_bowling_score_l3331_333145

/-- Proves Chris's bowling score given Sarah and Greg's score conditions -/
theorem chris_bowling_score (sarah_score greg_score : ℕ) : 
  sarah_score = greg_score + 60 →
  (sarah_score + greg_score) / 2 = 110 →
  let avg := (sarah_score + greg_score) / 2
  let chris_score := (avg * 120) / 100
  chris_score = 132 := by
sorry

end chris_bowling_score_l3331_333145


namespace min_value_theorem_min_value_is_four_l3331_333135

theorem min_value_theorem (x : ℝ) (h : x ≥ 2) :
  (∀ y : ℝ, y ≥ 2 → x + 4/x ≤ y + 4/y) ↔ x = 2 :=
by sorry

theorem min_value_is_four (x : ℝ) (h : x ≥ 2) :
  x + 4/x ≥ 4 :=
by sorry

end min_value_theorem_min_value_is_four_l3331_333135


namespace correct_paint_time_equation_l3331_333160

/-- Represents the time needed for three people to paint a room together, given their individual rates and a break time. -/
def paint_time (rate1 rate2 rate3 break_time : ℝ) (t : ℝ) : Prop :=
  (1 / rate1 + 1 / rate2 + 1 / rate3) * (t - break_time) = 1

/-- Theorem stating that the equation correctly represents the painting time for Doug, Dave, and Ralph. -/
theorem correct_paint_time_equation :
  ∀ t : ℝ, paint_time 6 8 12 1.5 t ↔ (1/6 + 1/8 + 1/12) * (t - 1.5) = 1 :=
by sorry

end correct_paint_time_equation_l3331_333160


namespace circular_track_circumference_l3331_333109

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem circular_track_circumference 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (meeting_time_minutes : ℝ) 
  (h1 : speed1 = 20) 
  (h2 : speed2 = 16) 
  (h3 : meeting_time_minutes = 36) : 
  speed1 + speed2 * meeting_time_minutes / 60 = 21.6 := by
  sorry

#check circular_track_circumference

end circular_track_circumference_l3331_333109


namespace at_least_two_inequalities_hold_l3331_333157

theorem at_least_two_inequalities_hold (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + b + c ≥ a * b * c) : 
  (2 / a + 3 / b + 6 / c ≥ 6 ∧ 2 / b + 3 / c + 6 / a ≥ 6) ∨
  (2 / b + 3 / c + 6 / a ≥ 6 ∧ 2 / c + 3 / a + 6 / b ≥ 6) ∨
  (2 / c + 3 / a + 6 / b ≥ 6 ∧ 2 / a + 3 / b + 6 / c ≥ 6) :=
by sorry

end at_least_two_inequalities_hold_l3331_333157


namespace tan_315_and_radian_conversion_l3331_333161

theorem tan_315_and_radian_conversion :
  Real.tan (315 * π / 180) = -1 ∧ 315 * π / 180 = 7 * π / 4 := by
  sorry

end tan_315_and_radian_conversion_l3331_333161


namespace remainder_count_l3331_333197

theorem remainder_count : 
  (Finset.filter (fun n => Nat.mod 2017 n = 1 ∨ Nat.mod 2017 n = 2) (Finset.range 2018)).card = 43 := by
  sorry

end remainder_count_l3331_333197


namespace fibonacci_sum_convergence_l3331_333152

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_sum_convergence :
  let S : ℝ := ∑' n, (fibonacci n : ℝ) / 5^n
  S = 5/19 := by sorry

end fibonacci_sum_convergence_l3331_333152


namespace cyclical_sequence_value_of_3_cyclical_sequence_properties_l3331_333195

def cyclical_sequence (n : ℕ) : ℕ :=
  match n % 5 with
  | 1 => 6
  | 2 => 12
  | 3 => 18  -- This is what we want to prove
  | 4 => 24
  | 0 => 30
  | _ => 0   -- This case should never occur

theorem cyclical_sequence_value_of_3 :
  cyclical_sequence 3 = 18 :=
by
  sorry

theorem cyclical_sequence_properties :
  (cyclical_sequence 1 = 6) ∧
  (cyclical_sequence 2 = 12) ∧
  (cyclical_sequence 4 = 24) ∧
  (cyclical_sequence 5 = 30) ∧
  (cyclical_sequence 6 = 1) :=
by
  sorry

end cyclical_sequence_value_of_3_cyclical_sequence_properties_l3331_333195


namespace system_solution_l3331_333164

theorem system_solution : 
  ∀ x y : ℝ, 
  (x + y + Real.sqrt (x * y) = 28 ∧ x^2 + y^2 + x * y = 336) ↔ 
  ((x = 4 ∧ y = 16) ∨ (x = 16 ∧ y = 4)) := by
sorry

end system_solution_l3331_333164


namespace biscuit_banana_cost_ratio_l3331_333108

-- Define variables
variable (b : ℚ) -- Cost of one biscuit
variable (x : ℚ) -- Cost of one banana

-- Define Susie's and Daisy's expenditures
def susie_expenditure : ℚ := 6 * b + 4 * x
def daisy_expenditure : ℚ := 4 * b + 20 * x

-- State the theorem
theorem biscuit_banana_cost_ratio :
  (susie_expenditure b x = daisy_expenditure b x / 3) →
  (b / x = 4 / 7) := by
  sorry

end biscuit_banana_cost_ratio_l3331_333108


namespace range_of_x_l3331_333118

open Set

def S : Set ℝ := {x | x ∈ Icc 2 5 ∨ x < 1 ∨ x > 4}

theorem range_of_x (h : ¬ ∀ x, x ∈ S) : 
  {x : ℝ | x ∈ Ico 1 2} = {x : ℝ | ¬ (x ∈ S)} := by sorry

end range_of_x_l3331_333118


namespace polynomial_coefficient_sum_of_squares_l3331_333125

theorem polynomial_coefficient_sum_of_squares 
  (a b c d e f : ℤ) 
  (h : ∀ x : ℝ, 8 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) : 
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 356 := by
  sorry

end polynomial_coefficient_sum_of_squares_l3331_333125


namespace time_puzzle_l3331_333111

theorem time_puzzle : 
  ∃ h : ℝ, h = (12 - h) + (2/5) * h ∧ h = 7.5 := by sorry

end time_puzzle_l3331_333111


namespace quadratic_has_two_distinct_roots_l3331_333170

theorem quadratic_has_two_distinct_roots 
  (a b c : ℝ) 
  (h1 : 5*a + 3*b + 2*c = 0) 
  (h2 : a ≠ 0) : 
  ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 :=
sorry

end quadratic_has_two_distinct_roots_l3331_333170


namespace elephant_received_503_pills_l3331_333127

/-- The number of pills given to four animals by Dr. Aibolit -/
def total_pills : ℕ := 2006

/-- The number of pills received by the crocodile -/
def crocodile_pills : ℕ := sorry

/-- The number of pills received by the rhinoceros -/
def rhinoceros_pills : ℕ := crocodile_pills + 1

/-- The number of pills received by the hippopotamus -/
def hippopotamus_pills : ℕ := rhinoceros_pills + 1

/-- The number of pills received by the elephant -/
def elephant_pills : ℕ := hippopotamus_pills + 1

/-- Theorem stating that the elephant received 503 pills -/
theorem elephant_received_503_pills : 
  crocodile_pills + rhinoceros_pills + hippopotamus_pills + elephant_pills = total_pills ∧ 
  elephant_pills = 503 := by
  sorry

end elephant_received_503_pills_l3331_333127


namespace indexCardsCostForCarl_l3331_333196

/-- Represents the cost of index cards for Carl's students. -/
def indexCardsCost (
  sixthGradeCards : ℕ
  ) (seventhGradeCards : ℕ
  ) (eighthGradeCards : ℕ
  ) (periodsPerDay : ℕ
  ) (sixthGradersPerPeriod : ℕ
  ) (seventhGradersPerPeriod : ℕ
  ) (eighthGradersPerPeriod : ℕ
  ) (cardsPerPack : ℕ
  ) (costPerPack : ℕ
  ) : ℕ :=
  let totalCards := 
    (sixthGradeCards * sixthGradersPerPeriod + 
     seventhGradeCards * seventhGradersPerPeriod + 
     eighthGradeCards * eighthGradersPerPeriod) * periodsPerDay
  let packsNeeded := (totalCards + cardsPerPack - 1) / cardsPerPack
  packsNeeded * costPerPack

/-- Theorem stating the total cost of index cards for Carl's students. -/
theorem indexCardsCostForCarl : 
  indexCardsCost 8 10 12 6 20 25 30 50 3 = 279 := by
  sorry

end indexCardsCostForCarl_l3331_333196


namespace paul_pencil_sales_l3331_333155

def pencils_sold (daily_production : ℕ) (work_days : ℕ) (starting_stock : ℕ) (ending_stock : ℕ) : ℕ :=
  daily_production * work_days + starting_stock - ending_stock

theorem paul_pencil_sales : pencils_sold 100 5 80 230 = 350 := by
  sorry

end paul_pencil_sales_l3331_333155


namespace eliminated_team_size_is_21_l3331_333179

/-- Represents a team in the competition -/
structure Team where
  size : ℕ
  is_girls : Bool

/-- Represents the state of the competition -/
structure Competition where
  teams : List Team
  eliminated_team_size : ℕ

def Competition.remaining_teams (c : Competition) : List Team :=
  c.teams.filter (λ t => t.size ≠ c.eliminated_team_size)

def Competition.total_players (c : Competition) : ℕ :=
  c.teams.map (λ t => t.size) |>.sum

def Competition.remaining_players (c : Competition) : ℕ :=
  c.total_players - c.eliminated_team_size

def Competition.boys_count (c : Competition) : ℕ :=
  c.remaining_teams.filter (λ t => ¬t.is_girls) |>.map (λ t => t.size) |>.sum

def Competition.girls_count (c : Competition) : ℕ :=
  c.remaining_players - c.boys_count

theorem eliminated_team_size_is_21 (c : Competition) : c.eliminated_team_size = 21 :=
  by
  have team_sizes : c.teams.map (λ t => t.size) = [9, 15, 17, 19, 21] := sorry
  have total_five_teams : c.teams.length = 5 := sorry
  have eliminated_is_girls : c.teams.filter (λ t => t.size = c.eliminated_team_size) |>.all (λ t => t.is_girls) := sorry
  have remaining_girls_triple_boys : c.girls_count = 3 * c.boys_count := sorry
  sorry

#check eliminated_team_size_is_21

end eliminated_team_size_is_21_l3331_333179


namespace mark_parking_tickets_l3331_333172

theorem mark_parking_tickets (total_tickets : ℕ) (sarah_speeding : ℕ)
  (h1 : total_tickets = 24)
  (h2 : sarah_speeding = 6) :
  ∃ (mark_parking sarah_parking : ℕ),
    mark_parking = 2 * sarah_parking ∧
    total_tickets = sarah_parking + mark_parking + 2 * sarah_speeding ∧
    mark_parking = 8 := by
  sorry

end mark_parking_tickets_l3331_333172


namespace fraction_equality_implies_product_l3331_333180

theorem fraction_equality_implies_product (a b : ℝ) : 
  a / 2 = 3 / b → a * b = 6 := by
  sorry

end fraction_equality_implies_product_l3331_333180


namespace function_equality_l3331_333116

theorem function_equality (f : ℕ+ → ℕ+) 
  (h : ∀ m n : ℕ+, m^2 + f n^2 + (m - f n)^2 ≥ f m^2 + n^2) : 
  ∀ n : ℕ+, f n = n :=
sorry

end function_equality_l3331_333116


namespace right_triangle_area_l3331_333140

theorem right_triangle_area (a b c : ℝ) (h1 : a = 48) (h2 : c = 50) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 336 := by
  sorry

end right_triangle_area_l3331_333140


namespace original_average_age_proof_l3331_333132

theorem original_average_age_proof (N : ℕ) (A : ℝ) : 
  A = 50 →
  (N * A + 12 * 32) / (N + 12) = 46 →
  A = 50 := by
sorry

end original_average_age_proof_l3331_333132


namespace kirsty_model_purchase_l3331_333185

/-- The number of models Kirsty can buy at the new price -/
def new_quantity : ℕ := 27

/-- The initial price of each model in dollars -/
def initial_price : ℚ := 45/100

/-- The new price of each model in dollars -/
def new_price : ℚ := 1/2

/-- The initial number of models Kirsty planned to buy -/
def initial_quantity : ℕ := 30

theorem kirsty_model_purchase :
  initial_quantity * initial_price = new_quantity * new_price :=
sorry


end kirsty_model_purchase_l3331_333185


namespace race_end_count_l3331_333131

/-- Represents the total number of people in all cars at the end of a race with given conditions. -/
def total_people_at_end (num_cars : ℕ) (initial_people_per_car : ℕ) 
  (first_quarter_gain : ℕ) (half_way_gain : ℕ) (three_quarter_gain : ℕ) : ℕ :=
  num_cars * (initial_people_per_car + first_quarter_gain + half_way_gain + three_quarter_gain)

/-- Theorem stating that under the given race conditions, the total number of people at the end is 450. -/
theorem race_end_count : 
  total_people_at_end 50 4 2 2 1 = 450 := by
  sorry

#eval total_people_at_end 50 4 2 2 1

end race_end_count_l3331_333131


namespace anusha_share_l3331_333143

theorem anusha_share (total : ℕ) (a b e : ℚ) : 
  total = 378 →
  12 * a = 8 * b →
  12 * a = 6 * e →
  a + b + e = total →
  a = 84 := by
sorry

end anusha_share_l3331_333143


namespace jolene_babysitting_charge_l3331_333190

theorem jolene_babysitting_charge 
  (num_families : ℕ) 
  (num_cars : ℕ) 
  (car_wash_fee : ℚ) 
  (total_raised : ℚ) :
  num_families = 4 →
  num_cars = 5 →
  car_wash_fee = 12 →
  total_raised = 180 →
  (num_families : ℚ) * (total_raised - num_cars * car_wash_fee) / num_families = 30 := by
  sorry

end jolene_babysitting_charge_l3331_333190


namespace min_n_for_constant_term_l3331_333153

theorem min_n_for_constant_term (x : ℝ) : 
  (∃ (n : ℕ), n > 0 ∧ (∃ (r : ℕ), r ≤ n ∧ 3 * n = 7 * r)) ∧
  (∀ (m : ℕ), m > 0 → (∃ (r : ℕ), r ≤ m ∧ 3 * m = 7 * r) → m ≥ 7) :=
by sorry

end min_n_for_constant_term_l3331_333153


namespace rhombus_perimeter_l3331_333138

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 40 := by
  sorry

end rhombus_perimeter_l3331_333138


namespace tanner_money_left_l3331_333148

def savings : List ℝ := [17, 48, 25, 55]
def video_game_price : ℝ := 49
def shoes_price : ℝ := 65
def discount_rate : ℝ := 0.1
def tax_rate : ℝ := 0.05

def total_savings : ℝ := savings.sum

def discounted_video_game_price : ℝ := video_game_price * (1 - discount_rate)

def total_cost_before_tax : ℝ := discounted_video_game_price + shoes_price

def sales_tax : ℝ := total_cost_before_tax * tax_rate

def total_cost_with_tax : ℝ := total_cost_before_tax + sales_tax

def money_left : ℝ := total_savings - total_cost_with_tax

theorem tanner_money_left :
  ∃ (ε : ℝ), money_left = 30.44 + ε ∧ abs ε < 0.005 := by
  sorry

end tanner_money_left_l3331_333148


namespace binomial_expansion_problem_l3331_333136

theorem binomial_expansion_problem (a b : ℝ) : 
  (∃ c d e : ℝ, (1 + a * x)^5 = 1 + 10*x + b*x^2 + c*x^3 + d*x^4 + a^5*x^5) → 
  a - b = -38 := by
sorry

end binomial_expansion_problem_l3331_333136


namespace largest_digit_divisible_by_six_l3331_333112

theorem largest_digit_divisible_by_six :
  ∃ (N : ℕ), N ≤ 9 ∧ 
  (∀ (M : ℕ), M ≤ 9 → 6 ∣ (5678 * 10 + M) → M ≤ N) ∧
  (6 ∣ (5678 * 10 + N)) :=
by sorry

end largest_digit_divisible_by_six_l3331_333112


namespace largest_integer_with_remainder_l3331_333105

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 120 ∧ 
  n % 8 = 5 ∧ 
  ∀ m : ℕ, m < 120 ∧ m % 8 = 5 → m ≤ n → 
  n = 117 :=
by sorry

end largest_integer_with_remainder_l3331_333105


namespace megans_vacation_pictures_l3331_333166

theorem megans_vacation_pictures (zoo_pics museum_pics deleted_pics : ℕ) 
  (h1 : zoo_pics = 15)
  (h2 : museum_pics = 18)
  (h3 : deleted_pics = 31) :
  zoo_pics + museum_pics - deleted_pics = 2 := by
  sorry

end megans_vacation_pictures_l3331_333166


namespace ellipse_foci_l3331_333174

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

/-- The coordinates of a focus -/
def focus_coordinate : ℝ × ℝ := (4, 0)

/-- Theorem stating that the foci of the given ellipse are at (±4, 0) -/
theorem ellipse_foci :
  ∀ (x y : ℝ), ellipse_equation x y → 
    (x = focus_coordinate.1 ∧ y = focus_coordinate.2) ∨
    (x = -focus_coordinate.1 ∧ y = focus_coordinate.2) := by
  sorry

end ellipse_foci_l3331_333174


namespace solution_is_correct_l3331_333173

/-- The imaginary unit i such that i^2 = -1 -/
noncomputable def i : ℂ := Complex.I

/-- The equation to be solved -/
def equation (z : ℂ) : Prop := 2 * z + (5 - 3 * i) = 6 + 11 * i

/-- The theorem stating that 1/2 + 7i is the solution to the equation -/
theorem solution_is_correct : equation (1/2 + 7 * i) := by
  sorry

end solution_is_correct_l3331_333173


namespace total_pictures_correct_l3331_333147

/-- The number of pictures Bianca uploaded to Facebook -/
def total_pictures : ℕ := 33

/-- The number of pictures in the first album -/
def first_album_pictures : ℕ := 27

/-- The number of additional albums -/
def additional_albums : ℕ := 3

/-- The number of pictures in each additional album -/
def pictures_per_additional_album : ℕ := 2

/-- Theorem stating that the total number of pictures is correct -/
theorem total_pictures_correct : 
  total_pictures = first_album_pictures + additional_albums * pictures_per_additional_album := by
  sorry

end total_pictures_correct_l3331_333147


namespace p_true_q_false_l3331_333115

theorem p_true_q_false (h1 : ¬(p ∧ q)) (h2 : ¬(¬p ∨ q)) : p ∧ ¬q :=
by sorry

end p_true_q_false_l3331_333115


namespace equation_solution_l3331_333181

theorem equation_solution :
  let y : ℚ := 20 / 7
  2 / y + (3 / y) / (6 / y) = 1.2 := by
  sorry

end equation_solution_l3331_333181


namespace prob_less_than_two_defective_l3331_333199

/-- The probability of selecting fewer than 2 defective products -/
theorem prob_less_than_two_defective (total : Nat) (defective : Nat) (selected : Nat) 
  (h1 : total = 10) (h2 : defective = 3) (h3 : selected = 2) : 
  (Nat.choose (total - defective) selected + 
   Nat.choose (total - defective) (selected - 1) * Nat.choose defective 1) / 
  Nat.choose total selected = 14 / 15 := by
  sorry

end prob_less_than_two_defective_l3331_333199
