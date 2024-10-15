import Mathlib

namespace NUMINAMATH_CALUDE_derivative_x_squared_sin_x_l3632_363237

/-- The derivative of x^2 * sin(x) is 2x * sin(x) + x^2 * cos(x) -/
theorem derivative_x_squared_sin_x (x : ℝ) :
  deriv (fun x => x^2 * Real.sin x) x = 2 * x * Real.sin x + x^2 * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_squared_sin_x_l3632_363237


namespace NUMINAMATH_CALUDE_trishul_investment_percentage_l3632_363204

theorem trishul_investment_percentage (vishal trishul raghu : ℝ) : 
  vishal = 1.1 * trishul →
  trishul + vishal + raghu = 5780 →
  raghu = 2000 →
  (raghu - trishul) / raghu = 0.1 := by
sorry

end NUMINAMATH_CALUDE_trishul_investment_percentage_l3632_363204


namespace NUMINAMATH_CALUDE_two_person_island_puzzle_l3632_363239

/-- Represents a person who can either be a liar or a truth-teller -/
inductive Person
  | Liar
  | TruthTeller

/-- The statement of a person about the number of truth-tellers -/
def statement (p : Person) (actual_truth_tellers : Nat) : Nat :=
  match p with
  | Person.Liar => actual_truth_tellers - 1  -- A liar reduces the number by one
  | Person.TruthTeller => actual_truth_tellers

/-- The main theorem -/
theorem two_person_island_puzzle (total_population : Nat) (liars truth_tellers : Nat)
    (h1 : total_population = liars + truth_tellers)
    (h2 : liars = 1000)
    (h3 : truth_tellers = 1000)
    (person1 person2 : Person)
    (h4 : statement person1 truth_tellers ≠ statement person2 truth_tellers) :
    person1 = Person.Liar ∧ person2 = Person.TruthTeller :=
  sorry


end NUMINAMATH_CALUDE_two_person_island_puzzle_l3632_363239


namespace NUMINAMATH_CALUDE_equation_solution_l3632_363298

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3632_363298


namespace NUMINAMATH_CALUDE_inequality_permutation_l3632_363215

theorem inequality_permutation (a b c d : ℝ) (h : a * b * c * d > 0) :
  ∃ (x y z w : ℝ), (({x, y, z, w} : Finset ℝ) = {a, b, c, d}) ∧
  (2 * (x * z + y * w)^2 > (x^2 + y^2) * (z^2 + w^2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_permutation_l3632_363215


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l3632_363206

theorem gcd_lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 24) (h2 : Nat.lcm a b = 3600) (h3 : b = 240) : a = 360 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l3632_363206


namespace NUMINAMATH_CALUDE_total_books_l3632_363220

theorem total_books (keith_books jason_books : ℕ) 
  (h1 : keith_books = 20) 
  (h2 : jason_books = 21) : 
  keith_books + jason_books = 41 := by
sorry

end NUMINAMATH_CALUDE_total_books_l3632_363220


namespace NUMINAMATH_CALUDE_glass_bottles_count_l3632_363257

/-- The number of glass bottles initially weighed -/
def initial_glass_bottles : ℕ := 3

/-- The weight of a plastic bottle in grams -/
def plastic_bottle_weight : ℕ := 50

/-- The weight of a glass bottle in grams -/
def glass_bottle_weight : ℕ := plastic_bottle_weight + 150

theorem glass_bottles_count :
  (initial_glass_bottles * glass_bottle_weight = 600) ∧
  (4 * glass_bottle_weight + 5 * plastic_bottle_weight = 1050) ∧
  (glass_bottle_weight = plastic_bottle_weight + 150) →
  initial_glass_bottles = 3 :=
by sorry

end NUMINAMATH_CALUDE_glass_bottles_count_l3632_363257


namespace NUMINAMATH_CALUDE_projection_theorem_l3632_363262

def proj_vector (a b : ℝ × ℝ) : ℝ × ℝ := sorry

theorem projection_theorem (a b : ℝ × ℝ) (angle : ℝ) :
  angle = 2 * Real.pi / 3 →
  norm a = 10 →
  b = (3, 4) →
  proj_vector a b = (-3, -4) := by sorry

end NUMINAMATH_CALUDE_projection_theorem_l3632_363262


namespace NUMINAMATH_CALUDE_work_time_relation_l3632_363265

/-- Represents the amount of work that can be done by a group of people in a given time -/
structure WorkCapacity where
  people : ℕ
  work : ℝ
  days : ℝ

/-- The work rate is constant for a given group size -/
axiom work_rate_constant (w : WorkCapacity) : w.work / w.days = w.people

/-- The theorem stating the relationship between work, people, and time -/
theorem work_time_relation (w1 w2 : WorkCapacity) 
  (h1 : w1.people = 3 ∧ w1.work = 3)
  (h2 : w2.people = 5 ∧ w2.work = 5)
  (h3 : w1.days = w2.days) :
  ∃ (original_work : WorkCapacity), 
    original_work.people = 3 ∧ 
    original_work.work = 1 ∧ 
    original_work.days = w1.days / 3 :=
sorry

end NUMINAMATH_CALUDE_work_time_relation_l3632_363265


namespace NUMINAMATH_CALUDE_batsman_total_score_l3632_363270

/-- Represents the score of a batsman in cricket --/
structure BatsmanScore where
  total : ℝ
  boundaries : ℕ
  sixes : ℕ
  runningPercentage : ℝ

/-- The total score of a batsman is 120 runs given the specified conditions --/
theorem batsman_total_score 
  (score : BatsmanScore) 
  (h1 : score.boundaries = 5) 
  (h2 : score.sixes = 5) 
  (h3 : score.runningPercentage = 58.333333333333336) :
  score.total = 120 := by
  sorry

end NUMINAMATH_CALUDE_batsman_total_score_l3632_363270


namespace NUMINAMATH_CALUDE_parabola_locus_l3632_363251

-- Define the parabola and its properties
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define the locus L
def locus (p : ℕ) (x y : ℝ) : Prop :=
  is_prime p ∧ p ≠ 2 ∧ y ≠ 0 ∧ 4 * y^2 = p * (x - p)

-- Theorem statement
theorem parabola_locus (p : ℕ) :
  is_prime p →
  p ≠ 2 →
  (∃ (x y : ℤ), locus p (x : ℝ) (y : ℝ)) ∧
  (∀ (x y : ℤ), locus p (x : ℝ) (y : ℝ) → ¬ ∃ (m : ℤ), (x : ℝ)^2 + (y : ℝ)^2 = (m : ℝ)^2) ∧
  (∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ locus p (x : ℝ) (y : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_parabola_locus_l3632_363251


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l3632_363263

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (1/3 : ℂ) + (1/2 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (1/3 : ℂ) - (1/2 : ℂ) * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l3632_363263


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l3632_363278

/-- Given a real number a, if f(x) = x^3 + ax^2 + (a + 2)x and f'(x) is an even function,
    then the equation of the tangent line to y=f(x) at the origin is y = 2x. -/
theorem tangent_line_at_origin (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + (a + 2)*x
  let f' : ℝ → ℝ := λ x ↦ (3*x^2 + 2*a*x + (a + 2))
  (∀ x, f' x = f' (-x)) →  -- f' is an even function
  (λ x ↦ 2*x) = (λ x ↦ f' 0 * x + f 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l3632_363278


namespace NUMINAMATH_CALUDE_max_a_value_l3632_363293

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0) → 
  a ≤ 1 ∧ ∀ b : ℝ, (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - b ≥ 0) → b ≤ a :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l3632_363293


namespace NUMINAMATH_CALUDE_twentieth_term_is_41_l3632_363273

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem twentieth_term_is_41 :
  arithmetic_sequence 3 2 20 = 41 := by sorry

end NUMINAMATH_CALUDE_twentieth_term_is_41_l3632_363273


namespace NUMINAMATH_CALUDE_triangle_area_qin_jiushao_l3632_363205

theorem triangle_area_qin_jiushao 
  (a b c : ℝ) 
  (h_positive : 0 < c ∧ 0 < b ∧ 0 < a) 
  (h_order : c < b ∧ b < a) 
  (h_a : a = 15) 
  (h_b : b = 14) 
  (h_c : c = 13) : 
  Real.sqrt ((1/4) * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2)) = 84 := by
  sorry

#check triangle_area_qin_jiushao

end NUMINAMATH_CALUDE_triangle_area_qin_jiushao_l3632_363205


namespace NUMINAMATH_CALUDE_age_ratio_five_years_ago_l3632_363200

/-- Represents the ages of Lucy and Lovely -/
structure Ages where
  lucy : ℕ
  lovely : ℕ

/-- The conditions of the problem -/
def problem_conditions (a : Ages) : Prop :=
  a.lucy = 50 ∧
  ∃ x : ℚ, (a.lucy - 5 : ℚ) = x * (a.lovely - 5 : ℚ) ∧
  (a.lucy + 10 : ℚ) = 2 * (a.lovely + 10 : ℚ)

/-- The theorem statement -/
theorem age_ratio_five_years_ago (a : Ages) :
  problem_conditions a →
  (a.lucy - 5 : ℚ) / (a.lovely - 5 : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_five_years_ago_l3632_363200


namespace NUMINAMATH_CALUDE_expression_equality_l3632_363297

theorem expression_equality : (2023^2 - 2015^2) / (2030^2 - 2008^2) = 4/11 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3632_363297


namespace NUMINAMATH_CALUDE_min_yellow_marbles_l3632_363218

-- Define the total number of marbles
variable (n : ℕ)

-- Define the number of yellow marbles
variable (y : ℕ)

-- Define the conditions
def blue_marbles := n / 3
def red_marbles := n / 4
def green_marbles := 9
def white_marbles := 2 * y

-- Define the total number of marbles equation
def total_marbles_equation : Prop :=
  n = blue_marbles n + red_marbles n + green_marbles + y + white_marbles y

-- Theorem statement
theorem min_yellow_marbles :
  (∃ n : ℕ, total_marbles_equation n y) → y ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_yellow_marbles_l3632_363218


namespace NUMINAMATH_CALUDE_share_ratio_l3632_363284

/-- 
Given:
- The total amount of money is $400
- A's share is $160
- A gets a certain fraction (x) as much as B and C together
- B gets 6/9 as much as A and C together

Prove that the ratio of A's share to the combined share of B and C is 2:3
-/
theorem share_ratio (total : ℕ) (a b c : ℕ) (x : ℚ) :
  total = 400 →
  a = 160 →
  a = x * (b + c) →
  b = (6/9 : ℚ) * (a + c) →
  a + b + c = total →
  (a : ℚ) / ((b + c) : ℚ) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_l3632_363284


namespace NUMINAMATH_CALUDE_fraction_sum_zero_l3632_363245

theorem fraction_sum_zero (a b : ℚ) (h : b + 1 ≠ 0) : 
  a / (b + 1) + 2 * a / (b + 1) - 3 * a / (b + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_l3632_363245


namespace NUMINAMATH_CALUDE_factorization_proof_l3632_363234

theorem factorization_proof (a : ℝ) : a^2 + 4*a + 4 = (a + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3632_363234


namespace NUMINAMATH_CALUDE_smallest_angle_equation_l3632_363226

theorem smallest_angle_equation (y : ℝ) : 
  (∀ z ∈ {x : ℝ | x > 0 ∧ 8 * Real.sin x * (Real.cos x)^3 - 8 * (Real.sin x)^3 * Real.cos x = 1}, y ≤ z) ∧ 
  (8 * Real.sin y * (Real.cos y)^3 - 8 * (Real.sin y)^3 * Real.cos y = 1) →
  y = π / 24 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_equation_l3632_363226


namespace NUMINAMATH_CALUDE_gcd_7979_3713_l3632_363202

theorem gcd_7979_3713 : Nat.gcd 7979 3713 = 79 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7979_3713_l3632_363202


namespace NUMINAMATH_CALUDE_samantha_sleep_hours_l3632_363299

/-- Represents the number of hours Samantha sleeps per night -/
def samantha_sleep : ℝ := 8

/-- Represents the number of hours Samantha's baby sister sleeps per night -/
def baby_sister_sleep : ℝ := 2.5 * samantha_sleep

/-- Represents the number of hours Samantha's father sleeps per night -/
def father_sleep : ℝ := 0.5 * baby_sister_sleep

theorem samantha_sleep_hours :
  samantha_sleep = 8 ∧
  baby_sister_sleep = 2.5 * samantha_sleep ∧
  father_sleep = 0.5 * baby_sister_sleep ∧
  7 * father_sleep = 70 := by
  sorry

#check samantha_sleep_hours

end NUMINAMATH_CALUDE_samantha_sleep_hours_l3632_363299


namespace NUMINAMATH_CALUDE_total_faces_painted_is_48_l3632_363267

/-- The number of outer faces of a cuboid -/
def cuboid_faces : ℕ := 6

/-- The number of cuboids -/
def num_cuboids : ℕ := 8

/-- The total number of faces painted -/
def total_faces_painted : ℕ := cuboid_faces * num_cuboids

/-- Theorem: The total number of faces painted on 8 identical cuboids is 48 -/
theorem total_faces_painted_is_48 : total_faces_painted = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_faces_painted_is_48_l3632_363267


namespace NUMINAMATH_CALUDE_segment_length_l3632_363292

/-- Given a line segment CD with points R and S on it, prove that CD has length 273.6 -/
theorem segment_length (C D R S : ℝ) : 
  (R > (C + D) / 2) →  -- R is on the same side of the midpoint as S
  (S > (C + D) / 2) →  -- S is on the same side of the midpoint as R
  (R - C) / (D - R) = 3 / 5 →  -- R divides CD in ratio 3:5
  (S - C) / (D - S) = 4 / 7 →  -- S divides CD in ratio 4:7
  S - R = 3 →  -- RS = 3
  D - C = 273.6 :=  -- CD = 273.6
by sorry

end NUMINAMATH_CALUDE_segment_length_l3632_363292


namespace NUMINAMATH_CALUDE_circle_equation_with_tangent_line_l3632_363289

/-- The equation of a circle with center (1, -1) tangent to the line x + y - √6 = 0 --/
theorem circle_equation_with_tangent_line :
  ∃ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 3 ∧
  (x + y - Real.sqrt 6 = 0 → 
    ∃ (x₀ y₀ : ℝ), (x₀ - 1)^2 + (y₀ + 1)^2 = 3 ∧ x₀ + y₀ - Real.sqrt 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_with_tangent_line_l3632_363289


namespace NUMINAMATH_CALUDE_aluminum_cans_collection_l3632_363238

theorem aluminum_cans_collection : 
  let sarah_yesterday : ℕ := 50
  let lara_yesterday : ℕ := sarah_yesterday + 30
  let sarah_today : ℕ := 40
  let lara_today : ℕ := 70
  let total_yesterday : ℕ := sarah_yesterday + lara_yesterday
  let total_today : ℕ := sarah_today + lara_today
  total_yesterday - total_today = 20 := by
sorry

end NUMINAMATH_CALUDE_aluminum_cans_collection_l3632_363238


namespace NUMINAMATH_CALUDE_circular_track_length_l3632_363252

/-- The length of the circular track in meters -/
def track_length : ℝ := 1250

/-- The initial speed of the Amur tiger in km/h -/
def amur_speed : ℝ := sorry

/-- The speed of the Bengal tiger in km/h -/
def bengal_speed : ℝ := sorry

/-- The number of additional laps run by the Amur tiger in the first 2 hours -/
def additional_laps_2h : ℝ := 6

/-- The speed increase of the Amur tiger after 2 hours in km/h -/
def speed_increase : ℝ := 10

/-- The total number of additional laps run by the Amur tiger in 3 hours -/
def total_additional_laps : ℝ := 17

theorem circular_track_length : 
  amur_speed - bengal_speed = 3 ∧
  (amur_speed - bengal_speed) * 2 = additional_laps_2h ∧
  (amur_speed + speed_increase - bengal_speed) * 1 + additional_laps_2h = total_additional_laps →
  track_length = 1250 := by sorry

end NUMINAMATH_CALUDE_circular_track_length_l3632_363252


namespace NUMINAMATH_CALUDE_inequality_solution_l3632_363255

theorem inequality_solution (x : ℝ) : 
  3 - 2 / (3 * x + 4) ≤ 5 ↔ x < -4/3 ∨ x > -5/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3632_363255


namespace NUMINAMATH_CALUDE_cos_240_degrees_l3632_363258

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l3632_363258


namespace NUMINAMATH_CALUDE_probability_adjacent_is_two_thirds_l3632_363274

/-- The number of ways to arrange 3 distinct objects in a row -/
def total_arrangements : ℕ := 6

/-- The number of arrangements where A and B are adjacent -/
def adjacent_arrangements : ℕ := 4

/-- The probability of A and B being adjacent when A, B, and C stand in a row -/
def probability_adjacent : ℚ := adjacent_arrangements / total_arrangements

theorem probability_adjacent_is_two_thirds :
  probability_adjacent = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_adjacent_is_two_thirds_l3632_363274


namespace NUMINAMATH_CALUDE_baseball_ticket_cost_is_8_l3632_363246

/-- Calculates the cost of a baseball ticket given initial amount, cost of hot dog, and remaining amount -/
def baseball_ticket_cost (initial_amount : ℕ) (hot_dog_cost : ℕ) (remaining_amount : ℕ) : ℕ :=
  initial_amount - hot_dog_cost - remaining_amount

/-- Proves that the cost of the baseball ticket is 8 given the specified conditions -/
theorem baseball_ticket_cost_is_8 :
  baseball_ticket_cost 20 3 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_baseball_ticket_cost_is_8_l3632_363246


namespace NUMINAMATH_CALUDE_T_always_one_smallest_n_correct_l3632_363212

/-- Definition of T_n -/
def T (n : ℕ+) : ℚ :=
  (Finset.filter (fun i => i ≠ 0) (Finset.range 2)).sum (fun i => 1 / i)

/-- Theorem: T_n is always 1 for any positive integer n -/
theorem T_always_one (n : ℕ+) : T n = 1 := by
  sorry

/-- The smallest positive integer n for which T_n is an integer -/
def smallest_n : ℕ+ := 1

/-- Theorem: smallest_n is indeed the smallest positive integer for which T_n is an integer -/
theorem smallest_n_correct : 
  ∀ k : ℕ+, (∃ m : ℤ, T k = m) → k ≥ smallest_n := by
  sorry

end NUMINAMATH_CALUDE_T_always_one_smallest_n_correct_l3632_363212


namespace NUMINAMATH_CALUDE_product_sum_relation_l3632_363236

theorem product_sum_relation (a b : ℝ) : 
  (a * b = 2 * (a + b) + 1) → (b = 7) → (b - a = 4) := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l3632_363236


namespace NUMINAMATH_CALUDE_bank_max_profit_rate_l3632_363281

/-- The bank's profit function --/
def profit (x : ℝ) : ℝ := 480 * x^2 - 10000 * x^3

/-- The derivative of the profit function --/
def profit_derivative (x : ℝ) : ℝ := 960 * x - 30000 * x^2

theorem bank_max_profit_rate :
  ∃ x : ℝ, x ∈ Set.Ioo 0 0.048 ∧
    (∀ y ∈ Set.Ioo 0 0.048, profit y ≤ profit x) ∧
    x = 0.032 := by
  sorry

end NUMINAMATH_CALUDE_bank_max_profit_rate_l3632_363281


namespace NUMINAMATH_CALUDE_solution_set_when_m_eq_3_m_range_when_f_geq_8_l3632_363282

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

-- Theorem for part I
theorem solution_set_when_m_eq_3 :
  {x : ℝ | f x 3 ≥ 6} = {x : ℝ | x ≤ -2 ∨ x ≥ 4} := by sorry

-- Theorem for part II
theorem m_range_when_f_geq_8 :
  (∀ x : ℝ, f x m ≥ 8) ↔ m ≤ -9 ∨ m ≥ 7 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_eq_3_m_range_when_f_geq_8_l3632_363282


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_conditions_l3632_363279

noncomputable def f (a b x : ℝ) : ℝ := a + (b * x - 1) * Real.exp x

theorem tangent_line_and_inequality_conditions 
  (a b : ℝ) 
  (h1 : f a b 0 = 0) 
  (h2 : (deriv (f a b)) 0 = 1) 
  (h3 : a < 1) 
  (h4 : b = 2) 
  (h5 : ∃! (n : ℤ), f a b n < a * n) :
  a = 1 ∧ b = 2 ∧ 3 / (2 * Real.exp 1) ≤ a := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequality_conditions_l3632_363279


namespace NUMINAMATH_CALUDE_carmen_additional_money_l3632_363229

/-- Calculates how much more money Carmen needs to have twice Jethro's amount -/
theorem carmen_additional_money (patricia_money jethro_money carmen_money : ℕ) : 
  patricia_money = 60 →
  patricia_money = 3 * jethro_money →
  carmen_money + patricia_money + jethro_money = 113 →
  (2 * jethro_money) - carmen_money = 7 :=
by
  sorry

#check carmen_additional_money

end NUMINAMATH_CALUDE_carmen_additional_money_l3632_363229


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l3632_363242

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ (x : ℂ), x = (-5 + Complex.I * Real.sqrt 171) / 14 ∧ 
   7 * x^2 + 5 * x + k = 0) → k = 7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l3632_363242


namespace NUMINAMATH_CALUDE_revolver_problem_l3632_363222

/-- Probability of the gun firing on any given shot -/
def p : ℚ := 1 / 6

/-- Probability of the gun not firing on any given shot -/
def q : ℚ := 1 - p

/-- The probability that the gun will fire while A is holding it -/
noncomputable def prob_A_fires : ℚ := sorry

theorem revolver_problem : prob_A_fires = 6 / 11 := by sorry

end NUMINAMATH_CALUDE_revolver_problem_l3632_363222


namespace NUMINAMATH_CALUDE_basketball_handshakes_l3632_363224

/-- Calculates the total number of handshakes in a basketball game scenario -/
theorem basketball_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : 
  team_size = 5 → num_teams = 2 → num_referees = 2 →
  (team_size * team_size) + (team_size * num_teams * num_referees) = 45 := by
  sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l3632_363224


namespace NUMINAMATH_CALUDE_erica_safari_animals_l3632_363232

/-- The number of animals Erica saw on Saturday -/
def saturday_animals : ℕ := 3 + 2

/-- The number of animals Erica saw on Sunday -/
def sunday_animals : ℕ := 2 + 5

/-- The number of animals Erica saw on Monday -/
def monday_animals : ℕ := 5 + 3

/-- The total number of animals Erica saw during her safari -/
def total_animals : ℕ := saturday_animals + sunday_animals + monday_animals

theorem erica_safari_animals : total_animals = 20 := by
  sorry

end NUMINAMATH_CALUDE_erica_safari_animals_l3632_363232


namespace NUMINAMATH_CALUDE_largest_mu_inequality_l3632_363241

theorem largest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  (∀ μ : ℝ, (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ a*b + μ*b*c + 2*c*d) → μ ≤ 3/4) ∧
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ a*b + 3/4*b*c + 2*c*d) :=
by sorry

end NUMINAMATH_CALUDE_largest_mu_inequality_l3632_363241


namespace NUMINAMATH_CALUDE_parallel_plane_line_l3632_363294

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelPL : Plane → Line → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularPL : Plane → Line → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the theorem
theorem parallel_plane_line 
  (l m n : Line) 
  (α β : Plane) 
  (h_distinct_lines : l ≠ m ∧ l ≠ n ∧ m ≠ n) 
  (h_distinct_planes : α ≠ β) 
  (h_parallel_planes : parallelPP α β) 
  (h_line_in_plane : subset l α) : 
  parallelPL β l :=
sorry

end NUMINAMATH_CALUDE_parallel_plane_line_l3632_363294


namespace NUMINAMATH_CALUDE_three_lines_vertically_opposite_angles_l3632_363264

/-- The number of pairs of vertically opposite angles formed by three intersecting lines in a plane -/
def vertically_opposite_angles_count (n : ℕ) : ℕ :=
  if n = 3 then 6 else 0

/-- Theorem stating that three intersecting lines in a plane form 6 pairs of vertically opposite angles -/
theorem three_lines_vertically_opposite_angles :
  vertically_opposite_angles_count 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_three_lines_vertically_opposite_angles_l3632_363264


namespace NUMINAMATH_CALUDE_percentage_passed_both_l3632_363240

theorem percentage_passed_both (total : ℕ) (h : total > 0) :
  let failed_hindi := (25 : ℕ) * total / 100
  let failed_english := (50 : ℕ) * total / 100
  let failed_both := (25 : ℕ) * total / 100
  let passed_both := total - (failed_hindi + failed_english - failed_both)
  (passed_both * 100 : ℕ) / total = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_passed_both_l3632_363240


namespace NUMINAMATH_CALUDE_sum_f_eq_518656_l3632_363233

/-- f(n) is the index of the highest power of 2 which divides n! -/
def f (n : ℕ+) : ℕ := sorry

/-- Sum of f(n) from 1 to 1023 -/
def sum_f : ℕ := sorry

theorem sum_f_eq_518656 : sum_f = 518656 := by sorry

end NUMINAMATH_CALUDE_sum_f_eq_518656_l3632_363233


namespace NUMINAMATH_CALUDE_product_equality_l3632_363295

theorem product_equality : 72519 * 31415.927 = 2277666538.233 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3632_363295


namespace NUMINAMATH_CALUDE_brothers_ages_sum_l3632_363216

theorem brothers_ages_sum (a b c : ℕ) : 
  a = 31 → b = a + 1 → c = b + 1 → a + b + c = 96 := by
  sorry

end NUMINAMATH_CALUDE_brothers_ages_sum_l3632_363216


namespace NUMINAMATH_CALUDE_unique_solution_l3632_363243

theorem unique_solution (x y : ℝ) : 
  |x - 2*y + 1| + (x + y - 5)^2 = 0 → x = 3 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3632_363243


namespace NUMINAMATH_CALUDE_find_k_value_l3632_363260

theorem find_k_value (k : ℝ) : 
  (∃ x y : ℝ, x = 2 ∧ y = -1 ∧ y - k * x = 7) → k = -4 := by
  sorry

end NUMINAMATH_CALUDE_find_k_value_l3632_363260


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_l3632_363221

theorem sum_of_even_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₀*x^7 + a₁*x^6 + a₂*x^5 + a₃*x^4 + a₄*x^3 + a₅*x^2 + a₆*x + a₇) →
  a₀ + a₂ + a₄ + a₆ = 4128 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_l3632_363221


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3632_363211

theorem arithmetic_calculation : 2 * (-5 + 3) + 2^3 / (-4) = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3632_363211


namespace NUMINAMATH_CALUDE_cupcakes_needed_l3632_363259

theorem cupcakes_needed (fourth_grade_classes : ℕ) (students_per_fourth_grade : ℕ)
  (pe_class_students : ℕ) (afterschool_clubs : ℕ) (students_per_club : ℕ) :
  fourth_grade_classes = 8 →
  students_per_fourth_grade = 40 →
  pe_class_students = 80 →
  afterschool_clubs = 2 →
  students_per_club = 35 →
  fourth_grade_classes * students_per_fourth_grade +
  pe_class_students +
  afterschool_clubs * students_per_club = 470 := by
sorry

end NUMINAMATH_CALUDE_cupcakes_needed_l3632_363259


namespace NUMINAMATH_CALUDE_ratio_equality_counterexample_l3632_363272

theorem ratio_equality_counterexample (a b c d : ℝ) 
  (h1 : a ≠ 0) (h2 : c ≠ 0) (h3 : a / b = c / d) : 
  ¬ ((a + d) / (b + c) = a / b) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_counterexample_l3632_363272


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l3632_363217

theorem magnitude_of_complex_power (z : ℂ) (n : ℕ) :
  z = 3/5 + 4/5 * I → n = 6 → Complex.abs (z^n) = 1 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l3632_363217


namespace NUMINAMATH_CALUDE_equal_debt_after_10_days_l3632_363227

/-- The number of days after which Darren and Fergie will owe the same amount -/
def days_to_equal_debt : ℕ := 10

/-- Darren's initial borrowed amount in clams -/
def darren_initial : ℕ := 200

/-- Fergie's initial borrowed amount in clams -/
def fergie_initial : ℕ := 150

/-- Daily interest rate as a percentage -/
def daily_interest_rate : ℚ := 10 / 100

theorem equal_debt_after_10_days :
  (darren_initial : ℚ) * (1 + daily_interest_rate * days_to_equal_debt) =
  (fergie_initial : ℚ) * (1 + daily_interest_rate * days_to_equal_debt) :=
sorry

end NUMINAMATH_CALUDE_equal_debt_after_10_days_l3632_363227


namespace NUMINAMATH_CALUDE_pedal_triangle_area_l3632_363213

-- Define the circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point
def Point := ℝ × ℝ

-- Function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Function to check if a triangle is inscribed in a circle
def isInscribed (t : Triangle) (c : Circle) : Prop := sorry

-- Function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ := sorry

-- Function to calculate the area of the pedal triangle
def pedalTriangleArea (t : Triangle) (p : Point) : ℝ := sorry

-- The main theorem
theorem pedal_triangle_area 
  (c : Circle) (t : Triangle) (p : Point) 
  (h1 : isInscribed t c) 
  (h2 : distance p c.center = d) :
  pedalTriangleArea t p = (1/4) * |1 - (d^2 / c.radius^2)| * triangleArea t := 
by sorry

end NUMINAMATH_CALUDE_pedal_triangle_area_l3632_363213


namespace NUMINAMATH_CALUDE_golden_ratio_expression_l3632_363291

theorem golden_ratio_expression (S : ℝ) (h : S^2 + S - 1 = 0) :
  S^(S^(S^2 - S⁻¹) - S⁻¹) - S⁻¹ = 0 := by sorry

end NUMINAMATH_CALUDE_golden_ratio_expression_l3632_363291


namespace NUMINAMATH_CALUDE_equation_solutions_l3632_363256

open Real

theorem equation_solutions (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    sin (x₁ - a) + cos (x₁ + 3 * a) = 0 ∧
    sin (x₂ - a) + cos (x₂ + 3 * a) = 0 ∧
    ∀ k : ℤ, x₁ - x₂ ≠ π * k) ↔
  ∃ t : ℤ, a = π * (4 * t + 1) / 8 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3632_363256


namespace NUMINAMATH_CALUDE_set_operations_l3632_363286

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x < 4}

theorem set_operations :
  (A ∪ B = {x | -1 < x ∧ x < 4}) ∧
  (A ∩ B = {x | 1 ≤ x ∧ x < 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3632_363286


namespace NUMINAMATH_CALUDE_monotonic_function_property_l3632_363266

/-- A monotonic function f satisfying f(f(x) - 3^x) = 4 for all x ∈ ℝ has f(2) = 10 -/
theorem monotonic_function_property (f : ℝ → ℝ) 
  (h_mono : Monotone f) 
  (h_prop : ∀ x, f (f x - 3^x) = 4) : 
  f 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_function_property_l3632_363266


namespace NUMINAMATH_CALUDE_rectangle_side_difference_l3632_363285

theorem rectangle_side_difference (A d x y : ℝ) (h1 : A > 0) (h2 : d > 0) (h3 : x > y) (h4 : x * y = A) (h5 : x^2 + y^2 = d^2) : x - y = 2 * Real.sqrt A := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_difference_l3632_363285


namespace NUMINAMATH_CALUDE_specific_prism_triangle_perimeter_l3632_363254

/-- A right prism with equilateral triangle bases -/
structure RightPrism where
  base_side_length : ℝ
  height : ℝ

/-- Midpoints of edges in the prism -/
structure PrismMidpoints (prism : RightPrism) where
  V : ℝ × ℝ × ℝ  -- Midpoint of PR
  W : ℝ × ℝ × ℝ  -- Midpoint of RQ
  X : ℝ × ℝ × ℝ  -- Midpoint of QT

/-- The perimeter of triangle VWX in the prism -/
def triangle_perimeter (prism : RightPrism) (midpoints : PrismMidpoints prism) : ℝ :=
  sorry

/-- Theorem stating the perimeter of triangle VWX in the specific prism -/
theorem specific_prism_triangle_perimeter :
  let prism : RightPrism := { base_side_length := 10, height := 20 }
  let midpoints : PrismMidpoints prism := sorry
  triangle_perimeter prism midpoints = 5 + 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_specific_prism_triangle_perimeter_l3632_363254


namespace NUMINAMATH_CALUDE_quartic_polynomial_satisfies_conditions_l3632_363275

theorem quartic_polynomial_satisfies_conditions :
  let p : ℝ → ℝ := λ x => -x^4 + 2*x^2 - 5*x + 1
  (p 1 = -3) ∧ (p 2 = -5) ∧ (p 3 = -11) ∧ (p 4 = -27) ∧ (p 5 = -59) := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_satisfies_conditions_l3632_363275


namespace NUMINAMATH_CALUDE_parabola_point_comparison_l3632_363261

/-- Parabola function -/
def f (x : ℝ) : ℝ := -x^2 + 2*x - 2

/-- Theorem: For the parabola y = -x^2 + 2x - 2, if (-2, y₁) and (3, y₂) are points on the parabola, then y₁ < y₂ -/
theorem parabola_point_comparison (y₁ y₂ : ℝ) 
  (h₁ : f (-2) = y₁) 
  (h₂ : f 3 = y₂) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_comparison_l3632_363261


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3632_363288

/-- Given a line L1 with equation x + 2y - 1 = 0 and a point A(1,2),
    prove that the line L2 passing through A and perpendicular to L1
    has the equation 2x - y = 0 -/
theorem perpendicular_line_equation (L1 : Set (ℝ × ℝ)) (A : ℝ × ℝ) :
  (L1 = {(x, y) | x + 2*y - 1 = 0}) →
  (A = (1, 2)) →
  (∃ L2 : Set (ℝ × ℝ), L2 = {(x, y) | 2*x - y = 0} ∧ 
    A ∈ L2 ∧ 
    (∀ (v w : ℝ × ℝ), v ∈ L1 → w ∈ L1 → v ≠ w → 
      (∀ (p q : ℝ × ℝ), p ∈ L2 → q ∈ L2 → p ≠ q → 
        ((v.1 - w.1) * (p.1 - q.1) + (v.2 - w.2) * (p.2 - q.2) = 0)))) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3632_363288


namespace NUMINAMATH_CALUDE_cheese_block_servings_l3632_363223

theorem cheese_block_servings (calories_per_serving : ℕ) (servings_eaten : ℕ) (calories_remaining : ℕ) :
  calories_per_serving = 110 →
  servings_eaten = 5 →
  calories_remaining = 1210 →
  (calories_remaining + servings_eaten * calories_per_serving) / calories_per_serving = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_cheese_block_servings_l3632_363223


namespace NUMINAMATH_CALUDE_prob_exactly_two_choose_A_l3632_363214

/-- The number of communities available for housing applications. -/
def num_communities : ℕ := 3

/-- The number of applicants. -/
def num_applicants : ℕ := 4

/-- The number of applicants required to choose community A. -/
def target_applicants : ℕ := 2

/-- The probability of an applicant choosing any specific community. -/
def prob_choose_community : ℚ := 1 / num_communities

/-- The probability that exactly 'target_applicants' out of 'num_applicants' 
    choose community A, given equal probability for each community. -/
theorem prob_exactly_two_choose_A : 
  (Nat.choose num_applicants target_applicants : ℚ) * 
  prob_choose_community ^ target_applicants * 
  (1 - prob_choose_community) ^ (num_applicants - target_applicants) = 8/27 :=
sorry

end NUMINAMATH_CALUDE_prob_exactly_two_choose_A_l3632_363214


namespace NUMINAMATH_CALUDE_chord_bisected_by_point_l3632_363201

def curve (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

def bisection_point : ℝ × ℝ := (3, -1)

def chord_equation (m b : ℝ) (x y : ℝ) : Prop := y = m*x + b

theorem chord_bisected_by_point (m b : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    curve x₁ y₁ ∧ 
    curve x₂ y₂ ∧ 
    chord_equation m b x₁ y₁ ∧ 
    chord_equation m b x₂ y₂ ∧ 
    ((x₁ + x₂)/2, (y₁ + y₂)/2) = bisection_point) →
  chord_equation (-3/4) (11/4) 3 (-1) ∧ 
  (∀ x y : ℝ, chord_equation (-3/4) (11/4) x y ↔ 3*x + 4*y - 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_chord_bisected_by_point_l3632_363201


namespace NUMINAMATH_CALUDE_roots_opposite_signs_l3632_363269

/-- Given an equation (x^2 - dx)/(cx - k) = (m-2)/(m+2) where c, d, and k are constants,
    prove that when m = 2(c - d)/(c + d), the equation has roots which are numerically
    equal but of opposite signs. -/
theorem roots_opposite_signs (c d k : ℝ) :
  let m := 2 * (c - d) / (c + d)
  let f := fun x => (x^2 - d*x) / (c*x - k) - (m - 2) / (m + 2)
  ∃ (r : ℝ), f r = 0 ∧ f (-r) = 0 := by
sorry

end NUMINAMATH_CALUDE_roots_opposite_signs_l3632_363269


namespace NUMINAMATH_CALUDE_ball_probability_l3632_363276

theorem ball_probability (m : ℕ) : 
  (3 : ℚ) / (3 + 4 + m) = 1 / 3 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l3632_363276


namespace NUMINAMATH_CALUDE_larger_number_in_sum_and_difference_l3632_363271

theorem larger_number_in_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 40) 
  (diff_eq : x - y = 6) : 
  max x y = 23 := by
sorry

end NUMINAMATH_CALUDE_larger_number_in_sum_and_difference_l3632_363271


namespace NUMINAMATH_CALUDE_expression_value_l3632_363253

theorem expression_value : 
  let x : ℤ := -2
  let y : ℤ := 1
  let z : ℤ := 4
  x^2 * y * z - x * y * z^2 = 48 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3632_363253


namespace NUMINAMATH_CALUDE_fixed_point_of_linear_function_l3632_363244

theorem fixed_point_of_linear_function (k : ℝ) :
  (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_linear_function_l3632_363244


namespace NUMINAMATH_CALUDE_sales_tax_difference_l3632_363296

theorem sales_tax_difference (price : ℝ) (high_rate low_rate : ℝ) 
  (h1 : price = 30)
  (h2 : high_rate = 0.075)
  (h3 : low_rate = 0.07) : 
  price * high_rate - price * low_rate = 0.15 := by
  sorry

#check sales_tax_difference

end NUMINAMATH_CALUDE_sales_tax_difference_l3632_363296


namespace NUMINAMATH_CALUDE_parabola_coef_sum_l3632_363280

/-- A parabola with equation x = ay^2 + by + c passing through points (6, -3) and (4, -1) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  point1 : 6 = a * (-3)^2 + b * (-3) + c
  point2 : 4 = a * (-1)^2 + b * (-1) + c

/-- The sum of coefficients a, b, and c of the parabola equals -2 -/
theorem parabola_coef_sum (p : Parabola) : p.a + p.b + p.c = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coef_sum_l3632_363280


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3632_363249

theorem complex_fraction_simplification :
  (1 + 3 * Complex.I) / (1 + Complex.I) = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3632_363249


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l3632_363235

/-- Proves that if a shopkeeper sells an article with a 5% discount and earns a 23.5% profit,
    then selling the same article without a discount would result in a 30% profit. -/
theorem shopkeeper_profit (cost_price : ℝ) (cost_price_pos : cost_price > 0) :
  let discount_rate := 0.05
  let profit_rate_with_discount := 0.235
  let selling_price_with_discount := cost_price * (1 + profit_rate_with_discount)
  let marked_price := selling_price_with_discount / (1 - discount_rate)
  let profit_without_discount := marked_price - cost_price
  profit_without_discount / cost_price = 0.3 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l3632_363235


namespace NUMINAMATH_CALUDE_graph_of_S_l3632_363287

theorem graph_of_S (a b t : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (sum_eq : a + b = 2) (prod_eq : a * b = t - 1) (ht : 1 < t ∧ t < 2) :
  (a - b)^2 = 8 - 4*t := by
  sorry

end NUMINAMATH_CALUDE_graph_of_S_l3632_363287


namespace NUMINAMATH_CALUDE_longest_pole_in_stadium_l3632_363210

theorem longest_pole_in_stadium (l w h : ℝ) (hl : l = 24) (hw : w = 18) (hh : h = 16) :
  Real.sqrt (l^2 + w^2 + h^2) = 34 := by
  sorry

end NUMINAMATH_CALUDE_longest_pole_in_stadium_l3632_363210


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3632_363283

/-- Given a quadratic inequality ax^2 - bx + 1 < 0 with solution set {x | x < -1/2 or x > 2}, 
    prove that a - b = 1/2 -/
theorem quadratic_inequality_solution (a b : ℝ) 
  (h : ∀ x, x < -1/2 ∨ x > 2 ↔ a * x^2 - b * x + 1 < 0) : 
  a - b = 1/2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3632_363283


namespace NUMINAMATH_CALUDE_jesses_room_difference_l3632_363209

theorem jesses_room_difference (width : ℝ) (length : ℝ) 
  (h1 : width = 19.7) (h2 : length = 20.25) : length - width = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_jesses_room_difference_l3632_363209


namespace NUMINAMATH_CALUDE_hidden_numbers_puzzle_l3632_363231

theorem hidden_numbers_puzzle (x y : ℕ) :
  x^2 + y^2 = 65 ∧
  x + y ≥ 10 ∧
  (∀ a b : ℕ, a^2 + b^2 = 65 ∧ a + b ≥ 10 → (a = x ∧ b = y) ∨ (a = y ∧ b = x)) →
  ((x = 7 ∧ y = 4) ∨ (x = 4 ∧ y = 7)) :=
by sorry

end NUMINAMATH_CALUDE_hidden_numbers_puzzle_l3632_363231


namespace NUMINAMATH_CALUDE_problem_statement_l3632_363250

theorem problem_statement 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h_a_order : a₁ < a₂ ∧ a₂ < a₃)
  (h_b_order : b₁ < b₂ ∧ b₂ < b₃)
  (h_sum : a₁ + a₂ + a₃ = b₁ + b₂ + b₃)
  (h_product_sum : a₁*a₂ + a₁*a₃ + a₂*a₃ = b₁*b₂ + b₁*b₃ + b₂*b₃)
  (h_a₁_b₁ : a₁ < b₁) : 
  (b₂ < a₂) ∧ 
  (a₃ < b₃) ∧ 
  (a₁*a₂*a₃ < b₁*b₂*b₃) ∧ 
  ((1-a₁)*(1-a₂)*(1-a₃) > (1-b₁)*(1-b₂)*(1-b₃)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3632_363250


namespace NUMINAMATH_CALUDE_luncheon_table_capacity_l3632_363208

theorem luncheon_table_capacity (invited : Nat) (no_shows : Nat) (tables : Nat) : Nat :=
  if invited = 18 ∧ no_shows = 12 ∧ tables = 2 then
    3
  else
    0

#check luncheon_table_capacity

end NUMINAMATH_CALUDE_luncheon_table_capacity_l3632_363208


namespace NUMINAMATH_CALUDE_decreasing_function_range_l3632_363219

/-- A function f: ℝ → ℝ is decreasing if for all x y, x < y implies f x > f y -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem decreasing_function_range (f : ℝ → ℝ) (m : ℝ) 
  (h1 : DecreasingFunction f)
  (h2 : ∀ x, f (-x) = -f x)
  (h3 : f (m - 1) + f (2*m - 1) > 0) :
  m < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_range_l3632_363219


namespace NUMINAMATH_CALUDE_pizzas_served_today_l3632_363228

theorem pizzas_served_today (lunch_pizzas dinner_pizzas : ℕ) 
  (h1 : lunch_pizzas = 9) 
  (h2 : dinner_pizzas = 6) : 
  lunch_pizzas + dinner_pizzas = 15 := by
  sorry

end NUMINAMATH_CALUDE_pizzas_served_today_l3632_363228


namespace NUMINAMATH_CALUDE_sophomore_count_l3632_363290

theorem sophomore_count (n : ℕ) : 
  n > 1000 → -- Ensure n is large enough to accommodate all students
  (60 : ℚ) / n = (27 : ℚ) / 450 →
  n = 450 + 250 + 300 :=
by
  sorry

end NUMINAMATH_CALUDE_sophomore_count_l3632_363290


namespace NUMINAMATH_CALUDE_smallest_positive_product_l3632_363230

def S : Set Int := {-4, -3, -1, 5, 6}

def is_valid_product (x y z : Int) : Prop :=
  x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z

def product (x y z : Int) : Int := x * y * z

theorem smallest_positive_product :
  ∃ (a b c : Int), is_valid_product a b c ∧ 
    product a b c > 0 ∧
    product a b c = 15 ∧
    ∀ (x y z : Int), is_valid_product x y z → product x y z > 0 → product x y z ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_product_l3632_363230


namespace NUMINAMATH_CALUDE_least_n_for_determinant_l3632_363277

theorem least_n_for_determinant (n : ℕ) : n ≥ 1 → (∀ k < n, 2^(k-1) < 2015) → 2^(n-1) ≥ 2015 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_least_n_for_determinant_l3632_363277


namespace NUMINAMATH_CALUDE_expected_unpaired_socks_l3632_363207

def n : ℕ := 2024

theorem expected_unpaired_socks (n : ℕ) :
  let total_socks := 2 * n
  let binom := Nat.choose total_socks n
  let expected_total := (4 : ℝ)^n / binom
  expected_total - 2 = (4 : ℝ)^n / Nat.choose (2 * n) n - 2 := by sorry

end NUMINAMATH_CALUDE_expected_unpaired_socks_l3632_363207


namespace NUMINAMATH_CALUDE_hayden_ironing_weeks_l3632_363203

/-- Calculates the number of weeks Hayden spends ironing given his daily routine and total ironing time. -/
def ironingWeeks (shirtTime minutesPerDay weekDays totalMinutes : ℕ) : ℕ :=
  totalMinutes / (shirtTime + minutesPerDay) / weekDays

/-- Proves that Hayden spends 4 weeks ironing given his routine and total ironing time. -/
theorem hayden_ironing_weeks :
  ironingWeeks 5 3 5 160 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hayden_ironing_weeks_l3632_363203


namespace NUMINAMATH_CALUDE_modular_inverse_5_mod_19_l3632_363225

theorem modular_inverse_5_mod_19 : ∃ x : ℕ, x < 19 ∧ (5 * x) % 19 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_5_mod_19_l3632_363225


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l3632_363248

-- Define the rectangle's dimensions
variable (l w : ℝ)

-- Define the conditions
def condition1 : Prop := (l + 3) * (w - 1) = l * w
def condition2 : Prop := (l - 1.5) * (w + 2) = l * w

-- State the theorem
theorem rectangle_area_theorem (h1 : condition1 l w) (h2 : condition2 l w) : l * w = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l3632_363248


namespace NUMINAMATH_CALUDE_average_movie_price_l3632_363247

theorem average_movie_price (dvd_count : ℕ) (dvd_price : ℚ) (bluray_count : ℕ) (bluray_price : ℚ) : 
  dvd_count = 8 → 
  dvd_price = 12 → 
  bluray_count = 4 → 
  bluray_price = 18 → 
  (dvd_count * dvd_price + bluray_count * bluray_price) / (dvd_count + bluray_count) = 14 := by
sorry

end NUMINAMATH_CALUDE_average_movie_price_l3632_363247


namespace NUMINAMATH_CALUDE_power_87_plus_3_mod_7_l3632_363268

theorem power_87_plus_3_mod_7 : (2^87 + 3) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_87_plus_3_mod_7_l3632_363268
